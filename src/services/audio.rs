//! Cross-platform desktop-audio capture → FFT → the `g_AudioSpectrum*`
//! spectra Wallpaper Engine shaders react to.
//!
//! [`cpal`] gives one capture code path across Linux (ALSA/PulseAudio),
//! macOS (CoreAudio) and Windows (WASAPI). The genuinely platform-specific
//! part is *which* device carries the desktop output ("loopback"):
//!
//! - **Linux:** PulseAudio/PipeWire expose a `.monitor` source that shows up
//!   as a normal capture device — we prefer an input whose name contains
//!   "monitor", so wallpapers react to whatever is playing.
//! - **macOS/Windows:** no native loopback through cpal; we fall back to the
//!   default input (a mic, or a virtual loopback device like BlackHole if the
//!   user installed one). System-audio capture there (ScreenCaptureKit /
//!   WASAPI loopback) can slot in behind this same PCM interface later.
//!
//! The reference (`Audio/AudioContext` + kissfft) processes a stereo FFT and
//! buckets it into 16/32/64 bands per channel; we mirror that shape. Capture
//! is best-effort: if no device/stream is available every band reads 0, which
//! is exactly what the shaders expect when nothing is playing.

use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{num_complex::Complex, FftPlanner};

/// FFT window size. 1024 samples ≈ 21 ms at 48 kHz — enough low-frequency
/// resolution for music bands without lagging the visual.
const FFT_SIZE: usize = 1024;

/// The per-channel band counts WE exposes. Each shader array is one of these.
const BANDS_16: usize = 16;
const BANDS_32: usize = 32;
const BANDS_64: usize = 64;

/// One frame's worth of spectra, mirroring WE's `g_AudioSpectrum{16,32,64}
/// {Left,Right}` uniforms. Each value is a normalized band magnitude ~[0,1].
#[derive(Clone)]
pub struct AudioSpectrum {
    pub s16_left: [f32; BANDS_16],
    pub s16_right: [f32; BANDS_16],
    pub s32_left: [f32; BANDS_32],
    pub s32_right: [f32; BANDS_32],
    pub s64_left: [f32; BANDS_64],
    pub s64_right: [f32; BANDS_64],
}

impl Default for AudioSpectrum {
    fn default() -> Self {
        Self {
            s16_left: [0.0; BANDS_16],
            s16_right: [0.0; BANDS_16],
            s32_left: [0.0; BANDS_32],
            s32_right: [0.0; BANDS_32],
            s64_left: [0.0; BANDS_64],
            s64_right: [0.0; BANDS_64],
        }
    }
}

/// Byte size of the std430 audio storage buffer the transpiler emits (see
/// `transpiler.rs`): six `float[N]` arrays, tightly packed at 4-byte stride.
pub const UNIFORM_BYTES: usize = (BANDS_16 * 2 + BANDS_32 * 2 + BANDS_64 * 2) * 4;

impl AudioSpectrum {
    /// Serialize into the std430 layout the emitted `WEAudio` block expects:
    /// arrays in canonical order (16L,16R,32L,32R,64L,64R), tightly packed
    /// (`float[N]` has a 4-byte stride in std430 — no vec4 padding).
    pub fn to_uniform_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(UNIFORM_BYTES);
        for arr in [
            &self.s16_left[..],
            &self.s16_right[..],
            &self.s32_left[..],
            &self.s32_right[..],
            &self.s64_left[..],
            &self.s64_right[..],
        ] {
            for &v in arr {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }
}

/// A running desktop-audio capture. The cpal stream fills a shared stereo
/// ring buffer on its own thread; [`Self::spectrum`] windows the newest
/// samples and runs the FFT on demand (once per rendered frame).
pub struct AudioCapture {
    /// Interleaved-deinterleaved: the latest `FFT_SIZE` samples per channel.
    left: Arc<Mutex<Vec<f32>>>,
    right: Arc<Mutex<Vec<f32>>>,
    planner: Mutex<std::sync::Arc<dyn rustfft::Fft<f32>>>,
    window: Vec<f32>,
    /// Kept alive so the callback keeps running; dropping it stops capture.
    _stream: cpal::Stream,
}

impl AudioCapture {
    /// Start capturing from the best available loopback/input device. Returns
    /// `None` (and the caller keeps a silent spectrum) when no device or the
    /// stream can't be built — audio reactivity is always optional.
    pub fn start() -> Option<Self> {
        let host = cpal::default_host();
        let Some(device) = pick_input_device(&host) else {
            tracing::warn!(target: "audio", "no capture device found — spectrum stays silent");
            return None;
        };
        let name = device.name().unwrap_or_default();
        let config = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(target: "audio", "'{name}' has no default input config: {e}");
                return None;
            }
        };
        let channels = config.channels() as usize;
        tracing::info!(target: "audio", "capturing from '{name}' ({} ch)", channels);

        let left = Arc::new(Mutex::new(vec![0.0f32; FFT_SIZE]));
        let right = Arc::new(Mutex::new(vec![0.0f32; FFT_SIZE]));
        let (cl, cr) = (left.clone(), right.clone());

        // Only f32 input is handled; every desktop host cpal targets delivers
        // f32 for monitor/loopback sources, so this covers the real cases.
        let stream = device
            .build_input_stream(
                &config.into(),
                move |data: &[f32], _| push_samples(data, channels, &cl, &cr),
                |err| tracing::warn!(target: "audio", "stream error: {err}"),
                None,
            )
            .ok()?;
        stream.play().ok()?;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        // Hann window kills spectral leakage so a pure tone lands in one band.
        let window = (0..FFT_SIZE)
            .map(|i| {
                let x = std::f32::consts::PI * i as f32 / (FFT_SIZE - 1) as f32;
                x.sin() * x.sin()
            })
            .collect();

        Some(Self {
            left,
            right,
            planner: Mutex::new(fft),
            window,
            _stream: stream,
        })
    }

    /// Compute the current spectra from the newest captured window.
    pub fn spectrum(&self) -> AudioSpectrum {
        let fft = self.planner.lock().unwrap().clone();
        let left = self.channel_bands(&self.left, &fft);
        let right = self.channel_bands(&self.right, &fft);
        AudioSpectrum {
            s16_left: downsample(&left),
            s16_right: downsample(&right),
            s32_left: downsample(&left),
            s32_right: downsample(&right),
            s64_left: left,
            s64_right: right,
        }
    }

    /// FFT one channel's window into 64 normalized magnitude bands.
    fn channel_bands(
        &self,
        buf: &Arc<Mutex<Vec<f32>>>,
        fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    ) -> [f32; BANDS_64] {
        let samples = buf.lock().unwrap().clone();
        let mut spectrum: Vec<Complex<f32>> = samples
            .iter()
            .zip(&self.window)
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();
        fft.process(&mut spectrum);

        // Only the first half of the FFT is unique (real input). Bucket the
        // usable bins into 64 bands, log-scaling the magnitude the way WE's
        // meters do so quiet detail is visible without clipping loud peaks.
        let usable = FFT_SIZE / 2;
        let per_band = (usable / BANDS_64).max(1);
        let mut bands = [0.0f32; BANDS_64];
        for (b, out) in bands.iter_mut().enumerate() {
            let start = b * per_band;
            let end = (start + per_band).min(usable);
            let mut mag = 0.0;
            for bin in &spectrum[start..end] {
                mag += bin.norm();
            }
            mag /= (end - start).max(1) as f32;
            // Normalize: FFT magnitudes scale with FFT_SIZE; log-compress.
            *out = (mag / (FFT_SIZE as f32 * 0.25)).min(1.0);
        }
        bands
    }
}

/// Average adjacent 64-band pairs down to a smaller band count.
fn downsample<const N: usize>(src: &[f32; BANDS_64]) -> [f32; N] {
    let mut out = [0.0f32; N];
    let group = BANDS_64 / N;
    for (i, o) in out.iter_mut().enumerate() {
        let slice = &src[i * group..(i + 1) * group];
        *o = slice.iter().copied().sum::<f32>() / group as f32;
    }
    out
}

/// De-interleave a callback buffer's newest samples into the per-channel
/// rings (mono sources feed both channels).
fn push_samples(
    data: &[f32],
    channels: usize,
    left: &Arc<Mutex<Vec<f32>>>,
    right: &Arc<Mutex<Vec<f32>>>,
) {
    let mut l = left.lock().unwrap();
    let mut r = right.lock().unwrap();
    for frame in data.chunks(channels) {
        l.push(frame[0]);
        r.push(if channels > 1 { frame[1] } else { frame[0] });
    }
    // Keep only the newest FFT_SIZE samples per channel.
    let trim = |v: &mut Vec<f32>| {
        if v.len() > FFT_SIZE {
            v.drain(0..v.len() - FFT_SIZE);
        }
    };
    trim(&mut l);
    trim(&mut r);
}

/// Prefer a desktop-output monitor (loopback) input; else the default input.
/// Device selection lives in the platform layer — "capture what the speakers
/// are playing" is spelled differently on every OS. See
/// [`crate::services::audio_devices`].
fn pick_input_device(host: &cpal::Host) -> Option<cpal::Device> {
    let (device, source) = crate::services::audio_devices::pick_capture_device(host)?;
    if source != crate::services::audio_devices::CaptureSource::Loopback {
        tracing::info!(
            target: "audio",
            "no desktop-loopback device found; falling back to {source:?} —              the spectrum will follow that input, not desktop audio"
        );
    }
    Some(device)
}

// ── Sound-object playback ─────────────────────────────────────────────────────

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Process-wide master gain (f32 bits) every wallpaper audio stream is scaled
/// by — the user's volume setting. Wallpaper Engine has one global volume, so
/// one atomic read on the audio thread is all the mixing this needs.
static MASTER_VOLUME: AtomicU32 = AtomicU32::new(0x3f80_0000); // 1.0
/// Silences every stream without tearing it down: set while the wallpaper is
/// paused by the playback gate, or muted by the pause policy.
static MUTED: AtomicBool = AtomicBool::new(false);

pub fn set_master_volume(v: f32) {
    let v = if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        1.0
    };
    MASTER_VOLUME.store(v.to_bits(), Ordering::Relaxed);
}

pub fn master_volume() -> f32 {
    f32::from_bits(MASTER_VOLUME.load(Ordering::Relaxed))
}

pub fn set_muted(muted: bool) {
    MUTED.store(muted, Ordering::Relaxed);
}

/// The gain a stream should apply right now: master volume, or 0 when muted.
pub fn output_gain() -> f32 {
    if MUTED.load(Ordering::Relaxed) {
        0.0
    } else {
        master_volume()
    }
}

/// Live controls for one playing sound, shared between the audio thread and
/// whoever drives it (the SceneScript sound API).
#[derive(Debug)]
pub struct SoundControl {
    /// Per-sound volume (f32 bits), before the master gain.
    volume: AtomicU32,
    playing: AtomicBool,
    /// Set to rewind to the first sample on the next callback.
    rewind: AtomicBool,
}

impl SoundControl {
    fn new(volume: f32, playing: bool) -> Self {
        Self {
            volume: AtomicU32::new(volume.to_bits()),
            playing: AtomicBool::new(playing),
            rewind: AtomicBool::new(false),
        }
    }
    pub fn volume(&self) -> f32 {
        f32::from_bits(self.volume.load(Ordering::Relaxed))
    }
    pub fn set_volume(&self, v: f32) {
        let v = if v.is_finite() {
            v.clamp(0.0, 4.0)
        } else {
            1.0
        };
        self.volume.store(v.to_bits(), Ordering::Relaxed);
    }
    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }
    pub fn play(&self) {
        self.playing.store(true, Ordering::Relaxed);
    }
    pub fn pause(&self) {
        self.playing.store(false, Ordering::Relaxed);
    }
    pub fn stop(&self) {
        self.playing.store(false, Ordering::Relaxed);
        self.rewind.store(true, Ordering::Relaxed);
    }
}

/// Plays a WE `sound` object's audio file (decoded via ffmpeg) out the default
/// output device, optionally looped. One stream per sound; the OS mixer sums
/// them. Dropping it stops playback.
pub struct SoundPlayback {
    _stream: cpal::Stream,
    control: Arc<SoundControl>,
}

impl SoundPlayback {
    /// Decode `path` and start playback (or hold it ready but silent when
    /// `start_playing` is false — WE's `startsilent`). Returns `None` if
    /// there's no output device or the file can't be decoded — sound is
    /// always best-effort.
    pub fn start(
        path: &std::path::Path,
        volume: f32,
        looping: bool,
        start_playing: bool,
    ) -> Option<Self> {
        let samples = decode_audio_file_for_default_output(path)?;
        Self::from_samples(
            samples,
            volume,
            looping,
            start_playing,
            &path.display().to_string(),
        )
    }

    fn from_samples(
        (samples, config): (Vec<f32>, cpal::SupportedStreamConfig),
        volume: f32,
        looping: bool,
        start_playing: bool,
        label: &str,
    ) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let device = cpal::default_host().default_output_device()?;
        let control = Arc::new(SoundControl::new(volume.clamp(0.0, 4.0), start_playing));
        let ctl = control.clone();
        let mut pos = 0usize;
        let stream = device
            .build_output_stream(
                &config.into(),
                move |out: &mut [f32], _| {
                    if ctl.rewind.swap(false, Ordering::Relaxed) {
                        pos = 0;
                    }
                    if !ctl.is_playing() {
                        out.fill(0.0);
                        return;
                    }
                    let gain = ctl.volume() * output_gain();
                    for s in out.iter_mut() {
                        if pos >= samples.len() {
                            if looping {
                                pos = 0;
                            } else {
                                // A one-shot sound finished: park it so a
                                // script's `play()` starts it from the top.
                                ctl.playing.store(false, Ordering::Relaxed);
                                ctl.rewind.store(true, Ordering::Relaxed);
                                *s = 0.0;
                                continue;
                            }
                        }
                        *s = samples[pos] * gain;
                        pos += 1;
                    }
                },
                |e| tracing::warn!(target: "audio", "sound stream error: {e}"),
                None,
            )
            .ok()?;
        stream.play().ok()?;
        tracing::info!(target: "audio", "sound '{label}' ready (vol {volume:.2}, loop {looping}, playing {start_playing})");
        Some(Self {
            _stream: stream,
            control,
        })
    }

    pub fn control(&self) -> Arc<SoundControl> {
        self.control.clone()
    }
}

/// Decode `path` for the default output device's native format.
fn decode_audio_file_for_default_output(
    path: &std::path::Path,
) -> Option<(Vec<f32>, cpal::SupportedStreamConfig)> {
    let device = cpal::default_host().default_output_device()?;
    let config = device.default_output_config().ok()?;
    let samples = decode_audio_file(path, config.sample_rate().0, config.channels() as usize)?;
    Some((samples, config))
}

/// A file's audio track playing on its own thread (a video wallpaper's
/// soundtrack). Decodes incrementally rather than up front — videos can run
/// for many minutes — and loops at the end. Dropping it stops playback.
pub struct StreamingAudio {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for StreamingAudio {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl StreamingAudio {
    /// Start playing `path`'s best audio stream; `None` when it has none or
    /// there is no output device.
    pub fn start(path: &std::path::Path) -> Option<Self> {
        ffmpeg_next::init().ok()?;
        let has_audio = ffmpeg_next::format::input(&path)
            .ok()?
            .streams()
            .best(ffmpeg_next::media::Type::Audio)
            .is_some();
        if !has_audio {
            return None;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let path = path.to_path_buf();
        let flag = stop.clone();
        // cpal streams aren't Send: the thread that decodes also owns the
        // output stream, and lives until `stop`.
        let thread = std::thread::Builder::new()
            .name("video-audio".into())
            .spawn(move || {
                if let Err(e) = stream_file_audio(&path, &flag) {
                    tracing::warn!(target: "audio", "video audio for {}: {e}", path.display());
                }
            })
            .ok()?;
        Some(Self {
            stop,
            thread: Some(thread),
        })
    }
}

fn stream_file_audio(path: &std::path::Path, stop: &AtomicBool) -> anyhow::Result<()> {
    use ffmpeg_next::util::format::sample::{Sample, Type};
    use std::sync::mpsc::{sync_channel, TryRecvError};

    let device = cpal::default_host()
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("no audio output device"))?;
    let config = device.default_output_config()?;
    let out_rate = config.sample_rate().0;
    let out_ch = config.channels() as usize;

    let (tx, rx) = sync_channel::<Vec<f32>>(16);
    let mut chunk: Vec<f32> = Vec::new();
    let mut pos = 0usize;
    let stream = device.build_output_stream(
        &config.into(),
        move |out: &mut [f32], _| {
            let gain = output_gain();
            for s in out.iter_mut() {
                if pos >= chunk.len() {
                    match rx.try_recv() {
                        Ok(next) => {
                            chunk = next;
                            pos = 0;
                        }
                        Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => {
                            *s = 0.0;
                            continue;
                        }
                    }
                }
                *s = chunk.get(pos).copied().unwrap_or(0.0) * gain;
                pos += 1;
            }
        },
        |e| tracing::warn!(target: "audio", "video audio stream error: {e}"),
        None,
    )?;
    stream.play()?;

    let mut ictx = ffmpeg_next::format::input(&path)?;
    let audio = ictx
        .streams()
        .best(ffmpeg_next::media::Type::Audio)
        .ok_or_else(|| anyhow::anyhow!("no audio stream"))?;
    let index = audio.index();
    let mut decoder = ffmpeg_next::codec::context::Context::from_parameters(audio.parameters())?
        .decoder()
        .audio()?;
    let out_layout = if out_ch >= 2 {
        ffmpeg_next::util::channel_layout::ChannelLayout::STEREO
    } else {
        ffmpeg_next::util::channel_layout::ChannelLayout::MONO
    };
    let in_layout = if decoder.channel_layout().is_empty() {
        if decoder.channels() >= 2 {
            ffmpeg_next::util::channel_layout::ChannelLayout::STEREO
        } else {
            ffmpeg_next::util::channel_layout::ChannelLayout::MONO
        }
    } else {
        decoder.channel_layout()
    };
    let mut resampler = ffmpeg_next::software::resampling::Context::get(
        decoder.format(),
        in_layout,
        decoder.rate(),
        Sample::F32(Type::Packed),
        out_layout,
        out_rate,
    )?;

    let mut frame = ffmpeg_next::frame::Audio::empty();
    let mut resampled = ffmpeg_next::frame::Audio::empty();
    // Send a resampled frame; false once the stream should end.
    let mut push = |resampled: &ffmpeg_next::frame::Audio| -> bool {
        let n = resampled.samples() * out_ch.min(2);
        let data: &[f32] = bytemuck::cast_slice(resampled.data(0));
        let mut samples = data[..n.min(data.len())].to_vec();
        // A device with more than two channels gets stereo on its first two.
        if out_ch > 2 {
            samples = samples
                .chunks(2)
                .flat_map(|lr| {
                    let mut f = vec![0.0; out_ch];
                    f[0] = lr[0];
                    f[1] = *lr.get(1).unwrap_or(&lr[0]);
                    f
                })
                .collect();
        }
        loop {
            if stop.load(Ordering::Relaxed) {
                return false;
            }
            match tx.try_send(samples) {
                Ok(()) => return true,
                Err(std::sync::mpsc::TrySendError::Full(back)) => {
                    samples = back;
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => return false,
            }
        }
    };

    loop {
        for (s, packet) in ictx.packets() {
            if s.index() != index || decoder.send_packet(&packet).is_err() {
                continue;
            }
            while decoder.receive_frame(&mut frame).is_ok() {
                if resampler.run(&frame, &mut resampled).is_ok() && !push(&resampled) {
                    return Ok(());
                }
            }
        }
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        // Loop: back to the start without re-opening the file.
        ictx.seek(0, ..)?;
        decoder.flush();
    }
}

/// Decode an audio file to interleaved f32 samples at `out_rate`/`out_ch` using
/// ffmpeg's decoder + software resampler. Returns `None` on any failure.
fn decode_audio_file(path: &std::path::Path, out_rate: u32, out_ch: usize) -> Option<Vec<f32>> {
    use ffmpeg_next::util::format::sample::{Sample, Type};
    ffmpeg_next::init().ok()?;
    let mut ictx = ffmpeg_next::format::input(&path).ok()?;
    let stream = ictx.streams().best(ffmpeg_next::media::Type::Audio)?;
    let stream_index = stream.index();
    let mut decoder = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
        .ok()?
        .decoder()
        .audio()
        .ok()?;

    let out_layout = if out_ch >= 2 {
        ffmpeg_next::util::channel_layout::ChannelLayout::STEREO
    } else {
        ffmpeg_next::util::channel_layout::ChannelLayout::MONO
    };
    let in_layout = {
        let l = decoder.channel_layout();
        if l.is_empty() {
            // Some decoders don't set a layout until the first frame; guess.
            if decoder.channels() >= 2 {
                ffmpeg_next::util::channel_layout::ChannelLayout::STEREO
            } else {
                ffmpeg_next::util::channel_layout::ChannelLayout::MONO
            }
        } else {
            l
        }
    };
    let mut resampler = ffmpeg_next::software::resampling::Context::get(
        decoder.format(),
        in_layout,
        decoder.rate(),
        Sample::F32(Type::Packed),
        out_layout,
        out_rate,
    )
    .ok()?;

    let mut out = Vec::new();
    let mut drain = |resampler: &mut ffmpeg_next::software::resampling::Context,
                     frame: &ffmpeg_next::frame::Audio,
                     out: &mut Vec<f32>| {
        let mut resampled = ffmpeg_next::frame::Audio::empty();
        if resampler.run(frame, &mut resampled).is_ok() {
            let n = resampled.samples() * out_ch;
            let data: &[f32] = bytemuck::cast_slice(resampled.data(0));
            out.extend_from_slice(&data[..n.min(data.len())]);
        }
    };
    for (s, packet) in ictx.packets() {
        if s.index() != stream_index {
            continue;
        }
        if decoder.send_packet(&packet).is_err() {
            continue;
        }
        let mut frame = ffmpeg_next::frame::Audio::empty();
        while decoder.receive_frame(&mut frame).is_ok() {
            drain(&mut resampler, &frame, &mut out);
        }
    }
    let _ = decoder.send_eof();
    let mut frame = ffmpeg_next::frame::Audio::empty();
    while decoder.receive_frame(&mut frame).is_ok() {
        drain(&mut resampler, &frame, &mut out);
    }
    (!out.is_empty()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn master_volume_clamps_and_mute_zeroes_gain() {
        set_master_volume(2.0);
        assert_eq!(master_volume(), 1.0);
        set_master_volume(0.25);
        assert_eq!(output_gain(), 0.25);
        set_muted(true);
        assert_eq!(output_gain(), 0.0);
        set_muted(false);
        set_master_volume(f32::NAN);
        assert_eq!(master_volume(), 1.0);
    }

    #[test]
    fn sound_control_stop_rewinds_and_pauses() {
        let c = SoundControl::new(0.5, true);
        c.stop();
        assert!(!c.is_playing());
        assert!(c.rewind.load(Ordering::Relaxed));
        c.set_volume(9.0);
        assert_eq!(c.volume(), 4.0);
    }

    #[test]
    fn uniform_bytes_layout_is_std430() {
        let mut s = AudioSpectrum::default();
        s.s16_left[0] = 1.0;
        s.s64_right[63] = 0.5;
        let bytes = s.to_uniform_bytes();
        assert_eq!(bytes.len(), UNIFORM_BYTES);
        // First array element sits at offset 0.
        assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), 1.0);
        // Last element of the last array (tight 4-byte std430 stride):
        // 16L+16R+32L+32R+64L done, then the 64th of 64Right.
        let off = (BANDS_16 * 2 + BANDS_32 * 2 + BANDS_64 + 63) * 4;
        assert_eq!(
            f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()),
            0.5
        );
    }

    #[test]
    fn downsample_averages_groups() {
        let mut src = [0.0f32; BANDS_64];
        src[0] = 1.0;
        src[1] = 1.0;
        src[2] = 0.0;
        src[3] = 0.0;
        // 16-band: each band groups 4 of 64 → band 0 = (1+1+0+0)/4 = 0.5.
        let out: [f32; BANDS_16] = downsample(&src);
        assert_eq!(out[0], 0.5);
        assert_eq!(out[1], 0.0);
    }
}
