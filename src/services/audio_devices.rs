//! Platform-specific capture-device selection for the audio-reactive path.
//!
//! Capture, FFT and the `g_AudioSpectrum*` uniforms are all cross-platform and
//! live in [`crate::services::audio`] on top of cpal. The one genuinely
//! OS-dependent decision is *which* device carries desktop output, because
//! "record what the speakers are playing" is spelled differently everywhere:
//!
//! - **Linux** (PulseAudio/PipeWire): output sinks expose a paired `.monitor`
//!   source. Real loopback, present by default.
//! - **macOS**: the OS ships no loopback source. It needs a virtual device
//!   (BlackHole, Soundflower, Loopback) that the user has installed and set as
//!   an output, so we match those by name.
//! - **Windows**: WASAPI exposes loopback on the *output* device rather than as
//!   an input, so the default output is the right handle.
//!
//! Falling through to the default input device (usually a microphone) is
//! deliberate: a mic still makes the wallpaper react to sound, which is closer
//! to the intent than silence. The fallback is logged, because "reacting to the
//! room" and "reacting to the music" look similar enough to hide a
//! misconfiguration for a long time.
//!
//! On Linux the hint list alone isn't enough: cpal's ALSA backend enumerates
//! `pipewire`/`pulse`/`default`, never the per-sink `.monitor` sources, so a
//! substring match finds nothing on a PulseAudio/PipeWire desktop. But ALSA's
//! `pulse` plugin honours the `PULSE_SOURCE` environment variable, so naming
//! the monitor there and opening `pulse` gets real desktop capture without a
//! libpulse dependency. See [`route_pulse_monitor`].

use cpal::traits::{DeviceTrait, HostTrait};

/// Device-name fragments that indicate a desktop-output capture source on this
/// platform, in priority order. Matched case-insensitively as substrings.
#[cfg(target_os = "linux")]
const LOOPBACK_HINTS: &[&str] = &["monitor"];

#[cfg(target_os = "macos")]
const LOOPBACK_HINTS: &[&str] = &["blackhole", "soundflower", "loopback", "aggregate"];

#[cfg(target_os = "windows")]
const LOOPBACK_HINTS: &[&str] = &["stereo mix", "what u hear", "loopback"];

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const LOOPBACK_HINTS: &[&str] = &["monitor", "loopback"];

/// How the chosen device was found — for logging, and so callers can tell a
/// real desktop capture from a microphone fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureSource {
    /// A real desktop-output loopback: the wallpaper reacts to what's playing.
    Loopback,
    /// Whatever the OS calls the default input — usually a microphone.
    DefaultInput,
    /// Windows-style loopback, which hangs off the output device.
    OutputLoopback,
}

/// User-chosen capture device, set from the UI. `None` = automatic.
///
/// Global because the renderer builds its `AudioCapture` deep inside a scene
/// load on its own thread, with no path to thread a setting through — the same
/// reason `workshop::properties` keeps its overrides this way.
static PREFERRED: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

/// Override automatic selection. Pass `None` to go back to automatic.
/// Takes effect the next time a wallpaper is applied.
pub fn set_preferred_device(name: Option<String>) {
    if let Ok(mut w) = PREFERRED.write() {
        *w = name;
    }
}

pub fn preferred_device() -> Option<String> {
    PREFERRED.read().ok().and_then(|g| g.clone())
}

/// Speakers to offer in the UI.
///
/// These are *output* devices — the thing the user thinks of as "where sound
/// comes out" — because that's the mental model the OS presents. That we
/// capture desktop audio by recording an output's monitor source is an
/// implementation detail the picker never exposes.
pub fn list_audio_outputs() -> Vec<CaptureOption> {
    let mut out = vec![CaptureOption {
        label: "Automatic".to_string(),
        device: None,
    }];
    #[cfg(target_os = "linux")]
    {
        let default = default_sink();
        for (name, description) in pulse_sinks() {
            let is_default = Some(&name) == default.as_ref();
            out.push(CaptureOption {
                label: if is_default {
                    format!("{description}  (default)")
                } else {
                    description
                },
                device: Some(name),
            });
        }
    }
    // Non-Linux: cpal's output devices are already user-facing names.
    #[cfg(not(target_os = "linux"))]
    if let Ok(devices) = cpal::default_host().output_devices() {
        for name in devices.filter_map(|d| d.name().ok()) {
            out.push(CaptureOption {
                label: name.clone(),
                device: Some(name),
            });
        }
    }
    out
}

/// One selectable entry for the device picker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureOption {
    /// Shown to the user.
    pub label: String,
    /// `None` = automatic selection.
    pub device: Option<String>,
}

/// Whether a wallpaper drives anything from audio, so the UI only offers a
/// device picker where it can matter.
///
/// Reads `general.supportsaudioprocessing` (WE's own flag) and falls back to
/// scanning for per-effect `audioprocessing` keys, which some scenes set
/// without the general flag.
pub fn wallpaper_uses_audio(dir: &std::path::Path) -> bool {
    let json = std::fs::read_to_string(dir.join("scene.json"))
        .ok()
        .or_else(|| {
            let pkg = crate::formats::pkg::Package::from_file(&dir.join("scene.pkg")).ok()?;
            pkg.get("scene.json")
                .map(|b| String::from_utf8_lossy(b).into_owned())
        });
    let Some(json) = json else { return false };
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
        if v.pointer("/general/supportsaudioprocessing")
            .and_then(|x| x.as_bool())
            == Some(true)
        {
            return true;
        }
    }
    // Case matters: scenes spell the per-effect combo `"AUDIOPROCESSING"` in
    // caps and the general flag in lowercase.
    json.to_lowercase().contains("audioprocessing")
}

/// `(sink name, human description)` for every output PulseAudio/PipeWire knows.
/// The description is what the OS's own volume UI shows.
#[cfg(target_os = "linux")]
fn pulse_sinks() -> Vec<(String, String)> {
    let Some(out) = std::process::Command::new("pactl")
        .arg("list")
        .arg("sinks")
        .output()
        .ok()
        .filter(|o| o.status.success())
    else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let mut sinks = Vec::new();
    let mut name: Option<String> = None;
    for line in text.lines().map(str::trim) {
        if let Some(v) = line.strip_prefix("Name: ") {
            name = Some(v.to_string());
        } else if let Some(v) = line.strip_prefix("Description: ") {
            // Description always follows its Name in pactl's output.
            if let Some(n) = name.take() {
                sinks.push((n, v.to_string()));
            }
        }
    }
    sinks
}

#[cfg(target_os = "linux")]
fn default_sink() -> Option<String> {
    let out = std::process::Command::new("pactl")
        .arg("get-default-sink")
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

/// Honour a user-chosen device: either a PulseAudio monitor source (routed
/// through `PULSE_SOURCE`) or a cpal input device by name.
fn use_preferred(host: &cpal::Host) -> Option<(cpal::Device, CaptureSource)> {
    let want = preferred_device()?;
    // The UI stores a speaker (sink); capture comes from its monitor source.
    #[cfg(target_os = "linux")]
    {
        let monitor = if want.ends_with(".monitor") {
            want.clone()
        } else {
            format!("{want}.monitor")
        };
        if let Some(device) = host
            .input_devices()
            .ok()
            .and_then(|mut d| d.find(|d| d.name().map(|n| n == "pulse").unwrap_or(false)))
        {
            std::env::set_var("PULSE_SOURCE", &monitor);
            tracing::info!(target: "audio", "capturing from selected speaker '{want}'");
            return Some((device, CaptureSource::Loopback));
        }
    }
    let device = host
        .input_devices()
        .ok()?
        .find(|d| d.name().map(|n| n == want).unwrap_or(false))?;
    tracing::info!(target: "audio", "using user-selected capture device '{want}'");
    Some((device, CaptureSource::DefaultInput))
}

/// Pick the best capture device for audio reactivity on this platform.
///
/// Returns the device plus how it was found. `None` means the machine exposes
/// no usable capture device at all, and the caller keeps a silent spectrum.
pub fn pick_capture_device(host: &cpal::Host) -> Option<(cpal::Device, CaptureSource)> {
    // An explicit user choice always wins over auto-detection.
    if let Some(picked) = use_preferred(host) {
        return Some(picked);
    }
    // Preferred on Linux: point ALSA's pulse plugin at the default sink's
    // monitor. Tried first because it's the only path that reaches real
    // desktop audio on PipeWire.
    #[cfg(target_os = "linux")]
    if let Some(d) = route_pulse_monitor(host) {
        return Some((d, CaptureSource::Loopback));
    }
    if let Some(d) = match_hint(host) {
        return Some((d, CaptureSource::Loopback));
    }
    // WASAPI does loopback through the render endpoint, so the default output
    // is a capture handle there — but not on hosts where it isn't.
    #[cfg(target_os = "windows")]
    if let Some(d) = host.default_output_device() {
        return Some((d, CaptureSource::OutputLoopback));
    }
    host.default_input_device()
        .map(|d| (d, CaptureSource::DefaultInput))
}

/// Route ALSA's `pulse` device at the default sink's `.monitor` source.
///
/// `pactl` ships with PulseAudio/PipeWire, so it's present wherever this can
/// work at all; a missing binary just means we fall through. Setting
/// `PULSE_SOURCE` is process-global and must happen before the stream opens,
/// which is why it lives here rather than at the call site.
#[cfg(target_os = "linux")]
fn route_pulse_monitor(host: &cpal::Host) -> Option<cpal::Device> {
    let sink = std::process::Command::new("pactl")
        .arg("get-default-sink")
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let sink = String::from_utf8(sink.stdout).ok()?.trim().to_string();
    if sink.is_empty() {
        return None;
    }
    let monitor = format!("{sink}.monitor");

    // Confirm the monitor actually exists before claiming Loopback — a stale
    // or renamed sink would otherwise open a stream that never delivers.
    let sources = std::process::Command::new("pactl")
        .args(["list", "short", "sources"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    if !String::from_utf8_lossy(&sources.stdout).contains(&monitor) {
        tracing::debug!(target: "audio", "monitor source '{monitor}' not present");
        return None;
    }

    let device = host
        .input_devices()
        .ok()?
        .find(|d| d.name().map(|n| n == "pulse").unwrap_or(false))?;
    // SAFETY-adjacent note: single-threaded setup path, set before any audio
    // stream exists, and the ALSA pulse plugin reads it at stream open.
    std::env::set_var("PULSE_SOURCE", &monitor);
    tracing::info!(target: "audio", "capturing desktop audio from '{monitor}'");
    Some(device)
}

fn match_hint(host: &cpal::Host) -> Option<cpal::Device> {
    let devices = host.input_devices().ok()?;
    // Collect once: `Devices` is a one-shot iterator, and we need several
    // passes to honour hint priority rather than device enumeration order.
    let named: Vec<(String, cpal::Device)> = devices
        .filter_map(|d| Some((d.name().ok()?.to_lowercase(), d)))
        .collect();
    LOOPBACK_HINTS.iter().find_map(|hint| {
        named
            .iter()
            .find(|(name, _)| name.contains(hint))
            .map(|(_, d)| d.clone())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal on-disk wallpaper carrying `scene.json`, for the detector.
    struct Dir(std::path::PathBuf);
    impl Dir {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tempdir_with(scene_json: &str) -> Dir {
        let p = std::env::temp_dir().join(format!("wpaudio{}", std::process::id()));
        let _ = std::fs::create_dir_all(&p);
        std::fs::write(p.join("scene.json"), scene_json).expect("write scene.json");
        Dir(p)
    }

    /// Every platform must offer at least one hint, or `match_hint` silently
    /// degrades to the microphone fallback on that OS.
    #[test]
    fn platform_has_loopback_hints() {
        assert!(!LOOPBACK_HINTS.is_empty());
        assert!(
            LOOPBACK_HINTS.iter().all(|h| *h == h.to_lowercase()),
            "hints are matched against a lowercased name, so they must be lowercase"
        );
    }

    /// The combo key is spelled `AUDIOPROCESSING` in caps inside `combos`,
    /// while the general flag is lowercase — matching one case detects
    /// neither reliably (a case-sensitive check found 0 of 197 scenes).
    #[test]
    fn audio_detection_is_case_insensitive() {
        let dir = tempdir_with(
            r#"{"general":{},"objects":[
            {"effects":[{"passes":[{"combos":{"AUDIOPROCESSING":3}}]}]}]}"#,
        );
        assert!(wallpaper_uses_audio(dir.path()));
    }

    /// The picker is a speaker list, so no plumbing may leak into a label:
    /// no `.monitor` suffix, no `alsa_output.`/`bluez_output.` prefixes.
    /// Users pick "Echo Dot-0TS", not "bluez_output.28_73_….1.monitor".
    #[test]
    fn labels_expose_no_implementation_detail() {
        for opt in list_audio_outputs() {
            for leak in [".monitor", "alsa_output.", "bluez_output.", "alsa_input."] {
                assert!(
                    !opt.label.contains(leak),
                    "label '{}' leaks '{leak}'",
                    opt.label
                );
            }
        }
    }

    /// "Automatic" must stay first so the default selection index (0) means
    /// automatic.
    #[test]
    fn automatic_is_the_first_option() {
        let outs = list_audio_outputs();
        assert_eq!(outs[0].label, "Automatic");
        assert!(outs[0].device.is_none());
    }

    /// Hint order is priority order, so the list must not contain a substring
    /// of an earlier entry (which could never be reached).
    #[test]
    fn hints_are_not_shadowed_by_earlier_ones() {
        for (i, h) in LOOPBACK_HINTS.iter().enumerate() {
            for earlier in &LOOPBACK_HINTS[..i] {
                assert!(
                    !h.contains(earlier),
                    "'{h}' is unreachable: '{earlier}' matches it first"
                );
            }
        }
    }
}
