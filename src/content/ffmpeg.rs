use anyhow::{anyhow, Result};
use image::RgbaImage;
use std::path::Path;
use std::sync::mpsc::SyncSender;
use std::sync::Arc;

/// Decode every frame of the video file and send each as an `Arc<RgbaImage>`
/// through `tx`, paced to the video's own timestamps. At the end it seeks
/// back to the start (no re-open, so no hitch) and keeps going. Exits when
/// `tx.send()` returns `Err` (receiver dropped).
pub fn video_decode_loop(path: &Path, tx: &SyncSender<Arc<RgbaImage>>) -> Result<()> {
    use ffmpeg_next::{
        codec::context::Context as CodecCtx,
        format,
        format::Pixel,
        media::Type,
        software::scaling::{context::Context as ScaleCtx, flag::Flags},
        util::frame::video::Video as VideoFrame,
    };
    use std::thread;
    use std::time::{Duration, Instant};

    ffmpeg_next::init().map_err(|e| anyhow!("FFmpeg init: {e}"))?;

    let mut ictx =
        format::input(&path).map_err(|e| anyhow!("cannot open '{}': {e}", path.display()))?;
    let stream_idx = ictx
        .streams()
        .best(Type::Video)
        .ok_or_else(|| anyhow!("no video stream in '{}'", path.display()))?
        .index();
    let time_base = ictx.stream(stream_idx).unwrap().time_base();

    let mut decoder = {
        let stream = ictx.stream(stream_idx).unwrap();
        let mut ctx = CodecCtx::from_parameters(stream.parameters())
            .map_err(|e| anyhow!("codec context: {e}"))?;
        // Let FFmpeg decode on every core (frame threading; 0 = auto).
        ctx.set_threading(ffmpeg_next::threading::Config {
            kind: ffmpeg_next::threading::Type::Frame,
            count: 0,
        });
        ctx.decoder()
            .video()
            .map_err(|e| anyhow!("video decoder: {e}"))?
    };

    // Decoded pixel format → packed RGBA8 at the video's own size; scaling
    // to each output happens in the platform layer.
    let mut scaler = ScaleCtx::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        Pixel::RGBA,
        decoder.width(),
        decoder.height(),
        Flags::BILINEAR,
    )
    .map_err(|e| anyhow!("scaler context: {e}"))?;

    let mut raw = VideoFrame::empty();
    let mut scaled = VideoFrame::empty();
    let (tb_num, tb_den) = (time_base.numerator() as f64, time_base.denominator() as f64);

    loop {
        // PTS pacing state, reset each pass.
        let mut first_pts: Option<i64> = None;
        let mut loop_start: Option<Instant> = None;
        let mut emit = |raw: &VideoFrame, scaled: &mut VideoFrame| -> bool {
            if let (Some(pts), true) = (raw.pts(), tb_den > 0.0) {
                let first = *first_pts.get_or_insert(pts);
                let start = *loop_start.get_or_insert_with(Instant::now);
                let offset =
                    Duration::from_secs_f64(((pts - first) as f64 * tb_num / tb_den).max(0.0));
                if let Some(wait) = offset.checked_sub(start.elapsed()) {
                    thread::sleep(wait);
                }
            }
            if scaler.run(raw, scaled).is_err() {
                return true;
            }
            match frame_to_rgba(scaled) {
                // `send` blocks when the channel is full (back-pressure) and
                // fails once the receiver is gone.
                Some(img) => tx.send(Arc::new(img)).is_ok(),
                None => true,
            }
        };

        for (stream, packet) in ictx.packets() {
            if stream.index() != stream_idx || decoder.send_packet(&packet).is_err() {
                continue;
            }
            while decoder.receive_frame(&mut raw).is_ok() {
                if !emit(&raw, &mut scaled) {
                    return Ok(());
                }
            }
        }
        // Drain frames still buffered in the (threaded) decoder.
        let _ = decoder.send_eof();
        while decoder.receive_frame(&mut raw).is_ok() {
            if !emit(&raw, &mut scaled) {
                return Ok(());
            }
        }

        ictx.seek(0, ..)
            .map_err(|e| anyhow!("seeking '{}' to loop: {e}", path.display()))?;
        decoder.flush();
    }
}

/// Decode the first video frame from `path` and return it as an `RgbaImage`.
/// Useful for extracting a static thumbnail from a video texture layer.
pub fn decode_first_frame(path: &Path) -> Result<RgbaImage> {
    use ffmpeg_next::{
        codec::context::Context as CodecCtx,
        format,
        format::Pixel,
        media::Type,
        software::scaling::{context::Context as ScaleCtx, flag::Flags},
        util::frame::video::Video as VideoFrame,
    };

    ffmpeg_next::init().map_err(|e| anyhow!("FFmpeg init: {e}"))?;

    let mut ictx =
        format::input(&path).map_err(|e| anyhow!("cannot open '{}': {e}", path.display()))?;

    let stream_idx = ictx
        .streams()
        .best(Type::Video)
        .ok_or_else(|| anyhow!("no video stream in '{}'", path.display()))?
        .index();

    let mut decoder = {
        let stream = ictx.stream(stream_idx).unwrap();
        CodecCtx::from_parameters(stream.parameters())
            .map_err(|e| anyhow!("codec context: {e}"))?
            .decoder()
            .video()
            .map_err(|e| anyhow!("video decoder: {e}"))?
    };

    let mut scaler = ScaleCtx::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        Pixel::RGBA,
        decoder.width(),
        decoder.height(),
        Flags::BILINEAR,
    )
    .map_err(|e| anyhow!("scaler context: {e}"))?;

    let mut raw = VideoFrame::empty();
    let mut scaled = VideoFrame::empty();

    for (stream, packet) in ictx.packets() {
        if stream.index() != stream_idx {
            continue;
        }
        if decoder.send_packet(&packet).is_err() {
            continue;
        }
        while decoder.receive_frame(&mut raw).is_ok() {
            if scaler.run(&raw, &mut scaled).is_ok() {
                if let Some(img) = frame_to_rgba(&scaled) {
                    return Ok(img);
                }
            }
        }
    }

    Err(anyhow!(
        "no decodable video frame found in '{}'",
        path.display()
    ))
}

/// Copy one RGBA `VideoFrame` into an `RgbaImage`, trimming stride padding.
fn frame_to_rgba(frame: &ffmpeg_next::util::frame::video::Video) -> Option<RgbaImage> {
    let w = frame.width();
    let h = frame.height();
    let stride = frame.stride(0);
    let data = frame.data(0);

    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h as usize {
        let start = row * stride;
        pixels.extend_from_slice(&data[start..start + w as usize * 4]);
    }

    RgbaImage::from_raw(w, h, pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1 s, 10 fps clip loops: more frames arrive than it holds, without
    /// the decoder exiting. Needs the `ffmpeg` CLI to make the clip.
    #[test]
    fn video_loops_past_its_end() {
        let dir = std::env::temp_dir().join(format!("wp-video-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let clip = dir.join("clip.mp4");
        let made = std::process::Command::new("ffmpeg")
            .args([
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=64x48:rate=10",
            ])
            .args(["-f", "lavfi", "-i", "sine=duration=1", "-shortest"])
            .arg(&clip)
            .status()
            .is_ok_and(|s| s.success());
        if !made {
            eprintln!("ffmpeg CLI unavailable; skipping");
            return;
        }
        let (tx, rx) = std::sync::mpsc::sync_channel(2);
        let path = clip.clone();
        std::thread::spawn(move || video_decode_loop(&path, &tx));
        let start = std::time::Instant::now();
        let mut frames = 0;
        while frames < 25 && start.elapsed() < std::time::Duration::from_secs(10) {
            if rx.recv_timeout(std::time::Duration::from_secs(2)).is_ok() {
                frames += 1;
            } else {
                break;
            }
        }
        drop(rx);
        assert_eq!(frames, 25, "decoder kept producing frames across the loop");
        // ~2.5 s of a 10 fps clip: paced, not decoded as fast as possible.
        assert!(start.elapsed() >= std::time::Duration::from_millis(2000));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
