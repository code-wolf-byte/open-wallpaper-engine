use anyhow::{anyhow, Result};
use image::RgbaImage;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::thread;

use super::content::WallpaperContent;

/// A live source of RGBA frames for the Wayland renderer.
///
/// `Static` always returns the same frame. `Video` owns a background decoder
/// thread that streams decoded frames through a bounded channel (size 2),
/// providing natural back-pressure so the decoder never runs more than two
/// frames ahead of the display.
pub enum FrameSource {
    Static(Arc<RgbaImage>),
    Video {
        /// The most recently decoded frame available to draw.
        current: Arc<RgbaImage>,
        /// Receive end of the bounded channel from the decoder thread.
        rx: Receiver<Arc<RgbaImage>>,
        /// `Some` only for a web wallpaper — lets the platform layer forward
        /// mouse input into the embedded page (see `render::web::WebInputEvent`).
        /// `None` for real video and CPU-scene-fallback, which share this
        /// same variant but have no page to receive input.
        web_input: Option<SyncSender<super::web::WebInputEvent>>,
        /// A video's soundtrack, playing while this source lives.
        audio: Option<crate::engine::audio::StreamingAudio>,
    },
}

impl FrameSource {
    /// Build a `FrameSource` from parsed `WallpaperContent`.
    ///
    /// For video this blocks until the first frame is decoded so the wallpaper
    /// surface is never shown blank.
    pub fn from_content(content: WallpaperContent) -> Result<Self> {
        Self::from_content_with(content, super::web::WebOptions::default())
    }

    /// [`Self::from_content`] with a web wallpaper's page size and frame
    /// rate (the output it will show on).
    pub fn from_content_with(
        content: WallpaperContent,
        web: super::web::WebOptions,
    ) -> Result<Self> {
        match content {
            WallpaperContent::Static(img) => Ok(FrameSource::Static(img)),
            WallpaperContent::Scene { dir } => {
                let (tx, rx) = sync_channel::<Arc<RgbaImage>>(2);
                let target_fps = 30.0;

                thread::spawn(move || {
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        crate::engine::gpu_renderer::gpu_scene_render_loop(&dir, &tx, target_fps)
                    }));
                    match r {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => {
                            tracing::warn!(target: "render", "GPU scene render failed, falling back to CPU: {e}");
                            if let Err(e2) =
                                crate::engine::animated::scene_render_loop(&dir, &tx, target_fps)
                            {
                                tracing::error!(target: "render", "CPU scene render error: {e2}");
                            }
                        }
                        Err(panic_val) => {
                            let msg = panic_val
                                .downcast_ref::<String>()
                                .cloned()
                                .or_else(|| panic_val.downcast_ref::<&str>().map(|s| s.to_string()))
                                .unwrap_or_else(|| "(unknown panic)".into());
                            tracing::error!(target: "render", "GPU render thread panicked: {msg} — falling back to CPU");
                            if let Err(e2) =
                                crate::engine::animated::scene_render_loop(&dir, &tx, target_fps)
                            {
                                tracing::error!(target: "render", "CPU scene render error: {e2}");
                            }
                        }
                    }
                });

                let first = rx
                    .recv()
                    .map_err(|_| anyhow!("scene renderer exited before producing any frames"))?;

                Ok(FrameSource::Video {
                    current: first,
                    rx,
                    web_input: None,
                    audio: None,
                })
            }
            // Web wallpapers reuse the `Video` variant rather than adding one:
            // it already means "a stream of frames plus the latest", which is
            // exactly what CEF's off-screen painting produces. Scene fallback
            // rendering shares it for the same reason.
            WallpaperContent::Web { html } => {
                let (first, rx, input_tx) = super::web::start_web_stream(&html, web)?;
                Ok(FrameSource::Video {
                    current: Arc::new(first),
                    rx,
                    web_input: Some(input_tx),
                    audio: None,
                })
            }
            WallpaperContent::Video { path } => {
                let (tx, rx) = sync_channel::<Arc<RgbaImage>>(2);
                let audio = crate::engine::audio::StreamingAudio::start(&path);

                thread::spawn(move || {
                    if let Err(e) = super::ffmpeg::video_decode_loop(&path, &tx) {
                        tracing::error!(target: "video", "video decoder error for '{}': {e}", path.display());
                    }
                });

                // Wait for the very first frame before returning so callers
                // can draw immediately without a blank flash.
                let first = rx
                    .recv()
                    .map_err(|_| anyhow!("video decoder exited before producing any frames"))?;

                Ok(FrameSource::Video {
                    current: first,
                    rx,
                    web_input: None,
                    audio,
                })
            }
        }
    }

    /// The channel to push mouse input into, for a web wallpaper — `None`
    /// for every other content type (including real video and CPU-scene
    /// fallback, which share `Video`'s shape but have no page to receive it).
    pub fn web_input(&self) -> Option<&SyncSender<super::web::WebInputEvent>> {
        match self {
            FrameSource::Video { web_input, .. } => web_input.as_ref(),
            FrameSource::Static(_) => None,
        }
    }

    /// The most recently decoded frame. `Arc::clone` by the caller is cheap
    /// (atomic refcount bump — no pixel data is copied).
    #[inline]
    pub fn current_frame(&self) -> &Arc<RgbaImage> {
        match self {
            FrameSource::Static(img) => img,
            FrameSource::Video { current, .. } => current,
        }
    }

    /// Advance to the next decoded frame if one is available.
    /// Returns `true` if the frame changed (caller should redraw).
    ///
    /// Takes exactly one frame per call so the 8 ms poll timer in the Wayland
    /// loop consumes frames at the same rate the PTS-paced decoder produces
    /// them.
    pub fn try_advance(&mut self) -> bool {
        match self {
            FrameSource::Static(_) => false,
            FrameSource::Video { current, rx, .. } => {
                if let Ok(frame) = rx.try_recv() {
                    *current = frame;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// `true` for animated sources that require a per-frame timer.
    #[inline]
    pub fn is_animated(&self) -> bool {
        matches!(self, FrameSource::Video { .. })
    }
}
