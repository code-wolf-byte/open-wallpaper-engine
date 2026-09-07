//! X11 display backend — draws into the root window's background pixmap.
//!
//! This mirrors `X11Output.cpp` in the reference rather than creating a
//! `_NET_WM_WINDOW_TYPE_DESKTOP` window: every WM and DE honours the root
//! pixmap plus the `_XROOTPMAP_ID` / `ESETROOT_PMAP_ID` convention (it is what
//! `feh --bg` and `hsetroot` set), whereas a desktop-type window competes with
//! whatever desktop surface GNOME/KDE/Xfce already draw and loses differently
//! on each.
//!
//! Unlike the Wayland backend there is no direct-presentation path: X11 has no
//! equivalent of handing the compositor a GPU surface for the desktop
//! background, so every frame is read back to RGBA and pushed with `PutImage`.
//! That readback is the same one `draw_shm` already performs, so scenes cost
//! roughly what they cost on the Wayland SHM fallback.

use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use x11rb::connection::{Connection, RequestConnection as _};
use x11rb::protocol::randr::ConnectionExt as _;
use x11rb::protocol::xproto::{
    AtomEnum, ChangeWindowAttributesAux, ConnectionExt as _, CreateGCAux, Gcontext, ImageFormat,
    KeyButMask, Pixmap, PropMode, Window,
};
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;

use super::display::{DisplayPlatform, WallpaperHandle, WallpaperHandleInner};
use crate::engine::gpu_renderer::GpuSceneInstance;
use crate::platform;
use crate::render::{FrameSource, RenderSettings, ScreenContent, ScreenSettings, WallpaperContent};

/// Frames per second for animated content. Matches `FrameSource`'s own target.
const TARGET_FPS: f32 = 30.0;

// ── Platform implementation ───────────────────────────────────────────────────

pub(super) struct X11Platform;

impl DisplayPlatform for X11Platform {
    fn spawn_wallpaper(&self, content: ScreenContent, settings: ScreenSettings) -> Result<WallpaperHandle> {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = Arc::clone(&stop);

        let thread = thread::spawn(move || {
            if let Err(e) = wallpaper_loop(content, settings, stop_thread) {
                tracing::error!(target: "wallpaper", "X11 wallpaper thread error: {e}");
            }
        });

        Ok(WallpaperHandle::new(Box::new(X11Handle { stop, thread })))
    }
}

struct X11Handle {
    stop: Arc<AtomicBool>,
    thread: thread::JoinHandle<()>,
}

impl WallpaperHandleInner for X11Handle {
    fn stop(self: Box<Self>) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.thread.join();
    }

    fn wait(self: Box<Self>) {
        let _ = self.thread.join();
    }
}

// ── Outputs ───────────────────────────────────────────────────────────────────

/// One monitor's rectangle in root-window coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutputRect {
    pub x: i16,
    pub y: i16,
    pub width: u16,
    pub height: u16,
}

/// One monitor: its rectangle plus its RandR output name (e.g. `"HDMI-1"`)
/// when available — the same string `wp-engine config set-screen` takes.
/// `name` is `None` on the whole-root-window fallback (no RandR, or RandR
/// present but nothing named/active decoded) — a per-screen assignment can
/// never match it, so it always renders the default wallpaper.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NamedOutput {
    pub name: Option<String>,
    pub rect: OutputRect,
}

/// Enumerate connected+active RandR outputs (name + rectangle), falling back
/// to the whole root window when RandR is missing or reports nothing usable
/// (headless X, Xvfb, old servers).
fn discover_outputs(conn: &RustConnection, root: Window, root_w: u16, root_h: u16) -> Vec<NamedOutput> {
    let whole = vec![NamedOutput {
        name: None,
        rect: OutputRect {
            x: 0,
            y: 0,
            width: root_w,
            height: root_h,
        },
    }];

    let Ok(cookie) = conn.randr_get_screen_resources_current(root) else {
        return whole;
    };
    let Ok(resources) = cookie.reply() else {
        return whole;
    };

    // CRTC id -> rectangle, for CRTCs that actually have a mode set.
    let crtc_rects: std::collections::HashMap<u32, OutputRect> = resources
        .crtcs
        .iter()
        .filter_map(|&crtc| {
            let info = conn
                .randr_get_crtc_info(crtc, resources.config_timestamp)
                .ok()?
                .reply()
                .ok()?;
            // A CRTC with no mode is disconnected/disabled.
            (info.width > 0 && info.height > 0).then_some((
                crtc,
                OutputRect {
                    x: info.x,
                    y: info.y,
                    width: info.width,
                    height: info.height,
                },
            ))
        })
        .collect();

    // Walk RandR *output* objects (not CRTCs) to recover names, then join
    // each back to its driving CRTC's rectangle. An output with no CRTC (or
    // one that turned out inactive above) is disconnected/off — skipped.
    let mut outputs: Vec<NamedOutput> = resources
        .outputs
        .iter()
        .filter_map(|&output| {
            let info = conn
                .randr_get_output_info(output, resources.config_timestamp)
                .ok()?
                .reply()
                .ok()?;
            let rect = crtc_rects.get(&info.crtc)?;
            Some(NamedOutput {
                name: Some(String::from_utf8_lossy(&info.name).into_owned()),
                rect: *rect,
            })
        })
        .collect();

    // Mirrored outputs report identical rectangles under different names —
    // legitimately different `NamedOutput`s (per-screen assignment keys on
    // the name), so no dedup here unlike the geometry-only path below.
    if !outputs.is_empty() {
        return outputs;
    }

    // RandR present but no *named* output resolved (e.g. output objects
    // unsupported by this server) — fall back to bare CRTC geometry, unnamed.
    outputs = crtc_rects
        .values()
        .map(|&rect| NamedOutput { name: None, rect })
        .collect();
    if outputs.is_empty() {
        whole
    } else {
        outputs
    }
}

// ── Content ───────────────────────────────────────────────────────────────────

/// How the wallpaper produces pixels. Both arms end in a CPU RGBA frame —
/// the root pixmap has no GPU-surface equivalent.
enum ContentRenderer {
    Frames(FrameSource),
    Scene(Box<GpuSceneInstance>),
}

impl ContentRenderer {
    fn is_animated(&self) -> bool {
        match self {
            ContentRenderer::Frames(fs) => fs.is_animated(),
            ContentRenderer::Scene(_) => true,
        }
    }

    fn next_frame(&mut self) -> Result<Arc<image::RgbaImage>> {
        match self {
            ContentRenderer::Frames(fs) => {
                fs.try_advance();
                Ok(Arc::clone(fs.current_frame()))
            }
            ContentRenderer::Scene(instance) => Ok(Arc::new(instance.render_rgba()?)),
        }
    }
}

// ── PutImage ──────────────────────────────────────────────────────────────────

/// Bytes of fixed overhead in a `PutImage` request (24-byte header, plus slack
/// for the length field's 4-byte padding).
const PUT_IMAGE_OVERHEAD: usize = 64;

/// Push `pixels` (BGRA, `rect.width * rect.height * 4` bytes) into `pixmap` at
/// `rect`'s offset, split into horizontal bands that each fit in one request.
///
/// A single 4K frame is ~33 MB, far past the ~16 MB ceiling even with
/// BIG-REQUESTS, so chunking is required rather than defensive.
fn put_image_chunked(
    conn: &RustConnection,
    pixmap: Pixmap,
    gc: Gcontext,
    depth: u8,
    rect: OutputRect,
    pixels: &[u8],
) -> Result<()> {
    let row_bytes = rect.width as usize * 4;
    if row_bytes == 0 {
        return Ok(());
    }

    let budget = conn
        .maximum_request_bytes()
        .saturating_sub(PUT_IMAGE_OVERHEAD);
    let rows_per_chunk = (budget / row_bytes).max(1).min(rect.height as usize);

    for band_start in (0..rect.height as usize).step_by(rows_per_chunk) {
        let band_rows = rows_per_chunk.min(rect.height as usize - band_start);
        let start = band_start * row_bytes;
        let end = start + band_rows * row_bytes;
        let Some(band) = pixels.get(start..end) else {
            return Err(anyhow!(
                "frame buffer is {} bytes, short of the {} needed for {}x{}",
                pixels.len(),
                rect.height as usize * row_bytes,
                rect.width,
                rect.height
            ));
        };

        conn.put_image(
            ImageFormat::Z_PIXMAP,
            pixmap,
            gc,
            rect.width,
            band_rows as u16,
            rect.x,
            rect.y + band_start as i16,
            0,
            depth,
            band,
        )?;
    }
    Ok(())
}

// ── Main loop ─────────────────────────────────────────────────────────────────

/// Build the renderer for one output's resolved content — the X11 twin of
/// the Wayland backend's `build_content_renderer`. Split out so each output
/// gets its own independent `GpuSceneInstance`/`FrameSource` instead of one
/// shared across the whole root pixmap.
fn build_content_renderer(content: WallpaperContent, device: wgpu::Device, queue: wgpu::Queue) -> Result<ContentRenderer> {
    Ok(match content {
        WallpaperContent::Scene { dir } => {
            match GpuSceneInstance::with_device(device, queue, &dir) {
                Ok(instance) => ContentRenderer::Scene(Box::new(instance)),
                Err(e) => {
                    tracing::warn!(target: "wallpaper", "GPU scene init failed ({e}); using frame-loop fallback");
                    ContentRenderer::Frames(FrameSource::from_content(WallpaperContent::Scene {
                        dir,
                    })?)
                }
            }
        }
        other => ContentRenderer::Frames(FrameSource::from_content(other)?),
    })
}

/// One monitor's independent render state: its own content, own `ContentRenderer`,
/// and its own click-edge tracking (X11 has no press/release *event* here — see
/// `left_down`'s doc comment below — so each output needs its own last-known state,
/// not one shared pair that would cross-talk between two different web wallpapers).
struct OutputInstance {
    named: NamedOutput,
    renderer: ContentRenderer,
    /// This output's own independent `RenderSettings` — resolved from
    /// `ScreenSettings` the same way `renderer` is resolved from
    /// `ScreenContent`, so a heavy scene on one monitor can't force every
    /// other monitor's quality down with it.
    settings: Arc<Mutex<RenderSettings>>,
    left_down: bool,
    right_down: bool,
}

fn wallpaper_loop(content: ScreenContent, settings: ScreenSettings, stop: Arc<AtomicBool>) -> Result<()> {
    let (conn, screen_num) =
        x11rb::connect(None).map_err(|e| anyhow!("cannot connect to X display: {e}"))?;
    let screen = &conn.setup().roots[screen_num];
    let root = screen.root;
    let depth = screen.root_depth;
    let (root_w, root_h) = (screen.width_in_pixels, screen.height_in_pixels);

    // `GpuScaler::from_device` consumes the device, so clone the parts the
    // scene renderer needs first (same dance as the Wayland backend).
    let gpu = platform::GpuDevice::open_low_power()
        .or_else(|_| platform::GpuDevice::open_best())
        .map_err(|e| anyhow!("no GPU device available: {e}"))?;
    let device = gpu.device.clone();
    let queue = gpu.queue.clone();
    let gpu_scaler = platform::GpuScaler::from_device(gpu)
        .map_err(|e| anyhow!("GPU scaler init failed: {e}"))?;

    let named_outputs = discover_outputs(&conn, root, root_w, root_h);
    tracing::info!(
        target: "wallpaper",
        "X11 root pixmap {root_w}x{root_h}, depth {depth}, {} output(s)",
        named_outputs.len()
    );

    let mut outputs: Vec<OutputInstance> = Vec::with_capacity(named_outputs.len());
    for named in named_outputs {
        let per_screen = named
            .name
            .as_deref()
            .is_some_and(|n| content.by_output.contains_key(n));
        let resolved = content.resolve(named.name.as_deref());
        match build_content_renderer(resolved, device.clone(), queue.clone()) {
            Ok(renderer) => {
                tracing::info!(
                    target: "wallpaper",
                    "output {:?}: {} wallpaper",
                    named.name,
                    if per_screen { "per-screen" } else { "default" }
                );
                let output_settings = settings.resolve(named.name.as_deref());
                outputs.push(OutputInstance {
                    named,
                    renderer,
                    settings: output_settings,
                    left_down: false,
                    right_down: false,
                });
            }
            // Don't take down every other (working) output over one bad
            // content path — log and leave this output out of the pixmap
            // entirely (it keeps whatever the desktop already had there).
            Err(e) => tracing::error!(
                target: "wallpaper",
                "failed to load wallpaper content for output {:?}: {e} — leaving it blank",
                named.name
            ),
        }
    }
    if outputs.is_empty() {
        return Err(anyhow!("no output's wallpaper content loaded successfully"));
    }

    // The pixmap stays owned by this connection: when the process exits the
    // server frees it and the previous desktop background comes back.
    let pixmap = conn.generate_id()?;
    conn.create_pixmap(depth, pixmap, root, root_w, root_h)?;
    let gc = conn.generate_id()?;
    conn.create_gc(gc, pixmap, &CreateGCAux::new())?;

    let prop_root = conn.intern_atom(false, b"_XROOTPMAP_ID")?.reply()?.atom;
    let prop_esetroot = conn.intern_atom(false, b"ESETROOT_PMAP_ID")?.reply()?.atom;

    let frame_budget = Duration::from_secs_f32(1.0 / TARGET_FPS);
    // Any output still animating keeps the whole loop going; a genuinely
    // static desktop (every output static) holds the pixmap and sleeps.
    let animated = outputs.iter().any(|o| o.renderer.is_animated());

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let started = Instant::now();

        for output in &mut outputs {
            let quality = output.settings.lock().unwrap().quality;
            let frame = output.renderer.next_frame()?;
            // `GpuScaler` emits ARGB8888-LE, i.e. bytes [B, G, R, A] — already
            // the byte order a Z_PIXMAP wants on a little-endian server, and
            // the same order the reference gets from its GL_BGRA readback. Do
            // not "fix" this into an RGBA swap.
            let pixels = gpu_scaler.scale(
                frame.as_ref(),
                output.named.rect.width as u32,
                output.named.rect.height as u32,
                quality,
            );
            put_image_chunked(&conn, pixmap, gc, depth, output.named.rect, &pixels)?;
        }

        // Publish the pixmap. Compositors (picom et al.) watch these atoms and
        // will otherwise paint over the background themselves.
        conn.change_property32(
            PropMode::REPLACE,
            root,
            prop_root,
            AtomEnum::PIXMAP,
            &[pixmap],
        )?;
        conn.change_property32(
            PropMode::REPLACE,
            root,
            prop_esetroot,
            AtomEnum::PIXMAP,
            &[pixmap],
        )?;
        conn.change_window_attributes(
            root,
            &ChangeWindowAttributesAux::new().background_pixmap(pixmap),
        )?;
        // Repaint the root from its new background.
        conn.clear_area(false, root, 0, 0, 0, 0)?;
        conn.flush()?;

        if !animated {
            // Every output is static: hold the pixmap until asked to stop.
            while !stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(100));
            }
            break;
        }

        // Global cursor position drives parallax. X11 hands us this regardless
        // of which window has focus, so unlike the Wayland backend there is no
        // "cursor left our surface" blind spot. With multiple outputs, forward
        // it only to whichever output's rect actually contains the cursor —
        // each output's own last-known left/right state (`OutputInstance`)
        // keeps two different web wallpapers on two different screens from
        // cross-talking through a would-be shared click-edge flag.
        if let Ok(pointer) = conn.query_pointer(root)?.reply() {
            for output in &mut outputs {
                let rect = output.named.rect;
                let (px, py) = (pointer.root_x as i32, pointer.root_y as i32);
                let inside = px >= rect.x as i32
                    && py >= rect.y as i32
                    && px < rect.x as i32 + rect.width as i32
                    && py < rect.y as i32 + rect.height as i32;
                if !inside {
                    continue;
                }
                let norm = [
                    ((px - rect.x as i32) as f32 / rect.width.max(1) as f32).clamp(0.0, 1.0),
                    ((py - rect.y as i32) as f32 / rect.height.max(1) as f32).clamp(0.0, 1.0),
                ];
                match &mut output.renderer {
                    ContentRenderer::Scene(scene) => scene.set_mouse(norm),
                    ContentRenderer::Frames(fs) => {
                        // Polled, not event-driven (see `OutputInstance`'s own
                        // doc comment) — mirrors the C++ reference's own
                        // per-frame `CWeb::updateMouse` poll, not just
                        // Wayland's event model adapted here.
                        let Some(tx) = fs.web_input() else { continue };
                        let _ = tx.try_send(crate::render::web::WebInputEvent::MouseMove {
                            x_norm: norm[0],
                            y_norm: norm[1],
                        });

                        let mask: u16 = pointer.mask.into();
                        let left_now = mask & u16::from(KeyButMask::BUTTON1) != 0;
                        let right_now = mask & u16::from(KeyButMask::BUTTON3) != 0;
                        if left_now != output.left_down {
                            output.left_down = left_now;
                            let _ = tx.try_send(crate::render::web::WebInputEvent::MouseButton {
                                x_norm: norm[0],
                                y_norm: norm[1],
                                button: crate::render::web::WebMouseButton::Left,
                                pressed: left_now,
                            });
                        }
                        if right_now != output.right_down {
                            output.right_down = right_now;
                            let _ = tx.try_send(crate::render::web::WebInputEvent::MouseButton {
                                x_norm: norm[0],
                                y_norm: norm[1],
                                button: crate::render::web::WebMouseButton::Right,
                                pressed: right_now,
                            });
                        }
                    }
                }
            }
        }

        if let Some(rest) = frame_budget.checked_sub(started.elapsed()) {
            thread::sleep(rest);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bands must tile the image exactly: no gaps, no overlap, no lost rows.
    /// Getting this wrong shows up as horizontal stripes of stale pixels.
    #[test]
    fn chunk_bands_tile_the_image_exactly() {
        for (height, rows_per_chunk) in [(1080usize, 7usize), (4u32 as usize, 4), (1000, 999)] {
            let mut covered = vec![0u8; height];
            for band_start in (0..height).step_by(rows_per_chunk) {
                let band_rows = rows_per_chunk.min(height - band_start);
                assert!(band_rows > 0);
                for row in band_start..band_start + band_rows {
                    covered[row] += 1;
                }
            }
            assert!(
                covered.iter().all(|&c| c == 1),
                "height {height} in chunks of {rows_per_chunk} did not tile exactly"
            );
        }
    }

    /// Round-trip a known colour through a real X server to prove the byte
    /// order `GpuScaler` emits is what `Z_PIXMAP` expects. Skipped when no
    /// display is available (CI, headless builds).
    #[test]
    fn put_image_round_trips_bgra_through_the_server() {
        if std::env::var_os("DISPLAY").is_none() {
            eprintln!("skipping: no DISPLAY");
            return;
        }
        let Ok((conn, screen_num)) = x11rb::connect(None) else {
            eprintln!("skipping: cannot connect to X display");
            return;
        };
        let screen = &conn.setup().roots[screen_num];
        let (root, depth) = (screen.root, screen.root_depth);
        let rect = OutputRect {
            x: 0,
            y: 0,
            width: 8,
            height: 8,
        };

        let pixmap = conn.generate_id().unwrap();
        conn.create_pixmap(depth, pixmap, root, rect.width, rect.height)
            .unwrap();
        let gc = conn.generate_id().unwrap();
        conn.create_gc(gc, pixmap, &CreateGCAux::new()).unwrap();

        // Opaque red, in the [B, G, R, A] order scaler.rs documents emitting.
        let pixels: Vec<u8> = std::iter::repeat([0u8, 0, 255, 255])
            .take(rect.width as usize * rect.height as usize)
            .flatten()
            .collect();
        put_image_chunked(&conn, pixmap, gc, depth, rect, &pixels).unwrap();
        conn.flush().unwrap();

        let got = conn
            .get_image(
                ImageFormat::Z_PIXMAP,
                pixmap,
                0,
                0,
                rect.width,
                rect.height,
                !0,
            )
            .unwrap()
            .reply()
            .unwrap();

        // Red must come back in the third byte. If it lands in the first, the
        // server wants RGBA here and the scaler output needs swapping.
        assert_eq!(
            (got.data[0], got.data[1], got.data[2]),
            (0, 0, 255),
            "byte order mismatch: got {:?}, expected red in byte 2 (BGRA)",
            &got.data[..4]
        );

        conn.free_gc(gc).unwrap();
        conn.free_pixmap(pixmap).unwrap();
        conn.flush().unwrap();
    }

    #[test]
    fn outputs_dedup_mirrored_crtcs() {
        let mut rects = vec![
            OutputRect {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080,
            },
            OutputRect {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080,
            },
            OutputRect {
                x: 1920,
                y: 0,
                width: 2560,
                height: 1440,
            },
        ];
        rects.dedup();
        assert_eq!(rects.len(), 2);
    }
}
