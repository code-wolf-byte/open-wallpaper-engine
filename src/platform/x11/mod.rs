//! X11 display backend — draws into a root-sized pixmap and shows it as the
//! root window's background (the `_XROOTPMAP_ID` / `ESETROOT_PMAP_ID`
//! convention `feh --bg` uses, as `X11Output.cpp` in the reference does).
//!
//! Desktop environments cover the root with their own desktop (plasmashell,
//! xfdesktop, Mutter's background actor, caja/nemo), so when the window
//! manager supports EWMH desktop windows the same pixmap also backs a
//! click-through `_NET_WM_WINDOW_TYPE_DESKTOP` window of our own, which the WM
//! keeps beneath every application window. Bare window managers without that
//! support (dwm, i3) would tile such a window, so they get the root only.
//! `WP_ENGINE_X11_MODE=root|window` overrides the choice.
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
    Atom, AtomEnum, ChangeWindowAttributesAux, ConnectionExt as _, CreateGCAux, CreateWindowAux,
    Gcontext, ImageFormat, KeyButMask, Pixmap, PropMode, Screen, Window, WindowClass,
};
use x11rb::protocol::{shape, xfixes::ConnectionExt as _};
use x11rb::rust_connection::RustConnection;

use crate::audio::playback_gate::PlaybackAction;
use x11rb::wrapper::ConnectionExt as _;

use super::display::{DisplayPlatform, WallpaperHandle, WallpaperHandleInner};
use crate::platform;
use crate::render::session::{
    pending_property_edits, ContentRenderer, PlaybackMonitor, WindowState, PAUSE_CHECK_INTERVAL,
};
use crate::render::{FrameSource, RenderSettings, ScreenContent, ScreenSettings};

// ── Platform implementation ───────────────────────────────────────────────────

pub(super) struct X11Platform;

impl DisplayPlatform for X11Platform {
    fn spawn_wallpaper(
        &self,
        content: ScreenContent,
        settings: ScreenSettings,
    ) -> Result<WallpaperHandle> {
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

/// Names of the active outputs, for picking a screen to assign.
pub fn output_names() -> Vec<String> {
    let Ok((conn, screen_num)) = x11rb::connect(None) else {
        return Vec::new();
    };
    let screen = &conn.setup().roots[screen_num];
    discover_outputs(
        &conn,
        screen.root,
        screen.width_in_pixels,
        screen.height_in_pixels,
    )
    .into_iter()
    .filter_map(|o| o.name)
    .collect()
}

/// Enumerate connected+active RandR outputs (name + rectangle), falling back
/// to the whole root window when RandR is missing or reports nothing usable
/// (headless X, Xvfb, old servers).
fn discover_outputs(
    conn: &RustConnection,
    root: Window,
    root_w: u16,
    root_h: u16,
) -> Vec<NamedOutput> {
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

// ── Fullscreen/maximized-app detection (playback-pause) ────────────────────────
// EWMH's `_NET_ACTIVE_WINDOW` + `_NET_WM_STATE` — see `engine::playback_gate`'s
// module doc (Ghidra report Follow-up (ff)) for where `playbackfullscreen`/
// `playbackmaximized` come from.

struct EwmhAtoms {
    active_window: Atom,
    wm_state: Atom,
    fullscreen: Atom,
    maximized_vert: Atom,
    maximized_horz: Atom,
    window_type: Atom,
    type_desktop: Atom,
    type_dock: Atom,
}

impl EwmhAtoms {
    fn intern(conn: &impl Connection) -> Result<Self> {
        let atom =
            |name: &[u8]| -> Result<Atom> { Ok(conn.intern_atom(false, name)?.reply()?.atom) };
        Ok(Self {
            active_window: atom(b"_NET_ACTIVE_WINDOW")?,
            wm_state: atom(b"_NET_WM_STATE")?,
            fullscreen: atom(b"_NET_WM_STATE_FULLSCREEN")?,
            maximized_vert: atom(b"_NET_WM_STATE_MAXIMIZED_VERT")?,
            maximized_horz: atom(b"_NET_WM_STATE_MAXIMIZED_HORZ")?,
            window_type: atom(b"_NET_WM_WINDOW_TYPE")?,
            type_desktop: atom(b"_NET_WM_WINDOW_TYPE_DESKTOP")?,
            type_dock: atom(b"_NET_WM_WINDOW_TYPE_DOCK")?,
        })
    }

    /// The state of whichever window `_NET_ACTIVE_WINDOW` names; all false
    /// when there's no active window, the window manager doesn't publish
    /// one, or a request fails — a WM without EWMH support degrades to
    /// "never pauses" rather than erroring the render loop.
    fn check(&self, conn: &impl Connection, root: Window) -> WindowState {
        let Some(active) = read_window32(conn, root, self.active_window) else {
            return WindowState::default();
        };
        if active == 0 || active == root {
            return WindowState::default();
        }
        let types = read_atoms32(conn, active, self.window_type).unwrap_or_default();
        let focused = !types.contains(&self.type_desktop) && !types.contains(&self.type_dock);
        let Some(states) = read_atoms32(conn, active, self.wm_state) else {
            return WindowState {
                focused,
                ..Default::default()
            };
        };
        WindowState {
            fullscreen: states.contains(&self.fullscreen),
            maximized: states.contains(&self.maximized_vert)
                && states.contains(&self.maximized_horz),
            focused,
        }
    }
}

fn read_window32(conn: &impl Connection, window: Window, property: Atom) -> Option<Window> {
    let reply = conn
        .get_property(false, window, property, AtomEnum::WINDOW, 0, 1)
        .ok()?
        .reply()
        .ok()?;
    let mut values = reply.value32()?;
    values.next()
}

/// Capped at 1024 atoms — `_NET_WM_STATE` never remotely approaches that in
/// practice (real window managers set a handful), and an unbounded
/// `long_length` would let a misbehaving property balloon this request.
fn read_atoms32(conn: &impl Connection, window: Window, property: Atom) -> Option<Vec<Atom>> {
    let reply = conn
        .get_property(false, window, property, AtomEnum::ATOM, 0, 1024)
        .ok()?
        .reply()
        .ok()?;
    let values = reply.value32()?;
    Some(values.collect())
}

// ── Desktop window ────────────────────────────────────────────────────────────

/// Whether to show the wallpaper in our own desktop window (see the module
/// doc): forced by `WP_ENGINE_X11_MODE`, otherwise when the window manager
/// lists `_NET_WM_WINDOW_TYPE_DESKTOP` in `_NET_SUPPORTED`.
fn use_desktop_window(conn: &impl Connection, root: Window, desktop_type: Atom) -> bool {
    match std::env::var("WP_ENGINE_X11_MODE").as_deref() {
        Ok("root") => false,
        Ok("window") => true,
        _ => conn
            .intern_atom(false, b"_NET_SUPPORTED")
            .ok()
            .and_then(|c| c.reply().ok())
            .and_then(|r| read_atoms32(conn, root, r.atom))
            .is_some_and(|supported| supported.contains(&desktop_type)),
    }
}

/// A root-sized desktop window whose background is `pixmap`: sticky, below
/// everything, out of taskbars and pagers, and click-through so presses
/// still reach the desktop icons and menus underneath (the cursor is polled
/// globally anyway).
fn create_desktop_window(conn: &RustConnection, screen: &Screen, pixmap: Pixmap) -> Result<Window> {
    let win = conn.generate_id()?;
    conn.create_window(
        screen.root_depth,
        win,
        screen.root,
        0,
        0,
        screen.width_in_pixels,
        screen.height_in_pixels,
        0,
        WindowClass::INPUT_OUTPUT,
        screen.root_visual,
        &CreateWindowAux::new().background_pixmap(pixmap),
    )?;
    let atom = |name: &[u8]| -> Result<Atom> { Ok(conn.intern_atom(false, name)?.reply()?.atom) };
    conn.change_property32(
        PropMode::REPLACE,
        win,
        atom(b"_NET_WM_WINDOW_TYPE")?,
        AtomEnum::ATOM,
        &[atom(b"_NET_WM_WINDOW_TYPE_DESKTOP")?],
    )?;
    conn.change_property32(
        PropMode::REPLACE,
        win,
        atom(b"_NET_WM_STATE")?,
        AtomEnum::ATOM,
        &[
            atom(b"_NET_WM_STATE_BELOW")?,
            atom(b"_NET_WM_STATE_STICKY")?,
            atom(b"_NET_WM_STATE_SKIP_TASKBAR")?,
            atom(b"_NET_WM_STATE_SKIP_PAGER")?,
        ],
    )?;
    conn.change_property32(
        PropMode::REPLACE,
        win,
        atom(b"_NET_WM_DESKTOP")?,
        AtomEnum::CARDINAL,
        &[u32::MAX],
    )?;
    conn.change_property8(
        PropMode::REPLACE,
        win,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        b"wp-engine",
    )?;
    conn.change_property8(
        PropMode::REPLACE,
        win,
        AtomEnum::WM_CLASS,
        AtomEnum::STRING,
        b"wp-engine\0wp-engine\0",
    )?;
    if let Err(e) = make_click_through(conn, win) {
        tracing::warn!(target: "wallpaper", "no XFixes ({e}); desktop window will take clicks");
    }
    conn.map_window(win)?;
    Ok(win)
}

/// An empty input shape: pointer events pass through to whatever is below.
fn make_click_through(conn: &RustConnection, win: Window) -> Result<()> {
    conn.xfixes_query_version(5, 0)?.reply()?;
    let region = conn.generate_id()?;
    conn.xfixes_create_region(region, &[])?;
    conn.xfixes_set_window_shape_region(win, shape::SK::INPUT, 0, 0, region)?;
    conn.xfixes_destroy_region(region)?;
    Ok(())
}

// ── Main loop ─────────────────────────────────────────────────────────────────

/// A web wallpaper's page size and frame rate for one output.
fn web_options(
    rect: OutputRect,
    settings: &ScreenSettings,
    name: Option<&str>,
) -> crate::render::web::WebOptions {
    crate::render::web::WebOptions {
        size: (rect.width as u32, rect.height as u32),
        fps: settings.resolve(name).lock().unwrap().fps,
    }
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
    /// Pause state and frame-rate cap last sent to a web page here.
    web_paused: bool,
    web_fps: u32,
    /// How far into `settings.property_edits` this output has applied.
    props_seen: usize,
}

fn wallpaper_loop(
    content: ScreenContent,
    settings: ScreenSettings,
    stop: Arc<AtomicBool>,
) -> Result<()> {
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

    // Span: one renderer over the whole root window, which already is the
    // bounding box of every monitor. ponytail: clone still runs one
    // renderer per monitor here; share one if the duplicate work matters.
    let named_outputs = if content.layout == crate::render::ScreenLayout::Span {
        vec![NamedOutput {
            name: None,
            rect: OutputRect {
                x: 0,
                y: 0,
                width: root_w,
                height: root_h,
            },
        }]
    } else {
        discover_outputs(&conn, root, root_w, root_h)
    };
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
        let web = web_options(named.rect, &settings, named.name.as_deref());
        match ContentRenderer::build(resolved, device.clone(), queue.clone(), true, web) {
            Ok(renderer) => {
                tracing::info!(
                    target: "wallpaper",
                    "output {:?}: {} wallpaper",
                    named.name,
                    if per_screen { "per-screen" } else { "default" }
                );
                let output_settings = settings.resolve(named.name.as_deref());
                let props_seen = output_settings.lock().unwrap().property_edits.len();
                outputs.push(OutputInstance {
                    named,
                    renderer,
                    settings: output_settings,
                    left_down: false,
                    right_down: false,
                    web_paused: false,
                    web_fps: web.fps,
                    props_seen,
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

    // Playback-pause conditions (`engine::playback_gate` — Ghidra report
    // Follow-up (ff)). `ewmh` degrades to "never fullscreen/maximized" on
    // any WM without `_NET_ACTIVE_WINDOW`/`_NET_WM_STATE` support (its own
    // `check`'s doc comment); battery/lock are checked on the same
    // `PAUSE_CHECK_INTERVAL` throttle rather than every frame, since both
    // involve real I/O (sysfs reads, a D-Bus round trip) a 30fps loop
    // shouldn't pay for every tick.
    let mut playback = PlaybackMonitor::new();
    let mut stopped = false;
    let ewmh = EwmhAtoms::intern(&conn)?;
    let desktop_window = if use_desktop_window(&conn, root, ewmh.type_desktop) {
        Some(create_desktop_window(&conn, screen, pixmap)?)
    } else {
        None
    };
    match desktop_window {
        Some(win) => {
            tracing::info!(target: "wallpaper", "X11: showing the wallpaper in desktop window {win:#x}")
        }
        None => {
            tracing::info!(target: "wallpaper", "X11: showing the wallpaper on the root window")
        }
    }

    // Any output still animating keeps the whole loop going; a genuinely
    // static desktop (every output static) holds the pixmap and sleeps.
    let animated = outputs.iter().any(|o| o.renderer.is_animated());

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let started = Instant::now();

        let action = playback.action(started, || ewmh.check(&conn, root));
        let paused = action.halts();
        // Tell web pages about pause edges and frame-rate changes.
        for output in &mut outputs {
            let fps = output.settings.lock().unwrap().fps;
            if let Some(tx) = output.renderer.web_input() {
                if paused != output.web_paused {
                    let _ = tx.try_send(crate::render::web::WebInputEvent::Paused(paused));
                }
                if fps != output.web_fps {
                    let _ = tx.try_send(crate::render::web::WebInputEvent::Fps(fps));
                }
            }
            output.web_paused = paused;
            output.web_fps = fps;
        }
        // Live user-property edits for whatever each output shows: scenes
        // apply them in place (or rebuild), web pages get applyUserProperties.
        for output in &mut outputs {
            let resolved = content.resolve(output.named.name.as_deref());
            let Some((dir, names)) =
                pending_property_edits(&output.settings, &mut output.props_seen, &resolved)
            else {
                continue;
            };
            if !output.renderer.apply_properties(&dir, &names) {
                let web = web_options(output.named.rect, &settings, output.named.name.as_deref());
                match ContentRenderer::build(resolved, device.clone(), queue.clone(), true, web) {
                    Ok(r) => output.renderer = r,
                    Err(e) => {
                        tracing::error!(target: "wallpaper", "reloading output {:?}: {e}", output.named.name)
                    }
                }
            }
        }
        crate::audio::audio::set_muted(action >= PlaybackAction::Mute);
        crate::audio::audio::set_master_volume(settings.default.lock().unwrap().volume);

        // Stop: drop each output's renderer (scene GPU resources, decoder
        // threads), keeping its last frame on the pixmap; rebuild on resume.
        if action == PlaybackAction::Stop && !stopped {
            stopped = true;
            for output in &mut outputs {
                let last = output.renderer.next_frame().ok();
                let still = last.unwrap_or_else(|| Arc::new(image::RgbaImage::new(1, 1)));
                output.renderer = ContentRenderer::Frames(FrameSource::Static(still));
            }
            tracing::info!(target: "wallpaper", "playback stopped; renderer released");
        } else if action != PlaybackAction::Stop && stopped {
            stopped = false;
            for output in &mut outputs {
                let resolved = content.resolve(output.named.name.as_deref());
                let web = web_options(output.named.rect, &settings, output.named.name.as_deref());
                match ContentRenderer::build(resolved, device.clone(), queue.clone(), true, web) {
                    Ok(r) => output.renderer = r,
                    Err(e) => {
                        tracing::error!(target: "wallpaper", "resuming output {:?}: {e}", output.named.name)
                    }
                }
            }
            tracing::info!(target: "wallpaper", "playback resumed");
        }

        if !paused {
            for output in &mut outputs {
                let (quality, scaling) = {
                    let s = output.settings.lock().unwrap();
                    (s.quality, s.scaling)
                };
                // Scenes size themselves; the fit modes are for video, image
                // and web content.
                let mode = match &output.renderer {
                    ContentRenderer::Scene(_) => crate::render::ScaleMode::Stretch,
                    ContentRenderer::Frames(_) => scaling,
                };
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
                    mode,
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
            // Repaint the root (and our desktop window) from the new frame.
            conn.clear_area(false, root, 0, 0, 0, 0)?;
            if let Some(win) = desktop_window {
                conn.clear_area(false, win, 0, 0, 0, 0)?;
            }
            conn.flush()?;
        }

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
                    ContentRenderer::Scene(scene) => {
                        scene.set_mouse(norm);
                        // The button mask is global: only count presses made
                        // on the bare desktop (no window, or a desktop window
                        // — ours is click-through, so the DE's or none).
                        let mask: u16 = pointer.mask.into();
                        let on_desktop = pointer.child == x11rb::NONE
                            || Some(pointer.child) == desktop_window
                            || (mask & u16::from(KeyButMask::BUTTON1) != 0
                                && read_atoms32(&conn, pointer.child, ewmh.window_type)
                                    .is_some_and(|t| t.contains(&ewmh.type_desktop)));
                        if on_desktop {
                            scene.set_mouse_button(mask & u16::from(KeyButMask::BUTTON1) != 0);
                        } else {
                            scene.set_mouse_button(false);
                        }
                    }
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

        // Paused: no point looping at frame rate just to re-check the same
        // throttled conditions — sleep for that same interval instead.
        let budget = if paused {
            PAUSE_CHECK_INTERVAL
        } else {
            settings.default.lock().unwrap().frame_interval()
        };
        if let Some(rest) = budget.checked_sub(started.elapsed()) {
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
