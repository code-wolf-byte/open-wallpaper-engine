use anyhow::{anyhow, Result};
use calloop::LoopSignal;
use calloop_wayland_source::WaylandSource;
use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_layer, delegate_output, delegate_pointer, delegate_registry,
    delegate_seat, delegate_shm,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    seat::{
        pointer::{PointerEvent, PointerEventKind, PointerHandler},
        Capability, SeatHandler, SeatState,
    },
    shell::wlr_layer::{
        Anchor, Layer, LayerShell, LayerShellHandler, LayerSurface, LayerSurfaceConfigure,
    },
    shell::WaylandSurface,
    shm::{slot::SlotPool, Shm, ShmHandler},
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::{mpsc::SyncSender, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use wayland_client::{
    backend::ObjectId,
    globals::registry_queue_init,
    protocol::{wl_output, wl_pointer, wl_seat, wl_shm, wl_surface},
    Connection, Dispatch, Proxy, QueueHandle,
};
// Manually dispatched (no smithay-client-toolkit delegate exists for this
// protocol) — playback-pause's fullscreen/maximized-app detection, see
// `engine::playback_gate` (Ghidra report Follow-up (ff)).
use wayland_protocols_wlr::foreign_toplevel::v1::client::{
    zwlr_foreign_toplevel_handle_v1::{self, ZwlrForeignToplevelHandleV1},
    zwlr_foreign_toplevel_manager_v1::{self, ZwlrForeignToplevelManagerV1},
};

use super::display::{DisplayPlatform, WallpaperHandle, WallpaperHandleInner};
use crate::engine::playback_gate::PlaybackAction;
use crate::{
    engine::gpu_renderer::GpuSceneInstance,
    platform,
    render::{FrameSource, RenderSettings, ScreenContent, ScreenSettings, WallpaperContent},
};

/// How often to re-check playback-pause conditions (battery, session lock,
/// fullscreen/maximized app) — see `engine::playback_gate`. Real I/O (sysfs
/// reads, a D-Bus call), so this stays well below frame rate rather than
/// adding that cost to every tick — matches `platform::x11`'s own interval.
const PAUSE_CHECK_INTERVAL: Duration = Duration::from_millis(750);

// ── Platform implementation ───────────────────────────────────────────────────

pub(super) struct WaylandPlatform;

impl DisplayPlatform for WaylandPlatform {
    fn spawn_wallpaper(
        &self,
        content: ScreenContent,
        settings: ScreenSettings,
    ) -> Result<WallpaperHandle> {
        let wayland_handle = spawn_wayland_wallpaper(content, settings)?;
        Ok(WallpaperHandle::new(Box::new(wayland_handle)))
    }
}

// ── Internal handle ───────────────────────────────────────────────────────────

struct WaylandHandle {
    stop_signal: LoopSignal,
    thread: thread::JoinHandle<()>,
}

impl WallpaperHandleInner for WaylandHandle {
    fn stop(self: Box<Self>) {
        self.stop_signal.stop();
        let _ = self.thread.join();
    }

    fn wait(self: Box<Self>) {
        let _ = self.thread.join();
    }
}

fn spawn_wayland_wallpaper(
    content: ScreenContent,
    settings: ScreenSettings,
) -> Result<WaylandHandle> {
    let (signal_tx, signal_rx) = std::sync::mpsc::sync_channel::<LoopSignal>(0);

    let thread = thread::spawn(move || {
        if let Err(e) = wallpaper_loop(content, settings, signal_tx) {
            tracing::error!(target: "wallpaper", "wallpaper thread error: {e}");
        }
    });

    let stop_signal = signal_rx
        .recv()
        .map_err(|_| anyhow!("wallpaper thread exited before sending the loop signal"))?;

    Ok(WaylandHandle {
        stop_signal,
        thread,
    })
}

// ── Content renderer ──────────────────────────────────────────────────────────

/// How the wallpaper content produces pixels.
enum ContentRenderer {
    /// CPU frames (static images, videos, and the scene fallback loop);
    /// presented through SHM buffers.
    Frames(FrameSource),
    /// GPU-rendered scene sharing our device — presented directly into the
    /// wgpu surface (no readback) when the compositor allows, with an RGBA
    /// readback + SHM fallback otherwise.
    Scene(Box<GpuSceneInstance>),
}

impl ContentRenderer {
    fn is_animated(&self) -> bool {
        match self {
            ContentRenderer::Frames(fs) => fs.is_animated(),
            ContentRenderer::Scene(_) => true,
        }
    }
}

/// Build the renderer for one output's resolved content. Scenes render on
/// our own device so frames can be presented directly; everything else
/// produces CPU frames. Split out of `wallpaper_loop` so `new_output` can
/// call it per-output instead of once for the whole process — the core of
/// per-screen wallpapers: each output gets its own independent
/// `GpuSceneInstance`/`FrameSource`, not a shared one.
fn build_content_renderer(
    content: WallpaperContent,
    device: wgpu::Device,
    queue: wgpu::Queue,
    allow_gpu_surface: bool,
    web: crate::render::web::WebOptions,
) -> Result<ContentRenderer> {
    Ok(match content {
        WallpaperContent::Scene { dir } if allow_gpu_surface => {
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
        other => ContentRenderer::Frames(FrameSource::from_content_with(other, web)?),
    })
}

/// A surface's renderer. Span and clone layouts give every surface the
/// same one, so it renders once per frame and each surface presents its
/// part.
struct Shared {
    content: ContentRenderer,
    /// When `content` last rendered a frame, and that frame read back for
    /// the SHM path, so other surfaces sharing it within the same frame
    /// reuse it.
    rendered_at: Option<Instant>,
    readback: Option<Arc<image::RgbaImage>>,
}

type SharedRenderer = Rc<RefCell<Shared>>;

fn shared(content: ContentRenderer) -> SharedRenderer {
    Rc::new(RefCell::new(Shared {
        content,
        rendered_at: None,
        readback: None,
    }))
}

impl Shared {
    /// Whether to render a new frame now: always for a renderer of its own;
    /// once per half frame interval for one shared by several surfaces.
    fn due(&mut self, sharers: usize, interval: Duration) -> bool {
        let due = sharers <= 1 || self.rendered_at.is_none_or(|t| t.elapsed() >= interval / 2);
        if due {
            self.rendered_at = Some(Instant::now());
        }
        due
    }
}

// ── Internal renderer state ───────────────────────────────────────────────────

/// GPU presentation state for one output surface.
struct SurfaceGpu {
    surface: wgpu::Surface<'static>,
    format: wgpu::TextureFormat,
    configured: (u32, u32),
}

struct WallpaperSurface {
    /// wgpu surface — declared before `layer` so it drops before the
    /// wl_surface it was created from.
    gpu: Option<SurfaceGpu>,
    /// `None` once GPU surface creation failed for this output (don't retry).
    gpu_failed: bool,
    layer: LayerSurface,
    width: u32,
    height: u32,
    /// Previous frame's SHM pool — kept alive until compositor releases the buffer.
    pool: Option<SlotPool>,
    /// This output's content — resolved from `ScreenContent` by output name
    /// at `new_output` time, so different outputs can show different
    /// wallpapers; one renderer shared by every surface in span and clone
    /// layouts.
    renderer: SharedRenderer,
    output: wl_output::WlOutput,
    /// The output's logical rectangle (x, y, w, h), for its share of a span.
    rect: Option<(i32, i32, i32, i32)>,
    /// This output's own independent `RenderSettings` — resolved from
    /// `ScreenSettings` the same way `renderer` is, so a heavy scene on one
    /// monitor can't force every other monitor's quality down with it.
    settings: Arc<Mutex<RenderSettings>>,
    /// The output's name, to rebuild `renderer` after a playback stop.
    output_name: Option<String>,
    /// When this surface last rendered, for the frame-rate cap, and whether
    /// a capped draw is already scheduled.
    last_draw: Option<Instant>,
    draw_scheduled: bool,
    /// Frame-rate cap last sent to a web page on this surface.
    web_fps: u32,
}

struct WallpaperState {
    registry_state: RegistryState,
    output_state: OutputState,
    seat_state: SeatState,
    compositor_state: CompositorState,
    shm: Shm,
    layer_shell: LayerShell,
    /// Pointer for the first seat that advertises one — drives camera parallax
    /// and `g_PointerPosition`.
    pointer: Option<wl_pointer::WlPointer>,
    surfaces: Vec<WallpaperSurface>,
    /// Per-output content to resolve against each new output's name —
    /// `new_output` builds that output's own `ContentRenderer` from this.
    content: ScreenContent,
    gpu_scaler: platform::GpuScaler,
    /// Per-output `RenderSettings` to resolve against each new output's
    /// name, the same way `content` is.
    settings: ScreenSettings,
    /// Queue handle stored so draw_at can request wl_surface_frame callbacks.
    qh: Option<QueueHandle<WallpaperState>>,
    // GPU presentation
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    /// Cloned alongside `device` — every output builds its own
    /// `GpuSceneInstance`/`FrameSource` from the same shared wgpu queue.
    queue: wgpu::Queue,
    display_ptr: *mut c_void,
    allow_gpu_surface: bool,
    // ── Playback-pause (engine::playback_gate, Ghidra report Follow-up (ff)) ──
    /// `None` when the compositor doesn't advertise `zwlr_layer_shell_v1`'s
    /// sibling `zwlr_foreign_toplevel_manager_v1` at all — GNOME/Mutter
    /// notably don't (it's a wlroots-ecosystem protocol: Sway/Hyprland/river
    /// support it). Fullscreen/maximized-app pause just never triggers
    /// there, same graceful-degradation precedent `platform::x11`'s EWMH
    /// check uses for a window manager with no support.
    ///
    /// Never read after construction — its only job is to outlive the
    /// connection so the subscription (and the `Toplevel` events it
    /// delivers into `foreign_toplevels`) stays alive; dropping it would
    /// tear the binding down.
    #[allow(dead_code)]
    foreign_toplevel_manager: Option<ZwlrForeignToplevelManagerV1>,
    /// Per-toplevel state from `zwlr_foreign_toplevel_handle_v1` events,
    /// keyed by the handle's own object id.
    foreign_toplevels: HashMap<ObjectId, ForeignToplevelState>,
    policy: crate::engine::playback_gate::PlaybackPolicy,
    /// MPRIS watcher for the "another application is playing audio"
    /// condition; only started when the policy acts on it.
    media: Option<crate::engine::media::MediaWatcher>,
    media_playing: bool,
    /// Renderers were released by a playback stop.
    stopped: bool,
    /// For scheduling frame-rate-capped draws.
    loop_handle: Option<calloop::LoopHandle<'static, WallpaperState>>,
    lock_watcher: Option<platform::power::LockWatcher>,
    playback_state: crate::engine::playback_gate::PlaybackState,
    last_pause_check: Instant,
    /// Last known pause state, so the watchdog timer (`tick_pause_watchdog`)
    /// can tell "just resumed" apart from "still paused"/"still active" —
    /// see that method's doc for why detecting the *edge* matters here.
    paused: bool,
}

/// One tracked toplevel's committed state (`maximized`/`activated`/
/// `fullscreen`) plus whatever a `state` event staged but hasn't been
/// finalized by a `done` event yet — the protocol's own "atomic even
/// across multiple events" contract (see the `state`/`done` event docs in
/// `wlr-foreign-toplevel-management-unstable-v1.xml`).
#[derive(Debug, Clone, Copy, Default)]
struct ForeignToplevelState {
    maximized: bool,
    activated: bool,
    fullscreen: bool,
    pending: Option<(bool, bool, bool)>,
}

impl WallpaperState {
    /// Try to create + configure a wgpu surface for output `idx`.
    /// On any failure the output falls back to the SHM path permanently.
    fn ensure_gpu_surface(&mut self, idx: usize) {
        if !self.allow_gpu_surface
            || self.surfaces[idx].gpu_failed
            || !matches!(
                self.surfaces[idx].renderer.borrow().content,
                ContentRenderer::Scene(_)
            )
        {
            return;
        }
        let (width, height) = (self.surfaces[idx].width, self.surfaces[idx].height);
        if width == 0 || height == 0 {
            return;
        }

        if self.surfaces[idx].gpu.is_none() {
            match self.create_gpu_surface(idx) {
                Ok(gpu) => self.surfaces[idx].gpu = Some(gpu),
                Err(e) => {
                    tracing::warn!(
                        target: "wallpaper",
                        "GPU surface unavailable for output {idx}: {e} — using SHM path"
                    );
                    self.surfaces[idx].gpu_failed = true;
                    return;
                }
            }
        }

        let gpu = self.surfaces[idx].gpu.as_mut().unwrap();
        if gpu.configured != (width, height) {
            gpu.surface.configure(
                &self.device,
                &wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: gpu.format,
                    width,
                    height,
                    present_mode: wgpu::PresentMode::Fifo,
                    desired_maximum_frame_latency: 2,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                },
            );
            gpu.configured = (width, height);
        }
    }

    fn create_gpu_surface(&self, idx: usize) -> Result<SurfaceGpu> {
        let display =
            NonNull::new(self.display_ptr).ok_or_else(|| anyhow!("null wl_display pointer"))?;
        let surface_ptr = self.surfaces[idx].layer.wl_surface().id().as_ptr() as *mut c_void;
        let surface_ptr =
            NonNull::new(surface_ptr).ok_or_else(|| anyhow!("null wl_surface pointer"))?;

        let raw_display = RawDisplayHandle::Wayland(WaylandDisplayHandle::new(display));
        let raw_window = RawWindowHandle::Wayland(WaylandWindowHandle::new(surface_ptr));

        // Safety: the wl_display lives for the whole wallpaper thread and the
        // wl_surface is owned by our LayerSurface, which outlives the wgpu
        // surface (field order in WallpaperSurface drops `gpu` first).
        let surface = unsafe {
            self.instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle: raw_display,
                    raw_window_handle: raw_window,
                })
        }
        .map_err(|e| anyhow!("create_surface failed: {e}"))?;

        if !self.adapter.is_surface_supported(&surface) {
            return Err(anyhow!(
                "adapter does not support presenting to this surface"
            ));
        }

        let caps = surface.get_capabilities(&self.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| {
                matches!(
                    f,
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
                )
            })
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| anyhow!("surface reports no supported formats"))?;

        Ok(SurfaceGpu {
            surface,
            format,
            configured: (0, 0),
        })
    }

    /// The canvas surface `idx` shows part of, and which part (normalized
    /// `[x, y, w, h]`): its share of all outputs' bounding box in a span,
    /// otherwise its whole self.
    fn view_of(&self, idx: usize) -> ((u32, u32), [f32; 4]) {
        let s = &self.surfaces[idx];
        let own = ((s.width, s.height), crate::render::settings::FULL_REGION);
        if self.content.layout != crate::render::ScreenLayout::Span {
            return own;
        }
        match s.rect {
            Some(rect) => span_region(rect, self.surfaces.iter().filter_map(|s| s.rect)),
            None => own,
        }
    }

    fn draw_at(&mut self, idx: usize) {
        if self.surfaces[idx].width == 0 || self.surfaces[idx].height == 0 {
            return;
        }
        // Playback policy (`engine::playback_gate`): when it halts
        // rendering, skip the draw and — just as importantly — don't
        // request another frame callback, so the surface holds its last
        // frame and the compositor stops nudging us every vblank.
        // `tick_pause_watchdog` notices the condition clearing and restarts
        // the callback chain.
        let action = self.playback_action(Instant::now());
        crate::engine::audio::set_muted(action >= PlaybackAction::Mute);
        if action.halts() {
            if !self.paused {
                self.notify_web(crate::render::web::WebInputEvent::Paused(true));
            }
            self.paused = true;
            return;
        }
        if self.paused {
            self.notify_web(crate::render::web::WebInputEvent::Paused(false));
        }
        self.paused = false;
        // A changed frame-rate cap reaches web pages too.
        let fps = self.surfaces[idx].settings.lock().unwrap().fps;
        if self.surfaces[idx].web_fps != fps {
            self.surfaces[idx].web_fps = fps;
            if let ContentRenderer::Frames(fs) = &self.surfaces[idx].renderer.borrow().content {
                if let Some(tx) = fs.web_input() {
                    let _ = tx.try_send(crate::render::web::WebInputEvent::Fps(fps));
                }
            }
        }
        crate::engine::audio::set_master_volume(self.settings.default.lock().unwrap().volume);

        // Frame-rate cap: a callback that arrives early waits on a timer.
        let interval = self.surfaces[idx].settings.lock().unwrap().frame_interval();
        if let Some(last) = self.surfaces[idx].last_draw {
            let since = last.elapsed();
            if since < interval {
                if !self.surfaces[idx].draw_scheduled {
                    if let Some(handle) = &self.loop_handle {
                        self.surfaces[idx].draw_scheduled = true;
                        let _ = handle.insert_source(
                            calloop::timer::Timer::from_duration(interval - since),
                            move |_, (), state: &mut WallpaperState| {
                                if let Some(s) = state.surfaces.get_mut(idx) {
                                    s.draw_scheduled = false;
                                    state.draw_at(idx);
                                }
                                calloop::timer::TimeoutAction::Drop
                            },
                        );
                    }
                }
                return;
            }
        }
        self.surfaces[idx].last_draw = Some(Instant::now());
        self.ensure_gpu_surface(idx);
        if self.surfaces[idx].gpu.is_some() {
            self.draw_gpu(idx);
        } else {
            self.draw_shm(idx);
        }
    }

    /// Direct GPU presentation: render the scene straight into the acquired
    /// surface texture — no CPU readback, no SHM copy.
    fn draw_gpu(&mut self, idx: usize) {
        // Request the next frame callback before present() commits the surface.
        if self.surfaces[idx].renderer.borrow().content.is_animated() {
            if let Some(qh) = &self.qh {
                let wl_surf = self.surfaces[idx].layer.wl_surface();
                wl_surf.frame(qh, wl_surf.clone());
            }
        }

        let acquired = self.surfaces[idx]
            .gpu
            .as_ref()
            .unwrap()
            .surface
            .get_current_texture();
        let frame = match acquired {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Outdated) | Err(wgpu::SurfaceError::Lost) => {
                self.surfaces[idx].gpu.as_mut().unwrap().configured = (0, 0);
                self.ensure_gpu_surface(idx);
                match self.surfaces[idx]
                    .gpu
                    .as_ref()
                    .unwrap()
                    .surface
                    .get_current_texture()
                {
                    Ok(f) => f,
                    Err(e) => {
                        tracing::error!(target: "wallpaper", "surface acquire failed after reconfigure: {e}");
                        return;
                    }
                }
            }
            Err(e) => {
                tracing::error!(target: "wallpaper", "surface acquire failed: {e}");
                return;
            }
        };

        let view = frame.texture.create_view(&Default::default());
        let format = self.surfaces[idx].gpu.as_ref().unwrap().format;
        let (canvas, region) = self.view_of(idx);
        let interval = self.surfaces[idx].settings.lock().unwrap().frame_interval();
        let sharers = Rc::strong_count(&self.surfaces[idx].renderer);
        let mut r = self.surfaces[idx].renderer.borrow_mut();
        let due = r.due(sharers, interval);
        match &mut r.content {
            ContentRenderer::Scene(instance) => {
                if due {
                    instance.render();
                }
                instance.present_region(&view, format, canvas, region);
            }
            // GPU surfaces are only created for scene renderers.
            ContentRenderer::Frames(_) => return,
        }
        drop(r);
        frame.present();
    }

    /// SHM path: CPU frame → GPU scaler → SHM buffer (static/video content
    /// and scene fallback when direct presentation is unavailable).
    fn draw_shm(&mut self, idx: usize) {
        let width = self.surfaces[idx].width;
        let height = self.surfaces[idx].height;

        let interval = self.surfaces[idx].settings.lock().unwrap().frame_interval();
        let sharers = Rc::strong_count(&self.surfaces[idx].renderer);
        let frame: Arc<image::RgbaImage> = {
            let mut r = self.surfaces[idx].renderer.borrow_mut();
            let due = r.due(sharers, interval) || r.readback.is_none();
            match &mut r.content {
                ContentRenderer::Frames(fs) => Arc::clone(fs.current_frame()),
                ContentRenderer::Scene(_) if !due => Arc::clone(r.readback.as_ref().unwrap()),
                ContentRenderer::Scene(instance) => match instance.render_rgba() {
                    Ok(img) => {
                        let img = Arc::new(img);
                        r.readback = Some(Arc::clone(&img));
                        img
                    }
                    Err(e) => {
                        tracing::error!(target: "wallpaper", "scene readback failed: {e}");
                        return;
                    }
                },
            }
        };
        let row_bytes = width as usize * 4;
        let stride = row_bytes;
        let active_len = stride * height as usize;
        let mut pool = match SlotPool::new(active_len, &self.shm) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(target: "wallpaper", "failed to create shm pool: {e}");
                return;
            }
        };

        let (buffer, canvas) = match pool.create_buffer(
            width as i32,
            height as i32,
            stride as i32,
            wl_shm::Format::Argb8888,
        ) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(target: "wallpaper", "failed to create buffer: {e}");
                return;
            }
        };

        let (quality, scaling) = {
            let s = self.surfaces[idx].settings.lock().unwrap();
            (s.quality, s.scaling)
        };
        // Scenes fill the screen, as on the direct GPU path; the fit modes
        // are for video, image and web content.
        let mode = match &self.surfaces[idx].renderer.borrow().content {
            ContentRenderer::Scene(_) => crate::render::ScaleMode::Fill,
            ContentRenderer::Frames(_) => scaling,
        };
        let (span, region) = self.view_of(idx);
        let pixels = self.gpu_scaler.scale_region(
            frame.as_ref(),
            width,
            height,
            quality,
            mode,
            span,
            region,
        );
        if pixels.len() != active_len {
            tracing::error!(
                target: "wallpaper",
                "scaler returned {} bytes for {}x{} frame, expected {}",
                pixels.len(),
                width,
                height,
                active_len
            );
            return;
        }

        copy_frame_into_shm_canvas(canvas, &pixels, width as usize, height as usize, stride);

        let wl_surf = self.surfaces[idx].layer.wl_surface();

        // Request a wl_surface_frame callback before committing — only for
        // animated sources. The callback fires after the compositor presents
        // this frame, at which point we advance to the next frame and draw again.
        // Static sources draw once on configure and never request more callbacks.
        if self.surfaces[idx].renderer.borrow().content.is_animated() {
            if let Some(qh) = &self.qh {
                wl_surf.frame(qh, wl_surf.clone());
            }
        }

        wl_surf.attach(Some(buffer.wl_buffer()), 0, 0);
        wl_surf.damage_buffer(0, 0, width as i32, height as i32);
        wl_surf.commit();

        // Keep the SHM pool alive until the compositor reads the buffer.
        self.surfaces[idx].pool = Some(pool);
    }
}

fn copy_frame_into_shm_canvas(
    canvas: &mut [u8],
    pixels: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) {
    let row_bytes = width * 4;
    let active_len = stride * height;
    let copy_len = active_len.min(canvas.len());

    canvas.fill(0);
    if copy_len < active_len || pixels.len() < row_bytes * height {
        return;
    }

    for row in 0..height {
        let src_start = row * row_bytes;
        let src_end = src_start + row_bytes;
        let dst_start = row * stride;
        let dst_end = dst_start + row_bytes;
        canvas[dst_start..dst_end].copy_from_slice(&pixels[src_start..src_end]);
    }
}

fn wallpaper_loop(
    content: ScreenContent,
    settings: ScreenSettings,
    signal_tx: SyncSender<LoopSignal>,
) -> Result<()> {
    // Open GPU device (prefer iGPU for background tasks; fall back to best).
    let gpu = platform::GpuDevice::open_low_power()
        .or_else(|_| platform::GpuDevice::open_best())
        .map_err(|e| anyhow!("no GPU device available: {e}"))?;
    let instance = gpu.instance.clone();
    let adapter = gpu.adapter.clone();
    let device = gpu.device.clone();
    let queue = gpu.queue.clone();
    let gpu_scaler = platform::GpuScaler::from_device(gpu)
        .map_err(|e| anyhow!("GPU scaler init failed: {e}"))?;

    let allow_gpu_surface = std::env::var("WP_ENGINE_FORCE_SHM").is_err();

    let conn = Connection::connect_to_env()
        .map_err(|e| anyhow!("cannot connect to Wayland display: {e}"))?;
    let display_ptr = conn.backend().display_ptr() as *mut c_void;

    let (globals, mut event_queue) = registry_queue_init::<WallpaperState>(&conn)
        .map_err(|e| anyhow!("Wayland registry init failed: {e}"))?;

    let qh = event_queue.handle();

    let compositor_state = CompositorState::bind(&globals, &qh)
        .map_err(|_| anyhow!("compositor does not advertise wl_compositor"))?;
    let shm =
        Shm::bind(&globals, &qh).map_err(|_| anyhow!("compositor does not advertise wl_shm"))?;
    let output_state = OutputState::new(&globals, &qh);
    let layer_shell = LayerShell::bind(&globals, &qh).map_err(|_| {
        anyhow!("compositor does not support zwlr_layer_shell_v1 (wlr-layer-shell)")
    })?;
    let registry_state = RegistryState::new(&globals);
    let seat_state = SeatState::new(&globals, &qh);
    // Playback-pause's fullscreen/maximized detection — see the
    // `foreign_toplevel_manager` field doc for the "not every compositor
    // supports this" graceful-degradation note.
    let foreign_toplevel_manager = globals
        .bind::<ZwlrForeignToplevelManagerV1, _, _>(&qh, 1..=3, ())
        .ok();

    let policy = crate::engine::playback_gate::PlaybackPolicy::load();
    let mut state = WallpaperState {
        registry_state,
        output_state,
        seat_state,
        compositor_state,
        shm,
        layer_shell,
        pointer: None,
        surfaces: Vec::new(),
        content,
        gpu_scaler,
        settings,
        qh: Some(qh.clone()),
        instance,
        adapter,
        device,
        queue,
        display_ptr,
        allow_gpu_surface,
        foreign_toplevel_manager,
        foreign_toplevels: HashMap::new(),
        policy,
        media: if policy.watches_audio() {
            crate::engine::media::MediaWatcher::start()
        } else {
            None
        },
        media_playing: false,
        stopped: false,
        loop_handle: None,
        lock_watcher: platform::power::LockWatcher::start(),
        playback_state: crate::engine::playback_gate::PlaybackState::default(),
        last_pause_check: Instant::now() - PAUSE_CHECK_INTERVAL,
        paused: false,
    };

    // First roundtrip: discovers all current outputs → triggers new_output → creates surfaces.
    event_queue
        .roundtrip(&mut state)
        .map_err(|e| anyhow!("initial Wayland roundtrip failed: {e}"))?;

    // Second roundtrip: compositor sends configure events for our layer surfaces.
    event_queue
        .roundtrip(&mut state)
        .map_err(|e| anyhow!("second Wayland roundtrip failed: {e}"))?;

    // Hand the queue to calloop.
    let mut event_loop: calloop::EventLoop<'static, WallpaperState> =
        calloop::EventLoop::try_new().map_err(|e| anyhow!("calloop init failed: {e}"))?;

    state.loop_handle = Some(event_loop.handle());
    WaylandSource::new(conn, event_queue)
        .insert(event_loop.handle())
        .map_err(|e| anyhow!("WaylandSource insert failed: {e}"))?;

    // Playback-pause watchdog (`engine::playback_gate`, Ghidra report
    // Follow-up (ff)): `draw_at` stops requesting new frame callbacks once
    // paused (see its own doc comment), so nothing would ever notice the
    // pause clearing without this independent timer.
    event_loop
        .handle()
        .insert_source(
            calloop::timer::Timer::from_duration(PAUSE_CHECK_INTERVAL),
            |_deadline, (), state: &mut WallpaperState| {
                state.tick_pause_watchdog();
                calloop::timer::TimeoutAction::ToDuration(PAUSE_CHECK_INTERVAL)
            },
        )
        .map_err(|e| anyhow!("pause-watchdog timer insert failed: {e}"))?;

    // Give the loop signal to the spawning thread before blocking.
    let _ = signal_tx.send(event_loop.get_signal());

    // Run until stop_signal.stop() is called from the UI thread.
    let run_result = event_loop.run(None, &mut state, |_| {});

    // Destroy wgpu surfaces BEFORE the event loop (and with it the Wayland
    // connection) is dropped — vkDestroySurfaceKHR against a closed
    // wl_display segfaults.
    for surface in &mut state.surfaces {
        surface.gpu = None;
    }

    run_result.map_err(|e| anyhow!("event loop error: {e}"))?;
    Ok(())
}

// ── SCTK handler implementations ──────────────────────────────────────────────

impl CompositorHandler for WallpaperState {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: i32,
    ) {
    }
    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wl_output::Transform,
    ) {
    }
    fn frame(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        _time: u32,
    ) {
        // Compositor has presented the previous frame and is ready for the next.
        // This mirrors linux-wallpaperengine's surfaceFrameCallback pattern.
        let idx = self
            .surfaces
            .iter()
            .position(|s| s.layer.wl_surface() == surface);
        if let Some(idx) = idx {
            self.qh = Some(qh.clone());
            if let ContentRenderer::Frames(fs) =
                &mut self.surfaces[idx].renderer.borrow_mut().content
            {
                fs.try_advance();
            }
            self.draw_at(idx);
        }
    }
}

impl OutputHandler for WallpaperState {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        // The xdg-output/wl_output name (e.g. "DP-1") — the same string
        // `wp-engine config set-screen` takes — resolved via SCTK's own
        // output-info aggregation, `None` when the compositor never reports
        // one (no per-screen assignment can ever match; falls back to the
        // default wallpaper below, same as always).
        let info = self.output_state.info(&output);
        let output_name = info.as_ref().and_then(|info| info.name.clone());
        // A web page renders at the output's native mode.
        let web = crate::render::web::WebOptions {
            size: info
                .as_ref()
                .and_then(|i| i.modes.iter().find(|m| m.current).map(|m| m.dimensions))
                .map(|(w, h)| (w.max(1) as u32, h.max(1) as u32))
                .unwrap_or((
                    crate::render::web::WEB_WIDTH,
                    crate::render::web::WEB_HEIGHT,
                )),
            fps: self
                .settings
                .resolve(output_name.as_deref())
                .lock()
                .unwrap()
                .fps,
        };

        let resolved = self.content.resolve(output_name.as_deref());
        let per_screen = output_name
            .as_deref()
            .is_some_and(|name| self.content.by_output.contains_key(name));
        // Span and clone: every output shows the first one's renderer.
        // ponytail: a spanned web page renders at the first output's size.
        let existing = (self.content.layout != crate::render::ScreenLayout::PerScreen)
            .then(|| self.surfaces.first().map(|s| Rc::clone(&s.renderer)))
            .flatten();
        let renderer = match existing.map(Ok).unwrap_or_else(|| {
            build_content_renderer(
                resolved,
                self.device.clone(),
                self.queue.clone(),
                self.allow_gpu_surface,
                web,
            )
            .map(shared)
        }) {
            Ok(r) => r,
            Err(e) => {
                // Don't take down every other (working) output over one
                // bad content path — log and leave this output blank.
                tracing::error!(
                    target: "wallpaper",
                    "failed to load wallpaper content for output {output_name:?}: {e} — leaving it blank"
                );
                return;
            }
        };
        tracing::info!(
            target: "wallpaper",
            "output {output_name:?}: {} wallpaper",
            match self.content.layout {
                crate::render::ScreenLayout::PerScreen if per_screen => "per-screen",
                crate::render::ScreenLayout::PerScreen => "default",
                l => l.label(),
            }
        );
        let rect = info.as_ref().and_then(output_rect);

        let wl_surface = self.compositor_state.create_surface(qh);
        let layer = self.layer_shell.create_layer_surface(
            qh,
            wl_surface,
            Layer::Background,
            Some("wp-engine"),
            Some(&output),
        );

        layer.set_anchor(Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
        layer.set_exclusive_zone(-1);
        layer.set_size(0, 0);
        layer.commit();

        let settings = self.settings.resolve(output_name.as_deref());

        self.surfaces.push(WallpaperSurface {
            gpu: None,
            gpu_failed: false,
            layer,
            width: 0,
            height: 0,
            pool: None,
            renderer,
            output,
            rect,
            settings,
            output_name,
            last_draw: None,
            draw_scheduled: false,
            web_fps: web.fps,
        });
    }

    fn update_output(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        let rect = self
            .output_state
            .info(&output)
            .as_ref()
            .and_then(output_rect);
        for s in self.surfaces.iter_mut().filter(|s| s.output == output) {
            s.rect = rect;
        }
    }

    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
}

impl LayerShellHandler for WallpaperState {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface) {}

    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let idx = self.surfaces.iter().position(|s| s.layer == *layer);
        if let Some(idx) = idx {
            self.surfaces[idx].width = configure.new_size.0;
            self.surfaces[idx].height = configure.new_size.1;
            self.draw_at(idx);
        }
    }
}

impl ShmHandler for WallpaperState {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl SeatHandler for WallpaperState {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }

    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}

    fn new_capability(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        // ponytail: one pointer, first seat that offers one. Multi-seat setups
        // would need a pointer per seat; nobody runs a wallpaper on two.
        if capability == Capability::Pointer && self.pointer.is_none() {
            match self.seat_state.get_pointer(qh, &seat) {
                Ok(pointer) => self.pointer = Some(pointer),
                Err(e) => {
                    tracing::warn!(target: "wallpaper", "cannot get wl_pointer ({e}); parallax stays centered")
                }
            }
        }
    }

    fn remove_capability(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Pointer {
            if let Some(pointer) = self.pointer.take() {
                pointer.release();
            }
        }
    }

    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

/// The bounding box of `all` output rectangles, and `rect`'s normalized
/// `[x, y, w, h]` within it.
fn span_region(
    (x, y, w, h): (i32, i32, i32, i32),
    all: impl Iterator<Item = (i32, i32, i32, i32)> + Clone,
) -> ((u32, u32), [f32; 4]) {
    let x0 = all.clone().map(|r| r.0).min().unwrap_or(x).min(x);
    let y0 = all.clone().map(|r| r.1).min().unwrap_or(y).min(y);
    let x1 = all
        .clone()
        .map(|r| r.0 + r.2)
        .max()
        .unwrap_or(x + w)
        .max(x + w);
    let y1 = all.map(|r| r.1 + r.3).max().unwrap_or(y + h).max(y + h);
    let (bw, bh) = ((x1 - x0).max(1), (y1 - y0).max(1));
    (
        (bw as u32, bh as u32),
        [
            (x - x0) as f32 / bw as f32,
            (y - y0) as f32 / bh as f32,
            w as f32 / bw as f32,
            h as f32 / bh as f32,
        ],
    )
}

/// An output's logical rectangle `(x, y, w, h)` in the compositor's global
/// space.
fn output_rect(info: &smithay_client_toolkit::output::OutputInfo) -> Option<(i32, i32, i32, i32)> {
    let (x, y) = info.logical_position?;
    let (w, h) = info.logical_size?;
    Some((x, y, w, h))
}

/// Linux evdev button codes (`linux/input-event-codes.h`) — what
/// `PointerEventKind::Press`/`Release`'s raw `button` field actually carries.
const BTN_LEFT: u32 = 0x110;
const BTN_RIGHT: u32 = 0x111;
const BTN_MIDDLE: u32 = 0x112;

impl PointerHandler for WallpaperState {
    fn pointer_frame(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for event in events {
            // Leave keeps the last position rather than snapping back to
            // centre — the cursor moving onto a window shouldn't yank the
            // parallax. Axis (scroll) isn't wired to anything — matches the
            // C++ reference's own `CWeb::updateMouse`, which forwards move
            // and left/right click only (its own `// TODO: ANY OTHER MOUSE
            // EVENTS TO SEND?` confirms that's the real engine's actual
            // scope, not an approximation here).
            let mut wheel: Option<(f32, f32)> = None;
            let (button, pressed) = match &event.kind {
                PointerEventKind::Enter { .. } | PointerEventKind::Motion { .. } => (None, false),
                PointerEventKind::Press { button, .. } => (Some(*button), true),
                PointerEventKind::Release { button, .. } => (Some(*button), false),
                PointerEventKind::Axis {
                    horizontal,
                    vertical,
                    ..
                } => {
                    wheel = Some((horizontal.absolute as f32, vertical.absolute as f32));
                    (None, false)
                }
                _ => continue,
            };
            let Some(idx) = self
                .surfaces
                .iter()
                .position(|s| s.layer.wl_surface() == &event.surface)
            else {
                continue;
            };
            let surface = &self.surfaces[idx];
            if surface.width == 0 || surface.height == 0 {
                continue;
            }

            // Into the shared canvas when this output shows part of a span.
            let n = pointer_norm(event.position, surface.width, surface.height);
            let (_, r) = self.view_of(idx);
            let norm = [r[0] + n[0] * r[2], r[1] + n[1] * r[3]];

            match &mut self.surfaces[idx].renderer.borrow_mut().content {
                // ponytail: only tracks while the cursor is over our own
                // layer surface. The reference additionally queries
                // Hyprland's IPC socket for a global cursor when another
                // window has it; add that (or the equivalent per-compositor
                // call) if parallax-under-windows matters.
                ContentRenderer::Scene(scene) => {
                    scene.set_mouse(norm);
                    // Background layer surfaces only see the pointer when
                    // nothing covers the desktop, so every press here is
                    // a press on the wallpaper.
                    if button == Some(BTN_LEFT) {
                        scene.set_mouse_button(pressed);
                    }
                }
                ContentRenderer::Frames(fs) => {
                    let Some(tx) = fs.web_input() else { continue };
                    let event = match (button, wheel) {
                        (None, Some((dx, dy))) => crate::render::web::WebInputEvent::Wheel {
                            x_norm: norm[0],
                            y_norm: norm[1],
                            dx,
                            dy,
                        },
                        (None, None) => crate::render::web::WebInputEvent::MouseMove {
                            x_norm: norm[0],
                            y_norm: norm[1],
                        },
                        (Some(code), _) => {
                            let Some(button) = web_mouse_button(code) else {
                                continue;
                            };
                            crate::render::web::WebInputEvent::MouseButton {
                                x_norm: norm[0],
                                y_norm: norm[1],
                                button,
                                pressed,
                            }
                        }
                    };
                    let _ = tx.try_send(event);
                }
            }
        }
    }
}

/// Linux evdev button code → the two buttons `WebInputEvent` (and the C++
/// reference) actually forward — `None` for anything else (middle click,
/// side buttons), silently dropped rather than guessed at.
fn web_mouse_button(code: u32) -> Option<crate::render::web::WebMouseButton> {
    match code {
        BTN_LEFT => Some(crate::render::web::WebMouseButton::Left),
        BTN_RIGHT => Some(crate::render::web::WebMouseButton::Right),
        BTN_MIDDLE => Some(crate::render::web::WebMouseButton::Middle),
        _ => None,
    }
}

/// Surface-local pointer coordinates → the `[0,1]²` pair the engine wants for
/// `g_PointerPosition` and camera parallax.
///
/// Top-origin, i.e. y=0 is the top edge. That matches the reference by way of a
/// double negative: `WaylandMouseInput::update` stores `size.y - localY`
/// (bottom-origin) and `CScene::updateMouse` then computes
/// `mouseY = 1.0 - normalizedMouseY`, putting it back. Only
/// `m_mousePositionNormalized` (particles, the scripting `input` object) stays
/// bottom-origin — `m_mousePosition`, which feeds both uniforms we care about
/// here, does not.
fn pointer_norm(position: (f64, f64), width: u32, height: u32) -> [f32; 2] {
    [
        (position.0 as f32 / width as f32).clamp(0.0, 1.0),
        (position.1 as f32 / height as f32).clamp(0.0, 1.0),
    ]
}

impl ProvidesRegistryState for WallpaperState {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers!(OutputState, SeatState);
}

/// Names of the outputs the compositor reports, for picking a screen to
/// assign.
pub fn output_names() -> Vec<String> {
    struct Outputs {
        registry: RegistryState,
        outputs: OutputState,
    }
    impl OutputHandler for Outputs {
        fn output_state(&mut self) -> &mut OutputState {
            &mut self.outputs
        }
        fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
        fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {
        }
        fn output_destroyed(
            &mut self,
            _: &Connection,
            _: &QueueHandle<Self>,
            _: wl_output::WlOutput,
        ) {
        }
    }
    impl ProvidesRegistryState for Outputs {
        fn registry(&mut self) -> &mut RegistryState {
            &mut self.registry
        }
        registry_handlers!(OutputState);
    }
    delegate_output!(Outputs);
    delegate_registry!(Outputs);

    let Ok(conn) = Connection::connect_to_env() else {
        return Vec::new();
    };
    let Ok((globals, mut queue)) = registry_queue_init::<Outputs>(&conn) else {
        return Vec::new();
    };
    let qh = queue.handle();
    let mut state = Outputs {
        registry: RegistryState::new(&globals),
        outputs: OutputState::new(&globals, &qh),
    };
    // Two round trips: one to bind the outputs, one for their info events.
    for _ in 0..2 {
        if queue.roundtrip(&mut state).is_err() {
            return Vec::new();
        }
    }
    state
        .outputs
        .outputs()
        .filter_map(|o| state.outputs.info(&o)?.name)
        .collect()
}

delegate_compositor!(WallpaperState);
delegate_output!(WallpaperState);
delegate_shm!(WallpaperState);
delegate_layer!(WallpaperState);
delegate_seat!(WallpaperState);
delegate_pointer!(WallpaperState);
delegate_registry!(WallpaperState);

// ── Playback-pause: zwlr_foreign_toplevel_management (manual dispatch — no
// smithay-client-toolkit delegate exists for this protocol) ───────────────────
// See `engine::playback_gate`'s module doc (Ghidra report Follow-up (ff)).

impl Dispatch<ZwlrForeignToplevelManagerV1, ()> for WallpaperState {
    fn event(
        state: &mut Self,
        _proxy: &ZwlrForeignToplevelManagerV1,
        event: zwlr_foreign_toplevel_manager_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        // `Finished` (the manager itself going away) needs no cleanup here —
        // each handle's own `Closed` event already removes its own entry.
        if let zwlr_foreign_toplevel_manager_v1::Event::Toplevel { toplevel } = event {
            state
                .foreign_toplevels
                .insert(toplevel.id(), ForeignToplevelState::default());
        }
    }
}

impl Dispatch<ZwlrForeignToplevelHandleV1, ()> for WallpaperState {
    fn event(
        state: &mut Self,
        proxy: &ZwlrForeignToplevelHandleV1,
        event: zwlr_foreign_toplevel_handle_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        let id = proxy.id();
        match event {
            // `state`'s `array` arg is a raw byte buffer of native-endian
            // u32 enum values (maximized=0, minimized=1, activated=2,
            // fullscreen=3 — `wlr-foreign-toplevel-management-unstable-v1
            // .xml`'s own `state` enum) — staged, not applied yet, until
            // `Done` commits it atomically (the protocol's own contract).
            zwlr_foreign_toplevel_handle_v1::Event::State { state: bits } => {
                let mut maximized = false;
                let mut activated = false;
                let mut fullscreen = false;
                for chunk in bits.chunks_exact(4) {
                    match u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) {
                        0 => maximized = true,
                        2 => activated = true,
                        3 => fullscreen = true,
                        _ => {}
                    }
                }
                if let Some(t) = state.foreign_toplevels.get_mut(&id) {
                    t.pending = Some((maximized, activated, fullscreen));
                }
            }
            zwlr_foreign_toplevel_handle_v1::Event::Done => {
                if let Some(t) = state.foreign_toplevels.get_mut(&id) {
                    if let Some((maximized, activated, fullscreen)) = t.pending.take() {
                        t.maximized = maximized;
                        t.activated = activated;
                        t.fullscreen = fullscreen;
                    }
                }
            }
            zwlr_foreign_toplevel_handle_v1::Event::Closed => {
                state.foreign_toplevels.remove(&id);
            }
            _ => {}
        }
    }
}

impl WallpaperState {
    /// `(fullscreen, maximized)` — any currently-*activated* toplevel
    /// reporting that state, mirroring `platform::x11`'s own "check the
    /// active window" heuristic (not "some window somewhere is fullscreen,"
    /// which could be a minimized/background one the user can't even see).
    fn foreign_toplevel_pause_state(&self) -> (bool, bool) {
        let fullscreen = self
            .foreign_toplevels
            .values()
            .any(|t| t.activated && t.fullscreen);
        let maximized = self
            .foreign_toplevels
            .values()
            .any(|t| t.activated && t.maximized);
        (fullscreen, maximized)
    }

    /// Re-checks the playback conditions when `PAUSE_CHECK_INTERVAL` has
    /// elapsed and returns the action the policy calls for, releasing or
    /// rebuilding the renderers on a stop edge.
    fn playback_action(&mut self, now: Instant) -> PlaybackAction {
        if now.duration_since(self.last_pause_check) >= PAUSE_CHECK_INTERVAL {
            self.last_pause_check = now;
            let (fullscreen_app, maximized_app) = self.foreign_toplevel_pause_state();
            while let Some(info) = self.media.as_ref().and_then(|m| m.try_recv()) {
                self.media_playing =
                    info.playback_state == crate::engine::media::PlaybackState::Playing;
            }
            self.playback_state = crate::engine::playback_gate::PlaybackState {
                on_battery: platform::power::is_on_battery(),
                locked_or_sleeping: self
                    .lock_watcher
                    .as_mut()
                    .map(|w| w.is_locked())
                    .unwrap_or(false),
                fullscreen_app,
                maximized_app,
                focused_app: self.foreign_toplevels.values().any(|t| t.activated),
                other_audio: self.media_playing,
            };
        }
        let action = crate::engine::playback_gate::decide(&self.policy, &self.playback_state);
        if action == PlaybackAction::Stop && !self.stopped {
            self.stopped = true;
            for s in &mut self.surfaces {
                let mut r = s.renderer.borrow_mut();
                let still = match &r.content {
                    ContentRenderer::Frames(fs) => Arc::clone(fs.current_frame()),
                    ContentRenderer::Scene(_) => Arc::new(image::RgbaImage::new(1, 1)),
                };
                r.content = ContentRenderer::Frames(FrameSource::Static(still));
                r.readback = None;
                drop(r);
                s.gpu = None;
            }
            tracing::info!(target: "wallpaper", "playback stopped; renderers released");
        } else if action != PlaybackAction::Stop && self.stopped {
            self.stopped = false;
            let one_renderer = self.content.layout != crate::render::ScreenLayout::PerScreen;
            for i in 0..self.surfaces.len() {
                self.surfaces[i].gpu_failed = false;
                // Span and clone surfaces share surface 0's renderer.
                if one_renderer && i > 0 {
                    continue;
                }
                let resolved = self
                    .content
                    .resolve(self.surfaces[i].output_name.as_deref());
                let web = crate::render::web::WebOptions {
                    size: (
                        self.surfaces[i].width.max(1),
                        self.surfaces[i].height.max(1),
                    ),
                    fps: self.surfaces[i].settings.lock().unwrap().fps,
                };
                match build_content_renderer(
                    resolved,
                    self.device.clone(),
                    self.queue.clone(),
                    self.allow_gpu_surface,
                    web,
                ) {
                    Ok(r) => self.surfaces[i].renderer.borrow_mut().content = r,
                    Err(e) => tracing::error!(target: "wallpaper", "resuming output {i}: {e}"),
                }
            }
            tracing::info!(target: "wallpaper", "playback resumed");
        }
        action
    }

    /// Send `event` to every output showing a web wallpaper.
    fn notify_web(&self, event: crate::render::web::WebInputEvent) {
        for s in &self.surfaces {
            if let ContentRenderer::Frames(fs) = &s.renderer.borrow().content {
                if let Some(tx) = fs.web_input() {
                    let _ = tx.try_send(event);
                }
            }
        }
    }

    /// Runs on its own `PAUSE_CHECK_INTERVAL` timer (inserted once at
    /// startup, independent of any surface's frame callbacks) — the only
    /// thing that can notice a halt has cleared once `draw_at` has stopped
    /// requesting new callbacks.
    fn tick_pause_watchdog(&mut self) {
        let was_paused = self.paused;
        let action = self.playback_action(Instant::now());
        crate::engine::audio::set_muted(action >= PlaybackAction::Mute);
        if !was_paused && action.halts() {
            self.notify_web(crate::render::web::WebInputEvent::Paused(true));
        }
        if was_paused && !action.halts() {
            self.notify_web(crate::render::web::WebInputEvent::Paused(false));
            self.paused = false;
            for idx in 0..self.surfaces.len() {
                if self.surfaces[idx].renderer.borrow().content.is_animated() {
                    self.draw_at(idx);
                }
            }
        } else {
            self.paused = action.halts();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{pointer_norm, span_region, web_mouse_button, BTN_LEFT, BTN_MIDDLE, BTN_RIGHT};

    #[test]
    fn span_regions_split_the_bounding_box() {
        // 1080p beside a 1440p monitor that sits 360 px higher.
        let left = (0, 360, 1920, 1080);
        let right = (1920, 0, 2560, 1440);
        let all = [left, right];
        let (canvas, r) = span_region(left, all.iter().copied());
        assert_eq!(canvas, (4480, 1440));
        assert_eq!(r, [0.0, 0.25, 1920.0 / 4480.0, 0.75]);
        let (_, r) = span_region(right, all.iter().copied());
        assert_eq!(r, [1920.0 / 4480.0, 0.0, 2560.0 / 4480.0, 1.0]);
    }

    #[test]
    fn pointer_norm_is_top_origin_and_clamped() {
        // Top-left of the surface maps to (0,0), bottom-right to (1,1).
        // If someone "fixes" the Y axis to bottom-origin, this flips and
        // every parallax wallpaper drifts the wrong way vertically.
        assert_eq!(pointer_norm((0.0, 0.0), 1920, 1080), [0.0, 0.0]);
        assert_eq!(pointer_norm((1920.0, 1080.0), 1920, 1080), [1.0, 1.0]);
        assert_eq!(pointer_norm((960.0, 540.0), 1920, 1080), [0.5, 0.5]);

        // A quarter of the way down is 0.25, not 0.75.
        assert_eq!(pointer_norm((0.0, 270.0), 1920, 1080), [0.0, 0.25]);

        // Compositors can report positions outside the surface during drags.
        assert_eq!(pointer_norm((-40.0, 5000.0), 1920, 1080), [0.0, 1.0]);
    }

    /// Only left/right forward to CEF — matches the C++ reference's own
    /// `CWeb::updateMouse` scope exactly (see `WebInputEvent`'s doc comment).
    /// Middle click and anything else (side buttons) must map to `None`
    /// rather than a guessed-at mapping.
    #[test]
    fn web_mouse_button_maps_left_middle_and_right() {
        use crate::render::web::WebMouseButton;
        assert_eq!(web_mouse_button(BTN_LEFT), Some(WebMouseButton::Left));
        assert_eq!(web_mouse_button(BTN_RIGHT), Some(WebMouseButton::Right));
        assert_eq!(web_mouse_button(BTN_MIDDLE), Some(WebMouseButton::Middle));
        assert_eq!(web_mouse_button(0), None);
    }
}
