// macOS display backend for wp-engine.
//
// Gated on by `platform/mod.rs`'s `#[cfg(target_os = "macos")] pub mod macos;`,
// so nothing in this file needs its own cfg.
//
// Status: step 2 of the port — a scene renders into a winit window via a wgpu
// Metal surface, no CPU readback (`GpuSceneInstance::render_to_view`, the same
// path the Wayland backend presents through). Still a plain foreground window;
// desktop-level wallpaper placement (NSWindow level + collectionBehavior via
// objc2-app-kit), multi-monitor, and non-scene content are next — see PROGRESS.md.
//
// Threading: AppKit requires the event loop on the process MAIN thread, and
// winit enforces it (there is no `with_any_thread` on macOS). So unlike the
// Wayland backend — which runs its loop on a spawned thread and lets the main
// thread park in `WallpaperApplication::show()` — the macOS wallpaper must run
// on the main thread via `run_wallpaper_on_main`. `MacOSPlatform::spawn_wallpaper`
// therefore can't fulfil the "spawn + return handle" contract and errors,
// pointing callers at the main-thread entry (the app-lifecycle wiring that calls
// it is the next step; see DECISIONS.md / PROGRESS.md).
//
// wgpu selects Metal automatically; WGSL is compiled to MSL by naga inside wgpu.

use std::process::Child;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use objc2_app_kit::{NSView, NSWindowCollectionBehavior};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use super::display::{DisplayPlatform, WallpaperHandle, WallpaperHandleInner};
use super::GpuDevice;
use crate::engine::gpu_renderer::GpuSceneInstance;
use crate::render::{ScreenContent, ScreenSettings, WallpaperContent};

// ── Platform implementation ───────────────────────────────────────────────────

pub struct MacOSPlatform;

impl DisplayPlatform for MacOSPlatform {
    fn spawn_wallpaper(&self, content: ScreenContent, _settings: ScreenSettings) -> Result<WallpaperHandle> {
        // AppKit/winit demand the event loop on the process main thread, and the
        // GUI (eframe) already owns this process's main thread — two winit event
        // loops can't share it. So we can't run the wallpaper in-process here.
        // Instead launch it as a child process (`wp-engine set-file <dir>`) that
        // owns its own main thread and runs `run_wallpaper_on_main`. The returned
        // handle stops the wallpaper by killing that child.
        //
        // Per-screen assignment (`content.by_output`) isn't honored yet: this
        // backend doesn't enumerate NSScreens or place a window per display at
        // all yet (see the file doc comment above — "step 2 of the port"), so
        // there's nowhere to route a second wallpaper to. Always runs
        // `content.default`, same as before this was even a `ScreenContent`.
        let dir = match &content.default {
            WallpaperContent::Scene { dir } => dir.clone(),
            _ => bail!("macOS backend currently renders only scene wallpapers"),
        };
        let exe = std::env::current_exe()
            .map_err(|e| anyhow!("cannot locate wp-engine executable: {e}"))?;
        let child = std::process::Command::new(exe)
            .arg("set-file")
            .arg(&dir)
            .spawn()
            .map_err(|e| anyhow!("failed to launch wallpaper process: {e}"))?;
        Ok(WallpaperHandle::new(Box::new(ChildWallpaperHandle { child })))
    }
}

/// Handle to a wallpaper running in a child `wp-engine set-file` process.
/// `stop` kills it; `wait` blocks until it exits on its own.
struct ChildWallpaperHandle {
    child: Child,
}

impl WallpaperHandleInner for ChildWallpaperHandle {
    fn stop(mut self: Box<Self>) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
    fn wait(mut self: Box<Self>) {
        let _ = self.child.wait();
    }
}

/// Run a scene wallpaper on the current thread until the window is closed.
///
/// **Must be called on the process main thread** — AppKit/winit require it.
/// Blocks for the lifetime of the wallpaper.
pub fn run_wallpaper_on_main(content: WallpaperContent) -> Result<()> {
    // Prefer the low-power GPU for a background task; fall back to best.
    let gpu = GpuDevice::open_low_power()
        .or_else(|_| GpuDevice::open_best())
        .map_err(|e| anyhow!("no GPU device available: {e}"))?;

    // Only scene wallpapers render on the GPU surface for now.
    let scene = match content {
        WallpaperContent::Scene { dir } => {
            GpuSceneInstance::with_device(gpu.device.clone(), gpu.queue.clone(), &dir)
                .map_err(|e| anyhow!("scene init failed: {e}"))?
        }
        _ => bail!("macOS backend currently renders only scene wallpapers"),
    };

    let event_loop = EventLoop::new().map_err(|e| anyhow!("winit event loop init failed: {e}"))?;

    let mut app = MacOSApp {
        instance: gpu.instance,
        adapter: gpu.adapter,
        device: gpu.device,
        scene: Box::new(scene),
        format: wgpu::TextureFormat::Bgra8Unorm,
        configured: (0, 0),
        surface: None,
        window: None,
    };

    event_loop
        .run_app(&mut app)
        .map_err(|e| anyhow!("winit event loop error: {e}"))?;

    // Drop the wgpu surface before the window it was created from.
    app.surface = None;
    Ok(())
}

// ── winit application ─────────────────────────────────────────────────────────

struct MacOSApp {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    scene: Box<GpuSceneInstance>,
    format: wgpu::TextureFormat,
    configured: (u32, u32),
    // `surface` is declared before `window` so it drops first — the wgpu
    // surface must not outlive the window it was created from.
    surface: Option<wgpu::Surface<'static>>,
    window: Option<Arc<Window>>,
}

impl MacOSApp {
    fn reconfigure(&mut self, width: u32, height: u32) {
        if let Some(surface) = &self.surface {
            surface.configure(
                &self.device,
                &wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: self.format,
                    width,
                    height,
                    present_mode: wgpu::PresentMode::Fifo,
                    desired_maximum_frame_latency: 2,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                },
            );
        }
        self.configured = (width, height);
    }

    fn draw(&mut self) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }
        if self.configured != (size.width, size.height) {
            self.reconfigure(size.width, size.height);
        }

        let acquired = match &self.surface {
            Some(s) => s.get_current_texture(),
            None => return,
        };
        let frame = match acquired {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.reconfigure(size.width, size.height);
                match self
                    .surface
                    .as_ref()
                    .and_then(|s| s.get_current_texture().ok())
                {
                    Some(f) => f,
                    None => return,
                }
            }
            Err(e) => {
                tracing::error!(target: "wallpaper", "surface acquire failed: {e}");
                return;
            }
        };

        let view = frame.texture.create_view(&Default::default());
        self.scene
            .render_to_view(&view, size.width, size.height, self.format);
        frame.present();

        // Fifo present() paces to vsync, so a redraw request per frame gives a
        // steady animation loop rather than a busy spin.
        window.request_redraw();
    }
}

impl ApplicationHandler for MacOSApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        // Cover the primary monitor: borderless, positioned/sized to the whole
        // screen so the scene fills the desktop. Fall back to a windowed size if
        // no monitor is reported (headless/edge cases).
        let mut attrs = Window::default_attributes()
            .with_title("wp-engine")
            .with_decorations(false);
        attrs = match event_loop.primary_monitor() {
            Some(m) => attrs.with_position(m.position()).with_inner_size(m.size()),
            None => attrs.with_inner_size(winit::dpi::LogicalSize::new(960.0, 540.0)),
        };
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                tracing::error!(target: "wallpaper", "create_window failed: {e}");
                event_loop.exit();
                return;
            }
        };
        // Sink it to the desktop-wallpaper layer (behind apps + icons, all
        // Spaces, click-through). Main thread — `resumed` always is.
        pin_to_desktop(&window);
        let surface = match self.instance.create_surface(window.clone()) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(target: "wallpaper", "create_surface failed: {e}");
                event_loop.exit();
                return;
            }
        };

        let caps = surface.get_capabilities(&self.adapter);
        self.format = caps
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
            .unwrap_or(wgpu::TextureFormat::Bgra8Unorm);

        let size = window.inner_size();
        self.surface = Some(surface);
        self.window = Some(window.clone());
        self.reconfigure(size.width.max(1), size.height.max(1));
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    self.reconfigure(size.width, size.height);
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.draw(),
            _ => {}
        }
    }
}

/// Sink the winit window down to the desktop-wallpaper layer: behind app
/// windows and desktop icons, visible on every Space, and click-through so the
/// desktop stays usable. Best-effort — logs and returns if the AppKit handle
/// isn't there. Must run on the main thread (AppKit requirement; `resumed` is).
fn pin_to_desktop(window: &Window) {
    // kCGDesktopWindowLevel: Finder's desktop layer, below the desktop-icon
    // layer, so icons stay on top and app windows cover us. ponytail: this is
    // the numeric value of CGWindowLevelForKey(kCGDesktopWindowLevelKey), stable
    // across macOS versions; link CoreGraphics for the real call if it ever drifts.
    const DESKTOP_WINDOW_LEVEL: isize = -2_147_483_623;

    let raw = match window.window_handle() {
        Ok(h) => h.as_raw(),
        Err(e) => {
            tracing::warn!(target: "wallpaper", "no window handle; not pinning to desktop: {e}");
            return;
        }
    };
    let RawWindowHandle::AppKit(handle) = raw else {
        tracing::warn!(target: "wallpaper", "non-AppKit window handle; not pinning to desktop");
        return;
    };
    // SAFETY: on AppKit the handle points at the live NSView owned by `window`,
    // which outlives this call; we only touch it on the main thread.
    let view: &NSView = unsafe { &*handle.ns_view.as_ptr().cast::<NSView>() };
    let Some(ns_window) = view.window() else {
        tracing::warn!(target: "wallpaper", "NSView has no NSWindow; not pinning to desktop");
        return;
    };
    unsafe {
        ns_window.setLevel(DESKTOP_WINDOW_LEVEL);
        ns_window.setCollectionBehavior(
            NSWindowCollectionBehavior::CanJoinAllSpaces | NSWindowCollectionBehavior::Stationary,
        );
        ns_window.setIgnoresMouseEvents(true);
    }
    tracing::info!(target: "wallpaper", "window pinned to desktop level");
}
