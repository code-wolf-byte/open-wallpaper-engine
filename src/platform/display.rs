use anyhow::Result;

use crate::render::ScreenContent;
use crate::render::ScreenSettings;

// ── Private inner trait ───────────────────────────────────────────────────────

pub(crate) trait WallpaperHandleInner: Send {
    fn stop(self: Box<Self>);
    fn wait(self: Box<Self>);
}

// ── Public abstract handle ────────────────────────────────────────────────────

/// Platform-agnostic handle to a running wallpaper renderer.
///
/// Drop or call `stop()` to remove the wallpaper.
pub struct WallpaperHandle {
    inner: Box<dyn WallpaperHandleInner>,
}

impl WallpaperHandle {
    pub(crate) fn new(inner: Box<dyn WallpaperHandleInner>) -> Self {
        Self { inner }
    }

    /// Stop the renderer and wait for its thread to exit.
    pub fn stop(self) {
        self.inner.stop();
    }

    /// Block the calling thread until the renderer exits (e.g. on Ctrl-C).
    pub fn wait(self) {
        self.inner.wait();
    }
}

// ── Platform trait ────────────────────────────────────────────────────────────

pub trait DisplayPlatform {
    /// Spawn a wallpaper renderer, one independent content instance per
    /// discovered output — `content.resolve(output_name)` decides which.
    ///
    /// The platform decides how to render: scene wallpapers draw directly
    /// into GPU surfaces when the compositor allows it; other content (and
    /// fallback paths) go through CPU frames + SHM buffers.
    fn spawn_wallpaper(&self, content: ScreenContent, settings: ScreenSettings) -> Result<WallpaperHandle>;
}

// ── Runtime detection ─────────────────────────────────────────────────────────

/// Detect the current display platform at runtime and return a boxed implementation.
///
/// On Linux, prefers Wayland (`WAYLAND_DISPLAY` / `WAYLAND_SOCKET`) and falls
/// back to X11 (`DISPLAY`). Wayland wins when both are set, which is the normal
/// state under XWayland — the native path presents without a CPU readback.
/// Set `WP_ENGINE_FORCE_X11=1` to take the X11 path anyway.
pub fn detect_platform() -> Box<dyn DisplayPlatform> {
    #[cfg(target_os = "linux")]
    {
        let force_x11 = std::env::var_os("WP_ENGINE_FORCE_X11").is_some();
        let has_wayland = std::env::var_os("WAYLAND_DISPLAY").is_some()
            || std::env::var_os("WAYLAND_SOCKET").is_some();

        if has_wayland && !force_x11 {
            tracing::debug!(target: "platform", "detected Wayland display platform");
            return Box::new(super::wayland::WaylandPlatform);
        }
        if std::env::var_os("DISPLAY").is_some() {
            tracing::debug!(target: "platform", "detected X11 display platform");
            return Box::new(super::x11::X11Platform);
        }
    }

    #[cfg(target_os = "macos")]
    {
        tracing::debug!(target: "platform", "detected macOS display platform");
        return Box::new(super::macos::MacOSPlatform);
    }

    #[cfg(not(target_os = "macos"))]
    {
        tracing::error!(target: "platform", "no supported display platform detected (WAYLAND_DISPLAY/WAYLAND_SOCKET/DISPLAY unset)");
        eprintln!(
            "error: no supported display platform detected.\n\
             wp-engine needs either a Wayland compositor with wlr-layer-shell\n\
             support, or an X11 display.\n\
             (WAYLAND_DISPLAY, WAYLAND_SOCKET and DISPLAY are all unset)\n\
             \n\
             Wayland compositors: Sway, Hyprland, river, labwc, wayfire, etc."
        );
        std::process::exit(1);
    }
}
