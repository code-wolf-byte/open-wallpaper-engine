//! Application lifecycle (the Rust counterpart of the C++
//! `WallpaperApplication`).
//!
//! Owns the flow from configuration to a running wallpaper:
//! resolve content → install property overrides → spawn the platform
//! renderer → block until a stop request (SIGINT/SIGTERM or [`stop`]).

#[cfg(not(target_os = "macos"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_os = "macos"))]
use std::time::Duration;

use anyhow::{Context, Result};

use crate::platform::{self, WallpaperHandle};
use crate::render::{ScreenContent, ScreenSettings, WallpaperContent};

pub mod application_context;
pub use application_context::ApplicationContext;

/// Set by the signal handler; polled by [`WallpaperApplication::show`].
/// Linux/Wayland only — macOS runs the event loop inline on the main thread.
#[cfg(not(target_os = "macos"))]
static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

#[cfg(not(target_os = "macos"))]
extern "C" fn handle_stop_signal(_sig: libc::c_int) {
    STOP_REQUESTED.store(true, Ordering::SeqCst);
}

pub struct WallpaperApplication {
    context: ApplicationContext,
    handle: Option<WallpaperHandle>,
}

impl WallpaperApplication {
    /// Prepare the application: installs the context's property overrides so
    /// every scene loaded afterwards resolves user settings against them.
    pub fn new(context: ApplicationContext) -> Self {
        crate::engine::properties::set_global_overrides(context.properties.clone());
        Self {
            context,
            handle: None,
        }
    }

    pub fn context(&self) -> &ApplicationContext {
        &self.context
    }

    /// Resolves `context.background` (the default, shown on any output with
    /// no more specific entry) and `context.screens` (per-output overrides,
    /// only ever populated by `wp-engine run`) into a [`ScreenContent`] the
    /// platform layer can hand out per discovered output. A screen whose
    /// content fails to load is logged and skipped — falls back to the
    /// default for that one output rather than failing the whole wallpaper
    /// over one bad per-screen entry.
    fn resolve_screen_content(&self) -> Result<ScreenContent> {
        let default = WallpaperContent::from_any_path(&self.context.background).with_context(
            || format!("loading wallpaper {}", self.context.background.display()),
        )?;
        let mut by_output = std::collections::HashMap::new();
        for (screen, path) in &self.context.screens {
            match WallpaperContent::from_any_path(path) {
                Ok(content) => {
                    by_output.insert(screen.clone(), content);
                }
                Err(e) => {
                    tracing::error!(
                        target: "app",
                        "screen '{screen}': failed to load {}: {e} — falling back to the default wallpaper",
                        path.display()
                    );
                }
            }
        }
        Ok(ScreenContent { by_output, default })
    }

    /// The [`ScreenSettings`] twin of `resolve_screen_content` — `screen_
    /// settings` was already built per-screen at the CLI layer (`main.rs`,
    /// where each screen's saved quality override is looked up by its
    /// original workshop-ID-or-path string), so this is just a wrap, not a
    /// second resolution pass.
    fn resolve_screen_settings(&self) -> ScreenSettings {
        ScreenSettings {
            by_output: self.context.screen_settings.clone(),
            default: self.context.settings.clone(),
        }
    }

    /// Load the configured background(s) and start rendering.
    #[tracing::instrument(target = "app", level = "info", skip(self), fields(background = %self.context.background.display()))]
    pub fn setup(&mut self) -> Result<()> {
        // macOS runs the wallpaper on the main thread in `show()` (winit/AppKit
        // require the event loop there), so there's nothing to spawn here.
        #[cfg(target_os = "macos")]
        {
            tracing::debug!(target: "app", "macOS: wallpaper runs on the main thread in show()");
            return Ok(());
        }
        #[cfg(not(target_os = "macos"))]
        {
            tracing::info!(target: "app", "loading wallpaper content");
            let content = self.resolve_screen_content()?;
            let settings = self.resolve_screen_settings();
            tracing::debug!(target: "app", "content loaded; spawning platform renderer");
            let handle = platform::detect_platform().spawn_wallpaper(content, settings)?;
            self.handle = Some(handle);
            tracing::info!(target: "app", "wallpaper renderer started");
            Ok(())
        }
    }

    /// Run the wallpaper until SIGINT/SIGTERM (or [`Self::stop`] from another
    /// handle) is received, then shut the renderer down cleanly.
    ///
    /// Calls [`Self::setup`] first when it hasn't run yet.
    pub fn show(&mut self) -> Result<()> {
        // macOS: the winit/AppKit event loop must own the main thread, so run
        // the wallpaper inline here rather than spawning a renderer and parking.
        #[cfg(target_os = "macos")]
        {
            let content =
                WallpaperContent::from_any_path(&self.context.background).with_context(|| {
                    format!("loading wallpaper {}", self.context.background.display())
                })?;
            return platform::macos::run_wallpaper_on_main(content);
        }
        #[cfg(not(target_os = "macos"))]
        {
            if self.handle.is_none() {
                self.setup()?;
            }

            STOP_REQUESTED.store(false, Ordering::SeqCst);
            let handler =
                handle_stop_signal as extern "C" fn(libc::c_int) as *const () as libc::sighandler_t;
            unsafe {
                libc::signal(libc::SIGINT, handler);
                libc::signal(libc::SIGTERM, handler);
            }

            while !STOP_REQUESTED.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(100));
            }

            self.cleanup();
            Ok(())
        }
    }

    /// Request the render loop to end (usable from signal-safe contexts).
    pub fn stop() {
        #[cfg(not(target_os = "macos"))]
        STOP_REQUESTED.store(true, Ordering::SeqCst);
    }

    /// Stop the platform renderer and wait for its thread to exit.
    pub fn cleanup(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.stop();
        }
    }
}

impl Drop for WallpaperApplication {
    fn drop(&mut self) {
        self.cleanup();
    }
}
