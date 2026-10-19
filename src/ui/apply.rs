//! Showing what was picked: applying a wallpaper to the chosen display or
//! the saved setup (through the background daemon on Linux, in-process
//! elsewhere), Cancel's revert, and the daemon link that forwards live
//! setting changes.

use super::*;

impl WpApp {
    /// Show a file or folder that isn't a Workshop item on every display.
    pub(super) fn apply_path(&mut self, path: PathBuf, display_title: String) {
        let properties = self.saved.property_overrides(&path.to_string_lossy());
        #[cfg(target_os = "linux")]
        self.apply_via_daemon(path, HashMap::new(), display_title, properties);
        #[cfg(not(target_os = "linux"))]
        match WallpaperContent::from_path(&path) {
            Err(e) => self.status = StatusMsg::err(format!("Load failed: {e}")),
            Ok(content) => self.apply_content(content, display_title, properties),
        }
    }

    /// WE's click-to-apply: show wallpaper `idx` on the chosen display (or
    /// all of them) and remember it, as `wp-engine config` would.
    pub(super) fn apply_wallpaper(&mut self, idx: usize) {
        if let Err(reason) = can_apply(&self.wallpapers[idx]) {
            self.status = StatusMsg::err(reason);
            return;
        }
        let id = self.wallpapers[idx].workshop_id.clone();
        match self.target_screen.clone() {
            Some(screen) => {
                self.saved.screens.insert(screen, id);
            }
            None => {
                self.saved.default_background = Some(id);
                self.saved.screens.clear();
            }
        }
        self.save_settings();
        let title = self.wallpapers[idx].title().to_string();
        #[cfg(target_os = "linux")]
        self.apply_saved(title, self.wallpapers[idx].path.clone());
        #[cfg(not(target_os = "linux"))]
        {
            let properties = self
                .saved
                .property_overrides(&self.wallpapers[idx].path.to_string_lossy());
            match WallpaperContent::from_wallpaper(&self.wallpapers[idx]) {
                Err(e) => self.status = StatusMsg::err(e.to_string()),
                Ok(content) => self.apply_content(content, title, properties),
            }
        }
    }

    /// Show the saved default and per-display wallpapers; `fallback` stands
    /// in for a missing default.
    #[cfg(target_os = "linux")]
    pub(super) fn apply_saved(&mut self, title: String, fallback: PathBuf) {
        let path_of = |id: &str| {
            self.wallpapers
                .iter()
                .find(|w| w.workshop_id == id)
                .map(|w| w.path.clone())
                .or_else(|| workshop::find_item_dir(id, None))
        };
        let background = self
            .saved
            .default_background
            .as_deref()
            .and_then(path_of)
            .unwrap_or(fallback);
        let screens = self
            .saved
            .screens
            .iter()
            .filter_map(|(screen, id)| Some((screen.clone(), path_of(id)?)))
            .collect();
        let properties = self.saved.property_overrides(&background.to_string_lossy());
        self.apply_via_daemon(background, screens, title, properties);
    }

    #[cfg(not(target_os = "linux"))]
    pub(super) fn apply_content(
        &mut self,
        content: WallpaperContent,
        display_title: String,
        properties: HashMap<String, String>,
    ) {
        crate::workshop::properties::set_global_overrides(properties);
        let old_renderer = self.renderer.take();
        let settings = Arc::clone(&self.settings);
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            if let Some(old) = old_renderer {
                old.stop();
            }
            let result = detect_platform()
                .spawn_wallpaper(
                    crate::content::ScreenContent::single(content),
                    crate::content::ScreenSettings::single(settings),
                )
                .map(|handle| (Some(handle), display_title.clone()))
                .map_err(|e| format!("{e:#}"));
            let _ = tx.send(result);
        });

        self.pending_apply = Some(rx);
        self.status = StatusMsg::ok("Applying...");
    }

    /// Hand the wallpaper to the background daemon (starting it if needed),
    /// so it keeps running after this window closes.
    #[cfg(target_os = "linux")]
    pub(super) fn apply_via_daemon(
        &mut self,
        path: PathBuf,
        screens: HashMap<String, PathBuf>,
        display_title: String,
        properties: HashMap<String, String>,
    ) {
        use crate::daemon::{self, ApplySpec, Request};
        let spec = {
            let s = self.settings.lock().unwrap();
            ApplySpec {
                background: path,
                screens,
                properties,
                quality: Some(s.quality.label().to_string()),
                volume: Some(s.volume),
                title: Some(display_title.clone()),
                ..Default::default()
            }
        };
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            let device = crate::services::audio_devices::preferred_device();
            let result = daemon::request_or_spawn(&Request::SetAudioDevice { device })
                .and_then(|_| daemon::request(&Request::Apply(spec)))
                .map(|_| (None, display_title))
                .map_err(|e| format!("{e:#}"));
            let _ = tx.send(result);
        });

        self.pending_apply = Some(rx);
        self.status = StatusMsg::ok("Applying...");
    }

    /// Remove the wallpaper the daemon is showing (the daemon stays up).
    #[cfg(target_os = "linux")]
    pub(super) fn clear_wallpaper(&mut self) {
        std::thread::spawn(|| {
            if let Err(e) = crate::daemon::request(&crate::daemon::Request::Clear) {
                tracing::warn!(target: "app", "clearing wallpaper failed: {e:#}");
            }
        });
        self.active_title = None;
        self.status = StatusMsg::ok("Wallpaper removed");
    }

    /// WE's Cancel: restore the settings and wallpaper from when the window
    /// opened, then close.
    pub(super) fn cancel(&mut self, ctx: &egui::Context) {
        if self.saved == self.initial_saved {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        self.saved = self.initial_saved.clone();
        self.save_settings();
        #[cfg(target_os = "linux")]
        match self.initial.clone() {
            Some((path, title)) => {
                self.apply_saved(title, path);
                self.close_after_apply = true;
                return;
            }
            None => self.clear_wallpaper(),
        }
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
    }

    #[cfg(target_os = "linux")]
    pub(super) fn poll_initial_status(&mut self) {
        let Some(rx) = &self.initial_status else {
            return;
        };
        match rx.try_recv() {
            Ok(status) => {
                if let Some(active) = status.active {
                    let title = active
                        .title
                        .unwrap_or_else(|| active.background.display().to_string());
                    self.active_title = Some(title.clone());
                    self.initial = Some((active.background, title));
                    if let Some(q) = RenderQuality::parse(&active.quality) {
                        self.settings.lock().unwrap().quality = q;
                    }
                }
                self.initial_status = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => self.initial_status = None,
        }
    }

    pub(super) fn poll_pending_apply(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.pending_apply else {
            return;
        };
        let done = match rx.try_recv() {
            Ok(Ok((handle, title))) => {
                if let Some(handle) = handle {
                    self.renderer = Some(handle);
                }
                self.active_title = Some(title.clone());
                self.status = StatusMsg::ok(format!("Applied \"{title}\""));
                true
            }
            Ok(Err(e)) => {
                self.status = StatusMsg::err(e);
                true
            }
            Err(mpsc::TryRecvError::Empty) => false,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.status = StatusMsg::err("Apply thread crashed");
                true
            }
        };
        if done {
            self.pending_apply = None;
            if self.close_after_apply {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }
}

/// A worker thread that forwards live setting changes to the daemon, so a
/// dragged slider never blocks the UI on the socket (or on an apply the
/// daemon is still busy with). Changes are dropped when no daemon runs:
/// the next apply carries the current values anyway.
#[cfg(target_os = "linux")]
pub(super) struct DaemonLink {
    tx: mpsc::Sender<crate::daemon::Request>,
}

#[cfg(target_os = "linux")]
impl DaemonLink {
    pub(super) fn new() -> Self {
        use crate::daemon::{self, Request};
        let (tx, rx) = mpsc::channel::<Request>();
        std::thread::spawn(move || {
            while let Ok(first) = rx.recv() {
                // Coalesce a burst (slider drag) down to the latest of each kind.
                let mut latest = HashMap::new();
                for req in std::iter::once(first).chain(rx.try_iter()) {
                    latest.insert(std::mem::discriminant(&req), req);
                }
                if !daemon::is_running() {
                    continue;
                }
                for req in latest.into_values() {
                    if let Err(e) = daemon::request(&req) {
                        tracing::warn!(target: "app", "daemon update failed: {e:#}");
                    }
                }
            }
        });
        Self { tx }
    }

    pub(super) fn send(&self, req: crate::daemon::Request) {
        let _ = self.tx.send(req);
    }
}

/// Why `w` can't be shown here, if it can't.
pub(super) fn can_apply(w: &Wallpaper) -> Result<(), &'static str> {
    let file_ok = || w.wallpaper_file().is_some_and(|p| p.exists());
    match w.wallpaper_type() {
        WallpaperType::Scene | WallpaperType::Preset => Ok(()),
        WallpaperType::Application => Err("Application wallpapers only run on Windows"),
        WallpaperType::Web if !crate::content::web::is_supported() => {
            Err("Web wallpapers need a build with --features web")
        }
        _ if file_ok() => Ok(()),
        _ => Err("The wallpaper's file is missing"),
    }
}
