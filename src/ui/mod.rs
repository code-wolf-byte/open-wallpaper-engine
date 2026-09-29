use crate::platform::RenderQuality;
use crate::platform::WallpaperHandle;
use crate::render::RenderSettings;
#[cfg(not(target_os = "linux"))]
use crate::{platform::display::detect_platform, render::WallpaperContent};
use crate::workshop::{self, Wallpaper, WallpaperType};
use egui::{pos2, vec2, Align, Color32, CornerRadius, FontId, Layout, Rect, Sense};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

// ── Layout constants ──────────────────────────────────────────────────────────

const CARD_W: f32 = 220.0;
const CARD_H: f32 = 155.0;
const IMG_H: f32 = 124.0;
const CARD_GAP: f32 = 8.0;
const SIDEBAR_W: f32 = 300.0;

// ── Palette ───────────────────────────────────────────────────────────────────

const BG: Color32 = Color32::from_rgb(18, 18, 28);
const PANEL_BG: Color32 = Color32::from_rgb(24, 24, 36);
const CARD_BG: Color32 = Color32::from_rgb(32, 32, 48);
const CARD_HOVER: Color32 = Color32::from_rgb(44, 44, 64);
const CARD_SEL: Color32 = Color32::from_rgb(0, 80, 160);
const CARD_SEL_BORDER: Color32 = Color32::from_rgb(80, 150, 255);
const TEXT_PRIMARY: Color32 = Color32::from_rgb(205, 214, 244);
const TEXT_MUTED: Color32 = Color32::from_rgb(108, 112, 134);
const ACCENT_GREEN: Color32 = Color32::from_rgb(100, 200, 120);

// ── App state ─────────────────────────────────────────────────────────────────

pub struct WpApp {
    wallpapers: Vec<Wallpaper>,
    filtered: Vec<usize>, // indices into wallpapers matching current search/filter

    search: String,
    type_filter: TypeFilter,
    selected: Option<usize>,

    thumbnails: HashMap<String, egui::TextureHandle>,
    thumb_queue: Vec<usize>,

    /// In-process renderer — macOS only; on Linux the daemon owns it.
    renderer: Option<WallpaperHandle>,
    active_title: Option<String>,
    /// Live quality/volume changes forwarded to the daemon.
    #[cfg(target_os = "linux")]
    daemon: DaemonLink,
    /// What the daemon was already showing when the GUI opened.
    #[cfg(target_os = "linux")]
    initial_status: Option<mpsc::Receiver<crate::daemon::DaemonStatus>>,
    status: StatusMsg,

    file_input: String,
    settings: Arc<Mutex<RenderSettings>>,
    pending_apply: Option<mpsc::Receiver<ApplyResult>>,
    /// Cached speaker list for the audio picker, built on first use —
    /// enumerating shells out to `pactl`, so not every frame.
    audio_devices: Option<Vec<crate::platform::audio::CaptureOption>>,
    /// Index into `audio_devices`; 0 is "Automatic".
    audio_choice: usize,
}

/// `None` handle when the daemon owns the renderer.
type ApplyResult = Result<(Option<WallpaperHandle>, String), String>;

/// A worker thread that forwards live setting changes to the daemon, so a
/// dragged slider never blocks the UI on the socket (or on an apply the
/// daemon is still busy with). Changes are dropped when no daemon runs:
/// the next apply carries the current values anyway.
#[cfg(target_os = "linux")]
struct DaemonLink {
    tx: mpsc::Sender<crate::daemon::Request>,
}

#[cfg(target_os = "linux")]
impl DaemonLink {
    fn new() -> Self {
        use crate::daemon::{self, Request};
        let (tx, rx) = mpsc::channel::<Request>();
        std::thread::spawn(move || {
            while let Ok(first) = rx.recv() {
                // Coalesce a burst (slider drag) down to the latest of each.
                let (mut quality, mut volume) = (None, None);
                for req in std::iter::once(first).chain(rx.try_iter()) {
                    match req {
                        Request::SetVolume { .. } => volume = Some(req),
                        other => quality = Some(other),
                    }
                }
                if !daemon::is_running() {
                    continue;
                }
                for req in [quality, volume].into_iter().flatten() {
                    if let Err(e) = daemon::request(&req) {
                        tracing::warn!(target: "app", "daemon update failed: {e:#}");
                    }
                }
            }
        });
        Self { tx }
    }

    fn send(&self, req: crate::daemon::Request) {
        let _ = self.tx.send(req);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TypeFilter {
    All,
    Video,
    Scene,
    Web,
    Application,
}

impl TypeFilter {
    const ALL: [TypeFilter; 5] = [
        TypeFilter::All,
        TypeFilter::Video,
        TypeFilter::Scene,
        TypeFilter::Web,
        TypeFilter::Application,
    ];

    fn label(&self) -> &'static str {
        match self {
            TypeFilter::All => "All Types",
            TypeFilter::Video => "Video",
            TypeFilter::Scene => "Scene",
            TypeFilter::Web => "Web",
            TypeFilter::Application => "Application",
        }
    }

    fn matches(&self, w: &Wallpaper) -> bool {
        match self {
            TypeFilter::All => true,
            TypeFilter::Video => matches!(w.wallpaper_type(), WallpaperType::Video),
            TypeFilter::Scene => matches!(w.wallpaper_type(), WallpaperType::Scene),
            TypeFilter::Web => matches!(w.wallpaper_type(), WallpaperType::Web),
            TypeFilter::Application => matches!(w.wallpaper_type(), WallpaperType::Application),
        }
    }
}

struct StatusMsg {
    text: String,
    error: bool,
}

impl StatusMsg {
    fn ok(s: impl Into<String>) -> Self {
        Self {
            text: s.into(),
            error: false,
        }
    }
    fn err(s: impl Into<String>) -> Self {
        Self {
            text: s.into(),
            error: true,
        }
    }
}

// ── Constructor ───────────────────────────────────────────────────────────────

impl WpApp {
    pub fn new(cc: &eframe::CreationContext) -> Self {
        setup_theme(&cc.egui_ctx);

        let wallpapers = workshop::scan_wallpapers();
        let count = wallpapers.len();
        let filtered = (0..count).collect();

        let status = if count == 0 {
            StatusMsg::err("No Workshop wallpapers found — install Wallpaper Engine & subscribe")
        } else {
            StatusMsg::ok(format!("{count} wallpaper(s) found"))
        };

        #[cfg(target_os = "linux")]
        let initial_status = crate::daemon::is_running().then(|| {
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                if let Ok(crate::daemon::Response::Status(status)) =
                    crate::daemon::request(&crate::daemon::Request::Status)
                {
                    let _ = tx.send(status);
                }
            });
            rx
        });

        Self {
            wallpapers,
            filtered,
            search: String::new(),
            type_filter: TypeFilter::All,
            selected: None,
            thumbnails: HashMap::new(),
            thumb_queue: Vec::new(),
            renderer: None,
            active_title: None,
            #[cfg(target_os = "linux")]
            daemon: DaemonLink::new(),
            #[cfg(target_os = "linux")]
            initial_status,
            status,
            file_input: String::new(),
            settings: Arc::new(Mutex::new(RenderSettings::default())),
            pending_apply: None,
            audio_devices: None,
            audio_choice: 0,
        }
    }

    // ── Filter helpers ────────────────────────────────────────────────────────

    fn rebuild_filter(&mut self) {
        let q = self.search.to_lowercase();
        self.filtered = self
            .wallpapers
            .iter()
            .enumerate()
            .filter(|(_, w)| {
                let title_ok = q.is_empty() || w.title().to_lowercase().contains(&q);
                title_ok && self.type_filter.matches(w)
            })
            .map(|(i, _)| i)
            .collect();
    }

    // ── Thumbnail loading ─────────────────────────────────────────────────────

    fn enqueue_thumb(&mut self, idx: usize) {
        let id = &self.wallpapers[idx].workshop_id;
        if !self.thumbnails.contains_key(id) && !self.thumb_queue.contains(&idx) {
            self.thumb_queue.push(idx);
        }
    }

    fn flush_thumb_queue(&mut self, ctx: &egui::Context) {
        // Load at most 4 thumbnails per frame to keep frame times smooth
        let batch: Vec<usize> = self
            .thumb_queue
            .drain(..self.thumb_queue.len().min(4))
            .collect();
        for idx in batch {
            self.load_thumb(idx, ctx);
        }
    }

    fn load_thumb(&mut self, idx: usize, ctx: &egui::Context) {
        let w = &self.wallpapers[idx];
        let key = w.workshop_id.clone();
        if self.thumbnails.contains_key(&key) {
            return;
        }

        // Try preview file first, then look for any image in the directory
        let path = w.preview_file().or_else(|| {
            std::fs::read_dir(&w.path).ok()?.find_map(|e| {
                let p = e.ok()?.path();
                let ext = p.extension()?.to_str()?.to_lowercase();
                matches!(ext.as_str(), "jpg" | "jpeg" | "png").then_some(p)
            })
        });

        if let Some(p) = path {
            if let Ok(img) = image::open(&p) {
                let thumb = image::imageops::thumbnail(&img.into_rgba8(), 240, 135);
                let size = [thumb.width() as usize, thumb.height() as usize];
                let ci = egui::ColorImage::from_rgba_unmultiplied(size, &thumb.into_raw());
                let handle = ctx.load_texture(&key, ci, egui::TextureOptions::LINEAR);
                self.thumbnails.insert(key, handle);
            }
        }
    }

    // ── Wallpaper application ─────────────────────────────────────────────────

    fn apply_path(&mut self, path: PathBuf, display_title: String) {
        #[cfg(target_os = "linux")]
        self.apply_via_daemon(path, display_title);
        #[cfg(not(target_os = "linux"))]
        match WallpaperContent::from_path(&path) {
            Err(e) => self.status = StatusMsg::err(format!("Load failed: {e}")),
            Ok(content) => self.apply_content(content, display_title),
        }
    }

    fn apply_selected(&mut self) {
        let Some(idx) = self.selected else { return };

        let title = self.wallpapers[idx].title().to_string();
        #[cfg(target_os = "linux")]
        self.apply_via_daemon(self.wallpapers[idx].path.clone(), title);
        #[cfg(not(target_os = "linux"))]
        {
            // `from_wallpaper` borrows wallpapers[idx] and returns an owned Result —
            // the borrow ends here, so self is free to mutate below.
            let content = WallpaperContent::from_wallpaper(&self.wallpapers[idx]);

            match content {
                Err(e) => self.status = StatusMsg::err(e.to_string()),
                Ok(content) => self.apply_content(content, title),
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn apply_content(&mut self, content: WallpaperContent, display_title: String) {
        let old_renderer = self.renderer.take();
        let settings = Arc::clone(&self.settings);
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            if let Some(old) = old_renderer {
                old.stop();
            }
            let result = detect_platform()
                .spawn_wallpaper(
                    crate::render::ScreenContent::single(content),
                    crate::render::ScreenSettings::single(settings),
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
    fn apply_via_daemon(&mut self, path: PathBuf, display_title: String) {
        use crate::daemon::{self, ApplySpec, Request};
        let spec = {
            let s = self.settings.lock().unwrap();
            ApplySpec {
                background: path,
                quality: Some(s.quality.label().to_string()),
                volume: Some(s.volume),
                title: Some(display_title.clone()),
                ..Default::default()
            }
        };
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            let device = crate::platform::audio::preferred_device();
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
    fn clear_wallpaper(&mut self) {
        std::thread::spawn(|| {
            if let Err(e) = crate::daemon::request(&crate::daemon::Request::Clear) {
                tracing::warn!(target: "app", "clearing wallpaper failed: {e:#}");
            }
        });
        self.active_title = None;
        self.status = StatusMsg::ok("Wallpaper removed");
    }

    #[cfg(target_os = "linux")]
    fn poll_initial_status(&mut self) {
        let Some(rx) = &self.initial_status else {
            return;
        };
        match rx.try_recv() {
            Ok(status) => {
                if let Some(active) = status.active {
                    self.active_title = Some(
                        active
                            .title
                            .unwrap_or_else(|| active.background.display().to_string()),
                    );
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

    fn poll_pending_apply(&mut self) {
        let Some(rx) = &self.pending_apply else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok((handle, title))) => {
                if let Some(handle) = handle {
                    self.renderer = Some(handle);
                }
                self.active_title = Some(title.clone());
                self.status = StatusMsg::ok(format!("Applied \"{title}\""));
                self.pending_apply = None;
            }
            Ok(Err(e)) => {
                self.status = StatusMsg::err(e);
                self.pending_apply = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                self.status = StatusMsg::err("Apply thread crashed");
                self.pending_apply = None;
            }
        }
    }
}

// ── eframe::App ───────────────────────────────────────────────────────────────

impl eframe::App for WpApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_pending_apply();
        #[cfg(target_os = "linux")]
        self.poll_initial_status();
        #[cfg(target_os = "linux")]
        let waiting = self.pending_apply.is_some() || self.initial_status.is_some();
        #[cfg(not(target_os = "linux"))]
        let waiting = self.pending_apply.is_some();
        if waiting {
            ctx.request_repaint();
        }
        self.flush_thumb_queue(ctx);
        self.render_toolbar(ctx);
        self.render_sidebar(ctx);
        self.render_status_bar(ctx);
        self.render_grid(ctx);
    }
}

// ── Panels ────────────────────────────────────────────────────────────────────

impl WpApp {
    fn render_toolbar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("toolbar")
            .frame(
                egui::Frame::default()
                    .fill(PANEL_BG)
                    .inner_margin(egui::Margin::symmetric(12, 8)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("🖼 wp-engine")
                            .size(16.0)
                            .color(TEXT_PRIMARY)
                            .strong(),
                    );
                    ui.separator();

                    ui.label(egui::RichText::new("Search").color(TEXT_MUTED).size(13.0));
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut self.search)
                            .desired_width(220.0)
                            .hint_text("Filter by title…"),
                    );
                    if r.changed() {
                        self.rebuild_filter();
                    }

                    ui.add_space(8.0);

                    egui::ComboBox::from_id_salt("type_filter")
                        .selected_text(self.type_filter.label())
                        .width(130.0)
                        .show_ui(ui, |ui| {
                            for tf in TypeFilter::ALL.clone() {
                                let label = tf.label();
                                if ui
                                    .selectable_value(&mut self.type_filter, tf, label)
                                    .clicked()
                                {
                                    self.rebuild_filter();
                                }
                            }
                        });

                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(format!("{} shown", self.filtered.len()))
                            .color(TEXT_MUTED)
                            .size(12.0),
                    );
                });
            });
    }

    fn render_status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_bar")
            .frame(
                egui::Frame::default()
                    .fill(PANEL_BG)
                    .inner_margin(egui::Margin::symmetric(12, 6)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if let Some(title) = &self.active_title {
                        ui.label(
                            egui::RichText::new(format!("▶ {title}"))
                                .color(ACCENT_GREEN)
                                .size(12.0),
                        );
                        #[cfg(target_os = "linux")]
                        if ui.small_button("■ Stop").clicked() {
                            self.clear_wallpaper();
                        }
                        ui.separator();
                    }
                    let color = if self.status.error {
                        Color32::from_rgb(243, 139, 168)
                    } else {
                        TEXT_MUTED
                    };
                    ui.label(
                        egui::RichText::new(&self.status.text)
                            .color(color)
                            .size(12.0),
                    );
                });
            });
    }

    fn render_sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("sidebar")
            .exact_width(SIDEBAR_W)
            .frame(
                egui::Frame::default()
                    .fill(PANEL_BG)
                    .inner_margin(egui::Margin::same(12)),
            )
            .show(ctx, |ui| match self.selected {
                Some(idx) => self.render_details(ui, idx),
                None => self.render_no_selection(ui),
            });
    }

    // Details panel — collects all data first to avoid borrow conflicts
    fn render_details(&mut self, ui: &mut egui::Ui, idx: usize) {
        // ── Collect data from the wallpaper (end borrows before any mutation) ──
        let title = self.wallpapers[idx].title().to_string();
        let wtype = self.wallpapers[idx].wallpaper_type().clone();
        let tags = self.wallpapers[idx]
            .project
            .tags
            .clone()
            .unwrap_or_default();
        let workshop_id = self.wallpapers[idx].workshop_id.clone();
        let file = self.wallpapers[idx].wallpaper_file();
        let file_exists = file.as_ref().map(|p| p.exists()).unwrap_or(false);
        let file_name = file
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let can_apply = (matches!(
            wtype,
            WallpaperType::Video | WallpaperType::Scene | WallpaperType::Unknown
        ) || (matches!(wtype, WallpaperType::Web)
            && crate::render::web::is_supported()))
            && (file_exists || matches!(wtype, WallpaperType::Scene));
        let thumb = self.thumbnails.get(&workshop_id).cloned();

        // ── Render ────────────────────────────────────────────────────────────

        // Preview (full width, 16:9)
        let pw = ui.available_width();
        let ph = pw * 9.0 / 16.0;
        let (preview_rect, _) = ui.allocate_exact_size(vec2(pw, ph), Sense::hover());
        {
            let p = ui.painter_at(preview_rect);
            p.rect_filled(preview_rect, CornerRadius::same(6), Color32::from_gray(16));
            if let Some(tex) = &thumb {
                p.image(
                    tex.id(),
                    preview_rect,
                    Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                    Color32::WHITE,
                );
            } else {
                p.text(
                    preview_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "No Preview",
                    FontId::proportional(12.0),
                    TEXT_MUTED,
                );
            }
        }

        ui.add_space(10.0);

        // Title
        ui.label(
            egui::RichText::new(&title)
                .size(15.0)
                .color(TEXT_PRIMARY)
                .strong(),
        );
        ui.add_space(4.0);

        // Type badge
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Type ").color(TEXT_MUTED).size(12.0));
            egui::Frame::default()
                .fill(type_badge_color(&wtype))
                .corner_radius(CornerRadius::same(3))
                .inner_margin(egui::Margin::symmetric(6, 2))
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new(wtype.to_string())
                            .size(11.0)
                            .color(Color32::WHITE),
                    );
                });
        });

        // Tags
        if !tags.is_empty() {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Tags: ").color(TEXT_MUTED).size(12.0));
                for tag in tags.iter().take(8) {
                    egui::Frame::default()
                        .fill(Color32::from_gray(40))
                        .corner_radius(CornerRadius::same(3))
                        .inner_margin(egui::Margin::symmetric(5, 2))
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new(tag).size(11.0).color(TEXT_MUTED));
                        });
                }
            });
        }

        // Workshop ID + file
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(format!("ID: {workshop_id}"))
                .size(11.0)
                .color(TEXT_MUTED),
        );

        match &wtype {
            // Scene wallpapers are rendered via the built-in scene engine
            WallpaperType::Scene => {
                ui.label(
                    egui::RichText::new("Scene wallpaper — rendered via built-in engine")
                        .size(11.0)
                        .color(TEXT_MUTED),
                );
            }
            // For renderable types, show whether the file exists
            _ if !file_name.is_empty() => {
                let file_color = if file_exists {
                    TEXT_MUTED
                } else {
                    Color32::from_rgb(243, 139, 168)
                };
                ui.label(
                    egui::RichText::new(format!(
                        "File: {file_name}{}",
                        if file_exists { "" } else { " (missing)" }
                    ))
                    .size(11.0)
                    .color(file_color),
                );
            }
            _ => {}
        }

        ui.add_space(12.0);
        ui.separator();

        // Audio-reactive wallpapers can be driven from any capture device;
        // only offer the choice where it actually changes something.
        let audio_dir = self.wallpapers[idx].path.clone();
        let uses_audio = crate::platform::audio::wallpaper_uses_audio(&audio_dir);

        // Apply button + settings panel — pinned to bottom
        ui.with_layout(Layout::bottom_up(Align::Center), |ui| {
            ui.add_space(6.0);

            if uses_audio {
                let devices = self
                    .audio_devices
                    .get_or_insert_with(crate::platform::audio::list_audio_outputs);
                let current = devices
                    .get(self.audio_choice)
                    .map(|d| d.label.clone())
                    .unwrap_or_else(|| "Automatic".to_string());
                ui.add_space(4.0);
                let mut changed = None;
                egui::ComboBox::from_id_salt("audio_device")
                    .selected_text(current)
                    .width(260.0)
                    .show_ui(ui, |ui| {
                        for (i, d) in devices.iter().enumerate() {
                            if ui
                                .selectable_label(self.audio_choice == i, &d.label)
                                .clicked()
                            {
                                changed = Some((i, d.device.clone()));
                            }
                        }
                    });
                if let Some((i, device)) = changed {
                    self.audio_choice = i;
                    crate::platform::audio::set_preferred_device(device);
                    self.status =
                        StatusMsg::ok("Audio source set — re-apply the wallpaper to use it");
                }
                ui.label(
                    egui::RichText::new("🔊  React to sound from")
                        .size(11.0)
                        .weak(),
                );
                ui.add_space(2.0);
            }

            if can_apply {
                let btn = egui::Button::new(
                    egui::RichText::new("▶  Apply Wallpaper")
                        .size(14.0)
                        .color(Color32::WHITE),
                )
                .fill(Color32::from_rgb(38, 130, 65))
                .min_size(vec2(ui.available_width(), 38.0))
                .corner_radius(CornerRadius::same(6));

                if ui.add(btn).clicked() {
                    self.apply_selected();
                }
            } else {
                let reason = match wtype {
                    WallpaperType::Web => "Web wallpapers need a build with --features web",
                    WallpaperType::Application => "Windows-only",
                    _ => "File missing",
                };
                ui.add_enabled(
                    false,
                    egui::Button::new(egui::RichText::new(format!("✖  {reason}")).size(13.0))
                        .min_size(vec2(ui.available_width(), 38.0))
                        .corner_radius(CornerRadius::same(6)),
                );
            }

            // ── Render Settings ───────────────────────────────────────────────
            ui.separator();
            ui.label(
                egui::RichText::new("Audio playback coming soon")
                    .color(TEXT_MUTED)
                    .small(),
            );
            {
                let mut s = self.settings.lock().unwrap();
                let volume_changed = ui
                    .add(
                        egui::Slider::new(&mut s.volume, 0.0..=1.0)
                            .text("Volume")
                            .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
                    )
                    .changed();
                let mut quality_changed = false;
                egui::ComboBox::from_label("Quality")
                    .selected_text(s.quality.label())
                    .show_ui(ui, |ui| {
                        for q in RenderQuality::ALL {
                            quality_changed |=
                                ui.selectable_value(&mut s.quality, q, q.label()).changed();
                        }
                    });
                #[cfg(target_os = "linux")]
                {
                    use crate::daemon::Request;
                    if volume_changed {
                        self.daemon.send(Request::SetVolume { volume: s.volume });
                    }
                    if quality_changed {
                        self.daemon.send(Request::SetQuality {
                            quality: s.quality.label().to_string(),
                        });
                    }
                }
                let _ = (volume_changed, quality_changed);
            }
            ui.label(
                egui::RichText::new("Render Settings")
                    .color(TEXT_MUTED)
                    .size(12.0),
            );
            ui.separator();
        });
    }

    fn render_no_selection(&mut self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.add_space(32.0);
            ui.label(
                egui::RichText::new("No wallpaper selected")
                    .color(TEXT_MUTED)
                    .size(14.0),
            );
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new("Click a wallpaper in the grid,\nor set a file path below.")
                    .color(TEXT_MUTED)
                    .size(12.0),
            );

            ui.add_space(24.0);
            ui.separator();
            ui.add_space(16.0);

            ui.label(
                egui::RichText::new("Apply an image file directly:")
                    .color(TEXT_MUTED)
                    .size(12.0),
            );
            ui.add_space(4.0);
            ui.add(
                egui::TextEdit::singleline(&mut self.file_input)
                    .desired_width(ui.available_width())
                    .hint_text("/path/to/wallpaper.png"),
            );
            ui.add_space(6.0);

            let apply_btn = egui::Button::new(
                egui::RichText::new("Apply File")
                    .color(Color32::WHITE)
                    .size(13.0),
            )
            .fill(Color32::from_rgb(40, 100, 180))
            .min_size(vec2(ui.available_width(), 32.0))
            .corner_radius(CornerRadius::same(6));

            if ui.add(apply_btn).clicked() && !self.file_input.trim().is_empty() {
                let path = PathBuf::from(self.file_input.trim());
                let name = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                self.apply_path(path, name);
            }

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(10.0);
            ui.label(
                egui::RichText::new(
                    "Workshop path:\n\
                     ~/.local/share/Steam/steamapps/\n\
                     workshop/content/431960/",
                )
                .size(11.0)
                .color(TEXT_MUTED),
            );
        });
    }

    fn render_grid(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(BG)
                    .inner_margin(egui::Margin::same(10)),
            )
            .show(ctx, |ui| {
                if self.wallpapers.is_empty() {
                    ui.vertical_centered(|ui| {
                        ui.add_space(80.0);
                        ui.label(
                            egui::RichText::new("No Workshop wallpapers found")
                                .size(18.0)
                                .color(TEXT_MUTED),
                        );
                        ui.add_space(10.0);
                        ui.label(
                            egui::RichText::new(
                                "Install Wallpaper Engine on Steam and subscribe to wallpapers,\n\
                                 or use the file path box on the right to apply any PNG/JPEG.",
                            )
                            .size(13.0)
                            .color(TEXT_MUTED),
                        );
                    });
                    return;
                }

                let cols = ((ui.available_width() / (CARD_W + CARD_GAP)).floor() as usize).max(1);
                let filtered = self.filtered.clone();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    for chunk in filtered.chunks(cols) {
                        ui.horizontal(|ui| {
                            ui.set_height(CARD_H);
                            for &idx in chunk {
                                let id = &self.wallpapers[idx].workshop_id;
                                let selected = self.selected == Some(idx);
                                let thumb = self.thumbnails.get(id).cloned();

                                if thumb.is_none() {
                                    self.enqueue_thumb(idx);
                                }

                                let (rect, resp) =
                                    ui.allocate_exact_size(vec2(CARD_W, CARD_H), Sense::click());

                                if ui.is_rect_visible(rect) {
                                    let w = &self.wallpapers[idx];
                                    draw_card(ui, rect, &resp, w, thumb.as_ref(), selected);
                                }

                                if resp.clicked() {
                                    self.selected = Some(idx);
                                    self.enqueue_thumb(idx);
                                }

                                resp.on_hover_text(self.wallpapers[idx].title());
                            }
                        });
                        ui.add_space(CARD_GAP);
                    }
                });
            });
    }
}

// ── Card drawing ──────────────────────────────────────────────────────────────

fn draw_card(
    ui: &egui::Ui,
    rect: Rect,
    resp: &egui::Response,
    w: &Wallpaper,
    thumb: Option<&egui::TextureHandle>,
    selected: bool,
) {
    let painter = ui.painter_at(rect);

    // Background
    let bg = if selected {
        CARD_SEL
    } else if resp.hovered() {
        CARD_HOVER
    } else {
        CARD_BG
    };
    painter.rect_filled(rect, CornerRadius::same(6), bg);

    if selected {
        // Paint a 2px border by drawing a slightly-expanded rect beneath the card
        painter.rect_filled(rect.expand(2.0), CornerRadius::same(8), CARD_SEL_BORDER);
        painter.rect_filled(rect, CornerRadius::same(6), bg);
    }

    // Thumbnail
    let img_rect = Rect::from_min_size(rect.min, vec2(CARD_W, IMG_H));
    if let Some(tex) = thumb {
        painter.image(
            tex.id(),
            img_rect,
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    } else {
        painter.rect_filled(img_rect, CornerRadius::same(6), Color32::from_gray(22));
        painter.text(
            img_rect.center(),
            egui::Align2::CENTER_CENTER,
            "⏳",
            FontId::proportional(22.0),
            Color32::from_gray(55),
        );
    }

    // Type badge (top-right corner of the thumbnail)
    let badge_label = type_badge_label(w.wallpaper_type());
    let badge_font = FontId::proportional(9.0);
    let badge_galley = painter.layout_no_wrap(badge_label.to_string(), badge_font, Color32::WHITE);
    let badge_size = badge_galley.size() + vec2(8.0, 4.0);
    let badge_tl = pos2(rect.right() - badge_size.x - 4.0, rect.min.y + 4.0);
    let badge_rect = Rect::from_min_size(badge_tl, badge_size);
    painter.rect_filled(
        badge_rect,
        CornerRadius::same(3),
        type_badge_color(w.wallpaper_type()),
    );
    painter.galley(badge_tl + vec2(4.0, 2.0), badge_galley, Color32::WHITE);

    // Title
    let title = truncate(w.title(), 26);
    painter.text(
        pos2(rect.min.x + 6.0, rect.min.y + IMG_H + 5.0),
        egui::Align2::LEFT_TOP,
        title,
        FontId::proportional(11.5),
        TEXT_PRIMARY,
    );
}

// ── Utilities ─────────────────────────────────────────────────────────────────

fn type_badge_color(t: &WallpaperType) -> Color32 {
    match t {
        WallpaperType::Video => Color32::from_rgb(55, 95, 200),
        WallpaperType::Scene => Color32::from_rgb(130, 65, 200),
        WallpaperType::Web => Color32::from_rgb(50, 160, 80),
        WallpaperType::Application => Color32::from_rgb(180, 95, 50),
        WallpaperType::Unknown => Color32::from_gray(75),
    }
}

fn type_badge_label(t: &WallpaperType) -> &'static str {
    match t {
        WallpaperType::Video => "VIDEO",
        WallpaperType::Scene => "SCENE",
        WallpaperType::Web => "WEB",
        WallpaperType::Application => "APP",
        WallpaperType::Unknown => "?",
    }
}

fn truncate(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

fn setup_theme(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = PANEL_BG;
    v.window_fill = BG;
    v.extreme_bg_color = Color32::from_gray(12);
    v.widgets.noninteractive.bg_fill = Color32::from_gray(28);
    v.widgets.inactive.bg_fill = Color32::from_gray(36);
    v.widgets.hovered.bg_fill = Color32::from_gray(50);
    v.widgets.active.bg_fill = Color32::from_rgb(0, 80, 160);
    v.selection.bg_fill = Color32::from_rgb(0, 80, 160);
    ctx.set_visuals(v);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = vec2(6.0, 4.0);
    ctx.set_style(style);
}
