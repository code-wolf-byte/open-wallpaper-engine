//! The desktop app, laid out like Wallpaper Engine's own browser
//! (`ui/dist/views/views/browsewallpapers.html` in a WE install): source
//! tabs, search, filters and sorting over a thumbnail grid, the playlist bar
//! under it, the selected wallpaper's details and properties on the right,
//! and an "Open Wallpaper" / OK / Cancel footer. As in WE, clicking a
//! wallpaper applies it to the chosen display straight away; Cancel puts
//! back what was showing (and saved) when the window opened.

use crate::content::RenderSettings;
use crate::platform::{RenderQuality, WallpaperHandle};
use crate::workshop::{self, Wallpaper, WallpaperType};
mod apply;
use apply::*;

#[cfg(not(target_os = "linux"))]
use crate::{content::WallpaperContent, platform::display::detect_platform};
use egui::{pos2, vec2, Align, Color32, CornerRadius, FontId, Layout, Rect, Sense, Stroke};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

mod config;

// ── Layout ────────────────────────────────────────────────────────────────────

const TILE: f32 = 168.0;
const TILE_GAP: f32 = 6.0;
/// `.browseWallpaperDetailView` at full window width.
const DETAILS_W: f32 = 400.0;
const FILTER_W: f32 = 200.0;
const PLAYLIST_THUMB: f32 = 56.0;
/// `#999`, the muted text colour of both stock skins.
const TEXT_MUTED: Color32 = Color32::from_gray(0x99);
const DANGER: Color32 = Color32::from_rgb(0xd9, 0x53, 0x4f);

const WORKSHOP_URL: &str = "https://steamcommunity.com/app/431960/workshop/";
const AGE_RATINGS: [&str; 3] = ["Everyone", "Questionable", "Mature"];
const TYPES: [&str; 5] = ["Scene", "Video", "Web", "Application", "Preset"];

/// Wallpaper Engine's two stock skins (`styles/main.css` and
/// `styles/skindark.css`), picked by the system theme.
#[derive(Clone, Copy)]
struct Skin {
    bg: Color32,
    panel: Color32,
    tile: Color32,
    accent: Color32,
    text: Color32,
    pill: Color32,
}

impl Skin {
    const LIGHT: Skin = Skin {
        bg: Color32::from_rgb(0xf3, 0xf3, 0xf3),
        panel: Color32::WHITE,
        tile: Color32::from_gray(0xdd),
        accent: Color32::from_rgb(0x29, 0x73, 0xf3),
        text: Color32::from_gray(0x33),
        pill: Color32::from_gray(0xcc),
    };
    const DARK: Skin = Skin {
        bg: Color32::from_gray(0x22),
        panel: Color32::from_gray(0x2a),
        tile: Color32::from_gray(0x33),
        accent: Color32::from_rgb(0x41, 0x83, 0xf5),
        text: Color32::from_gray(0xdd),
        pill: Color32::from_gray(0x3b),
    };

    fn of(ctx: &egui::Context) -> Skin {
        if ctx.style().visuals.dark_mode {
            Self::DARK
        } else {
            Self::LIGHT
        }
    }
}

// ── App state ─────────────────────────────────────────────────────────────────

pub struct WpApp {
    wallpapers: Vec<Wallpaper>,
    /// When each wallpaper's folder last changed, for "Date installed".
    installed: Vec<SystemTime>,
    /// Every tag any wallpaper carries, for the filter panel.
    all_tags: Vec<String>,
    filtered: Vec<usize>, // indices into wallpapers matching current search/filter

    search: String,
    filters: Filters,
    show_filter: bool,
    sort: SortKey,
    descending: bool,
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
    /// The wallpaper showing when the window opened, for Cancel.
    initial: Option<(PathBuf, String)>,
    /// `settings.json` as it was when the window opened, for Cancel.
    initial_saved: crate::settings::WpSettings,
    /// Close once the pending apply finishes (Cancel's revert).
    close_after_apply: bool,
    status: StatusMsg,

    settings: Arc<Mutex<RenderSettings>>,
    pending_apply: Option<mpsc::Receiver<ApplyResult>>,
    /// Cached speaker list for the audio picker, built on first use —
    /// enumerating shells out to `pactl`, so not every frame.
    audio_devices: Option<Vec<crate::services::audio_devices::CaptureOption>>,
    /// Index into `audio_devices`; 0 is "Automatic".
    audio_choice: usize,

    /// `settings.json`, edited by the property editor and settings window.
    saved: crate::settings::WpSettings,
    prop_editor: Option<config::PropertyEditor>,
    show_settings: bool,
    settings_tab: config::SettingsTab,
    show_displays: bool,
    /// The display clicking a wallpaper applies to; `None` = all of them.
    target_screen: Option<String>,
    /// Output names, listed on first use.
    outputs: Option<Vec<String>>,
    /// The playlist the playlist bar edits.
    playlist: String,
    new_playlist: String,
    autostart: bool,
}

/// `None` handle when the daemon owns the renderer.
type ApplyResult = Result<(Option<WallpaperHandle>, String), String>;

/// The filter panel's unticked boxes; a wallpaper shows unless it is
/// excluded by type or rating, or every one of its tags is unticked.
#[derive(Default)]
struct Filters {
    hidden_types: HashSet<&'static str>,
    hidden_ratings: HashSet<&'static str>,
    hidden_tags: HashSet<String>,
}

impl Filters {
    fn matches(&self, w: &Wallpaper) -> bool {
        let tags = w.project.tags.as_deref().unwrap_or_default();
        !self.hidden_types.contains(type_name(w.wallpaper_type()))
            && !self.hidden_ratings.contains(age_rating(w))
            && (tags.is_empty() || tags.iter().any(|t| !self.hidden_tags.contains(t)))
    }

    fn is_active(&self) -> bool {
        !(self.hidden_types.is_empty()
            && self.hidden_ratings.is_empty()
            && self.hidden_tags.is_empty())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SortKey {
    Name,
    Type,
    Installed,
}

impl SortKey {
    const ALL: [SortKey; 3] = [SortKey::Name, SortKey::Type, SortKey::Installed];

    fn label(self) -> &'static str {
        match self {
            SortKey::Name => "Name",
            SortKey::Type => "Type",
            SortKey::Installed => "Date installed",
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
        let saved = crate::settings::WpSettings::load();
        let count = wallpapers.len();
        let installed = wallpapers
            .iter()
            .map(|w| {
                std::fs::metadata(&w.path)
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH)
            })
            .collect();
        let mut all_tags: Vec<String> = wallpapers
            .iter()
            .flat_map(|w| w.project.tags.clone().unwrap_or_default())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        all_tags.sort();

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

        let playlist = saved
            .active_playlist
            .clone()
            .or_else(|| saved.playlists.keys().next().cloned())
            .unwrap_or_else(|| "Default".into());

        let mut app = Self {
            wallpapers,
            installed,
            all_tags,
            filtered: Vec::new(),
            search: String::new(),
            filters: Filters::default(),
            show_filter: false,
            sort: SortKey::Name,
            descending: false,
            selected: None,
            thumbnails: HashMap::new(),
            thumb_queue: Vec::new(),
            renderer: None,
            active_title: None,
            #[cfg(target_os = "linux")]
            daemon: DaemonLink::new(),
            #[cfg(target_os = "linux")]
            initial_status,
            initial: None,
            initial_saved: saved.clone(),
            close_after_apply: false,
            status,
            settings: Arc::new(Mutex::new(RenderSettings {
                fps: saved.fps.unwrap_or(crate::content::settings::DEFAULT_FPS),
                scaling: saved.scaling.unwrap_or_default(),
                ..Default::default()
            })),
            pending_apply: None,
            audio_devices: None,
            audio_choice: 0,
            saved,
            prop_editor: None,
            show_settings: false,
            settings_tab: config::SettingsTab::Performance,
            show_displays: false,
            target_screen: None,
            outputs: None,
            playlist,
            new_playlist: String::new(),
            autostart: crate::settings::autostart_enabled(),
        };
        app.rebuild_filter();
        app
    }

    // ── Filter helpers ────────────────────────────────────────────────────────

    fn rebuild_filter(&mut self) {
        let q = self.search.to_lowercase();
        let mut shown: Vec<usize> = (0..self.wallpapers.len())
            .filter(|&i| {
                let w = &self.wallpapers[i];
                let text_ok = q.is_empty()
                    || w.title().to_lowercase().contains(&q)
                    || w.project
                        .tags
                        .iter()
                        .flatten()
                        .any(|t| t.to_lowercase() == q);
                text_ok && self.filters.matches(w)
            })
            .collect();
        let title = |i: &usize| self.wallpapers[*i].title().to_lowercase();
        match self.sort {
            SortKey::Name => shown.sort_by_cached_key(title),
            SortKey::Type => shown.sort_by_cached_key(|i| {
                (type_name(self.wallpapers[*i].wallpaper_type()), title(i))
            }),
            // Newest first, like WE.
            SortKey::Installed => shown.sort_by_key(|&i| std::cmp::Reverse(self.installed[i])),
        }
        if self.descending {
            shown.reverse();
        }
        self.filtered = shown;
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
                // Square tiles, centre-cropped like WE's grid.
                let img = img.into_rgba8();
                let side = img.width().min(img.height());
                let square = image::imageops::crop_imm(
                    &img,
                    (img.width() - side) / 2,
                    (img.height() - side) / 2,
                    side,
                    side,
                )
                .to_image();
                let thumb = image::imageops::thumbnail(&square, 256, 256);
                let size = [thumb.width() as usize, thumb.height() as usize];
                let ci = egui::ColorImage::from_rgba_unmultiplied(size, &thumb.into_raw());
                let handle = ctx.load_texture(&key, ci, egui::TextureOptions::LINEAR);
                self.thumbnails.insert(key, handle);
            }
        }
    }

    // ── Wallpaper application ─────────────────────────────────────────────────

    /// Workshop item path for `id`, for the displays dialog and playlist.
    fn index_of(&self, id: &str) -> Option<usize> {
        self.wallpapers.iter().position(|w| w.workshop_id == id)
    }
}

// ── eframe::App ───────────────────────────────────────────────────────────────

impl eframe::App for WpApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_pending_apply(ctx);
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
        let skin = Skin::of(ctx);
        self.render_footer(ctx, skin);
        self.render_details_panel(ctx, skin);
        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(skin.bg)
                    .inner_margin(egui::Margin::same(10)),
            )
            .show(ctx, |ui| {
                egui::TopBottomPanel::top("browse_header")
                    .frame(egui::Frame::NONE.inner_margin(egui::Margin {
                        bottom: 6,
                        ..Default::default()
                    }))
                    .show_inside(ui, |ui| self.render_header(ui, skin));
                egui::TopBottomPanel::bottom("playlist_bar")
                    .frame(egui::Frame::NONE.inner_margin(egui::Margin {
                        top: 6,
                        ..Default::default()
                    }))
                    .show_inside(ui, |ui| self.render_playlist_bar(ui, skin));
                if self.show_filter {
                    egui::SidePanel::left("filters")
                        .exact_width(FILTER_W)
                        .frame(egui::Frame::NONE.inner_margin(egui::Margin {
                            right: 8,
                            ..Default::default()
                        }))
                        .show_inside(ui, |ui| self.render_filters(ui));
                }
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show_inside(ui, |ui| self.render_grid(ui, skin));
            });
        self.render_displays(ctx, skin);
        self.render_settings_window(ctx);
    }
}

// ── Panels ────────────────────────────────────────────────────────────────────

impl WpApp {
    /// Source tabs with Displays/Settings, then search, filter and sort.
    fn render_header(&mut self, ui: &mut egui::Ui, skin: Skin) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for (label, active) in [
                ("Installed", true),
                ("Discover", false),
                ("Workshop", false),
            ] {
                if source_tab(ui, label, active, skin).clicked() && !active {
                    open_external(WORKSHOP_URL);
                }
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                if ui.button("⚙ Settings").clicked() {
                    self.show_settings = !self.show_settings;
                    // `wp-engine config` may have changed it meanwhile.
                    self.saved = crate::settings::WpSettings::load();
                    self.autostart = crate::settings::autostart_enabled();
                }
                let display = self.target_screen.as_deref().unwrap_or("All displays");
                if ui
                    .button(format!("🖵 Displays: {display}"))
                    .on_hover_text("Choose the display clicked wallpapers go to")
                    .clicked()
                {
                    self.show_displays = true;
                }
            });
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let r = ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .desired_width(260.0)
                    .hint_text("🔍 Search"),
            );
            let mut changed = r.changed();
            let filter_btn = egui::Button::new("☰ Filter results")
                .selected(self.show_filter || self.filters.is_active());
            if ui.add(filter_btn).clicked() {
                self.show_filter = !self.show_filter;
            }
            if self.filtered.len() != self.wallpapers.len() {
                ui.label(
                    egui::RichText::new(format!(
                        "({} of {})",
                        self.filtered.len(),
                        self.wallpapers.len()
                    ))
                    .color(TEXT_MUTED),
                );
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let arrow = if self.descending { "⬇" } else { "⬆" };
                if ui.button(arrow).on_hover_text("Sort direction").clicked() {
                    self.descending = !self.descending;
                    changed = true;
                }
                egui::ComboBox::from_id_salt("sort")
                    .selected_text(self.sort.label())
                    .width(140.0)
                    .show_ui(ui, |ui| {
                        for key in SortKey::ALL {
                            changed |= ui
                                .selectable_value(&mut self.sort, key, key.label())
                                .changed();
                        }
                    });
            });
            if changed {
                self.rebuild_filter();
            }
        });
    }

    /// WE's "Filter results" panel: type, age rating and tags.
    fn render_filters(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        egui::ScrollArea::vertical().show(ui, |ui| {
            if ui.button("Reset Filters").clicked() {
                self.filters = Filters::default();
                changed = true;
            }
            ui.add_space(6.0);
            ui.strong("Type");
            for t in TYPES {
                changed |= filter_box(ui, &mut self.filters.hidden_types, t, t);
            }
            ui.add_space(6.0);
            ui.strong("Age rating");
            for r in AGE_RATINGS {
                changed |= filter_box(ui, &mut self.filters.hidden_ratings, r, r);
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.strong("Tags");
                if ui.small_button("All").clicked() {
                    self.filters.hidden_tags.clear();
                    changed = true;
                }
                if ui.small_button("None").clicked() {
                    self.filters.hidden_tags = self.all_tags.iter().cloned().collect();
                    changed = true;
                }
            });
            for tag in self.all_tags.clone() {
                changed |= filter_box(ui, &mut self.filters.hidden_tags, tag.clone(), &tag);
            }
        });
        if changed {
            self.rebuild_filter();
        }
    }

    fn render_grid(&mut self, ui: &mut egui::Ui, skin: Skin) {
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
                         or use \"Open Wallpaper\" below to show any file.",
                    )
                    .color(TEXT_MUTED),
                );
            });
            return;
        }
        if self.filtered.is_empty() {
            ui.vertical_centered(|ui| {
                ui.add_space(95.0);
                ui.label(
                    egui::RichText::new(
                        "No wallpapers found for your search.\nExpand or reset the filters.",
                    )
                    .size(18.0)
                    .color(TEXT_MUTED),
                );
            });
            return;
        }

        // As many columns as fit at up to `TILE`, stretched to fill the row.
        let width = ui.available_width();
        let cols = (((width + TILE_GAP) / (TILE + TILE_GAP)).ceil() as usize).max(1);
        let tile = (width - TILE_GAP * (cols - 1) as f32) / cols as f32;
        let rows = self.filtered.len().div_ceil(cols);
        let filtered = self.filtered.clone();
        egui::ScrollArea::vertical().auto_shrink(false).show_rows(
            ui,
            tile + TILE_GAP,
            rows,
            |ui, range| {
                for row in range {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = TILE_GAP;
                        for &idx in filtered.iter().skip(row * cols).take(cols) {
                            self.grid_tile(ui, idx, tile, skin);
                        }
                    });
                    ui.add_space(TILE_GAP);
                }
            },
        );
    }

    fn grid_tile(&mut self, ui: &mut egui::Ui, idx: usize, size: f32, skin: Skin) {
        let id = self.wallpapers[idx].workshop_id.clone();
        let thumb = self.thumbnails.get(&id).cloned();
        if thumb.is_none() {
            self.enqueue_thumb(idx);
        }
        let (rect, resp) = ui.allocate_exact_size(vec2(size, size), Sense::click());
        if ui.is_rect_visible(rect) {
            draw_tile(
                ui,
                rect,
                &resp,
                &self.wallpapers[idx],
                thumb.as_ref(),
                self.selected == Some(idx),
                skin,
            );
        }
        let in_playlist = self.playlist_items().contains(&id);
        resp.context_menu(|ui| {
            let label = if in_playlist {
                "Remove from playlist"
            } else {
                "Add to playlist"
            };
            if ui.button(label).clicked() {
                self.toggle_in_playlist(&id);
                ui.close();
            }
            if id.bytes().all(|b| b.is_ascii_digit()) && ui.button("Open Workshop page").clicked() {
                open_external(&workshop_item_url(&id));
                ui.close();
            }
            if ui.button("Open folder").clicked() {
                open_external(&self.wallpapers[idx].path.to_string_lossy());
                ui.close();
            }
        });
        if resp.clicked() {
            self.selected = Some(idx);
            self.apply_wallpaper(idx);
        }
    }

    /// The right-hand column: preview, info, actions and properties.
    fn render_details_panel(&mut self, ctx: &egui::Context, skin: Skin) {
        egui::SidePanel::right("details")
            .exact_width(DETAILS_W)
            .resizable(false)
            .frame(
                egui::Frame::default()
                    .fill(skin.panel)
                    .inner_margin(egui::Margin::same(14)),
            )
            .show(ctx, |ui| match self.selected {
                Some(idx) => self.render_details(ui, idx, skin),
                None => {
                    ui.vertical_centered(|ui| {
                        ui.add_space(100.0);
                        ui.label(
                            egui::RichText::new("Please select a wallpaper")
                                .size(16.0)
                                .color(TEXT_MUTED),
                        );
                    });
                }
            });
    }

    fn render_details(&mut self, ui: &mut egui::Ui, idx: usize, skin: Skin) {
        let w = &self.wallpapers[idx];
        let title = w.title().to_string();
        let wtype = w.wallpaper_type().clone();
        let tags = w.project.tags.clone().unwrap_or_default();
        let rating = age_rating(w);
        let id = w.workshop_id.clone();
        let path = w.path.clone();
        let blocker = can_apply(w).err();
        let thumb = self.thumbnails.get(&id).cloned();
        if thumb.is_none() {
            self.enqueue_thumb(idx);
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Preview, square like the thumbnails it comes from.
            let side = ui.available_width().min(260.0);
            ui.vertical_centered(|ui| {
                let (rect, _) = ui.allocate_exact_size(vec2(side, side), Sense::hover());
                let p = ui.painter_at(rect);
                p.rect_filled(rect, CornerRadius::same(4), skin.tile);
                match &thumb {
                    Some(tex) => {
                        p.image(tex.id(), rect, full_uv(), Color32::WHITE);
                    }
                    None => {
                        p.text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            "No preview",
                            FontId::proportional(13.0),
                            TEXT_MUTED,
                        );
                    }
                }
            });
            ui.add_space(8.0);
            ui.label(egui::RichText::new(&title).size(18.0).color(skin.text));
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                pill(ui, type_name(&wtype), type_color(&wtype), Color32::WHITE);
                pill(ui, rating, skin.pill, skin.text);
                for tag in &tags {
                    pill(ui, tag, skin.pill, skin.text);
                }
            });
            if let Some(reason) = blocker {
                ui.add_space(4.0);
                ui.label(egui::RichText::new(reason).color(DANGER));
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if id.bytes().all(|b| b.is_ascii_digit()) && ui.button("🌐 Workshop page").clicked()
                {
                    open_external(&workshop_item_url(&id));
                }
                if ui.button("📂 Open folder").clicked() {
                    open_external(&path.to_string_lossy());
                }
                let in_playlist = self.playlist_items().contains(&id);
                let label = if in_playlist {
                    "➖ Playlist"
                } else {
                    "➕ Playlist"
                };
                if ui.button(label).clicked() {
                    self.toggle_in_playlist(&id);
                }
            });
            ui.add_space(10.0);
            self.render_properties(ui, idx);
        });
    }

    /// Under the grid: the playlist for the chosen display, as in WE.
    fn render_playlist_bar(&mut self, ui: &mut egui::Ui, skin: Skin) {
        ui.separator();
        ui.horizontal(|ui| {
            ui.strong("Playlist");
            ui.menu_button(format!("📋 {}", self.playlist), |ui| {
                let names: Vec<String> = self.saved.playlists.keys().cloned().collect();
                for name in names {
                    if ui.selectable_label(name == self.playlist, &name).clicked() {
                        self.playlist = name;
                        ui.close();
                    }
                }
                ui.separator();
                ui.add(
                    egui::TextEdit::singleline(&mut self.new_playlist)
                        .hint_text("New playlist…")
                        .desired_width(140.0),
                );
                let name = self.new_playlist.trim().to_string();
                if ui
                    .add_enabled(!name.is_empty(), egui::Button::new("Save current as"))
                    .clicked()
                {
                    let copy = self
                        .saved
                        .playlists
                        .get(&self.playlist)
                        .cloned()
                        .unwrap_or_else(config::empty_playlist);
                    self.saved.playlists.insert(name.clone(), copy);
                    self.playlist = name;
                    self.new_playlist.clear();
                    self.save_settings();
                    ui.close();
                }
            });
            let has_items = !self.playlist_items().is_empty();
            ui.add_enabled_ui(has_items, |ui| {
                ui.menu_button("Configure", |ui| self.playlist_config(ui));
                #[cfg(target_os = "linux")]
                {
                    use crate::daemon::Request;
                    if ui.button("▶ Play").clicked() {
                        self.daemon_request(Request::PlayPlaylist {
                            name: self.playlist.clone(),
                        });
                        self.active_title = Some(format!("playlist '{}'", self.playlist));
                    }
                    if ui.button("⏭").on_hover_text("Next").clicked() {
                        self.daemon_request(Request::NextItem);
                    }
                    if ui.button("■").on_hover_text("Stop rotating").clicked() {
                        self.daemon_request(Request::StopPlaylist);
                    }
                }
                if ui.button("Clear").clicked() {
                    if let Some(p) = self.saved.playlists.get_mut(&self.playlist) {
                        p.items.clear();
                        p.times.clear();
                    }
                    self.save_settings();
                }
            });
            if let Some(idx) = self.selected {
                let id = self.wallpapers[idx].workshop_id.clone();
                let label = if self.playlist_items().contains(&id) {
                    "Remove wallpaper"
                } else {
                    "Add wallpaper"
                };
                if ui.button(label).clicked() {
                    self.toggle_in_playlist(&id);
                }
            }
        });
        let items = self.playlist_items();
        if items.is_empty() {
            ui.label(
                egui::RichText::new(
                    "Right-click a wallpaper or use \"Add wallpaper\" to build a playlist.",
                )
                .size(11.0)
                .color(TEXT_MUTED),
            );
            return;
        }
        egui::ScrollArea::horizontal().show(ui, |ui| {
            ui.horizontal(|ui| {
                for id in items {
                    let idx = self.index_of(&id);
                    if let Some(idx) = idx {
                        self.enqueue_thumb(idx);
                    }
                    let (rect, resp) = ui
                        .allocate_exact_size(vec2(PLAYLIST_THUMB, PLAYLIST_THUMB), Sense::click());
                    let p = ui.painter_at(rect);
                    p.rect_filled(rect, CornerRadius::same(3), skin.tile);
                    if let Some(tex) = self.thumbnails.get(&id) {
                        p.image(tex.id(), rect, full_uv(), Color32::WHITE);
                    }
                    if idx.is_some() && idx == self.selected {
                        p.rect_stroke(
                            rect,
                            CornerRadius::same(3),
                            Stroke::new(2.0, skin.accent),
                            egui::StrokeKind::Inside,
                        );
                    }
                    let resp = resp.on_hover_text(self.title_of(&id));
                    if resp.clicked() {
                        self.selected = idx;
                    }
                }
            });
        });
    }

    /// Open file/folder on the left, what's showing in the middle, OK and
    /// Cancel on the right.
    fn render_footer(&mut self, ctx: &egui::Context, skin: Skin) {
        egui::TopBottomPanel::bottom("footer")
            .frame(
                egui::Frame::default()
                    .fill(skin.panel)
                    .inner_margin(egui::Margin::symmetric(12, 8)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.menu_button("📂 Open Wallpaper", |ui| {
                        if ui
                            .button("Open offline wallpaper (scene folder)…")
                            .clicked()
                        {
                            ui.close();
                            if let Some(p) = config::pick_path(true) {
                                self.open_path(PathBuf::from(p));
                            }
                        }
                        if ui.button("Open video, image or web page…").clicked() {
                            ui.close();
                            if let Some(p) = config::pick_path(false) {
                                self.open_path(PathBuf::from(p));
                            }
                        }
                    });
                    ui.separator();
                    if let Some(title) = self.active_title.clone() {
                        ui.label(egui::RichText::new(format!("▶ {title}")).color(skin.accent));
                        #[cfg(target_os = "linux")]
                        if ui.small_button("■ Remove").clicked() {
                            self.clear_wallpaper();
                        }
                    }
                    let color = if self.status.error {
                        DANGER
                    } else {
                        TEXT_MUTED
                    };
                    ui.label(egui::RichText::new(&self.status.text).color(color));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.button("Cancel").clicked() {
                            self.cancel(ctx);
                        }
                        let ok = egui::Button::new(
                            egui::RichText::new("   OK   ").color(Color32::WHITE),
                        )
                        .fill(skin.accent);
                        if ui.add(ok).clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                });
            });
    }

    fn open_path(&mut self, path: PathBuf) {
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        self.apply_path(path, name);
    }

    /// WE's "Choose Display": where clicked wallpapers go, the multi-monitor
    /// layout, and what each display shows.
    fn render_displays(&mut self, ctx: &egui::Context, skin: Skin) {
        if !self.show_displays {
            return;
        }
        let outputs = self
            .outputs
            .get_or_insert_with(crate::platform::display::output_names)
            .clone();
        let modal = egui::Modal::new(egui::Id::new("displays")).show(ctx, |ui| {
            ui.set_width(560.0);
            ui.heading("Choose Display");
            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                let all = self.saved.default_background.clone();
                self.display_box(ui, None, "All displays", all, skin);
                for out in &outputs {
                    let shown = self
                        .saved
                        .screens
                        .get(out)
                        .or(self.saved.default_background.as_ref())
                        .cloned();
                    self.display_box(ui, Some(out.clone()), out, shown, skin);
                }
            });
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label("Several displays:");
                let mut layout = self.saved.layout.unwrap_or_default();
                let before = layout;
                for (l, text) in [
                    (crate::content::ScreenLayout::PerScreen, "Separate"),
                    (crate::content::ScreenLayout::Span, "Stretch"),
                    (crate::content::ScreenLayout::Clone, "Clone"),
                ] {
                    ui.selectable_value(&mut layout, l, text);
                }
                if layout != before {
                    self.saved.layout = Some(layout);
                    self.save_settings();
                    #[cfg(target_os = "linux")]
                    if let Some(id) = self.saved.default_background.clone() {
                        let fallback = self
                            .index_of(&id)
                            .map(|i| self.wallpapers[i].path.clone())
                            .unwrap_or_default();
                        self.apply_saved(self.title_of(&id), fallback);
                    }
                }
            });
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui
                    .add(egui::Button::new(
                        egui::RichText::new("Remove Wallpaper").color(DANGER),
                    ))
                    .clicked()
                {
                    match self.target_screen.clone() {
                        Some(screen) => {
                            self.saved.screens.remove(&screen);
                        }
                        None => {
                            self.saved.default_background = None;
                            self.saved.screens.clear();
                            #[cfg(target_os = "linux")]
                            self.clear_wallpaper();
                        }
                    }
                    self.save_settings();
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.button("Close").clicked() {
                        ui.close();
                    }
                });
            });
        });
        if modal.should_close() {
            self.show_displays = false;
        }
    }

    /// One display in the "Choose Display" dialog; click to target it.
    fn display_box(
        &mut self,
        ui: &mut egui::Ui,
        screen: Option<String>,
        name: &str,
        shown: Option<String>,
        skin: Skin,
    ) {
        let (rect, resp) = ui.allocate_exact_size(vec2(170.0, 120.0), Sense::click());
        let p = ui.painter_at(rect);
        let screen_rect = Rect::from_min_size(rect.min, vec2(170.0, 96.0));
        p.rect_filled(screen_rect, CornerRadius::same(4), skin.tile);
        if let Some(idx) = shown.as_deref().and_then(|id| self.index_of(id)) {
            self.enqueue_thumb(idx);
            if let Some(tex) = self.thumbnails.get(&self.wallpapers[idx].workshop_id) {
                // The square thumbnail, cropped to the screen's 16:9.
                let uv = Rect::from_min_max(pos2(0.0, 0.22), pos2(1.0, 0.78));
                p.image(tex.id(), screen_rect, uv, Color32::WHITE);
            }
        }
        let selected = self.target_screen == screen;
        p.rect_stroke(
            screen_rect,
            CornerRadius::same(4),
            Stroke::new(
                if selected { 3.0 } else { 1.0 },
                if selected { skin.accent } else { TEXT_MUTED },
            ),
            egui::StrokeKind::Inside,
        );
        p.text(
            pos2(rect.center().x, rect.bottom() - 10.0),
            egui::Align2::CENTER_CENTER,
            name,
            FontId::proportional(13.0),
            skin.text,
        );
        if resp.clicked() {
            self.target_screen = screen;
        }
    }

    // ── Playlist helpers ──────────────────────────────────────────────────────

    fn playlist_items(&self) -> Vec<String> {
        self.saved
            .playlists
            .get(&self.playlist)
            .map(|p| p.items.clone())
            .unwrap_or_default()
    }

    fn toggle_in_playlist(&mut self, id: &str) {
        let p = self
            .saved
            .playlists
            .entry(self.playlist.clone())
            .or_insert_with(config::empty_playlist);
        match p.items.iter().position(|i| i == id) {
            Some(i) => {
                p.items.remove(i);
                if p.times.len() > i {
                    p.times.remove(i);
                }
            }
            None => p.items.push(id.to_string()),
        }
        self.save_settings();
    }
}

// ── Drawing ───────────────────────────────────────────────────────────────────

fn source_tab(ui: &mut egui::Ui, label: &str, active: bool, skin: Skin) -> egui::Response {
    let (fill, text) = if active {
        (skin.accent, Color32::WHITE)
    } else {
        (skin.panel, skin.text)
    };
    ui.add(
        egui::Button::new(egui::RichText::new(label).size(15.0).color(text))
            .fill(fill)
            .stroke(Stroke::new(1.0, skin.accent))
            .corner_radius(CornerRadius {
                nw: 3,
                ne: 3,
                sw: 0,
                se: 0,
            })
            .min_size(vec2(96.0, 28.0)),
    )
}

/// A ticked-by-default filter checkbox backed by a set of unticked keys.
fn filter_box<K: std::hash::Hash + Eq>(
    ui: &mut egui::Ui,
    hidden: &mut HashSet<K>,
    key: K,
    text: &str,
) -> bool {
    let mut on = !hidden.contains(&key);
    if !ui.checkbox(&mut on, text).changed() {
        return false;
    }
    if on {
        hidden.remove(&key);
    } else {
        hidden.insert(key);
    }
    true
}

fn pill(ui: &mut egui::Ui, text: &str, fill: Color32, color: Color32) {
    egui::Frame::default()
        .fill(fill)
        .corner_radius(CornerRadius::same(12))
        .inner_margin(egui::Margin::symmetric(8, 2))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).size(12.0).color(color));
        });
}

fn draw_tile(
    ui: &egui::Ui,
    rect: Rect,
    resp: &egui::Response,
    w: &Wallpaper,
    thumb: Option<&egui::TextureHandle>,
    selected: bool,
    skin: Skin,
) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, CornerRadius::same(2), skin.tile);
    match thumb {
        Some(tex) => {
            painter.image(tex.id(), rect, full_uv(), Color32::WHITE);
        }
        None => {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "⏳",
                FontId::proportional(22.0),
                TEXT_MUTED,
            );
        }
    }
    if resp.hovered() {
        painter.rect_filled(rect, CornerRadius::same(2), Color32::from_white_alpha(24));
    }

    // Title over a dark band along the bottom, as in WE's grid.
    let band = Rect::from_min_max(pos2(rect.min.x, rect.max.y - 24.0), rect.max);
    painter.rect_filled(band, CornerRadius::ZERO, Color32::from_black_alpha(170));
    let mut job = egui::text::LayoutJob::simple_singleline(
        w.title().to_string(),
        FontId::proportional(12.0),
        Color32::WHITE,
    );
    job.wrap = egui::text::TextWrapping::truncate_at_width(rect.width() - 10.0);
    painter.galley(
        pos2(band.min.x + 5.0, band.min.y + 5.0),
        painter.layout_job(job),
        Color32::WHITE,
    );

    // Type marker in the top corner for everything that isn't a scene.
    if !matches!(w.wallpaper_type(), WallpaperType::Scene) {
        let label = type_name(w.wallpaper_type()).to_uppercase();
        let galley = painter.layout_no_wrap(label, FontId::proportional(9.0), Color32::WHITE);
        let size = galley.size() + vec2(8.0, 4.0);
        let badge = Rect::from_min_size(pos2(rect.max.x - size.x - 4.0, rect.min.y + 4.0), size);
        painter.rect_filled(badge, CornerRadius::same(3), type_color(w.wallpaper_type()));
        painter.galley(badge.min + vec2(4.0, 2.0), galley, Color32::WHITE);
    }

    if selected {
        painter.rect_stroke(
            rect,
            CornerRadius::same(2),
            Stroke::new(3.0, skin.accent),
            egui::StrokeKind::Inside,
        );
    }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

fn full_uv() -> Rect {
    Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0))
}

fn age_rating(w: &Wallpaper) -> &'static str {
    match w.project.contentrating.as_deref() {
        Some("Mature") => "Mature",
        Some("Questionable") => "Questionable",
        _ => "Everyone",
    }
}

fn type_name(t: &WallpaperType) -> &'static str {
    match t {
        WallpaperType::Video => "Video",
        WallpaperType::Scene => "Scene",
        WallpaperType::Web => "Web",
        WallpaperType::Application => "Application",
        WallpaperType::Preset => "Preset",
        WallpaperType::Unknown => "Unknown",
    }
}

fn type_color(t: &WallpaperType) -> Color32 {
    match t {
        WallpaperType::Video => Color32::from_rgb(55, 95, 200),
        WallpaperType::Scene => Color32::from_rgb(130, 65, 200),
        WallpaperType::Web => Color32::from_rgb(50, 160, 80),
        WallpaperType::Application => Color32::from_rgb(180, 95, 50),
        WallpaperType::Preset => Color32::from_rgb(190, 150, 40),
        WallpaperType::Unknown => Color32::from_gray(75),
    }
}

fn workshop_item_url(id: &str) -> String {
    format!("https://steamcommunity.com/sharedfiles/filedetails/?id={id}")
}

/// Open a URL or folder with the desktop's default handler.
fn open_external(target: &str) {
    #[cfg(target_os = "macos")]
    let opener = "open";
    #[cfg(not(target_os = "macos"))]
    let opener = "xdg-open";
    if let Err(e) = std::process::Command::new(opener).arg(target).spawn() {
        tracing::warn!(target: "app", "{opener} {target}: {e}");
    }
}

fn truncate(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Wallpaper Engine's light and dark skins, following the system theme.
fn setup_theme(ctx: &egui::Context) {
    for (theme, skin, mut v) in [
        (egui::Theme::Dark, Skin::DARK, egui::Visuals::dark()),
        (egui::Theme::Light, Skin::LIGHT, egui::Visuals::light()),
    ] {
        v.panel_fill = skin.bg;
        v.window_fill = skin.panel;
        v.selection.bg_fill = skin.accent;
        v.hyperlink_color = skin.accent;
        v.widgets.active.bg_fill = skin.accent;
        v.widgets.hovered.bg_stroke = Stroke::new(1.0, skin.accent);
        v.override_text_color = Some(skin.text);
        ctx.set_visuals_of(theme, v);
    }
    ctx.set_theme(egui::ThemePreference::System);

    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = vec2(6.0, 4.0);
        style.spacing.button_padding = vec2(8.0, 4.0);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wallpaper(json: &str) -> Wallpaper {
        Wallpaper {
            workshop_id: "1".into(),
            path: PathBuf::new(),
            project: serde_json::from_str(json).unwrap(),
        }
    }

    #[test]
    fn filters_hide_by_type_rating_and_all_tags() {
        let w = wallpaper(r#"{"type":"video","contentrating":"Mature","tags":["Anime","Game"]}"#);
        let mut f = Filters::default();
        assert!(f.matches(&w));
        f.hidden_tags.insert("Anime".into());
        assert!(f.matches(&w), "one tag still ticked");
        f.hidden_tags.insert("Game".into());
        assert!(!f.matches(&w), "every tag unticked");
        f.hidden_tags.clear();
        f.hidden_ratings.insert("Mature");
        assert!(!f.matches(&w));
        f.hidden_ratings.clear();
        f.hidden_types.insert("Video");
        assert!(!f.matches(&w));
        assert!(f.matches(&wallpaper(r#"{"type":"scene"}"#)));
    }
}
