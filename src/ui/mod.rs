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

mod thumbnails;

mod browse;

mod details;

mod playlist;

mod displays;

mod footer;

#[cfg(not(target_os = "linux"))]
use crate::{content::WallpaperContent, platform::display::detect_platform};
use egui::{pos2, vec2, Align, Color32, CornerRadius, FontId, Layout, Rect, Sense, Stroke};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

mod config;
mod properties;

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
    prop_editor: Option<properties::PropertyEditor>,
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

    // ── Thumbnail loading ─────────────────────────────────────────────────────

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
    // ── Playlist helpers ──────────────────────────────────────────────────────
}

// ── Drawing ───────────────────────────────────────────────────────────────────

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
