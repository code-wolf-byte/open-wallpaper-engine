//! The playlist bar under the grid: choosing, saving, playing and editing
//! the current playlist.

use super::*;
use crate::settings::Playlist;

impl WpApp {
    /// Under the grid: the playlist for the chosen display, as in WE.
    pub(super) fn render_playlist_bar(&mut self, ui: &mut egui::Ui, skin: Skin) {
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
                        .unwrap_or_else(empty_playlist);
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

    pub(super) fn playlist_items(&self) -> Vec<String> {
        self.saved
            .playlists
            .get(&self.playlist)
            .map(|p| p.items.clone())
            .unwrap_or_default()
    }

    pub(super) fn toggle_in_playlist(&mut self, id: &str) {
        let p = self
            .saved
            .playlists
            .entry(self.playlist.clone())
            .or_insert_with(empty_playlist);
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

impl WpApp {
    /// The playlist bar's Configure popup: rotation, order, timing,
    /// whether it plays at login, and deleting it.
    pub(super) fn playlist_config(&mut self, ui: &mut egui::Ui) {
        let name = self.playlist.clone();
        let Some(p) = self.saved.playlists.get(&name) else {
            return;
        };
        let titles: Vec<String> = p.items.iter().map(|id| self.title_of(id)).collect();
        let mut changed = playlist_editor(
            ui,
            &name,
            self.saved.playlists.get_mut(&name).unwrap(),
            &titles,
        );
        ui.separator();
        let mut at_login = self.saved.active_playlist.as_deref() == Some(name.as_str());
        if ui
            .checkbox(&mut at_login, "Play at login")
            .on_hover_text("`wp-engine run` plays this playlist")
            .changed()
        {
            self.saved.active_playlist = at_login.then(|| name.clone());
            changed = true;
        }
        if ui
            .button(egui::RichText::new("🗑 Delete playlist").color(DANGER))
            .clicked()
        {
            self.saved.playlists.remove(&name);
            if self.saved.active_playlist.as_deref() == Some(name.as_str()) {
                self.saved.active_playlist = None;
            }
            changed = true;
            ui.close();
        }
        if changed {
            self.save_settings();
        }
    }
}

/// A new playlist with the saved format's defaults.
pub(super) fn empty_playlist() -> Playlist {
    serde_json::from_str(r#"{"items":[]}"#).expect("playlist defaults")
}

const PLAYLIST_MODES: [(&str, &str); 4] = [
    ("timer", "Every N minutes"),
    ("daytime", "At times of day"),
    ("dayofweek", "By day of week"),
    ("login", "Next one each login"),
];

/// Mode, order, timing and items of one playlist. Returns whether it changed.
fn playlist_editor(ui: &mut egui::Ui, name: &str, p: &mut Playlist, titles: &[String]) -> bool {
    let before = p.clone();
    egui::ComboBox::from_id_salt(("mode", name))
        .selected_text(
            PLAYLIST_MODES
                .iter()
                .find(|(m, _)| *m == p.mode)
                .map_or(p.mode.as_str(), |(_, l)| l),
        )
        .show_ui(ui, |ui| {
            for (m, l) in PLAYLIST_MODES {
                ui.selectable_value(&mut p.mode, m.to_string(), l);
            }
        });
    ui.horizontal(|ui| {
        ui.selectable_value(&mut p.order, "sequential".into(), "In order");
        ui.selectable_value(&mut p.order, "random".into(), "Shuffle");
        if p.mode == "timer" {
            ui.add(
                egui::DragValue::new(&mut p.delay_minutes)
                    .range(1..=10_080)
                    .suffix(" min"),
            );
        }
    });
    let daytime = p.mode == "daytime";
    if daytime {
        p.times.resize(p.items.len(), String::new());
    }
    let mut remove = None;
    let mut raise = None;
    for (i, title) in titles.iter().enumerate() {
        ui.horizontal(|ui| {
            if ui.small_button("✖").clicked() {
                remove = Some(i);
            }
            if i > 0 && ui.small_button("⬆").clicked() {
                raise = Some(i);
            }
            if daytime {
                let t = &mut p.times[i];
                let bad = !t.is_empty() && crate::schedule::parse_time_of_day(t).is_none();
                ui.add(
                    egui::TextEdit::singleline(t)
                        .desired_width(48.0)
                        .hint_text("HH:MM")
                        .text_color_opt(bad.then_some(egui::Color32::LIGHT_RED)),
                );
            } else if p.mode == "dayofweek" {
                const DAYS: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
                ui.label(DAYS[i % 7]);
            }
            ui.label(super::truncate(title, 48)).on_hover_text(title);
        });
    }
    if let Some(i) = raise {
        p.items.swap(i, i - 1);
        if p.times.len() > i {
            p.times.swap(i, i - 1);
        }
    }
    if let Some(i) = remove {
        p.items.remove(i);
        if p.times.len() > i {
            p.times.remove(i);
        }
    }
    *p != before
}
