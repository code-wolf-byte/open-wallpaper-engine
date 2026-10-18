//! The playlist bar under the grid: choosing, saving, playing and editing
//! the current playlist.

use super::*;

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
