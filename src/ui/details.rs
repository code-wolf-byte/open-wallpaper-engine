//! The details panel: the selected wallpaper's preview, information,
//! actions and properties.

use super::*;

impl WpApp {
    /// The right-hand column: preview, info, actions and properties.
    pub(super) fn render_details_panel(&mut self, ctx: &egui::Context, skin: Skin) {
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

    pub(super) fn render_details(&mut self, ui: &mut egui::Ui, idx: usize, skin: Skin) {
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
}

pub(super) fn pill(ui: &mut egui::Ui, text: &str, fill: Color32, color: Color32) {
    egui::Frame::default()
        .fill(fill)
        .corner_radius(CornerRadius::same(12))
        .inner_margin(egui::Margin::symmetric(8, 2))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).size(12.0).color(color));
        });
}
