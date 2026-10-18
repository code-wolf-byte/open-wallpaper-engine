//! The Choose Display dialog: which display clicked wallpapers go to, the
//! multi-display layout, and what each display shows.

use super::*;

impl WpApp {
    /// WE's "Choose Display": where clicked wallpapers go, the multi-monitor
    /// layout, and what each display shows.
    pub(super) fn render_displays(&mut self, ctx: &egui::Context, skin: Skin) {
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
    pub(super) fn display_box(
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
}
