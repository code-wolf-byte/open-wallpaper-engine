//! The footer: Open Wallpaper, what is showing, and OK / Cancel.

use super::*;

impl WpApp {
    /// Open file/folder on the left, what's showing in the middle, OK and
    /// Cancel on the right.
    pub(super) fn render_footer(&mut self, ctx: &egui::Context, skin: Skin) {
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

    pub(super) fn open_path(&mut self, path: PathBuf) {
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        self.apply_path(path, name);
    }
}
