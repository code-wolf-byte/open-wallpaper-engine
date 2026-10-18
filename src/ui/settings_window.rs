//! The settings window: playback rules, frame rate, quality, scaling, audio
//! and autostart. Everything here edits `settings.json`, the same file
//! `wp-engine config` writes.

use super::{StatusMsg, WpApp, TEXT_MUTED};
use crate::content::ScaleMode;
use crate::platform::RenderQuality;
use crate::services::playback_gate::PlaybackAction;
use crate::settings::WpSettings;

/// The settings window's tabs, after WE's own settings view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum SettingsTab {
    Performance,
    General,
    About,
}

const PLAYBACK_CONDITIONS: [(&str, &str); 6] = [
    ("focus", "Another app is focused"),
    ("maximized", "Another app is maximized"),
    ("fullscreen", "Another app is fullscreen"),
    ("audio", "Another app is playing audio"),
    ("battery", "On battery"),
    ("lock", "Session locked"),
];

const SCALE_MODES: [ScaleMode; 4] = [
    ScaleMode::Fill,
    ScaleMode::Fit,
    ScaleMode::Stretch,
    ScaleMode::Center,
];

impl WpApp {
    // ── Detail panel ─────────────────────────────────────────────────────────

    // ── Settings window ──────────────────────────────────────────────────────

    pub(super) fn render_settings_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_settings;
        egui::Window::new("Settings")
            .open(&mut open)
            .default_width(480.0)
            .vscroll(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    for (tab, text) in [
                        (SettingsTab::Performance, "Performance"),
                        (SettingsTab::General, "General"),
                        (SettingsTab::About, "About"),
                    ] {
                        ui.selectable_value(&mut self.settings_tab, tab, text);
                    }
                });
                ui.separator();
                match self.settings_tab {
                    SettingsTab::Performance => {
                        self.playback_settings(ui);
                        ui.separator();
                        self.quality_settings(ui);
                    }
                    SettingsTab::General => self.general_settings(ui),
                    SettingsTab::About => about(ui),
                }
            });
        self.show_settings = open;
    }

    fn quality_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Quality");
        let mut quality_changed = false;
        {
            let mut s = self.settings.lock().unwrap();
            egui::ComboBox::from_label("Render quality")
                .selected_text(s.quality.label())
                .show_ui(ui, |ui| {
                    for q in RenderQuality::ALL {
                        quality_changed |=
                            ui.selectable_value(&mut s.quality, q, q.label()).changed();
                    }
                });
            #[cfg(target_os = "linux")]
            if quality_changed {
                self.daemon.send(crate::daemon::Request::SetQuality {
                    quality: s.quality.label().to_string(),
                });
            }
        }
        let _ = quality_changed;
        let mut fps = self
            .saved
            .fps
            .unwrap_or(crate::content::settings::DEFAULT_FPS);
        let r = ui.add(egui::Slider::new(&mut fps, 1..=240).text("Frame-rate cap"));
        if r.changed() {
            self.saved.fps = Some(fps);
            self.settings.lock().unwrap().fps = fps;
            #[cfg(target_os = "linux")]
            self.daemon.send(crate::daemon::Request::SetFps { fps });
        }
        if r.drag_stopped() || (r.changed() && !r.dragged()) {
            self.save_settings();
        }
    }

    fn general_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("General");
        let mut autostart = self.autostart;
        if ui
            .checkbox(&mut autostart, "Start with my session")
            .on_hover_text("Runs `wp-engine run` at login (XDG autostart)")
            .changed()
        {
            match crate::settings::set_autostart(autostart) {
                Ok(()) => self.autostart = autostart,
                Err(e) => self.status = StatusMsg::err(format!("Autostart: {e:#}")),
            }
        }
        if ui
            .button("▶ Apply saved setup")
            .on_hover_text(
                "What `wp-engine run` shows: the login playlist, else the saved displays",
            )
            .clicked()
        {
            self.run_saved_setup();
        }
        ui.add_space(6.0);
        ui.heading("Audio");
        {
            let mut s = self.settings.lock().unwrap();
            let volume_changed = ui
                .add(
                    egui::Slider::new(&mut s.volume, 0.0..=1.0)
                        .text("Volume")
                        .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
                )
                .changed();
            #[cfg(target_os = "linux")]
            if volume_changed {
                self.daemon
                    .send(crate::daemon::Request::SetVolume { volume: s.volume });
            }
            let _ = volume_changed;
        }
        let devices = self
            .audio_devices
            .get_or_insert_with(crate::services::audio_devices::list_audio_outputs);
        let current = devices
            .get(self.audio_choice)
            .map(|d| d.label.clone())
            .unwrap_or_else(|| "Automatic".to_string());
        let mut changed = None;
        egui::ComboBox::from_label("Audio-reactive wallpapers listen to")
            .selected_text(current)
            .width(240.0)
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
            crate::services::audio_devices::set_preferred_device(device);
            self.status = StatusMsg::ok("Audio source set — re-apply the wallpaper to use it");
        }
        ui.add_space(6.0);
        ui.heading("Display");
        let mut scaling = self.saved.scaling.unwrap_or_default();
        let before = scaling;
        egui::ComboBox::from_label("Video, image and web scaling")
            .selected_text(scaling.label())
            .show_ui(ui, |ui| {
                for m in SCALE_MODES {
                    ui.selectable_value(&mut scaling, m, m.label());
                }
            });
        if scaling != before {
            self.saved.scaling = Some(scaling);
            self.settings.lock().unwrap().scaling = scaling;
            self.save_settings();
            self.status = StatusMsg::ok("Scaling saved — re-apply the wallpaper to use it");
        }
    }

    fn playback_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Playback");
        let mut changed = false;
        egui::Grid::new("playback_rules")
            .num_columns(2)
            .show(ui, |ui| {
                for (key, text) in PLAYBACK_CONDITIONS {
                    ui.label(text);
                    let mut action = playback_action(&self.saved, key);
                    egui::ComboBox::from_id_salt(("playback", key))
                        .selected_text(action.label())
                        .show_ui(ui, |ui| {
                            for a in PlaybackAction::ALL {
                                changed |= ui.selectable_value(&mut action, a, a.label()).changed();
                            }
                        });
                    self.saved.playback.set(key, action);
                    ui.end_row();
                }
            });
        if changed {
            self.save_settings();
            self.status = StatusMsg::ok("Playback rule saved — applies to the next wallpaper");
        }
        ui.label(
            egui::RichText::new("stop = pause and free the renderer's memory")
                .size(11.0)
                .color(TEXT_MUTED),
        );
    }

    /// What `wp-engine run` does: the active playlist, else the saved
    /// screens and default.
    fn run_saved_setup(&mut self) {
        match std::env::current_exe()
            .and_then(|exe| std::process::Command::new(exe).arg("run").spawn())
        {
            Ok(mut child) => {
                std::thread::spawn(move || child.wait());
                self.active_title = Some("saved setup".into());
                self.status = StatusMsg::ok("Applying saved setup…");
            }
            Err(e) => self.status = StatusMsg::err(format!("Couldn't run saved setup: {e}")),
        }
    }
}

fn about(ui: &mut egui::Ui) {
    ui.heading("wp-engine");
    ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
    ui.label("A Wallpaper Engine client for Linux and macOS.");
    ui.hyperlink("https://github.com/code-wolf-byte/open-wallpaper-engine");
    ui.label(
        egui::RichText::new(
            "Scene rendering builds on Almamu's linux-wallpaperengine. \
             Wallpaper Engine is a product of Kristjan Skutta; this project is unaffiliated.",
        )
        .size(11.0)
        .color(TEXT_MUTED),
    );
}

fn playback_action(s: &WpSettings, key: &str) -> PlaybackAction {
    let p = &s.playback;
    match key {
        "focus" => p.other_focused,
        "maximized" => p.other_maximized,
        "fullscreen" => p.other_fullscreen,
        "audio" => p.other_audio,
        "battery" => p.on_battery,
        _ => p.locked,
    }
}
