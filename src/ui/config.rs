//! Saved-settings UI: the selected wallpaper's properties, playlist and
//! screen assignment, and the settings window (frame rate, scaling,
//! playback rules, playlists, screens, autostart). Everything here edits
//! `settings.json`, the same file `wp-engine config` writes.

use super::{StatusMsg, WpApp, TEXT_MUTED, TEXT_PRIMARY};
use crate::engine::playback_gate::PlaybackAction;
use crate::engine::properties::SceneProperty;
use crate::render::{ScaleMode, ScreenLayout};
use crate::settings::{Playlist, WpSettings};
use serde_json::Value;

/// The selected wallpaper's declared properties, loaded once per selection,
/// in WE's display order (headings included).
pub(super) struct PropertyEditor {
    pub idx: usize,
    pub props: Vec<SceneProperty>,
    /// Which rows their `condition` currently shows; refreshed on each edit.
    pub visible: Vec<bool>,
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

const PLAYLIST_MODES: [(&str, &str); 4] = [
    ("timer", "Every N minutes"),
    ("daytime", "At times of day"),
    ("dayofweek", "By day of week"),
    ("login", "Next one each login"),
];

/// `text` without the HTML (tags and common entities) real content uses.
fn plain_text(text: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in text.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    let out = out
        .replace("&emsp;", " ")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&");
    out.trim_matches(|c: char| c.is_whitespace() || c == '—' || c == '-')
        .to_string()
}

/// A property's label, falling back to its name when the text is empty or
/// an untranslated `ui_` key.
fn label(p: &SceneProperty) -> String {
    let out = plain_text(&p.text);
    if out.is_empty() || out.starts_with("ui_") {
        p.name.replace('_', " ")
    } else {
        out.to_string()
    }
}

/// Ask the desktop for a file or folder via zenity or kdialog.
// ponytail: shells out instead of a file-dialog crate; use the XDG portal
// (zbus is already a dependency) if neither tool is common enough.
fn pick_path(directory: bool) -> Option<String> {
    let zenity: &[&str] = if directory {
        &["--file-selection", "--directory"]
    } else {
        &["--file-selection"]
    };
    let kdialog: &[&str] = if directory {
        &["--getexistingdirectory"]
    } else {
        &["--getopenfilename"]
    };
    [("zenity", zenity), ("kdialog", kdialog)]
        .iter()
        .find_map(|(cmd, args)| {
            let out = std::process::Command::new(cmd).args(*args).output().ok()?;
            let path = String::from_utf8(out.stdout).ok()?.trim().to_string();
            (out.status.success() && !path.is_empty()).then_some(path)
        })
}

/// A declared value in the string form overrides are saved in.
fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => if *b { "1" } else { "0" }.to_string(),
        other => other.to_string(),
    }
}

fn parse_color(s: &str) -> [f32; 3] {
    let mut c = [0.0; 3];
    for (slot, v) in c.iter_mut().zip(s.split_whitespace()) {
        *slot = v.parse::<f32>().unwrap_or(0.0).clamp(0.0, 1.0);
    }
    c
}

impl WpApp {
    fn save_settings(&mut self) {
        if let Err(e) = self.saved.save() {
            self.status = StatusMsg::err(format!("Saving settings failed: {e:#}"));
        }
    }

    /// Send a one-off request to the daemon, starting it if needed.
    #[cfg(target_os = "linux")]
    fn daemon_request(&mut self, req: crate::daemon::Request) {
        std::thread::spawn(move || {
            if let Err(e) = crate::daemon::request_or_spawn(&req) {
                tracing::warn!(target: "app", "daemon request failed: {e:#}");
            }
        });
    }

    fn title_of(&self, id: &str) -> String {
        self.wallpapers
            .iter()
            .find(|w| w.workshop_id == id)
            .map(|w| w.title().to_string())
            .unwrap_or_else(|| id.to_string())
    }

    // ── Detail panel ─────────────────────────────────────────────────────────

    /// "Add to playlist" and "Use on screen" for the selected wallpaper.
    pub(super) fn render_library_actions(&mut self, ui: &mut egui::Ui, idx: usize) {
        let id = self.wallpapers[idx].workshop_id.clone();
        ui.horizontal(|ui| {
            ui.menu_button("➕ Playlist", |ui| {
                let names: Vec<String> = self.saved.playlists.keys().cloned().collect();
                for name in names {
                    if ui.button(&name).clicked() {
                        let items = &mut self.saved.playlists.get_mut(&name).unwrap().items;
                        if !items.contains(&id) {
                            items.push(id.clone());
                        }
                        self.save_settings();
                        self.status = StatusMsg::ok(format!("Added to playlist '{name}'"));
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
                    .add_enabled(!name.is_empty(), egui::Button::new("Create"))
                    .clicked()
                {
                    self.saved
                        .playlists
                        .entry(name.clone())
                        .or_insert_with(|| Playlist {
                            items: Vec::new(),
                            delay_minutes: 60,
                            mode: "timer".into(),
                            order: "sequential".into(),
                            times: Vec::new(),
                        })
                        .items
                        .push(id.clone());
                    self.new_playlist.clear();
                    self.save_settings();
                    self.status = StatusMsg::ok(format!("Created playlist '{name}'"));
                    ui.close();
                }
            });
            ui.menu_button("🖵 Use on screen", |ui| {
                let outputs = self
                    .outputs
                    .get_or_insert_with(crate::platform::display::output_names)
                    .clone();
                if ui.button("All screens (default)").clicked() {
                    self.saved.default_background = Some(id.clone());
                    self.save_settings();
                    self.status = StatusMsg::ok("Saved as the default wallpaper");
                    ui.close();
                }
                for out in outputs {
                    if ui.button(&out).clicked() {
                        self.saved.screens.insert(out.clone(), id.clone());
                        self.save_settings();
                        self.status = StatusMsg::ok(format!(
                            "Saved for {out} — \"Apply saved setup\" in Settings shows it"
                        ));
                        ui.close();
                    }
                }
            });
        });
    }

    /// The selected wallpaper's user properties. Changes are saved at once
    /// and take effect the next time the wallpaper is applied.
    pub(super) fn render_properties(&mut self, ui: &mut egui::Ui, idx: usize) {
        if self.prop_editor.as_ref().map(|e| e.idx) != Some(idx) {
            let props = crate::engine::properties::list_properties(&self.wallpapers[idx].path)
                .unwrap_or_default();
            self.prop_editor = Some(PropertyEditor {
                idx,
                props,
                visible: Vec::new(),
            });
        }
        let mut editor = self.prop_editor.take().unwrap();
        if editor.visible.len() != editor.props.len() {
            editor.visible = self.visible_rows(idx, &editor.props);
        }
        let settable = editor.props.iter().filter(|p| p.is_settable()).count();
        if settable > 0 {
            egui::CollapsingHeader::new(
                egui::RichText::new(format!("Properties ({settable})")).color(TEXT_PRIMARY),
            )
            .default_open(true)
            .show(ui, |ui| {
                if self.property_rows(ui, idx, &editor.props, &editor.visible) {
                    editor.visible.clear();
                }
                ui.label(
                    egui::RichText::new("Changes apply live while this wallpaper is showing.")
                        .size(11.0)
                        .color(TEXT_MUTED),
                );
            });
        }
        self.prop_editor = Some(editor);
    }

    /// Each row's visibility under the current saved values.
    fn visible_rows(&self, idx: usize, props: &[SceneProperty]) -> Vec<bool> {
        let saved = self
            .saved
            .property_overrides(&self.wallpapers[idx].workshop_id);
        let values = props
            .iter()
            .filter_map(|p| {
                let raw = saved.get(&p.name)?;
                Some((
                    p.name.clone(),
                    crate::engine::properties::convert_override(p, raw),
                ))
            })
            .collect();
        crate::engine::properties::visible_rows(props, &values)
    }

    /// Draws the rows WE would show; returns whether a value was changed.
    fn property_rows(
        &mut self,
        ui: &mut egui::Ui,
        idx: usize,
        props: &[SceneProperty],
        visible: &[bool],
    ) -> bool {
        let key = self.wallpapers[idx].workshop_id.clone();
        let mut changed = false;
        for (p, _) in props.iter().zip(visible).filter(|(_, v)| **v) {
            if !p.is_settable() {
                // `text`/`group`/untyped rows are headings and spacers.
                let text = plain_text(&p.text);
                if text.is_empty() || text.starts_with("ui_") {
                    ui.add_space(4.0);
                } else if p.kind == "group" {
                    ui.add_space(4.0);
                    ui.strong(text);
                } else {
                    ui.label(egui::RichText::new(text).color(TEXT_MUTED));
                }
                continue;
            }
            let saved = self
                .saved
                .properties
                .get(&key)
                .and_then(|m| m.get(&p.name))
                .cloned();
            let current = saved.clone().unwrap_or_else(|| value_text(&p.value));
            let mut commit: Option<String> = None;
            ui.horizontal(|ui| {
                if saved.is_some()
                    && ui
                        .small_button("↺")
                        .on_hover_text("Reset to default")
                        .clicked()
                {
                    commit = Some(String::new());
                }
                match p.kind.as_str() {
                    "bool" => {
                        let mut on = matches!(current.as_str(), "1" | "true");
                        if ui.checkbox(&mut on, label(p)).changed() {
                            commit = Some(if on { "1" } else { "0" }.into());
                        }
                    }
                    "slider" => {
                        let mut v = current.parse::<f64>().unwrap_or(0.0);
                        let range = p.min.unwrap_or(0.0)..=p.max.unwrap_or(1.0);
                        let mut slider = egui::Slider::new(&mut v, range).text(label(p));
                        if let Some(step) = p.step.filter(|s| *s > 0.0) {
                            slider = slider.step_by(step);
                        }
                        let r = ui.add(slider);
                        if r.drag_stopped() || (r.changed() && !r.dragged()) {
                            commit = Some(v.to_string());
                        }
                    }
                    "color" => {
                        let mut c = parse_color(&current);
                        if ui.color_edit_button_rgb(&mut c).changed() {
                            commit = Some(format!("{} {} {}", c[0], c[1], c[2]));
                        }
                        ui.label(label(p));
                    }
                    "combo" => {
                        let shown = p
                            .options
                            .iter()
                            .find(|(v, _)| *v == current)
                            .map(|(_, l)| l.clone())
                            .unwrap_or_else(|| current.clone());
                        egui::ComboBox::from_id_salt(("prop", &p.name))
                            .selected_text(shown)
                            .show_ui(ui, |ui| {
                                for (v, l) in &p.options {
                                    if ui.selectable_label(*v == current, l).clicked() {
                                        commit = Some(v.clone());
                                    }
                                }
                            });
                        ui.label(label(p));
                    }
                    // textinput, file, directory, scenetexture: free text, with
                    // a picker for the path kinds.
                    _ => {
                        if matches!(p.kind.as_str(), "file" | "directory" | "scenetexture")
                            && ui.small_button("📂").on_hover_text("Browse…").clicked()
                        {
                            commit = pick_path(p.kind == "directory");
                        }
                        let id = ui.make_persistent_id(("prop_text", &p.name));
                        let mut text = ui
                            .data_mut(|d| d.get_temp::<String>(id))
                            .unwrap_or_else(|| current.clone());
                        let r = ui.add(
                            egui::TextEdit::singleline(&mut text)
                                .desired_width(150.0)
                                .hint_text(label(p)),
                        );
                        if r.lost_focus() && text != current {
                            commit = Some(text.clone());
                        }
                        if r.has_focus() {
                            ui.data_mut(|d| d.insert_temp(id, text));
                        } else {
                            ui.data_mut(|d| d.remove::<String>(id));
                        }
                        ui.label(label(p));
                    }
                }
            });
            if let Some(value) = commit {
                if value.is_empty() {
                    self.saved.unset_property(&key, &p.name);
                } else {
                    self.saved.set_property(&key, &p.name, &value);
                }
                self.save_settings();
                // Show it live on whichever screens display this wallpaper.
                #[cfg(target_os = "linux")]
                self.daemon.send(crate::daemon::Request::PropertiesChanged {
                    wallpaper: key.clone(),
                    names: vec![p.name.clone()],
                });
                changed = true;
            }
        }
        changed
    }

    // ── Settings window ──────────────────────────────────────────────────────

    pub(super) fn render_settings_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_settings;
        egui::Window::new("Settings")
            .open(&mut open)
            .default_width(460.0)
            .vscroll(true)
            .show(ctx, |ui| {
                self.general_settings(ui);
                ui.separator();
                self.playback_settings(ui);
                ui.separator();
                self.screen_settings(ui);
                ui.separator();
                self.playlist_settings(ui);
            });
        self.show_settings = open;
    }

    fn general_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("General");
        let mut fps = self
            .saved
            .fps
            .unwrap_or(crate::render::settings::DEFAULT_FPS);
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

        let mut layout = self.saved.layout.unwrap_or_default();
        let before = layout;
        egui::ComboBox::from_label("Multiple monitors")
            .selected_text(layout.label())
            .show_ui(ui, |ui| {
                for l in ScreenLayout::ALL {
                    ui.selectable_value(&mut layout, l, l.label());
                }
            });
        if layout != before {
            self.saved.layout = Some(layout);
            self.save_settings();
            self.status = StatusMsg::ok("Layout saved — re-apply the wallpaper to use it");
        }

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
    }

    fn playback_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("While…");
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

    fn screen_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Screens");
        let mut changed = false;
        let default = self.saved.default_background.clone();
        ui.horizontal(|ui| {
            ui.label("Default:");
            match &default {
                Some(id) => {
                    ui.label(self.title_of(id));
                    if ui.small_button("✖").clicked() {
                        self.saved.default_background = None;
                        changed = true;
                    }
                }
                None => {
                    ui.label(egui::RichText::new("none").color(TEXT_MUTED));
                }
            }
        });
        let screens: Vec<(String, String)> = self
            .saved
            .screens
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for (screen, id) in screens {
            ui.horizontal(|ui| {
                ui.label(format!("{screen}:"));
                ui.label(self.title_of(&id));
                if ui.small_button("✖").clicked() {
                    self.saved.screens.remove(&screen);
                    changed = true;
                }
            });
        }
        if changed {
            self.save_settings();
        }
        ui.label(
            egui::RichText::new("Assign wallpapers with \"Use on screen\" in the details panel.")
                .size(11.0)
                .color(TEXT_MUTED),
        );
        if ui.button("▶ Apply saved setup").clicked() {
            self.run_saved_setup();
        }
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

    fn playlist_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Playlists");
        if self.saved.playlists.is_empty() {
            ui.label(
                egui::RichText::new("Add wallpapers with \"➕ Playlist\" in the details panel.")
                    .size(11.0)
                    .color(TEXT_MUTED),
            );
            return;
        }
        #[cfg(target_os = "linux")]
        ui.horizontal(|ui| {
            use crate::daemon::Request;
            if ui.button("⏮ Previous").clicked() {
                self.daemon_request(Request::PreviousItem);
            }
            if ui.button("⏭ Next").clicked() {
                self.daemon_request(Request::NextItem);
            }
            if ui.button("■ Stop rotating").clicked() {
                self.daemon_request(Request::StopPlaylist);
            }
        });

        let names: Vec<String> = self.saved.playlists.keys().cloned().collect();
        let mut changed = false;
        let mut delete = None;
        for name in names {
            let active = self.saved.active_playlist.as_deref() == Some(name.as_str());
            egui::CollapsingHeader::new(if active {
                format!("★ {name}")
            } else {
                name.clone()
            })
            .id_salt(("playlist", &name))
            .default_open(active)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    #[cfg(target_os = "linux")]
                    if ui.button("▶ Play").clicked() {
                        self.daemon_request(crate::daemon::Request::PlayPlaylist {
                            name: name.clone(),
                        });
                        self.active_title = Some(format!("playlist '{name}'"));
                    }
                    let mut is_active = active;
                    if ui
                        .checkbox(&mut is_active, "Play at login")
                        .on_hover_text("`wp-engine run` plays this playlist")
                        .changed()
                    {
                        self.saved.active_playlist = is_active.then(|| name.clone());
                        changed = true;
                    }
                    if ui.button("🗑 Delete").clicked() {
                        delete = Some(name.clone());
                    }
                });
                let titles: Vec<String> = self.saved.playlists[&name]
                    .items
                    .iter()
                    .map(|id| self.title_of(id))
                    .collect();
                let p = self.saved.playlists.get_mut(&name).unwrap();
                changed |= playlist_editor(ui, &name, p, &titles);
            });
        }
        if let Some(name) = delete {
            self.saved.playlists.remove(&name);
            if self.saved.active_playlist.as_deref() == Some(name.as_str()) {
                self.saved.active_playlist = None;
            }
            changed = true;
        }
        if changed {
            self.save_settings();
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_drop_markup_and_fall_back_to_the_name() {
        let mut p = SceneProperty {
            name: "add_wave".into(),
            kind: "bool".into(),
            text: "<b>Add Wave —————</b>".into(),
            value: Value::Bool(true),
            options: Vec::new(),
            min: None,
            max: None,
            ..Default::default()
        };
        assert_eq!(label(&p), "Add Wave");
        p.text = "ui_browse_properties_scheme_color".into();
        assert_eq!(label(&p), "add wave");
        assert_eq!(value_text(&p.value), "1");
        assert_eq!(parse_color("1 0.5 2"), [1.0, 0.5, 1.0]);
    }
}
