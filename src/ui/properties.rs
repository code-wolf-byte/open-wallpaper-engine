//! The property editor in the details panel: a wallpaper's user properties
//! in Wallpaper Engine's order and wording, edited live and saved to
//! settings.json.

use super::pick_path;
use super::{WpApp, DANGER, TEXT_MUTED};
use crate::workshop::properties::SceneProperty;
use serde_json::Value;

impl WpApp {
    /// The selected wallpaper's user properties, as WE's two-column list.
    /// Changes are saved at once and shown live where it is displayed.
    pub(super) fn render_properties(&mut self, ui: &mut egui::Ui, idx: usize) {
        if self.prop_editor.as_ref().map(|e| e.idx) != Some(idx) {
            let props = crate::workshop::properties::list_properties(&self.wallpapers[idx].path)
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
        ui.horizontal(|ui| {
            ui.heading("Properties");
            if settable > 0 {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .button(egui::RichText::new("Reset").color(DANGER))
                        .on_hover_text("Back to the wallpaper's own defaults")
                        .clicked()
                    {
                        self.reset_properties(idx, &editor.props);
                        editor.visible.clear();
                    }
                });
            }
        });
        ui.separator();
        if settable == 0 {
            ui.label(egui::RichText::new("This wallpaper has no properties.").color(TEXT_MUTED));
        } else {
            egui::Grid::new(("properties", idx))
                .num_columns(2)
                .spacing([12.0, 8.0])
                .show(ui, |ui| {
                    if self.property_rows(ui, idx, &editor.props, &editor.visible) {
                        editor.visible.clear();
                    }
                });
        }
        self.prop_editor = Some(editor);
    }

    /// Drop every saved override of wallpaper `idx`.
    pub(super) fn reset_properties(&mut self, idx: usize, props: &[SceneProperty]) {
        let key = self.wallpapers[idx].workshop_id.clone();
        let names: Vec<String> = props
            .iter()
            .filter(|p| self.saved.unset_property(&key, &p.name))
            .map(|p| p.name.clone())
            .collect();
        if names.is_empty() {
            return;
        }
        self.save_settings();
        #[cfg(target_os = "linux")]
        self.daemon.send(crate::daemon::Request::PropertiesChanged {
            wallpaper: key,
            names,
        });
    }

    /// Each row's visibility under the current saved values.
    pub(super) fn visible_rows(&self, idx: usize, props: &[SceneProperty]) -> Vec<bool> {
        let saved = self
            .saved
            .property_overrides(&self.wallpapers[idx].workshop_id);
        let values = props
            .iter()
            .filter_map(|p| {
                let raw = saved.get(&p.name)?;
                Some((
                    p.name.clone(),
                    crate::workshop::properties::convert_override(p, raw),
                ))
            })
            .collect();
        crate::workshop::properties::visible_rows(props, &values)
    }

    /// Draws the rows WE would show; returns whether a value was changed.
    pub(super) fn property_rows(
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
                    ui.strong(text);
                } else {
                    ui.label(egui::RichText::new(text).color(TEXT_MUTED));
                }
                ui.end_row();
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
            ui.label(label(p));
            ui.horizontal(|ui| {
                match p.kind.as_str() {
                    "bool" => {
                        let mut on = matches!(current.as_str(), "1" | "true");
                        if ui.checkbox(&mut on, "").changed() {
                            commit = Some(if on { "1" } else { "0" }.into());
                        }
                    }
                    "slider" => {
                        let mut v = current.parse::<f64>().unwrap_or(0.0);
                        let range = p.min.unwrap_or(0.0)..=p.max.unwrap_or(1.0);
                        let mut slider = egui::Slider::new(&mut v, range);
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
                        let r = ui.add(egui::TextEdit::singleline(&mut text).desired_width(150.0));
                        if r.lost_focus() && text != current {
                            commit = Some(text.clone());
                        }
                        if r.has_focus() {
                            ui.data_mut(|d| d.insert_temp(id, text));
                        } else {
                            ui.data_mut(|d| d.remove::<String>(id));
                        }
                    }
                }
                if saved.is_some()
                    && ui
                        .small_button("↺")
                        .on_hover_text("Reset to default")
                        .clicked()
                {
                    commit = Some(String::new());
                }
            });
            ui.end_row();
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
}

/// The selected wallpaper's declared properties, loaded once per selection,
/// in WE's display order (headings included).
pub(super) struct PropertyEditor {
    pub idx: usize,
    pub props: Vec<SceneProperty>,
    /// Which rows their `condition` currently shows; refreshed on each edit.
    pub visible: Vec<bool>,
}

/// `text` without the HTML (tags and common entities) real content uses.
pub(super) fn plain_text(text: &str) -> String {
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
pub(super) fn label(p: &SceneProperty) -> String {
    let out = plain_text(&p.text);
    if out.is_empty() || out.starts_with("ui_") {
        p.name.replace('_', " ")
    } else {
        out.to_string()
    }
}

/// A declared value in the string form overrides are saved in.
pub(super) fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => if *b { "1" } else { "0" }.to_string(),
        other => other.to_string(),
    }
}

pub(super) fn parse_color(s: &str) -> [f32; 3] {
    let mut c = [0.0; 3];
    for (slot, v) in c.iter_mut().zip(s.split_whitespace()) {
        *slot = v.parse::<f32>().unwrap_or(0.0).clamp(0.0, 1.0);
    }
    c
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
