//! project.json user-property resolution.
//!
//! Mirrors linux-wallpaperengine's PropertyParser + UserSettingParser flow:
//! `project.json` declares `general.properties` (sliders, colors, booleans,
//! combos); any value inside `scene.json` may reference one with
//! `{"user": "propname", "value": <default>}`. The effective value is the
//! property's configured value, which the user may override per-run with
//! `--set-property name=value` (the C++ `--set-property` equivalent).

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

/// One declared property from project.json `general.properties`. Real-world
/// type distribution surveyed across 271 cached Workshop items: `slider`
/// (640), `color` (447), `bool` (429), `combo` (111), `text` (91, a static
/// HTML label/separator — no real value), `group` (39, a UI section
/// header — also no real value), `textinput` (19, free text), `file` (16),
/// `directory` (2), `scenetexture` (1). `is_settable()` distinguishes the
/// two decorative types (plus a missing/empty `type`, which real content
/// also uses for the same static-label purpose) from everything else.
#[derive(Debug, Clone, Default)]
pub struct SceneProperty {
    pub name: String,
    /// Declared type: "slider" | "color" | "bool" | "combo" | "text" | ...
    pub kind: String,
    /// Human-readable label ("ui_browse_properties_..." keys are common).
    pub text: String,
    /// The effective raw value (project default, then CLI override).
    pub value: Value,
    /// `combo`'s declared choices, `(value, label)` in declared order —
    /// `value` is stringified regardless of the declaration's own JSON type
    /// (a *scene* combo's `options[].value` is always a JSON string, 52/52
    /// surveyed; a *web* combo's is genuinely mixed, 43 numeric / 16 string
    /// of 59 surveyed) since this is for display/matching, not for
    /// reproducing the original type — see `convert_override`'s `"combo"`
    /// arm for where the type that actually matters (what gets stored back
    /// into `value` on an override) is handled correctly per-property.
    /// `label` is what a picker should display.
    pub options: Vec<(String, String)>,
    /// `slider`'s declared range/step, for clamping a `--set-property`
    /// override into range the way a real slider widget would (a raw CLI
    /// float has no widget to keep it in bounds otherwise).
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    /// Display position: WE lists properties by `order`, then `index`.
    pub order: i64,
    pub index: i64,
    /// JS expression over other properties (`show.value == true`) that
    /// decides whether WE shows this row; `None` = always shown.
    pub condition: Option<String>,
}

impl SceneProperty {
    /// `false` for the decorative, no-real-value declarations real content
    /// actually ships (`text` labels/HTML separators, `group` section
    /// headers, and a `type`-less declaration — real content uses all
    /// three for the same "this is UI chrome, not a setting" purpose).
    /// `list-properties`/a settings-picker UI should filter these out
    /// rather than present them as things a user can meaningfully set.
    pub fn is_settable(&self) -> bool {
        !matches!(self.kind.as_str(), "text" | "group" | "")
    }
}

/// The resolved property set for one wallpaper project.
#[derive(Debug, Clone, Default)]
pub struct SceneProperties {
    properties: HashMap<String, SceneProperty>,
}

impl SceneProperties {
    /// Load `general.properties` from the project.json inside `dir`.
    /// Returns an empty set when there is no project.json or no properties —
    /// scenes without user settings are the common case.
    pub fn from_project_dir(dir: &Path) -> Self {
        let mut props = Self::default();
        let path = dir.join("project.json");
        let Ok(data) = std::fs::read_to_string(&path) else {
            return props;
        };
        let Ok(json) = serde_json::from_str::<Value>(&data) else {
            return props;
        };
        props.load_from_project_json(&json);
        if let Some(preset) = preset_overrides_for(dir) {
            props.apply_overrides(preset);
        }
        props.apply_overrides(crate::settings::WpSettings::load().property_overrides_for_dir(dir));
        let (scope, overrides) = global_overrides().lock().unwrap().clone();
        if scope.is_none_or(|s| s == canonical(dir)) {
            props.apply_overrides(overrides);
        }
        props
    }

    fn load_from_project_json(&mut self, project: &Value) {
        let Some(map) = project
            .get("general")
            .and_then(|g| g.get("properties"))
            .and_then(|p| p.as_object())
        else {
            return;
        };
        let strings = localization(project);
        let tr = |s: &str| strings.get(s).cloned().unwrap_or_else(|| s.to_string());
        for (name, decl) in map {
            let kind = decl
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            let text = tr(decl.get("text").and_then(|t| t.as_str()).unwrap_or(""));
            let value = decl.get("value").cloned().unwrap_or(Value::Null);
            let options = decl
                .get("options")
                .and_then(|o| o.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|opt| {
                            // `value` is a JSON string on every real *scene*
                            // combo but a bare number on many real *web*
                            // ones (`backgroundsource`'s `{"value": 1}`,
                            // workshop item 893418273) — `.as_str()` alone
                            // silently dropped every numeric-valued option.
                            // Stringify either shape; this list is for
                            // display/matching (`list-properties`'
                            // "choices:" line, and `apply_overrides`'
                            // does-this-override-match-a-declared-choice
                            // check), where the string form is exactly what
                            // both need regardless of the JSON type
                            // underneath.
                            let value = match opt.get("value") {
                                Some(Value::String(s)) => s.clone(),
                                Some(v @ Value::Number(_)) => json_value_as_string(v),
                                _ => return None,
                            };
                            let label =
                                tr(opt.get("label").and_then(|l| l.as_str()).unwrap_or(&value));
                            Some((value, label))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let min = decl.get("min").and_then(|v| v.as_f64());
            let max = decl.get("max").and_then(|v| v.as_f64());
            let step = decl.get("step").and_then(|v| v.as_f64());
            let int = |k: &str| decl.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as i64;
            let condition = decl
                .get("condition")
                .and_then(|c| c.as_str())
                .map(str::trim)
                .filter(|c| !c.is_empty())
                .map(String::from);
            self.properties.insert(
                name.clone(),
                SceneProperty {
                    name: name.clone(),
                    kind,
                    text,
                    value,
                    options,
                    min,
                    max,
                    step,
                    order: int("order"),
                    index: int("index"),
                    condition,
                },
            );
        }
    }

    /// Apply `name=value` overrides, converting the raw string to the
    /// property's declared type (mirrors the C++ `--set-property` handling).
    pub fn apply_overrides(&mut self, overrides: HashMap<String, String>) {
        for (name, raw) in overrides {
            match self.properties.get_mut(&name) {
                Some(prop) => {
                    if prop.kind == "combo"
                        && !prop.options.is_empty()
                        && !prop.options.iter().any(|(value, _)| value == &raw)
                    {
                        tracing::warn!(
                            target: "properties",
                            "'{name}' override '{raw}' doesn't match any of this combo's declared \
                             option values ({}); setting it anyway",
                            prop.options
                                .iter()
                                .map(|(v, _)| v.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                    prop.value = convert_override(prop, &raw);
                }
                None => {
                    // Unknown property: keep it anyway so scenes that reference
                    // undeclared names (creators do this) still resolve.
                    self.properties.insert(
                        name.clone(),
                        SceneProperty {
                            name,
                            value: guess_value(&raw),
                            ..Default::default()
                        },
                    );
                }
            }
        }
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.properties.get(name).map(|p| &p.value)
    }

    pub fn is_empty(&self) -> bool {
        self.properties.is_empty()
    }

    /// Iterate declared properties in WE's display order.
    pub fn iter(&self) -> impl Iterator<Item = &SceneProperty> {
        let mut all: Vec<&SceneProperty> = self.properties.values().collect();
        all.sort_by(|a, b| (a.order, a.index, &a.name).cmp(&(b.order, b.index, &b.name)));
        all.into_iter()
    }

    /// Recursively resolve `{"user": ...}` references in a scene JSON tree.
    ///
    /// Wherever an object carries a `user` key naming a known property, its
    /// `value` is replaced by the property's effective value. The `{"value":X}`
    /// wrapper is preserved because downstream parsers unwrap it.
    ///
    /// Conditional references (`"user": {"name":..., "condition":...}`)
    /// evaluate to a boolean: the property's value, stringified, compared
    /// against the condition string (the reference's DynamicValue equality —
    /// `condition == newValue`, DynamicValue.cpp:176). Copying the raw
    /// property value instead made e.g. a combo value of `1` read as
    /// "truthy", turning layers visible whose scene default is
    /// `"value": false` (wallpaper 2952574984's fullscreen white "Solid"
    /// layer is gated on `areffects == 3`).
    pub fn resolve_scene_json(&self, node: &mut Value) {
        if self.properties.is_empty() {
            return;
        }
        self.resolve_node(node);
    }

    fn resolve_node(&self, node: &mut Value) {
        match node {
            Value::Object(map) => {
                enum UserRef {
                    Plain(String),
                    Conditional(String, String),
                }
                let user_ref = match map.get("user") {
                    Some(Value::String(s)) => Some(UserRef::Plain(s.clone())),
                    Some(Value::Object(user)) => {
                        let name = user.get("name").and_then(|n| n.as_str());
                        match (name, user.get("condition").and_then(|c| c.as_str())) {
                            (Some(n), Some(c)) => {
                                Some(UserRef::Conditional(n.to_string(), c.to_string()))
                            }
                            (Some(n), None) => Some(UserRef::Plain(n.to_string())),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                match user_ref {
                    Some(UserRef::Plain(name)) => {
                        if let Some(prop) = self.properties.get(&name) {
                            if !prop.value.is_null() {
                                map.insert("value".into(), prop.value.clone());
                            }
                        }
                    }
                    Some(UserRef::Conditional(name, condition)) => {
                        if let Some(prop) = self.properties.get(&name) {
                            if !prop.value.is_null() {
                                let matches = json_value_as_string(&prop.value) == condition;
                                map.insert("value".into(), Value::Bool(matches));
                            }
                        }
                    }
                    None => {}
                }
                for (_, child) in map.iter_mut() {
                    self.resolve_node(child);
                }
            }
            Value::Array(items) => {
                for child in items.iter_mut() {
                    self.resolve_node(child);
                }
            }
            _ => {}
        }
    }
}

/// Stringify a property value the way WE's condition equality sees it:
/// integers without a decimal point (combo values are ints — condition
/// strings like `"3"` must match a JSON `3` and a JSON `3.0` alike).
fn json_value_as_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => match n.as_i64() {
            Some(i) => i.to_string(),
            None => n
                .as_f64()
                .map(|f| {
                    if f.fract() == 0.0 {
                        format!("{}", f as i64)
                    } else {
                        f.to_string()
                    }
                })
                .unwrap_or_default(),
        },
        other => other.to_string(),
    }
}

pub fn convert_override(prop: &SceneProperty, raw: &str) -> Value {
    match prop.kind.as_str() {
        "bool" => Value::Bool(matches!(raw, "1" | "true" | "yes" | "on")),
        "slider" => {
            let clamped = match raw.parse::<f64>() {
                Ok(mut f) => {
                    // A real slider widget can't produce an out-of-range
                    // value; a raw CLI/saved override has no such widget to
                    // keep it honest, so clamp to the declared range here —
                    // real content always declares both (surveyed 640
                    // real `slider` declarations, min/max present on every
                    // one checked).
                    if let Some(min) = prop.min {
                        f = f.max(min);
                    }
                    if let Some(max) = prop.max {
                        f = f.min(max);
                    }
                    serde_json::Number::from_f64(f).map(Value::Number)
                }
                Err(_) => None,
            };
            clamped.unwrap_or_else(|| Value::String(raw.to_string()))
        }
        // Combo values track whatever JSON type the wallpaper's own
        // declaration used, not a fixed one: *scene* wallpapers are 100%
        // string (52/52 real `combo` declarations surveyed — matching
        // `resolve_scene_json`'s stringified-equality condition mechanic),
        // but *web* wallpapers are genuinely mixed (43 numeric / 16 string
        // of 59 surveyed) since their own author-written JS does its own
        // comparisons — a numeric `backgroundsource` combo (workshop item
        // 893418273) almost certainly does a strict `===`/`switch` check
        // that a forced-string override would silently break. An earlier
        // version of this always produced `Value::String`, correct for
        // scene wallpapers (where it was verified) but wrong for a web
        // wallpaper's numeric combo — matching the *already-declared*
        // value's own type, instead of assuming one, is correct for both.
        "combo" => match &prop.value {
            Value::Number(_) => raw
                .parse::<i64>()
                .map(|i| Value::Number(i.into()))
                .unwrap_or_else(|_| Value::String(raw.to_string())),
            _ => Value::String(raw.to_string()),
        },
        // color properties are "r g b" strings in WE; keep raw text
        _ => Value::String(raw.to_string()),
    }
}

fn guess_value(raw: &str) -> Value {
    if let Ok(i) = raw.parse::<i64>() {
        return Value::Number(i.into());
    }
    if let Ok(f) = raw.parse::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return Value::Number(n);
        }
    }
    match raw {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        _ => Value::String(raw.to_string()),
    }
}

/// The wallpaper's own label translations (`general.localization`) for the
/// user's language, falling back to `en-us`, plus WE's one built-in label.
fn localization(project: &Value) -> HashMap<String, String> {
    let mut out = HashMap::from([(
        "ui_browse_properties_scheme_color".to_string(),
        "Scheme color".to_string(),
    )]);
    let Some(langs) = project
        .pointer("/general/localization")
        .and_then(|l| l.as_object())
    else {
        return out;
    };
    // `de_DE.UTF-8` -> `de-de`.
    let lang = std::env::var("LANG")
        .unwrap_or_default()
        .split('.')
        .next()
        .unwrap_or("")
        .replace('_', "-")
        .to_lowercase();
    let table = langs
        .get(&lang)
        .or_else(|| {
            langs
                .iter()
                .find(|(k, _)| k.split('-').next() == lang.split('-').next())
                .map(|(_, v)| v)
        })
        .or_else(|| langs.get("en-us"));
    if let Some(table) = table.and_then(|t| t.as_object()) {
        out.extend(
            table
                .iter()
                .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))),
        );
    }
    out
}

/// Which of `props` WE would show: each row's `condition` (a JS expression
/// such as `show.value == true && mode.value != 2`) evaluated against
/// `values`, the current value of every property by name. A condition that
/// fails to evaluate shows its row rather than hiding a setting.
pub fn visible_rows(props: &[SceneProperty], values: &HashMap<String, Value>) -> Vec<bool> {
    use boa_engine::{Context, Source};
    if props.iter().all(|p| p.condition.is_none()) {
        return vec![true; props.len()];
    }
    let mut ctx = Context::default();
    // Conditions come from the wallpaper: keep a hostile one from hanging us.
    ctx.runtime_limits_mut().set_loop_iteration_limit(10_000);
    ctx.runtime_limits_mut().set_recursion_limit(64);
    let scope: serde_json::Map<String, Value> = props
        .iter()
        .map(|p| {
            let v = values.get(&p.name).unwrap_or(&p.value).clone();
            (p.name.clone(), serde_json::json!({ "value": v }))
        })
        .collect();
    let setup = format!(
        "(function(p){{for (var k in p) globalThis[k] = p[k];}})({});",
        Value::Object(scope)
    );
    if ctx.eval(Source::from_bytes(setup.as_bytes())).is_err() {
        return vec![true; props.len()];
    }
    props
        .iter()
        .map(|p| match &p.condition {
            None => true,
            Some(c) => ctx
                .eval(Source::from_bytes(format!("!!({c})").as_bytes()))
                .ok()
                .and_then(|v| v.as_boolean())
                .unwrap_or(true),
        })
        .collect()
}

// ── Global override store ─────────────────────────────────────────────────────
// Set at startup from CLI/application context; consulted every time a
// scene's properties are loaded, after the wallpaper's own saved settings.
// Scoped to one wallpaper directory so other screens' wallpapers don't
// inherit them.

type ScopedOverrides = (Option<std::path::PathBuf>, HashMap<String, String>);

fn global_overrides() -> &'static Mutex<ScopedOverrides> {
    static OVERRIDES: OnceLock<Mutex<ScopedOverrides>> = OnceLock::new();
    OVERRIDES.get_or_init(|| Mutex::new((None, HashMap::new())))
}

/// Install `--set-property` overrides for every scene this process loads.
pub fn set_global_overrides(overrides: HashMap<String, String>) {
    *global_overrides().lock().unwrap() = (None, overrides);
}

/// Install overrides that apply only to the wallpaper in `dir`.
pub fn set_overrides_for(dir: &Path, overrides: HashMap<String, String>) {
    *global_overrides().lock().unwrap() = (Some(canonical(dir)), overrides);
}

// ── Preset overrides ──────────────────────────────────────────────────────────
// A Workshop preset is a property set layered onto another item. Its values
// apply to that item's directory, under the user's own `--set-property`
// overrides. Keyed by the canonical dependency directory.

fn preset_store() -> &'static Mutex<HashMap<std::path::PathBuf, HashMap<String, String>>> {
    static PRESETS: OnceLock<Mutex<HashMap<std::path::PathBuf, HashMap<String, String>>>> =
        OnceLock::new();
    PRESETS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn canonical(dir: &Path) -> std::path::PathBuf {
    std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf())
}

/// Register `preset`'s values as overrides for wallpapers loaded from
/// `target_dir`. String values naming a file inside `preset_dir` (presets
/// ship their own `files/`) are rewritten to absolute paths, since the
/// wallpaper itself is loaded from `target_dir`. `null` entries (group
/// headers) are skipped.
pub fn register_preset(
    target_dir: &Path,
    preset_dir: &Path,
    preset: &serde_json::Map<String, Value>,
) {
    let overrides = preset_to_overrides(preset_dir, preset);
    preset_store()
        .lock()
        .unwrap()
        .insert(canonical(target_dir), overrides);
}

fn preset_to_overrides(
    preset_dir: &Path,
    preset: &serde_json::Map<String, Value>,
) -> HashMap<String, String> {
    preset
        .iter()
        .filter_map(|(name, value)| {
            let raw = match value {
                Value::Null => return None,
                Value::Bool(b) => if *b { "1" } else { "0" }.to_string(),
                Value::Number(n) => n.to_string(),
                Value::String(s) => {
                    let local = preset_dir.join(s);
                    if !s.is_empty() && !Path::new(s).is_absolute() && local.is_file() {
                        local.display().to_string()
                    } else {
                        s.clone()
                    }
                }
                other => other.to_string(),
            };
            Some((name.clone(), raw))
        })
        .collect()
}

fn preset_overrides_for(dir: &Path) -> Option<HashMap<String, String>> {
    preset_store().lock().unwrap().get(&canonical(dir)).cloned()
}

/// Parse a `name=value` CLI argument (bare `name` means boolean true).
pub fn parse_property_arg(arg: &str) -> (String, String) {
    match arg.split_once('=') {
        Some((name, value)) => (name.trim().to_string(), value.to_string()),
        None => (arg.trim().to_string(), "1".to_string()),
    }
}

/// List an item's declared properties with their effective defaults. For a
/// preset that's the item it builds on, with the preset's values applied.
pub fn list_properties(dir: &Path) -> Result<Vec<SceneProperty>> {
    let project = crate::workshop::load_project(dir).ok();
    let preset = project
        .as_ref()
        .and_then(|p| Some((p.dependency.as_deref()?, p.preset.as_ref()?)));
    let (base, preset) = match preset {
        Some((dep, values)) => (
            crate::workshop::find_item_dir(dep, Some(dir)).with_context(|| {
                format!("preset needs workshop item {dep}, which is not installed")
            })?,
            Some(preset_to_overrides(dir, values)),
        ),
        None => (dir.to_path_buf(), None),
    };
    let path = base.join("project.json");
    let data =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let json: Value =
        serde_json::from_str(&data).with_context(|| format!("parsing {}", path.display()))?;
    let mut props = SceneProperties::default();
    props.load_from_project_json(&json);
    if let Some(values) = preset {
        props.apply_overrides(values);
    }
    Ok(props.iter().cloned().collect())
}

#[cfg(test)]
mod preset_tests {
    use super::*;

    #[test]
    fn preset_values_become_override_strings_with_local_files_absolutized() {
        let dir = std::env::temp_dir().join(format!("wp-preset-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("files")).unwrap();
        std::fs::write(dir.join("files/bg.jpg"), b"x").unwrap();
        let preset: serde_json::Map<String, Value> = serde_json::from_str(
            r#"{"on":true,"off":false,"n":58,"col":"1 0 0","bg":"files/bg.jpg","hdr":null,"missing":"files/nope.jpg"}"#,
        )
        .unwrap();
        let o = preset_to_overrides(&dir, &preset);
        assert_eq!(o["on"], "1");
        assert_eq!(o["off"], "0");
        assert_eq!(o["n"], "58");
        assert_eq!(o["col"], "1 0 0");
        assert_eq!(o["bg"], dir.join("files/bg.jpg").display().to_string());
        assert_eq!(o["missing"], "files/nope.jpg");
        assert!(!o.contains_key("hdr"));
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditions_follow_current_values_and_survive_hostile_input() {
        let prop = |name: &str, kind: &str, value: Value, cond: Option<&str>| SceneProperty {
            name: name.into(),
            kind: kind.into(),
            value,
            condition: cond.map(String::from),
            ..Default::default()
        };
        let props = vec![
            prop("show", "bool", Value::Bool(true), None),
            prop("mode", "combo", Value::String("2".into()), None),
            prop(
                "a",
                "slider",
                1.into(),
                Some("show.value == true && mode.value != 2"),
            ),
            prop(
                "b",
                "slider",
                1.into(),
                Some("mode.value == 2 || !show.value"),
            ),
            prop(
                "hang",
                "bool",
                Value::Bool(true),
                Some("(function(){while(true){}})()"),
            ),
            prop("bad", "bool", Value::Bool(true), Some("nope.value ==")),
        ];
        assert_eq!(
            visible_rows(&props, &HashMap::new()),
            [true, true, false, true, true, true]
        );
        let values = HashMap::from([("mode".to_string(), Value::from(1))]);
        assert_eq!(visible_rows(&props, &values)[2..4], [true, false]);
    }

    #[test]
    fn labels_use_the_wallpapers_own_translations() {
        let project = serde_json::json!({"general": {
            "localization": {"en-us": {"ui_HEADER_AUDIO": "Audio"}},
            "properties": {
                "h": {"type": "text", "text": "ui_HEADER_AUDIO"},
                "c": {"type": "color", "text": "ui_browse_properties_scheme_color", "order": -1}
            }
        }});
        let mut props = SceneProperties::default();
        props.load_from_project_json(&project);
        let rows: Vec<_> = props
            .iter()
            .map(|p| (p.name.as_str(), p.text.as_str()))
            .collect();
        assert_eq!(rows, [("c", "Scheme color"), ("h", "Audio")]);
    }

    fn props_with(name: &str, kind: &str, value: Value) -> SceneProperties {
        let mut p = SceneProperties::default();
        p.properties.insert(
            name.to_string(),
            SceneProperty {
                name: name.to_string(),
                kind: kind.to_string(),
                text: String::new(),
                value,
                options: Vec::new(),
                min: None,
                max: None,
                ..Default::default()
            },
        );
        p
    }

    /// Conditional user references (`{"user": {"name", "condition"}}`)
    /// evaluate `stringify(property) == condition` into a boolean — copying
    /// the raw combo value (e.g. `1`) instead made `is_visible`'s non-bool
    /// fallback report `true` for layers whose scene default is `false`
    /// (2952574984's fullscreen white "Solid" layer, gated on
    /// `areffects == 3`).
    #[test]
    fn conditional_user_reference_evaluates_equality() {
        let props = props_with("areffects", "combo", Value::from(1));
        let mut node = serde_json::json!({
            "visible": { "user": { "condition": "3", "name": "areffects" }, "value": false }
        });
        props.resolve_scene_json(&mut node);
        assert_eq!(node["visible"]["value"], Value::Bool(false));

        let props = props_with("areffects", "combo", Value::from(3));
        let mut node = serde_json::json!({
            "visible": { "user": { "condition": "3", "name": "areffects" }, "value": false }
        });
        props.resolve_scene_json(&mut node);
        assert_eq!(node["visible"]["value"], Value::Bool(true));
    }

    /// Plain string references keep the existing copy-the-value behavior.
    #[test]
    fn plain_user_reference_copies_property_value() {
        let props = props_with("logoopacity", "slider", Value::from(0.25));
        let mut node = serde_json::json!({
            "alpha": { "user": "logoopacity", "value": 0.6 }
        });
        props.resolve_scene_json(&mut node);
        assert_eq!(node["alpha"]["value"], Value::from(0.25));
    }

    /// Shaped after a real declaration (workshop item 3598808038's
    /// `effect` property): every real `combo` declaration checked (111
    /// across 271 cached Workshop items) stores `value` as a JSON string
    /// matching one of `options[].value`, never a bare number — a
    /// `--set-property`/saved override must preserve that, or any
    /// downstream code expecting `.as_str()` on the effective value breaks.
    #[test]
    fn combo_override_stays_a_json_string_not_a_number() {
        let project = serde_json::json!({
            "general": { "properties": {
                "effect": {
                    "type": "combo",
                    "value": "1",
                    "options": [
                        {"label": "Layers", "value": "0"},
                        {"label": "Particles", "value": "1"}
                    ]
                }
            }}
        });
        let mut props = SceneProperties::default();
        props.load_from_project_json(&project);
        assert_eq!(props.get("effect"), Some(&Value::String("1".to_string())));

        let mut overrides = HashMap::new();
        overrides.insert("effect".to_string(), "0".to_string());
        props.apply_overrides(overrides);
        assert_eq!(
            props.get("effect"),
            Some(&Value::String("0".to_string())),
            "combo override must stay a JSON string, matching the scene's own declared type"
        );
    }

    /// Shaped after a real *web*-wallpaper combo declaration (workshop item
    /// 893418273's `backgroundsource`: `"value": 1`, options `1..4`, a bare
    /// JSON number, not a string). Surveyed 59 real web-wallpaper `combo`
    /// declarations: 43 numeric, 16 string — genuinely mixed, unlike scene
    /// wallpapers' 52/52 string. A forced-string override here would break
    /// a real wallpaper's own `===`/`switch` check against the number.
    #[test]
    fn combo_override_stays_numeric_when_the_declaration_is_numeric() {
        let project = serde_json::json!({
            "general": { "properties": {
                "backgroundsource": {
                    "type": "combo",
                    "value": 1,
                    "options": [
                        {"label": "Color", "value": 1},
                        {"label": "Image", "value": 2},
                        {"label": "ImageSlideShow", "value": 3},
                        {"label": "Video", "value": 4}
                    ]
                }
            }}
        });
        let mut props = SceneProperties::default();
        props.load_from_project_json(&project);
        assert_eq!(props.get("backgroundsource"), Some(&Value::from(1)));

        let mut overrides = HashMap::new();
        overrides.insert("backgroundsource".to_string(), "3".to_string());
        props.apply_overrides(overrides);
        assert_eq!(
            props.get("backgroundsource"),
            Some(&Value::from(3)),
            "combo override must stay numeric when the declaration itself is numeric"
        );
    }

    /// The same numeric-valued declaration's `options` must still parse —
    /// an earlier version's `.as_str()`-only extraction silently dropped
    /// every option whose `value` was a bare number, leaving `options`
    /// empty for exactly this real, common shape.
    #[test]
    fn combo_options_parse_numeric_values_too() {
        let project = serde_json::json!({
            "general": { "properties": {
                "backgroundsource": {
                    "type": "combo",
                    "value": 1,
                    "options": [
                        {"label": "Color", "value": 1},
                        {"label": "ImageSlideShow", "value": 3}
                    ]
                }
            }}
        });
        let mut props = SceneProperties::default();
        props.load_from_project_json(&project);
        let prop = props.properties.get("backgroundsource").unwrap();
        assert_eq!(
            prop.options,
            vec![
                ("1".to_string(), "Color".to_string()),
                ("3".to_string(), "ImageSlideShow".to_string()),
            ]
        );
    }

    /// `options` parses into `(value, label)` pairs a picker can present —
    /// shaped after the same real `effect` declaration.
    #[test]
    fn combo_options_parse_with_values_and_labels() {
        let project = serde_json::json!({
            "general": { "properties": {
                "effect": {
                    "type": "combo",
                    "value": "1",
                    "options": [
                        {"label": "Layers", "value": "0"},
                        {"label": "Particles", "value": "1"}
                    ]
                }
            }}
        });
        let mut props = SceneProperties::default();
        props.load_from_project_json(&project);
        let effect = props.properties.get("effect").unwrap();
        assert_eq!(
            effect.options,
            vec![
                ("0".to_string(), "Layers".to_string()),
                ("1".to_string(), "Particles".to_string()),
            ]
        );
    }

    /// Shaped after a real declaration (3598808038's `dirtmouth` slider):
    /// a raw CLI/saved override has no widget to keep it in range the way
    /// a real slider would, so it must be clamped to the declared
    /// `min`/`max` rather than passed through as-is.
    #[test]
    fn slider_override_clamps_to_declared_range() {
        let project = serde_json::json!({
            "general": { "properties": {
                "dirtmouth": {"type": "slider", "value": 0.5, "min": 0, "max": 1, "step": 0.1}
            }}
        });
        let mut props = SceneProperties::default();
        props.load_from_project_json(&project);

        let mut overrides = HashMap::new();
        overrides.insert("dirtmouth".to_string(), "5.0".to_string());
        props.apply_overrides(overrides);
        assert_eq!(props.get("dirtmouth"), Some(&Value::from(1.0)));

        let mut overrides = HashMap::new();
        overrides.insert("dirtmouth".to_string(), "-2.0".to_string());
        props.apply_overrides(overrides);
        assert_eq!(props.get("dirtmouth"), Some(&Value::from(0.0)));

        // In-range values pass through unclamped.
        let mut overrides = HashMap::new();
        overrides.insert("dirtmouth".to_string(), "0.7".to_string());
        props.apply_overrides(overrides);
        assert_eq!(props.get("dirtmouth"), Some(&Value::from(0.7)));
    }

    /// Real content ships decorative `text`/`group`/type-less declarations
    /// (HTML separators, section headers) that aren't real settings —
    /// `is_settable()` is what a `list-properties`-style UI should filter
    /// on to avoid presenting a `<hr>` as something the user can set.
    #[test]
    fn is_settable_excludes_decorative_property_types() {
        let project = serde_json::json!({
            "general": { "properties": {
                "speed": {"type": "slider", "value": 1.0},
                "tint": {"type": "color", "value": "1 1 1"},
                "enabled": {"type": "bool", "value": true},
                "mode": {"type": "combo", "value": "0", "options": []},
                "banner": {"type": "text", "value": "<hr>"},
                "section": {"type": "group", "value": ""},
                "untyped_label": {"value": "1"}
            }}
        });
        let mut props = SceneProperties::default();
        props.load_from_project_json(&project);

        let settable: std::collections::BTreeSet<&str> = props
            .properties
            .values()
            .filter(|p| p.is_settable())
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(
            settable,
            ["speed", "tint", "enabled", "mode"].into_iter().collect()
        );
    }
}
