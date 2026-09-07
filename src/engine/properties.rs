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
#[derive(Debug, Clone)]
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
        props.apply_overrides(global_overrides().lock().unwrap().clone());
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
        for (name, decl) in map {
            let kind = decl
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            let text = decl
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
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
                            let label = opt
                                .get("label")
                                .and_then(|l| l.as_str())
                                .unwrap_or(&value)
                                .to_string();
                            Some((value, label))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let min = decl.get("min").and_then(|v| v.as_f64());
            let max = decl.get("max").and_then(|v| v.as_f64());
            let step = decl.get("step").and_then(|v| v.as_f64());
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
                            kind: String::new(),
                            text: String::new(),
                            value: guess_value(&raw),
                            options: Vec::new(),
                            min: None,
                            max: None,
                            step: None,
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

    /// Iterate declared properties (for `list-properties`).
    pub fn iter(&self) -> impl Iterator<Item = &SceneProperty> {
        let mut all: Vec<&SceneProperty> = self.properties.values().collect();
        all.sort_by(|a, b| a.name.cmp(&b.name));
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

fn convert_override(prop: &SceneProperty, raw: &str) -> Value {
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

// ── Global override store ─────────────────────────────────────────────────────
// Set once at startup from CLI/application context; consulted every time a
// scene's properties are loaded (matches the C++ ApplicationContext's
// settings.general.properties map).

fn global_overrides() -> &'static Mutex<HashMap<String, String>> {
    static OVERRIDES: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    OVERRIDES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Install `--set-property` overrides for all scenes loaded in this process.
pub fn set_global_overrides(overrides: HashMap<String, String>) {
    *global_overrides().lock().unwrap() = overrides;
}

/// Parse a `name=value` CLI argument (bare `name` means boolean true).
pub fn parse_property_arg(arg: &str) -> (String, String) {
    match arg.split_once('=') {
        Some((name, value)) => (name.trim().to_string(), value.to_string()),
        None => (arg.trim().to_string(), "1".to_string()),
    }
}

/// List a project's declared properties (for the `list-properties` command).
pub fn list_properties(dir: &Path) -> Result<Vec<SceneProperty>> {
    let path = dir.join("project.json");
    let data =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let json: Value =
        serde_json::from_str(&data).with_context(|| format!("parsing {}", path.display()))?;
    let mut props = SceneProperties::default();
    props.load_from_project_json(&json);
    Ok(props.iter().cloned().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

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
                step: None,
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
