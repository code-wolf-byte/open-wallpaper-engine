// Agent output (qwen3-coder) — corrected field names to match our scene.rs serde structs:
//   SceneObject has no id/alpha/color fields → use index + defaults
//   Effect has no name field → empty string
//   Pass has no combos field → empty HashMap
//   json_to_animated scripted branch: inner_value.value, not inner_value directly
//   camera is Option<Camera>, must use as_ref()
use super::dynamic_value::{AnimatedValue, DynamicValue};
use super::objects::{EffectModel, ObjectModel, PassModel, SceneModel};
use crate::engine::model::scene;
use anyhow::Result;
use std::collections::HashMap;

pub fn scene_to_model(scene: &scene::Scene) -> Result<SceneModel> {
    let camera_eye = scene
        .camera
        .as_ref()
        .and_then(|c| c.parsed_eye())
        .map(|e| [e[0] as f32, e[1] as f32, e[2] as f32])
        .unwrap_or([0.0; 3]);

    let objects = scene
        .objects
        .iter()
        .enumerate()
        .map(|(i, obj)| object_to_model(i, obj))
        .collect();

    Ok(SceneModel {
        camera_eye,
        ambient_color: [1.0; 3],
        objects,
    })
}

fn object_to_model(index: usize, obj: &scene::SceneObject) -> ObjectModel {
    ObjectModel {
        index,
        name: obj.name.clone().unwrap_or_default(),
        image: obj.image.clone(),
        origin: obj
            .origin
            .as_ref()
            .map(json_to_animated)
            .unwrap_or_default(),
        // WE "scale" is a vec3 multiplier applied to the layer's native texture dimensions.
        // "size" is an optional pixel-dimension override (rarely set; texture size takes precedence).
        scale: obj
            .scale
            .as_ref()
            .map(json_to_animated)
            .unwrap_or(AnimatedValue::from([1.0f32, 1.0, 1.0])),
        angles: obj
            .angles
            .as_ref()
            .map(json_to_animated)
            .unwrap_or_default(),
        alpha: obj
            .alpha
            .as_ref()
            .map(json_to_animated)
            .unwrap_or(AnimatedValue::from(1.0f32)),
        color: obj
            .color
            .as_ref()
            .map(json_to_animated)
            .unwrap_or(AnimatedValue::from([1.0f32, 1.0, 1.0])),
        brightness: obj
            .brightness
            .as_ref()
            .map(json_to_animated)
            .unwrap_or(AnimatedValue::from(1.0f32)),
        visible: obj
            .visible
            .as_ref()
            .map(json_to_animated)
            .unwrap_or(AnimatedValue::from(true)),
        blend_mode: obj.color_blend_mode,
        effects: obj.effects.iter().map(effect_to_model).collect(),
        particle: obj.particle.clone(),
        copybackground: obj.copybackground,
    }
}

fn effect_to_model(eff: &scene::Effect) -> EffectModel {
    EffectModel {
        file: eff.file.clone(),
        name: String::new(),
        visible: eff.is_visible(),
        passes: eff.passes.iter().map(pass_to_model).collect(),
    }
}

fn pass_to_model(pass: &scene::Pass) -> PassModel {
    let shader_values: HashMap<String, AnimatedValue> = pass
        .constantshadervalues
        .iter()
        .map(|(k, v)| (k.clone(), json_to_animated(v)))
        .collect();

    // Positional: preserve `null` slots as `None` instead of dropping them,
    // otherwise a texture after a gap (e.g. nitro's `[null, "clouds", "mask"]`)
    // shifts into the wrong shader slot.
    let textures: Vec<Option<String>> = pass
        .textures
        .iter()
        .map(|t| {
            t.as_ref()
                .and_then(|t| t.file.clone().or_else(|| t.name.clone()))
        })
        .collect();

    PassModel {
        combos: pass.combos.clone(),
        shader_values,
        textures,
    }
}

pub fn json_to_animated(v: &serde_json::Value) -> AnimatedValue {
    match v {
        serde_json::Value::Object(map) => {
            if let (Some(script), Some(inner)) = (map.get("script"), map.get("value")) {
                // {"script": "...", "value": X} — animated/scripted value
                let script_src = script.as_str().unwrap_or("").to_string();
                let inner_val = json_to_animated(inner);
                AnimatedValue::scripted(inner_val.value, script_src)
            } else if let Some(inner) = map.get("value") {
                // {"user": "ui_editor_properties_*", "value": X} — user-configurable constant.
                // This is the dominant format in WE scene JSON for all effect parameters.
                // The "user" key is just a reference to the UI property name; we extract the value.
                json_to_animated(inner)
            } else {
                AnimatedValue::default()
            }
        }
        serde_json::Value::Array(arr) => {
            // [r, g, b] arrays used for Vec3 color values
            let floats: Vec<f32> = arr
                .iter()
                .filter_map(|x| x.as_f64().map(|f| f as f32))
                .collect();
            if floats.len() == 3 {
                AnimatedValue::static_val(DynamicValue::Vec3([floats[0], floats[1], floats[2]]))
            } else {
                AnimatedValue::default()
            }
        }
        serde_json::Value::Number(n) => {
            AnimatedValue::static_val(DynamicValue::Float(n.as_f64().unwrap_or(0.0) as f32))
        }
        serde_json::Value::Bool(b) => AnimatedValue::static_val(DynamicValue::Bool(*b)),
        serde_json::Value::String(s) => {
            let parts: Vec<f32> = s
                .split_whitespace()
                .filter_map(|p| p.parse().ok())
                .collect();
            if parts.len() >= 3 {
                AnimatedValue::static_val(DynamicValue::Vec3([parts[0], parts[1], parts[2]]))
            } else if parts.len() == 2 {
                // WE size strings are often "W H" without a z component; treat z as 1
                AnimatedValue::static_val(DynamicValue::Vec3([parts[0], parts[1], 1.0]))
            } else {
                AnimatedValue::static_val(DynamicValue::Str(s.clone()))
            }
        }
        _ => AnimatedValue::default(),
    }
}
