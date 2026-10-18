// Agent output (qwen3-coder) — lightly corrected for field alignment with scene.rs.
use super::dynamic_value::AnimatedValue;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SceneModel {
    pub camera_eye: [f32; 3],
    pub ambient_color: [f32; 3],
    pub objects: Vec<ObjectModel>,
}

#[derive(Debug, Clone)]
pub struct ObjectModel {
    pub index: usize,
    pub name: String,
    pub image: Option<String>,
    pub origin: AnimatedValue,     // Vec3
    pub scale: AnimatedValue,      // Vec3
    pub angles: AnimatedValue,     // Vec3
    pub alpha: AnimatedValue,      // Float
    pub color: AnimatedValue,      // Vec3
    pub brightness: AnimatedValue, // Float
    pub visible: AnimatedValue,    // Bool
    pub blend_mode: u32,
    pub effects: Vec<EffectModel>,
    pub particle: Option<serde_json::Value>,
    pub copybackground: bool,
}

impl ObjectModel {
    pub fn is_visible(&self) -> bool {
        self.visible.as_bool().unwrap_or(true)
    }

    pub fn origin_xyz(&self) -> [f32; 3] {
        self.origin.as_vec3().unwrap_or([0.0; 3])
    }

    pub fn scale_xyz(&self) -> [f32; 3] {
        self.scale.as_vec3().unwrap_or([1.0, 1.0, 1.0])
    }

    pub fn alpha_f32(&self) -> f32 {
        self.alpha.as_float().unwrap_or(1.0)
    }

    pub fn brightness_f32(&self) -> f32 {
        self.brightness.as_float().unwrap_or(1.0)
    }

    pub fn color_rgb(&self) -> [f32; 3] {
        self.color.as_vec3().unwrap_or([1.0, 1.0, 1.0])
    }
}

#[derive(Debug, Clone)]
pub struct EffectModel {
    pub file: Option<String>,
    pub name: String,
    pub visible: bool,
    pub passes: Vec<PassModel>,
}

#[derive(Debug, Clone)]
pub struct PassModel {
    pub combos: HashMap<String, i32>,
    pub shader_values: HashMap<String, AnimatedValue>,
    /// Positional, `None` for a `null`/absent slot — see `scene::Pass::textures`.
    pub textures: Vec<Option<String>>,
}
