// Agent output (qwen3-coder) — fixed:
//   regex dep removed → ComboAnnotation::parse_combos uses manual serde_json parse
//   UniformDefault::from_json String branch uses bounds-safe .get() indexing
//   from_resolved_glsl wraps Option<serde_json::Value> correctly
use crate::audio::gpu::shaders::uniform_meta::parse_uniform_metadata;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum UniformKind {
    Float,
    Vec2,
    Vec3,
    Vec4,
}

#[derive(Debug, Clone)]
pub enum UniformDefault {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    None,
}

impl UniformDefault {
    pub fn from_json(val: &serde_json::Value, kind: &UniformKind) -> Self {
        match val {
            serde_json::Value::Number(n) => {
                let f = n.as_f64().unwrap_or(0.0) as f32;
                match kind {
                    UniformKind::Float => UniformDefault::Float(f),
                    UniformKind::Vec2 => UniformDefault::Vec2([f, 0.0]),
                    UniformKind::Vec3 => UniformDefault::Vec3([f, 0.0, 0.0]),
                    UniformKind::Vec4 => UniformDefault::Vec4([f, 0.0, 0.0, 0.0]),
                }
            }
            serde_json::Value::Array(arr) => {
                let v: Vec<f32> = arr
                    .iter()
                    .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                    .collect();
                let g = |i| v.get(i).copied().unwrap_or(0.0);
                match kind {
                    UniformKind::Float => UniformDefault::Float(g(0)),
                    UniformKind::Vec2 => UniformDefault::Vec2([g(0), g(1)]),
                    UniformKind::Vec3 => UniformDefault::Vec3([g(0), g(1), g(2)]),
                    UniformKind::Vec4 => UniformDefault::Vec4([g(0), g(1), g(2), g(3)]),
                }
            }
            serde_json::Value::String(s) => {
                let v: Vec<f32> = s
                    .split_whitespace()
                    .filter_map(|x| x.parse().ok())
                    .collect();
                let g = |i| v.get(i).copied().unwrap_or(0.0);
                match kind {
                    UniformKind::Float => UniformDefault::Float(g(0)),
                    UniformKind::Vec2 => UniformDefault::Vec2([g(0), g(1)]),
                    UniformKind::Vec3 => UniformDefault::Vec3([g(0), g(1), g(2)]),
                    UniformKind::Vec4 => UniformDefault::Vec4([g(0), g(1), g(2), g(3)]),
                }
            }
            _ => UniformDefault::None,
        }
    }

    pub fn as_float(&self) -> f32 {
        match self {
            UniformDefault::Float(f) => *f,
            UniformDefault::Vec2([f, _]) => *f,
            UniformDefault::Vec3([f, ..]) => *f,
            UniformDefault::Vec4([f, ..]) => *f,
            UniformDefault::None => 0.0,
        }
    }

    pub fn as_vec4(&self) -> [f32; 4] {
        match self {
            // Scalar defaults broadcast so vec2/vec3 lookups of a scalar
            // annotation (e.g. `"default": 1`) fill every component.
            UniformDefault::Float(f) => [*f; 4],
            UniformDefault::Vec2([a, b]) => [*a, *b, 0.0, 0.0],
            UniformDefault::Vec3([a, b, c]) => [*a, *b, *c, 0.0],
            UniformDefault::Vec4(arr) => *arr,
            UniformDefault::None => [0.0; 4],
        }
    }

    pub fn as_vec3(&self) -> [f32; 3] {
        match self {
            UniformDefault::Float(f) => [*f, 0.0, 0.0],
            UniformDefault::Vec2([a, b]) => [*a, *b, 0.0],
            UniformDefault::Vec3(arr) => *arr,
            UniformDefault::Vec4([a, b, c, _]) => [*a, *b, *c],
            UniformDefault::None => [0.0; 3],
        }
    }
}

#[derive(Debug, Clone)]
pub struct ValueUniform {
    pub glsl_name: String,
    pub material_key: String,
    /// The annotation's `label` (`ui_editor_properties_outline_background`).
    /// Scenes key `constantshadervalues` by EITHER this or `material_key`, and
    /// the two are unrelated strings often enough to matter.
    pub label: Option<String>,
    pub kind: UniformKind,
    pub default: UniformDefault,
}

#[derive(Debug, Clone)]
pub struct TextureSlot {
    pub index: usize,
    pub glsl_name: String,
    pub material_key: Option<String>,
    pub default_path: Option<String>,
    pub is_framebuffer: bool,
    pub mode: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WEBlending {
    Normal,
    Additive,
    Translucent,
    Disabled,
}

impl WEBlending {
    pub fn from_str(s: &str) -> Self {
        match s {
            "normal" => WEBlending::Normal,
            "additive" => WEBlending::Additive,
            "translucent" => WEBlending::Translucent,
            _ => WEBlending::Disabled,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ComboAnnotation {
    pub combo_name: String,
    pub default_value: i32,
}

impl ComboAnnotation {
    /// Parse `// [COMBO] {"combo":"NAME","default":N,...}` lines from GLSL source.
    pub fn parse_combos(glsl_source: &str) -> Vec<ComboAnnotation> {
        let mut results = Vec::new();
        for line in glsl_source.lines() {
            let trimmed = line.trim();
            if !trimmed.starts_with("// [COMBO]") {
                continue;
            }
            let json_start = match trimmed.find('{') {
                Some(i) => i,
                None => continue,
            };
            let json_str = &trimmed[json_start..];
            let val: serde_json::Value = match serde_json::from_str(json_str) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let combo_name = match val.get("combo").and_then(|v| v.as_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let default_value = val.get("default").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            results.push(ComboAnnotation {
                combo_name,
                default_value,
            });
        }
        results
    }
}

/// Central typed model for a resolved WE shader.
/// Built once at wallpaper-load time; referenced throughout the render pipeline.
#[derive(Debug, Clone)]
pub struct ShaderModel {
    pub name: String,
    pub frag_glsl: String,
    pub value_uniforms: Vec<ValueUniform>,
    pub texture_slots: Vec<TextureSlot>,
    pub combos: HashMap<String, i32>,
    pub blending: WEBlending,
    pub combo_annotations: Vec<ComboAnnotation>,
}

impl ShaderModel {
    pub fn from_resolved_glsl(
        name: String,
        frag_glsl: String,
        combos: HashMap<String, i32>,
        blending: WEBlending,
    ) -> Self {
        let metas = parse_uniform_metadata(&frag_glsl);
        let mut value_uniforms = Vec::new();
        let mut texture_slots = Vec::new();
        let mut tex_index = 0usize;

        for meta in &metas {
            match meta.uniform_type.as_str() {
                "sampler2D" => {
                    let default_path = meta
                        .default_value
                        .as_ref()
                        .and_then(|v| v.as_str())
                        .map(str::to_string);
                    // Slot index comes from the NAME (`g_TextureN` -> N), not
                    // declaration order: a material's `textures` array is
                    // positional against those numbers, and shaders do declare
                    // them out of order (blend.frag declares g_Texture7 before
                    // g_Texture1). Counting declarations instead shifted every
                    // later slot, binding the scene's blend texture to the
                    // opacity-mask sampler and leaving the blend texture on its
                    // `util/white` default — which blends the layer to white.
                    let index = meta
                        .uniform_name
                        .strip_prefix("g_Texture")
                        .and_then(|n| n.parse::<usize>().ok())
                        .unwrap_or(tex_index);
                    texture_slots.push(TextureSlot {
                        index,
                        glsl_name: meta.uniform_name.clone(),
                        material_key: meta.material_key.clone(),
                        default_path,
                        is_framebuffer: meta.hidden,
                        mode: meta.mode.clone(),
                    });
                    tex_index += 1;
                }
                type_str => {
                    let kind = match type_str {
                        "float" => UniformKind::Float,
                        "vec2" => UniformKind::Vec2,
                        "vec3" => UniformKind::Vec3,
                        "vec4" => UniformKind::Vec4,
                        _ => continue,
                    };
                    let default = meta
                        .default_value
                        .as_ref()
                        .map(|v| UniformDefault::from_json(v, &kind))
                        .unwrap_or(UniformDefault::None);
                    value_uniforms.push(ValueUniform {
                        glsl_name: meta.uniform_name.clone(),
                        material_key: meta
                            .material_key
                            .clone()
                            .unwrap_or_else(|| meta.uniform_name.clone()),
                        label: meta.label.clone(),
                        kind,
                        default,
                    });
                }
            }
        }

        // Consumers read this positionally (slot 0 = g_Texture0, then the
        // material's texture list index-for-index), so order by slot index
        // rather than the order the shader happened to declare them in.
        texture_slots.sort_by_key(|s| s.index);

        let combo_annotations = ComboAnnotation::parse_combos(&frag_glsl);

        ShaderModel {
            name,
            frag_glsl,
            value_uniforms,
            texture_slots,
            combos,
            blending,
            combo_annotations,
        }
    }

    pub fn framebuffer_slot(&self) -> Option<&TextureSlot> {
        self.texture_slots.iter().find(|s| s.is_framebuffer)
    }

    pub fn extra_texture_slots(&self) -> impl Iterator<Item = &TextureSlot> {
        self.texture_slots.iter().filter(|s| !s.is_framebuffer)
    }

    /// Effective combo values: annotation defaults overridden by explicit combos.
    pub fn effective_combos(&self) -> HashMap<String, i32> {
        let mut result: HashMap<String, i32> = self
            .combo_annotations
            .iter()
            .map(|a| (a.combo_name.clone(), a.default_value))
            .collect();
        result.extend(self.combos.iter().map(|(k, v)| (k.clone(), *v)));
        result
    }

    /// UBO byte size: each value uniform is padded to 16 bytes (std140).
    pub fn uniform_byte_size(&self) -> u32 {
        self.value_uniforms.len() as u32 * 16
    }
}
