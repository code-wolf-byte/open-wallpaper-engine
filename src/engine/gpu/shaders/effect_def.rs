use super::resolver::AssetResolver;
use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct EffectPass {
    pub material: Option<String>,
    /// Named FBO to render into instead of the effect ping-pong chain.
    pub target: Option<String>,
    /// Bindings of named FBOs to texture slots for this pass.
    #[serde(default)]
    pub bind: Vec<EffectBind>,
}

/// One `bind` entry in an effect pass: exposes FBO `name` as `g_Texture{index}`.
#[derive(Debug, Deserialize)]
pub struct EffectBind {
    pub name: Option<String>,
    pub index: Option<u32>,
}

/// A named auxiliary framebuffer declared by an effect (e.g. half-res buffers).
#[derive(Debug, Deserialize)]
pub struct EffectFbo {
    pub name: Option<String>,
    /// Downscale divisor relative to the layer/scene size (1 = full size).
    pub scale: Option<f64>,
    pub format: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct EffectDef {
    pub version: Option<u32>,
    pub name: Option<String>,
    pub group: Option<String>,
    #[serde(default)]
    pub passes: Vec<EffectPass>,
    #[serde(default)]
    pub fbos: Vec<EffectFbo>,
    pub dependencies: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct MaterialPass {
    pub shader: Option<String>,
    pub blending: Option<String>,
    pub depthtest: Option<String>,
    pub depthwrite: Option<String>,
    pub cullmode: Option<String>,
    #[serde(default)]
    pub textures: Vec<Option<String>>,
    /// Shader combo defines set by the material (e.g. `{"KERNEL": 1}`).
    #[serde(default)]
    pub combos: std::collections::HashMap<String, i32>,
    /// Material-level constant defaults, keyed by the shader annotation's
    /// `material` name (overridden by scene.json pass constants).
    #[serde(default, rename = "constantshadervalues")]
    pub constant_shader_values: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct MaterialDef {
    #[serde(default)]
    pub passes: Vec<MaterialPass>,
}

pub fn parse_effect_json(data: &[u8]) -> Result<EffectDef> {
    serde_json::from_slice(data).context("parsing effect.json")
}

pub fn parse_material_json(data: &[u8]) -> Result<MaterialDef> {
    serde_json::from_slice(data).context("parsing material JSON")
}

/// Load an effect.json by its verbatim scene.json `file` path (e.g.
/// `"effects/clouds/effect.json"` or a workshop-nested
/// `"effects/workshop/2138904733/cutout_vignette/effect.json"`) — WE always
/// gives the exact path, so it must be used as-is rather than reconstructed
/// from a guessed effect name.
pub fn load_effect_by_file(resolver: &AssetResolver, file: &str) -> Result<EffectDef> {
    let data = resolver
        .read(file)
        .with_context(|| format!("reading {file}"))?;
    parse_effect_json(&data)
}

pub fn load_material_from_dir(
    resolver: &AssetResolver,
    material_path: &str,
) -> Result<MaterialDef> {
    let data = resolver
        .read(material_path)
        .with_context(|| format!("reading {material_path}"))?;
    parse_material_json(&data)
}

/// Load a pass's material, checking the effect bundle directory
/// (`{effect_dir}/{material_path}`, e.g. built-ins like
/// `effects/clouds/materials/effects/clouds.json`) before falling back to
/// `material_path` as a root-relative asset path (the convention used by
/// workshop-local effects, whose material path is already root-relative).
pub fn load_material_from_effect(
    resolver: &AssetResolver,
    effect_dir: &str,
    material_path: &str,
) -> Result<MaterialDef> {
    let bundled = format!("{effect_dir}/{material_path}");
    if let Some(data) = resolver.read(&bundled) {
        return parse_material_json(&data);
    }
    load_material_from_dir(resolver, material_path)
}
