use anyhow::Result;
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use std::collections::BTreeMap;

use crate::formats::assets::AssetStore;

pub type TextureMap = BTreeMap<u32, String>;
pub type ComboMap = BTreeMap<String, i32>;
pub type ConstantMap = BTreeMap<String, Value>;

#[derive(Debug, Clone)]
pub struct Model {
    pub filename: String,
    pub material: Material,
    pub solid_layer: bool,
    pub fullscreen: bool,
    pub passthrough: bool,
    pub autosize: bool,
    pub no_padding: bool,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub puppet: Option<String>,
}

impl Model {
    pub fn load(assets: &AssetStore, filename: &str) -> Result<Self> {
        let raw: RawModel = assets.read_json(filename)?;
        let material = Material::load(assets, &raw.material)?;
        Ok(Self {
            filename: filename.to_string(),
            material,
            solid_layer: raw.solid_layer,
            fullscreen: raw.fullscreen,
            passthrough: raw.passthrough,
            autosize: raw.autosize,
            no_padding: raw.no_padding,
            width: raw.width,
            height: raw.height,
            puppet: raw.puppet,
        })
    }

    pub fn first_texture(&self) -> Option<&str> {
        self.material.first_texture()
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RawModel {
    material: String,
    #[serde(default, rename = "solidlayer")]
    solid_layer: bool,
    #[serde(default)]
    fullscreen: bool,
    #[serde(default)]
    passthrough: bool,
    #[serde(default)]
    autosize: bool,
    #[serde(default, rename = "nopadding")]
    no_padding: bool,
    width: Option<u32>,
    height: Option<u32>,
    puppet: Option<String>,
    // Parsed for completeness. `cropoffset` shifts the sampled texture origin
    // (would need per-model UV offset plumbing to apply); `instanced` is a GPU
    // instancing hint — we draw each object normally, so it's a visual no-op.
    #[serde(default)]
    #[allow(dead_code)]
    cropoffset: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    instanced: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct Material {
    pub filename: String,
    pub passes: Vec<MaterialPass>,
}

impl Material {
    pub fn load(assets: &AssetStore, filename: &str) -> Result<Self> {
        let raw: RawMaterial = assets.read_json(filename)?;
        Ok(Self {
            filename: filename.to_string(),
            passes: raw.passes,
        })
    }

    pub fn first_texture(&self) -> Option<&str> {
        self.passes.iter().find_map(MaterialPass::first_texture)
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RawMaterial {
    #[serde(default)]
    passes: Vec<MaterialPass>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MaterialPass {
    #[serde(default = "default_blending")]
    pub blending: String,
    #[serde(default = "default_cullmode")]
    pub cullmode: String,
    #[serde(default = "default_depth_disabled")]
    pub depthtest: String,
    #[serde(default = "default_depth_disabled")]
    pub depthwrite: String,
    // Alpha-channel write mask. Parsed; the 2D compositor writes straight-alpha
    // already, so it has no separate effect here.
    #[serde(default)]
    pub alphawriting: Option<serde_json::Value>,
    pub shader: Option<String>,
    #[serde(default, deserialize_with = "deserialize_texture_map")]
    pub textures: TextureMap,
    #[serde(default, deserialize_with = "deserialize_texture_map")]
    pub usertextures: TextureMap,
    #[serde(default, deserialize_with = "deserialize_combo_map")]
    pub combos: ComboMap,
    #[serde(default, rename = "constantshadervalues")]
    pub constants: ConstantMap,
}

impl MaterialPass {
    pub fn first_texture(&self) -> Option<&str> {
        self.textures
            .get(&0)
            .or_else(|| self.textures.values().next())
            .map(String::as_str)
    }
}

#[derive(Debug, Clone)]
pub struct EffectDefinition {
    pub filename: String,
    pub name: String,
    pub description: String,
    pub group: String,
    pub preview: String,
    pub dependencies: Vec<String>,
    pub passes: Vec<EffectPass>,
    pub fbos: Vec<EffectFbo>,
}

impl EffectDefinition {
    pub fn load(assets: &AssetStore, filename: &str) -> Result<Self> {
        let raw: RawEffectDefinition = assets.read_json(filename)?;
        let mut passes = Vec::with_capacity(raw.passes.len());
        for pass in raw.passes {
            passes.push(pass.resolve(assets)?);
        }

        Ok(Self {
            filename: filename.to_string(),
            name: raw.name,
            description: raw.description,
            group: raw.group,
            preview: raw.preview,
            dependencies: raw.dependencies,
            passes,
            fbos: raw.fbos,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RawEffectDefinition {
    #[serde(default)]
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    group: String,
    #[serde(default)]
    preview: String,
    #[serde(default)]
    dependencies: Vec<String>,
    #[serde(default)]
    passes: Vec<RawEffectPass>,
    #[serde(default)]
    fbos: Vec<EffectFbo>,
}

#[derive(Debug, Clone)]
pub struct EffectPass {
    pub material: Option<Material>,
    pub binds: TextureMap,
    pub command: Option<PassCommand>,
    pub source: Option<String>,
    pub target: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawEffectPass {
    material: Option<String>,
    #[serde(default, rename = "bind", deserialize_with = "deserialize_bind_map")]
    binds: TextureMap,
    command: Option<String>,
    source: Option<String>,
    target: Option<String>,
}

impl RawEffectPass {
    fn resolve(self, assets: &AssetStore) -> Result<EffectPass> {
        let material = self
            .material
            .as_deref()
            .map(|filename| Material::load(assets, filename))
            .transpose()?;

        Ok(EffectPass {
            material,
            binds: self.binds,
            command: self.command.as_deref().map(PassCommand::from_str),
            source: self.source,
            target: self.target,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassCommand {
    Copy,
    Swap,
}

impl PassCommand {
    fn from_str(value: &str) -> Self {
        match value {
            "copy" => Self::Copy,
            _ => Self::Swap,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct EffectFbo {
    pub name: String,
    #[serde(default = "default_fbo_format")]
    pub format: String,
    #[serde(default = "default_fbo_scale")]
    pub scale: f32,
    #[serde(default)]
    pub unique: bool,
}

pub fn parse_texture_map_value(value: &Value) -> TextureMap {
    let Some(items) = value.as_array() else {
        return TextureMap::new();
    };

    let mut textures = TextureMap::new();
    for (index, item) in items.iter().enumerate() {
        if item.is_null() {
            continue;
        }

        let name = if let Some(s) = item.as_str() {
            Some(s)
        } else {
            item.get("name")
                .or_else(|| item.get("file"))
                .and_then(Value::as_str)
        };

        if let Some(name) = name {
            if !name.is_empty() {
                textures.insert(index as u32, name.to_string());
            }
        }
    }

    textures
}

fn deserialize_texture_map<'de, D>(deserializer: D) -> std::result::Result<TextureMap, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    Ok(parse_texture_map_value(&value))
}

fn deserialize_bind_map<'de, D>(deserializer: D) -> std::result::Result<TextureMap, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    let Some(items) = value.as_array() else {
        return Ok(TextureMap::new());
    };

    let mut binds = TextureMap::new();
    for item in items {
        let Some(index) = item.get("index").and_then(Value::as_u64) else {
            continue;
        };
        let Some(name) = item.get("name").and_then(Value::as_str) else {
            continue;
        };
        binds.insert(index as u32, name.to_string());
    }

    Ok(binds)
}

fn deserialize_combo_map<'de, D>(deserializer: D) -> std::result::Result<ComboMap, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = BTreeMap::<String, Value>::deserialize(deserializer)?;
    Ok(raw
        .into_iter()
        .filter_map(|(key, value)| {
            let value = value.as_i64().or_else(|| value.as_str()?.parse().ok())?;
            Some((key, value as i32))
        })
        .collect())
}

fn default_blending() -> String {
    "normal".to_string()
}

fn default_cullmode() -> String {
    "nocull".to_string()
}

fn default_depth_disabled() -> String {
    "disabled".to_string()
}

fn default_fbo_format() -> String {
    "rgba8888".to_string()
}

fn default_fbo_scale() -> f32 {
    1.0
}
