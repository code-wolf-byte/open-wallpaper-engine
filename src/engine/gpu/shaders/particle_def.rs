use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ControlPoint {
    pub flags: Option<u32>,
    pub id: Option<u32>,
    pub offset: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Emitter {
    pub name: Option<String>,
    pub directions: Option<String>,
    pub distancemax: Option<f64>,
    pub distancemin: Option<f64>,
    pub origin: Option<String>,
    pub rate: Option<f64>,
    pub id: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Initializer {
    pub name: Option<String>,
    pub max: Option<serde_json::Value>,
    pub min: Option<serde_json::Value>,
    pub id: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Operator {
    pub name: Option<String>,
    pub id: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ParticlePreset {
    #[serde(default)]
    pub controlpoint: Vec<ControlPoint>,
    #[serde(default)]
    pub emitter: Vec<Emitter>,
    #[serde(default)]
    pub initializer: Vec<Initializer>,
    #[serde(default)]
    pub operator: Vec<Operator>,
    pub animationmode: Option<serde_json::Value>,
    pub children: Option<serde_json::Value>,
    pub material: Option<String>,
    pub maxcount: Option<u32>,
    pub sequencemultiplier: Option<f64>,
}

pub fn parse_particle_preset(data: &[u8]) -> Result<ParticlePreset> {
    serde_json::from_slice(data).context("parsing particle preset")
}
