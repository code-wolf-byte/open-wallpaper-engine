//! The scene engine: turns a Wallpaper Engine scene into frames.
//!
//! - [`model`]: what the scene is — `scene.json` types, resolving them into
//!   layers, materials and shader metadata, static meshes
//! - [`sim`]: what changes over time — particles, timelines, SceneScript,
//!   camera motion, puppet animation
//! - [`gpu`]: how it is drawn — the renderer
//!   ([`gpu::renderer::GpuSceneInstance`] is the live path), render targets,
//!   blending, lighting, shadows, text, and the GLSL→WGSL shader pipeline
//! - [`cpu`]: the CPU fallback renderer, used when the GPU path can't start
//!
//! File formats live in [`crate::formats`], user properties in
//! [`crate::workshop::properties`].

pub mod cpu;
pub mod gpu;
pub mod model;
pub mod sim;

pub use cpu::graph::SceneGraph;
pub use model::resolve::ResolvedScene;
