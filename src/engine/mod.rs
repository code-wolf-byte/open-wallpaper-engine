//! Wallpaper Engine scene rendering engine.
//!
//! The production live-render path is [`gpu_renderer::GpuSceneInstance`]; the
//! backend-neutral [`graph::SceneGraph`] feeds the CPU compositor and
//! diagnostics.

pub mod assets;
pub mod audio;
pub mod cpu;
pub mod effect;
pub mod gpu;
pub mod graph;
pub mod material;
#[cfg(target_os = "linux")]
pub mod media;
pub mod model;
pub mod pkg;
pub mod playback_gate;
pub mod properties;
pub mod sim;
pub mod tex;

// Convenience re-exports.
pub use crate::engine::gpu::fbo::{RenderTarget, RenderTargetPool};
pub use crate::engine::model::resolve::ResolvedScene;
pub use crate::engine::model::scene::Scene;
pub use crate::engine::sim::camera_dynamics::{CameraDynamics, CameraFrameDynamics};
pub use graph::SceneGraph;
pub use pkg::Package;
pub use properties::SceneProperties;
pub use tex::TexFile;
