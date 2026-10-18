//! Wallpaper Engine scene rendering engine.
//!
//! The production live-render path is [`gpu_renderer::GpuSceneInstance`]; the
//! backend-neutral [`graph::SceneGraph`] feeds the CPU compositor and
//! diagnostics.

pub mod animated;
pub mod assets;
pub mod audio;
pub mod effect;
pub mod gpu;
pub mod graph;
pub mod material;
#[cfg(target_os = "linux")]
pub mod media;
pub mod mesh3d;
pub mod model;
pub mod particle;
pub mod pkg;
pub mod playback_gate;
pub mod properties;
pub mod puppet;
pub mod render;
pub mod scene;
pub mod script;
pub mod sim;
pub mod tex;

// Convenience re-exports.
pub use crate::engine::gpu::fbo::{RenderTarget, RenderTargetPool};
pub use crate::engine::sim::camera_dynamics::{CameraDynamics, CameraFrameDynamics};
pub use graph::SceneGraph;
pub use pkg::Package;
pub use properties::SceneProperties;
pub use render::ResolvedScene;
pub use scene::Scene;
pub use tex::TexFile;
