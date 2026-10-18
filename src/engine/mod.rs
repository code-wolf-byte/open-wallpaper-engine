//! Wallpaper Engine scene rendering engine.
//!
//! The production live-render path is [`gpu_renderer::GpuSceneInstance`]; the
//! backend-neutral [`graph::SceneGraph`] feeds the CPU compositor and
//! diagnostics.

pub mod cpu;
pub mod gpu;
#[cfg(target_os = "linux")]
pub mod media;
pub mod model;
pub mod playback_gate;
pub mod properties;
pub mod sim;

// Convenience re-exports.
pub use crate::engine::cpu::graph::SceneGraph;
pub use crate::engine::gpu::fbo::{RenderTarget, RenderTargetPool};
pub use crate::engine::model::resolve::ResolvedScene;
pub use crate::engine::model::scene::Scene;
pub use crate::engine::sim::camera_dynamics::{CameraDynamics, CameraFrameDynamics};
pub use crate::formats::pkg::Package;
pub use crate::formats::tex::TexFile;
pub use properties::SceneProperties;
