//! Wallpaper Engine scene rendering engine.
//!
//! The production live-render path is [`gpu_renderer::GpuSceneInstance`]; the
//! backend-neutral [`graph::SceneGraph`] feeds the CPU compositor and
//! diagnostics.

pub mod audio;
pub mod cpu;
pub mod gpu;
#[cfg(target_os = "linux")]
pub mod media;
pub mod model;
pub mod playback_gate;
pub mod properties;
pub mod sim;

// Convenience re-exports.
pub use crate::audio::cpu::graph::SceneGraph;
pub use crate::audio::gpu::fbo::{RenderTarget, RenderTargetPool};
pub use crate::audio::model::resolve::ResolvedScene;
pub use crate::audio::model::scene::Scene;
pub use crate::audio::sim::camera_dynamics::{CameraDynamics, CameraFrameDynamics};
pub use crate::formats::pkg::Package;
pub use crate::formats::tex::TexFile;
pub use properties::SceneProperties;
