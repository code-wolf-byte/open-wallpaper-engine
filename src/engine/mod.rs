//! Wallpaper Engine scene rendering engine.
//!
//! The production live-render path is [`gpu_renderer::GpuSceneInstance`]; the
//! backend-neutral [`graph::SceneGraph`] feeds the CPU compositor and
//! diagnostics.

pub mod animated;
pub mod assets;
pub mod audio;
pub mod blend;
pub mod camera;
pub mod camera3d;
pub mod camera_dynamics;
pub mod effect;
pub mod fbo;
pub mod gpu_renderer;
pub mod graph;
pub mod lighting;
pub mod material;
#[cfg(target_os = "linux")]
pub mod media;
pub mod mesh3d;
pub mod model;
pub mod noise;
pub mod particle;
pub mod pass;
pub mod pkg;
pub mod playback_gate;
pub mod properties;
pub mod puppet;
pub mod render;
pub mod resource;
pub mod scene;
pub mod script;
pub mod shaders;
pub mod shadow;
pub mod tex;
pub mod text;

// Convenience re-exports.
pub use camera::SceneCamera;
pub use camera_dynamics::{CameraDynamics, CameraFrameDynamics};
pub use fbo::{RenderTarget, RenderTargetPool};
pub use graph::SceneGraph;
pub use pass::{ScenePass, UniformValue};
pub use pkg::Package;
pub use properties::SceneProperties;
pub use render::ResolvedScene;
pub use resource::ResourceManager;
pub use scene::Scene;
pub use tex::TexFile;
