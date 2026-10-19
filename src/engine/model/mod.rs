//! What a scene is: the `scene.json` types ([`scene`]), resolving them into
//! drawable layers ([`resolve`]), static `.mdl` meshes, and the typed
//! object/shader model with its per-frame dynamic values.

pub mod dynamic_value;
pub mod mesh3d;
pub mod objects;
pub mod parser;
pub mod property;
pub mod resolve;
pub mod scene;
pub mod shader_model;

pub use dynamic_value::{AnimatedValue, DynamicValue};
pub use objects::{EffectModel, ObjectModel, PassModel, SceneModel};
pub use parser::{json_to_animated, scene_to_model};
pub use property::{Property, PropertyKind, UserSetting};
pub use shader_model::{
    ComboAnnotation, ShaderModel, TextureSlot, UniformDefault, UniformKind, ValueUniform,
    WEBlending,
};
