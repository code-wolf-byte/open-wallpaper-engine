//! GPU scene renderer: pipelines, effect/FBO pass chaining, and the
//! per-scene [`GpuSceneInstance`] runtime.
//!
//! Pass model (mirrors linux-wallpaperengine's CWallpaper/CPass chain):
//! - every layer's base texture runs through its effect passes, ping-ponging
//!   between two persistent scene-sized render targets;
//! - effect passes may render into named FBOs (`effect.json` `fbos` +
//!   `target`) and re-bind them as texture slots (`bind`);
//! - each layer is then composited into the scene target with its blend mode,
//!   camera-dynamics UV offset (shake + parallax) and fade opacity;
//! - when scene bloom is enabled a threshold → blur chain runs over
//!   quarter/eighth-res buffers (`_rt_4FrameBuffer` / `_rt_8FrameBuffer`)
//!   and is added back onto the scene target.

use anyhow::{anyhow, Context, Result};
use image::RgbaImage;
use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::SyncSender;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::engine::gpu::fbo::RenderTargetPool;
use crate::engine::gpu::shaders::{effect_def, loader, resolver, transpiler};
use crate::engine::model::scene::{parse_value_bool, parse_value_f32};
use crate::engine::model::{ShaderModel, WEBlending};
use crate::engine::sim::camera_dynamics::{CameraDynamics, CameraFrameDynamics};
use crate::engine::sim::particle;
use crate::platform::GpuDevice;

mod params;
use params::*;

mod pipelines;
use pipelines::*;

mod lights;
use lights::*;

mod draw;
use draw::*;

const SHADER_SRC: &str = include_str!("gpu_shaders.wgsl");

/// Per-layer ping-pong buffer key in the FBO pool (each layer's effect chain
/// gets its own pair, sized to that layer's object size).
fn pingpong_key(layer_idx: usize, slot: u8) -> String {
    format!("_rt_pingpong{slot}_{layer_idx}")
}

/// Built-in effects with handwritten WGSL kernels.
const HARDCODED_EFFECTS: &[&str] = &[
    "pulse",
    "scroll",
    "shake",
    "tint",
    "opacity",
    "waterripple",
    "waterwaves",
    "spin",
];

pub struct GpuSceneRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Indexed by `(no_interpolation, clamp_uvs)` — the `.tex` `NoInterpolation`
    /// and `ClampUVs`/`ClampUVsBorder` flags select nearest-vs-linear filtering
    /// and clamp-vs-repeat addressing per layer.
    samplers: [wgpu::Sampler; 4],
    base_bgl: wgpu::BindGroupLayout,
    effect_bgl: wgpu::BindGroupLayout,
    /// The g_AudioSpectrum* UBO (effect_bgl binding 1), rewritten each frame
    /// from the FFT. Zero-filled when no audio is captured.
    audio_buf: wgpu::Buffer,
    composite_pipelines: Vec<(u32, wgpu::RenderPipeline)>,
    /// Same vs_composite_quad/fs_composite pair as `composite_pipelines[0]`
    /// but with blending disabled (plain overwrite): used for the intra-chain
    /// base material pass, which always renders into a freshly-cleared,
    /// transparent object FBO. Reusing the alpha-blended composite pipeline
    /// there would double-attenuate rgb by alpha (once via the shader's own
    /// `s.rgb * opacity` and again via the GPU blend equation).
    base_pass_pipeline: wgpu::RenderPipeline,
    effect_pipelines: Vec<(&'static str, wgpu::RenderPipeline)>,
    dynamic_pipelines: HashMap<String, wgpu::RenderPipeline>,
    // pipeline key → UBO layout (uniform key, glsl_type, typed default)
    dynamic_uniform_keys: HashMap<String, Vec<transpiler::UniformEntry>>,
    // pipeline key → vertex attributes (glsl type, name); empty = fullscreen
    // triangle with no vertex buffers (synthetic VS path)
    dynamic_vertex_attrs: HashMap<String, Vec<(String, String)>>,
    dynamic_textures: HashMap<String, Vec<wgpu::Texture>>,
    // (kind, dims) → shared quad attribute buffer (kind: 0=pos NDC, 1=uv, 2=zero)
    attr_buffers: HashMap<(u8, u8), wgpu::Buffer>,
    composite_pipeline_layout: wgpu::PipelineLayout,
    effect_pipeline_layout: wgpu::PipelineLayout,
    shader_module: wgpu::ShaderModule,
    dummy_tex: wgpu::Texture,
    // Scene bloom chain
    bloom_threshold_pipeline: wgpu::RenderPipeline,
    bloom_blur_pipeline: wgpu::RenderPipeline,
    bloom_combine_pipeline: wgpu::RenderPipeline,
    // Format → blit pipeline (surface presentation, FBO up/down-sampling)
    blit_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    // GPU particle pipeline (vs_particles/fs_particles): storage-buffer
    // vertices, texture-array sprites, hardware blending straight into the
    // scene target — replaces the budgeted CPU particle rasterizer.
    particle_bgl: wgpu::BindGroupLayout,
    particle_pipeline_add: wgpu::RenderPipeline,
    particle_pipeline_over: wgpu::RenderPipeline,
    // Static 3D mesh pipeline (vs_mesh3d/fs_mesh3d): real indexed geometry
    // through the scene camera, and the only pipeline here with a depth
    // buffer — a sphere self-occludes, which no painter's ordering can fake.
    mesh3d_bgl: wgpu::BindGroupLayout,
    /// Indexed by the material's `nocull`: back-face culling is what makes a
    /// skybox work (its near hemisphere culls away instead of hiding the
    /// scene inside it), while `"cullmode": "nocull"` materials — hollow
    /// shells meant to be seen from both sides — must keep every face.
    mesh3d_pipelines: [wgpu::RenderPipeline; 4],
    // Shadow-caster depth pass: see `engine::gpu::shadow` and
    // `GpuSceneInstance::build_shadow_atlas`.
    shadow_pass_bgl: wgpu::BindGroupLayout,
    mesh3d_shadow_pipeline: wgpu::RenderPipeline,
    shadow_comparison_sampler: wgpu::Sampler,
    // Volumetric-light-shaft ray march: see `fs_volumetrics`,
    // `GpuSceneInstance::build_volumetrics`/`record_volumetrics`.
    volumetrics_bgl: wgpu::BindGroupLayout,
    volumetrics_pipeline: wgpu::RenderPipeline,
    // Perspective-projected 2D quads: see `vs_quad3d`/`fs_quad3d`.
    quad3d_bgl: wgpu::BindGroupLayout,
    quad3d_pipeline: wgpu::RenderPipeline,
}

/// Depth format for the 3D mesh pass. Depth32Float is guaranteed everywhere
/// wgpu runs, so there's no fallback to pick.
const MESH3D_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

// ── Effect runtimes ───────────────────────────────────────────────────────────

/// Scene.json overrides for one effect pass: shader combos plus constant
/// values (indexed by material key, with normalized aliases and per-component
/// `_r/_g/_b`/`_x/_y/_z/_w` entries).
#[derive(Debug, Clone, Default)]
struct PassOverride {
    combos: HashMap<String, i32>,
    values: ShaderVals,
    /// Positional scene.json texture override for this pass (e.g. a custom
    /// opacity mask or noise map); `None` entries mean "leave this slot at
    /// the material's own default". See `scene::Pass::textures`.
    textures: Vec<Option<String>>,
}

/// One effect attached to one layer (an *instance* — the same effect on two
/// layers gets separate pipelines because combos/constants can differ).
struct EffectInstanceDef {
    layer_idx: usize,
    /// Position in the object's own `effects` list (scripts address effects
    /// by it: `thisLayer.getEffect(1)`).
    effect_index: usize,
    name: String,
    /// Verbatim scene.json effect `file` path (e.g. `"effects/clouds/effect.json"`
    /// or a workshop-nested `"effects/workshop/2138904733/foo/effect.json"`) —
    /// used as-is rather than reconstructed from `name`, since custom-effect
    /// bundles don't follow the `effects/{name}/` convention.
    file: String,
    pass_overrides: Vec<PassOverride>,
}

/// One executable pass of an effect.
struct EffectPassRuntime {
    /// Pipeline lookup key: the effect name for hardcoded kernels,
    /// `"fx{instance}:{effect}#{pass}"` for translated WE shaders.
    key: String,
    /// `dynamic_textures` lookup key. Equals `key` for translated shaders;
    /// hardcoded kernels get their own `"fx{instance}:{effect}#0"` since
    /// their pipeline `key` (the bare effect name) is shared across every
    /// instance and would collide (e.g. two masked waterwaves on one layer).
    tex_key: String,
    hardcoded: bool,
    /// Named FBO to render into (skips the ping-pong advance).
    target: Option<String>,
    /// (texture slot index, FBO name) bindings.
    binds: Vec<(u32, String)>,
    /// Resolved constants: material defaults overridden by scene values.
    values: ShaderVals,
    /// Quad attribute buffers for the real-VS path; empty = synthetic VS
    /// (fullscreen triangle, no vertex buffers).
    vertex_buffers: Vec<wgpu::Buffer>,
}

/// A fully loaded effect instance: its passes plus auxiliary FBO declarations.
struct EffectRuntime {
    passes: Vec<EffectPassRuntime>,
    /// (name, downscale divisor) — allocated per instance in the pool.
    fbos: Vec<(String, f32)>,
    /// True when any pass reads `g_AudioSpectrum*` — drives lazy capture start.
    uses_audio: bool,
}

/// Per-frame values for WE's engine-provided uniforms (the counterpart of
/// CPass::setupUniforms in the reference).
#[derive(Debug, Clone, Copy)]
struct EngineUniforms {
    time: f32,
    /// Fraction of the day in [0,1).
    daytime: f32,
    texel_size: [f32; 2],
    pointer: [f32; 2],
    /// g_Texture{N}Resolution: (tex_w, tex_h, real_w, real_h) per slot.
    resolutions: [[f32; 4]; 8],
    color: [f32; 3],
    alpha: f32,
    brightness: f32,
}

impl EngineUniforms {
    /// Look up an engine uniform by its GLSL name. Returns up to 4 components.
    fn get(&self, key: &str) -> Option<[f32; 4]> {
        let v = |a: f32, b: f32, c: f32, d: f32| Some([a, b, c, d]);
        match key {
            "g_Time" | "g_AnimationTime" | "g_FrameTime" => v(self.time, 0.0, 0.0, 0.0),
            "g_Daytime" => v(self.daytime, 0.0, 0.0, 0.0),
            "g_TexelSize" => v(self.texel_size[0], self.texel_size[1], 0.0, 0.0),
            "g_TexelSizeHalf" => v(self.texel_size[0] * 0.5, self.texel_size[1] * 0.5, 0.0, 0.0),
            "g_PointerPosition" | "g_PointerPositionLast" => {
                v(self.pointer[0], self.pointer[1], 0.0, 0.0)
            }
            "g_Color" | "g_CompositeColor" | "g_TintColor" => {
                v(self.color[0], self.color[1], self.color[2], 1.0)
            }
            "g_Color4" => v(self.color[0], self.color[1], self.color[2], self.alpha),
            "g_Alpha" | "g_UserAlpha" | "g_BlendAlpha" | "g_CompositeAlpha" => {
                v(self.alpha, 0.0, 0.0, 0.0)
            }
            "g_Brightness" => v(self.brightness, 0.0, 0.0, 0.0),
            "g_TextureReductionScale" => v(1.0, 1.0, 1.0, 1.0),
            _ => {
                if let Some(rest) = key.strip_prefix("g_Texture") {
                    if let Some(idx) = rest.strip_suffix("Resolution") {
                        let i: usize = idx.parse().ok()?;
                        return self.resolutions.get(i).copied();
                    }
                }
                None
            }
        }
    }
}

/// Scene-level bloom settings (general.bloom / bloomstrength / bloomthreshold).
#[derive(Debug, Clone, Default)]
struct BloomSettings {
    enabled: bool,
    strength: f32,
    threshold: f32,
}

struct SceneLayerGpu {
    frames: Vec<wgpu::Texture>,
    /// Animated puppet runtime — `frames[0]` is re-rasterized from the
    /// posed mesh at `PUPPET_UPDATE_INTERVAL` (skinning + CPU raster costs
    /// ~30ms at 4K, so it runs at a capped rate rather than per frame;
    /// idle-style puppet animations are slow enough not to notice).
    puppet: Option<std::sync::Arc<crate::engine::sim::puppet::PuppetRuntime>>,
    /// Last animation time `frames[0]` was posed at, seconds.
    puppet_posed_at: f32,
    /// Embedded-video stream — `frames[0]` is replaced with the newest
    /// decoded frame each tick (the decoder thread paces itself by PTS;
    /// draining to the latest frame drops rather than lags).
    video:
        Option<std::sync::Arc<std::sync::Mutex<crate::engine::model::resolve::VideoLayerStream>>>,
    blend_mode: u32,
    /// Live alpha; SceneScript may change it per frame.
    alpha: f32,
    /// Script-driven text: re-evaluated each tick; when the output string
    /// changes, `frames[0]` is re-rasterized and `rect` re-derived (clocks
    /// tick, dates roll over — 42 of the corpus's 58 script wallpapers).
    text_dynamic: Option<crate::engine::model::resolve::TextDynamic>,
    color: [f32; 3],
    brightness: f32,
    frame_duration_ms: u32,
    parallax_depth: [f32; 2],
    /// z-rotation in radians (WE `angles.z`).
    angle: f32,
    /// Object quad: (center_ndc.x, center_ndc.y, half_extent_ndc.x, half_extent_ndc.y).
    rect: [f32; 4],
    /// True perspective-projected quad corners (NDC, already divided) in
    /// `project_quad_ndc`'s corner order — `Some` only for perspective-scene
    /// layers where every corner projects in front of the camera. `rect`
    /// above is still always populated (their axis-aligned bounding box, via
    /// `project_quad_ndc`) since depth-sort and the fallback draw path both
    /// need it regardless. See `project_quad_corners` and the Ghidra
    /// report's quad-perspective-warp finding: without this, an off-axis
    /// quad's silhouette collapses to a rectangle instead of the true
    /// trapezoid. Only consumed when `blend_mode == 0` (normal) — non-normal
    /// blend modes on perspective quads keep the existing rect-based
    /// approximation, a narrow and rare combination not worth a second
    /// pipeline variant per blend mode.
    quad_corners: Option<[[f32; 2]; 4]>,
    /// The object's own visibility (authored, then as scripts set it). The
    /// layer draws only when it and every ancestor are visible — see
    /// `GpuSceneInstance::object_visible`.
    visible: bool,
    /// Object size in px before scale — rebuilds `rect` when a script
    /// changes scale.
    effective_size: [f64; 2],
    /// Text a script set but not yet rasterized (throttled — see
    /// `flush_text`), and when the last raster happened and how long it took.
    pending_text: Option<String>,
    text_rastered_at: Option<Instant>,
    text_raster_cost: Duration,
    /// Composition/fullscreen layer: `frames[0]` is refilled each frame
    /// with the scene drawn beneath it (see `capture_backdrop`).
    backdrop: Option<crate::engine::model::resolve::Backdrop>,
    /// Spritesheet frame a script picked (`getTextureAnimation().setFrame`),
    /// overriding the timed animation.
    script_frame: Option<usize>,
    /// `alignment`'s shift of the quad centre from the authored origin, kept
    /// so a scripted origin moves the quad the same way.
    align_offset: [f64; 3],
    no_interpolation: bool,
    clamp_uvs: bool,
    /// Unscaled object size in pixels — matches the reference's `CImage::m_size`
    /// (explicit scene.json `size`, falling back to the texture's native
    /// resolution). Effect-chain FBOs (ping-pong + named targets) are sized to
    /// this, not the scene resolution, per `CImage.cpp`/`FBOProvider::create`.
    object_size: (u32, u32),
    /// This layer's position in `scene.visible_objects()` — lets `render()`
    /// interleave with particle systems in true scene z-order.
    order_index: usize,
    /// The scene object's `id`, for `_rt_imageLayerComposite_<id>_*` binds.
    object_id: Option<i64>,
    /// Perspective scenes only: view-space distance of the quad center from
    /// the camera, for painter's-algorithm back-to-front sorting. Negative =
    /// behind the camera (culled). Always 0.0 in orthographic scenes.
    depth: f32,
}

// ── GpuSceneInstance ──────────────────────────────────────────────────────────

/// A loaded scene ready to render frames on the GPU.
///
/// Owns the renderer, layer textures, effect runtimes, the persistent FBO
/// pool, and per-frame camera dynamics. Render either to RGBA (readback
/// paths: preview/testing/SHM fallback) or straight into an external texture
/// view (Wayland surface presentation — no readback).
/// One sprite texture array for a GPU particle draw unit.
struct ParticleGpuTex {
    view: wgpu::TextureView,
    frames: u32,
    overbright: f32,
    /// Ropes tile V past 1.0 (`uvscale`/scrolling) — sample with the repeat
    /// sampler; sprite quads clamp.
    repeat: bool,
}

/// Per-layer GPU particle textures: the parent system's sprite plus one per
/// child preset (child *instances* spawn at runtime; textures are per child).
struct ParticleGpuAssets {
    parent: ParticleGpuTex,
    children: Vec<ParticleGpuTex>,
}

/// A scene `sound` object's playback stream, kept with its owner's identity
/// so SceneScript can find it by layer name or id.
pub(crate) struct SceneSound {
    /// Index of the owning object in `scene.objects` (or a script-created one).
    pub object: usize,
    pub playback: crate::services::audio::SoundPlayback,
}

/// Start a `sound` object's first decodable file, held silent when
/// `startsilent` is set.
fn start_scene_sound(
    dir: &Path,
    obj: &crate::engine::model::scene::SceneObject,
) -> Option<crate::services::audio::SoundPlayback> {
    let sound = obj.sound.as_ref()?;
    let start_silent = obj
        .startsilent
        .as_ref()
        .and_then(|v| {
            v.as_bool()
                .or_else(|| v.get("value").and_then(|i| i.as_bool()))
        })
        .unwrap_or(false);
    let volume = obj
        .volume
        .as_ref()
        .and_then(crate::engine::model::scene::parse_value_f32)
        .unwrap_or(1.0);
    let mode = obj.playbackmode.as_ref().and_then(|v| v.as_str());
    let looping = mode != Some("nointerrupt") && mode != Some("once");
    let files: Vec<String> = match sound {
        serde_json::Value::Array(a) => a
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        serde_json::Value::String(s) => vec![s.clone()],
        _ => Vec::new(),
    };
    files.into_iter().find_map(|f| {
        crate::services::audio::SoundPlayback::start(&dir.join(&f), volume, looping, !start_silent)
    })
}

pub struct GpuSceneInstance {
    renderer: GpuSceneRenderer,
    layers: Vec<SceneLayerGpu>,
    scene_effects: Vec<EffectInstanceDef>,
    /// Parallel to `scene_effects`; None = the instance failed to load.
    effect_runtimes: Vec<Option<EffectRuntime>>,
    fbo_pool: RenderTargetPool,
    target: wgpu::Texture,
    /// Snapshot of `target` before each colorBlendMode>0 layer composites, so
    /// the dest-read Photoshop blend can sample "what's already there"
    /// without reading and writing the same attachment in one pass.
    scene_copy: wgpu::Texture,
    width: u32,
    height: u32,
    clear_color: [f64; 3],
    dynamics: CameraDynamics,
    bloom: BloomSettings,
    /// The volumetric-light-shaft ray march's static per-scene bind group —
    /// `None` unless a light sets `castvolumetrics` (see `build_volumetrics`,
    /// `record_volumetrics`, and the Ghidra report's `_rt_volumetrics*`
    /// follow-up). Static because the camera and every light's world
    /// transform are themselves static in this engine (mesh3d/lighting are
    /// baked once at load, not per-frame — same precedent `mesh3d_lighting_ubo`
    /// already sets).
    volumetrics_bind_group: Option<wgpu::BindGroup>,
    /// Each owns independent simulation state; stepped and CPU-rendered into a
    /// scene-sized RGBA buffer every frame, then uploaded and composited like
    /// any other layer (see `render()` step 3). A known simplification: real
    /// particle objects interleave into the scene's actual render order, but
    /// these always draw on top of images (before bloom, so bright particles
    /// still contribute to the bloom pass) — see [[wp_engine_project]] memory.
    particle_systems: Vec<particle::ParticleSystem>,
    /// Parallel to `particle_systems`: each system's resolved sprite (from
    /// its config's `material`, if any), sampled CPU-side by `render_onto`'s
    /// textured path. `None` falls back to flat-color circles.
    particle_sprites: Vec<Option<particle::ParticleSprite>>,
    /// Parallel to `particle_systems`: each system's `order_index` (from its
    /// source `ParticleLayer`), so `render()` can interleave particles with
    /// image layers in true scene z-order.
    particle_order: Vec<usize>,
    /// Parallel to `particle_systems`: true when the resolved sprite's
    /// material declares `"blending":"additive"` (fog/smoke/embers/rain/
    /// lightning — most real particle materials). Composited additively
    /// instead of the default alpha-over, which otherwise makes a sprite's
    /// near-black background visibly darken/box the scene behind it.
    particle_additive: Vec<bool>,
    /// Parallel to `particle_systems`: GPU sprite texture arrays (parent +
    /// per-child), built once at load for the GPU particle pipeline.
    particle_gpu_assets: Vec<ParticleGpuAssets>,
    start: Instant,
    last_time: f32,
    mouse_norm: [f32; 2],
    /// The scene's SceneScript runtime, ticked once per frame. Lives here (on
    /// the render thread) because boa's `Context` is `!Send`.
    scripts: crate::engine::sim::script::ScriptHost,
    /// What live user-property edits diff against.
    live_props: LiveProps,
    /// Per-object visibility as scripts leave it, indexed like `scene.objects`
    /// (authored visibility at load). A layer draws only when its object and
    /// every ancestor are visible.
    object_visible: Vec<bool>,
    /// Live `disablepropagation()` state per object.
    propagation_disabled: Vec<bool>,
    /// Whether a script registered audio buffers (feeds them each frame).
    scripts_want_audio: bool,
    /// Last cursor position in scene pixels, and button state — for
    /// `input.*` and cursor events.
    cursor_scene: [f32; 2],
    cursor_left_down: bool,
    /// Left-button transitions since the last frame, in order, so a click
    /// that starts and ends between two frames still registers.
    pending_buttons: Vec<bool>,
    /// Objects with cursor handlers (`cursorClick` & co), hit-tested each
    /// frame; empty for the vast majority of scenes.
    cursor_targets: Vec<usize>,
    /// The handler object under the cursor last frame, and the one a press
    /// started on (receives the matching `cursorUp`/`cursorClick`).
    hovered: Option<usize>,
    pressed: Option<usize>,
    last_cursor_scene: [f32; 2],
    /// Keyframed property animations (`"animation"` timelines).
    timelines: Vec<TimelineEntry>,
    /// Parallel to `scene_effects`: false once a script hides the effect.
    effect_enabled: Vec<bool>,
    /// Per-object draw position; starts as the object index, and
    /// `thisScene.sortLayer` moves objects between their neighbours.
    draw_rank: Vec<f64>,
    sort_count: u32,
    /// For `thisScene.createLayer`: the scene directory and its package.
    scene_dir: std::path::PathBuf,
    scene_pkg: Option<crate::formats::pkg::Package>,
    /// Desktop-audio capture, started only when an effect reads
    /// `g_AudioSpectrum*` (else `None` and the spectrum stays silent). Polled
    /// once per frame into the renderer's `audio_buf`.
    audio_capture: Option<crate::services::audio::AudioCapture>,
    /// Live playback streams for `sound` objects; kept alive for the scene's
    /// lifetime (dropping stops the audio).
    sounds: Vec<SceneSound>,
    /// MPRIS "what's playing" watcher (`services::media`) — Linux-only (see
    /// its own module doc). Started unconditionally rather than gated on
    /// whether any script actually reads `media.*` (matching
    /// `engine.runtime`'s own always-on treatment): the D-Bus connection is
    /// a cheap background poll, and unlike the CEF bridge's `wants_media`
    /// scan there's no equivalent "does this bundle reference it" signal to
    /// scan for across independent per-property scripts.
    #[cfg(target_os = "linux")]
    media: Option<crate::services::media::MediaWatcher>,
    /// True for genuine 3D scenes (`Scene::is_perspective`): layer rects were
    /// projected through a perspective camera and `render()` sorts image
    /// layers back-to-front by `SceneLayerGpu::depth` instead of scene order.
    perspective: bool,
    /// Static 3D meshes (`model` -> .mdl), drawn as real geometry before the
    /// 2D layers composite over them. Empty for every scene that has none.
    mesh3d: Vec<Mesh3dGpu>,
    /// Depth buffer for `mesh3d`; `None` when there are no meshes. The 2D
    /// layers never touch it — they keep their painter's ordering.
    mesh3d_depth: Option<wgpu::TextureView>,
    /// Scene-wide lighting/fog state, shared by every mesh3d draw call's
    /// bind group (binding 3) rather than duplicated per mesh. Built once at
    /// scene setup: `Scene::lights()`/`general.fog()` have no per-frame
    /// animation path today, matching `build_mesh3d`'s own "static in real
    /// content" assumption for MVP. Always allocated, even for scenes with no
    /// meshes — the cost is one small buffer, and it keeps this field
    /// unconditional rather than `Option`. Not read back after construction —
    /// kept alive here (rather than as a `build_mesh3d`-local) so its
    /// lifetime is explicit and it's in place for a future per-frame rebuild
    /// if scripted-parent lights ever need one, matching how `Mesh3dGpu::ubo`
    /// already gets rewritten for scripted-parent meshes.
    #[allow(dead_code)]
    mesh3d_lighting_ubo: wgpu::Buffer,
    /// Shared shadow atlas texture (see `build_shadow_atlas`) — kept alive
    /// for the same reason as `mesh3d_lighting_ubo`: every mesh3d bind group
    /// holds a view into it, built once here rather than per mesh.
    #[allow(dead_code)]
    shadow_atlas_tex: wgpu::Texture,
    /// Retained for the per-frame parent recompose, which has to rebuild each
    /// mesh's MVP and each layer's projected rect.
    camera3d: Option<crate::engine::sim::camera3d::PerspectiveCamera>,
    /// Parent chain per object, for recomposing transforms scripts change.
    transform_graph: crate::engine::model::resolve::TransformGraph,
    /// `general.zoom`, retained because the per-frame parent recompose rebuilds
    /// orthographic rects and has to apply the same scene-wide scale the build
    /// path does.
    zoom: f32,
    /// Live per-object local transforms, carried across frames. WE's
    /// `update(value)` is handed the CURRENT value, so a script written as
    /// `value.y += ...` accumulates — resetting to the authored value each
    /// frame turns continuous rotation into jitter in place.
    current_locals: Vec<crate::engine::model::resolve::Xform>,
}

/// One static 3D mesh, uploaded once: geometry never changes, and its MVP is
/// baked at load because these objects don't animate in real content.
/// World-space corners of a quad centered at `center`, rotated by `angles`,
/// sized `size_px` — bottom-left, bottom-right, top-right, top-left in the
/// quad's own unrotated local space. Shared by `project_quad_ndc` and
/// `project_quad_corners` so their corner math can't drift apart.
fn quad_world_corners(center: [f32; 3], angles: [f32; 3], size_px: [f64; 2]) -> [[f32; 3]; 4] {
    let hx = (size_px[0] / 2.0) as f32;
    let hy = (size_px[1] / 2.0) as f32;
    [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]].map(|[cx, cy]| {
        let off = crate::engine::sim::camera3d::rotate_euler([cx, cy, 0.0], angles);
        [center[0] + off[0], center[1] + off[1], center[2] + off[2]]
    })
}

/// Project a world-space quad through the perspective camera into an NDC rect.
///
/// Returns `(rect, depth)`; a negative depth means the quad was culled (some
/// corner sat at or behind the eye plane). Shared by the build path and the
/// per-frame parent recompose so the two can't drift apart.
///
/// Approximation: no in-quad perspective warp — the four projected corners
/// collapse to their axis-aligned bounding box. Used for depth-sort and as
/// the composite fallback in all cases; `project_quad_corners` (below) gives
/// the true quad shape for the common normal-blend case — see its own docs.
fn project_quad_ndc(
    cam: &crate::engine::sim::camera3d::PerspectiveCamera,
    center: [f32; 3],
    angles: [f32; 3],
    size_px: [f64; 2],
) -> ([f32; 4], f32) {
    let corners = quad_world_corners(center, angles, size_px);
    let ndc: Vec<[f32; 2]> = corners.iter().filter_map(|c| cam.project(*c)).collect();
    if ndc.len() < 4 {
        return ([-10.0, -10.0, 0.0, 0.0], -1.0);
    }
    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
    for p in &ndc {
        min_x = min_x.min(p[0]);
        min_y = min_y.min(p[1]);
        max_x = max_x.max(p[0]);
        max_y = max_y.max(p[1]);
    }
    (
        [
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            (max_x - min_x) / 2.0,
            (max_y - min_y) / 2.0,
        ],
        cam.view_depth(center),
    )
}

/// True perspective-projected quad corners (NDC, already divided), in the
/// same bottom-left/bottom-right/top-right/top-left order
/// `quad_world_corners` builds — `None` if any corner sits at or behind the
/// eye plane (same cull condition `project_quad_ndc` uses, kept consistent
/// rather than drawing a degenerate partial quad).
///
/// This fixes the silhouette bug `project_quad_ndc`'s AABB collapse has: an
/// off-axis or steeply-angled quad renders as a true trapezoid instead of a
/// rectangle. UV mapping across it is bilinear, not hardware
/// perspective-correct (that would need clip-space corners with real w, not
/// pre-divided NDC) — a scoped simplification: the silhouette is what the
/// Ghidra report flagged as visibly wrong, and bilinear already fixes that;
/// full perspective-correct texture mapping is a further, separate
/// improvement.
fn project_quad_corners(
    cam: &crate::engine::sim::camera3d::PerspectiveCamera,
    center: [f32; 3],
    angles: [f32; 3],
    size_px: [f64; 2],
) -> Option<[[f32; 2]; 4]> {
    let corners = quad_world_corners(center, angles, size_px);
    let mut ndc = [[0.0f32; 2]; 4];
    for (i, c) in corners.iter().enumerate() {
        ndc[i] = cam.project(*c)?;
    }
    Some(ndc)
}

struct Mesh3dGpu {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    index_count: u32,
    bind_group: wgpu::BindGroup,
    nocull: bool,
    depthtest: bool,
    /// Kept so a script-driven parent can rewrite the MVP each frame.
    ubo: wgpu::Buffer,
    /// Pre-parent transform + which scene object this is, so the per-frame
    /// pass can recompose the chain (see `render::TransformGraph`).
    local: ([f32; 3], [f32; 3], [f32; 3]),
    order_index: usize,
    /// View depth of the mesh centre, for the perspective back-to-front sort.
    /// Meshes are never culled on it (a skybox legitimately surrounds the
    /// camera); it only orders them against the 2D layers.
    depth: f32,
}

/// Point-light cap for the mesh3d lighting pass. Must match
/// `MESH3D_MAX_LIGHTS` in `gpu_shaders.wgsl` exactly — there's no shared
/// constant across the Rust/WGSL boundary, so keep the two in sync by hand.
const MESH3D_MAX_LIGHTS: usize = 8;

/// Shadow-casting-light cap, mirrored from `engine::gpu::shadow::MAX_SHADOW_LIGHTS`
/// — must also match `MESH3D_MAX_SHADOW_LIGHTS` in `gpu_shaders.wgsl` by
/// hand, same caveat as `MESH3D_MAX_LIGHTS`.
const MESH3D_MAX_SHADOW_LIGHTS: usize = crate::engine::gpu::shadow::MAX_SHADOW_LIGHTS;

/// Byte length of the `Mesh3dLighting` uniform buffer: 5 leading `vec4`s
/// (flags, ambient, fog_distance, fog_height, fog_extra), three
/// `MESH3D_MAX_LIGHTS`-length `vec4` arrays (positions, colors, spot
/// direction+exponent), then `MESH3D_MAX_SHADOW_LIGHTS` `mat4x4`s (shadow
/// view-projections) and `MESH3D_MAX_SHADOW_LIGHTS` `vec4`s (their atlas UV
/// sub-rects), then one trailing `mat4x4` (`cookie_view_proj`).
const MESH3D_LIGHTING_BYTES_LEN: usize =
    16 * 5 + 16 * MESH3D_MAX_LIGHTS * 3 + (64 + 16) * MESH3D_MAX_SHADOW_LIGHTS + 64;

impl GpuSceneInstance {
    /// Open a scene from a wallpaper directory with a freshly opened GPU device.
    pub fn open(dir: &Path) -> Result<Self> {
        let gpu = GpuDevice::open_low_power()
            .or_else(|_| GpuDevice::open_best())
            .context("opening GPU device")?;
        Self::with_gpu(gpu, dir)
    }

    /// Open a scene using an existing device/queue (e.g. the one that owns a
    /// Wayland surface, so rendered frames can be presented directly).
    pub fn with_device(device: wgpu::Device, queue: wgpu::Queue, dir: &Path) -> Result<Self> {
        let renderer = GpuSceneRenderer::with_device(device, queue)?;
        Self::build(renderer, dir)
    }

    pub fn with_gpu(gpu: GpuDevice, dir: &Path) -> Result<Self> {
        let renderer = GpuSceneRenderer::new(gpu)?;
        Self::build(renderer, dir)
    }

    fn build(mut renderer: GpuSceneRenderer, dir: &Path) -> Result<Self> {
        let resolved = crate::engine::model::resolve::ResolvedScene::from_directory(dir)?;
        // The unparsed scene.json: scripts and timelines read fields the
        // typed model drops.
        let raw_scene: serde_json::Value = crate::formats::assets::AssetStore::from_directory(dir)
            .ok()
            .and_then(|a| a.scene_json().ok())
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        let w = resolved.width;
        let h = resolved.height;

        let scene_model = crate::engine::model::scene_to_model(&resolved.scene)?;

        let general = resolved.scene.general.as_ref();
        let clear_color: [f64; 3] = general
            .and_then(|g| g.clear_color.as_ref())
            .and_then(|v| {
                let s = if let Some(s) = v.as_str() {
                    s.to_string()
                } else if let Some(inner) = v.get("value").and_then(|i| i.as_str()) {
                    inner.to_string()
                } else {
                    return None;
                };
                let parts: Vec<f64> = s
                    .split_whitespace()
                    .filter_map(|p| p.parse().ok())
                    .collect();
                if parts.len() >= 3 {
                    Some([parts[0], parts[1], parts[2]])
                } else {
                    None
                }
            })
            .unwrap_or([0.0, 0.0, 0.0]);
        // `clearenabled: false` → don't paint the clear color (the surface is
        // opaque, so this reads as black behind any gaps in the scene).
        let mut clear_color = if general
            .and_then(|g| g.clearenabled.as_ref())
            .and_then(parse_value_bool)
            == Some(false)
        {
            [0.0, 0.0, 0.0]
        } else {
            clear_color
        };
        // NOTE: `ambientcolor`/`skylightcolor` are inputs to LIT materials, not
        // a global tint. The whole corpus is unlit (0 shaders use LightingV1)
        // and these default to "0.3 0.3 0.3" on 195/197 scenes, so applying
        // them globally would darken every scene to 30%. They stay unused
        // (parsed only) — the visible lighting is the light-object glow.

        // HDR wallpapers author a separate `bloomhdr*` set; with a single SDR
        // bloom chain we pick that set when `hdr` is on, else the SDR one.
        let hdr = general
            .and_then(|g| g.hdr.as_ref())
            .and_then(parse_value_bool)
            .unwrap_or(false);
        let bloom = BloomSettings {
            enabled: hdr
                || general
                    .and_then(|g| g.bloom.as_ref())
                    .and_then(parse_value_bool)
                    .unwrap_or(false),
            strength: general
                .and_then(|g| {
                    if hdr {
                        g.bloom_hdr_strength.as_ref()
                    } else {
                        g.bloom_strength.as_ref()
                    }
                })
                .and_then(parse_value_f32)
                .unwrap_or(1.0),
            threshold: general
                .and_then(|g| {
                    if hdr {
                        g.bloom_hdr_threshold.as_ref()
                    } else {
                        g.bloom_threshold.as_ref()
                    }
                })
                .and_then(parse_value_f32)
                .unwrap_or(0.5),
        };

        // `general.zoom` (default 1.0) — a scene-wide scale about the center.
        let zoom = general
            .and_then(|g| g.zoom.as_ref())
            .and_then(parse_value_f32)
            .filter(|z| *z > 0.0)
            .unwrap_or(1.0);
        // Scene-global particle force (gravity + wind).
        let scene_force = general.map(|g| g.particle_force()).unwrap_or([0.0, 0.0]);

        let dynamics = CameraDynamics::from_scene(&resolved.scene);

        // Perspective (3D) scenes project layer quads through the scene
        // camera instead of mapping pixel origins to NDC directly.
        let camera3d = crate::engine::sim::camera3d::PerspectiveCamera::from_scene(
            &resolved.scene,
            w as f32 / h as f32,
        );

        let layers: Vec<SceneLayerGpu> = resolved
            .layers
            .iter()
            .map(|l| {
                let authored = resolved
                    .transform_graph
                    .local
                    .get(l.order_index)
                    .map(|x| x.t)
                    .unwrap_or(l.local_origin);
                gpu_layer_from(&renderer, l, w, h, zoom, camera3d.as_ref(), authored)
            })
            .collect();

        let scene_effects = collect_effects(&scene_model);
        let effect_runtimes = load_effect_runtimes(&mut renderer, dir, &scene_effects);

        // Start desktop-audio capture only for scenes that actually react to
        // it (avoids opening a capture device for every wallpaper).
        // Start capture for audio shaders OR audio-reactive particle emitters.
        let particles_use_audio = resolved.particle_layers.iter().any(|pl| {
            pl.config
                .emitter
                .iter()
                .any(|e| e.audioprocessingmode.is_some() || e.audioprocessingbounds.is_some())
        });
        let uses_audio =
            particles_use_audio || effect_runtimes.iter().flatten().any(|r| r.uses_audio);
        let audio_capture = if uses_audio {
            crate::services::audio::AudioCapture::start()
        } else {
            None
        };

        // Sound objects: decode each file and play it looped; `startsilent`
        // ones are held ready but silent so a script can `play()` them.
        let sounds: Vec<SceneSound> = resolved
            .scene
            .objects
            .iter()
            .enumerate()
            .filter_map(|(object, obj)| {
                start_scene_sound(dir, obj).map(|playback| SceneSound { object, playback })
            })
            .collect();

        // Computed early (not just before `build_shadow_atlas` needs it, its
        // original spot) so the FBO-pool setup below can also gate
        // `_rt_VolumetricsBuffer`'s allocation on whether any light actually
        // needs it — see `engine::model::scene::SceneObject::volumetrics_params`.
        let lights = resolved.scene.lights();
        let has_volumetrics = lights.iter().any(|(_, _, v, _)| v.is_some());

        // Allocate all persistent render targets up front so the render loop
        // can look them up immutably. Per-layer effect-chain FBOs (ping-pong
        // pair + named targets) are sized to that layer's own object size,
        // matching the reference (`CImage`'s m_mainFBO/m_subFBO and
        // `FBOProvider::create` both use the object's unscaled size, not the
        // scene resolution).
        let mut fbo_pool = RenderTargetPool::new();
        for (layer_idx, layer) in layers.iter().enumerate() {
            let (ow, oh) = (layer.object_size.0.max(1), layer.object_size.1.max(1));
            fbo_pool.get_or_create(renderer.device(), &pingpong_key(layer_idx, 0), ow, oh);
            fbo_pool.get_or_create(renderer.device(), &pingpong_key(layer_idx, 1), ow, oh);
        }
        for (instance_idx, runtime) in effect_runtimes.iter().enumerate() {
            if let Some(runtime) = runtime {
                // `inst.layer_idx` is a raw scene-object index; find the
                // loaded layer that came from that object (positional
                // indexing would misalign whenever particles/skipped
                // objects precede the layer, sizing FBOs off — and routing
                // effects onto — a completely unrelated layer).
                let (ow, oh) = scene_effects
                    .get(instance_idx)
                    .and_then(|inst| layers.iter().find(|l| l.order_index == inst.layer_idx))
                    .map(|layer| (layer.object_size.0.max(1), layer.object_size.1.max(1)))
                    .unwrap_or((w, h));
                for (fbo_name, scale) in &runtime.fbos {
                    let div = scale.max(1.0);
                    let key = named_fbo_key(instance_idx, fbo_name);
                    fbo_pool.get_or_create(
                        renderer.device(),
                        &key,
                        ((ow as f32 / div) as u32).max(1),
                        ((oh as f32 / div) as u32).max(1),
                    );
                }
            }
        }
        // `_rt_imageLayerComposite_<objectId>_<a|b>`: WE's per-object composite
        // buffer, referenced by effects on *other* objects (2321732083's
        // samurai adds 5x object 101's buffer). Create one scene-sized target
        // per referenced name; each is filled right after its owning layer
        // composites. Aliasing these to the whole-scene snapshot instead —
        // which is what we used to do — meant `blend` added 5x the entire lit
        // city onto the samurai and blew it out to white.
        let layer_composite_names: Vec<String> = scene_effects
            .iter()
            .flat_map(|inst| inst.pass_overrides.iter())
            .flat_map(|over| {
                over.textures
                    .iter()
                    .filter_map(|t| t.clone())
                    .collect::<Vec<_>>()
            })
            .filter(|n| n.starts_with("_rt_imageLayerComposite"))
            .collect();
        for name in &layer_composite_names {
            fbo_pool.get_or_create(renderer.device(), name, w, h);
        }

        if bloom.enabled {
            // Matches the reference's three-buffer chain exactly (CScene.cpp):
            // no ping-pong needed since each stage writes a *different* buffer.
            let (qw, qh) = ((w / 4).max(1), (h / 4).max(1));
            let (ew, eh) = ((w / 8).max(1), (h / 8).max(1));
            fbo_pool.get_or_create(renderer.device(), "_rt_4FrameBuffer", qw, qh);
            fbo_pool.get_or_create(renderer.device(), "_rt_8FrameBuffer", ew, eh);
            fbo_pool.get_or_create(renderer.device(), "_rt_Bloom", ew, eh);
        }

        // Volumetric-light-shaft ray march (see `build_volumetrics` and
        // `record_volumetrics`) — quarter-res like bloom's own chain, cheap
        // enough for a per-pixel ray march, upsampled naturally by the blur
        // pass. Gated on `has_volumetrics`: real content essentially never
        // sets `castvolumetrics` (see the Ghidra report's `_rt_volumetrics*`
        // follow-up), so this stays a zero-cost no-op for the common case.
        if has_volumetrics {
            let (qw, qh) = ((w / 4).max(1), (h / 4).max(1));
            fbo_pool.get_or_create(renderer.device(), "_rt_VolumetricsBuffer", qw, qh);
            fbo_pool.get_or_create(renderer.device(), "_rt_VolumetricsBlurTmp", qw, qh);
        }

        let target = renderer.create_render_target(w, h);
        let scene_copy = renderer.create_render_target(w, h);

        let particle_systems: Vec<particle::ParticleSystem> = resolved
            .particle_layers
            .iter()
            .map(|pl| {
                let spawn_center = [pl.origin[0] as f32, h as f32 - pl.origin[1] as f32];
                let mut system = particle::ParticleSystem::from_config(
                    &pl.config,
                    spawn_center,
                    pl.overrides.as_ref(),
                );
                system.set_scene_force(scene_force);
                if let Some(sprite) = &pl.sprite_texture {
                    system.set_sprite_frames(sprite.frames.len(), sprite.duration);
                }
                for child in &pl.children {
                    system.add_child(
                        child.config.clone(),
                        child.sprite.clone(),
                        child.additive,
                        &child.child_ref,
                        spawn_center,
                    );
                }
                system
            })
            .collect();
        let particle_sprites: Vec<Option<particle::ParticleSprite>> = resolved
            .particle_layers
            .iter()
            .map(|pl| pl.sprite_texture.clone())
            .collect();
        let particle_order: Vec<usize> = resolved
            .particle_layers
            .iter()
            .map(|pl| pl.order_index)
            .collect();
        let particle_additive: Vec<bool> = resolved
            .particle_layers
            .iter()
            .map(|pl| pl.additive_blend)
            .collect();

        // GPU particle sprite arrays. Sprite-less systems draw the CPU
        // path's soft radial-falloff disc (alpha = (1-d)^2), baked once;
        // texture-less ropes fall back to a solid white strip, matching the
        // CPU flat fill.
        let soft_disc = {
            let mut img = RgbaImage::new(64, 64);
            for y in 0..64u32 {
                for x in 0..64u32 {
                    let dx = (x as f32 + 0.5) / 32.0 - 1.0;
                    let dy = (y as f32 + 0.5) / 32.0 - 1.0;
                    let d = (dx * dx + dy * dy).sqrt().min(1.0);
                    let t = 1.0 - d;
                    let a = (t * t * 255.0) as u8;
                    img.put_pixel(x, y, image::Rgba([255, 255, 255, a]));
                }
            }
            img
        };
        let white = RgbaImage::from_pixel(1, 1, image::Rgba([255, 255, 255, 255]));
        let array_view = |tex: &wgpu::Texture| {
            tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let make_gpu_tex = |sprite: Option<&particle::ParticleSprite>,
                            rope: bool|
         -> ParticleGpuTex {
            match sprite {
                Some(sp) if !sp.frames.is_empty() => {
                    let (tex, frames) = renderer.upload_texture_array(&sp.frames);
                    ParticleGpuTex {
                        view: array_view(&tex),
                        frames,
                        overbright: sp.overbright,
                        repeat: rope,
                    }
                }
                _ => {
                    let img = if rope { &white } else { &soft_disc };
                    let (tex, frames) = renderer.upload_texture_array(std::slice::from_ref(img));
                    ParticleGpuTex {
                        view: array_view(&tex),
                        frames,
                        overbright: 1.0,
                        repeat: rope,
                    }
                }
            }
        };
        let particle_gpu_assets: Vec<ParticleGpuAssets> = resolved
            .particle_layers
            .iter()
            .zip(particle_systems.iter())
            .map(|(pl, system)| ParticleGpuAssets {
                parent: make_gpu_tex(pl.sprite_texture.as_ref(), system.is_rope()),
                children: system
                    .child_sprite_info()
                    .into_iter()
                    .map(|(sprite, rope)| make_gpu_tex(sprite, rope))
                    .collect(),
            })
            .collect();

        // Static 3D meshes: upload geometry + bake each MVP. Only reachable
        // when the scene has a perspective camera to draw them through.
        // Draw only meshes whose name contains this — the way to tell "not
        // drawn" apart from "drawn but occluded" in a scene the camera sits
        // inside of (same role as WP_DEBUG_DUMP_FRAME).
        let mesh_filter = std::env::var("WP_MESH_FILTER").ok();
        // Shadow atlas: one shared depth texture, packed with a tile per
        // shadow-casting `light` object — see `build_shadow_atlas` and
        // `engine::gpu::shadow`. Built before the lighting uniform and the final
        // mesh3d draws because both need to bind its (by-then-final) view.
        // `lights` itself was computed earlier, alongside `has_volumetrics`.
        let (shadow_atlas_tex, shadow_slots, shadow_slot_data) = if camera3d.is_some() {
            Self::build_shadow_atlas(&renderer, &resolved.mesh3d_layers, &lights)
        } else {
            (
                renderer.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("shadow_atlas_dummy"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: MESH3D_DEPTH_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                }),
                vec![None; lights.len()],
                Vec::new(),
            )
        };
        let shadow_atlas_view = shadow_atlas_tex.create_view(&Default::default());
        // Spot-light cookie texture (`usecookie`/`cookie` — see
        // `SceneObject::cookie_texture_path`). Scoped to the *first* Spot
        // light with a real, loadable cookie image — real content sets
        // `usecookie` on zero lights, ever (see the Ghidra report's
        // property-schema survey), so there's no real-content case to
        // generalize "one" into "N" against, same precedent
        // `build_volumetrics` already uses for its own "first light with
        // the flag" scoping. Only resolves against loose files under `dir`
        // (no `scene.pkg` fallback) — the pkg handle isn't otherwise
        // plumbed into `build`, and every real `usecookie` occurrence in
        // this sandbox's cache is zero either way.
        let cookie_data = {
            use crate::engine::gpu::lighting::Light;
            let (center, radius) =
                crate::engine::gpu::shadow::scene_bounds(&resolved.mesh3d_layers);
            lights
                .iter()
                .enumerate()
                .find_map(|(i, (light, _, _, cookie_path))| {
                    let path = cookie_path.as_ref()?;
                    let Light::Spot {
                        origin,
                        direction,
                        outer_cone_degrees,
                        ..
                    } = light
                    else {
                        return None;
                    };
                    let image = crate::engine::model::resolve::resolve_static_texture(
                        Some(dir),
                        None,
                        path,
                    )?;
                    let vp = crate::engine::gpu::shadow::spot_light_view_proj(
                        *origin,
                        *direction,
                        *outer_cone_degrees,
                        center,
                        radius,
                    );
                    Some((i, vp, image))
                })
        };
        let cookie_view = renderer
            .upload_texture(
                &cookie_data
                    .as_ref()
                    .map(|(_, _, img)| img.clone())
                    .unwrap_or_else(|| {
                        RgbaImage::from_pixel(1, 1, image::Rgba([255, 255, 255, 255]))
                    }),
            )
            .create_view(&Default::default());
        // Volumetric-light-shaft ray march's static per-scene bind group —
        // `None` unless `has_volumetrics` (checked again inside, cheaply,
        // since it also needs to find *which* light). See `build_volumetrics`.
        let volumetrics_bind_group = camera3d.as_ref().and_then(|cam| {
            Self::build_volumetrics(
                &renderer,
                cam,
                &lights,
                &shadow_slots,
                &shadow_slot_data,
                &shadow_atlas_view,
            )
        });
        // Scene-wide lighting/fog/shadow uniform, shared by every mesh3d
        // bind group (binding 3) — see `mesh3d_lighting_bytes`. Zeroed (and
        // thus read as "no lighting") when there's no perspective camera to
        // place lights through, matching `mesh3d` staying empty in that case.
        let mesh3d_lighting_bytes = camera3d
            .as_ref()
            .map(|cam| {
                mesh3d_lighting_bytes(
                    &resolved.scene,
                    cam,
                    &shadow_slots,
                    &shadow_slot_data,
                    cookie_data.as_ref().map(|(i, vp, _)| (*i, *vp)),
                )
            })
            .unwrap_or_else(|| vec![0u8; MESH3D_LIGHTING_BYTES_LEN]);
        let mesh3d_lighting_ubo = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh3d_lighting"),
            size: mesh3d_lighting_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        renderer
            .queue
            .write_buffer(&mesh3d_lighting_ubo, 0, &mesh3d_lighting_bytes);
        let mesh3d: Vec<Mesh3dGpu> = match &camera3d {
            Some(cam) => resolved
                .mesh3d_layers
                .iter()
                .filter(|m| {
                    mesh_filter
                        .as_ref()
                        .is_none_or(|f| m.name.contains(f.as_str()))
                })
                .map(|m| {
                    Self::build_mesh3d(
                        &renderer,
                        cam,
                        m,
                        &mesh3d_lighting_ubo,
                        &shadow_atlas_view,
                        &cookie_view,
                    )
                })
                .collect(),
            None => Vec::new(),
        };
        let mesh3d_depth = (!mesh3d.is_empty()).then(|| {
            renderer
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("mesh3d_depth"),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: MESH3D_DEPTH_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        });

        let mut instance = Self {
            renderer,
            layers,
            scene_effects,
            effect_runtimes,
            fbo_pool,
            target,
            scene_copy,
            width: w,
            height: h,
            clear_color,
            dynamics,
            bloom,
            volumetrics_bind_group,
            particle_systems,
            particle_sprites,
            particle_order,
            particle_additive,
            particle_gpu_assets,
            start: Instant::now(),
            last_time: 0.0,
            mouse_norm: [0.5, 0.5],
            scripts: build_script_host(
                dir,
                &resolved.scene,
                &resolved.transform_graph,
                [w, h],
                &raw_scene,
            ),
            timelines: build_timelines(&raw_scene, &resolved.transform_graph),
            live_props: LiveProps::new(dir, &raw_scene),
            object_visible: resolved
                .scene
                .objects
                .iter()
                .map(|o| o.is_visible())
                .collect(),
            propagation_disabled: vec![false; resolved.scene.objects.len()],
            scripts_want_audio: false,
            cursor_scene: [w as f32 * 0.5, h as f32 * 0.5],
            cursor_left_down: false,
            pending_buttons: Vec::new(),
            cursor_targets: Vec::new(),
            hovered: None,
            pressed: None,
            last_cursor_scene: [w as f32 * 0.5, h as f32 * 0.5],
            effect_enabled: Vec::new(),
            draw_rank: (0..resolved.scene.objects.len())
                .map(|i| i as f64)
                .collect(),
            sort_count: 0,
            scene_dir: dir.to_path_buf(),
            scene_pkg: None,
            audio_capture,
            sounds,
            #[cfg(target_os = "linux")]
            media: crate::services::media::MediaWatcher::start(),
            perspective: camera3d.is_some(),
            mesh3d,
            mesh3d_depth,
            mesh3d_lighting_ubo,
            shadow_atlas_tex,
            camera3d,
            current_locals: resolved.transform_graph.local.clone(),
            transform_graph: resolved.transform_graph,
            zoom,
        };
        // Hidden ancestors hide their subtree; script-referenced hidden
        // objects are built but start hidden.
        instance.refresh_layer_visibility();
        instance.effect_enabled = vec![true; instance.scene_effects.len()];
        instance.cursor_targets = instance
            .scripts
            .objects_with_handlers(crate::engine::sim::script::CURSOR_EVENTS);
        Ok(instance)
    }

    /// Interleave position+uv+normal into the vs_mesh3d vertex layout
    /// (stride 32: 3 + 2 + 3 floats) and upload it plus the index buffer.
    /// Shared by the final mesh3d draw (`build_mesh3d`) and the shadow-atlas
    /// depth pass (`build_shadow_atlas`), which both need the same geometry
    /// on the GPU but at different points in scene setup — the shadow atlas
    /// has to exist before `build_mesh3d` can bind it, so its geometry
    /// upload can't simply be reused from a `Mesh3dGpu` built later.
    fn upload_mesh3d_geometry(
        renderer: &GpuSceneRenderer,
        mesh: &crate::engine::model::mesh3d::Mesh3d,
    ) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        let mut verts: Vec<u8> = Vec::with_capacity(mesh.positions.len() * 32);
        for ((p, uv), n) in mesh.positions.iter().zip(&mesh.uvs).zip(&mesh.normals) {
            for f in [p[0], p[1], p[2], uv[0], uv[1], n[0], n[1], n[2]] {
                verts.extend_from_slice(&f.to_le_bytes());
            }
        }
        let vbuf = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh3d_verts"),
            size: verts.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        renderer.queue.write_buffer(&vbuf, 0, &verts);

        let idx: Vec<u8> = mesh.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        let ibuf = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh3d_indices"),
            size: idx.len() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        renderer.queue.write_buffer(&ibuf, 0, &idx);
        (vbuf, ibuf, mesh.indices.len() as u32)
    }

    /// Upload one mesh's geometry and bake its MVP. Geometry and transform are
    /// both static in real content, so this runs once at load.
    fn build_mesh3d(
        renderer: &GpuSceneRenderer,
        cam: &crate::engine::sim::camera3d::PerspectiveCamera,
        layer: &crate::engine::model::resolve::Mesh3dLayer,
        lighting_ubo: &wgpu::Buffer,
        shadow_atlas_view: &wgpu::TextureView,
        cookie_view: &wgpu::TextureView,
    ) -> Mesh3dGpu {
        let (vbuf, ibuf, index_count) = Self::upload_mesh3d_geometry(renderer, &layer.mesh);

        let params = mesh3d_transform_bytes(cam, layer.origin, layer.angles, layer.scale);
        let ubo = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh3d_xform"),
            size: params.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        renderer.queue.write_buffer(&ubo, 0, &params);

        let tex = renderer.upload_texture(&layer.texture);
        let view = tex.create_view(&Default::default());
        let bind_group = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mesh3d_bg"),
                layout: &renderer.mesh3d_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        // Meshes wrap their UVs (skyboxes/spheres tile).
                        resource: wgpu::BindingResource::Sampler(&renderer.samplers[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: lighting_ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(shadow_atlas_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(
                            &renderer.shadow_comparison_sampler,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(cookie_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(&renderer.samplers[0]),
                    },
                ],
            });

        Mesh3dGpu {
            vbuf,
            ibuf,
            index_count,
            bind_group,
            nocull: layer.nocull,
            depthtest: layer.depthtest,
            ubo,
            local: (layer.local_origin, layer.local_angles, layer.local_scale),
            order_index: layer.order_index,
            depth: cam.view_depth(layer.origin),
        }
    }

    /// Advance keyframe animations and apply the properties they move.
    fn tick_timelines(&mut self, delta: f32) {
        use crate::engine::sim::script::{ObjectChange, ScriptChanges};
        if self.timelines.is_empty() {
            return;
        }
        let mut changes: Vec<(usize, ObjectChange)> = Vec::new();
        let mut events: Vec<(usize, String)> = Vec::new();
        let mut synced: Vec<(usize, String, serde_json::Value)> = Vec::new();
        for t in &mut self.timelines {
            for name in t.anim.advance(delta) {
                events.push((t.object, name));
            }
            let mut v = t.anim.sample();
            if t.anim.relative {
                for (k, x) in v.iter_mut().enumerate() {
                    *x += t.base.get(k).copied().unwrap_or(0.0);
                }
            }
            if t.last.as_ref() == Some(&v) {
                continue;
            }
            t.last = Some(v.clone());
            let idx = match changes.iter().position(|(o, _)| *o == t.object) {
                Some(i) => i,
                None => {
                    changes.push((t.object, ObjectChange::default()));
                    changes.len() - 1
                }
            };
            let c = &mut changes[idx].1;
            let v3 = |fill: f32| -> [f32; 3] {
                std::array::from_fn(|k| {
                    v.get(k).copied().or(t.base.get(k).copied()).unwrap_or(fill)
                })
            };
            let json = if v.len() == 1 {
                serde_json::json!(v[0])
            } else {
                serde_json::json!(v)
            };
            match t.key.as_str() {
                "origin" => c.origin = Some(v3(0.0)),
                "scale" => c.scale = Some(v3(1.0)),
                "angles" => c.angles = Some(v3(0.0)),
                "color" => c.color = Some(v3(1.0)),
                "alpha" => c.alpha = v.first().copied(),
                "brightness" => c.brightness = v.first().copied(),
                key => {
                    c.extra.insert(key.to_string(), json.clone());
                }
            }
            synced.push((t.object, t.key.clone(), json));
        }
        if self.scripts.has_scripts() {
            for (object, key, value) in &synced {
                self.scripts.sync_property(*object, key, value);
            }
            for (object, name) in &events {
                self.scripts.dispatch_event(
                    *object,
                    "animationEvent",
                    &serde_json::json!({ "name": name }),
                );
            }
        }
        if !changes.is_empty() {
            self.apply_script_changes(ScriptChanges {
                objects: changes,
                commands: Vec::new(),
            });
        }
    }

    /// `thisLayer.getAnimation(name).play()` & co: control the animations on
    /// object `index` named `property` (by animation name or property key),
    /// along with the animations grouped with them.
    fn control_timeline(&mut self, index: usize, property: &str, op: &str, frame: Option<f32>) {
        let matched: Vec<String> = self
            .timelines
            .iter()
            .filter(|t| t.object == index)
            .filter(|t| t.key == property || t.anim.name.as_deref() == Some(property))
            .flat_map(|t| std::iter::once(t.key.clone()).chain(t.anim.children.iter().cloned()))
            .collect();
        for t in self.timelines.iter_mut().filter(|t| t.object == index) {
            let grouped = matched.contains(&t.key)
                || t.anim.parent.as_ref().is_some_and(|p| matched.contains(p));
            if grouped {
                t.anim.control(op, frame);
            }
        }
    }

    /// Run this frame's SceneScripts and apply what they changed.
    fn tick_scripts(&mut self, time: f32, delta: f32) {
        if !self.scripts.has_scripts() {
            return;
        }
        #[cfg(target_os = "linux")]
        if let Some(info) = self.media.as_ref().and_then(|w| w.try_recv()) {
            use crate::services::media::PlaybackState;
            self.scripts.set_media(
                info.available,
                match info.playback_state {
                    PlaybackState::Stopped => 0,
                    PlaybackState::Playing => 1,
                    PlaybackState::Paused => 2,
                },
                &info.title,
                &info.artist,
                &info.album,
                info.position_us,
                info.duration_us,
                info.art_url.as_deref(),
            );
        }
        if self.scripts_want_audio {
            if let Some(cap) = &self.audio_capture {
                let sp = cap.spectrum();
                self.scripts.set_audio([
                    &sp.s16_left,
                    &sp.s16_right,
                    &sp.s32_left,
                    &sp.s32_right,
                    &sp.s64_left,
                    &sp.s64_right,
                ]);
            }
        }
        self.scripts.set_cursor(
            self.cursor_scene,
            self.cursor_screen(),
            self.cursor_left_down,
        );
        self.dispatch_cursor_events();
        self.scripts
            .frame(time, crate::engine::sim::script::local_time_of_day(), delta);
        let changes = self.scripts.collect();
        self.apply_script_changes(changes);
        self.flush_text();

        // `registerAudioBuffers` usually runs at module load, so this is
        // known after the first frame; start capture then if nothing else did.
        if !self.scripts_want_audio && self.scripts.wants_audio() {
            self.scripts_want_audio = true;
            if self.audio_capture.is_none() {
                self.audio_capture = crate::services::audio::AudioCapture::start();
            }
        }
    }

    /// Fire `cursorEnter`/`Leave`/`Move`/`Down`/`Up`/`Click` on handler
    /// objects from this frame's pointer state.
    fn dispatch_cursor_events(&mut self) {
        let buttons = std::mem::take(&mut self.pending_buttons);
        if self.cursor_targets.is_empty() {
            return;
        }
        let pos = self.cursor_scene;
        let hit = self.hit_test(self.mouse_norm);
        if hit != self.hovered {
            if let Some(old) = self.hovered {
                self.scripts.dispatch_cursor(old, "cursorLeave", pos);
            }
            if let Some(new) = hit {
                self.scripts.dispatch_cursor(new, "cursorEnter", pos);
            }
            self.hovered = hit;
        }
        if pos != self.last_cursor_scene {
            self.last_cursor_scene = pos;
            if let Some(h) = hit {
                self.scripts.dispatch_cursor(h, "cursorMove", pos);
            }
            // A drag keeps reporting to the layer it started on.
            if let Some(p) = self.pressed.filter(|p| Some(*p) != hit) {
                self.scripts.dispatch_cursor(p, "cursorMove", pos);
            }
        }
        for down in buttons {
            if down {
                self.pressed = hit;
                if let Some(h) = hit {
                    self.scripts.dispatch_cursor(h, "cursorDown", pos);
                }
            } else if let Some(p) = self.pressed.take() {
                self.scripts.dispatch_cursor(p, "cursorUp", pos);
                if hit == Some(p) {
                    self.scripts.dispatch_cursor(p, "cursorClick", pos);
                }
            }
        }
    }

    /// The topmost cursor-handler object whose quad contains the cursor
    /// (`norm`: `[0,1]²`, top-left origin). Hidden layers still count:
    /// real content uses invisible hitboxes that toggle themselves.
    fn hit_test(&self, norm: [f32; 2]) -> Option<usize> {
        let (w, h) = (self.width as f32, self.height as f32);
        // Output pixels, Y-up, origin at the centre — the space `rect` is in.
        let p = [(norm[0] - 0.5) * w, (0.5 - norm[1]) * h];
        self.layers
            .iter()
            .filter(|l| self.cursor_targets.contains(&l.order_index))
            .filter(|l| {
                let corners: [[f32; 2]; 4] = match l.quad_corners {
                    Some(c) => c.map(|c| [c[0] * w * 0.5, c[1] * h * 0.5]),
                    None => quad_corners_px(l.rect, l.angle, w, h),
                };
                point_in_convex_quad(p, &corners)
            })
            .map(|l| l.order_index)
            .max()
    }

    /// Record a left-button press or release (from the platform layer).
    pub fn set_mouse_button(&mut self, left_down: bool) {
        if left_down != self.cursor_left_down {
            self.cursor_left_down = left_down;
            self.pending_buttons.push(left_down);
        }
    }

    /// The cursor in output pixels, top-left origin (`input.cursorScreenPosition`).
    fn cursor_screen(&self) -> [f32; 2] {
        [
            self.mouse_norm[0] * self.width as f32,
            self.mouse_norm[1] * self.height as f32,
        ]
    }

    /// Apply a frame's script changes: transforms recompose through the
    /// parent graph, everything else lands on the object's layers and sounds.
    fn apply_script_changes(&mut self, changes: crate::engine::sim::script::ScriptChanges) {
        use crate::engine::sim::script::ScriptCommand;
        let n = self.current_locals.len();
        let mut moved = vec![false; n];
        let mut visibility_changed = false;
        for (i, c) in &changes.objects {
            let i = *i;
            if i >= n {
                continue;
            }
            let x = &mut self.current_locals[i];
            if let Some(o) = c.origin {
                x.t = o.map(|v| v as f64);
                moved[i] = true;
            }
            if let Some(a) = c.angles {
                x.r = a.map(|v| v as f64);
                moved[i] = true;
            }
            if let Some(sc) = c.scale {
                x.s = sc.map(|v| v as f64);
                moved[i] = true;
            }
            if let Some(d) = c.propagation_disabled {
                self.propagation_disabled[i] = d;
                moved[i] = true;
            }
            if let Some(v) = c.visible {
                self.object_visible[i] = v;
                visibility_changed = true;
            }
            if let Some(v) = c.volume {
                for snd in self.sounds_of(i) {
                    snd.set_volume(v);
                }
            }
            for (key, value) in &c.extra {
                self.apply_script_extra(i, key, value);
            }
            for li in 0..self.layers.len() {
                if self.layers[li].order_index != i {
                    continue;
                }
                let layer = &mut self.layers[li];
                if let Some(v) = c.visible {
                    layer.visible = v;
                }
                if let Some(a) = c.alpha {
                    layer.alpha = a;
                }
                if let Some(col) = c.color {
                    layer.color = col;
                }
                if let Some(b) = c.brightness {
                    layer.brightness = b;
                }
                if let Some(f) = c.frame {
                    layer.script_frame = Some(f as usize);
                }
                if let Some(ps) = c.pointsize {
                    if let Some(td) = &mut layer.text_dynamic {
                        td.point_size = ps.max(1.0);
                        td.last_text.clear();
                    }
                }
                if c.text.is_some() || c.pointsize.is_some() {
                    self.rerasterize_text(li, c.text.as_deref());
                }
            }
        }
        for cmd in changes.commands {
            match cmd {
                ScriptCommand::Sound { index, op } => {
                    for snd in self.sounds_of(index) {
                        match op.as_str() {
                            "play" => snd.play(),
                            "pause" => snd.pause(),
                            "stop" => snd.stop(),
                            _ => {}
                        }
                    }
                }
                ScriptCommand::Sort { index, to } => self.sort_object(index, to),
                ScriptCommand::Animation {
                    index,
                    property,
                    op,
                    frame,
                } => self.control_timeline(index, &property, &op, frame),
                ScriptCommand::Create { index, config } => {
                    self.create_object(index, &config);
                    visibility_changed = true;
                }
                ScriptCommand::Destroy { index } => {
                    // Hidden for good rather than removed: per-layer render
                    // targets are keyed by position in `self.layers`.
                    if let Some(v) = self.object_visible.get_mut(index) {
                        *v = false;
                    }
                    self.cursor_targets.retain(|t| *t != index);
                    visibility_changed = true;
                }
                other => {
                    tracing::debug!(target: "script", "unhandled script command {other:?}");
                }
            }
        }
        if visibility_changed {
            self.refresh_layer_visibility();
        }
        if moved.iter().any(|m| *m) {
            self.apply_object_transforms(&moved);
        }
    }

    /// The user properties `names` changed in the saved settings: show them
    /// live. Returns `false` when the scene must be rebuilt instead (see
    /// [`Self::apply_user_properties`]).
    pub fn set_user_properties(&mut self, names: &[String]) -> bool {
        let props =
            crate::workshop::properties::SceneProperties::from_project_dir(&self.live_props.dir);
        self.apply_user_properties(props, names)
    }

    /// Re-resolve the scene against `props` and push each object's
    /// differences through the same path SceneScript writes take; scripts
    /// get `applyUserProperties` for `names`. Returns `false` when a change
    /// can't be applied live (scene-wide settings, an effect that wasn't
    /// built at load, a field scripts can't set).
    fn apply_user_properties(
        &mut self,
        props: crate::workshop::properties::SceneProperties,
        names: &[String],
    ) -> bool {
        let lp = &mut self.live_props;
        lp.props = props;
        let mut next = lp.raw.clone();
        lp.props.resolve_scene_json(&mut next);
        let prev = std::mem::replace(&mut lp.resolved, next);
        if prev.get("general") != lp.resolved.get("general") {
            return false;
        }
        let objects = |v: &serde_json::Value| {
            v.get("objects")
                .and_then(|o| o.as_array())
                .cloned()
                .unwrap_or_default()
        };
        let (before, after) = (objects(&prev), objects(&lp.resolved));
        if before.len() != after.len() {
            return false;
        }
        let user: serde_json::Map<String, serde_json::Value> = names
            .iter()
            .filter_map(|n| {
                Some((
                    n.clone(),
                    serde_json::json!({ "value": lp.props.get(n)?.clone() }),
                ))
            })
            .collect();
        let mut objects = Vec::new();
        for (i, (a, b)) in before.iter().zip(&after).enumerate() {
            let mut diffs = Vec::new();
            changed_leaves(a, b, &mut Vec::new(), &mut diffs);
            if diffs.is_empty() {
                continue;
            }
            let mut change = crate::engine::sim::script::ObjectChange::default();
            for (path, v) in &diffs {
                if self.live_change(i, path, v, &mut change).is_none() {
                    tracing::debug!(target: "properties", "object {i}: {} needs a reload", path.join("."));
                    return false;
                }
            }
            objects.push((i, change));
        }
        self.apply_script_changes(crate::engine::sim::script::ScriptChanges {
            objects,
            commands: Vec::new(),
        });
        self.scripts
            .apply_user_properties(&serde_json::Value::Object(user));
        true
    }

    /// One changed scene.json field of object `object`, as a script change.
    /// `None` when it has no live counterpart.
    fn live_change(
        &self,
        object: usize,
        path: &[String],
        v: &serde_json::Value,
        c: &mut crate::engine::sim::script::ObjectChange,
    ) -> Option<()> {
        use crate::engine::model::scene::{parse_value_bool, parse_value_f32, parse_value_vec3};
        // `{"user": …, "value": X}` bindings resolve to their `value`.
        let v = v.get("value").unwrap_or(v);
        let vec3 = |v| parse_value_vec3(v).map(|a| a.map(|x| x as f32));
        let path: Vec<&str> = path.iter().map(String::as_str).collect();
        match path.as_slice() {
            ["visible"] => c.visible = Some(parse_value_bool(v)?),
            ["alpha"] => c.alpha = Some(parse_value_f32(v)?),
            ["brightness"] => c.brightness = Some(parse_value_f32(v)?),
            ["color"] => c.color = Some(vec3(v)?),
            ["origin"] => c.origin = Some(vec3(v)?),
            ["scale"] => c.scale = Some(vec3(v)?),
            ["volume"] => c.volume = Some(parse_value_f32(v)?),
            ["pointsize"] => c.pointsize = Some(parse_value_f32(v)?),
            ["text"] => c.text = Some(v.as_str()?.to_string()),
            ["instanceoverride", k] => {
                c.extra.insert(format!("instance.{k}"), v.clone());
            }
            ["effects", e, rest @ ..] => {
                let e: usize = e.parse().ok()?;
                // Effects hidden at load were never built.
                self.scene_effects
                    .iter()
                    .any(|d| d.layer_idx == object && d.effect_index == e)
                    .then_some(())?;
                match rest {
                    ["visible"] => {
                        c.extra
                            .insert(format!("effects.{e}.visible"), parse_value_bool(v)?.into());
                    }
                    ["passes", pass, "constantshadervalues", k] => {
                        c.extra.insert(format!("effects.{e}.{pass}.{k}"), v.clone());
                    }
                    _ => return None,
                }
            }
            _ => return None,
        }
        Some(())
    }

    /// A script wrote a declared extra property: `instance.<key>` (particle
    /// override), `effects.<e>.visible`, or `effects.<e>.<pass>.<constant>`.
    fn apply_script_extra(&mut self, object: usize, key: &str, value: &serde_json::Value) {
        if let Some(k) = key.strip_prefix("instance.") {
            for (sys, _) in self
                .particle_systems
                .iter_mut()
                .zip(&self.particle_order)
                .filter(|(_, o)| **o == object)
            {
                sys.set_instance_value(k, value);
            }
            return;
        }
        let Some(rest) = key.strip_prefix("effects.") else {
            return;
        };
        let mut parts = rest.splitn(3, '.');
        let Some(e) = parts.next().and_then(|e| e.parse::<usize>().ok()) else {
            return;
        };
        let Some(k) = self
            .scene_effects
            .iter()
            .position(|d| d.layer_idx == object && d.effect_index == e)
        else {
            return;
        };
        match (parts.next(), parts.next()) {
            (Some("visible"), None) => {
                self.effect_enabled[k] = value.as_bool().unwrap_or(true);
            }
            (Some(pass), Some(name)) => {
                let Some(pass) = pass.parse::<usize>().ok() else {
                    return;
                };
                let Some(runtime) = self.effect_runtimes.get_mut(k).and_then(|r| r.as_mut()) else {
                    return;
                };
                if let Some(p) = runtime.passes.get_mut(pass) {
                    // Vectors come back as arrays; the value parser wants "x y z".
                    let v = match value {
                        serde_json::Value::Array(a) => serde_json::Value::String(
                            a.iter()
                                .map(|x| x.as_f64().unwrap_or(0.0).to_string())
                                .collect::<Vec<_>>()
                                .join(" "),
                        ),
                        other => other.clone(),
                    };
                    insert_value_components(&mut p.values, name, &v);
                }
            }
            _ => {}
        }
    }

    /// `thisScene.sortLayer(layer, index)`: draw `index`'s object just below
    /// the object currently at `to`, later sorts landing lower.
    fn sort_object(&mut self, index: usize, to: usize) {
        let Some(base) = self.draw_rank.get(to).copied() else {
            return;
        };
        self.sort_count += 1;
        if let Some(r) = self.draw_rank.get_mut(index) {
            *r = base - 0.5 + 0.49 / self.sort_count as f64;
        }
    }

    /// `thisScene.createLayer(config)`: build object `index` from a
    /// scene.json-shaped config. Image, solid and text objects only; their
    /// effects are not instantiated.
    fn create_object(&mut self, index: usize, config: &serde_json::Value) {
        use crate::engine::model::resolve::Xform;
        // Keep every per-object table indexed in step with the script side,
        // even when this object can't be built.
        let n = index + 1;
        if self.current_locals.len() < n {
            self.current_locals.resize(n, Xform::IDENTITY);
            self.transform_graph.parent.resize(n, None);
            self.transform_graph.local.resize(n, Xform::IDENTITY);
            self.transform_graph.name.resize(n, None);
            self.transform_graph.id.resize(n, None);
            self.object_visible.resize(n, true);
            self.propagation_disabled.resize(n, false);
            let len = self.draw_rank.len();
            self.draw_rank.extend((len..n).map(|i| i as f64));
        }
        let obj: crate::engine::model::scene::SceneObject =
            match serde_json::from_value(config.clone()) {
                Ok(o) => o,
                Err(e) => {
                    tracing::warn!(target: "script", "createLayer: bad config: {e}");
                    return;
                }
            };
        if obj.sound.is_some() {
            if let Some(playback) = start_scene_sound(&self.scene_dir, &obj) {
                self.sounds.push(SceneSound {
                    object: index,
                    playback,
                });
            }
            return;
        }
        if self.scene_pkg.is_none() {
            self.scene_pkg =
                crate::formats::pkg::Package::from_file(&self.scene_dir.join("scene.pkg")).ok();
        }
        let Some(mut layer) = crate::engine::model::resolve::runtime_layer(
            &self.scene_dir,
            self.scene_pkg.as_ref(),
            &obj,
        ) else {
            tracing::debug!(target: "script", "createLayer: object {index} has nothing to draw");
            return;
        };
        layer.order_index = index;
        let local = crate::engine::model::resolve::obj_local_xform(&obj);
        self.current_locals[index] = local;
        self.transform_graph.local[index] = local;
        self.transform_graph.name[index] = obj.name.clone();
        self.object_visible[index] = obj.is_visible();
        let gpu = gpu_layer_from(
            &self.renderer,
            &layer,
            self.width,
            self.height,
            self.zoom,
            self.camera3d.as_ref(),
            local.t,
        );
        let (ow, oh) = (gpu.object_size.0.max(1), gpu.object_size.1.max(1));
        let li = self.layers.len();
        let device = self.renderer.device().clone();
        self.fbo_pool
            .get_or_create(&device, &pingpong_key(li, 0), ow, oh);
        self.fbo_pool
            .get_or_create(&device, &pingpong_key(li, 1), ow, oh);
        self.layers.push(gpu);
    }

    /// Sound controls belonging to scene object `index`.
    fn sounds_of(&self, index: usize) -> Vec<std::sync::Arc<crate::services::audio::SoundControl>> {
        self.sounds
            .iter()
            .filter(|s| s.object == index)
            .map(|s| s.playback.control())
            .collect()
    }

    /// Queue new text (or, with `None`, a re-raster at a new point size) for
    /// a text layer; [`Self::flush_text`] rasterizes it.
    fn rerasterize_text(&mut self, li: usize, text: Option<&str>) {
        let layer = &mut self.layers[li];
        let Some(td) = &layer.text_dynamic else {
            return;
        };
        let new_text = text
            .map(str::to_string)
            .or_else(|| layer.pending_text.clone())
            .unwrap_or_else(|| td.last_text.clone());
        if new_text.is_empty() || (text.is_some() && new_text == td.last_text) {
            layer.pending_text = None;
            return;
        }
        layer.pending_text = Some(new_text);
    }

    /// Rasterize pending text. Throttled per layer to every 100 ms, or every
    /// 10× the last raster's cost when that is longer, so a script that
    /// rewrites a huge text block every frame (debug overlays, timers with
    /// milliseconds) can't stall rendering; the newest text always lands.
    fn flush_text(&mut self) {
        let now = Instant::now();
        for li in 0..self.layers.len() {
            let layer = &mut self.layers[li];
            let (Some(text), Some(td)) = (&layer.pending_text, &mut layer.text_dynamic) else {
                continue;
            };
            let wait = (layer.text_raster_cost * 10).max(Duration::from_millis(100));
            if layer
                .text_rastered_at
                .is_some_and(|t| now.duration_since(t) < wait)
            {
                continue;
            }
            let started = Instant::now();
            if let Some(img) =
                crate::engine::gpu::text::rasterize(&td.font_data, text, td.point_size)
            {
                // The quad keeps its authored box; only the texture changes.
                layer.frames[0] = self.renderer.upload_texture(&img);
                td.last_text = text.clone();
            }
            layer.text_raster_cost = started.elapsed();
            layer.text_rastered_at = Some(now);
            layer.pending_text = None;
        }
    }

    /// A layer draws only when its own object and every ancestor are visible.
    fn refresh_layer_visibility(&mut self) {
        let parent = &self.transform_graph.parent;
        let visible = &self.object_visible;
        let effective = |mut i: usize| -> bool {
            for _ in 0..64 {
                if !visible.get(i).copied().unwrap_or(true) {
                    return false;
                }
                match parent.get(i).copied().flatten() {
                    Some(p) if p != i => i = p,
                    _ => return true,
                }
            }
            true
        };
        for layer in &mut self.layers {
            layer.visible = effective(layer.order_index);
        }
    }

    /// Recompose world transforms from `current_locals` and push them into
    /// the layers and meshes they move: those whose own object or any
    /// ancestor `moved`.
    fn apply_object_transforms(&mut self, moved: &[bool]) {
        use crate::engine::model::resolve::Xform;
        let n = self.current_locals.len();
        let parent = self.transform_graph.parent.clone();
        // affected[i]: object i or an ancestor moved.
        let mut affected = vec![false; n];
        for i in 0..n {
            let mut cur = i;
            for _ in 0..64 {
                if moved.get(cur).copied().unwrap_or(false) {
                    affected[i] = true;
                    break;
                }
                match parent.get(cur).copied().flatten() {
                    Some(p) if p != cur => cur = p,
                    _ => break,
                }
            }
        }
        let locals = self.current_locals.clone();
        let worlds = self
            .transform_graph
            .world(&locals, &self.propagation_disabled);
        // The PARENT's world transform, not the object's own — `worlds[i]`
        // already folds in object i's local transform.
        let parent_world = |i: usize| -> Xform {
            if self.propagation_disabled.get(i).copied().unwrap_or(false) {
                return Xform::IDENTITY;
            }
            match parent.get(i).copied().flatten() {
                Some(p) if p != i => worlds[p],
                _ => Xform::IDENTITY,
            }
        };

        // 3D meshes: rebuild the MVP from the recomposed world transform.
        // Computed up front so the write-back can borrow `self.mesh3d` mutably.
        let mesh_updates: Vec<(usize, Vec<u8>, f32)> = match &self.camera3d {
            Some(cam) => self
                .mesh3d
                .iter()
                .filter(|m| affected.get(m.order_index).copied().unwrap_or(false))
                .map(|m| {
                    let w = parent_world(m.order_index);
                    let (lo, la, ls) = match locals.get(m.order_index) {
                        Some(x) => (
                            [x.t[0] as f32, x.t[1] as f32, x.t[2] as f32],
                            [x.r[0] as f32, x.r[1] as f32, x.r[2] as f32],
                            [x.s[0] as f32, x.s[1] as f32, x.s[2] as f32],
                        ),
                        None => m.local,
                    };
                    let o = w.apply_point([lo[0] as f64, lo[1] as f64, lo[2] as f64]);
                    let centre = [o[0] as f32, o[1] as f32, o[2] as f32];
                    let angles: [f32; 3] = std::array::from_fn(|i| la[i] + w.r[i] as f32);
                    let scale: [f32; 3] = std::array::from_fn(|i| ls[i] * w.s[i] as f32);
                    let bytes = mesh3d_transform_bytes(cam, centre, angles, scale);
                    (m.order_index, bytes, cam.view_depth(centre))
                })
                .collect(),
            None => Vec::new(),
        };
        for (idx, bytes, depth) in mesh_updates {
            let Some(m) = self.mesh3d.iter_mut().find(|m| m.order_index == idx) else {
                continue;
            };
            self.renderer.queue.write_buffer(&m.ubo, 0, &bytes);
            m.depth = depth;
        }

        // 2D layers, both projections. The rect derivations differ (the
        // perspective one projects the quad's corners through the camera) but
        // both start from the same recomposed world transform.
        let (w_px, h_px) = (self.width as f64, self.height as f64);
        for layer in &mut self.layers {
            let i = layer.order_index;
            if !affected.get(i).copied().unwrap_or(false) {
                continue;
            }
            let w = parent_world(i);
            let local = locals[i];
            let o = w.apply_point(std::array::from_fn(|k| local.t[k] + layer.align_offset[k]));
            let size_px = [
                layer.effective_size[0] * local.s[0] * w.s[0],
                layer.effective_size[1] * local.s[1] * w.s[1],
            ];
            let local_angles: [f32; 3] = std::array::from_fn(|k| local.r[k] as f32);
            match &self.camera3d {
                Some(cam) => {
                    let angles: [f32; 3] = std::array::from_fn(|k| local_angles[k] + w.r[k] as f32);
                    let centre = [o[0] as f32, o[1] as f32, o[2] as f32];
                    let (rect, depth) = project_quad_ndc(cam, centre, angles, size_px);
                    layer.rect = rect;
                    layer.quad_corners = project_quad_corners(cam, centre, angles, size_px);
                    // Feeds the painter's-algorithm sort in `render()`; a
                    // negative depth culls the quad, same as at build.
                    layer.depth = depth;
                }
                None => {
                    // Same `general.zoom` scene-wide scale the build path uses.
                    let z = self.zoom;
                    layer.rect = [
                        (2.0 * o[0] / w_px - 1.0) as f32 * z,
                        (2.0 * o[1] / h_px - 1.0) as f32 * z,
                        (size_px[0] / w_px) as f32 * z,
                        (size_px[1] / h_px) as f32 * z,
                    ];
                    layer.angle = -(local_angles[2] + w.r[2] as f32);
                }
            }
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn device(&self) -> &wgpu::Device {
        self.renderer.device()
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.renderer.queue()
    }

    /// Feed pointer position in [0,1]² (drives camera parallax).
    pub fn set_mouse(&mut self, norm: [f32; 2]) {
        self.mouse_norm = norm;
        // Scene space is Y-up with the origin bottom-left.
        self.cursor_scene = [
            norm[0] * self.width as f32,
            (1.0 - norm[1]) * self.height as f32,
        ];
    }
}

fn named_fbo_key(instance_idx: usize, fbo_name: &str) -> String {
    format!("fx{instance_idx}:{fbo_name}")
}

/// Load pipelines/textures for every effect instance in the scene: hardcoded
/// kernels get single-pass runtimes, everything else is loaded from the WE
/// assets directory pass-by-pass and translated GLSL→WGSL with the instance's
/// combos baked in.
fn load_effect_runtimes(
    renderer: &mut GpuSceneRenderer,
    wallpaper_dir: &Path,
    scene_effects: &[EffectInstanceDef],
) -> Vec<Option<EffectRuntime>> {
    let assets_dir = loader::find_we_assets_dir();
    // Wallpaper-local effects/materials/shaders (loose files or scene.pkg)
    // take priority over the global Steam assets dir, mirroring the
    // reference's Container mount order (dir, scene.pkg, gifscene.pkg, assets).
    let resolver = resolver::AssetResolver::new(Some(wallpaper_dir), assets_dir);

    // Debug escape hatch: WP_ENGINE_SKIP_EFFECTS=clouds,bloom disables the
    // named effects (useful to bisect a misbehaving shader translation).
    let skip: Vec<String> = std::env::var("WP_ENGINE_SKIP_EFFECTS")
        .map(|v| v.split(',').map(|s| s.trim().to_lowercase()).collect())
        .unwrap_or_default();

    scene_effects
        .iter()
        .enumerate()
        .map(|(instance_idx, inst)| {
            if skip.contains(&inst.name) {
                tracing::debug!(
                    target: "effect",
                    "SKIP '{}': disabled via WP_ENGINE_SKIP_EFFECTS",
                    inst.name
                );
                return None;
            }
            load_effect_instance(renderer, &resolver, instance_idx, inst)
        })
        .collect()
}

#[tracing::instrument(target = "effect", level = "debug", skip(renderer, resolver, inst), fields(effect = %inst.name, instance = instance_idx))]
fn load_effect_instance(
    renderer: &mut GpuSceneRenderer,
    resolver: &resolver::AssetResolver,
    instance_idx: usize,
    inst: &EffectInstanceDef,
) -> Option<EffectRuntime> {
    let effect_name = &inst.name;
    tracing::trace!(target: "effect", "loading effect instance");
    if HARDCODED_EFFECTS.contains(&effect_name.as_str()) {
        // Scene instances may override secondary texture slots (typically
        // slot 1 = an opacity mask, e.g. waterwaves/shake masks) — load
        // them under a per-instance key so the kernel's extra-slot binding
        // sees the real mask instead of the white 1×1 dummy.
        let tex_key = format!("fx{instance_idx}:{effect_name}#0");
        let mut textures: Vec<wgpu::Texture> = Vec::new();
        if let Some(over) = inst.pass_overrides.first() {
            for (slot_i, name) in over.textures.iter().skip(1).enumerate() {
                let img = name
                    .as_deref()
                    .filter(|n| !n.is_empty() && !n.starts_with("_rt_"))
                    .and_then(|n| {
                        let candidates = [format!("materials/{n}.tex"), format!("{n}.tex")];
                        candidates.iter().find_map(|rel| {
                            let bytes = resolver.read(rel)?;
                            crate::formats::tex::TexFile::parse(&bytes)
                                .ok()?
                                .to_rgba()
                                .ok()
                        })
                    });
                // Missing-slot fallbacks: normally white (= unmasked), but
                // shake's slot 1 is a FLOW map whose authored default is
                // util/noflow — rg 0.498 gray, "no pixel moves" (white would
                // shear the whole layer diagonally), and pulse's slot 1 is
                // its noise source, authored default util/noise.
                let img = img.or_else(|| {
                    if effect_name == "pulse" && slot_i == 0 {
                        let bytes = resolver.read("materials/util/noise.tex")?;
                        crate::formats::tex::TexFile::parse(&bytes)
                            .ok()?
                            .to_rgba()
                            .ok()
                    } else {
                        None
                    }
                });
                let fallback = if effect_name == "shake" && slot_i == 0 {
                    image::Rgba([127, 127, 127, 255])
                } else {
                    image::Rgba([255, 255, 255, 255])
                };
                textures.push(
                    renderer.upload_texture(
                        &img.unwrap_or_else(|| RgbaImage::from_pixel(1, 1, fallback)),
                    ),
                );
            }
        }
        // A shake with no texture overrides at all authors no flow map either
        // — bind the gray no-flow dummy so the effect stays static like WE.
        if effect_name == "shake" && textures.is_empty() {
            textures.push(renderer.upload_texture(&RgbaImage::from_pixel(
                1,
                1,
                image::Rgba([127, 127, 127, 255]),
            )));
        }
        if std::env::var("WP_DEBUG_TEX_SLOTS").is_ok() {
            eprintln!(
                "[texslots] {effect_name} (hardcoded, instance {instance_idx}): overrides={:?} loaded={}",
                inst.pass_overrides.first().map(|o| o.textures.clone()),
                textures.len()
            );
        }
        if !textures.is_empty() {
            renderer.dynamic_textures.insert(tex_key.clone(), textures);
        }
        return Some(EffectRuntime {
            passes: vec![EffectPassRuntime {
                key: effect_name.clone(),
                tex_key,
                hardcoded: true,
                target: None,
                // REVERTED (2026-08-26, see the Ghidra report's godrays/
                // waterwaves follow-up): this used to bind slot 0 to
                // `_rt_FullFrameBuffer` for waterripple/waterwaves, on the
                // theory that their `g_Texture0` ("hidden"/"framebuffer" in
                // the real shader's own annotation) meant the global scene
                // buffer. Real content proved that wrong: a real downloaded
                // wallpaper attaches 8 waterwaves instances directly to its
                // own background image (a shimmer filter on itself, not a
                // transparent water surface revealing what's behind it) —
                // at the point this object's own effect chain runs, the
                // global scene target is still blank (this object hasn't
                // composited onto it yet), so binding it there fed every
                // instance blank input, chained through 8 passes to a flat
                // grey wash-out. "framebuffer" evidently means this pass's
                // own chain input here, which is exactly what leaving
                // `binds` empty (falling through to `chain_view`) already
                // gives it — the pre-fix behavior, restored.
                binds: Vec::new(),
                values: {
                    // Hardcoded kernels read combos (e.g. shake's DIRECTION/
                    // NOISE) as pseudo-values with a `combo_` prefix.
                    let mut vals = inst
                        .pass_overrides
                        .first()
                        .map(|o| o.values.clone())
                        .unwrap_or_default();
                    if let Some(over) = inst.pass_overrides.first() {
                        for (k, v) in &over.combos {
                            vals.insert(format!("combo_{k}"), *v as f32);
                        }
                    }
                    vals
                },
                vertex_buffers: Vec::new(),
            }],
            fbos: Vec::new(),
            uses_audio: false,
        });
    }
    let Ok(eff_def) = effect_def::load_effect_by_file(resolver, &inst.file) else {
        tracing::warn!(
            target: "effect",
            "SKIP '{effect_name}': no effect.json at '{}'",
            inst.file
        );
        return None;
    };
    // Bundle directory for this effect (e.g. "effects/clouds" or a
    // workshop-nested "effects/workshop/2138904733/foo"), used to check for
    // bundled materials/shaders before falling back to root-relative paths.
    let effect_dir = inst
        .file
        .rsplit_once('/')
        .map(|(dir, _)| dir)
        .unwrap_or(inst.file.as_str());

    let fbos: Vec<(String, f32)> = eff_def
        .fbos
        .iter()
        .filter_map(|f| {
            f.name
                .as_ref()
                .map(|n| (n.clone(), f.scale.unwrap_or(1.0) as f32))
        })
        .collect();

    let default_override = PassOverride::default();
    let mut passes: Vec<EffectPassRuntime> = Vec::new();
    let mut uses_audio = false;
    for (pass_idx, pass) in eff_def.passes.iter().enumerate() {
        let Some(mat_path) = pass.material.as_deref() else {
            continue;
        };
        let Ok(mat_def) = effect_def::load_material_from_effect(resolver, effect_dir, mat_path)
        else {
            tracing::warn!(
                target: "effect",
                "SKIP '{effect_name}' pass {pass_idx}: material '{mat_path}' not found"
            );
            continue;
        };
        let Some(mat_pass) = mat_def.passes.first() else {
            continue;
        };
        let Some(shader_name) = mat_pass.shader.as_deref() else {
            continue;
        };
        let Ok((frag_glsl, vert_glsl)) =
            resolver.load_glsl_shader_for_effect(shader_name, Some(effect_dir))
        else {
            tracing::warn!(
                target: "effect",
                "SKIP '{effect_name}' pass {pass_idx}: shader '{shader_name}' not found"
            );
            continue;
        };
        // Some WE downsample/blur shaders read a fixed-size array varying
        // through a `for` loop (`v_TexCoord[i]`) rather than literal indices;
        // unroll that loop into literal accesses first so the array-varying
        // pass below (which only handles literal indices) can then unroll
        // the declaration too.
        let frag_glsl = transpiler::unroll_simple_for_loops(&frag_glsl);
        let vert_glsl = transpiler::unroll_simple_for_loops(&vert_glsl);
        // WE's downsample/blur/bloom shaders declare literally-indexed array
        // varyings (`v_TexCoord[4]`) that naga's GLSL frontend can't accept
        // as entry-point I/O; unroll them to scalar varyings before anything
        // else touches the source.
        let frag_glsl = transpiler::unroll_array_varyings(&frag_glsl);
        let vert_glsl = transpiler::unroll_array_varyings(&vert_glsl);
        // Some shaders call max/min/clamp with a bare integer literal where a
        // float is expected (e.g. nitro.frag's `max(0, albedo.rgb)`) — legacy
        // NVIDIA compilers implicitly promoted it, naga/shaderc don't.
        let frag_glsl = transpiler::coerce_int_literal_builtin_args(&frag_glsl);
        let vert_glsl = transpiler::coerce_int_literal_builtin_args(&vert_glsl);
        // Some shaders assign a swizzle of a wider vector from `mix()` using
        // the un-swizzled vector as the first arg (e.g. shift_hue.frag's
        // `albedo.rgb = mix(albedo, newAlbedo, mask)` where `albedo` is
        // vec4) — WE's compiler truncates implicitly, naga/shaderc don't.
        let frag_glsl = transpiler::coerce_swizzle_mismatched_mix_arg(&frag_glsl);
        let vert_glsl = transpiler::coerce_swizzle_mismatched_mix_arg(&vert_glsl);

        // Scene.json pass overrides align with effect.json pass indices
        // (the reference's ImageEffectPassOverride).
        let override_ = inst
            .pass_overrides
            .get(pass_idx)
            .unwrap_or(&default_override);

        // Combos: material's, then the scene override's on top, uppercased
        // (the reference uppercases names when emitting #defines).
        let mut combos: HashMap<String, i32> = HashMap::new();
        for (k, v) in &mat_pass.combos {
            combos.insert(k.to_uppercase(), *v);
        }
        for (k, v) in &override_.combos {
            combos.insert(k.to_uppercase(), *v);
        }
        // Texture bindings: the material's own list, with the scene
        // instance's per-pass override applied on top position-by-position
        // (a scene `null` entry means "keep the material's default", not
        // "clear it" — matches the reference's ImageEffectPassOverride).
        let merged_textures: Vec<Option<String>> = {
            let len = mat_pass.textures.len().max(override_.textures.len());
            (0..len)
                .map(|i| {
                    override_
                        .textures
                        .get(i)
                        .and_then(|t| t.clone())
                        .or_else(|| mat_pass.textures.get(i).and_then(|t| t.clone()))
                })
                .collect()
        };
        // Some sampler2D uniforms annotate a `"combo":"NAME"` (e.g. nitro's
        // opacitymask slot) whose real semantics is "enabled whenever a real
        // texture is assigned to this slot" rather than a JSON-declared
        // combo default. Without this, an optional mask texture the scene
        // *does* provide never actually gates anything — the masked effect
        // washes over the whole layer instead of being confined to the mask
        // (e.g. nitro.frag's `#if MASK` block silently compiling out).
        let sampler_combos: Vec<Option<String>> =
            crate::engine::gpu::shaders::uniform_meta::parse_uniform_metadata(&frag_glsl)
                .into_iter()
                .filter(|m| m.uniform_type == "sampler2D")
                .map(|m| m.combo)
                .collect();
        for (i, tex) in merged_textures.iter().enumerate() {
            let Some(name) = tex else { continue };
            if name.is_empty() || name.starts_with("_rt_") || name.starts_with("_alias_") {
                continue;
            }
            if let Some(Some(combo_name)) = sampler_combos.get(i) {
                combos.insert(combo_name.to_uppercase(), 1);
            }
        }

        // Constants: material JSON defaults, overridden by scene values.
        let mut values: ShaderVals = ShaderVals::new();
        for (k, v) in &mat_pass.constant_shader_values {
            insert_value_components(&mut values, k, v);
        }
        for (k, v) in &override_.values {
            values.insert(k.clone(), *v);
        }

        let blending_str = mat_pass.blending.as_deref().unwrap_or("normal");
        let model = ShaderModel::from_resolved_glsl(
            shader_name.to_string(),
            frag_glsl,
            combos,
            WEBlending::from_str(blending_str),
        );
        let translated = match transpiler::translate_full(&model, Some(&vert_glsl)) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(target: "effect", "SKIP '{effect_name}' pass {pass_idx}: GLSL→WGSL failed: {e:#}");
                continue;
            }
        };
        uses_audio |= translated.uses_audio;
        let key = format!("fx{instance_idx}:{effect_name}#{pass_idx}");
        // Temporary debug hook: dump translated WGSL per pass.
        if std::env::var("WP_DEBUG_DUMP_WGSL").is_ok() {
            let safe = key.replace([':', '/', '#'], "_");
            let _ = std::fs::write(format!("/tmp/wgsl_{safe}.frag.wgsl"), &translated.wgsl);
            if let Some(v) = &translated.vert_wgsl {
                let _ = std::fs::write(format!("/tmp/wgsl_{safe}.vert.wgsl"), v);
            }
        }
        let blend = wgpu_blend_state(&model.blending);
        let mut pipeline_attrs = translated.attributes.clone();
        let mut added = renderer.add_dynamic_pipeline(
            key.clone(),
            &translated.wgsl,
            translated.vert_wgsl.as_deref(),
            translated.uniform_keys.clone(),
            &pipeline_attrs,
            blend,
        );
        if added.is_err() && !pipeline_attrs.is_empty() {
            // Real-VS pipeline failed (usually an interface mismatch the
            // preprocessor couldn't reconcile) — retry with the synthetic VS.
            tracing::warn!(
                target: "effect",
                "'{effect_name}' pass {pass_idx}: real VS failed ({}), synthetic fallback",
                added.as_ref().err().map(|e| e.to_string()).unwrap_or_default()
            );
            if let Ok(fallback) = transpiler::translate(&model) {
                pipeline_attrs = Vec::new();
                added = renderer.add_dynamic_pipeline(
                    key.clone(),
                    &fallback.wgsl,
                    fallback.vert_wgsl.as_deref(),
                    fallback.uniform_keys.clone(),
                    &[],
                    blend,
                );
            }
        }
        if let Err(e) = added {
            tracing::warn!(target: "effect", "SKIP '{effect_name}' pass {pass_idx}: pipeline failed: {e}");
            continue;
        }
        let vertex_buffers = renderer.attr_buffers_for(&pipeline_attrs);

        // Secondary textures (g_Texture1..): shader-annotation defaults,
        // overridden by the material pass's texture list (with the scene
        // instance's own override merged in above). Slot indices must stay
        // aligned, so failed loads push a white 1×1 stand-in instead of
        // shifting later slots down.
        let mut texture_names: Vec<Option<String>> = model
            .texture_slots
            .iter()
            .skip(1)
            .map(|slot| slot.default_path.clone())
            .collect();
        // Framebuffer refs in the texture list (e.g. refraction's compose
        // pass: `"textures": [null, "_rt_FullFrameBuffer"]`) aren't files —
        // they become extra binds, resolved at draw time like `bind` entries.
        let mut fbo_ref_binds: Vec<(u32, String)> = Vec::new();
        for (i, tex) in merged_textures.iter().enumerate().skip(1) {
            let slot = i - 1;
            if slot >= texture_names.len() {
                texture_names.resize(slot + 1, None);
            }
            if let Some(name) = tex {
                if name.starts_with("_rt_") {
                    fbo_ref_binds.push((i as u32, name.clone()));
                } else if !name.starts_with("_alias_") {
                    texture_names[slot] = Some(name.clone());
                }
            }
        }
        for (slot, name) in texture_names.iter().enumerate() {
            if let Some(n) = name {
                if n.starts_with("_rt_") {
                    fbo_ref_binds.push((slot as u32 + 1, n.clone()));
                }
            }
        }
        texture_names
            .iter_mut()
            .filter(|n| n.as_deref().is_some_and(|n| n.starts_with("_rt_")))
            .for_each(|n| *n = None);
        if std::env::var("WP_DEBUG_TEX_SLOTS").is_ok() {
            eprintln!(
                "[texslots] {effect_name} pass{pass_idx}: merged={:?} names={:?} slots={:?}",
                merged_textures,
                texture_names,
                model
                    .texture_slots
                    .iter()
                    .map(|s| (
                        s.glsl_name.clone(),
                        s.default_path.clone(),
                        s.is_framebuffer
                    ))
                    .collect::<Vec<_>>()
            );
        }
        let mut textures: Vec<wgpu::Texture> = Vec::new();
        for name in &texture_names {
            let path = name.as_deref().unwrap_or("");
            // WE texture names resolve under materials/ first
            // (e.g. "util/clouds_256" → materials/util/clouds_256.tex).
            let candidates = [format!("materials/{path}.tex"), format!("{path}.tex")];
            let img = candidates.iter().find_map(|rel| {
                let bytes = resolver.read(rel)?;
                crate::formats::tex::TexFile::parse(&bytes)
                    .ok()?
                    .to_rgba()
                    .ok()
            });
            match img {
                Some(img) => textures.push(renderer.upload_texture(&img)),
                None => {
                    if !path.is_empty() {
                        tracing::warn!(
                            target: "effect",
                            "'{effect_name}': texture '{path}' not found — using white"
                        );
                    }
                    textures.push(renderer.upload_texture(&RgbaImage::from_pixel(
                        1,
                        1,
                        image::Rgba([255, 255, 255, 255]),
                    )));
                }
            }
        }
        renderer.dynamic_textures.insert(key.clone(), textures);

        passes.push(EffectPassRuntime {
            tex_key: key.clone(),
            key,
            hardcoded: false,
            target: pass.target.clone(),
            // Draw-time binding applies these in order, so the effect's own
            // `bind` entries must come LAST — CPass.cpp:809 "binds are set
            // last as they're the most important to be set". With the
            // material's texture-list _rt_ refs winning instead, godrays'
            // combine pass read the (still empty) scene snapshot as its
            // albedo and washed the whole layer to the clear colour.
            binds: fbo_ref_binds
                .into_iter()
                .chain(
                    pass.bind
                        .iter()
                        .filter_map(|b| b.name.as_ref().map(|n| (b.index.unwrap_or(0), n.clone()))),
                )
                .collect(),
            values,
            vertex_buffers,
        });
    }

    if passes.is_empty() {
        tracing::warn!(target: "effect", "SKIP '{effect_name}': no usable passes");
        return None;
    }
    tracing::debug!(target: "effect", "LOADED '{effect_name}' ({} passes)", passes.len());
    Some(EffectRuntime {
        passes,
        fbos,
        uses_audio,
    })
}

/// Continuously render scene frames into a channel (preview window, headless
/// tests, and the CPU/SHM fallback path).
#[tracing::instrument(target = "render", level = "debug", skip(tx), fields(dir = %dir.display(), target_fps))]
pub fn gpu_scene_render_loop(
    dir: &std::path::Path,
    tx: &SyncSender<Arc<RgbaImage>>,
    target_fps: f64,
) -> Result<()> {
    tracing::info!(target: "render", "opening GPU scene instance");
    let mut instance = GpuSceneInstance::open(dir)?;
    let frame_duration = Duration::from_secs_f64(1.0 / target_fps);
    let start = Instant::now();
    tracing::info!(target: "render", "entering GPU render loop");

    let mut frame_no: u64 = 0;
    loop {
        let frame = instance.render_rgba()?;
        tracing::trace!(target: "render", frame = frame_no, "rendered frame");
        frame_no += 1;
        if tx.send(Arc::new(frame)).is_err() {
            tracing::debug!(target: "render", frames = frame_no, "receiver dropped; ending render loop");
            return Ok(());
        }

        let elapsed = start.elapsed();
        let next = Duration::from_secs_f64(
            (elapsed.as_secs_f64() / frame_duration.as_secs_f64()).ceil()
                * frame_duration.as_secs_f64(),
        );
        if next > elapsed {
            std::thread::sleep(next - elapsed);
        }
    }
}

/// The unresolved scene.json, the wallpaper's user properties, and the scene
/// as last resolved against them.
struct LiveProps {
    dir: std::path::PathBuf,
    raw: serde_json::Value,
    props: crate::workshop::properties::SceneProperties,
    resolved: serde_json::Value,
}

impl LiveProps {
    fn new(dir: &Path, raw: &serde_json::Value) -> Self {
        let props = crate::workshop::properties::SceneProperties::from_project_dir(dir);
        let mut resolved = raw.clone();
        props.resolve_scene_json(&mut resolved);
        Self {
            dir: dir.to_path_buf(),
            raw: raw.clone(),
            props,
            resolved,
        }
    }
}

/// Collect the leaves that differ between `a` and `b` as (path, new value).
/// A `{"user": …}` binding counts as one leaf, as do arrays that changed
/// length.
fn changed_leaves(
    a: &serde_json::Value,
    b: &serde_json::Value,
    path: &mut Vec<String>,
    out: &mut Vec<(Vec<String>, serde_json::Value)>,
) {
    use serde_json::Value;
    if a == b {
        return;
    }
    match (a, b) {
        (Value::Object(x), Value::Object(y)) if !y.contains_key("user") => {
            for k in x.keys().chain(y.keys().filter(|k| !x.contains_key(*k))) {
                path.push(k.clone());
                changed_leaves(
                    x.get(k).unwrap_or(&Value::Null),
                    y.get(k).unwrap_or(&Value::Null),
                    path,
                    out,
                );
                path.pop();
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                path.push(i.to_string());
                changed_leaves(p, q, path, out);
                path.pop();
            }
        }
        _ => out.push((path.clone(), b.clone())),
    }
}

fn collect_effects(scene: &crate::engine::model::SceneModel) -> Vec<EffectInstanceDef> {
    let alias = |k: &str| -> String {
        k.strip_prefix("ui_editor_properties_")
            .unwrap_or(k)
            .replace('_', "")
            .to_lowercase()
    };
    let mut result = Vec::new();
    for (i, obj) in scene.objects.iter().enumerate() {
        if !obj.is_visible() {
            continue;
        }
        for (effect_index, eff) in obj.effects.iter().enumerate() {
            if !eff.visible {
                continue;
            }
            let file = match &eff.file {
                Some(s) => s,
                None => continue,
            };
            // WE effect paths look like "effects/{name}/effect.json"; the effect
            // name is the directory, not the filename (which is always "effect.json").
            let mut parts = file.rsplitn(3, '/');
            let _filename = parts.next().unwrap_or(""); // "effect.json"
            let name = parts
                .next()
                .unwrap_or(file) // "waterflow"
                .trim_end_matches(".json")
                .to_lowercase();

            // Scene.json pass entries align with effect.json pass indices.
            let pass_overrides: Vec<PassOverride> = eff
                .passes
                .iter()
                .map(|p| {
                    let mut values = ShaderVals::new();
                    for (k, v) in &p.shader_values {
                        insert_animated_components(&mut values, k, &alias(k), v);
                    }
                    PassOverride {
                        combos: p.combos.clone(),
                        values,
                        textures: p.textures.clone(),
                    }
                })
                .collect();

            result.push(EffectInstanceDef {
                layer_idx: i,
                effect_index,
                name,
                file: file.clone(),
                pass_overrides,
            });
        }
    }
    result
}

/// Upload one CPU `Layer` and derive its GPU-side quad. `authored_origin` is
/// the object's scene.json origin, before alignment.
fn gpu_layer_from(
    renderer: &GpuSceneRenderer,
    l: &crate::engine::model::resolve::Layer,
    w: u32,
    h: u32,
    zoom: f32,
    camera3d: Option<&crate::engine::sim::camera3d::PerspectiveCamera>,
    authored_origin: [f64; 3],
) -> SceneLayerGpu {
    let all_frames: Vec<&RgbaImage> = std::iter::once(&l.image)
        .chain(l.extra_frames.iter())
        .collect();
    // Textures stay at native resolution; the composite quad and
    // sampler handle scaling to the object's on-screen size.
    let mut frames: Vec<wgpu::Texture> = all_frames
        .iter()
        .map(|frame| renderer.upload_texture(frame))
        .collect();

    // Unscaled object size: explicit scene.json size, else the
    // texture's native resolution (CImage.cpp lines 213-222).
    // True scene-fullscreen layers (models/util/fullscreenlayer.json
    // etc.) never reach here — they're filtered out upstream by
    // `is_special_layer` — so an unspecified size always means
    // "use the texture's own dimensions", never "cover the scene".
    let effective_size = if l.size[0] > 0.0 && l.size[1] > 0.0 {
        [l.size[0], l.size[1]]
    } else {
        [l.image.width() as f64, l.image.height() as f64]
    };
    let object_size = (
        effective_size[0].round() as u32,
        effective_size[1].round() as u32,
    );

    // Object quad rect in NDC. WE origins are absolute scene
    // coordinates (Y-up, bottom-left origin) pointing at the
    // object's center.
    let size_px = [
        effective_size[0] * l.scale[0],
        effective_size[1] * l.scale[1],
    ];
    let (rect, angle, depth, quad_corners) = if let Some(cam) = camera3d {
        let center = [l.origin[0] as f32, l.origin[1] as f32, l.origin[2] as f32];
        let (rect, depth) = project_quad_ndc(cam, center, l.angles, size_px);
        let quad_corners = project_quad_corners(cam, center, l.angles, size_px);
        (rect, 0.0, depth, quad_corners)
    } else {
        // `general.zoom` scales the whole scene about its center;
        // NDC (0,0) is the scene center, so a uniform rect scale
        // does it (center and half-extent both × zoom).
        let rect = [
            (2.0 * l.origin[0] / w as f64 - 1.0) as f32 * zoom,
            (2.0 * l.origin[1] / h as f64 - 1.0) as f32 * zoom,
            (size_px[0] / w as f64) as f32 * zoom,
            (size_px[1] / h as f64) as f32 * zoom,
        ];
        (rect, l.angle, 0.0, None)
    };

    // A backdrop layer's texture is a copy of the scene under its quad,
    // sized to that on-screen region.
    if l.backdrop.is_some() {
        let px = (
            ((rect[2] * w as f32).round() as u32).clamp(1, w.max(1)),
            ((rect[3] * h as f32).round() as u32).clamp(1, h.max(1)),
        );
        frames = vec![renderer.create_render_target(px.0, px.1)];
    }
    SceneLayerGpu {
        frames,
        puppet: l.puppet.clone(),
        puppet_posed_at: 0.0,
        video: l.video.clone(),
        blend_mode: l.blend_mode,
        alpha: l.alpha,
        text_dynamic: l.text_dynamic.clone(),
        color: l.color,
        brightness: l.brightness,
        frame_duration_ms: l.frame_duration_ms,
        parallax_depth: [l.parallax_depth[0] as f32, l.parallax_depth[1] as f32],
        angle,
        rect,
        quad_corners,
        depth,
        object_size,
        no_interpolation: l.no_interpolation,
        clamp_uvs: l.clamp_uvs,
        order_index: l.order_index,
        object_id: l.object_id,
        visible: l.visible_base,
        effective_size,
        align_offset: std::array::from_fn(|i| l.local_origin[i] - authored_origin[i]),
        script_frame: None,
        backdrop: l.backdrop,
        pending_text: None,
        text_rastered_at: None,
        text_raster_cost: Duration::ZERO,
    }
}

/// A layer quad's corners in output pixels (Y-up, centred), from its NDC
/// `rect` (centre, half-extent) and z-rotation.
fn quad_corners_px(rect: [f32; 4], angle: f32, w: f32, h: f32) -> [[f32; 2]; 4] {
    let (cx, cy) = (rect[0] * w * 0.5, rect[1] * h * 0.5);
    let (hx, hy) = (rect[2] * w * 0.5, rect[3] * h * 0.5);
    let (s, c) = angle.sin_cos();
    [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]
        .map(|[x, y]| [cx + x * c - y * s, cy + x * s + y * c])
}

/// Whether `p` lies inside the convex quad `q` (either winding).
fn point_in_convex_quad(p: [f32; 2], q: &[[f32; 2]; 4]) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let a = q[i];
        let b = q[(i + 1) % 4];
        let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        if cross.abs() < f32::EPSILON {
            continue;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if cross.signum() != sign {
            return false;
        }
    }
    sign != 0.0
}

/// One property's keyframe animation on one object.
struct TimelineEntry {
    object: usize,
    /// `origin`/`alpha`/…, or a script-style extra key (`instance.rate`,
    /// `effects.<e>.<pass>.<constant>`).
    key: String,
    /// Base value, which transform keyframes are offsets from.
    base: Vec<f32>,
    anim: crate::engine::sim::timeline::PropertyAnimation,
    /// Last value applied, to skip re-applying an idle animation.
    last: Option<Vec<f32>>,
}

/// Collect every animated property from the raw scene.json objects.
fn build_timelines(
    raw_scene: &serde_json::Value,
    graph: &crate::engine::model::resolve::TransformGraph,
) -> Vec<TimelineEntry> {
    use crate::engine::sim::timeline::PropertyAnimation;
    let mut out = Vec::new();
    let objects = raw_scene.get("objects").and_then(|o| o.as_array());
    for (i, raw) in objects.into_iter().flatten().enumerate() {
        let mut props: Vec<(String, &serde_json::Value)> =
            ["origin", "scale", "angles", "alpha", "color", "brightness"]
                .iter()
                .filter_map(|k| raw.get(*k).map(|v| (k.to_string(), v)))
                .collect();
        if let Some(m) = raw.get("instanceoverride").and_then(|v| v.as_object()) {
            props.extend(m.iter().map(|(k, v)| (format!("instance.{k}"), v)));
        }
        for (e, eff) in raw
            .get("effects")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .enumerate()
        {
            for (p, pass) in eff
                .get("passes")
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
                .enumerate()
            {
                if let Some(m) = pass.get("constantshadervalues").and_then(|v| v.as_object()) {
                    props.extend(m.iter().map(|(k, v)| (format!("effects.{e}.{p}.{k}"), v)));
                }
            }
        }
        for (key, value) in props {
            let Some(anim) = value
                .get("animation")
                .and_then(|a| PropertyAnimation::parse(&key, a))
            else {
                continue;
            };
            let local = graph
                .local
                .get(i)
                .copied()
                .unwrap_or(crate::engine::model::resolve::Xform::IDENTITY);
            let base: Vec<f32> = match key.as_str() {
                "origin" => local.t.iter().map(|v| *v as f32).collect(),
                "scale" => local.s.iter().map(|v| *v as f32).collect(),
                "angles" => local.r.iter().map(|v| *v as f32).collect(),
                _ => Vec::new(),
            };
            out.push(TimelineEntry {
                object: i,
                key,
                base,
                anim,
                last: None,
            });
        }
    }
    out
}

/// Build the scene's SceneScript runtime: one layer proxy per scene object
/// (indexed like `scene.objects`, the same index `order_index` and the
/// transform graph use), then one instance per scripted property.
fn build_script_host(
    dir: &Path,
    scene: &crate::engine::model::scene::Scene,
    graph: &crate::engine::model::resolve::TransformGraph,
    canvas: [u32; 2],
    raw_scene: &serde_json::Value,
) -> crate::engine::sim::script::ScriptHost {
    use crate::engine::model::scene::{parse_value_f32, parse_value_vec3};
    use crate::engine::sim::script::{ObjectInit, ScriptHost};

    let mut host = ScriptHost::new();
    if scene.objects.iter().all(|o| o.scripts().is_empty()) {
        return host;
    }
    if let Some(path) = ScriptHost::storage_path_for(dir) {
        host = host.with_storage(path);
    }
    // The raw scene.json entries back `thisScene.getInitialLayerConfig`.
    let raw_objects: Vec<serde_json::Value> = raw_scene
        .get("objects")
        .and_then(|o| o.as_array())
        .cloned()
        .unwrap_or_default();
    if let Some(general) = raw_scene.get("general") {
        host.set_scene_general(general);
    }

    let user_props: serde_json::Map<String, serde_json::Value> =
        crate::workshop::properties::SceneProperties::from_project_dir(dir)
            .iter()
            .map(|p| (p.name.clone(), p.value.clone()))
            .collect();
    host.set_scene_info(
        [canvas[0] as f32, canvas[1] as f32],
        canvas,
        &serde_json::Value::Object(user_props),
    );

    let f32v = |v: [f64; 3]| [v[0] as f32, v[1] as f32, v[2] as f32];
    for (i, obj) in scene.objects.iter().enumerate() {
        let local = graph
            .local
            .get(i)
            .copied()
            .unwrap_or(crate::engine::model::resolve::Xform::IDENTITY);
        let text = obj.text.as_ref().and_then(|t| match t {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Object(m) => {
                m.get("value").and_then(|v| v.as_str()).map(str::to_string)
            }
            _ => None,
        });
        let init = ObjectInit {
            name: obj.name.clone(),
            id: obj.id,
            origin: f32v(local.t),
            angles: f32v(local.r),
            scale: f32v(local.s),
            size: f32v(obj.parsed_size()),
            color: obj
                .color
                .as_ref()
                .and_then(parse_value_vec3)
                .map(f32v)
                .unwrap_or([1.0; 3]),
            visible: obj.is_visible(),
            alpha: obj.alpha.as_ref().and_then(parse_value_f32).unwrap_or(1.0),
            brightness: obj
                .brightness
                .as_ref()
                .and_then(parse_value_f32)
                .unwrap_or(1.0),
            text,
            pointsize: obj
                .pointsize
                .as_ref()
                .and_then(parse_value_f32)
                .unwrap_or(32.0),
            volume: obj.volume.as_ref().and_then(parse_value_f32).unwrap_or(1.0),
            parallax_depth: obj
                .parallax_depth
                .as_ref()
                .and_then(parse_value_vec3)
                .map(|v| [v[0] as f32, v[1] as f32])
                .unwrap_or([0.0; 2]),
            parent_index: graph.parent.get(i).copied().flatten(),
            frame_count: 1,
            raw: raw_objects.get(i).cloned(),
        };
        host.register_object(i, &init);
    }

    for (i, obj) in scene.objects.iter().enumerate() {
        let props: [(&str, Option<&serde_json::Value>); 10] = [
            ("visible", obj.visible.as_ref()),
            ("origin", obj.origin.as_ref()),
            ("angles", obj.angles.as_ref()),
            ("scale", obj.scale.as_ref()),
            ("alpha", obj.alpha.as_ref()),
            ("color", obj.color.as_ref()),
            ("brightness", obj.brightness.as_ref()),
            ("text", obj.text.as_ref()),
            ("pointsize", obj.pointsize.as_ref()),
            ("volume", obj.volume.as_ref()),
        ];
        for (prop, value) in props {
            if let Some((script, authored)) = crate::engine::sim::script::scripted(value) {
                host.attach(i, Some(prop), script, authored);
            }
        }
        // Scripted particle overrides and effect constants/visibility, read
        // from the raw entry (the typed model drops their scripts).
        let Some(raw) = raw_objects.get(i) else {
            continue;
        };
        let mut extras: Vec<(String, &serde_json::Value)> = Vec::new();
        if let Some(m) = raw.get("instanceoverride").and_then(|v| v.as_object()) {
            extras.extend(m.iter().map(|(k, v)| (format!("instance.{k}"), v)));
        }
        for (e, eff) in raw
            .get("effects")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .enumerate()
        {
            if let Some(v) = eff.get("visible") {
                extras.push((format!("effects.{e}.visible"), v));
            }
            for (p, pass) in eff
                .get("passes")
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
                .enumerate()
            {
                if let Some(m) = pass.get("constantshadervalues").and_then(|v| v.as_object()) {
                    extras.extend(m.iter().map(|(k, v)| (format!("effects.{e}.{p}.{k}"), v)));
                }
            }
        }
        for (key, value) in extras {
            if let Some((script, authored)) = crate::engine::sim::script::scripted(Some(value)) {
                let initial = value
                    .get("value")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                host.declare_property(
                    i,
                    &key,
                    crate::engine::sim::script::value_kind(&initial),
                    &initial,
                );
                host.attach(i, Some(&key), script, authored);
            }
        }
    }
    host
}

#[cfg(test)]
mod tests;
