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

mod live;
use live::*;

mod build;
use build::*;

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

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn device(&self) -> &wgpu::Device {
        self.renderer.device()
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.renderer.queue()
    }
}

fn named_fbo_key(instance_idx: usize, fbo_name: &str) -> String {
    format!("fx{instance_idx}:{fbo_name}")
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

#[cfg(test)]
mod tests;
