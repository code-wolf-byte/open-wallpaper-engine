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

use crate::engine::camera_dynamics::{CameraDynamics, CameraFrameDynamics};
use crate::engine::fbo::RenderTargetPool;
use crate::engine::model::{ShaderModel, WEBlending};
use crate::engine::particle;
use crate::engine::scene::{parse_value_bool, parse_value_f32};
use crate::engine::shaders::{effect_def, loader, resolver, transpiler};
use crate::platform::GpuDevice;

const SHADER_SRC: &str = include_str!("shaders/gpu_shaders.wgsl");

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
    // Shadow-caster depth pass: see `engine::shadow` and
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

impl GpuSceneRenderer {
    pub fn new(gpu: GpuDevice) -> Result<Self> {
        let GpuDevice { device, queue, .. } = gpu;
        Self::with_device(device, queue)
    }

    pub fn with_device(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self> {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene_shaders"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let make_sampler = |no_interpolation: bool, clamp: bool| {
            let filter = if no_interpolation {
                wgpu::FilterMode::Nearest
            } else {
                wgpu::FilterMode::Linear
            };
            let address = if clamp {
                wgpu::AddressMode::ClampToEdge
            } else {
                wgpu::AddressMode::Repeat
            };
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("scene_sampler"),
                mag_filter: filter,
                min_filter: filter,
                address_mode_u: address,
                address_mode_v: address,
                ..Default::default()
            })
        };
        // Index by (no_interpolation as usize) << 1 | (clamp as usize).
        let samplers = [
            make_sampler(false, false),
            make_sampler(false, true),
            make_sampler(true, false),
            make_sampler(true, true),
        ];

        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
            },
            count: None,
        };
        let base_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("base_bgl"),
            entries: &[
                tex_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    // VERTEX too: vs_composite_quad reads the object rect.
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Extra texture slots for multi-texture effects (g_Texture1..7,
                // bound at N+2 — 7 is the highest the workshop corpus uses).
                tex_entry(3),
                tex_entry(4),
                tex_entry(5),
                tex_entry(6),
                tex_entry(7),
                tex_entry(8),
                tex_entry(9),
            ],
        });

        let effect_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("effect_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // VERTEX too: real WE vertex shaders read the shared UBO
                    // (g_ModelViewProjectionMatrix, animation params, ...).
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: the g_AudioSpectrum* storage buffer, updated per
                // frame from the FFT. Read-only storage (not uniform) because a
                // `float[N]` needs WGSL-illegal 16-byte uniform array stride;
                // storage packs tightly. Always present so every effect
                // pipeline's layout matches; audio shaders read it in the
                // fragment stage, others ignore the binding.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let audio_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("audio_spectrum"),
            size: crate::engine::audio::UNIFORM_BYTES as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("composite_layout"),
                bind_group_layouts: &[&base_bgl],
                push_constant_ranges: &[],
            });

        // Normal (colorBlendMode == 0) layers use hardware alpha blending.
        // Any other colorBlendMode is a Photoshop-style blend that needs to
        // read the destination in-shader (see fs_composite_blend), so that
        // pipeline writes its already-blended result straight through.
        let mut composite_pipelines = Vec::new();
        composite_pipelines.push((
            0u32,
            Self::create_pipeline(
                &device,
                &composite_pipeline_layout,
                &shader_module,
                "vs_composite_quad",
                "fs_composite",
                "composite_normal",
                Some(wgpu::BlendState::ALPHA_BLENDING),
                wgpu::TextureFormat::Rgba8Unorm,
            ),
        ));
        composite_pipelines.push((
            1u32,
            Self::create_pipeline(
                &device,
                &composite_pipeline_layout,
                &shader_module,
                "vs_composite_quad",
                "fs_composite_blend",
                "composite_blend",
                Some(wgpu::BlendState::REPLACE),
                wgpu::TextureFormat::Rgba8Unorm,
            ),
        ));
        // Base material pass: same quad/tint shader, no blending — the
        // destination (a layer's own FBO) is always freshly cleared to
        // transparent, so this is a plain "write the tinted pixel" pass, not
        // a composite (see BlendingMode_Normal == GL_ONE,GL_ZERO in CPass.cpp).
        let base_pass_pipeline = Self::create_pipeline(
            &device,
            &composite_pipeline_layout,
            &shader_module,
            "vs_composite_quad",
            "fs_composite",
            "base_material_pass",
            None,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let effect_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("effect_layout"),
            bind_group_layouts: &[&base_bgl, &effect_bgl],
            push_constant_ranges: &[],
        });

        let effect_entry_points: Vec<(&str, &str)> = HARDCODED_EFFECTS
            .iter()
            .map(|n| {
                (
                    *n,
                    match *n {
                        "pulse" => "fs_pulse",
                        "scroll" => "fs_scroll",
                        "shake" => "fs_shake",
                        "tint" => "fs_tint",
                        "opacity" => "fs_opacity",
                        "waterripple" => "fs_waterripple",
                        "waterwaves" => "fs_waterwaves",
                        "spin" => "fs_spin",
                        _ => "fs_composite",
                    },
                )
            })
            .collect();

        let mut effect_pipelines = Vec::new();
        for (name, entry) in &effect_entry_points {
            let pipeline = Self::create_pipeline(
                &device,
                &effect_layout,
                &shader_module,
                "vs_fullscreen",
                entry,
                name,
                None,
                wgpu::TextureFormat::Rgba8Unorm,
            );
            effect_pipelines.push((*name, pipeline));
        }

        // Bloom chain pipelines (all render into Rgba8Unorm pool targets).
        let bloom_threshold_pipeline = Self::create_pipeline(
            &device,
            &effect_layout,
            &shader_module,
            "vs_fullscreen",
            "fs_bloom_threshold",
            "bloom_threshold",
            None,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let bloom_blur_pipeline = Self::create_pipeline(
            &device,
            &effect_layout,
            &shader_module,
            "vs_fullscreen",
            "fs_blur9",
            "bloom_blur",
            None,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        // fs_bloom_combine now samples the pre-bloom scene copy itself and
        // adds bloom in-shader (matching combine.frag), so this pass is a
        // plain overwrite (WE's `combine.json` pass uses default/"normal"
        // blending, not a GPU-side additive blend).
        let bloom_combine_pipeline = Self::create_pipeline(
            &device,
            &effect_layout,
            &shader_module,
            "vs_fullscreen",
            "fs_bloom_combine",
            "bloom_combine",
            None,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let particle_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let particle_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle_layout"),
            bind_group_layouts: &[&particle_bgl],
            push_constant_ranges: &[],
        });
        // Additive: the reference draws particle quads with
        // glBlendFuncSeparate(SRC_ALPHA, ONE, SRC_ALPHA, ONE) (CPass.cpp).
        let particle_add_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let particle_over_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let particle_pipeline_add = Self::create_pipeline(
            &device,
            &particle_layout,
            &shader_module,
            "vs_particles",
            "fs_particles",
            "particles_add",
            Some(particle_add_blend),
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let particle_pipeline_over = Self::create_pipeline(
            &device,
            &particle_layout,
            &shader_module,
            "vs_particles",
            "fs_particles",
            "particles_over",
            Some(particle_over_blend),
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let mesh3d_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh3d_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Shared shadow atlas (one texture, every shadow-casting
                // light's tile packed into it — see `engine::shadow`) plus
                // its comparison sampler, for `textureSampleCompareLevel` PCF
                // lookups in `fs_mesh3d`.
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                // Spot-light cookie texture (`usecookie`/`cookie`) — a
                // harmless 1x1 white dummy when no light has one, same
                // "always-valid dummy binding" precedent as `shadow_atlas`.
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let mesh3d_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh3d_layout"),
            bind_group_layouts: &[&mesh3d_bgl],
            push_constant_ranges: &[],
        });
        let make_mesh3d_pipeline = |cull: Option<wgpu::Face>, depth: bool, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&mesh3d_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: Some("vs_mesh3d"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3],
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: Some("fs_mesh3d"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: MESH3D_DEPTH_FORMAT,
                    depth_write_enabled: depth,
                    depth_compare: if depth {
                        wgpu::CompareFunction::Less
                    } else {
                        wgpu::CompareFunction::Always
                    },
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };
        // Indexed by `nocull as usize | (!depthtest as usize) << 1`.
        let mesh3d_pipelines = [
            make_mesh3d_pipeline(Some(wgpu::Face::Back), true, "mesh3d"),
            make_mesh3d_pipeline(None, true, "mesh3d_nocull"),
            make_mesh3d_pipeline(Some(wgpu::Face::Back), false, "mesh3d_nodepth"),
            make_mesh3d_pipeline(None, false, "mesh3d_nocull_nodepth"),
        ];

        // Shadow-caster depth pass: renders every mesh3d object's geometry,
        // position-only, into one tile of the shared shadow atlas (see
        // `engine::shadow` and `GpuSceneInstance::build_shadow_atlas`). No
        // fragment stage — depth-only. No back-face culling: a caster should
        // block light through any face, not just the ones the main camera
        // would see.
        let shadow_pass_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_pass_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let shadow_pass_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_pass_layout"),
            bind_group_layouts: &[&shadow_pass_bgl],
            push_constant_ranges: &[],
        });
        let mesh3d_shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh3d_shadow"),
            layout: Some(&shadow_pass_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_shadow_depth"),
                // Same buffer the main mesh3d pipeline uses (stride 32,
                // pos+uv+normal) — only position is declared/read here, but
                // the stride must still match the real per-vertex layout.
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 32,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: MESH3D_DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let shadow_comparison_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_comparison_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Volumetric-light-shaft ray march (`fs_volumetrics`): its own small
        // bind group layout rather than extending `base_bgl`/`effect_bgl` —
        // it needs the shadow atlas + a comparison sampler, which those
        // don't declare, and nothing else. Downstream blur/combine passes
        // reuse the existing bloom pipelines unchanged (both are already
        // generic single-source-texture operations — see `record_volumetrics`).
        let volumetrics_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volumetrics_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let volumetrics_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volumetrics_layout"),
            bind_group_layouts: &[&volumetrics_bgl],
            push_constant_ranges: &[],
        });
        let volumetrics_pipeline = Self::create_pipeline(
            &device,
            &volumetrics_layout,
            &shader_module,
            "vs_fullscreen",
            "fs_volumetrics",
            "volumetrics",
            None,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        // Perspective-projected 2D quads (true silhouette, not the AABB
        // `project_quad_ndc` collapses to) — see `vs_quad3d`/`fs_quad3d` in
        // gpu_shaders.wgsl and the Ghidra report's quad-warp finding. Its
        // own small bind group layout rather than extending the widely
        // shared `base_bgl`: `Quad3DParams` (corners) has nothing to do with
        // any other pass that layout serves.
        let quad3d_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("quad3d_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let quad3d_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("quad3d_layout"),
            bind_group_layouts: &[&quad3d_bgl],
            push_constant_ranges: &[],
        });
        let quad3d_pipeline = Self::create_pipeline(
            &device,
            &quad3d_layout,
            &shader_module,
            "vs_quad3d",
            "fs_quad3d",
            "quad3d",
            Some(wgpu::BlendState::ALPHA_BLENDING),
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let dummy_tex = Self::create_white_1x1_texture(&device, &queue);

        Ok(Self {
            device,
            queue,
            samplers,
            base_bgl,
            effect_bgl,
            audio_buf,
            composite_pipelines,
            base_pass_pipeline,
            effect_pipelines,
            dynamic_pipelines: HashMap::new(),
            dynamic_uniform_keys: HashMap::new(),
            dynamic_vertex_attrs: HashMap::new(),
            dynamic_textures: HashMap::new(),
            attr_buffers: HashMap::new(),
            composite_pipeline_layout,
            effect_pipeline_layout: effect_layout,
            shader_module,
            dummy_tex,
            bloom_threshold_pipeline,
            bloom_blur_pipeline,
            bloom_combine_pipeline,
            blit_pipelines: HashMap::new(),
            mesh3d_bgl,
            mesh3d_pipelines,
            shadow_pass_bgl,
            mesh3d_shadow_pipeline,
            shadow_comparison_sampler,
            volumetrics_bgl,
            volumetrics_pipeline,
            quad3d_bgl,
            quad3d_pipeline,
            particle_bgl,
            particle_pipeline_add,
            particle_pipeline_over,
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    fn create_white_1x1_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy_white"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            tex.as_image_copy(),
            &[255u8, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        tex
    }

    #[allow(clippy::too_many_arguments)]
    fn create_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        vs_entry: &str,
        fs_entry: &str,
        label: &str,
        blend: Option<wgpu::BlendState>,
        format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some(vs_entry),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some(fs_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn create_pipeline_split_modules(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        vs_module: &wgpu::ShaderModule,
        vs_entry: &str,
        fs_module: &wgpu::ShaderModule,
        fs_entry: &str,
        label: &str,
        blend: Option<wgpu::BlendState>,
        vertex_layouts: &[wgpu::VertexBufferLayout<'_>],
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: vs_module,
                entry_point: Some(vs_entry),
                buffers: vertex_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: fs_module,
                entry_point: Some(fs_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_dynamic_pipeline(
        &mut self,
        key: String,
        wgsl: &str,
        vert_wgsl: Option<&str>,
        uniform_keys: Vec<transpiler::UniformEntry>,
        attributes: &[(String, String)],
        blend: Option<wgpu::BlendState>,
    ) -> Result<()> {
        use std::panic::AssertUnwindSafe;
        let wgsl_owned: std::borrow::Cow<'static, str> = wgsl.to_string().into();
        let label = key.clone();
        let device = &self.device;
        let fs_module = std::panic::catch_unwind(AssertUnwindSafe(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label.as_str()),
                source: wgpu::ShaderSource::Wgsl(wgsl_owned),
            })
        }))
        .map_err(|e| {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "(unknown)".into());
            anyhow!("shader module panicked for '{key}': {msg}")
        })?;

        // One vertex buffer per attribute; formats follow the GLSL types.
        let attr_descs: Vec<[wgpu::VertexAttribute; 1]> = attributes
            .iter()
            .enumerate()
            .map(|(i, (ty, _))| {
                [wgpu::VertexAttribute {
                    format: vertex_format_for(ty),
                    offset: 0,
                    shader_location: i as u32,
                }]
            })
            .collect();
        let vertex_layouts: Vec<wgpu::VertexBufferLayout> = attr_descs
            .iter()
            .map(|attrs| wgpu::VertexBufferLayout {
                array_stride: attrs[0].format.size(),
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: attrs,
            })
            .collect();

        let device = &self.device;
        let layout = &self.effect_pipeline_layout;
        let base = &self.shader_module;
        let label2 = key.clone();
        let vert_src_owned: Option<String> = vert_wgsl.map(|s| s.to_string());
        let pipeline = std::panic::catch_unwind(AssertUnwindSafe(|| {
            if let Some(ref vert_src) = vert_src_owned {
                let vert_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&format!("{label2}-vs")),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(vert_src.as_str())),
                });
                Self::create_pipeline_split_modules(
                    device,
                    layout,
                    &vert_module,
                    "main",
                    &fs_module,
                    "main",
                    &label2,
                    blend,
                    &vertex_layouts,
                )
            } else {
                Self::create_pipeline_split_modules(
                    device,
                    layout,
                    base,
                    "vs_we_effect",
                    &fs_module,
                    "main",
                    &label2,
                    blend,
                    &[],
                )
            }
        }))
        .map_err(|e| {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "(unknown)".into());
            anyhow!("pipeline panicked for '{key}': {msg}")
        })?;

        self.dynamic_pipelines.insert(key.clone(), pipeline);
        self.dynamic_uniform_keys.insert(key.clone(), uniform_keys);
        self.dynamic_vertex_attrs.insert(key, attributes.to_vec());
        Ok(())
    }

    /// Shared quad buffer for one vertex attribute. `kind`: 0 = NDC position,
    /// 1 = UV (v=0 at the vertex that lands on screen top after naga's Y
    /// flip — same convention as the synthetic VS), 2 = zeros.
    fn attr_buffer(&mut self, kind: u8, dims: u8) -> &wgpu::Buffer {
        let device = &self.device;
        self.attr_buffers.entry((kind, dims)).or_insert_with(|| {
            const CORNERS: [[f32; 2]; 6] = [
                [-1.0, -1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ];
            let mut data: Vec<f32> = Vec::with_capacity(6 * dims as usize);
            for [x, y] in CORNERS {
                let full: [f32; 4] = match kind {
                    0 => [x, y, 0.0, 1.0],
                    1 => {
                        let u = (x + 1.0) * 0.5;
                        let v = (y + 1.0) * 0.5;
                        [u, v, u, v]
                    }
                    _ => [0.0; 4],
                };
                data.extend_from_slice(&full[..dims as usize]);
            }
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("quad_attr"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::VERTEX,
                mapped_at_creation: true,
            });
            buf.slice(..).get_mapped_range_mut()[..].copy_from_slice(&bytes);
            buf.unmap();
            buf
        })
    }

    /// Resolve the quad buffers for a translated shader's attribute list.
    fn attr_buffers_for(&mut self, attributes: &[(String, String)]) -> Vec<wgpu::Buffer> {
        let specs: Vec<(u8, u8)> = attributes
            .iter()
            .map(|(ty, name)| {
                let dims = match ty.as_str() {
                    "float" => 1,
                    "vec2" => 2,
                    "vec3" => 3,
                    _ => 4,
                };
                let kind = if name.contains("Position") {
                    0
                } else if name.contains("TexCoord") {
                    1
                } else {
                    2
                };
                (kind, dims)
            })
            .collect();
        specs
            .into_iter()
            .map(|(kind, dims)| self.attr_buffer(kind, dims).clone())
            .collect()
    }

    /// Get (or lazily create) the blit pipeline for a given output format.
    /// Used for surface presentation and FBO up/down-sampling.
    pub fn blit_pipeline(&mut self, format: wgpu::TextureFormat) -> &wgpu::RenderPipeline {
        if !self.blit_pipelines.contains_key(&format) {
            let p = Self::create_pipeline(
                &self.device,
                &self.composite_pipeline_layout,
                &self.shader_module,
                "vs_fullscreen",
                "fs_blit",
                "blit",
                None,
                format,
            );
            self.blit_pipelines.insert(format, p);
        }
        &self.blit_pipelines[&format]
    }

    pub fn upload_texture(&self, img: &RgbaImage) -> wgpu::Texture {
        let (fw, fh) =
            crate::engine::fbo::fit_texture_limit(&self.device, img.width(), img.height());
        let downscaled;
        let img = if (fw, fh) != img.dimensions() {
            tracing::warn!(
                "layer texture {}x{} exceeds the GPU limit — downscaling to {fw}x{fh}",
                img.width(),
                img.height()
            );
            downscaled =
                image::imageops::resize(img, fw, fh, image::imageops::FilterType::Triangle);
            &downscaled
        } else {
            img
        };
        let (w, h) = img.dimensions();
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("layer_tex"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            img.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        tex
    }

    /// Uploads sprite-sheet frames as one 2D texture array (layer per frame)
    /// for the GPU particle pipeline. Frames are padded to the largest
    /// extent; TEXS sheets are uniform in practice, so padding is a
    /// theoretical safety net, not an expected path.
    pub fn upload_texture_array(&self, frames: &[RgbaImage]) -> (wgpu::Texture, u32) {
        let n = frames.len().max(1) as u32;
        let w = frames.iter().map(|f| f.width()).max().unwrap_or(1).max(1);
        let h = frames.iter().map(|f| f.height()).max().unwrap_or(1).max(1);
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("particle_sprite_array"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: n,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (i, frame) in frames.iter().enumerate() {
            let (fw, fh) = (frame.width(), frame.height());
            let padded = if fw == w && fh == h {
                frame.clone()
            } else {
                let mut img = RgbaImage::new(w, h);
                image::imageops::overlay(&mut img, frame, 0, 0);
                img
            };
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: i as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                padded.as_raw(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * w),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        }
        (tex, frames.len().max(1) as u32)
    }

    pub fn create_render_target(&self, w: u32, h: u32) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render_target"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn find_effect_pipeline(&self, key: &str) -> Option<&wgpu::RenderPipeline> {
        self.effect_pipelines
            .iter()
            .find(|(n, _)| *n == key)
            .map(|(_, p)| p)
            .or_else(|| self.dynamic_pipelines.get(key))
    }

    /// Pipeline key: 0 = normal alpha blend, 1 = Photoshop-style dest-read
    /// blend (any nonzero WE colorBlendMode; the mode itself travels in
    /// CompositeParams.mode).
    fn composite_pipeline(&self, blend_mode: u32) -> &wgpu::RenderPipeline {
        let key = if blend_mode == 0 { 0 } else { 1 };
        self.composite_pipelines
            .iter()
            .find(|(m, _)| *m == key)
            .or_else(|| self.composite_pipelines.first())
            .map(|(_, p)| p)
            .unwrap()
    }

    /// Select the sampler matching a `.tex`'s `NoInterpolation`/`ClampUVs`
    /// flags (nearest-vs-linear filtering, clamp-vs-repeat addressing).
    fn sampler_for(&self, no_interpolation: bool, clamp_uvs: bool) -> &wgpu::Sampler {
        let idx = ((no_interpolation as usize) << 1) | (clamp_uvs as usize);
        &self.samplers[idx]
    }

    /// Create the group(0) bind group: source texture, sampler, a uniform
    /// buffer, and up to six extra texture slots (dummy white when absent).
    fn make_base_bind_group(
        &self,
        src_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        uniform_buf: &wgpu::Buffer,
        extra: &[Option<wgpu::TextureView>],
    ) -> wgpu::BindGroup {
        let dummy_view = self.dummy_tex.create_view(&Default::default());
        let slot = |i: usize| -> &wgpu::TextureView {
            extra.get(i).and_then(|v| v.as_ref()).unwrap_or(&dummy_view)
        };
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.base_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(slot(0)),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(slot(1)),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(slot(2)),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(slot(3)),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(slot(4)),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(slot(5)),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(slot(6)),
                },
            ],
        })
    }

    fn make_uniform_buffer(&self, data: &[u8], min_size: u64) -> wgpu::Buffer {
        let size = (data.len() as u64).max(min_size);
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut padded = vec![0u8; size as usize];
        padded[..data.len()].copy_from_slice(data);
        self.queue.write_buffer(&buf, 0, &padded);
        buf
    }

    fn make_effect_bind_group(&self, param_buf: &wgpu::Buffer) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.effect_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.audio_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Record one pass drawing `vertex_count` vertices (3 = fullscreen
    /// triangle, 6 = object quad / real-VS quad with vertex buffers).
    #[allow(clippy::too_many_arguments)]
    fn run_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        base_bg: &wgpu::BindGroup,
        effect_bg: Option<&wgpu::BindGroup>,
        dst_view: &wgpu::TextureView,
        load: wgpu::LoadOp<wgpu::Color>,
        label: &str,
        vertex_count: u32,
        vertex_buffers: &[wgpu::Buffer],
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, base_bg, &[]);
        if let Some(bg) = effect_bg {
            pass.set_bind_group(1, bg, &[]);
        }
        for (i, buf) in vertex_buffers.iter().enumerate() {
            pass.set_vertex_buffer(i as u32, buf.slice(..));
        }
        pass.draw(0..vertex_count, 0..1);
    }

    pub fn readback(&self, target: &wgpu::Texture, w: u32, h: u32) -> Result<RgbaImage> {
        // wgpu requires bytes_per_row to be a multiple of COPY_BYTES_PER_ROW_ALIGNMENT (256).
        // Widths not divisible by 64 (e.g. 1080, 1366) would panic without this padding.
        let unpadded_bpr = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = (unpadded_bpr + align - 1) / align * align;
        let staging_size = (padded_bpr * h) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: staging_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv()
            .map_err(|_| anyhow!("readback channel disconnected"))?
            .map_err(|e| anyhow!("buffer map failed: {e:?}"))?;

        let data = slice.get_mapped_range();
        // Strip row padding to produce a tightly-packed RGBA image.
        let pixels: Vec<u8> = data
            .chunks(padded_bpr as usize)
            .flat_map(|row| &row[..unpadded_bpr as usize])
            .copied()
            .collect();
        drop(data);
        staging.unmap();

        RgbaImage::from_raw(w, h, pixels).context("creating RgbaImage from GPU readback")
    }
}

// ── Effect runtimes ───────────────────────────────────────────────────────────

fn vertex_format_for(glsl_type: &str) -> wgpu::VertexFormat {
    match glsl_type {
        "float" => wgpu::VertexFormat::Float32,
        "vec2" => wgpu::VertexFormat::Float32x2,
        "vec3" => wgpu::VertexFormat::Float32x3,
        _ => wgpu::VertexFormat::Float32x4,
    }
}

/// Map a material pass's `blending` mode to a GPU blend state. Effect passes
/// always render a fullscreen-covering quad into a just-cleared (transparent)
/// target, so this only changes output for passes whose shader legitimately
/// writes alpha<1 — Normal/Disabled write it through unmodified (replace),
/// while Additive/Translucent premultiply color by that alpha, matching the
/// reference's (ONE,ZERO) / (SRC_ALPHA,ONE) / (SRC_ALPHA,1-SRC_ALPHA) blend
/// functions for a zero destination.
/// Material pass blending → wgpu blend state, matching CPass::render's GL
/// states exactly: `normal` is GL_ONE/GL_ZERO and `disabled` is
/// glDisable(GL_BLEND) — both plain replace (`None` here); only
/// `additive`/`translucent` actually blend.
fn wgpu_blend_state(blending: &WEBlending) -> Option<wgpu::BlendState> {
    match blending {
        WEBlending::Normal | WEBlending::Disabled => None,
        WEBlending::Additive => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        WEBlending::Translucent => Some(wgpu::BlendState::ALPHA_BLENDING),
    }
}

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
    puppet: Option<std::sync::Arc<crate::engine::puppet::PuppetRuntime>>,
    /// Last animation time `frames[0]` was posed at, seconds.
    puppet_posed_at: f32,
    /// Embedded-video stream — `frames[0]` is replaced with the newest
    /// decoded frame each tick (the decoder thread paces itself by PTS;
    /// draining to the latest frame drops rather than lags).
    video: Option<std::sync::Arc<std::sync::Mutex<crate::engine::render::VideoLayerStream>>>,
    blend_mode: u32,
    /// Live alpha for the current frame. When `alpha_script` is present this is
    /// recomputed each frame from `alpha_base`; otherwise it stays constant.
    alpha: f32,
    /// Authored alpha, passed as the `value` argument to `alpha_script` each
    /// frame (WE feeds the property's base value, not the previous output).
    alpha_base: f32,
    /// Inline SceneScript source driving `alpha`, or `None` for static alpha.
    alpha_script: Option<String>,
    /// Script-driven text: re-evaluated each tick; when the output string
    /// changes, `frames[0]` is re-rasterized and `rect` re-derived (clocks
    /// tick, dates roll over — 42 of the corpus's 58 script wallpapers).
    text_dynamic: Option<crate::engine::render::TextDynamic>,
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
    /// Live visibility for this frame. Starts `true`; only a `visible` script
    /// can flip it off. `render()` skips the layer entirely when `false`.
    visible: bool,
    /// The authored `visible` value — fed to the `visible` script as `value`
    /// (a script with no `update()`, e.g. a cursor-handler-only hitbox, returns
    /// it unchanged, so this must not be hardcoded `true`).
    visible_base: bool,
    /// Per-frame transform scripts (visible/scale/origin/angles). When
    /// scale/origin are present, `rect` is recomputed each frame from the
    /// base values below; angles rebuilds `angle`. Only honored in the 2D
    /// (orthographic) path — perspective layers keep their projected rect.
    transform_scripts: crate::engine::render::TransformScripts,
    /// Base values fed to the transform scripts as `value` and used to rebuild
    /// `rect`: object size in px (pre-scale), authored origin (scene coords,
    /// alignment folded in), scale multiplier, and full angles (radians).
    effective_size: [f64; 2],
    origin_base: [f64; 3],
    scale_base: [f32; 3],
    angles_base: [f32; 3],
    /// Pre-parent transform, for the per-frame parent recompose (see
    /// `update_parent_transforms`). Distinct from `*_base`, which are the
    /// post-parent values a per-object script updates from.
    local_origin: [f64; 3],
    local_scale: [f64; 3],
    local_angles: [f32; 3],
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
    /// Persistent JS runtime for SceneScript-driven properties, ticked once per
    /// frame. Lives here (on the render thread) because boa's `Context` is
    /// `!Send`; created and used entirely within this instance.
    script_ctx: crate::engine::script::ScriptContext,
    /// Desktop-audio capture, started only when an effect reads
    /// `g_AudioSpectrum*` (else `None` and the spectrum stays silent). Polled
    /// once per frame into the renderer's `audio_buf`.
    audio_capture: Option<crate::engine::audio::AudioCapture>,
    /// Live playback streams for `sound` objects; kept alive for the scene's
    /// lifetime (dropping stops the audio).
    _sounds: Vec<crate::engine::audio::SoundPlayback>,
    /// MPRIS "what's playing" watcher (`engine::media`) — Linux-only (see
    /// its own module doc). Started unconditionally rather than gated on
    /// whether any script actually reads `media.*` (matching
    /// `engine.runtime`'s own always-on treatment): the D-Bus connection is
    /// a cheap background poll, and unlike the CEF bridge's `wants_media`
    /// scan there's no equivalent "does this bundle reference it" signal to
    /// scan for across independent per-property scripts.
    #[cfg(target_os = "linux")]
    media: Option<crate::engine::media::MediaWatcher>,
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
    camera3d: Option<crate::engine::camera3d::PerspectiveCamera>,
    /// Parent chain + transform scripts. Only consulted when
    /// `TransformGraph::needs_per_frame()` — 189 of 197 scenes skip it.
    transform_graph: crate::engine::render::TransformGraph,
    /// `general.zoom`, retained because the per-frame parent recompose rebuilds
    /// orthographic rects and has to apply the same scene-wide scale the build
    /// path does.
    zoom: f32,
    /// Live per-object local transforms, carried across frames. WE's
    /// `update(value)` is handed the CURRENT value, so a script written as
    /// `value.y += ...` accumulates — resetting to the authored value each
    /// frame turns continuous rotation into jitter in place.
    current_locals: Vec<crate::engine::render::Xform>,
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
        let off = crate::engine::camera3d::rotate_euler([cx, cy, 0.0], angles);
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
    cam: &crate::engine::camera3d::PerspectiveCamera,
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
    cam: &crate::engine::camera3d::PerspectiveCamera,
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

/// Shadow-casting-light cap, mirrored from `engine::shadow::MAX_SHADOW_LIGHTS`
/// — must also match `MESH3D_MAX_SHADOW_LIGHTS` in `gpu_shaders.wgsl` by
/// hand, same caveat as `MESH3D_MAX_LIGHTS`.
const MESH3D_MAX_SHADOW_LIGHTS: usize = crate::engine::shadow::MAX_SHADOW_LIGHTS;

/// Byte length of the `Mesh3dLighting` uniform buffer: 5 leading `vec4`s
/// (flags, ambient, fog_distance, fog_height, fog_extra), three
/// `MESH3D_MAX_LIGHTS`-length `vec4` arrays (positions, colors, spot
/// direction+exponent), then `MESH3D_MAX_SHADOW_LIGHTS` `mat4x4`s (shadow
/// view-projections) and `MESH3D_MAX_SHADOW_LIGHTS` `vec4`s (their atlas UV
/// sub-rects), then one trailing `mat4x4` (`cookie_view_proj`).
const MESH3D_LIGHTING_BYTES_LEN: usize =
    16 * 5 + 16 * MESH3D_MAX_LIGHTS * 3 + (64 + 16) * MESH3D_MAX_SHADOW_LIGHTS + 64;

/// Pack one mesh's `Mesh3dTransform` uniform (mvp, model_view, normal_view,
/// model) — layout must match `Mesh3dTransform` in `gpu_shaders.wgsl`
/// exactly. `model` (object→world, no view/projection) is what the shadow
/// lookup in `fs_mesh3d` needs `world_pos` for — the light's view-projection
/// is naturally built in world space, unlike everything else here which
/// works in the main camera's view space.
fn mesh3d_transform_bytes(
    cam: &crate::engine::camera3d::PerspectiveCamera,
    origin: [f32; 3],
    angles: [f32; 3],
    scale: [f32; 3],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(64 * 4);
    for mat in [
        cam.mvp(origin, angles, scale),
        cam.model_view(origin, angles, scale),
        cam.normal_view(angles, scale),
        crate::engine::camera3d::model_matrix(origin, angles, scale),
    ] {
        for col in mat.iter() {
            for f in col {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
        }
    }
    bytes
}

/// Pack the scene-wide `Mesh3dLighting` uniform: ambient + fog (already
/// resolved by `General::ambient_color`/`General::fog`), up to
/// `MESH3D_MAX_LIGHTS` point or spot lights from `Scene::lights` (positions
/// in the camera's view space, matching everything else `fs_mesh3d` works
/// in; a spot's facing direction goes into `light_spot`, also view-space),
/// and up to `MESH3D_MAX_SHADOW_LIGHTS` shadow view-projections + atlas UV
/// rects (world space — see `mesh3d_transform_bytes`). Layout must match
/// `Mesh3dLighting` in `gpu_shaders.wgsl` exactly.
///
/// `shadow_slots[i]` is the shadow-atlas slot index for `scene.lights()[i]`,
/// or `None` when that light doesn't cast a shadow (unlit, or beyond
/// `MESH3D_MAX_SHADOW_LIGHTS` — see `GpuSceneInstance::build_shadow_atlas`).
/// A light's `light_pos[i].w` carries `(slot + 1) as f32` so the shader can
/// tell "no shadow" (0.0) from "shadow slot 0" without a separate flags
/// field — `light_color.a <= 0.0` (unused light slot) already uses the same
/// "skip" convention.
///
/// `cookie` is `Some((light_index, view_proj))` when one of `scene.lights()`
/// is a Spot light with a resolvable cookie texture (see
/// `GpuSceneInstance::build`'s cookie-resolution step) — packed into
/// `flags.y` (as `light_index + 1`, `0.0` = none) and the trailing
/// `cookie_view_proj` field.
fn mesh3d_lighting_bytes(
    scene: &crate::engine::scene::Scene,
    cam: &crate::engine::camera3d::PerspectiveCamera,
    shadow_slots: &[Option<usize>],
    shadow_slot_data: &[(crate::engine::camera3d::Mat4, [f32; 4])],
    cookie: Option<(usize, crate::engine::camera3d::Mat4)>,
) -> Vec<u8> {
    let lights = scene.lights();
    let general = scene.general.as_ref();
    let fog = general.and_then(|g| g.fog());
    let ambient = general.and_then(|g| g.ambient_color()).unwrap_or([0.0; 3]);

    let mut bytes = Vec::with_capacity(MESH3D_LIGHTING_BYTES_LEN);
    let mut push4 = |v: [f32; 4]| {
        for f in v {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
    };

    // Gates on "at least one light this loop can actually render" — every
    // `Scene::lights()` variant is renderable now (Point, Spot, Directional,
    // Tube), so in practice this only matters for an empty `lights` list. A
    // scene with no renderable lights must still fall through to the plain
    // unlit look, not engage the lit branch with zero actual contributions
    // (which would multiply by `ambient` alone — usually black, per the
    // corpus's near-universal unset `ambientcolor`).
    let has_renderable_light = lights.iter().any(|(l, ..)| {
        matches!(
            l,
            crate::engine::lighting::Light::Point { .. }
                | crate::engine::lighting::Light::Spot { .. }
                | crate::engine::lighting::Light::Directional { .. }
                | crate::engine::lighting::Light::Tube { .. }
        )
    });
    let cookie_light_flag = cookie.map(|(idx, _)| (idx + 1) as f32).unwrap_or(0.0);
    push4([
        if has_renderable_light { 1.0 } else { 0.0 },
        cookie_light_flag,
        0.0,
        0.0,
    ]);
    push4([ambient[0], ambient[1], ambient[2], 0.0]);
    match &fog {
        Some(f) => {
            push4([f.distance_color[0], f.distance_color[1], f.distance_color[2], f.distance_density]);
            push4([f.height_color[0], f.height_color[1], f.height_color[2], f.height_density]);
            push4([f.height_exponent, f.height_offset, 0.0, 0.0]);
        }
        None => {
            push4([0.0; 4]);
            push4([0.0; 4]);
            push4([0.0; 4]);
        }
    }

    let mut positions = [[0.0f32; 4]; MESH3D_MAX_LIGHTS];
    let mut colors = [[0.0f32; 4]; MESH3D_MAX_LIGHTS];
    let mut spots = [[0.0f32; 4]; MESH3D_MAX_LIGHTS];
    for (i, ((pos_slot, (color_slot, spot_slot)), (light, ..))) in positions
        .iter_mut()
        .zip(colors.iter_mut().zip(spots.iter_mut()))
        .zip(lights.iter())
        .enumerate()
    {
        let shadow_w = || {
            shadow_slots
                .get(i)
                .copied()
                .flatten()
                .map(|s| (s + 1) as f32)
                .unwrap_or(0.0)
        };
        match light {
            crate::engine::lighting::Light::Point { origin, color, intensity, .. } => {
                let vp = cam.to_view_space(*origin);
                *pos_slot = [vp[0], vp[1], vp[2], shadow_w()];
                *color_slot = [color[0], color[1], color[2], *intensity];
                // spot_slot stays [0,0,0,0] — zero-length direction reads as
                // "not a spot" in `mesh3d_spot_factor`.
            }
            crate::engine::lighting::Light::Spot {
                origin,
                direction,
                exponent,
                color,
                intensity,
                ..
            } => {
                let vp = cam.to_view_space(*origin);
                // `build_shadow_atlas` now casts shadows from `Light::Spot`
                // too (see `engine::shadow::spot_light_view_proj`), so
                // `shadow_w()` is real here, not always 0.0.
                *pos_slot = [vp[0], vp[1], vp[2], shadow_w()];
                *color_slot = [color[0], color[1], color[2], *intensity];
                let vd = cam.to_view_direction(*direction);
                *spot_slot = [vd[0], vd[1], vd[2], *exponent];
            }
            crate::engine::lighting::Light::Directional {
                direction,
                color,
                intensity,
            } => {
                // Infinite light: `light_pos[i].xyz` carries a
                // pre-normalized view-space direction from the surface
                // toward the light (the negation of the light's own
                // travel direction, matching `to_light`'s convention for
                // Point/Spot), flagged via `w = -1.0` — a sentinel real
                // shadow_w values never take (0, or a positive slot+1).
                // See `fs_mesh3d`'s light loop in gpu_shaders.wgsl.
                let vd = cam.to_view_direction(*direction);
                let neg = [-vd[0], -vd[1], -vd[2]];
                let len = (neg[0] * neg[0] + neg[1] * neg[1] + neg[2] * neg[2])
                    .sqrt()
                    .max(1e-6);
                *pos_slot = [neg[0] / len, neg[1] / len, neg[2] / len, -1.0];
                *color_slot = [color[0], color[1], color[2], *intensity];
                // spot_slot stays [0,0,0,0] — mesh3d_spot_factor's own
                // "not a spot" check already no-ops for a directional light.
            }
            crate::engine::lighting::Light::Tube {
                origin_a,
                origin_b,
                color,
                intensity,
            } => {
                // Line-segment light: `light_pos[i].xyz` carries endpoint A
                // (view space), `light_spot[i].xyz` carries endpoint B
                // (view space, *not* a direction) — flagged via `w = -3.0`,
                // a sentinel distinct from directional's `-1.0` (both are
                // already `< 0.5`, so `mesh3d_shadow_factor` already treats
                // either as "unshadowed" with no extra branching needed
                // there; `mesh3d_spot_factor` does need an explicit skip,
                // since a non-zero-length `light_spot.xyz` would otherwise
                // be misread as a spot direction — see `fs_mesh3d`'s light
                // loop in gpu_shaders.wgsl).
                let va = cam.to_view_space(*origin_a);
                let vb = cam.to_view_space(*origin_b);
                *pos_slot = [va[0], va[1], va[2], -3.0];
                *color_slot = [color[0], color[1], color[2], *intensity];
                *spot_slot = [vb[0], vb[1], vb[2], 0.0];
            }
        }
    }
    for p in positions {
        push4(p);
    }
    for c in colors {
        push4(c);
    }
    for s in spots {
        push4(s);
    }

    // `Mesh3dLighting` in WGSL declares these as two *separate* fixed-size
    // arrays (`shadow_view_proj` then `shadow_uv_rect`), not an
    // array-of-structs — every matrix must be written before any rect, or
    // WGSL indexes into the wrong bytes entirely (confirmed the hard way: an
    // earlier interleaved version of this loop packed matrix+rect pairs
    // per-slot, which happened to leave `shadow_view_proj[0]` intact by
    // coincidence but made every `shadow_uv_rect[slot]` read garbage spliced
    // from a neighboring slot's matrix).
    for slot in 0..MESH3D_MAX_SHADOW_LIGHTS {
        let vp = shadow_slot_data
            .get(slot)
            .map(|(vp, _)| *vp)
            .unwrap_or_else(crate::engine::camera3d::identity);
        for col in vp.iter() {
            push4(*col);
        }
    }
    for slot in 0..MESH3D_MAX_SHADOW_LIGHTS {
        let uv_rect = shadow_slot_data.get(slot).map(|(_, r)| *r).unwrap_or([0.0; 4]);
        push4(uv_rect);
    }
    let cookie_view_proj = cookie
        .map(|(_, vp)| vp)
        .unwrap_or_else(crate::engine::camera3d::identity);
    for col in cookie_view_proj.iter() {
        push4(*col);
    }
    bytes
}

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
        let resolved = super::render::ResolvedScene::from_directory(dir)?;
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
        let camera3d = crate::engine::camera3d::PerspectiveCamera::from_scene(
            &resolved.scene,
            w as f32 / h as f32,
        );

        let layers: Vec<SceneLayerGpu> = resolved
            .layers
            .iter()
            .map(|l| {
                let all_frames: Vec<&RgbaImage> = std::iter::once(&l.image)
                    .chain(l.extra_frames.iter())
                    .collect();
                // Textures stay at native resolution; the composite quad and
                // sampler handle scaling to the object's on-screen size.
                let frames: Vec<wgpu::Texture> = all_frames
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
                let (rect, angle, depth, quad_corners) = if let Some(cam) = &camera3d {
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

                SceneLayerGpu {
                    frames,
                    puppet: l.puppet.clone(),
                    puppet_posed_at: 0.0,
                    video: l.video.clone(),
                    blend_mode: l.blend_mode,
                    alpha: l.alpha,
                    alpha_base: l.alpha,
                    alpha_script: l.alpha_script.clone(),
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
                    visible_base: l.visible_base,
                    transform_scripts: l.transform_scripts.clone(),
                    effective_size,
                    origin_base: l.origin,
                    scale_base: [l.scale[0] as f32, l.scale[1] as f32, l.scale[2] as f32],
                    angles_base: l.angles,
                    local_origin: l.local_origin,
                    local_scale: l.local_scale,
                    local_angles: l.local_angles,
                }
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
            crate::engine::audio::AudioCapture::start()
        } else {
            None
        };

        // Sound objects: decode + play each file (looped unless startsilent).
        let mut sounds = Vec::new();
        for obj in &resolved.scene.objects {
            let Some(sound) = obj.sound.as_ref() else {
                continue;
            };
            if obj
                .startsilent
                .as_ref()
                .and_then(|v| {
                    v.as_bool()
                        .or_else(|| v.get("value").and_then(|i| i.as_bool()))
                })
                .unwrap_or(false)
            {
                continue;
            }
            let volume = obj
                .volume
                .as_ref()
                .and_then(crate::engine::scene::parse_value_f32)
                .unwrap_or(1.0);
            let mode = obj.playbackmode.as_ref().and_then(|v| v.as_str());
            let looping = mode != Some("nointerrupt") && mode != Some("once");
            // `sound` is an array of file paths; play the first that decodes.
            let files: Vec<String> = match sound {
                serde_json::Value::Array(a) => a
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect(),
                serde_json::Value::String(s) => vec![s.clone()],
                _ => Vec::new(),
            };
            for f in files {
                if let Some(p) =
                    crate::engine::audio::SoundPlayback::start(&dir.join(&f), volume, looping)
                {
                    sounds.push(p);
                    break;
                }
            }
        }

        // Computed early (not just before `build_shadow_atlas` needs it, its
        // original spot) so the FBO-pool setup below can also gate
        // `_rt_VolumetricsBuffer`'s allocation on whether any light actually
        // needs it — see `engine::scene::SceneObject::volumetrics_params`.
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
        // `engine::shadow`. Built before the lighting uniform and the final
        // mesh3d draws because both need to bind its (by-then-final) view.
        // `lights` itself was computed earlier, alongside `has_volumetrics`.
        let (shadow_atlas_tex, shadow_slots, shadow_slot_data) = if camera3d.is_some() {
            Self::build_shadow_atlas(&renderer, &resolved.mesh3d_layers, &lights)
        } else {
            (
                renderer.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("shadow_atlas_dummy"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: MESH3D_DEPTH_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
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
            use crate::engine::lighting::Light;
            let (center, radius) = crate::engine::shadow::scene_bounds(&resolved.mesh3d_layers);
            lights.iter().enumerate().find_map(|(i, (light, _, _, cookie_path))| {
                let path = cookie_path.as_ref()?;
                let Light::Spot { origin, direction, outer_cone_degrees, .. } = light else {
                    return None;
                };
                let image = super::render::resolve_static_texture(Some(dir), None, path)?;
                let vp = crate::engine::shadow::spot_light_view_proj(
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
                    .unwrap_or_else(|| RgbaImage::from_pixel(1, 1, image::Rgba([255, 255, 255, 255]))),
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

        Ok(Self {
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
            script_ctx: crate::engine::script::ScriptContext::new(),
            audio_capture,
            _sounds: sounds,
            #[cfg(target_os = "linux")]
            media: crate::engine::media::MediaWatcher::start(),
            perspective: camera3d.is_some(),
            mesh3d,
            mesh3d_depth,
            mesh3d_lighting_ubo,
            shadow_atlas_tex,
            camera3d,
            current_locals: resolved.transform_graph.local.clone(),
            transform_graph: resolved.transform_graph,
            zoom,
        })
    }

    /// Build the volumetric-light-shaft ray march's static per-scene bind
    /// group (`VolumetricsParams` uniform + the shadow atlas + its
    /// comparison sampler — see `vs_fullscreen`/`fs_volumetrics` in
    /// gpu_shaders.wgsl), when the scene has a `castvolumetrics` light —
    /// `None` otherwise (the overwhelmingly common case: skipped entirely,
    /// no buffers allocated). See `SceneObject::volumetrics_params` and the
    /// Ghidra report's `_rt_volumetrics*` follow-up.
    ///
    /// Scoped to the *first* volumetric light found. Real content doesn't
    /// use this system at all (confirmed against the one downloaded
    /// Workshop sample this session found — see the report), so building
    /// multi-light support (looping N lights' worth of ray marches in one
    /// pass, like `fs_mesh3d`'s `MESH3D_MAX_LIGHTS` loop) would be
    /// speculative complexity with no real-content grounding to verify it
    /// against, for a feature that's already an approximation (a
    /// ray-marched-against-the-shadow-map technique, not WE's real
    /// procedural per-light cone/box mesh — see the report's scoping note).
    fn build_volumetrics(
        renderer: &GpuSceneRenderer,
        cam: &crate::engine::camera3d::PerspectiveCamera,
        lights: &[(
            crate::engine::lighting::Light,
            bool,
            Option<(f32, f32)>,
            Option<String>,
        )],
        shadow_slots: &[Option<usize>],
        shadow_slot_data: &[(crate::engine::camera3d::Mat4, [f32; 4])],
        shadow_atlas_view: &wgpu::TextureView,
    ) -> Option<wgpu::BindGroup> {
        use crate::engine::lighting::Light;

        let (idx, origin, color_intensity, density, exponent) =
            lights.iter().enumerate().find_map(|(i, (light, _, volumetrics, _))| {
                let (density, exponent) = (*volumetrics)?;
                let (origin, color, intensity) = match light {
                    // Directional has no position to march a shaft from;
                    // Tube isn't constructed by `Scene::lights()` yet (see
                    // `LightType::Tube`'s doc comment) — neither reaches here.
                    Light::Point { origin, color, intensity, .. }
                    | Light::Spot { origin, color, intensity, .. } => (*origin, *color, *intensity),
                    _ => return None,
                };
                let ci = [color[0] * intensity, color[1] * intensity, color[2] * intensity];
                Some((i, origin, ci, density, exponent))
            })?;

        // Reuses this light's shadow-atlas tile when it also casts a shadow
        // (`castshadow: true`) — the shaft respects real occlusion, not just
        // WE's own documented "unoccluded" fallback for when shadows aren't
        // built. `has_shadow = 0.0` (no matching slot) makes
        // `fs_volumetrics` skip the occlusion test entirely, matching that
        // same documented fallback for a volumetric light that isn't also a
        // shadow caster.
        let (shadow_view_proj, shadow_uv_rect, has_shadow) = shadow_slots
            .get(idx)
            .copied()
            .flatten()
            .and_then(|slot| shadow_slot_data.get(slot))
            .map(|(vp, rect)| (*vp, *rect, 1.0f32))
            .unwrap_or((crate::engine::camera3d::identity(), [0.0; 4], 0.0));

        let inv_view_proj = crate::engine::camera3d::mat4_inverse(&cam.view_proj_raw());
        let eye = cam.eye();
        // Fixed, honest default range for the ray march — no real content
        // to derive a better heuristic from (see the fn doc comment).
        // Comparable to `radius`'s own 256px-ish default scene scale.
        const MAX_RANGE: f32 = 40.0;

        let mut bytes = Vec::with_capacity(4 * 16 * 4 + 16 * 5);
        let mut push_mat = |m: &crate::engine::camera3d::Mat4| {
            for col in m.iter() {
                for f in col {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
            }
        };
        push_mat(&inv_view_proj);
        push_mat(&shadow_view_proj);
        let mut push4 = |v: [f32; 4]| {
            for f in v {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
        };
        push4([eye[0], eye[1], eye[2], 0.0]);
        push4([origin[0], origin[1], origin[2], density]);
        push4([color_intensity[0], color_intensity[1], color_intensity[2], exponent]);
        push4(shadow_uv_rect);
        push4([has_shadow, MAX_RANGE, 0.0, 0.0]);

        let ubo = renderer.make_uniform_buffer(&bytes, bytes.len() as u64);
        Some(renderer.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetrics_bg"),
            layout: &renderer.volumetrics_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(shadow_atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&renderer.shadow_comparison_sampler),
                },
            ],
        }))
    }

    /// Render the shadow atlas once at scene setup (shadow-casting lights
    /// are treated as static, same precedent as mesh MVPs — see
    /// `build_mesh3d`'s own doc comment). For every `light` object with
    /// `castshadow: true` (capped at `engine::shadow::MAX_SHADOW_LIGHTS`),
    /// depth-renders every mesh3d object into that light's tile of one
    /// shared atlas texture — the architecture recovered from the original
    /// binary's `_rt_shadowAtlas` (see the Ghidra report's shadow-mapping
    /// follow-up), not a separate texture per light.
    ///
    /// Returns the atlas texture (a harmless 1×1 dummy when there are no
    /// shadow-casting lights or no mesh3d geometry, so `build_mesh3d`'s bind
    /// group always has a valid view to reference), a per-`scene.lights()`
    /// shadow-slot index (`None` = doesn't cast a shadow, or capped out),
    /// and the per-slot `(view_proj, uv_rect)` data for `mesh3d_lighting_bytes`.
    fn build_shadow_atlas(
        renderer: &GpuSceneRenderer,
        mesh3d_layers: &[crate::engine::render::Mesh3dLayer],
        lights: &[(
            crate::engine::lighting::Light,
            bool,
            Option<(f32, f32)>,
            Option<String>,
        )],
    ) -> (
        wgpu::Texture,
        Vec<Option<usize>>,
        Vec<(crate::engine::camera3d::Mat4, [f32; 4])>,
    ) {
        use crate::engine::lighting::Light;
        use crate::engine::shadow;

        let mut shadow_slots: Vec<Option<usize>> = vec![None; lights.len()];

        // Needed to build either projection (`point_light_view_proj`'s own
        // aim-and-size target, `spot_light_view_proj`'s near/far sizing) —
        // computed once up front now that both light types consume it,
        // rather than after filtering casters like the point-only version
        // did.
        let (center, radius) = shadow::scene_bounds(mesh3d_layers);
        let shadow_casters: Vec<(usize, crate::engine::camera3d::Mat4)> = lights
            .iter()
            .enumerate()
            .filter_map(|(i, (light, casts_shadow, _volumetrics, _))| {
                if !*casts_shadow {
                    return None;
                }
                let view_proj = match light {
                    Light::Point { origin, .. } => {
                        shadow::point_light_view_proj(*origin, center, radius)
                    }
                    Light::Spot {
                        origin,
                        direction,
                        outer_cone_degrees,
                        ..
                    } => shadow::spot_light_view_proj(
                        *origin,
                        *direction,
                        *outer_cone_degrees,
                        center,
                        radius,
                    ),
                    // Directional/tube shadows aren't built yet — see the
                    // Ghidra report's shadow-mapping follow-up. Directional
                    // has no natural "position" to cast a perspective
                    // shadow from (it'd need an orthographic projection, a
                    // different pipeline shape); tube isn't constructed by
                    // `Scene::lights()` at all yet.
                    _ => return None,
                };
                Some((i, view_proj))
            })
            .take(shadow::MAX_SHADOW_LIGHTS)
            .collect();

        if shadow_casters.is_empty() || mesh3d_layers.is_empty() {
            let dummy = renderer.device.create_texture(&wgpu::TextureDescriptor {
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
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            return (dummy, shadow_slots, Vec::new());
        }

        let (atlas_w, atlas_h, tiles) = shadow::pack_tiles(
            shadow_casters.len(),
            shadow::SHADOW_TILE_SIZE,
            shadow::SHADOW_TILE_SIZE * 4,
        );

        let atlas = renderer.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_atlas"),
            size: wgpu::Extent3d {
                width: atlas_w,
                height: atlas_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: MESH3D_DEPTH_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let atlas_view = atlas.create_view(&Default::default());

        // Geometry uploaded once, reused for every shadow-casting light's
        // tile pass. Duplicates the upload `build_mesh3d` does later for the
        // final draw — real content has ~0 shadow-casting lights (see the
        // Ghidra report), so trading a little redundant GPU upload for a
        // simple, independent setup path is the right side of that trade.
        let geometry: Vec<(wgpu::Buffer, wgpu::Buffer, u32)> = mesh3d_layers
            .iter()
            .map(|m| Self::upload_mesh3d_geometry(renderer, &m.mesh))
            .collect();

        let mut slot_data = Vec::with_capacity(shadow_casters.len());
        let mut encoder = renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shadow_atlas_pass"),
            });
        for (slot, ((light_idx, view_proj), tile)) in shadow_casters.iter().zip(tiles.iter()).enumerate() {
            let view_proj = *view_proj;
            let uv_rect = shadow::tile_uv_rect(*tile, atlas_w, atlas_h);
            shadow_slots[*light_idx] = Some(slot);
            slot_data.push((view_proj, uv_rect));

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_tile"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &atlas_view,
                    depth_ops: Some(wgpu::Operations {
                        // Only the first tile clears — clearing is a
                        // whole-texture operation in wgpu, so clearing again
                        // per tile would erase every tile drawn before it.
                        // Every pixel outside every tile's scissor rect stays
                        // at this initial 1.0 (far) regardless.
                        load: if slot == 0 {
                            wgpu::LoadOp::Clear(1.0)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_viewport(tile.x as f32, tile.y as f32, tile.w as f32, tile.h as f32, 0.0, 1.0);
            pass.set_scissor_rect(tile.x, tile.y, tile.w, tile.h);
            pass.set_pipeline(&renderer.mesh3d_shadow_pipeline);
            for (layer, (vbuf, ibuf, index_count)) in mesh3d_layers.iter().zip(geometry.iter()) {
                let model = crate::engine::camera3d::model_matrix(layer.origin, layer.angles, layer.scale);
                let light_mvp = crate::engine::camera3d::mat4_mul(&view_proj, &model);
                let mut mvp_bytes = Vec::with_capacity(64);
                for col in light_mvp.iter() {
                    for f in col {
                        mvp_bytes.extend_from_slice(&f.to_le_bytes());
                    }
                }
                let mvp_buf = renderer.make_uniform_buffer(&mvp_bytes, 64);
                let bg = renderer.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("shadow_pass_bg"),
                    layout: &renderer.shadow_pass_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: mvp_buf.as_entire_binding(),
                    }],
                });
                pass.set_bind_group(0, &bg, &[]);
                pass.set_vertex_buffer(0, vbuf.slice(..));
                pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*index_count, 0, 0..1);
            }
        }
        renderer.queue.submit(std::iter::once(encoder.finish()));

        (atlas, shadow_slots, slot_data)
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
        mesh: &crate::engine::mesh3d::Mesh3d,
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
        cam: &crate::engine::camera3d::PerspectiveCamera,
        layer: &crate::engine::render::Mesh3dLayer,
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

    /// Recompose parent chains for this frame.
    ///
    /// `apply_parent_transforms` bakes the chain at load, which is correct only
    /// while the chain is static. A scripted parent moves every frame, and
    /// those parent nodes are usually image-less — they have no `Layer`, so the
    /// per-layer script loop never touches them. This evaluates every object's
    /// transform scripts (parents included), recomposes each chain, and pushes
    /// the result into whatever the object actually draws as.
    ///
    /// Skipped entirely unless the scene both parents something and scripts a
    /// transform — true for 8 of 197 real scenes.
    fn update_parent_transforms(&mut self) {
        if !self.transform_graph.needs_per_frame() {
            return;
        }
        use crate::engine::render::Xform;

        // Evaluate in declaration order: these scripts talk to each other
        // through the shared `shared` object (one writes `shared.gjz`, another
        // reads it), and declaration order is the only ordering WE implies.
        let mut locals = self.current_locals.clone();
        let mut disabled = vec![false; self.transform_graph.parent.len()];
        for (i, scripts) in self.transform_graph.scripts.iter().enumerate() {
            // Previous frame's value, not the authored one — see `current_locals`.
            let base = self.current_locals[i];
            // `thisLayer`/`disablepropagation()` — the object-attachment
            // scripting API (see `engine::script::OBJECT_ATTACHMENT_API_JS`)
            // — needs to know which object is about to run, and this is the
            // one place every object (including image-less parent nodes)
            // gets its scripts evaluated.
            self.script_ctx.set_current_object(
                self.transform_graph.name[i].as_deref(),
                self.transform_graph.id[i],
                [base.t[0] as f32, base.t[1] as f32, base.t[2] as f32],
                [base.r[0] as f32, base.r[1] as f32, base.r[2] as f32],
                [base.s[0] as f32, base.s[1] as f32, base.s[2] as f32],
                [0.0, 0.0],
            );
            let eval = |ctx: &mut crate::engine::script::ScriptContext,
                        src: &Option<String>,
                        b: [f64; 3]| match src {
                Some(src) => ctx
                    .eval_update_vec3(src, [b[0] as f32, b[1] as f32, b[2] as f32])
                    .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
                    .unwrap_or(b),
                None => b,
            };
            locals[i] = Xform {
                t: eval(&mut self.script_ctx, &scripts[0], base.t),
                r: eval(&mut self.script_ctx, &scripts[1], base.r),
                s: eval(&mut self.script_ctx, &scripts[2], base.s),
            };
            disabled[i] = self.script_ctx.take_propagation_disabled();
        }
        self.current_locals.clone_from(&locals);
        let worlds = self.transform_graph.world(&locals, &disabled);
        // The PARENT's world transform, not the object's own — `worlds[i]`
        // already folds in object i's local transform, and the layer/mesh
        // values we apply it to are pre-parent locals. Applying the full world
        // would compose i's own transform twice. This mirrors exactly what
        // `apply_parent_transforms` does at load.
        let parent_world = |i: usize| -> crate::engine::render::Xform {
            match self.transform_graph.parent.get(i).copied().flatten() {
                Some(p) if p != i => worlds[p],
                _ => Xform::IDENTITY,
            }
        };

        // 3D meshes: rebuild the MVP from the recomposed world transform.
        // Computed up front so the write-back can borrow `self.mesh3d` mutably.
        let mesh_updates: Vec<(Vec<u8>, f32)> = match &self.camera3d {
            Some(cam) => self
                .mesh3d
                .iter()
                .map(|m| {
                    let w = parent_world(m.order_index);
                    let (lo, la, ls) = m.local;
                    let o = w.apply_point([lo[0] as f64, lo[1] as f64, lo[2] as f64]);
                    let centre = [o[0] as f32, o[1] as f32, o[2] as f32];
                    let angles: [f32; 3] = std::array::from_fn(|i| la[i] + w.r[i] as f32);
                    let scale: [f32; 3] = std::array::from_fn(|i| ls[i] * w.s[i] as f32);
                    let bytes = mesh3d_transform_bytes(cam, centre, angles, scale);
                    (bytes, cam.view_depth(centre))
                })
                .collect(),
            None => Vec::new(),
        };
        for (m, (bytes, depth)) in self.mesh3d.iter_mut().zip(mesh_updates) {
            self.renderer.queue.write_buffer(&m.ubo, 0, &bytes);
            m.depth = depth;
        }

        // 2D layers, both projections. The rect derivations differ (the
        // perspective one projects the quad's corners through the camera) but
        // both start from the same recomposed world transform.
        let (w_px, h_px) = (self.width as f64, self.height as f64);
        for layer in &mut self.layers {
            let w = parent_world(layer.order_index);
            if w.is_identity() {
                continue;
            }
            let o = w.apply_point(layer.local_origin);
            let size_px = [
                layer.effective_size[0] * layer.local_scale[0] * w.s[0],
                layer.effective_size[1] * layer.local_scale[1] * w.s[1],
            ];
            match &self.camera3d {
                Some(cam) => {
                    let angles: [f32; 3] =
                        std::array::from_fn(|i| layer.local_angles[i] + w.r[i] as f32);
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
                    layer.angle = -(layer.local_angles[2] + w.r[2] as f32);
                }
            }
        }
    }

    /// Draw every static mesh in one depth-tested pass, over the cleared scene
    /// target and under the 2D layers.
    ///
    /// ponytail: meshes always draw before all 2D layers rather than
    /// interleaving by `order_index` — real mesh scenes put their 2D content on
    /// top (overlays/UI) so it hasn't mattered. Interleaving needs the depth
    /// buffer shared with the composite pipeline.
    /// Draw one mesh, depth-tested, into the scene target.
    ///
    /// Per-mesh rather than one batched pass because meshes interleave with 2D
    /// layers by `order_index` — a scene's background layers sort *before* its
    /// meshes and must not paint over them (3453730450's moon sits at order 10
    /// behind nine full-screen fills). `clear_depth` is set for the first mesh
    /// of a frame; later ones load the existing buffer so they still occlude
    /// each other correctly.
    fn draw_one_mesh3d(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        idx: usize,
        clear_depth: bool,
    ) {
        let (Some(depth), Some(m)) = (&self.mesh3d_depth, self.mesh3d.get(idx)) else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh3d_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: if clear_depth {
                        wgpu::LoadOp::Clear(1.0)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(
            &self.renderer.mesh3d_pipelines[m.nocull as usize | ((!m.depthtest as usize) << 1)],
        );
        pass.set_bind_group(0, &m.bind_group, &[]);
        pass.set_vertex_buffer(0, m.vbuf.slice(..));
        pass.set_index_buffer(m.ibuf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..m.index_count, 0, 0..1);
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
    }

    /// Render the next frame into the internal scene target.
    pub fn render(&mut self) {
        let time = self.start.elapsed().as_secs_f32();
        let delta = (time - self.last_time).max(0.0);
        self.last_time = time;
        let dynamics = self.dynamics.update(time, delta, self.mouse_norm);

        // Tick SceneScript-driven properties for this frame. `engine.timeOfDay`
        // is fractional hours in [0,24); we derive it from the wall clock (UTC
        // for now — local-time zoning is a follow-up).
        {
            let secs_into_day = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() % 86_400)
                .unwrap_or(0);
            let time_of_day = secs_into_day as f32 / 3600.0;
            let (w, h, persp) = (self.width, self.height, self.perspective);
            // Every scene object's current name/id/transform, for
            // `thisScene.getLayer(...)` (the object-attachment API — see
            // `engine::script::OBJECT_ATTACHMENT_API_JS`) to search. Built
            // from last frame's resolved locals (`update_parent_transforms`,
            // which refreshes them, runs later this same frame) — one frame
            // of staleness here, same as every other cross-script read in
            // this codebase (e.g. the `shared` object's declaration-order
            // dependency, documented below).
            let layer_snapshots: Vec<crate::engine::script::LayerSnapshot> = self
                .transform_graph
                .name
                .iter()
                .zip(self.transform_graph.id.iter())
                .zip(self.current_locals.iter())
                .map(|((name, id), xform)| crate::engine::script::LayerSnapshot {
                    name: name.as_deref(),
                    id: *id,
                    origin: [xform.t[0] as f32, xform.t[1] as f32, xform.t[2] as f32],
                    angles: [xform.r[0] as f32, xform.r[1] as f32, xform.r[2] as f32],
                    scale: [xform.s[0] as f32, xform.s[1] as f32, xform.s[2] as f32],
                })
                .collect();
            #[cfg(target_os = "linux")]
            let media_update = self.media.as_ref().and_then(|w| w.try_recv());
            let ctx = &mut self.script_ctx;
            // `delta`, not `time - self.last_time`: last_time was already
            // advanced to `time` above, so that expression is always exactly 0
            // and every `engine.frametime` script silently multiplied by zero.
            ctx.set_time(time, time_of_day, delta);
            ctx.set_layers(&layer_snapshots);
            #[cfg(target_os = "linux")]
            if let Some(info) = media_update {
                use crate::engine::media::PlaybackState;
                ctx.set_media(
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
            for layer in &mut self.layers {
                // Refreshes `thisLayer` so this object's own scripts can use
                // the object-attachment API (`lookAt`, `setParent`,
                // `getTransformMatrix`, etc.) against real data.
                ctx.set_current_object(
                    self.transform_graph
                        .name
                        .get(layer.order_index)
                        .and_then(|n| n.as_deref()),
                    layer.object_id,
                    [
                        layer.origin_base[0] as f32,
                        layer.origin_base[1] as f32,
                        layer.origin_base[2] as f32,
                    ],
                    layer.angles_base,
                    layer.scale_base,
                    layer.parallax_depth,
                );
                if let Some(script) = &layer.alpha_script {
                    layer.alpha = ctx
                        .eval_update(script, layer.alpha_base)
                        .unwrap_or(layer.alpha_base);
                }
                // Transform scripts. `visible` gates the draw; scale/origin
                // rebuild `rect`; angles rebuilds `angle`. The rect rebuild is
                // 2D-only (perspective layers keep their camera-projected rect).
                // Each script borrow is scoped to its eval so the `rect`/`angle`
                // writes below don't overlap it.
                if let Some(v) = layer
                    .transform_scripts
                    .visible
                    .as_ref()
                    .and_then(|s| ctx.eval_update_bool(s, layer.visible_base))
                {
                    layer.visible = v;
                }
                if !persp
                    && (layer.transform_scripts.scale.is_some()
                        || layer.transform_scripts.origin.is_some())
                {
                    let scale = match &layer.transform_scripts.scale {
                        Some(s) => ctx
                            .eval_update_vec3(s, layer.scale_base)
                            .unwrap_or(layer.scale_base),
                        None => layer.scale_base,
                    };
                    let origin_base = [
                        layer.origin_base[0] as f32,
                        layer.origin_base[1] as f32,
                        layer.origin_base[2] as f32,
                    ];
                    let origin = match &layer.transform_scripts.origin {
                        Some(s) => ctx.eval_update_vec3(s, origin_base).unwrap_or(origin_base),
                        None => origin_base,
                    };
                    let size_px = [
                        layer.effective_size[0] * scale[0] as f64,
                        layer.effective_size[1] * scale[1] as f64,
                    ];
                    layer.rect = [
                        (2.0 * origin[0] as f64 / w as f64 - 1.0) as f32,
                        (2.0 * origin[1] as f64 / h as f64 - 1.0) as f32,
                        (size_px[0] / w as f64) as f32,
                        (size_px[1] / h as f64) as f32,
                    ];
                }
                if !persp {
                    if let Some(a) = layer
                        .transform_scripts
                        .angles
                        .as_ref()
                        .and_then(|s| ctx.eval_update_vec3(s, layer.angles_base))
                    {
                        layer.angle = -a[2];
                    }
                }
                // Script-driven text: re-evaluate; only a CHANGED string pays
                // for rasterization + upload (a clock re-rasterizes once per
                // displayed unit — second or minute — not per frame).
                if let Some(td) = &mut layer.text_dynamic {
                    if let Some(new_text) = ctx.eval_update_string(
                        &td.script,
                        &td.last_text,
                        td.script_properties.as_ref(),
                    ) {
                        if new_text != td.last_text && !new_text.is_empty() {
                            if let Some(img) = crate::engine::text::rasterize(
                                &td.font_data,
                                &new_text,
                                td.point_size,
                            ) {
                                // The quad's rect was sized at build to the
                                // object's authored `size` box and projected
                                // through the camera/parent transform; the text
                                // bitmap is stretched to fill it. A content
                                // change only swaps that texture — the quad
                                // stays put, so leave `rect` untouched (the old
                                // per-change ortho recompute is what moved the
                                // clock off-screen / shrank it).
                                // ponytail: fixed authored-size quad — right for
                                // the clock/date (constant-width). Text that
                                // must grow to fit would need its box re-fit.
                                layer.frames[0] = self.renderer.upload_texture(&img);
                                td.last_text = new_text;
                            }
                        }
                    }
                }
            }
        }

        // Recompose parent chains now that this frame's scripts have run —
        // a scripted parent moves its whole subtree.
        self.update_parent_transforms();

        // Refresh the g_AudioSpectrum* UBO from the latest captured window, and
        // drive audio-reactive particle emitters from the overall loudness.
        if let Some(cap) = &self.audio_capture {
            let spectrum = cap.spectrum();
            self.renderer.queue.write_buffer(
                &self.renderer.audio_buf,
                0,
                &spectrum.to_uniform_bytes(),
            );
            let level = spectrum.average_level();
            for system in &mut self.particle_systems {
                system.set_audio_level(level);
            }
        }

        let target_view = self.target.create_view(&Default::default());
        let mut encoder =
            self.renderer
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frame_encoder"),
                });

        // 1. Clear scene target.
        {
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.clear_color[0],
                            g: self.clear_color[1],
                            b: self.clear_color[2],
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
        }

        // Re-pose animated puppet layers at a capped rate: skin the mesh at
        // the current animation time and re-upload frames[0]. CPU skinning +
        // rasterization runs ~30ms at 4K, so this ticks at
        // PUPPET_UPDATE_INTERVAL rather than every frame — the idle-style
        // animations these carry (breathing, hair sway) are far slower than
        // even that.
        // Pull the newest decoded frame for embedded-video layers (the
        // ffmpeg thread paces itself; an empty channel means no new frame
        // this tick and the current texture stays).
        for i in 0..self.layers.len() {
            let Some(stream) = self.layers[i].video.clone() else {
                continue;
            };
            let Some(frame) = stream.lock().ok().and_then(|s| s.latest_frame()) else {
                continue;
            };
            let tex = self.renderer.upload_texture(&frame);
            self.layers[i].frames[0] = tex;
        }

        const PUPPET_UPDATE_INTERVAL: f32 = 1.0 / 15.0;
        for i in 0..self.layers.len() {
            let Some(runtime) = self.layers[i].puppet.clone() else {
                continue;
            };
            if (time - self.layers[i].puppet_posed_at).abs() < PUPPET_UPDATE_INTERVAL {
                continue;
            }
            let (w, h) = (runtime.atlas.width(), runtime.atlas.height());
            let posed = runtime.render_at(time, w, h);
            let tex = self.renderer.upload_texture(&posed);
            self.layers[i].frames[0] = tex;
            self.layers[i].puppet_posed_at = time;
        }

        // 2/3. Image layers and particle systems, interleaved by
        // `order_index` (true scene z-order, matching the reference's single
        // shared per-object render order) instead of drawing all particles
        // after all images.
        enum DrawItem {
            Image(usize),
            Particle(usize),
            Mesh(usize),
        }
        let mut items: Vec<(usize, DrawItem)> = self
            .layers
            .iter()
            .enumerate()
            // A `visible` script can hide a layer this frame — drop it here.
            .filter(|(_, l)| l.visible)
            .map(|(i, l)| (l.order_index, DrawItem::Image(i)))
            .chain(
                self.particle_systems
                    .iter()
                    .enumerate()
                    .map(|(i, _)| (self.particle_order[i], DrawItem::Particle(i))),
            )
            // Meshes interleave by scene order like everything else: a scene's
            // background layers can sort before its meshes.
            .chain(
                self.mesh3d
                    .iter()
                    .enumerate()
                    .map(|(i, m)| (m.order_index, DrawItem::Mesh(i))),
            )
            .collect();
        items.sort_by_key(|(order, _)| *order);

        if self.perspective {
            // Painter's algorithm for 3D scenes: draw image layers strictly
            // back-to-front by view-space depth (no depth buffer yet), with
            // culled quads (depth < 0) dropped. Particles keep drawing after
            // images, as in the ortho path's known simplification.
            items.retain(|(_, item)| match item {
                DrawItem::Image(i) => self.layers[*i].depth >= 0.0,
                DrawItem::Particle(_) => true,
                // Never culled on centre depth — a skybox surrounds the camera.
                DrawItem::Mesh(_) => true,
            });
            items.sort_by(|(oa, a), (ob, b)| {
                let key = |it: &DrawItem, order: usize| match it {
                    DrawItem::Image(i) => (0, -self.layers[*i].depth, order),
                    // Meshes share the images' back-to-front ordering so a
                    // distant backdrop layer still draws behind them.
                    DrawItem::Mesh(i) => (0, -self.mesh3d[*i].depth, order),
                    DrawItem::Particle(_) => (1, 0.0, order),
                };
                key(a, *oa)
                    .partial_cmp(&key(b, *ob))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // The depth buffer is cleared by the frame's first mesh; later meshes
        // load it so they still occlude one another.
        let mut mesh_depth_cleared = false;
        for (_, item) in items {
            match item {
                DrawItem::Mesh(idx) => {
                    self.draw_one_mesh3d(&mut encoder, &target_view, idx, !mesh_depth_cleared);
                    mesh_depth_cleared = true;
                }
                DrawItem::Image(layer_idx) => {
                    self.draw_image_layer_gpu(
                        &mut encoder,
                        &target_view,
                        layer_idx,
                        time,
                        dynamics,
                    );
                }
                DrawItem::Particle(idx) => {
                    self.draw_particle_layer_gpu(&mut encoder, &target_view, idx, delta);
                }
            }
        }

        // 4. Volumetric light shafts (before bloom, so a bright shaft can
        // itself contribute to the bloom pass — matching how particles are
        // ordered before bloom for the same reason, see their own doc
        // comment on `particle_systems`).
        self.record_volumetrics(&mut encoder, &target_view);

        // 5. Scene bloom chain.
        if self.bloom.enabled {
            self.record_bloom(&mut encoder, &target_view);
        }

        self.renderer.queue.submit(Some(encoder.finish()));
    }

    fn draw_image_layer_gpu(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        layer_idx: usize,
        time: f32,
        dynamics: CameraFrameDynamics,
    ) {
        {
            let layer = &self.layers[layer_idx];
            let frame_idx = if layer.frames.len() > 1 && layer.frame_duration_ms > 0 {
                ((time * 1000.0 / layer.frame_duration_ms as f32) as usize) % layer.frames.len()
            } else {
                0
            };
            let base_tex = &layer.frames[frame_idx];
            let base_view = base_tex.create_view(&Default::default());
            let tex_res = |t: &wgpu::Texture| -> [f32; 4] {
                let (tw, th) = (t.width() as f32, t.height() as f32);
                [tw, th, tw, th]
            };

            // Engine uniforms shared by every pass of this layer.
            let engine_base = EngineUniforms {
                time,
                daytime: daytime_fraction(),
                texel_size: [1.0 / self.width as f32, 1.0 / self.height as f32],
                pointer: self.mouse_norm,
                resolutions: [[
                    self.width as f32,
                    self.height as f32,
                    self.width as f32,
                    self.height as f32,
                ]; 8],
                color: layer.color,
                alpha: layer.alpha,
                brightness: layer.brightness,
            };

            // Ping-pong through this layer's effect passes (own object-sized pair).
            let pp = [
                self.fbo_pool.get(&pingpong_key(layer_idx, 0)).unwrap(),
                self.fbo_pool.get(&pingpong_key(layer_idx, 1)).unwrap(),
            ];
            let layer_sampler = self
                .renderer
                .sampler_for(layer.no_interpolation, layer.clamp_uvs);
            // Effects reference their owner by raw scene-object index
            // (layer.order_index), NOT by position in `self.layers` —
            // positional matching applied effects to whichever layer
            // happened to share the count once particles/skipped objects
            // offset the two spaces.
            let obj_index = layer.order_index;
            let has_effects = self
                .scene_effects
                .iter()
                .any(|inst| inst.layer_idx == obj_index);

            // Base material pass (genericimage3-equivalent): reference always
            // renders `texture * g_Color4` into the object's own FBO *before*
            // any effects run, so effects see the tinted/alpha'd image rather
            // than the raw texture (CImage::setup pushes the material's own
            // passes first, ahead of the effect chain). Skipped when the layer
            // has no effects: with nothing in between, tinting at the final
            // composite step produces an identical result for a fraction of
            // the draw calls.
            let mut cur: Option<usize> = None; // None = still reading base texture
            if has_effects {
                let base_tint = [
                    layer.color[0] * layer.brightness,
                    layer.color[1] * layer.brightness,
                    layer.color[2] * layer.brightness,
                ];
                let base_pass_buf = self.renderer.make_uniform_buffer(
                    &composite_params(
                        layer.alpha,
                        0,
                        [0.0, 0.0],
                        base_tint,
                        0.0,
                        FULLSCREEN_RECT,
                        1.0,
                        [1.0, 1.0],
                    ),
                    64,
                );
                let base_pass_bg = self.renderer.make_base_bind_group(
                    &base_view,
                    layer_sampler,
                    &base_pass_buf,
                    &[],
                );
                self.renderer.run_pass(
                    encoder,
                    &self.renderer.base_pass_pipeline,
                    &base_pass_bg,
                    None,
                    &pp[0].view(),
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    "base_material_pass",
                    6,
                    &[],
                );
                cur = Some(0);
            }
            for (instance_idx, inst) in self
                .scene_effects
                .iter()
                .enumerate()
                .filter(|(_, inst)| inst.layer_idx == obj_index)
            {
                let Some(runtime) = self
                    .effect_runtimes
                    .get(instance_idx)
                    .and_then(|r| r.as_ref())
                else {
                    continue;
                };
                for pass in &runtime.passes {
                    let Some(pipeline) = self.renderer.find_effect_pipeline(&pass.key) else {
                        continue;
                    };

                    // Chain source before bind-0 overrides ("previous" binds
                    // and the default input both refer to it).
                    let (chain_view, chain_res) = match cur {
                        None => (base_view.clone(), tex_res(base_tex)),
                        Some(i) => (pp[i].view(), tex_res(&pp[i].texture)),
                    };

                    let mut engine = engine_base;
                    engine.resolutions[0] = chain_res;

                    // Extra texture slots: shader/material textures first,
                    // FBO binds override.
                    let mut extra: Vec<Option<wgpu::TextureView>> = vec![None; 7];
                    if let Some(texs) = self.renderer.dynamic_textures.get(&pass.tex_key) {
                        for (i, tex) in texs.iter().enumerate().take(7) {
                            extra[i] = Some(tex.create_view(&Default::default()));
                            engine.resolutions[i + 1] = tex_res(tex);
                        }
                    }
                    let mut src_view = chain_view.clone();
                    for (slot, fbo_name) in &pass.binds {
                        // A bind named "previous" is the chain input itself.
                        let (view, res) = if fbo_name == "previous" {
                            (chain_view.clone(), chain_res)
                        } else if let Some(rt) = fbo_name
                            .starts_with("_rt_imageLayerComposite")
                            .then(|| self.fbo_pool.get(fbo_name.as_str()))
                            .flatten()
                        {
                            // The owning layer's own composite, captured when it
                            // drew (see draw_image_layer_gpu). Transparent when
                            // that layer hasn't drawn yet this frame, which adds
                            // nothing — the safe direction.
                            (rt.view(), tex_res(&rt.texture))
                        } else if fbo_name == "_rt_FullFrameBuffer"
                            || fbo_name == "_rt_MipMappedFrameBuffer"
                            // A per-layer composite we never captured falls back
                            // to the scene-so-far snapshot: effects run before
                            // their layer composites, so the scene target holds
                            // "everything behind this layer". Without any bind
                            // the sampler keeps its `util/white` default — which
                            // is what blended 2821407073's mountains to white.
                            || fbo_name.starts_with("_rt_imageLayerComposite")
                        {
                            // The wallpaper-global scene buffer (CWallpaper.cpp
                            // creates it at scene size; MipMapped is an alias).
                            // Effects run before their layer composites, so the
                            // scene target currently holds exactly "the scene
                            // behind this layer" — snapshot it, since a pass
                            // can't sample its own render target.
                            encoder.copy_texture_to_texture(
                                self.target.as_image_copy(),
                                self.scene_copy.as_image_copy(),
                                wgpu::Extent3d {
                                    width: self.width,
                                    height: self.height,
                                    depth_or_array_layers: 1,
                                },
                            );
                            (
                                self.scene_copy.create_view(&Default::default()),
                                [
                                    self.width as f32,
                                    self.height as f32,
                                    self.width as f32,
                                    self.height as f32,
                                ],
                            )
                        } else {
                            let key = named_fbo_key(instance_idx, fbo_name);
                            match self.fbo_pool.get(&key) {
                                Some(rt) => (rt.view(), tex_res(&rt.texture)),
                                None => continue,
                            }
                        };
                        if *slot == 0 {
                            src_view = view;
                            engine.resolutions[0] = res;
                        } else if (*slot as usize) <= 7 {
                            extra[(*slot - 1) as usize] = Some(view);
                            engine.resolutions[*slot as usize] = res;
                        }
                    }

                    let params = if pass.hardcoded {
                        make_effect_params(&inst.name, time, &pass.values)
                    } else {
                        let keys = self
                            .renderer
                            .dynamic_uniform_keys
                            .get(&pass.key)
                            .map(|v| v.as_slice())
                            .unwrap_or(&[]);
                        make_params_from_translated_typed(keys, &engine, &pass.values)
                    };

                    let composite_buf = self
                        .renderer
                        .make_uniform_buffer(&passthrough_composite_params(), 64);
                    let layer_sampler = self
                        .renderer
                        .sampler_for(layer.no_interpolation, layer.clamp_uvs);
                    let base_bg = self.renderer.make_base_bind_group(
                        &src_view,
                        layer_sampler,
                        &composite_buf,
                        &extra,
                    );
                    let param_buf = self.renderer.make_uniform_buffer(&params, 16);
                    let effect_bg = self.renderer.make_effect_bind_group(&param_buf);

                    let next = cur.map(|i| 1 - i).unwrap_or(0);
                    let dst_view = match &pass.target {
                        Some(fbo_name) => {
                            let key = named_fbo_key(instance_idx, fbo_name);
                            match self.fbo_pool.get(&key) {
                                Some(rt) => rt.view(),
                                None => pp[next].view(),
                            }
                        }
                        None => pp[next].view(),
                    };

                    // Real-VS pipelines draw a 6-vertex quad from attribute
                    // buffers; synthetic-VS ones draw the fullscreen triangle.
                    let vertex_count = if pass.vertex_buffers.is_empty() { 3 } else { 6 };

                    self.renderer.run_pass(
                        encoder,
                        pipeline,
                        &base_bg,
                        Some(&effect_bg),
                        &dst_view,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        "effect_pass",
                        vertex_count,
                        &pass.vertex_buffers,
                    );

                    // Named-target passes render aside; the chain continues
                    // from the same source.
                    if pass.target.is_none() {
                        cur = Some(next);
                    }
                }
            }

            // Composite the layer onto the scene target with camera dynamics.
            let uv_offset = [
                dynamics.shake_offset[0]
                    + dynamics.parallax_displacement[0] * layer.parallax_depth[0],
                dynamics.shake_offset[1]
                    + dynamics.parallax_displacement[1] * layer.parallax_depth[1],
            ];
            // Color/alpha/brightness were already baked in by the base
            // material pass above when this layer has effects; re-applying
            // them here would double them up.
            let (tint, opacity) = if has_effects {
                ([1.0, 1.0, 1.0], dynamics.fade)
            } else {
                (
                    [
                        layer.color[0] * layer.brightness,
                        layer.color[1] * layer.brightness,
                        layer.color[2] * layer.brightness,
                    ],
                    dynamics.fade * layer.alpha,
                )
            };
            let src_view = match cur {
                None => base_view,
                Some(i) => pp[i].view(),
            };
            // True perspective-quad silhouette, normal blending only — see
            // `vs_quad3d`/`fs_quad3d` and the Ghidra report's quad-warp
            // finding. `uv_offset` (shake/parallax) isn't applied on this
            // path: under true perspective it belongs on the camera or the
            // object's world position, not a flat post-hoc UV shift, and
            // wiring that through is future work, not this fix. Falls
            // through to the same draw + `_rt_imageLayerComposite` capture
            // code below as the rect-based path, just with a different
            // pipeline/bind group, so neither can drift out of sync with
            // the other.
            let (pipeline, base_bg) = if layer.blend_mode == 0 && layer.quad_corners.is_some() {
                let corners = layer.quad_corners.unwrap();
                let mut params = Vec::with_capacity(80);
                for c in corners {
                    params.extend_from_slice(&c[0].to_le_bytes());
                    params.extend_from_slice(&c[1].to_le_bytes());
                    params.extend_from_slice(&0.0f32.to_le_bytes());
                    params.extend_from_slice(&0.0f32.to_le_bytes());
                }
                for f in [tint[0], tint[1], tint[2], opacity] {
                    params.extend_from_slice(&f.to_le_bytes());
                }
                let quad3d_buf = self.renderer.make_uniform_buffer(&params, 80);
                let quad3d_bg = self.renderer.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("quad3d_bg"),
                    layout: &self.renderer.quad3d_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: quad3d_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&src_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(layer_sampler),
                        },
                    ],
                });
                (&self.renderer.quad3d_pipeline, quad3d_bg)
            } else {
                let composite_buf = self.renderer.make_uniform_buffer(
                    &composite_params(
                        opacity,
                        layer.blend_mode as i32,
                        uv_offset,
                        tint,
                        layer.angle,
                        layer.rect,
                        self.width as f32 / self.height as f32,
                        [self.width as f32, self.height as f32],
                    ),
                    64,
                );
                // Photoshop-style blend modes read the destination
                // in-shader, so snapshot the scene target before drawing
                // over it.
                let extra: Vec<Option<wgpu::TextureView>> = if layer.blend_mode != 0 {
                    encoder.copy_texture_to_texture(
                        self.target.as_image_copy(),
                        self.scene_copy.as_image_copy(),
                        wgpu::Extent3d {
                            width: self.width,
                            height: self.height,
                            depth_or_array_layers: 1,
                        },
                    );
                    vec![Some(self.scene_copy.create_view(&Default::default()))]
                } else {
                    vec![]
                };
                let base_bg = self.renderer.make_base_bind_group(
                    &src_view,
                    layer_sampler,
                    &composite_buf,
                    &extra,
                );
                (self.renderer.composite_pipeline(layer.blend_mode), base_bg)
            };
            self.renderer.run_pass(
                encoder,
                pipeline,
                &base_bg,
                None,
                target_view,
                wgpu::LoadOp::Load,
                "composite_pass",
                6,
                &[],
            );

            // Capture this layer's own composite for effects on other objects
            // that bind `_rt_imageLayerComposite_<thisId>_*` (WE keeps one
            // buffer per object; ours is filled with the same draw, on a
            // cleared scene-sized target so only this layer is in it).
            if let Some(id) = layer.object_id {
                for suffix in ['a', 'b'] {
                    let name = format!("_rt_imageLayerComposite_{id}_{suffix}");
                    if let Some(rt) = self.fbo_pool.get(name.as_str()) {
                        self.renderer.run_pass(
                            encoder,
                            pipeline,
                            &base_bg,
                            None,
                            &rt.view(),
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            "layer_composite_capture",
                            6,
                            &[],
                        );
                    }
                }
            }
        }
    }

    /// Steps and CPU-renders one particle system into a scene-sized RGBA
    /// buffer, uploads it, and composites it like a normal fullscreen layer.
    /// Steps a particle system and, if anything is alive, CPU-rasters +
    /// uploads only its bounding box (`ParticleSystem::bounds()`) instead of
    /// a full scene-sized buffer, compositing it as a positioned quad (the
    /// same `rect`/`CompositeParams` mechanism image layers use) rather than
    /// a fullscreen passthrough. Systems with no living particles are
    /// skipped entirely — no raster, no allocation, no upload, no draw call.
    /// A further possible optimization (not done here): reusing one
    /// persistent GPU texture per system across frames instead of a fresh
    /// `upload_texture` each frame — deferred since safely doing that while
    /// the bbox size changes frame-to-frame needs UV-subregion plumbing the
    /// composite shader doesn't have today.
    fn draw_particle_layer_gpu(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        idx: usize,
        delta: f32,
    ) {
        // Escape hatch for A/B comparison against the old budgeted CPU
        // rasterizer during the GPU-pipeline transition.
        if std::env::var("WP_PARTICLE_CPU").is_ok() {
            self.draw_particle_layer_cpu_raster(encoder, target_view, idx, delta);
            return;
        }
        self.particle_systems[idx].step(delta);

        // Draw units: the parent system, then every live child instance
        // (each child preset has its own sprite/blending).
        let system = &self.particle_systems[idx];
        let assets = &self.particle_gpu_assets[idx];
        let mut units: Vec<(Vec<particle::GpuVertex>, &ParticleGpuTex, bool)> = Vec::new();
        let mut verts = Vec::new();
        system.emit_gpu_vertices(&mut verts, assets.parent.frames as usize);
        if !verts.is_empty() {
            units.push((verts, &assets.parent, self.particle_additive[idx]));
        }
        system.visit_gpu_children(&mut |inst, child_idx, child_additive| {
            let tex = assets.children.get(child_idx).unwrap_or(&assets.parent);
            let mut v = Vec::new();
            inst.emit_gpu_vertices(&mut v, tex.frames as usize);
            if !v.is_empty() {
                units.push((v, tex, child_additive));
            }
        });

        for (verts, tex, additive) in units {
            let bytes = particle::GpuVertex::as_bytes(&verts);
            let vbuf = self.renderer.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("particle_verts"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.renderer.queue.write_buffer(&vbuf, 0, &bytes);

            let mut params = Vec::with_capacity(16);
            for f in [
                self.width as f32,
                self.height as f32,
                tex.overbright,
                tex.frames as f32,
            ] {
                params.extend_from_slice(&f.to_le_bytes());
            }
            let pbuf = self.renderer.make_uniform_buffer(&params, 16);
            // Linear-repeat for ropes (V tiles past 1.0), linear-clamp for
            // sprite quads (padding-safe at the edges).
            let sampler = &self.renderer.samplers[if tex.repeat { 0 } else { 1 }];
            let bg = self
                .renderer
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("particle_bg"),
                    layout: &self.renderer.particle_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: vbuf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&tex.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: pbuf.as_entire_binding(),
                        },
                    ],
                });

            let pipeline = if additive {
                &self.renderer.particle_pipeline_add
            } else {
                &self.renderer.particle_pipeline_over
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("particle_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..verts.len() as u32, 0..1);
        }
    }

    /// The pre-GPU-pipeline particle path: budgeted CPU rasterization into a
    /// bbox buffer, uploaded and composited as a stretched quad. Kept behind
    /// WP_PARTICLE_CPU=1 for A/B comparison; the CPU compositor paths
    /// (render.rs/animated.rs) share the same underlying rasterizer.
    fn draw_particle_layer_cpu_raster(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        idx: usize,
        delta: f32,
    ) {
        let system = &mut self.particle_systems[idx];
        let _t_step = std::time::Instant::now();
        system.step(delta);
        let step_ms = _t_step.elapsed().as_secs_f32() * 1000.0;
        let Some((min_x, min_y, max_x, max_y)) = system.bounds() else {
            return;
        };
        let min_x = min_x.max(0.0);
        let min_y = min_y.max(0.0);
        let max_x = max_x.min(self.width as f32);
        let max_y = max_y.min(self.height as f32);
        if max_x <= min_x || max_y <= min_y {
            return;
        }
        let bw = (max_x - min_x).ceil().max(1.0) as u32;
        let bh = (max_y - min_y).ceil().max(1.0) as u32;

        // Cap the CPU rasterization cost: huge soft sprites (smoke clouds,
        // rain sheets with radii in the hundreds of pixels) can cover most
        // of a 4K scene — rasterize into a budgeted buffer and let the
        // composite quad stretch it back up. Cost scales with the square of
        // this factor; the blur from upscaling is invisible on sprites that
        // are soft gradients to begin with (e.g. 2491009392's rain+smoke
        // went from ~0.6s/frame of CPU to real-time).
        const PARTICLE_PIXEL_BUDGET: f32 = 1_500_000.0;
        let area = bw as f32 * bh as f32;
        // Budget against total *coverage* (sum of per-particle quad areas),
        // not the buffer size: overdraw from many huge overlapping sprites
        // is what actually costs, and it shrinks with scale² the same way.
        let coverage = system.coverage(area).max(area);
        let raster_scale = (PARTICLE_PIXEL_BUDGET / coverage).sqrt().min(1.0).max(0.2);
        let sw = ((bw as f32 * raster_scale).ceil() as u32).max(1);
        let sh = ((bh as f32 * raster_scale).ceil() as u32).max(1);

        let additive = self.particle_additive.get(idx).copied().unwrap_or(false);
        let mut buf = RgbaImage::new(sw, sh);
        let _t_raster = std::time::Instant::now();
        system.render_onto_scaled(
            &mut buf,
            self.particle_sprites[idx].as_ref(),
            [min_x, min_y],
            additive,
            raster_scale,
        );
        if std::env::var("WP_DEBUG_PARTICLE_TIMING").is_ok() {
            tracing::trace!(
                target: "timing",
                "particles[{idx}]: step {step_ms:.1}ms, raster {sw}x{sh} (scale {raster_scale:.2}) {:.1}ms",
                _t_raster.elapsed().as_secs_f32() * 1000.0
            );
        }
        let tex = self.renderer.upload_texture(&buf);
        let view = tex.create_view(&Default::default());
        let sampler = self.renderer.sampler_for(false, false);

        // Same NDC `rect` convention as image layers (`gpu_shaders.wgsl`'s
        // `vs_composite_quad`): center + half-extent, Y-up — flip our
        // pixel-space (Y-down) bbox center back into WE's Y-up scene
        // convention first, matching `spawn_center`'s own flip.
        let cx_px = min_x + bw as f32 / 2.0;
        let cy_px = min_y + bh as f32 / 2.0;
        let we_cy = self.height as f32 - cy_px;
        let rect = [
            2.0 * cx_px / self.width as f32 - 1.0,
            2.0 * we_cy / self.height as f32 - 1.0,
            bw as f32 / self.width as f32,
            bh as f32 / self.height as f32,
        ];
        // Additive particle materials (fog/smoke/embers/rain/lightning — the
        // overwhelming majority of real particle materials) need the
        // dest-read blend pipeline. Mode 30 is our pure premultiplied add:
        // the rasterizer above already accumulated `src * src_a` per
        // particle (the reference's GL_SRC_ALPHA/GL_ONE quad blending), so
        // the composite adds the buffer's RGB to the scene as-is. Plain
        // alpha-over (mode 0) otherwise makes a sprite's near-black
        // background visibly darken/box the scene instead of contributing
        // nothing, since particle compositing previously always used mode 0.
        const BLEND_MODE_PARTICLE_ADD: i32 = 100;
        let mode = if additive { BLEND_MODE_PARTICLE_ADD } else { 0 };
        let composite_buf = self.renderer.make_uniform_buffer(
            &composite_params(
                1.0,
                mode,
                [0.0, 0.0],
                [1.0, 1.0, 1.0],
                0.0,
                rect,
                self.width as f32 / self.height as f32,
                [self.width as f32, self.height as f32],
            ),
            64,
        );
        let extra: Vec<Option<wgpu::TextureView>> = if additive {
            encoder.copy_texture_to_texture(
                self.target.as_image_copy(),
                self.scene_copy.as_image_copy(),
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
            vec![Some(self.scene_copy.create_view(&Default::default()))]
        } else {
            vec![]
        };
        let base_bg = self
            .renderer
            .make_base_bind_group(&view, sampler, &composite_buf, &extra);
        self.renderer.run_pass(
            encoder,
            self.renderer.composite_pipeline(mode as u32),
            &base_bg,
            None,
            target_view,
            wgpu::LoadOp::Load,
            "particle_composite",
            6,
            &[],
        );
    }

    /// Bloom: threshold at quarter res, gaussian blur at quarter and eighth
    /// res, then additive combine back onto the scene target.
    /// Reproduces the reference's exact four-pass chain (`downsample_quarter_bloom`
    /// → `downsample_eighth_blur_v` → `blur_h_bloom` → `combine`, all real WE
    /// utility shaders the linux port wires up as a hidden fullscreen effect —
    /// see CScene.cpp/WallpaperApplication.cpp). All texel offsets use the
    /// *scene's* texel size, matching WE's always-scene-relative `g_TexelSize`.
    /// Volumetric-light-shaft ray march, generated at quarter res, blurred
    /// (reusing the bloom chain's own separable-blur pipeline unmodified —
    /// it's already a generic single-source-texture operation), then
    /// additively combined onto the scene (reusing bloom's combine pipeline
    /// too, for the same reason). See `build_volumetrics`/`fs_volumetrics`
    /// and the Ghidra report's `_rt_volumetrics*` follow-up. A no-op unless
    /// `self.volumetrics_bind_group` is `Some` (a `castvolumetrics` light
    /// exists) — checked by the one caller, `render()`.
    fn record_volumetrics(&self, encoder: &mut wgpu::CommandEncoder, target_view: &wgpu::TextureView) {
        let Some(vol_bg) = &self.volumetrics_bind_group else {
            return;
        };
        let buf = self.fbo_pool.get("_rt_VolumetricsBuffer").unwrap();
        let tmp = self.fbo_pool.get("_rt_VolumetricsBlurTmp").unwrap();
        let buf_texel = [1.0 / buf.width as f32, 1.0 / buf.height as f32];

        // Snapshot the pre-volumetrics scene: the combine pass both reads
        // and writes `self.target`, which a GPU can't do within one pass —
        // same constraint `record_bloom` works around the same way.
        encoder.copy_texture_to_texture(
            self.target.as_image_copy(),
            self.scene_copy.as_image_copy(),
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        let scene_copy_view = self.scene_copy.create_view(&Default::default());

        // Pass 1: the ray march itself — its own dedicated bind group
        // (shadow atlas + comparison sampler + this light's params), not
        // the generic base_bgl/effect_bgl shape the blur/combine passes
        // below use.
        self.renderer.run_pass(
            encoder,
            &self.renderer.volumetrics_pipeline,
            vol_bg,
            None,
            &buf.view(),
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            "volumetrics_march",
            3,
            &[],
        );

        let run = |encoder: &mut wgpu::CommandEncoder,
                   pipeline: &wgpu::RenderPipeline,
                   src: &wgpu::TextureView,
                   extra: &[Option<wgpu::TextureView>],
                   dst: &wgpu::TextureView,
                   params: &[u8]| {
            let default_sampler = self.renderer.sampler_for(false, false);
            let passthrough = self
                .renderer
                .make_uniform_buffer(&passthrough_composite_params(), 64);
            let base_bg =
                self.renderer
                    .make_base_bind_group(src, default_sampler, &passthrough, extra);
            let param_buf = self.renderer.make_uniform_buffer(params, 16);
            let effect_bg = self.renderer.make_effect_bind_group(&param_buf);
            self.renderer.run_pass(
                encoder,
                pipeline,
                &base_bg,
                Some(&effect_bg),
                dst,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                "volumetrics_blur_combine",
                3,
                &[],
            );
        };
        let f32s =
            |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|f| f.to_le_bytes()).collect() };

        // Passes 2-3: separable blur, in place within the same quarter-res
        // buffer (via the tmp buffer) — softens the fixed-step-count ray
        // march's banding and gives the shaft a less hard-edged look.
        run(
            encoder,
            &self.renderer.bloom_blur_pipeline,
            &buf.view(),
            &[],
            &tmp.view(),
            &f32s(&[buf_texel[0] * 2.0, 0.0, 0.0, 0.0]),
        );
        run(
            encoder,
            &self.renderer.bloom_blur_pipeline,
            &tmp.view(),
            &[],
            &buf.view(),
            &f32s(&[0.0, buf_texel[1] * 2.0, 0.0, 0.0]),
        );
        // Pass 4: additive combine onto the scene (combine.frag: plain
        // `scene + src`, the same math bloom's own combine pass uses).
        run(
            encoder,
            &self.renderer.bloom_combine_pipeline,
            &buf.view(),
            &[Some(scene_copy_view)],
            target_view,
            &f32s(&[0.0, 0.0, 0.0, 0.0]),
        );
    }

    fn record_bloom(&self, encoder: &mut wgpu::CommandEncoder, target_view: &wgpu::TextureView) {
        let q = self.fbo_pool.get("_rt_4FrameBuffer").unwrap();
        let e = self.fbo_pool.get("_rt_8FrameBuffer").unwrap();
        let bloom_rt = self.fbo_pool.get("_rt_Bloom").unwrap();
        let scene_view = self.target.create_view(&Default::default());
        let scene_texel = [1.0 / self.width as f32, 1.0 / self.height as f32];

        // Snapshot the pre-bloom scene: the final combine pass both reads
        // and writes `self.target`, which a GPU can't do within one pass.
        encoder.copy_texture_to_texture(
            self.target.as_image_copy(),
            self.scene_copy.as_image_copy(),
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        let scene_copy_view = self.scene_copy.create_view(&Default::default());

        let passthrough = self
            .renderer
            .make_uniform_buffer(&passthrough_composite_params(), 64);

        let run = |encoder: &mut wgpu::CommandEncoder,
                   pipeline: &wgpu::RenderPipeline,
                   src: &wgpu::TextureView,
                   extra: &[Option<wgpu::TextureView>],
                   dst: &wgpu::TextureView,
                   params: &[u8]| {
            let default_sampler = self.renderer.sampler_for(false, false);
            let base_bg =
                self.renderer
                    .make_base_bind_group(src, default_sampler, &passthrough, extra);
            let param_buf = self.renderer.make_uniform_buffer(params, 16);
            let effect_bg = self.renderer.make_effect_bind_group(&param_buf);
            self.renderer.run_pass(
                encoder,
                pipeline,
                &base_bg,
                Some(&effect_bg),
                dst,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                "bloom",
                3,
                &[],
            );
        };

        let f32s =
            |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|f| f.to_le_bytes()).collect() };

        // Pass 1: threshold + 4-tap box downsample, scene → quarter
        // (downsample_quarter_bloom.frag: texel/threshold/strength; tint fixed
        // at white since it isn't exposed as a scene setting).
        run(
            encoder,
            &self.renderer.bloom_threshold_pipeline,
            &scene_view,
            &[],
            &q.view(),
            &f32s(&[
                scene_texel[0],
                scene_texel[1],
                self.bloom.threshold,
                self.bloom.strength,
            ]),
        );
        // Pass 2: 13-tap blur along x (step = scene texel * 8), quarter → eighth.
        run(
            encoder,
            &self.renderer.bloom_blur_pipeline,
            &q.view(),
            &[],
            &e.view(),
            &f32s(&[scene_texel[0] * 8.0, 0.0, 0.0, 0.0]),
        );
        // Pass 3: 13-tap blur along y (step = scene texel * 8), eighth → bloom.
        run(
            encoder,
            &self.renderer.bloom_blur_pipeline,
            &e.view(),
            &[],
            &bloom_rt.view(),
            &f32s(&[0.0, scene_texel[1] * 8.0, 0.0, 0.0]),
        );
        // Pass 4: combine.frag — plain add of the pre-bloom scene and bloom.
        run(
            encoder,
            &self.renderer.bloom_combine_pipeline,
            &bloom_rt.view(),
            &[Some(scene_copy_view)],
            target_view,
            &f32s(&[0.0, 0.0, 0.0, 0.0]),
        );
    }

    /// Render one frame and read it back to CPU RGBA (preview / test / SHM
    /// fallback paths).
    pub fn render_rgba(&mut self) -> Result<RgbaImage> {
        self.render();
        // Temporary debug hook: dump every pooled render target to PNG.
        if std::env::var("WP_DEBUG_DUMP_FBOS").is_ok() {
            for (name, rt) in self.fbo_pool.iter() {
                if let Ok(img) = self.renderer.readback(&rt.texture, rt.width, rt.height) {
                    let safe = name.replace([':', '/'], "_");
                    let _ = img.save(format!("/tmp/fbo_{safe}.png"));
                    let mut a = img.clone();
                    for p in a.pixels_mut() {
                        p.0 = [p.0[3], p.0[3], p.0[3], 255];
                    }
                    let _ = a.save(format!("/tmp/fbo_{safe}_alpha.png"));
                }
            }
        }
        self.renderer
            .readback(&self.target, self.width, self.height)
    }

    /// Render one frame and blit it aspect-fill into an external texture view
    /// (e.g. an acquired Wayland surface frame). No CPU readback happens.
    pub fn render_to_view(
        &mut self,
        view: &wgpu::TextureView,
        view_w: u32,
        view_h: u32,
        format: wgpu::TextureFormat,
    ) {
        self.render();

        // Aspect-fill (cover): sample a centered sub-rect of the scene.
        let scene_aspect = self.width as f32 / self.height as f32;
        let view_aspect = view_w.max(1) as f32 / view_h.max(1) as f32;
        let (uv_scale, uv_offset) = if view_aspect >= scene_aspect {
            let sy = scene_aspect / view_aspect;
            ([1.0f32, sy], [0.0f32, (1.0 - sy) * 0.5])
        } else {
            let sx = view_aspect / scene_aspect;
            ([sx, 1.0f32], [(1.0 - sx) * 0.5, 0.0f32])
        };
        let mut params = [0u8; 16];
        params[0..4].copy_from_slice(&uv_scale[0].to_le_bytes());
        params[4..8].copy_from_slice(&uv_scale[1].to_le_bytes());
        params[8..12].copy_from_slice(&uv_offset[0].to_le_bytes());
        params[12..16].copy_from_slice(&uv_offset[1].to_le_bytes());

        let blit_buf = self.renderer.make_uniform_buffer(&params, 16);
        let scene_view = self.target.create_view(&Default::default());
        let default_sampler = self.renderer.sampler_for(false, false);
        let base_bg =
            self.renderer
                .make_base_bind_group(&scene_view, default_sampler, &blit_buf, &[]);

        // Ensure the pipeline exists before borrowing immutably for the pass.
        self.renderer.blit_pipeline(format);
        let pipeline = &self.renderer.blit_pipelines[&format];

        let mut encoder =
            self.renderer
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("present_blit"),
                });
        self.renderer.run_pass(
            &mut encoder,
            pipeline,
            &base_bg,
            None,
            view,
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            "present_blit",
            3,
            &[],
        );
        self.renderer
            .queue
            .submit(std::iter::once(encoder.finish()));
    }
}

fn named_fbo_key(instance_idx: usize, fbo_name: &str) -> String {
    format!("fx{instance_idx}:{fbo_name}")
}

/// Fraction of the local day in [0,1) (the reference's g_Daytime).
fn daytime_fraction() -> f32 {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    ((secs % 86_400) as f32) / 86_400.0
}

#[allow(clippy::too_many_arguments)]
fn composite_params(
    opacity: f32,
    mode: i32,
    uv_offset: [f32; 2],
    color: [f32; 3],
    angle: f32,
    rect: [f32; 4],
    aspect: f32,
    resolution: [f32; 2],
) -> [u8; 64] {
    // WGSL CompositeParams layout: opacity@0, mode@4, uv_offset@8, color@16,
    // angle@28, rect@32, aspect@48, resolution@56 (bytes 52-55 are padding).
    let mut data = [0u8; 64];
    data[0..4].copy_from_slice(&opacity.to_le_bytes());
    data[4..8].copy_from_slice(&mode.to_le_bytes());
    data[8..12].copy_from_slice(&uv_offset[0].to_le_bytes());
    data[12..16].copy_from_slice(&uv_offset[1].to_le_bytes());
    data[16..20].copy_from_slice(&color[0].to_le_bytes());
    data[20..24].copy_from_slice(&color[1].to_le_bytes());
    data[24..28].copy_from_slice(&color[2].to_le_bytes());
    data[28..32].copy_from_slice(&angle.to_le_bytes());
    for (i, v) in rect.iter().enumerate() {
        data[32 + i * 4..36 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    data[48..52].copy_from_slice(&aspect.to_le_bytes());
    data[56..60].copy_from_slice(&resolution[0].to_le_bytes());
    data[60..64].copy_from_slice(&resolution[1].to_le_bytes());
    data
}

/// Fullscreen rect: centered, full extents.
const FULLSCREEN_RECT: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// Always mode 0 (`fs_composite`, never reads `dest_copy_tex`), so the
/// resolution field is unused — a dummy value is harmless here.
fn passthrough_composite_params() -> [u8; 64] {
    composite_params(
        1.0,
        0,
        [0.0, 0.0],
        [1.0, 1.0, 1.0],
        0.0,
        FULLSCREEN_RECT,
        1.0,
        [1.0, 1.0],
    )
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
                            crate::engine::tex::TexFile::parse(&bytes)
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
                        crate::engine::tex::TexFile::parse(&bytes)
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
            crate::engine::shaders::uniform_meta::parse_uniform_metadata(&frag_glsl)
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
                crate::engine::tex::TexFile::parse(&bytes)
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

type ShaderVals = std::collections::HashMap<String, f32>;

/// Insert a constant under every spelling the packer might look up: the raw
/// key, its normalized alias (scene constants use
/// "ui_editor_properties_color_start" while asset shaders annotate materials
/// as "colorstart"), and per-component suffixes for vector values.
fn insert_value_components(map: &mut ShaderVals, key: &str, value: &serde_json::Value) {
    let alias = |k: &str| -> String {
        k.strip_prefix("ui_editor_properties_")
            .unwrap_or(k)
            .replace('_', "")
            .to_lowercase()
    };
    let animated = crate::engine::model::json_to_animated(value);
    insert_animated_components(map, key, &alias(key), &animated);
}

fn insert_animated_components(
    map: &mut ShaderVals,
    key: &str,
    alias: &str,
    v: &crate::engine::model::AnimatedValue,
) {
    if let Some(f) = v.as_float() {
        map.insert(key.to_string(), f);
        map.insert(alias.to_string(), f);
    }
    if let Some(comps) = v.components() {
        // Both spellings: colors are read as _r/_g/_b/_a, geometry as
        // _x/_y/_z/_w. A vec3 keeps its historical implicit _w/_a of 1.0.
        let mut named: Vec<(&str, f32)> = Vec::new();
        for (i, val) in comps.iter().enumerate() {
            named.push((["r", "g", "b", "a"][i.min(3)], *val));
            named.push((["x", "y", "z", "w"][i.min(3)], *val));
        }
        if comps.len() == 3 {
            named.push(("a", 1.0));
            named.push(("w", 1.0));
        }
        for (name, val) in named {
            map.insert(format!("{key}_{name}"), val);
            map.insert(format!("{alias}_{name}"), val);
        }
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
        for eff in &obj.effects {
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
                name,
                file: file.clone(),
                pass_overrides,
            });
        }
    }
    result
}

/// Pack effect uniform values using std140 layout rules — the transpiled
/// UBO (GLSL uniform block → SPIR-V → WGSL) uses std140 offsets, so each
/// member is aligned to its own alignment, NOT padded to 16 bytes:
/// float/int align 4, vec2 align 8, vec3/vec4/matrix columns align 16.
///
/// Value resolution per uniform (mirrors CPass::setupUniforms +
/// setupShaderVariables): engine-provided uniforms first, then material/scene
/// constants, then the shader annotation's default.
fn make_params_from_translated_typed(
    keys: &[transpiler::UniformEntry],
    engine: &EngineUniforms,
    vals: &ShaderVals,
) -> Vec<u8> {
    let mut bytes: Vec<u8> = Vec::new();

    // Resolve one uniform into up to 4 components. A scene may key the value
    // by the annotation's material key ("alpha") or by its UI label
    // ("ui_editor_properties_outline_background") — and those two are not
    // always spellings of each other, so both are tried.
    let resolve = |entry: &transpiler::UniformEntry, default: &[f32; 4]| -> [f32; 4] {
        let key = entry.key.as_str();
        if let Some(v) = engine.get(key) {
            return v;
        }
        let names = || std::iter::once(key).chain(entry.label.as_deref());
        let mut out = *default;
        if let Some(f) = names().find_map(|k| vals.get(k)) {
            out = [*f; 4];
        }
        // Per-component spellings win over the scalar broadcast.
        for (i, suffixes) in [
            &["_x", "_r"][..],
            &["_y", "_g"][..],
            &["_z", "_b"][..],
            &["_w", "_a"][..],
        ]
        .iter()
        .enumerate()
        {
            if let Some(f) = suffixes
                .iter()
                .flat_map(|s| names().map(move |k| format!("{k}{s}")))
                .find_map(|k| vals.get(&k))
            {
                out[i] = *f;
            }
        }
        out
    };

    fn align_to(bytes: &mut Vec<u8>, align: usize) {
        let rem = bytes.len() % align;
        if rem != 0 {
            bytes.resize(bytes.len() + (align - rem), 0);
        }
    }
    fn push_f32(bytes: &mut Vec<u8>, v: f32) {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    for entry in keys {
        match entry.glsl_type.as_str() {
            "vec2" => {
                let v = resolve(entry, &entry.default);
                align_to(&mut bytes, 8);
                push_f32(&mut bytes, v[0]);
                push_f32(&mut bytes, v[1]);
            }
            "vec3" => {
                let v = resolve(entry, &entry.default);
                align_to(&mut bytes, 16);
                push_f32(&mut bytes, v[0]);
                push_f32(&mut bytes, v[1]);
                push_f32(&mut bytes, v[2]);
                // std140 vec3 has size 12; a following float packs into the
                // tail 4 bytes, so do NOT pad here.
            }
            "vec4" => {
                let v = resolve(entry, &entry.default);
                align_to(&mut bytes, 16);
                for f in v {
                    push_f32(&mut bytes, f);
                }
            }
            "mat4" => {
                align_to(&mut bytes, 16);
                let identity: [f32; 16] = [
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ];
                for f in identity {
                    push_f32(&mut bytes, f);
                }
            }
            "mat3" => {
                // std140 mat3: 3 columns × (vec3 + 4 bytes pad) = 48 bytes
                align_to(&mut bytes, 16);
                let cols: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
                for col in cols {
                    for f in col {
                        push_f32(&mut bytes, f);
                    }
                    bytes.extend_from_slice(&[0u8; 4]);
                }
            }
            _ => {
                // float / int / bool / unknown: 4 bytes, align 4
                let v = resolve(entry, &entry.default);
                align_to(&mut bytes, 4);
                push_f32(&mut bytes, v[0]);
            }
        }
    }

    // Uniform buffers must be non-empty and sized to a 16-byte multiple.
    if bytes.is_empty() {
        bytes.resize(16, 0);
    }
    align_to(&mut bytes, 16);
    bytes
}

fn make_effect_params(name: &str, time: f32, vals: &ShaderVals) -> Vec<u8> {
    // Scene `constantshadervalues` are keyed by the shader annotation's
    // *material key* ("speed", "scale", ...), not its ui_editor_properties_*
    // label — look the bare key up first and keep the label as a fallback
    // for any older call sites below that still pass the long form.
    fn get(vals: &ShaderVals, key: &str, default: f32) -> f32 {
        if let Some(v) = vals.get(key) {
            return *v;
        }
        // Scenes key a value by the annotation's material key ("amount") or by
        // its UI label ("ui_editor_properties_pulse_amount", normalized to
        // "pulseamount") — and the label is not always the material key with a
        // prefix chopped off. LABEL_ALIASES covers the pairs the corpus
        // actually uses; the translated-effect packer gets this from the
        // shader annotation itself (see make_params_from_translated_typed).
        const LABEL_ALIASES: &[(&str, &[&str])] = &[
            ("amount", &["pulseamount"]),
            ("speed", &["pulsespeed"]),
            ("phase", &["pulsephase", "timeoffset"]),
            ("bounds", &["pulsebounds"]),
            ("scale", &["ripplescale"]),
        ];
        // Vector params arrive here already split ("bounds_r"), so match on the
        // stem and carry the suffix over.
        let (stem, suffix) = match key.rfind('_') {
            Some(i) if key.len() - i <= 2 => (&key[..i], &key[i..]),
            _ => (key, ""),
        };
        if let Some(v) = LABEL_ALIASES
            .iter()
            .find(|(k, _)| *k == stem)
            .into_iter()
            .flat_map(|(_, aliases)| aliases.iter())
            .find_map(|a| vals.get(&format!("{a}{suffix}")))
        {
            return *v;
        }
        key.strip_prefix("ui_editor_properties_")
            .and_then(|bare| vals.get(bare).copied())
            .unwrap_or(default)
    }
    fn pack(floats: &[f32]) -> Vec<u8> {
        floats.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    // Keys below are the shader annotations' material keys (the names
    // scene.json constantshadervalues actually use); defaults match each
    // annotation's declared default.
    match name {
        "pulse" => pack(&[
            time,
            get(vals, "speed", 3.0),
            get(vals, "amount", 1.0),
            get(vals, "power", 1.0),
            get(vals, "phase", 0.0),
            get(vals, "bounds_r", 0.0),
            get(vals, "bounds_g", 1.0),
            get(vals, "combo_BLENDMODE", 9.0),
            get(vals, "tintlow_r", 1.0),
            get(vals, "tintlow_g", 1.0),
            get(vals, "tintlow_b", 1.0),
            get(vals, "combo_PULSECOLOR", 1.0),
            get(vals, "tinthigh_r", 1.0),
            get(vals, "tinthigh_g", 1.0),
            get(vals, "tinthigh_b", 1.0),
            get(vals, "combo_PULSEALPHA", 0.0),
            get(vals, "noisespeed", 0.5),
            get(vals, "noiseamount", 0.0),
            get(vals, "combo_AUDIOPROCESSING", 0.0),
            get(vals, "frequencymin", 0.0),
            get(vals, "frequencymax", 1.0),
            get(vals, "audiobounds_x", 0.5),
            get(vals, "audiobounds_y", 1.0),
            get(vals, "audioexponent", 1.0),
            get(vals, "audioamount", 1.0),
            // PulseParams rounds up to 112 bytes (vec3 members force a
            // 16-byte struct alignment) — pad or wgpu rejects the bind group.
            0.0,
            0.0,
            0.0,
        ]),
        "scroll" => pack(&[
            time,
            get(vals, "speedx", 0.2),
            get(vals, "speedy", 0.2),
            get(vals, "repeat_r", 1.0),
            get(vals, "repeat_g", 1.0),
            0.0,
            0.0,
            0.0,
        ]),
        "shake" => {
            // shake.vert precomputes v_Bounds = (x, 1/(y-x)) from `bounds`.
            let bounds_x = get(vals, "bounds_x", 0.0);
            let bounds_y = get(vals, "bounds_y", 1.0);
            pack(&[
                time,
                get(vals, "speed", 1.0),
                get(vals, "strength", 0.1),
                get(vals, "combo_DIRECTION", 0.0),
                get(vals, "friction_x", 1.0),
                get(vals, "friction_y", 1.0),
                bounds_x,
                1.0 / (bounds_y - bounds_x).max(0.0001),
                get(vals, "combo_NOISE", 0.0),
                0.0,
                0.0,
                0.0,
            ])
        }
        "tint" => {
            let r = get(vals, "color_r", 1.0);
            let g = get(vals, "color_g", 0.0);
            let b = get(vals, "color_b", 0.0);
            let a = get(vals, "alpha", 1.0);
            pack(&[
                r,
                g,
                b,
                a,
                get(vals, "combo_BLENDMODE", 30.0),
                0.0,
                0.0,
                0.0,
            ])
        }
        "opacity" => pack(&[get(vals, "alpha", 1.0), 0.0, 0.0, 0.0]),
        "waterripple" => pack(&[
            time,
            get(vals, "ripplestrength", 0.1),
            get(vals, "animationspeed", 0.15),
            get(vals, "scale", 1.0),
        ]),
        "waterwaves" => {
            // Direction is the reference's rotateVec2((0, 1), angle) —
            // base vector (0, 1), not (1, 0).
            let dir = get(vals, "direction", 0.0);
            pack(&[
                time,
                get(vals, "speed", 5.0),
                get(vals, "scale", 200.0),
                get(vals, "strength", 0.1),
                -dir.sin(),
                dir.cos(),
                get(vals, "exponent", 1.0),
                0.0,
            ])
        }
        "spin" => pack(&[
            time,
            get(vals, "speed", 1.0),
            get(vals, "center_r", 0.5),
            get(vals, "center_g", 0.5),
            get(vals, "size", 0.1),
            get(vals, "feather", 0.002),
            get(vals, "combo_REPEAT", 1.0),
            0.0,
        ]),
        _ => vec![0u8; 32],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `gpu_shaders.wgsl` must parse and validate as WGSL on its own — the
    /// only check in this crate that catches a shader syntax/type error
    /// before it surfaces at runtime as an opaque wgpu pipeline-creation
    /// failure the first time a scene that uses that pass actually loads.
    /// Uses naga directly (no GPU adapter needed — pure frontend/validator).
    #[test]
    fn gpu_shaders_wgsl_parses_and_validates() {
        let module =
            naga::front::wgsl::parse_str(SHADER_SRC).expect("gpu_shaders.wgsl failed to parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("gpu_shaders.wgsl failed validation");
    }

    fn quad3d_test_camera() -> crate::engine::camera3d::PerspectiveCamera {
        let scene: crate::engine::scene::Scene = serde_json::from_str(
            r#"{"camera": {"eye": "0 0 10", "center": "0 0 0", "up": "0 1 0"},
                "general": {"orthogonalprojection": null, "fov": 60.0}}"#,
        )
        .unwrap();
        crate::engine::camera3d::PerspectiveCamera::from_scene(&scene, 1.0).unwrap()
    }

    /// A camera-facing quad (no rotation, centered on the view axis) must
    /// project to a symmetric rectangle: opposite corners mirror each other
    /// around the NDC origin. This is the case `project_quad_ndc`'s AABB
    /// collapse already gets right — `project_quad_corners` must agree with
    /// it exactly here, only diverging once the quad is off-axis/rotated.
    #[test]
    fn project_quad_corners_symmetric_for_camera_facing_quad() {
        let cam = quad3d_test_camera();
        let corners = project_quad_corners(&cam, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.0, 2.0])
            .expect("camera-facing quad in front of the eye must project");
        // Order: bottom-left, bottom-right, top-right, top-left.
        assert!((corners[0][0] + corners[1][0]).abs() < 1e-5, "{corners:?}");
        assert!((corners[0][1] + corners[3][1]).abs() < 1e-5, "{corners:?}");
        assert!(corners[0][0] < corners[1][0], "left must be left of right");
        assert!(corners[0][1] < corners[2][1], "bottom must be below top");
    }

    /// The whole point of this function: a quad rotated hard off-axis
    /// projects to a genuine trapezoid, not a rectangle. Rotating around Y
    /// swings one vertical edge toward the camera and the other away from
    /// it (left edge: x=-2 → z=+1.73, closer to the eye at z=10; right
    /// edge: x=+2 → z=-1.73, farther away) — the closer edge must span a
    /// taller NDC range than the farther one. `project_quad_ndc`'s AABB
    /// would hide this entirely (both edges would just extend its bounding
    /// box, indistinguishable from a same-size rectangle).
    #[test]
    fn project_quad_corners_is_a_true_trapezoid_when_rotated() {
        let cam = quad3d_test_camera();
        let corners = project_quad_corners(
            &cam,
            [0.0, 0.0, 0.0],
            [0.0, 60f32.to_radians(), 0.0],
            [4.0, 2.0],
        )
        .expect("quad must still project in front of the eye");
        // Left edge: corner0 (bottom-left) to corner3 (top-left).
        let left_height = (corners[3][1] - corners[0][1]).abs();
        // Right edge: corner1 (bottom-right) to corner2 (top-right).
        let right_height = (corners[2][1] - corners[1][1]).abs();
        assert!(
            left_height > right_height + 1e-4,
            "expected the closer (left) edge to project taller than the \
             farther (right) edge: left={left_height} right={right_height}"
        );
    }

    /// A corner behind the eye plane must cull the whole quad — same rule
    /// `project_quad_ndc` uses, kept consistent rather than drawing a
    /// partially-degenerate quad.
    #[test]
    fn project_quad_corners_none_when_behind_camera() {
        let cam = quad3d_test_camera();
        // Eye is at z=10 looking toward the origin; z=20 is behind it.
        assert!(project_quad_corners(&cam, [0.0, 0.0, 20.0], [0.0, 0.0, 0.0], [4.0, 2.0]).is_none());
    }

    /// Same rule as `hardcoded_effect_params_are_uniform_sized` below, for the
    /// mesh3d lighting UBO: the Rust-side byte length must match WGSL's
    /// std140 struct size (a 16-byte multiple) and `mesh3d_lighting_bytes`
    /// must actually produce that many bytes, or wgpu rejects the mesh3d bind
    /// group at draw time.
    #[test]
    fn mesh3d_lighting_bytes_are_uniform_sized_and_match_const() {
        assert_eq!(MESH3D_LIGHTING_BYTES_LEN % 16, 0);
        let scene: crate::engine::scene::Scene = serde_json::from_str(
            r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"}, "general": {"orthogonalprojection": null}}"#,
        )
        .unwrap();
        let cam = crate::engine::camera3d::PerspectiveCamera::from_scene(&scene, 1.0).unwrap();

        // The common case: no shadow-casting lights at all.
        let bytes = mesh3d_lighting_bytes(&scene, &cam, &[], &[], None);
        assert_eq!(bytes.len(), MESH3D_LIGHTING_BYTES_LEN);

        // One shadow-casting light occupying slot 0 — same length either way,
        // since the shadow arrays are fixed-size regardless of how many
        // slots are actually in use.
        let shadow_slots = vec![Some(0)];
        let shadow_slot_data = vec![(
            crate::engine::camera3d::identity(),
            [0.0, 0.0, 1.0, 1.0],
        )];
        let bytes = mesh3d_lighting_bytes(&scene, &cam, &shadow_slots, &shadow_slot_data, None);
        assert_eq!(bytes.len(), MESH3D_LIGHTING_BYTES_LEN);
    }

    /// `cookie: Some((light_index, view_proj))` must write `flags.y ==
    /// light_index + 1` and pack `view_proj` into the trailing
    /// `cookie_view_proj` field (the last `MESH3D_MAX_SHADOW_LIGHTS` shadow
    /// mat4s + the `MESH3D_MAX_SHADOW_LIGHTS` uv-rect vec4s precede it —
    /// see `MESH3D_LIGHTING_BYTES_LEN`'s own layout doc).
    #[test]
    fn mesh3d_lighting_bytes_encodes_cookie_flag_and_matrix() {
        let scene: crate::engine::scene::Scene = serde_json::from_str(
            r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"}, "general": {"orthogonalprojection": null}}"#,
        )
        .unwrap();
        let cam = crate::engine::camera3d::PerspectiveCamera::from_scene(&scene, 1.0).unwrap();

        let mut vp = crate::engine::camera3d::identity();
        vp[3][0] = 42.0; // an arbitrary, recognizable value to round-trip
        let bytes = mesh3d_lighting_bytes(&scene, &cam, &[], &[], Some((2, vp)));

        let flags_y = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(flags_y, 3.0, "flags.y must be light_index + 1");

        let cookie_mat_offset = MESH3D_LIGHTING_BYTES_LEN - 64;
        let read_f32 = |i: usize| {
            f32::from_le_bytes(
                bytes[cookie_mat_offset + i * 4..cookie_mat_offset + i * 4 + 4]
                    .try_into()
                    .unwrap(),
            )
        };
        // Column 3, row 0 (the value we perturbed) is float index 12.
        assert_eq!(read_f32(12), 42.0, "cookie_view_proj must round-trip the matrix");
    }

    /// A spot light must (a) count as renderable (`flags.x == 1.0` — the
    /// bug fixed alongside GPU spot support: `Scene::lights()` producing a
    /// `Light::Spot` must not silently fall back to the unlit branch) and
    /// (b) write a non-zero direction into its `light_spot` slot, so
    /// `mesh3d_spot_factor` in the shader doesn't read it as "not a spot".
    #[test]
    fn mesh3d_lighting_bytes_encodes_spot_direction() {
        let cam_scene: crate::engine::scene::Scene = serde_json::from_str(
            r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"}, "general": {"orthogonalprojection": null}}"#,
        )
        .unwrap();
        let cam = crate::engine::camera3d::PerspectiveCamera::from_scene(&cam_scene, 1.0).unwrap();

        let scene: crate::engine::scene::Scene = serde_json::from_str(
            r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"},
                "general": {"orthogonalprojection": null},
                "objects": [
                    {"id": 1, "light": true, "origin": "0 0 0", "radius": 100,
                     "intensity": 0.8, "color": "1 1 1",
                     "innercone": 20, "outercone": 45}
                ]}"#,
        )
        .unwrap();
        let bytes = mesh3d_lighting_bytes(&scene, &cam, &[], &[], None);

        // flags.x is the buffer's first f32.
        let flags_x = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(flags_x, 1.0, "a spot-only scene must still engage the lit branch");

        // light_spot[0] starts after 5 leading vec4s + 2 MESH3D_MAX_LIGHTS
        // vec4 arrays (positions, colors).
        let spot0_offset = 16 * 5 + 16 * MESH3D_MAX_LIGHTS * 2;
        let read_f32 = |i: usize| {
            f32::from_le_bytes(
                bytes[spot0_offset + i * 4..spot0_offset + i * 4 + 4]
                    .try_into()
                    .unwrap(),
            )
        };
        let dir = [read_f32(0), read_f32(1), read_f32(2)];
        let len2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        assert!(len2 > 0.25, "spot direction must not read as zero: {dir:?}");
        assert!(read_f32(3) >= 1.0, "exponent must be clamped to at least 1.0");
    }

    /// A directional light must flag itself via `light_pos[0].w == -1.0`
    /// (the sentinel `fs_mesh3d`'s light loop branches on) with a unit-length
    /// view-space direction in xyz, and still engage the lit branch.
    #[test]
    fn mesh3d_lighting_bytes_encodes_directional_sentinel_and_direction() {
        let cam_scene: crate::engine::scene::Scene = serde_json::from_str(
            r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"}, "general": {"orthogonalprojection": null}}"#,
        )
        .unwrap();
        let cam = crate::engine::camera3d::PerspectiveCamera::from_scene(&cam_scene, 1.0).unwrap();

        let scene: crate::engine::scene::Scene = serde_json::from_str(
            r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"},
                "general": {"orthogonalprojection": null},
                "objects": [
                    {"id": 1, "light": "ldirectional", "angles": "0 1.5707963 0",
                     "intensity": 0.6, "color": "1 1 1"}
                ]}"#,
        )
        .unwrap();
        let bytes = mesh3d_lighting_bytes(&scene, &cam, &[], &[], None);

        let flags_x = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(flags_x, 1.0, "a directional-only scene must still engage the lit branch");

        // light_pos[0] starts right after the 5 leading vec4s.
        let pos0_offset = 16 * 5;
        let read_f32 = |i: usize| {
            f32::from_le_bytes(
                bytes[pos0_offset + i * 4..pos0_offset + i * 4 + 4]
                    .try_into()
                    .unwrap(),
            )
        };
        let dir = [read_f32(0), read_f32(1), read_f32(2)];
        let len2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        assert!((len2 - 1.0).abs() < 1e-4, "direction must be unit-length: {dir:?}");
        assert_eq!(read_f32(3), -1.0, "w must carry the directional sentinel");
    }

    /// A tube light must flag itself via `light_pos[0].w == -3.0` (the
    /// sentinel `fs_mesh3d`'s light loop branches on), with endpoint A in
    /// `light_pos[0].xyz` and endpoint B in `light_spot[0].xyz` — not a
    /// direction, so its magnitude should reflect the real endpoint spacing,
    /// not be unit-length like a spot's.
    #[test]
    fn mesh3d_lighting_bytes_encodes_tube_sentinel_and_endpoints() {
        let cam_scene: crate::engine::scene::Scene = serde_json::from_str(
            r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"}, "general": {"orthogonalprojection": null}}"#,
        )
        .unwrap();
        let cam = crate::engine::camera3d::PerspectiveCamera::from_scene(&cam_scene, 1.0).unwrap();

        let scene: crate::engine::scene::Scene = serde_json::from_str(
            r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"},
                "general": {"orthogonalprojection": null},
                "objects": [
                    {"id": 1, "light": "ltube", "origin": "0 0 0",
                     "controlpoint": "0 0 -4", "intensity": 0.9, "color": "1 1 1"}
                ]}"#,
        )
        .unwrap();
        let bytes = mesh3d_lighting_bytes(&scene, &cam, &[], &[], None);

        let flags_x = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(flags_x, 1.0, "a tube-only scene must still engage the lit branch");

        let pos0_offset = 16 * 5;
        let spot0_offset = 16 * 5 + 16 * MESH3D_MAX_LIGHTS * 2;
        let read_f32 = |offset: usize, i: usize| {
            f32::from_le_bytes(bytes[offset + i * 4..offset + i * 4 + 4].try_into().unwrap())
        };
        assert_eq!(read_f32(pos0_offset, 3), -3.0, "w must carry the tube sentinel");
        let endpoint_b = [read_f32(spot0_offset, 0), read_f32(spot0_offset, 1), read_f32(spot0_offset, 2)];
        let len2 = endpoint_b[0] * endpoint_b[0]
            + endpoint_b[1] * endpoint_b[1]
            + endpoint_b[2] * endpoint_b[2];
        assert!(len2 > 1.0, "endpoint B should be a real view-space position, not a unit vector: {endpoint_b:?}");
    }

    /// WGSL rounds every uniform-address-space struct up to a 16-byte
    /// multiple, so a param buffer that isn't one can never match its shader
    /// struct — wgpu rejects the bind group at draw time ("Buffer is bound
    /// with size N where the shader expects M") and the wallpaper dies.
    #[test]
    fn hardcoded_effect_params_are_uniform_sized() {
        let vals = ShaderVals::new();
        for name in HARDCODED_EFFECTS {
            let n = make_effect_params(name, 0.0, &vals).len();
            assert_eq!(n % 16, 0, "{name} params are {n} bytes, not a 16-multiple");
        }
    }

    /// Hardcoded kernels must find a value the scene keyed by UI label
    /// ("ui_editor_properties_pulse_amount") as well as by material key.
    #[test]
    fn hardcoded_params_resolve_label_keyed_values() {
        let mut vals = ShaderVals::new();
        insert_value_components(
            &mut vals,
            "ui_editor_properties_pulse_amount",
            &serde_json::json!(0.25),
        );
        insert_value_components(
            &mut vals,
            "ui_editor_properties_pulse_bounds",
            &serde_json::json!("0.2 0.8"),
        );
        let bytes = make_effect_params("pulse", 0.0, &vals);
        let f = |i: usize| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(f(2), 0.25, "amount");
        assert_eq!(f(5), 0.2, "bounds low");
        assert_eq!(f(6), 0.8, "bounds high");
    }

    /// vec2 constants (audiobounds, texture scales, ...) must expose their
    /// components like vec3 colors do, or the packer silently falls back to
    /// the shader default and the artist's tuning is lost.
    #[test]
    fn vector_constants_expose_components_at_every_width() {
        let mut map = ShaderVals::new();
        insert_value_components(&mut map, "audiobounds", &serde_json::json!("0.25 0.75"));
        assert_eq!(map.get("audiobounds_x"), Some(&0.25));
        assert_eq!(map.get("audiobounds_y"), Some(&0.75));
        assert_eq!(map.get("audiobounds_r"), Some(&0.25));

        insert_value_components(&mut map, "tintlow", &serde_json::json!("1 0 0"));
        assert_eq!(map.get("tintlow_b"), Some(&0.0));
        // vec3s keep their implicit opaque alpha.
        assert_eq!(map.get("tintlow_a"), Some(&1.0));
    }
}
