//! Device-level GPU state shared by every layer of a scene: samplers, bind
//! group layouts, the pipeline cache, and the draw helpers that record
//! passes with them.

use super::*;

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
            size: crate::services::audio::UNIFORM_BYTES as u64,
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
                // light's tile packed into it — see `engine::gpu::shadow`) plus
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
        // `engine::gpu::shadow` and `GpuSceneInstance::build_shadow_atlas`). No
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
        let mesh3d_shadow_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

    pub(super) fn create_white_1x1_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> wgpu::Texture {
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
    pub(super) fn create_pipeline(
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
    pub(super) fn create_pipeline_split_modules(
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
    pub(super) fn attr_buffer(&mut self, kind: u8, dims: u8) -> &wgpu::Buffer {
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
    pub(super) fn attr_buffers_for(
        &mut self,
        attributes: &[(String, String)],
    ) -> Vec<wgpu::Buffer> {
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
            crate::engine::gpu::fbo::fit_texture_limit(&self.device, img.width(), img.height());
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

    pub(super) fn find_effect_pipeline(&self, key: &str) -> Option<&wgpu::RenderPipeline> {
        self.effect_pipelines
            .iter()
            .find(|(n, _)| *n == key)
            .map(|(_, p)| p)
            .or_else(|| self.dynamic_pipelines.get(key))
    }

    /// Pipeline key: 0 = normal alpha blend, 1 = Photoshop-style dest-read
    /// blend (any nonzero WE colorBlendMode; the mode itself travels in
    /// CompositeParams.mode).
    pub(super) fn composite_pipeline(&self, blend_mode: u32) -> &wgpu::RenderPipeline {
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
    pub(super) fn sampler_for(&self, no_interpolation: bool, clamp_uvs: bool) -> &wgpu::Sampler {
        let idx = ((no_interpolation as usize) << 1) | (clamp_uvs as usize);
        &self.samplers[idx]
    }

    /// Create the group(0) bind group: source texture, sampler, a uniform
    /// buffer, and up to six extra texture slots (dummy white when absent).
    pub(super) fn make_base_bind_group(
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

    pub(super) fn make_uniform_buffer(&self, data: &[u8], min_size: u64) -> wgpu::Buffer {
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

    pub(super) fn make_effect_bind_group(&self, param_buf: &wgpu::Buffer) -> wgpu::BindGroup {
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
    pub(super) fn run_pass(
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

pub(super) fn vertex_format_for(glsl_type: &str) -> wgpu::VertexFormat {
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
pub(super) fn wgpu_blend_state(blending: &WEBlending) -> Option<wgpu::BlendState> {
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
