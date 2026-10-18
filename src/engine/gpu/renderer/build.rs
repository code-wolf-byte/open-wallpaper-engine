//! Building a running scene: resolving scene.json into GPU layers, 3D
//! meshes and sounds, loading each effect's passes into runtimes, and
//! uploading the initial geometry.

use super::*;

impl GpuSceneInstance {
    pub(super) fn build(mut renderer: GpuSceneRenderer, dir: &Path) -> Result<Self> {
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
    pub(super) fn upload_mesh3d_geometry(
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
    pub(super) fn build_mesh3d(
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
}

/// Start a `sound` object's first decodable file, held silent when
/// `startsilent` is set.
pub(super) fn start_scene_sound(
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

/// Load pipelines/textures for every effect instance in the scene: hardcoded
/// kernels get single-pass runtimes, everything else is loaded from the WE
/// assets directory pass-by-pass and translated GLSL→WGSL with the instance's
/// combos baked in.
pub(super) fn load_effect_runtimes(
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
pub(super) fn load_effect_instance(
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

pub(super) fn collect_effects(scene: &crate::engine::model::SceneModel) -> Vec<EffectInstanceDef> {
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
pub(super) fn gpu_layer_from(
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
