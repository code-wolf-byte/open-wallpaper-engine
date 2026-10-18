//! Scene lights on the GPU: the shadow-map atlas for shadow-casting lights,
//! volumetric light shafts (built once, recorded each frame), and the
//! per-mesh transform and lighting uniform bytes.

use super::*;

impl GpuSceneInstance {
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
    pub(super) fn build_volumetrics(
        renderer: &GpuSceneRenderer,
        cam: &crate::engine::sim::camera3d::PerspectiveCamera,
        lights: &[(
            crate::engine::gpu::lighting::Light,
            bool,
            Option<(f32, f32)>,
            Option<String>,
        )],
        shadow_slots: &[Option<usize>],
        shadow_slot_data: &[(crate::engine::sim::camera3d::Mat4, [f32; 4])],
        shadow_atlas_view: &wgpu::TextureView,
    ) -> Option<wgpu::BindGroup> {
        use crate::engine::gpu::lighting::Light;

        let (idx, origin, color_intensity, density, exponent) = lights
            .iter()
            .enumerate()
            .find_map(|(i, (light, _, volumetrics, _))| {
                let (density, exponent) = (*volumetrics)?;
                let (origin, color, intensity) = match light {
                    // Directional has no position to march a shaft from;
                    // Tube isn't constructed by `Scene::lights()` yet (see
                    // `LightType::Tube`'s doc comment) — neither reaches here.
                    Light::Point {
                        origin,
                        color,
                        intensity,
                        ..
                    }
                    | Light::Spot {
                        origin,
                        color,
                        intensity,
                        ..
                    } => (*origin, *color, *intensity),
                    _ => return None,
                };
                let ci = [
                    color[0] * intensity,
                    color[1] * intensity,
                    color[2] * intensity,
                ];
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
            .unwrap_or((crate::engine::sim::camera3d::identity(), [0.0; 4], 0.0));

        let inv_view_proj = crate::engine::sim::camera3d::mat4_inverse(&cam.view_proj_raw());
        let eye = cam.eye();
        // Fixed, honest default range for the ray march — no real content
        // to derive a better heuristic from (see the fn doc comment).
        // Comparable to `radius`'s own 256px-ish default scene scale.
        const MAX_RANGE: f32 = 40.0;

        let mut bytes = Vec::with_capacity(4 * 16 * 4 + 16 * 5);
        let mut push_mat = |m: &crate::engine::sim::camera3d::Mat4| {
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
        push4([
            color_intensity[0],
            color_intensity[1],
            color_intensity[2],
            exponent,
        ]);
        push4(shadow_uv_rect);
        push4([has_shadow, MAX_RANGE, 0.0, 0.0]);

        let ubo = renderer.make_uniform_buffer(&bytes, bytes.len() as u64);
        Some(
            renderer
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
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
                            resource: wgpu::BindingResource::Sampler(
                                &renderer.shadow_comparison_sampler,
                            ),
                        },
                    ],
                }),
        )
    }

    /// Render the shadow atlas once at scene setup (shadow-casting lights
    /// are treated as static, same precedent as mesh MVPs — see
    /// `build_mesh3d`'s own doc comment). For every `light` object with
    /// `castshadow: true` (capped at `engine::gpu::shadow::MAX_SHADOW_LIGHTS`),
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
    pub(super) fn build_shadow_atlas(
        renderer: &GpuSceneRenderer,
        mesh3d_layers: &[crate::engine::model::resolve::Mesh3dLayer],
        lights: &[(
            crate::engine::gpu::lighting::Light,
            bool,
            Option<(f32, f32)>,
            Option<String>,
        )],
    ) -> (
        wgpu::Texture,
        Vec<Option<usize>>,
        Vec<(crate::engine::sim::camera3d::Mat4, [f32; 4])>,
    ) {
        use crate::engine::gpu::lighting::Light;
        use crate::engine::gpu::shadow;

        let mut shadow_slots: Vec<Option<usize>> = vec![None; lights.len()];

        // Needed to build either projection (`point_light_view_proj`'s own
        // aim-and-size target, `spot_light_view_proj`'s near/far sizing) —
        // computed once up front now that both light types consume it,
        // rather than after filtering casters like the point-only version
        // did.
        let (center, radius) = shadow::scene_bounds(mesh3d_layers);
        let shadow_casters: Vec<(usize, crate::engine::sim::camera3d::Mat4)> = lights
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
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
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
        for (slot, ((light_idx, view_proj), tile)) in
            shadow_casters.iter().zip(tiles.iter()).enumerate()
        {
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
            pass.set_viewport(
                tile.x as f32,
                tile.y as f32,
                tile.w as f32,
                tile.h as f32,
                0.0,
                1.0,
            );
            pass.set_scissor_rect(tile.x, tile.y, tile.w, tile.h);
            pass.set_pipeline(&renderer.mesh3d_shadow_pipeline);
            for (layer, (vbuf, ibuf, index_count)) in mesh3d_layers.iter().zip(geometry.iter()) {
                let model = crate::engine::sim::camera3d::model_matrix(
                    layer.origin,
                    layer.angles,
                    layer.scale,
                );
                let light_mvp = crate::engine::sim::camera3d::mat4_mul(&view_proj, &model);
                let mut mvp_bytes = Vec::with_capacity(64);
                for col in light_mvp.iter() {
                    for f in col {
                        mvp_bytes.extend_from_slice(&f.to_le_bytes());
                    }
                }
                let mvp_buf = renderer.make_uniform_buffer(&mvp_bytes, 64);
                let bg = renderer
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
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
    pub(super) fn record_volumetrics(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
    ) {
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
}

/// Pack one mesh's `Mesh3dTransform` uniform (mvp, model_view, normal_view,
/// model) — layout must match `Mesh3dTransform` in `gpu_shaders.wgsl`
/// exactly. `model` (object→world, no view/projection) is what the shadow
/// lookup in `fs_mesh3d` needs `world_pos` for — the light's view-projection
/// is naturally built in world space, unlike everything else here which
/// works in the main camera's view space.
pub(super) fn mesh3d_transform_bytes(
    cam: &crate::engine::sim::camera3d::PerspectiveCamera,
    origin: [f32; 3],
    angles: [f32; 3],
    scale: [f32; 3],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(64 * 4);
    for mat in [
        cam.mvp(origin, angles, scale),
        cam.model_view(origin, angles, scale),
        cam.normal_view(angles, scale),
        crate::engine::sim::camera3d::model_matrix(origin, angles, scale),
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
pub(super) fn mesh3d_lighting_bytes(
    scene: &crate::engine::model::scene::Scene,
    cam: &crate::engine::sim::camera3d::PerspectiveCamera,
    shadow_slots: &[Option<usize>],
    shadow_slot_data: &[(crate::engine::sim::camera3d::Mat4, [f32; 4])],
    cookie: Option<(usize, crate::engine::sim::camera3d::Mat4)>,
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
            crate::engine::gpu::lighting::Light::Point { .. }
                | crate::engine::gpu::lighting::Light::Spot { .. }
                | crate::engine::gpu::lighting::Light::Directional { .. }
                | crate::engine::gpu::lighting::Light::Tube { .. }
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
            push4([
                f.distance_color[0],
                f.distance_color[1],
                f.distance_color[2],
                f.distance_density,
            ]);
            push4([
                f.height_color[0],
                f.height_color[1],
                f.height_color[2],
                f.height_density,
            ]);
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
            crate::engine::gpu::lighting::Light::Point {
                origin,
                color,
                intensity,
                ..
            } => {
                let vp = cam.to_view_space(*origin);
                *pos_slot = [vp[0], vp[1], vp[2], shadow_w()];
                *color_slot = [color[0], color[1], color[2], *intensity];
                // spot_slot stays [0,0,0,0] — zero-length direction reads as
                // "not a spot" in `mesh3d_spot_factor`.
            }
            crate::engine::gpu::lighting::Light::Spot {
                origin,
                direction,
                exponent,
                color,
                intensity,
                ..
            } => {
                let vp = cam.to_view_space(*origin);
                // `build_shadow_atlas` now casts shadows from `Light::Spot`
                // too (see `engine::gpu::shadow::spot_light_view_proj`), so
                // `shadow_w()` is real here, not always 0.0.
                *pos_slot = [vp[0], vp[1], vp[2], shadow_w()];
                *color_slot = [color[0], color[1], color[2], *intensity];
                let vd = cam.to_view_direction(*direction);
                *spot_slot = [vd[0], vd[1], vd[2], *exponent];
            }
            crate::engine::gpu::lighting::Light::Directional {
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
            crate::engine::gpu::lighting::Light::Tube {
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
            .unwrap_or_else(crate::engine::sim::camera3d::identity);
        for col in vp.iter() {
            push4(*col);
        }
    }
    for slot in 0..MESH3D_MAX_SHADOW_LIGHTS {
        let uv_rect = shadow_slot_data
            .get(slot)
            .map(|(_, r)| *r)
            .unwrap_or([0.0; 4]);
        push4(uv_rect);
    }
    let cookie_view_proj = cookie
        .map(|(_, vp)| vp)
        .unwrap_or_else(crate::engine::sim::camera3d::identity);
    for col in cookie_view_proj.iter() {
        push4(*col);
    }
    bytes
}
