//! Drawing a frame: the per-frame render (layers in scene order, effects,
//! composite, bloom), the image, particle and 3D mesh layer passes, the
//! backdrop copy refraction effects sample, and handing the result over as
//! pixels or onto a surface.

use super::*;

impl GpuSceneInstance {
    /// For a composition/fullscreen layer, copy the scene drawn so far under
    /// its quad into its texture, so its effects process the backdrop.
    pub(super) fn capture_backdrop(&self, encoder: &mut wgpu::CommandEncoder, layer_idx: usize) {
        let layer = &self.layers[layer_idx];
        if layer.backdrop.is_none() {
            return;
        }
        let dst = &layer.frames[0];
        let (w, h) = (self.width as i64, self.height as i64);
        // Quad → target pixels (texture rows run top-down).
        let x0 = (((layer.rect[0] - layer.rect[2]) + 1.0) * 0.5 * w as f32).round() as i64;
        let y0 = ((1.0 - (layer.rect[1] + layer.rect[3])) * 0.5 * h as f32).round() as i64;
        let (sx, sy) = (x0.max(0), y0.max(0));
        let (dx, dy) = (sx - x0, sy - y0);
        let cw = (dst.width() as i64 - dx).min(w - sx);
        let ch = (dst.height() as i64 - dy).min(h - sy);
        if cw <= 0 || ch <= 0 {
            return;
        }
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: sx as u32,
                    y: sy as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: dst,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: dx as u32,
                    y: dy as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: cw as u32,
                height: ch as u32,
                depth_or_array_layers: 1,
            },
        );
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
    pub(super) fn draw_one_mesh3d(
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

    /// Render the next frame into the internal scene target.
    pub fn render(&mut self) {
        let time = self.start.elapsed().as_secs_f32();
        let delta = (time - self.last_time).max(0.0);
        self.last_time = time;
        let dynamics = self.dynamics.update(time, delta, self.mouse_norm);

        self.tick_timelines(delta);
        self.tick_scripts(time, delta);

        // Refresh the g_AudioSpectrum* UBO from the latest captured window, and
        // drive audio-reactive particle emitters from the overall loudness.
        if let Some(cap) = &self.audio_capture {
            let spectrum = cap.spectrum();
            self.renderer.queue.write_buffer(
                &self.renderer.audio_buf,
                0,
                &spectrum.to_uniform_bytes(),
            );
            let bands =
                std::array::from_fn(|i| (spectrum.s16_left[i] + spectrum.s16_right[i]) * 0.5);
            for system in &mut self.particle_systems {
                system.set_audio_bands(bands);
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
        let rank = |order: usize| self.draw_rank.get(order).copied().unwrap_or(order as f64);
        items.sort_by(|(a, _), (b, _)| rank(*a).total_cmp(&rank(*b)));

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

    pub(super) fn draw_image_layer_gpu(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        layer_idx: usize,
        time: f32,
        dynamics: CameraFrameDynamics,
    ) {
        self.capture_backdrop(encoder, layer_idx);
        {
            let layer = &self.layers[layer_idx];
            let frame_idx = if let Some(f) = layer.script_frame {
                f.min(layer.frames.len() - 1)
            } else if layer.frames.len() > 1 && layer.frame_duration_ms > 0 {
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
                .zip(&self.effect_enabled)
                .any(|(inst, on)| *on && inst.layer_idx == obj_index);

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
                .filter(|(k, inst)| inst.layer_idx == obj_index && self.effect_enabled[*k])
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
                let quad3d_bg =
                    self.renderer
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
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
    pub(super) fn draw_particle_layer_gpu(
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
    /// (model::resolve/cpu::animated) share the same underlying rasterizer.
    pub(super) fn draw_particle_layer_cpu_raster(
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

    pub(super) fn record_bloom(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
    ) {
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
        let full = crate::content::settings::FULL_REGION;
        self.present_region(view, format, (view_w, view_h), full);
    }

    /// Blit the last rendered frame into `view`, which shows `region` of a
    /// `canvas` the scene is aspect-filled into (a spanned or cloned
    /// wallpaper renders once and presents to every monitor this way).
    pub fn present_region(
        &mut self,
        view: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        canvas: (u32, u32),
        region: [f32; 4],
    ) {
        let (uv_scale, uv_offset) = crate::content::ScaleMode::Fill.uv_transform_region(
            (self.width, self.height),
            canvas,
            region,
        );
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

#[allow(clippy::too_many_arguments)]
pub(super) fn composite_params(
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
pub(super) const FULLSCREEN_RECT: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// Always mode 0 (`fs_composite`, never reads `dest_copy_tex`), so the
/// resolution field is unused — a dummy value is harmless here.
pub(super) fn passthrough_composite_params() -> [u8; 64] {
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
