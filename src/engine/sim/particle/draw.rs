//! Turning particles into pixels: the CPU rasterizer (textured sprites,
//! blending, sampling), rope/trail geometry, and the vertices the GPU
//! renderer draws.

use super::*;

impl ParticleSystem {
    /// Total pixel coverage the rasterizer would write for one frame: the
    /// sum of every particle's quad area, each clipped to the bounding-box
    /// area. Overdraw — many huge overlapping sprites — makes this far
    /// larger than the bounding box itself, and it is the number a raster
    /// budget must divide, not the buffer size.
    pub fn coverage(&self, bbox_area: f32) -> f32 {
        self.particles
            .iter()
            .map(|p| {
                let side = (p.size * 2.0).max(1.0);
                (side * side).min(bbox_area)
            })
            .sum()
    }

    /// Bounding box (`min_x, min_y, max_x, max_y`) over all alive particles'
    /// `position ± size` (with a small margin for the soft-falloff glow
    /// radius) — `None` when nothing is alive, so callers can skip a wasted
    /// raster/upload entirely. Lets `render_onto` be called with a canvas
    /// sized to just this box instead of the full scene.
    pub fn bounds(&self) -> Option<(f32, f32, f32, f32)> {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        for p in &self.particles {
            // `p.size` is a radius (see `sizerandom`'s halving comment) for
            // both the circle and textured-quad draws; pad by a couple of
            // pixels for the soft falloff's antialiasing.
            let half = p.size.max(1.0) + 2.0;
            min_x = min_x.min(p.x - half);
            min_y = min_y.min(p.y - half);
            max_x = max_x.max(p.x + half);
            max_y = max_y.max(p.y + half);
        }
        (!self.particles.is_empty()).then_some((min_x, min_y, max_x, max_y))
    }

    /// `origin` shifts every particle's canvas-space position by `-origin` —
    /// lets callers raster into a canvas sized to just a system's bounding
    /// box (see `bounds()`) instead of the full scene. Existing full-canvas
    /// callers pass `[0.0, 0.0]`.
    pub fn render_onto(
        &self,
        canvas: &mut RgbaImage,
        sprite: Option<&ParticleSprite>,
        origin: [f32; 2],
    ) {
        self.render_onto_blended(canvas, sprite, origin, false);
    }

    /// `render_onto` with the material's blending mode: `additive` layers
    /// accumulate premultiplied contributions (the reference renders each
    /// particle quad with `glBlendFunc(GL_SRC_ALPHA, GL_ONE)`, CPass.cpp),
    /// so overlapping particles brighten each other instead of the newer
    /// one occluding the older via "over" compositing.
    pub fn render_onto_blended(
        &self,
        canvas: &mut RgbaImage,
        sprite: Option<&ParticleSprite>,
        origin: [f32; 2],
        additive: bool,
    ) {
        self.render_onto_scaled(canvas, sprite, origin, additive, 1.0);
    }

    /// `render_onto_blended` with all positions/sizes multiplied by `scale`
    /// — lets a caller rasterize into a smaller-than-scene buffer that the
    /// GPU compositor then stretches back up. Soft, huge sprites (smoke
    /// clouds, rain sheets — radii in the hundreds of pixels) are the ones
    /// that need this, and they lose nothing visible to the upscale, while
    /// the CPU rasterization cost drops with scale².
    pub fn render_onto_scaled(
        &self,
        canvas: &mut RgbaImage,
        sprite: Option<&ParticleSprite>,
        origin: [f32; 2],
        additive: bool,
        scale: f32,
    ) {
        if self.rope_mode {
            self.render_rope_onto(canvas, sprite, origin, additive, scale);
            self.render_children_onto(canvas, origin, scale);
            return;
        }

        let w = canvas.width() as i32;
        let h = canvas.height() as i32;

        for p in &self.particles {
            // Authored alpha straight through — the reference draws
            // `v_Color.a = p.alpha` with no damping factor.
            let alpha = (p.alpha * 255.0).clamp(0.0, 255.0) as u8;
            if alpha == 0 {
                continue;
            }
            let sz = (p.size * scale) as i32;
            if sz <= 0 {
                continue;
            }
            let px_pos = (p.x - origin[0]) * scale;
            let py_pos = (p.y - origin[1]) * scale;

            if let Some(sprite) = sprite {
                // `p.frame` is only ever set (in `step`) when the sprite has
                // more than one frame; a single-frame sprite always draws
                // frame 0 regardless of its (unused, still -1.0) value.
                //
                // Cross-fade: the fractional part of `p.frame` blends toward
                // the NEXT frame, clamped at the last frame (the real
                // engine's ComputeSpriteFrame uses
                // `nextFrame = min(numFrames - 1, currentFrame + 1)` — no
                // wrap, in any animation mode). RandomFrame assigns integer
                // frames, so its fraction is 0 and no second sample happens.
                let (frame_idx, next_frame) = if sprite.frames.len() > 1 {
                    let f = p.frame.max(0.0);
                    let a = (f as usize).min(sprite.frames.len() - 1);
                    let b = (a + 1).min(sprite.frames.len() - 1);
                    let blend = f.fract();
                    if b != a && blend > 1.0 / 255.0 {
                        (a, Some((&sprite.frames[b], blend)))
                    } else {
                        (a, None)
                    }
                } else {
                    (0, None)
                };
                let tex = &sprite.frames[frame_idx];
                // `spritetrail`: align the quad's V axis with the particle's
                // velocity and stretch it by clamp(|v|*length, min, max) —
                // genericparticle.vert's TRAILRENDERER path, where `up` =
                // normalized velocity scaled by that clamp and texture-top
                // (v=0) points along it.
                let (half_w, half_h, rotation) = match self.sprite_trail {
                    Some(t) => {
                        let vel_len = (p.vx * p.vx + p.vy * p.vy).sqrt();
                        let trail = (vel_len * t.length)
                            .clamp(t.min_length.min(t.max_length), t.max_length)
                            .max(0.05);
                        let angle = if vel_len > 0.0001 {
                            p.vy.atan2(p.vx) + std::f32::consts::FRAC_PI_2
                        } else {
                            p.rotation
                        };
                        (p.size * scale, p.size * trail * scale, angle)
                    }
                    None => (p.size * scale, p.size * scale, p.rotation),
                };
                draw_textured_particle(
                    canvas,
                    px_pos,
                    py_pos,
                    half_w,
                    half_h,
                    rotation,
                    p.color,
                    alpha,
                    tex,
                    next_frame,
                    sprite.overbright,
                    additive,
                    scale < 0.9,
                );
                continue;
            }

            let cx = px_pos as i32;
            let cy = py_pos as i32;

            let sz2 = (sz * sz) as f32;
            for dy in -sz..=sz {
                for dx in -sz..=sz {
                    let d2 = (dx * dx + dy * dy) as f32;
                    if d2 > sz2 {
                        continue;
                    }
                    let px = cx + dx;
                    let py = cy + dy;
                    if px < 0 || py < 0 || px >= w || py >= h {
                        continue;
                    }
                    // Soft radial falloff — a flat-alpha disc reads as a harsh,
                    // hard-edged blob, especially for large ambient sprites
                    // (fog/light shafts) that are meant to be soft glows.
                    let t = 1.0 - (d2 / sz2).sqrt();
                    let falloff = t * t;
                    let src_a = (alpha as f32 / 255.0) * falloff;
                    let src_c = [
                        p.color[0] as f32 / 255.0,
                        p.color[1] as f32 / 255.0,
                        p.color[2] as f32 / 255.0,
                    ];
                    let dst = canvas.get_pixel_mut(px as u32, py as u32);
                    blend_pixel(dst, src_c, src_a, additive);
                }
            }
        }

        self.render_children_onto(canvas, origin, scale);
    }

    /// Draws every child instance over the parent's output, each with its
    /// own material sprite and blending.
    pub(super) fn render_children_onto(
        &self,
        canvas: &mut RgbaImage,
        origin: [f32; 2],
        scale: f32,
    ) {
        for child in &self.children {
            for inst in &child.instances {
                inst.system.render_onto_scaled(
                    canvas,
                    child.sprite.as_ref(),
                    origin,
                    child.additive,
                    scale,
                );
            }
        }
    }

    /// Draws a connected ribbon through living particles (oldest-first, since
    /// `step()`'s retain preserves order and new spawns are appended) instead
    /// of independent circles. Position is interpolated with the reference's
    /// exact Catmull-Rom formula (CParticle.cpp renderRope); size/color/alpha
    /// are linearly interpolated per sub-segment, and the material texture is
    /// stretched across the ribbon: U spans the width (the beam profile that
    /// makes a 100px-wide rope read as a hairline filament), V runs along the
    /// rope as `(trailPosition + uvY) / (trailLength - 1)` with `uvscale`
    /// tiling, optional arc-length smoothing, and 1-cycle/sec scrolling —
    /// CParticle::renderRope's exact vertex semantics, evaluated per-pixel
    /// here instead of in its rope shader.
    pub(super) fn render_rope_onto(
        &self,
        canvas: &mut RgbaImage,
        sprite: Option<&ParticleSprite>,
        origin: [f32; 2],
        additive: bool,
        scale: f32,
    ) {
        let Some(geo) = self.build_rope_geometry(origin, scale) else {
            return;
        };
        let tex = sprite.and_then(|s| s.frames.first());
        let overbright = sprite.map(|s| s.overbright).unwrap_or(1.0);
        for &(i, v_a, v_b) in &geo.quads {
            let (a, b) = (&geo.points[i], &geo.points[i + 1]);
            fill_rope_segment(
                canvas, a.pos, b.pos, a.normal, b.normal, a.size, b.size, a.color, b.color,
                a.alpha, b.alpha, additive, tex, overbright, v_a, v_b,
            );
        }
    }

    /// Builds the rope's shared geometry — Catmull-Rom subpoints, per-joint
    /// central-difference normals, and per-quad V ranges — consumed by both
    /// the CPU scanline fill and the GPU vertex emitter.
    pub(super) fn build_rope_geometry(&self, origin: [f32; 2], scale: f32) -> Option<RopeGeometry> {
        let n = self.particles.len();
        if n < 2 {
            return None;
        }

        let subdiv = self.rope_subdivision;
        let mut subpoints = Vec::with_capacity((n - 1) * subdiv + 1);
        for i in 0..(n - 1) {
            let idx0 = i.saturating_sub(1);
            let p0 = &self.particles[idx0];
            let p1 = &self.particles[i];
            let p2 = &self.particles[i + 1];
            let idx3 = (i + 2).min(n - 1);
            let p3 = &self.particles[idx3];

            for step in 0..subdiv {
                let t = step as f32 / subdiv as f32;
                let pos =
                    catmull_rom_vec2([p0.x, p0.y], [p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y], t);
                subpoints.push(RopePoint {
                    pos: [(pos[0] - origin[0]) * scale, (pos[1] - origin[1]) * scale],
                    size: lerp_f32(p1.size, p2.size, t) * scale,
                    alpha: lerp_f32(p1.alpha, p2.alpha, t),
                    color: [
                        lerp_f32(p1.color[0] as f32, p2.color[0] as f32, t) as u8,
                        lerp_f32(p1.color[1] as f32, p2.color[1] as f32, t) as u8,
                        lerp_f32(p1.color[2] as f32, p2.color[2] as f32, t) as u8,
                    ],
                    normal: [0.0, 1.0],
                });
            }
        }
        let last = &self.particles[n - 1];
        subpoints.push(RopePoint {
            pos: [(last.x - origin[0]) * scale, (last.y - origin[1]) * scale],
            size: last.size * scale,
            color: last.color,
            alpha: last.alpha,
            normal: [0.0, 1.0],
        });

        // V coordinates along the rope — CParticle::renderRope's exact
        // semantics: each sub-segment quad consumes 1/(trailLength-1) of UV
        // space, `uvscale` divides the usable length (values > 1 tile the
        // texture), `uvsmoothing` redistributes V by arc length (only when
        // all lifetimes match and scrolling is off), and `uvscrolling`
        // shifts one full UV cycle per second.
        let total_sub = (subpoints.len() - 1) as f32;
        let usable_len = (total_sub / self.rope_uv_scale).max(f32::EPSILON);
        let use_smoothing =
            self.rope_uv_smoothing && self.life_min == self.life_max && !self.rope_uv_scrolling;
        let mut cumulative_arc = Vec::new();
        let mut total_arc = 0.0f32;
        if use_smoothing {
            cumulative_arc = Vec::with_capacity(subpoints.len());
            cumulative_arc.push(0.0);
            for pair in subpoints.windows(2) {
                let (dx, dy) = (
                    pair[1].pos[0] - pair[0].pos[0],
                    pair[1].pos[1] - pair[0].pos[1],
                );
                total_arc += (dx * dx + dy * dy).sqrt();
                cumulative_arc.push(total_arc);
            }
        }
        let scroll_offset = if self.rope_uv_scrolling {
            (self.time % 10000.0) * usable_len
        } else {
            0.0
        };

        // Per-joint width directions — genericropeparticle.vert's
        // `trailRightStart = cross(eye, end − prev)` / `trailRightEnd =
        // cross(eye, after − start)`: the perpendicular of the CENTRAL-
        // DIFFERENCE tangent at each joint, so consecutive quads share
        // their edge exactly and the ribbon reads as connected trapezoids
        // instead of independently-angled facets. In screen space
        // `cross((0,0,1), t)` is the 90° rotate `(−t.y, t.x)`.
        let joint_normal = |i: usize| -> [f32; 2] {
            let prev = &subpoints[i.saturating_sub(1)];
            let next = &subpoints[(i + 1).min(subpoints.len() - 1)];
            let t = [next.pos[0] - prev.pos[0], next.pos[1] - prev.pos[1]];
            let len = (t[0] * t[0] + t[1] * t[1]).sqrt();
            if len < 0.0001 {
                [0.0, 1.0]
            } else {
                [-t[1] / len, t[0] / len]
            }
        };
        let normals: Vec<[f32; 2]> = (0..subpoints.len()).map(joint_normal).collect();
        for (i, n) in normals.into_iter().enumerate() {
            subpoints[i].normal = n;
        }

        let quads = (0..subpoints.len() - 1)
            .map(|s| {
                let trail_position = if use_smoothing && total_arc > 0.0 {
                    cumulative_arc[s] / total_arc * total_sub
                } else {
                    s as f32
                } + scroll_offset;
                (
                    s,
                    trail_position / usable_len,
                    (trail_position + 1.0) / usable_len,
                )
            })
            .collect();

        Some(RopeGeometry {
            points: subpoints,
            quads,
        })
    }

    /// Appends pre-transformed triangle-list vertices for the GPU particle
    /// pipeline: 6 per sprite quad (rotation/spritetrail applied here, the
    /// same math as the CPU textured draw) or 6 per rope sub-quad. Positions
    /// are absolute scene pixels; the vertex shader converts to NDC.
    /// `frame_count` is the sprite's frame count (0/1 = static frame 0).
    pub fn emit_gpu_vertices(&self, out: &mut Vec<GpuVertex>, frame_count: usize) {
        if self.rope_mode {
            self.emit_rope_gpu_vertices(out);
            return;
        }
        for p in &self.particles {
            if p.alpha <= 0.0 || p.size <= 0.0 {
                continue;
            }
            // Frame/cross-fade selection — mirrors the CPU textured draw.
            let (frame, blend) = if frame_count > 1 {
                let f = p.frame.max(0.0);
                let a = (f as usize).min(frame_count - 1);
                let b = (a + 1).min(frame_count - 1);
                let fract = f.fract();
                if b != a && fract > 1.0 / 255.0 {
                    (a as f32, fract)
                } else {
                    (a as f32, 0.0)
                }
            } else {
                (0.0, 0.0)
            };
            let (half_w, half_h, rotation) = match self.sprite_trail {
                Some(t) => {
                    let vel_len = (p.vx * p.vx + p.vy * p.vy).sqrt();
                    let trail = (vel_len * t.length)
                        .clamp(t.min_length.min(t.max_length), t.max_length)
                        .max(0.05);
                    let angle = if vel_len > 0.0001 {
                        p.vy.atan2(p.vx) + std::f32::consts::FRAC_PI_2
                    } else {
                        p.rotation
                    };
                    (p.size, p.size * trail, angle)
                }
                None => (p.size, p.size, p.rotation),
            };
            let (c, sn) = (rotation.cos(), rotation.sin());
            let color = [
                p.color[0] as f32 / 255.0,
                p.color[1] as f32 / 255.0,
                p.color[2] as f32 / 255.0,
                p.alpha.clamp(0.0, 1.0),
            ];
            let corner = |lx: f32, ly: f32, u: f32, v: f32| GpuVertex {
                pos: [p.x + lx * c - ly * sn, p.y + lx * sn + ly * c],
                uv: [u, v],
                color,
                frame_blend: [frame, blend, 0.0, 0.0],
            };
            let tl = corner(-half_w, -half_h, 0.0, 0.0);
            let tr = corner(half_w, -half_h, 1.0, 0.0);
            let br = corner(half_w, half_h, 1.0, 1.0);
            let bl = corner(-half_w, half_h, 0.0, 1.0);
            out.extend_from_slice(&[tl, tr, br, br, bl, tl]);
        }
    }

    pub(super) fn emit_rope_gpu_vertices(&self, out: &mut Vec<GpuVertex>) {
        let Some(geo) = self.build_rope_geometry([0.0, 0.0], 1.0) else {
            return;
        };
        for &(i, v_a, v_b) in &geo.quads {
            let (a, b) = (&geo.points[i], &geo.points[i + 1]);
            // Same convexity guard as fill_rope_segment: keep both joint
            // normals on the same side when the chain doubles back.
            let nb = if a.normal[0] * b.normal[0] + a.normal[1] * b.normal[1] < 0.0 {
                [-b.normal[0], -b.normal[1]]
            } else {
                b.normal
            };
            let (ha, hb) = (a.size.max(0.5), b.size.max(0.5));
            let col = |p: &RopePoint| {
                [
                    p.color[0] as f32 / 255.0,
                    p.color[1] as f32 / 255.0,
                    p.color[2] as f32 / 255.0,
                    p.alpha.clamp(0.0, 1.0),
                ]
            };
            let (ca, cb) = (col(a), col(b));
            let vert = |pos: [f32; 2], uv: [f32; 2], color: [f32; 4]| GpuVertex {
                pos,
                uv,
                color,
                frame_blend: [0.0, 0.0, 0.0, 0.0],
            };
            // U convention matches fill_rope_segment: +normal side = u 1.
            let a1 = vert(
                [a.pos[0] + a.normal[0] * ha, a.pos[1] + a.normal[1] * ha],
                [1.0, v_a],
                ca,
            );
            let b1 = vert(
                [b.pos[0] + nb[0] * hb, b.pos[1] + nb[1] * hb],
                [1.0, v_b],
                cb,
            );
            let b0 = vert(
                [b.pos[0] - nb[0] * hb, b.pos[1] - nb[1] * hb],
                [0.0, v_b],
                cb,
            );
            let a0 = vert(
                [a.pos[0] - a.normal[0] * ha, a.pos[1] - a.normal[1] * ha],
                [0.0, v_a],
                ca,
            );
            out.extend_from_slice(&[a1, b1, b0, b0, a0, a1]);
        }
    }

    /// Visits every live child instance with its child index (the GPU path
    /// keys uploaded textures by child, since instances spawn at runtime)
    /// and its material blending.
    pub fn visit_gpu_children(&self, f: &mut dyn FnMut(&ParticleSystem, usize, bool)) {
        for (child_idx, child) in self.children.iter().enumerate() {
            for inst in &child.instances {
                f(&inst.system, child_idx, child.additive);
            }
        }
    }

    /// Build-time child info for GPU texture upload: each child's sprite and
    /// whether its preset uses a rope renderer.
    pub fn child_sprite_info(&self) -> Vec<(Option<&ParticleSprite>, bool)> {
        self.children
            .iter()
            .map(|c| {
                let rope = c.config.renderer.iter().any(|r| {
                    r.get("name")
                        .and_then(|n| n.as_str())
                        .map(|n| n == "rope" || n == "ropetrail")
                        .unwrap_or(false)
                });
                (c.sprite.as_ref(), rope)
            })
            .collect()
    }

    pub fn is_rope(&self) -> bool {
        self.rope_mode
    }
}

/// One point along a rope's subdivided spine (see `build_rope_geometry`).
pub(super) struct RopePoint {
    pub(super) pos: [f32; 2],
    pub(super) size: f32,
    pub(super) color: [u8; 3],
    pub(super) alpha: f32,
    /// Per-joint central-difference perpendicular (width direction).
    pub(super) normal: [f32; 2],
}

/// Shared rope geometry: spine points and per-quad `(start_index, v_a, v_b)`.
pub(super) struct RopeGeometry {
    pub(super) points: Vec<RopePoint>,
    pub(super) quads: Vec<(usize, f32, f32)>,
}

/// One pre-transformed vertex for the GPU particle pipeline (triangle list,
/// read from a storage buffer by `vs_particles`). 48 bytes, std430-compatible.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuVertex {
    /// Absolute scene pixels (y-down).
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    /// rgb 0-1, a = particle alpha.
    pub color: [f32; 4],
    /// x = sprite-sheet frame, y = cross-fade weight toward frame+1.
    pub frame_blend: [f32; 4],
}

impl GpuVertex {
    /// Flattens vertices to the byte layout `vs_particles` reads.
    pub fn as_bytes(verts: &[GpuVertex]) -> Vec<u8> {
        let mut out = Vec::with_capacity(verts.len() * 48);
        for v in verts {
            for f in v
                .pos
                .iter()
                .chain(v.uv.iter())
                .chain(v.color.iter())
                .chain(v.frame_blend.iter())
            {
                out.extend_from_slice(&f.to_le_bytes());
            }
        }
        out
    }
}

/// Reference's exact Catmull-Rom formula (CParticle.cpp:2124-2130), applied
/// per-axis.
pub(super) fn catmull_rom_vec2(
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],
    p3: [f32; 2],
    t: f32,
) -> [f32; 2] {
    let t2 = t * t;
    let t3 = t2 * t;
    let mut out = [0.0; 2];
    for axis in 0..2 {
        let (p0, p1, p2, p3) = (p0[axis], p1[axis], p2[axis], p3[axis]);
        out[axis] = 0.5
            * ((2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
    }
    out
}

/// Rasterizes one ribbon quad between two consecutive sub-points: offsets
/// each endpoint by its half-width along its own JOINT normal (the central-
/// difference perpendicular computed by the caller), producing edge-sharing
/// trapezoids exactly like genericropeparticle.vert's
/// `position + mix(trailRightStart, trailRightEnd, uvY) * (uvX*2−1)` — where
/// `trailRight* = normalize(cross(eye, tangent)) * size`, i.e. the half-width
/// IS the particle size (spawn already halved `sizerandom`, so `p.size` is a
/// radius). With a material texture, each pixel samples it at (U = signed
/// offset across the width, V = lerp(v_a, v_b) along the rope, wrapping for
/// tiled `uvscale`) and modulates the particle color/alpha — the beam strip's
/// transparent edges are what make a wide rope render as a thin filament.
/// Without a texture (material failed to resolve), falls back to a flat fill.
#[allow(clippy::too_many_arguments)]
pub(super) fn fill_rope_segment(
    canvas: &mut RgbaImage,
    a: [f32; 2],
    b: [f32; 2],
    normal_a: [f32; 2],
    normal_b: [f32; 2],
    size_a: f32,
    size_b: f32,
    color_a: [u8; 3],
    color_b: [u8; 3],
    alpha_a: f32,
    alpha_b: f32,
    additive: bool,
    tex: Option<&RgbaImage>,
    overbright: f32,
    v_a: f32,
    v_b: f32,
) {
    let dir = [b[0] - a[0], b[1] - a[1]];
    let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
    if len < 0.0001 {
        return;
    }
    let dir = [dir[0] / len, dir[1] / len];
    let (half_a, half_b) = (size_a.max(0.5), size_b.max(0.5));

    // Keep both joint normals on the same side so the quad stays convex —
    // a chain doubling back >90° would otherwise produce a bowtie that the
    // point-in-quad test rejects entirely (the GPU draws such quads as two
    // crossing triangles; a same-side flip is the closest scanline analog).
    let normal_b = if normal_a[0] * normal_b[0] + normal_a[1] * normal_b[1] < 0.0 {
        [-normal_b[0], -normal_b[1]]
    } else {
        normal_b
    };

    let corners = [
        [a[0] + normal_a[0] * half_a, a[1] + normal_a[1] * half_a],
        [b[0] + normal_b[0] * half_b, b[1] + normal_b[1] * half_b],
        [b[0] - normal_b[0] * half_b, b[1] - normal_b[1] * half_b],
        [a[0] - normal_a[0] * half_a, a[1] - normal_a[1] * half_a],
    ];

    let min_x = corners
        .iter()
        .map(|c| c[0])
        .fold(f32::MAX, f32::min)
        .floor()
        .max(0.0) as i32;
    let max_x = corners
        .iter()
        .map(|c| c[0])
        .fold(f32::MIN, f32::max)
        .ceil()
        .min(canvas.width() as f32 - 1.0) as i32;
    let min_y = corners
        .iter()
        .map(|c| c[1])
        .fold(f32::MAX, f32::min)
        .floor()
        .max(0.0) as i32;
    let max_y = corners
        .iter()
        .map(|c| c[1])
        .fold(f32::MIN, f32::max)
        .ceil()
        .min(canvas.height() as f32 - 1.0) as i32;
    if min_x > max_x || min_y > max_y {
        return;
    }

    let w = canvas.width() as i32;
    let h = canvas.height() as i32;
    for py in min_y..=max_y {
        for px in min_x..=max_x {
            if px < 0 || py < 0 || px >= w || py >= h {
                continue;
            }
            let point = [px as f32 + 0.5, py as f32 + 0.5];
            if !point_in_convex_quad(point, &corners) {
                continue;
            }
            // Blend color/alpha by which endpoint the pixel is nearer along
            // the segment (a cheap stand-in for true per-pixel interpolation).
            let seg_t = ((point[0] - a[0]) * dir[0] + (point[1] - a[1]) * dir[1]) / len;
            let seg_t = seg_t.clamp(0.0, 1.0);

            if let Some(tex) = tex {
                // Signed offset from the centerline along the interpolated
                // joint normal, in [-half, +half] at this point's width.
                let center = [lerp_f32(a[0], b[0], seg_t), lerp_f32(a[1], b[1], seg_t)];
                let nl = [
                    lerp_f32(normal_a[0], normal_b[0], seg_t),
                    lerp_f32(normal_a[1], normal_b[1], seg_t),
                ];
                let nl_len = (nl[0] * nl[0] + nl[1] * nl[1]).sqrt().max(0.0001);
                let d_perp =
                    ((point[0] - center[0]) * nl[0] + (point[1] - center[1]) * nl[1]) / nl_len;
                let half_t = lerp_f32(half_a, half_b, seg_t).max(0.5);
                let u = (0.5 + 0.5 * (d_perp / half_t)).clamp(0.0, 1.0);
                let v = lerp_f32(v_a, v_b, seg_t).rem_euclid(1.0);
                let sample = sample_bilinear(tex, u, v);
                let src_a =
                    lerp_f32(alpha_a, alpha_b, seg_t).clamp(0.0, 1.0) * (sample[3] as f32 / 255.0);
                if src_a <= 0.0 {
                    continue;
                }
                let src_c = [
                    (sample[0] as f32 / 255.0)
                        * (lerp_f32(color_a[0] as f32, color_b[0] as f32, seg_t) / 255.0)
                        * overbright,
                    (sample[1] as f32 / 255.0)
                        * (lerp_f32(color_a[1] as f32, color_b[1] as f32, seg_t) / 255.0)
                        * overbright,
                    (sample[2] as f32 / 255.0)
                        * (lerp_f32(color_a[2] as f32, color_b[2] as f32, seg_t) / 255.0)
                        * overbright,
                ];
                let dst = canvas.get_pixel_mut(px as u32, py as u32);
                blend_pixel(dst, src_c, src_a, additive);
                continue;
            }

            let alpha = (lerp_f32(alpha_a, alpha_b, seg_t) * 180.0).clamp(0.0, 255.0) as u8;
            if alpha == 0 {
                continue;
            }
            let color = [
                lerp_f32(color_a[0] as f32, color_b[0] as f32, seg_t) as u8,
                lerp_f32(color_a[1] as f32, color_b[1] as f32, seg_t) as u8,
                lerp_f32(color_a[2] as f32, color_b[2] as f32, seg_t) as u8,
            ];

            let src_a = alpha as f32 / 255.0;
            let src_c = [
                color[0] as f32 / 255.0,
                color[1] as f32 / 255.0,
                color[2] as f32 / 255.0,
            ];
            let dst = canvas.get_pixel_mut(px as u32, py as u32);
            blend_pixel(dst, src_c, src_a, additive);
        }
    }
}

/// Point-in-convex-quad test via consistent cross-product sign across all
/// four edges. `corners` must be wound consistently (either all-CW or
/// all-CCW) — true by construction here (two parallel perpendicular offsets).
pub(super) fn point_in_convex_quad(point: [f32; 2], corners: &[[f32; 2]; 4]) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        let edge = [b[0] - a[0], b[1] - a[1]];
        let to_point = [point[0] - a[0], point[1] - a[1]];
        let cross = edge[0] * to_point[1] - edge[1] * to_point[0];
        if i == 0 {
            sign = cross;
        } else if cross * sign < 0.0 {
            return false;
        }
    }
    true
}

/// Draws one particle as a rotated, textured quad: inverse-rotates each
/// destination pixel in the quad's bounding box back into texture space,
/// bilinear-samples, and composites tinted-by-`color`/scaled-by-`alpha_byte`
/// over the canvas. The reference hands rotation+size to an external,
/// proprietary vertex shader to build the quad (not present in the
/// open-source reference repo) — this CPU inverse-mapping is our own design
/// for the same visual effect.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(super) fn draw_textured_particle(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    half_w: f32,
    half_h: f32,
    rotation: f32,
    color: [u8; 3],
    alpha_byte: u8,
    tex: &RgbaImage,
    next_frame: Option<(&RgbaImage, f32)>,
    overbright: f32,
    additive: bool,
    nearest: bool,
) {
    // `p.size` is already a radius, not a diameter (see `sizerandom`'s
    // comment on the halving done at spawn) — matches the flat-color circle
    // draw's `sz = p.size` convention, so a particle looks the same size
    // whether or not it has a material/texture. The two half-extents differ
    // only for `spritetrail` quads (stretched along velocity).
    let half_w = half_w.max(0.5);
    let half_h = half_h.max(0.5);
    let cos_r = rotation.cos();
    let sin_r = rotation.sin();

    // Bounding box of the rotated quad: the diagonal half-extent covers any
    // rotation angle.
    let diag = (half_w * half_w + half_h * half_h).sqrt();
    let w = canvas.width() as i32;
    let h = canvas.height() as i32;
    let min_x = ((cx - diag).floor() as i32).max(0);
    let max_x = ((cx + diag).ceil() as i32).min(w - 1);
    let min_y = ((cy - diag).floor() as i32).max(0);
    let max_y = ((cy + diag).ceil() as i32).min(h - 1);

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let dx = px as f32 + 0.5 - cx;
            let dy = py as f32 + 0.5 - cy;
            // Inverse-rotate the destination offset back into the quad's own
            // (unrotated) local space.
            let lx = dx * cos_r + dy * sin_r;
            let ly = -dx * sin_r + dy * cos_r;
            if lx < -half_w || lx > half_w || ly < -half_h || ly > half_h {
                continue;
            }
            let u = (lx + half_w) / (2.0 * half_w);
            let v = (ly + half_h) / (2.0 * half_h);
            let mut sample = if nearest {
                sample_nearest(tex, u, v)
            } else {
                sample_bilinear(tex, u, v)
            };
            // Spritesheet cross-fade: mix toward the next frame's sample
            // (genericparticle.frag's SPRITESHEETBLEND path).
            if let Some((tex_b, blend)) = next_frame {
                let b = if nearest {
                    sample_nearest(tex_b, u, v)
                } else {
                    sample_bilinear(tex_b, u, v)
                };
                for c in 0..4 {
                    sample[c] = (sample[c] as f32 + (b[c] as f32 - sample[c] as f32) * blend)
                        .clamp(0.0, 255.0) as u8;
                }
            }
            let src_a = (sample[3] as f32 / 255.0) * (alpha_byte as f32 / 255.0);
            if src_a <= 0.0 {
                continue;
            }
            // genericparticle.frag: `color.rgb *= g_Overbright` after the
            // texture × vertex-color multiply. blend_pixel clamps, so
            // overbright > 1 saturates exactly like the GPU's clamped u8 target.
            let src_c = [
                (sample[0] as f32 / 255.0) * (color[0] as f32 / 255.0) * overbright,
                (sample[1] as f32 / 255.0) * (color[1] as f32 / 255.0) * overbright,
                (sample[2] as f32 / 255.0) * (color[2] as f32 / 255.0) * overbright,
            ];
            let dst = canvas.get_pixel_mut(px as u32, py as u32);
            blend_pixel(dst, src_c, src_a, additive);
        }
    }
}

/// Composites one source sample into a canvas pixel.
///
/// Normal/translucent layers use standard "over" compositing (not a plain
/// lerp): the destination isn't always opaque — the GPU path renders
/// particles into their own transparent scene-sized buffer before
/// compositing it as a texture, so overlapping particles must accumulate
/// alpha correctly rather than assuming a fully-opaque backdrop.
///
/// Additive layers accumulate premultiplied contributions instead — the
/// reference draws every particle quad with
/// `glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_SRC_ALPHA, GL_ONE)`
/// (CPass.cpp), i.e. `dst += src * src_a` — so overlapping glows brighten
/// each other rather than the newest quad occluding the ones below. The
/// buffer then holds already-premultiplied color, which the GPU composite's
/// pure-add mode adds to the scene without re-weighting by alpha.
pub(super) fn blend_pixel(dst: &mut image::Rgba<u8>, src_c: [f32; 3], src_a: f32, additive: bool) {
    if additive {
        for i in 0..3 {
            let dst_c = dst[i] as f32 / 255.0;
            dst[i] = ((dst_c + src_c[i] * src_a) * 255.0).clamp(0.0, 255.0) as u8;
        }
        let dst_a = dst[3] as f32 / 255.0;
        dst[3] = ((dst_a + src_a * src_a) * 255.0).clamp(0.0, 255.0) as u8;
        return;
    }
    let dst_a = dst[3] as f32 / 255.0;
    let out_a = src_a + dst_a * (1.0 - src_a);
    if out_a > 0.0 {
        for i in 0..3 {
            let dst_c = dst[i] as f32 / 255.0;
            let out_c = (src_c[i] * src_a + dst_c * dst_a * (1.0 - src_a)) / out_a;
            dst[i] = (out_c * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    dst[3] = (out_a * 255.0).clamp(0.0, 255.0) as u8;
}

/// Point sampling — used when the raster target is itself downscaled (the
/// GPU upscale blurs anyway), saving the 4-tap filter per pixel.
pub(super) fn sample_nearest(tex: &RgbaImage, u: f32, v: f32) -> [u8; 4] {
    let (tw, th) = (tex.width(), tex.height());
    if tw == 0 || th == 0 {
        return [0; 4];
    }
    let x = ((u.clamp(0.0, 1.0) * (tw as f32 - 1.0)) + 0.5) as u32;
    let y = ((v.clamp(0.0, 1.0) * (th as f32 - 1.0)) + 0.5) as u32;
    tex.get_pixel(x.min(tw - 1), y.min(th - 1)).0
}

pub(super) fn sample_bilinear(tex: &RgbaImage, u: f32, v: f32) -> [u8; 4] {
    let (tw, th) = (tex.width(), tex.height());
    let fx = (u.clamp(0.0, 1.0) * (tw as f32 - 1.0).max(0.0)).max(0.0);
    let fy = (v.clamp(0.0, 1.0) * (th as f32 - 1.0).max(0.0)).max(0.0);
    let x0 = fx.floor() as u32;
    let y0 = fy.floor() as u32;
    let x1 = (x0 + 1).min(tw - 1);
    let y1 = (y0 + 1).min(th - 1);
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);

    let p00 = tex.get_pixel(x0, y0);
    let p10 = tex.get_pixel(x1, y0);
    let p01 = tex.get_pixel(x0, y1);
    let p11 = tex.get_pixel(x1, y1);

    let mut out = [0u8; 4];
    for i in 0..4 {
        let top = p00[i] as f32 * (1.0 - tx) + p10[i] as f32 * tx;
        let bottom = p01[i] as f32 * (1.0 - tx) + p11[i] as f32 * tx;
        out[i] = (top * (1.0 - ty) + bottom * ty).clamp(0.0, 255.0) as u8;
    }
    out
}
