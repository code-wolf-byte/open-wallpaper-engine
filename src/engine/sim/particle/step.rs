//! Simulation: advancing a system by one frame (emission, initializers,
//! operators, ageing) and its child systems, plus the inputs that steer it
//! (scene forces, audio bands, instance values, origin shifts).

use super::*;

impl ParticleSystem {
    /// Change one `instanceoverride` value live (SceneScript
    /// `thisLayer.instance.<key> = …`). `count` (the pool size) only applies
    /// at load and is ignored here.
    pub fn set_instance_value(&mut self, key: &str, value: &serde_json::Value) {
        let num = || match value {
            serde_json::Value::Number(n) => n.as_f64().map(|v| v as f32),
            serde_json::Value::String(s) => s.trim().parse().ok(),
            _ => None,
        };
        let vec3 = || -> Option<[f32; 3]> {
            match value {
                serde_json::Value::Array(a) if a.len() >= 3 => {
                    Some(std::array::from_fn(|i| a[i].as_f64().unwrap_or(0.0) as f32))
                }
                other => value_as_vec3_pub(other),
            }
        };
        match key {
            "alpha" => self.alpha_mult = num().unwrap_or(self.alpha_mult),
            "rate" => self.rate_mult = num().unwrap_or(self.rate_mult).max(0.0),
            "size" => self.size_mult = num().unwrap_or(self.size_mult),
            "speed" => self.speed_mult = num().unwrap_or(self.speed_mult),
            "lifetime" => self.lifetime_mult = num().unwrap_or(self.lifetime_mult),
            "colorn" => self.colorn_mult = vec3().unwrap_or(self.colorn_mult),
            "color" => {
                if let Some(c) = vec3() {
                    self.color_override = Some(c.map(|v| v.clamp(0.0, 255.0) as u8));
                }
            }
            _ => {}
        }
    }

    /// Apply the scene-global constant force (`general.gravity*`/`wind*`).
    pub fn set_scene_force(&mut self, force: [f32; 2]) {
        self.scene_force = force;
    }

    /// Set the 16-band spectrum (~[0,1] per band) driving audio-reactive emitters.
    pub fn set_audio_bands(&mut self, bands: [f32; 16]) {
        self.audio_bands = bands;
    }

    /// Tell the system its resolved sprite's frame count/loop duration, so
    /// `step()` can advance each particle's `frame` — called once by the
    /// caller right after resolving the sprite (this module itself never
    /// loads textures). A no-op frame count of 0/1 leaves every particle's
    /// `frame` at its spawn sentinel, matching `render_onto`'s "always frame
    /// 0" behavior for non-animated sprites.
    pub fn set_sprite_frames(&mut self, frame_count: usize, duration: f32) {
        self.sprite_frame_count = frame_count;
        self.sprite_duration = duration;
    }

    pub fn step(&mut self, dt: f32) {
        self.time += dt;
        // `boids`: needs every particle's position/velocity as of the START
        // of this frame (not yet mutated by this frame's own movement), so
        // it's computed as a separate read-only pass before the main
        // per-particle loop mutates anything.
        let boids_accel = self
            .boids
            .as_ref()
            .map(|b| compute_boids_accel(&self.particles, b));
        for (particle_index, p) in self.particles.iter_mut().enumerate() {
            // `movement` operator (CParticle.cpp createMovementOperator):
            // position integrates the CURRENT velocity first, and only then
            // do this frame's forces modify velocity (for the next frame) —
            // gravity scaled by the instance speed override, then drag decay
            // clamped so `drag*dt > 1` stops the particle instead of
            // reversing it.
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.vx += (self.gravity[0] * self.speed_mult + self.scene_force[0]) * dt;
            p.vy += (self.gravity[1] * self.speed_mult + self.scene_force[1]) * dt;
            let drag_factor = (1.0 - self.movement_drag * dt).max(0.0);
            p.vx *= drag_factor;
            p.vy *= drag_factor;

            // `turbulence` operator (CParticle.cpp createTurbulenceOperator):
            // normalized curl-noise direction, sampled at a time-scrolled,
            // scaled particle position, masked per axis, added to velocity.
            if let Some(t) = self.turbulence {
                if t.speed > 0.0001 {
                    let noise_pos = [
                        (p.x + t.phase + t.timescale * self.time) * t.noise_scale,
                        p.y * t.noise_scale,
                        0.0,
                    ];
                    let curl = crate::engine::sim::noise::curl_noise(noise_pos);
                    let len = vec3_length(curl);
                    if len > 0.0001 {
                        let s = t.speed / len * dt * self.speed_mult;
                        p.vx += curl[0] * t.mask[0] * s;
                        p.vy += curl[1] * t.mask[1] * s;
                    }
                }
            }

            // `vortex` operator (CParticle.cpp createVortexOperator).
            if let Some(v) = self.vortex {
                let center_base = self
                    .control_points
                    .get(v.control_point.max(0) as usize)
                    .copied()
                    .unwrap_or([0.0; 3]);
                let center = [
                    center_base[0] + v.offset[0],
                    center_base[1] + v.offset[1],
                    center_base[2] + v.offset[2],
                ];
                // The reference computes tangents in its y-up sim space and
                // projects y-up; our space is y-down (a reflection, det -1),
                // under which cross products flip sign: mirror the axis's y
                // and negate the resulting tangent to reproduce the same
                // on-screen spin direction.
                let axis = vec3_normalize_or([v.axis[0], -v.axis[1], v.axis[2]], [0.0, 0.0, 1.0]);
                let to_particle = [p.x - center[0], p.y - center[1], -center[2]];

                // Infinite axis: cylinder shape (project out the axis
                // component); else full 3D distance (sphere shape).
                let radial = if v.infinite_axis {
                    let axial = vec3_dot(to_particle, axis);
                    [
                        to_particle[0] - axis[0] * axial,
                        to_particle[1] - axis[1] * axial,
                        to_particle[2] - axis[2] * axial,
                    ]
                } else {
                    to_particle
                };
                let distance = vec3_length(radial);

                let tangent = vec3_cross(axis, radial);
                let tangent_len = vec3_length(tangent);
                if tangent_len > 0.001 {
                    // Negated: see the reflection note on `axis` above.
                    let tangent = [
                        -tangent[0] / tangent_len,
                        -tangent[1] / tangent_len,
                        -tangent[2] / tangent_len,
                    ];

                    let mut speed = 0.0;
                    let mut radial_force = [0.0f32; 3];
                    if v.ring_shape {
                        let ring_inner = v.ring_radius - v.ring_width * 0.5;
                        let ring_outer = v.ring_radius + v.ring_width * 0.5;
                        if distance < ring_inner {
                            // Hollow center: no spin.
                        } else if distance <= ring_outer {
                            let t = (distance - ring_inner) / v.ring_width;
                            speed = v.speed_inner + (v.speed_outer - v.speed_inner) * t;
                        } else if distance <= ring_outer + v.ring_pull_distance {
                            let pull_t = (distance - ring_outer) / v.ring_pull_distance;
                            speed = v.speed_outer * (1.0 - pull_t);
                            if distance > 0.001 {
                                let toward = vec3_normalize_or(radial, [0.0; 3]);
                                let f = -v.ring_pull_force * pull_t;
                                radial_force = [toward[0] * f, toward[1] * f, toward[2] * f];
                            }
                        }
                    } else {
                        let dis_mid = v.distance_outer - v.distance_inner + 0.1;
                        speed = if dis_mid < 0.0 || distance < v.distance_inner {
                            v.speed_inner
                        } else if distance > v.distance_outer {
                            v.speed_outer
                        } else {
                            let t = (distance - v.distance_inner) / dis_mid;
                            v.speed_inner + (v.speed_outer - v.speed_inner) * t
                        };
                    }

                    let k = dt * self.speed_mult;
                    p.vx += (tangent[0] * speed + radial_force[0]) * k;
                    p.vy += (tangent[1] * speed + radial_force[1]) * k;
                    if v.maintain_distance && distance > 0.001 {
                        let toward = vec3_normalize_or(radial, [0.0; 3]);
                        p.vx -= toward[0] * v.center_force * k;
                        p.vy -= toward[1] * v.center_force * k;
                    }
                }
            }

            // `boids` operator: apply this particle's precomputed
            // separation/alignment/cohesion acceleration, then cap speed.
            if let (Some(accel), Some(b)) = (&boids_accel, &self.boids) {
                let [ax, ay] = accel[particle_index];
                let strength = if b.variable_strength {
                    // Deterministic per-particle jitter in [0.5, 1.5] from a
                    // cheap integer hash of the particle's stable id — avoids
                    // adding a new per-particle RNG-state field for a value
                    // that only needs to be stable, not independently random
                    // each frame.
                    let h = p.id.wrapping_mul(2654435761);
                    0.5 + ((h >> 8) & 0xFFFF) as f32 / 65535.0
                } else {
                    1.0
                };
                let k = dt * self.speed_mult * strength;
                p.vx += ax * k;
                p.vy += ay * k;
                if b.max_speed > 0.0 {
                    let speed2 = p.vx * p.vx + p.vy * p.vy;
                    if speed2 > b.max_speed * b.max_speed {
                        let s = b.max_speed / speed2.sqrt();
                        p.vx *= s;
                        p.vy *= s;
                    }
                }
            }

            // `remapvalue` targeting `velocity`/`speed`: runs here (before
            // this frame's velocity is used for next frame's position
            // integration), matching the original, real-content-validated
            // timing exactly. Fields targeting other outputs run later,
            // after alpha/size/color are computed (see below) so
            // `remapvalue` can override them.
            let lifetime_pos_pre = if p.max_life > 0.0 {
                ((p.max_life - p.life) / p.max_life).clamp(0.0, 1.0)
            } else {
                1.0
            };
            for r in &self.remap {
                match r.output {
                    RemapOutput::Velocity => {
                        let [vx, vy, _] =
                            r.apply_vector(p, self.time, lifetime_pos_pre, [p.vx, p.vy, 0.0], 2);
                        p.vx = vx;
                        p.vy = vy;
                    }
                    RemapOutput::Speed => {
                        let t = r.sample01(0.0, p, self.time, lifetime_pos_pre, p.max_life);
                        let target = r.out_min[0] + (r.out_max[0] - r.out_min[0]) * t;
                        let mag = (p.vx * p.vx + p.vy * p.vy).sqrt();
                        let sp = r.operation.combine(mag, target);
                        if mag > 1e-4 {
                            let k = sp / mag;
                            p.vx *= k;
                            p.vy *= k;
                        }
                    }
                    _ => {}
                }
            }

            // Position-oscillate integrates a velocity (derivative of
            // scale*cos(w*t+phase)) into position each frame, matching
            // CParticle.cpp's createOscillatePositionOperator exactly.
            if let Some(osc) = &p.osc_pos {
                let age = p.max_life - p.life;
                for axis in 0..2 {
                    let w = osc.frequency[axis];
                    let d = -osc.scale[axis] * w * (w * age + osc.phase[axis]).sin() * dt;
                    if axis == 0 {
                        p.x += d;
                    } else {
                        p.y += d;
                    }
                }
            }

            if let Some((idx, origin, scale, threshold)) = self.control_point_attract {
                if let Some(cp) = self.control_points.get(idx) {
                    let center = [cp[0] + origin[0], cp[1] + origin[1]];
                    let to_center = [center[0] - p.x, center[1] - p.y];
                    let distance =
                        (to_center[0] * to_center[0] + to_center[1] * to_center[1]).sqrt();
                    let radius = threshold / 2.0;
                    if distance > 0.001 && distance < radius {
                        let force = scale * dt * self.speed_mult;
                        p.vx += (to_center[0] / distance) * force;
                        p.vy += (to_center[1] / distance) * force;
                    }
                }
            }

            // `collisionsphere`/`collisionbox`/`collisionbounds`/
            // `collisionquad`/`collisionplane`: checked last among the
            // motion-affecting operators, against this frame's final
            // position, so a collider reacts to where the particle actually
            // ended up (matching how a physics response is conventionally
            // the last word on a frame's motion).
            for collider in &self.collisions {
                resolve_collision(p, collider);
            }

            p.life -= dt;

            // Reference order (CParticle.cpp createAngularMovementOperator):
            // integrate rotation from the *current* angular velocity first,
            // then accelerate angular velocity, then decay it by drag.
            p.rotation += p.angular_velocity * dt;
            p.angular_velocity += self.angular_force * dt;
            let drag_factor = (1.0 - self.angular_drag * dt).max(0.0);
            p.angular_velocity *= drag_factor;
            p.rotation = wrap_angle(p.rotation);

            let age = p.max_life - p.life;
            let lifetime_pos = if p.max_life > 0.0 {
                (age / p.max_life).clamp(0.0, 1.0)
            } else {
                1.0
            };

            // Sprite-sheet frame advance (CParticle.cpp's "Update animation
            // frames"): only meaningful once the caller has told us the
            // resolved sprite actually has more than one frame.
            if self.sprite_frame_count > 1 {
                let frame_count = self.sprite_frame_count as f32;
                match self.animation_mode {
                    AnimationMode::RandomFrame => {
                        if p.frame < 0.0 {
                            p.frame = (fastrand::f32() * frame_count)
                                .floor()
                                .min(frame_count - 1.0);
                        }
                    }
                    AnimationMode::Once => {
                        p.frame = (lifetime_pos * frame_count * self.sequence_multiplier)
                            .min(frame_count - 1.0);
                    }
                    AnimationMode::Loop => {
                        p.frame = if self.sprite_duration > 0.0 {
                            let time_in_cycle =
                                (age * self.sequence_multiplier) % self.sprite_duration;
                            let cycle_pos = time_in_cycle / self.sprite_duration;
                            (cycle_pos * frame_count) % frame_count
                        } else {
                            (lifetime_pos * frame_count * self.sequence_multiplier) % frame_count
                        };
                    }
                }
            }

            // Alpha: alphafade takes precedence (its very common fade-in/hold/
            // fade-out shape), then a generic alphachange ramp, else fall back
            // to a plain linear death-fade so particles don't hard-cut when a
            // config has no alpha-affecting operator at all.
            let mut alpha = if let Some((fade_in, fade_out)) = self.alphafade {
                let fade = if lifetime_pos <= fade_in {
                    fade_value(lifetime_pos, 0.0, fade_in, 0.0, 1.0)
                } else if lifetime_pos > fade_out {
                    1.0 - fade_value(lifetime_pos, fade_out, 1.0, 0.0, 1.0)
                } else {
                    1.0
                };
                p.initial_alpha * fade
            } else if let Some((start, end, sv, ev)) = self.alphachange {
                p.initial_alpha * fade_value(lifetime_pos, start, end, sv, ev)
            } else {
                p.initial_alpha * (p.life / p.max_life).max(0.0)
            };

            let mut size = if let Some((start, end, sv, ev)) = self.sizechange {
                p.initial_size * fade_value(lifetime_pos, start, end, sv, ev)
            } else {
                p.size
            };

            if let Some(osc) = &p.osc_alpha {
                let cos_val = ((osc.frequency * age + osc.phase).cos() + 1.0) * 0.5;
                let mult = self
                    .oscillate_alpha
                    .map(|o| o.scale_min + (o.scale_max - o.scale_min) * cos_val)
                    .unwrap_or(1.0);
                alpha *= mult;
            }
            if let Some(osc) = &p.osc_size {
                let cos_val = ((osc.frequency * age + osc.phase).cos() + 1.0) * 0.5;
                let mult = self
                    .oscillate_size
                    .map(|o| o.scale_min + (o.scale_max - o.scale_min) * cos_val)
                    .unwrap_or(1.0);
                size *= mult;
            }

            if let Some((start, end, start_color, end_color)) = self.colorchange {
                let mult = [
                    fade_value(lifetime_pos, start, end, start_color[0], end_color[0]),
                    fade_value(lifetime_pos, start, end, start_color[1], end_color[1]),
                    fade_value(lifetime_pos, start, end, start_color[2], end_color[2]),
                ];
                p.color = [
                    (p.initial_color[0] * mult[0] * 255.0).clamp(0.0, 255.0) as u8,
                    (p.initial_color[1] * mult[1] * 255.0).clamp(0.0, 255.0) as u8,
                    (p.initial_color[2] * mult[2] * 255.0).clamp(0.0, 255.0) as u8,
                ];
            }

            // `remapvalue` targeting anything other than velocity/speed
            // (handled above, before position integration needs it): runs
            // last, after alphafade/sizechange/colorchange/oscillate, so an
            // author who explicitly attaches remapvalue to one of these
            // fields gets it as the final, overriding value.
            for r in &self.remap {
                match r.output {
                    RemapOutput::Velocity | RemapOutput::Speed => {}
                    RemapOutput::Opacity => {
                        let t = r.sample01(0.0, p, self.time, lifetime_pos, p.max_life);
                        let remapped = r.out_min[0] + (r.out_max[0] - r.out_min[0]) * t;
                        alpha = r.operation.combine(alpha, remapped);
                    }
                    RemapOutput::Size => {
                        let t = r.sample01(0.0, p, self.time, lifetime_pos, p.max_life);
                        let remapped = r.out_min[0] + (r.out_max[0] - r.out_min[0]) * t;
                        size = r.operation.combine(size, remapped);
                    }
                    RemapOutput::Rotation => {
                        let t = r.sample01(0.0, p, self.time, lifetime_pos, p.max_life);
                        let remapped = r.out_min[0] + (r.out_max[0] - r.out_min[0]) * t;
                        p.rotation = r.operation.combine(p.rotation, remapped);
                    }
                    RemapOutput::AngularSpeed => {
                        let t = r.sample01(0.0, p, self.time, lifetime_pos, p.max_life);
                        let remapped = r.out_min[0] + (r.out_max[0] - r.out_min[0]) * t;
                        p.angular_velocity = r.operation.combine(p.angular_velocity, remapped);
                    }
                    RemapOutput::Color => {
                        let current = [
                            p.color[0] as f32 / 255.0,
                            p.color[1] as f32 / 255.0,
                            p.color[2] as f32 / 255.0,
                        ];
                        let out = r.apply_vector(p, self.time, lifetime_pos, current, 3);
                        p.color = [
                            (out[0] * 255.0).clamp(0.0, 255.0) as u8,
                            (out[1] * 255.0).clamp(0.0, 255.0) as u8,
                            (out[2] * 255.0).clamp(0.0, 255.0) as u8,
                        ];
                    }
                    RemapOutput::Position => {
                        let [x, y, _] =
                            r.apply_vector(p, self.time, lifetime_pos, [p.x, p.y, 0.0], 2);
                        p.x = x;
                        p.y = y;
                    }
                }
            }

            // `self.alpha_mult` (the instance-override multiplier) is already
            // baked into `p.initial_alpha` at spawn, matching the
            // reference's `p.alpha = random(min,max) * override` — applying
            // it again here would square it whenever an override isn't 1.0.
            p.alpha = alpha;
            p.size = size;
        }

        // Child-system event capture: deaths before the retain, spawns as
        // they're pushed below (only bothering when children are attached).
        let mut died: Vec<(u64, [f32; 2])> = Vec::new();
        let mut spawned: Vec<(u64, [f32; 2])> = Vec::new();
        if !self.children.is_empty() {
            died.extend(
                self.particles
                    .iter()
                    .filter(|p| p.life <= 0.0)
                    .map(|p| (p.id, [p.x, p.y])),
            );
        }
        self.particles.retain(|p| p.life > 0.0);

        // Hold off emission until `starttime` — existing particles still
        // simulate and die, the burst just ramps in from zero at start_time.
        let emitting = self.time >= self.start_time;
        let since_start = self.time - self.start_time;
        for emitter in &mut self.emitters {
            // Per-emitter `duration` cutoff (after start_time).
            let expired = emitter.duration.is_some_and(|d| since_start > d);
            if !emitting || expired {
                continue;
            }
            // One-shot burst: seed the accumulator with the full count once, so
            // the spawn loop below emits it in this frame (capped by max_count).
            if !emitter.burst_done {
                emitter.accumulator += emitter.burst as f32;
                emitter.burst_done = true;
            }
            let audio_gain = emitter.audio.map_or(1.0, |a| a.gain(&self.audio_bands));
            emitter.accumulator += emitter.rate * self.rate_mult * audio_gain * dt;
            while emitter.accumulator >= 1.0 && self.particles.len() < self.max_count {
                emitter.accumulator -= 1.0;
                // `createLifetimeRandomInitializer`: `p.lifetime =
                // random(min, max) * lifetimeOverride`. `exponent` biases the
                // uniform sample toward one end before the lerp.
                let life = (self.life_min
                    + fastrand::f32().powf(self.life_exp)
                        * (self.life_max - self.life_min).max(0.0))
                    * self.lifetime_mult;
                // Reference halves the sizerandom range (CParticle.cpp: `size = (min +
                // t*(max-min)) * override / 2.0`) — the initializer's value is a
                // diameter, not a radius.
                let size = (self.size_min
                    + fastrand::f32().powf(self.size_exp)
                        * (self.size_max - self.size_min).max(0.0))
                    * self.size_mult
                    / 2.0;

                let ox = emitter.origin[0];
                let oy = emitter.origin[1];

                // Spawn offset — CParticle.cpp's create{Box,Sphere}Emitter:
                // a random point WITHIN the emitter volume, scaled per-axis
                // by `directions` (the emitter never encodes velocity in
                // its distances; rainperspective's 1024 radius is a spawn
                // band, which our old model collapsed into a ±15px column
                // at the emitter origin while misreading the radius as a
                // fallback speed).
                let mut offset = if emitter.is_sphere {
                    let (rmin, rmax) = (emitter.distance_min[0], emitter.distance_max[0]);
                    if self.shell_3d {
                        // flags & 4 (perspective): 3D spherical shell,
                        // cbrt-uniform radius; our 2D sim keeps xy.
                        let theta = fastrand::f32() * std::f32::consts::TAU;
                        let cos_t = fastrand::f32() * 2.0 - 1.0;
                        let sin_t = (1.0 - cos_t * cos_t).sqrt();
                        let r =
                            (rmin.powi(3) + fastrand::f32() * (rmax.powi(3) - rmin.powi(3))).cbrt();
                        [sin_t * theta.cos() * r, sin_t * theta.sin() * r]
                    } else {
                        // 2D disk/annulus, sqrt-uniform for even area density.
                        let angle = fastrand::f32() * std::f32::consts::TAU;
                        let r2 = rmin * rmin + fastrand::f32() * (rmax * rmax - rmin * rmin);
                        let r = r2.max(0.0).sqrt();
                        [angle.cos() * r, angle.sin() * r]
                    }
                } else {
                    // Box: per-axis distance in [min, max], random sign flip.
                    let mut o = [0.0f32; 2];
                    for (axis, out) in o.iter_mut().enumerate() {
                        let d = emitter.distance_min[axis]
                            + fastrand::f32()
                                * (emitter.distance_max[axis] - emitter.distance_min[axis]);
                        *out = if fastrand::f32() < 0.5 { -d } else { d };
                    }
                    o
                };
                offset[0] *= emitter.directions[0];
                offset[1] *= emitter.directions[1];
                // `sign`: 1 forces positive, -1 negative, 0 leaves both.
                for axis in 0..2 {
                    if emitter.sign[axis] >= 1.0 {
                        offset[axis] = offset[axis].abs();
                    } else if emitter.sign[axis] <= -1.0 {
                        offset[axis] = -offset[axis].abs();
                    }
                }
                let mut spawn_x = ox + offset[0];
                let mut spawn_y = oy + offset[1];

                // Velocity: velocityrandom initializer first; else the
                // emitter's own optional radial speed (`speedmin`/`speedmax`
                // fields, pointing outward from the offset); else zero —
                // "Emitter does not set velocity - initializers handle that".
                let (mut vx, mut vy) =
                    if let (Some(min), Some(max)) = (self.velocity_min, self.velocity_max) {
                        (
                            (min[0] + fastrand::f32() * (max[0] - min[0])) * self.speed_mult,
                            (min[1] + fastrand::f32() * (max[1] - min[1])) * self.speed_mult,
                        )
                    } else if emitter.speed_max > 0.0 || emitter.speed_min != 0.0 {
                        let len = (offset[0] * offset[0] + offset[1] * offset[1]).sqrt();
                        let dir = if len > 0.0 {
                            [offset[0] / len, offset[1] / len]
                        } else {
                            [0.0, 1.0]
                        };
                        let speed = (emitter.speed_min
                            + fastrand::f32() * (emitter.speed_max - emitter.speed_min))
                            * self.speed_mult;
                        (dir[0] * speed, dir[1] * speed)
                    } else {
                        (0.0, 0.0)
                    };

                // `mapsequencebetweencontrolpoints`: the Nth spawn sits at
                // the Nth slot along the cp0->cp1 segment (mirror =
                // ping-pong); velocityrandom still jitters it afterwards,
                // which is what shapes the discharge preset's lightning.
                if let Some(ms) = self.map_seq_between {
                    let last = ms.count - 1;
                    let idx = self.map_seq_between_index;
                    self.map_seq_between_index = self.map_seq_between_index.wrapping_add(1);
                    let slot = if ms.mirror {
                        let period = 2 * last;
                        let k = idx % period.max(1);
                        if k <= last {
                            k
                        } else {
                            period - k
                        }
                    } else {
                        idx % ms.count
                    };
                    let t = slot as f32 / last as f32;
                    if let (Some(a), Some(b)) =
                        (self.control_points.first(), self.control_points.get(1))
                    {
                        spawn_x = a[0] + (b[0] - a[0]) * t;
                        spawn_y = a[1] + (b[1] - a[1]) * t;
                    }
                }

                // `mapsequencearoundcontrolpoint` (CParticle.cpp): spawn at
                // the control point, launching successive particles at
                // evenly-spaced angles around a circle (the initializer
                // *assigns* position/velocity, unlike the additive ones).
                if let Some(ms) = self.map_sequence {
                    let angle =
                        (self.map_sequence_index as f32 / ms.count as f32) * std::f32::consts::TAU;
                    self.map_sequence_index = (self.map_sequence_index + 1) % ms.count;
                    if let Some(cp) = self.control_points.get(ms.control_point) {
                        spawn_x = cp[0];
                        spawn_y = cp[1];
                    }
                    let sx = ms.speed_min[0]
                        + fastrand::f32() * (ms.speed_max[0] - ms.speed_min[0]).max(0.0);
                    let sy = ms.speed_min[1]
                        + fastrand::f32() * (ms.speed_max[1] - ms.speed_min[1]).max(0.0);
                    let (sin, cos) = angle.sin_cos();
                    // The reference flips y, then rotates by
                    // (cos*x + sin*y, -sin*x + cos*y) (column-major
                    // glm::mat3); with the flip folded in that is the
                    // conjugate rotation followed by a y flip.
                    vx = (cos * sx - sin * sy) * self.speed_mult;
                    vy = -(sin * sx + cos * sy) * self.speed_mult;
                }

                // `turbulentvelocityrandom` (CParticle.cpp): curl-noise
                // spawn velocity, angle-limited around `forward`, added on
                // top of any plain `velocityrandom` contribution.
                for &t in &self.turbulent_velocity {
                    let forward = vec3_normalize_or(t.forward, [0.0, -1.0, 0.0]);
                    let right = vec3_normalize_or(t.right, [1.0, 0.0, 0.0]);
                    let speed =
                        t.speed_min + fastrand::f32() * (t.speed_max - t.speed_min).max(0.0);
                    let phase =
                        t.phase_min + fastrand::f32() * (t.phase_max - t.phase_min).max(0.0);
                    let time_shift = self.time * t.timescale;
                    let sample = [
                        spawn_x * 0.1 + time_shift + phase,
                        spawn_y * 0.1 + time_shift + phase * 0.7,
                        time_shift + phase * 1.3,
                    ];
                    let curl = crate::engine::sim::noise::curl_noise(sample);
                    let mut dir = vec3_normalize_or(curl, forward);

                    // `scale` < 2 limits how far the direction may deviate
                    // from `forward` (normalized angle, max = scale/2); 0
                    // means straight along `forward`.
                    if t.scale <= 0.0 {
                        dir = forward;
                    } else if t.scale < 2.0 {
                        let cos_angle = vec3_dot(dir, forward).clamp(-1.0, 1.0);
                        let angle = cos_angle.acos() / std::f32::consts::PI;
                        let max_angle = t.scale / 2.0;
                        if angle > max_angle {
                            let axis = vec3_cross(dir, forward);
                            if vec3_length(axis) > 0.0001 {
                                let axis = vec3_normalize_or(axis, [0.0, 0.0, 1.0]);
                                let rot = (angle - max_angle) * std::f32::consts::PI;
                                dir = vec3_rotate_axis(dir, axis, rot);
                            }
                        }
                    }

                    // `offset` tilts the result around `right`.
                    if t.offset.abs() > 0.0001 {
                        dir = vec3_rotate_axis(dir, right, -t.offset);
                    }

                    // 2D particles: project onto the XY plane and
                    // renormalize (the reference's `flags & 4 == 0` branch —
                    // our renderer is always orthographic/2D).
                    dir[2] = 0.0;
                    let dir = vec3_normalize_or(dir, forward);

                    vx += dir[0] * speed * self.speed_mult;
                    vy += dir[1] * speed * self.speed_mult;
                }

                // `rotationrandom`: uniform in [min.z, max.z], scaled by the
                // instance speed override (the reference multiplies rotation
                // by `speedOverride` too, odd as that reads).
                let spawn_rotation = self
                    .rotation_range
                    .map(|(min, max)| {
                        (min + fastrand::f32() * (max - min).max(0.0)) * self.speed_mult
                    })
                    .unwrap_or(0.0);

                // `createColorRandomInitializer`: `p.color = randomVec3(min,
                // max) * colorOverride` — colorn is a 0-1 per-channel tint.
                let color = self.color_override.unwrap_or_else(|| {
                    [
                        (lerp_u8(self.color_min[0], self.color_max[0]) as f32 * self.colorn_mult[0])
                            .clamp(0.0, 255.0) as u8,
                        (lerp_u8(self.color_min[1], self.color_max[1]) as f32 * self.colorn_mult[1])
                            .clamp(0.0, 255.0) as u8,
                        (lerp_u8(self.color_min[2], self.color_max[2]) as f32 * self.colorn_mult[2])
                            .clamp(0.0, 255.0) as u8,
                    ]
                });

                // `alpharandom`: CParticle.cpp's `p.alpha = random(min, max) *
                // override` — a plain multiply (unlike sizerandom, not halved).
                let spawn_alpha = (self.alpha_min
                    + fastrand::f32() * (self.alpha_max - self.alpha_min).max(0.0))
                    * self.alpha_mult;

                let particle_id = self.next_particle_id;
                self.next_particle_id += 1;
                if !self.children.is_empty() {
                    spawned.push((particle_id, [spawn_x, spawn_y]));
                }
                self.particles.push(Particle {
                    id: particle_id,
                    x: spawn_x,
                    y: spawn_y,
                    vx,
                    vy,
                    life,
                    max_life: life,
                    size,
                    alpha: spawn_alpha,
                    color,
                    initial_alpha: spawn_alpha,
                    initial_size: size,
                    initial_color: [
                        color[0] as f32 / 255.0,
                        color[1] as f32 / 255.0,
                        color[2] as f32 / 255.0,
                    ],
                    osc_alpha: self.oscillate_alpha.map(|o| o.sample()),
                    osc_size: self.oscillate_size.map(|o| o.sample()),
                    osc_pos: self.oscillate_position.map(|o| {
                        let a = o.sample();
                        let b = o.sample();
                        OscillatorPos {
                            frequency: [a.frequency, b.frequency],
                            scale: [a.scale, b.scale],
                            phase: [a.phase, b.phase],
                        }
                    }),
                    rotation: spawn_rotation,
                    angular_velocity: self.angular_velocity_min
                        + fastrand::f32() * (self.angular_velocity_max - self.angular_velocity_min),
                    frame: -1.0,
                });
            }
        }

        if !self.children.is_empty() {
            self.step_children(dt, &spawned, &died);
        }
    }

    /// Attaches a resolved child preset (see [`ChildRef`]). `static`
    /// children get their single persistent instance immediately, at the
    /// same spawn center the parent was built with.
    pub fn add_child(
        &mut self,
        config: ParticleConfig,
        sprite: Option<ParticleSprite>,
        additive: bool,
        child_ref: &ChildRef,
        spawn_center: [f32; 2],
    ) {
        let child_type = match child_ref.child_type.as_deref() {
            Some("eventfollow") => ChildType::Follow,
            Some("eventspawn") => ChildType::Spawn,
            Some("eventdeath") => ChildType::Death,
            // Absent or "static": a continuously-running sub-system.
            _ => ChildType::Static,
        };
        let mut child = ChildSystem {
            child_type,
            probability: child_ref.probability.unwrap_or(1.0) as f32,
            max_instances: child_ref.maxcount.unwrap_or(16).max(1) as usize,
            config,
            sprite,
            additive,
            instances: Vec::new(),
        };
        if child_type == ChildType::Static {
            let mut system = ParticleSystem::from_config(&child.config, spawn_center, None);
            if let Some(s) = &child.sprite {
                system.set_sprite_frames(s.frames.len(), s.duration);
            }
            child.instances.push(ChildInstance {
                parent_id: None,
                last_center: spawn_center,
                emit_until: None,
                persistent: true,
                system,
            });
        }
        self.children.push(child);
    }

    /// Spawns/updates/retires child instances for this frame's parent
    /// spawn/death events. No counterpart in the C++ reference (it ignores
    /// `children` entirely); semantics derived from real preset content.
    pub(super) fn step_children(
        &mut self,
        dt: f32,
        spawned: &[(u64, [f32; 2])],
        died: &[(u64, [f32; 2])],
    ) {
        // Burst instances stop emitting after this long — the child preset
        // itself has no end marker, and a one-shot puff is the authored
        // intent of eventspawn/eventdeath (vs. a permanent stream).
        const BURST_EMIT_SECS: f32 = 0.1;

        let mut children = std::mem::take(&mut self.children);
        for child in &mut children {
            let events: &[(u64, [f32; 2])] = match child.child_type {
                ChildType::Static => &[],
                ChildType::Follow | ChildType::Spawn => spawned,
                ChildType::Death => died,
            };
            for &(pid, pos) in events {
                if child.instances.len() >= child.max_instances {
                    break;
                }
                if fastrand::f32() > child.probability {
                    continue;
                }
                let mut system = ParticleSystem::from_config(&child.config, pos, None);
                if let Some(s) = &child.sprite {
                    system.set_sprite_frames(s.frames.len(), s.duration);
                }
                let follows = child.child_type == ChildType::Follow;
                child.instances.push(ChildInstance {
                    parent_id: follows.then_some(pid),
                    last_center: pos,
                    emit_until: (!follows).then_some(BURST_EMIT_SECS),
                    persistent: false,
                    system,
                });
            }

            for inst in &mut child.instances {
                if let Some(pid) = inst.parent_id {
                    if let Some(p) = self.particles.iter().find(|p| p.id == pid) {
                        let delta = [p.x - inst.last_center[0], p.y - inst.last_center[1]];
                        if delta != [0.0, 0.0] {
                            inst.system.shift_origin(delta);
                            inst.last_center = [p.x, p.y];
                        }
                    } else {
                        // Parent died: stop emitting, let the remainder live out.
                        inst.parent_id = None;
                        inst.system.stop_emitting();
                    }
                }
                if let Some(cutoff) = inst.emit_until {
                    if inst.system.time >= cutoff {
                        inst.system.stop_emitting();
                    }
                }
                inst.system.step(dt);
            }

            child.instances.retain(|inst| {
                inst.persistent || inst.parent_id.is_some() || !inst.system.particles.is_empty()
            });
        }
        self.children = children;
    }

    /// Moves every emitter origin and resolved control point by `delta` —
    /// how an `eventfollow` child tracks its parent particle.
    pub(super) fn shift_origin(&mut self, delta: [f32; 2]) {
        for emitter in &mut self.emitters {
            emitter.origin[0] += delta[0];
            emitter.origin[1] += delta[1];
        }
        for cp in &mut self.control_points {
            cp[0] += delta[0];
            cp[1] += delta[1];
        }
    }

    pub(super) fn stop_emitting(&mut self) {
        for emitter in &mut self.emitters {
            emitter.rate = 0.0;
        }
    }
}
