//! Operator parameters parsed from the config (oscillation, turbulence,
//! vortex, collision, boids, remap) and the vector math the operators use.

use super::*;

/// Parsed `oscillatealpha`/`oscillatesize`/`oscillateposition` operator
/// config — random ranges each spawned particle draws its own values from.
#[derive(Clone, Copy)]
pub(super) struct OscillateParams {
    pub(super) freq_min: f32,
    pub(super) freq_max: f32,
    pub(super) scale_min: f32,
    pub(super) scale_max: f32,
    pub(super) phase_min: f32,
    pub(super) phase_max: f32,
}

impl OscillateParams {
    pub(super) fn from_operator(op: &Operator) -> Self {
        let freq_min = op.frequencymin.unwrap_or(0.0) as f32;
        let freq_max = op.frequencymax.unwrap_or(freq_min as f64) as f32;
        let scale_min = op.scalemin.or(op.scalemax).unwrap_or(0.0) as f32;
        let scale_max = op.scalemax.or(op.scalemin).unwrap_or(0.0) as f32;
        let phase_min = op.phasemin.unwrap_or(0.0) as f32;
        let phase_max = op.phasemax.unwrap_or(0.0) as f32;
        Self {
            freq_min,
            freq_max,
            scale_min,
            scale_max,
            phase_min,
            phase_max,
        }
    }

    /// CParticle.cpp always draws phase from `[phaseMin, phaseMax + 2*PI]` —
    /// even with no phase fields at all (both 0), this yields a fully random
    /// phase in `[0, 2*PI]`, which is what real presets (lacking phase
    /// fields entirely) actually rely on.
    pub(super) fn sample(&self) -> Oscillator1D {
        Oscillator1D {
            frequency: self.freq_min + fastrand::f32() * (self.freq_max - self.freq_min),
            scale: self.scale_min + fastrand::f32() * (self.scale_max - self.scale_min),
            phase: self.phase_min
                + fastrand::f32() * (self.phase_max + std::f32::consts::TAU - self.phase_min),
        }
    }
}

/// `turbulentvelocityrandom` (CParticle.cpp): spawn velocity picked from a
/// curl-noise field, angle-limited around `forward`.
#[derive(Clone, Copy)]
pub(super) struct TurbulentVelocityParams {
    pub(super) speed_min: f32,
    pub(super) speed_max: f32,
    /// Tilt angle (radians) around `right`.
    pub(super) offset: f32,
    /// Direction cone: `< 2` limits deviation from `forward` to
    /// `scale/2 * pi`.
    pub(super) scale: f32,
    pub(super) forward: [f32; 3],
    pub(super) right: [f32; 3],
    pub(super) timescale: f32,
    pub(super) phase_min: f32,
    pub(super) phase_max: f32,
}

/// `turbulence` operator (CParticle.cpp): curl-noise acceleration. `phase`
/// and `speed` are drawn once per operator instance, not per particle.
#[derive(Clone, Copy)]
pub(super) struct TurbulenceParams {
    pub(super) noise_scale: f32,
    pub(super) timescale: f32,
    pub(super) mask: [f32; 3],
    pub(super) phase: f32,
    pub(super) speed: f32,
}

/// `spritetrail` renderer: velocity-aligned, velocity-stretched sprite quad.
#[derive(Clone, Copy)]
pub(super) struct SpriteTrailParams {
    pub(super) length: f32,
    pub(super) max_length: f32,
    pub(super) min_length: f32,
}

/// `mapsequencebetweencontrolpoints` initializer: spawns successive
/// particles at evenly spaced points along the segment from control point
/// 0 to control point 1 (`count` slots). `limitbehavior: "mirror"`
/// ping-pongs the sequence; anything else wraps. Neither the C++ reference
/// nor its parser implements this one — semantics derived from real
/// content (the `discharge` preset draws a rope through the chain,
/// yielding a lightning arc between the two anchors).
#[derive(Clone, Copy)]
pub(super) struct MapSequenceBetweenParams {
    pub(super) count: u32,
    pub(super) mirror: bool,
}

/// `mapsequencearoundcontrolpoint` initializer (CParticle.cpp): spawns each
/// particle at the control point with a velocity rotated by the next angle
/// in an evenly-divided circle (`sequence / count * 2pi`).
#[derive(Clone, Copy)]
pub(super) struct MapSequenceParams {
    pub(super) control_point: usize,
    pub(super) count: u32,
    pub(super) speed_min: [f32; 3],
    pub(super) speed_max: [f32; 3],
}

/// `vortex` operator (CParticle.cpp): tangential spin around a control
/// point, with optional ring shape and center attraction.
#[derive(Clone, Copy)]
pub(super) struct VortexParams {
    pub(super) control_point: i64,
    pub(super) infinite_axis: bool,
    pub(super) maintain_distance: bool,
    pub(super) ring_shape: bool,
    pub(super) axis: [f32; 3],
    pub(super) offset: [f32; 3],
    pub(super) distance_inner: f32,
    pub(super) distance_outer: f32,
    pub(super) speed_inner: f32,
    pub(super) speed_outer: f32,
    pub(super) center_force: f32,
    pub(super) ring_radius: f32,
    pub(super) ring_width: f32,
    pub(super) ring_pull_distance: f32,
    pub(super) ring_pull_force: f32,
}

/// Collider shape for a `collisionsphere`/`collisionbox`/`collisionbounds`/
/// `collisionquad`/`collisionplane` operator (see the `Operator` field docs
/// for the "no C++ reference, reasoned reconstruction" caveat). `Box` also
/// covers `collisionbounds`/`collisionquad` — this 2D engine has no distinct
/// notion of "world bounds" vs. "an oriented flat quad" vs. "a box", so all
/// three collapse to the same axis-aligned-rectangle test.
#[derive(Clone, Copy)]
pub(super) enum CollisionShape {
    Sphere {
        radius: f32,
    },
    Box {
        half_extent: [f32; 2],
    },
    /// `normal` points away from the solid side (into free space).
    Plane {
        normal: [f32; 2],
    },
}

/// `collisionbehavior`: what happens to a particle that penetrates a
/// collider this frame.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum CollisionBehavior {
    Bounce,
    Slide,
    Delete,
}

impl CollisionBehavior {
    pub(super) fn parse(s: Option<&str>) -> Self {
        match s {
            Some("slide") => Self::Slide,
            Some("delete") => Self::Delete,
            _ => Self::Bounce,
        }
    }
}

/// One resolved collider: anchored at a `controlpoint` (+ `offset`), same
/// static-at-construction simplification `vortex`/`controlpointattract`
/// already accept for control points.
#[derive(Clone, Copy)]
pub(super) struct CollisionParams {
    pub(super) shape: CollisionShape,
    pub(super) center: [f32; 2],
    pub(super) behavior: CollisionBehavior,
    pub(super) bounce_factor: f32,
}

/// `boids` operator: standard separation/alignment/cohesion flocking rules.
/// Real field names recovered from the binary (see the `Operator` field
/// docs); the flocking algorithm itself is the industry-standard
/// three-rule formulation, not independently verified against WE's own
/// (unrecoverable) per-pixel/per-particle GPU dispatch.
#[derive(Clone, Copy)]
pub(super) struct BoidsParams {
    pub(super) separation_threshold: f32,
    pub(super) neighbor_threshold: f32,
    pub(super) max_speed: f32,
    pub(super) separation_factor: f32,
    pub(super) alignment_factor: f32,
    pub(super) cohesion_factor: f32,
    /// `variablestrength != 0`: jitter each particle's flocking-force
    /// magnitude by a deterministic per-particle hash of its id, so a flock
    /// doesn't move as one mechanically uniform block.
    pub(super) variable_strength: bool,
}

/// `remapvalue`'s `input`/`output` field selector. Real, named fields
/// recovered directly from the retail binary's own dispatch table (Ghidra
/// report Follow-up (x)); only the subset this engine already tracks
/// per-particle state for is implemented. The rest
/// (`distancetocontrolpoint`/`positionbetweentwocontrolpoints`/`runtime`/
/// `timeofday`/`particlesystemtime`/`layertime`/`controlpoint`/
/// `deltatocontrolpoint`/`directiontocontrolpoint`/`layerorigin`) are real
/// names too, but need control-point/scene-clock plumbing this operator
/// doesn't have access to and no cached real content (271 Workshop items
/// checked) exercises any of them — falling through to `None` (noise mode)
/// rather than silently misreading a wrong field.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum RemapField {
    /// No `input` configured: sample noise at the particle position/time —
    /// the original, real-content-validated behavior, unchanged.
    None,
    LifetimeFraction,
    MaxLifetime,
    Size,
    Opacity,
    Speed,
    Rotation,
    AngularSpeed,
    Color,
    Position,
    Velocity,
}

impl RemapField {
    pub(super) fn parse(s: Option<&str>) -> Self {
        match s {
            Some("lifetimefraction") => Self::LifetimeFraction,
            Some("maxlifetime") => Self::MaxLifetime,
            Some("size") => Self::Size,
            Some("opacity") => Self::Opacity,
            Some("speed") => Self::Speed,
            Some("rotation") => Self::Rotation,
            Some("angularspeed") => Self::AngularSpeed,
            Some("color") => Self::Color,
            Some("position") => Self::Position,
            Some("velocity") => Self::Velocity,
            _ => Self::None,
        }
    }
}

/// `remapvalue`'s `output` field — same real dispatch table as
/// [`RemapField`], restricted to fields this operator can write back to.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum RemapOutput {
    Velocity,
    Speed,
    Size,
    Opacity,
    Rotation,
    AngularSpeed,
    Color,
    Position,
}

impl RemapOutput {
    pub(super) fn parse(s: Option<&str>) -> Self {
        match s {
            Some("speed") => Self::Speed,
            Some("size") => Self::Size,
            Some("opacity") => Self::Opacity,
            Some("rotation") => Self::Rotation,
            Some("angularspeed") => Self::AngularSpeed,
            Some("color") => Self::Color,
            Some("position") => Self::Position,
            // "velocity" and anything unrecognized: existing default.
            _ => Self::Velocity,
        }
    }
}

/// `remapvalue`'s `operation` — how the remapped value combines with the
/// output field's current value. `Multiply` is corroborated directly: the
/// retail binary's own `alphafade`-family default-preset builders
/// (`FUN_1401bfbb0`/`FUN_1401bc4b0`, Follow-up (x)) construct exactly
/// `operation=multiply, input=lifetimefraction` to fade a field by a
/// lifetime-driven factor — the same shape as this engine's own
/// (independently, already-validated) `alphafade` — so "multiply the
/// current value" is a corroborated reading, not a guess. `Add`/`Subtract`
/// follow the same current-value-combine pattern; no real cached content
/// exercises them to cross-check against.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum RemapOp {
    /// Default: replace the output field outright (the original,
    /// real-content-validated `remapvalue` behavior).
    Remap,
    Multiply,
    Add,
    Subtract,
}

impl RemapOp {
    pub(super) fn parse(s: Option<&str>) -> Self {
        match s {
            Some("multiply") => Self::Multiply,
            Some("add") => Self::Add,
            Some("subtract") => Self::Subtract,
            _ => Self::Remap,
        }
    }

    pub(super) fn combine(self, current: f32, remapped: f32) -> f32 {
        match self {
            Self::Remap => remapped,
            Self::Multiply => current * remapped,
            Self::Add => current + remapped,
            Self::Subtract => current - remapped,
        }
    }
}

/// `remapvalue`'s `transformfunction` — shapes the normalized `[0,1]` input
/// before it's remapped into the output range. The retail binary gave up
/// only the *names* (Follow-up (x)), not formulas: `Sine`/`Square`/
/// `Triangle` are the standard one-period constructions for each name, an
/// honest best-effort reading rather than a verified port. `Saw` is a
/// linear ramp over a single non-wrapping `[0,1]` pass, which makes it
/// indistinguishable here from `None` — documented, not hidden.
/// `Simplex`/`Fbm` are unchanged from before: real, position/time-driven
/// noise sampling, validated against real content
/// (`particles/presets/rain_screen.json`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum RemapTransform {
    None,
    Sine,
    Square,
    Saw,
    Triangle,
    Simplex,
    Fbm,
}

impl RemapTransform {
    /// Parses an explicit, non-empty `transformfunction` value. Whether an
    /// *absent* key defaults to `Simplex` (noise generator) or `None`
    /// (passthrough) depends on whether `input` was also given — see the
    /// call site in `ParticleSystem::new` — so that's handled there, not
    /// here.
    pub(super) fn parse(s: &str) -> Self {
        match s {
            "none" => Self::None,
            "sine" => Self::Sine,
            "square" => Self::Square,
            "saw" => Self::Saw,
            "triangle" => Self::Triangle,
            "fbmnoise" => Self::Fbm,
            // "simplexnoise" and anything unrecognized.
            _ => Self::Simplex,
        }
    }

    pub(super) fn is_noise(self) -> bool {
        matches!(self, Self::Simplex | Self::Fbm)
    }

    /// Shape an already-clamped `[0,1]` value. Noise variants are handled
    /// separately (via position/time sampling, not this function).
    pub(super) fn shape(self, t: f32) -> f32 {
        match self {
            Self::None | Self::Saw | Self::Simplex | Self::Fbm => t,
            Self::Sine => 0.5 + 0.5 * (t * std::f32::consts::TAU).sin(),
            Self::Square => {
                if t < 0.5 {
                    0.0
                } else {
                    1.0
                }
            }
            Self::Triangle => 1.0 - (2.0 * t - 1.0).abs(),
        }
    }
}

/// `remapvalue`'s `inputcomponent`/`outputcomponent` — real, named
/// selectors recovered from the retail binary (Follow-up (x)).
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum RemapComponent {
    All,
    X,
    Y,
    Z,
    Sum,
    Average,
    Max,
    Min,
}

impl RemapComponent {
    pub(super) fn parse(s: Option<&str>) -> Self {
        match s {
            Some("x") => Self::X,
            Some("y") => Self::Y,
            Some("z") => Self::Z,
            Some("sum") => Self::Sum,
            Some("average") => Self::Average,
            Some("max") => Self::Max,
            Some("min") => Self::Min,
            _ => Self::All,
        }
    }

    /// Collapse a (2- or 3-axis) vector to one scalar, for reading a
    /// vector-valued input field. `All`/`Average` both average — a vector
    /// input must collapse to one scalar for the remap math regardless, and
    /// averaging is the least-arbitrary choice among these (stays in the
    /// field's native range, unlike `Sum`, which scales with axis count).
    pub(super) fn reduce(self, v: [f32; 3], axes: usize) -> f32 {
        let axes = axes.min(3);
        match self {
            Self::X => v[0],
            Self::Y => v[1],
            Self::Z => v[2],
            Self::Sum => v[..axes].iter().sum(),
            Self::Max => v[..axes].iter().cloned().fold(f32::MIN, f32::max),
            Self::Min => v[..axes].iter().cloned().fold(f32::MAX, f32::min),
            Self::All | Self::Average => v[..axes].iter().sum::<f32>() / axes as f32,
        }
    }
}

/// `remapvalue`: reads `input` (a named particle field, or noise sampled at
/// the particle position when unset), shapes it, remaps it into
/// `[out_min, out_max]`, and combines it into `output` via `operation`. See
/// [`RemapField`]/[`RemapOutput`]/[`RemapOp`]/[`RemapTransform`] for what's
/// grounded in the retail binary vs. an honest best-effort reading.
#[derive(Clone, Copy)]
pub(super) struct RemapParams {
    pub(super) input: RemapField,
    pub(super) input_component: RemapComponent,
    pub(super) output: RemapOutput,
    pub(super) output_component: RemapComponent,
    pub(super) operation: RemapOp,
    pub(super) transform: RemapTransform,
    pub(super) in_min: f32,
    pub(super) in_max: f32,
    pub(super) out_min: [f32; 3],
    pub(super) out_max: [f32; 3],
    pub(super) input_scale: f32,
}

impl RemapParams {
    /// Read the configured input field as a single scalar (vector fields
    /// collapse via `input_component`). Only meaningful when `input !=
    /// RemapField::None` — noise mode is handled entirely in `sample01`.
    pub(super) fn read_input(&self, p: &Particle, lifetime_pos: f32, max_life: f32) -> f32 {
        match self.input {
            RemapField::None => 0.0,
            RemapField::LifetimeFraction => lifetime_pos,
            RemapField::MaxLifetime => max_life,
            RemapField::Size => p.size,
            RemapField::Opacity => p.alpha,
            RemapField::Speed => (p.vx * p.vx + p.vy * p.vy).sqrt(),
            RemapField::Rotation => p.rotation,
            RemapField::AngularSpeed => p.angular_velocity,
            RemapField::Color => self.input_component.reduce(
                [
                    p.color[0] as f32 / 255.0,
                    p.color[1] as f32 / 255.0,
                    p.color[2] as f32 / 255.0,
                ],
                3,
            ),
            RemapField::Position => self.input_component.reduce([p.x, p.y, 0.0], 2),
            RemapField::Velocity => self.input_component.reduce([p.vx, p.vy, 0.0], 2),
        }
    }

    /// The shaped `[0,1]` value to lerp into `[out_min, out_max]`.
    /// `axis_seed` only matters in noise mode (a different seed per output
    /// axis, exactly the original x/y-decorrelation this operator always
    /// used); field mode ignores it, which is what gives a scalar field's
    /// single value its "broadcast to every output axis" behavior.
    pub(super) fn sample01(
        &self,
        axis_seed: f64,
        p: &Particle,
        time: f32,
        lifetime_pos: f32,
        max_life: f32,
    ) -> f32 {
        if self.input == RemapField::None || self.transform.is_noise() {
            use crate::engine::sim::noise::fbm_noise;
            let s = self.input_scale;
            let (x, y, t) = ((p.x * s) as f64, (p.y * s) as f64, time as f64);
            let octaves = if self.transform == RemapTransform::Fbm {
                4
            } else {
                1
            };
            let n = fbm_noise(x + axis_seed, y, t, octaves);
            ((n * 0.5 + 0.5) as f32).clamp(0.0, 1.0) // [-1,1] → [0,1]
        } else {
            let raw = self.read_input(p, lifetime_pos, max_life);
            let range = (self.in_max - self.in_min).abs().max(1e-6);
            let normalized = ((raw - self.in_min) / range).clamp(0.0, 1.0);
            self.transform.shape(normalized)
        }
    }

    /// Apply this operator to a 2- or 3-axis vector field (`velocity`/
    /// `position`/`color`), honoring `output_component` (writes only that
    /// one axis, leaving the rest of `current` untouched) or every in-range
    /// axis when it's `All`. Distinct per-axis noise seeds match the
    /// original x/y-decorrelated noise sampling exactly; field-mode inputs
    /// naturally broadcast the same value to every written axis instead
    /// (`sample01` ignores the seed outside noise mode).
    pub(super) fn apply_vector(
        &self,
        p: &Particle,
        time: f32,
        lifetime_pos: f32,
        current: [f32; 3],
        axes: usize,
    ) -> [f32; 3] {
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let seeds = [0.0f64, 13.7, 27.4];
        let mut out = current;
        let apply_axis = |axis: usize, out: &mut [f32; 3]| {
            let t = self.sample01(seeds[axis], p, time, lifetime_pos, p.max_life);
            let remapped = lerp(self.out_min[axis], self.out_max[axis], t);
            out[axis] = self.operation.combine(current[axis], remapped);
        };
        match self.output_component {
            RemapComponent::X => apply_axis(0, &mut out),
            RemapComponent::Y if axes >= 2 => apply_axis(1, &mut out),
            RemapComponent::Z if axes >= 3 => apply_axis(2, &mut out),
            _ => {
                for axis in 0..axes {
                    apply_axis(axis, &mut out);
                }
            }
        }
        out
    }
}

/// `boids` operator: O(n^2) separation/alignment/cohesion, using each
/// particle's position/velocity as of the start of the frame (the caller
/// snapshots `self.particles` before any per-particle mutation happens this
/// step). Fine for the particle counts real presets use (`maxcount` is
/// typically in the hundreds, not thousands).
pub(super) fn compute_boids_accel(particles: &[Particle], b: &BoidsParams) -> Vec<[f32; 2]> {
    let n = particles.len();
    let mut accel = vec![[0.0f32; 2]; n];
    for i in 0..n {
        let pi = &particles[i];
        let mut separation = [0.0f32; 2];
        let mut sep_count: u32 = 0;
        let mut avg_vel = [0.0f32; 2];
        let mut avg_pos = [0.0f32; 2];
        let mut neighbor_count: u32 = 0;
        for (j, pj) in particles.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = [pi.x - pj.x, pi.y - pj.y];
            let dist = (d[0] * d[0] + d[1] * d[1]).sqrt();
            if dist > 0.0001 && dist < b.separation_threshold {
                let f = 1.0 / dist;
                separation[0] += d[0] * f;
                separation[1] += d[1] * f;
                sep_count += 1;
            }
            if dist < b.neighbor_threshold {
                avg_vel[0] += pj.vx;
                avg_vel[1] += pj.vy;
                avg_pos[0] += pj.x;
                avg_pos[1] += pj.y;
                neighbor_count += 1;
            }
        }
        let mut a = [0.0f32; 2];
        if sep_count > 0 {
            let n = sep_count as f32;
            a[0] += separation[0] / n * b.separation_factor;
            a[1] += separation[1] / n * b.separation_factor;
        }
        if neighbor_count > 0 {
            let n = neighbor_count as f32;
            a[0] += (avg_vel[0] / n - pi.vx) * b.alignment_factor;
            a[1] += (avg_vel[1] / n - pi.vy) * b.alignment_factor;
            a[0] += (avg_pos[0] / n - pi.x) * b.cohesion_factor;
            a[1] += (avg_pos[1] / n - pi.y) * b.cohesion_factor;
        }
        accel[i] = a;
    }
    accel
}

/// Apply one collider's contact response to a particle already moved this
/// frame — see `CollisionShape`/`CollisionBehavior` docs for the "no C++
/// reference" caveat on exact semantics.
pub(super) fn resolve_collision(p: &mut Particle, c: &CollisionParams) {
    // `normal` points from the collider's solid surface toward free space;
    // `penetration` is how far past the surface the particle currently is
    // (<= 0.0 means "not colliding", skipped by the caller's-eye-view check
    // below in each shape branch).
    let (normal, penetration): ([f32; 2], f32) = match c.shape {
        CollisionShape::Sphere { radius } => {
            let d = [p.x - c.center[0], p.y - c.center[1]];
            let dist = (d[0] * d[0] + d[1] * d[1]).sqrt();
            if dist >= radius || dist < 1e-5 {
                return;
            }
            ([d[0] / dist, d[1] / dist], radius - dist)
        }
        CollisionShape::Box { half_extent } => {
            let d = [p.x - c.center[0], p.y - c.center[1]];
            if d[0].abs() >= half_extent[0] || d[1].abs() >= half_extent[1] {
                return;
            }
            let px = half_extent[0] - d[0].abs();
            let py = half_extent[1] - d[1].abs();
            if px < py {
                ([d[0].signum(), 0.0], px)
            } else {
                ([0.0, d[1].signum()], py)
            }
        }
        CollisionShape::Plane { normal } => {
            let d = [p.x - c.center[0], p.y - c.center[1]];
            let signed = d[0] * normal[0] + d[1] * normal[1];
            if signed >= 0.0 {
                return;
            }
            (normal, -signed)
        }
    };

    match c.behavior {
        CollisionBehavior::Delete => {
            p.life = -1.0;
        }
        CollisionBehavior::Slide => {
            p.x += normal[0] * penetration;
            p.y += normal[1] * penetration;
            let vn = p.vx * normal[0] + p.vy * normal[1];
            if vn < 0.0 {
                p.vx -= normal[0] * vn;
                p.vy -= normal[1] * vn;
            }
        }
        CollisionBehavior::Bounce => {
            p.x += normal[0] * penetration;
            p.y += normal[1] * penetration;
            let vn = p.vx * normal[0] + p.vy * normal[1];
            if vn < 0.0 {
                let restitution = 1.0 + c.bounce_factor;
                p.vx -= normal[0] * vn * restitution;
                p.vy -= normal[1] * vn * restitution;
            }
        }
    }
}

/// Wraps an angle to `[-pi, pi]`, matching `createAngularMovementOperator`'s
/// per-axis wrap (CParticle.cpp).
pub(super) fn wrap_angle(a: f32) -> f32 {
    let mut a = a % std::f32::consts::TAU;
    if a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    } else if a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

pub(super) fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(super) fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(super) fn vec3_length(v: [f32; 3]) -> f32 {
    vec3_dot(v, v).sqrt()
}

/// Normalize, or return `fallback` for a (near-)zero vector — the
/// reference's recurring `length > 0.0001 ? normalize(v) : default` guard.
pub(super) fn vec3_normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = vec3_length(v);
    if len > 0.0001 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        fallback
    }
}

/// Rotate `v` by `angle` radians around unit-length `axis` (Rodrigues'
/// formula — equivalent to the reference's `glm::rotate` mat3 use).
pub(super) fn vec3_rotate_axis(v: [f32; 3], axis: [f32; 3], angle: f32) -> [f32; 3] {
    let (sin, cos) = angle.sin_cos();
    let cross = vec3_cross(axis, v);
    let dot = vec3_dot(axis, v) * (1.0 - cos);
    [
        v[0] * cos + cross[0] * sin + axis[0] * dot,
        v[1] * cos + cross[1] * sin + axis[1] * dot,
        v[2] * cos + cross[2] * sin + axis[2] * dot,
    ]
}
