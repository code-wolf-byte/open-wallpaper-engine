use image::RgbaImage;
use serde::Deserialize;

mod config;
pub use config::*;

mod operators;
pub use operators::*;

mod draw;
pub use draw::*;

/// Per-particle oscillator state for `oscillatealpha`/`oscillatesize`: a
/// random frequency/scale/phase drawn once at spawn (CParticle.cpp captures
/// these lazily on first use, but since we never re-use particle slots
/// across configs, drawing them at spawn is equivalent).
#[derive(Clone, Copy)]
struct Oscillator1D {
    frequency: f32,
    scale: f32,
    phase: f32,
}

/// Same idea as `Oscillator1D` but per-axis for `oscillateposition`, which
/// integrates a velocity (derivative of `scale*cos(w*t+phase)`) into position
/// each frame rather than overwriting an absolute value.
#[derive(Clone, Copy)]
struct OscillatorPos {
    frequency: [f32; 2],
    scale: [f32; 2],
    phase: [f32; 2],
}

#[derive(Clone)]
struct Particle {
    /// Stable identity across `retain` compaction — child `eventfollow`
    /// instances track their parent particle by this, not by index.
    id: u64,
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    life: f32,
    max_life: f32,
    size: f32,
    alpha: f32,
    color: [u8; 3],
    /// Alpha/size/color as set by the initializers at spawn — `alphafade`/
    /// `sizechange`/`colorchange` scale *this*, not the live (already-modified)
    /// value. `initial_color` is 0..1 (matching `colorchange`'s multiplier).
    initial_alpha: f32,
    initial_size: f32,
    initial_color: [f32; 3],
    osc_alpha: Option<Oscillator1D>,
    osc_size: Option<Oscillator1D>,
    osc_pos: Option<OscillatorPos>,
    /// Screen-space (Z-axis) rotation angle, radians. Only visually relevant
    /// once a sprite texture is drawn (`render_onto`'s textured path) — a
    /// flat-color circle is rotation-invariant.
    rotation: f32,
    angular_velocity: f32,
    /// Current sprite-sheet frame (fractional, truncated at draw time);
    /// `-1.0` at spawn means "not yet assigned" (only meaningful for
    /// `AnimationMode::RandomFrame`, which assigns it once and freezes it).
    frame: f32,
}

pub struct ParticleSystem {
    particles: Vec<Particle>,
    emitters: Vec<EmitterState>,
    max_count: usize,
    life_min: f32,
    life_max: f32,
    size_min: f32,
    size_max: f32,
    /// `exponent` distribution bias for lifetime/size random sampling (1.0 = uniform).
    life_exp: f32,
    size_exp: f32,
    /// `alpharandom` initializer range (CParticle.cpp's `AlphaRandomInitializer`):
    /// defaults to `(1.0, 1.0)` — a no-op — when the config has no such
    /// initializer at all, matching `ParticleInstance::alpha`'s `1.0f` default.
    alpha_min: f32,
    alpha_max: f32,
    velocity_min: Option<[f32; 3]>,
    velocity_max: Option<[f32; 3]>,
    color_min: [f32; 3],
    color_max: [f32; 3],
    gravity: [f32; 2],
    /// Scene-global constant acceleration (`general.gravity*` + `wind*`),
    /// added to every particle's velocity each step. Set post-construction.
    scene_force: [f32; 2],
    /// Latest 16-band spectrum (L/R averaged), set per frame; audio-reactive
    /// emitters scale their emission rate by it.
    audio_bands: [f32; 16],
    /// `movement.drag`: velocity decays by `1 - drag*dt` (clamped at 0)
    /// each step, matching the reference's reversal-preventing clamp.
    movement_drag: f32,
    /// `alphafade`: (fadeintime, fadeouttime), normalized-lifetime fractions.
    alphafade: Option<(f32, f32)>,
    /// `sizechange`/`alphachange`: (starttime, endtime, startvalue, endvalue).
    sizechange: Option<(f32, f32, f32, f32)>,
    alphachange: Option<(f32, f32, f32, f32)>,
    /// `colorchange`: (starttime, endtime, start_color, end_color), colors in
    /// 0..1 (unlike `colorrandom`'s 0..255 — matches the real presets, e.g.
    /// `"startvalue": "1 0.75 0"`).
    colorchange: Option<(f32, f32, [f32; 3], [f32; 3])>,
    oscillate_alpha: Option<OscillateParams>,
    oscillate_size: Option<OscillateParams>,
    oscillate_position: Option<OscillateParams>,
    alpha_mult: f32,
    size_mult: f32,
    speed_mult: f32,
    rate_mult: f32,
    /// `instanceoverride.lifetime` multiplier on spawned lifetimes.
    lifetime_mult: f32,
    /// `instanceoverride.colorn` per-channel multiplier (0-1) on the
    /// `colorrandom` result.
    colorn_mult: [f32; 3],
    color_override: Option<[u8; 3]>,
    /// Resolved once at construction: `spawn_center + offset` per control
    /// point, indexed positionally (falls back to declaration order when a
    /// config omits `id`). Nested parent-space transforms and mouse-linked
    /// (`locktopointer`)/world-space (`flags & 2`) control points are not
    /// modeled — same simplification already accepted for emitter origins.
    control_points: Vec<[f32; 3]>,
    /// `controlpointattract`: (control point index, origin offset, scale, threshold).
    control_point_attract: Option<(usize, [f32; 3], f32, f32)>,
    /// True when `config.renderer` names a `"rope"`/`"ropetrail"` renderer —
    /// draws a connected ribbon through living particles instead of
    /// independent circles (CParticle::renderRope).
    rope_mode: bool,
    rope_subdivision: usize,
    /// Rope UV controls (`uvscale`/`uvscrolling`/`uvsmoothing` renderer keys).
    rope_uv_scale: f32,
    rope_uv_scrolling: bool,
    rope_uv_smoothing: bool,
    /// `angularvelocityrandom` initializer range (z-axis component only).
    angular_velocity_min: f32,
    angular_velocity_max: f32,
    /// `angularmovement` operator (z-axis `force` component, plus `drag`).
    angular_force: f32,
    angular_drag: f32,
    /// Sprite-sheet animation: how many frames the resolved sprite has (0 if
    /// no sprite, or a plain single-frame one) and its total loop duration
    /// in seconds — set post-construction via `set_sprite_frames` once the
    /// caller has actually resolved the sprite texture (this module stays
    /// asset-agnostic, same convention as `material`/`render_onto`).
    sprite_frame_count: usize,
    sprite_duration: f32,
    animation_mode: AnimationMode,
    sequence_multiplier: f32,
    /// `rotationrandom` initializer range (z-axis component, radians) —
    /// `None` leaves spawn rotation at 0, matching the reference's absent-
    /// initializer default.
    rotation_range: Option<(f32, f32)>,
    /// `spritetrail` renderer parameters.
    sprite_trail: Option<SpriteTrailParams>,
    /// `mapsequencearoundcontrolpoint` initializer + its running sequence
    /// counter (shared across spawns, wraps at `count`).
    map_sequence: Option<MapSequenceParams>,
    map_sequence_index: u32,
    /// `mapsequencebetweencontrolpoints` initializer + its counter.
    map_seq_between: Option<MapSequenceBetweenParams>,
    map_seq_between_index: u32,
    /// `turbulentvelocityrandom` initializers; each adds its own velocity.
    turbulent_velocity: Vec<TurbulentVelocityParams>,
    /// `turbulence` operator parameters.
    turbulence: Option<TurbulenceParams>,
    /// `vortex` operator parameters.
    vortex: Option<VortexParams>,
    /// `remapvalue` operator parameters — a `Vec` since real content
    /// (`particles/presets/rain_screen.json`) attaches more than one (one
    /// targeting `velocity`, one targeting `speed`) and both must run; a
    /// single `Option` here used to silently drop every instance after the
    /// first.
    remap: Vec<RemapParams>,
    /// `collisionsphere`/`collisionbox`/`collisionbounds`/`collisionquad`/
    /// `collisionplane` operators — every one attached, since a system can
    /// have more than one collider (e.g. a floor plane plus a wall box).
    collisions: Vec<CollisionParams>,
    /// `boids` operator parameters.
    boids: Option<BoidsParams>,
    /// Simulation clock (sum of `step` dts) — drives the time-scrolled
    /// noise fields, the reference's `m_time`/`currentTime`.
    time: f32,
    /// `starttime`: emission is suppressed until the clock reaches this.
    start_time: f32,
    /// Preset `flags & 4` (perspective particles): sphere emitters use the
    /// 3D spherical-shell spawn distribution instead of the 2D disk.
    shell_3d: bool,
    /// Monotonic source for `Particle::id`.
    next_particle_id: u64,
    /// Attached child presets (see [`ChildRef`]); each holds its own live
    /// instances, stepped and drawn with this system.
    children: Vec<ChildSystem>,
}

/// How a child preset binds to its parent (see [`ChildRef`]).
#[derive(Debug, Clone, Copy, PartialEq)]
enum ChildType {
    Static,
    Follow,
    Spawn,
    Death,
}

/// A resolved child preset plus its live instances. The child's material
/// sprite/blending are its own (a glow child can be additive over a
/// normal-blended parent); both stages draw into the same canvas.
pub struct ChildSystem {
    child_type: ChildType,
    probability: f32,
    max_instances: usize,
    config: ParticleConfig,
    sprite: Option<ParticleSprite>,
    additive: bool,
    instances: Vec<ChildInstance>,
}

struct ChildInstance {
    /// `Some` while following a live parent particle (`eventfollow`);
    /// cleared when the parent dies so the instance stops emitting and
    /// retires once its remaining particles expire.
    parent_id: Option<u64>,
    /// Where this instance's emitters currently sit (parent particle
    /// position for `eventfollow`, event position for bursts).
    last_center: [f32; 2],
    /// Burst instances (`eventspawn`/`eventdeath`) stop emitting at this
    /// system-time cutoff — the preset itself has no end marker.
    emit_until: Option<f32>,
    /// `static` children run for the parent system's whole life and are
    /// never retired, even while momentarily empty between spawns.
    persistent: bool,
    system: ParticleSystem,
}

struct EmitterState {
    id: u32,
    rate: f32,
    origin: [f32; 3],
    directions: [f32; 3],
    sign: [f32; 3],
    /// True for `sphererandom` (annulus/shell spawn), false for box.
    is_sphere: bool,
    /// Spawn-offset range: sphere radius in `[0]` (min/max annulus), or the
    /// box emitter's per-axis half-extents. CParticle.cpp spawns particles
    /// randomly WITHIN this volume, scaled per-axis by `directions` —
    /// distancemax is a POSITION range, never a velocity.
    distance_min: [f32; 3],
    distance_max: [f32; 3],
    /// Optional emitter-driven radial speed (`speedmin`/`speedmax` JSON
    /// fields — distinct from distance!). 0 = the reference's default:
    /// emitters set no velocity, initializers do.
    speed_min: f32,
    speed_max: f32,
    accumulator: f32,
    /// `instantaneous` one-shot burst count, emitted once when the emitter
    /// activates; `burst_done` tracks that it's fired.
    burst: u32,
    burst_done: bool,
    /// `duration`: emitter stops after `start_time + duration` (None = forever).
    duration: Option<f32>,
    /// Set when the emitter has `audioprocessing*` — its rate follows the
    /// audio spectrum.
    audio: Option<AudioResponse>,
}

impl ParticleSystem {
    /// Build a running particle system for one scene object.
    ///
    /// `spawn_center` is the object's own scene position, already converted to
    /// canvas pixel coordinates (see `model::resolve`'s WE-origin-to-pixel
    /// conversion) — a WE emitter's own `origin` field is a *local* offset
    /// from the object's position, not an absolute canvas coordinate.
    pub fn from_config(
        config: &ParticleConfig,
        spawn_center: [f32; 2],
        overrides: Option<&InstanceOverride>,
    ) -> Self {
        let emitters = config
            .emitter
            .iter()
            .map(|e| {
                let local_origin = parse_f32_vec3(e.origin.as_deref());
                // Reference default (1, 1, 0) — a zero default would scale
                // every spawn offset to nothing for emitters that omit the
                // field, collapsing them to a point at the origin.
                let directions = e
                    .directions
                    .as_deref()
                    .map(|d| parse_f32_vec3(Some(d)))
                    .unwrap_or([1.0, 1.0, 0.0]);
                let sign = parse_f32_vec3(e.sign.as_deref());
                let is_sphere = e.name == "sphererandom";

                // Sphere radii are scalars, box extents "x y z" vectors —
                // accept either shape for both (splat scalars).
                let dist = |v: &Option<serde_json::Value>, default: f32| -> [f32; 3] {
                    v.as_ref()
                        .and_then(|v| {
                            value_as_vec3(v).or_else(|| value_as_f32(v).map(|f| [f, f, f]))
                        })
                        .unwrap_or([default, default, default])
                };
                let distance_min = dist(&e.distancemin, 0.0);
                // Reference default 256 for both emitter kinds.
                let distance_max = dist(&e.distancemax, 256.0);
                let speed_min = e.speedmin.as_ref().and_then(value_as_f32).unwrap_or(0.0);
                let speed_max = e.speedmax.as_ref().and_then(value_as_f32).unwrap_or(0.0);

                EmitterState {
                    id: e.id.unwrap_or(0),
                    rate: e.rate.unwrap_or(10.0) as f32,
                    origin: [
                        spawn_center[0] + local_origin[0],
                        // The emitter's local origin is authored y-up
                        // (CParticle.cpp negates it in both createBoxEmitter
                        // and createSphereEmitter): "0 768 0" means 768
                        // ABOVE the object — e.g. rainperspective's spawn
                        // band sits above the frame so drops fall through
                        // it, not 768 below (which pushed the whole band
                        // off-screen on 1080p scenes and rain never showed).
                        spawn_center[1] - local_origin[1],
                        local_origin[2],
                    ],
                    directions,
                    sign,
                    is_sphere,
                    distance_min,
                    distance_max,
                    speed_min,
                    speed_max,
                    accumulator: 0.0,
                    burst: e.instantaneous.unwrap_or(0.0).max(0.0) as u32,
                    burst_done: false,
                    duration: e.duration.map(|d| d as f32),
                    audio: AudioResponse::from_emitter(e),
                }
            })
            .collect();

        // `instanceoverride.count` scales the pool before the default kicks
        // in (CParticle.cpp: `adjustedMaxCount = maxCount * countMultiplier;
        // maxParticles = adjusted > 0 ? adjusted : DEFAULT`).
        let count_mult = overrides.and_then(|o| o.count).unwrap_or(1.0);
        let max_count = match config.maxcount {
            Some(n) if n > 0 => match (n as f64 * count_mult) as usize {
                0 => 500,
                adjusted => adjusted,
            },
            _ => 500,
        };

        // Initializer-absent defaults are the reference's exact spawn values
        // (CParticle.cpp: `p.lifetime = 1.0`, `p.size = 20.0` before
        // initializers run) — size passes through the sizerandom /2, so the
        // range default is 40.
        let (life_min, life_max) =
            scalar_range_from_initializers(&config.initializer, "lifetimerandom", 1.0, 1.0);
        let (size_min, size_max) =
            scalar_range_from_initializers(&config.initializer, "sizerandom", 40.0, 40.0);
        let life_exp = initializer_exponent(&config.initializer, "lifetimerandom");
        let size_exp = initializer_exponent(&config.initializer, "sizerandom");
        let (alpha_min, alpha_max) =
            scalar_range_from_initializers(&config.initializer, "alpharandom", 1.0, 1.0);
        // Particle vectors are authored y-up; this simulation is y-down, so
        // y flips on the way in (CParticle.cpp does the same for velocity,
        // gravity and turbulent directions, as it does for emitter origins).
        let flip_y = |v: [f32; 3]| [v[0], -v[1], v[2]];
        let velocity_range = vec3_range_from_initializers(
            &config.initializer,
            "velocityrandom",
            [-32.0; 3],
            [32.0; 3],
        );
        let (velocity_min, velocity_max) = match velocity_range {
            Some((min, max)) => (Some(flip_y(min)), Some(flip_y(max))),
            None => (None, None),
        };
        let (color_min, color_max) =
            vec3_range_from_initializers(&config.initializer, "colorrandom", [0.0; 3], [255.0; 3])
                .unwrap_or(([255.0; 3], [255.0; 3]));

        let movement_op = config.operator.iter().find(|op| op.name == "movement");
        let gravity = movement_op
            .and_then(|op| op.gravity.as_deref())
            .map(|s| parse_f32_vec3(Some(s)))
            .map(|v| [v[0], -v[1]])
            .unwrap_or([0.0, 0.0]);
        // `movement.drag`: per-second velocity decay (CParticle.cpp
        // createMovementOperator) — without it, operators that keep adding
        // velocity every frame (turbulence, vortex) grow unbounded instead
        // of settling into the drifting equilibrium real presets tune for.
        let movement_drag = movement_op.and_then(|op| op.drag).unwrap_or(0.0) as f32;

        let color_override = overrides
            .and_then(|o| o.color.as_deref())
            .map(|s| parse_f32_vec3(Some(s)))
            .map(|v| [v[0] as u8, v[1] as u8, v[2] as u8]);

        let alphafade = config
            .operator
            .iter()
            .find(|op| op.name == "alphafade")
            .map(|op| {
                (
                    op.fadeintime.unwrap_or(0.0) as f32,
                    op.fadeouttime.unwrap_or(1.0) as f32,
                )
            });
        let fade_range_op = |name: &str| -> Option<(f32, f32, f32, f32)> {
            config.operator.iter().find(|op| op.name == name).map(|op| {
                (
                    op.starttime.unwrap_or(0.0) as f32,
                    op.endtime.unwrap_or(1.0) as f32,
                    op.startvalue.as_ref().and_then(value_as_f32).unwrap_or(0.0),
                    op.endvalue.as_ref().and_then(value_as_f32).unwrap_or(1.0),
                )
            })
        };
        let sizechange = fade_range_op("sizechange");
        let alphachange = fade_range_op("alphachange");
        let colorchange = config
            .operator
            .iter()
            .find(|op| op.name == "colorchange")
            .map(|op| {
                (
                    op.starttime.unwrap_or(0.0) as f32,
                    op.endtime.unwrap_or(1.0) as f32,
                    op.startvalue
                        .as_ref()
                        .and_then(value_as_vec3)
                        .unwrap_or([1.0; 3]),
                    op.endvalue
                        .as_ref()
                        .and_then(value_as_vec3)
                        .unwrap_or([1.0; 3]),
                )
            });
        let oscillate_op = |name: &str| -> Option<OscillateParams> {
            config
                .operator
                .iter()
                .find(|op| op.name == name)
                .map(OscillateParams::from_operator)
        };
        let oscillate_alpha = oscillate_op("oscillatealpha");
        let oscillate_size = oscillate_op("oscillatesize");
        let oscillate_position = oscillate_op("oscillateposition");

        // Positional indexing (declaration order), not `id`-keyed slot
        // placement — real presets declare control points in id order, and
        // consumers (e.g. `controlpointattract`) reference them by a plain
        // array index anyway.
        let control_points: Vec<[f32; 3]> = config
            .controlpoint
            .iter()
            .map(|cp| {
                let mut offset = cp
                    .offset
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0; 3]);
                // `angles.z` rotates the offset about the object origin.
                if let Some(az) = cp
                    .angles
                    .as_ref()
                    .and_then(value_as_vec3)
                    .map(|a| a[2])
                    .filter(|z| z.abs() > 1e-6)
                {
                    let (s, c) = az.sin_cos();
                    offset = [
                        offset[0] * c - offset[1] * s,
                        offset[0] * s + offset[1] * c,
                        offset[2],
                    ];
                }
                [
                    spawn_center[0] + offset[0],
                    spawn_center[1] + offset[1],
                    offset[2],
                ]
            })
            .collect();

        let control_point_attract = config
            .operator
            .iter()
            .find(|op| op.name == "controlpointattract")
            .map(|op| {
                let idx = op.controlpoint.unwrap_or(0).max(0) as usize;
                let origin = op
                    .origin
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0; 3]);
                let scale = op.scale.unwrap_or(100.0) as f32;
                let threshold = op.threshold.unwrap_or(1000.0) as f32;
                (idx, origin, scale, threshold)
            });

        let rope_renderer = config.renderer.iter().find(|r| {
            r.get("name")
                .and_then(|n| n.as_str())
                .map(|n| n == "rope" || n == "ropetrail")
                .unwrap_or(false)
        });
        let rope_mode = rope_renderer.is_some();
        let rope_subdivision = rope_renderer
            .and_then(|r| r.get("subdivision"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
            .max(1) as usize;
        // Rope UV controls (ObjectParser::parseParticleRenderer defaults:
        // uvscale 1, uvscrolling false, uvsmoothing true). `uvscale` divides
        // the rope's UV length so values > 1 tile the texture along it.
        let json_bool = |v: &serde_json::Value| v.as_bool().or_else(|| v.as_u64().map(|n| n != 0));
        let rope_uv_scale = rope_renderer
            .and_then(|r| r.get("uvscale"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .filter(|v| *v > 0.0)
            .unwrap_or(1.0);
        let rope_uv_scrolling = rope_renderer
            .and_then(|r| r.get("uvscrolling"))
            .and_then(json_bool)
            .unwrap_or(false);
        let rope_uv_smoothing = rope_renderer
            .and_then(|r| r.get("uvsmoothing"))
            .and_then(json_bool)
            .unwrap_or(true);

        // `spritetrail` renderer: each particle draws as a quad stretched
        // along its velocity by `clamp(|v| * length, minlength, maxlength)`
        // (genericparticle.vert's TRAILRENDERER `ComputeParticleTrailTangents`;
        // parser defaults from ObjectParser::parseParticleRenderer).
        let sprite_trail = config
            .renderer
            .iter()
            .find(|r| r.get("name").and_then(|n| n.as_str()) == Some("spritetrail"))
            .map(|r| {
                let f = |key: &str, default: f32| -> f32 {
                    r.get(key)
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32)
                        .unwrap_or(default)
                };
                SpriteTrailParams {
                    length: f("length", 0.05),
                    max_length: f("maxlength", 10.0),
                    min_length: f("minlength", 0.0),
                }
            });

        // `angularvelocityrandom`'s min/max are vec3 (reference default
        // `(0,0,-5)`/`(0,0,5)`) — we only model the z-axis (screen-space)
        // component, matching how `rotation.z` is the only axis our flat
        // 2D renderer can express.
        let (angular_velocity_min, angular_velocity_max) = config
            .initializer
            .iter()
            .find(|init| init.name == "angularvelocityrandom")
            .map(|init| {
                let min = init
                    .min
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0, 0.0, -5.0]);
                let max = init
                    .max
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0, 0.0, 5.0]);
                (min[2], max[2])
            })
            .unwrap_or((0.0, 0.0));

        let angular_op = config
            .operator
            .iter()
            .find(|op| op.name == "angularmovement");
        let angular_drag = angular_op.and_then(|op| op.drag).unwrap_or(0.0) as f32;
        let angular_force = angular_op
            .and_then(|op| op.force.as_ref())
            .and_then(value_as_vec3)
            .map(|v| v[2])
            .unwrap_or(0.0);

        // `rotationrandom`: z component of the random vec3 (our renderer is
        // flat 2D, same reduction as `angularvelocityrandom`). Reference
        // defaults: min (0,0,0), max (0,0,2pi).
        let rotation_range = config
            .initializer
            .iter()
            .find(|init| init.name == "rotationrandom")
            .map(|init| {
                let min = init
                    .min
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0; 3]);
                let max = init.max.as_ref().and_then(value_as_vec3).unwrap_or([
                    0.0,
                    0.0,
                    std::f32::consts::TAU,
                ]);
                (min[2], max[2])
            });

        let turbulent_velocity = config
            .initializer
            .iter()
            .filter(|init| init.name == "turbulentvelocityrandom")
            .map(|init| TurbulentVelocityParams {
                speed_min: init
                    .speedmin
                    .as_ref()
                    .and_then(value_as_f32)
                    .unwrap_or(100.0),
                speed_max: init
                    .speedmax
                    .as_ref()
                    .and_then(value_as_f32)
                    .unwrap_or(250.0),
                offset: init.offset.unwrap_or(0.0) as f32,
                scale: init.scale.unwrap_or(1.0) as f32,
                forward: init
                    .forward
                    .as_ref()
                    .and_then(value_as_vec3)
                    .map(flip_y)
                    .unwrap_or([0.0, -1.0, 0.0]),
                timescale: init.timescale.unwrap_or(1.0) as f32,
                phase_min: init.phasemin.unwrap_or(0.0) as f32,
                phase_max: init.phasemax.unwrap_or(0.1) as f32,
                right: init
                    .right
                    .as_ref()
                    .and_then(value_as_vec3)
                    .map(flip_y)
                    .unwrap_or([0.0, 0.0, 1.0]),
            })
            .collect();

        // `turbulence`: phase and speed are randomized once per operator
        // instance (the reference draws them in the factory, not per
        // particle or per frame).
        let turbulence = config
            .operator
            .iter()
            .find(|op| op.name == "turbulence")
            .map(|op| {
                let phase_min = op.phasemin.unwrap_or(0.0) as f32;
                let phase_max = op.phasemax.unwrap_or(0.0) as f32;
                let speed_min = op.speedmin.unwrap_or(500.0) as f32;
                let speed_max = op.speedmax.unwrap_or(1000.0) as f32;
                TurbulenceParams {
                    noise_scale: op.scale.unwrap_or(0.005) as f32 * 2.0,
                    timescale: op.timescale.unwrap_or(0.01) as f32,
                    mask: op
                        .mask
                        .as_ref()
                        .and_then(value_as_vec3)
                        .unwrap_or([1.0, 1.0, 0.0]),
                    phase: phase_min + fastrand::f32() * (phase_max - phase_min).max(0.0),
                    speed: speed_min + fastrand::f32() * (speed_max - speed_min).max(0.0),
                }
            });

        let map_sequence = config
            .initializer
            .iter()
            .find(|init| init.name == "mapsequencearoundcontrolpoint")
            .map(|init| MapSequenceParams {
                control_point: init.controlpoint.unwrap_or(0).max(0) as usize,
                count: (init.count.unwrap_or(1.0) as u32).max(1),
                speed_min: init
                    .speedmin
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0; 3]),
                speed_max: init
                    .speedmax
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([100.0; 3]),
            });

        let map_seq_between = config
            .initializer
            .iter()
            .find(|init| init.name == "mapsequencebetweencontrolpoints")
            .map(|init| MapSequenceBetweenParams {
                count: (init.count.unwrap_or(2.0) as u32).max(2),
                mirror: init.limitbehavior.as_deref() == Some("mirror"),
            });

        let vortex = config
            .operator
            .iter()
            .find(|op| op.name == "vortex" || op.name == "vortex_v2")
            .map(|op| {
                let flags = op.flags.as_ref().and_then(|v| v.as_i64()).unwrap_or(0);
                VortexParams {
                    control_point: op.controlpoint.unwrap_or(0),
                    infinite_axis: flags & 1 != 0,
                    maintain_distance: flags & 2 != 0,
                    ring_shape: flags & 4 != 0,
                    axis: op
                        .axis
                        .as_ref()
                        .and_then(value_as_vec3)
                        .unwrap_or([0.0, 0.0, 1.0]),
                    offset: op
                        .offset
                        .as_ref()
                        .and_then(value_as_vec3)
                        .unwrap_or([0.0; 3]),
                    distance_inner: op.distanceinner.unwrap_or(500.0) as f32,
                    distance_outer: op.distanceouter.unwrap_or(650.0) as f32,
                    speed_inner: op.speedinner.unwrap_or(2500.0) as f32,
                    speed_outer: op.speedouter.unwrap_or(0.0) as f32,
                    center_force: op.centerforce.unwrap_or(1.0) as f32,
                    ring_radius: op.ringradius.unwrap_or(300.0) as f32,
                    ring_width: op.ringwidth.unwrap_or(50.0) as f32,
                    ring_pull_distance: op.ringpulldistance.unwrap_or(50.0) as f32,
                    ring_pull_force: op.ringpullforce.unwrap_or(10.0) as f32,
                }
            });

        let remap: Vec<RemapParams> = config
            .operator
            .iter()
            .filter(|op| op.name == "remapvalue")
            .map(|op| {
                let range = |v: &Option<serde_json::Value>, d: f32| {
                    v.as_ref()
                        .and_then(value_as_vec3)
                        .or_else(|| v.as_ref().and_then(value_as_f32).map(|s| [s, s, s]))
                        .unwrap_or([d, d, d])
                };
                let scalar = |v: &Option<serde_json::Value>, d: f32| {
                    v.as_ref().and_then(value_as_f32).unwrap_or(d)
                };
                let input = RemapField::parse(op.input.as_deref());
                // No `transformfunction` key at all: noise-generator default
                // when there's no `input` field either (the classic,
                // real-content-validated case, `rain_screen.json`);
                // passthrough when an `input` field IS given, so setting
                // `input` alone uses the field's value directly instead of
                // silently overriding it with noise. Confirmed against the
                // retail binary's own `alphafade`-preset builder
                // (Follow-up (x)), which pairs `input=lifetimefraction` with
                // an *explicit* `transformfunction=none` — this default
                // reproduces that pairing even when a wallpaper's own config
                // omits the (redundant) explicit "none".
                let transform = match op.transformfunction.as_deref() {
                    Some(s) => RemapTransform::parse(s),
                    None if input == RemapField::None => RemapTransform::Simplex,
                    None => RemapTransform::None,
                };
                RemapParams {
                    input,
                    input_component: RemapComponent::parse(op.inputcomponent.as_deref()),
                    output: RemapOutput::parse(op.output.as_deref()),
                    output_component: RemapComponent::parse(op.outputcomponent.as_deref()),
                    operation: RemapOp::parse(op.operation.as_deref()),
                    transform,
                    in_min: scalar(&op.inputrangemin, 0.0),
                    in_max: scalar(&op.inputrangemax, 1.0),
                    out_min: range(&op.outputrangemin, 0.0),
                    out_max: range(&op.outputrangemax, 0.0),
                    input_scale: op.transforminputscale.unwrap_or(1.0) as f32,
                }
            })
            .collect();

        // `collisionsphere`/`collisionbox`/`collisionbounds`/`collisionquad`/
        // `collisionplane` — see the `Operator`/`CollisionParams` field docs
        // for the "no C++ reference" caveat. Anchored at `controlpoint` (+
        // `offset`), same static-resolve-once convention as
        // `controlpointattract`/`vortex`.
        const COLLISION_NAMES: [&str; 5] = [
            "collisionsphere",
            "collisionbox",
            "collisionbounds",
            "collisionquad",
            "collisionplane",
        ];
        let collisions: Vec<CollisionParams> = config
            .operator
            .iter()
            .filter(|op| COLLISION_NAMES.contains(&op.name.as_str()))
            .map(|op| {
                let cp_idx = op.controlpoint.unwrap_or(0).max(0) as usize;
                let cp = control_points.get(cp_idx).copied().unwrap_or([0.0; 3]);
                let offset = op
                    .offset
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0; 3]);
                let center = [cp[0] + offset[0], cp[1] + offset[1]];
                let shape = match op.name.as_str() {
                    "collisionsphere" => CollisionShape::Sphere {
                        radius: op.radius.unwrap_or(100.0) as f32,
                    },
                    "collisionplane" => {
                        // Default normal (0,1) is "up" in our y-down space
                        // (screen-up); `angles.z` rotates it, same convention
                        // `mapsequencearoundcontrolpoint`-adjacent offset
                        // rotation already uses elsewhere in this module.
                        let az = op
                            .angles
                            .as_ref()
                            .and_then(value_as_vec3)
                            .map(|a| a[2])
                            .unwrap_or(0.0);
                        let (s, c) = az.sin_cos();
                        CollisionShape::Plane { normal: [-s, c] }
                    }
                    // collisionbox/collisionbounds/collisionquad
                    _ => {
                        let size = op
                            .size
                            .as_ref()
                            .and_then(value_as_vec3)
                            .unwrap_or([200.0, 200.0, 0.0]);
                        CollisionShape::Box {
                            half_extent: [size[0] * 0.5, size[1] * 0.5],
                        }
                    }
                };
                CollisionParams {
                    shape,
                    center,
                    behavior: CollisionBehavior::parse(op.collisionbehavior.as_deref()),
                    bounce_factor: op.bouncefactor.unwrap_or(1.0) as f32,
                }
            })
            .collect();

        // `boids`: standard separation/alignment/cohesion flocking.
        let boids = config
            .operator
            .iter()
            .find(|op| op.name == "boids")
            .map(|op| BoidsParams {
                separation_threshold: op.separationthreshold.unwrap_or(50.0) as f32,
                neighbor_threshold: op.neighborthreshold.unwrap_or(150.0) as f32,
                max_speed: op.maxspeed.unwrap_or(500.0) as f32,
                separation_factor: op.separationfactor.unwrap_or(1.0) as f32,
                alignment_factor: op.alignmentfactor.unwrap_or(1.0) as f32,
                cohesion_factor: op.cohesionfactor.unwrap_or(1.0) as f32,
                variable_strength: op.variablestrength.unwrap_or(0.0) != 0.0,
            });

        Self {
            particles: Vec::with_capacity(max_count),
            emitters,
            max_count,
            life_min,
            life_max,
            size_min,
            size_max,
            life_exp,
            size_exp,
            alpha_min,
            alpha_max,
            velocity_min,
            velocity_max,
            color_min,
            color_max,
            gravity,
            movement_drag,
            alphafade,
            sizechange,
            alphachange,
            colorchange,
            oscillate_alpha,
            oscillate_size,
            oscillate_position,
            alpha_mult: overrides.and_then(|o| o.alpha).unwrap_or(1.0) as f32,
            lifetime_mult: overrides.and_then(|o| o.lifetime).unwrap_or(1.0) as f32,
            colorn_mult: overrides
                .and_then(|o| o.colorn.as_ref())
                .and_then(value_as_vec3_pub)
                .unwrap_or([1.0, 1.0, 1.0]),
            size_mult: overrides.and_then(|o| o.size).unwrap_or(1.0) as f32,
            speed_mult: overrides.and_then(|o| o.speed).unwrap_or(1.0) as f32,
            rate_mult: overrides.and_then(|o| o.rate).unwrap_or(1.0) as f32,
            color_override,
            control_points,
            control_point_attract,
            rope_mode,
            rope_subdivision,
            rope_uv_scale,
            rope_uv_scrolling,
            rope_uv_smoothing,
            shell_3d: config.flags.unwrap_or(0) & 4 != 0,
            next_particle_id: 0,
            children: Vec::new(),
            angular_velocity_min,
            angular_velocity_max,
            angular_force,
            angular_drag,
            sprite_frame_count: 0,
            sprite_duration: 0.0,
            animation_mode: AnimationMode::from_config(config.animationmode.as_deref()),
            sequence_multiplier: config
                .sequencemultiplier
                .map(|v| v as f32)
                .filter(|v| *v > 0.0)
                .unwrap_or(1.0),
            rotation_range,
            sprite_trail,
            map_sequence,
            map_sequence_index: 0,
            map_seq_between,
            map_seq_between_index: 0,
            turbulent_velocity,
            turbulence,
            vortex,
            remap,
            collisions,
            boids,
            time: 0.0,
            start_time: config.starttime.unwrap_or(0.0).max(0.0) as f32,
            scene_force: [0.0, 0.0],
            audio_bands: [0.0; 16],
        }
    }

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
    fn step_children(&mut self, dt: f32, spawned: &[(u64, [f32; 2])], died: &[(u64, [f32; 2])]) {
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
    fn shift_origin(&mut self, delta: [f32; 2]) {
        for emitter in &mut self.emitters {
            emitter.origin[0] += delta[0];
            emitter.origin[1] += delta[1];
        }
        for cp in &mut self.control_points {
            cp[0] += delta[0];
            cp[1] += delta[1];
        }
    }

    fn stop_emitting(&mut self) {
        for emitter in &mut self.emitters {
            emitter.rate = 0.0;
        }
    }
}

fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// Minimal vec3 math for the turbulent/vortex ports (the reference uses glm;
// pulling in a linear-algebra crate for five one-liners isn't worth it).

fn lerp_u8(min: f32, max: f32) -> u8 {
    (min + fastrand::f32() * (max - min)).clamp(0.0, 255.0) as u8
}

/// Linear ramp from `start_val` to `end_val` over `[start, end]` of `t`,
/// clamped flat outside that range. Matches `Maths::fadeValue` exactly.
fn fade_value(t: f32, start: f32, end: f32, start_val: f32, end_val: f32) -> f32 {
    if t <= start {
        start_val
    } else if t >= end {
        end_val
    } else {
        let f = (t - start) / (end - start);
        start_val + f * (end_val - start_val)
    }
}

#[cfg(test)]
mod tests;
