//! Building a particle system from its config: reading every emitter,
//! initializer, operator and renderer entry into the runtime parameters.

use super::*;

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
}
