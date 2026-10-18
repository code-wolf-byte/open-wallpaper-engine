use super::*;

/// Real preset fields (velocityrandom, colorrandom, angularvelocityrandom)
/// use `"x y z"` vector strings for min/max, not numbers — the old
/// `Option<f64>` typing rejected these outright.
#[test]
fn parses_real_preset_with_vector_initializers() {
    let json = r#"{
        "maxcount": 20,
        "emitter": [{"name":"sphererandom","rate":1.5,"distancemin":0,"distancemax":512}],
        "initializer": [
            {"id":2,"name":"lifetimerandom","min":3,"max":5},
            {"id":3,"name":"sizerandom","min":1000,"max":2200},
            {"id":4,"name":"velocityrandom","min":"25 0 0","max":"80 0 0"},
            {"id":5,"name":"colorrandom","min":"255 255 255","max":"255 255 255"}
        ],
        "operator": [{"id":9,"name":"movement","gravity":"0 0 0"}],
        "renderer": [{"id":1,"name":"sprite"}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let sys = ParticleSystem::from_config(&config, [100.0, 200.0], None);
    assert_eq!(sys.velocity_min, Some([25.0, 0.0, 0.0]));
    assert_eq!(sys.velocity_max, Some([80.0, 0.0, 0.0]));
    assert_eq!(sys.color_min, [255.0, 255.0, 255.0]);
}

#[test]
fn scalar_initializer_without_vector_fields_still_parses() {
    let json = r#"{
        "maxcount": 16,
        "emitter": [{"name":"box","rate":1}],
        "initializer": [{"id":2,"name":"lifetimerandom","min":1,"max":1}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    assert_eq!(sys.life_min, 1.0);
    assert_eq!(sys.life_max, 1.0);
    assert!(sys.velocity_min.is_none());
}

#[test]
fn instance_override_sets_color_and_multipliers() {
    let json = r#"{
        "maxcount": 5,
        "emitter": [{"name":"box","rate":1}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let overrides = InstanceOverride {
        alpha: Some(0.5),
        color: Some("192 192 192".to_string()),
        rate: Some(2.0),
        ..Default::default()
    };
    let sys = ParticleSystem::from_config(&config, [0.0, 0.0], Some(&overrides));
    assert_eq!(sys.color_override, Some([192, 192, 192]));
    assert_eq!(sys.alpha_mult, 0.5);
    assert_eq!(sys.rate_mult, 2.0);
}

/// `count` scales the pool, `lifetime` scales spawned lifetimes, and
/// `colorn` tints the colorrandom result — the multiplier semantics of
/// CParticle.cpp (lines 59, 755-765, 712-720).
#[test]
fn instance_override_count_lifetime_colorn() {
    let json = r#"{
        "maxcount": 100,
        "emitter": [{"name":"box","rate":1000}],
        "initializer": [
            {"id":1,"name":"lifetimerandom","min":2,"max":2},
            {"id":2,"name":"colorrandom","min":"200 100 50","max":"200 100 50"}
        ]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let overrides = InstanceOverride {
        count: Some(0.5),
        lifetime: Some(2.0),
        colorn: Some(serde_json::json!("0.5 1 1")),
        ..Default::default()
    };
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], Some(&overrides));
    assert_eq!(sys.max_count, 50, "count 0.5 should halve maxcount 100");
    sys.step(0.01);
    let p = &sys.particles[0];
    assert_eq!(p.life, 4.0, "lifetime 2.0 should double the 2s lifetime");
    assert_eq!(
        p.color,
        [100, 100, 50],
        "colorn should tint the red channel only"
    );
}

/// `enabled: false` in an instanceoverride must drop the whole system at
/// the layer level (checked in render.rs/animated.rs, config-level here).
#[test]
fn instance_override_enabled_parses() {
    let o: InstanceOverride =
        serde_json::from_str(r#"{"enabled": false, "id": 3}"#).expect("should parse");
    assert_eq!(o.enabled.as_ref().and_then(|v| v.as_bool()), Some(false));
}

/// Emitter local origins are authored y-up (CParticle.cpp negates the y
/// in both emitter constructors): "10 20 0" sits 20 ABOVE the object in
/// screen space — rainperspective's "0 768 0" spawn band must end up
/// above the frame, not below it.
#[test]
fn emitter_origin_is_local_offset_from_spawn_center() {
    let json = r#"{
        "maxcount": 5,
        "emitter": [{"name":"box","rate":1,"origin":"10 20 0"}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let sys = ParticleSystem::from_config(&config, [100.0, 200.0], None);
    assert_eq!(sys.emitters[0].origin[0], 110.0);
    assert_eq!(sys.emitters[0].origin[1], 180.0);
}

#[test]
fn movement_operator_gravity_field_parses() {
    let json = r#"{
        "maxcount": 5,
        "emitter": [{"name":"box","rate":1}],
        "operator": [{"id":1,"name":"movement","gravity":"0 -50 0"}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    // Authored y-up: "0 -50 0" pulls down the screen, +y in this
    // y-down simulation.
    assert_eq!(sys.gravity, [0.0, 50.0]);
}

#[test]
fn fade_value_clamps_outside_range_and_lerps_inside() {
    assert_eq!(fade_value(0.0, 0.3, 0.7, 0.0, 1.0), 0.0);
    assert_eq!(fade_value(0.3, 0.3, 0.7, 0.0, 1.0), 0.0);
    assert_eq!(fade_value(0.5, 0.3, 0.7, 0.0, 1.0), 0.5);
    assert_eq!(fade_value(0.7, 0.3, 0.7, 0.0, 1.0), 1.0);
    assert_eq!(fade_value(1.0, 0.3, 0.7, 0.0, 1.0), 1.0);
}

/// A real `alphafade` preset (fog1.json): fadeintime=0.5 means alpha
/// should be 0 at spawn, ~full partway through the ramp, and full for
/// the rest of the particle's life (no fadeouttime = holds at 1.0).
#[test]
fn alphafade_ramps_in_then_holds() {
    let json = r#"{
        "maxcount": 5,
        "emitter": [{"name":"box","rate":1}],
        "initializer": [{"id":1,"name":"alpharandom","min":1,"max":1}],
        "operator": [{"id":1,"name":"alphafade","fadeintime":0.5}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    // Force-spawn one particle with a known, fixed lifetime.
    sys.life_min = 10.0;
    sys.life_max = 10.0;
    sys.emitters[0].rate = 1000.0; // spawn immediately
    sys.step(0.001); // this step only spawns; the fade math runs on existing particles first
    assert_eq!(sys.particles.len(), 1);

    // Just after spawn (age≈0), fully faded in from 0.
    sys.step(0.001);
    assert!(sys.particles[0].alpha < 0.1);

    // Halfway through the fade-in window (age=2.5s of 10s life → lifetime_pos=0.25,
    // half of fadeintime=0.5): alpha should be roughly half.
    sys.step(2.499);
    let mid = sys.particles[0].alpha;
    assert!((0.3..0.7).contains(&mid), "expected ~0.5, got {mid}");

    // Well past the fade-in window, alpha should be at full (no fadeouttime set → holds at 1.0).
    sys.step(5.0);
    assert!(sys.particles[0].alpha > 0.9);
}

/// A real `sizechange` preset (magic_pulse.json): startvalue=0, endvalue=2,
/// endtime=1 (normalized) — size should start near 0 and grow past the
/// initializer-sampled size as lifetime_pos approaches endtime.
#[test]
fn sizechange_scales_initial_size_over_lifetime() {
    let json = r#"{
        "maxcount": 5,
        "emitter": [{"name":"box","rate":1}],
        "initializer": [{"id":1,"name":"sizerandom","min":100,"max":100}],
        "operator": [{"id":1,"name":"sizechange","startvalue":0,"endvalue":2,"endtime":1}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.life_min = 10.0;
    sys.life_max = 10.0;
    sys.emitters[0].rate = 1000.0;
    sys.step(0.001); // this step only spawns; the fade math runs on existing particles first
    assert_eq!(sys.particles.len(), 1);
    let initial_size = sys.particles[0].initial_size;
    sys.step(0.001);
    assert!(sys.particles[0].size < initial_size * 0.1);

    // lifetime_pos=1.0 reached at age=10s (endtime=1 normalized == full life here).
    sys.step(9.99);
    assert!((sys.particles[0].size - initial_size * 2.0).abs() < initial_size * 0.1);
}

/// A real `colorchange` preset (torch.json): "r g b" 0..1 vector strings
/// for startvalue/endvalue — these previously broke parsing entirely
/// (Operator.startvalue/endvalue were typed as bare f64).
#[test]
fn colorchange_parses_and_multiplies_initial_color_over_lifetime() {
    let json = r#"{
        "maxcount": 5,
        "emitter": [{"name":"box","rate":1}],
        "initializer": [{"id":1,"name":"colorrandom","min":"255 255 255","max":"255 255 255"}],
        "operator": [{"id":1,"name":"colorchange","startvalue":"1 0.75 0","endvalue":"1 0 0","endtime":1}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.life_min = 10.0;
    sys.life_max = 10.0;
    sys.emitters[0].rate = 1000.0;
    sys.step(0.001); // spawns the particle; fade math runs on existing particles first
    assert_eq!(sys.particles.len(), 1);

    sys.step(0.001); // age≈0 → at startvalue (1, 0.75, 0) * white
    let c0 = sys.particles[0].color;
    assert_eq!(c0[0], 255);
    assert!((c0[1] as i32 - 191).abs() <= 2); // 0.75*255≈191
    assert_eq!(c0[2], 0);

    sys.step(9.99); // age≈10 → lifetime_pos=1.0 → at endvalue (1, 0, 0) * white
    let c1 = sys.particles[0].color;
    assert_eq!(c1, [255, 0, 0]);
}

/// A real `controlpointattract` preset pulls particles within `threshold/2`
/// of a control point toward it, and leaves particles outside that radius
/// untouched (CParticle.cpp createControlPointAttractOperator).
#[test]
fn controlpointattract_pulls_nearby_particles_toward_control_point() {
    let json = r#"{
        "maxcount": 5,
        "emitter": [{"name":"box","rate":1}],
        "controlpoint": [{"id":0,"offset":"200 0 0"}],
        "operator": [{"id":1,"name":"controlpointattract","controlpoint":0,"scale":500,"threshold":1000}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    // spawn_center [0,0] + control point offset "200 0 0" -> control point at [200, 0].
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    assert_eq!(sys.control_points, vec![[200.0, 0.0, 0.0]]);

    sys.life_min = 10.0;
    sys.life_max = 10.0;
    sys.emitters[0].rate = 1000.0;
    sys.step(0.001); // spawns the particle at/near [0,0]
    assert_eq!(sys.particles.len(), 1);
    let start_dist = ((sys.particles[0].x - 200.0).powi(2) + sys.particles[0].y.powi(2)).sqrt();

    for _ in 0..10 {
        sys.step(0.05);
    }
    let end_dist = ((sys.particles[0].x - 200.0).powi(2) + sys.particles[0].y.powi(2)).sqrt();
    assert!(
        end_dist < start_dist,
        "expected particle to be pulled toward control point: start={start_dist} end={end_dist}"
    );
}

/// A particle outside the attract radius (`threshold/2`) should be left
/// alone entirely.
#[test]
fn controlpointattract_ignores_particles_outside_threshold() {
    let json = r#"{
        "maxcount": 5,
        "emitter": [{"name":"box","rate":1}],
        "controlpoint": [{"id":0,"offset":"5000 0 0"}],
        "operator": [{"id":1,"name":"controlpointattract","controlpoint":0,"scale":500,"threshold":10}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.life_min = 10.0;
    sys.life_max = 10.0;
    sys.emitters[0].rate = 1000.0;
    sys.step(0.001);
    assert_eq!(sys.particles.len(), 1);
    // Capture the emitter-assigned spawn velocity (unrelated to the
    // control point), then confirm a step leaves it untouched since the
    // particle is well outside the attract radius.
    let (vx0, vy0) = (sys.particles[0].vx, sys.particles[0].vy);
    sys.step(0.05);
    assert_eq!(sys.particles[0].vx, vx0);
    assert_eq!(sys.particles[0].vy, vy0);
}

/// A `rope`/`ropetrail` renderer should draw a connected ribbon through
/// living particles rather than isolated circles — verify pixels are
/// filled *between* particle centers, not just at them.
#[test]
fn rope_renderer_fills_ribbon_between_particles() {
    let json = r#"{
        "maxcount": 5,
        "emitter": [{"name":"box","rate":1}],
        "initializer": [{"id":1,"name":"sizerandom","min":40,"max":40}],
        "renderer": [{"id":1,"name":"rope","subdivision":4}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    assert!(sys.rope_mode);
    assert_eq!(sys.rope_subdivision, 4);

    // Manually place three particles in a horizontal line, all alive.
    let mut sys = sys;
    for (i, x) in [0.0f32, 50.0, 100.0].into_iter().enumerate() {
        sys.particles.push(Particle {
            id: 0,
            x,
            y: 100.0,
            vx: 0.0,
            vy: 0.0,
            life: 5.0,
            max_life: 5.0,
            size: 20.0,
            alpha: 1.0,
            color: [255, 0, 0],
            initial_alpha: 1.0,
            initial_size: 20.0,
            initial_color: [1.0, 0.0, 0.0],
            osc_alpha: None,
            osc_size: None,
            osc_pos: None,
            rotation: 0.0,
            angular_velocity: 0.0,
            frame: -1.0,
        });
        let _ = i;
    }

    let mut canvas = RgbaImage::new(200, 200);
    sys.render_onto(&mut canvas, None, [0.0, 0.0]);

    // Midpoint between the first two particles (25, 100) should be filled
    // — a plain per-particle circle draw (radius 10) would leave this gap
    // empty since it's 25px from the nearest particle center.
    let mid_pixel = canvas.get_pixel(25, 100);
    assert!(
        mid_pixel[3] > 0,
        "expected ribbon fill at midpoint, got {mid_pixel:?}"
    );
}

/// A rope with a material texture must sample it across the ribbon width
/// (U = perpendicular offset): a beam strip that's transparent except for
/// its center column should render as a thin filament along the rope's
/// spine, leaving the outer ribbon area untouched — the fix for fat
/// solid lightning bolts (discharge presets) and giant trail wedges.
#[test]
fn rope_renderer_samples_texture_across_width() {
    let json = r#"{
        "maxcount": 5,
        "emitter": [{"name":"box","rate":1}],
        "renderer": [{"id":1,"name":"rope","subdivision":1}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    for x in [0.0f32, 50.0, 100.0] {
        sys.particles.push(make_particle(x, 100.0, 20.0, 0.0));
    }

    // Beam strip: transparent except the 3 center columns (white, opaque).
    let mut beam = RgbaImage::new(33, 4);
    for y in 0..4 {
        for x in 15..=17 {
            beam.put_pixel(x, y, image::Rgba([255, 255, 255, 255]));
        }
    }
    let sprite = ParticleSprite {
        frames: vec![beam],
        duration: 0.0,
        overbright: 1.0,
    };

    let mut canvas = RgbaImage::new(200, 200);
    sys.render_onto(&mut canvas, Some(&sprite), [0.0, 0.0]);

    // The rope spine (u = 0.5 → the white center column) is filled…
    let spine = canvas.get_pixel(25, 100);
    assert!(
        spine[3] > 0,
        "expected filament on rope spine, got {spine:?}"
    );
    // …but the ribbon area off-center (still inside the 10px half-width
    // the old flat fill covered) samples the transparent texture edge.
    let edge = canvas.get_pixel(25, 92);
    assert_eq!(edge[3], 0, "expected transparent ribbon edge, got {edge:?}");
}

fn child_ref(child_type: Option<&str>, probability: f64) -> ChildRef {
    ChildRef {
        name: "child.json".into(),
        child_type: child_type.map(str::to_string),
        probability: Some(probability),
        maxcount: None,
    }
}

fn simple_child_config() -> ParticleConfig {
    serde_json::from_str(
        r#"{
            "maxcount": 10,
            "emitter": [{"name":"box","rate":1000}],
            "initializer": [{"id":1,"name":"lifetimerandom","min":5,"max":5}]
        }"#,
    )
    .expect("child config should parse")
}

/// An `eventfollow` child spawns one instance per parent particle and
/// keeps its emitters glued to the parent as it moves.
#[test]
fn child_eventfollow_tracks_parent_particle() {
    let json = r#"{
        "maxcount": 1,
        "emitter": [{"name":"box","rate":1000}],
        "initializer": [
            {"id":1,"name":"lifetimerandom","min":10,"max":10},
            {"id":2,"name":"velocityrandom","min":"100 0 0","max":"100 0 0"}
        ]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.add_child(
        simple_child_config(),
        None,
        false,
        &child_ref(Some("eventfollow"), 1.0),
        [0.0, 0.0],
    );

    sys.step(0.01); // spawns the parent + its follow instance
    assert_eq!(sys.particles.len(), 1);
    assert_eq!(sys.children[0].instances.len(), 1);

    sys.step(1.0); // parent moves +100x; instance must follow
    let parent = [sys.particles[0].x, sys.particles[0].y];
    let inst = &sys.children[0].instances[0];
    assert_eq!(inst.last_center, parent, "instance should track the parent");
    assert!(
        !inst.system.particles.is_empty(),
        "follow instance should be emitting its own particles"
    );
}

/// Probability 0 must never attach an instance; a dead parent stops the
/// instance's emitters and lets it retire once its particles expire.
#[test]
fn child_probability_and_parent_death() {
    let json = r#"{
        "maxcount": 1,
        "emitter": [{"name":"box","rate":1000}],
        "initializer": [{"id":1,"name":"lifetimerandom","min":0.05,"max":0.05}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");

    let mut never = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    never.add_child(
        simple_child_config(),
        None,
        false,
        &child_ref(Some("eventfollow"), 0.0),
        [0.0, 0.0],
    );
    never.step(0.01);
    assert!(never.children[0].instances.is_empty());

    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.add_child(
        simple_child_config(),
        None,
        false,
        &child_ref(Some("eventfollow"), 1.0),
        [0.0, 0.0],
    );
    sys.step(0.01);
    assert_eq!(sys.children[0].instances.len(), 1);
    sys.step(0.1); // parent (life 0.05) dies
    let inst = &sys.children[0].instances[0];
    assert!(
        inst.parent_id.is_none(),
        "dead parent should detach the instance"
    );
    assert!(
        inst.system.emitters.iter().all(|e| e.rate == 0.0),
        "detached instance should stop emitting"
    );
}

/// A type-less (static) child gets one persistent instance at the parent
/// object's origin that emits continuously, surviving empty stretches.
#[test]
fn child_static_runs_continuously() {
    let json = r#"{
        "maxcount": 1,
        "emitter": [{"name":"box","rate":0}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [40.0, 40.0], None);
    sys.add_child(
        simple_child_config(),
        None,
        false,
        &child_ref(None, 1.0),
        [40.0, 40.0],
    );
    assert_eq!(
        sys.children[0].instances.len(),
        1,
        "static child instantiates immediately"
    );
    for _ in 0..10 {
        sys.step(0.05);
    }
    assert!(!sys.children[0].instances[0].system.particles.is_empty());
    assert!(sys.children[0].instances[0].persistent);
}

fn make_particle(x: f32, y: f32, size: f32, rotation: f32) -> Particle {
    Particle {
        id: 0,
        x,
        y,
        vx: 0.0,
        vy: 0.0,
        life: 5.0,
        max_life: 5.0,
        size,
        alpha: 1.0,
        color: [255, 255, 255],
        initial_alpha: 1.0,
        initial_size: size,
        initial_color: [1.0, 1.0, 1.0],
        osc_alpha: None,
        osc_size: None,
        osc_pos: None,
        rotation,
        angular_velocity: 0.0,
        frame: -1.0,
    }
}

/// A textured particle's rotation should actually affect what's drawn —
/// sampling the same canvas location at two different rotation angles
/// must produce different colors, proving the quad (and the texture
/// sampled onto it) is genuinely rotated, not just passed through.
#[test]
fn textured_particle_rotation_changes_sampled_footprint() {
    // 8x8 sprite: left half red, right half blue, fully opaque.
    let mut tex = RgbaImage::new(8, 8);
    for y in 0..8 {
        for x in 0..8 {
            let color = if x < 4 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            };
            tex.put_pixel(x, y, image::Rgba(color));
        }
    }

    let config: ParticleConfig =
        serde_json::from_str(r#"{"maxcount":1,"emitter":[{"name":"box","rate":1}]}"#)
            .expect("should parse");

    let sprite = ParticleSprite::single(tex.clone());
    let render_at = |rotation: f32| -> image::Rgba<u8> {
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.particles
            .push(make_particle(50.0, 50.0, 20.0, rotation));
        let mut canvas = RgbaImage::new(100, 100);
        sys.render_onto(&mut canvas, Some(&sprite), [0.0, 0.0]);
        *canvas.get_pixel(44, 50)
    };

    let unrotated = render_at(0.0);
    let rotated = render_at(std::f32::consts::FRAC_PI_2);
    assert_ne!(
        unrotated, rotated,
        "expected rotation to change the sampled footprint at a fixed canvas point"
    );
}

/// `rotationrandom` with a degenerate `[min.z, max.z]` range must spawn
/// every particle at exactly that rotation (CParticle.cpp
/// createRotationRandomInitializer, z-axis reduction); absent, spawn
/// rotation stays 0.
#[test]
fn rotationrandom_sets_spawn_rotation() {
    let json = r#"{
        "maxcount": 4,
        "emitter": [{"name":"box","rate":1000}],
        "initializer": [{"id":1,"name":"rotationrandom","min":"0 0 1","max":"0 0 1"}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.step(0.001);
    assert!((sys.particles[0].rotation - 1.0).abs() < 1e-4);

    let plain: ParticleConfig =
        serde_json::from_str(r#"{"maxcount":1,"emitter":[{"name":"box","rate":1000}]}"#)
            .expect("should parse");
    let mut sys = ParticleSystem::from_config(&plain, [0.0, 0.0], None);
    sys.step(0.001);
    assert_eq!(sys.particles[0].rotation, 0.0);
}

/// `turbulentvelocityrandom` adds a curl-noise-directed velocity of
/// magnitude in `[speedmin, speedmax]` on top of the plain velocity —
/// with a zero base velocity, the spawn speed must equal the turbulent
/// speed exactly (the direction is noise, the magnitude is not).
#[test]
fn turbulentvelocityrandom_sets_spawn_speed_magnitude() {
    let json = r#"{
        "maxcount": 8,
        "emitter": [{"name":"box","rate":1000}],
        "initializer": [
            {"id":1,"name":"velocityrandom","min":"0 0 0","max":"0 0 0"},
            {"id":2,"name":"turbulentvelocityrandom","speedmin":100,"speedmax":100}
        ]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.step(0.001);
    assert!(!sys.particles.is_empty());
    for p in &sys.particles {
        let speed = (p.vx * p.vx + p.vy * p.vy).sqrt();
        assert!(
            (speed - 100.0).abs() < 0.5,
            "expected turbulent spawn speed 100, got {speed}"
        );
    }
}

/// The `turbulence` operator must push otherwise-motionless particles
/// around (curl-noise acceleration, CParticle.cpp
/// createTurbulenceOperator).
#[test]
fn turbulence_operator_accelerates_particles() {
    let json = r#"{
        "maxcount": 4,
        "emitter": [{"name":"box","rate":1000}],
        "initializer": [{"id":1,"name":"velocityrandom","min":"0 0 0","max":"0 0 0"}],
        "operator": [{"id":1,"name":"turbulence","speedmin":500,"speedmax":500,"scale":0.01}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [10.0, 20.0], None);
    for _ in 0..10 {
        sys.step(1.0 / 30.0);
    }
    let moving = sys
        .particles
        .iter()
        .any(|p| (p.vx * p.vx + p.vy * p.vy).sqrt() > 1.0);
    assert!(moving, "turbulence should have accelerated particles");
}

/// `vortex` spins particles tangentially around its control point: a
/// particle at +X from the center with a +Z axis must gain -Y velocity
/// in our y-down space (the reference's y-up cross(axis, radial),
/// mirrored — see the reflection note in `step`), scaled by speedinner
/// inside distanceinner.
#[test]
fn vortex_operator_adds_tangential_velocity() {
    let json = r#"{
        "maxcount": 4,
        "emitter": [{"name":"box","rate":0}],
        "controlpoint": [{"id":0,"offset":"0 0 0"}],
        "operator": [{"id":1,"name":"vortex","controlpoint":0,
                      "distanceinner":500,"distanceouter":650,
                      "speedinner":100,"speedouter":0}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.particles.push(make_particle(10.0, 0.0, 5.0, 0.0));
    sys.step(0.1);
    let p = &sys.particles[0];
    assert!(
        p.vy < -0.5,
        "expected tangential -Y velocity from vortex, got vy={}",
        p.vy
    );
    assert!(
        p.vx.abs() < 1e-3,
        "no radial force expected, got vx={}",
        p.vx
    );
}

/// `collisionsphere` with the default `bounce` behavior: a particle that
/// has penetrated the sphere gets pushed back to the surface and its
/// into-surface velocity component reflected (scaled by `1+bouncefactor`).
#[test]
fn collisionsphere_bounces_particle_off_surface() {
    let json = r#"{
        "maxcount": 1,
        "emitter": [{"name":"box","rate":0}],
        "controlpoint": [{"id":0,"offset":"0 0 0"}],
        "operator": [{"id":1,"name":"collisionsphere","controlpoint":0,"radius":50}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    let mut p = make_particle(10.0, 0.0, 5.0, 0.0);
    p.vx = -50.0;
    sys.particles.push(p);
    sys.step(0.1);
    let p = &sys.particles[0];
    assert!(
        (p.x - 50.0).abs() < 1e-3,
        "expected pushed to surface, x={}",
        p.x
    );
    assert!(
        p.vx > 0.0,
        "expected reflected outward velocity, vx={}",
        p.vx
    );
}

/// `collisionbox` with `slide`: the into-wall velocity component is
/// zeroed (not reflected), and the particle is pushed back to the wall.
#[test]
fn collisionbox_slide_zeroes_into_wall_velocity() {
    let json = r#"{
        "maxcount": 1,
        "emitter": [{"name":"box","rate":0}],
        "controlpoint": [{"id":0,"offset":"0 0 0"}],
        "operator": [{"id":1,"name":"collisionbox","controlpoint":0,
                      "size":"100 100 0","collisionbehavior":"slide"}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    let mut p = make_particle(60.0, 0.0, 5.0, 0.0);
    p.vx = -100.0;
    sys.particles.push(p);
    sys.step(0.15);
    let p = &sys.particles[0];
    assert!(
        (p.x - 50.0).abs() < 1e-3,
        "expected pushed to wall, x={}",
        p.x
    );
    assert!(
        p.vx.abs() < 1e-3,
        "expected into-wall velocity zeroed, vx={}",
        p.vx
    );
}

/// `collisionplane` with `delete`: a particle on the solid side of the
/// plane is killed outright (removed by `step`'s own retain).
#[test]
fn collisionplane_delete_removes_particle() {
    let json = r#"{
        "maxcount": 1,
        "emitter": [{"name":"box","rate":0}],
        "controlpoint": [{"id":0,"offset":"0 0 0"}],
        "operator": [{"id":1,"name":"collisionplane","controlpoint":0,
                      "collisionbehavior":"delete"}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.particles.push(make_particle(0.0, -10.0, 5.0, 0.0));
    sys.step(0.1);
    assert!(
        sys.particles.is_empty(),
        "particle should be deleted on contact"
    );
}

/// Audio response: WE's parser defaults, band clamp/swap, and the gain
/// ramp runs between the `bounds` volume levels over the selected bands.
#[test]
fn audio_response_defaults_bands_and_bounds() {
    let em = |j: &str| serde_json::from_str::<Emitter>(j).unwrap();
    assert_eq!(AudioResponse::from_emitter(&em("{}")), None);
    let d = AudioResponse::from_emitter(&em(r#"{"audioprocessingmode": 1}"#)).unwrap();
    assert_eq!((d.exponent, d.bounds, d.bands), (2.0, [0.8, 1.0], [0, 1]));
    let r = AudioResponse::from_emitter(&em(
        r#"{"audioprocessingbounds": "0.2 1.5", "audioprocessingexponent": 1,
            "audioprocessingfrequencystart": 40, "audioprocessingfrequencyend": 6}"#,
    ))
    .unwrap();
    assert_eq!(r.bands, [6, 15]);
    let mut bands = [0.0; 16];
    assert_eq!(r.gain(&bands), 0.0); // silence: emitter inactive
    bands[0] = 1.0; // outside 6..=15: no effect
    assert_eq!(r.gain(&bands), 0.0);
    bands[6..].fill(0.85);
    assert!((r.gain(&bands) - 0.5).abs() < 1e-6); // (0.85-0.2)/(1.5-0.2)
    bands[6..].fill(1.0);
    assert!((r.gain(&bands) - 0.8 / 1.3).abs() < 1e-6);
}

/// `boids` cohesion pulls two distant particles toward each other.
#[test]
fn boids_cohesion_pulls_particles_together() {
    let json = r#"{
        "maxcount": 2,
        "emitter": [{"name":"box","rate":0}],
        "operator": [{"id":1,"name":"boids","neighborthreshold":500,
                      "separationthreshold":1,"cohesionfactor":1.0,"alignmentfactor":0}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.particles.push(make_particle(-100.0, 0.0, 5.0, 0.0));
    sys.particles.push(make_particle(100.0, 0.0, 5.0, 0.0));
    sys.step(0.1);
    assert!(
        sys.particles[0].vx > 1.0,
        "left particle should move right, vx={}",
        sys.particles[0].vx
    );
    assert!(
        sys.particles[1].vx < -1.0,
        "right particle should move left, vx={}",
        sys.particles[1].vx
    );
}

/// `boids` separation pushes two nearby particles apart.
#[test]
fn boids_separation_pushes_particles_apart() {
    let json = r#"{
        "maxcount": 2,
        "emitter": [{"name":"box","rate":0}],
        "operator": [{"id":1,"name":"boids","separationthreshold":50,
                      "neighborthreshold":1,"separationfactor":1.0}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.particles.push(make_particle(-2.0, 0.0, 5.0, 0.0));
    sys.particles.push(make_particle(2.0, 0.0, 5.0, 0.0));
    sys.step(0.1);
    assert!(
        sys.particles[0].vx < 0.0,
        "left particle should move further left, vx={}",
        sys.particles[0].vx
    );
    assert!(
        sys.particles[1].vx > 0.0,
        "right particle should move further right, vx={}",
        sys.particles[1].vx
    );
}

/// `movement.drag` decays velocity by `1 - drag*dt` per step, clamped
/// at zero so an over-large `drag*dt` stops the particle instead of
/// reversing it (CParticle.cpp createMovementOperator).
#[test]
fn movement_drag_decays_velocity_without_reversal() {
    let json = r#"{
        "maxcount": 1,
        "emitter": [{"name":"box","rate":0}],
        "operator": [{"id":1,"name":"movement","gravity":"0 0 0","drag":4.0}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    let mut p = make_particle(50.0, 50.0, 5.0, 0.0);
    p.vx = 100.0;
    sys.particles.push(p);

    sys.step(0.1); // drag*dt = 0.4 -> vx *= 0.6
    assert!((sys.particles[0].vx - 60.0).abs() < 1e-3);

    sys.step(0.5); // drag*dt = 2.0 -> clamps to 0, must not reverse
    assert_eq!(sys.particles[0].vx, 0.0);
}

/// `mapsequencebetweencontrolpoints` walks spawn positions along the
/// cp0->cp1 segment; `limitbehavior: "mirror"` ping-pongs (0, 1/3, 2/3,
/// 1, 2/3, 1/3, 0, ...) instead of wrapping.
#[test]
fn mapsequence_between_walks_segment_with_mirror() {
    let json = r#"{
        "maxcount": 8,
        "emitter": [{"name":"box","rate":100000}],
        "controlpoint": [{"id":0,"offset":"0 0 0"},{"id":1,"offset":"90 0 0"}],
        "initializer": [
            {"id":1,"name":"velocityrandom","min":"0 0 0","max":"0 0 0"},
            {"id":2,"name":"mapsequencebetweencontrolpoints","count":4,"limitbehavior":"mirror"}
        ]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.step(0.001);
    assert!(sys.particles.len() >= 7, "expected at least 7 spawns");
    let xs: Vec<f32> = sys.particles.iter().take(7).map(|p| p.x).collect();
    let expect = [0.0, 30.0, 60.0, 90.0, 60.0, 30.0, 0.0];
    for (i, (&x, &e)) in xs.iter().zip(expect.iter()).enumerate() {
        assert!((x - e).abs() < 1e-3, "slot {i}: expected x={e}, got {x}");
    }
}

/// `mapsequencearoundcontrolpoint` spawns at the control point and
/// launches successive particles at evenly-divided angles: with
/// `count: 4` and a +X-only speed, consecutive spawn velocities must
/// point right/down-rotated/left/up-rotated (the reference's clockwise
/// mat3), all with the same magnitude.
#[test]
fn mapsequence_spawns_at_control_point_with_rotating_velocity() {
    let json = r#"{
        "maxcount": 4,
        "emitter": [{"name":"box","rate":100000}],
        "controlpoint": [{"id":0,"offset":"50 60 0"}],
        "initializer": [{"id":1,"name":"mapsequencearoundcontrolpoint",
                         "controlpoint":0,"count":4,
                         "speedmin":"80 0 0","speedmax":"80 0 0"}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.step(0.001);
    assert!(sys.particles.len() >= 4, "expected at least 4 spawns");
    for p in &sys.particles {
        assert_eq!((p.x, p.y), (50.0, 60.0), "must spawn at the control point");
        let speed = (p.vx * p.vx + p.vy * p.vy).sqrt();
        assert!(
            (speed - 80.0).abs() < 1e-3,
            "speed must stay 80, got {speed}"
        );
    }
    // First two spawns: angle 0 (+X) then 2pi/4 = 90deg rotated.
    assert!((sys.particles[0].vx - 80.0).abs() < 1e-3);
    assert!(sys.particles[0].vy.abs() < 1e-3);
    assert!(sys.particles[1].vx.abs() < 1e-3);
    assert!(sys.particles[1].vy.abs() > 79.0);
}

fn two_frame_sprite() -> ParticleSprite {
    ParticleSprite {
        frames: vec![
            RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255])),
            RgbaImage::from_pixel(4, 4, image::Rgba([0, 0, 255, 255])),
        ],
        duration: 1.0,
        overbright: 1.0,
    }
}

fn render_center_with_frame(frame: f32) -> image::Rgba<u8> {
    let config: ParticleConfig =
        serde_json::from_str(r#"{"maxcount":1,"emitter":[{"name":"box","rate":0}]}"#)
            .expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    let mut p = make_particle(20.0, 20.0, 8.0, 0.0);
    p.frame = frame;
    sys.particles.push(p);
    let sprite = two_frame_sprite();
    let mut canvas = RgbaImage::new(40, 40);
    sys.render_onto(&mut canvas, Some(&sprite), [0.0, 0.0]);
    *canvas.get_pixel(20, 20)
}

/// A fractional frame position cross-fades between the two adjacent
/// frames (genericparticle.frag's SPRITESHEETBLEND): halfway between a
/// red and a blue frame must paint ~50/50 purple, not hard-switch.
#[test]
fn fractional_frame_cross_fades_adjacent_frames() {
    let px = render_center_with_frame(0.5);
    assert!(px[0] > 100 && px[0] < 155, "red ~half, got {px:?}");
    assert!(px[2] > 100 && px[2] < 155, "blue ~half, got {px:?}");
    assert_eq!(px[1], 0);
}

/// The next frame is clamped at the last frame (ComputeSpriteFrame's
/// `min(numFrames - 1, currentFrame + 1)`) — a fraction past the final
/// frame must NOT wrap-blend back toward frame 0.
#[test]
fn cross_fade_clamps_at_last_frame_without_wrapping() {
    let px = render_center_with_frame(1.5);
    assert_eq!(px[0], 0, "no red from a wrap to frame 0: {px:?}");
    assert!(px[2] > 200, "pure last frame, got {px:?}");
}

/// Integer frame positions (RandomFrame assigns whole frames) take the
/// single-sample path — exactly one frame's color, no blending.
#[test]
fn integer_frame_draws_single_frame_unblended() {
    let px = render_center_with_frame(1.0);
    assert_eq!(px[0], 0, "{px:?}");
    assert!(px[2] > 200, "{px:?}");
    let px0 = render_center_with_frame(0.0);
    assert!(px0[0] > 200, "{px0:?}");
    assert_eq!(px0[2], 0, "{px0:?}");
}

/// `spritetrail` stretches the quad along the particle's velocity:
/// a fast horizontal particle must paint a footprint wider than tall
/// (its quad's long axis lies along X), unlike the square default.
#[test]
fn spritetrail_stretches_quad_along_velocity() {
    let json = r#"{
        "maxcount": 1,
        "emitter": [{"name":"box","rate":0}],
        "renderer": [{"name":"spritetrail","length":1.0,"maxlength":10}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    let mut p = make_particle(50.0, 50.0, 5.0, 0.0);
    p.vx = 8.0; // trail = clamp(8 * 1.0, 0, 10) = 8 -> half-height 40 along X
    let mut sys = sys;
    sys.particles.push(p);

    let tex = RgbaImage::from_pixel(4, 4, image::Rgba([255, 255, 255, 255]));
    let sprite = ParticleSprite::single(tex);
    let mut canvas = RgbaImage::new(100, 100);
    sys.render_onto(&mut canvas, Some(&sprite), [0.0, 0.0]);

    let painted_x = canvas.get_pixel(80, 50)[3] > 0; // 30px right of center
    let painted_y = canvas.get_pixel(50, 80)[3] > 0; // 30px below center
    assert!(painted_x, "trail should extend along +X (velocity)");
    assert!(!painted_y, "trail should stay narrow across velocity");
}

/// Name matching must be exact: `turbulentvelocityrandom` declared
/// before `velocityrandom` (and `angularmovement` before `movement`)
/// must not shadow them — the old substring matcher (`contains`)
/// read the turbulent initializer's fields as the plain velocity
/// range and the angular operator's (absent) gravity as movement's.
/// An omitted `min`/`max` takes WE's default instead of discarding the
/// initializer: 3598808038's vapor sets only `colorrandom.max = 0 0 0`
/// and must come out black, not the no-initializer white.
#[test]
fn partial_vec3_initializers_use_we_defaults() {
    let json = r#"{
        "maxcount": 4,
        "emitter": [{"name":"box","rate":1}],
        "initializer": [
            {"id":1,"name":"colorrandom","max":"0 0 0"},
            {"id":2,"name":"velocityrandom","min":"0 10 0"}
        ]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    assert_eq!((sys.color_min, sys.color_max), ([0.0; 3], [0.0; 3]));
    assert_eq!(sys.velocity_min, Some([0.0, -10.0, 0.0]));
    assert_eq!(sys.velocity_max, Some([32.0, -32.0, 32.0]));
}

#[test]
fn similarly_named_initializers_do_not_shadow_each_other() {
    let json = r#"{
        "maxcount": 4,
        "emitter": [{"name":"box","rate":1000}],
        "initializer": [
            {"id":1,"name":"turbulentvelocityrandom","speedmin":500,"speedmax":900},
            {"id":2,"name":"velocityrandom","min":"3 3 0","max":"3 3 0"}
        ],
        "operator": [
            {"id":1,"name":"angularmovement","drag":0.5},
            {"id":2,"name":"movement","gravity":"0 -7 0"}
        ]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    // Authored y-up vectors flip into this y-down simulation. Real
    // content settles the direction: snow, rain and leaf presets fall
    // with -y velocity, embers rise with +y gravity.
    assert_eq!(sys.velocity_min, Some([3.0, -3.0, 0.0]));
    assert_eq!(sys.velocity_max, Some([3.0, -3.0, 0.0]));
    assert_eq!(sys.gravity[1], 7.0);
}

/// A config with no `alpharandom` initializer at all must stay a no-op
/// (full alpha, matching `ParticleInstance::alpha`'s `1.0` default) —
/// the common case, and the one that must not regress.
#[test]
fn no_alpharandom_initializer_defaults_to_full_alpha() {
    let config: ParticleConfig =
        serde_json::from_str(r#"{"maxcount":1,"emitter":[{"name":"box","rate":1000}]}"#)
            .expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.step(0.001);
    assert_eq!(sys.particles[0].initial_alpha, 1.0);
}

/// `alpharandom` (e.g. fog1.json's `"min":0.15,"max":0.2`) must actually
/// constrain spawn alpha to that range instead of always spawning at
/// full opacity — this was silently ignored before.
#[test]
fn alpharandom_constrains_spawn_alpha_to_its_range() {
    let json = r#"{
        "maxcount": 20,
        "emitter": [{"name":"box","rate":1000}],
        "initializer": [{"id":1,"name":"alpharandom","min":0.15,"max":0.2}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.step(0.05); // spawn several at once (rate 1000 * 0.05s = 50, capped at maxcount)
    assert!(!sys.particles.is_empty());
    for p in &sys.particles {
        assert!(
            (0.15..=0.2).contains(&p.initial_alpha),
            "expected initial_alpha in [0.15, 0.2], got {}",
            p.initial_alpha
        );
        assert_eq!(
            p.alpha, p.initial_alpha,
            "no alphafade — alpha should equal initial_alpha before any fade operator runs"
        );
    }
}

/// An instance-override alpha multiplier must apply exactly once (not
/// squared) even when combined with `alpharandom`.
#[test]
fn alpha_override_multiplier_applies_once_not_twice() {
    let json = r#"{
        "maxcount": 1,
        "emitter": [{"name":"box","rate":1000}],
        "initializer": [{"id":1,"name":"alpharandom","min":1.0,"max":1.0}]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let overrides = InstanceOverride {
        alpha: Some(0.5),
        ..Default::default()
    };
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], Some(&overrides));
    sys.step(0.01);
    assert_eq!(sys.particles[0].initial_alpha, 0.5);
    sys.step(0.01);
    // No alphafade/alphachange configured, so the plain lifetime-based
    // death-fade fallback (`initial_alpha * life/max_life`) applies —
    // barely distinguishable from 0.5 this soon after spawn, but not
    // bit-exact. The point of this assertion is that it's nowhere near
    // 0.5*0.5=0.25 (the double-multiply bug), not exact equality.
    assert!(
        (sys.particles[0].alpha - 0.5).abs() < 0.01,
        "override should apply once (~0.5), not 0.5*0.5=0.25; got {}",
        sys.particles[0].alpha
    );
}

/// Default (no `animationmode`) sprite-sheet playback loops continuously
/// based on lifetime position: at half life with 4 frames, the particle
/// should be sitting on frame 2.
#[test]
fn sprite_frame_loops_by_lifetime_position() {
    let config: ParticleConfig =
        serde_json::from_str(r#"{"maxcount":1,"emitter":[{"name":"box","rate":1000}]}"#)
            .expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.life_min = 10.0;
    sys.life_max = 10.0;
    sys.set_sprite_frames(4, 0.0);
    sys.step(0.001); // spawn
    assert_eq!(sys.particles.len(), 1);
    sys.step(5.0); // half of a 10s life
    assert!((sys.particles[0].frame - 2.0).abs() < 0.01);
}

/// `animationmode: "once"` plays through the sheet across the particle's
/// lifetime and clamps at the last frame instead of wrapping.
#[test]
fn sprite_frame_clamps_at_last_frame_when_animation_mode_once() {
    let config: ParticleConfig = serde_json::from_str(
        r#"{"maxcount":1,"animationmode":"once","emitter":[{"name":"box","rate":1000}]}"#,
    )
    .expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.life_min = 10.0;
    sys.life_max = 10.0;
    sys.set_sprite_frames(4, 0.0);
    sys.step(0.001); // spawn
    sys.step(9.99); // nearly the whole 10s life
    assert_eq!(sys.particles[0].frame, 3.0);
}

/// `animationmode: "randomframe"` assigns one frame at spawn and freezes
/// it — a second `step()` must not change it.
#[test]
fn sprite_random_frame_stays_fixed_after_assignment() {
    let config: ParticleConfig = serde_json::from_str(
        r#"{"maxcount":1,"animationmode":"randomframe","emitter":[{"name":"box","rate":1000}]}"#,
    )
    .expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.life_min = 10.0;
    sys.life_max = 10.0;
    sys.set_sprite_frames(5, 0.0);
    sys.step(0.001); // spawn (frame assignment lags by one step, like other operators)
    sys.step(0.001); // assigns a random frame
    let frame = sys.particles[0].frame;
    assert!((0.0..5.0).contains(&frame));
    sys.step(1.0);
    assert_eq!(sys.particles[0].frame, frame);
}

/// A sprite with only one frame should never advance `p.frame` at all —
/// `render_onto` always draws frame 0 for it regardless.
#[test]
fn single_frame_sprite_leaves_frame_at_spawn_sentinel() {
    let config: ParticleConfig =
        serde_json::from_str(r#"{"maxcount":1,"emitter":[{"name":"box","rate":1000}]}"#)
            .expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.life_min = 10.0;
    sys.life_max = 10.0;
    sys.set_sprite_frames(1, 0.0);
    sys.step(0.001);
    sys.step(5.0);
    assert_eq!(sys.particles[0].frame, -1.0);
}

#[test]
fn bounds_is_none_when_no_particles_alive() {
    let config: ParticleConfig =
        serde_json::from_str(r#"{"maxcount":5,"emitter":[{"name":"box","rate":0}]}"#)
            .expect("should parse");
    let sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    assert!(sys.bounds().is_none());
}

#[test]
fn bounds_covers_alive_particle_extents() {
    let json = r#"{
        "maxcount": 5,
        "emitter": [{"name":"box","rate":1,"distancemax":"0 0 0"}],
        "initializer": [
            {"id":1,"name":"lifetimerandom","min":10,"max":10},
            {"id":2,"name":"sizerandom","min":20,"max":20},
            {"id":3,"name":"velocityrandom","min":"0 0 0","max":"0 0 0"}
        ]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [100.0, 100.0], None);
    sys.emitters[0].rate = 1000.0;
    sys.step(0.001);
    assert_eq!(sys.particles.len(), 1);

    let (min_x, min_y, max_x, max_y) = sys.bounds().expect("should have bounds");
    // Particle spawned at (100,100) (spread disabled via distancemax=0).
    // `sizerandom`'s value is halved at spawn (diameter → radius, see
    // the comment on that halving elsewhere in this file), so a
    // `sizerandom` of 20 gives an actual radius of 10; `bounds()` adds a
    // couple of pixels of margin on top of that.
    assert!(min_x < 100.0 - 9.0 && min_x > 100.0 - 16.0, "min_x={min_x}");
    assert!(max_x > 100.0 + 9.0 && max_x < 100.0 + 16.0, "max_x={max_x}");
    assert!(min_y < 100.0 - 9.0 && min_y > 100.0 - 16.0, "min_y={min_y}");
    assert!(max_y > 100.0 + 9.0 && max_y < 100.0 + 16.0, "max_y={max_y}");
}

/// `render_onto`'s `origin` param should shift particle canvas positions,
/// letting a caller raster into a canvas sized to just the bbox instead
/// of the full scene.
#[test]
fn render_onto_origin_shifts_particle_positions() {
    let json = r#"{
        "maxcount": 5,
        "emitter": [{"name":"box","rate":1}],
        "initializer": [
            {"id":1,"name":"lifetimerandom","min":10,"max":10},
            {"id":2,"name":"sizerandom","min":10,"max":10},
            {"id":3,"name":"velocityrandom","min":"0 0 0","max":"0 0 0"},
            {"id":4,"name":"alpharandom","min":1,"max":1}
        ]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
    let mut sys = ParticleSystem::from_config(&config, [50.0, 50.0], None);
    sys.emitters[0].rate = 1000.0;
    sys.step(0.001);
    assert_eq!(sys.particles.len(), 1);

    // A canvas the same size as the bbox, with origin shifted to the
    // bbox's top-left, should show the particle rendered near its
    // center rather than clipped at (0,0).
    let (min_x, min_y, max_x, max_y) = sys.bounds().unwrap();
    let w = (max_x - min_x).ceil() as u32;
    let h = (max_y - min_y).ceil() as u32;
    let mut canvas = RgbaImage::new(w, h);
    sys.render_onto(&mut canvas, None, [min_x, min_y]);

    let cx = (w / 2).min(w - 1);
    let cy = (h / 2).min(h - 1);
    assert!(
        canvas.get_pixel(cx, cy)[3] > 0,
        "expected particle to render near the shifted canvas center"
    );
}

#[test]
fn oscillate_params_defaults_scale_min_from_scale_max_when_absent() {
    let op = Operator {
        name: "oscillatealpha".to_string(),
        frequencymin: Some(0.5),
        frequencymax: Some(1.0),
        scalemax: Some(10.0),
        ..Default::default()
    };
    let params = OscillateParams::from_operator(&op);
    assert_eq!(params.scale_min, 10.0);
    assert_eq!(params.scale_max, 10.0);
    assert_eq!(params.freq_min, 0.5);
    assert_eq!(params.freq_max, 1.0);
}

#[test]
fn oscillate_sample_phase_covers_full_circle_when_unspecified() {
    let op = Operator {
        name: "oscillatealpha".to_string(),
        frequencymin: Some(1.0),
        frequencymax: Some(1.0),
        scalemin: Some(1.0),
        scalemax: Some(1.0),
        ..Default::default()
    };
    let params = OscillateParams::from_operator(&op);
    for _ in 0..50 {
        let s = params.sample();
        assert!((0.0..std::f32::consts::TAU).contains(&s.phase));
    }
}

#[test]
fn remapvalue_sets_velocity_within_output_range() {
    let json = r#"{
        "maxcount": 20,
        "emitter": [{"name":"boxrandom","rate":1000,"distancemin":"0 0 0","distancemax":"0 0 0"}],
        "initializer": [
            {"id":1,"name":"lifetimerandom","min":100,"max":100},
            {"id":2,"name":"sizerandom","min":10,"max":10}
        ],
        "operator": [
            {"id":1,"name":"remapvalue","output":"velocity",
             "outputrangemin":"-100 -200 0","outputrangemax":"100 -1200 1",
             "transformfunction":"simplexnoise","transforminputscale":10}
        ]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).unwrap();
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    for _ in 0..10 {
        sys.step(1.0 / 30.0);
    }
    assert!(!sys.particles.is_empty(), "should have spawned particles");
    for p in &sys.particles {
        // Velocity is set (then position-integrated) into the remapped
        // range each frame; check it lands within bounds (+ a small margin
        // for the integration/other operators).
        assert!(
            (-101.0..=101.0).contains(&p.vx),
            "vx {} out of remapped x-range",
            p.vx
        );
        assert!(
            (-1201.0..=-199.0).contains(&p.vy),
            "vy {} out of remapped y-range",
            p.vy
        );
    }
}

/// Real content (`particles/presets/rain_screen.json`, workshop item
/// `3564529588`) attaches *two* `remapvalue` operators — one targeting
/// `velocity`, one `speed` — and both must take effect. The old
/// `Option<RemapParams>` silently dropped every instance after the
/// first; this is the regression test for the `Vec` fix.
#[test]
fn multiple_remapvalue_operators_all_apply() {
    let json = r#"{
        "maxcount": 20,
        "emitter": [{"name":"boxrandom","rate":1000,"distancemin":"0 0 0","distancemax":"0 0 0"}],
        "initializer": [
            {"id":1,"name":"lifetimerandom","min":100,"max":100}
        ],
        "operator": [
            {"id":1,"name":"remapvalue","output":"velocity",
             "outputrangemin":"5 5 0","outputrangemax":"5 5 0",
             "transformfunction":"simplexnoise","transforminputscale":10},
            {"id":2,"name":"remapvalue","output":"speed",
             "outputrangemin":"50","outputrangemax":"50",
             "transformfunction":"fbmnoise","transforminputscale":8}
        ]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).unwrap();
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    // Particles spawn at the *end* of `step()`, after the per-particle
    // update loop that applies `remapvalue` — a freshly-spawned particle
    // hasn't had it applied yet on the frame it's born (same reasoning
    // `remapvalue_sets_velocity_within_output_range` above steps
    // multiple times for), so step twice before checking.
    sys.step(1.0 / 30.0);
    sys.step(1.0 / 30.0);
    assert!(!sys.particles.is_empty(), "should have spawned particles");
    for p in &sys.particles {
        // A degenerate [5,5]x[5,5] range collapses velocity's own remap
        // to a fixed heading; the speed operator then rescales its
        // magnitude to 50 — only observable if *both* operators ran.
        let mag = (p.vx * p.vx + p.vy * p.vy).sqrt();
        assert!((mag - 50.0).abs() < 0.5, "speed {mag} not rescaled to ~50");
    }
}

/// The general field-to-field form: `input=lifetimefraction,
/// operation=multiply, output=opacity` is exactly the shape the retail
/// binary's own `alphafade`-preset builders construct internally
/// (Ghidra report Follow-up (x), `FUN_1401bfbb0`/`FUN_1401bc4b0`) — a
/// lifetime-driven fade-out multiplier. Checks opacity decreases
/// monotonically as the particle ages toward its declared output range.
#[test]
fn remapvalue_multiply_by_lifetimefraction_fades_opacity() {
    let json = r#"{
        "maxcount": 5,
        "emitter": [{"name":"boxrandom","rate":1000,"distancemin":"0 0 0","distancemax":"0 0 0"}],
        "initializer": [
            {"id":1,"name":"lifetimerandom","min":30,"max":30},
            {"id":2,"name":"alpharandom","min":1,"max":1}
        ],
        "operator": [
            {"id":1,"name":"remapvalue","input":"lifetimefraction",
             "operation":"multiply","output":"opacity",
             "inputrangemin":0,"inputrangemax":1,
             "outputrangemin":1,"outputrangemax":0,
             "transformfunction":"none"}
        ]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).unwrap();
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    sys.step(1.0 / 30.0);
    assert!(!sys.particles.is_empty(), "should have spawned particles");
    let alpha_early = sys.particles[0].alpha;
    for _ in 0..20 {
        sys.step(1.0 / 30.0);
    }
    assert!(!sys.particles.is_empty(), "particle should still be alive");
    let alpha_late = sys.particles[0].alpha;
    assert!(
        alpha_late < alpha_early,
        "opacity should fade as lifetimefraction rises toward 1 (early={alpha_early}, late={alpha_late})"
    );
}

/// `transformfunction:"none"` with a real `input` field is a straight
/// remap (default `operation`), no shaping — the simplest, most literal
/// case of the general form, targeting `size` this time.
#[test]
fn remapvalue_remaps_size_from_maxlifetime() {
    let json = r#"{
        "maxcount": 5,
        "emitter": [{"name":"boxrandom","rate":1000,"distancemin":"0 0 0","distancemax":"0 0 0"}],
        "initializer": [
            {"id":1,"name":"lifetimerandom","min":40,"max":40},
            {"id":2,"name":"sizerandom","min":1,"max":1}
        ],
        "operator": [
            {"id":1,"name":"remapvalue","input":"maxlifetime",
             "output":"size","transformfunction":"none",
             "inputrangemin":0,"inputrangemax":40,
             "outputrangemin":0,"outputrangemax":200}
        ]
    }"#;
    let config: ParticleConfig = serde_json::from_str(json).unwrap();
    let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
    // See the comment in `multiple_remapvalue_operators_all_apply`: a
    // freshly-spawned particle hasn't had `remapvalue` applied yet on
    // its spawn frame.
    sys.step(1.0 / 30.0);
    sys.step(1.0 / 30.0);
    assert!(!sys.particles.is_empty(), "should have spawned particles");
    for p in &sys.particles {
        // maxlifetime=40 sits at the top of [0,40] -> size should land
        // at (or very near) the top of [0,200], overriding sizerandom's
        // spawn value of 1.
        assert!(
            (p.size - 200.0).abs() < 1.0,
            "size {} not remapped from maxlifetime",
            p.size
        );
    }
}

mod colorn_probe {
    use super::*;

    #[test]
    fn colorn_override_tints_the_spawn_color() {
        let preset = r#"{
            "maxcount": 8,
            "emitter": [{"name": "sphererandom", "id": 1, "rate": 100, "distancemax": "1 1 1"}],
            "initializer": [
                {"id": 4, "max": "255 255 255", "min": "255 255 255", "name": "colorrandom"},
                {"id": 3, "max": 10, "min": 10, "name": "sizerandom"},
                {"id": 2, "max": 1, "min": 1, "name": "lifetimerandom"}
            ],
            "operator": []
        }"#;
        let cfg: ParticleConfig = serde_json::from_str(preset).expect("preset parses");
        let ov: InstanceOverride =
            serde_json::from_str(r#"{"colorn": "0.180 0.408 0.773", "id": null, "lifetime": 2.0}"#)
                .expect("override parses");
        assert_eq!(
            ov.colorn.as_ref().and_then(value_as_vec3_pub),
            Some([0.18, 0.408, 0.773])
        );
        let mut sys = ParticleSystem::from_config(&cfg, [0.0, 0.0], Some(&ov));
        for _ in 0..20 {
            sys.step(0.05);
        }
        let p = sys.particles.first().expect("something spawned");
        // `colorrandom` is pinned to white (min == max) so the tint is the only
        // thing acting on the colour, and every spawned particle is identical.
        // With a random base this assertion was only true for most samples: a
        // blue draw near the low end of [47,255] tints to ~36, which is *below*
        // the fixed red of 255*0.180 ≈ 46, so the suite failed at random.
        let expected = [255.0 * 0.180, 255.0 * 0.408, 255.0 * 0.773];
        for (channel, want) in expected.iter().enumerate() {
            let got = p.color[channel] as f32;
            assert!(
                (got - want).abs() <= 2.0,
                "channel {channel} is {got}, expected ~{want}: {:?}",
                p.color
            );
        }
    }
}
