use super::*;

#[test]
fn compose_and_fullscreen_objects_become_backdrop_layers() {
    let objs: Vec<SceneObject> = serde_json::from_str(
        r#"[{"image": "models/util/composelayer.json", "size": "200 100"},
            {"image": "models/util/fullscreenlayer.json", "size": "200 100"},
            {"image": "models/util/projectlayer.json"},
            {"image": "models/other.json"}]"#,
    )
    .unwrap();
    let a = backdrop_layer_from_object(&objs[0], objs[0].image.as_deref().unwrap()).unwrap();
    assert_eq!(a.backdrop, Some(Backdrop::Region));
    assert_eq!([a.size[0], a.size[1]], [200.0, 100.0]);
    let b = backdrop_layer_from_object(&objs[1], objs[1].image.as_deref().unwrap()).unwrap();
    assert_eq!(b.backdrop, Some(Backdrop::FullScene));
    assert_eq!(
        [b.size[0], b.size[1]],
        [0.0, 0.0],
        "sized to the scene later"
    );
    assert!(backdrop_layer_from_object(&objs[2], "models/util/projectlayer.json").is_some());
    assert!(backdrop_layer_from_object(&objs[3], "models/other.json").is_none());
}

/// Untextured mesh materials are common in real content
/// (`"textures": [null,null,null]`), and their flat `color` is the only
/// thing that keeps the mesh from drawing plain white.
/// `maxwidth`/`maxrows` are only active when their `limit*` checkbox is
/// on. WE keeps the numbers around while the box is unchecked, so applying
/// them unconditionally wrapped text that should run free — 3091474852's
/// day label wrapped at 500px despite `limitwidth: false`.
/// `disablepropagation` severs transform inheritance while leaving the
/// `parent` link intact for grouping/visibility.
#[test]
fn disablepropagation_severs_transform_inheritance() {
    let objs: Vec<SceneObject> = serde_json::from_str(
        r#"[{"id": 1, "origin": "10 0 0"},
            {"id": 2, "origin": "1 0 0", "parent": 1},
            {"id": 3, "origin": "1 0 0", "parent": 1, "disablepropagation": true}]"#,
    )
    .expect("valid objects");
    let id_to_idx: HashMap<i64, usize> = objs
        .iter()
        .enumerate()
        .filter_map(|(i, o)| o.id.map(|d| (d, i)))
        .collect();
    let name_to_idx: HashMap<&str, usize> = HashMap::new();
    assert_eq!(parent_index(&objs[1], &id_to_idx, &name_to_idx), Some(0));
    assert_eq!(
        parent_index(&objs[2], &id_to_idx, &name_to_idx),
        None,
        "disablepropagation must stop transform inheritance"
    );
}

/// The object-attachment API's `thisLayer.disablepropagation()` (see
/// `engine::sim::script::OBJECT_ATTACHMENT_API_JS`) is this same
/// `disablepropagation` rule, just evaluated live per-frame instead of
/// baked once from scene.json — `TransformGraph::world`'s `disabled`
/// slice must sever object `i`'s *own* inheritance from its parent
/// (matching `parent_index` above exactly), not affect what `i`'s
/// children inherit from `i`.
#[test]
fn live_disablepropagation_severs_only_the_flagged_objects_own_inheritance() {
    // 0 (root, moves) -> 1 (child of 0) -> 2 (grandchild, child of 1)
    let graph = TransformGraph {
        parent: vec![None, Some(0), Some(1)],
        local: vec![Xform::IDENTITY; 3],
        name: vec![None, None, None],
        id: vec![None, None, None],
    };
    let locals = vec![
        Xform {
            t: [10.0, 0.0, 0.0],
            r: [0.0; 3],
            s: [1.0; 3],
        },
        Xform {
            t: [1.0, 0.0, 0.0],
            r: [0.0; 3],
            s: [1.0; 3],
        },
        Xform {
            t: [1.0, 0.0, 0.0],
            r: [0.0; 3],
            s: [1.0; 3],
        },
    ];

    let worlds = graph.world(&locals, &[false, false, false]);
    assert_eq!(
        worlds[2].t,
        [12.0, 0.0, 0.0],
        "baseline: full chain composes"
    );

    // Sever object 1's own inheritance from object 0.
    let worlds = graph.world(&locals, &[false, true, false]);
    assert_eq!(worlds[0].t, [10.0, 0.0, 0.0], "object 0 is untouched");
    assert_eq!(
        worlds[1].t,
        [1.0, 0.0, 0.0],
        "object 1 must stop inheriting object 0's transform"
    );
    assert_eq!(
        worlds[2].t,
        [2.0, 0.0, 0.0],
        "object 2 still inherits normally from (now-severed) object 1"
    );
}

/// `depthtest: "disabled"` on the object overrides the material pass.
#[test]
fn object_depthtest_overrides_material() {
    let read = |o: &SceneObject, mat: Option<&str>| -> bool {
        o.depthtest
            .as_ref()
            .and_then(|v| v.as_str())
            .or(mat)
            .map(|m| m != "disabled")
            .unwrap_or(true)
    };
    let plain: SceneObject = serde_json::from_str("{}").expect("valid");
    let off: SceneObject = serde_json::from_str(r#"{"depthtest": "disabled"}"#).expect("valid");
    assert!(read(&plain, None), "default is depth-tested");
    assert!(!read(&plain, Some("disabled")), "material pass applies");
    assert!(!read(&off, Some("enabled")), "object wins over material");
}

#[test]
fn wrap_limits_respect_their_gates() {
    let obj = |lw: bool| -> SceneObject {
        serde_json::from_value(serde_json::json!({
            "text": "SUNDAY", "maxwidth": 500.0, "maxrows": 1,
            "limitwidth": lw, "limitrows": lw,
        }))
        .expect("valid object")
    };
    let active = |o: &SceneObject| {
        o.maxwidth
            .as_ref()
            .and_then(crate::engine::model::scene::parse_value_f32)
            .filter(|w| *w > 0.0)
            .filter(|_| {
                o.limitwidth
                    .as_ref()
                    .and_then(crate::engine::model::scene::parse_value_bool)
                    .unwrap_or(false)
            })
            .is_some()
    };
    assert!(active(&obj(true)), "gate on: maxwidth applies");
    assert!(!active(&obj(false)), "gate off: maxwidth must be ignored");
}

#[test]
fn material_constant_color_reads_flat_color() {
    // `color` wins over the separate (usually white) `Color` tint.
    let m: serde_json::Value = serde_json::from_str(
        r#"{"textures":[null],"constantshadervalues":
           {"Color":"1.0 1.0 1.0","color":"0.6 0.0 0.2"}}"#,
    )
    .expect("valid json");
    assert_eq!(pass_constant_color(&m), Some([153, 0, 51]));
    assert_eq!(pass_constant_color(&serde_json::json!({})), None);
}

fn solid_particle_config(color: &str) -> particle::ParticleConfig {
    let json = format!(
        r#"{{
            "maxcount": 2,
            "emitter": [{{"name":"box","rate":1000,"distancemin":"0 0 0","distancemax":"0 0 0"}}],
            "initializer": [
                {{"id":1,"name":"lifetimerandom","min":100,"max":100}},
                {{"id":2,"name":"sizerandom","min":300,"max":300}},
                {{"id":3,"name":"velocityrandom","min":"0 0 0","max":"0 0 0"}},
                {{"id":4,"name":"colorrandom","min":"{color}","max":"{color}"}},
                {{"id":5,"name":"alpharandom","min":1,"max":1}}
            ]
        }}"#
    );
    serde_json::from_str(&json).expect("should parse")
}

/// Particles should interleave with image layers by `order_index` in
/// true scene z-order, not always draw on top of every image — build a
/// scene with [particle(order 0, blue), image(order 1, opaque red),
/// particle(order 2, green)] and confirm the final canvas shows the
/// blue particle fully hidden under the opaque red layer, with the green
/// particle visible on top of it.
#[test]
fn particles_and_images_interleave_by_order_index() {
    let scene: Scene = serde_json::from_str("{}").unwrap();
    let red_image = RgbaImage::from_pixel(100, 100, image::Rgba([255, 0, 0, 255]));

    let layers = vec![Layer {
        backdrop: None,
        local_origin: [0.0; 3],
        local_scale: [1.0; 3],
        local_angles: [0.0; 3],
        name: "red".to_string(),
        image: red_image,
        extra_frames: Vec::new(),
        frame_duration_ms: 0,
        origin: [50.0, 50.0, 0.0],
        size: [100.0, 100.0, 0.0],
        scale: [1.0, 1.0, 1.0],
        parallax_depth: [0.0, 0.0],
        angle: 0.0,
        angles: [0.0, 0.0, 0.0],
        blend_mode: 0,
        alpha: 1.0,
        color: [1.0, 1.0, 1.0],
        brightness: 1.0,
        copybackground: false,
        no_interpolation: true,
        clamp_uvs: false,
        puppet: None,
        video: None,
        text_dynamic: None,
        order_index: 1,
        object_id: None,
        visible_base: true,
    }];

    let particle_layers = vec![
        ParticleLayer {
            local_origin: [0.0; 3],
            name: "blue_particles".to_string(),
            origin: [50.0, 50.0, 0.0],
            parallax_depth: [0.0, 0.0],
            config: solid_particle_config("0 0 255"),
            overrides: None,
            sprite_texture: None,
            additive_blend: false,
            order_index: 0,
            children: Vec::new(),
        },
        ParticleLayer {
            local_origin: [0.0; 3],
            name: "green_particles".to_string(),
            origin: [50.0, 50.0, 0.0],
            parallax_depth: [0.0, 0.0],
            config: solid_particle_config("0 255 0"),
            overrides: None,
            sprite_texture: None,
            additive_blend: false,
            order_index: 2,
            children: Vec::new(),
        },
    ];

    let resolved = ResolvedScene {
        transform_graph: build_transform_graph(&scene),
        width: 100,
        height: 100,
        layers,
        particle_layers,
        mesh3d_layers: Vec::new(),
        scene,
    };

    let canvas = resolved.render();
    let px = canvas.get_pixel(50, 50);
    assert!(
        px[2] < 20,
        "blue particle (order 0) should be fully hidden under the opaque red layer (order 1), got {px:?}"
    );
    assert!(
        px[1] > px[0],
        "green particle (order 2) should be visible on top of the red layer, got {px:?}"
    );
}

#[test]
fn parallax_depth_number_broadcasts_to_both_axes() {
    let v = serde_json::json!(0.5);
    assert_eq!(parse_parallax_depth(&v), [0.5, 0.5]);
}

#[test]
fn parallax_depth_vec2_string_keeps_axes_independent() {
    let v = serde_json::json!("0.5 0.0");
    assert_eq!(parse_parallax_depth(&v), [0.5, 0.0]);
}

#[test]
fn parallax_depth_user_wrapped_value_unwraps() {
    let v = serde_json::json!({"user": "ui_editor_properties_depth", "value": "0.25 0.75"});
    assert_eq!(parse_parallax_depth(&v), [0.25, 0.75]);
}

#[test]
fn alignment_none_produces_no_offset() {
    assert_eq!(alignment_offset(None, [100.0, 200.0]), [0.0, 0.0]);
}

#[test]
fn alignment_top_shifts_down_by_half_height() {
    assert_eq!(alignment_offset(Some("top"), [100.0, 200.0]), [0.0, -100.0]);
}

#[test]
fn alignment_bottom_shifts_up_by_half_height() {
    assert_eq!(
        alignment_offset(Some("bottom"), [100.0, 200.0]),
        [0.0, 100.0]
    );
}

#[test]
fn alignment_left_shifts_right_by_half_width() {
    assert_eq!(alignment_offset(Some("left"), [100.0, 200.0]), [50.0, 0.0]);
}

#[test]
fn alignment_right_shifts_left_by_half_width() {
    assert_eq!(
        alignment_offset(Some("right"), [100.0, 200.0]),
        [-50.0, 0.0]
    );
}

#[test]
fn alignment_combines_vertical_and_horizontal() {
    assert_eq!(
        alignment_offset(Some("top left"), [100.0, 200.0]),
        [50.0, -100.0]
    );
}

#[test]
fn alignment_top_and_bottom_together_only_honors_top() {
    // Matches the reference's if/else-if: mutually exclusive per axis.
    assert_eq!(
        alignment_offset(Some("top bottom"), [100.0, 200.0]),
        [0.0, -100.0]
    );
}

#[test]
fn alignment_is_case_insensitive() {
    assert_eq!(alignment_offset(Some("TOP"), [100.0, 200.0]), [0.0, -100.0]);
}
