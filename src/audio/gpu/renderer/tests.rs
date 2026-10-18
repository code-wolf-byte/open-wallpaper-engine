use super::*;

/// `gpu_shaders.wgsl` must parse and validate as WGSL on its own — the
/// only check in this crate that catches a shader syntax/type error
/// before it surfaces at runtime as an opaque wgpu pipeline-creation
/// failure the first time a scene that uses that pass actually loads.
/// Uses naga directly (no GPU adapter needed — pure frontend/validator).
#[test]
fn gpu_shaders_wgsl_parses_and_validates() {
    let module =
        naga::front::wgsl::parse_str(SHADER_SRC).expect("gpu_shaders.wgsl failed to parse");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("gpu_shaders.wgsl failed validation");
}

fn quad3d_test_camera() -> crate::audio::sim::camera3d::PerspectiveCamera {
    let scene: crate::audio::model::scene::Scene = serde_json::from_str(
        r#"{"camera": {"eye": "0 0 10", "center": "0 0 0", "up": "0 1 0"},
            "general": {"orthogonalprojection": null, "fov": 60.0}}"#,
    )
    .unwrap();
    crate::audio::sim::camera3d::PerspectiveCamera::from_scene(&scene, 1.0).unwrap()
}

/// A camera-facing quad (no rotation, centered on the view axis) must
/// project to a symmetric rectangle: opposite corners mirror each other
/// around the NDC origin. This is the case `project_quad_ndc`'s AABB
/// collapse already gets right — `project_quad_corners` must agree with
/// it exactly here, only diverging once the quad is off-axis/rotated.
#[test]
fn project_quad_corners_symmetric_for_camera_facing_quad() {
    let cam = quad3d_test_camera();
    let corners = project_quad_corners(&cam, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.0, 2.0])
        .expect("camera-facing quad in front of the eye must project");
    // Order: bottom-left, bottom-right, top-right, top-left.
    assert!((corners[0][0] + corners[1][0]).abs() < 1e-5, "{corners:?}");
    assert!((corners[0][1] + corners[3][1]).abs() < 1e-5, "{corners:?}");
    assert!(corners[0][0] < corners[1][0], "left must be left of right");
    assert!(corners[0][1] < corners[2][1], "bottom must be below top");
}

/// The whole point of this function: a quad rotated hard off-axis
/// projects to a genuine trapezoid, not a rectangle. Rotating around Y
/// swings one vertical edge toward the camera and the other away from
/// it (left edge: x=-2 → z=+1.73, closer to the eye at z=10; right
/// edge: x=+2 → z=-1.73, farther away) — the closer edge must span a
/// taller NDC range than the farther one. `project_quad_ndc`'s AABB
/// would hide this entirely (both edges would just extend its bounding
/// box, indistinguishable from a same-size rectangle).
#[test]
fn project_quad_corners_is_a_true_trapezoid_when_rotated() {
    let cam = quad3d_test_camera();
    let corners = project_quad_corners(
        &cam,
        [0.0, 0.0, 0.0],
        [0.0, 60f32.to_radians(), 0.0],
        [4.0, 2.0],
    )
    .expect("quad must still project in front of the eye");
    // Left edge: corner0 (bottom-left) to corner3 (top-left).
    let left_height = (corners[3][1] - corners[0][1]).abs();
    // Right edge: corner1 (bottom-right) to corner2 (top-right).
    let right_height = (corners[2][1] - corners[1][1]).abs();
    assert!(
        left_height > right_height + 1e-4,
        "expected the closer (left) edge to project taller than the \
         farther (right) edge: left={left_height} right={right_height}"
    );
}

/// A corner behind the eye plane must cull the whole quad — same rule
/// `project_quad_ndc` uses, kept consistent rather than drawing a
/// partially-degenerate quad.
#[test]
fn project_quad_corners_none_when_behind_camera() {
    let cam = quad3d_test_camera();
    // Eye is at z=10 looking toward the origin; z=20 is behind it.
    assert!(project_quad_corners(&cam, [0.0, 0.0, 20.0], [0.0, 0.0, 0.0], [4.0, 2.0]).is_none());
}

/// Same rule as `hardcoded_effect_params_are_uniform_sized` below, for the
/// mesh3d lighting UBO: the Rust-side byte length must match WGSL's
/// std140 struct size (a 16-byte multiple) and `mesh3d_lighting_bytes`
/// must actually produce that many bytes, or wgpu rejects the mesh3d bind
/// group at draw time.
#[test]
fn mesh3d_lighting_bytes_are_uniform_sized_and_match_const() {
    assert_eq!(MESH3D_LIGHTING_BYTES_LEN % 16, 0);
    let scene: crate::audio::model::scene::Scene = serde_json::from_str(
        r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"}, "general": {"orthogonalprojection": null}}"#,
    )
    .unwrap();
    let cam = crate::audio::sim::camera3d::PerspectiveCamera::from_scene(&scene, 1.0).unwrap();

    // The common case: no shadow-casting lights at all.
    let bytes = mesh3d_lighting_bytes(&scene, &cam, &[], &[], None);
    assert_eq!(bytes.len(), MESH3D_LIGHTING_BYTES_LEN);

    // One shadow-casting light occupying slot 0 — same length either way,
    // since the shadow arrays are fixed-size regardless of how many
    // slots are actually in use.
    let shadow_slots = vec![Some(0)];
    let shadow_slot_data = vec![(
        crate::audio::sim::camera3d::identity(),
        [0.0, 0.0, 1.0, 1.0],
    )];
    let bytes = mesh3d_lighting_bytes(&scene, &cam, &shadow_slots, &shadow_slot_data, None);
    assert_eq!(bytes.len(), MESH3D_LIGHTING_BYTES_LEN);
}

/// `cookie: Some((light_index, view_proj))` must write `flags.y ==
/// light_index + 1` and pack `view_proj` into the trailing
/// `cookie_view_proj` field (the last `MESH3D_MAX_SHADOW_LIGHTS` shadow
/// mat4s + the `MESH3D_MAX_SHADOW_LIGHTS` uv-rect vec4s precede it —
/// see `MESH3D_LIGHTING_BYTES_LEN`'s own layout doc).
#[test]
fn mesh3d_lighting_bytes_encodes_cookie_flag_and_matrix() {
    let scene: crate::audio::model::scene::Scene = serde_json::from_str(
        r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"}, "general": {"orthogonalprojection": null}}"#,
    )
    .unwrap();
    let cam = crate::audio::sim::camera3d::PerspectiveCamera::from_scene(&scene, 1.0).unwrap();

    let mut vp = crate::audio::sim::camera3d::identity();
    vp[3][0] = 42.0; // an arbitrary, recognizable value to round-trip
    let bytes = mesh3d_lighting_bytes(&scene, &cam, &[], &[], Some((2, vp)));

    let flags_y = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
    assert_eq!(flags_y, 3.0, "flags.y must be light_index + 1");

    let cookie_mat_offset = MESH3D_LIGHTING_BYTES_LEN - 64;
    let read_f32 = |i: usize| {
        f32::from_le_bytes(
            bytes[cookie_mat_offset + i * 4..cookie_mat_offset + i * 4 + 4]
                .try_into()
                .unwrap(),
        )
    };
    // Column 3, row 0 (the value we perturbed) is float index 12.
    assert_eq!(
        read_f32(12),
        42.0,
        "cookie_view_proj must round-trip the matrix"
    );
}

/// A spot light must (a) count as renderable (`flags.x == 1.0` — the
/// bug fixed alongside GPU spot support: `Scene::lights()` producing a
/// `Light::Spot` must not silently fall back to the unlit branch) and
/// (b) write a non-zero direction into its `light_spot` slot, so
/// `mesh3d_spot_factor` in the shader doesn't read it as "not a spot".
#[test]
fn mesh3d_lighting_bytes_encodes_spot_direction() {
    let cam_scene: crate::audio::model::scene::Scene = serde_json::from_str(
        r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"}, "general": {"orthogonalprojection": null}}"#,
    )
    .unwrap();
    let cam = crate::audio::sim::camera3d::PerspectiveCamera::from_scene(&cam_scene, 1.0).unwrap();

    let scene: crate::audio::model::scene::Scene = serde_json::from_str(
        r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"},
            "general": {"orthogonalprojection": null},
            "objects": [
                {"id": 1, "light": true, "origin": "0 0 0", "radius": 100,
                 "intensity": 0.8, "color": "1 1 1",
                 "innercone": 20, "outercone": 45}
            ]}"#,
    )
    .unwrap();
    let bytes = mesh3d_lighting_bytes(&scene, &cam, &[], &[], None);

    // flags.x is the buffer's first f32.
    let flags_x = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(
        flags_x, 1.0,
        "a spot-only scene must still engage the lit branch"
    );

    // light_spot[0] starts after 5 leading vec4s + 2 MESH3D_MAX_LIGHTS
    // vec4 arrays (positions, colors).
    let spot0_offset = 16 * 5 + 16 * MESH3D_MAX_LIGHTS * 2;
    let read_f32 = |i: usize| {
        f32::from_le_bytes(
            bytes[spot0_offset + i * 4..spot0_offset + i * 4 + 4]
                .try_into()
                .unwrap(),
        )
    };
    let dir = [read_f32(0), read_f32(1), read_f32(2)];
    let len2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
    assert!(len2 > 0.25, "spot direction must not read as zero: {dir:?}");
    assert!(
        read_f32(3) >= 1.0,
        "exponent must be clamped to at least 1.0"
    );
}

/// A directional light must flag itself via `light_pos[0].w == -1.0`
/// (the sentinel `fs_mesh3d`'s light loop branches on) with a unit-length
/// view-space direction in xyz, and still engage the lit branch.
#[test]
fn mesh3d_lighting_bytes_encodes_directional_sentinel_and_direction() {
    let cam_scene: crate::audio::model::scene::Scene = serde_json::from_str(
        r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"}, "general": {"orthogonalprojection": null}}"#,
    )
    .unwrap();
    let cam = crate::audio::sim::camera3d::PerspectiveCamera::from_scene(&cam_scene, 1.0).unwrap();

    let scene: crate::audio::model::scene::Scene = serde_json::from_str(
        r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"},
            "general": {"orthogonalprojection": null},
            "objects": [
                {"id": 1, "light": "ldirectional", "angles": "0 1.5707963 0",
                 "intensity": 0.6, "color": "1 1 1"}
            ]}"#,
    )
    .unwrap();
    let bytes = mesh3d_lighting_bytes(&scene, &cam, &[], &[], None);

    let flags_x = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(
        flags_x, 1.0,
        "a directional-only scene must still engage the lit branch"
    );

    // light_pos[0] starts right after the 5 leading vec4s.
    let pos0_offset = 16 * 5;
    let read_f32 = |i: usize| {
        f32::from_le_bytes(
            bytes[pos0_offset + i * 4..pos0_offset + i * 4 + 4]
                .try_into()
                .unwrap(),
        )
    };
    let dir = [read_f32(0), read_f32(1), read_f32(2)];
    let len2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
    assert!(
        (len2 - 1.0).abs() < 1e-4,
        "direction must be unit-length: {dir:?}"
    );
    assert_eq!(read_f32(3), -1.0, "w must carry the directional sentinel");
}

/// A tube light must flag itself via `light_pos[0].w == -3.0` (the
/// sentinel `fs_mesh3d`'s light loop branches on), with endpoint A in
/// `light_pos[0].xyz` and endpoint B in `light_spot[0].xyz` — not a
/// direction, so its magnitude should reflect the real endpoint spacing,
/// not be unit-length like a spot's.
#[test]
fn mesh3d_lighting_bytes_encodes_tube_sentinel_and_endpoints() {
    let cam_scene: crate::audio::model::scene::Scene = serde_json::from_str(
        r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"}, "general": {"orthogonalprojection": null}}"#,
    )
    .unwrap();
    let cam = crate::audio::sim::camera3d::PerspectiveCamera::from_scene(&cam_scene, 1.0).unwrap();

    let scene: crate::audio::model::scene::Scene = serde_json::from_str(
        r#"{"camera": {"eye": "0 0 10", "center": "0 0 0"},
            "general": {"orthogonalprojection": null},
            "objects": [
                {"id": 1, "light": "ltube", "origin": "0 0 0",
                 "controlpoint": "0 0 -4", "intensity": 0.9, "color": "1 1 1"}
            ]}"#,
    )
    .unwrap();
    let bytes = mesh3d_lighting_bytes(&scene, &cam, &[], &[], None);

    let flags_x = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(
        flags_x, 1.0,
        "a tube-only scene must still engage the lit branch"
    );

    let pos0_offset = 16 * 5;
    let spot0_offset = 16 * 5 + 16 * MESH3D_MAX_LIGHTS * 2;
    let read_f32 = |offset: usize, i: usize| {
        f32::from_le_bytes(
            bytes[offset + i * 4..offset + i * 4 + 4]
                .try_into()
                .unwrap(),
        )
    };
    assert_eq!(
        read_f32(pos0_offset, 3),
        -3.0,
        "w must carry the tube sentinel"
    );
    let endpoint_b = [
        read_f32(spot0_offset, 0),
        read_f32(spot0_offset, 1),
        read_f32(spot0_offset, 2),
    ];
    let len2 = endpoint_b[0] * endpoint_b[0]
        + endpoint_b[1] * endpoint_b[1]
        + endpoint_b[2] * endpoint_b[2];
    assert!(
        len2 > 1.0,
        "endpoint B should be a real view-space position, not a unit vector: {endpoint_b:?}"
    );
}

/// WGSL rounds every uniform-address-space struct up to a 16-byte
/// multiple, so a param buffer that isn't one can never match its shader
/// struct — wgpu rejects the bind group at draw time ("Buffer is bound
/// with size N where the shader expects M") and the wallpaper dies.
#[test]
fn hardcoded_effect_params_are_uniform_sized() {
    let vals = ShaderVals::new();
    for name in HARDCODED_EFFECTS {
        let n = make_effect_params(name, 0.0, &vals).len();
        assert_eq!(n % 16, 0, "{name} params are {n} bytes, not a 16-multiple");
    }
}

/// Hardcoded kernels must find a value the scene keyed by UI label
/// ("ui_editor_properties_pulse_amount") as well as by material key.
#[test]
fn hardcoded_params_resolve_label_keyed_values() {
    let mut vals = ShaderVals::new();
    insert_value_components(
        &mut vals,
        "ui_editor_properties_pulse_amount",
        &serde_json::json!(0.25),
    );
    insert_value_components(
        &mut vals,
        "ui_editor_properties_pulse_bounds",
        &serde_json::json!("0.2 0.8"),
    );
    let bytes = make_effect_params("pulse", 0.0, &vals);
    let f = |i: usize| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    assert_eq!(f(2), 0.25, "amount");
    assert_eq!(f(5), 0.2, "bounds low");
    assert_eq!(f(6), 0.8, "bounds high");
}

/// vec2 constants (audiobounds, texture scales, ...) must expose their
/// components like vec3 colors do, or the packer silently falls back to
/// the shader default and the artist's tuning is lost.
#[test]
fn vector_constants_expose_components_at_every_width() {
    let mut map = ShaderVals::new();
    insert_value_components(&mut map, "audiobounds", &serde_json::json!("0.25 0.75"));
    assert_eq!(map.get("audiobounds_x"), Some(&0.25));
    assert_eq!(map.get("audiobounds_y"), Some(&0.75));
    assert_eq!(map.get("audiobounds_r"), Some(&0.25));

    insert_value_components(&mut map, "tintlow", &serde_json::json!("1 0 0"));
    assert_eq!(map.get("tintlow_b"), Some(&0.0));
    // vec3s keep their implicit opaque alpha.
    assert_eq!(map.get("tintlow_a"), Some(&1.0));
}

mod hit_tests {
    use super::*;

    #[test]
    fn hit_test_geometry() {
        // A 200x100 px quad centred at (100, 50) px in a 1000x500 output.
        let rect = [0.2, 0.2, 0.2, 0.2];
        let q = quad_corners_px(rect, 0.0, 1000.0, 500.0);
        assert!(point_in_convex_quad([100.0, 50.0], &q));
        assert!(point_in_convex_quad([199.0, 74.0], &q));
        assert!(!point_in_convex_quad([201.0, 50.0], &q));
        // Rotated 90°: the long axis is now vertical.
        let q = quad_corners_px(rect, std::f32::consts::FRAC_PI_2, 1000.0, 500.0);
        assert!(point_in_convex_quad([100.0, 140.0], &q));
        assert!(!point_in_convex_quad([190.0, 50.0], &q));
    }
    /// Debug aid: run a scene's scripts headlessly (no GPU) for a few
    /// frames and print every script log line. `WP_SCRIPT_SCENE=<id|dir>`.
    #[test]
    #[ignore]
    fn run_scene_scripts() {
        let Ok(target) = std::env::var("WP_SCRIPT_SCENE") else {
            return;
        };
        let dir = crate::workshop::find_item_dir(&target, None)
            .unwrap_or_else(|| std::path::PathBuf::from(&target));
        let resolved = crate::audio::model::resolve::ResolvedScene::from_directory(&dir).unwrap();
        let raw_scene: serde_json::Value = crate::formats::assets::AssetStore::from_directory(&dir)
            .ok()
            .and_then(|a| a.scene_json().ok())
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        let mut host = build_script_host(
            &dir,
            &resolved.scene,
            &resolved.transform_graph,
            [resolved.width, resolved.height],
            &raw_scene,
        );
        for f in 0..30 {
            host.frame(f as f32 * 0.5, 0.5, 0.5);
            let c = host.collect();
            if f == 0 {
                eprintln!(
                    "frame 0: {} objects changed, {} commands",
                    c.objects.len(),
                    c.commands.len()
                );
            }
        }
    }

    /// A real intro sequence (3453730450): layers keyed to fade in over the
    /// first ~13 s start hidden and end fully visible.
    #[test]
    #[ignore]
    fn keyframed_alpha_fades_in() {
        let Some(dir) = crate::workshop::find_item_dir("3453730450", None) else {
            eprintln!("workshop item 3453730450 not installed; skipping");
            return;
        };
        let mut scene = GpuSceneInstance::open(&dir).expect("scene loads");
        let animated: Vec<usize> = scene
            .timelines
            .iter()
            .filter(|t| t.key == "alpha")
            .map(|t| t.object)
            .collect();
        assert!(!animated.is_empty());
        let alpha_of = |s: &GpuSceneInstance, o: usize| {
            s.layers
                .iter()
                .find(|l| l.order_index == o)
                .map(|l| l.alpha)
        };
        scene.tick_timelines(0.1);
        let early: Vec<f32> = animated
            .iter()
            .filter_map(|o| alpha_of(&scene, *o))
            .collect();
        assert!(
            early.iter().all(|a| *a < 0.05),
            "hidden at start: {early:?}"
        );
        for _ in 0..140 {
            scene.tick_timelines(0.1);
        }
        let late: Vec<f32> = animated
            .iter()
            .filter_map(|o| alpha_of(&scene, *o))
            .collect();
        assert!(
            late.iter().filter(|a| **a > 0.3).count() >= late.len() - 1,
            "faded in: {late:?}"
        );
    }

    #[test]
    fn changed_leaves_stops_at_user_bindings() {
        let a = serde_json::json!({"visible": {"user": "grid", "value": false},
            "effects": [{"passes": [{"constantshadervalues": {"speed": 1}}]}], "name": "x"});
        let b = serde_json::json!({"visible": {"user": "grid", "value": true},
            "effects": [{"passes": [{"constantshadervalues": {"speed": 2}}]}], "name": "x"});
        let mut out = Vec::new();
        changed_leaves(&a, &b, &mut Vec::new(), &mut out);
        let paths: Vec<String> = out.iter().map(|(p, _)| p.join(".")).collect();
        assert_eq!(
            paths,
            ["effects.0.passes.0.constantshadervalues.speed", "visible"]
        );
        assert_eq!(out[1].1["value"], true);
    }

    /// A user setting bound to layer visibility (2156479578: `grid` shows
    /// the Grid layer) applies live, without rebuilding the scene.
    #[test]
    #[ignore]
    fn bool_setting_toggles_a_layer_live() {
        let Some(dir) = crate::workshop::find_item_dir("2156479578", None) else {
            eprintln!("workshop item 2156479578 not installed; skipping");
            return;
        };
        let mut scene = GpuSceneInstance::open(&dir).expect("scene loads");
        let before = scene.object_visible[4];
        let flipped = if before { "0" } else { "1" };
        assert!(
            {
                let mut props = scene.live_props.props.clone();
                props.apply_overrides(HashMap::from([("grid".to_string(), flipped.to_string())]));
                scene.apply_user_properties(props, &["grid".to_string()])
            },
            "applied live"
        );
        assert_eq!(scene.object_visible[4], !before);
        scene.render();
    }

    /// End to end against a real Workshop scene with clickable layers
    /// (2348914357: clicking the radio/music buttons toggles them). Needs
    /// the item installed and a GPU, hence ignored by default.
    #[test]
    #[ignore]
    fn clicking_a_scripted_button_toggles_it() {
        let Some(dir) = crate::workshop::find_item_dir("2348914357", None) else {
            eprintln!("workshop item 2348914357 not installed; skipping");
            return;
        };
        let mut scene = GpuSceneInstance::open(&dir).expect("scene loads");
        assert!(!scene.cursor_targets.is_empty(), "scene has click targets");
        scene.render();
        let target = scene
            .layers
            .iter()
            .find(|l| scene.cursor_targets.contains(&l.order_index))
            .expect("a click target has a layer");
        let (idx, before) = (target.order_index, scene.object_visible[target.order_index]);
        let norm = [(target.rect[0] + 1.0) * 0.5, (1.0 - target.rect[1]) * 0.5];
        assert_eq!(
            scene.hit_test(norm),
            Some(idx),
            "hit test finds the target at its centre"
        );
        scene.set_mouse(norm);
        scene.render();
        scene.set_mouse_button(true);
        scene.set_mouse_button(false);
        scene.render();
        assert_eq!(
            scene.object_visible[idx], !before,
            "the click toggled the layer"
        );
    }
}
