use std::collections::HashMap;

use wp_engine::engine::properties::{parse_property_arg, SceneProperties};
use wp_engine::engine::scene::Scene;
use wp_engine::engine::sim::camera_dynamics::CameraDynamics;

fn props_from_project(project: serde_json::Value, overrides: &[(&str, &str)]) -> SceneProperties {
    // Build via a temp dir so the public loader path is exercised.
    // Unique per call — tests run in parallel.
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "wp_engine_props_test_{}_{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("project.json"), project.to_string()).unwrap();
    let mut props = SceneProperties::from_project_dir(&dir);
    let map: HashMap<String, String> = overrides
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    props.apply_overrides(map);
    let _ = std::fs::remove_dir_all(&dir);
    props
}

fn sample_project() -> serde_json::Value {
    serde_json::json!({
        "title": "test",
        "general": {
            "properties": {
                "rate": { "type": "slider", "text": "Rate", "value": 0.25, "min": 0, "max": 1 },
                "tintcolor": { "type": "color", "text": "Tint", "value": "1 0.5 0" },
                "glow": { "type": "bool", "text": "Glow", "value": true }
            }
        }
    })
}

#[test]
fn resolves_user_references_in_scene_json() {
    let props = props_from_project(sample_project(), &[]);
    let mut scene = serde_json::json!({
        "general": {
            "bloomstrength": { "user": "rate", "value": 2.0 }
        },
        "objects": [
            { "color": { "user": "tintcolor", "value": "1 1 1" } }
        ]
    });
    props.resolve_scene_json(&mut scene);

    assert_eq!(scene["general"]["bloomstrength"]["value"], 0.25);
    assert_eq!(scene["objects"][0]["color"]["value"], "1 0.5 0");
}

#[test]
fn cli_overrides_replace_project_defaults() {
    let props = props_from_project(sample_project(), &[("rate", "0.9"), ("glow", "0")]);
    let mut scene = serde_json::json!({
        "a": { "user": "rate", "value": 0.0 },
        "b": { "user": "glow", "value": true }
    });
    props.resolve_scene_json(&mut scene);

    assert_eq!(scene["a"]["value"], 0.9);
    assert_eq!(scene["b"]["value"], false);
}

/// Conditional references evaluate `stringify(property) == condition` into
/// a boolean (the reference's DynamicValue equality, DynamicValue.cpp:176)
/// — they never copy the raw property value like plain references do, and
/// a non-matching condition must leave gated layers at `false` rather than
/// letting a truthy combo value force them visible.
#[test]
fn conditional_user_reference_evaluates_condition_equality() {
    let props = props_from_project(sample_project(), &[]);
    let mut scene = serde_json::json!({
        "matched": { "user": { "name": "rate", "condition": "0.25" }, "value": false },
        "unmatched": { "user": { "name": "rate", "condition": "3" }, "value": false }
    });
    props.resolve_scene_json(&mut scene);
    assert_eq!(scene["matched"]["value"], true);
    assert_eq!(scene["unmatched"]["value"], false);
}

#[test]
fn parse_property_arg_forms() {
    assert_eq!(
        parse_property_arg("speed=0.5"),
        ("speed".into(), "0.5".into())
    );
    assert_eq!(parse_property_arg("glow"), ("glow".into(), "1".into()));
}

// ── Camera dynamics ───────────────────────────────────────────────────────────

fn scene_with_general(general: serde_json::Value) -> Scene {
    let json = serde_json::json!({ "general": general, "objects": [] });
    Scene::from_value(json).unwrap()
}

#[test]
fn camera_dynamics_parse_user_wrapped_values() {
    let scene = scene_with_general(serde_json::json!({
        "camerashake": { "user": "shake", "value": true },
        "camerashakeamplitude": { "value": 1.5 },
        "camerafade": true,
        "cameraparallax": true,
        "cameraparallaxamount": 2.0
    }));
    let dynamics = CameraDynamics::from_scene(&scene);
    assert!(dynamics.shake_enabled);
    assert_eq!(dynamics.shake_amplitude, 1.5);
    assert!(dynamics.fade_enabled);
    assert!(dynamics.parallax_enabled);
    assert_eq!(dynamics.parallax_amount, 2.0);
    assert!(dynamics.is_active());
}

#[test]
fn fade_ramps_from_zero_to_one() {
    let scene = scene_with_general(serde_json::json!({ "camerafade": true }));
    let mut dynamics = CameraDynamics::from_scene(&scene);
    let start = dynamics.update(0.0, 0.0, [0.5, 0.5]);
    let end = dynamics.update(10.0, 0.1, [0.5, 0.5]);
    assert!(
        start.fade < 0.05,
        "fade should start near 0, got {}",
        start.fade
    );
    assert!((end.fade - 1.0).abs() < 1e-6, "fade should settle at 1");
}

#[test]
fn shake_produces_bounded_time_varying_offset() {
    let scene = scene_with_general(serde_json::json!({
        "camerashake": true,
        "camerashakeamplitude": 1.0,
        "camerashakespeed": 2.0
    }));
    let mut dynamics = CameraDynamics::from_scene(&scene);
    let a = dynamics.update(0.3, 0.016, [0.5, 0.5]).shake_offset;
    let b = dynamics.update(0.9, 0.016, [0.5, 0.5]).shake_offset;
    assert!(a != b, "shake offset should vary over time");
    for v in a.iter().chain(b.iter()) {
        assert!(v.abs() < 0.05, "shake offset should stay subtle, got {v}");
    }
}

#[test]
fn parallax_follows_mouse_and_rests_at_center() {
    let scene = scene_with_general(serde_json::json!({
        "cameraparallax": true,
        "cameraparallaxamount": 1.0,
        "cameraparallaxmouseinfluence": 1.0
    }));
    let mut dynamics = CameraDynamics::from_scene(&scene);
    let centered = dynamics.update(0.1, 0.016, [0.5, 0.5]);
    assert_eq!(centered.parallax_displacement, [0.0, 0.0]);
    let moved = dynamics.update(0.2, 0.016, [1.0, 0.5]);
    assert!(moved.parallax_displacement[0] > 0.4);
    assert_eq!(moved.parallax_displacement[1], 0.0);
}

#[test]
fn static_scene_has_inactive_dynamics() {
    let scene = scene_with_general(serde_json::json!({}));
    let dynamics = CameraDynamics::from_scene(&scene);
    assert!(!dynamics.is_active());
    let frame = CameraDynamics::from_scene(&scene).update(1.0, 0.016, [0.5, 0.5]);
    assert_eq!(frame.fade, 1.0);
    assert_eq!(frame.shake_offset, [0.0, 0.0]);
}

// Regression test for the copybackground bug: objects flagged
// `copybackground` still carry a real image and must load it (the reference
// engine ignores the flag). Uses a real workshop scene when available.
#[test]
fn copybackground_objects_load_their_image() {
    let dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".local/share/Steam/steamapps/workshop/content/431960/1275921440");
    if !dir.exists() {
        return; // fixture not installed on this machine
    }
    let resolved = wp_engine::engine::ResolvedScene::from_directory(&dir).unwrap();
    let layer = resolved
        .layers
        .iter()
        .find(|l| l.copybackground)
        .expect("scene should have a copybackground layer");
    let c = layer
        .image
        .get_pixel(layer.image.width() / 2, layer.image.height() / 2);
    assert_ne!(
        [c[0], c[1], c[2]],
        [140, 140, 140],
        "copybackground layer must load its real texture, not the placeholder"
    );
}
