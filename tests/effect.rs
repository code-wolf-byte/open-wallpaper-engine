use wp_engine::audio::cpu::effect::{HandwrittenEffectFallback, SceneEffect};

#[test]
fn recognizes_scroll_and_spin_fallbacks() {
    assert_eq!(
        HandwrittenEffectFallback::from_label("effects/scroll.json"),
        Some(HandwrittenEffectFallback::Scroll)
    );
    assert_eq!(
        HandwrittenEffectFallback::from_label("effects/spin.json"),
        Some(HandwrittenEffectFallback::Spin)
    );
}

#[test]
fn constructs_scroll_and_spin_effects() {
    let values = serde_json::json!({
        "speed": 0.25,
        "strength": 0.5,
    });

    assert!(matches!(
        SceneEffect::from_effect_name("scroll", &values, 1.0),
        Some(SceneEffect::Scroll { .. })
    ));
    assert!(matches!(
        SceneEffect::from_effect_name("spin", &values, 1.0),
        Some(SceneEffect::Spin { .. })
    ));
}
