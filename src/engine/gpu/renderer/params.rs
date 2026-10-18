//! Effect uniform packing: turning a pass's shader constants (scene values,
//! user properties, script overrides) into the std140 bytes its uniform
//! buffer expects.

use super::*;

/// Fraction of the local day in [0,1) (the reference's g_Daytime).
/// `g_Daytime`: fraction of the local day elapsed.
pub(super) fn daytime_fraction() -> f32 {
    crate::engine::sim::script::local_time_of_day()
}

pub(super) type ShaderVals = std::collections::HashMap<String, f32>;

/// Insert a constant under every spelling the packer might look up: the raw
/// key, its normalized alias (scene constants use
/// "ui_editor_properties_color_start" while asset shaders annotate materials
/// as "colorstart"), and per-component suffixes for vector values.
pub(super) fn insert_value_components(map: &mut ShaderVals, key: &str, value: &serde_json::Value) {
    let alias = |k: &str| -> String {
        k.strip_prefix("ui_editor_properties_")
            .unwrap_or(k)
            .replace('_', "")
            .to_lowercase()
    };
    let animated = crate::engine::model::json_to_animated(value);
    insert_animated_components(map, key, &alias(key), &animated);
}

pub(super) fn insert_animated_components(
    map: &mut ShaderVals,
    key: &str,
    alias: &str,
    v: &crate::engine::model::AnimatedValue,
) {
    if let Some(f) = v.as_float() {
        map.insert(key.to_string(), f);
        map.insert(alias.to_string(), f);
    }
    if let Some(comps) = v.components() {
        // Both spellings: colors are read as _r/_g/_b/_a, geometry as
        // _x/_y/_z/_w. A vec3 keeps its historical implicit _w/_a of 1.0.
        let mut named: Vec<(&str, f32)> = Vec::new();
        for (i, val) in comps.iter().enumerate() {
            named.push((["r", "g", "b", "a"][i.min(3)], *val));
            named.push((["x", "y", "z", "w"][i.min(3)], *val));
        }
        if comps.len() == 3 {
            named.push(("a", 1.0));
            named.push(("w", 1.0));
        }
        for (name, val) in named {
            map.insert(format!("{key}_{name}"), val);
            map.insert(format!("{alias}_{name}"), val);
        }
    }
}

/// Pack effect uniform values using std140 layout rules — the transpiled
/// UBO (GLSL uniform block → SPIR-V → WGSL) uses std140 offsets, so each
/// member is aligned to its own alignment, NOT padded to 16 bytes:
/// float/int align 4, vec2 align 8, vec3/vec4/matrix columns align 16.
///
/// Value resolution per uniform (mirrors CPass::setupUniforms +
/// setupShaderVariables): engine-provided uniforms first, then material/scene
/// constants, then the shader annotation's default.
pub(super) fn make_params_from_translated_typed(
    keys: &[transpiler::UniformEntry],
    engine: &EngineUniforms,
    vals: &ShaderVals,
) -> Vec<u8> {
    let mut bytes: Vec<u8> = Vec::new();

    // Resolve one uniform into up to 4 components. A scene may key the value
    // by the annotation's material key ("alpha") or by its UI label
    // ("ui_editor_properties_outline_background") — and those two are not
    // always spellings of each other, so both are tried.
    let resolve = |entry: &transpiler::UniformEntry, default: &[f32; 4]| -> [f32; 4] {
        let key = entry.key.as_str();
        if let Some(v) = engine.get(key) {
            return v;
        }
        let names = || std::iter::once(key).chain(entry.label.as_deref());
        let mut out = *default;
        if let Some(f) = names().find_map(|k| vals.get(k)) {
            out = [*f; 4];
        }
        // Per-component spellings win over the scalar broadcast.
        for (i, suffixes) in [
            &["_x", "_r"][..],
            &["_y", "_g"][..],
            &["_z", "_b"][..],
            &["_w", "_a"][..],
        ]
        .iter()
        .enumerate()
        {
            if let Some(f) = suffixes
                .iter()
                .flat_map(|s| names().map(move |k| format!("{k}{s}")))
                .find_map(|k| vals.get(&k))
            {
                out[i] = *f;
            }
        }
        out
    };

    fn align_to(bytes: &mut Vec<u8>, align: usize) {
        let rem = bytes.len() % align;
        if rem != 0 {
            bytes.resize(bytes.len() + (align - rem), 0);
        }
    }
    fn push_f32(bytes: &mut Vec<u8>, v: f32) {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    for entry in keys {
        match entry.glsl_type.as_str() {
            "vec2" => {
                let v = resolve(entry, &entry.default);
                align_to(&mut bytes, 8);
                push_f32(&mut bytes, v[0]);
                push_f32(&mut bytes, v[1]);
            }
            "vec3" => {
                let v = resolve(entry, &entry.default);
                align_to(&mut bytes, 16);
                push_f32(&mut bytes, v[0]);
                push_f32(&mut bytes, v[1]);
                push_f32(&mut bytes, v[2]);
                // std140 vec3 has size 12; a following float packs into the
                // tail 4 bytes, so do NOT pad here.
            }
            "vec4" => {
                let v = resolve(entry, &entry.default);
                align_to(&mut bytes, 16);
                for f in v {
                    push_f32(&mut bytes, f);
                }
            }
            "mat4" => {
                align_to(&mut bytes, 16);
                let identity: [f32; 16] = [
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ];
                for f in identity {
                    push_f32(&mut bytes, f);
                }
            }
            "mat3" => {
                // std140 mat3: 3 columns × (vec3 + 4 bytes pad) = 48 bytes
                align_to(&mut bytes, 16);
                let cols: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
                for col in cols {
                    for f in col {
                        push_f32(&mut bytes, f);
                    }
                    bytes.extend_from_slice(&[0u8; 4]);
                }
            }
            _ => {
                // float / int / bool / unknown: 4 bytes, align 4
                let v = resolve(entry, &entry.default);
                align_to(&mut bytes, 4);
                push_f32(&mut bytes, v[0]);
            }
        }
    }

    // Uniform buffers must be non-empty and sized to a 16-byte multiple.
    if bytes.is_empty() {
        bytes.resize(16, 0);
    }
    align_to(&mut bytes, 16);
    bytes
}

pub(super) fn make_effect_params(name: &str, time: f32, vals: &ShaderVals) -> Vec<u8> {
    // Scene `constantshadervalues` are keyed by the shader annotation's
    // *material key* ("speed", "scale", ...), not its ui_editor_properties_*
    // label — look the bare key up first and keep the label as a fallback
    // for any older call sites below that still pass the long form.
    fn get(vals: &ShaderVals, key: &str, default: f32) -> f32 {
        if let Some(v) = vals.get(key) {
            return *v;
        }
        // Scenes key a value by the annotation's material key ("amount") or by
        // its UI label ("ui_editor_properties_pulse_amount", normalized to
        // "pulseamount") — and the label is not always the material key with a
        // prefix chopped off. LABEL_ALIASES covers the pairs the corpus
        // actually uses; the translated-effect packer gets this from the
        // shader annotation itself (see make_params_from_translated_typed).
        const LABEL_ALIASES: &[(&str, &[&str])] = &[
            ("amount", &["pulseamount"]),
            ("speed", &["pulsespeed"]),
            ("phase", &["pulsephase", "timeoffset"]),
            ("bounds", &["pulsebounds"]),
            ("scale", &["ripplescale"]),
        ];
        // Vector params arrive here already split ("bounds_r"), so match on the
        // stem and carry the suffix over.
        let (stem, suffix) = match key.rfind('_') {
            Some(i) if key.len() - i <= 2 => (&key[..i], &key[i..]),
            _ => (key, ""),
        };
        if let Some(v) = LABEL_ALIASES
            .iter()
            .find(|(k, _)| *k == stem)
            .into_iter()
            .flat_map(|(_, aliases)| aliases.iter())
            .find_map(|a| vals.get(&format!("{a}{suffix}")))
        {
            return *v;
        }
        key.strip_prefix("ui_editor_properties_")
            .and_then(|bare| vals.get(bare).copied())
            .unwrap_or(default)
    }
    fn pack(floats: &[f32]) -> Vec<u8> {
        floats.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    // Keys below are the shader annotations' material keys (the names
    // scene.json constantshadervalues actually use); defaults match each
    // annotation's declared default.
    match name {
        "pulse" => pack(&[
            time,
            get(vals, "speed", 3.0),
            get(vals, "amount", 1.0),
            get(vals, "power", 1.0),
            get(vals, "phase", 0.0),
            get(vals, "bounds_r", 0.0),
            get(vals, "bounds_g", 1.0),
            get(vals, "combo_BLENDMODE", 9.0),
            get(vals, "tintlow_r", 1.0),
            get(vals, "tintlow_g", 1.0),
            get(vals, "tintlow_b", 1.0),
            get(vals, "combo_PULSECOLOR", 1.0),
            get(vals, "tinthigh_r", 1.0),
            get(vals, "tinthigh_g", 1.0),
            get(vals, "tinthigh_b", 1.0),
            get(vals, "combo_PULSEALPHA", 0.0),
            get(vals, "noisespeed", 0.5),
            get(vals, "noiseamount", 0.0),
            get(vals, "combo_AUDIOPROCESSING", 0.0),
            get(vals, "frequencymin", 0.0),
            get(vals, "frequencymax", 1.0),
            get(vals, "audiobounds_x", 0.5),
            get(vals, "audiobounds_y", 1.0),
            get(vals, "audioexponent", 1.0),
            get(vals, "audioamount", 1.0),
            // PulseParams rounds up to 112 bytes (vec3 members force a
            // 16-byte struct alignment) — pad or wgpu rejects the bind group.
            0.0,
            0.0,
            0.0,
        ]),
        "scroll" => pack(&[
            time,
            get(vals, "speedx", 0.2),
            get(vals, "speedy", 0.2),
            get(vals, "repeat_r", 1.0),
            get(vals, "repeat_g", 1.0),
            0.0,
            0.0,
            0.0,
        ]),
        "shake" => {
            // shake.vert precomputes v_Bounds = (x, 1/(y-x)) from `bounds`.
            let bounds_x = get(vals, "bounds_x", 0.0);
            let bounds_y = get(vals, "bounds_y", 1.0);
            pack(&[
                time,
                get(vals, "speed", 1.0),
                get(vals, "strength", 0.1),
                get(vals, "combo_DIRECTION", 0.0),
                get(vals, "friction_x", 1.0),
                get(vals, "friction_y", 1.0),
                bounds_x,
                1.0 / (bounds_y - bounds_x).max(0.0001),
                get(vals, "combo_NOISE", 0.0),
                0.0,
                0.0,
                0.0,
            ])
        }
        "tint" => {
            let r = get(vals, "color_r", 1.0);
            let g = get(vals, "color_g", 0.0);
            let b = get(vals, "color_b", 0.0);
            let a = get(vals, "alpha", 1.0);
            pack(&[
                r,
                g,
                b,
                a,
                get(vals, "combo_BLENDMODE", 30.0),
                0.0,
                0.0,
                0.0,
            ])
        }
        "opacity" => pack(&[get(vals, "alpha", 1.0), 0.0, 0.0, 0.0]),
        "waterripple" => pack(&[
            time,
            get(vals, "ripplestrength", 0.1),
            get(vals, "animationspeed", 0.15),
            get(vals, "scale", 1.0),
        ]),
        "waterwaves" => {
            // Direction is the reference's rotateVec2((0, 1), angle) —
            // base vector (0, 1), not (1, 0).
            let dir = get(vals, "direction", 0.0);
            pack(&[
                time,
                get(vals, "speed", 5.0),
                get(vals, "scale", 200.0),
                get(vals, "strength", 0.1),
                -dir.sin(),
                dir.cos(),
                get(vals, "exponent", 1.0),
                0.0,
            ])
        }
        "spin" => pack(&[
            time,
            get(vals, "speed", 1.0),
            get(vals, "center_r", 0.5),
            get(vals, "center_g", 0.5),
            get(vals, "size", 0.1),
            get(vals, "feather", 0.002),
            get(vals, "combo_REPEAT", 1.0),
            0.0,
        ]),
        _ => vec![0u8; 32],
    }
}
