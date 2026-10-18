use super::*;

/// Research check (not permanent coverage — depends on an untracked,
/// locally-downloaded Workshop sample that won't exist in CI): does the
/// real "godrays" effect's GLSL, exactly as authored by Valve/WE and
/// shipped in a real wallpaper, transpile through the *fully generic*
/// effect pipeline with zero wp-engine-specific code? See
/// ~/Applications/ghidra-dump/REPORT.md's volumetric-lighting follow-up
/// for why this matters: godrays is what real content actually uses for
/// light shafts, not the Light-object-driven `castvolumetrics` system.
#[test]
fn real_godrays_effect_transpiles_through_generic_pipeline() {
    let wallpaper_dir = std::path::Path::new("/home/tupretiw/wp-engine/analysis/orig_1137423685");
    if !wallpaper_dir.exists() {
        eprintln!("SKIP: analysis/ sample not present locally");
        return;
    }
    // No real WE assets dir in this sandbox — forces the embedded
    // WE_COMMON_H/WE_COMMON_BLENDING_H fallback in loader.rs, exactly
    // the path a Linux/macOS install without the Steam client hits too.
    let resolver = super::super::resolver::AssetResolver::new(Some(wallpaper_dir), None);

    let cases: &[(&str, &str, &[(&str, i32)])] = &[
        (
            "effects/godrays_downsample2",
            "effects/godrays_downsample2",
            &[],
        ),
        (
            "effects/godrays_cast",
            "effects/godrays_cast",
            &[("CASTER", 0), ("SAMPLES", 0)],
        ),
        (
            "effects/godrays_gaussian_x",
            "effects/godrays_gaussian",
            &[("KERNEL", 1), ("VERTICAL", 0)],
        ),
        (
            "effects/godrays_gaussian_y",
            "effects/godrays_gaussian",
            &[("KERNEL", 1), ("VERTICAL", 1)],
        ),
        (
            "effects/godrays_combine",
            "effects/godrays_combine",
            &[("BLENDMODE", 9)],
        ),
    ];

    let mut failures = Vec::new();
    for (label, shader_name, combo_pairs) in cases {
        let (frag_glsl, vert_glsl) = match super::super::loader::load_glsl_shader_with_resolver(
            &resolver,
            shader_name,
            None,
        ) {
            Ok(pair) => pair,
            Err(e) => {
                failures.push(format!("{label}: load failed: {e:#}"));
                continue;
            }
        };
        let combos: std::collections::HashMap<String, i32> = combo_pairs
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();
        let model = crate::audio::model::ShaderModel::from_resolved_glsl(
            label.to_string(),
            frag_glsl,
            combos,
            crate::audio::model::WEBlending::Normal,
        );
        if let Err(e) = translate_full(&model, Some(&vert_glsl)) {
            failures.push(format!("{label}: {e:#}"));
        }
    }
    assert!(
        failures.is_empty(),
        "godrays passes failed to transpile:\n{}",
        failures.join("\n")
    );
}

#[test]
fn unrolls_literal_indexed_array_varying() {
    let src = "varying vec2 v_TexCoord[4];\nvarying vec2 v_TexCoordBase;\n\
               void main() {\n\
               \tv_TexCoord[0] = a;\n\tv_TexCoord[1] = b;\n\
               \tv_TexCoord[2] = c;\n\tv_TexCoord[3] = d;\n}\n";
    let out = unroll_array_varyings(src);
    assert!(out.contains("varying vec2 v_TexCoord_0;"));
    assert!(out.contains("varying vec2 v_TexCoord_1;"));
    assert!(out.contains("varying vec2 v_TexCoord_2;"));
    assert!(out.contains("varying vec2 v_TexCoord_3;"));
    assert!(out.contains("varying vec2 v_TexCoordBase;"));
    assert!(!out.contains('['));
    assert!(out.contains("v_TexCoord_0 = a;"));
    assert!(out.contains("v_TexCoord_3 = d;"));
}

#[test]
fn leaves_source_without_array_varyings_untouched() {
    let src = "varying vec2 v_TexCoord;\nvoid main() { gl_FragColor = vec4(1.0); }\n";
    assert_eq!(unroll_array_varyings(src), src);
}

#[test]
fn leaves_dynamically_sized_arrays_alone() {
    let src = "varying vec2 v_TexCoord[LIGHTS_POINT];\n";
    let out = unroll_array_varyings(src);
    assert_eq!(out, src);
}

/// blur_downsample4.frag: a *fixed*-size array (`[4]`) but indexed by a
/// loop variable (`v_TexCoord[i]`), not literals. Unrolling only the
/// declaration here previously left `v_TexCoord[i]` referring to a name
/// that no longer existed ("undeclared identifier"), instead of leaving
/// the whole thing untouched for the honest "array varying not
/// supported" bail-out.
#[test]
fn leaves_loop_indexed_fixed_size_arrays_alone() {
    let src = "varying vec2 v_TexCoord[4];\n\
               void main() {\n\
               \tfor (int i = 0; i < 4; ++i) {\n\
               \t\tvec4 s = texSample2D(g_Texture0, v_TexCoord[i]);\n\
               \t}\n}\n";
    let out = unroll_array_varyings(src);
    assert_eq!(out, src);
}

#[test]
fn unroll_simple_for_loops_expands_fixed_bound_loop() {
    let src = "void main() {\n\tfor (int i = 0; i < 3; ++i) {\n\t\tx += arr[i];\n\t}\n}\n";
    let out = unroll_simple_for_loops(src);
    assert!(!out.contains("for ("));
    assert!(out.contains("x += arr[0];"));
    assert!(out.contains("x += arr[1];"));
    assert!(out.contains("x += arr[2];"));
    assert!(!out.contains("arr[3]"));
}

#[test]
fn unroll_simple_for_loops_handles_post_increment_and_less_equal() {
    let src = "for (int i = 0; i <= 2; i++) { y += a[i]; }";
    let out = unroll_simple_for_loops(src);
    assert!(out.contains("y += a[0];"));
    assert!(out.contains("y += a[1];"));
    assert!(out.contains("y += a[2];"));
    assert!(!out.contains("a[3]"));
}

#[test]
fn unroll_simple_for_loops_leaves_non_zero_start_untouched() {
    let src = "for (int i = 1; i < 4; ++i) { x += arr[i]; }";
    assert_eq!(unroll_simple_for_loops(src), src);
}

#[test]
fn unroll_simple_for_loops_leaves_non_literal_bound_untouched() {
    let src = "for (int i = 0; i < N; ++i) { x += arr[i]; }";
    assert_eq!(unroll_simple_for_loops(src), src);
}

#[test]
fn unroll_simple_for_loops_leaves_mismatched_loop_var_untouched() {
    let src = "for (int i = 0; j < 4; ++i) { x += arr[i]; }";
    assert_eq!(unroll_simple_for_loops(src), src);
}

#[test]
fn unroll_simple_for_loops_ignores_braces_inside_line_comments() {
    let src = "for (int i = 0; i < 2; ++i) {\n\t// a stray brace: }\n\tx += arr[i];\n}\n";
    let out = unroll_simple_for_loops(src);
    assert!(!out.contains("for ("));
    assert!(out.contains("x += arr[0];"));
    assert!(out.contains("x += arr[1];"));
}

/// The exact real-world shape from blur_downsample4.frag: loop-unroll
/// then array-varying-unroll together must fully eliminate the array.
#[test]
fn loop_then_array_unroll_fully_resolves_real_world_pattern() {
    let src = "varying vec2 v_TexCoord[4];\n\
               uniform sampler2D g_Texture0;\n\
               void main() {\n\
               \tfloat weight = 0.0;\n\
               \tvec4 result = CAST4(0.0);\n\
               \tfor (int i = 0; i < 4; ++i)\n\
               \t{\n\
               \t\tvec4 sample = texSample2D(g_Texture0, v_TexCoord[i]);\n\
               \t\tresult += sample * sample.a;\n\
               \t\tweight += sample.a;\n\
               \t}\n\
               \tgl_FragColor.rgb = result.rgb / max(0.001, weight);\n\
               }\n";
    let loop_unrolled = unroll_simple_for_loops(src);
    let fully_unrolled = unroll_array_varyings(&loop_unrolled);
    assert!(!fully_unrolled.contains('['));
    assert!(fully_unrolled.contains("varying vec2 v_TexCoord_0;"));
    assert!(fully_unrolled.contains("varying vec2 v_TexCoord_3;"));
    assert!(fully_unrolled.contains("v_TexCoord_0"));
    assert!(fully_unrolled.contains("v_TexCoord_3"));
}

/// The exact real-world case (nitro.frag): `max(0, albedo.rgb)` is wrong
/// two ways — GLSL only defines `genType max(genType, float)` (vector
/// first), and the literal must be a genuine float. Both need fixing.
#[test]
fn coerce_int_literal_fixes_bare_integer_and_swaps_order_in_max_call() {
    let src = "gl_FragColor = vec4(max(0, albedo.rgb), albedo.a);";
    let out = coerce_int_literal_builtin_args(src);
    assert_eq!(out, "gl_FragColor = vec4(max(albedo.rgb, 0.0), albedo.a);");
}

#[test]
fn coerce_int_literal_handles_min_and_clamp_and_negative() {
    assert_eq!(coerce_int_literal_builtin_args("min(0, x)"), "min(x, 0.0)");
    assert_eq!(
        coerce_int_literal_builtin_args("clamp(x, 0, 1)"),
        "clamp(x, 0.0, 1.0)"
    );
    assert_eq!(
        coerce_int_literal_builtin_args("max(-1, x)"),
        "max(x, -1.0)"
    );
}

/// A literal that's already a float still needs reordering if it's in
/// the (invalid) first position — order matters independently of type.
/// `clamp`'s args are never reordered (not commutative), and calls that
/// aren't max/min/clamp at all are left alone entirely.
#[test]
fn coerce_int_literal_reorders_already_float_literal_and_ignores_other_calls() {
    assert_eq!(
        coerce_int_literal_builtin_args("max(0.0, x)"),
        "max(x, 0.0)"
    );
    assert_eq!(
        coerce_int_literal_builtin_args("clamp(y, a, b)"),
        "clamp(y, a, b)"
    );
    assert_eq!(coerce_int_literal_builtin_args("foo(0, 1)"), "foo(0, 1)");
}

#[test]
fn coerce_int_literal_leaves_correctly_ordered_call_untouched() {
    assert_eq!(
        coerce_int_literal_builtin_args("max(albedo.rgb, 0.0)"),
        "max(albedo.rgb, 0.0)"
    );
}

#[test]
fn coerce_int_literal_handles_nested_calls() {
    let src = "max(0, texture(tex, uv).rgb)";
    assert_eq!(
        coerce_int_literal_builtin_args(src),
        "max(texture(tex, uv).rgb, 0.0)"
    );
}

#[test]
fn coerce_swizzle_mix_fixes_real_shift_hue_pattern() {
    let src = "albedo.rgb = mix(albedo, newAlbedo, mask);";
    assert_eq!(
        coerce_swizzle_mismatched_mix_arg(src),
        "albedo.rgb = mix(albedo.rgb, newAlbedo, mask);"
    );
}

#[test]
fn coerce_swizzle_mix_leaves_already_matching_types_untouched() {
    let src = "color.xyz = mix(color.xyz, other, t);";
    assert_eq!(coerce_swizzle_mismatched_mix_arg(src), src);
}

#[test]
fn coerce_swizzle_mix_ignores_unrelated_mix_calls() {
    let src = "vec3 c = mix(a, b, t);";
    assert_eq!(coerce_swizzle_mismatched_mix_arg(src), src);
}

#[test]
fn coerce_swizzle_mix_ignores_comparison_not_assignment() {
    let src = "if (foo.rgb == mix(foo, bar, t)) { }";
    assert_eq!(coerce_swizzle_mismatched_mix_arg(src), src);
}

#[test]
fn coerce_swizzle_mix_ignores_different_base_identifier() {
    let src = "albedo.rgb = mix(other, newAlbedo, mask);";
    assert_eq!(coerce_swizzle_mismatched_mix_arg(src), src);
}

/// godrays_gaussian.frag declares the same varying at three different
/// sizes under mutually-exclusive `#if KERNEL==N` branches; only one
/// (max-sized) declaration block should survive, with no duplicates.
#[test]
fn merges_same_name_declared_at_multiple_sizes_across_if_branches() {
    let src = "#if KERNEL == 0\nvarying vec2 v_TexCoord[13];\n#endif\n\
               #if KERNEL == 1\nvarying vec2 v_TexCoord[7];\n#endif\n\
               #if KERNEL == 2\nvarying vec2 v_TexCoord[3];\n#endif\n\
               void main() {\n\
               #if KERNEL == 2\n\tvec2 a = v_TexCoord[0] + v_TexCoord[2];\n#endif\n}\n";
    let out = unroll_array_varyings(src);
    assert_eq!(out.matches("varying vec2 v_TexCoord_0;").count(), 1);
    assert_eq!(out.matches("varying vec2 v_TexCoord_12;").count(), 1);
    assert!(!out.contains("v_TexCoord_13"));
    assert!(out.contains("v_TexCoord_0 + v_TexCoord_2"));
    assert!(!out.contains('['));
}

#[test]
fn shadowed_builtin_definitions_are_renamed_but_calls_are_not() {
    // crt_scan_line.frag defines its own mod(); glslang rejects the
    // precision-qualifier clash with the builtin.
    let src = "float mod(float x, float y) {\n return x - y * floor(x/y);\n}\nfloat f() { return mod(a, b); }\n";
    let out = rename_shadowed_builtins(src);
    assert!(out.contains("float _wp_mod(float x"), "{out}");
    assert!(out.contains("return _wp_mod(a, b)"), "{out}");
    // A shader that only CALLS frac keeps the macro name.
    let call_only = "float f() {\n return frac(x);\n}\n";
    assert_eq!(rename_shadowed_builtins(call_only), call_only);
}

#[test]
fn coerce_leaves_call_arguments_and_splats_scalar_swizzles() {
    let mut w = HashMap::new();
    w.insert("v_TexCoord".to_string(), 4u8);
    // contrast_based_sharpness: get_noise(vec4) must keep its vec4 argument.
    let out = coerce_vector_widths("vec3 noise_color = get_noise(v_TexCoord);", &mut w);
    assert!(!out.contains("v_TexCoord."), "{out}");
    // lens_flare_sun: `.xyz` on a float is a splat in WE's compiler.
    let mut w2 = HashMap::new();
    let _ = coerce_vector_widths("float timer = sin(g_Time);", &mut w2);
    let out2 = coerce_vector_widths("vec3 c = rotation + timer.xyz;", &mut w2);
    assert!(out2.contains("vec3(timer)"), "{out2}");
}

#[test]
fn coerce_truncates_wide_varying_inside_narrow_declaration() {
    // clipping_mask.frag: vec4 varying driving vec2 math (HLSL truncates).
    let mut w = HashMap::new();
    w.insert("v_TexCoord".to_string(), 4u8);
    let out = coerce_vector_widths(
        "vec2 uvTex = ((v_TexCoord * 2.0 - 1.0 - texScaleCenter) / u_scale + 1.0) / 2.0;",
        &mut w,
    );
    assert!(out.contains("v_TexCoord.xy"), "{out}");
    // Already-swizzled and unknown identifiers stay untouched.
    let out2 = coerce_vector_widths("vec2 b = v_TexCoord.zw + something;", &mut w);
    assert!(!out2.contains(".zw.xy"), "{out2}");
}

#[test]
fn coerce_truncates_unswizzled_texture_sample_to_declared_width() {
    let mut vw = HashMap::new();
    let out = coerce_vector_widths(
        "vec3 albedo = texSample2D(g_Texture0, v_TexCoord.xy);",
        &mut vw,
    );
    assert_eq!(
        out,
        "vec3 albedo = texSample2D(g_Texture0, v_TexCoord.xy).xyz;"
    );
    assert_eq!(vw.get("albedo"), Some(&3));
}

#[test]
fn coerce_truncates_plain_reassignment_using_tracked_width() {
    let mut vw = HashMap::new();
    vw.insert("albedo".to_string(), 3u8);
    let out = coerce_vector_widths("albedo = texSample2D(g_Texture0, uv);", &mut vw);
    assert_eq!(out, "albedo = texSample2D(g_Texture0, uv).xyz;");
}

#[test]
fn coerce_truncates_wide_varying_against_cast_broadcast() {
    let mut vw = HashMap::new();
    vw.insert("v_TexCoord".to_string(), 3u8);
    let out = coerce_vector_widths(
        "float scale = pow(length(abs(v_TexCoord - CAST2(u_offset)) * 1.0), 3.0);",
        &mut vw,
    );
    assert_eq!(
        out,
        "float scale = pow(length(abs(v_TexCoord.xy - CAST2(u_offset)) * 1.0), 3.0);"
    );
}

#[test]
fn coerce_leaves_matching_widths_untouched() {
    let mut vw = HashMap::new();
    vw.insert("v_TexCoord".to_string(), 2u8);
    let line = "float scale = length(v_TexCoord - CAST2(u_offset));";
    assert_eq!(coerce_vector_widths(line, &mut vw), line);
}

#[test]
fn coerce_leaves_swizzled_texture_sample_untouched() {
    let mut vw = HashMap::new();
    let line = "vec3 albedo = texSample2D(g_Texture0, uv).rgb;";
    assert_eq!(coerce_vector_widths(line, &mut vw), line);
}

#[test]
fn coerce_is_idempotent_on_already_truncated_operand() {
    let mut vw = HashMap::new();
    vw.insert("v_TexCoord".to_string(), 3u8);
    let once = coerce_vector_widths(
        "float scale = length(v_TexCoord - CAST2(u_offset));",
        &mut vw,
    );
    let twice = coerce_vector_widths(&once, &mut vw);
    assert_eq!(once, twice);
}

/// The reference's applyLinkedVaryingCompatibility: a vertex-side
/// `varying vec2` matching a fragment-side `varying vec4` gets upgraded,
/// with whole-variable assignments wrapped in vec4(expr, 0.0, 1.0);
/// swizzled stores and unrelated varyings stay untouched.
#[test]
fn harmonize_upgrades_mismatched_varying_and_assignments() {
    let vert = "varying vec2 v_TexCoord;\nvarying vec2 v_Other;\nvoid main() {\n\tv_TexCoord = a_TexCoord * 2.0;\n\tv_TexCoord.xy = a_TexCoord;\n\tv_Other = a_TexCoord;\n}\n";
    let frag = "varying vec4 v_TexCoord;\nvarying vec2 v_Other;\nvoid main() { gl_FragColor = v_TexCoord; }\n";
    let out = harmonize_varying_widths(vert, frag);
    assert!(out.contains("varying vec4 v_TexCoord;"));
    assert!(out.contains("v_TexCoord = vec4(a_TexCoord * 2.0, 0.0, 1.0);"));
    assert!(
        out.contains("v_TexCoord.xy = a_TexCoord;"),
        "swizzled store untouched"
    );
    assert!(
        out.contains("varying vec2 v_Other;"),
        "matching widths untouched"
    );
    assert!(out.contains("v_Other = a_TexCoord;"));
}

#[test]
fn harmonize_leaves_matched_interfaces_alone() {
    let vert = "varying vec2 v_TexCoord;\nvoid main() { v_TexCoord = a_TexCoord; }\n";
    let frag =
        "varying vec2 v_TexCoord;\nvoid main() { gl_FragColor = vec4(v_TexCoord, 0.0, 1.0); }\n";
    assert_eq!(harmonize_varying_widths(vert, frag), vert);
}

#[test]
fn wrap_assignment_rhs_handles_for_loop_init() {
    // The RHS must stop at the for-init's own `;`, and `<=` must not be
    // mistaken for an assignment (blur_gaussian.frag's exact shape).
    let line = "for (int i = -iterations; i <= iterations; i++)";
    let out = wrap_assignment_rhs(line, "int(", ")").unwrap();
    assert_eq!(out, "for (int i = int(-iterations); i <= iterations; i++)");
}

#[test]
fn wrap_assignment_rhs_skips_comparisons_but_wraps_compound_assign() {
    // Comparisons are never assignment targets.
    assert!(wrap_assignment_rhs("if (a == b) return;", "(", ").x").is_none());
    // Compound assignments ARE targets: `int x *= floatExpr` needs the RHS
    // coerced just like a plain assignment (simple_audio_bars' `bar *= step`).
    assert_eq!(
        wrap_assignment_rhs("\tbarLeft *= step(a, b);", "int(", ")").as_deref(),
        Some("\tbarLeft *= int(step(a, b));")
    );
}

#[test]
fn repair_patches_lines_named_in_shaderc_errors() {
    let glsl = "float pointer = g_PointerPosition.xy * u_pointerSpeed;\n\
                int i = -iterations;\n\
                vec3 c = 0.5;";
    let errs = "shader.frag:1: error: '=' : cannot convert from ' temp highp 2-component vector of float' to ' temp highp float'\n\
                shader.frag:2: error: '=' : cannot convert from ' temp highp float' to ' temp highp int'\n\
                shader.frag:3: error: '=' : cannot convert from ' const float' to ' temp highp 3-component vector of float'";
    let out = repair_assignment_conversions(glsl, errs).expect("should repair");
    assert!(out.contains("float pointer = (g_PointerPosition.xy * u_pointerSpeed).x;"));
    assert!(out.contains("int i = int(-iterations);"));
    assert!(out.contains("vec3 c = vec3(0.5);"));
}

/// End-to-end: a shader with the HLSL-leniency patterns from real
/// workshop content (lens_flare_sun's vec2→float, blur_gaussian's
/// float→int loop bound) must compile after the repair loop.
#[test]
fn repairs_implicit_conversions_through_real_compile() {
    let glsl = "#version 450\n\
        layout(location=0) out vec4 fragColor;\n\
        void main() {\n\
        \tvec2 v = vec2(1.5, 2.5);\n\
        \tfloat pointer = v * 2.0;\n\
        \tfloat iterations = 3.0;\n\
        \tfloat acc = 0.0;\n\
        \tfor (int i = -iterations; i <= iterations; i++) { acc += float(i) + pointer; }\n\
        \tfragColor = vec4(acc);\n\
        }\n";
    assert!(glsl_to_spirv(glsl, naga::ShaderStage::Fragment).is_err());
    glsl_to_spirv_with_repairs(glsl, naga::ShaderStage::Fragment)
        .expect("repair loop should make this compile");
}

/// depth_map.frag branches on a float uniform; HLSL calls nonzero true.
#[test]
fn bool_condition_repair_wraps_float_conditions() {
    let glsl = "#version 450\n\
        layout(location=0) out vec4 fragColor;\n\
        void main() {\n\
        \tfloat flag = 1.0;\n\
        \tfloat d = 0.25;\n\
        \tif(flag){ d = 1.0 - d; }\n\
        \tfragColor = vec4(d);\n\
        }\n";
    assert!(glsl_to_spirv(glsl, naga::ShaderStage::Fragment).is_err());
    glsl_to_spirv_with_repairs(glsl, naga::ShaderStage::Fragment)
        .expect("repair loop should make this compile");
}

/// `pow(vec3, float)` (color_grading.frag's shape) is illegal in strict
/// GLSL; the `_wp_pow` overload shims must let it translate.
#[test]
fn pow_scalar_broadcast_translates() {
    let src = "uniform sampler2D g_Texture0;\n\
               varying vec2 v_TexCoord;\n\
               void main() {\n\
               \tvec3 c = pow(texSample2D(g_Texture0, v_TexCoord).rgb, 2.2);\n\
               \tgl_FragColor = vec4(c, 1.0);\n\
               }\n";
    let model = ShaderModel::from_resolved_glsl(
        "pow_test".to_string(),
        src.to_string(),
        HashMap::new(),
        crate::audio::model::shader_model::WEBlending::Normal,
    );
    translate(&model).expect("pow(vec3, float) should translate via _wp_pow shims");
}

/// The embedded common_perspective.h body (squareToQuad) must be valid
/// GLSL: a vertex shader shaped like the `perspective` effect's vert —
/// `inverse(squareToQuad(p0..p3)) * vec3(uv,1)` — compiles with it inlined.
/// Without the body squareToQuad is undefined and this fails, which is the
/// exact macOS case the embedded-header fallback repairs.
#[test]
fn embedded_perspective_header_compiles() {
    let glsl = format!(
        "#version 450\n\
         layout(location=0) in vec3 a_Position;\n\
         layout(location=1) in vec2 a_TexCoord;\n\
         layout(location=0) out vec3 v_TexCoord;\n\
         {WE_COMMON_PERSPECTIVE_H}\n\
         void main() {{\n\
         \tmat3 xform = inverse(squareToQuad(vec2(0.0), vec2(1.0, 0.0), vec2(1.0), vec2(0.0, 1.0)));\n\
         \tv_TexCoord = xform * vec3(a_TexCoord, 1.0);\n\
         \tgl_Position = vec4(a_Position, 1.0);\n\
         }}\n"
    );
    glsl_to_spirv_with_repairs(&glsl, naga::ShaderStage::Vertex)
        .expect("perspective vert with embedded squareToQuad should compile");
}

#[test]
fn int_assigned_float_step_repairs() {
    let glsl = "#version 450\n\
        layout(location=0) out vec4 fragColor;\n\
        void main() {\n\
        \tfloat a = 0.3; float b = 0.7;\n\
        \tint barLeft = step(a, b);\n\
        \tfragColor = vec4(float(barLeft));\n\
        }\n";
    assert!(glsl_to_spirv(glsl, naga::ShaderStage::Fragment).is_err());
    glsl_to_spirv_with_repairs(glsl, naga::ShaderStage::Fragment)
        .expect("int = step(float) should repair via int() wrap");
}

#[test]
fn rewrite_modulo_handles_real_audio_bar_patterns() {
    // simple_audio_bars.frag's two shapes.
    assert_eq!(
        rewrite_modulo_on_line("\tuint barFreq1 = frequency % RESOLUTION;").as_deref(),
        Some("\tuint barFreq1 = mod(frequency, RESOLUTION);")
    );
    assert_eq!(
        rewrite_modulo_on_line("\tuint barFreq2 = (barFreq1 + 1) % RESOLUTION;").as_deref(),
        Some("\tuint barFreq2 = mod((barFreq1 + 1), RESOLUTION);")
    );
}

#[test]
fn rewrite_modulo_leaves_compound_assign_and_plain_lines() {
    assert_eq!(rewrite_modulo_on_line("\tx %= 4;"), None);
    assert_eq!(rewrite_modulo_on_line("\tfloat y = a + b;"), None);
}

/// End-to-end: a `float % int` shader (int-only `%` in strict GLSL) must
/// compile through the repair loop, mirroring simple_audio_bars.
#[test]
fn float_modulo_repairs_through_real_compile() {
    let glsl = "#version 450\n\
        layout(location=0) out vec4 fragColor;\n\
        void main() {\n\
        \tfloat frequency = 6.5;\n\
        \tuint bar = frequency % 4;\n\
        \tfragColor = vec4(float(bar));\n\
        }\n";
    assert!(glsl_to_spirv(glsl, naga::ShaderStage::Fragment).is_err());
    glsl_to_spirv_with_repairs(glsl, naga::ShaderStage::Fragment)
        .expect("float % int + float->uint assign should repair");
}

/// A shader reading `g_AudioSpectrum64Left` must translate to WGSL with the
/// array backed by a live uniform binding (not a zero `const`), and the
/// result must flag `uses_audio` so the renderer starts capture.
#[test]
fn audio_spectrum_becomes_live_uniform() {
    let src = "uniform sampler2D g_Texture0;\n\
               uniform float g_AudioSpectrum64Left[64];\n\
               varying vec2 v_TexCoord;\n\
               void main() {\n\
               \tint i = int(v_TexCoord.x * 64.0);\n\
               \tfloat amp = g_AudioSpectrum64Left[i];\n\
               \tgl_FragColor = vec4(amp, amp, amp, 1.0);\n\
               }\n";
    let model = ShaderModel::from_resolved_glsl(
        "audio_test".to_string(),
        src.to_string(),
        HashMap::new(),
        crate::audio::model::shader_model::WEBlending::Normal,
    );
    let t = translate(&model).expect("audio spectrum shader should translate");
    assert!(
        t.uses_audio,
        "uses_audio must be set for g_AudioSpectrum reads"
    );
    // The array is now a bound uniform member, so the emitted WGSL must not
    // contain a zero-const fallback for it.
    assert!(
        !t.wgsl.contains("array<f32, 64>(0.0"),
        "audio array must not be a zero const:\n{}",
        t.wgsl
    );
}
