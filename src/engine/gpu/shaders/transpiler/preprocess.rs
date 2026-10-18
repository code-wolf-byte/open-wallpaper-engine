//! Assembling the GLSL to compile from a shader model: the fragment and
//! matching vertex stages with Wallpaper Engine's headers, combo defines and
//! uniform declarations, or a synthetic vertex shader when the effect has
//! none that fits.

use super::*;

pub(super) fn preprocess_frag(model: &ShaderModel, extra_scalars: &[(String, String)]) -> String {
    let src = &rename_shadowed_builtins(&model.frag_glsl);

    // First pass: collect declarations in order
    let mut sampler_names: Vec<String> = Vec::new(); // g_Texture0, g_Texture1, ...
    let mut scalars: Vec<(String, String)> = Vec::new(); // (type, name) → UBO
    let mut inputs: Vec<(String, String)> = Vec::new(); // varyings
    let mut zero_arrays: Vec<(String, usize)> = Vec::new(); // float name[N] → const zeros

    for line in src.lines() {
        let t = line.trim();
        let decl = t.split("//").next().unwrap_or(t).trim();
        let tok: Vec<&str> = decl.split_whitespace().collect();
        if tok.len() >= 3 && tok[0] == "uniform" {
            let ty = tok[1];
            let name = tok[2].trim_end_matches(';');
            if ty.starts_with("sampler") {
                sampler_names.push(name.to_string());
            } else if !ty.is_empty() && !name.contains('[') {
                scalars.push((ty.to_string(), name.to_string()));
            } else if ty == "float" {
                // Float-array uniforms (`uniform float g_AudioSpectrum64Left[64];`)
                // don't fit the 16-byte-per-scalar UBO model, so they're
                // re-emitted below as zero-filled const arrays instead of
                // silently vanishing (which left the identifiers undeclared
                // and failed the whole compile). The only such uniforms in
                // practice are WE's audio-spectrum bands, and zero is exactly
                // what they hold when no audio is playing — the right output
                // for an engine with no audio-capture pipeline (yet).
                if let Some((base, rest)) = name.split_once('[') {
                    if let Ok(n) = rest.trim_end_matches(']').parse::<usize>() {
                        if n > 0 && !zero_arrays.iter().any(|(b, _)| b == base) {
                            zero_arrays.push((base.to_string(), n));
                        }
                    }
                }
            }
        } else if t.starts_with("varying ") || t.starts_with("attribute ") {
            if tok.len() >= 3 {
                let ty = tok[1];
                let name_raw = decl
                    .split_whitespace()
                    .nth(2)
                    .unwrap_or("")
                    .trim_end_matches(';');
                // Skip array varyings (e.g. `varying vec2 v_TexCoord[7]`) — naga cannot
                // represent arrays as entry-point I/O; the shader would fail at SPIR-V parse.
                if name_raw.contains('[') {
                    continue;
                }
                inputs.push((ty.to_string(), name_raw.to_string()));
            }
        }
    }

    // Texture binding indices: g_Texture0 → 0, g_TextureN → N+2 (3, 4, 5, ...).
    //
    // The binding MUST come from the `N` in the uniform's own name, not from
    // its position in the declaration scan: the scan reads the raw source
    // and ignores `#if` blocks, so a conditionally-declared sampler that the
    // compile strips (e.g. shine_combine.frag declares `g_Texture2` inside
    // `#if COPYBG` *above* g_Texture0/g_Texture1) would otherwise shift every
    // later sampler's binding by one — the shader then samples neighbouring
    // slots (or the white 1×1 dummy) instead of its real inputs. WE itself
    // keys texture units off the name's index (CPass binds unit N to
    // g_TextureN), so the name is the authoritative slot.
    let tex_binding = |i: usize, name: &str| -> u32 {
        let n = name
            .strip_prefix("g_Texture")
            .and_then(|rest| rest.parse::<usize>().ok())
            .unwrap_or(i);
        match n {
            0 => 0,
            n => 2 + n as u32,
        } // 0→0, 1→3, 2→4, 3→5
    };

    let mut out = String::new();
    out.push_str("#version 450\n");
    out.push_str("#define frac(x) fract(x)\n");
    out.push_str("#define saturate(x) clamp((x), 0.0, 1.0)\n");
    out.push_str("#define CAST2(x) vec2(x)\n");
    out.push_str("#define CAST3(x) vec3(x)\n");
    out.push_str("#define CAST4(x) vec4(x)\n");
    // Use separate texture2D + sampler for all texture sample macros so naga sees
    // OpSampledImage (from constructor) rather than OpLoad of a combined sampler2D.
    out.push_str("#define texSample2D(s, uv) texture(sampler2D(s, _wp_sampler), (uv).xy)\n");
    out.push_str(
        "#define texSample2DLod(s, uv, lod) textureLod(sampler2D(s, _wp_sampler), (uv).xy, lod)\n",
    );
    out.push_str("#define textureSample2D(s, uv) texture(sampler2D(s, _wp_sampler), (uv).xy)\n");
    out.push_str("#define texture2D(s, uv) texture(sampler2D(s, _wp_sampler), (uv).xy)\n");
    out.push_str(
        "#define texture2DLod(s, uv, lod) textureLod(sampler2D(s, _wp_sampler), (uv).xy, lod)\n",
    );
    out.push_str("#define textureCube(s, uv) texture(samplerCube(s, _wp_sampler), (uv).xyz)\n");
    out.push_str("#define lerp(a, b, t) mix(a, b, t)\n");
    out.push_str("#define mul(a, b) ((b) * (a))\n");
    out.push_str(&combo_defines(&model.effective_combos()));
    // HLSL type aliases used pervasively in WE shaders.
    // These must come BEFORE any inlined shader body (common.h etc. are inlined by the loader).
    // We do NOT emit our WE_COMMON_H / WE_COMMON_BLENDING_H function bodies here because
    // the loader already inlines the real WE common.h content into the source, and emitting
    // our copies would cause "redefinition" errors in shaderc.
    out.push_str("#define float2 vec2\n");
    out.push_str("#define float3 vec3\n");
    out.push_str("#define float4 vec4\n");
    out.push_str("#define int2 ivec2\n");
    out.push_str("#define int3 ivec3\n");
    out.push_str("#define int4 ivec4\n");
    out.push_str("#define CAST3X3(x) mat3(x)\n");
    out.push_str("#define atan2(y,x) atan(y,x)\n");
    out.push_str("#define fmod(x,y) ((x)-(y)*trunc((x)/(y)))\n");
    out.push_str("#define ddx dFdx\n");
    out.push_str("#define ddy(x) dFdy(-(x))\n");
    out.push_str("#define log10(x) (log2(x)*0.301029995663981)\n");
    // Constants sometimes needed by shaders that don't include common.h
    out.push_str("#ifndef M_PI\n#define M_PI 3.14159265359\n#endif\n");
    out.push_str("#ifndef M_PI_2\n#define M_PI_2 6.28318530718\n#endif\n");

    // HLSL's pow() broadcasts scalar arguments to the other side's vector
    // width; strict GLSL only defines pow(genType, genType). Real workshop
    // shaders (e.g. color_grading.frag's `pow(vec3, float)`) rely on the
    // lenient form, so every body call site is renamed to these overloads —
    // the same-shape ones are passthroughs, so legal calls are unaffected.
    out.push_str(concat!(
        "float _wp_pow(float a, float b) { return pow(a, b); }\n",
        "vec2 _wp_pow(vec2 a, vec2 b) { return pow(a, b); }\n",
        "vec3 _wp_pow(vec3 a, vec3 b) { return pow(a, b); }\n",
        "vec4 _wp_pow(vec4 a, vec4 b) { return pow(a, b); }\n",
        "vec2 _wp_pow(vec2 a, float b) { return pow(a, vec2(b)); }\n",
        "vec3 _wp_pow(vec3 a, float b) { return pow(a, vec3(b)); }\n",
        "vec4 _wp_pow(vec4 a, float b) { return pow(a, vec4(b)); }\n",
        "vec2 _wp_pow(float a, vec2 b) { return pow(vec2(a), b); }\n",
        "vec3 _wp_pow(float a, vec3 b) { return pow(vec3(a), b); }\n",
        "vec4 _wp_pow(float a, vec4 b) { return pow(vec4(a), b); }\n",
    ));

    // Separate texture2D declarations with matching bindings
    for (i, name) in sampler_names.iter().enumerate() {
        let b = tex_binding(i, name);
        out.push_str(&format!(
            "layout(set=0, binding={b}) uniform texture2D {name};\n"
        ));
    }
    // Shared sampler at binding 1
    if !sampler_names.is_empty() {
        out.push_str("layout(set=0, binding=1) uniform sampler _wp_sampler;\n");
    }
    // UBO for scalars at group 1 binding 0 (matches effect_bgl). The block is
    // the union of fragment + vertex scalars (fragment first) so both stages
    // agree on std140 offsets.
    if !scalars.is_empty() || !extra_scalars.is_empty() {
        out.push_str("layout(set=1, binding=0) uniform WEUniforms {\n");
        for (ty, name) in scalars.iter().chain(extra_scalars.iter()) {
            out.push_str(&format!("    {ty} {name};\n"));
        }
        out.push_str("};\n");
    }
    // Audio-spectrum arrays (`g_AudioSpectrum{16,32,64}{Left,Right}`) become a
    // live read-only STORAGE buffer (std430) at set=1 binding=1 filled from FFT
    // each frame. The full canonical set is always declared in a fixed order
    // (so element offsets are deterministic regardless of which arrays this
    // shader happens to reference); the renderer always binds the buffer, and
    // shaders that read none of it simply ignore the binding.
    let uses_audio = zero_arrays
        .iter()
        .any(|(name, _)| name.contains("AudioSpectrum"));
    if uses_audio {
        // std430 storage: `float[N]` packs at 4-byte stride (naga-valid),
        // unlike a std140 uniform member's WGSL-illegal 16-byte array stride.
        out.push_str("layout(set=1, binding=1, std430) readonly buffer WEAudio {\n");
        for (name, n) in [
            ("g_AudioSpectrum16Left", 16),
            ("g_AudioSpectrum16Right", 16),
            ("g_AudioSpectrum32Left", 32),
            ("g_AudioSpectrum32Right", 32),
            ("g_AudioSpectrum64Left", 64),
            ("g_AudioSpectrum64Right", 64),
        ] {
            out.push_str(&format!("    float {name}[{n}];\n"));
        }
        out.push_str("};\n");
    }
    // Any remaining (non-audio) float-array uniforms stay zero-filled consts.
    for (name, n) in &zero_arrays {
        if name.contains("AudioSpectrum") {
            continue;
        }
        let zeros = vec!["0.0"; *n].join(", ");
        out.push_str(&format!("const float {name}[{n}] = float[{n}]({zeros});\n"));
    }
    // Input varyings with explicit locations
    for (i, (ty, name)) in inputs.iter().enumerate() {
        out.push_str(&format!("layout(location={i}) in {ty} {name};\n"));
    }
    out.push_str("layout(location=0) out vec4 fragColor;\n");

    // Seed known vector widths (varyings + uniforms) for the truncation-coercion
    // pass below; it grows this map with local declarations as it walks the body.
    let mut var_width: HashMap<String, u8> = HashMap::new();
    for (ty, name) in inputs.iter().chain(scalars.iter()) {
        if let Some(w) = vec_width(ty) {
            var_width.insert(name.clone(), w);
        }
    }

    // Emit shader body (skip declarations already emitted above)
    for line in src.lines() {
        let t = line.trim_start();
        if t.starts_with("// [COMBO]") || t.starts_with("#version") || t.starts_with("precision ") {
            continue;
        }
        let decl = t.split("//").next().unwrap_or(t).trim();
        let first = decl.split_whitespace().next().unwrap_or("");
        if first == "varying" || first == "attribute" || first == "uniform" {
            continue;
        }

        // Skip #include directives — included content is inlined in the header.
        if t.starts_with("#include ") {
            continue;
        }

        // A shader that spells a prelude macro differently (M_PI as
        // 3.1415926535897932384626433832795, say) is a hard glslang error, not
        // a warning. Let the shader's own definition win, like HLSL does.
        if let Some(name) = redefined_macro(t) {
            out.push_str(&format!("#undef {name}\n"));
        }

        // Replace #require LightingV1 with a no-op stub (same as linux-wallpaperengine).
        if t.starts_with("#require LightingV1") {
            out.push_str(
                "vec3 PerformLighting_V1(vec3 worldPos, vec3 albedo, vec3 normal, vec3 viewDir,\n",
            );
            out.push_str(
                "    vec3 specularTint, vec3 baseReflectance, float roughness, float metallic)\n",
            );
            out.push_str("{ return vec3(0.0); }\n");
            continue;
        }

        let coerced = coerce_vector_widths(line, &mut var_width);
        let renamed = rename_reserved_word(&coerced, "sample", "_wp_s");
        // Route pow through the scalar-broadcasting overload shims above.
        let renamed = rename_reserved_word(&renamed, "pow", "_wp_pow");
        let l = renamed
            .replace("gl_FragColor", "fragColor")
            .replace("gl_FragData[0]", "fragColor");
        out.push_str(&l);
        out.push('\n');
    }
    out
}

/// Preprocess a real WE vertex shader into Vulkan GLSL 4.50.
///
/// - Emits the SAME `WEUniforms` block as the fragment stage (union scalar
///   list, set=1 binding=0) so std140 offsets agree across stages.
/// - Assigns each varying the location of the same-named fragment input;
///   fragment inputs the VS never writes are declared and zero-initialized at
///   the top of `main()` (wgpu requires every FS input to have a VS output).
/// - Returns the processed source plus the attribute list (type, name) in
///   declaration order for vertex-buffer layout construction.
pub(super) fn preprocess_vert_matched(
    glsl: &str,
    union_scalars: &[(String, String)],
    frag_inputs: &[(String, String)],
    combos: &HashMap<String, i32>,
) -> Result<(String, Vec<(String, String)>)> {
    // Vertex shaders that sample textures need image bindings we don't
    // provide in the vertex stage; let the synthetic fallback handle those.
    if glsl.contains("texSample2D") || glsl.contains("texture2D") {
        anyhow::bail!("vertex shader samples textures");
    }

    let mut out = String::new();
    out.push_str("#version 450\n");
    out.push_str("#define frac(x) fract(x)\n");
    out.push_str("#define saturate(x) clamp((x), 0.0, 1.0)\n");
    out.push_str("#define CAST2(x) vec2(x)\n");
    out.push_str("#define CAST3(x) vec3(x)\n");
    out.push_str("#define CAST4(x) vec4(x)\n");
    out.push_str("#define lerp(a, b, t) mix(a, b, t)\n");
    out.push_str("#define mul(a, b) ((b) * (a))\n");
    out.push_str(&combo_defines(combos));
    out.push_str("#define float2 vec2\n");
    out.push_str("#define float3 vec3\n");
    out.push_str("#define float4 vec4\n");
    out.push_str("#define int2 ivec2\n");
    out.push_str("#define int3 ivec3\n");
    out.push_str("#define int4 ivec4\n");
    out.push_str("#define CAST3X3(x) mat3(x)\n");
    out.push_str("#define atan2(y,x) atan(y,x)\n");
    out.push_str("#define fmod(x,y) ((x)-(y)*trunc((x)/(y)))\n");
    out.push_str("#define log10(x) (log2(x)*0.301029995663981)\n");
    out.push_str("#ifndef M_PI\n#define M_PI 3.14159265359\n#endif\n");
    out.push_str("#ifndef M_PI_2\n#define M_PI_2 6.28318530718\n#endif\n");

    if !union_scalars.is_empty() {
        out.push_str("layout(set=1, binding=0) uniform WEUniforms {\n");
        for (ty, name) in union_scalars {
            out.push_str(&format!("    {ty} {name};\n"));
        }
        out.push_str("};\n");
    }

    // Fragment-input locations by name; varyings must line up with them.
    let frag_loc: HashMap<&str, usize> = frag_inputs
        .iter()
        .enumerate()
        .map(|(i, (_, name))| (name.as_str(), i))
        .collect();
    let mut next_free_loc = frag_inputs.len();

    let mut attributes: Vec<(String, String)> = Vec::new();
    let mut declared_varyings: Vec<String> = Vec::new();
    let mut body = String::new();

    for line in glsl.lines() {
        let t = line.trim_start();
        if t.starts_with("#version")
            || t.starts_with("precision ")
            || t.starts_with("// [COMBO]")
            || t.starts_with("#include ")
        {
            continue;
        }
        // Shader-local redefinition of a prelude macro wins (see frag stage).
        if let Some(name) = redefined_macro(t) {
            body.push_str(&format!("#undef {name}\n"));
        }
        let decl = t.split("//").next().unwrap_or(t).trim();
        let tok: Vec<&str> = decl.split_whitespace().collect();
        if tok.len() >= 3 && tok[0] == "attribute" {
            let ty = tok[1].to_string();
            let name = tok[2].trim_end_matches(';').to_string();
            out.push_str(&format!(
                "layout(location={}) in {ty} {name};\n",
                attributes.len()
            ));
            attributes.push((ty, name));
            continue;
        }
        if tok.len() >= 3 && tok[0] == "varying" {
            let ty = tok[1].to_string();
            let name = tok[2].trim_end_matches(';').to_string();
            if name.contains('[') {
                anyhow::bail!("array varying '{name}' not supported");
            }
            let loc = frag_loc.get(name.as_str()).copied().unwrap_or_else(|| {
                let l = next_free_loc;
                next_free_loc += 1;
                l
            });
            out.push_str(&format!("layout(location={loc}) out {ty} {name};\n"));
            declared_varyings.push(name);
            continue;
        }
        if tok.first() == Some(&"uniform") {
            continue; // already in the shared UBO
        }
        body.push_str(line);
        body.push('\n');
    }

    // Fragment inputs the vertex shader never declares: emit + zero-init so
    // the wgpu interface check passes.
    let mut missing_inits = String::new();
    for (ty, name) in frag_inputs {
        if !declared_varyings.iter().any(|v| v == name) {
            out.push_str(&format!(
                "layout(location={}) out {ty} {name};\n",
                frag_loc[name.as_str()]
            ));
            missing_inits.push_str(&format!("    {name} = {ty}(0.0);\n"));
        }
    }
    if !missing_inits.is_empty() {
        // Insert right after the opening brace of main().
        let main_pos = body
            .find("void main")
            .context("vertex shader has no main()")?;
        let brace = body[main_pos..]
            .find('{')
            .map(|i| main_pos + i + 1)
            .context("vertex shader main() has no body")?;
        body.insert_str(brace, &format!("\n{missing_inits}"));
    }

    out.push_str(&body);
    Ok((out, attributes))
}

/// Extract (type, name) pairs from fragment `varying` declarations (array varyings excluded).
pub(super) fn frag_inputs(frag_glsl: &str) -> Vec<(String, String)> {
    let mut inputs = Vec::new();
    for line in frag_glsl.lines() {
        let t = line.trim();
        if !t.starts_with("varying ") && !t.starts_with("attribute ") {
            continue;
        }
        let tok: Vec<&str> = t.split_whitespace().collect();
        if tok.len() < 3 {
            continue;
        }
        let name_raw = t
            .split("//")
            .next()
            .unwrap_or(t)
            .split_whitespace()
            .nth(2)
            .unwrap_or("")
            .trim_end_matches(';');
        if name_raw.contains('[') {
            continue;
        } // array varyings already rejected upstream
        inputs.push((tok[1].to_string(), name_raw.to_string()));
    }
    inputs
}

/// Return true when the fragment shader's varyings differ from the vs_we_effect interface
/// (vec4 v_TexCoord @ loc 0, vec2 v_Scroll @ loc 1).  A mismatch causes a wgpu panic.
pub(super) fn needs_custom_vs(inputs: &[(String, String)]) -> bool {
    match inputs {
        [] => false,
        [(t0, _)] => t0 != "vec4",
        [(t0, _), (t1, _)] => t0 != "vec4" || t1 != "vec2",
        _ => true, // more than 2 varyings → vs_we_effect definitely can't match
    }
}

/// Build a Vulkan GLSL 4.50 vertex shader that outputs every declared varying as a
/// simple UV-derived value.  No uniforms — this shader works with any bind group layout.
pub(super) fn synthetic_vs_glsl(inputs: &[(String, String)]) -> String {
    let mut out = String::new();
    out.push_str("#version 450\n");
    for (i, (ty, name)) in inputs.iter().enumerate() {
        out.push_str(&format!("layout(location={i}) out {ty} {name};\n"));
    }
    out.push_str("void main() {\n");
    out.push_str("    int vi = gl_VertexIndex;\n");
    out.push_str("    float x = float(vi & 1) * 4.0 - 1.0;\n");
    out.push_str("    float y = float(vi >> 1) * 4.0 - 1.0;\n");
    out.push_str("    gl_Position = vec4(x, y, 0.0, 1.0);\n");
    out.push_str("    float u = (x + 1.0) * 0.5;\n");
    // naga's SPIR-V frontend runs with adjust_coordinate_space=true, which
    // negates gl_Position.y. UVs must be derived from the POST-flip screen
    // position, so v maps y=-1 (screen top after flip) → 0.
    out.push_str("    float v = (y + 1.0) * 0.5;\n");
    for (ty, name) in inputs {
        let val = match ty.as_str() {
            "float" => "u".to_string(),
            "vec2" => "vec2(u, v)".to_string(),
            "vec3" => "vec3(u, v, 0.0)".to_string(),
            "vec4" => "vec4(u, v, u, v)".to_string(),
            _ => format!("{ty}(0.0)"),
        };
        out.push_str(&format!("    {name} = {val};\n"));
    }
    out.push_str("}\n");
    out
}

pub fn combo_defines(combos: &HashMap<String, i32>) -> String {
    combos
        .iter()
        .map(|(name, value)| format!("#define {} {}\n", name, value))
        .collect()
}
