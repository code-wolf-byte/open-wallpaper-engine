use crate::engine::model::ShaderModel;
use anyhow::{Context, Result};
use std::collections::HashMap;

mod rewrite;
pub use rewrite::*;

mod repair;
use repair::*;

pub(crate) const WE_COMMON_H: &str = "\
#define M_PI 3.14159265359\n\
#define M_PI_HALF 1.57079632679\n\
#define M_PI_2 6.28318530718\n\
#define SQRT_2 1.41421356237\n\
#define SQRT_3 1.73205080756\n\
float greyscale(vec3 c){return dot(c,vec3(0.299,0.587,0.114));}\n\
vec3 hsv2rgb(vec3 c){vec4 K=vec4(1.0,2.0/3.0,1.0/3.0,3.0);vec3 p=abs(fract(c.xxx+K.xyz)*6.0-K.www);return c.z*mix(K.xxx,clamp(p-K.xxx,0.0,1.0),c.y);}\n\
vec3 rgb2hsv(vec3 RGB){vec4 P=(RGB.g<RGB.b)?vec4(RGB.bg,-1.0,2.0/3.0):vec4(RGB.gb,0.0,-1.0/3.0);vec4 Q=(RGB.r<P.x)?vec4(P.xyw,RGB.r):vec4(RGB.r,P.yzx);float C=Q.x-min(Q.w,Q.y);float H=abs((Q.w-Q.y)/(6.0*C+1e-10)+Q.z);vec3 HCV=vec3(H,C,Q.x);float S=HCV.y/(HCV.z+1e-10);return vec3(HCV.x,S,HCV.z);}\n\
vec2 rotateVec2(vec2 v,float r){vec2 cs=vec2(cos(r),sin(r));return vec2(v.x*cs.x-v.y*cs.y,v.x*cs.y+v.y*cs.x);}\n\
";

pub(crate) const WE_COMMON_BLENDING_H: &str = "\
vec3 _we_RGBToHSL(vec3 color){vec3 hsl;float fmin=min(min(color.r,color.g),color.b);float fmax=max(max(color.r,color.g),color.b);float delta=fmax-fmin;hsl.z=(fmax+fmin)/2.0;if(delta==0.0){hsl.x=0.0;hsl.y=0.0;}else{if(hsl.z<0.5)hsl.y=delta/(fmax+fmin);else hsl.y=delta/(2.0-fmax-fmin);float dR=(((fmax-color.r)/6.0)+(delta/2.0))/delta;float dG=(((fmax-color.g)/6.0)+(delta/2.0))/delta;float dB=(((fmax-color.b)/6.0)+(delta/2.0))/delta;if(color.r==fmax)hsl.x=dB-dG;else if(color.g==fmax)hsl.x=(1.0/3.0)+dR-dB;else hsl.x=(2.0/3.0)+dG-dR;if(hsl.x<0.0)hsl.x+=1.0;else if(hsl.x>1.0)hsl.x-=1.0;}return hsl;}\n\
float _we_HueToRGB(float f1,float f2,float hue){if(hue<0.0)hue+=1.0;else if(hue>1.0)hue-=1.0;if((6.0*hue)<1.0)return f1+(f2-f1)*6.0*hue;if((2.0*hue)<1.0)return f2;if((3.0*hue)<2.0)return f1+(f2-f1)*((2.0/3.0)-hue)*6.0;return f1;}\n\
vec3 _we_HSLToRGB(vec3 hsl){if(hsl.y==0.0)return vec3(hsl.z);float f2=hsl.z<0.5?hsl.z*(1.0+hsl.y):(hsl.z+hsl.y)-(hsl.y*hsl.z);float f1=2.0*hsl.z-f2;return vec3(_we_HueToRGB(f1,f2,hsl.x+(1.0/3.0)),_we_HueToRGB(f1,f2,hsl.x),_we_HueToRGB(f1,f2,hsl.x-(1.0/3.0)));}\n\
vec4 Desaturate(vec3 color,float Desaturation){vec3 grayXfer=vec3(0.3,0.59,0.11);vec3 gray=vec3(dot(grayXfer,color));return vec4(mix(color,gray,Desaturation),1.0);}\n\
#define BlendLinearDodgef(base,blend) (base+blend)\n\
#define BlendLinearBurnf(base,blend) max(base+blend-1.0,0.0)\n\
#define BlendLightenf(base,blend) max(blend,base)\n\
#define BlendDarkenf(base,blend) min(blend,base)\n\
#define BlendScreenf(base,blend) (1.0-((1.0-base)*(1.0-blend)))\n\
#define BlendOverlayf(base,blend) (base<0.5?(2.0*base*blend):(1.0-2.0*(1.0-base)*(1.0-blend)))\n\
#define BlendSoftLightf(base,blend) ((blend<0.5)?(2.0*base*blend+base*base*(1.0-2.0*blend)):(sqrt(base)*(2.0*blend-1.0)+2.0*base*(1.0-blend)))\n\
#define BlendColorDodgef(base,blend) ((blend==1.0)?blend:min(base/(1.0-blend),1.0))\n\
#define BlendColorBurnf(base,blend) ((blend==0.0)?blend:max((1.0-((1.0-base)/blend)),0.0))\n\
#define BlendVividLightf(base,blend) ((blend<0.5)?BlendColorBurnf(base,(2.0*blend)):BlendColorDodgef(base,(2.0*(blend-0.5))))\n\
#define BlendPinLightf(base,blend) ((blend<0.5)?BlendDarkenf(base,(2.0*blend)):BlendLightenf(base,(2.0*(blend-0.5))))\n\
#define BlendHardMixf(base,blend) ((BlendVividLightf(base,blend)<0.5)?0.0:1.0)\n\
#define BlendReflectf(base,blend) ((blend==1.0)?blend:min(base*base/(1.0-blend),1.0))\n\
#define BlendNormal(base,blend) (blend)\n\
#define BlendLighten BlendLightenf\n\
#define BlendDarken BlendDarkenf\n\
#define BlendMultiply(base,blend) (base*blend)\n\
#define BlendAverage(base,blend) ((base+blend)/2.0)\n\
#define BlendAdd(base,blend) min(base+blend,vec3(1.0))\n\
#define BlendSubstract(base,blend) max(base+blend-vec3(1.0),vec3(0.0))\n\
#define BlendDifference(base,blend) abs(base-blend)\n\
#define BlendNegation(base,blend) (vec3(1.0)-abs(vec3(1.0)-base-blend))\n\
#define BlendExclusion(base,blend) (base+blend-2.0*base*blend)\n\
#define BlendScreen(base,blend) vec3(BlendScreenf(base.r,blend.r),BlendScreenf(base.g,blend.g),BlendScreenf(base.b,blend.b))\n\
#define BlendOverlay(base,blend) vec3(BlendOverlayf(base.r,blend.r),BlendOverlayf(base.g,blend.g),BlendOverlayf(base.b,blend.b))\n\
#define BlendSoftLight(base,blend) vec3(BlendSoftLightf(base.r,blend.r),BlendSoftLightf(base.g,blend.g),BlendSoftLightf(base.b,blend.b))\n\
#define BlendHardLight(base,blend) BlendOverlay(blend,base)\n\
#define BlendColorDodge(base,blend) vec3(BlendColorDodgef(base.r,blend.r),BlendColorDodgef(base.g,blend.g),BlendColorDodgef(base.b,blend.b))\n\
#define BlendColorBurn(base,blend) vec3(BlendColorBurnf(base.r,blend.r),BlendColorBurnf(base.g,blend.g),BlendColorBurnf(base.b,blend.b))\n\
#define BlendLinearLight(base,blend) vec3(BlendLinearLightf(base.r,blend.r),BlendLinearLightf(base.g,blend.g),BlendLinearLightf(base.b,blend.b))\n\
#define BlendVividLight(base,blend) vec3(BlendVividLightf(base.r,blend.r),BlendVividLightf(base.g,blend.g),BlendVividLightf(base.b,blend.b))\n\
#define BlendPinLight(base,blend) vec3(BlendPinLightf(base.r,blend.r),BlendPinLightf(base.g,blend.g),BlendPinLightf(base.b,blend.b))\n\
#define BlendHardMix(base,blend) vec3(BlendHardMixf(base.r,blend.r),BlendHardMixf(base.g,blend.g),BlendHardMixf(base.b,blend.b))\n\
#define BlendReflect(base,blend) vec3(BlendReflectf(base.r,blend.r),BlendReflectf(base.g,blend.g),BlendReflectf(base.b,blend.b))\n\
#define BlendGlow(base,blend) BlendReflect(blend,base)\n\
#define BlendPhoenix(base,blend) (min(base,blend)-max(base,blend)+vec3(1.0))\n\
#define BlendLinearDodge(base,blend) min(base+blend,vec3(1.0))\n\
#define BlendLinearBurn(base,blend) max(base+blend-vec3(1.0),vec3(0.0))\n\
#define BlendTint(base,blend) (vec3(max(base.x,max(base.y,base.z)))*blend)\n\
#define BlendOpacity(base,opacity,BlendFunc,blend) mix(base,BlendFunc(base,blend),opacity)\n\
vec3 BlendHue(vec3 base,vec3 blend){vec3 b=_we_RGBToHSL(base);return _we_HSLToRGB(vec3(_we_RGBToHSL(blend).r,b.g,b.b));}\n\
vec3 BlendSaturation(vec3 base,vec3 blend){vec3 b=_we_RGBToHSL(base);return _we_HSLToRGB(vec3(b.r,_we_RGBToHSL(blend).g,b.b));}\n\
vec3 BlendColor(vec3 base,vec3 blend){vec3 b=_we_RGBToHSL(blend);return _we_HSLToRGB(vec3(b.r,b.g,_we_RGBToHSL(base).b));}\n\
vec3 BlendLuminosity(vec3 base,vec3 blend){vec3 b=_we_RGBToHSL(base);return _we_HSLToRGB(vec3(b.r,b.g,_we_RGBToHSL(blend).b));}\n\
vec3 ApplyBlending(const int blendMode,in vec3 A,in vec3 B,in float opacity){\n\
#if BLENDMODE==1\nreturn mix(A,BlendDarken(A,B),opacity);\n#endif\n\
#if BLENDMODE==2\nreturn mix(A,BlendMultiply(A,B),opacity);\n#endif\n\
#if BLENDMODE==3\nreturn mix(A,BlendColorBurn(A,B),opacity);\n#endif\n\
#if BLENDMODE==4\nreturn mix(A,BlendSubstract(A,B),opacity);\n#endif\n\
#if BLENDMODE==6\nreturn mix(A,BlendLighten(A,B),opacity);\n#endif\n\
#if BLENDMODE==7\nreturn mix(A,BlendScreen(A,B),opacity);\n#endif\n\
#if BLENDMODE==8\nreturn mix(A,BlendColorDodge(A,B),opacity);\n#endif\n\
#if BLENDMODE==9\nreturn mix(A,BlendAdd(A,B),opacity);\n#endif\n\
#if BLENDMODE==11\nreturn mix(A,BlendOverlay(A,B),opacity);\n#endif\n\
#if BLENDMODE==12\nreturn mix(A,BlendSoftLight(A,B),opacity);\n#endif\n\
#if BLENDMODE==18\nreturn mix(A,BlendDifference(A,B),opacity);\n#endif\n\
#if BLENDMODE==26\nreturn mix(A,BlendHue(A,B),opacity);\n#endif\n\
#if BLENDMODE==27\nreturn mix(A,BlendSaturation(A,B),opacity);\n#endif\n\
#if BLENDMODE==28\nreturn mix(A,BlendColor(A,B),opacity);\n#endif\n\
#if BLENDMODE==29\nreturn mix(A,BlendLuminosity(A,B),opacity);\n#endif\n\
return mix(A,B,opacity);}\n\
";

// squareToQuad from WE's common_perspective.h (assets/shaders/common_perspective.h),
// verbatim math. Builds the unit-square→quad homography for the `perspective`
// distort effect; its vert does inverse(squareToQuad(p0..p3)) * vec3(uv,1). The
// file's own mat3 inverse is behind `#if HLSL`, which our shaderc/GLSL path never
// defines (GLSL 4.50 has a builtin inverse), so only squareToQuad is carried.
pub(crate) const WE_COMMON_PERSPECTIVE_H: &str = "\
mat3 squareToQuad(vec2 p0,vec2 p1,vec2 p2,vec2 p3){\n\
mat3 m=mat3(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);\n\
float dx0=p0.x;float dy0=p0.y;float dx1=p1.x;float dy1=p1.y;\n\
float dx2=p3.x;float dy2=p3.y;float dx3=p2.x;float dy3=p2.y;\n\
float diffx1=dx1-dx3;float diffy1=dy1-dy3;float diffx2=dx2-dx3;float diffy2=dy2-dy3;\n\
float det=diffx1*diffy2-diffx2*diffy1;\n\
float sumx=dx0-dx1+dx3-dx2;float sumy=dy0-dy1+dy3-dy2;\n\
if(det==0.0||(sumx==0.0&&sumy==0.0)){\n\
m[0][0]=dx1-dx0;m[0][1]=dy1-dy0;m[0][2]=0.0;\n\
m[1][0]=dx3-dx1;m[1][1]=dy3-dy1;m[1][2]=0.0;\n\
m[2][0]=dx0;m[2][1]=dy0;m[2][2]=1.0;\n\
return m;}else{\n\
float ovdet=1.0/det;\n\
float g=(sumx*diffy2-diffx2*sumy)*ovdet;\n\
float h=(diffx1*sumy-sumx*diffy1)*ovdet;\n\
m[0][0]=dx1-dx0+g*dx1;m[0][1]=dy1-dy0+g*dy1;m[0][2]=g;\n\
m[1][0]=dx2-dx0+h*dx2;m[1][1]=dy2-dy0+h*dy2;m[1][2]=h;\n\
m[2][0]=dx0;m[2][1]=dy0;m[2][2]=1.0;\n\
return m;}}\n\
";

/// A single scalar-or-vector uniform to be packed into the effect UBO.
/// `glsl_type` is the GLSL type token ("float", "vec2", "vec3", "vec4", "mat4", …).
#[derive(Debug, Clone)]
pub struct UniformEntry {
    pub key: String,
    /// The annotation's UI label, when it differs from `key`. Scenes key their
    /// `constantshadervalues` by either name (edgedetection's
    /// `ui_editor_properties_outline_background` IS `outlinecolorbg`).
    pub label: Option<String>,
    pub glsl_type: String,
    pub default: [f32; 4],
}

#[derive(Debug)]
pub struct TranslatedShader {
    pub wgsl: String,
    pub vert_wgsl: Option<String>,
    pub uniform_keys: Vec<UniformEntry>,
    pub texture_count: usize,
    /// Vertex attributes (glsl type, name) in declaration order when the real
    /// WE vertex shader was translated; empty for the synthetic-VS path.
    pub attributes: Vec<(String, String)>,
    /// True when the shader reads `g_AudioSpectrum*` — the renderer only
    /// starts audio capture for scenes that actually react to it.
    pub uses_audio: bool,
}

pub fn translate(model: &ShaderModel) -> Result<TranslatedShader> {
    translate_full(model, None)
}

/// Translate the fragment shader and, when `vert_glsl` is given, the real WE
/// vertex shader. Both stages share one UBO layout (union of their scalar
/// uniforms, fragment's first) at group(1) binding(0), and varying locations
/// are matched by name so the naga-generated interfaces line up.
#[tracing::instrument(target = "shader", level = "trace", skip_all, fields(shader = %model.name, real_vs = vert_glsl.is_some()))]
pub fn translate_full(model: &ShaderModel, vert_glsl: Option<&str>) -> Result<TranslatedShader> {
    tracing::trace!(target: "shader", "translating GLSL → WGSL");

    // Unroll literal-indexed array varyings (`varying vec2 v[7]` → `v_0`,
    // `v_1`, ...) before anything else touches the source — naga's SPIR-V
    // reader rejects array-typed entry-point I/O outright, so this has to
    // happen up front. `unroll_array_varyings` was already fully
    // implemented and tested (see the tests below) but was never actually
    // wired into this function — real content that uses this exact pattern
    // (godrays_gaussian.frag's `v_TexCoord[13|7|3]` per KERNEL combo, its
    // own motivating comment) failed to transpile as a result. Rebinding
    // `model` to an owned, patched clone means every function below that
    // takes `&ShaderModel` sees the unrolled source with no signature
    // changes needed.
    let mut model = model.clone();
    model.frag_glsl = unroll_array_varyings(&model.frag_glsl);
    let vert_unrolled = vert_glsl.map(unroll_array_varyings);
    let vert_glsl = vert_unrolled.as_deref();
    let model = &model;

    // Shaders that still have an array varying at this point use a
    // variable index somewhere in the body (a `for` loop reading `tex[i]`)
    // — `unroll_array_varyings` deliberately leaves those alone rather than
    // unrolling a declaration out from under a reference that's still
    // there (see its own doc comment), so this honestly rejects them: naga's
    // SPIR-V reader rejects array-typed entry-point I/O either way.
    for line in model.frag_glsl.lines() {
        let t = line.trim();
        if t.starts_with("varying ") || t.starts_with("attribute ") {
            let tok: Vec<&str> = t.split_whitespace().collect();
            if tok.len() >= 3 {
                let name_raw = t
                    .split("//")
                    .next()
                    .unwrap_or(t)
                    .split_whitespace()
                    .nth(2)
                    .unwrap_or("")
                    .trim_end_matches(';');
                if name_raw.contains('[') {
                    anyhow::bail!(
                        "array varying '{}' not supported by naga — skipping shader '{}'",
                        name_raw,
                        model.name
                    );
                }
            }
        }
    }

    // Some workshop shaders declare a varying `vec2` in the vertex stage
    // but `vec4` in the fragment (WE's own compiler links them anyway) —
    // upgrade the vertex side so our strict pipeline's interface matches
    // (the reference does the same, ShaderUnit.cpp
    // applyLinkedVaryingCompatibility).
    let vert_patched: Option<String> =
        vert_glsl.map(|v| harmonize_varying_widths(v, &model.frag_glsl));
    let vert_glsl = vert_patched.as_deref();

    // Union scalar-uniform list: fragment's declarations first (their order
    // defines the front of the UBO), then vertex-only ones appended. Both
    // stages emit the SAME block so std140 offsets agree.
    let frag_scalars = collect_scalar_uniforms(&model.frag_glsl);
    let vert_only_scalars: Vec<(String, String)> = vert_glsl
        .map(|v| {
            collect_scalar_uniforms(v)
                .into_iter()
                .filter(|(_, name)| !frag_scalars.iter().any(|(_, f)| f == name))
                .collect()
        })
        .unwrap_or_default();

    let glsl = preprocess_frag(model, &vert_only_scalars);
    let spv = glsl_to_spirv_with_repairs(&glsl, naga::ShaderStage::Fragment)
        .context("GLSL→SPIR-V compilation failed")?;
    let wgsl = spirv_to_wgsl(&spv).context("SPIR-V→WGSL conversion failed")?;

    // glsl_name → (material_key, typed default) from `// {"material": ...}`
    // annotations, collected from BOTH stages (fragment wins on conflicts).
    let mut annotated: std::collections::HashMap<String, (String, Option<String>, [f32; 4])> =
        std::collections::HashMap::new();
    if let Some(v) = vert_glsl {
        for meta in crate::engine::gpu::shaders::uniform_meta::parse_uniform_metadata(v) {
            annotated.insert(
                meta.uniform_name.clone(),
                (
                    meta.material_key
                        .unwrap_or_else(|| meta.uniform_name.clone()),
                    meta.label.clone(),
                    meta.default_value
                        .as_ref()
                        .map(json_default_vec4)
                        .unwrap_or([0.0; 4]),
                ),
            );
        }
    }
    for u in &model.value_uniforms {
        annotated.insert(
            u.glsl_name.clone(),
            (u.material_key.clone(), u.label.clone(), u.default.as_vec4()),
        );
    }

    // The UBO contains all union scalars in emission order. We store the GLSL
    // type so the renderer can compute proper std140 layout/padding.
    let uniform_keys: Vec<UniformEntry> = frag_scalars
        .iter()
        .chain(vert_only_scalars.iter())
        .map(|(ty, name)| {
            let (key, label, default) = annotated
                .get(name)
                .cloned()
                .unwrap_or_else(|| (name.clone(), None, [0.0; 4]));
            UniformEntry {
                key,
                label,
                glsl_type: ty.clone(),
                default,
            }
        })
        .collect();

    let inputs = frag_inputs(&model.frag_glsl);

    // Prefer the real WE vertex shader: it computes scrolled/scaled UVs, wave
    // phases, blur offsets etc. Fall back to a synthetic passthrough VS when
    // translation fails (the caller falls back again if the pipeline fails).
    let mut attributes: Vec<(String, String)> = Vec::new();
    let mut vert_wgsl: Option<String> = None;
    if let Some(v) = vert_glsl {
        let union_scalars: Vec<(String, String)> = frag_scalars
            .iter()
            .chain(vert_only_scalars.iter())
            .cloned()
            .collect();
        match preprocess_vert_matched(v, &union_scalars, &inputs, &model.effective_combos()) {
            Ok((vsrc, attrs)) => {
                match glsl_to_spirv_with_repairs(&vsrc, naga::ShaderStage::Vertex)
                    .and_then(|spv| spirv_to_wgsl(&spv))
                {
                    Ok(w) => {
                        vert_wgsl = Some(w);
                        attributes = attrs;
                    }
                    Err(e) => tracing::warn!(
                        target: "shader",
                        "'{}': real VS translation failed ({e}); using synthetic VS",
                        model.name
                    ),
                }
            }
            Err(e) => tracing::warn!(
                target: "shader",
                "'{}': real VS preprocess failed ({e}); using synthetic VS",
                model.name
            ),
        }
    }

    // Synthetic fallback: outputs match the fragment shader's declared varyings.
    // vs_we_effect always outputs (vec4 v_TexCoord, vec2 v_Scroll) which causes
    // a pipeline validation panic when the frag expects different types.
    if vert_wgsl.is_none() && needs_custom_vs(&inputs) {
        let vert_glsl = synthetic_vs_glsl(&inputs);
        vert_wgsl = glsl_to_spirv(&vert_glsl, naga::ShaderStage::Vertex)
            .and_then(|spv| spirv_to_wgsl(&spv))
            .ok();
    }

    Ok(TranslatedShader {
        wgsl,
        vert_wgsl,
        uniform_keys,
        texture_count: model.texture_slots.len(),
        attributes,
        uses_audio: model.frag_glsl.contains("AudioSpectrum"),
    })
}

/// Convert an annotation `default` JSON value into a vec4-shaped default.
fn json_default_vec4(val: &serde_json::Value) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    match val {
        serde_json::Value::Number(n) => {
            let f = n.as_f64().unwrap_or(0.0) as f32;
            out = [f; 4];
        }
        serde_json::Value::String(s) => {
            for (i, p) in s.split_whitespace().take(4).enumerate() {
                out[i] = p.parse().unwrap_or(0.0);
            }
        }
        serde_json::Value::Array(a) => {
            for (i, p) in a.iter().take(4).enumerate() {
                out[i] = p.as_f64().unwrap_or(0.0) as f32;
            }
        }
        _ => {}
    }
    out
}

// ── GLSL → SPIR-V via shaderc ────────────────────────────────────────────────
//
// shaderc wraps glslangValidator as a linkable library — no external binary
// required, works identically on Linux (Vulkan) and macOS (Metal via wgpu).

// ── SPIR-V → WGSL via naga ───────────────────────────────────────────────────

// ── Preprocessors ─────────────────────────────────────────────────────────────
//
// Convert WE-style GLSL (OpenGL 2.x) to Vulkan-compatible GLSL 4.50 with
// separate texture2D + sampler bindings (naga's SPIR-V reader doesn't support
// combined sampler2D loads via OpLoad; it requires OpSampledImage from separate parts).
//
// Binding layout matches our GPU renderer's bind groups:
//   Group 0, binding 0    : texture 0 (source / framebuffer)
//   Group 0, binding 1    : shared sampler
//   Group 0, binding 2    : (buffer — unused by dynamic shader, reserved)
//   Group 0, binding 3    : texture 1
//   Group 0, binding 4    : texture 2
//   Group 0, binding 5    : texture 3
//   Group 1, binding 0    : effect UBO (scalar uniforms)

fn preprocess_frag(model: &ShaderModel, extra_scalars: &[(String, String)]) -> String {
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
fn preprocess_vert_matched(
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

// ── Synthetic vertex shader helpers ──────────────────────────────────────────

/// Extract (type, name) pairs from fragment `varying` declarations (array varyings excluded).
fn frag_inputs(frag_glsl: &str) -> Vec<(String, String)> {
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
fn needs_custom_vs(inputs: &[(String, String)]) -> bool {
    match inputs {
        [] => false,
        [(t0, _)] => t0 != "vec4",
        [(t0, _), (t1, _)] => t0 != "vec4" || t1 != "vec2",
        _ => true, // more than 2 varyings → vs_we_effect definitely can't match
    }
}

/// Build a Vulkan GLSL 4.50 vertex shader that outputs every declared varying as a
/// simple UV-derived value.  No uniforms — this shader works with any bind group layout.
fn synthetic_vs_glsl(inputs: &[(String, String)]) -> String {
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

#[cfg(test)]
mod tests;
