use crate::engine::model::ShaderModel;
use anyhow::{Context, Result};
use std::collections::HashMap;

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

/// Collect non-sampler `uniform` declarations (type, name) in source order.
/// The macro name of a `#define NAME ...` line, if this line is one.
fn redefined_macro(trimmed_line: &str) -> Option<&str> {
    let rest = trimmed_line.strip_prefix("#define")?;
    if !rest.starts_with([' ', '\t']) {
        return None;
    }
    let name = rest.trim_start();
    let end = name
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(name.len());
    (end > 0).then(|| &name[..end])
}

fn collect_scalar_uniforms(src: &str) -> Vec<(String, String)> {
    let mut scalars = Vec::new();
    for line in src.lines() {
        let decl = line.trim().split("//").next().unwrap_or("").trim();
        let tok: Vec<&str> = decl.split_whitespace().collect();
        if tok.len() >= 3 && tok[0] == "uniform" {
            let ty = tok[1];
            if !ty.starts_with("sampler") && !ty.is_empty() {
                let name = tok[2].trim_end_matches(';');
                // Array uniforms (audio spectra etc.) can't be packed by the
                // renderer's flat UBO writer; skip them (they read as zero).
                if !name.contains('[') {
                    scalars.push((ty.to_string(), name.to_string()));
                }
            }
        }
    }
    scalars
}

/// True if every `name[...]` access in `src` has a plain non-negative-integer
/// literal inside the brackets (no variables/expressions).
fn all_indices_are_literal(src: &str, name: &str) -> bool {
    let pattern = format!("{name}[");
    let mut search_from = 0;
    while let Some(rel) = src[search_from..].find(pattern.as_str()) {
        let start = search_from + rel + pattern.len();
        let Some(end_rel) = src[start..].find(']') else {
            return false; // unterminated — be conservative
        };
        let content = src[start..start + end_rel].trim();
        if content.parse::<usize>().is_err() {
            return false;
        }
        search_from = start + end_rel + 1;
    }
    true
}

/// Finds the balanced-bracket span starting at `open_pos` (which must point
/// at an `open` byte), skipping bracket-like bytes inside `//` line comments.
/// Returns `(span including both brackets, index just past the closer)`.
fn extract_balanced(s: &str, open_pos: usize, open: u8, close: u8) -> Option<(&str, usize)> {
    let bytes = s.as_bytes();
    if bytes.get(open_pos).copied() != Some(open) {
        return None;
    }
    let mut depth = 0i32;
    let mut i = open_pos;
    let mut in_line_comment = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_line_comment {
            if b == b'\n' {
                in_line_comment = false;
            }
            i += 1;
            continue;
        }
        if b == b'/' && bytes.get(i + 1) == Some(&b'/') {
            in_line_comment = true;
            i += 2;
            continue;
        }
        if b == open {
            depth += 1;
        } else if b == close {
            depth -= 1;
            if depth == 0 {
                return Some((&s[open_pos..=i], i + 1));
            }
        }
        i += 1;
    }
    None
}

/// Parses a `for` header's three `;`-separated clauses into
/// `(loop_var_name, iteration_count)`, only recognizing the exact shape real
/// WE shaders use: `int NAME = 0; NAME < N; NAME++` (or `++NAME`, or `<=`).
/// Returns `None` for anything else — callers must leave the input untouched
/// rather than guess.
fn parse_simple_for_header(header_inner: &str) -> Option<(String, usize)> {
    let clauses: Vec<&str> = header_inner.split(';').collect();
    if clauses.len() != 3 {
        return None;
    }
    let init = clauses[0].trim();
    let init = init
        .strip_prefix("int ")
        .or_else(|| init.strip_prefix("uint "))?;
    let (name, start_val) = init.split_once('=')?;
    let name = name.trim();
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    if start_val.trim().parse::<i64>() != Ok(0) {
        return None; // only handle the universal `= 0` starting point
    }

    let cond = clauses[1].trim();
    let (cmp_name, op, end_str) = if let Some((n, v)) = cond.split_once("<=") {
        (n.trim(), "<=", v.trim())
    } else if let Some((n, v)) = cond.split_once('<') {
        (n.trim(), "<", v.trim())
    } else {
        return None;
    };
    if cmp_name != name {
        return None;
    }
    let end_val: i64 = end_str.parse().ok()?;
    let count = match op {
        "<" => end_val,
        "<=" => end_val + 1,
        _ => return None,
    };
    if count <= 0 || count > 64 {
        return None; // sanity bound — real shaders use small fixed kernels
    }

    let incr = clauses[2].trim();
    if incr != format!("{name}++") && incr != format!("++{name}") {
        return None;
    }

    Some((name.to_string(), count as usize))
}

/// Unrolls simple, fixed-bound `for (int i = 0; i < N; i++) { BODY }` loops
/// into N literal copies of `BODY` with `i` replaced by each literal index.
///
/// Needed alongside `unroll_array_varyings`: naga's GLSL frontend rejects
/// array-typed varyings outright, and that pass only unrolls *literally*
/// indexed arrays — but some real shaders (e.g. `blur_downsample4.frag`) read
/// a fixed-size array varying through exactly this kind of loop
/// (`for (i...) texSample2D(tex, v_TexCoord[i])`). Unrolling the loop first
/// turns `v_TexCoord[i]` into `v_TexCoord[0]`, `v_TexCoord[1]`, ... so the
/// array-varying pass can then unroll the declaration too, and the shader
/// translates instead of being skipped.
///
/// Only ever touches text that matches this exact shape; anything else (a
/// different loop form, a non-zero start, a non-literal bound, nested loops
/// it can't balance) is left completely untouched — no partial rewrites.
pub fn unroll_simple_for_loops(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    loop {
        let Some(for_rel) = rest.find("for") else {
            out.push_str(rest);
            break;
        };
        let before_ok = for_rel == 0 || !is_word_char(rest.as_bytes()[for_rel - 1]);
        let after_rel = for_rel + 3;
        let after_ok = rest
            .as_bytes()
            .get(after_rel)
            .is_none_or(|&b| !is_word_char(b));
        if !before_ok || !after_ok {
            out.push_str(&rest[..after_rel]);
            rest = &rest[after_rel..];
            continue;
        }

        let paren_start =
            after_rel + rest[after_rel..].len() - rest[after_rel..].trim_start().len();
        if rest.as_bytes().get(paren_start).copied() != Some(b'(') {
            out.push_str(&rest[..after_rel]);
            rest = &rest[after_rel..];
            continue;
        }
        let Some((header, header_end)) = extract_balanced(rest, paren_start, b'(', b')') else {
            out.push_str(&rest[..after_rel]);
            rest = &rest[after_rel..];
            continue;
        };
        let Some((name, count)) = parse_simple_for_header(&header[1..header.len() - 1]) else {
            out.push_str(&rest[..header_end]);
            rest = &rest[header_end..];
            continue;
        };

        let after_header = &rest[header_end..];
        let body_open_rel = after_header.len() - after_header.trim_start().len();
        let body_open_abs = header_end + body_open_rel;
        if rest.as_bytes().get(body_open_abs).copied() != Some(b'{') {
            out.push_str(&rest[..header_end]);
            rest = &rest[header_end..];
            continue;
        }
        let Some((body, body_end)) = extract_balanced(rest, body_open_abs, b'{', b'}') else {
            out.push_str(&rest[..header_end]);
            rest = &rest[header_end..];
            continue;
        };
        let body_inner = &body[1..body.len() - 1];

        out.push_str(&rest[..for_rel]);
        for i in 0..count {
            // Each iteration gets its own `{}` scope, matching the original
            // loop body's scoping — otherwise a variable declared inside the
            // loop (e.g. `vec4 sample = ...`) becomes a flat redeclaration
            // once concatenated N times into the same enclosing scope.
            out.push_str("{\n");
            out.push_str(&rename_reserved_word(body_inner, &name, &i.to_string()));
            out.push_str("\n}\n");
        }
        rest = &rest[body_end..];
    }
    out
}

/// Splits a function-call argument list on *top-level* commas (not inside
/// nested parens), returning each argument's `(text, start_offset)` within
/// `args`.
fn split_top_level_args(args: &str) -> Vec<(&str, usize)> {
    let bytes = args.as_bytes();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut out = Vec::new();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b',' if depth == 0 => {
                out.push((&args[start..i], start));
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push((&args[start..], start));
    out
}

/// True if `s` (already trimmed) is a bare numeric literal: an optional `-`,
/// digits, and an optional `.` plus more digits — no identifiers, swizzles,
/// or function calls.
fn is_bare_number(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let s = s.strip_prefix('-').unwrap_or(s);
    if s.is_empty() {
        return false;
    }
    let mut seen_digit = false;
    let mut seen_dot = false;
    for c in s.chars() {
        match c {
            '0'..='9' => seen_digit = true,
            '.' if !seen_dot => seen_dot = true,
            _ => return false,
        }
    }
    seen_digit
}

fn ensure_float_literal(trimmed: &str) -> String {
    if trimmed.contains('.') {
        trimmed.to_string()
    } else {
        format!("{trimmed}.0")
    }
}

/// Legacy NVIDIA GLSL compilers tolerated `max`/`min` calls with a bare
/// numeric literal as the *first* argument and a vector expression second
/// (e.g. nitro.frag's `max(0, albedo.rgb)`), including when the literal was
/// an integer where a float is needed. Strict Vulkan GLSL (naga/shaderc)
/// requires both: the literal to be a genuine float, *and* the vector
/// argument first — GLSL only defines `genType max(genType x, float y)`, not
/// the reverse. `max`/`min` are commutative, so swapping is always safe (not
/// just when the second argument turns out to be a vector). `clamp`'s
/// numeric arguments are only ever coerced to float, never reordered, since
/// `clamp(x, minVal, maxVal)` isn't commutative.
pub fn coerce_int_literal_builtin_args(src: &str) -> String {
    const COMMUTATIVE_FUNCS: &[&str] = &["max(", "min("];
    const OTHER_FUNCS: &[&str] = &["clamp("];
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    'outer: loop {
        let mut best: Option<(usize, usize, bool)> = None; // (rel_pos, fn_len, commutative)
        for f in COMMUTATIVE_FUNCS {
            if let Some(rel) = rest.find(f) {
                if best.is_none_or(|(bp, ..)| rel < bp) {
                    best = Some((rel, f.len(), true));
                }
            }
        }
        for f in OTHER_FUNCS {
            if let Some(rel) = rest.find(f) {
                if best.is_none_or(|(bp, ..)| rel < bp) {
                    best = Some((rel, f.len(), false));
                }
            }
        }
        let Some((rel, flen, commutative)) = best else {
            out.push_str(rest);
            break 'outer;
        };
        let open_pos = rel + flen - 1; // index of the '('
        out.push_str(&rest[..open_pos]);
        let Some((call, call_end)) = extract_balanced(rest, open_pos, b'(', b')') else {
            out.push_str(&rest[open_pos..open_pos + 1]);
            rest = &rest[open_pos + 1..];
            continue;
        };
        let inner = &call[1..call.len() - 1];
        let args = split_top_level_args(inner);
        let trimmed: Vec<&str> = args.iter().map(|(a, _)| a.trim()).collect();

        let mut fixed_args: Vec<String> = trimmed.iter().map(|t| t.to_string()).collect();
        for t in fixed_args.iter_mut() {
            if is_bare_number(t) {
                *t = ensure_float_literal(t);
            }
        }
        // Only max/min, only the exact 2-arg "literal first, non-literal
        // second" shape — swap so the literal ends up in the (required)
        // second position.
        if commutative
            && fixed_args.len() == 2
            && is_bare_number(trimmed[0])
            && !is_bare_number(trimmed[1])
        {
            fixed_args.swap(0, 1);
        }

        out.push('(');
        out.push_str(&fixed_args.join(", "));
        out.push(')');
        rest = &rest[call_end..];
    }
    out
}

/// WE's shader compiler tolerates truncating a wider vector to match a
/// narrower swizzled assignment target — e.g. `albedo.rgb = mix(albedo,
/// newAlbedo, mask)` where `albedo` is `vec4` but the assignment (and
/// `newAlbedo`) are `vec3`. Strict GLSL (naga/shaderc) requires `mix()`'s
/// first two arguments to have the same component count and rejects this.
/// When an assignment's LHS is `IDENT.SWIZZLE` and the RHS is a `mix(...)`
/// call whose first argument is the same bare `IDENT`, apply the LHS's
/// swizzle to that argument too: a no-op when the types already matched, and
/// exactly the truncation WE's compiler performed implicitly otherwise.
pub fn coerce_swizzle_mismatched_mix_arg(src: &str) -> String {
    fn is_swizzle(s: &str) -> bool {
        (1..=4).contains(&s.len())
            && (s.bytes().all(|b| b"xyzw".contains(&b)) || s.bytes().all(|b| b"rgba".contains(&b)))
    }
    fn is_ident(s: &str) -> bool {
        !s.is_empty()
            && s.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
            && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    }

    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    loop {
        let Some(rel) = rest.find("mix(") else {
            out.push_str(rest);
            break;
        };
        // Skip matches that are just the tail of a longer identifier (e.g. a
        // hypothetical `remix(`).
        if rel > 0 {
            let prev = rest.as_bytes()[rel - 1];
            if prev.is_ascii_alphanumeric() || prev == b'_' {
                out.push_str(&rest[..rel + 4]);
                rest = &rest[rel + 4..];
                continue;
            }
        }
        let open_pos = rel + 3; // index of '(' in "mix("
        let Some((call, call_end)) = extract_balanced(rest, open_pos, b'(', b')') else {
            out.push_str(&rest[..open_pos + 1]);
            rest = &rest[open_pos + 1..];
            continue;
        };

        // Find the assignment this call is the RHS of: walk back over
        // whitespace and a trailing '=' immediately preceding "mix(".
        let before_trim = rest[..rel].trim_end();
        let Some(lhs_and_stmt) = before_trim.strip_suffix('=') else {
            out.push_str(&rest[..call_end]);
            rest = &rest[call_end..];
            continue;
        };
        // '==' is a comparison, not an assignment — reject if another '='
        // immediately precedes.
        if lhs_and_stmt.ends_with('=') {
            out.push_str(&rest[..call_end]);
            rest = &rest[call_end..];
            continue;
        }
        let lhs = lhs_and_stmt.trim_end();
        let stmt_start = lhs.rfind(['\n', ';', '{', '}']).map(|i| i + 1).unwrap_or(0);
        let lhs_expr = lhs[stmt_start..].trim();
        let Some((base, swizzle)) = lhs_expr.split_once('.') else {
            out.push_str(&rest[..call_end]);
            rest = &rest[call_end..];
            continue;
        };
        if !is_ident(base) || !is_swizzle(swizzle) {
            out.push_str(&rest[..call_end]);
            rest = &rest[call_end..];
            continue;
        }

        let inner = &call[1..call.len() - 1];
        let args = split_top_level_args(inner);
        if args.is_empty() || args[0].0.trim() != base {
            out.push_str(&rest[..call_end]);
            rest = &rest[call_end..];
            continue;
        }

        out.push_str(&rest[..open_pos + 1]);
        let mut fixed: Vec<String> = args.iter().map(|(a, _)| a.trim().to_string()).collect();
        fixed[0] = format!("{base}.{swizzle}");
        out.push_str(&fixed.join(", "));
        out.push(')');
        rest = &rest[call_end..];
    }
    out
}

pub fn unroll_array_varyings(src: &str) -> String {
    // name -> (qualifier, type, count)
    let mut arrays: Vec<(String, String, String, usize)> = Vec::new();
    for line in src.lines() {
        let t = line.trim();
        let (qualifier, rest) = if let Some(r) = t.strip_prefix("varying ") {
            ("varying", r)
        } else if let Some(r) = t.strip_prefix("attribute ") {
            ("attribute", r)
        } else {
            continue;
        };
        let rest = rest.split("//").next().unwrap_or(rest).trim();
        let rest = rest.trim_end_matches(';').trim();
        let mut parts = rest.splitn(2, char::is_whitespace);
        let Some(ty) = parts.next() else { continue };
        let Some(name_and_dim) = parts.next().map(str::trim) else {
            continue;
        };
        let Some(bracket) = name_and_dim.find('[') else {
            continue;
        };
        let name = &name_and_dim[..bracket];
        let Some(close) = name_and_dim[bracket + 1..].find(']') else {
            continue;
        };
        let dim_str = &name_and_dim[bracket + 1..bracket + 1 + close];
        let Ok(n) = dim_str.parse::<usize>() else {
            continue;
        };
        if n == 0 {
            continue;
        }
        arrays.push((name.to_string(), qualifier.to_string(), ty.to_string(), n));
    }

    if arrays.is_empty() {
        return src.to_string();
    }

    // WE shaders often declare the same array varying multiple times at
    // different sizes under mutually-exclusive `#if COMBO==N` branches (e.g.
    // godrays_gaussian.frag's `v_TexCoord[13|7|3]` per KERNEL quality level).
    // Since this pass runs before `#if` resolution, all branches are still
    // literally present; collapse same-named declarations to one, sized to
    // the largest variant seen (extra unused varyings are harmless).
    let mut merged: Vec<(String, String, String, usize)> = Vec::new();
    for (name, qualifier, ty, n) in arrays {
        if let Some(existing) = merged.iter_mut().find(|(en, ..)| *en == name) {
            existing.3 = existing.3.max(n);
        } else {
            merged.push((name, qualifier, ty, n));
        }
    }
    // Only unroll names indexed *exclusively* by literal integers everywhere
    // in the shader. Some real shaders (e.g. blur_downsample4.frag) loop over
    // the array with a variable index (`for (i...) tex[i]`); rewriting just
    // the literal-looking declaration while leaving `tex[i]` untouched would
    // unroll the declaration out from under a body reference that's still
    // there, producing "undeclared identifier" instead of a clean bail-out.
    // Leaving such names alone lets the existing "array varying not
    // supported" check below reject the shader honestly instead.
    let arrays: Vec<_> = merged
        .into_iter()
        .filter(|(name, ..)| all_indices_are_literal(src, name))
        .collect();

    // Re-extract this line's declared array name (if any) the same way as the
    // collection pass, so the declaration line is matched precisely instead
    // of by fragile substring checks against the qualifier/type prefix.
    let declared_name = |t: &str| -> Option<String> {
        let rest = t
            .strip_prefix("varying ")
            .or_else(|| t.strip_prefix("attribute "))?;
        let rest = rest.split("//").next().unwrap_or(rest).trim();
        let rest = rest.trim_end_matches(';').trim();
        let mut parts = rest.splitn(2, char::is_whitespace);
        parts.next()?;
        let name_and_dim = parts.next()?.trim();
        let bracket = name_and_dim.find('[')?;
        Some(name_and_dim[..bracket].to_string())
    };

    let mut emitted: Vec<String> = Vec::new();
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        let t = line.trim();
        if let Some(decl_name) = declared_name(t) {
            if emitted.contains(&decl_name) {
                // Same name already declared (a smaller #if/#else variant) —
                // drop this line rather than redeclare it.
                continue;
            }
            if let Some((name, qualifier, ty, n)) = arrays.iter().find(|(n, ..)| *n == decl_name) {
                for i in 0..*n {
                    out.push_str(&format!("{qualifier} {ty} {name}_{i};\n"));
                }
                emitted.push(decl_name);
                continue;
            }
        }
        let mut rewritten = line.to_string();
        for (name, _, _, n) in &arrays {
            for i in 0..*n {
                rewritten = rewritten.replace(&format!("{name}[{i}]"), &format!("{name}_{i}"));
            }
        }
        out.push_str(&rewritten);
        out.push('\n');
    }
    out
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
        for meta in crate::engine::shaders::uniform_meta::parse_uniform_metadata(v) {
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

/// Port of the reference's `applyLinkedVaryingCompatibility`
/// (ShaderUnit.cpp:379-415): for every `varying vec4 NAME;` the fragment
/// declares where the vertex declares `varying vec2 NAME;`, upgrade the
/// vertex declaration to vec4 and rewrite the vertex's whole-variable
/// assignments (`NAME = expr;` at statement start — swizzled stores like
/// `NAME.xy = ...` are already type-correct and left alone) to
/// `NAME = vec4(expr, 0.0, 1.0);`. WE's own compiler links such mismatched
/// interfaces silently; strict Vulkan pipelines reject them.
fn harmonize_varying_widths(vert: &str, frag: &str) -> String {
    // Fragment-side vec4 varyings.
    let mut vec4_names: Vec<String> = Vec::new();
    for line in frag.lines() {
        let decl = line.trim().split("//").next().unwrap_or("").trim();
        let tok: Vec<&str> = decl.split_whitespace().collect();
        if tok.len() >= 3 && tok[0] == "varying" && tok[1] == "vec4" {
            let name = tok[2].trim_end_matches(';');
            if !name.contains('[') {
                vec4_names.push(name.to_string());
            }
        }
    }
    if vec4_names.is_empty() {
        return vert.to_string();
    }

    let mut out: Vec<String> = Vec::new();
    let mut upgraded: Vec<String> = Vec::new();
    for line in vert.lines() {
        let decl = line.trim().split("//").next().unwrap_or("").trim();
        let tok: Vec<&str> = decl.split_whitespace().collect();
        if tok.len() >= 3 && tok[0] == "varying" && tok[1] == "vec2" {
            let name = tok[2].trim_end_matches(';');
            if vec4_names.iter().any(|n| n == name) {
                out.push(format!("varying vec4 {name};"));
                upgraded.push(name.to_string());
                continue;
            }
        }
        out.push(line.to_string());
    }
    if upgraded.is_empty() {
        return vert.to_string();
    }

    // Rewrite whole-variable assignments of upgraded varyings.
    for line in &mut out {
        let trimmed = line.trim_start();
        let indent_len = line.len() - trimmed.len();
        for name in &upgraded {
            let Some(rest) = trimmed.strip_prefix(name.as_str()) else {
                continue;
            };
            let rest = rest.trim_start();
            // Plain `=` only (not `.xy =`, `==`, `+=`); expression runs to
            // the trailing `;`.
            let Some(expr_and_tail) = rest.strip_prefix('=') else {
                continue;
            };
            if expr_and_tail.starts_with('=') {
                continue;
            }
            let Some(semi) = expr_and_tail.find(';') else {
                continue;
            };
            let expr = expr_and_tail[..semi].trim();
            let tail = &expr_and_tail[semi + 1..];
            *line = format!(
                "{}{name} = vec4({expr}, 0.0, 1.0);{tail}",
                &line[..indent_len]
            );
            break;
        }
    }
    out.join("\n")
}

/// `glsl_to_spirv` with a bounded repair loop for HLSL-style implicit
/// conversions that strict GLSL rejects but Wallpaper Engine's own (D3D)
/// compiler accepts silently — e.g. `float p = vec2Expr;` (truncates to .x),
/// `int i = -floatVar;`, or a scalar assigned to a vec3. shaderc's error
/// messages carry the line number and both types, which is exactly enough
/// to wrap the right-hand side in the coercion HLSL would have applied,
/// then recompile. Bails out (returning the latest error) once a round
/// fixes nothing or after a few rounds, so pathological shaders still fail
/// cleanly instead of looping.
fn glsl_to_spirv_with_repairs(glsl: &str, stage: naga::ShaderStage) -> Result<Vec<u8>> {
    // shaderc reports line numbers in the *generated* source, which nothing
    // else ever prints — WP_DEBUG_DUMP_GLSL=1 makes those numbers usable.
    let dump = |src: &str| {
        if std::env::var("WP_DEBUG_DUMP_GLSL").is_err() {
            return;
        }
        let path = format!(
            "/tmp/glsl_fail_{:x}.{}",
            src.len() as u64 ^ src.as_bytes().iter().map(|b| *b as u64).sum::<u64>(),
            if stage == naga::ShaderStage::Vertex {
                "vert"
            } else {
                "frag"
            }
        );
        let _ = std::fs::write(&path, src);
        tracing::warn!("dumped failing GLSL to {path}");
    };
    let mut src: Option<String> = None;
    let mut last_err = match glsl_to_spirv(glsl, stage) {
        Ok(spv) => return Ok(spv),
        Err(e) => {
            dump(glsl);
            e
        }
    };
    for _ in 0..5 {
        let current = src.as_deref().unwrap_or(glsl);
        let err = format!("{last_err:#}");
        // Apply both repairs against the same error text: `float % int` becomes
        // `mod()` and HLSL-implicit assignment conversions get their casts.
        // Either may fix a different error the same round.
        let mut patched = repair_float_modulo(current, &err);
        for repair in [
            repair_assignment_conversions,
            repair_bool_conditions,
            repair_input_assignment,
            repair_bool_operands,
            repair_call_arg_widths,
            repair_operand_widths,
        ] {
            let so_far = patched.as_deref().unwrap_or(current);
            if let Some(p) = repair(so_far, &err) {
                patched = Some(p);
            }
        }
        let Some(patched) = patched else {
            return Err(last_err);
        };
        match glsl_to_spirv(&patched, stage) {
            Ok(spv) => return Ok(spv),
            Err(e) => last_err = e,
        }
        src = Some(patched);
    }
    Err(last_err)
}

/// Best-effort classification of a shaderc type description ("temp highp
/// 2-component vector of float", "const float", "temp highp int", ...).
fn parse_shaderc_type(desc: &str) -> Option<(&'static str, u8)> {
    if let Some(pos) = desc.find("-component vector of float") {
        let n: u8 = desc[..pos].split_whitespace().last()?.parse().ok()?;
        return Some(("vec", n));
    }
    if desc.ends_with("float") {
        return Some(("float", 1));
    }
    if desc.ends_with("uint") {
        return Some(("uint", 1));
    }
    if desc.ends_with("int") {
        return Some(("int", 1));
    }
    if desc.ends_with("bool") {
        return Some(("bool", 1));
    }
    None
}

/// Rewrite every `a % b` on `line` to `mod(a, b)`. HLSL's `%` accepts float
/// operands (D3D `fmod` semantics); strict GLSL's `%` is integer-only, so WE
/// shaders that write `float % N` (e.g. simple_audio_bars' bar bucketing) are
/// rejected. `mod()` is GLSL's float remainder and int→float promotion makes
/// the second operand fit. Returns `None` when the line has no modulo.
fn rewrite_modulo_on_line(line: &str) -> Option<String> {
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b == b'.';
    let mut out = line.to_string();
    // Re-scan from the start after each rewrite (indices shift).
    loop {
        let bytes = out.as_bytes();
        // Find a modulo `%` that isn't the compound-assign `%=`.
        let Some(pct) =
            (0..bytes.len()).find(|&i| bytes[i] == b'%' && bytes.get(i + 1) != Some(&b'='))
        else {
            break;
        };

        // Left operand: skip whitespace, then take a parenthesized group (plus
        // any function name in front of it) or a bare identifier/number.
        let mut le = pct;
        while le > 0 && bytes[le - 1].is_ascii_whitespace() {
            le -= 1;
        }
        let ls = if le > 0 && (bytes[le - 1] == b')' || bytes[le - 1] == b']') {
            let (open, close) = if bytes[le - 1] == b')' {
                (b'(', b')')
            } else {
                (b'[', b']')
            };
            let mut depth = 0i32;
            let mut k = le - 1;
            loop {
                if bytes[k] == close {
                    depth += 1;
                } else if bytes[k] == open {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                if k == 0 {
                    break;
                }
                k -= 1;
            }
            let mut s = k;
            while s > 0 && is_ident(bytes[s - 1]) {
                s -= 1;
            }
            s
        } else {
            let mut s = le;
            while s > 0 && is_ident(bytes[s - 1]) {
                s -= 1;
            }
            s
        };

        // Right operand: skip whitespace, then a parenthesized group or a
        // bare identifier/number optionally followed by a call's parens.
        let mut rs = pct + 1;
        while rs < bytes.len() && bytes[rs].is_ascii_whitespace() {
            rs += 1;
        }
        let re = if rs < bytes.len() && (bytes[rs] == b'(' || bytes[rs] == b'[') {
            let close = if bytes[rs] == b'(' { b')' } else { b']' };
            let open = bytes[rs];
            let mut depth = 0i32;
            let mut k = rs;
            while k < bytes.len() {
                if bytes[k] == open {
                    depth += 1;
                } else if bytes[k] == close {
                    depth -= 1;
                    if depth == 0 {
                        k += 1;
                        break;
                    }
                }
                k += 1;
            }
            k
        } else {
            let mut k = rs;
            while k < bytes.len() && is_ident(bytes[k]) {
                k += 1;
            }
            if k < bytes.len() && bytes[k] == b'(' {
                let mut depth = 0i32;
                while k < bytes.len() {
                    if bytes[k] == b'(' {
                        depth += 1;
                    } else if bytes[k] == b')' {
                        depth -= 1;
                        if depth == 0 {
                            k += 1;
                            break;
                        }
                    }
                    k += 1;
                }
            }
            k
        };

        // Bail if either operand came out empty (malformed `%`) — leave as-is.
        if ls >= le || rs >= re {
            break;
        }
        let left = out[ls..le].trim().to_string();
        let right = out[rs..re].trim().to_string();
        out = format!("{}mod({left}, {right}){}", &out[..ls], &out[re..]);
    }
    (out != line).then_some(out)
}

/// Parse `shader.frag:N: error: '%' : wrong operand types` messages and rewrite
/// the flagged line's modulo(s) to `mod()`. Returns `None` when nothing matched.
fn repair_float_modulo(glsl: &str, err_text: &str) -> Option<String> {
    let mut lines: Vec<String> = glsl.lines().map(str::to_string).collect();
    let mut repaired = false;
    for msg in err_text.lines() {
        if !msg.contains("'%'") || !msg.contains("operand") {
            continue;
        }
        let Some(rest) = msg
            .split_once("shader.frag:")
            .or_else(|| msg.split_once("shader.vert:"))
            .map(|(_, r)| r)
        else {
            continue;
        };
        let Some((line_str, _)) = rest.split_once(':') else {
            continue;
        };
        let Ok(line_no) = line_str.trim().parse::<usize>() else {
            continue;
        };
        if let Some(line) = lines.get_mut(line_no.saturating_sub(1)) {
            if let Some(patched) = rewrite_modulo_on_line(line) {
                *line = patched;
                repaired = true;
            }
        }
    }
    repaired.then(|| lines.join("\n"))
}

/// Parse `shader.frag:N: error: '=' : cannot convert from ' X' to ' Y'`
/// messages and wrap the offending line's assignment RHS in the conversion
/// HLSL would apply implicitly. Returns `None` when nothing was repaired.
fn repair_assignment_conversions(glsl: &str, err_text: &str) -> Option<String> {
    let mut lines: Vec<String> = glsl.lines().map(str::to_string).collect();
    let mut repaired = false;

    for msg in err_text.lines() {
        let Some(rest) = msg
            .split_once("shader.frag:")
            .or_else(|| msg.split_once("shader.vert:"))
            .map(|(_, r)| r)
        else {
            continue;
        };
        let Some((line_str, detail)) = rest.split_once(':') else {
            continue;
        };
        let Ok(line_no) = line_str.trim().parse::<usize>() else {
            continue;
        };
        let Some(conv) = detail.split("cannot convert from '").nth(1) else {
            continue;
        };
        // detail shape: ` <from>' to ' <to>'`
        let mut parts = conv.split('\'');
        let from_desc = parts.next().unwrap_or("").trim();
        let to_desc = parts.nth(1).unwrap_or("").trim();
        let (Some(from), Some(to)) = (parse_shaderc_type(from_desc), parse_shaderc_type(to_desc))
        else {
            continue;
        };

        // Pick the wrapper HLSL's implicit conversion corresponds to.
        let wrap: (String, String) = match (from, to) {
            // vecN -> scalar: HLSL truncation warning, takes the leading
            // component(s).
            (("vec", _), ("float", _)) => ("(".into(), ").x".into()),
            (("vec", m), ("vec", n)) if m > n => {
                ("(".into(), format!(").{}", &"xyzw"[..n as usize]))
            }
            // scalar -> vecN: HLSL splats.
            (("float", _) | ("int", _) | (("bool", _)), ("vec", n)) => {
                (format!("vec{n}("), ")".into())
            }
            // bool -> number: HLSL's true/false are 1/0 (simple_audio_bars
            // assigns a comparison straight into an int bar index).
            (("bool", _), ("float", _)) => ("float(".into(), ")".into()),
            (("bool", _), ("int", _)) => ("int(".into(), ")".into()),
            (("bool", _), ("uint", _)) => ("uint(".into(), ")".into()),
            // float -> int/uint: HLSL truncates toward zero, same as GLSL's
            // explicit constructor.
            (("float", _), ("int", _)) => ("int(".into(), ")".into()),
            (("float", _), ("uint", _)) => ("uint(".into(), ")".into()),
            _ => continue,
        };

        let Some(line) = lines.get_mut(line_no.saturating_sub(1)) else {
            continue;
        };
        if let Some(patched) = wrap_assignment_rhs(line, &wrap.0, &wrap.1) {
            *line = patched;
            repaired = true;
        }
    }

    repaired.then(|| lines.join("\n"))
}

/// Declared widths of everything named in a generated shader: input varyings,
/// UBO members and plain local declarations. Feeds the error-driven repairs
/// below, which run on the final GLSL text and so have no access to the
/// preprocessor's own tracking.
fn declared_widths(glsl: &str) -> HashMap<String, u8> {
    let mut map = HashMap::new();
    for line in glsl.lines() {
        let t = line.trim_start();
        // "layout(location=0) in vec4 v_TexCoord;" / "    vec2 u_rOffset;" /
        // "	vec4 timer = texSample2D(...);"
        let decl = t.rsplit(')').next().unwrap_or(t).trim();
        let mut tok = decl.split_whitespace();
        let (Some(first), Some(second)) = (tok.next(), tok.next()) else {
            continue;
        };
        let (ty, name) = if first == "in" || first == "out" || first == "uniform" {
            match tok.next() {
                Some(n) => (second, n),
                None => continue,
            }
        } else {
            (first, second)
        };
        let Some(w) = vec_width(ty).or(matches!(ty, "float" | "int").then_some(1)) else {
            continue;
        };
        let name = name
            .trim_end_matches([';', ','])
            .split('=')
            .next()
            .unwrap_or("")
            .trim();
        if !name.is_empty() && name.chars().all(|c| is_word_char(c as u8)) {
            map.insert(name.to_string(), w);
        }
    }
    map
}

/// Line numbers shaderc reported for messages containing `needle`.
fn error_lines(err_text: &str, needle: &str) -> Vec<usize> {
    err_text
        .lines()
        .filter(|m| m.contains(needle))
        .filter_map(|m| {
            let rest = m
                .split_once("shader.frag:")
                .or_else(|| m.split_once("shader.vert:"))?
                .1;
            rest.split(':').next()?.trim().parse::<usize>().ok()
        })
        .collect()
}

/// `v_TexCoord += ...` — chromatic_aberration.frag writes to its own varying,
/// which WE's compiler allows and GLSL does not ("l-value required ... can't
/// modify shader input"). Rename the input and give `main()` a mutable copy
/// under the original name, so the body compiles unchanged.
fn repair_input_assignment(glsl: &str, err_text: &str) -> Option<String> {
    let mut out = glsl.to_string();
    let mut repaired = false;
    for msg in err_text.lines() {
        if !msg.contains("can't modify shader input") {
            continue;
        }
        let Some(name) = msg
            .split('"')
            .nth(1)
            .filter(|n| !n.is_empty() && n.chars().all(|c| is_word_char(c as u8)))
        else {
            continue;
        };
        let alias = format!("_wp_in_{name}");
        if out.contains(&alias) {
            continue;
        }
        // Find the input declaration to learn its type, and rename it there.
        let Some(decl) = out.lines().find(|l| {
            let t = l.trim_start();
            t.starts_with("layout(") && t.contains(" in ") && t.ends_with(&format!(" {name};"))
        }) else {
            continue;
        };
        let ty = decl.split_whitespace().nth(2)?.to_string();
        let renamed = decl.replace(&format!(" {name};"), &format!(" {alias};"));
        out = out.replace(decl, &renamed);
        // Shadow it with a local copy at the top of main().
        let main_brace = out
            .find("void main")
            .and_then(|p| out[p..].find('{').map(|i| p + i + 1))?;
        out.insert_str(main_brace, &format!("\n\t{ty} {name} = {alias};"));
        repaired = true;
    }
    repaired.then_some(out)
}

/// `depth *= (depth < limit) * 6.0` — bokeh_blur.frag does arithmetic on a
/// comparison. HLSL reads true as 1.0; GLSL refuses. Wrap the parenthesised
/// comparison in `float(...)`.
fn repair_bool_operands(glsl: &str, err_text: &str) -> Option<String> {
    let mut lines: Vec<String> = glsl.lines().map(str::to_string).collect();
    let mut repaired = false;
    for line_no in error_lines(err_text, "operand of type ' temp bool'") {
        let Some(line) = lines.get_mut(line_no.saturating_sub(1)) else {
            continue;
        };
        let mut patched = line.clone();
        let mut from = 0;
        while let Some(rel) = patched[from..].find('(') {
            let open = from + rel;
            let Some((group, after)) = extract_balanced(&patched, open, b'(', b')') else {
                break;
            };
            let inner = &group[1..group.len() - 1];
            let is_comparison = ["<", ">", "<=", ">=", "==", "!="]
                .iter()
                .any(|op| inner.contains(op))
                && !inner.contains("float(");
            if is_comparison && !patched[..open].ends_with("float") {
                let replacement = format!("float{group}");
                patched = format!("{}{replacement}{}", &patched[..open], &patched[after..]);
                from = open + replacement.len();
                repaired = true;
                continue;
            }
            from = open + 1;
        }
        *line = patched;
    }
    repaired.then(|| lines.join("\n"))
}

/// `rotateVec2(v_TexCoord, ...)` where the varying is a vec4 — usually because
/// `harmonize_varying_widths` widened the fragment's declaration to match the
/// vertex stage. Coerce each argument to the callee's declared parameter type.
fn repair_call_arg_widths(glsl: &str, err_text: &str) -> Option<String> {
    let widths = declared_widths(glsl);
    let mut lines: Vec<String> = glsl.lines().map(str::to_string).collect();
    let mut repaired = false;
    for msg in err_text.lines() {
        if !msg.contains("no matching overloaded function found") {
            continue;
        }
        let Some(func) = msg.split('\'').nth(1) else {
            continue;
        };
        let Some(line_no) = error_lines(msg, "no matching overloaded function")
            .first()
            .copied()
        else {
            continue;
        };
        // The callee's parameter widths, from its definition in this source.
        let Some(params) = glsl.lines().find_map(|l| {
            let t = l.trim_start();
            let rest = t.split_once(&format!(" {func}("))?.1;
            let end = rest.find(')')?;
            Some(
                rest[..end]
                    .split(',')
                    .filter_map(|p| {
                        let ty = p.split_whitespace().next()?;
                        vec_width(ty).or(matches!(ty, "float" | "int").then_some(1))
                    })
                    .collect::<Vec<u8>>(),
            )
        }) else {
            continue;
        };
        let Some(line) = lines.get_mut(line_no.saturating_sub(1)) else {
            continue;
        };
        let Some(call) = line.find(&format!("{func}(")) else {
            continue;
        };
        let open = call + func.len();
        let Some((group, after)) = extract_balanced(line, open, b'(', b')') else {
            continue;
        };
        let args: Vec<String> = split_top_level_args(&group[1..group.len() - 1])
            .iter()
            .map(|(a, _)| a.trim().to_string())
            .collect();
        if args.len() != params.len() {
            continue;
        }
        let coerced: Vec<String> = args
            .iter()
            .zip(&params)
            .map(|(arg, &want)| match widths.get(arg.as_str()) {
                Some(&have) if have > want && want >= 2 => {
                    format!("{arg}{}", swizzle_prefix(want))
                }
                _ => arg.clone(),
            })
            .collect();
        if coerced == args {
            continue;
        }
        *line = format!(
            "{}{func}({}){}",
            &line[..call],
            coerced.join(", "),
            &line[after..]
        );
        repaired = true;
    }
    repaired.then(|| lines.join("\n"))
}

/// `u_rOffset * timer` where one is a vec2 and the other a vec4 — HLSL
/// truncates the wider side, GLSL rejects the pair. Swizzle the wider
/// identifiers on the offending line down to the narrower width.
fn repair_operand_widths(glsl: &str, err_text: &str) -> Option<String> {
    let widths = declared_widths(glsl);
    let mut lines: Vec<String> = glsl.lines().map(str::to_string).collect();
    let mut repaired = false;
    for msg in err_text.lines() {
        if !msg.contains("wrong operand types") {
            continue;
        }
        let mut sides = msg.split("operand of type").skip(1).filter_map(|s| {
            let desc = s.split('\'').nth(1)?;
            parse_shaderc_type(desc.trim()).map(|(_, n)| n)
        });
        let (Some(a), Some(b)) = (sides.next(), sides.next()) else {
            continue;
        };
        let (wide, narrow) = (a.max(b), a.min(b));
        if wide == narrow || narrow < 2 {
            continue;
        }
        let Some(line_no) = error_lines(msg, "wrong operand types").first().copied() else {
            continue;
        };
        let Some(line) = lines.get_mut(line_no.saturating_sub(1)) else {
            continue;
        };
        let patched = truncate_wide_identifiers_of(line, wide, narrow, &widths);
        if patched != *line {
            *line = patched;
            repaired = true;
        }
    }
    repaired.then(|| lines.join("\n"))
}

/// Swizzle every bare identifier of exactly `wide` components down to
/// `narrow`. Unlike `truncate_wide_identifiers` this runs on a whole
/// statement (call arguments included) and only when shaderc already
/// rejected that line, so it cannot narrow a call that was fine.
fn truncate_wide_identifiers_of(
    line: &str,
    wide: u8,
    narrow: u8,
    widths: &HashMap<String, u8>,
) -> String {
    let mut out = String::with_capacity(line.len());
    let bytes = line.as_bytes();
    let mut prev_word: Option<String> = None;
    let mut i = 0;
    while i < bytes.len() {
        if !is_word_char(bytes[i])
            || (i > 0 && (is_word_char(bytes[i - 1]) || bytes[i - 1] == b'.'))
        {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && is_word_char(bytes[i]) {
            i += 1;
        }
        let ident = &line[start..i];
        out.push_str(ident);
        let next = bytes.get(i).copied().unwrap_or(b' ');
        // `vec4 rValue = ...` declares rValue — swizzling the declared name
        // is a syntax error, and the declaration's own width is not the bug.
        let is_declaration = matches!(
            prev_word.as_deref(),
            Some("vec2" | "vec3" | "vec4" | "float" | "int" | "uint" | "bool")
        );
        if !is_declaration
            && next != b'.'
            && next != b'('
            && next != b'['
            && widths.get(ident) == Some(&wide)
        {
            out.push_str(swizzle_prefix(narrow));
        }
        prev_word = Some(ident.to_string());
    }
    out
}

/// `if (u_userInvertDepthMap)` where the uniform is a float — HLSL treats
/// nonzero as true, strict GLSL wants a bool ("boolean expression expected").
/// Wrap the condition in `bool(...)`. Returns `None` when nothing was repaired.
fn repair_bool_conditions(glsl: &str, err_text: &str) -> Option<String> {
    let mut lines: Vec<String> = glsl.lines().map(str::to_string).collect();
    let mut repaired = false;
    for msg in err_text.lines() {
        if !msg.contains("boolean expression expected") {
            continue;
        }
        let Some(rest) = msg
            .split_once("shader.frag:")
            .or_else(|| msg.split_once("shader.vert:"))
            .map(|(_, r)| r)
        else {
            continue;
        };
        let Ok(line_no) = rest.split(':').next().unwrap_or("").trim().parse::<usize>() else {
            continue;
        };
        let Some(line) = lines.get_mut(line_no.saturating_sub(1)) else {
            continue;
        };
        for kw in ["if", "while"] {
            let Some(kw_pos) = line
                .find(&format!("{kw}("))
                .or_else(|| line.find(&format!("{kw} (")))
            else {
                continue;
            };
            let Some(open) = line[kw_pos..].find('(').map(|i| kw_pos + i) else {
                continue;
            };
            // extract_balanced yields the parens too, and `after` is the index
            // just past the closing one.
            let Some((parens, after)) = extract_balanced(line, open, b'(', b')') else {
                continue;
            };
            let cond = parens[1..parens.len() - 1].trim();
            if cond.is_empty() || cond.contains("bool(") {
                continue;
            }
            *line = format!("{}(bool({cond})){}", &line[..open], &line[after..]);
            repaired = true;
            break;
        }
    }
    repaired.then(|| lines.join("\n"))
}

/// Wrap the right-hand side of the first plain assignment on `line` in
/// `prefix`/`suffix`. The RHS runs from the `=` to the first `;` or `,` at
/// paren depth zero relative to the RHS start (so a `for (int i = x; ...)`
/// init terminates at its own `;`). Compound (`+=`) and comparison
/// (`==`/`<=`/...) operators are not assignments and are skipped.
fn wrap_assignment_rhs(line: &str, prefix: &str, suffix: &str) -> Option<String> {
    let bytes = line.as_bytes();
    let mut eq = None;
    for (i, &b) in bytes.iter().enumerate() {
        if b != b'=' {
            continue;
        }
        let prev = if i > 0 { bytes[i - 1] } else { b' ' };
        let next = bytes.get(i + 1).copied().unwrap_or(b' ');
        // Skip comparisons (`==`, `!=`, `<=`, `>=`). Compound assignments
        // (`+= -= *= /=`) ARE valid targets: `int x *= floatExpr` needs the
        // RHS wrapped just like a plain `int x = floatExpr` does.
        if matches!(prev, b'=' | b'!' | b'<' | b'>') || next == b'=' {
            continue;
        }
        eq = Some(i);
        break;
    }
    let eq = eq?;

    let rhs_start = eq + 1;
    let mut depth = 0i32;
    let mut rhs_end = line.len();
    for (i, ch) in line[rhs_start..].char_indices() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ';' | ',' if depth <= 0 => {
                rhs_end = rhs_start + i;
                break;
            }
            _ => {}
        }
    }
    let rhs = line[rhs_start..rhs_end].trim();
    if rhs.is_empty() {
        return None;
    }
    if rhs.starts_with(prefix.trim_end_matches('(')) && rhs.ends_with(suffix) {
        // Already wrapped on an earlier repair round — move on to the next
        // assignment on this line. simple_audio_bars.frag puts two of them on
        // one line ("barLeft *= isLeftChannel; barRight *= isRightChannel;")
        // and both need the same cast.
        let rest = wrap_assignment_rhs(&line[rhs_end..], prefix, suffix)?;
        return Some(format!("{}{rest}", &line[..rhs_end]));
    }
    Some(format!(
        "{} {prefix}{rhs}{suffix}{}",
        &line[..rhs_start],
        &line[rhs_end..]
    ))
}

fn glsl_to_spirv(glsl: &str, stage: naga::ShaderStage) -> Result<Vec<u8>> {
    let (kind, filename) = match stage {
        naga::ShaderStage::Fragment => (shaderc::ShaderKind::Fragment, "shader.frag"),
        naga::ShaderStage::Vertex => (shaderc::ShaderKind::Vertex, "shader.vert"),
        _ => (shaderc::ShaderKind::Compute, "shader.comp"),
    };

    let compiler = shaderc::Compiler::new().context("failed to create shaderc compiler")?;
    let mut options = shaderc::CompileOptions::new().context("failed to create shaderc options")?;
    // Target Vulkan SPIR-V — required for naga's spv-in reader.
    // Our preprocessor already emits Vulkan GLSL 4.50 with explicit set/binding/location qualifiers.
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_1 as u32,
    );
    options.set_source_language(shaderc::SourceLanguage::GLSL);
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    let artifact = compiler
        .compile_into_spirv(glsl, kind, filename, "main", Some(&options))
        .context("shaderc GLSL→SPIR-V compilation failed")?;

    Ok(artifact.as_binary_u8().to_vec())
}

// ── SPIR-V → WGSL via naga ───────────────────────────────────────────────────

fn spirv_to_wgsl(spv: &[u8]) -> Result<String> {
    let module = naga::front::spv::parse_u8_slice(
        spv,
        &naga::front::spv::Options {
            adjust_coordinate_space: true,
            ..Default::default()
        },
    )
    .context("naga SPIR-V parse failed")?;

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .context("naga validation failed")?;

    naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
        .context("naga WGSL output failed")
}

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

fn vec_width(ty: &str) -> Option<u8> {
    match ty {
        "vec2" => Some(2),
        "vec3" => Some(3),
        "vec4" => Some(4),
        _ => None,
    }
}

fn swizzle_prefix(n: u8) -> &'static str {
    match n {
        1 => ".x",
        2 => ".xy",
        3 => ".xyz",
        4 => ".xyzw",
        _ => "",
    }
}

const TEX_SAMPLE_CALLS: &[&str] = &[
    "texSample2DLod(",
    "texSample2D(",
    "texture2DLod(",
    "textureSample2D(",
    "texture2D(",
    "textureCube(",
    "texture(",
];

/// Legacy NVIDIA GLSL compilers silently truncated the wider operand when
/// vector widths mismatched in an assignment or binary op (e.g. assigning a
/// vec4 texture sample to a `vec3` without a `.rgb` swizzle, or subtracting a
/// `vec2` from a `vec3` varying). Workshop shaders authored and only ever
/// tested on NVIDIA rely on this; naga/shaderc's stricter frontend rejects it
/// outright. `var_width` is updated in place as declarations are seen so the
/// caller can process a shader body line by line.
/// Rename functions the shader defines under a name that is already a GLSL
/// builtin or one of our HLSL-compat macros. `crt_scan_line.frag` ships its
/// own `float mod(float, float)`, which glslang rejects (the builtin's
/// parameters are `highp`, the shader's are not); `lerp`/`saturate` would
/// expand through the macro into an even stranger error.
fn rename_shadowed_builtins(src: &str) -> String {
    const SHADOWABLE: &[&str] = &["mod", "lerp", "saturate", "frac", "mul", "atan2", "log10"];
    let mut out = src.to_string();
    for name in SHADOWABLE {
        let defined = src.lines().any(|l| {
            let t = l.trim_start();
            let Some(rest) = t.split_once(&format!(" {name}(")) else {
                return false;
            };
            // A definition starts the line with just a return type — `return
            // frac(x);` also has one leading word, so match real types only.
            matches!(
                rest.0,
                "float"
                    | "vec2"
                    | "vec3"
                    | "vec4"
                    | "int"
                    | "uint"
                    | "bool"
                    | "void"
                    | "mat2"
                    | "mat3"
                    | "mat4"
            )
        });
        if defined {
            out = rename_reserved_word(&out, name, &format!("_wp_{name}"));
        }
    }
    out
}

/// Swizzle every bare identifier in `expr` that is known to be wider than
/// `w` down to `w` components (`v_TexCoord` → `v_TexCoord.xy`). Identifiers
/// that already carry a swizzle, are being called, or are struct/array
/// accesses are left alone.
fn truncate_wide_identifiers(expr: &str, w: u8, var_width: &HashMap<String, u8>) -> String {
    let mut out = String::with_capacity(expr.len());
    let bytes = expr.as_bytes();
    // A call's arguments are typed by its signature, not by the declaration
    // being assigned to: narrowing `get_noise(v_TexCoord)` to `.xyz` breaks the
    // call it was fine in. Only bare arithmetic operands get truncated.
    let mut call_depth = 0usize;
    let mut last_word_end: Option<usize> = None;
    let mut i = 0;
    while i < bytes.len() {
        if !is_word_char(bytes[i])
            || (i > 0 && (is_word_char(bytes[i - 1]) || bytes[i - 1] == b'.'))
        {
            match bytes[i] {
                b'(' => {
                    let is_call = last_word_end == Some(i);
                    if is_call || call_depth > 0 {
                        call_depth += 1;
                    }
                }
                b')' => call_depth = call_depth.saturating_sub(1),
                _ => {}
            }
            if bytes[i] != b' ' {
                last_word_end = None;
            }
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && is_word_char(bytes[i]) {
            i += 1;
        }
        let ident = &expr[start..i];
        out.push_str(ident);
        last_word_end = Some(i);
        let next = bytes.get(i).copied().unwrap_or(b' ');
        if call_depth == 0 && next != b'.' && next != b'(' && next != b'[' {
            if let Some(&iw) = var_width.get(ident) {
                if iw > w {
                    out.push_str(swizzle_prefix(w));
                }
            }
        }
    }
    out
}

/// `timer.xyz` where `timer` is a `float` — lens_flare_sun.frag does this and
/// WE's compiler treats it as a splat. Strict GLSL calls it a swizzle out of
/// range, so rewrite it to `vec3(timer)`.
fn splat_scalar_swizzles(line: &str, var_width: &HashMap<String, u8>) -> String {
    let mut out = String::with_capacity(line.len());
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if !is_word_char(bytes[i])
            || (i > 0 && (is_word_char(bytes[i - 1]) || bytes[i - 1] == b'.'))
        {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && is_word_char(bytes[i]) {
            i += 1;
        }
        let ident = &line[start..i];
        // A swizzle of 2-4 components on something declared `float`.
        let swz_start = i + 1;
        let swz_end = if bytes.get(i) == Some(&b'.') {
            let mut j = swz_start;
            while j < bytes.len() && b"xyzwrgba".contains(&bytes[j]) {
                j += 1;
            }
            j
        } else {
            swz_start
        };
        let n = swz_end.saturating_sub(swz_start);
        if (2..=4).contains(&n)
            && var_width.get(ident) == Some(&1)
            && !bytes.get(swz_end).is_some_and(|b| is_word_char(*b))
        {
            out.push_str(&format!("vec{n}({ident})"));
            i = swz_end;
            continue;
        }
        out.push_str(ident);
    }
    out
}

fn coerce_vector_widths(line: &str, var_width: &mut HashMap<String, u8>) -> String {
    let indent_len = line.len() - line.trim_start().len();
    let indent = &line[..indent_len];
    let trimmed = line.trim_start();
    let trimmed = trimmed.strip_suffix('\r').unwrap_or(trimmed);

    // `vecN name = <expr>;` — track the declared width, and if <expr> is an
    // unswizzled texture sample (always vec4) narrower than N, truncate it.
    // Scalars are tracked too — only so a `.xyz` on one can be splatted below.
    if let Some(rest) = trimmed.strip_prefix("float ") {
        if let Some(eq) = rest.find('=') {
            let name = rest[..eq].trim();
            if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                var_width.insert(name.to_string(), 1);
            }
        }
    }
    let mut declared: Option<(u8, &str)> = None;
    for w in [2u8, 3u8] {
        let prefix = format!("vec{w} ");
        if let Some(rest) = trimmed.strip_prefix(prefix.as_str()) {
            if let Some(eq_pos) = rest.find('=') {
                let name = rest[..eq_pos].trim();
                if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    declared = Some((w, name));
                }
            }
        }
    }
    if let Some((w, name)) = declared {
        var_width.insert(name.to_string(), w);
        if let Some(body) = trimmed.strip_suffix(';') {
            if let Some(eq_pos) = body.find('=') {
                let rhs = body[eq_pos + 1..].trim();
                if TEX_SAMPLE_CALLS.iter().any(|c| rhs.starts_with(c))
                    && rhs.ends_with(')')
                    && w < 4
                {
                    let lhs = &body[..=eq_pos];
                    return format!("{indent}{lhs} {rhs}{};", swizzle_prefix(w));
                }
                // Wider operands inside a narrower declaration's expression:
                // clipping_mask.frag computes `vec2 uv = (v_TexCoord * 2.0 -
                // 1.0 - texScaleCenter) / ...` with a vec4 varying. HLSL
                // truncates the wide side with a warning; GLSL rejects it.
                let narrowed = truncate_wide_identifiers(rhs, w, var_width);
                if narrowed != rhs {
                    let lhs = &body[..=eq_pos];
                    return format!("{indent}{lhs} {narrowed};");
                }
            }
        }
    }

    // Plain reassignment `name = <texture sample>;` to an already-known narrower var.
    if declared.is_none() {
        if let Some(body) = trimmed.strip_suffix(';') {
            if let Some(eq_pos) = body.find('=') {
                let (lhs, rhs) = (body[..eq_pos].trim(), body[eq_pos + 1..].trim());
                if lhs.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    if let Some(&w) = var_width.get(lhs) {
                        if w < 4
                            && TEX_SAMPLE_CALLS.iter().any(|c| rhs.starts_with(c))
                            && rhs.ends_with(')')
                        {
                            return format!("{indent}{lhs} = {rhs}{};", swizzle_prefix(w));
                        }
                    }
                }
            }
        }
    }

    // Binary arithmetic against a `CASTn(...)` broadcast: `wide_var - CASTn(x)` where
    // wide_var's known width exceeds n. Truncate wide_var's trailing components.
    let mut out = splat_scalar_swizzles(trimmed, var_width);
    for cast_w in [2u8, 3u8] {
        let marker = format!("CAST{cast_w}(");
        let mut search_from = 0;
        while let Some(rel) = out[search_from..].find(marker.as_str()) {
            let cast_pos = search_from + rel;
            // Walk left over whitespace, then the binary operator, then whitespace,
            // to find the end of the preceding operand.
            let mut i = cast_pos;
            let bytes = out.as_bytes();
            while i > 0 && bytes[i - 1] == b' ' {
                i -= 1;
            }
            if i == 0 || !matches!(bytes[i - 1], b'-' | b'+' | b'*' | b'/') {
                search_from = cast_pos + marker.len();
                continue;
            }
            i -= 1; // consume operator
            while i > 0 && bytes[i - 1] == b' ' {
                i -= 1;
            }
            let operand_end = i;
            let mut start = i;
            while start > 0
                && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_')
            {
                start -= 1;
            }
            let operand = &out[start..operand_end];
            if let Some(&w) = var_width.get(operand) {
                if w > cast_w && start > 0 && bytes[start - 1] != b'.' {
                    let swz = swizzle_prefix(cast_w);
                    out = format!("{}{}{}", &out[..operand_end], swz, &out[operand_end..]);
                    search_from = operand_end + swz.len() + marker.len();
                    continue;
                }
            }
            search_from = cast_pos + marker.len();
        }
    }
    format!("{indent}{out}")
}

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

// Replace a reserved keyword with `replacement` only at word boundaries (i.e., not when
// adjacent to alphanumeric chars or underscore).  Used to rename `sample` → `_wp_s`.
fn rename_reserved_word(src: &str, keyword: &str, replacement: &str) -> String {
    let klen = keyword.len();
    let sbytes = src.as_bytes();
    let kbytes = keyword.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    while i < sbytes.len() {
        if sbytes[i..].starts_with(kbytes) {
            let before_ok = i == 0 || !is_word_char(sbytes[i - 1]);
            let after_ok = i + klen >= sbytes.len() || !is_word_char(sbytes[i + klen]);
            if before_ok && after_ok {
                out.push_str(replacement);
                i += klen;
                continue;
            }
        }
        out.push(sbytes[i] as char);
        i += 1;
    }
    out
}

#[inline]
fn is_word_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
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
        let wallpaper_dir =
            std::path::Path::new("/home/tupretiw/wp-engine/analysis/orig_1137423685");
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
            let model = crate::engine::model::ShaderModel::from_resolved_glsl(
                label.to_string(),
                frag_glsl,
                combos,
                crate::engine::model::WEBlending::Normal,
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
        let frag = "varying vec2 v_TexCoord;\nvoid main() { gl_FragColor = vec4(v_TexCoord, 0.0, 1.0); }\n";
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
            crate::engine::model::shader_model::WEBlending::Normal,
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
            crate::engine::model::shader_model::WEBlending::Normal,
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
}
