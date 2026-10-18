//! Source rewrites applied before compiling: real Workshop shaders rely on
//! lenient GLSL compilers, so loops with constant bounds are unrolled, array
//! varyings expanded, int literals and vector widths coerced, and names that
//! clash with built-ins renamed.

use super::*;

/// Collect non-sampler `uniform` declarations (type, name) in source order.
/// The macro name of a `#define NAME ...` line, if this line is one.
pub(super) fn redefined_macro(trimmed_line: &str) -> Option<&str> {
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

pub(super) fn collect_scalar_uniforms(src: &str) -> Vec<(String, String)> {
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
pub(super) fn all_indices_are_literal(src: &str, name: &str) -> bool {
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
pub(super) fn extract_balanced(
    s: &str,
    open_pos: usize,
    open: u8,
    close: u8,
) -> Option<(&str, usize)> {
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
pub(super) fn parse_simple_for_header(header_inner: &str) -> Option<(String, usize)> {
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
pub(super) fn split_top_level_args(args: &str) -> Vec<(&str, usize)> {
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
pub(super) fn is_bare_number(s: &str) -> bool {
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

pub(super) fn ensure_float_literal(trimmed: &str) -> String {
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

/// Port of the reference's `applyLinkedVaryingCompatibility`
/// (ShaderUnit.cpp:379-415): for every `varying vec4 NAME;` the fragment
/// declares where the vertex declares `varying vec2 NAME;`, upgrade the
/// vertex declaration to vec4 and rewrite the vertex's whole-variable
/// assignments (`NAME = expr;` at statement start — swizzled stores like
/// `NAME.xy = ...` are already type-correct and left alone) to
/// `NAME = vec4(expr, 0.0, 1.0);`. WE's own compiler links such mismatched
/// interfaces silently; strict Vulkan pipelines reject them.
pub(super) fn harmonize_varying_widths(vert: &str, frag: &str) -> String {
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

pub(super) fn vec_width(ty: &str) -> Option<u8> {
    match ty {
        "vec2" => Some(2),
        "vec3" => Some(3),
        "vec4" => Some(4),
        _ => None,
    }
}

pub(super) fn swizzle_prefix(n: u8) -> &'static str {
    match n {
        1 => ".x",
        2 => ".xy",
        3 => ".xyz",
        4 => ".xyzw",
        _ => "",
    }
}

pub(super) const TEX_SAMPLE_CALLS: &[&str] = &[
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
pub(super) fn rename_shadowed_builtins(src: &str) -> String {
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
pub(super) fn truncate_wide_identifiers(
    expr: &str,
    w: u8,
    var_width: &HashMap<String, u8>,
) -> String {
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
pub(super) fn splat_scalar_swizzles(line: &str, var_width: &HashMap<String, u8>) -> String {
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

pub(super) fn coerce_vector_widths(line: &str, var_width: &mut HashMap<String, u8>) -> String {
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

// Replace a reserved keyword with `replacement` only at word boundaries (i.e., not when
// adjacent to alphanumeric chars or underscore).  Used to rename `sample` → `_wp_s`.
pub(super) fn rename_reserved_word(src: &str, keyword: &str, replacement: &str) -> String {
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
pub(super) fn is_word_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
