//! Compiling to SPIR-V and translating to WGSL, with repairs driven by
//! shaderc's error messages: when a shader fails to compile, the failing
//! lines are patched (float modulo, implicit conversions, operand and call
//! argument widths, bool conditions) and it is compiled again.

use super::*;

/// `glsl_to_spirv` with a bounded repair loop for HLSL-style implicit
/// conversions that strict GLSL rejects but Wallpaper Engine's own (D3D)
/// compiler accepts silently — e.g. `float p = vec2Expr;` (truncates to .x),
/// `int i = -floatVar;`, or a scalar assigned to a vec3. shaderc's error
/// messages carry the line number and both types, which is exactly enough
/// to wrap the right-hand side in the coercion HLSL would have applied,
/// then recompile. Bails out (returning the latest error) once a round
/// fixes nothing or after a few rounds, so pathological shaders still fail
/// cleanly instead of looping.
pub(super) fn glsl_to_spirv_with_repairs(glsl: &str, stage: naga::ShaderStage) -> Result<Vec<u8>> {
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
pub(super) fn parse_shaderc_type(desc: &str) -> Option<(&'static str, u8)> {
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
pub(super) fn rewrite_modulo_on_line(line: &str) -> Option<String> {
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
pub(super) fn repair_float_modulo(glsl: &str, err_text: &str) -> Option<String> {
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
pub(super) fn repair_assignment_conversions(glsl: &str, err_text: &str) -> Option<String> {
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
pub(super) fn declared_widths(glsl: &str) -> HashMap<String, u8> {
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
pub(super) fn error_lines(err_text: &str, needle: &str) -> Vec<usize> {
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
pub(super) fn repair_input_assignment(glsl: &str, err_text: &str) -> Option<String> {
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
pub(super) fn repair_bool_operands(glsl: &str, err_text: &str) -> Option<String> {
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
pub(super) fn repair_call_arg_widths(glsl: &str, err_text: &str) -> Option<String> {
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
pub(super) fn repair_operand_widths(glsl: &str, err_text: &str) -> Option<String> {
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
pub(super) fn truncate_wide_identifiers_of(
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
pub(super) fn repair_bool_conditions(glsl: &str, err_text: &str) -> Option<String> {
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
pub(super) fn wrap_assignment_rhs(line: &str, prefix: &str, suffix: &str) -> Option<String> {
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

pub(super) fn glsl_to_spirv(glsl: &str, stage: naga::ShaderStage) -> Result<Vec<u8>> {
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

pub(super) fn spirv_to_wgsl(spv: &[u8]) -> Result<String> {
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
