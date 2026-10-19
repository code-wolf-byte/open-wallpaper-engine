#[derive(Debug, Clone)]
pub struct UniformMeta {
    pub uniform_name: String,
    pub uniform_type: String,
    pub material_key: Option<String>,
    pub label: Option<String>,
    pub default_value: Option<serde_json::Value>,
    pub range: Option<(f64, f64)>,
    pub hidden: bool,
    pub combo: Option<String>,
    pub mode: Option<String>,
}

pub fn parse_uniform_metadata(glsl_source: &str) -> Vec<UniformMeta> {
    let mut uniforms = Vec::new();

    for line in glsl_source.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("uniform ") {
            continue;
        }

        let json_start = match trimmed.find("// {") {
            Some(pos) => pos + 3,
            None => continue,
        };
        let json_str = &trimmed[json_start..];

        let parts: Vec<&str> = trimmed[..json_start - 3]
            .trim()
            .split_whitespace()
            .collect();
        if parts.len() < 3 {
            continue;
        }
        let uniform_type = parts[1].to_string();
        let uniform_name = parts[2].trim_end_matches(';').to_string();

        let json_val: serde_json::Value = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let range = json_val
            .get("range")
            .and_then(|v| v.as_array())
            .and_then(|a| {
                let min = a.first()?.as_f64()?;
                let max = a.get(1)?.as_f64()?;
                Some((min, max))
            });

        uniforms.push(UniformMeta {
            uniform_name,
            uniform_type,
            material_key: json_val
                .get("material")
                .and_then(|v| v.as_str())
                .map(String::from),
            label: json_val
                .get("label")
                .and_then(|v| v.as_str())
                .map(String::from),
            default_value: json_val.get("default").cloned(),
            range,
            hidden: json_val
                .get("hidden")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            combo: json_val
                .get("combo")
                .and_then(|v| v.as_str())
                .map(String::from),
            mode: json_val
                .get("mode")
                .and_then(|v| v.as_str())
                .map(String::from),
        });
    }

    uniforms
}
