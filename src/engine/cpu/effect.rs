use image::RgbaImage;

pub enum SceneEffect {
    Tint { r: f32, g: f32, b: f32, alpha: f32 },
    Opacity { alpha: f32 },
    Shake { strength: f32, time: f32 },
    Pulse { strength: f32, time: f32 },
    Scroll { speed: f32, time: f32 },
    Spin { speed: f32, time: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandwrittenEffectFallback {
    Tint,
    Opacity,
    Pulse,
    Shake,
    Scroll,
    Spin,
}

impl HandwrittenEffectFallback {
    pub fn from_label(label: &str) -> Option<Self> {
        let label = label.to_ascii_lowercase();
        if label.contains("tint") {
            Some(Self::Tint)
        } else if label.contains("opacity") || label.contains("alpha") {
            Some(Self::Opacity)
        } else if label.contains("pulse") {
            Some(Self::Pulse)
        } else if label.contains("shake") {
            Some(Self::Shake)
        } else if label.contains("scroll") {
            Some(Self::Scroll)
        } else if label.contains("spin") {
            Some(Self::Spin)
        } else {
            None
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Tint => "tint",
            Self::Opacity => "opacity",
            Self::Pulse => "pulse",
            Self::Shake => "shake",
            Self::Scroll => "scroll",
            Self::Spin => "spin",
        }
    }

    pub fn fragment_entry_point(self) -> &'static str {
        match self {
            Self::Tint => "fs_effect_tint",
            Self::Opacity => "fs_effect_opacity",
            Self::Pulse => "fs_effect_pulse",
            Self::Shake => "fs_effect_shake",
            Self::Scroll => "fs_effect_scroll",
            Self::Spin => "fs_effect_spin",
        }
    }
}

impl SceneEffect {
    pub fn from_effect_name(name: &str, values: &serde_json::Value, time: f32) -> Option<Self> {
        let name_lower = name.to_lowercase();
        if name_lower.contains("tint") {
            let alpha = extract_f32(values, "alpha").unwrap_or(1.0);
            let color = extract_string(values, "tintcolor").unwrap_or_default();
            let rgb = parse_color(&color);
            Some(SceneEffect::Tint {
                r: rgb[0],
                g: rgb[1],
                b: rgb[2],
                alpha,
            })
        } else if name_lower.contains("opacity") {
            let alpha = extract_f32(values, "alpha").unwrap_or(1.0);
            Some(SceneEffect::Opacity { alpha })
        } else if name_lower.contains("shake") {
            let strength = extract_f32(values, "ui_editor_properties_strength")
                .or_else(|| extract_f32(values, "strength"))
                .unwrap_or(0.05);
            Some(SceneEffect::Shake { strength, time })
        } else if name_lower.contains("pulse") {
            let strength = extract_f32(values, "ui_editor_properties_strength")
                .or_else(|| extract_f32(values, "strength"))
                .unwrap_or(0.1);
            Some(SceneEffect::Pulse { strength, time })
        } else if name_lower.contains("scroll") {
            let speed = extract_f32(values, "speed")
                .or_else(|| extract_f32(values, "strength"))
                .unwrap_or(0.02);
            Some(SceneEffect::Scroll { speed, time })
        } else if name_lower.contains("spin") {
            let speed = extract_f32(values, "speed")
                .or_else(|| extract_f32(values, "strength"))
                .unwrap_or(0.5);
            Some(SceneEffect::Spin { speed, time })
        } else {
            None
        }
    }

    pub fn apply(&self, img: &mut RgbaImage) {
        match self {
            SceneEffect::Tint { r, g, b, alpha } => apply_tint(img, *r, *g, *b, *alpha),
            SceneEffect::Opacity { alpha } => apply_opacity(img, *alpha),
            SceneEffect::Shake { strength, time } => apply_shake(img, *strength, *time),
            SceneEffect::Pulse { strength, time } => apply_pulse(img, *strength, *time),
            SceneEffect::Scroll { speed, time } => apply_scroll(img, *speed, *time),
            SceneEffect::Spin { speed, time } => apply_spin(img, *speed, *time),
        }
    }
}

fn apply_tint(img: &mut RgbaImage, r: f32, g: f32, b: f32, alpha: f32) {
    let tr = (r * 255.0).clamp(0.0, 255.0) as u16;
    let tg = (g * 255.0).clamp(0.0, 255.0) as u16;
    let tb = (b * 255.0).clamp(0.0, 255.0) as u16;
    let a = (alpha * 255.0).clamp(0.0, 255.0) as u16;
    let inv = 255 - a;

    for pixel in img.pixels_mut() {
        pixel[0] = ((pixel[0] as u16 * inv + tr * a) / 255) as u8;
        pixel[1] = ((pixel[1] as u16 * inv + tg * a) / 255) as u8;
        pixel[2] = ((pixel[2] as u16 * inv + tb * a) / 255) as u8;
    }
}

fn apply_opacity(img: &mut RgbaImage, alpha: f32) {
    let a = (alpha * 255.0).clamp(0.0, 255.0) as u8;
    for pixel in img.pixels_mut() {
        pixel[3] = ((pixel[3] as u16 * a as u16) / 255) as u8;
    }
}

fn apply_shake(img: &mut RgbaImage, strength: f32, time: f32) {
    let w = img.width() as f32;
    let offset_x = (time * 3.0).sin() * strength * w * 0.5;
    let offset_y = (time * 2.7).cos() * strength * w * 0.3;

    let ox = offset_x as i32;
    let oy = offset_y as i32;
    if ox == 0 && oy == 0 {
        return;
    }

    let clone = img.clone();
    let iw = img.width() as i32;
    let ih = img.height() as i32;

    for y in 0..ih {
        for x in 0..iw {
            let sx = (x + ox).clamp(0, iw - 1) as u32;
            let sy = (y + oy).clamp(0, ih - 1) as u32;
            *img.get_pixel_mut(x as u32, y as u32) = *clone.get_pixel(sx, sy);
        }
    }
}

fn apply_pulse(img: &mut RgbaImage, strength: f32, time: f32) {
    let pulse = ((time * 2.0).sin() * 0.5 + 0.5) * strength;
    let factor = 1.0 + pulse;

    for pixel in img.pixels_mut() {
        pixel[0] = (pixel[0] as f32 * factor).clamp(0.0, 255.0) as u8;
        pixel[1] = (pixel[1] as f32 * factor).clamp(0.0, 255.0) as u8;
        pixel[2] = (pixel[2] as f32 * factor).clamp(0.0, 255.0) as u8;
    }
}

fn apply_scroll(img: &mut RgbaImage, speed: f32, time: f32) {
    let shift = (time * speed * img.width() as f32) as i32;
    if shift == 0 {
        return;
    }

    let clone = img.clone();
    let width = img.width() as i32;
    for y in 0..img.height() {
        for x in 0..img.width() {
            let sx = (x as i32 - shift).rem_euclid(width) as u32;
            *img.get_pixel_mut(x, y) = *clone.get_pixel(sx, y);
        }
    }
}

fn apply_spin(img: &mut RgbaImage, speed: f32, time: f32) {
    let angle = time * speed;
    let (sin, cos) = angle.sin_cos();
    let cx = (img.width() as f32 - 1.0) * 0.5;
    let cy = (img.height() as f32 - 1.0) * 0.5;
    let clone = img.clone();

    for y in 0..img.height() {
        for x in 0..img.width() {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let sx = dx * cos + dy * sin + cx;
            let sy = -dx * sin + dy * cos + cy;
            let sx = sx.round().clamp(0.0, img.width() as f32 - 1.0) as u32;
            let sy = sy.round().clamp(0.0, img.height() as f32 - 1.0) as u32;
            *img.get_pixel_mut(x, y) = *clone.get_pixel(sx, sy);
        }
    }
}

fn extract_f32(v: &serde_json::Value, key: &str) -> Option<f32> {
    let val = v.get(key)?;
    match val {
        serde_json::Value::Number(n) => n.as_f64().map(|f| f as f32),
        serde_json::Value::Object(m) => m.get("value").and_then(|v| v.as_f64()).map(|f| f as f32),
        _ => None,
    }
}

fn extract_string(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key)?.as_str().map(|s| s.to_string())
}

pub(crate) fn parse_color(s: &str) -> [f32; 3] {
    let parts: Vec<f32> = s
        .split_whitespace()
        .filter_map(|p| p.parse().ok())
        .collect();
    [
        parts.first().copied().unwrap_or(1.0),
        parts.get(1).copied().unwrap_or(1.0),
        parts.get(2).copied().unwrap_or(1.0),
    ]
}
