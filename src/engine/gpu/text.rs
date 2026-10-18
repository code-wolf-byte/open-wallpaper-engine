//! Text object rendering (`CText.cpp` port).
//!
//! Matches the reference's own stated "Phase 1" scope: a single line of text
//! (no wrapping/padding), rasterized once from a system or wallpaper-bundled
//! TrueType font. We have no JS scripting engine, so scripted/dynamic text
//! renders whatever its initial value is and stays static — the reference's
//! own live-update path (`ScriptEngine::tickLayer`) has no equivalent here.
//!
//! Glyphs are rasterized in white with coverage as alpha; the object's own
//! `color`/`alpha` tint is applied later by the normal image-layer compositing
//! path (`Layer::color`/`Layer::alpha`), not baked in here — this lets a text
//! object reuse 100% of the existing image-layer machinery in both render
//! paths once it's turned into an `RgbaImage`.

use image::RgbaImage;

/// Rasterize `text` (one or more `\n`-separated lines) at `point_size` pixels.
/// Returns `None` for empty text. Multi-line strings (scripted clocks/dates
/// like `"7/13/2026\n8:35 AM"`) stack their lines vertically, centered.
pub fn rasterize(font_data: &[u8], text: &str, point_size: f32) -> Option<RgbaImage> {
    if text.is_empty() {
        return None;
    }
    if !text.contains('\n') {
        return rasterize_line(font_data, text, point_size);
    }
    // Multi-line: rasterize each line, then stack. Blank lines reserve a gap.
    let font = fontdue::Font::from_bytes(font_data, fontdue::FontSettings::default()).ok()?;
    let line_h = font
        .horizontal_line_metrics(point_size)
        .map(|m| (m.ascent - m.descent + m.line_gap).ceil() as u32)
        .unwrap_or((point_size * 1.2).ceil() as u32)
        .max(1);
    let lines: Vec<Option<RgbaImage>> = text
        .split('\n')
        .map(|l| rasterize_line(font_data, l, point_size))
        .collect();
    let width = lines
        .iter()
        .filter_map(|l| l.as_ref().map(|i| i.width()))
        .max()
        .unwrap_or(1)
        .max(1);
    let height = (line_h * lines.len() as u32).max(1);
    let mut img = RgbaImage::from_pixel(width, height, image::Rgba([255, 255, 255, 0]));
    for (i, line) in lines.iter().enumerate() {
        let Some(line) = line else { continue };
        // Center each line horizontally within the block, top-align in its row.
        let ox = (width - line.width()) / 2;
        let oy = i as u32 * line_h;
        for (x, y, px) in line.enumerate_pixels() {
            if px[3] > 0 {
                if let Some(dst) = img.get_pixel_mut_checked(ox + x, oy + y) {
                    *dst = *px;
                }
            }
        }
    }
    Some(img)
}

/// Rasterize a single line of `text` at `point_size` pixels using `font_data`.
/// Returns `None` for empty text (nothing to draw).
fn rasterize_line(font_data: &[u8], text: &str, point_size: f32) -> Option<RgbaImage> {
    if text.is_empty() {
        return None;
    }
    let font = fontdue::Font::from_bytes(font_data, fontdue::FontSettings::default()).ok()?;

    struct Glyph {
        metrics: fontdue::Metrics,
        bitmap: Vec<u8>,
        pen_x: f32,
    }

    let mut glyphs = Vec::with_capacity(text.chars().count());
    let mut pen_x: f32 = 0.0;
    let mut max_ascent: i32 = 1;
    let mut max_descent: i32 = 0;
    for ch in text.chars() {
        let (metrics, bitmap) = font.rasterize(ch, point_size);
        max_ascent = max_ascent.max(metrics.ymin + metrics.height as i32);
        max_descent = max_descent.max(-metrics.ymin);
        glyphs.push(Glyph {
            metrics,
            bitmap,
            pen_x,
        });
        pen_x += metrics.advance_width;
    }

    let width = (pen_x.ceil() as u32).max(1);
    let height = ((max_ascent + max_descent).max(1)) as u32;
    // Fill RGB white with alpha 0 (not the default all-zero = black/transparent).
    // Glyphs are white with coverage-as-alpha, so linear filtering / downscaling
    // interpolates white↔white at edges instead of white↔black — the latter
    // leaves a dark halo/outline around every glyph.
    let mut img = RgbaImage::from_pixel(width, height, image::Rgba([255, 255, 255, 0]));

    for g in &glyphs {
        let dst_x0 = g.pen_x as i32 + g.metrics.xmin;
        let dst_y0 = max_ascent - (g.metrics.ymin + g.metrics.height as i32);
        for row in 0..g.metrics.height {
            for col in 0..g.metrics.width {
                let coverage = g.bitmap[row * g.metrics.width + col];
                if coverage == 0 {
                    continue;
                }
                let dx = dst_x0 + col as i32;
                let dy = dst_y0 + row as i32;
                if dx < 0 || dy < 0 || dx as u32 >= width || dy as u32 >= height {
                    continue;
                }
                // "over" onto a transparent canvas: glyphs from neighboring
                // chars essentially never overlap for a normal font/spacing,
                // but max (not overwrite) avoids any anti-aliased seam if they
                // do (e.g. slightly negative advance/kerning-like fonts).
                let dst = img.get_pixel_mut(dx as u32, dy as u32);
                if coverage > dst[3] {
                    *dst = image::Rgba([255, 255, 255, coverage]);
                }
            }
        }
    }
    Some(img)
}

/// Word-wrap `text` so each line's rendered width stays under `max_width` px
/// at `point_size`, inserting `\n`. Existing `\n` are preserved. `max_rows`,
/// when >0, truncates to that many lines. Used for text objects that set
/// `maxwidth`/`maxrows`.
pub fn wrap_text(
    font_data: &[u8],
    text: &str,
    point_size: f32,
    max_width: f32,
    max_rows: usize,
    ellipsis: bool,
) -> String {
    let Some(font) = fontdue::Font::from_bytes(font_data, fontdue::FontSettings::default()).ok()
    else {
        return text.to_string();
    };
    let width_of = |s: &str| -> f32 {
        s.chars()
            .map(|c| font.metrics(c, point_size).advance_width)
            .sum()
    };
    let mut out: Vec<String> = Vec::new();
    for paragraph in text.split('\n') {
        let mut line = String::new();
        for word in paragraph.split(' ') {
            let candidate = if line.is_empty() {
                word.to_string()
            } else {
                format!("{line} {word}")
            };
            if !line.is_empty() && width_of(&candidate) > max_width {
                out.push(std::mem::take(&mut line));
                line = word.to_string();
            } else {
                line = candidate;
            }
        }
        out.push(line);
    }
    if max_rows > 0 && out.len() > max_rows {
        out.truncate(max_rows);
        // `limituseellipsis`: mark the clip instead of cutting mid-thought.
        if ellipsis {
            if let Some(last) = out.last_mut() {
                last.push('…');
            }
        }
    }
    out.join("\n")
}

/// Composite white-coverage `glyphs` (tinted `text_color`, 0-1) onto an opaque
/// `bg` box (0-1) expanded by `pad` px on every side. Colors are baked in, so
/// the caller draws the result with no further tint. `opaquebackground`.
pub fn with_background(
    glyphs: &RgbaImage,
    text_color: [f32; 3],
    bg: [f32; 3],
    pad: u32,
) -> RgbaImage {
    let w = glyphs.width() + pad * 2;
    let h = glyphs.height() + pad * 2;
    let bgp = image::Rgba([
        (bg[0] * 255.0) as u8,
        (bg[1] * 255.0) as u8,
        (bg[2] * 255.0) as u8,
        255,
    ]);
    let mut out = RgbaImage::from_pixel(w, h, bgp);
    for (x, y, px) in glyphs.enumerate_pixels() {
        let a = px[3] as f32 / 255.0;
        if a <= 0.0 {
            continue;
        }
        // Glyph is white-with-coverage; tint by text_color, then over bg.
        let dst = out.get_pixel_mut(x + pad, y + pad);
        for i in 0..3 {
            let fg = text_color[i] * 255.0;
            dst[i] = (fg * a + dst[i] as f32 * (1.0 - a)) as u8;
        }
    }
    out
}

/// Resolve font bytes for a text object's `font` field: a wallpaper-bundled
/// path (checked in the loose directory, then the pkg) if given and not a
/// `"systemfont_*"` placeholder, else a system font.
pub fn resolve_font_data(
    font_ref: Option<&str>,
    dir: Option<&std::path::Path>,
    pkg: Option<&crate::engine::pkg::Package>,
) -> Option<Vec<u8>> {
    if let Some(font_ref) = font_ref {
        if !font_ref.is_empty() && !font_ref.starts_with("systemfont_") {
            if let Some(dir) = dir {
                if let Ok(data) = std::fs::read(dir.join(font_ref)) {
                    return Some(data);
                }
            }
            if let Some(pkg) = pkg {
                if let Some(data) = pkg.get(font_ref) {
                    return Some(data.to_vec());
                }
            }
        }
    }
    find_system_font()
}

/// `fc-match` (fontconfig) works across distros without hardcoding paths;
/// the hardcoded list is a fallback for systems without it installed.
fn find_system_font() -> Option<Vec<u8>> {
    if let Ok(output) = std::process::Command::new("fc-match")
        .arg("--format=%{file}")
        .arg("sans-serif")
        .output()
    {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                if let Ok(data) = std::fs::read(&path) {
                    return Some(data);
                }
            }
        }
    }

    const CANDIDATES: &[&str] = &[
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation-fonts/LiberationSans-Regular.ttf",
    ];
    for path in CANDIDATES {
        if let Ok(data) = std::fs::read(path) {
            return Some(data);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `limituseellipsis` marks a row-clipped block instead of cutting it
    /// silently; without the gate the text is hard-truncated.
    #[test]
    fn ellipsis_marks_clipped_text() {
        let f = test_font();
        let long = "aaa bbb ccc ddd eee fff ggg hhh";
        let clipped = wrap_text(&f, long, 24.0, 60.0, 1, false);
        let ellipsed = wrap_text(&f, long, 24.0, 60.0, 1, true);
        assert!(!clipped.ends_with('…'), "gate off: no ellipsis");
        assert!(ellipsed.ends_with('…'), "gate on: ellipsis appended");
        assert_eq!(clipped.lines().count(), 1, "maxrows still clips to 1 row");
    }

    fn test_font() -> Vec<u8> {
        find_system_font().expect("a system font must be available to test rasterization")
    }

    #[test]
    fn rasterize_empty_text_returns_none() {
        assert!(rasterize(&test_font(), "", 32.0).is_none());
    }

    #[test]
    fn rasterize_produces_nonempty_bitmap_with_visible_glyphs() {
        let img = rasterize(&test_font(), "Hi", 32.0).expect("should rasterize");
        assert!(img.width() > 0 && img.height() > 0);
        let has_visible_pixel = img.pixels().any(|p| p.0[3] > 0);
        assert!(
            has_visible_pixel,
            "expected at least one non-transparent pixel"
        );
    }

    #[test]
    fn rasterize_glyphs_are_white_with_coverage_alpha() {
        let img = rasterize(&test_font(), "I", 48.0).expect("should rasterize");
        for p in img.pixels() {
            if p.0[3] > 0 {
                assert_eq!([p.0[0], p.0[1], p.0[2]], [255, 255, 255]);
            }
        }
    }

    #[test]
    fn longer_text_produces_wider_bitmap() {
        let short = rasterize(&test_font(), "I", 32.0).unwrap();
        let long = rasterize(&test_font(), "IIIIIIIIII", 32.0).unwrap();
        assert!(long.width() > short.width());
    }
}
