//! Turning scene objects into layers: image, particle, text, 3D mesh,
//! light and backdrop layers, with their placement (alignment, parallax
//! depth, spritesheet frame) and the puppet mesh an image may carry.

use super::*;

/// WE `parallaxDepth` is always a vec2 (independent x/y depth), but a bare
/// number in scene.json means "same depth on both axes".
pub(super) fn parse_parallax_depth(v: &serde_json::Value) -> [f64; 2] {
    use crate::engine::model::DynamicValue;
    match crate::engine::model::json_to_animated(v).value {
        DynamicValue::Float(f) => [f as f64; 2],
        DynamicValue::Vec3([x, y, _]) => [x as f64, y as f64],
        _ => [0.0, 0.0],
    }
}

/// WE origin convention: Y-up, (0,0) at bottom-left (same as `Layer::origin`).
/// "top"/"bottom" shift the object vertically by half its scaled height so
/// the origin becomes that edge instead of the center; "left"/"right" do the
/// same horizontally. Matches CImage.cpp lines 242-256 exactly (each pair is
/// mutually exclusive — "top bottom" together, like the reference, only
/// honors "top").
pub(crate) fn alignment_offset(alignment: Option<&str>, scaled_size: [f64; 2]) -> [f64; 2] {
    let a = alignment.unwrap_or("").to_lowercase();
    let mut offset = [0.0, 0.0];
    if a.contains("top") {
        offset[1] -= scaled_size[1] / 2.0;
    } else if a.contains("bottom") {
        offset[1] += scaled_size[1] / 2.0;
    }
    if a.contains("left") {
        offset[0] += scaled_size[0] / 2.0;
    } else if a.contains("right") {
        offset[0] -= scaled_size[0] / 2.0;
    }
    offset
}

/// `alignment_override`, when given, replaces `obj.alignment` (text objects
/// combine separate `horizontalalign`/`verticalalign` fields into one string
/// before calling this, rather than reading `obj.alignment` directly).
/// Brightness multiplier for additive (colorBlendMode 9) image layers — see the
/// call site. Default 0.55; override live with `WP_ENGINE_ADDITIVE_BRIGHTNESS`.
pub(super) fn additive_brightness_scale() -> f32 {
    std::env::var("WP_ENGINE_ADDITIVE_BRIGHTNESS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.55)
}

/// For a `SPRITESHEET` layer, pick the single frame the object's frame script
/// selects and make it the layer image. Frames are the `.tex` TEXS sub-rects; the
/// index comes from the object's frame-select script (WE hangs
/// `getTextureAnimation().setFrame` off the `visible` script — e.g. AM/PM by
/// `getHours()`). Falls back to frame 0 if the script sets none.
pub(super) fn select_spritesheet_frame(obj: &SceneObject, loaded: &mut LoadedImage) {
    if loaded.sprite_frames.len() < 2 {
        return;
    }
    let frame = obj
        .visible
        .as_ref()
        .and_then(|v| v.get("script"))
        .and_then(|s| s.as_str())
        .and_then(|script| {
            let init = crate::engine::sim::script::ObjectInit {
                visible: true,
                alpha: 1.0,
                frame_count: loaded.sprite_frames.len() as u32,
                ..Default::default()
            };
            crate::engine::sim::script::evaluate_once("visible", script, &init, None).frame
        })
        .unwrap_or(0)
        .min(loaded.sprite_frames.len() as u32 - 1);
    loaded.image = loaded.sprite_frames[frame as usize].clone();
}

/// Build the layer for one object outside the scene load — SceneScript's
/// `thisScene.createLayer`. Covers image, solid and text objects; particle
/// and model objects return `None`. `pkg` is the scene's package, if any.
pub fn runtime_layer(dir: &Path, pkg: Option<&Package>, obj: &SceneObject) -> Option<Layer> {
    if obj.text.is_some() {
        return text_layer_from_object(obj, Some(dir), pkg);
    }
    let path = obj.image.as_deref()?;
    if is_special_layer(path) {
        return backdrop_layer_from_object(obj, path);
    }
    let resolved_path = resolve_workshop_asset(dir, pkg, path);
    let path = resolved_path.as_str();
    let loaded = if is_solid_layer_path(path) {
        LoadedImage::single(white_pixel())
    } else {
        let from_pkg = pkg.map(|p| load_texture_from_pkg(p, path));
        match from_pkg {
            Some(Ok(l)) => l,
            _ => load_texture_from_dir(dir, path)
                .inspect_err(
                    |e| tracing::warn!(target: "scene", "createLayer: texture '{path}': {e:#}"),
                )
                .ok()?,
        }
    };
    Some(layer_from_object(obj, loaded, None))
}

pub(super) fn layer_from_object(
    obj: &SceneObject,
    mut loaded: LoadedImage,
    alignment_override: Option<&str>,
) -> Layer {
    // Spritesheet: replace the flattened atlas with the single cell the object's
    // frame script selects (e.g. AM/PM by `getHours()`), so the one-cell quad
    // shows one glyph instead of every cell squished together.
    // ponytail: frame is picked once at load from the current wall clock; a live
    // wallpaper won't flip AM↔PM mid-session without a reload (rare — re-select in
    // the render loop if that ceiling ever matters).
    select_spritesheet_frame(obj, &mut loaded);
    // Attach puppet `animationlayers` (blended each frame) to the fresh runtime.
    if let (Some(rt), Some(entries)) = (
        loaded.puppet.as_mut(),
        obj.animationlayers.as_ref().and_then(|v| v.as_array()),
    ) {
        if let Some(rt) = std::sync::Arc::get_mut(rt) {
            let n = rt.model.animations.len();
            if n > 0 {
                let layers: Vec<crate::engine::sim::puppet::AnimLayer> = entries
                    .iter()
                    .enumerate()
                    // `visible` is often a user-property binding
                    // (`{"user": "head", "value": false}`), not a bare bool.
                    .filter(|(_, e)| {
                        e.get("visible")
                            .and_then(crate::engine::model::scene::parse_value_bool)
                            .unwrap_or(true)
                    })
                    .map(|(i, e)| {
                        // `animation` is the MDLA record's own id (2321732083's
                        // samurai layers name 60/63/80/105). Match it; fall back
                        // to index-then-declaration-order for files whose ids we
                        // failed to read.
                        let raw = e
                            .get("animation")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(i as u64);
                        let anim_idx = rt
                            .model
                            .animations
                            .iter()
                            .position(|a| a.id as u64 == raw)
                            .unwrap_or_else(|| {
                                if (raw as usize) < n {
                                    raw as usize
                                } else {
                                    i.min(n - 1)
                                }
                            });
                        crate::engine::sim::puppet::AnimLayer {
                            anim_idx,
                            additive: e.get("additive").and_then(|v| v.as_bool()).unwrap_or(false),
                            blend: e.get("blend").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                            rate: e.get("rate").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                        }
                    })
                    .collect();
                rt.layers = layers;
            }
        }
    }
    let parallax = obj
        .parallax_depth
        .as_ref()
        .map(parse_parallax_depth)
        .unwrap_or([0.0, 0.0]);

    // WE stores angles already in radians; only the z component (2D roll)
    // applies to a flat layer quad. The reference negates it (rotate(-angle,
    // ...)) when building the object's screen transform.
    let angles_animated = obj
        .angles
        .as_ref()
        .map(crate::engine::model::json_to_animated);
    let angles3 = angles_animated
        .as_ref()
        .and_then(|v| v.as_vec3())
        .unwrap_or([0.0, 0.0, 0.0]);
    let angle = -angles3[2];

    let scale_animated = obj
        .scale
        .as_ref()
        .map(crate::engine::model::json_to_animated);
    let scale = scale_animated
        .as_ref()
        .and_then(|v| v.as_vec3())
        .map(|s| [s[0] as f64, s[1] as f64, s[2] as f64])
        .unwrap_or([1.0, 1.0, 1.0]);
    let alpha_animated = obj
        .alpha
        .as_ref()
        .map(crate::engine::model::json_to_animated);
    let alpha = alpha_animated
        .as_ref()
        .and_then(|v| v.as_float())
        .unwrap_or(1.0);
    let color = obj
        .color
        .as_ref()
        .map(crate::engine::model::json_to_animated)
        .and_then(|v| v.as_vec3())
        .unwrap_or([1.0, 1.0, 1.0]);
    let brightness = obj
        .brightness
        .as_ref()
        .map(crate::engine::model::json_to_animated)
        .and_then(|v| v.as_float())
        .unwrap_or(1.0);
    // ponytail: additive (colorBlendMode 9) image planes with a near-white base
    // (e.g. LonelyCat's water-ripple layer) blow out to a flat white disc,
    // because our water effects refract the layer's *own* bright base instead of
    // the real backdrop (backdrop capture is unimplemented — see PROGRESS.md).
    // Until that lands, scale additive layers down so the ripple structure stays
    // visible. Ceiling: this dims *every* additive image layer, not just ripples;
    // remove once screen-space backdrop capture exists. Tunable live via
    // WP_ENGINE_ADDITIVE_BRIGHTNESS (default 0.55).
    let brightness = if obj.color_blend_mode == 9 {
        brightness * additive_brightness_scale()
    } else {
        brightness
    };

    // Alignment shifts the object so its origin becomes an edge rather than
    // its center (CImage.cpp lines 242-256), expressed here as an offset
    // folded directly into `origin` — every consumer of `Layer::origin`
    // (both render paths) then gets alignment for free.
    // Text objects (the only caller passing `alignment_override`) size their
    // quad from the rasterized glyph bitmap, not the scene `size` field —
    // WE's CText uses `m_quadSize` (the raster dims) and ignores `size`
    // (which is editor bounding-box metadata). Honoring it would stretch the
    // glyphs to an unrelated box.
    let is_text = alignment_override.is_some();
    let raw_size = obj.parsed_size();
    let effective_size = if !is_text && raw_size[0] > 0.0 && raw_size[1] > 0.0 {
        [raw_size[0], raw_size[1]]
    } else {
        [loaded.image.width() as f64, loaded.image.height() as f64]
    };
    let scaled_size = [effective_size[0] * scale[0], effective_size[1] * scale[1]];
    let alignment = alignment_override.or(obj.alignment.as_deref());
    let align_offset = alignment_offset(alignment, scaled_size);
    let origin = {
        let o = obj.parsed_origin();
        [o[0] + align_offset[0], o[1] + align_offset[1], o[2]]
    };

    Layer {
        backdrop: None,
        name: obj.name.clone().unwrap_or_default(),
        image: loaded.image,
        extra_frames: loaded.extra_frames,
        frame_duration_ms: loaded.frame_duration_ms,
        origin,
        // Text: leave size 0 so both render paths fall back to the raster dims.
        size: if is_text {
            [0.0, 0.0, 0.0]
        } else {
            obj.parsed_size()
        },
        scale,
        local_origin: origin,
        local_scale: scale,
        local_angles: angles3,
        parallax_depth: parallax,
        angle,
        angles: angles3,
        blend_mode: obj.color_blend_mode,
        alpha,
        color,
        brightness,
        copybackground: obj.copybackground,
        no_interpolation: loaded.no_interpolation,
        // Object-level `clampuvs` overrides the .tex ClampUVs flag when set.
        clamp_uvs: loaded.clamp_uvs
            || obj
                .clampuvs
                .as_ref()
                .and_then(|v| {
                    v.as_bool()
                        .or_else(|| v.get("value").and_then(|i| i.as_bool()))
                })
                .unwrap_or(false),
        puppet: loaded.puppet,
        video: loaded.video,
        text_dynamic: None,
        order_index: 0,
        object_id: obj.id,
        visible_base: obj.is_visible(),
    }
}

pub(super) fn read_particle_json(
    dir: Option<&Path>,
    pkg: Option<&Package>,
    rel_path: &str,
) -> Option<String> {
    String::from_utf8(read_asset_bytes(dir, pkg, rel_path)?).ok()
}

/// Build a `ParticleLayer` from a scene object whose `particle` field is a
/// preset path, resolving and parsing the referenced JSON. Returns `None`
/// (silently skipped, like a missing texture) if the preset can't be found
/// or parsed.
pub(super) fn particle_layer_from_object(
    obj: &SceneObject,
    dir: Option<&Path>,
    pkg: Option<&Package>,
) -> Option<ParticleLayer> {
    let particle_ref = match &obj.particle {
        Some(serde_json::Value::String(s)) => s.as_str(),
        _ => return None,
    };
    let json = read_particle_json(dir, pkg, particle_ref)?;
    let mut config: particle::ParticleConfig = serde_json::from_str(&json)
        .inspect_err(
            |e| tracing::warn!(target: "particle", "failed to parse '{particle_ref}': {e}"),
        )
        .ok()?;
    let overrides: Option<particle::InstanceOverride> = obj
        .instanceoverride
        .as_ref()
        .and_then(|v| serde_json::from_value(v.clone()).ok());

    // `instanceoverride.enabled: false` turns the whole system off
    // (ObjectParser defaults it to true).
    if overrides.as_ref().is_some_and(|o| {
        o.enabled
            .as_ref()
            .is_some_and(|v| v.as_bool() == Some(false) || v.as_u64() == Some(0))
    }) {
        return None;
    }

    // Control-point plumbing the simulation can't do itself (it only knows
    // spawn-relative screen offsets):
    // 1. `instanceoverride.controlpointN` repositions preset control points
    //    with absolute scene coordinates (e.g. the discharge preset's arc
    //    endpoint, "1586.7 993.2") — absolute values are world-space, so
    //    force flag 2 on.
    // 2. Any world-space control point (flags & 2) is then converted to a
    //    spawn-relative offset here, where the object's origin is known:
    //    offset = (x_we - origin_x, origin_y - y_we) in the simulation's
    //    y-down screen space.
    if let Some(over) = overrides.as_ref() {
        for (key, value) in &over.extra {
            let Some(n) = key
                .strip_prefix("controlpoint")
                .and_then(|s| s.parse::<usize>().ok())
            else {
                continue;
            };
            while config.controlpoint.len() <= n {
                config
                    .controlpoint
                    .push(particle::ControlPointConfig::default());
            }
            config.controlpoint[n].offset = Some(value.clone());
            config.controlpoint[n].flags = Some(config.controlpoint[n].flags.unwrap_or(0) | 2);
        }
    }
    let origin = obj.parsed_origin();
    for cp in &mut config.controlpoint {
        if cp.flags.unwrap_or(0) & 2 == 0 {
            continue;
        }
        if let Some(world) = cp.offset.as_ref().and_then(particle::value_as_vec3_pub) {
            let rel = [
                world[0] - origin[0] as f32,
                origin[1] as f32 - world[1],
                world[2],
            ];
            cp.offset = Some(serde_json::json!(format!(
                "{} {} {}",
                rel[0], rel[1], rel[2]
            )));
            cp.flags = Some(cp.flags.unwrap_or(0) & !2);
        }
    }
    let parallax_depth = obj
        .parallax_depth
        .as_ref()
        .map(parse_parallax_depth)
        .unwrap_or([0.0, 0.0]);

    let resolved_sprite = config.material.as_deref().and_then(|mat_path| {
        // A failure here silently degrades to the flat-color circle draw —
        // visually plausible for dust/snow but wrong for shaped sprites, so
        // it must at least be visible in the log.
        if let Some(pkg) = pkg {
            resolve_particle_sprite_pkg(pkg, mat_path)
                .inspect_err(|e| {
                    eprintln!("[particle] sprite '{mat_path}' failed (untextured fallback): {e:#}")
                })
                .ok()
        } else {
            dir.and_then(|dir| {
                resolve_particle_sprite_dir(dir, mat_path)
                    .inspect_err(|e| {
                        eprintln!(
                            "[particle] sprite '{mat_path}' failed (untextured fallback): {e:#}"
                        )
                    })
                    .ok()
            })
        }
    });
    let additive_blend = resolved_sprite
        .as_ref()
        .is_some_and(|(_, blending)| blending.as_deref() == Some("additive"));
    let sprite_texture = resolved_sprite.map(|(sprite, _)| sprite);

    // Child presets: same preset-JSON + material resolution as the parent.
    // Failures degrade to "child skipped" (logged), never sink the layer.
    let children: Vec<ResolvedChildParticle> = config
        .children
        .clone()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|child_ref| {
            let json = read_particle_json(dir, pkg, &child_ref.name).or_else(|| {
                eprintln!("[particle] child preset '{}' not found", child_ref.name);
                None
            })?;
            let child_cfg: particle::ParticleConfig = serde_json::from_str(&json)
                .inspect_err(|e| {
                    eprintln!("[particle] child '{}' parse failed: {e}", child_ref.name)
                })
                .ok()?;
            let resolved = child_cfg.material.as_deref().and_then(|mat_path| {
                if let Some(pkg) = pkg {
                    resolve_particle_sprite_pkg(pkg, mat_path).ok()
                } else {
                    dir.and_then(|dir| resolve_particle_sprite_dir(dir, mat_path).ok())
                }
            });
            let additive = resolved
                .as_ref()
                .is_some_and(|(_, blending)| blending.as_deref() == Some("additive"));
            let sprite = resolved.map(|(sprite, _)| sprite);
            Some(ResolvedChildParticle {
                config: child_cfg,
                sprite,
                additive,
                child_ref,
            })
        })
        .collect();

    Some(ParticleLayer {
        name: obj.name.clone().unwrap_or_default(),
        origin: obj.parsed_origin(),
        local_origin: obj.parsed_origin(),
        parallax_depth,
        config,
        overrides,
        sprite_texture,
        additive_blend,
        order_index: 0,
        children,
    })
}

/// `text` is either a plain JSON string or a `{"user": ..., "value": "..."}`
/// wrapper (scene.json's usual user-setting shape) — deliberately NOT routed
/// through `json_to_animated`, which treats 2-3 whitespace-separated numbers
/// as a vector; that heuristic is wrong for text content that happens to be
/// purely numeric (e.g. a countdown reading "12 34").
pub(super) fn extract_text_string(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(map) => map.get("value").and_then(extract_text_string),
        _ => None,
    }
}

/// Build a text object's rendered bitmap as a normal `Layer` (see
/// `engine::gpu::text` module docs for why: no scripting engine means text never
/// changes after this one rasterization, so it can reuse 100% of the
/// existing image-layer pipeline in both render paths for free).
pub(super) fn text_layer_from_object(
    obj: &SceneObject,
    dir: Option<&Path>,
    pkg: Option<&Package>,
) -> Option<Layer> {
    // A text object's `text` is either a plain string, a {"value": …}
    // wrapper, or a scripted {"script": "…", "value"?: …} — clocks/dates
    // (the dominant case in real content) often ship the script with NO
    // static value at all, so the script's first evaluation IS the content.
    let text_value = obj.text.as_ref()?;
    let script = text_value
        .get("script")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let script_properties = text_value.get("scriptproperties").cloned();
    let static_str = extract_text_string(text_value).unwrap_or_default();
    let text_str = match (&script, static_str.is_empty()) {
        (Some(script), _) => {
            // One-shot load-time evaluation so the layer rasterizes real
            // content (and sizes itself correctly) from frame 0.
            let init = crate::engine::sim::script::ObjectInit {
                name: obj.name.clone(),
                id: obj.id,
                visible: true,
                alpha: 1.0,
                text: Some(static_str.clone()),
                ..Default::default()
            };
            crate::engine::sim::script::evaluate_once(
                "text",
                script,
                &init,
                script_properties.as_ref(),
            )
            .text
            .filter(|t| !t.is_empty())
            .unwrap_or(static_str)
        }
        (None, false) => static_str,
        (None, true) => return None,
    };
    if text_str.is_empty() {
        return None;
    }
    let point_size = obj
        .pointsize
        .as_ref()
        .and_then(|v| {
            v.as_f64()
                .or_else(|| v.get("value").and_then(|v| v.as_f64()))
        })
        .unwrap_or(32.0) as f32;
    let scale = obj
        .scale
        .as_ref()
        .and_then(|v| crate::engine::model::json_to_animated(v).as_vec3())
        .unwrap_or([1.0, 1.0, 1.0]);
    // The on-screen text height is driven by the object's `size` BOX height,
    // not pointsize: `layer_from_object` scales the rasterized bitmap by the
    // object's `scale`, and WE authors size text via that box (a big box +
    // scale>1 = big text; a modest box + scale<1 = small text). Rasterize at
    // the box height so the bitmap ≈ its on-screen size (crisp, natural aspect,
    // no stretch). `pointsize` here is only the fallback resolution when the
    // object ships no box (then compensate for `scale` like CText does, so the
    // glyphs don't collapse to a couple of pixels).
    let box_h = obj.parsed_size()[1] as f32;
    let raster_px = if box_h > 0.0 {
        box_h
    } else {
        let avg_scale = (scale[0] + scale[1]) * 0.5;
        let compensate = if avg_scale > 0.0 && avg_scale < 1.0 {
            (1.0 / avg_scale).min(32.0)
        } else {
            1.0
        };
        point_size * compensate
    }
    .clamp(1.0, 1024.0);
    let font_data = crate::engine::gpu::text::resolve_font_data(obj.font.as_deref(), dir, pkg)?;
    // Word-wrap to `maxwidth` (scene px == bitmap px at our raster resolution)
    // and cap at `maxrows`, if authored.
    let gate = |v: &Option<serde_json::Value>| {
        v.as_ref()
            .and_then(crate::engine::model::scene::parse_value_bool)
            .unwrap_or(false)
    };
    let text_str = match obj
        .maxwidth
        .as_ref()
        .and_then(crate::engine::model::scene::parse_value_f32)
        .filter(|w| *w > 0.0)
        .filter(|_| gate(&obj.limitwidth))
    {
        Some(mw) => {
            let rows = obj
                .maxrows
                .as_ref()
                .and_then(crate::engine::model::scene::parse_value_f32)
                .filter(|_| gate(&obj.limitrows))
                .unwrap_or(0.0)
                .max(0.0) as usize;
            crate::engine::gpu::text::wrap_text(
                &font_data,
                &text_str,
                raster_px,
                mw,
                rows,
                gate(&obj.limituseellipsis),
            )
        }
        None => text_str,
    };
    let image = crate::engine::gpu::text::rasterize(&font_data, &text_str, raster_px)?;

    // Opaque background box: bake the text color + bg color + padding into the
    // bitmap and neutralize the layer tint (color set to white below). `pad`
    // is authored in scene px, which maps 1:1 to bitmap px (we raster at the
    // box height). Off by default — `opaquebackground` is usually false.
    let opaque_bg = obj
        .opaquebackground
        .as_ref()
        .and_then(|v| {
            v.as_bool()
                .or_else(|| v.get("value").and_then(|i| i.as_bool()))
        })
        .unwrap_or(false);
    let (image, bg_baked) = if opaque_bg {
        let text_color = obj
            .color
            .as_ref()
            .and_then(|v| crate::engine::model::scene::parse_value_vec3(v))
            .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
            .unwrap_or([1.0, 1.0, 1.0]);
        let bright = obj
            .backgroundbrightness
            .as_ref()
            .and_then(crate::engine::model::scene::parse_value_f32)
            .unwrap_or(1.0);
        let bg = obj
            .backgroundcolor
            .as_ref()
            .and_then(|v| crate::engine::model::scene::parse_value_vec3(v))
            .map(|c| {
                [
                    c[0] as f32 * bright,
                    c[1] as f32 * bright,
                    c[2] as f32 * bright,
                ]
            })
            .unwrap_or([0.0, 0.0, 0.0]);
        let pad = obj
            .padding
            .as_ref()
            .and_then(crate::engine::model::scene::parse_value_f32)
            .unwrap_or(0.0)
            .max(0.0) as u32;
        (
            crate::engine::gpu::text::with_background(&image, text_color, bg, pad),
            true,
        )
    } else {
        (image, false)
    };

    let halign = obj
        .horizontalalign
        .as_deref()
        .or(obj.alignment.as_deref())
        .unwrap_or("center");
    let valign = obj.verticalalign.as_deref().unwrap_or("center");
    let combined = format!("{halign} {valign}");

    let mut layer = layer_from_object(obj, LoadedImage::single(image), Some(&combined));
    // WE fits the glyphs into the authored `size` box preserving their
    // aspect (letterbox), not stretch-to-fill: a tall clock box (375×161)
    // over ~99×26 glyphs would render the block ~6× too tall and collide with
    // the layer below it. Shrink the box to the bitmap aspect — the
    // constrained dimension keeps the box, the other pulls in.
    // ponytail: refit only; origin's alignment offset was folded for the
    // original box, so this is exact for center-aligned text (all clocks/dates
    // here) and would need an offset re-fold for left/top anchors.
    let bw = layer.image.width() as f64;
    let bh = layer.image.height() as f64;
    if bw > 0.0 && bh > 0.0 && layer.size[0] > 0.0 && layer.size[1] > 0.0 {
        let aspect = bw / bh;
        let (box_w, box_h) = (layer.size[0], layer.size[1]);
        let fitted = if box_w / box_h > aspect {
            [box_h * aspect, box_h]
        } else {
            [box_w, box_w / aspect]
        };
        layer.size[0] = fitted[0];
        layer.size[1] = fitted[1];
    }
    if bg_baked {
        // Text + bg colors are baked into the bitmap; don't tint it again.
        layer.color = [1.0, 1.0, 1.0];
    }
    // Any text layer can be re-rasterized: its own script, or another
    // object's script, may set its text.
    layer.text_dynamic = Some(TextDynamic {
        font_data,
        point_size: raster_px,
        last_text: text_str,
    });
    Some(layer)
}

/// Build a static-3D-mesh layer for an object whose `model` names a `.mdl`.
/// `read` fetches an asset by relative path (dir- or pkg-backed), and
/// `resolve_tex` turns the mesh's embedded material path into its texture.
pub(super) fn mesh3d_layer_from_object(
    obj: &SceneObject,
    dir: Option<&Path>,
    pkg: Option<&Package>,
) -> Option<Mesh3dLayer> {
    let path = obj.mesh3d_path()?;
    // The pkg stores every path lowercased, while scene.json and the `.mdl`
    // header keep the author's casing (`models/LP/LP.mdl`, and real content
    // has non-ASCII names too) — so retry lowercased on any miss.
    let bytes = read_asset_bytes(dir, pkg, path)
        .or_else(|| read_asset_bytes(dir, pkg, &path.to_lowercase()))?;
    let mesh = crate::engine::model::mesh3d::parse(&bytes)?;
    let resolve_tex = |p: &str| {
        dir.and_then(|d| load_texture_from_dir(d, p).ok())
            .or_else(|| pkg.and_then(|p2| load_texture_from_pkg(p2, p).ok()))
            .map(|l| l.image)
    };
    let pass = material_first_pass(dir, pkg, &mesh.material);
    let texture = resolve_tex(&mesh.material)
        .or_else(|| resolve_tex(&mesh.material.to_lowercase()))
        // Untextured materials are normal here (`"textures": [null,null,null]`
        // with a flat `color` constant): draw them as that color. An
        // unresolvable material falls through to white rather than vanishing.
        .unwrap_or_else(|| {
            let [r, g, b] = pass
                .as_ref()
                .and_then(pass_constant_color)
                .unwrap_or([255; 3]);
            RgbaImage::from_pixel(1, 1, image::Rgba([r, g, b, 255]))
        });
    let nocull = pass
        .as_ref()
        .and_then(|p| p.get("cullmode")?.as_str())
        .is_some_and(|m| m == "nocull");
    // The object's own `depthtest` wins over the material pass's.
    let depthtest = obj
        .depthtest
        .as_ref()
        .and_then(|v| v.as_str())
        .or_else(|| pass.as_ref().and_then(|p| p.get("depthtest")?.as_str()))
        .map(|m| m != "disabled")
        .unwrap_or(true);

    let scale = obj
        .scale
        .as_ref()
        .and_then(|v| crate::engine::model::json_to_animated(v).as_vec3())
        .unwrap_or([1.0, 1.0, 1.0]);
    let angles = obj
        .angles
        .as_ref()
        .and_then(|v| crate::engine::model::json_to_animated(v).as_vec3())
        .unwrap_or([0.0, 0.0, 0.0]);
    let o = obj.parsed_origin();

    tracing::debug!(
        target: "scene",
        "mesh3d '{path}': {} verts, {} tris, material '{}'",
        mesh.positions.len(),
        mesh.indices.len() / 3,
        mesh.material
    );
    Some(Mesh3dLayer {
        name: obj.name.clone().unwrap_or_default(),
        mesh,
        texture,
        origin: [o[0] as f32, o[1] as f32, o[2] as f32],
        scale,
        angles,
        local_origin: [o[0] as f32, o[1] as f32, o[2] as f32],
        local_scale: scale,
        local_angles: angles,
        nocull,
        depthtest,
        order_index: 0,
    })
}

/// A WE light object rendered as an additive radial glow — a minimal stand-in
/// for the full lighting model. Colored by `color`, sized by `radius`, with a
/// smooth quadratic falloff scaled by `intensity`. Additively blended so it
/// brightens whatever is behind it (like a real light).
pub(super) fn light_layer_from_object(obj: &SceneObject) -> Option<Layer> {
    let (radius, intensity, color) = obj.light_params()?;

    let d = (radius * 2.0).min(2048.0) as u32;
    let mut img = RgbaImage::new(d, d);
    let c = d as f32 / 2.0;
    for y in 0..d {
        for x in 0..d {
            let dist = (((x as f32 - c).powi(2) + (y as f32 - c).powi(2)).sqrt() / c).min(1.0);
            let a = (1.0 - dist).powi(2) * intensity; // quadratic falloff
            img.put_pixel(
                x,
                y,
                image::Rgba([
                    (color[0] * 255.0) as u8,
                    (color[1] * 255.0) as u8,
                    (color[2] * 255.0) as u8,
                    (a * 255.0) as u8,
                ]),
            );
        }
    }

    let mut layer = layer_from_object(obj, LoadedImage::single(img), None);
    // colorBlendMode 9 = Add (linear dodge) — the glow adds light.
    layer.blend_mode = 9;
    layer.color = [1.0, 1.0, 1.0];
    Some(layer)
}

/// If the model JSON names a `"puppet"` mesh, its texture is a packed UV
/// atlas — reassemble it by rasterizing the rest-pose mesh (see
/// `engine::sim::puppet`), and keep the parsed model + atlas around as a
/// `PuppetRuntime` so the live render loop can re-pose it over time when
/// the model carries MDLA animations. On any parse failure the decoded
/// texture passes through unchanged, so a malformed .mdl degrades to the
/// old scrambled-quad behavior instead of dropping the layer.
pub(super) fn apply_puppet_mesh(
    atlas: RgbaImage,
    model_json_path: &str,
    read: impl Fn(&str) -> Option<Vec<u8>>,
) -> LoadedImage {
    let plain = |atlas: RgbaImage| LoadedImage::single(atlas);
    let Some(model_bytes) = read(model_json_path) else {
        return plain(atlas);
    };
    let Ok(model_json) = serde_json::from_slice::<serde_json::Value>(&model_bytes) else {
        return plain(atlas);
    };
    let Some(puppet_path) = model_json.get("puppet").and_then(|v| v.as_str()) else {
        return plain(atlas);
    };
    let Some(mdl_bytes) = read(puppet_path) else {
        tracing::warn!(target: "scene", "puppet mesh '{puppet_path}' not found — drawing raw atlas");
        return plain(atlas);
    };
    let Some(model) = crate::engine::sim::puppet::parse_model(&mdl_bytes) else {
        tracing::warn!(target: "scene", "puppet mesh '{puppet_path}' unparsable — drawing raw atlas");
        return plain(atlas);
    };
    tracing::debug!(
        target: "scene",
        "puppet '{puppet_path}': {} vertices, {} triangles, {} bones, {} animations",
        model.mesh.positions.len(),
        model.mesh.indices.len() / 3,
        model.bones.len(),
        model.animations.len()
    );
    // With a skeleton + animation, frame 0 of the animation IS the
    // assembled pose (skin = worldAnim * inverse(atlas-space bind)); the
    // raw mesh alone only reproduces the packed atlas layout.
    let runtime = crate::engine::sim::puppet::PuppetRuntime {
        model,
        atlas,
        layers: Vec::new(),
    };
    let assembled = if runtime.model.has_animation() {
        runtime.render_at(0.0, runtime.atlas.width(), runtime.atlas.height())
    } else {
        crate::engine::sim::puppet::rasterize(
            &runtime.model.mesh,
            &runtime.atlas,
            runtime.atlas.width(),
            runtime.atlas.height(),
        )
    };
    let mut loaded = LoadedImage::single(assembled);
    if runtime.model.has_animation() {
        loaded.puppet = Some(std::sync::Arc::new(runtime));
    }
    loaded
}

/// Same model -> material -> texture chain as `resolve_model_chain_{pkg,dir}`,
/// but for particle sprites: keeps every sprite-sheet frame (sliced from the
/// `.tex`'s TEXS table, if any) instead of collapsing to one flat image, plus
/// the animation's total loop duration so `ParticleSystem::set_sprite_frames`
/// can drive per-particle frame advance, and the material pass's own
/// `"blending"` string so the caller can composite additive-glow presets
/// (fog/smoke/embers/rain/lightning — the overwhelming majority of real
/// particle materials) correctly instead of always alpha-blending, which
/// makes a sprite's near-black background visibly darken the scene instead
/// of contributing nothing.
pub(super) fn resolve_particle_sprite_pkg(
    pkg: &Package,
    json_path: &str,
) -> Result<(particle::ParticleSprite, Option<String>)> {
    let (tex, blending, overbright) = find_model_chain_tex_pkg(pkg, json_path)?;
    let duration: f32 = tex.frames().iter().map(|f| f.frametime).sum();
    Ok((
        particle::ParticleSprite {
            frames: tex.to_particle_rgba_frames()?,
            duration,
            overbright,
        },
        blending,
    ))
}

pub(super) fn resolve_particle_sprite_dir(
    dir: &Path,
    json_path: &str,
) -> Result<(particle::ParticleSprite, Option<String>)> {
    let (tex, blending, overbright) = find_model_chain_tex_dir(dir, json_path)?;
    let duration: f32 = tex.frames().iter().map(|f| f.frametime).sum();
    Ok((
        particle::ParticleSprite {
            frames: tex.to_particle_rgba_frames()?,
            duration,
            overbright,
        },
        blending,
    ))
}

pub(super) fn guess_scene_dimensions(scene: &Scene, layers: &[Layer]) -> (u32, u32) {
    // Perspective 3D scenes have no orthogonal projection and their layer
    // sizes are world-space, not pixels — inferring pixel dimensions from
    // them yields garbage (e.g. 1619x29). Render at a fixed 16:9 target; the
    // perspective camera maps world → NDC independently of this resolution.
    if scene.is_perspective() {
        return (1920, 1080);
    }

    // 1. Try orthogonal projection dimensions from camera or general settings
    if let Some(cam) = &scene.camera {
        if let Some(proj) = &cam.orthogonal_projection {
            if let (Some(w), Some(h)) = (proj.width, proj.height) {
                if w > 0 && h > 0 {
                    return (w, h);
                }
            }
        }
    }
    if let Some(gen) = &scene.general {
        if let Some(proj) = &gen.orthogonal_projection {
            if let (Some(w), Some(h)) = (proj.width, proj.height) {
                if w > 0 && h > 0 {
                    return (w, h);
                }
            }
        }
    }

    // 2. Use the largest layer dimensions
    let mut max_w = 0u32;
    let mut max_h = 0u32;
    for layer in layers {
        max_w = max_w.max(layer.image.width());
        max_h = max_h.max(layer.image.height());
    }
    if max_w > 0 && max_h > 0 {
        return (max_w, max_h);
    }

    // 3. Fallback
    (1920, 1080)
}

/// For model/copybackground layers we can't render, create a transparent
/// placeholder so any screen-space effects applied to them have a base to work
/// with. It must be transparent, not opaque grey: these are usually full-screen
/// effect-only overlays (e.g. "Light shafts"), so an opaque base paints a giant
/// grey box over the scene whenever the effect is a SKIP or an additive overlay.
/// ponytail: transparent base; revisit if an effect needs to sample the real
/// scene behind it (backdrop capture — not implemented).
pub(super) fn placeholder_for(obj: &SceneObject) -> RgbaImage {
    let size = obj.parsed_size();
    let w = if size[0] > 0.0 { size[0] as u32 } else { 1920 };
    let h = if size[1] > 0.0 { size[1] as u32 } else { 1080 };
    RgbaImage::from_pixel(w, h, image::Rgba([0u8, 0, 0, 0]))
}

pub(super) fn is_special_layer(image_path: &str) -> bool {
    let p = image_path.to_lowercase();
    p.contains("projectlayer") || p.contains("composelayer") || p.contains("fullscreenlayer")
}

/// WE's solid-color model bundles (`models/util/solidlayer.json`,
/// `solid_instance_model_*.json`) reference a "flat" shader that ignores its
/// texture entirely and just outputs `vec4(g_Color, g_Alpha)` — i.e. the
/// object's own color/alpha, no image. A 1×1 white texture reproduces that
/// exactly once tinted by `Layer::color`/`alpha`/`brightness`.
pub(super) fn is_solid_layer_path(image_path: &str) -> bool {
    let p = image_path.to_lowercase();
    p.contains("solidlayer") || p.contains("solid_instance_model")
}

pub(super) fn white_pixel() -> RgbaImage {
    RgbaImage::from_pixel(1, 1, image::Rgba([255, 255, 255, 255]))
}

/// WE sizes a solid layer to the full scene when it has no explicit size
/// (`CImage.cpp:199`: `if (solidlayer && size == 0) size = sceneSize`) — but
/// only to give an effect chain a dummy target. A size-less solid layer with
/// no effects is a script/cursor holder ("Player Options" in 2988724897);
/// stretching its opaque white over the scene hides the whole wallpaper, so
/// callers only register those that carry effects.
/// Composition (`composelayer`), fullscreen (`fullscreenlayer`) and
/// projection (`projectlayer`) layers have no image of their own: their
/// texture is the scene already drawn beneath them — the region under the
/// layer for a composition layer, the whole scene otherwise — which their
/// effects then process (audio bars, fisheye, colour grading, blur…).
/// Built with a placeholder image; the live renderer captures the backdrop
/// each frame. Fullscreen and projection layers (and composition layers
/// without a size) are sized to the scene by `fill_solid_layer_sizes`.
pub(super) fn backdrop_layer_from_object(obj: &SceneObject, path: &str) -> Option<Layer> {
    let p = path.to_lowercase();
    let kind = if p.contains("composelayer") {
        Backdrop::Region
    } else if p.contains("fullscreenlayer") || p.contains("projectlayer") {
        Backdrop::FullScene
    } else {
        return None;
    };
    let mut layer = layer_from_object(
        obj,
        LoadedImage::single(RgbaImage::from_pixel(1, 1, image::Rgba([0, 0, 0, 0]))),
        None,
    );
    if kind == Backdrop::FullScene {
        layer.size = [0.0, 0.0, 0.0];
    }
    layer.backdrop = Some(kind);
    Some(layer)
}

pub(super) fn fill_solid_layer_sizes(
    layers: &mut [Layer],
    solid_indices: &[usize],
    width: u32,
    height: u32,
) {
    for &i in solid_indices {
        if layers[i].size[0] == 0.0 && layers[i].size[1] == 0.0 {
            layers[i].size = [width as f64, height as f64, 0.0];
        }
    }
}
