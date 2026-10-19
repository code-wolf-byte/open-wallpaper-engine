//! The scene while it runs: keyframed timelines, SceneScript ticks and the
//! property changes scripts make, cursor events and hit tests, live user
//! property edits, objects scripts create, text re-rasterization, and the
//! per-frame object transforms.

use super::*;

impl GpuSceneInstance {
    /// Advance keyframe animations and apply the properties they move.
    pub(super) fn tick_timelines(&mut self, delta: f32) {
        use crate::engine::sim::script::{ObjectChange, ScriptChanges};
        if self.timelines.is_empty() {
            return;
        }
        let mut changes: Vec<(usize, ObjectChange)> = Vec::new();
        let mut events: Vec<(usize, String)> = Vec::new();
        let mut synced: Vec<(usize, String, serde_json::Value)> = Vec::new();
        for t in &mut self.timelines {
            for name in t.anim.advance(delta) {
                events.push((t.object, name));
            }
            let mut v = t.anim.sample();
            if t.anim.relative {
                for (k, x) in v.iter_mut().enumerate() {
                    *x += t.base.get(k).copied().unwrap_or(0.0);
                }
            }
            if t.last.as_ref() == Some(&v) {
                continue;
            }
            t.last = Some(v.clone());
            let idx = match changes.iter().position(|(o, _)| *o == t.object) {
                Some(i) => i,
                None => {
                    changes.push((t.object, ObjectChange::default()));
                    changes.len() - 1
                }
            };
            let c = &mut changes[idx].1;
            let v3 = |fill: f32| -> [f32; 3] {
                std::array::from_fn(|k| {
                    v.get(k).copied().or(t.base.get(k).copied()).unwrap_or(fill)
                })
            };
            let json = if v.len() == 1 {
                serde_json::json!(v[0])
            } else {
                serde_json::json!(v)
            };
            match t.key.as_str() {
                "origin" => c.origin = Some(v3(0.0)),
                "scale" => c.scale = Some(v3(1.0)),
                "angles" => c.angles = Some(v3(0.0)),
                "color" => c.color = Some(v3(1.0)),
                "alpha" => c.alpha = v.first().copied(),
                "brightness" => c.brightness = v.first().copied(),
                key => {
                    c.extra.insert(key.to_string(), json.clone());
                }
            }
            synced.push((t.object, t.key.clone(), json));
        }
        if self.scripts.has_scripts() {
            for (object, key, value) in &synced {
                self.scripts.sync_property(*object, key, value);
            }
            for (object, name) in &events {
                self.scripts.dispatch_event(
                    *object,
                    "animationEvent",
                    &serde_json::json!({ "name": name }),
                );
            }
        }
        if !changes.is_empty() {
            self.apply_script_changes(ScriptChanges {
                objects: changes,
                commands: Vec::new(),
            });
        }
    }

    /// `thisLayer.getAnimation(name).play()` & co: control the animations on
    /// object `index` named `property` (by animation name or property key),
    /// along with the animations grouped with them.
    pub(super) fn control_timeline(
        &mut self,
        index: usize,
        property: &str,
        op: &str,
        frame: Option<f32>,
    ) {
        let matched: Vec<String> = self
            .timelines
            .iter()
            .filter(|t| t.object == index)
            .filter(|t| t.key == property || t.anim.name.as_deref() == Some(property))
            .flat_map(|t| std::iter::once(t.key.clone()).chain(t.anim.children.iter().cloned()))
            .collect();
        for t in self.timelines.iter_mut().filter(|t| t.object == index) {
            let grouped = matched.contains(&t.key)
                || t.anim.parent.as_ref().is_some_and(|p| matched.contains(p));
            if grouped {
                t.anim.control(op, frame);
            }
        }
    }

    /// Run this frame's SceneScripts and apply what they changed.
    pub(super) fn tick_scripts(&mut self, time: f32, delta: f32) {
        if !self.scripts.has_scripts() {
            return;
        }
        #[cfg(target_os = "linux")]
        if let Some(info) = self.media.as_ref().and_then(|w| w.try_recv()) {
            use crate::services::media::PlaybackState;
            self.scripts.set_media(
                info.available,
                match info.playback_state {
                    PlaybackState::Stopped => 0,
                    PlaybackState::Playing => 1,
                    PlaybackState::Paused => 2,
                },
                &info.title,
                &info.artist,
                &info.album,
                info.position_us,
                info.duration_us,
                info.art_url.as_deref(),
            );
        }
        if self.scripts_want_audio {
            if let Some(cap) = &self.audio_capture {
                let sp = cap.spectrum();
                self.scripts.set_audio([
                    &sp.s16_left,
                    &sp.s16_right,
                    &sp.s32_left,
                    &sp.s32_right,
                    &sp.s64_left,
                    &sp.s64_right,
                ]);
            }
        }
        self.scripts.set_cursor(
            self.cursor_scene,
            self.cursor_screen(),
            self.cursor_left_down,
        );
        self.dispatch_cursor_events();
        self.scripts
            .frame(time, crate::engine::sim::script::local_time_of_day(), delta);
        let changes = self.scripts.collect();
        self.apply_script_changes(changes);
        self.flush_text();

        // `registerAudioBuffers` usually runs at module load, so this is
        // known after the first frame; start capture then if nothing else did.
        if !self.scripts_want_audio && self.scripts.wants_audio() {
            self.scripts_want_audio = true;
            if self.audio_capture.is_none() {
                self.audio_capture = crate::services::audio::AudioCapture::start();
            }
        }
    }

    /// Fire `cursorEnter`/`Leave`/`Move`/`Down`/`Up`/`Click` on handler
    /// objects from this frame's pointer state.
    pub(super) fn dispatch_cursor_events(&mut self) {
        let buttons = std::mem::take(&mut self.pending_buttons);
        if self.cursor_targets.is_empty() {
            return;
        }
        let pos = self.cursor_scene;
        let hit = self.hit_test(self.mouse_norm);
        if hit != self.hovered {
            if let Some(old) = self.hovered {
                self.scripts.dispatch_cursor(old, "cursorLeave", pos);
            }
            if let Some(new) = hit {
                self.scripts.dispatch_cursor(new, "cursorEnter", pos);
            }
            self.hovered = hit;
        }
        if pos != self.last_cursor_scene {
            self.last_cursor_scene = pos;
            if let Some(h) = hit {
                self.scripts.dispatch_cursor(h, "cursorMove", pos);
            }
            // A drag keeps reporting to the layer it started on.
            if let Some(p) = self.pressed.filter(|p| Some(*p) != hit) {
                self.scripts.dispatch_cursor(p, "cursorMove", pos);
            }
        }
        for down in buttons {
            if down {
                self.pressed = hit;
                if let Some(h) = hit {
                    self.scripts.dispatch_cursor(h, "cursorDown", pos);
                }
            } else if let Some(p) = self.pressed.take() {
                self.scripts.dispatch_cursor(p, "cursorUp", pos);
                if hit == Some(p) {
                    self.scripts.dispatch_cursor(p, "cursorClick", pos);
                }
            }
        }
    }

    /// The topmost cursor-handler object whose quad contains the cursor
    /// (`norm`: `[0,1]²`, top-left origin). Hidden layers still count:
    /// real content uses invisible hitboxes that toggle themselves.
    pub(super) fn hit_test(&self, norm: [f32; 2]) -> Option<usize> {
        let (w, h) = (self.width as f32, self.height as f32);
        // Output pixels, Y-up, origin at the centre — the space `rect` is in.
        let p = [(norm[0] - 0.5) * w, (0.5 - norm[1]) * h];
        self.layers
            .iter()
            .filter(|l| self.cursor_targets.contains(&l.order_index))
            .filter(|l| {
                let corners: [[f32; 2]; 4] = match l.quad_corners {
                    Some(c) => c.map(|c| [c[0] * w * 0.5, c[1] * h * 0.5]),
                    None => quad_corners_px(l.rect, l.angle, w, h),
                };
                point_in_convex_quad(p, &corners)
            })
            .map(|l| l.order_index)
            .max()
    }

    /// Record a left-button press or release (from the platform layer).
    pub fn set_mouse_button(&mut self, left_down: bool) {
        if left_down != self.cursor_left_down {
            self.cursor_left_down = left_down;
            self.pending_buttons.push(left_down);
        }
    }

    /// The cursor in output pixels, top-left origin (`input.cursorScreenPosition`).
    pub(super) fn cursor_screen(&self) -> [f32; 2] {
        [
            self.mouse_norm[0] * self.width as f32,
            self.mouse_norm[1] * self.height as f32,
        ]
    }

    /// Apply a frame's script changes: transforms recompose through the
    /// parent graph, everything else lands on the object's layers and sounds.
    pub(super) fn apply_script_changes(
        &mut self,
        changes: crate::engine::sim::script::ScriptChanges,
    ) {
        use crate::engine::sim::script::ScriptCommand;
        let n = self.current_locals.len();
        let mut moved = vec![false; n];
        let mut visibility_changed = false;
        for (i, c) in &changes.objects {
            let i = *i;
            if i >= n {
                continue;
            }
            let x = &mut self.current_locals[i];
            if let Some(o) = c.origin {
                x.t = o.map(|v| v as f64);
                moved[i] = true;
            }
            if let Some(a) = c.angles {
                x.r = a.map(|v| v as f64);
                moved[i] = true;
            }
            if let Some(sc) = c.scale {
                x.s = sc.map(|v| v as f64);
                moved[i] = true;
            }
            if let Some(d) = c.propagation_disabled {
                self.propagation_disabled[i] = d;
                moved[i] = true;
            }
            if let Some(v) = c.visible {
                self.object_visible[i] = v;
                visibility_changed = true;
            }
            if let Some(v) = c.volume {
                for snd in self.sounds_of(i) {
                    snd.set_volume(v);
                }
            }
            for (key, value) in &c.extra {
                self.apply_script_extra(i, key, value);
            }
            for li in 0..self.layers.len() {
                if self.layers[li].order_index != i {
                    continue;
                }
                let layer = &mut self.layers[li];
                if let Some(v) = c.visible {
                    layer.visible = v;
                }
                if let Some(a) = c.alpha {
                    layer.alpha = a;
                }
                if let Some(col) = c.color {
                    layer.color = col;
                }
                if let Some(b) = c.brightness {
                    layer.brightness = b;
                }
                if let Some(f) = c.frame {
                    layer.script_frame = Some(f as usize);
                }
                if let Some(ps) = c.pointsize {
                    if let Some(td) = &mut layer.text_dynamic {
                        td.point_size = ps.max(1.0);
                        td.last_text.clear();
                    }
                }
                if c.text.is_some() || c.pointsize.is_some() {
                    self.rerasterize_text(li, c.text.as_deref());
                }
            }
        }
        for cmd in changes.commands {
            match cmd {
                ScriptCommand::Sound { index, op } => {
                    for snd in self.sounds_of(index) {
                        match op.as_str() {
                            "play" => snd.play(),
                            "pause" => snd.pause(),
                            "stop" => snd.stop(),
                            _ => {}
                        }
                    }
                }
                ScriptCommand::Sort { index, to } => self.sort_object(index, to),
                ScriptCommand::Animation {
                    index,
                    property,
                    op,
                    frame,
                } => self.control_timeline(index, &property, &op, frame),
                ScriptCommand::Create { index, config } => {
                    self.create_object(index, &config);
                    visibility_changed = true;
                }
                ScriptCommand::Destroy { index } => {
                    // Hidden for good rather than removed: per-layer render
                    // targets are keyed by position in `self.layers`.
                    if let Some(v) = self.object_visible.get_mut(index) {
                        *v = false;
                    }
                    self.cursor_targets.retain(|t| *t != index);
                    visibility_changed = true;
                }
                other => {
                    tracing::debug!(target: "script", "unhandled script command {other:?}");
                }
            }
        }
        if visibility_changed {
            self.refresh_layer_visibility();
        }
        if moved.iter().any(|m| *m) {
            self.apply_object_transforms(&moved);
        }
    }

    /// The user properties `names` changed in the saved settings: show them
    /// live. Returns `false` when the scene must be rebuilt instead (see
    /// [`Self::apply_user_properties`]).
    pub fn set_user_properties(&mut self, names: &[String]) -> bool {
        let props =
            crate::workshop::properties::SceneProperties::from_project_dir(&self.live_props.dir);
        self.apply_user_properties(props, names)
    }

    /// Re-resolve the scene against `props` and push each object's
    /// differences through the same path SceneScript writes take; scripts
    /// get `applyUserProperties` for `names`. Returns `false` when a change
    /// can't be applied live (scene-wide settings, an effect that wasn't
    /// built at load, a field scripts can't set).
    pub(super) fn apply_user_properties(
        &mut self,
        props: crate::workshop::properties::SceneProperties,
        names: &[String],
    ) -> bool {
        let lp = &mut self.live_props;
        lp.props = props;
        let mut next = lp.raw.clone();
        lp.props.resolve_scene_json(&mut next);
        let prev = std::mem::replace(&mut lp.resolved, next);
        if prev.get("general") != lp.resolved.get("general") {
            return false;
        }
        let objects = |v: &serde_json::Value| {
            v.get("objects")
                .and_then(|o| o.as_array())
                .cloned()
                .unwrap_or_default()
        };
        let (before, after) = (objects(&prev), objects(&lp.resolved));
        if before.len() != after.len() {
            return false;
        }
        let user: serde_json::Map<String, serde_json::Value> = names
            .iter()
            .filter_map(|n| {
                Some((
                    n.clone(),
                    serde_json::json!({ "value": lp.props.get(n)?.clone() }),
                ))
            })
            .collect();
        let mut objects = Vec::new();
        for (i, (a, b)) in before.iter().zip(&after).enumerate() {
            let mut diffs = Vec::new();
            changed_leaves(a, b, &mut Vec::new(), &mut diffs);
            if diffs.is_empty() {
                continue;
            }
            let mut change = crate::engine::sim::script::ObjectChange::default();
            for (path, v) in &diffs {
                if self.live_change(i, path, v, &mut change).is_none() {
                    tracing::debug!(target: "properties", "object {i}: {} needs a reload", path.join("."));
                    return false;
                }
            }
            objects.push((i, change));
        }
        self.apply_script_changes(crate::engine::sim::script::ScriptChanges {
            objects,
            commands: Vec::new(),
        });
        self.scripts
            .apply_user_properties(&serde_json::Value::Object(user));
        true
    }

    /// One changed scene.json field of object `object`, as a script change.
    /// `None` when it has no live counterpart.
    pub(super) fn live_change(
        &self,
        object: usize,
        path: &[String],
        v: &serde_json::Value,
        c: &mut crate::engine::sim::script::ObjectChange,
    ) -> Option<()> {
        use crate::engine::model::scene::{parse_value_bool, parse_value_f32, parse_value_vec3};
        // `{"user": …, "value": X}` bindings resolve to their `value`.
        let v = v.get("value").unwrap_or(v);
        let vec3 = |v| parse_value_vec3(v).map(|a| a.map(|x| x as f32));
        let path: Vec<&str> = path.iter().map(String::as_str).collect();
        match path.as_slice() {
            ["visible"] => c.visible = Some(parse_value_bool(v)?),
            ["alpha"] => c.alpha = Some(parse_value_f32(v)?),
            ["brightness"] => c.brightness = Some(parse_value_f32(v)?),
            ["color"] => c.color = Some(vec3(v)?),
            ["origin"] => c.origin = Some(vec3(v)?),
            ["scale"] => c.scale = Some(vec3(v)?),
            ["volume"] => c.volume = Some(parse_value_f32(v)?),
            ["pointsize"] => c.pointsize = Some(parse_value_f32(v)?),
            ["text"] => c.text = Some(v.as_str()?.to_string()),
            ["instanceoverride", k] => {
                c.extra.insert(format!("instance.{k}"), v.clone());
            }
            ["effects", e, rest @ ..] => {
                let e: usize = e.parse().ok()?;
                // Effects hidden at load were never built.
                self.scene_effects
                    .iter()
                    .any(|d| d.layer_idx == object && d.effect_index == e)
                    .then_some(())?;
                match rest {
                    ["visible"] => {
                        c.extra
                            .insert(format!("effects.{e}.visible"), parse_value_bool(v)?.into());
                    }
                    ["passes", pass, "constantshadervalues", k] => {
                        c.extra.insert(format!("effects.{e}.{pass}.{k}"), v.clone());
                    }
                    _ => return None,
                }
            }
            _ => return None,
        }
        Some(())
    }

    /// A script wrote a declared extra property: `instance.<key>` (particle
    /// override), `effects.<e>.visible`, or `effects.<e>.<pass>.<constant>`.
    pub(super) fn apply_script_extra(
        &mut self,
        object: usize,
        key: &str,
        value: &serde_json::Value,
    ) {
        if let Some(k) = key.strip_prefix("instance.") {
            for (sys, _) in self
                .particle_systems
                .iter_mut()
                .zip(&self.particle_order)
                .filter(|(_, o)| **o == object)
            {
                sys.set_instance_value(k, value);
            }
            return;
        }
        let Some(rest) = key.strip_prefix("effects.") else {
            return;
        };
        let mut parts = rest.splitn(3, '.');
        let Some(e) = parts.next().and_then(|e| e.parse::<usize>().ok()) else {
            return;
        };
        let Some(k) = self
            .scene_effects
            .iter()
            .position(|d| d.layer_idx == object && d.effect_index == e)
        else {
            return;
        };
        match (parts.next(), parts.next()) {
            (Some("visible"), None) => {
                self.effect_enabled[k] = value.as_bool().unwrap_or(true);
            }
            (Some(pass), Some(name)) => {
                let Some(pass) = pass.parse::<usize>().ok() else {
                    return;
                };
                let Some(runtime) = self.effect_runtimes.get_mut(k).and_then(|r| r.as_mut()) else {
                    return;
                };
                if let Some(p) = runtime.passes.get_mut(pass) {
                    // Vectors come back as arrays; the value parser wants "x y z".
                    let v = match value {
                        serde_json::Value::Array(a) => serde_json::Value::String(
                            a.iter()
                                .map(|x| x.as_f64().unwrap_or(0.0).to_string())
                                .collect::<Vec<_>>()
                                .join(" "),
                        ),
                        other => other.clone(),
                    };
                    insert_value_components(&mut p.values, name, &v);
                }
            }
            _ => {}
        }
    }

    /// `thisScene.sortLayer(layer, index)`: draw `index`'s object just below
    /// the object currently at `to`, later sorts landing lower.
    pub(super) fn sort_object(&mut self, index: usize, to: usize) {
        let Some(base) = self.draw_rank.get(to).copied() else {
            return;
        };
        self.sort_count += 1;
        if let Some(r) = self.draw_rank.get_mut(index) {
            *r = base - 0.5 + 0.49 / self.sort_count as f64;
        }
    }

    /// `thisScene.createLayer(config)`: build object `index` from a
    /// scene.json-shaped config. Image, solid and text objects only; their
    /// effects are not instantiated.
    pub(super) fn create_object(&mut self, index: usize, config: &serde_json::Value) {
        use crate::engine::model::resolve::Xform;
        // Keep every per-object table indexed in step with the script side,
        // even when this object can't be built.
        let n = index + 1;
        if self.current_locals.len() < n {
            self.current_locals.resize(n, Xform::IDENTITY);
            self.transform_graph.parent.resize(n, None);
            self.transform_graph.local.resize(n, Xform::IDENTITY);
            self.transform_graph.name.resize(n, None);
            self.transform_graph.id.resize(n, None);
            self.object_visible.resize(n, true);
            self.propagation_disabled.resize(n, false);
            let len = self.draw_rank.len();
            self.draw_rank.extend((len..n).map(|i| i as f64));
        }
        let obj: crate::engine::model::scene::SceneObject =
            match serde_json::from_value(config.clone()) {
                Ok(o) => o,
                Err(e) => {
                    tracing::warn!(target: "script", "createLayer: bad config: {e}");
                    return;
                }
            };
        if obj.sound.is_some() {
            if let Some(playback) = start_scene_sound(&self.scene_dir, &obj) {
                self.sounds.push(SceneSound {
                    object: index,
                    playback,
                });
            }
            return;
        }
        if self.scene_pkg.is_none() {
            self.scene_pkg =
                crate::formats::pkg::Package::from_file(&self.scene_dir.join("scene.pkg")).ok();
        }
        let Some(mut layer) = crate::engine::model::resolve::runtime_layer(
            &self.scene_dir,
            self.scene_pkg.as_ref(),
            &obj,
        ) else {
            tracing::debug!(target: "script", "createLayer: object {index} has nothing to draw");
            return;
        };
        layer.order_index = index;
        let local = crate::engine::model::resolve::obj_local_xform(&obj);
        self.current_locals[index] = local;
        self.transform_graph.local[index] = local;
        self.transform_graph.name[index] = obj.name.clone();
        self.object_visible[index] = obj.is_visible();
        let gpu = gpu_layer_from(
            &self.renderer,
            &layer,
            self.width,
            self.height,
            self.zoom,
            self.camera3d.as_ref(),
            local.t,
        );
        let (ow, oh) = (gpu.object_size.0.max(1), gpu.object_size.1.max(1));
        let li = self.layers.len();
        let device = self.renderer.device().clone();
        self.fbo_pool
            .get_or_create(&device, &pingpong_key(li, 0), ow, oh);
        self.fbo_pool
            .get_or_create(&device, &pingpong_key(li, 1), ow, oh);
        self.layers.push(gpu);
    }

    /// Sound controls belonging to scene object `index`.
    pub(super) fn sounds_of(
        &self,
        index: usize,
    ) -> Vec<std::sync::Arc<crate::services::audio::SoundControl>> {
        self.sounds
            .iter()
            .filter(|s| s.object == index)
            .map(|s| s.playback.control())
            .collect()
    }

    /// Queue new text (or, with `None`, a re-raster at a new point size) for
    /// a text layer; [`Self::flush_text`] rasterizes it.
    pub(super) fn rerasterize_text(&mut self, li: usize, text: Option<&str>) {
        let layer = &mut self.layers[li];
        let Some(td) = &layer.text_dynamic else {
            return;
        };
        let new_text = text
            .map(str::to_string)
            .or_else(|| layer.pending_text.clone())
            .unwrap_or_else(|| td.last_text.clone());
        if new_text.is_empty() || (text.is_some() && new_text == td.last_text) {
            layer.pending_text = None;
            return;
        }
        layer.pending_text = Some(new_text);
    }

    /// Rasterize pending text. Throttled per layer to every 100 ms, or every
    /// 10× the last raster's cost when that is longer, so a script that
    /// rewrites a huge text block every frame (debug overlays, timers with
    /// milliseconds) can't stall rendering; the newest text always lands.
    pub(super) fn flush_text(&mut self) {
        let now = Instant::now();
        for li in 0..self.layers.len() {
            let layer = &mut self.layers[li];
            let (Some(text), Some(td)) = (&layer.pending_text, &mut layer.text_dynamic) else {
                continue;
            };
            let wait = (layer.text_raster_cost * 10).max(Duration::from_millis(100));
            if layer
                .text_rastered_at
                .is_some_and(|t| now.duration_since(t) < wait)
            {
                continue;
            }
            let started = Instant::now();
            if let Some(img) =
                crate::engine::gpu::text::rasterize(&td.font_data, text, td.point_size)
            {
                // The quad keeps its authored box; only the texture changes.
                layer.frames[0] = self.renderer.upload_texture(&img);
                td.last_text = text.clone();
            }
            layer.text_raster_cost = started.elapsed();
            layer.text_rastered_at = Some(now);
            layer.pending_text = None;
        }
    }

    /// A layer draws only when its own object and every ancestor are visible.
    pub(super) fn refresh_layer_visibility(&mut self) {
        let parent = &self.transform_graph.parent;
        let visible = &self.object_visible;
        let effective = |mut i: usize| -> bool {
            for _ in 0..64 {
                if !visible.get(i).copied().unwrap_or(true) {
                    return false;
                }
                match parent.get(i).copied().flatten() {
                    Some(p) if p != i => i = p,
                    _ => return true,
                }
            }
            true
        };
        for layer in &mut self.layers {
            layer.visible = effective(layer.order_index);
        }
    }

    /// Recompose world transforms from `current_locals` and push them into
    /// the layers and meshes they move: those whose own object or any
    /// ancestor `moved`.
    pub(super) fn apply_object_transforms(&mut self, moved: &[bool]) {
        use crate::engine::model::resolve::Xform;
        let n = self.current_locals.len();
        let parent = self.transform_graph.parent.clone();
        // affected[i]: object i or an ancestor moved.
        let mut affected = vec![false; n];
        for i in 0..n {
            let mut cur = i;
            for _ in 0..64 {
                if moved.get(cur).copied().unwrap_or(false) {
                    affected[i] = true;
                    break;
                }
                match parent.get(cur).copied().flatten() {
                    Some(p) if p != cur => cur = p,
                    _ => break,
                }
            }
        }
        let locals = self.current_locals.clone();
        let worlds = self
            .transform_graph
            .world(&locals, &self.propagation_disabled);
        // The PARENT's world transform, not the object's own — `worlds[i]`
        // already folds in object i's local transform.
        let parent_world = |i: usize| -> Xform {
            if self.propagation_disabled.get(i).copied().unwrap_or(false) {
                return Xform::IDENTITY;
            }
            match parent.get(i).copied().flatten() {
                Some(p) if p != i => worlds[p],
                _ => Xform::IDENTITY,
            }
        };

        // 3D meshes: rebuild the MVP from the recomposed world transform.
        // Computed up front so the write-back can borrow `self.mesh3d` mutably.
        let mesh_updates: Vec<(usize, Vec<u8>, f32)> = match &self.camera3d {
            Some(cam) => self
                .mesh3d
                .iter()
                .filter(|m| affected.get(m.order_index).copied().unwrap_or(false))
                .map(|m| {
                    let w = parent_world(m.order_index);
                    let (lo, la, ls) = match locals.get(m.order_index) {
                        Some(x) => (
                            [x.t[0] as f32, x.t[1] as f32, x.t[2] as f32],
                            [x.r[0] as f32, x.r[1] as f32, x.r[2] as f32],
                            [x.s[0] as f32, x.s[1] as f32, x.s[2] as f32],
                        ),
                        None => m.local,
                    };
                    let o = w.apply_point([lo[0] as f64, lo[1] as f64, lo[2] as f64]);
                    let centre = [o[0] as f32, o[1] as f32, o[2] as f32];
                    let angles: [f32; 3] = std::array::from_fn(|i| la[i] + w.r[i] as f32);
                    let scale: [f32; 3] = std::array::from_fn(|i| ls[i] * w.s[i] as f32);
                    let bytes = mesh3d_transform_bytes(cam, centre, angles, scale);
                    (m.order_index, bytes, cam.view_depth(centre))
                })
                .collect(),
            None => Vec::new(),
        };
        for (idx, bytes, depth) in mesh_updates {
            let Some(m) = self.mesh3d.iter_mut().find(|m| m.order_index == idx) else {
                continue;
            };
            self.renderer.queue.write_buffer(&m.ubo, 0, &bytes);
            m.depth = depth;
        }

        // 2D layers, both projections. The rect derivations differ (the
        // perspective one projects the quad's corners through the camera) but
        // both start from the same recomposed world transform.
        let (w_px, h_px) = (self.width as f64, self.height as f64);
        for layer in &mut self.layers {
            let i = layer.order_index;
            if !affected.get(i).copied().unwrap_or(false) {
                continue;
            }
            let w = parent_world(i);
            let local = locals[i];
            let o = w.apply_point(std::array::from_fn(|k| local.t[k] + layer.align_offset[k]));
            let size_px = [
                layer.effective_size[0] * local.s[0] * w.s[0],
                layer.effective_size[1] * local.s[1] * w.s[1],
            ];
            let local_angles: [f32; 3] = std::array::from_fn(|k| local.r[k] as f32);
            match &self.camera3d {
                Some(cam) => {
                    let angles: [f32; 3] = std::array::from_fn(|k| local_angles[k] + w.r[k] as f32);
                    let centre = [o[0] as f32, o[1] as f32, o[2] as f32];
                    let (rect, depth) = project_quad_ndc(cam, centre, angles, size_px);
                    layer.rect = rect;
                    layer.quad_corners = project_quad_corners(cam, centre, angles, size_px);
                    // Feeds the painter's-algorithm sort in `render()`; a
                    // negative depth culls the quad, same as at build.
                    layer.depth = depth;
                }
                None => {
                    // Same `general.zoom` scene-wide scale the build path uses.
                    let z = self.zoom;
                    layer.rect = [
                        (2.0 * o[0] / w_px - 1.0) as f32 * z,
                        (2.0 * o[1] / h_px - 1.0) as f32 * z,
                        (size_px[0] / w_px) as f32 * z,
                        (size_px[1] / h_px) as f32 * z,
                    ];
                    layer.angle = -(local_angles[2] + w.r[2] as f32);
                }
            }
        }
    }

    /// Feed pointer position in [0,1]² (drives camera parallax).
    pub fn set_mouse(&mut self, norm: [f32; 2]) {
        self.mouse_norm = norm;
        // Scene space is Y-up with the origin bottom-left.
        self.cursor_scene = [
            norm[0] * self.width as f32,
            (1.0 - norm[1]) * self.height as f32,
        ];
    }
}

impl LiveProps {
    pub(super) fn new(dir: &Path, raw: &serde_json::Value) -> Self {
        let props = crate::workshop::properties::SceneProperties::from_project_dir(dir);
        let mut resolved = raw.clone();
        props.resolve_scene_json(&mut resolved);
        Self {
            dir: dir.to_path_buf(),
            raw: raw.clone(),
            props,
            resolved,
        }
    }
}

/// Collect the leaves that differ between `a` and `b` as (path, new value).
/// A `{"user": …}` binding counts as one leaf, as do arrays that changed
/// length.
pub(super) fn changed_leaves(
    a: &serde_json::Value,
    b: &serde_json::Value,
    path: &mut Vec<String>,
    out: &mut Vec<(Vec<String>, serde_json::Value)>,
) {
    use serde_json::Value;
    if a == b {
        return;
    }
    match (a, b) {
        (Value::Object(x), Value::Object(y)) if !y.contains_key("user") => {
            for k in x.keys().chain(y.keys().filter(|k| !x.contains_key(*k))) {
                path.push(k.clone());
                changed_leaves(
                    x.get(k).unwrap_or(&Value::Null),
                    y.get(k).unwrap_or(&Value::Null),
                    path,
                    out,
                );
                path.pop();
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                path.push(i.to_string());
                changed_leaves(p, q, path, out);
                path.pop();
            }
        }
        _ => out.push((path.clone(), b.clone())),
    }
}

/// A layer quad's corners in output pixels (Y-up, centred), from its NDC
/// `rect` (centre, half-extent) and z-rotation.
pub(super) fn quad_corners_px(rect: [f32; 4], angle: f32, w: f32, h: f32) -> [[f32; 2]; 4] {
    let (cx, cy) = (rect[0] * w * 0.5, rect[1] * h * 0.5);
    let (hx, hy) = (rect[2] * w * 0.5, rect[3] * h * 0.5);
    let (s, c) = angle.sin_cos();
    [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]
        .map(|[x, y]| [cx + x * c - y * s, cy + x * s + y * c])
}

/// Whether `p` lies inside the convex quad `q` (either winding).
pub(super) fn point_in_convex_quad(p: [f32; 2], q: &[[f32; 2]; 4]) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let a = q[i];
        let b = q[(i + 1) % 4];
        let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        if cross.abs() < f32::EPSILON {
            continue;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if cross.signum() != sign {
            return false;
        }
    }
    sign != 0.0
}

/// Collect every animated property from the raw scene.json objects.
pub(super) fn build_timelines(
    raw_scene: &serde_json::Value,
    graph: &crate::engine::model::resolve::TransformGraph,
) -> Vec<TimelineEntry> {
    use crate::engine::sim::timeline::PropertyAnimation;
    let mut out = Vec::new();
    let objects = raw_scene.get("objects").and_then(|o| o.as_array());
    for (i, raw) in objects.into_iter().flatten().enumerate() {
        let mut props: Vec<(String, &serde_json::Value)> =
            ["origin", "scale", "angles", "alpha", "color", "brightness"]
                .iter()
                .filter_map(|k| raw.get(*k).map(|v| (k.to_string(), v)))
                .collect();
        if let Some(m) = raw.get("instanceoverride").and_then(|v| v.as_object()) {
            props.extend(m.iter().map(|(k, v)| (format!("instance.{k}"), v)));
        }
        for (e, eff) in raw
            .get("effects")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .enumerate()
        {
            for (p, pass) in eff
                .get("passes")
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
                .enumerate()
            {
                if let Some(m) = pass.get("constantshadervalues").and_then(|v| v.as_object()) {
                    props.extend(m.iter().map(|(k, v)| (format!("effects.{e}.{p}.{k}"), v)));
                }
            }
        }
        for (key, value) in props {
            let Some(anim) = value
                .get("animation")
                .and_then(|a| PropertyAnimation::parse(&key, a))
            else {
                continue;
            };
            let local = graph
                .local
                .get(i)
                .copied()
                .unwrap_or(crate::engine::model::resolve::Xform::IDENTITY);
            let base: Vec<f32> = match key.as_str() {
                "origin" => local.t.iter().map(|v| *v as f32).collect(),
                "scale" => local.s.iter().map(|v| *v as f32).collect(),
                "angles" => local.r.iter().map(|v| *v as f32).collect(),
                _ => Vec::new(),
            };
            out.push(TimelineEntry {
                object: i,
                key,
                base,
                anim,
                last: None,
            });
        }
    }
    out
}

/// Build the scene's SceneScript runtime: one layer proxy per scene object
/// (indexed like `scene.objects`, the same index `order_index` and the
/// transform graph use), then one instance per scripted property.
pub(super) fn build_script_host(
    dir: &Path,
    scene: &crate::engine::model::scene::Scene,
    graph: &crate::engine::model::resolve::TransformGraph,
    canvas: [u32; 2],
    raw_scene: &serde_json::Value,
) -> crate::engine::sim::script::ScriptHost {
    use crate::engine::model::scene::{parse_value_f32, parse_value_vec3};
    use crate::engine::sim::script::{ObjectInit, ScriptHost};

    let mut host = ScriptHost::new();
    if scene.objects.iter().all(|o| o.scripts().is_empty()) {
        return host;
    }
    if let Some(path) = ScriptHost::storage_path_for(dir) {
        host = host.with_storage(path);
    }
    // The raw scene.json entries back `thisScene.getInitialLayerConfig`.
    let raw_objects: Vec<serde_json::Value> = raw_scene
        .get("objects")
        .and_then(|o| o.as_array())
        .cloned()
        .unwrap_or_default();
    if let Some(general) = raw_scene.get("general") {
        host.set_scene_general(general);
    }

    let user_props: serde_json::Map<String, serde_json::Value> =
        crate::workshop::properties::SceneProperties::from_project_dir(dir)
            .iter()
            .map(|p| (p.name.clone(), p.value.clone()))
            .collect();
    host.set_scene_info(
        [canvas[0] as f32, canvas[1] as f32],
        canvas,
        &serde_json::Value::Object(user_props),
    );

    let f32v = |v: [f64; 3]| [v[0] as f32, v[1] as f32, v[2] as f32];
    for (i, obj) in scene.objects.iter().enumerate() {
        let local = graph
            .local
            .get(i)
            .copied()
            .unwrap_or(crate::engine::model::resolve::Xform::IDENTITY);
        let text = obj.text.as_ref().and_then(|t| match t {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Object(m) => {
                m.get("value").and_then(|v| v.as_str()).map(str::to_string)
            }
            _ => None,
        });
        let init = ObjectInit {
            name: obj.name.clone(),
            id: obj.id,
            origin: f32v(local.t),
            angles: f32v(local.r),
            scale: f32v(local.s),
            size: f32v(obj.parsed_size()),
            color: obj
                .color
                .as_ref()
                .and_then(parse_value_vec3)
                .map(f32v)
                .unwrap_or([1.0; 3]),
            visible: obj.is_visible(),
            alpha: obj.alpha.as_ref().and_then(parse_value_f32).unwrap_or(1.0),
            brightness: obj
                .brightness
                .as_ref()
                .and_then(parse_value_f32)
                .unwrap_or(1.0),
            text,
            pointsize: obj
                .pointsize
                .as_ref()
                .and_then(parse_value_f32)
                .unwrap_or(32.0),
            volume: obj.volume.as_ref().and_then(parse_value_f32).unwrap_or(1.0),
            parallax_depth: obj
                .parallax_depth
                .as_ref()
                .and_then(parse_value_vec3)
                .map(|v| [v[0] as f32, v[1] as f32])
                .unwrap_or([0.0; 2]),
            parent_index: graph.parent.get(i).copied().flatten(),
            frame_count: 1,
            raw: raw_objects.get(i).cloned(),
        };
        host.register_object(i, &init);
    }

    for (i, obj) in scene.objects.iter().enumerate() {
        let props: [(&str, Option<&serde_json::Value>); 10] = [
            ("visible", obj.visible.as_ref()),
            ("origin", obj.origin.as_ref()),
            ("angles", obj.angles.as_ref()),
            ("scale", obj.scale.as_ref()),
            ("alpha", obj.alpha.as_ref()),
            ("color", obj.color.as_ref()),
            ("brightness", obj.brightness.as_ref()),
            ("text", obj.text.as_ref()),
            ("pointsize", obj.pointsize.as_ref()),
            ("volume", obj.volume.as_ref()),
        ];
        for (prop, value) in props {
            if let Some((script, authored)) = crate::engine::sim::script::scripted(value) {
                host.attach(i, Some(prop), script, authored);
            }
        }
        // Scripted particle overrides and effect constants/visibility, read
        // from the raw entry (the typed model drops their scripts).
        let Some(raw) = raw_objects.get(i) else {
            continue;
        };
        let mut extras: Vec<(String, &serde_json::Value)> = Vec::new();
        if let Some(m) = raw.get("instanceoverride").and_then(|v| v.as_object()) {
            extras.extend(m.iter().map(|(k, v)| (format!("instance.{k}"), v)));
        }
        for (e, eff) in raw
            .get("effects")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .enumerate()
        {
            if let Some(v) = eff.get("visible") {
                extras.push((format!("effects.{e}.visible"), v));
            }
            for (p, pass) in eff
                .get("passes")
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
                .enumerate()
            {
                if let Some(m) = pass.get("constantshadervalues").and_then(|v| v.as_object()) {
                    extras.extend(m.iter().map(|(k, v)| (format!("effects.{e}.{p}.{k}"), v)));
                }
            }
        }
        for (key, value) in extras {
            if let Some((script, authored)) = crate::engine::sim::script::scripted(Some(value)) {
                let initial = value
                    .get("value")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                host.declare_property(
                    i,
                    &key,
                    crate::engine::sim::script::value_kind(&initial),
                    &initial,
                );
                host.attach(i, Some(&key), script, authored);
            }
        }
    }
    host
}
