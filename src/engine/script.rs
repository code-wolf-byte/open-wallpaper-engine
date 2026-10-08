//! JavaScript evaluation for SceneScript-driven properties, backed by the
//! [`boa_engine`] interpreter.
//!
//! Wallpaper Engine lets any animated property carry a small script with a
//! `function update(value) { ... return value; }` that runs each frame. We
//! evaluate it in a real JS engine so full expressions, locals, conditionals,
//! loops and `Math.*` all work — not just the arithmetic the previous
//! hand-rolled parser handled. The exposed API mirrors what those scripts
//! reference:
//!
//! - `value` — the property's current value (also passed as the `update` arg)
//! - `engine.runtime` — seconds since the wallpaper started
//! - `engine.timeOfDay` — fractional hours in `[0, 24)`
//! - `Math.*` — the standard library (native to boa)
//! - `WEMath.smoothStep(a, b, x)` — WE's smoothstep helper
//!
//! NOTE: each call spins up a fresh [`Context`]. That is fine for evaluating
//! independent property scripts, but a per-frame render loop wants a persistent
//! context (globals, the `update` closure, and the realm reused across frames);
//! that is the next milestone and will live behind a `ScriptContext` type.

use boa_engine::{Context, Source};

pub struct ScriptEnv {
    pub runtime: f32,
    pub time_of_day: f32,
}

/// Format an `f32` as a JS numeric literal, mapping non-finite values to `0`
/// so a stray `inf`/`NaN` can't inject invalid tokens into the source.
fn js_num(v: f32) -> String {
    if v.is_finite() {
        format!("{v:?}")
    } else {
        "0".to_string()
    }
}

/// Format a `&str` as a JS string literal (proper escaping via
/// `serde_json`'s compatible-with-JS escaping rules — same trick
/// [`ScriptContext::eval_update_string`] already relies on).
fn js_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string())
}

/// One scene object's current name/id/transform, as fed to
/// [`ScriptContext::set_layers`] for `thisScene.getLayer(...)` to search.
/// `name`/`id` of `None` never match a script's `getLayer` lookup (mirrors
/// how a nameless/idless object can't be looked up in real WE either).
pub struct LayerSnapshot<'a> {
    pub name: Option<&'a str>,
    pub id: Option<i64>,
    pub origin: [f32; 3],
    pub angles: [f32; 3],
    pub scale: [f32; 3],
}

/// The WE scripting globals, declared ahead of the user script.
fn prelude(current_value: f32, env: &ScriptEnv) -> String {
    format!(
        "var value = {v};\n\
         var engine = {{ runtime: {r}, timeOfDay: {t} }};\n\
         var WEMath = {{ smoothStep: function(a, b, x) {{ \
             var t = Math.min(Math.max((x - a) / (b - a), 0), 1); \
             return t * t * (3 - 2 * t); \
         }} }};\n",
        v = js_num(current_value),
        r = js_num(env.runtime),
        t = js_num(env.time_of_day),
    )
}

/// Run a SceneScript `update(value)` and return its numeric result, or `None`
/// if the script has no `update`, throws, or doesn't yield a finite number —
/// callers then keep the property's current value.
pub fn eval_update(script: &str, current_value: f32, env: &ScriptEnv) -> Option<f32> {
    let source = format!(
        "{prelude}\n{script}\n; (typeof update === 'function') ? update(value) : value;",
        prelude = prelude(current_value, env),
    );

    let mut context = Context::default();
    let result = context.eval(Source::from_bytes(source.as_bytes())).ok()?;
    let n = result.to_number(&mut context).ok()?;
    n.is_finite().then_some(n as f32)
}

/// WE exposes an object/attachment scripting API on every scene object
/// (`thisLayer`, and anything `thisScene.getLayer(...)` returns): xref'd from
/// the binary as 11 functions/properties all registered from one method
/// table (`lookAt`, `lookAtYaw`, `rotateObjectSpace`, `setParent`,
/// `parallaxDepth`, `getTransformMatrix`, `getAttachmentOrigin`,
/// `getAttachmentAngles`, `getAttachmentIndex`, `getAttachmentMatrix`,
/// `disablepropagation` — see the Ghidra report's object-attachment
/// follow-up). wp-engine had none of this before; `thisLayer`/`thisScene`
/// were static stubs.
///
/// No real downloaded Workshop content calls any of these (searched the
/// local corpus), and the vendored C++ reference doesn't implement this API
/// either (`parallaxDepth` there is only a parsed, non-scripted property) —
/// so beyond the names and the fact that they're one object's method table,
/// there's no ground truth for the exact semantics. This is an honest
/// best-effort implementation grounded in wp-engine's *own* transform
/// conventions (the same additive-Euler rotation composition
/// `render::Xform::compose` already uses for the static scene.json `parent`
/// field, and `camera3d::model_matrix`'s own Rz·Ry·Rx convention for
/// `getTransformMatrix`), not a verified match to the original binary.
///
/// No skeleton/bone/attachment-point data exists anywhere in this engine (or
/// in the C++ reference), so the `getAttachment*` family can't resolve a
/// real socket: `getAttachmentIndex` always reports "not found" (-1), and
/// `getAttachmentOrigin`/`Angles`/`Matrix` always resolve to the queried
/// object's own root transform regardless of `index`. `setParent` only
/// takes effect on `thisLayer` itself (the object whose own script is
/// running) — this engine has no live scene-graph re-parenting, so mutating
/// an object obtained via `getLayer` doesn't feed back into rendering.
const OBJECT_ATTACHMENT_API_JS: &str = r#"
function wpRotateEuler(v, angles) {
    var sx = Math.sin(angles.x), cx = Math.cos(angles.x);
    var sy = Math.sin(angles.y), cy = Math.cos(angles.y);
    var sz = Math.sin(angles.z), cz = Math.cos(angles.z);
    var x1 = v.x, y1 = cx * v.y - sx * v.z, z1 = sx * v.y + cx * v.z;
    var x2 = cy * x1 + sy * z1, y2 = y1, z2 = -sy * x1 + cy * z1;
    var x3 = cz * x2 - sz * y2, y3 = sz * x2 + cz * y2, z3 = z2;
    return new Vec3(x3, y3, z3);
}

function wpBuildMatrix(origin, angles, scale) {
    var sx = Math.sin(angles.x), cx = Math.cos(angles.x);
    var sy = Math.sin(angles.y), cy = Math.cos(angles.y);
    var sz = Math.sin(angles.z), cz = Math.cos(angles.z);
    var r00 = cz*cy,               r10 = sz*cy,               r20 = -sy;
    var r01 = cz*sy*sx - sz*cx,    r11 = sz*sy*sx + cz*cx,    r21 = cy*sx;
    var r02 = cz*sy*cx + sz*sx,    r12 = sz*sy*cx - cz*sx,    r22 = cy*cx;
    return [
        r00*scale.x, r10*scale.x, r20*scale.x, 0,
        r01*scale.y, r11*scale.y, r21*scale.y, 0,
        r02*scale.z, r12*scale.z, r22*scale.z, 0,
        origin.x, origin.y, origin.z, 1
    ];
}

function wpMakeSceneObject(name, id, origin, angles, scale, parallaxDepth) {
    var obj = {
        name: name || '',
        id: (id === null || id === undefined) ? -1 : id,
        origin: origin, angles: angles, scale: scale,
        parallaxDepth: parallaxDepth || new Vec3(0, 0, 0),
        visible: true, pointsize: 32, font: '', text: '', __frame: -1,
        __propagationDisabled: false,
        getTextureAnimation: function() {
            return {
                setFrame: function(f) { obj.__frame = f; },
                setFrameByName: function() {},
                getFrameCount: function() { return 1; }
            };
        },
        getTransformMatrix: function() {
            return wpBuildMatrix(obj.origin, obj.angles, obj.scale);
        },
        // Faces `target` (world position) by solving yaw (angles.y) and
        // pitch (angles.x) directly from wpBuildMatrix's own Rz*Ry*Rx
        // convention. Roll (angles.z) is left untouched — a full
        // look-at-with-roll decomposition needs a target "up" vector this
        // API doesn't supply.
        lookAt: function(target) {
            var d = target.subtract(obj.origin);
            var len = Math.sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
            if (len > 1e-6) {
                obj.angles.y = Math.atan2(d.x, d.z);
                obj.angles.x = Math.asin(Math.max(-1, Math.min(1, -d.y / len)));
            }
            return obj.angles;
        },
        lookAtYaw: function(target) {
            var d = target.subtract(obj.origin);
            if (Math.abs(d.x) > 1e-6 || Math.abs(d.z) > 1e-6) {
                obj.angles.y = Math.atan2(d.x, d.z);
            }
            return obj.angles;
        },
        // Additive, matching `Xform::compose`'s own simplified Euler-sum
        // rotation model — this engine never composes rotations as true
        // matrices (see render.rs), so staying additive here keeps this
        // consistent with every other rotation composition in the codebase.
        rotateObjectSpace: function(x, y, z) {
            obj.angles.x += x; obj.angles.y += y; obj.angles.z += z;
            return obj.angles;
        },
        // Treats this object's own origin/angles/scale as a local offset and
        // composes them onto `parent`'s current transform, via the exact
        // rule `Xform::compose`/`apply_point` already use for this engine's
        // static scene.json `parent` field.
        setParent: function(parent) {
            if (!parent) { return obj; }
            var scaledLocal = new Vec3(obj.origin.x * parent.scale.x, obj.origin.y * parent.scale.y, obj.origin.z * parent.scale.z);
            var rotated = wpRotateEuler(scaledLocal, parent.angles);
            obj.origin = rotated.add(parent.origin);
            obj.angles = new Vec3(obj.angles.x + parent.angles.x, obj.angles.y + parent.angles.y, obj.angles.z + parent.angles.z);
            obj.scale = new Vec3(obj.scale.x * parent.scale.x, obj.scale.y * parent.scale.y, obj.scale.z * parent.scale.z);
            return obj;
        },
        getAttachmentOrigin: function(index) { return obj.origin.copy(); },
        getAttachmentAngles: function(index) { return obj.angles.copy(); },
        getAttachmentMatrix: function(index) { return obj.getTransformMatrix(); },
        getAttachmentIndex: function(attachmentName) { return -1; },
        disablepropagation: function() { obj.__propagationDisabled = true; }
    };
    return obj;
}

var __wp_layers = [];
var thisLayer = wpMakeSceneObject('', -1, new Vec3(0,0,0), new Vec3(0,0,0), new Vec3(1,1,1), new Vec3(0,0,0));
var thisScene = {
    createLayer: function() { return { visible: false, origin: new Vec3(0,0,0), angles: new Vec3(0,0,0), scale: new Vec3(1,1,1), text: '' }; },
    sortLayer: function() {},
    getLayerIndex: function() { return 0; },
    // Searches the snapshot `set_layers` refreshes once per frame. `null`
    // when not found, matching WE's own documented behavior.
    getLayer: function(nameOrId) {
        for (var i = 0; i < __wp_layers.length; i++) {
            if (__wp_layers[i].name === nameOrId || __wp_layers[i].id === nameOrId) {
                return __wp_layers[i];
            }
        }
        return null;
    }
};
"#;

/// A persistent JS runtime for the render loop: the realm, intrinsics, and the
/// WE globals (`engine`, `WEMath`, `value`, `update`) are created once and
/// reused across frames, so a per-frame tick only mutates `engine.*` and runs
/// each property's `update`. This avoids rebuilding the entire JS realm on
/// every property, every frame (what the free [`eval_update`] does).
pub struct ScriptContext {
    context: Context,
}

impl Default for ScriptContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Real workshop scripts are authored as ES modules: `import * as WEMath
/// from 'WEMath'`, `export function update(value)`, `export let
/// __workshopId`, `export var scriptProperties = createScriptProperties()…`.
/// boa evaluates our snippets in script (non-module) mode where `import`/
/// `export` are syntax errors — and the WE globals they import already exist
/// in our realm anyway. Strip module syntax line-by-line: drop `import`
/// lines, peel `export ` off declarations.
fn strip_module_syntax(script: &str) -> String {
    let out = script
        .lines()
        .map(|line| {
            let t = line.trim_start();
            if t.starts_with("import ") {
                ""
            } else if let Some(rest) = t.strip_prefix("export ") {
                rest
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    // Minified/concatenated modules pack several statements per line, e.g.
    // `"use strict";export var x=…;…;export function update(){}` — the
    // line-start peel above misses those inner `export`s. Strip them at their
    // statement boundary (after `;` or `}`) too. `export` is a reserved word,
    // so it can only appear as a statement keyword outside strings/comments,
    // which these date/clock scripts don't contain.
    out.replace(";export ", ";").replace("}export ", "}")
}

/// Wrap a user script in an IIFE so it can be re-evaluated every frame in
/// the persistent realm: top-level `let`/`const` become function-scoped
/// (re-declaring a global `let` would throw on the second frame), `update`
/// and `scriptProperties` stay local to the closure, and WE's startup call
/// to `applyUserProperties(props)` (which initializes script config like
/// delimiters from the property defaults) runs before `update`. Trade-off:
/// module-level state does not persist across frames — fine for the
/// clock/date scripts that dominate real content.
fn iife_body(script: &str, authored_props: Option<&serde_json::Value>) -> String {
    // The scene JSON's `scriptproperties` object holds the values the author
    // chose in the editor (e.g. this clock instance shows hours only) —
    // merged over the script's own builder defaults before `init()` runs.
    let merge_props = authored_props
        .filter(|v| v.is_object())
        .map(|v| {
            format!(
                ";if (typeof scriptProperties !== 'undefined') {{ var __wp_p = {v}; for (var __wp_k in __wp_p) {{ scriptProperties[__wp_k] = __wp_p[__wp_k]; }} }}"
            )
        })
        .unwrap_or_default();
    format!(
        "(function() {{\n{script}\n{merge_props}\n;if (typeof init === 'function') {{ try {{ init(); }} catch (e) {{}} }}\nif (typeof applyUserProperties === 'function' && typeof scriptProperties !== 'undefined') {{ try {{ applyUserProperties(scriptProperties); }} catch (e) {{}} }}\nreturn (typeof update === 'function') ? update(value) : value;\n}})();",
        script = strip_module_syntax(script),
    )
}

impl ScriptContext {
    pub fn new() -> Self {
        let mut context = Context::default();
        // Declare the globals once. `value`/`update` are reset per evaluation;
        // `engine`/`WEMath` persist and are mutated by `set_time`.
        //
        // `createScriptProperties()` mirrors WE's chainable builder: every
        // `add*` records the property's default under its name on the same
        // object the script keeps (`scriptProperties.showHours` etc.), since
        // we have no editor UI feeding user-chosen values yet.
        let _ = context.eval(Source::from_bytes(
            "var value;\n\
             var update;\n\
             var engine = { runtime: 0, timeOfDay: 0, frametime: 0, registerTimeEvent: function(){}, userProperties: {} };\n\
             var shared = {};\n\
             var console = { log: function(){}, warn: function(){}, error: function(){} };\n\
             function createScriptProperties() {\n\
                 var api = {};\n\
                 function add(def) { if (def && def.name !== undefined) { api[def.name] = def.value; } return api; }\n\
                 ['addCheckbox','addSlider','addText','addTextInput','addCombo','addColorPicker','addFile','addDirectory','addBool'].forEach(function(m) { api[m] = add; });\n\
                 api.finish = function() { return api; };\n\
                 return api;\n\
             }\n\
             var WEMath = { \
                 smoothStep: function(a, b, x) { \
                     var t = Math.min(Math.max((x - a) / (b - a), 0), 1); \
                     return t * t * (3 - 2 * t); \
                 }, \
                 mix: function(a, b, t) { return a + (b - a) * t; }, \
                 clamp: function(x, a, b) { return Math.min(Math.max(x, a), b); }, \
                 frac: function(x) { return x - Math.floor(x); } \
             };\n\
             function Vec3(x, y, z) { this.x = x || 0; this.y = y || 0; this.z = z || 0; }\n\
             Vec3.prototype.copy = function() { return new Vec3(this.x, this.y, this.z); };\n\
             Vec3.prototype.add = function(o) { return new Vec3(this.x + o.x, this.y + o.y, this.z + o.z); };\n\
             Vec3.prototype.subtract = function(o) { return new Vec3(this.x - o.x, this.y - o.y, this.z - o.z); };\n\
             Vec3.prototype.multiply = function(o) { var s = (typeof o === 'number'); return new Vec3(this.x * (s ? o : o.x), this.y * (s ? o : o.y), this.z * (s ? o : o.z)); };\n\
             Vec3.prototype.divide = function(o) { var s = (typeof o === 'number'); return new Vec3(this.x / (s ? o : o.x), this.y / (s ? o : o.y), this.z / (s ? o : o.z)); };\n\
             var input = { cursorWorldPosition: new Vec3(0,0,0), cursorPosition: new Vec3(0,0,0) };\n\
             var media = { available: false, playbackState: 0, title: '', artist: '', album: '', position: 0, duration: 0, artUrl: null };",
        ));
        // The object-attachment scripting API (`lookAt`, `setParent`,
        // `getAttachmentX`, `getTransformMatrix`, `disablepropagation` —
        // see `OBJECT_ATTACHMENT_API_JS`'s own doc comment) is sizeable
        // enough to keep as its own eval rather than folding into the
        // block above.
        let _ = context.eval(Source::from_bytes(OBJECT_ATTACHMENT_API_JS.as_bytes()));
        Self { context }
    }

    /// Update the clock globals for the current frame. Call once per frame
    /// before evaluating this frame's property scripts. `frametime` is the
    /// delta since the previous frame in seconds (real transform scripts read
    /// `engine.frametime` to make motion frame-rate independent).
    pub fn set_time(&mut self, runtime: f32, time_of_day: f32, frametime: f32) {
        let src = format!(
            "engine.runtime = {}; engine.timeOfDay = {}; engine.frametime = {};",
            js_num(runtime),
            js_num(time_of_day),
            js_num(frametime),
        );
        let _ = self.context.eval(Source::from_bytes(src.as_bytes()));
    }

    /// Refresh the `media` global (currently-playing track info — see
    /// `engine::media`) for this frame.
    ///
    /// The real engine pushes media changes as events, calling
    /// `mediaPropertiesChanged`/`mediaPlaybackChanged`/`mediaTimelineChanged`/
    /// `mediaThumbnailChanged` as *named exports* on every loaded script
    /// module (confirmed from the vendored C++ reference's
    /// `ScriptEngine::notifyMediaUpdate`) — this codebase's SceneScript
    /// engine has no equivalent "loaded modules with named exports" concept
    /// to broadcast to (see `ScriptContext`'s own module doc: it evaluates
    /// one property's `update(value)` at a time). Exposing `media` as a
    /// plain read-model global instead — queryable from any property
    /// script's own body — is an honest adaptation to how this engine's
    /// scripts actually run, not an attempt at the reference's exact
    /// push-callback dispatch shape. Album-art color fields
    /// (`primaryColor` etc.) aren't included here at all: the reference's
    /// own values for those are hardcoded defaults too (its own `// TODO:
    /// PROCESS THESE COLORS INSTEAD OF HARDCODING THEM`), so there's
    /// nothing real to expose yet.
    pub fn set_media(
        &mut self,
        available: bool,
        playback_state: i32,
        title: &str,
        artist: &str,
        album: &str,
        position_us: i64,
        duration_us: i64,
        art_url: Option<&str>,
    ) {
        let art_url_js = art_url.map(js_str).unwrap_or_else(|| "null".to_string());
        let src = format!(
            "media.available = {available}; media.playbackState = {playback_state}; \
             media.title = {title}; media.artist = {artist}; media.album = {album}; \
             media.position = {position_us}; media.duration = {duration_us}; \
             media.artUrl = {art_url_js};",
            title = js_str(title),
            artist = js_str(artist),
            album = js_str(album),
        );
        let _ = self.context.eval(Source::from_bytes(src.as_bytes()));
    }

    /// Refresh `thisLayer` to the scene object about to have its own
    /// SceneScripts evaluated — the object-attachment API (`lookAt`,
    /// `setParent`, `getTransformMatrix`, etc., see
    /// [`OBJECT_ATTACHMENT_API_JS`]) reads and mutates whatever `thisLayer`
    /// currently holds, so callers must call this before evaluating that
    /// object's `origin`/`angles`/`scale`/`visible`/etc. scripts. Replaces
    /// `thisLayer` wholesale (a fresh `wpMakeSceneObject` call) rather than
    /// mutating fields in place, so a previous object's `disablepropagation`
    /// flag or attachment state can't leak into the next.
    pub fn set_current_object(
        &mut self,
        name: Option<&str>,
        id: Option<i64>,
        origin: [f32; 3],
        angles: [f32; 3],
        scale: [f32; 3],
        parallax_depth: [f32; 2],
    ) {
        let src = format!(
            "thisLayer = wpMakeSceneObject({name}, {id}, new Vec3({ox},{oy},{oz}), new Vec3({ax},{ay},{az}), new Vec3({sx},{sy},{sz}), new Vec3({px},{py},0));",
            name = js_str(name.unwrap_or("")),
            id = id.map(|v| v.to_string()).unwrap_or_else(|| "-1".to_string()),
            ox = js_num(origin[0]), oy = js_num(origin[1]), oz = js_num(origin[2]),
            ax = js_num(angles[0]), ay = js_num(angles[1]), az = js_num(angles[2]),
            sx = js_num(scale[0]), sy = js_num(scale[1]), sz = js_num(scale[2]),
            px = js_num(parallax_depth[0]), py = js_num(parallax_depth[1]),
        );
        let _ = self.context.eval(Source::from_bytes(src.as_bytes()));
    }

    /// Refresh `thisScene`'s object registry — `getLayer(nameOrId)` searches
    /// this. Call once per frame (not once per object) with every scene
    /// object's current name/id/origin/angles/scale.
    pub fn set_layers(&mut self, layers: &[LayerSnapshot]) {
        let mut src = String::from("__wp_layers = [");
        for l in layers {
            src.push_str(&format!(
                "wpMakeSceneObject({name}, {id}, new Vec3({ox},{oy},{oz}), new Vec3({ax},{ay},{az}), new Vec3({sx},{sy},{sz}), new Vec3(0,0,0)),",
                name = js_str(l.name.unwrap_or("")),
                id = l.id.map(|v| v.to_string()).unwrap_or_else(|| "-1".to_string()),
                ox = js_num(l.origin[0]), oy = js_num(l.origin[1]), oz = js_num(l.origin[2]),
                ax = js_num(l.angles[0]), ay = js_num(l.angles[1]), az = js_num(l.angles[2]),
                sx = js_num(l.scale[0]), sy = js_num(l.scale[1]), sz = js_num(l.scale[2]),
            ));
        }
        src.push_str("];");
        let _ = self.context.eval(Source::from_bytes(src.as_bytes()));
    }

    /// Whether the object last set via [`Self::set_current_object`] called
    /// `disablepropagation()` on itself. Call right after evaluating that
    /// object's scripts, before the next `set_current_object` overwrites
    /// `thisLayer`.
    pub fn take_propagation_disabled(&mut self) -> bool {
        self.context
            .eval(Source::from_bytes(b"thisLayer.__propagationDisabled"))
            .ok()
            .map(|v| v.to_boolean())
            .unwrap_or(false)
    }

    /// Evaluate a Vec3-valued property's `update(value)` (WE `origin`/`scale`/
    /// `angles`): `value` is passed as a `Vec3(x, y, z)` and the returned Vec3
    /// is read back. `None` when the script throws or yields a non-finite/
    /// non-Vec3 result, so callers keep the property's base value.
    ///
    /// The result is marshalled through a `"x y z"` string rather than boa's
    /// object-field API — same plumbing as [`Self::eval_update_string`], and
    /// it sidesteps the accessor churn across boa versions.
    pub fn eval_update_vec3(&mut self, script: &str, base: [f32; 3]) -> Option<[f32; 3]> {
        let _ = self
            .context
            .eval(Source::from_bytes(b"update = undefined;"));
        let src = format!(
            "value = new Vec3({x}, {y}, {z});\nvar __out = {body};\n\
             (__out && typeof __out === 'object') ? (__out.x + ' ' + __out.y + ' ' + __out.z) : ''",
            x = js_num(base[0]),
            y = js_num(base[1]),
            z = js_num(base[2]),
            body = iife_body(script, None),
        );
        let result = self.context.eval(Source::from_bytes(src.as_bytes())).ok()?;
        let s = result
            .to_string(&mut self.context)
            .ok()?
            .to_std_string_escaped();
        let parts: Vec<f32> = s
            .split_whitespace()
            .filter_map(|p| p.parse().ok())
            .collect();
        (parts.len() == 3 && parts.iter().all(|v| v.is_finite()))
            .then(|| [parts[0], parts[1], parts[2]])
    }

    /// Evaluate a boolean-valued property's `update(value)` (WE `visible`):
    /// `value` is passed as `0`/`1` and the truthy result maps back to a bool.
    /// `None` when the script throws, so callers keep the base visibility.
    pub fn eval_update_bool(&mut self, script: &str, base: bool) -> Option<bool> {
        self.eval_update(script, if base { 1.0 } else { 0.0 })
            .map(|n| n >= 0.5)
    }

    /// Evaluate one property's `update(value)` against the persistent context,
    /// returning its finite numeric result or `None` (no `update`, threw, or
    /// non-finite) so the caller keeps the property's current value.
    ///
    /// `update` is reset before the user script runs so a previous property's
    /// leftover `update` can't leak into a script that doesn't define its own.
    pub fn eval_update(&mut self, script: &str, current_value: f32) -> Option<f32> {
        // Reset in its own program: a function-declaration hoist in the user
        // script must win, which it can't if the reset shared the same program.
        let _ = self
            .context
            .eval(Source::from_bytes(b"update = undefined;"));
        let src = format!(
            "value = {v};\n{body}",
            v = js_num(current_value),
            body = iife_body(script, None),
        );
        let result = self.context.eval(Source::from_bytes(src.as_bytes())).ok()?;
        let n = result.to_number(&mut self.context).ok()?;
        n.is_finite().then_some(n as f32)
    }

    /// Evaluate a text object's `update(value)`, returning its string result —
    /// the dominant script kind in real content (clocks, dates, countdowns:
    /// 42 of the 58 script-using wallpapers in the local corpus). `None` when
    /// the script throws or has no usable `update`, so callers keep the
    /// previous text.
    pub fn eval_update_string(
        &mut self,
        script: &str,
        current_value: &str,
        authored_props: Option<&serde_json::Value>,
    ) -> Option<String> {
        let _ = self
            .context
            .eval(Source::from_bytes(b"update = undefined;"));
        // serde_json string-escapes into a valid JS string literal.
        let quoted = serde_json::to_string(current_value).ok()?;
        let src = format!(
            "value = {quoted};\n{body}",
            body = iife_body(script, authored_props)
        );
        let result = self.context.eval(Source::from_bytes(src.as_bytes())).ok()?;
        // A script that returns `undefined`/`null` (e.g. reads an unstubbed
        // scene API) stringifies to the literal "undefined"/"null" — keep the
        // caller's current text instead of rendering that.
        if result.is_undefined() || result.is_null() {
            return None;
        }
        let js_str = result.to_string(&mut self.context).ok()?;
        Some(js_str.to_std_string_escaped())
    }

    /// Evaluate a script for its `thisLayer.getTextureAnimation().setFrame(n)`
    /// side effect (spritesheet frame selection — e.g. an AM/PM clock cell picked
    /// by `getHours()`). Returns the selected frame index, or `None` if the
    /// script set none. `set_time` should be called first so `Date`/`timeOfDay`
    /// reflect the intended clock.
    pub fn eval_frame(&mut self, script: &str) -> Option<u32> {
        let _ = self.context.eval(Source::from_bytes(
            b"update = undefined; thisLayer.__frame = -1;",
        ));
        // Frame scripts hang the setFrame off `update`; run with a truthy `value`
        // (these are usually `visible` scripts that just return the value back).
        let src = format!("value = true;\n{body}", body = iife_body(script, None));
        let _ = self.context.eval(Source::from_bytes(src.as_bytes()));
        let frame = self
            .context
            .eval(Source::from_bytes(b"thisLayer.__frame"))
            .ok()?
            .to_number(&mut self.context)
            .ok()?;
        (frame.is_finite() && frame >= 0.0).then_some(frame as u32)
    }
}

pub fn eval_script_opt(script: Option<&str>, current_value: f32, env: &ScriptEnv) -> f32 {
    match script {
        Some(s) => eval_update(s, current_value, env).unwrap_or(current_value),
        None => current_value,
    }
}

pub fn eval_vec3_script_opt(
    animated: &crate::engine::model::AnimatedValue,
    env: &ScriptEnv,
) -> [f32; 3] {
    // Default to [1.0; 3] so scale and color don't zero-out the output when missing
    let base = animated.as_vec3().unwrap_or([1.0; 3]);
    match &animated.script {
        None => base,
        Some(script) => [
            eval_update(script, base[0], env).unwrap_or(base[0]),
            eval_update(script, base[1], env).unwrap_or(base[1]),
            eval_update(script, base[2], env).unwrap_or(base[2]),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(runtime: f32, time_of_day: f32) -> ScriptEnv {
        ScriptEnv {
            runtime,
            time_of_day,
        }
    }

    #[test]
    fn returns_current_value_when_no_update_function() {
        let out = eval_update("var x = 3;", 7.0, &env(0.0, 0.0));
        assert_eq!(out, Some(7.0));
    }

    #[test]
    fn simple_arithmetic_update() {
        let script = "function update(value) { return value * 2 + 1; }";
        assert_eq!(eval_update(script, 5.0, &env(0.0, 0.0)), Some(11.0));
    }

    #[test]
    fn reads_engine_runtime() {
        let script = "function update(value) { return engine.runtime; }";
        assert_eq!(eval_update(script, 0.0, &env(4.5, 0.0)), Some(4.5));
    }

    #[test]
    fn multi_statement_body_with_locals_and_math() {
        // The old hand-rolled parser could only handle a single `return <expr>`.
        let script = "function update(value) {\n\
                          var a = Math.floor(value);\n\
                          var b = Math.sin(0) + 1;\n\
                          return a * b;\n\
                      }";
        assert_eq!(eval_update(script, 3.7, &env(0.0, 0.0)), Some(3.0));
    }

    #[test]
    fn conditional_and_loop() {
        let script = "function update(value) {\n\
                          var sum = 0;\n\
                          for (var i = 0; i < 4; i++) { sum += i; }\n\
                          return value > 0 ? sum : -sum;\n\
                      }";
        assert_eq!(eval_update(script, 1.0, &env(0.0, 0.0)), Some(6.0));
        assert_eq!(eval_update(script, -1.0, &env(0.0, 0.0)), Some(-6.0));
    }

    #[test]
    fn wemath_smoothstep_matches_definition() {
        let script = "function update(value) { return WEMath.smoothStep(0, 1, value); }";
        // midpoint of smoothstep is exactly 0.5
        assert_eq!(eval_update(script, 0.5, &env(0.0, 0.0)), Some(0.5));
        // clamps below/above the edges
        assert_eq!(eval_update(script, -1.0, &env(0.0, 0.0)), Some(0.0));
        assert_eq!(eval_update(script, 2.0, &env(0.0, 0.0)), Some(1.0));
    }

    #[test]
    fn throwing_script_falls_back_to_current_value() {
        let script = "function update(value) { return nonexistent.field; }";
        assert_eq!(eval_update(script, 9.0, &env(0.0, 0.0)), None);
        assert_eq!(eval_script_opt(Some(script), 9.0, &env(0.0, 0.0)), 9.0);
    }

    #[test]
    fn persistent_context_reevaluates_across_frames() {
        let mut ctx = ScriptContext::new();
        let script = "function update(value) { return engine.runtime * 2; }";

        ctx.set_time(1.0, 0.0, 0.016);
        assert_eq!(ctx.eval_update(script, 0.0), Some(2.0));

        // Same context, new frame: only the clock changed, result tracks it.
        ctx.set_time(5.5, 0.0, 0.016);
        assert_eq!(ctx.eval_update(script, 0.0), Some(11.0));
    }

    #[test]
    fn persistent_context_passes_current_value_through() {
        let mut ctx = ScriptContext::new();
        let script = "function update(value) { return value + 1; }";
        assert_eq!(ctx.eval_update(script, 4.0), Some(5.0));
        assert_eq!(ctx.eval_update(script, 40.0), Some(41.0));
    }

    #[test]
    fn persistent_context_does_not_leak_update_between_scripts() {
        let mut ctx = ScriptContext::new();
        // First property defines an `update`.
        assert_eq!(
            ctx.eval_update("function update(value) { return 99; }", 1.0),
            Some(99.0)
        );
        // Second property has no `update`: it must keep its own value, not
        // reuse the leftover `update` from the first property.
        assert_eq!(ctx.eval_update("var unrelated = 3;", 7.0), Some(7.0));
    }

    #[test]
    fn persistent_context_survives_a_throwing_script() {
        let mut ctx = ScriptContext::new();
        assert_eq!(
            ctx.eval_update("function update(v){ return boom.x; }", 2.0),
            None
        );
        // The context is still usable afterwards.
        assert_eq!(
            ctx.eval_update("function update(v){ return v * 10; }", 2.0),
            Some(20.0)
        );
    }

    /// Real workshop text scripts are ES modules with `import`/`export` and a
    /// `createScriptProperties()` builder — the whole shape must evaluate and
    /// return a string.
    #[test]
    fn module_style_text_script_evaluates_to_string() {
        let mut ctx = ScriptContext::new();
        let script = r#"'use strict';
export let __workshopId = '2741588178';
import * as WEMath from 'WEMath';
export var scriptProperties = createScriptProperties()
    .addCheckbox({ name: 'showLabel', value: true })
    .addText({ name: 'format', value: 'H:M' });
export function update(value) {
    var d = new Date();
    var label = scriptProperties.showLabel ? 'T-' : '';
    return label + scriptProperties.format + '!';
}"#;
        assert_eq!(
            ctx.eval_update_string(script, "old", None).as_deref(),
            Some("T-H:M!")
        );
    }

    /// A Vec3 `scale`/`origin`/`angles` script: `value` arrives as a Vec3 and
    /// the returned Vec3 is read back component-wise.
    #[test]
    fn vec3_script_reads_and_returns_vec3() {
        let mut ctx = ScriptContext::new();
        // Doubles x, passes y through, zeroes z.
        let script = "export function update(value) { \
            return new Vec3(value.x * 2, value.y, 0); }";
        assert_eq!(
            ctx.eval_update_vec3(script, [3.0, 5.0, 9.0]),
            Some([6.0, 5.0, 0.0])
        );
    }

    /// The real `angles` idiom mutates `value` in place and returns it.
    #[test]
    fn vec3_script_mutate_in_place() {
        let mut ctx = ScriptContext::new();
        ctx.set_time(0.0, 0.0, 0.5);
        let script = "export function update(value) { \
            value.x += engine.frametime; return value; }";
        assert_eq!(
            ctx.eval_update_vec3(script, [1.0, 2.0, 3.0]),
            Some([1.5, 2.0, 3.0])
        );
    }

    /// A `visible` script returns a boolean; `value` arrives as 0/1.
    #[test]
    fn bool_script_toggles_visibility() {
        let mut ctx = ScriptContext::new();
        assert_eq!(
            ctx.eval_update_bool("export function update(value) { return !value; }", true),
            Some(false)
        );
        assert_eq!(
            ctx.eval_update_bool("export function update(value) { return value > 0; }", true),
            Some(true)
        );
    }

    /// Real content has invisible click/hover hitboxes: `visible` is authored
    /// `false` and the script defines ONLY cursor handlers — no `update()`. The
    /// script must then return the authored value unchanged, or the hitbox
    /// renders as an opaque box (3509243656's white box).
    #[test]
    fn bool_script_without_update_keeps_authored_value() {
        let mut ctx = ScriptContext::new();
        let hitbox = "'use strict';\n\
            export function cursorClick(event) { shared.yp = 1; }\n\
            export function cursorEnter(event) { shared.cret1 = 1; }";
        assert_eq!(ctx.eval_update_bool(hitbox, false), Some(false));
        assert_eq!(ctx.eval_update_bool(hitbox, true), Some(true));
    }

    /// A text script that returns `undefined` (unstubbed scene API) must not
    /// render the literal "undefined" — the caller keeps its current text.
    #[test]
    fn text_script_returning_undefined_yields_none() {
        let mut ctx = ScriptContext::new();
        assert_eq!(
            ctx.eval_update_string(
                "export function update(v) { return undefined; }",
                "keep",
                None
            ),
            None
        );
        assert_eq!(
            ctx.eval_update_string("export function update(v) { return 'ok'; }", "keep", None)
                .as_deref(),
            Some("ok")
        );
    }

    /// A clock-style script using `Date` must produce plausible digits.
    #[test]
    fn date_script_returns_time_digits() {
        let mut ctx = ScriptContext::new();
        let script = r#"export function update(value) {
            var d = new Date();
            var h = d.getHours();
            return (h < 10 ? '0' : '') + h;
        }"#;
        let out = ctx
            .eval_update_string(script, "", None)
            .expect("should evaluate");
        let n: u32 = out.parse().expect("two digits");
        assert!(n < 24, "hours out of range: {out}");
    }

    /// Minified module scripts pack `"use strict";export var …;…;export
    /// function update(){}` onto one line — every inner `export` must be
    /// stripped, not just the line-leading one.
    #[test]
    fn strips_inner_statement_exports() {
        let s = strip_module_syntax(
            "\"use strict\";export var a=1;export function update(v){return a;}",
        );
        assert!(!s.contains("export"), "leftover export in: {s}");
    }

    // --- object-attachment API (`OBJECT_ATTACHMENT_API_JS`) ---

    #[test]
    fn look_at_faces_the_target_straight_ahead() {
        let mut ctx = ScriptContext::new();
        ctx.set_current_object(
            Some("obj"),
            Some(1),
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0],
        );
        let script =
            "export function update(value) { return thisLayer.lookAt(new Vec3(0, 0, 5)); }";
        let angles = ctx
            .eval_update_vec3(script, [0.0, 0.0, 0.0])
            .expect("should evaluate");
        // Target is straight down local +Z: both pitch and yaw are 0.
        assert!(angles[0].abs() < 1e-4, "pitch: {angles:?}");
        assert!(angles[1].abs() < 1e-4, "yaw: {angles:?}");
    }

    #[test]
    fn look_at_yaw_touches_only_the_y_axis() {
        let mut ctx = ScriptContext::new();
        ctx.set_current_object(
            None,
            None,
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.7],
            [1.0, 1.0, 1.0],
            [0.0, 0.0],
        );
        // Target far to +X, high up: yaw should swing toward +X (atan2(5,0) = pi/2)
        // while the authored pitch/roll pass through untouched.
        let script =
            "export function update(value) { return thisLayer.lookAtYaw(new Vec3(5, 100, 0)); }";
        let angles = ctx
            .eval_update_vec3(script, [0.3, 0.0, 0.7])
            .expect("should evaluate");
        assert!(
            (angles[0] - 0.3).abs() < 1e-4,
            "pitch must stay unchanged: {angles:?}"
        );
        assert!(
            (angles[2] - 0.7).abs() < 1e-4,
            "roll must stay unchanged: {angles:?}"
        );
        assert!(
            (angles[1] - std::f32::consts::FRAC_PI_2).abs() < 1e-3,
            "yaw should point toward +X: {angles:?}"
        );
    }

    #[test]
    fn rotate_object_space_is_additive() {
        let mut ctx = ScriptContext::new();
        ctx.set_current_object(None, None, [0.0; 3], [0.1, 0.2, 0.3], [1.0; 3], [0.0, 0.0]);
        let script =
            "export function update(value) { return thisLayer.rotateObjectSpace(0.5, -0.1, 0.2); }";
        let angles = ctx
            .eval_update_vec3(script, [0.1, 0.2, 0.3])
            .expect("should evaluate");
        assert!((angles[0] - 0.6).abs() < 1e-5, "{angles:?}");
        assert!((angles[1] - 0.1).abs() < 1e-5, "{angles:?}");
        assert!((angles[2] - 0.5).abs() < 1e-5, "{angles:?}");
    }

    #[test]
    fn get_transform_matrix_translation_column_matches_origin() {
        let mut ctx = ScriptContext::new();
        ctx.set_current_object(None, None, [3.0, 4.0, 5.0], [0.0; 3], [1.0; 3], [0.0, 0.0]);
        let script = "export function update(value) { \
            var m = thisLayer.getTransformMatrix(); \
            return new Vec3(m[12], m[13], m[14]); }";
        let t = ctx
            .eval_update_vec3(script, [0.0; 3])
            .expect("should evaluate");
        assert_eq!(t, [3.0, 4.0, 5.0]);
    }

    #[test]
    fn set_parent_composes_local_offset_onto_the_named_parent() {
        let mut ctx = ScriptContext::new();
        ctx.set_layers(&[LayerSnapshot {
            name: Some("Anchor"),
            id: Some(9),
            origin: [10.0, 0.0, 0.0],
            angles: [0.0, 0.0, 0.0],
            scale: [2.0, 2.0, 2.0],
        }]);
        ctx.set_current_object(
            Some("Child"),
            Some(1),
            [1.0, 0.0, 0.0],
            [0.0; 3],
            [1.0; 3],
            [0.0, 0.0],
        );
        let script = "export function update(value) { \
            return thisLayer.setParent(thisScene.getLayer('Anchor')).origin; }";
        let origin = ctx
            .eval_update_vec3(script, [1.0, 0.0, 0.0])
            .expect("should evaluate");
        // local (1,0,0) * parent scale (2,2,2) = (2,0,0), unrotated, + parent origin (10,0,0)
        assert_eq!(origin, [12.0, 0.0, 0.0]);
    }

    #[test]
    fn get_layer_returns_null_for_an_unknown_name() {
        let mut ctx = ScriptContext::new();
        ctx.set_layers(&[]);
        let script = "export function update(value) { \
            return thisScene.getLayer('Nope') === null ? new Vec3(1,1,1) : new Vec3(0,0,0); }";
        let v = ctx
            .eval_update_vec3(script, [0.0; 3])
            .expect("should evaluate");
        assert_eq!(v, [1.0, 1.0, 1.0]);
    }

    /// No skeleton/bone/attachment-point data exists anywhere in this engine
    /// — `getAttachmentIndex` always reports -1, and `getAttachmentOrigin`
    /// always resolves to the object's own transform regardless of index.
    #[test]
    fn attachment_queries_have_no_bone_table_so_resolve_to_self_or_not_found() {
        let mut ctx = ScriptContext::new();
        ctx.set_current_object(None, None, [7.0, 8.0, 9.0], [0.0; 3], [1.0; 3], [0.0, 0.0]);
        let script = "export function update(value) { \
            var idx = thisLayer.getAttachmentIndex('hand'); \
            var o = thisLayer.getAttachmentOrigin(idx); \
            return new Vec3(idx, o.x, o.z); }";
        let out = ctx
            .eval_update_vec3(script, [0.0; 3])
            .expect("should evaluate");
        assert_eq!(out, [-1.0, 7.0, 9.0]);
    }

    #[test]
    fn disablepropagation_sets_a_readable_flag_reset_by_the_next_object() {
        let mut ctx = ScriptContext::new();
        ctx.set_current_object(None, None, [0.0; 3], [0.0; 3], [1.0; 3], [0.0, 0.0]);
        assert!(!ctx.take_propagation_disabled());

        let script =
            "export function update(value) { thisLayer.disablepropagation(); return value; }";
        ctx.eval_update_vec3(script, [0.0; 3]);
        assert!(ctx.take_propagation_disabled());

        // A fresh object (no call to disablepropagation) starts clean again.
        ctx.set_current_object(None, None, [0.0; 3], [0.0; 3], [1.0; 3], [0.0, 0.0]);
        assert!(!ctx.take_propagation_disabled());
    }

    // --- media (engine::media, see set_media's own doc comment) ---

    #[test]
    fn media_global_starts_unavailable() {
        let mut ctx = ScriptContext::new();
        let script = "export function update(value) { return media.available ? 1 : 0; }";
        assert_eq!(ctx.eval_update(script, 0.0), Some(0.0));
    }

    #[test]
    fn set_media_updates_the_global_a_script_can_read() {
        let mut ctx = ScriptContext::new();
        ctx.set_media(
            true,
            1,
            "Song Title",
            "The Artist",
            "An Album",
            30_000_000,
            180_000_000,
            None,
        );

        let script = "export function update(value) { return media.available ? 1 : 0; }";
        assert_eq!(ctx.eval_update(script, 0.0), Some(1.0));

        let script = "export function update(value) { return media.playbackState; }";
        assert_eq!(ctx.eval_update(script, 0.0), Some(1.0));

        let script = "export function update(value) { return media.position / 1000000; }";
        assert_eq!(ctx.eval_update(script, 0.0), Some(30.0));

        let script = "export function update(value) { return media.artUrl === null ? 1 : 0; }";
        assert_eq!(ctx.eval_update(script, 0.0), Some(1.0));
    }

    #[test]
    fn set_media_escapes_string_fields_safely() {
        let mut ctx = ScriptContext::new();
        // A title containing a quote and a backslash must not break out of
        // the JS string literal `set_media` builds.
        ctx.set_media(
            true,
            0,
            "Quote\" \\ Test",
            "",
            "",
            0,
            0,
            Some("http://example.com/art.jpg"),
        );

        let script = "export function update(value) { return media.title; }";
        assert_eq!(
            ctx.eval_update_string(script, "", None).as_deref(),
            Some("Quote\" \\ Test")
        );
        let script = "export function update(value) { return media.artUrl; }";
        assert_eq!(
            ctx.eval_update_string(script, "", None).as_deref(),
            Some("http://example.com/art.jpg")
        );
    }
}
