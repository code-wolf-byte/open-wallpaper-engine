//! SceneScript: Wallpaper Engine's per-object JavaScript, run in a persistent
//! [`boa_engine`] realm.
//!
//! Each scene object gets one long-lived layer proxy (`thisLayer`), and each
//! script attached to it is compiled once into an instance whose module-level
//! state persists for the life of the wallpaper: `init(value)` runs once,
//! `update(value)` every frame, and the event handlers (`cursorClick`,
//! `applyUserProperties`, the `media*` callbacks…) when their event fires.
//! The JS side of the API lives in `scenescript_runtime.js`.
//!
//! A frame is two calls into JS: [`ScriptHost::frame`] runs timers and every
//! instance's `update`, then [`ScriptHost::collect`] returns one diff of every
//! layer property, command (sound play/pause, layer creation…) and log line
//! the scripts produced, for the renderer to apply.
//!
//! Units: scene.json stores angles in radians, but scripts see degrees, so
//! angles are converted at this boundary in both directions.

use boa_engine::{Context, Source};
use serde_json::Value;
use std::path::PathBuf;

const RUNTIME_JS: &str = include_str!("scenescript_runtime.js");

/// Handlers a script module may export. The factory wrapper returns each one
/// that the script defines.
const EXPORTS: &[&str] = &[
    "init",
    "update",
    "destroy",
    "applyUserProperties",
    "resizeScreen",
    "cursorClick",
    "cursorDown",
    "cursorUp",
    "cursorMove",
    "cursorEnter",
    "cursorLeave",
    "mediaPlaybackChanged",
    "mediaPropertiesChanged",
    "mediaThumbnailChanged",
    "mediaTimelineChanged",
    "mediaStatusChanged",
    "animationEvent",
    "scriptProperties",
];

/// The cursor events a layer can receive.
pub const CURSOR_EVENTS: &[&str] = &[
    "cursorClick",
    "cursorDown",
    "cursorUp",
    "cursorMove",
    "cursorEnter",
    "cursorLeave",
];

/// Format an `f32` as a JS numeric literal, mapping non-finite values to `0`
/// so a stray `inf`/`NaN` can't inject invalid tokens into the source.
fn js_num(v: f32) -> String {
    if v.is_finite() {
        format!("{v:?}")
    } else {
        "0".to_string()
    }
}

fn js_vec(v: &[f32]) -> String {
    v.iter().map(|c| js_num(*c)).collect::<Vec<_>>().join(" ")
}

/// Format a `&str` as a JS string literal (serde_json's escaping is valid JS).
fn js_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string())
}

/// Initial state of one scene object's layer proxy.
#[derive(Debug, Clone, Default)]
pub struct ObjectInit {
    pub name: Option<String>,
    pub id: Option<i64>,
    pub origin: [f32; 3],
    /// Radians, as stored in scene.json.
    pub angles: [f32; 3],
    pub scale: [f32; 3],
    pub size: [f32; 3],
    pub color: [f32; 3],
    pub visible: bool,
    pub alpha: f32,
    pub brightness: f32,
    pub text: Option<String>,
    pub pointsize: f32,
    pub volume: f32,
    pub parallax_depth: [f32; 2],
    /// Index of the parent object, for `getParent()`.
    pub parent_index: Option<usize>,
    /// Spritesheet frame count, for `getTextureAnimation().getFrameCount()`.
    pub frame_count: u32,
    /// The object's scene.json entry, for `thisScene.getInitialLayerConfig`
    /// and `createLayer` copies.
    pub raw: Option<Value>,
}

/// What scripts changed on one object since the last [`ScriptHost::collect`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ObjectChange {
    pub origin: Option<[f32; 3]>,
    /// Radians.
    pub angles: Option<[f32; 3]>,
    pub scale: Option<[f32; 3]>,
    pub color: Option<[f32; 3]>,
    pub visible: Option<bool>,
    pub alpha: Option<f32>,
    pub brightness: Option<f32>,
    pub text: Option<String>,
    pub pointsize: Option<f32>,
    pub volume: Option<f32>,
    pub rate: Option<f32>,
    pub frame: Option<u32>,
    pub propagation_disabled: Option<bool>,
    /// Extra declared properties (shader constants, particle overrides…),
    /// keyed as declared via [`ScriptHost::attach`].
    pub extra: serde_json::Map<String, Value>,
}

/// A side effect a script requested.
#[derive(Debug, Clone, PartialEq)]
pub enum ScriptCommand {
    /// `play`/`pause`/`stop` on a sound layer.
    Sound { index: usize, op: String },
    /// `play`/`pause`/`stop` on a layer's spritesheet animation.
    TextureAnimation { index: usize, op: String },
    /// Control of a property timeline animation (`getAnimation(prop)`).
    Animation {
        index: usize,
        property: String,
        op: String,
        frame: Option<f32>,
    },
    /// Move a layer to a new draw position.
    Sort { index: usize, to: usize },
    /// `thisScene.createLayer(config)` — the new object's index and config.
    Create { index: usize, config: Value },
    /// `thisScene.destroyLayer(layer)`.
    Destroy { index: usize },
    /// `engine.openUserShortcut(name)`.
    Shortcut(String),
}

#[derive(Debug, Default)]
pub struct ScriptChanges {
    pub objects: Vec<(usize, ObjectChange)>,
    pub commands: Vec<ScriptCommand>,
}

impl ScriptChanges {
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty() && self.commands.is_empty()
    }
}

/// Real workshop scripts are ES modules (`import * as WEMath from 'WEMath'`,
/// `export function update(value)`), but boa runs them here as function
/// bodies, where `import`/`export` are syntax errors — and the modules they
/// import are realm globals anyway. Imports become plain bindings; `export`
/// is peeled off declarations.
fn module_to_body(script: &str) -> String {
    let mut out = Vec::new();
    for line in script.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("import ") {
            out.push(import_to_bindings(rest));
        } else if let Some(rest) = t.strip_prefix("export default ") {
            out.push(rest.to_string());
        } else if let Some(rest) = t.strip_prefix("export ") {
            out.push(rest.to_string());
        } else {
            out.push(line.to_string());
        }
    }
    // Minified modules pack statements onto one line; peel the inner
    // `export`s at their statement boundary too. `export` is reserved, so it
    // only appears as a keyword outside strings and comments.
    out.join("\n")
        .replace(";export ", ";")
        .replace("}export ", "}")
}

/// `* as X from 'Mod'` → `var X = Mod;`, `{ a, b as c } from 'Mod'` →
/// `var a = Mod.a, c = Mod.b;`. Anything else is dropped.
fn import_to_bindings(rest: &str) -> String {
    let rest = rest.trim().trim_end_matches(';');
    let Some((what, from)) = rest.rsplit_once(" from ") else {
        return String::new();
    };
    let module = from.trim().trim_matches(|c| c == '\'' || c == '"');
    if !module
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
        || module.is_empty()
    {
        return String::new();
    }
    let what = what.trim();
    if let Some(alias) = what.strip_prefix("* as ") {
        let alias = alias.trim();
        return if alias == module {
            String::new()
        } else {
            format!("var {alias} = {module};")
        };
    }
    if let Some(list) = what.strip_prefix('{').and_then(|w| w.strip_suffix('}')) {
        let bindings: Vec<String> = list
            .split(',')
            .filter_map(|item| {
                let item = item.trim();
                if item.is_empty() {
                    return None;
                }
                let (name, alias) = match item.split_once(" as ") {
                    Some((n, a)) => (n.trim(), a.trim()),
                    None => (item, item),
                };
                Some(format!("{alias} = {module}.{name}"))
            })
            .collect();
        if !bindings.is_empty() {
            return format!("var {};", bindings.join(", "));
        }
    }
    String::new()
}

/// The factory `__wp_register_instance` calls: the script body, then an
/// object of whichever handlers it defined.
fn factory_source(script: &str) -> String {
    let exports = EXPORTS
        .iter()
        .map(|e| format!("{e}: (typeof {e} !== 'undefined') ? {e} : undefined"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "function (thisLayer) {{\n{body}\n;return {{ {exports} }};\n}}",
        body = module_to_body(script)
    )
}

/// Fraction of the local day elapsed, `[0, 1)` — `engine.timeOfDay`.
pub fn local_time_of_day() -> f32 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let offset = local_utc_offset_secs(now as i64);
    let secs = (now + offset as f64).rem_euclid(86_400.0);
    (secs / 86_400.0) as f32
}

#[cfg(unix)]
fn local_utc_offset_secs(t: i64) -> i64 {
    let time = t as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: localtime_r writes only into `tm`, which outlives the call.
    let ok = unsafe { !libc::localtime_r(&time, &mut tm).is_null() };
    if ok {
        tm.tm_gmtoff
    } else {
        0
    }
}

#[cfg(not(unix))]
fn local_utc_offset_secs(_t: i64) -> i64 {
    0
}

/// The persistent SceneScript runtime for one loaded scene.
pub struct ScriptHost {
    context: Context,
    storage_path: Option<PathBuf>,
    instances: usize,
}

impl Default for ScriptHost {
    fn default() -> Self {
        Self::new()
    }
}

impl ScriptHost {
    pub fn new() -> Self {
        let mut context = Context::default();
        if let Err(e) = context.eval(Source::from_bytes(RUNTIME_JS.as_bytes())) {
            tracing::error!(target: "script", "SceneScript runtime failed to load: {e}");
        }
        Self {
            context,
            storage_path: None,
            instances: 0,
        }
    }

    /// Persist `localStorage` to `path` (a JSON object), loading what's there.
    pub fn with_storage(mut self, path: PathBuf) -> Self {
        if let Ok(text) = std::fs::read_to_string(&path) {
            if serde_json::from_str::<serde_json::Map<String, Value>>(&text).is_ok() {
                self.run(&format!("__wp.storage = {text};"));
            }
        }
        self.storage_path = Some(path);
        self
    }

    /// The `localStorage` file for the wallpaper at `dir`.
    pub fn storage_path_for(dir: &std::path::Path) -> Option<PathBuf> {
        let key = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
        let name = key.file_name()?.to_string_lossy().to_string();
        Some(
            dirs::data_dir()?
                .join("wp-engine")
                .join("storage")
                .join(format!("{name}.json")),
        )
    }

    fn run(&mut self, src: &str) -> Option<boa_engine::JsValue> {
        match self.context.eval(Source::from_bytes(src.as_bytes())) {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::debug!(target: "script", "eval failed: {e}");
                None
            }
        }
    }

    fn run_string(&mut self, src: &str) -> Option<String> {
        let v = self.run(src)?;
        v.as_string().map(|s| s.to_std_string_escaped())
    }

    /// Whether any script was attached — callers can skip per-frame script
    /// work entirely when not.
    pub fn has_scripts(&self) -> bool {
        self.instances > 0
    }

    /// Create the persistent layer proxy for scene object `index`.
    pub fn register_object(&mut self, index: usize, init: &ObjectInit) {
        let deg = 180.0 / std::f32::consts::PI;
        let angles = init.angles.map(|a| a * deg);
        let config = serde_json::json!({
            "raw": init.raw.clone().unwrap_or(Value::Null),
            "parentIndex": init.parent_index.map(|p| p as i64).unwrap_or(-1),
            "frameCount": init.frame_count.max(1),
        });
        let src = format!(
            "__wp_register_object({index}, {{ name: {name}, id: {id}, origin: {origin}, angles: {angles}, \
             scale: {scale}, size: {size}, color: {color}, visible: {visible}, alpha: {alpha}, \
             brightness: {brightness}, text: {text}, pointsize: {pointsize}, volume: {volume}, \
             parallaxDepth: {parallax}, config: {config} }});",
            name = js_str(init.name.as_deref().unwrap_or("")),
            id = init.id.map(|v| v.to_string()).unwrap_or_else(|| "-1".into()),
            origin = js_str(&js_vec(&init.origin)),
            angles = js_str(&js_vec(&angles)),
            scale = js_str(&js_vec(&init.scale)),
            size = js_str(&js_vec(&init.size)),
            color = js_str(&js_vec(&init.color)),
            visible = init.visible,
            alpha = js_num(init.alpha),
            brightness = js_num(init.brightness),
            text = js_str(init.text.as_deref().unwrap_or("")),
            pointsize = js_num(init.pointsize),
            volume = js_num(init.volume),
            parallax = js_str(&js_vec(&init.parallax_depth)),
            config = config,
        );
        self.run(&src);
    }

    /// Declare an extra scriptable property on an object (a shader constant,
    /// a particle override…) so scripts can read and write it by `key`.
    /// `kind` is `"number"`, `"bool"`, `"string"`, `"vec2"`, `"vec3"` or
    /// `"vec4"`; `value` is its initial value.
    pub fn declare_property(&mut self, index: usize, key: &str, kind: &str, value: &Value) {
        let src = format!(
            "(function(){{ var l = __wp.objects[{index}]; if (l) {{ l.__declare({key}, {kind}, {value}); }} }})();",
            key = js_str(key),
            kind = js_str(kind),
        );
        self.run(&src);
    }

    /// Compile `script` as an instance attached to object `index`. With
    /// `property`, its `update(value)` drives that property each frame;
    /// without, it only receives events. `authored_props` is the scene's
    /// `scriptproperties` object, merged over the script's own defaults.
    /// Returns false when the script failed to compile.
    pub fn attach(
        &mut self,
        index: usize,
        property: Option<&str>,
        script: &str,
        authored_props: Option<&Value>,
    ) -> bool {
        let src = format!(
            "__wp_register_instance({index}, {prop}, null, {authored}, {factory});",
            prop = property.map(js_str).unwrap_or_else(|| "null".into()),
            authored = authored_props
                .filter(|v| v.is_object())
                .map(|v| v.to_string())
                .unwrap_or_else(|| "null".into()),
            factory = factory_source(script),
        );
        match self.context.eval(Source::from_bytes(src.as_bytes())) {
            Ok(v) if v.as_number().is_some_and(|n| n >= 0.0) => {
                self.instances += 1;
                true
            }
            Ok(_) => false,
            Err(e) => {
                tracing::warn!(
                    target: "script",
                    "script on object {index} ({}) failed to compile: {e}",
                    property.unwrap_or("events")
                );
                false
            }
        }
    }

    /// Scene-wide values scripts read: `engine.canvasSize` (scene size),
    /// `engine.screenResolution`, and the wallpaper's user property values.
    pub fn set_scene_info(&mut self, canvas: [f32; 2], screen: [u32; 2], user_props: &Value) {
        let src = format!(
            "engine.canvasSize = new Vec2({}, {}); engine.screenResolution = new Vec2({}, {}); engine.userProperties = {};",
            js_num(canvas[0]),
            js_num(canvas[1]),
            screen[0],
            screen[1],
            if user_props.is_object() { user_props.to_string() } else { "{}".into() },
        );
        self.run(&src);
    }

    /// Expose scene.json's `general` settings (`bloom`, `camerashake`,
    /// `clearcolor`…) on `thisScene`, unwrapping `{"value": …}` entries.
    /// Read-only: changing them from a script has no effect yet.
    pub fn set_scene_general(&mut self, general: &Value) {
        let Some(m) = general.as_object() else {
            return;
        };
        let mut flat = serde_json::Map::new();
        for (k, v) in m {
            let v = match v {
                Value::Object(o) => o.get("value").cloned().unwrap_or(Value::Null),
                other => other.clone(),
            };
            if !v.is_null() && !k.starts_with("__") {
                flat.insert(k.clone(), v);
            }
        }
        self.run(&format!(
            "(function(g) {{ for (var k in g) {{ if (typeof thisScene[k] !== 'function') {{ thisScene[k] = __wp_raw_value(g[k]); }} }} }})({});",
            Value::Object(flat)
        ));
    }

    /// Live user-property change: updates `engine.userProperties` and calls
    /// every instance's `applyUserProperties(changed)`.
    pub fn apply_user_properties(&mut self, changed: &Value) {
        if changed.is_object() {
            self.run(&format!("__wp_set_user_properties({changed});"));
        }
    }

    /// Cursor state for `input.*`: scene-space position, screen pixels, and
    /// whether the left button is held.
    pub fn set_cursor(&mut self, world: [f32; 2], screen: [f32; 2], left_down: bool) {
        let src = format!(
            "input.cursorWorldPosition = new Vec3({}, {}, 0); input.cursorScreenPosition = new Vec2({}, {}); input.cursorLeftDown = {left_down};",
            js_num(world[0]),
            js_num(world[1]),
            js_num(screen[0]),
            js_num(screen[1]),
        );
        self.run(&src);
    }

    /// Whether a script called `engine.registerAudioBuffers`.
    pub fn wants_audio(&mut self) -> bool {
        self.run("__wp.audioWanted")
            .map(|v| v.to_boolean())
            .unwrap_or(false)
    }

    /// Feed this frame's spectrum to registered audio buffers.
    pub fn set_audio(&mut self, bands: [&[f32]; 6]) {
        let arr = |b: &[f32]| {
            format!(
                "[{}]",
                b.iter().map(|v| js_num(*v)).collect::<Vec<_>>().join(",")
            )
        };
        let src = format!(
            "__wp_set_audio({}, {}, {}, {}, {}, {});",
            arr(bands[0]),
            arr(bands[1]),
            arr(bands[2]),
            arr(bands[3]),
            arr(bands[4]),
            arr(bands[5]),
        );
        self.run(&src);
    }

    /// Update the `media` global and fire the `media*Changed` handlers.
    #[allow(clippy::too_many_arguments)]
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
             media.artUrl = {art_url_js}; \
             __wp_dispatch(-1, 'mediaPropertiesChanged', {{ title: media.title, artist: media.artist, albumTitle: media.album }}); \
             __wp_dispatch(-1, 'mediaPlaybackChanged', {{ state: media.playbackState }}); \
             __wp_dispatch(-1, 'mediaTimelineChanged', {{ position: media.position / 1e6, duration: media.duration / 1e6 }}); \
             __wp_dispatch(-1, 'mediaStatusChanged', {{ enabled: media.available }});",
            title = js_str(title),
            artist = js_str(artist),
            album = js_str(album),
        );
        self.run(&src);
    }

    /// Fire `event` (e.g. `"cursorClick"`) on the scripts of object `index`.
    /// `world` is the cursor's scene-space position.
    pub fn dispatch_cursor(&mut self, index: usize, event: &str, world: [f32; 2]) {
        let src = format!(
            "__wp_dispatch({index}, {event}, {{ worldPosition: new Vec3({x}, {y}, 0) }});",
            event = js_str(event),
            x = js_num(world[0]),
            y = js_num(world[1]),
        );
        self.run(&src);
    }

    /// Objects with at least one handler among `events`.
    pub fn objects_with_handlers(&mut self, events: &[&str]) -> Vec<usize> {
        let list = serde_json::to_string(events).unwrap_or_else(|_| "[]".into());
        self.run_string(&format!("JSON.stringify(__wp_handler_objects({list}))"))
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Run timers and every instance's `init`/`update` for this frame.
    /// `time_of_day` is `[0, 1)` — see [`local_time_of_day`].
    pub fn frame(&mut self, runtime: f32, time_of_day: f32, frametime: f32) {
        self.run(&format!(
            "__wp_frame({}, {}, {});",
            js_num(runtime),
            js_num(time_of_day),
            js_num(frametime)
        ));
    }

    /// Everything scripts changed since the last call.
    pub fn collect(&mut self) -> ScriptChanges {
        let Some(json) = self.run_string("__wp_collect()") else {
            return ScriptChanges::default();
        };
        let Ok(v) = serde_json::from_str::<Value>(&json) else {
            return ScriptChanges::default();
        };
        for line in v["logs"].as_array().into_iter().flatten() {
            if let Some(line) = line.as_str() {
                if cfg!(test) && std::env::var_os("WP_SCRIPT_SCENE").is_some() {
                    eprintln!("{line}");
                }
                if line.starts_with("error") {
                    tracing::warn!(target: "script", "{line}");
                } else {
                    tracing::debug!(target: "script", "{line}");
                }
            }
        }
        if let (Some(storage), Some(path)) = (v.get("storage"), &self.storage_path) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(e) = std::fs::write(path, storage.to_string()) {
                tracing::warn!(target: "script", "saving localStorage to {}: {e}", path.display());
            }
        }
        ScriptChanges {
            objects: v["objects"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(parse_object_change)
                .collect(),
            commands: v["commands"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(parse_command)
                .collect(),
        }
    }
}

/// Run `script` once against a single object, as on its first frame, and
/// return what it changed — for load-time values (a clock's initial text, a
/// spritesheet frame picked by the hour).
pub fn evaluate_once(
    property: &str,
    script: &str,
    init: &ObjectInit,
    authored_props: Option<&Value>,
) -> ObjectChange {
    let mut host = ScriptHost::new();
    host.register_object(0, init);
    if !host.attach(0, Some(property), script, authored_props) {
        return ObjectChange::default();
    }
    host.frame(0.0, local_time_of_day(), 0.0);
    host.collect()
        .objects
        .into_iter()
        .find(|(i, _)| *i == 0)
        .map(|(_, c)| c)
        .unwrap_or_default()
}

fn vec3(v: &Value) -> Option<[f32; 3]> {
    let a = v.as_array()?;
    let f = |i: usize| a.get(i).and_then(|x| x.as_f64()).map(|x| x as f32);
    let out = [f(0)?, f(1)?, f(2)?];
    out.iter().all(|c| c.is_finite()).then_some(out)
}

fn parse_object_change(entry: &Value) -> Option<(usize, ObjectChange)> {
    let index = entry.get(0)?.as_u64()? as usize;
    let map = entry.get(1)?.as_object()?;
    let rad = std::f32::consts::PI / 180.0;
    let mut c = ObjectChange::default();
    for (k, v) in map {
        let num = || v.as_f64().map(|n| n as f32).filter(|n| n.is_finite());
        match k.as_str() {
            "origin" => c.origin = vec3(v),
            "angles" => c.angles = vec3(v).map(|a| a.map(|d| d * rad)),
            "scale" => c.scale = vec3(v),
            "color" => c.color = vec3(v),
            "visible" => c.visible = v.as_bool(),
            "alpha" => c.alpha = num(),
            "brightness" => c.brightness = num(),
            "text" => c.text = v.as_str().map(str::to_string),
            "pointsize" => c.pointsize = num(),
            "volume" => c.volume = num(),
            "rate" => c.rate = num(),
            "frame" => c.frame = v.as_f64().filter(|f| *f >= 0.0).map(|f| f as u32),
            "propagationDisabled" => c.propagation_disabled = v.as_bool(),
            "size" | "parallaxDepth" | "font" => {}
            _ => {
                c.extra.insert(k.clone(), v.clone());
            }
        }
    }
    Some((index, c))
}

fn parse_command(cmd: &Value) -> Option<ScriptCommand> {
    let a = cmd.as_array()?;
    let kind = a.first()?.as_str()?;
    let index = a.get(1).and_then(|v| v.as_i64()).unwrap_or(-1);
    let idx = || (index >= 0).then_some(index as usize);
    let s = |i: usize| a.get(i).and_then(|v| v.as_str()).map(str::to_string);
    Some(match kind {
        "sound" => ScriptCommand::Sound {
            index: idx()?,
            op: s(2)?,
        },
        "texanim" => ScriptCommand::TextureAnimation {
            index: idx()?,
            op: s(2)?,
        },
        "anim" => ScriptCommand::Animation {
            index: idx()?,
            property: s(2)?,
            op: s(3)?,
            frame: a.get(4).and_then(|v| v.as_f64()).map(|f| f as f32),
        },
        "sort" => ScriptCommand::Sort {
            index: idx()?,
            to: a.get(2)?.as_f64()?.max(0.0) as usize,
        },
        "create" => ScriptCommand::Create {
            index: idx()?,
            config: a.get(2).cloned().unwrap_or(Value::Null),
        },
        "destroy" => ScriptCommand::Destroy { index: idx()? },
        "shortcut" => ScriptCommand::Shortcut(s(2)?),
        _ => return None,
    })
}

/// The scripting kind of a scene.json value: `"number"`, `"bool"`,
/// `"vec2"`/`"vec3"`/`"vec4"` (numeric strings like `"1 0 0"`), or `"string"`.
pub fn value_kind(v: &Value) -> &'static str {
    match v {
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(s) => {
            let parts: Vec<&str> = s.split_whitespace().collect();
            let numeric = parts.iter().all(|p| p.parse::<f64>().is_ok());
            match (numeric, parts.len()) {
                (true, 1) => "number",
                (true, 2) => "vec2",
                (true, 3) => "vec3",
                (true, 4) => "vec4",
                _ => "string",
            }
        }
        _ => "number",
    }
}

/// The `(property, script, scriptproperties)` triples of a scene.json value
/// shaped `{"value": …, "script": "…", "scriptproperties": {…}}`.
pub fn scripted(value: Option<&Value>) -> Option<(&str, Option<&Value>)> {
    let m = value?.as_object()?;
    let script = m.get("script")?.as_str()?;
    Some((script, m.get("scriptproperties")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_with_object() -> ScriptHost {
        let mut h = ScriptHost::new();
        h.register_object(
            0,
            &ObjectInit {
                name: Some("clock".into()),
                id: Some(7),
                origin: [100.0, 200.0, 0.0],
                scale: [1.0, 1.0, 1.0],
                color: [1.0, 1.0, 1.0],
                visible: true,
                alpha: 1.0,
                brightness: 1.0,
                pointsize: 32.0,
                volume: 1.0,
                ..Default::default()
            },
        );
        h
    }

    fn change(h: &mut ScriptHost) -> ObjectChange {
        h.collect()
            .objects
            .into_iter()
            .find(|(i, _)| *i == 0)
            .map(|(_, c)| c)
            .unwrap_or_default()
    }

    #[test]
    fn module_state_persists_and_init_runs_once() {
        let mut h = host_with_object();
        let script = "'use strict';\nlet n = 0;\nexport function init(v) { n = 10; }\nexport function update(value) { n += 1; return n / 100; }";
        assert!(h.attach(0, Some("alpha"), script, None));
        h.frame(0.0, 0.0, 0.016);
        assert_eq!(change(&mut h).alpha, Some(0.11));
        h.frame(0.016, 0.0, 0.016);
        assert_eq!(change(&mut h).alpha, Some(0.12));
    }

    #[test]
    fn unchanged_values_are_not_reported() {
        let mut h = host_with_object();
        h.attach(
            0,
            Some("alpha"),
            "export function update(v) { return 1; }",
            None,
        );
        h.frame(0.0, 0.0, 0.016);
        assert!(h.collect().objects.is_empty());
    }

    #[test]
    fn angles_are_degrees_in_scripts_and_radians_out() {
        let mut h = host_with_object();
        h.attach(
            0,
            Some("angles"),
            "export function update(value) { value.z = -90; return value; }",
            None,
        );
        h.frame(0.0, 0.0, 0.016);
        let a = change(&mut h).angles.unwrap();
        assert!((a[2] + std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    }

    #[test]
    fn scripts_write_other_layers_and_this_layer() {
        let mut h = host_with_object();
        h.register_object(
            1,
            &ObjectInit {
                name: Some("label".into()),
                visible: true,
                ..Default::default()
            },
        );
        h.attach(
            0,
            Some("visible"),
            "export function update(value) { thisScene.getLayer('label').text = 'hi'; thisLayer.alpha = 0.5; return value; }",
            None,
        );
        h.frame(0.0, 0.0, 0.016);
        let changes = h.collect();
        let get = |i: usize| {
            changes
                .objects
                .iter()
                .find(|(j, _)| *j == i)
                .unwrap()
                .1
                .clone()
        };
        assert_eq!(get(1).text.as_deref(), Some("hi"));
        assert_eq!(get(0).alpha, Some(0.5));
    }

    #[test]
    fn vector_getters_return_copies() {
        let mut h = host_with_object();
        h.attach(
            0,
            Some("alpha"),
            "export function update(v) { let o = thisLayer.origin; o.x = 999; return v; }",
            None,
        );
        h.frame(0.0, 0.0, 0.016);
        assert_eq!(change(&mut h).origin, None);
    }

    #[test]
    fn timers_fire_on_schedule() {
        let mut h = host_with_object();
        h.attach(
            0,
            None,
            "export function init() { engine.setTimeout(function() { thisLayer.visible = false; }, 500); }",
            None,
        );
        h.frame(0.0, 0.0, 0.016);
        assert_eq!(change(&mut h).visible, None);
        h.frame(0.6, 0.0, 0.016);
        assert_eq!(change(&mut h).visible, Some(false));
    }

    #[test]
    fn audio_buffers_are_fed() {
        let mut h = host_with_object();
        h.attach(
            0,
            Some("alpha"),
            "let buf = engine.registerAudioBuffers(engine.AUDIO_RESOLUTION_16);\nexport function update(v) { return buf.average[0]; }",
            None,
        );
        assert!(h.wants_audio());
        let l16 = [0.5f32; 16];
        let r16 = [0.25f32; 16];
        h.set_audio([&l16, &r16, &[0.0; 32], &[0.0; 32], &[0.0; 64], &[0.0; 64]]);
        h.frame(0.0, 0.0, 0.016);
        assert_eq!(change(&mut h).alpha, Some(0.375));
    }

    #[test]
    fn cursor_events_reach_the_target_object_only() {
        let mut h = host_with_object();
        h.register_object(
            1,
            &ObjectInit {
                visible: true,
                ..Default::default()
            },
        );
        h.attach(
            0,
            Some("visible"),
            "export function cursorClick(e) { thisLayer.visible = !thisLayer.visible; }",
            None,
        );
        assert_eq!(h.objects_with_handlers(CURSOR_EVENTS), vec![0]);
        h.dispatch_cursor(1, "cursorClick", [0.0, 0.0]);
        assert!(h.collect().objects.is_empty());
        h.dispatch_cursor(0, "cursorClick", [0.0, 0.0]);
        assert_eq!(change(&mut h).visible, Some(false));
    }

    #[test]
    fn script_properties_merge_authored_values_and_user_props_apply() {
        let mut h = host_with_object();
        h.set_scene_info(
            [1920.0, 1080.0],
            [1920, 1080],
            &serde_json::json!({"hudcolor": "1 0 0"}),
        );
        let script = "export var scriptProperties = createScriptProperties().addCheckbox({name: 'h24', value: false}).finish();\n\
                      let c = '';\n\
                      export function applyUserProperties(p) { if (p.hasOwnProperty('hudcolor')) { c = p.hudcolor; } }\n\
                      export function update(value) { return (scriptProperties.h24 ? '24' : '12') + c; }";
        h.attach(
            0,
            Some("text"),
            script,
            Some(&serde_json::json!({"h24": {"value": true}})),
        );
        h.frame(0.0, 0.0, 0.016);
        assert_eq!(change(&mut h).text.as_deref(), Some("241 0 0"));
    }

    #[test]
    fn sound_and_create_commands_are_reported() {
        let mut h = host_with_object();
        h.attach(
            0,
            None,
            "export function init() { thisLayer.play(); var l = thisScene.createLayer(thisScene.getInitialLayerConfig(thisLayer)); thisScene.destroyLayer(l); }",
            None,
        );
        h.frame(0.0, 0.0, 0.016);
        let cmds = h.collect().commands;
        assert_eq!(
            cmds[0],
            ScriptCommand::Sound {
                index: 0,
                op: "play".into()
            }
        );
        assert!(matches!(cmds[1], ScriptCommand::Create { index: 1, .. }));
        assert_eq!(cmds[2], ScriptCommand::Destroy { index: 1 });
    }

    #[test]
    fn a_throwing_script_keeps_running_others() {
        let mut h = host_with_object();
        h.attach(
            0,
            Some("alpha"),
            "export function update(v) { throw new Error('x'); }",
            None,
        );
        h.attach(
            0,
            Some("brightness"),
            "export function update(v) { return 2; }",
            None,
        );
        h.frame(0.0, 0.0, 0.016);
        let c = change(&mut h);
        assert_eq!(c.alpha, None);
        assert_eq!(c.brightness, Some(2.0));
    }

    #[test]
    fn syntax_errors_fail_to_attach() {
        let mut h = host_with_object();
        assert!(!h.attach(
            0,
            Some("alpha"),
            "export function update(v) { return ; ) }",
            None
        ));
        assert!(!h.has_scripts());
    }

    #[test]
    fn named_imports_become_bindings() {
        assert_eq!(
            import_to_bindings("{ smoothStep, mix as m } from 'WEMath';"),
            "var smoothStep = WEMath.smoothStep, m = WEMath.mix;"
        );
        assert_eq!(import_to_bindings("* as WEMath from 'WEMath';"), "");
        assert_eq!(
            import_to_bindings("* as M from 'WEMath';"),
            "var M = WEMath;"
        );
    }

    #[test]
    fn minified_inner_exports_are_stripped() {
        let body =
            module_to_body("'use strict';export var x=1;function a(){}export function update(){}");
        assert!(!body.contains("export"));
    }

    #[test]
    fn color_helpers_round_trip() {
        let mut h = host_with_object();
        h.attach(
            0,
            Some("color"),
            "export function update(v) { return WEColor.hsv2rgb(WEColor.rgb2hsv(new Vec3(0.2, 0.4, 0.6))); }",
            None,
        );
        h.frame(0.0, 0.0, 0.016);
        let c = change(&mut h).color.unwrap();
        assert!(
            (c[0] - 0.2).abs() < 1e-5 && (c[1] - 0.4).abs() < 1e-5 && (c[2] - 0.6).abs() < 1e-5
        );
    }

    #[test]
    fn local_storage_persists_to_disk() {
        let path = std::env::temp_dir().join(format!("wp-ls-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&path);
        {
            let mut h = ScriptHost::new().with_storage(path.clone());
            h.register_object(0, &ObjectInit::default());
            h.attach(
                0,
                None,
                "export function init() { localStorage.set('n', 3); }",
                None,
            );
            h.frame(0.0, 0.0, 0.016);
            h.collect();
        }
        let mut h = ScriptHost::new().with_storage(path.clone());
        h.register_object(
            0,
            &ObjectInit {
                alpha: 1.0,
                ..Default::default()
            },
        );
        h.attach(
            0,
            Some("alpha"),
            "export function update(v) { return localStorage.get('n'); }",
            None,
        );
        h.frame(0.0, 0.0, 0.016);
        assert_eq!(change(&mut h).alpha, Some(3.0));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn scene_general_settings_are_readable() {
        let mut h = host_with_object();
        h.set_scene_general(
            &serde_json::json!({"bloom": {"value": true}, "bloomstrength": 2.5, "getLayer": 1}),
        );
        h.attach(
            0,
            Some("text"),
            "export function update(v) { return thisScene.bloom + ' ' + thisScene.bloomstrength + ' ' + typeof thisScene.getLayer; }",
            None,
        );
        h.frame(0.0, 0.0, 0.016);
        assert_eq!(change(&mut h).text.as_deref(), Some("true 2.5 function"));
    }

    fn host_with_effects() -> ScriptHost {
        let mut h = ScriptHost::new();
        h.register_object(
            0,
            &ObjectInit {
                name: Some("dots".into()),
                visible: true,
                raw: Some(serde_json::json!({
                    "instanceoverride": {"rate": 0.5},
                    "effects": [
                        {"file": "effects/pulse/effect.json", "passes": [{"constantshadervalues": {"speed": 1.0}}]},
                        {"file": "effects/scroll/effect.json", "name": "scroll", "passes": [{"constantshadervalues": {"tint": "1 0 0"}}]}
                    ]
                })),
                ..Default::default()
            },
        );
        h
    }

    #[test]
    fn effect_materials_and_instance_are_scriptable() {
        let mut h = host_with_effects();
        h.attach(
            0,
            Some("visible"),
            "export function update(v) {\n\
               let m = thisLayer.getEffect(1).getMaterial(0);\n\
               let t = m.tint; t.y = 0.5; m.tint = t;\n\
               thisLayer.getEffect('pulse').getMaterial(0).speed = m.tint.y * 4;\n\
               thisLayer.getEffect(0).visible = false;\n\
               thisLayer.instance.rate = thisLayer.instance.rate * 2;\n\
               return v; }",
            None,
        );
        h.frame(0.0, 0.0, 0.016);
        let c = change(&mut h);
        assert_eq!(c.extra["effects.1.0.tint"], serde_json::json!([1, 0.5, 0]));
        assert_eq!(c.extra["effects.0.0.speed"], serde_json::json!(2));
        assert_eq!(c.extra["effects.0.visible"], serde_json::json!(false));
        assert_eq!(c.extra["instance.rate"], serde_json::json!(1));
    }

    #[test]
    fn declared_extra_properties_drive_update_scripts() {
        let mut h = host_with_effects();
        h.declare_property(0, "instance.rate", "number", &serde_json::json!(0.5));
        h.attach(
            0,
            Some("instance.rate"),
            "export function update(v) { return v + 1; }",
            None,
        );
        h.frame(0.0, 0.0, 0.016);
        assert_eq!(
            change(&mut h).extra["instance.rate"],
            serde_json::json!(1.5)
        );
        assert_eq!(value_kind(&serde_json::json!("1 0 0")), "vec3");
        assert_eq!(value_kind(&serde_json::json!("0.5")), "number");
    }

    #[test]
    fn time_of_day_is_a_fraction() {
        let t = local_time_of_day();
        assert!((0.0..1.0).contains(&t));
    }
}
