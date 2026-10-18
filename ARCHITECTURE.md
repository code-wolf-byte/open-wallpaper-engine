# Architecture

How a wallpaper becomes pixels, stage by stage. This is the companion to the
[README](README.md) — read that first for what the project does and how to run
it.

The scene engine is deliberately split into small stages so Wallpaper Engine's
original OpenGL pass model can be rebuilt incrementally in Rust, and so each
stage can be inspected on its own when a wallpaper renders wrongly.

## Source layout

```text
src/
  main.rs, bin/        the wp-engine CLI and the scan-wallpapers tool
  workshop/            finding Workshop items; project.json and user properties
  formats/             .pkg archives, .tex textures, the asset store
  engine/              the scene engine
    model/             what a scene is: scene.json types, resolving into layers
    sim/               what changes: particles, timelines, SceneScript, cameras, puppets
    gpu/               how it's drawn: the renderer, targets, lighting, shadows, shaders
    cpu/               the CPU fallback renderer
  content/             wallpaper content (scene/video/web/image), frame sources,
                       render settings, the per-output session the backends share
  services/            what wallpapers react to: audio, media, battery/lock, pause rules
  platform/            presentation: wayland/, x11/, macos/, GPU device, scaler
  daemon/, application/, settings.rs, schedule.rs   lifecycle and saved setup
  ui/                  the desktop app
```

## Pipeline

### 1. Workshop Content Detection

The app starts from either a Steam Workshop item or a direct file path.

- `src/workshop` scans Wallpaper Engine Workshop projects.
- `src/content/wallpaper.rs` classifies content as static image, video, scene, or
  web.
- Static, video, and web wallpapers flow through the frame-source path.
- Scene wallpapers are handed to the engine pipeline below.

### 2. Asset Store

Scene assets can exist as loose files or packed archives.

- `src/formats/assets.rs` mounts the wallpaper directory.
- Loose files are preferred.
- `scene.pkg` and `gifscene.pkg` are loaded as fallbacks.
- Asset paths are normalized to Wallpaper Engine's forward-slash layout.
- Textures are resolved from material-style names such as `foo` to candidates
  like `materials/foo.tex`.

This stage is the Rust equivalent of the original engine's asset/container
lookup layer.

### 3. Scene Parsing

The engine reads `scene.json` into loose, tolerant Rust structs.

- `src/engine/model/scene.rs` parses objects, camera data, effects, visibility,
  origins, sizes, parallax, and texture overrides.
- Fields that vary between Wallpaper Engine versions are kept as
  `serde_json::Value` until a later stage knows how to interpret them.

The goal is to avoid rejecting valid wallpapers just because a creator used a
slightly different schema.

### 4. Model, Material, And Effect Loading

The scene graph resolves object references into typed engine data.

- `src/engine/cpu/material.rs` loads model JSON files for the CPU path.
- Models point at material JSON files.
- Materials contain passes, shaders, texture bindings, combos, constants,
  blending, culling, and depth flags.
- Effect JSON files are loaded into effect passes and FBO declarations.

This mirrors the original engine's structure:

```text
scene object
  -> model
    -> material
      -> material pass
        -> shader + textures + constants
  -> effects
    -> effect pass
      -> material/command/source/target
```

### 5. Scene Graph

`src/engine/cpu/graph.rs` builds a resolved scene graph.

For each visible object it records:

- the source scene object
- resolved model, if the object references one
- resolved base texture
- resolved effects
- render-target/fullscreen hints
- parent chain and transforms
- the draw order (`order_index`) both render paths share, so images, particles,
  and meshes interleave in true scene z-order rather than by category

The scene graph is backend-neutral, and useful for both diagnostics and
rendering.

### 6. Shader Translation Layer

`src/engine/gpu/shaders/transpiler/` turns Wallpaper Engine's GLSL into WGSL:
resolve `shaders/<name>.vert|frag`, expand includes and combo `#define`s,
compile to SPIR-V with shaderc, then translate to WGSL with Naga.

Real workshop shaders lean on legacy NVIDIA GLSL leniency, so the translator
also repairs them: vector-width coercion, `for`-loop unrolling ahead of array
varying expansion, `%` to `mod`, int literals to floats in `min`/`max`/`clamp`,
and float-to-int assignment casts. Engine uniforms (`g_Time`,
`g_PointerPosition`, the `g_AudioSpectrum*` arrays, …) are packed to std140 and
bound alongside.

`WP_DEBUG_DUMP_GLSL=1` writes the generated GLSL of any shader that fails to
compile — shaderc's line numbers refer to that text and nothing else prints it.
`WP_DEBUG_DUMP_WGSL=1` dumps each pass's translated WGSL.

### 7. GPU Scene Renderer

`src/engine/gpu/renderer/` is the production renderer — `GpuSceneInstance` and
`gpu_scene_render_loop`, used by `set`, `set-file`, and `test-scene`.

It owns the full pass model: per-object composite buffers, ping-pong FBOs,
named effect targets (`target`/`bind`), multi-pass effects, the reference's
bloom chain, particle systems, puppet skinning, 3D meshes for perspective
scenes, text layers, and SceneScript-driven properties ticked per frame.

`WP_DEBUG_DUMP_FBOS=1` dumps every pooled render target per frame;
`WP_ENGINE_SKIP_EFFECTS=name,name` disables effects by name;
`WP_DEBUG_DUMP_FRAME=path` saves the last GPU frame headlessly, which is the
only way to inspect the live path off-screen.

### 8. Web Wallpapers

`src/content/web.rs` renders HTML wallpapers with an embedded Chromium through
CEF, off-screen: each painted frame becomes an `RgbaImage` on the same channel
video and CPU scene rendering use, so every presentation path works unchanged.

It also injects the Wallpaper Engine browser API that pages expect —
`wallpaperPropertyListener.applyUserProperties` fed from `project.json`, and
`wallpaperRegisterAudioListener` fed from live capture as 64 left plus 64 right
bands. Behind the off-by-default `web` feature; see **Building**.

### 9. CPU Fallback Renderer

`ResolvedScene::render` (`src/engine/model/resolve/`) keeps the older CPU
compositor alive, with its frame loop and scene graph in `src/engine/cpu/`.

It:

- resolves visible image layers
- decodes texture/image assets to RGBA
- resizes each layer to the output size
- overlays layers into one `RgbaImage`

Run it with:

```bash
cargo run -- render-scene <workshop-id-or-directory> --output scene_output.png
```

The CPU path is not feature-equivalent to Wallpaper Engine, but it remains useful
for smoke tests and as a stable fallback while the `wgpu` renderer grows.

### 10. Live Output

`src/platform/display.rs` picks a backend at runtime: Wayland when
`WAYLAND_DISPLAY`/`WAYLAND_SOCKET` is set, else X11 when `DISPLAY` is, else
macOS. `WP_ENGINE_FORCE_X11=1` takes the X11 path even under a Wayland session.

On **Wayland**, scene wallpapers render on the GPU and present *directly into
the layer-shell surface* through a `wgpu::Surface` — no CPU readback, no SHM
copy. `src/platform/wayland` creates the surface from the `wl_surface` raw
handle; if that fails (or `WP_ENGINE_FORCE_SHM` is set) it falls back to RGBA
readback + SHM buffers. The same module feeds the real cursor position into
camera parallax and `g_PointerPosition`.

On **X11**, `src/platform/x11` draws each frame into a root-sized pixmap and
publishes it as the root background (`_XROOTPMAP_ID`/`ESETROOT_PMAP_ID`, what
`feh --bg` sets). Desktop environments cover the root with their own desktop, so
when the window manager supports EWMH desktop windows the same pixmap also backs
a click-through `_NET_WM_WINDOW_TYPE_DESKTOP` window of ours. Frames are read
back and pushed with `PutImage`; there is no direct-presentation path yet. X11 gets
better parallax for free: `QueryPointer` reports the global cursor regardless of
which window has focus.

Both the X11 and Wayland backends also implement the real Wallpaper Engine's
own "pause wallpaper rendering when..." settings (`services::playback_gate` —
recovered from the Ghidra dump's import table, not the usual property scan;
see the report's Follow-up (ff)): on battery power, when the session is
locked, and when another application is fullscreen or maximized (X11 via
EWMH's `_NET_ACTIVE_WINDOW`/`_NET_WM_STATE`; Wayland via
`zwlr_foreign_toplevel_management`, so it only works on compositors that
support that wlroots-ecosystem protocol — Sway/Hyprland/river, not
GNOME/Mutter). Session-lock detection is Linux-only (logind over D-Bus, via
`services::power::LockWatcher`). Each condition defaults per
`services::playback_gate::PlaybackPolicy::default`'s own doc comment (WE's
real defaults for these aren't recoverable from the binary) and is
overridable with `WP_ENGINE_PAUSE_ON_BATTERY`/`_LOCK`/`_FULLSCREEN`/
`_MAXIMIZED=0`/`1`. macOS currently has none of this wired in (see
`platform::macos`'s own "step 2 of the port" status note) — `services::
power::is_on_battery` already works there (via `pmset`) for whenever that
backend's render loop is built out far enough to use it.

Static images, videos, and web wallpapers flow through `src/content/frame.rs`
frame sources. What every output does whichever backend shows it — building its
renderer, live property edits, the pause rules — lives once in
`src/content/session.rs`.

### 11. Application Lifecycle

`src/application` owns the run flow (the Rust counterpart of the C++
`WallpaperApplication`): `ApplicationContext` carries the background path,
`--set-property` overrides, and shared render settings;
`WallpaperApplication::setup()/show()` resolve content, spawn the platform
renderer, and block until SIGINT/SIGTERM. `main.rs` `set`/`set-file`
delegate to it.

On Linux `src/daemon` makes that lifecycle a per-session singleton: `wp-engine
daemon` takes an exclusive `flock` on `$XDG_RUNTIME_DIR/wp-engine/daemon.lock`,
listens on `daemon.sock`, and owns the one `WallpaperApplication`. `set`,
`set-file`, `run`, and the GUI are clients — they serialize their
`ApplicationContext` into an `ApplySpec` (one JSON line per connection),
spawning the daemon detached (`setsid`) if it isn't running, and the daemon
drops the old application before building the new one, so applies replace
rather than stack. The GUI forwards live quality/volume changes the same way.
`--foreground` keeps the in-process path; macOS always uses it.

### 12. User Properties

`src/workshop/properties.rs` loads `project.json` `general.properties`,
applies `--set-property NAME=VALUE` overrides, and rewrites `{"user": ...}`
references inside `scene.json` before parsing — the Rust counterpart of the
reference `PropertyParser`/`UserSettingParser`. `wp-engine list-properties
<id>` lists what a wallpaper exposes.

### 13. Camera Dynamics

`src/engine/sim/camera_dynamics.rs` implements scene-level `camerashake`,
`camerafade`, and `cameraparallax` from the resolved general settings. Shake
and per-layer parallax feed the composite pass as UV offsets; fade drives
global scene opacity.
