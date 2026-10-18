# Feature-gap audit: wp-engine vs. Wallpaper Engine

Audit (taken at commit `6906b93`) of what this project lacked to be a
complete, drop-in replacement for the Wallpaper Engine **client** (the
playback app, not the editor). Each item cites where the gap lived in the
code. The **Status** column tracks the work on the `beta` branch.

Priorities: **P0** breaks or degrades real Workshop content · **P1** core
client feature users expect · **P2** parity / polish · **OOS** out of scope.

## 1. Already solid at audit time

`.pkg`/`.tex`/`.mdl` loading, the scene pass model (ping-pong FBOs, named
targets, bloom), GLSL→WGSL translation, particles, puppet `.mdl` skinning,
text layers, lights (point/spot/directional/tube, shadows, volumetrics),
audio-reactive uniforms, camera shake/fade/parallax, user properties incl.
`condition` refs, MPRIS media, Wayland layer-shell / X11 root-pixmap
presentation, per-screen content, quality tiers, pause on
battery/lock/fullscreen/maximized, a single-instance daemon, and an egui
browser.

## 2. SceneScript

| Pri | Gap | Where |
| --- | --- | --- |
| P0 | Script state didn't persist between frames: every evaluation re-wrapped the script in an IIFE and re-ran `init()` each frame. | `engine/script.rs` `iife_body` |
| P0 | Cursor events (`cursorClick`/`Down`/`Up`/`Move`/`Enter`/`Leave`) were never dispatched; `input.cursor*` was fixed at the origin. | `engine/script.rs`, `gpu_renderer.rs` |
| P1 | Scriptable properties limited to alpha/visible/scale/origin/angles/text/frame (no color, shader constants, particle instance, sound). | `gpu_renderer.rs` |
| P1 | Scene API gaps: layer create/destroy, sound/animation/video/particle control. | `engine/script.rs` |
| P1 | `registerTimeEvent` no-op; `screenResolution`, `canvasSize`, `isDesktopDevice`/`isWallpaper`/`isScreensaver`, `localStorage` missing. | `engine/script.rs` |
| P2 | Math library partial: no `Vec2`/`Vec4`, `WEColor`, `WEVector`, few `WEMath` helpers. | `engine/script.rs` |

## 3. Scene renderer

| Pri | Gap | Where |
| --- | --- | --- |
| P0 | Keyframed property timeline animations (`"animation": {"c0": …}`) were not parsed or played. | `engine/scene.rs` |
| P0 | Screen-space backdrop capture for refraction/water effects. | `gpu_renderer.rs` |
| P1 | Per-object cameras / 3D camera paths. | — |
| P1 | Normal-mapped 2D lighting, reflections, HDR output, MSAA. | — |

## 4. Content types

| Pri | Gap | Where |
| --- | --- | --- |
| P1 | `preset` workshop items fell into `Unknown`. | `workshop/mod.rs` |
| P1 | `dependency` items (assets from another workshop item) not mounted. | `engine/assets.rs` |
| P1 | Video: no audio, always stretched, re-open hitch on loop, CPU decode only. Images also stretched. | `render/ffmpeg.rs`, `platform/scaler.rs` |
| P1 | Web: fixed 1920×1080; bridge missing `applyGeneralProperties`, `setPaused`, the `wallpaperRegisterMedia*Listener` family; no keyboard/scroll input. | `render/web.rs` |
| OOS | `application` (Windows .exe) wallpapers. | — |

## 5. Playback and performance

| Pri | Gap | Where |
| --- | --- | --- |
| P0 | Volume setting stored but never applied. | `render/settings.rs` |
| P1 | Playlists persisted but never rotated. | `settings.rs` |
| P1 | FPS hard-coded to 30. | `x11/mod.rs`, `render/frame.rs` |
| P1 | Pause policy lacked "other app focused" and "stop" modes. | `engine/playback_gate.rs` |
| P1 | No span/clone multi-monitor modes. | `render/content.rs` |

## 6. Steam Workshop

| Pri | Gap | Where |
| --- | --- | --- |
| P1 | Library discovery didn't read `libraryfolders.vdf`; missed secondary libraries and Flatpak/Snap Steam. | `workshop/mod.rs` |
| P1 | No Workshop browsing/downloading. | — |

## 7. Desktop app

| Pri | Gap | Where |
| --- | --- | --- |
| P1 | GUI had no property editor, playlist editor or monitor picker. | `ui/mod.rs` |
| P1 | No tray icon, hotkeys or autostart install. | — |

## 8. Platforms

| Pri | Gap | Where |
| --- | --- | --- |
| P1 | macOS: scenes only, single display, no pause, no daemon. | `platform/macos/mod.rs` |
| OOS | Windows. | — |

## 9. Engineering hygiene

| Pri | Gap | Where |
| --- | --- | --- |
| P1 | No CI. | `.github/` |
| P2 | README "Known gaps" stale; badges/clone URL pointed at `wp-engine`. | `README.md` |
