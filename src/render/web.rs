//! Web (HTML) wallpapers, rendered by an embedded Chromium through CEF.
//!
//! CEF runs windowless (off-screen rendering): Chromium paints into a CPU BGRA
//! buffer instead of a real window, and each painted frame is pushed down the
//! same `SyncSender<Arc<RgbaImage>>` that video and CPU scene rendering use.
//! Every presentation path — Wayland SHM, the X11 root pixmap, the GPU scaler
//! — therefore works unchanged; a web wallpaper is just another frame producer.
//!
//! The whole module is behind the off-by-default `web` cargo feature, because
//! building it downloads the CEF binary distribution (~400 MB extracted) and
//! the resulting binary needs `libcef.so` plus Chromium's resource blobs beside
//! it at runtime. Without the feature the stubs below report that clearly
//! rather than silently doing nothing.

use anyhow::Result;
use image::RgbaImage;
use std::path::Path;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::Arc;

/// Off-screen render size for web wallpapers.
///
/// ponytail: fixed 1080p. CEF needs explicit dimensions up front and
/// `FrameSource` is built before any output is known, so the platform scaler
/// resizes to each monitor exactly as it does for a 1080p video. Plumb real
/// output dimensions through `FrameSource::from_content` if someone runs these
/// on a 4K panel and complains about softness.
pub const WEB_WIDTH: u32 = 1920;
pub const WEB_HEIGHT: u32 = 1080;

/// Start rendering `html` and return its first frame, a stream of the rest,
/// and a sender the caller can push mouse input into (see [`WebInputEvent`]).
///
/// Mirrors `render::ffmpeg::video_decode_loop`'s contract: the receiver yields
/// frames until the sender is dropped, and blocking for the first frame means
/// callers never present a blank surface.
pub fn start_web_stream(
    html: &Path,
) -> Result<(RgbaImage, Receiver<Arc<RgbaImage>>, SyncSender<WebInputEvent>)> {
    imp::start_web_stream(html)
}

/// `true` when this binary can actually render web wallpapers.
pub fn is_supported() -> bool {
    cfg!(feature = "web")
}

/// A mouse input event to forward into the embedded page — mirrors exactly
/// what the C++ reference itself forwards (`CWeb::updateMouse`): move plus
/// left/right click. Nothing else (scroll, middle click, keyboard) is
/// forwarded there either — its own `// TODO: ANY OTHER MOUSE EVENTS TO
/// SEND?` confirms that's the real engine's actual current scope, not an
/// approximation. Coordinates are normalized `[0,1]²` (top-left origin,
/// matching every other consumer of the platform layer's own pointer
/// tracking) so callers never need to know CEF's fixed pixel canvas size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WebInputEvent {
    MouseMove { x_norm: f32, y_norm: f32 },
    MouseButton { x_norm: f32, y_norm: f32, button: WebMouseButton, pressed: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebMouseButton {
    Left,
    Right,
}

/// Handle a CEF subprocess launch, if this process is one.
///
/// CEF starts its renderer/GPU/utility processes by re-executing this same
/// binary with `--type=...`. `main` must call this before parsing arguments and
/// exit with the returned code when it is `Some` — otherwise the child runs the
/// whole wallpaper app instead of a Chromium subprocess.
pub fn subprocess_main() -> Option<i32> {
    #[cfg(feature = "web")]
    {
        imp::subprocess_main()
    }
    #[cfg(not(feature = "web"))]
    {
        None
    }
}

#[cfg(not(feature = "web"))]
mod imp {
    use super::*;
    use anyhow::anyhow;

    pub fn start_web_stream(
        html: &Path,
    ) -> Result<(RgbaImage, Receiver<Arc<RgbaImage>>, SyncSender<WebInputEvent>)> {
        Err(anyhow!(
            "cannot render web wallpaper {}: this build has no web support.\n\
             Rebuild with `cargo build --features web` (downloads the CEF/Chromium \
             runtime, ~400 MB; set CEF_PATH to reuse an existing distribution).",
            html.display()
        ))
    }
}

#[cfg(feature = "web")]
mod imp {
    use super::*;
    use anyhow::{anyhow, Context};
    // Glob imports: the `wrap_*!` macros expand to impls of `WrapClient` /
    // `ImplClient` / `WrapRenderHandler` / `ImplRenderHandler` and the `Rc`
    // machinery, all of which must be in scope at the expansion site.
    use cef::rc::*;
    use cef::*;
    use cef::{args::Args, wrap_client, wrap_render_handler};
    use std::fmt::Write as _;
    use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};

    /// How long to wait for Chromium to paint the first frame before giving up.
    const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(30);

    /// Work item for the CEF thread. CEF may only be initialised once per
    /// process and its message loop must be pumped from the one thread that
    /// initialised it, so every browser is created there rather than on the
    /// caller's thread.
    struct OpenRequest {
        url: String,
        frames: SyncSender<Arc<RgbaImage>>,
        /// project.json's `general.properties`, already serialised, ready to
        /// hand to the page's `applyUserProperties`.
        properties: String,
        /// Whether this wallpaper calls `wallpaperRegisterAudioListener`. Only
        /// then do we open a capture device — a page that ignores audio has no
        /// business making us grab the desktop's output stream.
        wants_audio: bool,
        /// Whether this wallpaper calls `wallpaperMediaIntegration.listen`.
        /// Same reasoning as `wants_audio`: only then do we start the D-Bus
        /// MPRIS watcher (`engine::media`).
        wants_media: bool,
        /// Mouse events the platform layer pushes in — drained once per
        /// message-loop tick and forwarded to CEF's `BrowserHost`. See
        /// `WebInputEvent`.
        input_rx: Receiver<WebInputEvent>,
    }

    /// The Wallpaper Engine browser API, as much of it as this corpus uses.
    ///
    /// Injected into every frame before page scripts run. `wallpaperPropertyListener`
    /// is an accessor rather than a plain slot on purpose: properties are pushed
    /// from Rust as soon as the browser exists, which usually beats the page
    /// assigning its listener, so the setter re-delivers whatever arrived early.
    /// Without that the common case is a silent no-op.
    const BOOTSTRAP_JS: &str = r#"
(function () {
  if (window.__wpEngineBridge) return;
  var pendingProps = null, propListener = null, audioCb = null;
  var mediaListener = null, pendingMedia = null;

  Object.defineProperty(window, 'wallpaperPropertyListener', {
    configurable: true,
    get: function () { return propListener; },
    set: function (v) {
      propListener = v;
      if (v && pendingProps && typeof v.applyUserProperties === 'function') {
        try { v.applyUserProperties(pendingProps); } catch (e) { console.error(e); }
      }
    }
  });

  window.wallpaperRegisterAudioListener = function (cb) { audioCb = cb; };

  // Rust bundles a `__files` array (file:// URLs) onto any `directory`-type
  // property whose current value resolves to a real, readable directory
  // (see `user_properties_json`) — pick one at random here rather than a
  // real native round-trip per call; same user-visible behavior (a
  // randomized slideshow) for confirmed real usage (workshop item
  // 893418273), far less machinery than the genuine async CEF IPC a
  // literal port of the real per-call API would need.
  window.wallpaperRequestRandomFileForProperty = function (name, cb) {
    if (!pendingProps) return;
    var prop = pendingProps[name];
    var files = prop && prop.__files;
    if (!files || !files.length) return;
    var file = files[Math.floor(Math.random() * files.length)];
    try { cb(name, file); } catch (e) { console.error(e); }
  };

  // Real shape confirmed from the vendored C++ reference's ScriptEngine.cpp
  // (`notifyMediaUpdate`) — WE calls all four callbacks a listener defines,
  // every time *any* of them changes, not just the one that actually did.
  // `primaryColor`/`secondaryColor`/`tertiaryColor`/`highContrastColor` are
  // fixed defaults there too (its own `// TODO: PROCESS THESE COLORS
  // INSTEAD OF HARDCODING THEM`) — matched here rather than attempting real
  // album-art color extraction the reference itself doesn't have either.
  window.wallpaperMediaIntegration = {
    listen: function (cb) {
      mediaListener = cb;
      if (pendingMedia) { deliverMedia(pendingMedia); }
    }
  };
  function deliverMedia(m) {
    if (!mediaListener) { return; }
    var calls = [
      ['mediaPropertiesChanged', { title: m.title, artist: m.artist, albumTitle: m.album }],
      ['mediaPlaybackChanged', { state: m.playbackState }],
      ['mediaTimelineChanged', { position: m.position, duration: m.duration }],
      ['mediaThumbnailChanged', {
        hasThumbnail: !!m.artUrl,
        primaryColor: { x: 0.12, y: 0.12, z: 0.12 },
        secondaryColor: { x: 0.0, y: 0.0, z: 0.0 },
        tertiaryColor: { x: 0.25, y: 0.25, z: 0.25 },
        highContrastColor: { x: 1.0, y: 1.0, z: 1.0 }
      }]
    ];
    for (var i = 0; i < calls.length; i++) {
      var fn = mediaListener[calls[i][0]];
      if (typeof fn === 'function') {
        try { fn(calls[i][1]); } catch (e) { console.error(e); }
      }
    }
  }

  window.__wpProps = function (p) {
    pendingProps = p;
    if (propListener && typeof propListener.applyUserProperties === 'function') {
      try { propListener.applyUserProperties(p); } catch (e) { console.error(e); }
    }
  };
  window.__wpAudio = function (a) {
    if (audioCb) { try { audioCb(a); } catch (e) { console.error(e); } }
  };
  window.__wpMedia = function (m) {
    pendingMedia = m;
    deliverMedia(m);
  };
  window.__wpEngineBridge = true;
})();
"#;

    fn cef_thread() -> &'static SyncSender<OpenRequest> {
        static TX: OnceLock<SyncSender<OpenRequest>> = OnceLock::new();
        TX.get_or_init(|| {
            let (tx, rx) = sync_channel::<OpenRequest>(1);
            std::thread::Builder::new()
                .name("cef".into())
                .spawn(move || cef_main(rx))
                .expect("spawning the CEF thread");
            tx
        })
    }

    /// Pin the CEF API version for this process.
    ///
    /// CEF 148 introduced API versioning: every call crossing the libcef
    /// boundary checks that the host has pinned a version first, and without
    /// this libcef aborts with `CefClient_0_CToCpp called with invalid version
    /// -1` the moment it calls back into us. It must run in BOTH roles — the
    /// subprocess path and the browser-process path — before any other CEF
    /// call. `CEF_API_VERSION_LAST` is the newest version these bindings were
    /// generated against.
    fn pin_api_version() {
        let _ = api_hash(cef::sys::CEF_API_VERSION_LAST, 0);
    }

    // CEF requires an App on every initialize path. Ours also carries the
    // render-process handler, which is the only place the WE bridge can be
    // injected early enough: `on_context_created` runs in the render process
    // before the page's own scripts, whereas anything driven from the browser
    // process races them.
    wrap_app! {
        struct WallpaperApp {
            render_process_handler: RenderProcessHandler,
        }

        impl App {
            fn render_process_handler(&self) -> Option<RenderProcessHandler> {
                Some(self.render_process_handler.clone())
            }

            /// Some sandboxes (containers with restricted process-spawn/
            /// namespace permissions) fail to launch Chromium's separate GPU
            /// process at all — not a missing-driver problem
            /// (`/dev/dri/renderD128` present and accessible is not enough;
            /// the crash is a zygote/process-launch failure:
            /// `GPU process launch failed`, then a hard `GPU process isn't
            /// usable. Goodbye.` before any page even loads). `--in-process-
            /// gpu` is the standard fix for exactly this failure mode: it
            /// runs GPU code inside the browser process instead of spawning
            /// a separate one, sidestepping the broken spawn path entirely.
            /// `--disable-gpu-sandbox` additionally skips the per-process GPU
            /// sandbox (redundant with `Settings.no_sandbox` already being
            /// set, but that field doesn't necessarily suppress every
            /// internal Chromium sandbox attempt on its own — the zygote
            /// failure above is exactly that kind of leftover). Applied
            /// unconditionally (every process type, including the initial
            /// browser-process call where `process_type` is `None`) —
            /// standard practice for flags like this that child processes
            /// need to inherit consistently.
            fn on_before_command_line_processing(
                &self,
                _process_type: Option<&CefString>,
                command_line: Option<&mut CommandLine>,
            ) {
                let Some(command_line) = command_line else {
                    return;
                };
                command_line.append_switch(Some(&"disable-gpu-sandbox".into()));
                // Chromium restricts a `file://` page to reading `file://`
                // resources from its OWN directory tree by default — real
                // impact here, not theoretical: a `file`/`directory`-type
                // property's user-chosen path (`user_properties_json` in
                // this module) is almost never inside the wallpaper's own
                // bundle directory (a user picks their own ~/Pictures
                // folder for a slideshow, workshop item 893418273's real
                // use case), so without this flag every such override —
                // and the `__files` directory-listing feature built on top
                // of it — would silently fail to load, indistinguishable
                // from a wallpaper that simply renders solid black.
                command_line.append_switch(Some(&"allow-file-access-from-files".into()));
            }
        }
    }

    fn make_app() -> App {
        WallpaperApp::new(WallpaperRenderProcess::new())
    }

    wrap_render_process_handler! {
        struct WallpaperRenderProcess;

        impl RenderProcessHandler {
            fn on_context_created(
                &self,
                _browser: Option<&mut Browser>,
                frame: Option<&mut Frame>,
                _context: Option<&mut V8Context>,
            ) {
                if let Some(frame) = frame {
                    frame.execute_java_script(
                        Some(&BOOTSTRAP_JS.into()),
                        Some(&"wp-engine://bridge".into()),
                        0,
                    );
                }
            }
        }
    }

    pub fn subprocess_main() -> Option<i32> {
        pin_api_version();
        let args = Args::new();
        let mut app = make_app();
        let code = cef::execute_process(
            Some(args.as_main_args()),
            Some(&mut app),
            std::ptr::null_mut(),
        );
        (code >= 0).then_some(code)
    }

    /// How often the audio spectrum is pushed into the page. WE's own listener
    /// fires at roughly frame rate; 30 Hz is smooth for a visualiser and keeps
    /// the per-push `execute_java_script` cost off the CEF thread's back.
    const AUDIO_PUSH_INTERVAL: Duration = Duration::from_millis(33);

    /// Owns CEF for the life of the process: initialise once, then pump the
    /// message loop forever, creating browsers as requests arrive.
    ///
    /// ponytail: never calls `cef::shutdown`. Re-initialising CEF in the same
    /// process is not supported, and a wallpaper switch would otherwise do
    /// exactly that; letting process exit reclaim it is the honest trade. The
    /// old browser is closed when its frame channel drops.
    fn cef_main(rx: std::sync::mpsc::Receiver<OpenRequest>) {
        pin_api_version();
        let args = Args::new();
        let mut app = make_app();
        let settings = Settings {
            no_sandbox: 1,
            windowless_rendering_enabled: 1,
            // Keep Chromium's own cache out of the user's cwd.
            root_cache_path: cef_cache_dir().as_str().into(),
            ..Default::default()
        };

        if initialize(
            Some(args.as_main_args()),
            Some(&settings),
            Some(&mut app),
            std::ptr::null_mut(),
        ) != 1
        {
            tracing::error!(target: "web", "cef::initialize failed — web wallpapers unavailable");
            return;
        }

        // Browsers are kept alive here; dropping one closes it. Paired with
        // its own input receiver since each `start_web_stream` call creates a
        // fresh channel (see `OpenRequest::input_rx`).
        let mut open: Vec<(Browser, Receiver<WebInputEvent>)> = Vec::new();
        let mut properties = String::new();
        let mut audio: Option<crate::engine::audio::AudioCapture> = None;
        let mut last_audio_push = Instant::now();
        // MPRIS is a freedesktop.org/Linux-only surface — see `engine::media`.
        #[cfg(target_os = "linux")]
        let mut media: Option<crate::engine::media::MediaWatcher> = None;

        loop {
            match rx.try_recv() {
                Ok(req) => {
                    // A new wallpaper replaces the previous one.
                    open.clear();
                    audio = None;
                    #[cfg(target_os = "linux")]
                    {
                        media = None;
                    }
                    properties = req.properties.clone();
                    let wants_audio = req.wants_audio;
                    #[cfg(target_os = "linux")]
                    let wants_media = req.wants_media;
                    let browser = create_browser(&req);
                    match browser {
                        Some(browser) => open.push((browser, req.input_rx)),
                        None => tracing::error!(target: "web", "failed to create CEF browser"),
                    }
                    if wants_audio {
                        audio = crate::engine::audio::AudioCapture::start();
                        if audio.is_none() {
                            tracing::warn!(
                                target: "web",
                                "page uses wallpaperRegisterAudioListener but audio capture \
                                 could not start; the visualiser will sit at silence"
                            );
                        }
                    }
                    #[cfg(target_os = "linux")]
                    if wants_media {
                        media = crate::engine::media::MediaWatcher::start();
                        if media.is_none() {
                            tracing::warn!(
                                target: "web",
                                "page uses wallpaperMediaIntegration but the D-Bus session bus \
                                 could not be reached; media info will never update"
                            );
                        }
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }

            if let Some((browser, input_rx)) = open.first() {
                // Properties are re-sent every tick rather than once: the page
                // may not have assigned its listener yet, and the injected
                // setter re-delivers on assignment anyway, so this is cheap
                // insurance against a lost first delivery. `__wpProps` is
                // idempotent.
                if !properties.is_empty() {
                    eval_in_page(
                        browser,
                        &format!("window.__wpProps&&window.__wpProps({properties});"),
                    );
                    // One delivery per navigation is enough once it lands.
                    properties.clear();
                }

                if let Some(capture) = &audio {
                    if last_audio_push.elapsed() >= AUDIO_PUSH_INTERVAL {
                        last_audio_push = Instant::now();
                        eval_in_page(browser, &audio_push_script(&capture.spectrum()));
                    }
                }

                // `try_recv` already collapses to "changed since last call"
                // (see its own doc comment) — no extra interval gate needed
                // the way the audio spectrum's continuous stream needs one.
                #[cfg(target_os = "linux")]
                if let Some(watcher) = &media {
                    if let Some(info) = watcher.try_recv() {
                        eval_in_page(browser, &media_push_script(&info));
                    }
                }

                // Drain every pending event rather than just the latest: a
                // dropped click is a real bug in a way a dropped mouse-move
                // sample never is.
                while let Ok(event) = input_rx.try_recv() {
                    forward_input_event(browser, event);
                }
            }

            // Chromium paints from inside this call — without it there are no
            // OnPaint callbacks at all and the wallpaper never advances.
            do_message_loop_work();
            std::thread::sleep(Duration::from_millis(4));
        }
    }

    fn eval_in_page(browser: &Browser, script: &str) {
        if let Some(frame) = browser.main_frame() {
            frame.execute_java_script(Some(&script.into()), None, 0);
        }
    }

    /// Forward one mouse event to CEF — mirrors `CWeb::updateMouse` exactly
    /// (`SendMouseMoveEvent`/`SendMouseClickEvent`, no other event types; see
    /// `WebInputEvent`'s own doc comment). Normalized `[0,1]²` coordinates
    /// convert to CEF's fixed `WEB_WIDTH`×`WEB_HEIGHT` pixel canvas here, so
    /// callers only ever deal in the same normalized space every other
    /// pointer consumer in this codebase already uses.
    fn forward_input_event(browser: &Browser, event: WebInputEvent) {
        let Some(host) = browser.host() else {
            return;
        };
        let to_px = |x_norm: f32, y_norm: f32| MouseEvent {
            x: (x_norm.clamp(0.0, 1.0) * WEB_WIDTH as f32) as i32,
            y: (y_norm.clamp(0.0, 1.0) * WEB_HEIGHT as f32) as i32,
            modifiers: 0,
        };
        match event {
            WebInputEvent::MouseMove { x_norm, y_norm } => {
                host.send_mouse_move_event(Some(&to_px(x_norm, y_norm)), 0);
            }
            WebInputEvent::MouseButton { x_norm, y_norm, button, pressed } => {
                let type_ = match button {
                    WebMouseButton::Left => MouseButtonType::LEFT,
                    WebMouseButton::Right => MouseButtonType::RIGHT,
                };
                host.send_mouse_click_event(
                    Some(&to_px(x_norm, y_norm)),
                    type_,
                    (!pressed) as i32,
                    1,
                );
            }
        }
    }

    /// Build the `__wpAudio` call for one spectrum snapshot.
    ///
    /// WE hands the listener 128 values: 64 left-channel bands followed by 64
    /// right-channel bands — precisely `AudioSpectrum`'s `s64_*` pair, so no
    /// resampling is needed.
    fn audio_push_script(spectrum: &crate::engine::audio::AudioSpectrum) -> String {
        let mut s = String::with_capacity(1024);
        s.push_str("window.__wpAudio&&window.__wpAudio([");
        for (i, v) in spectrum
            .s64_left
            .iter()
            .chain(spectrum.s64_right.iter())
            .enumerate()
        {
            if i > 0 {
                s.push(',');
            }
            // Three decimals is well past what a visualiser can show and keeps
            // the script string small enough to parse cheaply at 30 Hz.
            let _ = write!(s, "{v:.3}");
        }
        s.push_str("]);");
        s
    }

    /// Build the `__wpMedia` call for one `MediaInfo` snapshot — the JS
    /// bridge (`BOOTSTRAP_JS`'s `deliverMedia`) fans this single object out
    /// into the four real `mediaPropertiesChanged`/`mediaPlaybackChanged`/
    /// `mediaTimelineChanged`/`mediaThumbnailChanged` callbacks itself, so
    /// only one payload needs building here. `position`/`duration` pass
    /// through in the same raw microseconds `MediaInfo` itself carries —
    /// see its own doc comment for why.
    #[cfg(target_os = "linux")]
    fn media_push_script(info: &crate::engine::media::MediaInfo) -> String {
        let json_str = |s: &str| serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string());
        let art_url = info
            .art_url
            .as_deref()
            .map(json_str)
            .unwrap_or_else(|| "null".to_string());
        format!(
            "window.__wpMedia&&window.__wpMedia({{title:{title},artist:{artist},album:{album},\
             playbackState:{state},position:{pos},duration:{dur},artUrl:{art_url}}});",
            title = json_str(&info.title),
            artist = json_str(&info.artist),
            album = json_str(&info.album),
            state = info.playback_state as i32,
            pos = info.position_us,
            dur = info.duration_us,
        )
    }

    fn create_browser(req: &OpenRequest) -> Option<Browser> {
        let render_handler = WallpaperRenderHandler::new(req.frames.clone());
        let mut client = WallpaperClient::new(render_handler);
        let window_info = WindowInfo {
            windowless_rendering_enabled: 1,
            bounds: Rect {
                x: 0,
                y: 0,
                width: WEB_WIDTH as i32,
                height: WEB_HEIGHT as i32,
            },
            ..Default::default()
        }
        .set_as_windowless(0);
        let browser_settings = BrowserSettings {
            windowless_frame_rate: 60,
            // CEF's OSR default is fully transparent (`0x00000000`) — meant
            // for embedding a browser view over other content, not for a
            // desktop background. A wallpaper whose page hasn't painted an
            // opaque background yet (or, on this run, is genuinely stuck on
            // an early transparent paint — see `on_before_command_line_
            // processing`'s doc comment on `--in-process-gpu`) should still
            // show *something* solid, not whatever the desktop had behind
            // it. `0xFF000000` = opaque black, ARGB.
            background_color: 0xFF000000,
            ..Default::default()
        };

        browser_host_create_browser_sync(
            Some(&window_info),
            Some(&mut client),
            Some(&req.url.as_str().into()),
            Some(&browser_settings),
            None,
            None,
        )
    }

    /// Builds the JSON `applyUserProperties` receives, starting from the raw
    /// `project.json` declarations (preserving every field real wallpaper JS
    /// might read beyond `.value` — `condition`, `index`, `order`, etc. —
    /// verbatim) and then:
    ///
    /// 1. **Applying `--set-property`/saved overrides** — this used to send
    ///    the project.json defaults straight through, meaning web wallpapers
    ///    silently ignored every override scene wallpapers already got via
    ///    `engine::properties::SceneProperties`. Same override source
    ///    (`global_overrides`), so `wp-engine set <web-wallpaper>
    ///    --set-property name=value` (and a saved `settings::WpSettings`
    ///    override) now actually reaches the page.
    /// 2. **`file`-type values are passed through as bare filesystem paths,
    ///    unmodified** — real-content ground truth, confirmed against TWO
    ///    independent wallpapers' actual code, not assumed: 893418273's
    ///    `backgroundimage` handler does `imagePath = "file:///" +
    ///    properties.backgroundimage.value`, and 1396475780's `audiOrbits.js`
    ///    `setImgSrc` does the identical `"file:///" + srcVal` — both
    ///    wallpapers do their *own* `file://` prefixing and expect a bare
    ///    path in `.value`. An earlier version of this function pre-
    ///    converted `.value` to a full `file://` URL, which looked
    ///    plausible (a bare path isn't normally a valid `<img src>`) but
    ///    was never actually verified against a real wallpaper's *full*
    ///    property-consumption code, only assumed from seeing `.value` read
    ///    directly — that produced a broken double-prefixed
    ///    `file:///file:///...` URL for exactly the wallpaper it was meant
    ///    to fix, caught by an end-to-end visual smoke test (a magenta test
    ///    image that rendered as flat black — every sampled pixel exactly
    ///    matched the browser's own background-fill color, meaning the
    ///    image never loaded at all). A bare absolute path used directly in
    ///    a CSS `url(...)`/`<img src>` still resolves correctly on a page
    ///    loaded from a `file://` origin (standard URL-resolution rules
    ///    treat a leading `/` as absolute-path-from-the-current-origin's
    ///    root, and a `file://` origin's root *is* the filesystem root) —
    ///    so passing the raw path through is correct for both conventions,
    ///    not just the one this was tested against.
    /// 3. **Bundling a file listing for `directory`-type properties** under
    ///    `__files` (bare paths, same reasoning as above) — real usage
    ///    (workshop item 893418273) calls
    ///    `wallpaperRequestRandomFileForProperty(name, cb)` expecting a
    ///    *different* random file back on each call (a slideshow). Rather
    ///    than build genuine async native↔JS IPC (the CEF `CefProcessMessage`
    ///    round-trip a literal port would need) for one confirmed real
    ///    caller, the directory is listed once, here, and
    ///    `wallpaperRequestRandomFileForProperty`'s JS-side implementation
    ///    (`BOOTSTRAP_JS`) picks randomly from the bundled list — same
    ///    user-visible behavior (a randomized slideshow), far less
    ///    machinery. `__` prefix so it can't collide with any real WE
    ///    property field.
    fn user_properties_json(dir: &Path) -> String {
        let Ok(text) = std::fs::read_to_string(dir.join("project.json")) else {
            return String::new();
        };
        let Ok(mut project) = serde_json::from_str::<serde_json::Value>(&text) else {
            return String::new();
        };
        let Some(props) = project
            .get_mut("general")
            .and_then(|g| g.get_mut("properties"))
            .and_then(|p| p.as_object_mut())
        else {
            return String::new();
        };

        let resolved = crate::engine::properties::SceneProperties::from_project_dir(dir);
        for (name, decl) in props.iter_mut() {
            if let Some(value) = resolved.get(name) {
                decl["value"] = value.clone();
            }

            if decl.get("type").and_then(|t| t.as_str()) == Some("directory") {
                let dir_path = decl.get("value").and_then(|v| v.as_str()).unwrap_or("");
                if !dir_path.is_empty() {
                    if let Ok(entries) = std::fs::read_dir(dir_path) {
                        let files: Vec<serde_json::Value> = entries
                            .flatten()
                            .filter(|e| e.path().is_file())
                            .filter_map(|e| e.path().canonicalize().ok())
                            .map(|p| serde_json::Value::String(p.display().to_string()))
                            .collect();
                        if !files.is_empty() {
                            decl["__files"] = serde_json::Value::Array(files);
                        }
                    }
                }
            }
        }

        serde_json::to_string(props).unwrap_or_default()
    }

    /// Does this wallpaper's bundle reference `needle` anywhere in its
    /// .html/.js files? Grepping is cruder than asking the page, but asking
    /// means an async round-trip into the render process for something the
    /// answer needs to gate a side effect *before* the browser even exists
    /// (opening a capture device, starting the D-Bus watcher) — see
    /// `uses_audio_listener`/`uses_media_listener`.
    ///
    /// ponytail: scans up to 2 levels deep. A wallpaper hiding the call in a
    /// .json blob or deeper tree just gets silence; widen the walk if one
    /// turns up.
    fn scans_bundle_for(dir: &Path, needle: &[u8]) -> bool {
        fn scan(dir: &Path, depth: usize, needle: &[u8]) -> bool {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return false;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if depth > 0 && scan(&path, depth - 1, needle) {
                        return true;
                    }
                    continue;
                }
                let is_script = matches!(
                    path.extension().and_then(|e| e.to_str()),
                    Some("html" | "htm" | "js")
                );
                if is_script
                    && std::fs::read(&path).is_ok_and(|bytes| memmem_contains(&bytes, needle))
                {
                    return true;
                }
            }
            false
        }
        scan(dir, 2, needle)
    }

    /// Does this wallpaper call `wallpaperRegisterAudioListener`? The answer
    /// decides whether we open a desktop-audio capture device at all — a
    /// side effect worth avoiding for the pages that never use it.
    fn uses_audio_listener(dir: &Path) -> bool {
        scans_bundle_for(dir, b"wallpaperRegisterAudioListener")
    }

    /// Does this wallpaper call `wallpaperMediaIntegration.listen`? The
    /// answer decides whether we start the D-Bus MPRIS watcher
    /// (`engine::media`) at all — same reasoning as `uses_audio_listener`.
    #[cfg(target_os = "linux")]
    fn uses_media_listener(dir: &Path) -> bool {
        scans_bundle_for(dir, b"wallpaperMediaIntegration")
    }

    /// Substring search over raw bytes — these bundles are minified and not
    /// always valid UTF-8, so `str::contains` is not an option.
    fn memmem_contains(haystack: &[u8], needle: &[u8]) -> bool {
        haystack
            .windows(needle.len())
            .any(|window| window == needle)
    }

    fn cef_cache_dir() -> String {
        let dir = dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("wp-engine/cef");
        let _ = std::fs::create_dir_all(&dir);
        dir.to_string_lossy().into_owned()
    }

    wrap_client! {
        struct WallpaperClient {
            render_handler: RenderHandler,
        }

        impl Client {
            fn render_handler(&self) -> Option<RenderHandler> {
                Some(self.render_handler.clone())
            }
        }
    }

    wrap_render_handler! {
        struct WallpaperRenderHandler {
            frames: SyncSender<Arc<RgbaImage>>,
        }

        impl RenderHandler {
            fn view_rect(&self, _browser: Option<&mut Browser>, rect: Option<&mut Rect>) {
                // CEF treats an empty rect as an error and paints nothing.
                if let Some(rect) = rect {
                    *rect = Rect {
                        x: 0,
                        y: 0,
                        width: WEB_WIDTH as i32,
                        height: WEB_HEIGHT as i32,
                    };
                }
            }

            fn on_paint(
                &self,
                _browser: Option<&mut Browser>,
                type_: PaintElementType,
                _dirty_rects: Option<&[Rect]>,
                buffer: *const u8,
                width: ::std::os::raw::c_int,
                height: ::std::os::raw::c_int,
            ) {
                // PET_POPUP is the dropdown/select overlay drawn separately; we
                // only present the main view.
                if type_ != PaintElementType::VIEW || buffer.is_null() || width <= 0 || height <= 0 {
                    return;
                }
                let (w, h) = (width as u32, height as u32);
                let len = w as usize * h as usize * 4;
                // SAFETY: CEF guarantees `buffer` holds width*height*4 bytes of
                // BGRA for the duration of this callback. We copy out before
                // returning and never retain the pointer.
                let src = unsafe { std::slice::from_raw_parts(buffer, len) };

                let mut rgba = Vec::with_capacity(len);
                for px in src.chunks_exact(4) {
                    // CEF paints BGRA; RgbaImage wants RGBA.
                    rgba.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
                }
                let Some(img) = RgbaImage::from_raw(w, h, rgba) else {
                    return;
                };

                // Drop frames rather than block: this runs on the CEF UI
                // thread, and stalling it stalls Chromium itself.
                match self.frames.try_send(Arc::new(img)) {
                    Ok(()) | Err(TrySendError::Full(_)) => {}
                    Err(TrySendError::Disconnected(_)) => {}
                }
            }
        }
    }

    pub fn start_web_stream(
        html: &Path,
    ) -> Result<(RgbaImage, Receiver<Arc<RgbaImage>>, SyncSender<WebInputEvent>)> {
        let html = html
            .canonicalize()
            .with_context(|| format!("resolving web wallpaper path {}", html.display()))?;
        let url = format!("file://{}", html.to_string_lossy());

        let dir = html.parent().unwrap_or(&html).to_path_buf();
        let (frames, rx) = sync_channel::<Arc<RgbaImage>>(2);
        // Bounded, but generously so relative to `cef_main`'s ~4ms poll —
        // this should never realistically fill. `try_send` on the platform
        // side means a full channel drops rather than blocking the render
        // loop, same trade `frames` itself already makes.
        let (input_tx, input_rx) = sync_channel::<WebInputEvent>(64);
        cef_thread()
            .send(OpenRequest {
                url,
                frames,
                properties: user_properties_json(&dir),
                wants_audio: uses_audio_listener(&dir),
                #[cfg(target_os = "linux")]
                wants_media: uses_media_listener(&dir),
                #[cfg(not(target_os = "linux"))]
                wants_media: false,
                input_rx,
            })
            .map_err(|_| anyhow!("the CEF thread is not running"))?;

        // Block for the first paint so callers never present a blank surface,
        // matching the video decoder's contract.
        let deadline = Instant::now() + FIRST_FRAME_TIMEOUT;
        let first = loop {
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .ok_or_else(|| anyhow!("timed out waiting for the first web frame"))?;
            match rx.recv_timeout(remaining) {
                Ok(frame) => break frame,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    return Err(anyhow!("timed out waiting for the first web frame"))
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(anyhow!("the CEF browser closed before painting a frame"))
                }
            }
        };

        Ok((
            Arc::try_unwrap(first).unwrap_or_else(|arc| (*arc).clone()),
            rx,
            input_tx,
        ))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A fresh temp directory nobody else can collide with, for a
        /// project.json + (for the `directory` case) some real files.
        fn tempdir(name: &str) -> std::path::PathBuf {
            let dir = std::env::temp_dir().join(format!(
                "wp-engine-web-props-test-{name}-{}-{:?}",
                std::process::id(),
                std::time::SystemTime::now()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }

        /// `file`-type property values pass through as bare filesystem
        /// paths, unmodified — confirmed against TWO real wallpapers' full
        /// consumption code (893418273's `backgroundimage` handler and
        /// 1396475780's `audiOrbits.js` `setImgSrc`), both of which do
        /// their *own* `"file:///" + value` prefixing and would break on
        /// an already-prefixed value (a real regression an earlier version
        /// of this had — a visual smoke test caught it rendering flat
        /// black, the double-`file:///file:///`-prefixed URL never
        /// loading).
        #[test]
        fn file_property_value_passes_through_as_a_bare_path() {
            let dir = tempdir("file-prop");
            std::fs::write(
                dir.join("project.json"),
                r#"{"general":{"properties":{
                    "img_background": {"type":"file","text":"BG","value":"/home/user/bg.png"}
                }}}"#,
            )
            .unwrap();

            let json = user_properties_json(&dir);
            let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed["img_background"]["value"], "/home/user/bg.png");
        }

        /// `directory`-type properties get a `__files` listing bundled in
        /// (bare paths, same reasoning as the `file`-type case above) when
        /// the declared value resolves to a real, readable directory —
        /// what `wallpaperRequestRandomFileForProperty` picks randomly from
        /// (confirmed real usage, workshop item 893418273's image-slideshow
        /// `customrandomdirectory`).
        #[test]
        fn directory_property_gets_a_bundled_file_listing() {
            let dir = tempdir("directory-prop");
            let images = dir.join("images");
            std::fs::create_dir_all(&images).unwrap();
            std::fs::write(images.join("a.jpg"), b"fake").unwrap();
            std::fs::write(images.join("b.jpg"), b"fake").unwrap();
            std::fs::create_dir_all(images.join("subdir")).unwrap(); // not a file, must be excluded

            let project = serde_json::json!({"general": {"properties": {
                "customrandomdirectory": {
                    "type": "directory",
                    "text": "Image Folder",
                    "value": images.to_str().unwrap(),
                }
            }}});
            std::fs::write(dir.join("project.json"), project.to_string()).unwrap();

            let json = user_properties_json(&dir);
            let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
            let files = parsed["customrandomdirectory"]["__files"]
                .as_array()
                .expect("__files should be a bundled array");
            assert_eq!(files.len(), 2, "should list exactly the 2 real files, not the subdirectory");
            for f in files {
                let s = f.as_str().unwrap();
                assert!(s.starts_with('/'), "{s} should be a bare absolute path");
                assert!(!s.starts_with("file://"), "{s} should not be pre-prefixed with file://");
                assert!(s.ends_with(".jpg"));
            }
        }

        /// An empty/unset directory value (the common case — no default
        /// makes sense for "pick your own folder") bundles no listing
        /// rather than erroring.
        #[test]
        fn directory_property_with_no_value_gets_no_file_listing() {
            let dir = tempdir("directory-prop-empty");
            std::fs::write(
                dir.join("project.json"),
                r#"{"general":{"properties":{
                    "customrandomdirectory": {"type":"directory","text":"Image Folder","value":""}
                }}}"#,
            )
            .unwrap();

            let json = user_properties_json(&dir);
            let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert!(parsed["customrandomdirectory"].get("__files").is_none());
        }

        /// Every other declared field on a property (real content puts real
        /// weight on `condition`/`index`/`order`) must survive untouched —
        /// this rebuilds from the raw declaration and only overlays
        /// `value`/`__files`, it doesn't reconstruct from a narrower typed
        /// view that would silently drop them.
        #[test]
        fn unrelated_declaration_fields_pass_through_untouched() {
            let dir = tempdir("passthrough-fields");
            std::fs::write(
                dir.join("project.json"),
                r#"{"general":{"properties":{
                    "effect": {"type":"combo","text":"Effect","value":"1","index":7,"order":107,
                               "condition":"somecond","options":[{"label":"A","value":"0"}]}
                }}}"#,
            )
            .unwrap();

            let json = user_properties_json(&dir);
            let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed["effect"]["index"], 7);
            assert_eq!(parsed["effect"]["order"], 107);
            assert_eq!(parsed["effect"]["condition"], "somecond");
            assert_eq!(parsed["effect"]["options"][0]["label"], "A");
        }
    }
}
