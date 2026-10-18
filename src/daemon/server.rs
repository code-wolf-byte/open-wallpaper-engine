//! The daemon side: singleton lock, socket listener, and the one renderer.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::DirBuilderExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};

use super::protocol::{ActiveWallpaper, DaemonStatus, Request, Response};
use crate::application::WallpaperApplication;
use crate::platform::RenderQuality;

/// Set by SIGINT/SIGTERM or a `Shutdown` request; polled by the accept loop.
static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_stop_signal(_sig: libc::c_int) {
    STOP_REQUESTED.store(true, Ordering::SeqCst);
}

/// A client that connects but never finishes its request line must not pin
/// a handler thread forever.
const CLIENT_READ_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Default)]
pub(super) struct State {
    pub(super) app: Option<WallpaperApplication>,
    pub(super) title: Option<String>,
    pub(super) playlist: Option<crate::schedule::PlaylistRun>,
}

impl State {
    /// Show a wallpaper from its saved per-wallpaper settings.
    fn apply_item(&mut self, id_or_path: &str) -> Result<()> {
        self.app = None;
        self.title = None;
        let context = crate::application::ApplicationContext::for_wallpaper(id_or_path)?;
        let mut app = WallpaperApplication::new(context);
        app.setup()?;
        self.title = Some(
            crate::workshop::find_by_id(id_or_path)
                .map(|w| w.title().to_string())
                .unwrap_or_else(|| id_or_path.to_string()),
        );
        self.app = Some(app);
        Ok(())
    }

    /// Switch the playing playlist to item `index`.
    fn play_index(&mut self, index: usize) -> Result<()> {
        let Some(run) = self.playlist.as_mut() else {
            return Err(anyhow!("no playlist is playing"));
        };
        run.switch_to(index, std::time::Instant::now());
        let (name, idx, item) = (run.name.clone(), run.index, run.current().to_string());
        crate::schedule::save_position(&name, idx);
        tracing::info!(target: "app", playlist = %name, item = %item, "playlist item");
        self.apply_item(&item)
    }

    /// Advance the playlist when its schedule says so.
    fn tick_playlist(&mut self) {
        let Some(run) = &self.playlist else {
            return;
        };
        if let Some(next) = run.due(std::time::Instant::now(), crate::schedule::LocalTime::now()) {
            if let Err(e) = self.play_index(next) {
                tracing::warn!(target: "app", "playlist switch failed: {e:#}");
            }
        }
    }
}

/// Holds the exclusive `flock` for as long as it lives; the kernel drops the
/// lock when the process exits, however it exits, so a crashed daemon never
/// leaves a stale lock behind.
pub struct InstanceLock {
    _file: File,
}

impl InstanceLock {
    /// Take the singleton lock, or fail naming the pid that holds it.
    pub fn acquire(path: &Path) -> Result<Self> {
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                let holder = std::fs::read_to_string(path).unwrap_or_default();
                return Err(anyhow!(
                    "another wp-engine daemon is already running (pid {})",
                    holder.trim()
                ));
            }
            return Err(err).with_context(|| format!("locking {}", path.display()));
        }
        file.set_len(0)?;
        write!(file, "{}", std::process::id())?;
        Ok(Self { _file: file })
    }
}

/// Run the daemon in the foreground until SIGINT/SIGTERM or a `Shutdown`
/// request. Fails immediately if another daemon already holds the lock.
pub fn run_daemon() -> Result<()> {
    let dir = super::runtime_dir();
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&dir)
        .with_context(|| format!("creating runtime dir {}", dir.display()))?;

    let lock = InstanceLock::acquire(&super::lock_path())?;

    // Holding the lock proves any socket file left here is from a daemon
    // that died without cleaning up.
    let socket = super::socket_path();
    let _ = std::fs::remove_file(&socket);
    let listener =
        UnixListener::bind(&socket).with_context(|| format!("binding {}", socket.display()))?;
    listener.set_nonblocking(true)?;

    STOP_REQUESTED.store(false, Ordering::SeqCst);
    let handler =
        handle_stop_signal as extern "C" fn(libc::c_int) as *const () as libc::sighandler_t;
    unsafe {
        libc::signal(libc::SIGINT, handler);
        libc::signal(libc::SIGTERM, handler);
    }

    tracing::info!(target: "app", pid = std::process::id(), socket = %socket.display(), "daemon listening");
    println!("wp-engine daemon listening on {}", socket.display());

    let state = Arc::new(Mutex::new(State::default()));
    let tray = (std::env::var_os("WP_ENGINE_NO_TRAY").is_none())
        .then(|| super::tray::spawn(Arc::clone(&state)))
        .flatten();
    while !STOP_REQUESTED.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                let state = Arc::clone(&state);
                std::thread::spawn(move || handle_client(stream, &state));
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
                // `try_lock`: an apply in progress holds the lock; the
                // schedule can wait for the next pass.
                if let Ok(mut s) = state.try_lock() {
                    s.tick_playlist();
                }
            }
            Err(e) => tracing::warn!(target: "app", "daemon accept failed: {e}"),
        }
    }

    tracing::info!(target: "app", "daemon shutting down");
    if let Some(tray) = tray {
        tray.shutdown().wait();
    }
    // Unlink before the lock goes so a new daemon never sees our socket.
    let _ = std::fs::remove_file(&socket);
    if let Ok(mut state) = state.lock() {
        state.app = None;
    }
    drop(lock);
    Ok(())
}

fn handle_client(stream: UnixStream, state: &Mutex<State>) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(CLIENT_READ_TIMEOUT));

    let mut line = String::new();
    let response = match BufReader::new(&stream).read_line(&mut line) {
        Err(e) => Response::Error {
            message: format!("reading request: {e}"),
        },
        Ok(_) => match serde_json::from_str::<Request>(line.trim()) {
            Err(e) => Response::Error {
                message: format!("malformed request: {e}"),
            },
            Ok(req) => {
                tracing::debug!(target: "app", ?req, "daemon request");
                handle_request(req, state)
            }
        },
    };

    let mut out = serde_json::to_string(&response).unwrap_or_default();
    out.push('\n');
    let _ = (&stream).write_all(out.as_bytes());
}

pub(super) fn handle_request(req: Request, state: &Mutex<State>) -> Response {
    // A panic inside an earlier apply poisons the mutex; the state is still
    // meaningful (worst case: no wallpaper), so keep serving.
    let mut state = state.lock().unwrap_or_else(|p| p.into_inner());
    match req {
        Request::Ping => Response::Ok,
        Request::Status => Response::Status(status(&state)),
        Request::Apply(spec) => {
            // An explicit wallpaper ends any playlist rotation.
            state.playlist = None;
            let title = spec.title.clone();
            // Stop the old renderer before `WallpaperApplication::new`
            // swaps the process-global property overrides under it.
            state.app = None;
            state.title = None;
            let mut app = WallpaperApplication::new(spec.into_context());
            match app.setup() {
                Ok(()) => {
                    tracing::info!(target: "app", background = %app.context().background.display(), "daemon applied wallpaper");
                    state.app = Some(app);
                    state.title = title;
                    Response::Ok
                }
                Err(e) => Response::Error {
                    message: format!("{e:#}"),
                },
            }
        }
        Request::SetQuality { quality } => {
            let Some(q) = RenderQuality::parse(&quality) else {
                return Response::Error {
                    message: format!(
                        "invalid quality '{quality}' (expected Ultra, High, Medium, or Low)"
                    ),
                };
            };
            // No screen named: the whole wallpaper, every output included.
            if let Some(app) = &state.app {
                let context = app.context();
                context.settings.lock().unwrap().quality = q;
                for s in context.screen_settings.values() {
                    s.lock().unwrap().quality = q;
                }
            }
            Response::Ok
        }
        Request::SetFps { fps } => {
            if let Some(app) = &state.app {
                let context = app.context();
                let fps = fps.clamp(1, 240);
                context.settings.lock().unwrap().fps = fps;
                for s in context.screen_settings.values() {
                    s.lock().unwrap().fps = fps;
                }
            }
            Response::Ok
        }
        Request::PropertiesChanged { wallpaper, names } => {
            let Some(dir) = crate::workshop::find_item_dir(&wallpaper, None)
                .or_else(|| Some(std::path::PathBuf::from(&wallpaper)).filter(|p| p.is_dir()))
            else {
                return Response::Error {
                    message: format!("unknown wallpaper '{wallpaper}'"),
                };
            };
            let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
            if let Some(app) = &state.app {
                // Every output's settings: each applies only edits for the
                // wallpaper it is actually showing.
                let context = app.context();
                for s in std::iter::once(&context.settings).chain(context.screen_settings.values())
                {
                    let mut s = s.lock().unwrap();
                    for name in &names {
                        let (dir, name) = (dir.clone(), name.clone());
                        s.property_edits
                            .push(crate::content::settings::PropertyEdit { dir, name });
                    }
                }
            }
            Response::Ok
        }
        Request::SetVolume { volume } => {
            if let Some(app) = &state.app {
                app.context().settings.lock().unwrap().volume = volume.clamp(0.0, 1.0);
            }
            Response::Ok
        }
        Request::SetAudioDevice { device } => {
            crate::services::audio_devices::set_preferred_device(device);
            Response::Ok
        }
        Request::PlayPlaylist { name } => {
            let settings = crate::settings::WpSettings::load();
            let Some(playlist) = settings.playlists.get(&name).cloned() else {
                return Response::Error {
                    message: format!("no playlist named '{name}'"),
                };
            };
            let resume = crate::schedule::load_positions().get(&name).copied();
            let Some(run) = crate::schedule::PlaylistRun::start(
                &name,
                playlist,
                resume,
                crate::schedule::LocalTime::now(),
            ) else {
                return Response::Error {
                    message: format!("playlist '{name}' is empty"),
                };
            };
            let first = run.index;
            state.playlist = Some(run);
            match state.play_index(first) {
                Ok(()) => Response::Ok,
                Err(e) => Response::Error {
                    message: format!("{e:#}"),
                },
            }
        }
        Request::NextItem | Request::PreviousItem => {
            let Some(run) = &state.playlist else {
                return Response::Error {
                    message: "no playlist is playing".into(),
                };
            };
            let index = if matches!(req, Request::NextItem) {
                run.next_index()
            } else {
                run.previous_index()
            };
            match state.play_index(index) {
                Ok(()) => Response::Ok,
                Err(e) => Response::Error {
                    message: format!("{e:#}"),
                },
            }
        }
        Request::StopPlaylist => {
            state.playlist = None;
            Response::Ok
        }
        Request::Clear => {
            state.playlist = None;
            state.app = None;
            state.title = None;
            Response::Ok
        }
        Request::Shutdown => {
            STOP_REQUESTED.store(true, Ordering::SeqCst);
            Response::Ok
        }
    }
}

fn status(state: &State) -> DaemonStatus {
    DaemonStatus {
        pid: std::process::id(),
        active: state.app.as_ref().map(|app| {
            let context = app.context();
            ActiveWallpaper {
                background: context.background.clone(),
                title: state.title.clone(),
                quality: context.settings.lock().unwrap().quality.label().to_string(),
            }
        }),
        playlist: state
            .playlist
            .as_ref()
            .map(|r| (r.name.clone(), r.index, r.playlist.items.len())),
    }
}

#[cfg(test)]
mod tests {
    use super::InstanceLock;

    #[test]
    fn second_lock_on_the_same_file_is_refused() {
        let dir = std::env::temp_dir().join(format!("wp-engine-lock-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("daemon.lock");

        let first = InstanceLock::acquire(&path).unwrap();
        let err = InstanceLock::acquire(&path)
            .err()
            .expect("second lock must fail");
        assert!(
            err.to_string().contains(&std::process::id().to_string()),
            "{err}"
        );

        drop(first);
        InstanceLock::acquire(&path).expect("lock is free again once dropped");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
