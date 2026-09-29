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
struct State {
    app: Option<WallpaperApplication>,
    title: Option<String>,
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
    let listener = UnixListener::bind(&socket)
        .with_context(|| format!("binding {}", socket.display()))?;
    listener.set_nonblocking(true)?;

    STOP_REQUESTED.store(false, Ordering::SeqCst);
    let handler = handle_stop_signal as extern "C" fn(libc::c_int) as *const () as libc::sighandler_t;
    unsafe {
        libc::signal(libc::SIGINT, handler);
        libc::signal(libc::SIGTERM, handler);
    }

    tracing::info!(target: "app", pid = std::process::id(), socket = %socket.display(), "daemon listening");
    println!("wp-engine daemon listening on {}", socket.display());

    let state = Arc::new(Mutex::new(State::default()));
    while !STOP_REQUESTED.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                let state = Arc::clone(&state);
                std::thread::spawn(move || handle_client(stream, &state));
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => tracing::warn!(target: "app", "daemon accept failed: {e}"),
        }
    }

    tracing::info!(target: "app", "daemon shutting down");
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
        Err(e) => Response::Error { message: format!("reading request: {e}") },
        Ok(_) => match serde_json::from_str::<Request>(line.trim()) {
            Err(e) => Response::Error { message: format!("malformed request: {e}") },
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

fn handle_request(req: Request, state: &Mutex<State>) -> Response {
    // A panic inside an earlier apply poisons the mutex; the state is still
    // meaningful (worst case: no wallpaper), so keep serving.
    let mut state = state.lock().unwrap_or_else(|p| p.into_inner());
    match req {
        Request::Ping => Response::Ok,
        Request::Status => Response::Status(status(&state)),
        Request::Apply(spec) => {
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
                Err(e) => Response::Error { message: format!("{e:#}") },
            }
        }
        Request::SetQuality { quality } => {
            let Some(q) = RenderQuality::parse(&quality) else {
                return Response::Error {
                    message: format!("invalid quality '{quality}' (expected Ultra, High, Medium, or Low)"),
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
        Request::SetVolume { volume } => {
            if let Some(app) = &state.app {
                app.context().settings.lock().unwrap().volume = volume.clamp(0.0, 1.0);
            }
            Response::Ok
        }
        Request::SetAudioDevice { device } => {
            crate::platform::audio::set_preferred_device(device);
            Response::Ok
        }
        Request::Clear => {
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
        let err = InstanceLock::acquire(&path).err().expect("second lock must fail");
        assert!(err.to_string().contains(&std::process::id().to_string()), "{err}");

        drop(first);
        InstanceLock::acquire(&path).expect("lock is free again once dropped");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
