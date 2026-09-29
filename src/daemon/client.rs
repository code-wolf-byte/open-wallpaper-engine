//! The client side: talk to the running daemon, starting it if needed.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};

use super::protocol::{Request, Response};

/// How long a freshly spawned daemon gets to bind its socket.
const SPAWN_TIMEOUT: Duration = Duration::from_secs(10);

/// `-v` count forwarded to a daemon this process spawns.
static SPAWN_VERBOSITY: AtomicU8 = AtomicU8::new(0);

/// Forward this `-v` count to any daemon started from this process.
pub fn set_spawn_verbosity(verbosity: u8) {
    SPAWN_VERBOSITY.store(verbosity, Ordering::Relaxed);
}

/// True when a daemon is accepting connections.
pub fn is_running() -> bool {
    UnixStream::connect(super::socket_path()).is_ok()
}

/// Send `req` to the running daemon. Fails if none is running; a daemon-side
/// [`Response::Error`] comes back as `Err` too.
pub fn request(req: &Request) -> Result<Response> {
    let socket = super::socket_path();
    let stream = UnixStream::connect(&socket)
        .with_context(|| format!("no wp-engine daemon at {}", socket.display()))?;
    send(stream, req)
}

/// Like [`request`], but starts a background daemon first when none is
/// running.
pub fn request_or_spawn(req: &Request) -> Result<Response> {
    let stream = match UnixStream::connect(super::socket_path()) {
        Ok(stream) => stream,
        Err(_) => spawn_daemon()?,
    };
    send(stream, req)
}

fn send(mut stream: UnixStream, req: &Request) -> Result<Response> {
    let mut line = serde_json::to_string(req)?;
    line.push('\n');
    stream.write_all(line.as_bytes()).context("sending request to daemon")?;

    // No read timeout: an apply loads the whole scene before answering.
    let mut reply = String::new();
    BufReader::new(&stream)
        .read_line(&mut reply)
        .context("reading daemon reply")?;
    if reply.is_empty() {
        return Err(anyhow!("daemon closed the connection without replying"));
    }
    match serde_json::from_str(reply.trim()).context("malformed daemon reply")? {
        Response::Error { message } => Err(anyhow!(message)),
        response => Ok(response),
    }
}

/// Start `wp-engine daemon` detached from this process's session (so it
/// survives the terminal or GUI that launched it) and wait for its socket.
fn spawn_daemon() -> Result<UnixStream> {
    let exe = std::env::current_exe().context("locating the wp-engine binary")?;
    let log_path = super::log_path();
    if let Some(dir) = log_path.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating {}", dir.display()))?;
    }
    let log = std::fs::File::create(&log_path)
        .with_context(|| format!("opening daemon log {}", log_path.display()))?;

    let mut cmd = Command::new(exe);
    cmd.arg("daemon");
    for _ in 0..SPAWN_VERBOSITY.load(Ordering::Relaxed) {
        cmd.arg("-v");
    }
    cmd.stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = cmd.spawn().context("starting wp-engine daemon")?;
    tracing::info!(target: "app", pid = child.id(), log = %log_path.display(), "started background daemon");

    let deadline = Instant::now() + SPAWN_TIMEOUT;
    let stream = loop {
        if let Ok(stream) = UnixStream::connect(super::socket_path()) {
            break stream;
        }
        // Exiting early is fine if it lost a start-up race to another daemon
        // (the loop connects to that one); otherwise it failed.
        if let Ok(Some(code)) = child.try_wait() {
            if let Ok(stream) = UnixStream::connect(super::socket_path()) {
                break stream;
            }
            return Err(anyhow!(
                "wp-engine daemon exited during start-up ({code}) — see {}",
                log_path.display()
            ));
        }
        if Instant::now() >= deadline {
            return Err(anyhow!(
                "wp-engine daemon did not come up within {}s — see {}",
                SPAWN_TIMEOUT.as_secs(),
                log_path.display()
            ));
        }
        std::thread::sleep(Duration::from_millis(50));
    };

    // Reap it if it exits while this process (e.g. the GUI) is still alive.
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(stream)
}
