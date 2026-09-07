//! Battery and session-lock detection for `engine::playback_gate`'s
//! `on_battery`/`on_lock_or_sleep` conditions — see that module's doc for
//! where these come from (the Ghidra report's Follow-up (ff)).

use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::time::Duration;

const POLL_INTERVAL: Duration = Duration::from_secs(2);

/// True when running on battery power. Linux: `/sys/class/power_supply`
/// (a real "Mains" supply reporting `online` wins over any battery's own
/// status, so a laptop on AC with a battery present still reads as "not on
/// battery"). macOS: `pmset -g batt`'s own "Battery Power"/"AC Power"
/// wording. `false` (never pauses) everywhere else, or when neither source
/// is readable — a desktop with no battery at all is the common case this
/// must not misreport.
pub fn is_on_battery() -> bool {
    #[cfg(target_os = "linux")]
    {
        linux_on_battery().unwrap_or(false)
    }
    #[cfg(target_os = "macos")]
    {
        macos_on_battery().unwrap_or(false)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        false
    }
}

#[cfg(target_os = "linux")]
fn linux_on_battery() -> Option<bool> {
    let dir = std::fs::read_dir("/sys/class/power_supply").ok()?;
    let mut ac_online = None;
    let mut any_discharging = false;
    for entry in dir.flatten() {
        let path = entry.path();
        let kind = std::fs::read_to_string(path.join("type")).unwrap_or_default();
        match kind.trim() {
            "Mains" | "USB" => {
                let online = std::fs::read_to_string(path.join("online"))
                    .ok()
                    .map(|s| s.trim() == "1");
                if let Some(online) = online {
                    // Any supply reporting online wins immediately — a
                    // USB-PD dock being unplugged while a Mains adapter is
                    // still connected (or vice versa) must not flip this.
                    if online {
                        return Some(false);
                    }
                    ac_online.get_or_insert(false);
                }
            }
            "Battery" => {
                let status = std::fs::read_to_string(path.join("status")).unwrap_or_default();
                if status.trim() == "Discharging" {
                    any_discharging = true;
                }
            }
            _ => {}
        }
    }
    // A real AC/USB supply that's present but not online is authoritative
    // (unplugged, regardless of what any battery's own status says — some
    // drivers report a battery as "Not charging" rather than "Discharging"
    // while genuinely on AC). Only fall back to the battery's own status
    // when no AC/USB supply exists to ask at all.
    match ac_online {
        Some(_) => Some(true),
        None => Some(any_discharging),
    }
}

#[cfg(target_os = "macos")]
fn macos_on_battery() -> Option<bool> {
    let out = std::process::Command::new("pmset")
        .arg("-g")
        .arg("batt")
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    // First line reads e.g. "Now drawing from 'Battery Power'" or
    // "'AC Power'" — the only part of the output with fixed wording across
    // macOS versions.
    Some(text.contains("Battery Power"))
}

/// Owns the background polling thread; drop to stop it (mirrors
/// `engine::media::MediaWatcher`'s exact lifetime/polling convention).
/// Linux-only — session lock state is a freedesktop.org logind convention
/// with no macOS equivalent this codebase reaches for elsewhere either
/// (see `engine::media`'s own "Linux-only" precedent).
#[cfg(target_os = "linux")]
pub struct LockWatcher {
    rx: Receiver<bool>,
    last: bool,
}

#[cfg(target_os = "linux")]
impl LockWatcher {
    /// Connects to the system bus and starts polling. `None` if the bus
    /// itself is unreachable (no logind on this system, or no permission) —
    /// matches `MediaWatcher::start`'s own graceful give-up.
    pub fn start() -> Option<Self> {
        let conn = zbus::blocking::Connection::system().ok()?;
        let (tx, rx) = sync_channel(4);
        std::thread::Builder::new()
            .name("session-lock".into())
            .spawn(move || watch_loop(conn, tx))
            .ok()?;
        Some(Self { rx, last: false })
    }

    /// The current locked state — non-blocking, draining to the newest
    /// pending update (a lock/unlock event is fully described by its
    /// latest value, same reasoning `MediaWatcher::try_recv` uses).
    pub fn is_locked(&mut self) -> bool {
        while let Ok(locked) = self.rx.try_recv() {
            self.last = locked;
        }
        self.last
    }
}

#[cfg(target_os = "linux")]
fn watch_loop(conn: zbus::blocking::Connection, tx: SyncSender<bool>) {
    let mut last = None;
    loop {
        let locked = any_session_locked(&conn).unwrap_or(false);
        if last != Some(locked) {
            if tx.send(locked).is_err() {
                return;
            }
            last = Some(locked);
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// `true` if any of the current user's logind sessions reports
/// `LockedHint` — deliberately not narrowed to "the seat's active session"
/// (a real distinction logind makes) since a single-graphical-session
/// desktop, the overwhelming common case, has only one candidate anyway,
/// and ORing every session of this UID together degrades gracefully
/// (multi-session setups just get the more conservative "any locked"
/// reading) rather than needing seat-arbitration logic this feature
/// doesn't otherwise depend on.
#[cfg(target_os = "linux")]
fn any_session_locked(conn: &zbus::blocking::Connection) -> Option<bool> {
    let uid = unsafe { libc::getuid() };
    let msg = conn
        .call_method(
            Some("org.freedesktop.login1"),
            "/org/freedesktop/login1",
            Some("org.freedesktop.login1.Manager"),
            "ListSessions",
            &(),
        )
        .ok()?;
    let sessions: Vec<(String, u32, String, String, zbus::zvariant::OwnedObjectPath)> =
        msg.body().deserialize().ok()?;

    for (_id, session_uid, _user, _seat, path) in sessions {
        if session_uid != uid {
            continue;
        }
        let props = zbus::blocking::fdo::PropertiesProxy::builder(conn)
            .destination("org.freedesktop.login1")
            .ok()?
            .path(path)
            .ok()?
            .build()
            .ok()?;
        let iface = zbus::names::InterfaceName::try_from("org.freedesktop.login1.Session")
            .expect("valid interface name");
        if let Ok(locked) = props.get(iface, "LockedHint") {
            if bool::try_from(locked).unwrap_or(false) {
                return Some(true);
            }
        }
    }
    Some(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `is_on_battery` must never panic regardless of what this sandbox's
    /// `/sys/class/power_supply` (or lack thereof) actually contains —
    /// the real assertion here is "doesn't crash," since the true/false
    /// answer depends on real hardware this test can't control.
    #[test]
    fn is_on_battery_does_not_panic() {
        let _ = is_on_battery();
    }
}
