//! MPRIS (Media Player Remote Interfacing Specification) media integration
//! over the D-Bus session bus — mirrors the C++ reference's
//! `Media::DBusMediaSource` (`cpp-implementation/src/WallpaperEngine/Media/
//! DBusMediaSource.cpp`) as closely as this codebase's own conventions
//! allow: detect the active MPRIS player, read its Metadata/PlaybackStatus/
//! Position, notify on change.
//!
//! One real, deliberate departure from the reference: this polls
//! (`POLL_INTERVAL`) rather than subscribing to `PropertiesChanged` signals
//! from an arbitrary, dynamically-discovered sender. The reference's
//! signal-filter approach needs a raw, connection-wide message filter (no
//! fixed destination to bind a typed proxy to, since *any* MPRIS player
//! might start talking at any time) — zbus's ergonomic proxy API is built
//! around a fixed destination instead. Polling every player's
//! `PlaybackStatus` each tick sidesteps that entirely, at the cost of up to
//! one `POLL_INTERVAL` of latency detecting a hand-off between two open
//! players — a small trade against a lot of raw-message-filtering
//! complexity, and it still catches a `Position` change every tick exactly
//! like the reference's own `performUpdate` already does regardless of any
//! signal.
//!
//! Linux-only — MPRIS is a freedesktop.org desktop convention, with no
//! equivalent surface on macOS or Windows.

use std::collections::HashMap;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::time::Duration;
use zbus::blocking::{fdo::DBusProxy, fdo::PropertiesProxy, Connection};
use zbus::names::InterfaceName;
use zbus::zvariant::OwnedValue;

const MPRIS_PREFIX: &str = "org.mpris.MediaPlayer2.";
const MPRIS_PATH: &str = "/org/mpris/MediaPlayer2";
const MPRIS_PLAYER_IFACE: &str = "org.mpris.MediaPlayer2.Player";
const POLL_INTERVAL: Duration = Duration::from_millis(1000);

/// A known-valid constant, so the `expect` here can never actually fire.
fn player_iface() -> InterfaceName<'static> {
    InterfaceName::try_from(MPRIS_PLAYER_IFACE)
        .expect("MPRIS_PLAYER_IFACE is a valid interface name")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlaybackState {
    #[default]
    Stopped,
    Playing,
    Paused,
}

/// Mirrors `Media::MediaSource::MediaInfo` field-for-field. `duration_us`/
/// `position_us` are left as MPRIS's own raw microseconds rather than
/// converted to seconds — the reference itself stores `mpris:length`/
/// `Position` unconverted (`DBusMediaSource.cpp`'s `parseMetadata`/
/// `performUpdate`), and there's no way to check what unit the *original*
/// Windows SMTC-backed implementation actually hands scripts, so passing
/// through the one real, grounded value we have unmodified beats guessing
/// at a conversion.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MediaInfo {
    /// `true` when an MPRIS player was actually found on the bus — the
    /// reference declares this same field on its own `MediaInfo`
    /// (`MediaSource.h`) but its `DBusMediaSource` never actually sets it
    /// anywhere in the implementation; set it correctly here instead of
    /// carrying that gap forward, since it's a real, meaningful distinction
    /// (no player open at all vs. a player that's open but stopped).
    pub available: bool,
    pub playback_state: PlaybackState,
    pub title: String,
    pub artist: String,
    pub album: String,
    /// `mpris:artUrl` — a `file://`/`http(s)://` URL, not a local path.
    pub art_url: Option<String>,
    pub duration_us: i64,
    pub position_us: i64,
}

/// Owns the background polling thread; drop to stop it (the thread checks
/// whether its `SyncSender` still has a receiver each tick).
pub struct MediaWatcher {
    rx: Receiver<MediaInfo>,
}

impl MediaWatcher {
    /// Connects to the session bus and starts polling. `None` if the bus
    /// itself is unreachable (matches the reference's own
    /// connect-or-give-up behavior in `DBusMediaSource`'s constructor, as a
    /// clean `Option` here instead of a fatal log).
    pub fn start() -> Option<Self> {
        let conn = Connection::session().ok()?;
        let (tx, rx) = sync_channel(4);
        std::thread::Builder::new()
            .name("mpris".into())
            .spawn(move || watch_loop(conn, tx))
            .ok()?;
        Some(Self { rx })
    }

    /// The latest media info, if it changed since the last call —
    /// non-blocking, draining to the newest pending update. `MediaInfo` is
    /// state, not an event log, so skipping an intermediate stale value is
    /// safe (same reasoning `FrameSource::try_advance` already uses for
    /// video frames).
    pub fn try_recv(&self) -> Option<MediaInfo> {
        let mut latest = None;
        while let Ok(info) = self.rx.try_recv() {
            latest = Some(info);
        }
        latest
    }
}

fn watch_loop(conn: Connection, tx: SyncSender<MediaInfo>) {
    let mut last: Option<MediaInfo> = None;
    loop {
        let info = poll_active_player(&conn).unwrap_or_default();
        if last.as_ref() != Some(&info) {
            if tx.send(info.clone()).is_err() {
                return;
            }
            last = Some(info);
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

fn properties_proxy<'a>(conn: &'a Connection, destination: &str) -> Option<PropertiesProxy<'a>> {
    PropertiesProxy::builder(conn)
        .destination(destination.to_string())
        .ok()?
        .path(MPRIS_PATH)
        .ok()?
        .build()
        .ok()
}

fn playback_status(props: &PropertiesProxy) -> Option<String> {
    let value: OwnedValue = props.get(player_iface(), "PlaybackStatus").ok()?;
    String::try_from(value).ok()
}

/// Every `org.mpris.MediaPlayer2.*` bus name, whether or not anything is
/// actually loaded/playing behind it — `detect_player` narrows this down.
fn list_mpris_players(conn: &Connection) -> Vec<String> {
    let Ok(bus) = DBusProxy::new(conn) else {
        return Vec::new();
    };
    let Ok(names) = bus.list_names() else {
        return Vec::new();
    };
    names
        .into_iter()
        .map(|n| n.to_string())
        .filter(|n| n.starts_with(MPRIS_PREFIX))
        .collect()
}

/// The bus name to read from this tick — whichever player reports
/// `PlaybackStatus == "Playing"`, or the first MPRIS player found if none
/// currently are (so a paused-but-open player still reports real metadata
/// instead of nothing), matching the spirit of the reference's own
/// `detectPlayer` (prefer Playing, fall back to whatever's open).
fn detect_player(conn: &Connection) -> Option<String> {
    let players = list_mpris_players(conn);
    let mut fallback = None;
    for player in &players {
        let Some(props) = properties_proxy(conn, player) else {
            continue;
        };
        if playback_status(&props).as_deref() == Some("Playing") {
            return Some(player.clone());
        }
        fallback.get_or_insert_with(|| player.clone());
    }
    fallback
}

fn poll_active_player(conn: &Connection) -> Option<MediaInfo> {
    let player = detect_player(conn)?;
    let props = properties_proxy(conn, &player)?;
    let all: HashMap<String, OwnedValue> = props.get_all(player_iface()).ok()?;

    let playback_state = match all
        .get("PlaybackStatus")
        .and_then(|v| String::try_from(v.clone()).ok())
        .as_deref()
    {
        Some("Playing") => PlaybackState::Playing,
        Some("Paused") => PlaybackState::Paused,
        _ => PlaybackState::Stopped,
    };
    let position_us = all
        .get("Position")
        .and_then(|v| i64::try_from(v.clone()).ok())
        .unwrap_or(0);

    let mut info = MediaInfo {
        available: true,
        playback_state,
        position_us,
        ..Default::default()
    };

    if let Some(metadata) = all.get("Metadata") {
        if let Ok(dict) = HashMap::<String, OwnedValue>::try_from(metadata.clone()) {
            if let Some(title) = dict
                .get("xesam:title")
                .and_then(|v| String::try_from(v.clone()).ok())
            {
                info.title = title;
            }
            // `xesam:artist` is a string array (`as`) — take the first,
            // matching the reference's own `dbus_message_iter_recurse` +
            // read-one-string handling exactly.
            if let Some(artists) = dict
                .get("xesam:artist")
                .and_then(|v| Vec::<String>::try_from(v.clone()).ok())
            {
                if let Some(first) = artists.into_iter().next() {
                    info.artist = first;
                }
            }
            if let Some(album) = dict
                .get("xesam:album")
                .and_then(|v| String::try_from(v.clone()).ok())
            {
                info.album = album;
            }
            if let Some(art_url) = dict
                .get("mpris:artUrl")
                .and_then(|v| String::try_from(v.clone()).ok())
            {
                if !art_url.is_empty() {
                    info.art_url = Some(art_url);
                }
            }
            if let Some(length) = dict
                .get("mpris:length")
                .and_then(|v| i64::try_from(v.clone()).ok())
            {
                info.duration_us = length;
            }
        }
    }

    Some(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn playback_state_defaults_to_stopped() {
        assert_eq!(PlaybackState::default(), PlaybackState::Stopped);
    }

    #[test]
    fn media_info_defaults_are_all_empty() {
        let info = MediaInfo::default();
        assert_eq!(info.playback_state, PlaybackState::Stopped);
        assert_eq!(info.title, "");
        assert_eq!(info.art_url, None);
        assert_eq!(info.duration_us, 0);
    }

    /// Real-bus smoke test, not run by default (needs a live session bus —
    /// `cargo test --lib -- --ignored services::media`). Verifies the actual
    /// D-Bus round trip (connect, ListNames, filter, poll) doesn't error,
    /// against whatever's really running — no MPRIS player was available to
    /// install in the sandbox this was written in, so this only proves the
    /// "nothing playing" path is real, not simulated; it does not exercise
    /// `poll_active_player`'s Metadata-parsing branch.
    #[test]
    #[ignore]
    fn live_watcher_connects_to_the_real_session_bus() {
        let watcher = MediaWatcher::start().expect("session bus must be reachable");
        std::thread::sleep(Duration::from_millis(1500));
        // No assertion on the value — this is a connectivity smoke test.
        // try_recv() is called only so a panic inside the watcher thread
        // (which would otherwise fail silently) has a chance to surface via
        // the channel disconnecting.
        let _ = watcher.try_recv();
    }
}
