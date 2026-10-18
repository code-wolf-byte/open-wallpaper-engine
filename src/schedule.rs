//! Playlist playback: which item of a playlist should be showing, and when
//! to move on. Pure logic over a [`Playlist`] and the local clock, so the
//! daemon only has to apply what it says.
//!
//! Modes (`Playlist::mode`):
//! - `timer` — advance every `delay_minutes`.
//! - `daytime` — `times[i]` (`"HH:MM"`) is when item `i` starts each day;
//!   the item showing is the one whose start time most recently passed.
//! - `dayofweek` — item `i` shows on weekday `i` (Monday = 0), wrapping.
//! - `login` — advance once each time playback starts (the daemon starting
//!   with the session), remembering the position between runs.
//!
//! `order` is `sequential` or `random` (random never repeats the item just
//! shown when there is a choice).

use crate::settings::Playlist;
use std::time::{Duration, Instant};

/// Local wall-clock facts the schedule depends on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalTime {
    /// Seconds since local midnight.
    pub seconds_of_day: u32,
    /// Monday = 0 … Sunday = 6.
    pub weekday: u32,
}

impl LocalTime {
    pub fn now() -> Self {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let local = secs + local_offset(secs);
        let days = local.div_euclid(86_400);
        Self {
            seconds_of_day: local.rem_euclid(86_400) as u32,
            // 1970-01-01 was a Thursday (Monday = 0 → Thursday = 3).
            weekday: (days + 3).rem_euclid(7) as u32,
        }
    }
}

#[cfg(unix)]
fn local_offset(t: i64) -> i64 {
    let time = t as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: localtime_r only writes into `tm`.
    if unsafe { libc::localtime_r(&time, &mut tm).is_null() } {
        0
    } else {
        tm.tm_gmtoff
    }
}

#[cfg(not(unix))]
fn local_offset(_t: i64) -> i64 {
    0
}

/// Parse `"HH:MM"` (or `"HH"`) into seconds of the day.
pub fn parse_time_of_day(s: &str) -> Option<u32> {
    let (h, m) = match s.trim().split_once(':') {
        Some((h, m)) => (h.parse::<u32>().ok()?, m.parse::<u32>().ok()?),
        None => (s.trim().parse::<u32>().ok()?, 0),
    };
    (h < 24 && m < 60).then_some(h * 3600 + m * 60)
}

/// A playlist being played.
#[derive(Debug, Clone)]
pub struct PlaylistRun {
    pub name: String,
    pub playlist: Playlist,
    /// Index into `playlist.items` currently showing.
    pub index: usize,
    /// When the current item started (timer mode counts from here).
    switched_at: Instant,
}

impl PlaylistRun {
    /// Start playing. `resume_index` is where a previous run left off
    /// (`login` mode advances from it).
    pub fn start(
        name: &str,
        playlist: Playlist,
        resume_index: Option<usize>,
        now: LocalTime,
    ) -> Option<Self> {
        if playlist.items.is_empty() {
            return None;
        }
        let mut run = Self {
            name: name.to_string(),
            playlist,
            index: 0,
            switched_at: Instant::now(),
        };
        run.index = match run.playlist.mode.as_str() {
            "daytime" => run.scheduled_index(now).unwrap_or(0),
            "dayofweek" => now.weekday as usize % run.len(),
            "login" => match resume_index {
                Some(i) => run.step_from(i),
                None => run.first(),
            },
            _ => run.first(),
        };
        Some(run)
    }

    fn len(&self) -> usize {
        self.playlist.items.len()
    }

    fn random(&self) -> bool {
        self.playlist.order == "random"
    }

    fn first(&self) -> usize {
        if self.random() {
            fastrand::usize(..self.len())
        } else {
            0
        }
    }

    /// The item after `i` in this playlist's order.
    fn step_from(&self, i: usize) -> usize {
        let n = self.len();
        if self.random() && n > 1 {
            let mut j = fastrand::usize(..n - 1);
            if j >= i % n {
                j += 1;
            }
            j
        } else {
            (i + 1) % n
        }
    }

    fn scheduled_index(&self, now: LocalTime) -> Option<usize> {
        let times: Vec<(usize, u32)> = self
            .playlist
            .times
            .iter()
            .enumerate()
            .take(self.len())
            .filter_map(|(i, t)| parse_time_of_day(t).map(|s| (i, s)))
            .collect();
        if times.is_empty() {
            return None;
        }
        // Latest start at or before now; before the first start of the day,
        // the previous day's last item is still showing.
        times
            .iter()
            .filter(|(_, s)| *s <= now.seconds_of_day)
            .max_by_key(|(_, s)| *s)
            .or_else(|| times.iter().max_by_key(|(_, s)| *s))
            .map(|(i, _)| *i)
    }

    pub fn current(&self) -> &str {
        &self.playlist.items[self.index]
    }

    /// The item to switch to now, if any.
    pub fn due(&self, now: Instant, local: LocalTime) -> Option<usize> {
        let target = match self.playlist.mode.as_str() {
            "daytime" => self.scheduled_index(local)?,
            "dayofweek" => local.weekday as usize % self.len(),
            "login" => return None,
            _ => {
                let delay = Duration::from_secs(u64::from(self.playlist.delay_minutes.max(1)) * 60);
                if now.duration_since(self.switched_at) < delay {
                    return None;
                }
                self.step_from(self.index)
            }
        };
        (target != self.index || self.playlist.mode == "timer").then_some(target)
    }

    /// Move to `index` (after the daemon applied it).
    pub fn switch_to(&mut self, index: usize, now: Instant) {
        self.index = index % self.len();
        self.switched_at = now;
    }

    pub fn next_index(&self) -> usize {
        self.step_from(self.index)
    }

    pub fn previous_index(&self) -> usize {
        (self.index + self.len() - 1) % self.len()
    }
}

/// Where `login`-mode playlists left off, between daemon runs.
pub fn load_positions() -> std::collections::BTreeMap<String, usize> {
    std::fs::read_to_string(positions_path())
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

pub fn save_position(name: &str, index: usize) {
    let mut all = load_positions();
    all.insert(name.to_string(), index);
    let path = positions_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, serde_json::to_string(&all).unwrap_or_default());
}

fn positions_path() -> std::path::PathBuf {
    dirs::state_dir()
        .or_else(dirs::data_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("wp-engine")
        .join("playlists.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn playlist(mode: &str, order: &str, n: usize) -> Playlist {
        Playlist {
            items: (0..n).map(|i| format!("item{i}")).collect(),
            delay_minutes: 1,
            mode: mode.into(),
            order: order.into(),
            times: Vec::new(),
        }
    }

    fn at(h: u32, m: u32, weekday: u32) -> LocalTime {
        LocalTime {
            seconds_of_day: h * 3600 + m * 60,
            weekday,
        }
    }

    #[test]
    fn timer_advances_after_the_delay() {
        let run =
            PlaylistRun::start("p", playlist("timer", "sequential", 3), None, at(0, 0, 0)).unwrap();
        let t0 = Instant::now();
        assert_eq!(run.due(t0, at(0, 0, 0)), None);
        assert_eq!(run.due(t0 + Duration::from_secs(61), at(0, 0, 0)), Some(1));
    }

    #[test]
    fn daytime_picks_the_latest_started_item_and_wraps_overnight() {
        let mut p = playlist("daytime", "sequential", 3);
        p.times = vec!["06:00".into(), "12:30".into(), "20:00".into()];
        let run = PlaylistRun::start("p", p, None, at(13, 0, 0)).unwrap();
        assert_eq!(run.index, 1);
        assert_eq!(run.due(Instant::now(), at(20, 1, 0)), Some(2));
        assert_eq!(run.due(Instant::now(), at(13, 1, 0)), None);
        // 03:00 is still last night's item.
        assert_eq!(run.due(Instant::now(), at(3, 0, 0)), Some(2));
    }

    #[test]
    fn day_of_week_indexes_by_weekday() {
        let run = PlaylistRun::start(
            "p",
            playlist("dayofweek", "sequential", 7),
            None,
            at(9, 0, 4),
        )
        .unwrap();
        assert_eq!(run.current(), "item4");
        assert_eq!(run.due(Instant::now(), at(0, 1, 5)), Some(5));
    }

    #[test]
    fn login_advances_from_the_saved_position() {
        let run = PlaylistRun::start(
            "p",
            playlist("login", "sequential", 3),
            Some(2),
            at(0, 0, 0),
        )
        .unwrap();
        assert_eq!(run.index, 0);
        assert_eq!(
            run.due(Instant::now() + Duration::from_secs(9999), at(0, 0, 0)),
            None
        );
    }

    #[test]
    fn random_order_never_repeats_immediately() {
        let run =
            PlaylistRun::start("p", playlist("timer", "random", 2), None, at(0, 0, 0)).unwrap();
        for _ in 0..20 {
            assert_ne!(run.next_index(), run.index);
        }
    }

    #[test]
    fn parses_times() {
        assert_eq!(parse_time_of_day("07:30"), Some(27_000));
        assert_eq!(parse_time_of_day("7"), Some(25_200));
        assert_eq!(parse_time_of_day("24:00"), None);
    }

    #[test]
    fn empty_playlists_do_not_start() {
        assert!(
            PlaylistRun::start("p", playlist("timer", "sequential", 0), None, at(0, 0, 0))
                .is_none()
        );
    }
}
