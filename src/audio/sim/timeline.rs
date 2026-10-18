//! Keyframed property animations: the timeline Wallpaper Engine's editor
//! records on a property as `{"value": …, "animation": {"c0": [...], "options": {...}}}`.
//!
//! Each component of the property is a channel (`c0`, `c1`, …) of keyframes
//! `{frame, value, back: {enabled, x, y}, front: {enabled, x, y}}`, where
//! `front`/`back` are bezier handles relative to the key (x in frames, y in
//! value units). `options` carries `fps`, `length` (frames), `mode`
//! (`"single"` plays once and holds the last frame, `"loop"`, `"mirror"`),
//! `name`, `startpaused`, timeline `events`, and grouping (`parent`/`children`
//! property keys, which play together).
//!
//! Measured on real content: transform keyframes (origin/scale/angles) are
//! offsets from the property's base value — they end at 0 — while every
//! other property's keyframes are absolute values.

use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Handle {
    enabled: bool,
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Keyframe {
    frame: f32,
    value: f32,
    back: Handle,
    front: Handle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Single,
    Loop,
    Mirror,
}

/// One property's timeline and its playback state.
#[derive(Debug, Clone)]
pub struct PropertyAnimation {
    channels: Vec<Vec<Keyframe>>,
    pub fps: f32,
    /// Timeline length in frames.
    pub length: f32,
    pub mode: Mode,
    pub name: Option<String>,
    /// Property key of the animation this one plays along with.
    pub parent: Option<String>,
    /// Property keys of animations that play along with this one.
    pub children: Vec<String>,
    /// `(frame, name)` events, fired as playback crosses them.
    pub events: Vec<(f32, String)>,
    /// Whether keyframes are offsets from the base value (transforms).
    pub relative: bool,
    /// Playback position in frames.
    pub frame: f32,
    pub playing: bool,
    pub rate: f32,
}

fn handle(v: Option<&Value>) -> Handle {
    let f = |k: &str| {
        v.and_then(|h| h.get(k))
            .and_then(Value::as_f64)
            .unwrap_or(0.0) as f32
    };
    Handle {
        enabled: v
            .and_then(|h| h.get("enabled"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        x: f("x"),
        y: f("y"),
    }
}

impl PropertyAnimation {
    /// Parse the `animation` object of property `key`. `None` when it has no
    /// usable keyframes.
    pub fn parse(key: &str, animation: &Value) -> Option<Self> {
        let m = animation.as_object()?;
        let mut channels: Vec<(usize, Vec<Keyframe>)> = m
            .iter()
            .filter_map(|(k, v)| {
                let idx: usize = k.strip_prefix('c')?.parse().ok()?;
                let mut keys: Vec<Keyframe> = v
                    .as_array()?
                    .iter()
                    .filter_map(|kf| {
                        Some(Keyframe {
                            frame: kf.get("frame")?.as_f64()? as f32,
                            value: kf.get("value")?.as_f64()? as f32,
                            back: handle(kf.get("back")),
                            front: handle(kf.get("front")),
                        })
                    })
                    .collect();
                keys.sort_by(|a, b| a.frame.total_cmp(&b.frame));
                Some((idx, keys))
            })
            .collect();
        channels.sort_by_key(|(i, _)| *i);
        if channels.iter().all(|(_, k)| k.is_empty()) {
            return None;
        }
        let opts = m.get("options");
        let opt_f = |k: &str| opts.and_then(|o| o.get(k)).and_then(Value::as_f64);
        let opt_s = |k: &str| opts.and_then(|o| o.get(k)).and_then(Value::as_str);
        let last_key = channels
            .iter()
            .flat_map(|(_, k)| k.last())
            .map(|k| k.frame)
            .fold(0.0f32, f32::max);
        let fps = opt_f("fps").filter(|f| *f > 0.0).unwrap_or(30.0) as f32;
        let length = opt_f("length")
            .map(|l| l as f32)
            .filter(|l| *l > 0.0)
            .unwrap_or(last_key.max(1.0));
        let leaf = key.rsplit('.').next().unwrap_or(key);
        Some(Self {
            channels: channels.into_iter().map(|(_, k)| k).collect(),
            fps,
            length,
            mode: match opt_s("mode") {
                Some("loop") => Mode::Loop,
                Some("mirror") => Mode::Mirror,
                _ => Mode::Single,
            },
            name: opt_s("name").map(str::to_string),
            parent: opts
                .and_then(|o| o.get("parent"))
                .and_then(|p| p.get("key"))
                .and_then(Value::as_str)
                .map(str::to_string),
            children: opts
                .and_then(|o| o.get("children"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|c| c.get("key").and_then(Value::as_str).map(str::to_string))
                .collect(),
            events: opts
                .and_then(|o| o.get("events"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|e| {
                    Some((
                        e.get("frame")?.as_f64()? as f32,
                        e.get("name")?.as_str()?.to_string(),
                    ))
                })
                .collect(),
            relative: matches!(leaf, "origin" | "scale" | "angles"),
            frame: 0.0,
            playing: !opts
                .and_then(|o| o.get("startpaused"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            rate: 1.0,
        })
    }

    pub fn components(&self) -> usize {
        self.channels.len()
    }

    /// Advance playback by `dt` seconds; returns the names of events crossed.
    pub fn advance(&mut self, dt: f32) -> Vec<String> {
        if !self.playing || dt <= 0.0 {
            return Vec::new();
        }
        let before = self.frame;
        let span = self.length.max(1e-3);
        let mut after = before + dt * self.fps * self.rate;
        let mut crossed: Vec<String> = Vec::new();
        let mut collect = |from: f32, to: f32, crossed: &mut Vec<String>| {
            for (f, name) in &self.events {
                if *f > from && *f <= to {
                    crossed.push(name.clone());
                }
            }
        };
        match self.mode {
            Mode::Single => {
                if after >= span {
                    after = span;
                    self.playing = false;
                }
                collect(before, after, &mut crossed);
            }
            Mode::Loop => {
                collect(before, after.min(span), &mut crossed);
                if after >= span {
                    let wrapped = after.rem_euclid(span);
                    collect(-1.0, wrapped, &mut crossed);
                    after = wrapped;
                }
            }
            // Mirror keeps an unbounded phase in `frame` and folds it at
            // sample time; events fire on the forward leg only.
            Mode::Mirror => {
                let fold = |p: f32| {
                    let t = p.rem_euclid(2.0 * span);
                    if t > span {
                        2.0 * span - t
                    } else {
                        t
                    }
                };
                if (before.rem_euclid(2.0 * span)) < span {
                    collect(fold(before), fold(after.min(before + span)), &mut crossed);
                }
            }
        }
        self.frame = after;
        crossed
    }

    fn position(&self) -> f32 {
        match self.mode {
            Mode::Mirror => {
                let span = self.length.max(1e-3);
                let t = self.frame.rem_euclid(2.0 * span);
                if t > span {
                    2.0 * span - t
                } else {
                    t
                }
            }
            _ => self.frame,
        }
    }

    /// Current value of each channel.
    pub fn sample(&self) -> Vec<f32> {
        let f = self.position();
        self.channels
            .iter()
            .map(|keys| sample_channel(keys, f))
            .collect()
    }

    /// Control from `thisLayer.getAnimation(...)`.
    pub fn control(&mut self, op: &str, frame: Option<f32>) {
        match op {
            "play" => {
                if self.mode == Mode::Single && self.frame >= self.length {
                    self.frame = 0.0;
                }
                self.playing = true;
            }
            "pause" => self.playing = false,
            "stop" => {
                self.playing = false;
                self.frame = 0.0;
            }
            "frame" => {
                if let Some(f) = frame {
                    self.frame = f.clamp(0.0, self.length);
                }
            }
            _ => {}
        }
    }
}

fn sample_channel(keys: &[Keyframe], f: f32) -> f32 {
    match keys {
        [] => 0.0,
        [k] => k.value,
        _ => {
            if f <= keys[0].frame {
                return keys[0].value;
            }
            let last = keys[keys.len() - 1];
            if f >= last.frame {
                return last.value;
            }
            let i = keys.partition_point(|k| k.frame <= f).saturating_sub(1);
            segment(&keys[i], &keys[i + 1], f)
        }
    }
}

/// Value between two keys at frame `f`: a cubic bezier through the keys'
/// handles (a disabled handle sits on its key, so two disabled handles give
/// a straight line).
fn segment(a: &Keyframe, b: &Keyframe, f: f32) -> f32 {
    let span = (b.frame - a.frame).max(1e-6);
    let (p1x, p1y) = if a.front.enabled {
        (a.frame + a.front.x.clamp(0.0, span), a.value + a.front.y)
    } else {
        (a.frame, a.value)
    };
    let (p2x, p2y) = if b.back.enabled {
        (b.frame + b.back.x.clamp(-span, 0.0), b.value + b.back.y)
    } else {
        (b.frame, b.value)
    };
    if !a.front.enabled && !b.back.enabled {
        return a.value + (b.value - a.value) * ((f - a.frame) / span);
    }
    let bez = |p0: f32, p1: f32, p2: f32, p3: f32, t: f32| {
        let u = 1.0 - t;
        u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t * p3
    };
    // x(t) is monotone with the handles clamped inside the segment: bisect.
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..32 {
        let mid = 0.5 * (lo + hi);
        if bez(a.frame, p1x, p2x, b.frame, mid) < f {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    bez(a.value, p1y, p2y, b.value, 0.5 * (lo + hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anim(json: &str) -> PropertyAnimation {
        PropertyAnimation::parse("alpha", &serde_json::from_str(json).unwrap()).unwrap()
    }

    #[test]
    fn linear_keys_interpolate_and_single_mode_holds() {
        let mut a = anim(
            r#"{"c0":[{"frame":0,"value":0},{"frame":10,"value":1}],"options":{"fps":10,"length":10,"mode":"single"}}"#,
        );
        a.advance(0.5);
        assert!((a.sample()[0] - 0.5).abs() < 1e-5);
        a.advance(5.0);
        assert_eq!(a.sample()[0], 1.0);
        assert!(!a.playing);
    }

    #[test]
    fn loop_wraps_and_fires_events() {
        let mut a = anim(
            r#"{"c0":[{"frame":0,"value":0},{"frame":10,"value":10}],"options":{"fps":10,"length":10,"mode":"loop","events":[{"frame":5,"name":"half"}]}}"#,
        );
        assert_eq!(a.advance(0.6), vec!["half".to_string()]);
        a.advance(0.6);
        assert!((a.sample()[0] - 2.0).abs() < 1e-3);
        assert!(a.playing);
    }

    #[test]
    fn mirror_ping_pongs() {
        let mut a = anim(
            r#"{"c0":[{"frame":0,"value":0},{"frame":10,"value":10}],"options":{"fps":10,"length":10,"mode":"mirror"}}"#,
        );
        a.advance(1.5);
        assert!((a.sample()[0] - 5.0).abs() < 1e-3);
    }

    #[test]
    fn bezier_handles_ease() {
        let a = anim(
            r#"{"c0":[{"frame":0,"value":0,"front":{"enabled":true,"x":5,"y":0}},{"frame":10,"value":1,"back":{"enabled":true,"x":-5,"y":0}}],"options":{"fps":10,"length":10}}"#,
        );
        let mut early = a.clone();
        early.frame = 2.0;
        let mut mid = a.clone();
        mid.frame = 5.0;
        // Ease-in: slower than linear at the start, symmetric in the middle.
        assert!(early.sample()[0] < 0.2);
        assert!((mid.sample()[0] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn start_paused_and_control() {
        let mut a = anim(
            r#"{"c0":[{"frame":0,"value":0},{"frame":10,"value":1}],"options":{"fps":10,"length":10,"startpaused":true,"name":"hover"}}"#,
        );
        a.advance(0.5);
        assert_eq!(a.sample()[0], 0.0);
        a.control("play", None);
        a.advance(0.5);
        assert!(a.sample()[0] > 0.0);
        a.control("stop", None);
        assert_eq!(a.frame, 0.0);
        assert_eq!(a.name.as_deref(), Some("hover"));
    }

    #[test]
    fn transforms_are_relative() {
        let a = PropertyAnimation::parse(
            "origin",
            &serde_json::json!({"c0": [{"frame": 0, "value": 5}], "options": {}}),
        )
        .unwrap();
        assert!(a.relative);
    }

    #[test]
    fn empty_channels_are_rejected() {
        assert!(
            PropertyAnimation::parse("ratio", &serde_json::json!({"c0": [], "options": {}}))
                .is_none()
        );
    }
}
