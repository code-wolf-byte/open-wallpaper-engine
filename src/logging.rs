//! Verbose, step-by-step diagnostic logging for the whole engine.
//!
//! Every subsystem emits through the [`tracing`] facade with a stable
//! `target` (`scene`, `effect`, `particle`, `shader`, `wallpaper`, `timing`,
//! …) so a single run can be filtered down to whatever stage you're
//! debugging. This module owns the one-time subscriber setup.
//!
//! # Controlling verbosity
//!
//! Two knobs, checked in this order:
//!
//! 1. `RUST_LOG` — if set, it wins outright and uses the full
//!    [`EnvFilter`](tracing_subscriber::EnvFilter) syntax, e.g.
//!    `RUST_LOG=wp_engine::audio::cpu::effect=trace,wp_engine=info`.
//! 2. `-v` flags on the CLI — mapped to a level for the `wp_engine` crate,
//!    with noisy dependencies (wgpu, naga, …) held back unless `-vvv`.
//!
//! | flags  | wp_engine | dependencies |
//! |--------|-----------|--------------|
//! | (none) | info      | warn         |
//! | `-v`   | debug     | warn         |
//! | `-vv`  | trace     | warn         |
//! | `-vvv` | trace     | debug        |

use std::io::IsTerminal;
use std::sync::Once;

use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();

/// Every short `target` the engine logs under. These are deliberately not
/// module paths, so [`EnvFilter`] wouldn't associate them with the
/// `wp_engine` crate on its own — we enumerate them here so the `-v` level
/// applies to our own diagnostics while unrelated dependencies stay quiet.
///
/// Filter one subsystem at runtime with e.g. `RUST_LOG=effect=trace`.
pub const TARGETS: &[&str] = &[
    "cli",       // top-level CLI dispatch
    "app",       // application lifecycle (setup/show/cleanup)
    "platform",  // display-platform detection / probing
    "scaler",    // GPU→CPU frame scaling
    "workshop",  // Steam Workshop scan / project.json parsing
    "content",   // wallpaper content-type resolution
    "scene",     // scene graph + resolved scene loading
    "particle",  // particle system parsing
    "shader",    // GLSL→WGSL transpile / util textures
    "effect",    // effect instance loading + pass compilation
    "pkg",       // .pkg archive loading
    "timing",    // per-frame timing instrumentation (trace)
    "render",    // render loops (GPU/CPU scene)
    "video",     // ffmpeg video decode
    "audio",     // desktop-audio capture + FFT spectrum
    "wallpaper", // Wayland surface / output management
];

/// Install the global tracing subscriber. Idempotent — safe to call from any
/// entry point (the GUI, each CLI subcommand, or a test harness); only the
/// first call takes effect.
///
/// `verbosity` is the count of `-v` flags (0–3+). `RUST_LOG`, when present,
/// overrides the level mapping entirely.
pub fn init(verbosity: u8) {
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(default_filter(verbosity)));

        tracing_subscriber::fmt()
            .with_env_filter(filter)
            // Show the subsystem `target` so `[scene]`/`[effect]` style
            // prefixes are recoverable from the log line itself.
            .with_target(true)
            .with_writer(std::io::stderr)
            // Colour only when stderr is a real terminal; keep logs clean when
            // redirected to a file or piped.
            .with_ansi(std::io::stderr().is_terminal())
            .init();
    });
}

/// Build the `EnvFilter` directive string used when `RUST_LOG` is unset.
///
/// `dep_level` is the global default (applies to third-party crates); the
/// `wp_engine` crate path and each of our short [`TARGETS`] are raised to
/// `crate_level` so our own diagnostics show while dependencies stay quiet
/// until `-vvv`.
fn default_filter(verbosity: u8) -> String {
    let (crate_level, dep_level) = match verbosity {
        0 => ("info", "warn"),
        1 => ("debug", "warn"),
        2 => ("trace", "warn"),
        _ => ("trace", "debug"),
    };

    let mut directives = Vec::with_capacity(TARGETS.len() + 2);
    directives.push(dep_level.to_string());
    directives.push(format!("wp_engine={crate_level}"));
    for target in TARGETS {
        directives.push(format!("{target}={crate_level}"));
    }
    directives.join(",")
}

#[cfg(test)]
mod tests {
    use super::{default_filter, TARGETS};

    #[test]
    fn verbosity_sets_global_default_and_raises_app_targets() {
        // v0: dependencies at warn, our subsystems at info.
        let f0 = default_filter(0);
        assert!(f0.starts_with("warn,"), "global default is warn: {f0}");
        assert!(f0.contains("wp_engine=info"));
        assert!(f0.contains("scene=info") && f0.contains("effect=info"));

        // -vvv raises the global default (dependencies) to debug too.
        assert!(default_filter(3).starts_with("debug,"));

        // Every declared target is present in the directive string.
        let f1 = default_filter(1);
        for t in TARGETS {
            assert!(f1.contains(&format!("{t}=debug")), "missing {t} in {f1}");
        }
    }
}
