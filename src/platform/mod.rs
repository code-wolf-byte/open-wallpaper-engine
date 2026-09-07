pub mod audio;
pub mod device;
pub mod display;
pub mod gpu;
#[cfg(target_os = "macos")]
pub mod macos;
pub mod platform_info;
pub mod power;
pub mod scaler;
#[cfg(target_os = "linux")]
pub mod wayland;
#[cfg(target_os = "linux")]
pub mod x11;

pub use device::GpuDevice;
pub use display::{detect_platform, WallpaperHandle};
pub use scaler::GpuScaler;

// ── Render quality ────────────────────────────────────────────────────────────

/// Controls GPU render quality / source down-sample factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderQuality {
    Ultra,
    High,
    Medium,
    Low,
}

impl RenderQuality {
    /// Fraction of the source image dimensions to keep before uploading to GPU.
    /// Lower values reduce GPU memory bandwidth at the cost of quality.
    pub fn source_scale(self) -> f32 {
        match self {
            Self::Ultra => 1.0,
            Self::High => 0.75,
            Self::Medium => 0.5,
            Self::Low => 0.25,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Ultra => "Ultra",
            Self::High => "High",
            Self::Medium => "Medium",
            Self::Low => "Low",
        }
    }

    /// Case-insensitive inverse of [`Self::label`] — for parsing
    /// `--quality`/`wp-engine config set-quality` CLI input and the saved
    /// per-wallpaper value in `settings::WpSettings`.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|q| q.label().eq_ignore_ascii_case(s))
    }

    pub const ALL: [Self; 4] = [Self::Ultra, Self::High, Self::Medium, Self::Low];
}

// ── Backend ───────────────────────────────────────────────────────────────────

/// Graphics API backend reported by the driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendType {
    Vulkan,
    Metal,
    Dx12,
    OpenGl,
    Unknown,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Vulkan => write!(f, "Vulkan"),
            BackendType::Metal => write!(f, "Metal"),
            BackendType::Dx12 => write!(f, "D3D12"),
            BackendType::OpenGl => write!(f, "OpenGL"),
            BackendType::Unknown => write!(f, "Unknown"),
        }
    }
}

// ── Device type ───────────────────────────────────────────────────────────────

/// Physical category of a GPU adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuDeviceType {
    /// Dedicated GPU (discrete card, highest performance).
    Discrete,
    /// GPU share system memory with the CPU.
    Integrated,
    /// Virtualised / paravirtualised GPU.
    Virtual,
    /// Pure software rasteriser (LLVMpipe, SwiftShader, …).
    Software,
    Unknown,
}

impl std::fmt::Display for GpuDeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuDeviceType::Discrete => write!(f, "Discrete"),
            GpuDeviceType::Integrated => write!(f, "Integrated"),
            GpuDeviceType::Virtual => write!(f, "Virtual"),
            GpuDeviceType::Software => write!(f, "Software"),
            GpuDeviceType::Unknown => write!(f, "Unknown"),
        }
    }
}

// ── Adapter info ──────────────────────────────────────────────────────────────

/// Information about a GPU adapter visible to the process.
///
/// One entry is returned per (physical-device, backend) combination — on
/// Linux a single card can appear as both Vulkan and OpenGL.
#[derive(Debug, Clone)]
pub struct GpuAdapter {
    pub name: String,
    pub backend: BackendType,
    pub device_type: GpuDeviceType,
    /// PCI device ID (0 if unavailable).
    pub device_id: u32,
    /// PCI vendor ID (0 if unavailable).
    pub vendor_id: u32,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Enumerate every GPU adapter available on this system.
///
/// Fast and non-blocking — only queries driver metadata, no device is
/// created and no GPU work is performed.
pub fn probe_adapters() -> Vec<GpuAdapter> {
    gpu::enumerate_adapters()
}

/// Open the highest-performance GPU device available.
///
/// Prefers a discrete GPU; falls back to integrated, then software.
pub fn open_device() -> anyhow::Result<GpuDevice> {
    GpuDevice::open_best()
}
