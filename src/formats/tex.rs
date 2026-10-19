//! Wallpaper Engine `.tex` container parser.
//!
//! Mirrors `WallpaperEngine::Data::Parsers::TextureParser` /
//! `Data::Assets::Texture` in the C++ reference: `TEXV0005` header,
//! `TEXI0001` metadata, a `TEXB000{1..4}` mipmap container (one mipmap chain
//! per image), and an optional `TEXS000{1..3}` animation/spritesheet table
//! gated by the `IsGif` flag.

use anyhow::{bail, Context, Result};
use image::RgbaImage;
use std::io::{Cursor, Read};

/// Raw pixel formats a mipmap's bytes can be in when there's no FreeImage
/// payload (`freeImageFormat == FIF_UNKNOWN`). IDs match `TextureFormat` in
/// the reference's `Data/Assets/Texture.h`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TexFormat {
    Rgba8, // ARGB8888, stored BGRA in memory
    Rgb888,
    Rgb565,
    Dxt5,
    Dxt3,
    Dxt1,
    Rg88,
    R8,
    Rg1616f,
    R16f,
    Bc7,
    RgBa1010102,
    Rgba16161616f,
    Rgb161616f,
}

impl TexFormat {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::Rgba8),
            1 => Ok(Self::Rgb888),
            2 => Ok(Self::Rgb565),
            4 => Ok(Self::Dxt5),
            6 => Ok(Self::Dxt3),
            7 => Ok(Self::Dxt1),
            8 => Ok(Self::Rg88),
            9 => Ok(Self::R8),
            10 => Ok(Self::Rg1616f),
            11 => Ok(Self::R16f),
            12 => Ok(Self::Bc7),
            13 => Ok(Self::RgBa1010102),
            14 => Ok(Self::Rgba16161616f),
            15 => Ok(Self::Rgb161616f),
            other => bail!("unknown .tex format id: {other}"),
        }
    }
}

/// `FIF` (FreeImage Format) IDs the reference recognizes. When a mipmap's
/// container declares one of these (anything but `Unknown`), its bytes are
/// an encoded image blob (PNG/JPEG/...), not raw pixels in `TexFormat`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum FreeImageFormat {
    Unknown,
    Mp4,
    Other,
}

impl FreeImageFormat {
    fn from_u32(v: u32) -> Self {
        match v {
            // FIF_UNKNOWN is -1 (i.e. u32::MAX) in the reference enum.
            u32::MAX => Self::Unknown,
            35 => Self::Other, // FIF_WEBP doubles as FIF_MP4 in the reference; container decides below
            _ => Self::Other,
        }
    }
}

const CONTAINER_TEXB0001: u32 = 1;
const CONTAINER_TEXB0002: u32 = 2;
const CONTAINER_TEXB0003: u32 = 3;
const CONTAINER_TEXB0004: u32 = 4;

const FLAG_NO_INTERPOLATION: u32 = 1;
const FLAG_CLAMP_UVS: u32 = 2;
const FLAG_IS_GIF: u32 = 4;
const FLAG_CLAMP_UVS_BORDER: u32 = 8;

struct Mipmap {
    width: u32,
    height: u32,
    /// Raw pixel bytes (TexFormat) or an encoded image blob (FreeImage).
    data: Vec<u8>,
}

/// One TEXS-table animation/spritesheet frame: a sub-rect of `images[0]`'s
/// first mipmap, shown for `frametime` seconds.
#[derive(Debug, Clone, Copy)]
pub struct TexFrame {
    pub frametime: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

pub struct TexFile {
    format: TexFormat,
    /// Embedded-video texture (mp4 payload in the first mipmap; WE flag 32
    /// or a TEXB0004 video container). Pixel decoding doesn't apply —
    /// callers extract `video_bytes()` and hand them to ffmpeg.
    is_video: bool,
    free_image: FreeImageFormat,
    flags: u32,
    /// Real (unpadded) image dimensions.
    pub image_width: u32,
    pub image_height: u32,
    /// In-memory (power-of-two padded) texture dimensions.
    pub texture_width: u32,
    pub texture_height: u32,
    /// One mipmap chain per image (`imageCount` images; almost always 1).
    images: Vec<Vec<Mipmap>>,
    frames: Vec<TexFrame>,
}

impl TexFile {
    pub fn parse(data: &[u8]) -> Result<Self> {
        match Self::parse_we(data) {
            Ok(tex) => Ok(tex),
            Err(e) => {
                // Not a real .tex container (e.g. a test fixture that's a
                // bare encoded image) — fall back to sniffing it directly.
                if let Ok(img) = image::load_from_memory(data) {
                    let (w, h) = (img.width(), img.height());
                    return Ok(Self {
                        // free_image != Unknown routes decode_mipmap through
                        // image::load_from_memory instead of the raw BGRA
                        // unswizzle path, so the original encoded bytes (not
                        // already-decoded RGBA) must be stored here.
                        format: TexFormat::Rgba8,
                        is_video: false,
                        free_image: FreeImageFormat::Other,
                        flags: 0,
                        image_width: w,
                        image_height: h,
                        texture_width: w,
                        texture_height: h,
                        images: vec![vec![Mipmap {
                            width: w,
                            height: h,
                            data: data.to_vec(),
                        }]],
                        frames: Vec::new(),
                    });
                }
                Err(e)
            }
        }
    }

    fn parse_we(data: &[u8]) -> Result<Self> {
        let mut cur = Cursor::new(data);

        let magic = read_tag(&mut cur)?;
        if &magic != b"TEXV0005\0" {
            bail!("not a .tex file: expected TEXV0005, got {:?}", magic);
        }
        let sub = read_tag(&mut cur)?;
        if &sub != b"TEXI0001\0" {
            bail!(
                "unexpected .tex sub-container: expected TEXI0001, got {:?}",
                sub
            );
        }

        let format_id = read_u32(&mut cur)?;
        let flags = read_u32(&mut cur)?;
        let texture_width = read_u32(&mut cur)?;
        let texture_height = read_u32(&mut cur)?;
        let image_width = read_u32(&mut cur)?;
        let image_height = read_u32(&mut cur)?;
        let _ignored = read_u32(&mut cur)?;

        let container_magic = read_tag(&mut cur)?;
        let image_count = read_u32(&mut cur)?;

        let mut free_image = FreeImageFormat::Unknown;
        let mut is_video = false;
        let mut container_version = match &container_magic {
            b"TEXB0001\0" => CONTAINER_TEXB0001,
            b"TEXB0002\0" => CONTAINER_TEXB0002,
            b"TEXB0003\0" => CONTAINER_TEXB0003,
            b"TEXB0004\0" => CONTAINER_TEXB0004,
            other => bail!("unknown .tex container: {:?}", other),
        };

        if container_version == CONTAINER_TEXB0004 {
            let fif_id = read_u32(&mut cur)?;
            free_image = FreeImageFormat::from_u32(fif_id);
            is_video = read_u32(&mut cur)? == 1;
            if free_image == FreeImageFormat::Unknown && is_video {
                free_image = FreeImageFormat::Mp4;
            }
            // The reference downgrades to the TEXB0003 mipmap layout unless
            // this is actually an MP4 payload.
            if free_image != FreeImageFormat::Mp4 {
                container_version = CONTAINER_TEXB0003;
            }
        } else if container_version == CONTAINER_TEXB0003 {
            let fif_id = read_u32(&mut cur)?;
            free_image = FreeImageFormat::from_u32(fif_id);
        }

        let is_video_tex = is_video || (flags & 32) != 0;

        let mut images = Vec::with_capacity(image_count as usize);
        for _ in 0..image_count {
            let mipmap_count = read_u32(&mut cur)?;
            let mut mipmaps = Vec::with_capacity(mipmap_count as usize);
            for _ in 0..mipmap_count {
                mipmaps.push(read_mipmap(&mut cur, container_version)?);
            }
            images.push(mipmaps);
        }

        let mut frames = Vec::new();
        if flags & FLAG_IS_GIF != 0 {
            frames = read_frames(&mut cur)?;
        }

        // The raw pixel format only matters when there's no FreeImage payload
        // to decode instead; an unrecognized id is otherwise harmless.
        let format = if free_image == FreeImageFormat::Unknown {
            TexFormat::from_u32(format_id)?
        } else {
            TexFormat::from_u32(format_id).unwrap_or(TexFormat::Rgba8)
        };

        Ok(Self {
            format,
            is_video: is_video_tex,
            free_image,
            flags,
            image_width,
            image_height,
            texture_width,
            texture_height,
            images,
            frames,
        })
    }

    fn decode_mipmap(&self, mip: &Mipmap, channel_alpha: bool) -> Result<RgbaImage> {
        if self.is_video {
            bail!("embedded video texture: decode via video_bytes()/ffmpeg, not as pixels");
        }
        if self.free_image != FreeImageFormat::Unknown {
            let img = image::load_from_memory(&mip.data)
                .context("decoding FreeImage-format .tex payload")?;
            return Ok(img.into_rgba8());
        }
        let rgba = decode_raw(self.format, &mip.data, mip.width, mip.height, channel_alpha);
        RgbaImage::from_raw(mip.width, mip.height, rgba)
            .context("failed to create RgbaImage from decoded texture data")
    }

    /// Decode image 0's first (highest-res) mipmap, cropped to the real
    /// (unpadded) image dimensions.
    pub fn to_rgba(&self) -> Result<RgbaImage> {
        self.to_rgba_with(false)
    }

    fn to_rgba_with(&self, channel_alpha: bool) -> Result<RgbaImage> {
        let mip = self
            .images
            .first()
            .and_then(|mips| mips.first())
            .context("no mipmaps in .tex file")?;
        let img = self.decode_mipmap(mip, channel_alpha)?;

        if self.image_width != img.width() || self.image_height != img.height() {
            if self.image_width <= img.width() && self.image_height <= img.height() {
                Ok(
                    image::imageops::crop_imm(&img, 0, 0, self.image_width, self.image_height)
                        .to_image(),
                )
            } else {
                Ok(img)
            }
        } else {
            Ok(img)
        }
    }

    /// True if this texture carries a TEXS spritesheet/animation table
    /// (the reference's `Texture::isAnimated()`, gated by the `IsGif` flag).
    pub fn is_animated(&self) -> bool {
        self.flags & FLAG_IS_GIF != 0 && !self.frames.is_empty()
    }

    pub fn format(&self) -> TexFormat {
        self.format
    }

    pub fn flags(&self) -> u32 {
        self.flags
    }

    pub fn is_video(&self) -> bool {
        self.is_video
    }

    /// The embedded video container bytes (e.g. mp4) for video textures.
    pub fn video_bytes(&self) -> Option<&[u8]> {
        if !self.is_video {
            return None;
        }
        self.images.first()?.first().map(|m| m.data.as_slice())
    }

    /// Nearest-neighbor sampling instead of the default linear/bilinear.
    pub fn no_interpolation(&self) -> bool {
        self.flags & FLAG_NO_INTERPOLATION != 0
    }

    /// Clamp-to-edge (or clamp-to-border) instead of the default repeat.
    pub fn clamp_uvs(&self) -> bool {
        self.flags & (FLAG_CLAMP_UVS | FLAG_CLAMP_UVS_BORDER) != 0
    }

    pub fn frames(&self) -> &[TexFrame] {
        &self.frames
    }

    /// Decode every animation frame as a separate RGBA image:
    /// - if TEXS spritesheet data is present, crop each frame's sub-rect out
    ///   of image 0's packed texture;
    /// - else if the container holds more than one image, decode each image's
    ///   first mipmap as a frame;
    /// - else fall back to the single `to_rgba()` frame.
    pub fn to_rgba_frames(&self) -> Result<Vec<RgbaImage>> {
        self.to_rgba_frames_with(false)
    }

    /// `to_rgba_frames()` for particle sprites: R8/RG88 (and their f16
    /// variants) become white-RGB/luminance with the channel as alpha,
    /// unconditionally by format — the real engine's `ConvertTexture0Format`
    /// applied by the generic particle shaders (see `decode_raw`).
    pub fn to_particle_rgba_frames(&self) -> Result<Vec<RgbaImage>> {
        self.to_rgba_frames_with(true)
    }

    fn to_rgba_frames_with(&self, channel_alpha: bool) -> Result<Vec<RgbaImage>> {
        if self.is_animated() {
            let atlas = self.to_rgba_atlas(channel_alpha)?;
            let mut frames = Vec::with_capacity(self.frames.len());
            for f in &self.frames {
                let (x, y, w, h) = (
                    f.x.round() as u32,
                    f.y.round() as u32,
                    f.width.round().max(1.0) as u32,
                    f.height.round().max(1.0) as u32,
                );
                if x + w <= atlas.width() && y + h <= atlas.height() {
                    frames.push(image::imageops::crop_imm(&atlas, x, y, w, h).to_image());
                }
            }
            if !frames.is_empty() {
                return Ok(frames);
            }
        }

        if self.images.len() > 1 {
            let mut frames = Vec::with_capacity(self.images.len());
            for mips in &self.images {
                if let Some(mip) = mips.first() {
                    frames.push(self.decode_mipmap(mip, channel_alpha)?);
                }
            }
            if !frames.is_empty() {
                return Ok(frames);
            }
        }

        Ok(vec![self.to_rgba_with(channel_alpha)?])
    }

    /// The uncropped, undecoded-atlas version of `to_rgba()` (image 0's
    /// first mipmap at its native decoded size), used to slice TEXS frames.
    fn to_rgba_atlas(&self, channel_alpha: bool) -> Result<RgbaImage> {
        let mip = self
            .images
            .first()
            .and_then(|mips| mips.first())
            .context("no mipmaps in .tex file")?;
        self.decode_mipmap(mip, channel_alpha)
    }
}

fn decode_raw(
    format: TexFormat,
    data: &[u8],
    width: u32,
    height: u32,
    channel_alpha: bool,
) -> Vec<u8> {
    match format {
        // Despite the reference's "ARGB8888" name, CTexture.cpp uploads raw
        // (non-FreeImage-embedded) bytes for this format directly via
        // `glTexImage2D(..., GL_RGBA, GL_UNSIGNED_BYTE, dataptr)` with no byte
        // reordering — the data on disk is already straight RGBA. A previous
        // R/B swap here was wrong and visibly shifted every wallpaper's hues
        // (teal skies rendering yellow-green, blues rendering magenta/red).
        TexFormat::Rgba8 => data.to_vec(),
        // `channel_alpha` mirrors the real engine's `ConvertTexture0Format`
        // (assets/shaders/common_fragment.h): shaders that opt in — the
        // generic particle/rope-particle/fur shaders — treat single/dual
        // channel formats as alpha carriers, unconditionally by format
        // (CPass.cpp sets the TEX0FORMAT combo from the pixel format alone;
        // the `AlphaChannelPriority` texture flag exists but is never read).
        // R8/R16F → vec4(1, 1, 1, r): white tinted by the particle color,
        // shaped purely by alpha. Without the conversion every sprite draws
        // its full bounding rectangle as an opaque box (black for R8, red-
        // tinted for RG88's [r, g, 0] expansion).
        TexFormat::R8 => data
            .iter()
            .flat_map(|&r| {
                if channel_alpha {
                    [255, 255, 255, r]
                } else {
                    [r, r, r, 255]
                }
            })
            .collect(),
        TexFormat::Rg88 => data
            .chunks(2)
            .flat_map(|rg| {
                let r = rg.first().copied().unwrap_or(0);
                let g = rg.get(1).copied().unwrap_or(0);
                if channel_alpha {
                    // _sample.rrrg: R = luminance, G = alpha (independent of
                    // brightness — e.g. a beam that stays bright along its
                    // whole length but tapers alpha only at the very ends).
                    [r, r, r, g]
                } else {
                    [r, g, 0, 255]
                }
            })
            .collect(),
        TexFormat::Rgb888 => data
            .chunks(3)
            .flat_map(|rgb| {
                let r = rgb.first().copied().unwrap_or(0);
                let g = rgb.get(1).copied().unwrap_or(0);
                let b = rgb.get(2).copied().unwrap_or(0);
                [r, g, b, 255]
            })
            .collect(),
        TexFormat::Rgb565 => data
            .chunks_exact(2)
            .flat_map(|b| rgb565_to_rgba(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        TexFormat::R16f => data
            .chunks_exact(2)
            .flat_map(|b| {
                let v = f16_to_u8(u16::from_le_bytes([b[0], b[1]]));
                if channel_alpha {
                    [255, 255, 255, v]
                } else {
                    [v, v, v, 255]
                }
            })
            .collect(),
        TexFormat::Rg1616f => data
            .chunks_exact(4)
            .flat_map(|b| {
                let r = f16_to_u8(u16::from_le_bytes([b[0], b[1]]));
                let g = f16_to_u8(u16::from_le_bytes([b[2], b[3]]));
                if channel_alpha {
                    [r, r, r, g]
                } else {
                    [r, g, 0, 255]
                }
            })
            .collect(),
        TexFormat::Rgba16161616f => data
            .chunks_exact(8)
            .flat_map(|b| {
                [
                    f16_to_u8(u16::from_le_bytes([b[0], b[1]])),
                    f16_to_u8(u16::from_le_bytes([b[2], b[3]])),
                    f16_to_u8(u16::from_le_bytes([b[4], b[5]])),
                    f16_to_u8(u16::from_le_bytes([b[6], b[7]])),
                ]
            })
            .collect(),
        TexFormat::Rgb161616f => data
            .chunks_exact(6)
            .flat_map(|b| {
                [
                    f16_to_u8(u16::from_le_bytes([b[0], b[1]])),
                    f16_to_u8(u16::from_le_bytes([b[2], b[3]])),
                    f16_to_u8(u16::from_le_bytes([b[4], b[5]])),
                    255,
                ]
            })
            .collect(),
        TexFormat::RgBa1010102 => data
            .chunks_exact(4)
            .flat_map(|b| {
                let v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                let r = ((v & 0x3FF) * 255 / 1023) as u8;
                let g = (((v >> 10) & 0x3FF) * 255 / 1023) as u8;
                let bl = (((v >> 20) & 0x3FF) * 255 / 1023) as u8;
                let a = (((v >> 30) & 0x3) * 255 / 3) as u8;
                [r, g, bl, a]
            })
            .collect(),
        TexFormat::Dxt1 => decode_dxt1(data, width, height),
        TexFormat::Dxt3 => decode_dxt3(data, width, height),
        TexFormat::Dxt5 => decode_dxt5(data, width, height),
        TexFormat::Bc7 => decode_bc7(data, width, height),
    }
}

/// Approximate f16→u8 conversion for display purposes (HDR/linear values
/// outside [0,1] are clamped).
fn f16_to_u8(bits: u16) -> u8 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1F;
    let frac = bits & 0x3FF;
    let value = if exp == 0 {
        (frac as f32) / 1024.0 * 2f32.powi(-14)
    } else if exp == 0x1F {
        if frac == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + frac as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    let value = if sign == 1 { -value } else { value };
    (value.clamp(0.0, 1.0) * 255.0) as u8
}

// ── DXT/BC decoders ────────────────────────────────────────────────────────────

fn rgb565_to_rgba(c: u16) -> [u8; 4] {
    let r = ((c >> 11) & 0x1F) as u8;
    let g = ((c >> 5) & 0x3F) as u8;
    let b = (c & 0x1F) as u8;
    [
        (r << 3) | (r >> 2),
        (g << 2) | (g >> 4),
        (b << 3) | (b >> 2),
        255,
    ]
}

fn lerp_rgb(a: [u8; 4], b: [u8; 4], num_a: u16, num_b: u16, den: u16) -> [u8; 4] {
    [
        ((a[0] as u16 * num_a + b[0] as u16 * num_b) / den) as u8,
        ((a[1] as u16 * num_a + b[1] as u16 * num_b) / den) as u8,
        ((a[2] as u16 * num_a + b[2] as u16 * num_b) / den) as u8,
        255,
    ]
}

fn decode_dxt1(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let bw = ((width + 3) / 4) as usize;
    let bh = ((height + 3) / 4) as usize;
    let mut out = vec![0u8; (width * height * 4) as usize];
    let stride = width as usize * 4;

    for by in 0..bh {
        for bx in 0..bw {
            let offset = (by * bw + bx) * 8;
            if offset + 8 > data.len() {
                break;
            }
            let block = &data[offset..offset + 8];

            let c0 = u16::from_le_bytes([block[0], block[1]]);
            let c1 = u16::from_le_bytes([block[2], block[3]]);
            let lut = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);

            let p0 = rgb565_to_rgba(c0);
            let p1 = rgb565_to_rgba(c1);

            let palette = if c0 > c1 {
                [p0, p1, lerp_rgb(p0, p1, 2, 1, 3), lerp_rgb(p0, p1, 1, 2, 3)]
            } else {
                [p0, p1, lerp_rgb(p0, p1, 1, 1, 2), [0, 0, 0, 0]]
            };

            for py in 0..4u32 {
                for px in 0..4u32 {
                    let x = bx as u32 * 4 + px;
                    let y = by as u32 * 4 + py;
                    if x >= width || y >= height {
                        continue;
                    }
                    let idx = ((lut >> (2 * (py * 4 + px))) & 3) as usize;
                    let dst = y as usize * stride + x as usize * 4;
                    out[dst..dst + 4].copy_from_slice(&palette[idx]);
                }
            }
        }
    }
    out
}

fn decode_dxt3(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let bw = ((width + 3) / 4) as usize;
    let bh = ((height + 3) / 4) as usize;
    let mut out = vec![0u8; (width * height * 4) as usize];
    let stride = width as usize * 4;

    for by in 0..bh {
        for bx in 0..bw {
            let offset = (by * bw + bx) * 16;
            if offset + 16 > data.len() {
                break;
            }
            let block = &data[offset..offset + 16];

            // First 8 bytes: explicit 4-bit alpha for each pixel
            let alpha_bits = u64::from_le_bytes(block[0..8].try_into().unwrap());

            // Next 8 bytes: DXT1 color block
            let c0 = u16::from_le_bytes([block[8], block[9]]);
            let c1 = u16::from_le_bytes([block[10], block[11]]);
            let lut = u32::from_le_bytes([block[12], block[13], block[14], block[15]]);

            let p0 = rgb565_to_rgba(c0);
            let p1 = rgb565_to_rgba(c1);
            let palette = [p0, p1, lerp_rgb(p0, p1, 2, 1, 3), lerp_rgb(p0, p1, 1, 2, 3)];

            for py in 0..4u32 {
                for px in 0..4u32 {
                    let x = bx as u32 * 4 + px;
                    let y = by as u32 * 4 + py;
                    if x >= width || y >= height {
                        continue;
                    }
                    let idx = ((lut >> (2 * (py * 4 + px))) & 3) as usize;
                    let pixel_index = py * 4 + px;
                    let a4 = ((alpha_bits >> (pixel_index * 4)) & 0xF) as u8;
                    let alpha = a4 | (a4 << 4);

                    let dst = y as usize * stride + x as usize * 4;
                    out[dst] = palette[idx][0];
                    out[dst + 1] = palette[idx][1];
                    out[dst + 2] = palette[idx][2];
                    out[dst + 3] = alpha;
                }
            }
        }
    }
    out
}

fn decode_dxt5(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let bw = ((width + 3) / 4) as usize;
    let bh = ((height + 3) / 4) as usize;
    let mut out = vec![0u8; (width * height * 4) as usize];
    let stride = width as usize * 4;

    for by in 0..bh {
        for bx in 0..bw {
            let offset = (by * bw + bx) * 16;
            if offset + 16 > data.len() {
                break;
            }
            let block = &data[offset..offset + 16];

            // Alpha block: 2 reference alphas + 6 bytes of 3-bit indices
            let a0 = block[0];
            let a1 = block[1];
            let alpha_bits = u64::from(block[2])
                | (u64::from(block[3]) << 8)
                | (u64::from(block[4]) << 16)
                | (u64::from(block[5]) << 24)
                | (u64::from(block[6]) << 32)
                | (u64::from(block[7]) << 40);

            let alpha_palette = if a0 > a1 {
                [
                    a0,
                    a1,
                    ((6 * a0 as u16 + a1 as u16) / 7) as u8,
                    ((5 * a0 as u16 + 2 * a1 as u16) / 7) as u8,
                    ((4 * a0 as u16 + 3 * a1 as u16) / 7) as u8,
                    ((3 * a0 as u16 + 4 * a1 as u16) / 7) as u8,
                    ((2 * a0 as u16 + 5 * a1 as u16) / 7) as u8,
                    ((a0 as u16 + 6 * a1 as u16) / 7) as u8,
                ]
            } else {
                [
                    a0,
                    a1,
                    ((4 * a0 as u16 + a1 as u16) / 5) as u8,
                    ((3 * a0 as u16 + 2 * a1 as u16) / 5) as u8,
                    ((2 * a0 as u16 + 3 * a1 as u16) / 5) as u8,
                    ((a0 as u16 + 4 * a1 as u16) / 5) as u8,
                    0,
                    255,
                ]
            };

            // Color block (same as DXT1)
            let c0 = u16::from_le_bytes([block[8], block[9]]);
            let c1 = u16::from_le_bytes([block[10], block[11]]);
            let lut = u32::from_le_bytes([block[12], block[13], block[14], block[15]]);

            let p0 = rgb565_to_rgba(c0);
            let p1 = rgb565_to_rgba(c1);
            let palette = [p0, p1, lerp_rgb(p0, p1, 2, 1, 3), lerp_rgb(p0, p1, 1, 2, 3)];

            for py in 0..4u32 {
                for px in 0..4u32 {
                    let x = bx as u32 * 4 + px;
                    let y = by as u32 * 4 + py;
                    if x >= width || y >= height {
                        continue;
                    }
                    let cidx = ((lut >> (2 * (py * 4 + px))) & 3) as usize;
                    let pixel_index = py * 4 + px;
                    let aidx = ((alpha_bits >> (pixel_index * 3)) & 7) as usize;

                    let dst = y as usize * stride + x as usize * 4;
                    out[dst] = palette[cidx][0];
                    out[dst + 1] = palette[cidx][1];
                    out[dst + 2] = palette[cidx][2];
                    out[dst + 3] = alpha_palette[aidx];
                }
            }
        }
    }
    out
}

fn decode_bc7(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let bw = ((width + 3) / 4) as usize;
    let bh = ((height + 3) / 4) as usize;
    let mut out = vec![0u8; (width * height * 4) as usize];
    let stride = width as usize * 4;

    for by in 0..bh {
        for bx in 0..bw {
            let offset = (by * bw + bx) * 16;
            if offset + 16 > data.len() {
                break;
            }
            let block = &data[offset..offset + 16];

            let mode = block[0].trailing_zeros();
            let rgba = if mode == 6 {
                decode_bc7_mode6(block)
            } else {
                [[128, 128, 128, 255]; 16]
            };

            for py in 0..4u32 {
                for px in 0..4u32 {
                    let x = bx as u32 * 4 + px;
                    let y = by as u32 * 4 + py;
                    if x >= width || y >= height {
                        continue;
                    }
                    let src = (py * 4 + px) as usize;
                    let dst = y as usize * stride + x as usize * 4;
                    out[dst..dst + 4].copy_from_slice(&rgba[src]);
                }
            }
        }
    }
    out
}

fn decode_bc7_mode6(block: &[u8]) -> [[u8; 4]; 16] {
    const WEIGHTS: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

    let bits = u128::from_le_bytes(block[0..16].try_into().unwrap());
    let mut pos = 7; // skip mode bits (bit 6 set)

    let mut endpoints = [[0u8; 4]; 2];
    for ch in 0..4 {
        for ep in 0..2 {
            endpoints[ep][ch] = ((bits >> pos) & 0x7F) as u8;
            pos += 7;
        }
    }
    let p0 = ((bits >> pos) & 1) as u8;
    pos += 1;
    let p1 = ((bits >> pos) & 1) as u8;
    pos += 1;

    for ch in 0..4 {
        endpoints[0][ch] = (endpoints[0][ch] << 1) | p0;
        endpoints[1][ch] = (endpoints[1][ch] << 1) | p1;
    }

    let mut result = [[0u8; 4]; 16];
    for i in 0..16 {
        let nbits = if i == 0 { 3 } else { 4 };
        let idx = ((bits >> pos) & ((1 << nbits) - 1)) as usize;
        pos += nbits;
        let w = WEIGHTS[idx];
        for ch in 0..4 {
            let e0 = endpoints[0][ch] as u32;
            let e1 = endpoints[1][ch] as u32;
            result[i][ch] = (((64 - w) * e0 + w * e1 + 32) >> 6) as u8;
        }
    }
    result
}

// ── Container parsing helpers ─────────────────────────────────────────────────

fn read_mipmap(cur: &mut Cursor<&[u8]>, container_version: u32) -> Result<Mipmap> {
    if container_version == CONTAINER_TEXB0004 {
        let _ignored0 = read_u32(cur)?;
        let _ignored1 = read_u32(cur)?;
        let _json = read_cstring(cur)?;
        let _ignored2 = read_u32(cur)?;
    }

    let width = read_u32(cur)?;
    let height = read_u32(cur)?;

    let mut compression = 0u32;
    let mut uncompressed_size = 0i64;
    if matches!(
        container_version,
        CONTAINER_TEXB0002 | CONTAINER_TEXB0003 | CONTAINER_TEXB0004
    ) {
        compression = read_u32(cur)?;
        uncompressed_size = read_i32(cur)? as i64;
    }

    let compressed_size = read_i32(cur)? as i64;
    if compression == 0 {
        uncompressed_size = compressed_size;
    }

    if uncompressed_size < 0 || compressed_size < 0 {
        bail!("negative .tex mipmap size");
    }

    let data = if compression == 1 {
        let mut compressed = vec![0u8; compressed_size as usize];
        cur.read_exact(&mut compressed)
            .context("reading compressed mipmap data")?;
        lz4_flex::decompress(&compressed, uncompressed_size as usize)
            .context("LZ4 decompress mipmap")?
    } else {
        let mut raw = vec![0u8; uncompressed_size as usize];
        cur.read_exact(&mut raw).context("reading mipmap data")?;
        raw
    };

    Ok(Mipmap {
        width,
        height,
        data,
    })
}

fn read_frames(cur: &mut Cursor<&[u8]>) -> Result<Vec<TexFrame>> {
    let magic = read_tag(cur)?;
    let version = match &magic {
        b"TEXS0001\0" => 1,
        b"TEXS0002\0" => 2,
        b"TEXS0003\0" => 3,
        other => bail!("unknown .tex animation section: {:?}", other),
    };

    let frame_count = read_u32(cur)?;
    if version == 3 {
        let _gif_width = read_u32(cur)?;
        let _gif_height = read_u32(cur)?;
    }

    let mut frames = Vec::with_capacity(frame_count as usize);
    for _ in 0..frame_count {
        if version == 1 {
            let _frame_number = read_u32(cur)?;
            let frametime = read_f32(cur)?;
            let x = read_u32(cur)? as f32;
            let y = read_u32(cur)? as f32;
            let width = read_u32(cur)? as f32;
            let _unk0 = read_u32(cur)?;
            let _unk1 = read_u32(cur)?;
            let height = read_u32(cur)? as f32;
            frames.push(TexFrame {
                frametime,
                x,
                y,
                width,
                height,
            });
        } else {
            let _frame_number = read_u32(cur)?;
            let frametime = read_f32(cur)?;
            let x = read_f32(cur)?;
            let y = read_f32(cur)?;
            let width = read_f32(cur)?;
            let _width2 = read_f32(cur)?;
            let _height2 = read_f32(cur)?;
            let height = read_f32(cur)?;
            frames.push(TexFrame {
                frametime,
                x,
                y,
                width,
                height,
            });
        }
    }
    Ok(frames)
}

/// Reads a fixed 8-byte tag followed by its NUL terminator (WE always writes
/// section names as `"TEXV0005"` + `\0`, 9 bytes total).
fn read_tag(cur: &mut Cursor<&[u8]>) -> Result<[u8; 9]> {
    let mut buf = [0u8; 9];
    cur.read_exact(&mut buf).context("reading .tex tag")?;
    Ok(buf)
}

fn read_cstring(cur: &mut Cursor<&[u8]>) -> Result<String> {
    let mut bytes = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        cur.read_exact(&mut byte).context("reading .tex string")?;
        if byte[0] == 0 {
            break;
        }
        bytes.push(byte[0]);
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn read_u32(cur: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    cur.read_exact(&mut buf)
        .context("unexpected EOF reading u32")?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(cur: &mut Cursor<&[u8]>) -> Result<i32> {
    let mut buf = [0u8; 4];
    cur.read_exact(&mut buf)
        .context("unexpected EOF reading i32")?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32(cur: &mut Cursor<&[u8]>) -> Result<f32> {
    let mut buf = [0u8; 4];
    cur.read_exact(&mut buf)
        .context("unexpected EOF reading f32")?;
    Ok(f32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tag(out: &mut Vec<u8>, tag: &[u8; 8]) {
        out.extend_from_slice(tag);
        out.push(0);
    }

    /// Build a minimal TEXV0005/TEXI0001/TEXB0001 file: 1 image, 1 mipmap,
    /// `format_id`/`flags` as given, with `pixels` as the raw mipmap bytes
    /// (2x2 texture).
    fn minimal_tex_with(format_id: u32, flags: u32, pixels: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        write_tag(&mut out, b"TEXV0005");
        write_tag(&mut out, b"TEXI0001");
        out.extend_from_slice(&format_id.to_le_bytes());
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes()); // texture_width
        out.extend_from_slice(&2u32.to_le_bytes()); // texture_height
        out.extend_from_slice(&2u32.to_le_bytes()); // image_width
        out.extend_from_slice(&2u32.to_le_bytes()); // image_height
        out.extend_from_slice(&0u32.to_le_bytes()); // ignored

        write_tag(&mut out, b"TEXB0001");
        out.extend_from_slice(&1u32.to_le_bytes()); // imageCount = 1
        out.extend_from_slice(&1u32.to_le_bytes()); // mipmapCount = 1

        // mipmap: width, height, compressedSize, then raw bytes (TEXB0001 has
        // no compression/uncompressedSize fields).
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(pixels.len() as i32).to_le_bytes());
        out.extend_from_slice(pixels);
        out
    }

    /// Uncompressed R8 pixels (2x2, value 0x80 everywhere), no flags.
    fn minimal_tex() -> Vec<u8> {
        minimal_tex_with(9, 0, &[0x80u8; 4])
    }

    #[test]
    fn parses_minimal_texb0001_r8() {
        let data = minimal_tex();
        let tex = TexFile::parse(&data).unwrap();
        assert_eq!(tex.image_width, 2);
        assert_eq!(tex.image_height, 2);
        let img = tex.to_rgba().unwrap();
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
        assert_eq!(img.get_pixel(0, 0).0, [0x80, 0x80, 0x80, 255]);
    }

    /// Generic (non-particle) decode: R8 stays a plain opaque luminance
    /// mask, regardless of any texture flag.
    #[test]
    fn r8_generic_decode_is_opaque_luminance() {
        let data = minimal_tex_with(9, 0, &[0x40u8; 4]);
        let img = TexFile::parse(&data).unwrap().to_rgba().unwrap();
        assert_eq!(img.get_pixel(0, 0).0, [0x40, 0x40, 0x40, 255]);
    }

    /// Particle decode mirrors `ConvertTexture0Format`: R8 becomes
    /// vec4(1, 1, 1, r) — white shaped purely by the channel-as-alpha —
    /// unconditionally by format (no flag involved; the real engine keys
    /// the conversion off TEX0FORMAT alone). This is the bug behind
    /// "particle effects have hard black boxes around them": debris/smoke
    /// masks are R8, and decoding them as opaque made every particle draw
    /// its full bounding rectangle as a solid dark box.
    #[test]
    fn r8_particle_decode_is_white_with_channel_alpha() {
        let data = minimal_tex_with(9, 0, &[0x40u8; 4]);
        let frames = TexFile::parse(&data)
            .unwrap()
            .to_particle_rgba_frames()
            .unwrap();
        assert_eq!(frames[0].get_pixel(0, 0).0, [255, 255, 255, 0x40]);
    }

    /// Generic (non-particle) decode: RG88 keeps its plain
    /// two-color-channel expansion (blue=0, fully opaque).
    #[test]
    fn rg88_generic_decode_is_two_channel_color() {
        let pixels = [0x10u8, 0x20, 0x10, 0x20, 0x10, 0x20, 0x10, 0x20];
        let data = minimal_tex_with(8, 0, &pixels);
        let img = TexFile::parse(&data).unwrap().to_rgba().unwrap();
        assert_eq!(img.get_pixel(0, 0).0, [0x10, 0x20, 0, 255]);
    }

    /// Particle decode: RG88 becomes _sample.rrrg — R broadcast to RGB as
    /// luminance, G as alpha (independent of brightness, e.g. a beam sprite
    /// that stays bright along its whole length but tapers alpha only at
    /// the very ends). Decoding it generically instead shows as an opaque
    /// red-tinted box ([r, g, 0, 255] is red-dominant).
    #[test]
    fn rg88_particle_decode_uses_green_as_alpha() {
        let pixels = [0x10u8, 0x20, 0x10, 0x20, 0x10, 0x20, 0x10, 0x20];
        let data = minimal_tex_with(8, 0, &pixels);
        let frames = TexFile::parse(&data)
            .unwrap()
            .to_particle_rgba_frames()
            .unwrap();
        assert_eq!(frames[0].get_pixel(0, 0).0, [0x10, 0x10, 0x10, 0x20]);
    }

    #[test]
    fn falls_back_to_plain_image_for_non_tex_data() {
        // A 1x1 PNG (bare, no TEXV wrapper) should still decode.
        let png = image::RgbaImage::from_pixel(1, 1, image::Rgba([10, 20, 30, 255]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgba8(png)
            .write_to(&mut Cursor::new(&mut bytes), image::ImageOutputFormat::Png)
            .unwrap();
        let tex = TexFile::parse(&bytes).unwrap();
        let img = tex.to_rgba().unwrap();
        assert_eq!(img.get_pixel(0, 0).0, [10, 20, 30, 255]);
    }

    #[test]
    fn rejects_wrong_magic_without_valid_fallback_image() {
        let data = b"not a tex file at all, and not an image either";
        assert!(TexFile::parse(data).is_err());
    }

    /// Build a TEXV0005/TEXI0001/TEXB0003 file wrapping a real PNG payload
    /// (FreeImage format = PNG), exercising the FreeImage decode path.
    fn free_image_tex() -> Vec<u8> {
        let png_img = image::RgbaImage::from_pixel(3, 3, image::Rgba([200, 100, 50, 255]));
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgba8(png_img)
            .write_to(
                &mut Cursor::new(&mut png_bytes),
                image::ImageOutputFormat::Png,
            )
            .unwrap();

        let mut out = Vec::new();
        write_tag(&mut out, b"TEXV0005");
        write_tag(&mut out, b"TEXI0001");
        out.extend_from_slice(&0xFFFFFFFFu32.to_le_bytes()); // format = UNKNOWN (FreeImage takes over)
        out.extend_from_slice(&0u32.to_le_bytes()); // flags
        out.extend_from_slice(&3u32.to_le_bytes()); // texture_width
        out.extend_from_slice(&3u32.to_le_bytes()); // texture_height
        out.extend_from_slice(&3u32.to_le_bytes()); // image_width
        out.extend_from_slice(&3u32.to_le_bytes()); // image_height
        out.extend_from_slice(&0u32.to_le_bytes()); // ignored

        write_tag(&mut out, b"TEXB0003");
        out.extend_from_slice(&1u32.to_le_bytes()); // imageCount = 1
        out.extend_from_slice(&13u32.to_le_bytes()); // FIF_PNG = 13

        out.extend_from_slice(&1u32.to_le_bytes()); // mipmapCount = 1
                                                    // mipmap: width, height, compression=0, uncompressedSize, compressedSize, bytes
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // compression = 0 (raw payload bytes)
        out.extend_from_slice(&(png_bytes.len() as i32).to_le_bytes()); // uncompressedSize (ignored, compression==0)
        out.extend_from_slice(&(png_bytes.len() as i32).to_le_bytes()); // compressedSize
        out.extend_from_slice(&png_bytes);
        out
    }

    #[test]
    fn decodes_free_image_payload() {
        let data = free_image_tex();
        let tex = TexFile::parse(&data).unwrap();
        let img = tex.to_rgba().unwrap();
        assert_eq!(img.width(), 3);
        assert_eq!(img.height(), 3);
        assert_eq!(img.get_pixel(1, 1).0, [200, 100, 50, 255]);
    }

    /// Build a 4x2 R8 spritesheet (two 2x2 frames side by side, left=0x40,
    /// right=0xC0) with a TEXS0002 animation table and the IsGif flag set.
    fn spritesheet_tex() -> Vec<u8> {
        let mut out = Vec::new();
        write_tag(&mut out, b"TEXV0005");
        write_tag(&mut out, b"TEXI0001");
        out.extend_from_slice(&9u32.to_le_bytes()); // format = R8
        out.extend_from_slice(&FLAG_IS_GIF.to_le_bytes()); // flags
        out.extend_from_slice(&4u32.to_le_bytes()); // texture_width
        out.extend_from_slice(&2u32.to_le_bytes()); // texture_height
        out.extend_from_slice(&4u32.to_le_bytes()); // image_width
        out.extend_from_slice(&2u32.to_le_bytes()); // image_height
        out.extend_from_slice(&0u32.to_le_bytes()); // ignored

        write_tag(&mut out, b"TEXB0001");
        out.extend_from_slice(&1u32.to_le_bytes()); // imageCount = 1
        out.extend_from_slice(&1u32.to_le_bytes()); // mipmapCount = 1

        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes());
        let pixels = [0x40, 0x40, 0xC0, 0xC0, 0x40, 0x40, 0xC0, 0xC0];
        out.extend_from_slice(&(pixels.len() as i32).to_le_bytes());
        out.extend_from_slice(&pixels);

        write_tag(&mut out, b"TEXS0002");
        out.extend_from_slice(&2u32.to_le_bytes()); // frameCount = 2
        for (frame_num, x) in [(0u32, 0.0f32), (1, 2.0)] {
            out.extend_from_slice(&frame_num.to_le_bytes());
            out.extend_from_slice(&0.1f32.to_le_bytes()); // frametime
            out.extend_from_slice(&x.to_le_bytes()); // x
            out.extend_from_slice(&0.0f32.to_le_bytes()); // y
            out.extend_from_slice(&2.0f32.to_le_bytes()); // width1
            out.extend_from_slice(&2.0f32.to_le_bytes()); // width2
            out.extend_from_slice(&2.0f32.to_le_bytes()); // height2
            out.extend_from_slice(&2.0f32.to_le_bytes()); // height1
        }
        out
    }

    #[test]
    fn slices_texs_spritesheet_frames() {
        let data = spritesheet_tex();
        let tex = TexFile::parse(&data).unwrap();
        assert!(tex.is_animated());
        let frames = tex.to_rgba_frames().unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].width(), 2);
        assert_eq!(frames[0].height(), 2);
        assert_eq!(frames[0].get_pixel(0, 0).0, [0x40, 0x40, 0x40, 255]);
        assert_eq!(frames[1].get_pixel(0, 0).0, [0xC0, 0xC0, 0xC0, 255]);
        assert!((tex.frames()[0].frametime - 0.1).abs() < 1e-6);
    }
}
