use crate::{
    data_io::io_error,
    error::{Error, Result},
};
use std::{io::Write, path::Path};
pub const PALETTE: [[u8; 3]; 12] = [
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [0, 255, 255],
    [255, 0, 255],
    [255, 255, 0],
    [0, 255, 128],
    [0, 128, 255],
    [255, 128, 0],
    [255, 0, 128],
    [128, 128, 255],
    [128, 255, 128],
];
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pixels: Vec<u8>,
}

impl Canvas {
    pub fn new(width: usize, height: usize, color: [u8; 3]) -> Result<Self> {
        let len = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .filter(|n| *n <= 600_000_000)
            .ok_or_else(|| Error::value("image dimensions exceed native rendering limit"))?;
        let mut pixels = vec![0; len];
        for p in pixels.chunks_exact_mut(3) {
            p.copy_from_slice(&color);
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }
    fn pixel(&mut self, x: i32, y: i32, color: [u8; 3]) {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            let i = (y as usize * self.width + x as usize) * 3;
            self.pixels[i..i + 3].copy_from_slice(&color);
        }
    }
    pub fn rect(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: [u8; 3]) {
        for y in y0.max(0)..y1.min(self.height as i32) {
            for x in x0.max(0)..x1.min(self.width as i32) {
                self.pixel(x, y, color);
            }
        }
    }
    pub fn circle(&mut self, x: f64, y: f64, r: i32, color: [u8; 3], filled: bool) {
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        let (x, y) = (x.round_ties_even() as i32, y.round_ties_even() as i32);
        if x < -r || y < -r || x > self.width as i32 + r || y > self.height as i32 + r {
            return;
        }
        for b in -r..=r {
            for a in -r..=r {
                let d = a * a + b * b;
                if d <= r * r && (filled || d >= (r - 1) * (r - 1)) {
                    self.pixel(x + a, y + b, color);
                }
            }
        }
    }
    pub fn line(
        &mut self,
        mut x0: f64,
        mut y0: f64,
        mut x1: f64,
        mut y1: f64,
        color: [u8; 3],
        width: i32,
        dashed: bool,
    ) {
        if ![x0, y0, x1, y1].iter().all(|x| x.is_finite()) {
            return;
        }
        let (dx, dy) = (x1 - x0, y1 - y0);
        let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
        for (p, q) in [
            (-dx, x0),
            (dx, self.width as f64 - 1. - x0),
            (-dy, y0),
            (dy, self.height as f64 - 1. - y0),
        ] {
            if p == 0. {
                if q < 0. {
                    return;
                }
            } else {
                let r = q / p;
                if p < 0. {
                    lo = lo.max(r);
                } else {
                    hi = hi.min(r);
                }
                if lo > hi {
                    return;
                }
            }
        }
        x1 = x0 + hi * dx;
        y1 = y0 + hi * dy;
        x0 += lo * dx;
        y0 += lo * dy;
        let steps = (x1 - x0).abs().max((y1 - y0).abs()).ceil() as usize;
        for i in 0..=steps {
            if dashed && i % 12 >= 6 {
                continue;
            }
            let a = if steps == 0 {
                0.
            } else {
                i as f64 / steps as f64
            };
            let x = (x0 + a * (x1 - x0)).round_ties_even() as i32;
            let y = (y0 + a * (y1 - y0)).round_ties_even() as i32;
            for oy in -(width / 2)..=(width / 2) {
                for ox in -(width / 2)..=(width / 2) {
                    self.pixel(x + ox, y + oy, color);
                }
            }
        }
    }
    pub fn text(&mut self, x: i32, y: i32, text: &str, color: [u8; 3], scale: i32) {
        const LETTERS: [u64; 36] = [
            0x0e11111f111111,
            0x1e11111e11111e,
            0x0e11101010110e,
            0x1e11111111111e,
            0x1f10101e10101f,
            0x1f10101e101010,
            0x0e11101711110f,
            0x1111111f111111,
            0x0e04040404040e,
            0x0702020212120c,
            0x11121418141211,
            0x1010101010101f,
            0x111b1515111111,
            0x11191513111111,
            0x0e11111111110e,
            0x1e11111e101010,
            0x0e11111115120d,
            0x1e11111e141211,
            0x0f10100e01011e,
            0x1f040404040404,
            0x1111111111110e,
            0x11111111110a04,
            0x11111115151b11,
            0x11110a040a1111,
            0x11110a04040404,
            0x1f01020408101f,
            0x0e11131519110e,
            0x040c040404040e,
            0x0e11010204081f,
            0x1e01010601011e,
            0x02060a121f0202,
            0x1f10101e01011e,
            0x0e10101e11110e,
            0x1f010204080808,
            0x0e11110e11110e,
            0x0e11110f01010e,
        ];
        let mut cursor = x;
        for c in text.to_ascii_uppercase().chars() {
            let glyph = match c {
                'A'..='Z' => LETTERS[(c as u8 - b'A') as usize],
                '0'..='9' => LETTERS[26 + (c as u8 - b'0') as usize],
                '-' => 0x0000001f000000,
                '.' => 0x00000000000c0c,
                ':' => 0x000c0c000c0c00,
                '/' => 0x01010204081010,
                '=' => 0x00001f001f0000,
                ' ' => 0,
                _ => 0x0e110204000400,
            };
            for row in 0..7 {
                for col in 0..5 {
                    if (glyph >> (8 * (6 - row))) & (1 << (4 - col)) != 0 {
                        for sy in 0..scale {
                            for sx in 0..scale {
                                self.pixel(cursor + col * scale + sx, y + row * scale + sy, color);
                            }
                        }
                    }
                }
            }
            cursor += 6 * scale;
        }
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(io_error)?;
        }
        let mut raw = Vec::with_capacity(self.pixels.len() + self.height);
        for row in self.pixels.chunks_exact(self.width * 3) {
            raw.push(0);
            raw.extend_from_slice(row);
        }
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        z.write_all(&raw).map_err(io_error)?;
        let data = z.finish().map_err(io_error)?;
        let mut out = Vec::from(&b"\x89PNG\r\n\x1a\n"[..]);
        fn chunk(out: &mut Vec<u8>, name: &[u8; 4], data: &[u8]) {
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            out.extend_from_slice(name);
            out.extend_from_slice(data);
            let mut crc = 0xffffffff_u32;
            for b in name.iter().chain(data) {
                crc ^= *b as u32;
                for _ in 0..8 {
                    crc = (crc >> 1) ^ if crc & 1 != 0 { 0xedb88320 } else { 0 };
                }
            }
            out.extend_from_slice(&(!crc).to_be_bytes());
        }
        let mut header = vec![];
        header.extend_from_slice(&(self.width as u32).to_be_bytes());
        header.extend_from_slice(&(self.height as u32).to_be_bytes());
        header.extend_from_slice(&[8, 2, 0, 0, 0]);
        chunk(&mut out, b"IHDR", &header);
        chunk(&mut out, b"IDAT", &data);
        chunk(&mut out, b"IEND", &[]);
        std::fs::write(path, out).map_err(io_error)
    }
}
