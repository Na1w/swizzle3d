use glam::{vec3, Vec3};
use crate::types::Varyings;
use crate::materials::{Material, sample_material};

pub const WIDTH: usize = 800;
pub const HEIGHT: usize = 600;
pub const CHUNK_ROWS: usize = 16;

pub struct Band<'a> {
    pub color: &'a mut [u32],
    pub depth: &'a mut [f32],
    pub w: usize,
    pub y0: i32,
    pub y1: i32,
    pub tint: u32,
    pub debug: bool,
}

#[inline]
fn band_idx(band: &Band, x: i32, y: i32) -> usize {
    let local_row = (y - band.y0) as usize;
    local_row * band.w + (x as usize)
}

#[inline]
fn lerp_u8(a: u8, b: u8, t_num: u8, t_den: u8) -> u8 {
    let a16 = a as u16;
    let b16 = b as u16;
    let t = t_num as u16;
    let den = t_den as u16;
    let res = (a16 * (den - t) + b16 * t + (den / 2)) / den;
    res as u8
}

#[inline]
fn blend_tint_argb(color: u32, tint: u32, t_num: u8, t_den: u8) -> u32 {
    let a = ((color >> 24) & 0xFF) as u8;
    let cr = ((color >> 16) & 0xFF) as u8;
    let cg = ((color >> 8) & 0xFF) as u8;
    let cb = (color & 0xFF) as u8;

    let tr = ((tint >> 16) & 0xFF) as u8;
    let tg = ((tint >> 8) & 0xFF) as u8;
    let tb = (tint & 0xFF) as u8;

    let r = lerp_u8(cr, tr, t_num, t_den) as u32;
    let g = lerp_u8(cg, tg, t_num, t_den) as u32;
    let b = lerp_u8(cb, tb, t_num, t_den) as u32;
    ((a as u32) << 24) | (r << 16) | (g << 8) | b
}

fn ndc_depth_to_zbuf(ndc_z: f32) -> f32 {
    0.5 * (ndc_z + 1.0)
}

pub fn edge(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

#[inline]
pub fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 { v.max(lo).min(hi) }

fn lambert_shade(base_color: Vec3, normal_ws: Vec3, light_dir_ws: Vec3, view_dir_ws: Vec3) -> Vec3 {
    let n = normal_ws.normalize();
    let l = (-light_dir_ws).normalize();
    let diffuse = n.dot(l).max(0.0);
    let ambient = 0.15;
    let h = (l + view_dir_ws.normalize()).normalize();
    let spec = n.dot(h).max(0.0).powf(32.0) * 0.25;
    base_color * (ambient + 0.85 * diffuse) + vec3(spec, spec, spec)
}

pub fn to_screen(p: Vec3) -> Vec3 {
    let x = ((p.x + 1.0) * 0.5) * (WIDTH as f32);
    let y = (1.0 - (p.y + 1.0) * 0.5) * (HEIGHT as f32);
    vec3(x, y, p.z)
}

#[allow(clippy::too_many_arguments)]
pub fn draw_triangle_band_ext(
    band: &mut Band,
    v0: &Varyings,
    v1: &Varyings,
    v2: &Varyings,
    light_dir: Vec3,
    cam_pos: Vec3,
    time: f32,
    material: Material,
    no_cull: bool,
    no_depth: bool,
) {
    let p0 = v0.screen; let p1 = v1.screen; let p2 = v2.screen;

    let area = edge(p0, p1, p2);
    if !no_cull && area >= 0.0 { return; }

    let min_x = p0.x.min(p1.x).min(p2.x).floor().max(0.0) as i32;
    let max_x = p0.x.max(p1.x).max(p2.x).ceil().min((band.w - 1) as f32) as i32;
    let mut min_y = p0.y.min(p1.y).min(p2.y).floor() as i32;
    let mut max_y = p0.y.max(p1.y).max(p2.y).ceil() as i32;
    min_y = clamp_i32(min_y, band.y0, band.y1 - 1);
    max_y = clamp_i32(max_y, band.y0, band.y1 - 1);
    if min_y > max_y { return; }

    if area == 0.0 { return; }
    let inv_area = 1.0 / area;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = vec3(x as f32 + 0.5, y as f32 + 0.5, 0.0);
            let w0 = edge(p1, p2, p) * inv_area;
            let w1 = edge(p2, p0, p) * inv_area;
            let w2 = edge(p0, p1, p) * inv_area;

            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 { continue; }

            let invw = v0.inv_w * w0 + v1.inv_w * w1 + v2.inv_w * w2;
            let recip = 1.0 / invw;

            let clip_z = v0.clip.z * w0 + v1.clip.z * w1 + v2.clip.z * w2;
            let clip_w = v0.clip.w * w0 + v1.clip.w * w1 + v2.clip.w * w2;
            let ndc_z = clip_z / clip_w;
            let z = ndc_depth_to_zbuf(ndc_z);
            let idx = band_idx(band, x, y);
            if !no_depth && z >= band.depth[idx] { continue; }

            let uv = (v0.uv * v0.inv_w * w0 + v1.uv * v1.inv_w * w1 + v2.uv * v2.inv_w * w2) * recip;
            let normal = (v0.normal * v0.inv_w * w0 + v1.normal * v1.inv_w * w1 + v2.normal * v2.inv_w * w2) * recip;
            let world_pos = (v0.world_pos * v0.inv_w * w0 + v1.world_pos * v1.inv_w * w1 + v2.world_pos * v2.inv_w * w2) * recip;

            let base = sample_material(uv, material, time);
            let view_dir = (cam_pos - world_pos).normalize();
            let rgb = lambert_shade(base, normal, light_dir, view_dir).clamp(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0));
            let r = (rgb.x * 255.0) as u32;
            let g = (rgb.y * 255.0) as u32;
            let b = (rgb.z * 255.0) as u32;
            let mut px = (0xFF << 24) | (r << 16) | (g << 8) | b;
            if band.debug {
                px = blend_tint_argb(px, band.tint, 51, 255);
            }
            band.color[idx] = px;
            if !no_depth {
                band.depth[idx] = z;
            }
        }
    }
}
