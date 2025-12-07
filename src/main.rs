use glam::{vec2, vec3, Mat4, Vec2, Vec3, Vec4};
use rayon::prelude::*;
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use std::time::Instant;
use tokio::sync::{mpsc, watch};

const WIDTH: usize = 800;
const HEIGHT: usize = 600;
// Fine-grained work partitioning: number of rows per chunk (tile height)
const CHUNK_ROWS: usize = 16;

#[derive(Clone, Copy, Debug)]
struct Vertex {
    pos: Vec3,
    normal: Vec3,
    uv: Vec2,
}

#[derive(Clone, Copy, Debug)]
struct Varyings {
    screen: Vec3, // x,y in pixels, z in NDC [-1,1]
    inv_w: f32,   // 1 / clip_w for perspective correction
    world_pos: Vec3,
    normal: Vec3,
    uv: Vec2,
}

fn checkerboard(uv: Vec2) -> Vec3 {
    // Simple, tile 8x8 checker
    let scale = 8.0;
    let u = (uv.x * scale).floor() as i32;
    let v = (uv.y * scale).floor() as i32;
    let c = ((u ^ v) & 1) as f32;
    // Two colors
    let a = vec3(0.12, 0.12, 0.12);
    let b = vec3(0.85, 0.85, 0.85);
    a * (1.0 - c) + b * c
}

// --- Procedural materials ---
#[derive(Clone, Copy)]
enum Material { Checker, Starfield, Fire }

// Note: Materials are bound per triangle (see tri_materials in worker loop).

#[inline]
fn hash2(x: Vec2) -> f32 {
    // Low-cost hash based on dot + sin, returns [0,1)
    let p = x;
    let h = (p.x * 127.1 + p.y * 311.7).sin() * 43758.5453;
    h.fract().abs()
}

#[inline]
fn value_noise(p: Vec2) -> f32 {
    // 2D value noise using bilinear interpolation of hashed grid points
    let i = vec2(p.x.floor(), p.y.floor());
    let f = p - i;
    let u = vec2(f.x * f.x * (3.0 - 2.0 * f.x), f.y * f.y * (3.0 - 2.0 * f.y));
    let n00 = hash2(i);
    let n10 = hash2(i + vec2(1.0, 0.0));
    let n01 = hash2(i + vec2(0.0, 1.0));
    let n11 = hash2(i + vec2(1.0, 1.0));
    let nx0 = n00 + (n10 - n00) * u.x;
    let nx1 = n01 + (n11 - n01) * u.x;
    nx0 + (nx1 - nx0) * u.y
}

#[inline]
fn fbm(mut p: Vec2, octaves: i32) -> f32 {
    let mut a = 0.0;
    let mut amp = 0.5;
    for _ in 0..octaves {
        a += value_noise(p) * amp;
        p = p * 2.0 + vec2(3.1, 1.7);
        amp *= 0.5;
    }
    a
}

#[inline]
fn sample_starfield(uv: Vec2, time: f32) -> Vec3 {
    // Tile space for stars
    let scale = 40.0; // higher -> more potential stars
    let p = uv * scale;
    let cell = vec2(p.x.floor(), p.y.floor());
    let r = hash2(cell);
    let density = 0.975; // keep sparse
    if r > density {
        // brightness based on hash and a twinkle factor
        let phase = hash2(cell + vec2(7.3, 9.2)) * 6.28318;
        let twinkle = 0.6 + 0.4 * (time * 6.0 + phase).sin().abs();
        let b = ((r - density) / (1.0 - density)).powf(4.0) * twinkle; // concentrate near 1
        let color = vec3(0.8, 0.9, 1.0) * b; // slightly bluish white
        return color.clamp(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0));
    }
    // faint background
    let bg = 0.02 + 0.03 * fbm(uv * 3.0 + vec2(0.0, time * 0.05), 3);
    vec3(bg, bg, bg)
}

#[inline]
fn fire_ramp(t: f32) -> Vec3 {
    // t in [0,1]
    let t = t.clamp(0.0, 1.0);
    // piecewise: black->red->orange->yellow->white
    if t < 0.3 {
        // black to red
        let k = t / 0.3;
        vec3(0.0, 0.0, 0.0) * (1.0 - k) + vec3(1.0, 0.0, 0.0) * k
    } else if t < 0.6 {
        // red to orange
        let k = (t - 0.3) / 0.3;
        vec3(1.0, 0.0, 0.0) * (1.0 - k) + vec3(1.0, 0.5, 0.0) * k
    } else if t < 0.85 {
        // orange to yellow
        let k = (t - 0.6) / 0.25;
        vec3(1.0, 0.5, 0.0) * (1.0 - k) + vec3(1.0, 1.0, 0.0) * k
    } else {
        // yellow to white
        let k = (t - 0.85) / 0.15;
        vec3(1.0, 1.0, 0.0) * (1.0 - k) + vec3(1.0, 1.0, 1.0) * k
    }
}

#[inline]
fn sample_fire(uv: Vec2, time: f32) -> Vec3 {
    // Flip Y so fire rises upward visually in [0,1] UV
    let p = vec2(uv.x, 1.0 - uv.y);
    // Horizontal turbulence and vertical advection
    let speed = 0.6;
    let scroll = time * speed;
    let turbulence = fbm(vec2(p.x * 3.0, p.y * 6.0 + scroll * 4.0), 4);
    // Base intensity increases near bottom and decreases toward top
    let base = (1.2 * (1.0 - p.y) + 0.8 * turbulence).clamp(0.0, 1.0);
    // Add flicker
    let flicker = 0.85 + 0.15 * (time * 10.0 + hash2(vec2((p.x * 5.0).floor(), (p.y * 8.0).floor())) * 6.28318).sin().abs();
    let t = (base.powf(1.2) * flicker).clamp(0.0, 1.0);
    fire_ramp(t)
}

// Default per-face materials; order +X,-X,+Y,-Y,+Z,-Z
const FACE_MATERIALS: [Material; 6] = [
    Material::Fire,      // +X
    Material::Checker,   // -X
    Material::Checker,   // +Y
    Material::Checker,   // -Y
    Material::Starfield, // +Z
    Material::Checker,   // -Z
];

#[inline]
fn sample_material(uv: Vec2, material: Material, time: f32) -> Vec3 {
    match material {
        Material::Checker => checkerboard(uv),
        Material::Starfield => sample_starfield(uv * 1.0, time),
        Material::Fire => sample_fire(uv * 1.0, time),
    }
}

fn lambert_shade(base_color: Vec3, normal_ws: Vec3, light_dir_ws: Vec3, view_dir_ws: Vec3) -> Vec3 {
    let n = normal_ws.normalize();
    let l = (-light_dir_ws).normalize(); // light_dir as direction from surface to light
    let diffuse = n.dot(l).max(0.0);
    let ambient = 0.15;
    // Simple Blinn-Phong
    let h = (l + view_dir_ws.normalize()).normalize();
    let spec = n.dot(h).max(0.0).powf(32.0) * 0.25;
    base_color * (ambient + 0.85 * diffuse) + vec3(spec, spec, spec)
}

fn to_screen(p: Vec3) -> Vec3 {
    // p in NDC [-1,1]
    let x = ((p.x + 1.0) * 0.5) * (WIDTH as f32);
    let y = ((1.0 - (p.y + 1.0) * 0.5)) * (HEIGHT as f32);
    vec3(x, y, p.z)
}

fn make_cube(size: f32) -> (Vec<Vertex>, Vec<[u32; 3]>) {
    let s = size * 0.5;
    // 24 unique vertices (per-face normals/uvs)
    let p = [
        vec3(-s, -s,  s), vec3( s, -s,  s), vec3( s,  s,  s), vec3(-s,  s,  s), // front +Z
        vec3(-s, -s, -s), vec3(-s,  s, -s), vec3( s,  s, -s), vec3( s, -s, -s), // back -Z
        vec3(-s,  s, -s), vec3(-s,  s,  s), vec3( s,  s,  s), vec3( s,  s, -s), // top +Y
        vec3(-s, -s, -s), vec3( s, -s, -s), vec3( s, -s,  s), vec3(-s, -s,  s), // bottom -Y
        vec3( s, -s, -s), vec3( s,  s, -s), vec3( s,  s,  s), vec3( s, -s,  s), // right +X
        vec3(-s, -s, -s), vec3(-s, -s,  s), vec3(-s,  s,  s), vec3(-s,  s, -s), // left -X
    ];
    let n: [Vec3; 24] = [
        // front +Z
        vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0),
        // back -Z
        vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, -1.0),
        // top +Y
        vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0),
        // bottom -Y
        vec3(0.0, -1.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0, -1.0, 0.0),
        // right +X
        vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0),
        // left -X
        vec3(-1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0),
    ];
    let uv = [
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0),
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0),
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0),
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0),
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0),
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0),
    ];

    let mut verts = Vec::with_capacity(24);
    for i in 0..24 {
        verts.push(Vertex { pos: p[i], normal: n[i], uv: uv[i] });
    }
    let idx: Vec<[u32; 3]> = vec![
        // front
        [0, 1, 2], [0, 2, 3],
        // back
        [4, 5, 6], [4, 6, 7],
        // top
        [8, 9, 10], [8, 10, 11],
        // bottom
        [12, 13, 14], [12, 14, 15],
        // right
        [16, 17, 18], [16, 18, 19],
        // left
        [20, 21, 22], [20, 22, 23],
    ];
    (verts, idx)
}

// Generate a UV sphere with given rings (latitude) and segments (longitude)
// Rings >= 2, Segments >= 3
fn make_uv_sphere(rings: u32, segments: u32, radius: f32) -> (Vec<Vertex>, Vec<[u32; 3]>) {
    let rings = rings.max(2);
    let segments = segments.max(3);
    let mut verts: Vec<Vertex> = Vec::with_capacity((rings as usize + 1) * (segments as usize + 1));
    for r in 0..=rings {
        let v = r as f32 / rings as f32; // [0,1]
        let theta = v * std::f32::consts::PI; // 0..PI
        let (st, ct) = theta.sin_cos();
        for s in 0..=segments {
            let u = s as f32 / segments as f32; // [0,1]
            let phi = u * std::f32::consts::TAU; // 0..2PI
            let (sp, cp) = phi.sin_cos();
            let n = vec3(cp * st, ct, sp * st);
            let pos = n * radius;
            verts.push(Vertex { pos, normal: n, uv: vec2(u, 1.0 - v) });
        }
    }
    let stride = (segments + 1) as usize;
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity((rings as usize) * (segments as usize) * 2);
    for r in 0..rings {
        for s in 0..segments {
            let i0 = (r as usize) * stride + (s as usize);
            let i1 = i0 + 1;
            let i2 = i0 + stride;
            let i3 = i2 + 1;
            // two triangles per quad; ensure winding matches outward-facing convention
            // Flip from previous (which produced inward normals) to outward normals
            tris.push([i0 as u32, i1 as u32, i2 as u32]);
            tris.push([i1 as u32, i3 as u32, i2 as u32]);
        }
    }
    (verts, tris)
}

fn ndc_depth_to_zbuf(ndc_z: f32) -> f32 {
    // Map [-1,1] where -1 is near to a positive value; we'll use direct compare on depth with smaller closer.
    // We initialized depth with +inf so we compare real z later.
    ndc_z
}

fn edge(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    // 2D edge function using top-left rule
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

#[inline]
fn is_front_facing(a_wp: Vec3, b_wp: Vec3, c_wp: Vec3, cam_pos: Vec3) -> bool {
    // World-space facing: front if triangle normal points toward camera
    let n = (b_wp - a_wp).cross(c_wp - a_wp);
    let to_cam = cam_pos - (a_wp + b_wp + c_wp) * (1.0 / 3.0);
    n.dot(to_cam) > 0.0
}

#[inline]
fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 { v.max(lo).min(hi) }

// Removed legacy single-threaded raster path to avoid dead code warnings.

// A band is a disjoint vertical range of rows [y0, y1) with exclusive access to its color/depth slices.
struct Band<'a> {
    color: &'a mut [u32],
    depth: &'a mut [f32],
    w: usize,
    y0: i32,
    y1: i32,
    // debug visualization
    tint: u32,   // ARGB tint assigned to this band
    debug: bool, // whether to apply visualization
}

#[inline]
fn band_idx(band: &Band, x: i32, y: i32) -> usize {
    let local_row = (y - band.y0) as usize;
    local_row * band.w + (x as usize)
}

#[inline]
fn lerp_u8(a: u8, b: u8, t_num: u8, t_den: u8) -> u8 {
    // integer linear interpolation: a*(1-t)+b*t with t=t_num/t_den
    let a16 = a as u16;
    let b16 = b as u16;
    let t = t_num as u16;
    let den = t_den as u16;
    let res = (a16 * (den - t) + b16 * t + (den / 2)) / den;
    res as u8
}

#[inline]
fn blend_tint_argb(color: u32, tint: u32, t_num: u8, t_den: u8) -> u32 {
    // ARGB channels
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

fn draw_triangle_band(band: &mut Band, v0: &Varyings, v1: &Varyings, v2: &Varyings, light_dir: Vec3, cam_pos: Vec3, time: f32, material: Material) {
    let p0 = v0.screen; let p1 = v1.screen; let p2 = v2.screen;

    // Backface culling in world space (robust w.r.t. projection/axis flips)
    if !is_front_facing(v0.world_pos, v1.world_pos, v2.world_pos, cam_pos) { return; }

    // Bounding box clamped to band rows
    let min_x = p0.x.min(p1.x).min(p2.x).floor().max(0.0) as i32;
    let max_x = p0.x.max(p1.x).max(p2.x).ceil().min((band.w - 1) as f32) as i32;
    let mut min_y = p0.y.min(p1.y).min(p2.y).floor() as i32;
    let mut max_y = p0.y.max(p1.y).max(p2.y).ceil() as i32;
    min_y = clamp_i32(min_y, band.y0, band.y1 - 1);
    max_y = clamp_i32(max_y, band.y0, band.y1 - 1);
    if min_y > max_y { return; }

    let area = edge(p0, p1, p2);
    if area == 0.0 { return; }
    let inv_area = 1.0 / area;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = vec3(x as f32 + 0.5, y as f32 + 0.5, 0.0);
            let mut w0 = edge(p1, p2, p) * inv_area;
            let mut w1 = edge(p2, p0, p) * inv_area;
            let mut w2 = edge(p0, p1, p) * inv_area;
            // Consistent inside-test regardless of winding
            if area > 0.0 {
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 { continue; }
            } else {
                if w0 > 0.0 || w1 > 0.0 || w2 > 0.0 { continue; }
                // make weights positive for later math
                w0 = -w0; w1 = -w1; w2 = -w2;
            }

            // Perspective-correct weights
            let invw = v0.inv_w * w0 + v1.inv_w * w1 + v2.inv_w * w2;
            let recip = 1.0 / invw;

            // Correct perspective depth: interpolate clip-space z and w, then divide
            // Using ndc_z directly in the 1/w scheme would introduce a 1/w^2 error.
            let clip_z = (v0.screen.z / v0.inv_w) * w0 + (v1.screen.z / v1.inv_w) * w1 + (v2.screen.z / v2.inv_w) * w2;
            let clip_w = (1.0 / v0.inv_w) * w0 + (1.0 / v1.inv_w) * w1 + (1.0 / v2.inv_w) * w2;
            let ndc_z = clip_z / clip_w;
            let z = ndc_depth_to_zbuf(ndc_z);
            let idx = band_idx(band, x, y);
            if z >= band.depth[idx] { continue; }

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
                // blend 20% of band tint
                px = blend_tint_argb(px, band.tint, 51, 255); // ~0.2
            }
            band.color[idx] = px;
            band.depth[idx] = z;
        }
    }
}

#[tokio::main]
async fn main() {
    let mut window = Window::new(
        "Software 3D Renderer (Rust)",
        WIDTH,
        HEIGHT,
        WindowOptions { resize: true, ..WindowOptions::default() },
    )
    .expect("Unable to create window");

    window.limit_update_rate(Some(std::time::Duration::from_micros(16_667))); // ~60 FPS

    let (verts, tris) = make_cube(1.6);
    let (sphere_verts, sphere_tris) = make_uv_sphere(32, 48, 0.8);

    // Camera
    let up = vec3(0.0, 1.0, 0.0);

    // Channels: free buffers -> worker, completed frames -> main
    let (to_worker_tx, mut to_worker_rx) = mpsc::channel::<Vec<u32>>(2);
    let (from_worker_tx, mut from_worker_rx) = mpsc::channel::<Vec<u32>>(2);
    // Scene params (watch latest parameters)
    #[derive(Clone, Copy)]
    struct SceneParams {
        angle: f32,
        debug: bool,
        time: f32,
        cam_pos: Vec3,
        cam_target: Vec3,
        light_dir: Vec3,
    }
    let (scene_tx, scene_rx) = watch::channel(SceneParams {
        angle: 0.0,
        debug: false,
        time: 0.0,
        cam_pos: vec3(0.0, 0.0, 3.0),
        cam_target: vec3(0.0, 0.0, 0.0),
        light_dir: vec3(0.6, 0.7, 0.2).normalize(),
    });

    // Preallocate double buffers and send to worker
    let buf_size = WIDTH * HEIGHT;
    to_worker_tx.send(vec![0xFF101018; buf_size]).await.ok();
    to_worker_tx.send(vec![0xFF101018; buf_size]).await.ok();

    // Spawn render worker (async task; heavy compute uses Rayon internally)
    let verts_w = verts.clone();
    let tris_w = tris.clone();
    let sphere_verts_w = sphere_verts.clone();
    let sphere_tris_w = sphere_tris.clone();
    tokio::spawn(async move {
        // Depth buffer reused inside worker
        let mut depth = vec![f32::INFINITY; WIDTH * HEIGHT];
        let up = up;
        let scene_rx = scene_rx;

        // Bind a material per triangle (fixed mapping to cube faces):
        // Triangles are ordered in make_cube as:
        // 0-1: front (+Z), 2-3: back (-Z), 4-5: top (+Y), 6-7: bottom (-Y), 8-9: right (+X), 10-11: left (-X)
        let mut tri_materials: Vec<Material> = Vec::with_capacity(tris_w.len());
        for (i, _) in tris_w.iter().enumerate() {
            let mat = if i <= 1 {
                // front +Z
                FACE_MATERIALS[4]
            } else if i <= 3 {
                // back -Z
                FACE_MATERIALS[5]
            } else if i <= 5 {
                // top +Y
                FACE_MATERIALS[2]
            } else if i <= 7 {
                // bottom -Y
                FACE_MATERIALS[3]
            } else if i <= 9 {
                // right +X
                FACE_MATERIALS[0]
            } else {
                // left -X (10-11)
                FACE_MATERIALS[1]
            };
            tri_materials.push(mat);
        }

        while let Some(mut color) = to_worker_rx.recv().await {
            // Read latest scene params
            let params = *scene_rx.borrow();

            // Clear buffers
            color.fill(0xFF101018);
            depth.fill(f32::INFINITY);

            // Compute transforms for this frame
            let angle = params.angle;
            let model = Mat4::from_rotation_y(angle) * Mat4::from_rotation_x(angle * 0.5);
            let view = Mat4::look_at_rh(params.cam_pos, params.cam_target, up);
            let aspect = WIDTH as f32 / HEIGHT as f32;
            let proj = Mat4::perspective_rh_gl(60f32.to_radians(), aspect, 0.1, 100.0);
            let mvp = proj * view * model;

            // Build varyings for cube
            let mut var: Vec<Varyings> = Vec::with_capacity(verts_w.len());
            for v in &verts_w {
                let world_pos = (model * v.pos.extend(1.0)).truncate();
                let clip: Vec4 = mvp * v.pos.extend(1.0);
                let inv_w = 1.0 / clip.w;
                let ndc = clip.truncate() * inv_w;
                let screen = to_screen(ndc);
                let normal_ws = (model * v.normal.extend(0.0)).truncate();
                var.push(Varyings { screen, inv_w, world_pos, normal: normal_ws, uv: v.uv });
            }

            // Partition into fine-grained row chunks and bin triangles per chunk
            let rows_per_band = CHUNK_ROWS;
            let num_bands = (HEIGHT + rows_per_band - 1) / rows_per_band;

            // Triangle bins per band (by y-range)
            let mut bins: Vec<Vec<usize>> = vec![Vec::new(); num_bands];
            for (ti, t) in tris_w.iter().enumerate() {
                // Backface cull early in world space to match raster rule
                let aw = var[t[0] as usize].world_pos;
                let bw = var[t[1] as usize].world_pos;
                let cw = var[t[2] as usize].world_pos;
                if !is_front_facing(aw, bw, cw, params.cam_pos) { continue; }
                let a = var[t[0] as usize].screen;
                let b = var[t[1] as usize].screen;
                let c = var[t[2] as usize].screen;
                let mut min_y = a.y.min(b.y).min(c.y).floor() as i32;
                let mut max_y = a.y.max(b.y).max(c.y).ceil() as i32;
                min_y = clamp_i32(min_y, 0, (HEIGHT - 1) as i32);
                max_y = clamp_i32(max_y, 0, (HEIGHT - 1) as i32);
                if min_y > max_y { continue; }
                let first_band = (min_y as usize) / rows_per_band;
                let last_band = (max_y as usize) / rows_per_band;
                for bi in first_band..=last_band {
                    bins[bi].push(ti);
                }
            }

            // Safety: split disjointly by rows
            let mut bands: Vec<Band> = Vec::new();
            let mut y0 = 0usize;
            // Distinct tint colors for debug visualization (ARGB)
            let tints: [u32; 12] = [
                0xFFFF0000, // red
                0xFF00FF00, // green
                0xFF0000FF, // blue
                0xFFFFFF00, // yellow
                0xFFFF00FF, // magenta
                0xFF00FFFF, // cyan
                0xFFFF8000, // orange
                0xFF8080FF, // light blue
                0xFF80FF80, // light green
                0xFFFF80FF, // pink
                0xFF80FFFF, // light cyan
                0xFFFFFF80, // light yellow
            ];
            let debug = params.debug;
            let mut band_id = 0usize;
            for (c_chunk, d_chunk) in color
                .chunks_mut(rows_per_band * WIDTH)
                .zip(depth.chunks_mut(rows_per_band * WIDTH))
            {
                let rows = c_chunk.len() / WIDTH;
                if rows == 0 { continue; }
                let y_start = y0 as i32;
                let y_end = (y0 + rows) as i32;
                let tint = tints[band_id % tints.len()];
                bands.push(Band { color: c_chunk, depth: d_chunk, w: WIDTH, y0: y_start, y1: y_end, tint, debug });
                y0 += rows;
                band_id += 1;
            }

            // Pair each band with its bin and process in parallel
            struct BandTask<'a> {
                band: Band<'a>,
                bin: Vec<usize>,
            }
            let mut tasks: Vec<BandTask> = Vec::with_capacity(bands.len());
            for (i, band) in bands.into_iter().enumerate() {
                // move bin out without cloning extra capacity
                use std::mem;
                let bin = mem::take(&mut bins[i]);
                tasks.push(BandTask { band, bin });
            }

            let time = params.time;
            let light_dir = params.light_dir.normalize();
            tasks.into_par_iter().for_each(|mut t|
            {
                for &ti in &t.bin {
                    let idxs = tris_w[ti];
                    let a = var[idxs[0] as usize];
                    let b = var[idxs[1] as usize];
                    let c = var[idxs[2] as usize];
                    let material = tri_materials[ti];
                    draw_triangle_band(&mut t.band, &a, &b, &c, light_dir, params.cam_pos, time, material);
                }
            });

            // Now render the sphere: build transforms (place sphere offset) and draw
            let sphere_model = Mat4::from_translation(vec3(-1.8, 0.0, 0.0));
            let sphere_mvp = proj * view * sphere_model;
            let mut svar: Vec<Varyings> = Vec::with_capacity(sphere_verts_w.len());
            for v in &sphere_verts_w {
                let world_pos = (sphere_model * v.pos.extend(1.0)).truncate();
                let clip: Vec4 = sphere_mvp * v.pos.extend(1.0);
                let inv_w = 1.0 / clip.w;
                let ndc = clip.truncate() * inv_w;
                let screen = to_screen(ndc);
                let normal_ws = (sphere_model * v.normal.extend(0.0)).truncate();
                svar.push(Varyings { screen, inv_w, world_pos, normal: normal_ws, uv: v.uv });
            }

            // Bin sphere triangles
            let rows_per_band = CHUNK_ROWS;
            let num_bands = (HEIGHT + rows_per_band - 1) / rows_per_band;
            let mut bins: Vec<Vec<usize>> = vec![Vec::new(); num_bands];
            for (ti, t) in sphere_tris_w.iter().enumerate() {
                let aw = svar[t[0] as usize].world_pos;
                let bw = svar[t[1] as usize].world_pos;
                let cw = svar[t[2] as usize].world_pos;
                if !is_front_facing(aw, bw, cw, params.cam_pos) { continue; }
                let a = svar[t[0] as usize].screen;
                let b = svar[t[1] as usize].screen;
                let c = svar[t[2] as usize].screen;
                let mut min_y = a.y.min(b.y).min(c.y).floor() as i32;
                let mut max_y = a.y.max(b.y).max(c.y).ceil() as i32;
                min_y = clamp_i32(min_y, 0, (HEIGHT - 1) as i32);
                max_y = clamp_i32(max_y, 0, (HEIGHT - 1) as i32);
                if min_y > max_y { continue; }
                let first_band = (min_y as usize) / rows_per_band;
                let last_band = (max_y as usize) / rows_per_band;
                for bi in first_band..=last_band { bins[bi].push(ti); }
            }

            // Rebuild bands over current color/depth slices
            let mut bands: Vec<Band> = Vec::new();
            let mut y0 = 0usize;
            let tints: [u32; 12] = [
                0xFFFF0000,0xFF00FF00,0xFF0000FF,0xFFFFFF00,0xFFFF00FF,0xFF00FFFF,
                0xFFFF8000,0xFF8080FF,0xFF80FF80,0xFFFF80FF,0xFF80FFFF,0xFFFFFF80,
            ];
            let debug = params.debug;
            let mut band_id = 0usize;
            for (c_chunk, d_chunk) in color.chunks_mut(rows_per_band * WIDTH).zip(depth.chunks_mut(rows_per_band * WIDTH)) {
                let rows = c_chunk.len() / WIDTH; if rows == 0 { continue; }
                let y_start = y0 as i32; let y_end = (y0 + rows) as i32;
                let tint = tints[band_id % tints.len()];
                bands.push(Band { color: c_chunk, depth: d_chunk, w: WIDTH, y0: y_start, y1: y_end, tint, debug });
                y0 += rows; band_id += 1;
            }

            struct BandTask2<'a> { band: Band<'a>, bin: Vec<usize> }
            let mut tasks2: Vec<BandTask2> = Vec::with_capacity(bands.len());
            for (i, band) in bands.into_iter().enumerate() {
                use std::mem; let bin = mem::take(&mut bins[i]);
                tasks2.push(BandTask2 { band, bin });
            }

            tasks2.into_par_iter().for_each(|mut t| {
                for &ti in &t.bin {
                    let idxs = sphere_tris_w[ti];
                    let a = svar[idxs[0] as usize];
                    let b = svar[idxs[1] as usize];
                    let c = svar[idxs[2] as usize];
                    // Sphere material: use checker to visualize UVs
                    draw_triangle_band(&mut t.band, &a, &b, &c, light_dir, params.cam_pos, time, Material::Checker);
                }
            });

            // Send completed frame back
            if from_worker_tx.send(color).await.is_err() {
                break;
            }
        }
    });

    // Main loop: update params, present completed frames, recycle buffers
    let mut angle: f32 = 0.0;
    let start_time = Instant::now();
    let mut debug: bool = false;
    let mut last_present = Instant::now();
    let mut fps_ema: f32 = 0.0;
    // Camera state
    let mut cam_pos = vec3(0.0, 0.0, 3.0);
    let mut yaw: f32 = 0.0;   // radians around Y
    let mut pitch: f32 = 0.0; // radians up/down
    let mut last_frame = Instant::now();
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now_loop = Instant::now();
        let dt = now_loop.duration_since(last_frame).as_secs_f32().min(0.05);
        last_frame = now_loop;
        // Toggle debug mode on key press (no repeat)
        if window.is_key_pressed(Key::D, KeyRepeat::No) {
            debug = !debug;
        }

        // Object rotation (cube)
        angle += 0.8 * dt;
        // Camera controls
        let turn_speed = 1.8; // rad/s
        if window.is_key_down(Key::Left) { yaw -= turn_speed * dt; }
        if window.is_key_down(Key::Right) { yaw += turn_speed * dt; }
        if window.is_key_down(Key::Up) { pitch = (pitch + turn_speed * dt).min(1.2); }
        if window.is_key_down(Key::Down) { pitch = (pitch - turn_speed * dt).max(-1.2); }

        let forward = vec3(yaw.sin()*pitch.cos(), pitch.sin(), yaw.cos()*pitch.cos());
        let right = vec3(forward.z, 0.0, -forward.x).normalize();
        let up_vec = vec3(0.0, 1.0, 0.0);
        let move_speed = if window.is_key_down(Key::LeftShift) { 4.0 } else { 2.0 };
        if window.is_key_down(Key::W) { cam_pos += forward * move_speed * dt; }
        if window.is_key_down(Key::S) { cam_pos -= forward * move_speed * dt; }
        if window.is_key_down(Key::A) { cam_pos -= right * move_speed * dt; }
        if window.is_key_down(Key::D) { cam_pos += right * move_speed * dt; }
        if window.is_key_down(Key::Q) { cam_pos -= up_vec * move_speed * dt; }
        if window.is_key_down(Key::E) { cam_pos += up_vec * move_speed * dt; }

        let time = start_time.elapsed().as_secs_f32();
        // Rotating light around Y axis
        let light_radius = 1.0;
        let light_dir = vec3((time * 0.7).cos() * light_radius, 0.6, (time * 0.7).sin() * light_radius).normalize();
        let cam_target = cam_pos + forward;
        let _ = scene_tx.send(SceneParams { angle, debug, time, cam_pos, cam_target, light_dir });

        if let Some(color) = from_worker_rx.try_recv().ok() {
            // Present
            window
                .update_with_buffer(&color, WIDTH, HEIGHT)
                .expect("Failed to update window");
            // FPS update
            let now = Instant::now();
            let dt = now.duration_since(last_present).as_secs_f32().max(1e-6);
            last_present = now;
            let inst_fps = 1.0 / dt;
            fps_ema = if fps_ema == 0.0 { inst_fps } else { fps_ema * 0.9 + inst_fps * 0.1 };
            window.set_title(&format!(
                "Software 3D Renderer (Rust) | {:5.1} FPS | Debug {} | WASD/QE move, Arrow keys look",
                fps_ema,
                if debug { "ON" } else { "OFF" }
            ));
            // Recycle buffer
            let _ = to_worker_tx.try_send(color);
        } else {
            // No new frame ready yet; keep window responsive
            window.update();
        }
    }
}
