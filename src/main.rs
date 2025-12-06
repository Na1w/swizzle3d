use glam::{vec2, vec3, Mat4, Vec2, Vec3, Vec4};
use rayon::prelude::*;
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use std::time::Instant;
use tokio::sync::{mpsc, watch};

const WIDTH: usize = 800;
const HEIGHT: usize = 600;

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

struct Framebuffer {
    color: Vec<u32>,
    depth: Vec<f32>, // z-buffer in NDC [-1,1], lower is nearer after mapping
    w: usize,
    h: usize,
}

impl Framebuffer {
    fn new(w: usize, h: usize) -> Self {
        Self { color: vec![0x000000; w * h], depth: vec![f32::INFINITY; w * h], w, h }
    }
    fn clear(&mut self, rgba: u32) {
        self.color.fill(rgba);
        self.depth.fill(f32::INFINITY);
    }
    #[inline]
    fn idx(&self, x: i32, y: i32) -> usize { (y as usize) * self.w + (x as usize) }
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
fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 { v.max(lo).min(hi) }

fn draw_triangle(fb: &mut Framebuffer, v0: &Varyings, v1: &Varyings, v2: &Varyings, light_dir: Vec3, cam_pos: Vec3) {
    let p0 = v0.screen; let p1 = v1.screen; let p2 = v2.screen;

    // Backface culling in screen space
    let area = edge(p0, p1, p2);
    if area >= 0.0 { return; }

    // Bounding box
    let min_x = p0.x.min(p1.x).min(p2.x).floor().max(0.0) as i32;
    let max_x = p0.x.max(p1.x).max(p2.x).ceil().min((fb.w - 1) as f32) as i32;
    let min_y = p0.y.min(p1.y).min(p2.y).floor().max(0.0) as i32;
    let max_y = p0.y.max(p1.y).max(p2.y).ceil().min((fb.h - 1) as f32) as i32;

    let inv_area = 1.0 / area;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = vec3(x as f32 + 0.5, y as f32 + 0.5, 0.0);
            let w0 = edge(p1, p2, p) * inv_area;
            let w1 = edge(p2, p0, p) * inv_area;
            let w2 = edge(p0, p1, p) * inv_area;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 { continue; }

            // Perspective-correct weights
            let invw = v0.inv_w * w0 + v1.inv_w * w1 + v2.inv_w * w2;
            let recip = 1.0 / invw;

            let ndc_z = (v0.screen.z * v0.inv_w * w0 + v1.screen.z * v1.inv_w * w1 + v2.screen.z * v2.inv_w * w2) * recip;
            let z = ndc_depth_to_zbuf(ndc_z);
            let idx = fb.idx(x, y);
            if z >= fb.depth[idx] { continue; }

            let uv = (v0.uv * v0.inv_w * w0 + v1.uv * v1.inv_w * w1 + v2.uv * v2.inv_w * w2) * recip;
            let normal = (v0.normal * v0.inv_w * w0 + v1.normal * v1.inv_w * w1 + v2.normal * v2.inv_w * w2) * recip;
            let world_pos = (v0.world_pos * v0.inv_w * w0 + v1.world_pos * v1.inv_w * w1 + v2.world_pos * v2.inv_w * w2) * recip;

            let base = checkerboard(uv);
            let view_dir = (cam_pos - world_pos).normalize();
            let rgb = lambert_shade(base, normal, light_dir, view_dir).clamp(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0));
            let r = (rgb.x * 255.0) as u32;
            let g = (rgb.y * 255.0) as u32;
            let b = (rgb.z * 255.0) as u32;
            fb.color[idx] = (0xFF << 24) | (r << 16) | (g << 8) | b;
            fb.depth[idx] = z;
        }
    }
}

// A band is a disjoint vertical range of rows [y0, y1) with exclusive access to its color/depth slices.
struct Band<'a> {
    color: &'a mut [u32],
    depth: &'a mut [f32],
    w: usize,
    y0: i32,
    y1: i32,
    // debug visualization
    id: usize,
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

fn draw_triangle_band(band: &mut Band, v0: &Varyings, v1: &Varyings, v2: &Varyings, light_dir: Vec3, cam_pos: Vec3) {
    let p0 = v0.screen; let p1 = v1.screen; let p2 = v2.screen;

    // Backface culling in screen space
    let area = edge(p0, p1, p2);
    if area >= 0.0 { return; }

    // Bounding box clamped to band rows
    let min_x = p0.x.min(p1.x).min(p2.x).floor().max(0.0) as i32;
    let max_x = p0.x.max(p1.x).max(p2.x).ceil().min((band.w - 1) as f32) as i32;
    let mut min_y = p0.y.min(p1.y).min(p2.y).floor() as i32;
    let mut max_y = p0.y.max(p1.y).max(p2.y).ceil() as i32;
    min_y = clamp_i32(min_y, band.y0, band.y1 - 1);
    max_y = clamp_i32(max_y, band.y0, band.y1 - 1);
    if min_y > max_y { return; }

    let inv_area = 1.0 / area;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = vec3(x as f32 + 0.5, y as f32 + 0.5, 0.0);
            let w0 = edge(p1, p2, p) * inv_area;
            let w1 = edge(p2, p0, p) * inv_area;
            let w2 = edge(p0, p1, p) * inv_area;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 { continue; }

            // Perspective-correct weights
            let invw = v0.inv_w * w0 + v1.inv_w * w1 + v2.inv_w * w2;
            let recip = 1.0 / invw;

            let ndc_z = (v0.screen.z * v0.inv_w * w0 + v1.screen.z * v1.inv_w * w1 + v2.screen.z * v2.inv_w * w2) * recip;
            let z = ndc_depth_to_zbuf(ndc_z);
            let idx = band_idx(band, x, y);
            if z >= band.depth[idx] { continue; }

            let uv = (v0.uv * v0.inv_w * w0 + v1.uv * v1.inv_w * w1 + v2.uv * v2.inv_w * w2) * recip;
            let normal = (v0.normal * v0.inv_w * w0 + v1.normal * v1.inv_w * w1 + v2.normal * v2.inv_w * w2) * recip;
            let world_pos = (v0.world_pos * v0.inv_w * w0 + v1.world_pos * v1.inv_w * w1 + v2.world_pos * v2.inv_w * w2) * recip;

            let base = checkerboard(uv);
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

    // Camera
    let cam_pos = vec3(0.0, 0.0, 3.0);
    let target = vec3(0.0, 0.0, 0.0);
    let up = vec3(0.0, 1.0, 0.0);
    let light_dir = vec3(0.6, 0.7, 0.2).normalize();

    // Channels: free buffers -> worker, completed frames -> main
    let (to_worker_tx, mut to_worker_rx) = mpsc::channel::<Vec<u32>>(2);
    let (from_worker_tx, mut from_worker_rx) = mpsc::channel::<Vec<u32>>(2);
    // Scene params (watch latest angle)
    #[derive(Clone, Copy)]
    struct SceneParams { angle: f32, debug: bool }
    let (scene_tx, scene_rx) = watch::channel(SceneParams { angle: 0.0, debug: false });

    // Preallocate double buffers and send to worker
    let buf_size = WIDTH * HEIGHT;
    to_worker_tx.send(vec![0xFF101018; buf_size]).await.ok();
    to_worker_tx.send(vec![0xFF101018; buf_size]).await.ok();

    // Spawn render worker (async task; heavy compute uses Rayon internally)
    let verts_w = verts.clone();
    let tris_w = tris.clone();
    tokio::spawn(async move {
        // Depth buffer reused inside worker
        let mut depth = vec![f32::INFINITY; WIDTH * HEIGHT];
        let cam_pos = cam_pos; // move into task
        let target = target;
        let up = up;
        let light_dir = light_dir;
        let mut scene_rx = scene_rx;

        while let Some(mut color) = to_worker_rx.recv().await {
            // Read latest scene params
            let params = *scene_rx.borrow();

            // Clear buffers
            color.fill(0xFF101018);
            depth.fill(f32::INFINITY);

            // Compute transforms for this frame
            let angle = params.angle;
            let model = Mat4::from_rotation_y(angle) * Mat4::from_rotation_x(angle * 0.5);
            let view = Mat4::look_at_rh(cam_pos, target, up);
            let aspect = WIDTH as f32 / HEIGHT as f32;
            let proj = Mat4::perspective_rh_gl(60f32.to_radians(), aspect, 0.1, 100.0);
            let mvp = proj * view * model;

            // Build varyings
            let mut var: Vec<Varyings> = Vec::with_capacity(verts_w.len());
            for v in &verts_w {
                let world_pos = (model * v.pos.extend(1.0)).truncate();
                let clip: Vec4 = mvp * v.pos.extend(1.0);
                let inv_w = 1.0 / clip.w;
                let ndc = (clip.truncate() * inv_w).clamp(vec3(-1.0, -1.0, -1.0), vec3(1.0, 1.0, 1.0));
                let screen = to_screen(ndc);
                let normal_ws = (model * v.normal.extend(0.0)).truncate();
                var.push(Varyings { screen, inv_w, world_pos, normal: normal_ws, uv: v.uv });
            }

            // Partition into bands over color/depth slices
            let threads = rayon::current_num_threads().max(1);
            let rows_per_band = (HEIGHT + threads - 1) / threads;

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
                bands.push(Band { color: c_chunk, depth: d_chunk, w: WIDTH, y0: y_start, y1: y_end, id: band_id, tint, debug });
                y0 += rows;
                band_id += 1;
            }

            bands.into_par_iter().for_each(|mut band| {
                for t in &tris_w {
                    let a = var[t[0] as usize];
                    let b = var[t[1] as usize];
                    let c = var[t[2] as usize];
                    draw_triangle_band(&mut band, &a, &b, &c, light_dir, cam_pos);
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
    let mut debug: bool = false;
    let mut last_present = Instant::now();
    let mut fps_ema: f32 = 0.0;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Toggle debug mode on key press (no repeat)
        if window.is_key_pressed(Key::D, KeyRepeat::No) {
            debug = !debug;
        }

        angle += 0.8 * (1.0 / 60.0);
        let _ = scene_tx.send(SceneParams { angle, debug });

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
                "Software 3D Renderer (Rust) | {:5.1} FPS | Debug {}",
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
