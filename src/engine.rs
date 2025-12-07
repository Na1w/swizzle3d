use glam::{vec3, Mat4, Vec3, Vec4};
use rayon::prelude::*;
use tokio::sync::{mpsc, watch};

use crate::types::{Varyings, Vertex};
use crate::materials::{Material, FACE_MATERIALS};
use crate::rasterizer::{WIDTH, HEIGHT, CHUNK_ROWS, Band, draw_triangle_band_ext, edge, clamp_i32, to_screen};
use crate::clipping::clip_triangle_near;

#[derive(Clone, Copy)]
pub struct SceneParams {
    pub angle: f32,
    pub debug: bool,
    pub debug_depth: bool,
    pub time: f32,
    pub cam_pos: Vec3,
    pub cam_target: Vec3,
    pub light_dir: Vec3,
    pub no_cull: bool,
    pub no_depth: bool,
}

pub struct RenderWorker {
    pub to_worker_rx: mpsc::Receiver<Vec<u32>>,
    pub from_worker_tx: mpsc::Sender<Vec<u32>>,
    pub scene_rx: watch::Receiver<SceneParams>,
    pub verts_w: Vec<Vertex>,
    pub tris_w: Vec<[u32; 3]>,
    pub sphere_verts_w: Vec<Vertex>,
    pub sphere_tris_w: Vec<[u32; 3]>,
}

impl RenderWorker {
    pub async fn run(mut self) {
        let mut depth = vec![1.0f32; WIDTH * HEIGHT];
        let up = vec3(0.0, 1.0, 0.0);

        let mut tri_materials: Vec<Material> = Vec::with_capacity(self.tris_w.len());
        for (i, _) in self.tris_w.iter().enumerate() {
            let mat = if i <= 1 { FACE_MATERIALS[4] }
            else if i <= 3 { FACE_MATERIALS[5] }
            else if i <= 5 { FACE_MATERIALS[2] }
            else if i <= 7 { FACE_MATERIALS[3] }
            else if i <= 9 { FACE_MATERIALS[0] }
            else { FACE_MATERIALS[1] };
            tri_materials.push(mat);
        }

        while let Some(mut color) = self.to_worker_rx.recv().await {
            let params = *self.scene_rx.borrow();

            for y in 0..HEIGHT {
                let gy = (255.0 * (y as f32 / (HEIGHT as f32 - 1.0))) as u32;
                let row_color = 0xFF000000 | (gy << 8) | gy;
                let row = &mut color[y*WIDTH..(y+1)*WIDTH];
                row.fill(row_color);
            }
            depth.fill(1.0);

            let angle = params.angle;
            let time = params.time;

            // Cube movement
            let cube_pos = vec3(
                (time * 0.5).sin() * 0.5,
                (time * 0.3).cos() * 0.3,
                (time * 0.4).sin() * 0.5
            );

            let model = Mat4::from_translation(cube_pos)
                * Mat4::from_rotation_y(angle)
                * Mat4::from_rotation_x(angle * 0.5);

            let view = Mat4::look_at_rh(params.cam_pos, params.cam_target, up);
            let aspect = WIDTH as f32 / HEIGHT as f32;
            let proj = Mat4::perspective_rh_gl(60f32.to_radians(), aspect, 0.1, 100.0);
            let mvp = proj * view * model;

            let mut var: Vec<Varyings> = Vec::with_capacity(self.verts_w.len());
            for v in &self.verts_w {
                let world_pos = (model * v.pos.extend(1.0)).truncate();
                let clip: Vec4 = mvp * v.pos.extend(1.0);
                let inv_w = 1.0 / clip.w;
                let ndc = clip.truncate() * inv_w;
                let screen = to_screen(ndc);
                let normal_ws = (model * v.normal.extend(0.0)).truncate();
                var.push(Varyings { clip, screen, inv_w, world_pos, normal: normal_ws, uv: v.uv });
            }

            let rows_per_band = CHUNK_ROWS;
            let num_bands = HEIGHT.div_ceil(rows_per_band);

            let mut clipped_tris: Vec<([Varyings;3], Material)> = Vec::new();
            for (ti, t) in self.tris_w.iter().enumerate() {
                let a = var[t[0] as usize];
                let b = var[t[1] as usize];
                let c = var[t[2] as usize];
                if !params.no_cull {
                    let area = edge(a.screen, b.screen, c.screen);
                    if area >= 0.0 { continue; }
                }
                for tri in clip_triangle_near(&a, &b, &c) {
                    clipped_tris.push((tri, tri_materials[ti]));
                }
            }

            let sphere_model = Mat4::IDENTITY;
            let sphere_mvp = proj * view * sphere_model;
            let mut svar: Vec<Varyings> = Vec::with_capacity(self.sphere_verts_w.len());
            for v in &self.sphere_verts_w {
                let world_pos = (sphere_model * v.pos.extend(1.0)).truncate();
                let clip: Vec4 = sphere_mvp * v.pos.extend(1.0);
                let inv_w = 1.0 / clip.w;
                let ndc = clip.truncate() * inv_w;
                let screen = to_screen(ndc);
                let normal_ws = (sphere_model * v.normal.extend(0.0)).truncate();
                svar.push(Varyings { clip, screen, inv_w, world_pos, normal: normal_ws, uv: v.uv });
            }

            for t in &self.sphere_tris_w {
                let a = svar[t[0] as usize];
                let b = svar[t[1] as usize];
                let c = svar[t[2] as usize];
                for tri in clip_triangle_near(&a, &b, &c) {
                    clipped_tris.push((tri, Material::Checker));
                }
            }

            let mut bins: Vec<Vec<usize>> = vec![Vec::new(); num_bands];
            for (ti, (tri, _)) in clipped_tris.iter().enumerate() {
                let a = tri[0].screen; let b = tri[1].screen; let c = tri[2].screen;
                let mut min_y = a.y.min(b.y).min(c.y).floor() as i32;
                let mut max_y = a.y.max(b.y).max(c.y).ceil() as i32;
                min_y = clamp_i32(min_y, 0, (HEIGHT - 1) as i32);
                max_y = clamp_i32(max_y, 0, (HEIGHT - 1) as i32);
                if min_y > max_y { continue; }
                let first_band = (min_y as usize) / rows_per_band;
                let last_band = (max_y as usize) / rows_per_band;
                for bin in bins.iter_mut().take(last_band + 1).skip(first_band) {
                    bin.push(ti);
                }
            }

            let mut bands: Vec<Band> = Vec::new();
            let mut y0 = 0usize;
            let tints: [u32; 12] = [
                0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFF00, 0xFFFF00FF, 0xFF00FFFF,
                0xFFFF8000, 0xFF8080FF, 0xFF80FF80, 0xFFFF80FF, 0xFF80FFFF, 0xFFFFFF80,
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

            struct BandTask<'a> {
                band: Band<'a>,
                bin: Vec<usize>,
            }
            let mut tasks: Vec<BandTask> = Vec::with_capacity(bands.len());
            for (i, band) in bands.into_iter().enumerate() {
                use std::mem;
                let bin = mem::take(&mut bins[i]);
                tasks.push(BandTask { band, bin });
            }

            let light_dir = params.light_dir.normalize();
            let no_cull = params.no_cull;
            let no_depth = params.no_depth;
            let cam_pos = params.cam_pos;

            tasks.into_par_iter().for_each(|mut t| {
                for &ti in &t.bin {
                    let (tri, material) = &clipped_tris[ti];
                    draw_triangle_band_ext(&mut t.band, &tri[0], &tri[1], &tri[2], light_dir, cam_pos, time, *material, no_cull, no_depth);
                }
            });

            if params.debug_depth {
                for i in 0..color.len() {
                    let z = depth[i];
                    if z < 1.0 {
                        let g = (z * 255.0).clamp(0.0, 255.0) as u32;
                        let px = 0xFF000000 | (g << 16) | (g << 8) | g;
                        color[i] = px;
                    } else {
                        color[i] = 0xFF101018;
                    }
                }
            }

            if self.from_worker_tx.send(color).await.is_err() {
                break;
            }
        }
    }
}
