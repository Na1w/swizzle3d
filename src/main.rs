mod types;
mod materials;
mod geometry;
mod rasterizer;
mod clipping;
mod engine;

use glam::{vec3};
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use std::time::Instant;
use tokio::sync::{mpsc, watch};

use geometry::{make_cube, make_uv_sphere};
use rasterizer::{WIDTH, HEIGHT};
use engine::{RenderWorker, SceneParams};

const WINDOW_TITLE: &str = "Swizzle3d";

#[tokio::main]
async fn main() {
    let mut window = Window::new(
        WINDOW_TITLE,
        WIDTH,
        HEIGHT,
        WindowOptions { resize: true, ..WindowOptions::default() },
    )
    .expect("Unable to create window");

    window.limit_update_rate(Some(std::time::Duration::from_micros(16_667)));

    let (verts, tris) = make_cube(1.6);
    let (sphere_verts, sphere_tris) = make_uv_sphere(32, 48, 1.2);

    let (to_worker_tx, to_worker_rx) = mpsc::channel::<Vec<u32>>(2);
    let (from_worker_tx, mut from_worker_rx) = mpsc::channel::<Vec<u32>>(2);

    let (scene_tx, scene_rx) = watch::channel(SceneParams {
        angle: 0.0,
        debug: false,
        debug_depth: false,
        time: 0.0,
        cam_pos: vec3(0.0, 0.0, 3.0),
        cam_target: vec3(0.0, 0.0, 0.0),
        light_dir: vec3(0.6, 0.7, 0.2).normalize(),
        no_cull: false,
        no_depth: false,
    });

    let buf_size = WIDTH * HEIGHT;
    to_worker_tx.send(vec![0xFF101018; buf_size]).await.ok();
    to_worker_tx.send(vec![0xFF101018; buf_size]).await.ok();

    let worker = RenderWorker {
        to_worker_rx,
        from_worker_tx,
        scene_rx,
        verts_w: verts,
        tris_w: tris,
        sphere_verts_w: sphere_verts,
        sphere_tris_w: sphere_tris,
    };
    tokio::spawn(worker.run());

    let mut angle: f32 = 0.0;
    let start_time = Instant::now();
    let mut debug: bool = false;
    let mut debug_depth: bool = false;
    let mut last_present = Instant::now();
    let mut fps_ema: f32 = 0.0;
    let mut cam_pos = vec3(0.0, 0.0, 3.0);
    let mut yaw: f32 = std::f32::consts::PI;
    let mut pitch: f32 = 0.0;
    let mut last_frame = Instant::now();
    let mut no_cull = false;
    let mut no_depth = false;
    let mut have_presented = false;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now_loop = Instant::now();
        let dt = now_loop.duration_since(last_frame).as_secs_f32().min(0.05);
        last_frame = now_loop;

        if window.is_key_pressed(Key::F1, KeyRepeat::No) { debug = !debug; }
        if window.is_key_pressed(Key::F2, KeyRepeat::No) { debug_depth = !debug_depth; }
        if window.is_key_pressed(Key::F3, KeyRepeat::No) { no_cull = !no_cull; }
        if window.is_key_pressed(Key::F4, KeyRepeat::No) { no_depth = !no_depth; }

        angle += 0.8 * dt;
        let turn_speed = 1.8;
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
        let light_radius = 1.0;
        let light_dir = vec3((time * 0.7).cos() * light_radius, 0.6, (time * 0.7).sin() * light_radius).normalize();
        let cam_target = cam_pos + forward;
        let _ = scene_tx.send(SceneParams { angle, debug, debug_depth, time, cam_pos, cam_target, light_dir, no_cull, no_depth });

        match from_worker_rx.try_recv() {
            Ok(color) => {
                window.update_with_buffer(&color, WIDTH, HEIGHT).expect("Failed to update window");
                have_presented = true;
                let now = Instant::now();
                let dt = now.duration_since(last_present).as_secs_f32().max(1e-6);
                last_present = now;
                let inst_fps = 1.0 / dt;
                fps_ema = if fps_ema == 0.0 { inst_fps } else { fps_ema * 0.9 + inst_fps * 0.1 };
                window.set_title(&format!(
                    "{} | {:5.1} FPS | Debug {} (F1) | Depth {} (F2) | NoCull {} (F3) | NoDepth {} (F4)",
                    WINDOW_TITLE,
                    fps_ema,
                    if debug { "ON" } else { "OFF" },
                    if debug_depth { "ON" } else { "OFF" },
                    if no_cull { "ON" } else { "OFF" },
                    if no_depth { "ON" } else { "OFF" }
                ));
                if let Err(e) = to_worker_tx.try_send(color) {
                    eprintln!("WARN: failed to recycle buffer to worker: {e:?}");
                }
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                window.update();
                if !have_presented && start_time.elapsed().as_secs_f32() > 0.5 {
                    window.set_title(&format!("{} | waiting for first frame...", WINDOW_TITLE));
                }
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                break;
            }
        }
    }
}
