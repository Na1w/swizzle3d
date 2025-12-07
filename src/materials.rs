use glam::{vec2, vec3, Vec2, Vec3};

#[derive(Clone, Copy)]
pub enum Material {
    Checker,
    Starfield,
    Fire,
}

#[inline]
fn hash2(x: Vec2) -> f32 {
    let p = x;
    let h = (p.x * 127.1 + p.y * 311.7).sin() * 43_758.547;
    h.fract().abs()
}

#[inline]
fn value_noise(p: Vec2) -> f32 {
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

fn checkerboard(uv: Vec2) -> Vec3 {
    let scale = 8.0;
    let u = (uv.x * scale).floor() as i32;
    let v = (uv.y * scale).floor() as i32;
    let c = ((u ^ v) & 1) as f32;
    let a = vec3(0.12, 0.12, 0.12);
    let b = vec3(0.85, 0.85, 0.85);
    a * (1.0 - c) + b * c
}

fn sample_starfield(uv: Vec2, time: f32) -> Vec3 {
    let scale = 40.0;
    let p = uv * scale;
    let cell = vec2(p.x.floor(), p.y.floor());
    let r = hash2(cell);
    let density = 0.975;
    if r > density {
        let phase = hash2(cell + vec2(7.3, 9.2)) * std::f32::consts::TAU;
        let twinkle = 0.6 + 0.4 * (time * 6.0 + phase).sin().abs();
        let b = ((r - density) / (1.0 - density)).powf(4.0) * twinkle;
        let color = vec3(0.8, 0.9, 1.0) * b;
        return color.clamp(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0));
    }
    let bg = 0.02 + 0.03 * fbm(uv * 3.0 + vec2(0.0, time * 0.05), 3);
    vec3(bg, bg, bg)
}

fn fire_ramp(t: f32) -> Vec3 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.3 {
        let k = t / 0.3;
        vec3(0.0, 0.0, 0.0) * (1.0 - k) + vec3(1.0, 0.0, 0.0) * k
    } else if t < 0.6 {
        let k = (t - 0.3) / 0.3;
        vec3(1.0, 0.0, 0.0) * (1.0 - k) + vec3(1.0, 0.5, 0.0) * k
    } else if t < 0.85 {
        let k = (t - 0.6) / 0.25;
        vec3(1.0, 0.5, 0.0) * (1.0 - k) + vec3(1.0, 1.0, 0.0) * k
    } else {
        let k = (t - 0.85) / 0.15;
        vec3(1.0, 1.0, 0.0) * (1.0 - k) + vec3(1.0, 1.0, 1.0) * k
    }
}

fn sample_fire(uv: Vec2, time: f32) -> Vec3 {
    let p = vec2(uv.x, 1.0 - uv.y);
    let speed = 0.6;
    let scroll = time * speed;
    let turbulence = fbm(vec2(p.x * 3.0, p.y * 6.0 + scroll * 4.0), 4);
    let base = (1.2 * (1.0 - p.y) + 0.8 * turbulence).clamp(0.0, 1.0);
    let flicker = 0.85 + 0.15 * (time * 10.0 + hash2(vec2((p.x * 5.0).floor(), (p.y * 8.0).floor())) * std::f32::consts::TAU).sin().abs();
    let t = (base.powf(1.2) * flicker).clamp(0.0, 1.0);
    fire_ramp(t)
}

#[inline]
pub fn sample_material(uv: Vec2, material: Material, time: f32) -> Vec3 {
    match material {
        Material::Checker => checkerboard(uv),
        Material::Starfield => sample_starfield(uv * 1.0, time),
        Material::Fire => sample_fire(uv * 1.0, time),
    }
}

pub const FACE_MATERIALS: [Material; 6] = [
    Material::Fire,
    Material::Checker,
    Material::Checker,
    Material::Checker,
    Material::Starfield,
    Material::Checker,
];
