use glam::{vec2, vec3, Vec3};
use crate::types::Vertex;

pub fn make_cube(size: f32) -> (Vec<Vertex>, Vec<[u32; 3]>) {
    let s = size * 0.5;
    let p = [
        vec3(-s, -s,  s), vec3( s, -s,  s), vec3( s,  s,  s), vec3(-s,  s,  s),
        vec3(-s, -s, -s), vec3(-s,  s, -s), vec3( s,  s, -s), vec3( s, -s, -s),
        vec3(-s,  s, -s), vec3(-s,  s,  s), vec3( s,  s,  s), vec3( s,  s, -s),
        vec3(-s, -s, -s), vec3( s, -s, -s), vec3( s, -s,  s), vec3(-s, -s,  s),
        vec3( s, -s, -s), vec3( s,  s, -s), vec3( s,  s,  s), vec3( s, -s,  s),
        vec3(-s, -s, -s), vec3(-s, -s,  s), vec3(-s,  s,  s), vec3(-s,  s, -s),
    ];
    let n: [Vec3; 24] = [
        vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0),
        vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, -1.0),
        vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0),
        vec3(0.0, -1.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0, -1.0, 0.0),
        vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0),
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
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [8, 9, 10], [8, 10, 11],
        [12, 13, 14], [12, 14, 15],
        [16, 17, 18], [16, 18, 19],
        [20, 21, 22], [20, 22, 23],
    ];
    (verts, idx)
}

pub fn make_uv_sphere(rings: u32, segments: u32, radius: f32) -> (Vec<Vertex>, Vec<[u32; 3]>) {
    let rings = rings.max(2);
    let segments = segments.max(3);
    let mut verts: Vec<Vertex> = Vec::with_capacity((rings as usize + 1) * (segments as usize + 1));
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let theta = v * std::f32::consts::PI;
        let (st, ct) = theta.sin_cos();
        for s in 0..=segments {
            let u = s as f32 / segments as f32;
            let phi = u * std::f32::consts::TAU;
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
            tris.push([i0 as u32, i1 as u32, i2 as u32]);
            tris.push([i1 as u32, i3 as u32, i2 as u32]);
        }
    }
    (verts, tris)
}
