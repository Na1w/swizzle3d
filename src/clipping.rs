use smallvec::SmallVec;
use crate::types::Varyings;
use crate::rasterizer::to_screen;

#[inline]
fn inside_near(v: &Varyings) -> bool { v.clip.z + v.clip.w >= 0.0 }

#[inline]
fn lerp_varyings(a: &Varyings, b: &Varyings, t: f32) -> Varyings {
    let clip = a.clip * (1.0 - t) + b.clip * t;
    let inv_w = 1.0 / clip.w;
    let ndc = (clip.truncate()) * inv_w;
    let screen = to_screen(ndc);
    let world_pos = a.world_pos * (1.0 - t) + b.world_pos * t;
    let normal = (a.normal * (1.0 - t) + b.normal * t).normalize();
    let uv = a.uv * (1.0 - t) + b.uv * t;
    Varyings { clip, screen, inv_w, world_pos, normal, uv }
}

pub fn clip_triangle_near(v0: &Varyings, v1: &Varyings, v2: &Varyings) -> SmallVec<[[Varyings;3]; 2]> {
    let mut inlist: SmallVec<[Varyings; 8]> = SmallVec::new();
    inlist.push(*v0); inlist.push(*v1); inlist.push(*v2);
    let mut outlist: SmallVec<[Varyings; 8]> = SmallVec::new();
    for i in 0..inlist.len() {
        let curr = inlist[i];
        let prev = inlist[(i + inlist.len() - 1) % inlist.len()];
        let curr_in = inside_near(&curr);
        let prev_in = inside_near(&prev);
        if curr_in {
            if prev_in {
                outlist.push(curr);
            } else {
                let sa = prev.clip.z + prev.clip.w;
                let sb = curr.clip.z + curr.clip.w;
                let t = sa / (sa - sb);
                outlist.push(lerp_varyings(&prev, &curr, t));
                outlist.push(curr);
            }
        } else if prev_in {
            let sa = prev.clip.z + prev.clip.w;
            let sb = curr.clip.z + curr.clip.w;
            let t = sa / (sa - sb);
            outlist.push(lerp_varyings(&prev, &curr, t));
        }
    }

    let mut out_tris: SmallVec<[[Varyings;3]; 2]> = SmallVec::new();
    if outlist.len() < 3 { return out_tris; }
    let v0 = outlist[0];
    for i in 1..(outlist.len()-1) {
        out_tris.push([v0, outlist[i], outlist[i+1]]);
    }
    out_tris
}
