use glam::{Vec2, Vec3, Vec4};

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

#[derive(Clone, Copy, Debug)]
pub struct Varyings {
    pub clip: Vec4,
    pub screen: Vec3,
    pub inv_w: f32,
    pub world_pos: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}
