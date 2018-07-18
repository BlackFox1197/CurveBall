#![cfg_attr(feature = "cargo-clippy", allow(ref_in_deref))]

use cgmath::Vector3;
use vulkano::impl_vertex;

#[derive(Debug, Clone)]
pub struct Vertex {
    position: [f32; 3],
}
impl_vertex!(Vertex, position);

impl From<Vector3<f32>> for Vertex {
    fn from(vec3: Vector3<f32>) -> Vertex {
        Vertex {
            position: [vec3.x, vec3.y, vec3.z],
        }
    }
}
