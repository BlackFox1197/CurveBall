pub mod vs {
    vulkano_shaders::shader!{ ty: "vertex", path: "src/shaders/vertex.glsl"}
}

pub mod fs {
    vulkano_shaders::shader!{ ty: "fragment", path: "src/shaders/fragment.glsl"}
}
