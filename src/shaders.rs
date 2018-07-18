#![allow(dead_code)]

pub mod vs {
    use vulkano_shader_derive::VulkanoShader;
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[path = "src/shaders/vertex.glsl"]
    struct Dummy;
}

pub mod fs {
    use vulkano_shader_derive::VulkanoShader;
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[path = "src/shaders/fragment.glsl"]
    struct Dummy;
}
