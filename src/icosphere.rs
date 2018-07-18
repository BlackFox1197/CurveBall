use cgmath::InnerSpace;
use cgmath::Vector3;

use crate::vertex::Vertex;

pub fn icosphere(subdivison_level: u8) -> (Vec<Vertex>, Vec<u32>) {
    let (mut vertices, mut indices) = create_icosahedron();

    let radius = vertices[0].magnitude();
    let mut num_triangles = 20;

    for _ in 0..subdivison_level {
        for offset in (0..num_triangles).map(|x| x * 3) {
            //           p0
            //           /\
            //          /  \
            //         /    \
            //      p3/------\ p5
            //       / \    / \
            //      /   \  /   \
            //     /     \/     \
            // p1 /--------------\ p2
            //           p4

            let p0_idx = indices[offset];
            let p1_idx = indices[offset + 1];
            let p2_idx = indices[offset + 2];

            let p0 = &vertices[p0_idx as usize];
            let p1 = &vertices[p1_idx as usize];
            let p2 = &vertices[p2_idx as usize];

            let p3_idx = vertices.len() as u32;
            let p4_idx = (vertices.len() + 1) as u32;
            let p5_idx = (vertices.len() + 2) as u32;

            let p3 = (p0 + 0.5 * (p1 - p0)).normalize_to(radius);
            let p4 = (p1 + 0.5 * (p2 - p1)).normalize_to(radius);
            let p5 = (p2 + 0.5 * (p0 - p2)).normalize_to(radius);

            vertices.push(p3);
            vertices.push(p4);
            vertices.push(p5);

            indices[offset + 1] = p3_idx;
            indices[offset + 2] = p5_idx;

            indices.push(p3_idx);
            indices.push(p1_idx);
            indices.push(p4_idx);

            indices.push(p5_idx);
            indices.push(p4_idx);
            indices.push(p2_idx);

            indices.push(p4_idx);
            indices.push(p5_idx);
            indices.push(p3_idx);
        }
        num_triangles *= 4;
    }

    (vertices.into_iter().map(|v| v.into()).collect(), indices)
}

fn create_icosahedron() -> (Vec<Vector3<f32>>, Vec<u32>) {
    let tau = (1. + 5f32.sqrt()) / 2.;

    let vertices = vec![
        Vector3::new(1., tau, 0.), // p0
        Vector3::new(1., -tau, 0.),
        Vector3::new(-1., -tau, 0.), // p2
        Vector3::new(-1., tau, 0.),
        //
        Vector3::new(0., 1., tau),
        Vector3::new(0., 1., -tau),
        Vector3::new(0., -1., -tau),
        Vector3::new(0., -1., tau),
        //
        Vector3::new(tau, 0., 1.),
        Vector3::new(-tau, 0., 1.),
        Vector3::new(-tau, 0., -1.),
        Vector3::new(tau, 0., -1.),
    ];
    let mut indices = Vec::new();
    // TODO: replace with static values and correct winding direction
    for i in 0..12 {
        for j in 0..12 {
            for k in 0..12 {
                if !(i < j && j < k) {
                    continue;
                }
                let a = (vertices[i] - vertices[j]).magnitude();
                let b = (vertices[i] - vertices[k]).magnitude();
                let c = (vertices[j] - vertices[k]).magnitude();
                if (2. - a).abs() < 0.1 && (a - b).abs() < 0.1 && (a - c).abs() < 0.1 {
                    indices.push(i as u32);
                    indices.push(j as u32);
                    indices.push(k as u32);
                }
            }
        }
    }

    (vertices, indices)
}
