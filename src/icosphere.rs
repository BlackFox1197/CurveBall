use cgmath::InnerSpace;
use cgmath::Vector3;
use std::collections::HashMap;

use crate::vertex::Vertex;

pub fn icosphere(subdivison_level: u8) -> (Vec<Vertex>, Vec<u32>) {
    let (mut vertices, mut indices) = create_icosahedron();

    let radius = vertices[0].magnitude();
    let mut num_faces = 20;
    debug_assert_eq!(num_faces * 3, indices.len());

    let final_faces = num_faces * (4usize.pow(subdivison_level.into()));
    // Euler's formula: Vertices = 2 + Edges - Faces
    let final_vertices = 2 + (final_faces * 3 / 2) - final_faces;
    let final_indices = final_faces * 3;
    vertices.reserve_exact(final_vertices - vertices.len());
    indices.reserve_exact(final_indices - indices.len());

    let mut vertex_cache: HashMap<(u32, u32), u32> = HashMap::new();
    let mut get_middle_point = |mut p0_idx: u32, mut p1_idx: u32| {
        if p1_idx < p0_idx {
            std::mem::swap(&mut p0_idx, &mut p1_idx);
        }

        match vertex_cache.get(&(p0_idx, p1_idx)) {
            Some(mid_idx) => *mid_idx,
            None => {
                let p0 = &vertices[p0_idx as usize];
                let p1 = &vertices[p1_idx as usize];
                let mid = (p0 + 0.5 * (p1 - p0)).normalize_to(radius);

                let index = vertices.len();
                vertices.push(mid);
                vertex_cache.insert((p0_idx, p1_idx), index as u32);
                index as u32
            }
        }
    };

    for _ in 0..subdivison_level {
        for offset in (0..num_faces).map(|x| x * 3) {
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

            let p3_idx = get_middle_point(p0_idx, p1_idx);
            let p4_idx = get_middle_point(p1_idx, p2_idx);
            let p5_idx = get_middle_point(p2_idx, p0_idx);

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
        num_faces *= 4;
    }

    debug_assert_eq!(final_vertices, vertices.len());
    debug_assert_eq!(final_indices, indices.len());

    (vertices.into_iter().map(|v| v.into()).collect(), indices)
}

// TODO: replace with static or const
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
