use glam::*;

use crate::{
    CameraTrait, Error, QueryHitPod, QueryHitResultPod, QueryResultCountBuffer, QueryResultPod,
    QueryResultsBuffer,
};

/// Download the query results from the GPU.
pub async fn download(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    result_count: &QueryResultCountBuffer,
    results: &QueryResultsBuffer,
) -> Result<Vec<QueryResultPod>, Error> {
    match result_count.download(device, queue).await? {
        0 => Ok(Vec::new()),
        count => results.download(device, queue, count).await,
    }
}

/// Get the world position of the [`QueryType::Hit`](crate::QueryType::Hit) query.
///
/// This uses the closest depth for finding the hit.
///
/// Returns the index of the hit result and the world position.
pub fn hit_pos_by_closest(
    query: &QueryHitPod,
    results: &[QueryHitResultPod],
    camera: &impl CameraTrait,
    texture_size: UVec2,
) -> Option<(usize, Vec3)> {
    let (index, hit) = results.iter().enumerate().min_by(|(_, a), (_, b)| {
        a.depth()
            .partial_cmp(&b.depth())
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;

    let world_pos = coords_and_depth_to_world(camera, query.coords(), hit.depth(), texture_size);

    Some((index, world_pos))
}

/// Get the world position of the [`QueryType::Hit`](crate::QueryType::Hit) query.
///
/// This uses the most alpha contribution after alpha blending.
///
/// This also sorts the results by depth.
///
/// Returns the index of the hit result after being sorted, the blended alpha of the hit Gaussian,
/// and the world position.
pub fn hit_pos_by_most_alpha(
    query: &QueryHitPod,
    results: &mut [QueryHitResultPod],
    camera: &impl CameraTrait,
    texture_size: UVec2,
) -> Option<(usize, f32, Vec3)> {
    results.sort_by(|a, b| {
        a.depth()
            .partial_cmp(&b.depth())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut src_alpha = 0.0;
    for result in results.iter_mut() {
        *result.alpha_mut() *= 1.0 - src_alpha;
        src_alpha = result.alpha();
    }

    let (index, (alpha, hit)) = results
        .iter()
        .scan(0.0, |src_alpha, result| {
            let dst_alpha = result.alpha() * (1.0 - *src_alpha);
            *src_alpha = dst_alpha;

            Some((dst_alpha, result))
        })
        .enumerate()
        .max_by(|(_, (a, _)), (_, (b, _))| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;

    let world_pos = coords_and_depth_to_world(camera, query.coords(), hit.depth(), texture_size);

    Some((index, alpha, world_pos))
}

/// Gets the world position from the texture coordinates and the normalized depth.
fn coords_and_depth_to_world(
    camera: &impl CameraTrait,
    coords: Vec2,
    depth: f32,
    texture_size: UVec2,
) -> Vec3 {
    let texture_size = texture_size.as_vec2();
    let pos_ndc = ((coords * vec2(1.0, -1.0) + vec2(0.0, texture_size.y - 1.0)) / texture_size
        * 2.0
        - Vec2::ONE)
        .extend(depth)
        .extend(1.0);

    let transform_mat = camera.projection(texture_size.x / texture_size.y) * camera.view();
    let pos_inverted = transform_mat.inverse() * pos_ndc;

    pos_inverted.xyz() / pos_inverted.w
}
