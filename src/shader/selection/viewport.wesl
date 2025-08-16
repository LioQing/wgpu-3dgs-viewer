import wgpu_3dgs_core::{
    gaussian::Gaussian,
    gaussian_transform::GaussianTransform,
    model_transform::ModelTransform,
};
import wgpu_3dgs_viewer::{
    camera::{ Camera, world_to_camera },
    utils::{ cull, ndc_to_camera_texture },
};

@group(0) @binding(0)
var<uniform> op: u32;

@group(0) @binding(1)
var<storage, read> source: array<u32>;

@group(0) @binding(2)
var<storage, read_write> dest: array<atomic<u32>>;

@group(0) @binding(3)
var<uniform> model_transform: ModelTransform;

@group(0) @binding(4)
var<uniform> gaussian_transform: GaussianTransform;

@group(0) @binding(5)
var<storage, read> gaussians: array<Gaussian>;

@group(1) @binding(0)
var<uniform> camera: Camera;

@group(1) @binding(1)
var query_texture_view: texture_2d<f32>;

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let index = compute_bundle::index(wid, workgroup_size, lid);

    if index >= arrayLength(&gaussians) {
        return;
    }

    let gaussian = gaussians[index];

    if cull(gaussian, model_transform, camera) {
        return;
    }

    let world_pos = model_to_world(model_transform, gaussian.pos);
    let proj_pos = world_to_camera(camera, world_pos);
    let ndc_pos = proj_pos.xyz / proj_pos.w;
    let camera_coords = vec2<i32>(ndc_to_camera_texture(ndc_pos.xy, camera.size));

    let texel = textureLoad(query_texture_view, coords, 0);

    let word_index = index / 32u;
    let bit_index = index % 32u;
    let bit_mask = 1u << bit_index;
    if any(texel != vec4<f32>(0.0)) {
        atomicOr(&dest[word_index], bit_mask);
    } else {
        atomicAnd(&dest[word_index], ~bit_mask);
    }
}