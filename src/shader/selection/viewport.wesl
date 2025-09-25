import wgpu_3dgs_core::{
    gaussian::Gaussian,
    gaussian_transform::GaussianTransform,
    model_transform::{ ModelTransform, model_to_world },
};
import package::{
    camera::{ Camera, world_to_camera, ndc_to_camera_texture },
    utils::cull,
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
var selection_texture: texture_2d<f32>;

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;

    if index >= arrayLength(&gaussians) {
        return;
    }

    let gaussian = gaussians[index];

    let world_pos = model_to_world(model_transform, gaussian.pos);
    let proj_pos = world_to_camera(camera, world_pos);
    let ndc_pos = proj_pos.xyz / proj_pos.w;

    let word_index = index / 32u;
    let bit_index = index % 32u;
    let bit_mask = 1u << bit_index;

    if cull(ndc_pos) {
        atomicAnd(&dest[word_index], ~bit_mask);
        return;
    }

    let coords = vec2<i32>(ndc_to_camera_texture(ndc_pos.xy, camera.size));

    let magnitude = textureLoad(selection_texture, coords, 0).r;

    if magnitude != 0.0 {
        atomicOr(&dest[word_index], bit_mask);
    } else {
        atomicAnd(&dest[word_index], ~bit_mask);
    }
}