struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    size: vec2<f32>,
}

fn camera_aspect_ratio(size: vec2<f32>) -> f32 {
    return size.y / size.x;
}