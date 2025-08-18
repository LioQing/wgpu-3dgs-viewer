struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    size: vec2<f32>,
}

// Get the aspect ratio of the camera.
fn camera_aspect_ratio(size: vec2<f32>) -> f32 {
    return size.y / size.x;
}

// Transform a world position to camera space.
fn world_to_camera(camera: Camera, world_pos: vec4<f32>) -> vec4<f32> {
    return camera.proj * camera.view * world_pos;
}

// Transform a normalized device coordinate (NDC) position to camera texture coordinates.
fn ndc_to_camera_texture(ndc_pos: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    return (ndc_pos * vec2<f32>(1.0, -1.0) + vec2<f32>(1.0)) * size * 0.5;
}

// Transform a camera texture coordinate to NDC.
fn camera_texture_to_ndc(coords: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    return (coords / size * 2.0 - 1.0) * vec2<f32>(1.0, -1.0);
}