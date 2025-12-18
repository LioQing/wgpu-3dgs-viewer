use glam::*;
use wgpu::util::DeviceExt;
use wgpu_3dgs_viewer::{Camera, CameraBuffer, CameraPod, CameraTrait, core::BufferWrapper};

use crate::common::TestContext;

#[test]
fn test_camera_buffer_new_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let buffer = CameraBuffer::new(&ctx.device);

    assert_eq!(
        buffer.buffer().size(),
        std::mem::size_of::<CameraPod>() as wgpu::BufferAddress
    );
}

#[test]
fn test_camera_buffer_update_should_update_buffer_correctly() {
    let ctx = TestContext::new();
    let buffer = CameraBuffer::try_from(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Camera Buffer"),
        size: std::mem::size_of::<CameraPod>() as wgpu::BufferAddress,
        usage: CameraBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }))
    .expect("try_from");

    let camera = Camera::new(0.1..100.0, std::f32::consts::FRAC_PI_4);
    let size = UVec2::new(800, 600);
    let pod = CameraPod::new(&camera, size);

    buffer.update(&ctx.queue, &camera, size);

    let downloaded = pollster::block_on(buffer.download::<CameraPod>(&ctx.device, &ctx.queue))
        .expect("download")[0];

    assert_eq!(downloaded, pod);
}

#[test]
fn test_camera_buffer_update_with_pod_should_update_buffer_correctly() {
    let ctx = TestContext::new();
    let buffer = CameraBuffer::try_from(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Camera Buffer"),
        size: std::mem::size_of::<CameraPod>() as wgpu::BufferAddress,
        usage: CameraBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }))
    .expect("try_from");

    let camera = Camera::new(0.1..100.0, std::f32::consts::FRAC_PI_4);
    let size = UVec2::new(1920, 1080);
    let pod = CameraPod::new(&camera, size);

    buffer.update_with_pod(&ctx.queue, &pod);

    let downloaded = pollster::block_on(buffer.download::<CameraPod>(&ctx.device, &ctx.queue))
        .expect("download")[0];

    assert_eq!(downloaded, pod);
}

#[test]
fn test_camera_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let camera = Camera::new(0.1..100.0, std::f32::consts::FRAC_PI_4);
    let size = UVec2::new(800, 600);
    let pod = CameraPod::new(&camera, size);
    let wgpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Camera Buffer"),
            contents: bytemuck::bytes_of(&pod),
            usage: CameraBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        });

    let converted_buffer = CameraBuffer::try_from(wgpu_buffer.clone()).expect("try_from");
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let wgpu_downloaded =
        pollster::block_on(wgpu_converted_buffer.download::<CameraPod>(&ctx.device, &ctx.queue))
            .expect("download");
    let converted_downloaded =
        pollster::block_on(converted_buffer.download::<CameraPod>(&ctx.device, &ctx.queue))
            .expect("download");
    let wgpu_converted_downloaded =
        pollster::block_on(wgpu_buffer.download::<CameraPod>(&ctx.device, &ctx.queue))
            .expect("download");

    assert_eq!(wgpu_downloaded, converted_downloaded);
    assert_eq!(wgpu_downloaded, wgpu_converted_downloaded);
}

#[test]
fn test_camera_pod_new_should_return_correct_pod() {
    let camera = Camera::new(0.1..100.0, std::f32::consts::FRAC_PI_4);
    let size = UVec2::new(1280, 720);
    let pod = CameraPod::new(&camera, size);

    let expected_view = camera.view();
    let expected_proj = camera.projection(size.x as f32 / size.y as f32);

    assert_eq!(pod.view, expected_view);
    assert_eq!(pod.proj, expected_proj);
    assert_eq!(pod.size, size.as_vec2());
}

#[test]
fn test_camera_pod_new_with_modified_camera_should_return_correct_pod() {
    let mut camera = Camera::new(0.1..100.0, std::f32::consts::FRAC_PI_4);
    camera.pos = Vec3::new(5.0, 10.0, 15.0);
    camera.pitch_by(std::f32::consts::FRAC_PI_6);
    camera.yaw_by(std::f32::consts::FRAC_PI_4);

    let size = UVec2::new(1024, 768);
    let pod = CameraPod::new(&camera, size);

    let expected_view = camera.view();
    let expected_proj = camera.projection(size.x as f32 / size.y as f32);

    assert_eq!(pod.view, expected_view);
    assert_eq!(pod.proj, expected_proj);
    assert_eq!(pod.size, size.as_vec2());
}
