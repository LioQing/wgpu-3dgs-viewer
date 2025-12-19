use glam::*;
use wgpu::util::DeviceExt;
use wgpu_3dgs_viewer::core::BufferWrapper;
use wgpu_3dgs_viewer::selection::{
    PreprocessorInvertSelectionBuffer, ViewportTexture, ViewportTextureF32Buffer,
    ViewportTexturePosBuffer,
};

use crate::common::TestContext;

#[test]
fn test_viewport_texture_new_should_return_correct_texture() {
    let ctx = TestContext::new();
    let size = UVec2::new(1024, 512);
    let texture = ViewportTexture::new(&ctx.device, size);

    assert_eq!(texture.texture().width(), size.x);
    assert_eq!(texture.texture().height(), size.y);
    assert_eq!(texture.texture().depth_or_array_layers(), 1);
    assert_eq!(texture.texture().mip_level_count(), 1);
    assert_eq!(texture.texture().sample_count(), 1);
    assert_eq!(texture.texture().dimension(), wgpu::TextureDimension::D2);
    assert_eq!(texture.texture().format(), wgpu::TextureFormat::R8Unorm);
    assert!(
        texture
            .texture()
            .usage()
            .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
    );
    assert!(
        texture
            .texture()
            .usage()
            .contains(wgpu::TextureUsages::TEXTURE_BINDING)
    );
}

#[test]
fn test_viewport_texture_pos_buffer_new_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let buffer = ViewportTexturePosBuffer::new(&ctx.device);

    assert_eq!(
        buffer.buffer().size(),
        std::mem::size_of::<Vec2>() as wgpu::BufferAddress
    );
    assert_eq!(
        buffer.buffer().usage(),
        ViewportTexturePosBuffer::DEFAULT_USAGES
    );
}

#[test]
fn test_viewport_texture_pos_buffer_update_should_update_buffer_correctly() {
    let ctx = TestContext::new();
    let buffer =
        ViewportTexturePosBuffer::try_from(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Viewport Texture Pos Buffer"),
            size: std::mem::size_of::<Vec2>() as wgpu::BufferAddress,
            usage: ViewportTexturePosBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
        .expect("try_from");

    let pos = Vec2::new(42.0, 24.0);

    buffer.update(&ctx.queue, pos);

    let downloaded =
        pollster::block_on(buffer.download::<Vec2>(&ctx.device, &ctx.queue)).expect("download")[0];

    assert_eq!(downloaded, pos);
}

#[test]
fn test_viewport_texture_pos_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let pos = Vec2::new(12.0, 34.0);
    let wgpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Viewport Texture Pos Buffer"),
            contents: bytemuck::bytes_of(&pos),
            usage: ViewportTexturePosBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        });

    let converted_buffer =
        ViewportTexturePosBuffer::try_from(wgpu_buffer.clone()).expect("try_from");
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let wgpu_downloaded =
        pollster::block_on(wgpu_converted_buffer.download::<Vec2>(&ctx.device, &ctx.queue))
            .expect("download");
    let converted_downloaded =
        pollster::block_on(converted_buffer.download::<Vec2>(&ctx.device, &ctx.queue))
            .expect("download");
    let wgpu_converted_downloaded =
        pollster::block_on(wgpu_buffer.download::<Vec2>(&ctx.device, &ctx.queue))
            .expect("download");

    assert_eq!(wgpu_downloaded, converted_downloaded);
    assert_eq!(wgpu_downloaded, wgpu_converted_downloaded);
}

#[test]
fn test_viewport_texture_f32_buffer_new_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let buffer = ViewportTextureF32Buffer::new(&ctx.device);

    assert_eq!(
        buffer.buffer().size(),
        std::mem::size_of::<f32>() as wgpu::BufferAddress
    );
    assert_eq!(
        buffer.buffer().usage(),
        ViewportTextureF32Buffer::DEFAULT_USAGES
    );
}

#[test]
fn test_viewport_texture_f32_buffer_update_should_update_buffer_correctly() {
    let ctx = TestContext::new();
    let buffer =
        ViewportTextureF32Buffer::try_from(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Viewport Texture F32 Buffer"),
            size: std::mem::size_of::<f32>() as wgpu::BufferAddress,
            usage: ViewportTextureF32Buffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
        .expect("try_from");

    let value = 0.75f32;

    buffer.update(&ctx.queue, value);

    let downloaded =
        pollster::block_on(buffer.download::<f32>(&ctx.device, &ctx.queue)).expect("download")[0];

    assert_eq!(downloaded, value);
}

#[test]
fn test_viewport_texture_f32_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let value = std::f32::consts::PI;
    let wgpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Viewport Texture F32 Buffer"),
            contents: bytemuck::bytes_of(&value),
            usage: ViewportTextureF32Buffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        });

    let converted_buffer =
        ViewportTextureF32Buffer::try_from(wgpu_buffer.clone()).expect("try_from");
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let wgpu_downloaded =
        pollster::block_on(wgpu_converted_buffer.download::<f32>(&ctx.device, &ctx.queue))
            .expect("download");
    let converted_downloaded =
        pollster::block_on(converted_buffer.download::<f32>(&ctx.device, &ctx.queue))
            .expect("download");
    let wgpu_converted_downloaded =
        pollster::block_on(wgpu_buffer.download::<f32>(&ctx.device, &ctx.queue)).expect("download");

    assert_eq!(wgpu_downloaded, converted_downloaded);
    assert_eq!(wgpu_downloaded, wgpu_converted_downloaded);
}

#[test]
fn test_preprocessor_invert_selection_buffer_new_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let buffer = PreprocessorInvertSelectionBuffer::new(&ctx.device);

    assert_eq!(
        buffer.buffer().size(),
        std::mem::size_of::<u32>() as wgpu::BufferAddress
    );
    assert_eq!(
        buffer.buffer().usage(),
        PreprocessorInvertSelectionBuffer::DEFAULT_USAGES
    );
}

#[test]
fn test_preprocessor_invert_selection_buffer_update_should_update_buffer_correctly() {
    let ctx = TestContext::new();
    let buffer = PreprocessorInvertSelectionBuffer::try_from(ctx.device.create_buffer(
        &wgpu::BufferDescriptor {
            label: Some("Test Preprocessor Invert Selection Buffer"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: PreprocessorInvertSelectionBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        },
    ))
    .expect("try_from");

    buffer.update(&ctx.queue, false);
    let downloaded_false =
        pollster::block_on(buffer.download::<u32>(&ctx.device, &ctx.queue)).expect("download")[0];
    assert_eq!(downloaded_false, 0u32);

    buffer.update(&ctx.queue, true);
    let downloaded_true =
        pollster::block_on(buffer.download::<u32>(&ctx.device, &ctx.queue)).expect("download")[0];
    assert_eq!(downloaded_true, 1u32);
}

#[test]
fn test_preprocessor_invert_selection_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let value = 0u32;
    let wgpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Preprocessor Invert Selection Buffer"),
            contents: bytemuck::bytes_of(&value),
            usage: PreprocessorInvertSelectionBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        });

    let converted_buffer =
        PreprocessorInvertSelectionBuffer::try_from(wgpu_buffer.clone()).expect("try_from");
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let wgpu_downloaded =
        pollster::block_on(wgpu_converted_buffer.download::<u32>(&ctx.device, &ctx.queue))
            .expect("download");
    let converted_downloaded =
        pollster::block_on(converted_buffer.download::<u32>(&ctx.device, &ctx.queue))
            .expect("download");
    let wgpu_converted_downloaded =
        pollster::block_on(wgpu_buffer.download::<u32>(&ctx.device, &ctx.queue)).expect("download");

    assert_eq!(wgpu_downloaded, converted_downloaded);
    assert_eq!(wgpu_downloaded, wgpu_converted_downloaded);
}
