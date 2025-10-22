use wgpu_3dgs_core::glam::*;
use wgpu_3dgs_viewer::{Camera, CameraPod};

use crate::common::TestContext;

pub fn camera() -> Camera {
    Camera {
        yaw: 0.1,
        pitch: 0.1,
        ..Camera::new(0.1..1e4, 60f32.to_radians())
    }
}

pub fn camera_pod() -> CameraPod {
    CameraPod::new(
        // TODO(#8): Fix camera orientation edge case when yaw or pitch is 0.0
        &camera(),
        UVec2::new(1024, 1024),
    )
}

pub fn render_target_texture(ctx: &TestContext) -> wgpu::Texture {
    ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Render Target"),
        size: wgpu::Extent3d {
            width: 1024,
            height: 1024,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}
