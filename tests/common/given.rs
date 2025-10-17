use wgpu_3dgs_core::{Gaussian, Gaussians, glam::*};
use wgpu_3dgs_viewer::{Camera, CameraPod};

use crate::common::TestContext;

pub fn gaussian_with_seed(seed: u32) -> Gaussian {
    let base = seed as f32;

    let rot_x = base + 0.1;
    let rot_y = base + 0.2;
    let rot_z = base + 0.3;
    let rot_w = base + 0.4;
    let rot = Quat::from_xyzw(rot_x, rot_y, rot_z, rot_w).normalize();

    let pos = Vec3::new(base + 1.0, base + 2.0, base + 3.0);

    let color = U8Vec4::new(
        ((base + 10.0) % 256.0) as u8,
        ((base + 20.0) % 256.0) as u8,
        ((base + 30.0) % 256.0) as u8,
        ((base + 40.0) % 256.0) as u8,
    );

    let mut sh = [Vec3::ZERO; 15];
    for i in 0..15 {
        let sh_base = base + (i as f32);
        sh[i] = Vec3::new(sh_base + 0.1, sh_base + 0.2, sh_base + 0.3);
    }

    let scale = Vec3::new(base + 0.1, base + 0.2, base + 0.3);

    Gaussian {
        rot,
        pos,
        color,
        sh,
        scale,
    }
}

pub fn gaussians() -> Gaussians {
    Gaussians {
        gaussians: vec![gaussian_with_seed(42), gaussian_with_seed(123)],
    }
}

pub fn camera() -> CameraPod {
    CameraPod::new(
        // TODO(#8): Fix camera orientation edge case when yaw or pitch is 0.0
        &Camera {
            yaw: 0.1,
            pitch: 0.1,
            ..Camera::new(0.1..1e4, 60f32.to_radians())
        },
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
