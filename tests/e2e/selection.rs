use glam::*;
use wgpu_3dgs_viewer::{
    Viewer,
    core::{
        BufferWrapper, Gaussian, GaussianPodWithShSingleCov3dSingleConfigs, Gaussians,
        GaussiansBuffer,
    },
};

use crate::common::{TestContext, given};

type G = GaussianPodWithShSingleCov3dSingleConfigs;

#[test]
fn test_viewer_when_gaussian_is_in_selected_region_should_be_selected() {
    let ctx = TestContext::new();
    let gaussians = Gaussians {
        gaussians: vec![Gaussian {
            rot: Quat::IDENTITY,
            pos: Vec3::ZERO + Vec3::Z,
            color: U8Vec4::new(255, 0, 0, 255),
            sh: [Vec3::ZERO; 15],
            scale: Vec3::splat(1.0),
        }],
    };

    let render_target = given::render_target_texture(&ctx);

    let mut viewer = Viewer::<G>::new_with(
        &ctx.device,
        wgpu::TextureFormat::Rgba8Unorm,
        None,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        &gaussians,
    )
    .expect("viewer");
}

#[test]
fn test_viewer_when_gaussian_is_not_in_selected_region_should_not_be_selected() {
    let ctx = TestContext::new();
    let gaussians = Gaussians {
        gaussians: vec![Gaussian {
            rot: Quat::IDENTITY,
            pos: Vec3::ZERO + Vec3::Z,
            color: U8Vec4::new(255, 0, 0, 255),
            sh: [Vec3::ZERO; 15],
            scale: Vec3::splat(1.0),
        }],
    };

    let render_target = given::render_target_texture(&ctx);

    let mut viewer = Viewer::<G>::new_with(
        &ctx.device,
        wgpu::TextureFormat::Rgba8Unorm,
        None,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        &gaussians,
    )
    .expect("viewer");
}

#[test]
fn test_viewer_when_gaussian_is_selected_and_modified_should_modify_correctly() {
    let ctx = TestContext::new();
    let gaussians = Gaussians {
        gaussians: vec![Gaussian {
            rot: Quat::IDENTITY,
            pos: Vec3::ZERO + Vec3::Z,
            color: U8Vec4::new(255, 0, 0, 255),
            sh: [Vec3::ZERO; 15],
            scale: Vec3::splat(1.0),
        }],
    };

    let render_target = given::render_target_texture(&ctx);

    let mut viewer = Viewer::<G>::new_with(
        &ctx.device,
        wgpu::TextureFormat::Rgba8Unorm,
        None,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        &gaussians,
    )
    .expect("viewer");
}
