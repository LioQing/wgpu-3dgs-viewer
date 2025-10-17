use glam::*;
use wgpu_3dgs_viewer::{
    Viewer,
    core::{
        Gaussian, GaussianDisplayMode, GaussianPodWithShSingleCov3dSingleConfigs, GaussianShDegree,
        GaussianTransformPod, Gaussians, ModelTransformPod,
    },
};

use crate::common::{TestContext, assert_render_target, given};

type G = GaussianPodWithShSingleCov3dSingleConfigs;

fn render_and_assert(
    ctx: &TestContext,
    viewer: &Viewer<G>,
    render_target: &wgpu::Texture,
    assertion: impl Fn(&[UVec4]),
) {
    let render_target_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

    viewer.render(&mut encoder, &render_target_view);

    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::PollType::Wait).expect("device poll");

    assert_render_target(ctx, &render_target_view, assertion);
}

#[test]
fn test_viewer_render_should_render_correctly() {
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

    let mut viewer =
        Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians).expect("viewer");

    viewer.update_camera_with_pod(&ctx.queue, &given::camera());

    render_and_assert(&ctx, &viewer, &render_target, |pixels: &[UVec4]| {
        let sum = pixels.iter().sum::<UVec4>();
        assert!(sum.x > 1);
        assert!(sum.y < 1);
        assert!(sum.z < 1);
        assert!(sum.w > 1);
    });
}

fn test_viewer_when_no_sh0_is_set_should_and_render_as_grayscale(
    update_gaussian_transform: impl FnOnce(&mut Viewer<G>, &wgpu::Queue),
) {
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

    let mut viewer =
        Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians).expect("viewer");

    viewer.update_camera_with_pod(&ctx.queue, &given::camera());
    update_gaussian_transform(&mut viewer, &ctx.queue);

    render_and_assert(&ctx, &viewer, &render_target, |pixels: &[UVec4]| {
        let sum = pixels.iter().sum::<UVec4>();
        assert!(sum.x > 1);
        assert!(sum.y > 1);
        assert!(sum.z > 1);
        assert!(sum.w > 1);
    });
}

#[test]
fn test_viewer_update_gaussian_transform_when_no_sh0_is_set_should_render_as_grayscale() {
    test_viewer_when_no_sh0_is_set_should_and_render_as_grayscale(|viewer, queue| {
        viewer.update_gaussian_transform(
            queue,
            1.0,
            GaussianDisplayMode::Splat,
            GaussianShDegree::new_unchecked(3),
            true,
            3.0,
        );
    });
}

#[test]
fn test_viewer_update_gaussian_transform_with_pod_when_no_sh0_is_set_should_render_as_grayscale() {
    test_viewer_when_no_sh0_is_set_should_and_render_as_grayscale(|viewer, queue| {
        viewer.update_gaussian_transform_with_pod(
            queue,
            &GaussianTransformPod::new(
                1.0,
                GaussianDisplayMode::Splat,
                GaussianShDegree::new_unchecked(3),
                true,
                3.0,
            ),
        );
    });
}

fn test_viewer_when_model_pos_is_behind_camera_should_not_render_gaussian(
    update_model_transform: impl FnOnce(&mut Viewer<G>, &wgpu::Queue),
) {
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

    let mut viewer =
        Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians).expect("viewer");

    viewer.update_camera_with_pod(&ctx.queue, &given::camera());
    update_model_transform(&mut viewer, &ctx.queue);

    render_and_assert(&ctx, &viewer, &render_target, |pixels: &[UVec4]| {
        let sum = pixels.iter().sum::<UVec4>();
        assert_eq!(sum.x, 0);
        assert_eq!(sum.y, 0);
        assert_eq!(sum.z, 0);
    });
}

#[test]
fn test_viewer_update_model_transform_when_model_pos_is_behind_camera_should_not_render_gaussian() {
    test_viewer_when_model_pos_is_behind_camera_should_not_render_gaussian(|viewer, queue| {
        viewer.update_model_transform(queue, Vec3::ZERO - Vec3::Z, Quat::IDENTITY, Vec3::ONE);
    });
}

#[test]
fn test_viewer_update_model_transform_with_pod_when_model_pos_is_behind_camera_should_not_render_gaussian()
 {
    test_viewer_when_model_pos_is_behind_camera_should_not_render_gaussian(|viewer, queue| {
        viewer.update_model_transform_with_pod(
            queue,
            &ModelTransformPod::new(Vec3::ZERO - Vec3::Z, Quat::IDENTITY, Vec3::ONE),
        );
    });
}
