use std::sync::atomic::{AtomicUsize, Ordering};

use glam::*;
use wgpu_3dgs_core::GaussianMaxStdDev;
use wgpu_3dgs_viewer::{
    CameraPod, DepthSortIndirectArgsBuffer, DepthSorter, Viewer,
    ViewerCreateDepthSorterFactoryContext, ViewerCreateError, ViewerCreateOptions,
    core::{
        BufferWrapper, CovSingle, Gaussian, GaussianDisplayMode, GaussianPod, GaussianShDegree,
        GaussianTransformPod, Gaussians, GaussiansBuffer, ModelTransformPod, PackedGaussian,
        ShDegree0, ShDegree1, ShDegree2, ShDegree4, ShHalf, ShNone, ShNorm8, ShSingle,
    },
};

use crate::common::{TestContext, assert_render_target, given};

type G = PackedGaussian<ShSingle, CovSingle>;

fn gaussian() -> Gaussian {
    Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::Z,
        color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::ONE,
    }
}

#[test]
fn test_viewer_when_runtime_storage_degree_mismatches_should_return_error() {
    let ctx = TestContext::new();

    for count in [0, 1] {
        let gaussians = Gaussians::from(vec![gaussian().convert_sh_degree::<ShDegree0>(); count]);

        let error = Viewer::<PackedGaussian<ShSingle, CovSingle>>::new(
            &ctx.device,
            wgpu::TextureFormat::Rgba8Unorm,
            &gaussians,
        )
        .expect_err("SH degree mismatch");
        assert!(matches!(
            error,
            ViewerCreateError::ShDegreeMismatch(error)
                if error.actual_degree == 0 && error.expected_degree == 3
        ));

        let error = Viewer::<PackedGaussian<ShSingle, CovSingle>>::new_with_options(
            &ctx.device,
            wgpu::TextureFormat::Rgba8Unorm,
            &gaussians,
            ViewerCreateOptions::default(),
        )
        .expect_err("SH degree mismatch");
        assert!(matches!(
            error,
            ViewerCreateError::ShDegreeMismatch(error)
                if error.actual_degree == 0 && error.expected_degree == 3
        ));
    }
}

fn render_matching_layout<G: GaussianPod>(ctx: &TestContext) {
    let gaussians = vec![gaussian().convert_sh_degree::<G::ShDegree>()];
    let mut viewer =
        Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians).expect("viewer");
    viewer.update_camera_with_pod(&ctx.queue, &given::camera_pod());

    let render_target = given::render_target_texture(ctx);
    let render_target_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    viewer.render(&mut encoder, &render_target_view);
    ctx.queue.submit(Some(encoder.finish()));
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");

    assert_render_target(ctx, &render_target_view, |pixels| {
        let sum = pixels.iter().sum::<UVec4>();
        assert!(sum.x > 1);
        assert_eq!(sum.y, 0);
        assert_eq!(sum.z, 0);
    });
}

#[test]
fn test_viewer_when_storage_degree_matches_should_render_packed_layouts() {
    let ctx = TestContext::new();

    render_matching_layout::<PackedGaussian<ShNone, CovSingle>>(&ctx);
    render_matching_layout::<PackedGaussian<ShSingle<ShDegree0>, CovSingle>>(&ctx);
    render_matching_layout::<PackedGaussian<ShSingle<ShDegree1>, CovSingle>>(&ctx);
    render_matching_layout::<PackedGaussian<ShHalf<ShDegree2>, CovSingle>>(&ctx);
    render_matching_layout::<PackedGaussian<ShNorm8, CovSingle>>(&ctx);
    render_matching_layout::<PackedGaussian<ShSingle<ShDegree4>, CovSingle>>(&ctx);
}

#[test]
fn test_viewer_when_fourth_sh_band_is_nonzero_should_render_it_only_at_degree_four() {
    let ctx = TestContext::new();
    let mut gaussian = gaussian().convert_sh_degree::<ShDegree4>();
    // Index 19 is the degree-4, m=0 coefficient (SH0 is stored in color).
    // Its basis is positive along the Z axis, so it adds green to the red fixture.
    gaussian.sh[19] = Vec3::Y;
    let gaussians = vec![gaussian];
    let mut viewer = Viewer::<PackedGaussian<ShSingle<ShDegree4>, CovSingle>>::new(
        &ctx.device,
        wgpu::TextureFormat::Rgba8Unorm,
        &gaussians,
    )
    .expect("viewer");
    viewer.update_camera_with_pod(&ctx.queue, &given::camera_pod());

    let render_target = given::render_target_texture(&ctx);
    let render_target_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());
    for degree in [3, 4] {
        viewer.update_gaussian_transform(
            &ctx.queue,
            1.0,
            GaussianDisplayMode::Splat,
            GaussianShDegree::new(degree).expect("sh deg"),
            false,
            GaussianMaxStdDev::new(3.0).expect("max std dev"),
        );

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        viewer.render(&mut encoder, &render_target_view);
        ctx.queue.submit(Some(encoder.finish()));
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");

        assert_render_target(&ctx, &render_target_view, |pixels| {
            let sum = pixels.iter().sum::<UVec4>();
            assert!(sum.x > 1, "the base red color must remain visible");
            assert_eq!(sum.z, 0);
            if degree == 3 {
                assert_eq!(sum.y, 0, "degree 3 must ignore the fourth SH band");
            } else {
                assert!(sum.y > 1, "degree 4 must evaluate the fourth SH band");
            }
        });
    }
}

#[test]
fn test_viewer_when_runtime_storage_degree_matches_should_create_viewer() {
    let ctx = TestContext::new();
    let gaussians = Gaussians::from(vec![gaussian()]);
    let viewer = Viewer::<PackedGaussian<ShSingle, CovSingle>>::new(
        &ctx.device,
        wgpu::TextureFormat::Rgba8Unorm,
        &gaussians,
    )
    .expect("viewer");

    assert_eq!(viewer.gaussians_buffer.len(), 1);
}

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
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");

    assert_render_target(ctx, &render_target_view, assertion);
}

#[test]
fn test_viewer_update_camera_when_with_or_without_pod_should_be_equal() {
    let ctx = TestContext::new();
    let gaussians: Vec<Gaussian> = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let render_target1 = given::render_target_texture(&ctx);
    let render_target2 = given::render_target_texture(&ctx);
    let size = UVec2::new(render_target1.size().width, render_target1.size().height);
    let camera = given::camera();
    let camera_pod = CameraPod::new(&camera, size);

    let mut viewer1 =
        Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians).expect("viewer");
    let mut viewer2 =
        Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians).expect("viewer");

    viewer1.update_camera_with_pod(&ctx.queue, &camera_pod);
    viewer2.update_camera(&ctx.queue, &camera, size);

    render_and_assert(&ctx, &viewer1, &render_target1, |pixels1: &[UVec4]| {
        render_and_assert(&ctx, &viewer2, &render_target2, |pixels2: &[UVec4]| {
            assert_eq!(pixels1, pixels2);
        });
    });
}

#[test]
fn test_viewer_render_should_render_correctly() {
    let ctx = TestContext::new();
    let gaussians: Vec<Gaussian> = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let render_target = given::render_target_texture(&ctx);

    let mut viewer =
        Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians).expect("viewer");

    viewer.update_camera_with_pod(&ctx.queue, &given::camera_pod());

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
    let gaussians: Vec<Gaussian> = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let render_target = given::render_target_texture(&ctx);

    let mut viewer =
        Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians).expect("viewer");

    viewer.update_camera_with_pod(&ctx.queue, &given::camera_pod());
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
            GaussianShDegree::new(3).expect("sh deg"),
            true,
            GaussianMaxStdDev::new(3.0).expect("max std dev"),
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
                GaussianShDegree::new(3).expect("sh deg"),
                true,
                GaussianMaxStdDev::new(3.0).expect("max std dev"),
            ),
        );
    });
}

fn test_viewer_when_model_pos_is_behind_camera_should_not_render_gaussian(
    update_model_transform: impl FnOnce(&mut Viewer<G>, &wgpu::Queue),
) {
    let ctx = TestContext::new();
    let gaussians: Vec<Gaussian> = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let render_target = given::render_target_texture(&ctx);

    let mut viewer =
        Viewer::<G>::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm, &gaussians).expect("viewer");

    viewer.update_camera_with_pod(&ctx.queue, &given::camera_pod());
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

#[test]
fn test_viewer_uses_custom_depth_sorter_factory() {
    #[derive(Debug)]
    struct CountingSorter {
        sort_calls: AtomicUsize,
    }

    impl CountingSorter {
        fn new(_ctx: ViewerCreateDepthSorterFactoryContext<G>) -> Self {
            Self {
                sort_calls: AtomicUsize::new(0),
            }
        }
    }

    impl DepthSorter for CountingSorter {
        fn sort(
            &self,
            _encoder: &mut wgpu::CommandEncoder,
            _indirect_args_buffer: &DepthSortIndirectArgsBuffer,
        ) {
            self.sort_calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    let ctx = TestContext::new();
    let gaussians: Vec<Gaussian> = vec![Gaussian {
        rot: Quat::IDENTITY,
        pos: Vec3::ZERO + Vec3::Z,
        color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        sh: [Vec3::ZERO; 15],
        scale: Vec3::splat(1.0),
    }];

    let viewer = Viewer::<G, CountingSorter>::new_with_options(
        &ctx.device,
        wgpu::TextureFormat::Rgba8Unorm,
        &gaussians,
        ViewerCreateOptions {
            depth_stencil: None,
            gaussians_buffer_usage: GaussiansBuffer::<G>::DEFAULT_USAGES,
            color_write_mask: wgpu::ColorWrites::ALL,
            cache: None,
            depth_sorter_factory: CountingSorter::new,
            phantom_data: Default::default(),
        },
    )
    .expect("viewer");

    assert_eq!(viewer.depth_sorter.sort_calls.load(Ordering::SeqCst), 0);

    let render_target = given::render_target_texture(&ctx);
    let render_target_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

    viewer.render(&mut encoder, &render_target_view);

    ctx.queue.submit(Some(encoder.finish()));
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");

    assert_eq!(viewer.depth_sorter.sort_calls.load(Ordering::SeqCst), 1);
}
