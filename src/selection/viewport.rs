use crate::{
    core::{self, ComputeBundle, ComputeBundleBuilder, GaussianPod},
    editor::SelectionBundle,
    shader,
};

/// The viewport selection bind group layout descriptor.
pub const VIEWPORT_BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
    wgpu::BindGroupLayoutDescriptor {
        label: Some("Viewport Selection Bind Group Layout"),
        entries: &[
            // Camera uniform buffer
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Viewport selection texture
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    };

/// Create a viewport selection operation.
///
/// - Bind group 0 is [`SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`].
/// - Bind group 1 is [`VIEWPORT_BIND_GROUP_LAYOUT_DESCRIPTOR`].
pub fn create_viewport_bundle<G: GaussianPod>(device: &wgpu::Device) -> ComputeBundle<()> {
    let mut resolver = wesl::PkgResolver::new();
    resolver.add_package(&core::shader::PACKAGE);
    resolver.add_package(&shader::PACKAGE);

    ComputeBundleBuilder::new()
        .label("Viewport Selection")
        .bind_groups([
            &SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR,
            &VIEWPORT_BIND_GROUP_LAYOUT_DESCRIPTOR,
        ])
        .main_shader(wesl::ModulePath::from_path(
            "wgpu_3dgs_viewer/selection/viewport",
        ))
        .entry_point("main")
        .compile_options(wesl::CompileOptions {
            features: G::wesl_features(),
            ..Default::default()
        })
        .resolver(resolver)
        .build_without_bind_groups(device)
        .map_err(|e| log::error!("{e}"))
        .expect("viewport selection compute bundle")
}
