use crate::{core, shader};

/// Get the WESL package resolver for this crate.
///
/// This resolver includes the [`wgpu-3dgs-viewer`](shader) and [`wgpu-3dgs-core`](core::shader)
/// packages.
pub fn resolver() -> wesl::PkgResolver {
    let mut resolver = wesl::PkgResolver::new();
    resolver.add_package(&core::shader::PACKAGE);
    resolver.add_package(&shader::PACKAGE);
    resolver
}
