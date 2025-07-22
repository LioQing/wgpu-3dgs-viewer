use crate::{core, shader};

pub fn resolver() -> wesl::PkgResolver {
    let mut resolver = wesl::PkgResolver::new();
    resolver.add_package(&core::shader::Mod);
    resolver.add_package(&shader::Mod);
    resolver
}
