use wesl::Wesl;

use crate::core;

pub fn compiler<'a>(
    features: impl IntoIterator<Item = (&'a str, bool)>,
) -> Wesl<wesl::StandardResolver> {
    let mut compiler = Wesl::new("src/shader");
    compiler.add_package(&core::shader::Mod);
    compiler.set_features(features);
    compiler
}

pub fn resolver() -> wesl::StandardResolver {
    let mut resolver = wesl::StandardResolver::new("src/shader");
    resolver.add_package(&core::shader::Mod);
    resolver
}
