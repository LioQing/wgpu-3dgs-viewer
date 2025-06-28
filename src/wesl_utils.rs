use wesl::Wesl;

use crate::core;

pub fn compiler<'a>(
    features: impl IntoIterator<Item = (&'a str, bool)>,
) -> Wesl<core::shader::Resolver> {
    let resolver = core::shader::Resolver::new("src/shader").with_package(&core::shader::Mod);
    let mut compiler = Wesl::new("src/shader").set_custom_resolver(resolver);
    compiler.set_features(features);
    compiler
}
