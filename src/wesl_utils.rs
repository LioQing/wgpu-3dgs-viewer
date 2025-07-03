use wesl::Wesl;

use crate::core;

pub fn compiler<'a>(
    features: impl IntoIterator<Item = (&'a str, bool)>,
) -> Wesl<core::wesl::VarStandardResolver> {
    let resolver =
        core::wesl::VarStandardResolver::new("src/shader").with_package(&core::shader::Mod);
    let mut compiler = Wesl::new("src/shader").set_custom_resolver(resolver);
    compiler.set_features(features);
    compiler
}
