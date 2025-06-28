use wesl::Wesl;

use crate::core;

pub fn compiler<'a>(
    features: impl IntoIterator<Item = (&'a str, bool)>,
) -> Wesl<wesl::StandardResolver> {
    let mut compiler = Wesl::new("src/shader");
    compiler.add_package(&core::wesl::Mod);
    compiler.set_features(features);
    compiler
}
