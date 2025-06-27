use wesl::*;

pub fn compiler<'a>(features: impl IntoIterator<Item = (&'a str, bool)>) -> Wesl<StandardResolver> {
    let mut compiler = Wesl::new("src/shader");
    compiler.set_features(features);
    compiler
}
