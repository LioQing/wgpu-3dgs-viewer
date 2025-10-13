#[macro_export]
macro_rules! inline_wesl_pkg {
    (use $deps:expr, $name:literal => $module_name:literal: $($body:tt)+) => {
        wesl::Pkg {
            crate_name: $name,
            root: &wesl::PkgModule {
                name: $module_name,
                source: {
                    stringify!($($body)+)
                },
                submodules: &[],
            },
            dependencies: &$deps,
        }
    };
    (use $deps:expr, $name:literal: $($body:tt)+) => {
        inline_wesl_pkg!(use $deps, $name => $name: $($body)+)
    };
    ($name:literal: $($body:tt)+) => {
        inline_wesl_pkg!(use [], $name => $name: $($body)+)
    };
}
