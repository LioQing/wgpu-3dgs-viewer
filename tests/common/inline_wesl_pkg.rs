#[macro_export]
macro_rules! inline_wesl_pkg {
    (
        $(use { $($deps:expr),+ $(,)? };)?

        crate $crate_name:ident;

        mod $module_name:ident { $($body:tt)+ }
    ) => {
        wesl::Pkg {
            crate_name: stringify!($crate_name),
            root: &wesl::PkgModule {
                name: stringify!($module_name),
                source: {
                    stringify!($($body)+)
                },
                submodules: &[],
            },
            dependencies: &[$($(&$deps),+)?],
        }
    };
    (
        $(use $($deps:expr),+ $(,)?;)?

        crate $crate_name:ident;

        mod $module_name:ident { $($body:tt)+ }
    ) => {
        inline_wesl_pkg! {
            $(use { $($deps),+ };)?

            crate $crate_name;

            mod $module_name { $($body)+ }
        }
    };
    (
        $(use { $($deps:expr),+ $(,)? };)?

        mod $module_name:ident { $($body:tt)+ }
    ) => {
        inline_wesl_pkg! {
            $(use { $($deps),+ };)?

            crate $module_name;

            mod $module_name { $($body)+ }
        }
    };
    (
        $(use $($deps:expr),+ $(,)?;)?

        mod $module_name:ident { $($body:tt)+ }
    ) => {
        inline_wesl_pkg! {
            $(use { $($deps),+ };)?

            crate $module_name;

            mod $module_name { $($body)+ }
        }
    };
}
