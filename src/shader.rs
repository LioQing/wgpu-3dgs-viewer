//! Shader modules for the [`wesl::Pkg`] `wgpu-3dgs-viewer`.
//!
//! See the documentation of each module for details.

use wesl::{Pkg, PkgModule};

use crate::core;

/// The `wgpu-3dgs-viewer` [`wesl::Pkg`].
pub const PACKAGE: Pkg = Pkg {
    crate_name: "wgpu-3dgs-viewer",
    root: &MODULE,
    dependencies: &[&core::shader::PACKAGE],
};

/// The root module of the `wgpu-3dgs-viewer` package.
pub const MODULE: PkgModule = PkgModule {
    name: "wgpu_3dgs_viewer",
    source: "",
    submodules: &[
        &camera::MODULE,
        &preprocess::MODULE,
        &render::MODULE,
        &utils::MODULE,
        #[cfg(feature = "selection")]
        &selection::MODULE,
    ],
};

pub mod camera {
    use super::PkgModule;

    #[doc = concat!("```wgsl\n", include_str!("shader/camera.wesl"), "\n```")]
    pub const MODULE: PkgModule = PkgModule {
        name: "camera",
        source: include_str!("shader/camera.wesl"),
        submodules: &[],
    };
}

pub mod preprocess {
    use super::PkgModule;

    #[doc = concat!("```wgsl\n", include_str!("shader/preprocess.wesl"), "\n```")]
    pub const MODULE: PkgModule = PkgModule {
        name: "preprocess",
        source: include_str!("shader/preprocess.wesl"),
        submodules: &[],
    };
}

pub mod render {
    use super::PkgModule;

    #[doc = concat!("```wgsl\n", include_str!("shader/render.wesl"), "\n```")]
    pub const MODULE: PkgModule = PkgModule {
        name: "render",
        source: include_str!("shader/render.wesl"),
        submodules: &[],
    };
}

pub mod utils {
    use super::PkgModule;

    #[doc = concat!("```wgsl\n", include_str!("shader/utils.wesl"), "\n```")]
    pub const MODULE: PkgModule = PkgModule {
        name: "utils",
        source: include_str!("shader/utils.wesl"),
        submodules: &[],
    };
}

#[cfg(feature = "selection")]
pub mod selection {
    use super::PkgModule;

    /// The root module of the viewport selection shaders.
    pub const MODULE: PkgModule = PkgModule {
        name: "selection",
        source: "",
        submodules: &[
            &viewport::MODULE,
            &viewport_texture_rectangle::MODULE,
            &viewport_texture_brush::MODULE,
        ],
    };

    pub mod viewport {
        use super::PkgModule;

        #[doc = concat!("```wgsl\n", include_str!("shader/selection/viewport.wesl"), "\n```")]
        pub const MODULE: PkgModule = PkgModule {
            name: "viewport",
            source: include_str!("shader/selection/viewport.wesl"),
            submodules: &[],
        };
    }

    pub mod viewport_texture_rectangle {
        use super::PkgModule;

        #[doc = concat!("```wgsl\n", include_str!("shader/selection/viewport_texture_rectangle.wesl"), "\n```")]
        pub const MODULE: PkgModule = PkgModule {
            name: "viewport_texture_rectangle",
            source: include_str!("shader/selection/viewport_texture_rectangle.wesl"),
            submodules: &[],
        };
    }

    pub mod viewport_texture_brush {
        use super::PkgModule;

        #[doc = concat!("```wgsl\n", include_str!("shader/selection/viewport_texture_brush.wesl"), "\n```")]
        pub const MODULE: PkgModule = PkgModule {
            name: "viewport_texture_brush",
            source: include_str!("shader/selection/viewport_texture_brush.wesl"),
            submodules: &[],
        };
    }
}
