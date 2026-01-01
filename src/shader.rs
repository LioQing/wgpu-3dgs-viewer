//! Shader modules for the [`wesl::CodegenPkg`] `wgpu-3dgs-viewer`.
//!
//! See the documentation of each module for details.

use wesl::{CodegenModule, CodegenPkg};

use crate::core;

/// The `wgpu-3dgs-viewer` [`wesl::CodegenPkg`].
pub const PACKAGE: CodegenPkg = CodegenPkg {
    crate_name: "wgpu-3dgs-viewer",
    root: &MODULE,
    dependencies: &[&core::shader::PACKAGE],
};

/// The root module of the `wgpu-3dgs-viewer` package.
pub const MODULE: CodegenModule = CodegenModule {
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
    use super::CodegenModule;

    #[doc = concat!("```wgsl\n", include_str!("shader/camera.wesl"), "\n```")]
    pub const MODULE: CodegenModule = CodegenModule {
        name: "camera",
        source: include_str!("shader/camera.wesl"),
        submodules: &[],
    };
}

pub mod preprocess {
    use super::CodegenModule;

    #[doc = concat!("```wgsl\n", include_str!("shader/preprocess.wesl"), "\n```")]
    pub const MODULE: CodegenModule = CodegenModule {
        name: "preprocess",
        source: include_str!("shader/preprocess.wesl"),
        submodules: &[],
    };
}

pub mod render {
    use super::CodegenModule;

    #[doc = concat!("```wgsl\n", include_str!("shader/render.wesl"), "\n```")]
    pub const MODULE: CodegenModule = CodegenModule {
        name: "render",
        source: include_str!("shader/render.wesl"),
        submodules: &[],
    };
}

pub mod utils {
    use super::CodegenModule;

    #[doc = concat!("```wgsl\n", include_str!("shader/utils.wesl"), "\n```")]
    pub const MODULE: CodegenModule = CodegenModule {
        name: "utils",
        source: include_str!("shader/utils.wesl"),
        submodules: &[],
    };
}

#[cfg(feature = "selection")]
pub mod selection {
    use super::CodegenModule;

    /// The root module of the viewport selection shaders.
    pub const MODULE: CodegenModule = CodegenModule {
        name: "selection",
        source: "",
        submodules: &[
            &viewport::MODULE,
            &viewport_texture_rectangle::MODULE,
            &viewport_texture_brush::MODULE,
        ],
    };

    pub mod viewport {
        use super::CodegenModule;

        #[doc = concat!("```wgsl\n", include_str!("shader/selection/viewport.wesl"), "\n```")]
        pub const MODULE: CodegenModule = CodegenModule {
            name: "viewport",
            source: include_str!("shader/selection/viewport.wesl"),
            submodules: &[],
        };
    }

    pub mod viewport_texture_rectangle {
        use super::CodegenModule;

        #[doc = concat!("```wgsl\n", include_str!("shader/selection/viewport_texture_rectangle.wesl"), "\n```")]
        pub const MODULE: CodegenModule = CodegenModule {
            name: "viewport_texture_rectangle",
            source: include_str!("shader/selection/viewport_texture_rectangle.wesl"),
            submodules: &[],
        };
    }

    pub mod viewport_texture_brush {
        use super::CodegenModule;

        #[doc = concat!("```wgsl\n", include_str!("shader/selection/viewport_texture_brush.wesl"), "\n```")]
        pub const MODULE: CodegenModule = CodegenModule {
            name: "viewport_texture_brush",
            source: include_str!("shader/selection/viewport_texture_brush.wesl"),
            submodules: &[],
        };
    }
}
