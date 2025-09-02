use wesl::{Pkg, PkgModule};

use crate::core;

pub const PACKAGE: Pkg = Pkg {
    crate_name: "wgpu-3dgs-viewer",
    root: &MODULE,
    dependencies: &[&core::shader::PACKAGE],
};

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

    pub const MODULE: PkgModule = PkgModule {
        name: "camera",
        source: include_str!("shader/camera.wesl"),
        submodules: &[],
    };
}

pub mod preprocess {
    use super::PkgModule;

    pub const MODULE: PkgModule = PkgModule {
        name: "preprocess",
        source: include_str!("shader/preprocess.wesl"),
        submodules: &[],
    };
}

pub mod render {
    use super::PkgModule;

    pub const MODULE: PkgModule = PkgModule {
        name: "render",
        source: include_str!("shader/render.wesl"),
        submodules: &[],
    };
}

pub mod utils {
    use super::PkgModule;

    pub const MODULE: PkgModule = PkgModule {
        name: "utils",
        source: include_str!("shader/utils.wesl"),
        submodules: &[],
    };
}

#[cfg(feature = "selection")]
pub mod selection {
    use super::PkgModule;

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

        pub const MODULE: PkgModule = PkgModule {
            name: "viewport",
            source: include_str!("shader/selection/viewport.wesl"),
            submodules: &[],
        };
    }

    pub mod viewport_texture_rectangle {
        use super::PkgModule;

        pub const MODULE: PkgModule = PkgModule {
            name: "viewport_texture_rectangle",
            source: include_str!("shader/selection/viewport_texture_rectangle.wesl"),
            submodules: &[],
        };
    }

    pub mod viewport_texture_brush {
        use super::PkgModule;

        pub const MODULE: PkgModule = PkgModule {
            name: "viewport_texture_brush",
            source: include_str!("shader/selection/viewport_texture_brush.wesl"),
            submodules: &[],
        };
    }
}
