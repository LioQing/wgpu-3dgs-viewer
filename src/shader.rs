use wesl::PkgModule;

pub struct Mod;

impl PkgModule for Mod {
    fn name(&self) -> &'static str {
        "wgpu_3dgs_viewer"
    }

    fn source(&self) -> &'static str {
        ""
    }

    fn submodules(&self) -> &[&dyn PkgModule] {
        static SUBMODULES: &[&dyn PkgModule] =
            &[&camera::Mod, &preprocess::Mod, &render::Mod, &utils::Mod];
        SUBMODULES
    }

    fn submodule(&self, name: &str) -> Option<&dyn PkgModule> {
        match name {
            "camera" => Some(&camera::Mod),
            "preprocess" => Some(&preprocess::Mod),
            "render" => Some(&render::Mod),
            "utils" => Some(&utils::Mod),
            "selection" => Some(&selection::Mod),
            _ => selection::Mod.submodule(name),
        }
    }
}

macro_rules! submodule {
    ($name:ident $(, $dir:literal)? override $mod_name:ident) => {
        paste::paste! {
            pub mod $mod_name {
                pub struct Mod;

                impl wesl::PkgModule for Mod {
                    fn name(&self) -> &'static str {
                        stringify!($name)
                    }

                    fn source(&self) -> &'static str {
                        include_str!(concat!("shader/", $($dir,)? stringify!($name), ".wesl"))
                    }

                    fn submodules(&self) -> &[&dyn wesl::PkgModule] {
                        &[]
                    }

                    fn submodule(&self, _name: &str) -> Option<&dyn wesl::PkgModule> {
                        None
                    }
                }
            }
        }
    };
    ($name:ident $(, $dir:literal)?) => {
        submodule!($name $(, $dir)? override $name);
    };
}

submodule!(camera);
submodule!(preprocess);
submodule!(render);
submodule!(utils);

pub mod selection {
    use super::*;

    macro_rules! selection_submodule {
        ($name:ident) => {
            submodule!($name, "selection/");
        };
        ($name:ident override $mod_name:ident) => {
            submodule!($name, "selection/" override $mod_name);
        };
    }

    pub struct Mod;

    impl PkgModule for Mod {
        fn name(&self) -> &'static str {
            "selection"
        }

        fn source(&self) -> &'static str {
            ""
        }

        fn submodules(&self) -> &[&dyn PkgModule] {
            static SUBMODULES: &[&dyn PkgModule] =
                &[&viewport::Mod, &viewport_texture_rectangle::Mod];
            SUBMODULES
        }

        fn submodule(&self, name: &str) -> Option<&dyn PkgModule> {
            match name {
                "viewport" => Some(&viewport::Mod),
                "viewport_texture_rectangle" => Some(&viewport_texture_rectangle::Mod),
                _ => None,
            }
        }
    }

    selection_submodule!(viewport);
    selection_submodule!(viewport_texture_rectangle);
}
