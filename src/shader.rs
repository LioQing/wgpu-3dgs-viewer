pub struct Mod;

impl wesl::PkgModule for Mod {
    fn name(&self) -> &'static str {
        "wgpu_3dgs_viewer"
    }

    fn source(&self) -> &'static str {
        ""
    }

    fn submodules(&self) -> &[&dyn wesl::PkgModule] {
        static SUBMODULES: &[&dyn wesl::PkgModule] =
            &[&camera::Mod, &preprocess::Mod, &render::Mod, &utils::Mod];
        SUBMODULES
    }

    fn submodule(&self, name: &str) -> Option<&dyn wesl::PkgModule> {
        match name {
            "camera" => Some(&camera::Mod),
            "preprocess" => Some(&preprocess::Mod),
            "render" => Some(&render::Mod),
            "utils" => Some(&utils::Mod),
            _ => None,
        }
    }
}

macro_rules! submodule {
    ($name:ident) => {
        paste::paste! {
            pub mod $name {
                pub struct Mod;

                impl wesl::PkgModule for Mod {
                    fn name(&self) -> &'static str {
                        stringify!($name)
                    }

                    fn source(&self) -> &'static str {
                        include_str!(concat!("shader/", stringify!($name), ".wesl"))
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
}

submodule!(camera);
submodule!(preprocess);
submodule!(render);
submodule!(utils);
