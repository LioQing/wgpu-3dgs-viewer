pub struct TestContext {
    #[allow(dead_code)]
    pub instance: wgpu::Instance,
    #[allow(dead_code)]
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl TestContext {
    pub fn new() -> Self {
        Self::new_with_requirements(|adapter| (wgpu::Features::empty(), adapter.limits()))
    }

    #[cfg(all(feature = "lampshade-sort", not(target_arch = "wasm32")))]
    pub fn new_with_lampshade() -> Self {
        Self::new_with_requirements(|adapter| {
            let requirements = lampshade::KeyValueSoaSorter::requirements(adapter);
            (
                requirements.features(wgpu::Features::empty()),
                requirements.limits(adapter.limits()),
            )
        })
    }

    fn new_with_requirements(
        requirements: impl FnOnce(&wgpu::Adapter) -> (wgpu::Features, wgpu::Limits),
    ) -> Self {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .expect("adapter");

            let (required_features, required_limits) = requirements(&adapter);
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features,
                    required_limits,
                    ..Default::default()
                })
                .await
                .expect("device");

            Self {
                instance,
                adapter,
                device,
                queue,
            }
        })
    }
}
