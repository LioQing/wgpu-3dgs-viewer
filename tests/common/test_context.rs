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
        pollster::block_on(async {
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .expect("adapter");

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_limits: adapter.limits(),
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

    pub fn new_with_subgroups() -> Self {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .expect("adapter");
            let required_features = adapter.features() & wgpu::Features::SUBGROUP;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("Subgroup Test Device"),
                    required_features,
                    required_limits: adapter.limits(),
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
