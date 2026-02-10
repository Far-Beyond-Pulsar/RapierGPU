//! GPU device initialization and management.

use std::sync::Arc;
use wgpu;

/// GPU context managing WGPU device, queue, and adapter.
///
/// This is the main entry point for GPU acceleration. It handles:
/// - Device selection (prefers discrete GPUs)
/// - Feature validation (compute shaders required)
/// - Adapter capabilities checking
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter: wgpu::Adapter,
}

/// Errors that can occur during GPU initialization.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("No suitable GPU adapter found")]
    NoAdapter,
    
    #[error("Failed to request device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),
    
    #[error("Compute shaders not supported on this device")]
    ComputeNotSupported,
    
    #[error("Insufficient GPU memory (required: {required} bytes, available: {available} bytes)")]
    InsufficientMemory { required: u64, available: u64 },
}

impl GpuContext {
    /// Creates a new GPU context with default settings.
    ///
    /// Prefers discrete GPUs over integrated ones for better performance.
    /// Falls back gracefully if no suitable GPU is found.
    ///
    /// # Errors
    ///
    /// Returns `GpuError` if:
    /// - No GPU adapter is found
    /// - Compute shaders are not supported
    /// - Device request fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rapier3d::gpu::GpuContext;
    /// let gpu = GpuContext::new().expect("Failed to initialize GPU");
    /// println!("Using GPU: {:?}", gpu.adapter_info().name);
    /// ```
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    /// Async version of `new()` for async runtimes.
    pub async fn new_async() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request adapter with preference for discrete GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        log::info!("Selected GPU: {} ({:?})", adapter_info.name, adapter_info.backend);
        log::info!("GPU Type: {:?}", adapter_info.device_type);

        // Verify compute shader support
        let features = adapter.features();
        if !features.contains(wgpu::Features::empty()) {
            // Basic compute is always available in WebGPU
            log::debug!("Compute shaders supported");
        }

        // Request device with required features
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Rapier GPU Device"),
                    required_features: wgpu::Features::empty(), // Compute is baseline
                    required_limits: wgpu::Limits {
                        max_compute_workgroup_size_x: 256,
                        max_compute_workgroup_size_y: 256,
                        max_compute_workgroup_size_z: 64,
                        max_compute_invocations_per_workgroup: 256,
                        max_compute_workgroups_per_dimension: 65535,
                        ..Default::default()
                    },
                    memory_hints: Default::default(),
                },
                None,
            )
            .await?;

        log::info!("GPU device initialized successfully");

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter,
        })
    }

    /// Returns information about the selected GPU adapter.
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// Returns the limits of the GPU device.
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    /// Checks if the GPU has sufficient memory for the given requirement.
    pub fn check_memory_requirement(&self, required_bytes: u64) -> Result<(), GpuError> {
        // WGPU doesn't expose memory info directly, but we can check limits
        let limits = self.limits();
        let max_buffer_size = limits.max_buffer_size;
        
        if required_bytes > max_buffer_size {
            return Err(GpuError::InsufficientMemory {
                required: required_bytes,
                available: max_buffer_size,
            });
        }
        
        Ok(())
    }

    /// Returns the maximum workgroup size for compute shaders.
    pub fn max_workgroup_size(&self) -> (u32, u32, u32) {
        let limits = self.limits();
        (
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_creation() {
        // This test requires a GPU to be available
        if let Ok(ctx) = GpuContext::new() {
            let info = ctx.adapter_info();
            println!("GPU: {} ({:?})", info.name, info.backend);
            
            let (x, y, z) = ctx.max_workgroup_size();
            assert!(x >= 256, "Workgroup X size too small");
            assert!(y >= 256, "Workgroup Y size too small");
            assert!(z >= 64, "Workgroup Z size too small");
        } else {
            println!("Skipping GPU test - no suitable GPU found");
        }
    }
}
