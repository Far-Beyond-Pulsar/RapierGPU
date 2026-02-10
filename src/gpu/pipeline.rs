//! Compute pipeline abstraction for GPU operations.

use wgpu;

/// Wrapper for WGPU compute pipelines.
///
/// Provides a simplified interface for creating and executing compute shaders.
pub struct GpuComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuComputePipeline {
    /// Creates a new compute pipeline from WGSL shader source.
    ///
    /// # Arguments
    ///
    /// * `device` - WGPU device
    /// * `shader_source` - WGSL shader source code
    /// * `entry_point` - Entry point function name (usually "main")
    /// * `bind_group_layout_entries` - Buffer binding descriptions
    pub fn new(
        device: &wgpu::Device,
        shader_source: &str,
        entry_point: &str,
        bind_group_layout_entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: bind_group_layout_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: entry_point,
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Creates a bind group for this pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - WGPU device
    /// * `buffers` - Buffer bindings (must match bind_group_layout order)
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &self.bind_group_layout,
            entries: &entries,
        })
    }

    /// Dispatches the compute shader.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder
    /// * `bind_group` - Bind group with buffer bindings
    /// * `workgroups` - Number of workgroups in (x, y, z)
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }
}

/// Helper to create a standard storage buffer binding layout entry.
pub fn storage_buffer_binding(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Helper to create a uniform buffer binding layout entry.
pub fn uniform_buffer_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_buffer_binding() {
        let entry = storage_buffer_binding(0, true);
        assert_eq!(entry.binding, 0);
        assert_eq!(entry.visibility, wgpu::ShaderStages::COMPUTE);
    }

    #[test]
    fn test_uniform_buffer_binding() {
        let entry = uniform_buffer_binding(1);
        assert_eq!(entry.binding, 1);
        assert_eq!(entry.visibility, wgpu::ShaderStages::COMPUTE);
    }
}
