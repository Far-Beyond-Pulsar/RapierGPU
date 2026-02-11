//! GPU buffer management for physics data.

use crate::dynamics::RigidBodySet;
use crate::math::Real;

#[cfg(feature = "dim3")]
use crate::glamx::{Vec3, Quat};
#[cfg(feature = "dim2")]
use crate::glamx::Vec2;

use wgpu;
use bytemuck::{Pod, Zeroable};

/// GPU-friendly representation of rigid body data using Structure-of-Arrays layout.
///
/// This layout optimizes for coalesced GPU memory access by grouping similar
/// data types together rather than per-body structures.
pub struct RigidBodyGpuBuffer {
    // Position data
    pub positions_buffer: wgpu::Buffer,
    pub rotations_buffer: wgpu::Buffer,
    
    // Velocity data
    pub lin_velocities_buffer: wgpu::Buffer,
    pub ang_velocities_buffer: wgpu::Buffer,
    
    // Force accumulation
    pub forces_buffer: wgpu::Buffer,
    pub torques_buffer: wgpu::Buffer,
    
    // Mass properties
    pub inv_masses_buffer: wgpu::Buffer,
    pub inv_inertias_buffer: wgpu::Buffer,
    
    // Metadata
    pub body_count: usize,
    pub capacity: usize,
}

/// Position data for GPU (3D vector).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuVector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _padding: f32, // Align to 16 bytes
}

/// Rotation data for GPU (quaternion in 3D, angle in 2D).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuRotation {
    #[cfg(feature = "dim3")]
    pub x: f32,
    #[cfg(feature = "dim3")]
    pub y: f32,
    #[cfg(feature = "dim3")]
    pub z: f32,
    #[cfg(feature = "dim3")]
    pub w: f32,
    
    #[cfg(feature = "dim2")]
    pub angle: f32,
    #[cfg(feature = "dim2")]
    pub _padding: [f32; 3],
}

/// 3x3 matrix for inertia tensors (stored as row-major).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuMatrix3 {
    pub data: [f32; 12], // 3x3 matrix + padding to 4x3
}

/// All four mutable body-state arrays downloaded from the GPU in one pass.
pub struct GpuFullState {
    pub positions: Vec<GpuVector3>,
    pub rotations: Vec<GpuRotation>,
    pub lin_velocities: Vec<GpuVector3>,
    pub ang_velocities: Vec<GpuVector3>,
}

/// Manages GPU buffer lifecycle and CPU↔GPU transfers.
pub struct BufferManager {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
}

impl BufferManager {
    /// Creates a new buffer manager.
    pub fn new(device: std::sync::Arc<wgpu::Device>, queue: std::sync::Arc<wgpu::Queue>) -> Self {
        Self { device, queue }
    }

    /// Creates GPU buffers for rigid body data.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of bodies to allocate space for
    pub fn create_rigid_body_buffer(&self, capacity: usize) -> RigidBodyGpuBuffer {
        let create_buffer = |label: &str, size: usize| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::STORAGE 
                    | wgpu::BufferUsages::COPY_DST 
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let vec3_size = std::mem::size_of::<GpuVector3>() * capacity;
        let rot_size = std::mem::size_of::<GpuRotation>() * capacity;
        let scalar_size = std::mem::size_of::<f32>() * capacity;
        let mat3_size = std::mem::size_of::<GpuMatrix3>() * capacity;

        RigidBodyGpuBuffer {
            positions_buffer: create_buffer("RigidBody Positions", vec3_size),
            rotations_buffer: create_buffer("RigidBody Rotations", rot_size),
            lin_velocities_buffer: create_buffer("RigidBody Linear Velocities", vec3_size),
            ang_velocities_buffer: create_buffer("RigidBody Angular Velocities", vec3_size),
            forces_buffer: create_buffer("RigidBody Forces", vec3_size),
            torques_buffer: create_buffer("RigidBody Torques", vec3_size),
            inv_masses_buffer: create_buffer("RigidBody Inverse Masses", scalar_size),
            inv_inertias_buffer: create_buffer("RigidBody Inverse Inertias", mat3_size),
            body_count: 0,
            capacity,
        }
    }

    /// Uploads rigid body data from CPU to GPU.
    ///
    /// # Arguments
    ///
    /// * `bodies` - The rigid body set to upload
    /// * `gpu_buffer` - The GPU buffer to write to
    pub fn upload_rigid_bodies(
        &self,
        bodies: &RigidBodySet,
        gpu_buffer: &mut RigidBodyGpuBuffer,
    ) {
        let body_count = bodies.len();
        
        if body_count > gpu_buffer.capacity {
            log::warn!(
                "Body count ({}) exceeds GPU buffer capacity ({}). Truncating.",
                body_count,
                gpu_buffer.capacity
            );
        }

        let count = body_count.min(gpu_buffer.capacity);
        gpu_buffer.body_count = count;

        // Prepare CPU-side data in SoA layout
        let mut positions = Vec::with_capacity(count);
        let mut rotations = Vec::with_capacity(count);
        let mut lin_vels = Vec::with_capacity(count);
        let mut ang_vels = Vec::with_capacity(count);
        let mut forces = Vec::with_capacity(count);
        let mut torques = Vec::with_capacity(count);
        let mut inv_masses = Vec::with_capacity(count);
        let mut inv_inertias = Vec::with_capacity(count);

        for (_handle, body) in bodies.iter().take(count) {
            let pos = &body.position().translation;
            #[cfg(feature = "dim3")]
            positions.push(Self::vector_to_gpu_from_vec3(pos));
            #[cfg(feature = "dim2")]
            positions.push(Self::vector_to_gpu_from_vec2(pos));
            
            #[cfg(feature = "dim3")]
            rotations.push(Self::rotation_to_gpu(&body.position().rotation));
            #[cfg(feature = "dim2")]
            rotations.push(Self::rotation_to_gpu(&body.position().rotation));
            
            let vel = body.linvel();
            #[cfg(feature = "dim3")]
            lin_vels.push(Self::vector_to_gpu_from_vec3(&vel));
            #[cfg(feature = "dim2")]
            lin_vels.push(Self::vector_to_gpu_from_vec2(&vel));
            
            #[cfg(feature = "dim3")]
            {
                let angvel = body.angvel();
                ang_vels.push(GpuVector3 {
                    x: angvel.x,
                    y: angvel.y,
                    z: angvel.z,
                    _padding: 0.0,
                });
            }
            #[cfg(feature = "dim2")]
            {
                let angvel = body.angvel();
                ang_vels.push(GpuVector3 {
                    x: 0.0,
                    y: 0.0,
                    z: angvel,
                    _padding: 0.0,
                });
            }
            
            // User forces
            let force = body.user_force();
            #[cfg(feature = "dim3")]
            forces.push(Self::vector_to_gpu_from_vec3(&force));
            #[cfg(feature = "dim2")]
            forces.push(Self::vector_to_gpu_from_vec2(&force));
            
            #[cfg(feature = "dim3")]
            {
                let torque = body.user_torque();
                torques.push(GpuVector3 {
                    x: torque.x,
                    y: torque.y,
                    z: torque.z,
                    _padding: 0.0,
                });
            }
            #[cfg(feature = "dim2")]
            {
                let torque = body.user_torque();
                torques.push(GpuVector3 {
                    x: 0.0,
                    y: 0.0,
                    z: torque,
                    _padding: 0.0,
                });
            }
            
            let inv_mass = body.mass_properties().local_mprops.inv_mass;
            inv_masses.push(inv_mass);
            
            #[cfg(feature = "dim3")]
            {
                // SdpMatrix3 is symmetric, extract as Matrix3
                let inv_inertia = body.mass_properties().effective_world_inv_inertia;
                let mut data = [0.0f32; 12];
                // SdpMatrix3 stores only 6 unique values (symmetric)
                data[0] = inv_inertia.m11;
                data[1] = inv_inertia.m12;
                data[2] = inv_inertia.m13;
                data[4] = inv_inertia.m12; // symmetric
                data[5] = inv_inertia.m22;
                data[6] = inv_inertia.m23;
                data[8] = inv_inertia.m13; // symmetric
                data[9] = inv_inertia.m23; // symmetric
                data[10] = inv_inertia.m33;
                inv_inertias.push(GpuMatrix3 { data });
            }
            #[cfg(feature = "dim2")]
            {
                let inv_inertia = body.mass_properties().effective_world_inv_inertia;
                inv_inertias.push(GpuMatrix3 {
                    data: [inv_inertia, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                });
            }
        }

        // Upload to GPU
        self.queue.write_buffer(&gpu_buffer.positions_buffer, 0, bytemuck::cast_slice(&positions));
        self.queue.write_buffer(&gpu_buffer.rotations_buffer, 0, bytemuck::cast_slice(&rotations));
        self.queue.write_buffer(&gpu_buffer.lin_velocities_buffer, 0, bytemuck::cast_slice(&lin_vels));
        self.queue.write_buffer(&gpu_buffer.ang_velocities_buffer, 0, bytemuck::cast_slice(&ang_vels));
        self.queue.write_buffer(&gpu_buffer.forces_buffer, 0, bytemuck::cast_slice(&forces));
        self.queue.write_buffer(&gpu_buffer.torques_buffer, 0, bytemuck::cast_slice(&torques));
        self.queue.write_buffer(&gpu_buffer.inv_masses_buffer, 0, bytemuck::cast_slice(&inv_masses));
        self.queue.write_buffer(&gpu_buffer.inv_inertias_buffer, 0, bytemuck::cast_slice(&inv_inertias));
    }

    /// Download rigid body data from GPU back to CPU.
    /// Returns vectors of positions and velocities in SoA layout.
    pub fn download_rigid_bodies(&self, gpu_buffer: &RigidBodyGpuBuffer) -> (Vec<GpuVector3>, Vec<GpuVector3>) {
        // Create staging buffers for readback
        let positions_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Positions Staging Buffer"),
            size: (gpu_buffer.body_count * std::mem::size_of::<GpuVector3>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let velocities_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Velocities Staging Buffer"),
            size: (gpu_buffer.body_count * std::mem::size_of::<GpuVector3>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder and copy data
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Download Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &gpu_buffer.positions_buffer,
            0,
            &positions_staging,
            0,
            (gpu_buffer.body_count * std::mem::size_of::<GpuVector3>()) as u64,
        );

        encoder.copy_buffer_to_buffer(
            &gpu_buffer.lin_velocities_buffer,
            0,
            &velocities_staging,
            0,
            (gpu_buffer.body_count * std::mem::size_of::<GpuVector3>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map and read the staging buffers
        let positions_slice = positions_staging.slice(..);
        let velocities_slice = velocities_staging.slice(..);

        positions_slice.map_async(wgpu::MapMode::Read, |_| {});
        velocities_slice.map_async(wgpu::MapMode::Read, |_| {});

        self.device.poll(wgpu::Maintain::Wait);

        let positions_data = positions_slice.get_mapped_range();
        let velocities_data = velocities_slice.get_mapped_range();

        let positions: Vec<GpuVector3> = bytemuck::cast_slice(&positions_data).to_vec();
        let velocities: Vec<GpuVector3> = bytemuck::cast_slice(&velocities_data).to_vec();

        drop(positions_data);
        drop(velocities_data);
        
        positions_staging.unmap();
        velocities_staging.unmap();

        (positions, velocities)
    }

    /// Download all four mutable state arrays from GPU in a single command pass.
    ///
    /// Returns positions, rotations, linear velocities, and angular velocities
    /// exactly as they exist on the GPU after integration.
    pub fn download_full_state(&self, gpu_buffer: &RigidBodyGpuBuffer) -> GpuFullState {
        let n = gpu_buffer.body_count;
        let vec3_bytes = (n * std::mem::size_of::<GpuVector3>()) as u64;
        let rot_bytes  = (n * std::mem::size_of::<GpuRotation>()) as u64;

        let make_staging = |label: &'static str, size: u64| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let pos_stg = make_staging("pos_staging",    vec3_bytes);
        let rot_stg = make_staging("rot_staging",    rot_bytes);
        let lv_stg  = make_staging("linvel_staging", vec3_bytes);
        let av_stg  = make_staging("angvel_staging", vec3_bytes);

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Full State Download"),
        });
        enc.copy_buffer_to_buffer(&gpu_buffer.positions_buffer,      0, &pos_stg, 0, vec3_bytes);
        enc.copy_buffer_to_buffer(&gpu_buffer.rotations_buffer,       0, &rot_stg, 0, rot_bytes);
        enc.copy_buffer_to_buffer(&gpu_buffer.lin_velocities_buffer,  0, &lv_stg,  0, vec3_bytes);
        enc.copy_buffer_to_buffer(&gpu_buffer.ang_velocities_buffer,  0, &av_stg,  0, vec3_bytes);
        self.queue.submit(Some(enc.finish()));

        pos_stg.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        rot_stg.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        lv_stg.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        av_stg.slice(..).map_async(wgpu::MapMode::Read, |_| {});

        self.device.poll(wgpu::Maintain::Wait);

        let positions = {
            let data = pos_stg.slice(..).get_mapped_range();
            bytemuck::cast_slice::<u8, GpuVector3>(&data).to_vec()
        };
        pos_stg.unmap();

        let rotations = {
            let data = rot_stg.slice(..).get_mapped_range();
            bytemuck::cast_slice::<u8, GpuRotation>(&data).to_vec()
        };
        rot_stg.unmap();

        let lin_velocities = {
            let data = lv_stg.slice(..).get_mapped_range();
            bytemuck::cast_slice::<u8, GpuVector3>(&data).to_vec()
        };
        lv_stg.unmap();

        let ang_velocities = {
            let data = av_stg.slice(..).get_mapped_range();
            bytemuck::cast_slice::<u8, GpuVector3>(&data).to_vec()
        };
        av_stg.unmap();

        GpuFullState { positions, rotations, lin_velocities, ang_velocities }
    }

    /// Helper to convert Rapier vector to GPU format (3D).
    #[cfg(feature = "dim3")]
    fn vector_to_gpu_from_vec3(v: &Vec3) -> GpuVector3 {
        GpuVector3 {
            x: v.x,
            y: v.y,
            z: v.z,
            _padding: 0.0,
        }
    }

    /// Helper to convert Rapier vector to GPU format (2D).
    #[cfg(feature = "dim2")]
    fn vector_to_gpu_from_vec2(v: &Vec2) -> GpuVector3 {
        GpuVector3 {
            x: v.x,
            y: v.y,
            z: 0.0,
            _padding: 0.0,
        }
    }

    /// Helper to convert Rapier vector to GPU format.
    fn vector_to_gpu(v: &na::Vector3<Real>) -> GpuVector3 {
        GpuVector3 {
            x: v.x,
            y: v.y,
            z: v.z,
            _padding: 0.0,
        }
    }

    /// Helper to convert Rapier rotation to GPU format (3D).
    #[cfg(feature = "dim3")]
    fn rotation_to_gpu(rot: &Quat) -> GpuRotation {
        GpuRotation {
            x: rot.x,
            y: rot.y,
            z: rot.z,
            w: rot.w,
        }
    }

    /// Helper to convert Rapier rotation to GPU format (2D).
    #[cfg(feature = "dim2")]
    fn rotation_to_gpu(rot: &crate::glamx::Rot2) -> GpuRotation {
        GpuRotation {
            angle: rot.angle(),
            _padding: [0.0, 0.0, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_vector3_size() {
        assert_eq!(std::mem::size_of::<GpuVector3>(), 16);
        assert_eq!(std::mem::align_of::<GpuVector3>(), 4);
    }

    #[test]
    fn test_gpu_rotation_size() {
        #[cfg(feature = "dim3")]
        assert_eq!(std::mem::size_of::<GpuRotation>(), 16);
        #[cfg(feature = "dim2")]
        assert_eq!(std::mem::size_of::<GpuRotation>(), 16);
    }
}
