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

/// Manages GPU buffer lifecycle and CPU↔GPU transfers.
pub struct BufferManager {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl BufferManager {
    /// Creates a new buffer manager.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
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
                    x: angvel,
                    y: 0.0,
                    z: 0.0,
                    _padding: 0.0,
                });
            }
            
            // Forces (accumulated)
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
                    x: torque,
                    y: 0.0,
                    z: 0.0,
                    _padding: 0.0,
                });
            }
            
            inv_masses.push(body.mass_properties().local_mprops.inv_mass);
            
            // Inverse inertia tensor
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
