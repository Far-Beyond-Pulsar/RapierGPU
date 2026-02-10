// Integration Kernel - Symplectic Euler Integration
//
// Computes next frame state for rigid bodies:
// 1. Velocity integration: v' = v + (F/m + gravity) * dt
// 2. Position integration: p' = p + v' * dt
// 3. Angular integration: ω' = I⁻¹(τ * dt), rotation update
// 4. Apply linear/angular damping

struct IntegrationParams {
    body_count: u32,
    dt: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    linear_damping: f32,
    angular_damping: f32,
    _padding: f32,
}

struct GpuVector3 {
    x: f32,
    y: f32,
    z: f32,
    _padding: f32,
}

struct GpuRotation {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

@group(0) @binding(0) var<uniform> params: IntegrationParams;
@group(0) @binding(1) var<storage, read_write> positions: array<GpuVector3>;
@group(0) @binding(2) var<storage, read_write> rotations: array<GpuRotation>;
@group(0) @binding(3) var<storage, read_write> lin_velocities: array<GpuVector3>;
@group(0) @binding(4) var<storage, read_write> ang_velocities: array<GpuVector3>;
@group(0) @binding(5) var<storage, read> forces: array<GpuVector3>;
@group(0) @binding(6) var<storage, read> torques: array<GpuVector3>;
@group(0) @binding(7) var<storage, read> inv_masses: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.body_count) {
        return;
    }
    
    let inv_mass = inv_masses[idx];
    
    // Skip fixed/static bodies (infinite mass = 0 inv_mass)
    if (inv_mass == 0.0) {
        return;
    }
    
    // Load current state
    var pos = vec3<f32>(positions[idx].x, positions[idx].y, positions[idx].z);
    var vel = vec3<f32>(lin_velocities[idx].x, lin_velocities[idx].y, lin_velocities[idx].z);
    var ang_vel = vec3<f32>(ang_velocities[idx].x, ang_velocities[idx].y, ang_velocities[idx].z);
    let force = vec3<f32>(forces[idx].x, forces[idx].y, forces[idx].z);
    let torque = vec3<f32>(torques[idx].x, torques[idx].y, torques[idx].z);
    
    let gravity = vec3<f32>(params.gravity_x, params.gravity_y, params.gravity_z);
    
    // 1. Velocity integration: v' = v + (F/m + g) * dt
    let acceleration = force * inv_mass + gravity;
    vel = vel + acceleration * params.dt;
    
    // 2. Apply linear damping: v' = v * (1 - damping)^dt
    // Approximate: v' ≈ v * (1 - damping * dt) for small dt
    vel = vel * (1.0 - params.linear_damping * params.dt);
    
    // 3. Position integration: p' = p + v * dt (symplectic Euler)
    pos = pos + vel * params.dt;
    
    // 4. Angular velocity damping
    ang_vel = ang_vel * (1.0 - params.angular_damping * params.dt);
    
    // 5. Rotation integration (simplified - quaternion integration would go here)
    // For now, we'll just update angular velocity
    // Full quaternion integration: q' = q + 0.5 * q * [0, ω] * dt
    // TODO: Implement proper quaternion integration
    
    // Write back updated state
    positions[idx].x = pos.x;
    positions[idx].y = pos.y;
    positions[idx].z = pos.z;
    
    lin_velocities[idx].x = vel.x;
    lin_velocities[idx].y = vel.y;
    lin_velocities[idx].z = vel.z;
    
    ang_velocities[idx].x = ang_vel.x;
    ang_velocities[idx].y = ang_vel.y;
    ang_velocities[idx].z = ang_vel.z;
}
