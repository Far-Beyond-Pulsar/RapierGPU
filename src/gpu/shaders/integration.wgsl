// Integration Kernel - COMPLETE Symplectic Euler Integration
//
// Implements full rigid body dynamics:
// 1. Force accumulation with gravity
// 2. Linear velocity integration: v' = v + (F/m + g) * dt
// 3. Position integration: p' = p + v' * dt
// 4. Angular velocity integration: ω' = ω + I⁻¹(τ) * dt
// 5. Rotation integration: q' = q + 0.5 * dt * [ω.x, ω.y, ω.z, 0] * q
// 6. Apply damping to velocities

struct IntegrationParams {
    body_count: u32,
    dt: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    linear_damping: f32,
    angular_damping: f32,
    clear_forces: u32,  // 1 = clear forces after integration, 0 = accumulate
}

struct GpuVector3 {
    x: f32,
    y: f32,
    z: f32,
    _padding: f32,
}

struct GpuRotation {
    x: f32,  // quaternion x
    y: f32,  // quaternion y
    z: f32,  // quaternion z
    w: f32,  // quaternion w (scalar part)
}

struct GpuMatrix3 {
    // 3x3 matrix stored as 3 vec4s (row-major, padded)
    m00: f32, m01: f32, m02: f32, _pad0: f32,
    m10: f32, m11: f32, m12: f32, _pad1: f32,
    m20: f32, m21: f32, m22: f32, _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: IntegrationParams;
@group(0) @binding(1) var<storage, read_write> positions: array<GpuVector3>;
@group(0) @binding(2) var<storage, read_write> rotations: array<GpuRotation>;
@group(0) @binding(3) var<storage, read_write> lin_velocities: array<GpuVector3>;
@group(0) @binding(4) var<storage, read_write> ang_velocities: array<GpuVector3>;
@group(0) @binding(5) var<storage, read_write> forces: array<GpuVector3>;
@group(0) @binding(6) var<storage, read_write> torques: array<GpuVector3>;
@group(0) @binding(7) var<storage, read> inv_masses: array<f32>;
@group(0) @binding(8) var<storage, read> inv_inertias: array<GpuMatrix3>;

// Quaternion multiplication: q1 * q2
fn quat_mul(q1: vec4<f32>, q2: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

// Normalize quaternion
fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
    let len = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (len > 0.0001) {
        return q / len;
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0); // Identity quaternion
}

// Rotate vector by quaternion: q * v * q^-1
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let qw = q.w;
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    return v + ((uv * qw) + uuv) * 2.0;
}

// Apply 3x3 matrix to vector
fn mat3_mul_vec3(m: GpuMatrix3, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        m.m00 * v.x + m.m01 * v.y + m.m02 * v.z,
        m.m10 * v.x + m.m11 * v.y + m.m12 * v.z,
        m.m20 * v.x + m.m21 * v.y + m.m22 * v.z
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.body_count) {
        return;
    }
    
    let inv_mass = inv_masses[idx];
    
    // TEMPORARILY DISABLED: Skip check to debug mass issues
    // Skip fixed/static bodies (infinite mass = 0 inv_mass)
    // if (inv_mass == 0.0) {
    //     return;
    // }
    
    // ========== Load current state ==========
    var pos = vec3<f32>(positions[idx].x, positions[idx].y, positions[idx].z);
    var lin_vel = vec3<f32>(lin_velocities[idx].x, lin_velocities[idx].y, lin_velocities[idx].z);
    var ang_vel = vec3<f32>(ang_velocities[idx].x, ang_velocities[idx].y, ang_velocities[idx].z);
    var quat = vec4<f32>(
        rotations[idx].x,
        rotations[idx].y,
        rotations[idx].z,
        rotations[idx].w
    );
    
    let force = vec3<f32>(forces[idx].x, forces[idx].y, forces[idx].z);
    let torque = vec3<f32>(torques[idx].x, torques[idx].y, torques[idx].z);
    let inv_inertia = inv_inertias[idx];
    let gravity = vec3<f32>(params.gravity_x, params.gravity_y, params.gravity_z);
    
    // ========== LINEAR INTEGRATION ==========
    
    // 1. Apply forces and gravity to linear velocity
    // v' = v + (F/m + g) * dt
    let linear_acceleration = force * inv_mass + gravity;
    lin_vel = lin_vel + linear_acceleration * params.dt;
    
    // 2. Update position using NEW velocity (symplectic Euler)
    // p' = p + v' * dt
    pos = pos + lin_vel * params.dt;
    
    // 3. Apply linear damping AFTER position update (correct formula)
    // v' = v / (1 + damping * dt)
    let linear_damping_divisor = 1.0 + params.linear_damping * params.dt;
    lin_vel = lin_vel / linear_damping_divisor;
    
    // ========== ANGULAR INTEGRATION ==========
    
    // 4. Transform torque to world space and apply inverse inertia
    // τ_world = R * τ_local
    // ω' = ω + I_world⁻¹ * τ * dt
    // where I_world⁻¹ = R * I_local⁻¹ * R^T
    
    // Transform torque from local to world space
    let torque_world = quat_rotate(quat, torque);
    
    // Apply inverse inertia tensor (in local space for efficiency)
    // Transform angular velocity to local space
    let ang_vel_local = quat_rotate(vec4<f32>(-quat.x, -quat.y, -quat.z, quat.w), ang_vel);
    
    // Apply local inverse inertia tensor
    let ang_accel_local = mat3_mul_vec3(inv_inertia, torque);
    
    // Transform back to world space
    let ang_accel_world = quat_rotate(quat, ang_accel_local);
    
    // Update angular velocity
    ang_vel = ang_vel + ang_accel_world * params.dt;
    
    // 5. Update rotation using quaternion derivative
    // q' = q + 0.5 * dt * quat(ω, 0) * q
    // This is the quaternion differential equation for rotation
    let half_dt = params.dt * 0.5;
    let ang_vel_quat = vec4<f32>(ang_vel.x, ang_vel.y, ang_vel.z, 0.0);
    let quat_deriv = quat_mul(ang_vel_quat, quat);
    quat = quat + quat_deriv * half_dt;
    
    // 6. Normalize quaternion to prevent drift
    quat = quat_normalize(quat);
    
    // 7. Apply angular damping AFTER rotation update (correct formula)
    // ω' = ω / (1 + damping) [note: no dt factor for angular in Rapier]
    let angular_damping_divisor = 1.0 + params.angular_damping;
    ang_vel = ang_vel / angular_damping_divisor;
    
    // ========== Write back updated state ==========
    positions[idx].x = pos.x;
    positions[idx].y = pos.y;
    positions[idx].z = pos.z;
    
    rotations[idx].x = quat.x;
    rotations[idx].y = quat.y;
    rotations[idx].z = quat.z;
    rotations[idx].w = quat.w;
    
    lin_velocities[idx].x = lin_vel.x;
    lin_velocities[idx].y = lin_vel.y;
    lin_velocities[idx].z = lin_vel.z;
    
    ang_velocities[idx].x = ang_vel.x;
    ang_velocities[idx].y = ang_vel.y;
    ang_velocities[idx].z = ang_vel.z;
    
    // Clear forces if requested (for per-frame force application)
    if (params.clear_forces != 0u) {
        forces[idx].x = 0.0;
        forces[idx].y = 0.0;
        forces[idx].z = 0.0;
        
        torques[idx].x = 0.0;
        torques[idx].y = 0.0;
        torques[idx].z = 0.0;
    }
}
