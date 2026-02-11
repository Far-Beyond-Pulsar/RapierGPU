//! GPU Parity Test — CPU vs GPU-Resident Integration
//!
//! Validates that the GPU integration kernel produces numerically equivalent
//! results to the CPU reference implementation by running 1 000 bodies
//! through 100 frames and comparing every field of every body.
//!
//! Run with:
//!   cargo run --bin gpu_parity_test --features rapier3d/gpu-acceleration

use rapier3d::prelude::*;
use rapier3d::gpu::{GpuContext, BufferManager, GpuIntegrator, GpuFullState, wgpu};
use std::time::Instant;
use owo_colors::OwoColorize;
use comfy_table::*;
use comfy_table::presets::UTF8_FULL;
use comfy_table::modifiers::UTF8_ROUND_CORNERS;

// ── Simulation parameters ────────────────────────────────────────────────────

const BODY_COUNT:      usize = 1000;
const FRAME_COUNT:     usize = 100;
const GRAVITY:         [f32; 3] = [0.0, -9.81, 0.0];
const DT:              f32 = 1.0 / 60.0;
const LINEAR_DAMPING:  f32 = 0.0;
const ANGULAR_DAMPING: f32 = 0.0;

/// Maximum tolerated absolute error per component after FRAME_COUNT steps.
/// f32 arithmetic over 100 frames accumulates ~1e-4 absolute error at these
/// scales; 1e-3 gives comfortable headroom while still catching real bugs.
const TOLERANCE: f32 = 1e-3;

// ── Per-body CPU state ───────────────────────────────────────────────────────

/// Raw f32 state mirroring exactly what lives in the GPU SoA buffers.
struct BodyState {
    pos:     [f32; 3],
    rot:     [f32; 4], // quaternion [x, y, z, w]
    lin_vel: [f32; 3],
    ang_vel: [f32; 3],
    /// Inverse mass (constant; uploaded once to GPU)
    inv_mass: f32,
    /// Row-major 3×3 inverse-inertia tensor + per-row f32 padding (12 floats).
    /// Layout matches GpuMatrix3 / the WGSL `inv_inertias` buffer.
    inv_inertia: [f32; 12],
}

// ── Maths helpers — exact mirrors of the WGSL shader ────────────────────────

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Quaternion product: q1 ⊗ q2, layout [x, y, z, w].
/// Matches `quat_mul` in integration.wgsl.
#[inline]
fn quat_mul(q1: [f32; 4], q2: [f32; 4]) -> [f32; 4] {
    let [x1, y1, z1, w1] = q1;
    let [x2, y2, z2, w2] = q2;
    [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]
}

/// Normalize quaternion.  Returns identity on near-zero input.
#[inline]
fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len > 0.0001 {
        [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// Rotate vector by quaternion: q * v * q⁻¹.
/// Matches `quat_rotate` in integration.wgsl.
#[inline]
fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let qv = [q[0], q[1], q[2]];
    let qw = q[3];
    let uv  = cross3(qv, v);
    let uuv = cross3(qv, uv);
    [
        v[0] + (uv[0] * qw + uuv[0]) * 2.0,
        v[1] + (uv[1] * qw + uuv[1]) * 2.0,
        v[2] + (uv[2] * qw + uuv[2]) * 2.0,
    ]
}

/// Multiply 3×3 matrix (stored as 12 f32s matching GpuMatrix3) by a vector.
/// Matches `mat3_mul_vec3` in integration.wgsl.
#[inline]
fn mat3_mul_vec3(m: &[f32; 12], v: [f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2]  * v[2],
        m[4] * v[0] + m[5] * v[1] + m[6]  * v[2],
        m[8] * v[0] + m[9] * v[1] + m[10] * v[2],
    ]
}

// ── CPU integration step — must be a bit-for-bit replica of integration.wgsl ─

fn cpu_step(s: &mut BodyState) {
    // All bodies start with zero user forces / torques; gravity is external.
    // This mirrors the GPU path with clear_forces = true.
    let force  = [0.0f32; 3];
    let torque = [0.0f32; 3];

    // ── Linear ────────────────────────────────────────────────────────────────

    // 1.  v' = v + (F/m + g) · dt
    let lin_accel = [
        force[0] * s.inv_mass + GRAVITY[0],
        force[1] * s.inv_mass + GRAVITY[1],
        force[2] * s.inv_mass + GRAVITY[2],
    ];
    s.lin_vel[0] += lin_accel[0] * DT;
    s.lin_vel[1] += lin_accel[1] * DT;
    s.lin_vel[2] += lin_accel[2] * DT;

    // 2.  p' = p + v' · dt  (symplectic Euler — uses the updated velocity)
    s.pos[0] += s.lin_vel[0] * DT;
    s.pos[1] += s.lin_vel[1] * DT;
    s.pos[2] += s.lin_vel[2] * DT;

    // 3.  Linear damping:  v' = v / (1 + d · dt)
    let ld = 1.0 + LINEAR_DAMPING * DT;
    s.lin_vel[0] /= ld;
    s.lin_vel[1] /= ld;
    s.lin_vel[2] /= ld;

    // ── Angular ───────────────────────────────────────────────────────────────

    // 4.  ang_accel_local = I⁻¹ · τ  (τ is zero here, but kept for symmetry)
    let ang_accel_local = mat3_mul_vec3(&s.inv_inertia, torque);

    // 5.  Transform to world space
    let ang_accel_world = quat_rotate(s.rot, ang_accel_local);

    // 6.  ω' = ω + α · dt
    s.ang_vel[0] += ang_accel_world[0] * DT;
    s.ang_vel[1] += ang_accel_world[1] * DT;
    s.ang_vel[2] += ang_accel_world[2] * DT;

    // 7.  Quaternion derivative:  q' = q + ½ · dt · [ω, 0] ⊗ q
    let half_dt  = DT * 0.5;
    let aw_quat  = [s.ang_vel[0], s.ang_vel[1], s.ang_vel[2], 0.0f32];
    let qd       = quat_mul(aw_quat, s.rot);
    s.rot[0] += qd[0] * half_dt;
    s.rot[1] += qd[1] * half_dt;
    s.rot[2] += qd[2] * half_dt;
    s.rot[3] += qd[3] * half_dt;

    // 8.  Normalise to prevent drift
    s.rot = quat_normalize(s.rot);

    // 9.  Angular damping:  ω' = ω / (1 + d)  [no dt factor — matches shader]
    let ad = 1.0 + ANGULAR_DAMPING;
    s.ang_vel[0] /= ad;
    s.ang_vel[1] /= ad;
    s.ang_vel[2] /= ad;

    // (Forces already zero; clear is a no-op but mirrors GPU clear_forces = true)
}

// ── Scene construction ───────────────────────────────────────────────────────

fn create_bodies() -> RigidBodySet {
    let mut bodies = RigidBodySet::new();

    for i in 0..BODY_COUNT {
        // 10 × 10 × 10 grid of bodies with 2.5 m spacing
        let x = (i % 10)       as Real * 2.5;
        let y = (i / 10 % 10)  as Real * 2.5 + 10.0;
        let z = (i / 100)      as Real * 2.5;

        // Varied initial linear velocity so each body takes a different path
        let vx = ((i % 5)  as f32 - 2.0) * 0.4;  // −0.8 … 0.8
        let vy = -1.0 - (i % 4) as f32 * 0.25;   // −1.0 … −1.75
        let vz = ((i % 7)  as f32 - 3.0) * 0.2;  // −0.6 … 0.6

        // Varied initial angular velocity to exercise quaternion integration
        let wx = ((i % 5)  as f32 - 2.0) * 0.3;  // −0.6 … 0.6
        let wy = ((i % 7)  as f32 - 3.0) * 0.2;  // −0.6 … 0.6
        let wz = ((i % 11) as f32 - 5.0) * 0.15; // −0.75 … 0.75

        // Uniform mass / inertia — matches the hardcoded CPU constants above
        let mass_props = MassProperties::new(
            Vector::new(0.0, 0.0, 0.0),
            1.0,
            Vector::new(1.0, 1.0, 1.0), // principal inertia → inv_inertia = I₃
        );

        let body = RigidBodyBuilder::dynamic()
            .translation(Vector::new(x, y, z))
            .linvel(Vector::new(vx, vy, vz))
            .angvel(Vector::new(wx, wy, wz))
            .additional_mass_properties(mass_props)
            .build();

        bodies.insert(body);
    }

    bodies
}

// ── Convert downloaded GPU initial state into BodyState structs ──────────────

/// inv_mass and inv_inertia for all bodies are identical in this test.
/// inv_mass = 1.0  (mass = 1.0 kg)
/// inv_inertia = I₃ (principal_inertia = [1, 1, 1])
const INV_MASS: f32 = 1.0;
const INV_INERTIA: [f32; 12] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
];

fn initial_cpu_states_from_gpu(snapshot: &GpuFullState) -> Vec<BodyState> {
    (0..snapshot.positions.len())
        .map(|i| {
            let p = &snapshot.positions[i];
            let r = &snapshot.rotations[i];
            let v = &snapshot.lin_velocities[i];
            let w = &snapshot.ang_velocities[i];
            BodyState {
                pos:         [p.x, p.y, p.z],
                rot:         [r.x, r.y, r.z, r.w],
                lin_vel:     [v.x, v.y, v.z],
                ang_vel:     [w.x, w.y, w.z],
                inv_mass:    INV_MASS,
                inv_inertia: INV_INERTIA,
            }
        })
        .collect()
}

// ── Comparison helpers ───────────────────────────────────────────────────────

struct FieldStats {
    name:       &'static str,
    max_err:    f32,
    mean_err:   f32,
    worst_body: usize,
    passed:     bool,
}

fn compare_field(
    name: &'static str,
    cpu_vals: impl Iterator<Item = f32>,
    gpu_vals: impl Iterator<Item = f32>,
) -> FieldStats {
    let mut max_err    = 0.0f32;
    let mut sum_err    = 0.0f32;
    let mut worst_body = 0usize;
    let mut count      = 0usize;

    for (i, (c, g)) in cpu_vals.zip(gpu_vals).enumerate() {
        let err = (c - g).abs();
        sum_err += err;
        if err > max_err {
            max_err    = err;
            worst_body = i;
        }
        count += 1;
    }

    FieldStats {
        name,
        max_err,
        mean_err: if count > 0 { sum_err / count as f32 } else { 0.0 },
        worst_body,
        passed: max_err <= TOLERANCE,
    }
}

fn format_err(v: f32) -> String {
    if v == 0.0 {
        "0.000 000".to_string()
    } else if v < 1e-5 {
        format!("{:.2e}", v)
    } else {
        format!("{:.6}", v)
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!(
        "{}",
        "╔═══════════════════════════════════════════════════════════╗"
            .bright_cyan().bold()
    );
    println!(
        "{}",
        "║      RAPIER GPU PARITY TEST — CPU vs GPU-RESIDENT         ║"
            .bright_cyan().bold()
    );
    println!(
        "{}",
        "╚═══════════════════════════════════════════════════════════╝"
            .bright_cyan().bold()
    );
    println!();
    println!(
        "  {} {} bodies  ×  {} frames  |  dt = {:.4} s  |  tol = {:.0e}",
        "Simulation:".bright_yellow(),
        BODY_COUNT.to_string().yellow().bold(),
        FRAME_COUNT.to_string().yellow().bold(),
        DT,
        TOLERANCE
    );
    println!(
        "  {}  [{}, {}, {}]",
        "Gravity:   ".bright_yellow(),
        GRAVITY[0], GRAVITY[1], GRAVITY[2]
    );
    println!();

    // ── GPU init ─────────────────────────────────────────────────────────────
    print!("{} Initializing GPU... ", "🔧".yellow());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let gpu_ctx = match GpuContext::new() {
        Ok(ctx) => {
            println!(
                "{} {}",
                "✓".green(),
                ctx.adapter.get_info().name.bright_green().bold()
            );
            ctx
        }
        Err(e) => {
            println!("{} GPU unavailable: {:?}", "✗".red(), e);
            println!("  Cannot run GPU parity test.");
            return;
        }
    };

    // ── Scene setup ───────────────────────────────────────────────────────────
    print!("{} Creating {} bodies... ", "🔩".yellow(), BODY_COUNT);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let bodies = create_bodies();
    println!("{}", "✓".green());

    let buffer_manager = BufferManager::new(gpu_ctx.device.clone(), gpu_ctx.queue.clone());
    let integrator     = GpuIntegrator::new(&gpu_ctx.device);
    let mut gpu_buffer = buffer_manager.create_rigid_body_buffer(bodies.len());

    // Upload initial state to GPU
    buffer_manager.upload_rigid_bodies(&bodies, &mut gpu_buffer);

    // Download that same initial state so the CPU starts from the *exact* same
    // f32 values that the GPU shader will read — no separate extraction needed.
    let initial = buffer_manager.download_full_state(&gpu_buffer);
    let mut cpu_states = initial_cpu_states_from_gpu(&initial);

    // ── CPU simulation ────────────────────────────────────────────────────────
    print!(
        "{} Running CPU integration ({} frames)... ",
        "💻".cyan(),
        FRAME_COUNT
    );
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let t0 = Instant::now();
    for _ in 0..FRAME_COUNT {
        for s in cpu_states.iter_mut() {
            cpu_step(s);
        }
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("{} in {:.2} ms", "Done".green(), cpu_ms);

    // ── GPU simulation ────────────────────────────────────────────────────────
    // The GPU buffer still holds the initial state from the upload above.
    print!(
        "{} Running GPU integration ({} frames)... ",
        "🚀".cyan(),
        FRAME_COUNT
    );
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let t1 = Instant::now();
    for _ in 0..FRAME_COUNT {
        integrator.integrate(
            &gpu_ctx.device,
            &gpu_ctx.queue,
            &mut gpu_buffer,
            DT,
            GRAVITY,
            LINEAR_DAMPING,
            ANGULAR_DAMPING,
            true, // clear_forces — mirrors cpu_step
        );
    }
    gpu_ctx.device.poll(wgpu::Maintain::Wait);
    let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!("{} in {:.2} ms", "Done".green(), gpu_ms);

    // ── Download GPU final state ──────────────────────────────────────────────
    print!("{} Downloading GPU final state... ", "📥".cyan());
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let gpu_final = buffer_manager.download_full_state(&gpu_buffer);
    println!("{}", "Done".green());

    // ── Compare every field of every body ─────────────────────────────────────
    println!();

    let n = cpu_states.len();

    let stats: Vec<FieldStats> = vec![
        compare_field(
            "Position X",
            cpu_states.iter().map(|s| s.pos[0]),
            gpu_final.positions.iter().map(|v| v.x),
        ),
        compare_field(
            "Position Y",
            cpu_states.iter().map(|s| s.pos[1]),
            gpu_final.positions.iter().map(|v| v.y),
        ),
        compare_field(
            "Position Z",
            cpu_states.iter().map(|s| s.pos[2]),
            gpu_final.positions.iter().map(|v| v.z),
        ),
        compare_field(
            "Rotation X",
            cpu_states.iter().map(|s| s.rot[0]),
            gpu_final.rotations.iter().map(|r| r.x),
        ),
        compare_field(
            "Rotation Y",
            cpu_states.iter().map(|s| s.rot[1]),
            gpu_final.rotations.iter().map(|r| r.y),
        ),
        compare_field(
            "Rotation Z",
            cpu_states.iter().map(|s| s.rot[2]),
            gpu_final.rotations.iter().map(|r| r.z),
        ),
        compare_field(
            "Rotation W",
            cpu_states.iter().map(|s| s.rot[3]),
            gpu_final.rotations.iter().map(|r| r.w),
        ),
        compare_field(
            "Linear Vel X",
            cpu_states.iter().map(|s| s.lin_vel[0]),
            gpu_final.lin_velocities.iter().map(|v| v.x),
        ),
        compare_field(
            "Linear Vel Y",
            cpu_states.iter().map(|s| s.lin_vel[1]),
            gpu_final.lin_velocities.iter().map(|v| v.y),
        ),
        compare_field(
            "Linear Vel Z",
            cpu_states.iter().map(|s| s.lin_vel[2]),
            gpu_final.lin_velocities.iter().map(|v| v.z),
        ),
        compare_field(
            "Angular Vel X",
            cpu_states.iter().map(|s| s.ang_vel[0]),
            gpu_final.ang_velocities.iter().map(|v| v.x),
        ),
        compare_field(
            "Angular Vel Y",
            cpu_states.iter().map(|s| s.ang_vel[1]),
            gpu_final.ang_velocities.iter().map(|v| v.y),
        ),
        compare_field(
            "Angular Vel Z",
            cpu_states.iter().map(|s| s.ang_vel[2]),
            gpu_final.ang_velocities.iter().map(|v| v.z),
        ),
    ];

    // ── Results table ─────────────────────────────────────────────────────────
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("Field")       .fg(Color::Cyan).add_attribute(Attribute::Bold),
            Cell::new("Max Error")   .fg(Color::Cyan).add_attribute(Attribute::Bold),
            Cell::new("Mean Error")  .fg(Color::Cyan).add_attribute(Attribute::Bold),
            Cell::new("Worst Body")  .fg(Color::Cyan).add_attribute(Attribute::Bold),
            Cell::new("Status")      .fg(Color::Cyan).add_attribute(Attribute::Bold),
        ]);

    let passed_count = stats.iter().filter(|s| s.passed).count();

    for st in &stats {
        let status_cell = if st.passed {
            Cell::new("✓ PASS").fg(Color::Green).add_attribute(Attribute::Bold)
        } else {
            Cell::new("✗ FAIL").fg(Color::Red).add_attribute(Attribute::Bold)
        };
        let err_color = if st.passed { Color::Green } else { Color::Red };

        table.add_row(vec![
            Cell::new(st.name).fg(Color::White),
            Cell::new(format_err(st.max_err))  .fg(err_color),
            Cell::new(format_err(st.mean_err)) .fg(Color::DarkGrey),
            Cell::new(format!("#{}", st.worst_body)).fg(Color::DarkGrey),
            status_cell,
        ]);
    }

    println!("{}", table);
    println!("  Tolerance: {:.0e}  |  Bodies: {}  |  Frames: {}", TOLERANCE, n, FRAME_COUNT);
    println!();

    // ── Overall verdict ───────────────────────────────────────────────────────
    let all_passed = passed_count == stats.len();
    let verdict_line = "═".repeat(55);

    if all_passed {
        println!("{}", verdict_line.bright_green().bold());
        println!(
            "{}",
            format!(
                "  OVERALL RESULT: ✓ ALL {} FIELDS PASSED ({}/{})  ",
                stats.len(),
                passed_count,
                stats.len()
            )
            .bright_green()
            .bold()
        );
        println!("{}", verdict_line.bright_green().bold());
        println!();
        println!(
            "  GPU integration is numerically equivalent to CPU (tol = {:.0e}).",
            TOLERANCE
        );
    } else {
        println!("{}", verdict_line.bright_red().bold());
        println!(
            "{}",
            format!(
                "  OVERALL RESULT: ✗ {} FIELD(S) FAILED ({}/{} passed)  ",
                stats.len() - passed_count,
                passed_count,
                stats.len()
            )
            .bright_red()
            .bold()
        );
        println!("{}", verdict_line.bright_red().bold());
        println!();
        println!("{}", "  Failing fields:".bright_red());
        for st in stats.iter().filter(|s| !s.passed) {
            println!(
                "    • {}  max_err = {:.2e}  (tol = {:.0e})",
                st.name.red().bold(),
                st.max_err,
                TOLERANCE
            );
        }
    }

    // ── Timing summary ────────────────────────────────────────────────────────
    println!();
    println!("{}", "  Timing".bright_yellow().bold());
    println!("    CPU  {} frames: {:.2} ms  ({:.2} µs/frame)", FRAME_COUNT, cpu_ms, cpu_ms * 1000.0 / FRAME_COUNT as f64);
    println!("    GPU  {} frames: {:.2} ms  ({:.2} µs/frame)", FRAME_COUNT, gpu_ms, gpu_ms * 1000.0 / FRAME_COUNT as f64);
    println!();
}
