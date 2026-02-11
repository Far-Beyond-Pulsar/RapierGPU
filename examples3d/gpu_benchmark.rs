//! GPU acceleration benchmark: CPU vs GPU Naive vs GPU Delta
//! 
//! Tests 3 strategies:
//! 1. **CPU**: Baseline single-threaded integration
//! 2. **GPU Naive**: Upload all + compute + download all every frame (worst case)
//! 3. **GPU Delta**: Compute only, GPU-resident data (realistic production scenario)

use rapier3d::prelude::*;
use rapier3d::gpu::{GpuContext, BufferManager, GpuIntegrator, wgpu};
use rapier3d::na::Point3;
use std::time::Instant;

const SCALES: &[usize] = &[10, 50, 100, 500, 1000, 5000, 10000, 50000/*, 100000, 1_000_000, 100_000_000*/];
const ITERATIONS: usize = 100;
const SIMULATION_FRAMES: usize = 10; // Run multiple frames per measurement

struct BenchmarkResult {
    scale: usize,
    cpu_time: f64,
    gpu_naive_time: f64,
    gpu_delta_time: f64,
}

fn create_test_bodies(count: usize) -> RigidBodySet {
    let mut bodies = RigidBodySet::new();
    
    for i in 0..count {
        let x = (i % 10) as Real;
        let y = (i / 10) as Real;
        let z = (i / 100) as Real;
        
        // Create mass properties manually since bodies have no colliders
        let mass_props = MassProperties::new(
            Vector::new(0.0, 0.0, 0.0),  // center of mass
            1.0,  // mass = 1.0 kg
            Vector::new(1.0, 1.0, 1.0)  // principal inertia
        );
        
        let body = RigidBodyBuilder::dynamic()
            .translation(Vector::new(x * 2.0, y * 2.0 + 10.0, z * 2.0))
            .linvel(Vector::new(0.0, -1.0, 0.0))
            .additional_mass_properties(mass_props)
            .build();
        
        bodies.insert(body);
    }
    
    bodies
}

fn benchmark_cpu_integration(bodies: &mut RigidBodySet, frames: usize, iterations: usize) -> f64 {
    let dt = 1.0 / 60.0;
    let gravity = Vector::new(0.0, -9.81, 0.0);
    
    let start = Instant::now();
    for _ in 0..iterations {
        for _ in 0..frames {
            // Simulate CPU integration
            for (_handle, body) in bodies.iter_mut() {
                if !body.is_dynamic() {
                    continue;
                }
                
                // Apply gravity
                let force = gravity * body.mass();
                let lin_vel = body.linvel() + (force / body.mass()) * dt;
                
                // Update position (symplectic Euler)
                let new_translation = body.translation() + lin_vel * dt;
                let mut new_isometry = *body.position();
                new_isometry.translation = new_translation.into();
                
                body.set_linvel(lin_vel, false);
                body.set_position(new_isometry, false);
            }
        }
    }
    let elapsed = start.elapsed();
    
    elapsed.as_micros() as f64 / iterations as f64
}

fn benchmark_gpu_naive(
    bodies: &RigidBodySet, 
    buffer_manager: &BufferManager, 
    integrator: &GpuIntegrator,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_buffer: &mut rapier3d::gpu::RigidBodyGpuBuffer,
    frames: usize,
    iterations: usize
) -> f64 {
    let dt = 1.0 / 60.0;
    let gravity = [0.0, -9.81, 0.0];
    
    let start = Instant::now();
    for _ in 0..iterations {
        for _ in 0..frames {
            // NAIVE: Upload ALL + compute + download ALL every frame
            buffer_manager.upload_rigid_bodies(bodies, gpu_buffer);
            integrator.integrate(device, queue, gpu_buffer, dt, gravity, 0.0, 0.0);
            let (_positions, _velocities) = buffer_manager.download_rigid_bodies(gpu_buffer);
        }
        // Wait for GPU to finish all work
        device.poll(wgpu::Maintain::Wait);
    }
    let elapsed = start.elapsed();
    
    elapsed.as_micros() as f64 / iterations as f64
}

fn benchmark_gpu_delta(
    integrator: &GpuIntegrator,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_buffer: &mut rapier3d::gpu::RigidBodyGpuBuffer,
    frames: usize,
    iterations: usize
) -> f64 {
    let dt = 1.0 / 60.0;
    let gravity = [0.0, -9.81, 0.0];
    
    let start = Instant::now();
    for _ in 0..iterations {
        for _ in 0..frames {
            // DELTA/GPU-RESIDENT: Compute only, NO transfers!
            // Data stays on GPU permanently (realistic production scenario)
            integrator.integrate(device, queue, gpu_buffer, dt, gravity, 0.0, 0.0);
        }
        // CRITICAL: Wait for GPU to actually finish!
        // queue.submit() is async, doesn't wait for completion
        device.poll(wgpu::Maintain::Wait);
    }
    let elapsed = start.elapsed();
    
    elapsed.as_micros() as f64 / iterations as f64
}

fn print_results_table(results: &[BenchmarkResult]) {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║             CPU vs GPU TRANSFER STRATEGY COMPARISON ({} frames/iter)                      ║", SIMULATION_FRAMES);
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Bodies  │    CPU     │ GPU Naive  │ GPU Delta  │ Naive vs CPU │ Delta vs CPU │   Winner   ║");
    println!("╠══════════╪════════════╪════════════╪════════════╪══════════════╪══════════════╪════════════╣");
    
    for result in results {
        let cpu = result.cpu_time;
        let naive = result.gpu_naive_time;
        let delta = result.gpu_delta_time;
        
        let naive_speedup = cpu / naive;
        let delta_speedup = cpu / delta;
        
        let naive_str = if naive_speedup > 1.0 {
            format!("GPU {:.1}x", naive_speedup)
        } else {
            format!("CPU {:.1}x", 1.0 / naive_speedup)
        };
        
        let delta_str = if delta_speedup > 1.0 {
            format!("GPU {:.1}x", delta_speedup)
        } else {
            format!("CPU {:.1}x", 1.0 / delta_speedup)
        };
        
        let winner = if delta_speedup > 1.0 {
            "🚀 GPU"
        } else if naive_speedup > 0.8 {
            "⚖️  Even"
        } else {
            "💻 CPU"
        };
        
        println!("║ {:>8} │ {:>10} │ {:>10} │ {:>10} │ {:>12} │ {:>12} │ {:>10} ║",
            result.scale,
            format_time(cpu),
            format_time(naive),
            format_time(delta),
            naive_str,
            delta_str,
            winner
        );
    }
    
    println!("╚══════════╧════════════╧════════════╧════════════╧══════════════╧══════════════╧════════════╝");
    println!("\n📊 Key Insights:");
    println!("  • **GPU Naive**: Upload all + compute + download all (current worst case)");
    println!("  • **GPU Delta**: Compute only, data GPU-resident (PhysX-style architecture)");
    println!("  • Delta strategy = production performance! Data lives on GPU.");
}

fn format_time(micros: f64) -> String {
    if micros < 1000.0 {
        format!("{:.2} µs", micros)
    } else if micros < 1_000_000.0 {
        format!("{:.2} ms", micros / 1000.0)
    } else {
        format!("{:.2} s", micros / 1_000_000.0)
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║              RAPIER GPU ACCELERATION: TRANSFER STRATEGY BENCHMARK             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝\n");
    
    println!("Initializing GPU...");
    let gpu_ctx = match GpuContext::new() {
        Ok(ctx) => {
            println!("✓ GPU initialized: {}", ctx.adapter.get_info().name);
            ctx
        },
        Err(e) => {
            println!("✗ GPU not available: {:?}", e);
            println!("  Cannot run GPU benchmarks");
            return;
        }
    };
    
    let mut results = Vec::new();
    
    for &scale in SCALES {
        print!("Benchmarking {} bodies... ", scale);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        
        let mut bodies = create_test_bodies(scale);
        
        // Benchmark CPU
        let cpu_time = benchmark_cpu_integration(&mut bodies, SIMULATION_FRAMES, ITERATIONS);
        
        // Benchmark GPU strategies
        let buffer_manager = BufferManager::new(gpu_ctx.device.clone(), gpu_ctx.queue.clone());
        let integrator = GpuIntegrator::new(&gpu_ctx.device);
        let mut gpu_buffer = buffer_manager.create_rigid_body_buffer(bodies.len());
        
        // Initial upload (one-time cost amortized over frames)
        buffer_manager.upload_rigid_bodies(&bodies, &mut gpu_buffer);
        
        // Strategy 1: Naive (worst case - transfer everything)
        let gpu_naive_time = benchmark_gpu_naive(
            &bodies, &buffer_manager, &integrator,
            &gpu_ctx.device, &gpu_ctx.queue, &mut gpu_buffer,
            SIMULATION_FRAMES, ITERATIONS
        );
        
        // Strategy 2: Delta/GPU-Resident (realistic - compute only)
        let gpu_delta_time = benchmark_gpu_delta(
            &integrator,
            &gpu_ctx.device, &gpu_ctx.queue, &mut gpu_buffer,
            SIMULATION_FRAMES, ITERATIONS
        );
        
        results.push(BenchmarkResult {
            scale,
            cpu_time,
            gpu_naive_time,
            gpu_delta_time,
        });
        
        println!("Done!");
    }
    
    print_results_table(&results);
}
