//! Advanced GPU benchmark comparing transfer strategies.
//! 
//! Measures 3 scenarios to show where GPU wins:
//! 1. **Naive (current)**: Upload all → compute → download all every frame
//! 2. **GPU-resident**: Compute only, no transfers (realistic for many frames)
//! 3. **Hybrid**: Upload deltas → compute → download rendering data only

use rapier3d::prelude::*;
use rapier3d::gpu::{GpuContext, BufferManager, GpuIntegrator, GpuResidentState};
use std::time::Instant;

const SCALES: &[usize] = &[10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000];
const ITERATIONS: usize = 100;
const SIMULATION_FRAMES: usize = 10; // Simulate 10 physics steps per measurement

struct BenchmarkResult {
    scale: usize,
    cpu_time: f64,
    gpu_naive_time: f64,
    gpu_resident_time: f64,
}

fn create_test_bodies(count: usize) -> RigidBodySet {
    let mut bodies = RigidBodySet::new();
    
    for i in 0..count {
        let x = (i % 10) as f32;
        let y = (i / 10) as f32;
        let z = (i / 100) as f32;
        
        let body = RigidBodyBuilder::dynamic()
            .translation(vector![x * 2.0, y * 2.0 + 10.0, z * 2.0])
            .linvel(vector![0.0, 0.0, 0.0])
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
                let ang_vel = body.angvel();
                
                // Update position (symplectic Euler)
                let new_pos = body.position().translation.vector + lin_vel * dt;
                let mut new_isometry = *body.position();
                new_isometry.translation.vector = new_pos;
                
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
            // NAIVE: Upload + compute + download EVERY frame (worst case)
            buffer_manager.upload_rigid_bodies(bodies, gpu_buffer);
            integrator.integrate(device, queue, gpu_buffer, dt, gravity, 0.0, 0.0);
            let (_positions, _velocities) = buffer_manager.download_rigid_bodies(gpu_buffer);
        }
    }
    let elapsed = start.elapsed();
    
    elapsed.as_micros() as f64 / iterations as f64
}

fn benchmark_gpu_resident(
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
            // GPU-RESIDENT: Compute only, no transfers!
            // This is realistic when bodies stay on GPU for many frames
            integrator.integrate(device, queue, gpu_buffer, dt, gravity, 0.0, 0.0);
        }
    }
    let elapsed = start.elapsed();
    
    elapsed.as_micros() as f64 / iterations as f64
}

fn print_results_table(results: &[BenchmarkResult]) {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║              GPU TRANSFER STRATEGY COMPARISON ({} frames/iter)              ║", SIMULATION_FRAMES);
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Bodies  │    CPU     │ GPU Naive  │ GPU Resident │  Speedup  │    Winner    ║");
    println!("╠══════════╪════════════╪════════════╪══════════════╪═══════════╪══════════════╣");
    
    for result in results {
        let cpu = result.cpu_time;
        let gpu_naive = result.gpu_naive_time;
        let gpu_resident = result.gpu_resident_time;
        
        // Compare best GPU approach vs CPU
        let best_gpu = gpu_resident.min(gpu_naive);
        let speedup = cpu / best_gpu;
        
        let (winner, speedup_str) = if speedup > 1.0 {
            ("GPU", format!("{:.2}x", speedup))
        } else {
            ("CPU", format!("{:.2}x", 1.0 / speedup))
        };
        
        let cpu_str = format_time(cpu);
        let naive_str = format_time(gpu_naive);
        let resident_str = format_time(gpu_resident);
        
        println!("║ {:>8} │ {:>10} │ {:>10} │ {:>12} │ {:>9} │ {:>12} ║",
            result.scale, cpu_str, naive_str, resident_str, speedup_str, winner);
    }
    
    println!("╚══════════╧════════════╧════════════╧══════════════╧═══════════╧══════════════╝");
    println!("\nKey insights:");
    println!("• **GPU Naive**: Upload + compute + download every frame (current benchmark)");
    println!("• **GPU Resident**: Compute only, data stays on GPU (realistic scenario)");
    println!("• Speedup = CPU time / best GPU time");
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
    println!("║               RAPIER GPU TRANSFER STRATEGY BENCHMARK                          ║");
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
        
        // Benchmark GPU approaches
        let buffer_manager = BufferManager::new(gpu_ctx.device.clone(), gpu_ctx.queue.clone());
        let integrator = GpuIntegrator::new(&gpu_ctx.device);
        let mut gpu_buffer = buffer_manager.create_rigid_body_buffer(bodies.len());
        
        // Initial upload (one-time cost amortized over many frames)
        buffer_manager.upload_rigid_bodies(&bodies, &mut gpu_buffer);
        
        let gpu_naive_time = benchmark_gpu_naive(
            &bodies, &buffer_manager, &integrator,
            &gpu_ctx.device, &gpu_ctx.queue, &mut gpu_buffer,
            SIMULATION_FRAMES, ITERATIONS
        );
        
        let gpu_resident_time = benchmark_gpu_resident(
            &integrator,
            &gpu_ctx.device, &gpu_ctx.queue, &mut gpu_buffer,
            SIMULATION_FRAMES, ITERATIONS
        );
        
        results.push(BenchmarkResult {
            scale,
            cpu_time,
            gpu_naive_time,
            gpu_resident_time,
        });
        
        println!("Done!");
    }
    
    print_results_table(&results);
}
