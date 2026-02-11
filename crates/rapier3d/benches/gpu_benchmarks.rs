//! GPU vs CPU benchmarks: Direct comparison with summary table.
//!
//! Compares CPU and GPU performance for physics operations
//! and prints an easy-to-read comparison table.

use rapier3d::prelude::*;
use rapier3d::gpu::{GpuContext, BufferManager};
use std::time::Instant;

/// Test scales: body counts
const SCALES: &[usize] = &[10, 50, 100, 500, 1_000, 5_000, 10_000];
const ITERATIONS: usize = 10; // Number of iterations to average

struct BenchmarkResult {
    scale: usize,
    cpu_time_us: f64,
    gpu_time_us: f64,
    speedup: f64,
}

/// Helper to create a scene with N bodies in a grid.
fn create_test_bodies(count: usize) -> RigidBodySet {
    let mut bodies = RigidBodySet::new();
    let side = (count as f32).cbrt().ceil() as usize;
    let spacing = 2.5;
    
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                if bodies.len() >= count {
                    break;
                }
                
                let pos = Vector::new(
                    i as Real * spacing,
                    j as Real * spacing,
                    k as Real * spacing,
                );
                
                let rb = RigidBodyBuilder::dynamic()
                    .translation(pos)
                    .linvel(Vector::new(
                        (i as Real - side as Real / 2.0) * 0.1,
                        -1.0,
                        (k as Real - side as Real / 2.0) * 0.1,
                    ))
                    .build();
                
                bodies.insert(rb);
            }
            if bodies.len() >= count {
                break;
            }
        }
        if bodies.len() >= count {
            break;
        }
    }
    
    bodies
}

fn benchmark_cpu_integration(bodies: &RigidBodySet, iterations: usize) -> f64 {
    let dt = 1.0 / 60.0;
    let mut bodies_copy = bodies.clone();
    
    let start = Instant::now();
    for _ in 0..iterations {
        for (_handle, body) in bodies_copy.iter_mut() {
            if body.is_dynamic() {
                let mut pos = body.position().clone();
                let linvel = body.linvel();
                
                pos.translation.x += linvel.x * dt;
                pos.translation.y += linvel.y * dt;
                pos.translation.z += linvel.z * dt;
                
                body.set_position(pos, false);
            }
        }
    }
    let elapsed = start.elapsed();
    
    elapsed.as_micros() as f64 / iterations as f64
}

fn benchmark_gpu_transfer(bodies: &RigidBodySet, buffer_manager: &BufferManager, gpu_buffer: &mut rapier3d::gpu::RigidBodyGpuBuffer, iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        buffer_manager.upload_rigid_bodies(bodies, gpu_buffer);
        let (_positions, _velocities) = buffer_manager.download_rigid_bodies(gpu_buffer);
    }
    let elapsed = start.elapsed();
    
    elapsed.as_micros() as f64 / iterations as f64
}

fn print_results_table(results: &[BenchmarkResult]) {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    CPU vs GPU PERFORMANCE COMPARISON                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Bodies  │    CPU Time    │    GPU Time    │   Speedup   │      Winner       ║");
    println!("╠══════════╪════════════════╪════════════════╪═════════════╪═══════════════════╣");
    
    for result in results {
        let winner = if result.speedup > 1.0 {
            format!("GPU {:>6.2}x faster", result.speedup)
        } else if result.speedup < 0.95 {
            format!("CPU {:>6.2}x faster", 1.0 / result.speedup)
        } else {
            "  ~Same speed   ".to_string()
        };
        
        let cpu_str = format_time(result.cpu_time_us);
        let gpu_str = format_time(result.gpu_time_us);
        
        println!("║ {:>8} │ {:>14} │ {:>14} │ {:>11.2}x │ {:>17} ║", 
                 result.scale, 
                 cpu_str,
                 gpu_str,
                 result.speedup,
                 winner);
    }
    
    println!("╚══════════╧════════════════╧════════════════╧═════════════╧═══════════════════╝");
    println!("\nNote: GPU times include CPU↔GPU transfer overhead.");
    println!("      Actual GPU compute will be added in Phase 2.");
}

fn format_time(us: f64) -> String {
    if us < 1_000.0 {
        format!("{:>10.2} µs", us)
    } else if us < 1_000_000.0 {
        format!("{:>10.2} ms", us / 1_000.0)
    } else {
        format!("{:>10.2} s", us / 1_000_000.0)
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                       RAPIER GPU ACCELERATION BENCHMARK                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝\n");
    
    println!("Initializing GPU...");
    let gpu_setup = match GpuContext::new() {
        Ok(ctx) => {
            println!("✓ GPU initialized: {}", ctx.adapter.get_info().name);
            let buffer_manager = BufferManager::new(ctx.device, ctx.queue);
            Some(buffer_manager)
        },
        Err(e) => {
            println!("✗ GPU not available: {:?}", e);
            println!("  Running CPU-only benchmarks...\n");
            None
        }
    };
    
    let mut results = Vec::new();
    
    for &scale in SCALES {
        print!("Benchmarking {} bodies... ", scale);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        
        let bodies = create_test_bodies(scale);
        
        // Benchmark CPU
        let cpu_time = benchmark_cpu_integration(&bodies, ITERATIONS);
        
        // Benchmark GPU if available
        let gpu_time = if let Some(ref buffer_manager) = gpu_setup {
            let mut gpu_buffer = buffer_manager.create_rigid_body_buffer(bodies.len());
            benchmark_gpu_transfer(&bodies, buffer_manager, &mut gpu_buffer, ITERATIONS)
        } else {
            0.0
        };
        
        let speedup = if gpu_time > 0.0 {
            cpu_time / gpu_time
        } else {
            0.0
        };
        
        results.push(BenchmarkResult {
            scale,
            cpu_time_us: cpu_time,
            gpu_time_us: gpu_time,
            speedup,
        });
        
        println!("Done!");
    }
    
    print_results_table(&results);
}
