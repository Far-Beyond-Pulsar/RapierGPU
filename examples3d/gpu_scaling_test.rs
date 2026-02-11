//! Minimal GPU compute test to verify shader execution

use rapier3d::prelude::*;
use rapier3d::gpu::{GpuContext, BufferManager, GpuIntegrator, wgpu};

fn main() {
    println!("🔬 GPU Compute Scaling Test\n");
    
    let gpu_ctx = GpuContext::new().expect("GPU required");
    println!("GPU: {}\n", gpu_ctx.adapter.get_info().name);
    
    let scales = [10, 100, 1_000, 10_000, 100_000, 1_000_000];
    
    for &count in &scales {
        let mut bodies = RigidBodySet::new();
        for i in 0..count {
            let mass_props = MassProperties::new(
                Vector::new(0.0, 0.0, 0.0),
                1.0,
                Vector::new(1.0, 1.0, 1.0)
            );
            
            let body = RigidBodyBuilder::dynamic()
                .translation(Vector::new(i as Real, 10.0, 0.0))
                .linvel(Vector::new(0.0, -1.0, 0.0))
                .additional_mass_properties(mass_props)
                .build();
            bodies.insert(body);
        }
        
        let buffer_manager = BufferManager::new(gpu_ctx.device.clone(), gpu_ctx.queue.clone());
        let integrator = GpuIntegrator::new(&gpu_ctx.device);
        let mut gpu_buffer = buffer_manager.create_rigid_body_buffer(count);
        
        buffer_manager.upload_rigid_bodies(&bodies, &mut gpu_buffer);
        
        // Run 100 iterations and time
        let start = std::time::Instant::now();
        for _ in 0..100 {
            integrator.integrate(&gpu_ctx.device, &gpu_ctx.queue, &mut gpu_buffer, 1.0/60.0, [0.0, -9.81, 0.0], 0.0, 0.0, true);
        }
        gpu_ctx.device.poll(wgpu::Maintain::Wait);
        let elapsed = start.elapsed();
        
        let workgroups = (count + 255) / 256;
        let avg_us = elapsed.as_micros() as f64 / 100.0;
        let per_body_ns = (avg_us * 1000.0) / count as f64;
        
        println!("{:>8} bodies | {:>4} workgroups | {:>8.2} µs/iter | {:>6.2} ns/body",
            count, workgroups, avg_us, per_body_ns);
    }
    
    println!("\n✅ If ns/body is constant (~same for all counts), GPU is working!");
    println!("❌ If µs/iter is constant, GPU isn't scaling (driver overhead only)");
}
