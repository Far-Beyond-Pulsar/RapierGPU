//! Debug GPU benchmark - adds profiling to see what's happening

use rapier3d::prelude::*;
use rapier3d::gpu::{GpuContext, BufferManager, GpuIntegrator, wgpu};
use rapier3d::na::Point3;
use std::time::Instant;

fn main() {
    println!("🔍 GPU Integration Debug Benchmark\n");
    
    let gpu_ctx = GpuContext::new().expect("GPU required");
    println!("✓ GPU: {}\n", gpu_ctx.adapter.get_info().name);
    
    let scales = [100, 1000, 10000, 100000];
    
    for &count in &scales {
        println!("════════════════════════════════════");
        println!("Testing {} bodies:", count);
        println!("════════════════════════════════════");
        
        let mut bodies = RigidBodySet::new();
        for i in 0..count {
            // Create mass properties manually since bodies have no colliders
            let mass_props = MassProperties::new(
                Vector::new(0.0, 0.0, 0.0),  // center of mass
                1.0,  // mass
                Vector::new(1.0, 1.0, 1.0)  // principal inertia
            );
            
            let body = RigidBodyBuilder::dynamic()
                .translation(Vector::new(i as Real, 10.0, 0.0))
                .linvel(Vector::new(0.0, -1.0, 0.0))
                .additional_mass_properties(mass_props)
                .build();
            bodies.insert(body);
        }
        
        // Check first body has non-zero mass
        let first = bodies.iter().next().unwrap().1;
        println!("First body inv_mass: {}", first.mass_properties().local_mprops.inv_mass);
        println!("First body position: {:?}", first.translation());
        println!("First body velocity: {:?}", first.linvel());
        
        let buffer_manager = BufferManager::new(gpu_ctx.device.clone(), gpu_ctx.queue.clone());
        let integrator = GpuIntegrator::new(&gpu_ctx.device);
        let mut gpu_buffer = buffer_manager.create_rigid_body_buffer(count);
        
        // Upload
        let t0 = Instant::now();
        buffer_manager.upload_rigid_bodies(&bodies, &mut gpu_buffer);
        let upload_time = t0.elapsed();
        
        // Compute (single frame)
        let t1 = Instant::now();
        integrator.integrate(&gpu_ctx.device, &gpu_ctx.queue, &mut gpu_buffer, 1.0/60.0, [0.0, -9.81, 0.0], 0.0, 0.0);
        gpu_ctx.device.poll(wgpu::Maintain::Wait);
        let compute_time = t1.elapsed();
        
        // Download
        let t2 = Instant::now();
        let (positions, velocities) = buffer_manager.download_rigid_bodies(&gpu_buffer);
        let download_time = t2.elapsed();
        
        println!("Upload:   {:?}", upload_time);
        println!("Compute:  {:?}", compute_time);
        println!("Download: {:?}", download_time);
        println!("Total:    {:?}", upload_time + compute_time + download_time);
        
        // Verify results
        if count <= 1000 {
            println!("\nFirst 3 positions after integration:");
            for i in 0..3.min(positions.len()) {
                println!("  Body {}: pos=({:.3}, {:.3}, {:.3}), vel=({:.3}, {:.3}, {:.3})",
                    i,
                    positions[i].x, positions[i].y, positions[i].z,
                    velocities[i].x, velocities[i].y, velocities[i].z
                );
            }
        }
        
        println!();
    }
}
