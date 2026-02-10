//! GPU validation tests - compare GPU results against CPU reference.
//!
//! This module provides a testing framework to ensure GPU-accelerated
//! physics produces identical (or numerically equivalent) results to
//! the CPU implementation.

#[cfg(all(test, feature = "gpu-acceleration"))]
mod tests {
    use crate::dynamics::{RigidBodyBuilder, RigidBodySet};
    use crate::geometry::{ColliderBuilder, ColliderSet};
    use crate::gpu::{GpuContext, BufferManager};
    use crate::math::{Real, Vector};
    use approx::assert_relative_eq;

    /// Tolerance for floating-point comparisons (CPU vs GPU).
    const TOLERANCE: Real = 1e-5;

    /// Helper to create a test scene with falling bodies.
    fn create_test_scene() -> (RigidBodySet, ColliderSet) {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();

        // Create a dynamic body
        let rb = RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 10.0, 0.0))
            .linvel(Vector::new(1.0, -2.0, 0.5))
            .build();
        let rb_handle = bodies.insert(rb);

        // Add a collider
        #[cfg(feature = "dim3")]
        let collider = ColliderBuilder::ball(0.5).build();
        #[cfg(feature = "dim2")]
        let collider = ColliderBuilder::ball(0.5).build();
        
        colliders.insert_with_parent(collider, rb_handle, &mut bodies);

        // Create another dynamic body
        let rb2 = RigidBodyBuilder::dynamic()
            .translation(Vector::new(2.0, 5.0, -1.0))
            .linvel(Vector::new(-0.5, 1.0, 0.0))
            .build();
        let rb2_handle = bodies.insert(rb2);

        #[cfg(feature = "dim3")]
        let collider2 = ColliderBuilder::cuboid(0.5, 0.5, 0.5).build();
        #[cfg(feature = "dim2")]
        let collider2 = ColliderBuilder::cuboid(0.5, 0.5).build();
        
        colliders.insert_with_parent(collider2, rb2_handle, &mut bodies);

        (bodies, colliders)
    }

    /// Helper to compare two rigid body sets for equality.
    fn assert_bodies_equivalent(cpu_bodies: &RigidBodySet, gpu_bodies: &RigidBodySet) {
        assert_eq!(
            cpu_bodies.len(),
            gpu_bodies.len(),
            "Body count mismatch"
        );

        for (handle, cpu_body) in cpu_bodies.iter() {
            let gpu_body = gpu_bodies
                .get(handle)
                .expect("GPU missing body that CPU has");

            // Compare positions
            let cpu_pos = cpu_body.position().translation.vector;
            let gpu_pos = gpu_body.position().translation.vector;
            
            #[cfg(feature = "dim3")]
            {
                assert_relative_eq!(cpu_pos.x, gpu_pos.x, epsilon = TOLERANCE);
                assert_relative_eq!(cpu_pos.y, gpu_pos.y, epsilon = TOLERANCE);
                assert_relative_eq!(cpu_pos.z, gpu_pos.z, epsilon = TOLERANCE);
            }
            #[cfg(feature = "dim2")]
            {
                assert_relative_eq!(cpu_pos.x, gpu_pos.x, epsilon = TOLERANCE);
                assert_relative_eq!(cpu_pos.y, gpu_pos.y, epsilon = TOLERANCE);
            }

            // Compare velocities
            let cpu_vel = cpu_body.linvel();
            let gpu_vel = gpu_body.linvel();
            
            #[cfg(feature = "dim3")]
            {
                assert_relative_eq!(cpu_vel.x, gpu_vel.x, epsilon = TOLERANCE);
                assert_relative_eq!(cpu_vel.y, gpu_vel.y, epsilon = TOLERANCE);
                assert_relative_eq!(cpu_vel.z, gpu_vel.z, epsilon = TOLERANCE);
            }
            #[cfg(feature = "dim2")]
            {
                assert_relative_eq!(cpu_vel.x, gpu_vel.x, epsilon = TOLERANCE);
                assert_relative_eq!(cpu_vel.y, gpu_vel.y, epsilon = TOLERANCE);
            }

            // Compare angular velocity
            #[cfg(feature = "dim3")]
            {
                let cpu_angvel = cpu_body.angvel();
                let gpu_angvel = gpu_body.angvel();
                assert_relative_eq!(cpu_angvel.x, gpu_angvel.x, epsilon = TOLERANCE);
                assert_relative_eq!(cpu_angvel.y, gpu_angvel.y, epsilon = TOLERANCE);
                assert_relative_eq!(cpu_angvel.z, gpu_angvel.z, epsilon = TOLERANCE);
            }
            #[cfg(feature = "dim2")]
            {
                let cpu_angvel = cpu_body.angvel();
                let gpu_angvel = gpu_body.angvel();
                assert_relative_eq!(*cpu_angvel, *gpu_angvel, epsilon = TOLERANCE);
            }
        }
    }

    #[test]
    fn test_gpu_context_initialization() {
        // Test that we can initialize GPU context
        let gpu_ctx = GpuContext::new();
        
        if gpu_ctx.is_err() {
            println!("Skipping GPU test - no suitable GPU found");
            return;
        }

        let gpu_ctx = gpu_ctx.unwrap();
        let info = gpu_ctx.adapter_info();
        println!("GPU: {} ({:?})", info.name, info.backend);
    }

    #[test]
    fn test_buffer_upload_download() {
        // Test that we can upload and download data to GPU
        let gpu_ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping GPU test - no suitable GPU found");
                return;
            }
        };

        let (bodies, _colliders) = create_test_scene();
        
        let buffer_manager = BufferManager::new(
            gpu_ctx.device.clone(),
            gpu_ctx.queue.clone()
        );

        let mut gpu_buffer = buffer_manager.create_rigid_body_buffer(bodies.len());
        
        // Upload to GPU
        buffer_manager.upload_rigid_bodies(&bodies, &mut gpu_buffer);
        
        println!(
            "Uploaded {} bodies to GPU buffer (capacity: {})",
            gpu_buffer.body_count,
            gpu_buffer.capacity
        );
        
        // TODO: Add download and comparison when we implement readback
        assert_eq!(gpu_buffer.body_count, bodies.len());
    }

    #[test]
    #[ignore] // Enable once integration kernel is implemented
    fn test_integration_cpu_vs_gpu() {
        // This test will compare CPU integration against GPU integration
        let gpu_ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping GPU test - no suitable GPU found");
                return;
            }
        };

        let (mut cpu_bodies, _colliders) = create_test_scene();
        let (mut gpu_bodies, _) = create_test_scene();

        let dt = 0.016; // 60 FPS

        // TODO: Run CPU integration
        // for (_handle, body) in cpu_bodies.iter_mut() {
        //     body.integrate_forces(dt);
        // }

        // TODO: Run GPU integration
        // let buffer_manager = BufferManager::new(
        //     gpu_ctx.device.clone(),
        //     gpu_ctx.queue.clone()
        // );
        // gpu_integrator.integrate(&mut gpu_bodies, dt);

        // Compare results
        assert_bodies_equivalent(&cpu_bodies, &gpu_bodies);
    }

    #[test]
    #[ignore] // Enable once collision detection is implemented
    fn test_collision_detection_cpu_vs_gpu() {
        let gpu_ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping GPU test - no suitable GPU found");
                return;
            }
        };

        // TODO: Create scene with colliding bodies
        // TODO: Run CPU collision detection
        // TODO: Run GPU collision detection
        // TODO: Compare contact manifolds
        
        println!("GPU collision detection test (not yet implemented)");
    }

    #[test]
    #[ignore] // Enable once constraint solver is implemented
    fn test_solver_cpu_vs_gpu() {
        let gpu_ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping GPU test - no suitable GPU found");
                return;
            }
        };

        // TODO: Create scene with constraints
        // TODO: Run CPU solver
        // TODO: Run GPU solver
        // TODO: Compare constraint impulses and body velocities
        
        println!("GPU solver test (not yet implemented)");
    }

    #[test]
    #[ignore] // Enable for long-running stability tests
    fn test_long_simulation_cpu_vs_gpu() {
        // Run a 10-second simulation and ensure CPU and GPU stay in sync
        let gpu_ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping GPU test - no suitable GPU found");
                return;
            }
        };

        let (mut cpu_bodies, cpu_colliders) = create_test_scene();
        let (mut gpu_bodies, gpu_colliders) = create_test_scene();

        let dt = 0.016; // 60 FPS
        let steps = 600; // 10 seconds

        for step in 0..steps {
            // TODO: Step CPU physics
            // TODO: Step GPU physics
            
            // Compare every 60 frames (1 second)
            if step % 60 == 0 {
                println!("Comparing at step {}", step);
                assert_bodies_equivalent(&cpu_bodies, &gpu_bodies);
            }
        }

        // Final comparison
        assert_bodies_equivalent(&cpu_bodies, &gpu_bodies);
    }

    #[test]
    fn test_stress_test_many_bodies() {
        // Test with 1000+ bodies to ensure GPU scales
        let gpu_ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("Skipping GPU test - no suitable GPU found");
                return;
            }
        };

        let mut bodies = RigidBodySet::new();
        
        // Create 1000 bodies in a grid
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    #[cfg(feature = "dim3")]
                    let pos = Vector::new(i as Real * 2.0, j as Real * 2.0, k as Real * 2.0);
                    #[cfg(feature = "dim2")]
                    let pos = Vector::new(i as Real * 2.0, j as Real * 2.0);
                    
                    let rb = RigidBodyBuilder::dynamic()
                        .translation(pos)
                        .build();
                    bodies.insert(rb);
                }
            }
        }

        println!("Created {} bodies", bodies.len());

        let buffer_manager = BufferManager::new(
            gpu_ctx.device.clone(),
            gpu_ctx.queue.clone()
        );

        let mut gpu_buffer = buffer_manager.create_rigid_body_buffer(bodies.len());
        buffer_manager.upload_rigid_bodies(&bodies, &mut gpu_buffer);

        assert_eq!(gpu_buffer.body_count, bodies.len());
        println!("Successfully uploaded {} bodies to GPU", gpu_buffer.body_count);
    }
}
