//! GPU vs CPU benchmarks at multiple scales.
//!
//! This benchmark suite compares CPU and GPU performance across 10 different
//! data scales to identify crossover points and validate GPU acceleration benefits.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rapier3d::prelude::*;
use rapier3d::gpu::{GpuContext, BufferManager};

/// Test scales: body counts from 10 to 500,000
const SCALES: &[usize] = &[
    10,
    50,
    100,
    500,
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    500_000,
];

/// Helper to create a scene with N bodies in a grid.
fn create_test_bodies(count: usize) -> RigidBodySet {
    let mut bodies = RigidBodySet::new();
    
    // Calculate grid dimensions (cube root for 3D)
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

/// Benchmark: Buffer upload (CPU → GPU transfer).
fn benchmark_buffer_upload(c: &mut Criterion) {
    let gpu_ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping GPU benchmarks: {:?}", e);
            return;
        }
    };
    
    let buffer_manager = BufferManager::new(gpu_ctx.device, gpu_ctx.queue);
    
    let mut group = c.benchmark_group("buffer_upload");
    
    for &scale in SCALES {
        let bodies = create_test_bodies(scale);
        
        group.throughput(Throughput::Elements(scale as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(scale),
            &scale,
            |b, _| {
                let mut gpu_buffer = buffer_manager.create_rigid_body_buffer(bodies.len());
                b.iter(|| {
                    buffer_manager.upload_rigid_bodies(&bodies, &mut gpu_buffer);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Memory allocation (GPU buffer creation).
fn benchmark_buffer_allocation(c: &mut Criterion) {
    let gpu_ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping GPU benchmarks: {:?}", e);
            return;
        }
    };
    
    let buffer_manager = BufferManager::new(gpu_ctx.device, gpu_ctx.queue);
    
    let mut group = c.benchmark_group("buffer_allocation");
    
    for &scale in SCALES {
        group.throughput(Throughput::Elements(scale as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(scale),
            &scale,
            |b, &capacity| {
                b.iter(|| {
                    buffer_manager.create_rigid_body_buffer(capacity);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark: CPU body iteration (baseline).
fn benchmark_cpu_body_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_body_iteration");
    
    for &scale in SCALES {
        let bodies = create_test_bodies(scale);
        
        group.throughput(Throughput::Elements(scale as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(scale),
            &scale,
            |b, _| {
                b.iter(|| {
                    let mut sum = Vector::new(0.0, 0.0, 0.0);
                    for (_handle, body) in bodies.iter() {
                        sum += body.linvel();
                    }
                    sum
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Full CPU → GPU → CPU roundtrip.
fn benchmark_roundtrip(c: &mut Criterion) {
    let gpu_ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping GPU benchmarks: {:?}", e);
            return;
        }
    };
    
    let buffer_manager = BufferManager::new(gpu_ctx.device, gpu_ctx.queue);
    
    let mut group = c.benchmark_group("roundtrip");
    
    for &scale in SCALES {
        let bodies = create_test_bodies(scale);
        
        group.throughput(Throughput::Elements(scale as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(scale),
            &scale,
            |b, _| {
                let mut gpu_buffer = buffer_manager.create_rigid_body_buffer(bodies.len());
                b.iter(|| {
                    // Upload
                    buffer_manager.upload_rigid_bodies(&bodies, &mut gpu_buffer);
                    
                    // TODO: GPU compute goes here
                    
                    // TODO: Download (not yet implemented)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Compare CPU vs GPU at critical scales.
fn benchmark_critical_comparison(c: &mut Criterion) {
    // Focus on the scales where CPU/GPU crossover is most interesting
    let critical_scales = &[100, 1_000, 10_000];
    
    for &scale in critical_scales {
        let mut group = c.benchmark_group(format!("comparison_{}", scale));
        let bodies = create_test_bodies(scale);
        
        group.throughput(Throughput::Elements(scale as u64));
        
        // CPU baseline
        group.bench_function("cpu", |b| {
            b.iter(|| {
                let mut sum = Vector::new(0.0, 0.0, 0.0);
                for (_handle, body) in bodies.iter() {
                    sum += body.linvel();
                }
                sum
            });
        });
        
        // GPU (if available)
        if let Ok(gpu_ctx) = GpuContext::new() {
            let buffer_manager = BufferManager::new(gpu_ctx.device, gpu_ctx.queue);
            
            group.bench_function("gpu_upload", |b| {
                let mut gpu_buffer = buffer_manager.create_rigid_body_buffer(bodies.len());
                b.iter(|| {
                    buffer_manager.upload_rigid_bodies(&bodies, &mut gpu_buffer);
                });
            });
        }
        
        group.finish();
    }
}

criterion_group!(
    benches,
    benchmark_buffer_upload,
    benchmark_buffer_allocation,
    benchmark_cpu_body_iteration,
    benchmark_roundtrip,
    benchmark_critical_comparison,
);

criterion_main!(benches);

