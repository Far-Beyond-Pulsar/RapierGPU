//! GPU acceleration module using WGPU compute shaders.
//!
//! This module provides GPU-accelerated implementations of compute-intensive
//! physics operations including collision detection, constraint solving, and
//! integration. It's designed as an optional, incremental enhancement that
//! coexists with the existing CPU implementation.
//!
//! # Features
//!
//! Enable GPU acceleration with the `gpu-acceleration` feature flag:
//! ```toml
//! rapier3d = { version = "0.32", features = ["gpu-acceleration"] }
//! ```
//!
//! # Architecture
//!
//! - **GpuContext**: Manages WGPU device and queue lifecycle
//! - **BufferManager**: Handles CPU↔GPU data transfers
//! - **GpuIntegrator**: Position/velocity integration on GPU
//! - **GpuBroadPhase**: BVH-based collision detection (future)
//! - **GpuConstraintSolver**: Constraint resolution (future)
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "gpu-acceleration")]
//! # {
//! use rapier3d::prelude::*;
//! use rapier3d::gpu::GpuContext;
//!
//! // Initialize GPU context
//! let gpu_ctx = GpuContext::new().expect("Failed to initialize GPU");
//!
//! // Use in physics pipeline (future API)
//! let pipeline = PhysicsPipeline::new()
//!     .with_gpu_context(gpu_ctx);
//! # }
//! ```

#[cfg(feature = "gpu-acceleration")]
mod device;
#[cfg(feature = "gpu-acceleration")]
mod buffer_manager;
#[cfg(feature = "gpu-acceleration")]
mod pipeline;
#[cfg(feature = "gpu-acceleration")]
mod integrator;
#[cfg(feature = "gpu-acceleration")]
mod tests;

#[cfg(feature = "gpu-acceleration")]
pub use device::GpuContext;
#[cfg(feature = "gpu-acceleration")]
pub use buffer_manager::{BufferManager, RigidBodyGpuBuffer};
#[cfg(feature = "gpu-acceleration")]
pub use pipeline::GpuComputePipeline;
#[cfg(feature = "gpu-acceleration")]
pub use integrator::GpuIntegrator;

/// Re-export WGPU types for convenience
#[cfg(feature = "gpu-acceleration")]
pub use wgpu;
