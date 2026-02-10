//! GPU-resident physics state manager.
//! 
//! This module implements the PhysX-style architecture where rigid body
//! state lives permanently on the GPU. Only deltas (new/changed/removed
//! bodies) are transferred, minimizing CPU↔GPU bandwidth.

use crate::dynamics::{RigidBodyHandle, RigidBodySet};
use super::buffer_manager::{BufferManager, RigidBodyGpuBuffer};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Tracks which bodies have been modified and need GPU updates.
#[derive(Default)]
struct DirtyTracker {
    /// Bodies added since last sync
    added: HashSet<RigidBodyHandle>,
    /// Bodies removed since last sync
    removed: HashSet<RigidBodyHandle>,
    /// Bodies whose state changed (forces, velocities, etc.)
    modified: HashSet<RigidBodyHandle>,
}

impl DirtyTracker {
    fn clear(&mut self) {
        self.added.clear();
        self.removed.clear();
        self.modified.clear();
    }
    
    fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.modified.is_empty()
    }
}

/// GPU-resident rigid body state with delta tracking.
///
/// Architecture:
/// - Body data lives on GPU permanently
/// - CPU tracks which bodies are dirty (added/modified/removed)
/// - `sync_to_gpu()` uploads only deltas
/// - `readback_for_rendering()` downloads only positions/rotations
pub struct GpuResidentState {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffer_manager: BufferManager,
    
    /// GPU buffers (persistent)
    gpu_buffer: Option<RigidBodyGpuBuffer>,
    
    /// Maps RigidBodyHandle → GPU buffer index
    handle_to_index: HashMap<RigidBodyHandle, usize>,
    
    /// Dirty tracking for incremental updates
    dirty: DirtyTracker,
    
    /// Current capacity
    capacity: usize,
    
    /// Total bodies currently on GPU
    gpu_body_count: usize,
}

impl GpuResidentState {
    /// Create a new GPU-resident state manager.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, initial_capacity: usize) -> Self {
        let buffer_manager = BufferManager::new(device.clone(), queue.clone());
        let gpu_buffer = Some(buffer_manager.create_rigid_body_buffer(initial_capacity));
        
        Self {
            device,
            queue,
            buffer_manager,
            gpu_buffer,
            handle_to_index: HashMap::new(),
            dirty: DirtyTracker::default(),
            capacity: initial_capacity,
            gpu_body_count: 0,
        }
    }
    
    /// Mark a body as added (will be uploaded on next sync).
    pub fn mark_added(&mut self, handle: RigidBodyHandle) {
        self.dirty.added.insert(handle);
    }
    
    /// Mark a body as modified (will be re-uploaded on next sync).
    pub fn mark_modified(&mut self, handle: RigidBodyHandle) {
        if !self.dirty.added.contains(&handle) {
            self.dirty.modified.insert(handle);
        }
    }
    
    /// Mark a body as removed (will be deleted from GPU on next sync).
    pub fn mark_removed(&mut self, handle: RigidBodyHandle) {
        self.dirty.removed.insert(handle);
        self.dirty.added.remove(&handle);
        self.dirty.modified.remove(&handle);
    }
    
    /// Sync dirty bodies to GPU (incremental upload).
    ///
    /// This is the key optimization: only upload what changed!
    pub fn sync_to_gpu(&mut self, bodies: &RigidBodySet) {
        if self.dirty.is_empty() {
            return; // Nothing to do!
        }
        
        // Handle removals
        for handle in &self.dirty.removed {
            if let Some(index) = self.handle_to_index.remove(handle) {
                // Swap-remove: move last body to this slot
                if index < self.gpu_body_count - 1 {
                    // Find handle of last body and update its index
                    let last_handle = self.handle_to_index
                        .iter()
                        .find(|(_, &idx)| idx == self.gpu_body_count - 1)
                        .map(|(h, _)| *h);
                    
                    if let Some(last_handle) = last_handle {
                        self.handle_to_index.insert(last_handle, index);
                        // Mark last body as modified so it gets copied to new slot
                        self.dirty.modified.insert(last_handle);
                    }
                }
                self.gpu_body_count -= 1;
            }
        }
        
        // Handle additions
        for handle in &self.dirty.added {
            if self.gpu_body_count >= self.capacity {
                // Resize GPU buffers
                self.resize_gpu_buffers(self.capacity * 2);
            }
            
            let index = self.gpu_body_count;
            self.handle_to_index.insert(*handle, index);
            self.gpu_body_count += 1;
        }
        
        // Upload modified + added bodies
        let mut bodies_to_upload = self.dirty.added.iter()
            .chain(self.dirty.modified.iter())
            .filter_map(|h| bodies.get(*h).map(|b| (*h, b)))
            .collect::<Vec<_>>();
        
        if !bodies_to_upload.is_empty() {
            // TODO: Implement partial buffer upload (currently uploads all)
            // For now, upload entire buffer (suboptimal but works)
            self.buffer_manager.upload_rigid_bodies(bodies, self.gpu_buffer.as_mut().unwrap());
        }
        
        self.dirty.clear();
    }
    
    /// Readback positions/rotations for rendering (minimal data transfer).
    ///
    /// Only downloads what's needed for visualization - doesn't sync full state.
    pub fn readback_for_rendering(&self) -> (Vec<super::buffer_manager::GpuVector3>, Vec<super::buffer_manager::GpuVector3>) {
        if let Some(ref gpu_buffer) = self.gpu_buffer {
            self.buffer_manager.download_rigid_bodies(gpu_buffer)
        } else {
            (Vec::new(), Vec::new())
        }
    }
    
    /// Get GPU buffer for compute operations.
    pub fn gpu_buffer_mut(&mut self) -> Option<&mut RigidBodyGpuBuffer> {
        self.gpu_buffer.as_mut()
    }
    
    /// Get current GPU body count.
    pub fn body_count(&self) -> usize {
        self.gpu_body_count
    }
    
    /// Resize GPU buffers to new capacity.
    fn resize_gpu_buffers(&mut self, new_capacity: usize) {
        log::info!("Resizing GPU buffers: {} → {}", self.capacity, new_capacity);
        
        // Create new larger buffer
        let new_buffer = self.buffer_manager.create_rigid_body_buffer(new_capacity);
        
        // TODO: Copy old buffer data to new buffer on GPU (avoid CPU roundtrip)
        // For now, mark all bodies as modified to trigger re-upload
        for handle in self.handle_to_index.keys() {
            self.dirty.modified.insert(*handle);
        }
        
        self.gpu_buffer = Some(new_buffer);
        self.capacity = new_capacity;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dirty_tracking() {
        let mut tracker = DirtyTracker::default();
        let handle = RigidBodyHandle::from_raw_parts(0, 0);
        
        tracker.added.insert(handle);
        assert!(!tracker.is_empty());
        
        tracker.clear();
        assert!(tracker.is_empty());
    }
}
