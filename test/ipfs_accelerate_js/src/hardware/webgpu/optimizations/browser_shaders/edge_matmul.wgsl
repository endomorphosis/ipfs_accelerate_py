/**
 * Edge-optimized matrix multiplication shader
 * 
 * Optimized for Edge's WebGPU implementation with:
 * - Workgroup size of 16x16 similar to Chrome but with Edge-specific enhancements
 * - Memory access patterns optimized for Edge's memory handling
 * - Hybrid approach combining Chrome's strengths with Edge-specific patterns
 */

struct Params {
  M: u32,
  N: u32,
  K: u32,
  batch_size: u32,
  alpha: f32,
  beta: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

// Define workgroup size based on Edge's optimal parameters
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  
  // Declare shared memory tiles - Edge benefits from square tiles like Chrome
  var<workgroup> A_shared: array<f32, 16 * 16>;
  var<workgroup> B_shared: array<f32, 16 * 16>;

  // Get indices
  let batch_idx = global_id.z;
  let row = global_id.x;
  let col = global_id.y;
  
  // Guard against out-of-bounds work items
  if (row >= params.M || col >= params.N || batch_idx >= params.batch_size) {
    return;
  }
  
  // Calculate base indices for batch
  let a_batch_offset = batch_idx * params.M * params.K;
  let b_batch_offset = batch_idx * params.K * params.N;
  let c_batch_offset = batch_idx * params.M * params.N;
  
  // Initialize sum - will accumulate dot product
  var sum: f32 = 0.0;
  
  // Calculate number of tiles needed
  let num_tiles = (params.K + 16u - 1u) / 16u;
  
  // Process each tile
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    // Calculate start of current tile
    let tile_start = t * 16u;
    
    // Collaborative loading of tiles into shared memory
    // Edge performs better with this simpler loading pattern
    let A_row = row;
    let A_col = tile_start + local_id.y;
    
    if (A_row < params.M && A_col < params.K) {
      A_shared[local_id.x * 16u + local_id.y] = A[a_batch_offset + A_row * params.K + A_col];
    } else {
      A_shared[local_id.x * 16u + local_id.y] = 0.0;
    }
    
    let B_row = tile_start + local_id.x;
    let B_col = col;
    
    if (B_row < params.K && B_col < params.N) {
      B_shared[local_id.x * 16u + local_id.y] = B[b_batch_offset + B_row * params.N + B_col];
    } else {
      B_shared[local_id.x * 16u + local_id.y] = 0.0;
    }
    
    // Ensure all threads have loaded their values
    workgroupBarrier();
    
    // Edge benefits from a hybrid approach - partially unrolled loops (pairs)
    var k = 0u;
    
    // Process pairs of elements - good compromise for Edge
    for (; k + 2u <= 16u && tile_start + k < params.K; k = k + 2u) {
      let a0 = A_shared[local_id.x * 16u + k];
      let a1 = A_shared[local_id.x * 16u + k + 1u];
      
      let b0 = B_shared[k * 16u + local_id.y];
      let b1 = B_shared[(k + 1u) * 16u + local_id.y];
      
      sum = sum + a0 * b0 + a1 * b1;
    }
    
    // Handle remaining values
    for (; k < 16u && tile_start + k < params.K; k = k + 1u) {
      sum = sum + A_shared[local_id.x * 16u + k] * B_shared[k * 16u + local_id.y];
    }
    
    // Ensure all threads are done with the tile before loading the next one
    workgroupBarrier();
  }
  
  // Apply alpha and beta if provided
  if (params.alpha != 1.0 || params.beta != 0.0) {
    // Get existing value for beta
    var existing: f32 = 0.0;
    if (params.beta != 0.0) {
      existing = C[c_batch_offset + row * params.N + col];
    }
    
    // Calculate final result with alpha and beta
    sum = params.alpha * sum + params.beta * existing;
  }
  
  // Store the result
  C[c_batch_offset + row * params.N + col] = sum;
}