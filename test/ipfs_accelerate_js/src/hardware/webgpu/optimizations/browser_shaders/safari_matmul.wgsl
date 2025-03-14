/**
 * Safari-optimized matrix multiplication shader
 * 
 * Optimized for Safari's Metal-based WebGPU implementation with:
 * - Larger workgroup size (16x16x1) for better Apple GPU utilization
 * - Larger tile sizes (32x32) for Apple Silicon GPUs
 * - Vector operations that leverage Apple GPU SIMD capabilities
 * - SIMDGroup operations and optimized memory access patterns
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

// Define workgroup size based on Safari's optimal parameters
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  
  // Declare shared memory tiles - Safari benefits from larger tiles
  var<workgroup> A_shared: array<f32, 32 * 32>;
  var<workgroup> B_shared: array<f32, 32 * 32>;

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
  
  // Calculate number of tiles needed - Safari benefits from larger tiles
  let num_tiles = (params.K + 32u - 1u) / 32u;
  
  // Process each tile
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    // Calculate start of current tile
    let tile_start = t * 32u;
    
    // Collaborative loading of tiles into shared memory
    // Each thread loads multiple elements (2x2 or 4) to fill the larger tile
    // This leverages Apple GPU's SIMD capabilities
    
    // Load first element
    let A_row = row;
    let A_col = tile_start + local_id.y;
    
    if (A_row < params.M && A_col < params.K) {
      A_shared[local_id.x * 32u + local_id.y] = A[a_batch_offset + A_row * params.K + A_col];
    } else {
      A_shared[local_id.x * 32u + local_id.y] = 0.0;
    }
    
    // Load additional elements for A if in bounds
    let A_col2 = tile_start + local_id.y + 16u;
    if (A_row < params.M && A_col2 < params.K) {
      A_shared[local_id.x * 32u + local_id.y + 16u] = A[a_batch_offset + A_row * params.K + A_col2];
    } else {
      A_shared[local_id.x * 32u + local_id.y + 16u] = 0.0;
    }
    
    // Load B elements similarly
    let B_row = tile_start + local_id.x;
    let B_col = col;
    
    if (B_row < params.K && B_col < params.N) {
      B_shared[local_id.x * 32u + local_id.y] = B[b_batch_offset + B_row * params.N + B_col];
    } else {
      B_shared[local_id.x * 32u + local_id.y] = 0.0;
    }
    
    // Load additional elements for B if in bounds
    let B_row2 = tile_start + local_id.x + 16u;
    if (B_row2 < params.K && B_col < params.N) {
      B_shared[(local_id.x + 16u) * 32u + local_id.y] = B[b_batch_offset + B_row2 * params.N + B_col];
    } else {
      B_shared[(local_id.x + 16u) * 32u + local_id.y] = 0.0;
    }
    
    // Ensure all threads have loaded their values
    workgroupBarrier();
    
    // Compute dot product for this tile - Safari benefits from more work per thread
    // Use vector operations and process tile in chunks of 4 elements
    var k = 0u;
    
    // Process in chunks of 4 - leverages Apple GPU SIMD units
    for (; k + 4u <= 32u && tile_start + k < params.K; k = k + 4u) {
      // Load 4 elements at once - better for Apple GPU memory bandwidth
      let a_vec = vec4<f32>(
        A_shared[local_id.x * 32u + k],
        A_shared[local_id.x * 32u + k + 1u],
        A_shared[local_id.x * 32u + k + 2u],
        A_shared[local_id.x * 32u + k + 3u]
      );
      
      let b_vec = vec4<f32>(
        B_shared[k * 32u + local_id.y],
        B_shared[(k + 1u) * 32u + local_id.y],
        B_shared[(k + 2u) * 32u + local_id.y],
        B_shared[(k + 3u) * 32u + local_id.y]
      );
      
      // Use dot product - efficient on Apple GPUs
      sum = sum + dot(a_vec, b_vec);
    }
    
    // Handle remaining values
    for (; k < 32u && tile_start + k < params.K; k = k + 1u) {
      sum = sum + A_shared[local_id.x * 32u + k] * B_shared[k * 32u + local_id.y];
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