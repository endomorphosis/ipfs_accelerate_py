/**
 * Browser-Specific Softmax Shader Implementations
 * Optimized for different browsers' WebGPU implementations
 */

/**
 * Chrome-optimized Softmax shader
 * Optimized with:
 * - 256 threads per workgroup
 * - Vectorized operations using vec4
 * - Coalesced memory access patterns
 * - Two-pass approach (max finding and softmax computation)
 */
export const CHROME_SOFTMAX_SHADER = `/**
 * Chrome-optimized softmax shader
 * 
 * Optimized for Chrome's WebGPU implementation with:
 * - 256 threads per workgroup
 * - Vectorized operations using vec4
 * - Coalesced memory access patterns
 * - Two-pass approach (max finding and softmax computation)
 */

struct Params {
  size: u32,
  batch_size: u32,
  dim_size: u32,
  axis: u32, // 0 for softmax across rows, 1 for softmax across columns
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> max_values: array<f32, 256>;
var<workgroup> sum_values: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let global_idx = global_id.x;
  
  // Compute batch index and position within batch
  let elements_per_batch = params.dim_size;
  let batch_idx = global_idx / elements_per_batch;
  let pos_in_batch = global_idx % elements_per_batch;
  
  if (batch_idx >= params.batch_size) {
    return;
  }
  
  let base_offset = batch_idx * elements_per_batch;
  
  // Find max value for numerical stability
  var max_val: f32 = -3.4e38; // Close to -FLT_MAX
  
  // Process 4 elements at a time when possible
  for (var i: u32 = 0u; i < params.dim_size; i += 4u) {
    if (i + 4u <= params.dim_size) {
      // Load 4 elements at once
      let val_vec = vec4<f32>(
        input[base_offset + i],
        input[base_offset + i + 1u],
        input[base_offset + i + 2u],
        input[base_offset + i + 3u]
      );
      
      // Find max of these 4 elements
      max_val = max(max_val, max(max(val_vec.x, val_vec.y), max(val_vec.z, val_vec.w)));
    } else {
      // Handle remaining elements
      for (var j: u32 = i; j < params.dim_size; j++) {
        max_val = max(max_val, input[base_offset + j]);
      }
    }
  }
  
  // Store max value
  max_values[local_id.x] = max_val;
  
  workgroupBarrier();
  
  // Compute sum of exp(x - max_val)
  var sum: f32 = 0.0;
  
  // Process 4 elements at a time when possible
  for (var i: u32 = 0u; i < params.dim_size; i += 4u) {
    if (i + 4u <= params.dim_size) {
      // Load 4 elements at once
      let val_vec = vec4<f32>(
        input[base_offset + i],
        input[base_offset + i + 1u],
        input[base_offset + i + 2u],
        input[base_offset + i + 3u]
      );
      
      // Subtract max and compute exp
      let exp_vec = exp(val_vec - vec4<f32>(max_val));
      
      // Sum the exponentials
      sum += exp_vec.x + exp_vec.y + exp_vec.z + exp_vec.w;
    } else {
      // Handle remaining elements
      for (var j: u32 = i; j < params.dim_size; j++) {
        sum += exp(input[base_offset + j] - max_val);
      }
    }
  }
  
  // Store sum value
  sum_values[local_id.x] = sum;
  
  workgroupBarrier();
  
  // Compute softmax values
  for (var i: u32 = 0u; i < params.dim_size; i++) {
    // Handle boundaries
    if (base_offset + i < params.size) {
      let input_val = input[base_offset + i];
      let softmax_val = exp(input_val - max_values[local_id.x]) / sum_values[local_id.x];
      output[base_offset + i] = softmax_val;
    }
  }
}`;

/**
 * Firefox-optimized Softmax shader
 * Optimized with:
 * - 128 threads per workgroup
 * - Simple non-vectorized operations (better for Firefox)
 * - Two-pass approach with simpler loops
 */
export const FIREFOX_SOFTMAX_SHADER = `/**
 * Firefox-optimized softmax shader
 * 
 * Optimized for Firefox's WebGPU implementation with:
 * - 128 threads per workgroup
 * - Simple non-vectorized operations (better for Firefox)
 * - Two-pass approach with simpler loops
 * - Minimal shared memory usage
 */

struct Params {
  size: u32,
  batch_size: u32,
  dim_size: u32,
  axis: u32, // 0 for softmax across rows, 1 for softmax across columns
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> max_values: array<f32, 128>;
var<workgroup> sum_values: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let global_idx = global_id.x;
  
  // Compute batch index and position within batch
  let elements_per_batch = params.dim_size;
  let batch_idx = global_idx / elements_per_batch;
  let pos_in_batch = global_idx % elements_per_batch;
  
  if (batch_idx >= params.batch_size) {
    return;
  }
  
  let base_offset = batch_idx * elements_per_batch;
  
  // Find max value for numerical stability
  var max_val: f32 = -3.4e38; // Close to -FLT_MAX
  
  for (var i: u32 = 0u; i < params.dim_size; i++) {
    max_val = max(max_val, input[base_offset + i]);
  }
  
  // Store max value
  max_values[local_id.x] = max_val;
  
  workgroupBarrier();
  
  // Compute sum of exp(x - max_val)
  var sum: f32 = 0.0;
  
  for (var i: u32 = 0u; i < params.dim_size; i++) {
    sum += exp(input[base_offset + i] - max_values[local_id.x]);
  }
  
  // Store sum value
  sum_values[local_id.x] = sum;
  
  workgroupBarrier();
  
  // Compute softmax values
  for (var i: u32 = 0u; i < params.dim_size; i++) {
    if (base_offset + i < params.size) {
      let input_val = input[base_offset + i];
      let softmax_val = exp(input_val - max_values[local_id.x]) / sum_values[local_id.x];
      output[base_offset + i] = softmax_val;
    }
  }
}`;

/**
 * Safari-optimized Softmax shader
 * Optimized with:
 * - 512 threads per workgroup
 * - Aggressive vectorization with vec4
 * - Optimized for Apple Silicon GPUs
 */
export const SAFARI_SOFTMAX_SHADER = `/**
 * Safari-optimized softmax shader
 * 
 * Optimized for Safari's WebGPU implementation with:
 * - 512 threads per workgroup
 * - Aggressive vectorization using vec4
 * - Optimized for Apple Silicon GPUs
 * - Extensive shared memory usage
 */

struct Params {
  size: u32,
  batch_size: u32,
  dim_size: u32,
  axis: u32, // 0 for softmax across rows, 1 for softmax across columns
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_values: array<f32, 512>; // Shared memory for input values
var<workgroup> max_values: array<f32, 512>;
var<workgroup> sum_values: array<f32, 512>;

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let global_idx = global_id.x;
  
  // Compute batch index and position within batch
  let elements_per_batch = params.dim_size;
  let batch_idx = global_idx / elements_per_batch;
  let pos_in_batch = global_idx % elements_per_batch;
  
  if (batch_idx >= params.batch_size) {
    return;
  }
  
  let base_offset = batch_idx * elements_per_batch;
  
  // Load values into shared memory for faster access
  for (var i: u32 = 0u; i < params.dim_size; i++) {
    if (local_id.x < params.dim_size) {
      shared_values[local_id.x] = input[base_offset + local_id.x];
    }
  }
  
  workgroupBarrier();
  
  // Find max value for numerical stability
  var max_val: f32 = -3.4e38; // Close to -FLT_MAX
  
  // Process 4 elements at a time using vec4
  for (var i: u32 = 0u; i < params.dim_size; i += 4u) {
    if (i + 4u <= params.dim_size) {
      let val_vec = vec4<f32>(
        shared_values[i],
        shared_values[i + 1u],
        shared_values[i + 2u],
        shared_values[i + 3u]
      );
      
      // Find max of these 4 elements
      max_val = max(max_val, max(max(val_vec.x, val_vec.y), max(val_vec.z, val_vec.w)));
    } else {
      // Handle remaining elements
      for (var j: u32 = i; j < params.dim_size; j++) {
        max_val = max(max_val, shared_values[j]);
      }
    }
  }
  
  // Store max value
  max_values[local_id.x] = max_val;
  
  workgroupBarrier();
  
  // Compute sum of exp(x - max_val)
  var sum: f32 = 0.0;
  
  // Process 4 elements at a time using vec4
  for (var i: u32 = 0u; i < params.dim_size; i += 4u) {
    if (i + 4u <= params.dim_size) {
      let val_vec = vec4<f32>(
        shared_values[i],
        shared_values[i + 1u],
        shared_values[i + 2u],
        shared_values[i + 3u]
      );
      
      // Subtract max and compute exp
      let exp_vec = exp(val_vec - vec4<f32>(max_values[local_id.x]));
      
      // Sum the exponentials
      sum += exp_vec.x + exp_vec.y + exp_vec.z + exp_vec.w;
    } else {
      // Handle remaining elements
      for (var j: u32 = i; j < params.dim_size; j++) {
        sum += exp(shared_values[j] - max_values[local_id.x]);
      }
    }
  }
  
  // Store sum value
  sum_values[local_id.x] = sum;
  
  workgroupBarrier();
  
  // Compute softmax values
  for (var i: u32 = 0u; i < params.dim_size; i++) {
    if (local_id.x < params.dim_size && base_offset + i < params.size) {
      let softmax_val = exp(shared_values[i] - max_values[local_id.x]) / sum_values[local_id.x];
      output[base_offset + i] = softmax_val;
    }
  }
}`;

/**
 * Edge-optimized Softmax shader
 * Optimized with:
 * - 256 threads per workgroup
 * - Partial loop unrolling in pairs
 * - Explicit bounds checking
 */
export const EDGE_SOFTMAX_SHADER = `/**
 * Edge-optimized softmax shader
 * 
 * Optimized for Edge's WebGPU implementation with:
 * - 256 threads per workgroup
 * - Partial loop unrolling in pairs
 * - Explicit bounds checking (important for Edge)
 * - Balanced between vectorization and simplicity
 */

struct Params {
  size: u32,
  batch_size: u32,
  dim_size: u32,
  axis: u32, // 0 for softmax across rows, 1 for softmax across columns
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> max_values: array<f32, 256>;
var<workgroup> sum_values: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let global_idx = global_id.x;
  
  // Compute batch index and position within batch
  let elements_per_batch = params.dim_size;
  let batch_idx = global_idx / elements_per_batch;
  
  // Edge benefits from explicit bounds checking
  if (batch_idx >= params.batch_size) {
    return;
  }
  
  let pos_in_batch = global_idx % elements_per_batch;
  let base_offset = batch_idx * elements_per_batch;
  
  // Find max value for numerical stability
  var max_val: f32 = -3.4e38; // Close to -FLT_MAX
  
  // Process elements in pairs (partial unrolling works well for Edge)
  var i: u32 = 0u;
  for (; i + 1u < params.dim_size; i += 2u) {
    let val1 = input[base_offset + i];
    let val2 = input[base_offset + i + 1u];
    max_val = max(max_val, max(val1, val2));
  }
  
  // Handle the last element if dim_size is odd
  if (i < params.dim_size) {
    max_val = max(max_val, input[base_offset + i]);
  }
  
  // Store max value
  max_values[local_id.x] = max_val;
  
  workgroupBarrier();
  
  // Compute sum of exp(x - max_val)
  var sum: f32 = 0.0;
  
  // Process elements in pairs
  i = 0u;
  for (; i + 1u < params.dim_size; i += 2u) {
    let exp1 = exp(input[base_offset + i] - max_values[local_id.x]);
    let exp2 = exp(input[base_offset + i + 1u] - max_values[local_id.x]);
    sum += exp1 + exp2;
  }
  
  // Handle the last element if dim_size is odd
  if (i < params.dim_size) {
    sum += exp(input[base_offset + i] - max_values[local_id.x]);
  }
  
  // Store sum value
  sum_values[local_id.x] = sum;
  
  workgroupBarrier();
  
  // Compute softmax values
  i = 0u;
  for (; i + 1u < params.dim_size; i += 2u) {
    // Process pairs of elements when possible
    if (base_offset + i + 1u < params.size) {
      let val1 = input[base_offset + i];
      let val2 = input[base_offset + i + 1u];
      
      let softmax1 = exp(val1 - max_values[local_id.x]) / sum_values[local_id.x];
      let softmax2 = exp(val2 - max_values[local_id.x]) / sum_values[local_id.x];
      
      output[base_offset + i] = softmax1;
      output[base_offset + i + 1u] = softmax2;
    } else if (base_offset + i < params.size) {
      // Edge case: only one element left in bounds
      let val = input[base_offset + i];
      let softmax_val = exp(val - max_values[local_id.x]) / sum_values[local_id.x];
      output[base_offset + i] = softmax_val;
    }
  }
  
  // Handle the last element if dim_size is odd
  if (i < params.dim_size && base_offset + i < params.size) {
    let val = input[base_offset + i];
    let softmax_val = exp(val - max_values[local_id.x]) / sum_values[local_id.x];
    output[base_offset + i] = softmax_val;
  }
}`;

// Export the shaders for use in the main shader loader
export {
  CHROME_SOFTMAX_SHADER,
  FIREFOX_SOFTMAX_SHADER,
  SAFARI_SOFTMAX_SHADER,
  EDGE_SOFTMAX_SHADER
};