/**
 * Browser-Specific Multi-Head Attention Shader Implementations
 * Optimized for different browsers' WebGPU implementations
 */

/**
 * Fast math functions for attention operations
 */
export const ATTENTION_MATH_FUNCTIONS = `
  // Fast softmax implementation
  fn fast_softmax(vec: array<f32, 256>, idx: i32, length: i32) -> f32 {
    var max_val: f32 = -3.4e38; // Close to -FLT_MAX
    
    // Find max value for numerical stability
    for (var i = 0; i < length; i = i + 1) {
      max_val = max(max_val, vec[i]);
    }
    
    // Compute exp(x - max) and sum
    var sum: f32 = 0.0;
    for (var i = 0; i < length; i = i + 1) {
      sum = sum + exp(vec[i] - max_val);
    }
    
    // Return softmax value for the requested index
    return exp(vec[idx] - max_val) / sum;
  }
  
  // Scaled dot-product attention for multi-head attention mechanism
  fn scaled_dot_product(q: f32, k: f32, scale: f32) -> f32 {
    return q * k * scale;
  }
`;

/**
 * Chrome-optimized Multi-Head Attention shader
 * Optimized with:
 * - 256 threads per workgroup
 * - Vectorized operations using vec4
 * - Coalesced memory access patterns
 * - Shared memory for attention scores
 * - 4-element vectorization for dot products
 */
export const CHROME_MULTI_HEAD_ATTENTION_SHADER = `/**
 * Chrome-optimized multi-head attention shader
 * 
 * Optimized for Chrome's WebGPU implementation with:
 * - 256 threads per workgroup
 * - Vectorized operations using vec4
 * - Coalesced memory access patterns
 * - Shared memory for attention scores
 * - 4-element vectorization for dot products
 */

${ATTENTION_MATH_FUNCTIONS}

struct Params {
  batch_size: u32,
  seq_length: u32,
  num_heads: u32,
  head_dim: u32,
  hidden_size: u32,
  scale: f32,
  use_causal_mask: u32,
  dropout_prob: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> key: array<f32>;
@group(0) @binding(3) var<storage, read> value: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> rng_state: array<u32>;

var<workgroup> attention_scores: array<f32, 256>; // Shared memory for attention scores
var<workgroup> attention_probs: array<f32, 256>;  // Shared memory for attention probabilities

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let global_idx = global_id.x;
  
  // Compute batch and head indices
  let elements_per_batch = params.seq_length * params.hidden_size;
  let batch_idx = global_idx / (params.seq_length * params.num_heads);
  let remaining = global_idx - batch_idx * (params.seq_length * params.num_heads);
  let seq_idx = remaining / params.num_heads;
  let head_idx = remaining % params.num_heads;
  
  if (batch_idx >= params.batch_size || seq_idx >= params.seq_length) {
    return;
  }
  
  let head_size = params.head_dim;
  let head_offset = head_idx * head_size;
  
  // Compute attention scores for this sequence position and head
  // Using vectorization for better performance on Chrome
  for (var target_seq_idx: u32 = 0u; target_seq_idx < params.seq_length; target_seq_idx++) {
    // Calculate input offsets
    let q_offset = batch_idx * elements_per_batch + seq_idx * params.hidden_size + head_offset;
    let k_offset = batch_idx * elements_per_batch + target_seq_idx * params.hidden_size + head_offset;
    
    // Compute dot product between query and key vectors
    var dot_product: f32 = 0.0;
    
    // Process 4 elements at a time using vectorization
    for (var d: u32 = 0u; d < head_size; d += 4u) {
      if (d + 4u <= head_size) {
        // Load query and key vectors (4 elements each)
        let q_vec = vec4<f32>(
          query[q_offset + d],
          query[q_offset + d + 1u],
          query[q_offset + d + 2u],
          query[q_offset + d + 3u]
        );
        
        let k_vec = vec4<f32>(
          key[k_offset + d],
          key[k_offset + d + 1u],
          key[k_offset + d + 2u],
          key[k_offset + d + 3u]
        );
        
        // Vectorized dot product
        dot_product += dot(q_vec, k_vec);
      } else {
        // Handle remaining elements
        for (var r: u32 = d; r < head_size; r++) {
          dot_product += query[q_offset + r] * key[k_offset + r];
        }
      }
    }
    
    // Scale dot product
    dot_product *= params.scale;
    
    // Apply causal mask if needed
    if (params.use_causal_mask == 1u && target_seq_idx > seq_idx) {
      dot_product = -3.4e38; // Large negative number
    }
    
    // Store in shared memory
    let score_idx = local_id.x * params.seq_length + target_seq_idx;
    if (score_idx < 256u) {
      attention_scores[score_idx] = dot_product;
    }
  }
  
  // Ensure all attention scores are computed
  workgroupBarrier();
  
  // Apply softmax
  // Find max for numerical stability
  var max_val: f32 = -3.4e38;
  for (var i: u32 = 0u; i < params.seq_length; i++) {
    let score_idx = local_id.x * params.seq_length + i;
    if (score_idx < 256u) {
      max_val = max(max_val, attention_scores[score_idx]);
    }
  }
  
  // Compute exp and sum
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < params.seq_length; i++) {
    let score_idx = local_id.x * params.seq_length + i;
    if (score_idx < 256u) {
      let exp_val = exp(attention_scores[score_idx] - max_val);
      attention_probs[score_idx] = exp_val;
      sum += exp_val;
    }
  }
  
  // Normalize and apply dropout
  for (var i: u32 = 0u; i < params.seq_length; i++) {
    let score_idx = local_id.x * params.seq_length + i;
    if (score_idx < 256u) {
      var prob = attention_probs[score_idx] / sum;
      
      // Apply dropout if needed
      if (params.dropout_prob > 0.0) {
        let rng_val = f32(rng_state[global_idx * params.seq_length + i]) / 4294967295.0;
        if (rng_val < params.dropout_prob) {
          prob = 0.0;
        } else {
          prob /= (1.0 - params.dropout_prob);
        }
      }
      
      attention_probs[score_idx] = prob;
    }
  }
  
  // Compute weighted sum of values
  for (var d: u32 = 0u; d < head_size; d++) {
    var weighted_sum: f32 = 0.0;
    
    for (var i: u32 = 0u; i < params.seq_length; i++) {
      let score_idx = local_id.x * params.seq_length + i;
      if (score_idx < 256u) {
        let v_offset = batch_idx * elements_per_batch + i * params.hidden_size + head_offset + d;
        weighted_sum += attention_probs[score_idx] * value[v_offset];
      }
    }
    
    let out_offset = batch_idx * elements_per_batch + seq_idx * params.hidden_size + head_offset + d;
    output[out_offset] = weighted_sum;
  }
}`;

/**
 * Firefox-optimized Multi-Head Attention shader
 * Optimized with:
 * - 128 threads per workgroup
 * - No vectorization (works better for Firefox)
 * - Simple loops with good memory access patterns
 * - Less shared memory usage
 */
export const FIREFOX_MULTI_HEAD_ATTENTION_SHADER = `/**
 * Firefox-optimized multi-head attention shader
 * 
 * Optimized for Firefox's WebGPU implementation with:
 * - 128 threads per workgroup
 * - Simple non-vectorized operations (better for Firefox)
 * - Straightforward memory access patterns
 * - Minimal shared memory usage
 * - Simple loops with fewer conditionals
 */

${ATTENTION_MATH_FUNCTIONS}

struct Params {
  batch_size: u32,
  seq_length: u32,
  num_heads: u32,
  head_dim: u32,
  hidden_size: u32,
  scale: f32,
  use_causal_mask: u32,
  dropout_prob: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> key: array<f32>;
@group(0) @binding(3) var<storage, read> value: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> rng_state: array<u32>;

var<workgroup> attention_scores: array<f32, 128>; // Shared memory for attention scores
var<workgroup> max_scores: array<f32, 128>;       // Shared memory for max scores
var<workgroup> sum_exps: array<f32, 128>;         // Shared memory for exp sums

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let global_idx = global_id.x;
  
  // Compute batch and head indices
  let elements_per_batch = params.seq_length * params.hidden_size;
  let batch_idx = global_idx / (params.seq_length * params.num_heads);
  let remaining = global_idx - batch_idx * (params.seq_length * params.num_heads);
  let seq_idx = remaining / params.num_heads;
  let head_idx = remaining % params.num_heads;
  
  if (batch_idx >= params.batch_size || seq_idx >= params.seq_length) {
    return;
  }
  
  let head_size = params.head_dim;
  let head_offset = head_idx * head_size;
  
  // Firefox performs better with multiple simpler loops rather than nested ones
  
  // 1. Compute max attention score for numerical stability
  var max_score: f32 = -3.4e38;
  
  for (var target_seq_idx: u32 = 0u; target_seq_idx < params.seq_length; target_seq_idx++) {
    // Calculate input offsets
    let q_offset = batch_idx * elements_per_batch + seq_idx * params.hidden_size + head_offset;
    let k_offset = batch_idx * elements_per_batch + target_seq_idx * params.hidden_size + head_offset;
    
    // Compute dot product between query and key vectors
    var dot_product: f32 = 0.0;
    
    for (var d: u32 = 0u; d < head_size; d++) {
      dot_product += query[q_offset + d] * key[k_offset + d];
    }
    
    // Scale dot product
    dot_product *= params.scale;
    
    // Apply causal mask if needed
    if (params.use_causal_mask == 1u && target_seq_idx > seq_idx) {
      dot_product = -3.4e38; // Large negative number
    }
    
    // Update max score
    max_score = max(max_score, dot_product);
    
    // Store in local array for later use
    attention_scores[local_id.x] = dot_product;
  }
  
  // Store max score in shared memory
  max_scores[local_id.x] = max_score;
  
  // 2. Compute sum of exps for softmax normalization
  var sum_exp: f32 = 0.0;
  
  for (var target_seq_idx: u32 = 0u; target_seq_idx < params.seq_length; target_seq_idx++) {
    // Calculate input offsets
    let q_offset = batch_idx * elements_per_batch + seq_idx * params.hidden_size + head_offset;
    let k_offset = batch_idx * elements_per_batch + target_seq_idx * params.hidden_size + head_offset;
    
    // Compute dot product between query and key vectors
    var dot_product: f32 = 0.0;
    
    for (var d: u32 = 0u; d < head_size; d++) {
      dot_product += query[q_offset + d] * key[k_offset + d];
    }
    
    // Scale dot product
    dot_product *= params.scale;
    
    // Apply causal mask if needed
    if (params.use_causal_mask == 1u && target_seq_idx > seq_idx) {
      dot_product = -3.4e38; // Large negative number
    }
    
    // Compute exp(score - max_score) for numerical stability
    let exp_val = exp(dot_product - max_score);
    sum_exp += exp_val;
    
    // Store in local array for later use
    attention_scores[local_id.x] = exp_val;
  }
  
  // Store sum in shared memory
  sum_exps[local_id.x] = sum_exp;
  
  // 3. Compute weighted sum of values
  for (var d: u32 = 0u; d < head_size; d++) {
    var weighted_sum: f32 = 0.0;
    
    for (var target_seq_idx: u32 = 0u; target_seq_idx < params.seq_length; target_seq_idx++) {
      // Calculate input offsets
      let q_offset = batch_idx * elements_per_batch + seq_idx * params.hidden_size + head_offset;
      let k_offset = batch_idx * elements_per_batch + target_seq_idx * params.hidden_size + head_offset;
      
      // Compute dot product between query and key vectors
      var dot_product: f32 = 0.0;
      
      for (var i: u32 = 0u; i < head_size; i++) {
        dot_product += query[q_offset + i] * key[k_offset + i];
      }
      
      // Scale dot product
      dot_product *= params.scale;
      
      // Apply causal mask if needed
      if (params.use_causal_mask == 1u && target_seq_idx > seq_idx) {
        dot_product = -3.4e38; // Large negative number
      }
      
      // Compute attention probability
      let exp_val = exp(dot_product - max_scores[local_id.x]);
      var attention_prob = exp_val / sum_exps[local_id.x];
      
      // Apply dropout if needed
      if (params.dropout_prob > 0.0) {
        let rng_val = f32(rng_state[global_idx * params.seq_length + target_seq_idx]) / 4294967295.0;
        if (rng_val < params.dropout_prob) {
          attention_prob = 0.0;
        } else {
          attention_prob /= (1.0 - params.dropout_prob);
        }
      }
      
      // Get value and compute weighted sum
      let v_offset = batch_idx * elements_per_batch + target_seq_idx * params.hidden_size + head_offset + d;
      weighted_sum += attention_prob * value[v_offset];
    }
    
    // Write output
    let out_offset = batch_idx * elements_per_batch + seq_idx * params.hidden_size + head_offset + d;
    output[out_offset] = weighted_sum;
  }
}`;

/**
 * Safari-optimized Multi-Head Attention shader
 * Optimized with:
 * - 512 threads per workgroup
 * - Aggressive vectorization with vec4
 * - Optimized for Apple Silicon GPUs
 * - Extensive use of shared memory
 */
export const SAFARI_MULTI_HEAD_ATTENTION_SHADER = `/**
 * Safari-optimized multi-head attention shader
 * 
 * Optimized for Safari's WebGPU implementation with:
 * - 512 threads per workgroup
 * - Aggressive vectorization using vec4
 * - Optimized for Apple Silicon GPUs
 * - Extensive use of shared memory for caching
 * - Optimized memory access patterns for Apple GPUs
 */

${ATTENTION_MATH_FUNCTIONS}

struct Params {
  batch_size: u32,
  seq_length: u32,
  num_heads: u32,
  head_dim: u32,
  hidden_size: u32,
  scale: f32,
  use_causal_mask: u32,
  dropout_prob: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> key: array<f32>;
@group(0) @binding(3) var<storage, read> value: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> rng_state: array<u32>;

var<workgroup> shared_q: array<f32, 512>; // Shared memory for query vectors
var<workgroup> shared_k: array<f32, 512>; // Shared memory for key vectors
var<workgroup> shared_v: array<f32, 512>; // Shared memory for value vectors
var<workgroup> attention_scores: array<f32, 512>; // Shared memory for attention scores

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let global_idx = global_id.x;
  
  // Compute batch and head indices
  let elements_per_batch = params.seq_length * params.hidden_size;
  let batch_idx = global_idx / (params.seq_length * params.num_heads);
  let remaining = global_idx - batch_idx * (params.seq_length * params.num_heads);
  let seq_idx = remaining / params.num_heads;
  let head_idx = remaining % params.num_heads;
  
  if (batch_idx >= params.batch_size || seq_idx >= params.seq_length) {
    return;
  }
  
  let head_size = params.head_dim;
  let head_offset = head_idx * head_size;
  
  // Safari performs best with aggressive vectorization and shared memory usage
  
  // Load query vector into shared memory
  let q_offset = batch_idx * elements_per_batch + seq_idx * params.hidden_size + head_offset;
  for (var d: u32 = 0u; d < head_size; d++) {
    if (local_id.x < head_size) {
      shared_q[local_id.x] = query[q_offset + local_id.x];
    }
  }
  
  workgroupBarrier();
  
  // Process each sequence position
  for (var target_seq_idx: u32 = 0u; target_seq_idx < params.seq_length; target_seq_idx++) {
    // Load key vector into shared memory
    let k_offset = batch_idx * elements_per_batch + target_seq_idx * params.hidden_size + head_offset;
    for (var d: u32 = 0u; d < head_size; d++) {
      if (local_id.x < head_size) {
        shared_k[local_id.x] = key[k_offset + local_id.x];
      }
    }
    
    // Load value vector into shared memory
    let v_offset = batch_idx * elements_per_batch + target_seq_idx * params.hidden_size + head_offset;
    for (var d: u32 = 0u; d < head_size; d++) {
      if (local_id.x < head_size) {
        shared_v[local_id.x] = value[v_offset + local_id.x];
      }
    }
    
    workgroupBarrier();
    
    // Compute attention score - using vectorized operations
    var dot_product: f32 = 0.0;
    
    // Process 4 elements at a time using vec4
    for (var d: u32 = 0u; d < head_size; d += 4u) {
      if (d + 4u <= head_size && local_id.x < 1u) {
        let q_vec = vec4<f32>(
          shared_q[d],
          shared_q[d + 1u],
          shared_q[d + 2u],
          shared_q[d + 3u]
        );
        
        let k_vec = vec4<f32>(
          shared_k[d],
          shared_k[d + 1u],
          shared_k[d + 2u],
          shared_k[d + 3u]
        );
        
        // Use dot product for vec4
        dot_product += dot(q_vec, k_vec);
      }
    }
    
    // Handle remaining elements
    for (var d: u32 = (head_size / 4u) * 4u; d < head_size; d++) {
      if (local_id.x < 1u) {
        dot_product += shared_q[d] * shared_k[d];
      }
    }
    
    // Scale dot product
    dot_product *= params.scale;
    
    // Apply causal mask if needed
    if (params.use_causal_mask == 1u && target_seq_idx > seq_idx) {
      dot_product = -3.4e38; // Large negative number
    }
    
    // Store attention score
    if (local_id.x < 1u) {
      attention_scores[target_seq_idx] = dot_product;
    }
    
    workgroupBarrier();
  }
  
  // Apply softmax to attention scores
  if (local_id.x < 1u) {
    // Find max score for numerical stability
    var max_score: f32 = -3.4e38;
    for (var i: u32 = 0u; i < params.seq_length; i++) {
      max_score = max(max_score, attention_scores[i]);
    }
    
    // Compute exp and sum
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.seq_length; i++) {
      let exp_val = exp(attention_scores[i] - max_score);
      attention_scores[i] = exp_val;
      sum += exp_val;
    }
    
    // Normalize and apply dropout
    for (var i: u32 = 0u; i < params.seq_length; i++) {
      var prob = attention_scores[i] / sum;
      
      // Apply dropout if needed
      if (params.dropout_prob > 0.0) {
        let rng_val = f32(rng_state[global_idx * params.seq_length + i]) / 4294967295.0;
        if (rng_val < params.dropout_prob) {
          prob = 0.0;
        } else {
          prob /= (1.0 - params.dropout_prob);
        }
      }
      
      attention_scores[i] = prob;
    }
  }
  
  workgroupBarrier();
  
  // Compute weighted sum of values
  if (local_id.x < head_size) {
    let d = local_id.x;
    var weighted_sum: f32 = 0.0;
    
    for (var target_seq_idx: u32 = 0u; target_seq_idx < params.seq_length; target_seq_idx++) {
      // Load value from shared memory
      let v_offset = batch_idx * elements_per_batch + target_seq_idx * params.hidden_size + head_offset + d;
      let v_val = value[v_offset];
      
      // Multiply by attention probability
      weighted_sum += attention_scores[target_seq_idx] * v_val;
    }
    
    // Write output
    let out_offset = batch_idx * elements_per_batch + seq_idx * params.hidden_size + head_offset + d;
    output[out_offset] = weighted_sum;
  }
}`;

/**
 * Edge-optimized Multi-Head Attention shader
 * Optimized with:
 * - 256 threads per workgroup
 * - Partial loop unrolling in pairs
 * - Explicit bounds checking
 * - Optimized for Edge's WebGPU implementation
 */
export const EDGE_MULTI_HEAD_ATTENTION_SHADER = `/**
 * Edge-optimized multi-head attention shader
 * 
 * Optimized for Edge's WebGPU implementation with:
 * - 256 threads per workgroup
 * - Partial loop unrolling in pairs
 * - Explicit bounds checking (important for Edge)
 * - Balanced between vectorization and simplicity
 * - Adaptive workgroups based on sequence length
 */

${ATTENTION_MATH_FUNCTIONS}

struct Params {
  batch_size: u32,
  seq_length: u32,
  num_heads: u32,
  head_dim: u32,
  hidden_size: u32,
  scale: f32,
  use_causal_mask: u32,
  dropout_prob: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> key: array<f32>;
@group(0) @binding(3) var<storage, read> value: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> rng_state: array<u32>;

var<workgroup> attention_scores: array<f32, 256>; // Shared memory for attention scores

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let global_idx = global_id.x;
  
  // Compute batch and head indices - Edge benefits from explicit bounds checking
  let elements_per_batch = params.seq_length * params.hidden_size;
  let batch_idx = global_idx / (params.seq_length * params.num_heads);
  
  if (batch_idx >= params.batch_size) {
    return;
  }
  
  let remaining = global_idx - batch_idx * (params.seq_length * params.num_heads);
  let seq_idx = remaining / params.num_heads;
  
  if (seq_idx >= params.seq_length) {
    return;
  }
  
  let head_idx = remaining % params.num_heads;
  let head_size = params.head_dim;
  let head_offset = head_idx * head_size;
  
  // Edge performs best with loop unrolling in pairs
  
  // Compute attention scores
  for (var target_seq_idx: u32 = 0u; target_seq_idx < params.seq_length; target_seq_idx++) {
    // Calculate input offsets
    let q_offset = batch_idx * elements_per_batch + seq_idx * params.hidden_size + head_offset;
    let k_offset = batch_idx * elements_per_batch + target_seq_idx * params.hidden_size + head_offset;
    
    // Compute dot product between query and key vectors
    var dot_product: f32 = 0.0;
    
    // Process elements in pairs (partial unrolling works well for Edge)
    var d: u32 = 0u;
    for (; d + 1u < head_size; d += 2u) {
      dot_product += query[q_offset + d] * key[k_offset + d];
      dot_product += query[q_offset + d + 1u] * key[k_offset + d + 1u];
    }
    
    // Handle remaining element if head_size is odd
    if (d < head_size) {
      dot_product += query[q_offset + d] * key[k_offset + d];
    }
    
    // Scale dot product
    dot_product *= params.scale;
    
    // Apply causal mask if needed
    if (params.use_causal_mask == 1u && target_seq_idx > seq_idx) {
      dot_product = -3.4e38; // Large negative number
    }
    
    // Store attention score
    attention_scores[local_id.x] = dot_product;
  }
  
  // Find max score for numerical stability
  var max_score: f32 = -3.4e38;
  for (var i: u32 = 0u; i < params.seq_length; i++) {
    max_score = max(max_score, attention_scores[local_id.x]);
  }
  
  // Compute exp and sum
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < params.seq_length; i++) {
    let exp_val = exp(attention_scores[local_id.x] - max_score);
    attention_scores[local_id.x] = exp_val;
    sum += exp_val;
  }
  
  // Compute weighted sum of values
  for (var d: u32 = 0u; d < head_size; d++) {
    var weighted_sum: f32 = 0.0;
    
    for (var target_seq_idx: u32 = 0u; target_seq_idx < params.seq_length; target_seq_idx++) {
      // Calculate input offsets
      let q_offset = batch_idx * elements_per_batch + seq_idx * params.hidden_size + head_offset;
      let k_offset = batch_idx * elements_per_batch + target_seq_idx * params.hidden_size + head_offset;
      
      // Compute dot product between query and key vectors
      var dot_product: f32 = 0.0;
      
      // Process elements in pairs
      var i: u32 = 0u;
      for (; i + 1u < head_size; i += 2u) {
        dot_product += query[q_offset + i] * key[k_offset + i];
        dot_product += query[q_offset + i + 1u] * key[k_offset + i + 1u];
      }
      
      // Handle remaining element if head_size is odd
      if (i < head_size) {
        dot_product += query[q_offset + i] * key[k_offset + i];
      }
      
      // Scale dot product
      dot_product *= params.scale;
      
      // Apply causal mask if needed
      if (params.use_causal_mask == 1u && target_seq_idx > seq_idx) {
        dot_product = -3.4e38; // Large negative number
      }
      
      // Compute attention probability with numerical stability
      let exp_val = exp(dot_product - max_score);
      var attention_prob = exp_val / sum;
      
      // Apply dropout if needed
      if (params.dropout_prob > 0.0) {
        let rng_val = f32(rng_state[global_idx * params.seq_length + target_seq_idx]) / 4294967295.0;
        if (rng_val < params.dropout_prob) {
          attention_prob = 0.0;
        } else {
          attention_prob /= (1.0 - params.dropout_prob);
        }
      }
      
      // Get value and compute weighted sum
      let v_offset = batch_idx * elements_per_batch + target_seq_idx * params.hidden_size + head_offset + d;
      weighted_sum += attention_prob * value[v_offset];
    }
    
    // Write output
    let out_offset = batch_idx * elements_per_batch + seq_idx * params.hidden_size + head_offset + d;
    output[out_offset] = weighted_sum;
  }
}`;

// Export the shaders for use in the main shader loader
export {
  CHROME_MULTI_HEAD_ATTENTION_SHADER,
  FIREFOX_MULTI_HEAD_ATTENTION_SHADER,
  SAFARI_MULTI_HEAD_ATTENTION_SHADER,
  EDGE_MULTI_HEAD_ATTENTION_SHADER
};