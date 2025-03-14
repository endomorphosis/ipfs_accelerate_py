/**
 * Chrome-optimized quantized matrix multiplication shader
 * 
 * Optimized for Chrome's WebGPU implementation with:
 * - Efficient 4-bit quantization unpacking
 * - Workgroup size of 8x16 for balanced performance with quantized operations
 * - Coalesced memory access patterns
 * - Optimized for Chrome's shader compiler
 */

struct Params {
  M: u32,                // Rows in A
  N: u32,                // Columns in B
  K: u32,                // Columns in A / Rows in B
  bits_per_weight: u32,  // Bits per quantized weight
  scale_mode: u32,       // 0 = per-tensor, 1 = per-channel
  zero_point_mode: u32,  // 0 = symmetric, 1 = asymmetric
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> activations: array<f32>;             // FP32 activation values
@group(0) @binding(2) var<storage, read> quantized_weights: array<u32>;       // Packed quantized weights
@group(0) @binding(3) var<storage, read> scales: array<f32>;                  // Scale factors
@group(0) @binding(4) var<storage, read> zero_points: array<f32>;             // Zero points
@group(0) @binding(5) var<storage, read_write> output: array<f32>;            // FP32 output

// Chrome performs well with 8x16 workgroup size for quantized operations
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  
  // Guard against out-of-bounds work items
  if (row >= params.M || col >= params.N) {
    return;
  }
  
  // Calculate indices
  let output_idx = row * params.N + col;
  
  // Initialize sum
  var sum: f32 = 0.0;
  
  // Calculate number of values per 32-bit word based on bit width
  let values_per_word = 32u / params.bits_per_weight;
  
  // Calculate number of words needed for each row
  let words_per_row = (params.K + values_per_word - 1u) / values_per_word;
  
  // Get scale and zero point
  let scale_idx = select(0u, col, params.scale_mode == 1u);
  let zero_point = select(0.0, zero_points[scale_idx], params.zero_point_mode == 1u);
  let scale = scales[scale_idx];

  // Chrome performs well with this unrolled loop approach
  for (var word_idx: u32 = 0u; word_idx < words_per_row; word_idx++) {
    // Calculate offset in quantized weights
    let weight_offset = row * words_per_row + word_idx;
    
    // Check if we're in bounds
    if (weight_offset < params.M * words_per_row) {
      // Load the packed word
      let packed_word = quantized_weights[weight_offset];
      
      // Calculate start index for this word
      let k_start = word_idx * values_per_word;
      
      // Optimize for 4-bit quantization - common case
      if (params.bits_per_weight == 4u) {
        // Process 8 activations at a time (efficient in Chrome)
        for (var i: u32 = 0u; i < 8u && k_start + i < params.K; i++) {
          let unpacked_value = unpack_4bit(packed_word, i);
          let dequantized = dequantize(unpacked_value, scale, zero_point);
          let activation = activations[k_start + i];
          sum += dequantized * activation;
        }
      } else if (params.bits_per_weight == 8u) {
        // Process 4 activations at a time for 8-bit
        for (var i: u32 = 0u; i < 4u && k_start + i < params.K; i++) {
          let unpacked_value = unpack_8bit(packed_word, i);
          let dequantized = dequantize(unpacked_value, scale, zero_point);
          let activation = activations[k_start + i];
          sum += dequantized * activation;
        }
      } else if (params.bits_per_weight == 2u) {
        // Process 16 activations at a time for 2-bit
        for (var i: u32 = 0u; i < 16u && k_start + i < params.K; i++) {
          let unpacked_value = unpack_2bit(packed_word, i);
          let dequantized = dequantize_2bit(unpacked_value, scale, zero_point);
          let activation = activations[k_start + i];
          sum += dequantized * activation;
        }
      }
      // other bit widths would be handled similarly
    }
  }
  
  // Store result
  output[output_idx] = sum;
}

// Unpacking functions optimized for Chrome's shader compiler

// Unpack 4-bit value from 32-bit word (8 values per word)
fn unpack_4bit(packed: u32, idx: u32) -> f32 {
  let shift = idx * 4u;
  let mask = 0xfu;
  let value = (packed >> shift) & mask;
  return f32(value);
}

// Unpack 2-bit value from 32-bit word (16 values per word)
fn unpack_2bit(packed: u32, idx: u32) -> f32 {
  let shift = idx * 2u;
  let mask = 0x3u;
  let value = (packed >> shift) & mask;
  return f32(value);
}

// Unpack 8-bit value from 32-bit word (4 values per word)
fn unpack_8bit(packed: u32, idx: u32) -> f32 {
  let shift = idx * 8u;
  let mask = 0xffu;
  let value = (packed >> shift) & mask;
  return f32(value);
}

// Standard dequantization function
fn dequantize(quantized: f32, scale: f32, zero_point: f32) -> f32 {
  return (quantized - zero_point) * scale;
}

// Special dequantization for 2-bit values with optimized mapping
fn dequantize_2bit(quantized: f32, scale: f32, zero_point: f32) -> f32 {
  // Map 2-bit values (0,1,2,3) to (-1.0, 0.0, 0.5, 1.0) - optimized for small values
  let value_map = array<f32, 4>(-1.0, 0.0, 0.5, 1.0);
  return value_map[u32(quantized)] * scale;
}