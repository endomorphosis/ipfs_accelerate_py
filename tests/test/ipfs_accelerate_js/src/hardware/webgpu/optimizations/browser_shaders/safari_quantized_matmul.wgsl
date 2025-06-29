/**
 * Safari-optimized quantized matrix multiplication shader
 * 
 * Optimized for Safari's Metal-based WebGPU implementation with:
 * - Special optimizations for Apple Silicon GPUs
 * - Larger workgroup size for better utilization of Apple GPU cores
 * - Vector operations for better performance on Apple Silicon GPUs
 * - Optimized 3-bit unpacking algorithm for Apple GPUs
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

// Safari performs well with larger workgroups for quantized operations
@compute @workgroup_size(16, 16, 1)
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
  var values_per_word: u32;
  
  // Safari handles switch statements efficiently
  switch(params.bits_per_weight) {
    case 8u: {
      values_per_word = 4u;
      break;
    }
    case 4u: {
      values_per_word = 8u;
      break;
    }
    case 3u: {
      // Special case for 3-bit (10 values per 32-bit word in Safari)
      values_per_word = 10u;
      break;
    }
    case 2u: {
      values_per_word = 16u;
      break;
    }
    case 1u: {
      values_per_word = 32u;
      break;
    }
    default: {
      values_per_word = 8u; // Default to 4-bit
    }
  }
  
  // Calculate number of words needed for each row
  let words_per_row = (params.K + values_per_word - 1u) / values_per_word;
  
  // Get scale and zero point
  let scale_idx = select(0u, col, params.scale_mode == 1u);
  let zero_point = select(0.0, zero_points[scale_idx], params.zero_point_mode == 1u);
  let scale = scales[scale_idx];

  // Safari benefits from vectorized processing when possible
  for (var word_idx: u32 = 0u; word_idx < words_per_row; word_idx++) {
    // Calculate offset in quantized weights
    let weight_offset = row * words_per_row + word_idx;
    
    // Check if we're in bounds
    if (weight_offset < params.M * words_per_row) {
      // Load the packed word
      let packed_word = quantized_weights[weight_offset];
      
      // Calculate start index for this word
      let k_start = word_idx * values_per_word;
      
      // Special case for 3-bit on Safari (optimized)
      if (params.bits_per_weight == 3u) {
        // Process 10 values at a time (3-bit special case)
        // This is highly optimized for Apple GPUs
        var i: u32 = 0u;
        
        // Process vector chunks for better performance on Apple Silicon
        for (; i + 4u <= 10u && k_start + i < params.K; i += 4u) {
          // Pre-fetch 4 activations at once - better for Apple GPU memory
          let activation_vec = vec4<f32>(
            activations[k_start + i],
            activations[k_start + i + 1u],
            activations[k_start + i + 2u],
            activations[k_start + i + 3u]
          );
          
          // Unpack 4 values at once
          let unpacked_values = vec4<f32>(
            unpack_3bit(packed_word, i),
            unpack_3bit(packed_word, i + 1u),
            unpack_3bit(packed_word, i + 2u),
            unpack_3bit(packed_word, i + 3u)
          );
          
          // Dequantize all 4 values
          let dequantized_values = (unpacked_values - vec4<f32>(zero_point)) * vec4<f32>(scale);
          
          // Multiply and accumulate using dot product
          sum += dot(dequantized_values, activation_vec);
        }
        
        // Handle remaining values
        for (; i < 10u && k_start + i < params.K; i++) {
          let unpacked_value = unpack_3bit(packed_word, i);
          let dequantized = (unpacked_value - zero_point) * scale;
          let activation = activations[k_start + i];
          sum += dequantized * activation;
        }
      } else if (params.bits_per_weight == 8u || params.bits_per_weight == 4u) {
        // Generic handling for 8-bit and 4-bit
        let unpack_fn = select(unpack_4bit, unpack_8bit, params.bits_per_weight == 8u);
        let vals_per_word = select(8u, 4u, params.bits_per_weight == 8u);
        
        for (var i: u32 = 0u; i < vals_per_word && k_start + i < params.K; i++) {
          let unpacked_value = unpack_fn(packed_word, i);
          let dequantized = (unpacked_value - zero_point) * scale;
          let activation = activations[k_start + i];
          sum += dequantized * activation;
        }
      } else if (params.bits_per_weight == 2u) {
        // Specialized 2-bit handler with optimal mapping for Apple Silicon
        for (var i: u32 = 0u; i < 16u && k_start + i < params.K; i++) {
          let unpacked_value = unpack_2bit(packed_word, i);
          let dequantized = dequantize_2bit(unpacked_value, scale, zero_point);
          let activation = activations[k_start + i];
          sum += dequantized * activation;
        }
      }
      // Other bit widths would be handled similarly
    }
  }
  
  // Store result
  output[output_idx] = sum;
}

// Unpacking functions optimized for Safari's Metal shader compiler

// Highly optimized 3-bit unpacking for Safari (10 values per 32-bit word)
fn unpack_3bit(packed: u32, idx: u32) -> f32 {
  // For 3-bit, we pack 10 values in a 32-bit word (10 * 3 = 30 bits)
  // With optimized bit position calculation for Apple GPUs
  let bit_pos = idx * 3u;
  let mask = 0x7u; // 3-bit mask (0b111)
  let value = (packed >> bit_pos) & mask;
  return f32(value);
}

// Unpack 4-bit value from 32-bit word
fn unpack_4bit(packed: u32, idx: u32) -> f32 {
  let shift = idx * 4u;
  let mask = 0xfu;
  let value = (packed >> shift) & mask;
  return f32(value);
}

// Unpack 8-bit value from 32-bit word
fn unpack_8bit(packed: u32, idx: u32) -> f32 {
  let shift = idx * 8u;
  let mask = 0xffu;
  let value = (packed >> shift) & mask;
  return f32(value);
}

// Unpack 2-bit value
fn unpack_2bit(packed: u32, idx: u32) -> f32 {
  let shift = idx * 2u;
  let mask = 0x3u;
  let value = (packed >> shift) & mask;
  return f32(value);
}

// Special dequantization for 2-bit values (Apple Silicon optimized)
fn dequantize_2bit(quantized: f32, scale: f32, zero_point: f32) -> f32 {
  // Apple Silicon optimized mapping
  let value_map = array<f32, 4>(-1.0, -0.33, 0.33, 1.0);
  return value_map[u32(quantized)] * scale;
}