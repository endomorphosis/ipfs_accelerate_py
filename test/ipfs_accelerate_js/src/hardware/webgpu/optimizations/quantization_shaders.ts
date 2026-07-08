/**
 * WebGPU Quantization Shaders
 * Specialized shaders for tensor quantization operations
 */

import { BrowserType } from '../browser_optimized_operations';

/**
 * Shader configuration options for quantization
 */
export interface QuantizationShaderOptions {
  /** Workgroup size for compute operations */
  workgroupSize?: number;
  
  /** Browser optimization target */
  browserType?: BrowserType;
  
  /** Whether to use fast math approximations */
  useFastMath?: boolean;
  
  /** Bits per weight (1, 2, 3, 4, 8) */
  bitsPerWeight?: 1 | 2 | 3 | 4 | 8;
  
  /** Use symmetric quantization */
  useSymmetricQuantization?: boolean;
  
  /** Whether to implement per-channel quantization */
  usePerChannelQuantization?: boolean;
}

/**
 * Default shader options
 */
const DEFAULT_OPTIONS: QuantizationShaderOptions = {
  workgroupSize: 256,
  browserType: BrowserType.UNKNOWN,
  useFastMath: true,
  bitsPerWeight: 4,
  useSymmetricQuantization: false,
  usePerChannelQuantization: true
};

/**
 * Generate shader for quantizing a tensor to N-bit precision
 * @param options Shader configuration options
 * @returns WGSL shader code
 */
export function generateQuantizeShader(options: QuantizationShaderOptions = {}): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // Determine appropriate workgroup size based on browser
  let workgroupSize = opts.workgroupSize || 256;
  if (opts.browserType === BrowserType.FIREFOX) {
    workgroupSize = 128; // Firefox often performs better with smaller workgroups
  } else if (opts.browserType === BrowserType.SAFARI) {
    workgroupSize = 512; // Safari/Apple GPUs can handle larger workgroups
  }
  
  // Calculate number of quantization levels based on bit depth
  const numLevels = 1 << opts.bitsPerWeight!;
  
  // Determine whether to use per-channel quantization (default: true)
  const usePerChannelQuant = opts.usePerChannelQuantization !== false;
  
  // Create the shader code based on quantization options
  return /* wgsl */`
  // Binding group layout:
  // binding 0: Input tensor (fp32)
  // binding 1: Output quantized tensor (packed bits)
  // binding 2: Scale factors (per whole tensor or per channel)
  // binding 3: Zero points (per whole tensor or per channel)
  // binding 4: Uniform parameters
  
  struct Params {
    tensor_size: u32,     // Total number of elements in tensor
    channel_size: u32,    // Size of each channel (for per-channel quantization)
    num_channels: u32,    // Number of channels
    bits_per_weight: u32, // Number of bits per weight (1-8)
    symmetric_quant: u32, // Whether using symmetric quantization (0 = false, 1 = true)
    per_channel_quant: u32, // Whether using per-channel quantization (0 = false, 1 = true)
  };
  
  @group(0) @binding(0) var<storage, read> input: array<f32>;
  @group(0) @binding(1) var<storage, read_write> output: array<u32>;
  @group(0) @binding(2) var<storage, read> scales: array<f32>;
  @group(0) @binding(3) var<storage, read> zero_points: array<f32>;
  @group(0) @binding(4) var<uniform> params: Params;
  
  // Helper function to pack multiple quantized values into a single u32
  fn pack_values(values: array<u32, 16>, num_values: u32, bits_per_value: u32) -> u32 {
    var result: u32 = 0u;
    let mask = (1u << bits_per_value) - 1u;
    
    for (var i: u32 = 0u; i < num_values; i = i + 1u) {
      // Ensure the value is within the appropriate range
      let value = values[i] & mask;
      
      // Shift the value to its position and OR it into the result
      let shift = i * bits_per_value;
      result = result | (value << shift);
    }
    
    return result;
  }
  
  @compute @workgroup_size(${workgroupSize})
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Each thread processes multiple elements based on bits per weight
    let values_per_u32 = 32u / params.bits_per_weight;
    let output_idx = idx;
    
    // Bounds check
    if (output_idx >= (params.tensor_size + values_per_u32 - 1u) / values_per_u32) {
      return;
    }
    
    // Determine how many values we'll pack in this u32
    let start_idx = output_idx * values_per_u32;
    let end_idx = min(start_idx + values_per_u32, params.tensor_size);
    let num_values = end_idx - start_idx;
    
    // Array to store quantized values before packing
    var quantized_values: array<u32, 16>;
    
    // Process each value that will be packed into this u32
    for (var i: u32 = 0u; i < num_values; i = i + 1u) {
      let input_idx = start_idx + i;
      
      // Get the fp32 value
      let fp_value = input[input_idx];
      
      // Determine channel index if using per-channel quantization
      var channel_idx: u32 = 0u;
      if (params.per_channel_quant == 1u) {
        channel_idx = (input_idx / params.channel_size) % params.num_channels;
      }
      
      // Get scale and zero point
      let scale = scales[channel_idx];
      let zero_point = zero_points[channel_idx];
      
      // Convert to integer representation
      var quant_value: i32;
      
      if (params.symmetric_quant == 1u) {
        // Symmetric quantization (no zero point)
        quant_value = i32(round(fp_value / scale));
        
        // Clamp to range for the given bit depth
        let max_value = i32((1u << (params.bits_per_weight - 1u)) - 1u);
        quant_value = clamp(quant_value, -max_value - 1, max_value);
        
        // Shift to unsigned range [0, 2^bits_per_weight - 1]
        quant_value = quant_value + i32(1u << (params.bits_per_weight - 1u));
      } else {
        // Asymmetric quantization (with zero point)
        quant_value = i32(round(fp_value / scale) + zero_point);
        
        // Clamp to range for the given bit depth
        quant_value = clamp(quant_value, 0, i32((1u << params.bits_per_weight) - 1u));
      }
      
      // Store the quantized value
      quantized_values[i] = u32(quant_value);
    }
    
    // Pack the values into a single u32
    let packed_value = pack_values(quantized_values, num_values, params.bits_per_weight);
    
    // Write the packed value to the output
    output[output_idx] = packed_value;
  }`;
}

/**
 * Generate shader for dequantizing a tensor from N-bit precision to FP32
 * @param options Shader configuration options
 * @returns WGSL shader code
 */
export function generateDequantizeShader(options: QuantizationShaderOptions = {}): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // Determine appropriate workgroup size based on browser
  let workgroupSize = opts.workgroupSize || 256;
  if (opts.browserType === BrowserType.FIREFOX) {
    workgroupSize = 128;
  } else if (opts.browserType === BrowserType.SAFARI) {
    workgroupSize = 512;
  }
  
  // Determine whether to use per-channel quantization (default: true)
  const usePerChannelQuant = opts.usePerChannelQuantization !== false;
  
  return /* wgsl */`
  // Binding group layout:
  // binding 0: Input quantized tensor (packed bits)
  // binding 1: Output tensor (fp32)
  // binding 2: Scale factors (per whole tensor or per channel)
  // binding 3: Zero points (per whole tensor or per channel)
  // binding 4: Uniform parameters
  
  struct Params {
    tensor_size: u32,     // Total number of elements in tensor
    channel_size: u32,    // Size of each channel (for per-channel quantization)
    num_channels: u32,    // Number of channels
    bits_per_weight: u32, // Number of bits per weight (1-8)
    symmetric_quant: u32, // Whether using symmetric quantization (0 = false, 1 = true)
    per_channel_quant: u32, // Whether using per-channel quantization (0 = false, 1 = true)
  };
  
  @group(0) @binding(0) var<storage, read> input: array<u32>;
  @group(0) @binding(1) var<storage, read_write> output: array<f32>;
  @group(0) @binding(2) var<storage, read> scales: array<f32>;
  @group(0) @binding(3) var<storage, read> zero_points: array<f32>;
  @group(0) @binding(4) var<uniform> params: Params;
  
  // Helper function to unpack multiple quantized values from a single u32
  fn unpack_value(packed: u32, index: u32, bits_per_value: u32) -> u32 {
    let mask = (1u << bits_per_value) - 1u;
    let shift = index * bits_per_value;
    return (packed >> shift) & mask;
  }
  
  @compute @workgroup_size(${workgroupSize})
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Bounds check
    if (idx >= params.tensor_size) {
      return;
    }
    
    // Determine which packed value and position within that value
    let values_per_u32 = 32u / params.bits_per_weight;
    let packed_idx = idx / values_per_u32;
    let value_pos = idx % values_per_u32;
    
    // Get the packed value and extract the specific quantized value
    let packed_value = input[packed_idx];
    let quant_value = unpack_value(packed_value, value_pos, params.bits_per_weight);
    
    // Determine channel index if using per-channel quantization
    var channel_idx: u32 = 0u;
    if (params.per_channel_quant == 1u) {
      channel_idx = (idx / params.channel_size) % params.num_channels;
    }
    
    // Get scale and zero point
    let scale = scales[channel_idx];
    let zero_point = zero_points[channel_idx];
    
    // Convert to float
    var fp_value: f32;
    
    if (params.symmetric_quant == 1u) {
      // Symmetric quantization (no zero point)
      // First shift back to signed range
      let signed_value = i32(quant_value) - i32(1u << (params.bits_per_weight - 1u));
      fp_value = f32(signed_value) * scale;
    } else {
      // Asymmetric quantization (with zero point)
      fp_value = (f32(quant_value) - zero_point) * scale;
    }
    
    // Write to output
    output[idx] = fp_value;
  }`;
}

/**
 * Generate shader for 4-bit matrix multiplication with KV cache optimization
 * Special shader for transformer blocks that optimizes attention computation
 * @param options Shader configuration options
 * @returns WGSL shader code
 */
export function generateQuantizedMatmulShader(options: QuantizationShaderOptions = {}): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // Fixed at 4-bit precision for attention computation
  opts.bitsPerWeight = 4;
  
  // Use 8x32 as default workgroup size for matrix multiplication
  const workgroupSizeX = 8;
  const workgroupSizeY = 32;
  
  return /* wgsl */`
  // Specialized 4-bit matrix multiplication shader for attention computation
  // A: quantized weight matrix [M,K] (4-bit packed)
  // B: activation matrix [K,N] (fp32)
  // C: output matrix [M,N] (fp32)
  
  struct Dimensions {
    M: u32,            // Rows in A
    K: u32,            // Columns in A / Rows in B
    N: u32,            // Columns in B
    bits_per_weight: u32, // Always 4 for this shader
  };
  
  @group(0) @binding(0) var<storage, read> weights: array<u32>;         // 4-bit packed weights
  @group(0) @binding(1) var<storage, read> activations: array<f32>;     // FP32 activations
  @group(0) @binding(2) var<storage, read_write> output: array<f32>;    // FP32 output
  @group(0) @binding(3) var<uniform> dimensions: Dimensions;            // Matrix dimensions
  @group(0) @binding(4) var<storage, read> scales: array<f32>;          // Scales for dequantization
  @group(0) @binding(5) var<storage, read> zero_points: array<f32>;     // Zero points
  
  // Helper function to unpack 4-bit values from a u32
  fn unpack_4bit_values(packed: u32) -> array<f32, 8> {
    var result: array<f32, 8>;
    
    for (var i = 0u; i < 8u; i = i + 1u) {
      // Extract 4-bit value
      let quant_value = (packed >> (i * 4u)) & 0xFu;
      
      // Convert to float (for now, assuming weights are per-tensor quantized)
      // This will be expanded later to support per-channel quantization
      let scale = scales[0];
      let zero_point = zero_points[0];
      
      // Dequantize: asymmetric by default
      result[i] = (f32(quant_value) - zero_point) * scale;
    }
    
    return result;
  }
  
  @compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY})
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    
    // Bounds check
    if (row >= dimensions.M || col >= dimensions.N) {
      return;
    }
    
    var sum: f32 = 0.0;
    
    // Process 8 elements (4-bit packed) at a time
    for (var k_base = 0u; k_base < dimensions.K; k_base = k_base + 8u) {
      // Calculate the packed weight index
      let weight_idx = (row * dimensions.K + k_base) / 8u;
      
      // Get the packed weights
      let packed_weights = weights[weight_idx];
      
      // Unpack weights
      let unpacked_weights = unpack_4bit_values(packed_weights);
      
      // Perform dot product for this section
      for (var i = 0u; i < 8u; i = i + 1u) {
        let k = k_base + i;
        if (k < dimensions.K) { // Bound check for K dimension
          let act_idx = k * dimensions.N + col;
          sum = sum + unpacked_weights[i] * activations[act_idx];
        }
      }
    }
    
    // Write output
    let output_idx = row * dimensions.N + col;
    output[output_idx] = sum;
  }`;
}

/**
 * Generate shader for quantized KV cache in transformer models
 * This allows saving memory for transformer's key-value cache by using 4-bit storage
 * @param options Shader configuration options
 * @returns WGSL shader code for KV cache write operation
 */
export function generateKVCacheQuantizeShader(options: QuantizationShaderOptions = {}): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // Set to 4-bit as default for KV cache
  opts.bitsPerWeight = 4;
  
  // Use 256 as default workgroup size for KV cache operations
  const workgroupSize = opts.workgroupSize || 256;
  
  return /* wgsl */`
  // Specialized shader for quantizing KV cache in transformer models
  
  struct Params {
    batch_size: u32,        // Number of sequences
    num_layers: u32,        // Number of transformer layers
    num_heads: u32,         // Number of attention heads
    head_dim: u32,          // Dimension of each attention head
    seq_len: u32,           // Current sequence length
    bits_per_value: u32,    // Bits per value (typically 4)
  };
  
  @group(0) @binding(0) var<storage, read> kv_cache_fp32: array<f32>;       // Input full-precision cache
  @group(0) @binding(1) var<storage, read_write> kv_cache_quantized: array<u32>; // Output quantized cache
  @group(0) @binding(2) var<storage, read_write> scales: array<f32>;        // Scales (computed per head)
  @group(0) @binding(3) var<storage, read_write> zero_points: array<f32>;   // Zero points
  @group(0) @binding(4) var<uniform> params: Params;                        // Parameters
  
  // Helper function to find min/max values
  fn find_min_max(start_idx: u32, count: u32) -> vec2<f32> {
    var min_val: f32 = 3.402823e+38f; // FLT_MAX
    var max_val: f32 = -3.402823e+38f; // -FLT_MAX
    
    for (var i = 0u; i < count; i = i + 1u) {
      let val = kv_cache_fp32[start_idx + i];
      min_val = min(min_val, val);
      max_val = max(max_val, val);
    }
    
    return vec2<f32>(min_val, max_val);
  }
  
  // Pack multiple 4-bit values into a u32
  fn pack_values(values: array<u32, 8>) -> u32 {
    var result: u32 = 0u;
    
    for (var i = 0u; i < 8u; i = i + 1u) {
      result = result | (values[i] << (i * 4u));
    }
    
    return result;
  }
  
  @compute @workgroup_size(${workgroupSize})
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Calculate total elements per head
    let head_size = params.head_dim * params.seq_len;
    
    // Total number of heads across all batches and layers
    let total_heads = params.batch_size * params.num_layers * params.num_heads;
    
    // Each thread processes one head for quantization parameters
    if (idx < total_heads) {
      // Calculate starting index for this head's data
      let head_start = idx * head_size;
      
      // Find min/max values for this head
      let min_max = find_min_max(head_start, head_size);
      let min_val = min_max.x;
      let max_val = min_max.y;
      
      // Compute scale and zero point for asymmetric quantization
      let range = max_val - min_val;
      let scale = range / f32((1u << params.bits_per_value) - 1u);
      let zero_point = min_val / scale;
      
      // Store quantization parameters
      scales[idx] = scale;
      zero_points[idx] = zero_point;
      
      // Now quantize this head's data
      let values_per_u32 = 32u / params.bits_per_value; // 8 values per u32 for 4-bit
      
      for (var offset = 0u; offset < head_size; offset = offset + values_per_u32) {
        var quant_values: array<u32, 8>;
        let max_offset = min(values_per_u32, head_size - offset);
        
        // Quantize each batch of values
        for (var i = 0u; i < max_offset; i = i + 1u) {
          let fp_val = kv_cache_fp32[head_start + offset + i];
          var quant_val = u32(round((fp_val / scale) + zero_point));
          
          // Clamp to valid range
          quant_val = min(quant_val, (1u << params.bits_per_value) - 1u);
          quant_values[i] = quant_val;
        }
        
        // Pack values into a single u32
        let packed = pack_values(quant_values);
        
        // Write to output
        let out_idx = (head_start + offset) / values_per_u32;
        kv_cache_quantized[out_idx] = packed;
      }
    }
  }`;
}

/**
 * Generate shader for using quantized KV cache in attention computation
 * @param options Shader configuration options
 * @returns WGSL shader code for KV cache read/use operation
 */
export function generateKVCacheDequantizeShader(options: QuantizationShaderOptions = {}): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // Set to 4-bit as default for KV cache
  opts.bitsPerWeight = 4;
  
  // Use 256 as default workgroup size for KV cache operations
  const workgroupSize = opts.workgroupSize || 256;
  
  return /* wgsl */`
  // Specialized shader for using quantized KV cache in attention computation
  
  struct Params {
    batch_size: u32,        // Number of sequences
    num_layers: u32,        // Number of transformer layers
    num_heads: u32,         // Number of attention heads
    head_dim: u32,          // Dimension of each attention head
    seq_len: u32,           // Current sequence length
    bits_per_value: u32,    // Bits per value (typically 4)
  };
  
  @group(0) @binding(0) var<storage, read> kv_cache_quantized: array<u32>; // Quantized KV cache
  @group(0) @binding(1) var<storage, read_write> kv_cache_fp32: array<f32>; // Output dequantized cache
  @group(0) @binding(2) var<storage, read> scales: array<f32>;            // Scales (per head)
  @group(0) @binding(3) var<storage, read> zero_points: array<f32>;       // Zero points
  @group(0) @binding(4) var<uniform> params: Params;                       // Parameters
  
  // Helper function to unpack 4-bit values from a u32
  fn unpack_values(packed: u32) -> array<u32, 8> {
    var result: array<u32, 8>;
    
    for (var i = 0u; i < 8u; i = i + 1u) {
      result[i] = (packed >> (i * 4u)) & 0xFu;
    }
    
    return result;
  }
  
  @compute @workgroup_size(${workgroupSize})
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Calculate total elements
    let total_elements = params.batch_size * params.num_layers * params.num_heads * params.head_dim * params.seq_len;
    
    // Calculate elements per head
    let head_size = params.head_dim * params.seq_len;
    
    // Each thread processes one element
    if (idx < total_elements) {
      // Calculate which head this element belongs to
      let head_idx = idx / head_size;
      
      // Get scale and zero point for this head
      let scale = scales[head_idx];
      let zero_point = zero_points[head_idx];
      
      // Calculate packed index
      let values_per_u32 = 32u / params.bits_per_value; // 8 values per u32 for 4-bit
      let packed_idx = idx / values_per_u32;
      let value_pos = idx % values_per_u32;
      
      // Get packed value
      let packed = kv_cache_quantized[packed_idx];
      
      // Unpack values
      let unpacked = unpack_values(packed);
      let quant_val = unpacked[value_pos];
      
      // Dequantize
      let fp_val = scale * (f32(quant_val) - zero_point);
      
      // Write to output
      kv_cache_fp32[idx] = fp_val;
    }
  }`;
}

/**
 * Generate optimized 2-bit / 3-bit quantization shader for extreme compression
 * @param options Shader configuration options
 * @returns WGSL shader code
 */
export function generateUltraLowPrecisionQuantizeShader(options: QuantizationShaderOptions = {}): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // Ensure bits are 2 or 3 for ultra low precision
  const bitsPerWeight = (opts.bitsPerWeight === 2 || opts.bitsPerWeight === 3) ? opts.bitsPerWeight : 3;
  
  // Use 256 as default workgroup size
  const workgroupSize = opts.workgroupSize || 256;
  
  // Create the shader specifically optimized for either 2-bit or 3-bit
  if (bitsPerWeight === 2) {
    // 2-bit specific shader (each u32 packs 16 values)
    return /* wgsl */`
    // Ultra-low precision 2-bit quantization shader
    struct Params {
      tensor_size: u32,      // Total number of elements
      channel_size: u32,     // Elements per channel
      num_channels: u32,     // Number of channels
      per_channel_quant: u32 // Whether to use per-channel quantization
    };
    
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<u32>;
    @group(0) @binding(2) var<storage, read_write> scales: array<f32>;
    @group(0) @binding(3) var<storage, read_write> min_values: array<f32>;
    @group(0) @binding(4) var<uniform> params: Params;
    
    // Each u32 packs 16 2-bit values
    @compute @workgroup_size(${workgroupSize})
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let idx = global_id.x;
      
      // Each thread processes 16 values (packed into one u32)
      let values_per_thread = 16u;
      let base_idx = idx * values_per_thread;
      
      // Check bounds 
      if (base_idx >= params.tensor_size) {
        return;
      }
      
      // Determine which channel(s) this thread processes
      var channel_idx: u32 = 0u;
      if (params.per_channel_quant == 1u) {
        channel_idx = (base_idx / params.channel_size) % params.num_channels;
        
        // For values that might span channel boundaries, we simplify and use the main channel
        // A more complex implementation would handle cross-boundary packing properly
      }
      
      // Find min/max for this block of values
      var min_val: f32 = 3.402823e+38f; // FLT_MAX
      var max_val: f32 = -3.402823e+38f; // -FLT_MAX
      
      // Number of values to process (handling edge case)
      let num_values = min(values_per_thread, params.tensor_size - base_idx);
      
      // Find min/max
      for (var i = 0u; i < num_values; i = i + 1u) {
        let val = input[base_idx + i];
        min_val = min(min_val, val);
        max_val = max(max_val, val);
      }
      
      // Compute scale
      var scale: f32 = 0.0;
      if (max_val > min_val) {
        scale = (max_val - min_val) / 3.0; // 3 is max value in 2-bit representation (00, 01, 10, 11)
      }
      
      // Store scale and min value for dequantization
      scales[idx] = scale;
      min_values[idx] = min_val;
      
      // Quantize and pack values
      var packed_value: u32 = 0u;
      
      for (var i = 0u; i < num_values; i = i + 1u) {
        let val = input[base_idx + i];
        var quant_val: u32;
        
        if (scale > 0.0) {
          // Quantize to 0-3 range
          quant_val = min(u32((val - min_val) / scale), 3u);
        } else {
          quant_val = 0u; // If all values identical, quantize to 0
        }
        
        // Pack into output u32, 2 bits at a time
        packed_value = packed_value | (quant_val << (i * 2u));
      }
      
      // Write packed output
      output[idx] = packed_value;
    }`;
  } else {
    // 3-bit specific shader (each u32 packs 10 values)
    return /* wgsl */`
    // Ultra-low precision 3-bit quantization shader
    struct Params {
      tensor_size: u32,      // Total number of elements
      channel_size: u32,     // Elements per channel
      num_channels: u32,     // Number of channels
      per_channel_quant: u32 // Whether to use per-channel quantization
    };
    
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<u32>;
    @group(0) @binding(2) var<storage, read_write> scales: array<f32>;
    @group(0) @binding(3) var<storage, read_write> min_values: array<f32>;
    @group(0) @binding(4) var<uniform> params: Params;
    
    // Each u32 packs 10 3-bit values (30 bits used)
    @compute @workgroup_size(${workgroupSize})
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let idx = global_id.x;
      
      // Each thread processes 10 values (packed into one u32)
      let values_per_thread = 10u;
      let base_idx = idx * values_per_thread;
      
      // Check bounds 
      if (base_idx >= params.tensor_size) {
        return;
      }
      
      // Determine which channel(s) this thread processes
      var channel_idx: u32 = 0u;
      if (params.per_channel_quant == 1u) {
        channel_idx = (base_idx / params.channel_size) % params.num_channels;
      }
      
      // Find min/max for this block of values
      var min_val: f32 = 3.402823e+38f; // FLT_MAX
      var max_val: f32 = -3.402823e+38f; // -FLT_MAX
      
      // Number of values to process (handling edge case)
      let num_values = min(values_per_thread, params.tensor_size - base_idx);
      
      // Find min/max
      for (var i = 0u; i < num_values; i = i + 1u) {
        let val = input[base_idx + i];
        min_val = min(min_val, val);
        max_val = max(max_val, val);
      }
      
      // Compute scale
      var scale: f32 = 0.0;
      if (max_val > min_val) {
        scale = (max_val - min_val) / 7.0; // 7 is max value in 3-bit representation (0-7)
      }
      
      // Store scale and min value for dequantization
      scales[idx] = scale;
      min_values[idx] = min_val;
      
      // Quantize and pack values
      var packed_value: u32 = 0u;
      
      for (var i = 0u; i < num_values; i = i + 1u) {
        let val = input[base_idx + i];
        var quant_val: u32;
        
        if (scale > 0.0) {
          // Quantize to 0-7 range
          quant_val = min(u32((val - min_val) / scale), 7u);
        } else {
          quant_val = 0u; // If all values identical, quantize to 0
        }
        
        // Pack into output u32, 3 bits at a time
        packed_value = packed_value | (quant_val << (i * 3u));
      }
      
      // Write packed output
      output[idx] = packed_value;
    }`;
  }
}

/**
 * Generate shader for dequantizing ultra-low precision values (2-bit/3-bit)
 * @param options Shader configuration options
 * @returns WGSL shader code
 */
export function generateUltraLowPrecisionDequantizeShader(options: QuantizationShaderOptions = {}): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // Ensure bits are 2 or 3 for ultra low precision
  const bitsPerWeight = (opts.bitsPerWeight === 2 || opts.bitsPerWeight === 3) ? opts.bitsPerWeight : 3;
  
  // Use 256 as default workgroup size
  const workgroupSize = opts.workgroupSize || 256;
  
  // Values per u32 based on bit depth
  const valuesPerU32 = bitsPerWeight === 2 ? 16 : 10;
  
  // Create the shader specifically optimized for either 2-bit or 3-bit
  return /* wgsl */`
  // Ultra-low precision ${bitsPerWeight}-bit dequantization shader
  struct Params {
    tensor_size: u32      // Total number of elements
  };
  
  @group(0) @binding(0) var<storage, read> input: array<u32>;
  @group(0) @binding(1) var<storage, read_write> output: array<f32>;
  @group(0) @binding(2) var<storage, read> scales: array<f32>;
  @group(0) @binding(3) var<storage, read> min_values: array<f32>;
  @group(0) @binding(4) var<uniform> params: Params;
  
  // Each u32 packs ${valuesPerU32} ${bitsPerWeight}-bit values
  @compute @workgroup_size(${workgroupSize})
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Check bounds
    if (idx >= params.tensor_size) {
      return;
    }
    
    // Calculate which packed value this element belongs to
    let values_per_u32 = ${valuesPerU32}u;
    let packed_idx = idx / values_per_u32;
    let value_pos = idx % values_per_u32;
    
    // Get packed value
    let packed_value = input[packed_idx];
    
    // Get scale and min value for this block
    let scale = scales[packed_idx];
    let min_val = min_values[packed_idx];
    
    // Extract quantized value
    let mask = (1u << ${bitsPerWeight}u) - 1u;
    let quant_val = (packed_value >> (value_pos * ${bitsPerWeight}u)) & mask;
    
    // Dequantize
    let fp_val = min_val + scale * f32(quant_val);
    
    // Write output
    output[idx] = fp_val;
  }`;
}

/**
 * Generate specialized fused qkv matmul shader for transformer/attention block
 * Performs Q = XWq, K = XWk, V = XWv in one operation with quantized weights
 * @param options Shader configuration options
 * @returns WGSL shader code
 */
export function generateFusedQKVMatmulShader(options: QuantizationShaderOptions = {}): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // Use 4-bit quantization for weights
  opts.bitsPerWeight = 4;
  
  // Set workgroup size
  const workgroupSizeX = 8;
  const workgroupSizeY = 16;
  
  return /* wgsl */`
  // Fused QKV matmul shader with quantized weights
  // X: input tensor [batch_size, seq_len, hidden_size]
  // Wq, Wk, Wv: quantized weight matrices [hidden_size, head_size * num_heads]
  // Q, K, V: output tensors [batch_size, seq_len, num_heads, head_size]
  
  struct Dimensions {
    batch_size: u32,      // Batch size
    seq_len: u32,         // Sequence length
    hidden_size: u32,     // Hidden size
    num_heads: u32,       // Number of attention heads
    head_size: u32,       // Size of each attention head
  };
  
  @group(0) @binding(0) var<storage, read> input: array<f32>;         // Input tensor X
  @group(0) @binding(1) var<storage, read> weights_q: array<u32>;      // Quantized Q weights
  @group(0) @binding(2) var<storage, read> weights_k: array<u32>;      // Quantized K weights
  @group(0) @binding(3) var<storage, read> weights_v: array<u32>;      // Quantized V weights
  @group(0) @binding(4) var<storage, read_write> output_q: array<f32>; // Output Q
  @group(0) @binding(5) var<storage, read_write> output_k: array<f32>; // Output K
  @group(0) @binding(6) var<storage, read_write> output_v: array<f32>; // Output V
  @group(0) @binding(7) var<storage, read> q_scales: array<f32>;      // Scales for Q weights
  @group(0) @binding(8) var<storage, read> k_scales: array<f32>;      // Scales for K weights
  @group(0) @binding(9) var<storage, read> v_scales: array<f32>;      // Scales for V weights
  @group(0) @binding(10) var<uniform> dimensions: Dimensions;         // Dimensions
  
  // Helper function to unpack 4-bit values from a u32
  fn unpack_4bit_values(packed: u32) -> array<u32, 8> {
    var result: array<u32, 8>;
    
    for (var i = 0u; i < 8u; i = i + 1u) {
      result[i] = (packed >> (i * 4u)) & 0xFu;
    }
    
    return result;
  }
  
  @compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY}, 1)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Each thread processes one output element in Q, K, and V
    
    // Global thread coordinates
    let batch_seq = global_id.x; // Batch * seq_len index
    let head_dim = global_id.y;  // head * head_dim index
    let qkv_idx = global_id.z;   // 0=Q, 1=K, 2=V
    
    // Check bounds
    if (batch_seq >= dimensions.batch_size * dimensions.seq_len || 
        head_dim >= dimensions.num_heads * dimensions.head_size || 
        qkv_idx >= 3u) {
      return;
    }
    
    // Calculate batch and sequence indices
    let batch_idx = batch_seq / dimensions.seq_len;
    let seq_idx = batch_seq % dimensions.seq_len;
    
    // Calculate head and dimension indices
    let head_idx = head_dim / dimensions.head_size;
    let dim_idx = head_dim % dimensions.head_size;
    
    // Calculate the starting index for this sequence in the input
    let input_seq_offset = (batch_idx * dimensions.seq_len + seq_idx) * dimensions.hidden_size;
    
    // Calculate the output index
    let output_idx = batch_idx * (dimensions.seq_len * dimensions.num_heads * dimensions.head_size) +
                    seq_idx * (dimensions.num_heads * dimensions.head_size) +
                    head_idx * dimensions.head_size +
                    dim_idx;
    
    // Select the appropriate weight matrix and scales based on QKV index
    var weight_matrix: array<u32>;
    var scales: array<f32>;
    
    if (qkv_idx == 0u) {
      weight_matrix = weights_q;
      scales = q_scales;
    } else if (qkv_idx == 1u) {
      weight_matrix = weights_k;
      scales = k_scales;
    } else {
      weight_matrix = weights_v;
      scales = v_scales;
    }
    
    // Compute matrix multiplication (X * W) for this output element
    var sum: f32 = 0.0;
    
    // Weight column index
    let weight_col = head_idx * dimensions.head_size + dim_idx;
    
    // Process input hidden states, 8 elements at a time (4-bit packed weights)
    for (var h_base = 0u; h_base < dimensions.hidden_size; h_base = h_base + 8u) {
      // Calculate the weight matrix row offset
      let weight_offset = (h_base * (dimensions.num_heads * dimensions.head_size) + weight_col) / 8u;
      
      // Get packed weights
      let packed_weights = weight_matrix[weight_offset];
      
      // Unpack weights
      let unpacked_weights = unpack_4bit_values(packed_weights);
      
      // Get scale for this column
      let weight_scale = scales[weight_col];
      
      // Compute dot product for 8 elements
      for (var i = 0u; i < 8u; i = i + 1u) {
        let h_idx = h_base + i;
        
        if (h_idx < dimensions.hidden_size) {
          // Get input value
          let input_val = input[input_seq_offset + h_idx];
          
          // Dequantize weight (simple, can be expanded for zero point handling)
          let weight_val = f32(unpacked_weights[i]) * weight_scale;
          
          // Accumulate
          sum = sum + input_val * weight_val;
        }
      }
    }
    
    // Write output based on QKV index
    if (qkv_idx == 0u) {
      output_q[output_idx] = sum;
    } else if (qkv_idx == 1u) {
      output_k[output_idx] = sum;
    } else {
      output_v[output_idx] = sum;
    }
  }`;
}