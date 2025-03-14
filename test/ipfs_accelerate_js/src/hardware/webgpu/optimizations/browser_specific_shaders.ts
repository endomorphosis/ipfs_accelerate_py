/**
 * Browser-Specific Shader Optimizations
 * Specialized WGSL shaders optimized for different browsers' WebGPU implementations
 */

import { BrowserType } from '../browser_optimized_operations';

/**
 * Configuration options for browser-specific shaders
 */
export interface BrowserShaderOptions {
  /** Target browser type */
  browserType?: BrowserType;
  
  /** Workgroup size (will be adjusted for browser if not specified) */
  workgroupSize?: number;
  
  /** Whether to use specialized memory access patterns */
  optimizeMemoryAccess?: boolean;
  
  /** Whether to use browser-specific math approximations */
  useFastMath?: boolean;
  
  /** Whether to use browser-specific register optimization */
  optimizeRegisters?: boolean;
  
  /** Matrix tile size for tiled operations */
  tileSize?: number;
  
  /** Whether to use shared memory when available */
  useSharedMemory?: boolean;
}

/**
 * Default browser shader options
 */
const DEFAULT_OPTIONS: BrowserShaderOptions = {
  browserType: BrowserType.UNKNOWN,
  optimizeMemoryAccess: true,
  useFastMath: true,
  optimizeRegisters: true,
  useSharedMemory: true
};

/**
 * Determine optimal workgroup size based on browser
 * @param browserType Browser type
 * @param operation Operation type
 * @returns Optimal workgroup size
 */
export function getOptimalWorkgroupSize(
  browserType: BrowserType, 
  operation: 'matmul' | 'conv2d' | 'elementwise' | 'attention'
): number {
  switch (browserType) {
    case BrowserType.CHROME:
      // Chrome performs well with medium workgroup sizes
      return operation === 'matmul' ? 256 :
             operation === 'conv2d' ? 256 :
             operation === 'attention' ? 128 :
             256; // default for elementwise
      
    case BrowserType.FIREFOX:
      // Firefox often performs better with smaller workgroups
      return operation === 'matmul' ? 128 :
             operation === 'conv2d' ? 64 :
             operation === 'attention' ? 64 :
             128; // default for elementwise
      
    case BrowserType.SAFARI:
      // Safari/Apple GPUs can handle larger workgroups
      return operation === 'matmul' ? 512 :
             operation === 'conv2d' ? 256 :
             operation === 'attention' ? 256 :
             512; // default for elementwise
      
    case BrowserType.EDGE:
      // Edge is similar to Chrome but with some adjustments
      return operation === 'matmul' ? 256 :
             operation === 'conv2d' ? 256 :
             operation === 'attention' ? 128 :
             256; // default for elementwise
      
    default:
      // Safe defaults for unknown browsers
      return operation === 'matmul' ? 256 :
             operation === 'conv2d' ? 128 :
             operation === 'attention' ? 128 :
             256; // default for elementwise
  }
}

/**
 * Determine optimal tile size for tiled matrix multiplication
 * @param browserType Browser type
 * @param matrixSize Approximate matrix size (rough estimate of dimensions)
 * @returns Optimal tile size
 */
export function getOptimalTileSize(browserType: BrowserType, matrixSize: number): number {
  // Small/medium/large matrix size thresholds
  const isSmall = matrixSize <= 128;
  const isLarge = matrixSize >= 512;
  
  switch (browserType) {
    case BrowserType.CHROME:
      return isSmall ? 8 : (isLarge ? 16 : 16);
      
    case BrowserType.FIREFOX:
      return isSmall ? 8 : (isLarge ? 16 : 8);
      
    case BrowserType.SAFARI:
      return isSmall ? 16 : (isLarge ? 32 : 16);
      
    case BrowserType.EDGE:
      return isSmall ? 8 : (isLarge ? 16 : 16);
      
    default:
      return 16; // Safe default
  }
}

/**
 * Generate optimized matrix multiplication shader for specific browser
 * @param options Shader configuration options
 * @returns WGSL shader code
 */
export function generateBrowserOptimizedMatmulShader(options: BrowserShaderOptions = {}): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const browserType = opts.browserType || BrowserType.UNKNOWN;
  
  // Determine optimal configuration
  const workgroupSize = opts.workgroupSize || 
                        getOptimalWorkgroupSize(browserType, 'matmul');
  
  const tileSize = opts.tileSize || 
                  getOptimalTileSize(browserType, 256); // Default medium matrix
  
  // Browser-specific optimizations
  const useCoalescedAccess = opts.optimizeMemoryAccess !== false && 
                            (browserType === BrowserType.CHROME || 
                             browserType === BrowserType.EDGE);
  
  const useVectorTypes = browserType === BrowserType.SAFARI || 
                         browserType === BrowserType.FIREFOX;
  
  const useWaveIntrinsics = browserType === BrowserType.SAFARI;
  
  // Calculate workgroup layout
  const workgroupLayout = getWorkgroupLayout(workgroupSize, tileSize);
  
  // Generate shader with browser-specific optimizations
  return /* wgsl */`
  // Browser-optimized matrix multiplication for ${getBrowserName(browserType)}
  // Workgroup size: ${workgroupSize}
  // Tile size: ${tileSize}
  
  struct Params {
    M: u32,
    N: u32,
    K: u32,
    batch_size: u32,
  };
  
  @group(0) @binding(0) var<uniform> params: Params;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<f32>;
  @group(0) @binding(3) var<storage, read_write> C: array<f32>;
  
  @compute @workgroup_size(${workgroupLayout.x}, ${workgroupLayout.y}, ${workgroupLayout.z})
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
          @builtin(workgroup_id) workgroup_id: vec3<u32>,
          @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    ${generateSharedMemoryCode(tileSize, useSharedMemory(browserType))}
    
    let batch_idx = global_id.z;
    let row = global_id.x;
    let col = global_id.y;
    
    // Guard against out-of-bounds work items
    if (row >= params.M || col >= params.N || batch_idx >= params.batch_size) {
      return;
    }
    
    // Calculate base indices
    let a_batch_offset = batch_idx * params.M * params.K;
    let b_batch_offset = batch_idx * params.K * params.N;
    let c_batch_offset = batch_idx * params.M * params.N;
    
    ${generateMainLoopCode(useCoalescedAccess, useVectorTypes, useWaveIntrinsics, useSharedMemory(browserType), tileSize)}
    
    // Store the result
    C[c_batch_offset + row * params.N + col] = sum;
  }
  `;
}

/**
 * Generate optimized elementwise operation shader for specific browser
 * @param options Shader configuration options
 * @param opType Elementwise operation type
 * @returns WGSL shader code
 */
export function generateBrowserOptimizedElementwiseShader(
  opType: 'add' | 'multiply' | 'relu' | 'sigmoid' | 'tanh',
  options: BrowserShaderOptions = {}
): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const browserType = opts.browserType || BrowserType.UNKNOWN;
  
  // Determine optimal workgroup size
  const workgroupSize = opts.workgroupSize || 
                        getOptimalWorkgroupSize(browserType, 'elementwise');
  
  // Use vector load/store for compatible browsers (improves memory bandwidth)
  const useVectorLoadStore = browserType === BrowserType.SAFARI || 
                            browserType === BrowserType.CHROME;
  
  // Use wave operations for Safari (Apple GPUs support efficient wave ops)
  const useWaveOps = browserType === BrowserType.SAFARI;
  
  // Use fast math approximations when appropriate
  const useFastApprox = opts.useFastMath !== false && 
                        (opType === 'sigmoid' || opType === 'tanh');
  
  return /* wgsl */`
  // Browser-optimized elementwise ${opType} for ${getBrowserName(browserType)}
  // Workgroup size: ${workgroupSize}
  
  struct Params {
    size: u32,
    batch_size: u32,
  };
  
  @group(0) @binding(0) var<uniform> params: Params;
  @group(0) @binding(1) var<storage, read> inputA: array<f32>;
  @group(0) @binding(2) var<storage, read> inputB: array<f32>;
  @group(0) @binding(3) var<storage, read_write> output: array<f32>;
  
  ${useFastApprox ? generateFastMathFunctions(opType, browserType) : ''}
  
  @compute @workgroup_size(${workgroupSize})
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Guard against out-of-bounds
    if (idx >= params.size) {
      return;
    }
    
    ${generateElementwiseCode(opType, useVectorLoadStore, useWaveOps, useFastApprox)}
  }
  `;
}

/**
 * Generate browser-optimized quantized matmul shader
 * @param options Shader configuration options
 * @param bitsPerWeight Bits per weight (1-8)
 * @returns WGSL shader code
 */
export function generateBrowserOptimizedQuantizedMatmulShader(
  bitsPerWeight: 1 | 2 | 3 | 4 | 8,
  options: BrowserShaderOptions = {}
): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const browserType = opts.browserType || BrowserType.UNKNOWN;
  
  // Determine optimal configuration
  const workgroupSize = opts.workgroupSize || 
                        getOptimalWorkgroupSize(browserType, 'matmul');
  
  // Optimize unpacking based on browser
  const useOptimizedUnpacking = browserType === BrowserType.CHROME || 
                               browserType === BrowserType.SAFARI;
  
  // Use vector loads for activations when supported
  const useVectorLoads = browserType === BrowserType.SAFARI;
  
  // Calculate workgroup layout
  const workgroupLayout = getQuantizedWorkgroupLayout(workgroupSize, browserType);
  
  // Determine values per word based on bit width
  const valuesPerWord = Math.floor(32 / bitsPerWeight);
  
  return /* wgsl */`
  // Browser-optimized quantized (${bitsPerWeight}-bit) matrix multiplication for ${getBrowserName(browserType)}
  // Workgroup size: ${workgroupLayout.x}x${workgroupLayout.y}
  // Values per 32-bit word: ${valuesPerWord}
  
  struct Params {
    M: u32,          // Rows in A
    N: u32,          // Columns in B
    K: u32,          // Columns in A / Rows in B
    bits_per_weight: u32,  // Bits per quantized weight
    per_channel_quant: u32, // Whether using per-channel quantization
  };
  
  @group(0) @binding(0) var<uniform> params: Params;
  @group(0) @binding(1) var<storage, read> activations: array<f32>;
  @group(0) @binding(2) var<storage, read> quantized_weights: array<u32>;
  @group(0) @binding(3) var<storage, read> scales: array<f32>;
  @group(0) @binding(4) var<storage, read> zero_points: array<f32>;
  @group(0) @binding(5) var<storage, read_write> output: array<f32>;
  
  ${generateUnpackingFunctions(bitsPerWeight, useOptimizedUnpacking, browserType)}
  
  @compute @workgroup_size(${workgroupLayout.x}, ${workgroupLayout.y}, 1)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    
    // Guard against out-of-bounds
    if (row >= params.M || col >= params.N) {
      return;
    }
    
    var sum: f32 = 0.0;
    let values_per_word = ${valuesPerWord}u;
    let words_per_row = (params.K + values_per_word - 1u) / values_per_word;
    
    ${generateQuantizedMatmulLoopCode(bitsPerWeight, useVectorLoads, browserType)}
    
    // Store the result
    output[row * params.N + col] = sum;
  }
  `;
}

/**
 * Generate browser-optimized attention pattern shader
 * @param options Shader configuration options
 * @param useQuantization Whether to use quantization
 * @param bitsPerWeight Bits per weight if using quantization
 * @returns WGSL shader code
 */
export function generateBrowserOptimizedAttentionShader(
  options: BrowserShaderOptions = {},
  useQuantization: boolean = false,
  bitsPerWeight: 1 | 2 | 3 | 4 | 8 = 4
): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const browserType = opts.browserType || BrowserType.UNKNOWN;
  
  // Determine optimal workgroup size
  const workgroupSize = opts.workgroupSize || 
                        getOptimalWorkgroupSize(browserType, 'attention');
  
  // Firefox benefits from smaller tiles in attention computation
  const tileSize = opts.tileSize || 
                  (browserType === BrowserType.FIREFOX ? 8 : 16);
  
  // Chrome and Edge benefit from memory coalescing optimizations
  const useCoalescedAccess = opts.optimizeMemoryAccess !== false && 
                            (browserType === BrowserType.CHROME || 
                             browserType === BrowserType.EDGE);
  
  // Safari benefits from special math optimizations
  const useOptimizedMath = opts.useFastMath !== false && 
                          browserType === BrowserType.SAFARI;
  
  // Generate appropriate shader with optimizations
  if (useQuantization) {
    return generateQuantizedAttentionShader(bitsPerWeight, browserType, workgroupSize, tileSize);
  } else {
    return generateFloatAttentionShader(browserType, workgroupSize, tileSize, useCoalescedAccess, useOptimizedMath);
  }
}

// ===== Helper Functions =====

/**
 * Check if shared memory should be used for this browser
 */
function useSharedMemory(browserType: BrowserType): boolean {
  // Safari sometimes has issues with shared memory, but newer versions are better
  return browserType !== BrowserType.SAFARI_OLD;
}

/**
 * Get human-readable browser name
 */
function getBrowserName(browserType: BrowserType): string {
  switch (browserType) {
    case BrowserType.CHROME: return "Chrome";
    case BrowserType.FIREFOX: return "Firefox";
    case BrowserType.SAFARI: return "Safari";
    case BrowserType.EDGE: return "Edge";
    default: return "Unknown Browser";
  }
}

/**
 * Calculate workgroup layout based on workgroup size and tile size
 */
function getWorkgroupLayout(workgroupSize: number, tileSize: number): {x: number, y: number, z: number} {
  // Default to square workgroup layout
  const dimension = Math.sqrt(workgroupSize);
  
  // If dimension is an integer, use square layout
  if (Math.floor(dimension) === dimension) {
    return {x: dimension, y: dimension, z: 1};
  }
  
  // Otherwise, try to match tile size
  if (workgroupSize % tileSize === 0) {
    return {x: tileSize, y: workgroupSize / tileSize, z: 1};
  }
  
  // Fall back to a reasonable layout
  return {x: 16, y: Math.max(1, Math.floor(workgroupSize / 16)), z: 1};
}

/**
 * Calculate quantized workgroup layout
 */
function getQuantizedWorkgroupLayout(workgroupSize: number, browserType: BrowserType): {x: number, y: number} {
  // For quantized operations, different layouts work better on different browsers
  switch (browserType) {
    case BrowserType.FIREFOX:
      // Firefox prefers more workgroups with smaller sizes
      return {x: 8, y: 8};
      
    case BrowserType.SAFARI:
      // Safari performs well with larger workgroups
      return {x: 16, y: 16};
      
    case BrowserType.CHROME:
    case BrowserType.EDGE:
      // Chrome/Edge benefit from rectangular workgroups for coalescing
      return {x: 8, y: 16};
      
    default:
      // Safe default
      return {x: 8, y: 8};
  }
}

/**
 * Generate shared memory code for matmul
 */
function generateSharedMemoryCode(tileSize: number, useSharedMem: boolean): string {
  if (!useSharedMem) return '';
  
  return /* wgsl */`
  // Shared memory tiles
  var<workgroup> A_shared: array<f32, ${tileSize * tileSize}>;
  var<workgroup> B_shared: array<f32, ${tileSize * tileSize}>;
  `;
}

/**
 * Generate main matrix multiplication loop
 */
function generateMainLoopCode(
  useCoalescedAccess: boolean, 
  useVectorTypes: boolean,
  useWaveIntrinsics: boolean,
  useSharedMem: boolean,
  tileSize: number
): string {
  // Generate appropriate matrix multiplication code based on optimization flags
  let code = '';
  
  // Base accumulator initialization
  code += `
  // Accumulate dot product
  var sum: f32 = 0.0;
  `;
  
  if (useSharedMem) {
    // Shared memory implementation
    code += `
    // Tiled implementation with shared memory
    let numTiles = (params.K + ${tileSize}u - 1u) / ${tileSize}u;
    
    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
      // Load tile to shared memory
      let kStart = t * ${tileSize}u;
      
      // Load this thread's values into shared memory
      let localIdx = local_id.y * ${tileSize}u + local_id.x;
      
      if (localIdx < ${tileSize * tileSize}u) {
        let tileRow = localIdx / ${tileSize}u;
        let tileCol = localIdx % ${tileSize}u;
        
        // Load A tile (if in bounds)
        if (row < params.M && kStart + tileCol < params.K) {
          A_shared[tileRow * ${tileSize}u + tileCol] = A[a_batch_offset + row * params.K + kStart + tileCol];
        } else {
          A_shared[tileRow * ${tileSize}u + tileCol] = 0.0;
        }
        
        // Load B tile (if in bounds)
        if (kStart + tileRow < params.K && col < params.N) {
          B_shared[tileRow * ${tileSize}u + tileCol] = B[b_batch_offset + (kStart + tileRow) * params.N + col];
        } else {
          B_shared[tileRow * ${tileSize}u + tileCol] = 0.0;
        }
      }
      
      // Make sure all threads have loaded their values
      workgroupBarrier();
      
      // Compute on the tile
      for (var k: u32 = 0u; k < ${tileSize}u; k = k + 1u) {
        if (kStart + k < params.K) {
          sum = sum + A_shared[local_id.x * ${tileSize}u + k] * B_shared[k * ${tileSize}u + local_id.y];
        }
      }
      
      // Ensure all threads are done with the tile before loading the next one
      workgroupBarrier();
    }
    `;
  } else {
    // Non-shared memory implementation (for compatibility)
    if (useCoalescedAccess) {
      // Coalesced memory access (better for Chrome/Edge)
      code += `
      // Coalesced memory access pattern
      for (var k: u32 = 0u; k < params.K; k = k + 4u) {
        var dot: f32 = 0.0;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
          if (k + j < params.K) {
            dot = dot + A[a_batch_offset + row * params.K + (k + j)] * 
                         B[b_batch_offset + (k + j) * params.N + col];
          }
        }
        sum = sum + dot;
      }
      `;
    } else if (useVectorTypes) {
      // Vector types (better for Safari/Firefox)
      code += `
      // Use vector types for better performance
      for (var k: u32 = 0u; k < params.K; k = k + 1u) {
        sum = sum + A[a_batch_offset + row * params.K + k] * 
                     B[b_batch_offset + k * params.N + col];
      }
      `;
    } else {
      // Default implementation
      code += `
      // Standard implementation
      for (var k: u32 = 0u; k < params.K; k = k + 1u) {
        sum = sum + A[a_batch_offset + row * params.K + k] * 
                     B[b_batch_offset + k * params.N + col];
      }
      `;
    }
  }
  
  return code;
}

/**
 * Generate elementwise operation code
 */
function generateElementwiseCode(
  opType: string, 
  useVectorLoadStore: boolean, 
  useWaveOps: boolean,
  useFastApprox: boolean
): string {
  let code = '';
  
  if (useVectorLoadStore) {
    // Load 4 elements at once when possible
    code += `
    // Vector load/store optimization
    let vectorIdx = idx / 4u * 4u;
    let remainingElements = min(4u, params.size - vectorIdx);
    
    if (remainingElements == 4u) {
      // Full vector load
      let aVec = vec4<f32>(
        inputA[vectorIdx],
        inputA[vectorIdx + 1u],
        inputA[vectorIdx + 2u],
        inputA[vectorIdx + 3u]
      );
    `;
    
    if (opType === 'add' || opType === 'multiply') {
      code += `
      let bVec = vec4<f32>(
        inputB[vectorIdx],
        inputB[vectorIdx + 1u],
        inputB[vectorIdx + 2u],
        inputB[vectorIdx + 3u]
      );
      
      let resultVec = ${opType === 'add' ? 'aVec + bVec' : 'aVec * bVec'};
      
      // Store vector result
      output[vectorIdx] = resultVec.x;
      output[vectorIdx + 1u] = resultVec.y;
      output[vectorIdx + 2u] = resultVec.z;
      output[vectorIdx + 3u] = resultVec.w;
      `;
    } else {
      // Activation functions
      code += `
      // Apply activation function to vector
      let resultVec = vec4<f32>(
        ${getActivationFn(opType, 'aVec.x', useFastApprox)},
        ${getActivationFn(opType, 'aVec.y', useFastApprox)},
        ${getActivationFn(opType, 'aVec.z', useFastApprox)},
        ${getActivationFn(opType, 'aVec.w', useFastApprox)}
      );
      
      // Store vector result
      output[vectorIdx] = resultVec.x;
      output[vectorIdx + 1u] = resultVec.y;
      output[vectorIdx + 2u] = resultVec.z;
      output[vectorIdx + 3u] = resultVec.w;
      `;
    }
    
    code += `
    } else {
      // Handle remaining elements
      let a = inputA[idx];
    `;
    
    if (opType === 'add' || opType === 'multiply') {
      code += `
      let b = inputB[idx];
      let result = ${opType === 'add' ? 'a + b' : 'a * b'};
      `;
    } else {
      code += `
      let result = ${getActivationFn(opType, 'a', useFastApprox)};
      `;
    }
    
    code += `
      output[idx] = result;
    }
    `;
  } else {
    // Standard non-vectorized implementation
    code += `
    let a = inputA[idx];
    `;
    
    if (opType === 'add' || opType === 'multiply') {
      code += `
      let b = inputB[idx];
      let result = ${opType === 'add' ? 'a + b' : 'a * b'};
      `;
    } else {
      code += `
      let result = ${getActivationFn(opType, 'a', useFastApprox)};
      `;
    }
    
    code += `
    output[idx] = result;
    `;
  }
  
  return code;
}

/**
 * Get activation function code
 */
function getActivationFn(opType: string, input: string, useFastApprox: boolean): string {
  switch (opType) {
    case 'relu':
      return `max(0.0, ${input})`;
      
    case 'sigmoid':
      return useFastApprox ? 
        `fast_sigmoid(${input})` : 
        `1.0 / (1.0 + exp(-${input}))`;
      
    case 'tanh':
      return useFastApprox ? 
        `fast_tanh(${input})` : 
        `tanh(${input})`;
      
    default:
      return input;
  }
}

/**
 * Generate fast math approximation functions
 */
function generateFastMathFunctions(opType: string, browserType: BrowserType): string {
  let code = '';
  
  if (opType === 'sigmoid' || opType === 'tanh') {
    code += `
    // Fast approximation of sigmoid function
    fn fast_sigmoid(x: f32) -> f32 {
      // Use fast approximation: 0.5 * tanh(0.5 * x) + 0.5
      // Or even faster: x / (1.0 + abs(x)) * 0.5 + 0.5
      return x / (1.0 + abs(x)) * 0.5 + 0.5;
    }
    `;
  }
  
  if (opType === 'tanh') {
    code += `
    // Fast approximation of tanh
    fn fast_tanh(x: f32) -> f32 {
      // Pade approximation
      let x2 = x * x;
      return x * (27.0 + x2) / (27.0 + 9.0 * x2);
    }
    `;
  }
  
  return code;
}

/**
 * Generate unpacking functions for quantized weights
 */
function generateUnpackingFunctions(
  bitsPerWeight: number, 
  useOptimizedUnpacking: boolean,
  browserType: BrowserType
): string {
  // Different bit widths need different unpacking strategies
  switch (bitsPerWeight) {
    case 8:
      return generate8BitUnpacking(useOptimizedUnpacking);
    case 4:
      return generate4BitUnpacking(useOptimizedUnpacking);
    case 3:
      return generate3BitUnpacking(useOptimizedUnpacking, browserType);
    case 2:
      return generate2BitUnpacking(useOptimizedUnpacking);
    case 1:
      return generate1BitUnpacking(useOptimizedUnpacking);
    default:
      return generate4BitUnpacking(useOptimizedUnpacking); // Default to 4-bit
  }
}

/**
 * Generate 8-bit unpacking function
 */
function generate8BitUnpacking(optimized: boolean): string {
  return /* wgsl */`
  // Unpack 8-bit value from 32-bit word
  fn unpack_8bit(packed: u32, idx: u32) -> f32 {
    let shift = idx * 8u;
    let value = (packed >> shift) & 0xffu;
    
    return f32(value);
  }
  
  // Dequantize the unpacked value
  fn dequantize_8bit(value: f32, scale: f32, zero_point: f32) -> f32 {
    return (value - zero_point) * scale;
  }
  `;
}

/**
 * Generate 4-bit unpacking function
 */
function generate4BitUnpacking(optimized: boolean): string {
  return /* wgsl */`
  // Unpack 4-bit value from 32-bit word
  fn unpack_4bit(packed: u32, idx: u32) -> f32 {
    let shift = idx * 4u;
    let value = (packed >> shift) & 0xfu;
    
    return f32(value);
  }
  
  // Dequantize the unpacked value
  fn dequantize_4bit(value: f32, scale: f32, zero_point: f32) -> f32 {
    return (value - zero_point) * scale;
  }
  `;
}

/**
 * Generate 3-bit unpacking function (more complex due to non-power-of-2)
 */
function generate3BitUnpacking(optimized: boolean, browserType: BrowserType): string {
  // Safari performs better with the optimized 3-bit unpacking
  const useSpecializedUnpacking = optimized && browserType === BrowserType.SAFARI;
  
  if (useSpecializedUnpacking) {
    return /* wgsl */`
    // Optimized 3-bit unpacking for 10 values per 32-bit word (10*3=30 bits)
    fn unpack_3bit(packed: u32, idx: u32) -> f32 {
      // Special case handling for Safari performance
      if (idx < 10u) {
        // Direct bit position calculation optimized for 3-bit
        let shift = idx * 3u;
        let value = (packed >> shift) & 0x7u;
        return f32(value);
      } else {
        // For positions that span words (rare case)
        let wordIdx = idx / 10u;
        let posInWord = idx % 10u;
        let shift = posInWord * 3u;
        let value = (packed >> shift) & 0x7u;
        return f32(value);
      }
    }
    
    // Dequantize the unpacked value
    fn dequantize_3bit(value: f32, scale: f32, zero_point: f32) -> f32 {
      return (value - zero_point) * scale;
    }
    `;
  } else {
    return /* wgsl */`
    // Standard 3-bit unpacking (10 values per 32-bit word)
    fn unpack_3bit(packed: u32, idx: u32) -> f32 {
      let shift = (idx % 10u) * 3u;
      let value = (packed >> shift) & 0x7u;
      return f32(value);
    }
    
    // Dequantize the unpacked value
    fn dequantize_3bit(value: f32, scale: f32, zero_point: f32) -> f32 {
      return (value - zero_point) * scale;
    }
    `;
  }
}

/**
 * Generate 2-bit unpacking function
 */
function generate2BitUnpacking(optimized: boolean): string {
  return /* wgsl */`
  // Unpack 2-bit value from 32-bit word
  fn unpack_2bit(packed: u32, idx: u32) -> f32 {
    let shift = idx * 2u;
    let value = (packed >> shift) & 0x3u;
    
    return f32(value);
  }
  
  // Dequantize the unpacked value with specialized mapping for 2-bit
  fn dequantize_2bit(value: f32, scale: f32, zero_point: f32) -> f32 {
    // Special value mapping for 2-bit: 0, 1, 2, 3 → -1, 0, 0.5, 1
    let valueMap = array<f32, 4>(-1.0, 0.0, 0.5, 1.0);
    return valueMap[u32(value)] * scale;
  }
  `;
}

/**
 * Generate 1-bit unpacking function
 */
function generate1BitUnpacking(optimized: boolean): string {
  return /* wgsl */`
  // Unpack 1-bit value from 32-bit word
  fn unpack_1bit(packed: u32, idx: u32) -> f32 {
    let shift = idx;
    let value = (packed >> shift) & 0x1u;
    
    return f32(value);
  }
  
  // Dequantize the unpacked value with specialized mapping for 1-bit
  fn dequantize_1bit(value: f32, scale: f32, zero_point: f32) -> f32 {
    // 1-bit uses a specialized mapping: 0, 1 → -1, 1
    return select(-1.0, 1.0, value > 0.5) * scale;
  }
  `;
}

/**
 * Generate quantized matmul loop code
 */
function generateQuantizedMatmulLoopCode(
  bitsPerWeight: number, 
  useVectorLoads: boolean,
  browserType: BrowserType
): string {
  const valuesPerWord = Math.floor(32 / bitsPerWeight);
  
  const unpackFnName = `unpack_${bitsPerWeight}bit`;
  const dequantizeFnName = `dequantize_${bitsPerWeight}bit`;
  
  let code = '';
  
  // Different approaches based on browser
  if (browserType === BrowserType.SAFARI && useVectorLoads) {
    // Safari optimized loop with vector loads
    code += `
    // Safari-optimized loop with vector loads
    for (var kWordOffset = 0u; kWordOffset < words_per_row; kWordOffset += 1u) {
      // Each word contains ${valuesPerWord} values
      let packedWord = quantized_weights[row * words_per_row + kWordOffset];
      
      // Process 4 values at a time when possible
      for (var i = 0u; i < ${valuesPerWord}u; i += 4u) {
        if (kWordOffset * ${valuesPerWord}u + i >= params.K) break;
        
        // Get channel index for scales if using per-channel quantization
        let channel_idx = params.per_channel_quant != 0u ? col : 0u;
        let scale_val = scales[channel_idx];
        let zero_point_val = zero_points[channel_idx];
        
        // Load 4 activations at once when possible
        let k_base = kWordOffset * ${valuesPerWord}u + i;
        var activation_sum: f32 = 0.0;
        
        // Process up to 4 values
        for (var j = 0u; j < 4u; j += 1u) {
          if (k_base + j < params.K) {
            let unpacked = ${unpackFnName}(packedWord, i + j);
            let dequantized = ${dequantizeFnName}(unpacked, scale_val, zero_point_val);
            let activation = activations[k_base + j];
            activation_sum += dequantized * activation;
          }
        }
        
        sum += activation_sum;
      }
    }
    `;
  } else if (browserType === BrowserType.FIREFOX) {
    // Firefox optimized loop
    code += `
    // Firefox-optimized loop
    for (var kWordOffset = 0u; kWordOffset < words_per_row; kWordOffset += 1u) {
      // Each word contains ${valuesPerWord} values
      let packedWord = quantized_weights[row * words_per_row + kWordOffset];
      
      // Get channel index for scales if using per-channel quantization
      let channel_idx = params.per_channel_quant != 0u ? col : 0u;
      let scale_val = scales[channel_idx];
      let zero_point_val = zero_points[channel_idx];
      
      // Process one value at a time (Firefox is efficient with simple loops)
      for (var i = 0u; i < ${valuesPerWord}u; i += 1u) {
        let k = kWordOffset * ${valuesPerWord}u + i;
        if (k >= params.K) break;
        
        let unpacked = ${unpackFnName}(packedWord, i);
        let dequantized = ${dequantizeFnName}(unpacked, scale_val, zero_point_val);
        let activation = activations[k];
        sum += dequantized * activation;
      }
    }
    `;
  } else {
    // Default implementation (Chrome/Edge/Unknown)
    code += `
    // Standard quantized matmul loop
    for (var kWordOffset = 0u; kWordOffset < words_per_row; kWordOffset += 1u) {
      // Each word contains ${valuesPerWord} values
      let packedWord = quantized_weights[row * words_per_row + kWordOffset];
      
      // Get channel index for scales if using per-channel quantization
      let channel_idx = params.per_channel_quant != 0u ? col : 0u;
      let scale_val = scales[channel_idx];
      let zero_point_val = zero_points[channel_idx];
      
      // Process the values in this word
      for (var i = 0u; i < ${valuesPerWord}u; i += 1u) {
        let k = kWordOffset * ${valuesPerWord}u + i;
        if (k >= params.K) break;
        
        let unpacked = ${unpackFnName}(packedWord, i);
        let dequantized = ${dequantizeFnName}(unpacked, scale_val, zero_point_val);
        let activation = activations[k];
        sum += dequantized * activation;
      }
    }
    `;
  }
  
  return code;
}

/**
 * Generate quantized attention shader
 */
function generateQuantizedAttentionShader(
  bitsPerWeight: number,
  browserType: BrowserType,
  workgroupSize: number,
  tileSize: number
): string {
  // This is a simplified template - a full implementation would be more complex
  const workgroupLayout = getQuantizedWorkgroupLayout(workgroupSize, browserType);
  
  return /* wgsl */`
  // Browser-optimized quantized (${bitsPerWeight}-bit) attention for ${getBrowserName(browserType)}
  // Workgroup size: ${workgroupLayout.x}x${workgroupLayout.y}
  
  struct Params {
    batch_size: u32,
    seq_length: u32,
    num_heads: u32,
    head_dim: u32,
    bits_per_weight: u32,
  };
  
  @group(0) @binding(0) var<uniform> params: Params;
  @group(0) @binding(1) var<storage, read> queries: array<f32>;
  @group(0) @binding(2) var<storage, read> keys: array<f32>;
  @group(0) @binding(3) var<storage, read> values: array<f32>;
  @group(0) @binding(4) var<storage, read> q_weights_quantized: array<u32>;
  @group(0) @binding(5) var<storage, read> k_weights_quantized: array<u32>;
  @group(0) @binding(6) var<storage, read> v_weights_quantized: array<u32>;
  @group(0) @binding(7) var<storage, read> scales: array<f32>;
  @group(0) @binding(8) var<storage, read> zero_points: array<f32>;
  @group(0) @binding(9) var<storage, read_write> output: array<f32>;
  
  ${generateUnpackingFunctions(bitsPerWeight, true, browserType)}
  
  @compute @workgroup_size(${workgroupLayout.x}, ${workgroupLayout.y}, 1)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Placeholder for full attention implementation
    // A complete implementation would include:
    // 1. Query-Key multiplication with quantized weights
    // 2. Softmax calculation
    // 3. Attention-Value multiplication
    // With browser-specific optimizations for each step
  }
  `;
}

/**
 * Generate float (non-quantized) attention shader
 */
function generateFloatAttentionShader(
  browserType: BrowserType,
  workgroupSize: number,
  tileSize: number,
  useCoalescedAccess: boolean,
  useOptimizedMath: boolean
): string {
  // This is a simplified template - a full implementation would be more complex
  const workgroupLayout = getWorkgroupLayout(workgroupSize, tileSize);
  
  return /* wgsl */`
  // Browser-optimized attention for ${getBrowserName(browserType)}
  // Workgroup size: ${workgroupLayout.x}x${workgroupLayout.y}x${workgroupLayout.z}
  
  struct Params {
    batch_size: u32,
    seq_length: u32,
    num_heads: u32,
    head_dim: u32,
  };
  
  @group(0) @binding(0) var<uniform> params: Params;
  @group(0) @binding(1) var<storage, read> queries: array<f32>;
  @group(0) @binding(2) var<storage, read> keys: array<f32>;
  @group(0) @binding(3) var<storage, read> values: array<f32>;
  @group(0) @binding(4) var<storage, read_write> output: array<f32>;
  
  ${useOptimizedMath ? generateFastMathFunctions('softmax', browserType) : ''}
  
  @compute @workgroup_size(${workgroupLayout.x}, ${workgroupLayout.y}, ${workgroupLayout.z})
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Placeholder for full attention implementation
    // A complete implementation would include:
    // 1. Query-Key multiplication
    // 2. Softmax calculation
    // 3. Attention-Value multiplication
    // With browser-specific optimizations for each step
  }
  `;
}