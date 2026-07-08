/**
 * Browser-Specific Shader Loader
 * Loads the appropriate browser-optimized shaders at runtime
 */

import { BrowserType, detectBrowserType } from '../browser_optimized_operations';
import {
  CHROME_MULTI_HEAD_ATTENTION_SHADER,
  FIREFOX_MULTI_HEAD_ATTENTION_SHADER,
  SAFARI_MULTI_HEAD_ATTENTION_SHADER,
  EDGE_MULTI_HEAD_ATTENTION_SHADER
} from './browser_specific_attention_shaders';
import {
  CHROME_SOFTMAX_SHADER,
  FIREFOX_SOFTMAX_SHADER,
  SAFARI_SOFTMAX_SHADER,
  EDGE_SOFTMAX_SHADER
} from './browser_specific_softmax_shaders';

/**
 * Available shader types enumeration
 */
export enum ShaderType {
  MATMUL = 'matmul',
  QUANTIZED_MATMUL = 'quantized_matmul',
  ELEMENTWISE = 'elementwise',
  ELEMENTWISE_RELU = 'elementwise_relu',
  ELEMENTWISE_ADD = 'elementwise_add',
  ELEMENTWISE_TANH = 'elementwise_tanh',
  ELEMENTWISE_SIGMOID = 'elementwise_sigmoid',
  ELEMENTWISE_GELU = 'elementwise_gelu',
  ELEMENTWISE_SILU = 'elementwise_silu',
  ELEMENTWISE_LEAKY_RELU = 'elementwise_leaky_relu',
  LAYER_NORMALIZATION = 'layer_normalization',
  MULTI_HEAD_ATTENTION = 'multi_head_attention',
  SOFTMAX = 'softmax',
  ATTENTION = 'attention'
}

/**
 * Cache for loaded shaders to avoid redundant loads
 * Key format: browserType_shaderType
 */
const shaderCache: Map<string, string> = new Map();

/**
 * Resolves the path to the appropriate shader file based on browser type and shader type
 * @param browserType The browser type to get the shader for
 * @param shaderType The type of shader to load
 * @returns Path to the shader file
 */
function resolveShaderPath(browserType: BrowserType, shaderType: ShaderType): string {
  let browserName: string;
  
  switch (browserType) {
    case BrowserType.CHROME:
      browserName = 'chrome';
      break;
    case BrowserType.FIREFOX:
      browserName = 'firefox';
      break;
    case BrowserType.SAFARI:
      browserName = 'safari';
      break;
    case BrowserType.EDGE:
      browserName = 'edge';
      break;
    default:
      browserName = 'chrome'; // Default to Chrome shaders if unknown
  }
  
  return `/src/hardware/webgpu/optimizations/browser_shaders/${browserName}_${shaderType}.wgsl`;
}

/**
 * In browser environments, fetch the shader from the server
 * @param path Path to the shader file
 * @returns Shader code as string
 */
async function fetchShader(path: string): Promise<string> {
  try {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`Failed to fetch shader: ${response.status} ${response.statusText}`);
    }
    return await response.text();
  } catch (error) {
    console.error(`Error loading shader from ${path}:`, error);
    throw error;
  }
}

/**
 * Gets embedded shader based on shader type and browser type
 * Used when fetching from server isn't possible
 * @param browserType Browser type
 * @param shaderType Shader type
 * @returns Embedded shader code
 */
function getEmbeddedShader(browserType: BrowserType, shaderType: ShaderType): string {
  // For matmul shaders
  if (shaderType === ShaderType.MATMUL) {
    // Return appropriate shader based on browser type
    switch (browserType) {
      case BrowserType.CHROME:
        return CHROME_MATMUL_SHADER;
      case BrowserType.FIREFOX:
        return FIREFOX_MATMUL_SHADER;
      case BrowserType.SAFARI:
        return SAFARI_MATMUL_SHADER;
      case BrowserType.EDGE:
        return EDGE_MATMUL_SHADER;
      default:
        return CHROME_MATMUL_SHADER; // Default to Chrome
    }
  }
  
  // For quantized matmul shaders
  if (shaderType === ShaderType.QUANTIZED_MATMUL) {
    switch (browserType) {
      case BrowserType.CHROME:
        return CHROME_QUANTIZED_MATMUL_SHADER;
      case BrowserType.FIREFOX:
        return FIREFOX_QUANTIZED_MATMUL_SHADER;
      case BrowserType.SAFARI:
        return SAFARI_QUANTIZED_MATMUL_SHADER;
      case BrowserType.EDGE:
        return EDGE_QUANTIZED_MATMUL_SHADER;
      default:
        return CHROME_QUANTIZED_MATMUL_SHADER; // Default to Chrome
    }
  }
  
  // For elementwise ReLU shaders
  if (shaderType === ShaderType.ELEMENTWISE_RELU) {
    switch (browserType) {
      case BrowserType.CHROME:
        return CHROME_ELEMENTWISE_RELU_SHADER;
      case BrowserType.FIREFOX:
        return FIREFOX_ELEMENTWISE_RELU_SHADER;
      case BrowserType.SAFARI:
        return SAFARI_ELEMENTWISE_RELU_SHADER;
      case BrowserType.EDGE:
        return EDGE_ELEMENTWISE_RELU_SHADER;
      default:
        return CHROME_ELEMENTWISE_RELU_SHADER; // Default to Chrome
    }
  }
  
  // For elementwise Add shaders
  if (shaderType === ShaderType.ELEMENTWISE_ADD) {
    switch (browserType) {
      case BrowserType.CHROME:
        return CHROME_ELEMENTWISE_ADD_SHADER;
      case BrowserType.FIREFOX:
        return FIREFOX_ELEMENTWISE_ADD_SHADER;
      case BrowserType.SAFARI:
        return SAFARI_ELEMENTWISE_ADD_SHADER;
      case BrowserType.EDGE:
        return EDGE_ELEMENTWISE_ADD_SHADER;
      default:
        return CHROME_ELEMENTWISE_ADD_SHADER; // Default to Chrome
    }
  }
  
  // For elementwise Tanh shaders
  if (shaderType === ShaderType.ELEMENTWISE_TANH) {
    switch (browserType) {
      case BrowserType.CHROME:
        return CHROME_ELEMENTWISE_TANH_SHADER;
      case BrowserType.FIREFOX:
        return FIREFOX_ELEMENTWISE_TANH_SHADER;
      case BrowserType.SAFARI:
        return SAFARI_ELEMENTWISE_TANH_SHADER;
      case BrowserType.EDGE:
        return EDGE_ELEMENTWISE_TANH_SHADER;
      default:
        return CHROME_ELEMENTWISE_TANH_SHADER; // Default to Chrome
    }
  }
  
  // For elementwise Sigmoid shaders
  if (shaderType === ShaderType.ELEMENTWISE_SIGMOID) {
    switch (browserType) {
      case BrowserType.CHROME:
        return CHROME_ELEMENTWISE_SIGMOID_SHADER;
      case BrowserType.FIREFOX:
        return FIREFOX_ELEMENTWISE_SIGMOID_SHADER;
      case BrowserType.SAFARI:
        return SAFARI_ELEMENTWISE_SIGMOID_SHADER;
      case BrowserType.EDGE:
        return EDGE_ELEMENTWISE_SIGMOID_SHADER;
      default:
        return CHROME_ELEMENTWISE_SIGMOID_SHADER; // Default to Chrome
    }
  }
  
  // For elementwise GELU shaders
  if (shaderType === ShaderType.ELEMENTWISE_GELU) {
    switch (browserType) {
      case BrowserType.CHROME:
        return CHROME_ELEMENTWISE_GELU_SHADER;
      case BrowserType.FIREFOX:
        return FIREFOX_ELEMENTWISE_GELU_SHADER;
      case BrowserType.SAFARI:
        return SAFARI_ELEMENTWISE_GELU_SHADER;
      case BrowserType.EDGE:
        return EDGE_ELEMENTWISE_GELU_SHADER;
      default:
        return CHROME_ELEMENTWISE_GELU_SHADER; // Default to Chrome
    }
  }
  
  // For elementwise SiLU shaders
  if (shaderType === ShaderType.ELEMENTWISE_SILU) {
    switch (browserType) {
      case BrowserType.CHROME:
        return CHROME_ELEMENTWISE_SILU_SHADER;
      case BrowserType.FIREFOX:
        return FIREFOX_ELEMENTWISE_SILU_SHADER;
      case BrowserType.SAFARI:
        return SAFARI_ELEMENTWISE_SILU_SHADER;
      case BrowserType.EDGE:
        return EDGE_ELEMENTWISE_SILU_SHADER;
      default:
        return CHROME_ELEMENTWISE_SILU_SHADER; // Default to Chrome
    }
  }
  
  // For elementwise Leaky ReLU shaders
  if (shaderType === ShaderType.ELEMENTWISE_LEAKY_RELU) {
    switch (browserType) {
      case BrowserType.CHROME:
        return CHROME_ELEMENTWISE_LEAKY_RELU_SHADER;
      case BrowserType.FIREFOX:
        return FIREFOX_ELEMENTWISE_LEAKY_RELU_SHADER;
      case BrowserType.SAFARI:
        return SAFARI_ELEMENTWISE_LEAKY_RELU_SHADER;
      case BrowserType.EDGE:
        return EDGE_ELEMENTWISE_LEAKY_RELU_SHADER;
      default:
        return CHROME_ELEMENTWISE_LEAKY_RELU_SHADER; // Default to Chrome
    }
  }
  
  // For Layer Normalization shaders
  if (shaderType === ShaderType.LAYER_NORMALIZATION) {
    switch (browserType) {
      case BrowserType.CHROME:
        return CHROME_LAYER_NORMALIZATION_SHADER;
      case BrowserType.FIREFOX:
        return FIREFOX_LAYER_NORMALIZATION_SHADER;
      case BrowserType.SAFARI:
        return SAFARI_LAYER_NORMALIZATION_SHADER;
      case BrowserType.EDGE:
        return EDGE_LAYER_NORMALIZATION_SHADER;
      default:
        return CHROME_LAYER_NORMALIZATION_SHADER; // Default to Chrome
    }
  }
  
  // For Multi-Head Attention shaders
  if (shaderType === ShaderType.MULTI_HEAD_ATTENTION) {
    switch (browserType) {
      case BrowserType.CHROME:
        return CHROME_MULTI_HEAD_ATTENTION_SHADER;
      case BrowserType.FIREFOX:
        return FIREFOX_MULTI_HEAD_ATTENTION_SHADER;
      case BrowserType.SAFARI:
        return SAFARI_MULTI_HEAD_ATTENTION_SHADER;
      case BrowserType.EDGE:
        return EDGE_MULTI_HEAD_ATTENTION_SHADER;
      default:
        return CHROME_MULTI_HEAD_ATTENTION_SHADER; // Default to Chrome
    }
  }
  
  // For Softmax shaders
  if (shaderType === ShaderType.SOFTMAX) {
    switch (browserType) {
      case BrowserType.CHROME:
        return CHROME_SOFTMAX_SHADER;
      case BrowserType.FIREFOX:
        return FIREFOX_SOFTMAX_SHADER;
      case BrowserType.SAFARI:
        return SAFARI_SOFTMAX_SHADER;
      case BrowserType.EDGE:
        return EDGE_SOFTMAX_SHADER;
      default:
        return CHROME_SOFTMAX_SHADER; // Default to Chrome
    }
  }
  
  // Generic elementwise shader (fallback)
  if (shaderType === ShaderType.ELEMENTWISE) {
    switch (browserType) {
      case BrowserType.CHROME:
        return CHROME_ELEMENTWISE_RELU_SHADER; // Use ReLU as default elementwise
      case BrowserType.FIREFOX:
        return FIREFOX_ELEMENTWISE_RELU_SHADER;
      case BrowserType.SAFARI:
        return SAFARI_ELEMENTWISE_RELU_SHADER;
      case BrowserType.EDGE:
        return EDGE_ELEMENTWISE_RELU_SHADER;
      default:
        return CHROME_ELEMENTWISE_RELU_SHADER;
    }
  }
  
  // Default to Chrome matmul shader if no match
  return CHROME_MATMUL_SHADER;
}

/**
 * Loads the browser-optimized shader for the specified operation
 * @param shaderType Type of shader to load
 * @param browserType Optional browser type override (autodetects if not provided)
 * @returns Shader code as string
 */
export async function loadBrowserShader(
  shaderType: ShaderType, 
  browserType?: BrowserType
): Promise<string> {
  // Detect browser type if not provided
  const detectedBrowserType = browserType || detectBrowserType();
  
  // Create a cache key
  const cacheKey = `${detectedBrowserType}_${shaderType}`;
  
  // Check if shader is already cached
  if (shaderCache.has(cacheKey)) {
    return shaderCache.get(cacheKey)!;
  }
  
  let shaderCode: string;
  
  // Try to fetch shader if in browser environment
  if (typeof window !== 'undefined' && typeof fetch === 'function') {
    try {
      const shaderPath = resolveShaderPath(detectedBrowserType, shaderType);
      shaderCode = await fetchShader(shaderPath);
    } catch (error) {
      // Fall back to embedded shaders if fetch fails
      console.warn('Failed to fetch shader, using embedded shader', error);
      shaderCode = getEmbeddedShader(detectedBrowserType, shaderType);
    }
  } else {
    // Use embedded shaders in non-browser environments
    shaderCode = getEmbeddedShader(detectedBrowserType, shaderType);
  }
  
  // Cache the shader
  shaderCache.set(cacheKey, shaderCode);
  
  return shaderCode;
}

/**
 * Gets a browser-specific shader synchronously (from cache or embedded)
 * @param shaderType Type of shader to get
 * @param browserType Optional browser type override (autodetects if not provided)
 * @returns Shader code as string
 */
export function getBrowserShaderSync(
  shaderType: ShaderType, 
  browserType?: BrowserType
): string {
  // Detect browser type if not provided
  const detectedBrowserType = browserType || detectBrowserType();
  
  // Create a cache key
  const cacheKey = `${detectedBrowserType}_${shaderType}`;
  
  // Check if shader is already cached
  if (shaderCache.has(cacheKey)) {
    return shaderCache.get(cacheKey)!;
  }
  
  // Get embedded shader
  const shaderCode = getEmbeddedShader(detectedBrowserType, shaderType);
  
  // Cache the shader
  shaderCache.set(cacheKey, shaderCode);
  
  return shaderCode;
}

// ===== Embedded Shaders =====
// These are used when fetching from server is not possible

// ===== Elementwise Operation Shaders =====

// Utility functions for fast math approximations
const FAST_MATH_FUNCTIONS = `
// Fast approximation of sigmoid: x / (1.0 + abs(x)) * 0.5 + 0.5
fn fast_sigmoid(x: f32) -> f32 {
  return x / (1.0 + abs(x)) * 0.5 + 0.5;
}

// Fast approximation of tanh using a Pade approximation
fn fast_tanh(x: f32) -> f32 {
  let x2 = x * x;
  return x * (27.0 + x2) / (27.0 + 9.0 * x2);
}

// Standard sigmoid implementation with clamping for numerical stability
fn sigmoid(x: f32) -> f32 {
  // Clamp to avoid overflow
  let clamped = clamp(x, -30.0, 30.0);
  return 1.0 / (1.0 + exp(-clamped));
}

// Standard tanh implementation using the formula: 2 * sigmoid(2x) - 1
fn tanh_approx(x: f32) -> f32 {
  return 2.0 * sigmoid(2.0 * x) - 1.0;
}

// Constants for GELU calculations
const SQRT_2_OVER_PI: f32 = 0.7978845608; // sqrt(2/π)
const GELU_COEF1: f32 = 0.044715;

// Standard GELU implementation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
fn gelu(x: f32) -> f32 {
  let x3 = x * x * x;
  let inner = SQRT_2_OVER_PI * (x + GELU_COEF1 * x3);
  return x * 0.5 * (1.0 + tanh_approx(inner));
}

// Fast approximation of GELU using simpler polynomial
// Approximation: x * sigmoid(1.702 * x)
fn fast_gelu(x: f32) -> f32 {
  return x * sigmoid(1.702 * x);
}

// SiLU (Swish) implementation: x * sigmoid(x)
fn silu(x: f32) -> f32 {
  return x * sigmoid(x);
}

// Fast approximation of SiLU using fast sigmoid
fn fast_silu(x: f32) -> f32 {
  return x * fast_sigmoid(x);
}

// Leaky ReLU implementation: max(alpha * x, x)
fn leaky_relu(x: f32, alpha: f32) -> f32 {
  return max(alpha * x, x);
}

// Standard Leaky ReLU with default alpha value of 0.01
fn std_leaky_relu(x: f32) -> f32 {
  return leaky_relu(x, 0.01);
}
`;

const FIREFOX_ELEMENTWISE_RELU_SHADER = `/**
 * Firefox-optimized elementwise ReLU shader
 * 
 * Optimized for Firefox's WebGPU implementation with:
 * - Smaller workgroup size (128) for better thread occupancy
 * - Simple, non-vectorized operations that perform well in Firefox
 * - Avoids complex control flow that can be problematic in Firefox's shader compiler
 */

struct Params {
  size: u32,
  batch_size: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Guard against out-of-bounds
  if (idx >= params.size) {
    return;
  }
  
  // Simple loop-based processing for Firefox
  let value = input[idx];
  output[idx] = max(0.0, value);
}`;

const EDGE_ELEMENTWISE_RELU_SHADER = `/**
 * Edge-optimized elementwise ReLU shader
 * 
 * Optimized for Edge's WebGPU implementation with:
 * - Uses 256 threads per workgroup
 * - Processes multiple elements per thread when possible
 * - Uses partial loop unrolling for better performance on Edge
 */

struct Params {
  size: u32,
  batch_size: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global_idx = global_id.x;
  let stride = 256u * 4u; // Process 4 elements per thread when possible
  
  // Process multiple elements per thread for better utilization
  for (var i = 0u; i < 4u; i++) {
    let idx = global_idx + i * 256u;
    
    // Edge's shader compiler benefits from explicit bounds checking
    if (idx < params.size) {
      let value = input[idx];
      output[idx] = max(0.0, value);
    }
  }
}`;

const CHROME_ELEMENTWISE_RELU_SHADER = `/**
 * Chrome-optimized elementwise ReLU shader
 * 
 * Optimized for Chrome's WebGPU implementation with:
 * - Uses 256 threads per workgroup for balanced performance
 * - Processes elements in vectors of 4 when possible
 * - Efficiently utilizes coalesced memory access
 */

struct Params {
  size: u32,
  batch_size: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Vector processing for 4 elements at once
  let vector_idx = idx * 4u;
  
  if (vector_idx + 3u < params.size) {
    // Full vector load is possible
    let v0 = input[vector_idx];
    let v1 = input[vector_idx + 1u];
    let v2 = input[vector_idx + 2u];
    let v3 = input[vector_idx + 3u];
    
    // Apply ReLU to each element
    output[vector_idx] = max(0.0, v0);
    output[vector_idx + 1u] = max(0.0, v1);
    output[vector_idx + 2u] = max(0.0, v2);
    output[vector_idx + 3u] = max(0.0, v3);
  } else {
    // Handle boundary with individual loads
    for (var i = 0u; i < 4u; i++) {
      let element_idx = vector_idx + i;
      if (element_idx < params.size) {
        output[element_idx] = max(0.0, input[element_idx]);
      }
    }
  }
}`;

const FIREFOX_ELEMENTWISE_ADD_SHADER = `/**
 * Firefox-optimized elementwise addition shader
 * 
 * Optimized for Firefox's WebGPU implementation with:
 * - Smaller workgroup size (128) for better thread occupancy
 * - Simple, non-vectorized operations that perform well in Firefox
 * - Avoids complex control flow that can be problematic in Firefox's shader compiler
 */

struct Params {
  size: u32,
  batch_size: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputA: array<f32>;
@group(0) @binding(2) var<storage, read> inputB: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Guard against out-of-bounds
  if (idx >= params.size) {
    return;
  }
  
  // Simple element-by-element processing for Firefox
  output[idx] = inputA[idx] + inputB[idx];
}`;

const EDGE_ELEMENTWISE_ADD_SHADER = `/**
 * Edge-optimized elementwise addition shader
 * 
 * Optimized for Edge's WebGPU implementation with:
 * - Uses 256 threads per workgroup
 * - Processes multiple elements per thread when possible
 * - Uses partial loop unrolling in pairs for better performance on Edge
 */

struct Params {
  size: u32,
  batch_size: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputA: array<f32>;
@group(0) @binding(2) var<storage, read> inputB: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global_idx = global_id.x;
  
  // Process elements in pairs (partial unrolling works well in Edge)
  for (var i = 0u; i < 4u; i = i + 2u) {
    let idx1 = global_idx + i * 256u;
    let idx2 = global_idx + (i + 1u) * 256u;
    
    // Edge benefits from explicit bounds checking
    if (idx1 < params.size) {
      output[idx1] = inputA[idx1] + inputB[idx1];
    }
    
    if (idx2 < params.size) {
      output[idx2] = inputA[idx2] + inputB[idx2];
    }
  }
}`;

const EDGE_ELEMENTWISE_TANH_SHADER = `/**
 * Edge-optimized elementwise tanh shader
 * 
 * Optimized for Edge's WebGPU implementation with:
 * - Uses 256 threads per workgroup
 * - Processes multiple elements per thread when possible
 * - Uses partial loop unrolling in pairs for better performance on Edge
 * - Explicit bounds checking that benefits Edge's shader compiler
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global_idx = global_id.x;
  
  // Process elements in pairs (partial unrolling works well in Edge)
  for (var i = 0u; i < 4u; i = i + 2u) {
    let idx1 = global_idx + i * 256u;
    let idx2 = global_idx + (i + 1u) * 256u;
    
    // Edge benefits from explicit bounds checking and switch statements
    if (idx1 < params.size) {
      let value = input[idx1];
      
      // Edge's shader compiler works better with switch than if/else
      switch(params.use_fast_math) {
        case 0u: {
          output[idx1] = tanh_approx(value);
          break;
        }
        case 1u: {
          output[idx1] = fast_tanh(value);
          break;
        }
        default: {
          output[idx1] = tanh_approx(value);
          break;
        }
      }
    }
    
    if (idx2 < params.size) {
      let value = input[idx2];
      
      switch(params.use_fast_math) {
        case 0u: {
          output[idx2] = tanh_approx(value);
          break;
        }
        case 1u: {
          output[idx2] = fast_tanh(value);
          break;
        }
        default: {
          output[idx2] = tanh_approx(value);
          break;
        }
      }
    }
  }
}`;

const EDGE_ELEMENTWISE_SIGMOID_SHADER = `/**
 * Edge-optimized elementwise sigmoid shader
 * 
 * Optimized for Edge's WebGPU implementation with:
 * - Uses 256 threads per workgroup
 * - Processes multiple elements per thread when possible
 * - Uses partial loop unrolling in pairs for better performance on Edge
 * - Explicit bounds checking that benefits Edge's shader compiler
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global_idx = global_id.x;
  
  // Process elements in pairs (partial unrolling works well in Edge)
  for (var i = 0u; i < 4u; i = i + 2u) {
    let idx1 = global_idx + i * 256u;
    let idx2 = global_idx + (i + 1u) * 256u;
    
    // Edge benefits from explicit bounds checking and switch statements
    if (idx1 < params.size) {
      let value = input[idx1];
      
      // Edge's shader compiler works better with switch than if/else
      switch(params.use_fast_math) {
        case 0u: {
          output[idx1] = sigmoid(value);
          break;
        }
        case 1u: {
          output[idx1] = fast_sigmoid(value);
          break;
        }
        default: {
          output[idx1] = sigmoid(value);
          break;
        }
      }
    }
    
    if (idx2 < params.size) {
      let value = input[idx2];
      
      switch(params.use_fast_math) {
        case 0u: {
          output[idx2] = sigmoid(value);
          break;
        }
        case 1u: {
          output[idx2] = fast_sigmoid(value);
          break;
        }
        default: {
          output[idx2] = sigmoid(value);
          break;
        }
      }
    }
  }
}`;

const EDGE_ELEMENTWISE_GELU_SHADER = `/**
 * Edge-optimized elementwise GELU shader
 * 
 * Optimized for Edge's WebGPU implementation with:
 * - Uses 256 threads per workgroup
 * - Processes multiple elements per thread when possible
 * - Uses partial loop unrolling in pairs for better performance on Edge
 * - Explicit bounds checking that benefits Edge's shader compiler
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global_idx = global_id.x;
  
  // Process elements in pairs (partial unrolling works well in Edge)
  for (var i = 0u; i < 4u; i = i + 2u) {
    let idx1 = global_idx + i * 256u;
    let idx2 = global_idx + (i + 1u) * 256u;
    
    // Edge benefits from explicit bounds checking and switch statements
    if (idx1 < params.size) {
      let value = input[idx1];
      
      // Edge's shader compiler works better with switch than if/else
      switch(params.use_fast_math) {
        case 0u: {
          output[idx1] = gelu(value);
          break;
        }
        case 1u: {
          output[idx1] = fast_gelu(value);
          break;
        }
        default: {
          output[idx1] = gelu(value);
          break;
        }
      }
    }
    
    if (idx2 < params.size) {
      let value = input[idx2];
      
      switch(params.use_fast_math) {
        case 0u: {
          output[idx2] = gelu(value);
          break;
        }
        case 1u: {
          output[idx2] = fast_gelu(value);
          break;
        }
        default: {
          output[idx2] = gelu(value);
          break;
        }
      }
    }
  }
}`;

const EDGE_ELEMENTWISE_SILU_SHADER = `/**
 * Edge-optimized elementwise SiLU shader
 * 
 * Optimized for Edge's WebGPU implementation with:
 * - Uses 256 threads per workgroup
 * - Processes multiple elements per thread when possible
 * - Uses partial loop unrolling in pairs for better performance on Edge
 * - Explicit bounds checking that benefits Edge's shader compiler
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global_idx = global_id.x;
  
  // Process elements in pairs (partial unrolling works well in Edge)
  for (var i = 0u; i < 4u; i = i + 2u) {
    let idx1 = global_idx + i * 256u;
    let idx2 = global_idx + (i + 1u) * 256u;
    
    // Edge benefits from explicit bounds checking and switch statements
    if (idx1 < params.size) {
      let value = input[idx1];
      
      // Edge's shader compiler works better with switch than if/else
      switch(params.use_fast_math) {
        case 0u: {
          output[idx1] = silu(value);
          break;
        }
        case 1u: {
          output[idx1] = fast_silu(value);
          break;
        }
        default: {
          output[idx1] = silu(value);
          break;
        }
      }
    }
    
    if (idx2 < params.size) {
      let value = input[idx2];
      
      switch(params.use_fast_math) {
        case 0u: {
          output[idx2] = silu(value);
          break;
        }
        case 1u: {
          output[idx2] = fast_silu(value);
          break;
        }
        default: {
          output[idx2] = silu(value);
          break;
        }
      }
    }
  }
}`;

const EDGE_ELEMENTWISE_LEAKY_RELU_SHADER = `/**
 * Edge-optimized elementwise Leaky ReLU shader
 * 
 * Optimized for Edge's WebGPU implementation with:
 * - Uses 256 threads per workgroup
 * - Processes multiple elements per thread when possible
 * - Uses partial loop unrolling in pairs for better performance on Edge
 * - Explicit bounds checking that benefits Edge's shader compiler
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  alpha: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global_idx = global_id.x;
  
  // Process elements in pairs (partial unrolling works well in Edge)
  for (var i = 0u; i < 4u; i = i + 2u) {
    let idx1 = global_idx + i * 256u;
    let idx2 = global_idx + (i + 1u) * 256u;
    
    // Edge benefits from explicit bounds checking
    if (idx1 < params.size) {
      let value = input[idx1];
      output[idx1] = leaky_relu(value, params.alpha);
    }
    
    if (idx2 < params.size) {
      let value = input[idx2];
      output[idx2] = leaky_relu(value, params.alpha);
    }
  }
}`;

const EDGE_LAYER_NORMALIZATION_SHADER = `/**
 * Edge-optimized Layer Normalization shader
 * 
 * Optimized for Edge's WebGPU implementation with:
 * - Uses 256 threads per workgroup
 * - Processes multiple elements per thread for better utilization
 * - Uses partial loop unrolling in pairs for better performance on Edge
 * - Explicit bounds checking that benefits Edge's shader compiler
 * - Two-pass algorithm with carefully structured control flow
 */

struct Params {
  hidden_size: u32,      // Size of the hidden dimension to normalize across
  batch_seq_len: u32,    // Batch size * sequence length
  epsilon: f32,          // Small value to avoid division by zero
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>; // Scale parameter
@group(0) @binding(3) var<storage, read> beta: array<f32>;  // Shift parameter
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Shared workgroup memory for reduction operations
var<workgroup> reduction_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  // For Edge, we use switch statements and explicit bounds checking for better performance
  let hidden_size = params.hidden_size;
  let batch_seq_len = params.batch_seq_len;
  let epsilon = params.epsilon;
  
  // Calculate which sequence this workgroup is processing
  let seq_idx = global_id.y;
  
  // Guard against out-of-bounds
  if (seq_idx >= batch_seq_len) {
    return;
  }
  
  // Base index for this sequence element
  let base_idx = seq_idx * hidden_size;
  
  // Thread index within workgroup
  let thread_idx = local_id.x;
  
  // ---------------------------
  // Step 1: Calculate the mean with explicit loop unrolling
  // ---------------------------
  var thread_sum: f32 = 0.0;
  
  // Each thread processes multiple elements with partial loop unrolling
  // This pattern works well for Edge's shader compiler
  let elements_per_thread = (hidden_size + 255u) / 256u;
  
  // Process elements in pairs (partial unrolling optimized for Edge)
  for (var i = 0u; i < elements_per_thread; i += 2u) {
    let hidden_offset1 = thread_idx + i * 256u;
    let hidden_offset2 = thread_idx + (i + 1u) * 256u;
    
    // First element
    if (hidden_offset1 < hidden_size) {
      thread_sum += input[base_idx + hidden_offset1];
    }
    
    // Second element (with bounds check)
    if (hidden_offset2 < hidden_size) {
      thread_sum += input[base_idx + hidden_offset2];
    }
  }
  
  // Store partial sum in shared memory
  reduction_data[thread_idx] = thread_sum;
  
  // Synchronize threads
  workgroupBarrier();
  
  // Parallel reduction optimized for Edge's shader compiler
  // Using powers of 2 with explicit checks for better performance
  if (thread_idx < 128u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 128u];
  }
  workgroupBarrier();
  
  if (thread_idx < 64u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 64u];
  }
  workgroupBarrier();
  
  if (thread_idx < 32u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 32u];
  }
  workgroupBarrier();
  
  if (thread_idx < 16u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 16u];
  }
  workgroupBarrier();
  
  if (thread_idx < 8u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 8u];
  }
  workgroupBarrier();
  
  if (thread_idx < 4u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 4u];
  }
  workgroupBarrier();
  
  if (thread_idx < 2u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 2u];
  }
  workgroupBarrier();
  
  if (thread_idx < 1u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 1u];
  }
  workgroupBarrier();
  
  // Mean is now in reduction_data[0]
  let mean = reduction_data[0] / f32(hidden_size);
  
  // --------------------------------
  // Step 2: Calculate the variance with explicit loop unrolling
  // --------------------------------
  var thread_variance: f32 = 0.0;
  
  // Process elements in pairs (optimization for Edge)
  for (var i = 0u; i < elements_per_thread; i += 2u) {
    let hidden_offset1 = thread_idx + i * 256u;
    let hidden_offset2 = thread_idx + (i + 1u) * 256u;
    
    // First element
    if (hidden_offset1 < hidden_size) {
      let diff = input[base_idx + hidden_offset1] - mean;
      thread_variance += diff * diff;
    }
    
    // Second element (with bounds check)
    if (hidden_offset2 < hidden_size) {
      let diff = input[base_idx + hidden_offset2] - mean;
      thread_variance += diff * diff;
    }
  }
  
  // Store partial variance in shared memory
  reduction_data[thread_idx] = thread_variance;
  
  // Synchronize threads
  workgroupBarrier();
  
  // Parallel reduction for variance with the same pattern as for mean
  if (thread_idx < 128u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 128u];
  }
  workgroupBarrier();
  
  if (thread_idx < 64u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 64u];
  }
  workgroupBarrier();
  
  if (thread_idx < 32u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 32u];
  }
  workgroupBarrier();
  
  if (thread_idx < 16u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 16u];
  }
  workgroupBarrier();
  
  if (thread_idx < 8u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 8u];
  }
  workgroupBarrier();
  
  if (thread_idx < 4u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 4u];
  }
  workgroupBarrier();
  
  if (thread_idx < 2u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 2u];
  }
  workgroupBarrier();
  
  if (thread_idx < 1u) {
    reduction_data[thread_idx] += reduction_data[thread_idx + 1u];
  }
  workgroupBarrier();
  
  // Compute inverse standard deviation
  let variance = reduction_data[0] / f32(hidden_size);
  let inv_std = 1.0 / sqrt(variance + epsilon);
  
  // --------------------------------
  // Step 3: Normalize and scale with explicit loop unrolling
  // --------------------------------
  // Process elements in pairs (optimization for Edge)
  for (var i = 0u; i < elements_per_thread; i += 2u) {
    let hidden_offset1 = thread_idx + i * 256u;
    let hidden_offset2 = thread_idx + (i + 1u) * 256u;
    
    // First element
    if (hidden_offset1 < hidden_size) {
      let idx = base_idx + hidden_offset1;
      let normalized = (input[idx] - mean) * inv_std;
      output[idx] = normalized * gamma[hidden_offset1] + beta[hidden_offset1];
    }
    
    // Second element (with bounds check)
    if (hidden_offset2 < hidden_size) {
      let idx = base_idx + hidden_offset2;
      let normalized = (input[idx] - mean) * inv_std;
      output[idx] = normalized * gamma[hidden_offset2] + beta[hidden_offset2];
    }
  }
}`;

const CHROME_ELEMENTWISE_ADD_SHADER = `/**
 * Chrome-optimized elementwise addition shader
 * 
 * Optimized for Chrome's WebGPU implementation with:
 * - Uses 256 threads per workgroup for balanced performance
 * - Processes elements in vectors of 4 when possible
 * - Efficiently utilizes coalesced memory access
 */

struct Params {
  size: u32,
  batch_size: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputA: array<f32>;
@group(0) @binding(2) var<storage, read> inputB: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Vector processing for 4 elements at once
  let vector_idx = idx * 4u;
  
  if (vector_idx + 3u < params.size) {
    // Full vector processing is possible
    // Load 4 elements from input A
    let a0 = inputA[vector_idx];
    let a1 = inputA[vector_idx + 1u];
    let a2 = inputA[vector_idx + 2u];
    let a3 = inputA[vector_idx + 3u];
    
    // Load 4 elements from input B
    let b0 = inputB[vector_idx];
    let b1 = inputB[vector_idx + 1u];
    let b2 = inputB[vector_idx + 2u];
    let b3 = inputB[vector_idx + 3u];
    
    // Add and store results
    output[vector_idx] = a0 + b0;
    output[vector_idx + 1u] = a1 + b1;
    output[vector_idx + 2u] = a2 + b2;
    output[vector_idx + 3u] = a3 + b3;
  } else {
    // Handle boundary with individual loads
    for (var i = 0u; i < 4u; i++) {
      let element_idx = vector_idx + i;
      if (element_idx < params.size) {
        output[element_idx] = inputA[element_idx] + inputB[element_idx];
      }
    }
  }
}`;

const CHROME_ELEMENTWISE_TANH_SHADER = `/**
 * Chrome-optimized elementwise tanh shader
 * 
 * Optimized for Chrome's WebGPU implementation with:
 * - Uses 256 threads per workgroup for balanced performance
 * - Processes elements in vectors of 4 when possible
 * - Efficiently utilizes coalesced memory access
 * - Uses vectorized operations for better performance
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Vector processing for 4 elements at once
  let vector_idx = idx * 4u;
  
  if (vector_idx + 3u < params.size) {
    // Full vector load is possible
    let v0 = input[vector_idx];
    let v1 = input[vector_idx + 1u];
    let v2 = input[vector_idx + 2u];
    let v3 = input[vector_idx + 3u];
    
    // Apply tanh to each element
    if (params.use_fast_math == 1u) {
      output[vector_idx] = fast_tanh(v0);
      output[vector_idx + 1u] = fast_tanh(v1);
      output[vector_idx + 2u] = fast_tanh(v2);
      output[vector_idx + 3u] = fast_tanh(v3);
    } else {
      output[vector_idx] = tanh_approx(v0);
      output[vector_idx + 1u] = tanh_approx(v1);
      output[vector_idx + 2u] = tanh_approx(v2);
      output[vector_idx + 3u] = tanh_approx(v3);
    }
  } else {
    // Handle boundary with individual loads
    for (var i = 0u; i < 4u; i++) {
      let element_idx = vector_idx + i;
      if (element_idx < params.size) {
        let value = input[element_idx];
        if (params.use_fast_math == 1u) {
          output[element_idx] = fast_tanh(value);
        } else {
          output[element_idx] = tanh_approx(value);
        }
      }
    }
  }
}`;

const CHROME_ELEMENTWISE_SIGMOID_SHADER = `/**
 * Chrome-optimized elementwise sigmoid shader
 * 
 * Optimized for Chrome's WebGPU implementation with:
 * - Uses 256 threads per workgroup for balanced performance
 * - Processes elements in vectors of 4 when possible
 * - Efficiently utilizes coalesced memory access
 * - Uses vectorized operations for better performance
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Vector processing for 4 elements at once
  let vector_idx = idx * 4u;
  
  if (vector_idx + 3u < params.size) {
    // Full vector load is possible
    let v0 = input[vector_idx];
    let v1 = input[vector_idx + 1u];
    let v2 = input[vector_idx + 2u];
    let v3 = input[vector_idx + 3u];
    
    // Apply sigmoid to each element
    if (params.use_fast_math == 1u) {
      output[vector_idx] = fast_sigmoid(v0);
      output[vector_idx + 1u] = fast_sigmoid(v1);
      output[vector_idx + 2u] = fast_sigmoid(v2);
      output[vector_idx + 3u] = fast_sigmoid(v3);
    } else {
      output[vector_idx] = sigmoid(v0);
      output[vector_idx + 1u] = sigmoid(v1);
      output[vector_idx + 2u] = sigmoid(v2);
      output[vector_idx + 3u] = sigmoid(v3);
    }
  } else {
    // Handle boundary with individual loads
    for (var i = 0u; i < 4u; i++) {
      let element_idx = vector_idx + i;
      if (element_idx < params.size) {
        let value = input[element_idx];
        if (params.use_fast_math == 1u) {
          output[element_idx] = fast_sigmoid(value);
        } else {
          output[element_idx] = sigmoid(value);
        }
      }
    }
  }
}`;

const CHROME_ELEMENTWISE_GELU_SHADER = `/**
 * Chrome-optimized elementwise GELU shader
 * 
 * Optimized for Chrome's WebGPU implementation with:
 * - Uses 256 threads per workgroup for balanced performance
 * - Processes elements in vectors of 4 when possible
 * - Efficiently utilizes coalesced memory access
 * - Uses vectorized operations for better performance
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Vector processing for 4 elements at once
  let vector_idx = idx * 4u;
  
  if (vector_idx + 3u < params.size) {
    // Full vector load is possible
    let v0 = input[vector_idx];
    let v1 = input[vector_idx + 1u];
    let v2 = input[vector_idx + 2u];
    let v3 = input[vector_idx + 3u];
    
    // Apply GELU to each element
    if (params.use_fast_math == 1u) {
      output[vector_idx] = fast_gelu(v0);
      output[vector_idx + 1u] = fast_gelu(v1);
      output[vector_idx + 2u] = fast_gelu(v2);
      output[vector_idx + 3u] = fast_gelu(v3);
    } else {
      output[vector_idx] = gelu(v0);
      output[vector_idx + 1u] = gelu(v1);
      output[vector_idx + 2u] = gelu(v2);
      output[vector_idx + 3u] = gelu(v3);
    }
  } else {
    // Handle boundary with individual loads
    for (var i = 0u; i < 4u; i++) {
      let element_idx = vector_idx + i;
      if (element_idx < params.size) {
        let value = input[element_idx];
        if (params.use_fast_math == 1u) {
          output[element_idx] = fast_gelu(value);
        } else {
          output[element_idx] = gelu(value);
        }
      }
    }
  }
}`;

const CHROME_ELEMENTWISE_SILU_SHADER = `/**
 * Chrome-optimized elementwise SiLU shader
 * 
 * Optimized for Chrome's WebGPU implementation with:
 * - Uses 256 threads per workgroup for balanced performance
 * - Processes elements in vectors of 4 when possible
 * - Efficiently utilizes coalesced memory access
 * - Uses vectorized operations for better performance
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Vector processing for 4 elements at once
  let vector_idx = idx * 4u;
  
  if (vector_idx + 3u < params.size) {
    // Full vector load is possible
    let v0 = input[vector_idx];
    let v1 = input[vector_idx + 1u];
    let v2 = input[vector_idx + 2u];
    let v3 = input[vector_idx + 3u];
    
    // Apply SiLU to each element
    if (params.use_fast_math == 1u) {
      output[vector_idx] = fast_silu(v0);
      output[vector_idx + 1u] = fast_silu(v1);
      output[vector_idx + 2u] = fast_silu(v2);
      output[vector_idx + 3u] = fast_silu(v3);
    } else {
      output[vector_idx] = silu(v0);
      output[vector_idx + 1u] = silu(v1);
      output[vector_idx + 2u] = silu(v2);
      output[vector_idx + 3u] = silu(v3);
    }
  } else {
    // Handle boundary with individual loads
    for (var i = 0u; i < 4u; i++) {
      let element_idx = vector_idx + i;
      if (element_idx < params.size) {
        let value = input[element_idx];
        if (params.use_fast_math == 1u) {
          output[element_idx] = fast_silu(value);
        } else {
          output[element_idx] = silu(value);
        }
      }
    }
  }
}`;

const CHROME_ELEMENTWISE_LEAKY_RELU_SHADER = `/**
 * Chrome-optimized elementwise Leaky ReLU shader
 * 
 * Optimized for Chrome's WebGPU implementation with:
 * - Uses 256 threads per workgroup for balanced performance
 * - Processes elements in vectors of 4 when possible
 * - Efficiently utilizes coalesced memory access
 * - Uses vectorized operations for better performance
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  alpha: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Vector processing for 4 elements at once
  let vector_idx = idx * 4u;
  
  if (vector_idx + 3u < params.size) {
    // Full vector load is possible
    let v0 = input[vector_idx];
    let v1 = input[vector_idx + 1u];
    let v2 = input[vector_idx + 2u];
    let v3 = input[vector_idx + 3u];
    
    // Apply Leaky ReLU to each element
    output[vector_idx] = leaky_relu(v0, params.alpha);
    output[vector_idx + 1u] = leaky_relu(v1, params.alpha);
    output[vector_idx + 2u] = leaky_relu(v2, params.alpha);
    output[vector_idx + 3u] = leaky_relu(v3, params.alpha);
  } else {
    // Handle boundary with individual loads
    for (var i = 0u; i < 4u; i++) {
      let element_idx = vector_idx + i;
      if (element_idx < params.size) {
        let value = input[element_idx];
        output[element_idx] = leaky_relu(value, params.alpha);
      }
    }
  }
}`;

const CHROME_LAYER_NORMALIZATION_SHADER = `/**
 * Chrome-optimized Layer Normalization shader
 * 
 * Optimized for Chrome's WebGPU implementation with:
 * - Uses 256 threads per workgroup for balanced performance
 * - Utilizes coalesced memory access patterns
 * - Efficiently uses shared memory for reductions
 * - Implements 4-element vectorization for better throughput
 * - Two-pass algorithm with vectorized operations
 */

struct Params {
  hidden_size: u32,      // Size of the hidden dimension to normalize across
  batch_seq_len: u32,    // Batch size * sequence length
  epsilon: f32,          // Small value to avoid division by zero
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>; // Scale parameter
@group(0) @binding(3) var<storage, read> beta: array<f32>;  // Shift parameter
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Shared workgroup memory for reduction operations (256 threads per workgroup)
var<workgroup> reduction_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  let hidden_size = params.hidden_size;
  let batch_seq_len = params.batch_seq_len;
  let epsilon = params.epsilon;
  
  // Calculate which sequence element this workgroup is processing
  let seq_idx = global_id.y;
  
  // Guard against out-of-bounds
  if (seq_idx >= batch_seq_len) {
    return;
  }
  
  // Base index for this sequence element
  let base_idx = seq_idx * hidden_size;
  
  // Each thread processes multiple elements of the hidden dimension
  let thread_idx = local_id.x;
  let elements_per_thread = (hidden_size + 255u) / 256u;
  
  // ---------------------------
  // Step 1: Calculate the mean using vectorization when possible
  // ---------------------------
  var thread_sum: f32 = 0.0;
  
  for (var i = 0u; i < elements_per_thread; i++) {
    let hidden_offset = thread_idx + i * 256u;
    
    // Use 4-element vectorization when possible
    if (hidden_offset % 4u == 0u && hidden_offset + 3u < hidden_size) {
      // Vector load for 4 consecutive elements
      let idx = base_idx + hidden_offset;
      let v0 = input[idx];
      let v1 = input[idx + 1u];
      let v2 = input[idx + 2u];
      let v3 = input[idx + 3u];
      
      thread_sum += v0 + v1 + v2 + v3;
    } else if (hidden_offset < hidden_size) {
      // Individual element
      let idx = base_idx + hidden_offset;
      thread_sum += input[idx];
    }
  }
  
  // Store partial sum in shared memory
  reduction_data[thread_idx] = thread_sum;
  
  // Synchronize threads in the workgroup
  workgroupBarrier();
  
  // Parallel reduction to compute the sum - optimized for Chrome
  // Each step reduces the active threads by half
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (thread_idx < stride) {
      reduction_data[thread_idx] += reduction_data[thread_idx + stride];
    }
    workgroupBarrier();
  }
  
  // Mean is now in reduction_data[0]
  let mean = reduction_data[0] / f32(hidden_size);
  
  // --------------------------------
  // Step 2: Calculate the variance using vectorization
  // --------------------------------
  var thread_variance: f32 = 0.0;
  
  for (var i = 0u; i < elements_per_thread; i++) {
    let hidden_offset = thread_idx + i * 256u;
    
    // Use 4-element vectorization when possible
    if (hidden_offset % 4u == 0u && hidden_offset + 3u < hidden_size) {
      // Vector load for 4 consecutive elements
      let idx = base_idx + hidden_offset;
      let v0 = input[idx] - mean;
      let v1 = input[idx + 1u] - mean;
      let v2 = input[idx + 2u] - mean;
      let v3 = input[idx + 3u] - mean;
      
      thread_variance += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    } else if (hidden_offset < hidden_size) {
      // Individual element
      let idx = base_idx + hidden_offset;
      let diff = input[idx] - mean;
      thread_variance += diff * diff;
    }
  }
  
  // Store partial variance in shared memory
  reduction_data[thread_idx] = thread_variance;
  
  // Synchronize threads
  workgroupBarrier();
  
  // Parallel reduction for variance
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (thread_idx < stride) {
      reduction_data[thread_idx] += reduction_data[thread_idx + stride];
    }
    workgroupBarrier();
  }
  
  // Compute standard deviation with epsilon for numerical stability
  let variance = reduction_data[0] / f32(hidden_size);
  let inv_std = 1.0 / sqrt(variance + epsilon);
  
  // --------------------------------
  // Step 3: Normalize and scale with vectorization
  // --------------------------------
  for (var i = 0u; i < elements_per_thread; i++) {
    let hidden_offset = thread_idx + i * 256u;
    
    // Use 4-element vectorization when possible
    if (hidden_offset % 4u == 0u && hidden_offset + 3u < hidden_size) {
      // Process 4 elements at once
      let idx = base_idx + hidden_offset;
      
      // Load input values
      let in0 = input[idx];
      let in1 = input[idx + 1u];
      let in2 = input[idx + 2u];
      let in3 = input[idx + 3u];
      
      // Load scale and shift parameters
      let g0 = gamma[hidden_offset];
      let g1 = gamma[hidden_offset + 1u];
      let g2 = gamma[hidden_offset + 2u];
      let g3 = gamma[hidden_offset + 3u];
      
      let b0 = beta[hidden_offset];
      let b1 = beta[hidden_offset + 1u];
      let b2 = beta[hidden_offset + 2u];
      let b3 = beta[hidden_offset + 3u];
      
      // Normalize and scale
      output[idx] = ((in0 - mean) * inv_std) * g0 + b0;
      output[idx + 1u] = ((in1 - mean) * inv_std) * g1 + b1;
      output[idx + 2u] = ((in2 - mean) * inv_std) * g2 + b2;
      output[idx + 3u] = ((in3 - mean) * inv_std) * g3 + b3;
    } else if (hidden_offset < hidden_size) {
      // Individual element processing
      let idx = base_idx + hidden_offset;
      output[idx] = ((input[idx] - mean) * inv_std) * gamma[hidden_offset] + beta[hidden_offset];
    }
  }
}`;

const SAFARI_ELEMENTWISE_ADD_SHADER = `/**
 * Safari-optimized elementwise addition shader
 * 
 * Optimized for Safari's Metal-based WebGPU implementation with:
 * - Uses vector types for efficient processing
 * - Uses SIMDGroup operations when available
 * - Optimized for Apple GPU architecture
 */

struct Params {
  size: u32,
  batch_size: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputA: array<f32>;
@group(0) @binding(2) var<storage, read> inputB: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Process elements in vectors of 4 when possible
  let vector_idx = idx * 4u;
  
  if (vector_idx + 3u < params.size) {
    // Full vector processing is possible
    let a_vec = vec4<f32>(
      inputA[vector_idx],
      inputA[vector_idx + 1u],
      inputA[vector_idx + 2u],
      inputA[vector_idx + 3u]
    );
    
    let b_vec = vec4<f32>(
      inputB[vector_idx],
      inputB[vector_idx + 1u],
      inputB[vector_idx + 2u],
      inputB[vector_idx + 3u]
    );
    
    // Add vectors and store results
    let result_vec = a_vec + b_vec;
    
    output[vector_idx] = result_vec.x;
    output[vector_idx + 1u] = result_vec.y;
    output[vector_idx + 2u] = result_vec.z;
    output[vector_idx + 3u] = result_vec.w;
  } else {
    // Handle boundary with individual loads
    for (var i = 0u; i < 4u; i++) {
      let element_idx = vector_idx + i;
      if (element_idx < params.size) {
        output[element_idx] = inputA[element_idx] + inputB[element_idx];
      }
    }
  }
}`;

const SAFARI_ELEMENTWISE_TANH_SHADER = `/**
 * Safari-optimized elementwise tanh shader
 * 
 * Optimized for Safari's Metal-based WebGPU implementation with:
 * - Uses vector types for efficient processing
 * - Uses SIMDGroup operations when available
 * - Optimized for Apple GPU architecture
 * - Leverages larger workgroups (512) for Apple Silicon GPUs
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Process elements in vectors of 4 when possible
  let vector_idx = idx * 4u;
  
  if (vector_idx + 3u < params.size) {
    // Full vector load is possible
    let input_vec = vec4<f32>(
      input[vector_idx],
      input[vector_idx + 1u],
      input[vector_idx + 2u],
      input[vector_idx + 3u]
    );
    
    // Apply tanh using vector operations if possible
    var result_vec: vec4<f32>;
    
    if (params.use_fast_math == 1u) {
      // Apply fast tanh to each component
      let x2_vec = input_vec * input_vec;
      result_vec = input_vec * (vec4<f32>(27.0) + x2_vec) / (vec4<f32>(27.0) + vec4<f32>(9.0) * x2_vec);
    } else {
      // Apply standard tanh to each component
      // First compute sigmoid(2x) for each component
      let two_x = 2.0 * input_vec;
      let clamped = clamp(two_x, vec4<f32>(-30.0), vec4<f32>(30.0));
      let sigmoid_vec = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-clamped));
      
      // tanh(x) = 2 * sigmoid(2x) - 1
      result_vec = 2.0 * sigmoid_vec - vec4<f32>(1.0);
    }
    
    // Store results
    output[vector_idx] = result_vec.x;
    output[vector_idx + 1u] = result_vec.y;
    output[vector_idx + 2u] = result_vec.z;
    output[vector_idx + 3u] = result_vec.w;
  } else {
    // Handle boundary with individual loads
    for (var i = 0u; i < 4u; i++) {
      let element_idx = vector_idx + i;
      if (element_idx < params.size) {
        let value = input[element_idx];
        if (params.use_fast_math == 1u) {
          output[element_idx] = fast_tanh(value);
        } else {
          output[element_idx] = tanh_approx(value);
        }
      }
    }
  }
}`;

const SAFARI_ELEMENTWISE_SIGMOID_SHADER = `/**
 * Safari-optimized elementwise sigmoid shader
 * 
 * Optimized for Safari's Metal-based WebGPU implementation with:
 * - Uses vector types for efficient processing
 * - Uses SIMDGroup operations when available
 * - Optimized for Apple GPU architecture
 * - Leverages larger workgroups (512) for Apple Silicon GPUs
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Process elements in vectors of 4 when possible
  let vector_idx = idx * 4u;
  
  if (vector_idx + 3u < params.size) {
    // Full vector load is possible
    let input_vec = vec4<f32>(
      input[vector_idx],
      input[vector_idx + 1u],
      input[vector_idx + 2u],
      input[vector_idx + 3u]
    );
    
    // Apply sigmoid using vector operations
    var result_vec: vec4<f32>;
    
    if (params.use_fast_math == 1u) {
      // Fast sigmoid approximation: x / (1 + abs(x)) * 0.5 + 0.5
      result_vec = input_vec / (vec4<f32>(1.0) + abs(input_vec)) * vec4<f32>(0.5) + vec4<f32>(0.5);
    } else {
      // Standard sigmoid with clamping for numerical stability
      let clamped = clamp(input_vec, vec4<f32>(-30.0), vec4<f32>(30.0));
      result_vec = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-clamped));
    }
    
    // Store results
    output[vector_idx] = result_vec.x;
    output[vector_idx + 1u] = result_vec.y;
    output[vector_idx + 2u] = result_vec.z;
    output[vector_idx + 3u] = result_vec.w;
  } else {
    // Handle boundary with individual loads
    for (var i = 0u; i < 4u; i++) {
      let element_idx = vector_idx + i;
      if (element_idx < params.size) {
        let value = input[element_idx];
        if (params.use_fast_math == 1u) {
          output[element_idx] = fast_sigmoid(value);
        } else {
          output[element_idx] = sigmoid(value);
        }
      }
    }
  }
}`;

const SAFARI_ELEMENTWISE_GELU_SHADER = `/**
 * Safari-optimized elementwise GELU shader
 * 
 * Optimized for Safari's Metal-based WebGPU implementation with:
 * - Uses vector types for efficient processing
 * - Uses SIMDGroup operations when available
 * - Optimized for Apple GPU architecture
 * - Leverages larger workgroups (512) for Apple Silicon GPUs
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Process elements in vectors of 4 when possible
  let vector_idx = idx * 4u;
  
  if (vector_idx + 3u < params.size) {
    // Full vector load is possible
    let input_vec = vec4<f32>(
      input[vector_idx],
      input[vector_idx + 1u],
      input[vector_idx + 2u],
      input[vector_idx + 3u]
    );
    
    // Apply GELU using vector operations
    var result_vec: vec4<f32>;
    
    if (params.use_fast_math == 1u) {
      // Fast GELU approximation: x * sigmoid(1.702 * x)
      let scaled_input = 1.702 * input_vec;
      let clamped = clamp(scaled_input, vec4<f32>(-30.0), vec4<f32>(30.0));
      let sigmoid_result = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-clamped));
      result_vec = input_vec * sigmoid_result;
    } else {
      // Standard GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
      let x3 = input_vec * input_vec * input_vec;
      let inner = vec4<f32>(SQRT_2_OVER_PI) * (input_vec + vec4<f32>(GELU_COEF1) * x3);
      
      // Compute tanh using the 2*sigmoid(2x)-1 formula with vector operations
      let two_inner = 2.0 * inner;
      let clamped = clamp(two_inner, vec4<f32>(-30.0), vec4<f32>(30.0));
      let sigmoid_result = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-clamped));
      let tanh_result = 2.0 * sigmoid_result - vec4<f32>(1.0);
      
      result_vec = input_vec * 0.5 * (vec4<f32>(1.0) + tanh_result);
    }
    
    // Store results
    output[vector_idx] = result_vec.x;
    output[vector_idx + 1u] = result_vec.y;
    output[vector_idx + 2u] = result_vec.z;
    output[vector_idx + 3u] = result_vec.w;
  } else {
    // Handle boundary with individual loads
    for (var i = 0u; i < 4u; i++) {
      let element_idx = vector_idx + i;
      if (element_idx < params.size) {
        let value = input[element_idx];
        if (params.use_fast_math == 1u) {
          output[element_idx] = fast_gelu(value);
        } else {
          output[element_idx] = gelu(value);
        }
      }
    }
  }
}`;

const SAFARI_ELEMENTWISE_SILU_SHADER = `/**
 * Safari-optimized elementwise SiLU shader
 * 
 * Optimized for Safari's Metal-based WebGPU implementation with:
 * - Uses vector types for efficient processing
 * - Uses SIMDGroup operations when available
 * - Optimized for Apple GPU architecture
 * - Leverages larger workgroups (512) for Apple Silicon GPUs
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Process elements in vectors of 4 when possible
  let vector_idx = idx * 4u;
  
  if (vector_idx + 3u < params.size) {
    // Full vector load is possible
    let input_vec = vec4<f32>(
      input[vector_idx],
      input[vector_idx + 1u],
      input[vector_idx + 2u],
      input[vector_idx + 3u]
    );
    
    // Apply SiLU using vector operations
    var result_vec: vec4<f32>;
    
    if (params.use_fast_math == 1u) {
      // Fast SiLU approximation: x * fast_sigmoid(x)
      let sigmoid_result = input_vec / (vec4<f32>(1.0) + abs(input_vec)) * vec4<f32>(0.5) + vec4<f32>(0.5);
      result_vec = input_vec * sigmoid_result;
    } else {
      // Standard SiLU: x * sigmoid(x)
      let clamped = clamp(input_vec, vec4<f32>(-30.0), vec4<f32>(30.0));
      let sigmoid_result = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-clamped));
      result_vec = input_vec * sigmoid_result;
    }
    
    // Store results
    output[vector_idx] = result_vec.x;
    output[vector_idx + 1u] = result_vec.y;
    output[vector_idx + 2u] = result_vec.z;
    output[vector_idx + 3u] = result_vec.w;
  } else {
    // Handle boundary with individual loads
    for (var i = 0u; i < 4u; i++) {
      let element_idx = vector_idx + i;
      if (element_idx < params.size) {
        let value = input[element_idx];
        if (params.use_fast_math == 1u) {
          output[element_idx] = fast_silu(value);
        } else {
          output[element_idx] = silu(value);
        }
      }
    }
  }
}`;

const SAFARI_ELEMENTWISE_LEAKY_RELU_SHADER = `/**
 * Safari-optimized elementwise Leaky ReLU shader
 * 
 * Optimized for Safari's Metal-based WebGPU implementation with:
 * - Uses vector types for efficient processing
 * - Uses SIMDGroup operations when available
 * - Optimized for Apple GPU architecture
 * - Leverages larger workgroups (512) for Apple Silicon GPUs
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  alpha: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Process elements in vectors of 4 when possible
  let vector_idx = idx * 4u;
  
  if (vector_idx + 3u < params.size) {
    // Full vector load is possible
    let input_vec = vec4<f32>(
      input[vector_idx],
      input[vector_idx + 1u],
      input[vector_idx + 2u],
      input[vector_idx + 3u]
    );
    
    // Apply Leaky ReLU using vector operations
    // Create a vector with all alpha values
    let alpha_vec = vec4<f32>(params.alpha);
    
    // Compute alpha * x for all elements
    let scaled_input = alpha_vec * input_vec;
    
    // Use max() with vectors for better performance on Apple GPUs
    let result_vec = max(scaled_input, input_vec);
    
    // Store results
    output[vector_idx] = result_vec.x;
    output[vector_idx + 1u] = result_vec.y;
    output[vector_idx + 2u] = result_vec.z;
    output[vector_idx + 3u] = result_vec.w;
  } else {
    // Handle boundary with individual loads
    for (var i = 0u; i < 4u; i++) {
      let element_idx = vector_idx + i;
      if (element_idx < params.size) {
        let value = input[element_idx];
        output[element_idx] = leaky_relu(value, params.alpha);
      }
    }
  }
}`;

const SAFARI_LAYER_NORMALIZATION_SHADER = `/**
 * Safari-optimized Layer Normalization shader
 * 
 * Optimized for Safari's Metal-based WebGPU implementation with:
 * - Uses vector types for efficient processing
 * - Optimized for Apple GPU architecture with vector operations
 * - Leverages larger workgroups (512) for Apple Silicon GPUs
 * - Specialized shared memory layout for Apple GPUs
 */

struct Params {
  hidden_size: u32,      // Size of the hidden dimension to normalize across
  batch_seq_len: u32,    // Batch size * sequence length
  epsilon: f32,          // Small value to avoid division by zero
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>; // Scale parameter
@group(0) @binding(3) var<storage, read> beta: array<f32>;  // Shift parameter
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Shared workgroup memory for reduction operations
var<workgroup> reduction_data: array<f32, 512>;

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  let hidden_size = params.hidden_size;
  let batch_seq_len = params.batch_seq_len;
  let epsilon = params.epsilon;
  
  // Calculate which sequence element this workgroup is processing
  let seq_idx = global_id.y;
  
  // Guard against out-of-bounds
  if (seq_idx >= batch_seq_len) {
    return;
  }
  
  // Base index for this sequence element
  let base_idx = seq_idx * hidden_size;
  
  // Thread index within workgroup
  let thread_idx = local_id.x;
  
  // Calculate how many elements each thread needs to process
  let elements_per_thread = (hidden_size + 511u) / 512u;
  
  // ---------------------------
  // Step 1: Calculate the mean with vector operations
  // ---------------------------
  var thread_sum: f32 = 0.0;
  
  for (var i = 0u; i < elements_per_thread; i++) {
    let hidden_offset = thread_idx + i * 512u;
    
    // Use vec4 for efficient processing - optimized for Apple GPUs
    if (hidden_offset + 3u < hidden_size && hidden_offset % 4u == 0u) {
      // Load 4 consecutive elements as a vector
      let idx = base_idx + hidden_offset;
      let input_vec = vec4<f32>(
        input[idx],
        input[idx + 1u],
        input[idx + 2u],
        input[idx + 3u]
      );
      
      // Sum all components of the vector
      thread_sum += input_vec.x + input_vec.y + input_vec.z + input_vec.w;
    } else if (hidden_offset < hidden_size) {
      // Individual processing for boundary elements
      thread_sum += input[base_idx + hidden_offset];
    }
  }
  
  // Store partial sum in shared memory
  reduction_data[thread_idx] = thread_sum;
  
  // Synchronize all threads in the workgroup
  workgroupBarrier();
  
  // Parallel reduction optimized for Safari's larger workgroup size
  // Use a binary tree reduction pattern
  for (var stride = 256u; stride > 0u; stride >>= 1u) {
    if (thread_idx < stride) {
      reduction_data[thread_idx] += reduction_data[thread_idx + stride];
    }
    workgroupBarrier();
  }
  
  // Mean is now in reduction_data[0]
  let mean = reduction_data[0] / f32(hidden_size);
  
  // --------------------------------
  // Step 2: Calculate the variance with vector operations
  // --------------------------------
  var thread_variance: f32 = 0.0;
  
  for (var i = 0u; i < elements_per_thread; i++) {
    let hidden_offset = thread_idx + i * 512u;
    
    // Use vec4 operations for better performance on Apple GPUs
    if (hidden_offset + 3u < hidden_size && hidden_offset % 4u == 0u) {
      // Load 4 consecutive elements
      let idx = base_idx + hidden_offset;
      let input_vec = vec4<f32>(
        input[idx],
        input[idx + 1u],
        input[idx + 2u],
        input[idx + 3u]
      );
      
      // Subtract mean from each element
      let diff_vec = input_vec - vec4<f32>(mean);
      
      // Square each difference and sum
      thread_variance += dot(diff_vec, diff_vec); // Efficient dot product
    } else if (hidden_offset < hidden_size) {
      // Individual processing for boundary
      let diff = input[base_idx + hidden_offset] - mean;
      thread_variance += diff * diff;
    }
  }
  
  // Store partial variance in shared memory
  reduction_data[thread_idx] = thread_variance;
  
  // Synchronize threads
  workgroupBarrier();
  
  // Parallel reduction for variance
  for (var stride = 256u; stride > 0u; stride >>= 1u) {
    if (thread_idx < stride) {
      reduction_data[thread_idx] += reduction_data[thread_idx + stride];
    }
    workgroupBarrier();
  }
  
  // Compute inverse standard deviation
  let variance = reduction_data[0] / f32(hidden_size);
  let inv_std = 1.0 / sqrt(variance + epsilon);
  
  // --------------------------------
  // Step 3: Normalize and scale with vector operations
  // --------------------------------
  for (var i = 0u; i < elements_per_thread; i++) {
    let hidden_offset = thread_idx + i * 512u;
    
    // Use vec4 operations for better performance on Apple GPUs
    if (hidden_offset + 3u < hidden_size && hidden_offset % 4u == 0u) {
      // Load 4 consecutive elements and parameters
      let idx = base_idx + hidden_offset;
      
      // Input values
      let input_vec = vec4<f32>(
        input[idx],
        input[idx + 1u],
        input[idx + 2u],
        input[idx + 3u]
      );
      
      // Scale factors
      let gamma_vec = vec4<f32>(
        gamma[hidden_offset],
        gamma[hidden_offset + 1u],
        gamma[hidden_offset + 2u],
        gamma[hidden_offset + 3u]
      );
      
      // Shift factors
      let beta_vec = vec4<f32>(
        beta[hidden_offset],
        beta[hidden_offset + 1u],
        beta[hidden_offset + 2u],
        beta[hidden_offset + 3u]
      );
      
      // Normalize, scale, and shift in vectorized form
      let mean_vec = vec4<f32>(mean);
      let inv_std_vec = vec4<f32>(inv_std);
      
      let normalized_vec = (input_vec - mean_vec) * inv_std_vec;
      let result_vec = normalized_vec * gamma_vec + beta_vec;
      
      // Store results
      output[idx] = result_vec.x;
      output[idx + 1u] = result_vec.y;
      output[idx + 2u] = result_vec.z;
      output[idx + 3u] = result_vec.w;
    } else if (hidden_offset < hidden_size) {
      // Individual processing for boundary
      let idx = base_idx + hidden_offset;
      let normalized = (input[idx] - mean) * inv_std;
      output[idx] = normalized * gamma[hidden_offset] + beta[hidden_offset];
    }
  }
}`;

const FIREFOX_ELEMENTWISE_TANH_SHADER = `/**
 * Firefox-optimized elementwise tanh shader
 * 
 * Optimized for Firefox's WebGPU implementation with:
 * - Smaller workgroup size (128) for better thread occupancy
 * - Simple, non-vectorized operations that perform well in Firefox
 * - Avoids complex control flow that can be problematic in Firefox's shader compiler
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Guard against out-of-bounds
  if (idx >= params.size) {
    return;
  }
  
  // Simple element-by-element processing for Firefox
  let value = input[idx];
  
  // Choose between fast and standard tanh based on configuration
  if (params.use_fast_math == 1u) {
    output[idx] = fast_tanh(value);
  } else {
    output[idx] = tanh_approx(value);
  }
}`;

const FIREFOX_ELEMENTWISE_SIGMOID_SHADER = `/**
 * Firefox-optimized elementwise sigmoid shader
 * 
 * Optimized for Firefox's WebGPU implementation with:
 * - Smaller workgroup size (128) for better thread occupancy
 * - Simple, non-vectorized operations that perform well in Firefox
 * - Avoids complex control flow that can be problematic in Firefox's shader compiler
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Guard against out-of-bounds
  if (idx >= params.size) {
    return;
  }
  
  // Simple element-by-element processing for Firefox
  let value = input[idx];
  
  // Choose between fast and standard sigmoid based on configuration
  if (params.use_fast_math == 1u) {
    output[idx] = fast_sigmoid(value);
  } else {
    output[idx] = sigmoid(value);
  }
}`;

const FIREFOX_ELEMENTWISE_GELU_SHADER = `/**
 * Firefox-optimized elementwise GELU shader
 * 
 * Optimized for Firefox's WebGPU implementation with:
 * - Smaller workgroup size (128) for better thread occupancy
 * - Simple, non-vectorized operations that perform well in Firefox
 * - Avoids complex control flow that can be problematic in Firefox's shader compiler
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Guard against out-of-bounds
  if (idx >= params.size) {
    return;
  }
  
  // Simple element-by-element processing for Firefox
  let value = input[idx];
  
  // Choose between fast and standard GELU based on configuration
  if (params.use_fast_math == 1u) {
    output[idx] = fast_gelu(value);
  } else {
    output[idx] = gelu(value);
  }
}`;

const FIREFOX_ELEMENTWISE_SILU_SHADER = `/**
 * Firefox-optimized elementwise SiLU shader
 * 
 * Optimized for Firefox's WebGPU implementation with:
 * - Smaller workgroup size (128) for better thread occupancy
 * - Simple, non-vectorized operations that perform well in Firefox
 * - Avoids complex control flow that can be problematic in Firefox's shader compiler
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  use_fast_math: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Guard against out-of-bounds
  if (idx >= params.size) {
    return;
  }
  
  // Simple element-by-element processing for Firefox
  let value = input[idx];
  
  // Choose between fast and standard SiLU based on configuration
  if (params.use_fast_math == 1u) {
    output[idx] = fast_silu(value);
  } else {
    output[idx] = silu(value);
  }
}`;

const FIREFOX_ELEMENTWISE_LEAKY_RELU_SHADER = `/**
 * Firefox-optimized elementwise Leaky ReLU shader
 * 
 * Optimized for Firefox's WebGPU implementation with:
 * - Smaller workgroup size (128) for better thread occupancy
 * - Simple, non-vectorized operations that perform well in Firefox
 * - Avoids complex control flow that can be problematic in Firefox's shader compiler
 */
${FAST_MATH_FUNCTIONS}

struct Params {
  size: u32,
  batch_size: u32,
  alpha: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Guard against out-of-bounds
  if (idx >= params.size) {
    return;
  }
  
  // Simple element-by-element processing for Firefox
  let value = input[idx];
  
  // Apply Leaky ReLU with custom alpha
  output[idx] = leaky_relu(value, params.alpha);
}`;

const FIREFOX_LAYER_NORMALIZATION_SHADER = `/**
 * Firefox-optimized Layer Normalization shader
 * 
 * Optimized for Firefox's WebGPU implementation with:
 * - Smaller workgroup size (128) for better thread occupancy
 * - Simple, non-vectorized operations that perform well in Firefox
 * - Avoids complex control flow that can be problematic in Firefox's shader compiler
 * - Two-pass algorithm to compute mean and variance
 */

struct Params {
  hidden_size: u32,      // Size of the hidden dimension to normalize across
  batch_seq_len: u32,    // Batch size * sequence length
  epsilon: f32,          // Small value to avoid division by zero
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>; // Scale parameter
@group(0) @binding(3) var<storage, read> beta: array<f32>;  // Shift parameter
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Shared workgroup memory for reduction operations
var<workgroup> reduction_data: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  let hidden_size = params.hidden_size;
  let batch_seq_len = params.batch_seq_len;
  let epsilon = params.epsilon;
  
  // Calculate which sequence element this workgroup is processing
  let seq_idx = global_id.x / 128u;
  
  // Guard against out-of-bounds
  if (seq_idx >= batch_seq_len) {
    return;
  }
  
  // Base index for this sequence element
  let base_idx = seq_idx * hidden_size;
  
  // Each thread processes a portion of the hidden dimension
  let elements_per_thread = (hidden_size + 127u) / 128u;
  let thread_idx = local_id.x;
  
  // ---------------------------
  // Step 1: Calculate the mean
  // ---------------------------
  var thread_sum: f32 = 0.0;
  
  for (var i = 0u; i < elements_per_thread; i++) {
    let idx = base_idx + thread_idx + i * 128u;
    if (idx < base_idx + hidden_size) {
      thread_sum += input[idx];
    }
  }
  
  // Store partial sum in shared memory
  reduction_data[thread_idx] = thread_sum;
  
  // Parallel reduction to compute the sum across the entire hidden dimension
  workgroupBarrier();
  
  // Reduce within workgroup (using parallel reduction)
  for (var stride = 64u; stride > 0u; stride >>= 1u) {
    if (thread_idx < stride) {
      reduction_data[thread_idx] += reduction_data[thread_idx + stride];
    }
    workgroupBarrier();
  }
  
  // Mean is now in reduction_data[0], divided by hidden_size
  let mean = reduction_data[0] / f32(hidden_size);
  
  // --------------------------------
  // Step 2: Calculate the variance
  // --------------------------------
  var thread_variance: f32 = 0.0;
  
  for (var i = 0u; i < elements_per_thread; i++) {
    let idx = base_idx + thread_idx + i * 128u;
    if (idx < base_idx + hidden_size) {
      let diff = input[idx] - mean;
      thread_variance += diff * diff;
    }
  }
  
  // Store partial variance in shared memory
  reduction_data[thread_idx] = thread_variance;
  
  // Parallel reduction to compute variance
  workgroupBarrier();
  
  // Reduce within workgroup
  for (var stride = 64u; stride > 0u; stride >>= 1u) {
    if (thread_idx < stride) {
      reduction_data[thread_idx] += reduction_data[thread_idx + stride];
    }
    workgroupBarrier();
  }
  
  // Variance is now in reduction_data[0], divided by hidden_size
  let variance = reduction_data[0] / f32(hidden_size);
  
  // Compute standard deviation
  let inv_std = 1.0 / sqrt(variance + epsilon);
  
  // --------------------------------
  // Step 3: Normalize and scale
  // --------------------------------
  for (var i = 0u; i < elements_per_thread; i++) {
    let hidden_idx = thread_idx + i * 128u;
    let idx = base_idx + hidden_idx;
    
    if (hidden_idx < hidden_size) {
      // Apply normalization, scaling, and shifting
      // y = ((x - mean) / sqrt(variance + epsilon)) * gamma + beta
      let normalized = (input[idx] - mean) * inv_std;
      let scaled = normalized * gamma[hidden_idx] + beta[hidden_idx];
      output[idx] = scaled;
    }
  }
}`;

const SAFARI_ELEMENTWISE_RELU_SHADER = `/**
 * Safari-optimized elementwise ReLU shader
 * 
 * Optimized for Safari's Metal-based WebGPU implementation with:
 * - Uses vector types for efficient processing
 * - Uses SIMDGroup operations when available
 * - Optimized for Apple GPU architecture
 */

struct Params {
  size: u32,
  batch_size: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(512)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Process elements in vectors of 4 when possible
  let vector_idx = idx * 4u;
  
  if (vector_idx + 3u < params.size) {
    // Full vector load is possible
    let input_vec = vec4<f32>(
      input[vector_idx],
      input[vector_idx + 1u],
      input[vector_idx + 2u],
      input[vector_idx + 3u]
    );
    
    // Apply ReLU using vector operation
    let result_vec = max(vec4<f32>(0.0), input_vec);
    
    // Store results
    output[vector_idx] = result_vec.x;
    output[vector_idx + 1u] = result_vec.y;
    output[vector_idx + 2u] = result_vec.z;
    output[vector_idx + 3u] = result_vec.w;
  } else {
    // Handle boundary with individual loads
    for (var i = 0u; i < 4u; i++) {
      let element_idx = vector_idx + i;
      if (element_idx < params.size) {
        output[element_idx] = max(0.0, input[element_idx]);
      }
    }
  }
}`;

const CHROME_MATMUL_SHADER = `/**
 * Chrome-optimized matrix multiplication shader
 * 
 * Optimized for Chrome's WebGPU implementation with:
 * - Workgroup size of 16x16 for balanced performance
 * - Coalesced memory access patterns for better performance on desktop GPUs
 * - 4-element unrolled loops for better vectorization
 * - Efficient shared memory usage
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

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  
  var<workgroup> A_shared: array<f32, 16 * 16>;
  var<workgroup> B_shared: array<f32, 16 * 16>;

  let batch_idx = global_id.z;
  let row = global_id.x;
  let col = global_id.y;
  
  if (row >= params.M || col >= params.N || batch_idx >= params.batch_size) {
    return;
  }
  
  let a_batch_offset = batch_idx * params.M * params.K;
  let b_batch_offset = batch_idx * params.K * params.N;
  let c_batch_offset = batch_idx * params.M * params.N;
  
  var sum: f32 = 0.0;
  let num_tiles = (params.K + 16u - 1u) / 16u;
  
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let tile_start = t * 16u;
    
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
    
    workgroupBarrier();
    
    var k = 0u;
    for (; k + 4u <= 16u && tile_start + k < params.K; k = k + 4u) {
      let a0 = A_shared[local_id.x * 16u + k];
      let a1 = A_shared[local_id.x * 16u + k + 1u];
      let a2 = A_shared[local_id.x * 16u + k + 2u];
      let a3 = A_shared[local_id.x * 16u + k + 3u];
      
      let b0 = B_shared[k * 16u + local_id.y];
      let b1 = B_shared[(k + 1u) * 16u + local_id.y];
      let b2 = B_shared[(k + 2u) * 16u + local_id.y];
      let b3 = B_shared[(k + 3u) * 16u + local_id.y];
      
      sum = sum + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    }
    
    for (; k < 16u && tile_start + k < params.K; k = k + 1u) {
      sum = sum + A_shared[local_id.x * 16u + k] * B_shared[k * 16u + local_id.y];
    }
    
    workgroupBarrier();
  }
  
  if (params.alpha != 1.0 || params.beta != 0.0) {
    var existing: f32 = 0.0;
    if (params.beta != 0.0) {
      existing = C[c_batch_offset + row * params.N + col];
    }
    
    sum = params.alpha * sum + params.beta * existing;
  }
  
  C[c_batch_offset + row * params.N + col] = sum;
}`;

const FIREFOX_MATMUL_SHADER = `/**
 * Firefox-optimized matrix multiplication shader
 * 
 * Optimized for Firefox's WebGPU implementation with:
 * - Smaller workgroup size (8x8) for better occupancy
 * - Simpler memory access patterns
 * - Non-unrolled loops that perform better in Firefox
 * - Reduced register usage
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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  
  var<workgroup> A_shared: array<f32, 8 * 8>;
  var<workgroup> B_shared: array<f32, 8 * 8>;

  let batch_idx = global_id.z;
  let row = global_id.x;
  let col = global_id.y;
  
  if (row >= params.M || col >= params.N || batch_idx >= params.batch_size) {
    return;
  }
  
  let a_batch_offset = batch_idx * params.M * params.K;
  let b_batch_offset = batch_idx * params.K * params.N;
  let c_batch_offset = batch_idx * params.M * params.N;
  
  var sum: f32 = 0.0;
  let num_tiles = (params.K + 8u - 1u) / 8u;
  
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let tile_start = t * 8u;
    
    let A_row = row;
    let A_col = tile_start + local_id.y;
    
    if (A_row < params.M && A_col < params.K) {
      A_shared[local_id.x * 8u + local_id.y] = A[a_batch_offset + A_row * params.K + A_col];
    } else {
      A_shared[local_id.x * 8u + local_id.y] = 0.0;
    }
    
    let B_row = tile_start + local_id.x;
    let B_col = col;
    
    if (B_row < params.K && B_col < params.N) {
      B_shared[local_id.x * 8u + local_id.y] = B[b_batch_offset + B_row * params.N + B_col];
    } else {
      B_shared[local_id.x * 8u + local_id.y] = 0.0;
    }
    
    workgroupBarrier();
    
    for (var k: u32 = 0u; k < 8u && tile_start + k < params.K; k = k + 1u) {
      sum = sum + A_shared[local_id.x * 8u + k] * B_shared[k * 8u + local_id.y];
    }
    
    workgroupBarrier();
  }
  
  if (params.alpha != 1.0 || params.beta != 0.0) {
    var existing: f32 = 0.0;
    if (params.beta != 0.0) {
      existing = C[c_batch_offset + row * params.N + col];
    }
    
    sum = params.alpha * sum + params.beta * existing;
  }
  
  C[c_batch_offset + row * params.N + col] = sum;
}`;

const SAFARI_MATMUL_SHADER = `/**
 * Safari-optimized matrix multiplication shader
 * 
 * Optimized for Safari's Metal-based WebGPU implementation with:
 * - Vector operations utilizing Apple GPU SIMD units
 * - Larger workgroup size and tile size for Apple Silicon GPUs
 * - Optimized memory access patterns for Metal GPU architecture
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

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  
  var<workgroup> A_shared: array<f32, 32 * 32>;
  var<workgroup> B_shared: array<f32, 32 * 32>;

  let batch_idx = global_id.z;
  let row = global_id.x;
  let col = global_id.y;
  
  if (row >= params.M || col >= params.N || batch_idx >= params.batch_size) {
    return;
  }
  
  let a_batch_offset = batch_idx * params.M * params.K;
  let b_batch_offset = batch_idx * params.K * params.N;
  let c_batch_offset = batch_idx * params.M * params.N;
  
  var sum: f32 = 0.0;
  let num_tiles = (params.K + 32u - 1u) / 32u;
  
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let tile_start = t * 32u;
    
    let A_row = row;
    let A_col = tile_start + local_id.y;
    
    if (A_row < params.M && A_col < params.K) {
      A_shared[local_id.x * 32u + local_id.y] = A[a_batch_offset + A_row * params.K + A_col];
    } else {
      A_shared[local_id.x * 32u + local_id.y] = 0.0;
    }
    
    let A_col2 = tile_start + local_id.y + 16u;
    if (A_row < params.M && A_col2 < params.K) {
      A_shared[local_id.x * 32u + local_id.y + 16u] = A[a_batch_offset + A_row * params.K + A_col2];
    } else {
      A_shared[local_id.x * 32u + local_id.y + 16u] = 0.0;
    }
    
    let B_row = tile_start + local_id.x;
    let B_col = col;
    
    if (B_row < params.K && B_col < params.N) {
      B_shared[local_id.x * 32u + local_id.y] = B[b_batch_offset + B_row * params.N + B_col];
    } else {
      B_shared[local_id.x * 32u + local_id.y] = 0.0;
    }
    
    let B_row2 = tile_start + local_id.x + 16u;
    if (B_row2 < params.K && B_col < params.N) {
      B_shared[(local_id.x + 16u) * 32u + local_id.y] = B[b_batch_offset + B_row2 * params.N + B_col];
    } else {
      B_shared[(local_id.x + 16u) * 32u + local_id.y] = 0.0;
    }
    
    workgroupBarrier();
    
    var k = 0u;
    for (; k + 4u <= 32u && tile_start + k < params.K; k = k + 4u) {
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
      
      sum = sum + dot(a_vec, b_vec);
    }
    
    for (; k < 32u && tile_start + k < params.K; k = k + 1u) {
      sum = sum + A_shared[local_id.x * 32u + k] * B_shared[k * 32u + local_id.y];
    }
    
    workgroupBarrier();
  }
  
  if (params.alpha != 1.0 || params.beta != 0.0) {
    var existing: f32 = 0.0;
    if (params.beta != 0.0) {
      existing = C[c_batch_offset + row * params.N + col];
    }
    
    sum = params.alpha * sum + params.beta * existing;
  }
  
  C[c_batch_offset + row * params.N + col] = sum;
}`;

const EDGE_MATMUL_SHADER = `/**
 * Edge-optimized matrix multiplication shader
 * 
 * Optimized for Edge's WebGPU implementation with:
 * - Hybrid approach combining Chrome-like performance with Edge-specific optimizations
 * - Balanced tile size and workgroup size
 * - Partially unrolled loops
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

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  
  var<workgroup> A_shared: array<f32, 16 * 16>;
  var<workgroup> B_shared: array<f32, 16 * 16>;

  let batch_idx = global_id.z;
  let row = global_id.x;
  let col = global_id.y;
  
  if (row >= params.M || col >= params.N || batch_idx >= params.batch_size) {
    return;
  }
  
  let a_batch_offset = batch_idx * params.M * params.K;
  let b_batch_offset = batch_idx * params.K * params.N;
  let c_batch_offset = batch_idx * params.M * params.N;
  
  var sum: f32 = 0.0;
  let num_tiles = (params.K + 16u - 1u) / 16u;
  
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let tile_start = t * 16u;
    
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
    
    workgroupBarrier();
    
    var k = 0u;
    for (; k + 2u <= 16u && tile_start + k < params.K; k = k + 2u) {
      let a0 = A_shared[local_id.x * 16u + k];
      let a1 = A_shared[local_id.x * 16u + k + 1u];
      
      let b0 = B_shared[k * 16u + local_id.y];
      let b1 = B_shared[(k + 1u) * 16u + local_id.y];
      
      sum = sum + a0 * b0 + a1 * b1;
    }
    
    for (; k < 16u && tile_start + k < params.K; k = k + 1u) {
      sum = sum + A_shared[local_id.x * 16u + k] * B_shared[k * 16u + local_id.y];
    }
    
    workgroupBarrier();
  }
  
  if (params.alpha != 1.0 || params.beta != 0.0) {
    var existing: f32 = 0.0;
    if (params.beta != 0.0) {
      existing = C[c_batch_offset + row * params.N + col];
    }
    
    sum = params.alpha * sum + params.beta * existing;
  }
  
  C[c_batch_offset + row * params.N + col] = sum;
}`;

const CHROME_QUANTIZED_MATMUL_SHADER = `/**
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

@compute @workgroup_size(8, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  
  if (row >= params.M || col >= params.N) {
    return;
  }
  
  let output_idx = row * params.N + col;
  var sum: f32 = 0.0;
  
  let values_per_word = 32u / params.bits_per_weight;
  let words_per_row = (params.K + values_per_word - 1u) / values_per_word;
  
  let scale_idx = select(0u, col, params.scale_mode == 1u);
  let zero_point = select(0.0, zero_points[scale_idx], params.zero_point_mode == 1u);
  let scale = scales[scale_idx];

  for (var word_idx: u32 = 0u; word_idx < words_per_row; word_idx++) {
    let weight_offset = row * words_per_row + word_idx;
    
    if (weight_offset < params.M * words_per_row) {
      let packed_word = quantized_weights[weight_offset];
      let k_start = word_idx * values_per_word;
      
      if (params.bits_per_weight == 4u) {
        for (var i: u32 = 0u; i < 8u && k_start + i < params.K; i++) {
          let unpacked_value = unpack_4bit(packed_word, i);
          let dequantized = dequantize(unpacked_value, scale, zero_point);
          let activation = activations[k_start + i];
          sum += dequantized * activation;
        }
      } else if (params.bits_per_weight == 8u) {
        for (var i: u32 = 0u; i < 4u && k_start + i < params.K; i++) {
          let unpacked_value = unpack_8bit(packed_word, i);
          let dequantized = dequantize(unpacked_value, scale, zero_point);
          let activation = activations[k_start + i];
          sum += dequantized * activation;
        }
      } else if (params.bits_per_weight == 2u) {
        for (var i: u32 = 0u; i < 16u && k_start + i < params.K; i++) {
          let unpacked_value = unpack_2bit(packed_word, i);
          let dequantized = dequantize_2bit(unpacked_value, scale, zero_point);
          let activation = activations[k_start + i];
          sum += dequantized * activation;
        }
      }
    }
  }
  
  output[output_idx] = sum;
}

fn unpack_4bit(packed: u32, idx: u32) -> f32 {
  let shift = idx * 4u;
  let mask = 0xfu;
  let value = (packed >> shift) & mask;
  return f32(value);
}

fn unpack_2bit(packed: u32, idx: u32) -> f32 {
  let shift = idx * 2u;
  let mask = 0x3u;
  let value = (packed >> shift) & mask;
  return f32(value);
}

fn unpack_8bit(packed: u32, idx: u32) -> f32 {
  let shift = idx * 8u;
  let mask = 0xffu;
  let value = (packed >> shift) & mask;
  return f32(value);
}

fn dequantize(quantized: f32, scale: f32, zero_point: f32) -> f32 {
  return (quantized - zero_point) * scale;
}

fn dequantize_2bit(quantized: f32, scale: f32, zero_point: f32) -> f32 {
  let value_map = array<f32, 4>(-1.0, 0.0, 0.5, 1.0);
  return value_map[u32(quantized)] * scale;
}`;

const FIREFOX_QUANTIZED_MATMUL_SHADER = `/**
 * Firefox-optimized quantized matrix multiplication shader
 * 
 * Optimized for Firefox's WebGPU implementation with:
 * - Uses smaller workgroup size (8x8) for better occupancy
 * - Simpler memory access patterns that perform better in Firefox
 * - Reduced unrolling for better Firefox shader compiler performance
 * - Optimized bit extraction operations for different bit widths
 */

@group(0) @binding(0) var<storage, read> a: array<f32>; // Input matrix A
@group(0) @binding(1) var<storage, read> b: array<u32>; // Quantized input matrix B
@group(0) @binding(2) var<storage, read_write> c: array<f32>; // Output matrix
@group(0) @binding(3) var<storage, read> scales: array<f32>; // Scales for dequantization
@group(0) @binding(4) var<storage, read> zeroPoints: array<f32>; // Zero points for dequantization

// Uniforms for matrix dimensions and quantization parameters
struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    bitsPerWeight: u32,
    usePerChannelQuantization: u32,
    paddingDummy1: u32,
    paddingDummy2: u32,
    paddingDummy3: u32,
};
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

// Shared memory for tiled matrix multiplication
var<workgroup> aTile: array<f32, 8 * 8>;
var<workgroup> bTile: array<f32, 8 * 8>;

// Returns a dequantized value from a packed quantized weight
fn dequantize1Bit(packedValue: u32, bitIndex: u32, scale: f32, zeroPoint: f32) -> f32 {
    let bitValue = (packedValue >> bitIndex) & 1u;
    return f32(bitValue) * scale + zeroPoint;
}

fn dequantize2Bit(packedValue: u32, bitIndex: u32, scale: f32, zeroPoint: f32) -> f32 {
    let bitValue = (packedValue >> (bitIndex * 2u)) & 3u;
    return f32(bitValue) * scale + zeroPoint;
}

fn dequantize3Bit(packedValue: u32, bitIndex: u32, scale: f32, zeroPoint: f32) -> f32 {
    // For 3-bit, we pack 10 values in each 32-bit word (3*10=30 bits used)
    // For 0-9 indexes: [0-2], [3-5], [6-8], [9-11], [12-14], [15-17], [18-20], [21-23], [24-26], [27-29]
    let startBit = bitIndex * 3u;
    let mask = 7u; // 0b111
    let bitValue = (packedValue >> startBit) & mask;
    return f32(bitValue) * scale + zeroPoint;
}

fn dequantize4Bit(packedValue: u32, bitIndex: u32, scale: f32, zeroPoint: f32) -> f32 {
    let bitValue = (packedValue >> (bitIndex * 4u)) & 15u;
    return f32(bitValue) * scale + zeroPoint;
}

fn dequantize8Bit(packedValue: u32, bitIndex: u32, scale: f32, zeroPoint: f32) -> f32 {
    let bitValue = (packedValue >> (bitIndex * 8u)) & 255u;
    return f32(bitValue) * scale + zeroPoint;
}

// Main compute shader
@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;
    let bitsPerWeight = uniforms.bitsPerWeight;
    let usePerChannelQuantization = uniforms.usePerChannelQuantization;

    // Matrix dimensions validation
    if (global_id.x >= N || global_id.y >= M) {
        return;
    }

    let row = global_id.y;
    let col = global_id.x;
    
    // Calculate tile indices
    let tileRow = local_id.y;
    let tileCol = local_id.x;
    
    // Initialize accumulator
    var acc = 0.0;
    
    // Number of tiles needed to cover the K dimension
    let numTiles = (K + 7u) / 8u;
    
    // Values per 32-bit word depends on bit width
    var valuesPerWord = 32u / bitsPerWeight;
    if (bitsPerWeight == 3u) {
        valuesPerWord = 10u; // Special case for 3-bit: 10 values per 32-bit word
    }
    
    // Load appropriate scale and zero point
    var scale = 0.0;
    var zeroPoint = 0.0;
    
    if (usePerChannelQuantization == 1u) {
        scale = scales[col];
        zeroPoint = zeroPoints[col];
    } else {
        scale = scales[0];
        zeroPoint = zeroPoints[0];
    }
    
    // Process tiles
    for (var t = 0u; t < numTiles; t = t + 1u) {
        // Load tile of matrix A into shared memory
        let aRow = row;
        let aCol = t * 8u + tileCol;
        
        if (aCol < K) {
            aTile[tileRow * 8u + tileCol] = a[aRow * K + aCol];
        } else {
            aTile[tileRow * 8u + tileCol] = 0.0;
        }
        
        // Load and dequantize tile of matrix B into shared memory
        let bRow = t * 8u + tileRow;
        let bCol = col;
        
        if (bRow < K && bCol < N) {
            // Calculate the index in the packed quantized data
            let packedIdx = (bRow * N + bCol) / valuesPerWord;
            let valueIdx = (bRow * N + bCol) % valuesPerWord;
            let packedValue = b[packedIdx];
            
            // Dequantize based on bit width
            var dequantizedValue = 0.0;
            
            if (bitsPerWeight == 1u) {
                dequantizedValue = dequantize1Bit(packedValue, valueIdx, scale, zeroPoint);
            } else if (bitsPerWeight == 2u) {
                dequantizedValue = dequantize2Bit(packedValue, valueIdx, scale, zeroPoint);
            } else if (bitsPerWeight == 3u) {
                dequantizedValue = dequantize3Bit(packedValue, valueIdx, scale, zeroPoint);
            } else if (bitsPerWeight == 4u) {
                dequantizedValue = dequantize4Bit(packedValue, valueIdx, scale, zeroPoint);
            } else if (bitsPerWeight == 8u) {
                dequantizedValue = dequantize8Bit(packedValue, valueIdx, scale, zeroPoint);
            }
            
            bTile[tileRow * 8u + tileCol] = dequantizedValue;
        } else {
            bTile[tileRow * 8u + tileCol] = 0.0;
        }
        
        // Synchronize to make sure both tiles are loaded
        workgroupBarrier();
        
        // Compute partial dot product for this tile
        for (var k = 0u; k < 8u; k = k + 1u) {
            acc = acc + aTile[tileRow * 8u + k] * bTile[k * 8u + tileCol];
        }
        
        // Synchronize before loading the next tile
        workgroupBarrier();
    }
    
    // Write result
    c[row * N + col] = acc;
}`;

const EDGE_QUANTIZED_MATMUL_SHADER = `/**
 * Edge-optimized quantized matrix multiplication shader
 * 
 * Optimized for Microsoft Edge's WebGPU implementation with:
 * - Uses 16x16 workgroup size similar to Chrome but with Edge-specific enhancements
 * - Implements partially unrolled loops for better performance
 * - Takes a hybrid approach between Chrome and Firefox patterns
 * - Specialized handling for WebNN integration
 */

@group(0) @binding(0) var<storage, read> a: array<f32>; // Input matrix A
@group(0) @binding(1) var<storage, read> b: array<u32>; // Quantized input matrix B
@group(0) @binding(2) var<storage, read_write> c: array<f32>; // Output matrix
@group(0) @binding(3) var<storage, read> scales: array<f32>; // Scales for dequantization
@group(0) @binding(4) var<storage, read> zeroPoints: array<f32>; // Zero points for dequantization

// Uniforms for matrix dimensions and quantization parameters
struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    bitsPerWeight: u32,
    usePerChannelQuantization: u32,
    paddingDummy1: u32,
    paddingDummy2: u32,
    paddingDummy3: u32,
};
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

// Shared memory for tiled matrix multiplication
var<workgroup> aTile: array<f32, 16 * 16>;
var<workgroup> bTile: array<f32, 16 * 16>;

// Helper functions for dequantization
fn dequantize1Bit(packedValue: u32, bitIndex: u32, scale: f32, zeroPoint: f32) -> f32 {
    let bitValue = (packedValue >> bitIndex) & 1u;
    return f32(bitValue) * scale + zeroPoint;
}

fn dequantize2Bit(packedValue: u32, bitIndex: u32, scale: f32, zeroPoint: f32) -> f32 {
    let bitValue = (packedValue >> (bitIndex * 2u)) & 3u;
    return f32(bitValue) * scale + zeroPoint;
}

fn dequantize3Bit(packedValue: u32, bitIndex: u32, scale: f32, zeroPoint: f32) -> f32 {
    // For 3-bit, we pack 10 values in each 32-bit word (3*10=30 bits used)
    // For 0-9 indexes: [0-2], [3-5], [6-8], [9-11], [12-14], [15-17], [18-20], [21-23], [24-26], [27-29]
    let startBit = bitIndex * 3u;
    let mask = 7u; // 0b111
    let bitValue = (packedValue >> startBit) & mask;
    return f32(bitValue) * scale + zeroPoint;
}

fn dequantize4Bit(packedValue: u32, bitIndex: u32, scale: f32, zeroPoint: f32) -> f32 {
    let bitValue = (packedValue >> (bitIndex * 4u)) & 15u;
    return f32(bitValue) * scale + zeroPoint;
}

fn dequantize8Bit(packedValue: u32, bitIndex: u32, scale: f32, zeroPoint: f32) -> f32 {
    let bitValue = (packedValue >> (bitIndex * 8u)) & 255u;
    return f32(bitValue) * scale + zeroPoint;
}

// Edge-optimized matrix multiplication
@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;
    let bitsPerWeight = uniforms.bitsPerWeight;
    let usePerChannelQuantization = uniforms.usePerChannelQuantization;

    // Matrix dimensions validation
    if (global_id.x >= N || global_id.y >= M) {
        return;
    }

    let row = global_id.y;
    let col = global_id.x;
    
    // Calculate tile indices
    let tileRow = local_id.y;
    let tileCol = local_id.x;
    
    // Initialize accumulator
    var acc = 0.0;
    
    // Number of tiles needed to cover the K dimension
    let numTiles = (K + 15u) / 16u;
    
    // Values per 32-bit word depends on bit width
    var valuesPerWord = 32u / bitsPerWeight;
    if (bitsPerWeight == 3u) {
        valuesPerWord = 10u; // Special case for 3-bit: 10 values per 32-bit word
    }
    
    // Load appropriate scale and zero point
    var scale = 0.0;
    var zeroPoint = 0.0;
    
    if (usePerChannelQuantization == 1u) {
        scale = scales[col];
        zeroPoint = zeroPoints[col];
    } else {
        scale = scales[0];
        zeroPoint = zeroPoints[0];
    }
    
    // Process tiles - Edge optimized loop with partial unrolling in pairs
    for (var t = 0u; t < numTiles; t = t + 1u) {
        // Load tile of matrix A into shared memory
        let aRow = row;
        let aCol = t * 16u + tileCol;
        
        if (aCol < K) {
            aTile[tileRow * 16u + tileCol] = a[aRow * K + aCol];
        } else {
            aTile[tileRow * 16u + tileCol] = 0.0;
        }
        
        // Load and dequantize tile of matrix B into shared memory
        let bRow = t * 16u + tileRow;
        let bCol = col;
        
        if (bRow < K && bCol < N) {
            // Calculate the index in the packed quantized data
            let packedIdx = (bRow * N + bCol) / valuesPerWord;
            let valueIdx = (bRow * N + bCol) % valuesPerWord;
            let packedValue = b[packedIdx];
            
            // Dequantize based on bit width
            var dequantizedValue = 0.0;
            
            switch(bitsPerWeight) {
                case 1u: {
                    dequantizedValue = dequantize1Bit(packedValue, valueIdx, scale, zeroPoint);
                    break;
                }
                case 2u: {
                    dequantizedValue = dequantize2Bit(packedValue, valueIdx, scale, zeroPoint);
                    break;
                }
                case 3u: {
                    dequantizedValue = dequantize3Bit(packedValue, valueIdx, scale, zeroPoint);
                    break;
                }
                case 4u: {
                    dequantizedValue = dequantize4Bit(packedValue, valueIdx, scale, zeroPoint);
                    break;
                }
                case 8u: {
                    dequantizedValue = dequantize8Bit(packedValue, valueIdx, scale, zeroPoint);
                    break;
                }
                default: {
                    dequantizedValue = 0.0;
                    break;
                }
            }
            
            bTile[tileRow * 16u + tileCol] = dequantizedValue;
        } else {
            bTile[tileRow * 16u + tileCol] = 0.0;
        }
        
        // Synchronize to make sure both tiles are loaded
        workgroupBarrier();
        
        // Compute partial dot product for this tile with partial unrolling in pairs
        // Edge's shader compiler benefits from this partially unrolled pattern
        for (var k = 0u; k < 16u; k = k + 2u) {
            // Process two elements at once - partial unrolling works well in Edge
            acc = acc + aTile[tileRow * 16u + k] * bTile[k * 16u + tileCol];
            
            // Edge's compiler benefits from explicit bounds checking vs Chrome
            if (k + 1u < 16u) {
                acc = acc + aTile[tileRow * 16u + (k + 1u)] * bTile[(k + 1u) * 16u + tileCol];
            }
        }
        
        // Synchronize before loading the next tile
        workgroupBarrier();
    }
    
    // Write result
    c[row * N + col] = acc;
}`;

const SAFARI_QUANTIZED_MATMUL_SHADER = `/**
 * Safari-optimized quantized matrix multiplication shader
 * 
 * Optimized for Safari's Metal-based WebGPU implementation with:
 * - Special optimizations for Apple Silicon GPUs
 * - Larger workgroup size for better utilization of Apple GPU cores
 * - Vector operations for better performance on Apple Silicon
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

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  
  if (row >= params.M || col >= params.N) {
    return;
  }
  
  let output_idx = row * params.N + col;
  var sum: f32 = 0.0;
  
  var values_per_word: u32;
  
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
      values_per_word = 8u;
    }
  }
  
  let words_per_row = (params.K + values_per_word - 1u) / values_per_word;
  
  let scale_idx = select(0u, col, params.scale_mode == 1u);
  let zero_point = select(0.0, zero_points[scale_idx], params.zero_point_mode == 1u);
  let scale = scales[scale_idx];

  for (var word_idx: u32 = 0u; word_idx < words_per_row; word_idx++) {
    let weight_offset = row * words_per_row + word_idx;
    
    if (weight_offset < params.M * words_per_row) {
      let packed_word = quantized_weights[weight_offset];
      let k_start = word_idx * values_per_word;
      
      if (params.bits_per_weight == 3u) {
        var i: u32 = 0u;
        
        for (; i + 4u <= 10u && k_start + i < params.K; i += 4u) {
          let activation_vec = vec4<f32>(
            activations[k_start + i],
            activations[k_start + i + 1u],
            activations[k_start + i + 2u],
            activations[k_start + i + 3u]
          );
          
          let unpacked_values = vec4<f32>(
            unpack_3bit(packed_word, i),
            unpack_3bit(packed_word, i + 1u),
            unpack_3bit(packed_word, i + 2u),
            unpack_3bit(packed_word, i + 3u)
          );
          
          let dequantized_values = (unpacked_values - vec4<f32>(zero_point)) * vec4<f32>(scale);
          
          sum += dot(dequantized_values, activation_vec);
        }
        
        for (; i < 10u && k_start + i < params.K; i++) {
          let unpacked_value = unpack_3bit(packed_word, i);
          let dequantized = (unpacked_value - zero_point) * scale;
          let activation = activations[k_start + i];
          sum += dequantized * activation;
        }
      } else if (params.bits_per_weight == 8u || params.bits_per_weight == 4u) {
        for (var i: u32 = 0u; i < values_per_word && k_start + i < params.K; i++) {
          let unpacked_value = params.bits_per_weight == 4u ? 
                              unpack_4bit(packed_word, i) : 
                              unpack_8bit(packed_word, i);
          let dequantized = (unpacked_value - zero_point) * scale;
          let activation = activations[k_start + i];
          sum += dequantized * activation;
        }
      } else if (params.bits_per_weight == 2u) {
        for (var i: u32 = 0u; i < 16u && k_start + i < params.K; i++) {
          let unpacked_value = unpack_2bit(packed_word, i);
          let dequantized = dequantize_2bit(unpacked_value, scale, zero_point);
          let activation = activations[k_start + i];
          sum += dequantized * activation;
        }
      }
    }
  }
  
  output[output_idx] = sum;
}

fn unpack_3bit(packed: u32, idx: u32) -> f32 {
  let bit_pos = idx * 3u;
  let mask = 0x7u;
  let value = (packed >> bit_pos) & mask;
  return f32(value);
}

fn unpack_4bit(packed: u32, idx: u32) -> f32 {
  let shift = idx * 4u;
  let mask = 0xfu;
  let value = (packed >> shift) & mask;
  return f32(value);
}

fn unpack_8bit(packed: u32, idx: u32) -> f32 {
  let shift = idx * 8u;
  let mask = 0xffu;
  let value = (packed >> shift) & mask;
  return f32(value);
}

fn unpack_2bit(packed: u32, idx: u32) -> f32 {
  let shift = idx * 2u;
  let mask = 0x3u;
  let value = (packed >> shift) & mask;
  return f32(value);
}

fn dequantize_2bit(quantized: f32, scale: f32, zero_point: f32) -> f32 {
  let value_map = array<f32, 4>(-1.0, -0.33, 0.33, 1.0);
  return value_map[u32(quantized)] * scale;
}`;