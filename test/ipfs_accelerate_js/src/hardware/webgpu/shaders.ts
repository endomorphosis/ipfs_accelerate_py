/**
 * WebGPU Shader Collection
 * Contains WGSL compute shaders for tensor operations
 */

/**
 * Shader for basic element-wise operations
 */
export const elementwiseOperationShader = /* wgsl */`
// Binding group layout:
// binding 0: Input buffer A
// binding 1: Input buffer B (for binary operations)
// binding 2: Output buffer
// binding 3: Uniform buffer with operation parameters

struct Params {
  // Operation type: 0=add, 1=subtract, 2=multiply, 3=divide,
  // 4=pow, 5=exp, 6=log, 7=sqrt, 8=relu, 9=sigmoid, 10=tanh
  op: u32,
  
  // Shapes and dimensions
  dim: u32,
  length: u32,
  output_size: u32,
  
  // For unary operations with optional parameters
  alpha: f32,
  beta: f32,
};

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// Helper function for sigmoid
fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

// Helper function for tanh
fn tanh_approx(x: f32) -> f32 {
  // Tanh can be expressed in terms of sigmoid: tanh(x) = 2 * sigmoid(2x) - 1
  let s = sigmoid(2.0 * x);
  return 2.0 * s - 1.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Bounds check
  if (idx >= params.output_size) {
    return;
  }
  
  var result: f32;
  
  // Perform the operation
  switch(params.op) {
    // Binary operations
    case 0u: { // Add
      result = input_a[idx] + input_b[idx];
      break;
    }
    case 1u: { // Subtract
      result = input_a[idx] - input_b[idx];
      break;
    }
    case 2u: { // Multiply
      result = input_a[idx] * input_b[idx];
      break;
    }
    case 3u: { // Divide
      // Avoid division by zero
      if (abs(input_b[idx]) < 1e-7) {
        result = 0.0;
      } else {
        result = input_a[idx] / input_b[idx];
      }
      break;
    }
    case 4u: { // Power
      result = pow(input_a[idx], input_b[idx]);
      break;
    }
    
    // Unary operations
    case 5u: { // Exponential
      result = exp(input_a[idx]);
      break;
    }
    case 6u: { // Natural logarithm
      // Avoid log of negative/zero values
      if (input_a[idx] <= 0.0) {
        result = -999999.0; // A very negative number to indicate error
      } else {
        result = log(input_a[idx]);
      }
      break;
    }
    case 7u: { // Square root
      // Avoid sqrt of negative values
      if (input_a[idx] < 0.0) {
        result = 0.0;
      } else {
        result = sqrt(input_a[idx]);
      }
      break;
    }
    case 8u: { // ReLU
      result = max(0.0, input_a[idx]);
      break;
    }
    case 9u: { // Sigmoid
      result = sigmoid(input_a[idx]);
      break;
    }
    case 10u: { // Tanh
      result = tanh_approx(input_a[idx]);
      break;
    }
    default: {
      result = input_a[idx]; // Identity (fallback)
    }
  }
  
  // Write the result
  output[idx] = result;
}`;

/**
 * Shader for matrix multiplication
 */
export const matrixMultiplicationShader = /* wgsl */`
// Binding group layout:
// binding 0: Input matrix A
// binding 1: Input matrix B
// binding 2: Output matrix
// binding 3: Uniform buffer with matrix dimensions

struct Dimensions {
  M: u32,  // Rows in A
  K: u32,  // Columns in A / Rows in B
  N: u32,  // Columns in B
};

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: Dimensions;

// Tiled matrix multiplication for better performance
// Workgroup size of 16x16 is common for matrix operations
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  
  // Bounds check
  if (row >= dimensions.M || col >= dimensions.N) {
    return;
  }
  
  var sum: f32 = 0.0;
  
  // Compute the dot product for this output element
  for (var k: u32 = 0u; k < dimensions.K; k = k + 1u) {
    let a_index = row * dimensions.K + k;
    let b_index = k * dimensions.N + col;
    sum = sum + matrix_a[a_index] * matrix_b[b_index];
  }
  
  // Write the result
  let c_index = row * dimensions.N + col;
  matrix_c[c_index] = sum;
}`;

/**
 * Shader for matrix transposition
 */
export const transposeShader = /* wgsl */`
// Binding group layout:
// binding 0: Input matrix
// binding 1: Output matrix
// binding 2: Uniform buffer with matrix dimensions

struct Dimensions {
  rows: u32,
  cols: u32,
};

@group(0) @binding(0) var<storage, read> input_matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_matrix: array<f32>;
@group(0) @binding(2) var<uniform> dimensions: Dimensions;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  
  // Bounds check
  if (row >= dimensions.rows || col >= dimensions.cols) {
    return;
  }
  
  // Compute indices
  let input_idx = row * dimensions.cols + col;
  let output_idx = col * dimensions.rows + row;
  
  // Write the transposed value
  output_matrix[output_idx] = input_matrix[input_idx];
}`;

/**
 * Shader for softmax operation
 */
export const softmaxShader = /* wgsl */`
// Binding group layout:
// binding 0: Input tensor
// binding 1: Output tensor
// binding 2: Uniform buffer with tensor dimensions and axis

struct Params {
  total_elements: u32,
  inner_dim: u32,     // Size of dimension along which to compute softmax
  outer_dim: u32,     // Product of dimensions before the axis
  stride: u32,        // Product of dimensions after the axis
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Workgroup shared memory for reduction operations
var<workgroup> reduction_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Each workgroup handles one outer slice
  let outer_idx = workgroup_id.x;
  
  // Bounds check
  if (outer_idx >= params.outer_dim) {
    return;
  }
  
  // Find the max value in this slice (for numerical stability)
  var max_val = -3.402823e+38f;  // Minimum float value
  
  for (var i: u32 = 0u; i < params.inner_dim; i = i + 1u) {
    let input_idx = outer_idx * params.inner_dim * params.stride + i * params.stride;
    max_val = max(max_val, input[input_idx]);
  }
  
  // Compute exp(x - max_val) for each element and sum
  var sum = 0.0;
  
  for (var i: u32 = 0u; i < params.inner_dim; i = i + 1u) {
    let input_idx = outer_idx * params.inner_dim * params.stride + i * params.stride;
    let v = exp(input[input_idx] - max_val);
    sum = sum + v;
    
    // Store in temp buffer for second pass
    let output_idx = outer_idx * params.inner_dim * params.stride + i * params.stride;
    output[output_idx] = v;
  }
  
  // Normalize by the sum
  for (var i: u32 = 0u; i < params.inner_dim; i = i + 1u) {
    let output_idx = outer_idx * params.inner_dim * params.stride + i * params.stride;
    output[output_idx] = output[output_idx] / sum;
  }
}`;

/**
 * Shader for reduction operations (sum, mean, max, min)
 */
export const reductionShader = /* wgsl */`
// Binding group layout:
// binding 0: Input tensor
// binding 1: Output tensor
// binding 2: Uniform buffer with tensor dimensions and reduction type

struct Params {
  total_elements: u32,
  reduce_size: u32,    // Size of dimension to reduce
  outer_size: u32,     // Product of dimensions before the axis
  inner_size: u32,     // Product of dimensions after the axis
  reduce_op: u32,      // 0=sum, 1=mean, 2=max, 3=min
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let output_idx = global_id.x;
  
  // Bounds check
  if (output_idx >= params.outer_size * params.inner_size) {
    return;
  }
  
  // Calculate the outer and inner indices
  let outer_idx = output_idx / params.inner_size;
  let inner_idx = output_idx % params.inner_size;
  
  var result: f32;
  
  // Initialize based on reduction operation
  switch(params.reduce_op) {
    case 0u: { // Sum
      result = 0.0;
      break;
    }
    case 1u: { // Mean (start with sum)
      result = 0.0;
      break;
    }
    case 2u: { // Max
      result = -3.402823e+38f;  // Minimum float value
      break;
    }
    case 3u: { // Min
      result = 3.402823e+38f;   // Maximum float value
      break;
    }
    default: {
      result = 0.0;
    }
  }
  
  // Perform the reduction
  for (var i: u32 = 0u; i < params.reduce_size; i = i + 1u) {
    let input_idx = outer_idx * params.reduce_size * params.inner_size + 
                    i * params.inner_size + 
                    inner_idx;
    
    let val = input[input_idx];
    
    switch(params.reduce_op) {
      case 0u, 1u: { // Sum or Mean
        result = result + val;
        break;
      }
      case 2u: { // Max
        result = max(result, val);
        break;
      }
      case 3u: { // Min
        result = min(result, val);
        break;
      }
    }
  }
  
  // Finalize the result
  if (params.reduce_op == 1u) {
    // Mean = Sum / Count
    result = result / f32(params.reduce_size);
  }
  
  // Write the result
  output[output_idx] = result;
}`;

/**
 * Get a shader based on the operation name
 * @param name Operation name
 * @returns WGSL shader code
 */
export function getShader(name: string): string {
  switch (name) {
    case 'elementwise':
    case 'add':
    case 'subtract':
    case 'multiply':
    case 'divide':
    case 'relu':
    case 'sigmoid':
    case 'tanh':
      return elementwiseOperationShader;
    
    case 'matmul':
      return matrixMultiplicationShader;
    
    case 'transpose':
      return transposeShader;
    
    case 'softmax':
      return softmaxShader;
    
    case 'reduction':
    case 'sum':
    case 'mean':
    case 'max':
    case 'min':
      return reductionShader;
    
    default:
      throw new Error(`Shader not found for operation: ${name}`);
  }
}