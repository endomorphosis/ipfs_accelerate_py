/**
 * Specialized WGSL Shaders
 * Optimized implementations for common tensor operations
 */

/**
 * Configuration options for specialized shaders
 */
export interface SpecializedShaderOptions {
  /** Workgroup size for compute shaders */
  workgroupSize?: number;
  
  /** Whether to use specialized memory layout */
  useSpecializedLayout?: boolean;
  
  /** Whether to use fast math approximations */
  useFastMath?: boolean;
  
  /** Buffer size threshold for switching algorithms */
  algorithmSwitchThreshold?: number;
  
  /** Browser-specific optimization */
  browserOptimized?: 'chrome' | 'firefox' | 'safari' | 'edge';
}

/**
 * Default options for specialized shaders
 */
const DEFAULT_OPTIONS: SpecializedShaderOptions = {
  workgroupSize: 256,
  useSpecializedLayout: true,
  useFastMath: true,
  algorithmSwitchThreshold: 1024 * 1024 // 1M elements
};

/**
 * Get specialized matrix multiplication shader optimized for shape characteristics
 * @param M Number of rows in A
 * @param K Number of columns in A / rows in B
 * @param N Number of columns in B
 * @param options Specialization options
 * @returns WGSL shader code
 */
export function getSpecializedMatmulShader(
  M: number, 
  K: number, 
  N: number,
  options: SpecializedShaderOptions = {}
): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // Choose optimal workgroup size based on matrix dimensions
  // This can significantly affect performance based on hardware characteristics
  let workgroupSize = opts.workgroupSize!;
  
  // Tune workgroup dimensions based on matrix size
  let workgroupSizeX = 16;
  let workgroupSizeY = 16;
  
  // For tall matrices (M >> N), optimize for row parallelism
  if (M > 4 * N) {
    workgroupSizeX = 32;
    workgroupSizeY = 8;
  }
  // For wide matrices (N >> M), optimize for column parallelism
  else if (N > 4 * M) {
    workgroupSizeX = 8;
    workgroupSizeY = 32;
  }
  // For large, square matrices, use larger workgroups
  else if (M > 2048 && N > 2048) {
    workgroupSizeX = 32;
    workgroupSizeY = 32;
  }
  
  // Special optimization for Firefox (which sometimes benefits from different configurations)
  if (opts.browserOptimized === 'firefox') {
    // Firefox often benefits from smaller workgroups due to different scheduler
    workgroupSizeX = Math.min(workgroupSizeX, 16);
    workgroupSizeY = Math.min(workgroupSizeY, 16);
  }
  
  // Use tiling for large matrices to improve cache locality
  const useTiling = M * K * N > opts.algorithmSwitchThreshold!;
  
  if (useTiling) {
    return generateTiledMatmulShader(workgroupSizeX, workgroupSizeY, opts);
  } else {
    return generateSimpleMatmulShader(workgroupSizeX, workgroupSizeY, opts);
  }
}

/**
 * Generate simple matrix multiplication shader
 * @param workgroupSizeX X dimension of workgroup
 * @param workgroupSizeY Y dimension of workgroup
 * @param options Shader options
 * @returns WGSL shader code
 */
function generateSimpleMatmulShader(
  workgroupSizeX: number,
  workgroupSizeY: number,
  options: SpecializedShaderOptions
): string {
  return /* wgsl */`
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

@compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY})
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
}

/**
 * Generate tiled matrix multiplication shader for better performance
 * @param workgroupSizeX X dimension of workgroup
 * @param workgroupSizeY Y dimension of workgroup
 * @param options Shader options
 * @returns WGSL shader code
 */
function generateTiledMatmulShader(
  workgroupSizeX: number,
  workgroupSizeY: number,
  options: SpecializedShaderOptions
): string {
  // Define tile size for shared memory
  const TILE_SIZE = 16; // Can be tuned based on hardware characteristics
  
  return /* wgsl */`
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

// Shared memory for tiled multiplication
var<workgroup> tile_a: array<f32, ${TILE_SIZE * TILE_SIZE}>;
var<workgroup> tile_b: array<f32, ${TILE_SIZE * TILE_SIZE}>;

@compute @workgroup_size(${TILE_SIZE}, ${TILE_SIZE})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let row = workgroup_id.x * ${TILE_SIZE} + local_id.x;
  let col = workgroup_id.y * ${TILE_SIZE} + local_id.y;
  
  // Initialize accumulator
  var sum: f32 = 0.0;
  
  // Loop over tiles
  let num_tiles = (dimensions.K + ${TILE_SIZE} - 1) / ${TILE_SIZE};
  
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    // Load tile from matrix A
    let a_row = row;
    let a_col = t * ${TILE_SIZE} + local_id.y;
    
    if (a_row < dimensions.M && a_col < dimensions.K) {
      tile_a[local_id.x * ${TILE_SIZE} + local_id.y] = matrix_a[a_row * dimensions.K + a_col];
    } else {
      tile_a[local_id.x * ${TILE_SIZE} + local_id.y] = 0.0;
    }
    
    // Load tile from matrix B
    let b_row = t * ${TILE_SIZE} + local_id.x;
    let b_col = col;
    
    if (b_row < dimensions.K && b_col < dimensions.N) {
      tile_b[local_id.x * ${TILE_SIZE} + local_id.y] = matrix_b[b_row * dimensions.N + b_col];
    } else {
      tile_b[local_id.x * ${TILE_SIZE} + local_id.y] = 0.0;
    }
    
    // Synchronize to ensure tiles are loaded
    workgroupBarrier();
    
    // Perform tile multiplication and accumulation
    for (var k: u32 = 0u; k < ${TILE_SIZE}; k = k + 1u) {
      sum = sum + tile_a[local_id.x * ${TILE_SIZE} + k] * tile_b[k * ${TILE_SIZE} + local_id.y];
    }
    
    // Synchronize before loading next tiles
    workgroupBarrier();
  }
  
  // Write the result
  if (row < dimensions.M && col < dimensions.N) {
    matrix_c[row * dimensions.N + col] = sum;
  }
}`;
}

/**
 * Get specialized elementwise operation shader
 * @param operation Operation type ('add', 'subtract', 'multiply', 'divide')
 * @param options Specialization options
 * @returns WGSL shader code
 */
export function getSpecializedElementwiseShader(
  operation: 'add' | 'subtract' | 'multiply' | 'divide',
  options: SpecializedShaderOptions = {}
): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // Select appropriate workgroup size
  let workgroupSize = opts.workgroupSize!;
  
  // Firefox tends to do better with smaller workgroups
  if (opts.browserOptimized === 'firefox') {
    workgroupSize = 128;
  }
  
  // Map operation to WGSL operator
  const operatorMap: {[key: string]: string} = {
    'add': '+',
    'subtract': '-',
    'multiply': '*',
    'divide': '/'
  };
  
  const operator = operatorMap[operation];
  
  // Special case for division to handle division by zero
  const resultComputation = operation === 'divide' 
    ? `(abs(input_b[idx]) < 1e-7) ? 0.0 : (input_a[idx] ${operator} input_b[idx])`
    : `input_a[idx] ${operator} input_b[idx]`;
  
  return /* wgsl */`
// Binding group layout:
// binding 0: Input buffer A
// binding 1: Input buffer B
// binding 2: Output buffer
// binding 3: Uniform params

struct Params {
  length: u32,
};

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Bounds check
  if (idx >= params.length) {
    return;
  }
  
  // Compute result
  let result = ${resultComputation};
  
  // Write the result
  output[idx] = result;
}`;
}

/**
 * Get specialized activation function shader
 * @param activationType Activation function type ('relu', 'sigmoid', 'tanh', 'softmax')
 * @param options Specialization options
 * @returns WGSL shader code
 */
export function getSpecializedActivationShader(
  activationType: 'relu' | 'sigmoid' | 'tanh' | 'softmax',
  options: SpecializedShaderOptions = {}
): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // Select appropriate workgroup size
  let workgroupSize = opts.workgroupSize!;
  
  // Softmax is a special case requiring different shader structure
  if (activationType === 'softmax') {
    return getSpecializedSoftmaxShader(opts);
  }
  
  // Helper functions
  const activationFunctions = `
// Helper function for sigmoid
fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

// Helper function for tanh
fn tanh_approx(x: f32) -> f32 {
  ${opts.useFastMath 
    ? '// Fast approximation using padé approximant\n  let x2 = x * x;\n  return x * (27.0 + x2) / (27.0 + 9.0 * x2);' 
    : '// Standard implementation\n  let s = sigmoid(2.0 * x);\n  return 2.0 * s - 1.0;'}
}
`;
  
  // Map activation type to computation expression
  let activationExpression: string;
  switch (activationType) {
    case 'relu':
      activationExpression = 'max(0.0, input[idx])';
      break;
    case 'sigmoid':
      activationExpression = 'sigmoid(input[idx])';
      break;
    case 'tanh':
      activationExpression = 'tanh_approx(input[idx])';
      break;
    default:
      activationExpression = 'input[idx]';
  }
  
  return /* wgsl */`
// Binding group layout:
// binding 0: Input buffer
// binding 1: Output buffer
// binding 2: Uniform params

struct Params {
  length: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

${activationFunctions}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Bounds check
  if (idx >= params.length) {
    return;
  }
  
  // Apply activation function
  let result = ${activationExpression};
  
  // Write the result
  output[idx] = result;
}`;
}

/**
 * Get specialized softmax shader
 * @param options Specialization options
 * @returns WGSL shader code
 */
function getSpecializedSoftmaxShader(options: SpecializedShaderOptions): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // Softmax requires a different approach with reduction operations
  return /* wgsl */`
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
}

/**
 * Get specialized transpose shader
 * @param options Specialization options
 * @returns WGSL shader code
 */
export function getSpecializedTransposeShader(options: SpecializedShaderOptions = {}): string {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // Select appropriate workgroup sizes
  let workgroupSizeX = 16;
  let workgroupSizeY = 16;
  
  // Browser-specific optimizations
  if (opts.browserOptimized === 'firefox') {
    workgroupSizeX = 8;
    workgroupSizeY = 8;
  }
  
  return /* wgsl */`
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

@compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY})
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
}

/**
 * Generate browser-optimized shader for a specific operation
 * @param operation Operation type
 * @param browserType Browser to optimize for
 * @returns WGSL shader code
 */
export function getBrowserOptimizedShader(
  operation: 'matmul' | 'add' | 'subtract' | 'multiply' | 'divide' | 'relu' | 'sigmoid' | 'tanh' | 'softmax' | 'transpose',
  browserType: 'chrome' | 'firefox' | 'safari' | 'edge'
): string {
  // Create options specific to this browser
  const options: SpecializedShaderOptions = {
    browserOptimized: browserType
  };
  
  // Set browser-specific options
  switch (browserType) {
    case 'chrome':
      // Chrome generally does well with larger workgroups
      options.workgroupSize = 256;
      options.useFastMath = true;
      break;
    
    case 'firefox':
      // Firefox tends to do better with smaller workgroups and more threads
      options.workgroupSize = 128;
      options.useFastMath = true;
      break;
    
    case 'safari':
      // Safari prefers larger workgroups for better GPU utilization on Apple hardware
      options.workgroupSize = 512;
      // Use high precision math on Apple GPUs which handle it well
      options.useFastMath = false;
      break;
    
    case 'edge':
      // Similar to Chrome, but with some adjustments for DirectX
      options.workgroupSize = 256;
      options.useFastMath = true;
      break;
  }
  
  // Generate the appropriate shader
  switch (operation) {
    case 'matmul':
      // For matrix multiplication, we provide a placeholder size
      // In practice, the actual dimensions would determine the optimal shader
      return getSpecializedMatmulShader(1024, 1024, 1024, options);
    
    case 'add':
    case 'subtract':
    case 'multiply':
    case 'divide':
      return getSpecializedElementwiseShader(operation, options);
    
    case 'relu':
    case 'sigmoid':
    case 'tanh':
    case 'softmax':
      return getSpecializedActivationShader(operation, options);
    
    case 'transpose':
      return getSpecializedTransposeShader(options);
    
    default:
      throw new Error(`Unsupported operation for browser optimization: ${operation}`);
  }
}