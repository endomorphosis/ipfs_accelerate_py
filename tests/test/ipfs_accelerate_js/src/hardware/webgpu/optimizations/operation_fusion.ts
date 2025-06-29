/**
 * WebGPU Operation Fusion
 * Optimizes performance by combining multiple operations into a single shader
 */

import { Tensor } from '../../../tensor/tensor';
import { WebGPUBackend, WebGPUOperationType } from '../backend';

/**
 * Defines a fused operation sequence
 */
export interface FusedOperationSequence {
  /** Unique identifier for this fusion pattern */
  id: string;
  
  /** Names of operations in the sequence */
  operations: string[];
  
  /** Whether the sequence is valid for fusion */
  valid: boolean;
}

/**
 * Operation fusion patterns supported by the WebGPU backend
 */
export enum FusionPattern {
  /** Linear + Activation (e.g., MatMul + ReLU) */
  LinearActivation = 'linear_activation',
  
  /** Element-wise operations chain (e.g., Add + Multiply + Add) */
  ElementWiseChain = 'elementwise_chain',
  
  /** Element-wise binary + Unary (e.g., Add + ReLU) */
  BinaryUnary = 'binary_unary',
  
  /** Reshape + Operation (e.g., Reshape + Softmax) */
  ReshapeOp = 'reshape_op',
  
  /** Multiple activations in sequence (e.g., ReLU + Dropout) */
  ActivationChain = 'activation_chain',
  
  /** Normalization + Activation (e.g., LayerNorm + GELU) */
  NormActivation = 'norm_activation',
  
  /** Attention pattern for transformer models (MatMul + Scale + Softmax + MatMul) */
  AttentionPattern = 'attention_pattern',
  
  /** Matrix operation chain (matmul + matmul) for faster multi-layer execution */
  MatrixChain = 'matrix_chain',
  
  /** Quantized matrix multiplication for memory efficiency */
  QuantizedMatmul = 'quantized_matmul',
  
  /** Quantized matrix multiplication with activation function */
  QuantizedMatmulActivation = 'quantized_matmul_activation',
  
  /** Quantized attention mechanism for transformer models */
  QuantizedAttention = 'quantized_attention',
  
  /** Custom defined sequence */
  Custom = 'custom'
}

/**
 * Supported fusion operation types
 */
export type FusionOpType = 
  // Basic arithmetic operations
  | 'add' 
  | 'subtract' 
  | 'multiply' 
  | 'divide'
  | 'pow'
  | 'max'
  | 'min'
  
  // Matrix operations
  | 'matmul' 
  | 'transpose'
  | 'dot' 
  
  // Shape operations
  | 'reshape'
  | 'flatten'
  | 'expand_dims'
  | 'squeeze'
  
  // Activation functions
  | 'relu' 
  | 'sigmoid' 
  | 'tanh' 
  | 'softmax'
  | 'gelu'
  | 'silu' // SiLU/Swish activation (x * sigmoid(x))
  | 'leaky_relu'
  | 'elu'

  // Normalization operations
  | 'layer_norm'
  | 'batch_norm'
  
  // Dropout (for training)
  | 'dropout'
  
  // Pooling operations
  | 'max_pool'
  | 'avg_pool'
  
  // Advanced operations for attention mechanisms
  | 'scale' // Scaling operation for attention
  | 'mask' // Masking operation for attention
  | 'softmax_with_mask'; // Masked softmax for attention

/**
 * Configuration for operation fusion
 */
export interface FusionConfig {
  /** Maximum operations to fuse in a chain */
  maxFusionLength?: number;
  
  /** Whether to enable auto-fusion detection */
  enableAutoFusion?: boolean;
  
  /** Specific patterns to enable/disable */
  enabledPatterns?: FusionPattern[];
  
  /** Custom fusion patterns */
  customPatterns?: {[key: string]: FusionOpType[]};
  
  /** Whether to use quantized weights for matrix operations */
  useQuantizedWeights?: boolean;
  
  /** Bits per weight for quantized operations */
  bitsPerWeight?: 1 | 2 | 3 | 4 | 8;
  
  /** Whether to use browser-specific optimizations */
  useBrowserOptimizations?: boolean;
  
  /** Type of browser for optimizations (detected automatically if useBrowserOptimizations is true) */
  browserOptimizationType?: 'firefox' | 'chrome' | 'safari' | 'edge' | 'default';
}

/**
 * Default fusion configuration
 */
const DEFAULT_FUSION_CONFIG: FusionConfig = {
  maxFusionLength: 10,
  enableAutoFusion: true,
  enabledPatterns: [
    FusionPattern.LinearActivation,
    FusionPattern.ElementWiseChain,
    FusionPattern.BinaryUnary,
    FusionPattern.ReshapeOp,
    FusionPattern.ActivationChain,
    FusionPattern.NormActivation,
    FusionPattern.AttentionPattern,
    FusionPattern.MatrixChain
  ],
  useQuantizedWeights: false,
  bitsPerWeight: 4,
  useBrowserOptimizations: true
};

/**
 * Utility for fusing operations in WebGPU
 */
export class WebGPUOperationFusion {
  /** Backend reference */
  private backend: WebGPUBackend;
  
  /** Fusion configuration */
  private config: FusionConfig;
  
  /** Cache of compiled fusion shaders */
  private shaderCache: Map<string, GPUComputePipeline> = new Map();
  
  /** Cache of bind group layouts */
  private bindGroupLayoutCache: Map<string, GPUBindGroupLayout> = new Map();
  
  /**
   * Constructor
   * @param backend WebGPU backend instance
   * @param config Fusion configuration
   */
  constructor(backend: WebGPUBackend, config: FusionConfig = DEFAULT_FUSION_CONFIG) {
    this.backend = backend;
    this.config = {
      ...DEFAULT_FUSION_CONFIG,
      ...config
    };
    
    // Enable browser-specific optimizations by detecting the browser type
    if (this.config.useBrowserOptimizations) {
      this.detectAndSetBrowserOptimizations();
    }
  }
  
  /**
   * Check if a sequence of operations can be fused
   * @param operations Operation types to check
   * @returns Whether the operations can be fused
   */
  canFuse(operations: FusionOpType[]): boolean {
    // Check fusion length limit
    if (operations.length > this.config.maxFusionLength!) {
      return false;
    }
    
    // Check for known fusion patterns
    if (this.matchesPattern(operations)) {
      return true;
    }
    
    // Check for custom patterns
    if (this.matchesCustomPattern(operations)) {
      return true;
    }
    
    // Check for quantized matrix operations
    if (this.config.useQuantizedWeights && 
        operations.length >= 1 && 
        operations[0] === 'matmul') {
      return true;
    }
    
    // For auto-fusion, check compatibility
    if (this.config.enableAutoFusion) {
      return this.areCompatibleForFusion(operations);
    }
    
    return false;
  }
  
  /**
   * Check if operations match a predefined pattern
   * @param operations Operations to check
   * @returns Whether the operations match a pattern
   */
  private matchesPattern(operations: FusionOpType[]): boolean {
    const enabledPatterns = this.config.enabledPatterns || [];
    
    // Linear + Activation pattern
    if (enabledPatterns.includes(FusionPattern.LinearActivation)) {
      if (operations.length === 2 && 
          (operations[0] === 'matmul') && 
          this.isActivationOp(operations[1])) {
        return true;
      }
    }
    
    // Element-wise chain pattern
    if (enabledPatterns.includes(FusionPattern.ElementWiseChain)) {
      const allElementWise = operations.every(op => this.isElementWiseOp(op));
      
      if (allElementWise) {
        return true;
      }
    }
    
    // Binary + Unary pattern
    if (enabledPatterns.includes(FusionPattern.BinaryUnary)) {
      if (operations.length === 2 && 
          this.isElementWiseOp(operations[0]) &&
          this.isActivationOp(operations[1])) {
        return true;
      }
    }
    
    // Reshape + Op pattern
    if (enabledPatterns.includes(FusionPattern.ReshapeOp)) {
      if (operations.length === 2 && 
          this.isShapeOp(operations[0]) &&
          !this.isShapeOp(operations[1])) {
        return true;
      }
    }
    
    // Activation Chain pattern
    if (enabledPatterns.includes(FusionPattern.ActivationChain)) {
      const allActivation = operations.every(op => this.isActivationOp(op) || op === 'dropout');
      
      if (allActivation && operations.length >= 2) {
        return true;
      }
    }
    
    // Normalization + Activation pattern
    if (enabledPatterns.includes(FusionPattern.NormActivation)) {
      if (operations.length === 2 && 
          this.isNormalizationOp(operations[0]) &&
          this.isActivationOp(operations[1])) {
        return true;
      }
    }
    
    // Attention pattern (for transformer models)
    if (enabledPatterns.includes(FusionPattern.AttentionPattern)) {
      // MatMul + Scale + Softmax + MatMul pattern for self-attention
      if (operations.length >= 3 && 
          operations[0] === 'matmul' &&
          (operations[1] === 'scale' || operations[1] === 'multiply') &&
          (operations[2] === 'softmax' || operations[2] === 'softmax_with_mask')) {
        return true;
      }
    }
    
    // Matrix Chain pattern (matmul + matmul)
    if (enabledPatterns.includes(FusionPattern.MatrixChain)) {
      // Multiple matrix multiplications in sequence
      if (operations.length >= 2 && 
          operations[0] === 'matmul' && 
          operations[1] === 'matmul') {
        return true;
      }
    }
    
    return false;
  }
  
  /**
   * Check if an operation is an element-wise operation
   * @param op Operation to check
   * @returns Whether it's an element-wise operation
   */
  private isElementWiseOp(op: FusionOpType): boolean {
    return ['add', 'subtract', 'multiply', 'divide', 'pow', 'max', 'min'].includes(op);
  }
  
  /**
   * Check if an operation is an activation function
   * @param op Operation to check
   * @returns Whether it's an activation function
   */
  private isActivationOp(op: FusionOpType): boolean {
    return ['relu', 'sigmoid', 'tanh', 'gelu', 'silu', 'leaky_relu', 'elu', 'softmax'].includes(op);
  }
  
  /**
   * Check if an operation is a shape operation
   * @param op Operation to check
   * @returns Whether it's a shape operation
   */
  private isShapeOp(op: FusionOpType): boolean {
    return ['reshape', 'transpose', 'flatten', 'expand_dims', 'squeeze'].includes(op);
  }
  
  /**
   * Check if an operation is a normalization operation
   * @param op Operation to check
   * @returns Whether it's a normalization operation
   */
  private isNormalizationOp(op: FusionOpType): boolean {
    return ['layer_norm', 'batch_norm'].includes(op);
  }
  
  /**
   * Check if operations match a custom pattern
   * @param operations Operations to check
   * @returns Whether the operations match a custom pattern
   */
  private matchesCustomPattern(operations: FusionOpType[]): boolean {
    const customPatterns = this.config.customPatterns || {};
    
    for (const patternName of Object.keys(customPatterns)) {
      const pattern = customPatterns[patternName];
      
      if (operations.length !== pattern.length) {
        continue;
      }
      
      let matches = true;
      for (let i = 0; i < pattern.length; i++) {
        if (operations[i] !== pattern[i]) {
          matches = false;
          break;
        }
      }
      
      if (matches) {
        return true;
      }
    }
    
    return false;
  }
  
  /**
   * Check if operations are compatible for auto-fusion
   * @param operations Operations to check
   * @returns Whether the operations are compatible
   */
  private areCompatibleForFusion(operations: FusionOpType[]): boolean {
    // For now, we only support auto-fusion for element-wise operations
    const elementWiseOps = ['add', 'subtract', 'multiply', 'divide', 'relu', 'sigmoid', 'tanh'];
    const allElementWise = operations.every(op => elementWiseOps.includes(op));
    
    if (allElementWise) {
      return true;
    }
    
    return false;
  }
  
  /**
   * Generate a unique ID for a fusion sequence
   * @param operations Operations in the sequence
   * @returns Unique ID
   */
  generateFusionId(operations: FusionOpType[]): string {
    return `fusion_${operations.join('_')}`;
  }
  
  /**
   * Generate a WGSL shader for a fusion sequence
   * @param operations Operations to fuse
   * @returns WGSL shader code
   */
  generateFusionShader(operations: FusionOpType[]): string {
    // Check if this is a special pattern that needs custom handling
    if (this.isLinearActivation(operations)) {
      // Check if we should use quantized version for better memory efficiency
      if (this.config.useQuantizedWeights && operations[0] === 'matmul') {
        return this.generateQuantizedMatmulActivationShader(operations[1]);
      }
      return this.generateLinearActivationShader(operations[1]);
    }
    
    if (this.isElementWiseChain(operations)) {
      return this.generateElementWiseChainShader(operations);
    }
    
    if (this.isBinaryUnary(operations)) {
      return this.generateBinaryUnaryShader(
        operations[0],
        operations[1]
      );
    }
    
    // Check for activation chain pattern
    if (this.isActivationChain(operations)) {
      return this.generateActivationChainShader(operations);
    }
    
    // Check for attention pattern
    if (this.isAttentionPattern(operations)) {
      // Check if we should use quantized version for better memory efficiency
      if (this.config.useQuantizedWeights) {
        return this.generateQuantizedAttentionShader(operations);
      }
      return this.generateAttentionPatternShader(operations);
    }
    
    // Check for matrix chain pattern
    if (this.isMatrixChain(operations)) {
      // Check if we should use quantized version
      if (this.config.useQuantizedWeights) {
        return this.generateQuantizedMatrixChainShader(operations);
      }
      return this.generateMatrixChainShader(operations);
    }
    
    // Check for normalization + activation pattern
    if (this.isNormActivation(operations)) {
      return this.generateNormActivationShader(operations);
    }
    
    // Check for standalone matrix multiplication that could use quantization
    if (this.config.useQuantizedWeights && operations.length >= 1 && operations[0] === 'matmul') {
      return this.generateQuantizedMatmulShader();
    }
    
    // Default to element-wise fusion shader (generic)
    return this.generateGenericFusionShader(operations);
  }
  
  /**
   * Check if this is an ActivationChain fusion
   * @param operations Operations to check
   * @returns Whether it's an ActivationChain fusion
   */
  private isActivationChain(operations: FusionOpType[]): boolean {
    return operations.length >= 2 && 
           operations.every(op => this.isActivationOp(op) || op === 'dropout');
  }
  
  /**
   * Check if this is an AttentionPattern fusion
   * @param operations Operations to check
   * @returns Whether it's an AttentionPattern fusion
   */
  private isAttentionPattern(operations: FusionOpType[]): boolean {
    return operations.length >= 3 && 
           operations[0] === 'matmul' &&
           (operations[1] === 'scale' || operations[1] === 'multiply') &&
           (operations[2] === 'softmax' || operations[2] === 'softmax_with_mask');
  }
  
  /**
   * Check if this is a MatrixChain fusion
   * @param operations Operations to check
   * @returns Whether it's a MatrixChain fusion
   */
  private isMatrixChain(operations: FusionOpType[]): boolean {
    return operations.length >= 2 && 
           operations[0] === 'matmul' && 
           operations[1] === 'matmul';
  }
  
  /**
   * Check if this is a NormActivation fusion
   * @param operations Operations to check
   * @returns Whether it's a NormActivation fusion
   */
  private isNormActivation(operations: FusionOpType[]): boolean {
    return operations.length === 2 && 
           this.isNormalizationOp(operations[0]) &&
           this.isActivationOp(operations[1]);
  }
  
  /**
   * Check if this is a LinearActivation fusion
   * @param operations Operations to check
   * @returns Whether it's a LinearActivation fusion
   */
  private isLinearActivation(operations: FusionOpType[]): boolean {
    return operations.length === 2 && 
           operations[0] === 'matmul' && 
           ['relu', 'sigmoid', 'tanh'].includes(operations[1]);
  }
  
  /**
   * Check if this is an ElementWiseChain fusion
   * @param operations Operations to check
   * @returns Whether it's an ElementWiseChain fusion
   */
  private isElementWiseChain(operations: FusionOpType[]): boolean {
    return operations.every(op => ['add', 'subtract', 'multiply', 'divide'].includes(op));
  }
  
  /**
   * Check if this is a BinaryUnary fusion
   * @param operations Operations to check
   * @returns Whether it's a BinaryUnary fusion
   */
  private isBinaryUnary(operations: FusionOpType[]): boolean {
    return operations.length === 2 && 
           ['add', 'subtract', 'multiply', 'divide'].includes(operations[0]) &&
           ['relu', 'sigmoid', 'tanh'].includes(operations[1]);
  }
  
  /**
   * Generate a MatMul+Activation fusion shader
   * @param activationType Activation function type
   * @returns WGSL shader code
   */
  private generateLinearActivationShader(activationType: FusionOpType): string {
    // Helper functions for activations
    const activationFunctions = `
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

// Helper function for GELU (Gaussian Error Linear Unit)
fn gelu(x: f32) -> f32 {
  // Approximation of GELU: x * 0.5 * (1.0 + tanh(sqrt(2.0/pi) * (x + 0.044715 * x^3)))
  let sqrt2overpi = 0.7978845608028654;
  let coeff = 0.044715;
  let x3 = x * x * x;
  return x * 0.5 * (1.0 + tanh_approx(sqrt2overpi * (x + coeff * x3)));
}

// Helper function for SiLU / Swish: x * sigmoid(x)
fn silu(x: f32) -> f32 {
  return x * sigmoid(x);
}

// Helper function for Leaky ReLU
fn leaky_relu(x: f32) -> f32 {
  return x < 0.0 ? 0.01 * x : x;
}

// Helper function for ELU (Exponential Linear Unit)
fn elu(x: f32) -> f32 {
  return x < 0.0 ? exp(x) - 1.0 : x;
}
`;
    
    // Determine the activation function to use
    let activationCode: string;
    switch (activationType) {
      case 'relu':
        activationCode = 'max(0.0, sum)';
        break;
      case 'sigmoid':
        activationCode = 'sigmoid(sum)';
        break;
      case 'tanh':
        activationCode = 'tanh_approx(sum)';
        break;
      case 'gelu':
        activationCode = 'gelu(sum)';
        break;
      case 'silu':
        activationCode = 'silu(sum)';
        break;
      case 'leaky_relu':
        activationCode = 'leaky_relu(sum)';
        break;
      case 'elu':
        activationCode = 'elu(sum)';
        break;
      case 'softmax':
        // For softmax after matmul, we'd need special handling
        // This is simplified here
        activationCode = 'exp(sum)'; // Just the exp part, normalization would be done separately
        break;
      default:
        activationCode = 'sum'; // No activation (identity)
    }
    
    // Generate the shader
    return /* wgsl */`
// Binding group layout:
// binding 0: Input matrix A
// binding 1: Input matrix B
// binding 2: Output matrix
// binding 3: Uniform buffer with matrix dimensions

struct Dimensions {
  M: u32,  // Rows in A
  K: u32,  // Cols in A / Rows in B
  N: u32,  // Cols in B
};

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: Dimensions;

${activationFunctions}

// Fused matrix multiplication with activation
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
  
  // Apply activation function
  let activated = ${activationCode};
  
  // Write the result
  let c_index = row * dimensions.N + col;
  matrix_c[c_index] = activated;
}`;
  }
  
  /**
   * Generate an element-wise chain fusion shader
   * @param operations Element-wise operations
   * @returns WGSL shader code
   */
  private generateElementWiseChainShader(
    operations: Array<'add' | 'subtract' | 'multiply' | 'divide'>
  ): string {
    // Convert operations to WGSL code snippets
    const opMappings: {[key: string]: string} = {
      'add': '+',
      'subtract': '-',
      'multiply': '*',
      'divide': '/'
    };
    
    // Number of input buffers needed (operations + 1)
    const numInputs = operations.length + 1;
    
    // Generate input bindings
    let inputBindings = '';
    for (let i = 0; i < numInputs; i++) {
      inputBindings += `@group(0) @binding(${i}) var<storage, read> input_${i}: array<f32>;\n`;
    }
    
    // Generate computation expression
    let computation = `input_0[idx]`;
    for (let i = 0; i < operations.length; i++) {
      const op = opMappings[operations[i]];
      computation = `(${computation} ${op} input_${i+1}[idx])`;
    }
    
    // Generate the shader
    return /* wgsl */`
// Binding group layout:
// binding 0..N-1: Input buffers
// binding N: Output buffer
// binding N+1: Uniform params

struct Params {
  length: u32,
};

${inputBindings}
@group(0) @binding(${numInputs}) var<storage, read_write> output: array<f32>;
@group(0) @binding(${numInputs + 1}) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Bounds check
  if (idx >= params.length) {
    return;
  }
  
  // Compute fused element-wise operations
  let result = ${computation};
  
  // Write the result
  output[idx] = result;
}`;
  }
  
  /**
   * Generate a binary+unary fusion shader
   * @param binaryOp Binary operation type
   * @param unaryOp Unary operation type
   * @returns WGSL shader code
   */
  private generateBinaryUnaryShader(
    binaryOp: FusionOpType,
    unaryOp: FusionOpType
  ): string {
    // Map binary operation to WGSL operator
    const binaryOpMappings: {[key: string]: string} = {
      'add': '+',
      'subtract': '-',
      'multiply': '*',
      'divide': '/',
      'pow': '**',
      'max': 'max',
      'min': 'min'
    };
    
    // Helper functions for activations
    const activationFunctions = `
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

// Helper function for GELU
fn gelu(x: f32) -> f32 {
  // Approximation of GELU
  let sqrt2overpi = 0.7978845608028654;
  let coeff = 0.044715;
  let x3 = x * x * x;
  return x * 0.5 * (1.0 + tanh_approx(sqrt2overpi * (x + coeff * x3)));
}

// Helper function for SiLU / Swish
fn silu(x: f32) -> f32 {
  return x * sigmoid(x);
}

// Helper function for Leaky ReLU
fn leaky_relu(x: f32) -> f32 {
  return x < 0.0 ? 0.01 * x : x;
}

// Helper function for ELU
fn elu(x: f32) -> f32 {
  return x < 0.0 ? exp(x) - 1.0 : x;
}
`;
    
    // Build the binary operation expression
    let binaryExpression: string;
    if (['max', 'min'].includes(binaryOp)) {
      binaryExpression = `${binaryOp}(input_a[idx], input_b[idx])`;
    } else if (binaryOp === 'pow') {
      binaryExpression = `pow(input_a[idx], input_b[idx])`;
    } else {
      binaryExpression = `input_a[idx] ${binaryOpMappings[binaryOp]} input_b[idx]`;
    }
    
    // Determine the activation function to use
    let activationCode: string;
    switch (unaryOp) {
      case 'relu':
        activationCode = 'max(0.0, binary_result)';
        break;
      case 'sigmoid':
        activationCode = 'sigmoid(binary_result)';
        break;
      case 'tanh':
        activationCode = 'tanh_approx(binary_result)';
        break;
      case 'gelu':
        activationCode = 'gelu(binary_result)';
        break;
      case 'silu':
        activationCode = 'silu(binary_result)';
        break;
      case 'leaky_relu':
        activationCode = 'leaky_relu(binary_result)';
        break;
      case 'elu':
        activationCode = 'elu(binary_result)';
        break;
      default:
        activationCode = 'binary_result'; // No activation (identity)
    }
    
    // Generate the shader
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

${activationFunctions}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Bounds check
  if (idx >= params.length) {
    return;
  }
  
  // Compute binary operation
  let binary_result = ${binaryExpression};
  
  // Apply activation function
  let result = ${activationCode};
  
  // Write the result
  output[idx] = result;
}`;
  }
  
  /**
   * Generate shader for a chain of activation functions
   * @param operations Operations to fuse
   * @returns WGSL shader code
   */
  private generateActivationChainShader(operations: FusionOpType[]): string {
    // Include helper functions for all activation functions
    const activationFunctions = `
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

// Helper function for GELU
fn gelu(x: f32) -> f32 {
  // Approximation of GELU
  let sqrt2overpi = 0.7978845608028654;
  let coeff = 0.044715;
  let x3 = x * x * x;
  return x * 0.5 * (1.0 + tanh_approx(sqrt2overpi * (x + coeff * x3)));
}

// Helper function for SiLU / Swish
fn silu(x: f32) -> f32 {
  return x * sigmoid(x);
}

// Helper function for Leaky ReLU
fn leaky_relu(x: f32) -> f32 {
  return x < 0.0 ? 0.01 * x : x;
}

// Helper function for ELU
fn elu(x: f32) -> f32 {
  return x < 0.0 ? exp(x) - 1.0 : x;
}

// Helper function for ReLU
fn relu(x: f32) -> f32 {
  return max(0.0, x);
}

// Helper function for dropout (training mode only)
fn dropout(x: f32, rate: f32, mask: f32) -> f32 {
  // In inference mode, we just scale the output
  return x * (1.0 - rate);
}
`;
    
    // Generate computation steps for each activation
    let computationSteps = '';
    let resultVar = 'input[idx]';
    
    for (let i = 0; i < operations.length; i++) {
      const op = operations[i];
      const nextVar = `temp_${i+1}`;
      
      switch (op) {
        case 'relu':
          computationSteps += `  let ${nextVar} = relu(${resultVar});\n`;
          break;
        case 'sigmoid':
          computationSteps += `  let ${nextVar} = sigmoid(${resultVar});\n`;
          break;
        case 'tanh':
          computationSteps += `  let ${nextVar} = tanh_approx(${resultVar});\n`;
          break;
        case 'gelu':
          computationSteps += `  let ${nextVar} = gelu(${resultVar});\n`;
          break;
        case 'silu': 
          computationSteps += `  let ${nextVar} = silu(${resultVar});\n`;
          break;
        case 'leaky_relu':
          computationSteps += `  let ${nextVar} = leaky_relu(${resultVar});\n`;
          break;
        case 'elu':
          computationSteps += `  let ${nextVar} = elu(${resultVar});\n`;
          break;
        case 'dropout':
          // For inference only - just a scale operation
          computationSteps += `  let ${nextVar} = dropout(${resultVar}, 0.1, 1.0);\n`;
          break;
        case 'softmax':
          // This is simplified, as softmax usually requires multiple passes
          computationSteps += `  let ${nextVar} = exp(${resultVar});\n`;
          break;
        default:
          computationSteps += `  let ${nextVar} = ${resultVar};\n`;
      }
      
      resultVar = nextVar;
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Bounds check
  if (idx >= params.length) {
    return;
  }
  
  // Apply activation chain
${computationSteps}
  
  // Write the result
  output[idx] = ${resultVar};
}`;
  }
  
  /**
   * Generate shader for matrix chain operations (matmul sequence)
   * @param operations Matrix operations to fuse
   * @returns WGSL shader code
   */
  private generateMatrixChainShader(operations: FusionOpType[]): string {
    // For a sequence of matrix multiplications, we can optimize by computing the entire chain
    // in a single pass, avoiding intermediate result materialization
    
    // This is a simplified implementation - a real implementation would use tiling
    // and other optimizations for better performance
    
    return /* wgsl */`
// Binding group layout:
// binding 0: Input matrix A
// binding 1: Input matrix B
// binding 2: Input matrix C (if needed for second matmul)
// binding 3: Output matrix
// binding 4: Uniform buffer with matrix dimensions

struct Dimensions {
  M: u32,  // Rows in A
  K: u32,  // Cols in A / Rows in B
  N: u32,  // Cols in B
  P: u32,  // Cols in C (for second matmul)
};

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read> matrix_c: array<f32>;
@group(0) @binding(3) var<storage, read_write> matrix_out: array<f32>;
@group(0) @binding(4) var<uniform> dimensions: Dimensions;

// Fused matrix multiplication for A*B*C
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  
  // Bounds check for final output dimensions
  if (row >= dimensions.M || col >= dimensions.P) {
    return;
  }
  
  var sum: f32 = 0.0;
  
  // For matrix chain A * B * C, we compute (A * B) * C
  // But we don't materialize the intermediate result A * B
  for (var k: u32 = 0u; k < dimensions.K; k = k + 1u) {
    // For each element in row of A
    let a_idx = row * dimensions.K + k;
    let a_val = matrix_a[a_idx];
    
    // For each element in column of C (iterating through B*C)
    for (var j: u32 = 0u; j < dimensions.N; j = j + 1u) {
      // Get element from B
      let b_idx = k * dimensions.N + j;
      let b_val = matrix_b[b_idx];
      
      // Get element from C
      let c_idx = j * dimensions.P + col;
      let c_val = matrix_c[c_idx];
      
      // Accumulate (A*B)*C contribution
      sum = sum + a_val * b_val * c_val;
    }
  }
  
  // Write the result
  let out_idx = row * dimensions.P + col;
  matrix_out[out_idx] = sum;
}`;
  }
  
  /**
   * Generate shader for attention pattern fusion (for transformer models)
   * @param operations Attention operations to fuse
   * @returns WGSL shader code
   */
  private generateAttentionPatternShader(operations: FusionOpType[]): string {
    // Self-attention pattern: MatMul + Scale + Softmax
    // This is a simplified implementation focusing on fusion pattern
    
    return /* wgsl */`
// Binding group layout:
// binding 0: Query matrix
// binding 1: Key matrix (transposed)
// binding 2: Value matrix
// binding 3: Output matrix
// binding 4: Uniform buffer with dimensions and scale

struct Dimensions {
  batch_size: u32,
  seq_len: u32,
  num_heads: u32,
  head_dim: u32,
  scale: f32,
};

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key_t: array<f32>; // Transposed key
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> dimensions: Dimensions;

// Workgroup shared memory for softmax normalization
var<workgroup> max_values: array<f32, 64>; // Assuming max sequence length per workgroup
var<workgroup> sum_values: array<f32, 64>; // Assuming max sequence length per workgroup

// Self-attention fusion: Q*K^T*scale -> Softmax -> *V
@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let batch_idx = workgroup_id.z;
  let head_idx = workgroup_id.y;
  
  let row = global_id.x; // Sequence position in query
  let col = global_id.y; // Sequence position in value
  
  // First phase: compute attention scores (Q * K^T) * scale
  if (row < dimensions.seq_len && col < dimensions.seq_len) {
    // Compute attention score
    var score: f32 = 0.0;
    
    // Compute dot product of query and key vectors
    for (var i: u32 = 0u; i < dimensions.head_dim; i = i + 1u) {
      let q_idx = ((batch_idx * dimensions.num_heads + head_idx) * dimensions.seq_len + row) * dimensions.head_dim + i;
      let k_idx = ((batch_idx * dimensions.num_heads + head_idx) * dimensions.seq_len + col) * dimensions.head_dim + i;
      
      score = score + query[q_idx] * key_t[k_idx];
    }
    
    // Apply scaling
    score = score * dimensions.scale;
    
    // Store for softmax calculation
    // This is simplified; a complete implementation would handle masking here
    let attn_idx = ((batch_idx * dimensions.num_heads + head_idx) * dimensions.seq_len + row) * dimensions.seq_len + col;
    output[attn_idx] = score;
  }
  
  // Synchronize before softmax
  workgroupBarrier();
  
  // Second phase: Compute softmax for each query position
  // Find max for numerical stability
  if (col == 0 && row < dimensions.seq_len) {
    var max_val: f32 = -3.402823e+38;
    
    for (var i: u32 = 0u; i < dimensions.seq_len; i = i + 1u) {
      let attn_idx = ((batch_idx * dimensions.num_heads + head_idx) * dimensions.seq_len + row) * dimensions.seq_len + i;
      max_val = max(max_val, output[attn_idx]);
    }
    
    // Store max value for this row
    max_values[row] = max_val;
  }
  
  // Synchronize after finding max
  workgroupBarrier();
  
  // Apply exp(x - max) and sum
  if (col == 0 && row < dimensions.seq_len) {
    var sum: f32 = 0.0;
    
    for (var i: u32 = 0u; i < dimensions.seq_len; i = i + 1u) {
      let attn_idx = ((batch_idx * dimensions.num_heads + head_idx) * dimensions.seq_len + row) * dimensions.seq_len + i;
      
      // Apply exp(score - max) for numerical stability
      let exp_val = exp(output[attn_idx] - max_values[row]);
      output[attn_idx] = exp_val;
      sum = sum + exp_val;
    }
    
    // Store sum for normalization
    sum_values[row] = sum;
  }
  
  // Synchronize before normalization
  workgroupBarrier();
  
  // Normalize with sum to get softmax
  if (row < dimensions.seq_len && col < dimensions.seq_len) {
    let attn_idx = ((batch_idx * dimensions.num_heads + head_idx) * dimensions.seq_len + row) * dimensions.seq_len + col;
    
    // Normalize to get softmax value
    output[attn_idx] = output[attn_idx] / sum_values[row];
  }
  
  // Synchronize before matmul with values
  workgroupBarrier();
  
  // Final phase: multiply with V to get attention output
  if (row < dimensions.seq_len && col < dimensions.head_dim) {
    // For each query position (row) and output dimension (col),
    // compute weighted sum of value vectors
    var weighted_sum: f32 = 0.0;
    
    for (var i: u32 = 0u; i < dimensions.seq_len; i = i + 1u) {
      // Get attention weight
      let attn_idx = ((batch_idx * dimensions.num_heads + head_idx) * dimensions.seq_len + row) * dimensions.seq_len + i;
      let attn_weight = output[attn_idx];
      
      // Get value element
      let v_idx = ((batch_idx * dimensions.num_heads + head_idx) * dimensions.seq_len + i) * dimensions.head_dim + col;
      let value_elem = value[v_idx];
      
      weighted_sum = weighted_sum + attn_weight * value_elem;
    }
    
    // Store the final output
    let out_idx = ((batch_idx * dimensions.num_heads + head_idx) * dimensions.seq_len + row) * dimensions.head_dim + col;
    output[out_idx] = weighted_sum;
  }
}`;
  }
  
  /**
   * Generate shader for layer normalization + activation fusion
   * @param operations Operations to fuse
   * @returns WGSL shader code
   */
  private generateNormActivationShader(operations: FusionOpType[]): string {
    // Layer normalization followed by activation is common in many transformer models
    
    // Get the activation type
    const activationType = operations[1];
    
    // Activation helper functions
    const activationFunctions = `
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

// Helper function for GELU
fn gelu(x: f32) -> f32 {
  // Approximation of GELU
  let sqrt2overpi = 0.7978845608028654;
  let coeff = 0.044715;
  let x3 = x * x * x;
  return x * 0.5 * (1.0 + tanh_approx(sqrt2overpi * (x + coeff * x3)));
}

// Helper function for SiLU / Swish
fn silu(x: f32) -> f32 {
  return x * sigmoid(x);
}

// Helper function for ReLU
fn relu(x: f32) -> f32 {
  return max(0.0, x);
}

// Helper function for Leaky ReLU
fn leaky_relu(x: f32) -> f32 {
  return x < 0.0 ? 0.01 * x : x;
}

// Helper function for ELU
fn elu(x: f32) -> f32 {
  return x < 0.0 ? exp(x) - 1.0 : x;
}
`;
    
    // Determine activation function
    let activationCode: string;
    switch (activationType) {
      case 'relu':
        activationCode = 'relu(normalized)';
        break;
      case 'gelu':
        activationCode = 'gelu(normalized)';
        break;
      case 'sigmoid':
        activationCode = 'sigmoid(normalized)';
        break;
      case 'tanh':
        activationCode = 'tanh_approx(normalized)';
        break;
      case 'silu':
        activationCode = 'silu(normalized)';
        break;
      case 'leaky_relu':
        activationCode = 'leaky_relu(normalized)';
        break;
      case 'elu':
        activationCode = 'elu(normalized)';
        break;
      default:
        activationCode = 'normalized';
    }
    
    return /* wgsl */`
// Binding group layout:
// binding 0: Input tensor
// binding 1: Gamma (scale parameter)
// binding 2: Beta (shift parameter)
// binding 3: Output tensor
// binding 4: Uniform buffer with dimensions

struct Dimensions {
  batch_size: u32,
  seq_len: u32,
  hidden_size: u32,
  eps: f32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> dimensions: Dimensions;

// Workgroup shared memory for mean and variance calculation
var<workgroup> mean_values: array<f32, 64>; // Assuming max batch*seq_len per workgroup
var<workgroup> var_values: array<f32, 64>; // Assuming max batch*seq_len per workgroup

${activationFunctions}

// Fused layer normalization and activation
@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let batch_seq_idx = workgroup_id.x;
  let hidden_idx = global_id.y;
  
  // First pass: compute mean for this sequence position
  if (hidden_idx == 0) {
    var sum: f32 = 0.0;
    
    for (var i: u32 = 0u; i < dimensions.hidden_size; i = i + 1u) {
      let idx = batch_seq_idx * dimensions.hidden_size + i;
      sum = sum + input[idx];
    }
    
    let mean = sum / f32(dimensions.hidden_size);
    mean_values[batch_seq_idx] = mean;
  }
  
  // Synchronize after computing mean
  workgroupBarrier();
  
  // Second pass: compute variance
  if (hidden_idx == 0) {
    var sum_sq_diff: f32 = 0.0;
    let mean = mean_values[batch_seq_idx];
    
    for (var i: u32 = 0u; i < dimensions.hidden_size; i = i + 1u) {
      let idx = batch_seq_idx * dimensions.hidden_size + i;
      let diff = input[idx] - mean;
      sum_sq_diff = sum_sq_diff + diff * diff;
    }
    
    let variance = sum_sq_diff / f32(dimensions.hidden_size);
    var_values[batch_seq_idx] = variance;
  }
  
  // Synchronize after computing variance
  workgroupBarrier();
  
  // Apply layer normalization and activation
  if (hidden_idx < dimensions.hidden_size) {
    let idx = batch_seq_idx * dimensions.hidden_size + hidden_idx;
    
    // Get mean and variance for this sequence position
    let mean = mean_values[batch_seq_idx];
    let variance = var_values[batch_seq_idx];
    
    // Apply layer normalization
    let normalized = (input[idx] - mean) * inverseSqrt(variance + dimensions.eps);
    
    // Apply scale and shift (gamma and beta)
    let scaled = normalized * gamma[hidden_idx] + beta[hidden_idx];
    
    // Apply activation function
    let activated = ${activationCode};
    
    // Write to output
    output[idx] = activated;
  }
}`;
  }
  
  /**
   * Generate a generic fusion shader for arbitrary operation sequences
   * @param operations Operations to fuse
   * @returns WGSL shader code
   */
  private generateGenericFusionShader(operations: FusionOpType[]): string {
    // This is a simplified generic fusion
    // In a real implementation, this would be more sophisticated
    
    // For now, we'll implement a simple element-wise operation chain
    // with special cases for activation functions
    
    // Helper functions for activations
    const helperFunctions = `
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

// Helper function for relu
fn relu(x: f32) -> f32 {
  return max(0.0, x);
}

// Helper function for GELU
fn gelu(x: f32) -> f32 {
  // Approximation of GELU
  let sqrt2overpi = 0.7978845608028654;
  let coeff = 0.044715;
  let x3 = x * x * x;
  return x * 0.5 * (1.0 + tanh_approx(sqrt2overpi * (x + coeff * x3)));
}

// Helper function for SiLU / Swish
fn silu(x: f32) -> f32 {
  return x * sigmoid(x);
}

// Helper function for Leaky ReLU
fn leaky_relu(x: f32) -> f32 {
  return x < 0.0 ? 0.01 * x : x;
}

// Helper function for ELU
fn elu(x: f32) -> f32 {
  return x < 0.0 ? exp(x) - 1.0 : x;
}

// Helper function for Layer Normalization
fn layer_norm(x: f32, mean: f32, variance: f32, gamma: f32, beta: f32) -> f32 {
  return gamma * (x - mean) / sqrt(variance + 1e-5) + beta;
}
`;
    
    // Generate a sequential chain of operations
    let computationSteps = '';
    let resultVar = 'input_0[idx]';
    
    for (let i = 0; i < operations.length; i++) {
      const op = operations[i];
      const nextVar = `temp_${i+1}`;
      
      switch (op) {
        case 'add':
          computationSteps += `  let ${nextVar} = ${resultVar} + input_${i+1}[idx];\n`;
          break;
        case 'subtract':
          computationSteps += `  let ${nextVar} = ${resultVar} - input_${i+1}[idx];\n`;
          break;
        case 'multiply':
          computationSteps += `  let ${nextVar} = ${resultVar} * input_${i+1}[idx];\n`;
          break;
        case 'divide':
          computationSteps += `  let ${nextVar} = ${resultVar} / input_${i+1}[idx];\n`;
          break;
        case 'pow':
          computationSteps += `  let ${nextVar} = pow(${resultVar}, input_${i+1}[idx]);\n`;
          break;
        case 'max':
          computationSteps += `  let ${nextVar} = max(${resultVar}, input_${i+1}[idx]);\n`;
          break;
        case 'min':
          computationSteps += `  let ${nextVar} = min(${resultVar}, input_${i+1}[idx]);\n`;
          break;
        case 'relu':
          computationSteps += `  let ${nextVar} = relu(${resultVar});\n`;
          break;
        case 'sigmoid':
          computationSteps += `  let ${nextVar} = sigmoid(${resultVar});\n`;
          break;
        case 'tanh':
          computationSteps += `  let ${nextVar} = tanh_approx(${resultVar});\n`;
          break;
        case 'gelu':
          computationSteps += `  let ${nextVar} = gelu(${resultVar});\n`;
          break;
        case 'silu':
          computationSteps += `  let ${nextVar} = silu(${resultVar});\n`;
          break;
        case 'leaky_relu':
          computationSteps += `  let ${nextVar} = leaky_relu(${resultVar});\n`;
          break;
        case 'elu':
          computationSteps += `  let ${nextVar} = elu(${resultVar});\n`;
          break;
        case 'scale':
          computationSteps += `  let ${nextVar} = ${resultVar} * input_${i+1}[0]; // Scale factor\n`;
          break;
        default:
          // For unsupported operations, pass through
          computationSteps += `  let ${nextVar} = ${resultVar};\n`;
      }
      
      resultVar = nextVar;
    }
    
    // Number of input buffers needed (could be operations.length+1 or fewer)
    const numInputBindings = operations.filter(op => 
      ['add', 'subtract', 'multiply', 'divide'].includes(op)).length + 1;
    
    // Generate input bindings
    let inputBindings = '';
    for (let i = 0; i < numInputBindings; i++) {
      inputBindings += `@group(0) @binding(${i}) var<storage, read> input_${i}: array<f32>;\n`;
    }
    
    // Generate the shader
    return /* wgsl */`
// Binding group layout:
// binding 0..N-1: Input buffers
// binding N: Output buffer
// binding N+1: Uniform params

struct Params {
  length: u32,
};

${inputBindings}
@group(0) @binding(${numInputBindings}) var<storage, read_write> output: array<f32>;
@group(0) @binding(${numInputBindings + 1}) var<uniform> params: Params;

${helperFunctions}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  
  // Bounds check
  if (idx >= params.length) {
    return;
  }
  
  // Compute fused operations
${computationSteps}
  
  // Write the result
  output[idx] = ${resultVar};
}`;
  }
  
  /**
   * Execute a fused operation sequence
   * @param inputs Input tensors
   * @param operations Operations to fuse
   * @returns Output tensor
   */
  async executeFusedOperations<T>(
    inputs: Tensor<T>[],
    operations: FusionOpType[]
  ): Promise<Tensor<T>> {
    if (!this.canFuse(operations)) {
      throw new Error(`Cannot fuse operations: ${operations.join(', ')}`);
    }
    
    // Get the device
    const device = (this.backend as any).device;
    if (!device) {
      throw new Error('WebGPU device not available');
    }
    
    // Generate fusion ID
    const fusionId = this.generateFusionId(operations);
    
    // Get or create the shader and pipeline
    let pipeline: GPUComputePipeline;
    if (this.shaderCache.has(fusionId)) {
      pipeline = this.shaderCache.get(fusionId)!;
    } else {
      // Generate the shader
      const shaderCode = this.generateFusionShader(operations);
      
      // Create shader module
      const shaderModule = device.createShaderModule({
        code: shaderCode,
        label: `fusion_${fusionId}_shader`
      });
      
      // Create bind group layout
      let bindGroupLayout: GPUBindGroupLayout;
      if (this.bindGroupLayoutCache.has(fusionId)) {
        bindGroupLayout = this.bindGroupLayoutCache.get(fusionId)!;
      } else {
        // Create bind group layout based on the fusion type
        bindGroupLayout = this.createBindGroupLayout(device, operations, inputs.length);
        this.bindGroupLayoutCache.set(fusionId, bindGroupLayout);
      }
      
      // Create pipeline layout
      const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: `fusion_${fusionId}_layout`
      });
      
      // Create compute pipeline
      pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        },
        label: `fusion_${fusionId}_pipeline`
      });
      
      // Cache the pipeline
      this.shaderCache.set(fusionId, pipeline);
    }
    
    // Determine output shape based on fusion pattern
    let outputShape: number[];
    
    // Calculate output shape based on operation type
    if (this.isLinearActivation(operations)) {
      // For MatMul + Activation, output shape is [M, N]
      outputShape = [inputs[0].shape[0], inputs[1].shape[1]];
    } else if (this.isElementWiseChain(operations) || this.isBinaryUnary(operations)) {
      // For element-wise operations, output shape is same as input
      outputShape = [...inputs[0].shape];
    } else if (operations[0] === 'reshape' && operations.length > 1) {
      // For reshape + op, we need to handle this separately
      // This assumes inputs[1] contains the new shape
      outputShape = [...inputs[1].shape];
    } else {
      // Default: use first input's shape
      outputShape = [...inputs[0].shape];
    }
    
    // Create output tensor
    const outputTensor = new Tensor<T>(
      outputShape,
      null,
      {
        dataType: inputs[0].dataType,
        backend: 'webgpu',
        device: this.backend.id
      }
    );
    
    // Calculate total elements in output
    const totalElements = outputShape.reduce((a, b) => a * b, 1);
    
    // Create a command encoder
    const commandEncoder = device.createCommandEncoder({
      label: `fusion_${fusionId}_encoder`
    });
    
    // Create GPU buffers for inputs, output, and params
    const inputBuffers: GPUBuffer[] = [];
    const stagingBuffersToDestroy: GPUBuffer[] = [];
    
    // Create input buffers and upload data
    for (let i = 0; i < inputs.length; i++) {
      const input = inputs[i];
      // Get the buffer from the tensor or create a new one
      const buffer = (input as any).buffer || device.createBuffer({
        size: input.size * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: `input_${i}_buffer`
      });
      
      // If the buffer is new, upload the data
      if (!(input as any).buffer) {
        // Create a staging buffer for the upload
        const stagingBuffer = device.createBuffer({
          size: input.size * Float32Array.BYTES_PER_ELEMENT,
          usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE,
          label: `input_${i}_staging_buffer`
        });
        
        stagingBuffersToDestroy.push(stagingBuffer);
        
        // Upload data to staging buffer
        await stagingBuffer.mapAsync(GPUMapMode.WRITE);
        const arrayBuffer = stagingBuffer.getMappedRange();
        const data = input.data || new Float32Array(input.size).fill(0);
        new Float32Array(arrayBuffer).set(data);
        stagingBuffer.unmap();
        
        // Copy from staging buffer to input buffer
        commandEncoder.copyBufferToBuffer(
          stagingBuffer, 0,
          buffer, 0,
          input.size * Float32Array.BYTES_PER_ELEMENT
        );
      }
      
      inputBuffers.push(buffer);
    }
    
    // Create output buffer
    const outputBuffer = device.createBuffer({
      size: outputTensor.size * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'output_buffer'
    });
    
    // Create buffer for reading back the result
    const resultBuffer = device.createBuffer({
      size: outputTensor.size * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'result_buffer'
    });
    
    // Determine uniform buffer size based on operation pattern
    // Different fusion patterns need different uniform parameters
    let uniformBufferSize = 16 * 4; // Default: 16 u32/f32 values (64 bytes)
    if (this.isLinearActivation(operations)) {
      uniformBufferSize = 4 * 4; // 4 values for matmul: M, K, N, padding
    } else if (this.isSoftmaxOperation(operations)) {
      uniformBufferSize = 8 * 4; // 8 values for softmax: total_elements, inner_dim, outer_dim, stride, etc.
    }
    
    // Create uniform buffer for parameters
    const uniformBuffer = device.createBuffer({
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'uniform_buffer'
    });
    
    // Create parameter data based on fusion pattern
    const uniformData = new ArrayBuffer(uniformBufferSize);
    const uniformView = new DataView(uniformData);
    
    // Write uniform parameters based on operation type
    if (this.isLinearActivation(operations)) {
      // For matrix multiplication + activation
      // M, K, N dimensions
      const M = inputs[0].shape[0];
      const K = inputs[0].shape[1];
      const N = inputs[1].shape[1];
      
      uniformView.setUint32(0, M, true); // M (rows in A)
      uniformView.setUint32(4, K, true); // K (cols in A, rows in B)
      uniformView.setUint32(8, N, true); // N (cols in B)
    } else if (this.isSoftmaxOperation(operations)) {
      // For softmax operations
      const totalSize = inputs[0].size;
      
      // For simplicity, assuming 2D tensor with softmax along last dimension
      // In a full implementation, would check which axis to apply softmax
      const outerDim = inputs[0].shape[0];
      const innerDim = inputs[0].shape[1] || 1;
      const stride = 1;
      
      uniformView.setUint32(0, totalSize, true);   // total_elements
      uniformView.setUint32(4, innerDim, true);    // inner_dim (dim to apply softmax)
      uniformView.setUint32(8, outerDim, true);    // outer_dim
      uniformView.setUint32(12, stride, true);     // stride
    } else {
      // For element-wise operations
      uniformView.setUint32(0, totalElements, true); // length
      
      // If there are more shape dimensions, add them
      // This is used for reshape + activation patterns 
      if (operations[0] === 'reshape' && operations.length > 1) {
        // Store output shape dimensions
        const rank = outputShape.length;
        uniformView.setUint32(4, rank, true); // rank
        
        for (let i = 0; i < Math.min(rank, 4); i++) {
          uniformView.setUint32(8 + i * 4, outputShape[i], true);
        }
      }
    }
    
    // Upload uniform data
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    
    // Create bind group based on operation type
    let bindGroupEntries: GPUBindGroupEntry[] = [];
    
    // Add input bindings
    for (let i = 0; i < inputBuffers.length; i++) {
      bindGroupEntries.push({
        binding: i,
        resource: { buffer: inputBuffers[i] }
      });
    }
    
    // Add output binding
    bindGroupEntries.push({
      binding: inputBuffers.length,
      resource: { buffer: outputBuffer }
    });
    
    // Add uniform binding
    bindGroupEntries.push({
      binding: inputBuffers.length + 1,
      resource: { buffer: uniformBuffer }
    });
    
    // Create bind group
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindGroupEntries,
      label: `fusion_${fusionId}_bind_group`
    });
    
    // Create compute pass
    const passEncoder = commandEncoder.beginComputePass({
      label: `fusion_${fusionId}_pass`
    });
    
    // Set pipeline and bind group
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    
    // Dispatch workgroups based on operation type
    if (this.isLinearActivation(operations)) {
      // For matrix multiplication
      const M = inputs[0].shape[0];
      const N = inputs[1].shape[1];
      
      // Dispatch with 16x16 workgroup size
      const workgroupCountX = Math.ceil(M / 16);
      const workgroupCountY = Math.ceil(N / 16);
      
      passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    } else if (this.isSoftmaxOperation(operations)) {
      // For softmax operations
      // Each workgroup handles one outer dimension slice
      const outerDim = inputs[0].shape[0];
      passEncoder.dispatchWorkgroups(outerDim);
    } else {
      // For element-wise operations
      // 256 threads per workgroup
      const workgroupCount = Math.ceil(totalElements / 256);
      passEncoder.dispatchWorkgroups(workgroupCount);
    }
    
    // End the compute pass
    passEncoder.end();
    
    // Copy output to result buffer
    commandEncoder.copyBufferToBuffer(
      outputBuffer, 0,
      resultBuffer, 0,
      outputTensor.size * Float32Array.BYTES_PER_ELEMENT
    );
    
    // Submit command buffer
    const commandBuffer = commandEncoder.finish();
    device.queue.submit([commandBuffer]);
    
    try {
      // Read back the result
      await resultBuffer.mapAsync(GPUMapMode.READ);
      const resultArrayBuffer = resultBuffer.getMappedRange();
      const resultData = new Float32Array(resultArrayBuffer.byteLength / Float32Array.BYTES_PER_ELEMENT);
      resultData.set(new Float32Array(resultArrayBuffer));
      resultBuffer.unmap();
      
      // Set the output tensor data
      (outputTensor as any).data = resultData;
      
      // Store buffer reference for possible reuse
      (outputTensor as any).buffer = outputBuffer;
      
      return outputTensor;
    } catch (error) {
      // Handle errors and clean up
      console.error('Error executing fused operations:', error);
      throw new Error(`Failed to execute fused operations: ${error}`);
    } finally {
      // Clean up - destroy temporary buffers
      resultBuffer.destroy();
      
      // Clean up staging buffers
      for (const buffer of stagingBuffersToDestroy) {
        if (buffer) buffer.destroy();
      }
    }
  }
  
  /**
   * Check if this operation sequence involves softmax
   * @param operations Operations to check
   * @returns Whether it involves softmax
   */
  private isSoftmaxOperation(operations: FusionOpType[]): boolean {
    return operations.includes('softmax');
  }
  
  /**
   * Generate shader for quantized matrix multiplication
   * @returns WGSL shader code for quantized matrix multiplication
   */
  private generateQuantizedMatmulShader(): string {
    // Get the bit precision to use
    const bitsPerWeight = this.config.bitsPerWeight || 4;
    
    // Set workgroup sizes for optimized matrix multiplication
    const workgroupSizeX = 8;
    const workgroupSizeY = 32;
    
    // Generate shader code for quantized matrix multiplication
    return /* wgsl */`
    // Quantized matrix multiplication shader (${bitsPerWeight}-bit weights)
    // A: quantized weight matrix [M,K] (${bitsPerWeight}-bit packed)
    // B: activation matrix [K,N] (fp32)
    // C: output matrix [M,N] (fp32)
    
    struct Dimensions {
      M: u32,            // Rows in A
      K: u32,            // Columns in A / Rows in B
      N: u32,            // Columns in B
      bits_per_weight: u32, // Bits per weight (typically ${bitsPerWeight})
    };
    
    @group(0) @binding(0) var<storage, read> weights: array<u32>;         // Quantized weights
    @group(0) @binding(1) var<storage, read> activations: array<f32>;     // FP32 activations
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;    // FP32 output
    @group(0) @binding(3) var<uniform> dimensions: Dimensions;            // Matrix dimensions
    @group(0) @binding(4) var<storage, read> scales: array<f32>;          // Scales for dequantization
    @group(0) @binding(5) var<storage, read> zero_points: array<f32>;     // Zero points
    
    // Helper function to unpack ${bitsPerWeight}-bit values from a u32
    fn unpack_values(packed: u32) -> array<f32, ${Math.floor(32 / bitsPerWeight)}> {
      var result: array<f32, ${Math.floor(32 / bitsPerWeight)}>;
      let values_per_u32 = 32u / ${bitsPerWeight}u;
      let mask = (1u << ${bitsPerWeight}u) - 1u;
      
      for (var i = 0u; i < values_per_u32; i = i + 1u) {
        // Extract quantized value
        let quant_value = (packed >> (i * ${bitsPerWeight}u)) & mask;
        
        // Dequantize using scale and zero point
        // For simplicity using per-tensor quantization - can be expanded for per-channel
        let scale = scales[0];
        let zero_point = zero_points[0];
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
      let values_per_u32 = 32u / dimensions.bits_per_weight;
      
      // Process multiple elements (quantized) at a time
      for (var k_base = 0u; k_base < dimensions.K; k_base = k_base + values_per_u32) {
        // Calculate the packed weight index
        let weight_idx = (row * dimensions.K + k_base) / values_per_u32;
        
        // Get the packed weights
        let packed_weights = weights[weight_idx];
        
        // Unpack weights
        let unpacked_weights = unpack_values(packed_weights);
        
        // Perform dot product for this section
        for (var i = 0u; i < values_per_u32; i = i + 1u) {
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
   * Generate shader for quantized matrix multiplication with activation function
   * @param activationType Type of activation function to apply
   * @returns WGSL shader code
   */
  private generateQuantizedMatmulActivationShader(activationType: FusionOpType): string {
    // Get the bit precision to use
    const bitsPerWeight = this.config.bitsPerWeight || 4;
    
    // Set workgroup sizes for optimized matrix multiplication
    const workgroupSizeX = 8;
    const workgroupSizeY = 32;
    
    // Generate activation function code
    let activationFunction = '';
    let activationCall = 'sum';
    
    switch (activationType) {
      case 'relu':
        activationFunction = 'fn activation(x: f32) -> f32 { return max(0.0, x); }';
        activationCall = 'activation(sum)';
        break;
      case 'sigmoid':
        activationFunction = 'fn activation(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }';
        activationCall = 'activation(sum)';
        break;
      case 'tanh':
        activationFunction = 'fn activation(x: f32) -> f32 { let s = 1.0 / (1.0 + exp(-2.0 * x)); return 2.0 * s - 1.0; }';
        activationCall = 'activation(sum)';
        break;
      case 'gelu':
        activationFunction = `fn activation(x: f32) -> f32 { 
          let sqrt2overpi = 0.7978845608028654;
          let coeff = 0.044715;
          let x3 = x * x * x;
          return 0.5 * x * (1.0 + tanh(sqrt2overpi * (x + coeff * x3)));
        }`;
        activationCall = 'activation(sum)';
        break;
      case 'silu': // Swish activation
        activationFunction = 'fn activation(x: f32) -> f32 { return x * (1.0 / (1.0 + exp(-x))); }';
        activationCall = 'activation(sum)';
        break;
      case 'leaky_relu':
        activationFunction = 'fn activation(x: f32) -> f32 { return x < 0.0 ? 0.01 * x : x; }';
        activationCall = 'activation(sum)';
        break;
      default:
        activationFunction = '// No activation function';
        activationCall = 'sum';
    }
    
    // Generate shader code
    return /* wgsl */`
    // Quantized matrix multiplication with ${activationType} activation (${bitsPerWeight}-bit weights)
    // A: quantized weight matrix [M,K] (${bitsPerWeight}-bit packed)
    // B: activation matrix [K,N] (fp32)
    // C: output matrix [M,N] (fp32)
    
    struct Dimensions {
      M: u32,            // Rows in A
      K: u32,            // Columns in A / Rows in B
      N: u32,            // Columns in B
      bits_per_weight: u32, // Bits per weight (typically ${bitsPerWeight})
      has_bias: u32,     // Whether to use bias (0 = no, 1 = yes)
    };
    
    @group(0) @binding(0) var<storage, read> weights: array<u32>;         // Quantized weights
    @group(0) @binding(1) var<storage, read> activations: array<f32>;     // FP32 activations
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;    // FP32 output
    @group(0) @binding(3) var<uniform> dimensions: Dimensions;            // Matrix dimensions
    @group(0) @binding(4) var<storage, read> scales: array<f32>;          // Scales for dequantization
    @group(0) @binding(5) var<storage, read> zero_points: array<f32>;     // Zero points
    @group(0) @binding(6) var<storage, read> bias: array<f32>;            // Optional bias
    
    // Helper function to unpack ${bitsPerWeight}-bit values from a u32
    fn unpack_values(packed: u32) -> array<f32, ${Math.floor(32 / bitsPerWeight)}> {
      var result: array<f32, ${Math.floor(32 / bitsPerWeight)}>;
      let values_per_u32 = 32u / ${bitsPerWeight}u;
      let mask = (1u << ${bitsPerWeight}u) - 1u;
      
      for (var i = 0u; i < values_per_u32; i = i + 1u) {
        // Extract quantized value
        let quant_value = (packed >> (i * ${bitsPerWeight}u)) & mask;
        
        // Dequantize using scale and zero point
        let scale = scales[0];
        let zero_point = zero_points[0];
        result[i] = (f32(quant_value) - zero_point) * scale;
      }
      
      return result;
    }
    
    // Activation function: ${activationType}
    ${activationFunction}
    
    @compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY})
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let row = global_id.x;
      let col = global_id.y;
      
      // Bounds check
      if (row >= dimensions.M || col >= dimensions.N) {
        return;
      }
      
      var sum: f32 = 0.0;
      let values_per_u32 = 32u / dimensions.bits_per_weight;
      
      // Process multiple elements (quantized) at a time
      for (var k_base = 0u; k_base < dimensions.K; k_base = k_base + values_per_u32) {
        // Calculate the packed weight index
        let weight_idx = (row * dimensions.K + k_base) / values_per_u32;
        
        // Get the packed weights
        let packed_weights = weights[weight_idx];
        
        // Unpack weights
        let unpacked_weights = unpack_values(packed_weights);
        
        // Perform dot product for this section
        for (var i = 0u; i < values_per_u32; i = i + 1u) {
          let k = k_base + i;
          if (k < dimensions.K) { // Bound check for K dimension
            let act_idx = k * dimensions.N + col;
            sum = sum + unpacked_weights[i] * activations[act_idx];
          }
        }
      }
      
      // Add bias if enabled
      if (dimensions.has_bias == 1u) {
        sum = sum + bias[col];
      }
      
      // Apply activation function
      let activated_sum = ${activationCall};
      
      // Write output
      let output_idx = row * dimensions.N + col;
      output[output_idx] = activated_sum;
    }`;
  }
  
  /**
   * Generate shader for quantized attention mechanism for transformer models
   * @param operations Operations involved in the attention pattern
   * @returns WGSL shader code
   */
  private generateQuantizedAttentionShader(operations: FusionOpType[]): string {
    // Get the bit precision to use
    const bitsPerWeight = this.config.bitsPerWeight || 4;
    
    // Determine browser-specific optimizations
    let useOptimizations = this.config.useBrowserOptimizations !== false;
    
    return /* wgsl */`
    // Quantized Attention Mechanism (${bitsPerWeight}-bit weights)
    // Optimized for transformer model architectures
    
    struct AttentionParams {
      batch_size: u32,       // Batch size
      seq_len: u32,          // Sequence length
      hidden_size: u32,      // Hidden size
      num_heads: u32,        // Number of attention heads
      head_size: u32,        // Size of each head
      bits_per_weight: u32,  // Bits per weight for quantization
    };
    
    @group(0) @binding(0) var<storage, read> input: array<f32>;           // Input activations [batch_size, seq_len, hidden_size]
    @group(0) @binding(1) var<storage, read> qw: array<u32>;              // Quantized query weights
    @group(0) @binding(2) var<storage, read> kw: array<u32>;              // Quantized key weights
    @group(0) @binding(3) var<storage, read> vw: array<u32>;              // Quantized value weights
    @group(0) @binding(4) var<storage, read_write> output: array<f32>;    // Output [batch_size, seq_len, hidden_size]
    @group(0) @binding(5) var<storage, read> q_scales: array<f32>;        // Scales for query weights
    @group(0) @binding(6) var<storage, read> k_scales: array<f32>;        // Scales for key weights
    @group(0) @binding(7) var<storage, read> v_scales: array<f32>;        // Scales for value weights
    @group(0) @binding(8) var<uniform> params: AttentionParams;           // Attention parameters
    
    // Helper function to unpack quantized values
    fn unpack_values(packed: u32, bits_per_weight: u32) -> array<f32, 8> {
      var result: array<f32, 8>;
      let values_per_u32 = 32u / bits_per_weight;
      let mask = (1u << bits_per_weight) - 1u;
      
      for (var i = 0u; i < 8u; i = i + 1u) {
        if (i < values_per_u32) {
          // Extract quantized value
          let shift = i * bits_per_weight;
          let quant_value = (packed >> shift) & mask;
          
          // Using simplified dequantization with scale only
          // This can be expanded to use zero points and per-channel quantization
          result[i] = f32(quant_value) * q_scales[0];
        } else {
          result[i] = 0.0;
        }
      }
      
      return result;
    }
    
    // Workgroup shared memory for softmax normalization
    var<workgroup> max_values: array<f32, 64>; // Assuming max sequence length per workgroup
    var<workgroup> sum_values: array<f32, 64>; // Assuming max sequence length per workgroup
    
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      // Global position
      let batch_head = global_id.z; // Batch * head position
      let seq_idx = global_id.x;    // Sequence position
      let head_dim = global_id.y;   // Head dimension
      
      // Calculate indices
      let batch_idx = batch_head / params.num_heads;
      let head_idx = batch_head % params.num_heads;
      
      // Bounds check
      if (batch_idx >= params.batch_size || seq_idx >= params.seq_len || 
          head_idx >= params.num_heads || head_dim >= params.head_size) {
        return;
      }
      
      // Input position
      let input_offset = (batch_idx * params.seq_len + seq_idx) * params.hidden_size;
      
      // Output position for Q, K, V (separately)
      let qkv_offset = batch_idx * params.seq_len * params.num_heads * params.head_size + 
                       seq_idx * params.num_heads * params.head_size +
                       head_idx * params.head_size + head_dim;
      
      // Compute query, key, and value projections
      var q_val: f32 = 0.0;
      var k_val: f32 = 0.0;
      var v_val: f32 = 0.0;
      
      // Calculate column offset in weight matrices
      let weight_col_offset = head_idx * params.head_size + head_dim;
      
      // Calculate values per u32 based on bits per weight
      let values_per_u32 = 32u / params.bits_per_weight;
      
      // Process hidden dimension in chunks of values_per_u32
      for (var h_offset = 0u; h_offset < params.hidden_size; h_offset += values_per_u32) {
        // Calculate weight indices for Q, K, V
        let q_weight_idx = (h_offset * (params.num_heads * params.head_size) + weight_col_offset) / values_per_u32;
        let k_weight_idx = (h_offset * (params.num_heads * params.head_size) + weight_col_offset) / values_per_u32;
        let v_weight_idx = (h_offset * (params.num_heads * params.head_size) + weight_col_offset) / values_per_u32;
        
        // Get packed weights
        let q_packed = qw[q_weight_idx];
        let k_packed = kw[k_weight_idx];
        let v_packed = vw[v_weight_idx];
        
        // Dequantize weights
        let q_weights = unpack_values(q_packed, params.bits_per_weight);
        let k_weights = unpack_values(k_packed, params.bits_per_weight);
        let v_weights = unpack_values(v_packed, params.bits_per_weight);
        
        // Process input for this chunk
        for (var i = 0u; i < values_per_u32; i += 1u) {
          if (h_offset + i < params.hidden_size) {
            let input_val = input[input_offset + h_offset + i];
            q_val += input_val * q_weights[i];
            k_val += input_val * k_weights[i];
            v_val += input_val * v_weights[i];
          }
        }
      }
      
      // Write projections to Q, K, V output areas
      // (In a complete implementation, would process attention scores here)
      let q_out_idx = qkv_offset;
      let k_out_idx = q_out_idx + params.batch_size * params.seq_len * params.num_heads * params.head_size;
      let v_out_idx = k_out_idx + params.batch_size * params.seq_len * params.num_heads * params.head_size;
      
      output[q_out_idx] = q_val;
      output[k_out_idx] = k_val;
      output[v_out_idx] = v_val;
    }`;
  }
  
  /**
   * Generate shader for quantized matrix chain operations
   * @param operations Matrix operations to fuse
   * @returns WGSL shader code
   */
  private generateQuantizedMatrixChainShader(operations: FusionOpType[]): string {
    // Get the bit precision to use
    const bitsPerWeight = this.config.bitsPerWeight || 4;
    
    return /* wgsl */`
    // Quantized Matrix Chain Shader (${bitsPerWeight}-bit weights)
    // Fuses multiple matrix multiplications in sequence
    
    struct Dimensions {
      M: u32,            // Rows in A
      K: u32,            // Columns in A / Rows in B
      N: u32,            // Columns in B
      P: u32,            // Columns in C (for A*B*C)
      bits_per_weight: u32, // Bits per weight (typically ${bitsPerWeight})
    };
    
    @group(0) @binding(0) var<storage, read> matrix_a_weights: array<u32>;  // Quantized weights for A
    @group(0) @binding(1) var<storage, read> matrix_b_weights: array<u32>;  // Quantized weights for B
    @group(0) @binding(2) var<storage, read> matrix_c: array<f32>;          // Full precision C
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;      // Output matrix
    @group(0) @binding(4) var<storage, read> a_scales: array<f32>;          // Scales for A
    @group(0) @binding(5) var<storage, read> b_scales: array<f32>;          // Scales for B
    @group(0) @binding(6) var<uniform> dimensions: Dimensions;              // Matrix dimensions
    
    // Helper function to unpack quantized values
    fn unpack_values(packed: u32, bits_per_weight: u32) -> array<f32, 8> {
      var result: array<f32, 8>;
      let values_per_u32 = 32u / bits_per_weight;
      let mask = (1u << bits_per_weight) - 1u;
      
      for (var i = 0u; i < 8u; i = i + 1u) {
        if (i < values_per_u32) {
          // Extract quantized value
          let shift = i * bits_per_weight;
          let quant_value = (packed >> shift) & mask;
          
          // Simplified dequantization using only scale
          result[i] = f32(quant_value) * a_scales[0];
        } else {
          result[i] = 0.0;
        }
      }
      
      return result;
    }
    
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      // Process (A*B)*C matrix chain
      let row = global_id.x;
      let col = global_id.y;
      
      // Bounds check
      if (row >= dimensions.M || col >= dimensions.P) {
        return;
      }
      
      // Matrix chain computes (A*B)*C
      // Implement first A*B
      var ab_result: f32 = 0.0;
      let values_per_u32 = 32u / dimensions.bits_per_weight;
      
      // Compute A*B with quantized weights
      for (var k_base = 0u; k_base < dimensions.K; k_base += values_per_u32) {
        // Get packed weights for A
        let a_weight_idx = (row * dimensions.K + k_base) / values_per_u32;
        let a_packed = matrix_a_weights[a_weight_idx];
        let a_weights = unpack_values(a_packed, dimensions.bits_per_weight);
        
        // For each column in B
        for (var n = 0u; n < dimensions.N; n += 1u) {
          var ab_element: f32 = 0.0;
          
          // Process this segment of weights
          for (var i = 0u; i < values_per_u32; i += 1u) {
            if (k_base + i < dimensions.K) {
              // Get packed weights for B
              let b_weight_idx = ((k_base + i) * dimensions.N + n) / values_per_u32;
              let b_weight_offset = ((k_base + i) * dimensions.N + n) % values_per_u32;
              let b_packed = matrix_b_weights[b_weight_idx];
              
              // Extract B weight
              let b_mask = (1u << dimensions.bits_per_weight) - 1u;
              let b_shift = b_weight_offset * dimensions.bits_per_weight;
              let b_quant = (b_packed >> b_shift) & b_mask;
              let b_value = f32(b_quant) * b_scales[0];
              
              // Accumulate product
              ab_element += a_weights[i] * b_value;
            }
          }
          
          // Now multiply by C
          let c_idx = n * dimensions.P + col;
          ab_result += ab_element * matrix_c[c_idx];
        }
      }
      
      // Write final result
      output[row * dimensions.P + col] = ab_result;
    }`;
  }
  
  /**
   * Create a bind group layout for a fusion operation
   * @param device WebGPU device
   * @param operations Operations to fuse
   * @param numInputs Number of input tensors
   * @returns Bind group layout
   */
  /**
   * Detect the browser type and configure optimizations accordingly
   */
  private detectAndSetBrowserOptimizations(): void {
    // Detect browser type
    const userAgent = navigator.userAgent.toLowerCase();
    
    // Set appropriate optimizations based on browser
    if (userAgent.includes('firefox')) {
      // Firefox-specific settings
      console.log('Applying Firefox-specific WebGPU optimizations');
      // Firefox often performs better with smaller workgroups
      this.config.browserOptimizationType = 'firefox';
      
    } else if (userAgent.includes('safari') || userAgent.includes('applewebkit') && !userAgent.includes('chrome')) {
      // Safari-specific settings
      console.log('Applying Safari-specific WebGPU optimizations');
      // Safari/Metal can handle larger workgroups but requires specific memory patterns
      this.config.browserOptimizationType = 'safari';
      
    } else if (userAgent.includes('edg/')) {
      // Edge-specific settings
      console.log('Applying Edge-specific WebGPU optimizations');
      // Edge (Chromium-based) benefits from specific shader optimizations
      this.config.browserOptimizationType = 'edge';
      
    } else if (userAgent.includes('chrome')) {
      // Chrome-specific settings
      console.log('Applying Chrome-specific WebGPU optimizations');
      // Chrome benefits from specific shader optimizations
      this.config.browserOptimizationType = 'chrome';
      
    } else {
      // Default settings
      console.log('Using default WebGPU settings (browser not specifically optimized)');
      this.config.browserOptimizationType = 'default';
    }
  }
  
  private createBindGroupLayout(
    device: GPUDevice,
    operations: FusionOpType[],
    numInputs: number
  ): GPUBindGroupLayout {
    // This is a simplified implementation
    // The actual layout would depend on the specific fusion pattern
    
    const entries: GPUBindGroupLayoutEntry[] = [];
    
    // Add input bindings
    for (let i = 0; i < numInputs; i++) {
      entries.push({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' }
      });
    }
    
    // Add output binding
    entries.push({
      binding: numInputs,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' }
    });
    
    // Add uniform binding
    entries.push({
      binding: numInputs + 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'uniform' }
    });
    
    // Create and return the layout
    return device.createBindGroupLayout({
      entries,
      label: `fusion_layout_${operations.join('_')}`
    });
  }
}