/**
 * TypeScript definitions for WebNN (Web Neural Network API)
 * Based on the WebNN specification: https://webmachinelearning.github.io/webnn/
 */

/**
 * MLContext - The execution context for ML operations
 */
interface MLContext {
  // This interface is primarily used as a marker for the ML context
}

/**
 * MLOperand - Represents a multi-dimensional array (tensor)
 */
interface MLOperand {
  // This interface is primarily used as a marker for operands
}

/**
 * MLOperandDescriptor - Description of an operand to be created
 */
interface MLOperandDescriptor {
  type: MLOperandType;
  dimensions: number[];
}

/**
 * MLOperandType - Type of elements in an operand
 */
type MLOperandType = 'float32' | 'float16' | 'int32' | 'uint32' | 'int8' | 'uint8';

/**
 * MLGraph - A computational graph for ML operations
 */
interface MLGraph {
  /**
   * Computes the outputs of the graph with the given inputs
   */
  compute(inputs: Record<string, MLOperand>): Record<string, MLOperand>;
}

/**
 * MLGraphBuilder - Builds a computational graph for ML operations
 */
interface MLGraphBuilder {
  /**
   * Creates an input operand for the graph
   */
  input(name: string, descriptor: MLOperandDescriptor): MLOperand;
  
  /**
   * Creates a constant operand for the graph
   */
  constant(descriptor: MLOperandDescriptor, value: ArrayBufferView): MLOperand;
  
  /**
   * Builds the graph with the given outputs
   */
  build(outputs: Record<string, MLOperand>): Promise<MLGraph>;

  // Arithmetic operations
  
  /**
   * Adds two operands element-wise
   */
  add(a: MLOperand, b: MLOperand): MLOperand;
  
  /**
   * Subtracts the second operand from the first element-wise
   */
  sub(a: MLOperand, b: MLOperand): MLOperand;
  
  /**
   * Multiplies two operands element-wise
   */
  mul(a: MLOperand, b: MLOperand): MLOperand;
  
  /**
   * Divides the first operand by the second element-wise
   */
  div(a: MLOperand, b: MLOperand): MLOperand;
  
  /**
   * Computes the maximum of two operands element-wise
   */
  max(a: MLOperand, b: MLOperand): MLOperand;
  
  /**
   * Computes the minimum of two operands element-wise
   */
  min(a: MLOperand, b: MLOperand): MLOperand;
  
  /**
   * Computes the power of the first operand to the second element-wise
   */
  pow(a: MLOperand, b: MLOperand): MLOperand;
  
  /**
   * Computes the absolute value of an operand element-wise
   */
  abs(x: MLOperand): MLOperand;
  
  /**
   * Computes the natural exponential of an operand element-wise
   */
  exp(x: MLOperand): MLOperand;
  
  /**
   * Computes the floor of an operand element-wise
   */
  floor(x: MLOperand): MLOperand;
  
  /**
   * Computes the ceiling of an operand element-wise
   */
  ceil(x: MLOperand): MLOperand;
  
  /**
   * Computes the natural logarithm of an operand element-wise
   */
  log(x: MLOperand): MLOperand;
  
  /**
   * Computes the square root of an operand element-wise
   */
  sqrt(x: MLOperand): MLOperand;
  
  /**
   * Computes the sine of an operand element-wise
   */
  sin(x: MLOperand): MLOperand;
  
  /**
   * Computes the cosine of an operand element-wise
   */
  cos(x: MLOperand): MLOperand;
  
  /**
   * Computes the tangent of an operand element-wise
   */
  tan(x: MLOperand): MLOperand;
  
  /**
   * Computes the arc-sine of an operand element-wise
   */
  asin(x: MLOperand): MLOperand;
  
  /**
   * Computes the arc-cosine of an operand element-wise
   */
  acos(x: MLOperand): MLOperand;
  
  /**
   * Computes the arc-tangent of an operand element-wise
   */
  atan(x: MLOperand): MLOperand;
  
  /**
   * Computes the hyperbolic sine of an operand element-wise
   */
  sinh(x: MLOperand): MLOperand;
  
  /**
   * Computes the hyperbolic cosine of an operand element-wise
   */
  cosh(x: MLOperand): MLOperand;
  
  /**
   * Computes the hyperbolic tangent of an operand element-wise
   */
  tanh(x: MLOperand): MLOperand;
  
  /**
   * Computes the inverse of an operand element-wise
   */
  inverse(x: MLOperand): MLOperand;
  
  /**
   * Computes the natural logarithm of (1 + x) element-wise
   */
  log1p(x: MLOperand): MLOperand;
  
  /**
   * Negates an operand element-wise
   */
  neg(x: MLOperand): MLOperand;
  
  // Neural network operations
  
  /**
   * Applies the rectified linear unit function element-wise
   */
  relu(x: MLOperand): MLOperand;
  
  /**
   * Applies the sigmoid function element-wise
   */
  sigmoid(x: MLOperand): MLOperand;
  
  /**
   * Performs batch normalization
   */
  batchNormalization(
    input: MLOperand,
    mean: MLOperand,
    variance: MLOperand,
    options?: {
      scale?: MLOperand;
      bias?: MLOperand;
      epsilon?: number;
    }
  ): MLOperand;
  
  /**
   * Applies the softmax function
   */
  softmax(x: MLOperand): MLOperand;
  
  /**
   * Performs 2D convolution
   */
  conv2d(
    input: MLOperand,
    filter: MLOperand,
    options?: {
      padding?: [number, number, number, number] | number;
      strides?: [number, number] | number;
      dilations?: [number, number] | number;
      groups?: number;
      layout?: 'nchw' | 'nhwc';
      bias?: MLOperand;
      activation?: MLActivation;
    }
  ): MLOperand;
  
  /**
   * Performs 2D transposed convolution (deconvolution)
   */
  convTranspose2d(
    input: MLOperand,
    filter: MLOperand,
    options?: {
      padding?: [number, number, number, number] | number;
      strides?: [number, number] | number;
      dilations?: [number, number] | number;
      outputPadding?: [number, number] | number;
      groups?: number;
      layout?: 'nchw' | 'nhwc';
      bias?: MLOperand;
      activation?: MLActivation;
    }
  ): MLOperand;
  
  /**
   * Performs max pooling
   */
  maxPool2d(
    input: MLOperand,
    options?: {
      windowDimensions?: [number, number] | number;
      padding?: [number, number, number, number] | number;
      strides?: [number, number] | number;
      dilations?: [number, number] | number;
      layout?: 'nchw' | 'nhwc';
    }
  ): MLOperand;
  
  /**
   * Performs average pooling
   */
  averagePool2d(
    input: MLOperand,
    options?: {
      windowDimensions?: [number, number] | number;
      padding?: [number, number, number, number] | number;
      strides?: [number, number] | number;
      dilations?: [number, number] | number;
      layout?: 'nchw' | 'nhwc';
    }
  ): MLOperand;
  
  /**
   * Performs a matrix multiplication (fully connected layer)
   */
  gemm(
    a: MLOperand,
    b: MLOperand,
    options?: {
      c?: MLOperand;
      alpha?: number;
      beta?: number;
      aTranspose?: boolean;
      bTranspose?: boolean;
    }
  ): MLOperand;
  
  /**
   * Performs a matrix multiplication with bias (fully connected layer)
   */
  matmul(a: MLOperand, b: MLOperand): MLOperand;
  
  /**
   * Reshapes an operand to a new shape
   */
  reshape(input: MLOperand, newShape: number[]): MLOperand;
  
  /**
   * Transposes an operand by permuting its dimensions
   */
  transpose(input: MLOperand, permutation?: number[]): MLOperand;
  
  /**
   * Concatenates operands along a dimension
   */
  concat(inputs: MLOperand[], axis: number): MLOperand;
  
  /**
   * Splits an operand into multiple operands along a dimension
   */
  split(
    input: MLOperand,
    splits: number[],
    axis: number
  ): MLOperand[];
  
  /**
   * Slices an operand
   */
  slice(
    input: MLOperand,
    starts: number[],
    sizes: number[]
  ): MLOperand;
  
  /**
   * Reduces an operand by computing the mean along specified dimensions
   */
  reduceMean(
    input: MLOperand,
    options?: {
      axes?: number[];
      keepDimensions?: boolean;
    }
  ): MLOperand;
  
  /**
   * Reduces an operand by computing the sum along specified dimensions
   */
  reduceSum(
    input: MLOperand,
    options?: {
      axes?: number[];
      keepDimensions?: boolean;
    }
  ): MLOperand;
  
  /**
   * Reduces an operand by computing the maximum along specified dimensions
   */
  reduceMax(
    input: MLOperand,
    options?: {
      axes?: number[];
      keepDimensions?: boolean;
    }
  ): MLOperand;
  
  /**
   * Reduces an operand by computing the minimum along specified dimensions
   */
  reduceMin(
    input: MLOperand,
    options?: {
      axes?: number[];
      keepDimensions?: boolean;
    }
  ): MLOperand;
  
  /**
   * Reduces an operand by computing the product along specified dimensions
   */
  reduceProduct(
    input: MLOperand,
    options?: {
      axes?: number[];
      keepDimensions?: boolean;
    }
  ): MLOperand;
  
  /**
   * Gathers elements from an operand
   */
  gather(
    input: MLOperand,
    indices: MLOperand,
    options?: {
      axis?: number;
    }
  ): MLOperand;
  
  /**
   * Scatters elements from an operand
   */
  scatter(
    input: MLOperand,
    indices: MLOperand,
    updates: MLOperand,
    options?: {
      axis?: number;
    }
  ): MLOperand;
  
  /**
   * Pads an operand
   */
  pad(
    input: MLOperand,
    padding: number[][],
    options?: {
      value?: number;
    }
  ): MLOperand;
  
  /**
   * Clips (clamps) an operand element-wise
   */
  clamp(
    input: MLOperand,
    options?: {
      minValue?: number;
      maxValue?: number;
    }
  ): MLOperand;
  
  /**
   * Resizes an operand spatially
   */
  resample2d(
    input: MLOperand,
    scales: [number, number],
    options?: {
      mode?: 'nearest' | 'linear';
      layout?: 'nchw' | 'nhwc';
    }
  ): MLOperand;
  
  /**
   * Selects elements from two operands based on a condition
   */
  where(
    condition: MLOperand,
    a: MLOperand,
    b: MLOperand
  ): MLOperand;
  
  /**
   * Creates a one-hot encoded operand
   */
  oneHot(
    indices: MLOperand,
    depth: number,
    options?: {
      onValue?: number;
      offValue?: number;
      axis?: number;
    }
  ): MLOperand;
  
  /**
   * Performs a depthwise 2D convolution
   */
  conv2dTranspose(
    input: MLOperand,
    filter: MLOperand,
    options?: {
      padding?: [number, number, number, number] | number;
      strides?: [number, number] | number;
      dilations?: [number, number] | number;
      outputPadding?: [number, number] | number;
      groups?: number;
      layout?: 'nchw' | 'nhwc';
      bias?: MLOperand;
      activation?: MLActivation;
    }
  ): MLOperand;
  
  /**
   * Performs element-wise binary logical operations
   */
  logicalAnd(a: MLOperand, b: MLOperand): MLOperand;
  logicalOr(a: MLOperand, b: MLOperand): MLOperand;
  logicalXor(a: MLOperand, b: MLOperand): MLOperand;
  logicalNot(x: MLOperand): MLOperand;
  
  /**
   * Performs element-wise equal comparison
   */
  equal(a: MLOperand, b: MLOperand): MLOperand;
  
  /**
   * Performs element-wise greater comparison
   */
  greater(a: MLOperand, b: MLOperand): MLOperand;
  
  /**
   * Performs element-wise greater or equal comparison
   */
  greaterOrEqual(a: MLOperand, b: MLOperand): MLOperand;
  
  /**
   * Performs element-wise lesser comparison
   */
  lesser(a: MLOperand, b: MLOperand): MLOperand;
  
  /**
   * Performs element-wise lesser or equal comparison
   */
  lesserOrEqual(a: MLOperand, b: MLOperand): MLOperand;
  
  /**
   * Performs element-wise not equal comparison
   */
  notEqual(a: MLOperand, b: MLOperand): MLOperand;
  
  /**
   * Performs a depth-to-space rearrangement
   */
  depthToSpace(
    input: MLOperand,
    blockSize: number,
    options?: {
      mode?: 'blocks_first' | 'depth_first';
      layout?: 'nchw' | 'nhwc';
    }
  ): MLOperand;
  
  /**
   * Performs a space-to-depth rearrangement
   */
  spaceToDepth(
    input: MLOperand,
    blockSize: number,
    options?: {
      layout?: 'nchw' | 'nhwc';
    }
  ): MLOperand;
  
  /**
   * Adds a dimension of size 1 to an operand's shape
   */
  unsqueeze(
    input: MLOperand,
    axes: number[]
  ): MLOperand;
  
  /**
   * Removes dimensions of size 1 from an operand's shape
   */
  squeeze(
    input: MLOperand,
    axes?: number[]
  ): MLOperand;
}

/**
 * MLActivation - Activation function for neural network operations
 */
type MLActivation = 'relu' | 'sigmoid' | 'softmax' | 'tanh' | 'leakyRelu' | 'hardSwish' | 'elu';

/**
 * NavigatorML - Extension of Navigator with ML support
 */
interface NavigatorML {
  ml: ML;
}

/**
 * ML - Entry point for WebNN
 */
interface ML {
  /**
   * Creates an ML context
   */
  createContext(options?: MLContextOptions): MLContext;
  
  /**
   * Creates a graph builder for the given context
   */
  createGraphBuilder(context: MLContext, options?: MLGraphBuilderOptions): MLGraphBuilder;
}

/**
 * MLContextOptions - Options for creating an ML context
 */
interface MLContextOptions {
  devicePreference?: 'gpu' | 'cpu';
  powerPreference?: 'default' | 'low-power' | 'high-performance';
}

/**
 * MLGraphBuilderOptions - Options for creating a graph builder
 */
interface MLGraphBuilderOptions {
  // Currently no options are defined in the spec
}

/**
 * Extend Navigator to include ML
 */
interface Navigator extends NavigatorML {}

/**
 * Extend Worker's Navigator to include ML
 */
interface WorkerNavigator extends NavigatorML {}