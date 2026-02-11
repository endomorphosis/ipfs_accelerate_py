/**
 * WebNN TypeScript definitions
 * 
 * These are simplified type definitions for WebNN to use in our SDK.
 * For complete definitions, see:
 * https://www.w3.org/TR/webnn/
 */

declare global {
  /**
   * The entry point to WebNN functionality in the browser
   */
  interface Navigator {
    readonly ml?: ML;
  }

  /**
   * Main ML interface provided by the browser
   */
  interface ML {
    /**
     * Create a neural network context
     */
    createContext(options?: MLContextOptions): Promise<MLContext>;
  }

  /**
   * Context options
   */
  interface MLContextOptions {
    /**
     * Device preference
     */
    devicePreference?: MLDevicePreference;
    
    /**
     * Power preference
     */
    powerPreference?: MLPowerPreference;
  }

  /**
   * Device preference
   */
  type MLDevicePreference = 'gpu' | 'cpu';

  /**
   * Power preference
   */
  type MLPowerPreference = 'default' | 'low-power' | 'high-performance';

  /**
   * Neural network context
   */
  interface MLContext {
    /**
     * Create a graph builder
     */
    createGraphBuilder(): MLGraphBuilder;
    
    /**
     * Compute the result of a graph
     */
    compute(graph: MLGraph, inputs: MLNamedInputs): Promise<MLNamedOutputs>;
  }

  /**
   * Graph builder
   */
  interface MLGraphBuilder {
    /**
     * Input
     */
    input(name: string, options: MLInputOperandOptions): MLOperand;
    
    /**
     * Constant
     */
    constant(options: MLOperandOptions, buffer: BufferSource): MLOperand;
    
    /**
     * Build a graph
     */
    build(outputs: MLNamedOperands): MLGraph;
    
    // Basic operations
    /**
     * Add operation
     */
    add(a: MLOperand, b: MLOperand): MLOperand;
    
    /**
     * Sub operation
     */
    sub(a: MLOperand, b: MLOperand): MLOperand;
    
    /**
     * Mul operation
     */
    mul(a: MLOperand, b: MLOperand): MLOperand;
    
    /**
     * Div operation
     */
    div(a: MLOperand, b: MLOperand): MLOperand;
    
    // Neural network operations
    /**
     * Conv2d operation
     */
    conv2d(
      input: MLOperand,
      filter: MLOperand,
      options?: MLConv2dOptions
    ): MLOperand;
    
    /**
     * Gemm (matrix multiplication) operation
     */
    gemm(
      a: MLOperand,
      b: MLOperand,
      options?: MLGemmOptions
    ): MLOperand;
    
    /**
     * Matmul operation
     */
    matmul(
      a: MLOperand,
      b: MLOperand
    ): MLOperand;
    
    /**
     * Resample2d operation
     */
    resample2d(
      input: MLOperand,
      options: MLResample2dOptions
    ): MLOperand;
    
    /**
     * Concat operation
     */
    concat(
      inputs: MLOperand[],
      axis: number
    ): MLOperand;
    
    // Activation operations
    /**
     * Relu operation
     */
    relu(input: MLOperand): MLOperand;
    
    /**
     * Sigmoid operation
     */
    sigmoid(input: MLOperand): MLOperand;
    
    /**
     * Tanh operation
     */
    tanh(input: MLOperand): MLOperand;
    
    /**
     * Leaky relu operation
     */
    leakyRelu(input: MLOperand, options?: MLLeakyReluOptions): MLOperand;
    
    /**
     * Softmax operation
     */
    softmax(input: MLOperand): MLOperand;
    
    // Tensor operations
    /**
     * Reshape operation
     */
    reshape(
      input: MLOperand,
      newShape: number[]
    ): MLOperand;
    
    /**
     * Transpose operation
     */
    transpose(
      input: MLOperand,
      options?: MLTransposeOptions
    ): MLOperand;
    
    /**
     * Slice operation
     */
    slice(
      input: MLOperand,
      starts: number[],
      sizes: number[]
    ): MLOperand;
    
    /**
     * Split operation
     */
    split(
      input: MLOperand,
      splits: number[],
      axis: number
    ): MLOperand[];
    
    // Pooling operations
    /**
     * AvgPool2d operation
     */
    averagePool2d(
      input: MLOperand,
      options: MLPool2dOptions
    ): MLOperand;
    
    /**
     * MaxPool2d operation
     */
    maxPool2d(
      input: MLOperand,
      options: MLPool2dOptions
    ): MLOperand;
    
    // Reduction operations
    /**
     * ReduceMean operation
     */
    reduceMean(
      input: MLOperand,
      options: MLReduceOptions
    ): MLOperand;
    
    /**
     * ReduceMax operation
     */
    reduceMax(
      input: MLOperand,
      options: MLReduceOptions
    ): MLOperand;
    
    /**
     * ReduceMin operation
     */
    reduceMin(
      input: MLOperand,
      options: MLReduceOptions
    ): MLOperand;
    
    /**
     * ReduceSum operation
     */
    reduceSum(
      input: MLOperand,
      options: MLReduceOptions
    ): MLOperand;
    
    // Normalization operations
    /**
     * BatchNormalization operation
     */
    batchNormalization(
      input: MLOperand,
      mean: MLOperand,
      variance: MLOperand,
      options?: MLBatchNormalizationOptions
    ): MLOperand;
    
    /**
     * InstanceNormalization operation
     */
    instanceNormalization(
      input: MLOperand,
      options?: MLInstanceNormalizationOptions
    ): MLOperand;
    
    /**
     * LayerNormalization operation
     */
    layerNormalization(
      input: MLOperand,
      options?: MLLayerNormalizationOptions
    ): MLOperand;
    
    // Attention and transformer operations
    /**
     * Attention operation (Transformers)
     */
    attention(
      query: MLOperand,
      key: MLOperand,
      value: MLOperand,
      options?: MLAttentionOptions
    ): MLOperand;
    
    /**
     * GELU operation (Gaussian Error Linear Units)
     */
    gelu(input: MLOperand): MLOperand;
  }

  /**
   * Operand options
   */
  interface MLOperandOptions {
    /**
     * Data type
     */
    type: MLOperandType;
    
    /**
     * Dimensions
     */
    dimensions: number[];
  }

  /**
   * Input operand options
   */
  interface MLInputOperandOptions extends MLOperandOptions {}

  /**
   * Operand
   */
  interface MLOperand {}

  /**
   * Named operands (mapping of output name to operand)
   */
  interface MLNamedOperands {
    [outputName: string]: MLOperand;
  }

  /**
   * Named inputs (mapping of input name to buffer)
   */
  interface MLNamedInputs {
    [inputName: string]: MLNamedArrayBufferViews;
  }

  /**
   * Named outputs (mapping of output name to buffer)
   */
  interface MLNamedOutputs {
    [outputName: string]: MLNamedArrayBufferViews;
  }

  /**
   * Named array buffer views
   */
  interface MLNamedArrayBufferViews {
    /**
     * Get resource as array buffer view
     */
    resource: ArrayBufferView;
  }

  /**
   * Type of operand
   */
  type MLOperandType = 'float32' | 'float16' | 'int32' | 'uint32' | 'int64' | 'uint64' | 'int8' | 'uint8';

  /**
   * Graph
   */
  interface MLGraph {}

  /**
   * Conv2d options
   */
  interface MLConv2dOptions {
    /**
     * Padding
     */
    padding?: [number, number, number, number] | number;
    
    /**
     * Strides
     */
    strides?: [number, number] | number;
    
    /**
     * Dilations
     */
    dilations?: [number, number] | number;
    
    /**
     * Groups
     */
    groups?: number;
    
    /**
     * Input layout
     */
    inputLayout?: 'nchw' | 'nhwc';
    
    /**
     * Filter layout
     */
    filterLayout?: 'oihw' | 'hwio' | 'ohwi' | 'ihwo';
    
    /**
     * Bias
     */
    bias?: MLOperand;
    
    /**
     * Activation
     */
    activation?: MLActivation;
  }

  /**
   * Gemm options
   */
  interface MLGemmOptions {
    /**
     * Alpha scalar
     */
    alpha?: number;
    
    /**
     * Beta scalar
     */
    beta?: number;
    
    /**
     * Transposition for A
     */
    aTranspose?: boolean;
    
    /**
     * Transposition for B
     */
    bTranspose?: boolean;
    
    /**
     * C matrix
     */
    c?: MLOperand;
  }

  /**
   * Activation function
   */
  type MLActivation = 'relu' | 'sigmoid' | 'leakyRelu' | 'tanh';

  /**
   * Leaky ReLU options
   */
  interface MLLeakyReluOptions {
    /**
     * Alpha value
     */
    alpha: number;
  }

  /**
   * Pool2d options
   */
  interface MLPool2dOptions {
    /**
     * Window dimensions
     */
    windowDimensions: [number, number];
    
    /**
     * Padding
     */
    padding?: [number, number, number, number] | number;
    
    /**
     * Strides
     */
    strides?: [number, number] | number;
    
    /**
     * Dilations
     */
    dilations?: [number, number] | number;
    
    /**
     * Layout
     */
    layout?: 'nchw' | 'nhwc';
  }

  /**
   * Reduce options
   */
  interface MLReduceOptions {
    /**
     * Axes
     */
    axes: number[];
    
    /**
     * Keep dimensions
     */
    keepDimensions?: boolean;
  }

  /**
   * Resample2d options
   */
  interface MLResample2dOptions {
    /**
     * Sizes
     */
    sizes: [number, number];
    
    /**
     * Mode
     */
    mode?: 'nearest' | 'linear';
    
    /**
     * Scales
     */
    scales?: [number, number];
    
    /**
     * Layout
     */
    layout?: 'nchw' | 'nhwc';
  }

  /**
   * Transpose options
   */
  interface MLTransposeOptions {
    /**
     * Permutation
     */
    permutation: number[];
  }

  /**
   * Batch normalization options
   */
  interface MLBatchNormalizationOptions {
    /**
     * Scale
     */
    scale?: MLOperand;
    
    /**
     * Bias
     */
    bias?: MLOperand;
    
    /**
     * Epsilon
     */
    epsilon?: number;
    
    /**
     * Activation
     */
    activation?: MLActivation;
  }

  /**
   * Instance normalization options
   */
  interface MLInstanceNormalizationOptions {
    /**
     * Scale
     */
    scale?: MLOperand;
    
    /**
     * Bias
     */
    bias?: MLOperand;
    
    /**
     * Epsilon
     */
    epsilon?: number;
    
    /**
     * Layout
     */
    layout?: 'nchw' | 'nhwc';
  }

  /**
   * Layer normalization options
   */
  interface MLLayerNormalizationOptions {
    /**
     * Scale
     */
    scale?: MLOperand;
    
    /**
     * Bias
     */
    bias?: MLOperand;
    
    /**
     * Epsilon
     */
    epsilon?: number;
    
    /**
     * Axis
     */
    axis?: number;
  }

  /**
   * Attention options
   */
  interface MLAttentionOptions {
    /**
     * Mask
     */
    mask?: MLOperand;
    
    /**
     * Bias
     */
    bias?: MLOperand;
    
    /**
     * Number of attention heads
     */
    heads?: number;
    
    /**
     * Scale factor
     */
    scale?: number;
  }
}

// This is needed to make this a module
export {};