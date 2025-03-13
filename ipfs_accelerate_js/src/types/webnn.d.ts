/**
 * WebNN type definitions
 * 
 * This file provides TypeScript type definitions for WebNN API.
 * Note: WebNN API is still evolving and may change in future browser versions.
 */

declare global {
  interface Navigator {
    readonly ml?: ML;
  }

  interface ML {
    createContext(options?: MLContextOptions): Promise<MLContext>;
  }

  interface MLContextOptions {
    deviceType?: 'cpu' | 'gpu';
    powerPreference?: 'default' | 'high-performance' | 'low-power';
  }

  interface MLContext {
    readonly deviceType?: string;
    readonly deviceInfo?: MLDeviceInfo;
    
    createOperand(descriptor: MLOperandDescriptor, bufferView?: BufferSource): MLOperand;
  }

  interface MLDeviceInfo {
    readonly name?: string;
    readonly type?: string;
  }

  interface MLOperandDescriptor {
    type: MLOperandType;
    dimensions: number[];
  }

  type MLOperandType = 'float32' | 'float16' | 'int32' | 'uint32' | 'int8' | 'uint8';

  interface MLOperand {}

  interface MLGraphBuilder {
    readonly context: MLContext;

    // Input and constant methods
    input(name: string, descriptor: MLOperandDescriptor): MLOperand;
    constant(descriptor: MLOperandDescriptor, bufferView: BufferSource): MLOperand;

    // Activation operations
    relu(input: MLOperand): MLOperand;
    sigmoid(input: MLOperand): MLOperand;
    tanh(input: MLOperand): MLOperand;
    leakyRelu(input: MLOperand, options?: { alpha?: number }): MLOperand;
    prelu(input: MLOperand, alpha: MLOperand): MLOperand;
    softmax(input: MLOperand): MLOperand;

    // Element-wise operations
    abs(input: MLOperand): MLOperand;
    exp(input: MLOperand): MLOperand;
    ceil(input: MLOperand): MLOperand;
    floor(input: MLOperand): MLOperand;
    log(input: MLOperand): MLOperand;
    neg(input: MLOperand): MLOperand;
    sin(input: MLOperand): MLOperand;
    cos(input: MLOperand): MLOperand;
    tan(input: MLOperand): MLOperand;
    asin(input: MLOperand): MLOperand;
    acos(input: MLOperand): MLOperand;
    atan(input: MLOperand): MLOperand;
    sinh(input: MLOperand): MLOperand;
    cosh(input: MLOperand): MLOperand;
    asinh(input: MLOperand): MLOperand;
    acosh(input: MLOperand): MLOperand;
    atanh(input: MLOperand): MLOperand;
    erf(input: MLOperand): MLOperand;

    // Binary operations
    add(a: MLOperand, b: MLOperand): MLOperand;
    sub(a: MLOperand, b: MLOperand): MLOperand;
    mul(a: MLOperand, b: MLOperand): MLOperand;
    div(a: MLOperand, b: MLOperand): MLOperand;
    pow(a: MLOperand, b: MLOperand): MLOperand;
    max(a: MLOperand, b: MLOperand): MLOperand;
    min(a: MLOperand, b: MLOperand): MLOperand;

    // Tensor operations
    concat(inputs: MLOperand[], axis: number): MLOperand;
    matmul(a: MLOperand, b: MLOperand): MLOperand;
    reshape(input: MLOperand, newShape: number[]): MLOperand;
    transpose(input: MLOperand, permutation?: number[]): MLOperand;
    slice(input: MLOperand, starts: number[], sizes: number[]): MLOperand;
    split(input: MLOperand, splits: number[], axis: number): MLOperand[];
    
    // Convolution operations
    conv2d(
      input: MLOperand, 
      filter: MLOperand, 
      options?: {
        padding?: number | [number, number, number, number];
        strides?: number | [number, number];
        dilations?: number | [number, number];
        groups?: number;
        layout?: 'nchw' | 'nhwc';
      }
    ): MLOperand;
    
    // Pooling operations
    averagePool2d(
      input: MLOperand,
      options?: {
        windowDimensions?: number | [number, number];
        padding?: number | [number, number, number, number];
        strides?: number | [number, number];
        dilations?: number | [number, number];
        layout?: 'nchw' | 'nhwc';
      }
    ): MLOperand;

    maxPool2d(
      input: MLOperand,
      options?: {
        windowDimensions?: number | [number, number];
        padding?: number | [number, number, number, number];
        strides?: number | [number, number];
        dilations?: number | [number, number];
        layout?: 'nchw' | 'nhwc';
      }
    ): MLOperand;

    // Normalization operations
    instanceNormalization(
      input: MLOperand,
      options?: {
        scale?: MLOperand;
        bias?: MLOperand;
        epsilon?: number;
        layout?: 'nchw' | 'nhwc';
      }
    ): MLOperand;

    layerNormalization(
      input: MLOperand,
      options?: {
        scale?: MLOperand;
        bias?: MLOperand;
        axes?: number[];
        epsilon?: number;
      }
    ): MLOperand;

    batchNormalization(
      input: MLOperand,
      mean: MLOperand,
      variance: MLOperand,
      options?: {
        scale?: MLOperand;
        bias?: MLOperand;
        epsilon?: number;
        layout?: 'nchw' | 'nhwc';
      }
    ): MLOperand;

    // Graph building
    build(options: MLGraphBuildOptions): Promise<MLGraph>;
  }

  interface MLGraphBuildOptions {
    inputs: MLOperand[] | Record<string, MLOperand>;
    outputs: MLOperand[] | Record<string, MLOperand>;
  }

  interface MLGraph {
    compute(inputs: Record<string, MLOperand>, outputs?: Record<string, MLOperand>): Promise<Record<string, MLOperand>>;
  }
}

// Declare the constructor for MLGraphBuilder
declare var MLGraphBuilder: {
  prototype: MLGraphBuilder;
  new(context: MLContext): MLGraphBuilder;
};