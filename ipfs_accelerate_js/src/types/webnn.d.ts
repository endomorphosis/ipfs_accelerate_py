/**
 * Enhanced TypeScript definitions for WebNN
 */

// Base interfaces
interface MLContext {
  // Empty interface for type safety
}

interface MLOperandDescriptor {
  type: MLOperandType;
  dimensions?: number[];
}

type MLOperandType =
  | 'float32'
  | 'float16'
  | 'int32'
  | 'uint32'
  | 'int8'
  | 'uint8'
  | 'int64'
  | 'uint64';

// MLGraph for executing the neural network
interface MLGraph {
  compute(inputs: Record<string, MLOperand>): Promise<Record<string, MLOperand>>;
}

// MLOperand represents a tensor in WebNN
interface MLOperand {
  // Empty interface for type safety
}

// Graph builder for constructing neural networks
interface MLGraphBuilder {
  // Input and constant creation
  input(name: string, descriptor: MLOperandDescriptor): MLOperand;
  constant(descriptor: MLOperandDescriptor, value: ArrayBufferView): MLOperand;
  
  // Basic operations
  add(a: MLOperand, b: MLOperand): MLOperand;
  sub(a: MLOperand, b: MLOperand): MLOperand;
  mul(a: MLOperand, b: MLOperand): MLOperand;
  div(a: MLOperand, b: MLOperand): MLOperand;
  
  // Neural network operations
  relu(x: MLOperand): MLOperand;
  sigmoid(x: MLOperand): MLOperand;
  tanh(x: MLOperand): MLOperand;
  leakyRelu(x: MLOperand, alpha?: number): MLOperand;
  softmax(x: MLOperand): MLOperand;
  
  // Tensor operations
  concat(inputs: MLOperand[], axis: number): MLOperand;
  reshape(input: MLOperand, newShape: number[]): MLOperand;
  
  // Convolution operations
  conv2d(
    input: MLOperand,
    filter: MLOperand,
    options?: {
      padding?: [number, number, number, number];
      strides?: [number, number];
      dilations?: [number, number];
      groups?: number;
    }
  ): MLOperand;
  
  // Pooling operations
  averagePool2d(
    input: MLOperand,
    options?: {
      windowDimensions?: [number, number];
      padding?: [number, number, number, number];
      strides?: [number, number];
      dilations?: [number, number];
      layout?: 'nchw' | 'nhwc';
    }
  ): MLOperand;
  
  maxPool2d(
    input: MLOperand,
    options?: {
      windowDimensions?: [number, number];
      padding?: [number, number, number, number];
      strides?: [number, number];
      dilations?: [number, number];
      layout?: 'nchw' | 'nhwc';
    }
  ): MLOperand;
  
  // Matrix operations
  matmul(a: MLOperand, b: MLOperand): MLOperand;
  
  // Build the graph
  build(outputs: Record<string, MLOperand>): Promise<MLGraph>;
}

// WebNN API interfaces
interface MLContextOptions {
  devicePreference?: 'gpu' | 'cpu';
}

interface ML {
  createContext(options?: MLContextOptions): Promise<MLContext>;
  createGraphBuilder(context: MLContext): MLGraphBuilder;
}

interface Navigator {
  readonly ml?: ML;
}

// Helper types for our SDK
export type WebNNBackendType = 'webnn';
