/**
 * WebNN Backend Implementation
 * Provides hardware acceleration for tensor operations using WebNN
 */

import { HardwareBackend, HardwareCapabilities } from '../interfaces/hardware_backend';
import { Tensor } from '../../tensor/tensor';

/**
 * WebNN operation type enumeration
 */
export enum WebNNOperationType {
  Add = 'add',
  Subtract = 'sub',
  Multiply = 'mul',
  Divide = 'div',
  MatMul = 'matmul',
  Relu = 'relu',
  Sigmoid = 'sigmoid',
  Tanh = 'tanh',
  Softmax = 'softmax',
  Reshape = 'reshape',
  Transpose = 'transpose'
}

/**
 * Interface for WebNN module
 * This provides type definitions for the WebNN API
 * Note: WebNN is still evolving, so these types might need updates
 */
interface MLContext {
  compute(graph: MLGraph): Promise<void>;
}

interface MLGraph {
  input(name: string, desc: MLOperandDescriptor): MLOperand;
  constant(desc: MLOperandDescriptor, value: ArrayBufferView): MLOperand;
  output(name: string, operand: MLOperand): void;
  relu(input: MLOperand): MLOperand;
  sigmoid(input: MLOperand): MLOperand;
  tanh(input: MLOperand): MLOperand;
  softmax(input: MLOperand, options?: any): MLOperand;
  add(a: MLOperand, b: MLOperand): MLOperand;
  sub(a: MLOperand, b: MLOperand): MLOperand;
  mul(a: MLOperand, b: MLOperand): MLOperand;
  div(a: MLOperand, b: MLOperand): MLOperand;
  matmul(a: MLOperand, b: MLOperand): MLOperand;
  reshape(input: MLOperand, newShape: number[]): MLOperand;
  transpose(input: MLOperand, permutation?: number[]): MLOperand;
}

interface MLOperand {
  readonly type: MLOperandType;
  readonly dimensions: number[];
}

interface MLOperandDescriptor {
  type: MLOperandType;
  dimensions: number[];
}

type MLOperandType = 'float32' | 'float16' | 'int32' | 'uint32' | 'int8' | 'uint8';

/**
 * WebNN backend implementation
 */
export class WebNNBackend implements HardwareBackend {
  /** Unique identifier for this backend */
  readonly id: string = 'webnn';
  
  /** Type of backend */
  readonly type: string = 'webnn';
  
  /** WebNN context for neural network computation */
  private context: MLContext | null = null;
  
  /** Whether the backend is initialized */
  private initialized: boolean = false;
  
  /** Map of cached graphs for reuse */
  private graphCache: Map<string, MLGraph> = new Map();
  
  /** Map of tensor to MLOperand output */
  private tensorOperands: Map<string, ArrayBuffer> = new Map();
  
  /** WebNN backend capabilities */
  readonly capabilities: HardwareCapabilities = {
    maxDimensions: 4,
    maxMatrixSize: 16384,
    supportedDataTypes: ['float32'],
    supportsAsync: true,
    supportedOperations: {
      basicArithmetic: true,
      matrixMultiplication: true,
      convolution: true,
      reduction: true,
      activation: true
    }
  };
  
  /**
   * Constructor
   */
  constructor() {
    // Generate a unique ID
    this.id = `webnn_${Date.now()}`;
  }
  
  /**
   * Check if WebNN is available in the current environment
   */
  get isAvailable(): boolean {
    // Check if the WebNN API is available
    return typeof navigator !== 'undefined' && 'ml' in navigator;
  }
  
  /**
   * Initialize the WebNN backend
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }
    
    if (!this.isAvailable) {
      throw new Error('WebNN is not available in this environment');
    }
    
    try {
      // Get the WebNN context
      // @ts-ignore - WebNN is not yet in TypeScript standard lib
      this.context = await navigator.ml.createContext();
      
      if (!this.context) {
        throw new Error('Failed to create WebNN context');
      }
      
      this.initialized = true;
      
      console.log('WebNN backend initialized successfully');
    } catch (error) {
      console.error('Failed to initialize WebNN backend:', error);
      throw error;
    }
  }
  
  /**
   * Check if the backend is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }
  
  /**
   * Generate a unique tensor ID
   * @param tensor Tensor to generate ID for
   * @returns Unique tensor ID
   */
  private getTensorId<T>(tensor: Tensor<T>): string {
    return `tensor_${tensor.shape.join('x')}_${tensor.dataType}_${tensor.backend}`;
  }
  
  /**
   * Convert tensor data type to WebNN data type
   * @param dataType Tensor data type
   * @returns WebNN data type
   */
  private convertDataType(dataType: string): MLOperandType {
    switch (dataType) {
      case 'float32':
        return 'float32';
      case 'float64':
      case 'float16':
        return 'float32'; // Fall back to float32 for now
      case 'int32':
        return 'int32';
      case 'uint8':
        return 'uint8';
      case 'int8':
        return 'int8';
      default:
        return 'float32'; // Default to float32
    }
  }
  
  /**
   * Allocate a tensor on the backend
   * @param tensor Tensor to allocate
   */
  async allocateTensor<T>(tensor: Tensor<T>): Promise<void> {
    if (!this.initialized || !this.context) {
      throw new Error('WebNN backend not initialized');
    }
    
    // Generate a unique tensor ID
    const tensorId = this.getTensorId(tensor);
    
    // If already allocated, do nothing
    if (this.tensorOperands.has(tensorId)) {
      return;
    }
    
    // Convert tensor data to appropriate typed array
    const typedArray = this.convertToTypedArray(tensor);
    
    // Store the array buffer
    this.tensorOperands.set(tensorId, typedArray.buffer.slice(0));
  }
  
  /**
   * Release a tensor from the backend
   * @param tensor Tensor to release
   */
  releaseTensor<T>(tensor: Tensor<T>): void {
    if (!this.initialized) {
      return;
    }
    
    const tensorId = this.getTensorId(tensor);
    
    if (this.tensorOperands.has(tensorId)) {
      this.tensorOperands.delete(tensorId);
    }
  }
  
  /**
   * Convert tensor data to an appropriate typed array
   * @param tensor Tensor to convert
   * @returns Typed array containing tensor data
   */
  private convertToTypedArray<T>(tensor: Tensor<T>): ArrayBufferView {
    switch (tensor.dataType) {
      case 'float32':
        return new Float32Array(tensor.data as unknown as ArrayLike<number>);
      case 'int32':
        return new Int32Array(tensor.data as unknown as ArrayLike<number>);
      case 'float64':
        // Convert to float32 for WebNN compatibility
        const float64Data = tensor.data as unknown as ArrayLike<number>;
        const float32Data = new Float32Array(float64Data.length);
        for (let i = 0; i < float64Data.length; i++) {
          float32Data[i] = Number(float64Data[i]);
        }
        return float32Data;
      case 'uint8':
        return new Uint8Array(tensor.data as unknown as ArrayLike<number>);
      case 'int8':
        return new Int8Array(tensor.data as unknown as ArrayLike<number>);
      default:
        // Default to float32
        return new Float32Array(tensor.data as unknown as ArrayLike<number>);
    }
  }
  
  /**
   * Create a typed array from an array buffer
   * @param buffer Array buffer
   * @param dataType Data type
   * @returns Typed array
   */
  private createTypedArrayFromBuffer(buffer: ArrayBuffer, dataType: string): ArrayBufferView {
    switch (dataType) {
      case 'float32':
        return new Float32Array(buffer);
      case 'int32':
        return new Int32Array(buffer);
      case 'float64':
        // WebNN doesn't support float64, so we use float32
        return new Float32Array(buffer);
      case 'uint8':
        return new Uint8Array(buffer);
      case 'int8':
        return new Int8Array(buffer);
      default:
        // Default to float32
        return new Float32Array(buffer);
    }
  }
  
  /**
   * Execute tensor addition
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  async add<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    return this.binaryOperation(a, b, WebNNOperationType.Add);
  }
  
  /**
   * Execute tensor subtraction
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  async subtract<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    return this.binaryOperation(a, b, WebNNOperationType.Subtract);
  }
  
  /**
   * Execute tensor multiplication (element-wise)
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  async multiply<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    return this.binaryOperation(a, b, WebNNOperationType.Multiply);
  }
  
  /**
   * Execute tensor division
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  async divide<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    return this.binaryOperation(a, b, WebNNOperationType.Divide);
  }
  
  /**
   * Execute a binary operation
   * @param a First tensor
   * @param b Second tensor
   * @param operation Operation type
   * @returns Resulting tensor
   */
  private async binaryOperation<T>(
    a: Tensor<T>,
    b: Tensor<T>,
    operation: WebNNOperationType
  ): Promise<Tensor<T>> {
    if (!this.initialized || !this.context) {
      throw new Error('WebNN backend not initialized');
    }
    
    // Ensure tensors are allocated
    await this.allocateTensor(a);
    await this.allocateTensor(b);
    
    // Get tensor IDs
    const tensorIdA = this.getTensorId(a);
    const tensorIdB = this.getTensorId(b);
    
    // Get tensor data
    const bufferA = this.tensorOperands.get(tensorIdA)!;
    const bufferB = this.tensorOperands.get(tensorIdB)!;
    
    // Create output tensor with same shape as input
    const outputShape = [...a.shape]; // Assuming shapes are compatible
    const outputTensor = new Tensor<T>(
      outputShape,
      null,
      {
        dataType: a.dataType,
        backend: 'webnn',
        device: this.id
      }
    );
    
    // Create a WebNN graph for this operation
    // @ts-ignore - WebNN is not yet in TypeScript standard lib
    const graph = await navigator.ml.createGraph();
    
    // Create input operands
    const inputA = graph.input('a', {
      type: this.convertDataType(a.dataType),
      dimensions: a.shape
    });
    
    const inputB = graph.input('b', {
      type: this.convertDataType(b.dataType),
      dimensions: b.shape
    });
    
    // Create the operation
    let output;
    switch (operation) {
      case WebNNOperationType.Add:
        output = graph.add(inputA, inputB);
        break;
      case WebNNOperationType.Subtract:
        output = graph.sub(inputA, inputB);
        break;
      case WebNNOperationType.Multiply:
        output = graph.mul(inputA, inputB);
        break;
      case WebNNOperationType.Divide:
        output = graph.div(inputA, inputB);
        break;
      default:
        throw new Error(`Unsupported operation: ${operation}`);
    }
    
    // Set the output
    graph.output('output', output);
    
    // Create input data
    const inputs = {
      'a': this.createTypedArrayFromBuffer(bufferA, a.dataType),
      'b': this.createTypedArrayFromBuffer(bufferB, b.dataType)
    };
    
    // Create output buffer
    const outputSize = outputShape.reduce((acc, dim) => acc * dim, 1);
    let outputBuffer;
    
    switch (a.dataType) {
      case 'float32':
        outputBuffer = new Float32Array(outputSize);
        break;
      case 'int32':
        outputBuffer = new Int32Array(outputSize);
        break;
      case 'uint8':
        outputBuffer = new Uint8Array(outputSize);
        break;
      case 'int8':
        outputBuffer = new Int8Array(outputSize);
        break;
      default:
        outputBuffer = new Float32Array(outputSize);
    }
    
    // Define outputs
    const outputs = {
      'output': outputBuffer
    };
    
    // Compute the graph
    // @ts-ignore - WebNN is not yet in TypeScript standard lib
    const results = await this.context.compute(graph, inputs, outputs);
    
    // Convert output buffer to tensor data
    for (let i = 0; i < outputSize; i++) {
      outputTensor.data[i] = outputBuffer[i] as unknown as T;
    }
    
    return outputTensor;
  }
  
  /**
   * Execute matrix multiplication
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  async matmul<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    if (!this.initialized || !this.context) {
      throw new Error('WebNN backend not initialized');
    }
    
    // Validate tensor shapes for matrix multiplication
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error('Matrix multiplication requires 2D tensors');
    }
    
    if (a.shape[1] !== b.shape[0]) {
      throw new Error(`Matrix dimensions mismatch: ${a.shape} and ${b.shape}`);
    }
    
    // Ensure tensors are allocated
    await this.allocateTensor(a);
    await this.allocateTensor(b);
    
    // Get tensor IDs
    const tensorIdA = this.getTensorId(a);
    const tensorIdB = this.getTensorId(b);
    
    // Get tensor data
    const bufferA = this.tensorOperands.get(tensorIdA)!;
    const bufferB = this.tensorOperands.get(tensorIdB)!;
    
    // Calculate output shape [M, N]
    const M = a.shape[0];
    const K = a.shape[1];
    const N = b.shape[1];
    const outputShape = [M, N];
    
    // Create output tensor
    const outputTensor = new Tensor<T>(
      outputShape,
      null,
      {
        dataType: a.dataType,
        backend: 'webnn',
        device: this.id
      }
    );
    
    // Create a WebNN graph for this operation
    // @ts-ignore - WebNN is not yet in TypeScript standard lib
    const graph = await navigator.ml.createGraph();
    
    // Create input operands
    const inputA = graph.input('a', {
      type: this.convertDataType(a.dataType),
      dimensions: a.shape
    });
    
    const inputB = graph.input('b', {
      type: this.convertDataType(b.dataType),
      dimensions: b.shape
    });
    
    // Create the matmul operation
    const output = graph.matmul(inputA, inputB);
    
    // Set the output
    graph.output('output', output);
    
    // Create input data
    const inputs = {
      'a': this.createTypedArrayFromBuffer(bufferA, a.dataType),
      'b': this.createTypedArrayFromBuffer(bufferB, b.dataType)
    };
    
    // Create output buffer
    const outputSize = outputShape.reduce((acc, dim) => acc * dim, 1);
    let outputBuffer;
    
    switch (a.dataType) {
      case 'float32':
        outputBuffer = new Float32Array(outputSize);
        break;
      case 'int32':
        outputBuffer = new Int32Array(outputSize);
        break;
      case 'uint8':
        outputBuffer = new Uint8Array(outputSize);
        break;
      case 'int8':
        outputBuffer = new Int8Array(outputSize);
        break;
      default:
        outputBuffer = new Float32Array(outputSize);
    }
    
    // Define outputs
    const outputs = {
      'output': outputBuffer
    };
    
    // Compute the graph
    // @ts-ignore - WebNN is not yet in TypeScript standard lib
    const results = await this.context.compute(graph, inputs, outputs);
    
    // Convert output buffer to tensor data
    for (let i = 0; i < outputSize; i++) {
      outputTensor.data[i] = outputBuffer[i] as unknown as T;
    }
    
    return outputTensor;
  }
  
  /**
   * Execute transpose operation
   * @param tensor Input tensor
   * @returns Transposed tensor
   */
  async transpose<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    if (!this.initialized || !this.context) {
      throw new Error('WebNN backend not initialized');
    }
    
    // Validate tensor shape
    if (tensor.shape.length !== 2) {
      throw new Error('Transpose requires a 2D tensor');
    }
    
    // Ensure tensor is allocated
    await this.allocateTensor(tensor);
    
    // Get tensor ID
    const tensorId = this.getTensorId(tensor);
    
    // Get tensor data
    const buffer = this.tensorOperands.get(tensorId)!;
    
    // Calculate output shape by swapping dimensions
    const outputShape = [tensor.shape[1], tensor.shape[0]];
    
    // Create output tensor
    const outputTensor = new Tensor<T>(
      outputShape,
      null,
      {
        dataType: tensor.dataType,
        backend: 'webnn',
        device: this.id
      }
    );
    
    // Create a WebNN graph for this operation
    // @ts-ignore - WebNN is not yet in TypeScript standard lib
    const graph = await navigator.ml.createGraph();
    
    // Create input operand
    const input = graph.input('input', {
      type: this.convertDataType(tensor.dataType),
      dimensions: tensor.shape
    });
    
    // Create the transpose operation with permutation [1, 0]
    const output = graph.transpose(input, [1, 0]);
    
    // Set the output
    graph.output('output', output);
    
    // Create input data
    const inputs = {
      'input': this.createTypedArrayFromBuffer(buffer, tensor.dataType)
    };
    
    // Create output buffer
    const outputSize = outputShape.reduce((acc, dim) => acc * dim, 1);
    let outputBuffer;
    
    switch (tensor.dataType) {
      case 'float32':
        outputBuffer = new Float32Array(outputSize);
        break;
      case 'int32':
        outputBuffer = new Int32Array(outputSize);
        break;
      case 'uint8':
        outputBuffer = new Uint8Array(outputSize);
        break;
      case 'int8':
        outputBuffer = new Int8Array(outputSize);
        break;
      default:
        outputBuffer = new Float32Array(outputSize);
    }
    
    // Define outputs
    const outputs = {
      'output': outputBuffer
    };
    
    // Compute the graph
    // @ts-ignore - WebNN is not yet in TypeScript standard lib
    const results = await this.context.compute(graph, inputs, outputs);
    
    // Convert output buffer to tensor data
    for (let i = 0; i < outputSize; i++) {
      outputTensor.data[i] = outputBuffer[i] as unknown as T;
    }
    
    return outputTensor;
  }
  
  /**
   * Execute ReLU activation function
   * @param tensor Input tensor
   * @returns Tensor with ReLU applied
   */
  async relu<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    return this.unaryOperation(tensor, WebNNOperationType.Relu);
  }
  
  /**
   * Execute sigmoid activation function
   * @param tensor Input tensor
   * @returns Tensor with sigmoid applied
   */
  async sigmoid<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    return this.unaryOperation(tensor, WebNNOperationType.Sigmoid);
  }
  
  /**
   * Execute tanh activation function
   * @param tensor Input tensor
   * @returns Tensor with tanh applied
   */
  async tanh<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    return this.unaryOperation(tensor, WebNNOperationType.Tanh);
  }
  
  /**
   * Execute a unary operation
   * @param tensor Input tensor
   * @param operation Operation type
   * @returns Resulting tensor
   */
  private async unaryOperation<T>(
    tensor: Tensor<T>,
    operation: WebNNOperationType
  ): Promise<Tensor<T>> {
    if (!this.initialized || !this.context) {
      throw new Error('WebNN backend not initialized');
    }
    
    // Ensure tensor is allocated
    await this.allocateTensor(tensor);
    
    // Get tensor ID
    const tensorId = this.getTensorId(tensor);
    
    // Get tensor data
    const buffer = this.tensorOperands.get(tensorId)!;
    
    // Create output tensor with same shape as input
    const outputShape = [...tensor.shape];
    const outputTensor = new Tensor<T>(
      outputShape,
      null,
      {
        dataType: tensor.dataType,
        backend: 'webnn',
        device: this.id
      }
    );
    
    // Create a WebNN graph for this operation
    // @ts-ignore - WebNN is not yet in TypeScript standard lib
    const graph = await navigator.ml.createGraph();
    
    // Create input operand
    const input = graph.input('input', {
      type: this.convertDataType(tensor.dataType),
      dimensions: tensor.shape
    });
    
    // Create the operation
    let output;
    switch (operation) {
      case WebNNOperationType.Relu:
        output = graph.relu(input);
        break;
      case WebNNOperationType.Sigmoid:
        output = graph.sigmoid(input);
        break;
      case WebNNOperationType.Tanh:
        output = graph.tanh(input);
        break;
      default:
        throw new Error(`Unsupported operation: ${operation}`);
    }
    
    // Set the output
    graph.output('output', output);
    
    // Create input data
    const inputs = {
      'input': this.createTypedArrayFromBuffer(buffer, tensor.dataType)
    };
    
    // Create output buffer
    const outputSize = outputShape.reduce((acc, dim) => acc * dim, 1);
    let outputBuffer;
    
    switch (tensor.dataType) {
      case 'float32':
        outputBuffer = new Float32Array(outputSize);
        break;
      case 'int32':
        outputBuffer = new Int32Array(outputSize);
        break;
      case 'uint8':
        outputBuffer = new Uint8Array(outputSize);
        break;
      case 'int8':
        outputBuffer = new Int8Array(outputSize);
        break;
      default:
        outputBuffer = new Float32Array(outputSize);
    }
    
    // Define outputs
    const outputs = {
      'output': outputBuffer
    };
    
    // Compute the graph
    // @ts-ignore - WebNN is not yet in TypeScript standard lib
    const results = await this.context.compute(graph, inputs, outputs);
    
    // Convert output buffer to tensor data
    for (let i = 0; i < outputSize; i++) {
      outputTensor.data[i] = outputBuffer[i] as unknown as T;
    }
    
    return outputTensor;
  }
  
  /**
   * Execute softmax activation function
   * @param tensor Input tensor
   * @param axis Axis to apply softmax on
   * @returns Tensor with softmax applied
   */
  async softmax<T>(tensor: Tensor<T>, axis: number = -1): Promise<Tensor<T>> {
    if (!this.initialized || !this.context) {
      throw new Error('WebNN backend not initialized');
    }
    
    // Handle negative axis
    if (axis < 0) {
      axis = tensor.shape.length + axis;
    }
    
    // Validate axis
    if (axis < 0 || axis >= tensor.shape.length) {
      throw new Error(`Invalid softmax axis: ${axis}`);
    }
    
    // Ensure tensor is allocated
    await this.allocateTensor(tensor);
    
    // Get tensor ID
    const tensorId = this.getTensorId(tensor);
    
    // Get tensor data
    const buffer = this.tensorOperands.get(tensorId)!;
    
    // Create output tensor with same shape as input
    const outputShape = [...tensor.shape];
    const outputTensor = new Tensor<T>(
      outputShape,
      null,
      {
        dataType: tensor.dataType,
        backend: 'webnn',
        device: this.id
      }
    );
    
    // Create a WebNN graph for this operation
    // @ts-ignore - WebNN is not yet in TypeScript standard lib
    const graph = await navigator.ml.createGraph();
    
    // Create input operand
    const input = graph.input('input', {
      type: this.convertDataType(tensor.dataType),
      dimensions: tensor.shape
    });
    
    // Create the softmax operation with the specified axis
    const output = graph.softmax(input, { axis });
    
    // Set the output
    graph.output('output', output);
    
    // Create input data
    const inputs = {
      'input': this.createTypedArrayFromBuffer(buffer, tensor.dataType)
    };
    
    // Create output buffer
    const outputSize = outputShape.reduce((acc, dim) => acc * dim, 1);
    let outputBuffer;
    
    switch (tensor.dataType) {
      case 'float32':
        outputBuffer = new Float32Array(outputSize);
        break;
      case 'int32':
        outputBuffer = new Int32Array(outputSize);
        break;
      case 'uint8':
        outputBuffer = new Uint8Array(outputSize);
        break;
      case 'int8':
        outputBuffer = new Int8Array(outputSize);
        break;
      default:
        outputBuffer = new Float32Array(outputSize);
    }
    
    // Define outputs
    const outputs = {
      'output': outputBuffer
    };
    
    // Compute the graph
    // @ts-ignore - WebNN is not yet in TypeScript standard lib
    const results = await this.context.compute(graph, inputs, outputs);
    
    // Convert output buffer to tensor data
    for (let i = 0; i < outputSize; i++) {
      outputTensor.data[i] = outputBuffer[i] as unknown as T;
    }
    
    return outputTensor;
  }
  
  /**
   * Execute tensor reshape
   * @param tensor Input tensor
   * @param newShape New shape
   * @returns Reshaped tensor
   */
  async reshape<T>(tensor: Tensor<T>, newShape: number[]): Promise<Tensor<T>> {
    if (!this.initialized || !this.context) {
      throw new Error('WebNN backend not initialized');
    }
    
    // Validate new shape
    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== tensor.size) {
      throw new Error(`Cannot reshape tensor of size ${tensor.size} to shape ${newShape} (size ${newSize})`);
    }
    
    // Ensure tensor is allocated
    await this.allocateTensor(tensor);
    
    // Get tensor ID
    const tensorId = this.getTensorId(tensor);
    
    // Get tensor data
    const buffer = this.tensorOperands.get(tensorId)!;
    
    // Create output tensor with new shape
    const outputTensor = new Tensor<T>(
      newShape,
      null,
      {
        dataType: tensor.dataType,
        backend: 'webnn',
        device: this.id
      }
    );
    
    // Create a WebNN graph for this operation
    // @ts-ignore - WebNN is not yet in TypeScript standard lib
    const graph = await navigator.ml.createGraph();
    
    // Create input operand
    const input = graph.input('input', {
      type: this.convertDataType(tensor.dataType),
      dimensions: tensor.shape
    });
    
    // Create the reshape operation
    const output = graph.reshape(input, newShape);
    
    // Set the output
    graph.output('output', output);
    
    // Create input data
    const inputs = {
      'input': this.createTypedArrayFromBuffer(buffer, tensor.dataType)
    };
    
    // Create output buffer
    const outputSize = newShape.reduce((acc, dim) => acc * dim, 1);
    let outputBuffer;
    
    switch (tensor.dataType) {
      case 'float32':
        outputBuffer = new Float32Array(outputSize);
        break;
      case 'int32':
        outputBuffer = new Int32Array(outputSize);
        break;
      case 'uint8':
        outputBuffer = new Uint8Array(outputSize);
        break;
      case 'int8':
        outputBuffer = new Int8Array(outputSize);
        break;
      default:
        outputBuffer = new Float32Array(outputSize);
    }
    
    // Define outputs
    const outputs = {
      'output': outputBuffer
    };
    
    // Compute the graph
    // @ts-ignore - WebNN is not yet in TypeScript standard lib
    const results = await this.context.compute(graph, inputs, outputs);
    
    // Convert output buffer to tensor data
    for (let i = 0; i < outputSize; i++) {
      outputTensor.data[i] = outputBuffer[i] as unknown as T;
    }
    
    return outputTensor;
  }
  
  /**
   * Synchronize backend execution
   * Ensures all queued operations are complete
   */
  async sync(): Promise<void> {
    // WebNN operations are already synchronous after awaiting compute
    return Promise.resolve();
  }
  
  /**
   * Free all resources associated with this backend
   */
  dispose(): void {
    if (!this.initialized) {
      return;
    }
    
    // Clear graph cache
    this.graphCache.clear();
    
    // Clear tensor operands
    this.tensorOperands.clear();
    
    // Reset context
    this.context = null;
    
    // Reset initialization flag
    this.initialized = false;
  }
}

/**
 * Create and initialize a WebNN backend
 * @returns Initialized WebNN backend
 */
export async function createWebNNBackend(): Promise<WebNNBackend> {
  const backend = new WebNNBackend();
  await backend.initialize();
  return backend;
}