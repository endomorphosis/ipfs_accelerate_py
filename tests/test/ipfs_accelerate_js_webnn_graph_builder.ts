/**
 * WebNN Graph Builder
 * Provides a high-level API for building and executing WebNN neural network graphs
 */

import { WebNNBackend } from './ipfs_accelerate_js_webnn_backend';
import { Tensor } from './ipfs_accelerate_js_tensor';

/**
 * WebNN Graph input configuration
 */
export interface GraphInput {
  name: string;
  tensor: Tensor<any>;
}

/**
 * WebNN Graph output configuration
 */
export interface GraphOutput {
  name: string;
  operand: any;
}

/**
 * Convolution options
 */
export interface ConvolutionOptions {
  strides?: [number, number];
  padding?: [number, number, number, number] | number;
  dilations?: [number, number];
  groups?: number;
  layout?: 'nchw' | 'nhwc';
  activation?: ActivationFunction;
}

/**
 * Pooling options
 */
export interface PoolingOptions {
  windowDimensions: [number, number];
  strides?: [number, number];
  padding?: [number, number, number, number] | number;
  dilations?: [number, number];
  layout?: 'nchw' | 'nhwc';
}

/**
 * GEMM (General Matrix Multiplication) options
 */
export interface GemmOptions {
  alpha?: number;
  beta?: number;
  aTranspose?: boolean;
  bTranspose?: boolean;
  c?: Tensor<any>;
}

/**
 * Activation function
 */
export type ActivationFunction = 'relu' | 'sigmoid' | 'tanh' | 'leakyRelu' | 'none';

/**
 * WebNN Graph Node - represents a node in the computation graph
 */
class GraphNode {
  /** Unique identifier for this node */
  id: string;
  
  /** Operation type */
  operationType: string;
  
  /** Input nodes */
  inputs: GraphNode[];
  
  /** WebNN operand */
  operand: any;
  
  /** Output shape */
  shape: number[];
  
  /** Data type */
  dataType: string;
  
  /**
   * Constructor
   */
  constructor(
    id: string,
    operationType: string,
    operand: any,
    shape: number[],
    dataType: string,
    inputs: GraphNode[] = []
  ) {
    this.id = id;
    this.operationType = operationType;
    this.operand = operand;
    this.shape = shape;
    this.dataType = dataType;
    this.inputs = inputs;
  }
}

/**
 * A caching mechanism for WebNN graphs
 */
class GraphCache {
  /** Map of graph key to compiled graph */
  private cache: Map<string, any> = new Map();
  
  /**
   * Get a cached graph
   * @param key - Cache key
   * @returns Cached graph or undefined
   */
  get(key: string): any {
    return this.cache.get(key);
  }
  
  /**
   * Set a cached graph
   * @param key - Cache key
   * @param graph - Compiled graph
   */
  set(key: string, graph: any): void {
    this.cache.set(key, graph);
  }
  
  /**
   * Check if a key exists in the cache
   * @param key - Cache key
   * @returns Whether the key exists
   */
  has(key: string): boolean {
    return this.cache.has(key);
  }
  
  /**
   * Clear the cache
   */
  clear(): void {
    this.cache.clear();
  }
}

/**
 * WebNN Graph Builder
 * Provides a convenient API for building and executing WebNN neural network graphs
 */
export class WebNNGraphBuilder {
  /** WebNN graph builder instance */
  private graphBuilder: any;
  
  /** WebNN context */
  private context: any;
  
  /** WebNN backend */
  private backend: WebNNBackend;
  
  /** Next node ID */
  private nextNodeId: number = 0;
  
  /** Graph inputs */
  private graphInputs: Map<string, GraphNode> = new Map();
  
  /** Graph outputs */
  private graphOutputs: Map<string, GraphNode> = new Map();
  
  /** Graph nodes */
  private nodes: Map<string, GraphNode> = new Map();
  
  /** Graph cache */
  private cache: GraphCache = new GraphCache();
  
  /** Whether the builder is initialized */
  private initialized: boolean = false;
  
  /**
   * Constructor
   * @param backend - WebNN backend
   */
  constructor(backend: WebNNBackend) {
    this.backend = backend;
  }
  
  /**
   * Initialize the graph builder
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }
    
    if (!this.backend.isInitialized()) {
      await this.backend.initialize();
    }
    
    // Access the context and graph builder from the backend
    this.context = this.backend['context'];
    
    if (!this.context) {
      throw new Error('WebNN context is not available');
    }
    
    try {
      // @ts-ignore - WebNN is not yet in TypeScript standard lib
      this.graphBuilder = await navigator.ml.createGraphBuilder(this.context);
      this.initialized = true;
    } catch (error) {
      console.error('Failed to create WebNN graph builder:', error);
      throw error;
    }
  }
  
  /**
   * Generate a unique node ID
   * @returns Unique node ID
   */
  private generateNodeId(): string {
    return `node_${this.nextNodeId++}`;
  }
  
  /**
   * Add a graph input
   * @param name - Input name
   * @param tensor - Input tensor
   * @returns Graph node
   */
  input(name: string, tensor: Tensor<any>): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    // Create an input operand descriptor
    const descriptor = {
      type: this.convertDataType(tensor.dataType),
      dimensions: tensor.shape
    };
    
    // Create the input operand
    const operand = this.graphBuilder.input(name, descriptor);
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      'input',
      operand,
      tensor.shape,
      tensor.dataType
    );
    
    this.graphInputs.set(name, node);
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Add a constant node to the graph
   * @param tensor - Constant tensor
   * @returns Graph node
   */
  constant(tensor: Tensor<any>): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    // Create an operand descriptor
    const descriptor = {
      type: this.convertDataType(tensor.dataType),
      dimensions: tensor.shape
    };
    
    // Convert tensor data to a typed array
    const typedArray = this.convertToTypedArray(tensor);
    
    // Create the constant operand
    const operand = this.graphBuilder.constant(descriptor, typedArray);
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      'constant',
      operand,
      tensor.shape,
      tensor.dataType
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Set a graph output
   * @param name - Output name
   * @param node - Output node
   */
  output(name: string, node: GraphNode): void {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    this.graphOutputs.set(name, node);
  }
  
  /**
   * Build and compile the graph
   * @returns Compiled graph
   */
  async buildAndCompile(): Promise<any> {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    if (this.graphOutputs.size === 0) {
      throw new Error('Graph must have at least one output');
    }
    
    // Set outputs in the graph builder
    const outputs: Record<string, any> = {};
    this.graphOutputs.forEach((node, name) => {
      outputs[name] = node.operand;
    });
    
    // Build the graph
    try {
      const graph = await this.graphBuilder.build(outputs);
      return graph;
    } catch (error) {
      console.error('Failed to build WebNN graph:', error);
      throw error;
    }
  }
  
  /**
   * Execute the compiled graph
   * @param inputs - Input tensors
   * @param graph - Compiled graph (optional, will build if not provided)
   * @returns Output tensors
   */
  async execute(
    inputs: Record<string, Tensor<any>>,
    graph?: any
  ): Promise<Record<string, Tensor<any>>> {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    // Build the graph if not provided
    if (!graph) {
      graph = await this.buildAndCompile();
    }
    
    // Prepare input operands
    const inputOperands: Record<string, any> = {};
    for (const [name, tensor] of Object.entries(inputs)) {
      inputOperands[name] = this.convertToTypedArray(tensor);
    }
    
    // Prepare output operands
    const outputOperands: Record<string, any> = {};
    for (const [name, node] of this.graphOutputs.entries()) {
      // Calculate output size
      const outputSize = node.shape.reduce((acc, dim) => acc * dim, 1);
      
      // Create output buffer
      outputOperands[name] = this.createTypedArray(node.dataType, outputSize);
    }
    
    // Execute the graph
    try {
      await this.context.compute(graph, inputOperands, outputOperands);
      
      // Convert output buffers to tensors
      const results: Record<string, Tensor<any>> = {};
      for (const [name, node] of this.graphOutputs.entries()) {
        const outputData = outputOperands[name];
        results[name] = new Tensor(
          node.shape,
          Array.from(outputData),
          {
            dataType: node.dataType,
            backend: 'webnn',
            device: this.backend['id']
          }
        );
      }
      
      return results;
    } catch (error) {
      console.error('Failed to execute WebNN graph:', error);
      throw error;
    }
  }
  
  /**
   * Execute graph with caching
   * @param inputs - Input tensors
   * @param cacheKey - Cache key for the graph
   * @returns Output tensors
   */
  async executeWithCache(
    inputs: Record<string, Tensor<any>>,
    cacheKey: string
  ): Promise<Record<string, Tensor<any>>> {
    // Check if the graph is cached
    let graph = this.cache.get(cacheKey);
    
    // If not cached, build and cache the graph
    if (!graph) {
      graph = await this.buildAndCompile();
      this.cache.set(cacheKey, graph);
    }
    
    // Execute the graph
    return this.execute(inputs, graph);
  }
  
  /**
   * Add a convolution operation to the graph
   * @param input - Input node
   * @param filter - Filter weights node
   * @param options - Convolution options
   * @returns Output node
   */
  conv2d(
    input: GraphNode,
    filter: GraphNode,
    options: ConvolutionOptions = {}
  ): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    const {
      strides = [1, 1],
      padding = [0, 0, 0, 0],
      dilations = [1, 1],
      groups = 1,
      layout = 'nhwc',
      activation = 'none'
    } = options;
    
    // Build options for convolution
    const convOptions: Record<string, any> = {
      strides,
      padding,
      dilations,
      groups,
      layout
    };
    
    // Create the convolution operation
    let outputOperand = this.graphBuilder.conv2d(
      input.operand,
      filter.operand,
      convOptions
    );
    
    // Apply activation function if specified
    if (activation !== 'none') {
      outputOperand = this.applyActivation(outputOperand, activation);
    }
    
    // Calculate output shape
    // For NHWC layout: [batch, height, width, channels]
    // For NCHW layout: [batch, channels, height, width]
    let inputHeight, inputWidth, inputChannels, outputChannels, batch;
    let filterHeight, filterWidth;
    
    if (layout === 'nhwc') {
      [batch, inputHeight, inputWidth, inputChannels] = input.shape;
      [filterHeight, filterWidth, inputChannels, outputChannels] = filter.shape;
    } else {
      [batch, inputChannels, inputHeight, inputWidth] = input.shape;
      [outputChannels, inputChannels, filterHeight, filterWidth] = filter.shape;
    }
    
    // Calculate output dimensions
    const paddingTotal = padding instanceof Array 
      ? [padding[0] + padding[2], padding[1] + padding[3]]
      : [padding * 2, padding * 2];
    
    const outputHeight = Math.floor(
      (inputHeight - filterHeight * dilations[0] + paddingTotal[0]) / strides[0] + 1
    );
    const outputWidth = Math.floor(
      (inputWidth - filterWidth * dilations[1] + paddingTotal[1]) / strides[1] + 1
    );
    
    // Output shape matches the layout of input
    const outputShape = layout === 'nhwc'
      ? [batch, outputHeight, outputWidth, outputChannels]
      : [batch, outputChannels, outputHeight, outputWidth];
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      'conv2d',
      outputOperand,
      outputShape,
      input.dataType,
      [input, filter]
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Add a pooling operation to the graph
   * @param input - Input node
   * @param type - Pooling type ('max' or 'average')
   * @param options - Pooling options
   * @returns Output node
   */
  pool(
    input: GraphNode,
    type: 'max' | 'average',
    options: PoolingOptions
  ): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    const {
      windowDimensions,
      strides = [1, 1],
      padding = [0, 0, 0, 0],
      dilations = [1, 1],
      layout = 'nhwc'
    } = options;
    
    // Build options for pooling
    const poolOptions: Record<string, any> = {
      windowDimensions,
      strides,
      padding,
      dilations,
      layout
    };
    
    // Create the pooling operation
    let outputOperand;
    if (type === 'max') {
      outputOperand = this.graphBuilder.maxPool2d(input.operand, poolOptions);
    } else if (type === 'average') {
      outputOperand = this.graphBuilder.averagePool2d(input.operand, poolOptions);
    } else {
      throw new Error(`Unsupported pooling type: ${type}`);
    }
    
    // Calculate output shape
    // For NHWC layout: [batch, height, width, channels]
    // For NCHW layout: [batch, channels, height, width]
    let inputHeight, inputWidth, channels, batch;
    
    if (layout === 'nhwc') {
      [batch, inputHeight, inputWidth, channels] = input.shape;
    } else {
      [batch, channels, inputHeight, inputWidth] = input.shape;
    }
    
    // Calculate output dimensions
    const paddingTotal = padding instanceof Array 
      ? [padding[0] + padding[2], padding[1] + padding[3]]
      : [padding * 2, padding * 2];
    
    const outputHeight = Math.floor(
      (inputHeight - windowDimensions[0] * dilations[0] + paddingTotal[0]) / strides[0] + 1
    );
    const outputWidth = Math.floor(
      (inputWidth - windowDimensions[1] * dilations[1] + paddingTotal[1]) / strides[1] + 1
    );
    
    // Output shape matches the layout of input
    const outputShape = layout === 'nhwc'
      ? [batch, outputHeight, outputWidth, channels]
      : [batch, channels, outputHeight, outputWidth];
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      `${type}Pool`,
      outputOperand,
      outputShape,
      input.dataType,
      [input]
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Add a matrix multiplication (GEMM) operation to the graph
   * @param a - First input node
   * @param b - Second input node
   * @param options - GEMM options
   * @returns Output node
   */
  gemm(
    a: GraphNode,
    b: GraphNode,
    options: GemmOptions = {}
  ): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    // Build the GEMM options
    const gemmOptions: Record<string, any> = {};
    
    if (options.alpha !== undefined) {
      gemmOptions.alpha = options.alpha;
    }
    
    if (options.beta !== undefined) {
      gemmOptions.beta = options.beta;
    }
    
    if (options.aTranspose !== undefined) {
      gemmOptions.aTranspose = options.aTranspose;
    }
    
    if (options.bTranspose !== undefined) {
      gemmOptions.bTranspose = options.bTranspose;
    }
    
    if (options.c) {
      // Add bias tensor C
      const cNode = this.constant(options.c);
      gemmOptions.c = cNode.operand;
    }
    
    // Create the GEMM operation
    const outputOperand = this.graphBuilder.gemm(
      a.operand,
      b.operand,
      gemmOptions
    );
    
    // Calculate output shape
    // GEMM: C = alpha * (A @ B) + beta * C
    // Where A is (M x K), B is (K x N), and C is (M x N)
    
    let M, N, K;
    if (options.aTranspose) {
      [K, M] = a.shape;
    } else {
      [M, K] = a.shape;
    }
    
    if (options.bTranspose) {
      [N, K] = b.shape;
    } else {
      [K, N] = b.shape;
    }
    
    const outputShape = [M, N];
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      'gemm',
      outputOperand,
      outputShape,
      a.dataType,
      [a, b]
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Add a matrix multiplication to the graph
   * @param a - First input node
   * @param b - Second input node
   * @returns Output node
   */
  matmul(a: GraphNode, b: GraphNode): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    // Create the matmul operation
    const outputOperand = this.graphBuilder.matmul(a.operand, b.operand);
    
    // Calculate output shape for matmul: (M x K) @ (K x N) = (M x N)
    const M = a.shape[0];
    const N = b.shape[1];
    const outputShape = [M, N];
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      'matmul',
      outputOperand,
      outputShape,
      a.dataType,
      [a, b]
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Add a normalization operation to the graph
   * @param input - Input node
   * @param mean - Mean values
   * @param variance - Variance values
   * @param scale - Scale values (optional)
   * @param bias - Bias values (optional)
   * @param epsilon - Epsilon value for numerical stability
   * @returns Output node
   */
  batchNormalization(
    input: GraphNode,
    mean: GraphNode,
    variance: GraphNode,
    scale?: GraphNode,
    bias?: GraphNode,
    epsilon: number = 1e-5
  ): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    // Build the batch norm options
    const bnOptions: Record<string, any> = {
      epsilon
    };
    
    if (scale) {
      bnOptions.scale = scale.operand;
    }
    
    if (bias) {
      bnOptions.bias = bias.operand;
    }
    
    // Create the batch normalization operation
    const outputOperand = this.graphBuilder.batchNormalization(
      input.operand,
      mean.operand,
      variance.operand,
      bnOptions
    );
    
    // Batch normalization preserves the input shape
    const outputShape = [...input.shape];
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      'batchNormalization',
      outputOperand,
      outputShape,
      input.dataType,
      [input, mean, variance, ...(scale ? [scale] : []), ...(bias ? [bias] : [])]
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Add an element-wise addition to the graph
   * @param a - First input node
   * @param b - Second input node
   * @returns Output node
   */
  add(a: GraphNode, b: GraphNode): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    // Create the add operation
    const outputOperand = this.graphBuilder.add(a.operand, b.operand);
    
    // Determine output shape (accounting for broadcasting)
    const outputShape = this.calculateBroadcastShape(a.shape, b.shape);
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      'add',
      outputOperand,
      outputShape,
      a.dataType,
      [a, b]
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Add an element-wise subtraction to the graph
   * @param a - First input node
   * @param b - Second input node
   * @returns Output node
   */
  sub(a: GraphNode, b: GraphNode): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    // Create the sub operation
    const outputOperand = this.graphBuilder.sub(a.operand, b.operand);
    
    // Determine output shape (accounting for broadcasting)
    const outputShape = this.calculateBroadcastShape(a.shape, b.shape);
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      'sub',
      outputOperand,
      outputShape,
      a.dataType,
      [a, b]
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Add an element-wise multiplication to the graph
   * @param a - First input node
   * @param b - Second input node
   * @returns Output node
   */
  mul(a: GraphNode, b: GraphNode): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    // Create the mul operation
    const outputOperand = this.graphBuilder.mul(a.operand, b.operand);
    
    // Determine output shape (accounting for broadcasting)
    const outputShape = this.calculateBroadcastShape(a.shape, b.shape);
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      'mul',
      outputOperand,
      outputShape,
      a.dataType,
      [a, b]
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Add an element-wise division to the graph
   * @param a - First input node
   * @param b - Second input node
   * @returns Output node
   */
  div(a: GraphNode, b: GraphNode): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    // Create the div operation
    const outputOperand = this.graphBuilder.div(a.operand, b.operand);
    
    // Determine output shape (accounting for broadcasting)
    const outputShape = this.calculateBroadcastShape(a.shape, b.shape);
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      'div',
      outputOperand,
      outputShape,
      a.dataType,
      [a, b]
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Add a reshape operation to the graph
   * @param input - Input node
   * @param newShape - New shape
   * @returns Output node
   */
  reshape(input: GraphNode, newShape: number[]): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    // Create the reshape operation
    const outputOperand = this.graphBuilder.reshape(input.operand, newShape);
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      'reshape',
      outputOperand,
      newShape,
      input.dataType,
      [input]
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Add a transpose operation to the graph
   * @param input - Input node
   * @param permutation - Permutation of dimensions
   * @returns Output node
   */
  transpose(input: GraphNode, permutation?: number[]): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    // If permutation is not specified, reverse the dimensions
    if (!permutation) {
      permutation = Array.from(input.shape.keys()).reverse();
    }
    
    // Create the transpose operation
    const outputOperand = this.graphBuilder.transpose(input.operand, permutation);
    
    // Calculate the output shape
    const outputShape = permutation.map(p => input.shape[p]);
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      'transpose',
      outputOperand,
      outputShape,
      input.dataType,
      [input]
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Add an activation function to the graph
   * @param input - Input node
   * @param activation - Activation function
   * @returns Output node
   */
  activation(input: GraphNode, activation: ActivationFunction): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    // Create the activation operation
    const outputOperand = this.applyActivation(input.operand, activation);
    
    // Activation functions preserve shape
    const outputShape = [...input.shape];
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      activation,
      outputOperand,
      outputShape,
      input.dataType,
      [input]
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Add a concatenation operation to the graph
   * @param inputs - Input nodes
   * @param axis - Concatenation axis
   * @returns Output node
   */
  concat(inputs: GraphNode[], axis: number): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    if (inputs.length < 2) {
      throw new Error('Concat requires at least two inputs');
    }
    
    // Create the concat operation
    const outputOperand = this.graphBuilder.concat(
      inputs.map(input => input.operand),
      axis
    );
    
    // Calculate output shape
    const baseShape = [...inputs[0].shape];
    let concatDimSize = baseShape[axis];
    
    for (let i = 1; i < inputs.length; i++) {
      concatDimSize += inputs[i].shape[axis];
    }
    
    baseShape[axis] = concatDimSize;
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      'concat',
      outputOperand,
      baseShape,
      inputs[0].dataType,
      inputs
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Add a softmax operation to the graph
   * @param input - Input node
   * @param axis - Axis to apply softmax on
   * @returns Output node
   */
  softmax(input: GraphNode, axis: number = -1): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    // Handle negative axis
    if (axis < 0) {
      axis = input.shape.length + axis;
    }
    
    // Create the softmax operation
    const outputOperand = this.graphBuilder.softmax(
      input.operand,
      { axis }
    );
    
    // Softmax preserves shape
    const outputShape = [...input.shape];
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      'softmax',
      outputOperand,
      outputShape,
      input.dataType,
      [input]
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Add a fully connected layer to the graph
   * @param input - Input node
   * @param weights - Weights node
   * @param bias - Bias node (optional)
   * @param activation - Activation function (optional)
   * @returns Output node
   */
  fullyConnected(
    input: GraphNode,
    weights: GraphNode,
    bias?: GraphNode,
    activation: ActivationFunction = 'none'
  ): GraphNode {
    if (!this.initialized) {
      throw new Error('WebNN graph builder not initialized');
    }
    
    // Reshape input if needed to 2D
    let reshapedInput = input;
    const originalShape = input.shape;
    
    if (input.shape.length > 2) {
      // Flatten all dimensions except the batch dimension
      const batchSize = input.shape[0];
      const flattenedSize = input.shape.slice(1).reduce((a, b) => a * b, 1);
      reshapedInput = this.reshape(input, [batchSize, flattenedSize]);
    }
    
    // Create the matmul operation
    let outputOperand = this.graphBuilder.matmul(reshapedInput.operand, weights.operand);
    
    // Add bias if provided
    if (bias) {
      outputOperand = this.graphBuilder.add(outputOperand, bias.operand);
    }
    
    // Apply activation if specified
    if (activation !== 'none') {
      outputOperand = this.applyActivation(outputOperand, activation);
    }
    
    // Calculate output shape
    const outputFeatures = weights.shape[1];
    let outputShape: number[];
    
    if (input.shape.length > 2) {
      // Restore the batch dimension and append the output features
      outputShape = [originalShape[0], outputFeatures];
    } else {
      outputShape = [input.shape[0], outputFeatures];
    }
    
    // Create and store the node
    const node = new GraphNode(
      this.generateNodeId(),
      'fullyConnected',
      outputOperand,
      outputShape,
      input.dataType,
      [input, weights, ...(bias ? [bias] : [])]
    );
    
    this.nodes.set(node.id, node);
    
    return node;
  }
  
  /**
   * Create a neural network layer with weights, bias, and activation
   * @param input - Input node
   * @param weights - Weights tensor
   * @param bias - Bias tensor (optional)
   * @param activation - Activation function (optional)
   * @returns Output node
   */
  layer(
    input: GraphNode,
    weights: Tensor<any>,
    bias?: Tensor<any>,
    activation: ActivationFunction = 'none'
  ): GraphNode {
    // Create constant nodes for weights and bias
    const weightsNode = this.constant(weights);
    const biasNode = bias ? this.constant(bias) : undefined;
    
    // Create the fully connected layer
    return this.fullyConnected(input, weightsNode, biasNode, activation);
  }
  
  /**
   * Apply a neural network residual block
   * @param input - Input node
   * @param fc1Weights - First fully connected layer weights
   * @param fc1Bias - First fully connected layer bias
   * @param fc2Weights - Second fully connected layer weights
   * @param fc2Bias - Second fully connected layer bias
   * @param activation - Activation function
   * @returns Output node
   */
  residualBlock(
    input: GraphNode,
    fc1Weights: Tensor<any>,
    fc1Bias: Tensor<any>,
    fc2Weights: Tensor<any>,
    fc2Bias: Tensor<any>,
    activation: ActivationFunction = 'relu'
  ): GraphNode {
    // First fully connected layer with activation
    const fc1 = this.layer(input, fc1Weights, fc1Bias, activation);
    
    // Second fully connected layer without activation
    const fc2 = this.layer(fc1, fc2Weights, fc2Bias, 'none');
    
    // Add residual connection
    const residual = this.add(input, fc2);
    
    // Apply activation after residual connection
    return this.activation(residual, activation);
  }
  
  /**
   * Create a sequential neural network model
   * @param input - Input node
   * @param layers - Array of layer definitions
   * @returns Output node
   */
  sequential(
    input: GraphNode,
    layers: Array<{
      weights: Tensor<any>;
      bias?: Tensor<any>;
      activation?: ActivationFunction;
    }>
  ): GraphNode {
    let current = input;
    
    // Apply each layer sequentially
    for (const layer of layers) {
      current = this.layer(
        current,
        layer.weights,
        layer.bias,
        layer.activation || 'none'
      );
    }
    
    return current;
  }
  
  // ======== Helper Methods ========
  
  /**
   * Apply an activation function to an operand
   * @param operand - Input operand
   * @param activation - Activation function
   * @returns Output operand
   */
  private applyActivation(operand: any, activation: ActivationFunction): any {
    switch (activation) {
      case 'relu':
        return this.graphBuilder.relu(operand);
        
      case 'sigmoid':
        return this.graphBuilder.sigmoid(operand);
        
      case 'tanh':
        return this.graphBuilder.tanh(operand);
        
      case 'leakyRelu':
        // WebNN doesn't have a built-in leakyRelu, so implement it
        // leakyRelu(x) = x if x > 0, alpha * x if x <= 0
        const alpha = 0.01;
        
        // Create a constant for alpha
        const alphaOperand = this.graphBuilder.constant(
          { type: 'float32', dimensions: [1] },
          new Float32Array([alpha])
        );
        
        // x * alpha
        const alphaX = this.graphBuilder.mul(operand, alphaOperand);
        
        // max(x, alpha * x)
        return this.graphBuilder.max(operand, alphaX);
        
      case 'none':
      default:
        return operand;
    }
  }
  
  /**
   * Calculate broadcast shape for binary operations
   * @param shapeA - First shape
   * @param shapeB - Second shape
   * @returns Broadcast shape
   */
  private calculateBroadcastShape(shapeA: number[], shapeB: number[]): number[] {
    const result = [];
    const maxLength = Math.max(shapeA.length, shapeB.length);
    
    // Pad shapes with 1s if needed
    const paddedA = Array(maxLength - shapeA.length).fill(1).concat(shapeA);
    const paddedB = Array(maxLength - shapeB.length).fill(1).concat(shapeB);
    
    // Calculate output shape
    for (let i = 0; i < maxLength; i++) {
      result.push(Math.max(paddedA[i], paddedB[i]));
    }
    
    return result;
  }
  
  /**
   * Convert tensor data type to WebNN data type
   * @param dataType - Tensor data type
   * @returns WebNN data type
   */
  private convertDataType(dataType: string): string {
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
   * Convert tensor to typed array
   * @param tensor - Input tensor
   * @returns Typed array
   */
  private convertToTypedArray(tensor: Tensor<any>): ArrayBufferView {
    switch (tensor.dataType) {
      case 'float32':
        return new Float32Array(tensor.data);
      case 'int32':
        return new Int32Array(tensor.data);
      case 'float64':
        // Convert to float32 for WebNN compatibility
        const float64Data = tensor.data as number[];
        const float32Data = new Float32Array(float64Data.length);
        for (let i = 0; i < float64Data.length; i++) {
          float32Data[i] = Number(float64Data[i]);
        }
        return float32Data;
      case 'uint8':
        return new Uint8Array(tensor.data);
      case 'int8':
        return new Int8Array(tensor.data);
      default:
        // Default to float32
        return new Float32Array(tensor.data);
    }
  }
  
  /**
   * Create a typed array for a specific data type
   * @param dataType - Data type
   * @param size - Array size
   * @returns Typed array
   */
  private createTypedArray(dataType: string, size: number): ArrayBufferView {
    switch (dataType) {
      case 'float32':
        return new Float32Array(size);
      case 'int32':
        return new Int32Array(size);
      case 'float64':
        // WebNN doesn't support float64, so use float32
        return new Float32Array(size);
      case 'uint8':
        return new Uint8Array(size);
      case 'int8':
        return new Int8Array(size);
      default:
        return new Float32Array(size);
    }
  }
  
  /**
   * Reset the graph builder
   */
  reset(): void {
    this.nextNodeId = 0;
    this.graphInputs.clear();
    this.graphOutputs.clear();
    this.nodes.clear();
  }
  
  /**
   * Clear the cache
   */
  clearCache(): void {
    this.cache.clear();
  }
  
  /**
   * Dispose of resources
   */
  dispose(): void {
    this.reset();
    this.clearCache();
    this.initialized = false;
  }
}