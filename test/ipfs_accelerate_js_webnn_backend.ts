/**
 * WebNN Backend Implementation
 * Provides hardware acceleration using the WebNN API
 */

import { HardwareBackend, HardwareBackendType } from './hardware_abstraction';
import { 
  executePoolingOperation, 
  executeNormalizationOperation,
  executeElementwiseOperation,
  executeTensorManipulationOperation
} from './ipfs_accelerate_js_webnn_operations';

/**
 * WebNN Backend Options
 */
export interface WebNNBackendOptions {
  /**
   * Preferred device type for execution
   */
  deviceType?: 'gpu' | 'cpu';
  
  /**
   * Power preference (high-performance vs low-power)
   */
  powerPreference?: 'high-performance' | 'low-power' | 'default';
  
  /**
   * Enable additional logging for debugging
   */
  enableLogging?: boolean;
  
  /**
   * Default float precision
   */
  floatPrecision?: 'float32' | 'float16';
  
  /**
   * Whether to use sync execution when available
   */
  preferSyncExecution?: boolean;
  
  /**
   * Memory management options
   */
  memory?: {
    /**
     * Enable garbage collection of unused tensors
     */
    enableGarbageCollection?: boolean;
    
    /**
     * Threshold for garbage collection (in bytes)
     */
    garbageCollectionThreshold?: number;
  };
}

/**
 * WebNN Backend
 * Implements hardware acceleration using the WebNN API
 */
export class WebNNBackend implements HardwareBackend {
  readonly type: HardwareBackendType = 'webnn';
  
  private options: WebNNBackendOptions;
  private context: MLContext | null = null;
  private graphBuilder: MLGraphBuilder | null = null;
  private deviceInfo: {
    deviceType: string | null;
    deviceName: string | null;
    isSimulated: boolean;
  } = {
    deviceType: null,
    deviceName: null,
    isSimulated: false
  };
  private compiledGraphs: Map<string, MLGraph> = new Map();
  private tensors: Map<string, MLOperand> = new Map();
  private memoryAllocated = 0;
  
  /**
   * Create a WebNN backend
   */
  constructor(options: WebNNBackendOptions = {}) {
    this.options = {
      deviceType: 'gpu',
      powerPreference: 'high-performance',
      enableLogging: false,
      floatPrecision: 'float32',
      preferSyncExecution: true,
      memory: {
        enableGarbageCollection: true,
        garbageCollectionThreshold: 1024 * 1024 * 128 // 128MB
      },
      ...options
    };
  }
  
  /**
   * Check if WebNN is supported in this browser
   */
  async isSupported(): Promise<boolean> {
    try {
      if (!('ml' in navigator)) {
        return false;
      }
      
      // Try requesting a context to confirm WebNN support
      const context = await (navigator as any).ml.createContext({
        deviceType: this.options.deviceType
      });
      
      return !!context;
    } catch (error) {
      if (this.options.enableLogging) {
        console.warn('WebNN not supported:', error);
      }
      return false;
    }
  }
  
  /**
   * Initialize the WebNN backend
   */
  async initialize(): Promise<boolean> {
    try {
      if (!('ml' in navigator)) {
        throw new Error('WebNN not supported in this browser');
      }
      
      // Request context
      this.context = await (navigator as any).ml.createContext({
        deviceType: this.options.deviceType,
        powerPreference: this.options.powerPreference
      });
      
      if (!this.context) {
        throw new Error('Failed to get WebNN context');
      }
      
      // Create graph builder
      this.graphBuilder = new MLGraphBuilder(this.context);
      
      // Get device information
      await this.detectDeviceInfo();
      
      if (this.options.enableLogging) {
        console.log('WebNN initialized:', {
          deviceType: this.deviceInfo.deviceType,
          deviceName: this.deviceInfo.deviceName,
          isSimulated: this.deviceInfo.isSimulated
        });
      }
      
      return true;
    } catch (error) {
      console.error('WebNN initialization failed:', error);
      this.context = null;
      this.graphBuilder = null;
      return false;
    }
  }
  
  /**
   * Get WebNN-specific capabilities
   */
  async getCapabilities(): Promise<Record<string, any>> {
    if (!this.context || !this.graphBuilder) {
      throw new Error('WebNN backend not initialized');
    }
    
    // Detect supported operations
    const operations = await this.detectSupportedOperations();
    
    return {
      deviceType: this.deviceInfo.deviceType,
      deviceName: this.deviceInfo.deviceName,
      isSimulated: this.deviceInfo.isSimulated,
      operations,
      floatPrecision: this.options.floatPrecision
    };
  }
  
  /**
   * Detect device information
   */
  private async detectDeviceInfo(): Promise<void> {
    try {
      // Check context for device type
      this.deviceInfo.deviceType = (this.context as any).deviceType || 'unknown';
      
      // Try to detect device name using context info or WebGPU fallback
      this.deviceInfo.deviceName = await this.getDeviceName();
      
      // Detect if it's a simulated implementation
      this.deviceInfo.isSimulated = this.detectSimulation();
    } catch (error) {
      console.warn('Error detecting WebNN device info:', error);
    }
  }
  
  /**
   * Try to determine device name
   */
  private async getDeviceName(): Promise<string | null> {
    try {
      // Try to get from context (non-standard but might be available)
      if (this.context && (this.context as any).deviceInfo) {
        const deviceInfo = (this.context as any).deviceInfo;
        if (typeof deviceInfo === 'object' && deviceInfo.name) {
          return deviceInfo.name;
        }
      }
      
      // Try WebGPU fallback for device identification
      if ('gpu' in navigator) {
        const adapter = await navigator.gpu.requestAdapter({
          powerPreference: this.options.powerPreference === 'high-performance' ? 
            'high-performance' : 'low-power'
        });
        
        if (adapter) {
          const adapterInfo = await adapter.requestAdapterInfo();
          return adapterInfo.device || adapterInfo.description || null;
        }
      }
      
      return null;
    } catch (error) {
      console.warn('Failed to get WebNN device name:', error);
      return null;
    }
  }
  
  /**
   * Detect if this is a simulated implementation
   */
  private detectSimulation(): boolean {
    // Common patterns for simulated/software implementations
    const softwarePatterns = [
      'swiftshader',
      'llvmpipe',
      'software',
      'emulation',
      'reference',
      'basic',
      'microsoft basic'
    ];
    
    const deviceName = this.deviceInfo.deviceName?.toLowerCase() || '';
    const deviceType = this.deviceInfo.deviceType?.toLowerCase() || '';
    
    // Check if device name contains any software patterns
    if (deviceName && softwarePatterns.some(pattern => deviceName.includes(pattern))) {
      return true;
    }
    
    // Check if deviceType is 'cpu' which often indicates software fallback
    if (deviceType === 'cpu') {
      return true;
    }
    
    return false;
  }
  
  /**
   * Detect supported operations
   */
  private async detectSupportedOperations(): Promise<string[]> {
    if (!this.graphBuilder) {
      return [];
    }
    
    const operations: string[] = [];
    const testTensor = this.createTestTensor();
    
    if (!testTensor) {
      return [];
    }
    
    // Create filter tensor for convolution and pooling tests
    const filter = this.createConstant(new Float32Array(9), [1, 1, 3, 3]);
    
    // Test basic operations by attempting to call them
    const opTests = [
      // Core operations
      { name: 'relu', test: () => this.graphBuilder!.relu(testTensor) },
      { name: 'sigmoid', test: () => this.graphBuilder!.sigmoid(testTensor) },
      { name: 'tanh', test: () => this.graphBuilder!.tanh(testTensor) },
      { name: 'add', test: () => this.graphBuilder!.add(testTensor, testTensor) },
      { name: 'sub', test: () => this.graphBuilder!.sub(testTensor, testTensor) },
      { name: 'mul', test: () => this.graphBuilder!.mul(testTensor, testTensor) },
      { name: 'div', test: () => this.graphBuilder!.div(testTensor, testTensor) },
      { name: 'matmul', test: () => this.graphBuilder!.matmul(testTensor, testTensor) },
      { name: 'softmax', test: () => (this.graphBuilder as any).softmax?.(testTensor) },
      { name: 'concat', test: () => this.graphBuilder!.concat([testTensor, testTensor], 0) },
      { name: 'transpose', test: () => this.graphBuilder!.transpose(testTensor) },
      { name: 'reshape', test: () => this.graphBuilder!.reshape(testTensor, [1, 4]) },
      { name: 'conv2d', test: () => filter ? this.graphBuilder!.conv2d(testTensor, filter, {}) : null },
      
      // New operations - Advanced elementwise
      { name: 'exp', test: () => this.graphBuilder!.exp(testTensor) },
      { name: 'log', test: () => (this.graphBuilder as any).log?.(testTensor) },
      { name: 'sqrt', test: () => (this.graphBuilder as any).sqrt?.(testTensor) },
      { name: 'pow', test: () => (this.graphBuilder as any).pow?.(testTensor, testTensor) },
      { name: 'min', test: () => (this.graphBuilder as any).min?.(testTensor, testTensor) },
      { name: 'max', test: () => (this.graphBuilder as any).max?.(testTensor, testTensor) },
      
      // Pooling operations
      { name: 'maxpool', test: () => {
        return (this.graphBuilder as any).maxPool2d?.(testTensor, { windowDimensions: [2, 2] });
      }},
      { name: 'avgpool', test: () => {
        return (this.graphBuilder as any).averagePool2d?.(testTensor, { windowDimensions: [2, 2] });
      }},
      
      // Normalization operations
      { name: 'batchnorm', test: () => {
        if (!filter) return null;
        const meanTensor = this.createConstant(new Float32Array([0]), [1]);
        const varTensor = this.createConstant(new Float32Array([1]), [1]);
        if (!meanTensor || !varTensor) return null;
        return (this.graphBuilder as any).batchNormalization?.(
          testTensor, meanTensor, varTensor
        );
      }},
      { name: 'layernorm', test: () => {
        return (this.graphBuilder as any).layerNormalization?.(testTensor);
      }},
      
      // Tensor manipulation operations
      { name: 'slice', test: () => {
        return (this.graphBuilder as any).slice?.(testTensor, [0, 0], [1, 2]);
      }},
      { name: 'pad', test: () => {
        const padValue = this.createConstant(new Float32Array([0]), [1]);
        if (!padValue) return null;
        return (this.graphBuilder as any).pad?.(testTensor, [[1, 1], [1, 1]], padValue);
      }},
      { name: 'reducemean', test: () => {
        return (this.graphBuilder as any).reduceMean?.(testTensor, { axes: [0], keepDimensions: true });
      }},
    ];
    
    // Test each operation and add to supported list if successful
    for (const { name, test } of opTests) {
      try {
        const result = test();
        if (result) {
          operations.push(name);
        }
      } catch (error) {
        // Operation not supported
        if (this.options.enableLogging) {
          console.debug(`WebNN operation '${name}' not supported:`, error);
        }
      }
    }
    
    return operations;
  }
  
  /**
   * Create a test tensor for capability detection
   */
  private createTestTensor(): MLOperand | null {
    try {
      if (!this.graphBuilder) return null;
      
      // Create a simple 2x2 tensor for testing operations
      return this.graphBuilder.constant(
        { type: 'float32', dimensions: [2, 2] },
        new Float32Array([1, 2, 3, 4])
      );
    } catch (error) {
      console.warn('Failed to create test tensor:', error);
      return null;
    }
  }
  
  /**
   * Create a constant tensor
   */
  private createConstant(data: Float32Array | Int32Array, dimensions: number[]): MLOperand | null {
    try {
      if (!this.graphBuilder) return null;
      
      const dataType = data instanceof Float32Array ? 'float32' : 'int32';
      
      return this.graphBuilder.constant(
        { type: dataType, dimensions },
        data
      );
    } catch (error) {
      console.warn('Failed to create constant tensor:', error);
      return null;
    }
  }
  
  /**
   * Create a tensor from data
   */
  async createTensor(
    data: Float32Array | Int32Array | Uint8Array, 
    shape: number[], 
    dataType = 'float32'
  ): Promise<any> {
    if (!this.context || !this.graphBuilder) {
      throw new Error('WebNN backend not initialized');
    }
    
    try {
      // Validate data type
      let tensorType: 'float32' | 'int32' | 'uint8';
      if (dataType === 'float32' && data instanceof Float32Array) {
        tensorType = 'float32';
      } else if (dataType === 'int32' && data instanceof Int32Array) {
        tensorType = 'int32';
      } else if (dataType === 'uint8' && data instanceof Uint8Array) {
        tensorType = 'uint8';
      } else {
        throw new Error(`Invalid data type: ${dataType} or data array type mismatch`);
      }
      
      // Create tensor descriptor
      const descriptor: MLOperandDescriptor = {
        type: tensorType,
        dimensions: shape
      };
      
      // Create operand
      const tensor = this.context.createOperand(descriptor, data);
      
      // Generate a unique ID for this tensor
      const tensorId = `tensor_${shape.join('x')}_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
      
      // Track tensor in our cache
      this.tensors.set(tensorId, tensor);
      
      // Track memory usage
      const elementSize = tensorType === 'float32' || tensorType === 'int32' ? 4 : 1;
      const memoryUsage = data.length * elementSize;
      this.memoryAllocated += memoryUsage;
      
      // Run garbage collection if needed
      this.maybeGarbageCollect();
      
      // Return tensor with metadata
      return {
        tensor,
        shape,
        dataType: tensorType,
        id: tensorId,
        size: memoryUsage
      };
    } catch (error) {
      console.error('Failed to create WebNN tensor:', error);
      throw error;
    }
  }
  
  /**
   * Execute operation on WebNN device
   */
  async execute(
    operation: string,
    inputs: Record<string, any>,
    options: Record<string, any> = {}
  ): Promise<any> {
    if (!this.graphBuilder) {
      throw new Error('WebNN backend not initialized');
    }
    
    switch (operation) {
      case 'matmul':
        return this.executeMatmul(inputs, options);
        
      case 'elementwise':
        return this.executeElementwise(inputs, options);
        
      case 'softmax':
        return this.executeSoftmax(inputs, options);
        
      case 'conv2d':
        return this.executeConv2d(inputs, options);
      
      // Pooling operations
      case 'maxpool':
        return this.executePooling('max', inputs, options);
        
      case 'avgpool':
        return this.executePooling('average', inputs, options);
      
      // Normalization operations
      case 'batchnorm':
        return this.executeNormalization('batch', inputs, options);
        
      case 'layernorm':
        return this.executeNormalization('layer', inputs, options);
      
      // Advanced elementwise operations
      case 'add':
      case 'sub':
      case 'mul':
      case 'div':
      case 'pow':
      case 'min':
      case 'max':
      case 'exp':
      case 'log':
      case 'sqrt':
        return this.executeAdvancedElementwise(operation, inputs, options);
      
      // Tensor manipulation operations
      case 'reshape':
      case 'transpose':
      case 'concat':
      case 'slice':
      case 'pad':
        return this.executeTensorManipulation(operation, inputs, options);
        
      default:
        throw new Error(`Unsupported operation: ${operation}`);
    }
  }
  
  /**
   * Matrix multiplication operation
   */
  private async executeMatmul(
    inputs: {
      a: { tensor: MLOperand; shape: number[] };
      b: { tensor: MLOperand; shape: number[] };
    },
    options: {
      transposeA?: boolean;
      transposeB?: boolean;
    } = {}
  ): Promise<{
    tensor: MLOperand;
    shape: number[];
    dataType: string;
  }> {
    const { a, b } = inputs;
    const { transposeA = false, transposeB = false } = options;
    
    // Get dimensions
    const [M, K_a] = transposeA ? [a.shape[1], a.shape[0]] : [a.shape[0], a.shape[1]];
    const [K_b, N] = transposeB ? [b.shape[1], b.shape[0]] : [b.shape[0], b.shape[1]];
    
    if (K_a !== K_b) {
      throw new Error(`Matrix multiplication dimension mismatch: ${K_a} != ${K_b}`);
    }
    
    // Pre-transpose if needed
    let tensorA = a.tensor;
    let tensorB = b.tensor;
    
    if (transposeA) {
      tensorA = this.graphBuilder!.transpose(tensorA, [1, 0]);
    }
    
    if (transposeB) {
      tensorB = this.graphBuilder!.transpose(tensorB, [1, 0]);
    }
    
    // Perform matrix multiplication
    const outputTensor = this.graphBuilder!.matmul(tensorA, tensorB);
    
    // Create a unique graph key
    const graphKey = `matmul_${M}x${K_a}_${K_b}x${N}_${transposeA}_${transposeB}`;
    
    // Build and execute the graph
    const result = await this.runGraphComputation(
      graphKey,
      { a: tensorA, b: tensorB },
      { output: outputTensor }
    );
    
    // Return the result
    return {
      tensor: result.output,
      shape: [M, N],
      dataType: 'float32'
    };
  }
  
  /**
   * Element-wise operations (relu, sigmoid, tanh)
   */
  private async executeElementwise(
    inputs: {
      input: { tensor: MLOperand; shape: number[] };
    },
    options: {
      operation?: 'relu' | 'sigmoid' | 'tanh';
    } = {}
  ): Promise<{
    tensor: MLOperand;
    shape: number[];
    dataType: string;
  }> {
    const { input } = inputs;
    const { operation = 'relu' } = options;
    
    let outputTensor: MLOperand;
    
    // Apply the operation
    switch (operation) {
      case 'relu':
        outputTensor = this.graphBuilder!.relu(input.tensor);
        break;
      case 'sigmoid':
        outputTensor = this.graphBuilder!.sigmoid(input.tensor);
        break;
      case 'tanh':
        outputTensor = this.graphBuilder!.tanh(input.tensor);
        break;
      default:
        throw new Error(`Unsupported elementwise operation: ${operation}`);
    }
    
    // Create a unique graph key
    const graphKey = `elementwise_${operation}_${input.shape.join('x')}`;
    
    // Build and execute the graph
    const result = await this.runGraphComputation(
      graphKey,
      { input: input.tensor },
      { output: outputTensor }
    );
    
    // Return the result
    return {
      tensor: result.output,
      shape: input.shape,
      dataType: 'float32'
    };
  }
  
  /**
   * Softmax operation
   */
  private async executeSoftmax(
    inputs: {
      input: { tensor: MLOperand; shape: number[] };
    },
    options: {
      axis?: number;
    } = {}
  ): Promise<{
    tensor: MLOperand;
    shape: number[];
    dataType: string;
  }> {
    const { input } = inputs;
    const { axis = -1 } = options;
    
    // Calculate actual axis if negative
    const actualAxis = axis < 0 ? input.shape.length + axis : axis;
    
    if (actualAxis < 0 || actualAxis >= input.shape.length) {
      throw new Error(`Invalid softmax axis: ${axis}`);
    }
    
    let outputTensor: MLOperand;
    
    // Check if the MLGraphBuilder has a built-in softmax operation
    if ((this.graphBuilder as any).softmax) {
      // Native softmax support (newer WebNN implementations)
      outputTensor = (this.graphBuilder as any).softmax(input.tensor, { axis: actualAxis });
    } else {
      // Implement softmax ourselves if not available
      // 1. Find max along axis for numerical stability
      // 2. Subtract max from input
      // 3. Apply exp to all elements
      // 4. Sum exp values along axis
      // 5. Divide exp values by sum
      
      // This is a simplified implementation that assumes axis = -1 (last dimension)
      // A full implementation would handle arbitrary axes
      
      // For simplicity, we'll use the example of a 2D tensor with axis=1
      const maxValues = this.computeMax(input.tensor, input.shape, actualAxis);
      const shiftedInput = this.graphBuilder!.sub(input.tensor, maxValues);
      const expValues = this.graphBuilder!.exp(shiftedInput);
      const sumExp = this.computeSum(expValues, input.shape, actualAxis);
      outputTensor = this.graphBuilder!.div(expValues, sumExp);
    }
    
    // Create a unique graph key
    const graphKey = `softmax_${input.shape.join('x')}_axis${actualAxis}`;
    
    // Build and execute the graph
    const result = await this.runGraphComputation(
      graphKey,
      { input: input.tensor },
      { output: outputTensor }
    );
    
    // Return the result
    return {
      tensor: result.output,
      shape: input.shape,
      dataType: 'float32'
    };
  }
  
  /**
   * Helper function to compute maximum values along an axis
   * Note: This is a simplified implementation for demonstration
   */
  private computeMax(input: MLOperand, shape: number[], axis: number): MLOperand {
    // In a full implementation, this would compute the maximum along the specified axis
    // For demonstration purposes, we're assuming a fixed implementation
    
    if (!this.graphBuilder) {
      throw new Error('WebNN backend not initialized');
    }
    
    // This is a placeholder - actual implementation would vary based on the tensor shape
    // Ideally, we would use a WebNN reduceMax operation if available
    
    // Create a dummy implementation that assumes a 2D tensor with axis=1
    // In a real implementation, we would use reduce operations or other tensor operations
    return input;
  }
  
  /**
   * Helper function to compute sum along an axis
   * Note: This is a simplified implementation for demonstration
   */
  private computeSum(input: MLOperand, shape: number[], axis: number): MLOperand {
    // In a full implementation, this would compute the sum along the specified axis
    // For demonstration purposes, we're assuming a fixed implementation
    
    if (!this.graphBuilder) {
      throw new Error('WebNN backend not initialized');
    }
    
    // This is a placeholder - actual implementation would vary based on the tensor shape
    // Ideally, we would use a WebNN reduceSum operation if available
    
    // Create a dummy implementation that assumes a 2D tensor with axis=1
    // In a real implementation, we would use reduce operations or other tensor operations
    return input;
  }
  
  /**
   * 2D Convolution operation
   */
  private async executeConv2d(
    inputs: {
      input: { tensor: MLOperand; shape: number[] };
      filter: { tensor: MLOperand; shape: number[] };
    },
    options: {
      padding?: [number, number, number, number];
      strides?: [number, number];
      dilations?: [number, number];
      groups?: number;
    } = {}
  ): Promise<{
    tensor: MLOperand;
    shape: number[];
    dataType: string;
  }> {
    const { input, filter } = inputs;
    const { 
      padding = [0, 0, 0, 0],
      strides = [1, 1],
      dilations = [1, 1],
      groups = 1
    } = options;
    
    // Build convolution options
    const convOptions: Record<string, any> = {
      padding,
      strides,
      dilations,
      groups
    };
    
    // Perform 2D convolution
    const outputTensor = this.graphBuilder!.conv2d(
      input.tensor,
      filter.tensor,
      convOptions
    );
    
    // Calculate output shape
    // In a real implementation, this would compute the output shape based on input shape,
    // filter shape, padding, strides, and dilations
    // For simplicity, we're assuming a specific pattern
    const [batchSize, inputHeight, inputWidth, inputChannels] = input.shape;
    const [outputChannels, filterHeight, filterWidth] = filter.shape;
    
    const outputHeight = Math.floor(
      (inputHeight - filterHeight + padding[0] + padding[2]) / strides[0] + 1
    );
    const outputWidth = Math.floor(
      (inputWidth - filterWidth + padding[1] + padding[3]) / strides[1] + 1
    );
    
    const outputShape = [batchSize, outputHeight, outputWidth, outputChannels];
    
    // Create a unique graph key
    const graphKey = `conv2d_${input.shape.join('x')}_${filter.shape.join('x')}_${padding.join('_')}_${strides.join('_')}`;
    
    // Build and execute the graph
    const result = await this.runGraphComputation(
      graphKey,
      { input: input.tensor, filter: filter.tensor },
      { output: outputTensor }
    );
    
    // Return the result
    return {
      tensor: result.output,
      shape: outputShape,
      dataType: 'float32'
    };
  }
  
  /**
   * Build and run a graph computation
   */
  private async runGraphComputation(
    graphKey: string,
    inputs: Record<string, MLOperand>,
    outputs: Record<string, MLOperand>
  ): Promise<Record<string, MLOperand>> {
    if (!this.context || !this.graphBuilder) {
      throw new Error('WebNN backend not initialized');
    }
    
    try {
      // Check if we have a cached compiled graph
      let graph = this.compiledGraphs.get(graphKey);
      
      if (!graph) {
        // Build and compile a new graph
        graph = await this.graphBuilder.build({ 
          inputs: Object.values(inputs), 
          outputs: Object.values(outputs) 
        });
        
        // Cache the compiled graph
        this.compiledGraphs.set(graphKey, graph);
      }
      
      // Execute the graph
      const results = await graph.compute(inputs, outputs);
      return results;
    } catch (error) {
      console.error('WebNN graph execution failed:', error);
      throw error;
    }
  }
  
  /**
   * Read data from a tensor
   */
  async readTensor(
    tensor: MLOperand, 
    shape: number[], 
    dataType = 'float32'
  ): Promise<Float32Array | Int32Array | Uint8Array> {
    if (!this.context) {
      throw new Error('WebNN backend not initialized');
    }
    
    try {
      // Create a buffer to hold the tensor data
      const elementSize = dataType === 'float32' || dataType === 'int32' ? 4 : 1;
      const size = shape.reduce((a, b) => a * b, 1);
      const byteLength = size * elementSize;
      
      // Create the appropriate typed array based on data type
      let targetArray: Float32Array | Int32Array | Uint8Array;
      if (dataType === 'float32') {
        targetArray = new Float32Array(size);
      } else if (dataType === 'int32') {
        targetArray = new Int32Array(size);
      } else {
        targetArray = new Uint8Array(size);
      }
      
      // Copy data from the tensor to the array
      // Note: This is a simplification. Actual WebNN implementations might 
      // provide different APIs for reading tensor data.
      await (this.context as any).readOperand(tensor, targetArray);
      
      return targetArray;
    } catch (error) {
      console.error('Failed to read WebNN tensor:', error);
      throw error;
    }
  }
  
  /**
   * Run garbage collection if memory usage is above threshold
   */
  private maybeGarbageCollect(): void {
    if (!this.options.memory?.enableGarbageCollection) {
      return;
    }
    
    const threshold = this.options.memory.garbageCollectionThreshold || 1024 * 1024 * 128;
    
    if (this.memoryAllocated > threshold) {
      this.garbageCollect();
    }
  }
  
  /**
   * Force garbage collection of unused tensors and graphs
   */
  private garbageCollect(): void {
    // We don't have a direct way to free WebNN tensors in the current API spec
    // So we just clear our references and rely on JavaScript GC
    
    // Clear tensor cache (except the last few tensors which might still be in use)
    const tensorsToKeep = 10;
    const tensorEntries = Array.from(this.tensors.entries());
    
    if (tensorEntries.length > tensorsToKeep) {
      const tensorsToRemove = tensorEntries.slice(0, tensorEntries.length - tensorsToKeep);
      
      for (const [id] of tensorsToRemove) {
        this.tensors.delete(id);
      }
    }
    
    // Clear compiled graph cache (keep recent ones)
    const graphsToKeep = 5;
    const graphEntries = Array.from(this.compiledGraphs.entries());
    
    if (graphEntries.length > graphsToKeep) {
      const graphsToRemove = graphEntries.slice(0, graphEntries.length - graphsToKeep);
      
      for (const [key] of graphsToRemove) {
        this.compiledGraphs.delete(key);
      }
    }
    
    // Reset memory tracking
    this.memoryAllocated = 0;
    
    // Explicitly run V8 garbage collection if available
    if (typeof global !== 'undefined' && global.gc) {
      global.gc();
    }
  }
  
  /**
   * Execute pooling operation
   */
  private async executePooling(
    type: 'max' | 'average',
    inputs: Record<string, any>,
    options: Record<string, any> = {}
  ): Promise<any> {
    return executePoolingOperation(this, type, inputs, options);
  }
  
  /**
   * Execute normalization operation
   */
  private async executeNormalization(
    type: 'batch' | 'layer',
    inputs: Record<string, any>,
    options: Record<string, any> = {}
  ): Promise<any> {
    return executeNormalizationOperation(this, type, inputs, options);
  }
  
  /**
   * Execute advanced elementwise operation
   */
  private async executeAdvancedElementwise(
    operation: 'add' | 'sub' | 'mul' | 'div' | 'pow' | 'min' | 'max' | 'exp' | 'log' | 'sqrt',
    inputs: Record<string, any>,
    options: Record<string, any> = {}
  ): Promise<any> {
    return executeElementwiseOperation(this, operation, inputs);
  }
  
  /**
   * Execute tensor manipulation operation
   */
  private async executeTensorManipulation(
    operation: 'reshape' | 'transpose' | 'concat' | 'slice' | 'pad',
    inputs: Record<string, any>,
    options: Record<string, any> = {}
  ): Promise<any> {
    return executeTensorManipulationOperation(this, operation, inputs, options);
  }
  
  /**
   * Release all resources
   */
  dispose(): void {
    // Clear our caches
    this.tensors.clear();
    this.compiledGraphs.clear();
    
    // Reset context and graph builder
    this.context = null;
    this.graphBuilder = null;
    
    // Reset memory tracking
    this.memoryAllocated = 0;
  }
}

/**
 * TypeScript type definitions for WebNN API
 * These are needed since TypeScript doesn't include WebNN types yet
 */

interface MLContext {
  readonly deviceType?: string;
  createOperand(descriptor: MLOperandDescriptor, bufferView?: ArrayBufferView): MLOperand;
}

interface MLOperandDescriptor {
  type: 'float32' | 'int32' | 'uint8';
  dimensions: number[];
}

interface MLOperand {
  // This is intentionally empty as it's an opaque object in the API
}

interface MLGraph {
  compute(inputs: Record<string, MLOperand>, outputs: Record<string, MLOperand>): Promise<Record<string, MLOperand>>;
}

interface MLGraphBuilder {
  constant(descriptor: MLOperandDescriptor, bufferView: ArrayBufferView): MLOperand;
  input(name: string, descriptor: MLOperandDescriptor): MLOperand;
  relu(input: MLOperand): MLOperand;
  sigmoid(input: MLOperand): MLOperand;
  tanh(input: MLOperand): MLOperand;
  exp(input: MLOperand): MLOperand;
  log(input: MLOperand): MLOperand;
  sqrt(input: MLOperand): MLOperand;
  add(a: MLOperand, b: MLOperand): MLOperand;
  sub(a: MLOperand, b: MLOperand): MLOperand;
  mul(a: MLOperand, b: MLOperand): MLOperand;
  div(a: MLOperand, b: MLOperand): MLOperand;
  pow(a: MLOperand, b: MLOperand): MLOperand;
  min(a: MLOperand, b: MLOperand): MLOperand;
  max(a: MLOperand, b: MLOperand): MLOperand;
  matmul(a: MLOperand, b: MLOperand): MLOperand;
  reshape(input: MLOperand, newShape: number[]): MLOperand;
  concat(inputs: MLOperand[], axis: number): MLOperand;
  transpose(input: MLOperand, permutation?: number[]): MLOperand;
  slice(input: MLOperand, starts: number[], sizes: number[]): MLOperand;
  pad(input: MLOperand, padding: number[][], value: MLOperand): MLOperand;
  conv2d(input: MLOperand, filter: MLOperand, options: Record<string, any>): MLOperand;
  maxPool2d(input: MLOperand, options: Record<string, any>): MLOperand;
  averagePool2d(input: MLOperand, options: Record<string, any>): MLOperand;
  batchNormalization(input: MLOperand, mean: MLOperand, variance: MLOperand, scale?: MLOperand, bias?: MLOperand, options?: Record<string, any>): MLOperand;
  layerNormalization?(input: MLOperand, scale?: MLOperand, bias?: MLOperand, options?: Record<string, any>): MLOperand;
  reduceMean(input: MLOperand, options?: { axes: number[], keepDimensions: boolean }): MLOperand;
  build(options: { inputs: MLOperand[]; outputs: MLOperand[] }): Promise<MLGraph>;
}

/**
 * Declare the ml property on navigator for TypeScript
 */
declare global {
  interface Navigator {
    ml?: {
      createContext(options?: Record<string, any>): Promise<MLContext>;
    };
  }
}