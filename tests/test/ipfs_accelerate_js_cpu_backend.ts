/**
 * CPU Backend Implementation
 * Provides fallback implementation for all operations using JavaScript
 */

import { HardwareBackend, HardwareBackendType } from './hardware_abstraction';

/**
 * CPU Backend Options
 */
export interface CPUBackendOptions {
  /**
   * Maximum number of worker threads to use (if Web Workers are available)
   */
  maxWorkers?: number;
  
  /**
   * Enable multi-threading with Web Workers
   */
  useWebWorkers?: boolean;
  
  /**
   * Use typed arrays for memory efficiency
   */
  useTypedArrays?: boolean;
  
  /**
   * Enable operation optimization for modern browsers
   */
  enableOptimizations?: boolean;
  
  /**
   * Cache intermediate results for repeated operations
   */
  cacheResults?: boolean;
  
  /**
   * Memory management options
   */
  memory?: {
    /**
     * Enable garbage collection for unused tensors
     */
    enableGarbageCollection?: boolean;
    
    /**
     * Threshold for garbage collection (in bytes)
     */
    garbageCollectionThreshold?: number;
  };
}

/**
 * CPU Backend
 * Implements fallback operations in pure JavaScript
 */
export class CPUBackend implements HardwareBackend {
  readonly type: HardwareBackendType = 'cpu';
  
  private options: CPUBackendOptions;
  private workers: Worker[] = [];
  private resultCache: Map<string, any> = new Map();
  private memoryAllocated = 0;
  private initialized = false;
  
  /**
   * Create a CPU backend
   */
  constructor(options: CPUBackendOptions = {}) {
    this.options = {
      maxWorkers: typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 2 : 2,
      useWebWorkers: typeof Worker !== 'undefined',
      useTypedArrays: true,
      enableOptimizations: true,
      cacheResults: true,
      memory: {
        enableGarbageCollection: true,
        garbageCollectionThreshold: 1024 * 1024 * 256 // 256MB
      },
      ...options
    };
  }
  
  /**
   * Check if this backend is supported
   * CPU backend is always supported as a fallback
   */
  async isSupported(): Promise<boolean> {
    // CPU backend is always supported as a fallback
    return true;
  }
  
  /**
   * Initialize the CPU backend
   */
  async initialize(): Promise<boolean> {
    if (this.initialized) return true;
    
    try {
      // Initialize web workers if enabled and available
      if (this.options.useWebWorkers && typeof Worker !== 'undefined') {
        await this.initializeWorkers();
      }
      
      this.initialized = true;
      return true;
    } catch (error) {
      console.error('CPU backend initialization failed:', error);
      return false;
    }
  }
  
  /**
   * Initialize Web Workers for parallel processing
   */
  private async initializeWorkers(): Promise<void> {
    // Clear any existing workers
    this.terminateWorkers();
    
    // Create new workers up to the maximum
    const maxWorkers = this.options.maxWorkers || 2;
    
    // Web Worker creation is browser-specific and requires a Worker file
    // This is a simplified implementation that would need a proper worker file in production
    // For now, we'll just create dummy workers for demonstration
    try {
      for (let i = 0; i < maxWorkers; i++) {
        // In a real implementation, this would point to an actual worker JavaScript file
        // const worker = new Worker('./cpu_worker.js');
        // this.workers.push(worker);
        
        // For demonstration purposes, we'll just log that workers would be created
        console.log(`Would create CPU worker ${i + 1}/${maxWorkers} in a real implementation`);
      }
    } catch (error) {
      console.warn('Failed to initialize web workers:', error);
      // Continue without web workers
    }
  }
  
  /**
   * Terminate all web workers
   */
  private terminateWorkers(): void {
    for (const worker of this.workers) {
      worker.terminate();
    }
    this.workers = [];
  }
  
  /**
   * Get CPU-specific capabilities
   */
  async getCapabilities(): Promise<Record<string, any>> {
    return {
      useWebWorkers: this.options.useWebWorkers && this.workers.length > 0,
      workers: this.workers.length,
      typedArrays: this.options.useTypedArrays,
      optimizations: this.options.enableOptimizations,
      cache: this.options.cacheResults,
      cacheSize: this.resultCache.size
    };
  }
  
  /**
   * Create a tensor from data
   */
  async createTensor(
    data: Float32Array | Int32Array | Uint8Array, 
    shape: number[], 
    dataType = 'float32'
  ): Promise<any> {
    // Create appropriate typed array if not already
    let typedArray: Float32Array | Int32Array | Uint8Array;
    
    if (data instanceof Float32Array && dataType === 'float32') {
      typedArray = data;
    } else if (data instanceof Int32Array && dataType === 'int32') {
      typedArray = data;
    } else if (data instanceof Uint8Array && dataType === 'uint8') {
      typedArray = data;
    } else {
      // Convert to the appropriate typed array
      if (dataType === 'float32') {
        typedArray = new Float32Array(data as any);
      } else if (dataType === 'int32') {
        typedArray = new Int32Array(data as any);
      } else {
        typedArray = new Uint8Array(data as any);
      }
    }
    
    // Calculate size in bytes
    const elementSize = dataType === 'float32' || dataType === 'int32' ? 4 : 1;
    const size = typedArray.length * elementSize;
    
    // Update memory tracking
    this.memoryAllocated += size;
    this.maybeGarbageCollect();
    
    // Return tensor metadata
    return {
      data: typedArray,
      shape,
      dataType,
      size
    };
  }
  
  /**
   * Execute operation on CPU
   */
  async execute(
    operation: string,
    inputs: Record<string, any>,
    options: Record<string, any> = {}
  ): Promise<any> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    // Generate cache key if caching is enabled
    let cacheKey = '';
    if (this.options.cacheResults) {
      cacheKey = this.generateCacheKey(operation, inputs, options);
      const cachedResult = this.resultCache.get(cacheKey);
      if (cachedResult) {
        return cachedResult;
      }
    }
    
    // Execute the appropriate operation
    let result;
    switch (operation) {
      case 'matmul':
        result = this.executeMatmul(inputs, options);
        break;
        
      case 'elementwise':
        result = this.executeElementwise(inputs, options);
        break;
        
      case 'softmax':
        result = this.executeSoftmax(inputs, options);
        break;
        
      case 'conv2d':
        result = this.executeConv2d(inputs, options);
        break;
        
      case 'maxpool':
      case 'avgpool':
        result = this.executePooling(operation, inputs, options);
        break;
        
      case 'relu':
      case 'sigmoid':
      case 'tanh':
        // Redirect to elementwise for activation functions
        result = this.executeElementwise({ input: inputs.input }, { operation });
        break;
        
      case 'add':
      case 'sub':
      case 'mul':
      case 'div':
      case 'pow':
      case 'min':
      case 'max':
        result = this.executeBinaryOperation(operation, inputs, options);
        break;
        
      case 'exp':
      case 'log':
      case 'sqrt':
        result = this.executeUnaryOperation(operation, inputs, options);
        break;
        
      case 'reshape':
      case 'transpose':
      case 'concat':
      case 'slice':
        result = this.executeTensorManipulation(operation, inputs, options);
        break;
        
      case 'quantize':
        result = this.executeQuantize(inputs, options);
        break;
        
      case 'dequantize':
        result = this.executeDequantize(inputs, options);
        break;
        
      default:
        throw new Error(`Unsupported operation: ${operation}`);
    }
    
    // Cache the result if enabled
    if (this.options.cacheResults && cacheKey) {
      this.resultCache.set(cacheKey, result);
    }
    
    return result;
  }
  
  /**
   * Generate a cache key for an operation
   */
  private generateCacheKey(
    operation: string,
    inputs: Record<string, any>,
    options: Record<string, any>
  ): string {
    // Simple cache key generation
    // In a real implementation, this would be more sophisticated
    let key = `${operation}:`;
    
    // Add shape and type info from inputs
    for (const [name, input] of Object.entries(inputs)) {
      if (input && input.shape) {
        key += `${name}:${input.shape.join('x')}:${input.dataType || 'float32'}:`;
      }
    }
    
    // Add options
    key += JSON.stringify(options);
    
    return key;
  }
  
  /**
   * Matrix multiplication operation
   */
  private executeMatmul(
    inputs: {
      a: { data: Float32Array; shape: number[] };
      b: { data: Float32Array; shape: number[] };
    },
    options: {
      transposeA?: boolean;
      transposeB?: boolean;
    } = {}
  ): any {
    const { a, b } = inputs;
    const { transposeA = false, transposeB = false } = options;
    
    // Get dimensions
    const [M, K_a] = transposeA ? [a.shape[1], a.shape[0]] : [a.shape[0], a.shape[1]];
    const [K_b, N] = transposeB ? [b.shape[1], b.shape[0]] : [b.shape[0], b.shape[1]];
    
    if (K_a !== K_b) {
      throw new Error(`Matrix multiplication dimension mismatch: ${K_a} != ${K_b}`);
    }
    
    const K = K_a;
    
    // Create output array
    const output = new Float32Array(M * N);
    
    // Naive matrix multiplication implementation
    // In a real implementation, this would use more efficient algorithms
    for (let m = 0; m < M; m++) {
      for (let n = 0; n < N; n++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          // Access elements with appropriate indices based on transpose flags
          const aIndex = transposeA ? k * M + m : m * K + k;
          const bIndex = transposeB ? n * K + k : k * N + n;
          sum += a.data[aIndex] * b.data[bIndex];
        }
        output[m * N + n] = sum;
      }
    }
    
    return {
      data: output,
      shape: [M, N],
      dataType: 'float32'
    };
  }
  
  /**
   * Element-wise operations (relu, sigmoid, tanh)
   */
  private executeElementwise(
    inputs: {
      input: { data: Float32Array | Int32Array | Uint8Array; shape: number[] };
    },
    options: {
      operation?: 'relu' | 'sigmoid' | 'tanh';
    } = {}
  ): any {
    const { input } = inputs;
    const { operation = 'relu' } = options;
    
    // Get dimensions
    const size = input.data.length;
    
    // Create output array (same shape as input)
    const output = new Float32Array(size);
    
    // Apply element-wise operation
    for (let i = 0; i < size; i++) {
      const x = input.data[i];
      switch (operation) {
        case 'relu':
          output[i] = Math.max(0, x as number);
          break;
        case 'sigmoid':
          output[i] = 1 / (1 + Math.exp(-(x as number)));
          break;
        case 'tanh':
          output[i] = Math.tanh(x as number);
          break;
        default:
          output[i] = x as number;
      }
    }
    
    return {
      data: output,
      shape: input.shape,
      dataType: 'float32'
    };
  }
  
  /**
   * Softmax operation
   */
  private executeSoftmax(
    inputs: {
      input: { data: Float32Array; shape: number[] };
    },
    options: {
      axis?: number;
    } = {}
  ): any {
    const { input } = inputs;
    const { axis = -1 } = options;
    
    // Get dimensions
    const shape = input.shape;
    const actualAxis = axis < 0 ? shape.length + axis : axis;
    
    if (actualAxis < 0 || actualAxis >= shape.length) {
      throw new Error(`Invalid softmax axis: ${axis}`);
    }
    
    // Compute softmax along the specified axis
    // For simplicity, this implementation assumes 1D or 2D tensors
    // A full implementation would handle arbitrary dimensions
    
    const output = new Float32Array(input.data.length);
    
    if (shape.length === 1 || (shape.length === 2 && actualAxis === 1)) {
      // Simple case: softmax along last dimension for 1D tensor or rows of 2D tensor
      let offset = 0;
      const rowSize = actualAxis === 0 ? 1 : shape[1];
      const numRows = shape.length === 1 ? 1 : shape[0];
      
      for (let i = 0; i < numRows; i++) {
        // Find max for numerical stability
        let maxVal = -Infinity;
        for (let j = 0; j < rowSize; j++) {
          maxVal = Math.max(maxVal, input.data[offset + j]);
        }
        
        // Compute exp(x - max) and sum
        let sumExp = 0;
        for (let j = 0; j < rowSize; j++) {
          const expVal = Math.exp(input.data[offset + j] - maxVal);
          output[offset + j] = expVal;
          sumExp += expVal;
        }
        
        // Normalize
        for (let j = 0; j < rowSize; j++) {
          output[offset + j] /= sumExp;
        }
        
        offset += rowSize;
      }
    } else {
      // For more complex shapes, we'd need a more general implementation
      throw new Error('Softmax implementation for tensors with rank > 2 or axis != last dimension not yet implemented');
    }
    
    return {
      data: output,
      shape: input.shape,
      dataType: 'float32'
    };
  }
  
  /**
   * 2D Convolution operation
   */
  private executeConv2d(
    inputs: {
      input: { data: Float32Array; shape: number[] };
      filter: { data: Float32Array; shape: number[] };
    },
    options: {
      strides?: [number, number];
      padding?: [number, number, number, number]; // [top, right, bottom, left]
      dilations?: [number, number];
    } = {}
  ): any {
    const { input, filter } = inputs;
    const {
      strides = [1, 1],
      padding = [0, 0, 0, 0],
      dilations = [1, 1]
    } = options;
    
    // Get dimensions
    // Assuming NHWC format: [batch, height, width, channels]
    const [batchSize, inputHeight, inputWidth, inputChannels] = input.shape;
    
    // Assuming OIHW format for the filter: [outputChannels, inputChannels, filterHeight, filterWidth]
    // or HWIO format: [filterHeight, filterWidth, inputChannels, outputChannels]
    // We'll assume OIHW here
    const [outputChannels, filterInputChannels, filterHeight, filterWidth] = filter.shape;
    
    if (inputChannels !== filterInputChannels) {
      throw new Error(`Channel dimension mismatch: ${inputChannels} != ${filterInputChannels}`);
    }
    
    // Calculate output dimensions
    const outputHeight = Math.floor(
      (inputHeight + padding[0] + padding[2] - (dilations[0] * (filterHeight - 1) + 1)) / strides[0] + 1
    );
    const outputWidth = Math.floor(
      (inputWidth + padding[1] + padding[3] - (dilations[1] * (filterWidth - 1) + 1)) / strides[1] + 1
    );
    
    // Create output array
    const output = new Float32Array(batchSize * outputHeight * outputWidth * outputChannels);
    
    // Naive convolution implementation
    // In a real implementation, this would use more efficient algorithms
    for (let b = 0; b < batchSize; b++) {
      for (let oh = 0; oh < outputHeight; oh++) {
        for (let ow = 0; ow < outputWidth; ow++) {
          for (let oc = 0; oc < outputChannels; oc++) {
            let sum = 0;
            
            // Apply filter
            for (let fh = 0; fh < filterHeight; fh++) {
              for (let fw = 0; fw < filterWidth; fw++) {
                const ih = oh * strides[0] + fh * dilations[0] - padding[0];
                const iw = ow * strides[1] + fw * dilations[1] - padding[3];
                
                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                  for (let ic = 0; ic < inputChannels; ic++) {
                    const inputIndex = ((b * inputHeight + ih) * inputWidth + iw) * inputChannels + ic;
                    const filterIndex = ((oc * filterInputChannels + ic) * filterHeight + fh) * filterWidth + fw;
                    
                    sum += input.data[inputIndex] * filter.data[filterIndex];
                  }
                }
              }
            }
            
            const outputIndex = ((b * outputHeight + oh) * outputWidth + ow) * outputChannels + oc;
            output[outputIndex] = sum;
          }
        }
      }
    }
    
    return {
      data: output,
      shape: [batchSize, outputHeight, outputWidth, outputChannels],
      dataType: 'float32'
    };
  }
  
  /**
   * Pooling operations (max, average)
   */
  private executePooling(
    type: 'maxpool' | 'avgpool',
    inputs: {
      input: { data: Float32Array; shape: number[] };
    },
    options: {
      windowDimensions?: [number, number];
      strides?: [number, number];
      padding?: [number, number, number, number]; // [top, right, bottom, left]
    } = {}
  ): any {
    const { input } = inputs;
    const {
      windowDimensions = [2, 2],
      strides = [2, 2],
      padding = [0, 0, 0, 0]
    } = options;
    
    // Get dimensions
    // Assuming NHWC format: [batch, height, width, channels]
    const [batchSize, inputHeight, inputWidth, channels] = input.shape;
    
    // Calculate output dimensions
    const outputHeight = Math.floor(
      (inputHeight + padding[0] + padding[2] - windowDimensions[0]) / strides[0] + 1
    );
    const outputWidth = Math.floor(
      (inputWidth + padding[1] + padding[3] - windowDimensions[1]) / strides[1] + 1
    );
    
    // Create output array
    const output = new Float32Array(batchSize * outputHeight * outputWidth * channels);
    
    // Implementation
    for (let b = 0; b < batchSize; b++) {
      for (let oh = 0; oh < outputHeight; oh++) {
        for (let ow = 0; ow < outputWidth; ow++) {
          for (let c = 0; c < channels; c++) {
            let pooledValue = type === 'maxpool' ? -Infinity : 0;
            let count = 0;
            
            // Apply pooling window
            for (let kh = 0; kh < windowDimensions[0]; kh++) {
              for (let kw = 0; kw < windowDimensions[1]; kw++) {
                const ih = oh * strides[0] + kh - padding[0];
                const iw = ow * strides[1] + kw - padding[3];
                
                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                  const inputIndex = ((b * inputHeight + ih) * inputWidth + iw) * channels + c;
                  const value = input.data[inputIndex];
                  
                  if (type === 'maxpool') {
                    pooledValue = Math.max(pooledValue, value);
                  } else { // avgpool
                    pooledValue += value;
                    count++;
                  }
                }
              }
            }
            
            // For average pooling, divide by count of values
            if (type === 'avgpool' && count > 0) {
              pooledValue /= count;
            }
            
            const outputIndex = ((b * outputHeight + oh) * outputWidth + ow) * channels + c;
            output[outputIndex] = pooledValue;
          }
        }
      }
    }
    
    return {
      data: output,
      shape: [batchSize, outputHeight, outputWidth, channels],
      dataType: 'float32'
    };
  }
  
  /**
   * Binary operations (add, sub, mul, div, pow, min, max)
   */
  private executeBinaryOperation(
    operation: 'add' | 'sub' | 'mul' | 'div' | 'pow' | 'min' | 'max',
    inputs: {
      a: { data: Float32Array | Int32Array | Uint8Array; shape: number[] };
      b: { data: Float32Array | Int32Array | Uint8Array; shape: number[] };
    },
    options: Record<string, any> = {}
  ): any {
    const { a, b } = inputs;
    
    // Check for broadcasting
    const outputShape = this.getBroadcastShape(a.shape, b.shape);
    
    // Create output array
    const output = new Float32Array(outputShape.reduce((acc, dim) => acc * dim, 1));
    
    // Handle broadcasting
    const aStrides = this.getStrides(a.shape, outputShape);
    const bStrides = this.getStrides(b.shape, outputShape);
    const outputStrides = this.getStrides(outputShape, outputShape);
    
    // Apply operation with broadcasting
    const size = output.length;
    for (let i = 0; i < size; i++) {
      const indices = this.getIndices(i, outputStrides, outputShape);
      const aIndex = this.getFlatIndex(indices, aStrides, a.shape);
      const bIndex = this.getFlatIndex(indices, bStrides, b.shape);
      
      const aValue = aIndex !== -1 ? a.data[aIndex] as number : 0;
      const bValue = bIndex !== -1 ? b.data[bIndex] as number : 0;
      
      switch (operation) {
        case 'add':
          output[i] = aValue + bValue;
          break;
        case 'sub':
          output[i] = aValue - bValue;
          break;
        case 'mul':
          output[i] = aValue * bValue;
          break;
        case 'div':
          output[i] = aValue / bValue;
          break;
        case 'pow':
          output[i] = Math.pow(aValue, bValue);
          break;
        case 'min':
          output[i] = Math.min(aValue, bValue);
          break;
        case 'max':
          output[i] = Math.max(aValue, bValue);
          break;
      }
    }
    
    return {
      data: output,
      shape: outputShape,
      dataType: 'float32'
    };
  }
  
  /**
   * Unary operations (exp, log, sqrt)
   */
  private executeUnaryOperation(
    operation: 'exp' | 'log' | 'sqrt',
    inputs: {
      input: { data: Float32Array | Int32Array | Uint8Array; shape: number[] };
    },
    options: Record<string, any> = {}
  ): any {
    const { input } = inputs;
    
    // Create output array (same shape as input)
    const output = new Float32Array(input.data.length);
    
    // Apply unary operation
    for (let i = 0; i < input.data.length; i++) {
      const x = input.data[i] as number;
      switch (operation) {
        case 'exp':
          output[i] = Math.exp(x);
          break;
        case 'log':
          output[i] = Math.log(x);
          break;
        case 'sqrt':
          output[i] = Math.sqrt(x);
          break;
      }
    }
    
    return {
      data: output,
      shape: input.shape,
      dataType: 'float32'
    };
  }
  
  /**
   * Tensor manipulation operations (reshape, transpose, concat, slice)
   */
  private executeTensorManipulation(
    operation: 'reshape' | 'transpose' | 'concat' | 'slice',
    inputs: Record<string, any>,
    options: Record<string, any> = {}
  ): any {
    switch (operation) {
      case 'reshape':
        return this.executeReshape(inputs, options);
      case 'transpose':
        return this.executeTranspose(inputs, options);
      case 'concat':
        return this.executeConcat(inputs, options);
      case 'slice':
        return this.executeSlice(inputs, options);
      default:
        throw new Error(`Unsupported tensor manipulation: ${operation}`);
    }
  }
  
  /**
   * Reshape operation
   */
  private executeReshape(
    inputs: {
      input: { data: Float32Array | Int32Array | Uint8Array; shape: number[] };
    },
    options: {
      newShape: number[];
    }
  ): any {
    const { input } = inputs;
    const { newShape } = options;
    
    // Check that total size remains the same
    const inputSize = input.shape.reduce((acc, dim) => acc * dim, 1);
    const outputSize = newShape.reduce((acc, dim) => acc * dim, 1);
    
    if (inputSize !== outputSize) {
      throw new Error(`Invalid reshape: input size (${inputSize}) !== output size (${outputSize})`);
    }
    
    // For reshape, we simply copy the data and change the shape
    return {
      data: input.data.slice(),
      shape: newShape,
      dataType: input.dataType || 'float32'
    };
  }
  
  /**
   * Transpose operation
   */
  private executeTranspose(
    inputs: {
      input: { data: Float32Array | Int32Array | Uint8Array; shape: number[] };
    },
    options: {
      permutation?: number[];
    } = {}
  ): any {
    const { input } = inputs;
    let { permutation } = options;
    
    // Default permutation reverses all axes
    if (!permutation) {
      permutation = [];
      for (let i = input.shape.length - 1; i >= 0; i--) {
        permutation.push(i);
      }
    }
    
    // Check permutation validity
    if (permutation.length !== input.shape.length) {
      throw new Error('Permutation must have the same length as the input shape');
    }
    
    // Create output shape
    const outputShape = permutation.map(p => input.shape[p]);
    
    // Create output array
    const output = new (input.data.constructor as any)(input.data.length);
    
    // Calculate strides for input and output
    const inputStrides = this.calculateStrides(input.shape);
    const outputStrides = this.calculateStrides(outputShape);
    
    // Perform transpose
    for (let i = 0; i < input.data.length; i++) {
      // Get input indices
      const inputIndices = this.getIndicesFromFlatIndex(i, inputStrides);
      
      // Get output indices by applying permutation
      const outputIndices = permutation.map(p => inputIndices[p]);
      
      // Calculate output flat index
      const outputIndex = this.getFlatIndexFromIndices(outputIndices, outputStrides);
      
      // Copy data
      output[outputIndex] = input.data[i];
    }
    
    return {
      data: output,
      shape: outputShape,
      dataType: input.dataType || 'float32'
    };
  }
  
  /**
   * Concatenate tensors along axis
   */
  private executeConcat(
    inputs: {
      inputs: Array<{ data: Float32Array | Int32Array | Uint8Array; shape: number[] }>;
    },
    options: {
      axis?: number;
    } = {}
  ): any {
    const { inputs: tensors } = inputs;
    const { axis = 0 } = options;
    
    // Check that all tensors have the same shape except for the concatenation axis
    const shape = [...tensors[0].shape];
    let totalAxisSize = shape[axis];
    
    for (let i = 1; i < tensors.length; i++) {
      const tensor = tensors[i];
      
      if (tensor.shape.length !== shape.length) {
        throw new Error('All tensors must have the same number of dimensions');
      }
      
      for (let j = 0; j < shape.length; j++) {
        if (j === axis) {
          totalAxisSize += tensor.shape[j];
        } else if (shape[j] !== tensor.shape[j]) {
          throw new Error(`Tensor shapes don't match on non-concat axis: ${shape[j]} vs ${tensor.shape[j]}`);
        }
      }
    }
    
    // Update shape
    shape[axis] = totalAxisSize;
    
    // Create output array
    const outputSize = shape.reduce((acc, dim) => acc * dim, 1);
    const output = new Float32Array(outputSize);
    
    // Calculate strides
    const outputStrides = this.calculateStrides(shape);
    
    // Keep track of the current offset for each tensor
    let axisOffset = 0;
    
    // Copy data from each tensor
    for (const tensor of tensors) {
      const tensorStrides = this.calculateStrides(tensor.shape);
      const tensorAxisSize = tensor.shape[axis];
      
      for (let i = 0; i < tensor.data.length; i++) {
        // Get input indices
        const inputIndices = this.getIndicesFromFlatIndex(i, tensorStrides);
        
        // Adjust axis index for output
        inputIndices[axis] += axisOffset;
        
        // Calculate output flat index
        const outputIndex = this.getFlatIndexFromIndices(inputIndices, outputStrides);
        
        // Copy data
        output[outputIndex] = tensor.data[i] as number;
      }
      
      // Update axis offset
      axisOffset += tensorAxisSize;
    }
    
    return {
      data: output,
      shape,
      dataType: 'float32'
    };
  }
  
  /**
   * Slice operation
   */
  private executeSlice(
    inputs: {
      input: { data: Float32Array | Int32Array | Uint8Array; shape: number[] };
    },
    options: {
      begin: number[];
      size: number[];
    }
  ): any {
    const { input } = inputs;
    const { begin, size } = options;
    
    // Check validity
    if (begin.length !== input.shape.length || size.length !== input.shape.length) {
      throw new Error('Begin and size must have the same length as the input shape');
    }
    
    // Calculate output shape
    const outputShape = size;
    
    // Create output array
    const outputSize = outputShape.reduce((acc, dim) => acc * dim, 1);
    const output = new (input.data.constructor as any)(outputSize);
    
    // Calculate strides
    const inputStrides = this.calculateStrides(input.shape);
    const outputStrides = this.calculateStrides(outputShape);
    
    // Perform slice
    for (let i = 0; i < outputSize; i++) {
      // Get output indices
      const outputIndices = this.getIndicesFromFlatIndex(i, outputStrides);
      
      // Calculate input indices by adding begin offsets
      const inputIndices = outputIndices.map((index, dim) => index + begin[dim]);
      
      // Check if the input indices are within bounds
      const inBounds = inputIndices.every((index, dim) => index >= 0 && index < input.shape[dim]);
      
      if (inBounds) {
        // Calculate input flat index
        const inputIndex = this.getFlatIndexFromIndices(inputIndices, inputStrides);
        
        // Copy data
        output[i] = input.data[inputIndex];
      }
    }
    
    return {
      data: output,
      shape: outputShape,
      dataType: input.dataType || 'float32'
    };
  }
  
  /**
   * Calculate strides for a given shape
   */
  private calculateStrides(shape: number[]): number[] {
    const rank = shape.length;
    const strides = new Array(rank);
    
    strides[rank - 1] = 1;
    for (let i = rank - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    
    return strides;
  }
  
  /**
   * Get indices from flat index
   */
  private getIndicesFromFlatIndex(flatIndex: number, strides: number[]): number[] {
    const indices = [];
    let remaining = flatIndex;
    
    for (let i = 0; i < strides.length; i++) {
      const index = Math.floor(remaining / strides[i]);
      indices.push(index);
      remaining -= index * strides[i];
    }
    
    return indices;
  }
  
  /**
   * Get flat index from indices
   */
  private getFlatIndexFromIndices(indices: number[], strides: number[]): number {
    let flatIndex = 0;
    
    for (let i = 0; i < indices.length; i++) {
      flatIndex += indices[i] * strides[i];
    }
    
    return flatIndex;
  }
  
  /**
   * Get broadcast shape for binary operations
   */
  private getBroadcastShape(shapeA: number[], shapeB: number[]): number[] {
    const result = [];
    const maxLen = Math.max(shapeA.length, shapeB.length);
    
    // Pad shapes with ones for alignment
    const paddedA = Array(maxLen - shapeA.length).fill(1).concat(shapeA);
    const paddedB = Array(maxLen - shapeB.length).fill(1).concat(shapeB);
    
    // Compute broadcast shape
    for (let i = 0; i < maxLen; i++) {
      if (paddedA[i] === 1 || paddedB[i] === 1) {
        result.push(Math.max(paddedA[i], paddedB[i]));
      } else if (paddedA[i] === paddedB[i]) {
        result.push(paddedA[i]);
      } else {
        throw new Error(`Cannot broadcast shapes ${shapeA} and ${shapeB}`);
      }
    }
    
    return result;
  }
  
  /**
   * Calculate strides for broadcasting
   */
  private getStrides(shape: number[], broadcastShape: number[]): number[] {
    const result = [];
    const rankDiff = broadcastShape.length - shape.length;
    
    // Calculate strides for original shape
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      result.unshift(stride);
      stride *= shape[i];
    }
    
    // Pad with zeros for broadcasting
    return Array(rankDiff).fill(0).concat(result);
  }
  
  /**
   * Get flat index with broadcasting
   */
  private getFlatIndex(indices: number[], strides: number[], shape: number[]): number {
    // Check if dimensions are within shape - return -1 for out of bounds
    if (indices.length > shape.length) {
      const relevantIndices = indices.slice(indices.length - shape.length);
      for (let i = 0; i < shape.length; i++) {
        if (relevantIndices[i] >= shape[i]) {
          return -1;
        }
      }
    }
    
    let flatIndex = 0;
    const offset = indices.length - strides.length;
    
    for (let i = 0; i < strides.length; i++) {
      const index = offset + i < 0 ? 0 : indices[offset + i];
      flatIndex += index * strides[i];
    }
    
    return flatIndex;
  }
  
  /**
   * Get indices from flat index with strides
   */
  private getIndices(flatIndex: number, strides: number[], shape: number[]): number[] {
    const indices = [];
    
    for (let i = 0; i < shape.length; i++) {
      if (strides[i] > 0) {
        const index = Math.floor(flatIndex / strides[i]) % shape[i];
        indices.push(index);
      } else {
        indices.push(0);
      }
    }
    
    return indices;
  }
  
  /**
   * Execute quantization (float32 to int8)
   */
  private executeQuantize(
    inputs: {
      input: { data: Float32Array; shape: number[] };
    },
    options: Record<string, any> = {}
  ): any {
    const { input } = inputs;
    
    // Create output arrays
    const output = new Int32Array(input.data.length);
    const scale = new Float32Array(1);
    
    // Find max abs value for scaling
    let maxAbs = 0;
    for (let i = 0; i < input.data.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(input.data[i]));
    }
    
    // Compute scale factor (127.0 for int8)
    scale[0] = maxAbs > 0 ? 127.0 / maxAbs : 1.0;
    
    // Quantize values
    for (let i = 0; i < input.data.length; i++) {
      const quantized = Math.round(input.data[i] * scale[0]);
      output[i] = Math.max(-127, Math.min(127, quantized)); // Clamp to int8 range (-127 to 127)
    }
    
    return {
      data: output,
      scale: scale,
      shape: input.shape,
      dataType: 'int32'
    };
  }
  
  /**
   * Execute dequantization (int8 to float32)
   */
  private executeDequantize(
    inputs: {
      input: { data: Int32Array; shape: number[] };
      scale: { data: Float32Array };
    },
    options: Record<string, any> = {}
  ): any {
    const { input, scale } = inputs;
    
    // Create output array
    const output = new Float32Array(input.data.length);
    
    // Dequantize values
    for (let i = 0; i < input.data.length; i++) {
      output[i] = input.data[i] / scale.data[0];
    }
    
    return {
      data: output,
      shape: input.shape,
      dataType: 'float32'
    };
  }
  
  /**
   * Run garbage collection if memory usage is above threshold
   */
  private maybeGarbageCollect(): void {
    if (!this.options.memory?.enableGarbageCollection) {
      return;
    }
    
    const threshold = this.options.memory.garbageCollectionThreshold || 1024 * 1024 * 256;
    
    if (this.memoryAllocated > threshold) {
      this.garbageCollect();
    }
  }
  
  /**
   * Force garbage collection
   */
  private garbageCollect(): void {
    // Clear cache
    this.resultCache.clear();
    
    // Reset memory tracking
    this.memoryAllocated = 0;
  }
  
  /**
   * Release all resources
   */
  dispose(): void {
    // Terminate any web workers
    this.terminateWorkers();
    
    // Clear cache
    this.resultCache.clear();
    
    // Reset memory tracking
    this.memoryAllocated = 0;
    
    // Reset initialized state
    this.initialized = false;
  }
}