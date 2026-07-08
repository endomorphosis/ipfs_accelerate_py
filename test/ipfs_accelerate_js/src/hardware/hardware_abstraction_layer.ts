/**
 * Hardware Abstraction Layer (HAL)
 * Provides unified interface for hardware acceleration across different backends (WebGPU, WebNN, CPU)
 * Automatically selects the most appropriate backend based on model requirements and hardware availability
 */

import { Tensor } from '../tensor/tensor';
import { HardwareBackend, HardwareCapabilities } from './interfaces/hardware_backend';
import { SharedTensor } from '../tensor/shared_tensor';
import { BrowserType, detectBrowserType } from './webgpu/browser_optimized_operations';
import { PerformanceTracker, PerformanceRecord } from './performance_tracking';

/**
 * Types of hardware backends
 */
export type BackendType = 'webgpu' | 'webnn' | 'cpu';

/**
 * Hardware recommendation for a model type
 */
export interface HardwareRecommendation {
  /** Preferred backend type for this model */
  preferredBackend: BackendType;
  /** Browser preference if applicable */
  browserPreference?: BrowserType;
  /** Fallback backend type if preferred is not available */
  fallbackBackend: BackendType;
  /** Specific optimizations to enable */
  optimizations: {
    /** Use browser-specific optimizations */
    useBrowserOptimizations: boolean;
    /** Use operation fusion when possible */
    useOperationFusion: boolean;
    /** Use optimized memory management */
    optimizedMemory: boolean;
    /** Use quantization for lower memory usage */
    quantization?: {
      enabled: boolean;
      bits?: number;
      type?: 'int' | 'float';
    }
  };
}

/**
 * Model type categories for hardware recommendations
 */
export type ModelType = 'vision' | 'text' | 'audio' | 'multimodal';

/**
 * Backend selection criteria
 */
export interface BackendSelectionCriteria {
  /** Model type */
  modelType: ModelType;
  /** Explicit backend preference (overrides model type recommendations) */
  backendPreference?: BackendType[];
  /** Model size in parameters */
  modelSize?: 'tiny' | 'small' | 'base' | 'large';
  /** Whether to prioritize speed over accuracy */
  prioritizeSpeed?: boolean;
  /** Whether to prioritize memory efficiency */
  memoryConstrained?: boolean;
  /** Whether to use quantization */
  useQuantization?: boolean;
  /** Browser type if known */
  browserType?: BrowserType;
}

/**
 * Hardware Abstraction Layer configuration
 */
export interface HardwareAbstractionLayerConfig {
  /** Available backends */
  backends: HardwareBackend[];
  /** Default backend to use if no suitable backend is found */
  defaultBackend?: BackendType;
  /** Whether to automatically initialize backends when requested */
  autoInitialize?: boolean;
  /** Whether to use browser-specific optimizations */
  useBrowserOptimizations?: boolean;
  /** Browser type if known, otherwise it will be auto-detected */
  browserType?: BrowserType;
  /** Whether to enable tensor sharing between models */
  enableTensorSharing?: boolean;
  /** Whether to enable operation fusion */
  enableOperationFusion?: boolean;
  /** Memory cache size in bytes */
  memoryCacheSize?: number;
}

/**
 * Hardware Abstraction Layer class
 * Provides a unified interface to multiple hardware backends
 */
export class HardwareAbstractionLayer {
  private backends: Map<BackendType, HardwareBackend> = new Map();
  private activeBackend: HardwareBackend | null = null;
  private initialized = false;
  private config: HardwareAbstractionLayerConfig;
  private sharedTensors: Map<string, SharedTensor> = new Map();
  private browserType: BrowserType;
  /** Performance tracker for monitoring operation performance */
  private performanceTracker: PerformanceTracker;
  
  // Hardware recommendations by model type
  private modelRecommendations: Record<ModelType, HardwareRecommendation> = {
    vision: {
      preferredBackend: 'webgpu',
      browserPreference: 'chrome',
      fallbackBackend: 'cpu',
      optimizations: {
        useBrowserOptimizations: true,
        useOperationFusion: true,
        optimizedMemory: true,
        quantization: {
          enabled: false
        }
      }
    },
    text: {
      preferredBackend: 'webgpu',
      browserPreference: 'edge',
      fallbackBackend: 'cpu',
      optimizations: {
        useBrowserOptimizations: true,
        useOperationFusion: true,
        optimizedMemory: true,
        quantization: {
          enabled: false
        }
      }
    },
    audio: {
      preferredBackend: 'webgpu',
      browserPreference: 'firefox',
      fallbackBackend: 'cpu',
      optimizations: {
        useBrowserOptimizations: true,
        useOperationFusion: true,
        optimizedMemory: true,
        quantization: {
          enabled: false
        }
      }
    },
    multimodal: {
      preferredBackend: 'webgpu',
      fallbackBackend: 'cpu',
      optimizations: {
        useBrowserOptimizations: true,
        useOperationFusion: true,
        optimizedMemory: true,
        quantization: {
          enabled: false
        }
      }
    }
  };
  
  /**
   * Constructor for Hardware Abstraction Layer
   * @param config HAL configuration
   */
  constructor(config: HardwareAbstractionLayerConfig) {
    this.config = {
      autoInitialize: true,
      useBrowserOptimizations: true,
      enableTensorSharing: true,
      enableOperationFusion: true,
      memoryCacheSize: 1024 * 1024 * 100, // 100 MB default cache
      ...config
    };
    
    // Register available backends
    for (const backend of config.backends) {
      this.backends.set(backend.type as BackendType, backend);
    }
    
    // Auto-detect browser type if not specified
    this.browserType = config.browserType || detectBrowserType();
    
    // Use the first backend as active by default if no default specified
    if (!this.activeBackend && this.backends.size > 0) {
      const defaultBackend = this.config.defaultBackend || 'cpu';
      this.activeBackend = this.backends.get(defaultBackend) || this.backends.values().next().value;
    }
    
    // Initialize performance tracker
    this.performanceTracker = new PerformanceTracker();
    
    console.info('Hardware Abstraction Layer initialized with performance tracking');
  }
  
  /**
   * Get performance history for a specific operation on the current backend
   * @param operation Name of the operation
   * @returns Array of performance records
   */
  public getOperationPerformanceHistory(operation: string): PerformanceRecord[] {
    if (!this.activeBackend) {
      return [];
    }
    
    return this.performanceTracker.getOperationHistory(
      operation, 
      this.activeBackend.type
    );
  }
  
  /**
   * Get recommended backend for a specific operation based on performance history
   * @param operation Name of the operation
   * @returns Recommended backend type or null if not enough data
   */
  public getRecommendedBackendForOperation(operation: string): BackendType | null {
    return this.performanceTracker.getRecommendedBackend(operation);
  }
  
  /**
   * Generate a performance report with all data
   * @returns Comprehensive performance data
   */
  public generatePerformanceReport(): Record<string, any> {
    return this.performanceTracker.exportPerformanceData();
  }
  
  /**
   * Execute an operation with performance tracking
   * @param operation Name of the operation
   * @param inputShapes Shapes of input tensors
   * @param fn Function to execute
   * @returns Result of the operation
   */
  private async trackOperationPerformance<T>(
    operation: string,
    inputShapes: number[][],
    fn: () => Promise<T>
  ): Promise<T> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    const startTime = performance.now();
    let success = true;
    let errorType = undefined;
    let result: T;
    let outputShape: number[] = [];
    
    try {
      result = await fn();
      
      // Extract output shape if result is a Tensor
      if (result instanceof Tensor) {
        outputShape = result.dimensions;
      }
      
    } catch (error) {
      success = false;
      errorType = (error as Error).name;
      throw error;
    } finally {
      const endTime = performance.now();
      const durationMs = endTime - startTime;
      
      // Create performance record
      const record: PerformanceRecord = {
        timestamp: Date.now(),
        operation,
        backendType: this.activeBackend.type,
        browserType: this.browserType,
        durationMs,
        inputShapes,
        outputShape,
        success,
        errorType
      };
      
      // Track operation
      this.performanceTracker.trackOperation(record);
    }
    
    return result!;
  }
  
  /**
   * Initialize the hardware abstraction layer
   * @returns Promise that resolves when initialization is complete
   */
  public async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }
    
    // Initialize the default backend
    if (this.activeBackend && !this.activeBackend.isInitialized()) {
      await this.activeBackend.initialize();
    }
    
    this.initialized = true;
  }
  
  /**
   * Check if the hardware abstraction layer is initialized
   * @returns Whether the HAL is initialized
   */
  public isInitialized(): boolean {
    return this.initialized;
  }
  
  /**
   * Get hardware recommendations for a model type
   * @param modelType Type of model
   * @returns Hardware recommendations
   */
  public getRecommendations(modelType: ModelType): HardwareRecommendation {
    return this.modelRecommendations[modelType];
  }
  
  /**
   * Select the best backend for a given model type and criteria
   * @param criteria Backend selection criteria
   * @returns Selected backend or null if no suitable backend is found
   */
  public async selectBackend(criteria: BackendSelectionCriteria): Promise<HardwareBackend | null> {
    // Get recommendations for this model type
    const recommendations = this.getRecommendations(criteria.modelType);
    
    // If there's an explicit preference, try those first
    if (criteria.backendPreference && criteria.backendPreference.length > 0) {
      for (const preferredType of criteria.backendPreference) {
        const backend = this.backends.get(preferredType);
        if (backend && backend.isAvailable) {
          // Initialize the backend if needed
          if (this.config.autoInitialize && !backend.isInitialized()) {
            await backend.initialize();
          }
          return backend;
        }
      }
    }
    
    // Otherwise use recommended backend
    const recommendedType = criteria.memoryConstrained ? 
      recommendations.fallbackBackend : recommendations.preferredBackend;
    
    const backend = this.backends.get(recommendedType);
    if (backend && backend.isAvailable) {
      // Initialize the backend if needed
      if (this.config.autoInitialize && !backend.isInitialized()) {
        await backend.initialize();
      }
      return backend;
    }
    
    // Fallback to any available backend
    for (const [_, backend] of this.backends) {
      if (backend.isAvailable) {
        // Initialize the backend if needed
        if (this.config.autoInitialize && !backend.isInitialized()) {
          await backend.initialize();
        }
        return backend;
      }
    }
    
    // No suitable backend found
    return null;
  }
  
  /**
   * Set the active backend
   * @param backend Backend to set as active
   */
  public setActiveBackend(backend: HardwareBackend): void {
    this.activeBackend = backend;
  }
  
  /**
   * Get the active backend
   * @returns Active backend
   */
  public getActiveBackend(): HardwareBackend | null {
    return this.activeBackend;
  }
  
  /**
   * Get the active backend type
   * @returns Active backend type
   */
  public getBackendType(): string {
    return this.activeBackend ? this.activeBackend.type : 'none';
  }
  
  /**
   * Create a tensor on the active backend
   * @param options Tensor creation options
   * @returns Created tensor
   */
  public async createTensor<T>(options: {
    dimensions: number[],
    data: T[] | Float32Array | Int32Array | Uint8Array,
    dtype: string
  }): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    if (!this.initialized) {
      await this.initialize();
    }
    
    // Create tensor
    const tensor = new Tensor<T>(
      options.dimensions,
      options.data,
      options.dtype
    );
    
    // Allocate on backend
    await this.activeBackend.allocateTensor(tensor);
    
    return tensor;
  }
  
  /**
   * Release a tensor from the active backend
   * @param tensor Tensor to release
   */
  public async releaseTensor<T>(tensor: Tensor<T>): Promise<void> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    this.activeBackend.releaseTensor(tensor);
  }
  
  /**
   * Execute tensor addition
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  public async add<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    return this.activeBackend.add(a, b);
  }
  
  /**
   * Execute tensor subtraction
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  public async subtract<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    return this.activeBackend.subtract(a, b);
  }
  
  /**
   * Execute tensor multiplication (element-wise)
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  public async multiply<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    return this.activeBackend.multiply(a, b);
  }
  
  /**
   * Execute tensor division
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  public async divide<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    return this.activeBackend.divide(a, b);
  }
  
  /**
   * Execute matrix multiplication
   * @param a First tensor
   * @param b Second tensor
   * @param options Additional options
   * @returns Resulting tensor
   */
  public async matmul<T>(
    a: Tensor<T>, 
    b: Tensor<T>, 
    options?: {
      useOptimization?: boolean
    }
  ): Promise<Tensor<T>> {
    // Check if there's a recommended backend based on performance
    const recommendedBackend = this.performanceTracker.getRecommendedBackend('matmul');
    const operationName = 'matmul';
    
    // If we have a recommendation and it's different from current backend
    if (recommendedBackend && 
        this.activeBackend && 
        recommendedBackend !== this.activeBackend.type &&
        this.backends.has(recommendedBackend)) {
        
      // Get the recommended backend
      const recommended = this.backends.get(recommendedBackend)!;
      
      // Only switch if it's available
      if (recommended.isAvailable) {
        // Store original backend for restoration
        const originalBackend = this.activeBackend;
        
        // Switch to recommended backend
        this.activeBackend = recommended;
        
        // Make sure it's initialized
        if (!recommended.isInitialized()) {
          await recommended.initialize();
        }
        
        console.info(`Using recommended backend ${recommendedBackend} for ${operationName} operation based on performance history`);
      }
    }
    
    // Track the operation performance
    return this.trackOperationPerformance<Tensor<T>>(
      operationName,
      [a.dimensions, b.dimensions],
      async () => {
        if (!this.activeBackend) {
          throw new Error('No active backend');
        }
        
        // If the backend supports optimized matmul and optimization is requested
        if (options?.useOptimization && 
            this.config.useBrowserOptimizations && 
            'optimizedMatmul' in this.activeBackend) {
          // Use optimized implementation if available
          const optimizedBackend = this.activeBackend as any;
          return optimizedBackend.optimizedMatmul(a, b, {
            browserType: this.browserType
          });
        }
        
        // Otherwise use standard implementation
        return this.activeBackend.matmul(a, b);
      }
    );
  }
  
  /**
   * Execute tensor transpose
   * @param tensor Input tensor
   * @param axes Permutation of axes (optional)
   * @returns Transposed tensor
   */
  public async transpose<T>(tensor: Tensor<T>, axes?: number[]): Promise<Tensor<T>> {
    // Check if there's a recommended backend based on performance
    const recommendedBackend = this.performanceTracker.getRecommendedBackend('transpose');
    const operationName = 'transpose';
    
    // If we have a recommendation and it's different from current backend
    if (recommendedBackend && 
        this.activeBackend && 
        recommendedBackend !== this.activeBackend.type &&
        this.backends.has(recommendedBackend)) {
        
      // Get the recommended backend
      const recommended = this.backends.get(recommendedBackend)!;
      
      // Only switch if it's available
      if (recommended.isAvailable) {
        // Store original backend for restoration
        const originalBackend = this.activeBackend;
        
        // Switch to recommended backend
        this.activeBackend = recommended;
        
        // Make sure it's initialized
        if (!recommended.isInitialized()) {
          await recommended.initialize();
        }
        
        console.info(`Using recommended backend ${recommendedBackend} for ${operationName} operation based on performance history`);
      }
    }
    
    // Track the operation performance
    return this.trackOperationPerformance<Tensor<T>>(
      operationName,
      [tensor.dimensions],
      async () => {
        if (!this.activeBackend) {
          throw new Error('No active backend');
        }
        
        if (axes && 'transposeWithAxes' in this.activeBackend) {
          // Use transposeWithAxes if available
          const extendedBackend = this.activeBackend as any;
          return extendedBackend.transposeWithAxes(tensor, axes);
        }
        
        return this.activeBackend.transpose(tensor);
      }
    );
  }
  
  /**
   * Execute tensor reshape
   * @param tensor Input tensor
   * @param newShape New shape
   * @returns Reshaped tensor
   */
  public async reshape<T>(tensor: Tensor<T>, newShape: number[]): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    return this.activeBackend.reshape(tensor, newShape);
  }
  
  /**
   * Execute GELU activation function
   * @param tensor Input tensor
   * @returns Tensor with GELU applied
   */
  public async gelu<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    // Check if backend has native GELU implementation
    if ('gelu' in this.activeBackend) {
      const extendedBackend = this.activeBackend as any;
      return extendedBackend.gelu(tensor);
    }
    
    // Otherwise implement using existing operations
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const x = tensor;
    
    // Create constant tensors
    const constHalf = await this.createTensor({
      dimensions: [1],
      data: new Float32Array([0.5]),
      dtype: 'float32'
    });
    
    const constSqrt2OverPi = await this.createTensor({
      dimensions: [1],
      data: new Float32Array([0.7978845608]),  // sqrt(2/π)
      dtype: 'float32'
    });
    
    const constCoeff = await this.createTensor({
      dimensions: [1],
      data: new Float32Array([0.044715]),
      dtype: 'float32'
    });
    
    // x^3
    const xCubed = await this.multiply(x, await this.multiply(x, x));
    
    // 0.044715 * x^3
    const xCubedScaled = await this.multiply(constCoeff, xCubed);
    
    // x + 0.044715 * x^3
    const inner = await this.add(x, xCubedScaled);
    
    // sqrt(2/π) * (x + 0.044715 * x^3)
    const innerScaled = await this.multiply(constSqrt2OverPi, inner);
    
    // tanh(sqrt(2/π) * (x + 0.044715 * x^3))
    const tanhInner = await this.activeBackend.tanh(innerScaled);
    
    // 1 + tanh(...)
    const onePlusTanh = await this.add(
      await this.createTensor({
        dimensions: [1],
        data: new Float32Array([1.0]),
        dtype: 'float32'
      }),
      tanhInner
    );
    
    // x * (1 + tanh(...))
    const xTimesOnePlusTanh = await this.multiply(x, onePlusTanh);
    
    // 0.5 * x * (1 + tanh(...))
    const result = await this.multiply(constHalf, xTimesOnePlusTanh);
    
    // Release temporary tensors
    await this.releaseTensor(constHalf);
    await this.releaseTensor(constSqrt2OverPi);
    await this.releaseTensor(constCoeff);
    await this.releaseTensor(xCubed);
    await this.releaseTensor(xCubedScaled);
    await this.releaseTensor(inner);
    await this.releaseTensor(innerScaled);
    await this.releaseTensor(tanhInner);
    await this.releaseTensor(onePlusTanh);
    await this.releaseTensor(xTimesOnePlusTanh);
    
    return result;
  }
  
  /**
   * Execute ReLU activation function
   * @param tensor Input tensor
   * @returns Tensor with ReLU applied
   */
  public async relu<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    return this.activeBackend.relu(tensor);
  }
  
  /**
   * Execute sigmoid activation function
   * @param tensor Input tensor
   * @returns Tensor with sigmoid applied
   */
  public async sigmoid<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    return this.activeBackend.sigmoid(tensor);
  }
  
  /**
   * Execute tanh activation function
   * @param tensor Input tensor
   * @returns Tensor with tanh applied
   */
  public async tanh<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    return this.activeBackend.tanh(tensor);
  }
  
  /**
   * Execute softmax activation function
   * @param tensor Input tensor
   * @param axis Axis to apply softmax on
   * @returns Tensor with softmax applied
   */
  public async softmax<T>(tensor: Tensor<T>, axis: number): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    return this.activeBackend.softmax(tensor, axis);
  }
  
  /**
   * Execute layer normalization
   * @param input Input tensor
   * @param weight Scale parameter
   * @param bias Bias parameter
   * @param epsilon Small constant for numerical stability
   * @returns Normalized tensor
   */
  public async layerNorm<T>(
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    epsilon: number = 1e-5
  ): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    // Check if backend has native layerNorm implementation
    if ('layerNorm' in this.activeBackend) {
      const extendedBackend = this.activeBackend as any;
      return extendedBackend.layerNorm(input, weight, bias, {
        epsilon,
        useBrowserOptimizations: this.config.useBrowserOptimizations,
        browserType: this.browserType
      });
    }
    
    // Otherwise implement using existing operations
    // Get last dimension size (normalized over last dim)
    const lastDimSize = input.dimensions[input.dimensions.length - 1];
    
    // Calculate mean across last dimension
    const sum = await this.reduceSum(input, -1, true);
    const mean = await this.divide(sum, 
      await this.createTensor<T>({
        dimensions: [1],
        data: new Float32Array([lastDimSize]),
        dtype: 'float32'
      } as any)
    );
    
    // Calculate variance
    const diff = await this.subtract(input, mean);
    const diffSquared = await this.multiply(diff, diff);
    const varianceSum = await this.reduceSum(diffSquared, -1, true);
    const variance = await this.divide(varianceSum,
      await this.createTensor<T>({
        dimensions: [1],
        data: new Float32Array([lastDimSize]),
        dtype: 'float32'
      } as any)
    );
    
    // Add epsilon for numerical stability
    const epsilonTensor = await this.createTensor<T>({
      dimensions: [1],
      data: new Float32Array([epsilon]),
      dtype: 'float32'
    } as any);
    
    const variancePlusEpsilon = await this.add(variance, epsilonTensor);
    
    // Calculate standard deviation
    const stdDev = await this.sqrt(variancePlusEpsilon);
    
    // Normalize
    const normalized = await this.divide(diff, stdDev);
    
    // Scale and shift
    const scaled = await this.multiply(normalized, weight);
    const result = await this.add(scaled, bias);
    
    // Release temporary tensors
    await this.releaseTensor(sum);
    await this.releaseTensor(mean);
    await this.releaseTensor(diff);
    await this.releaseTensor(diffSquared);
    await this.releaseTensor(varianceSum);
    await this.releaseTensor(variance);
    await this.releaseTensor(epsilonTensor);
    await this.releaseTensor(variancePlusEpsilon);
    await this.releaseTensor(stdDev);
    await this.releaseTensor(normalized);
    await this.releaseTensor(scaled);
    
    return result;
  }
  
  /**
   * Execute square root operation
   * @param tensor Input tensor
   * @returns Tensor with sqrt applied
   */
  public async sqrt<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    // Check if backend has native sqrt implementation
    if ('sqrt' in this.activeBackend) {
      const extendedBackend = this.activeBackend as any;
      return extendedBackend.sqrt(tensor);
    }
    
    // Otherwise implement using existing operations
    // For sqrt, we can use x^0.5
    // First create a tensor with 0.5
    const halfTensor = await this.createTensor<T>({
      dimensions: [1],
      data: new Float32Array([0.5]),
      dtype: 'float32'
    } as any);
    
    // Then use pow operation if available
    if ('pow' in this.activeBackend) {
      const extendedBackend = this.activeBackend as any;
      const result = await extendedBackend.pow(tensor, halfTensor);
      await this.releaseTensor(halfTensor);
      return result;
    }
    
    // If pow is not available, use exp(0.5 * log(x))
    if ('log' in this.activeBackend && 'exp' in this.activeBackend) {
      const extendedBackend = this.activeBackend as any;
      const logX = await extendedBackend.log(tensor);
      const halfLogX = await this.multiply(halfTensor, logX);
      const result = await extendedBackend.exp(halfLogX);
      
      await this.releaseTensor(halfTensor);
      await this.releaseTensor(logX);
      await this.releaseTensor(halfLogX);
      
      return result;
    }
    
    throw new Error('Backend does not support sqrt, pow, or log/exp operations');
  }
  
  /**
   * Execute sum reduction
   * @param tensor Input tensor
   * @param axis Axis to reduce along
   * @param keepDims Whether to keep reduced dimensions
   * @returns Reduced tensor
   */
  public async reduceSum<T>(
    tensor: Tensor<T>, 
    axis: number,
    keepDims: boolean = false
  ): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    // Check if backend has native reduceSum implementation
    if ('reduceSum' in this.activeBackend) {
      const extendedBackend = this.activeBackend as any;
      return extendedBackend.reduceSum(tensor, axis, keepDims);
    }
    
    throw new Error('Backend does not support reduction operations');
  }
  
  /**
   * Execute mean reduction
   * @param tensor Input tensor
   * @param axis Axis to reduce along
   * @param keepDims Whether to keep reduced dimensions
   * @returns Reduced tensor
   */
  public async reduceMean<T>(
    tensor: Tensor<T>, 
    axis: number,
    keepDims: boolean = false
  ): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    // Check if backend has native reduceMean implementation
    if ('reduceMean' in this.activeBackend) {
      const extendedBackend = this.activeBackend as any;
      return extendedBackend.reduceMean(tensor, axis, keepDims);
    }
    
    // Otherwise implement using sum and division
    const sum = await this.reduceSum(tensor, axis, keepDims);
    
    // Get size of the reduced dimension
    let reduceSize: number;
    if (axis < 0) {
      axis = tensor.dimensions.length + axis;
    }
    reduceSize = tensor.dimensions[axis];
    
    const divTensor = await this.createTensor<T>({
      dimensions: [1],
      data: new Float32Array([reduceSize]),
      dtype: 'float32'
    } as any);
    
    const result = await this.divide(sum, divTensor);
    
    await this.releaseTensor(sum);
    await this.releaseTensor(divTensor);
    
    return result;
  }
  
  /**
   * Slice a tensor
   * @param input Input tensor
   * @param starts Start indices
   * @param ends End indices
   * @returns Sliced tensor
   */
  public async slice<T>(
    input: Tensor<T>,
    starts: number[],
    ends: number[]
  ): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    // Check if backend has native slice implementation
    if ('slice' in this.activeBackend) {
      const extendedBackend = this.activeBackend as any;
      return extendedBackend.slice(input, starts, ends);
    }
    
    throw new Error('Backend does not support slice operation');
  }
  
  /**
   * Concatenate tensors along an axis
   * @param tensors Tensors to concatenate
   * @param axis Axis to concatenate along
   * @returns Concatenated tensor
   */
  public async concat<T>(
    tensors: Tensor<T>[],
    axis: number
  ): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    // Check if backend has native concat implementation
    if ('concat' in this.activeBackend) {
      const extendedBackend = this.activeBackend as any;
      return extendedBackend.concat(tensors, axis);
    }
    
    throw new Error('Backend does not support concat operation');
  }
  
  /**
   * Gather elements from a tensor
   * @param input Input tensor
   * @param indices Indices tensor
   * @param axis Axis to gather along
   * @returns Gathered tensor
   */
  public async gather<T>(
    input: Tensor<T>,
    indices: Tensor<number>,
    axis: number = 0
  ): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    // Check if backend has native gather implementation
    if ('gather' in this.activeBackend) {
      const extendedBackend = this.activeBackend as any;
      return extendedBackend.gather(input, indices, axis);
    }
    
    throw new Error('Backend does not support gather operation');
  }
  
  /**
   * Gather embedding from a lookup table
   * @param embedding Embedding table
   * @param indices Indices to lookup
   * @returns Gathered embeddings
   */
  public async gatherEmbedding<T>(
    embedding: Tensor<T>,
    indices: Tensor<number>
  ): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    // Check if backend has native gatherEmbedding implementation
    if ('gatherEmbedding' in this.activeBackend) {
      const extendedBackend = this.activeBackend as any;
      return extendedBackend.gatherEmbedding(embedding, indices);
    }
    
    // Otherwise use regular gather with axis=0
    return this.gather(embedding, indices, 0);
  }
  
  /**
   * Repeat a tensor along dimensions
   * @param input Input tensor
   * @param repeats Number of repeats along each dimension
   * @returns Repeated tensor
   */
  public async repeat<T>(
    input: Tensor<T>,
    repeats: number[]
  ): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    // Check if backend has native repeat implementation
    if ('repeat' in this.activeBackend) {
      const extendedBackend = this.activeBackend as any;
      return extendedBackend.repeat(input, repeats);
    }
    
    throw new Error('Backend does not support repeat operation');
  }
  
  /**
   * Execute FFNN layer with optimizations if available
   * @param input Input tensor
   * @param weights Weight tensor
   * @param bias Bias tensor
   * @param activation Activation function to apply
   * @returns Output tensor
   */
  public async ffnn<T>(
    input: Tensor<T>,
    weights: Tensor<T>,
    bias: Tensor<T>,
    activation: 'relu' | 'gelu' | 'sigmoid' | 'tanh' | 'none' = 'none'
  ): Promise<Tensor<T>> {
    if (!this.activeBackend) {
      throw new Error('No active backend');
    }
    
    // Check if backend supports operation fusion and it's enabled
    if (this.config.enableOperationFusion && 'executeOperations' in this.activeBackend) {
      const extendedBackend = this.activeBackend as any;
      
      // Define operations sequence
      const operations = ['matmul', 'add'];
      if (activation !== 'none') {
        operations.push(activation);
      }
      
      // Execute fused operations
      return extendedBackend.executeOperations(
        [input, weights, bias],
        operations,
        {
          useBrowserOptimizations: this.config.useBrowserOptimizations,
          browserType: this.browserType,
          useFusion: true
        }
      );
    }
    
    // Otherwise implement using individual operations
    const matmulResult = await this.matmul(input, weights);
    const biasedResult = await this.add(matmulResult, bias);
    
    let result: Tensor<T>;
    
    // Apply activation function
    switch (activation) {
      case 'relu':
        result = await this.relu(biasedResult);
        break;
      case 'gelu':
        result = await this.gelu(biasedResult);
        break;
      case 'sigmoid':
        result = await this.sigmoid(biasedResult);
        break;
      case 'tanh':
        result = await this.tanh(biasedResult);
        break;
      case 'none':
      default:
        result = biasedResult;
        break;
    }
    
    // Release temporary tensors
    if (result !== matmulResult) {
      await this.releaseTensor(matmulResult);
    }
    if (result !== biasedResult) {
      await this.releaseTensor(biasedResult);
    }
    
    return result;
  }
  
  /**
   * Create a shared tensor that can be used by other models
   * @param tensor Tensor to share
   * @param outputType Type of output to share
   * @param modelId Model identifier
   * @returns Shared tensor reference
   */
  public createSharedTensor<T>(
    tensor: Tensor<T>,
    outputType: string,
    modelId: string
  ): SharedTensor {
    if (!this.config.enableTensorSharing) {
      throw new Error('Tensor sharing is disabled');
    }
    
    // Create a new shared tensor
    const sharedTensor = new SharedTensor(
      tensor,
      outputType,
      modelId,
      this.getBackendType()
    );
    
    // Store in shared tensors map
    const key = `${outputType}_${modelId}`;
    this.sharedTensors.set(key, sharedTensor);
    
    return sharedTensor;
  }
  
  /**
   * Get a shared tensor if available
   * @param outputType Type of output to get
   * @param modelId Model identifier
   * @returns Shared tensor or null if not found
   */
  public getSharedTensor(
    outputType: string,
    modelId: string
  ): SharedTensor | null {
    if (!this.config.enableTensorSharing) {
      return null;
    }
    
    const key = `${outputType}_${modelId}`;
    return this.sharedTensors.get(key) || null;
  }
  
  /**
   * Synchronize backend execution
   * @returns Promise that resolves when sync is complete
   */
  public async sync(): Promise<void> {
    if (!this.activeBackend) {
      return;
    }
    
    return this.activeBackend.sync();
  }
  
  /**
   * Dispose of all resources
   */
  public dispose(): void {
    // Release all shared tensors
    for (const [key, sharedTensor] of this.sharedTensors.entries()) {
      sharedTensor.release();
      this.sharedTensors.delete(key);
    }
    
    // Dispose backends
    for (const [_, backend] of this.backends) {
      backend.dispose();
    }
    
    this.activeBackend = null;
    this.initialized = false;
  }
}

/**
 * Factory function to create Hardware Abstraction Layer
 * @param config HAL configuration
 * @returns New HAL instance
 */
export function createHardwareAbstractionLayer(
  config: HardwareAbstractionLayerConfig
): HardwareAbstractionLayer {
  return new HardwareAbstractionLayer(config);
}