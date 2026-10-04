/**
 * Hardware Backend Interface
 * Defines the common interface that all hardware acceleration backends must implement
 */

import { Tensor } from '../../tensor/tensor';

/**
 * Interface for hardware backend capabilities
 */
export interface HardwareCapabilities {
  /** Maximum supported tensor dimensions */
  maxDimensions: number;
  
  /** Maximum matrix size for efficient multiplication */
  maxMatrixSize: number;
  
  /** Supported data types */
  supportedDataTypes: string[];
  
  /** Available memory in bytes (if known) */
  availableMemory?: number;
  
  /** Whether the backend supports async execution */
  supportsAsync: boolean;
  
  /** Supported operations */
  supportedOperations: {
    basicArithmetic: boolean;
    matrixMultiplication: boolean;
    convolution: boolean;
    reduction: boolean;
    activation: boolean;
  };
}

/**
 * Interface for all hardware acceleration backends
 */
export interface HardwareBackend {
  /** Unique identifier for this backend */
  readonly id: string;
  
  /** Type of backend (cpu, webgpu, webnn, etc.) */
  readonly type: string;
  
  /** Whether the backend is available in the current environment */
  readonly isAvailable: boolean;
  
  /** Capabilities of this backend */
  readonly capabilities: HardwareCapabilities;
  
  /**
   * Initialize the hardware backend
   * @returns Promise that resolves when backend is ready
   */
  initialize(): Promise<void>;
  
  /**
   * Check if the backend is initialized
   */
  isInitialized(): boolean;
  
  /**
   * Allocate a tensor on the backend
   * @param tensor Tensor to allocate
   * @returns Promise that resolves when allocation is complete
   */
  allocateTensor<T>(tensor: Tensor<T>): Promise<void>;
  
  /**
   * Release a tensor from the backend
   * @param tensor Tensor to release
   */
  releaseTensor<T>(tensor: Tensor<T>): void;
  
  /**
   * Execute tensor addition
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  add<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>>;
  
  /**
   * Execute tensor subtraction
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  subtract<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>>;
  
  /**
   * Execute tensor multiplication (element-wise)
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  multiply<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>>;
  
  /**
   * Execute tensor division
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  divide<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>>;
  
  /**
   * Execute matrix multiplication
   * @param a First tensor
   * @param b Second tensor
   * @returns Resulting tensor
   */
  matmul<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>>;
  
  /**
   * Execute transpose operation
   * @param tensor Input tensor
   * @returns Transposed tensor
   */
  transpose<T>(tensor: Tensor<T>): Promise<Tensor<T>>;
  
  /**
   * Execute ReLU activation function
   * @param tensor Input tensor
   * @returns Tensor with ReLU applied
   */
  relu<T>(tensor: Tensor<T>): Promise<Tensor<T>>;
  
  /**
   * Execute sigmoid activation function
   * @param tensor Input tensor
   * @returns Tensor with sigmoid applied
   */
  sigmoid<T>(tensor: Tensor<T>): Promise<Tensor<T>>;
  
  /**
   * Execute tanh activation function
   * @param tensor Input tensor
   * @returns Tensor with tanh applied
   */
  tanh<T>(tensor: Tensor<T>): Promise<Tensor<T>>;
  
  /**
   * Execute softmax activation function
   * @param tensor Input tensor
   * @param axis Axis to apply softmax on
   * @returns Tensor with softmax applied
   */
  softmax<T>(tensor: Tensor<T>, axis: number): Promise<Tensor<T>>;
  
  /**
   * Execute tensor reshape
   * @param tensor Input tensor
   * @param newShape New shape
   * @returns Reshaped tensor
   */
  reshape<T>(tensor: Tensor<T>, newShape: number[]): Promise<Tensor<T>>;
  
  /**
   * Synchronize backend execution
   * Ensures all queued operations are complete
   * @returns Promise that resolves when sync is complete
   */
  sync(): Promise<void>;
  
  /**
   * Free all resources associated with this backend
   */
  dispose(): void;
}