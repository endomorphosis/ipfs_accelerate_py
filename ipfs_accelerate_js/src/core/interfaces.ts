/**
 * Core interfaces for IPFS Accelerate JS SDK
 */

export type ModelType = 'text' | 'vision' | 'audio' | 'multimodal';
export type HardwareBackendType = 'webgpu' | 'webnn' | 'wasm' | 'cpu';
export type TensorDataType = 'float32' | 'float16' | 'int32' | 'int16' | 'int8' | 'uint8';
export type QuantizationMode = 'int8' | 'int4' | 'float16' | 'none';

/**
 * Generic interface for model input
 */
export interface ModelInput<T = any> {
  [key: string]: T;
}

/**
 * Generic interface for model output
 */
export interface ModelOutput<T = any> {
  [key: string]: T;
}

/**
 * Common options for all model types
 */
export interface ModelOptions {
  /** Preferred hardware backend */
  preferredBackend?: HardwareBackendType;
  /** Enable cache for model weights */
  enableCache?: boolean;
  /** Quantization mode for model weights */
  quantization?: QuantizationMode;
  /** Enable performance tracking */
  trackPerformance?: boolean;
  /** Logging level */
  logLevel?: 'none' | 'error' | 'warn' | 'info' | 'debug';
}

/**
 * Backend interface that all hardware backends must implement 
 */
export interface IBackend {
  /** Initialize the backend */
  initialize(): Promise<boolean>;
  /** Check if the backend is initialized */
  isInitialized(): boolean;
  /** Check if backend is using real hardware acceleration */
  isRealHardware(): boolean;
  /** Get backend type */
  getType(): HardwareBackendType;
  /** Get backend capabilities */
  getCapabilities(): any;
  /** Get backend information */
  getInfo(): any;
  /** Dispose backend resources */
  dispose(): void;
}

/**
 * Tensor interface for cross-platform tensor operations
 */
export interface ITensor {
  /** Get tensor dimensions */
  getDimensions(): number[];
  /** Get tensor data type */
  getDataType(): TensorDataType;
  /** Get total number of elements */
  getSize(): number;
  /** Get tensor data as typed array */
  getData<T extends ArrayBufferView>(): T;
  /** Get tensor name or identifier */
  getName(): string;
  /** Dispose tensor resources */
  dispose(): void;
}

/**
 * Model interface that all models must implement
 */
export interface IModel<
  TInput extends ModelInput = ModelInput,
  TOutput extends ModelOutput = ModelOutput
> {
  /** Load model weights and prepare for inference */
  load(): Promise<boolean>;
  /** Run inference with provided input */
  predict(input: TInput): Promise<TOutput>;
  /** Check if model is loaded */
  isLoaded(): boolean;
  /** Get model type */
  getType(): ModelType;
  /** Get model name */
  getName(): string;
  /** Get model metadata */
  getMetadata(): Record<string, any>;
  /** Dispose model resources */
  dispose(): void;
}

/**
 * Hardware abstraction layer interface
 */
export interface IHardwareAbstraction {
  /** Initialize hardware detection and setup */
  initialize(): Promise<boolean>;
  /** Get active backend */
  getActiveBackend(): HardwareBackendType | null;
  /** Switch to a different backend */
  switchBackend(backend: HardwareBackendType): Promise<boolean>;
  /** Get optimal backend for a specific model type */
  getOptimalBackendForModel(modelType: ModelType): HardwareBackendType;
  /** Get hardware capabilities */
  getCapabilities(): any;
  /** Check if a backend is supported */
  isBackendSupported(backend: HardwareBackendType): boolean;
  /** Dispose all resources */
  dispose(): void;
}

/**
 * Resource pool interface for managing resource allocation
 */
export interface IResourcePool {
  /** Initialize the resource pool */
  initialize(): Promise<boolean>;
  /** Acquire a resource */
  acquireResource(resourceType: string, options?: any): Promise<any>;
  /** Release a resource */
  releaseResource(resource: any): Promise<void>;
  /** Get statistics about resource usage */
  getStats(): Record<string, any>;
  /** Dispose all resources */
  dispose(): void;
}

/**
 * Storage interface for model weights and tensors
 */
export interface IStorage {
  /** Initialize storage */
  initialize(): Promise<boolean>;
  /** Store item */
  setItem(key: string, value: any): Promise<void>;
  /** Retrieve item */
  getItem<T>(key: string): Promise<T | null>;
  /** Check if item exists */
  hasItem(key: string): Promise<boolean>;
  /** Remove item */
  removeItem(key: string): Promise<void>;
  /** Clear all items */
  clear(): Promise<void>;
}

/**
 * Performance monitor interface for tracking and analyzing performance
 */
export interface IPerformanceMonitor {
  /** Start recording an operation */
  start(operationName: string, metadata?: Record<string, any>): string;
  /** End recording an operation */
  end(id: string): void;
  /** Get performance statistics */
  getStats(): Record<string, any>;
  /** Clear all performance records */
  clear(): void;
}