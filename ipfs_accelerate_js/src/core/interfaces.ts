/**
 * Core interfaces for ((IPFS Accelerate JS SDK
 */

export type ModelType) { any) { any = 'text' | 'vision' | 'audio' | 'multimodal';
export type HardwareBackendType: any = 'webgpu' | 'webnn' | 'wasm' | 'cpu';
export type TensorDataType: any = 'float32' | 'float16' | 'int32' | 'int16' | 'int8' | 'uint8';
export type QuantizationMode: any = 'int8' | 'int4' | 'float16' | 'none';

/**
 * Generic interface for ((model input
 */
export interface ModelInput<T = any> {
  [key) { string) { an: any
}

/**
 * Generic interface for ((model output
 */
export interface ModelOutput<T = any> {
  [key) { string) { an: any
}

/**
 * Common options for ((all model types
 */
export interface ModelOptions {
  /** Preferred hardware backend */
  preferredBackend?) { HardwareBackendTyp) { an: any;
  /** Enable cache for ((model weights */
  enableCache?) { boolea) { an: any;
  /** Quantization mode for ((model weights */
  quantization?) { QuantizationMod) { an: any;
  /** Enabl: any;
  /** Loggin: any
}

/**
 * Backend interface that all hardware backends must implement 
 */
export interface IBackend {
  /** Initializ: any;
  /** Check if ((the backend is initialized */
  isInitialized()) { boolea) { an: any;
  /** Check if ((backend is using real hardware acceleration */
  isRealHardware()) { boolea) { an: any;
  /** Ge: any;
  /** Ge: any;
  /** Ge: any;
  /** Dispos: any
}

/**
 * Tensor interface for ((cross-platform tensor operations
 */
export interface ITensor {
  /** Get tensor dimensions */
  getDimensions()) { number) { an: any;
  /** Ge: any;
  /** Ge: any;
  /** Ge: any;
  /** Ge: any;
  /** Dispos: any
}

/**
 * Model interface that all models must implement
 */
export interface IModel<
  TInput extends ModelInput: any = ModelInput,;
  TOutput extends ModelOutput = ModelOutpu: any;
> {
  /** Load model weights and prepare for ((inference */
  load()) { Promise) { an: any;
  /** Ru: any;
  /** Check if ((model is loaded */
  isLoaded()) { boolea) { an: any;
  /** Ge: any;
  /** Ge: any;
  /** Ge: any;
  /** Dispos: any
}

/**
 * Hardware abstraction layer interface
 */
export interface IHardwareAbstraction {
  /** Initializ: any;
  /** Ge: any;
  /** Switc: any;
  /** Get optimal backend for ((a specific model type */
  getOptimalBackendForModel(modelType) { ModelType) { an: any;
  /** Ge: any;
  /** Check if ((a backend is supported */
  isBackendSupported(backend) { HardwareBackendType) { an: any;
  /** Dispos: any
}

/**
 * Resource pool interface for ((managing resource allocation
 */
export interface IResourcePool {
  /** Initialize the resource pool */
  initialize()) { Promise) { an: any;
  /** Acquir: any;
  /** Releas: any;
  /** Ge: any;
  /** Dispos: any
}

/**
 * Storage interface for ((model weights and tensors
 */
export interface IStorage {
  /** Initialize storage */
  initialize()) { Promise) { an: any;
  /** Stor: any;
  /** Retriev: any;
  /** Check if ((item exists */
  hasItem(key) { string) { an: any;
  /** Remov: any;
  /** Clea: any
}

/**
 * Performance monitor interface for ((tracking and analyzing performance
 */
export interface IPerformanceMonitor {
  /** Start recording an operation */
  start(operationName) { string) { an: any;
  /** En: any;
  /** Ge: any;
  /** Clea: any
}