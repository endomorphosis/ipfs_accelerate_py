/**
 * Core interfaces for IPFS Accelerate JavaScript SDK
 */

// Hardware interfaces
export interface HardwareBackend {
  initialize(): Promise<boolean>;
  destroy(): void;
  execute<T = any, U = any>(inputs: T): Promise<U>;
}

export interface HardwarePreferences {
  backendOrder: string[];
  modelPreferences: Record<string, string>;
  options: Record<string, any>;
}

// Model interfaces
export interface ModelConfig {
  id: string;
  type: string;
  path: string;
  options: Record<string, any>;
}

export interface Model {
  id: string;
  type: string;
  execute<T = any, U = any>(inputs: T): Promise<U>;
}

// WebGPU interfaces
export interface GPUBufferDescriptor {
  size: number;
  usage: number;
  mappedAtCreation?: boolean;
}

export interface GPUShaderModuleDescriptor {
  code: string;
}

export interface GPUBindGroupDescriptor {
  layout: any;
  entries: GPUBindGroupEntry[];
}

export interface GPUBindGroupEntry {
  binding: number;
  resource: any;
}

export interface GPUComputePipelineDescriptor {
  layout: any;
  compute: {
    module: any;
    entryPoint: string;
  };
}

// WebNN interfaces
export interface MLOperandDescriptor {
  type: string;
  dimensions: number[];
}

export interface MLOperand {}

export interface MLGraph {
  compute(inputs: Record<string, any>): Promise<Record<string, any>>;
}

export interface MLContext {}

export interface MLGraphBuilder {
  constant(value: any, dimensions?: number[]): MLOperand;
  input(name: string, descriptor: MLOperandDescriptor): MLOperand;
  build(outputs: Record<string, MLOperand>): Promise<MLGraph>;
}

// Resource Pool interfaces
export interface ResourcePoolConnection {
  id: string;
  type: string;
  status: string;
  created: Date;
  resources: Record<string, any>;
}

export interface ResourcePoolOptions {
  maxConnections: number;
  browserPreferences: Record<string, string>;
  adaptiveScaling: boolean;
  enableFaultTolerance: boolean;
  recoveryStrategy: string;
  stateSyncInterval: number;
  redundancyFactor: number;
}

// Browser interfaces
export interface BrowserCapabilities {
  browserName: string;
  browserVersion: string;
  isMobile: boolean;
  platform: string;
  osVersion: string;
  webgpuSupported: boolean;
  webgpuFeatures: string[];
  webnnSupported: boolean;
  webnnFeatures: string[];
  wasmSupported: boolean;
  wasmFeatures: string[];
  metalApiSupported: boolean;
  metalApiVersion: string;
  recommendedBackend: string;
  memoryLimitMB: number;
}

// Optimization interfaces
export interface OptimizationConfig {
  memoryOptimization: boolean;
  progressiveLoading: boolean;
  useQuantization: boolean;
  precision: string;
  maxChunkSizeMB: number;
  parallelLoading: boolean;
  specialOptimizations: Record<string, any>;
}

// Tensor interfaces
export interface Tensor {
  shape: number[];
  data: Float32Array | Int32Array | Uint8Array;
  dtype: string;
}

// Shared tensor memory interface
export interface TensorSharing {
  shareableTypes: string[];
  enableSharing: boolean;
  shareTensor(id: string, tensor: Tensor): any;
  getTensor(id: string): Tensor | null;
  releaseSharedTensors(ids: string[]): void;
}
