/**
 * Common interfaces for IPFS Accelerate JavaScript SDK
 */

// Hardware interfaces
export interface HardwareBackend {
  initialize(): Promise<boolean>;
  dispose(): void;
  execute<T = any, U = any>(inputs: T): Promise<U>;
}

export interface HardwarePreferences {
  backendOrder?: string[];
  modelPreferences?: Record<string, string>;
  options?: Record<string, any>;
}

// Model interfaces
export interface ModelConfig {
  id: string;
  type: string;
  path?: string;
  options?: Record<string, any>;
}

export interface Model {
  id: string;
  type: string;
  execute<T = any, U = any>(inputs: T): Promise<U>;
}

// Tensor interfaces
export interface TensorOptions {
  dtype?: string;
  device?: string;
  requiresGrad?: boolean;
}

export interface TensorShape {
  dimensions: number[];
  numel: number;
  strides?: number[];
}

export interface SharedTensorOptions {
  name: string;
  shape: number[];
  dtype?: string;
  storage_type?: string;
  producer_model?: string;
  consumer_models?: string[];
}

// Resource Pool interfaces
export interface ResourcePoolOptions {
  maxConnections?: number;
  browserPreferences?: Record<string, string>;
  adaptiveScaling?: boolean;
  enableFaultTolerance?: boolean;
  recoveryStrategy?: string;
  stateSyncInterval?: number;
  redundancyFactor?: number;
}
