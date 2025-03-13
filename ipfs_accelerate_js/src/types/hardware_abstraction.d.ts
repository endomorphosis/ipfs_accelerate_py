/**
 * Type definitions for hardware abstraction layer
 */

import { WebGPUBackendType } from './webgpu';
import { WebNNBackendType } from './webnn';

export type HardwareBackendType = WebGPUBackendType | WebNNBackendType | 'wasm' | 'cpu';

export interface HardwareCapabilities {
  browserName: string;
  browserVersion: string;
  platform: string;
  osVersion: string;
  isMobile: boolean;
  webgpuSupported: boolean;
  webgpuFeatures: string[];
  webnnSupported: boolean;
  webnnFeatures: string[];
  wasmSupported: boolean;
  wasmFeatures: string[];
  recommendedBackend: HardwareBackendType;
  memoryLimitMB: number;
}

export interface HardwareBackend {
  initialize(): Promise<boolean>;
  dispose(): void;
  execute?<T = any, U = any>(inputs: T): Promise<U>;
}

export interface ModelLoaderOptions {
  modelId: string;
  modelType: string;
  path?: string;
  backend?: HardwareBackendType;
  options?: Record<string, any>;
}
