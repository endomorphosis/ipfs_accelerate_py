/**
 * Type definitions for hardware abstraction layer
 */

export type HardwareBackendType = 'webgpu' | 'webnn' | 'wasm' | 'cpu';

export interface HardwareInfo {
  name: string;
  type: HardwareBackendType;
  isSimulated: boolean;
  vendor?: string;
  version?: string;
  memorySize?: number;
  capabilities: {
    shaderModel?: string;
    computeSupport?: boolean;
    int8Support?: boolean;
    int4Support?: boolean;
    float16Support?: boolean;
  };
}

export interface HardwareBackend {
  initialize(): Promise<boolean>;
  dispose(): void;
  getInfo(): HardwareInfo;
  runInference(model: any, inputs: any): Promise<any>;
  getCapabilities(): any;
}

export interface HardwareAbstractionOptions {
  logging?: boolean;
  preferredBackends?: HardwareBackendType[];
  fallbackOrder?: HardwareBackendType[];
}

export interface HardwareCapabilities {
  browserName: string;
  backends: {
    webgpu?: any;
    webnn?: any;
    wasm?: any;
    cpu?: any;
  };
}

export interface HardwarePreferences {
  priorityList?: HardwareBackendType[];
  disallowList?: HardwareBackendType[];
  requireSharedMemory?: boolean;
  requireInt8?: boolean;
  requireFloat16?: boolean;
  preferSimulation?: boolean;
}

export interface ModelType {
  text: string;
  vision: string;
  audio: string;
  multimodal: string;
}