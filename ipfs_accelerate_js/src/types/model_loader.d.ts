/**
 * Type definitions for model loader
 */

import { HardwareBackendType } from './hardware_abstraction';

export type ModelType = 'text' | 'vision' | 'audio' | 'multimodal';

export interface ModelConfig {
  backend?: HardwareBackendType;
  quantized?: boolean;
  bits?: number;
  mixedPrecision?: boolean;
  tensorFormat?: string;
  cacheResults?: boolean;
  enableOptimizations?: boolean;
  customOptions?: Record<string, any>;
}

export interface ModelLoadOptions {
  modelId: string;
  modelType: ModelType;
  backend?: HardwareBackendType;
  config?: ModelConfig;
}

export interface ModelInfo {
  id: string;
  type: ModelType;
  backend: HardwareBackendType;
  memoryUsage: number;
  size: number;
  parameters: number;
  precision: number;
  isQuantized: boolean;
  isSharded: boolean;
}

export interface Model {
  getInfo(): ModelInfo;
  getBackend(): HardwareBackendType;
  dispose(): void;
  process(input: any): Promise<any>;
  processText?(text: string): Promise<any>;
  processImage?(image: any): Promise<any>;
  processAudio?(audio: any): Promise<any>;
  processMultimodal?(input: any): Promise<any>;
}

export interface ModelLoader {
  initialize(): Promise<void>;
  loadModel(options: ModelLoadOptions): Promise<Model>;
  disposeModel(modelId: string): Promise<void>;
  getLoadedModels(): string[];
  dispose(): Promise<void>;
}