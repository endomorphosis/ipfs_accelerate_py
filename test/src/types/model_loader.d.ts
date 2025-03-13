/**
 * Type definitions for model loader
 */

import { HardwareBackendType } from './hardware_abstraction';

export interface ModelConfig {
  id: string;
  type: string;
  path: string;
  options?: Record<string, any>;
}

export interface Model {
  id: string;
  type: string;
  execute<T = any, U = any>(inputs: T): Promise<U>;
}

export interface ModelLoaderOptions {
  modelId: string;
  modelType: string;
  backend?: HardwareBackendType;
  options?: Record<string, any>;
}
