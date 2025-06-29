/**
 * Type definitions for OpenVINO Model Server (OVMS) API
 */

import { ApiRequestOptions } from '../types';

export interface OVMSRequestData {
  // Standard OVMS request format
  instances?: any[];
  inputs?: any;
  signature_name?: string;
  [key: string]: any;
}

export interface OVMSResponse {
  predictions?: any[];
  outputs?: any;
  model_name?: string;
  model_version?: string;
  [key: string]: any;
}

export interface OVMSModelMetadata {
  name: string;
  versions: string[];
  platform: string;
  inputs: OVMSModelInput[];
  outputs: OVMSModelOutput[];
  [key: string]: any;
}

export interface OVMSModelInput {
  name: string;
  datatype: string;
  shape: number[];
  layout?: string;
  [key: string]: any;
}

export interface OVMSModelOutput {
  name: string;
  datatype: string;
  shape: number[];
  layout?: string;
  [key: string]: any;
}

export interface OVMSRequestOptions extends ApiRequestOptions {
  version?: string;
  shape?: number[];
  precision?: string;
  config?: any;
  [key: string]: any;
}

export interface OVMSModelConfig {
  batch_size?: number;
  preferred_batch?: number;
  instance_count?: number;
  execution_mode?: 'latency' | 'throughput';
  [key: string]: any;
}

export interface OVMSServerStatistics {
  server_uptime?: number;
  server_version?: string;
  active_models?: number;
  total_requests?: number;
  requests_per_second?: number;
  avg_inference_time?: number;
  cpu_usage?: number;
  memory_usage?: number;
  [key: string]: any;
}

export interface OVMSModelStatistics {
  model: string;
  statistics: {
    requests_processed?: number;
    tokens_generated?: number;
    avg_inference_time?: number;
    throughput?: number;
    errors?: number;
    [key: string]: any;
  };
  [key: string]: any;
}

export interface OVMSQuantizationConfig {
  enabled: boolean;
  method?: string;
  bits?: number;
  [key: string]: any;
}