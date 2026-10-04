/**
 * Type definitions for VLLM API
 */

import { ApiRequestOptions } from '../types';

export interface VLLMRequestData {
  prompt?: string;
  messages?: Array<{ role: string; content: string | any[] }>;
  model?: string;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  use_beam_search?: boolean;
  stop?: string | string[];
  n?: number;
  best_of?: number;
  stream?: boolean;
  [key: string]: any;
}

export interface VLLMResponse {
  text?: string;
  texts?: string[];
  message?: {
    content: string;
    role: string;
    [key: string]: any;
  };
  choices?: Array<{
    text?: string;
    message?: {
      content?: string;
      role?: string;
      [key: string]: any;
    };
    delta?: {
      content?: string;
      role?: string;
      [key: string]: any;
    };
    index?: number;
    finish_reason?: string;
    [key: string]: any;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    [key: string]: any;
  };
  metadata?: {
    finish_reason?: string;
    model?: string;
    usage?: {
      prompt_tokens?: number;
      completion_tokens?: number;
      total_tokens?: number;
      [key: string]: any;
    };
    is_streaming?: boolean;
    [key: string]: any;
  };
  [key: string]: any;
}

export interface VLLMModelInfo {
  model: string;
  max_model_len: number;
  num_gpu: number;
  dtype: string;
  gpu_memory_utilization: number;
  quantization?: {
    enabled: boolean;
    method?: string;
    [key: string]: any;
  };
  lora_adapters?: string[];
  [key: string]: any;
}

export interface VLLMModelStatistics {
  model: string;
  statistics: {
    requests_processed?: number;
    tokens_generated?: number;
    avg_tokens_per_request?: number;
    max_tokens_per_request?: number;
    avg_generation_time?: number;
    throughput?: number;
    errors?: number;
    uptime?: number;
    [key: string]: any;
  };
  [key: string]: any;
}

export interface VLLMLoraAdapter {
  id: string;
  name: string;
  base_model: string;
  size_mb: number;
  active: boolean;
  [key: string]: any;
}

export interface VLLMQuantizationConfig {
  enabled: boolean;
  method?: string;
  bits?: number;
  [key: string]: any;
}

export interface VLLMRequestOptions extends ApiRequestOptions {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  stop?: string | string[];
  [key: string]: any;
}