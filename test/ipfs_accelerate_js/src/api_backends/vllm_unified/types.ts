import { Message } from '../types';

export interface VllmRequest {
  prompt?: string;
  messages?: Message[];
  model?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  use_beam_search?: boolean;
  n?: number;
  stop?: string[] | string;
  presence_penalty?: number;
  frequency_penalty?: number;
  best_of?: number;
  ignore_eos?: boolean;
  stream?: boolean;
  echo?: boolean;
}

export interface VllmUnifiedResponse {
  text: string;
  metadata: {
    finish_reason?: string;
    model?: string;
    is_streaming?: boolean;
    usage?: {
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
    };
  };
}

export interface VllmBatchResponse {
  texts: string[];
  metadata: {
    finish_reasons?: string[];
    model?: string;
    usage?: {
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
    };
  };
}

export interface VllmStreamChunk {
  text: string;
  metadata: {
    finish_reason?: string | null;
    is_streaming?: boolean;
  };
}

export interface VllmModelInfo {
  model: string;
  max_model_len: number;
  num_gpu: number;
  dtype: string;
  gpu_memory_utilization: number;
  quantization?: {
    enabled: boolean;
    method?: string;
    bits?: number;
  };
  lora_adapters?: any[];
}

export interface VllmModelStatistics {
  model: string;
  statistics: {
    requests_processed: number;
    tokens_generated: number;
    avg_tokens_per_request: number;
    max_tokens_per_request: number;
    avg_generation_time: number;
    throughput: number;
    errors: number;
    uptime: number;
  };
}

export interface VllmLoraAdapter {
  id: string;
  name: string;
  base_model: string;
  size_mb: number;
  active: boolean;
}

export interface VllmLoadLoraResponse {
  success: boolean;
  adapter_id: string;
  message: string;
}

export interface VllmQuantizationConfig {
  enabled: boolean;
  method?: string;
  bits?: number;
}

export interface VllmQuantizationResponse {
  success: boolean;
  message: string;
  model: string;
  quantization: VllmQuantizationConfig;
}

/**
 * Container configuration for the VLLM Unified backend
 */
export interface VllmContainerConfig {
  /**
   * Docker image to use for the container
   */
  image: string;
  
  /**
   * Whether to enable GPU support
   */
  gpu: boolean;
  
  /**
   * Path to the models directory on the host
   */
  models_path: string;
  
  /**
   * Path to the configuration directory on the host
   */
  config_path: string;
  
  /**
   * API port to expose
   */
  api_port: number;
  
  /**
   * Number of GPUs to use for tensor parallelism
   */
  tensor_parallel_size: number;
  
  /**
   * Maximum model context length (0 means use default)
   */
  max_model_len: number;
  
  /**
   * GPU memory utilization (0.0-1.0)
   */
  gpu_memory_utilization: number;
  
  /**
   * Quantization method to use (null = no quantization)
   */
  quantization: string | null;
  
  /**
   * Whether to trust remote code when loading models
   */
  trust_remote_code: boolean;
  
  /**
   * Custom arguments to pass to the container
   */
  custom_args: string;
}

/**
 * Container status for the VLLM Unified backend
 */
export type VllmContainerStatus = 'stopped' | 'starting' | 'running' | 'stopping' | 'error';

/**
 * Container metrics for the VLLM Unified backend
 */
export interface VllmContainerMetrics {
  /**
   * CPU usage percentage (0-100)
   */
  cpu_usage: number;
  
  /**
   * Memory usage in MB
   */
  memory_usage: number;
  
  /**
   * GPU memory usage in MB (if GPU is enabled)
   */
  gpu_memory_usage?: number;
  
  /**
   * Number of requests processed
   */
  requests_processed: number;
  
  /**
   * Number of tokens generated
   */
  tokens_generated: number;
  
  /**
   * Average latency in ms
   */
  average_latency: number;
  
  /**
   * Throughput in tokens per second
   */
  throughput: number;
}