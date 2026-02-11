import { ApiBackendOptions, ApiMetadata } from '../types';

/**
 * LLVM API specific options
 */
export interface LlvmOptions extends ApiBackendOptions {
  /** Base URL for the LLVM API */
  base_url?: string;
  /** Maximum concurrent requests */
  max_concurrent_requests?: number;
  /** Maximum retry attempts */
  max_retries?: number;
  /** Retry delay in milliseconds */
  retry_delay?: number;
  /** Queue size (maximum number of pending requests) */
  queue_size?: number;
}

/**
 * Model information returned by the LLVM API
 */
export interface ModelInfo {
  /** Model ID */
  model_id: string;
  /** Model status (e.g., 'loaded', 'loading', 'failed') */
  status: string;
  /** Optional model details */
  details?: Record<string, any>;
}

/**
 * Response from the run_inference method
 */
export interface InferenceResponse {
  /** Model ID used for inference */
  model_id: string;
  /** Inference output */
  outputs: string | Record<string, any>;
  /** Inference metadata */
  metadata?: Record<string, any>;
}

/**
 * Response from the list_models method
 */
export interface ListModelsResponse {
  /** List of available model IDs */
  models: string[];
}

/**
 * Options for the run_inference method
 */
export interface InferenceOptions {
  /** Timeout for the request in milliseconds */
  timeout?: number;
  /** Maximum number of tokens to generate (for text models) */
  max_tokens?: number;
  /** Temperature for controlling randomness */
  temperature?: number;
  /** Sample top-k most likely tokens */
  top_k?: number;
  /** Nucleus sampling (top-p) */
  top_p?: number;
  /** Custom model parameters */
  params?: Record<string, any>;
}