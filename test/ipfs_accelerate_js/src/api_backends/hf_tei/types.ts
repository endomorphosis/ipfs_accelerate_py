import { ApiEndpoint, Message } from '../types';

/**
 * Response from the Hugging Face Text Embedding Inference API
 * Can be multiple different formats:
 * - Array format: number[][] (array of embedding vectors)
 * - Object format: { embeddings: number[][] }
 */
export interface HfTeiResponse {
  embeddings?: number[][];
  embedding?: number[];
  model?: string;
  dimensions?: number;
  implementation_type?: string;
  request_id?: string;
}

/**
 * Request to the Hugging Face Text Embedding Inference API
 */
export interface HfTeiRequest {
  inputs: string | string[];
  model?: string;
  options?: {
    normalize?: boolean;
    use_cache?: boolean;
    wait_for_model?: boolean;
    [key: string]: any;
  };
}

/**
 * Options for Hugging Face Text Embedding Inference API
 */
export interface HfTeiOptions {
  model?: string;
  normalize?: boolean;
  endpointId?: string;
  requestId?: string;
  [key: string]: any;
}

/**
 * Endpoint configuration for Hugging Face Text Embedding Inference
 * Extends the base ApiEndpoint interface with HF TEI specific properties
 */
export interface HfTeiEndpoint extends ApiEndpoint {
  // Standard properties from ApiEndpoint
  id: string;
  apiKey: string;
  model?: string;
  maxConcurrentRequests?: number;
  queueSize?: number;
  maxRetries?: number;
  initialRetryDelay?: number;
  backoffFactor?: number;
  timeout?: number;
  
  // HF TEI specific properties
  api_key: string;         // HF-specific key (aliases apiKey)
  model_id: string;        // HF-specific model ID (aliases model)
  endpoint_url: string;    // HF-specific endpoint URL
  max_retries?: number;    // HF-specific retry count (aliases maxRetries)
  max_concurrent_requests?: number; // HF-specific concurrency (aliases maxConcurrentRequests)
  initial_retry_delay?: number;     // HF-specific retry delay (aliases initialRetryDelay)
  backoff_factor?: number;         // HF-specific backoff factor

  // HF TEI tracking properties  
  successful_requests: number;
  failed_requests: number;
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  current_requests: number;
  queue_processing: boolean;
  request_queue: Array<any>;
  last_request_at: number | null;
  created_at: number;
  
  // Allow additional properties 
  [key: string]: any;
}

/**
 * Statistics for a Hugging Face Text Embedding Inference endpoint
 */
export interface HfTeiStats {
  endpoint_id: string;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  created_at: number;
  last_request_at: number | null;
  current_queue_size: number;
  current_requests: number;
}
