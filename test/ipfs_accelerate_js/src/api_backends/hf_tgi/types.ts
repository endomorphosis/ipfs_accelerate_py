import { Message, ApiEndpoint } from '../types';

/**
 * Hugging Face Text Generation Inference API Response
 * Can be multiple different formats depending on the endpoint and request:
 * - Single completion: { generated_text: string, ... }
 * - Multiple completions: Array<{ generated_text: string, ... }>
 * - Streaming: { token: { text: string, ... }, ... }
 */
export interface HfTgiResponse {
  generated_text?: string;
  token?: {
    text: string;
    id?: number;
    logprob?: number;
    special?: boolean;
  };
  details?: {
    finish_reason: string;
    logprobs?: Array<{
      token: string;
      logprob: number;
    }>;
    generated_tokens: number;
  };
}

/**
 * Request to the Hugging Face Text Generation Inference API
 */
export interface HfTgiRequest {
  inputs: string;
  parameters?: {
    max_new_tokens?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    return_full_text?: boolean;
    do_sample?: boolean;
    repetition_penalty?: number;
    seed?: number;
    watermark?: boolean;
    [key: string]: any;
  };
  stream?: boolean;
  options?: {
    use_cache?: boolean;
    wait_for_model?: boolean;
    [key: string]: any;
  };
}

/**
 * Options for Hugging Face Text Generation Inference chat API
 */
export interface HfTgiChatOptions {
  model?: string;
  temperature?: number;
  max_new_tokens?: number;
  top_p?: number;
  top_k?: number;
  return_full_text?: boolean;
  do_sample?: boolean;
  repetition_penalty?: number;
  endpointId?: string;
  requestId?: string;
  [key: string]: any;
}

/**
 * Endpoint configuration for Hugging Face Text Generation Inference
 * Extends the base ApiEndpoint interface with HF TGI specific properties
 */
export interface HfTgiEndpoint extends ApiEndpoint {
  // Standard properties from ApiEndpoint (repeated here for clarity)
  id: string;
  apiKey?: string;
  model?: string;
  maxConcurrentRequests?: number;
  queueSize?: number;
  maxRetries?: number;
  initialRetryDelay?: number;
  backoffFactor?: number;
  timeout?: number;
  
  // HF TGI specific properties
  api_key: string;         // HF-specific key (aliases apiKey)
  model_id: string;        // HF-specific model ID (aliases model)
  endpoint_url: string;    // HF-specific endpoint URL
  max_retries?: number;    // HF-specific retry count (aliases maxRetries)
  max_concurrent_requests?: number; // HF-specific concurrency (aliases maxConcurrentRequests)
  initial_retry_delay?: number;     // HF-specific retry delay (aliases initialRetryDelay)
  backoff_factor?: number;         // HF-specific backoff factor

  // HF TGI tracking properties  
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
 * Statistics for a Hugging Face Text Generation Inference endpoint
 */
export interface HfTgiStats {
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
