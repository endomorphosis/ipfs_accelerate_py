import { Message } from '../types';

/**
 * Ollama API response format
 */
export interface OllamaResponse {
  model?: string;
  message?: {
    role?: string;
    content?: string;
  };
  response?: string;
  done?: boolean;
  context?: number[];
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
  [key: string]: any;
}

/**
 * Ollama streaming response chunk
 */
export interface OllamaStreamChunk {
  model?: string;
  message?: {
    role?: string;
    content?: string;
  };
  response?: string;
  done?: boolean;
  prompt_eval_count?: number;
  eval_count?: number;
}

/**
 * Ollama API request format
 */
export interface OllamaRequest {
  model: string;
  messages: OllamaMessage[];
  stream: boolean;
  options?: OllamaOptions;
  [key: string]: any;
}

/**
 * Ollama message format
 */
export interface OllamaMessage {
  role: string;
  content: string;
}

/**
 * Ollama request options
 */
export interface OllamaOptions {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repeat_penalty?: number;
  num_predict?: number;
  [key: string]: any;
}

/**
 * Ollama usage statistics
 */
export interface OllamaUsageStats {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  totalTokens: number;
  totalPromptTokens: number;
  totalCompletionTokens: number;
}

/**
 * Ollama request metadata
 */
export interface OllamaRequestMetadata {
  requestId: string;
  startTime: number;
  model: string;
  priority: number;
}
