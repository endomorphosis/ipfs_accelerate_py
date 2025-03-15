/**
 * Common types used across all API backends
 */

export interface ApiResources {
  [key: string]: any;
}

export interface ApiMetadata {
  [key: string]: any;
}

export interface ApiEndpoint {
  id: string;
  apiKey: string;  // Must be required for strict typing compliance
  model?: string;
  maxConcurrentRequests?: number;
  queueSize?: number;
  maxRetries?: number;
  initialRetryDelay?: number;
  backoffFactor?: number;
  timeout?: number;
  [key: string]: any;  // Allow additional properties for implementation-specific requirements
}

export interface ApiEndpointStats {
  requests: number;
  success: number;
  errors: number;
  [key: string]: any;
}

export interface Message {
  role: string;
  content: string | any[];
  [key: string]: any;
}

export interface ChatCompletionResponse {
  id?: string;
  content?: any;
  role?: string;
  model?: string;
  usage?: {
    inputTokens?: number;
    outputTokens?: number;
    [key: string]: any;
  };
  [key: string]: any;
}

export interface StreamChunk {
  content?: string;
  role?: string;
  type?: string;
  delta?: any;
  [key: string]: any;
}

export interface ApiRequestOptions {
  signal?: AbortSignal;
  requestId?: string;
  timeout?: number;
  [key: string]: any;
}

export interface ApiError extends Error {
  statusCode?: number;
  type?: string;
  isRateLimitError?: boolean;
  isAuthError?: boolean;
  isTransientError?: boolean;
  isTimeout?: boolean;
  retryAfter?: number;
  [key: string]: any;
}

export interface RequestQueueItem {
  data: any;
  apiKey: string;
  requestId: string;
  resolve: (value: any) => void;
  reject: (reason: any) => void;
  options?: ApiRequestOptions;
}

export interface TestResults {
  [key: string]: string | boolean | number | object;
}