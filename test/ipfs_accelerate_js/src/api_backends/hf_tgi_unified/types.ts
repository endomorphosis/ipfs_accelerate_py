import { ApiMetadata, ChatMessage, ChatOptions, BaseStreamChunk, PriorityLevel } from '../types';

/**
 * HF TGI Unified configuration options
 */
export interface HfTgiUnifiedOptions {
  /** Base URL for API requests */
  apiUrl?: string;
  /** Base URL for container requests */
  containerUrl?: string;
  /** Maximum number of retries for failed requests */
  maxRetries?: number;
  /** Timeout in milliseconds for API requests */
  requestTimeout?: number;
  /** Whether to use request queue for managing concurrent requests */
  useRequestQueue?: boolean;
  /** Enable debug logging */
  debug?: boolean;
  /** Use container mode instead of API mode */
  useContainer?: boolean;
  /** Path to container image */
  containerImagePath?: string;
  /** Docker registry for container */
  dockerRegistry?: string;
  /** Container tag */
  containerTag?: string;
  /** GPU device to use for container */
  gpuDevice?: string;
  /** Maximum tokens to generate */
  maxTokens?: number;
  /** Sampling temperature */
  temperature?: number;
  /** Top-p sampling parameter */
  topP?: number;
  /** Top-k sampling parameter */
  topK?: number;
  /** Repetition penalty */
  repetitionPenalty?: number;
  /** Stop sequences */
  stopSequences?: string[];
}

/**
 * HF TGI Unified API metadata
 */
export type HfTgiUnifiedApiMetadata = ApiMetadata & {
  /** Hugging Face API key */
  hf_api_key?: string;
  /** Container URL for hosted HF TGI */
  hf_container_url?: string;
  /** Docker registry for container deployment */
  docker_registry?: string;
  /** Container tag for deployment */
  container_tag?: string;
  /** GPU device ID for container */
  gpu_device?: string;
  /** Model ID for text generation */
  model_id?: string;
}

/**
 * Container information for HF TGI Unified
 */
export interface ContainerInfo {
  /** Container ID */
  containerId: string;
  /** Container host */
  host: string;
  /** Container port */
  port: number;
  /** Container status */
  status: 'running' | 'stopped' | 'failed' | 'unknown';
  /** Time container was started */
  startTime?: Date;
  /** Error message if container failed to start */
  error?: string;
}

/**
 * Model information response from HF TGI
 */
export interface HfTgiModelInfo {
  /** Model ID from Hugging Face Hub */
  model_id: string;
  /** Model status */
  status: string;
  /** Model revision */
  revision?: string;
  /** Model framework */
  framework?: string;
  /** Whether the model is quantized */
  quantized?: boolean;
  /** Available parameters for text generation */
  parameters?: string[];
  /** Maximum context size */
  max_input_length?: number;
  /** Maximum new tokens */
  max_total_tokens?: number;
}

/**
 * Text generation request options
 */
export interface TextGenerationOptions {
  /** Model ID to use */
  model?: string;
  /** Maximum tokens to generate */
  maxTokens?: number;
  /** Sampling temperature */
  temperature?: number;
  /** Top-p sampling parameter */
  topP?: number;
  /** Top-k sampling parameter */
  topK?: number;
  /** Repetition penalty */
  repetitionPenalty?: number;
  /** Whether to stream the response */
  stream?: boolean;
  /** Stop sequences */
  stopSequences?: string[];
  /** Request priority */
  priority?: PriorityLevel;
  /** Whether to include input text in response */
  includeInputText?: boolean;
  /** Stream callback for streaming responses */
  streamCallback?: (chunk: StreamChunk) => void;
}

/**
 * Chat generation options for HF TGI
 */
export interface ChatGenerationOptions extends TextGenerationOptions {
  /** Chat prompt template */
  promptTemplate?: string;
  /** System message */
  systemMessage?: string;
}

/**
 * Text generation request sent to HF TGI
 */
export interface HfTgiTextGenerationRequest {
  /** Input text to generate from */
  inputs: string;
  /** Parameters for text generation */
  parameters?: {
    /** Maximum new tokens to generate */
    max_new_tokens?: number;
    /** Sampling temperature */
    temperature?: number;
    /** Top-p sampling parameter */
    top_p?: number;
    /** Top-k sampling parameter */
    top_k?: number;
    /** Repetition penalty */
    repetition_penalty?: number;
    /** Whether to return the full text or just the generated part */
    return_full_text?: boolean;
    /** Whether to stream the response */
    stream?: boolean;
    /** Stop sequences */
    stop?: string[];
  };
}

/**
 * Text generation response from HF TGI API
 */
export interface HfTgiTextGenerationResponse {
  /** Generated text */
  generated_text: string;
  /** Token details if requested */
  details?: {
    /** Finished reason */
    finish_reason: string;
    /** Number of generated tokens */
    generated_tokens: number;
    /** Additional details */
    [key: string]: any;
  };
}

/**
 * Streaming response chunk from HF TGI API
 */
export interface HfTgiStreamChunk {
  /** Generated token text */
  token: {
    /** Token ID */
    id: number;
    /** Token text */
    text: string;
    /** Token logprob */
    logprob: number;
    /** Whether this is a special token */
    special: boolean;
  };
  /** Whether the generation is complete */
  generated_text?: string | null;
  /** Details if generation is complete */
  details?: {
    /** Finished reason */
    finish_reason: string;
    /** Number of generated tokens */
    generated_tokens: number;
  };
}

/**
 * Stream chunk for text generation
 */
export interface StreamChunk extends BaseStreamChunk {
  /** Generated text (incremental or complete) */
  text: string;
  /** Whether this is the final chunk */
  done: boolean;
  /** Full generated text (only on final chunk) */
  fullText?: string;
  /** Token details */
  token?: {
    /** Token ID */
    id: number;
    /** Token text */
    text: string;
    /** Token logprob */
    logprob: number;
    /** Whether this is a special token */
    special: boolean;
  };
}

/**
 * Deployment configuration for container-based HF TGI
 */
export interface DeploymentConfig {
  /** Docker registry to use */
  dockerRegistry: string;
  /** Container tag */
  containerTag: string;
  /** GPU device to use */
  gpuDevice: string;
  /** Model ID to load */
  modelId: string;
  /** Container port */
  port: number;
  /** Container environment variables */
  env?: Record<string, string>;
  /** Container volume mounts */
  volumes?: string[];
  /** Container network */
  network?: string;
  /** Additional container parameters */
  parameters?: string[];
  /** Maximum input length */
  maxInputLength?: number;
  /** Disable GPU for testing */
  disableGpu?: boolean;
}

/**
 * Performance metrics for HF TGI
 */
export interface PerformanceMetrics {
  /** Time in milliseconds for single generation */
  singleGenerationTime: number;
  /** Tokens per second */
  tokensPerSecond: number;
  /** Generated tokens */
  generatedTokens: number;
  /** Input tokens */
  inputTokens: number;
  /** Memory usage in MB */
  memoryUsageMb?: number;
  /** Test timestamp */
  timestamp: string;
}