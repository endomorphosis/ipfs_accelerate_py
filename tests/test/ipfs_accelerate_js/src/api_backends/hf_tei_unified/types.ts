import { ApiMetadata, ChatMessage, ChatOptions, BaseStreamChunk } from '../types';

/**
 * HF TEI Unified configuration options
 */
export interface HfTeiUnifiedOptions {
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
}

/**
 * HF TEI Unified API metadata
 */
export type HfTeiUnifiedApiMetadata = ApiMetadata & {
  /** Hugging Face API key */
  hf_api_key?: string;
  /** Container URL for hosted HF TEI */
  hf_container_url?: string;
  /** Docker registry for container deployment */
  docker_registry?: string;
  /** Container tag for deployment */
  container_tag?: string;
  /** GPU device ID for container */
  gpu_device?: string;
  /** Model ID for embedding model */
  model_id?: string;
}

/**
 * Container information for HF TEI Unified
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
 * Model information response from HF TEI
 */
export interface HfTeiModelInfo {
  /** Model ID from Hugging Face Hub */
  model_id: string;
  /** Embedding dimension */
  dim: number;
  /** Model status */
  status: string;
  /** Model revision */
  revision?: string;
  /** Model framework */
  framework?: string;
  /** Whether the model is quantized */
  quantized?: boolean;
}

/**
 * Embedding request options
 */
export interface EmbeddingOptions {
  /** Model ID to use for embedding */
  model?: string;
  /** Embedding normalization method */
  normalize?: boolean;
  /** Truncation strategy */
  truncation?: boolean;
  /** Maximum tokens to process */
  maxTokens?: number;
  /** Request priority */
  priority?: 'HIGH' | 'NORMAL' | 'LOW';
}

/**
 * Embedding request sent to HF TEI
 */
export interface HfTeiEmbeddingRequest {
  /** Input text or texts to embed */
  inputs: string | string[];
  /** Whether to normalize the embeddings */
  normalize?: boolean;
  /** Truncation settings */
  truncation?: boolean;
  /** Maximum tokens to process */
  max_tokens?: number;
}

/**
 * Embedding response from HF TEI API
 */
export type HfTeiEmbeddingResponse = number[] | number[][];

/**
 * Performance metrics for HF TEI
 */
export interface PerformanceMetrics {
  /** Time in milliseconds for single embedding */
  singleEmbeddingTime: number;
  /** Time in milliseconds for batch embedding */
  batchEmbeddingTime: number;
  /** Sentences processed per second */
  sentencesPerSecond: number;
  /** Speedup factor for batch processing */
  batchSpeedupFactor: number;
  /** Memory usage in MB */
  memoryUsageMb?: number;
  /** Test timestamp */
  timestamp: string;
}

/**
 * Deployment configuration for container-based HF TEI
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
}