import { ApiBackendOptions, ApiMetadata } from '../types';

/**
 * Configuration for an S3 endpoint
 */
export interface S3EndpointConfig {
  /** The S3 endpoint URL */
  endpoint_url: string;
  /** Access key for the S3 service */
  access_key: string;
  /** Secret key for the S3 service */
  secret_key: string;
  /** Maximum concurrent requests to this endpoint (default: 5) */
  max_concurrent?: number;
  /** Number of failures before the circuit breaker opens (default: 3) */
  circuit_breaker_threshold?: number;
  /** Number of retry attempts for failed requests (default: 2) */
  retries?: number;
  /** Initial delay before retrying a failed request in ms (default: 1000) */
  initial_retry_delay?: number;
  /** Backoff factor for exponential retry delay (default: 2) */
  backoff_factor?: number;
  /** Timeout for S3 requests in ms (default: 30000) */
  timeout?: number;
}

/**
 * S3 endpoint handler metadata
 */
export interface S3EndpointHandlerMetadata {
  endpoint_url: string;
  access_key: string;
  secret_key: string;
  max_concurrent: number;
  circuit_breaker_threshold: number;
  retries: number;
  initial_retry_delay: number;
  backoff_factor: number;
  timeout: number;
}

/**
 * Request options for S3 operations
 */
export interface S3RequestOptions {
  /** The S3 bucket name */
  bucket: string;
  /** The S3 object key */
  key?: string;
  /** The file path for upload/download operations */
  file_path?: string;
  /** The prefix for list operations */
  prefix?: string;
  /** Maximum number of keys to return in list operations */
  max_keys?: number;
  /** Custom request timeout in ms */
  timeout?: number;
  /** Priority level for the request */
  priority?: 'HIGH' | 'NORMAL' | 'LOW';
}

/**
 * S3 circuit breaker states
 */
export enum CircuitBreakerState {
  CLOSED = 'CLOSED',
  OPEN = 'OPEN',
  HALF_OPEN = 'HALF_OPEN'
}

/**
 * Options specific to the S3 Kit backend
 */
export interface S3KitOptions extends ApiBackendOptions {
  /** Default S3 endpoint URL */
  s3_endpoint_url?: string;
  /** Maximum concurrent requests (default: 10) */
  max_concurrent_requests?: number;
  /** Size limit for the request queue (default: 100) */
  queue_size?: number;
  /** Maximum retry attempts for failed requests (default: 3) */
  max_retries?: number;
  /** Initial delay before retrying a failed request in ms (default: 1000) */
  initial_retry_delay?: number;
  /** Backoff factor for exponential retry delay (default: 2) */
  backoff_factor?: number;
  /** Default timeout for S3 requests in ms (default: 30000) */
  default_timeout?: number;
  /** Selection strategy for endpoint multiplexing (default: 'round-robin') */
  endpoint_selection_strategy?: 'round-robin' | 'least-loaded';
  /** Circuit breaker configuration */
  circuit_breaker?: {
    /** Threshold for consecutive failures before opening the breaker */
    threshold: number;
    /** Timeout period in ms before attempting to half-open the breaker */
    timeout: number;
  };
}