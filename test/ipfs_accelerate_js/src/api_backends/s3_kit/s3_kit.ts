import { BaseApiBackend } from '../base';
import { ApiBackendOptions, ApiMetadata, BaseRequestOptions } from '../types';
import {
  S3KitOptions,
  S3EndpointConfig,
  S3EndpointHandlerMetadata,
  S3RequestOptions,
  CircuitBreakerState
} from './types';
import * as fs from 'fs';
import * as path from 'path';

/**
 * S3 Kit backend for handling S3-compatible object storage operations
 * Supports multiple endpoints, circuit breaker pattern, and connection multiplexing
 */
export class S3Kit extends BaseApiBackend {
  private endpoints: Map<string, S3EndpointHandler>;
  private endpointsLock: { acquire: () => void; release: () => void };
  private lastUsed: Map<string, number>;
  private requestsPerEndpoint: Map<string, number>;
  private endpointSelectionStrategy: 'round-robin' | 'least-loaded';
  
  // Queue and request management properties
  protected queueProcessing: boolean;
  protected requestQueue: Array<{
    id: string;
    operation: string;
    options: S3RequestOptions;
    priority: 'HIGH' | 'NORMAL' | 'LOW';
    resolve: (value: any) => void;
    reject: (reason: any) => void;
    timestamp: number;
    retryCount: number;
  }>;
  
  // Configuration properties
  protected maxConcurrentRequests: number;
  protected queueSize: number;
  protected maxRetries: number;
  protected initialRetryDelay: number;
  protected backoffFactor: number;
  protected defaultTimeout: number;
  protected activeRequests: number;
  protected circuitBreakerThreshold: number;
  protected circuitBreakerTimeout: number;

  /**
   * Creates a new S3 Kit instance for handling S3-compatible storage operations
   * @param options - Configuration options for the S3 Kit
   * @param metadata - API metadata including S3 access credentials
   */
  constructor(options: S3KitOptions = {}, metadata: ApiMetadata = {}) {
    super(options, metadata);
    
    // Initialize configuration from options
    this.maxConcurrentRequests = options.max_concurrent_requests || 10;
    this.queueSize = options.queue_size || 100;
    this.maxRetries = options.max_retries || 3;
    this.initialRetryDelay = options.initial_retry_delay || 1000;
    this.backoffFactor = options.backoff_factor || 2;
    this.defaultTimeout = options.default_timeout || 30000;
    this.endpointSelectionStrategy = options.endpoint_selection_strategy || 'round-robin';
    this.circuitBreakerThreshold = options.circuit_breaker?.threshold || 3;
    this.circuitBreakerTimeout = options.circuit_breaker?.timeout || 60000;
    
    // Initialize request management
    this.activeRequests = 0;
    this.queueProcessing = false;
    this.requestQueue = [];
    
    // Initialize endpoint management
    this.endpoints = new Map();
    this.endpointsLock = this.createMutex();
    this.lastUsed = new Map();
    this.requestsPerEndpoint = new Map();
    
    // Initialize with default endpoint if provided
    const defaultEndpoint = this.getDefaultS3Endpoint();
    if (defaultEndpoint) {
      this.addEndpoint('default', defaultEndpoint, this.getDefaultS3AccessKey(), this.getDefaultS3SecretKey());
    }
  }
  
  /**
   * Get the API key from metadata or environment variables
   * @returns The API key or null if not found
   */
  getApiKey(): string | null {
    return this.metadata?.s3cfg?.accessKey || 
           process.env.S3_ACCESS_KEY || 
           null;
  }
  
  /**
   * Get the default model for this API (not applicable for S3)
   * @returns Empty string (S3 doesn't use models)
   */
  getDefaultModel(): string {
    return '';
  }
  
  /**
   * Check if a model is compatible with this API (not applicable for S3)
   * @param model - Model name
   * @returns Always false (S3 doesn't support models)
   */
  isCompatibleModel(model: string): boolean {
    return false;
  }
  
  /**
   * Get the default S3 endpoint URL from metadata or environment variables
   * @returns The S3 endpoint URL or null if not found
   */
  getDefaultS3Endpoint(): string | null {
    return this.metadata?.s3cfg?.endpoint || 
           process.env.S3_ENDPOINT ||
           null;
  }
  
  /**
   * Get the default S3 access key from metadata or environment variables
   * @returns The S3 access key or null if not found
   */
  getDefaultS3AccessKey(): string | null {
    return this.metadata?.s3cfg?.accessKey || 
           process.env.S3_ACCESS_KEY || 
           null;
  }
  
  /**
   * Get the default S3 secret key from metadata or environment variables
   * @returns The S3 secret key or null if not found
   */
  getDefaultS3SecretKey(): string | null {
    return this.metadata?.s3cfg?.secretKey || 
           process.env.S3_SECRET_KEY || 
           null;
  }
  
  /**
   * Create a mutex for thread-safe operations
   * Simple implementation for JavaScript single-threaded environment
   */
  private createMutex() {
    let locked = false;
    return {
      acquire: function() {
        // In a single-threaded environment like JavaScript, this is mainly
        // to detect programming errors if acquire is called twice without release
        if (locked) {
          console.warn('Mutex already acquired');
        }
        locked = true;
      },
      release: function() {
        if (!locked) {
          console.warn('Mutex not acquired, cannot release');
        }
        locked = false;
      }
    };
  }
  
  /**
   * Create a new endpoint handler for a specific S3-compatible service
   * @param name - Unique name for this endpoint
   * @param endpointUrl - The S3 endpoint URL
   * @param accessKey - Access key for the S3 service
   * @param secretKey - Secret key for the S3 service
   * @param maxConcurrent - Maximum concurrent requests to this endpoint
   * @param circuitBreakerThreshold - Number of failures before the circuit breaker opens
   * @param retries - Number of retry attempts for failed requests
   * @returns The created endpoint handler
   */
  addEndpoint(
    name: string,
    endpointUrl: string,
    accessKey: string | null,
    secretKey: string | null,
    maxConcurrent: number = 5,
    circuitBreakerThreshold: number = 3,
    retries: number = 2
  ): S3EndpointHandler {
    if (!endpointUrl) {
      throw new Error('Endpoint URL is required');
    }
    
    if (!accessKey || !secretKey) {
      throw new Error('Access key and secret key are required');
    }
    
    const handler = new S3EndpointHandler({
      endpoint_url: endpointUrl,
      access_key: accessKey,
      secret_key: secretKey,
      max_concurrent: maxConcurrent,
      circuit_breaker_threshold: circuitBreakerThreshold,
      retries: retries,
      initial_retry_delay: this.initialRetryDelay,
      backoff_factor: this.backoffFactor,
      timeout: this.defaultTimeout
    });
    
    this.endpointsLock.acquire();
    try {
      this.endpoints.set(name, handler);
      this.lastUsed.set(name, 0);
      this.requestsPerEndpoint.set(name, 0);
    } finally {
      this.endpointsLock.release();
    }
    
    return handler;
  }
  
  /**
   * Get an endpoint by name or using a selection strategy
   * @param name - Optional endpoint name
   * @param strategy - Strategy for selecting an endpoint ('round-robin' or 'least-loaded')
   * @returns The selected endpoint handler
   */
  getEndpoint(name?: string, strategy?: 'round-robin' | 'least-loaded'): S3EndpointHandler {
    this.endpointsLock.acquire();
    try {
      if (this.endpoints.size === 0) {
        throw new Error('No S3 endpoints have been added');
      }
      
      // Return specific endpoint if requested
      if (name && this.endpoints.has(name)) {
        this.lastUsed.set(name, Date.now());
        this.requestsPerEndpoint.set(name, (this.requestsPerEndpoint.get(name) || 0) + 1);
        return this.endpoints.get(name)!;
      }
      
      // Use provided strategy or default
      const useStrategy = strategy || this.endpointSelectionStrategy;
      
      // Apply selection strategy
      let selected: string;
      
      if (useStrategy === 'round-robin') {
        // Choose least recently used endpoint
        let minTime = Infinity;
        let minEndpoint = '';
        
        for (const [endpoint, time] of this.lastUsed.entries()) {
          if (time < minTime) {
            minTime = time;
            minEndpoint = endpoint;
          }
        }
        
        selected = minEndpoint;
      } else if (useStrategy === 'least-loaded') {
        // Choose endpoint with fewest requests
        let minRequests = Infinity;
        let minEndpoint = '';
        
        for (const [endpoint, count] of this.requestsPerEndpoint.entries()) {
          if (count < minRequests) {
            minRequests = count;
            minEndpoint = endpoint;
          }
        }
        
        selected = minEndpoint;
      } else {
        // Default to first endpoint
        selected = this.endpoints.keys().next().value;
      }
      
      this.lastUsed.set(selected, Date.now());
      this.requestsPerEndpoint.set(selected, (this.requestsPerEndpoint.get(selected) || 0) + 1);
      return this.endpoints.get(selected)!;
    } finally {
      this.endpointsLock.release();
    }
  }
  
  /**
   * Test if an S3 endpoint is accessible
   * @param endpointUrl - The S3 endpoint URL to test
   * @param accessKey - Access key for the S3 service
   * @param secretKey - Secret key for the S3 service
   * @returns True if the endpoint is accessible, false otherwise
   */
  async testS3Endpoint(
    endpointUrl: string,
    accessKey?: string,
    secretKey?: string
  ): Promise<boolean> {
    const useAccessKey = accessKey || this.getDefaultS3AccessKey();
    const useSecretKey = secretKey || this.getDefaultS3SecretKey();
    
    if (!useAccessKey || !useSecretKey) {
      throw new Error('Access key and secret key are required');
    }
    
    // Just check if we can connect to the endpoint by making a simple request
    try {
      const response = await fetch(`${endpointUrl}?location`, {
        method: 'GET',
        headers: {
          'Authorization': `AWS ${useAccessKey}:${this.calculateS3Signature(useSecretKey, 'GET', `${endpointUrl}?location`, '')}`
        }
      });
      
      return response.status === 200;
    } catch (error) {
      console.error(`S3 endpoint test failed: ${error}`);
      return false;
    }
  }
  
  /**
   * Calculate AWS S3 signature (simplified for demo purposes)
   * Note: In a real implementation, use a library like aws-sdk or aws4
   */
  private calculateS3Signature(secretKey: string, method: string, url: string, payload: string): string {
    // This is a simplified placeholder - in real code, use a proper S3 signature calculation
    // In practice, use aws-sdk or aws4 library for this
    const stringToSign = `${method}\n\n\n${new Date().toUTCString()}\n${url}`;
    
    // In a browser environment, you'd need to use Web Crypto API or a library
    // This is just a placeholder that doesn't actually calculate a valid signature
    return 'SignatureWouldBeCalculatedHere';
  }
  
  /**
   * Upload a file to an S3 bucket
   * @param filePath - Path to the file to upload
   * @param bucket - S3 bucket name
   * @param key - S3 object key
   * @param options - Additional options for the upload
   * @returns Promise resolving to the upload result
   */
  async uploadFile(
    filePath: string,
    bucket: string,
    key: string,
    options: Partial<S3RequestOptions> = {}
  ): Promise<any> {
    const requestId = this.generateRequestId();
    
    return new Promise((resolve, reject) => {
      this.queueRequest({
        id: requestId,
        operation: 'upload_file',
        options: {
          bucket,
          key,
          file_path: filePath,
          ...options
        },
        priority: options.priority || 'NORMAL',
        resolve,
        reject,
        timestamp: Date.now(),
        retryCount: 0
      });
    });
  }
  
  /**
   * Download a file from an S3 bucket
   * @param bucket - S3 bucket name
   * @param key - S3 object key
   * @param filePath - Path where the file should be saved
   * @param options - Additional options for the download
   * @returns Promise resolving to the download result
   */
  async downloadFile(
    bucket: string,
    key: string,
    filePath: string,
    options: Partial<S3RequestOptions> = {}
  ): Promise<any> {
    const requestId = this.generateRequestId();
    
    return new Promise((resolve, reject) => {
      this.queueRequest({
        id: requestId,
        operation: 'download_file',
        options: {
          bucket,
          key,
          file_path: filePath,
          ...options
        },
        priority: options.priority || 'NORMAL',
        resolve,
        reject,
        timestamp: Date.now(),
        retryCount: 0
      });
    });
  }
  
  /**
   * List objects in an S3 bucket
   * @param bucket - S3 bucket name
   * @param prefix - Optional prefix to filter objects
   * @param options - Additional options for listing objects
   * @returns Promise resolving to the list of objects
   */
  async listObjects(
    bucket: string,
    prefix?: string,
    options: Partial<S3RequestOptions> = {}
  ): Promise<any> {
    const requestId = this.generateRequestId();
    
    return new Promise((resolve, reject) => {
      this.queueRequest({
        id: requestId,
        operation: 'list_objects',
        options: {
          bucket,
          prefix,
          ...options
        },
        priority: options.priority || 'NORMAL',
        resolve,
        reject,
        timestamp: Date.now(),
        retryCount: 0
      });
    });
  }
  
  /**
   * Delete an object from an S3 bucket
   * @param bucket - S3 bucket name
   * @param key - S3 object key
   * @param options - Additional options for the delete operation
   * @returns Promise resolving to the delete result
   */
  async deleteObject(
    bucket: string,
    key: string,
    options: Partial<S3RequestOptions> = {}
  ): Promise<any> {
    const requestId = this.generateRequestId();
    
    return new Promise((resolve, reject) => {
      this.queueRequest({
        id: requestId,
        operation: 'delete_object',
        options: {
          bucket,
          key,
          ...options
        },
        priority: options.priority || 'NORMAL',
        resolve,
        reject,
        timestamp: Date.now(),
        retryCount: 0
      });
    });
  }
  
  /**
   * Queue a request for processing
   * @param request - The request to queue
   */
  private queueRequest(request: any): void {
    // Check if queue is full
    if (this.requestQueue.length >= this.queueSize) {
      request.reject(new Error(`Request queue is full (max size: ${this.queueSize})`));
      return;
    }
    
    // Add request to queue based on priority
    if (request.priority === 'HIGH') {
      // High priority requests go to the front of the queue
      this.requestQueue.unshift(request);
    } else if (request.priority === 'LOW') {
      // Low priority requests go to the back of the queue
      this.requestQueue.push(request);
    } else {
      // Normal priority requests go after existing high priority requests
      const highPriorityCount = this.requestQueue.filter(r => r.priority === 'HIGH').length;
      this.requestQueue.splice(highPriorityCount, 0, request);
    }
    
    // Start processing the queue if not already processing
    if (!this.queueProcessing) {
      this.processQueue();
    }
  }
  
  /**
   * Process the request queue
   */
  private async processQueue(): Promise<void> {
    if (this.queueProcessing) {
      return; // Already processing
    }
    
    this.queueProcessing = true;
    
    while (this.requestQueue.length > 0 && this.activeRequests < this.maxConcurrentRequests) {
      const request = this.requestQueue.shift()!;
      this.activeRequests++;
      
      // Process the request
      try {
        // Select an endpoint using a strategy or specific endpoint
        const endpoint = this.getEndpoint();
        const result = await endpoint.processRequest(request.operation, request.options);
        request.resolve(result);
      } catch (error) {
        console.error(`Error processing request ${request.id}: ${error}`);
        
        // Retry if we haven't reached the maximum retry count
        if (request.retryCount < this.maxRetries) {
          request.retryCount++;
          const delay = this.initialRetryDelay * Math.pow(this.backoffFactor, request.retryCount - 1);
          
          console.log(`Retrying request ${request.id} after ${delay}ms (attempt ${request.retryCount}/${this.maxRetries})`);
          
          setTimeout(() => {
            this.queueRequest(request);
          }, delay);
        } else {
          // Maximum retries reached, reject the request
          request.reject(error);
        }
      } finally {
        this.activeRequests--;
      }
    }
    
    this.queueProcessing = false;
    
    // If there are still requests in the queue and we have available capacity, continue processing
    if (this.requestQueue.length > 0 && this.activeRequests < this.maxConcurrentRequests) {
      this.processQueue();
    }
  }
  
  /**
   * Generate a unique request ID
   * @returns A unique request ID
   */
  private generateRequestId(): string {
    return `s3-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  /**
   * Create an endpoint handler for making S3 API requests
   * @param endpoint - The endpoint URL
   * @returns A function that can be used to make requests to the endpoint
   */
  createEndpointHandler(endpoint: string): (operation: string, options: any) => Promise<any> {
    return async (operation: string, options: any): Promise<any> => {
      // This is a placeholder implementation
      // In a real implementation, use a library like aws-sdk or minio-js
      console.log(`Endpoint handler for ${endpoint} called with operation ${operation}`);
      
      switch (operation) {
        case 'upload_file':
          return this.mockUploadFile(options);
        case 'download_file':
          return this.mockDownloadFile(options);
        case 'list_objects':
          return this.mockListObjects(options);
        case 'delete_object':
          return this.mockDeleteObject(options);
        default:
          throw new Error(`Unsupported operation: ${operation}`);
      }
    };
  }
  
  /**
   * Test the API endpoint
   * Not applicable for S3 Kit, use testS3Endpoint instead
   */
  async testEndpoint(): Promise<boolean> {
    const endpoint = this.getDefaultS3Endpoint();
    if (!endpoint) {
      return false;
    }
    
    return this.testS3Endpoint(endpoint);
  }
  
  /**
   * Make a POST request (not applicable for S3 Kit)
   */
  async makePostRequest(url: string, data: any, options?: BaseRequestOptions): Promise<any> {
    throw new Error('Method not applicable for S3 Kit');
  }
  
  /**
   * Make a streaming request (not applicable for S3 Kit)
   */
  async makeStreamRequest(url: string, data: any, options?: BaseRequestOptions): Promise<any> {
    throw new Error('Method not applicable for S3 Kit');
  }
  
  /**
   * Generate a chat completion (not applicable for S3 Kit)
   */
  async chat(model: string, messages: Array<any>, options?: any): Promise<any> {
    throw new Error('Method not applicable for S3 Kit');
  }
  
  /**
   * Generate a streaming chat completion (not applicable for S3 Kit)
   */
  async *streamChat(model: string, messages: Array<any>, options?: any): AsyncGenerator<any> {
    throw new Error('Method not applicable for S3 Kit');
  }
  
  // Mock implementations for testing - in a real implementation, use aws-sdk or minio-js
  
  private mockUploadFile(options: S3RequestOptions): Promise<any> {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        console.log(`Mock upload file ${options.file_path} to ${options.bucket}/${options.key}`);
        resolve({
          ETag: '"mockETag"',
          Location: `https://${options.bucket}.s3.amazonaws.com/${options.key}`,
          Bucket: options.bucket,
          Key: options.key
        });
      }, 50);
    });
  }
  
  private mockDownloadFile(options: S3RequestOptions): Promise<any> {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        console.log(`Mock download file from ${options.bucket}/${options.key} to ${options.file_path}`);
        resolve({
          Bucket: options.bucket,
          Key: options.key,
          Body: 'Mock file content'
        });
      }, 50);
    });
  }
  
  private mockListObjects(options: S3RequestOptions): Promise<any> {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        console.log(`Mock list objects in ${options.bucket} with prefix ${options.prefix || ''}`);
        resolve({
          Contents: [
            { Key: 'file1.txt', Size: 100, LastModified: new Date().toISOString() },
            { Key: 'file2.txt', Size: 200, LastModified: new Date().toISOString() }
          ],
          Name: options.bucket,
          Prefix: options.prefix || '',
          MaxKeys: options.max_keys || 1000,
          IsTruncated: false
        });
      }, 50);
    });
  }
  
  private mockDeleteObject(options: S3RequestOptions): Promise<any> {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        console.log(`Mock delete object ${options.key} from ${options.bucket}`);
        resolve({
          DeleteMarker: true,
          VersionId: 'mockVersionId'
        });
      }, 50);
    });
  }
}

/**
 * Handler for a specific S3 endpoint with its own circuit breaker and configuration
 */
class S3EndpointHandler {
  // Endpoint configuration
  readonly metadata: S3EndpointHandlerMetadata;
  
  // Circuit breaker state
  private circuitState: CircuitBreakerState;
  private failureCount: number;
  private lastStateChange: number;
  
  constructor(config: S3EndpointConfig) {
    this.metadata = {
      endpoint_url: config.endpoint_url,
      access_key: config.access_key,
      secret_key: config.secret_key,
      max_concurrent: config.max_concurrent || 5,
      circuit_breaker_threshold: config.circuit_breaker_threshold || 3,
      retries: config.retries || 2,
      initial_retry_delay: config.initial_retry_delay || 1000,
      backoff_factor: config.backoff_factor || 2,
      timeout: config.timeout || 30000
    };
    
    // Initialize circuit breaker
    this.circuitState = CircuitBreakerState.CLOSED;
    this.failureCount = 0;
    this.lastStateChange = Date.now();
  }
  
  /**
   * Process a request through this endpoint handler
   * @param operation - The operation to perform
   * @param options - Options for the operation
   * @returns Promise resolving to the operation result
   */
  async processRequest(operation: string, options: any): Promise<any> {
    // Check circuit breaker state
    if (this.circuitState === CircuitBreakerState.OPEN) {
      const timeInOpen = Date.now() - this.lastStateChange;
      
      if (timeInOpen < this.metadata.circuit_breaker_threshold * 1000) {
        throw new Error(`Circuit breaker is open for ${this.metadata.endpoint_url}`);
      }
      
      // Transition to half-open state
      this.circuitState = CircuitBreakerState.HALF_OPEN;
      this.lastStateChange = Date.now();
    }
    
    try {
      // In a real implementation, use aws-sdk or minio-js to make the actual request
      let result: any;
      
      switch (operation) {
        case 'upload_file':
          result = await this.uploadFile(options);
          break;
        case 'download_file':
          result = await this.downloadFile(options);
          break;
        case 'list_objects':
          result = await this.listObjects(options);
          break;
        case 'delete_object':
          result = await this.deleteObject(options);
          break;
        default:
          throw new Error(`Unsupported operation: ${operation}`);
      }
      
      // Reset failure count on success
      this.failureCount = 0;
      
      // Transition from half-open to closed on success
      if (this.circuitState === CircuitBreakerState.HALF_OPEN) {
        this.circuitState = CircuitBreakerState.CLOSED;
        this.lastStateChange = Date.now();
      }
      
      return result;
    } catch (error) {
      // Increment failure count
      this.failureCount++;
      
      // If in half-open state or failure count exceeds threshold, open the circuit
      if (this.circuitState === CircuitBreakerState.HALF_OPEN || 
          this.failureCount >= this.metadata.circuit_breaker_threshold) {
        this.circuitState = CircuitBreakerState.OPEN;
        this.lastStateChange = Date.now();
      }
      
      throw error;
    }
  }
  
  /**
   * Upload a file to S3
   * @param options - Upload options
   * @returns Promise resolving to the upload result
   */
  private async uploadFile(options: S3RequestOptions): Promise<any> {
    // Mock implementation - in a real application, use aws-sdk or minio-js
    return {
      ETag: '"mockETag"',
      Location: `https://${options.bucket}.s3.amazonaws.com/${options.key}`,
      Bucket: options.bucket,
      Key: options.key
    };
  }
  
  /**
   * Download a file from S3
   * @param options - Download options
   * @returns Promise resolving to the download result
   */
  private async downloadFile(options: S3RequestOptions): Promise<any> {
    // Mock implementation - in a real application, use aws-sdk or minio-js
    return {
      Bucket: options.bucket,
      Key: options.key,
      Body: 'Mock file content'
    };
  }
  
  /**
   * List objects in an S3 bucket
   * @param options - List options
   * @returns Promise resolving to the list of objects
   */
  private async listObjects(options: S3RequestOptions): Promise<any> {
    // Mock implementation - in a real application, use aws-sdk or minio-js
    return {
      Contents: [
        { Key: 'file1.txt', Size: 100, LastModified: new Date().toISOString() },
        { Key: 'file2.txt', Size: 200, LastModified: new Date().toISOString() }
      ],
      Name: options.bucket,
      Prefix: options.prefix || '',
      MaxKeys: options.max_keys || 1000,
      IsTruncated: false
    };
  }
  
  /**
   * Delete an object from S3
   * @param options - Delete options
   * @returns Promise resolving to the delete result
   */
  private async deleteObject(options: S3RequestOptions): Promise<any> {
    // Mock implementation - in a real application, use aws-sdk or minio-js
    return {
      DeleteMarker: true,
      VersionId: 'mockVersionId'
    };
  }
}