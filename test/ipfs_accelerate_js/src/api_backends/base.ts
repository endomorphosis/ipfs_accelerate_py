/**
 * Base class for all API backends
 */

import { 
  ApiResources, 
  ApiMetadata, 
  ApiEndpoint, 
  ApiEndpointStats,
  Message,
  ChatCompletionResponse,
  StreamChunk,
  ApiRequestOptions,
  ApiError,
  RequestQueueItem
} from './types';

export abstract class BaseApiBackend {
  // Common properties all backends will have
  protected resources: ApiResources;
  protected metadata: ApiMetadata;
  protected apiKey: string;
  protected model: string;
  protected endpoints: Record<string, ApiEndpoint> = {};
  
  // Request tracking and queue management
  protected queueEnabled: boolean = true;
  protected requestQueue: RequestQueueItem[] = [];
  protected circuitState: 'OPEN' | 'CLOSED' | 'HALF-OPEN' = 'CLOSED';
  protected currentRequests: number = 0;
  protected maxConcurrentRequests: number = 5;
  protected queueSize: number = 100;
  
  // Retry and backoff settings
  protected maxRetries: number = 3;
  protected initialRetryDelay: number = 1000; // milliseconds
  protected backoffFactor: number = 2;
  protected timeout: number = 30000; // milliseconds
  
  // Usage tracking
  protected requestTracking: boolean = true;
  protected recentRequests: Record<string, {
    timestamp: number;
    success: boolean;
    error?: string;
  }> = {};

  constructor(resources: ApiResources = {}, metadata: ApiMetadata = {}) {
    this.resources = resources;
    this.metadata = metadata;
    this.apiKey = this.getApiKey(metadata);
    this.model = metadata.model || this.getDefaultModel();
    
    // Initialize with metadata values if provided
    if (metadata.maxRetries !== undefined) {
      this.maxRetries = metadata.maxRetries;
    }
    if (metadata.initialRetryDelay !== undefined) {
      this.initialRetryDelay = metadata.initialRetryDelay;
    }
    if (metadata.backoffFactor !== undefined) {
      this.backoffFactor = metadata.backoffFactor;
    }
    if (metadata.timeout !== undefined) {
      this.timeout = metadata.timeout;
    }
    if (metadata.maxConcurrentRequests !== undefined) {
      this.maxConcurrentRequests = metadata.maxConcurrentRequests;
    }
    if (metadata.queueSize !== undefined) {
      this.queueSize = metadata.queueSize;
    }
  }

  /**
   * Get the API key from metadata or environment
   * Each implementation should override this with its specific API key logic
   */
  protected abstract getApiKey(metadata: ApiMetadata): string;

  /**
   * Get default model for this API
   */
  protected abstract getDefaultModel(): string;

  /**
   * Create an endpoint handler for this API
   */
  abstract createEndpointHandler(): (data: any) => Promise<any>;

  /**
   * Test the API endpoint
   */
  abstract testEndpoint(): Promise<boolean>;

  /**
   * Make a POST request to the API
   */
  abstract makePostRequest(data: any, apiKey?: string, options?: ApiRequestOptions): Promise<any>;

  /**
   * Make a streaming request to the API
   */
  abstract makeStreamRequest(data: any, options?: ApiRequestOptions): AsyncGenerator<StreamChunk>;

  /**
   * Generate a chat completion
   */
  abstract chat(messages: Message[], options?: ApiRequestOptions): Promise<ChatCompletionResponse>;

  /**
   * Generate a streaming chat completion
   */
  abstract streamChat(messages: Message[], options?: ApiRequestOptions): AsyncGenerator<StreamChunk>;

  /**
   * Create a new API endpoint for multiplexing
   */
  createEndpoint(endpoint: Partial<ApiEndpoint>): string {
    const endpointId = `endpoint_${Object.keys(this.endpoints).length}`;
    this.endpoints[endpointId] = {
      id: endpointId,
      apiKey: endpoint.apiKey || this.apiKey,
      model: endpoint.model || this.model,
      maxConcurrentRequests: endpoint.maxConcurrentRequests || this.maxConcurrentRequests,
      queueSize: endpoint.queueSize || this.queueSize,
      maxRetries: endpoint.maxRetries || this.maxRetries,
      initialRetryDelay: endpoint.initialRetryDelay || this.initialRetryDelay,
      backoffFactor: endpoint.backoffFactor || this.backoffFactor,
      timeout: endpoint.timeout || this.timeout,
      ...endpoint
    };
    return endpointId;
  }

  /**
   * Get statistics for an endpoint
   */
  getStats(endpointId: string): ApiEndpointStats {
    if (!this.endpoints[endpointId]) {
      throw new Error(`Endpoint ${endpointId} not found`);
    }
    
    // Count successful and failed requests for this endpoint
    let requests = 0;
    let success = 0;
    let errors = 0;
    
    for (const requestId in this.recentRequests) {
      if (requestId.startsWith(endpointId)) {
        requests++;
        if (this.recentRequests[requestId].success) {
          success++;
        } else {
          errors++;
        }
      }
    }
    
    return {
      requests,
      success,
      errors
    };
  }

  /**
   * Make a request using a specific endpoint
   */
  async makeRequestWithEndpoint(endpointId: string, data: any, options?: ApiRequestOptions): Promise<any> {
    if (!this.endpoints[endpointId]) {
      throw new Error(`Endpoint ${endpointId} not found`);
    }
    
    const endpoint = this.endpoints[endpointId];
    const requestId = options?.requestId || `${endpointId}_${Date.now()}`;
    
    try {
      const response = await this.makePostRequest(
        data, 
        endpoint.apiKey,
        { 
          ...options,
          requestId,
          timeout: options?.timeout || endpoint.timeout
        }
      );
      
      this.trackRequestResult(true, requestId);
      return response;
    } catch (error) {
      this.trackRequestResult(false, requestId, error as Error);
      throw error;
    }
  }

  /**
   * Check if a model is compatible with this API
   */
  isCompatibleModel(model: string): boolean {
    return false; // Base implementation, override in derived classes
  }

  /**
   * Track request result for monitoring
   */
  protected trackRequestResult(success: boolean, requestId?: string, error?: Error): void {
    if (!this.requestTracking) return;
    
    const id = requestId || `anon_${Date.now()}`;
    this.recentRequests[id] = {
      timestamp: Date.now(),
      success,
      error: error?.message
    };
    
    // Clean up old requests (older than 1 hour)
    const now = Date.now();
    for (const id in this.recentRequests) {
      if (now - this.recentRequests[id].timestamp > 3600000) {
        delete this.recentRequests[id];
      }
    }
  }

  /**
   * Process the request queue
   */
  protected async processQueue(): Promise<void> {
    if (!this.queueEnabled || this.requestQueue.length === 0 || this.currentRequests >= this.maxConcurrentRequests) {
      return;
    }
    
    if (this.circuitState === 'OPEN') {
      // Circuit is open, don't process any requests
      return;
    }
    
    // Process the oldest request in the queue
    const nextRequest = this.requestQueue.shift();
    if (!nextRequest) return;
    
    this.currentRequests++;
    
    try {
      const response = await this.makePostRequest(
        nextRequest.data,
        nextRequest.apiKey,
        nextRequest.options
      );
      
      nextRequest.resolve(response);
      this.trackRequestResult(true, nextRequest.requestId);
      
      // If circuit was half-open and this succeeded, close the circuit
      if (this.circuitState === 'HALF-OPEN') {
        this.circuitState = 'CLOSED';
      }
    } catch (error) {
      nextRequest.reject(error);
      this.trackRequestResult(false, nextRequest.requestId, error as Error);
      
      // Open the circuit if we're seeing multiple failures
      const errorRate = this.calculateErrorRate();
      if (errorRate > 0.5 && this.circuitState === 'CLOSED') {
        this.circuitState = 'OPEN';
        
        // Set a timeout to try again after a delay
        setTimeout(() => {
          this.circuitState = 'HALF-OPEN';
        }, this.initialRetryDelay * 1000);
      }
    } finally {
      this.currentRequests--;
      
      // Trigger processing the next item in the queue
      setTimeout(() => this.processQueue(), 0);
    }
  }
  
  /**
   * Calculate current error rate for circuit breaker
   */
  private calculateErrorRate(): number {
    const recentTimeWindow = Date.now() - 60000; // Last minute
    let totalRequests = 0;
    let failedRequests = 0;
    
    for (const id in this.recentRequests) {
      if (this.recentRequests[id].timestamp > recentTimeWindow) {
        totalRequests++;
        if (!this.recentRequests[id].success) {
          failedRequests++;
        }
      }
    }
    
    return totalRequests > 0 ? failedRequests / totalRequests : 0;
  }
  
  /**
   * Create an error object with appropriate properties based on the type
   */
  protected createApiError(message: string, statusCode?: number, type?: string): ApiError {
    const error = new Error(message) as ApiError;
    error.statusCode = statusCode;
    error.type = type;
    
    // Set appropriate flags based on status code and type
    error.isAuthError = statusCode === 401 || statusCode === 403;
    error.isRateLimitError = statusCode === 429;
    error.isTransientError = statusCode === 500 || statusCode === 502 || statusCode === 503 || statusCode === 504;
    
    return error;
  }
  
  /**
   * Retryable request - performs retries with exponential backoff
   */
  protected async retryableRequest<T>(
    requestFn: () => Promise<T>, 
    maxRetries: number = this.maxRetries,
    initialDelay: number = this.initialRetryDelay,
    backoffFactor: number = this.backoffFactor
  ): Promise<T> {
    let lastError: Error | null = null;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await requestFn();
      } catch (error) {
        lastError = error as Error;
        
        const apiError = error as ApiError;
        
        // Don't retry auth errors
        if (apiError.isAuthError) {
          break;
        }
        
        // If this was the last attempt, don't delay
        if (attempt === maxRetries) {
          break;
        }
        
        // Calculate delay, with custom retry-after if provided
        let delay = apiError.retryAfter 
          ? apiError.retryAfter * 1000 
          : initialDelay * Math.pow(backoffFactor, attempt);
          
        // Add some jitter to prevent thundering herd problems
        delay = delay * (0.9 + Math.random() * 0.2);
        
        // Wait before retrying
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    // If we got here, we've exhausted our retries
    throw lastError || new Error('Request failed after retries');
  }
}