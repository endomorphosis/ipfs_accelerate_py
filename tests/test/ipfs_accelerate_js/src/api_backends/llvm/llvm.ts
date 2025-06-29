import { BaseApiBackend } from '../base';
import { ApiMetadata, BaseRequestOptions, ChatMessage } from '../types';
import { 
  LlvmOptions, 
  ModelInfo, 
  InferenceResponse, 
  ListModelsResponse,
  InferenceOptions
} from './types';

/**
 * LLVM API Backend for working with LLVM-based inference servers
 */
export class LLVM extends BaseApiBackend {
  private apiKey: string | null;
  private baseUrl: string;
  private requestCount: number;
  
  // Request management properties
  protected maxRetries: number;
  protected retryDelay: number;
  protected maxConcurrentRequests: number;
  protected queueSize: number;
  protected requestQueue: Array<{
    id: string;
    operation: string;
    params: any;
    priority: 'HIGH' | 'NORMAL' | 'LOW';
    resolve: (value: any) => void;
    reject: (reason: any) => void;
    timestamp: number;
    retryCount: number;
  }>;
  protected queueProcessing: boolean;
  protected activeRequests: number;

  /**
   * Create a new LLVM API Backend
   * @param options - Options for the LLVM client
   * @param metadata - API metadata including API key
   */
  constructor(options: LlvmOptions = {}, metadata: ApiMetadata = {}) {
    super(options, metadata);
    
    // Initialize API key from metadata or environment
    this.apiKey = this.getApiKey();
    
    // Initialize base URL
    this.baseUrl = options.base_url || process.env.LLVM_BASE_URL || 'http://localhost:8000';
    
    // Initialize request counter (for debugging/metrics)
    this.requestCount = 0;
    
    // Initialize request management
    this.maxRetries = options.max_retries || 3;
    this.retryDelay = options.retry_delay || 1000;
    this.maxConcurrentRequests = options.max_concurrent_requests || 10;
    this.queueSize = options.queue_size || 100;
    this.requestQueue = [];
    this.queueProcessing = false;
    this.activeRequests = 0;
  }

  /**
   * Get the API key from metadata or environment
   * @returns The API key, or null if not found
   */
  getApiKey(): string | null {
    return this.metadata?.llvm_api_key || 
           process.env.LLVM_API_KEY || 
           null;
  }

  /**
   * Get the default model for this API
   * @returns The default model ID (or empty string if none specified)
   */
  getDefaultModel(): string {
    return this.metadata?.llvm_default_model || 
           process.env.LLVM_DEFAULT_MODEL || 
           '';
  }

  /**
   * Set the API key
   * @param apiKey - The new API key to use
   */
  setApiKey(apiKey: string): void {
    this.apiKey = apiKey;
  }

  /**
   * Check if a model is compatible with this API backend
   * @param model - Model name/ID to check
   * @returns True if the model is compatible, false otherwise
   */
  isCompatibleModel(model: string): boolean {
    // LLVM can work with a variety of models
    // This is a simple check based on naming pattern
    return model.startsWith('llvm-') || 
           model.startsWith('llvm_') || 
           model.endsWith('-llvm') || 
           model.endsWith('_llvm');
  }

  /**
   * Test the endpoint connection
   * @returns Promise resolving to true if connection successful, false otherwise
   */
  async testEndpoint(): Promise<boolean> {
    try {
      await this.listModels();
      return true;
    } catch (error) {
      console.error('LLVM endpoint test failed:', error);
      return false;
    }
  }

  /**
   * Make a POST request to the LLVM API
   * @param url - API endpoint URL
   * @param data - Request payload
   * @param options - Additional request options
   * @returns Promise resolving to the API response
   */
  async makePostRequest(url: string, data: any, options?: BaseRequestOptions): Promise<any> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(data),
        signal: options?.abortSignal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`LLVM API error (${response.status}): ${errorText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('LLVM API request failed:', error);
      throw error;
    }
  }

  /**
   * Make a streaming request (not supported in LLVM API)
   */
  async makeStreamRequest(url: string, data: any, options?: BaseRequestOptions): Promise<any> {
    throw new Error('Streaming is not supported by the LLVM API');
  }

  /**
   * List all available models
   * @returns Promise resolving to a list of available models
   */
  async listModels(): Promise<ListModelsResponse> {
    const requestId = this.generateRequestId();
    
    return new Promise((resolve, reject) => {
      this.queueRequest({
        id: requestId,
        operation: 'list_models',
        params: {},
        priority: 'NORMAL',
        resolve,
        reject,
        timestamp: Date.now(),
        retryCount: 0
      });
    });
  }

  /**
   * Get information about a specific model
   * @param modelId - ID of the model to get information about
   * @returns Promise resolving to model information
   */
  async getModelInfo(modelId: string): Promise<ModelInfo> {
    const requestId = this.generateRequestId();
    
    return new Promise((resolve, reject) => {
      this.queueRequest({
        id: requestId,
        operation: 'get_model_info',
        params: { modelId },
        priority: 'NORMAL',
        resolve,
        reject,
        timestamp: Date.now(),
        retryCount: 0
      });
    });
  }

  /**
   * Run inference with a model
   * @param modelId - ID of the model to use
   * @param inputs - Input data for the model
   * @param options - Additional inference options
   * @returns Promise resolving to inference results
   */
  async runInference(
    modelId: string,
    inputs: string | Record<string, any>,
    options: InferenceOptions = {}
  ): Promise<InferenceResponse> {
    const requestId = this.generateRequestId();
    
    return new Promise((resolve, reject) => {
      this.queueRequest({
        id: requestId,
        operation: 'run_inference',
        params: { modelId, inputs, options },
        priority: 'NORMAL',
        resolve,
        reject,
        timestamp: Date.now(),
        retryCount: 0
      });
    });
  }

  /**
   * Execute LLVM API operation with retry mechanism
   * @param operation - API operation name
   * @param params - Operation parameters
   * @returns Promise resolving to the operation result
   */
  private async executeOperation(operation: string, params: any): Promise<any> {
    let result: any;
    
    try {
      switch (operation) {
        case 'list_models':
          result = await this.executeListModels();
          break;
        case 'get_model_info':
          result = await this.executeGetModelInfo(params.modelId);
          break;
        case 'run_inference':
          result = await this.executeRunInference(
            params.modelId,
            params.inputs,
            params.options
          );
          break;
        default:
          throw new Error(`Unknown operation: ${operation}`);
      }
      
      this.requestCount++;
      return result;
    } catch (error) {
      console.error(`LLVM operation ${operation} failed:`, error);
      throw error;
    }
  }

  /**
   * Execute the list_models operation
   * @returns Promise resolving to list of models
   */
  private async executeListModels(): Promise<ListModelsResponse> {
    const url = `${this.baseUrl}/models`;
    const response = await fetch(url, {
      method: 'GET',
      headers: this.getHeaders(),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to list models: ${response.statusText}`);
    }
    
    return await response.json();
  }

  /**
   * Execute the get_model_info operation
   * @param modelId - ID of the model
   * @returns Promise resolving to model information
   */
  private async executeGetModelInfo(modelId: string): Promise<ModelInfo> {
    const url = `${this.baseUrl}/models/${encodeURIComponent(modelId)}`;
    const response = await fetch(url, {
      method: 'GET',
      headers: this.getHeaders(),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to get model info: ${response.statusText}`);
    }
    
    return await response.json();
  }

  /**
   * Execute the run_inference operation
   * @param modelId - ID of the model
   * @param inputs - Input data
   * @param options - Inference options
   * @returns Promise resolving to inference results
   */
  private async executeRunInference(
    modelId: string,
    inputs: string | Record<string, any>,
    options: InferenceOptions = {}
  ): Promise<InferenceResponse> {
    const url = `${this.baseUrl}/models/${encodeURIComponent(modelId)}/inference`;
    const data = {
      inputs,
      ...options
    };
    
    const response = await this.makePostRequest(url, data);
    return response;
  }

  /**
   * Get request headers with authentication
   * @returns Headers object with authentication
   */
  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };
    
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
    
    return headers;
  }

  /**
   * Queue a request for processing
   * @param request - Request to queue
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
        // Execute operation with retries
        let result = null;
        let success = false;
        let lastError = null;
        
        for (let attempt = 0; attempt <= request.retryCount; attempt++) {
          try {
            result = await this.executeOperation(request.operation, request.params);
            success = true;
            break;
          } catch (error) {
            console.error(`Attempt ${attempt + 1}/${this.maxRetries + 1} failed:`, error);
            lastError = error;
            
            if (attempt < this.maxRetries) {
              // Wait before retrying
              const delay = this.retryDelay * Math.pow(2, attempt);
              await new Promise(resolve => setTimeout(resolve, delay));
            }
          }
        }
        
        if (success) {
          request.resolve(result);
        } else {
          request.reject(lastError || new Error('All retry attempts failed'));
        }
      } catch (error) {
        console.error(`Error processing request ${request.id}:`, error);
        request.reject(error);
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
   * @returns Unique request ID
   */
  private generateRequestId(): string {
    return `llvm-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Chat completion - not directly supported by LLVM API
   * This implementation converts chat messages to text and uses runInference
   */
  async chat(model: string, messages: ChatMessage[], options?: any): Promise<any> {
    // Convert chat messages to a single string for LLVM
    const formattedPrompt = this.formatChatMessages(messages);
    
    // Run inference with the formatted prompt
    const response = await this.runInference(model, formattedPrompt, options);
    
    // Format the response as a chat message
    return {
      id: `llvm-${Date.now()}`,
      model,
      object: 'chat.completion',
      created: Date.now(),
      content: response.outputs,
      role: 'assistant'
    };
  }

  /**
   * Format chat messages as a single string for LLVM
   * @param messages - Array of chat messages
   * @returns Formatted text string
   */
  private formatChatMessages(messages: ChatMessage[]): string {
    return messages.map(msg => {
      if (typeof msg.content === 'string') {
        return `${msg.role}: ${msg.content}`;
      } else {
        // Handle array content (multimodal) - convert to string
        return `${msg.role}: ${JSON.stringify(msg.content)}`;
      }
    }).join('\n');
  }

  /**
   * Stream chat completion - not supported by LLVM API
   */
  async *streamChat(model: string, messages: ChatMessage[], options?: any): AsyncGenerator<any> {
    throw new Error('Streaming is not supported by the LLVM API');
  }
}