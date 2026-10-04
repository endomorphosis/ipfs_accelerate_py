import { BaseApiBackend } from '../base';
import { ApiMetadata, ApiRequestOptions, Message, ChatCompletionResponse, StreamChunk, ApiError, ApiEndpoint } from '../types';
import { HfTgiResponse, HfTgiRequest, HfTgiChatOptions, HfTgiEndpoint, HfTgiStats } from './types';

/**
 * Hugging Face Text Generation Inference API
 * 
 * This class provides access to the Hugging Face Text Generation Inference API,
 * supporting both hosted inference endpoints and self-hosted deployments.
 */
export class HfTgi extends BaseApiBackend {
  private baseApiUrl: string = 'https://api-inference.huggingface.co/models';
  private defaultModel: string = 'google/t5-efficient-tiny';
  private useDefaultApiEndpoint: boolean = true;
  
  // Use the base class endpoints property
  // We'll cast to HfTgiEndpoint when accessing our specific properties

  constructor(resources: Record<string, any> = {}, metadata: ApiMetadata = {}) {
    super(resources, metadata);
    
    // Initialize with custom API base if provided
    if (metadata.api_base) {
      this.baseApiUrl = metadata.api_base;
      this.useDefaultApiEndpoint = false;
    }
    
    // Set default model if provided
    if (metadata.model_id) {
      this.defaultModel = metadata.model_id;
    }
    
    // Initialize endpoints with default if provided
    if (metadata.hf_api_key || metadata.hfApiKey) {
      this.createEndpoint({
        id: 'default',
        apiKey: this.getApiKey(metadata),    // Standard property
        model: this.getDefaultModel(),       // Standard property
        api_key: this.getApiKey(metadata),   // HF TGI specific property
        model_id: this.getDefaultModel(),    // HF TGI specific property
        maxConcurrentRequests: this.maxConcurrentRequests,  // Standard property
        queueSize: this.queueSize,           // Standard property
        max_concurrent_requests: this.maxConcurrentRequests, // HF TGI specific property
        queue_size: this.queueSize           // HF TGI specific property
      });
    }
  }

  /**
   * Get API key from metadata or environment variables
   */
  protected getApiKey(metadata: ApiMetadata): string {
    return metadata.hf_api_key || 
           metadata.hfApiKey || 
           (typeof process !== 'undefined' ? process.env.HF_API_KEY || process.env.HUGGINGFACE_API_KEY || '' : '');
  }

  /**
   * Get the default model for this API
   */
  protected getDefaultModel(): string {
    return this.defaultModel;
  }

  /**
   * Check if a model is compatible with this API
   */
  isCompatibleModel(model: string): boolean {
    // HuggingFace models typically have a namespace/model format or are from known families
    return (
      model.includes('/') || // Most HF models have a namespace/model format
      model.startsWith('t5') ||
      model.startsWith('bert') ||
      model.startsWith('gpt') ||
      model.startsWith('llama') ||
      model.startsWith('mistral') ||
      model.startsWith('falcon') ||
      model.startsWith('bloom') ||
      model.startsWith('opt') ||
      model.startsWith('pythia')
    );
  }

  /**
   * Create an endpoint handler that can be used for generating text
   * Implements the abstract method from BaseApiBackend
   */
  createEndpointHandler(): (data: any) => Promise<any> {
    const url = `${this.baseApiUrl}/${this.getDefaultModel()}`;
    const key = this.getApiKey(this.metadata);
    
    return async (data: any): Promise<any> => {
      try {
        const requestId = `handler_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
        
        // Extract request options
        const options = {
          requestId,
          ...(data.options || {})
        };
        
        // Check if this is a streaming request
        if (data.stream) {
          // For streaming, return an async generator using arrow function to preserve 'this'
          const makeGen = async function* (self: HfTgi) {
            // Use the instance's makeStreamRequest method with the standardized signature
            yield* self.makeStreamRequest(data, { 
              ...options,
              endpointUrl: url,
              apiKey: key
            });
          };
          
          return makeGen(this);
        } else {
          // For regular requests, use the standard post request method
          return await this.makePostRequest(data, key, { 
            ...options,
            endpointUrl: url 
          });
        }
      } catch (error) {
        const apiError = error as ApiError;
        throw this.createApiError(
          `HF TGI endpoint error: ${apiError.message}`,
          apiError.statusCode || 500,
          apiError.type || 'endpoint_error'
        );
      }
    };
  }
  
  /**
   * Create an endpoint handler with specific URL and API key
   * This is a HF TGI-specific extension method
   */
  createEndpointHandlerWithParams(endpointUrl?: string, apiKey?: string): (data: any) => Promise<any> {
    const url = endpointUrl || `${this.baseApiUrl}/${this.getDefaultModel()}`;
    const key = apiKey || this.getApiKey(this.metadata);
    
    return async (data: any): Promise<any> => {
      try {
        const requestId = `handler_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
        
        // Extract request options
        const options = {
          requestId,
          ...(data.options || {})
        };
        
        // Check if this is a streaming request
        if (data.stream) {
          // For streaming, return an async generator using parameter to preserve 'this'
          const makeGen = async function* (self: HfTgi) {
            // Use the instance's makeStreamRequest method
            yield* self.makeStreamRequest(data, { 
              ...options,
              endpointUrl: url,
              apiKey: key
            });
          };
          
          return makeGen(this);
        } else {
          // For regular requests, use the standard post request method
          return await this.makePostRequest(data, key, { 
            ...options,
            endpointUrl: url 
          });
        }
      } catch (error) {
        const apiError = error as ApiError;
        throw this.createApiError(
          `HF TGI endpoint error: ${apiError.message}`,
          apiError.statusCode || 500,
          apiError.type || 'endpoint_error'
        );
      }
    };
  }

  /**
   * Create a remote text generation endpoint handler
   */
  createRemoteTextGenerationEndpointHandler(endpointUrl?: string, apiKey?: string): (data: any) => Promise<any> {
    // This is essentially the same as createEndpointHandlerWithParams but named to match Python implementation
    return this.createEndpointHandlerWithParams(endpointUrl, apiKey);
  }

  /**
   * Test the HF TGI endpoint for availability and correct configuration
   */
  async testEndpoint(): Promise<boolean>;
  async testEndpoint(endpointUrl?: string, apiKey?: string, modelName?: string): Promise<boolean>;
  async testEndpoint(endpointUrl?: string, apiKey?: string, modelName?: string): Promise<boolean> {
    try {
      const url = endpointUrl || `${this.baseApiUrl}/${modelName || this.getDefaultModel()}`;
      const key = apiKey || this.getApiKey(this.metadata);
      
      // Simple test prompt
      const testRequest: HfTgiRequest = {
        inputs: "Testing the Hugging Face TGI API. Please respond with a short message.",
        parameters: {
          max_new_tokens: 5,
          do_sample: false
        }
      };
      
      // Make a request with minimal timeout
      await this.makePostRequest(testRequest, key, { 
        timeout: 5000,
        endpointUrl: url
      });
      
      return true;
    } catch (error) {
      console.error(`HF TGI endpoint test failed:`, error);
      return false;
    }
  }

  /**
   * Test a specific TGI endpoint
   */
  async testTgiEndpoint(endpointUrl: string, apiKey: string, modelName: string): Promise<any> {
    return this.testEndpoint(endpointUrl, apiKey, modelName);
  }

  /**
   * Make a POST request to the HF TGI API
   */
  async makePostRequest(data: any, apiKey?: string, options?: ApiRequestOptions): Promise<any> {
    const key = apiKey || this.getApiKey(this.metadata);
    const endpointUrl = options?.endpointUrl || `${this.baseApiUrl}/${options?.model || this.getDefaultModel()}`;
    const requestId = options?.requestId || `req_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    
    // Check if we should use endpoint-specific settings
    const endpointId = options?.endpointId;
    let endpoint: HfTgiEndpoint | undefined;
    
    if (endpointId && this.endpoints[endpointId]) {
      endpoint = this.endpoints[endpointId] as HfTgiEndpoint;
    }
    
    // Process with queue, circuit breaker, and retries
    return this.retryableRequest(async () => {
      // Prepare headers
      const headers: Record<string, string> = {
        'Content-Type': 'application/json'
      };
      
      // Add authorization if we have an API key
      if (key) {
        headers['Authorization'] = `Bearer ${key}`;
      }
      
      // Add request ID for tracking
      if (requestId) {
        headers['X-Request-ID'] = requestId;
      }
      
      // Prepare request body
      const requestBody = JSON.stringify(data);
      
      // Set up timeout
      const timeoutMs = options?.timeout || endpoint?.timeout || this.timeout;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
      
      try {
        // Make the request
        const response = await fetch(endpointUrl, {
          method: 'POST',
          headers,
          body: requestBody,
          signal: controller.signal
        });
        
        // Check for errors
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          
          // Create an appropriate error based on status code
          const error = this.createApiError(
            errorData.error?.message || `HTTP error ${response.status}`,
            response.status,
            errorData.error?.type || 'api_error'
          );
          
          // Check for rate limiting headers
          const retryAfter = response.headers.get('retry-after');
          if (retryAfter && !isNaN(Number(retryAfter))) {
            error.retryAfter = Number(retryAfter);
          }
          
          // Track endpoint stats if we're using an endpoint
          if (endpoint) {
            endpoint.failed_requests++;
            endpoint.last_request_at = Date.now();
          }
          
          // Properly cast the error before throwing
          throw error as ApiError;
        }
        
        // Parse response
        const responseData = await response.json();
        
        // Track endpoint stats if we're using an endpoint
        if (endpoint) {
          endpoint.successful_requests++;
          endpoint.last_request_at = Date.now();
          
          // Update token counts if available in the response
          if (responseData.usage) {
            endpoint.total_tokens += responseData.usage.total_tokens || 0;
            endpoint.input_tokens += responseData.usage.prompt_tokens || 0;
            endpoint.output_tokens += responseData.usage.completion_tokens || 0;
          }
        }
        
        return responseData;
      } catch (error) {
        if ((error as Error).name === 'AbortError') {
          throw this.createApiError(`Request timed out after ${timeoutMs}ms`, 408, 'timeout_error');
        }
        throw error;
      } finally {
        clearTimeout(timeoutId);
      }
    }, 
    options?.maxRetries || endpoint?.max_retries || this.maxRetries,
    endpoint?.initial_retry_delay || this.initialRetryDelay,
    endpoint?.backoff_factor || this.backoffFactor
    );
  }

  /**
   * Make a POST request to the HF TGI API with endpoint-specific handling
   */
  makePostRequestHfTgi(
    endpointUrl: string, 
    data: any, 
    apiKey?: string, 
    requestId?: string, 
    endpointId?: string
  ): Promise<any> {
    return this.makePostRequest(data, apiKey, { 
      endpointUrl,
      requestId,
      endpointId
    });
  }

  /**
   * Make a streaming request to the HF TGI API
   * Implements the abstract method from BaseApiBackend
   */
  async *makeStreamRequest(data: any, options?: ApiRequestOptions): AsyncGenerator<StreamChunk> {
    const key = options?.apiKey || this.getApiKey(this.metadata);
    const modelName = options?.model || this.getDefaultModel();
    const url = options?.endpointUrl || `${this.baseApiUrl}/${modelName}`;
    const requestId = options?.requestId || `stream_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    
    // Ensure stream parameter is set
    const streamData = {
      ...data,
      stream: true
    };
    
    // Prepare headers
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };
    
    // Add authorization if we have an API key
    if (key) {
      headers['Authorization'] = `Bearer ${key}`;
    }
    
    // Add request ID for tracking
    if (requestId) {
      headers['X-Request-ID'] = requestId;
    }
    
    // Set up timeout
    const timeoutMs = options?.timeout || this.timeout;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    
    try {
      // Make the request
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(streamData),
        signal: controller.signal
      });
      
      // Check for errors
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw this.createApiError(
          errorData.error?.message || `HTTP error ${response.status}`,
          response.status,
          errorData.error?.type || 'api_error'
        );
      }
      
      if (!response.body) {
        throw this.createApiError('Response body is null', 500, 'stream_error');
      }
      
      // Process the stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep the last incomplete line in the buffer
        
        for (const line of lines) {
          if (line.trim() === '') continue;
          
          try {
            const parsed = JSON.parse(line);
            
            // Extract content from different possible formats
            const content = parsed.generated_text || parsed.token?.text || '';
            const done = parsed.details?.finish_reason !== undefined;
            
            yield {
              content,
              type: 'delta',
              done,
              requestId,
              model: modelName
            };
          } catch (e) {
            console.warn('Failed to parse stream data:', line);
          }
        }
      }
      
      // Handle any remaining data in the buffer
      if (buffer.trim() !== '') {
        try {
          const parsed = JSON.parse(buffer);
          
          // Extract content from different possible formats
          const content = parsed.generated_text || parsed.token?.text || '';
          const done = parsed.details?.finish_reason !== undefined;
          
          yield {
            content,
            type: 'delta',
            done,
            requestId,
            model: modelName
          };
        } catch (e) {
          console.warn('Failed to parse final stream data:', buffer);
        }
      }
      
      // Final chunk to indicate completion
      yield {
        content: '',
        type: 'delta',
        done: true,
        requestId,
        model: modelName
      };
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        throw this.createApiError(`Stream request timed out after ${timeoutMs}ms`, 408, 'timeout_error');
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }
  
  /**
   * Make a streaming request to the HF TGI API with specific parameters
   * This is a HF TGI-specific extension method
   */
  async *makeStreamRequestWithParams(
    data: any, 
    endpointUrl?: string, 
    apiKey?: string, 
    options?: ApiRequestOptions
  ): AsyncGenerator<StreamChunk> {
    // Create combined options
    const combinedOptions: ApiRequestOptions = {
      ...options,
      endpointUrl: endpointUrl || options?.endpointUrl,
      apiKey: apiKey || options?.apiKey
    };
    
    // Call the standardized method
    yield* this.makeStreamRequest(data, combinedOptions);
  }

  /**
   * Make a streaming request to the HF TGI API with endpoint-specific handling
   */
  makeStreamRequestHfTgi(
    endpointUrl: string, 
    data: any, 
    apiKey?: string, 
    requestId?: string, 
    endpointId?: string
  ): AsyncGenerator<StreamChunk> {
    return this.makeStreamRequest(data, { 
      endpointUrl,
      apiKey,
      requestId,
      endpointId
    });
  }
  
  /**
   * Generate text with HF TGI API
   */
  async generateText(
    modelId: string, 
    inputs: string, 
    parameters?: any, 
    apiKey?: string, 
    requestId?: string, 
    endpointId?: string
  ): Promise<any> {
    // Format endpoint URL for the model
    const endpointUrl = `${this.baseApiUrl}/${modelId}`;
    
    // Prepare data
    const data: HfTgiRequest = {
      inputs,
      parameters
    };
    
    // Make request with all the necessary parameters
    const response = await this.makePostRequestHfTgi(
      endpointUrl,
      data,
      apiKey,
      requestId,
      endpointId
    );
    
    // Process response based on format
    if (Array.isArray(response)) {
      // Some models return list of generated texts
      return response[0] || { generated_text: "" };
    } else if (typeof response === 'object') {
      // Some models return dict with generated_text key
      return response;
    } else {
      // Default fallback
      return { generated_text: String(response) };
    }
  }
  
  /**
   * Stream text generation from HF TGI API
   */
  async *streamGenerate(
    modelId: string, 
    inputs: string, 
    parameters?: any, 
    apiKey?: string, 
    requestId?: string, 
    endpointId?: string
  ): AsyncGenerator<any> {
    // Format endpoint URL for the model
    const endpointUrl = `${this.baseApiUrl}/${modelId}`;
    
    // Prepare data with streaming enabled
    const data: HfTgiRequest = {
      inputs,
      parameters,
      stream: true
    };
    
    // Prepare options for the request
    const options: ApiRequestOptions = {
      endpointUrl,
      apiKey,
      requestId,
      endpointId,
      model: modelId
    };
    
    // Get the streaming response using the standardized method
    const stream = this.makeStreamRequest(data, options);
    
    // Yield each chunk with request ID if provided
    for await (const chunk of stream) {
      yield chunk;
    }
  }
  
  /**
   * Format messages into a prompt suitable for text generation models
   */
  private formatChatMessages(messages: Message[]): string {
    let formattedPrompt = '';
    
    for (const message of messages) {
      const role = message.role || 'user';
      const content = typeof message.content === 'string' 
        ? message.content 
        : JSON.stringify(message.content);
      
      if (role === 'system') {
        formattedPrompt += `<|system|>\n${content}\n`;
      } else if (role === 'assistant') {
        formattedPrompt += `<|assistant|>\n${content}\n`;
      } else {  // user or default
        formattedPrompt += `<|user|>\n${content}\n`;
      }
    }
    
    // Add final assistant marker for completion
    formattedPrompt += '<|assistant|>\n';
    
    return formattedPrompt;
  }
  
  /**
   * Estimate token usage (rough approximation)
   */
  private estimateUsage(prompt: string, response: string): any {
    // Very rough approximation: 4 chars ~= 1 token
    const promptTokens = Math.ceil(prompt.length / 4);
    const completionTokens = Math.ceil(response.length / 4);
    
    return {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens
    };
  }

  /**
   * Format request parameters for HF TGI API
   */
  formatRequest(
    handler: (data: any) => Promise<any>, 
    text: string, 
    max_new_tokens?: number, 
    temperature?: number, 
    top_p?: number, 
    top_k?: number, 
    repetition_penalty?: number
  ): Promise<any> {
    // Prepare the request data
    const data: HfTgiRequest = {
      inputs: text
    };
    
    // Add parameters if provided
    if (max_new_tokens !== undefined || temperature !== undefined || 
        top_p !== undefined || top_k !== undefined || repetition_penalty !== undefined) {
      data.parameters = {};
      
      if (max_new_tokens !== undefined) data.parameters.max_new_tokens = max_new_tokens;
      if (temperature !== undefined) data.parameters.temperature = temperature;
      if (top_p !== undefined) data.parameters.top_p = top_p;
      if (top_k !== undefined) data.parameters.top_k = top_k;
      if (repetition_penalty !== undefined) data.parameters.repetition_penalty = repetition_penalty;
      
      // Set do_sample based on temperature
      data.parameters.do_sample = temperature !== 0 && temperature !== undefined;
    }
    
    // Call the handler with the formatted request
    return handler(data);
  }

  /**
   * Create a new endpoint with dedicated settings
   * Override from BaseApiBackend to handle HF TGI specific properties
   */
  createEndpoint(params: Partial<ApiEndpoint> | Partial<HfTgiEndpoint>): string {
    // Generate a unique endpoint ID if not provided
    const id = params.id || `endpoint_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    
    // Get HF TGI specific properties from params
    const hfParams = params as Partial<HfTgiEndpoint>;
    
    // Get API key - always ensure we have one to satisfy ApiEndpoint requirements
    const apiKey = hfParams.api_key || params.apiKey || this.getApiKey(this.metadata) || "";
    
    // Create a base endpoint compatible with both ApiEndpoint and HfTgiEndpoint
    const baseEndpoint: ApiEndpoint & Partial<HfTgiEndpoint> = {
      // Standard ApiEndpoint properties - all required properties must be present
      id,
      apiKey, // Required by ApiEndpoint
      model: params.model || this.getDefaultModel(),
      maxConcurrentRequests: params.maxConcurrentRequests || this.maxConcurrentRequests,
      queueSize: params.queueSize || this.queueSize,
      maxRetries: params.maxRetries || this.maxRetries,
      initialRetryDelay: params.initialRetryDelay || this.initialRetryDelay,
      backoffFactor: params.backoffFactor || this.backoffFactor,
      timeout: params.timeout || this.timeout,
      
      // HF TGI specific properties
      api_key: apiKey,
      model_id: hfParams.model_id || params.model || this.getDefaultModel(),
      endpoint_url: hfParams.endpoint_url || `${this.baseApiUrl}/${hfParams.model_id || params.model || this.getDefaultModel()}`,
      max_concurrent_requests: hfParams.max_concurrent_requests || params.maxConcurrentRequests || this.maxConcurrentRequests,
      queue_size: hfParams.queue_size || params.queueSize || this.queueSize,
      max_retries: hfParams.max_retries || params.maxRetries || this.maxRetries,
      initial_retry_delay: hfParams.initial_retry_delay || params.initialRetryDelay || this.initialRetryDelay,
      backoff_factor: hfParams.backoff_factor || params.backoffFactor || this.backoffFactor,
      
      // HF TGI tracking properties
      successful_requests: 0,
      failed_requests: 0,
      total_tokens: 0,
      input_tokens: 0,
      output_tokens: 0,
      current_requests: 0,
      queue_processing: false,
      request_queue: [],
      last_request_at: null,
      created_at: Date.now()
    };
    
    // Add any additional properties from params, ensuring we don't override required properties with undefined
    const mergedEndpoint: ApiEndpoint = {
      ...baseEndpoint,
      ...params,
      // Ensure required fields aren't overridden with undefined
      id,
      apiKey: params.apiKey || apiKey
    };
    
    // Set in the endpoints dictionary
    this.endpoints[id] = mergedEndpoint;
    
    return id;
  }

  /**
   * Get an endpoint by ID or create a default one
   */
  getEndpoint(endpointId?: string): HfTgiEndpoint {
    // If no endpoint ID provided, use the first one or create a default
    if (!endpointId) {
      if (Object.keys(this.endpoints).length === 0) {
        endpointId = this.createEndpoint({});
      } else {
        endpointId = Object.keys(this.endpoints)[0];
      }
    }
    
    // If endpoint doesn't exist, create it
    if (!this.endpoints[endpointId]) {
      endpointId = this.createEndpoint({ id: endpointId });
    }
    
    // Cast to HfTgiEndpoint since we know our implementation uses this type
    return this.endpoints[endpointId] as HfTgiEndpoint;
  }

  /**
   * Update an endpoint's settings
   */
  updateEndpoint(endpointId: string, params: Partial<HfTgiEndpoint>): HfTgiEndpoint {
    if (!this.endpoints[endpointId]) {
      throw new Error(`Endpoint ${endpointId} not found`);
    }
    
    // Get the endpoint as a HfTgiEndpoint
    const endpoint = this.endpoints[endpointId] as HfTgiEndpoint;
    
    // Update only the provided settings
    for (const [key, value] of Object.entries(params)) {
      if (key in endpoint) {
        // Use type assertion since we know this property exists
        (endpoint as any)[key] = value;
      }
    }
    
    // Cast the updated endpoint back to HfTgiEndpoint
    return endpoint;
  }

  /**
   * Get usage statistics for an endpoint or global stats
   */
  getStats(endpointId?: string): HfTgiStats | any {
    if (endpointId && this.endpoints[endpointId]) {
      // Cast to HfTgiEndpoint to access specific properties
      const endpoint = this.endpoints[endpointId] as HfTgiEndpoint;
      
      return {
        endpoint_id: endpointId,
        total_requests: endpoint.successful_requests + endpoint.failed_requests,
        successful_requests: endpoint.successful_requests,
        failed_requests: endpoint.failed_requests,
        total_tokens: endpoint.total_tokens,
        input_tokens: endpoint.input_tokens,
        output_tokens: endpoint.output_tokens,
        created_at: endpoint.created_at,
        last_request_at: endpoint.last_request_at,
        current_queue_size: endpoint.request_queue.length,
        current_requests: endpoint.current_requests
      };
    } else {
      // Aggregate stats across all endpoints
      const endpointCount = Object.keys(this.endpoints).length;
      
      // If no endpoints, return empty stats
      if (endpointCount === 0) {
        return {
          endpoints_count: 0,
          total_requests: 0,
          successful_requests: 0,
          failed_requests: 0,
          total_tokens: 0,
          input_tokens: 0,
          output_tokens: 0,
          global_queue_size: 0,
          global_current_requests: 0
        };
      }
      
      // Cast endpoints to HfTgiEndpoint array for type safety
      const typedEndpoints = Object.values(this.endpoints) as HfTgiEndpoint[];
      
      // Sum stats across all endpoints
      const totalRequests = typedEndpoints.reduce(
        (sum, e) => sum + e.successful_requests + e.failed_requests, 0
      );
      
      const successfulRequests = typedEndpoints.reduce(
        (sum, e) => sum + e.successful_requests, 0
      );
      
      const failedRequests = typedEndpoints.reduce(
        (sum, e) => sum + e.failed_requests, 0
      );
      
      const totalTokens = typedEndpoints.reduce(
        (sum, e) => sum + e.total_tokens, 0
      );
      
      const inputTokens = typedEndpoints.reduce(
        (sum, e) => sum + e.input_tokens, 0
      );
      
      const outputTokens = typedEndpoints.reduce(
        (sum, e) => sum + e.output_tokens, 0
      );
      
      const queueSize = typedEndpoints.reduce(
        (sum, e) => sum + e.request_queue.length, 0
      );
      
      const currentRequests = typedEndpoints.reduce(
        (sum, e) => sum + e.current_requests, 0
      );
      
      return {
        endpoints_count: endpointCount,
        total_requests: totalRequests,
        successful_requests: successfulRequests,
        failed_requests: failedRequests,
        total_tokens: totalTokens,
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        global_queue_size: queueSize,
        global_current_requests: currentRequests
      };
    }
  }

  /**
   * Reset usage statistics for an endpoint or globally
   */
  resetStats(endpointId?: string): void {
    if (endpointId && this.endpoints[endpointId]) {
      // Reset stats for a specific endpoint
      const endpoint = this.endpoints[endpointId] as HfTgiEndpoint;
      endpoint.successful_requests = 0;
      endpoint.failed_requests = 0;
      endpoint.total_tokens = 0;
      endpoint.input_tokens = 0;
      endpoint.output_tokens = 0;
    } else if (!endpointId) {
      // Reset stats for all endpoints
      const typedEndpoints = Object.values(this.endpoints) as HfTgiEndpoint[];
      for (const endpoint of typedEndpoints) {
        endpoint.successful_requests = 0;
        endpoint.failed_requests = 0;
        endpoint.total_tokens = 0;
        endpoint.input_tokens = 0;
        endpoint.output_tokens = 0;
      }
    } else {
      throw new Error(`Endpoint ${endpointId} not found`);
    }
  }

  /**
   * Make a request using a specific endpoint
   * Implements the method from BaseApiBackend
   */
  async makeRequestWithEndpoint(
    endpointId: string,
    data: any,
    options?: ApiRequestOptions
  ): Promise<any> {
    if (!this.endpoints[endpointId]) {
      throw new Error(`Endpoint ${endpointId} not found`);
    }
    
    // Cast to HfTgiEndpoint to access specific properties
    const endpoint = this.endpoints[endpointId] as HfTgiEndpoint;
    const reqId = options?.requestId || `${endpointId}_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    
    // Combine options with endpoint-specific settings
    const combinedOptions: ApiRequestOptions = {
      ...options,
      endpointUrl: endpoint.endpoint_url,
      requestId: reqId,
      endpointId
    };
    
    try {
      const response = await this.makePostRequest(data, endpoint.api_key, combinedOptions);
      
      // Track the request result
      this.trackRequestResult(true, reqId);
      
      return response;
    } catch (error) {
      // Track the failed request
      this.trackRequestResult(false, reqId, error as Error);
      throw error;
    }
  }

  /**
   * Chat completion with HF TGI API
   * Implements the abstract method from BaseApiBackend
   */
  async chat(messages: Message[], options?: ApiRequestOptions): Promise<ChatCompletionResponse> {
    // Convert ApiRequestOptions to HfTgiChatOptions if needed
    const chatOptions: HfTgiChatOptions = options as HfTgiChatOptions || {};
    
    // Format messages into a prompt
    const prompt = this.formatChatMessages(messages);
    
    // Extract parameters from options
    const parameters: any = {};
    if (chatOptions) {
      if (chatOptions.temperature !== undefined) parameters.temperature = chatOptions.temperature;
      if (chatOptions.max_new_tokens !== undefined) parameters.max_new_tokens = chatOptions.max_new_tokens;
      if (chatOptions.top_p !== undefined) parameters.top_p = chatOptions.top_p;
      if (chatOptions.top_k !== undefined) parameters.top_k = chatOptions.top_k;
      if (chatOptions.repetition_penalty !== undefined) parameters.repetition_penalty = chatOptions.repetition_penalty;
      if (chatOptions.return_full_text !== undefined) parameters.return_full_text = chatOptions.return_full_text;
      parameters.do_sample = chatOptions.do_sample ?? (chatOptions.temperature !== 0 && chatOptions.temperature !== undefined);
    }
    
    // Generate request ID if not provided
    const requestId = options?.requestId || `chat_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    
    // Use endpoint ID from options if provided
    const endpointId = chatOptions.endpointId;
    
    // Get the model to use
    const modelId = chatOptions.model || this.getDefaultModel();
    
    try {
      // Create request data
      const data: HfTgiRequest = {
        inputs: prompt,
        parameters
      };
      
      // Create API request options
      const apiOptions: ApiRequestOptions = {
        ...options,
        requestId,
        endpointId,
        model: modelId,
        endpointUrl: `${this.baseApiUrl}/${modelId}`
      };
      
      // Make request
      const response = await this.makePostRequest(data, undefined, apiOptions);
      
      // Extract text from response
      let text = '';
      if (typeof response === 'object') {
        text = response.generated_text || '';
      } else {
        text = String(response);
      }
      
      // Calculate usage estimates
      const usage = this.estimateUsage(prompt, text);
      
      // Track the request result
      this.trackRequestResult(true, requestId);
      
      // Return standardized response
      return {
        text,
        model: modelId,
        usage,
        requestId,
        implementation_type: '(REAL)'
      };
    } catch (error) {
      // Track the failed request
      this.trackRequestResult(false, requestId, error as Error);
      throw error;
    }
  }

  /**
   * Stream chat completions with HF TGI API
   * Implements the abstract method from BaseApiBackend
   */
  async *streamChat(messages: Message[], options?: ApiRequestOptions): AsyncGenerator<StreamChunk> {
    // Convert ApiRequestOptions to HfTgiChatOptions if needed
    const chatOptions: HfTgiChatOptions = options as HfTgiChatOptions || {};
    
    // Format messages into a prompt
    const prompt = this.formatChatMessages(messages);
    
    // Extract parameters from options
    const parameters: any = {};
    if (chatOptions) {
      if (chatOptions.temperature !== undefined) parameters.temperature = chatOptions.temperature;
      if (chatOptions.max_new_tokens !== undefined) parameters.max_new_tokens = chatOptions.max_new_tokens;
      if (chatOptions.top_p !== undefined) parameters.top_p = chatOptions.top_p;
      if (chatOptions.top_k !== undefined) parameters.top_k = chatOptions.top_k;
      if (chatOptions.repetition_penalty !== undefined) parameters.repetition_penalty = chatOptions.repetition_penalty;
      if (chatOptions.return_full_text !== undefined) parameters.return_full_text = chatOptions.return_full_text;
      parameters.do_sample = chatOptions.do_sample ?? (chatOptions.temperature !== 0 && chatOptions.temperature !== undefined);
    }
    
    // Generate request ID if not provided
    const requestId = options?.requestId || `stream_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    
    // Use endpoint ID from options if provided
    const endpointId = chatOptions.endpointId;
    
    // Get the model to use
    const modelId = chatOptions.model || this.getDefaultModel();
    
    // Create data object
    const data: HfTgiRequest = {
      inputs: prompt,
      parameters,
      stream: true
    };
    
    // Prepare API request options
    const apiOptions: ApiRequestOptions = {
      ...options,
      requestId,
      endpointId,
      model: modelId
    };
    
    // Generate streaming text using the standardized method
    const stream = this.makeStreamRequest(data, apiOptions);
    
    // Process and yield each chunk
    for await (const chunk of stream) {
      yield {
        content: chunk.content || '',
        type: 'delta',
        done: chunk.done || false,
        model: modelId,
        requestId,
        implementation_type: '(REAL)'
      };
    }
  }
}