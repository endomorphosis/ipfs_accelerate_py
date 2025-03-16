import { BaseApiBackend } from '../base';
import { ApiMetadata, ApiRequestOptions, Message, ChatCompletionResponse, StreamChunk } from '../types';
import {
  VllmRequest,
  VllmUnifiedResponse,
  VllmBatchResponse,
  VllmStreamChunk,
  VllmModelInfo,
  VllmModelStatistics,
  VllmLoraAdapter,
  VllmLoadLoraResponse,
  VllmQuantizationConfig,
  VllmQuantizationResponse
} from './types';

/**
 * VLLM Unified API backend for advanced text generation
 * Supports standard and batch inference, streaming, and advanced model management
 */
export class VllmUnified extends BaseApiBackend {
  private defaultApiUrl: string = 'http://localhost:8000';
  private defaultModel: string = 'meta-llama/Llama-2-7b-chat-hf';
  private defaultTimeout: number = 30000; // 30 seconds

  // Standard test inputs for consistency tests
  private testInputs = [
    "This is a simple text input",
    ["Input 1", "Input 2", "Input 3"], // Batch of text inputs
    { prompt: "Test input with parameters", parameters: { temperature: 0.7 }},
    { input: "Standard format input", parameters: { max_tokens: 100 }}
  ];

  // Test parameters for various configurations
  private testParameters = {
    "default": {},
    "creative": { temperature: 0.9, top_p: 0.95 },
    "precise": { temperature: 0.1, top_p: 0.1 },
    "fast": { max_tokens: 50, top_k: 10 },
    "sampling": { use_beam_search: false, temperature: 0.8, top_p: 0.9 },
    "beam_search": { use_beam_search: true, n: 3 }
  };

  constructor(resources: Record<string, any> = {}, metadata: ApiMetadata = {}) {
    super(resources, metadata);
  }

  /**
   * Get API key from metadata or environment variables
   * Note: VLLM often doesn't require an API key for local deployments
   */
  protected getApiKey(metadata: ApiMetadata): string {
    return metadata.vllm_api_key || 
           metadata.vllmApiKey || 
           (typeof process !== 'undefined' ? process.env.VLLM_API_KEY || '' : '');
  }

  /**
   * Get default model for this API backend
   */
  protected getDefaultModel(): string {
    return this.metadata.vllm_model || 
           this.metadata.vllmModel || 
           (typeof process !== 'undefined' ? process.env.VLLM_MODEL || this.defaultModel : this.defaultModel);
  }

  /**
   * Check if a model is compatible with VLLM
   * VLLM generally supports most transformer-based models
   */
  isCompatibleModel(model: string): boolean {
    // VLLM supports most transformer models, especially those from HuggingFace
    return (
      model.includes('llama') || 
      model.includes('falcon') || 
      model.includes('bloom') || 
      model.includes('gpt') || 
      model.includes('mistral') || 
      model.includes('mixtral') || 
      model.includes('mpt') || 
      model.includes('opt') || 
      model.startsWith('meta/') ||
      model.startsWith('meta-llama/') ||
      model.startsWith('tiiuae/') ||
      model.includes('codellama') ||
      model.includes('vicuna') ||
      model.includes('starcoder') ||
      model.includes('pythia') ||
      model.includes('stablelm') ||
      model.includes('cerebras') ||
      model.includes('qwen')
    );
  }

  /**
   * Create an endpoint handler for VLLM
   */
  createEndpointHandler(): (data: any) => Promise<any> {
    return async (data: any) => {
      try {
        const endpointUrl = this.metadata.vllm_api_url || 
                            this.metadata.vllmApiUrl || 
                            (typeof process !== 'undefined' ? process.env.VLLM_API_URL || this.defaultApiUrl : this.defaultApiUrl);
        const model = this.getDefaultModel();
        
        return await this.makeRequest(endpointUrl, data, model);
      } catch (error) {
        throw this.createApiError(`${this.constructor.name} endpoint error: ${error.message}`, 500);
      }
    };
  }

  /**
   * Create a VLLM endpoint handler for a specific URL and model
   */
  createVllmEndpointHandler(endpointUrl: string, model?: string): (data: any) => Promise<any> {
    return async (data: any) => {
      try {
        const modelName = model || this.getDefaultModel();
        return await this.makeRequest(endpointUrl, data, modelName);
      } catch (error) {
        throw this.createApiError(`VLLM endpoint error: ${error.message}`, 500);
      }
    };
  }

  /**
   * Create a VLLM endpoint handler with specific parameters
   */
  createVllmEndpointHandlerWithParams(
    endpointUrl: string, 
    model?: string, 
    parameters?: Record<string, any>
  ): (data: any) => Promise<any> {
    return async (data: any) => {
      try {
        const modelName = model || this.getDefaultModel();
        const requestData = this.prepareRequestData(data, modelName, parameters);
        return await this.makePostRequestVllm(endpointUrl, requestData);
      } catch (error) {
        throw this.createApiError(`VLLM endpoint error: ${error.message}`, 500);
      }
    };
  }

  /**
   * Test the VLLM endpoint
   */
  async testEndpoint(): Promise<boolean> {
    try {
      const endpointUrl = this.metadata.vllm_api_url || 
                          this.metadata.vllmApiUrl || 
                          (typeof process !== 'undefined' ? process.env.VLLM_API_URL || this.defaultApiUrl : this.defaultApiUrl);
      const model = this.getDefaultModel();

      // Make a minimal request to verify the endpoint works
      await this.makeRequest(endpointUrl, { prompt: "Hello", max_tokens: 5 }, model);
      return true;
    } catch (error) {
      console.error(`${this.constructor.name} endpoint test failed:`, error);
      return false;
    }
  }

  /**
   * Test a specific VLLM endpoint
   */
  async testVllmEndpoint(endpointUrl: string, model?: string): Promise<boolean> {
    try {
      const modelName = model || this.getDefaultModel();
      
      // Prepare a minimal test request
      const testRequest: VllmRequest = {
        prompt: "Hello, world!",
        max_tokens: 5,
        model: modelName
      };
      
      // Make the request
      await this.makePostRequestVllm(endpointUrl, testRequest);
      return true;
    } catch (error) {
      console.error(`VLLM endpoint test failed:`, error);
      return false;
    }
  }

  /**
   * Test a VLLM endpoint with specific parameters
   */
  async testVllmEndpointWithParams(
    endpointUrl: string, 
    model?: string, 
    parameters?: Record<string, any>
  ): Promise<boolean> {
    try {
      const modelName = model || this.getDefaultModel();
      
      // Prepare test request with parameters
      const testRequest: VllmRequest = {
        prompt: "Hello, world!",
        model: modelName,
        ...parameters
      };
      
      // Make the request
      await this.makePostRequestVllm(endpointUrl, testRequest);
      return true;
    } catch (error) {
      console.error(`VLLM endpoint test with parameters failed:`, error);
      return false;
    }
  }

  /**
   * Make a request to the VLLM API
   */
  async makeRequest(
    endpointUrl: string, 
    data: any, 
    model?: string, 
    options?: ApiRequestOptions
  ): Promise<any> {
    const modelName = model || this.getDefaultModel();
    const requestData = this.prepareRequestData(data, modelName);
    return await this.makePostRequestVllm(endpointUrl, requestData, options?.requestId, options);
  }

  /**
   * Make a POST request to the VLLM API
   */
  async makePostRequestVllm(
    endpointUrl: string,
    data: any,
    requestId?: string,
    options?: ApiRequestOptions
  ): Promise<VllmUnifiedResponse> {
    // Process with queue and circuit breaker
    return this.retryableRequest(async () => {
      // Prepare request headers
      const headers: Record<string, string> = {
        'Content-Type': 'application/json'
      };

      // Add API key if available
      const apiKey = options?.apiKey || this.getApiKey(this.metadata);
      if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`;
      }
      
      // Add request ID if provided
      if (requestId) {
        headers['X-Request-ID'] = requestId;
      }

      // Prepare request body
      const requestBody = JSON.stringify(data);

      // Set up timeout
      const timeoutMs = options?.timeout || 
                       this.metadata.timeout || 
                       (typeof process !== 'undefined' ? parseInt(process.env.VLLM_TIMEOUT || '30000', 10) : this.defaultTimeout);
      
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
          const errorMessage = errorData.error || `HTTP error ${response.status}`;
          
          // Handle rate limiting
          if (response.status === 429) {
            const retryAfter = response.headers.get('retry-after');
            const error = this.createApiError(errorMessage, response.status, 'rate_limit_error');
            error.retryAfter = retryAfter ? parseInt(retryAfter, 10) : 1;
            error.isRateLimitError = true;
            throw error;
          }
          
          throw this.createApiError(errorMessage, response.status);
        }

        // Parse response
        const responseData = await response.json();
        return responseData;
      } catch (error) {
        if (error.name === 'AbortError') {
          throw this.createApiError(`Request timed out after ${timeoutMs}ms`, 408, 'timeout_error');
        }
        throw error;
      } finally {
        clearTimeout(timeoutId);
      }
    }, options?.maxRetries || this.maxRetries);
  }

  /**
   * Make a streaming request to the VLLM API
   */
  async *makeStreamRequestVllm(
    endpointUrl: string,
    data: any,
    options?: ApiRequestOptions
  ): AsyncGenerator<VllmStreamChunk> {
    // Prepare request headers
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };

    // Add API key if available
    const apiKey = options?.apiKey || this.getApiKey(this.metadata);
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }
    
    // Add request ID if provided
    if (options?.requestId) {
      headers['X-Request-ID'] = options.requestId;
    }

    // Ensure stream option is set
    const streamData = { ...data, stream: true };

    // Prepare request body
    const requestBody = JSON.stringify(streamData);

    // Set up timeout
    const timeoutMs = options?.timeout || 
                     this.metadata.timeout || 
                     (typeof process !== 'undefined' ? parseInt(process.env.VLLM_TIMEOUT || '60000', 10) : 60000);
    
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
        throw this.createApiError(
          errorData.error || `HTTP error ${response.status}`,
          response.status
        );
      }

      // Process the stream
      const reader = response.body?.getReader();
      if (!reader) {
        throw this.createApiError('Response body is null', 500, 'stream_error');
      }

      // VLLM typically returns newline-delimited JSON
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
            // Parse the JSON chunk
            const parsed = JSON.parse(line);
            yield parsed;
          } catch (e) {
            console.warn('Failed to parse stream data:', line);
          }
        }
      }

      // Process any remaining data in the buffer
      if (buffer.trim() !== '') {
        try {
          const parsed = JSON.parse(buffer);
          yield parsed;
        } catch (e) {
          console.warn('Failed to parse final stream data:', buffer);
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        throw this.createApiError(`Stream request timed out after ${timeoutMs}ms`, 408, 'timeout_error');
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Process a batch of requests
   */
  async processBatch(
    endpointUrl: string,
    batchData: string[],
    model?: string,
    parameters?: Record<string, any>
  ): Promise<string[]> {
    // Prepare batch request
    const modelName = model || this.getDefaultModel();
    const requestData = {
      prompts: batchData,
      model: modelName,
      ...parameters
    };
    
    // Make the request
    const response = await this.makePostRequestVllm(`${endpointUrl}/batch`, requestData) as VllmBatchResponse;
    
    // Return the results
    return response.texts || [];
  }

  /**
   * Process a batch of requests with parameters
   */
  async processBatchWithParams(
    endpointUrl: string,
    batchData: string[],
    model?: string,
    parameters?: Record<string, any>
  ): Promise<string[]> {
    // Use the base batch processing method with parameters
    return this.processBatch(endpointUrl, batchData, model, parameters);
  }

  /**
   * Process a batch with detailed metrics
   */
  async processBatchWithMetrics(
    endpointUrl: string,
    batchData: string[],
    model?: string,
    parameters?: Record<string, any>
  ): Promise<[string[], Record<string, any>]> {
    // Prepare batch request
    const modelName = model || this.getDefaultModel();
    const requestData = {
      prompts: batchData,
      model: modelName,
      ...parameters
    };
    
    // Make the request
    const response = await this.makePostRequestVllm(`${endpointUrl}/batch`, requestData) as VllmBatchResponse;
    
    // Extract metrics
    const metrics = {
      model: response.metadata?.model || modelName,
      finish_reasons: response.metadata?.finish_reasons || [],
      usage: response.metadata?.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };
    
    // Return the results and metrics
    return [response.texts || [], metrics];
  }

  /**
   * Stream text generation
   */
  async *streamGeneration(
    endpointUrl: string,
    prompt: string,
    model?: string,
    parameters?: Record<string, any>
  ): AsyncGenerator<VllmStreamChunk> {
    // Prepare request data
    const modelName = model || this.getDefaultModel();
    const requestData = {
      prompt,
      model: modelName,
      stream: true,
      ...parameters
    };
    
    // Get the stream
    yield* this.makeStreamRequestVllm(endpointUrl, requestData);
  }

  /**
   * Stream text generation with parameters
   */
  async *streamGenerationWithParams(
    endpointUrl: string,
    prompt: string,
    model?: string,
    parameters?: Record<string, any>
  ): AsyncGenerator<VllmStreamChunk> {
    // Use the base streaming method with parameters
    yield* this.streamGeneration(endpointUrl, prompt, model, parameters);
  }

  /**
   * Get information about a model
   */
  async getModelInfo(
    endpointUrl: string,
    model?: string
  ): Promise<VllmModelInfo> {
    const modelName = model || this.getDefaultModel();
    
    // Make a GET request to the model info endpoint
    const response = await fetch(`${endpointUrl}/models/${modelName}`);
    
    // Check for errors
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw this.createApiError(
        errorData.error || `HTTP error ${response.status}`,
        response.status
      );
    }
    
    // Parse and return model info
    return await response.json() as VllmModelInfo;
  }

  /**
   * Get model statistics
   */
  async getModelStatistics(
    endpointUrl: string,
    model?: string
  ): Promise<VllmModelStatistics> {
    const modelName = model || this.getDefaultModel();
    
    // Make a GET request to the model statistics endpoint
    const response = await fetch(`${endpointUrl}/models/${modelName}/stats`);
    
    // Check for errors
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw this.createApiError(
        errorData.error || `HTTP error ${response.status}`,
        response.status
      );
    }
    
    // Parse and return model statistics
    return await response.json() as VllmModelStatistics;
  }

  /**
   * List LoRA adapters
   */
  async listLoraAdapters(
    endpointUrl: string
  ): Promise<VllmLoraAdapter[]> {
    // Make a GET request to the LoRA adapters endpoint
    const response = await fetch(`${endpointUrl}/lora_adapters`);
    
    // Check for errors
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw this.createApiError(
        errorData.error || `HTTP error ${response.status}`,
        response.status
      );
    }
    
    // Parse and return LoRA adapters list
    const responseJson = await response.json();
    return responseJson.lora_adapters || [];
  }

  /**
   * Load a LoRA adapter
   */
  async loadLoraAdapter(
    endpointUrl: string,
    adapterData: Record<string, any>
  ): Promise<VllmLoadLoraResponse> {
    // Prepare request headers
    const headers = {
      'Content-Type': 'application/json'
    };
    
    // Make a POST request to the LoRA adapter loading endpoint
    const response = await fetch(`${endpointUrl}/lora_adapters/load`, {
      method: 'POST',
      headers,
      body: JSON.stringify(adapterData)
    });
    
    // Check for errors
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw this.createApiError(
        errorData.error || `HTTP error ${response.status}`,
        response.status
      );
    }
    
    // Parse and return response
    return await response.json() as VllmLoadLoraResponse;
  }

  /**
   * Set quantization configuration
   */
  async setQuantization(
    endpointUrl: string,
    model: string,
    config: VllmQuantizationConfig
  ): Promise<VllmQuantizationResponse> {
    // Prepare request headers
    const headers = {
      'Content-Type': 'application/json'
    };
    
    // Prepare request data
    const requestData = {
      model,
      quantization: config
    };
    
    // Make a POST request to the quantization endpoint
    const response = await fetch(`${endpointUrl}/models/${model}/quantize`, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestData)
    });
    
    // Check for errors
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw this.createApiError(
        errorData.error || `HTTP error ${response.status}`,
        response.status
      );
    }
    
    // Parse and return response
    return await response.json() as VllmQuantizationResponse;
  }

  /**
   * Format a request for the VLLM API
   */
  formatRequest(
    handler: (data: any) => Promise<any>,
    input: any
  ): Promise<any> {
    // Format different types of inputs
    if (typeof input === 'string') {
      // Simple string prompt
      return handler({ prompt: input });
    } else if (Array.isArray(input)) {
      // Batch of prompts
      return handler({ prompts: input });
    } else if (typeof input === 'object') {
      // Object with prompt and parameters
      if (input.prompt) {
        return handler({ prompt: input.prompt, ...input.parameters });
      } else if (input.input) {
        // Alternative input field
        return handler({ prompt: input.input, ...input.parameters });
      }
    }
    
    // Default fallback
    return handler(input);
  }

  /**
   * Format a request with parameters
   */
  formatRequestWithParams(
    handler: (data: any) => Promise<any>,
    input: any,
    parameters: Record<string, any>
  ): Promise<any> {
    // Format different types of inputs with parameters
    if (typeof input === 'string') {
      // Simple string prompt with parameters
      return handler({ prompt: input, ...parameters });
    } else if (Array.isArray(input)) {
      // Batch of prompts with parameters
      return handler({ prompts: input, ...parameters });
    } else if (typeof input === 'object') {
      // Merge parameters with existing ones
      const mergedParams = { ...input.parameters, ...parameters };
      
      if (input.prompt) {
        return handler({ prompt: input.prompt, ...mergedParams });
      } else if (input.input) {
        return handler({ prompt: input.input, ...mergedParams });
      }
    }
    
    // Default fallback
    return handler({ ...input, ...parameters });
  }

  /**
   * Format a chat request
   */
  formatChatRequest(
    handler: (data: any) => Promise<any>,
    messages: Message[]
  ): Promise<any> {
    // Format chat messages for VLLM
    return handler({ messages });
  }

  /**
   * Chat completion implementation
   */
  async chat(
    messages: Message[],
    options?: ApiRequestOptions
  ): Promise<ChatCompletionResponse> {
    // Get API URL and model from metadata or environment
    const endpointUrl = this.metadata.vllm_api_url || 
                        this.metadata.vllmApiUrl || 
                        (typeof process !== 'undefined' ? process.env.VLLM_API_URL || this.defaultApiUrl : this.defaultApiUrl);
    const modelName = options?.model || this.getDefaultModel();
    
    // Prepare request data
    const requestData: VllmRequest = {
      messages,
      model: modelName,
      max_tokens: options?.maxTokens,
      temperature: options?.temperature,
      top_p: options?.topP,
      stream: false
    };
    
    // Make the request
    const response = await this.makePostRequestVllm(endpointUrl, requestData, options?.requestId, options);
    
    // Convert to standard format
    return {
      model: modelName,
      content: response.text || '',
      usage: response.metadata?.usage || { 
        prompt_tokens: 0, 
        completion_tokens: 0, 
        total_tokens: 0 
      }
    };
  }

  /**
   * Streaming chat completion
   */
  async *streamChat(
    messages: Message[],
    options?: ApiRequestOptions
  ): AsyncGenerator<StreamChunk> {
    // Get API URL and model from metadata or environment
    const endpointUrl = this.metadata.vllm_api_url || 
                        this.metadata.vllmApiUrl || 
                        (typeof process !== 'undefined' ? process.env.VLLM_API_URL || this.defaultApiUrl : this.defaultApiUrl);
    const modelName = options?.model || this.getDefaultModel();
    
    // Prepare request data
    const requestData: VllmRequest = {
      messages,
      model: modelName,
      max_tokens: options?.maxTokens,
      temperature: options?.temperature,
      top_p: options?.topP,
      stream: true
    };
    
    // Get the stream
    const stream = this.makeStreamRequestVllm(endpointUrl, requestData, options);
    
    // Process and yield stream chunks
    for await (const chunk of stream) {
      yield {
        content: chunk.text || '',
        type: 'delta',
        done: chunk.metadata?.finish_reason !== null && chunk.metadata?.is_streaming === false
      };
    }
  }

  /**
   * Make a request with a specific endpoint
   */
  async makeRequestWithEndpoint(
    endpointId: string,
    prompt: any,
    model?: string,
    parameters?: Record<string, any>
  ): Promise<any> {
    // Check if endpoint exists
    if (!this.endpoints[endpointId]) {
      throw this.createApiError(`Endpoint ${endpointId} not found`, 404);
    }
    
    const endpoint = this.endpoints[endpointId];
    const requestId = `${endpointId}_${Date.now()}`;
    const modelName = model || endpoint.model || this.getDefaultModel();
    
    // Get endpoint URL from metadata
    const endpointUrl = this.metadata.vllm_api_url || 
                        this.metadata.vllmApiUrl || 
                        (typeof process !== 'undefined' ? process.env.VLLM_API_URL || this.defaultApiUrl : this.defaultApiUrl);
    
    // Prepare request data based on input type
    let requestData;
    if (typeof prompt === 'string') {
      requestData = {
        prompt,
        model: modelName,
        ...parameters
      };
    } else if (Array.isArray(prompt)) {
      requestData = {
        messages: prompt,
        model: modelName,
        ...parameters
      };
    } else {
      requestData = {
        ...prompt,
        model: modelName,
        ...parameters
      };
    }
    
    // Make the request with endpoint configuration
    try {
      const response = await this.makePostRequestVllm(
        endpointUrl,
        requestData,
        requestId,
        {
          apiKey: endpoint.apiKey,
          maxRetries: endpoint.maxRetries,
          timeout: endpoint.timeout
        }
      );
      
      this.trackRequestResult(true, requestId);
      return response;
    } catch (error) {
      this.trackRequestResult(false, requestId, error);
      throw error;
    }
  }

  /**
   * Helper method to prepare request data
   */
  private prepareRequestData(
    data: any,
    model: string,
    parameters?: Record<string, any>
  ): VllmRequest {
    // Format different types of inputs with optional parameters
    if (typeof data === 'string') {
      // Simple string prompt
      return {
        prompt: data,
        model,
        ...parameters
      };
    } else if (Array.isArray(data)) {
      if (data.every(item => typeof item === 'object' && ('role' in item || 'content' in item))) {
        // Array of messages
        return {
          messages: data,
          model,
          ...parameters
        };
      } else {
        // Batch of prompts
        return {
          prompts: data,
          model,
          ...parameters
        };
      }
    } else if (data.messages) {
      // Chat messages
      return {
        messages: data.messages,
        model,
        ...parameters,
        ...data
      };
    } else if (data.prompt) {
      // Object with prompt field
      return {
        prompt: data.prompt,
        model,
        ...parameters,
        ...data
      };
    } else if (data.input) {
      // Object with input field
      return {
        prompt: data.input,
        model,
        ...parameters,
        ...data
      };
    }
    
    // Default fallback
    return {
      ...data,
      model,
      ...parameters
    };
  }
}