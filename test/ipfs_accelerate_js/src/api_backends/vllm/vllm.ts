/**
 * VLLM API Backend
 */

import { 
  ApiResources, 
  ApiMetadata, 
  ApiRequestOptions,
  Message,
  ChatCompletionResponse,
  StreamChunk
} from '../types';
import { BaseApiBackend } from '../base';
import { 
  VLLMRequestData, 
  VLLMResponse, 
  VLLMRequestOptions,
  VLLMModelInfo,
  VLLMModelStatistics,
  VLLMLoraAdapter,
  VLLMQuantizationConfig
} from './types';

export class VLLM extends BaseApiBackend {
  protected apiUrl: string;
  protected defaultModel: string;
  
  constructor(resources: ApiResources = {}, metadata: ApiMetadata = {}) {
    super(resources, metadata);
    
    // Initialize VLLM-specific properties
    this.apiUrl = metadata.vllm_api_url || process.env.VLLM_API_URL || 'http://localhost:8000';
    this.defaultModel = metadata.vllm_model || process.env.VLLM_MODEL || 'meta-llama/Llama-2-7b-chat-hf';
    this.model = this.defaultModel;
    
    // Set timeout for VLLM requests
    this.timeout = metadata.timeout || parseInt(process.env.VLLM_TIMEOUT || '30') * 1000;
  }
  
  /**
   * Get API key - VLLM typically doesn't use API keys, but implementation
   * supports them for secure deployments
   */
  protected getApiKey(metadata: ApiMetadata): string {
    return metadata.vllm_api_key || process.env.VLLM_API_KEY || '';
  }
  
  /**
   * Get the default model
   */
  protected getDefaultModel(): string {
    return 'meta-llama/Llama-2-7b-chat-hf';
  }
  
  /**
   * Create an endpoint handler for making VLLM requests
   */
  createEndpointHandler(
    endpointUrl: string = `${this.apiUrl}/v1/completions`
  ): (data: VLLMRequestData) => Promise<VLLMResponse> {
    return async (data: VLLMRequestData): Promise<VLLMResponse> => {
      return this.makePostRequestVLLM(endpointUrl, data);
    };
  }
  
  /**
   * Create VLLM chat endpoint handler
   */
  createVLLMChatEndpointHandler(
    endpointUrl: string = `${this.apiUrl}/v1/chat/completions`
  ): (data: VLLMRequestData) => Promise<VLLMResponse> {
    return async (data: VLLMRequestData): Promise<VLLMResponse> => {
      return this.makePostRequestVLLM(endpointUrl, data);
    };
  }
  
  /**
   * Test the VLLM endpoint
   */
  async testEndpoint(
    endpointUrl: string = `${this.apiUrl}/v1/completions`,
    model: string = this.model
  ): Promise<boolean> {
    try {
      // Create a simple test prompt
      const testData: VLLMRequestData = {
        prompt: "Test prompt",
        max_tokens: 5,
        model
      };
      
      const response = await this.makePostRequestVLLM(endpointUrl, testData);
      return !!response && (!!response.text || (!!response.choices && response.choices.length > 0));
    } catch (error) {
      console.error('VLLM endpoint test failed:', error);
      return false;
    }
  }
  
  /**
   * Test the VLLM chat endpoint
   */
  async testVLLMChatEndpoint(
    endpointUrl: string = `${this.apiUrl}/v1/chat/completions`,
    model: string = this.model
  ): Promise<boolean> {
    try {
      // Create a simple test message
      const testData: VLLMRequestData = {
        messages: [
          { role: "user", content: "Hello" }
        ],
        max_tokens: 5,
        model
      };
      
      const response = await this.makePostRequestVLLM(endpointUrl, testData);
      return !!response && 
        (!!response.choices && response.choices.length > 0) && 
        (!!response.choices[0].message || !!response.choices[0].text);
    } catch (error) {
      console.error('VLLM chat endpoint test failed:', error);
      return false;
    }
  }
  
  /**
   * Make a POST request to the VLLM server
   */
  async makePostRequestVLLM(
    endpointUrl: string,
    data: VLLMRequestData,
    options: VLLMRequestOptions = {}
  ): Promise<VLLMResponse> {
    const requestId = options.requestId || `vllm_${Date.now()}`;
    const apiKey = options.apiKey || this.apiKey;
    
    // Set headers
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };
    
    // Add API key if available
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }
    
    // Add request tracking if enabled
    if (this.requestTracking) {
      headers['X-Request-ID'] = requestId;
    }
    
    // Apply request options to data
    const requestData = { ...data };
    
    // Apply generation parameters if provided
    if (options.temperature !== undefined) requestData.temperature = options.temperature;
    if (options.top_p !== undefined) requestData.top_p = options.top_p;
    if (options.top_k !== undefined) requestData.top_k = options.top_k;
    if (options.max_tokens !== undefined) requestData.max_tokens = options.max_tokens;
    if (options.stop !== undefined) requestData.stop = options.stop;
    
    try {
      return await this.retryableRequest(async () => {
        const response = await fetch(endpointUrl, {
          method: 'POST',
          headers,
          body: JSON.stringify(requestData),
          signal: options.signal || AbortSignal.timeout(options.timeout || this.timeout)
        });
        
        if (!response.ok) {
          const errorText = await response.text();
          throw this.createApiError(
            `VLLM API error: ${errorText}`,
            response.status,
            'vllm_error'
          );
        }
        
        return await response.json();
      });
    } catch (error) {
      // Track the error if request tracking is enabled
      this.trackRequestResult(false, requestId, error as Error);
      
      // Re-throw the error
      throw error;
    }
  }
  
  /**
   * Make a streaming request to the VLLM server
   */
  async makeStreamRequestVLLM(
    endpointUrl: string,
    data: VLLMRequestData,
    options: VLLMRequestOptions = {}
  ): Promise<AsyncGenerator<VLLMResponse>> {
    const requestId = options.requestId || `vllm_stream_${Date.now()}`;
    const apiKey = options.apiKey || this.apiKey;
    
    // Set headers
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };
    
    // Add API key if available
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }
    
    // Add request tracking if enabled
    if (this.requestTracking) {
      headers['X-Request-ID'] = requestId;
    }
    
    // Apply request options to data and ensure streaming is enabled
    const requestData = { 
      ...data,
      stream: true
    };
    
    // Apply generation parameters if provided
    if (options.temperature !== undefined) requestData.temperature = options.temperature;
    if (options.top_p !== undefined) requestData.top_p = options.top_p;
    if (options.top_k !== undefined) requestData.top_k = options.top_k;
    if (options.max_tokens !== undefined) requestData.max_tokens = options.max_tokens;
    if (options.stop !== undefined) requestData.stop = options.stop;
    
    try {
      const response = await fetch(endpointUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify(requestData),
        signal: options.signal || AbortSignal.timeout(options.timeout || this.timeout)
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw this.createApiError(
          `VLLM API streaming error: ${errorText}`,
          response.status,
          'vllm_error'
        );
      }
      
      if (!response.body) {
        throw new Error('Response body is null');
      }
      
      // Return an async generator that yields each chunk of the response
      const stream = this.parseVLLMStream(response);
      return stream;
    } catch (error) {
      // Track the error if request tracking is enabled
      this.trackRequestResult(false, requestId, error as Error);
      
      // Re-throw the error
      throw error;
    }
  }
  
  /**
   * Parse VLLM streaming response
   */
  private async *parseVLLMStream(response: Response): AsyncGenerator<VLLMResponse> {
    if (!response.body) {
      throw new Error('Response body is null');
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        // Decode the response chunk
        const chunkText = decoder.decode(value, { stream: true });
        
        // VLLM returns each chunk as a separate JSON object
        // Split by lines and parse each line as JSON
        const lines = chunkText.split('\n').filter(line => line.trim() !== '');
        
        for (const line of lines) {
          try {
            // Remove "data: " prefix if present (SSE format)
            const jsonText = line.startsWith('data: ') ? line.slice(5) : line;
            
            // Special case for "[DONE]" marker
            if (jsonText.trim() === '[DONE]') {
              continue;
            }
            
            const chunk = JSON.parse(jsonText) as VLLMResponse;
            yield chunk;
          } catch (e) {
            console.error('Error parsing SSE chunk:', e);
            console.error('Chunk text:', line);
          }
        }
      }
    } catch (error) {
      // Release the reader if there's an error
      reader.cancel();
      throw error;
    } finally {
      // Ensure reader is released
      reader.releaseLock();
    }
  }
  
  /**
   * Get model information
   */
  async getModelInfo(model: string = this.model): Promise<VLLMModelInfo> {
    const url = `${this.apiUrl}/v1/models/${model}`;
    
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw this.createApiError(
          `Failed to get model info: ${response.statusText}`,
          response.status,
          'vllm_error'
        );
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to get model info:', error);
      throw error;
    }
  }
  
  /**
   * Get model statistics
   */
  async getModelStatistics(model: string = this.model): Promise<VLLMModelStatistics> {
    const url = `${this.apiUrl}/v1/models/${model}/statistics`;
    
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw this.createApiError(
          `Failed to get model statistics: ${response.statusText}`,
          response.status,
          'vllm_error'
        );
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to get model statistics:', error);
      throw error;
    }
  }
  
  /**
   * List LoRA adapters
   */
  async listLoraAdapters(): Promise<VLLMLoraAdapter[]> {
    const url = `${this.apiUrl}/v1/lora_adapters`;
    
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw this.createApiError(
          `Failed to list LoRA adapters: ${response.statusText}`,
          response.status,
          'vllm_error'
        );
      }
      
      const result = await response.json();
      return result.lora_adapters || [];
    } catch (error) {
      console.error('Failed to list LoRA adapters:', error);
      throw error;
    }
  }
  
  /**
   * Load LoRA adapter
   */
  async loadLoraAdapter(adapterData: {
    adapter_name: string;
    adapter_path: string;
    base_model: string;
  }): Promise<any> {
    const url = `${this.apiUrl}/v1/lora_adapters`;
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(adapterData)
      });
      
      if (!response.ok) {
        throw this.createApiError(
          `Failed to load LoRA adapter: ${response.statusText}`,
          response.status,
          'vllm_error'
        );
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to load LoRA adapter:', error);
      throw error;
    }
  }
  
  /**
   * Set quantization
   */
  async setQuantization(
    model: string = this.model,
    config: VLLMQuantizationConfig
  ): Promise<any> {
    const url = `${this.apiUrl}/v1/models/${model}/quantization`;
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(config)
      });
      
      if (!response.ok) {
        throw this.createApiError(
          `Failed to set quantization: ${response.statusText}`,
          response.status,
          'vllm_error'
        );
      }
      
      return await response.json();
    } catch (error) {
      console.error('Failed to set quantization:', error);
      throw error;
    }
  }
  
  /**
   * Process a batch of prompts
   */
  async processBatch(
    endpointUrl: string = `${this.apiUrl}/v1/completions`,
    batchData: string[],
    model: string = this.model,
    options: VLLMRequestOptions = {}
  ): Promise<string[]> {
    // VLLM can handle batched requests via the n parameter, but the request still contains one prompt
    // In case of truly different prompts, we need to make multiple requests
    // This is a convenience method for processing a batch of different prompts
    
    const results: string[] = [];
    
    // Make individual requests for each prompt in the batch
    for (const prompt of batchData) {
      const requestData: VLLMRequestData = {
        prompt,
        model,
        max_tokens: options.max_tokens || 100,
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        stop: options.stop
      };
      
      const response = await this.makePostRequestVLLM(endpointUrl, requestData, options);
      
      // Extract the generated text
      if (response.text) {
        results.push(response.text);
      } else if (response.choices && response.choices.length > 0) {
        results.push(response.choices[0].text || '');
      } else {
        results.push('');
      }
    }
    
    return results;
  }
  
  /**
   * Process a batch with metrics
   */
  async processBatchWithMetrics(
    endpointUrl: string = `${this.apiUrl}/v1/completions`,
    batchData: string[],
    model: string = this.model,
    options: VLLMRequestOptions = {}
  ): Promise<[string[], Record<string, any>]> {
    const startTime = Date.now();
    const results = await this.processBatch(endpointUrl, batchData, model, options);
    const endTime = Date.now();
    
    // Calculate metrics
    const metrics = {
      total_time_ms: endTime - startTime,
      average_time_per_item_ms: (endTime - startTime) / batchData.length,
      batch_size: batchData.length,
      successful_items: results.filter(r => r.length > 0).length
    };
    
    return [results, metrics];
  }
  
  /**
   * Stream generation
   */
  async *streamGeneration(
    endpointUrl: string = `${this.apiUrl}/v1/completions`,
    prompt: string,
    model: string = this.model,
    options: VLLMRequestOptions = {}
  ): AsyncGenerator<string> {
    const requestData: VLLMRequestData = {
      prompt,
      model,
      stream: true,
      max_tokens: options.max_tokens || 100,
      temperature: options.temperature,
      top_p: options.top_p,
      top_k: options.top_k,
      stop: options.stop
    };
    
    const stream = await this.makeStreamRequestVLLM(endpointUrl, requestData, options);
    
    let accumulatedText = '';
    for await (const chunk of stream) {
      if (chunk.choices && chunk.choices.length > 0) {
        // Extract delta text from the chunk
        const delta = chunk.choices[0].text || chunk.choices[0]?.delta?.content || '';
        accumulatedText += delta;
        yield delta;
      }
    }
  }
  
  /**
   * Check if a model is compatible with VLLM
   */
  isCompatibleModel(model: string): boolean {
    // VLLM supports most HuggingFace models, though this is a simplified check
    if (!model) return false;
    
    // Common model prefixes supported by VLLM
    const supportedPrefixes = [
      'meta-llama/',
      'mistralai/',
      'facebook/',
      'databricks/',
      'mosaicml/',
      'EleutherAI/',
      'bigscience/',
      'allenai/',
      'microsoft/',
      'together/'
    ];
    
    // Common model architectures supported by VLLM
    const supportedArchitectures = [
      'llama',
      'mistral',
      'mixtral',
      'falcon',
      'mpt',
      'opt',
      'bloom',
      'gpt-neox',
      'pythia',
      'cerebras',
      'stable',
      'dolly'
    ];
    
    // Check if model starts with a supported prefix
    const hasPrefix = supportedPrefixes.some(prefix => model.toLowerCase().startsWith(prefix.toLowerCase()));
    
    // Check if model contains a supported architecture
    const hasArchitecture = supportedArchitectures.some(arch => 
      model.toLowerCase().includes(arch.toLowerCase())
    );
    
    return hasPrefix || hasArchitecture;
  }
  
  /**
   * Required abstract method implementations
   */
  
  async makePostRequest(data: any, apiKey?: string, options?: ApiRequestOptions): Promise<any> {
    // Default endpoint based on data structure
    let endpointUrl = `${this.apiUrl}/v1/completions`;
    
    // Use chat endpoint if messages are present
    if (data.messages) {
      endpointUrl = `${this.apiUrl}/v1/chat/completions`;
    }
    
    // Allow overriding endpoint via options
    if (options?.endpoint) {
      endpointUrl = options.endpoint;
    }
    
    return this.makePostRequestVLLM(endpointUrl, data, { 
      ...options, 
      apiKey
    });
  }

  async *makeStreamRequest(data: any, options?: ApiRequestOptions): AsyncGenerator<StreamChunk> {
    // Default endpoint based on data structure
    let endpointUrl = `${this.apiUrl}/v1/completions`;
    
    // Use chat endpoint if messages are present
    if (data.messages) {
      endpointUrl = `${this.apiUrl}/v1/chat/completions`;
    }
    
    // Allow overriding endpoint via options
    if (options?.endpoint) {
      endpointUrl = options.endpoint;
    }
    
    // Ensure streaming is enabled
    const streamData = { ...data, stream: true };
    
    // Get the streaming response
    const stream = await this.makeStreamRequestVLLM(endpointUrl, streamData, options);
    
    // Process the stream and yield standard chunks
    for await (const chunk of stream) {
      // Check if this is a chat response or completion response
      if (chunk.choices && chunk.choices.length > 0) {
        const choice = chunk.choices[0];
        
        if (choice.delta) {
          // Chat format
          yield {
            content: choice.delta.content || '',
            role: choice.delta.role || 'assistant',
            type: 'token',
            done: choice.finish_reason !== null
          };
        } else {
          // Completion format
          yield {
            content: choice.text || '',
            type: 'token',
            done: choice.finish_reason !== null
          };
        }
      }
    }
    
    // Final chunk indicating completion
    yield {
      content: '',
      type: 'end',
      done: true
    };
  }

  /**
   * Convert messages to VLLM chat format and generate a response
   */
  async chat(messages: Message[], options?: ApiRequestOptions): Promise<ChatCompletionResponse> {
    const requestData: VLLMRequestData = {
      messages: messages.map(m => ({
        role: m.role,
        content: m.content
      })),
      model: options?.model || this.model,
      max_tokens: options?.max_tokens || 100,
      temperature: options?.temperature,
      top_p: options?.top_p,
      top_k: options?.top_k,
      stop: options?.stop
    };
    
    const endpointUrl = `${this.apiUrl}/v1/chat/completions`;
    const response = await this.makePostRequestVLLM(endpointUrl, requestData, options);
    
    // Extract response from the VLLM response format
    let content = '';
    let role = 'assistant';
    
    if (response.choices && response.choices.length > 0) {
      if (response.choices[0].message) {
        content = response.choices[0].message.content || '';
        role = response.choices[0].message.role || 'assistant';
      } else if (response.choices[0].text) {
        content = response.choices[0].text;
      }
    } else if (response.message) {
      content = response.message.content || '';
      role = response.message.role || 'assistant';
    } else if (response.text) {
      content = response.text;
    }
    
    // Format response
    return {
      content,
      role,
      model: options?.model || this.model,
      usage: response.usage
    };
  }

  /**
   * Stream chat completion
   */
  async *streamChat(messages: Message[], options?: ApiRequestOptions): AsyncGenerator<StreamChunk> {
    const requestData: VLLMRequestData = {
      messages: messages.map(m => ({
        role: m.role,
        content: m.content
      })),
      model: options?.model || this.model,
      max_tokens: options?.max_tokens || 100,
      temperature: options?.temperature,
      top_p: options?.top_p,
      top_k: options?.top_k,
      stop: options?.stop,
      stream: true
    };
    
    const endpointUrl = `${this.apiUrl}/v1/chat/completions`;
    const stream = await this.makeStreamRequestVLLM(endpointUrl, requestData, options);
    
    let accumulatedContent = '';
    
    for await (const chunk of stream) {
      if (chunk.choices && chunk.choices.length > 0) {
        const choice = chunk.choices[0];
        let content = '';
        let role = 'assistant';
        
        if (choice.delta) {
          // Standard delta format
          content = choice.delta.content || '';
          role = choice.delta.role || 'assistant';
        } else if (choice.message) {
          // Some implementations might return full message
          content = choice.message.content || '';
          role = choice.message.role || 'assistant';
        } else if (choice.text) {
          // Plain text format
          content = choice.text;
        }
        
        // Update accumulated content
        accumulatedContent += content;
        
        // Yield the chunk in standardized format
        yield {
          content,
          role,
          type: 'token',
          done: !!choice.finish_reason,
          finish_reason: choice.finish_reason
        };
      }
    }
    
    // Final chunk with complete response (may be useful for some consumers)
    yield {
      content: accumulatedContent,
      role: 'assistant',
      type: 'complete',
      done: true
    };
  }
}

// Default export
export default VLLM;