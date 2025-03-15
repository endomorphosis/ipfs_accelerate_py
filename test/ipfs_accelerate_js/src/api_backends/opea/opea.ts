/**
 * OpenAI Proxy API Extension (OPEA) Backend
 * 
 * This backend provides a compatible interface for self-hosted OpenAI-compatible APIs
 * including those running behind custom proxies.
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
  OPEARequestData, 
  OPEAResponse, 
  OPEAStreamChunk,
  OPEARequestOptions
} from './types';

export class OPEA extends BaseApiBackend {
  protected apiUrl: string;
  protected defaultModel: string;
  
  constructor(resources: ApiResources = {}, metadata: ApiMetadata = {}) {
    super(resources, metadata);
    
    // Initialize OPEA-specific properties
    this.apiUrl = metadata.opea_api_url || process.env.OPEA_API_URL || 'http://localhost:8000';
    this.defaultModel = metadata.opea_model || process.env.OPEA_MODEL || 'gpt-3.5-turbo';
    this.model = this.defaultModel;
    
    // Set timeout for OPEA requests
    this.timeout = metadata.timeout || parseInt(process.env.OPEA_TIMEOUT || '30') * 1000;
  }
  
  /**
   * Get API key
   */
  protected getApiKey(metadata: ApiMetadata): string {
    return metadata.opea_api_key || process.env.OPEA_API_KEY || '';
  }
  
  /**
   * Get the default model
   */
  protected getDefaultModel(): string {
    return 'gpt-3.5-turbo';
  }
  
  /**
   * Create an endpoint handler for OPEA
   */
  createEndpointHandler(
    endpointUrl: string = `${this.apiUrl}/v1/chat/completions`
  ): (data: OPEARequestData) => Promise<OPEAResponse> {
    return async (data: OPEARequestData): Promise<OPEAResponse> => {
      return this.makePostRequestOPEA(endpointUrl, data);
    };
  }
  
  /**
   * Test the OPEA endpoint
   */
  async testEndpoint(
    endpointUrl: string = `${this.apiUrl}/v1/chat/completions`
  ): Promise<boolean> {
    try {
      // Create a simple test message
      const testData: OPEARequestData = {
        messages: [
          { role: "user", content: "Hello" }
        ],
        max_tokens: 5,
        model: this.model
      };
      
      const response = await this.makePostRequestOPEA(endpointUrl, testData);
      return !!response && 
        (!!response.choices && response.choices.length > 0) && 
        (!!response.choices[0].message || !!response.choices[0].text);
    } catch (error) {
      console.error('OPEA endpoint test failed:', error);
      return false;
    }
  }
  
  /**
   * Make a POST request to the OPEA server
   */
  async makePostRequestOPEA(
    endpointUrl: string,
    data: OPEARequestData,
    options: OPEARequestOptions = {}
  ): Promise<OPEAResponse> {
    const requestId = options.requestId || `opea_${Date.now()}`;
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
            `OPEA API error: ${errorText}`,
            response.status,
            'opea_error'
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
   * Make a streaming request to the OPEA server
   */
  async makeStreamRequestOPEA(
    endpointUrl: string,
    data: OPEARequestData,
    options: OPEARequestOptions = {}
  ): Promise<AsyncGenerator<OPEAStreamChunk>> {
    const requestId = options.requestId || `opea_stream_${Date.now()}`;
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
          `OPEA API streaming error: ${errorText}`,
          response.status,
          'opea_error'
        );
      }
      
      if (!response.body) {
        throw new Error('Response body is null');
      }
      
      // Return an async generator that yields each chunk of the response
      const stream = this.parseOPEAStream(response);
      return stream;
    } catch (error) {
      // Track the error if request tracking is enabled
      this.trackRequestResult(false, requestId, error as Error);
      
      // Re-throw the error
      throw error;
    }
  }
  
  /**
   * Parse OPEA streaming response
   */
  private async *parseOPEAStream(response: Response): AsyncGenerator<OPEAStreamChunk> {
    if (!response.body) {
      throw new Error('Response body is null');
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        // Decode the response chunk
        const chunkText = decoder.decode(value, { stream: true });
        buffer += chunkText;
        
        // Process each line (each chunk may contain multiple data lines)
        let lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep the last line which might be incomplete
        
        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine) continue; // Skip empty lines
          
          // Handle SSE format (lines starting with "data: ")
          if (trimmedLine.startsWith('data: ')) {
            const jsonText = trimmedLine.slice(5).trim();
            
            // Special case for "[DONE]" marker
            if (jsonText === '[DONE]') {
              continue;
            }
            
            try {
              const chunk = JSON.parse(jsonText) as OPEAStreamChunk;
              yield chunk;
            } catch (e) {
              console.error('Error parsing SSE chunk:', e);
              console.error('Chunk text:', jsonText);
            }
          }
        }
      }
      
      // Process any remaining buffer
      if (buffer.trim()) {
        const lines = buffer.split('\n');
        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine || !trimmedLine.startsWith('data: ')) continue;
          
          const jsonText = trimmedLine.slice(5).trim();
          if (jsonText === '[DONE]') continue;
          
          try {
            const chunk = JSON.parse(jsonText) as OPEAStreamChunk;
            yield chunk;
          } catch (e) {
            console.error('Error parsing final SSE chunk:', e);
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
   * Check if a model is compatible with OPEA
   */
  isCompatibleModel(model: string): boolean {
    if (!model) return false;
    
    // OPEA typically supports OpenAI-compatible models
    const supportedPrefixes = [
      'gpt-',
      'text-',
      'claude-',
      'llama-',
      'mistral-',
      'mixtral-'
    ];
    
    // Check if model starts with a supported prefix
    return supportedPrefixes.some(prefix => model.toLowerCase().startsWith(prefix.toLowerCase()));
  }
  
  /**
   * Required abstract method implementations
   */
  
  async makePostRequest(data: any, apiKey?: string, options?: ApiRequestOptions): Promise<any> {
    // Default endpoint based on data structure
    let endpointUrl = `${this.apiUrl}/v1/chat/completions`;
    
    // Allow overriding endpoint via options
    if (options?.endpoint) {
      endpointUrl = options.endpoint;
    }
    
    return this.makePostRequestOPEA(endpointUrl, data, { 
      ...options, 
      apiKey
    });
  }

  async *makeStreamRequest(data: any, options?: ApiRequestOptions): AsyncGenerator<StreamChunk> {
    // Default endpoint is chat completions for OPEA
    let endpointUrl = `${this.apiUrl}/v1/chat/completions`;
    
    // Allow overriding endpoint via options
    if (options?.endpoint) {
      endpointUrl = options.endpoint;
    }
    
    // Ensure streaming is enabled
    const streamData = { ...data, stream: true };
    
    // Get the streaming response
    const stream = await this.makeStreamRequestOPEA(endpointUrl, streamData, options);
    
    // Process the stream and yield standard chunks
    for await (const chunk of stream) {
      if (chunk.choices && chunk.choices.length > 0) {
        const choice = chunk.choices[0];
        
        if (choice.delta) {
          // OpenAI-compatible delta format
          yield {
            content: choice.delta.content || '',
            role: choice.delta.role || 'assistant',
            type: 'token',
            done: !!choice.finish_reason
          };
        } else {
          // Some implementations might not use delta format
          yield {
            content: '',
            type: 'token',
            done: !!choice.finish_reason
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

  async chat(messages: Message[], options?: ApiRequestOptions): Promise<ChatCompletionResponse> {
    const requestData: OPEARequestData = {
      messages: messages.map(m => ({
        role: m.role,
        content: m.content
      })),
      model: options?.model || this.model,
      max_tokens: options?.max_tokens || 100,
      temperature: options?.temperature,
      top_p: options?.top_p,
      stop: options?.stop
    };
    
    const endpointUrl = `${this.apiUrl}/v1/chat/completions`;
    const response = await this.makePostRequestOPEA(endpointUrl, requestData, options);
    
    // Extract response from the OPEA response format
    let content = '';
    let role = 'assistant';
    
    if (response.choices && response.choices.length > 0) {
      if (response.choices[0].message) {
        content = response.choices[0].message.content || '';
        role = response.choices[0].message.role || 'assistant';
      } else if (response.choices[0].text) {
        content = response.choices[0].text;
      }
    }
    
    // Format response
    return {
      content,
      role,
      model: options?.model || this.model,
      usage: response.usage
    };
  }

  async *streamChat(messages: Message[], options?: ApiRequestOptions): AsyncGenerator<StreamChunk> {
    const requestData: OPEARequestData = {
      messages: messages.map(m => ({
        role: m.role,
        content: m.content
      })),
      model: options?.model || this.model,
      max_tokens: options?.max_tokens || 100,
      temperature: options?.temperature,
      top_p: options?.top_p,
      stop: options?.stop,
      stream: true
    };
    
    const endpointUrl = `${this.apiUrl}/v1/chat/completions`;
    const stream = await this.makeStreamRequestOPEA(endpointUrl, requestData, options);
    
    let accumulatedContent = '';
    
    for await (const chunk of stream) {
      if (chunk.choices && chunk.choices.length > 0) {
        const choice = chunk.choices[0];
        
        if (choice.delta) {
          const content = choice.delta.content || '';
          const role = choice.delta.role || 'assistant';
          
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
    }
    
    // Final chunk with complete response
    yield {
      content: accumulatedContent,
      role: 'assistant',
      type: 'complete',
      done: true
    };
  }
}

// Default export
export default OPEA;