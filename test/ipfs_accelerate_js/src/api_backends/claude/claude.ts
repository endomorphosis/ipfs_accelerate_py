/**
 * Claude (Anthropic) API Backend for IPFS Accelerate
 */

import { BaseApiBackend } from '../base';
import { 
  ApiResources, 
  ApiMetadata, 
  Message, 
  ChatCompletionResponse,
  StreamChunk,
  ApiRequestOptions,
  ApiError
} from '../types';

interface ClaudeMessage {
  role: string;
  content: string | any[];
}

interface ClaudeRequestData {
  model: string;
  messages: ClaudeMessage[];
  temperature?: number;
  max_tokens?: number;
  system?: string;
  stream?: boolean;
  [key: string]: any;
}

interface ClaudeContentItem {
  type: string;
  text?: string;
  [key: string]: any;
}

interface ClaudeResponse {
  id: string;
  type: string;
  role: string;
  content: ClaudeContentItem[];
  model: string;
  stop_reason: string | null;
  stop_sequence: string | null;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

interface ClaudeDeltaResponse {
  type: string;
  [key: string]: any;
}

export class Claude extends BaseApiBackend {
  private apiEndpoint: string = 'https://api.anthropic.com/v1/messages';
  private apiVersion: string = '2023-06-01';
  
  constructor(resources: ApiResources = {}, metadata: ApiMetadata = {}) {
    super(resources, metadata);
  }
  
  protected getApiKey(metadata: ApiMetadata): string {
    // Try to get from metadata, then from environment
    return metadata.claude_api_key || 
           metadata.claudeApiKey || 
           (typeof process !== 'undefined' ? process.env.ANTHROPIC_API_KEY || '' : '');
  }
  
  protected getDefaultModel(): string {
    return 'claude-3-haiku-20240307';
  }
  
  /**
   * Create an endpoint handler for Claude API
   */
  createEndpointHandler(): (data: any) => Promise<any> {
    // Return a function that can handle requests to the Claude API
    return async (data: any) => {
      try {
        // Validate input
        if (!data || !data.messages || !Array.isArray(data.messages)) {
          throw new Error('Invalid request data: messages array is required');
        }
        
        // Prepare request data
        const requestData: ClaudeRequestData = {
          model: data.model || this.model,
          messages: data.messages,
          temperature: data.temperature,
          max_tokens: data.max_tokens,
          system: data.system,
          stream: data.stream || false
        };
        
        // Make the request
        if (requestData.stream) {
          return this.makeStreamRequest(requestData);
        } else {
          return this.makePostRequest(requestData);
        }
      } catch (error) {
        console.error('Claude endpoint handler error:', error);
        throw error;
      }
    };
  }
  
  /**
   * Test the Claude API endpoint
   */
  async testEndpoint(): Promise<boolean> {
    try {
      // Simple message to test API connectivity
      const testMessage: ClaudeMessage[] = [
        { role: 'user', content: 'Hello, this is a test message. Please respond with a short greeting.' }
      ];
      
      // Prepare request with minimal tokens
      const requestData: ClaudeRequestData = {
        model: this.model,
        messages: testMessage,
        max_tokens: 10
      };
      
      // Make the request with retries
      const response = await this.retryableRequest(() => 
        this.makePostRequest(requestData, this.apiKey, { timeout: 5000 })
      );
      
      return !!response && !!response.content;
    } catch (error) {
      console.error('Claude API test failed:', error);
      return false;
    }
  }
  
  /**
   * Make a POST request to the Claude API
   */
  async makePostRequest(
    data: any, 
    apiKey: string = this.apiKey, 
    options: ApiRequestOptions = {}
  ): Promise<ClaudeResponse> {
    if (!apiKey) {
      throw this.createApiError('API key is required', 401, 'authentication_error');
    }
    
    const controller = new AbortController();
    const { signal } = controller;
    
    // Set timeout if specified
    const timeoutId = options.timeout ? 
      setTimeout(() => controller.abort(), options.timeout) : 
      null;
    
    try {
      const response = await fetch(this.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': this.apiVersion,
          'anthropic-beta': 'messages-2023-12-15'
        },
        body: JSON.stringify(data),
        signal
      });
      
      // Clear timeout if set
      if (timeoutId) clearTimeout(timeoutId);
      
      // Handle HTTP errors
      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        const message = errorBody.error?.message || `HTTP error ${response.status}`;
        
        throw this.createApiError(
          message,
          response.status,
          errorBody.error?.type || 'api_error'
        );
      }
      
      // Parse successful response
      const result = await response.json();
      return result as ClaudeResponse;
    } catch (error) {
      // Clear timeout if set
      if (timeoutId) clearTimeout(timeoutId);
      
      // Handle AbortError specially
      if (error instanceof DOMException && error.name === 'AbortError') {
        const timeoutError = this.createApiError(
          'Request timed out',
          408,
          'timeout_error'
        );
        timeoutError.isTimeout = true;
        throw timeoutError;
      }
      
      // Re-throw API errors
      if ((error as ApiError).statusCode) {
        throw error;
      }
      
      // Wrap other errors
      throw this.createApiError(
        `Request failed: ${error instanceof Error ? error.message : String(error)}`,
        500,
        'request_error'
      );
    }
  }
  
  /**
   * Make a streaming request to the Claude API
   */
  async *makeStreamRequest(
    data: any, 
    options: ApiRequestOptions = {}
  ): AsyncGenerator<StreamChunk> {
    if (!this.apiKey) {
      throw this.createApiError('API key is required', 401, 'authentication_error');
    }
    
    // Ensure stream is enabled
    const streamData = { ...data, stream: true };
    
    const controller = new AbortController();
    const { signal } = controller;
    
    // Set timeout if specified
    const timeoutId = options.timeout ? 
      setTimeout(() => controller.abort(), options.timeout) : 
      null;
    
    try {
      const response = await fetch(this.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': this.apiKey,
          'anthropic-version': this.apiVersion,
          'anthropic-beta': 'messages-2023-12-15'
        },
        body: JSON.stringify(streamData),
        signal
      });
      
      // Handle HTTP errors
      if (!response.ok) {
        if (timeoutId) clearTimeout(timeoutId);
        
        const errorBody = await response.json().catch(() => ({}));
        const message = errorBody.error?.message || `HTTP error ${response.status}`;
        
        throw this.createApiError(
          message,
          response.status,
          errorBody.error?.type || 'api_error'
        );
      }
      
      // Process the stream
      if (!response.body) {
        if (timeoutId) clearTimeout(timeoutId);
        throw this.createApiError('Response body is null', 500, 'stream_error');
      }
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          buffer += decoder.decode(value, { stream: true });
          
          // Process any complete events (separated by double newlines)
          const lines = buffer.split('\n\n');
          buffer = lines.pop() || ''; // Keep the last incomplete chunk in the buffer
          
          for (const line of lines) {
            if (!line.trim() || !line.startsWith('data: ')) continue;
            
            try {
              const eventData = JSON.parse(line.substring(6));
              
              if (eventData.type === 'content_block_delta') {
                yield {
                  content: eventData.delta.text,
                  type: 'delta'
                };
              } else if (eventData.type === 'message_start') {
                yield {
                  role: 'assistant',
                  type: 'start'
                };
              } else if (eventData.type === 'message_stop') {
                yield {
                  type: 'stop'
                };
              }
            } catch (e) {
              console.warn('Error parsing Claude stream data:', e);
              continue;
            }
          }
        }
        
        // Process any remaining data
        const remainingText = decoder.decode();
        if (remainingText && remainingText.startsWith('data: ')) {
          try {
            const eventData = JSON.parse(remainingText.substring(6));
            
            if (eventData.type === 'content_block_delta') {
              yield {
                content: eventData.delta.text,
                type: 'delta'
              };
            }
          } catch (e) {
            console.warn('Error parsing Claude final stream data:', e);
          }
        }
      } finally {
        // Clear timeout if set
        if (timeoutId) clearTimeout(timeoutId);
      }
    } catch (error) {
      // Clear timeout if set
      if (timeoutId) clearTimeout(timeoutId);
      
      // Handle AbortError specially
      if (error instanceof DOMException && error.name === 'AbortError') {
        const timeoutError = this.createApiError(
          'Request timed out',
          408,
          'timeout_error'
        );
        timeoutError.isTimeout = true;
        throw timeoutError;
      }
      
      // Re-throw API errors
      if ((error as ApiError).statusCode) {
        throw error;
      }
      
      // Wrap other errors
      throw this.createApiError(
        `Stream request failed: ${error instanceof Error ? error.message : String(error)}`,
        500,
        'stream_error'
      );
    }
  }
  
  /**
   * Generate a chat completion
   */
  async chat(messages: Message[], options: ApiRequestOptions = {}): Promise<ChatCompletionResponse> {
    const requestData: ClaudeRequestData = {
      model: options.model as string || this.model,
      messages: messages as ClaudeMessage[],
      temperature: options.temperature as number,
      max_tokens: options.max_tokens as number,
      system: options.system as string,
      stream: false
    };
    
    const response = await this.makePostRequest(requestData, options.apiKey as string, options);
    
    return {
      id: response.id,
      content: response.content,
      role: response.role,
      model: response.model,
      usage: {
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens
      }
    };
  }
  
  /**
   * Generate a streaming chat completion
   */
  async *streamChat(messages: Message[], options: ApiRequestOptions = {}): AsyncGenerator<StreamChunk> {
    const requestData: ClaudeRequestData = {
      model: options.model as string || this.model,
      messages: messages as ClaudeMessage[],
      temperature: options.temperature as number,
      max_tokens: options.max_tokens as number,
      system: options.system as string,
      stream: true
    };
    
    for await (const chunk of this.makeStreamRequest(requestData, options)) {
      yield chunk;
    }
  }
  
  /**
   * Check if a model is compatible with the Claude API
   */
  isCompatibleModel(model: string): boolean {
    const lowerModel = model.toLowerCase();
    return lowerModel.includes('claude');
  }
}