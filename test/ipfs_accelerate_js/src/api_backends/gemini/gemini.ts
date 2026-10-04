import { BaseApiBackend } from '../base';
import { ApiMetadata, ApiRequestOptions, Message, ChatCompletionResponse, StreamChunk } from '../types';
import { GeminiResponse, GeminiRequest } from './types';

export class Gemini extends BaseApiBackend {

  private apiEndpoint: string = 'https://generativelanguage.googleapis.com/v1/models';
  private apiVersion: string = 'v1';

  constructor(resources: Record<string, any> = {}, metadata: ApiMetadata = {}) {
    super(resources, metadata);
  }

  protected getApiKey(metadata: ApiMetadata): string {
    return metadata.gemini_api_key || 
           metadata.geminiApiKey || 
           (typeof process !== 'undefined' ? process.env.GEMINI_API_KEY || '' : '');
  }

  protected getDefaultModel(): string {
    return "gemini-pro";
  }

  isCompatibleModel(model: string): boolean {
    // Gemini models
    return (
      model.startsWith('gemini-') || 
      model.toLowerCase().includes('gemini') ||
      model.toLowerCase() === 'bison' ||
      model.toLowerCase().includes('palm')
    );
  }

  createEndpointHandler(): (data: any) => Promise<any> {
    return async (data: any) => {
      try {
        return await this.makePostRequest(data);
      } catch (error) {
        throw this.createApiError(`${this.constructor.name} endpoint error: ${error.message}`, 500);
      }
    };
  }

  async testEndpoint(): Promise<boolean> {
    try {
      const apiKey = this.getApiKey(this.metadata);
      if (!apiKey) {
        throw this.createApiError('API key is required', 401, 'authentication_error');
      }

      // Make a minimal request to verify the endpoint works
      const model = this.getDefaultModel();
      const testRequest = {
        contents: [{ 
          role: 'user',
          parts: [{ text: 'Hello' }]
        }],
        generationConfig: {
          maxOutputTokens: 5
        }
      };

      const endpoint = `${this.apiEndpoint}/${model}:generateContent?key=${apiKey}`;
      await this.makePostRequestDirect(endpoint, testRequest, { timeoutMs: 5000 });
      return true;
    } catch (error) {
      console.error(`${this.constructor.name} endpoint test failed:`, error);
      return false;
    }
  }

  private async makePostRequestDirect(endpoint: string, data: any, options?: ApiRequestOptions): Promise<any> {
    // Set up timeout
    const timeoutMs = options?.timeout || 30000;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      // Make the request
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data),
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
  }

  async makePostRequest(data: any, apiKey?: string, options?: ApiRequestOptions): Promise<any> {
    const key = apiKey || this.getApiKey(this.metadata);
    if (!key) {
      throw this.createApiError('API key is required', 401, 'authentication_error');
    }

    // Process with queue and circuit breaker
    return this.retryableRequest(async () => {
      const model = data.model || this.getDefaultModel();
      
      // Gemini API has a different structure - convert from standard if needed
      let geminiData: any;
      
      if (data.messages) {
        // Convert from standard chat format to Gemini format
        geminiData = {
          contents: data.messages.map((message: Message) => ({
            role: message.role === 'assistant' ? 'model' : message.role,
            parts: [{ text: message.content }]
          })),
          generationConfig: {
            maxOutputTokens: data.max_tokens,
            temperature: data.temperature,
            topP: data.top_p
          }
        };
      } else {
        geminiData = data;
      }
      
      // Gemini endpoint includes model and API key in URL
      const endpoint = `${this.apiEndpoint}/${model}:${data.stream ? 'streamGenerateContent' : 'generateContent'}?key=${key}`;
      return this.makePostRequestDirect(endpoint, geminiData, options);
    }, options?.maxRetries || this.maxRetries);
  }

  async *makeStreamRequest(data: any, options?: ApiRequestOptions): AsyncGenerator<StreamChunk> {
    const apiKey = this.getApiKey(this.metadata);
    if (!apiKey) {
      throw this.createApiError('API key is required', 401, 'authentication_error');
    }

    const model = data.model || this.getDefaultModel();
    
    // Convert standard format to Gemini format if needed
    let geminiData: any;
    
    if (data.messages) {
      // Convert from standard chat format to Gemini format
      geminiData = {
        contents: data.messages.map((message: Message) => ({
          role: message.role === 'assistant' ? 'model' : message.role,
          parts: [{ text: message.content }]
        })),
        generationConfig: {
          maxOutputTokens: data.max_tokens,
          temperature: data.temperature,
          topP: data.top_p
        }
      };
    } else {
      geminiData = data;
    }

    // Set up timeout
    const timeoutMs = options?.timeout || 30000;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      // Gemini stream endpoint
      const endpoint = `${this.apiEndpoint}/${model}:streamGenerateContent?key=${apiKey}`;
      
      // Make the request
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(geminiData),
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
            // Extract text content from Gemini's specific format
            const content = parsed.candidates?.[0]?.content?.parts?.[0]?.text || '';
            
            yield {
              content,
              type: 'delta'
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
          const content = parsed.candidates?.[0]?.content?.parts?.[0]?.text || '';
          
          yield {
            content,
            type: 'delta'
          };
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

  async chat(messages: Message[], options?: ApiRequestOptions): Promise<ChatCompletionResponse> {
    // Prepare request data
    const modelName = options?.model || this.getDefaultModel();
    
    // Convert to Gemini format
    const geminiData = {
      model: modelName,
      contents: messages.map((message: Message) => ({
        role: message.role === 'assistant' ? 'model' : message.role,
        parts: [{ text: message.content }]
      })),
      generationConfig: {
        maxOutputTokens: options?.maxTokens,
        temperature: options?.temperature,
        topP: options?.topP
      }
    };

    // Make the request
    const response = await this.makePostRequest(geminiData, undefined, options);

    // Convert Gemini response to standard format
    return {
      id: response.id || '',
      model: modelName,
      content: response.candidates?.[0]?.content?.parts?.[0]?.text || '',
      created: Date.now(),
      usage: { 
        prompt_tokens: response.usageMetadata?.promptTokenCount || 0, 
        completion_tokens: response.usageMetadata?.candidatesTokenCount || 0,
        total_tokens: (response.usageMetadata?.promptTokenCount || 0) + 
                     (response.usageMetadata?.candidatesTokenCount || 0)
      }
    };
  }

  async *streamChat(messages: Message[], options?: ApiRequestOptions): AsyncGenerator<StreamChunk> {
    // Prepare request data
    const modelName = options?.model || this.getDefaultModel();
    
    // Convert to Gemini format
    const geminiData = {
      model: modelName,
      contents: messages.map((message: Message) => ({
        role: message.role === 'assistant' ? 'model' : message.role,
        parts: [{ text: message.content }]
      })),
      generationConfig: {
        maxOutputTokens: options?.maxTokens,
        temperature: options?.temperature,
        topP: options?.topP
      }
    };

    // Make the stream request
    const stream = this.makeStreamRequest(geminiData, options);
    
    // Pass through the stream chunks
    yield* stream;
  }
}