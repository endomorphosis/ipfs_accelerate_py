import { BaseApiBackend } from '../base';
import { ApiMetadata, ApiRequestOptions, Message, ChatCompletionResponse, StreamChunk } from '../types';
import { GroqResponse, GroqRequest } from './types';

export class Groq extends BaseApiBackend {

  private apiEndpoint: string = 'https://api.groq.com/openai/v1/chat/completions';

  constructor(resources: Record<string, any> = {}, metadata: ApiMetadata = {}) {
    super(resources, metadata);
  }

  protected getApiKey(metadata: ApiMetadata): string {
    return metadata.groq_api_key || 
           metadata.groqApiKey || 
           (typeof process !== 'undefined' ? process.env.GROQ_API_KEY || '' : '');
  }

  protected getDefaultModel(): string {
    return "llama2-70b-4096";
  }

  isCompatibleModel(model: string): boolean {
    // Groq models - typically using either OpenAI compatible models or Llama variants
    return (
      model.startsWith('llama') || 
      model.startsWith('mistral') || 
      model.startsWith('mixtral') || 
      model.toLowerCase().includes('groq')
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
      const testRequest = {
        model: this.getDefaultModel(),
        messages: [{ role: 'user', content: 'Hello' }],
        max_tokens: 5
      };

      await this.makePostRequest(testRequest, apiKey, { timeoutMs: 5000 });
      return true;
    } catch (error) {
      console.error(`${this.constructor.name} endpoint test failed:`, error);
      return false;
    }
  }

  async makePostRequest(data: any, apiKey?: string, options?: ApiRequestOptions): Promise<any> {
    const key = apiKey || this.getApiKey(this.metadata);
    if (!key) {
      throw this.createApiError('API key is required', 401, 'authentication_error');
    }

    // Process with queue and circuit breaker
    return this.retryableRequest(async () => {
      // Prepare request headers
      const headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${key}`
      };

      // Prepare request body
      const requestBody = JSON.stringify(data);

      // Set up timeout
      const timeoutMs = options?.timeout || 30000;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

      try {
        // Make the request
        const response = await fetch(this.apiEndpoint, {
          method: 'POST',
          headers,
          body: requestBody,
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
    }, options?.maxRetries || this.maxRetries);
  }

  async *makeStreamRequest(data: any, options?: ApiRequestOptions): AsyncGenerator<StreamChunk> {
    const apiKey = this.getApiKey(this.metadata);
    if (!apiKey) {
      throw this.createApiError('API key is required', 401, 'authentication_error');
    }

    // Ensure stream option is set
    const streamData = { ...data, stream: true };

    // Prepare request headers
    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    };

    // Prepare request body
    const requestBody = JSON.stringify(streamData);

    // Set up timeout
    const timeoutMs = options?.timeout || 30000;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      // Make the request
      const response = await fetch(this.apiEndpoint, {
        method: 'POST',
        headers,
        body: requestBody,
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
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;

            try {
              const parsed = JSON.parse(data);
              yield {
                content: parsed.choices?.[0]?.delta?.content || '',
                type: 'delta'
              };
            } catch (e) {
              console.warn('Failed to parse stream data:', data);
            }
          }
        }
      }

      // Handle any remaining data in the buffer
      if (buffer.trim() !== '') {
        if (buffer.startsWith('data: ') && buffer !== 'data: [DONE]') {
          try {
            const data = buffer.slice(6);
            const parsed = JSON.parse(data);
            yield {
              content: parsed.choices?.[0]?.delta?.content || '',
              type: 'delta'
            };
          } catch (e) {
            console.warn('Failed to parse final stream data:', buffer);
          }
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
    
    const requestData: GroqRequest = {
      model: modelName,
      messages,
      max_tokens: options?.maxTokens,
      temperature: options?.temperature,
      top_p: options?.topP
    };

    // Make the request
    const response = await this.makePostRequest(requestData, undefined, options);

    // Convert to standard format
    return {
      id: response.id || '',
      model: response.model || modelName,
      content: response.choices?.[0]?.message?.content || '',
      created: response.created || Date.now(),
      usage: response.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    };
  }

  async *streamChat(messages: Message[], options?: ApiRequestOptions): AsyncGenerator<StreamChunk> {
    // Prepare request data
    const modelName = options?.model || this.getDefaultModel();
    
    const requestData: GroqRequest = {
      model: modelName,
      messages,
      stream: true,
      max_tokens: options?.maxTokens,
      temperature: options?.temperature,
      top_p: options?.topP
    };

    // Make the stream request
    const stream = this.makeStreamRequest(requestData, options);
    
    // Pass through the stream chunks
    yield* stream;
  }
}