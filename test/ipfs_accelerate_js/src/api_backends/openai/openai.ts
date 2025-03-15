import { BaseApiBackend } from '../base';
import { ApiMetadata, ApiRequestOptions, Message, ChatCompletionResponse, StreamChunk } from '../types';
import { 
  OpenAIResponse, 
  OpenAIRequest, 
  OpenAIEmbeddingRequest,
  OpenAIEmbeddingResponse,
  OpenAIModerationRequest,
  OpenAIModerationResponse,
  OpenAIImageRequest,
  OpenAIImageResponse,
  OpenAISpeechRequest,
  OpenAITranscriptionRequest,
  OpenAITranscriptionResponse
} from './types';

export class OpenAI extends BaseApiBackend {
  private apiEndpoints: Record<string, string> = {
    chat: 'https://api.openai.com/v1/chat/completions',
    embeddings: 'https://api.openai.com/v1/embeddings',
    moderations: 'https://api.openai.com/v1/moderations',
    images: 'https://api.openai.com/v1/images/generations',
    speech: 'https://api.openai.com/v1/audio/speech',
    transcriptions: 'https://api.openai.com/v1/audio/transcriptions'
  };

  constructor(resources: Record<string, any> = {}, metadata: ApiMetadata = {}) {
    super(resources, metadata);
  }

  protected getApiKey(metadata: ApiMetadata): string {
    return metadata.openai_api_key || 
           metadata.openaiApiKey || 
           (typeof process !== 'undefined' ? process.env.OPENAI_API_KEY || '' : '');
  }

  protected getDefaultModel(): string {
    return "gpt-4o";
  }

  isCompatibleModel(model: string): boolean {
    // OpenAI models
    return (
      model.startsWith('gpt-') || 
      model.startsWith('text-embedding-') || 
      model === 'text-moderation-latest' ||
      model.startsWith('dall-e-') ||
      model.startsWith('tts-') ||
      model === 'whisper-1'
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
        model: 'text-embedding-3-small',
        input: 'Hello'
      };

      await this.makePostRequest(testRequest, apiKey, { 
        timeout: 5000,
        endpoint: this.apiEndpoints.embeddings 
      });
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

    // Determine the API endpoint to use
    const endpoint = options?.endpoint || this.apiEndpoints.chat;

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
        const response = await fetch(endpoint, {
          method: 'POST',
          headers,
          body: requestBody,
          signal: controller.signal
        });

        // Check for errors
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          const errorMessage = errorData.error?.message || `HTTP error ${response.status}`;
          const errorType = errorData.error?.type || 'api_error';
          
          // Handle rate limiting
          if (response.status === 429) {
            const retryAfter = response.headers.get('retry-after');
            const error = this.createApiError(errorMessage, response.status, errorType);
            error.retryAfter = retryAfter ? parseInt(retryAfter, 10) : 1;
            error.isRateLimitError = true;
            throw error;
          }
          
          throw this.createApiError(errorMessage, response.status, errorType);
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

  async makeFormDataRequest(formData: FormData, apiKey?: string, options?: ApiRequestOptions): Promise<any> {
    const key = apiKey || this.getApiKey(this.metadata);
    if (!key) {
      throw this.createApiError('API key is required', 401, 'authentication_error');
    }

    // Determine the API endpoint to use
    const endpoint = options?.endpoint || this.apiEndpoints.transcriptions;

    // Process with queue and circuit breaker
    return this.retryableRequest(async () => {
      // Prepare request headers
      const headers = {
        'Authorization': `Bearer ${key}`
      };

      // Set up timeout
      const timeoutMs = options?.timeout || 30000;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

      try {
        // Make the request
        const response = await fetch(endpoint, {
          method: 'POST',
          headers,
          body: formData,
          signal: controller.signal
        });

        // Check for errors
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          const errorMessage = errorData.error?.message || `HTTP error ${response.status}`;
          const errorType = errorData.error?.type || 'api_error';
          
          // Handle rate limiting
          if (response.status === 429) {
            const retryAfter = response.headers.get('retry-after');
            const error = this.createApiError(errorMessage, response.status, errorType);
            error.retryAfter = retryAfter ? parseInt(retryAfter, 10) : 1;
            error.isRateLimitError = true;
            throw error;
          }
          
          throw this.createApiError(errorMessage, response.status, errorType);
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
    const timeoutMs = options?.timeout || 60000;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      // Make the request
      const response = await fetch(this.apiEndpoints.chat, {
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
              
              // Check for tool calls in streaming response
              if (parsed.choices?.[0]?.delta?.tool_calls) {
                const toolCall = parsed.choices[0].delta.tool_calls[0];
                yield {
                  content: '',
                  type: 'tool_call',
                  tool_call: toolCall
                };
              } else {
                yield {
                  content: parsed.choices?.[0]?.delta?.content || '',
                  role: parsed.choices?.[0]?.delta?.role,
                  type: 'delta'
                };
              }
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
            
            if (parsed.choices?.[0]?.delta?.tool_calls) {
              const toolCall = parsed.choices[0].delta.tool_calls[0];
              yield {
                content: '',
                type: 'tool_call',
                tool_call: toolCall
              };
            } else {
              yield {
                content: parsed.choices?.[0]?.delta?.content || '',
                role: parsed.choices?.[0]?.delta?.role,
                type: 'delta'
              };
            }
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
    
    const requestData: OpenAIRequest = {
      model: modelName,
      messages,
      max_tokens: options?.maxTokens,
      temperature: options?.temperature,
      top_p: options?.topP
    };

    // Add function calling if provided
    if (options?.functions) {
      requestData.functions = options.functions;
    }

    if (options?.tools) {
      requestData.tools = options.tools;
    }

    if (options?.toolChoice) {
      requestData.tool_choice = options.toolChoice;
    }

    // Make the request
    const response = await this.makePostRequest(requestData, undefined, options);

    // Convert to standard format
    return {
      id: response.id || '',
      model: response.model || modelName,
      content: response.choices?.[0]?.message?.content || '',
      role: response.choices?.[0]?.message?.role || 'assistant',
      created: response.created || Date.now(),
      tool_calls: response.choices?.[0]?.message?.tool_calls,
      usage: response.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    };
  }

  async *streamChat(messages: Message[], options?: ApiRequestOptions): AsyncGenerator<StreamChunk> {
    // Prepare request data
    const modelName = options?.model || this.getDefaultModel();
    
    const requestData: OpenAIRequest = {
      model: modelName,
      messages,
      stream: true,
      max_tokens: options?.maxTokens,
      temperature: options?.temperature,
      top_p: options?.topP
    };

    // Add function calling if provided
    if (options?.functions) {
      requestData.functions = options.functions;
    }

    if (options?.tools) {
      requestData.tools = options.tools;
    }

    if (options?.toolChoice) {
      requestData.tool_choice = options.toolChoice;
    }

    // Make the stream request
    const stream = this.makeStreamRequest(requestData, options);
    
    // Pass through the stream chunks
    yield* stream;
  }

  async embedding(text: string | string[], options?: ApiRequestOptions): Promise<number[][]> {
    const modelName = options?.model || 'text-embedding-3-small';
    
    const requestData: OpenAIEmbeddingRequest = {
      model: modelName,
      input: text,
      encoding_format: 'float'
    };

    const response = await this.makePostRequest(
      requestData, 
      undefined, 
      { ...options, endpoint: this.apiEndpoints.embeddings }
    ) as OpenAIEmbeddingResponse;

    // Extract embeddings from response
    const embeddings = response.data.map(item => item.embedding);
    return embeddings;
  }

  async moderation(text: string, options?: ApiRequestOptions): Promise<OpenAIModerationResponse> {
    const modelName = options?.model || 'text-moderation-latest';
    
    const requestData: OpenAIModerationRequest = {
      input: text,
      model: modelName
    };

    return await this.makePostRequest(
      requestData,
      undefined,
      { ...options, endpoint: this.apiEndpoints.moderations }
    ) as OpenAIModerationResponse;
  }

  async textToImage(prompt: string, options?: ApiRequestOptions): Promise<OpenAIImageResponse> {
    const modelName = options?.model || 'dall-e-3';
    
    const requestData: OpenAIImageRequest = {
      model: modelName,
      prompt,
      n: options?.n || 1,
      size: options?.size || '1024x1024',
      style: options?.style || 'vivid',
      quality: options?.quality || 'standard',
      response_format: options?.responseFormat || 'url'
    };

    return await this.makePostRequest(
      requestData,
      undefined,
      { ...options, endpoint: this.apiEndpoints.images }
    ) as OpenAIImageResponse;
  }

  async textToSpeech(text: string, voice: string, options?: ApiRequestOptions): Promise<ArrayBuffer> {
    const modelName = options?.model || 'tts-1';
    
    const requestData: OpenAISpeechRequest = {
      model: modelName,
      input: text,
      voice,
      response_format: options?.responseFormat || 'mp3',
      speed: options?.speed || 1.0
    };

    const response = await fetch(this.apiEndpoints.speech, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify(requestData)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw this.createApiError(
        errorData.error?.message || `HTTP error ${response.status}`,
        response.status,
        errorData.error?.type || 'api_error'
      );
    }

    return await response.arrayBuffer();
  }

  async speechToText(audioData: Blob, options?: ApiRequestOptions): Promise<OpenAITranscriptionResponse> {
    const modelName = options?.model || 'whisper-1';
    
    const formData = new FormData();
    formData.append('file', audioData);
    formData.append('model', modelName);
    
    if (options?.language) {
      formData.append('language', options.language);
    }
    
    if (options?.prompt) {
      formData.append('prompt', options.prompt);
    }
    
    if (options?.responseFormat) {
      formData.append('response_format', options.responseFormat);
    }
    
    if (options?.temperature) {
      formData.append('temperature', options.temperature.toString());
    }

    return await this.makeFormDataRequest(
      formData,
      undefined,
      { ...options, endpoint: this.apiEndpoints.transcriptions }
    ) as OpenAITranscriptionResponse;
  }

  // Helper method to determine best model to use
  private determineModel(modelType: string, requestedModel?: string): string {
    if (requestedModel) {
      return requestedModel;
    }

    // Default models per type
    const models: Record<string, string> = {
      'chat': 'gpt-4o',
      'embedding': 'text-embedding-3-small',
      'moderation': 'text-moderation-latest',
      'image': 'dall-e-3',
      'speech': 'tts-1',
      'transcription': 'whisper-1'
    };

    return models[modelType] || this.getDefaultModel();
  }

  // Helper method to process messages and add system message if needed
  processMessages(messages: Message[], systemMessage?: string): Message[] {
    // If there's already a system message, don't add another one
    const hasSystemMessage = messages.some(msg => msg.role === 'system');
    
    if (!hasSystemMessage && systemMessage) {
      return [
        { role: 'system', content: systemMessage },
        ...messages
      ];
    }
    
    return messages;
  }
}