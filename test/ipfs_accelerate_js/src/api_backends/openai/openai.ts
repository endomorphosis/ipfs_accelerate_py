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
    transcriptions: 'https://api.openai.com/v1/audio/transcriptions',
    translations: 'https://api.openai.com/v1/audio/translations'
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
  
  /**
   * Enhanced metrics collection for API requests
   * @param requestType Type of request (chat, embedding, speech, etc.)
   * @param startTime Request start timestamp
   * @param endTime Request end timestamp
   * @param success Whether the request was successful
   * @param model The model used for the request
   * @param tokenCount Token usage if available
   * @param options Additional metrics options
   */
  trackRequestMetrics(
    requestType: string, 
    startTime: number, 
    endTime: number, 
    success: boolean, 
    model?: string,
    tokenCount?: { input?: number, output?: number, total?: number },
    options?: { 
      errorType?: string, 
      retryCount?: number, 
      requestSize?: number, 
      responseSize?: number
    }
  ): void {
    // Only track if metrics are enabled in parent class
    if (!this.requestTracking) return;
    
    const duration = endTime - startTime;
    const requestId = `${requestType}_${startTime}`;
    
    // Basic metrics that are always tracked
    const metrics = {
      requestType,
      timestamp: startTime,
      duration,
      success,
      model: model || this.getDefaultModel(),
    };
    
    // Add token usage if available
    if (tokenCount) {
      Object.assign(metrics, {
        inputTokens: tokenCount.input,
        outputTokens: tokenCount.output,
        totalTokens: tokenCount.total
      });
    }
    
    // Add additional metrics if provided
    if (options) {
      Object.assign(metrics, {
        errorType: options.errorType,
        retryCount: options.retryCount,
        requestSize: options.requestSize,
        responseSize: options.responseSize
      });
    }
    
    // Store metrics
    this.recentRequests[requestId] = metrics;
    
    // Clean up old metrics (older than 1 hour)
    this.cleanupOldMetrics();
    
    // If distributed testing framework is available, report metrics
    this.reportMetricsToDistributedFramework(requestId, metrics);
  }
  
  /**
   * Clean up old metrics to prevent memory leaks
   * @param maxAgeMs Maximum age in milliseconds (default: 1 hour)
   */
  private cleanupOldMetrics(maxAgeMs: number = 3600000): void {
    const now = Date.now();
    for (const id in this.recentRequests) {
      if (this.recentRequests[id].timestamp && 
          now - this.recentRequests[id].timestamp > maxAgeMs) {
        delete this.recentRequests[id];
      }
    }
  }
  
  /**
   * Report metrics to distributed testing framework if available
   * @param requestId Unique identifier for the request
   * @param metrics Metrics to report
   */
  private reportMetricsToDistributedFramework(requestId: string, metrics: any): void {
    // Check if distributed testing framework is available
    if (typeof window !== 'undefined' && 
        window['ipfs_accelerate_distributed_testing']) {
      try {
        window['ipfs_accelerate_distributed_testing'].reportMetrics(
          'openai',
          requestId,
          metrics
        );
      } catch (error) {
        console.warn('Failed to report metrics to distributed testing framework:', error);
      }
    }
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

  /**
   * Generate speech from text with enhanced voice capabilities
   * @param text Text to convert to speech
   * @param voice Voice type to use (alloy, echo, fable, onyx, nova, shimmer)
   * @param options Additional TTS options
   * @returns Audio data as ArrayBuffer
   */
  async textToSpeech(
    text: string, 
    voice: string | OpenAIVoiceType = OpenAIVoiceType.ALLOY, 
    options?: ApiRequestOptions
  ): Promise<ArrayBuffer> {
    const startTime = Date.now();
    const modelName = options?.model || 'tts-1';
    let success = false;
    let errorType;
    
    try {
      const requestData: OpenAISpeechRequest = {
        model: modelName,
        input: text,
        voice,
        response_format: options?.responseFormat || OpenAIAudioFormat.MP3,
        speed: options?.speed || 1.0
      };

      const response = await this.retryableRequest(async () => {
        const res = await fetch(this.apiEndpoints.speech, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.getApiKey(this.metadata)}`
          },
          body: JSON.stringify(requestData)
        });

        if (!res.ok) {
          const errorData = await res.json().catch(() => ({}));
          errorType = errorData.error?.type;
          throw this.createApiError(
            errorData.error?.message || `HTTP error ${res.status}`,
            res.status,
            errorData.error?.type || 'api_error'
          );
        }

        return res;
      }, options?.maxRetries || this.maxRetries);

      const arrayBuffer = await response.arrayBuffer();
      success = true;
      
      // Track metrics
      this.trackRequestMetrics(
        'tts',
        startTime,
        Date.now(),
        true,
        modelName,
        undefined,
        {
          requestSize: text.length,
          responseSize: arrayBuffer.byteLength
        }
      );
      
      return arrayBuffer;
    } catch (error) {
      // Track failed request
      this.trackRequestMetrics(
        'tts',
        startTime,
        Date.now(),
        false,
        modelName,
        undefined,
        {
          errorType: errorType || 'api_error',
          requestSize: text.length
        }
      );
      
      throw error;
    }
  }
  
  /**
   * Create voice agent for interactive voice applications
   * @param initialPrompt Initial system prompt for the agent
   * @param voiceSettings Voice settings for text-to-speech
   * @param options Additional options
   * @returns Voice agent interface
   */
  createVoiceAgent(
    initialPrompt: string = "You are a helpful voice assistant.",
    voiceSettings: {
      voice?: string | OpenAIVoiceType,
      model?: string,
      speed?: number,
      format?: string | OpenAIAudioFormat
    } = {},
    options: {
      chatModel?: string,
      temperature?: number,
      maxTokens?: number
    } = {}
  ) {
    // Use defaults if not provided
    const voice = voiceSettings.voice || OpenAIVoiceType.ALLOY;
    const speechModel = voiceSettings.model || 'tts-1';
    const speed = voiceSettings.speed || 1.0;
    const format = voiceSettings.format || OpenAIAudioFormat.MP3;
    const chatModel = options.chatModel || 'gpt-4o';
    
    // Initialize conversation with system message
    const messages: Message[] = [
      { role: 'system', content: initialPrompt }
    ];
    
    // Return voice agent interface
    return {
      /**
       * Process speech input (audio) and get a spoken response
       * @param audioData Audio data from user
       * @param ttsOptions Additional TTS options
       * @param chatOptions Additional chat options
       * @returns Speech audio data
       */
      async processAudio(
        audioData: Blob,
        ttsOptions: Partial<OpenAISpeechRequest> = {},
        chatOptions: ApiRequestOptions = {}
      ): Promise<{
        audioResponse: ArrayBuffer,
        textResponse: string,
        messages: Message[]
      }> {
        // Convert speech to text
        const transcription = await this.speechToText(audioData, {
          model: 'whisper-1',
          language: chatOptions?.language
        });
        
        // Add user message to conversation
        messages.push({ role: 'user', content: transcription.text });
        
        // Get chat completion
        const chatResponse = await this.chat(messages, {
          model: chatModel,
          temperature: options.temperature,
          maxTokens: options.maxTokens,
          ...chatOptions
        });
        
        // Add assistant response to conversation
        messages.push({ role: 'assistant', content: chatResponse.content });
        
        // Convert response to speech
        const audioResponse = await this.textToSpeech(
          chatResponse.content.toString(),
          voice,
          {
            model: speechModel,
            responseFormat: format,
            speed: speed,
            ...ttsOptions
          }
        );
        
        return {
          audioResponse,
          textResponse: chatResponse.content.toString(),
          messages: [...messages]
        };
      },
      
      /**
       * Process text input and get a spoken response
       * @param text Text input from user
       * @param ttsOptions Additional TTS options
       * @param chatOptions Additional chat options
       * @returns Speech audio data
       */
      async processText(
        text: string,
        ttsOptions: Partial<OpenAISpeechRequest> = {},
        chatOptions: ApiRequestOptions = {}
      ): Promise<{
        audioResponse: ArrayBuffer,
        textResponse: string,
        messages: Message[]
      }> {
        // Add user message to conversation
        messages.push({ role: 'user', content: text });
        
        // Get chat completion
        const chatResponse = await this.chat(messages, {
          model: chatModel,
          temperature: options.temperature,
          maxTokens: options.maxTokens,
          ...chatOptions
        });
        
        // Add assistant response to conversation
        messages.push({ role: 'assistant', content: chatResponse.content });
        
        // Convert response to speech
        const audioResponse = await this.textToSpeech(
          chatResponse.content.toString(),
          voice,
          {
            model: speechModel,
            responseFormat: format,
            speed: speed,
            ...ttsOptions
          }
        );
        
        return {
          audioResponse,
          textResponse: chatResponse.content.toString(),
          messages: [...messages]
        };
      },
      
      /**
       * Get the current conversation messages
       * @returns Array of conversation messages
       */
      getMessages(): Message[] {
        return [...messages];
      },
      
      /**
       * Reset the conversation history
       * @param newInitialPrompt Optional new system prompt
       */
      reset(newInitialPrompt?: string): void {
        messages.length = 0;
        messages.push({ 
          role: 'system', 
          content: newInitialPrompt || initialPrompt 
        });
      }
    };
  }

  /**
   * Transcribe audio to text with enhanced features
   * @param audioData Audio data to transcribe
   * @param options Transcription options
   * @returns Transcription response
   */
  async speechToText(audioData: Blob, options?: ApiRequestOptions): Promise<OpenAITranscriptionResponse> {
    const startTime = Date.now();
    const modelName = options?.model || 'whisper-1';
    let success = false;
    let errorType;
    
    try {
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
      
      // Add timestamp granularities if provided
      if (options?.timestamp_granularities) {
        const granularities = Array.isArray(options.timestamp_granularities) 
          ? options.timestamp_granularities 
          : [options.timestamp_granularities];
          
        granularities.forEach(granularity => {
          formData.append('timestamp_granularities[]', granularity);
        });
      }

      const response = await this.makeFormDataRequest(
        formData,
        undefined,
        { ...options, endpoint: this.apiEndpoints.transcriptions }
      ) as OpenAITranscriptionResponse;
      
      success = true;
      
      // Track metrics
      this.trackRequestMetrics(
        'stt',
        startTime,
        Date.now(),
        true,
        modelName,
        undefined,
        {
          requestSize: audioData.size,
          responseSize: JSON.stringify(response).length
        }
      );
      
      return response;
    } catch (error) {
      // Track failed request
      this.trackRequestMetrics(
        'stt',
        startTime,
        Date.now(),
        false,
        modelName,
        undefined,
        {
          errorType: errorType || 'api_error',
          requestSize: audioData.size
        }
      );
      
      throw error;
    }
  }
  
  /**
   * Translate audio to text
   * @param audioData Audio data to translate
   * @param options Translation options
   * @returns Translation response
   */
  async translateAudio(audioData: Blob, options?: ApiRequestOptions): Promise<OpenAITranscriptionResponse> {
    const startTime = Date.now();
    const modelName = options?.model || 'whisper-1';
    let success = false;
    let errorType;
    
    try {
      const formData = new FormData();
      formData.append('file', audioData);
      formData.append('model', modelName);
      
      if (options?.prompt) {
        formData.append('prompt', options.prompt);
      }
      
      if (options?.responseFormat) {
        formData.append('response_format', options.responseFormat);
      }
      
      if (options?.temperature) {
        formData.append('temperature', options.temperature.toString());
      }

      const response = await this.makeFormDataRequest(
        formData,
        undefined,
        { ...options, endpoint: this.apiEndpoints.translations }
      ) as OpenAITranscriptionResponse;
      
      success = true;
      
      // Track metrics
      this.trackRequestMetrics(
        'translate',
        startTime,
        Date.now(),
        true,
        modelName,
        undefined,
        {
          requestSize: audioData.size,
          responseSize: JSON.stringify(response).length
        }
      );
      
      return response;
    } catch (error) {
      // Track failed request
      this.trackRequestMetrics(
        'translate',
        startTime,
        Date.now(),
        false,
        modelName,
        undefined,
        {
          errorType: errorType || 'api_error',
          requestSize: audioData.size
        }
      );
      
      throw error;
    }
  }

  /**
   * Defines available GPT models with their properties
   */
  readonly gptModels = {
    // GPT-4 family
    'gpt-4o': { tokens: 128000, updated: '2024-05' },
    'gpt-4o-mini': { tokens: 128000, updated: '2024-05' },
    'gpt-4-turbo': { tokens: 128000, updated: '2023-12' },
    'gpt-4': { tokens: 8192, updated: '2023-04' },
    'gpt-4-32k': { tokens: 32768, updated: '2023-04' },
    'gpt-4-vision-preview': { tokens: 128000, updated: '2023-10', vision: true },
    
    // GPT-3.5 family
    'gpt-3.5-turbo': { tokens: 16385, updated: '2023-04' },
    'gpt-3.5-turbo-16k': { tokens: 16385, updated: '2023-06' }
  };
  
  /**
   * Defines available embedding models with their properties
   */
  readonly embeddingModels = {
    'text-embedding-3-small': { dimensions: 1536, updated: '2023-10' },
    'text-embedding-3-large': { dimensions: 3072, updated: '2023-10' },
    'text-embedding-ada-002': { dimensions: 1536, updated: '2022-12' }
  };
  
  /**
   * Defines available speech models with their properties
   */
  readonly speechModels = {
    'tts-1': { updated: '2023-11', quality: 'standard' },
    'tts-1-hd': { updated: '2023-11', quality: 'high' },
    'whisper-1': { updated: '2023-06', type: 'transcription' }
  };

  /**
   * Helper method to determine best model to use based on the task type
   * @param modelType Type of model needed
   * @param requestedModel Specific model requested (optional)
   * @returns Best model ID to use
   */
  private determineModel(modelType: string, requestedModel?: string): string {
    if (requestedModel) {
      return requestedModel;
    }

    // Default models per type
    const models: Record<string, string> = {
      'chat': 'gpt-4o',
      'chat-economic': 'gpt-4o-mini',  // For cost-sensitive applications
      'embedding': 'text-embedding-3-small',
      'embedding-high-quality': 'text-embedding-3-large',
      'moderation': 'text-moderation-latest',
      'image': 'dall-e-3',
      'speech': 'tts-1',
      'speech-hd': 'tts-1-hd',
      'transcription': 'whisper-1'
    };

    return models[modelType] || this.getDefaultModel();
  }
  
  /**
   * Get whether a model supports vision capabilities
   * @param model Model ID to check
   * @returns Whether the model supports vision
   */
  supportsVision(model: string): boolean {
    return model === 'gpt-4-vision-preview' || model === 'gpt-4o';
  }
  
  /**
   * Get maximum token context for a model
   * @param model Model ID to check
   * @returns Maximum token context or undefined if unknown
   */
  getModelMaxTokens(model: string): number | undefined {
    // Check direct matches in GPT models
    if (this.gptModels[model]) {
      return this.gptModels[model].tokens;
    }
    
    // Handle pattern-based matches
    if (model.startsWith('gpt-4-32k')) {
      return 32768;
    } else if (model.startsWith('gpt-4-')) {
      return 8192;
    } else if (model.startsWith('gpt-3.5-turbo-16k')) {
      return 16384;
    } else if (model.startsWith('gpt-3.5-turbo')) {
      return 4096;
    }
    
    return undefined;
  }

  /**
   * Helper method to process messages and add system message if needed
   * @param messages Array of chat messages
   * @param systemMessage Optional system message
   * @returns Processed messages array
   */
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
  
  /**
   * Execute parallel function calls
   * @param toolCalls Array of tool calls from model response
   * @param functions Object mapping function names to implementations
   * @param options Execution options
   * @returns Array of tool results
   */
  async executeParallelFunctions(
    toolCalls: Array<{
      id: string;
      type: string;
      function: {
        name: string;
        arguments: string;
      };
    }>,
    functions: Record<string, (args: any) => Promise<any>>,
    options: {
      timeout?: number;
      abortSignal?: AbortSignal;
      failOnError?: boolean;
    } = {}
  ): Promise<Array<{
    id: string;
    name: string;
    content: string;
    error?: string;
  }>> {
    // Default options
    const timeout = options.timeout || 30000;
    const failOnError = options.failOnError !== undefined ? options.failOnError : false;
    
    // Create promise for each tool call
    const functionPromises = toolCalls.map(async (toolCall) => {
      const { id, function: { name, arguments: argsString } } = toolCall;
      
      try {
        // Check if function exists
        if (!functions[name]) {
          throw new Error(`Function "${name}" not found`);
        }
        
        // Parse arguments
        const args = JSON.parse(argsString);
        
        // Execute function with timeout
        const resultPromise = functions[name](args);
        
        // Add timeout
        const timeoutPromise = new Promise<never>((_, reject) => {
          setTimeout(() => reject(new Error(`Function "${name}" timed out after ${timeout}ms`)), timeout);
        });
        
        // Race function execution against timeout
        const result = await Promise.race([resultPromise, timeoutPromise]);
        
        // Convert result to string
        const content = typeof result === 'string' ? result : JSON.stringify(result);
        
        return {
          id,
          name,
          content
        };
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        
        if (failOnError) {
          throw new Error(`Error executing function "${name}": ${errorMessage}`);
        }
        
        return {
          id,
          name,
          content: `Error: ${errorMessage}`,
          error: errorMessage
        };
      }
    });
    
    // Execute all functions in parallel
    return Promise.all(functionPromises);
  }
  
  /**
   * Complete a conversation with function calls in a single request
   * @param messages Initial messages in the conversation
   * @param functions Functions available for the model to call
   * @param options Chat and execution options
   * @returns Final conversation response
   */
  async chatWithFunctions(
    messages: Message[],
    functions: Record<string, (args: any) => Promise<any>>,
    options: ApiRequestOptions & {
      maxRounds?: number;
      functionTimeout?: number;
      failOnFunctionError?: boolean;
    } = {}
  ): Promise<ChatCompletionResponse> {
    // Configuration
    const maxRounds = options.maxRounds || 5;
    const functionTimeout = options.functionTimeout || 30000;
    
    // Create tools array from functions
    const tools = Object.entries(functions).map(([name, fn]) => {
      const metadata = fn.toString();
      // Extract parameter types from function signature
      const paramMatch = metadata.match(/\(([^)]*)\)/);
      const hasParams = paramMatch && paramMatch[1].trim() !== '';
      
      return {
        type: 'function',
        function: {
          name,
          description: `Function ${name}`,
          parameters: {
            type: 'object',
            properties: hasParams ? { args: { type: 'object' } } : {},
            required: hasParams ? ['args'] : []
          }
        }
      };
    });
    
    // Clone messages to avoid mutation
    let currentMessages = [...messages];
    let round = 0;
    
    while (round < maxRounds) {
      // Get response from model
      const response = await this.chat(currentMessages, {
        ...options,
        tools,
        toolChoice: 'auto'
      });
      
      // If no tool calls, we're done
      if (!response.tool_calls || response.tool_calls.length === 0) {
        return response;
      }
      
      // Execute functions in parallel
      const toolResults = await this.executeParallelFunctions(
        response.tool_calls,
        functions,
        {
          timeout: functionTimeout,
          failOnError: options.failOnFunctionError
        }
      );
      
      // Add assistant response with tool calls to messages
      currentMessages.push({
        role: 'assistant',
        content: null,
        tool_calls: response.tool_calls
      });
      
      // Add tool responses to messages
      for (const result of toolResults) {
        currentMessages.push({
          role: 'tool',
          tool_call_id: result.id,
          content: result.content
        });
      }
      
      round++;
    }
    
    // Get final response after all function calls
    return this.chat(currentMessages, {
      ...options,
      tools: undefined,
      toolChoice: undefined
    });
  }
}