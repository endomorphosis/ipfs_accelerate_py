import { BaseApiBackend } from '../base';
import { 
  OpenAiMiniOptions, 
  OpenAiMiniApiMetadata, 
  OpenAiMiniChatRequest, 
  OpenAiMiniChatResponse,
  OpenAiMiniStreamChunk,
  OpenAiMiniAudioOptions,
  OpenAiMiniTTSOptions,
  OpenAiMiniImageOptions,
  OpenAiMiniImageResponse,
  OpenAiMiniFileResponse,
  OpenAiMiniFileUploadOptions
} from './types';
import { ChatMessage, ChatOptions, EndpointHandler, PriorityLevel, StreamChunk } from '../types';
import * as fs from 'fs';
import * as path from 'path';
import { CircuitBreaker } from '../utils/circuit_breaker';

/**
 * OpenAI Mini API client - Lightweight implementation of OpenAI API client
 * Supports core functionality with optimized implementation
 */
export class OpenAiMini extends BaseApiBackend {
  private options: OpenAiMiniOptions;
  private apiKey: string | null = null;
  private baseUrl: string;
  private circuitBreaker: CircuitBreaker;

  /**
   * Creates a new OpenAI Mini API client
   * @param options Configuration options
   * @param metadata API metadata including API keys
   */
  constructor(options: OpenAiMiniOptions = {}, metadata: OpenAiMiniApiMetadata = {}) {
    super(options, metadata);
    
    this.options = {
      apiUrl: 'https://api.openai.com/v1',
      maxRetries: 3,
      requestTimeout: 30000,
      useRequestQueue: true,
      debug: false,
      ...options
    };
    
    this.baseUrl = this.options.apiUrl as string;
    this.apiKey = this.getApiKey('openai_mini_api_key', 'OPENAI_API_KEY');
    
    // Initialize circuit breaker
    this.circuitBreaker = new CircuitBreaker({
      failureThreshold: 3,
      resetTimeout: 30000,
      name: 'openai_mini'
    });
  }

  /**
   * Get the default model for OpenAI Mini API
   * @returns The default model ID
   */
  getDefaultModel(): string {
    return 'gpt-3.5-turbo';
  }

  /**
   * Check if the provided model is compatible with OpenAI Mini API
   * @param model Model ID to check
   * @returns Whether the model is compatible
   */
  isCompatibleModel(model: string): boolean {
    const compatibleModels = [
      // Chat models
      'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 
      'gpt-4', 'gpt-4-32k', 
      'gpt-4-turbo', 'gpt-4-turbo-preview',
      // Embedding models
      'text-embedding-ada-002',
      // Speech models
      'whisper-1', 'tts-1', 'tts-1-hd',
      // Image models
      'dall-e-2', 'dall-e-3'
    ];
    
    // Check exact matches
    if (compatibleModels.includes(model)) {
      return true;
    }
    
    // Check prefixes
    const modelPrefixes = ['gpt-3.5', 'gpt-4', 'text-embedding'];
    return modelPrefixes.some(prefix => model.startsWith(prefix));
  }

  /**
   * Create an endpoint handler for API requests
   * @returns Endpoint handler
   */
  createEndpointHandler(): EndpointHandler {
    return {
      makeRequest: async (endpoint: string, requestOptions: RequestInit, priority: PriorityLevel = 'NORMAL') => {
        if (!this.apiKey) {
          throw new Error('API key not found. Set OPENAI_API_KEY environment variable or provide in metadata.');
        }
        
        const url = `${this.baseUrl}${endpoint}`;
        
        // Add API key header
        const headers = {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
          ...requestOptions.headers,
        };
        
        const options: RequestInit = {
          ...requestOptions,
          headers
        };
        
        if (this.options.useRequestQueue) {
          return this.addToRequestQueue(url, options, priority);
        } else {
          return this.circuitBreaker.execute(() => fetch(url, options));
        }
      }
    };
  }

  /**
   * Test the API endpoint connectivity
   * @returns Whether the endpoint is working
   */
  async testEndpoint(): Promise<boolean> {
    try {
      const endpoint = '/models';
      const handler = this.createEndpointHandler();
      const response = await handler.makeRequest(endpoint, { method: 'GET' }, 'HIGH');
      
      if (!response.ok) {
        if (this.options.debug) {
          console.error(`Error testing endpoint: ${response.status} ${response.statusText}`);
        }
        return false;
      }
      
      const data = await response.json();
      return Array.isArray(data.data) && data.data.length > 0;
    } catch (error) {
      if (this.options.debug) {
        console.error('Error testing endpoint:', error);
      }
      return false;
    }
  }

  /**
   * Make a POST request to the OpenAI Mini API
   * @param endpoint API endpoint
   * @param data Request data
   * @param priority Request priority
   * @returns API response
   */
  async makePostRequest<T>(endpoint: string, data: any, priority: PriorityLevel = 'NORMAL'): Promise<T> {
    const handler = this.createEndpointHandler();
    
    const response = await handler.makeRequest(
      endpoint,
      {
        method: 'POST',
        body: JSON.stringify(data),
        headers: {
          'Content-Type': 'application/json'
        }
      },
      priority
    );
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(`OpenAI Mini API error (${response.status}): ${JSON.stringify(errorData)}`);
    }
    
    return response.json();
  }

  /**
   * Make a streaming request to the OpenAI Mini API
   * @param endpoint API endpoint
   * @param data Request data
   * @param priority Request priority
   * @returns Async generator for streaming responses
   */
  async *makeStreamRequest(endpoint: string, data: any, priority: PriorityLevel = 'NORMAL'): AsyncGenerator<StreamChunk> {
    const handler = this.createEndpointHandler();
    
    const response = await handler.makeRequest(
      endpoint,
      {
        method: 'POST',
        body: JSON.stringify({ ...data, stream: true }),
        headers: {
          'Content-Type': 'application/json'
        }
      },
      priority
    );
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(`OpenAI Mini API streaming error (${response.status}): ${JSON.stringify(errorData)}`);
    }
    
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
        
        buffer += decoder.decode(value, { stream: true });
        
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (line.trim() === '') continue;
          if (line.trim() === 'data: [DONE]') return;
          
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.choices && data.choices.length > 0) {
                const choice = data.choices[0];
                const content = choice.delta.content || '';
                
                const chunk: StreamChunk = {
                  content,
                  done: !!choice.finish_reason,
                  meta: { ...data }
                };
                
                yield chunk;
              }
            } catch (e) {
              if (this.options.debug) {
                console.error('Error parsing stream chunk:', e);
              }
            }
          }
        }
      }
    } finally {
      // Ensure the reader is released
      reader.releaseLock();
    }
  }

  /**
   * Generate a chat completion
   * @param messages Array of chat messages
   * @param options Chat options
   * @returns Chat completion response
   */
  async chat(messages: ChatMessage[], options: ChatOptions = {}): Promise<StreamChunk> {
    const model = options.model || this.getDefaultModel();
    
    const requestData: OpenAiMiniChatRequest = {
      model,
      messages,
      temperature: options.temperature,
      top_p: options.top_p,
      max_tokens: options.max_tokens,
      stop: options.stop
    };
    
    const response = await this.makePostRequest<OpenAiMiniChatResponse>(
      '/chat/completions',
      requestData,
      options.priority
    );
    
    if (!response.choices || response.choices.length === 0) {
      throw new Error('No choices returned from API');
    }
    
    return {
      content: response.choices[0].message.content,
      done: true,
      meta: response
    };
  }

  /**
   * Generate a streaming chat completion
   * @param messages Array of chat messages
   * @param options Chat options
   * @returns Async generator for streaming responses
   */
  async *streamChat(messages: ChatMessage[], options: ChatOptions = {}): AsyncGenerator<StreamChunk> {
    const model = options.model || this.getDefaultModel();
    
    const requestData: OpenAiMiniChatRequest = {
      model,
      messages,
      temperature: options.temperature,
      top_p: options.top_p,
      max_tokens: options.max_tokens,
      stop: options.stop
    };
    
    yield* this.makeStreamRequest('/chat/completions', requestData, options.priority);
  }

  /**
   * Upload a file to OpenAI API
   * @param filePath Path to the file to upload
   * @param options File upload options
   * @returns File upload response
   */
  async uploadFile(filePath: string, options: OpenAiMiniFileUploadOptions = {}): Promise<OpenAiMiniFileResponse> {
    if (!fs.existsSync(filePath)) {
      throw new Error(`File not found: ${filePath}`);
    }
    
    const fileName = options.fileName || path.basename(filePath);
    const purpose = options.purpose || 'fine-tune';
    
    const formData = new FormData();
    formData.append('purpose', purpose);
    
    // Read file as Blob
    const fileBuffer = fs.readFileSync(filePath);
    const fileBlob = new Blob([fileBuffer]);
    formData.append('file', fileBlob, fileName);
    
    const handler = this.createEndpointHandler();
    
    const response = await handler.makeRequest(
      '/files',
      {
        method: 'POST',
        body: formData,
        headers: {
          // Don't set Content-Type header, it will be set automatically with the boundary
        }
      },
      'NORMAL'
    );
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(`File upload error (${response.status}): ${JSON.stringify(errorData)}`);
    }
    
    return response.json();
  }

  /**
   * Transcribe audio to text
   * @param audioFilePath Path to audio file
   * @param options Audio transcription options
   * @returns Transcription text
   */
  async transcribeAudio(audioFilePath: string, options: OpenAiMiniAudioOptions = {}): Promise<string> {
    if (!fs.existsSync(audioFilePath)) {
      throw new Error(`Audio file not found: ${audioFilePath}`);
    }
    
    const model = options.model || 'whisper-1';
    const formData = new FormData();
    formData.append('model', model);
    
    if (options.prompt) {
      formData.append('prompt', options.prompt);
    }
    
    if (options.response_format) {
      formData.append('response_format', options.response_format);
    }
    
    if (options.temperature) {
      formData.append('temperature', options.temperature.toString());
    }
    
    if (options.language) {
      formData.append('language', options.language);
    }
    
    // Read audio file as Blob
    const fileBuffer = fs.readFileSync(audioFilePath);
    const audioBlob = new Blob([fileBuffer]);
    formData.append('file', audioBlob, path.basename(audioFilePath));
    
    const handler = this.createEndpointHandler();
    
    const response = await handler.makeRequest(
      '/audio/transcriptions',
      {
        method: 'POST',
        body: formData
      },
      'NORMAL'
    );
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(`Audio transcription error (${response.status}): ${JSON.stringify(errorData)}`);
    }
    
    const result = await response.json();
    return result.text;
  }

  /**
   * Generate speech from text
   * @param text Text to convert to speech
   * @param options TTS options
   * @returns Audio data as Buffer
   */
  async textToSpeech(text: string, options: OpenAiMiniTTSOptions = {}): Promise<Buffer> {
    const model = options.model || 'tts-1';
    const voice = options.voice || 'alloy';
    const speed = options.speed || 1.0;
    const response_format = options.response_format || 'mp3';
    
    const handler = this.createEndpointHandler();
    
    const response = await handler.makeRequest(
      '/audio/speech',
      {
        method: 'POST',
        body: JSON.stringify({
          model,
          input: text,
          voice,
          speed,
          response_format
        }),
        headers: {
          'Content-Type': 'application/json'
        }
      },
      'NORMAL'
    );
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(`Text-to-speech error (${response.status}): ${JSON.stringify(errorData)}`);
    }
    
    const arrayBuffer = await response.arrayBuffer();
    return Buffer.from(arrayBuffer);
  }

  /**
   * Generate images from a prompt
   * @param prompt Text prompt for image generation
   * @param options Image generation options
   * @returns Generated image data
   */
  async generateImage(prompt: string, options: OpenAiMiniImageOptions = {}): Promise<OpenAiMiniImageResponse> {
    const model = options.model || 'dall-e-3';
    const size = options.size || '1024x1024';
    const n = options.n || 1;
    const response_format = options.response_format || 'url';
    
    const requestData = {
      model,
      prompt,
      n,
      size,
      response_format,
      quality: options.quality,
      style: options.style
    };
    
    return this.makePostRequest<OpenAiMiniImageResponse>(
      '/images/generations',
      requestData,
      'NORMAL'
    );
  }
}