import { BaseApiBackend } from '../base';
import { ApiMetadata, ApiRequestOptions, Message, ChatCompletionResponse, StreamChunk } from '../types';
import { OllamaResponse, OllamaRequest, OllamaOptions, OllamaMessage, OllamaUsageStats } from './types';
import { v4 as uuidv4 } from 'uuid';

/**
 * Ollama API Backend for local LLM deployments
 * 
 * This implementation provides integration with Ollama API for running
 * local language models. Features:
 * - Request queue with concurrency limits
 * - Exponential backoff for error handling
 * - Request tracking with unique IDs
 * - Streaming support for chat completions
 * - Circuit breaker pattern for resilience
 */
export class Ollama extends BaseApiBackend {
  private apiBase: string;
  private defaultModel: string;
  private usageStats: OllamaUsageStats;
  
  // Request queue and concurrency management
  private requestQueue: Array<any> = [];
  private currentRequests: number = 0;
  private maxConcurrentRequests: number = 5;
  private queueProcessing: boolean = false;
  
  // Circuit breaker pattern
  private circuitState: 'CLOSED' | 'OPEN' | 'HALF-OPEN' = 'CLOSED';
  private failureCount: number = 0;
  private failureThreshold: number = 5;
  private circuitTimeout: number = 30; // seconds
  private lastFailureTime: number = 0;
  
  // Backoff configuration
  private maxRetries: number = 5;
  private initialRetryDelay: number = 1000; // ms
  private backoffFactor: number = 2;
  private maxRetryDelay: number = 16000; // ms
  
  // Priority levels
  private readonly PRIORITY_HIGH = 0;
  private readonly PRIORITY_NORMAL = 1;
  private readonly PRIORITY_LOW = 2;

  constructor(resources: Record<string, any> = {}, metadata: ApiMetadata = {}) {
    super(resources, metadata);
    
    // Initialize Ollama API URL
    this.apiBase = this.getOllamaApiUrl();
    
    // Initialize usage stats
    this.usageStats = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      totalTokens: 0,
      totalPromptTokens: 0,
      totalCompletionTokens: 0
    };
    
    // Set default model
    this.defaultModel = process.env.OLLAMA_MODEL || "llama3";
    
    // Start queue processor
    this.startQueueProcessor();
  }

  /**
   * Get the Ollama API URL from metadata or environment variables
   */
  private getOllamaApiUrl(): string {
    // Try to get from metadata
    const apiUrl = this.metadata.ollama_api_url;
    if (apiUrl) {
      return apiUrl;
    }
    
    // Try to get from environment
    const envUrl = process.env.OLLAMA_API_URL;
    if (envUrl) {
      return envUrl;
    }
    
    // Return default if no URL found
    return "http://localhost:11434/api";
  }

  /**
   * Start processing the request queue
   */
  private async startQueueProcessor(): Promise<void> {
    if (this.queueProcessing) {
      return; // Already processing queue
    }
    
    this.queueProcessing = true;
    
    while (true) {
      try {
        // Wait for requests in the queue
        if (this.requestQueue.length === 0) {
          await new Promise(resolve => setTimeout(resolve, 100));
          continue;
        }
        
        // Check if we can process more requests
        if (this.currentRequests >= this.maxConcurrentRequests) {
          await new Promise(resolve => setTimeout(resolve, 100));
          continue;
        }
        
        // Get the next request from the queue
        const request = this.requestQueue.shift();
        if (!request) continue;
        
        const { resolve, reject, endpoint, data, stream, requestId, priority } = request;
        
        this.currentRequests++;
        
        // Check circuit breaker
        if (this.circuitState === 'OPEN') {
          // Circuit is open, check if timeout has elapsed
          if (Date.now() - this.lastFailureTime > this.circuitTimeout * 1000) {
            // Transition to half-open state
            this.circuitState = 'HALF-OPEN';
          } else {
            // Circuit is open and timeout hasn't elapsed, fail fast
            this.currentRequests--;
            reject(new Error(`Circuit breaker is OPEN. Service unavailable.`));
            continue;
          }
        }
        
        // Process with retry logic
        this.processRequestWithRetry(endpoint, data, stream, requestId, resolve, reject);
      } catch (error) {
        console.error(`Error in queue processor: ${error}`);
      }
    }
  }

  /**
   * Process a request with retry logic
   */
  private async processRequestWithRetry(
    endpoint: string,
    data: any,
    stream: boolean,
    requestId: string,
    resolve: (value: any) => void,
    reject: (reason: any) => void
  ): Promise<void> {
    let retryCount = 0;
    let success = false;

    while (retryCount <= this.maxRetries && !success) {
      try {
        // Construct headers
        const headers: Record<string, string> = {
          'Content-Type': 'application/json'
        };
        
        // Include request ID in headers if provided
        if (requestId) {
          headers['X-Request-ID'] = requestId;
        }
        
        // Make request with proper error handling
        let response;
        if (stream) {
          // For streaming, we need to handle differently with fetch
          const fetchResponse = await fetch(endpoint, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(data)
          });
          
          if (!fetchResponse.ok) {
            throw new Error(`HTTP error! status: ${fetchResponse.status}`);
          }
          
          if (!fetchResponse.body) {
            throw new Error('Response body is null');
          }
          
          // Handle streaming response
          const reader = fetchResponse.body.getReader();
          const decoder = new TextDecoder('utf-8');
          
          // Create a generator function for streaming responses
          const streamGenerator = async function* () {
            let buffer = '';
            
            while (true) {
              const { done, value } = await reader.read();
              
              if (done) {
                break;
              }
              
              // Decode the chunk and add to buffer
              buffer += decoder.decode(value, { stream: true });
              
              // Process complete JSON objects
              let newlineIndex;
              while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
                const line = buffer.slice(0, newlineIndex);
                buffer = buffer.slice(newlineIndex + 1);
                
                if (line.trim()) {
                  try {
                    const chunk = JSON.parse(line);
                    yield chunk;
                  } catch (e) {
                    console.error(`Error parsing JSON: ${e}`);
                  }
                }
              }
            }
            
            // Handle any remaining data
            if (buffer.trim()) {
              try {
                const chunk = JSON.parse(buffer);
                yield chunk;
              } catch (e) {
                console.error(`Error parsing JSON: ${e}`);
              }
            }
          };
          
          // Update circuit breaker on success
          if (this.circuitState === 'HALF-OPEN') {
            // Success in half-open state, close the circuit
            this.circuitState = 'CLOSED';
            this.failureCount = 0;
          } else if (this.circuitState === 'CLOSED') {
            // Reset failure count on successful request
            this.failureCount = 0;
          }
          
          // Return the generator
          resolve(streamGenerator());
          success = true;
          this.currentRequests--;
          return;
        } else {
          // For non-streaming requests
          response = await fetch(endpoint, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(data)
          });
          
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          // Parse JSON response
          const result = await response.json();
          
          // Update circuit breaker on success
          if (this.circuitState === 'HALF-OPEN') {
            // Success in half-open state, close the circuit
            this.circuitState = 'CLOSED';
            this.failureCount = 0;
          } else if (this.circuitState === 'CLOSED') {
            // Reset failure count on successful request
            this.failureCount = 0;
          }
          
          // Update usage stats
          this.usageStats.totalRequests++;
          this.usageStats.successfulRequests++;
          
          resolve(result);
          success = true;
        }
      } catch (error) {
        retryCount++;
        
        // Update circuit breaker on failure
        this.failureCount++;
        this.lastFailureTime = Date.now();
        
        // Check if we should open the circuit
        if (this.circuitState === 'CLOSED' && this.failureCount >= this.failureThreshold) {
          this.circuitState = 'OPEN';
        } else if (this.circuitState === 'HALF-OPEN') {
          // Failed in half-open state, reopen the circuit
          this.circuitState = 'OPEN';
        }
        
        if (retryCount > this.maxRetries) {
          // Update usage stats
          this.usageStats.totalRequests++;
          this.usageStats.failedRequests++;
          
          this.currentRequests--;
          reject(error);
          return;
        }
        
        // Calculate backoff delay
        const delay = Math.min(
          this.initialRetryDelay * (this.backoffFactor ** (retryCount - 1)),
          this.maxRetryDelay
        );
        
        // Sleep with backoff
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    this.currentRequests--;
  }

  /**
   * Make a request to the Ollama API with queue and backoff
   */
  private async makePostRequestOllama(
    endpoint: string, 
    data: any, 
    stream: boolean = false, 
    requestId?: string,
    priority?: number
  ): Promise<any> {
    // Generate unique request ID if not provided
    if (!requestId) {
      requestId = uuidv4();
    }
    
    // Set default priority if not provided
    if (priority === undefined) {
      priority = this.PRIORITY_NORMAL;
    }
    
    // Return a promise that will be resolved when the request is processed
    return new Promise((resolve, reject) => {
      this.requestQueue.push({
        resolve,
        reject,
        endpoint,
        data,
        stream,
        requestId,
        priority
      });
    });
  }

  /**
   * Format messages to match Ollama API requirements
   */
  private formatMessages(messages: Message[]): OllamaMessage[] {
    const formattedMessages: OllamaMessage[] = [];
    
    if (!messages || messages.length === 0) {
      return [{ role: 'user', content: 'Hello' }];
    }
    
    for (const message of messages) {
      let role = message.role || 'user';
      const content = message.content || '';
      
      // Map standard roles to Ollama roles
      if (role === 'assistant') {
        role = 'assistant';
      } else if (role === 'system') {
        role = 'system';
      } else {
        role = 'user';
      }
      
      formattedMessages.push({
        role,
        content
      });
    }
    
    return formattedMessages;
  }

  /**
   * Extract usage information from Ollama response
   */
  private extractUsage(response: OllamaResponse): { prompt_tokens: number; completion_tokens: number; total_tokens: number } {
    return {
      prompt_tokens: response.prompt_eval_count || 0,
      completion_tokens: response.eval_count || 0,
      total_tokens: (response.prompt_eval_count || 0) + (response.eval_count || 0)
    };
  }

  /**
   * Create endpoint URL by path
   */
  private createEndpointUrl(path: string): string {
    return `${this.apiBase}/${path}`;
  }

  /**
   * Send a chat request to Ollama API
   */
  async chat(model: string, messages: Message[], options: ApiRequestOptions = {}): Promise<ChatCompletionResponse> {
    // Use provided model or default model
    const modelToUse = model || this.metadata.model || this.defaultModel;
    
    // Construct the proper endpoint URL
    const endpoint = this.createEndpointUrl('chat');
    
    // Format messages for Ollama API
    const formattedMessages = this.formatMessages(messages);
    
    // Prepare options dictionary
    const optionsDict: OllamaOptions = {};
    
    // Add max_tokens and temperature to options if provided
    if (options.max_tokens !== undefined) {
      optionsDict.num_predict = options.max_tokens;
    }
    
    if (options.temperature !== undefined) {
      optionsDict.temperature = options.temperature;
    }
    
    // Handle additional options
    if (options.top_p !== undefined) {
      optionsDict.top_p = options.top_p;
    }
    
    if (options.top_k !== undefined) {
      optionsDict.top_k = options.top_k;
    }
    
    // Prepare request data
    const data: OllamaRequest = {
      model: modelToUse,
      messages: formattedMessages,
      stream: false
    };
    
    // Add options if any exist
    if (Object.keys(optionsDict).length > 0) {
      data.options = optionsDict;
    }
    
    // Make request with queue and backoff
    try {
      const response = await this.makePostRequestOllama(endpoint, data) as OllamaResponse;
      
      // Update token usage stats
      if (response.prompt_eval_count !== undefined && response.eval_count !== undefined) {
        this.usageStats.totalPromptTokens += response.prompt_eval_count;
        this.usageStats.totalCompletionTokens += response.eval_count;
        this.usageStats.totalTokens += response.prompt_eval_count + response.eval_count;
      }
      
      // Process and normalize response
      return {
        text: response.message?.content || response.response || '',
        model: modelToUse,
        usage: this.extractUsage(response),
        implementation_type: '(REAL)'
      };
    } catch (error) {
      return {
        text: `Error: ${error}`,
        model: modelToUse,
        error: `${error}`,
        implementation_type: '(ERROR)'
      };
    }
  }

  /**
   * Generate text completions using the Ollama API
   */
  async generate(model: string, prompt: string, options: ApiRequestOptions = {}): Promise<ChatCompletionResponse> {
    // Construct a message from the prompt
    const messages: Message[] = [{ role: 'user', content: prompt }];
    
    // Call chat method
    return this.chat(model, messages, options);
  }

  /**
   * Generate streaming chat completions from Ollama API
   */
  async *streamChat(model: string, messages: Message[], options: ApiRequestOptions = {}): AsyncGenerator<StreamChunk> {
    // Use provided model or default model
    const modelToUse = model || this.metadata.model || this.defaultModel;
    
    // Construct the proper endpoint URL
    const endpoint = this.createEndpointUrl('chat');
    
    // Format messages for Ollama API
    const formattedMessages = this.formatMessages(messages);
    
    // Prepare options dictionary
    const optionsDict: OllamaOptions = {};
    
    // Add max_tokens and temperature to options if provided
    if (options.max_tokens !== undefined) {
      optionsDict.num_predict = options.max_tokens;
    }
    
    if (options.temperature !== undefined) {
      optionsDict.temperature = options.temperature;
    }
    
    // Handle additional options
    if (options.top_p !== undefined) {
      optionsDict.top_p = options.top_p;
    }
    
    if (options.top_k !== undefined) {
      optionsDict.top_k = options.top_k;
    }
    
    // Prepare request data
    const data: OllamaRequest = {
      model: modelToUse,
      messages: formattedMessages,
      stream: true
    };
    
    // Add options if any exist
    if (Object.keys(optionsDict).length > 0) {
      data.options = optionsDict;
    }
    
    // Make streaming request
    try {
      const responseStream = await this.makePostRequestOllama(endpoint, data, true);
      
      // Process streaming response
      let completionTokens = 0;
      
      for await (const chunk of responseStream) {
        completionTokens++;
        
        yield {
          text: chunk.message?.content || chunk.response || '',
          done: chunk.done || false,
          model: modelToUse
        };
        
        // If this is the final chunk, update token usage
        if (chunk.done) {
          const promptTokens = chunk.prompt_eval_count || 0;
          this.usageStats.totalPromptTokens += promptTokens;
          this.usageStats.totalCompletionTokens += completionTokens;
          this.usageStats.totalTokens += promptTokens + completionTokens;
        }
      }
    } catch (error) {
      yield {
        text: `Error: ${error}`,
        error: `${error}`,
        done: true,
        model: modelToUse
      };
    }
  }

  /**
   * List available models in Ollama
   */
  async listModels(): Promise<any[]> {
    const endpoint = this.createEndpointUrl('tags');
    
    try {
      const response = await fetch(endpoint, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data.models || [];
    } catch (error) {
      console.error(`Error listing models: ${error}`);
      return [];
    }
  }

  /**
   * Test the Ollama endpoint
   */
  async testOllamaEndpoint(endpointUrl?: string): Promise<boolean> {
    if (!endpointUrl) {
      endpointUrl = this.createEndpointUrl('chat');
    }
    
    const model = this.metadata.ollama_model || this.defaultModel;
    const messages: Message[] = [{ 
      role: 'user', 
      content: 'Testing the Ollama API. Please respond with a short message.'
    }];
    
    try {
      const response = await this.chat(model, messages);
      return 'text' in response && response.implementation_type === '(REAL)';
    } catch (error) {
      console.error(`Error testing Ollama endpoint: ${error}`);
      return false;
    }
  }

  /**
   * Reset usage statistics to zero
   */
  resetUsageStats(): void {
    this.usageStats = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      totalTokens: 0,
      totalPromptTokens: 0,
      totalCompletionTokens: 0
    };
  }

  /**
   * Returns true if the model is compatible with this backend
   */
  isCompatibleModel(modelName: string): boolean {
    // Check if model name follows Ollama naming convention
    return (
      modelName.startsWith('llama') || 
      modelName.startsWith('mistral') || 
      modelName.includes('/ollama/') ||
      modelName.includes(':latest')
    );
  }

  /**
   * Returns information about the backend implementation
   */
  getBackendInfo(): Record<string, any> {
    return {
      name: 'ollama',
      version: '1.0.0',
      api_type: 'ollama',
      description: 'Ollama API Backend for local LLM deployments',
      homepage: 'https://ollama.ai/',
      capabilities: {
        chat: true,
        streaming: true,
        batch_processing: false,
        fine_tuning: false,
        embeddings: true
      },
      models: ['llama3', 'mistral', 'phi', 'codellama', 'gemma'],
      supported_parameters: {
        temperature: true,
        top_p: true,
        top_k: true,
        repeat_penalty: true,
        num_predict: true
      }
    };
  }
}