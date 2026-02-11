import { BaseApiBackend } from '../base';
import { 
  HfTgiUnifiedOptions, 
  HfTgiUnifiedApiMetadata,
  ContainerInfo,
  HfTgiModelInfo,
  TextGenerationOptions,
  ChatGenerationOptions,
  HfTgiTextGenerationRequest,
  HfTgiTextGenerationResponse,
  HfTgiStreamChunk,
  StreamChunk,
  DeploymentConfig,
  PerformanceMetrics
} from './types';
import { ChatMessage, ChatOptions, EndpointHandler, PriorityLevel } from '../types';
import { CircuitBreaker } from '../utils/circuit_breaker';
import { execSync, spawn } from 'child_process';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';
import fetch from 'node-fetch';

// Add missing exports from node-fetch when working with ESM
const Response = fetch.Response;
const Headers = fetch.Headers;

/**
 * HF TGI Unified - Hugging Face Text Generation Inference API client with unified interface
 * Supports both direct API access and container-based deployment
 */
export class HfTgiUnified extends BaseApiBackend {
  private options: HfTgiUnifiedOptions;
  private apiKey: string | null = null;
  private baseApiUrl: string;
  private containerUrl: string;
  private modelId: string;
  private circuitBreaker: CircuitBreaker;
  private useContainer: boolean;
  private containerInfo: ContainerInfo | null = null;
  private dockerRegistry: string;
  private containerTag: string;
  private gpuDevice: string;
  private promptTemplates: Record<string, string> = {
    // For generic models
    default: "{input_text}",
    // For instruction models
    instruction: "### Instruction:\n{input_text}\n\n### Response:",
    // For chat models with system message
    chat: "{system_message}\n\n{messages}",
    // For Llama2 chat format
    llama2: "<s>[INST] {input_text} [/INST]",
    // For Falcon chat format
    falcon: "User: {input_text}\n\nAssistant:",
    // For Mistral chat format
    mistral: "<s>[INST] {input_text} [/INST]",
    // For ChatML format
    chatml: "<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
  };
  
  // Test prompts for benchmarking
  private testPrompts: string[] = [
    "Summarize the benefits of exercise for mental health.",
    "Explain the difference between machine learning and deep learning in simple terms.",
    "Write a short paragraph about climate change and its impact on ecosystems.",
    "Describe the main features of the Python programming language.",
    "What are three effective strategies for time management?"
  ];

  /**
   * Creates a new HF TGI Unified API client
   * @param options Configuration options
   * @param metadata API metadata including API keys
   */
  constructor(options: HfTgiUnifiedOptions = {}, metadata: HfTgiUnifiedApiMetadata = {}) {
    super(options, metadata);
    
    this.options = {
      apiUrl: 'https://api-inference.huggingface.co/models',
      containerUrl: metadata.hf_container_url || 'http://localhost:8080',
      maxRetries: 3,
      requestTimeout: 60000,
      useRequestQueue: true,
      debug: false,
      useContainer: false,
      dockerRegistry: metadata.docker_registry || 'ghcr.io/huggingface/text-generation-inference',
      containerTag: metadata.container_tag || 'latest',
      gpuDevice: metadata.gpu_device || '0',
      maxTokens: 512,
      temperature: 0.7,
      topP: 0.95,
      topK: 50,
      repetitionPenalty: 1.0,
      stopSequences: [],
      ...options
    };
    
    this.baseApiUrl = this.options.apiUrl as string;
    this.containerUrl = this.options.containerUrl as string;
    this.useContainer = this.options.useContainer as boolean;
    this.dockerRegistry = this.options.dockerRegistry as string;
    this.containerTag = this.options.containerTag as string;
    this.gpuDevice = this.options.gpuDevice as string;
    
    this.apiKey = this.getApiKey('hf_api_key', 'HF_API_KEY');
    this.modelId = metadata.model_id || 'google/flan-t5-small';
    
    // Initialize circuit breaker for API requests
    this.circuitBreaker = new CircuitBreaker({
      failureThreshold: 3,
      resetTimeout: 30000,
      name: 'hf_tgi_unified'
    });
  }

  /**
   * Get the default model for HF TGI
   * @returns The default model ID
   */
  getDefaultModel(): string {
    return this.modelId;
  }

  /**
   * Check if the provided model is compatible with HF TGI
   * @param model Model ID to check
   * @returns Whether the model is compatible
   */
  isCompatibleModel(model: string): boolean {
    // Models known to be compatible with text-generation-inference
    const knownCompatibleModels = [
      'google/flan-t5-small',
      'google/flan-t5-base',
      'google/flan-t5-large',
      'facebook/opt-125m',
      'facebook/opt-350m',
      'facebook/opt-1.3b',
      'bigscience/bloom-560m',
      'bigscience/bloom-1b1',
      'tiiuae/falcon-7b',
      'meta-llama/Llama-2-7b-hf',
      'meta-llama/Llama-2-7b-chat-hf',
      'mistralai/Mistral-7B-v0.1',
      'mistralai/Mistral-7B-Instruct-v0.1',
      'microsoft/phi-1_5',
      'microsoft/phi-2'
    ];
    
    if (knownCompatibleModels.includes(model)) {
      return true;
    }
    
    // Check for common text generation model patterns
    const textGenPatterns = [
      /flan-t5/i,
      /opt-/i,
      /llama-/i,
      /bloom-/i,
      /falcon-/i,
      /mistral/i,
      /stablelm/i,
      /gpt2/i,
      /gpt-/i,
      /phi-/i,
      /qwen/i
    ];
    
    return textGenPatterns.some(pattern => pattern.test(model));
  }

  /**
   * Create an endpoint handler for API requests
   * @returns Endpoint handler
   */
  createEndpointHandler(): EndpointHandler {
    return {
      makeRequest: async (endpoint: string, requestOptions: RequestInit, priority: PriorityLevel = 'NORMAL') => {
        // Determine which URL to use based on mode
        const baseUrl = this.useContainer ? this.containerUrl : this.baseApiUrl;
        const url = `${baseUrl}${endpoint}`;
        
        // Add headers based on mode (API key for HF API, no auth for container)
        const headers: Record<string, string> = {
          'Content-Type': 'application/json',
          ...requestOptions.headers as Record<string, string>
        };
        
        // Only add Authorization header for API mode, not container mode
        if (!this.useContainer && this.apiKey) {
          headers['Authorization'] = `Bearer ${this.apiKey}`;
        }
        
        const options: RequestInit = {
          ...requestOptions,
          headers
        };
        
        if (this.options.debug) {
          console.log(`Making request to ${url}`);
          console.log('Request options:', options);
        }
        
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
      if (this.useContainer) {
        // For container mode, check if container is running first
        if (!this.containerInfo || this.containerInfo.status !== 'running') {
          return false;
        }
        
        // Use health endpoint for container
        const handler = this.createEndpointHandler();
        const response = await handler.makeRequest('/health', { 
          method: 'GET' 
        }, 'HIGH');
        
        return response.ok;
      } else {
        // For API mode, check if model exists
        const modelEndpoint = `/${encodeURIComponent(this.modelId)}`;
        const handler = this.createEndpointHandler();
        const response = await handler.makeRequest(modelEndpoint, { 
          method: 'GET' 
        }, 'HIGH');
        
        if (!response.ok) {
          return false;
        }
        
        const data = await response.json();
        return data && typeof data === 'object';
      }
    } catch (error) {
      if (this.options.debug) {
        console.error('Error testing endpoint:', error);
      }
      return false;
    }
  }

  /**
   * Make a POST request to the HF TGI API
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
      // Handle specific error codes
      if (response.status === 404) {
        throw new Error(`Model not found: ${this.modelId}`);
      } else if (response.status === 403) {
        throw new Error('Authorization error: Invalid API key or insufficient permissions');
      } else if (response.status === 503) {
        throw new Error('The model is currently loading or busy. Try again later.');
      }
      
      const errorText = await response.text();
      throw new Error(`HF TGI API error (${response.status}): ${errorText}`);
    }
    
    return response.json();
  }

  /**
   * Start a container instance of HF TGI
   * @param config Container deployment configuration
   * @returns Container information
   */
  async startContainer(config?: Partial<DeploymentConfig>): Promise<ContainerInfo> {
    if (this.containerInfo && this.containerInfo.status === 'running') {
      return this.containerInfo;
    }
    
    const deployConfig: DeploymentConfig = {
      dockerRegistry: config?.dockerRegistry || this.dockerRegistry,
      containerTag: config?.containerTag || this.containerTag,
      gpuDevice: config?.gpuDevice || this.gpuDevice,
      modelId: config?.modelId || this.modelId,
      port: config?.port || 8080,
      env: config?.env || {},
      volumes: config?.volumes || [],
      network: config?.network || 'bridge',
      parameters: config?.parameters || [],
      maxInputLength: config?.maxInputLength || 2048,
      disableGpu: config?.disableGpu || false
    };
    
    try {
      // Check if Docker is available
      execSync('docker --version', { stdio: 'ignore' });
      
      // Pull the container image if not already pulled
      const imageName = `${deployConfig.dockerRegistry}:${deployConfig.containerTag}`;
      if (this.options.debug) {
        console.log(`Pulling container image: ${imageName}`);
      }
      
      try {
        execSync(`docker pull ${imageName}`, { stdio: 'inherit' });
      } catch (error) {
        if (this.options.debug) {
          console.error('Error pulling container image:', error);
        }
        throw new Error(`Failed to pull container image: ${imageName}`);
      }
      
      // Start the container
      const volumes = deployConfig.volumes?.length 
        ? deployConfig.volumes.map(v => `-v ${v}`).join(' ') 
        : '';
      
      const envVars = Object.entries(deployConfig.env || {})
        .map(([key, value]) => `-e ${key}=${value}`)
        .join(' ');
      
      const gpuArgs = deployConfig.disableGpu ? '' : 
        (deployConfig.gpuDevice ? `--gpus device=${deployConfig.gpuDevice}` : '--gpus all');
      
      const networkArg = deployConfig.network 
        ? `--network=${deployConfig.network}` 
        : '';
      
      const additionalParams = deployConfig.parameters?.length
        ? deployConfig.parameters.join(' ')
        : '';
      
      const maxInputLengthArg = deployConfig.maxInputLength
        ? `--max-input-length ${deployConfig.maxInputLength}`
        : '';
      
      const containerName = `hf-tgi-${Date.now()}`;
      const command = `docker run -d --name ${containerName} \
        -p ${deployConfig.port}:80 \
        ${gpuArgs} \
        ${envVars} \
        ${volumes} \
        ${networkArg} \
        ${imageName} \
        --model-id ${deployConfig.modelId} \
        ${maxInputLengthArg} \
        ${additionalParams}`;
      
      if (this.options.debug) {
        console.log(`Starting container with command: ${command}`);
      }
      
      const containerId = execSync(command).toString().trim();
      
      // Wait for container to start
      await new Promise(resolve => setTimeout(resolve, 10000));
      
      // Update container URL to point to the launched container
      this.containerUrl = `http://localhost:${deployConfig.port}`;
      
      this.containerInfo = {
        containerId,
        host: 'localhost',
        port: deployConfig.port,
        status: 'running',
        startTime: new Date()
      };
      
      return this.containerInfo;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      
      this.containerInfo = {
        containerId: '',
        host: '',
        port: 0,
        status: 'failed',
        error: errorMessage
      };
      
      throw new Error(`Failed to start HF TGI container: ${errorMessage}`);
    }
  }

  /**
   * Stop and remove the HF TGI container
   * @returns Whether the container was stopped successfully
   */
  async stopContainer(): Promise<boolean> {
    if (!this.containerInfo || !this.containerInfo.containerId) {
      return true;
    }
    
    try {
      // Stop the container
      execSync(`docker stop ${this.containerInfo.containerId}`, { stdio: 'ignore' });
      
      // Remove the container
      execSync(`docker rm ${this.containerInfo.containerId}`, { stdio: 'ignore' });
      
      this.containerInfo = null;
      return true;
    } catch (error) {
      if (this.options.debug) {
        console.error('Error stopping container:', error);
      }
      return false;
    }
  }

  /**
   * Get model information (parameters, revision, etc.)
   * @param modelId Optional model ID, defaults to the instance model ID
   * @returns Model information
   */
  async getModelInfo(modelId?: string): Promise<HfTgiModelInfo> {
    const model = modelId || this.modelId;
    
    if (this.useContainer) {
      // For container mode, use info API endpoint
      const handler = this.createEndpointHandler();
      const response = await handler.makeRequest('/info', { 
        method: 'GET' 
      }, 'HIGH');
      
      if (!response.ok) {
        throw new Error(`Failed to get model info: ${response.statusText}`);
      }
      
      const data = await response.json();
      return {
        model_id: model,
        status: 'ok',
        revision: data.model_id || data.revision,
        framework: data.framework || 'unknown',
        max_input_length: data.max_input_length,
        max_total_tokens: data.max_total_tokens,
        parameters: Object.keys(data.parameters || {})
      };
    } else {
      // For API mode, need to use model info endpoint
      const handler = this.createEndpointHandler();
      const endpoint = `/${encodeURIComponent(model)}`;
      
      const response = await handler.makeRequest(
        endpoint,
        {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        },
        'HIGH'
      );
      
      if (!response.ok) {
        throw new Error(`Failed to get model info: ${response.statusText}`);
      }
      
      const info = await response.json();
      return {
        model_id: model,
        status: 'ok',
        framework: info.framework || 'unknown',
        revision: info.sha || info.revision,
        parameters: info.parameters ? Object.keys(info.parameters) : undefined
      };
    }
  }

  /**
   * Format a request for text generation
   * @param text Text to generate from
   * @param options Generation options
   * @returns Formatted request
   */
  private formatRequest(text: string, options: TextGenerationOptions = {}): HfTgiTextGenerationRequest {
    return {
      inputs: text,
      parameters: {
        max_new_tokens: options.maxTokens || this.options.maxTokens as number,
        temperature: options.temperature || this.options.temperature as number,
        top_p: options.topP || this.options.topP as number,
        top_k: options.topK || this.options.topK as number,
        repetition_penalty: options.repetitionPenalty || this.options.repetitionPenalty as number,
        return_full_text: options.includeInputText || false,
        stream: options.stream || false,
        stop: options.stopSequences || this.options.stopSequences as string[]
      }
    };
  }

  /**
   * Format a chat prompt based on the model and messages
   * @param messages Array of chat messages
   * @param options Chat generation options
   * @returns Formatted prompt string
   */
  private formatChatPrompt(messages: ChatMessage[], options: ChatGenerationOptions = {}): string {
    // Determine prompt template based on model or provided template
    let templateName = 'default';
    const modelId = options.model || this.modelId;
    
    if (options.promptTemplate) {
      // Use custom template if provided
      templateName = options.promptTemplate;
    } else if (modelId.includes('llama-2') || modelId.includes('llama2')) {
      templateName = 'llama2';
    } else if (modelId.includes('falcon')) {
      templateName = 'falcon';
    } else if (modelId.includes('mistral')) {
      templateName = 'mistral';
    } else if (modelId.includes('chatbot') || modelId.includes('chat') || modelId.includes('gpt')) {
      templateName = 'chatml';
    } else if (modelId.includes('instruct')) {
      templateName = 'instruction';
    }
    
    let template = this.promptTemplates[templateName] || this.promptTemplates.default;
    
    // Format system message if present
    const systemMessage = options.systemMessage || '';
    
    // Format user messages
    let formattedMessages = '';
    
    if (templateName === 'chatml') {
      formattedMessages = messages.map(msg => {
        const role = msg.role === 'user' ? 'user' : 'assistant';
        return `<|im_start|>${role}\n${msg.content}<|im_end|>\n`;
      }).join('');
      
      // Add the final assistant prefix for generation
      formattedMessages += '<|im_start|>assistant\n';
    } else if (templateName === 'llama2' || templateName === 'mistral') {
      // For Llama2/Mistral format, we build a conversation
      formattedMessages = messages.map((msg, idx) => {
        if (msg.role === 'user') {
          return `<s>[INST] ${msg.content} [/INST]`;
        } else {
          return msg.content;
        }
      }).join('\n');
      
      // If last message is from assistant, add user prefix for next message
      if (messages.length > 0 && messages[messages.length - 1].role === 'assistant') {
        formattedMessages += '\n<s>[INST] ';
      }
    } else {
      // For other formats, just concatenate the last user message
      const userMessages = messages.filter(msg => msg.role === 'user');
      if (userMessages.length > 0) {
        formattedMessages = userMessages[userMessages.length - 1].content;
      }
    }
    
    // Apply the template
    let prompt = template
      .replace('{system_message}', systemMessage)
      .replace('{messages}', formattedMessages)
      .replace('{input_text}', formattedMessages || '');
    
    return prompt;
  }

  /**
   * Generate text using the HF TGI API
   * @param text Text prompt to generate from
   * @param options Text generation options
   * @returns Generated text
   */
  async generateText(text: string, options: TextGenerationOptions = {}): Promise<string> {
    const model = options.model || this.modelId;
    const request = this.formatRequest(text, options);
    
    // In container mode, endpoint is '/generate'
    // In API mode, endpoint includes the model name
    const endpoint = this.useContainer 
      ? '/generate' 
      : `/${encodeURIComponent(model)}`;
    
    // If streaming is requested, use the streaming method
    if (options.stream) {
      const streamResponse = this.streamGenerateText(text, options);
      
      // Collect all chunks into a single response
      let fullText = '';
      let lastChunk: StreamChunk | null = null;
      
      for await (const chunk of streamResponse) {
        lastChunk = chunk;
        // For API mode, we get incremental chunks; for container mode we may get full text
        if (chunk.fullText && chunk.done) {
          fullText = chunk.fullText;
          break;
        } else {
          fullText += chunk.text;
        }
      }
      
      return fullText;
    }
    
    // For non-streaming mode
    const response = await this.makePostRequest<HfTgiTextGenerationResponse>(
      endpoint,
      request,
      options.priority || 'NORMAL'
    );
    
    return response.generated_text;
  }

  /**
   * Generate text as a stream of chunks
   * @param text Text prompt to generate from 
   * @param options Text generation options
   * @returns Async generator of stream chunks
   */
  async *streamGenerateText(text: string, options: TextGenerationOptions = {}): AsyncGenerator<StreamChunk> {
    const model = options.model || this.modelId;
    const request = this.formatRequest(text, {
      ...options,
      stream: true
    });
    
    // In container mode, endpoint is '/generate'
    // In API mode, endpoint includes the model name
    const endpoint = this.useContainer 
      ? '/generate' 
      : `/${encodeURIComponent(model)}`;
    
    const handler = this.createEndpointHandler();
    
    const response = await handler.makeRequest(
      endpoint,
      {
        method: 'POST',
        body: JSON.stringify(request),
        headers: {
          'Content-Type': 'application/json'
        }
      },
      options.priority || 'NORMAL'
    );
    
    if (!response.ok) {
      throw new Error(`Text generation API error: ${response.statusText}`);
    }
    
    if (!response.body) {
      throw new Error('Response body is null');
    }
    
    // For container mode: parse SSE stream
    if (this.useContainer) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullGeneratedText = '';
      
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          // Add to buffer and process all complete events
          buffer += decoder.decode(value, { stream: true });
          
          // Split buffer by "data: " lines
          const lines = buffer.split('\n');
          buffer = '';
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const content = line.slice(6).trim();
                
                // Skip empty or heartbeat messages
                if (!content || content === '[DONE]') {
                  continue;
                }
                
                const chunk = JSON.parse(content) as HfTgiStreamChunk;
                fullGeneratedText += chunk.token?.text || '';
                
                if (chunk.generated_text !== undefined) {
                  // Final chunk with complete generation
                  yield {
                    text: chunk.token?.text || '',
                    token: chunk.token,
                    done: true,
                    fullText: chunk.generated_text || fullGeneratedText,
                    requestId: options.priority
                  };
                  return;
                } else {
                  // Incremental chunk
                  yield {
                    text: chunk.token?.text || '',
                    token: chunk.token,
                    done: false,
                    requestId: options.priority
                  };
                  
                  // Call stream callback if provided
                  if (options.streamCallback) {
                    options.streamCallback({
                      text: chunk.token?.text || '',
                      token: chunk.token,
                      done: false,
                      requestId: options.priority
                    });
                  }
                }
              } catch (error) {
                console.error('Error parsing SSE chunk:', error);
                // Skip invalid chunks
                continue;
              }
            } else {
              // Keep incomplete lines in buffer
              buffer += line + '\n';
            }
          }
        }
        
        // Yield final chunk
        yield {
          text: '',
          done: true,
          fullText: fullGeneratedText,
          requestId: options.priority
        };
      } catch (error) {
        console.error('Error reading stream:', error);
        throw error;
      } finally {
        reader.releaseLock();
      }
    } 
    // For API mode: HF API doesn't support true streaming, so we simulate it
    else {
      const text = await response.text();
      try {
        const data = JSON.parse(text);
        
        // Split the generated text into tokens (this is a simulation)
        const tokens = data.generated_text.split(/(\s+)/).filter(Boolean);
        
        let generatedSoFar = '';
        for (let i = 0; i < tokens.length; i++) {
          generatedSoFar += tokens[i];
          
          const isLast = i === tokens.length - 1;
          const chunk: StreamChunk = {
            text: tokens[i],
            done: isLast,
            fullText: isLast ? data.generated_text : undefined,
            requestId: options.priority
          };
          
          // Call stream callback if provided
          if (options.streamCallback) {
            options.streamCallback(chunk);
          }
          
          yield chunk;
          
          // Add a small delay to simulate streaming
          if (!isLast) {
            await new Promise(resolve => setTimeout(resolve, 10));
          }
        }
      } catch (error) {
        console.error('Error parsing API response:', error);
        throw new Error(`Failed to parse API response: ${error}`);
      }
    }
  }

  /**
   * Generate a chat response based on conversation history
   * @param messages Array of chat messages
   * @param options Chat options
   * @returns Generated response
   */
  async chat(messages: ChatMessage[], options: ChatOptions = {}): Promise<StreamChunk> {
    if (messages.length === 0) {
      throw new Error('Chat messages array cannot be empty');
    }
    
    // Format the chat prompt based on model and messages
    const prompt = this.formatChatPrompt(messages, options);
    
    // Convert chat options to text generation options
    const genOptions: TextGenerationOptions = {
      model: options.model || this.modelId,
      maxTokens: options.maxTokens || this.options.maxTokens as number,
      temperature: options.temperature || this.options.temperature as number,
      topP: options.topP || this.options.topP as number,
      topK: options.topK || this.options.topK as number,
      repetitionPenalty: options.repetitionPenalty || this.options.repetitionPenalty as number,
      stream: false,
      priority: options.priority || 'NORMAL',
      includeInputText: false
    };
    
    // Generate text from the formatted prompt
    const response = await this.generateText(prompt, genOptions);
    
    return {
      text: response,
      fullText: response,
      done: true,
      requestId: options.priority
    };
  }

  /**
   * Stream a chat response based on conversation history
   * @param messages Array of chat messages
   * @param options Chat options
   * @returns Async generator of stream chunks
   */
  async *streamChat(messages: ChatMessage[], options: ChatOptions = {}): AsyncGenerator<StreamChunk> {
    if (messages.length === 0) {
      throw new Error('Chat messages array cannot be empty');
    }
    
    // Format the chat prompt based on model and messages
    const prompt = this.formatChatPrompt(messages, options);
    
    // Convert chat options to text generation options
    const genOptions: TextGenerationOptions = {
      model: options.model || this.modelId,
      maxTokens: options.maxTokens || this.options.maxTokens as number,
      temperature: options.temperature || this.options.temperature as number,
      topP: options.topP || this.options.topP as number,
      topK: options.topK || this.options.topK as number,
      repetitionPenalty: options.repetitionPenalty || this.options.repetitionPenalty as number,
      stream: true,
      priority: options.priority || 'NORMAL',
      includeInputText: false,
      streamCallback: options.streamCallback
    };
    
    // Stream text from the formatted prompt
    for await (const chunk of this.streamGenerateText(prompt, genOptions)) {
      yield chunk;
    }
  }

  /**
   * Run a benchmark on the text generation model
   * @param options Benchmark options
   * @returns Performance metrics
   */
  async runBenchmark(options: {
    iterations?: number;
    model?: string;
    maxTokens?: number;
  } = {}): Promise<PerformanceMetrics> {
    const iterations = options.iterations || 5;
    const model = options.model || this.modelId;
    const maxTokens = options.maxTokens || 100;
    
    // Create test prompt
    const samplePrompt = this.testPrompts[0];
    
    // Benchmark text generation
    const startSingle = performance.now();
    let tokens = 0;
    let inputTokens = samplePrompt.split(/\s+/).length; // Rough estimate
    
    for (let i = 0; i < iterations; i++) {
      const response = await this.generateText(samplePrompt, { 
        model,
        maxTokens
      });
      tokens += response.split(/\s+/).length; // Rough estimate
    }
    
    const endSingle = performance.now();
    const singleTime = (endSingle - startSingle) / iterations;
    
    // Calculate metrics
    const tokensPerSecond = (tokens / iterations) / (singleTime / 1000);
    
    const result: PerformanceMetrics = {
      singleGenerationTime: singleTime,
      tokensPerSecond,
      generatedTokens: tokens / iterations,
      inputTokens,
      timestamp: new Date().toISOString()
    };
    
    return result;
  }
  
  /**
   * Switch between API and container modes
   * @param useContainer Whether to use container mode
   * @returns Updated mode setting
   */
  setMode(useContainer: boolean): boolean {
    this.useContainer = useContainer;
    return this.useContainer;
  }
  
  /**
   * Get the current mode (API or container)
   * @returns Current mode setting
   */
  getMode(): string {
    return this.useContainer ? 'container' : 'api';
  }
}