import { BaseApiBackend } from '../base';
import { 
  HfTeiUnifiedOptions, 
  HfTeiUnifiedApiMetadata,
  ContainerInfo,
  HfTeiModelInfo,
  HfTeiEmbeddingRequest,
  HfTeiEmbeddingResponse,
  EmbeddingOptions,
  PerformanceMetrics,
  DeploymentConfig
} from './types';
import { ChatMessage, ChatOptions, EndpointHandler, PriorityLevel, StreamChunk } from '../types';
import { CircuitBreaker } from '../utils/circuit_breaker';
import { execSync, spawn } from 'child_process';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';

/**
 * HF TEI Unified - Hugging Face Text Embedding Inference API client with unified interface
 * Supports both direct API access and container-based deployment
 */
export class HfTeiUnified extends BaseApiBackend {
  private options: HfTeiUnifiedOptions;
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
  private testSentences: string[] = [
    "The quick brown fox jumps over the lazy dog.",
    "I enjoy walking in the park on sunny days.",
    "Machine learning models can be used for natural language processing tasks.",
    "The capital city of France is Paris, which is known for the Eiffel Tower.",
    "Deep learning has revolutionized the field of computer vision in recent years."
  ];

  /**
   * Creates a new HF TEI Unified API client
   * @param options Configuration options
   * @param metadata API metadata including API keys
   */
  constructor(options: HfTeiUnifiedOptions = {}, metadata: HfTeiUnifiedApiMetadata = {}) {
    super(options, metadata);
    
    this.options = {
      apiUrl: 'https://api-inference.huggingface.co/models',
      containerUrl: metadata.hf_container_url || 'http://localhost:8080',
      maxRetries: 3,
      requestTimeout: 30000,
      useRequestQueue: true,
      debug: false,
      useContainer: false,
      dockerRegistry: metadata.docker_registry || 'ghcr.io/huggingface/text-embeddings-inference',
      containerTag: metadata.container_tag || 'latest',
      gpuDevice: metadata.gpu_device || '0',
      ...options
    };
    
    this.baseApiUrl = this.options.apiUrl as string;
    this.containerUrl = this.options.containerUrl as string;
    this.useContainer = this.options.useContainer as boolean;
    this.dockerRegistry = this.options.dockerRegistry as string;
    this.containerTag = this.options.containerTag as string;
    this.gpuDevice = this.options.gpuDevice as string;
    
    this.apiKey = this.getApiKey('hf_api_key', 'HF_API_KEY');
    this.modelId = metadata.model_id || 'BAAI/bge-small-en-v1.5';
    
    // Initialize circuit breaker for API requests
    this.circuitBreaker = new CircuitBreaker({
      failureThreshold: 3,
      resetTimeout: 30000,
      name: 'hf_tei_unified'
    });
  }

  /**
   * Get the default model for HF TEI
   * @returns The default model ID
   */
  getDefaultModel(): string {
    return this.modelId;
  }

  /**
   * Check if the provided model is compatible with HF TEI
   * @param model Model ID to check
   * @returns Whether the model is compatible
   */
  isCompatibleModel(model: string): boolean {
    // Models compatible with text-embeddings-inference
    const knownCompatibleModels = [
      'BAAI/bge-small-en-v1.5',
      'BAAI/bge-base-en-v1.5',
      'BAAI/bge-large-en-v1.5',
      'sentence-transformers/all-MiniLM-L6-v2',
      'sentence-transformers/all-mpnet-base-v2',
      'intfloat/e5-base-v2',
      'intfloat/e5-large-v2',
      'thenlper/gte-base',
      'thenlper/gte-large',
      'jinaai/jina-embeddings-v2-base-en',
      'jinaai/jina-embeddings-v2-small-en',
      'microsoft/unixcoder-base'
    ];
    
    if (knownCompatibleModels.includes(model)) {
      return true;
    }
    
    // Check for common embedding model patterns
    const embeddingPatterns = [
      /bge-.*-v\d/i,
      /sentence-transformers/i,
      /all-.*-v\d/i,
      /e5-.*-v\d/i, 
      /gte-.*/i,
      /jina-embeddings/i,
      /multilingual-e5/i,
      /instructor/i
    ];
    
    return embeddingPatterns.some(pattern => pattern.test(model));
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
        if (!this.containerInfo && this.containerInfo?.status !== 'running') {
          return false;
        }
        
        // Use health endpoint for container
        const handler = this.createEndpointHandler();
        const response = await handler.makeRequest('', { 
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
   * Make a POST request to the HF TEI API
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
      throw new Error(`HF TEI API error (${response.status}): ${errorText}`);
    }
    
    return response.json();
  }

  /**
   * Start a container instance of HF TEI
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
      network: config?.network || 'bridge'
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
      
      const gpuArgs = deployConfig.gpuDevice 
        ? `--gpus device=${deployConfig.gpuDevice}` 
        : '';
      
      const networkArg = deployConfig.network 
        ? `--network=${deployConfig.network}` 
        : '';
      
      const containerName = `hf-tei-${Date.now()}`;
      const command = `docker run -d --name ${containerName} \
        -p ${deployConfig.port}:80 \
        ${gpuArgs} \
        ${envVars} \
        ${volumes} \
        ${networkArg} \
        ${imageName} \
        --model-id ${deployConfig.modelId}`;
      
      if (this.options.debug) {
        console.log(`Starting container with command: ${command}`);
      }
      
      const containerId = execSync(command).toString().trim();
      
      // Wait for container to start
      await new Promise(resolve => setTimeout(resolve, 5000));
      
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
      
      throw new Error(`Failed to start HF TEI container: ${errorMessage}`);
    }
  }

  /**
   * Stop and remove the HF TEI container
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
   * Get model information (dim, revision, etc.)
   * @param modelId Optional model ID, defaults to the instance model ID
   * @returns Model information
   */
  async getModelInfo(modelId?: string): Promise<HfTeiModelInfo> {
    const model = modelId || this.modelId;
    
    if (this.useContainer) {
      // For container mode, use API endpoint to get model info
      const handler = this.createEndpointHandler();
      const response = await handler.makeRequest('', { 
        method: 'GET' 
      }, 'HIGH');
      
      if (!response.ok) {
        throw new Error(`Failed to get model info: ${response.statusText}`);
      }
      
      const data = await response.json();
      return {
        model_id: model,
        dim: data.dim || 0,
        status: data.status || 'unknown',
        revision: data.revision,
        framework: data.framework,
        quantized: data.quantized
      };
    } else {
      // For API mode, need to make a test embedding request to get dimensions
      const handler = this.createEndpointHandler();
      const endpoint = `/${encodeURIComponent(model)}`;
      
      const response = await handler.makeRequest(
        endpoint,
        {
          method: 'POST',
          body: JSON.stringify({ inputs: "Test embedding" }),
          headers: { 'Content-Type': 'application/json' }
        },
        'HIGH'
      );
      
      if (!response.ok) {
        throw new Error(`Failed to get model info: ${response.statusText}`);
      }
      
      const embeddings = await response.json();
      const dim = Array.isArray(embeddings) ? embeddings.length : 0;
      
      return {
        model_id: model,
        dim,
        status: 'ok'
      };
    }
  }

  /**
   * Generate embeddings for a text or array of texts
   * @param text Text or array of texts to embed
   * @param options Embedding options
   * @returns Array of embeddings
   */
  async generateEmbeddings(
    text: string | string[], 
    options: EmbeddingOptions = {}
  ): Promise<number[][]> {
    const model = options.model || this.modelId;
    const texts = Array.isArray(text) ? text : [text];
    
    const request: HfTeiEmbeddingRequest = {
      inputs: texts,
      normalize: options.normalize,
      truncation: options.truncation,
      max_tokens: options.maxTokens
    };
    
    // In container mode, endpoint is empty string
    // In API mode, endpoint includes the model name
    const endpoint = this.useContainer 
      ? '' 
      : `/${encodeURIComponent(model)}`;
    
    const response = await this.makePostRequest<HfTeiEmbeddingResponse>(
      endpoint,
      request,
      options.priority || 'NORMAL'
    );
    
    // Normalize response to always be an array of arrays
    if (Array.isArray(response) && response.length > 0) {
      // If it's a single embedding (array of numbers), wrap it in an array
      if (typeof response[0] === 'number') {
        return [response as number[]];
      }
      // Otherwise it's already an array of arrays
      return response as number[][];
    }
    
    throw new Error('Invalid response from embedding API');
  }

  /**
   * Generate embeddings for a batch of texts
   * @param texts Array of texts to embed
   * @param options Embedding options
   * @returns Array of embeddings
   */
  async batchEmbeddings(
    texts: string[], 
    options: EmbeddingOptions = {}
  ): Promise<number[][]> {
    return this.generateEmbeddings(texts, options);
  }
  
  /**
   * Run a benchmark on the embedding model
   * @param options Benchmark options
   * @returns Performance metrics
   */
  async runBenchmark(options: {
    iterations?: number;
    batchSize?: number;
    model?: string;
  } = {}): Promise<PerformanceMetrics> {
    const iterations = options.iterations || 10;
    const batchSize = options.batchSize || 5;
    const model = options.model || this.modelId;
    
    // Create test sentences if not enough provided
    const sampleTexts = [...this.testSentences];
    while (sampleTexts.length < batchSize) {
      sampleTexts.push(...this.testSentences);
    }
    
    // Use only as many as needed
    const texts = sampleTexts.slice(0, batchSize);
    
    // Benchmark single embedding
    const startSingle = performance.now();
    for (let i = 0; i < iterations; i++) {
      await this.generateEmbeddings(texts[0], { model });
    }
    const endSingle = performance.now();
    const singleTime = (endSingle - startSingle) / iterations;
    
    // Benchmark batch embedding
    const startBatch = performance.now();
    for (let i = 0; i < iterations; i++) {
      await this.batchEmbeddings(texts, { model });
    }
    const endBatch = performance.now();
    const batchTime = (endBatch - startBatch) / iterations;
    
    // Calculate metrics
    const sentencesPerSecond = texts.length / (batchTime / 1000);
    const batchSpeedupFactor = (singleTime * texts.length) / batchTime;
    
    const result: PerformanceMetrics = {
      singleEmbeddingTime: singleTime,
      batchEmbeddingTime: batchTime,
      sentencesPerSecond,
      batchSpeedupFactor,
      timestamp: new Date().toISOString()
    };
    
    return result;
  }

  /**
   * Not directly supported for embedding-only API, 
   * throws an error to indicate this.
   */
  async chat(messages: ChatMessage[], options: ChatOptions = {}): Promise<StreamChunk> {
    throw new Error('Chat completion not supported for HF TEI Unified. Use generateEmbeddings instead.');
  }

  /**
   * Not directly supported for embedding-only API,
   * throws an error to indicate this.
   */
  async *streamChat(messages: ChatMessage[], options: ChatOptions = {}): AsyncGenerator<StreamChunk> {
    throw new Error('Streaming chat completion not supported for HF TEI Unified. Use generateEmbeddings instead.');
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