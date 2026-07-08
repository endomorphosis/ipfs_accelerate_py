// Tests for HfTeiUnified API Backend
import { HfTeiUnified } from '../../src/api_backends/hf_tei_unified/hf_tei_unified';
import { 
  HfTeiUnifiedOptions, 
  HfTeiUnifiedApiMetadata,
  HfTeiEmbeddingRequest,
  HfTeiEmbeddingResponse,
  EmbeddingOptions
} from '../../src/api_backends/hf_tei_unified/types';
import { ApiMetadata, ChatMessage, PriorityLevel } from '../../src/api_backends/types';

// Mock child_process for container tests
jest.mock('child_process', () => ({
  execSync: jest.fn().mockReturnValue(Buffer.from('container-id')),
  spawn: jest.fn()
}));

// Mock os for platform checks
jest.mock('os', () => ({
  platform: jest.fn().mockReturnValue('linux')
}));

// Mock fs for file operations
jest.mock('fs', () => ({
  existsSync: jest.fn().mockReturnValue(true),
  writeFileSync: jest.fn(),
  readFileSync: jest.fn().mockReturnValue(Buffer.from('{}'))
}));

// Mock fetch for testing
global.fetch = jest.fn();

// Mock Response and Headers from node-fetch
const { Response, Headers } = require('node-fetch');
global.Response = Response;
global.Headers = Headers;

// Mock performance.now() for benchmarking tests
const originalPerformanceNow = global.performance.now;
jest.spyOn(global.performance, 'now')
  .mockImplementationOnce(() => 0)
  .mockImplementationOnce(() => 500)
  .mockImplementationOnce(() => 1000)
  .mockImplementationOnce(() => 2000);

describe('HfTeiUnified API Backend', () => {
  let backend: HfTeiUnified;
  let mockMetadata: HfTeiUnifiedApiMetadata;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock successful fetch response for embeddings
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    });
    
    // Set up test data
    mockMetadata = {
      hf_api_key: 'test-api-key',
      model_id: 'BAAI/bge-small-en-v1.5'
    };
    
    // Create backend instance in API mode (default)
    backend = new HfTeiUnified({}, mockMetadata);
  });
  
  afterAll(() => {
    // Restore original performance.now
    global.performance.now = originalPerformanceNow;
  });
  
  // Core Functionality Tests
  
  test('should initialize correctly', () => {
    expect(backend).toBeDefined();
    expect(backend.getDefaultModel()).toBe('BAAI/bge-small-en-v1.5');
  });
  
  test('should get API key from metadata', () => {
    // Testing the getApiKey method
    expect(backend['apiKey']).toBe('test-api-key');
    
    // Create backend with different key formats
    const backend1 = new HfTeiUnified({}, { hf_tei_api_key: 'specific-key' });
    expect(backend1['apiKey']).toBe('specific-key');
    
    const backend2 = new HfTeiUnified({}, { hf_api_key: 'general-key' });
    expect(backend2['apiKey']).toBe('general-key');
  });
  
  test('should initialize with custom options', () => {
    const customOptions: HfTeiUnifiedOptions = {
      apiUrl: 'https://custom-api.example.com',
      containerUrl: 'http://localhost:8888',
      maxRetries: 5,
      requestTimeout: 120000,
      useContainer: true,
      dockerRegistry: 'custom-registry',
      containerTag: 'custom-tag',
      gpuDevice: '1'
    };
    
    const customBackend = new HfTeiUnified(customOptions, mockMetadata);
    
    // Verify options were set
    expect(customBackend['baseApiUrl']).toBe('https://custom-api.example.com');
    expect(customBackend['containerUrl']).toBe('http://localhost:8888');
    expect(customBackend['useContainer']).toBe(true);
    expect(customBackend['dockerRegistry']).toBe('custom-registry');
    expect(customBackend['containerTag']).toBe('custom-tag');
    expect(customBackend['gpuDevice']).toBe('1');
  });
  
  test('should create endpoint handler', () => {
    const handler = backend.createEndpointHandler();
    expect(handler).toBeDefined();
    expect(handler.makeRequest).toBeDefined();
    expect(typeof handler.makeRequest).toBe('function');
  });
  
  test('should test endpoint successfully in API mode', async () => {
    // Using API mode (default)
    const result = await backend.testEndpoint();
    expect(result).toBe(true);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('BAAI/bge-small-en-v1.5'),
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          'Authorization': 'Bearer test-api-key'
        })
      })
    );
  });
  
  test('should test endpoint with failure', async () => {
    // Mock a failed response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 404,
      statusText: 'Not Found'
    });
    
    const result = await backend.testEndpoint();
    expect(result).toBe(false);
  });
  
  test('should test endpoint with error', async () => {
    // Mock an error
    (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));
    
    const result = await backend.testEndpoint();
    expect(result).toBe(false);
  });
  
  test('should make POST request with correct parameters', async () => {
    const endpoint = '/test-endpoint';
    const data = { inputs: 'test text' };
    const priority: PriorityLevel = 'HIGH';
    
    const response = await backend.makePostRequest(endpoint, data, priority);
    
    expect(response).toBeDefined();
    expect(Array.isArray(response)).toBe(true);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining(endpoint),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify(data),
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          'Authorization': 'Bearer test-api-key'
        })
      })
    );
  });
  
  test('should handle API errors in makePostRequest', async () => {
    // Mock error responses for different status codes
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 404,
      text: jest.fn().mockResolvedValue('Model not found')
    });
    
    await expect(backend.makePostRequest('/test', {})).rejects.toThrow('Model not found');
    
    // 403 error
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 403,
      text: jest.fn().mockResolvedValue('Unauthorized')
    });
    
    await expect(backend.makePostRequest('/test', {})).rejects.toThrow('Authorization error');
    
    // 503 error
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 503,
      text: jest.fn().mockResolvedValue('Model loading')
    });
    
    await expect(backend.makePostRequest('/test', {})).rejects.toThrow('The model is currently loading or busy');
    
    // Generic error
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: jest.fn().mockResolvedValue('Internal server error')
    });
    
    await expect(backend.makePostRequest('/test', {})).rejects.toThrow('HF TEI API error (500)');
  });

  // Embedding Tests
  
  test('should generate embeddings for a single text', async () => {
    const text = 'This is a test text';
    const options: EmbeddingOptions = {
      normalize: true
    };
    
    const response = await backend.generateEmbeddings(text, options);
    
    expect(response).toBeDefined();
    expect(Array.isArray(response)).toBe(true);
    expect(response.length).toBe(1); // One embedding vector
    expect(Array.isArray(response[0])).toBe(true);
    expect(response[0].length).toBeGreaterThan(0); // Vector has dimensions
    
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('BAAI/bge-small-en-v1.5'),
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining(text),
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          'Authorization': 'Bearer test-api-key'
        })
      })
    );
  });
  
  test('should generate embeddings with custom model', async () => {
    const text = 'Generate embeddings with custom model';
    const options = {
      model: 'sentence-transformers/all-MiniLM-L6-v2',
      normalize: true
    };
    
    const response = await backend.generateEmbeddings(text, options);
    
    expect(response).toBeDefined();
    expect(Array.isArray(response)).toBe(true);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('sentence-transformers/all-MiniLM-L6-v2'),
      expect.anything()
    );
  });
  
  test('should handle batch embeddings', async () => {
    // Mock batch response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
      ])
    });
    
    const texts = ['Text 1', 'Text 2'];
    const options: EmbeddingOptions = {
      normalize: true,
      truncation: true,
      maxTokens: 100
    };
    
    const response = await backend.generateEmbeddings(texts, options);
    
    expect(response).toBeDefined();
    expect(Array.isArray(response)).toBe(true);
    expect(response.length).toBe(2); // Two embedding vectors
    expect(Array.isArray(response[0])).toBe(true);
    expect(response[0].length).toBe(4); // Vector has dimensions
    
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('BAAI/bge-small-en-v1.5'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          inputs: texts,
          normalize: true,
          truncation: true,
          max_tokens: 100
        }),
        headers: expect.any(Object)
      })
    );
  });
  
  test('should handle various embedding response formats', async () => {
    // Test with array response (single embedding)
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue([0.1, 0.2, 0.3, 0.4])
    });
    
    const singleResponse = await backend.generateEmbeddings('Single text');
    expect(singleResponse).toBeDefined();
    expect(singleResponse.length).toBe(1);
    expect(singleResponse[0].length).toBe(4);
    
    // Test with array of arrays response (batch embeddings)
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
      ])
    });
    
    const batchResponse = await backend.generateEmbeddings(['Text 1', 'Text 2']);
    expect(batchResponse).toBeDefined();
    expect(batchResponse.length).toBe(2);
    expect(batchResponse[0].length).toBe(4);
  });
  
  test('should handle invalid embedding response', async () => {
    // Mock invalid response format
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        unexpected: 'format'
      })
    });
    
    await expect(backend.generateEmbeddings('Test text')).rejects.toThrow('Invalid response from embedding API');
  });

  test('should use batchEmbeddings for multiple texts', async () => {
    // Spy on generateEmbeddings to see if it's called by batchEmbeddings
    const spy = jest.spyOn(backend, 'generateEmbeddings');
    
    const texts = ['Text 1', 'Text 2', 'Text 3'];
    const options: EmbeddingOptions = {
      normalize: true
    };
    
    await backend.batchEmbeddings(texts, options);
    
    expect(spy).toHaveBeenCalledWith(texts, options);
    
    // Clean up
    spy.mockRestore();
  });
  
  // Container Mode Tests
  
  test('should start container successfully', async () => {
    // Mock successful container start
    const { execSync } = require('child_process');
    execSync.mockReturnValueOnce('container-id\n');
    
    const containerInfo = await backend.startContainer();
    
    expect(containerInfo).toBeDefined();
    expect(containerInfo.status).toBe('running');
    expect(containerInfo.containerId).toBe('container-id');
    expect(execSync).toHaveBeenCalled();
    expect(execSync.mock.calls[0][0]).toContain('docker run');
    expect(execSync.mock.calls[0][0]).toContain('--model-id BAAI/bge-small-en-v1.5');
  });
  
  test('should start container with custom configuration', async () => {
    // Mock successful container start
    const { execSync } = require('child_process');
    execSync.mockReturnValueOnce('custom-container-id\n');
    
    const containerInfo = await backend.startContainer({
      dockerRegistry: 'custom-registry',
      containerTag: 'custom-tag',
      gpuDevice: '1,2',
      modelId: 'custom-model',
      port: 8888,
      env: { CUDA_VISIBLE_DEVICES: '1,2' },
      volumes: ['/host/path:/container/path'],
      network: 'host'
    });
    
    expect(containerInfo).toBeDefined();
    expect(containerInfo.status).toBe('running');
    expect(containerInfo.containerId).toBe('custom-container-id');
    expect(execSync).toHaveBeenCalled();
    
    // Check that custom parameters were passed correctly
    const command = execSync.mock.calls[0][0];
    expect(command).toContain('custom-registry:custom-tag');
    expect(command).toContain('--gpus device=1,2');
    expect(command).toContain('-p 8888:80');
    expect(command).toContain('-e CUDA_VISIBLE_DEVICES=1,2');
    expect(command).toContain('-v /host/path:/container/path');
    expect(command).toContain('--network=host');
    expect(command).toContain('--model-id custom-model');
  });
  
  test('should handle container start error', async () => {
    // Mock failed container start
    const { execSync } = require('child_process');
    execSync.mockImplementationOnce(() => {
      throw new Error('Docker error');
    });
    
    await expect(backend.startContainer()).rejects.toThrow('Failed to start HF TEI container: Docker error');
    
    // Container info should be set to failed status
    expect(backend['containerInfo']).toBeDefined();
    expect(backend['containerInfo']?.status).toBe('failed');
    expect(backend['containerInfo']?.error).toBeDefined();
  });
  
  test('should stop container successfully', async () => {
    // Set up container info
    backend['containerInfo'] = {
      containerId: 'test-container-id',
      host: 'localhost',
      port: 8080,
      status: 'running',
      startTime: new Date()
    };
    
    const { execSync } = require('child_process');
    
    const result = await backend.stopContainer();
    
    expect(result).toBe(true);
    expect(execSync).toHaveBeenCalledWith(
      'docker stop test-container-id',
      expect.anything()
    );
    expect(execSync).toHaveBeenCalledWith(
      'docker rm test-container-id',
      expect.anything()
    );
    expect(backend['containerInfo']).toBeNull();
  });
  
  test('should handle container stop when no container is running', async () => {
    backend['containerInfo'] = null;
    
    const result = await backend.stopContainer();
    
    expect(result).toBe(true);
    const { execSync } = require('child_process');
    expect(execSync).not.toHaveBeenCalled();
  });
  
  test('should handle container stop error', async () => {
    // Set up container info
    backend['containerInfo'] = {
      containerId: 'test-container-id',
      host: 'localhost',
      port: 8080,
      status: 'running',
      startTime: new Date()
    };
    
    // Mock error on container stop
    const { execSync } = require('child_process');
    execSync.mockImplementationOnce(() => {
      throw new Error('Docker stop error');
    });
    
    // Even with error, should return false
    const result = await backend.stopContainer();
    expect(result).toBe(false);
  });
  
  test('should test endpoint in container mode', async () => {
    // Set backend to container mode
    backend['useContainer'] = true;
    backend['containerInfo'] = {
      containerId: 'test-container-id',
      host: 'localhost',
      port: 8080,
      status: 'running',
      startTime: new Date()
    };
    
    // Mock successful health check
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true
    });
    
    const result = await backend.testEndpoint();
    
    expect(result).toBe(true);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('http://localhost:8080'),
      expect.objectContaining({
        method: 'GET'
      })
    );
  });
  
  test('should fail endpoint test in container mode if container not running', async () => {
    // Set backend to container mode but no running container
    backend['useContainer'] = true;
    backend['containerInfo'] = null;
    
    const result = await backend.testEndpoint();
    
    expect(result).toBe(false);
    expect(global.fetch).not.toHaveBeenCalled();
  });
  
  test('should get model info in API mode', async () => {
    // Mock successful model info response for API mode
    // For embedding models, we need to return an array to get the dimensions
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    });
    
    const modelInfo = await backend.getModelInfo();
    
    expect(modelInfo).toBeDefined();
    expect(modelInfo.model_id).toBe('BAAI/bge-small-en-v1.5');
    expect(modelInfo.status).toBe('ok');
    expect(modelInfo.dim).toBe(8); // Number of dimensions in the embedding
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('BAAI/bge-small-en-v1.5'),
      expect.objectContaining({
        method: 'POST'
      })
    );
  });
  
  test('should get model info in container mode', async () => {
    // Set backend to container mode
    backend['useContainer'] = true;
    backend['containerInfo'] = {
      containerId: 'test-container-id',
      host: 'localhost',
      port: 8080,
      status: 'running',
      startTime: new Date()
    };
    
    // Mock successful info response for container mode
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        model_id: 'BAAI/bge-small-en-v1.5',
        dim: 768,
        status: 'loaded',
        revision: 'abc123',
        framework: 'pytorch',
        quantized: false
      })
    });
    
    const modelInfo = await backend.getModelInfo();
    
    expect(modelInfo).toBeDefined();
    expect(modelInfo.model_id).toBe('BAAI/bge-small-en-v1.5');
    expect(modelInfo.dim).toBe(768);
    expect(modelInfo.status).toBe('loaded');
    expect(modelInfo.revision).toBe('abc123');
    expect(modelInfo.framework).toBe('pytorch');
    expect(modelInfo.quantized).toBe(false);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('http://localhost:8080'),
      expect.objectContaining({
        method: 'GET'
      })
    );
  });
  
  test('should generate embeddings in container mode', async () => {
    // Set backend to container mode
    backend['useContainer'] = true;
    backend['containerInfo'] = {
      containerId: 'test-container-id',
      host: 'localhost',
      port: 8080,
      status: 'running',
      startTime: new Date()
    };
    
    const text = 'Generate embeddings in container mode';
    const response = await backend.generateEmbeddings(text);
    
    expect(response).toBeDefined();
    expect(Array.isArray(response)).toBe(true);
    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8080',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining(text)
      })
    );
  });
  
  test('should run benchmark successfully', async () => {
    // Mock generateEmbeddings to return predictable response for benchmarking
    jest.spyOn(backend, 'generateEmbeddings').mockResolvedValue([Array(768).fill(0.1)]);
    jest.spyOn(backend, 'batchEmbeddings').mockResolvedValue([Array(768).fill(0.1), Array(768).fill(0.2)]);
    
    const result = await backend.runBenchmark({
      iterations: 3,
      batchSize: 2,
      model: 'BAAI/bge-small-en-v1.5'
    });
    
    expect(result).toBeDefined();
    expect(result.singleEmbeddingTime).toBeGreaterThan(0);
    expect(result.batchEmbeddingTime).toBeGreaterThan(0);
    expect(result.sentencesPerSecond).toBeGreaterThan(0);
    expect(result.batchSpeedupFactor).toBeGreaterThan(0);
    expect(result.timestamp).toBeDefined();
    
    // Test with default values
    await backend.runBenchmark();
  });
  
  test('should switch between API and container modes', () => {
    // Start in API mode
    expect(backend.getMode()).toBe('api');
    
    // Switch to container mode
    backend.setMode(true);
    expect(backend.getMode()).toBe('container');
    
    // Switch back to API mode
    backend.setMode(false);
    expect(backend.getMode()).toBe('api');
  });
  
  // Model Compatibility Test
  
  test('should check model compatibility', () => {
    // Known compatible models
    expect(backend.isCompatibleModel('BAAI/bge-small-en-v1.5')).toBe(true);
    expect(backend.isCompatibleModel('sentence-transformers/all-MiniLM-L6-v2')).toBe(true);
    expect(backend.isCompatibleModel('intfloat/e5-base-v2')).toBe(true);
    
    // Models matching patterns
    expect(backend.isCompatibleModel('random/bge-large-v1')).toBe(true);
    expect(backend.isCompatibleModel('random/sentence-transformers-model')).toBe(true);
    expect(backend.isCompatibleModel('random/e5-small-v2')).toBe(true);
    
    // Incompatible models
    expect(backend.isCompatibleModel('llama-7b')).toBe(false);
    expect(backend.isCompatibleModel('gpt2')).toBe(false);
    expect(backend.isCompatibleModel('t5-base')).toBe(false);
  });
  
  // Error Cases
  
  test('should throw error for chat method', async () => {
    await expect(backend.chat([{ role: 'user', content: 'Hello' }])).rejects.toThrow(
      'Chat completion not supported for HF TEI Unified'
    );
  });
  
  test('should throw error for streamChat method', async () => {
    const messages = [{ role: 'user', content: 'Hello' }];
    const streamChat = backend.streamChat(messages);
    
    await expect(async () => {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      for await (const chunk of streamChat) {
        // This should not be reached
      }
    }).rejects.toThrow('Streaming chat completion not supported for HF TEI Unified');
  });
});