// Tests for HfTgiUnified API Backend
import { HfTgiUnified } from '../../src/api_backends/hf_tgi_unified/hf_tgi_unified';
import { 
  HfTgiUnifiedOptions, 
  HfTgiUnifiedApiMetadata,
  HfTgiTextGenerationRequest,
  HfTgiTextGenerationResponse,
  HfTgiStreamChunk
} from '../../src/api_backends/hf_tgi_unified/types';
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

// Mock TextEncoder and TextDecoder for streaming response tests
global.TextEncoder = require('util').TextEncoder;
global.TextDecoder = require('util').TextDecoder;

// Mock Response and Headers from node-fetch
const { Response, Headers } = require('node-fetch');
global.Response = Response;
global.Headers = Headers;

describe('HfTgiUnified API Backend', () => {
  let backend: HfTgiUnified;
  let mockMetadata: HfTgiUnifiedApiMetadata;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock successful fetch response for text generation
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        generated_text: 'This is a generated response',
        details: {
          finish_reason: 'length',
          generated_tokens: 20
        }
      })
    });
    
    // Set up test data
    mockMetadata = {
      hf_api_key: 'test-api-key',
      model_id: 'google/flan-t5-small'
    };
    
    // Create backend instance in API mode (default)
    backend = new HfTgiUnified({}, mockMetadata);
  });
  
  // Core Functionality Tests
  
  test('should initialize correctly', () => {
    expect(backend).toBeDefined();
    expect(backend.getDefaultModel()).toBe('google/flan-t5-small');
  });
  
  test('should get API key from metadata', () => {
    // Testing the getApiKey method
    expect(backend['apiKey']).toBe('test-api-key');
    
    // Create backend with different key formats
    const backend1 = new HfTgiUnified({}, { hf_tgi_api_key: 'specific-key' });
    expect(backend1['apiKey']).toBe('specific-key');
    
    const backend2 = new HfTgiUnified({}, { hf_api_key: 'general-key' });
    expect(backend2['apiKey']).toBe('general-key');
  });
  
  test('should initialize with custom options', () => {
    const customOptions: HfTgiUnifiedOptions = {
      apiUrl: 'https://custom-api.example.com',
      containerUrl: 'http://localhost:8888',
      maxRetries: 5,
      requestTimeout: 120000,
      useContainer: true,
      dockerRegistry: 'custom-registry',
      containerTag: 'custom-tag',
      gpuDevice: '1',
      maxTokens: 1024,
      temperature: 0.5,
      topP: 0.8,
      topK: 40,
      repetitionPenalty: 1.2,
      stopSequences: ['END']
    };
    
    const customBackend = new HfTgiUnified(customOptions, mockMetadata);
    
    // Verify options were set
    expect(customBackend['baseApiUrl']).toBe('https://custom-api.example.com');
    expect(customBackend['containerUrl']).toBe('http://localhost:8888');
    expect(customBackend['useContainer']).toBe(true);
    expect(customBackend['dockerRegistry']).toBe('custom-registry');
    expect(customBackend['containerTag']).toBe('custom-tag');
    expect(customBackend['gpuDevice']).toBe('1');
    expect(customBackend['options'].maxTokens).toBe(1024);
    expect(customBackend['options'].temperature).toBe(0.5);
    expect(customBackend['options'].topP).toBe(0.8);
    expect(customBackend['options'].topK).toBe(40);
    expect(customBackend['options'].repetitionPenalty).toBe(1.2);
    expect(customBackend['options'].stopSequences).toEqual(['END']);
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
      expect.stringContaining('google/flan-t5-small'),
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
    const data = { key: 'value' };
    const priority: PriorityLevel = 'HIGH';
    
    const response = await backend.makePostRequest(endpoint, data, priority);
    
    expect(response).toBeDefined();
    expect(response.generated_text).toBe('This is a generated response');
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
    
    await expect(backend.makePostRequest('/test', {})).rejects.toThrow('HF TGI API error (500)');
  });

  // Text Generation Tests
  
  test('should generate text with API mode', async () => {
    const text = 'Generate a response to this prompt';
    const options = {
      maxTokens: 100,
      temperature: 0.7,
      topP: 0.9
    };
    
    const response = await backend.generateText(text, options);
    
    expect(response).toBe('This is a generated response');
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('google/flan-t5-small'),
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
  
  test('should generate text with custom model', async () => {
    const text = 'Generate with custom model';
    const options = {
      model: 'facebook/opt-125m',
      maxTokens: 100
    };
    
    const response = await backend.generateText(text, options);
    
    expect(response).toBe('This is a generated response');
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('facebook/opt-125m'),
      expect.anything()
    );
  });
  
  test('should collect output from streaming text generation', async () => {
    // Mock the streamGenerateText method to return an async generator
    const mockStream = async function* () {
      yield { text: 'This ', done: false };
      yield { text: 'is ', done: false };
      yield { text: 'a ', done: false };
      yield { text: 'streamed ', done: false };
      yield { text: 'response.', done: true, fullText: 'This is a streamed response.' };
    };
    
    jest.spyOn(backend, 'streamGenerateText').mockImplementation(() => mockStream());
    
    const text = 'Generate a streaming response';
    const options = {
      stream: true
    };
    
    const response = await backend.generateText(text, options);
    
    expect(response).toBe('This is a streamed response.');
    expect(backend.streamGenerateText).toHaveBeenCalledWith(text, options);
  });
  
  test('should handle streaming text generation from API', async () => {
    // Create a mock implementation for streamGenerateText
    // This is a more complex test as it involves async generators
    
    // Mock response for API mode simulation
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      text: jest.fn().mockResolvedValue(JSON.stringify({
        generated_text: 'This is a simulated response'
      }))
    });
    
    const text = 'Generate a streaming response';
    const streamGenerator = backend.streamGenerateText(text, { stream: true });
    
    // Collect all chunks
    const chunks = [];
    for await (const chunk of streamGenerator) {
      chunks.push(chunk);
    }
    
    // Verify we got chunks with the correct structure
    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks[0].text).toBeDefined();
    
    // Last chunk should have done=true
    const lastChunk = chunks[chunks.length - 1];
    expect(lastChunk.done).toBe(true);
  });
  
  // Chat Tests
  
  test('should generate chat response', async () => {
    const messages: ChatMessage[] = [
      { role: 'user', content: 'Hello, how are you?' }
    ];
    
    const response = await backend.chat(messages);
    
    expect(response).toBeDefined();
    expect(response.text).toBe('This is a generated response');
    expect(response.done).toBe(true);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('google/flan-t5-small'),
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('Hello, how are you?')
      })
    );
  });
  
  test('should reject chat with empty messages', async () => {
    const messages: ChatMessage[] = [];
    
    await expect(backend.chat(messages)).rejects.toThrow('Chat messages array cannot be empty');
  });
  
  test('should format chat prompts correctly for different models', async () => {
    // Create spy to inspect formatChatPrompt method
    const spy = jest.spyOn(backend as any, 'formatChatPrompt');
    
    // Test with Llama model format
    const llamaBackend = new HfTgiUnified({}, { model_id: 'meta-llama/Llama-2-7b-chat-hf' });
    await llamaBackend.chat([{ role: 'user', content: 'Hello' }]);
    expect(spy).toHaveBeenCalled();
    
    // Test format for Mistral
    const mistralBackend = new HfTgiUnified({}, { model_id: 'mistralai/Mistral-7B-Instruct-v0.1' });
    await mistralBackend.chat([{ role: 'user', content: 'Hello' }]);
    expect(spy).toHaveBeenCalled();
    
    // Test format for ChatML
    const chatmlBackend = new HfTgiUnified({}, { model_id: 'openchat/openchat-3.5' });
    await chatmlBackend.chat([{ role: 'user', content: 'Hello' }]);
    expect(spy).toHaveBeenCalled();
    
    // Clean up spy
    spy.mockRestore();
  });
  
  test('should format multi-turn chat conversation correctly', async () => {
    const spy = jest.spyOn(backend as any, 'formatChatPrompt');
    
    const messages: ChatMessage[] = [
      { role: 'user', content: 'Hello, how are you?' },
      { role: 'assistant', content: 'I am doing well, thank you for asking. How can I help you today?' },
      { role: 'user', content: 'Tell me about language models' }
    ];
    
    await backend.chat(messages);
    
    // Verify formatChatPrompt was called correctly
    expect(spy).toHaveBeenCalledWith(messages, expect.anything());
    
    // Clean up spy
    spy.mockRestore();
  });
  
  test('should stream chat response', async () => {
    // Mock the streamGenerateText method
    const mockStream = async function* () {
      yield { text: 'This ', done: false };
      yield { text: 'is ', done: false };
      yield { text: 'a ', done: false };
      yield { text: 'streamed ', done: false };
      yield { text: 'response.', done: true, fullText: 'This is a streamed response.' };
    };
    
    jest.spyOn(backend, 'streamGenerateText').mockImplementation(() => mockStream());
    
    const messages: ChatMessage[] = [
      { role: 'user', content: 'Stream a response to me' }
    ];
    
    const streamGenerator = backend.streamChat(messages);
    
    // Collect all chunks
    const chunks = [];
    for await (const chunk of streamGenerator) {
      chunks.push(chunk);
    }
    
    // Verify we got the expected chunks
    expect(chunks.length).toBe(5);
    expect(chunks[0].text).toBe('This ');
    expect(chunks[4].text).toBe('response.');
    expect(chunks[4].done).toBe(true);
  });
  
  test('should reject streamChat with empty messages', async () => {
    const messages: ChatMessage[] = [];
    
    const streamGenerator = backend.streamChat(messages);
    
    await expect(async () => {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      for await (const chunk of streamGenerator) {
        // This should not be reached
      }
    }).rejects.toThrow('Chat messages array cannot be empty');
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
    expect(execSync.mock.calls[0][0]).toContain('--model-id google/flan-t5-small');
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
      network: 'host',
      parameters: ['--quantize'],
      maxInputLength: 4096
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
    expect(command).toContain('--max-input-length 4096');
    expect(command).toContain('--quantize');
  });
  
  test('should handle container start error', async () => {
    // Mock failed container start
    const { execSync } = require('child_process');
    execSync.mockImplementationOnce(() => {
      throw new Error('Docker error');
    });
    
    await expect(backend.startContainer()).rejects.toThrow('Failed to start HF TGI container: Docker error');
    
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
      expect.stringContaining('/health'),
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
    // Mock successful model info response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        sha: 'abc123',
        framework: 'pytorch',
        parameters: { max_length: 512 }
      })
    });
    
    const modelInfo = await backend.getModelInfo();
    
    expect(modelInfo).toBeDefined();
    expect(modelInfo.model_id).toBe('google/flan-t5-small');
    expect(modelInfo.status).toBe('ok');
    expect(modelInfo.revision).toBe('abc123');
    expect(modelInfo.framework).toBe('pytorch');
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('google/flan-t5-small'),
      expect.objectContaining({
        method: 'GET'
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
    
    // Mock successful info response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        model_id: 'google/flan-t5-small',
        framework: 'pytorch',
        max_input_length: 2048,
        max_total_tokens: 4096,
        parameters: { max_length: 512 }
      })
    });
    
    const modelInfo = await backend.getModelInfo();
    
    expect(modelInfo).toBeDefined();
    expect(modelInfo.model_id).toBe('google/flan-t5-small');
    expect(modelInfo.framework).toBe('pytorch');
    expect(modelInfo.max_input_length).toBe(2048);
    expect(modelInfo.max_total_tokens).toBe(4096);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/info'),
      expect.objectContaining({
        method: 'GET'
      })
    );
  });
  
  test('should generate text in container mode', async () => {
    // Set backend to container mode
    backend['useContainer'] = true;
    backend['containerInfo'] = {
      containerId: 'test-container-id',
      host: 'localhost',
      port: 8080,
      status: 'running',
      startTime: new Date()
    };
    
    const text = 'Generate a response in container mode';
    const response = await backend.generateText(text);
    
    expect(response).toBe('This is a generated response');
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/generate'),
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining(text)
      })
    );
  });
  
  test('should run benchmark successfully', async () => {
    // Mock generateText to return predictable response for benchmarking
    jest.spyOn(backend, 'generateText').mockResolvedValue('This is a benchmark test response with 10 tokens');
    
    const result = await backend.runBenchmark({
      iterations: 3,
      model: 'google/flan-t5-small',
      maxTokens: 50
    });
    
    expect(result).toBeDefined();
    expect(result.singleGenerationTime).toBeGreaterThan(0);
    expect(result.tokensPerSecond).toBeGreaterThan(0);
    expect(result.generatedTokens).toBeGreaterThan(0);
    expect(result.inputTokens).toBeGreaterThan(0);
    expect(result.timestamp).toBeDefined();
    expect(backend.generateText).toHaveBeenCalledTimes(3);
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
    expect(backend.isCompatibleModel('google/flan-t5-small')).toBe(true);
    expect(backend.isCompatibleModel('facebook/opt-125m')).toBe(true);
    expect(backend.isCompatibleModel('meta-llama/Llama-2-7b-chat-hf')).toBe(true);
    expect(backend.isCompatibleModel('mistralai/Mistral-7B-v0.1')).toBe(true);
    
    // Models matching patterns
    expect(backend.isCompatibleModel('random/flan-t5-model')).toBe(true);
    expect(backend.isCompatibleModel('random/opt-model')).toBe(true);
    expect(backend.isCompatibleModel('random/llama-model')).toBe(true);
    expect(backend.isCompatibleModel('random/mistral-model')).toBe(true);
    
    // Incompatible models
    expect(backend.isCompatibleModel('clip-vit')).toBe(false);
    expect(backend.isCompatibleModel('embedding-model')).toBe(false);
    expect(backend.isCompatibleModel('random-model-name')).toBe(false);
  });
  
  // Request Formatting Tests
  
  test('should format text generation request correctly', () => {
    const formatRequest = backend['formatRequest'].bind(backend);
    
    // Basic request
    const request1 = formatRequest('Test prompt');
    expect(request1.inputs).toBe('Test prompt');
    expect(request1.parameters.max_new_tokens).toBe(512); // Default from options
    
    // Request with custom parameters
    const request2 = formatRequest('Test prompt', {
      maxTokens: 100,
      temperature: 0.5,
      topP: 0.8,
      topK: 40,
      repetitionPenalty: 1.2,
      includeInputText: true,
      stream: true,
      stopSequences: ['END']
    });
    
    expect(request2.inputs).toBe('Test prompt');
    expect(request2.parameters.max_new_tokens).toBe(100);
    expect(request2.parameters.temperature).toBe(0.5);
    expect(request2.parameters.top_p).toBe(0.8);
    expect(request2.parameters.top_k).toBe(40);
    expect(request2.parameters.repetition_penalty).toBe(1.2);
    expect(request2.parameters.return_full_text).toBe(true);
    expect(request2.parameters.stream).toBe(true);
    expect(request2.parameters.stop).toEqual(['END']);
  });
  
  test('should format chat prompt correctly for different models', () => {
    const formatChatPrompt = backend['formatChatPrompt'].bind(backend);
    
    // Test with default template
    const messages1: ChatMessage[] = [
      { role: 'user', content: 'Hello, how are you?' }
    ];
    const prompt1 = formatChatPrompt(messages1);
    expect(prompt1).toBe('Hello, how are you?');
    
    // Test with Llama2 template
    const prompt2 = formatChatPrompt(messages1, {
      promptTemplate: 'llama2'
    });
    expect(prompt2).toContain('<s>[INST] Hello, how are you? [/INST]');
    
    // Test with ChatML template
    const prompt3 = formatChatPrompt(messages1, {
      promptTemplate: 'chatml'
    });
    expect(prompt3).toContain('<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n');
    
    // Test with multi-turn conversation
    const messages2: ChatMessage[] = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there!' },
      { role: 'user', content: 'How are you?' }
    ];
    
    // For ChatML format
    const prompt4 = formatChatPrompt(messages2, {
      promptTemplate: 'chatml'
    });
    expect(prompt4).toContain('<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n');
    
    // Test with system message
    const prompt5 = formatChatPrompt(messages1, {
      promptTemplate: 'chat',
      systemMessage: 'You are a helpful assistant.'
    });
    expect(prompt5).toContain('You are a helpful assistant.');
    expect(prompt5).toContain('Hello, how are you?');
  });
});