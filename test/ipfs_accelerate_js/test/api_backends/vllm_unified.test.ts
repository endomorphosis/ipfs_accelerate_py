// Tests for VLLM Unified API Backend
import { VLLM } from '../../src/api_backends/vllm/vllm';
import { 
  VLLMRequestData, 
  VLLMResponse, 
  VLLMRequestOptions,
  VLLMModelInfo,
  VLLMLoraAdapter,
  VLLMQuantizationConfig
} from '../../src/api_backends/vllm/types';
import { ApiMetadata, Message, ApiRequestOptions, PriorityLevel } from '../../src/api_backends/types';

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

// Mock AbortSignal for timeout
global.AbortSignal = {
  timeout: jest.fn().mockReturnValue({
    aborted: false
  })
} as any;

// Mock TextEncoder and TextDecoder for streaming response tests
global.TextEncoder = require('util').TextEncoder;
global.TextDecoder = require('util').TextDecoder;

// Mock ReadableStream for testing streaming
const mockReadableStream = () => {
  const chunks = [
    `data: ${JSON.stringify({
      choices: [{ delta: { content: 'Hello', role: 'assistant' }, finish_reason: null }]
    })}`,
    `data: ${JSON.stringify({
      choices: [{ delta: { content: ', world!', role: 'assistant' }, finish_reason: null }]
    })}`,
    `data: ${JSON.stringify({
      choices: [{ delta: { content: '', role: 'assistant' }, finish_reason: 'stop' }]
    })}`,
    'data: [DONE]'
  ];
  
  let currentIndex = 0;
  
  return {
    getReader: () => ({
      read: () => {
        if (currentIndex < chunks.length) {
          const chunk = chunks[currentIndex];
          currentIndex++;
          return Promise.resolve({
            done: false,
            value: new TextEncoder().encode(chunk + '\n')
          });
        } else {
          return Promise.resolve({ done: true, value: undefined });
        }
      },
      cancel: jest.fn(),
      releaseLock: jest.fn()
    })
  };
};

describe('VLLM Unified API Backend', () => {
  let backend: VLLM;
  let mockMetadata: ApiMetadata;
  
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock successful fetch response
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        id: 'test-id',
        model: 'test-model',
        choices: [
          {
            message: {
              role: 'assistant',
              content: 'Hello, I am an AI assistant.'
            },
            finish_reason: 'stop'
          }
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 20,
          total_tokens: 30
        }
      })
    });
    
    // Set up test data
    mockMetadata = {
      vllm_api_key: 'test-api-key',
      vllm_api_url: 'https://vllm-api.example.com',
      vllm_model: 'meta-llama/Llama-2-7b-chat-hf'
    };
    
    // Create backend instance
    backend = new VLLM({}, mockMetadata);
  });
  
  // Core Functionality Tests
  
  test('should initialize correctly', () => {
    expect(backend).toBeDefined();
    
    // Test with custom API URL
    const customBackend = new VLLM({}, { 
      vllm_api_url: 'https://custom-vllm-url.com',
      vllm_model: 'custom-model'
    });
    expect(customBackend).toBeDefined();
    
    // @ts-ignore - Accessing protected property for testing
    expect(customBackend.apiUrl).toBe('https://custom-vllm-url.com');
    // @ts-ignore - Accessing protected property for testing
    expect(customBackend.defaultModel).toBe('custom-model');
  });
  
  test('should get API key from metadata', () => {
    // @ts-ignore - Testing protected method
    const apiKey = backend.getApiKey(mockMetadata);
    expect(apiKey).toBe('test-api-key');
    
    // Test environment variable fallback
    const originalEnv = process.env.VLLM_API_KEY;
    process.env.VLLM_API_KEY = 'env-api-key';
    // @ts-ignore - Testing protected method
    const envApiKey = backend.getApiKey({});
    expect(envApiKey).toBe('env-api-key');
    
    // Restore original environment
    process.env.VLLM_API_KEY = originalEnv;
  });
  
  test('should get default model', () => {
    // @ts-ignore - Testing protected method
    const model = backend.getDefaultModel();
    expect(model).toBeDefined();
    expect(typeof model).toBe('string');
    expect(model).toBe('meta-llama/Llama-2-7b-chat-hf');
  });
  
  test('should create endpoint handler', () => {
    const handler = backend.createEndpointHandler();
    expect(handler).toBeDefined();
    expect(typeof handler).toBe('function');
    
    // Test with custom URL
    const customHandler = backend.createEndpointHandler('https://custom-endpoint.com');
    expect(customHandler).toBeDefined();
    expect(typeof customHandler).toBe('function');
  });
  
  test('should create chat endpoint handler', () => {
    // @ts-ignore - Testing method not in interface but in implementation
    const handler = backend.createVLLMChatEndpointHandler();
    expect(handler).toBeDefined();
    expect(typeof handler).toBe('function');
    
    // Test with custom URL
    // @ts-ignore - Testing method not in interface but in implementation
    const customHandler = backend.createVLLMChatEndpointHandler('https://custom-chat-endpoint.com');
    expect(customHandler).toBeDefined();
    expect(typeof customHandler).toBe('function');
  });
  
  test('should test endpoint', async () => {
    const result = await backend.testEndpoint();
    expect(result).toBe(true);
    expect(global.fetch).toHaveBeenCalled();
    
    // Test with custom URL and model
    const customResult = await backend.testEndpoint('https://custom-endpoint.com', 'custom-model');
    expect(customResult).toBe(true);
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });
  
  test('should test chat endpoint', async () => {
    // @ts-ignore - Testing method not in interface but in implementation
    const result = await backend.testVLLMChatEndpoint();
    expect(result).toBe(true);
    expect(global.fetch).toHaveBeenCalled();
    
    // Test with custom URL and model
    // @ts-ignore - Testing method not in interface but in implementation
    const customResult = await backend.testVLLMChatEndpoint('https://custom-chat-endpoint.com', 'custom-model');
    expect(customResult).toBe(true);
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });
  
  test('should make POST request to VLLM', async () => {
    // @ts-ignore - Testing protected method
    const response = await backend.makePostRequestVLLM(
      'https://vllm-api.example.com/v1/completions',
      { prompt: 'Hello', max_tokens: 10 }
    );
    
    expect(response).toBeDefined();
    expect(global.fetch).toHaveBeenCalledWith(
      'https://vllm-api.example.com/v1/completions',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        }),
        body: expect.any(String)
      })
    );
    
    // Test with options
    // @ts-ignore - Testing protected method
    const responseWithOptions = await backend.makePostRequestVLLM(
      'https://vllm-api.example.com/v1/completions',
      { prompt: 'Hello', max_tokens: 10 },
      { temperature: 0.5, top_p: 0.9, apiKey: 'custom-key' }
    );
    
    expect(responseWithOptions).toBeDefined();
    expect(global.fetch).toHaveBeenCalledTimes(2);
    expect(global.fetch).toHaveBeenLastCalledWith(
      'https://vllm-api.example.com/v1/completions',
      expect.objectContaining({
        body: expect.stringContaining('"temperature":0.5')
      })
    );
  });
  
  test('should handle API errors in makePostRequestVLLM', async () => {
    // Mock error responses for different status codes
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 404,
      text: jest.fn().mockResolvedValue('Model not found')
    });
    
    // @ts-ignore - Testing protected method
    await expect(backend.makePostRequestVLLM('https://vllm-api.example.com/v1/completions', {}))
      .rejects.toThrow('VLLM API error: Model not found');
    
    // 401 error
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 401,
      text: jest.fn().mockResolvedValue('Unauthorized')
    });
    
    // @ts-ignore - Testing protected method
    await expect(backend.makePostRequestVLLM('https://vllm-api.example.com/v1/completions', {}))
      .rejects.toThrow('VLLM API error: Unauthorized');
    
    // 500 error
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: jest.fn().mockResolvedValue('Internal server error')
    });
    
    // @ts-ignore - Testing protected method
    await expect(backend.makePostRequestVLLM('https://vllm-api.example.com/v1/completions', {}))
      .rejects.toThrow('VLLM API error: Internal server error');
  });
  
  test('should make streaming request to VLLM', async () => {
    // Mock streaming response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      body: mockReadableStream(),
      status: 200
    });
    
    // @ts-ignore - Testing protected method
    const stream = await backend.makeStreamRequestVLLM(
      'https://vllm-api.example.com/v1/completions',
      { prompt: 'Hello', max_tokens: 10 }
    );
    
    expect(stream).toBeDefined();
    
    // Collect items from the stream
    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    
    expect(chunks.length).toBeGreaterThan(0);
    expect(global.fetch).toHaveBeenCalledWith(
      'https://vllm-api.example.com/v1/completions',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"stream":true')
      })
    );
  });
  
  test('should handle errors in makeStreamRequestVLLM', async () => {
    // Mock error response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 401,
      text: jest.fn().mockResolvedValue('Unauthorized')
    });
    
    // @ts-ignore - Testing protected method
    await expect(backend.makeStreamRequestVLLM(
      'https://vllm-api.example.com/v1/completions',
      { prompt: 'Hello', max_tokens: 10 }
    )).rejects.toThrow('VLLM API streaming error: Unauthorized');
  });
  
  test('should handle null body in streaming response', async () => {
    // Mock response with null body
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      body: null,
      status: 200
    });
    
    // @ts-ignore - Testing protected method
    await expect(backend.makeStreamRequestVLLM(
      'https://vllm-api.example.com/v1/completions',
      { prompt: 'Hello', max_tokens: 10 }
    )).rejects.toThrow('Response body is null');
  });
  
  test('should handle chat completion', async () => {
    const messages: Message[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    const response = await backend.chat(messages);
    
    expect(response).toBeDefined();
    expect(response.content).toBe('Hello, I am an AI assistant.');
    expect(global.fetch).toHaveBeenCalled();
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/chat/completions'),
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"messages"')
      })
    );
    
    // Test with options
    const responseWithOptions = await backend.chat(messages, {
      model: 'custom-model',
      temperature: 0.5,
      top_p: 0.9,
      max_tokens: 50
    });
    
    expect(responseWithOptions).toBeDefined();
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/chat/completions'),
      expect.objectContaining({
        body: expect.stringContaining('"model":"custom-model"')
      })
    );
  });
  
  test('should stream chat completion', async () => {
    // Mock streaming response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      body: mockReadableStream(),
      status: 200
    });
    
    const messages: Message[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    const stream = backend.streamChat(messages);
    expect(stream).toBeDefined();
    
    // Collect items from the stream
    const chunks = [];
    for await (const chunk of await stream) {
      chunks.push(chunk);
    }
    
    expect(chunks.length).toBeGreaterThan(0);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/chat/completions'),
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"stream":true')
      })
    );
  });
  
  test('should implement makePostRequest abstract method', async () => {
    const data = { prompt: 'Hello', max_tokens: 10 };
    const response = await backend.makePostRequest(data);
    
    expect(response).toBeDefined();
    expect(global.fetch).toHaveBeenCalled();
    
    // Test with messages
    const messageData = { 
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 10
    };
    const messageResponse = await backend.makePostRequest(messageData);
    
    expect(messageResponse).toBeDefined();
    expect(global.fetch).toHaveBeenCalledTimes(2);
    expect(global.fetch).toHaveBeenLastCalledWith(
      expect.stringContaining('/v1/chat/completions'),
      expect.any(Object)
    );
    
    // Test with custom endpoint
    const customResponse = await backend.makePostRequest(data, undefined, {
      endpoint: 'https://custom-endpoint.com'
    });
    
    expect(customResponse).toBeDefined();
    expect(global.fetch).toHaveBeenCalledTimes(3);
    expect(global.fetch).toHaveBeenLastCalledWith(
      'https://custom-endpoint.com',
      expect.any(Object)
    );
  });
  
  test('should implement makeStreamRequest abstract method', async () => {
    // Mock streaming response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      body: mockReadableStream(),
      status: 200
    });
    
    const data = { prompt: 'Hello', max_tokens: 10 };
    const stream = await backend.makeStreamRequest(data);
    
    expect(stream).toBeDefined();
    
    // Collect items from the stream
    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    
    expect(chunks.length).toBeGreaterThan(0);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/completions'),
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"stream":true')
      })
    );
    
    // Test with messages
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      body: mockReadableStream(),
      status: 200
    });
    
    const messageData = { 
      messages: [{ role: 'user', content: 'Hello' }]
    };
    const messageStream = await backend.makeStreamRequest(messageData);
    
    expect(messageStream).toBeDefined();
    expect(global.fetch).toHaveBeenCalledTimes(2);
    expect(global.fetch).toHaveBeenLastCalledWith(
      expect.stringContaining('/v1/chat/completions'),
      expect.any(Object)
    );
  });
  
  test('should check model compatibility', () => {
    // Test with compatible model prefixes
    expect(backend.isCompatibleModel('meta-llama/llama-7b')).toBe(true);
    expect(backend.isCompatibleModel('mistralai/mistral-7b')).toBe(true);
    
    // Test with compatible model architectures
    expect(backend.isCompatibleModel('custom-llama-model')).toBe(true);
    expect(backend.isCompatibleModel('my-falcon-model')).toBe(true);
    
    // Test with incompatible model
    expect(backend.isCompatibleModel('')).toBe(false);
    expect(backend.isCompatibleModel('clip-vit-base')).toBe(false);
    expect(backend.isCompatibleModel('BAAI/bge-small-en')).toBe(false);
  });
  
  test('should process batch of prompts', async () => {
    // @ts-ignore - Testing method not in interface but in implementation
    const results = await backend.processBatch(
      'https://vllm-api.example.com/v1/completions',
      ['Hello', 'How are you?'],
      'test-model'
    );
    
    expect(results).toBeDefined();
    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBe(2);
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });
  
  test('should process batch with metrics', async () => {
    // @ts-ignore - Testing method not in interface but in implementation
    const [results, metrics] = await backend.processBatchWithMetrics(
      'https://vllm-api.example.com/v1/completions',
      ['Hello', 'How are you?'],
      'test-model'
    );
    
    expect(results).toBeDefined();
    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBe(2);
    
    expect(metrics).toBeDefined();
    expect(metrics.batch_size).toBe(2);
    expect(metrics.successful_items).toBe(2);
    expect(metrics.total_time_ms).toBeGreaterThanOrEqual(0);
    expect(metrics.average_time_per_item_ms).toBeGreaterThanOrEqual(0);
    
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });
  
  test('should stream generation', async () => {
    // Mock streaming response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      body: mockReadableStream(),
      status: 200
    });
    
    // @ts-ignore - Testing method not in interface but in implementation
    const stream = backend.streamGeneration(
      'https://vllm-api.example.com/v1/completions',
      'Hello, how are you?',
      'test-model'
    );
    
    expect(stream).toBeDefined();
    
    // Collect items from the stream
    const chunks = [];
    for await (const chunk of await stream) {
      chunks.push(chunk);
    }
    
    expect(chunks.length).toBeGreaterThan(0);
    expect(global.fetch).toHaveBeenCalledWith(
      'https://vllm-api.example.com/v1/completions',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"stream":true')
      })
    );
  });
  
  test('should get model info', async () => {
    // Mock model info response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        model: 'meta-llama/Llama-2-7b-chat-hf',
        max_model_len: 4096,
        num_gpu: 1,
        dtype: 'float16',
        gpu_memory_utilization: 0.85,
        quantization: {
          enabled: false
        }
      })
    });
    
    const modelInfo = await backend.getModelInfo();
    
    expect(modelInfo).toBeDefined();
    expect(modelInfo.model).toBe('meta-llama/Llama-2-7b-chat-hf');
    expect(modelInfo.max_model_len).toBe(4096);
    expect(modelInfo.num_gpu).toBe(1);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/models/'),
      expect.objectContaining({
        method: 'GET'
      })
    );
    
    // Test with custom model name
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        model: 'custom-model',
        max_model_len: 2048,
        num_gpu: 2,
        dtype: 'float16',
        gpu_memory_utilization: 0.75
      })
    });
    
    const customModelInfo = await backend.getModelInfo('custom-model');
    
    expect(customModelInfo).toBeDefined();
    expect(customModelInfo.model).toBe('custom-model');
    expect(global.fetch).toHaveBeenLastCalledWith(
      expect.stringContaining('/v1/models/custom-model'),
      expect.any(Object)
    );
  });
  
  test('should handle errors in getModelInfo', async () => {
    // Mock error response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 404,
      statusText: 'Not Found'
    });
    
    await expect(backend.getModelInfo()).rejects.toThrow('Failed to get model info: Not Found');
    
    // Test network error
    (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Connection error'));
    
    await expect(backend.getModelInfo()).rejects.toThrow('Connection error');
  });
  
  test('should get model statistics', async () => {
    // Mock model statistics response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        model: 'test-model',
        statistics: {
          requests_processed: 100,
          tokens_generated: 5000,
          avg_tokens_per_request: 50,
          max_tokens_per_request: 200,
          avg_generation_time: 0.5,
          throughput: 100,
          errors: 2,
          uptime: 3600
        }
      })
    });
    
    const stats = await backend.getModelStatistics();
    
    expect(stats).toBeDefined();
    expect(stats.model).toBe('test-model');
    expect(stats.statistics.requests_processed).toBe(100);
    expect(stats.statistics.tokens_generated).toBe(5000);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/models/'),
      expect.objectContaining({
        method: 'GET'
      })
    );
    
    // Test with custom model name
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        model: 'custom-model',
        statistics: {
          requests_processed: 50,
          tokens_generated: 2500
        }
      })
    });
    
    const customStats = await backend.getModelStatistics('custom-model');
    
    expect(customStats).toBeDefined();
    expect(customStats.model).toBe('custom-model');
    expect(customStats.statistics.requests_processed).toBe(50);
    expect(global.fetch).toHaveBeenLastCalledWith(
      expect.stringContaining('/v1/models/custom-model/statistics'),
      expect.any(Object)
    );
  });
  
  test('should handle errors in getModelStatistics', async () => {
    // Mock error response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error'
    });
    
    await expect(backend.getModelStatistics()).rejects.toThrow('Failed to get model statistics: Internal Server Error');
  });
  
  test('should list LoRA adapters', async () => {
    // Mock LoRA adapter response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        lora_adapters: [
          {
            id: 'adapter1',
            name: 'test-adapter',
            base_model: 'meta-llama/Llama-2-7b-chat-hf',
            size_mb: 10,
            active: true
          },
          {
            id: 'adapter2',
            name: 'another-adapter',
            base_model: 'meta-llama/Llama-2-7b-chat-hf',
            size_mb: 15,
            active: false
          }
        ]
      })
    });
    
    const adapters = await backend.listLoraAdapters();
    
    expect(adapters).toBeDefined();
    expect(Array.isArray(adapters)).toBe(true);
    expect(adapters.length).toBe(2);
    expect(adapters[0].id).toBe('adapter1');
    expect(adapters[1].name).toBe('another-adapter');
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/lora_adapters'),
      expect.objectContaining({
        method: 'GET'
      })
    );
  });
  
  test('should handle empty LoRA adapter response', async () => {
    // Mock empty LoRA adapter response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({})
    });
    
    const adapters = await backend.listLoraAdapters();
    
    expect(adapters).toBeDefined();
    expect(Array.isArray(adapters)).toBe(true);
    expect(adapters.length).toBe(0);
  });
  
  test('should handle errors in listLoraAdapters', async () => {
    // Mock error response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error'
    });
    
    await expect(backend.listLoraAdapters()).rejects.toThrow('Failed to list LoRA adapters: Internal Server Error');
  });
  
  test('should load LoRA adapter', async () => {
    // Mock successful LoRA adapter loading
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        success: true,
        adapter: {
          id: 'adapter1',
          name: 'test-adapter',
          base_model: 'meta-llama/Llama-2-7b-chat-hf',
          size_mb: 10,
          active: true
        }
      })
    });
    
    const adapterData = {
      adapter_name: 'test-adapter',
      adapter_path: '/path/to/adapter',
      base_model: 'meta-llama/Llama-2-7b-chat-hf'
    };
    
    const result = await backend.loadLoraAdapter(adapterData);
    
    expect(result).toBeDefined();
    expect(result.success).toBe(true);
    expect(result.adapter.name).toBe('test-adapter');
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/lora_adapters'),
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"adapter_name":"test-adapter"')
      })
    );
  });
  
  test('should handle errors in loadLoraAdapter', async () => {
    // Mock error response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 400,
      statusText: 'Bad Request'
    });
    
    const adapterData = {
      adapter_name: 'invalid-adapter',
      adapter_path: '/path/to/invalid',
      base_model: 'meta-llama/Llama-2-7b-chat-hf'
    };
    
    await expect(backend.loadLoraAdapter(adapterData)).rejects.toThrow('Failed to load LoRA adapter: Bad Request');
  });
  
  test('should set quantization', async () => {
    // Mock successful quantization update
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        success: true,
        model: 'test-model',
        quantization: {
          enabled: true,
          method: 'awq',
          bits: 4
        }
      })
    });
    
    const quantConfig: VLLMQuantizationConfig = {
      enabled: true,
      method: 'awq',
      bits: 4
    };
    
    const result = await backend.setQuantization('test-model', quantConfig);
    
    expect(result).toBeDefined();
    expect(result.success).toBe(true);
    expect(result.quantization.enabled).toBe(true);
    expect(result.quantization.method).toBe('awq');
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/models/test-model/quantization'),
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"method":"awq"')
      })
    );
  });
  
  test('should handle errors in setQuantization', async () => {
    // Mock error response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 400,
      statusText: 'Bad Request'
    });
    
    const quantConfig: VLLMQuantizationConfig = {
      enabled: true,
      method: 'invalid-method',
      bits: 3
    };
    
    await expect(backend.setQuantization('test-model', quantConfig)).rejects.toThrow('Failed to set quantization: Bad Request');
  });
});