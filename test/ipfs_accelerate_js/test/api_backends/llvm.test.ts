// Tests for LLVM API Backend
import { LLVM } from '../../src/api_backends/llvm';
import { ChatMessage } from '../../src/api_backends/types';

// Mock global fetch
global.fetch = jest.fn();
const mockFetch = global.fetch as jest.Mock;

describe('LLVM API Backend', () => {
  let backend: LLVM;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    mockFetch.mockReset();

    // Mock successful fetch response
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ success: true })
    });

    // Set up environment variables
    process.env.LLVM_API_KEY = 'test-api-key';
    process.env.LLVM_BASE_URL = 'http://test-llvm-api.com';
    process.env.LLVM_DEFAULT_MODEL = 'llvm-test-model';

    // Create backend instance
    backend = new LLVM();
  });

  afterEach(() => {
    // Clean up environment variables
    delete process.env.LLVM_API_KEY;
    delete process.env.LLVM_BASE_URL;
    delete process.env.LLVM_DEFAULT_MODEL;
  });

  test('should initialize correctly', () => {
    expect(backend).toBeDefined();
    
    // @ts-ignore - accessing private properties for testing
    expect(backend.apiKey).toBe('test-api-key');
    // @ts-ignore - accessing private properties for testing
    expect(backend.baseUrl).toBe('http://test-llvm-api.com');
    // @ts-ignore - accessing private properties for testing
    expect(backend.maxRetries).toBe(3);
  });

  test('should initialize with custom options', () => {
    const customBackend = new LLVM({
      base_url: 'https://custom-llvm-url.com',
      max_concurrent_requests: 20,
      max_retries: 5,
      retry_delay: 2000,
      queue_size: 200
    });

    // @ts-ignore - accessing private properties for testing
    expect(customBackend.baseUrl).toBe('https://custom-llvm-url.com');
    // @ts-ignore - accessing private properties for testing
    expect(customBackend.maxConcurrentRequests).toBe(20);
    // @ts-ignore - accessing private properties for testing
    expect(customBackend.maxRetries).toBe(5);
    // @ts-ignore - accessing private properties for testing
    expect(customBackend.retryDelay).toBe(2000);
    // @ts-ignore - accessing private properties for testing
    expect(customBackend.queueSize).toBe(200);
  });

  test('should get API key from environment', () => {
    const apiKey = backend.getApiKey();
    expect(apiKey).toBe('test-api-key');
  });

  test('should get API key from metadata', () => {
    const customBackend = new LLVM({}, {
      llvm_api_key: 'metadata-api-key'
    });

    const apiKey = customBackend.getApiKey();
    expect(apiKey).toBe('metadata-api-key');
  });

  test('should set API key', () => {
    backend.setApiKey('new-api-key');
    
    // @ts-ignore - accessing private properties for testing
    expect(backend.apiKey).toBe('new-api-key');
  });

  test('should get default model from environment', () => {
    const model = backend.getDefaultModel();
    expect(model).toBe('llvm-test-model');
  });

  test('should get default model from metadata', () => {
    const customBackend = new LLVM({}, {
      llvm_default_model: 'metadata-model'
    });

    const model = customBackend.getDefaultModel();
    expect(model).toBe('metadata-model');
  });

  test('should check model compatibility correctly', () => {
    // Compatible models
    expect(backend.isCompatibleModel('llvm-model')).toBe(true);
    expect(backend.isCompatibleModel('llvm_model')).toBe(true);
    expect(backend.isCompatibleModel('model-llvm')).toBe(true);
    expect(backend.isCompatibleModel('model_llvm')).toBe(true);
    
    // Incompatible models
    expect(backend.isCompatibleModel('gpt-4')).toBe(false);
    expect(backend.isCompatibleModel('bert-base')).toBe(false);
    expect(backend.isCompatibleModel('ggml-model')).toBe(false);
  });

  test('should test endpoint successfully', async () => {
    // Mock list models response
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: ['model1', 'model2', 'model3']
      })
    });

    const result = await backend.testEndpoint();
    expect(result).toBe(true);
    
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://test-llvm-api.com/models',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          'Authorization': 'Bearer test-api-key'
        })
      })
    );
  });

  test('should handle failed endpoint test', async () => {
    // Mock error response
    mockFetch.mockResolvedValueOnce({
      ok: false,
      statusText: 'Server Error'
    });

    const result = await backend.testEndpoint();
    expect(result).toBe(false);
  });

  test('should make POST request with authentication', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ result: 'success' })
    });

    const response = await backend.makePostRequest(
      'http://test-llvm-api.com/endpoint',
      { test: 'data' }
    );

    expect(response).toEqual({ result: 'success' });
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://test-llvm-api.com/endpoint',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          'Authorization': 'Bearer test-api-key'
        }),
        body: JSON.stringify({ test: 'data' })
      })
    );
  });

  test('should throw error for stream request', async () => {
    await expect(
      backend.makeStreamRequest('url', { data: 'test' })
    ).rejects.toThrow('Streaming is not supported by the LLVM API');
  });

  test('should list models', async () => {
    // Mock processQueue to resolve immediately
    // @ts-ignore - mocking private method for testing
    jest.spyOn(backend, 'processQueue').mockImplementation(async () => {
      // @ts-ignore - accessing private property for testing
      const request = backend.requestQueue[0];
      const result = await backend.executeOperation(request.operation, request.params);
      request.resolve(result);
    });
    
    // Mock executeListModels to return model list
    // @ts-ignore - mocking private method for testing
    jest.spyOn(backend, 'executeListModels').mockResolvedValue({
      models: ['model1', 'model2', 'model3']
    });

    const result = await backend.listModels();

    expect(result).toEqual({
      models: ['model1', 'model2', 'model3']
    });
    
    // @ts-ignore - verify private method was called
    expect(backend.processQueue).toHaveBeenCalled();
    // @ts-ignore - verify private method was called
    expect(backend.executeListModels).toHaveBeenCalled();
  });

  test('should get model info', async () => {
    // Mock processQueue to resolve immediately
    // @ts-ignore - mocking private method for testing
    jest.spyOn(backend, 'processQueue').mockImplementation(async () => {
      // @ts-ignore - accessing private property for testing
      const request = backend.requestQueue[0];
      const result = await backend.executeOperation(request.operation, request.params);
      request.resolve(result);
    });
    
    // Mock executeGetModelInfo to return model info
    // @ts-ignore - mocking private method for testing
    jest.spyOn(backend, 'executeGetModelInfo').mockResolvedValue({
      model_id: 'test-model',
      status: 'loaded',
      details: {
        size: '7B',
        type: 'transformer'
      }
    });

    const modelId = 'test-model';
    const result = await backend.getModelInfo(modelId);

    expect(result).toEqual({
      model_id: 'test-model',
      status: 'loaded',
      details: {
        size: '7B',
        type: 'transformer'
      }
    });
    
    // @ts-ignore - verify private method was called
    expect(backend.processQueue).toHaveBeenCalled();
    // @ts-ignore - verify private method was called
    expect(backend.executeGetModelInfo).toHaveBeenCalledWith(modelId);
  });

  test('should run inference', async () => {
    // Mock processQueue to resolve immediately
    // @ts-ignore - mocking private method for testing
    jest.spyOn(backend, 'processQueue').mockImplementation(async () => {
      // @ts-ignore - accessing private property for testing
      const request = backend.requestQueue[0];
      const result = await backend.executeOperation(request.operation, request.params);
      request.resolve(result);
    });
    
    // Mock executeRunInference to return inference results
    // @ts-ignore - mocking private method for testing
    jest.spyOn(backend, 'executeRunInference').mockResolvedValue({
      model_id: 'test-model',
      outputs: 'Generated text response',
      metadata: {
        tokens_generated: 20,
        generation_time_ms: 500
      }
    });

    const modelId = 'test-model';
    const inputs = 'Test prompt';
    const options = {
      max_tokens: 100,
      temperature: 0.7
    };
    
    const result = await backend.runInference(modelId, inputs, options);

    expect(result).toEqual({
      model_id: 'test-model',
      outputs: 'Generated text response',
      metadata: {
        tokens_generated: 20,
        generation_time_ms: 500
      }
    });
    
    // @ts-ignore - verify private method was called
    expect(backend.processQueue).toHaveBeenCalled();
    // @ts-ignore - verify private method was called
    expect(backend.executeRunInference).toHaveBeenCalledWith(modelId, inputs, options);
  });

  test('should execute list_models operation', async () => {
    // Mock successful response
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: ['model1', 'model2', 'model3']
      })
    });

    // @ts-ignore - testing private method
    const result = await backend.executeListModels();

    expect(result).toEqual({
      models: ['model1', 'model2', 'model3']
    });
    
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://test-llvm-api.com/models',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Authorization': 'Bearer test-api-key'
        })
      })
    );
  });

  test('should execute get_model_info operation', async () => {
    // Mock successful response
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        model_id: 'test-model',
        status: 'loaded',
        details: {
          size: '7B',
          type: 'transformer'
        }
      })
    });

    // @ts-ignore - testing private method
    const result = await backend.executeGetModelInfo('test-model');

    expect(result).toEqual({
      model_id: 'test-model',
      status: 'loaded',
      details: {
        size: '7B',
        type: 'transformer'
      }
    });
    
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://test-llvm-api.com/models/test-model',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          'Authorization': 'Bearer test-api-key'
        })
      })
    );
  });

  test('should execute run_inference operation', async () => {
    // Mock successful POST request
    // @ts-ignore - mocking private method for testing
    jest.spyOn(backend, 'makePostRequest').mockResolvedValue({
      model_id: 'test-model',
      outputs: 'Generated text response',
      metadata: {
        tokens_generated: 20,
        generation_time_ms: 500
      }
    });

    const modelId = 'test-model';
    const inputs = 'Test prompt';
    const options = {
      max_tokens: 100,
      temperature: 0.7
    };
    
    // @ts-ignore - testing private method
    const result = await backend.executeRunInference(modelId, inputs, options);

    expect(result).toEqual({
      model_id: 'test-model',
      outputs: 'Generated text response',
      metadata: {
        tokens_generated: 20,
        generation_time_ms: 500
      }
    });
    
    // @ts-ignore - verify private method was called
    expect(backend.makePostRequest).toHaveBeenCalledWith(
      'http://test-llvm-api.com/models/test-model/inference',
      {
        inputs: 'Test prompt',
        max_tokens: 100,
        temperature: 0.7
      }
    );
  });

  test('should format chat messages correctly', async () => {
    const messages: ChatMessage[] = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Hello, how are you?' },
      { role: 'assistant', content: 'I am doing well, thank you for asking!' },
      { role: 'user', content: 'Tell me a joke.' }
    ];
    
    // @ts-ignore - testing private method
    const formattedPrompt = backend.formatChatMessages(messages);
    
    const expectedPrompt = 
      'system: You are a helpful assistant.\n' +
      'user: Hello, how are you?\n' +
      'assistant: I am doing well, thank you for asking!\n' +
      'user: Tell me a joke.';
    
    expect(formattedPrompt).toBe(expectedPrompt);
  });

  test('should handle multimodal content in chat messages', async () => {
    const messages: ChatMessage[] = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { 
        role: 'user', 
        content: [
          { type: 'text', text: 'What is this image?' },
          { type: 'image_url', image_url: { url: 'https://example.com/image.jpg' } }
        ]
      }
    ];
    
    // @ts-ignore - testing private method
    const formattedPrompt = backend.formatChatMessages(messages);
    
    const expectedPrompt = 
      'system: You are a helpful assistant.\n' +
      'user: [{"type":"text","text":"What is this image?"},{"type":"image_url","image_url":{"url":"https://example.com/image.jpg"}}]';
    
    expect(formattedPrompt).toBe(expectedPrompt);
  });

  test('should implement chat method using runInference', async () => {
    // Mock runInference to return test response
    jest.spyOn(backend, 'runInference').mockResolvedValue({
      model_id: 'test-model',
      outputs: 'This is a response to your query.',
      metadata: {
        tokens_generated: 10
      }
    });

    const modelId = 'test-model';
    const messages: ChatMessage[] = [
      { role: 'user', content: 'Hello, how are you?' }
    ];
    
    const result = await backend.chat(modelId, messages);

    expect(result).toEqual({
      id: expect.stringMatching(/^llvm-\d+$/),
      model: 'test-model',
      object: 'chat.completion',
      created: expect.any(Number),
      content: 'This is a response to your query.',
      role: 'assistant'
    });
    
    // Verify runInference was called with formatted prompt
    expect(backend.runInference).toHaveBeenCalledWith(
      modelId,
      'user: Hello, how are you?',
      undefined
    );
  });

  test('should throw error for streamChat', async () => {
    const modelId = 'test-model';
    const messages: ChatMessage[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    // Need to wrap in a function since streamChat returns an AsyncGenerator
    await expect(async () => {
      for await (const chunk of backend.streamChat(modelId, messages)) {
        // This should not execute
      }
    }).rejects.toThrow('Streaming is not supported by the LLVM API');
  });

  test('should retry failed operations', async () => {
    // Mock executeOperation to fail first time, then succeed
    // @ts-ignore - mocking private method for testing
    jest.spyOn(backend, 'executeOperation')
      .mockRejectedValueOnce(new Error('Temporary error'))
      .mockResolvedValueOnce({
        models: ['model1', 'model2', 'model3']
      });
    
    // Mock setTimeout to resolve immediately
    jest.spyOn(global, 'setTimeout').mockImplementation((cb: any) => {
      cb();
      return 0 as any;
    });

    // Mock processQueue to use the executeOperation properly
    // @ts-ignore - mocking private method for testing
    jest.spyOn(backend, 'processQueue').mockImplementation(async () => {
      // @ts-ignore - accessing private property for testing
      const request = backend.requestQueue[0];
      
      try {
        let result;
        let success = false;
        let lastError;
        
        for (let attempt = 0; attempt <= backend.maxRetries; attempt++) {
          try {
            // @ts-ignore - accessing private method for testing
            result = await backend.executeOperation(request.operation, request.params);
            success = true;
            break;
          } catch (error) {
            lastError = error;
            
            if (attempt < backend.maxRetries) {
              await new Promise<void>(resolve => {
                setTimeout(resolve, 0);
              });
            }
          }
        }
        
        if (success) {
          request.resolve(result);
        } else {
          request.reject(lastError);
        }
      } catch (error) {
        request.reject(error);
      }
    });

    const result = await backend.listModels();
    
    expect(result).toEqual({
      models: ['model1', 'model2', 'model3']
    });
    
    // @ts-ignore - verify the executeOperation method was called twice
    expect(backend.executeOperation).toHaveBeenCalledTimes(2);
  });

  test('should handle queue full condition', async () => {
    // Create backend with small queue size
    const smallQueueBackend = new LLVM({
      queue_size: 1
    });
    
    // Fill the queue
    // @ts-ignore - accessing private property for testing
    smallQueueBackend.requestQueue = [{ id: 'test' }];
    
    // Try to queue a request
    await expect(
      smallQueueBackend.listModels()
    ).rejects.toThrow('Request queue is full');
  });
});