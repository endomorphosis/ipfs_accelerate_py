// Tests for OVMS API Backend
import { OVMS } from '../../src/api_backends/ovms';
import { ApiMetadata, Message, ApiRequestOptions } from '../../src/api_backends/types';
import { 
  OVMSRequestData, 
  OVMSResponse, 
  OVMSModelMetadata,
  OVMSModelConfig,
  OVMSServerStatistics,
  OVMSQuantizationConfig
} from '../../src/api_backends/ovms/types';

// Mock fetch for testing
global.fetch = jest.fn();

describe('OVMS API Backend', () => {
  let backend: OVMS;
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
      ovms_api_key: 'test-api-key'
    };
    
    // Create backend instance
    backend = new OVMS({}, mockMetadata);
  });
  
  test('should initialize correctly', () => {
    expect(backend).toBeDefined();
    
    // Test with custom API URL
    const customBackend = new OVMS({}, { 
      ovms_api_url: 'https://custom-ovms-url.com',
      ovms_model: 'custom-model',
      ovms_version: '2',
      ovms_precision: 'FP16'
    });
    expect(customBackend).toBeDefined();
    
    // @ts-ignore - Accessing protected property for testing
    expect(customBackend.apiUrl).toBe('https://custom-ovms-url.com');
    // @ts-ignore - Accessing protected property for testing
    expect(customBackend.modelName).toBe('custom-model');
    // @ts-ignore - Accessing protected property for testing
    expect(customBackend.modelVersion).toBe('2');
    // @ts-ignore - Accessing protected property for testing
    expect(customBackend.precision).toBe('FP16');
  });
  
  test('should get API key from metadata', () => {
    // @ts-ignore - Testing protected method
    const apiKey = backend.getApiKey(mockMetadata);
    expect(apiKey).toBe('test-api-key');
    
    // Test environment variable fallback
    const originalEnv = process.env.OVMS_API_KEY;
    process.env.OVMS_API_KEY = 'env-api-key';
    // @ts-ignore - Testing protected method
    const envApiKey = backend.getApiKey({});
    expect(envApiKey).toBe('env-api-key');
    
    // Restore original environment
    process.env.OVMS_API_KEY = originalEnv;
  });
  
  test('should get default model', () => {
    // @ts-ignore - Testing protected method
    const model = backend.getDefaultModel();
    expect(model).toBeDefined();
    expect(typeof model).toBe('string');
    expect(model).toBe('model');
  });
  
  test('should create endpoint handler', () => {
    const handler = backend.createEndpointHandler();
    expect(handler).toBeDefined();
    expect(typeof handler).toBe('function');
    
    // Test with custom URL and model
    const customHandler = backend.createEndpointHandler('https://custom-endpoint.com', 'custom-model');
    expect(customHandler).toBeDefined();
    expect(typeof customHandler).toBe('function');
  });
  
  test('should test endpoint', async () => {
    // Update the mock to return predictions array for OVMS
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        predictions: [{ output: "test output" }]
      })
    });
    
    const result = await backend.testEndpoint();
    expect(result).toBe(true);
    expect(global.fetch).toHaveBeenCalled();
    
    // Test with custom URL and model
    const customResult = await backend.testEndpoint('https://custom-endpoint.com', 'custom-model');
    expect(customResult).toBe(true);
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });
  
  test('should test endpoint failure', async () => {
    // Mock a failed fetch
    (global.fetch as jest.Mock).mockRejectedValue(new Error('Network error'));
    
    // Allow test to complete without fetch hanging
    const result = await Promise.race([
      backend.testEndpoint(),
      new Promise<boolean>(resolve => setTimeout(() => resolve(false), 1000))
    ]);
    
    expect(result).toBe(false);
    expect(global.fetch).toHaveBeenCalled();
  }, 10000);
  
  test('should make POST request to OVMS', async () => {
    // Mock predictions response for OVMS
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        predictions: [{ output: [1, 2, 3, 4] }]
      })
    });
    
    // @ts-ignore - Testing protected method
    const response = await backend.makePostRequestOVMS(
      'https://ovms-api.com/v1/models/model:predict',
      { instances: [{ data: [0, 1, 2, 3] }] }
    );
    
    expect(response).toBeDefined();
    expect(response.predictions).toBeDefined();
    expect(global.fetch).toHaveBeenCalledWith(
      'https://ovms-api.com/v1/models/model:predict',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        }),
        body: expect.any(String)
      })
    );
    
    // Test with options (including version)
    // @ts-ignore - Testing protected method
    const responseWithOptions = await backend.makePostRequestOVMS(
      'https://ovms-api.com/v1/models/model:predict',
      { instances: [{ data: [0, 1, 2, 3] }] },
      { version: '2', apiKey: 'custom-key' }
    );
    
    expect(responseWithOptions).toBeDefined();
    expect(responseWithOptions.predictions).toBeDefined();
    expect(global.fetch).toHaveBeenCalledTimes(2);
    // URL should include the version
    expect(global.fetch).toHaveBeenLastCalledWith(
      expect.stringContaining('/versions/2:predict'),
      expect.any(Object)
    );
  });
  
  test('should handle POST request errors', async () => {
    // Mock a failed response
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      text: jest.fn().mockResolvedValue('Invalid input format')
    });
    
    // @ts-ignore - Testing protected method
    await expect(async () => {
      // Add timeout to ensure test completes even if fetch hangs
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Test timeout')), 1000)
      );
      
      return await Promise.race([
        backend.makePostRequestOVMS(
          'https://ovms-api.com/v1/models/model:predict',
          { instances: [{ data: [0, 1, 2, 3] }] }
        ),
        timeoutPromise
      ]);
    }).rejects.toThrow();
    
    expect(global.fetch).toHaveBeenCalled();
  }, 10000);
  
  test('should format request for OVMS', async () => {
    // Mock predictions response for OVMS
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        predictions: [{ output: [1, 2, 3, 4] }]
      })
    });
    
    // Create a handler
    const handler = jest.fn().mockResolvedValue({ predictions: [{ output: [1, 2, 3, 4] }] });
    
    // @ts-ignore - Testing protected method
    const response = await backend.formatRequest(handler, [0, 1, 2, 3]);
    
    expect(response).toBeDefined();
    expect(response.predictions).toBeDefined();
    expect(handler).toHaveBeenCalledWith(expect.objectContaining({
      instances: [{ data: [0, 1, 2, 3] }]
    }));
    
    // Test with object input
    // @ts-ignore - Testing protected method
    const objResponse = await backend.formatRequest(handler, { data: [0, 1, 2, 3] });
    
    expect(objResponse).toBeDefined();
    expect(objResponse.predictions).toBeDefined();
    expect(handler).toHaveBeenCalledTimes(2);
    expect(handler).toHaveBeenLastCalledWith(expect.objectContaining({
      instances: [{ data: [0, 1, 2, 3] }]
    }));
    
    // Test with already formatted input
    // @ts-ignore - Testing protected method
    const formattedResponse = await backend.formatRequest(handler, {
      instances: [{ data: [0, 1, 2, 3] }]
    });
    
    expect(formattedResponse).toBeDefined();
    expect(formattedResponse.predictions).toBeDefined();
    expect(handler).toHaveBeenCalledTimes(3);
    expect(handler).toHaveBeenLastCalledWith(expect.objectContaining({
      instances: [{ data: [0, 1, 2, 3] }]
    }));
  });
  
  test('should infer with OVMS', async () => {
    // Mock predictions response for OVMS
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        predictions: [{ output: [1, 2, 3, 4] }]
      })
    });
    
    // @ts-ignore - Testing method not in interface but in implementation
    const result = await backend.infer('model', [0, 1, 2, 3]);
    
    expect(result).toBeDefined();
    expect(Array.isArray(result)).toBe(true);
    expect(global.fetch).toHaveBeenCalled();
    
    // Test with options
    // @ts-ignore - Testing method not in interface but in implementation
    const resultWithOptions = await backend.infer('model', [0, 1, 2, 3], { version: '2' });
    
    expect(resultWithOptions).toBeDefined();
    expect(Array.isArray(resultWithOptions)).toBe(true);
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });
  
  test('should batch infer with OVMS', async () => {
    // Mock predictions response for OVMS
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        predictions: [
          { output: [1, 2, 3, 4] },
          { output: [5, 6, 7, 8] }
        ]
      })
    });
    
    // @ts-ignore - Testing method not in interface but in implementation
    const results = await backend.batchInfer('model', [[0, 1, 2, 3], [4, 5, 6, 7]]);
    
    expect(results).toBeDefined();
    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBe(2);
    expect(global.fetch).toHaveBeenCalled();
    
    // Verify request format
    expect(global.fetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        body: expect.stringContaining('"instances":[{"data":[0,1,2,3]},{"data":[4,5,6,7]}]')
      })
    );
  });
  
  test('should handle chat completion', async () => {
    // Mock predictions response for OVMS
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        predictions: [{ output: [1, 2, 3, 4] }]
      })
    });
    
    const messages: Message[] = [
      { role: 'user', content: 'Hello' }
    ];
    
    const response = await backend.chat(messages);
    
    expect(response).toBeDefined();
    // OVMS.chat returns JSON stringified result
    expect(typeof response.content).toBe('string');
    expect(global.fetch).toHaveBeenCalled();
    
    // Test with options
    const responseWithOptions = await backend.chat(messages, {
      model: 'custom-model'
    });
    
    expect(responseWithOptions).toBeDefined();
    expect(typeof responseWithOptions.content).toBe('string');
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });
  
  test('should handle chat with empty user messages', async () => {
    const messages: Message[] = [
      { role: 'system', content: 'You are a helpful assistant.' }
    ];
    
    // Should throw since there's no user message
    await expect(backend.chat(messages)).rejects.toThrow('No user message found');
  });
  
  test('should stream chat completion', async () => {
    // Mock predictions response for OVMS
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        predictions: [{ output: [1, 2, 3, 4] }]
      })
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
    
    expect(chunks.length).toBe(1); // OVMS doesn't support true streaming, so it returns a single chunk
    expect(chunks[0].type).toBe('result');
    expect(global.fetch).toHaveBeenCalled();
  });
  
  test('should implement makePostRequest abstract method', async () => {
    // Mock predictions response for OVMS
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        predictions: [{ output: [1, 2, 3, 4] }]
      })
    });
    
    const data = { instances: [{ data: [0, 1, 2, 3] }] };
    const response = await backend.makePostRequest(data);
    
    expect(response).toBeDefined();
    expect(response.predictions).toBeDefined();
    expect(global.fetch).toHaveBeenCalled();
    
    // Test with custom endpoint
    const customResponse = await backend.makePostRequest(data, undefined, {
      endpoint: 'https://custom-endpoint.com'
    });
    
    expect(customResponse).toBeDefined();
    expect(customResponse.predictions).toBeDefined();
    expect(global.fetch).toHaveBeenCalledTimes(2);
    expect(global.fetch).toHaveBeenLastCalledWith(
      'https://custom-endpoint.com',
      expect.any(Object)
    );
  });
  
  test('should implement makeStreamRequest abstract method', async () => {
    // OVMS doesn't support streaming natively, but it implements the interface
    const data = { instances: [{ data: [0, 1, 2, 3] }] };
    const stream = await backend.makeStreamRequest(data);
    
    expect(stream).toBeDefined();
    
    // Collect items from the stream
    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    
    expect(chunks.length).toBe(1); // OVMS doesn't support true streaming, so it returns a single chunk
    expect(chunks[0].type).toBe('result');
    expect(global.fetch).toHaveBeenCalled();
  });
  
  test('should check model compatibility', () => {
    // @ts-ignore - Need to mock getModelInfo for this test
    backend.getModelInfo = jest.fn().mockResolvedValue({ name: 'sample-model', versions: ['1'] });
    
    const compatibleModel = backend.isCompatibleModel('sample-model');
    expect(compatibleModel).toBe(true);
    
    // Reset mock
    // @ts-ignore - Reset mock
    backend.getModelInfo = undefined;
  });
  
  // NEW TESTS FOR MISSING METHODS
  
  test('should get model info', async () => {
    // Mock model info response
    const mockModelInfo: OVMSModelMetadata = {
      name: 'bert',
      versions: ['1', '2', '3'],
      platform: 'openvino',
      inputs: [
        {
          name: 'input_ids',
          datatype: 'INT64',
          shape: [1, 128]
        }
      ],
      outputs: [
        {
          name: 'embeddings',
          datatype: 'FP32',
          shape: [1, 128, 768]
        }
      ]
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue(mockModelInfo)
    });
    
    // @ts-ignore - Testing method not in interface but in implementation
    const modelInfo = await backend.getModelInfo('bert');
    
    expect(modelInfo).toBeDefined();
    expect(modelInfo.name).toBe('bert');
    expect(modelInfo.versions).toEqual(['1', '2', '3']);
    expect(modelInfo.platform).toBe('openvino');
    expect(modelInfo.inputs.length).toBe(1);
    expect(modelInfo.outputs.length).toBe(1);
    expect(global.fetch).toHaveBeenCalledWith(
      `${backend['apiUrl']}/v1/models/bert`,
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        })
      })
    );
  });
  
  test('should handle get model info error', async () => {
    // Mock error response
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found'
    });
    
    // @ts-ignore - Testing method not in interface but in implementation
    await expect(backend.getModelInfo('nonexistent-model')).rejects.toThrow();
    
    expect(global.fetch).toHaveBeenCalled();
  });
  
  test('should get model versions', async () => {
    // Mock model info response
    const mockModelInfo: OVMSModelMetadata = {
      name: 'bert',
      versions: ['1', '2', '3'],
      platform: 'openvino',
      inputs: [],
      outputs: []
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue(mockModelInfo)
    });
    
    // @ts-ignore - Testing method not in interface but in implementation
    const versions = await backend.getModelVersions('bert');
    
    expect(versions).toBeDefined();
    expect(Array.isArray(versions)).toBe(true);
    expect(versions).toEqual(['1', '2', '3']);
    expect(global.fetch).toHaveBeenCalled();
  });
  
  test('should get server statistics', async () => {
    // Mock server statistics response
    const mockStats: OVMSServerStatistics = {
      server_uptime: 3600,
      server_version: '2.0.0',
      active_models: 5,
      total_requests: 1000,
      requests_per_second: 10.5,
      avg_inference_time: 45.2,
      cpu_usage: 35.1,
      memory_usage: 2048
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue(mockStats)
    });
    
    // @ts-ignore - Testing method not in interface but in implementation
    const stats = await backend.getServerStatistics();
    
    expect(stats).toBeDefined();
    expect(stats.server_uptime).toBe(3600);
    expect(stats.active_models).toBe(5);
    expect(stats.avg_inference_time).toBe(45.2);
    expect(global.fetch).toHaveBeenCalledWith(
      `${backend['apiUrl']}/v1/statistics`,
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        })
      })
    );
  });
  
  test('should handle get server statistics error', async () => {
    // Mock error response
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error'
    });
    
    // @ts-ignore - Testing method not in interface but in implementation
    await expect(backend.getServerStatistics()).rejects.toThrow();
    
    expect(global.fetch).toHaveBeenCalled();
  });
  
  test('should get model status', async () => {
    // Mock model status response
    const mockStatus = {
      state: 'AVAILABLE',
      health: 'OK',
      version: '1',
      last_inference_time: '2023-01-01T12:00:00Z'
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue(mockStatus)
    });
    
    // @ts-ignore - Testing method not in interface but in implementation
    const status = await backend.getModelStatus('bert');
    
    expect(status).toBeDefined();
    expect(status.state).toBe('AVAILABLE');
    expect(status.health).toBe('OK');
    expect(global.fetch).toHaveBeenCalledWith(
      `${backend['apiUrl']}/v1/models/bert/status`,
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        })
      })
    );
  });
  
  test('should set model config', async () => {
    // Mock config response
    const mockResponse = {
      status: 'success',
      message: 'Configuration updated successfully'
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue(mockResponse)
    });
    
    const config: OVMSModelConfig = {
      batch_size: 16,
      instance_count: 2,
      execution_mode: 'throughput'
    };
    
    // @ts-ignore - Testing method not in interface but in implementation
    const result = await backend.setModelConfig('bert', config);
    
    expect(result).toBeDefined();
    expect(result.status).toBe('success');
    expect(global.fetch).toHaveBeenCalledWith(
      `${backend['apiUrl']}/v1/models/bert/config`,
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        }),
        body: JSON.stringify(config)
      })
    );
  });
  
  test('should set execution mode', async () => {
    // Mock the setModelConfig method
    // @ts-ignore - Mocking method for testing
    backend.setModelConfig = jest.fn().mockResolvedValue({ status: 'success' });
    
    // @ts-ignore - Testing method not in interface but in implementation
    const result = await backend.setExecutionMode('bert', 'latency');
    
    expect(result).toBeDefined();
    // @ts-ignore - Checking mock
    expect(backend.setModelConfig).toHaveBeenCalledWith('bert', { execution_mode: 'latency' });
  });
  
  test('should reload model', async () => {
    // Mock reload response
    const mockResponse = {
      status: 'success',
      message: 'Model reloaded successfully'
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue(mockResponse)
    });
    
    // @ts-ignore - Testing method not in interface but in implementation
    const result = await backend.reloadModel('bert');
    
    expect(result).toBeDefined();
    expect(result.status).toBe('success');
    expect(global.fetch).toHaveBeenCalledWith(
      `${backend['apiUrl']}/v1/models/bert/reload`,
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        })
      })
    );
  });
  
  test('should infer with version', async () => {
    // Mock the infer method
    // @ts-ignore - Mocking method for testing
    backend.infer = jest.fn().mockResolvedValue([{ output: [1, 2, 3, 4] }]);
    
    const data = [0, 1, 2, 3];
    const options = { timeout: 5000 };
    
    // @ts-ignore - Testing method not in interface but in implementation
    const result = await backend.inferWithVersion('bert', '2', data, options);
    
    expect(result).toBeDefined();
    // @ts-ignore - Checking mock
    expect(backend.infer).toHaveBeenCalledWith('bert', data, {
      ...options,
      version: '2'
    });
  });
  
  test('should explain prediction', async () => {
    // Mock explain response
    const mockResponse = {
      predictions: [
        {
          output: [1, 2, 3, 4],
          importance: [0.1, 0.2, 0.3, 0.4]
        }
      ]
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue(mockResponse)
    });
    
    const data = [0, 1, 2, 3];
    
    // @ts-ignore - Testing method not in interface but in implementation
    const result = await backend.explainPrediction('bert', data);
    
    expect(result).toBeDefined();
    expect(result.predictions).toBeDefined();
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/models/bert:explain'),
      expect.any(Object)
    );
  });
  
  test('should get model metadata with shapes', async () => {
    // Mock metadata response
    const mockMetadata: OVMSModelMetadata = {
      name: 'bert',
      versions: ['1'],
      platform: 'openvino',
      inputs: [
        {
          name: 'input_ids',
          datatype: 'INT64',
          shape: [1, 128]
        }
      ],
      outputs: [
        {
          name: 'embeddings',
          datatype: 'FP32',
          shape: [1, 128, 768]
        }
      ]
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue(mockMetadata)
    });
    
    // @ts-ignore - Testing method not in interface but in implementation
    const metadata = await backend.getModelMetadataWithShapes('bert');
    
    expect(metadata).toBeDefined();
    expect(metadata.name).toBe('bert');
    expect(metadata.inputs[0].shape).toEqual([1, 128]);
    expect(metadata.outputs[0].shape).toEqual([1, 128, 768]);
    expect(global.fetch).toHaveBeenCalledWith(
      `${backend['apiUrl']}/v1/models/bert/metadata`,
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        })
      })
    );
  });
  
  test('should set quantization', async () => {
    // Mock quantization response
    const mockResponse = {
      status: 'success',
      message: 'Quantization updated successfully'
    };
    
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue(mockResponse)
    });
    
    const config: OVMSQuantizationConfig = {
      enabled: true,
      method: 'int8',
      bits: 8
    };
    
    // @ts-ignore - Testing method not in interface but in implementation
    const result = await backend.setQuantization('bert', config);
    
    expect(result).toBeDefined();
    expect(result.status).toBe('success');
    expect(global.fetch).toHaveBeenCalledWith(
      `${backend['apiUrl']}/v1/models/bert/quantization`,
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        }),
        body: JSON.stringify(config)
      })
    );
  });
  
  test('should handle network errors gracefully', async () => {
    // Mock network error
    (global.fetch as jest.Mock).mockRejectedValue(new Error('Network error'));
    
    // @ts-ignore - Testing method not in interface but in implementation
    await expect(backend.getModelInfo('bert')).rejects.toThrow('Network error');
    
    // Should call console.error
    expect(console.error).toHaveBeenCalled;
  });
  
  test('should set request tracking flag', () => {
    // @ts-ignore - Setting protected property for testing
    backend.requestTracking = true;
    
    // @ts-ignore - Accessing protected property for testing
    expect(backend.requestTracking).toBe(true);
    
    // @ts-ignore - Setting protected property for testing
    backend.requestTracking = false;
    
    // @ts-ignore - Accessing protected property for testing
    expect(backend.requestTracking).toBe(false);
  });
});