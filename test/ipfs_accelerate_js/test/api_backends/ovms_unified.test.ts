// Tests for OVMS Unified API Backend
import { OVMS } from '../../src/api_backends/ovms/ovms';
import { 
  OVMSRequestData, 
  OVMSResponse, 
  OVMSRequestOptions,
  OVMSModelMetadata,
  OVMSModelConfig,
  OVMSServerStatistics,
  OVMSModelStatistics,
  OVMSQuantizationConfig
} from '../../src/api_backends/ovms/types';
import { ApiMetadata, Message, ApiRequestOptions, PriorityLevel } from '../../src/api_backends/types';

// Mock child_process for potential container mode tests
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
    JSON.stringify({
      predictions: [
        { output: [0.1, 0.2, 0.3, 0.4] }
      ]
    })
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
            value: new TextEncoder().encode(chunk)
          });
        } else {
          return Promise.resolve({ done: true });
        }
      },
      releaseLock: () => {}
    })
  };
};

describe('OVMS Unified API Backend', () => {
  let backend: OVMS;
  let mockMetadata: ApiMetadata;
  
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock successful fetch response for OVMS
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        predictions: [
          { output: [0.1, 0.2, 0.3, 0.4] }
        ]
      })
    });
    
    // Set up test data
    mockMetadata = {
      ovms_api_key: 'test-api-key',
      ovms_api_url: 'http://ovms-server:9000',
      ovms_model: 'bert',
      ovms_version: '1',
      ovms_precision: 'FP32'
    };
    
    // Create backend instance
    backend = new OVMS({}, mockMetadata);
  });
  
  // Core Functionality Tests
  
  test('should initialize correctly', () => {
    expect(backend).toBeDefined();
    
    // @ts-ignore - Accessing protected property for testing
    expect(backend.apiUrl).toBe('http://ovms-server:9000');
    // @ts-ignore - Accessing protected property for testing
    expect(backend.modelName).toBe('bert');
    // @ts-ignore - Accessing protected property for testing
    expect(backend.modelVersion).toBe('1');
    // @ts-ignore - Accessing protected property for testing
    expect(backend.precision).toBe('FP32');
  });
  
  test('should initialize with environment variables', () => {
    // Save original environment
    const originalEnv = { ...process.env };
    
    // Set environment variables
    process.env.OVMS_API_URL = 'http://env-server:9000';
    process.env.OVMS_MODEL = 'env-model';
    process.env.OVMS_VERSION = 'env-version';
    process.env.OVMS_PRECISION = 'FP16';
    process.env.OVMS_API_KEY = 'env-api-key';
    process.env.OVMS_TIMEOUT = '60';
    
    // Create backend with environment variables
    const envBackend = new OVMS();
    
    // @ts-ignore - Accessing protected property for testing
    expect(envBackend.apiUrl).toBe('http://env-server:9000');
    // @ts-ignore - Accessing protected property for testing
    expect(envBackend.modelName).toBe('env-model');
    // @ts-ignore - Accessing protected property for testing
    expect(envBackend.modelVersion).toBe('env-version');
    // @ts-ignore - Accessing protected property for testing
    expect(envBackend.precision).toBe('FP16');
    // @ts-ignore - Accessing protected property for testing
    expect(envBackend.apiKey).toBe('env-api-key');
    // @ts-ignore - Accessing protected property for testing
    expect(envBackend.timeout).toBe(60000);
    
    // Restore original environment
    process.env = originalEnv;
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
    expect(model).toBe('model');
    
    // Create backend with custom model
    const customBackend = new OVMS({}, { ovms_model: 'custom-model' });
    // @ts-ignore - Testing protected method
    expect(customBackend.getDefaultModel()).toBe('model');
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
  
  test('should test endpoint successfully', async () => {
    const result = await backend.testEndpoint();
    expect(result).toBe(true);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/models/bert:predict'),
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          'Authorization': 'Bearer test-api-key'
        }),
        body: expect.any(String)
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
  
  // API Request Tests
  
  test('should make POST request to OVMS with correct parameters', async () => {
    const endpointUrl = 'http://ovms-server:9000/v1/models/bert:predict';
    const data: OVMSRequestData = {
      instances: [{ data: [0, 1, 2, 3] }]
    };
    
    // @ts-ignore - Testing protected method
    const response = await backend.makePostRequestOVMS(endpointUrl, data);
    
    expect(response).toBeDefined();
    expect(response.predictions).toBeDefined();
    expect(global.fetch).toHaveBeenCalledWith(
      endpointUrl,
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          'Authorization': 'Bearer test-api-key'
        }),
        body: JSON.stringify(data)
      })
    );
  });
  
  test('should add version to URL when making requests', async () => {
    const endpointUrl = 'http://ovms-server:9000/v1/models/bert:predict';
    const data: OVMSRequestData = {
      instances: [{ data: [0, 1, 2, 3] }]
    };
    const options: OVMSRequestOptions = {
      version: '2'
    };
    
    // @ts-ignore - Testing protected method
    const response = await backend.makePostRequestOVMS(endpointUrl, data, options);
    
    expect(response).toBeDefined();
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/versions/2:predict'),
      expect.any(Object)
    );
  });
  
  test('should handle POST request errors', async () => {
    // Mock a failed response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      text: jest.fn().mockResolvedValue('Invalid input format')
    });
    
    // @ts-ignore - Testing protected method
    await expect(backend.makePostRequestOVMS(
      'http://ovms-server:9000/v1/models/bert:predict',
      { instances: [{ data: [0, 1, 2, 3] }] }
    )).rejects.toThrow('OVMS API error');
    
    // 401 error
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 401,
      text: jest.fn().mockResolvedValue('Unauthorized')
    });
    
    // @ts-ignore - Testing protected method
    await expect(backend.makePostRequestOVMS(
      'http://ovms-server:9000/v1/models/bert:predict',
      { instances: [{ data: [0, 1, 2, 3] }] }
    )).rejects.toThrow('OVMS API error');
    
    // 404 error
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 404,
      text: jest.fn().mockResolvedValue('Model not found')
    });
    
    // @ts-ignore - Testing protected method
    await expect(backend.makePostRequestOVMS(
      'http://ovms-server:9000/v1/models/nonexistent:predict',
      { instances: [{ data: [0, 1, 2, 3] }] }
    )).rejects.toThrow('OVMS API error');
    
    // 500 error
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: jest.fn().mockResolvedValue('Internal server error')
    });
    
    // @ts-ignore - Testing protected method
    await expect(backend.makePostRequestOVMS(
      'http://ovms-server:9000/v1/models/bert:predict',
      { instances: [{ data: [0, 1, 2, 3] }] }
    )).rejects.toThrow('OVMS API error');
  });
  
  test('should format request for OVMS with array input', async () => {
    const handler = jest.fn().mockResolvedValue({ 
      predictions: [{ output: [0.1, 0.2, 0.3, 0.4] }] 
    });
    
    // Test with array input
    // @ts-ignore - Testing protected method
    const response = await backend.formatRequest(handler, [0, 1, 2, 3]);
    
    expect(response).toBeDefined();
    expect(response.predictions).toBeDefined();
    expect(handler).toHaveBeenCalledWith(expect.objectContaining({
      instances: [{ data: [0, 1, 2, 3] }]
    }));
  });
  
  test('should format request for OVMS with object input', async () => {
    const handler = jest.fn().mockResolvedValue({ 
      predictions: [{ output: [0.1, 0.2, 0.3, 0.4] }] 
    });
    
    // Test with object input
    // @ts-ignore - Testing protected method
    const response = await backend.formatRequest(handler, { data: [0, 1, 2, 3] });
    
    expect(response).toBeDefined();
    expect(response.predictions).toBeDefined();
    expect(handler).toHaveBeenCalledWith(expect.objectContaining({
      instances: [{ data: [0, 1, 2, 3] }]
    }));
  });
  
  test('should format request for OVMS with already formatted input', async () => {
    const handler = jest.fn().mockResolvedValue({ 
      predictions: [{ output: [0.1, 0.2, 0.3, 0.4] }] 
    });
    
    // Test with already formatted input
    // @ts-ignore - Testing protected method
    const response = await backend.formatRequest(handler, {
      instances: [{ data: [0, 1, 2, 3] }]
    });
    
    expect(response).toBeDefined();
    expect(response.predictions).toBeDefined();
    expect(handler).toHaveBeenCalledWith(expect.objectContaining({
      instances: [{ data: [0, 1, 2, 3] }]
    }));
  });
  
  test('should format request for OVMS with scalar input', async () => {
    const handler = jest.fn().mockResolvedValue({ 
      predictions: [{ output: [0.1] }] 
    });
    
    // Test with scalar input
    // @ts-ignore - Testing protected method
    const response = await backend.formatRequest(handler, 42);
    
    expect(response).toBeDefined();
    expect(response.predictions).toBeDefined();
    expect(handler).toHaveBeenCalledWith(expect.objectContaining({
      instances: [{ data: [42] }]
    }));
  });
  
  // Inference Tests
  
  test('should run inference with OVMS', async () => {
    // @ts-ignore - Testing method not in interface but in implementation
    const result = await backend.infer('bert', [0, 1, 2, 3]);
    
    expect(result).toBeDefined();
    expect(Array.isArray(result)).toBe(true);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/models/bert:predict'),
      expect.any(Object)
    );
  });
  
  test('should run batch inference with OVMS', async () => {
    const dataBatch = [
      [0, 1, 2, 3],
      [4, 5, 6, 7]
    ];
    
    // Mock batch response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        predictions: [
          { output: [0.1, 0.2, 0.3, 0.4] },
          { output: [0.5, 0.6, 0.7, 0.8] }
        ]
      })
    });
    
    // @ts-ignore - Testing method not in interface but in implementation
    const results = await backend.batchInfer('bert', dataBatch);
    
    expect(results).toBeDefined();
    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBe(2);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/models/bert:predict'),
      expect.objectContaining({
        body: expect.stringContaining('"instances":[{"data":[0,1,2,3]},{"data":[4,5,6,7]}]')
      })
    );
  });
  
  // OpenVINO-Specific Tests
  
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
    
    (global.fetch as jest.Mock).mockResolvedValueOnce({
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
      'http://ovms-server:9000/v1/models/bert',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        })
      })
    );
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
    
    (global.fetch as jest.Mock).mockResolvedValueOnce({
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
    
    (global.fetch as jest.Mock).mockResolvedValueOnce({
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
      'http://ovms-server:9000/v1/models/bert/metadata',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        })
      })
    );
  });
  
  test('should get model status', async () => {
    // Mock model status response
    const mockStatus = {
      state: 'AVAILABLE',
      health: 'OK',
      version: '1',
      last_inference_time: '2023-01-01T12:00:00Z'
    };
    
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue(mockStatus)
    });
    
    // @ts-ignore - Testing method not in interface but in implementation
    const status = await backend.getModelStatus('bert');
    
    expect(status).toBeDefined();
    expect(status.state).toBe('AVAILABLE');
    expect(status.health).toBe('OK');
    expect(global.fetch).toHaveBeenCalledWith(
      'http://ovms-server:9000/v1/models/bert/status',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        })
      })
    );
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
    
    (global.fetch as jest.Mock).mockResolvedValueOnce({
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
      'http://ovms-server:9000/v1/statistics',
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
    
    (global.fetch as jest.Mock).mockResolvedValueOnce({
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
      'http://ovms-server:9000/v1/models/bert/config',
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
    
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue(mockResponse)
    });
    
    // @ts-ignore - Testing method not in interface but in implementation
    const result = await backend.reloadModel('bert');
    
    expect(result).toBeDefined();
    expect(result.status).toBe('success');
    expect(global.fetch).toHaveBeenCalledWith(
      'http://ovms-server:9000/v1/models/bert/reload',
      expect.objectContaining({
        method: 'POST',
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
    
    (global.fetch as jest.Mock).mockResolvedValueOnce({
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
      'http://ovms-server:9000/v1/models/bert/quantization',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        }),
        body: JSON.stringify(config)
      })
    );
  });
  
  test('should infer with specific version', async () => {
    // Mock the infer method
    // @ts-ignore - Mocking method for testing
    backend.infer = jest.fn().mockResolvedValue([{ output: [0.1, 0.2, 0.3, 0.4] }]);
    
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
          output: [0.1, 0.2, 0.3, 0.4],
          importance: [0.01, 0.02, 0.03, 0.04]
        }
      ]
    };
    
    (global.fetch as jest.Mock).mockResolvedValueOnce({
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
  
  // Chat and Stream Tests
  
  test('should handle chat with OVMS adaptation', async () => {
    const messages: Message[] = [
      { role: 'user', content: 'Get embeddings for this text' }
    ];
    
    const response = await backend.chat(messages);
    
    expect(response).toBeDefined();
    expect(response.content).toBeDefined();
    expect(response.role).toBe('assistant');
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/v1/models/bert:predict'),
      expect.any(Object)
    );
  });
  
  test('should handle chat with empty user messages', async () => {
    const messages: Message[] = [
      { role: 'system', content: 'You are a helpful assistant.' }
    ];
    
    // Should throw since there's no user message
    await expect(backend.chat(messages)).rejects.toThrow('No user message found');
  });
  
  test('should handle chat with multimodal content', async () => {
    const messages: Message[] = [
      { 
        role: 'user', 
        content: [
          { type: 'text', text: 'Get embeddings for this text' },
          { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,ABC123' } }
        ]
      }
    ];
    
    const response = await backend.chat(messages);
    
    expect(response).toBeDefined();
    expect(response.content).toBeDefined();
    expect(global.fetch).toHaveBeenCalled();
  });
  
  test('should simulate streaming in OVMS', async () => {
    const messages: Message[] = [
      { role: 'user', content: 'Get embeddings for this text' }
    ];
    
    const stream = backend.streamChat(messages);
    
    // Collect items from the stream
    const chunks = [];
    for await (const chunk of await stream) {
      chunks.push(chunk);
    }
    
    expect(chunks.length).toBe(1); // OVMS doesn't support true streaming, so it returns a single chunk
    expect(chunks[0].type).toBe('result');
    expect(global.fetch).toHaveBeenCalled();
  });
  
  // Base API Implementation Tests
  
  test('should implement makePostRequest from base class', async () => {
    const data = { instances: [{ data: [0, 1, 2, 3] }] };
    const apiKey = 'custom-api-key';
    const options: ApiRequestOptions = {
      endpoint: 'http://custom-endpoint.com'
    };
    
    const response = await backend.makePostRequest(data, apiKey, options);
    
    expect(response).toBeDefined();
    expect(response.predictions).toBeDefined();
    expect(global.fetch).toHaveBeenCalledWith(
      'http://custom-endpoint.com',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        }),
        body: expect.any(String)
      })
    );
  });
  
  test('should implement makeStreamRequest from base class', async () => {
    const data = { instances: [{ data: [0, 1, 2, 3] }] };
    const options: ApiRequestOptions = {};
    
    const streamIterator = await backend.makeStreamRequest(data, options);
    
    expect(streamIterator).toBeDefined();
    
    // Collect chunks from the stream
    const chunks = [];
    for await (const chunk of streamIterator) {
      chunks.push(chunk);
    }
    
    expect(chunks.length).toBe(1); // OVMS doesn't support true streaming
    expect(chunks[0].type).toBe('result');
    expect(chunks[0].done).toBe(true);
    expect(global.fetch).toHaveBeenCalled();
  });
  
  // Model Compatibility Tests
  
  test('should check model compatibility', () => {
    // @ts-ignore - Need to mock getModelInfo for this test
    backend.getModelInfo = jest.fn().mockResolvedValue({ name: 'bert', versions: ['1'] });
    
    const compatibleModel = backend.isCompatibleModel('bert');
    expect(compatibleModel).toBe(true);
  });
  
  test('should handle network errors gracefully', async () => {
    // Mock network error
    (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));
    
    // @ts-ignore - Testing method not in interface but in implementation
    await expect(backend.getModelInfo('bert')).rejects.toThrow('Network error');
    
    // Should call console.error
    expect(console.error).toHaveBeenCalled;
  });
});