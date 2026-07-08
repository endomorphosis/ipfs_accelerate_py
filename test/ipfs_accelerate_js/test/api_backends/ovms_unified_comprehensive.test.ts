/**
 * Comprehensive test file for OVMS Unified API Backend
 * This test covers all aspects of the OVMS Unified backend
 */
import { OVMSUnified } from '../../src/api_backends/ovms_unified/ovms_unified';
import { OVMSModelConfig, OVMSPipelineConfig, OVMSResponse } from '../../src/api_backends/ovms_unified/types';
import { ApiMetadata, ApiRequestOptions, Message, PriorityLevel } from '../../src/api_backends/types';

// Mock child_process for container tests
jest.mock('child_process', () => ({
  execSync: jest.fn().mockReturnValue(Buffer.from('container-id')),
  spawn: jest.fn().mockImplementation(() => ({
    stdout: {
      on: jest.fn()
    },
    stderr: {
      on: jest.fn()
    },
    on: jest.fn()
  }))
}));

// Mock fs for file operations
jest.mock('fs', () => ({
  existsSync: jest.fn().mockReturnValue(true),
  writeFileSync: jest.fn(),
  readFileSync: jest.fn().mockReturnValue(Buffer.from('{"models": {}}'))
}));

// Mock fetch for API calls
global.fetch = jest.fn();

// Mock AbortSignal for timeout
global.AbortSignal = {
  timeout: jest.fn().mockReturnValue({
    aborted: false
  })
} as any;

describe('OVMS Unified API Backend', () => {
  let backend: OVMSUnified;
  let mockMetadata: ApiMetadata;
  
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock successful fetch response
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        model_version: "1",
        outputs: [
          {
            name: "output",
            shape: [1, 768],
            datatype: "FP32",
            data: Array(768).fill(0).map(() => Math.random())
          }
        ]
      })
    });
    
    // Set up test data
    mockMetadata = {
      ovms_api_url: 'http://localhost:8080',
      ovms_model_name: 'test-model',
      ovms_container_image: 'openvino/model-server:latest'
    };
    
    // Create backend instance
    backend = new OVMSUnified({}, mockMetadata);
  });
  
  // Core functionality tests
  
  describe('Core initialization and configuration', () => {
    test('should initialize correctly', () => {
      expect(backend).toBeDefined();
      
      // Test with custom API URL
      const customBackend = new OVMSUnified({}, { 
        ovms_api_url: 'http://custom-server:9000',
        ovms_model_name: 'custom-model'
      });
      expect(customBackend).toBeDefined();
      
      // @ts-ignore - Accessing protected property for testing
      expect(customBackend.apiUrl).toBe('http://custom-server:9000');
      // @ts-ignore - Accessing protected property for testing
      expect(customBackend.defaultModel).toBe('custom-model');
    });
    
    test('should set default model name from metadata', () => {
      // @ts-ignore - Testing protected method
      const modelName = backend.getDefaultModel();
      expect(modelName).toBe('test-model');
      
      // Test with custom model name
      const customBackend = new OVMSUnified({}, { 
        ovms_model_name: 'another-model'
      });
      // @ts-ignore - Testing protected method
      expect(customBackend.getDefaultModel()).toBe('another-model');
      
      // Test with missing model name
      const fallbackBackend = new OVMSUnified({}, {});
      // @ts-ignore - Testing protected method
      expect(fallbackBackend.getDefaultModel()).toBe('bert-base-uncased');
    });
    
    test('should extract container configuration from metadata', () => {
      // @ts-ignore - Testing protected property
      expect(backend.containerImage).toBe('openvino/model-server:latest');
      
      // Test with custom container config
      const customBackend = new OVMSUnified({}, { 
        ovms_container_image: 'custom/image:latest',
        ovms_container_ports: '8080:8080'
      });
      // @ts-ignore - Testing protected property
      expect(customBackend.containerImage).toBe('custom/image:latest');
      // @ts-ignore - Testing protected property
      expect(customBackend.containerPorts).toBe('8080:8080');
    });
  });
  
  describe('API endpoint and request handling', () => {
    test('should create endpoint handler', () => {
      const handler = backend.createEndpointHandler();
      expect(handler).toBeDefined();
      expect(typeof handler).toBe('function');
      
      // Test with custom URL
      const customHandler = backend.createEndpointHandler('http://custom-endpoint:8080');
      expect(customHandler).toBeDefined();
      expect(typeof customHandler).toBe('function');
    });
    
    test('should test endpoint', async () => {
      const result = await backend.testEndpoint();
      expect(result).toBe(true);
      expect(global.fetch).toHaveBeenCalled();
      
      // Test with custom URL and model
      const customResult = await backend.testEndpoint('http://custom-endpoint:8080', 'custom-model');
      expect(customResult).toBe(true);
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });
    
    test('should handle endpoint failures', async () => {
      // Mock fetch failure
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Connection refused'));
      
      const result = await backend.testEndpoint();
      expect(result).toBe(false);
      
      // Mock HTTP error
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found'
      });
      
      const result2 = await backend.testEndpoint();
      expect(result2).toBe(false);
    });
    
    test('should make POST request', async () => {
      const data = {
        inputs: [
          {
            name: "input_ids",
            shape: [1, 10],
            datatype: "INT64",
            data: Array(10).fill(0).map((_, i) => i)
          }
        ]
      };
      
      const response = await backend.makePostRequest(data);
      
      expect(response).toBeDefined();
      expect(global.fetch).toHaveBeenCalled();
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/models/test-model/infer'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json'
          }),
          body: expect.stringContaining('"input_ids"')
        })
      );
      
      // Test with options
      const options: ApiRequestOptions = {
        endpoint: 'http://custom-endpoint:8080',
        model: 'custom-model',
        priority: PriorityLevel.HIGH,
        timeoutMs: 5000
      };
      
      const customResponse = await backend.makePostRequest(data, undefined, options);
      
      expect(customResponse).toBeDefined();
      expect(global.fetch).toHaveBeenCalledTimes(2);
      expect(global.fetch).toHaveBeenLastCalledWith(
        'http://custom-endpoint:8080',
        expect.any(Object)
      );
    });
    
    test('should handle API errors', async () => {
      // Mock error response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: jest.fn().mockResolvedValue({
          error: 'Invalid input format'
        })
      });
      
      const data = {
        inputs: [
          {
            name: "input_ids",
            shape: [1, 10],
            datatype: "INT64",
            data: Array(10).fill(0).map((_, i) => i)
          }
        ]
      };
      
      await expect(backend.makePostRequest(data)).rejects.toThrow('OVMS API error: Bad Request');
    });
    
    test('should handle network errors', async () => {
      // Mock network error
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));
      
      const data = {
        inputs: [
          {
            name: "input_ids",
            shape: [1, 10],
            datatype: "INT64",
            data: Array(10).fill(0).map((_, i) => i)
          }
        ]
      };
      
      await expect(backend.makePostRequest(data)).rejects.toThrow('Network error');
    });
  });
  
  describe('Model management', () => {
    test('should check model compatibility', () => {
      // Test with compatible models
      expect(backend.isCompatibleModel('bert-base-uncased')).toBe(true);
      expect(backend.isCompatibleModel('resnet50')).toBe(true);
      expect(backend.isCompatibleModel('yolov5')).toBe(true);
      
      // Test with OpenVINO-specific models
      expect(backend.isCompatibleModel('openvino-bert')).toBe(true);
      expect(backend.isCompatibleModel('intel/bert-base')).toBe(true);
      
      // Test with incompatible models
      expect(backend.isCompatibleModel('llama-7b')).toBe(false);
      expect(backend.isCompatibleModel('gpt-4')).toBe(false);
      expect(backend.isCompatibleModel('claude-3')).toBe(false);
    });
    
    test('should get model metadata', async () => {
      // Mock model metadata response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          name: 'test-model',
          version: '1',
          platform: 'openvino',
          inputs: [
            {
              name: 'input_ids',
              datatype: 'INT64',
              shape: [1, 10]
            }
          ],
          outputs: [
            {
              name: 'output',
              datatype: 'FP32',
              shape: [1, 768]
            }
          ]
        })
      });
      
      // @ts-ignore - Testing method not exposed in interface
      const metadata = await backend.getModelMetadata('test-model');
      
      expect(metadata).toBeDefined();
      expect(metadata.name).toBe('test-model');
      expect(metadata.version).toBe('1');
      expect(metadata.inputs.length).toBe(1);
      expect(metadata.outputs.length).toBe(1);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/models/test-model'),
        expect.any(Object)
      );
    });
    
    test('should list models', async () => {
      // Mock model list response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          models: [
            {
              name: 'bert-base-uncased',
              version: '1',
              state: 'AVAILABLE'
            },
            {
              name: 'resnet50',
              version: '1',
              state: 'AVAILABLE'
            }
          ]
        })
      });
      
      // @ts-ignore - Testing method not exposed in interface
      const models = await backend.listModels();
      
      expect(models).toBeDefined();
      expect(Array.isArray(models)).toBe(true);
      expect(models.length).toBe(2);
      expect(models[0].name).toBe('bert-base-uncased');
      expect(models[1].name).toBe('resnet50');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/models'),
        expect.any(Object)
      );
    });
    
    test('should handle errors in model operations', async () => {
      // Mock error response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found'
      });
      
      // @ts-ignore - Testing method not exposed in interface
      await expect(backend.getModelMetadata('non-existent-model')).rejects.toThrow('Failed to get model metadata: Not Found');
    });
    
    test('should load model', async () => {
      // Mock success response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          status: 'success',
          model_name: 'new-model',
          model_version: '1'
        })
      });
      
      const modelConfig: OVMSModelConfig = {
        model_name: 'new-model',
        model_path: '/models/new-model',
        model_version: '1',
        device: 'CPU'
      };
      
      // @ts-ignore - Testing method not exposed in interface
      const result = await backend.loadModel(modelConfig);
      
      expect(result).toBeDefined();
      expect(result.status).toBe('success');
      expect(result.model_name).toBe('new-model');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/models/load'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"model_name":"new-model"')
        })
      );
    });
    
    test('should unload model', async () => {
      // Mock success response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          status: 'success',
          model_name: 'test-model'
        })
      });
      
      // @ts-ignore - Testing method not exposed in interface
      const result = await backend.unloadModel('test-model');
      
      expect(result).toBeDefined();
      expect(result.status).toBe('success');
      expect(result.model_name).toBe('test-model');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/models/unload'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"model_name":"test-model"')
        })
      );
    });
  });
  
  describe('Pipeline management', () => {
    test('should create pipeline', async () => {
      // Mock success response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          status: 'success',
          pipeline_name: 'test-pipeline'
        })
      });
      
      const pipelineConfig: OVMSPipelineConfig = {
        pipeline_name: 'test-pipeline',
        nodes: [
          {
            name: 'preprocessing',
            model_name: 'preprocessor',
            type: 'DL model'
          },
          {
            name: 'inference',
            model_name: 'test-model',
            type: 'DL model'
          },
          {
            name: 'postprocessing',
            model_name: 'postprocessor',
            type: 'DL model'
          }
        ],
        connections: [
          {
            source: 'preprocessing:output',
            target: 'inference:input'
          },
          {
            source: 'inference:output',
            target: 'postprocessing:input'
          }
        ]
      };
      
      // @ts-ignore - Testing method not exposed in interface
      const result = await backend.createPipeline(pipelineConfig);
      
      expect(result).toBeDefined();
      expect(result.status).toBe('success');
      expect(result.pipeline_name).toBe('test-pipeline');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/pipelines/create'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"pipeline_name":"test-pipeline"')
        })
      );
    });
    
    test('should remove pipeline', async () => {
      // Mock success response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          status: 'success',
          pipeline_name: 'test-pipeline'
        })
      });
      
      // @ts-ignore - Testing method not exposed in interface
      const result = await backend.removePipeline('test-pipeline');
      
      expect(result).toBeDefined();
      expect(result.status).toBe('success');
      expect(result.pipeline_name).toBe('test-pipeline');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/pipelines/remove'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"pipeline_name":"test-pipeline"')
        })
      );
    });
    
    test('should list pipelines', async () => {
      // Mock pipeline list response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          pipelines: [
            {
              name: 'pipeline1',
              nodes: 3,
              connections: 2
            },
            {
              name: 'pipeline2',
              nodes: 2,
              connections: 1
            }
          ]
        })
      });
      
      // @ts-ignore - Testing method not exposed in interface
      const pipelines = await backend.listPipelines();
      
      expect(pipelines).toBeDefined();
      expect(Array.isArray(pipelines)).toBe(true);
      expect(pipelines.length).toBe(2);
      expect(pipelines[0].name).toBe('pipeline1');
      expect(pipelines[1].name).toBe('pipeline2');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/pipelines'),
        expect.any(Object)
      );
    });
    
    test('should handle errors in pipeline operations', async () => {
      // Mock error response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request'
      });
      
      const pipelineConfig: OVMSPipelineConfig = {
        pipeline_name: 'invalid-pipeline',
        nodes: [],
        connections: []
      };
      
      // @ts-ignore - Testing method not exposed in interface
      await expect(backend.createPipeline(pipelineConfig)).rejects.toThrow('Failed to create pipeline: Bad Request');
    });
  });
  
  describe('Container management', () => {
    test('should start container', async () => {
      const child_process = require('child_process');
      
      // @ts-ignore - Testing method not exposed in interface
      const containerId = await backend.startContainer();
      
      expect(containerId).toBe('container-id');
      expect(child_process.execSync).toHaveBeenCalled();
      expect(child_process.execSync).toHaveBeenCalledWith(
        expect.stringContaining('docker run'),
        expect.any(Object)
      );
    });
    
    test('should stop container', async () => {
      const child_process = require('child_process');
      
      // @ts-ignore - Testing method not exposed in interface
      await backend.stopContainer('container-id');
      
      expect(child_process.execSync).toHaveBeenCalled();
      expect(child_process.execSync).toHaveBeenCalledWith(
        expect.stringContaining('docker stop container-id'),
        expect.any(Object)
      );
    });
    
    test('should handle container errors', async () => {
      const child_process = require('child_process');
      
      // Mock command error
      child_process.execSync.mockImplementationOnce(() => {
        throw new Error('Container error');
      });
      
      // @ts-ignore - Testing method not exposed in interface
      await expect(backend.startContainer()).rejects.toThrow('Failed to start OVMS container: Container error');
    });
    
    test('should write configuration file', async () => {
      const fs = require('fs');
      
      const config = {
        model_config_list: [
          {
            config: {
              name: 'test-model',
              base_path: '/models/test-model',
              target_device: 'CPU'
            }
          }
        ]
      };
      
      // @ts-ignore - Testing method not exposed in interface
      await backend.writeConfigFile(config, '/tmp/config.json');
      
      expect(fs.writeFileSync).toHaveBeenCalled();
      expect(fs.writeFileSync).toHaveBeenCalledWith(
        '/tmp/config.json',
        expect.stringContaining('test-model'),
        'utf8'
      );
    });
  });
  
  describe('Embedding and inference', () => {
    test('should get embeddings', async () => {
      // Mock embedding response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          model_version: "1",
          outputs: [
            {
              name: "embedding",
              shape: [1, 768],
              datatype: "FP32",
              data: Array(768).fill(0).map(() => Math.random())
            }
          ]
        })
      });
      
      const embeddings = await backend.getEmbedding('Hello, world\!');
      
      expect(embeddings).toBeDefined();
      expect(Array.isArray(embeddings)).toBe(true);
      expect(embeddings.length).toBe(768);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/models/test-model/infer'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('Hello, world\!')
        })
      );
      
      // Test with options
      const optionsEmbeddings = await backend.getEmbedding('Another text', {
        model: 'custom-model'
      });
      
      expect(optionsEmbeddings).toBeDefined();
      expect(global.fetch).toHaveBeenCalledTimes(2);
      expect(global.fetch).toHaveBeenLastCalledWith(
        expect.stringContaining('/v1/models/custom-model/infer'),
        expect.any(Object)
      );
      
      // Test with array of texts
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          model_version: "1",
          outputs: [
            {
              name: "embedding",
              shape: [2, 768],
              datatype: "FP32",
              data: Array(2 * 768).fill(0).map(() => Math.random())
            }
          ]
        })
      });
      
      const multiEmbeddings = await backend.getEmbedding(['Text 1', 'Text 2']);
      
      expect(multiEmbeddings).toBeDefined();
      expect(Array.isArray(multiEmbeddings)).toBe(true);
      expect(multiEmbeddings.length).toBe(2);
      expect(Array.isArray(multiEmbeddings[0])).toBe(true);
      expect(multiEmbeddings[0].length).toBe(768);
    });
    
    test('should handle errors in getEmbedding', async () => {
      // Mock error response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      });
      
      await expect(backend.getEmbedding('Hello, world\!')).rejects.toThrow('OVMS API error: Internal Server Error');
    });
    
    test('should infer using custom input format', async () => {
      // Mock inference response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          model_version: "1",
          outputs: [
            {
              name: "detection_output",
              shape: [1, 1, 100, 7],
              datatype: "FP32",
              data: Array(700).fill(0).map(() => Math.random())
            }
          ]
        })
      });
      
      const imageData = new Float32Array(3 * 224 * 224).fill(0).map(() => Math.random());
      
      const inferenceData = {
        inputs: [
          {
            name: "image_tensor",
            shape: [1, 3, 224, 224],
            datatype: "FP32",
            data: Array.from(imageData)
          }
        ]
      };
      
      // @ts-ignore - Testing method not exposed in interface
      const result = await backend.infer(inferenceData, 'detection-model');
      
      expect(result).toBeDefined();
      expect(result.outputs).toBeDefined();
      expect(result.outputs.length).toBe(1);
      expect(result.outputs[0].name).toBe('detection_output');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/models/detection-model/infer'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"image_tensor"')
        })
      );
    });
  });
  
  describe('Chat interface', () => {
    test('should implement chat method', async () => {
      // Mock chat response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          model_version: "1",
          outputs: [
            {
              name: "text_output",
              shape: [1],
              datatype: "STRING",
              data: ["Hello, I am an AI assistant."]
            }
          ]
        })
      });
      
      const messages: Message[] = [
        { role: 'user', content: 'Hi, who are you?' }
      ];
      
      const response = await backend.chat(messages);
      
      expect(response).toBeDefined();
      expect(response.content).toBe('Hello, I am an AI assistant.');
      expect(response.model).toBe('test-model');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/models/test-model/infer'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('Hi, who are you?')
        })
      );
      
      // Test with options
      const optionsResponse = await backend.chat(messages, {
        model: 'custom-model',
        temperature: 0.7,
        maxTokens: 100
      });
      
      expect(optionsResponse).toBeDefined();
      expect(global.fetch).toHaveBeenCalledTimes(2);
      expect(global.fetch).toHaveBeenLastCalledWith(
        expect.stringContaining('/v1/models/custom-model/infer'),
        expect.any(Object)
      );
    });
    
    test('should handle errors in chat', async () => {
      // Mock error response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      });
      
      const messages: Message[] = [
        { role: 'user', content: 'Hi, who are you?' }
      ];
      
      await expect(backend.chat(messages)).rejects.toThrow('OVMS API error: Internal Server Error');
    });
    
    test('should handle message formatting in chat', async () => {
      // Mock chat response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          model_version: "1",
          outputs: [
            {
              name: "text_output",
              shape: [1],
              datatype: "STRING",
              data: ["Hello, I am an AI assistant."]
            }
          ]
        })
      });
      
      const messages: Message[] = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'Hi, who are you?' }
      ];
      
      const response = await backend.chat(messages);
      
      expect(response).toBeDefined();
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('You are a helpful assistant')
        })
      );
    });
  });
  
  describe('API-specific methods', () => {
    test('should get server status', async () => {
      // Mock server status response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          version: "2022.3",
          online_models: 5,
          running_since: "2023-01-01T00:00:00Z",
          status: "online"
        })
      });
      
      // @ts-ignore - Testing method not exposed in interface
      const status = await backend.getServerStatus();
      
      expect(status).toBeDefined();
      expect(status.version).toBe("2022.3");
      expect(status.online_models).toBe(5);
      expect(status.status).toBe("online");
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/status'),
        expect.any(Object)
      );
    });
    
    test('should get model performance statistics', async () => {
      // Mock performance stats response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          model_name: "test-model",
          version: "1",
          execution_count: 100,
          avg_latency_ms: 15.5,
          min_latency_ms: 10.2,
          max_latency_ms: 25.7,
          success_count: 98,
          error_count: 2
        })
      });
      
      // @ts-ignore - Testing method not exposed in interface
      const stats = await backend.getModelPerformance('test-model');
      
      expect(stats).toBeDefined();
      expect(stats.model_name).toBe("test-model");
      expect(stats.execution_count).toBe(100);
      expect(stats.avg_latency_ms).toBe(15.5);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/models/test-model/performance'),
        expect.any(Object)
      );
    });
    
    test('should handle errors in API-specific methods', async () => {
      // Mock error response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found'
      });
      
      // @ts-ignore - Testing method not exposed in interface
      await expect(backend.getModelPerformance('non-existent-model')).rejects.toThrow('Failed to get model performance: Not Found');
    });
  });
  
  describe('Configuration management', () => {
    test('should update server configuration', async () => {
      // Mock success response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          status: 'success',
          config: {
            log_level: 'INFO',
            server_threads: 8
          }
        })
      });
      
      const config = {
        log_level: 'INFO',
        server_threads: 8
      };
      
      // @ts-ignore - Testing method not exposed in interface
      const result = await backend.updateServerConfig(config);
      
      expect(result).toBeDefined();
      expect(result.status).toBe('success');
      expect(result.config.log_level).toBe('INFO');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/config'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"log_level":"INFO"')
        })
      );
    });
    
    test('should get current server configuration', async () => {
      // Mock config response
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({
          log_level: 'INFO',
          server_threads: 8,
          model_directory: '/models',
          model_config_path: '/config/model_config.json'
        })
      });
      
      // @ts-ignore - Testing method not exposed in interface
      const config = await backend.getServerConfig();
      
      expect(config).toBeDefined();
      expect(config.log_level).toBe('INFO');
      expect(config.server_threads).toBe(8);
      expect(config.model_directory).toBe('/models');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/config'),
        expect.objectContaining({
          method: 'GET'
        })
      );
    });
  });
  
  describe('Error handling and recovery', () => {
    test('should handle and create API errors', () => {
      // @ts-ignore - Testing protected method
      const error = backend.createApiError('Test error', 400, 'bad_request');
      
      expect(error).toBeDefined();
      expect(error.message).toContain('Test error');
      expect(error.code).toBe(400);
      expect(error.type).toBe('bad_request');
      
      // Test with default values
      // @ts-ignore - Testing protected method
      const defaultError = backend.createApiError('Another error');
      
      expect(defaultError).toBeDefined();
      expect(defaultError.message).toContain('Another error');
      expect(defaultError.code).toBe(500);
      expect(defaultError.type).toBe('api_error');
    });
    
    test('should implement circuit breaker pattern', async () => {
      // Create backend with circuit breaker config
      const circuitBackend = new OVMSUnified({}, {
        ...mockMetadata,
        circuit_breaker_threshold: 3,
        circuit_breaker_timeout_ms: 5000
      });
      
      // Mock repeated failures to trigger circuit breaker
      (global.fetch as jest.Mock)
        .mockRejectedValueOnce(new Error('Server error 1'))
        .mockRejectedValueOnce(new Error('Server error 2'))
        .mockRejectedValueOnce(new Error('Server error 3'));
      
      // First failure
      try {
        await circuitBackend.makePostRequest({});
      } catch (error) {
        expect(error.message).toContain('Server error 1');
      }
      
      // Second failure
      try {
        await circuitBackend.makePostRequest({});
      } catch (error) {
        expect(error.message).toContain('Server error 2');
      }
      
      // Third failure should trigger circuit breaker
      try {
        await circuitBackend.makePostRequest({});
      } catch (error) {
        expect(error.message).toContain('Server error 3');
      }
      
      // Fourth attempt should fail fast with circuit breaker error
      try {
        await circuitBackend.makePostRequest({});
      } catch (error) {
        expect(error.message).toContain('Circuit breaker is open');
      }
      
      // Verify only 3 API calls were made (not 4)
      expect(global.fetch).toHaveBeenCalledTimes(3);
    });
    
    test('should implement request queue with priority', async () => {
      // Create backend with queue config
      const queueBackend = new OVMSUnified({}, {
        ...mockMetadata,
        queue_size: 5
      });
      
      // Mock successful response after delay
      (global.fetch as jest.Mock).mockImplementation(() => {
        return new Promise(resolve => {
          setTimeout(() => {
            resolve({
              ok: true,
              json: () => Promise.resolve({
                model_version: "1",
                outputs: [
                  {
                    name: "output",
                    shape: [1, 768],
                    datatype: "FP32",
                    data: Array(768).fill(0).map(() => Math.random())
                  }
                ]
              })
            });
          }, 100);
        });
      });
      
      // Send multiple requests with different priorities
      const highPriorityPromise = queueBackend.makePostRequest({}, undefined, { priority: PriorityLevel.HIGH });
      const normalPriorityPromise = queueBackend.makePostRequest({}, undefined, { priority: PriorityLevel.NORMAL });
      const lowPriorityPromise = queueBackend.makePostRequest({}, undefined, { priority: PriorityLevel.LOW });
      
      // All should complete successfully
      const results = await Promise.all([highPriorityPromise, normalPriorityPromise, lowPriorityPromise]);
      
      expect(results).toBeDefined();
      expect(results.length).toBe(3);
      expect(results[0]).toBeDefined();
      expect(results[1]).toBeDefined();
      expect(results[2]).toBeDefined();
      
      // Verify 3 API calls were made
      expect(global.fetch).toHaveBeenCalledTimes(3);
    });
    
    test('should implement retries for transient errors', async () => {
      // Create backend with retry config
      const retryBackend = new OVMSUnified({}, {
        ...mockMetadata,
        retry_count: 2,
        retry_delay_ms: 100
      });
      
      // Mock failure then success
      (global.fetch as jest.Mock)
        .mockRejectedValueOnce(new Error('Transient error'))
        .mockResolvedValueOnce({
          ok: true,
          json: jest.fn().mockResolvedValue({
            model_version: "1",
            outputs: [
              {
                name: "output",
                shape: [1, 768],
                datatype: "FP32",
                data: Array(768).fill(0).map(() => Math.random())
              }
            ]
          })
        });
      
      // Should retry and eventually succeed
      const result = await retryBackend.makePostRequest({});
      
      expect(result).toBeDefined();
      
      // Verify 2 API calls were made (1 failure, 1 success)
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });
  });
});
