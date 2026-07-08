// Basic tests for VLLM Unified API Backend with Container Management
// For comprehensive tests, see vllm_unified_comprehensive.test.ts
import { VllmUnified } from '../../src/api_backends/vllm_unified/vllm_unified';
import { 
  VllmRequest,
  VllmUnifiedResponse,
  VllmBatchResponse,
  VllmStreamChunk,
  VllmModelInfo,
  VllmLoraAdapter,
  VllmQuantizationConfig,
  VllmContainerConfig,
  VllmContainerStatus
} from '../../src/api_backends/vllm_unified/types';
import { ApiMetadata, Message, ApiRequestOptions } from '../../src/api_backends/types';

// Mock child_process for container tests
jest.mock('child_process', () => ({
  execSync: jest.fn().mockReturnValue(Buffer.from('container-id')),
  spawn: jest.fn()
}));

// Mock os for platform checks and path operations
jest.mock('os', () => ({
  platform: jest.fn().mockReturnValue('linux'),
  homedir: jest.fn().mockReturnValue('/home/user')
}));

// Mock fs for file operations
jest.mock('fs', () => ({
  existsSync: jest.fn().mockReturnValue(true),
  writeFileSync: jest.fn(),
  readFileSync: jest.fn().mockReturnValue(Buffer.from('{}')),
  mkdirSync: jest.fn(),
  statSync: jest.fn().mockReturnValue({
    isDirectory: jest.fn().mockReturnValue(true)
  }),
  readdirSync: jest.fn().mockReturnValue(['file1', 'file2']),
  copyFileSync: jest.fn()
}));

// Mock path for path operations
jest.mock('path', () => ({
  join: jest.fn((...args) => args.join('/')),
  basename: jest.fn((path) => path.split('/').pop())
}));

// Mock fetch for testing
global.fetch = jest.fn();

// Mock AbortController for timeout
global.AbortController = class {
  signal = { aborted: false };
  abort() { this.signal.aborted = true; }
};
AbortController.timeout = jest.fn().mockReturnValue(new AbortController());

// Mock TextEncoder and TextDecoder for streaming response tests
global.TextEncoder = require('util').TextEncoder;
global.TextDecoder = require('util').TextDecoder;

// Mock ReadableStream for testing streaming
const mockReadableStream = () => {
  const chunks = [
    `data: ${JSON.stringify({
      text: 'Hello',
      metadata: { finish_reason: null, is_streaming: true }
    })}`,
    `data: ${JSON.stringify({
      text: ', world!',
      metadata: { finish_reason: null, is_streaming: true }
    })}`,
    `data: ${JSON.stringify({
      text: '',
      metadata: { finish_reason: 'stop', is_streaming: false }
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

describe('VLLM Unified API Backend with Container Management', () => {
  let backend: VllmUnified;
  let mockMetadata: ApiMetadata;
  const childProcess = require('child_process');
  const fs = require('fs');
  const os = require('os');
  
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock successful fetch response
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue({
        id: 'test-id',
        model: 'test-model',
        text: 'Hello, I am an AI assistant.',
        metadata: {
          finish_reason: 'stop',
          usage: {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30
          }
        }
      })
    });
    
    // Set up test data
    mockMetadata = {
      vllm_api_key: 'test-api-key',
      vllm_api_url: 'http://localhost:8000',
      vllm_model: 'meta-llama/Llama-2-7b-chat-hf',
      vllm_container_enabled: true,
      vllm_container_image: 'vllm/vllm-openai:latest',
      vllm_container_gpu: true,
      vllm_tensor_parallel_size: 2,
      vllm_gpu_memory_utilization: 0.8
    };
    
    // Create backend instance
    backend = new VllmUnified({}, mockMetadata);
  });
  
  // Core Functionality Tests
  
  test('should initialize with container configuration', () => {
    expect(backend).toBeDefined();
    
    // Check container configuration
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerEnabled).toBe(true);
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerConfig).toBeDefined();
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerConfig.image).toBe('vllm/vllm-openai:latest');
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerConfig.gpu).toBe(true);
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerConfig.tensor_parallel_size).toBe(2);
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerConfig.gpu_memory_utilization).toBe(0.8);
    
    // Check environment variables
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerEnvVars).toBeDefined();
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerEnvVars.VLLM_MODEL_PATH).toBe('meta-llama/Llama-2-7b-chat-hf');
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerEnvVars.VLLM_TENSOR_PARALLEL_SIZE).toBe('2');
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerEnvVars.VLLM_GPU_MEMORY_UTILIZATION).toBe('0.8');
    
    // Check that configuration directories are created
    expect(fs.mkdirSync).toHaveBeenCalled();
    expect(fs.writeFileSync).toHaveBeenCalled();
  });
  
  test('should initialize without container management when not enabled', () => {
    // Create backend instance without container management
    const nonContainerBackend = new VllmUnified({}, {
      vllm_api_url: 'http://localhost:8000',
      vllm_model: 'meta-llama/Llama-2-7b-chat-hf'
    });
    
    expect(nonContainerBackend).toBeDefined();
    // @ts-ignore - Accessing private property for testing
    expect(nonContainerBackend.containerEnabled).toBe(false);
    // @ts-ignore - Accessing private property for testing
    expect(nonContainerBackend.containerConfig).toBeNull();
  });
  
  test('should start container successfully', async () => {
    // Mock Docker available and image exists
    childProcess.execSync.mockImplementation((cmd: string) => {
      if (cmd.includes('docker --version')) {
        return Buffer.from('Docker version 20.10.21');
      } else if (cmd.includes('docker image inspect')) {
        return Buffer.from('{"Id": "image-id"}');
      } else if (cmd.includes('docker run')) {
        return Buffer.from('container-id');
      } else {
        return Buffer.from('');
      }
    });
    
    // Mock successful API connection
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({ models: ['model1', 'model2'] })
    });
    
    // Start the container
    const result = await backend.startContainer();
    
    // Check result
    expect(result).toBe(true);
    
    // Check container status and ID
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerStatus).toBe('running');
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerId).toBe('container-id');
    
    // Check Docker commands
    expect(childProcess.execSync).toHaveBeenCalledWith('docker --version', expect.anything());
    expect(childProcess.execSync).toHaveBeenCalledWith(expect.stringContaining('docker run -d --rm '), expect.anything());
    
    // Check health check is started
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerHealthCheckTimer).toBeDefined();
  });
  
  test('should handle container startup failure', async () => {
    // Mock Docker command failure
    childProcess.execSync.mockImplementationOnce(() => { throw new Error('Docker not found'); });
    
    // Start the container
    const result = await backend.startContainer();
    
    // Check result
    expect(result).toBe(false);
    
    // Check container status
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerStatus).toBe('error');
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerId).toBeNull();
  });
  
  test('should stop container successfully', async () => {
    // Setup container as running
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'running';
    // @ts-ignore - Setting private property for testing
    backend.containerId = 'container-id';
    
    // Stop the container
    const result = await backend.stopContainer();
    
    // Check result
    expect(result).toBe(true);
    
    // Check container status and ID
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerStatus).toBe('stopped');
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerId).toBeNull();
    
    // Check Docker command
    expect(childProcess.execSync).toHaveBeenCalledWith('docker stop container-id');
  });
  
  test('should handle container stop when not running', async () => {
    // Setup container as not running
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'stopped';
    // @ts-ignore - Setting private property for testing
    backend.containerId = null;
    
    // Stop the container
    const result = await backend.stopContainer();
    
    // Check result
    expect(result).toBe(true);
    
    // Check Docker command was not called
    expect(childProcess.execSync).not.toHaveBeenCalledWith(expect.stringContaining('docker stop'));
  });
  
  test('should restart container successfully', async () => {
    // Setup container as running
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'running';
    // @ts-ignore - Setting private property for testing
    backend.containerId = 'container-id';
    
    // Mock Docker command success
    childProcess.execSync.mockImplementation((cmd: string) => {
      if (cmd.includes('docker --version')) {
        return Buffer.from('Docker version 20.10.21');
      } else if (cmd.includes('docker stop')) {
        return Buffer.from('');
      } else if (cmd.includes('docker image inspect')) {
        return Buffer.from('{"Id": "image-id"}');
      } else if (cmd.includes('docker run')) {
        return Buffer.from('new-container-id');
      } else {
        return Buffer.from('');
      }
    });
    
    // Mock successful API connection
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({ models: ['model1', 'model2'] })
    });
    
    // Restart the container
    const result = await backend.restartContainer();
    
    // Check result
    expect(result).toBe(true);
    
    // Check container status and ID
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerStatus).toBe('running');
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerId).toBe('new-container-id');
    
    // Check Docker commands
    expect(childProcess.execSync).toHaveBeenCalledWith('docker stop container-id');
    expect(childProcess.execSync).toHaveBeenCalledWith(expect.stringContaining('docker run -d --rm '), expect.anything());
  });
  
  test('should get container status', () => {
    // Set status for test
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'running';
    
    // Get status
    const status = backend.getContainerStatus();
    
    // Check result
    expect(status).toBe('running');
  });
  
  test('should get container logs', () => {
    // Set logs for test
    const testLogs = ['[2025-03-16T12:00:00.000Z] Log entry 1', '[2025-03-16T12:01:00.000Z] Log entry 2'];
    // @ts-ignore - Setting private property for testing
    backend.containerLogs = testLogs;
    
    // Get logs
    const logs = backend.getContainerLogs();
    
    // Check result
    expect(logs).toEqual(testLogs);
    expect(logs).not.toBe(testLogs); // Should be a copy, not the same reference
  });
  
  test('should check container health', async () => {
    // Setup container as running
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'running';
    // @ts-ignore - Setting private property for testing
    backend.containerId = 'container-id';
    
    // Mock Docker command for container running
    childProcess.execSync.mockImplementationOnce((cmd: string) => {
      if (cmd.includes('docker inspect')) {
        return Buffer.from('true');
      } else {
        return Buffer.from('');
      }
    });
    
    // Mock successful API connection
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({ models: ['model1', 'model2'] })
    });
    
    // Check health
    // @ts-ignore - Calling private method for testing
    await backend.checkContainerHealth();
    
    // Container should still be running
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerStatus).toBe('running');
    
    // Docker inspect should have been called
    expect(childProcess.execSync).toHaveBeenCalledWith(expect.stringContaining('docker inspect container-id'));
  });
  
  test('should handle container stopped in health check', async () => {
    // Setup container as running
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'running';
    // @ts-ignore - Setting private property for testing
    backend.containerId = 'container-id';
    // @ts-ignore - Setting private property for testing
    backend.containerAutomaticRetryEnabled = false;
    
    // Mock Docker command for container stopped
    childProcess.execSync.mockImplementationOnce((cmd: string) => {
      if (cmd.includes('docker inspect')) {
        return Buffer.from('false');
      } else {
        return Buffer.from('');
      }
    });
    
    // Check health
    // @ts-ignore - Calling private method for testing
    await backend.checkContainerHealth();
    
    // Container should be marked as stopped
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerStatus).toBe('stopped');
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerId).toBeNull();
  });
  
  test('should handle API not responsive in health check', async () => {
    // Setup container as running
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'running';
    // @ts-ignore - Setting private property for testing
    backend.containerId = 'container-id';
    
    // Mock Docker command for container running
    childProcess.execSync.mockImplementationOnce((cmd: string) => {
      if (cmd.includes('docker inspect')) {
        return Buffer.from('true');
      } else {
        return Buffer.from('');
      }
    });
    
    // Mock API connection failure
    (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Connection refused'));
    
    // Check health
    // @ts-ignore - Calling private method for testing
    await backend.checkContainerHealth();
    
    // Container should still be running (we wait for Docker check to handle it)
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerStatus).toBe('running');
    
    // Logs should mention API failure
    // @ts-ignore - Accessing private property for testing
    const logs = backend.containerLogs;
    expect(logs[logs.length - 1]).toContain('API check failed');
  });
  
  test('should set container configuration', () => {
    // Update configuration
    const newConfig: Partial<VllmContainerConfig> = {
      gpu_memory_utilization: 0.9,
      tensor_parallel_size: 4,
      max_model_len: 4096,
      quantization: 'awq'
    };
    
    backend.setContainerConfig(newConfig);
    
    // Check configuration was updated
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerConfig.gpu_memory_utilization).toBe(0.9);
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerConfig.tensor_parallel_size).toBe(4);
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerConfig.max_model_len).toBe(4096);
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerConfig.quantization).toBe('awq');
    
    // Check environment variables were updated
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerEnvVars.VLLM_GPU_MEMORY_UTILIZATION).toBe('0.9');
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerEnvVars.VLLM_TENSOR_PARALLEL_SIZE).toBe('4');
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerEnvVars.VLLM_MAX_MODEL_LEN).toBe('4096');
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerEnvVars.VLLM_QUANTIZATION).toBe('awq');
    
    // Check configuration file was updated
    expect(fs.writeFileSync).toHaveBeenCalled();
  });
  
  test('should get container configuration', () => {
    // Set configuration for test
    // @ts-ignore - Setting private property for testing
    backend.containerConfig = {
      image: 'vllm/vllm-openai:latest',
      gpu: true,
      models_path: '/home/user/.vllm/models',
      config_path: '/home/user/.vllm',
      api_port: 8000,
      tensor_parallel_size: 2,
      max_model_len: 4096,
      gpu_memory_utilization: 0.8,
      quantization: 'awq',
      trust_remote_code: false,
      custom_args: ''
    };
    
    // Get configuration
    const config = backend.getContainerConfig();
    
    // Check result
    expect(config).toBeDefined();
    expect(config?.image).toBe('vllm/vllm-openai:latest');
    expect(config?.gpu).toBe(true);
    expect(config?.tensor_parallel_size).toBe(2);
    expect(config?.quantization).toBe('awq');
  });
  
  test('should enable container mode', () => {
    // Create backend without container mode
    const nonContainerBackend = new VllmUnified({}, {
      vllm_api_url: 'http://localhost:8000',
      vllm_model: 'meta-llama/Llama-2-7b-chat-hf'
    });
    
    // Check container mode is disabled
    // @ts-ignore - Accessing private property for testing
    expect(nonContainerBackend.containerEnabled).toBe(false);
    
    // Enable container mode
    nonContainerBackend.enableContainerMode({
      gpu: true,
      tensor_parallel_size: 4
    });
    
    // Check container mode is enabled
    // @ts-ignore - Accessing private property for testing
    expect(nonContainerBackend.containerEnabled).toBe(true);
    // @ts-ignore - Accessing private property for testing
    expect(nonContainerBackend.containerConfig).toBeDefined();
    // @ts-ignore - Accessing private property for testing
    expect(nonContainerBackend.containerConfig.gpu).toBe(true);
    // @ts-ignore - Accessing private property for testing
    expect(nonContainerBackend.containerConfig.tensor_parallel_size).toBe(4);
  });
  
  test('should disable container mode', async () => {
    // Setup container as running
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'running';
    // @ts-ignore - Setting private property for testing
    backend.containerId = 'container-id';
    
    // Disable container mode
    await backend.disableContainerMode();
    
    // Check container mode is disabled
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerEnabled).toBe(false);
    
    // Container should be stopped
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerStatus).toBe('stopped');
    
    // Docker stop should have been called
    expect(childProcess.execSync).toHaveBeenCalledWith('docker stop container-id');
  });
  
  test('should load models into container', async () => {
    // Setup container as running
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'running';
    // @ts-ignore - Setting private property for testing
    backend.containerId = 'container-id';
    
    // Load models
    const result = await backend.loadModels(['/path/to/model1', '/path/to/model2']);
    
    // Check result
    expect(result).toBe(true);
    
    // Check directory creation and file copying
    expect(fs.mkdirSync).toHaveBeenCalled();
    expect(fs.copyFileSync).toHaveBeenCalled();
  });
  
  test('should validate container capabilities', async () => {
    // Setup container as running
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'running';
    // @ts-ignore - Setting private property for testing
    backend.containerId = 'container-id';
    
    // Mock Docker command success
    childProcess.execSync.mockImplementation((cmd: string) => {
      if (cmd.includes('docker --version')) {
        return Buffer.from('Docker version 20.10.21');
      } else if (cmd.includes('nvidia-smi')) {
        return Buffer.from('NVIDIA-SMI 535.154.05');
      } else {
        return Buffer.from('');
      }
    });
    
    // Mock successful API connection
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({ models: ['model1', 'model2'] })
    }).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({ model: 'meta-llama/Llama-2-7b-chat-hf' })
    });
    
    // Validate capabilities
    const capabilities = await backend.validateContainerCapabilities();
    
    // Check result
    expect(capabilities).toBeDefined();
    expect(capabilities.docker_available).toBe(true);
    expect(capabilities.gpu_support).toBe(true);
    expect(capabilities.api_accessible).toBe(true);
    expect(capabilities.model_loaded).toBe(true);
  });
  
  test('should auto-start container when making a request', async () => {
    // Setup container as not running
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'stopped';
    // @ts-ignore - Setting private property for testing
    backend.containerId = null;
    
    // Mock Docker command success
    childProcess.execSync.mockImplementation((cmd: string) => {
      if (cmd.includes('docker --version')) {
        return Buffer.from('Docker version 20.10.21');
      } else if (cmd.includes('docker image inspect')) {
        return Buffer.from('{"Id": "image-id"}');
      } else if (cmd.includes('docker run')) {
        return Buffer.from('container-id');
      } else {
        return Buffer.from('');
      }
    });
    
    // Mock successful API connection for container startup
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({ models: ['model1', 'model2'] })
    }).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        id: 'test-id',
        model: 'test-model',
        text: 'Hello, I am an AI assistant.',
        metadata: {
          finish_reason: 'stop',
          usage: {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30
          }
        }
      })
    });
    
    // Make a request
    const result = await backend.makeRequest(
      'http://localhost:8000',
      { prompt: 'Hello', max_tokens: 10 },
      'test-model'
    );
    
    // Check request was successful
    expect(result).toBeDefined();
    
    // Container should be started
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerStatus).toBe('running');
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerId).toBe('container-id');
    
    // Docker commands should have been called
    expect(childProcess.execSync).toHaveBeenCalledWith('docker --version', expect.anything());
    expect(childProcess.execSync).toHaveBeenCalledWith(expect.stringContaining('docker run -d --rm '), expect.anything());
  });

  test('should successfully process batch with metrics', async () => {
    // Setup for test
    const batchData = ['Question 1', 'Question 2', 'Question 3'];
    const model = 'meta-llama/Llama-2-7b-chat-hf';
    
    // Mock API response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        texts: ['Answer 1', 'Answer 2', 'Answer 3'],
        metadata: {
          model: 'meta-llama/Llama-2-7b-chat-hf',
          finish_reasons: ['stop', 'stop', 'stop'],
          usage: {
            prompt_tokens: 30,
            completion_tokens: 60,
            total_tokens: 90
          }
        }
      })
    });
    
    // Process batch
    const [results, metrics] = await backend.processBatchWithMetrics(
      batchData,
      model
    );
    
    // Check results
    expect(results).toEqual(['Answer 1', 'Answer 2', 'Answer 3']);
    
    // Check metrics
    expect(metrics).toBeDefined();
    expect(metrics.model).toBe('meta-llama/Llama-2-7b-chat-hf');
    expect(metrics.batch_size).toBe(3);
    expect(metrics.successful_items).toBe(3);
    expect(metrics.total_time_ms).toBeGreaterThanOrEqual(0);
    expect(metrics.average_time_per_item_ms).toBeGreaterThanOrEqual(0);
    expect(metrics.usage).toBeDefined();
    expect(metrics.usage.total_tokens).toBe(90);
  });
  
  test('should stream generation with container support', async () => {
    // Setup for test
    const prompt = 'Tell me about quantum computing';
    const model = 'meta-llama/Llama-2-7b-chat-hf';
    
    // Mock API streaming response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      body: mockReadableStream(),
      status: 200
    });
    
    // Start streaming
    const stream = backend.streamGeneration(
      prompt,
      model
    );
    
    // Collect chunks from the stream
    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    
    // Check results
    expect(chunks).toEqual(['Hello', ', world!']);
    
    // Check API request was made correctly
    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/v1/completions',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"prompt":"Tell me about quantum computing"'),
        body: expect.stringContaining('"model":"meta-llama/Llama-2-7b-chat-hf"'),
        body: expect.stringContaining('"stream":true')
      })
    );
  });
  
  test('should get model info with container support', async () => {
    // Setup for test
    const model = 'meta-llama/Llama-2-7b-chat-hf';
    
    // Mock API response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        model: 'meta-llama/Llama-2-7b-chat-hf',
        max_model_len: 4096,
        num_gpu: 2,
        dtype: 'float16',
        gpu_memory_utilization: 0.8,
        quantization: {
          enabled: true,
          method: 'awq',
          bits: 4
        }
      })
    });
    
    // Get model info
    const modelInfo = await backend.getModelInfo(model);
    
    // Check results
    expect(modelInfo).toBeDefined();
    expect(modelInfo.model).toBe('meta-llama/Llama-2-7b-chat-hf');
    expect(modelInfo.max_model_len).toBe(4096);
    expect(modelInfo.num_gpu).toBe(2);
    expect(modelInfo.dtype).toBe('float16');
    expect(modelInfo.gpu_memory_utilization).toBe(0.8);
    expect(modelInfo.quantization).toBeDefined();
    expect(modelInfo.quantization?.enabled).toBe(true);
    expect(modelInfo.quantization?.method).toBe('awq');
    expect(modelInfo.quantization?.bits).toBe(4);
    
    // Check API request was made correctly
    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/v1/models/meta-llama/Llama-2-7b-chat-hf',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        })
      })
    );
  });
  
  test('should get model statistics with container support', async () => {
    // Setup for test
    const model = 'meta-llama/Llama-2-7b-chat-hf';
    
    // Mock API response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValue({
        model: 'meta-llama/Llama-2-7b-chat-hf',
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
    
    // Get model statistics
    const stats = await backend.getModelStatistics(model);
    
    // Check results
    expect(stats).toBeDefined();
    expect(stats.model).toBe('meta-llama/Llama-2-7b-chat-hf');
    expect(stats.statistics).toBeDefined();
    expect(stats.statistics.requests_processed).toBe(100);
    expect(stats.statistics.tokens_generated).toBe(5000);
    expect(stats.statistics.avg_tokens_per_request).toBe(50);
    expect(stats.statistics.throughput).toBe(100);
    
    // Check API request was made correctly
    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/v1/models/meta-llama/Llama-2-7b-chat-hf/statistics',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        })
      })
    );
  });
  
  test('should start container automatically when testing endpoint', async () => {
    // Setup container as not running
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'stopped';
    
    // Mock Docker command success
    childProcess.execSync.mockImplementation((cmd: string) => {
      if (cmd.includes('docker --version')) {
        return Buffer.from('Docker version 20.10.21');
      } else if (cmd.includes('docker image inspect')) {
        return Buffer.from('{"Id": "image-id"}');
      } else if (cmd.includes('docker run')) {
        return Buffer.from('container-id');
      } else {
        return Buffer.from('');
      }
    });
    
    // Mock successful API connection
    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({ // For the container readiness check
        ok: true,
        json: jest.fn().mockResolvedValue({ models: ['model1', 'model2'] })
      })
      .mockResolvedValueOnce({ // For the actual test endpoint call
        ok: true,
        json: jest.fn().mockResolvedValue({
          id: 'test-id',
          model: 'test-model',
          text: 'Hello, I am an AI assistant.',
          metadata: { finish_reason: 'stop' }
        })
      });
    
    // Test endpoint
    const result = await backend.testEndpoint();
    
    // Check result
    expect(result).toBe(true);
    
    // Container should be started
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerStatus).toBe('running');
  });
  
  test('should handle container start failure when testing endpoint', async () => {
    // Setup container as not running
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'stopped';
    
    // Mock Docker command failure
    childProcess.execSync.mockImplementationOnce(() => { throw new Error('Docker not found'); });
    
    // Test endpoint
    const result = await backend.testEndpoint();
    
    // Check result
    expect(result).toBe(false);
    
    // Container should be in error state
    // @ts-ignore - Accessing private property for testing
    expect(backend.containerStatus).toBe('error');
  });
  
  // Error Handling Tests
  
  test('should handle API errors with container', async () => {
    // Setup container as running
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'running';
    // @ts-ignore - Setting private property for testing
    backend.containerId = 'container-id';
    
    // Mock API error
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: jest.fn().mockResolvedValue({ error: 'Internal server error' })
    });
    
    // Make request
    await expect(backend.makeRequest(
      'http://localhost:8000',
      { prompt: 'Hello', max_tokens: 10 },
      'test-model'
    )).rejects.toThrow('Internal server error');
  });
  
  test('should handle timeout errors with container', async () => {
    // Setup container as running
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'running';
    // @ts-ignore - Setting private property for testing
    backend.containerId = 'container-id';
    
    // Mock timeout error
    (global.fetch as jest.Mock).mockImplementationOnce(() => {
      const error = new Error('AbortError');
      error.name = 'AbortError';
      throw error;
    });
    
    // Make request
    await expect(backend.makeRequest(
      'http://localhost:8000',
      { prompt: 'Hello', max_tokens: 10 },
      'test-model'
    )).rejects.toThrow('Request timed out');
  });
  
  test('should handle container start failure when making request', async () => {
    // Setup container as not running
    // @ts-ignore - Setting private property for testing
    backend.containerStatus = 'stopped';
    
    // Mock Docker command failure
    childProcess.execSync.mockImplementationOnce(() => { throw new Error('Docker not found'); });
    
    // Make request
    await expect(backend.makeRequest(
      'http://localhost:8000',
      { prompt: 'Hello', max_tokens: 10 },
      'test-model'
    )).rejects.toThrow('VLLM container failed to start');
  });
});