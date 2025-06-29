/**
 * WebGPU Tensor Sharing Integration Tests
 * 
 * Tests for WebGPU integration with the Cross-Model Tensor Sharing system
 */

import { WebGPUTensorSharing, SharedTensorLocation } from './ipfs_accelerate_js_webgpu_tensor_sharing';
import { TensorSharingIntegration } from './ipfs_accelerate_js_tensor_sharing_integration';
import { WebGPUBackend } from './ipfs_accelerate_js_webgpu_backend';
import { StorageManager } from './ipfs_accelerate_js_storage_manager';
import { SharedTensor } from './ipfs_accelerate_js_shared_tensor';

// Mock implementations for testing
jest.mock('./ipfs_accelerate_js_webgpu_backend');
jest.mock('./ipfs_accelerate_js_tensor_sharing_integration');
jest.mock('./ipfs_accelerate_js_storage_manager');
jest.mock('./ipfs_accelerate_js_shared_tensor');

describe('WebGPUTensorSharing', () => {
  let tensorSharing: TensorSharingIntegration;
  let webgpuBackend: WebGPUBackend;
  let storageManager: StorageManager;
  let webgpuTensorSharing: WebGPUTensorSharing;
  
  // Setup before each test
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Create mock instances
    tensorSharing = new TensorSharingIntegration();
    webgpuBackend = new WebGPUBackend();
    storageManager = new StorageManager();
    
    // Setup WebGPU backend mock
    (webgpuBackend.isSupported as jest.Mock).mockResolvedValue(true);
    (webgpuBackend.initialize as jest.Mock).mockResolvedValue(true);
    
    // Setup tensor sharing mock
    (tensorSharing.getSharedTensor as jest.Mock).mockImplementation((name, model) => {
      const tensor = new SharedTensor();
      (tensor.getData as jest.Mock).mockReturnValue(new Float32Array([1, 2, 3, 4]));
      (tensor.getShape as jest.Mock).mockReturnValue([2, 2]);
      (tensor.getDataType as jest.Mock).mockReturnValue('float32');
      (tensor.getSize as jest.Mock).mockReturnValue(4);
      return Promise.resolve(tensor);
    });
    
    // Create webgpu tensor sharing instance
    webgpuTensorSharing = new WebGPUTensorSharing(tensorSharing, {
      webgpuBackend,
      storageManager,
      debug: false
    });
  });
  
  describe('initialization', () => {
    test('should initialize successfully when WebGPU is supported', async () => {
      const result = await webgpuTensorSharing.initialize();
      
      expect(result).toBe(true);
      expect(webgpuBackend.isSupported).toHaveBeenCalled();
      expect(webgpuBackend.initialize).toHaveBeenCalled();
      expect(webgpuTensorSharing.isInitialized()).toBe(true);
    });
    
    test('should fail initialization when WebGPU is not supported', async () => {
      (webgpuBackend.isSupported as jest.Mock).mockResolvedValue(false);
      
      const result = await webgpuTensorSharing.initialize();
      
      expect(result).toBe(false);
      expect(webgpuBackend.isSupported).toHaveBeenCalled();
      expect(webgpuBackend.initialize).not.toHaveBeenCalled();
      expect(webgpuTensorSharing.isInitialized()).toBe(false);
    });
    
    test('should fail initialization when WebGPU backend fails to initialize', async () => {
      (webgpuBackend.initialize as jest.Mock).mockResolvedValue(false);
      
      const result = await webgpuTensorSharing.initialize();
      
      expect(result).toBe(false);
      expect(webgpuBackend.isSupported).toHaveBeenCalled();
      expect(webgpuBackend.initialize).toHaveBeenCalled();
      expect(webgpuTensorSharing.isInitialized()).toBe(false);
    });
  });
  
  describe('GPU buffer creation', () => {
    beforeEach(async () => {
      await webgpuTensorSharing.initialize();
      
      // Setup WebGPU backend mock for creating tensors
      (webgpuBackend.createTensor as jest.Mock).mockImplementation((data, shape, dataType) => {
        const buffer = {} as GPUBuffer;
        return Promise.resolve({
          buffer,
          shape,
          dataType,
          size: data.byteLength
        });
      });
    });
    
    test('should create GPU buffer from shared tensor', async () => {
      const result = await webgpuTensorSharing.createGPUBufferFromSharedTensor('tensor1', 'model1');
      
      expect(tensorSharing.getSharedTensor).toHaveBeenCalledWith('tensor1', 'model1');
      expect(webgpuBackend.createTensor).toHaveBeenCalled();
      expect(result.shape).toEqual([2, 2]);
      expect(result.dataType).toBe('float32');
      expect(result.location).toBe(SharedTensorLocation.WEBGPU);
    });
    
    test('should return cached GPU buffer if already created', async () => {
      // Create first time
      await webgpuTensorSharing.createGPUBufferFromSharedTensor('tensor1', 'model1');
      
      // Reset mocks to verify cache hit
      jest.clearAllMocks();
      
      // Request same tensor again
      const result = await webgpuTensorSharing.createGPUBufferFromSharedTensor('tensor1', 'model1');
      
      // Should not call these again
      expect(tensorSharing.getSharedTensor).not.toHaveBeenCalled();
      expect(webgpuBackend.createTensor).not.toHaveBeenCalled();
      
      expect(result.shape).toEqual([2, 2]);
      expect(result.dataType).toBe('float32');
      expect(result.location).toBe(SharedTensorLocation.WEBGPU);
    });
    
    test('should throw error when shared tensor not found', async () => {
      (tensorSharing.getSharedTensor as jest.Mock).mockResolvedValue(null);
      
      await expect(
        webgpuTensorSharing.createGPUBufferFromSharedTensor('nonexistent', 'model1')
      ).rejects.toThrow('Shared tensor nonexistent not found for model model1');
    });
  });
  
  describe('tensor operations', () => {
    beforeEach(async () => {
      await webgpuTensorSharing.initialize();
      
      // Setup WebGPU backend mock for tensor creation and operations
      (webgpuBackend.createTensor as jest.Mock).mockImplementation((data, shape, dataType) => {
        const buffer = {} as GPUBuffer;
        return Promise.resolve({
          buffer,
          shape,
          dataType,
          size: data.byteLength
        });
      });
      
      // Setup buffer read mock
      (webgpuBackend.readBuffer as jest.Mock).mockResolvedValue(new Float32Array([5, 6, 7, 8]));
      
      // Setup operation execution mock
      (webgpuBackend.execute as jest.Mock).mockImplementation((operation, inputs, options) => {
        const buffer = {} as GPUBuffer;
        const shape = [2, 2];
        
        if (operation === 'quantize') {
          const scaleBuffer = {} as GPUBuffer;
          return Promise.resolve({
            buffer,
            scale: scaleBuffer,
            shape,
            dataType: 'int32'
          });
        }
        
        return Promise.resolve({
          buffer,
          shape,
          dataType: 'float32'
        });
      });
    });
    
    test('should execute matrix multiplication operation', async () => {
      const result = await webgpuTensorSharing.matmul(
        'tensor1', 'model1',
        'tensor2', 'model2',
        'output', 'model3'
      );
      
      expect(webgpuBackend.execute).toHaveBeenCalledWith(
        'matmul',
        expect.objectContaining({
          a: expect.anything(),
          b: expect.anything()
        }),
        expect.objectContaining({
          transposeA: false,
          transposeB: false
        })
      );
      
      expect(tensorSharing.registerSharedTensor).toHaveBeenCalledWith(
        'output',
        [2, 2],
        expect.any(Float32Array),
        'float32',
        'model3',
        ['model3']
      );
      
      expect(result).toBe('output');
    });
    
    test('should execute elementwise operation', async () => {
      const result = await webgpuTensorSharing.elementwise(
        'tensor1', 'model1',
        'output', 'model3',
        'relu'
      );
      
      expect(webgpuBackend.execute).toHaveBeenCalledWith(
        'elementwise',
        expect.objectContaining({
          input: expect.anything()
        }),
        expect.objectContaining({
          operation: 'relu'
        })
      );
      
      expect(tensorSharing.registerSharedTensor).toHaveBeenCalled();
      expect(result).toBe('output');
    });
    
    test('should execute softmax operation', async () => {
      const result = await webgpuTensorSharing.softmax(
        'tensor1', 'model1',
        'output', 'model3',
        -1
      );
      
      expect(webgpuBackend.execute).toHaveBeenCalledWith(
        'softmax',
        expect.objectContaining({
          input: expect.anything()
        }),
        expect.objectContaining({
          axis: -1
        })
      );
      
      expect(tensorSharing.registerSharedTensor).toHaveBeenCalled();
      expect(result).toBe('output');
    });
    
    test('should execute quantize operation', async () => {
      const result = await webgpuTensorSharing.quantize(
        'tensor1', 'model1',
        'output', 'model3'
      );
      
      expect(webgpuBackend.execute).toHaveBeenCalledWith(
        'quantize',
        expect.objectContaining({
          input: expect.anything()
        }),
        expect.any(Object)
      );
      
      expect(tensorSharing.registerSharedTensor).toHaveBeenCalledTimes(2); // Once for tensor, once for scale
      expect(result).toEqual({
        tensorName: 'output',
        scaleTensorName: 'output_scale'
      });
    });
    
    test('should execute dequantize operation', async () => {
      const result = await webgpuTensorSharing.dequantize(
        'tensor1', 'model1',
        'scale1', 'model1',
        'output', 'model3'
      );
      
      expect(webgpuBackend.execute).toHaveBeenCalledWith(
        'dequantize',
        expect.objectContaining({
          input: expect.anything(),
          scale: expect.anything()
        }),
        expect.any(Object)
      );
      
      expect(tensorSharing.registerSharedTensor).toHaveBeenCalled();
      expect(result).toBe('output');
    });
  });
  
  describe('tensor sharing and views', () => {
    beforeEach(async () => {
      await webgpuTensorSharing.initialize();
      
      // Mock WebGPU backend device
      webgpuBackend['device'] = {
        createBuffer: jest.fn().mockReturnValue({}),
        createCommandEncoder: jest.fn().mockReturnValue({
          copyBufferToBuffer: jest.fn(),
          finish: jest.fn().mockReturnValue({})
        }),
        queue: {
          submit: jest.fn()
        }
      };
      
      // Setup tensor creation
      (webgpuBackend.createTensor as jest.Mock).mockImplementation((data, shape, dataType) => {
        const buffer = {} as GPUBuffer;
        return Promise.resolve({
          buffer,
          shape,
          dataType,
          size: data.byteLength
        });
      });
    });
    
    test('should create tensor view', async () => {
      // First create a source tensor
      await webgpuTensorSharing.createGPUBufferFromSharedTensor('sourceT', 'modelA');
      
      // Create a view
      const viewTensorName = await webgpuTensorSharing.createTensorView(
        'sourceT', 'modelA',
        'viewT', 'modelB',
        2, 2
      );
      
      expect(tensorSharing.createTensorView).toHaveBeenCalledWith(
        'sourceT', 'viewT', 2, 2, 'modelA'
      );
      
      expect(tensorSharing.shareTensorBetweenModels).toHaveBeenCalledWith(
        'viewT', 'modelA', ['modelB']
      );
      
      expect(viewTensorName).toBe('viewT');
      
      // Verify GPU buffer was created for view
      expect(webgpuBackend['device'].createBuffer).toHaveBeenCalled();
      expect(webgpuBackend['device'].createCommandEncoder).toHaveBeenCalled();
    });
    
    test('should share tensor between models', async () => {
      // First create a source tensor
      await webgpuTensorSharing.createGPUBufferFromSharedTensor('sourceT', 'modelA');
      
      // Share with other models
      await webgpuTensorSharing.shareTensorBetweenModels(
        'sourceT', 'modelA', ['modelB', 'modelC']
      );
      
      expect(tensorSharing.shareTensorBetweenModels).toHaveBeenCalledWith(
        'sourceT', 'modelA', ['modelB', 'modelC']
      );
      
      // Verify caches were created for each model
      const cacheB = await webgpuTensorSharing.createGPUBufferFromSharedTensor('sourceT', 'modelB');
      const cacheC = await webgpuTensorSharing.createGPUBufferFromSharedTensor('sourceT', 'modelC');
      
      expect(cacheB).toBeDefined();
      expect(cacheC).toBeDefined();
      
      // Shouldn't have called these again since using cache
      expect(webgpuBackend.createTensor).toHaveBeenCalledTimes(1);
    });
  });
  
  describe('memory management', () => {
    beforeEach(async () => {
      await webgpuTensorSharing.initialize();
      
      // Setup tensor creation with different sizes
      (webgpuBackend.createTensor as jest.Mock).mockImplementation((data, shape, dataType) => {
        const buffer = {} as GPUBuffer;
        const size = shape[0] * shape[1] * 4; // Float32 = 4 bytes
        return Promise.resolve({
          buffer,
          shape,
          dataType,
          size
        });
      });
      
      // Mock tensor sharing memory functions
      (tensorSharing.getTensorMemoryUsage as jest.Mock).mockResolvedValue(1024 * 1024);
      (tensorSharing.getModelMemoryUsage as jest.Mock).mockResolvedValue({
        model1: 512 * 1024,
        model2: 512 * 1024
      });
    });
    
    test('should report correct GPU memory usage', async () => {
      // Create tensors of different sizes
      await webgpuTensorSharing.createGPUBufferFromSharedTensor('small', 'model1');
      
      // Mock a larger tensor
      (tensorSharing.getSharedTensor as jest.Mock).mockImplementation((name, model) => {
        if (name === 'large') {
          const tensor = new SharedTensor();
          (tensor.getData as jest.Mock).mockReturnValue(new Float32Array(1000));
          (tensor.getShape as jest.Mock).mockReturnValue([500, 2]);
          (tensor.getDataType as jest.Mock).mockReturnValue('float32');
          (tensor.getSize as jest.Mock).mockReturnValue(1000);
          return Promise.resolve(tensor);
        }
        
        const tensor = new SharedTensor();
        (tensor.getData as jest.Mock).mockReturnValue(new Float32Array([1, 2, 3, 4]));
        (tensor.getShape as jest.Mock).mockReturnValue([2, 2]);
        (tensor.getDataType as jest.Mock).mockReturnValue('float32');
        (tensor.getSize as jest.Mock).mockReturnValue(4);
        return Promise.resolve(tensor);
      });
      
      await webgpuTensorSharing.createGPUBufferFromSharedTensor('large', 'model2');
      
      const usage = webgpuTensorSharing.getGPUMemoryUsage();
      
      // Should be sum of small (2*2*4 = 16) and large (500*2*4 = 4000)
      expect(usage).toBeGreaterThan(0);
    });
    
    test('should clean up unused tensors', async () => {
      // Create a couple of tensors
      await webgpuTensorSharing.createGPUBufferFromSharedTensor('tensor1', 'model1');
      await webgpuTensorSharing.createGPUBufferFromSharedTensor('tensor2', 'model2');
      
      // Mock last used time to be old
      const map = (webgpuTensorSharing as any).tensorBufferCache;
      for (const entry of map.values()) {
        entry.lastUsed = Date.now() - 60000; // 1 minute ago
      }
      
      // Clean up with short max age
      webgpuTensorSharing.cleanupUnusedTensors(1000); // 1 second max age
      
      // getSharedTensor will be called to verify if tensor exists
      expect(tensorSharing.getSharedTensor).toHaveBeenCalled();
    });
    
    test('should optimize memory usage', async () => {
      // Create tensors
      await webgpuTensorSharing.createGPUBufferFromSharedTensor('tensor1', 'model1');
      
      // Set up a large tensor for optimization
      (tensorSharing.getSharedTensor as jest.Mock).mockImplementation((name, model) => {
        if (name === 'large') {
          const tensor = new SharedTensor();
          (tensor.getData as jest.Mock).mockReturnValue(new Float32Array(250000)); // 1MB
          (tensor.getShape as jest.Mock).mockReturnValue([500, 500]);
          (tensor.getDataType as jest.Mock).mockReturnValue('float32');
          (tensor.getSize as jest.Mock).mockReturnValue(250000);
          return Promise.resolve(tensor);
        }
        
        const tensor = new SharedTensor();
        (tensor.getData as jest.Mock).mockReturnValue(new Float32Array([1, 2, 3, 4]));
        (tensor.getShape as jest.Mock).mockReturnValue([2, 2]);
        (tensor.getDataType as jest.Mock).mockReturnValue('float32');
        (tensor.getSize as jest.Mock).mockReturnValue(4);
        return Promise.resolve(tensor);
      });
      
      await webgpuTensorSharing.createGPUBufferFromSharedTensor('large', 'model2');
      
      // Make it old
      const map = (webgpuTensorSharing as any).tensorBufferCache;
      for (const entry of map.values()) {
        entry.lastUsed = Date.now() - 10 * 60 * 1000; // 10 minutes ago
      }
      
      // Mock readBuffer for synchronization
      (webgpuBackend.readBuffer as jest.Mock).mockResolvedValue(new Float32Array(250000));
      
      // Optimize with memory priority
      (webgpuTensorSharing as any).options.priorityMode = 'memory';
      await webgpuTensorSharing.optimizeMemoryUsage();
      
      // Should have called readBuffer for synchronizing to CPU
      expect(webgpuBackend.readBuffer).toHaveBeenCalled();
    });
  });
  
  describe('custom shaders', () => {
    beforeEach(async () => {
      await webgpuTensorSharing.initialize();
      
      // Mock WebGPU backend device and shader creation
      webgpuBackend['device'] = {
        createShaderModule: jest.fn().mockReturnValue({}),
        createComputePipeline: jest.fn().mockReturnValue({
          getBindGroupLayout: jest.fn().mockReturnValue({})
        }),
        createBindGroup: jest.fn().mockReturnValue({}),
        createCommandEncoder: jest.fn().mockReturnValue({
          beginComputePass: jest.fn().mockReturnValue({
            setPipeline: jest.fn(),
            setBindGroup: jest.fn(),
            dispatchWorkgroups: jest.fn(),
            end: jest.fn()
          }),
          finish: jest.fn().mockReturnValue({})
        }),
        queue: {
          submit: jest.fn()
        },
        createBuffer: jest.fn().mockReturnValue({})
      };
      
      webgpuBackend['shaderCache'] = new Map();
      
      // Setup tensor creation
      (webgpuBackend.createTensor as jest.Mock).mockImplementation((data, shape, dataType) => {
        const buffer = {} as GPUBuffer;
        return Promise.resolve({
          buffer,
          shape,
          dataType,
          size: data.byteLength
        });
      });
      
      // Setup buffer read
      (webgpuBackend.readBuffer as jest.Mock).mockResolvedValue(new Float32Array([10, 20, 30, 40]));
    });
    
    test('should create custom shader', async () => {
      const shaderCode = `
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x;
          output[idx] = input[idx] * 2.0;
        }
      `;
      
      await webgpuTensorSharing.createCustomShader(
        'double_values',
        shaderCode,
        { input: 'float32' },
        { output: 'float32' }
      );
      
      // Should have created shader module
      expect(webgpuBackend['device'].createShaderModule).toHaveBeenCalledWith({
        label: 'custom_double_values_shader',
        code: shaderCode
      });
      
      // Should have stored in shader cache
      expect(webgpuBackend['shaderCache'].get('double_values')).toBeDefined();
    });
    
    test('should execute custom shader', async () => {
      // First create shader
      const shaderCode = `@compute @workgroup_size(1) fn main() {}`;
      await webgpuTensorSharing.createCustomShader('test_shader', shaderCode);
      
      // Make sure we have GPU buffers for input
      await webgpuTensorSharing.createGPUBufferFromSharedTensor('input1', 'modelA');
      
      // Execute custom shader
      const results = await webgpuTensorSharing.executeCustomShader(
        'test_shader',
        {
          input: { tensorName: 'input1', modelName: 'modelA' }
        },
        {
          output: { tensorName: 'output1', shape: [2, 2], dataType: 'float32' }
        },
        'modelB',
        [1, 1, 1],
        [1, 1, 1]
      );
      
      // Should have created compute pipeline
      expect(webgpuBackend['device'].createComputePipeline).toHaveBeenCalled();
      
      // Should have created bind group
      expect(webgpuBackend['device'].createBindGroup).toHaveBeenCalled();
      
      // Should have dispatched compute
      expect(webgpuBackend['device'].createCommandEncoder).toHaveBeenCalled();
      
      // Should have created output buffer
      expect(webgpuBackend['device'].createBuffer).toHaveBeenCalled();
      
      // Should have registered shared tensor
      expect(tensorSharing.registerSharedTensor).toHaveBeenCalledWith(
        'output1',
        [2, 2],
        expect.any(Float32Array),
        'float32',
        'modelB',
        ['modelB']
      );
      
      // Should return tensor names
      expect(results).toEqual({ output: 'output1' });
    });
  });
  
  describe('synchronization', () => {
    beforeEach(async () => {
      await webgpuTensorSharing.initialize();
      
      // Setup tensor creation
      (webgpuBackend.createTensor as jest.Mock).mockImplementation((data, shape, dataType) => {
        const buffer = {} as GPUBuffer;
        return Promise.resolve({
          buffer,
          shape,
          dataType,
          size: data.byteLength
        });
      });
      
      // Setup buffer read
      (webgpuBackend.readBuffer as jest.Mock).mockResolvedValue(new Float32Array([10, 20, 30, 40]));
    });
    
    test('should synchronize tensor from GPU to CPU', async () => {
      // First create GPU buffer
      await webgpuTensorSharing.createGPUBufferFromSharedTensor('tensor1', 'model1');
      
      // Set up mock for testing
      const mockTensor = new SharedTensor();
      (tensorSharing.getSharedTensor as jest.Mock).mockResolvedValue(mockTensor);
      
      // Synchronize GPU to CPU
      await webgpuTensorSharing.synchronizeTensor('tensor1', 'model1', 'gpu-to-cpu');
      
      // Should have read buffer
      expect(webgpuBackend.readBuffer).toHaveBeenCalled();
      
      // Should have updated CPU tensor data
      expect(mockTensor.setData).toHaveBeenCalledWith(expect.any(Float32Array));
    });
    
    test('should synchronize tensor from CPU to GPU', async () => {
      // First create GPU buffer
      await webgpuTensorSharing.createGPUBufferFromSharedTensor('tensor1', 'model1');
      
      // Synchronize CPU to GPU
      await webgpuTensorSharing.synchronizeTensor('tensor1', 'model1', 'cpu-to-gpu');
      
      // Should have created new tensor
      expect(webgpuBackend.createTensor).toHaveBeenCalledTimes(2);
    });
    
    test('should create GPU buffer when synchronizing non-existent GPU tensor', async () => {
      // Set up mock for testing
      const mockTensor = new SharedTensor();
      (mockTensor.getData as jest.Mock).mockReturnValue(new Float32Array([1, 2, 3, 4]));
      (mockTensor.getShape as jest.Mock).mockReturnValue([2, 2]);
      (mockTensor.getDataType as jest.Mock).mockReturnValue('float32');
      (tensorSharing.getSharedTensor as jest.Mock).mockResolvedValue(mockTensor);
      
      // Synchronize GPU to CPU for non-existent GPU tensor
      await webgpuTensorSharing.synchronizeTensor('new_tensor', 'model1', 'gpu-to-cpu');
      
      // Should create GPU buffer
      expect(webgpuBackend.createTensor).toHaveBeenCalled();
    });
  });
  
  describe('cleanup', () => {
    beforeEach(async () => {
      await webgpuTensorSharing.initialize();
      
      // Setup tensor creation
      (webgpuBackend.createTensor as jest.Mock).mockImplementation((data, shape, dataType) => {
        const buffer = {
          destroy: jest.fn()
        } as unknown as GPUBuffer;
        
        return Promise.resolve({
          buffer,
          shape,
          dataType,
          size: data.byteLength
        });
      });
    });
    
    test('should dispose resources', async () => {
      // Create some tensors first
      await webgpuTensorSharing.createGPUBufferFromSharedTensor('tensor1', 'model1');
      await webgpuTensorSharing.createGPUBufferFromSharedTensor('tensor2', 'model2');
      
      // Dispose
      webgpuTensorSharing.dispose();
      
      // Should call dispose on backend (if we created it)
      expect(webgpuBackend.dispose).toHaveBeenCalled();
      
      // Should no longer be initialized
      expect(webgpuTensorSharing.isInitialized()).toBe(false);
      
      // Should have cleared cache
      expect(webgpuTensorSharing.getGPUMemoryUsage()).toBe(0);
    });
  });
});