/**
 * Unit tests for the TensorSharingIntegration
 * Tests cover integration with storage manager, WebNN backend, and cross-model sharing
 */

import { TensorSharingIntegration } from './ipfs_accelerate_js_tensor_sharing_integration';
import { SharedTensor, TensorSharingManager } from './ipfs_accelerate_js/src/tensor/shared_tensor';
import { WebNNBackend } from './ipfs_accelerate_js_webnn_backend';
import StorageManager from './ipfs_accelerate_js_storage_manager';

// Mock classes for testing
jest.mock('./ipfs_accelerate_js/src/tensor/shared_tensor');
jest.mock('./ipfs_accelerate_js_webnn_backend');
jest.mock('./ipfs_accelerate_js_storage_manager');

describe('TensorSharingIntegration', () => {
  let integration: TensorSharingIntegration;
  let webnnBackend: jest.Mocked<WebNNBackend>;
  let mockTensor: jest.Mocked<SharedTensor>;
  
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Create mocked WebNN backend
    webnnBackend = new WebNNBackend() as jest.Mocked<WebNNBackend>;
    webnnBackend.initialize.mockResolvedValue(true);
    webnnBackend.createTensor.mockResolvedValue({
      tensor: { id: 'webnn_tensor_1' },
      shape: [1, 768]
    });
    webnnBackend.getTensorData.mockResolvedValue(new Float32Array(768));
    
    // Create integration with mocked dependencies
    integration = new TensorSharingIntegration({
      enablePersistence: true,
      enableLogging: true,
      maxMemoryMb: 1024,
      dbName: 'test-db'
    });
    
    // Mock the internal sharingManager
    const mockedManager = (TensorSharingManager as jest.Mock).mock.instances[0];
    mockedManager.registerSharedTensor.mockImplementation((name, shape, storageType, producerModel, consumerModels, dtype) => {
      mockTensor = {
        name,
        shape,
        storageType,
        producerModel,
        data: null,
        dtype,
        acquire: jest.fn().mockReturnValue(true),
        release: jest.fn().mockReturnValue(true),
        pin: jest.fn(),
        unpin: jest.fn(),
        getMemoryUsage: jest.fn().mockReturnValue(shape[0] * shape[1] * 4),
        canBeFreed: jest.fn().mockReturnValue(false),
        referenceCount: 0
      } as unknown as jest.Mocked<SharedTensor>;
      return mockTensor;
    });
    
    mockedManager.getSharedTensor.mockImplementation((name, modelName) => {
      if (name === 'existing_tensor') {
        return mockTensor;
      }
      return null;
    });
    
    mockedManager.shareTensorBetweenModels.mockReturnValue(true);
    mockedManager.createTensorView.mockReturnValue({
      name: 'test_view',
      parent: mockTensor,
      offset: [0, 0],
      size: [1, 384],
      acquire: jest.fn().mockReturnValue(true),
      release: jest.fn().mockReturnValue(true)
    });
    mockedManager.optimizeMemoryUsage.mockReturnValue({
      initial_memory_bytes: 10000,
      current_memory_bytes: 8000,
      freed_memory_bytes: 2000,
      freed_tensors_count: 1,
      memory_reduction_percent: 20
    });
    mockedManager.getStats.mockReturnValue({
      total_tensors: 3,
      total_models: 2,
      memory_usage_mb: 5.2,
      hit_rate: 0.85
    });
  });
  
  describe('Initialization', () => {
    it('should initialize successfully', async () => {
      const result = await integration.initialize(webnnBackend);
      expect(result).toBe(true);
      expect(StorageManager.prototype.initialize).toHaveBeenCalled();
    });
    
    it('should handle storage initialization failure', async () => {
      (StorageManager.prototype.initialize as jest.Mock).mockResolvedValueOnce(false);
      const result = await integration.initialize(webnnBackend);
      expect(result).toBe(false);
    });
    
    it('should handle WebNN initialization failure', async () => {
      webnnBackend.initialize.mockResolvedValueOnce(false);
      const result = await integration.initialize(webnnBackend);
      expect(result).toBe(false);
    });
  });
  
  describe('Tensor Registration', () => {
    beforeEach(async () => {
      await integration.initialize(webnnBackend);
    });
    
    it('should register a shared tensor successfully', async () => {
      const data = new Float32Array(768);
      const tensor = await integration.registerSharedTensor(
        'test_tensor',
        [1, 768],
        data,
        'cpu',
        'test-model',
        null
      );
      
      expect(tensor).toBeDefined();
      expect(tensor.name).toBe('test_tensor');
      expect(tensor.shape).toEqual([1, 768]);
      expect(tensor.storageType).toBe('cpu');
      expect(tensor.producerModel).toBe('test-model');
      
      // Should call persistSharedTensor if enablePersistence is true
      expect(StorageManager.prototype.storeTensor).toHaveBeenCalled();
      expect(StorageManager.prototype.storeModelMetadata).toHaveBeenCalled();
    });
    
    it('should handle tensor registration error', async () => {
      const data = new Float32Array(768);
      
      // Make the TensorSharingManager throw an error
      (TensorSharingManager.prototype.registerSharedTensor as jest.Mock).mockImplementationOnce(() => {
        throw new Error('Registration error');
      });
      
      await expect(
        integration.registerSharedTensor(
          'test_tensor',
          [1, 768],
          data,
          'cpu',
          'test-model',
          null
        )
      ).rejects.toThrow('Registration error');
    });
  });
  
  describe('Tensor Retrieval', () => {
    beforeEach(async () => {
      await integration.initialize(webnnBackend);
      
      // Setup a mock tensor for testing
      mockTensor = {
        name: 'existing_tensor',
        shape: [1, 768],
        storageType: 'cpu',
        producerModel: 'bert',
        data: new Float32Array(768),
        dtype: 'float32',
        acquire: jest.fn().mockReturnValue(true),
        release: jest.fn().mockReturnValue(true),
        pin: jest.fn(),
        unpin: jest.fn(),
        getMemoryUsage: jest.fn().mockReturnValue(768 * 4),
        canBeFreed: jest.fn().mockReturnValue(false),
        referenceCount: 1
      } as unknown as jest.Mocked<SharedTensor>;
      
      (TensorSharingManager.prototype.getSharedTensor as jest.Mock).mockImplementation((name, modelName) => {
        if (name === 'existing_tensor') {
          return mockTensor;
        }
        return null;
      });
    });
    
    it('should get a tensor from memory', async () => {
      const tensor = await integration.getSharedTensor('existing_tensor', 'model1');
      expect(tensor).toBe(mockTensor);
      expect(mockTensor.acquire).toHaveBeenCalledWith('model1');
    });
    
    it('should load a tensor from persistent storage if not in memory', async () => {
      // Setup mockTensor to be found in storage
      const persistentInfo = { storageId: 'storage_id_1', lastSynced: Date.now() };
      (integration as any).persistentTensors.set('persistent_tensor', persistentInfo);
      
      // Mock storage manager to return model metadata
      (StorageManager.prototype.getModelMetadata as jest.Mock).mockResolvedValueOnce({
        id: 'storage_id_1',
        tensorIds: ['tensor_id_1'],
        metadata: {
          tensorName: 'persistent_tensor',
          shape: [1, 768],
          storageType: 'cpu',
          producerModel: 'bert'
        }
      });
      
      // Mock storage manager to return tensor data
      (StorageManager.prototype.getTensorData as jest.Mock).mockResolvedValueOnce(new Float32Array(768));
      (StorageManager.prototype.getTensor as jest.Mock).mockResolvedValueOnce({
        shape: [1, 768],
        dataType: 'float32'
      });
      
      // Mock the registerSharedTensor method to return a new tensor
      const storageTensor = {
        name: 'persistent_tensor',
        shape: [1, 768],
        storageType: 'cpu',
        producerModel: 'bert',
        data: null,
        acquire: jest.fn().mockReturnValue(true)
      } as unknown as jest.Mocked<SharedTensor>;
      
      (TensorSharingManager.prototype.registerSharedTensor as jest.Mock).mockReturnValueOnce(storageTensor);
      
      // Get the tensor that should be loaded from storage
      const tensor = await integration.getSharedTensor('persistent_tensor', 'model1');
      
      expect(tensor).toBeDefined();
      expect(StorageManager.prototype.getModelMetadata).toHaveBeenCalledWith('storage_id_1');
      expect(StorageManager.prototype.getTensorData).toHaveBeenCalled();
      expect(tensor?.acquire).toHaveBeenCalledWith('model1');
    });
    
    it('should return null if tensor not found in memory or storage', async () => {
      const tensor = await integration.getSharedTensor('non_existent_tensor', 'model1');
      expect(tensor).toBeNull();
    });
  });
  
  describe('Tensor Sharing', () => {
    beforeEach(async () => {
      await integration.initialize(webnnBackend);
    });
    
    it('should share tensor between models', async () => {
      const result = await integration.shareTensorBetweenModels(
        'existing_tensor',
        'bert',
        ['t5', 'gpt2']
      );
      
      expect(result.status).toBe('success');
      expect(TensorSharingManager.prototype.shareTensorBetweenModels).toHaveBeenCalledWith(
        'existing_tensor',
        'bert',
        ['t5', 'gpt2']
      );
    });
    
    it('should handle tensor not found', async () => {
      (TensorSharingManager.prototype.getSharedTensor as jest.Mock).mockReturnValueOnce(null);
      
      const result = await integration.shareTensorBetweenModels(
        'non_existent_tensor',
        'bert',
        ['t5', 'gpt2']
      );
      
      expect(result.status).toBe('not_found');
    });
    
    it('should handle sharing error', async () => {
      (TensorSharingManager.prototype.shareTensorBetweenModels as jest.Mock).mockReturnValueOnce(false);
      
      const result = await integration.shareTensorBetweenModels(
        'existing_tensor',
        'bert',
        ['t5', 'gpt2']
      );
      
      expect(result.status).toBe('error');
    });
  });
  
  describe('Tensor Views', () => {
    beforeEach(async () => {
      await integration.initialize(webnnBackend);
    });
    
    it('should create a tensor view', async () => {
      const view = await integration.createTensorView(
        'existing_tensor',
        'half_view',
        [0, 0],
        [1, 384],
        'distilbert'
      );
      
      expect(view).toBeDefined();
      expect(TensorSharingManager.prototype.createTensorView).toHaveBeenCalledWith(
        'existing_tensor',
        'half_view',
        [0, 0],
        [1, 384],
        'distilbert'
      );
    });
    
    it('should handle parent tensor not found', async () => {
      (TensorSharingManager.prototype.getSharedTensor as jest.Mock).mockReturnValueOnce(null);
      
      const view = await integration.createTensorView(
        'non_existent_tensor',
        'half_view',
        [0, 0],
        [1, 384],
        'distilbert'
      );
      
      expect(view).toBeNull();
    });
  });
  
  describe('WebNN Integration', () => {
    beforeEach(async () => {
      await integration.initialize(webnnBackend);
    });
    
    it('should create WebNN tensors from shared tensors', async () => {
      // Mock getSharedTensor to return a tensor with data
      mockTensor = {
        name: 'existing_tensor',
        shape: [1, 768],
        storageType: 'cpu',
        producerModel: 'bert',
        data: new Float32Array(768),
        dtype: 'float32',
        acquire: jest.fn().mockReturnValue(true),
        release: jest.fn().mockReturnValue(true)
      } as unknown as jest.Mocked<SharedTensor>;
      
      (TensorSharingManager.prototype.getSharedTensor as jest.Mock).mockReturnValueOnce(mockTensor);
      
      const webnnTensors = await integration.createWebNNTensorsFromShared(
        ['existing_tensor'],
        'llama-7b'
      );
      
      expect(webnnTensors).toBeDefined();
      expect(webnnTensors?.size).toBe(1);
      expect(webnnBackend.createTensor).toHaveBeenCalled();
      expect(mockTensor.acquire).toHaveBeenCalledWith('llama-7b');
    });
    
    it('should handle missing WebNN backend', async () => {
      // Re-create integration without WebNN backend
      integration = new TensorSharingIntegration({
        enablePersistence: true,
        enableLogging: true
      });
      await integration.initialize(); // No WebNN backend
      
      const webnnTensors = await integration.createWebNNTensorsFromShared(
        ['existing_tensor'],
        'llama-7b'
      );
      
      expect(webnnTensors).toBeNull();
    });
    
    it('should save shared tensors as a WebNN model', async () => {
      // Mock WebNN storage integration
      const mockStoreModel = jest.fn().mockResolvedValue(true);
      (integration as any).webnnStorageIntegration = {
        storeModel: mockStoreModel
      };
      
      // Mock getSharedTensor to return tensors with data
      const mockTensors = [
        {
          name: 'text_embedding',
          shape: [1, 768],
          storageType: 'cpu',
          producerModel: 'bert',
          data: new Float32Array(768),
          dtype: 'float32'
        },
        {
          name: 'vision_embedding',
          shape: [1, 1024],
          storageType: 'webgpu',
          producerModel: 'vit',
          data: new Float32Array(1024),
          dtype: 'float32'
        }
      ];
      
      (TensorSharingManager.prototype.getSharedTensor as jest.Mock).mockImplementation((name) => {
        if (name === 'text_embedding') return mockTensors[0];
        if (name === 'vision_embedding') return mockTensors[1];
        return null;
      });
      
      const result = await integration.saveAsWebNNModel(
        'shared_model_1',
        'Shared Tensors Model',
        ['text_embedding', 'vision_embedding']
      );
      
      expect(result).toBe(true);
      expect(mockStoreModel).toHaveBeenCalledWith(
        'shared_model_1',
        'Shared Tensors Model',
        expect.any(Map),
        expect.objectContaining({
          type: 'shared_tensors',
          tensorNames: ['text_embedding', 'vision_embedding']
        })
      );
    });
    
    it('should load shared tensors from a WebNN model', async () => {
      // Mock WebNN storage integration
      const mockLoadModel = jest.fn().mockResolvedValue(new Map([
        ['text_embedding', { tensor: 'webnn_tensor_1', shape: [1, 768] }],
        ['vision_embedding', { tensor: 'webnn_tensor_2', shape: [1, 1024] }]
      ]));
      
      const mockListModels = jest.fn().mockResolvedValue([
        { id: 'model_id_1', name: 'Shared Tensors Model' }
      ]);
      
      (integration as any).webnnStorageIntegration = {
        loadModel: mockLoadModel,
        listModels: mockListModels
      };
      
      // Mock registerSharedTensor implementation
      const registerSharedTensorSpy = jest.spyOn(integration, 'registerSharedTensor');
      registerSharedTensorSpy.mockImplementation(async () => {
        return { name: 'new_tensor' } as SharedTensor;
      });
      
      const tensorNames = await integration.loadFromWebNNModel(
        'model_id_1',
        'new-producer'
      );
      
      expect(tensorNames).toHaveLength(2);
      expect(webnnBackend.getTensorData).toHaveBeenCalledTimes(2);
      expect(registerSharedTensorSpy).toHaveBeenCalledTimes(2);
    });
  });
  
  describe('Memory Management', () => {
    beforeEach(async () => {
      await integration.initialize(webnnBackend);
    });
    
    it('should release model tensors', async () => {
      const releasedCount = await integration.releaseModelTensors('model1');
      expect(TensorSharingManager.prototype.releaseModelTensors).toHaveBeenCalledWith('model1');
    });
    
    it('should optimize memory usage', async () => {
      const result = await integration.optimizeMemoryUsage();
      expect(TensorSharingManager.prototype.optimizeMemoryUsage).toHaveBeenCalled();
      expect(result.memory_reduction_percent).toBe(20);
    });
    
    it('should get tensor memory usage', () => {
      integration.getTensorMemoryUsage();
      expect(TensorSharingManager.prototype.getTensorMemoryUsage).toHaveBeenCalled();
    });
    
    it('should get model memory usage', () => {
      integration.getModelMemoryUsage();
      expect(TensorSharingManager.prototype.getModelMemoryUsage).toHaveBeenCalled();
    });
    
    it('should get optimization recommendations', () => {
      integration.getOptimizationRecommendations();
      expect(TensorSharingManager.prototype.getOptimizationRecommendations).toHaveBeenCalled();
    });
    
    it('should get statistics', () => {
      const stats = integration.getStats();
      expect(TensorSharingManager.prototype.getStats).toHaveBeenCalled();
      expect(stats.total_tensors).toBe(3);
      expect(stats.total_models).toBe(2);
    });
  });
  
  describe('Persistent Storage', () => {
    beforeEach(async () => {
      await integration.initialize(webnnBackend);
    });
    
    it('should synchronize with persistent storage when enabled', async () => {
      // Mock tensor memory usage
      (TensorSharingManager.prototype.getTensorMemoryUsage as jest.Mock).mockReturnValue({
        'tensor1': { memory_mb: 3.0 },
        'tensor2': { memory_mb: 2.2 }
      });
      
      // Mock getSharedTensor to return tensors with data
      const mockTensors = [
        {
          name: 'tensor1',
          shape: [1, 768],
          storageType: 'cpu',
          producerModel: 'bert',
          data: new Float32Array(768),
          dtype: 'float32',
          referenceCount: 2
        },
        {
          name: 'tensor2',
          shape: [1, 512],
          storageType: 'webgpu',
          producerModel: 'vit',
          data: new Float32Array(512),
          dtype: 'float32',
          referenceCount: 1
        }
      ];
      
      (TensorSharingManager.prototype.getSharedTensor as jest.Mock).mockImplementation((name) => {
        if (name === 'tensor1') return mockTensors[0];
        if (name === 'tensor2') return mockTensors[1];
        return null;
      });
      
      // Mock the persistSharedTensor private method
      const persistSpy = jest.spyOn(integration as any, 'persistSharedTensor');
      persistSpy.mockResolvedValue(true);
      
      const result = await integration.synchronizePersistentStorage();
      
      expect(result.success).toBe(true);
      expect(result.updatedCount).toBe(2);
      expect(persistSpy).toHaveBeenCalledTimes(2);
    });
    
    it('should handle persistence disabled', async () => {
      // Create integration with persistence disabled
      integration = new TensorSharingIntegration({
        enablePersistence: false,
        enableLogging: true
      });
      await integration.initialize(webnnBackend);
      
      const result = await integration.synchronizePersistentStorage();
      
      expect(result.success).toBe(false);
      expect(result.message).toContain('Persistence is not enabled');
    });
    
    it('should handle synchronization errors', async () => {
      // Mock tensor memory usage
      (TensorSharingManager.prototype.getTensorMemoryUsage as jest.Mock).mockReturnValue({
        'tensor1': { memory_mb: 3.0 }
      });
      
      // Mock getSharedTensor to return a tensor with data
      const mockTensor = {
        name: 'tensor1',
        shape: [1, 768],
        storageType: 'cpu',
        producerModel: 'bert',
        data: new Float32Array(768),
        dtype: 'float32',
        referenceCount: 2
      };
      
      (TensorSharingManager.prototype.getSharedTensor as jest.Mock).mockReturnValue(mockTensor);
      
      // Mock the persistSharedTensor private method to throw an error
      const persistSpy = jest.spyOn(integration as any, 'persistSharedTensor');
      persistSpy.mockRejectedValue(new Error('Storage error'));
      
      const result = await integration.synchronizePersistentStorage();
      
      expect(result.success).toBe(false);
      expect(result.message).toContain('Error: Storage error');
    });
  });
});