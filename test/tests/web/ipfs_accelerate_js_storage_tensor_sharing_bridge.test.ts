/**
 * Storage Tensor Sharing Bridge Tests
 */

import { StorageTensorSharingBridge, TensorSharingType } from './ipfs_accelerate_js_storage_tensor_sharing_bridge';
import StorageManager, { TensorStorage } from './ipfs_accelerate_js_storage_manager';

// Mock StorageManager
jest.mock('./ipfs_accelerate_js_storage_manager', () => {
  const originalModule = jest.requireActual('./ipfs_accelerate_js_storage_manager');
  
  // Create a mock class that extends the original
  class MockStorageManager {
    initialize = jest.fn().mockResolvedValue(true);
    storeTensor = jest.fn().mockResolvedValue(true);
    getTensor = jest.fn();
    getTensorData = jest.fn();
    deleteTensor = jest.fn().mockResolvedValue(true);
    getAllFromObjectStore = jest.fn();
  }
  
  return {
    __esModule: true,
    ...originalModule,
    default: MockStorageManager,
    StorageManager: MockStorageManager
  };
});

describe('StorageTensorSharingBridge', () => {
  let bridge: StorageTensorSharingBridge;
  let mockStorageManager: jest.Mocked<StorageManager>;
  
  beforeEach(() => {
    // Clear all mocks
    jest.clearAllMocks();
    
    // Create bridge instance with logging enabled for better debugging
    bridge = new StorageTensorSharingBridge({
      enableLogging: true,
      dbName: 'test-sharing-bridge'
    });
    
    // Get reference to the mocked storage manager
    mockStorageManager = (bridge as any).storageManager;
    
    // Setup getAllFromObjectStore mock to return empty array by default
    mockStorageManager.getAllFromObjectStore.mockResolvedValue([]);
  });
  
  describe('initialize', () => {
    it('should initialize successfully', async () => {
      const result = await bridge.initialize();
      
      expect(result).toBe(true);
      expect(mockStorageManager.initialize).toHaveBeenCalled();
    });
    
    it('should load existing shared tensors metadata on initialization', async () => {
      // Setup mock shared tensor data
      const mockTensors: TensorStorage[] = [
        {
          id: 'tensor1',
          shape: [1, 768],
          dataType: 'float32',
          data: new ArrayBuffer(0),
          metadata: {
            sharing: {
              sharingType: 'text_embedding' as TensorSharingType,
              originalModel: 'model1',
              sharedBy: ['model1', 'model2'],
              refCount: 2
            }
          },
          createdAt: Date.now(),
          lastAccessed: Date.now(),
          byteSize: 3072
        },
        {
          id: 'tensor2',
          shape: [1, 1024],
          dataType: 'float32',
          data: new ArrayBuffer(0),
          metadata: {
            sharing: {
              sharingType: 'vision_embedding' as TensorSharingType,
              originalModel: 'model3',
              sharedBy: ['model3'],
              refCount: 1
            }
          },
          createdAt: Date.now(),
          lastAccessed: Date.now(),
          byteSize: 4096
        }
      ];
      
      mockStorageManager.getAllFromObjectStore.mockResolvedValue(mockTensors);
      
      const result = await bridge.initialize();
      
      expect(result).toBe(true);
      expect(mockStorageManager.getAllFromObjectStore).toHaveBeenCalledWith('tensors');
      
      // Verify that the tensors were loaded into memory
      const sharingStats = await bridge.getSharingStats();
      expect(sharingStats.totalSharedTensors).toBe(2);
      expect(sharingStats.totalRefCount).toBe(3); // 2 + 1
    });
  });
  
  describe('registerSharedTensor', () => {
    beforeEach(async () => {
      await bridge.initialize();
    });
    
    it('should register a new shared tensor', async () => {
      const tensorId = 'test-tensor-1';
      const modelId = 'model1';
      const sharingType = 'text_embedding';
      const data = new Float32Array([1, 2, 3, 4]);
      const shape = [1, 4];
      
      mockStorageManager.getTensor.mockResolvedValue(null); // Tensor doesn't exist yet
      
      const result = await bridge.registerSharedTensor(
        tensorId,
        modelId,
        sharingType as TensorSharingType,
        data,
        shape,
        'float32'
      );
      
      expect(result).toBe(true);
      expect(mockStorageManager.storeTensor).toHaveBeenCalledWith(
        tensorId,
        data,
        shape,
        'float32',
        expect.objectContaining({
          sharing: expect.objectContaining({
            sharingType,
            originalModel: modelId,
            sharedBy: [modelId],
            refCount: 1
          })
        })
      );
      
      // Check that in-memory metadata was updated
      const sharingStats = await bridge.getSharingStats();
      expect(sharingStats.totalSharedTensors).toBe(1);
      expect(sharingStats.totalRefCount).toBe(1);
      expect(sharingStats.sharingByType.text_embedding).toBe(1);
    });
    
    it('should update sharing metadata for existing tensor', async () => {
      const tensorId = 'test-tensor-1';
      const modelId = 'model2';
      const sharingType = 'text_embedding';
      const data = new Float32Array([1, 2, 3, 4]);
      const shape = [1, 4];
      
      // Tensor already exists
      mockStorageManager.getTensor.mockResolvedValue({
        id: tensorId,
        shape,
        dataType: 'float32',
        data: new ArrayBuffer(0),
        metadata: {
          sharing: {
            sharingType,
            originalModel: 'model1',
            sharedBy: ['model1'],
            refCount: 1
          }
        },
        createdAt: Date.now(),
        lastAccessed: Date.now(),
        byteSize: 16
      });
      
      mockStorageManager.getTensorData.mockResolvedValue(data);
      
      const result = await bridge.registerSharedTensor(
        tensorId,
        modelId,
        sharingType as TensorSharingType,
        data,
        shape,
        'float32'
      );
      
      expect(result).toBe(true);
      expect(mockStorageManager.storeTensor).toHaveBeenCalledWith(
        tensorId,
        data,
        shape,
        'float32',
        expect.objectContaining({
          sharing: expect.objectContaining({
            sharingType,
            originalModel: 'model1',
            sharedBy: expect.arrayContaining(['model1', 'model2']),
            refCount: 2
          })
        })
      );
      
      // Check that in-memory metadata was updated
      const sharingStats = await bridge.getSharingStats();
      expect(sharingStats.totalSharedTensors).toBe(1);
      expect(sharingStats.totalRefCount).toBe(2);
    });
  });
  
  describe('shareTensor', () => {
    beforeEach(async () => {
      await bridge.initialize();
      
      // Register a shared tensor for testing
      const tensorId = 'test-tensor-1';
      const modelId = 'model1';
      const sharingType = 'text_embedding';
      const data = new Float32Array([1, 2, 3, 4]);
      const shape = [1, 4];
      
      mockStorageManager.getTensor.mockResolvedValue(null);
      
      await bridge.registerSharedTensor(
        tensorId,
        modelId,
        sharingType as TensorSharingType,
        data,
        shape,
        'float32'
      );
      
      // Update mock to return the tensor when requested
      mockStorageManager.getTensor.mockResolvedValue({
        id: tensorId,
        shape,
        dataType: 'float32',
        data: new ArrayBuffer(0),
        metadata: {
          sharing: {
            sharingType,
            originalModel: modelId,
            sharedBy: [modelId],
            refCount: 1
          }
        },
        createdAt: Date.now(),
        lastAccessed: Date.now(),
        byteSize: 16
      });
      
      mockStorageManager.getTensorData.mockResolvedValue(data);
    });
    
    it('should share a tensor with another model', async () => {
      const tensorId = 'test-tensor-1';
      const sourceModelId = 'model1';
      const targetModelId = 'model2';
      
      const result = await bridge.shareTensor(tensorId, sourceModelId, targetModelId);
      
      expect(result).toBe(true);
      expect(mockStorageManager.storeTensor).toHaveBeenCalledWith(
        tensorId,
        expect.any(Float32Array),
        [1, 4],
        'float32',
        expect.objectContaining({
          sharing: expect.objectContaining({
            sharedBy: expect.arrayContaining(['model1', 'model2']),
            refCount: 2
          })
        })
      );
      
      // Check that in-memory metadata was updated
      const sharingStats = await bridge.getSharingStats();
      expect(sharingStats.totalSharedTensors).toBe(1);
      expect(sharingStats.totalRefCount).toBe(2);
      
      // Should be able to get the shared tensor for the new model
      const modelTensors = await bridge.getModelSharedTensors(targetModelId);
      expect(modelTensors.length).toBe(1);
      expect(modelTensors[0].tensorId).toBe(tensorId);
    });
    
    it('should not share tensor again if already shared', async () => {
      const tensorId = 'test-tensor-1';
      const sourceModelId = 'model1';
      const targetModelId = 'model2';
      
      // First share
      await bridge.shareTensor(tensorId, sourceModelId, targetModelId);
      
      // Reset mock to see if it's called again
      mockStorageManager.storeTensor.mockClear();
      
      // Try to share again
      const result = await bridge.shareTensor(tensorId, sourceModelId, targetModelId);
      
      expect(result).toBe(true);
      expect(mockStorageManager.storeTensor).not.toHaveBeenCalled();
      
      // Stats should remain the same
      const sharingStats = await bridge.getSharingStats();
      expect(sharingStats.totalRefCount).toBe(2);
    });
  });
  
  describe('getSharedTensor', () => {
    const tensorId = 'test-tensor-1';
    const modelId = 'model1';
    const sharingType = 'text_embedding';
    const data = new Float32Array([1, 2, 3, 4]);
    const shape = [1, 4];
    
    beforeEach(async () => {
      await bridge.initialize();
      
      // Register a shared tensor for testing
      mockStorageManager.getTensor.mockResolvedValue(null);
      
      await bridge.registerSharedTensor(
        tensorId,
        modelId,
        sharingType as TensorSharingType,
        data,
        shape,
        'float32'
      );
      
      // Update mock to return the tensor when requested
      mockStorageManager.getTensor.mockResolvedValue({
        id: tensorId,
        shape,
        dataType: 'float32',
        data: new ArrayBuffer(16),
        metadata: {
          sharing: {
            sharingType,
            originalModel: modelId,
            sharedBy: [modelId],
            refCount: 1
          }
        },
        createdAt: Date.now(),
        lastAccessed: Date.now(),
        byteSize: 16
      });
      
      mockStorageManager.getTensorData.mockResolvedValue(data);
    });
    
    it('should get a shared tensor', async () => {
      const result = await bridge.getSharedTensor(tensorId, modelId);
      
      expect(result).not.toBeNull();
      expect(result!.data).toBe(data);
      expect(result!.shape).toEqual(shape);
      expect(result!.metadata).toBeDefined();
    });
    
    it('should return null if tensor is not shared with the model', async () => {
      const result = await bridge.getSharedTensor(tensorId, 'unshared-model');
      
      expect(result).toBeNull();
    });
    
    it('should use the cache for subsequent requests', async () => {
      // First request
      await bridge.getSharedTensor(tensorId, modelId);
      
      // Clear mocks to see if they're called again
      mockStorageManager.getTensorData.mockClear();
      
      // Second request should use cache
      await bridge.getSharedTensor(tensorId, modelId);
      
      expect(mockStorageManager.getTensorData).not.toHaveBeenCalled();
    });
  });
  
  describe('findCompatibleTensors', () => {
    beforeEach(async () => {
      await bridge.initialize();
      
      // Register a few shared tensors of different types
      const data = new Float32Array([1, 2, 3, 4]);
      
      mockStorageManager.getTensor.mockImplementation((tensorId) => {
        if (tensorId === 'text-tensor-1' || tensorId === 'text-tensor-2') {
          return Promise.resolve({
            id: tensorId,
            shape: [1, 4],
            dataType: 'float32',
            data: new ArrayBuffer(16),
            metadata: {
              sharing: {
                sharingType: 'text_embedding',
                originalModel: 'model1',
                sharedBy: ['model1'],
                refCount: 1
              }
            },
            createdAt: Date.now(),
            lastAccessed: Date.now(),
            byteSize: 16
          });
        } else if (tensorId === 'vision-tensor-1') {
          return Promise.resolve({
            id: tensorId,
            shape: [1, 4],
            dataType: 'float32',
            data: new ArrayBuffer(16),
            metadata: {
              sharing: {
                sharingType: 'vision_embedding',
                originalModel: 'model2',
                sharedBy: ['model2'],
                refCount: 1
              }
            },
            createdAt: Date.now(),
            lastAccessed: Date.now(),
            byteSize: 16
          });
        }
        return Promise.resolve(null);
      });
      
      // Register tensors
      await bridge.registerSharedTensor(
        'text-tensor-1',
        'model1',
        'text_embedding',
        data,
        [1, 4],
        'float32'
      );
      
      await bridge.registerSharedTensor(
        'text-tensor-2',
        'model1',
        'text_embedding',
        data,
        [1, 4],
        'float32'
      );
      
      await bridge.registerSharedTensor(
        'vision-tensor-1',
        'model2',
        'vision_embedding',
        data,
        [1, 4],
        'float32'
      );
    });
    
    it('should find compatible tensors by sharing type', async () => {
      const results = await bridge.findCompatibleTensors('model3', 'text_embedding');
      
      expect(results.length).toBe(2);
      expect(results.map(r => r.tensorId)).toEqual(expect.arrayContaining(['text-tensor-1', 'text-tensor-2']));
    });
    
    it('should filter by model type if provided', async () => {
      const results = await bridge.findCompatibleTensors('model3', 'text_embedding', 'bert');
      
      expect(results.length).toBe(2);
      
      // Try with incompatible model type
      const emptyResults = await bridge.findCompatibleTensors('model3', 'vision_embedding', 'bert');
      
      expect(emptyResults.length).toBe(0);
    });
  });
  
  describe('releaseTensor', () => {
    const tensorId = 'test-tensor-1';
    const modelId1 = 'model1';
    const modelId2 = 'model2';
    const sharingType = 'text_embedding';
    const data = new Float32Array([1, 2, 3, 4]);
    const shape = [1, 4];
    
    beforeEach(async () => {
      await bridge.initialize();
      
      // Register and share a tensor
      mockStorageManager.getTensor.mockResolvedValue(null);
      
      await bridge.registerSharedTensor(
        tensorId,
        modelId1,
        sharingType as TensorSharingType,
        data,
        shape,
        'float32'
      );
      
      // Update mock to return the tensor when requested
      mockStorageManager.getTensor.mockResolvedValue({
        id: tensorId,
        shape,
        dataType: 'float32',
        data: new ArrayBuffer(16),
        metadata: {
          sharing: {
            sharingType,
            originalModel: modelId1,
            sharedBy: [modelId1],
            refCount: 1
          }
        },
        createdAt: Date.now(),
        lastAccessed: Date.now(),
        byteSize: 16
      });
      
      mockStorageManager.getTensorData.mockResolvedValue(data);
      
      // Share with second model
      await bridge.shareTensor(tensorId, modelId1, modelId2);
      
      // Update mock to reflect the sharing
      mockStorageManager.getTensor.mockResolvedValue({
        id: tensorId,
        shape,
        dataType: 'float32',
        data: new ArrayBuffer(16),
        metadata: {
          sharing: {
            sharingType,
            originalModel: modelId1,
            sharedBy: [modelId1, modelId2],
            refCount: 2
          }
        },
        createdAt: Date.now(),
        lastAccessed: Date.now(),
        byteSize: 16
      });
    });
    
    it('should reduce reference count when a model releases a tensor', async () => {
      const result = await bridge.releaseTensor(tensorId, modelId2);
      
      expect(result).toBe(true);
      expect(mockStorageManager.storeTensor).toHaveBeenCalledWith(
        tensorId,
        data,
        shape,
        'float32',
        expect.objectContaining({
          sharing: expect.objectContaining({
            sharedBy: [modelId1],
            refCount: 1
          })
        })
      );
      
      // Stats should show reduced ref count
      const sharingStats = await bridge.getSharingStats();
      expect(sharingStats.totalRefCount).toBe(1);
    });
    
    it('should remove tensor when last reference is released', async () => {
      // Release from second model
      await bridge.releaseTensor(tensorId, modelId2);
      
      mockStorageManager.storeTensor.mockClear();
      mockStorageManager.deleteTensor.mockClear();
      
      // Release from first (original) model
      const result = await bridge.releaseTensor(tensorId, modelId1);
      
      expect(result).toBe(true);
      expect(mockStorageManager.deleteTensor).toHaveBeenCalledWith(tensorId);
      expect(mockStorageManager.storeTensor).not.toHaveBeenCalled();
      
      // Stats should show no tensors
      const sharingStats = await bridge.getSharingStats();
      expect(sharingStats.totalSharedTensors).toBe(0);
      expect(sharingStats.totalRefCount).toBe(0);
    });
  });
  
  describe('getSharingStats', () => {
    beforeEach(async () => {
      await bridge.initialize();
      
      // Register a few shared tensors of different types
      const data = new Float32Array([1, 2, 3, 4]);
      
      mockStorageManager.getTensor.mockImplementation((tensorId) => {
        if (tensorId === 'text-tensor-1') {
          return Promise.resolve({
            id: tensorId,
            shape: [1, 4],
            dataType: 'float32',
            data: new ArrayBuffer(16),
            metadata: {
              sharing: {
                sharingType: 'text_embedding',
                originalModel: 'model1',
                sharedBy: ['model1', 'model2'],
                refCount: 2
              }
            },
            createdAt: Date.now(),
            lastAccessed: Date.now(),
            byteSize: 16
          });
        } else if (tensorId === 'vision-tensor-1') {
          return Promise.resolve({
            id: tensorId,
            shape: [1, 4],
            dataType: 'float32',
            data: new ArrayBuffer(16),
            metadata: {
              sharing: {
                sharingType: 'vision_embedding',
                originalModel: 'model2',
                sharedBy: ['model2', 'model3'],
                refCount: 2
              }
            },
            createdAt: Date.now(),
            lastAccessed: Date.now(),
            byteSize: 16
          });
        }
        return Promise.resolve(null);
      });
      
      // Register tensors
      await bridge.registerSharedTensor(
        'text-tensor-1',
        'model1',
        'text_embedding',
        data,
        [1, 4],
        'float32'
      );
      
      await bridge.registerSharedTensor(
        'vision-tensor-1',
        'model2',
        'vision_embedding',
        data,
        [1, 4],
        'float32'
      );
      
      // Share tensors
      await bridge.shareTensor('text-tensor-1', 'model1', 'model2');
      await bridge.shareTensor('vision-tensor-1', 'model2', 'model3');
    });
    
    it('should return accurate sharing statistics', async () => {
      const stats = await bridge.getSharingStats();
      
      expect(stats.totalSharedTensors).toBe(2);
      expect(stats.totalRefCount).toBe(4); // 2 + 2
      expect(stats.bytesSaved).toBe(32); // 16 + 16 (1 ref each saved)
      expect(stats.sharingByType.text_embedding).toBe(1);
      expect(stats.sharingByType.vision_embedding).toBe(1);
    });
  });
  
  describe('garbageCollect', () => {
    beforeEach(async () => {
      await bridge.initialize();
      
      // Set up some shared tensors with zero reference count
      const sharedTensors = new Map();
      sharedTensors.set('tensor1', {
        sharingType: 'text_embedding' as TensorSharingType,
        originalModel: 'model1',
        sharedBy: [],
        refCount: 0
      });
      sharedTensors.set('tensor2', {
        sharingType: 'text_embedding' as TensorSharingType,
        originalModel: 'model1',
        sharedBy: ['model1'],
        refCount: 1
      });
      sharedTensors.set('tensor3', {
        sharingType: 'vision_embedding' as TensorSharingType,
        originalModel: 'model2',
        sharedBy: [],
        refCount: 0
      });
      
      // Inject the shared tensors map
      (bridge as any).sharedTensors = sharedTensors;
      
      // Set up the sharing type map
      const textEmbeddings = new Set(['tensor1', 'tensor2']);
      const visionEmbeddings = new Set(['tensor3']);
      
      const sharingTypeMap = new Map();
      sharingTypeMap.set('text_embedding', textEmbeddings);
      sharingTypeMap.set('vision_embedding', visionEmbeddings);
      
      (bridge as any).sharingTypeMap = sharingTypeMap;
      
      // Set up the model-tensor map
      const modelTensorMap = new Map();
      modelTensorMap.set('model1', new Set(['tensor2']));
      
      (bridge as any).modelTensorMap = modelTensorMap;
      
      // Mock tensor data
      mockStorageManager.getTensor.mockImplementation((tensorId) => {
        if (tensorId === 'tensor1' || tensorId === 'tensor2' || tensorId === 'tensor3') {
          return Promise.resolve({
            id: tensorId,
            shape: [1, 4],
            dataType: 'float32',
            data: new ArrayBuffer(16),
            metadata: {},
            createdAt: Date.now(),
            lastAccessed: Date.now(),
            byteSize: 16
          });
        }
        return Promise.resolve(null);
      });
    });
    
    it('should remove tensors with zero reference count', async () => {
      const result = await bridge.garbageCollect();
      
      expect(result.tensorsRemoved).toBe(2); // tensor1 and tensor3
      expect(result.bytesFreed).toBeGreaterThan(0);
      expect(mockStorageManager.deleteTensor).toHaveBeenCalledWith('tensor1');
      expect(mockStorageManager.deleteTensor).toHaveBeenCalledWith('tensor3');
      expect(mockStorageManager.deleteTensor).not.toHaveBeenCalledWith('tensor2');
      
      // Check that in-memory data was updated
      const sharingStats = await bridge.getSharingStats();
      expect(sharingStats.totalSharedTensors).toBe(1);
      expect(sharingStats.totalRefCount).toBe(1);
    });
  });
  
  describe('clearSharedTensors', () => {
    beforeEach(async () => {
      await bridge.initialize();
      
      // Register some shared tensors
      const data = new Float32Array([1, 2, 3, 4]);
      
      mockStorageManager.getTensor.mockResolvedValue(null);
      
      await bridge.registerSharedTensor(
        'tensor1',
        'model1',
        'text_embedding',
        data,
        [1, 4],
        'float32'
      );
      
      await bridge.registerSharedTensor(
        'tensor2',
        'model2',
        'vision_embedding',
        data,
        [1, 4],
        'float32'
      );
      
      // Set up mock getAllFromObjectStore
      mockStorageManager.getAllFromObjectStore.mockResolvedValue([
        {
          id: 'tensor1',
          metadata: { sharing: {} }
        },
        {
          id: 'tensor2',
          metadata: { sharing: {} }
        },
        {
          id: 'tensor3',
          metadata: {} // Not a shared tensor
        }
      ]);
    });
    
    it('should clear all shared tensors', async () => {
      const result = await bridge.clearSharedTensors();
      
      expect(result).toBe(true);
      expect(mockStorageManager.deleteTensor).toHaveBeenCalledWith('tensor1');
      expect(mockStorageManager.deleteTensor).toHaveBeenCalledWith('tensor2');
      expect(mockStorageManager.deleteTensor).not.toHaveBeenCalledWith('tensor3');
      
      // Check that in-memory data was cleared
      const sharingStats = await bridge.getSharingStats();
      expect(sharingStats.totalSharedTensors).toBe(0);
      expect(sharingStats.totalRefCount).toBe(0);
    });
  });
  
  describe('Cache management', () => {
    const createMockTensor = (size: number) => {
      // Create a Float32Array of the specified size (in elements)
      return new Float32Array(size);
    };
    
    beforeEach(async () => {
      await bridge.initialize();
      
      // Set a small cache size for testing
      (bridge as any).options.maxCacheSize = 1024; // 1KB
      
      // Register some shared tensors
      mockStorageManager.getTensor.mockImplementation((tensorId) => {
        const shapes: Record<string, number[]> = {
          'small-tensor': [1, 100], // 400 bytes
          'medium-tensor': [1, 150], // 600 bytes
          'large-tensor': [1, 200]   // 800 bytes
        };
        
        if (tensorId in shapes) {
          return Promise.resolve({
            id: tensorId,
            shape: shapes[tensorId],
            dataType: 'float32',
            data: new ArrayBuffer(shapes[tensorId][1] * 4),
            metadata: {
              sharing: {
                sharingType: 'text_embedding',
                originalModel: 'model1',
                sharedBy: ['model1'],
                refCount: 1
              }
            },
            createdAt: Date.now(),
            lastAccessed: Date.now(),
            byteSize: shapes[tensorId][1] * 4
          });
        }
        return Promise.resolve(null);
      });
      
      // Set up the tensor data
      mockStorageManager.getTensorData.mockImplementation((tensorId) => {
        const sizes: Record<string, number> = {
          'small-tensor': 100,  // 400 bytes
          'medium-tensor': 150, // 600 bytes
          'large-tensor': 200   // 800 bytes
        };
        
        if (tensorId in sizes) {
          return Promise.resolve(createMockTensor(sizes[tensorId]));
        }
        return Promise.resolve(null);
      });
      
      // Register tensors with sharing
      for (const tensorId of ['small-tensor', 'medium-tensor', 'large-tensor']) {
        (bridge as any).sharedTensors.set(tensorId, {
          sharingType: 'text_embedding',
          originalModel: 'model1',
          sharedBy: ['model1'],
          refCount: 1
        });
        
        if (!(bridge as any).modelTensorMap.has('model1')) {
          (bridge as any).modelTensorMap.set('model1', new Set());
        }
        (bridge as any).modelTensorMap.get('model1').add(tensorId);
        
        if (!(bridge as any).sharingTypeMap.has('text_embedding')) {
          (bridge as any).sharingTypeMap.set('text_embedding', new Set());
        }
        (bridge as any).sharingTypeMap.get('text_embedding').add(tensorId);
      }
    });
    
    it('should cache tensors up to the maximum size', async () => {
      // Load small tensor (400 bytes)
      await bridge.getSharedTensor('small-tensor', 'model1');
      
      // Check that it's in the cache
      expect((bridge as any).tensorCache.has('small-tensor')).toBe(true);
      expect((bridge as any).cacheSize).toBe(400);
      
      // Load medium tensor (600 bytes)
      await bridge.getSharedTensor('medium-tensor', 'model1');
      
      // Both should be in cache (1000 bytes total)
      expect((bridge as any).tensorCache.has('small-tensor')).toBe(true);
      expect((bridge as any).tensorCache.has('medium-tensor')).toBe(true);
      expect((bridge as any).cacheSize).toBe(1000);
      
      // Reset getTensorData mock to see if it's called again
      mockStorageManager.getTensorData.mockClear();
      
      // Load large tensor (800 bytes)
      // This should remove the small tensor from cache (oldest)
      await bridge.getSharedTensor('large-tensor', 'model1');
      
      // Cache should now have medium and large tensors
      expect((bridge as any).tensorCache.has('small-tensor')).toBe(false);
      expect((bridge as any).tensorCache.has('medium-tensor')).toBe(true);
      expect((bridge as any).tensorCache.has('large-tensor')).toBe(true);
      expect((bridge as any).cacheSize).toBe(1400);
      
      // Try to load small tensor again
      await bridge.getSharedTensor('small-tensor', 'model1');
      
      // This should have triggered a storage read
      expect(mockStorageManager.getTensorData).toHaveBeenCalledWith('small-tensor');
    });
  });
});