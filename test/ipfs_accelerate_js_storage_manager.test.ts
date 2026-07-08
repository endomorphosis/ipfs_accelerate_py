/**
 * WebNN Storage Manager Tests
 */

import { StorageManager, TensorStorage, ModelMetadata } from './ipfs_accelerate_js_storage_manager';

// Mock IndexedDB implementation
const mockIndexedDB = {
  open: jest.fn(),
};

const mockDBRequest = {
  onupgradeneeded: null as Function | null,
  onsuccess: null as Function | null,
  onerror: null as Function | null,
  error: null,
};

const mockObjectStore = {
  put: jest.fn(),
  get: jest.fn(),
  getAll: jest.fn(),
  delete: jest.fn(),
  clear: jest.fn(),
  count: jest.fn(),
  createIndex: jest.fn(),
  index: jest.fn(),
};

const mockTransaction = {
  objectStore: jest.fn().mockReturnValue(mockObjectStore),
};

const mockDB = {
  transaction: jest.fn().mockReturnValue(mockTransaction),
  objectStoreNames: {
    contains: jest.fn().mockReturnValue(false),
  },
  createObjectStore: jest.fn().mockReturnValue(mockObjectStore),
};

const mockIndex = {
  openCursor: jest.fn(),
};

// Mock cursor for query operations
const createMockCursor = (values: any[]) => {
  let currentIndex = 0;
  
  return {
    value: null,
    continue: jest.fn().mockImplementation(() => {
      currentIndex++;
      
      if (currentIndex < values.length) {
        const event = {
          target: {
            result: {
              value: values[currentIndex],
              continue: jest.fn(),
            },
          },
        };
        (mockIndex.openCursor as jest.Mock).mock.results[0].value.onsuccess(event);
      } else {
        const event = {
          target: {
            result: null,
          },
        };
        (mockIndex.openCursor as jest.Mock).mock.results[0].value.onsuccess(event);
      }
    }),
  };
};

// Set up global mocks
(global as any).indexedDB = mockIndexedDB;
(global as any).IDBKeyRange = {
  upperBound: jest.fn().mockReturnValue('upperBound'),
};

describe('StorageManager', () => {
  let storageManager: StorageManager;
  
  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Set up mock implementations
    mockIndexedDB.open.mockReturnValue(mockDBRequest);
    mockObjectStore.index.mockReturnValue(mockIndex);
    
    // Create storage manager instance
    storageManager = new StorageManager({
      dbName: 'test-db',
      dbVersion: 1,
      enableLogging: false,
    });
  });
  
  describe('initialize', () => {
    test('should initialize successfully', async () => {
      // Set up mock for successful initialization
      mockIndexedDB.open.mockImplementation(() => {
        setTimeout(() => {
          mockDBRequest.onsuccess?.({ target: { result: mockDB } } as any);
        }, 0);
        return mockDBRequest;
      });
      
      // Mock for updateStorageStats
      mockObjectStore.count.mockImplementation(() => {
        const countRequest = { result: 0, onsuccess: null as Function | null };
        setTimeout(() => {
          countRequest.onsuccess?.({ target: countRequest } as any);
        }, 0);
        return countRequest;
      });
      
      mockObjectStore.getAll.mockImplementation(() => {
        const getAllRequest = { result: [], onsuccess: null as Function | null };
        setTimeout(() => {
          getAllRequest.onsuccess?.({ target: getAllRequest } as any);
        }, 0);
        return getAllRequest;
      });
      
      const result = await storageManager.initialize();
      
      expect(result).toBe(true);
      expect(mockIndexedDB.open).toHaveBeenCalledWith('test-db', 1);
    });
    
    test('should handle initialization failure', async () => {
      // Set up mock for initialization failure
      mockIndexedDB.open.mockImplementation(() => {
        setTimeout(() => {
          mockDBRequest.error = new Error('Test error');
          mockDBRequest.onerror?.({ target: mockDBRequest } as any);
        }, 0);
        return mockDBRequest;
      });
      
      const result = await storageManager.initialize();
      
      expect(result).toBe(false);
    });
    
    test('should create object stores during upgrade', async () => {
      // Simulate onupgradeneeded event
      mockIndexedDB.open.mockImplementation(() => {
        setTimeout(() => {
          mockDBRequest.onupgradeneeded?.({ target: { result: mockDB } } as any);
          mockDBRequest.onsuccess?.({ target: { result: mockDB } } as any);
        }, 0);
        return mockDBRequest;
      });
      
      // Mock for updateStorageStats
      mockObjectStore.count.mockImplementation(() => {
        const countRequest = { result: 0, onsuccess: null as Function | null };
        setTimeout(() => {
          countRequest.onsuccess?.({ target: countRequest } as any);
        }, 0);
        return countRequest;
      });
      
      mockObjectStore.getAll.mockImplementation(() => {
        const getAllRequest = { result: [], onsuccess: null as Function | null };
        setTimeout(() => {
          getAllRequest.onsuccess?.({ target: getAllRequest } as any);
        }, 0);
        return getAllRequest;
      });
      
      await storageManager.initialize();
      
      expect(mockDB.createObjectStore).toHaveBeenCalledWith('models', { keyPath: 'id' });
      expect(mockDB.createObjectStore).toHaveBeenCalledWith('tensors', { keyPath: 'id' });
      expect(mockObjectStore.createIndex).toHaveBeenCalledWith('name', 'name', { unique: false });
      expect(mockObjectStore.createIndex).toHaveBeenCalledWith('createdAt', 'createdAt', { unique: false });
      expect(mockObjectStore.createIndex).toHaveBeenCalledWith('lastAccessed', 'lastAccessed', { unique: false });
    });
  });
  
  describe('Model operations', () => {
    beforeEach(async () => {
      // Set up mock for successful initialization
      mockIndexedDB.open.mockImplementation(() => {
        setTimeout(() => {
          mockDBRequest.onsuccess?.({ target: { result: mockDB } } as any);
        }, 0);
        return mockDBRequest;
      });
      
      // Mock for updateStorageStats
      mockObjectStore.count.mockImplementation(() => {
        const countRequest = { result: 0, onsuccess: null as Function | null };
        setTimeout(() => {
          countRequest.onsuccess?.({ target: countRequest } as any);
        }, 0);
        return countRequest;
      });
      
      mockObjectStore.getAll.mockImplementation(() => {
        const getAllRequest = { result: [], onsuccess: null as Function | null };
        setTimeout(() => {
          getAllRequest.onsuccess?.({ target: getAllRequest } as any);
        }, 0);
        return getAllRequest;
      });
      
      await storageManager.initialize();
    });
    
    test('should store model metadata', async () => {
      // Mock for putInObjectStore
      mockObjectStore.put.mockImplementation(() => {
        const putRequest = { onsuccess: null as Function | null };
        setTimeout(() => {
          putRequest.onsuccess?.({ target: putRequest } as any);
        }, 0);
        return putRequest;
      });
      
      const modelMetadata = {
        id: 'model1',
        name: 'Test Model',
        version: '1.0.0',
        totalSize: 1000,
        tensorIds: ['tensor1', 'tensor2'],
      };
      
      const result = await storageManager.storeModelMetadata(modelMetadata);
      
      expect(result).toBe(true);
      expect(mockTransaction.objectStore).toHaveBeenCalledWith('models');
      expect(mockObjectStore.put).toHaveBeenCalled();
      expect(mockObjectStore.put.mock.calls[0][0]).toMatchObject({
        id: 'model1',
        name: 'Test Model',
        version: '1.0.0',
        totalSize: 1000,
        tensorIds: ['tensor1', 'tensor2'],
      });
    });
    
    test('should get model metadata', async () => {
      // Mock for getFromObjectStore
      const mockModel: ModelMetadata = {
        id: 'model1',
        name: 'Test Model',
        version: '1.0.0',
        totalSize: 1000,
        tensorIds: ['tensor1', 'tensor2'],
        createdAt: Date.now(),
        lastAccessed: Date.now(),
      };
      
      mockObjectStore.get.mockImplementation(() => {
        const getRequest = { result: mockModel, onsuccess: null as Function | null };
        setTimeout(() => {
          getRequest.onsuccess?.({ target: getRequest } as any);
        }, 0);
        return getRequest;
      });
      
      // Mock for putInObjectStore (for lastAccessed update)
      mockObjectStore.put.mockImplementation(() => {
        const putRequest = { onsuccess: null as Function | null };
        setTimeout(() => {
          putRequest.onsuccess?.({ target: putRequest } as any);
        }, 0);
        return putRequest;
      });
      
      const result = await storageManager.getModelMetadata('model1');
      
      expect(result).toEqual(mockModel);
      expect(mockTransaction.objectStore).toHaveBeenCalledWith('models');
      expect(mockObjectStore.get).toHaveBeenCalledWith('model1');
    });
    
    test('should list all models', async () => {
      // Mock for getAllFromObjectStore
      const mockModels: ModelMetadata[] = [
        {
          id: 'model1',
          name: 'Test Model 1',
          version: '1.0.0',
          totalSize: 1000,
          tensorIds: ['tensor1', 'tensor2'],
          createdAt: Date.now(),
          lastAccessed: Date.now(),
        },
        {
          id: 'model2',
          name: 'Test Model 2',
          version: '1.0.0',
          totalSize: 2000,
          tensorIds: ['tensor3', 'tensor4'],
          createdAt: Date.now(),
          lastAccessed: Date.now(),
        },
      ];
      
      mockObjectStore.getAll.mockImplementation(() => {
        const getAllRequest = { result: mockModels, onsuccess: null as Function | null };
        setTimeout(() => {
          getAllRequest.onsuccess?.({ target: getAllRequest } as any);
        }, 0);
        return getAllRequest;
      });
      
      const result = await storageManager.listModels();
      
      expect(result).toEqual(mockModels);
      expect(mockTransaction.objectStore).toHaveBeenCalledWith('models');
      expect(mockObjectStore.getAll).toHaveBeenCalled();
    });
  });
  
  describe('Tensor operations', () => {
    beforeEach(async () => {
      // Set up mock for successful initialization
      mockIndexedDB.open.mockImplementation(() => {
        setTimeout(() => {
          mockDBRequest.onsuccess?.({ target: { result: mockDB } } as any);
        }, 0);
        return mockDBRequest;
      });
      
      // Mock for updateStorageStats
      mockObjectStore.count.mockImplementation(() => {
        const countRequest = { result: 0, onsuccess: null as Function | null };
        setTimeout(() => {
          countRequest.onsuccess?.({ target: countRequest } as any);
        }, 0);
        return countRequest;
      });
      
      mockObjectStore.getAll.mockImplementation(() => {
        const getAllRequest = { result: [], onsuccess: null as Function | null };
        setTimeout(() => {
          getAllRequest.onsuccess?.({ target: getAllRequest } as any);
        }, 0);
        return getAllRequest;
      });
      
      await storageManager.initialize();
    });
    
    test('should store tensor data', async () => {
      // Mock for putInObjectStore
      mockObjectStore.put.mockImplementation(() => {
        const putRequest = { onsuccess: null as Function | null };
        setTimeout(() => {
          putRequest.onsuccess?.({ target: putRequest } as any);
        }, 0);
        return putRequest;
      });
      
      const data = new Float32Array([1, 2, 3, 4]);
      
      const result = await storageManager.storeTensor(
        'tensor1',
        data,
        [2, 2],
        'float32',
        { name: 'activation' }
      );
      
      expect(result).toBe(true);
      expect(mockTransaction.objectStore).toHaveBeenCalledWith('tensors');
      expect(mockObjectStore.put).toHaveBeenCalled();
      
      // Check that the stored tensor has the correct properties
      const storedTensor = mockObjectStore.put.mock.calls[0][0];
      expect(storedTensor.id).toBe('tensor1');
      expect(storedTensor.shape).toEqual([2, 2]);
      expect(storedTensor.dataType).toBe('float32');
      expect(storedTensor.metadata).toEqual({ name: 'activation' });
      expect(storedTensor.byteSize).toBe(16); // 4 floats * 4 bytes
    });
    
    test('should get tensor data', async () => {
      // Create mock tensor
      const buffer = new ArrayBuffer(16);
      const floatView = new Float32Array(buffer);
      floatView.set([1, 2, 3, 4]);
      
      const mockTensor: TensorStorage = {
        id: 'tensor1',
        shape: [2, 2],
        dataType: 'float32',
        data: buffer,
        createdAt: Date.now(),
        lastAccessed: Date.now(),
        byteSize: 16,
      };
      
      // Mock for getFromObjectStore
      mockObjectStore.get.mockImplementation(() => {
        const getRequest = { result: mockTensor, onsuccess: null as Function | null };
        setTimeout(() => {
          getRequest.onsuccess?.({ target: getRequest } as any);
        }, 0);
        return getRequest;
      });
      
      // Mock for putInObjectStore (for lastAccessed update)
      mockObjectStore.put.mockImplementation(() => {
        const putRequest = { onsuccess: null as Function | null };
        setTimeout(() => {
          putRequest.onsuccess?.({ target: putRequest } as any);
        }, 0);
        return putRequest;
      });
      
      const result = await storageManager.getTensorData('tensor1');
      
      expect(result).toBeInstanceOf(Float32Array);
      expect(Array.from(result as Float32Array)).toEqual([1, 2, 3, 4]);
    });
  });
  
  describe('Storage management', () => {
    beforeEach(async () => {
      // Set up mock for successful initialization
      mockIndexedDB.open.mockImplementation(() => {
        setTimeout(() => {
          mockDBRequest.onsuccess?.({ target: { result: mockDB } } as any);
        }, 0);
        return mockDBRequest;
      });
      
      // Mock for updateStorageStats
      mockObjectStore.count.mockImplementation(() => {
        const countRequest = { result: 0, onsuccess: null as Function | null };
        setTimeout(() => {
          countRequest.onsuccess?.({ target: countRequest } as any);
        }, 0);
        return countRequest;
      });
      
      mockObjectStore.getAll.mockImplementation(() => {
        const getAllRequest = { result: [], onsuccess: null as Function | null };
        setTimeout(() => {
          getAllRequest.onsuccess?.({ target: getAllRequest } as any);
        }, 0);
        return getAllRequest;
      });
      
      await storageManager.initialize();
    });
    
    test('should clear all stored data', async () => {
      // Mock for clearObjectStore
      mockObjectStore.clear.mockImplementation(() => {
        const clearRequest = { onsuccess: null as Function | null };
        setTimeout(() => {
          clearRequest.onsuccess?.({ target: clearRequest } as any);
        }, 0);
        return clearRequest;
      });
      
      const result = await storageManager.clear();
      
      expect(result).toBe(true);
      expect(mockTransaction.objectStore).toHaveBeenCalledWith('models');
      expect(mockTransaction.objectStore).toHaveBeenCalledWith('tensors');
      expect(mockObjectStore.clear).toHaveBeenCalledTimes(2);
    });
    
    test('should get storage info', async () => {
      // Mock storage stats
      mockObjectStore.count.mockImplementationOnce(() => {
        const countRequest = { result: 2, onsuccess: null as Function | null };
        setTimeout(() => {
          countRequest.onsuccess?.({ target: countRequest } as any);
        }, 0);
        return countRequest;
      }).mockImplementationOnce(() => {
        const countRequest = { result: 5, onsuccess: null as Function | null };
        setTimeout(() => {
          countRequest.onsuccess?.({ target: countRequest } as any);
        }, 0);
        return countRequest;
      });
      
      const mockTensors = [
        { byteSize: 100 },
        { byteSize: 200 },
        { byteSize: 300 },
        { byteSize: 400 },
        { byteSize: 500 },
      ];
      
      mockObjectStore.getAll.mockImplementation(() => {
        const getAllRequest = { result: mockTensors, onsuccess: null as Function | null };
        setTimeout(() => {
          getAllRequest.onsuccess?.({ target: getAllRequest } as any);
        }, 0);
        return getAllRequest;
      });
      
      // Mock navigator.storage.estimate
      (global as any).navigator = {
        storage: {
          estimate: jest.fn().mockResolvedValue({
            usage: 2000,
            quota: 10000,
          }),
        },
      };
      
      const result = await storageManager.getStorageInfo();
      
      expect(result).toEqual({
        modelCount: 2,
        tensorCount: 5,
        totalSize: 1500, // Sum of tensor sizes
        remainingQuota: 8000, // 10000 - 2000
        dbName: 'test-db',
        dbVersion: 1,
      });
    });
    
    test('should clean up old tensors', async () => {
      // Mock for runObjectStoreQuery to return unused tensors
      const mockUnusedTensors = [
        { id: 'tensor1', byteSize: 100 },
        { id: 'tensor2', byteSize: 200 },
      ];
      
      // Create mock cursor with fake results
      const firstCursor = createMockCursor(mockUnusedTensors);
      
      mockIndex.openCursor.mockImplementation(() => {
        const cursorRequest = {
          onsuccess: null as Function | null,
          result: firstCursor,
        };
        
        setTimeout(() => {
          cursorRequest.onsuccess?.({ target: cursorRequest } as any);
        }, 0);
        
        return cursorRequest;
      });
      
      // Mock for deleteFromObjectStore
      mockObjectStore.delete.mockImplementation(() => {
        const deleteRequest = { onsuccess: null as Function | null };
        setTimeout(() => {
          deleteRequest.onsuccess?.({ target: deleteRequest } as any);
        }, 0);
        return deleteRequest;
      });
      
      jest.spyOn(Date, 'now').mockReturnValue(1000 * 60 * 60 * 24 * 10); // 10 days
      
      const result = await storageManager.cleanup();
      
      expect(result).toBe(300); // Total freed: 100 + 200
      expect(mockObjectStore.delete).toHaveBeenCalledTimes(2);
    });
  });
});