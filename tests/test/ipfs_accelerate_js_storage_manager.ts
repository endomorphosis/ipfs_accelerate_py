/**
 * WebNN Storage Manager Implementation
 * Provides storage and caching for model weights and tensors using IndexedDB
 */

/**
 * Data types supported by the storage manager
 */
export type StorageDataType = 'float32' | 'int32' | 'uint8' | 'float16';

/**
 * Tensor storage format
 */
export interface TensorStorage {
  /**
   * Unique identifier for the tensor
   */
  id: string;
  
  /**
   * Shape of the tensor
   */
  shape: number[];
  
  /**
   * Data type of the tensor values
   */
  dataType: StorageDataType;
  
  /**
   * Binary data for the tensor
   */
  data: ArrayBuffer;
  
  /**
   * Metadata for the tensor (e.g. name, layer, etc.)
   */
  metadata?: Record<string, any>;
  
  /**
   * Creation timestamp
   */
  createdAt: number;
  
  /**
   * Last access timestamp
   */
  lastAccessed: number;
  
  /**
   * Size in bytes
   */
  byteSize: number;
}

/**
 * Model metadata structure
 */
export interface ModelMetadata {
  /**
   * Unique identifier for the model
   */
  id: string;
  
  /**
   * Model name
   */
  name: string;
  
  /**
   * Model version
   */
  version: string;
  
  /**
   * Framework the model is from (e.g., "tensorflow", "pytorch", "onnx")
   */
  framework?: string;
  
  /**
   * Total size in bytes
   */
  totalSize: number;
  
  /**
   * List of tensor IDs that belong to this model
   */
  tensorIds: string[];
  
  /**
   * Creation timestamp
   */
  createdAt: number;
  
  /**
   * Last access timestamp
   */
  lastAccessed: number;
  
  /**
   * Additional model information
   */
  metadata?: Record<string, any>;
}

/**
 * Storage manager options
 */
export interface StorageManagerOptions {
  /**
   * Name of the IndexedDB database
   */
  dbName?: string;
  
  /**
   * Database version
   */
  dbVersion?: number;
  
  /**
   * Enable compression for stored tensors
   */
  enableCompression?: boolean;
  
  /**
   * Maximum storage size in bytes (0 for unlimited)
   */
  maxStorageSize?: number;
  
  /**
   * Enable automatic cleanup of unused items
   */
  enableAutoCleanup?: boolean;
  
  /**
   * Time in milliseconds after which unused items can be removed
   */
  cleanupThreshold?: number;
  
  /**
   * Enable logging of storage operations
   */
  enableLogging?: boolean;
}

/**
 * Storage information
 */
export interface StorageInfo {
  /**
   * Total number of models stored
   */
  modelCount: number;
  
  /**
   * Total number of tensors stored
   */
  tensorCount: number;
  
  /**
   * Total size in bytes used
   */
  totalSize: number;
  
  /**
   * Remaining storage quota in bytes (if available)
   */
  remainingQuota?: number;
  
  /**
   * Database name
   */
  dbName: string;
  
  /**
   * Database version
   */
  dbVersion: number;
}

/**
 * Storage Manager for model weights and tensors
 * Uses IndexedDB for persistent storage
 */
export class StorageManager {
  private options: Required<StorageManagerOptions>;
  private db: IDBDatabase | null = null;
  private isInitialized = false;
  private initPromise: Promise<boolean> | null = null;
  
  // Store objects count and size for quick access
  private modelCount = 0;
  private tensorCount = 0;
  private totalSize = 0;
  
  /**
   * Creates a new storage manager instance
   */
  constructor(options: StorageManagerOptions = {}) {
    // Set default options
    this.options = {
      dbName: options.dbName || 'webnn-storage',
      dbVersion: options.dbVersion || 1,
      enableCompression: options.enableCompression || false,
      maxStorageSize: options.maxStorageSize || 0, // 0 means no limit
      enableAutoCleanup: options.enableAutoCleanup || true,
      cleanupThreshold: options.cleanupThreshold || 1000 * 60 * 60 * 24 * 7, // 7 days
      enableLogging: options.enableLogging || false
    };
  }
  
  /**
   * Initialize the storage manager and open the database
   */
  async initialize(): Promise<boolean> {
    // If already initializing, return the existing promise
    if (this.initPromise) {
      return this.initPromise;
    }
    
    // If already initialized, return true
    if (this.isInitialized) {
      return true;
    }
    
    // Create a new initialization promise
    this.initPromise = new Promise<boolean>((resolve, reject) => {
      try {
        // Check if IndexedDB is supported
        if (!('indexedDB' in window)) {
          this.log('IndexedDB is not supported in this browser');
          resolve(false);
          return;
        }
        
        // Open the database
        const request = indexedDB.open(this.options.dbName, this.options.dbVersion);
        
        // Handle database upgrade (creating object stores)
        request.onupgradeneeded = (event) => {
          const db = (event.target as IDBOpenDBRequest).result;
          
          // Create models store
          if (!db.objectStoreNames.contains('models')) {
            const modelsStore = db.createObjectStore('models', { keyPath: 'id' });
            modelsStore.createIndex('name', 'name', { unique: false });
            modelsStore.createIndex('createdAt', 'createdAt', { unique: false });
            modelsStore.createIndex('lastAccessed', 'lastAccessed', { unique: false });
          }
          
          // Create tensors store
          if (!db.objectStoreNames.contains('tensors')) {
            const tensorsStore = db.createObjectStore('tensors', { keyPath: 'id' });
            tensorsStore.createIndex('createdAt', 'createdAt', { unique: false });
            tensorsStore.createIndex('lastAccessed', 'lastAccessed', { unique: false });
            tensorsStore.createIndex('byteSize', 'byteSize', { unique: false });
          }
        };
        
        // Handle successful database open
        request.onsuccess = async (event) => {
          this.db = (event.target as IDBOpenDBRequest).result;
          this.isInitialized = true;
          this.log('Storage manager initialized successfully');
          
          // Update storage statistics
          await this.updateStorageStats();
          
          // Run cleanup if auto-cleanup is enabled
          if (this.options.enableAutoCleanup) {
            this.cleanup().catch(error => {
              this.log(`Auto cleanup failed: ${error.message}`);
            });
          }
          
          resolve(true);
        };
        
        // Handle errors
        request.onerror = (event) => {
          const error = (event.target as IDBOpenDBRequest).error;
          this.log(`Failed to initialize storage manager: ${error?.message}`);
          resolve(false);
        };
      } catch (error) {
        this.log(`Initialization error: ${error.message}`);
        resolve(false);
      }
    });
    
    return this.initPromise;
  }
  
  /**
   * Store a model's metadata
   */
  async storeModelMetadata(metadata: Omit<ModelMetadata, 'createdAt' | 'lastAccessed'>): Promise<boolean> {
    if (!await this.ensureInitialized()) {
      return false;
    }
    
    try {
      // Add timestamps
      const model: ModelMetadata = {
        ...metadata,
        createdAt: Date.now(),
        lastAccessed: Date.now()
      };
      
      // Store in database
      await this.putInObjectStore('models', model);
      
      // Update stats
      this.modelCount++;
      this.totalSize += model.totalSize;
      
      this.log(`Stored model metadata: ${model.id} (${model.name})`);
      return true;
    } catch (error) {
      this.log(`Failed to store model metadata: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Store a tensor
   */
  async storeTensor(
    id: string, 
    data: Float32Array | Int32Array | Uint8Array, 
    shape: number[], 
    dataType: StorageDataType,
    metadata?: Record<string, any>
  ): Promise<boolean> {
    if (!await this.ensureInitialized()) {
      return false;
    }
    
    try {
      // Calculate byte size
      const byteSize = data.byteLength;
      
      // Create tensor storage object
      const tensor: TensorStorage = {
        id,
        shape,
        dataType,
        data: data.buffer.slice(0), // Create a copy of the buffer
        metadata,
        createdAt: Date.now(),
        lastAccessed: Date.now(),
        byteSize
      };
      
      // Check if we need to compress the data
      if (this.options.enableCompression && byteSize > 1024) {
        // In a real implementation, we would compress the data here
        // For now, we'll just store it as is
      }
      
      // Check if we have enough storage space
      if (this.options.maxStorageSize > 0) {
        const newTotalSize = this.totalSize + byteSize;
        if (newTotalSize > this.options.maxStorageSize) {
          // Try to free up space
          const freedSpace = await this.freeUpSpace(byteSize);
          if (freedSpace < byteSize) {
            this.log(`Not enough storage space for tensor: ${id}`);
            return false;
          }
        }
      }
      
      // Store in database
      await this.putInObjectStore('tensors', tensor);
      
      // Update stats
      this.tensorCount++;
      this.totalSize += byteSize;
      
      this.log(`Stored tensor: ${id} (${shape.join('x')}, ${dataType})`);
      return true;
    } catch (error) {
      this.log(`Failed to store tensor: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Get a model's metadata
   */
  async getModelMetadata(modelId: string): Promise<ModelMetadata | null> {
    if (!await this.ensureInitialized()) {
      return null;
    }
    
    try {
      // Get model metadata from database
      const model = await this.getFromObjectStore('models', modelId) as ModelMetadata;
      
      if (!model) {
        return null;
      }
      
      // Update last accessed timestamp
      model.lastAccessed = Date.now();
      await this.putInObjectStore('models', model);
      
      return model;
    } catch (error) {
      this.log(`Failed to get model metadata: ${error.message}`);
      return null;
    }
  }
  
  /**
   * Get a list of all stored models
   */
  async listModels(): Promise<ModelMetadata[]> {
    if (!await this.ensureInitialized()) {
      return [];
    }
    
    try {
      // Get all models from database
      return await this.getAllFromObjectStore('models') as ModelMetadata[];
    } catch (error) {
      this.log(`Failed to list models: ${error.message}`);
      return [];
    }
  }
  
  /**
   * Get a tensor by ID
   */
  async getTensor(tensorId: string): Promise<TensorStorage | null> {
    if (!await this.ensureInitialized()) {
      return null;
    }
    
    try {
      // Get tensor from database
      const tensor = await this.getFromObjectStore('tensors', tensorId) as TensorStorage;
      
      if (!tensor) {
        return null;
      }
      
      // Update last accessed timestamp
      tensor.lastAccessed = Date.now();
      await this.putInObjectStore('tensors', tensor);
      
      // Decompress if necessary
      if (this.options.enableCompression) {
        // In a real implementation, we would decompress the data here
        // For now, we'll just return it as is
      }
      
      return tensor;
    } catch (error) {
      this.log(`Failed to get tensor: ${error.message}`);
      return null;
    }
  }
  
  /**
   * Get tensor data and convert to appropriate TypedArray
   */
  async getTensorData(tensorId: string): Promise<Float32Array | Int32Array | Uint8Array | null> {
    const tensor = await this.getTensor(tensorId);
    
    if (!tensor) {
      return null;
    }
    
    // Convert ArrayBuffer to appropriate TypedArray based on dataType
    switch (tensor.dataType) {
      case 'float32':
        return new Float32Array(tensor.data);
      case 'int32':
        return new Int32Array(tensor.data);
      case 'uint8':
        return new Uint8Array(tensor.data);
      case 'float16':
        // In a real implementation, we would handle float16 conversion
        // For now, we'll just return it as a Uint8Array
        return new Uint8Array(tensor.data);
      default:
        this.log(`Unknown tensor data type: ${tensor.dataType}`);
        return null;
    }
  }
  
  /**
   * Get storage information
   */
  async getStorageInfo(): Promise<StorageInfo> {
    if (!await this.ensureInitialized()) {
      return {
        modelCount: 0,
        tensorCount: 0,
        totalSize: 0,
        dbName: this.options.dbName,
        dbVersion: this.options.dbVersion
      };
    }
    
    // Refresh storage statistics
    await this.updateStorageStats();
    
    // Try to get remaining quota if available
    let remainingQuota: number | undefined = undefined;
    
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      try {
        const estimate = await navigator.storage.estimate();
        if (estimate.quota && estimate.usage) {
          remainingQuota = estimate.quota - estimate.usage;
        }
      } catch (error) {
        this.log(`Failed to get storage estimate: ${error.message}`);
      }
    }
    
    return {
      modelCount: this.modelCount,
      tensorCount: this.tensorCount,
      totalSize: this.totalSize,
      remainingQuota,
      dbName: this.options.dbName,
      dbVersion: this.options.dbVersion
    };
  }
  
  /**
   * Delete a model and all its tensors
   */
  async deleteModel(modelId: string): Promise<boolean> {
    if (!await this.ensureInitialized()) {
      return false;
    }
    
    try {
      // Get model metadata
      const model = await this.getModelMetadata(modelId);
      
      if (!model) {
        this.log(`Model not found: ${modelId}`);
        return false;
      }
      
      // Delete all tensors
      for (const tensorId of model.tensorIds) {
        await this.deleteTensor(tensorId);
      }
      
      // Delete model metadata
      await this.deleteFromObjectStore('models', modelId);
      
      // Update stats
      this.modelCount--;
      
      this.log(`Deleted model: ${modelId}`);
      return true;
    } catch (error) {
      this.log(`Failed to delete model: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Delete a tensor
   */
  async deleteTensor(tensorId: string): Promise<boolean> {
    if (!await this.ensureInitialized()) {
      return false;
    }
    
    try {
      // Get tensor to get its size
      const tensor = await this.getTensor(tensorId);
      
      if (!tensor) {
        this.log(`Tensor not found: ${tensorId}`);
        return false;
      }
      
      // Delete tensor
      await this.deleteFromObjectStore('tensors', tensorId);
      
      // Update stats
      this.tensorCount--;
      this.totalSize -= tensor.byteSize;
      
      this.log(`Deleted tensor: ${tensorId}`);
      return true;
    } catch (error) {
      this.log(`Failed to delete tensor: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Clear all stored data
   */
  async clear(): Promise<boolean> {
    if (!await this.ensureInitialized()) {
      return false;
    }
    
    try {
      // Clear object stores
      await this.clearObjectStore('models');
      await this.clearObjectStore('tensors');
      
      // Reset stats
      this.modelCount = 0;
      this.tensorCount = 0;
      this.totalSize = 0;
      
      this.log('Cleared all stored data');
      return true;
    } catch (error) {
      this.log(`Failed to clear stored data: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Clean up old unused data
   */
  async cleanup(): Promise<number> {
    if (!await this.ensureInitialized()) {
      return 0;
    }
    
    try {
      const threshold = Date.now() - this.options.cleanupThreshold;
      let freedSpace = 0;
      
      // First, find unused tensors
      const unusedTensors = await this.runObjectStoreQuery<TensorStorage>(
        'tensors',
        'lastAccessed',
        IDBKeyRange.upperBound(threshold)
      );
      
      // Delete unused tensors
      for (const tensor of unusedTensors) {
        await this.deleteFromObjectStore('tensors', tensor.id);
        freedSpace += tensor.byteSize;
        this.tensorCount--;
        this.totalSize -= tensor.byteSize;
      }
      
      this.log(`Cleanup freed ${freedSpace} bytes from ${unusedTensors.length} tensors`);
      return freedSpace;
    } catch (error) {
      this.log(`Cleanup failed: ${error.message}`);
      return 0;
    }
  }
  
  /**
   * Free up space for new data
   */
  private async freeUpSpace(requiredBytes: number): Promise<number> {
    if (!await this.ensureInitialized()) {
      return 0;
    }
    
    // First try to run cleanup
    const freedFromCleanup = await this.cleanup();
    if (freedFromCleanup >= requiredBytes) {
      return freedFromCleanup;
    }
    
    try {
      let totalFreed = freedFromCleanup;
      
      // If still not enough, start deleting oldest tensors until we have enough space
      while (totalFreed < requiredBytes) {
        // Get oldest tensor
        const oldestTensors = await this.runObjectStoreQuery<TensorStorage>(
          'tensors',
          'lastAccessed',
          null,
          1
        );
        
        if (oldestTensors.length === 0) {
          break; // No more tensors to delete
        }
        
        const tensor = oldestTensors[0];
        await this.deleteFromObjectStore('tensors', tensor.id);
        
        totalFreed += tensor.byteSize;
        this.tensorCount--;
        this.totalSize -= tensor.byteSize;
      }
      
      this.log(`Freed ${totalFreed} bytes of space`);
      return totalFreed;
    } catch (error) {
      this.log(`Failed to free up space: ${error.message}`);
      return freedFromCleanup;
    }
  }
  
  /**
   * Update storage statistics
   */
  private async updateStorageStats(): Promise<void> {
    if (!this.db) return;
    
    try {
      // Count models
      this.modelCount = await this.countObjectStore('models');
      
      // Count tensors
      this.tensorCount = await this.countObjectStore('tensors');
      
      // Calculate total size from tensor sizes
      const tensors = await this.getAllFromObjectStore('tensors') as TensorStorage[];
      this.totalSize = tensors.reduce((total, tensor) => total + tensor.byteSize, 0);
    } catch (error) {
      this.log(`Failed to update storage stats: ${error.message}`);
    }
  }
  
  /**
   * Ensure the storage manager is initialized
   */
  private async ensureInitialized(): Promise<boolean> {
    if (this.isInitialized) {
      return true;
    }
    
    return await this.initialize();
  }
  
  /**
   * Log message if logging is enabled
   */
  private log(message: string): void {
    if (this.options.enableLogging) {
      console.log(`[StorageManager] ${message}`);
    }
  }
  
  // IndexedDB helper methods
  
  /**
   * Put an item in an object store
   */
  private putInObjectStore(storeName: string, item: any): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }
      
      try {
        const transaction = this.db.transaction(storeName, 'readwrite');
        const store = transaction.objectStore(storeName);
        const request = store.put(item);
        
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      } catch (error) {
        reject(error);
      }
    });
  }
  
  /**
   * Get an item from an object store
   */
  private getFromObjectStore(storeName: string, key: string): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }
      
      try {
        const transaction = this.db.transaction(storeName, 'readonly');
        const store = transaction.objectStore(storeName);
        const request = store.get(key);
        
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      } catch (error) {
        reject(error);
      }
    });
  }
  
  /**
   * Get all items from an object store
   */
  private getAllFromObjectStore(storeName: string): Promise<any[]> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }
      
      try {
        const transaction = this.db.transaction(storeName, 'readonly');
        const store = transaction.objectStore(storeName);
        const request = store.getAll();
        
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      } catch (error) {
        reject(error);
      }
    });
  }
  
  /**
   * Delete an item from an object store
   */
  private deleteFromObjectStore(storeName: string, key: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }
      
      try {
        const transaction = this.db.transaction(storeName, 'readwrite');
        const store = transaction.objectStore(storeName);
        const request = store.delete(key);
        
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      } catch (error) {
        reject(error);
      }
    });
  }
  
  /**
   * Clear an object store
   */
  private clearObjectStore(storeName: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }
      
      try {
        const transaction = this.db.transaction(storeName, 'readwrite');
        const store = transaction.objectStore(storeName);
        const request = store.clear();
        
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      } catch (error) {
        reject(error);
      }
    });
  }
  
  /**
   * Count items in an object store
   */
  private countObjectStore(storeName: string): Promise<number> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }
      
      try {
        const transaction = this.db.transaction(storeName, 'readonly');
        const store = transaction.objectStore(storeName);
        const request = store.count();
        
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      } catch (error) {
        reject(error);
      }
    });
  }
  
  /**
   * Run a query on an object store using an index
   */
  private runObjectStoreQuery<T>(
    storeName: string,
    indexName: string,
    range: IDBKeyRange | null,
    limit: number = Infinity
  ): Promise<T[]> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }
      
      try {
        const transaction = this.db.transaction(storeName, 'readonly');
        const store = transaction.objectStore(storeName);
        const index = store.index(indexName);
        const request = range ? index.openCursor(range) : index.openCursor();
        
        const results: T[] = [];
        
        request.onsuccess = (event) => {
          const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result;
          
          if (cursor && results.length < limit) {
            results.push(cursor.value);
            cursor.continue();
          } else {
            resolve(results);
          }
        };
        
        request.onerror = () => reject(request.error);
      } catch (error) {
        reject(error);
      }
    });
  }
}

// Default export for easier imports
export default StorageManager;