/**
 * IndexedDB-based storage manager for model weights and tensors
 * 
 * This file implements a storage manager for persisting model weights and tensors
 * in the browser's IndexedDB storage. It supports versioning, caching, and 
 * automatic cleanup of old or unused models.
 */

/**
 * Database structure
 */
interface ModelWeightDB {
  version: number;
  models: Record<string, ModelEntry>;
  lastAccessed: Record<string, number>;
  storageUsage: number;
}

/**
 * Model entry in database
 */
interface ModelEntry {
  modelId: string;
  name: string;
  version: string;
  weights: Record<string, WeightTensor>;
  metadata: Record<string, any>;
  size: number;
  createdAt: number;
  lastAccessed: number;
}

/**
 * Weight tensor stored in database
 */
interface WeightTensor {
  name: string;
  shape: number[];
  dataType: string;
  data: ArrayBuffer;
  size: number;
}

/**
 * Storage manager configuration
 */
export interface StorageManagerConfig {
  dbName?: string;
  dbVersion?: number;
  maxStorageSize?: number;
  cacheTTL?: number;
  compressionEnabled?: boolean;
  cleanupInterval?: number;
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: StorageManagerConfig = {
  dbName: 'ipfs_accelerate_model_storage',
  dbVersion: 1,
  maxStorageSize: 500 * 1024 * 1024, // 500MB default
  cacheTTL: 7 * 24 * 60 * 60 * 1000, // 7 days
  compressionEnabled: true,
  cleanupInterval: 60 * 60 * 1000, // 1 hour
};

/**
 * IndexedDB-based storage manager for model weights
 */
export class StorageManager {
  private db: IDBDatabase | null = null;
  private config: StorageManagerConfig;
  private cleanupIntervalId: number | null = null;
  private modelCache: Map<string, ModelEntry> = new Map();
  private initialized: boolean = false;
  private initPromise: Promise<void> | null = null;

  /**
   * Create a new storage manager
   */
  constructor(config: StorageManagerConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Initialize the storage manager
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = new Promise<void>((resolve, reject) => {
      // Check if IndexedDB is supported
      if (!window.indexedDB) {
        reject(new Error('IndexedDB is not supported in this browser'));
        return;
      }

      // Open database
      const request = window.indexedDB.open(
        this.config.dbName!, 
        this.config.dbVersion!
      );

      // Create database schema if needed
      request.onupgradeneeded = (event) => {
        const db = request.result;
        
        // Create object stores if they don't exist
        if (!db.objectStoreNames.contains('models')) {
          db.createObjectStore('models', { keyPath: 'modelId' });
        }
        
        if (!db.objectStoreNames.contains('metadata')) {
          db.createObjectStore('metadata', { keyPath: 'key' });
        }
      };

      // Handle success
      request.onsuccess = (event) => {
        this.db = request.result;
        this.initialized = true;
        
        // Initialize metadata if needed
        this.initializeMetadata().then(() => {
          // Start cleanup interval
          if (this.config.cleanupInterval && this.config.cleanupInterval > 0) {
            this.cleanupIntervalId = window.setInterval(
              () => this.runCleanup(),
              this.config.cleanupInterval
            );
          }
          
          resolve();
        }).catch(reject);
      };

      // Handle error
      request.onerror = (event) => {
        reject(new Error(`Failed to open database: ${request.error?.message}`));
      };
    });

    return this.initPromise;
  }

  /**
   * Initialize metadata if needed
   */
  private async initializeMetadata(): Promise<void> {
    const metadata = await this.getMetadata('dbInfo');
    
    if (!metadata) {
      // Initialize metadata
      await this.setMetadata('dbInfo', {
        createdAt: Date.now(),
        storageUsage: 0,
        modelCount: 0,
        lastCleanup: Date.now(),
      });
    }
  }

  /**
   * Get metadata from the database
   */
  private getMetadata(key: string): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction(['metadata'], 'readonly');
      const store = transaction.objectStore('metadata');
      const request = store.get(key);

      request.onsuccess = () => {
        resolve(request.result?.value || null);
      };

      request.onerror = () => {
        reject(new Error(`Failed to get metadata: ${request.error?.message}`));
      };
    });
  }

  /**
   * Set metadata in the database
   */
  private setMetadata(key: string, value: any): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction(['metadata'], 'readwrite');
      const store = transaction.objectStore('metadata');
      const request = store.put({ key, value });

      request.onsuccess = () => {
        resolve();
      };

      request.onerror = () => {
        reject(new Error(`Failed to set metadata: ${request.error?.message}`));
      };
    });
  }

  /**
   * Store a model in the database
   */
  async storeModel(
    modelId: string,
    name: string,
    version: string,
    weights: Record<string, WeightTensor>,
    metadata: Record<string, any> = {}
  ): Promise<void> {
    await this.initialize();

    // Calculate total size
    let totalSize = 0;
    for (const key in weights) {
      totalSize += weights[key].size;
    }

    // Check if we have enough space
    const dbInfo = await this.getMetadata('dbInfo');
    const currentUsage = dbInfo.storageUsage || 0;
    
    if (currentUsage + totalSize > this.config.maxStorageSize!) {
      // Need to free up space
      await this.freeUpSpace(totalSize);
    }

    // Create model entry
    const modelEntry: ModelEntry = {
      modelId,
      name,
      version,
      weights,
      metadata,
      size: totalSize,
      createdAt: Date.now(),
      lastAccessed: Date.now(),
    };

    // Store in database
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction(['models'], 'readwrite');
      const store = transaction.objectStore('models');
      const request = store.put(modelEntry);

      request.onsuccess = async () => {
        // Update metadata
        const dbInfo = await this.getMetadata('dbInfo');
        await this.setMetadata('dbInfo', {
          ...dbInfo,
          storageUsage: (dbInfo.storageUsage || 0) + totalSize,
          modelCount: (dbInfo.modelCount || 0) + 1,
        });

        // Update cache
        this.modelCache.set(modelId, modelEntry);
        
        resolve();
      };

      request.onerror = () => {
        reject(new Error(`Failed to store model: ${request.error?.message}`));
      };
    });
  }

  /**
   * Load a model from the database
   */
  async loadModel(modelId: string): Promise<ModelEntry | null> {
    await this.initialize();

    // Check cache first
    if (this.modelCache.has(modelId)) {
      const cachedModel = this.modelCache.get(modelId)!;
      
      // Update last accessed
      cachedModel.lastAccessed = Date.now();
      this.updateModelLastAccessed(modelId, Date.now());
      
      return cachedModel;
    }

    // Load from database
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction(['models'], 'readwrite');
      const store = transaction.objectStore('models');
      const request = store.get(modelId);

      request.onsuccess = () => {
        const model = request.result as ModelEntry | undefined;
        
        if (model) {
          // Update last accessed
          model.lastAccessed = Date.now();
          this.updateModelLastAccessed(modelId, Date.now());
          
          // Add to cache
          this.modelCache.set(modelId, model);
        }
        
        resolve(model || null);
      };

      request.onerror = () => {
        reject(new Error(`Failed to load model: ${request.error?.message}`));
      };
    });
  }

  /**
   * Update model last accessed time
   */
  private updateModelLastAccessed(modelId: string, timestamp: number): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction(['models'], 'readwrite');
      const store = transaction.objectStore('models');
      const request = store.get(modelId);

      request.onsuccess = () => {
        const model = request.result as ModelEntry | undefined;
        
        if (model) {
          model.lastAccessed = timestamp;
          store.put(model);
        }
        
        resolve();
      };

      request.onerror = () => {
        reject(new Error(`Failed to update model: ${request.error?.message}`));
      };
    });
  }

  /**
   * Delete a model from the database
   */
  async deleteModel(modelId: string): Promise<void> {
    await this.initialize();

    // Check if model exists
    const model = await this.loadModel(modelId);
    
    if (!model) {
      return; // Model doesn't exist
    }

    // Remove from cache
    this.modelCache.delete(modelId);

    // Delete from database
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction(['models'], 'readwrite');
      const store = transaction.objectStore('models');
      const request = store.delete(modelId);

      request.onsuccess = async () => {
        // Update metadata
        const dbInfo = await this.getMetadata('dbInfo');
        await this.setMetadata('dbInfo', {
          ...dbInfo,
          storageUsage: Math.max(0, (dbInfo.storageUsage || 0) - model.size),
          modelCount: Math.max(0, (dbInfo.modelCount || 0) - 1),
        });
        
        resolve();
      };

      request.onerror = () => {
        reject(new Error(`Failed to delete model: ${request.error?.message}`));
      };
    });
  }

  /**
   * List all models in the database
   */
  async listModels(): Promise<Array<{ modelId: string, name: string, version: string, size: number, lastAccessed: number }>> {
    await this.initialize();

    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction(['models'], 'readonly');
      const store = transaction.objectStore('models');
      const request = store.getAll();

      request.onsuccess = () => {
        const models = request.result as ModelEntry[];
        
        resolve(models.map(model => ({
          modelId: model.modelId,
          name: model.name,
          version: model.version,
          size: model.size,
          lastAccessed: model.lastAccessed,
        })));
      };

      request.onerror = () => {
        reject(new Error(`Failed to list models: ${request.error?.message}`));
      };
    });
  }

  /**
   * Free up space by deleting old models
   */
  private async freeUpSpace(requiredSpace: number): Promise<void> {
    const models = await this.listModels();
    
    // Sort by last accessed time (oldest first)
    models.sort((a, b) => a.lastAccessed - b.lastAccessed);
    
    let freedSpace = 0;
    
    // Delete models until we have enough space
    for (const model of models) {
      if (freedSpace >= requiredSpace) {
        break;
      }
      
      await this.deleteModel(model.modelId);
      freedSpace += model.size;
    }
    
    // Check if we freed enough space
    if (freedSpace < requiredSpace) {
      throw new Error(`Not enough space available. Required: ${requiredSpace}, freed: ${freedSpace}`);
    }
  }

  /**
   * Run cleanup to remove old models
   */
  private async runCleanup(): Promise<void> {
    const now = Date.now();
    const models = await this.listModels();
    
    // Find models older than TTL
    const oldModels = models.filter(model => 
      now - model.lastAccessed > this.config.cacheTTL!
    );
    
    // Delete old models
    for (const model of oldModels) {
      await this.deleteModel(model.modelId);
    }
    
    // Update last cleanup time
    const dbInfo = await this.getMetadata('dbInfo');
    await this.setMetadata('dbInfo', {
      ...dbInfo,
      lastCleanup: now,
    });
  }

  /**
   * Get storage usage information
   */
  async getStorageInfo(): Promise<{
    used: number,
    total: number,
    modelCount: number,
    lastCleanup: number
  }> {
    await this.initialize();
    
    const dbInfo = await this.getMetadata('dbInfo');
    
    return {
      used: dbInfo.storageUsage || 0,
      total: this.config.maxStorageSize!,
      modelCount: dbInfo.modelCount || 0,
      lastCleanup: dbInfo.lastCleanup || 0,
    };
  }

  /**
   * Close the database connection
   */
  dispose(): void {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
    
    if (this.cleanupIntervalId !== null) {
      window.clearInterval(this.cleanupIntervalId);
      this.cleanupIntervalId = null;
    }
    
    this.modelCache.clear();
    this.initialized = false;
    this.initPromise = null;
  }
}

/**
 * Create a new storage manager instance
 */
export function createStorageManager(config: StorageManagerConfig = {}): StorageManager {
  return new StorageManager(config);
}
