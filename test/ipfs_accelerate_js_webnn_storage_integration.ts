/**
 * WebNN Backend Storage Integration
 * Integrates the WebNN backend with the storage manager
 */

import { WebNNBackend } from './ipfs_accelerate_js_webnn_backend';
import StorageManager, { ModelMetadata, StorageDataType } from './ipfs_accelerate_js_storage_manager';

/**
 * WebNN Storage Integration Options
 */
export interface WebNNStorageIntegrationOptions {
  /**
   * Enable model weights caching
   */
  enableModelCaching?: boolean;
  
  /**
   * Enable automatic cleanup of unused models
   */
  enableAutoCleanup?: boolean;
  
  /**
   * Threshold for cleanup (in milliseconds)
   */
  cleanupThreshold?: number;
  
  /**
   * Maximum storage size (in bytes)
   */
  maxStorageSize?: number;
  
  /**
   * Enable compression for stored tensors
   */
  enableCompression?: boolean;
  
  /**
   * Enable logging
   */
  enableLogging?: boolean;
  
  /**
   * Database name
   */
  dbName?: string;
}

/**
 * WebNN Storage Integration
 * Integrates WebNN backend with the storage manager for efficient model loading and caching
 */
export class WebNNStorageIntegration {
  private backend: WebNNBackend;
  private storageManager: StorageManager;
  private options: Required<WebNNStorageIntegrationOptions>;
  private modelCache: Map<string, Set<string>> = new Map();
  
  /**
   * Creates a new WebNN Storage Integration instance
   */
  constructor(
    backend: WebNNBackend, 
    options: WebNNStorageIntegrationOptions = {}
  ) {
    this.backend = backend;
    
    // Set default options
    this.options = {
      enableModelCaching: options.enableModelCaching ?? true,
      enableAutoCleanup: options.enableAutoCleanup ?? true,
      cleanupThreshold: options.cleanupThreshold ?? 1000 * 60 * 60 * 24 * 7, // 7 days
      maxStorageSize: options.maxStorageSize ?? 1024 * 1024 * 1024, // 1GB
      enableCompression: options.enableCompression ?? false,
      enableLogging: options.enableLogging ?? false,
      dbName: options.dbName ?? 'webnn-model-cache',
    };
    
    // Create storage manager
    this.storageManager = new StorageManager({
      dbName: this.options.dbName,
      enableCompression: this.options.enableCompression,
      maxStorageSize: this.options.maxStorageSize,
      enableAutoCleanup: this.options.enableAutoCleanup,
      cleanupThreshold: this.options.cleanupThreshold,
      enableLogging: this.options.enableLogging,
    });
  }
  
  /**
   * Initialize the storage integration
   */
  async initialize(): Promise<boolean> {
    try {
      // Initialize the storage manager
      const storageInitialized = await this.storageManager.initialize();
      
      if (!storageInitialized) {
        this.log('Failed to initialize storage manager');
        return false;
      }
      
      // Initialize the WebNN backend
      const backendInitialized = await this.backend.initialize();
      
      if (!backendInitialized) {
        this.log('Failed to initialize WebNN backend');
        return false;
      }
      
      this.log('WebNN Storage Integration initialized successfully');
      return true;
    } catch (error) {
      this.log(`Initialization error: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Store a model in the cache
   */
  async storeModel(
    modelId: string,
    modelName: string,
    weights: Map<string, { data: Float32Array | Int32Array | Uint8Array; shape: number[]; dataType: StorageDataType }>,
    metadata: Record<string, any> = {}
  ): Promise<boolean> {
    try {
      // Create list of tensor IDs
      const tensorIds: string[] = [];
      
      // Store each tensor
      for (const [name, tensor] of weights.entries()) {
        // Create a unique tensor ID
        const tensorId = `${modelId}_${name}`;
        tensorIds.push(tensorId);
        
        // Store tensor data
        const success = await this.storageManager.storeTensor(
          tensorId,
          tensor.data,
          tensor.shape,
          tensor.dataType,
          { name, ...metadata }
        );
        
        if (!success) {
          this.log(`Failed to store tensor: ${tensorId}`);
          return false;
        }
      }
      
      // Calculate total size
      let totalSize = 0;
      for (const [, tensor] of weights.entries()) {
        totalSize += tensor.data.byteLength;
      }
      
      // Store model metadata
      const modelMetadata: Omit<ModelMetadata, 'createdAt' | 'lastAccessed'> = {
        id: modelId,
        name: modelName,
        version: metadata.version || '1.0.0',
        framework: metadata.framework,
        totalSize,
        tensorIds,
        metadata,
      };
      
      const success = await this.storageManager.storeModelMetadata(modelMetadata);
      
      if (!success) {
        this.log(`Failed to store model metadata: ${modelId}`);
        return false;
      }
      
      // Update in-memory cache
      this.modelCache.set(modelId, new Set(tensorIds));
      
      this.log(`Stored model: ${modelId} (${modelName})`);
      return true;
    } catch (error) {
      this.log(`Failed to store model: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Load a model from the cache
   */
  async loadModel(
    modelId: string
  ): Promise<Map<string, { tensor: any; shape: number[] }> | null> {
    try {
      // Get model metadata
      const model = await this.storageManager.getModelMetadata(modelId);
      
      if (!model) {
        this.log(`Model not found: ${modelId}`);
        return null;
      }
      
      // Load each tensor
      const tensors = new Map<string, { tensor: any; shape: number[] }>();
      
      for (const tensorId of model.tensorIds) {
        // Get tensor data
        const tensorData = await this.storageManager.getTensorData(tensorId);
        
        if (!tensorData) {
          this.log(`Tensor not found: ${tensorId}`);
          continue;
        }
        
        // Get tensor metadata
        const tensor = await this.storageManager.getTensor(tensorId);
        
        if (!tensor) {
          this.log(`Tensor metadata not found: ${tensorId}`);
          continue;
        }
        
        // Extract name from tensor ID (remove model ID prefix)
        const name = tensor.metadata?.name || tensorId.replace(`${modelId}_`, '');
        
        // Create tensor in WebNN backend
        const webnnTensor = await this.backend.createTensor(
          tensorData,
          tensor.shape,
          tensor.dataType
        );
        
        tensors.set(name, { tensor: webnnTensor.tensor, shape: tensor.shape });
      }
      
      // Update in-memory cache
      this.modelCache.set(modelId, new Set(model.tensorIds));
      
      this.log(`Loaded model: ${modelId} (${model.name})`);
      return tensors;
    } catch (error) {
      this.log(`Failed to load model: ${error.message}`);
      return null;
    }
  }
  
  /**
   * Check if a model is available in the cache
   */
  async isModelCached(modelId: string): Promise<boolean> {
    // First check in-memory cache
    if (this.modelCache.has(modelId)) {
      return true;
    }
    
    // Then check storage
    const model = await this.storageManager.getModelMetadata(modelId);
    return !!model;
  }
  
  /**
   * Delete a model from the cache
   */
  async deleteModel(modelId: string): Promise<boolean> {
    const success = await this.storageManager.deleteModel(modelId);
    
    if (success) {
      // Remove from in-memory cache
      this.modelCache.delete(modelId);
      this.log(`Deleted model: ${modelId}`);
    }
    
    return success;
  }
  
  /**
   * Get storage statistics
   */
  async getStorageStats(): Promise<{
    modelCount: number;
    totalSize: number;
    remainingQuota?: number;
  }> {
    const info = await this.storageManager.getStorageInfo();
    
    return {
      modelCount: info.modelCount,
      totalSize: info.totalSize,
      remainingQuota: info.remainingQuota,
    };
  }
  
  /**
   * List all available models
   */
  async listModels(): Promise<{
    id: string;
    name: string;
    version: string;
    size: number;
    lastAccessed: number;
  }[]> {
    const models = await this.storageManager.listModels();
    
    return models.map(model => ({
      id: model.id,
      name: model.name,
      version: model.version,
      size: model.totalSize,
      lastAccessed: model.lastAccessed,
    }));
  }
  
  /**
   * Clear all cached models
   */
  async clearCache(): Promise<boolean> {
    const success = await this.storageManager.clear();
    
    if (success) {
      // Clear in-memory cache
      this.modelCache.clear();
      this.log('Cleared model cache');
    }
    
    return success;
  }
  
  /**
   * Log message if logging is enabled
   */
  private log(message: string): void {
    if (this.options.enableLogging) {
      console.log(`[WebNNStorageIntegration] ${message}`);
    }
  }
}

// Default export for easier imports
export default WebNNStorageIntegration;