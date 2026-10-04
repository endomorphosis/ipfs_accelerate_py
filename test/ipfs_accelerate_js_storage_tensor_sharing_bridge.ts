/**
 * Storage Tensor Sharing Bridge
 * 
 * This component bridges the gap between the storage manager and the cross-model tensor sharing system.
 * It enables efficient sharing of tensors between multiple models while leveraging the storage manager
 * for persistent storage and caching.
 * 
 * Key features:
 * - Reference counting for shared tensors
 * - Tensor reuse between models
 * - Intelligent cache management
 * - Automatic garbage collection
 * - Storage-backed tensor sharing
 */

import StorageManager, { ModelMetadata, StorageDataType, TensorStorage } from './ipfs_accelerate_js_storage_manager';
import { WebNNStorageIntegration } from './ipfs_accelerate_js_webnn_storage_integration';

/**
 * Tensor sharing metadata
 */
export interface TensorSharingMetadata {
  /**
   * List of model IDs that share this tensor
   */
  sharedBy: string[];
  
  /**
   * Original model ID that created the tensor
   */
  originalModel: string;
  
  /**
   * Sharing type (e.g., 'text_embedding', 'vision_embedding', etc.)
   */
  sharingType: TensorSharingType;
  
  /**
   * Additional sharing information
   */
  sharingInfo?: Record<string, any>;
  
  /**
   * Reference count
   */
  refCount: number;
}

/**
 * Tensor sharing information
 */
export interface TensorSharingInfo {
  /**
   * Tensor ID
   */
  tensorId: string;
  
  /**
   * Tensor sharing type
   */
  sharingType: TensorSharingType;
  
  /**
   * Models that share this tensor
   */
  sharedBy: string[];
  
  /**
   * Original model ID
   */
  originalModel: string;
  
  /**
   * Sharing timestamp
   */
  sharedAt: number;
  
  /**
   * Reference count
   */
  refCount: number;
  
  /**
   * Tensor byte size
   */
  byteSize: number;
  
  /**
   * Tensor shape
   */
  shape: number[];
  
  /**
   * Tensor data type
   */
  dataType: StorageDataType;
}

/**
 * Types of tensor sharing
 */
export type TensorSharingType = 
  'text_embedding' | 
  'vision_embedding' | 
  'audio_embedding' | 
  'vision_text_joint' | 
  'audio_text_joint' | 
  'custom';

/**
 * Compatibility registry for tensor sharing
 */
export const TENSOR_SHARING_COMPATIBILITY: Record<TensorSharingType, string[]> = {
  'text_embedding': ['bert', 't5', 'llama', 'bart'],
  'vision_embedding': ['vit', 'clip', 'detr'],
  'audio_embedding': ['whisper', 'wav2vec2', 'clap'],
  'vision_text_joint': ['clip', 'llava', 'blip'],
  'audio_text_joint': ['clap', 'whisper-text'],
  'custom': []
};

/**
 * Storage Tensor Sharing Bridge options
 */
export interface StorageTensorSharingBridgeOptions {
  /**
   * Enable logging
   */
  enableLogging?: boolean;
  
  /**
   * Database name
   */
  dbName?: string;
  
  /**
   * Enable automatic garbage collection
   */
  enableAutoGC?: boolean;
  
  /**
   * Garbage collection interval (in milliseconds)
   */
  gcInterval?: number;
  
  /**
   * Maximum cache size (in bytes)
   */
  maxCacheSize?: number;
  
  /**
   * Enable tensor compression
   */
  enableCompression?: boolean;
}

/**
 * Storage Tensor Sharing Bridge
 * Enables efficient sharing of tensors between multiple models with storage backing
 */
export class StorageTensorSharingBridge {
  private storageManager: StorageManager;
  private webnnIntegration?: WebNNStorageIntegration;
  private options: Required<StorageTensorSharingBridgeOptions>;
  
  // In-memory reference counting
  private sharedTensors: Map<string, TensorSharingMetadata> = new Map();
  private modelTensorMap: Map<string, Set<string>> = new Map();
  private sharingTypeMap: Map<TensorSharingType, Set<string>> = new Map();
  
  // Cache for loaded tensors
  private tensorCache: Map<string, {
    data: Float32Array | Int32Array | Uint8Array,
    lastAccessed: number,
    size: number
  }> = new Map();
  
  private cacheSize = 0;
  private isInitialized = false;
  private gcTimer: any = null;
  
  /**
   * Creates a new storage tensor sharing bridge
   */
  constructor(
    options: StorageTensorSharingBridgeOptions = {},
    webnnIntegration?: WebNNStorageIntegration
  ) {
    // Set default options
    this.options = {
      enableLogging: options.enableLogging ?? false,
      dbName: options.dbName ?? 'tensor-sharing-bridge',
      enableAutoGC: options.enableAutoGC ?? true,
      gcInterval: options.gcInterval ?? 1000 * 60 * 10, // 10 minutes
      maxCacheSize: options.maxCacheSize ?? 1024 * 1024 * 512, // 512MB
      enableCompression: options.enableCompression ?? false
    };
    
    // Create storage manager
    this.storageManager = new StorageManager({
      dbName: this.options.dbName,
      enableCompression: this.options.enableCompression,
      enableLogging: this.options.enableLogging
    });
    
    // Store WebNN integration if provided
    this.webnnIntegration = webnnIntegration;
    
    // Initialize sharing type map
    for (const sharingType of Object.keys(TENSOR_SHARING_COMPATIBILITY) as TensorSharingType[]) {
      this.sharingTypeMap.set(sharingType, new Set());
    }
  }
  
  /**
   * Initialize the tensor sharing bridge
   */
  async initialize(): Promise<boolean> {
    if (this.isInitialized) {
      return true;
    }
    
    try {
      // Initialize storage manager
      const initialized = await this.storageManager.initialize();
      
      if (!initialized) {
        this.log('Failed to initialize storage manager');
        return false;
      }
      
      // Load shared tensor metadata from storage
      await this.loadSharedTensorsMetadata();
      
      // Start garbage collection timer if enabled
      if (this.options.enableAutoGC) {
        this.startGarbageCollection();
      }
      
      this.isInitialized = true;
      this.log('Storage tensor sharing bridge initialized successfully');
      return true;
    } catch (error) {
      this.log(`Initialization error: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Register a tensor for sharing
   */
  async registerSharedTensor(
    tensorId: string,
    modelId: string,
    sharingType: TensorSharingType,
    data: Float32Array | Int32Array | Uint8Array,
    shape: number[],
    dataType: StorageDataType,
    metadata: Record<string, any> = {}
  ): Promise<boolean> {
    if (!await this.ensureInitialized()) {
      return false;
    }
    
    try {
      // Check if tensor already exists
      const existingTensor = await this.storageManager.getTensor(tensorId);
      if (existingTensor) {
        // Update sharing metadata
        return await this.updateSharingMetadata(tensorId, modelId, sharingType);
      }
      
      // Store tensor in storage manager
      const stored = await this.storageManager.storeTensor(
        tensorId,
        data,
        shape,
        dataType,
        {
          ...metadata,
          sharing: {
            sharingType,
            originalModel: modelId,
            sharedBy: [modelId],
            refCount: 1
          }
        }
      );
      
      if (!stored) {
        this.log(`Failed to store shared tensor: ${tensorId}`);
        return false;
      }
      
      // Update in-memory metadata
      this.updateInMemorySharing(tensorId, modelId, sharingType, {
        originalModel: modelId,
        sharedBy: [modelId],
        refCount: 1,
        sharingType
      });
      
      this.log(`Registered shared tensor: ${tensorId} (type: ${sharingType}) for model: ${modelId}`);
      return true;
    } catch (error) {
      this.log(`Failed to register shared tensor: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Share a tensor with an additional model
   */
  async shareTensor(
    tensorId: string,
    sourceModelId: string,
    targetModelId: string
  ): Promise<boolean> {
    if (!await this.ensureInitialized()) {
      return false;
    }
    
    try {
      // Get tensor sharing metadata
      const sharingMetadata = this.sharedTensors.get(tensorId);
      
      if (!sharingMetadata) {
        this.log(`Tensor not found for sharing: ${tensorId}`);
        return false;
      }
      
      // Check if tensor is already shared with the target model
      if (sharingMetadata.sharedBy.includes(targetModelId)) {
        this.log(`Tensor ${tensorId} is already shared with model ${targetModelId}`);
        return true;
      }
      
      // Update sharing metadata
      sharingMetadata.sharedBy.push(targetModelId);
      sharingMetadata.refCount++;
      
      // Update tensor metadata in storage
      const tensor = await this.storageManager.getTensor(tensorId);
      
      if (!tensor) {
        this.log(`Tensor not found in storage: ${tensorId}`);
        return false;
      }
      
      // Update tensor metadata
      const updatedMetadata = {
        ...tensor.metadata,
        sharing: {
          ...sharingMetadata
        }
      };
      
      // Store updated tensor with new metadata
      const stored = await this.storageManager.storeTensor(
        tensorId,
        await this.storageManager.getTensorData(tensorId) as Float32Array | Int32Array | Uint8Array,
        tensor.shape,
        tensor.dataType,
        updatedMetadata
      );
      
      if (!stored) {
        this.log(`Failed to update shared tensor metadata: ${tensorId}`);
        return false;
      }
      
      // Update model-tensor map
      if (!this.modelTensorMap.has(targetModelId)) {
        this.modelTensorMap.set(targetModelId, new Set());
      }
      this.modelTensorMap.get(targetModelId)!.add(tensorId);
      
      this.log(`Shared tensor ${tensorId} from model ${sourceModelId} with model ${targetModelId}`);
      return true;
    } catch (error) {
      this.log(`Failed to share tensor: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Get a shared tensor
   */
  async getSharedTensor(
    tensorId: string,
    modelId: string
  ): Promise<{ data: Float32Array | Int32Array | Uint8Array, shape: number[], metadata: any } | null> {
    if (!await this.ensureInitialized()) {
      return null;
    }
    
    try {
      // Check if model has access to this tensor
      const sharingMetadata = this.sharedTensors.get(tensorId);
      
      if (!sharingMetadata || !sharingMetadata.sharedBy.includes(modelId)) {
        this.log(`Model ${modelId} does not have access to tensor ${tensorId}`);
        return null;
      }
      
      // Check if tensor is in cache
      if (this.tensorCache.has(tensorId)) {
        // Update last accessed time
        const cached = this.tensorCache.get(tensorId)!;
        cached.lastAccessed = Date.now();
        
        // Get tensor metadata
        const tensor = await this.storageManager.getTensor(tensorId);
        
        if (!tensor) {
          this.log(`Tensor metadata not found in storage: ${tensorId}`);
          return null;
        }
        
        return {
          data: cached.data,
          shape: tensor.shape,
          metadata: tensor.metadata
        };
      }
      
      // Get tensor from storage
      const tensorData = await this.storageManager.getTensorData(tensorId);
      
      if (!tensorData) {
        this.log(`Tensor data not found in storage: ${tensorId}`);
        return null;
      }
      
      // Get tensor metadata
      const tensor = await this.storageManager.getTensor(tensorId);
      
      if (!tensor) {
        this.log(`Tensor metadata not found in storage: ${tensorId}`);
        return null;
      }
      
      // Add to cache if there's space
      const byteSize = tensorData.byteLength;
      
      if (this.cacheSize + byteSize <= this.options.maxCacheSize) {
        this.tensorCache.set(tensorId, {
          data: tensorData,
          lastAccessed: Date.now(),
          size: byteSize
        });
        
        this.cacheSize += byteSize;
      } else {
        // Try to make space in the cache
        const freed = this.freeCacheSpace(byteSize);
        
        if (freed >= byteSize) {
          this.tensorCache.set(tensorId, {
            data: tensorData,
            lastAccessed: Date.now(),
            size: byteSize
          });
          
          this.cacheSize += byteSize;
        }
      }
      
      return {
        data: tensorData,
        shape: tensor.shape,
        metadata: tensor.metadata
      };
    } catch (error) {
      this.log(`Failed to get shared tensor: ${error.message}`);
      return null;
    }
  }
  
  /**
   * Find compatible shared tensors by sharing type
   */
  async findCompatibleTensors(
    modelId: string,
    sharingType: TensorSharingType,
    modelType?: string
  ): Promise<TensorSharingInfo[]> {
    if (!await this.ensureInitialized()) {
      return [];
    }
    
    try {
      const tensorIds = this.sharingTypeMap.get(sharingType) || new Set();
      const results: TensorSharingInfo[] = [];
      
      for (const tensorId of tensorIds) {
        const sharingMetadata = this.sharedTensors.get(tensorId);
        
        if (!sharingMetadata) {
          continue;
        }
        
        // Check model compatibility if modelType is provided
        if (modelType) {
          const compatibleModels = TENSOR_SHARING_COMPATIBILITY[sharingType];
          if (!compatibleModels.includes(modelType)) {
            continue;
          }
        }
        
        // Get tensor metadata
        const tensor = await this.storageManager.getTensor(tensorId);
        
        if (!tensor) {
          continue;
        }
        
        results.push({
          tensorId,
          sharingType,
          sharedBy: sharingMetadata.sharedBy,
          originalModel: sharingMetadata.originalModel,
          sharedAt: tensor.createdAt,
          refCount: sharingMetadata.refCount,
          byteSize: tensor.byteSize,
          shape: tensor.shape,
          dataType: tensor.dataType
        });
      }
      
      return results;
    } catch (error) {
      this.log(`Failed to find compatible tensors: ${error.message}`);
      return [];
    }
  }
  
  /**
   * Release a shared tensor
   */
  async releaseTensor(
    tensorId: string,
    modelId: string
  ): Promise<boolean> {
    if (!await this.ensureInitialized()) {
      return false;
    }
    
    try {
      // Get tensor sharing metadata
      const sharingMetadata = this.sharedTensors.get(tensorId);
      
      if (!sharingMetadata) {
        this.log(`Tensor not found for release: ${tensorId}`);
        return false;
      }
      
      // Check if model has access to this tensor
      if (!sharingMetadata.sharedBy.includes(modelId)) {
        this.log(`Model ${modelId} does not have access to tensor ${tensorId}`);
        return false;
      }
      
      // Update sharing metadata
      sharingMetadata.sharedBy = sharingMetadata.sharedBy.filter(id => id !== modelId);
      sharingMetadata.refCount--;
      
      // Update model-tensor map
      if (this.modelTensorMap.has(modelId)) {
        this.modelTensorMap.get(modelId)!.delete(tensorId);
      }
      
      // If no more references, remove tensor
      if (sharingMetadata.refCount <= 0) {
        this.sharedTensors.delete(tensorId);
        this.sharingTypeMap.get(sharingMetadata.sharingType)?.delete(tensorId);
        
        // Remove from cache
        if (this.tensorCache.has(tensorId)) {
          const cached = this.tensorCache.get(tensorId)!;
          this.cacheSize -= cached.size;
          this.tensorCache.delete(tensorId);
        }
        
        // Delete from storage
        await this.storageManager.deleteTensor(tensorId);
        
        this.log(`Removed shared tensor with no references: ${tensorId}`);
      } else {
        // Update tensor metadata in storage
        const tensor = await this.storageManager.getTensor(tensorId);
        
        if (tensor) {
          // Update tensor metadata
          const updatedMetadata = {
            ...tensor.metadata,
            sharing: {
              ...sharingMetadata
            }
          };
          
          // Store updated tensor with new metadata
          await this.storageManager.storeTensor(
            tensorId,
            await this.storageManager.getTensorData(tensorId) as Float32Array | Int32Array | Uint8Array,
            tensor.shape,
            tensor.dataType,
            updatedMetadata
          );
        }
      }
      
      this.log(`Released tensor ${tensorId} from model ${modelId}`);
      return true;
    } catch (error) {
      this.log(`Failed to release tensor: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Get all tensors shared by a model
   */
  async getModelSharedTensors(
    modelId: string
  ): Promise<TensorSharingInfo[]> {
    if (!await this.ensureInitialized()) {
      return [];
    }
    
    try {
      const results: TensorSharingInfo[] = [];
      
      // Get tensor IDs from model-tensor map
      const tensorIds = this.modelTensorMap.get(modelId) || new Set();
      
      for (const tensorId of tensorIds) {
        const sharingMetadata = this.sharedTensors.get(tensorId);
        
        if (!sharingMetadata) {
          continue;
        }
        
        // Get tensor metadata
        const tensor = await this.storageManager.getTensor(tensorId);
        
        if (!tensor) {
          continue;
        }
        
        results.push({
          tensorId,
          sharingType: sharingMetadata.sharingType,
          sharedBy: sharingMetadata.sharedBy,
          originalModel: sharingMetadata.originalModel,
          sharedAt: tensor.createdAt,
          refCount: sharingMetadata.refCount,
          byteSize: tensor.byteSize,
          shape: tensor.shape,
          dataType: tensor.dataType
        });
      }
      
      return results;
    } catch (error) {
      this.log(`Failed to get model shared tensors: ${error.message}`);
      return [];
    }
  }
  
  /**
   * Get sharing statistics
   */
  async getSharingStats(): Promise<{
    totalSharedTensors: number;
    totalRefCount: number;
    bytesSaved: number;
    sharingByType: Record<TensorSharingType, number>;
  }> {
    if (!await this.ensureInitialized()) {
      return {
        totalSharedTensors: 0,
        totalRefCount: 0,
        bytesSaved: 0,
        sharingByType: {} as Record<TensorSharingType, number>
      };
    }
    
    try {
      let totalSharedTensors = 0;
      let totalRefCount = 0;
      let bytesSaved = 0;
      const sharingByType: Record<TensorSharingType, number> = {} as Record<TensorSharingType, number>;
      
      // Initialize sharing by type
      for (const sharingType of Object.keys(TENSOR_SHARING_COMPATIBILITY) as TensorSharingType[]) {
        sharingByType[sharingType] = 0;
      }
      
      // Calculate statistics
      for (const [tensorId, metadata] of this.sharedTensors.entries()) {
        totalSharedTensors++;
        totalRefCount += metadata.refCount;
        
        // Get tensor size
        const tensor = await this.storageManager.getTensor(tensorId);
        
        if (tensor) {
          // Calculate bytes saved (size * (refCount - 1))
          // refCount - 1 because the first reference doesn't save anything
          bytesSaved += tensor.byteSize * (metadata.refCount - 1);
        }
        
        // Update sharing by type
        sharingByType[metadata.sharingType] = (sharingByType[metadata.sharingType] || 0) + 1;
      }
      
      return {
        totalSharedTensors,
        totalRefCount,
        bytesSaved,
        sharingByType
      };
    } catch (error) {
      this.log(`Failed to get sharing stats: ${error.message}`);
      return {
        totalSharedTensors: 0,
        totalRefCount: 0,
        bytesSaved: 0,
        sharingByType: {} as Record<TensorSharingType, number>
      };
    }
  }
  
  /**
   * Clear all shared tensors
   */
  async clearSharedTensors(): Promise<boolean> {
    if (!await this.ensureInitialized()) {
      return false;
    }
    
    try {
      // Clear in-memory data
      this.sharedTensors.clear();
      this.modelTensorMap.clear();
      
      for (const sharingType of Object.keys(TENSOR_SHARING_COMPATIBILITY) as TensorSharingType[]) {
        this.sharingTypeMap.set(sharingType, new Set());
      }
      
      // Clear tensor cache
      this.tensorCache.clear();
      this.cacheSize = 0;
      
      // Clear tensors from storage
      const tensors = await this.storageManager.getAllFromObjectStore('tensors') as TensorStorage[];
      
      for (const tensor of tensors) {
        if (tensor.metadata?.sharing) {
          await this.storageManager.deleteTensor(tensor.id);
        }
      }
      
      this.log('Cleared all shared tensors');
      return true;
    } catch (error) {
      this.log(`Failed to clear shared tensors: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Run garbage collection
   */
  async garbageCollect(): Promise<{
    tensorsRemoved: number;
    bytesFreed: number;
  }> {
    if (!await this.ensureInitialized()) {
      return { tensorsRemoved: 0, bytesFreed: 0 };
    }
    
    try {
      let tensorsRemoved = 0;
      let bytesFreed = 0;
      
      // Find tensors with no references
      const tensorsToRemove: string[] = [];
      
      for (const [tensorId, metadata] of this.sharedTensors.entries()) {
        if (metadata.refCount <= 0 || metadata.sharedBy.length === 0) {
          tensorsToRemove.push(tensorId);
        }
      }
      
      // Remove tensors
      for (const tensorId of tensorsToRemove) {
        const sharingMetadata = this.sharedTensors.get(tensorId);
        
        if (!sharingMetadata) {
          continue;
        }
        
        // Get tensor size
        const tensor = await this.storageManager.getTensor(tensorId);
        
        if (tensor) {
          bytesFreed += tensor.byteSize;
        }
        
        // Remove from in-memory data
        this.sharedTensors.delete(tensorId);
        this.sharingTypeMap.get(sharingMetadata.sharingType)?.delete(tensorId);
        
        // Remove from cache
        if (this.tensorCache.has(tensorId)) {
          const cached = this.tensorCache.get(tensorId)!;
          this.cacheSize -= cached.size;
          this.tensorCache.delete(tensorId);
        }
        
        // Delete from storage
        await this.storageManager.deleteTensor(tensorId);
        
        tensorsRemoved++;
      }
      
      // Clean up cache
      const cacheFreed = await this.cleanCache();
      bytesFreed += cacheFreed;
      
      this.log(`Garbage collection: removed ${tensorsRemoved} tensors, freed ${bytesFreed} bytes`);
      return { tensorsRemoved, bytesFreed };
    } catch (error) {
      this.log(`Garbage collection error: ${error.message}`);
      return { tensorsRemoved: 0, bytesFreed: 0 };
    }
  }
  
  // Private helper methods
  
  /**
   * Ensure the bridge is initialized
   */
  private async ensureInitialized(): Promise<boolean> {
    if (this.isInitialized) {
      return true;
    }
    
    return await this.initialize();
  }
  
  /**
   * Load shared tensors metadata from storage
   */
  private async loadSharedTensorsMetadata(): Promise<void> {
    try {
      // Get all tensors from storage
      const tensors = await this.storageManager.getAllFromObjectStore('tensors') as TensorStorage[];
      
      for (const tensor of tensors) {
        const sharingMetadata = tensor.metadata?.sharing as TensorSharingMetadata | undefined;
        
        if (sharingMetadata) {
          const { sharingType, sharedBy, originalModel, refCount } = sharingMetadata;
          
          // Update in-memory metadata
          this.updateInMemorySharing(tensor.id, originalModel, sharingType, sharingMetadata);
        }
      }
      
      this.log(`Loaded metadata for ${this.sharedTensors.size} shared tensors`);
    } catch (error) {
      this.log(`Failed to load shared tensors metadata: ${error.message}`);
    }
  }
  
  /**
   * Update sharing metadata for a tensor
   */
  private async updateSharingMetadata(
    tensorId: string,
    modelId: string,
    sharingType: TensorSharingType
  ): Promise<boolean> {
    try {
      // Get tensor
      const tensor = await this.storageManager.getTensor(tensorId);
      
      if (!tensor) {
        return false;
      }
      
      // Get current sharing metadata
      const sharingMetadata = tensor.metadata?.sharing as TensorSharingMetadata | undefined;
      
      if (sharingMetadata) {
        // Check if model is already in the list
        if (sharingMetadata.sharedBy.includes(modelId)) {
          return true;
        }
        
        // Update sharing metadata
        const updatedMetadata = {
          ...sharingMetadata,
          sharedBy: [...sharingMetadata.sharedBy, modelId],
          refCount: sharingMetadata.refCount + 1
        };
        
        // Update tensor metadata
        const updatedTensorMetadata = {
          ...tensor.metadata,
          sharing: updatedMetadata
        };
        
        // Store updated tensor with new metadata
        const stored = await this.storageManager.storeTensor(
          tensorId,
          await this.storageManager.getTensorData(tensorId) as Float32Array | Int32Array | Uint8Array,
          tensor.shape,
          tensor.dataType,
          updatedTensorMetadata
        );
        
        if (stored) {
          // Update in-memory metadata
          this.updateInMemorySharing(tensorId, modelId, sharingType, updatedMetadata);
          return true;
        }
      } else {
        // Create new sharing metadata
        const newMetadata = {
          sharingType,
          originalModel: modelId,
          sharedBy: [modelId],
          refCount: 1
        };
        
        // Update tensor metadata
        const updatedTensorMetadata = {
          ...tensor.metadata,
          sharing: newMetadata
        };
        
        // Store updated tensor with new metadata
        const stored = await this.storageManager.storeTensor(
          tensorId,
          await this.storageManager.getTensorData(tensorId) as Float32Array | Int32Array | Uint8Array,
          tensor.shape,
          tensor.dataType,
          updatedTensorMetadata
        );
        
        if (stored) {
          // Update in-memory metadata
          this.updateInMemorySharing(tensorId, modelId, sharingType, newMetadata);
          return true;
        }
      }
      
      return false;
    } catch (error) {
      this.log(`Failed to update sharing metadata: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Update in-memory sharing metadata
   */
  private updateInMemorySharing(
    tensorId: string,
    modelId: string,
    sharingType: TensorSharingType,
    metadata: TensorSharingMetadata
  ): void {
    // Update shared tensors map
    this.sharedTensors.set(tensorId, metadata);
    
    // Update sharing type map
    if (!this.sharingTypeMap.has(sharingType)) {
      this.sharingTypeMap.set(sharingType, new Set());
    }
    this.sharingTypeMap.get(sharingType)!.add(tensorId);
    
    // Update model-tensor map for each model that shares this tensor
    for (const sharedBy of metadata.sharedBy) {
      if (!this.modelTensorMap.has(sharedBy)) {
        this.modelTensorMap.set(sharedBy, new Set());
      }
      this.modelTensorMap.get(sharedBy)!.add(tensorId);
    }
  }
  
  /**
   * Start garbage collection timer
   */
  private startGarbageCollection(): void {
    if (this.gcTimer) {
      clearInterval(this.gcTimer);
    }
    
    this.gcTimer = setInterval(() => {
      this.garbageCollect().catch(error => {
        this.log(`Automatic garbage collection error: ${error.message}`);
      });
    }, this.options.gcInterval);
  }
  
  /**
   * Clean up the tensor cache
   */
  private async cleanCache(): Promise<number> {
    if (this.cacheSize <= this.options.maxCacheSize * 0.8) {
      return 0; // No cleanup needed
    }
    
    // Sort cache entries by last accessed time
    const entries = Array.from(this.tensorCache.entries())
      .map(([id, info]) => ({ id, ...info }))
      .sort((a, b) => a.lastAccessed - b.lastAccessed);
    
    let bytesFreed = 0;
    const targetSize = this.options.maxCacheSize * 0.6; // Target 60% of max cache size
    
    // Remove oldest entries until we're below target size
    for (const entry of entries) {
      if (this.cacheSize <= targetSize) {
        break;
      }
      
      this.tensorCache.delete(entry.id);
      this.cacheSize -= entry.size;
      bytesFreed += entry.size;
    }
    
    return bytesFreed;
  }
  
  /**
   * Free up space in the cache for a new tensor
   */
  private freeCacheSpace(requiredBytes: number): number {
    if (this.cacheSize + requiredBytes <= this.options.maxCacheSize) {
      return requiredBytes;
    }
    
    // Sort cache entries by last accessed time
    const entries = Array.from(this.tensorCache.entries())
      .map(([id, info]) => ({ id, ...info }))
      .sort((a, b) => a.lastAccessed - b.lastAccessed);
    
    let bytesFreed = 0;
    
    // Remove oldest entries until we have enough space
    for (const entry of entries) {
      if (bytesFreed >= requiredBytes) {
        break;
      }
      
      this.tensorCache.delete(entry.id);
      this.cacheSize -= entry.size;
      bytesFreed += entry.size;
    }
    
    return bytesFreed;
  }
  
  /**
   * Get all objects from an object store (private method for system use)
   */
  private async getAllFromObjectStore(storeName: string): Promise<any[]> {
    return await this.storageManager.getAllFromObjectStore(storeName);
  }
  
  /**
   * Log message if logging is enabled
   */
  private log(message: string): void {
    if (this.options.enableLogging) {
      console.log(`[StorageTensorSharingBridge] ${message}`);
    }
  }
}

// Default export for easier imports
export default StorageTensorSharingBridge;