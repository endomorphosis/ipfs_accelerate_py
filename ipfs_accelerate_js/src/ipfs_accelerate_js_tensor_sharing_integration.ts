/**
 * Cross-Model Tensor Sharing Integration with Storage Manager
 * Integrates the TensorSharingManager with the StorageManager for persistent shared tensors
 */

import { TensorSharingManager, SharedTensor, SharedTensorView, StorageType } from './ipfs_accelerate_js/src/tensor/shared_tensor';
import StorageManager, { StorageDataType } from './ipfs_accelerate_js_storage_manager';
import { WebNNBackend } from './ipfs_accelerate_js_webnn_backend';
import { WebNNStorageIntegration } from './ipfs_accelerate_js_webnn_storage_integration';

/**
 * Options for the Tensor Sharing Integration
 */
export interface TensorSharingIntegrationOptions {
  /**
   * Enable persistence for shared tensors
   */
  enablePersistence?: boolean;
  
  /**
   * Maximum memory to use for shared tensors (in MB)
   */
  maxMemoryMb?: number;
  
  /**
   * Enable automatic cleanup of unused tensors
   */
  enableAutoCleanup?: boolean;
  
  /**
   * Time in milliseconds after which unused tensors can be removed
   */
  cleanupThreshold?: number;
  
  /**
   * Enable compression for stored tensors
   */
  enableCompression?: boolean;
  
  /**
   * Enable logging
   */
  enableLogging?: boolean;
  
  /**
   * Database name for persistent storage
   */
  dbName?: string;
}

/**
 * Status of a sharing operation
 */
export enum SharingStatus {
  SUCCESS = 'success',
  NOT_FOUND = 'not_found',
  INCOMPATIBLE = 'incompatible',
  ERROR = 'error'
}

/**
 * Result of a sharing operation
 */
export interface SharingResult {
  status: SharingStatus;
  message: string;
  tensorInfo?: {
    name: string;
    shape: number[];
    storageType: StorageType;
    memoryBytes: number;
  };
}

/**
 * Tensor Sharing Integration
 * Connects TensorSharingManager with Storage Manager for persistent shared tensors
 */
export class TensorSharingIntegration {
  private sharingManager: TensorSharingManager;
  private storageManager: StorageManager;
  private webnnBackend: WebNNBackend | null = null;
  private webnnStorageIntegration: WebNNStorageIntegration | null = null;
  private options: Required<TensorSharingIntegrationOptions>;
  private isInitialized = false;
  
  // Track which tensors are stored persistently
  private persistentTensors: Map<string, { storageId: string, lastSynced: number }> = new Map();
  
  // Cache of database tensor IDs to shared tensor names for quick lookup
  private tensorIdMap: Map<string, string> = new Map();
  
  /**
   * Creates a new Tensor Sharing Integration
   */
  constructor(options: TensorSharingIntegrationOptions = {}) {
    // Set default options
    this.options = {
      enablePersistence: options.enablePersistence ?? true,
      maxMemoryMb: options.maxMemoryMb ?? 2048, // 2GB
      enableAutoCleanup: options.enableAutoCleanup ?? true,
      cleanupThreshold: options.cleanupThreshold ?? 1000 * 60 * 60 * 24 * 7, // 7 days
      enableCompression: options.enableCompression ?? false,
      enableLogging: options.enableLogging ?? false,
      dbName: options.dbName ?? 'tensor-sharing-db',
    };
    
    // Create tensor sharing manager
    this.sharingManager = new TensorSharingManager(this.options.maxMemoryMb);
    
    // Create storage manager if persistence is enabled
    if (this.options.enablePersistence) {
      this.storageManager = new StorageManager({
        dbName: this.options.dbName,
        enableCompression: this.options.enableCompression,
        enableAutoCleanup: this.options.enableAutoCleanup,
        cleanupThreshold: this.options.cleanupThreshold,
        enableLogging: this.options.enableLogging,
      });
    }
  }
  
  /**
   * Initialize the tensor sharing integration
   */
  async initialize(webnnBackend?: WebNNBackend): Promise<boolean> {
    if (this.isInitialized) {
      return true;
    }
    
    try {
      // Initialize WebNN backend integration if provided
      if (webnnBackend) {
        this.webnnBackend = webnnBackend;
        this.webnnStorageIntegration = new WebNNStorageIntegration(
          webnnBackend,
          {
            dbName: `${this.options.dbName}-webnn`,
            enableCompression: this.options.enableCompression,
            enableAutoCleanup: this.options.enableAutoCleanup,
            cleanupThreshold: this.options.cleanupThreshold,
            enableLogging: this.options.enableLogging,
          }
        );
        
        await this.webnnStorageIntegration.initialize();
      }
      
      // Initialize storage manager if persistence is enabled
      if (this.options.enablePersistence) {
        const storageInitialized = await this.storageManager.initialize();
        
        if (!storageInitialized) {
          this.log('Failed to initialize storage manager');
          return false;
        }
        
        // Load persistent tensors from storage
        await this.loadPersistentTensors();
      }
      
      this.isInitialized = true;
      this.log('Tensor Sharing Integration initialized successfully');
      return true;
    } catch (error) {
      this.log(`Initialization error: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Load persistent tensors from storage
   */
  private async loadPersistentTensors(): Promise<void> {
    try {
      // Get all models from storage
      const models = await this.storageManager.listModels();
      
      // Find models that represent shared tensors
      const sharedTensorModels = models.filter(model => 
        model.metadata && model.metadata.type === 'shared_tensor'
      );
      
      // Load each shared tensor
      for (const model of sharedTensorModels) {
        // Get tensor metadata
        const tensorName = model.metadata.tensorName;
        
        // Track in persistent tensors map
        this.persistentTensors.set(tensorName, {
          storageId: model.id,
          lastSynced: model.lastAccessed
        });
        
        // Track tensor ID mapping
        for (const tensorId of model.tensorIds) {
          this.tensorIdMap.set(tensorId, tensorName);
        }
        
        this.log(`Loaded persistent tensor metadata: ${tensorName}`);
      }
      
      this.log(`Loaded ${sharedTensorModels.length} persistent shared tensors`);
    } catch (error) {
      this.log(`Failed to load persistent tensors: ${error.message}`);
    }
  }
  
  /**
   * Register a shared tensor
   */
  async registerSharedTensor(
    name: string,
    shape: number[],
    data: Float32Array | Int32Array | Uint8Array,
    storageType: StorageType = "cpu",
    producerModel: string | null = null,
    consumerModels: string[] | null = null,
  ): Promise<SharedTensor> {
    try {
      // Determine data type from the data array
      let dataType: StorageDataType = 'float32';
      if (data instanceof Int32Array) {
        dataType = 'int32';
      } else if (data instanceof Uint8Array) {
        dataType = 'uint8';
      }
      
      // Register in tensor sharing manager
      const tensor = this.sharingManager.registerSharedTensor(
        name,
        shape,
        storageType,
        producerModel,
        consumerModels,
        dataType
      );
      
      // Set the data
      tensor.data = data.slice(); // Create a copy
      
      // Store persistently if enabled
      if (this.options.enablePersistence) {
        await this.persistSharedTensor(tensor, data, dataType);
      }
      
      this.log(`Registered shared tensor: ${name} (${shape.join('x')}, ${storageType})`);
      return tensor;
    } catch (error) {
      this.log(`Failed to register shared tensor: ${error.message}`);
      throw error;
    }
  }
  
  /**
   * Persistently store a shared tensor
   */
  private async persistSharedTensor(
    tensor: SharedTensor,
    data: Float32Array | Int32Array | Uint8Array,
    dataType: StorageDataType
  ): Promise<boolean> {
    try {
      // Create a unique ID for storing the tensor
      const storageId = `shared_tensor_${tensor.name}_${Date.now()}`;
      
      // Store the tensor data
      const tensorId = `${storageId}_data`;
      const success = await this.storageManager.storeTensor(
        tensorId,
        data,
        tensor.shape,
        dataType,
        {
          name: tensor.name,
          storageType: tensor.storageType,
          producerModel: tensor.producerModel,
          isPersistent: true
        }
      );
      
      if (!success) {
        this.log(`Failed to store tensor data: ${tensor.name}`);
        return false;
      }
      
      // Store model metadata to represent the shared tensor
      const modelMetadata = {
        id: storageId,
        name: `Shared Tensor: ${tensor.name}`,
        version: '1.0',
        totalSize: tensor.getMemoryUsage(),
        tensorIds: [tensorId],
        metadata: {
          type: 'shared_tensor',
          tensorName: tensor.name,
          shape: tensor.shape,
          dataType: tensor.dtype,
          storageType: tensor.storageType,
          producerModel: tensor.producerModel
        }
      };
      
      const metadataSuccess = await this.storageManager.storeModelMetadata(modelMetadata);
      
      if (!metadataSuccess) {
        this.log(`Failed to store tensor metadata: ${tensor.name}`);
        return false;
      }
      
      // Track in our maps
      this.persistentTensors.set(tensor.name, {
        storageId,
        lastSynced: Date.now()
      });
      
      this.tensorIdMap.set(tensorId, tensor.name);
      
      this.log(`Persisted shared tensor: ${tensor.name}`);
      return true;
    } catch (error) {
      this.log(`Failed to persist shared tensor: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Get a shared tensor by name
   */
  async getSharedTensor(
    name: string,
    modelName: string | null = null
  ): Promise<SharedTensor | null> {
    try {
      // First try to get from memory
      let tensor = this.sharingManager.getSharedTensor(name, modelName);
      
      // If not found and persistence is enabled, try to load from storage
      if (!tensor && this.options.enablePersistence && this.persistentTensors.has(name)) {
        tensor = await this.loadSharedTensorFromStorage(name, modelName);
      }
      
      if (tensor) {
        this.log(`Retrieved shared tensor: ${name}${modelName ? ` for model ${modelName}` : ''}`);
      } else {
        this.log(`Shared tensor not found: ${name}`);
      }
      
      return tensor;
    } catch (error) {
      this.log(`Failed to get shared tensor: ${error.message}`);
      return null;
    }
  }
  
  /**
   * Load a shared tensor from persistent storage
   */
  private async loadSharedTensorFromStorage(
    name: string,
    modelName: string | null = null
  ): Promise<SharedTensor | null> {
    try {
      const persistentInfo = this.persistentTensors.get(name);
      
      if (!persistentInfo) {
        return null;
      }
      
      // Get model metadata
      const modelMetadata = await this.storageManager.getModelMetadata(persistentInfo.storageId);
      
      if (!modelMetadata) {
        // Remove stale reference
        this.persistentTensors.delete(name);
        return null;
      }
      
      // Get the tensor data
      const tensorId = modelMetadata.tensorIds[0];
      const tensorData = await this.storageManager.getTensorData(tensorId);
      
      if (!tensorData) {
        return null;
      }
      
      // Get tensor metadata
      const tensor = await this.storageManager.getTensor(tensorId);
      
      if (!tensor) {
        return null;
      }
      
      // Create the shared tensor in memory
      const sharedTensor = this.sharingManager.registerSharedTensor(
        name,
        tensor.shape,
        modelMetadata.metadata.storageType,
        modelMetadata.metadata.producerModel,
        null,
        tensor.dataType
      );
      
      // Set the data
      sharedTensor.data = tensorData;
      
      // If model name provided, acquire for this model
      if (modelName) {
        sharedTensor.acquire(modelName);
      }
      
      this.log(`Loaded shared tensor from storage: ${name}`);
      return sharedTensor;
    } catch (error) {
      this.log(`Failed to load shared tensor from storage: ${error.message}`);
      return null;
    }
  }
  
  /**
   * Share a tensor between models
   */
  async shareTensorBetweenModels(
    tensorName: string,
    fromModel: string,
    toModels: string[]
  ): Promise<SharingResult> {
    try {
      // Get the tensor
      const tensor = await this.getSharedTensor(tensorName, null);
      
      if (!tensor) {
        return {
          status: SharingStatus.NOT_FOUND,
          message: `Tensor ${tensorName} not found`
        };
      }
      
      // Share in memory
      const success = this.sharingManager.shareTensorBetweenModels(
        tensorName,
        fromModel,
        toModels
      );
      
      if (!success) {
        return {
          status: SharingStatus.ERROR,
          message: `Failed to share tensor ${tensorName} from ${fromModel}`
        };
      }
      
      // Return success result
      return {
        status: SharingStatus.SUCCESS,
        message: `Shared tensor ${tensorName} from ${fromModel} to ${toModels.join(', ')}`,
        tensorInfo: {
          name: tensorName,
          shape: tensor.shape,
          storageType: tensor.storageType,
          memoryBytes: tensor.getMemoryUsage()
        }
      };
    } catch (error) {
      return {
        status: SharingStatus.ERROR,
        message: `Error sharing tensor: ${error.message}`
      };
    }
  }
  
  /**
   * Create a view into a shared tensor
   */
  async createTensorView(
    tensorName: string,
    viewName: string,
    offset: number[],
    size: number[],
    modelName: string | null = null
  ): Promise<SharedTensorView | null> {
    try {
      // Get the tensor
      const tensor = await this.getSharedTensor(tensorName, null);
      
      if (!tensor) {
        this.log(`Parent tensor ${tensorName} not found for creating view`);
        return null;
      }
      
      // Create the view
      return this.sharingManager.createTensorView(
        tensorName,
        viewName,
        offset,
        size,
        modelName
      );
    } catch (error) {
      this.log(`Failed to create tensor view: ${error.message}`);
      return null;
    }
  }
  
  /**
   * Release tensors used by a model
   */
  async releaseModelTensors(modelName: string): Promise<number> {
    return this.sharingManager.releaseModelTensors(modelName);
  }
  
  /**
   * Optimize memory usage by freeing unused tensors
   */
  async optimizeMemoryUsage(): Promise<Record<string, any>> {
    return this.sharingManager.optimizeMemoryUsage();
  }
  
  /**
   * Analyze sharing opportunities
   */
  analyzeSharingOpportunities(): Record<string, string[]> {
    return this.sharingManager.analyzeSharingOpportunities();
  }
  
  /**
   * Get tensor memory usage
   */
  getTensorMemoryUsage(): Record<string, Record<string, any>> {
    return this.sharingManager.getTensorMemoryUsage();
  }
  
  /**
   * Get model memory usage
   */
  getModelMemoryUsage(): Record<string, Record<string, any>> {
    return this.sharingManager.getModelMemoryUsage();
  }
  
  /**
   * Get optimization recommendations
   */
  getOptimizationRecommendations(): Record<string, any> {
    return this.sharingManager.getOptimizationRecommendations();
  }
  
  /**
   * Get statistics about the tensor sharing manager
   */
  getStats(): Record<string, any> {
    const stats = this.sharingManager.getStats();
    
    // Add persistent storage stats if available
    if (this.options.enablePersistence) {
      stats.persistentTensorCount = this.persistentTensors.size;
    }
    
    return stats;
  }
  
  /**
   * Synchronize in-memory tensors with persistent storage
   * This ensures tensors that have been modified are updated in storage
   */
  async synchronizePersistentStorage(): Promise<Record<string, any>> {
    if (!this.options.enablePersistence) {
      return {
        success: false,
        message: 'Persistence is not enabled',
        updatedCount: 0
      };
    }
    
    try {
      let updatedCount = 0;
      const failedUpdates: string[] = [];
      
      // Get tensor memory usage to find all tensors
      const tensorMemoryUsage = this.sharingManager.getTensorMemoryUsage();
      
      for (const [name, usage] of Object.entries(tensorMemoryUsage)) {
        // Check if tensor is already persistent
        const persistentInfo = this.persistentTensors.get(name);
        
        // Get the tensor
        const tensor = this.sharingManager.getSharedTensor(name, null);
        
        if (!tensor || !tensor.data) {
          continue; // Skip if tensor doesn't have data
        }
        
        // Determine if we need to update in persistent storage
        const needsUpdate = !persistentInfo || 
          (tensor.referenceCount > 0 && Date.now() - persistentInfo.lastSynced > 60000); // 1 minute
        
        if (needsUpdate) {
          // Convert data to typed array
          let typedData: Float32Array | Int32Array | Uint8Array;
          if (tensor.dtype === 'float32') {
            typedData = new Float32Array(tensor.data);
          } else if (tensor.dtype === 'int32') {
            typedData = new Int32Array(tensor.data);
          } else {
            typedData = new Uint8Array(tensor.data);
          }
          
          // Store in persistent storage
          const success = await this.persistSharedTensor(
            tensor,
            typedData,
            tensor.dtype as StorageDataType
          );
          
          if (success) {
            updatedCount++;
          } else {
            failedUpdates.push(name);
          }
        }
      }
      
      return {
        success: true,
        message: `Synchronized ${updatedCount} tensors with persistent storage`,
        updatedCount,
        failedUpdates
      };
    } catch (error) {
      this.log(`Failed to synchronize with persistent storage: ${error.message}`);
      return {
        success: false,
        message: `Error: ${error.message}`,
        updatedCount: 0
      };
    }
  }
  
  /**
   * Create WebNN tensors from shared tensors
   * This is useful when working with the WebNN backend
   */
  async createWebNNTensorsFromShared(
    sharedTensorNames: string[],
    modelName: string
  ): Promise<Map<string, any> | null> {
    if (!this.webnnBackend || !this.webnnStorageIntegration) {
      this.log('WebNN backend not initialized');
      return null;
    }
    
    try {
      const webnnTensors = new Map<string, any>();
      
      for (const name of sharedTensorNames) {
        // Get shared tensor
        const sharedTensor = await this.getSharedTensor(name, modelName);
        
        if (!sharedTensor || !sharedTensor.data) {
          this.log(`Shared tensor not found or missing data: ${name}`);
          continue;
        }
        
        // Convert data to typed array
        let typedData: Float32Array | Int32Array | Uint8Array;
        if (sharedTensor.dtype === 'float32') {
          typedData = new Float32Array(sharedTensor.data);
        } else if (sharedTensor.dtype === 'int32') {
          typedData = new Int32Array(sharedTensor.data);
        } else {
          typedData = new Uint8Array(sharedTensor.data);
        }
        
        // Create WebNN tensor
        const webnnTensor = await this.webnnBackend.createTensor(
          typedData,
          sharedTensor.shape,
          sharedTensor.dtype as StorageDataType
        );
        
        webnnTensors.set(name, webnnTensor.tensor);
      }
      
      return webnnTensors;
    } catch (error) {
      this.log(`Failed to create WebNN tensors: ${error.message}`);
      return null;
    }
  }
  
  /**
   * Save shared tensors as a WebNN model
   * This allows utilizing the WebNN storage for shared tensors
   */
  async saveAsWebNNModel(
    modelId: string,
    modelName: string,
    sharedTensorNames: string[]
  ): Promise<boolean> {
    if (!this.webnnBackend || !this.webnnStorageIntegration) {
      this.log('WebNN backend not initialized');
      return false;
    }
    
    try {
      // Create weights map
      const weights = new Map<string, { data: Float32Array | Int32Array | Uint8Array; shape: number[]; dataType: StorageDataType }>();
      
      for (const name of sharedTensorNames) {
        // Get shared tensor
        const sharedTensor = await this.getSharedTensor(name, null);
        
        if (!sharedTensor || !sharedTensor.data) {
          this.log(`Shared tensor not found or missing data: ${name}`);
          continue;
        }
        
        // Convert data to typed array
        let typedData: Float32Array | Int32Array | Uint8Array;
        if (sharedTensor.dtype === 'float32') {
          typedData = new Float32Array(sharedTensor.data);
        } else if (sharedTensor.dtype === 'int32') {
          typedData = new Int32Array(sharedTensor.data);
        } else {
          typedData = new Uint8Array(sharedTensor.data);
        }
        
        weights.set(name, {
          data: typedData,
          shape: sharedTensor.shape,
          dataType: sharedTensor.dtype as StorageDataType
        });
      }
      
      // Store as WebNN model
      const success = await this.webnnStorageIntegration.storeModel(
        modelId,
        modelName,
        weights,
        {
          type: 'shared_tensors',
          tensorNames: sharedTensorNames
        }
      );
      
      if (success) {
        this.log(`Saved shared tensors as WebNN model: ${modelId}`);
      }
      
      return success;
    } catch (error) {
      this.log(`Failed to save as WebNN model: ${error.message}`);
      return false;
    }
  }
  
  /**
   * Load shared tensors from a WebNN model
   */
  async loadFromWebNNModel(
    modelId: string,
    producerModel: string | null = null
  ): Promise<string[]> {
    if (!this.webnnBackend || !this.webnnStorageIntegration) {
      this.log('WebNN backend not initialized');
      return [];
    }
    
    try {
      // Load WebNN model
      const tensors = await this.webnnStorageIntegration.loadModel(modelId);
      
      if (!tensors) {
        this.log(`WebNN model not found: ${modelId}`);
        return [];
      }
      
      // Get model metadata
      const models = await this.webnnStorageIntegration.listModels();
      const modelInfo = models.find(m => m.id === modelId);
      
      if (!modelInfo) {
        this.log(`WebNN model metadata not found: ${modelId}`);
        return [];
      }
      
      const registeredTensors: string[] = [];
      
      // Register as shared tensors
      for (const [name, { tensor, shape }] of tensors.entries()) {
        // Create a unique name if none provided
        const tensorName = name || `webnn_tensor_${Date.now()}_${registeredTensors.length}`;
        
        // Extract data
        const data = await this.webnnBackend.getTensorData(tensor);
        
        if (!data) {
          this.log(`Failed to get tensor data for: ${name}`);
          continue;
        }
        
        // Register as shared tensor
        await this.registerSharedTensor(
          tensorName,
          shape,
          data,
          'webnn',
          producerModel,
          null
        );
        
        registeredTensors.push(tensorName);
      }
      
      this.log(`Loaded ${registeredTensors.length} shared tensors from WebNN model: ${modelId}`);
      return registeredTensors;
    } catch (error) {
      this.log(`Failed to load from WebNN model: ${error.message}`);
      return [];
    }
  }
  
  /**
   * Log message if logging is enabled
   */
  private log(message: string): void {
    if (this.options.enableLogging) {
      console.log(`[TensorSharingIntegration] ${message}`);
    }
  }
}

export default TensorSharingIntegration;