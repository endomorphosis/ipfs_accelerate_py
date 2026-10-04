/**
 * Cross-Model Tensor Sharing for WebGPU/WebNN Resource Pool Integration
 *
 * This module implements efficient tensor sharing across multiple models in the WebGPU/WebNN
 * resource pool, enabling:
 *
 * 1. Shared tensor memory across models running on the same hardware
 * 2. Efficient multimodal applications with shared representations
 * 3. Memory optimization through tensor reuse
 * 4. Cached intermediate representations for common model components
 *
 * Key features:
 * - Tensor reference counting for efficient memory management
 * - Support for different tensor storage formats (WebGPU, WebNN, CPU)
 * - Tensor view support for zero-copy tensor slicing
 * - Smart caching of shared embedding spaces
 * - Cross-model intermediate representation sharing
 */

import { Tensor } from './tensor';

/**
 * Types of storage for shared tensors
 */
export type StorageType = 'cpu' | 'webgpu' | 'webnn';

/**
 * Interface for shared tensor options
 */
export interface SharedTensorOptions {
  name: string;
  shape: number[];
  dtype?: string;
  storageType?: StorageType;
  producerModel?: string;
}

/**
 * A tensor that can be shared between multiple models.
 * 
 * Implements reference counting and intelligent memory management
 * to ensure tensors are only freed when no longer needed by any model.
 */
export class SharedTensor {
  /** Unique name for this tensor */
  readonly name: string;
  
  /** Shape of the tensor */
  readonly shape: number[];
  
  /** Data type of the tensor */
  readonly dtype: string;
  
  /** Where the tensor is stored ('cpu', 'webgpu', 'webnn') */
  readonly storageType: StorageType;
  
  /** Name of the model that created this tensor */
  readonly producerModel: string | null;
  
  /** Set of models consuming this tensor */
  private consumerModels: Set<string>;
  
  /** Current reference count */
  referenceCount: number;
  
  /** Last time this tensor was accessed */
  private lastAccessed: number;
  
  /** Actual tensor data */
  data: any | null;
  
  /** Views into this tensor */
  private views: Map<string, SharedTensorView>;
  
  /** If true, will not be freed regardless of reference count */
  isPinned: boolean;
  
  /** Additional metadata for this tensor */
  private metadata: Record<string, any>;
  
  /** Storage-specific attributes */
  private gpuBufferId: string | null;
  private webnnTensorId: string | null;
  
  /**
   * Initialize a shared tensor.
   * 
   * @param name Unique name for this tensor
   * @param shape Shape of the tensor
   * @param dtype Data type of the tensor
   * @param storageType Where the tensor is stored ('cpu', 'webgpu', 'webnn')
   * @param producerModel Name of the model that created this tensor
   */
  constructor({
    name,
    shape,
    dtype = "float32",
    storageType = "cpu",
    producerModel = null
  }: SharedTensorOptions) {
    this.name = name;
    this.shape = shape;
    this.dtype = dtype;
    this.storageType = storageType as StorageType;
    this.producerModel = producerModel;
    this.consumerModels = new Set<string>();
    this.referenceCount = 0;
    this.lastAccessed = Date.now();
    this.data = null;
    this.views = new Map<string, SharedTensorView>();
    this.isPinned = false;
    this.metadata = {};
    
    // Storage-specific attributes
    this.gpuBufferId = null;
    this.webnnTensorId = null;
    
    console.debug(`Created shared tensor ${name} with shape ${shape} and type ${storageType}`);
  }
  
  /**
   * Acquire this tensor for use by a model.
   * 
   * @param modelName Name of the model acquiring the tensor
   * @returns True if acquisition was successful
   */
  acquire(modelName: string): boolean {
    this.consumerModels.add(modelName);
    this.referenceCount += 1;
    this.lastAccessed = Date.now();
    console.debug(`Model ${modelName} acquired tensor ${this.name}, reference count: ${this.referenceCount}`);
    return true;
  }
  
  /**
   * Release this tensor from use by a model.
   * 
   * @param modelName Name of the model releasing the tensor
   * @returns True if release was successful
   */
  release(modelName: string): boolean {
    if (this.consumerModels.has(modelName)) {
      this.consumerModels.delete(modelName);
      this.referenceCount = Math.max(0, this.referenceCount - 1);
      console.debug(`Model ${modelName} released tensor ${this.name}, reference count: ${this.referenceCount}`);
      return true;
    }
    return false;
  }
  
  /**
   * Pin the tensor to prevent automatic release.
   */
  pin(): void {
    this.isPinned = true;
    console.debug(`Tensor ${this.name} pinned in memory`);
  }
  
  /**
   * Unpin the tensor to allow automatic release.
   */
  unpin(): void {
    this.isPinned = false;
    console.debug(`Tensor ${this.name} unpinned from memory`);
  }
  
  /**
   * Check if this tensor can be freed from memory.
   * 
   * @returns True if the tensor can be freed
   */
  canBeFreed(): boolean {
    return (
      !this.isPinned &&
      this.referenceCount === 0 &&
      this.consumerModels.size === 0 &&
      Date.now() - this.lastAccessed > 30000  // 30 second grace period
    );
  }
  
  /**
   * Create a view into this tensor.
   * 
   * @param name Name for the view
   * @param offset Start indices for the view
   * @param size Size of the view
   * @returns SharedTensorView object
   */
  createView(name: string, offset: number[], size: number[]): SharedTensorView {
    const view = new SharedTensorView(this, name, offset, size);
    this.views.set(name, view);
    return view;
  }
  
  /**
   * Copy this tensor to a different storage type.
   * 
   * @param targetStorageType The target storage type
   * @returns New SharedTensor with the copied data
   */
  copyTo(targetStorageType: StorageType): SharedTensor {
    // Create a new tensor with the target storage type
    const newTensor = new SharedTensor({
      name: `${this.name}_${targetStorageType}`,
      shape: this.shape,
      dtype: this.dtype,
      storageType: targetStorageType,
      producerModel: this.producerModel
    });
    
    // In a real implementation, we would copy the data between storage types
    // This would involve WebGPU/WebNN specific code
    
    // Simulate data copy
    console.info(`Copying tensor ${this.name} from ${this.storageType} to ${targetStorageType}`);
    newTensor.data = this.data;  // In a real implementation, this would be a proper copy
    
    return newTensor;
  }
  
  /**
   * Get the memory usage of this tensor in bytes.
   * 
   * @returns Memory usage in bytes
   */
  getMemoryUsage(): number {
    let elementSize = 4;  // Assume float32 (4 bytes)
    
    if (this.dtype === "float16") {
      elementSize = 2;
    } else if (this.dtype === "int8") {
      elementSize = 1;
    }
    
    let numElements = 1;
    for (const dim of this.shape) {
      numElements *= dim;
    }
    
    return numElements * elementSize;
  }
  
  /**
   * String representation of the tensor
   */
  toString(): string {
    return `SharedTensor(name=${this.name}, shape=${this.shape}, type=${this.dtype}, storage=${this.storageType}, refs=${this.referenceCount}, producer=${this.producerModel})`;
  }
}

/**
 * A view into a shared tensor, representing a slice or subset of the tensor.
 * 
 * This allows multiple models to use different parts of the same tensor
 * without duplicating memory.
 */
export class SharedTensorView {
  /** The parent tensor this is a view into */
  readonly parent: SharedTensor;
  
  /** Unique name for this view */
  readonly name: string;
  
  /** Start indices for the view */
  readonly offset: number[];
  
  /** Size of the view */
  readonly size: number[];
  
  /** Set of models consuming this tensor view */
  private consumerModels: Set<string>;
  
  /** Current reference count */
  private referenceCount: number;
  
  /** Last time this tensor view was accessed */
  private lastAccessed: number;
  
  /**
   * Initialize a tensor view.
   * 
   * @param parent The parent tensor this is a view into
   * @param name Unique name for this view
   * @param offset Start indices for the view
   * @param size Size of the view
   */
  constructor(
    parent: SharedTensor,
    name: string,
    offset: number[],
    size: number[]
  ) {
    this.parent = parent;
    this.name = name;
    this.offset = offset;
    this.size = size;
    this.consumerModels = new Set<string>();
    this.referenceCount = 0;
    this.lastAccessed = Date.now();
    
    console.debug(`Created tensor view ${name} into ${parent.name} with offset ${offset} and size ${size}`);
  }
  
  /**
   * Acquire this tensor view for use by a model.
   * 
   * @param modelName Name of the model acquiring the view
   * @returns True if acquisition was successful
   */
  acquire(modelName: string): boolean {
    // Acquire both the view and the parent tensor
    this.consumerModels.add(modelName);
    this.referenceCount += 1;
    this.lastAccessed = Date.now();
    this.parent.acquire(modelName);
    
    console.debug(`Model ${modelName} acquired tensor view ${this.name}, reference count: ${this.referenceCount}`);
    return true;
  }
  
  /**
   * Release this tensor view from use by a model.
   * 
   * @param modelName Name of the model releasing the view
   * @returns True if release was successful
   */
  release(modelName: string): boolean {
    if (this.consumerModels.has(modelName)) {
      this.consumerModels.delete(modelName);
      this.referenceCount = Math.max(0, this.referenceCount - 1);
      this.parent.release(modelName);
      
      console.debug(`Model ${modelName} released tensor view ${this.name}, reference count: ${this.referenceCount}`);
      return true;
    }
    return false;
  }
  
  /**
   * Get the data for this view.
   * 
   * @returns The tensor view data
   */
  getData(): any {
    this.lastAccessed = Date.now();
    
    // In a real implementation, this would return a slice or view of the parent tensor
    // based on the offset and size
    return null;  // Placeholder
  }
  
  /**
   * String representation of the tensor view
   */
  toString(): string {
    return `SharedTensorView(name=${this.name}, parent=${this.parent.name}, offset=${this.offset}, size=${this.size}, refs=${this.referenceCount})`;
  }
}

/**
 * Manager for shared tensors across multiple models.
 * 
 * This class handles tensor registration, sharing, memory optimization,
 * and lifecycle management for tensors shared across models.
 */
export class TensorSharingManager {
  /** Map of tensor name to tensor object */
  private tensors: Map<string, SharedTensor>;
  
  /** Maps model names to sets of tensor names */
  private modelTensors: Map<string, Set<string>>;
  
  /** Maximum memory to use for shared tensors (in MB) */
  private maxMemoryMb: number | null;
  
  /** Current memory usage in bytes */
  private currentMemoryUsage: number;
  
  /** Number of cache hits */
  private cacheHits: number;
  
  /** Number of cache misses */
  private cacheMisses: number;
  
  /** Stats for tensor usage */
  private tensorUsageStats: Map<string, Record<string, any>>;
  
  /** Set up cross-model sharing patterns */
  private sharingPatterns: Record<string, string[]>;
  
  /**
   * Initialize the tensor sharing manager.
   * 
   * @param maxMemoryMb Maximum memory to use for shared tensors (in MB)
   */
  constructor(maxMemoryMb: number | null = null) {
    this.tensors = new Map<string, SharedTensor>();
    this.modelTensors = new Map<string, Set<string>>();
    this.maxMemoryMb = maxMemoryMb;
    this.currentMemoryUsage = 0;
    this.cacheHits = 0;
    this.cacheMisses = 0;
    this.tensorUsageStats = new Map<string, Record<string, any>>();
    
    // Set up cross-model sharing patterns
    this.sharingPatterns = {
      // Common embedding spaces that can be shared
      "text_embedding": ["bert", "t5", "llama", "bart"],
      "vision_embedding": ["vit", "clip", "detr"],
      "audio_embedding": ["whisper", "wav2vec2", "clap"],
      // Multimodal shared representations
      "vision_text_joint": ["clip", "llava", "blip"],
      "audio_text_joint": ["clap", "whisper_text"]
    };
    
    console.info(`TensorSharingManager initialized with max memory: ${maxMemoryMb} MB`);
  }
  
  /**
   * Register a new shared tensor.
   * 
   * @param name Unique name for this tensor
   * @param shape Shape of the tensor
   * @param storageType Where the tensor is stored ('cpu', 'webgpu', 'webnn')
   * @param producerModel Name of the model that created this tensor
   * @param consumerModels List of models that will use this tensor
   * @param dtype Data type of the tensor
   * @returns The created SharedTensor
   */
  registerSharedTensor(
    name: string,
    shape: number[],
    storageType: StorageType = "cpu",
    producerModel: string | null = null,
    consumerModels: string[] | null = null,
    dtype: string = "float32"
  ): SharedTensor {
    if (this.tensors.has(name)) {
      console.warn(`Tensor ${name} already registered. Returning existing tensor.`);
      return this.tensors.get(name)!;
    }
    
    // Create the shared tensor
    const tensor = new SharedTensor({
      name,
      shape,
      dtype,
      storageType,
      producerModel
    });
    
    // Register the tensor
    this.tensors.set(name, tensor);
    
    // Track memory usage
    const tensorMemory = tensor.getMemoryUsage();
    this.currentMemoryUsage += tensorMemory;
    
    // Track by producer model
    if (producerModel) {
      if (!this.modelTensors.has(producerModel)) {
        this.modelTensors.set(producerModel, new Set<string>());
      }
      this.modelTensors.get(producerModel)!.add(name);
      
      // Acquire reference for producer
      tensor.acquire(producerModel);
    }
    
    // Register for consumer models
    if (consumerModels) {
      for (const model of consumerModels) {
        if (!this.modelTensors.has(model)) {
          this.modelTensors.set(model, new Set<string>());
        }
        this.modelTensors.get(model)!.add(name);
      }
    }
    
    // Initialize usage stats
    this.tensorUsageStats.set(name, {
      "created_at": Date.now(),
      "access_count": 0,
      "last_accessed": Date.now(),
      "memory_bytes": tensorMemory,
      "producer": producerModel,
      "consumers": new Set(consumerModels || [])
    });
    
    console.info(`Registered shared tensor ${name} with shape ${shape} and storage type ${storageType}`);
    return tensor;
  }
  
  /**
   * Get a shared tensor by name.
   * 
   * @param name Name of the tensor to get
   * @param modelName Name of the model requesting the tensor
   * @returns The shared tensor or null if not found
   */
  getSharedTensor(name: string, modelName: string | null = null): SharedTensor | null {
    if (!this.tensors.has(name)) {
      console.warn(`Tensor ${name} not found`);
      this.cacheMisses += 1;
      return null;
    }
    
    const tensor = this.tensors.get(name)!;
    
    // Update usage stats
    const stats = this.tensorUsageStats.get(name)!;
    stats.access_count += 1;
    stats.last_accessed = Date.now();
    this.cacheHits += 1;
    
    // If model name provided, acquire for this model
    if (modelName) {
      tensor.acquire(modelName);
      
      // Add to model's tensor set
      if (!this.modelTensors.has(modelName)) {
        this.modelTensors.set(modelName, new Set<string>());
      }
      this.modelTensors.get(modelName)!.add(name);
      
      // Update consumers in stats
      stats.consumers.add(modelName);
    }
    
    return tensor;
  }
  
  /**
   * Create a view into a shared tensor.
   * 
   * @param tensorName Name of the parent tensor
   * @param viewName Name for the new view
   * @param offset Start indices for the view
   * @param size Size of the view
   * @param modelName Name of the model creating the view
   * @returns The created SharedTensorView or null if parent tensor not found
   */
  createTensorView(
    tensorName: string,
    viewName: string,
    offset: number[],
    size: number[],
    modelName: string | null = null
  ): SharedTensorView | null {
    if (!this.tensors.has(tensorName)) {
      console.warn(`Parent tensor ${tensorName} not found`);
      return null;
    }
    
    const parent = this.tensors.get(tensorName)!;
    
    // Create the view
    const view = parent.createView(viewName, offset, size);
    
    // If model name provided, acquire for this model
    if (modelName) {
      view.acquire(modelName);
    }
    
    console.info(`Created tensor view ${viewName} into ${tensorName} for model ${modelName}`);
    return view;
  }
  
  /**
   * Share a tensor from one model to others.
   * 
   * @param tensorName Name of the tensor to share
   * @param fromModel Model sharing the tensor
   * @param toModels Models to share the tensor with
   * @returns True if sharing was successful
   */
  shareTensorBetweenModels(
    tensorName: string,
    fromModel: string,
    toModels: string[]
  ): boolean {
    if (!this.tensors.has(tensorName)) {
      console.warn(`Tensor ${tensorName} not found for sharing`);
      return false;
    }
    
    const tensor = this.tensors.get(tensorName)!;
    
    // Make sure the fromModel is the producer or a consumer
    const modelTensorSet = this.modelTensors.get(fromModel);
    if (tensor.producerModel !== fromModel && (!modelTensorSet || !modelTensorSet.has(tensorName))) {
      console.warn(`Model ${fromModel} does not own tensor ${tensorName}`);
      return false;
    }
    
    // Share with target models
    for (const model of toModels) {
      if (!this.modelTensors.has(model)) {
        this.modelTensors.set(model, new Set<string>());
      }
      
      // Add to model's tensor set
      this.modelTensors.get(model)!.add(tensorName);
      
      // Update usage stats
      this.tensorUsageStats.get(tensorName)!.consumers.add(model);
    }
    
    console.info(`Shared tensor ${tensorName} from ${fromModel} to ${toModels}`);
    return true;
  }
  
  /**
   * Optimize memory usage by freeing unused tensors.
   * 
   * @returns Dictionary with optimization results
   */
  optimizeMemoryUsage(): Record<string, any> {
    const initialMemory = this.currentMemoryUsage;
    const freedTensors: string[] = [];
    let freedMemory = 0;
    
    // Check for unused tensors that can be freed
    for (const [name, tensor] of this.tensors.entries()) {
      if (tensor.canBeFreed()) {
        // Calculate memory to be freed
        const tensorMemory = tensor.getMemoryUsage();
        freedMemory += tensorMemory;
        
        // Remove from manager
        this.tensors.delete(name);
        this.tensorUsageStats.delete(name);
        
        // Remove from model mappings
        for (const [model, tensorSet] of this.modelTensors.entries()) {
          if (tensorSet.has(name)) {
            tensorSet.delete(name);
          }
        }
        
        freedTensors.push(name);
        console.info(`Freed unused tensor ${name}, saved ${tensorMemory / (1024 * 1024)} MB`);
      }
    }
    
    // Update current memory usage
    this.currentMemoryUsage -= freedMemory;
    
    // Prepare result dictionary
    const result = {
      "initial_memory_bytes": initialMemory,
      "current_memory_bytes": this.currentMemoryUsage,
      "freed_memory_bytes": freedMemory,
      "freed_tensors_count": freedTensors.length,
      "freed_tensors": freedTensors,
      "memory_reduction_percent": (freedMemory / initialMemory * 100) || 0,
      "remaining_tensor_count": this.tensors.size
    };
    
    console.info(`Memory optimization complete: freed ${freedMemory / (1024 * 1024)} MB (${result.memory_reduction_percent}%)`);
    return result;
  }
  
  /**
   * Analyze the current models and tensors to identify sharing opportunities.
   * 
   * @returns Dictionary of tensor names to lists of models that could share them
   */
  analyzeSharingOpportunities(): Record<string, string[]> {
    const opportunities: Record<string, string[]> = {};
    
    // Identify potential sharing opportunities based on model combinations
    const activeModels = new Set(this.modelTensors.keys());
    
    // Check each sharing pattern
    for (const [tensorType, compatibleModels] of Object.entries(this.sharingPatterns)) {
      // Find active models that match this pattern
      const matchingModels: string[] = [];
      for (const model of activeModels) {
        if (compatibleModels.includes(model)) {
          matchingModels.push(model);
        }
      }
      
      if (matchingModels.length >= 2) {
        // There are at least 2 active models that could share this tensor type
        opportunities[tensorType] = matchingModels;
      }
    }
    
    console.info(`Identified ${Object.keys(opportunities).length} tensor sharing opportunities`);
    return opportunities;
  }
  
  /**
   * Get detailed memory usage by tensor.
   * 
   * @returns Dictionary mapping tensor names to memory usage info
   */
  getTensorMemoryUsage(): Record<string, Record<string, any>> {
    const memoryUsage: Record<string, Record<string, any>> = {};
    
    for (const [name, tensor] of this.tensors.entries()) {
      const memoryBytes = tensor.getMemoryUsage();
      // Get consumer models from our tracked data
      const consumerModels = new Set<string>();
      for (const [model, tensorSet] of this.modelTensors.entries()) {
        if (tensorSet.has(name) && model !== tensor.producerModel) {
          consumerModels.add(model);
        }
      }
      
      memoryUsage[name] = {
        "memory_bytes": memoryBytes,
        "memory_mb": memoryBytes / (1024 * 1024),
        "shape": tensor.shape,
        "dtype": tensor.dtype,
        "storage_type": tensor.storageType,
        "reference_count": tensor.referenceCount,
        "consumer_count": consumerModels.size,
        "consumers": Array.from(consumerModels),
        "producer": tensor.producerModel
      };
    }
    
    return memoryUsage;
  }
  
  /**
   * Get detailed memory usage by model.
   * 
   * @returns Dictionary mapping model names to memory usage info
   */
  getModelMemoryUsage(): Record<string, Record<string, any>> {
    const modelMemory: Record<string, Record<string, any>> = {};
    
    for (const [modelName, tensorNames] of this.modelTensors.entries()) {
      let totalMemory = 0;
      const tensorDetails: Record<string, Record<string, any>> = {};
      
      for (const tensorName of tensorNames) {
        if (this.tensors.has(tensorName)) {
          const tensor = this.tensors.get(tensorName)!;
          const memoryBytes = tensor.getMemoryUsage();
          totalMemory += memoryBytes;
          
          tensorDetails[tensorName] = {
            "memory_bytes": memoryBytes,
            "memory_mb": memoryBytes / (1024 * 1024),
            "shape": tensor.shape
          };
        }
      }
      
      modelMemory[modelName] = {
        "total_memory_bytes": totalMemory,
        "total_memory_mb": totalMemory / (1024 * 1024),
        "tensor_count": tensorNames.size,
        "tensors": tensorDetails
      };
    }
    
    return modelMemory;
  }
  
  /**
   * Get recommendations for memory optimization.
   * 
   * @returns Dictionary with optimization recommendations
   */
  getOptimizationRecommendations(): Record<string, any> {
    // Analyze current memory usage
    const modelMemory = this.getModelMemoryUsage();
    const tensorMemory = this.getTensorMemoryUsage();
    
    // Find the largest tensors
    const largestTensors = Object.entries(tensorMemory)
      .map(([name, info]) => ({ name, memoryBytes: info.memory_bytes }))
      .sort((a, b) => b.memoryBytes - a.memoryBytes)
      .slice(0, 5);  // Top 5 largest tensors
    
    // Find tensors with low reference counts
    const lowRefTensors: string[] = [];
    for (const [name, tensor] of this.tensors.entries()) {
      if (tensor.referenceCount <= 1 && !tensor.isPinned) {
        lowRefTensors.push(name);
      }
    }
    
    // Find shared tensor opportunities
    const sharingOpportunities = this.analyzeSharingOpportunities();
    
    // Calculate potential savings
    let potentialSavingsBytes = 0;
    for (const tensor of this.tensors.values()) {
      if (tensor.canBeFreed()) {
        potentialSavingsBytes += tensor.getMemoryUsage();
      }
    }
    
    // Prepare recommendations
    const recommendations = {
      "largest_tensors": largestTensors.map(({ name, memoryBytes }) => ({
        name,
        memory_mb: memoryBytes / (1024 * 1024)
      })),
      "low_reference_tensors": lowRefTensors,
      "sharing_opportunities": sharingOpportunities,
      "total_memory_mb": this.currentMemoryUsage / (1024 * 1024),
      "potential_savings_mb": potentialSavingsBytes / (1024 * 1024),
      "cache_efficiency": {
        "hits": this.cacheHits,
        "misses": this.cacheMisses,
        "hit_rate": this.cacheHits / (this.cacheHits + this.cacheMisses) || 0
      }
    };
    
    return recommendations;
  }
  
  /**
   * Release all tensors used by a model.
   * 
   * @param modelName Name of the model to release tensors for
   * @returns Number of tensors released
   */
  releaseModelTensors(modelName: string): number {
    if (!this.modelTensors.has(modelName)) {
      console.warn(`Model ${modelName} not found in tensor manager`);
      return 0;
    }
    
    let releasedCount = 0;
    const tensorNames = Array.from(this.modelTensors.get(modelName)!);
    
    for (const tensorName of tensorNames) {
      if (this.tensors.has(tensorName)) {
        const tensor = this.tensors.get(tensorName)!;
        tensor.release(modelName);
        releasedCount += 1;
      }
    }
    
    // Remove model from tracking
    this.modelTensors.delete(modelName);
    
    console.info(`Released ${releasedCount} tensors for model ${modelName}`);
    return releasedCount;
  }
  
  /**
   * Get statistics about the tensor sharing manager.
   * 
   * @returns Dictionary with statistics
   */
  getStats(): Record<string, any> {
    return {
      "total_tensors": this.tensors.size,
      "total_models": this.modelTensors.size,
      "memory_usage_bytes": this.currentMemoryUsage,
      "memory_usage_mb": this.currentMemoryUsage / (1024 * 1024),
      "cache_hits": this.cacheHits,
      "cache_misses": this.cacheMisses,
      "hit_rate": this.cacheHits / (this.cacheHits + this.cacheMisses) || 0,
      "models": Array.from(this.modelTensors.keys()),
      "sharing_opportunities": this.analyzeSharingOpportunities()
    };
  }
}

/**
 * Get models that can share a tensor of the given type.
 * 
 * @param tensorType Type of tensor to check
 * @returns List of compatible model names
 */
export function getCompatibleModelsForTensor(tensorType: string): string[] {
  // Default sharing patterns for common tensor types
  const sharingPatterns: Record<string, string[]> = {
    "text_embedding": ["bert", "t5", "llama", "bart", "roberta", "gpt2"],
    "vision_embedding": ["vit", "clip", "detr", "swin", "dino"],
    "audio_embedding": ["whisper", "wav2vec2", "clap", "hubert"],
    "vision_text_joint": ["clip", "llava", "blip", "xclip"],
    "audio_text_joint": ["clap", "whisper_text", "wav2vec2_text"],
  };
  
  return sharingPatterns[tensorType] || [];
}

/**
 * Create a demonstration of tensor sharing functionality.
 * 
 * @returns Dictionary with demonstration results
 */
export function createTensorSharingDemo(): Record<string, any> {
  // Create tensor sharing manager
  const manager = new TensorSharingManager(2048);
  
  // Register example tensors
  const textEmbedding = manager.registerSharedTensor(
    "bert_embedding",
    [1, 768],
    "cpu",
    "bert",
    ["t5", "llama"],
    "float32"
  );
  
  const visionEmbedding = manager.registerSharedTensor(
    "vit_embedding",
    [1, 1024],
    "webgpu",
    "vit",
    ["clip"],
    "float32"
  );
  
  // Create a tensor view
  const embeddingView = manager.createTensorView(
    "bert_embedding",
    "bert_embedding_first_half",
    [0, 0],
    [1, 384],
    "t5"
  );
  
  // Share tensor with additional models
  manager.shareTensorBetweenModels(
    "vit_embedding",
    "vit",
    ["llava", "xclip"]
  );
  
  // Analyze sharing opportunities
  const opportunities = manager.analyzeSharingOpportunities();
  
  // Get memory usage
  const modelMemory = manager.getModelMemoryUsage();
  const tensorMemory = manager.getTensorMemoryUsage();
  
  // Get optimization recommendations
  const recommendations = manager.getOptimizationRecommendations();
  
  // Release model tensors
  const releasedCount = manager.releaseModelTensors("llama");
  
  // Run memory optimization
  const optimizationResults = manager.optimizeMemoryUsage();
  
  // Get final stats
  const stats = manager.getStats();
  
  // Prepare result for demonstration
  const result = {
    "registered_tensors": {
      "text_embedding": textEmbedding.toString(),
      "vision_embedding": visionEmbedding.toString(),
      "embedding_view": embeddingView ? embeddingView.toString() : null
    },
    "sharing_opportunities": opportunities,
    "model_memory_usage": modelMemory,
    "tensor_memory_usage": tensorMemory,
    "optimization_recommendations": recommendations,
    "released_count": releasedCount,
    "optimization_results": optimizationResults,
    "final_stats": stats
  };
  
  return result;
}

// Example usage demonstration
if (typeof window !== 'undefined') {
  console.log("Cross-Model Tensor Sharing Demo");
  console.log("===============================\n");
  
  const demoResults = createTensorSharingDemo();
  
  console.log("Registered Tensors:");
  for (const [name, tensor] of Object.entries(demoResults.registered_tensors)) {
    console.log(`  ${name}: ${tensor}`);
  }
  
  console.log("\nSharing Opportunities:");
  for (const [tensorType, models] of Object.entries(demoResults.sharing_opportunities)) {
    console.log(`  ${tensorType}: ${models}`);
  }
  
  console.log("\nOptimization Recommendations:");
  const recommendations = demoResults.optimization_recommendations;
  console.log(`  Largest tensors: ${JSON.stringify(recommendations.largest_tensors)}`);
  console.log(`  Low reference tensors: ${recommendations.low_reference_tensors}`);
  console.log(`  Total memory: ${recommendations.total_memory_mb.toFixed(2)} MB`);
  console.log(`  Potential savings: ${recommendations.potential_savings_mb.toFixed(2)} MB`);
  
  console.log("\nOptimization Results:");
  const results = demoResults.optimization_results;
  console.log(`  Initial memory: ${(results.initial_memory_bytes / (1024 * 1024)).toFixed(2)} MB`);
  console.log(`  Current memory: ${(results.current_memory_bytes / (1024 * 1024)).toFixed(2)} MB`);
  console.log(`  Memory reduction: ${results.memory_reduction_percent.toFixed(2)}%`);
  console.log(`  Freed tensors: ${results.freed_tensors_count}`);
  
  console.log("\nFinal Stats:");
  const stats = demoResults.final_stats;
  console.log(`  Total tensors: ${stats.total_tensors}`);
  console.log(`  Total models: ${stats.total_models}`);
  console.log(`  Memory usage: ${stats.memory_usage_mb.toFixed(2)} MB`);
  console.log(`  Cache hit rate: ${(stats.hit_rate * 100).toFixed(2)}%`);
}