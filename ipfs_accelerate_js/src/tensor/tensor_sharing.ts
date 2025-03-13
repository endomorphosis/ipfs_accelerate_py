/**
 * Cross-Model Tensor Sharing System
 * 
 * This module implements a tensor sharing system that allows multiple models
 * to efficiently share tensor memory, reducing memory usage and improving
 * performance for multi-model workloads.
 */

import { Tensor } from './tensor';
import { TensorDescriptor, TensorDataType, TensorStorage } from '../types/tensor_types';

/**
 * Tensor sharing types
 */
export type TensorSharingType = 
  | 'text_embedding'      // Text embeddings for NLP models (BERT, T5, LLAMA, BART)
  | 'vision_embedding'    // Vision embeddings for image models (ViT, CLIP, DETR)
  | 'audio_embedding'     // Audio embeddings for speech/audio models (Whisper, Wav2Vec2, CLAP)
  | 'vision_text_joint'   // Joint embeddings for multimodal models (CLIP, LLaVA, BLIP)
  | 'audio_text_joint'    // Joint embeddings for audio-text models (CLAP, Whisper-Text)
  | 'custom'              // Custom tensor sharing type

/**
 * Shared tensor entry
 */
export interface SharedTensorEntry {
  /** Shared tensor instance */
  tensor: Tensor;
  
  /** Models currently sharing this tensor */
  models: Set<string>;
  
  /** Last access timestamp */
  lastAccess: number;
  
  /** Sharing type */
  sharingType: TensorSharingType;
  
  /** Tag for custom categorization */
  tag?: string;
}

/**
 * TensorSharingManager manages shared tensors across multiple models
 */
export class TensorSharingManager {
  /** Map of shared tensors by sharing type */
  private sharedTensors: Map<string, Map<string, SharedTensorEntry>> = new Map();
  
  /** Reference to model IDs using shared tensors */
  private modelTensors: Map<string, Set<string>> = new Map();
  
  /** Auto-cleanup interval */
  private cleanupIntervalId: number | null = null;
  
  /**
   * Constructor
   * @param options Configuration options
   */
  constructor(private options: {
    /** Enable auto-cleanup for unused tensors */
    enableAutoCleanup?: boolean;
    
    /** Interval for auto-cleanup in ms */
    cleanupInterval?: number;
    
    /** Maximum age for unused tensors in ms */
    maxTensorAge?: number;
  } = {}) {
    // Set defaults
    this.options = {
      enableAutoCleanup: true,
      cleanupInterval: 60000, // 1 minute
      maxTensorAge: 300000,   // 5 minutes
      ...options
    };
    
    // Start cleanup interval if enabled
    if (this.options.enableAutoCleanup) {
      this.cleanupIntervalId = window.setInterval(
        () => this.cleanupOldTensors(),
        this.options.cleanupInterval
      );
    }
  }
  
  /**
   * Get or create a shared tensor
   * @param modelId Model requesting the tensor
   * @param sharingType Type of tensor sharing
   * @param descriptor Tensor descriptor for creation if needed
   * @param key Optional key for distinguishing tensors of same type
   * @returns Shared tensor
   */
  getOrCreate(
    modelId: string,
    sharingType: TensorSharingType,
    descriptor: TensorDescriptor,
    key: string = 'default'
  ): Tensor {
    // Create maps if they don't exist
    if (!this.sharedTensors.has(sharingType)) {
      this.sharedTensors.set(sharingType, new Map());
    }
    
    const typeMap = this.sharedTensors.get(sharingType)!;
    const fullKey = `${sharingType}:${key}`;
    
    // Check if tensor already exists
    if (typeMap.has(fullKey)) {
      const entry = typeMap.get(fullKey)!;
      
      // Update access info
      entry.lastAccess = Date.now();
      entry.models.add(modelId);
      
      // Add reference for the model
      this.addModelTensorReference(modelId, fullKey);
      
      // Increment tensor reference count
      entry.tensor.addReference();
      
      return entry.tensor;
    }
    
    // Create new tensor
    const tensor = new Tensor(descriptor);
    
    // Create entry
    const entry: SharedTensorEntry = {
      tensor,
      models: new Set([modelId]),
      lastAccess: Date.now(),
      sharingType
    };
    
    // Store entry
    typeMap.set(fullKey, entry);
    
    // Add reference for the model
    this.addModelTensorReference(modelId, fullKey);
    
    return tensor;
  }
  
  /**
   * Share an existing tensor
   * @param modelId Model sharing the tensor
   * @param tensor Tensor to share
   * @param sharingType Type of tensor sharing
   * @param key Optional key for distinguishing tensors of same type
   * @returns Shared tensor (same instance)
   */
  share(
    modelId: string,
    tensor: Tensor,
    sharingType: TensorSharingType,
    key: string = 'default'
  ): Tensor {
    // Create maps if they don't exist
    if (!this.sharedTensors.has(sharingType)) {
      this.sharedTensors.set(sharingType, new Map());
    }
    
    const typeMap = this.sharedTensors.get(sharingType)!;
    const fullKey = `${sharingType}:${key}`;
    
    // Check if tensor already exists with this key
    if (typeMap.has(fullKey)) {
      // We have a conflict - release the provided tensor and use existing one
      const existingEntry = typeMap.get(fullKey)!;
      
      // Update access info
      existingEntry.lastAccess = Date.now();
      existingEntry.models.add(modelId);
      
      // Add reference for the model
      this.addModelTensorReference(modelId, fullKey);
      
      // Increment tensor reference count for existing tensor
      existingEntry.tensor.addReference();
      
      return existingEntry.tensor;
    }
    
    // Create entry with provided tensor
    const entry: SharedTensorEntry = {
      tensor,
      models: new Set([modelId]),
      lastAccess: Date.now(),
      sharingType
    };
    
    // Store entry
    typeMap.set(fullKey, entry);
    
    // Add reference for the model
    this.addModelTensorReference(modelId, fullKey);
    
    // Increment tensor reference count
    tensor.addReference();
    
    return tensor;
  }
  
  /**
   * Release a shared tensor for a specific model
   * @param modelId Model releasing the tensor
   * @param sharingType Type of tensor sharing
   * @param key Optional key for distinguishing tensors of same type
   */
  release(modelId: string, sharingType: TensorSharingType, key: string = 'default'): void {
    // Get map for sharing type
    const typeMap = this.sharedTensors.get(sharingType);
    if (!typeMap) return;
    
    const fullKey = `${sharingType}:${key}`;
    
    // Check if tensor exists
    const entry = typeMap.get(fullKey);
    if (!entry) return;
    
    // Update model reference
    entry.models.delete(modelId);
    
    // Remove from model tensor references
    this.removeModelTensorReference(modelId, fullKey);
    
    // Decrement tensor reference count
    const shouldDispose = entry.tensor.releaseReference();
    
    // Clean up entry if no models are using it and reference count is 0
    if (entry.models.size === 0 && shouldDispose) {
      entry.tensor.dispose();
      typeMap.delete(fullKey);
      
      // Remove map if empty
      if (typeMap.size === 0) {
        this.sharedTensors.delete(sharingType);
      }
    }
  }
  
  /**
   * Get a shared tensor if it exists
   * @param sharingType Type of tensor sharing
   * @param key Optional key for distinguishing tensors of same type
   * @returns Shared tensor or null if not found
   */
  getTensor(sharingType: TensorSharingType, key: string = 'default'): Tensor | null {
    // Get map for sharing type
    const typeMap = this.sharedTensors.get(sharingType);
    if (!typeMap) return null;
    
    const fullKey = `${sharingType}:${key}`;
    
    // Check if tensor exists
    const entry = typeMap.get(fullKey);
    if (!entry) return null;
    
    // Update access time
    entry.lastAccess = Date.now();
    
    return entry.tensor;
  }
  
  /**
   * Release all tensors for a specific model
   * @param modelId Model ID
   */
  releaseAllForModel(modelId: string): void {
    // Get tensors for model
    const tensorKeys = this.modelTensors.get(modelId);
    if (!tensorKeys) return;
    
    // Copy keys to avoid modification during iteration
    const keys = Array.from(tensorKeys);
    
    // Release each tensor
    for (const fullKey of keys) {
      const [sharingType, key] = fullKey.split(':') as [TensorSharingType, string];
      this.release(modelId, sharingType, key);
    }
    
    // Remove model entry
    this.modelTensors.delete(modelId);
  }
  
  /**
   * Add a tensor reference for a model
   * @param modelId Model ID
   * @param fullKey Full tensor key
   */
  private addModelTensorReference(modelId: string, fullKey: string): void {
    // Create set if it doesn't exist
    if (!this.modelTensors.has(modelId)) {
      this.modelTensors.set(modelId, new Set());
    }
    
    // Add reference
    this.modelTensors.get(modelId)!.add(fullKey);
  }
  
  /**
   * Remove a tensor reference for a model
   * @param modelId Model ID
   * @param fullKey Full tensor key
   */
  private removeModelTensorReference(modelId: string, fullKey: string): void {
    // Get set for model
    const tensorKeys = this.modelTensors.get(modelId);
    if (!tensorKeys) return;
    
    // Remove reference
    tensorKeys.delete(fullKey);
    
    // Remove model entry if empty
    if (tensorKeys.size === 0) {
      this.modelTensors.delete(modelId);
    }
  }
  
  /**
   * Clean up old tensors that haven't been accessed recently
   */
  private cleanupOldTensors(): void {
    const now = Date.now();
    const maxAge = this.options.maxTensorAge!;
    
    // Iterate over all sharing types
    for (const [sharingType, typeMap] of this.sharedTensors.entries()) {
      // Iterate over all entries for this type
      for (const [fullKey, entry] of typeMap.entries()) {
        // Check if tensor is old enough to clean up
        if (now - entry.lastAccess > maxAge) {
          // Clean up tensor if no models are using it
          if (entry.models.size === 0) {
            entry.tensor.dispose();
            typeMap.delete(fullKey);
          }
        }
      }
      
      // Remove map if empty
      if (typeMap.size === 0) {
        this.sharedTensors.delete(sharingType);
      }
    }
  }
  
  /**
   * Get statistics about shared tensors
   */
  getStats(): {
    totalTensors: number;
    totalModels: number;
    byType: Record<string, { tensors: number, models: number }>;
  } {
    let totalTensors = 0;
    let totalModels = 0;
    const modelSet = new Set<string>();
    const byType: Record<string, { tensors: number, models: number }> = {};
    
    // Iterate over all sharing types
    for (const [sharingType, typeMap] of this.sharedTensors.entries()) {
      // Initialize stats for this type
      byType[sharingType] = { tensors: 0, models: 0 };
      const typeModelSet = new Set<string>();
      
      // Count tensors and models for this type
      for (const [_, entry] of typeMap.entries()) {
        byType[sharingType].tensors++;
        totalTensors++;
        
        // Count models
        for (const modelId of entry.models) {
          typeModelSet.add(modelId);
          modelSet.add(modelId);
        }
      }
      
      // Update model count for this type
      byType[sharingType].models = typeModelSet.size;
    }
    
    // Update total model count
    totalModels = modelSet.size;
    
    return {
      totalTensors,
      totalModels,
      byType
    };
  }
  
  /**
   * Dispose of all resources
   */
  dispose(): void {
    // Clear cleanup interval
    if (this.cleanupIntervalId !== null) {
      window.clearInterval(this.cleanupIntervalId);
      this.cleanupIntervalId = null;
    }
    
    // Dispose of all tensors
    for (const [_, typeMap] of this.sharedTensors.entries()) {
      for (const [_, entry] of typeMap.entries()) {
        entry.tensor.dispose();
      }
      typeMap.clear();
    }
    
    // Clear all maps
    this.sharedTensors.clear();
    this.modelTensors.clear();
  }
}

/**
 * Create a tensor sharing manager
 */
export function createTensorSharingManager(options = {}): TensorSharingManager {
  return new TensorSharingManager(options);
}

/**
 * Global default tensor sharing manager instance
 */
export const defaultTensorSharingManager = new TensorSharingManager();
