/**
 * Hardware Abstracted CLIP Implementation
 * Implements CLIP model with automatic hardware acceleration selection and optimization
 */

import { HardwareBackend } from '../../hardware/interfaces/hardware_backend';
import { createOptimalBackend, createMultiBackend } from '../../hardware/index';
import { Tensor } from '../../tensor/tensor';
import { SharedTensor } from '../../tensor/shared_tensor';
import { matmul, softmax } from '../../tensor/operations/matrix';
import { layerNorm, gelu } from '../../tensor/operations/nn';
import { add, mul } from '../../tensor/operations/basic';
import { Clip as CLIPImplementation, ClipConfig, ClipImageInput, ClipTextInput, ClipOutput } from '../vision/clip';

/**
 * Storage Manager Interface for model weights storage
 */
export interface StorageManager {
  initialize(): Promise<void>;
  getItem(key: string): Promise<any>;
  setItem(key: string, value: any): Promise<void>;
  hasItem(key: string): Promise<boolean>;
  removeItem(key: string): Promise<void>;
}

/**
 * Configuration for Hardware Abstracted CLIP
 */
export interface HardwareAbstractedCLIPConfig extends ClipConfig {
  /**
   * Quantization settings
   */
  quantization?: {
    enabled: boolean;
    bits: number;
    blockSize?: number;
  };
  
  /**
   * Backend preference order
   */
  backendPreference?: ('webgpu' | 'webnn' | 'wasm' | 'cpu')[];
  
  /**
   * Whether to allow automatic fallback to next backend
   */
  allowFallback?: boolean;
  
  /**
   * Whether to collect performance metrics
   */
  collectMetrics?: boolean;
  
  /**
   * Browser-specific optimizations
   */
  browserOptimizations?: boolean;
  
  /**
   * Task type - determines output format and processing
   */
  taskType?: 'image_embedding' | 'text_embedding' | 'zero_shot_classification' | 'similarity';
}

/**
 * Default Hardware Abstracted CLIP Configuration (clip-vit-base-patch32)
 */
export const DEFAULT_HARDWARE_ABSTRACTED_CLIP_CONFIG: HardwareAbstractedCLIPConfig = {
  modelId: 'openai/clip-vit-base-patch32',
  imageSize: 224,
  patchSize: 32,
  hiddenSize: 768,
  projectionDim: 512,
  visionNumLayers: 12,
  textNumLayers: 12,
  visionNumHeads: 12,
  textNumHeads: 12,
  visionIntermediateSize: 3072,
  textIntermediateSize: 3072,
  layerNormEps: 1e-5,
  maxTextLength: 77,
  backendPreference: ['webgpu', 'webnn', 'cpu'],
  useOptimizedOps: true,
  useBrowserOptimizations: true,
  useOperationFusion: true,
  allowFallback: true,
  collectMetrics: true,
  browserOptimizations: true,
  taskType: 'similarity',
  quantization: {
    enabled: false,
    bits: 8
  }
};

/**
 * Performance metric type
 */
interface PerformanceMetric {
  avg: number;
  min: number;
  max: number;
  count: number;
  total: number;
}

/**
 * Hardware Abstracted CLIP class
 * Provides hardware optimization for CLIP model
 */
export class HardwareAbstractedCLIP {
  private config: HardwareAbstractedCLIPConfig;
  private storageManager: StorageManager;
  private hardware: HardwareBackend | null = null;
  private modelImpl: CLIPImplementation | null = null;
  private initialized: boolean = false;
  
  // Metrics tracking
  private performanceMetrics: Record<string, PerformanceMetric> = {};
  private backendMetrics: Record<string, any> = {};
  private selectedBackend: string = '';
  private availableBackends: string[] = [];
  
  /**
   * Constructor for Hardware Abstracted CLIP
   * @param config Configuration options
   * @param storageManager Storage manager for model weights
   */
  constructor(
    config: Partial<HardwareAbstractedCLIPConfig> = {},
    storageManager: StorageManager
  ) {
    this.config = { ...DEFAULT_HARDWARE_ABSTRACTED_CLIP_CONFIG, ...config };
    this.storageManager = storageManager;
  }
  
  /**
   * Initialize the model
   * Sets up hardware backend and model implementation
   */
  public async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }
    
    try {
      const startTime = performance.now();
      
      // Initialize hardware backend
      await this.initializeHardware();
      
      // Initialize model implementation
      if (this.hardware) {
        this.modelImpl = new CLIPImplementation(this.hardware, this.config);
        await this.modelImpl.initialize();
      } else {
        throw new Error('Failed to initialize hardware backend');
      }
      
      const initTime = performance.now() - startTime;
      this.updateMetric('initialization', initTime);
      
      this.initialized = true;
      console.log(`CLIP model initialized (${this.selectedBackend}) in ${initTime.toFixed(2)}ms`);
    } catch (error) {
      console.error('Failed to initialize Hardware Abstracted CLIP:', error);
      throw error;
    }
  }
  
  /**
   * Initialize hardware backend
   * Selects optimal backend based on config preferences
   */
  private async initializeHardware(): Promise<void> {
    // Collect available backends for metrics
    this.availableBackends = this.getAvailableBackends();
    
    try {
      // For CLIP, we generally prefer WebGPU for vision processing
      // But for Firefox, Chrome, and Edge, there are different performance characteristics
      const isFirefox = typeof navigator !== 'undefined' && navigator.userAgent.toLowerCase().includes('firefox');
      const isChrome = typeof navigator !== 'undefined' && (navigator.userAgent.toLowerCase().includes('chrome') || navigator.userAgent.toLowerCase().includes('chromium'));
      const isEdge = typeof navigator !== 'undefined' && navigator.userAgent.toLowerCase().includes('edg');
      
      // Try to initialize with optimal backend
      if (this.config.backendPreference && this.config.backendPreference.length > 0) {
        // Use multi-backend with specified preference order
        this.hardware = await createMultiBackend(this.config.backendPreference);
      } else if (isChrome) {
        // Chrome has generally good all-around WebGPU support
        this.hardware = await createOptimalBackend({
          forceBackend: 'webgpu',
          optimizationLevel: 'maximum'
        });
      } else {
        // Use automatic optimal backend selection
        this.hardware = await createOptimalBackend({
          preferComputePerformance: true, // CLIP benefits from compute performance
          modelType: 'vision'             // Specify this is a vision model
        });
      }
      
      if (!this.hardware) {
        throw new Error('Failed to initialize hardware backend');
      }
      
      // Store selected backend type
      this.selectedBackend = this.hardware.type;
      
      // Record backend capabilities
      this.backendMetrics = {
        type: this.hardware.type,
        capabilities: this.hardware.capabilities,
        isAvailable: this.hardware.isAvailable,
        browserInfo: typeof navigator !== 'undefined' ? {
          userAgent: navigator.userAgent,
          platform: navigator.platform,
          isFirefox: isFirefox,
          isChrome: isChrome,
          isEdge: isEdge
        } : 'unknown'
      };
      
      console.log(`Selected backend: ${this.selectedBackend}`);
    } catch (error) {
      console.error('Error initializing hardware backend:', error);
      
      if (this.config.allowFallback) {
        console.warn('Falling back to CPU backend');
        // Fallback to CPU backend
        this.hardware = await createOptimalBackend({ forceBackend: 'cpu' });
        if (this.hardware) {
          this.selectedBackend = this.hardware.type;
          console.log(`Fallback to ${this.selectedBackend} successful`);
        } else {
          throw new Error('Failed to initialize fallback CPU backend');
        }
      } else {
        throw error;
      }
    }
  }
  
  /**
   * Get list of available backends
   * Depends on browser capabilities
   */
  private getAvailableBackends(): string[] {
    const backends: string[] = ['cpu']; // CPU is always available
    
    // Check for WebGPU
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      backends.push('webgpu');
    }
    
    // Check for WebNN
    if (typeof navigator !== 'undefined' && 'ml' in navigator) {
      backends.push('webnn');
    }
    
    return backends;
  }
  
  /**
   * Preprocess image for CLIP model
   * @param image Image input
   * @returns Preprocessed input
   */
  public async preprocessImage(image: HTMLImageElement | ImageData | {
    data: Uint8Array | Float32Array;
    width: number;
    height: number;
  }): Promise<ClipImageInput> {
    const startTime = performance.now();
    
    // Convert to ClipImageInput format
    let clipImageInput: ClipImageInput;
    
    if (image instanceof HTMLImageElement) {
      // Convert HTMLImageElement to ImageData
      const canvas = document.createElement('canvas');
      canvas.width = image.width;
      canvas.height = image.height;
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        throw new Error('Failed to get canvas context');
      }
      
      // Draw image to canvas
      ctx.drawImage(image, 0, 0);
      
      // Get pixel data
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      // Convert to ClipImageInput
      clipImageInput = {
        imageData: new Uint8Array(imageData.data),
        width: imageData.width,
        height: imageData.height
      };
    } else if ('data' in image && 'width' in image && 'height' in image) {
      // Already in compatible format
      clipImageInput = {
        imageData: image.data,
        width: image.width,
        height: image.height
      };
    } else {
      // Must be an ImageData
      const imageData = image as ImageData;
      clipImageInput = {
        imageData: new Uint8Array(imageData.data),
        width: imageData.width,
        height: imageData.height
      };
    }
    
    // Resize image if needed
    if (clipImageInput.width !== this.config.imageSize || clipImageInput.height !== this.config.imageSize) {
      // In a real implementation, we would resize the image
      // Here we'll just throw an error for now
      throw new Error(`Image must be resized to ${this.config.imageSize}x${this.config.imageSize} before processing`);
    }
    
    const preprocessTime = performance.now() - startTime;
    this.updateMetric('image_preprocessing', preprocessTime);
    
    return clipImageInput;
  }
  
  /**
   * Execute CLIP model with image input
   * @param image Image input
   * @returns Model output (image embeddings)
   */
  public async encodeImage(image: HTMLImageElement | ImageData | ClipImageInput | {
    data: Uint8Array | Float32Array;
    width: number;
    height: number;
  }): Promise<Float32Array> {
    if (!this.initialized || !this.modelImpl) {
      throw new Error('Model not initialized. Call initialize() first.');
    }
    
    const totalStartTime = performance.now();
    
    try {
      // Preprocess image if needed
      let clipImageInput: ClipImageInput;
      
      if ('imageData' in image && 'width' in image && 'height' in image) {
        // Already in ClipImageInput format
        clipImageInput = image as ClipImageInput;
      } else {
        // Convert to ClipImageInput
        clipImageInput = await this.preprocessImage(image);
      }
      
      // Run model inference
      const inferenceStartTime = performance.now();
      const output = await this.modelImpl.process({ image: clipImageInput });
      const inferenceTime = performance.now() - inferenceStartTime;
      
      this.updateMetric('inference_time', inferenceTime);
      
      // Extract image embeddings
      if (!output.imageEmbeddings) {
        throw new Error('No image embeddings in model output');
      }
      
      // Create shared tensor for cross-model sharing
      if (this.modelImpl) {
        const sharedTensor = this.modelImpl.getSharedTensor('vision_embedding');
        if (sharedTensor) {
          this.updateMetric('image_embedding_size', sharedTensor.tensor.data.length * 4); // Size in bytes
        }
      }
      
      const totalTime = performance.now() - totalStartTime;
      this.updateMetric('total_image_processing', totalTime);
      
      return new Float32Array(output.imageEmbeddings);
    } catch (error) {
      console.error('Error during CLIP image encoding:', error);
      throw error;
    }
  }
  
  /**
   * Execute CLIP model with text input
   * @param text Text input
   * @returns Model output (text embeddings)
   */
  public async encodeText(text: string | ClipTextInput): Promise<Float32Array> {
    if (!this.initialized || !this.modelImpl) {
      throw new Error('Model not initialized. Call initialize() first.');
    }
    
    const totalStartTime = performance.now();
    
    try {
      // Prepare text input
      let clipTextInput: ClipTextInput;
      
      if (typeof text === 'string') {
        clipTextInput = { text };
      } else {
        clipTextInput = text;
      }
      
      // Run model inference
      const inferenceStartTime = performance.now();
      const output = await this.modelImpl.process({ text: clipTextInput });
      const inferenceTime = performance.now() - inferenceStartTime;
      
      this.updateMetric('inference_time', inferenceTime);
      
      // Extract text embeddings
      if (!output.textEmbeddings) {
        throw new Error('No text embeddings in model output');
      }
      
      // Create shared tensor for cross-model sharing
      if (this.modelImpl) {
        const sharedTensor = this.modelImpl.getSharedTensor('text_embedding');
        if (sharedTensor) {
          this.updateMetric('text_embedding_size', sharedTensor.tensor.data.length * 4); // Size in bytes
        }
      }
      
      const totalTime = performance.now() - totalStartTime;
      this.updateMetric('total_text_processing', totalTime);
      
      return new Float32Array(output.textEmbeddings);
    } catch (error) {
      console.error('Error during CLIP text encoding:', error);
      throw error;
    }
  }
  
  /**
   * Compute similarity between image and text
   * @param image Image input
   * @param text Text input
   * @returns Similarity score (higher means more similar)
   */
  public async computeSimilarity(
    image: HTMLImageElement | ImageData | ClipImageInput | {
      data: Uint8Array | Float32Array;
      width: number;
      height: number;
    },
    text: string | ClipTextInput
  ): Promise<number> {
    if (!this.initialized || !this.modelImpl) {
      throw new Error('Model not initialized. Call initialize() first.');
    }
    
    const totalStartTime = performance.now();
    
    try {
      // Prepare inputs
      let clipImageInput: ClipImageInput;
      let clipTextInput: ClipTextInput;
      
      if ('imageData' in image && 'width' in image && 'height' in image) {
        // Already in ClipImageInput format
        clipImageInput = image as ClipImageInput;
      } else {
        // Convert to ClipImageInput
        clipImageInput = await this.preprocessImage(image);
      }
      
      if (typeof text === 'string') {
        clipTextInput = { text };
      } else {
        clipTextInput = text;
      }
      
      // Run model inference
      const inferenceStartTime = performance.now();
      const output = await this.modelImpl.process({
        image: clipImageInput,
        text: clipTextInput
      });
      const inferenceTime = performance.now() - inferenceStartTime;
      
      this.updateMetric('inference_time', inferenceTime);
      
      // Extract similarity score
      if (output.similarity === undefined) {
        throw new Error('No similarity score in model output');
      }
      
      const totalTime = performance.now() - totalStartTime;
      this.updateMetric('total_similarity_processing', totalTime);
      
      return output.similarity;
    } catch (error) {
      console.error('Error during CLIP similarity computation:', error);
      throw error;
    }
  }
  
  /**
   * Run zero-shot classification on an image
   * @param image Image input
   * @param classes Array of class labels to classify image into
   * @returns Array of {label, score} pairs, sorted by score descending
   */
  public async classifyImage(
    image: HTMLImageElement | ImageData | ClipImageInput | {
      data: Uint8Array | Float32Array;
      width: number;
      height: number;
    },
    classes: string[]
  ): Promise<Array<{label: string, score: number}>> {
    if (!this.initialized || !this.modelImpl) {
      throw new Error('Model not initialized. Call initialize() first.');
    }
    
    const totalStartTime = performance.now();
    
    try {
      // First encode the image
      let clipImageInput: ClipImageInput;
      
      if ('imageData' in image && 'width' in image && 'height' in image) {
        // Already in ClipImageInput format
        clipImageInput = image as ClipImageInput;
      } else {
        // Convert to ClipImageInput
        clipImageInput = await this.preprocessImage(image);
      }
      
      // Process image
      const imageEncodingStartTime = performance.now();
      const imageOutput = await this.modelImpl.process({ image: clipImageInput });
      const imageEncodingTime = performance.now() - imageEncodingStartTime;
      
      this.updateMetric('image_encoding_time', imageEncodingTime);
      
      // Extract image embeddings
      if (!imageOutput.imageEmbeddings) {
        throw new Error('No image embeddings in model output');
      }
      
      // Encode each class label with template: "a photo of a {class}"
      const results: Array<{label: string, score: number}> = [];
      
      for (const className of classes) {
        const prompt = `a photo of a ${className}`;
        
        // Encode text
        const textEncodingStartTime = performance.now();
        const textOutput = await this.modelImpl.process({ text: { text: prompt } });
        const textEncodingTime = performance.now() - textEncodingStartTime;
        
        this.updateMetric('text_encoding_time', textEncodingTime);
        
        // Extract text embeddings
        if (!textOutput.textEmbeddings) {
          console.warn(`No text embeddings for class "${className}"`);
          continue;
        }
        
        // Compute similarity
        const similarityStartTime = performance.now();
        const fullOutput = await this.modelImpl.process({
          image: clipImageInput,
          text: { text: prompt }
        });
        const similarityTime = performance.now() - similarityStartTime;
        
        this.updateMetric('similarity_computation_time', similarityTime);
        
        // Extract similarity score
        if (fullOutput.similarity === undefined) {
          console.warn(`No similarity score for class "${className}"`);
          continue;
        }
        
        // Add to results
        results.push({
          label: className,
          score: fullOutput.similarity
        });
      }
      
      // Sort results by score descending
      results.sort((a, b) => b.score - a.score);
      
      const totalTime = performance.now() - totalStartTime;
      this.updateMetric('total_classification_time', totalTime);
      
      return results;
    } catch (error) {
      console.error('Error during CLIP zero-shot classification:', error);
      throw error;
    }
  }
  
  /**
   * Run performance comparison across all available backends
   * @param image Test image
   * @param text Test text
   * @returns Execution time by backend
   */
  public async compareBackends(
    image: ClipImageInput,
    text: string
  ): Promise<Record<string, number>> {
    const results: Record<string, number> = {};
    const availableBackends = this.getAvailableBackends();
    
    // Test each backend
    for (const backendType of availableBackends) {
      try {
        console.log(`Testing ${backendType} backend...`);
        
        // Create backend
        const hardware = await createOptimalBackend({ forceBackend: backendType as any });
        
        if (!hardware) {
          console.warn(`Backend ${backendType} initialization failed`);
          results[backendType] = -1; // Mark as failed
          continue;
        }
        
        // Create model
        const model = new CLIPImplementation(hardware, this.config);
        await model.initialize();
        
        // Warm-up run
        await model.process({ image, text: { text } });
        
        // Timed run
        const startTime = performance.now();
        await model.process({ image, text: { text } });
        const endTime = performance.now();
        
        // Record result
        results[backendType] = endTime - startTime;
        
        // Clean up
        try {
          await model.dispose();
          await hardware.dispose();
        } catch (e) {
          console.warn(`Error disposing resources for ${backendType}:`, e);
        }
      } catch (error) {
        console.error(`Error testing ${backendType} backend:`, error);
        results[backendType] = -1; // Mark as failed
      }
    }
    
    console.log('Backend comparison results:', results);
    return results;
  }
  
  /**
   * Get performance metrics collected during execution
   * @returns Performance metrics by operation
   */
  public getPerformanceMetrics(): Record<string, PerformanceMetric> {
    return this.performanceMetrics;
  }
  
  /**
   * Get backend-specific metrics and capabilities
   * @returns Backend metrics
   */
  public getBackendMetrics(): Record<string, any> {
    return this.backendMetrics;
  }
  
  /**
   * Get model information
   * @returns Model info
   */
  public getModelInfo(): Record<string, any> {
    return {
      modelId: this.config.modelId,
      modelType: 'CLIP',
      taskType: this.config.taskType,
      hiddenSize: this.config.hiddenSize,
      projectionDim: this.config.projectionDim,
      selectedBackend: this.selectedBackend,
      availableBackends: this.availableBackends,
      imageSize: this.config.imageSize,
      visionLayers: this.config.visionNumLayers,
      textLayers: this.config.textNumLayers,
      quantization: this.config.quantization?.enabled
        ? `${this.config.quantization.bits}-bit`
        : 'none'
    };
  }
  
  /**
   * Update a performance metric with a new timing value
   * @param name Metric name
   * @param value Timing value
   */
  private updateMetric(name: string, value: number): void {
    if (!this.config.collectMetrics) {
      return;
    }
    
    if (!this.performanceMetrics[name]) {
      this.performanceMetrics[name] = {
        avg: value,
        min: value,
        max: value,
        count: 1,
        total: value
      };
      return;
    }
    
    const metric = this.performanceMetrics[name];
    metric.count++;
    metric.total += value;
    metric.avg = metric.total / metric.count;
    metric.min = Math.min(metric.min, value);
    metric.max = Math.max(metric.max, value);
  }
  
  /**
   * Release all resources
   */
  public async dispose(): Promise<void> {
    if (this.modelImpl) {
      await this.modelImpl.dispose();
      this.modelImpl = null;
    }
    
    if (this.hardware) {
      await this.hardware.dispose();
      this.hardware = null;
    }
    
    this.initialized = false;
  }
  
  /**
   * Get a shared tensor from the model
   * @param outputType Type of output to get (e.g. 'vision_embedding', 'text_embedding')
   * @returns Shared tensor or null if not found
   */
  public getSharedTensor(outputType: string = 'vision_embedding'): SharedTensor | null {
    if (!this.modelImpl) {
      return null;
    }
    
    return this.modelImpl.getSharedTensor(outputType);
  }
  
  /**
   * Reset performance metrics
   */
  public resetMetrics(): void {
    this.performanceMetrics = {};
  }
}

/**
 * Factory function to create a Hardware Abstracted CLIP
 * @param config Configuration options
 * @param storageManager Storage manager for model weights
 * @returns HardwareAbstractedCLIP instance
 */
export function createHardwareAbstractedCLIP(
  config: Partial<HardwareAbstractedCLIPConfig> = {},
  storageManager: StorageManager
): HardwareAbstractedCLIP {
  return new HardwareAbstractedCLIP(config, storageManager);
}