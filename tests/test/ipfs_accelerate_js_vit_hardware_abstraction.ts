/**
 * Vision Transformer (ViT) implementation using the Hardware Abstraction Layer
 * 
 * This implementation uses the Hardware Abstraction Layer (HAL) to automatically
 * select the optimal backend (WebGPU, WebNN, CPU) for ViT inference based on
 * the available hardware capabilities and model requirements.
 */

import { HardwareAbstraction, HardwareBackend, createHardwareAbstraction } from './ipfs_accelerate_js_hardware_abstraction';
import { StorageManager } from './ipfs_accelerate_js_storage_manager';

/**
 * Configuration options for ViT model
 */
export interface ViTConfig {
  imageSize: number;         // Input image size (224 for ViT-Base)
  patchSize: number;         // Patch size (16 for ViT-Base)
  numLayers: number;         // Number of transformer layers (12 for ViT-Base)
  hiddenSize: number;        // Hidden dimension size (768 for ViT-Base)
  numHeads: number;          // Number of attention heads (12 for ViT-Base)
  mlpDim: number;            // MLP/FFN dimension (3072 for ViT-Base)
  numClasses: number;        // Number of output classes (1000 for ImageNet)
  quantization?: {           // Optional quantization settings
    enabled: boolean;        // Whether to use quantization
    bits: number;            // Quantization bit depth (4 or 8)
    blockSize?: number;      // Quantization block size
  };
  batchSize?: number;        // Batch size for inference (default: 1)
  useOptimizedAttention?: boolean; // Whether to use flash attention (default: true)
  modelId?: string;          // Model ID for storage and caching
}

/**
 * Vision Transformer (ViT) implementation using the Hardware Abstraction Layer
 */
export class HardwareAbstractedVIT {
  private config: ViTConfig;
  private hal: HardwareAbstraction;
  private storageManager: StorageManager;
  private initialized: boolean = false;
  private weights: Map<string, any> = new Map();
  private hardwareCapabilities: any = null;
  
  /**
   * Create a new HAL-accelerated ViT model
   * @param config Model configuration
   * @param storageManager Storage manager for model weights
   */
  constructor(config: ViTConfig, storageManager: StorageManager) {
    this.config = {
      batchSize: 1,
      useOptimizedAttention: true,
      ...config
    };
    this.storageManager = storageManager;
  }
  
  /**
   * Initialize the model and load weights
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;
    
    // Initialize Hardware Abstraction Layer
    this.hal = await createHardwareAbstraction({
      modelPreferences: {
        'vision': 'webgpu',  // Vision models generally perform best on WebGPU
        'default': 'auto'
      },
      backendOptions: {
        'webgpu': {
          enableOptimizations: true,
          precompileShaders: true,
          optimizationLevel: 'maximum'
        },
        'webnn': {
          enableTensorCaching: true,
          preferredDevice: 'gpu'
        },
        'cpu': {
          numThreads: 'auto',
          enableSimd: true
        }
      },
      autoFallback: true,  // Try next best backend if preferred fails
      autoSelection: true  // Use best backend based on model type and hardware
    });
    
    // Get hardware capabilities for logging and optimization
    this.hardwareCapabilities = this.hal.getCapabilities();
    
    // Load model weights from storage
    await this.loadWeights();
    
    this.initialized = true;
    
    console.log(`ViT model initialized with Hardware Abstraction Layer`);
    console.log(`Available backends: ${this.hal.getAvailableBackends().join(', ')}`);
    const bestBackend = this.hal.getBestBackend('vision');
    console.log(`Selected backend for vision: ${bestBackend?.type}`);
  }
  
  /**
   * Load weights from storage
   */
  private async loadWeights(): Promise<void> {
    if (!this.config.modelId) {
      throw new Error('Model ID required for loading weights');
    }
    
    const modelWeights = await this.storageManager.getModelWeights(this.config.modelId);
    if (!modelWeights) {
      throw new Error(`Model weights not found for ${this.config.modelId}`);
    }
    
    // For each weight, create a tensor using the appropriate backend
    for (const [name, tensorData] of Object.entries(modelWeights)) {
      // Choose the best backend based on the tensor type and size
      const backend = this.hal.getBestBackend('vision');
      
      if (!backend) {
        throw new Error('No suitable backend available for tensor creation');
      }
      
      // Create tensor with the chosen backend
      const tensor = await backend.createTensor(
        tensorData.data,
        tensorData.dims,
        tensorData.dtype || 'float32'
      );
      
      // Store tensor with its backend type for later use
      this.weights.set(name, {
        tensor,
        backendType: backend.type
      });
    }
    
    console.log(`Loaded ${this.weights.size} weight tensors for ViT model`);
  }
  
  /**
   * Process input image through the ViT model
   * @param image Input image as Float32Array or HTML image element
   * @returns Class probabilities
   */
  async predict(image: Float32Array | HTMLImageElement): Promise<Float32Array> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    // Convert image to input tensor if it's an HTML element
    let inputData: Float32Array;
    
    if (image instanceof HTMLImageElement) {
      inputData = await this.imageToTensor(image);
    } else {
      inputData = image;
    }
    
    // Get best backend for vision models
    const backend = this.hal.getBestBackend('vision');
    if (!backend) {
      throw new Error('No suitable backend available for ViT inference');
    }
    
    // Create input tensor
    const inputTensor = await backend.createTensor(
      inputData,
      [this.config.batchSize || 1, this.config.imageSize, this.config.imageSize, 3],
      'float32'
    );
    
    // Execute inference through the HAL
    try {
      // Start timing
      console.time('vit_inference');
      
      // 1. Apply patch embedding
      const patchEmbeddings = await this.hal.execute(
        'patchEmbedding',
        {
          input: inputTensor,
          weights: this.weights.get('patch_embedding.weight')?.tensor,
          bias: this.weights.get('patch_embedding.bias')?.tensor
        },
        {
          modelType: 'vision',
          backendOptions: {
            imageSize: this.config.imageSize,
            patchSize: this.config.patchSize,
            hiddenSize: this.config.hiddenSize,
            prependClassToken: true
          }
        }
      );
      
      // 2. Add position embeddings
      const positionEmbeddings = this.weights.get('position_embeddings')?.tensor;
      
      const embeddings = await this.hal.execute(
        'add',
        {
          a: patchEmbeddings,
          b: positionEmbeddings
        },
        { modelType: 'vision' }
      );
      
      // 3. Process through transformer layers
      let layerInput = embeddings;
      
      for (let i = 0; i < this.config.numLayers; i++) {
        // Apply layer norm 1
        const layerNorm1Output = await this.hal.execute(
          'layerNorm',
          {
            input: layerInput,
            weight: this.weights.get(`transformer.layers.${i}.ln_1.weight`)?.tensor,
            bias: this.weights.get(`transformer.layers.${i}.ln_1.bias`)?.tensor
          },
          { 
            modelType: 'vision',
            backendOptions: { epsilon: 1e-5 }
          }
        );
        
        // Apply self-attention
        const attentionOutput = await this.hal.execute(
          'selfAttention',
          {
            input: layerNorm1Output,
            queryWeight: this.weights.get(`transformer.layers.${i}.attn.q_proj`)?.tensor,
            keyWeight: this.weights.get(`transformer.layers.${i}.attn.k_proj`)?.tensor,
            valueWeight: this.weights.get(`transformer.layers.${i}.attn.v_proj`)?.tensor,
            outputWeight: this.weights.get(`transformer.layers.${i}.attn.out_proj`)?.tensor
          },
          {
            modelType: 'vision',
            backendOptions: {
              numHeads: this.config.numHeads,
              hiddenSize: this.config.hiddenSize,
              useOptimizedAttention: this.config.useOptimizedAttention
            }
          }
        );
        
        // Residual connection
        const attentionWithResidual = await this.hal.execute(
          'add',
          {
            a: layerInput,
            b: attentionOutput
          },
          { modelType: 'vision' }
        );
        
        // Apply layer norm 2
        const layerNorm2Output = await this.hal.execute(
          'layerNorm',
          {
            input: attentionWithResidual,
            weight: this.weights.get(`transformer.layers.${i}.ln_2.weight`)?.tensor,
            bias: this.weights.get(`transformer.layers.${i}.ln_2.bias`)?.tensor
          },
          { 
            modelType: 'vision',
            backendOptions: { epsilon: 1e-5 }
          }
        );
        
        // Apply MLP
        const mlpOutput = await this.hal.execute(
          'mlp',
          {
            input: layerNorm2Output,
            fc1Weight: this.weights.get(`transformer.layers.${i}.mlp.fc1`)?.tensor,
            fc2Weight: this.weights.get(`transformer.layers.${i}.mlp.fc2`)?.tensor
          },
          {
            modelType: 'vision',
            backendOptions: {
              hiddenSize: this.config.hiddenSize,
              mlpDim: this.config.mlpDim,
              activationType: 'gelu'
            }
          }
        );
        
        // Residual connection
        layerInput = await this.hal.execute(
          'add',
          {
            a: attentionWithResidual,
            b: mlpOutput
          },
          { modelType: 'vision' }
        );
      }
      
      // 4. Final layer norm
      const finalLayerNorm = await this.hal.execute(
        'layerNorm',
        {
          input: layerInput,
          weight: this.weights.get('transformer.ln_f.weight')?.tensor,
          bias: this.weights.get('transformer.ln_f.bias')?.tensor
        },
        { 
          modelType: 'vision',
          backendOptions: { epsilon: 1e-5 }
        }
      );
      
      // 5. Extract class token (first token)
      const classEmbedding = await this.hal.execute(
        'extractFirstToken',
        {
          input: finalLayerNorm
        },
        { modelType: 'vision' }
      );
      
      // 6. Apply classification head
      const logits = await this.hal.execute(
        'linear',
        {
          input: classEmbedding,
          weight: this.weights.get('classifier.weight')?.tensor,
          bias: this.weights.get('classifier.bias')?.tensor
        },
        { modelType: 'vision' }
      );
      
      // 7. Apply softmax to get probabilities
      const probabilities = await this.hal.execute(
        'softmax',
        {
          input: logits
        },
        { modelType: 'vision' }
      );
      
      // Get result as CPU array
      const result = await backend.execute(
        'getTensorData',
        { input: probabilities }
      ) as Float32Array;
      
      // End timing
      console.timeEnd('vit_inference');
      
      return result;
      
    } catch (error) {
      console.error('Error during ViT inference:', error);
      throw error;
    }
  }
  
  /**
   * Convert an HTMLImageElement to a Float32Array tensor
   */
  private async imageToTensor(image: HTMLImageElement): Promise<Float32Array> {
    const { imageSize } = this.config;
    
    // Create a canvas to resize and process the image
    const canvas = document.createElement('canvas');
    canvas.width = imageSize;
    canvas.height = imageSize;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      throw new Error('Could not get canvas context');
    }
    
    // Draw and resize the image
    ctx.drawImage(image, 0, 0, imageSize, imageSize);
    
    // Get pixel data
    const imageData = ctx.getImageData(0, 0, imageSize, imageSize);
    const pixels = imageData.data;
    
    // Convert RGBA to RGB float array with normalization
    const rgbData = new Float32Array(imageSize * imageSize * 3);
    for (let i = 0; i < pixels.length / 4; i++) {
      // Normalize to [0, 1] and apply preprocessing
      rgbData[i * 3] = (pixels[i * 4] / 255.0 - 0.485) / 0.229;     // R, normalized with ImageNet mean/std
      rgbData[i * 3 + 1] = (pixels[i * 4 + 1] / 255.0 - 0.456) / 0.224; // G, normalized with ImageNet mean/std
      rgbData[i * 3 + 2] = (pixels[i * 4 + 2] / 255.0 - 0.406) / 0.225; // B, normalized with ImageNet mean/std
    }
    
    return rgbData;
  }
  
  /**
   * Get model information
   */
  getModelInfo(): Record<string, any> {
    const bestBackend = this.hal?.getBestBackend('vision');
    
    return {
      modelType: 'ViT',
      imageSize: this.config.imageSize,
      patchSize: this.config.patchSize,
      numLayers: this.config.numLayers,
      hiddenSize: this.config.hiddenSize,
      numHeads: this.config.numHeads,
      mlpDim: this.config.mlpDim,
      numClasses: this.config.numClasses,
      quantization: this.config.quantization,
      selectedBackend: bestBackend?.type || 'unknown',
      availableBackends: this.hal?.getAvailableBackends() || [],
      capabilities: this.hardwareCapabilities
    };
  }
  
  /**
   * Release all resources
   */
  async dispose(): Promise<void> {
    // Clean up tensors
    for (const { tensor, backendType } of this.weights.values()) {
      const backend = this.hal.getBackend(backendType);
      if (backend) {
        await backend.execute('releaseTensor', { tensor });
      }
    }
    
    this.weights.clear();
    
    // Dispose HAL
    this.hal?.dispose();
    
    this.initialized = false;
  }
}