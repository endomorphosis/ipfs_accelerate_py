/**
 * Vision Transformer (ViT) implementation with browser-optimized WebGPU acceleration
 * 
 * This implementation uses the WebGPU Tensor Sharing system with browser-specific
 * optimizations to accelerate the core tensor operations in ViT.
 */

import { TensorStorage, TensorView, TensorDimensions } from './ipfs_accelerate_js_tensor.ts';
import { WebGPUTensorSharing } from './ipfs_accelerate_js_webgpu_tensor_sharing.ts';
import { getOptimizedShader, BrowserCapabilities, ShaderOptimizationSettings } from './ipfs_accelerate_js_browser_optimized_shaders.ts';
import { StorageManager } from './ipfs_accelerate_js_storage_manager.ts';

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
 * Vision Transformer (ViT) implementation with browser-optimized WebGPU acceleration
 */
export class WebGPUOptimizedViT {
  private config: ViTConfig;
  private tensorSharing: WebGPUTensorSharing;
  private storageManager: StorageManager;
  private initialized: boolean = false;
  private weights: Map<string, TensorView> = new Map();
  private browserCapabilities: BrowserCapabilities | null = null;
  private optimizationSettings: ShaderOptimizationSettings | null = null;
  private cachedMatmulBindGroups: Map<string, GPUBindGroup> = new Map();
  private cachedLayerNormBindGroups: Map<string, GPUBindGroup> = new Map();
  private device: GPUDevice | null = null;
  
  // Precompiled pipeline cache
  private patchEmbeddingPipeline: GPUComputePipeline | null = null;
  private matmulPipeline: GPUComputePipeline | null = null;
  private softmaxPipeline: GPUComputePipeline | null = null;
  private layerNormPipeline: GPUComputePipeline | null = null;
  private attentionPipeline: GPUComputePipeline | null = null;
  private mlpPipeline: GPUComputePipeline | null = null;
  
  /**
   * Create a new WebGPU accelerated ViT model
   * @param config Model configuration
   * @param tensorSharing WebGPU tensor sharing instance
   * @param storageManager Storage manager for model weights
   */
  constructor(
    config: ViTConfig, 
    tensorSharing: WebGPUTensorSharing,
    storageManager: StorageManager
  ) {
    this.config = {
      batchSize: 1,
      useOptimizedAttention: true,
      ...config
    };
    this.tensorSharing = tensorSharing;
    this.storageManager = storageManager;
  }
  
  /**
   * Initialize the model and load weights
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;
    
    // Get WebGPU device and browser capabilities
    this.device = this.tensorSharing.getDevice();
    if (!this.device) {
      throw new Error('WebGPU not supported or not initialized');
    }
    
    // Get browser capabilities and optimization settings
    this.browserCapabilities = await this.tensorSharing.getBrowserCapabilities();
    this.optimizationSettings = this.tensorSharing.getOptimizationSettings();
    
    // Load model weights from storage
    await this.loadWeights();
    
    // Precompile pipelines with browser-optimized shaders
    await this.precompilePipelines();
    
    this.initialized = true;
    
    console.log(`ViT model initialized with browser-optimized WebGPU acceleration`);
    console.log(`Browser: ${this.browserCapabilities?.browserType}, GPU: ${this.browserCapabilities?.gpuVendor}`);
    console.log(`Optimization level: ${this.optimizationSettings?.optimizationLevel}`);
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
    
    // Process and upload weights to GPU
    for (const [name, tensorData] of Object.entries(modelWeights)) {
      const tensorView = await this.tensorSharing.createTensorFromData(
        tensorData.data, 
        tensorData.dims as TensorDimensions,
        tensorData.dtype || 'float32'
      );
      
      // Apply quantization if enabled
      if (this.config.quantization?.enabled && 
          name.includes('weight') && 
          !name.includes('embedding')) {
        const quantizedTensor = await this.tensorSharing.quantizeTensor(
          tensorView,
          this.config.quantization.bits,
          this.config.quantization.blockSize || 32
        );
        this.weights.set(name, quantizedTensor);
      } else {
        this.weights.set(name, tensorView);
      }
    }
    
    console.log(`Loaded ${this.weights.size} weight tensors for ViT model`);
  }
  
  /**
   * Precompile compute pipelines with browser-optimized shaders
   */
  private async precompilePipelines(): Promise<void> {
    if (!this.device || !this.browserCapabilities || !this.optimizationSettings) {
      throw new Error('Device or browser capabilities not initialized');
    }
    
    // Patch embedding pipeline
    const patchEmbeddingShader = getOptimizedShader(
      this.device,
      'convolution',
      {
        ...this.optimizationSettings,
        kernelSize: this.config.patchSize,
        stride: this.config.patchSize,
        tensorDims: [this.config.patchSize, this.config.patchSize, 3, this.config.hiddenSize]
      }
    );
    
    this.patchEmbeddingPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({
          code: patchEmbeddingShader
        }),
        entryPoint: 'main'
      }
    });
    
    // Matrix multiplication pipeline (used in attention and MLP)
    const matmulShader = getOptimizedShader(
      this.device,
      'matmul',
      {
        ...this.optimizationSettings,
        precisionLevel: this.config.quantization?.enabled ? 'reduced' : 'high',
        workgroupSize: this.browserCapabilities.optimalWorkgroupSizes.matmul
      }
    );
    
    this.matmulPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({
          code: matmulShader
        }),
        entryPoint: 'main'
      }
    });
    
    // Softmax pipeline (used in attention)
    const softmaxShader = getOptimizedShader(
      this.device,
      'softmax',
      {
        ...this.optimizationSettings,
        workgroupSize: this.browserCapabilities.optimalWorkgroupSizes.reduction
      }
    );
    
    this.softmaxPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({
          code: softmaxShader
        }),
        entryPoint: 'main'
      }
    });
    
    // Layer normalization pipeline
    const layerNormShader = getOptimizedShader(
      this.device,
      'layernorm',
      {
        ...this.optimizationSettings,
        epsilon: 1e-5,
        workgroupSize: this.browserCapabilities.optimalWorkgroupSizes.reduction
      }
    );
    
    this.layerNormPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({
          code: layerNormShader
        }),
        entryPoint: 'main'
      }
    });
    
    // Optimized attention pipeline (if supported)
    if (this.config.useOptimizedAttention) {
      const flashAttentionSupported = this.browserCapabilities.flashAttentionSupported;
      
      const attentionShader = getOptimizedShader(
        this.device,
        flashAttentionSupported ? 'flashAttention' : 'attention',
        {
          ...this.optimizationSettings,
          numHeads: this.config.numHeads,
          headDim: this.config.hiddenSize / this.config.numHeads,
          batchSize: this.config.batchSize || 1,
          useSharedMemory: this.browserCapabilities.sharedMemorySupported
        }
      );
      
      this.attentionPipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: this.device.createShaderModule({
            code: attentionShader
          }),
          entryPoint: 'main'
        }
      });
    }
    
    // MLP pipeline with fused activation
    const mlpShader = getOptimizedShader(
      this.device,
      'mlpWithActivation',
      {
        ...this.optimizationSettings,
        activationType: 'gelu',
        fusedOperations: true,
        workgroupSize: this.browserCapabilities.optimalWorkgroupSizes.elementwise
      }
    );
    
    this.mlpPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({
          code: mlpShader
        }),
        entryPoint: 'main'
      }
    });
    
    console.log('Precompiled all compute pipelines with browser-optimized shaders');
  }
  
  /**
   * Run inference on input images
   * @param images Input images as tensor or array
   * @returns Class probabilities
   */
  async predict(images: TensorView | Float32Array): Promise<Float32Array> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    // Create input tensor if needed
    let inputTensor: TensorView;
    if (images instanceof Float32Array) {
      const batchSize = this.config.batchSize || 1;
      inputTensor = await this.tensorSharing.createTensorFromData(
        images,
        [batchSize, this.config.imageSize, this.config.imageSize, 3],
        'float32'
      );
    } else {
      inputTensor = images;
    }
    
    // Apply patch embedding
    const patchEmbeddings = await this.applyPatchEmbedding(inputTensor);
    
    // Add position embeddings
    const positionEmbeddings = this.weights.get('position_embeddings');
    if (!positionEmbeddings) {
      throw new Error('Position embeddings not found');
    }
    
    // Combine patch and position embeddings
    const embeddings = await this.tensorSharing.executeElementwiseOperation(
      patchEmbeddings,
      positionEmbeddings,
      'add'
    );
    
    // Process through transformer layers
    let layerInput = embeddings;
    for (let i = 0; i < this.config.numLayers; i++) {
      // Apply layer norm 1
      const layerNorm1Output = await this.applyLayerNorm(
        layerInput,
        `transformer.layers.${i}.ln_1.weight`,
        `transformer.layers.${i}.ln_1.bias`
      );
      
      // Apply self-attention
      const attentionOutput = await this.applySelfAttention(
        layerNorm1Output,
        `transformer.layers.${i}.attn.q_proj`,
        `transformer.layers.${i}.attn.k_proj`,
        `transformer.layers.${i}.attn.v_proj`,
        `transformer.layers.${i}.attn.out_proj`
      );
      
      // Residual connection
      const attentionWithResidual = await this.tensorSharing.executeElementwiseOperation(
        layerInput,
        attentionOutput,
        'add'
      );
      
      // Apply layer norm 2
      const layerNorm2Output = await this.applyLayerNorm(
        attentionWithResidual,
        `transformer.layers.${i}.ln_2.weight`,
        `transformer.layers.${i}.ln_2.bias`
      );
      
      // Apply MLP
      const mlpOutput = await this.applyMLP(
        layerNorm2Output,
        `transformer.layers.${i}.mlp.fc1`,
        `transformer.layers.${i}.mlp.fc2`
      );
      
      // Residual connection
      layerInput = await this.tensorSharing.executeElementwiseOperation(
        attentionWithResidual,
        mlpOutput,
        'add'
      );
    }
    
    // Final layer norm
    const finalLayerNorm = await this.applyLayerNorm(
      layerInput,
      'transformer.ln_f.weight',
      'transformer.ln_f.bias'
    );
    
    // Get class embedding (first token)
    const classEmbedding = await this.tensorSharing.extractFirstToken(finalLayerNorm);
    
    // Apply classification head
    const logits = await this.applyClassificationHead(
      classEmbedding,
      'classifier.weight',
      'classifier.bias'
    );
    
    // Convert logits to probabilities
    const probabilities = await this.tensorSharing.executeSoftmax(logits);
    
    // Get result as CPU array
    const result = await this.tensorSharing.getTensorData(probabilities) as Float32Array;
    
    // Clean up tensors
    await this.tensorSharing.releaseTensor(inputTensor);
    await this.tensorSharing.releaseTensor(patchEmbeddings);
    await this.tensorSharing.releaseTensor(embeddings);
    await this.tensorSharing.releaseTensor(layerInput);
    await this.tensorSharing.releaseTensor(finalLayerNorm);
    await this.tensorSharing.releaseTensor(classEmbedding);
    await this.tensorSharing.releaseTensor(logits);
    await this.tensorSharing.releaseTensor(probabilities);
    
    return result;
  }
  
  /**
   * Apply patch embedding to input images
   */
  private async applyPatchEmbedding(inputTensor: TensorView): Promise<TensorView> {
    const patchEmbeddingWeight = this.weights.get('patch_embedding.weight');
    const patchEmbeddingBias = this.weights.get('patch_embedding.bias');
    
    if (!patchEmbeddingWeight || !patchEmbeddingBias) {
      throw new Error('Patch embedding weights not found');
    }
    
    // Use the precompiled patch embedding pipeline
    const outputTensor = await this.tensorSharing.executeCustomPipeline(
      this.patchEmbeddingPipeline!,
      [inputTensor, patchEmbeddingWeight, patchEmbeddingBias],
      [this.config.batchSize || 1, (this.config.imageSize / this.config.patchSize) ** 2 + 1, this.config.hiddenSize],
      {
        workgroupSize: this.browserCapabilities?.optimalWorkgroupSizes.convolution || [8, 8, 1],
        inputLayout: [this.config.batchSize || 1, this.config.imageSize, this.config.imageSize, 3],
        kernelSize: this.config.patchSize,
        stride: this.config.patchSize,
        prependClassToken: true
      }
    );
    
    return outputTensor;
  }
  
  /**
   * Apply layer normalization
   */
  private async applyLayerNorm(
    input: TensorView,
    weightKey: string,
    biasKey: string
  ): Promise<TensorView> {
    const weight = this.weights.get(weightKey);
    const bias = this.weights.get(biasKey);
    
    if (!weight || !bias) {
      throw new Error(`Layer norm weights not found: ${weightKey}, ${biasKey}`);
    }
    
    // Use the browser-optimized layer norm pipeline
    const output = await this.tensorSharing.executeCustomPipeline(
      this.layerNormPipeline!,
      [input, weight, bias],
      input.dims,
      {
        workgroupSize: this.browserCapabilities?.optimalWorkgroupSizes.reduction || [256, 1, 1],
        epsilon: 1e-5,
        axis: -1, // Normalize across hidden dimension
      }
    );
    
    return output;
  }
  
  /**
   * Apply self-attention mechanism
   */
  private async applySelfAttention(
    input: TensorView,
    qWeightKey: string,
    kWeightKey: string,
    vWeightKey: string,
    outWeightKey: string
  ): Promise<TensorView> {
    const qWeight = this.weights.get(qWeightKey);
    const kWeight = this.weights.get(kWeightKey);
    const vWeight = this.weights.get(vWeightKey);
    const outWeight = this.weights.get(outWeightKey);
    
    if (!qWeight || !kWeight || !vWeight || !outWeight) {
      throw new Error('Attention weights not found');
    }
    
    if (this.config.useOptimizedAttention && this.attentionPipeline && this.browserCapabilities?.flashAttentionSupported) {
      // Use optimized flash attention if supported
      return await this.tensorSharing.executeCustomPipeline(
        this.attentionPipeline,
        [input, qWeight, kWeight, vWeight, outWeight],
        input.dims,
        {
          workgroupSize: this.browserCapabilities.optimalWorkgroupSizes.attention || [128, 1, 1],
          numHeads: this.config.numHeads,
          headDim: this.config.hiddenSize / this.config.numHeads,
          seqLen: (this.config.imageSize / this.config.patchSize) ** 2 + 1, // +1 for class token
          batchSize: this.config.batchSize || 1,
          causalMask: false, // ViT uses bidirectional attention
          useSharedMemory: this.browserCapabilities.sharedMemorySupported
        }
      );
    } else {
      // Fallback to standard attention implementation
      
      // Apply query projection
      const queryProjection = await this.tensorSharing.executeMatmul(input, qWeight);
      
      // Apply key projection
      const keyProjection = await this.tensorSharing.executeMatmul(input, kWeight);
      
      // Apply value projection
      const valueProjection = await this.tensorSharing.executeMatmul(input, vWeight);
      
      // Reshape projections for multi-head attention
      const batchSize = this.config.batchSize || 1;
      const seqLen = (this.config.imageSize / this.config.patchSize) ** 2 + 1; // +1 for class token
      const numHeads = this.config.numHeads;
      const headDim = this.config.hiddenSize / numHeads;
      
      const reshapedQuery = await this.tensorSharing.reshapeTensor(
        queryProjection,
        [batchSize, seqLen, numHeads, headDim]
      );
      
      const reshapedKey = await this.tensorSharing.reshapeTensor(
        keyProjection,
        [batchSize, seqLen, numHeads, headDim]
      );
      
      const reshapedValue = await this.tensorSharing.reshapeTensor(
        valueProjection,
        [batchSize, seqLen, numHeads, headDim]
      );
      
      // Transpose key for attention score computation
      const transposedKey = await this.tensorSharing.transposeTensor(
        reshapedKey,
        [0, 2, 1, 3], // [batch, head, seq, dim]
        [0, 2, 3, 1]  // [batch, head, dim, seq]
      );
      
      // Compute attention scores (batch matrix multiplication)
      const attentionScores = await this.tensorSharing.executeCustomPipeline(
        this.matmulPipeline!,
        [reshapedQuery, transposedKey],
        [batchSize, numHeads, seqLen, seqLen],
        {
          transposeA: false,
          transposeB: false,
          workgroupSize: this.browserCapabilities?.optimalWorkgroupSizes.matmul || [8, 8, 1],
          alpha: 1.0 / Math.sqrt(headDim) // Scale by inverse sqrt of head dimension
        }
      );
      
      // Apply softmax to attention scores
      const attentionProbs = await this.tensorSharing.executeCustomPipeline(
        this.softmaxPipeline!,
        [attentionScores],
        attentionScores.dims,
        {
          workgroupSize: this.browserCapabilities?.optimalWorkgroupSizes.reduction || [256, 1, 1],
          axis: -1 // Softmax across sequence dimension
        }
      );
      
      // Apply attention to values (batch matrix multiplication)
      const transposedValue = await this.tensorSharing.transposeTensor(
        reshapedValue,
        [0, 1, 2, 3], // [batch, seq, head, dim]
        [0, 2, 1, 3]  // [batch, head, seq, dim]
      );
      
      const attentionOutput = await this.tensorSharing.executeCustomPipeline(
        this.matmulPipeline!,
        [attentionProbs, transposedValue],
        [batchSize, numHeads, seqLen, headDim],
        {
          transposeA: false,
          transposeB: false,
          workgroupSize: this.browserCapabilities?.optimalWorkgroupSizes.matmul || [8, 8, 1]
        }
      );
      
      // Reshape and transpose back
      const transposedOutput = await this.tensorSharing.transposeTensor(
        attentionOutput,
        [0, 1, 2, 3], // [batch, head, seq, dim]
        [0, 2, 1, 3]  // [batch, seq, head, dim]
      );
      
      const reshapedOutput = await this.tensorSharing.reshapeTensor(
        transposedOutput,
        [batchSize, seqLen, this.config.hiddenSize]
      );
      
      // Apply output projection
      const finalOutput = await this.tensorSharing.executeMatmul(reshapedOutput, outWeight);
      
      // Clean up intermediate tensors
      await this.tensorSharing.releaseTensor(queryProjection);
      await this.tensorSharing.releaseTensor(keyProjection);
      await this.tensorSharing.releaseTensor(valueProjection);
      await this.tensorSharing.releaseTensor(reshapedQuery);
      await this.tensorSharing.releaseTensor(reshapedKey);
      await this.tensorSharing.releaseTensor(reshapedValue);
      await this.tensorSharing.releaseTensor(transposedKey);
      await this.tensorSharing.releaseTensor(attentionScores);
      await this.tensorSharing.releaseTensor(attentionProbs);
      await this.tensorSharing.releaseTensor(transposedValue);
      await this.tensorSharing.releaseTensor(attentionOutput);
      await this.tensorSharing.releaseTensor(transposedOutput);
      await this.tensorSharing.releaseTensor(reshapedOutput);
      
      return finalOutput;
    }
  }
  
  /**
   * Apply MLP with browser-optimized compute shaders
   */
  private async applyMLP(
    input: TensorView,
    fc1WeightKey: string,
    fc2WeightKey: string
  ): Promise<TensorView> {
    const fc1Weight = this.weights.get(fc1WeightKey);
    const fc2Weight = this.weights.get(fc2WeightKey);
    
    if (!fc1Weight || !fc2Weight) {
      throw new Error('MLP weights not found');
    }
    
    // If fused MLP pipeline is available, use it
    if (this.mlpPipeline && this.browserCapabilities?.fusedOperationsSupported) {
      return await this.tensorSharing.executeCustomPipeline(
        this.mlpPipeline,
        [input, fc1Weight, fc2Weight],
        input.dims,
        {
          workgroupSize: this.browserCapabilities.optimalWorkgroupSizes.elementwise || [256, 1, 1],
          hiddenDim: this.config.mlpDim,
          activationType: 'gelu',
          quantized: this.config.quantization?.enabled || false,
          bitsUsed: this.config.quantization?.bits || 32
        }
      );
    } else {
      // Fallback implementation without fused operations
      
      // Linear projection 1
      const fc1Output = await this.tensorSharing.executeMatmul(input, fc1Weight);
      
      // Apply GELU activation
      const geluOutput = await this.tensorSharing.executeElementwiseOperation(
        fc1Output,
        null,
        'gelu'
      );
      
      // Linear projection 2
      const fc2Output = await this.tensorSharing.executeMatmul(geluOutput, fc2Weight);
      
      // Clean up intermediate tensors
      await this.tensorSharing.releaseTensor(fc1Output);
      await this.tensorSharing.releaseTensor(geluOutput);
      
      return fc2Output;
    }
  }
  
  /**
   * Apply classification head to class token
   */
  private async applyClassificationHead(
    classToken: TensorView,
    weightKey: string,
    biasKey: string
  ): Promise<TensorView> {
    const weight = this.weights.get(weightKey);
    const bias = this.weights.get(biasKey);
    
    if (!weight) {
      throw new Error('Classification weight not found');
    }
    
    // Apply linear projection
    const logits = await this.tensorSharing.executeMatmul(classToken, weight);
    
    // Add bias if available
    if (bias) {
      const outputWithBias = await this.tensorSharing.executeElementwiseOperation(
        logits,
        bias,
        'add'
      );
      
      await this.tensorSharing.releaseTensor(logits);
      return outputWithBias;
    }
    
    return logits;
  }
  
  /**
   * Release resources
   */
  async dispose(): Promise<void> {
    // Release weight tensors
    for (const tensor of this.weights.values()) {
      await this.tensorSharing.releaseTensor(tensor);
    }
    this.weights.clear();
    
    // Clear bind group cache
    this.cachedMatmulBindGroups.clear();
    this.cachedLayerNormBindGroups.clear();
    
    this.initialized = false;
  }
  
  /**
   * Get model information
   */
  getModelInfo(): Record<string, any> {
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
      browserOptimized: true,
      browserType: this.browserCapabilities?.browserType || 'unknown',
      gpuVendor: this.browserCapabilities?.gpuVendor || 'unknown',
      optimizationLevel: this.optimizationSettings?.optimizationLevel || 'standard',
      useOptimizedAttention: this.config.useOptimizedAttention,
      flashAttentionSupported: this.browserCapabilities?.flashAttentionSupported || false,
      fusedOperationsSupported: this.browserCapabilities?.fusedOperationsSupported || false,
      sharedMemorySupported: this.browserCapabilities?.sharedMemorySupported || false
    };
  }
}