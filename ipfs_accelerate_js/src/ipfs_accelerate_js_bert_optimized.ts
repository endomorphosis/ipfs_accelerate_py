/**
 * BERT (Bidirectional Encoder Representations from Transformers) implementation 
 * with browser-optimized WebGPU acceleration
 * 
 * This implementation uses the WebGPU Tensor Sharing system with browser-specific
 * optimizations to accelerate the core tensor operations in BERT.
 */

import { TensorStorage, TensorView, TensorDimensions } from './ipfs_accelerate_js_tensor.ts';
import { WebGPUTensorSharing } from './ipfs_accelerate_js_webgpu_tensor_sharing.ts';
import { getOptimizedShader, BrowserCapabilities, ShaderOptimizationSettings } from './ipfs_accelerate_js_browser_optimized_shaders.ts';
import { StorageManager } from './ipfs_accelerate_js_storage_manager.ts';

/**
 * Configuration options for BERT model
 */
export interface BERTConfig {
  vocabSize: number;         // Vocabulary size (30522 for BERT Base)
  hiddenSize: number;        // Hidden dimension size (768 for BERT Base)
  numLayers: number;         // Number of transformer layers (12 for BERT Base)
  numHeads: number;          // Number of attention heads (12 for BERT Base)
  intermediateSize: number;  // Intermediate feed-forward dimension (3072 for BERT Base)
  maxSequenceLength: number; // Maximum sequence length (512 for BERT Base)
  quantization?: {           // Optional quantization settings
    enabled: boolean;        // Whether to use quantization
    bits: number;            // Quantization bit depth (4 or 8)
    blockSize?: number;      // Quantization block size
  };
  useOptimizedAttention?: boolean; // Whether to use flash attention (default: true)
  modelId?: string;          // Model ID for storage and caching
  taskType?: 'embedding' | 'sequence_classification' | 'token_classification' | 'question_answering';
  numLabels?: number;        // Number of output labels for classification tasks
}

/**
 * BERT token embedding
 */
interface BERTEmbedding {
  inputIds: number[];
  tokenTypeIds?: number[];
  attentionMask: number[];
}

/**
 * BERT output type (varies based on task)
 */
type BERTOutput = 
  | Float32Array                // For embeddings
  | { logits: Float32Array }    // For classification
  | { start_logits: Float32Array; end_logits: Float32Array } // For QA

/**
 * BERT implementation with browser-optimized WebGPU acceleration
 */
export class WebGPUOptimizedBERT {
  private config: BERTConfig;
  private tensorSharing: WebGPUTensorSharing;
  private storageManager: StorageManager;
  private initialized: boolean = false;
  private weights: Map<string, TensorView> = new Map();
  private browserCapabilities: BrowserCapabilities | null = null;
  private optimizationSettings: ShaderOptimizationSettings | null = null;
  private device: GPUDevice | null = null;
  
  // Precompiled pipeline cache
  private embeddingPipeline: GPUComputePipeline | null = null;
  private matmulPipeline: GPUComputePipeline | null = null;
  private softmaxPipeline: GPUComputePipeline | null = null;
  private layerNormPipeline: GPUComputePipeline | null = null;
  private attentionPipeline: GPUComputePipeline | null = null;
  private geluPipeline: GPUComputePipeline | null = null;
  private intermediateMatmulPipeline: GPUComputePipeline | null = null;
  private outputMatmulPipeline: GPUComputePipeline | null = null;
  
  // Performance tracking
  private opExecutionTimes: Map<string, number[]> = new Map();
  
  /**
   * Create a new WebGPU accelerated BERT model
   * @param config Model configuration
   * @param tensorSharing WebGPU tensor sharing instance
   * @param storageManager Storage manager for model weights
   */
  constructor(
    config: BERTConfig, 
    tensorSharing: WebGPUTensorSharing,
    storageManager: StorageManager
  ) {
    this.config = {
      maxSequenceLength: 512,
      useOptimizedAttention: true,
      taskType: 'embedding',
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
    
    console.log(`Initializing BERT model with browser-optimized WebGPU acceleration`);
    console.log(`Browser: ${this.browserCapabilities?.browserType}, GPU: ${this.browserCapabilities?.gpuVendor}`);
    console.log(`Optimization level: ${this.optimizationSettings?.optimizationLevel}`);
    
    // Load model weights from storage
    await this.loadWeights();
    
    // Precompile pipelines with browser-optimized shaders
    await this.precompilePipelines();
    
    this.initialized = true;
  }
  
  /**
   * Load weights from storage
   */
  private async loadWeights(): Promise<void> {
    if (!this.config.modelId) {
      throw new Error('Model ID required for loading weights');
    }
    
    console.log(`Loading weights for model: ${this.config.modelId}`);
    
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
      
      // Apply quantization if enabled to weight matrices
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
    
    console.log(`Loaded ${this.weights.size} weight tensors for BERT model`);
  }
  
  /**
   * Precompile compute pipelines with browser-optimized shaders
   */
  private async precompilePipelines(): Promise<void> {
    if (!this.device || !this.browserCapabilities || !this.optimizationSettings) {
      throw new Error('Device or browser capabilities not initialized');
    }
    
    const startTime = performance.now();
    console.log("Precompiling pipelines for BERT model...");
    
    // Embedding lookup pipeline
    const embeddingShader = getOptimizedShader(
      this.device,
      'embedding',
      {
        ...this.optimizationSettings,
        vocabSize: this.config.vocabSize,
        embeddingDim: this.config.hiddenSize
      }
    );
    
    this.embeddingPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({
          code: embeddingShader
        }),
        entryPoint: 'main'
      }
    });
    
    // Matrix multiplication pipeline (used in attention and feed-forward)
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
    
    // Specialized intermediate matmul pipeline for feed-forward network
    const intermediateMatmulShader = getOptimizedShader(
      this.device,
      'matmul',
      {
        ...this.optimizationSettings,
        precisionLevel: this.config.quantization?.enabled ? 'reduced' : 'high',
        workgroupSize: this.browserCapabilities.optimalWorkgroupSizes.matmul,
        specializedFor: 'bert_intermediate'
      }
    );
    
    this.intermediateMatmulPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({
          code: intermediateMatmulShader
        }),
        entryPoint: 'main'
      }
    });
    
    // Specialized output matmul pipeline for feed-forward network
    const outputMatmulShader = getOptimizedShader(
      this.device,
      'matmul',
      {
        ...this.optimizationSettings,
        precisionLevel: this.config.quantization?.enabled ? 'reduced' : 'high',
        workgroupSize: this.browserCapabilities.optimalWorkgroupSizes.matmul,
        specializedFor: 'bert_output'
      }
    );
    
    this.outputMatmulPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({
          code: outputMatmulShader
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
        workgroupSize: this.browserCapabilities.optimalWorkgroupSizes.reduction,
        attentionMaskSupport: true // Specialized for BERT attention masking
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
        epsilon: 1e-12, // BERT uses 1e-12 for layer norm
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
          maxSequenceLength: this.config.maxSequenceLength,
          useSharedMemory: this.browserCapabilities.sharedMemorySupported,
          attentionMaskSupport: true,
          causalMask: false, // BERT uses bidirectional attention
          specializedFor: 'bert'
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
    
    // GELU activation pipeline for feed-forward network
    const geluShader = getOptimizedShader(
      this.device,
      'gelu',
      {
        ...this.optimizationSettings,
        workgroupSize: this.browserCapabilities.optimalWorkgroupSizes.elementwise,
        approximation: 'tanh' // BERT typically uses tanh approximation
      }
    );
    
    this.geluPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({
          code: geluShader
        }),
        entryPoint: 'main'
      }
    });
    
    const endTime = performance.now();
    console.log(`Precompiled all compute pipelines in ${(endTime - startTime).toFixed(2)}ms`);
  }
  
  /**
   * Tokenize input text (simplified example - in a real implementation, 
   * we would use a proper tokenizer)
   * @param text Input text
   * @returns Tokenized input
   */
  tokenize(text: string): BERTEmbedding {
    // This is a simplified tokenization for demonstration
    // In real implementation, we would use a proper WordPiece tokenizer
    
    // Mock implementation returning random token IDs
    const length = Math.min(text.length, this.config.maxSequenceLength - 2); // -2 for [CLS] and [SEP]
    
    const inputIds = [101]; // [CLS] token
    for (let i = 0; i < length; i++) {
      // Generate a random token ID between 1000 and 30000
      // This is just a mock implementation
      inputIds.push(1000 + Math.floor(Math.random() * 29000));
    }
    inputIds.push(102); // [SEP] token
    
    // Create attention mask (1 for all tokens)
    const attentionMask = new Array(inputIds.length).fill(1);
    
    // Create token type IDs (all 0 for single sequence)
    const tokenTypeIds = new Array(inputIds.length).fill(0);
    
    return {
      inputIds,
      tokenTypeIds,
      attentionMask
    };
  }
  
  /**
   * Run inference on input text
   * @param text Input text or pre-tokenized input
   * @returns Model output based on task type
   */
  async predict(text: string | BERTEmbedding): Promise<BERTOutput> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    // Tokenize input if needed
    const tokens = typeof text === 'string' ? this.tokenize(text) : text;
    
    // Create input tensors
    const inputTensors = await this.createInputTensors(tokens);
    
    // Get embeddings
    const embeddings = await this.getEmbeddings(
      inputTensors.inputIds,
      inputTensors.tokenTypeIds,
      inputTensors.positionIds
    );
    
    // Apply attention mask to embeddings if provided
    const attentionMask = inputTensors.attentionMask;
    
    // Process through transformer layers
    let layerOutput = embeddings;
    for (let i = 0; i < this.config.numLayers; i++) {
      layerOutput = await this.processLayer(
        layerOutput,
        attentionMask,
        i
      );
    }
    
    // Get output based on task type
    const output = await this.getTaskOutput(layerOutput);
    
    // Clean up tensors
    await this.tensorSharing.releaseTensor(embeddings);
    await this.tensorSharing.releaseTensor(layerOutput);
    await this.tensorSharing.releaseTensor(inputTensors.inputIds);
    await this.tensorSharing.releaseTensor(inputTensors.tokenTypeIds);
    await this.tensorSharing.releaseTensor(inputTensors.positionIds);
    await this.tensorSharing.releaseTensor(inputTensors.attentionMask);
    
    return output;
  }
  
  /**
   * Create input tensors from tokens
   */
  private async createInputTensors(tokens: BERTEmbedding): Promise<{
    inputIds: TensorView;
    tokenTypeIds: TensorView;
    positionIds: TensorView;
    attentionMask: TensorView;
  }> {
    const seq_length = tokens.inputIds.length;
    
    // Input IDs tensor
    const inputIds = await this.tensorSharing.createTensorFromData(
      new Int32Array(tokens.inputIds),
      [1, seq_length],
      'int32'
    );
    
    // Token type IDs tensor (0 for sentence A, 1 for sentence B)
    const tokenTypeIds = await this.tensorSharing.createTensorFromData(
      new Int32Array(tokens.tokenTypeIds || new Array(seq_length).fill(0)),
      [1, seq_length],
      'int32'
    );
    
    // Position IDs tensor (0, 1, 2, ...)
    const positionIdsArray = new Array(seq_length).fill(0).map((_, i) => i);
    const positionIds = await this.tensorSharing.createTensorFromData(
      new Int32Array(positionIdsArray),
      [1, seq_length],
      'int32'
    );
    
    // Attention mask tensor (1 for tokens, 0 for padding)
    const attentionMask = await this.tensorSharing.createTensorFromData(
      new Int32Array(tokens.attentionMask),
      [1, seq_length],
      'int32'
    );
    
    return {
      inputIds,
      tokenTypeIds,
      positionIds,
      attentionMask
    };
  }
  
  /**
   * Get token embeddings from input IDs, token type IDs, and position IDs
   */
  private async getEmbeddings(
    inputIds: TensorView,
    tokenTypeIds: TensorView,
    positionIds: TensorView
  ): Promise<TensorView> {
    const startTime = performance.now();
    
    // Get embeddings from weights
    const wordEmbeddings = this.weights.get('bert.embeddings.word_embeddings.weight');
    const tokenTypeEmbeddings = this.weights.get('bert.embeddings.token_type_embeddings.weight');
    const positionEmbeddings = this.weights.get('bert.embeddings.position_embeddings.weight');
    const layerNormWeight = this.weights.get('bert.embeddings.LayerNorm.weight');
    const layerNormBias = this.weights.get('bert.embeddings.LayerNorm.bias');
    
    if (!wordEmbeddings || !tokenTypeEmbeddings || !positionEmbeddings || !layerNormWeight || !layerNormBias) {
      throw new Error('Embedding weights not found');
    }
    
    // Embedding lookup (optimized for browser)
    const wordEmbeddingOutput = await this.tensorSharing.executeEmbeddingLookup(
      inputIds,
      wordEmbeddings
    );
    
    const tokenTypeEmbeddingOutput = await this.tensorSharing.executeEmbeddingLookup(
      tokenTypeIds,
      tokenTypeEmbeddings
    );
    
    const positionEmbeddingOutput = await this.tensorSharing.executeEmbeddingLookup(
      positionIds,
      positionEmbeddings
    );
    
    // Add embeddings
    const embeddingsSum1 = await this.tensorSharing.executeElementwiseOperation(
      wordEmbeddingOutput,
      tokenTypeEmbeddingOutput,
      'add'
    );
    
    const embeddingsSum2 = await this.tensorSharing.executeElementwiseOperation(
      embeddingsSum1,
      positionEmbeddingOutput,
      'add'
    );
    
    // Apply layer normalization
    const embeddings = await this.tensorSharing.executeLayerNorm(
      embeddingsSum2,
      layerNormWeight,
      layerNormBias,
      1e-12 // BERT uses 1e-12 for layer norm epsilon
    );
    
    // Release intermediate tensors
    await this.tensorSharing.releaseTensor(wordEmbeddingOutput);
    await this.tensorSharing.releaseTensor(tokenTypeEmbeddingOutput);
    await this.tensorSharing.releaseTensor(positionEmbeddingOutput);
    await this.tensorSharing.releaseTensor(embeddingsSum1);
    await this.tensorSharing.releaseTensor(embeddingsSum2);
    
    const endTime = performance.now();
    this.recordOperationTime('embedding', endTime - startTime);
    
    return embeddings;
  }
  
  /**
   * Process a single transformer layer
   */
  private async processLayer(
    input: TensorView,
    attentionMask: TensorView,
    layerIdx: number
  ): Promise<TensorView> {
    const startTime = performance.now();
    
    // 1. Self-attention
    const attentionOutput = await this.applySelfAttention(
      input,
      attentionMask,
      layerIdx
    );
    
    // 2. Add & LayerNorm (after attention)
    const attentionLayerNorm = await this.applyAddAndLayerNorm(
      input,
      attentionOutput,
      `bert.encoder.layer.${layerIdx}.attention.output.LayerNorm.weight`,
      `bert.encoder.layer.${layerIdx}.attention.output.LayerNorm.bias`
    );
    
    // 3. Feed-forward network
    const feedForwardOutput = await this.applyFeedForward(
      attentionLayerNorm,
      layerIdx
    );
    
    // 4. Add & LayerNorm (after feed-forward)
    const outputLayerNorm = await this.applyAddAndLayerNorm(
      attentionLayerNorm,
      feedForwardOutput,
      `bert.encoder.layer.${layerIdx}.output.LayerNorm.weight`,
      `bert.encoder.layer.${layerIdx}.output.LayerNorm.bias`
    );
    
    // Release intermediate tensors
    await this.tensorSharing.releaseTensor(attentionOutput);
    await this.tensorSharing.releaseTensor(attentionLayerNorm);
    await this.tensorSharing.releaseTensor(feedForwardOutput);
    
    const endTime = performance.now();
    this.recordOperationTime(`layer_${layerIdx}`, endTime - startTime);
    
    return outputLayerNorm;
  }
  
  /**
   * Apply self-attention mechanism
   */
  private async applySelfAttention(
    input: TensorView,
    attentionMask: TensorView,
    layerIdx: number
  ): Promise<TensorView> {
    const startTime = performance.now();
    
    const queryWeight = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.self.query.weight`);
    const queryBias = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.self.query.bias`);
    const keyWeight = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.self.key.weight`);
    const keyBias = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.self.key.bias`);
    const valueWeight = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.self.value.weight`);
    const valueBias = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.self.value.bias`);
    const outputWeight = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.output.dense.weight`);
    const outputBias = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.output.dense.bias`);
    
    if (!queryWeight || !queryBias || !keyWeight || !keyBias || !valueWeight || !valueBias || !outputWeight || !outputBias) {
      throw new Error(`Attention weights not found for layer ${layerIdx}`);
    }
    
    if (this.config.useOptimizedAttention && 
        this.attentionPipeline && 
        this.browserCapabilities?.flashAttentionSupported) {
      // Use optimized flash attention if supported
      const batchSize = input.dims[0];
      const seqLen = input.dims[1];
      
      // Create extended attention mask
      // Convert mask from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
      // and adjust values: 0 -> -10000.0, 1 -> 0.0
      const extendedAttentionMask = await this.createExtendedAttentionMask(attentionMask);
      
      // Process with optimized attention
      const result = await this.tensorSharing.executeCustomPipeline(
        this.attentionPipeline,
        [
          input, 
          queryWeight, queryBias,
          keyWeight, keyBias,
          valueWeight, valueBias,
          outputWeight, outputBias,
          extendedAttentionMask
        ],
        input.dims,
        {
          workgroupSize: this.browserCapabilities.optimalWorkgroupSizes.attention || [128, 1, 1],
          numHeads: this.config.numHeads,
          headDim: this.config.hiddenSize / this.config.numHeads,
          seqLen: seqLen,
          batchSize: batchSize,
          causalMask: false, // BERT uses bidirectional attention
          useSharedMemory: this.browserCapabilities.sharedMemorySupported,
          attentionScale: 1.0 / Math.sqrt(this.config.hiddenSize / this.config.numHeads)
        }
      );
      
      // Release extended attention mask
      await this.tensorSharing.releaseTensor(extendedAttentionMask);
      
      const endTime = performance.now();
      this.recordOperationTime(`attention_${layerIdx}`, endTime - startTime);
      
      return result;
    } else {
      // Standard attention implementation
      
      // Apply query, key, value projections
      const query = await this.tensorSharing.executeMatmulWithBias(input, queryWeight, queryBias);
      const key = await this.tensorSharing.executeMatmulWithBias(input, keyWeight, keyBias);
      const value = await this.tensorSharing.executeMatmulWithBias(input, valueWeight, valueBias);
      
      // Reshape for multi-head attention
      const batchSize = input.dims[0];
      const seqLen = input.dims[1];
      const numHeads = this.config.numHeads;
      const headDim = this.config.hiddenSize / numHeads;
      
      const reshapedQuery = await this.tensorSharing.reshapeTensor(
        query,
        [batchSize, seqLen, numHeads, headDim]
      );
      
      const reshapedKey = await this.tensorSharing.reshapeTensor(
        key,
        [batchSize, seqLen, numHeads, headDim]
      );
      
      const reshapedValue = await this.tensorSharing.reshapeTensor(
        value,
        [batchSize, seqLen, numHeads, headDim]
      );
      
      // Transpose for attention computation
      const transposedQuery = await this.tensorSharing.transposeTensor(
        reshapedQuery,
        [0, 2, 1, 3], // [batch, head, seq, dim]
        [0, 2, 1, 3]  // [batch, head, seq, dim]
      );
      
      const transposedKey = await this.tensorSharing.transposeTensor(
        reshapedKey,
        [0, 2, 1, 3], // [batch, head, seq, dim]
        [0, 2, 3, 1]  // [batch, head, dim, seq]
      );
      
      const transposedValue = await this.tensorSharing.transposeTensor(
        reshapedValue,
        [0, 2, 1, 3], // [batch, head, seq, dim]
        [0, 2, 1, 3]  // [batch, head, seq, dim]
      );
      
      // Compute attention scores
      const attentionScores = await this.tensorSharing.executeMatmul(
        transposedQuery,
        transposedKey
      );
      
      // Scale attention scores
      const scaleFactor = 1.0 / Math.sqrt(headDim);
      const scaledAttentionScores = await this.tensorSharing.executeScalarMultiplication(
        attentionScores,
        scaleFactor
      );
      
      // Create extended attention mask
      const extendedAttentionMask = await this.createExtendedAttentionMask(attentionMask);
      
      // Add attention mask to scores
      const maskedAttentionScores = await this.tensorSharing.executeElementwiseOperation(
        scaledAttentionScores,
        extendedAttentionMask,
        'add'
      );
      
      // Apply softmax
      const attentionProbs = await this.tensorSharing.executeSoftmax(
        maskedAttentionScores,
        -1 // Softmax along sequence dimension
      );
      
      // Apply attention to values
      const contextLayer = await this.tensorSharing.executeMatmul(
        attentionProbs,
        transposedValue
      );
      
      // Transpose back and reshape
      const transposedContext = await this.tensorSharing.transposeTensor(
        contextLayer,
        [0, 2, 1, 3], // [batch, head, seq, dim]
        [0, 1, 2, 3]  // [batch, seq, head, dim]
      );
      
      const reshapedContext = await this.tensorSharing.reshapeTensor(
        transposedContext,
        [batchSize, seqLen, this.config.hiddenSize]
      );
      
      // Apply output projection
      const attentionOutput = await this.tensorSharing.executeMatmulWithBias(
        reshapedContext,
        outputWeight,
        outputBias
      );
      
      // Release intermediate tensors
      await this.tensorSharing.releaseTensor(query);
      await this.tensorSharing.releaseTensor(key);
      await this.tensorSharing.releaseTensor(value);
      await this.tensorSharing.releaseTensor(reshapedQuery);
      await this.tensorSharing.releaseTensor(reshapedKey);
      await this.tensorSharing.releaseTensor(reshapedValue);
      await this.tensorSharing.releaseTensor(transposedQuery);
      await this.tensorSharing.releaseTensor(transposedKey);
      await this.tensorSharing.releaseTensor(transposedValue);
      await this.tensorSharing.releaseTensor(attentionScores);
      await this.tensorSharing.releaseTensor(scaledAttentionScores);
      await this.tensorSharing.releaseTensor(extendedAttentionMask);
      await this.tensorSharing.releaseTensor(maskedAttentionScores);
      await this.tensorSharing.releaseTensor(attentionProbs);
      await this.tensorSharing.releaseTensor(contextLayer);
      await this.tensorSharing.releaseTensor(transposedContext);
      await this.tensorSharing.releaseTensor(reshapedContext);
      
      const endTime = performance.now();
      this.recordOperationTime(`attention_${layerIdx}`, endTime - startTime);
      
      return attentionOutput;
    }
  }
  
  /**
   * Create extended attention mask for attention mechanism
   */
  private async createExtendedAttentionMask(attentionMask: TensorView): Promise<TensorView> {
    // Convert mask from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
    const batchSize = attentionMask.dims[0];
    const seqLen = attentionMask.dims[1];
    
    // Reshape to [batch_size, 1, 1, seq_len]
    const reshapedMask = await this.tensorSharing.reshapeTensor(
      attentionMask,
      [batchSize, 1, 1, seqLen]
    );
    
    // Convert mask values: 0 -> -10000.0, 1 -> 0.0
    // First, convert from int32 to float32 if needed
    let floatMask = reshapedMask;
    if (reshapedMask.dtype === 'int32') {
      floatMask = await this.tensorSharing.convertTensor(reshapedMask, 'float32');
      await this.tensorSharing.releaseTensor(reshapedMask);
    }
    
    // Apply transformation: (1 - mask) * -10000.0
    const negatedMask = await this.tensorSharing.executeElementwiseOperation(
      floatMask,
      null,
      'negate'
    );
    
    const onesTensor = await this.tensorSharing.createTensorFromData(
      new Float32Array([1.0]),
      [1],
      'float32'
    );
    
    const invertedMask = await this.tensorSharing.executeElementwiseOperation(
      negatedMask,
      onesTensor,
      'add'
    );
    
    const scaleFactor = -10000.0;
    const extendedMask = await this.tensorSharing.executeScalarMultiplication(
      invertedMask,
      scaleFactor
    );
    
    // Release intermediate tensors
    if (floatMask !== reshapedMask) {
      await this.tensorSharing.releaseTensor(floatMask);
    }
    await this.tensorSharing.releaseTensor(negatedMask);
    await this.tensorSharing.releaseTensor(onesTensor);
    await this.tensorSharing.releaseTensor(invertedMask);
    
    return extendedMask;
  }
  
  /**
   * Apply feed-forward network to transformer layer
   */
  private async applyFeedForward(
    input: TensorView,
    layerIdx: number
  ): Promise<TensorView> {
    const startTime = performance.now();
    
    const intermediateWeight = this.weights.get(`bert.encoder.layer.${layerIdx}.intermediate.dense.weight`);
    const intermediateBias = this.weights.get(`bert.encoder.layer.${layerIdx}.intermediate.dense.bias`);
    const outputWeight = this.weights.get(`bert.encoder.layer.${layerIdx}.output.dense.weight`);
    const outputBias = this.weights.get(`bert.encoder.layer.${layerIdx}.output.dense.bias`);
    
    if (!intermediateWeight || !intermediateBias || !outputWeight || !outputBias) {
      throw new Error(`Feed-forward weights not found for layer ${layerIdx}`);
    }
    
    // Use specialized pipeline for intermediate matmul
    const intermediateMatmulOutput = await this.tensorSharing.executeCustomPipeline(
      this.intermediateMatmulPipeline!,
      [input, intermediateWeight, intermediateBias],
      [input.dims[0], input.dims[1], this.config.intermediateSize],
      {
        transposeA: false,
        transposeB: true,
        workgroupSize: this.browserCapabilities?.optimalWorkgroupSizes.matmul || [8, 8, 1],
        addBias: true
      }
    );
    
    // Apply GELU activation
    const geluOutput = await this.tensorSharing.executeCustomPipeline(
      this.geluPipeline!,
      [intermediateMatmulOutput],
      intermediateMatmulOutput.dims,
      {
        workgroupSize: this.browserCapabilities?.optimalWorkgroupSizes.elementwise || [256, 1, 1],
        approximation: 'tanh' // BERT typically uses tanh approximation
      }
    );
    
    // Use specialized pipeline for output matmul
    const outputMatmulOutput = await this.tensorSharing.executeCustomPipeline(
      this.outputMatmulPipeline!,
      [geluOutput, outputWeight, outputBias],
      [input.dims[0], input.dims[1], this.config.hiddenSize],
      {
        transposeA: false,
        transposeB: true,
        workgroupSize: this.browserCapabilities?.optimalWorkgroupSizes.matmul || [8, 8, 1],
        addBias: true
      }
    );
    
    // Release intermediate tensors
    await this.tensorSharing.releaseTensor(intermediateMatmulOutput);
    await this.tensorSharing.releaseTensor(geluOutput);
    
    const endTime = performance.now();
    this.recordOperationTime(`feedforward_${layerIdx}`, endTime - startTime);
    
    return outputMatmulOutput;
  }
  
  /**
   * Apply add and layer normalization
   */
  private async applyAddAndLayerNorm(
    input: TensorView,
    residual: TensorView,
    weightKey: string,
    biasKey: string
  ): Promise<TensorView> {
    const startTime = performance.now();
    
    const layerNormWeight = this.weights.get(weightKey);
    const layerNormBias = this.weights.get(biasKey);
    
    if (!layerNormWeight || !layerNormBias) {
      throw new Error(`Layer norm weights not found: ${weightKey}, ${biasKey}`);
    }
    
    // Add residual connection
    const residualSum = await this.tensorSharing.executeElementwiseOperation(
      input,
      residual,
      'add'
    );
    
    // Apply layer normalization
    const output = await this.tensorSharing.executeCustomPipeline(
      this.layerNormPipeline!,
      [residualSum, layerNormWeight, layerNormBias],
      residualSum.dims,
      {
        workgroupSize: this.browserCapabilities?.optimalWorkgroupSizes.reduction || [256, 1, 1],
        epsilon: 1e-12, // BERT uses 1e-12 for layer norm
        axis: -1 // Normalize across hidden dimension
      }
    );
    
    // Release intermediate tensor
    await this.tensorSharing.releaseTensor(residualSum);
    
    const endTime = performance.now();
    this.recordOperationTime('layer_norm', endTime - startTime);
    
    return output;
  }
  
  /**
   * Get task-specific output based on config.taskType
   */
  private async getTaskOutput(lastHiddenState: TensorView): Promise<BERTOutput> {
    const startTime = performance.now();
    
    switch (this.config.taskType) {
      case 'embedding':
        // For embedding task, return the [CLS] token embedding
        const clsEmbedding = await this.tensorSharing.extractFirstToken(lastHiddenState);
        
        // Get the data as Float32Array
        const embeddingData = await this.tensorSharing.getTensorData(clsEmbedding) as Float32Array;
        
        // Release tensor
        await this.tensorSharing.releaseTensor(clsEmbedding);
        
        const endTime = performance.now();
        this.recordOperationTime('task_output', endTime - startTime);
        
        return embeddingData;
        
      case 'sequence_classification':
        // For sequence classification, apply classifier to [CLS] token
        if (!this.config.numLabels) {
          throw new Error('numLabels must be specified for sequence_classification task');
        }
        
        const poolerWeight = this.weights.get('bert.pooler.dense.weight');
        const poolerBias = this.weights.get('bert.pooler.dense.bias');
        const classifierWeight = this.weights.get('classifier.weight');
        const classifierBias = this.weights.get('classifier.bias');
        
        if (!poolerWeight || !poolerBias || !classifierWeight || !classifierBias) {
          throw new Error('Classification weights not found');
        }
        
        // Extract [CLS] token (first token)
        const firstToken = await this.tensorSharing.extractFirstToken(lastHiddenState);
        
        // Apply pooler (dense layer with tanh activation)
        const poolerOutput = await this.tensorSharing.executeMatmulWithBias(
          firstToken,
          poolerWeight,
          poolerBias
        );
        
        const tanhPoolerOutput = await this.tensorSharing.executeElementwiseOperation(
          poolerOutput,
          null,
          'tanh'
        );
        
        // Apply classifier layer
        const logits = await this.tensorSharing.executeMatmulWithBias(
          tanhPoolerOutput,
          classifierWeight,
          classifierBias
        );
        
        // Get logits as Float32Array
        const logitsData = await this.tensorSharing.getTensorData(logits) as Float32Array;
        
        // Release tensors
        await this.tensorSharing.releaseTensor(firstToken);
        await this.tensorSharing.releaseTensor(poolerOutput);
        await this.tensorSharing.releaseTensor(tanhPoolerOutput);
        await this.tensorSharing.releaseTensor(logits);
        
        const endTime2 = performance.now();
        this.recordOperationTime('task_output', endTime2 - startTime);
        
        return { logits: logitsData };
        
      case 'token_classification':
        // For token classification, apply classifier to all tokens
        if (!this.config.numLabels) {
          throw new Error('numLabels must be specified for token_classification task');
        }
        
        const tokenClassifierWeight = this.weights.get('classifier.weight');
        const tokenClassifierBias = this.weights.get('classifier.bias');
        
        if (!tokenClassifierWeight || !tokenClassifierBias) {
          throw new Error('Token classification weights not found');
        }
        
        // Apply classifier to all token embeddings
        const tokenLogits = await this.tensorSharing.executeMatmulWithBias(
          lastHiddenState,
          tokenClassifierWeight,
          tokenClassifierBias
        );
        
        // Get logits as Float32Array
        const tokenLogitsData = await this.tensorSharing.getTensorData(tokenLogits) as Float32Array;
        
        // Release tensor
        await this.tensorSharing.releaseTensor(tokenLogits);
        
        const endTime3 = performance.now();
        this.recordOperationTime('task_output', endTime3 - startTime);
        
        return { logits: tokenLogitsData };
        
      case 'question_answering':
        // For question answering, predict start and end positions
        const qaStartWeight = this.weights.get('qa_outputs.weight');
        const qaStartBias = this.weights.get('qa_outputs.bias');
        
        if (!qaStartWeight || !qaStartBias) {
          throw new Error('Question answering weights not found');
        }
        
        // Apply classifier to get start and end logits
        const qaLogits = await this.tensorSharing.executeMatmulWithBias(
          lastHiddenState,
          qaStartWeight,
          qaStartBias
        );
        
        // Split into start and end logits
        // In this simplified implementation, we assume the classifier outputs [seq_len, 2]
        // where [:, 0] are start logits and [:, 1] are end logits
        const [startLogits, endLogits] = await this.tensorSharing.splitTensor(
          qaLogits,
          -1, // Split along last dimension
          2   // Into 2 parts
        );
        
        // Get logits as Float32Array
        const startLogitsData = await this.tensorSharing.getTensorData(startLogits) as Float32Array;
        const endLogitsData = await this.tensorSharing.getTensorData(endLogits) as Float32Array;
        
        // Release tensors
        await this.tensorSharing.releaseTensor(qaLogits);
        await this.tensorSharing.releaseTensor(startLogits);
        await this.tensorSharing.releaseTensor(endLogits);
        
        const endTime4 = performance.now();
        this.recordOperationTime('task_output', endTime4 - startTime);
        
        return {
          start_logits: startLogitsData,
          end_logits: endLogitsData
        };
        
      default:
        throw new Error(`Unsupported task type: ${this.config.taskType}`);
    }
  }
  
  /**
   * Record operation execution time for performance tracking
   */
  private recordOperationTime(operation: string, time: number): void {
    if (!this.opExecutionTimes.has(operation)) {
      this.opExecutionTimes.set(operation, []);
    }
    this.opExecutionTimes.get(operation)!.push(time);
  }
  
  /**
   * Get performance metrics for all operations
   */
  getPerformanceMetrics(): Record<string, { 
    avg: number; 
    min: number; 
    max: number; 
    count: number;
    total: number;
  }> {
    const metrics: Record<string, { avg: number; min: number; max: number; count: number; total: number }> = {};
    
    for (const [operation, times] of this.opExecutionTimes.entries()) {
      const count = times.length;
      if (count === 0) continue;
      
      const total = times.reduce((sum, time) => sum + time, 0);
      const avg = total / count;
      const min = Math.min(...times);
      const max = Math.max(...times);
      
      metrics[operation] = { avg, min, max, count, total };
    }
    
    return metrics;
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
    
    this.initialized = false;
    this.opExecutionTimes.clear();
  }
  
  /**
   * Get model information
   */
  getModelInfo(): Record<string, any> {
    return {
      modelType: 'BERT',
      vocabSize: this.config.vocabSize,
      hiddenSize: this.config.hiddenSize,
      numLayers: this.config.numLayers,
      numHeads: this.config.numHeads,
      intermediateSize: this.config.intermediateSize,
      maxSequenceLength: this.config.maxSequenceLength,
      quantization: this.config.quantization,
      browserOptimized: true,
      browserType: this.browserCapabilities?.browserType || 'unknown',
      gpuVendor: this.browserCapabilities?.gpuVendor || 'unknown',
      optimizationLevel: this.optimizationSettings?.optimizationLevel || 'standard',
      useOptimizedAttention: this.config.useOptimizedAttention,
      flashAttentionSupported: this.browserCapabilities?.flashAttentionSupported || false,
      fusedOperationsSupported: this.browserCapabilities?.fusedOperationsSupported || false,
      sharedMemorySupported: this.browserCapabilities?.sharedMemorySupported || false,
      taskType: this.config.taskType
    };
  }
}