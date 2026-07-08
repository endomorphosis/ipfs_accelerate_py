/**
 * BERT (Bidirectional Encoder Representations from Transformers) implementation 
 * using the Hardware Abstraction Layer
 * 
 * This implementation uses the Hardware Abstraction Layer (HAL) to automatically
 * select the optimal backend (WebGPU, WebNN, CPU) for BERT inference based on
 * the available hardware capabilities and model requirements.
 */

import { HardwareAbstraction, HardwareBackend, createHardwareAbstraction } from './ipfs_accelerate_js_hardware_abstraction';
import { StorageManager } from './ipfs_accelerate_js_storage_manager';

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
export interface BERTEmbedding {
  inputIds: number[];
  tokenTypeIds?: number[];
  attentionMask: number[];
}

/**
 * BERT output type (varies based on task)
 */
export type BERTOutput = 
  | Float32Array                // For embeddings
  | { logits: Float32Array }    // For classification
  | { start_logits: Float32Array; end_logits: Float32Array } // For QA

/**
 * BERT implementation using the Hardware Abstraction Layer
 */
export class HardwareAbstractedBERT {
  private config: BERTConfig;
  private hal: HardwareAbstraction;
  private storageManager: StorageManager;
  private initialized: boolean = false;
  private weights: Map<string, any> = new Map();
  private hardwareCapabilities: any = null;
  private opExecutionTimes: Map<string, number[]> = new Map();
  
  /**
   * Create a new HAL-accelerated BERT model
   * @param config Model configuration
   * @param storageManager Storage manager for model weights
   */
  constructor(config: BERTConfig, storageManager: StorageManager) {
    this.config = {
      maxSequenceLength: 512,
      useOptimizedAttention: true,
      taskType: 'embedding',
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
        'text': 'webnn',     // Text models like BERT generally perform best on WebNN
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
    
    console.log(`BERT model initialized with Hardware Abstraction Layer`);
    console.log(`Available backends: ${this.hal.getAvailableBackends().join(', ')}`);
    const bestBackend = this.hal.getBestBackend('text');
    console.log(`Selected backend for text: ${bestBackend?.type}`);
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
      const backend = this.hal.getBestBackend('text');
      
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
    
    console.log(`Loaded ${this.weights.size} weight tensors for BERT model`);
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
    
    const startTime = performance.now();
    
    // Tokenize input if needed
    const tokens = typeof text === 'string' ? this.tokenize(text) : text;
    
    // Get best backend for text models
    const backend = this.hal.getBestBackend('text');
    if (!backend) {
      throw new Error('No suitable backend available for BERT inference');
    }
    
    try {
      // Start timing
      console.time('bert_inference');
      
      // Create input tensors
      const inputTensors = await this.createInputTensors(tokens, backend);
      
      // Get embeddings
      const embeddings = await this.getEmbeddings(
        inputTensors.inputIds,
        inputTensors.tokenTypeIds,
        inputTensors.positionIds
      );
      
      // Create extended attention mask
      const extendedAttentionMask = await this.createExtendedAttentionMask(
        inputTensors.attentionMask
      );
      
      // Process through transformer layers
      let layerOutput = embeddings;
      for (let i = 0; i < this.config.numLayers; i++) {
        const layerStartTime = performance.now();
        layerOutput = await this.processLayer(
          layerOutput,
          extendedAttentionMask,
          i
        );
        
        const layerEndTime = performance.now();
        this.recordOperationTime(`layer_${i}`, layerEndTime - layerStartTime);
      }
      
      // Get output based on task type
      const output = await this.getTaskOutput(layerOutput);
      
      // Clean up tensors
      await this.cleanupTensors([
        embeddings, 
        layerOutput, 
        extendedAttentionMask,
        ...Object.values(inputTensors)
      ]);
      
      // End timing
      console.timeEnd('bert_inference');
      
      const endTime = performance.now();
      this.recordOperationTime('total_inference', endTime - startTime);
      
      return output;
      
    } catch (error) {
      console.error('Error during BERT inference:', error);
      
      // Attempt automatic fallback if enabled
      if (this.hal.hasBackend('cpu')) {
        console.warn('Attempting fallback to CPU backend...');
        
        // Force CPU backend for next iteration
        const cpuResult = await this.hal.execute(
          'predictBert',
          { text, config: this.config },
          { 
            modelType: 'text', 
            preferredBackend: 'cpu' 
          }
        );
        
        return cpuResult as BERTOutput;
      }
      
      throw error;
    }
  }
  
  /**
   * Create input tensors from tokens
   */
  private async createInputTensors(tokens: BERTEmbedding, backend: HardwareBackend): Promise<{
    inputIds: any;
    tokenTypeIds: any;
    positionIds: any;
    attentionMask: any;
  }> {
    const createTensorStartTime = performance.now();
    
    const seq_length = tokens.inputIds.length;
    
    // Input IDs tensor
    const inputIds = await backend.createTensor(
      new Int32Array(tokens.inputIds),
      [1, seq_length],
      'int32'
    );
    
    // Token type IDs tensor (0 for sentence A, 1 for sentence B)
    const tokenTypeIds = await backend.createTensor(
      new Int32Array(tokens.tokenTypeIds || new Array(seq_length).fill(0)),
      [1, seq_length],
      'int32'
    );
    
    // Position IDs tensor (0, 1, 2, ...)
    const positionIdsArray = new Array(seq_length).fill(0).map((_, i) => i);
    const positionIds = await backend.createTensor(
      new Int32Array(positionIdsArray),
      [1, seq_length],
      'int32'
    );
    
    // Attention mask tensor (1 for tokens, 0 for padding)
    const attentionMask = await backend.createTensor(
      new Int32Array(tokens.attentionMask),
      [1, seq_length],
      'int32'
    );
    
    const createTensorEndTime = performance.now();
    this.recordOperationTime('create_input_tensors', createTensorEndTime - createTensorStartTime);
    
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
    inputIds: any,
    tokenTypeIds: any,
    positionIds: any
  ): Promise<any> {
    const embeddingStartTime = performance.now();
    
    // Get embeddings from weights
    const wordEmbeddings = this.weights.get('bert.embeddings.word_embeddings.weight')?.tensor;
    const tokenTypeEmbeddings = this.weights.get('bert.embeddings.token_type_embeddings.weight')?.tensor;
    const positionEmbeddings = this.weights.get('bert.embeddings.position_embeddings.weight')?.tensor;
    const layerNormWeight = this.weights.get('bert.embeddings.LayerNorm.weight')?.tensor;
    const layerNormBias = this.weights.get('bert.embeddings.LayerNorm.bias')?.tensor;
    
    if (!wordEmbeddings || !tokenTypeEmbeddings || !positionEmbeddings || 
        !layerNormWeight || !layerNormBias) {
      throw new Error('Embedding weights not found');
    }
    
    // Execute embedding lookup using HAL
    const wordEmbeddingOutput = await this.hal.execute(
      'embeddingLookup',
      {
        indices: inputIds,
        weights: wordEmbeddings
      },
      { modelType: 'text' }
    );
    
    const tokenTypeEmbeddingOutput = await this.hal.execute(
      'embeddingLookup',
      {
        indices: tokenTypeIds,
        weights: tokenTypeEmbeddings
      },
      { modelType: 'text' }
    );
    
    const positionEmbeddingOutput = await this.hal.execute(
      'embeddingLookup',
      {
        indices: positionIds,
        weights: positionEmbeddings
      },
      { modelType: 'text' }
    );
    
    // Add embeddings
    const embeddingsSum1 = await this.hal.execute(
      'add',
      {
        a: wordEmbeddingOutput,
        b: tokenTypeEmbeddingOutput
      },
      { modelType: 'text' }
    );
    
    const embeddingsSum2 = await this.hal.execute(
      'add',
      {
        a: embeddingsSum1,
        b: positionEmbeddingOutput
      },
      { modelType: 'text' }
    );
    
    // Apply layer normalization
    const embeddings = await this.hal.execute(
      'layerNorm',
      {
        input: embeddingsSum2,
        weight: layerNormWeight,
        bias: layerNormBias
      },
      { 
        modelType: 'text',
        backendOptions: { epsilon: 1e-12 } // BERT uses 1e-12 for layer norm epsilon
      }
    );
    
    // Release intermediate tensors
    await this.cleanupTensors([
      wordEmbeddingOutput,
      tokenTypeEmbeddingOutput,
      positionEmbeddingOutput,
      embeddingsSum1,
      embeddingsSum2
    ]);
    
    const embeddingEndTime = performance.now();
    this.recordOperationTime('embeddings', embeddingEndTime - embeddingStartTime);
    
    return embeddings;
  }
  
  /**
   * Create extended attention mask for attention mechanism
   */
  private async createExtendedAttentionMask(attentionMask: any): Promise<any> {
    const maskStartTime = performance.now();
    
    // Convert mask from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
    // and adjust values: 0 -> -10000.0, 1 -> 0.0
    
    // Use HAL to reshape attention mask
    const extendedMask = await this.hal.execute(
      'createAttentionMask',
      {
        attentionMask,
        attentionScale: -10000.0
      },
      { modelType: 'text' }
    );
    
    const maskEndTime = performance.now();
    this.recordOperationTime('create_attention_mask', maskEndTime - maskStartTime);
    
    return extendedMask;
  }
  
  /**
   * Process a single transformer layer
   */
  private async processLayer(
    input: any,
    attentionMask: any,
    layerIdx: number
  ): Promise<any> {
    // 1. Self-attention
    const attentionStartTime = performance.now();
    const attentionOutput = await this.applySelfAttention(
      input,
      attentionMask,
      layerIdx
    );
    const attentionEndTime = performance.now();
    this.recordOperationTime(`attention_${layerIdx}`, attentionEndTime - attentionStartTime);
    
    // 2. Add & LayerNorm (after attention)
    const layerNorm1StartTime = performance.now();
    const attentionLayerNorm = await this.applyAddAndLayerNorm(
      input,
      attentionOutput,
      `bert.encoder.layer.${layerIdx}.attention.output.LayerNorm.weight`,
      `bert.encoder.layer.${layerIdx}.attention.output.LayerNorm.bias`
    );
    const layerNorm1EndTime = performance.now();
    this.recordOperationTime(`layernorm1_${layerIdx}`, layerNorm1EndTime - layerNorm1StartTime);
    
    // Release attention output tensor
    await this.cleanupTensors([attentionOutput]);
    
    // 3. Feed-forward network
    const ffnStartTime = performance.now();
    const feedForwardOutput = await this.applyFeedForward(
      attentionLayerNorm,
      layerIdx
    );
    const ffnEndTime = performance.now();
    this.recordOperationTime(`ffn_${layerIdx}`, ffnEndTime - ffnStartTime);
    
    // 4. Add & LayerNorm (after feed-forward)
    const layerNorm2StartTime = performance.now();
    const outputLayerNorm = await this.applyAddAndLayerNorm(
      attentionLayerNorm,
      feedForwardOutput,
      `bert.encoder.layer.${layerIdx}.output.LayerNorm.weight`,
      `bert.encoder.layer.${layerIdx}.output.LayerNorm.bias`
    );
    const layerNorm2EndTime = performance.now();
    this.recordOperationTime(`layernorm2_${layerIdx}`, layerNorm2EndTime - layerNorm2StartTime);
    
    // Release intermediate tensors
    await this.cleanupTensors([
      attentionLayerNorm,
      feedForwardOutput
    ]);
    
    return outputLayerNorm;
  }
  
  /**
   * Apply self-attention mechanism
   */
  private async applySelfAttention(
    input: any,
    attentionMask: any,
    layerIdx: number
  ): Promise<any> {
    // Get attention weights from model weights
    const queryWeight = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.self.query.weight`)?.tensor;
    const queryBias = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.self.query.bias`)?.tensor;
    const keyWeight = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.self.key.weight`)?.tensor;
    const keyBias = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.self.key.bias`)?.tensor;
    const valueWeight = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.self.value.weight`)?.tensor;
    const valueBias = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.self.value.bias`)?.tensor;
    const outputWeight = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.output.dense.weight`)?.tensor;
    const outputBias = this.weights.get(`bert.encoder.layer.${layerIdx}.attention.output.dense.bias`)?.tensor;
    
    if (!queryWeight || !queryBias || !keyWeight || !keyBias || 
        !valueWeight || !valueBias || !outputWeight || !outputBias) {
      throw new Error(`Attention weights not found for layer ${layerIdx}`);
    }
    
    // Use optimized attention if enabled
    if (this.config.useOptimizedAttention) {
      // Execute self-attention using optimized HAL implementation
      return await this.hal.execute(
        'selfAttention',
        {
          input,
          queryWeight,
          queryBias,
          keyWeight,
          keyBias,
          valueWeight,
          valueBias,
          outputWeight,
          outputBias,
          attentionMask
        },
        {
          modelType: 'text',
          backendOptions: {
            numHeads: this.config.numHeads,
            hiddenSize: this.config.hiddenSize,
            useFlashAttention: true,
            causalMask: false, // BERT uses bidirectional attention
            attentionScale: 1.0 / Math.sqrt(this.config.hiddenSize / this.config.numHeads)
          }
        }
      );
    } else {
      // Standard attention implementation
      
      // Apply query, key, value projections
      const query = await this.hal.execute(
        'linearWithBias',
        {
          input,
          weight: queryWeight,
          bias: queryBias
        },
        { modelType: 'text' }
      );
      
      const key = await this.hal.execute(
        'linearWithBias',
        {
          input,
          weight: keyWeight,
          bias: keyBias
        },
        { modelType: 'text' }
      );
      
      const value = await this.hal.execute(
        'linearWithBias',
        {
          input,
          weight: valueWeight,
          bias: valueBias
        },
        { modelType: 'text' }
      );
      
      // Reshape and transpose for multi-head attention
      const batchSize = 1; // For simplicity in this example
      const seqLen = input.dims[1];
      const numHeads = this.config.numHeads;
      const headDim = this.config.hiddenSize / numHeads;
      
      // Execute attention score computation
      const attentionOutput = await this.hal.execute(
        'multiHeadAttention',
        {
          query,
          key,
          value,
          attentionMask,
          outputWeight,
          outputBias
        },
        {
          modelType: 'text',
          backendOptions: {
            batchSize,
            seqLen,
            numHeads,
            headDim,
            causalMask: false, // BERT uses bidirectional attention
            attentionScale: 1.0 / Math.sqrt(headDim)
          }
        }
      );
      
      // Release intermediate tensors
      await this.cleanupTensors([query, key, value]);
      
      return attentionOutput;
    }
  }
  
  /**
   * Apply feed-forward network to transformer layer
   */
  private async applyFeedForward(
    input: any,
    layerIdx: number
  ): Promise<any> {
    // Get feed-forward weights
    const intermediateWeight = this.weights.get(`bert.encoder.layer.${layerIdx}.intermediate.dense.weight`)?.tensor;
    const intermediateBias = this.weights.get(`bert.encoder.layer.${layerIdx}.intermediate.dense.bias`)?.tensor;
    const outputWeight = this.weights.get(`bert.encoder.layer.${layerIdx}.output.dense.weight`)?.tensor;
    const outputBias = this.weights.get(`bert.encoder.layer.${layerIdx}.output.dense.bias`)?.tensor;
    
    if (!intermediateWeight || !intermediateBias || !outputWeight || !outputBias) {
      throw new Error(`Feed-forward weights not found for layer ${layerIdx}`);
    }
    
    // Execute feed-forward network using HAL
    return await this.hal.execute(
      'feedForward',
      {
        input,
        intermediateWeight,
        intermediateBias,
        outputWeight,
        outputBias
      },
      {
        modelType: 'text',
        backendOptions: {
          hiddenSize: this.config.hiddenSize,
          intermediateSize: this.config.intermediateSize,
          activationType: 'gelu',
          geluApproximation: 'tanh' // BERT typically uses tanh approximation
        }
      }
    );
  }
  
  /**
   * Apply add and layer normalization
   */
  private async applyAddAndLayerNorm(
    input: any,
    residual: any,
    weightKey: string,
    biasKey: string
  ): Promise<any> {
    const layerNormWeight = this.weights.get(weightKey)?.tensor;
    const layerNormBias = this.weights.get(biasKey)?.tensor;
    
    if (!layerNormWeight || !layerNormBias) {
      throw new Error(`Layer norm weights not found: ${weightKey}, ${biasKey}`);
    }
    
    // Execute add and layer norm using HAL
    return await this.hal.execute(
      'addLayerNorm',
      {
        input,
        residual,
        weight: layerNormWeight,
        bias: layerNormBias
      },
      {
        modelType: 'text',
        backendOptions: { epsilon: 1e-12 } // BERT uses 1e-12 for layer norm
      }
    );
  }
  
  /**
   * Get task-specific output based on config.taskType
   */
  private async getTaskOutput(lastHiddenState: any): Promise<BERTOutput> {
    const outputStartTime = performance.now();
    
    switch (this.config.taskType) {
      case 'embedding':
        // For embedding task, return the [CLS] token embedding
        const clsEmbedding = await this.hal.execute(
          'extractFirstToken',
          {
            input: lastHiddenState
          },
          { modelType: 'text' }
        );
        
        // Get the data as Float32Array
        const embeddingData = await this.hal.execute(
          'getTensorData',
          { input: clsEmbedding },
          { modelType: 'text' }
        ) as Float32Array;
        
        // Release tensor
        await this.cleanupTensors([clsEmbedding]);
        
        const embeddingEndTime = performance.now();
        this.recordOperationTime('embedding_output', embeddingEndTime - outputStartTime);
        
        return embeddingData;
        
      case 'sequence_classification':
        // For sequence classification, apply classifier to [CLS] token
        if (!this.config.numLabels) {
          throw new Error('numLabels must be specified for sequence_classification task');
        }
        
        const poolerWeight = this.weights.get('bert.pooler.dense.weight')?.tensor;
        const poolerBias = this.weights.get('bert.pooler.dense.bias')?.tensor;
        const classifierWeight = this.weights.get('classifier.weight')?.tensor;
        const classifierBias = this.weights.get('classifier.bias')?.tensor;
        
        if (!poolerWeight || !poolerBias || !classifierWeight || !classifierBias) {
          throw new Error('Classification weights not found');
        }
        
        // Execute classification using HAL
        const logits = await this.hal.execute(
          'bertClassification',
          {
            hiddenState: lastHiddenState,
            poolerWeight,
            poolerBias,
            classifierWeight,
            classifierBias
          },
          { modelType: 'text' }
        );
        
        // Get logits as Float32Array
        const logitsData = await this.hal.execute(
          'getTensorData',
          { input: logits },
          { modelType: 'text' }
        ) as Float32Array;
        
        // Release tensors
        await this.cleanupTensors([logits]);
        
        const classificationEndTime = performance.now();
        this.recordOperationTime('classification_output', classificationEndTime - outputStartTime);
        
        return { logits: logitsData };
        
      case 'question_answering':
        // For question answering, predict start and end positions
        const qaWeight = this.weights.get('qa_outputs.weight')?.tensor;
        const qaBias = this.weights.get('qa_outputs.bias')?.tensor;
        
        if (!qaWeight || !qaBias) {
          throw new Error('Question answering weights not found');
        }
        
        // Execute QA using HAL
        const qaLogits = await this.hal.execute(
          'bertQuestionAnswering',
          {
            hiddenState: lastHiddenState,
            qaWeight,
            qaBias
          },
          { modelType: 'text' }
        );
        
        // Get start and end logits
        const startLogitsData = await this.hal.execute(
          'getTensorData',
          { input: qaLogits.startLogits },
          { modelType: 'text' }
        ) as Float32Array;
        
        const endLogitsData = await this.hal.execute(
          'getTensorData',
          { input: qaLogits.endLogits },
          { modelType: 'text' }
        ) as Float32Array;
        
        // Release tensors
        await this.cleanupTensors([qaLogits.startLogits, qaLogits.endLogits]);
        
        const qaEndTime = performance.now();
        this.recordOperationTime('qa_output', qaEndTime - outputStartTime);
        
        return { 
          start_logits: startLogitsData, 
          end_logits: endLogitsData 
        };
        
      default:
        throw new Error(`Unsupported task type: ${this.config.taskType}`);
    }
  }
  
  /**
   * Clean up multiple tensors at once
   */
  private async cleanupTensors(tensors: any[]): Promise<void> {
    for (const tensor of tensors) {
      if (tensor) {
        try {
          await this.hal.execute(
            'releaseTensor',
            { tensor },
            { modelType: 'text' }
          );
        } catch (error) {
          console.warn('Error releasing tensor:', error);
          // Continue with cleanup even if one tensor fails
        }
      }
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
   * Get backend-specific metrics and information
   */
  getBackendMetrics(): Record<string, any> {
    const availableBackends = this.hal.getAvailableBackends();
    const metrics: Record<string, any> = {};
    
    for (const backendType of availableBackends) {
      const backend = this.hal.getBackend(backendType);
      if (backend) {
        metrics[backendType] = {
          isSelected: backend === this.hal.getBestBackend('text'),
          capabilities: backend.getCapabilities()
        };
      }
    }
    
    return metrics;
  }
  
  /**
   * Compare inference performance across all available backends
   */
  async compareBackends(text: string): Promise<Record<string, number>> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    const tokens = this.tokenize(text);
    const results: Record<string, number> = {};
    const availableBackends = this.hal.getAvailableBackends();
    
    for (const backendType of availableBackends) {
      try {
        console.log(`Testing backend: ${backendType}`);
        const startTime = performance.now();
        
        // Force specific backend
        await this.hal.execute(
          'predictBert',
          { 
            text: tokens,
            config: this.config 
          },
          { 
            modelType: 'text',
            preferredBackend: backendType as any
          }
        );
        
        const endTime = performance.now();
        results[backendType] = endTime - startTime;
        
      } catch (error) {
        console.error(`Error with backend ${backendType}:`, error);
        results[backendType] = -1; // Error indicator
      }
    }
    
    return results;
  }
  
  /**
   * Get model information
   */
  getModelInfo(): Record<string, any> {
    const bestBackend = this.hal?.getBestBackend('text');
    
    return {
      modelType: 'BERT',
      vocabSize: this.config.vocabSize,
      hiddenSize: this.config.hiddenSize,
      numLayers: this.config.numLayers,
      numHeads: this.config.numHeads,
      intermediateSize: this.config.intermediateSize,
      maxSequenceLength: this.config.maxSequenceLength,
      quantization: this.config.quantization,
      taskType: this.config.taskType,
      numLabels: this.config.numLabels,
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
    this.opExecutionTimes.clear();
  }
}