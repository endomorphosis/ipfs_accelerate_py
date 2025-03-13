/**
 * BERT model implementation with hardware acceleration support
 * Implements a BERT model with support for WebGPU and WebNN backends
 */

import { Tensor } from '../../tensor/tensor';
import { SharedTensor } from '../../tensor/shared_tensor';
import { TensorBackendType } from '../../tensor/index';
import { HardwareBackend } from '../../hardware/interfaces/hardware_backend';
import { matmul, softmax } from '../../tensor/operations/matrix';
import { layerNorm, gelu } from '../../tensor/operations/nn';
import { add, mul } from '../../tensor/operations/basic';

/**
 * Configuration for the BERT model
 */
export interface BertConfig {
  /** Model identifier (e.g., "bert-base-uncased") */
  modelId: string;
  /** Vocabulary size */
  vocabSize: number;
  /** Hidden size (typically 768 for base models) */
  hiddenSize: number;
  /** Number of layers/blocks in the model */
  numLayers: number;
  /** Number of attention heads */
  numHeads: number;
  /** Intermediate size in feed-forward networks */
  intermediateSize: number;
  /** Maximum sequence length */
  maxPositions: number;
  /** Layer normalization epsilon */
  layerNormEps: number;
  /** Hardware backend type preference */
  backendPreference?: TensorBackendType[];
  /** Whether to use optimized operations */
  useOptimizedOps?: boolean;
}

/**
 * Default BERT configuration (bert-base-uncased)
 */
export const DEFAULT_BERT_CONFIG: BertConfig = {
  modelId: 'bert-base-uncased',
  vocabSize: 30522,
  hiddenSize: 768,
  numLayers: 12,
  numHeads: 12,
  intermediateSize: 3072,
  maxPositions: 512,
  layerNormEps: 1e-12,
  backendPreference: ['webgpu', 'webnn', 'cpu'],
  useOptimizedOps: true
};

/**
 * Attention output from BERT self-attention
 */
interface BertAttentionOutput {
  /** Attention output */
  output: Tensor;
  /** Attention weights (optional, for analysis) */
  weights?: Tensor;
}

/**
 * Input format for BERT inference
 */
export interface BertInput {
  /** Input token IDs */
  inputIds: number[];
  /** Attention mask (1 for tokens to attend to, 0 for padding) */
  attentionMask?: number[];
  /** Token type IDs for segment embeddings */
  tokenTypeIds?: number[];
}

/**
 * Output format for BERT inference
 */
export interface BertOutput {
  /** Last hidden state (sequence output) */
  lastHiddenState: number[][];
  /** Pooled output (CLS token representation) */
  pooledOutput?: number[];
  /** Model identifier */
  model: string;
  /** Backend used for inference */
  backend: string;
}

/**
 * BERT model implementation with hardware acceleration
 */
export class Bert {
  private config: BertConfig;
  private hardware: HardwareBackend;
  private initialized: boolean = false;
  
  // Model weights (loaded on demand)
  private weights: {
    // Embeddings
    wordEmbeddings?: Tensor,
    positionEmbeddings?: Tensor,
    tokenTypeEmbeddings?: Tensor,
    layerNorm?: { weight?: Tensor, bias?: Tensor },
    
    // Encoder layers
    layers?: Array<{
      // Self Attention
      attention: {
        query: { weight?: Tensor, bias?: Tensor },
        key: { weight?: Tensor, bias?: Tensor },
        value: { weight?: Tensor, bias?: Tensor },
        output: { weight?: Tensor, bias?: Tensor },
        layerNorm: { weight?: Tensor, bias?: Tensor }
      },
      // Feed Forward
      intermediate: { weight?: Tensor, bias?: Tensor },
      output: { weight?: Tensor, bias?: Tensor },
      layerNorm: { weight?: Tensor, bias?: Tensor }
    }>,
    
    // Pooler
    pooler?: { weight?: Tensor, bias?: Tensor }
  } = {};
  
  private sharedTensors: Map<string, SharedTensor> = new Map();
  
  /**
   * Constructor for BERT model
   * @param hardware Hardware backend for tensor operations
   * @param config BERT configuration
   */
  constructor(
    hardware: HardwareBackend,
    config: Partial<BertConfig> = {}
  ) {
    this.hardware = hardware;
    this.config = { ...DEFAULT_BERT_CONFIG, ...config };
  }
  
  /**
   * Initialize the model by loading weights and preparing for inference
   */
  public async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }
    
    try {
      // Initialize the hardware backend if not already initialized
      if (!this.hardware.isInitialized()) {
        await this.hardware.initialize();
      }
      
      // Load model weights
      await this.loadWeights();
      
      this.initialized = true;
    } catch (error) {
      console.error('Error initializing BERT model:', error);
      throw error;
    }
  }
  
  /**
   * Load model weights from storage or remote source
   */
  private async loadWeights(): Promise<void> {
    // In a real implementation, this would load weights from a file or API
    // For now, we'll simulate with placeholder weights
    console.log(`Loading weights for ${this.config.modelId}...`);
    
    // Create placeholder for embedding weights
    this.weights.wordEmbeddings = await this.hardware.createTensor({
      dimensions: [this.config.vocabSize, this.config.hiddenSize],
      data: new Float32Array(this.config.vocabSize * this.config.hiddenSize),
      dtype: 'float32'
    });
    
    this.weights.positionEmbeddings = await this.hardware.createTensor({
      dimensions: [this.config.maxPositions, this.config.hiddenSize],
      data: new Float32Array(this.config.maxPositions * this.config.hiddenSize),
      dtype: 'float32'
    });
    
    this.weights.tokenTypeEmbeddings = await this.hardware.createTensor({
      dimensions: [2, this.config.hiddenSize],
      data: new Float32Array(2 * this.config.hiddenSize),
      dtype: 'float32'
    });
    
    // Create layer norm weights
    this.weights.layerNorm = {
      weight: await this.hardware.createTensor({
        dimensions: [this.config.hiddenSize],
        data: new Float32Array(this.config.hiddenSize).fill(1.0), // Initialize to ones
        dtype: 'float32'
      }),
      bias: await this.hardware.createTensor({
        dimensions: [this.config.hiddenSize],
        data: new Float32Array(this.config.hiddenSize).fill(0.0), // Initialize to zeros
        dtype: 'float32'
      })
    };
    
    // Create encoder layers
    this.weights.layers = [];
    for (let i = 0; i < this.config.numLayers; i++) {
      const layer = {
        // Self Attention
        attention: {
          query: {
            weight: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize, this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize * this.config.hiddenSize),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize),
              dtype: 'float32'
            })
          },
          key: {
            weight: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize, this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize * this.config.hiddenSize),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize),
              dtype: 'float32'
            })
          },
          value: {
            weight: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize, this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize * this.config.hiddenSize),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize),
              dtype: 'float32'
            })
          },
          output: {
            weight: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize, this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize * this.config.hiddenSize),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize),
              dtype: 'float32'
            })
          },
          layerNorm: {
            weight: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize).fill(1.0),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize).fill(0.0),
              dtype: 'float32'
            })
          }
        },
        // Feed Forward
        intermediate: {
          weight: await this.hardware.createTensor({
            dimensions: [this.config.hiddenSize, this.config.intermediateSize],
            data: new Float32Array(this.config.hiddenSize * this.config.intermediateSize),
            dtype: 'float32'
          }),
          bias: await this.hardware.createTensor({
            dimensions: [this.config.intermediateSize],
            data: new Float32Array(this.config.intermediateSize),
            dtype: 'float32'
          })
        },
        output: {
          weight: await this.hardware.createTensor({
            dimensions: [this.config.intermediateSize, this.config.hiddenSize],
            data: new Float32Array(this.config.intermediateSize * this.config.hiddenSize),
            dtype: 'float32'
          }),
          bias: await this.hardware.createTensor({
            dimensions: [this.config.hiddenSize],
            data: new Float32Array(this.config.hiddenSize),
            dtype: 'float32'
          })
        },
        layerNorm: {
          weight: await this.hardware.createTensor({
            dimensions: [this.config.hiddenSize],
            data: new Float32Array(this.config.hiddenSize).fill(1.0),
            dtype: 'float32'
          }),
          bias: await this.hardware.createTensor({
            dimensions: [this.config.hiddenSize],
            data: new Float32Array(this.config.hiddenSize).fill(0.0),
            dtype: 'float32'
          })
        }
      };
      
      this.weights.layers.push(layer);
    }
    
    // Pooler weights
    this.weights.pooler = {
      weight: await this.hardware.createTensor({
        dimensions: [this.config.hiddenSize, this.config.hiddenSize],
        data: new Float32Array(this.config.hiddenSize * this.config.hiddenSize),
        dtype: 'float32'
      }),
      bias: await this.hardware.createTensor({
        dimensions: [this.config.hiddenSize],
        data: new Float32Array(this.config.hiddenSize),
        dtype: 'float32'
      })
    };
    
    console.log(`Weights loaded for ${this.config.modelId}`);
  }
  
  /**
   * Tokenize input text into token IDs
   * @param text Input text to tokenize
   * @returns Tokenized input object
   */
  public async tokenize(text: string): Promise<BertInput> {
    // In a real implementation, this would use a tokenizer
    // For now, we'll return a simple placeholder
    console.log(`Tokenizing text: ${text}`);
    
    // Create dummy token IDs (in a real implementation, this would use a tokenizer)
    const tokenIds = Array.from({length: Math.min(text.length + 2, this.config.maxPositions)}, 
      (_, i) => i === 0 ? 101 : (i === text.length + 1 ? 102 : 1000 + i)); // [CLS], tokens, [SEP]
    
    return {
      inputIds: tokenIds,
      attentionMask: tokenIds.map(() => 1), // All tokens attended to
      tokenTypeIds: tokenIds.map(() => 0)   // All tokens from segment 0
    };
  }
  
  /**
   * Run BERT model inference on tokenized input
   * @param input Tokenized input
   * @returns BERT output with embeddings
   */
  public async process(input: BertInput): Promise<BertOutput> {
    // Ensure model is initialized
    if (!this.initialized) {
      await this.initialize();
    }
    
    // Convert input to tensors
    const inputTensors = await this.prepareInputTensors(input);
    
    // Get embeddings
    const embeddings = await this.getEmbeddings(
      inputTensors.inputIds,
      inputTensors.tokenTypeIds,
      inputTensors.attentionMask
    );
    
    // Run encoder layers
    let sequenceOutput = embeddings;
    for (let i = 0; i < this.config.numLayers; i++) {
      sequenceOutput = await this.runEncoderLayer(sequenceOutput, inputTensors.attentionMask, i);
    }
    
    // Get pooled output (CLS token representation)
    const clsOutput = await this.runPooler(sequenceOutput);
    
    // Convert output tensors to arrays
    const lastHiddenState = await sequenceOutput.toArray() as number[][];
    const pooledOutput = await clsOutput.toArray() as number[];
    
    // Release temporary tensors
    await this.hardware.releaseTensor(inputTensors.inputIds);
    await this.hardware.releaseTensor(inputTensors.tokenTypeIds);
    await this.hardware.releaseTensor(inputTensors.attentionMask);
    await this.hardware.releaseTensor(sequenceOutput);
    await this.hardware.releaseTensor(clsOutput);
    
    return {
      lastHiddenState,
      pooledOutput,
      model: this.config.modelId,
      backend: this.hardware.getBackendType()
    };
  }
  
  /**
   * Prepare input tensors from tokenized input
   * @param input Tokenized input
   * @returns Tensor representations of inputs
   */
  private async prepareInputTensors(input: BertInput): Promise<{
    inputIds: Tensor,
    tokenTypeIds: Tensor,
    attentionMask: Tensor
  }> {
    const seqLength = input.inputIds.length;
    
    // Create input ID tensor
    const inputIdsTensor = await this.hardware.createTensor({
      dimensions: [1, seqLength],
      data: new Int32Array(input.inputIds),
      dtype: 'int32'
    });
    
    // Create token type ID tensor (default to all zeros if not provided)
    const tokenTypeIds = input.tokenTypeIds || new Array(seqLength).fill(0);
    const tokenTypeIdsTensor = await this.hardware.createTensor({
      dimensions: [1, seqLength],
      data: new Int32Array(tokenTypeIds),
      dtype: 'int32'
    });
    
    // Create attention mask tensor (default to all ones if not provided)
    const attentionMask = input.attentionMask || new Array(seqLength).fill(1);
    const attentionMaskTensor = await this.hardware.createTensor({
      dimensions: [1, seqLength],
      data: new Float32Array(attentionMask),
      dtype: 'float32'
    });
    
    return {
      inputIds: inputIdsTensor,
      tokenTypeIds: tokenTypeIdsTensor,
      attentionMask: attentionMaskTensor
    };
  }
  
  /**
   * Get embeddings from input IDs, token type IDs, and position IDs
   * @param inputIds Input token IDs
   * @param tokenTypeIds Token type IDs
   * @param attentionMask Attention mask
   * @returns Embedded representation
   */
  private async getEmbeddings(
    inputIds: Tensor,
    tokenTypeIds: Tensor,
    attentionMask: Tensor
  ): Promise<Tensor> {
    // Get dimensions
    const [batchSize, seqLength] = inputIds.dimensions;
    
    // Get word embeddings for input IDs
    const wordEmbeddings = await this.hardware.gatherEmbedding(
      this.weights.wordEmbeddings!,
      inputIds
    );
    
    // Create position IDs tensor [0, 1, 2, ...]
    const positionIds = await this.hardware.createTensor({
      dimensions: [1, seqLength],
      data: new Int32Array(Array.from({length: seqLength}, (_, i) => i)),
      dtype: 'int32'
    });
    
    // Get position embeddings
    const posEmbeddings = await this.hardware.gatherEmbedding(
      this.weights.positionEmbeddings!,
      positionIds
    );
    
    // Get token type embeddings
    const typeEmbeddings = await this.hardware.gatherEmbedding(
      this.weights.tokenTypeEmbeddings!,
      tokenTypeIds
    );
    
    // Add all embeddings
    const embeddings1 = await add(wordEmbeddings, posEmbeddings, this.hardware);
    const combinedEmbeddings = await add(embeddings1, typeEmbeddings, this.hardware);
    
    // Apply layer normalization
    const normalizedEmbeddings = await layerNorm(
      combinedEmbeddings,
      this.weights.layerNorm!.weight!,
      this.weights.layerNorm!.bias!,
      this.config.layerNormEps,
      this.hardware
    );
    
    // Release temporary tensors
    await this.hardware.releaseTensor(wordEmbeddings);
    await this.hardware.releaseTensor(positionIds);
    await this.hardware.releaseTensor(posEmbeddings);
    await this.hardware.releaseTensor(typeEmbeddings);
    await this.hardware.releaseTensor(embeddings1);
    await this.hardware.releaseTensor(combinedEmbeddings);
    
    return normalizedEmbeddings;
  }
  
  /**
   * Run a single BERT encoder layer
   * @param hiddenStates Input hidden states
   * @param attentionMask Attention mask
   * @param layerIndex Index of the encoder layer
   * @returns Output hidden states
   */
  private async runEncoderLayer(
    hiddenStates: Tensor,
    attentionMask: Tensor,
    layerIndex: number
  ): Promise<Tensor> {
    const layer = this.weights.layers![layerIndex];
    
    // Run self-attention
    const { output: attentionOutput } = await this.runSelfAttention(
      hiddenStates,
      attentionMask,
      layer.attention
    );
    
    // Apply attention output layer normalization
    const attentionNormalized = await layerNorm(
      attentionOutput,
      layer.attention.layerNorm.weight!,
      layer.attention.layerNorm.bias!,
      this.config.layerNormEps,
      this.hardware
    );
    
    // Run feed-forward network
    const intermediate = await this.runIntermediate(
      attentionNormalized,
      layer.intermediate.weight!,
      layer.intermediate.bias!
    );
    
    const ffnOutput = await this.runFeedForwardOutput(
      intermediate,
      attentionNormalized,
      layer.output.weight!,
      layer.output.bias!
    );
    
    // Apply final layer normalization
    const outputNormalized = await layerNorm(
      ffnOutput,
      layer.layerNorm.weight!,
      layer.layerNorm.bias!,
      this.config.layerNormEps,
      this.hardware
    );
    
    // Release temporary tensors
    await this.hardware.releaseTensor(attentionOutput);
    await this.hardware.releaseTensor(attentionNormalized);
    await this.hardware.releaseTensor(intermediate);
    await this.hardware.releaseTensor(ffnOutput);
    
    return outputNormalized;
  }
  
  /**
   * Run self-attention mechanism
   * @param hiddenStates Input hidden states
   * @param attentionMask Attention mask
   * @param weights Attention weights
   * @returns Attention output
   */
  private async runSelfAttention(
    hiddenStates: Tensor,
    attentionMask: Tensor,
    weights: {
      query: { weight?: Tensor, bias?: Tensor },
      key: { weight?: Tensor, bias?: Tensor },
      value: { weight?: Tensor, bias?: Tensor },
      output: { weight?: Tensor, bias?: Tensor },
      layerNorm: { weight?: Tensor, bias?: Tensor }
    }
  ): Promise<BertAttentionOutput> {
    // Get dimensions
    const [batchSize, seqLength, hiddenSize] = hiddenStates.dimensions;
    const numAttentionHeads = this.config.numHeads;
    const attentionHeadSize = hiddenSize / numAttentionHeads;
    
    // Create query projection
    const query = await matmul(
      hiddenStates,
      weights.query.weight!,
      this.hardware,
      { useOptimization: this.config.useOptimizedOps }
    );
    const queryBiased = await add(query, weights.query.bias!, this.hardware);
    
    // Create key projection
    const key = await matmul(
      hiddenStates,
      weights.key.weight!,
      this.hardware,
      { useOptimization: this.config.useOptimizedOps }
    );
    const keyBiased = await add(key, weights.key.bias!, this.hardware);
    
    // Create value projection
    const value = await matmul(
      hiddenStates,
      weights.value.weight!,
      this.hardware,
      { useOptimization: this.config.useOptimizedOps }
    );
    const valueBiased = await add(value, weights.value.bias!, this.hardware);
    
    // Reshape query, key, value to multi-head format
    const queryHeads = await this.hardware.reshape(
      queryBiased,
      [batchSize, seqLength, numAttentionHeads, attentionHeadSize]
    );
    const keyHeads = await this.hardware.reshape(
      keyBiased,
      [batchSize, seqLength, numAttentionHeads, attentionHeadSize]
    );
    const valueHeads = await this.hardware.reshape(
      valueBiased,
      [batchSize, seqLength, numAttentionHeads, attentionHeadSize]
    );
    
    // Transpose for matrix multiplication
    const queryTransposed = await this.hardware.transpose(
      queryHeads,
      [0, 2, 1, 3] // [batch, heads, seq_len, head_size]
    );
    const keyTransposed = await this.hardware.transpose(
      keyHeads,
      [0, 2, 3, 1] // [batch, heads, head_size, seq_len]
    );
    const valueTransposed = await this.hardware.transpose(
      valueHeads,
      [0, 2, 1, 3] // [batch, heads, seq_len, head_size]
    );
    
    // Calculate attention scores
    const attentionScores = await matmul(
      queryTransposed,
      keyTransposed,
      this.hardware,
      { useOptimization: this.config.useOptimizedOps }
    );
    
    // Scale attention scores
    const scaleFactor = Math.sqrt(attentionHeadSize);
    const scaledScores = await this.hardware.mul(
      attentionScores,
      await this.hardware.createTensor({
        dimensions: [1],
        data: new Float32Array([1.0 / scaleFactor]),
        dtype: 'float32'
      })
    );
    
    // Apply attention mask
    const extendedAttentionMask = await this.hardware.reshape(
      attentionMask,
      [batchSize, 1, 1, seqLength]
    );
    
    // Create a large negative tensor to mask attention where mask is 0
    const negativeInfinity = await this.hardware.createTensor({
      dimensions: [1],
      data: new Float32Array([-10000.0]),
      dtype: 'float32'
    });
    
    const invertedMask = await this.hardware.sub(
      await this.hardware.createTensor({
        dimensions: extendedAttentionMask.dimensions,
        data: new Float32Array(
          extendedAttentionMask.dimensions.reduce((a, b) => a * b, 1)
        ).fill(1.0),
        dtype: 'float32'
      }),
      extendedAttentionMask
    );
    
    const maskedNegativeInfinity = await this.hardware.mul(
      invertedMask,
      negativeInfinity
    );
    
    const maskedScores = await add(
      scaledScores,
      maskedNegativeInfinity,
      this.hardware
    );
    
    // Apply softmax to get attention probabilities
    const attentionProbs = await softmax(
      maskedScores,
      -1, // axis = -1 (last dimension)
      this.hardware
    );
    
    // Apply attention to values
    const contextLayer = await matmul(
      attentionProbs,
      valueTransposed,
      this.hardware,
      { useOptimization: this.config.useOptimizedOps }
    );
    
    // Transpose and reshape context layer
    const contextTransposed = await this.hardware.transpose(
      contextLayer,
      [0, 2, 1, 3] // [batch, seq_len, heads, head_size]
    );
    
    const contextReshaped = await this.hardware.reshape(
      contextTransposed,
      [batchSize, seqLength, hiddenSize]
    );
    
    // Apply output projection
    const attentionOutput = await matmul(
      contextReshaped,
      weights.output.weight!,
      this.hardware,
      { useOptimization: this.config.useOptimizedOps }
    );
    
    const outputBiased = await add(
      attentionOutput,
      weights.output.bias!,
      this.hardware
    );
    
    // Add residual connection
    const outputWithResidual = await add(
      outputBiased,
      hiddenStates,
      this.hardware
    );
    
    // Release temporary tensors
    await this.hardware.releaseTensor(query);
    await this.hardware.releaseTensor(queryBiased);
    await this.hardware.releaseTensor(key);
    await this.hardware.releaseTensor(keyBiased);
    await this.hardware.releaseTensor(value);
    await this.hardware.releaseTensor(valueBiased);
    await this.hardware.releaseTensor(queryHeads);
    await this.hardware.releaseTensor(keyHeads);
    await this.hardware.releaseTensor(valueHeads);
    await this.hardware.releaseTensor(queryTransposed);
    await this.hardware.releaseTensor(keyTransposed);
    await this.hardware.releaseTensor(valueTransposed);
    await this.hardware.releaseTensor(attentionScores);
    await this.hardware.releaseTensor(scaledScores);
    await this.hardware.releaseTensor(extendedAttentionMask);
    await this.hardware.releaseTensor(negativeInfinity);
    await this.hardware.releaseTensor(invertedMask);
    await this.hardware.releaseTensor(maskedNegativeInfinity);
    await this.hardware.releaseTensor(maskedScores);
    await this.hardware.releaseTensor(attentionProbs);
    await this.hardware.releaseTensor(contextLayer);
    await this.hardware.releaseTensor(contextTransposed);
    await this.hardware.releaseTensor(contextReshaped);
    await this.hardware.releaseTensor(attentionOutput);
    await this.hardware.releaseTensor(outputBiased);
    
    return {
      output: outputWithResidual
    };
  }
  
  /**
   * Run intermediate/feed-forward layer
   * @param hiddenStates Input hidden states
   * @param weight Intermediate weight matrix
   * @param bias Intermediate bias
   * @returns Intermediate hidden states
   */
  private async runIntermediate(
    hiddenStates: Tensor,
    weight: Tensor,
    bias: Tensor
  ): Promise<Tensor> {
    // Linear projection
    const intermediate = await matmul(
      hiddenStates,
      weight,
      this.hardware,
      { useOptimization: this.config.useOptimizedOps }
    );
    
    // Add bias
    const biased = await add(intermediate, bias, this.hardware);
    
    // Apply GELU activation
    const activated = await gelu(biased, this.hardware);
    
    // Release temporary tensors
    await this.hardware.releaseTensor(intermediate);
    await this.hardware.releaseTensor(biased);
    
    return activated;
  }
  
  /**
   * Run feed-forward output layer
   * @param intermediate Intermediate hidden states
   * @param inputTensor Original input tensor (for residual connection)
   * @param weight Output weight matrix
   * @param bias Output bias
   * @returns Output hidden states
   */
  private async runFeedForwardOutput(
    intermediate: Tensor,
    inputTensor: Tensor,
    weight: Tensor,
    bias: Tensor
  ): Promise<Tensor> {
    // Linear projection
    const output = await matmul(
      intermediate,
      weight,
      this.hardware,
      { useOptimization: this.config.useOptimizedOps }
    );
    
    // Add bias
    const biased = await add(output, bias, this.hardware);
    
    // Add residual connection
    const withResidual = await add(biased, inputTensor, this.hardware);
    
    // Release temporary tensors
    await this.hardware.releaseTensor(output);
    await this.hardware.releaseTensor(biased);
    
    return withResidual;
  }
  
  /**
   * Run pooler on the output of the encoder
   * @param sequenceOutput Output of the encoder
   * @returns Pooled output (typically CLS token representation)
   */
  private async runPooler(sequenceOutput: Tensor): Promise<Tensor> {
    // Get the first token (CLS) hidden state
    const firstTokenTensor = await this.hardware.slice(
      sequenceOutput,
      [0, 0, 0],
      [sequenceOutput.dimensions[0], 1, sequenceOutput.dimensions[2]]
    );
    
    const reshapedCls = await this.hardware.reshape(
      firstTokenTensor,
      [sequenceOutput.dimensions[0], sequenceOutput.dimensions[2]]
    );
    
    // Apply linear layer
    const pooled = await matmul(
      reshapedCls,
      this.weights.pooler!.weight!,
      this.hardware,
      { useOptimization: this.config.useOptimizedOps }
    );
    
    // Add bias
    const biased = await add(pooled, this.weights.pooler!.bias!, this.hardware);
    
    // Apply tanh activation
    const activated = await this.hardware.tanh(biased);
    
    // Release temporary tensors
    await this.hardware.releaseTensor(firstTokenTensor);
    await this.hardware.releaseTensor(reshapedCls);
    await this.hardware.releaseTensor(pooled);
    await this.hardware.releaseTensor(biased);
    
    return activated;
  }
  
  /**
   * Create a shared tensor that can be used by other models
   * @param sequenceOutput Output of encoder 
   * @param outputType Type of output to share
   * @returns Shared tensor reference
   */
  public async createSharedTensor(
    sequenceOutput: Tensor,
    outputType: string = 'text_embedding'
  ): Promise<SharedTensor> {
    // Create a new shared tensor
    const sharedTensor = new SharedTensor(
      sequenceOutput,
      outputType,
      this.config.modelId,
      this.hardware.getBackendType()
    );
    
    // Store in shared tensors map
    const key = `${outputType}_${this.config.modelId}`;
    this.sharedTensors.set(key, sharedTensor);
    
    return sharedTensor;
  }
  
  /**
   * Get a shared tensor if available
   * @param outputType Type of output to get
   * @returns Shared tensor or null if not found
   */
  public getSharedTensor(outputType: string = 'text_embedding'): SharedTensor | null {
    const key = `${outputType}_${this.config.modelId}`;
    return this.sharedTensors.get(key) || null;
  }
  
  /**
   * Dispose of all resources used by the model
   */
  public async dispose(): Promise<void> {
    // Release all weights
    if (this.weights.wordEmbeddings) {
      await this.hardware.releaseTensor(this.weights.wordEmbeddings);
    }
    
    if (this.weights.positionEmbeddings) {
      await this.hardware.releaseTensor(this.weights.positionEmbeddings);
    }
    
    if (this.weights.tokenTypeEmbeddings) {
      await this.hardware.releaseTensor(this.weights.tokenTypeEmbeddings);
    }
    
    if (this.weights.layerNorm) {
      if (this.weights.layerNorm.weight) {
        await this.hardware.releaseTensor(this.weights.layerNorm.weight);
      }
      if (this.weights.layerNorm.bias) {
        await this.hardware.releaseTensor(this.weights.layerNorm.bias);
      }
    }
    
    // Release encoder layer weights
    if (this.weights.layers) {
      for (const layer of this.weights.layers) {
        // Attention weights
        if (layer.attention.query.weight) {
          await this.hardware.releaseTensor(layer.attention.query.weight);
        }
        if (layer.attention.query.bias) {
          await this.hardware.releaseTensor(layer.attention.query.bias);
        }
        
        if (layer.attention.key.weight) {
          await this.hardware.releaseTensor(layer.attention.key.weight);
        }
        if (layer.attention.key.bias) {
          await this.hardware.releaseTensor(layer.attention.key.bias);
        }
        
        if (layer.attention.value.weight) {
          await this.hardware.releaseTensor(layer.attention.value.weight);
        }
        if (layer.attention.value.bias) {
          await this.hardware.releaseTensor(layer.attention.value.bias);
        }
        
        if (layer.attention.output.weight) {
          await this.hardware.releaseTensor(layer.attention.output.weight);
        }
        if (layer.attention.output.bias) {
          await this.hardware.releaseTensor(layer.attention.output.bias);
        }
        
        if (layer.attention.layerNorm.weight) {
          await this.hardware.releaseTensor(layer.attention.layerNorm.weight);
        }
        if (layer.attention.layerNorm.bias) {
          await this.hardware.releaseTensor(layer.attention.layerNorm.bias);
        }
        
        // Intermediate and output weights
        if (layer.intermediate.weight) {
          await this.hardware.releaseTensor(layer.intermediate.weight);
        }
        if (layer.intermediate.bias) {
          await this.hardware.releaseTensor(layer.intermediate.bias);
        }
        
        if (layer.output.weight) {
          await this.hardware.releaseTensor(layer.output.weight);
        }
        if (layer.output.bias) {
          await this.hardware.releaseTensor(layer.output.bias);
        }
        
        if (layer.layerNorm.weight) {
          await this.hardware.releaseTensor(layer.layerNorm.weight);
        }
        if (layer.layerNorm.bias) {
          await this.hardware.releaseTensor(layer.layerNorm.bias);
        }
      }
    }
    
    // Release pooler weights
    if (this.weights.pooler) {
      if (this.weights.pooler.weight) {
        await this.hardware.releaseTensor(this.weights.pooler.weight);
      }
      if (this.weights.pooler.bias) {
        await this.hardware.releaseTensor(this.weights.pooler.bias);
      }
    }
    
    // Release shared tensors
    for (const [key, sharedTensor] of this.sharedTensors.entries()) {
      await sharedTensor.release();
      this.sharedTensors.delete(key);
    }
    
    // Reset initialized state
    this.initialized = false;
  }
}

/**
 * Factory function to create a BERT model
 * @param hardware Hardware backend for tensor operations
 * @param config BERT configuration
 * @returns BERT model instance
 */
export function createBertModel(
  hardware: HardwareBackend,
  config: Partial<BertConfig> = {}
): Bert {
  return new Bert(hardware, config);
}