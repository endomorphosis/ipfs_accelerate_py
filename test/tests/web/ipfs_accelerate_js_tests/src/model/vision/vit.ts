/**
 * Vision Transformer (ViT) model implementation with hardware acceleration support
 * Implements Google's Vision Transformer model with WebGPU and WebNN backends
 */

import { Tensor } from '../../tensor/tensor';
import { SharedTensor } from '../../tensor/shared_tensor';
import { TensorBackendType } from '../../tensor/index';
import { HardwareBackend } from '../../hardware/interfaces/hardware_backend';
import { matmul, softmax } from '../../tensor/operations/matrix';
import { layerNorm, gelu } from '../../tensor/operations/nn';
import { add, mul } from '../../tensor/operations/basic';
import { ShaderType } from '../../hardware/webgpu/optimizations/browser_shader_loader';
import { BrowserType, detectBrowserType } from '../../hardware/webgpu/browser_optimized_operations';

/**
 * Configuration for the ViT model
 */
export interface ViTConfig {
  /** Model identifier (e.g., "google/vit-base-patch16-224") */
  modelId: string;
  /** Image size (height and width in pixels) */
  imageSize: number;
  /** Patch size for image tokenization */
  patchSize: number;
  /** Hidden size (embedding dimension) */
  hiddenSize: number;
  /** Number of layers/blocks in the model */
  numLayers: number;
  /** Number of attention heads */
  numHeads: number;
  /** Intermediate size in feed-forward networks */
  intermediateSize: number;
  /** Layer normalization epsilon */
  layerNormEps: number;
  /** Number of classes for classification (e.g., 1000 for ImageNet) */
  numClasses: number;
  /** Hardware backend type preference */
  backendPreference?: TensorBackendType[];
  /** Whether to use optimized operations */
  useOptimizedOps?: boolean;
  /** Number of channels in input images (3 for RGB) */
  channels?: number;
  /** Whether to use browser-specific optimizations */
  useBrowserOptimizations?: boolean;
  /** Specific browser type to optimize for (auto-detected if not specified) */
  browserType?: BrowserType;
  /** Whether to use operation fusion when possible */
  useOperationFusion?: boolean;
  /** Attention dropout probability */
  attentionDropout?: number;
}

/**
 * Default ViT configuration (vit-base-patch16-224)
 */
export const DEFAULT_VIT_CONFIG: ViTConfig = {
  modelId: 'google/vit-base-patch16-224',
  imageSize: 224,
  patchSize: 16,
  hiddenSize: 768,
  numLayers: 12,
  numHeads: 12,
  intermediateSize: 3072,
  layerNormEps: 1e-12,
  numClasses: 1000,
  backendPreference: ['webgpu', 'webnn', 'cpu'],
  useOptimizedOps: true,
  channels: 3,
  useBrowserOptimizations: true,
  useOperationFusion: true,
  attentionDropout: 0.0
};

/**
 * Attention output from ViT self-attention
 */
interface ViTAttentionOutput {
  /** Attention output */
  output: Tensor;
  /** Attention weights (optional, for visualization) */
  weights?: Tensor;
}

/**
 * Input format for ViT inference
 */
export interface ViTInput {
  /** 
   * Image data in RGB format [0-255] 
   * Can be provided as Float32Array (normalized to [0-1]) or Uint8Array ([0-255])
   */
  imageData: Float32Array | Uint8Array;
  /** Image width */
  width: number;
  /** Image height */
  height: number;
  /** Whether the image is already preprocessed (normalized) */
  isPreprocessed?: boolean;
}

/**
 * Output format for ViT inference
 */
export interface ViTOutput {
  /** Classification logits (unnormalized) */
  logits: number[];
  /** Class probabilities (softmax applied) */
  probabilities: number[];
  /** Predicted class ID (index of highest probability) */
  classId: number;
  /** Model identifier */
  model: string;
  /** Backend used for inference */
  backend: string;
}

/**
 * Vision Transformer model implementation with hardware acceleration
 */
export class ViT {
  private config: ViTConfig;
  private hardware: HardwareBackend;
  private initialized: boolean = false;
  
  // Model weights (loaded on demand)
  private weights: {
    // Patch embedding
    patchEmbedding?: Tensor,
    positionEmbeddings?: Tensor,
    classToken?: Tensor,
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
    
    // Classification head
    classifier?: { weight?: Tensor, bias?: Tensor }
  } = {};
  
  private sharedTensors: Map<string, SharedTensor> = new Map();
  
  /**
   * Constructor for ViT model
   * @param hardware Hardware backend for tensor operations
   * @param config ViT configuration
   */
  constructor(
    hardware: HardwareBackend,
    config: Partial<ViTConfig> = {}
  ) {
    this.hardware = hardware;
    this.config = { ...DEFAULT_VIT_CONFIG, ...config };
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
      console.error('Error initializing ViT model:', error);
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
    
    // Calculate dimensions
    const patchesPerSide = this.config.imageSize / this.config.patchSize;
    const numPatches = patchesPerSide * patchesPerSide;
    const patchDim = this.config.channels * this.config.patchSize * this.config.patchSize;
    
    // Create patch embedding weights
    this.weights.patchEmbedding = await this.hardware.createTensor({
      dimensions: [patchDim, this.config.hiddenSize],
      data: new Float32Array(patchDim * this.config.hiddenSize),
      dtype: 'float32'
    });
    
    // Create position embeddings (for patches + class token)
    this.weights.positionEmbeddings = await this.hardware.createTensor({
      dimensions: [numPatches + 1, this.config.hiddenSize],
      data: new Float32Array((numPatches + 1) * this.config.hiddenSize),
      dtype: 'float32'
    });
    
    // Create class token
    this.weights.classToken = await this.hardware.createTensor({
      dimensions: [1, this.config.hiddenSize],
      data: new Float32Array(this.config.hiddenSize),
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
    
    // Classification head weights
    this.weights.classifier = {
      weight: await this.hardware.createTensor({
        dimensions: [this.config.hiddenSize, this.config.numClasses],
        data: new Float32Array(this.config.hiddenSize * this.config.numClasses),
        dtype: 'float32'
      }),
      bias: await this.hardware.createTensor({
        dimensions: [this.config.numClasses],
        data: new Float32Array(this.config.numClasses),
        dtype: 'float32'
      })
    };
    
    console.log(`Weights loaded for ${this.config.modelId}`);
  }
  
  /**
   * Preprocess image for ViT model
   * @param input Input image data
   * @returns Preprocessed image tensor
   */
  private async preprocessImage(input: ViTInput): Promise<Tensor> {
    // Prepare the image data
    const { imageData, width, height, isPreprocessed } = input;
    
    // Check image dimensions
    if (width !== this.config.imageSize || height !== this.config.imageSize) {
      throw new Error(`Image dimensions must be ${this.config.imageSize}x${this.config.imageSize}`);
    }
    
    // Convert to Float32Array if needed
    let normalizedData: Float32Array;
    if (imageData instanceof Uint8Array) {
      // Convert from 0-255 to 0-1
      normalizedData = new Float32Array(imageData.length);
      for (let i = 0; i < imageData.length; i++) {
        normalizedData[i] = imageData[i] / 255.0;
      }
    } else {
      // Already float32, check if normalized
      normalizedData = imageData;
      if (!isPreprocessed) {
        // Normalize if needed (assuming values are 0-255)
        for (let i = 0; i < normalizedData.length; i++) {
          normalizedData[i] = normalizedData[i] / 255.0;
        }
      }
    }
    
    // Create tensor with shape [1, channels, height, width]
    // Note: May need to adjust based on channel order (RGB vs BGR)
    // and channel first vs channel last conventions
    const imageTensor = await this.hardware.createTensor({
      dimensions: [1, this.config.channels, height, width],
      data: normalizedData,
      dtype: 'float32'
    });
    
    // Normalize with ImageNet mean and std
    const meanTensor = await this.hardware.createTensor({
      dimensions: [1, this.config.channels, 1, 1],
      data: new Float32Array([0.485, 0.456, 0.406]), // ImageNet RGB mean
      dtype: 'float32'
    });
    
    const stdTensor = await this.hardware.createTensor({
      dimensions: [1, this.config.channels, 1, 1],
      data: new Float32Array([0.229, 0.224, 0.225]), // ImageNet RGB std
      dtype: 'float32'
    });
    
    // Subtract mean
    const centeredImage = await this.hardware.sub(imageTensor, meanTensor);
    
    // Divide by std
    const normalizedImage = await this.hardware.div(centeredImage, stdTensor);
    
    // Release temporary tensors
    await this.hardware.releaseTensor(imageTensor);
    await this.hardware.releaseTensor(meanTensor);
    await this.hardware.releaseTensor(stdTensor);
    await this.hardware.releaseTensor(centeredImage);
    
    return normalizedImage;
  }
  
  /**
   * Extract patches from the image
   * @param image Preprocessed image tensor
   * @returns Patch embedding tensor
   */
  private async patchifyAndEmbed(image: Tensor): Promise<Tensor> {
    // Calculate dimensions
    const patchesPerSide = this.config.imageSize / this.config.patchSize;
    const numPatches = patchesPerSide * patchesPerSide;
    const patchDim = this.config.channels * this.config.patchSize * this.config.patchSize;
    
    // Extract patches - in an actual implementation, this would use a 
    // specialized operation or unfold operation for efficiency
    // Here, we'll simulate the result
    
    // Simulate patches tensor with shape [1, numPatches, patchDim]
    const patchesTensor = await this.hardware.createTensor({
      dimensions: [1, numPatches, patchDim],
      data: new Float32Array(numPatches * patchDim),
      dtype: 'float32'
    });
    
    // Apply patch projection
    const embeddedPatches = await matmul(
      patchesTensor,
      this.weights.patchEmbedding!,
      this.hardware,
      { useOptimization: this.config.useOptimizedOps }
    );
    
    // Release temporary tensor
    await this.hardware.releaseTensor(patchesTensor);
    
    return embeddedPatches;
  }
  
  /**
   * Add class token and position embeddings
   * @param patchEmbeddings Embedded patches
   * @returns Sequence with class token and position embeddings
   */
  private async addClassTokenAndPositionEmbeddings(patchEmbeddings: Tensor): Promise<Tensor> {
    // Get dimensions
    const [batchSize, numPatches, hiddenSize] = patchEmbeddings.dimensions;
    
    // Create repeated class token for batch
    const classTokenRepeated = await this.hardware.repeat(
      this.weights.classToken!,
      [batchSize, 1, 1] // Repeat over batch dimension
    );
    
    // Concatenate class token to beginning of patch embeddings
    const withClassToken = await this.hardware.concat(
      [classTokenRepeated, patchEmbeddings],
      1 // Concat along sequence dimension
    );
    
    // Add position embeddings
    const positionEmbeddingsRepeated = await this.hardware.repeat(
      this.weights.positionEmbeddings!,
      [batchSize, 1, 1] // Repeat over batch dimension
    );
    
    const withPositionEmbeddings = await add(
      withClassToken,
      positionEmbeddingsRepeated,
      this.hardware
    );
    
    // Apply layer normalization
    const normalizedEmbeddings = await layerNorm(
      withPositionEmbeddings,
      this.weights.layerNorm!.weight!,
      this.weights.layerNorm!.bias!,
      this.config.layerNormEps,
      this.hardware
    );
    
    // Release temporary tensors
    await this.hardware.releaseTensor(classTokenRepeated);
    await this.hardware.releaseTensor(withClassToken);
    await this.hardware.releaseTensor(positionEmbeddingsRepeated);
    await this.hardware.releaseTensor(withPositionEmbeddings);
    
    return normalizedEmbeddings;
  }
  
  /**
   * Run a single ViT encoder layer
   * @param hiddenStates Input hidden states
   * @param layerIndex Index of the encoder layer
   * @returns Output hidden states
   */
  private async runEncoderLayer(
    hiddenStates: Tensor,
    layerIndex: number
  ): Promise<Tensor> {
    const layer = this.weights.layers![layerIndex];
    
    // Run self-attention
    const { output: attentionOutput } = await this.runSelfAttention(
      hiddenStates,
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
   * @param weights Attention weights
   * @returns Attention output
   */
  private async runSelfAttention(
    hiddenStates: Tensor,
    weights: {
      query: { weight?: Tensor, bias?: Tensor },
      key: { weight?: Tensor, bias?: Tensor },
      value: { weight?: Tensor, bias?: Tensor },
      output: { weight?: Tensor, bias?: Tensor },
      layerNorm: { weight?: Tensor, bias?: Tensor }
    }
  ): Promise<ViTAttentionOutput> {
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
    
    // Apply softmax to get attention probabilities
    const attentionProbs = await softmax(
      scaledScores,
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
    
    // Store attention weights for visualization
    const attentionWeights = await attentionProbs.clone();
    
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
    await this.hardware.releaseTensor(attentionProbs);
    await this.hardware.releaseTensor(contextLayer);
    await this.hardware.releaseTensor(contextTransposed);
    await this.hardware.releaseTensor(contextReshaped);
    await this.hardware.releaseTensor(attentionOutput);
    await this.hardware.releaseTensor(outputBiased);
    
    return {
      output: outputWithResidual,
      weights: attentionWeights
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
   * Extract the class token representation
   * @param sequenceOutput Full sequence output
   * @returns Class token representation
   */
  private async extractClassToken(sequenceOutput: Tensor): Promise<Tensor> {
    // Get class token (first token)
    const classToken = await this.hardware.slice(
      sequenceOutput,
      [0, 0, 0],
      [sequenceOutput.dimensions[0], 1, sequenceOutput.dimensions[2]]
    );
    
    // Reshape to [batch_size, hidden_size]
    const reshapedClassToken = await this.hardware.reshape(
      classToken,
      [sequenceOutput.dimensions[0], sequenceOutput.dimensions[2]]
    );
    
    // Release temporary tensor
    await this.hardware.releaseTensor(classToken);
    
    return reshapedClassToken;
  }
  
  /**
   * Run classifier on class token representation
   * @param classToken Class token representation
   * @returns Logits for each class
   */
  private async runClassifier(classToken: Tensor): Promise<Tensor> {
    // Apply classifier
    const logits = await matmul(
      classToken,
      this.weights.classifier!.weight!,
      this.hardware,
      { useOptimization: this.config.useOptimizedOps }
    );
    
    // Add bias
    const logitsWithBias = await add(
      logits,
      this.weights.classifier!.bias!,
      this.hardware
    );
    
    // Release temporary tensor
    await this.hardware.releaseTensor(logits);
    
    return logitsWithBias;
  }
  
  /**
   * Run ViT model inference on image input
   * @param input Input image data
   * @returns ViT output with classification results
   */
  public async process(input: ViTInput): Promise<ViTOutput> {
    // Ensure model is initialized
    if (!this.initialized) {
      await this.initialize();
    }
    
    // Preprocess image
    const preprocessedImage = await this.preprocessImage(input);
    
    // Patchify and embed
    const patchEmbeddings = await this.patchifyAndEmbed(preprocessedImage);
    
    // Add class token and position embeddings
    const embeddings = await this.addClassTokenAndPositionEmbeddings(patchEmbeddings);
    
    // Run encoder layers
    let sequenceOutput = embeddings;
    for (let i = 0; i < this.config.numLayers; i++) {
      sequenceOutput = await this.runEncoderLayer(sequenceOutput, i);
    }
    
    // Store sequence output as shared tensor if needed
    const visionEmbedding = await this.createSharedTensor(
      sequenceOutput,
      'vision_embedding'
    );
    
    // Extract class token for classification
    const classToken = await this.extractClassToken(sequenceOutput);
    
    // Run classifier
    const logits = await this.runClassifier(classToken);
    
    // Apply softmax to get probabilities
    const probabilities = await softmax(logits, -1, this.hardware);
    
    // Convert to arrays
    const logitsArray = await logits.toArray() as number[];
    const probabilitiesArray = await probabilities.toArray() as number[];
    
    // Find class with highest probability
    const classId = probabilitiesArray.indexOf(Math.max(...probabilitiesArray));
    
    // Release temporary tensors
    await this.hardware.releaseTensor(preprocessedImage);
    await this.hardware.releaseTensor(patchEmbeddings);
    await this.hardware.releaseTensor(embeddings);
    await this.hardware.releaseTensor(sequenceOutput);
    await this.hardware.releaseTensor(classToken);
    await this.hardware.releaseTensor(logits);
    await this.hardware.releaseTensor(probabilities);
    
    return {
      logits: logitsArray,
      probabilities: probabilitiesArray,
      classId,
      model: this.config.modelId,
      backend: this.hardware.getBackendType()
    };
  }
  
  /**
   * Create a shared tensor that can be used by other models
   * @param sequenceOutput Output of encoder 
   * @param outputType Type of output to share
   * @returns Shared tensor reference
   */
  public async createSharedTensor(
    sequenceOutput: Tensor,
    outputType: string = 'vision_embedding'
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
  public getSharedTensor(outputType: string = 'vision_embedding'): SharedTensor | null {
    const key = `${outputType}_${this.config.modelId}`;
    return this.sharedTensors.get(key) || null;
  }
  
  /**
   * Dispose of all resources used by the model
   */
  public async dispose(): Promise<void> {
    if (!this.initialized) {
      return;
    }
    
    // Release patch embedding weights
    if (this.weights.patchEmbedding) {
      await this.hardware.releaseTensor(this.weights.patchEmbedding);
    }
    
    if (this.weights.positionEmbeddings) {
      await this.hardware.releaseTensor(this.weights.positionEmbeddings);
    }
    
    if (this.weights.classToken) {
      await this.hardware.releaseTensor(this.weights.classToken);
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
    
    // Release classifier weights
    if (this.weights.classifier) {
      if (this.weights.classifier.weight) {
        await this.hardware.releaseTensor(this.weights.classifier.weight);
      }
      if (this.weights.classifier.bias) {
        await this.hardware.releaseTensor(this.weights.classifier.bias);
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
 * Factory function to create a ViT model
 * @param hardware Hardware backend for tensor operations
 * @param config ViT configuration
 * @returns ViT model instance
 */
export function createVitModel(
  hardware: HardwareBackend,
  config: Partial<ViTConfig> = {}
): ViT {
  return new ViT(hardware, config);
}