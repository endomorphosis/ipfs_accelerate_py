/**
 * Hardware Abstracted Vision Transformer (ViT) Implementation
 * Leverages the Hardware Abstraction Layer to choose the optimal backend
 * for ViT model execution based on available hardware and browser
 */

import { Tensor } from '../../tensor/tensor';
import { HardwareAbstractionLayer, ModelType, BackendSelectionCriteria } from '../../hardware/hardware_abstraction_layer';
import { SharedTensor } from '../../tensor/shared_tensor';
import { BrowserType } from '../../hardware/webgpu/browser_optimized_operations';

/**
 * Configuration for the Hardware Abstracted ViT model
 */
export interface HardwareAbstractedViTConfig {
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
  /** Number of channels in input images (3 for RGB) */
  channels?: number;
  /** Backend preference (will override automatic selection if specified) */
  backendPreference?: string[];
  /** Whether to prioritize speed over accuracy */
  prioritizeSpeed?: boolean;
  /** Whether to use quantization for lower memory usage */
  useQuantization?: boolean;
  /** Whether to enable tensor sharing with other models */
  enableTensorSharing?: boolean;
  /** Browser specific type to optimize for (auto-detected if not specified) */
  browserType?: BrowserType;
}

/**
 * Default ViT configuration (vit-base-patch16-224)
 */
export const DEFAULT_HARDWARE_ABSTRACTED_VIT_CONFIG: HardwareAbstractedViTConfig = {
  modelId: 'google/vit-base-patch16-224',
  imageSize: 224,
  patchSize: 16,
  hiddenSize: 768,
  numLayers: 12,
  numHeads: 12,
  intermediateSize: 3072,
  layerNormEps: 1e-12,
  numClasses: 1000,
  channels: 3,
  prioritizeSpeed: false,
  useQuantization: false,
  enableTensorSharing: true
};

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
  /** Browser type used for inference (if applicable) */
  browserType?: string;
}

/**
 * Hardware Abstracted Vision Transformer (ViT) model
 */
export class HardwareAbstractedViT {
  private config: HardwareAbstractedViTConfig;
  private hal: HardwareAbstractionLayer;
  private initialized: boolean = false;
  
  // Model weights
  private weights: {
    // Patch embedding
    patchEmbedding?: Tensor<number>,
    positionEmbeddings?: Tensor<number>,
    classToken?: Tensor<number>,
    layerNorm?: { weight?: Tensor<number>, bias?: Tensor<number> },
    
    // Encoder layers
    layers?: Array<{
      // Self Attention
      attention: {
        query: { weight?: Tensor<number>, bias?: Tensor<number> },
        key: { weight?: Tensor<number>, bias?: Tensor<number> },
        value: { weight?: Tensor<number>, bias?: Tensor<number> },
        output: { weight?: Tensor<number>, bias?: Tensor<number> },
        layerNorm: { weight?: Tensor<number>, bias?: Tensor<number> }
      },
      // Feed Forward
      intermediate: { weight?: Tensor<number>, bias?: Tensor<number> },
      output: { weight?: Tensor<number>, bias?: Tensor<number> },
      layerNorm: { weight?: Tensor<number>, bias?: Tensor<number> }
    }>,
    
    // Classification head
    classifier?: { weight?: Tensor<number>, bias?: Tensor<number> }
  } = {};
  
  // Shared tensors for cross-model sharing
  private sharedTensors: Map<string, SharedTensor> = new Map();
  
  /**
   * Constructor for Hardware Abstracted ViT model
   * @param hal Hardware Abstraction Layer instance
   * @param config ViT configuration
   */
  constructor(
    hal: HardwareAbstractionLayer,
    config: Partial<HardwareAbstractedViTConfig> = {}
  ) {
    this.hal = hal;
    this.config = { ...DEFAULT_HARDWARE_ABSTRACTED_VIT_CONFIG, ...config };
  }
  
  /**
   * Initialize the model by selecting the optimal backend and loading weights
   */
  public async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }
    
    console.log(`Initializing Hardware Abstracted ViT model: ${this.config.modelId}`);

    try {
      // Initialize the HAL if not already initialized
      if (!this.hal.isInitialized()) {
        await this.hal.initialize();
      }
      
      // Select the optimal backend for vision models
      const criteria: BackendSelectionCriteria = {
        modelType: 'vision' as ModelType,
        backendPreference: this.config.backendPreference as any[],
        modelSize: this.getModelSize(),
        prioritizeSpeed: this.config.prioritizeSpeed,
        memoryConstrained: this.isMemoryConstrained(),
        useQuantization: this.config.useQuantization,
        browserType: this.config.browserType
      };
      
      const selectedBackend = await this.hal.selectBackend(criteria);
      if (!selectedBackend) {
        throw new Error('No suitable backend found for ViT model');
      }
      
      // Set the selected backend as active
      this.hal.setActiveBackend(selectedBackend);
      console.log(`Selected backend for ViT model: ${selectedBackend.type}`);
      
      // Load model weights
      await this.loadWeights();
      
      this.initialized = true;
    } catch (error) {
      console.error('Error initializing Hardware Abstracted ViT model:', error);
      throw error;
    }
  }
  
  /**
   * Determine model size category based on parameters
   */
  private getModelSize(): 'tiny' | 'small' | 'base' | 'large' {
    const numParams = this.estimateModelParameters();
    
    if (numParams < 10_000_000) return 'tiny';
    if (numParams < 50_000_000) return 'small';
    if (numParams < 200_000_000) return 'base';
    return 'large';
  }
  
  /**
   * Estimate model parameters based on configuration
   */
  private estimateModelParameters(): number {
    const hiddenSize = this.config.hiddenSize;
    const intermediateSize = this.config.intermediateSize;
    const numLayers = this.config.numLayers;
    const numClasses = this.config.numClasses;
    const patchDim = this.config.channels! * this.config.patchSize * this.config.patchSize;
    
    // Embedding parameters
    const embeddingParams = patchDim * hiddenSize; // patch embedding
    const posEmbeddingParams = ((this.config.imageSize / this.config.patchSize) ** 2 + 1) * hiddenSize; // position embedding + class token
    const layerNormParams = 2 * hiddenSize; // layer norm (weight + bias)
    
    // Per-layer parameters
    const attentionParams = 4 * (hiddenSize * hiddenSize + hiddenSize); // query, key, value, output with biases
    const attentionLayerNormParams = 2 * hiddenSize; // layer norm (weight + bias)
    const ffnParams = hiddenSize * intermediateSize + intermediateSize + intermediateSize * hiddenSize + hiddenSize; // intermediate + output with biases
    const ffnLayerNormParams = 2 * hiddenSize; // layer norm (weight + bias)
    const layerParams = attentionParams + attentionLayerNormParams + ffnParams + ffnLayerNormParams;
    
    // Classification head parameters
    const classifierParams = hiddenSize * numClasses + numClasses; // weight + bias
    
    // Total parameters
    const totalParams = embeddingParams + posEmbeddingParams + layerNormParams + numLayers * layerParams + classifierParams;
    
    return totalParams;
  }
  
  /**
   * Determine if model should be considered memory constrained
   */
  private isMemoryConstrained(): boolean {
    // For web environments, consider large models memory constrained
    const numParams = this.estimateModelParameters();
    return numParams > 100_000_000;
  }
  
  /**
   * Load model weights
   */
  private async loadWeights(): Promise<void> {
    // In a real implementation, this would load weights from a file or API
    console.log(`Loading weights for ${this.config.modelId}...`);
    
    // Calculate dimensions
    const patchesPerSide = this.config.imageSize / this.config.patchSize;
    const numPatches = patchesPerSide * patchesPerSide;
    const patchDim = this.config.channels! * this.config.patchSize * this.config.patchSize;
    
    // Create patch embedding weights
    this.weights.patchEmbedding = await this.hal.createTensor({
      dimensions: [patchDim, this.config.hiddenSize],
      data: new Float32Array(patchDim * this.config.hiddenSize),
      dtype: 'float32'
    });
    
    // Create position embeddings (for patches + class token)
    this.weights.positionEmbeddings = await this.hal.createTensor({
      dimensions: [numPatches + 1, this.config.hiddenSize],
      data: new Float32Array((numPatches + 1) * this.config.hiddenSize),
      dtype: 'float32'
    });
    
    // Create class token
    this.weights.classToken = await this.hal.createTensor({
      dimensions: [1, this.config.hiddenSize],
      data: new Float32Array(this.config.hiddenSize),
      dtype: 'float32'
    });
    
    // Create layer norm weights
    this.weights.layerNorm = {
      weight: await this.hal.createTensor({
        dimensions: [this.config.hiddenSize],
        data: new Float32Array(this.config.hiddenSize).fill(1.0), // Initialize to ones
        dtype: 'float32'
      }),
      bias: await this.hal.createTensor({
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
            weight: await this.hal.createTensor({
              dimensions: [this.config.hiddenSize, this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize * this.config.hiddenSize),
              dtype: 'float32'
            }),
            bias: await this.hal.createTensor({
              dimensions: [this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize),
              dtype: 'float32'
            })
          },
          key: {
            weight: await this.hal.createTensor({
              dimensions: [this.config.hiddenSize, this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize * this.config.hiddenSize),
              dtype: 'float32'
            }),
            bias: await this.hal.createTensor({
              dimensions: [this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize),
              dtype: 'float32'
            })
          },
          value: {
            weight: await this.hal.createTensor({
              dimensions: [this.config.hiddenSize, this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize * this.config.hiddenSize),
              dtype: 'float32'
            }),
            bias: await this.hal.createTensor({
              dimensions: [this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize),
              dtype: 'float32'
            })
          },
          output: {
            weight: await this.hal.createTensor({
              dimensions: [this.config.hiddenSize, this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize * this.config.hiddenSize),
              dtype: 'float32'
            }),
            bias: await this.hal.createTensor({
              dimensions: [this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize),
              dtype: 'float32'
            })
          },
          layerNorm: {
            weight: await this.hal.createTensor({
              dimensions: [this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize).fill(1.0),
              dtype: 'float32'
            }),
            bias: await this.hal.createTensor({
              dimensions: [this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize).fill(0.0),
              dtype: 'float32'
            })
          }
        },
        // Feed Forward
        intermediate: {
          weight: await this.hal.createTensor({
            dimensions: [this.config.hiddenSize, this.config.intermediateSize],
            data: new Float32Array(this.config.hiddenSize * this.config.intermediateSize),
            dtype: 'float32'
          }),
          bias: await this.hal.createTensor({
            dimensions: [this.config.intermediateSize],
            data: new Float32Array(this.config.intermediateSize),
            dtype: 'float32'
          })
        },
        output: {
          weight: await this.hal.createTensor({
            dimensions: [this.config.intermediateSize, this.config.hiddenSize],
            data: new Float32Array(this.config.intermediateSize * this.config.hiddenSize),
            dtype: 'float32'
          }),
          bias: await this.hal.createTensor({
            dimensions: [this.config.hiddenSize],
            data: new Float32Array(this.config.hiddenSize),
            dtype: 'float32'
          })
        },
        layerNorm: {
          weight: await this.hal.createTensor({
            dimensions: [this.config.hiddenSize],
            data: new Float32Array(this.config.hiddenSize).fill(1.0),
            dtype: 'float32'
          }),
          bias: await this.hal.createTensor({
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
      weight: await this.hal.createTensor({
        dimensions: [this.config.hiddenSize, this.config.numClasses],
        data: new Float32Array(this.config.hiddenSize * this.config.numClasses),
        dtype: 'float32'
      }),
      bias: await this.hal.createTensor({
        dimensions: [this.config.numClasses],
        data: new Float32Array(this.config.numClasses),
        dtype: 'float32'
      })
    };
    
    console.log(`Weights loaded for ${this.config.modelId}`);
  }
  
  /**
   * Preprocess image for ViT model
   */
  private async preprocessImage(input: ViTInput): Promise<Tensor<number>> {
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
    const imageTensor = await this.hal.createTensor({
      dimensions: [1, this.config.channels!, height, width],
      data: normalizedData,
      dtype: 'float32'
    });
    
    // Normalize with ImageNet mean and std
    const meanTensor = await this.hal.createTensor({
      dimensions: [1, this.config.channels!, 1, 1],
      data: new Float32Array([0.485, 0.456, 0.406]), // ImageNet RGB mean
      dtype: 'float32'
    });
    
    const stdTensor = await this.hal.createTensor({
      dimensions: [1, this.config.channels!, 1, 1],
      data: new Float32Array([0.229, 0.224, 0.225]), // ImageNet RGB std
      dtype: 'float32'
    });
    
    // Subtract mean
    const centeredImage = await this.hal.subtract(imageTensor, meanTensor);
    
    // Divide by std
    const normalizedImage = await this.hal.divide(centeredImage, stdTensor);
    
    // Release temporary tensors
    await this.hal.releaseTensor(imageTensor);
    await this.hal.releaseTensor(meanTensor);
    await this.hal.releaseTensor(stdTensor);
    await this.hal.releaseTensor(centeredImage);
    
    return normalizedImage;
  }
  
  /**
   * Extract patches from the image
   */
  private async patchifyAndEmbed(image: Tensor<number>): Promise<Tensor<number>> {
    // Calculate dimensions
    const patchesPerSide = this.config.imageSize / this.config.patchSize;
    const numPatches = patchesPerSide * patchesPerSide;
    const patchDim = this.config.channels! * this.config.patchSize * this.config.patchSize;
    
    // Extract patches - in an actual implementation, this would use a 
    // specialized operation or unfold operation for efficiency
    // Here, we'll simulate the result
    
    // Simulate patches tensor with shape [1, numPatches, patchDim]
    const patchesTensor = await this.hal.createTensor({
      dimensions: [1, numPatches, patchDim],
      data: new Float32Array(numPatches * patchDim),
      dtype: 'float32'
    });
    
    // Apply patch projection
    const embeddedPatches = await this.hal.matmul(
      patchesTensor,
      this.weights.patchEmbedding!,
      { useOptimization: true }
    );
    
    // Release temporary tensor
    await this.hal.releaseTensor(patchesTensor);
    
    return embeddedPatches;
  }
  
  /**
   * Add class token and position embeddings
   */
  private async addClassTokenAndPositionEmbeddings(patchEmbeddings: Tensor<number>): Promise<Tensor<number>> {
    // Get dimensions
    const [batchSize, numPatches, hiddenSize] = patchEmbeddings.dimensions;
    
    // Create repeated class token for batch
    const classTokenRepeated = await this.hal.repeat(
      this.weights.classToken!,
      [batchSize, 1, 1] // Repeat over batch dimension
    );
    
    // Concatenate class token to beginning of patch embeddings
    const withClassToken = await this.hal.concat(
      [classTokenRepeated, patchEmbeddings],
      1 // Concat along sequence dimension
    );
    
    // Add position embeddings
    const positionEmbeddingsRepeated = await this.hal.repeat(
      this.weights.positionEmbeddings!,
      [batchSize, 1, 1] // Repeat over batch dimension
    );
    
    const withPositionEmbeddings = await this.hal.add(
      withClassToken,
      positionEmbeddingsRepeated
    );
    
    // Apply layer normalization
    const normalizedEmbeddings = await this.hal.layerNorm(
      withPositionEmbeddings,
      this.weights.layerNorm!.weight!,
      this.weights.layerNorm!.bias!,
      this.config.layerNormEps
    );
    
    // Release temporary tensors
    await this.hal.releaseTensor(classTokenRepeated);
    await this.hal.releaseTensor(withClassToken);
    await this.hal.releaseTensor(positionEmbeddingsRepeated);
    await this.hal.releaseTensor(withPositionEmbeddings);
    
    return normalizedEmbeddings;
  }
  
  /**
   * Run a single encoder layer
   */
  private async runEncoderLayer(
    hiddenStates: Tensor<number>,
    layerIndex: number
  ): Promise<Tensor<number>> {
    const layer = this.weights.layers![layerIndex];
    
    // Run self-attention
    const attentionOutput = await this.runSelfAttention(
      hiddenStates,
      layer.attention
    );
    
    // Apply attention output layer normalization
    const attentionNormalized = await this.hal.layerNorm(
      attentionOutput,
      layer.attention.layerNorm.weight!,
      layer.attention.layerNorm.bias!,
      this.config.layerNormEps
    );
    
    // Run feed-forward network with potential operation fusion
    const ffnOutput = await this.hal.ffnn(
      attentionNormalized,
      layer.intermediate.weight!,
      layer.intermediate.bias!,
      'gelu'
    );
    
    const ffnProjected = await this.hal.ffnn(
      ffnOutput,
      layer.output.weight!,
      layer.output.bias!,
      'none'
    );
    
    // Add residual connection
    const ffnWithResidual = await this.hal.add(ffnProjected, attentionNormalized);
    
    // Apply final layer normalization
    const outputNormalized = await this.hal.layerNorm(
      ffnWithResidual,
      layer.layerNorm.weight!,
      layer.layerNorm.bias!,
      this.config.layerNormEps
    );
    
    // Release temporary tensors
    await this.hal.releaseTensor(attentionOutput);
    await this.hal.releaseTensor(attentionNormalized);
    await this.hal.releaseTensor(ffnOutput);
    await this.hal.releaseTensor(ffnProjected);
    await this.hal.releaseTensor(ffnWithResidual);
    
    return outputNormalized;
  }
  
  /**
   * Run self-attention mechanism
   */
  private async runSelfAttention(
    hiddenStates: Tensor<number>,
    weights: {
      query: { weight?: Tensor<number>, bias?: Tensor<number> },
      key: { weight?: Tensor<number>, bias?: Tensor<number> },
      value: { weight?: Tensor<number>, bias?: Tensor<number> },
      output: { weight?: Tensor<number>, bias?: Tensor<number> },
      layerNorm: { weight?: Tensor<number>, bias?: Tensor<number> }
    }
  ): Promise<Tensor<number>> {
    // Get dimensions
    const [batchSize, seqLength, hiddenSize] = hiddenStates.dimensions;
    const numHeads = this.config.numHeads;
    const headSize = hiddenSize / numHeads;
    
    // Check if we can use a fused attention implementation
    if ('multiHeadAttention' in this.hal) {
      const extendedHal = this.hal as any;
      
      // Use fused multi-head attention
      const attentionOut = await extendedHal.multiHeadAttention(
        hiddenStates,
        weights.query.weight!,
        weights.query.bias!,
        weights.key.weight!,
        weights.key.bias!,
        weights.value.weight!,
        weights.value.bias!,
        weights.output.weight!,
        weights.output.bias!,
        {
          numHeads,
          headSize,
          addResidual: true
        }
      );
      
      return attentionOut;
    }
    
    // Otherwise implement attention using individual operations
    
    // Create query projection
    const query = await this.hal.ffnn(
      hiddenStates,
      weights.query.weight!,
      weights.query.bias!,
      'none'
    );
    
    // Create key projection
    const key = await this.hal.ffnn(
      hiddenStates,
      weights.key.weight!,
      weights.key.bias!,
      'none'
    );
    
    // Create value projection
    const value = await this.hal.ffnn(
      hiddenStates,
      weights.value.weight!,
      weights.value.bias!,
      'none'
    );
    
    // Reshape query, key, value to multi-head format
    const queryHeads = await this.hal.reshape(
      query,
      [batchSize, seqLength, numHeads, headSize]
    );
    const keyHeads = await this.hal.reshape(
      key,
      [batchSize, seqLength, numHeads, headSize]
    );
    const valueHeads = await this.hal.reshape(
      value,
      [batchSize, seqLength, numHeads, headSize]
    );
    
    // Transpose for matrix multiplication
    const queryTransposed = await this.hal.transpose(
      queryHeads,
      [0, 2, 1, 3] // [batch, heads, seq_len, head_size]
    );
    const keyTransposed = await this.hal.transpose(
      keyHeads,
      [0, 2, 3, 1] // [batch, heads, head_size, seq_len]
    );
    const valueTransposed = await this.hal.transpose(
      valueHeads,
      [0, 2, 1, 3] // [batch, heads, seq_len, head_size]
    );
    
    // Calculate attention scores
    const attentionScores = await this.hal.matmul(
      queryTransposed,
      keyTransposed,
      { useOptimization: true }
    );
    
    // Scale attention scores
    const scaleFactor = Math.sqrt(headSize);
    const scaleFactorTensor = await this.hal.createTensor({
      dimensions: [1],
      data: new Float32Array([1.0 / scaleFactor]),
      dtype: 'float32'
    });
    
    const scaledScores = await this.hal.multiply(
      attentionScores,
      scaleFactorTensor
    );
    
    // Apply softmax to get attention probabilities
    const attentionProbs = await this.hal.softmax(
      scaledScores,
      -1 // last dimension
    );
    
    // Apply attention to values
    const contextLayer = await this.hal.matmul(
      attentionProbs,
      valueTransposed,
      { useOptimization: true }
    );
    
    // Transpose and reshape context layer
    const contextTransposed = await this.hal.transpose(
      contextLayer,
      [0, 2, 1, 3] // [batch, seq_len, heads, head_size]
    );
    
    const contextReshaped = await this.hal.reshape(
      contextTransposed,
      [batchSize, seqLength, hiddenSize]
    );
    
    // Apply output projection
    const attentionOutput = await this.hal.ffnn(
      contextReshaped,
      weights.output.weight!,
      weights.output.bias!,
      'none'
    );
    
    // Add residual connection
    const outputWithResidual = await this.hal.add(
      attentionOutput,
      hiddenStates
    );
    
    // Release temporary tensors
    await this.hal.releaseTensor(query);
    await this.hal.releaseTensor(key);
    await this.hal.releaseTensor(value);
    await this.hal.releaseTensor(queryHeads);
    await this.hal.releaseTensor(keyHeads);
    await this.hal.releaseTensor(valueHeads);
    await this.hal.releaseTensor(queryTransposed);
    await this.hal.releaseTensor(keyTransposed);
    await this.hal.releaseTensor(valueTransposed);
    await this.hal.releaseTensor(attentionScores);
    await this.hal.releaseTensor(scaleFactorTensor);
    await this.hal.releaseTensor(scaledScores);
    await this.hal.releaseTensor(attentionProbs);
    await this.hal.releaseTensor(contextLayer);
    await this.hal.releaseTensor(contextTransposed);
    await this.hal.releaseTensor(contextReshaped);
    await this.hal.releaseTensor(attentionOutput);
    
    return outputWithResidual;
  }
  
  /**
   * Extract the class token representation for classification
   */
  private async extractClassToken(sequenceOutput: Tensor<number>): Promise<Tensor<number>> {
    // Get class token (first token)
    const classToken = await this.hal.slice(
      sequenceOutput,
      [0, 0, 0],
      [sequenceOutput.dimensions[0], 1, sequenceOutput.dimensions[2]]
    );
    
    // Reshape to [batch_size, hidden_size]
    const reshapedClassToken = await this.hal.reshape(
      classToken,
      [sequenceOutput.dimensions[0], sequenceOutput.dimensions[2]]
    );
    
    // Release temporary tensor
    await this.hal.releaseTensor(classToken);
    
    return reshapedClassToken;
  }
  
  /**
   * Run classifier on class token representation
   */
  private async runClassifier(classToken: Tensor<number>): Promise<Tensor<number>> {
    // Apply classifier as a feed-forward network
    return this.hal.ffnn(
      classToken,
      this.weights.classifier!.weight!,
      this.weights.classifier!.bias!,
      'none'
    );
  }
  
  /**
   * Run ViT model inference on image input
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
    
    // Store sequence output as shared tensor if tensor sharing is enabled
    if (this.config.enableTensorSharing) {
      const visionEmbedding = this.hal.createSharedTensor(
        sequenceOutput,
        'vision_embedding',
        this.config.modelId
      );
    }
    
    // Extract class token for classification
    const classToken = await this.extractClassToken(sequenceOutput);
    
    // Run classifier
    const logits = await this.runClassifier(classToken);
    
    // Apply softmax to get probabilities
    const probabilities = await this.hal.softmax(logits, -1);
    
    // Convert to arrays
    const logitsArray = await logits.toArray() as number[];
    const probabilitiesArray = await probabilities.toArray() as number[];
    
    // Find class with highest probability
    const classId = probabilitiesArray.indexOf(Math.max(...probabilitiesArray));
    
    // Release temporary tensors
    await this.hal.releaseTensor(preprocessedImage);
    await this.hal.releaseTensor(patchEmbeddings);
    await this.hal.releaseTensor(embeddings);
    await this.hal.releaseTensor(sequenceOutput);
    await this.hal.releaseTensor(classToken);
    await this.hal.releaseTensor(logits);
    await this.hal.releaseTensor(probabilities);
    
    return {
      logits: logitsArray,
      probabilities: probabilitiesArray,
      classId,
      model: this.config.modelId,
      backend: this.hal.getBackendType(),
      browserType: this.config.browserType
    };
  }
  
  /**
   * Get a shared tensor if available
   */
  public getSharedTensor(outputType: string = 'vision_embedding'): SharedTensor | null {
    if (!this.config.enableTensorSharing) {
      return null;
    }
    
    return this.hal.getSharedTensor(outputType, this.config.modelId);
  }
  
  /**
   * Dispose of all resources used by the model
   */
  public async dispose(): Promise<void> {
    if (!this.initialized) {
      return;
    }
    
    // Release all weights
    for (const key in this.weights) {
      const weight = this.weights[key as keyof typeof this.weights];
      
      if (weight instanceof Tensor) {
        await this.hal.releaseTensor(weight);
      } else if (typeof weight === 'object' && weight !== null) {
        // Handle nested weights
        for (const subKey in weight) {
          const subWeight = weight[subKey as keyof typeof weight];
          
          if (subWeight instanceof Tensor) {
            await this.hal.releaseTensor(subWeight);
          } else if (typeof subWeight === 'object' && subWeight !== null) {
            // Handle deeply nested weights
            for (const deepKey in subWeight) {
              const deepWeight = subWeight[deepKey as keyof typeof subWeight];
              
              if (deepWeight instanceof Tensor) {
                await this.hal.releaseTensor(deepWeight);
              }
            }
          }
        }
      }
    }
    
    // Clear weights object
    this.weights = {};
    
    // Reset initialized state
    this.initialized = false;
  }
}

/**
 * Factory function to create a Hardware Abstracted ViT model
 * @param hal Hardware Abstraction Layer instance
 * @param config ViT configuration
 * @returns Hardware Abstracted ViT model instance
 */
export function createHardwareAbstractedViT(
  hal: HardwareAbstractionLayer,
  config: Partial<HardwareAbstractedViTConfig> = {}
): HardwareAbstractedViT {
  return new HardwareAbstractedViT(hal, config);
}