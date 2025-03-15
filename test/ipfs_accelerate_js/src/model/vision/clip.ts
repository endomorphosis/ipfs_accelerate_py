/**
 * CLIP (Contrastive Language-Image Pre-training) model implementation
 * Implements OpenAI's CLIP model with WebGPU and WebNN backends
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
 * Configuration for the CLIP model
 */
export interface ClipConfig {
  /** Model identifier (e.g., "openai/clip-vit-base-patch32") */
  modelId: string;
  /** Image size (height and width in pixels) */
  imageSize: number;
  /** Patch size for image tokenization */
  patchSize: number;
  /** Hidden size (embedding dimension) */
  hiddenSize: number;
  /** Text embedding dimension */
  textEmbeddingSize?: number;
  /** Vision embedding dimension */
  visionEmbeddingSize?: number;
  /** Projection dimension (joint text-image space) */
  projectionDim: number;
  /** Number of layers in the vision transformer */
  visionNumLayers: number;
  /** Number of layers in the text transformer */
  textNumLayers: number;
  /** Number of attention heads in vision transformer */
  visionNumHeads: number;
  /** Number of attention heads in text transformer */
  textNumHeads: number;
  /** Intermediate size in vision feed-forward networks */
  visionIntermediateSize: number;
  /** Intermediate size in text feed-forward networks */
  textIntermediateSize: number;
  /** Layer normalization epsilon */
  layerNormEps: number;
  /** Maximum text length */
  maxTextLength: number;
  /** Hardware backend preference */
  backendPreference?: TensorBackendType[];
  /** Whether to use optimized operations */
  useOptimizedOps?: boolean;
  /** Whether to use browser-specific optimizations */
  useBrowserOptimizations?: boolean;
  /** Specific browser type to optimize for (auto-detected if not specified) */
  browserType?: BrowserType;
  /** Whether to use operation fusion when possible */
  useOperationFusion?: boolean;
}

/**
 * Default CLIP configuration (clip-vit-base-patch32)
 */
export const DEFAULT_CLIP_CONFIG: ClipConfig = {
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
  useOperationFusion: true
};

/**
 * Input format for CLIP image encoder
 */
export interface ClipImageInput {
  /** Raw image data */
  imageData: Float32Array | Uint8Array;
  /** Image width */
  width: number;
  /** Image height */
  height: number;
  /** Whether the image is already preprocessed (normalized) */
  isPreprocessed?: boolean;
}

/**
 * Input format for CLIP text encoder
 */
export interface ClipTextInput {
  /** Input text */
  text: string;
  /** Token IDs (if already tokenized) */
  inputIds?: number[];
  /** Attention mask */
  attentionMask?: number[];
}

/**
 * Output format for CLIP
 */
export interface ClipOutput {
  /** Text embeddings */
  textEmbeddings?: number[];
  /** Image embeddings */
  imageEmbeddings?: number[];
  /** Similarity scores (if both text and image provided) */
  similarity?: number;
  /** Model identifier */
  model: string;
  /** Backend used for inference */
  backend: string;
}

/**
 * CLIP model implementation with hardware acceleration
 */
export class Clip {
  private config: ClipConfig;
  private hardware: HardwareBackend;
  private initialized: boolean = false;
  
  // Model weights (loaded on demand)
  private weights: {
    // Vision model weights
    vision?: {
      embedding?: Tensor,
      positionalEmbedding?: Tensor,
      layers?: Array<{
        attention: {
          query: { weight?: Tensor, bias?: Tensor },
          key: { weight?: Tensor, bias?: Tensor },
          value: { weight?: Tensor, bias?: Tensor },
          output: { weight?: Tensor, bias?: Tensor },
        },
        layerNorm1: { weight?: Tensor, bias?: Tensor },
        mlp: {
          fc1: { weight?: Tensor, bias?: Tensor },
          fc2: { weight?: Tensor, bias?: Tensor },
        },
        layerNorm2: { weight?: Tensor, bias?: Tensor },
      }>,
      finalLayerNorm?: { weight?: Tensor, bias?: Tensor },
      visualProjection?: Tensor,
    },
    
    // Text model weights
    text?: {
      embedding?: Tensor,
      positionalEmbedding?: Tensor,
      layers?: Array<{
        attention: {
          query: { weight?: Tensor, bias?: Tensor },
          key: { weight?: Tensor, bias?: Tensor },
          value: { weight?: Tensor, bias?: Tensor },
          output: { weight?: Tensor, bias?: Tensor },
        },
        layerNorm1: { weight?: Tensor, bias?: Tensor },
        mlp: {
          fc1: { weight?: Tensor, bias?: Tensor },
          fc2: { weight?: Tensor, bias?: Tensor },
        },
        layerNorm2: { weight?: Tensor, bias?: Tensor },
      }>,
      finalLayerNorm?: { weight?: Tensor, bias?: Tensor },
      textProjection?: Tensor,
    },
    
    // Joint Projection
    logitScale?: Tensor,
  } = {};
  
  // Shared tensors for cross-model sharing
  private sharedTensors: Map<string, SharedTensor> = new Map();
  
  /**
   * Constructor for CLIP model
   * @param hardware Hardware backend for tensor operations
   * @param config CLIP configuration
   */
  constructor(
    hardware: HardwareBackend,
    config: Partial<ClipConfig> = {}
  ) {
    this.hardware = hardware;
    this.config = { ...DEFAULT_CLIP_CONFIG, ...config };
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
      console.log(`CLIP model initialized (${this.hardware.type})`);
    } catch (error) {
      console.error('Error initializing CLIP model:', error);
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
    
    // Initialize vision model weights
    this.weights.vision = {
      embedding: await this.hardware.createTensor({
        dimensions: [this.config.patchSize * this.config.patchSize * 3, this.config.hiddenSize],
        data: new Float32Array(this.config.patchSize * this.config.patchSize * 3 * this.config.hiddenSize),
        dtype: 'float32'
      }),
      positionalEmbedding: await this.hardware.createTensor({
        dimensions: [(this.config.imageSize / this.config.patchSize) ** 2 + 1, this.config.hiddenSize],
        data: new Float32Array(((this.config.imageSize / this.config.patchSize) ** 2 + 1) * this.config.hiddenSize),
        dtype: 'float32'
      }),
      layers: [],
      finalLayerNorm: {
        weight: await this.hardware.createTensor({
          dimensions: [this.config.hiddenSize],
          data: new Float32Array(this.config.hiddenSize).fill(1.0),
          dtype: 'float32'
        }),
        bias: await this.hardware.createTensor({
          dimensions: [this.config.hiddenSize],
          data: new Float32Array(this.config.hiddenSize),
          dtype: 'float32'
        }),
      },
      visualProjection: await this.hardware.createTensor({
        dimensions: [this.config.hiddenSize, this.config.projectionDim],
        data: new Float32Array(this.config.hiddenSize * this.config.projectionDim),
        dtype: 'float32'
      }),
    };
    
    // Create vision layers
    for (let i = 0; i < this.config.visionNumLayers; i++) {
      const layer = {
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
            }),
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
            }),
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
            }),
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
            }),
          },
        },
        layerNorm1: {
          weight: await this.hardware.createTensor({
            dimensions: [this.config.hiddenSize],
            data: new Float32Array(this.config.hiddenSize).fill(1.0),
            dtype: 'float32'
          }),
          bias: await this.hardware.createTensor({
            dimensions: [this.config.hiddenSize],
            data: new Float32Array(this.config.hiddenSize),
            dtype: 'float32'
          }),
        },
        mlp: {
          fc1: {
            weight: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize, this.config.visionIntermediateSize],
              data: new Float32Array(this.config.hiddenSize * this.config.visionIntermediateSize),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [this.config.visionIntermediateSize],
              data: new Float32Array(this.config.visionIntermediateSize),
              dtype: 'float32'
            }),
          },
          fc2: {
            weight: await this.hardware.createTensor({
              dimensions: [this.config.visionIntermediateSize, this.config.hiddenSize],
              data: new Float32Array(this.config.visionIntermediateSize * this.config.hiddenSize),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize),
              dtype: 'float32'
            }),
          },
        },
        layerNorm2: {
          weight: await this.hardware.createTensor({
            dimensions: [this.config.hiddenSize],
            data: new Float32Array(this.config.hiddenSize).fill(1.0),
            dtype: 'float32'
          }),
          bias: await this.hardware.createTensor({
            dimensions: [this.config.hiddenSize],
            data: new Float32Array(this.config.hiddenSize),
            dtype: 'float32'
          }),
        },
      };
      
      this.weights.vision.layers!.push(layer);
    }
    
    // Initialize text model weights
    const textEmbeddingSize = this.config.textEmbeddingSize || this.config.hiddenSize;
    
    this.weights.text = {
      embedding: await this.hardware.createTensor({
        dimensions: [49408, textEmbeddingSize], // CLIP uses 49408 vocab size
        data: new Float32Array(49408 * textEmbeddingSize),
        dtype: 'float32'
      }),
      positionalEmbedding: await this.hardware.createTensor({
        dimensions: [this.config.maxTextLength, textEmbeddingSize],
        data: new Float32Array(this.config.maxTextLength * textEmbeddingSize),
        dtype: 'float32'
      }),
      layers: [],
      finalLayerNorm: {
        weight: await this.hardware.createTensor({
          dimensions: [textEmbeddingSize],
          data: new Float32Array(textEmbeddingSize).fill(1.0),
          dtype: 'float32'
        }),
        bias: await this.hardware.createTensor({
          dimensions: [textEmbeddingSize],
          data: new Float32Array(textEmbeddingSize),
          dtype: 'float32'
        }),
      },
      textProjection: await this.hardware.createTensor({
        dimensions: [textEmbeddingSize, this.config.projectionDim],
        data: new Float32Array(textEmbeddingSize * this.config.projectionDim),
        dtype: 'float32'
      }),
    };
    
    // Create text layers
    for (let i = 0; i < this.config.textNumLayers; i++) {
      const layer = {
        attention: {
          query: {
            weight: await this.hardware.createTensor({
              dimensions: [textEmbeddingSize, textEmbeddingSize],
              data: new Float32Array(textEmbeddingSize * textEmbeddingSize),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [textEmbeddingSize],
              data: new Float32Array(textEmbeddingSize),
              dtype: 'float32'
            }),
          },
          key: {
            weight: await this.hardware.createTensor({
              dimensions: [textEmbeddingSize, textEmbeddingSize],
              data: new Float32Array(textEmbeddingSize * textEmbeddingSize),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [textEmbeddingSize],
              data: new Float32Array(textEmbeddingSize),
              dtype: 'float32'
            }),
          },
          value: {
            weight: await this.hardware.createTensor({
              dimensions: [textEmbeddingSize, textEmbeddingSize],
              data: new Float32Array(textEmbeddingSize * textEmbeddingSize),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [textEmbeddingSize],
              data: new Float32Array(textEmbeddingSize),
              dtype: 'float32'
            }),
          },
          output: {
            weight: await this.hardware.createTensor({
              dimensions: [textEmbeddingSize, textEmbeddingSize],
              data: new Float32Array(textEmbeddingSize * textEmbeddingSize),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [textEmbeddingSize],
              data: new Float32Array(textEmbeddingSize),
              dtype: 'float32'
            }),
          },
        },
        layerNorm1: {
          weight: await this.hardware.createTensor({
            dimensions: [textEmbeddingSize],
            data: new Float32Array(textEmbeddingSize).fill(1.0),
            dtype: 'float32'
          }),
          bias: await this.hardware.createTensor({
            dimensions: [textEmbeddingSize],
            data: new Float32Array(textEmbeddingSize),
            dtype: 'float32'
          }),
        },
        mlp: {
          fc1: {
            weight: await this.hardware.createTensor({
              dimensions: [textEmbeddingSize, this.config.textIntermediateSize],
              data: new Float32Array(textEmbeddingSize * this.config.textIntermediateSize),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [this.config.textIntermediateSize],
              data: new Float32Array(this.config.textIntermediateSize),
              dtype: 'float32'
            }),
          },
          fc2: {
            weight: await this.hardware.createTensor({
              dimensions: [this.config.textIntermediateSize, textEmbeddingSize],
              data: new Float32Array(this.config.textIntermediateSize * textEmbeddingSize),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [textEmbeddingSize],
              data: new Float32Array(textEmbeddingSize),
              dtype: 'float32'
            }),
          },
        },
        layerNorm2: {
          weight: await this.hardware.createTensor({
            dimensions: [textEmbeddingSize],
            data: new Float32Array(textEmbeddingSize).fill(1.0),
            dtype: 'float32'
          }),
          bias: await this.hardware.createTensor({
            dimensions: [textEmbeddingSize],
            data: new Float32Array(textEmbeddingSize),
            dtype: 'float32'
          }),
        },
      };
      
      this.weights.text.layers!.push(layer);
    }
    
    // Initialize logit scale (temperature parameter)
    this.weights.logitScale = await this.hardware.createTensor({
      dimensions: [1],
      data: new Float32Array([Math.log(1/0.07)]), // Default temperature from CLIP paper
      dtype: 'float32'
    });
    
    console.log(`Weights loaded for ${this.config.modelId}`);
  }
  
  /**
   * Preprocess image data
   * @param imageInput Image data
   * @returns Preprocessed image tensor
   */
  private async preprocessImage(imageInput: ClipImageInput): Promise<Tensor> {
    const { imageData, width, height, isPreprocessed } = imageInput;
    
    // Check dimensions
    if (width !== this.config.imageSize || height !== this.config.imageSize) {
      throw new Error(`Image must be ${this.config.imageSize}x${this.config.imageSize}`);
    }
    
    // Convert to Float32Array if needed
    let normalizedData: Float32Array;
    if (imageData instanceof Uint8Array) {
      // Convert from 0-255 to normalized range
      normalizedData = new Float32Array(imageData.length);
      for (let i = 0; i < imageData.length; i++) {
        normalizedData[i] = imageData[i] / 255.0;
      }
    } else {
      normalizedData = imageData;
      if (!isPreprocessed) {
        // Normalize if needed
        for (let i = 0; i < normalizedData.length; i++) {
          normalizedData[i] = normalizedData[i] / 255.0;
        }
      }
    }
    
    // Create tensor with shape [1, 3, height, width]
    const imageTensor = await this.hardware.createTensor({
      dimensions: [1, 3, height, width],
      data: normalizedData,
      dtype: 'float32'
    });
    
    // Normalize with CLIP-specific mean and std
    const meanTensor = await this.hardware.createTensor({
      dimensions: [1, 3, 1, 1],
      data: new Float32Array([0.48145466, 0.4578275, 0.40821073]), // CLIP-specific RGB mean
      dtype: 'float32'
    });
    
    const stdTensor = await this.hardware.createTensor({
      dimensions: [1, 3, 1, 1],
      data: new Float32Array([0.26862954, 0.26130258, 0.27577711]), // CLIP-specific RGB std
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
   * Tokenize text input for CLIP model
   * @param text Text input
   * @returns Token IDs and attention mask
   */
  private async tokenizeText(text: string): Promise<{ inputIds: number[], attentionMask: number[] }> {
    // In a real implementation, this would use CLIP's tokenizer
    // For now, we'll create a simple placeholder
    
    // Create simple token IDs (simulated)
    const tokenIds = Array.from({ length: Math.min(text.length + 2, this.config.maxTextLength) }, 
      (_, i) => i === 0 ? 49406 : (i === text.length + 1 ? 49407 : 1000 + i)); // BOS, text, EOS
    
    // Pad to max length
    while (tokenIds.length < this.config.maxTextLength) {
      tokenIds.push(0); // Padding token
    }
    
    // Create attention mask (1 for real tokens, 0 for padding)
    const attentionMask = tokenIds.map(id => id === 0 ? 0 : 1);
    
    return { inputIds, attentionMask };
  }
  
  /**
   * Encode image with vision transformer
   * @param preprocessedImage Preprocessed image tensor
   * @returns Image embeddings
   */
  private async encodeImage(preprocessedImage: Tensor): Promise<Tensor> {
    // Calculate number of patches
    const patchesPerSide = this.config.imageSize / this.config.patchSize;
    const numPatches = patchesPerSide * patchesPerSide;
    
    // Patchify image
    const patchesShape = [
      1,                          // batch size
      numPatches,                 // number of patches
      3 * this.config.patchSize * this.config.patchSize // patch dimension
    ];
    
    // In a real implementation, we would properly extract patches from the image
    // For now, we'll simulate the result
    const patchesTensor = await this.hardware.createTensor({
      dimensions: patchesShape,
      data: new Float32Array(patchesShape[0] * patchesShape[1] * patchesShape[2]),
      dtype: 'float32'
    });
    
    // Project patches to embedding dimension
    const patchEmbeddings = await matmul(
      patchesTensor,
      this.weights.vision!.embedding!,
      this.hardware,
      { useOptimization: this.config.useOptimizedOps }
    );
    
    // Add class token (similar to BERT's CLS token)
    const classTokenTensor = await this.hardware.createTensor({
      dimensions: [1, 1, this.config.hiddenSize],
      data: new Float32Array(this.config.hiddenSize),
      dtype: 'float32'
    });
    
    // Concatenate class token with patch embeddings
    const withClassToken = await this.hardware.concat(
      [classTokenTensor, patchEmbeddings],
      1 // concat along sequence dimension
    );
    
    // Add position embeddings
    const posEmbeddings = await this.hardware.slice(
      this.weights.vision!.positionalEmbedding!,
      [0, 0],
      [numPatches + 1, this.config.hiddenSize]
    );
    
    const expandedPosEmbeddings = await this.hardware.reshape(
      posEmbeddings,
      [1, numPatches + 1, this.config.hiddenSize]
    );
    
    const embeddingsWithPos = await add(
      withClassToken,
      expandedPosEmbeddings,
      this.hardware
    );
    
    // Process through transformer layers
    let hiddenStates = embeddingsWithPos;
    
    for (let i = 0; i < this.config.visionNumLayers; i++) {
      const layer = this.weights.vision!.layers![i];
      
      // Layer norm before self-attention
      const layerNorm1 = await layerNorm(
        hiddenStates,
        layer.layerNorm1.weight!,
        layer.layerNorm1.bias!,
        this.config.layerNormEps,
        this.hardware
      );
      
      // Self-attention
      const queryProjection = await matmul(
        layerNorm1,
        layer.attention.query.weight!,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      const queryBiased = await add(queryProjection, layer.attention.query.bias!, this.hardware);
      
      const keyProjection = await matmul(
        layerNorm1,
        layer.attention.key.weight!,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      const keyBiased = await add(keyProjection, layer.attention.key.bias!, this.hardware);
      
      const valueProjection = await matmul(
        layerNorm1,
        layer.attention.value.weight!,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      const valueBiased = await add(valueProjection, layer.attention.value.bias!, this.hardware);
      
      // Shape for multi-head attention
      const [batchSize, seqLength, hiddenSize] = layerNorm1.dimensions;
      const headSize = hiddenSize / this.config.visionNumHeads;
      
      // Reshape for attention
      const queryReshaped = await this.hardware.reshape(
        queryBiased,
        [batchSize, seqLength, this.config.visionNumHeads, headSize]
      );
      const keyReshaped = await this.hardware.reshape(
        keyBiased,
        [batchSize, seqLength, this.config.visionNumHeads, headSize]
      );
      const valueReshaped = await this.hardware.reshape(
        valueBiased,
        [batchSize, seqLength, this.config.visionNumHeads, headSize]
      );
      
      // Transpose for attention computation
      const queryTransposed = await this.hardware.transpose(
        queryReshaped,
        [0, 2, 1, 3] // [batch, heads, seq_len, head_size]
      );
      const keyTransposed = await this.hardware.transpose(
        keyReshaped,
        [0, 2, 3, 1] // [batch, heads, head_size, seq_len]
      );
      const valueTransposed = await this.hardware.transpose(
        valueReshaped,
        [0, 2, 1, 3] // [batch, heads, seq_len, head_size]
      );
      
      // Compute attention scores
      const scores = await matmul(
        queryTransposed,
        keyTransposed,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      
      // Scale attention scores
      const scaleFactor = Math.sqrt(headSize);
      const scaleConstant = await this.hardware.createTensor({
        dimensions: [1],
        data: new Float32Array([1 / scaleFactor]),
        dtype: 'float32'
      });
      const scaledScores = await this.hardware.mul(scores, scaleConstant);
      
      // Apply softmax
      const attentionProbs = await softmax(scaledScores, -1, this.hardware);
      
      // Apply attention to values
      const contextLayer = await matmul(
        attentionProbs,
        valueTransposed,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      
      // Transpose and reshape results
      const contextTransposed = await this.hardware.transpose(
        contextLayer,
        [0, 2, 1, 3] // [batch, seq_len, heads, head_size]
      );
      
      const contextReshaped = await this.hardware.reshape(
        contextTransposed,
        [batchSize, seqLength, hiddenSize]
      );
      
      // Output projection
      const attentionOutput = await matmul(
        contextReshaped,
        layer.attention.output.weight!,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      const attentionWithBias = await add(
        attentionOutput,
        layer.attention.output.bias!,
        this.hardware
      );
      
      // Add residual connection to input
      const attentionWithResidual = await add(attentionWithBias, hiddenStates, this.hardware);
      
      // Layer norm before MLP
      const layerNorm2 = await layerNorm(
        attentionWithResidual,
        layer.layerNorm2.weight!,
        layer.layerNorm2.bias!,
        this.config.layerNormEps,
        this.hardware
      );
      
      // MLP forward pass
      const mlpIntermediate = await matmul(
        layerNorm2,
        layer.mlp.fc1.weight!,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      const intermediateWithBias = await add(
        mlpIntermediate,
        layer.mlp.fc1.bias!,
        this.hardware
      );
      
      // GELU activation
      const intermediateWithActivation = await this.hardware.gelu(intermediateWithBias);
      
      // Output projection
      const mlpOutput = await matmul(
        intermediateWithActivation,
        layer.mlp.fc2.weight!,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      const outputWithBias = await add(
        mlpOutput,
        layer.mlp.fc2.bias!,
        this.hardware
      );
      
      // Add residual connection
      const outputWithResidual = await add(
        outputWithBias,
        attentionWithResidual,
        this.hardware
      );
      
      // Update hidden states for next layer
      hiddenStates = outputWithResidual;
      
      // Release temporary tensors
      await this.hardware.releaseTensor(layerNorm1);
      await this.hardware.releaseTensor(queryProjection);
      await this.hardware.releaseTensor(queryBiased);
      await this.hardware.releaseTensor(keyProjection);
      await this.hardware.releaseTensor(keyBiased);
      await this.hardware.releaseTensor(valueProjection);
      await this.hardware.releaseTensor(valueBiased);
      await this.hardware.releaseTensor(queryReshaped);
      await this.hardware.releaseTensor(keyReshaped);
      await this.hardware.releaseTensor(valueReshaped);
      await this.hardware.releaseTensor(queryTransposed);
      await this.hardware.releaseTensor(keyTransposed);
      await this.hardware.releaseTensor(valueTransposed);
      await this.hardware.releaseTensor(scores);
      await this.hardware.releaseTensor(scaleConstant);
      await this.hardware.releaseTensor(scaledScores);
      await this.hardware.releaseTensor(attentionProbs);
      await this.hardware.releaseTensor(contextLayer);
      await this.hardware.releaseTensor(contextTransposed);
      await this.hardware.releaseTensor(contextReshaped);
      await this.hardware.releaseTensor(attentionOutput);
      await this.hardware.releaseTensor(attentionWithBias);
      if (i < this.config.visionNumLayers - 1) {
        await this.hardware.releaseTensor(attentionWithResidual);
      }
      await this.hardware.releaseTensor(layerNorm2);
      await this.hardware.releaseTensor(mlpIntermediate);
      await this.hardware.releaseTensor(intermediateWithBias);
      await this.hardware.releaseTensor(intermediateWithActivation);
      await this.hardware.releaseTensor(mlpOutput);
      await this.hardware.releaseTensor(outputWithBias);
      if (i < this.config.visionNumLayers - 1) {
        await this.hardware.releaseTensor(outputWithResidual);
      }
    }
    
    // Final layer norm
    const finalLayerNorm = await layerNorm(
      hiddenStates,
      this.weights.vision!.finalLayerNorm!.weight!,
      this.weights.vision!.finalLayerNorm!.bias!,
      this.config.layerNormEps,
      this.hardware
    );
    
    // Extract the class token representation (first token)
    const classRepresentation = await this.hardware.slice(
      finalLayerNorm,
      [0, 0, 0],
      [1, 1, this.config.hiddenSize]
    );
    
    // Reshape to [batch_size, hidden_size]
    const reshapedClassRepresentation = await this.hardware.reshape(
      classRepresentation,
      [1, this.config.hiddenSize]
    );
    
    // Project to joint embedding space
    const imageEmbedding = await matmul(
      reshapedClassRepresentation,
      this.weights.vision!.visualProjection!,
      this.hardware,
      { useOptimization: this.config.useOptimizedOps }
    );
    
    // Normalize embedding
    const norm = await this.hardware.norm(imageEmbedding, 2, -1, true);
    const normalizedImageEmbedding = await this.hardware.div(imageEmbedding, norm);
    
    // Release temporary tensors
    await this.hardware.releaseTensor(patchesTensor);
    await this.hardware.releaseTensor(patchEmbeddings);
    await this.hardware.releaseTensor(classTokenTensor);
    await this.hardware.releaseTensor(withClassToken);
    await this.hardware.releaseTensor(posEmbeddings);
    await this.hardware.releaseTensor(expandedPosEmbeddings);
    await this.hardware.releaseTensor(embeddingsWithPos);
    await this.hardware.releaseTensor(hiddenStates);
    await this.hardware.releaseTensor(finalLayerNorm);
    await this.hardware.releaseTensor(classRepresentation);
    await this.hardware.releaseTensor(reshapedClassRepresentation);
    await this.hardware.releaseTensor(imageEmbedding);
    await this.hardware.releaseTensor(norm);
    
    return normalizedImageEmbedding;
  }
  
  /**
   * Encode text with text transformer
   * @param textInput Text input
   * @returns Text embeddings
   */
  private async encodeText(textInput: ClipTextInput): Promise<Tensor> {
    // Get the text embedding size
    const textEmbeddingSize = this.config.textEmbeddingSize || this.config.hiddenSize;
    
    // Tokenize text if needed
    let inputIds: number[];
    let attentionMask: number[];
    
    if (textInput.inputIds && textInput.attentionMask) {
      inputIds = textInput.inputIds;
      attentionMask = textInput.attentionMask;
    } else {
      const tokenized = await this.tokenizeText(textInput.text);
      inputIds = tokenized.inputIds;
      attentionMask = tokenized.attentionMask;
    }
    
    // Create input tensors
    const inputIdsTensor = await this.hardware.createTensor({
      dimensions: [1, inputIds.length],
      data: new Int32Array(inputIds),
      dtype: 'int32'
    });
    
    const attentionMaskTensor = await this.hardware.createTensor({
      dimensions: [1, attentionMask.length],
      data: new Int32Array(attentionMask),
      dtype: 'int32'
    });
    
    // Convert attention mask to float
    const attentionMaskFloat = await this.hardware.cast(attentionMaskTensor, 'float32');
    
    // Get token embeddings
    const tokenEmbeddings = await this.hardware.gatherEmbedding(
      this.weights.text!.embedding!,
      inputIdsTensor
    );
    
    // Get position embeddings
    const positionIds = await this.hardware.createTensor({
      dimensions: [1, this.config.maxTextLength],
      data: new Int32Array(Array.from({ length: this.config.maxTextLength }, (_, i) => i)),
      dtype: 'int32'
    });
    
    const positionEmbeddings = await this.hardware.gatherEmbedding(
      this.weights.text!.positionalEmbedding!,
      positionIds
    );
    
    // Add embeddings
    const embeddingsWithPos = await add(tokenEmbeddings, positionEmbeddings, this.hardware);
    
    // Process through transformer layers
    let hiddenStates = embeddingsWithPos;
    
    for (let i = 0; i < this.config.textNumLayers; i++) {
      const layer = this.weights.text!.layers![i];
      
      // Layer norm before self-attention
      const layerNorm1 = await layerNorm(
        hiddenStates,
        layer.layerNorm1.weight!,
        layer.layerNorm1.bias!,
        this.config.layerNormEps,
        this.hardware
      );
      
      // Self-attention with causal mask (for text)
      const queryProjection = await matmul(
        layerNorm1,
        layer.attention.query.weight!,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      const queryBiased = await add(queryProjection, layer.attention.query.bias!, this.hardware);
      
      const keyProjection = await matmul(
        layerNorm1,
        layer.attention.key.weight!,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      const keyBiased = await add(keyProjection, layer.attention.key.bias!, this.hardware);
      
      const valueProjection = await matmul(
        layerNorm1,
        layer.attention.value.weight!,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      const valueBiased = await add(valueProjection, layer.attention.value.bias!, this.hardware);
      
      // Shape for multi-head attention
      const [batchSize, seqLength, hiddenSize] = layerNorm1.dimensions;
      const headSize = hiddenSize / this.config.textNumHeads;
      
      // Reshape for attention
      const queryReshaped = await this.hardware.reshape(
        queryBiased,
        [batchSize, seqLength, this.config.textNumHeads, headSize]
      );
      const keyReshaped = await this.hardware.reshape(
        keyBiased,
        [batchSize, seqLength, this.config.textNumHeads, headSize]
      );
      const valueReshaped = await this.hardware.reshape(
        valueBiased,
        [batchSize, seqLength, this.config.textNumHeads, headSize]
      );
      
      // Transpose for attention computation
      const queryTransposed = await this.hardware.transpose(
        queryReshaped,
        [0, 2, 1, 3] // [batch, heads, seq_len, head_size]
      );
      const keyTransposed = await this.hardware.transpose(
        keyReshaped,
        [0, 2, 3, 1] // [batch, heads, head_size, seq_len]
      );
      const valueTransposed = await this.hardware.transpose(
        valueReshaped,
        [0, 2, 1, 3] // [batch, heads, seq_len, head_size]
      );
      
      // Compute attention scores
      const scores = await matmul(
        queryTransposed,
        keyTransposed,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      
      // Scale attention scores
      const scaleFactor = Math.sqrt(headSize);
      const scaleConstant = await this.hardware.createTensor({
        dimensions: [1],
        data: new Float32Array([1 / scaleFactor]),
        dtype: 'float32'
      });
      const scaledScores = await this.hardware.mul(scores, scaleConstant);
      
      // Create causal mask (lower triangular)
      const causalMask = await this.hardware.createTensor({
        dimensions: [1, 1, seqLength, seqLength],
        data: new Float32Array(seqLength * seqLength).fill(Number.NEGATIVE_INFINITY),
        dtype: 'float32'
      });
      
      // Fill in lower triangular part with zeros
      // In a real implementation, this would be more efficient
      // Here we simulate the result
      
      // Add causal mask to attention scores
      const maskedScores = await add(scaledScores, causalMask, this.hardware);
      
      // Create attention mask
      const attentionMaskExpanded = await this.hardware.reshape(
        attentionMaskFloat,
        [batchSize, 1, 1, seqLength]
      );
      const attentionMaskAdditive = await this.hardware.sub(
        await this.hardware.createTensor({
          dimensions: attentionMaskExpanded.dimensions,
          data: new Float32Array(attentionMaskExpanded.dimensions.reduce((a, b) => a * b, 1)).fill(1.0),
          dtype: 'float32'
        }),
        attentionMaskExpanded
      );
      const attentionMaskScaled = await this.hardware.mul(
        attentionMaskAdditive,
        await this.hardware.createTensor({
          dimensions: [1],
          data: new Float32Array([-10000.0]),
          dtype: 'float32'
        })
      );
      
      // Apply attention mask
      const maskedWithAttention = await add(maskedScores, attentionMaskScaled, this.hardware);
      
      // Apply softmax
      const attentionProbs = await softmax(maskedWithAttention, -1, this.hardware);
      
      // Apply attention to values
      const contextLayer = await matmul(
        attentionProbs,
        valueTransposed,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      
      // Transpose and reshape results
      const contextTransposed = await this.hardware.transpose(
        contextLayer,
        [0, 2, 1, 3] // [batch, seq_len, heads, head_size]
      );
      
      const contextReshaped = await this.hardware.reshape(
        contextTransposed,
        [batchSize, seqLength, hiddenSize]
      );
      
      // Output projection
      const attentionOutput = await matmul(
        contextReshaped,
        layer.attention.output.weight!,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      const attentionWithBias = await add(
        attentionOutput,
        layer.attention.output.bias!,
        this.hardware
      );
      
      // Add residual connection to input
      const attentionWithResidual = await add(attentionWithBias, hiddenStates, this.hardware);
      
      // Layer norm before MLP
      const layerNorm2 = await layerNorm(
        attentionWithResidual,
        layer.layerNorm2.weight!,
        layer.layerNorm2.bias!,
        this.config.layerNormEps,
        this.hardware
      );
      
      // MLP forward pass
      const mlpIntermediate = await matmul(
        layerNorm2,
        layer.mlp.fc1.weight!,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      const intermediateWithBias = await add(
        mlpIntermediate,
        layer.mlp.fc1.bias!,
        this.hardware
      );
      
      // GELU activation
      const intermediateWithActivation = await this.hardware.gelu(intermediateWithBias);
      
      // Output projection
      const mlpOutput = await matmul(
        intermediateWithActivation,
        layer.mlp.fc2.weight!,
        this.hardware,
        { useOptimization: this.config.useOptimizedOps }
      );
      const outputWithBias = await add(
        mlpOutput,
        layer.mlp.fc2.bias!,
        this.hardware
      );
      
      // Add residual connection
      const outputWithResidual = await add(
        outputWithBias,
        attentionWithResidual,
        this.hardware
      );
      
      // Update hidden states for next layer
      hiddenStates = outputWithResidual;
      
      // Release temporary tensors
      await this.hardware.releaseTensor(layerNorm1);
      await this.hardware.releaseTensor(queryProjection);
      await this.hardware.releaseTensor(queryBiased);
      await this.hardware.releaseTensor(keyProjection);
      await this.hardware.releaseTensor(keyBiased);
      await this.hardware.releaseTensor(valueProjection);
      await this.hardware.releaseTensor(valueBiased);
      await this.hardware.releaseTensor(queryReshaped);
      await this.hardware.releaseTensor(keyReshaped);
      await this.hardware.releaseTensor(valueReshaped);
      await this.hardware.releaseTensor(queryTransposed);
      await this.hardware.releaseTensor(keyTransposed);
      await this.hardware.releaseTensor(valueTransposed);
      await this.hardware.releaseTensor(scores);
      await this.hardware.releaseTensor(scaleConstant);
      await this.hardware.releaseTensor(scaledScores);
      await this.hardware.releaseTensor(causalMask);
      await this.hardware.releaseTensor(maskedScores);
      await this.hardware.releaseTensor(attentionMaskExpanded);
      await this.hardware.releaseTensor(attentionMaskAdditive);
      await this.hardware.releaseTensor(attentionMaskScaled);
      await this.hardware.releaseTensor(maskedWithAttention);
      await this.hardware.releaseTensor(attentionProbs);
      await this.hardware.releaseTensor(contextLayer);
      await this.hardware.releaseTensor(contextTransposed);
      await this.hardware.releaseTensor(contextReshaped);
      await this.hardware.releaseTensor(attentionOutput);
      await this.hardware.releaseTensor(attentionWithBias);
      if (i < this.config.textNumLayers - 1) {
        await this.hardware.releaseTensor(attentionWithResidual);
      }
      await this.hardware.releaseTensor(layerNorm2);
      await this.hardware.releaseTensor(mlpIntermediate);
      await this.hardware.releaseTensor(intermediateWithBias);
      await this.hardware.releaseTensor(intermediateWithActivation);
      await this.hardware.releaseTensor(mlpOutput);
      await this.hardware.releaseTensor(outputWithBias);
      if (i < this.config.textNumLayers - 1) {
        await this.hardware.releaseTensor(outputWithResidual);
      }
    }
    
    // Final layer norm
    const finalLayerNorm = await layerNorm(
      hiddenStates,
      this.weights.text!.finalLayerNorm!.weight!,
      this.weights.text!.finalLayerNorm!.bias!,
      this.config.layerNormEps,
      this.hardware
    );
    
    // Extract sequence output at EOS token
    // In a real implementation, find the EOS token position from input IDs
    // Here we'll just use the last token
    
    // Get last token hidden states
    const lastTokenIdx = inputIds.findIndex(id => id === 49407); // EOS token ID
    const lastTokenPos = lastTokenIdx > 0 ? lastTokenIdx : inputIds.length - 1;
    
    const lastTokenRepresentation = await this.hardware.slice(
      finalLayerNorm,
      [0, lastTokenPos, 0],
      [1, 1, textEmbeddingSize]
    );
    
    // Reshape to [batch_size, hidden_size]
    const reshapedTextRepresentation = await this.hardware.reshape(
      lastTokenRepresentation,
      [1, textEmbeddingSize]
    );
    
    // Project to joint embedding space
    const textEmbedding = await matmul(
      reshapedTextRepresentation,
      this.weights.text!.textProjection!,
      this.hardware,
      { useOptimization: this.config.useOptimizedOps }
    );
    
    // Normalize embedding
    const norm = await this.hardware.norm(textEmbedding, 2, -1, true);
    const normalizedTextEmbedding = await this.hardware.div(textEmbedding, norm);
    
    // Release temporary tensors
    await this.hardware.releaseTensor(inputIdsTensor);
    await this.hardware.releaseTensor(attentionMaskTensor);
    await this.hardware.releaseTensor(attentionMaskFloat);
    await this.hardware.releaseTensor(tokenEmbeddings);
    await this.hardware.releaseTensor(positionIds);
    await this.hardware.releaseTensor(positionEmbeddings);
    await this.hardware.releaseTensor(embeddingsWithPos);
    await this.hardware.releaseTensor(hiddenStates);
    await this.hardware.releaseTensor(finalLayerNorm);
    await this.hardware.releaseTensor(lastTokenRepresentation);
    await this.hardware.releaseTensor(reshapedTextRepresentation);
    await this.hardware.releaseTensor(textEmbedding);
    await this.hardware.releaseTensor(norm);
    
    return normalizedTextEmbedding;
  }
  
  /**
   * Compute similarity between text and image embeddings
   * @param imageEmbedding Image embedding
   * @param textEmbedding Text embedding
   * @returns Similarity score (higher means more similar)
   */
  private async computeSimilarity(
    imageEmbedding: Tensor,
    textEmbedding: Tensor
  ): Promise<number> {
    // Get logit scale
    const logitScale = await this.hardware.exp(this.weights.logitScale!);
    
    // Compute dot product similarity
    const similarity = await matmul(
      imageEmbedding,
      await this.hardware.transpose(textEmbedding, [0, 2, 1]),
      this.hardware,
      { useOptimization: this.config.useOptimizedOps }
    );
    
    // Scale similarity
    const scaledSimilarity = await this.hardware.mul(similarity, logitScale);
    
    // Extract similarity score
    const similarityArray = await scaledSimilarity.toArray() as number[][];
    
    // Release temporary tensors
    await this.hardware.releaseTensor(similarity);
    await this.hardware.releaseTensor(scaledSimilarity);
    
    return similarityArray[0][0];
  }
  
  /**
   * Process image and/or text inputs through CLIP model
   * @param input Image and/or text inputs
   * @returns CLIP output with embeddings and optional similarity
   */
  public async process(input: {
    image?: ClipImageInput,
    text?: ClipTextInput
  }): Promise<ClipOutput> {
    if (!this.initialized) {
      throw new Error('Model not initialized. Call initialize() first.');
    }
    
    let imageEmbedding: Tensor | null = null;
    let textEmbedding: Tensor | null = null;
    let similarity: number | undefined;
    
    try {
      // Process image if provided
      if (input.image) {
        const preprocessedImage = await this.preprocessImage(input.image);
        imageEmbedding = await this.encodeImage(preprocessedImage);
        
        // Store image embedding as shared tensor
        await this.createSharedTensor(imageEmbedding, 'vision_embedding');
        
        // Release preprocessed image tensor
        await this.hardware.releaseTensor(preprocessedImage);
      }
      
      // Process text if provided
      if (input.text) {
        textEmbedding = await this.encodeText(input.text);
        
        // Store text embedding as shared tensor
        await this.createSharedTensor(textEmbedding, 'text_embedding');
      }
      
      // Compute similarity if both image and text provided
      if (imageEmbedding && textEmbedding) {
        similarity = await this.computeSimilarity(imageEmbedding, textEmbedding);
      }
      
      // Prepare output
      const output: ClipOutput = {
        model: this.config.modelId,
        backend: this.hardware.type
      };
      
      // Add embeddings to output if available
      if (imageEmbedding) {
        output.imageEmbeddings = await imageEmbedding.toArray() as number[];
      }
      
      if (textEmbedding) {
        output.textEmbeddings = await textEmbedding.toArray() as number[];
      }
      
      // Add similarity if computed
      if (similarity !== undefined) {
        output.similarity = similarity;
      }
      
      return output;
    } catch (error) {
      console.error('Error during CLIP processing:', error);
      throw error;
    } finally {
      // Release embedding tensors
      if (imageEmbedding) {
        await this.hardware.releaseTensor(imageEmbedding);
      }
      
      if (textEmbedding) {
        await this.hardware.releaseTensor(textEmbedding);
      }
    }
  }
  
  /**
   * Create a shared tensor for cross-model integration
   * @param tensor Tensor to share
   * @param outputType Type of output ('vision_embedding' or 'text_embedding')
   * @returns Shared tensor
   */
  public async createSharedTensor(
    tensor: Tensor,
    outputType: string = 'vision_embedding'
  ): Promise<SharedTensor> {
    // Create shared tensor
    const sharedTensor = new SharedTensor(
      tensor,
      outputType,
      this.config.modelId,
      this.hardware.type
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
   * Dispose of all resources associated with the model
   */
  public async dispose(): Promise<void> {
    if (!this.initialized) {
      return;
    }
    
    try {
      // Release vision model weights
      if (this.weights.vision) {
        if (this.weights.vision.embedding) {
          await this.hardware.releaseTensor(this.weights.vision.embedding);
        }
        
        if (this.weights.vision.positionalEmbedding) {
          await this.hardware.releaseTensor(this.weights.vision.positionalEmbedding);
        }
        
        if (this.weights.vision.finalLayerNorm) {
          if (this.weights.vision.finalLayerNorm.weight) {
            await this.hardware.releaseTensor(this.weights.vision.finalLayerNorm.weight);
          }
          
          if (this.weights.vision.finalLayerNorm.bias) {
            await this.hardware.releaseTensor(this.weights.vision.finalLayerNorm.bias);
          }
        }
        
        if (this.weights.vision.visualProjection) {
          await this.hardware.releaseTensor(this.weights.vision.visualProjection);
        }
        
        // Release vision layers
        if (this.weights.vision.layers) {
          for (const layer of this.weights.vision.layers) {
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
            
            // Layer norms
            if (layer.layerNorm1.weight) {
              await this.hardware.releaseTensor(layer.layerNorm1.weight);
            }
            if (layer.layerNorm1.bias) {
              await this.hardware.releaseTensor(layer.layerNorm1.bias);
            }
            
            if (layer.layerNorm2.weight) {
              await this.hardware.releaseTensor(layer.layerNorm2.weight);
            }
            if (layer.layerNorm2.bias) {
              await this.hardware.releaseTensor(layer.layerNorm2.bias);
            }
            
            // MLP weights
            if (layer.mlp.fc1.weight) {
              await this.hardware.releaseTensor(layer.mlp.fc1.weight);
            }
            if (layer.mlp.fc1.bias) {
              await this.hardware.releaseTensor(layer.mlp.fc1.bias);
            }
            
            if (layer.mlp.fc2.weight) {
              await this.hardware.releaseTensor(layer.mlp.fc2.weight);
            }
            if (layer.mlp.fc2.bias) {
              await this.hardware.releaseTensor(layer.mlp.fc2.bias);
            }
          }
        }
      }
      
      // Release text model weights
      if (this.weights.text) {
        if (this.weights.text.embedding) {
          await this.hardware.releaseTensor(this.weights.text.embedding);
        }
        
        if (this.weights.text.positionalEmbedding) {
          await this.hardware.releaseTensor(this.weights.text.positionalEmbedding);
        }
        
        if (this.weights.text.finalLayerNorm) {
          if (this.weights.text.finalLayerNorm.weight) {
            await this.hardware.releaseTensor(this.weights.text.finalLayerNorm.weight);
          }
          
          if (this.weights.text.finalLayerNorm.bias) {
            await this.hardware.releaseTensor(this.weights.text.finalLayerNorm.bias);
          }
        }
        
        if (this.weights.text.textProjection) {
          await this.hardware.releaseTensor(this.weights.text.textProjection);
        }
        
        // Release text layers
        if (this.weights.text.layers) {
          for (const layer of this.weights.text.layers) {
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
            
            // Layer norms
            if (layer.layerNorm1.weight) {
              await this.hardware.releaseTensor(layer.layerNorm1.weight);
            }
            if (layer.layerNorm1.bias) {
              await this.hardware.releaseTensor(layer.layerNorm1.bias);
            }
            
            if (layer.layerNorm2.weight) {
              await this.hardware.releaseTensor(layer.layerNorm2.weight);
            }
            if (layer.layerNorm2.bias) {
              await this.hardware.releaseTensor(layer.layerNorm2.bias);
            }
            
            // MLP weights
            if (layer.mlp.fc1.weight) {
              await this.hardware.releaseTensor(layer.mlp.fc1.weight);
            }
            if (layer.mlp.fc1.bias) {
              await this.hardware.releaseTensor(layer.mlp.fc1.bias);
            }
            
            if (layer.mlp.fc2.weight) {
              await this.hardware.releaseTensor(layer.mlp.fc2.weight);
            }
            if (layer.mlp.fc2.bias) {
              await this.hardware.releaseTensor(layer.mlp.fc2.bias);
            }
          }
        }
      }
      
      // Release logit scale
      if (this.weights.logitScale) {
        await this.hardware.releaseTensor(this.weights.logitScale);
      }
      
      // Release shared tensors
      for (const [key, sharedTensor] of this.sharedTensors.entries()) {
        await sharedTensor.release();
        this.sharedTensors.delete(key);
      }
      
      this.initialized = false;
    } catch (error) {
      console.error('Error disposing CLIP model:', error);
      throw error;
    }
  }
}

/**
 * Factory function to create a CLIP model
 * @param hardware Hardware backend for tensor operations
 * @param config CLIP configuration
 * @returns CLIP model instance
 */
export function createClipModel(
  hardware: HardwareBackend,
  config: Partial<ClipConfig> = {}
): Clip {
  return new Clip(hardware, config);
}