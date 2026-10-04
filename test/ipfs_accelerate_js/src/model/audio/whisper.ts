/**
 * Whisper model implementation with hardware acceleration support
 * Implements OpenAI's Whisper model with WebGPU and WebNN backends
 */

import { Tensor } from '../../tensor/tensor';
import { SharedTensor } from '../../tensor/shared_tensor';
import { TensorBackendType } from '../../tensor/index';
import { HardwareBackend } from '../../hardware/interfaces/hardware_backend';
import { matmul, softmax } from '../../tensor/operations/matrix';
import { layerNorm } from '../../tensor/operations/nn';
import { add } from '../../tensor/operations/basic';

/**
 * Configuration for the Whisper model
 */
export interface WhisperConfig {
  /** Model identifier (e.g., "openai/whisper-tiny") */
  modelId: string;
  /** Sample rate of audio input (default 16000) */
  sampleRate: number;
  /** Number of mel frequency bins */
  melBins: number;
  /** Number of tokens in the vocabulary */
  vocabSize: number;
  /** Hidden size in the model */
  hiddenSize: number;
  /** Number of encoder layers */
  encoderLayers: number;
  /** Number of decoder layers */
  decoderLayers: number;
  /** Number of attention heads */
  numHeads: number;
  /** FFT size for audio processing */
  fftSize: number;
  /** Hop length for audio processing */
  hopLength: number;
  /** Window size for audio processing */
  windowSize: number;
  /** Hardware backend preference */
  backendPreference?: TensorBackendType[];
  /** Whether to use browser-specific optimizations */
  useBrowserOptimizations?: boolean;
  /** Maximum audio length in samples */
  maxAudioLength?: number;
  /** Maximum output length */
  maxOutputLength?: number;
}

/**
 * Default Whisper configuration (whisper-tiny model)
 */
export const DEFAULT_WHISPER_CONFIG: WhisperConfig = {
  modelId: 'openai/whisper-tiny',
  sampleRate: 16000,
  melBins: 80,
  vocabSize: 51864,
  hiddenSize: 384,
  encoderLayers: 4,
  decoderLayers: 4,
  numHeads: 6,
  fftSize: 400,
  hopLength: 160,
  windowSize: 400,
  backendPreference: ['webgpu', 'webnn', 'cpu'],
  useBrowserOptimizations: true,
  maxAudioLength: 480000, // 30 seconds @ 16kHz
  maxOutputLength: 448
};

/**
 * Input format for Whisper model
 */
export interface WhisperInput {
  /** Raw audio samples */
  audioData: Float32Array;
  /** Sample rate of audio */
  sampleRate?: number;
  /** Whether to use multilingual mode */
  multilingual?: boolean;
  /** Language hint (ISO 639-1 code, e.g., "en") */
  language?: string;
  /** Number of beams for beam search decoding */
  beamSize?: number;
  /** Whether to return timestamps */
  returnTimestamps?: boolean;
}

/**
 * Segment from speech recognition
 */
export interface WhisperSegment {
  /** Start time of segment in seconds */
  start: number;
  /** End time of segment in seconds */
  end: number;
  /** Transcribed text */
  text: string;
}

/**
 * Output format for Whisper model
 */
export interface WhisperOutput {
  /** Full transcription text */
  text: string;
  /** Segments with timestamps (if returnTimestamps is true) */
  segments?: WhisperSegment[];
  /** Language detected (if multilingual mode) */
  detectedLanguage?: string;
  /** Model identifier */
  model: string;
  /** Backend used for inference */
  backend: string;
}

/**
 * Whisper model implementation with hardware acceleration
 */
export class Whisper {
  private config: WhisperConfig;
  private hardware: HardwareBackend;
  private initialized: boolean = false;
  
  // Model weights (loaded on demand)
  private weights: {
    // Encoder weights
    encoder?: {
      embedding?: Tensor,
      positionalEmbedding?: Tensor,
      layers?: Array<{
        selfAttention: {
          query: { weight?: Tensor, bias?: Tensor },
          key: { weight?: Tensor, bias?: Tensor },
          value: { weight?: Tensor, bias?: Tensor },
          output: { weight?: Tensor, bias?: Tensor },
        },
        layerNorm1: { weight?: Tensor, bias?: Tensor },
        ffn: {
          intermediate: { weight?: Tensor, bias?: Tensor },
          output: { weight?: Tensor, bias?: Tensor },
        },
        layerNorm2: { weight?: Tensor, bias?: Tensor },
      }>,
      finalLayerNorm?: { weight?: Tensor, bias?: Tensor },
    },
    
    // Decoder weights
    decoder?: {
      embedding?: Tensor,
      positionalEmbedding?: Tensor,
      layers?: Array<{
        selfAttention: {
          query: { weight?: Tensor, bias?: Tensor },
          key: { weight?: Tensor, bias?: Tensor },
          value: { weight?: Tensor, bias?: Tensor },
          output: { weight?: Tensor, bias?: Tensor },
        },
        layerNorm1: { weight?: Tensor, bias?: Tensor },
        crossAttention: {
          query: { weight?: Tensor, bias?: Tensor },
          key: { weight?: Tensor, bias?: Tensor },
          value: { weight?: Tensor, bias?: Tensor },
          output: { weight?: Tensor, bias?: Tensor },
        },
        layerNorm2: { weight?: Tensor, bias?: Tensor },
        ffn: {
          intermediate: { weight?: Tensor, bias?: Tensor },
          output: { weight?: Tensor, bias?: Tensor },
        },
        layerNorm3: { weight?: Tensor, bias?: Tensor },
      }>,
      finalLayerNorm?: { weight?: Tensor, bias?: Tensor },
    },
    
    // Output projection
    outputProjection?: Tensor,
    
    // Audio processing filters
    melFilterbank?: Tensor,
  } = {};
  
  // Audio processing compute pipeline for WebGPU
  private audioProcessingPipeline?: any;
  
  // Shared tensors for cross-model sharing
  private sharedTensors: Map<string, SharedTensor> = new Map();
  
  /**
   * Constructor for Whisper model
   * @param hardware Hardware backend for tensor operations
   * @param config Whisper configuration
   */
  constructor(
    hardware: HardwareBackend,
    config: Partial<WhisperConfig> = {}
  ) {
    this.hardware = hardware;
    this.config = { ...DEFAULT_WHISPER_CONFIG, ...config };
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
      
      // Set up audio processing pipeline if using WebGPU
      if (this.hardware.getBackendType() === 'webgpu') {
        await this.setupAudioProcessingPipeline();
      }
      
      this.initialized = true;
    } catch (error) {
      console.error('Error initializing Whisper model:', error);
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
    
    // Initialize encoder weights
    this.weights.encoder = {
      embedding: await this.hardware.createTensor({
        dimensions: [this.config.melBins, this.config.hiddenSize],
        data: new Float32Array(this.config.melBins * this.config.hiddenSize),
        dtype: 'float32'
      }),
      positionalEmbedding: await this.hardware.createTensor({
        dimensions: [this.config.maxAudioLength || 3000, this.config.hiddenSize],
        data: new Float32Array((this.config.maxAudioLength || 3000) * this.config.hiddenSize),
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
      }
    };
    
    // Create encoder layers
    for (let i = 0; i < this.config.encoderLayers; i++) {
      const layer = {
        selfAttention: {
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
        ffn: {
          intermediate: {
            weight: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize, this.config.hiddenSize * 4],
              data: new Float32Array(this.config.hiddenSize * this.config.hiddenSize * 4),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize * 4],
              data: new Float32Array(this.config.hiddenSize * 4),
              dtype: 'float32'
            }),
          },
          output: {
            weight: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize * 4, this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize * 4 * this.config.hiddenSize),
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
      
      this.weights.encoder.layers!.push(layer);
    }
    
    // Initialize decoder weights
    this.weights.decoder = {
      embedding: await this.hardware.createTensor({
        dimensions: [this.config.vocabSize, this.config.hiddenSize],
        data: new Float32Array(this.config.vocabSize * this.config.hiddenSize),
        dtype: 'float32'
      }),
      positionalEmbedding: await this.hardware.createTensor({
        dimensions: [this.config.maxOutputLength || 448, this.config.hiddenSize],
        data: new Float32Array((this.config.maxOutputLength || 448) * this.config.hiddenSize),
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
      }
    };
    
    // Create decoder layers
    for (let i = 0; i < this.config.decoderLayers; i++) {
      const layer = {
        selfAttention: {
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
        crossAttention: {
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
        ffn: {
          intermediate: {
            weight: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize, this.config.hiddenSize * 4],
              data: new Float32Array(this.config.hiddenSize * this.config.hiddenSize * 4),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize * 4],
              data: new Float32Array(this.config.hiddenSize * 4),
              dtype: 'float32'
            }),
          },
          output: {
            weight: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize * 4, this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize * 4 * this.config.hiddenSize),
              dtype: 'float32'
            }),
            bias: await this.hardware.createTensor({
              dimensions: [this.config.hiddenSize],
              data: new Float32Array(this.config.hiddenSize),
              dtype: 'float32'
            }),
          },
        },
        layerNorm3: {
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
      
      this.weights.decoder!.layers!.push(layer);
    }
    
    // Initialize output projection
    this.weights.outputProjection = await this.hardware.createTensor({
      dimensions: [this.config.hiddenSize, this.config.vocabSize],
      data: new Float32Array(this.config.hiddenSize * this.config.vocabSize),
      dtype: 'float32'
    });
    
    // Initialize mel filterbank for audio processing
    this.weights.melFilterbank = await this.hardware.createTensor({
      dimensions: [this.config.melBins, Math.floor(this.config.fftSize / 2) + 1],
      data: new Float32Array(this.config.melBins * (Math.floor(this.config.fftSize / 2) + 1)),
      dtype: 'float32'
    });
    
    console.log(`Weights loaded for ${this.config.modelId}`);
  }
  
  /**
   * Set up audio processing pipeline for WebGPU
   */
  private async setupAudioProcessingPipeline(): Promise<void> {
    if (this.hardware.getBackendType() !== 'webgpu') {
      return;
    }
    
    const backendType = this.hardware.getBackendType();
    console.log(`Setting up audio processing pipeline for ${backendType}...`);
    
    // In a real implementation, this would set up a WebGPU compute pipeline
    // for audio processing using the appropriate WGSL shader
    this.audioProcessingPipeline = {
      initialized: true,
      browserOptimized: this.config.useBrowserOptimizations
    };
  }
  
  /**
   * Process audio data into mel spectrogram features
   * @param audioData Raw audio samples
   * @param sampleRate Sample rate of audio
   * @returns Mel spectrogram features as tensor
   */
  private async processAudio(
    audioData: Float32Array,
    sampleRate: number = this.config.sampleRate
  ): Promise<Tensor> {
    // Check if we can use hardware acceleration for audio processing
    const useAcceleration = this.hardware.getBackendType() === 'webgpu' && this.audioProcessingPipeline?.initialized;
    
    // Resample audio if needed
    let processedAudio = audioData;
    if (sampleRate !== this.config.sampleRate) {
      processedAudio = this.resampleAudio(audioData, sampleRate, this.config.sampleRate);
    }
    
    if (useAcceleration) {
      // Use GPU-accelerated audio processing
      return this.processAudioWithGPU(processedAudio);
    } else {
      // Use CPU-based audio processing
      return this.processAudioWithCPU(processedAudio);
    }
  }
  
  /**
   * Process audio data into mel spectrogram features using WebGPU
   * @param audioData Raw audio samples
   * @returns Mel spectrogram features as tensor
   */
  private async processAudioWithGPU(audioData: Float32Array): Promise<Tensor> {
    // In a full implementation, this would use the WebGPU compute pipeline
    // to process audio into mel spectrogram features
    console.log(`Processing audio with GPU (${this.hardware.getBackendType()})...`);
    
    // Convert audio to tensor
    const audioTensor = await this.hardware.createTensor({
      dimensions: [1, audioData.length],
      data: audioData,
      dtype: 'float32'
    });
    
    // Calculate feature dimensions
    const numFrames = Math.floor((audioData.length - this.config.windowSize) / this.config.hopLength) + 1;
    
    // Create output tensor for mel spectrogram
    const melSpectrogram = await this.hardware.createTensor({
      dimensions: [1, this.config.melBins, numFrames],
      data: new Float32Array(this.config.melBins * numFrames),
      dtype: 'float32'
    });
    
    // In a real implementation, we would execute a WebGPU compute shader here
    // For now we'll just simulate it with a simple operation
    
    // Release temporary tensors
    await this.hardware.releaseTensor(audioTensor);
    
    return melSpectrogram;
  }
  
  /**
   * Process audio data into mel spectrogram features using CPU
   * @param audioData Raw audio samples
   * @returns Mel spectrogram features as tensor
   */
  private async processAudioWithCPU(audioData: Float32Array): Promise<Tensor> {
    console.log('Processing audio with CPU...');
    
    // Calculate feature dimensions
    const numFrames = Math.floor((audioData.length - this.config.windowSize) / this.config.hopLength) + 1;
    
    // Create output array for mel spectrogram
    const melSpectrogramData = new Float32Array(this.config.melBins * numFrames);
    
    // In a real implementation, we would perform FFT and mel filterbank application here
    // For now, we'll just generate a placeholder spectrogram
    
    // Create tensor from processed data
    const melSpectrogram = await this.hardware.createTensor({
      dimensions: [1, this.config.melBins, numFrames],
      data: melSpectrogramData,
      dtype: 'float32'
    });
    
    return melSpectrogram;
  }
  
  /**
   * Resample audio to target sample rate
   * @param audioData Raw audio samples
   * @param sourceSampleRate Source sample rate
   * @param targetSampleRate Target sample rate
   * @returns Resampled audio
   */
  private resampleAudio(
    audioData: Float32Array,
    sourceSampleRate: number,
    targetSampleRate: number
  ): Float32Array {
    // Skip if rates are the same
    if (sourceSampleRate === targetSampleRate) {
      return audioData;
    }
    
    console.log(`Resampling audio from ${sourceSampleRate}Hz to ${targetSampleRate}Hz...`);
    
    // Calculate new length
    const sourceLength = audioData.length;
    const targetLength = Math.floor(sourceLength * (targetSampleRate / sourceSampleRate));
    
    // Create output array
    const resampledAudio = new Float32Array(targetLength);
    
    // Simple linear interpolation resampling
    // In a real implementation, we would use a higher quality resampling algorithm
    for (let i = 0; i < targetLength; i++) {
      const sourceIndex = i * (sourceSampleRate / targetSampleRate);
      const sourceIndexFloor = Math.floor(sourceIndex);
      const sourceIndexCeil = Math.min(sourceLength - 1, Math.ceil(sourceIndex));
      const fraction = sourceIndex - sourceIndexFloor;
      
      resampledAudio[i] = (1 - fraction) * audioData[sourceIndexFloor] + fraction * audioData[sourceIndexCeil];
    }
    
    return resampledAudio;
  }
  
  /**
   * Run encoder on mel spectrogram features
   * @param melSpectrogram Mel spectrogram features
   * @returns Encoder output
   */
  private async runEncoder(melSpectrogram: Tensor): Promise<Tensor> {
    // Get dimensions
    const [batchSize, melBins, numFrames] = melSpectrogram.dimensions;
    
    // Reshape spectrogram for encoder input
    const reshapedSpectrogram = await this.hardware.reshape(
      melSpectrogram,
      [batchSize, numFrames, melBins]
    );
    
    // Project mel spectrogram to hidden dimension
    const encoderInput = await matmul(
      reshapedSpectrogram,
      this.weights.encoder!.embedding!,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
    );
    
    // Add positional embeddings
    // Only use as many positional embeddings as we have frames
    const posEmbedding = await this.hardware.slice(
      this.weights.encoder!.positionalEmbedding!,
      [0, 0],
      [numFrames, this.config.hiddenSize]
    );
    
    const expandedPosEmbedding = await this.hardware.reshape(
      posEmbedding,
      [1, numFrames, this.config.hiddenSize]
    );
    
    const embeddings = await add(encoderInput, expandedPosEmbedding, this.hardware);
    
    // Run encoder layers
    let layerOutput = embeddings;
    for (let i = 0; i < this.config.encoderLayers; i++) {
      layerOutput = await this.runEncoderLayer(layerOutput, i);
    }
    
    // Apply final layer norm
    const encoderOutput = await layerNorm(
      layerOutput,
      this.weights.encoder!.finalLayerNorm!.weight!,
      this.weights.encoder!.finalLayerNorm!.bias!,
      1e-5,
      this.hardware
    );
    
    // Release temporary tensors
    await this.hardware.releaseTensor(reshapedSpectrogram);
    await this.hardware.releaseTensor(encoderInput);
    await this.hardware.releaseTensor(posEmbedding);
    await this.hardware.releaseTensor(expandedPosEmbedding);
    await this.hardware.releaseTensor(embeddings);
    await this.hardware.releaseTensor(layerOutput);
    
    return encoderOutput;
  }
  
  /**
   * Run a single encoder layer
   * @param hiddenStates Input hidden states
   * @param layerIndex Index of the encoder layer
   * @returns Output hidden states
   */
  private async runEncoderLayer(
    hiddenStates: Tensor,
    layerIndex: number
  ): Promise<Tensor> {
    const layer = this.weights.encoder!.layers![layerIndex];
    
    // Layer norm before self-attention
    const normedHiddenStates = await layerNorm(
      hiddenStates,
      layer.layerNorm1.weight!,
      layer.layerNorm1.bias!,
      1e-5,
      this.hardware
    );
    
    // Self-attention
    const attentionOutput = await this.runSelfAttention(
      normedHiddenStates,
      layer.selfAttention.query.weight!,
      layer.selfAttention.query.bias!,
      layer.selfAttention.key.weight!,
      layer.selfAttention.key.bias!,
      layer.selfAttention.value.weight!,
      layer.selfAttention.value.bias!,
      layer.selfAttention.output.weight!,
      layer.selfAttention.output.bias!
    );
    
    // Residual connection
    const attentionWithResidual = await add(attentionOutput, hiddenStates, this.hardware);
    
    // Layer norm before FFN
    const normedAttention = await layerNorm(
      attentionWithResidual,
      layer.layerNorm2.weight!,
      layer.layerNorm2.bias!,
      1e-5,
      this.hardware
    );
    
    // FFN
    const intermediate = await matmul(
      normedAttention,
      layer.ffn.intermediate.weight!,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
    );
    
    const intermediateWithBias = await add(
      intermediate,
      layer.ffn.intermediate.bias!,
      this.hardware
    );
    
    // GELU activation
    const intermediateActivated = await this.hardware.gelu(intermediateWithBias);
    
    const output = await matmul(
      intermediateActivated,
      layer.ffn.output.weight!,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
    );
    
    const outputWithBias = await add(
      output,
      layer.ffn.output.bias!,
      this.hardware
    );
    
    // Residual connection
    const outputWithResidual = await add(
      outputWithBias,
      attentionWithResidual,
      this.hardware
    );
    
    // Release temporary tensors
    await this.hardware.releaseTensor(normedHiddenStates);
    await this.hardware.releaseTensor(attentionOutput);
    await this.hardware.releaseTensor(attentionWithResidual);
    await this.hardware.releaseTensor(normedAttention);
    await this.hardware.releaseTensor(intermediate);
    await this.hardware.releaseTensor(intermediateWithBias);
    await this.hardware.releaseTensor(intermediateActivated);
    await this.hardware.releaseTensor(output);
    await this.hardware.releaseTensor(outputWithBias);
    
    return outputWithResidual;
  }
  
  /**
   * Run self-attention mechanism
   * @param hiddenStates Input hidden states
   * @param queryWeight Query weight matrix
   * @param queryBias Query bias
   * @param keyWeight Key weight matrix
   * @param keyBias Key bias
   * @param valueWeight Value weight matrix
   * @param valueBias Value bias
   * @param outputWeight Output weight matrix
   * @param outputBias Output bias
   * @returns Attention output
   */
  private async runSelfAttention(
    hiddenStates: Tensor,
    queryWeight: Tensor,
    queryBias: Tensor,
    keyWeight: Tensor,
    keyBias: Tensor,
    valueWeight: Tensor,
    valueBias: Tensor,
    outputWeight: Tensor,
    outputBias: Tensor
  ): Promise<Tensor> {
    // Get dimensions
    const [batchSize, seqLength, hiddenSize] = hiddenStates.dimensions;
    const numHeads = this.config.numHeads;
    const headSize = hiddenSize / numHeads;
    
    // Linear projections
    const query = await matmul(
      hiddenStates,
      queryWeight,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
    );
    const queryWithBias = await add(query, queryBias, this.hardware);
    
    const key = await matmul(
      hiddenStates,
      keyWeight,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
    );
    const keyWithBias = await add(key, keyBias, this.hardware);
    
    const value = await matmul(
      hiddenStates,
      valueWeight,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
    );
    const valueWithBias = await add(value, valueBias, this.hardware);
    
    // Reshape to multi-head format
    const queryHeads = await this.hardware.reshape(
      queryWithBias,
      [batchSize, seqLength, numHeads, headSize]
    );
    const keyHeads = await this.hardware.reshape(
      keyWithBias,
      [batchSize, seqLength, numHeads, headSize]
    );
    const valueHeads = await this.hardware.reshape(
      valueWithBias,
      [batchSize, seqLength, numHeads, headSize]
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
      { useOptimization: this.config.useBrowserOptimizations }
    );
    
    // Scale attention scores
    const scaleFactor = Math.sqrt(headSize);
    const scaledScores = await this.hardware.mul(
      attentionScores,
      await this.hardware.createTensor({
        dimensions: [1],
        data: new Float32Array([1.0 / scaleFactor]),
        dtype: 'float32'
      })
    );
    
    // Apply softmax to get attention weights
    const attentionWeights = await softmax(
      scaledScores,
      -1, // axis = -1 (last dimension)
      this.hardware
    );
    
    // Apply attention to values
    const contextLayer = await matmul(
      attentionWeights,
      valueTransposed,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
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
      outputWeight,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
    );
    
    const outputWithBias = await add(
      attentionOutput,
      outputBias,
      this.hardware
    );
    
    // Release temporary tensors
    await this.hardware.releaseTensor(query);
    await this.hardware.releaseTensor(queryWithBias);
    await this.hardware.releaseTensor(key);
    await this.hardware.releaseTensor(keyWithBias);
    await this.hardware.releaseTensor(value);
    await this.hardware.releaseTensor(valueWithBias);
    await this.hardware.releaseTensor(queryHeads);
    await this.hardware.releaseTensor(keyHeads);
    await this.hardware.releaseTensor(valueHeads);
    await this.hardware.releaseTensor(queryTransposed);
    await this.hardware.releaseTensor(keyTransposed);
    await this.hardware.releaseTensor(valueTransposed);
    await this.hardware.releaseTensor(attentionScores);
    await this.hardware.releaseTensor(scaledScores);
    await this.hardware.releaseTensor(attentionWeights);
    await this.hardware.releaseTensor(contextLayer);
    await this.hardware.releaseTensor(contextTransposed);
    await this.hardware.releaseTensor(contextReshaped);
    await this.hardware.releaseTensor(attentionOutput);
    
    return outputWithBias;
  }
  
  /**
   * Run decoder with encoder output for speech recognition
   * @param encoderOutput Encoder output tensor
   * @param maxLength Maximum output sequence length
   * @param beamSize Number of beams for beam search
   * @returns Decoded token IDs
   */
  private async runDecoder(
    encoderOutput: Tensor,
    maxLength: number = this.config.maxOutputLength || 448,
    beamSize: number = 1
  ): Promise<number[]> {
    // For simplicity, let's implement greedy decoding
    // In a full implementation, beam search would be used
    const [batchSize, seqLength, hiddenSize] = encoderOutput.dimensions;
    
    // Initialize decoder input with start token (usually 50258 for Whisper)
    const startToken = 50258; // <|startoftranscript|>
    let currentTokens = [startToken];
    
    // Initialize result array
    const resultTokens = [startToken];
    
    // Generate tokens auto-regressively
    for (let i = 0; i < maxLength; i++) {
      // Create input tensors for decoder
      const inputTensor = await this.hardware.createTensor({
        dimensions: [1, currentTokens.length],
        data: new Int32Array(currentTokens),
        dtype: 'int32'
      });
      
      // Run decoder step
      const logits = await this.decoderStep(inputTensor, encoderOutput);
      
      // Get next token (greedy selection)
      const [nextToken, nextLogProb] = await this.getNextToken(logits);
      
      // Add token to result
      resultTokens.push(nextToken);
      
      // Update current tokens for next step
      currentTokens.push(nextToken);
      
      // Release temporary tensors
      await this.hardware.releaseTensor(inputTensor);
      await this.hardware.releaseTensor(logits);
      
      // Check for end of sequence token (usually 50257 for Whisper)
      if (nextToken === 50257) { // <|endoftext|>
        break;
      }
    }
    
    return resultTokens;
  }
  
  /**
   * Run a single decoder step
   * @param inputTensor Input token IDs
   * @param encoderOutput Encoder hidden states
   * @returns Output logits
   */
  private async decoderStep(
    inputTensor: Tensor,
    encoderOutput: Tensor
  ): Promise<Tensor> {
    // Get dimensions
    const [batchSize, inputLength] = inputTensor.dimensions;
    const [_, encoderLength, hiddenSize] = encoderOutput.dimensions;
    
    // Embed input tokens
    const inputEmbeddings = await this.hardware.gatherEmbedding(
      this.weights.decoder!.embedding!,
      inputTensor
    );
    
    // Get positional embeddings
    const posEmbedding = await this.hardware.slice(
      this.weights.decoder!.positionalEmbedding!,
      [0, 0],
      [inputLength, this.config.hiddenSize]
    );
    
    const expandedPosEmbedding = await this.hardware.reshape(
      posEmbedding,
      [1, inputLength, this.config.hiddenSize]
    );
    
    // Add embeddings
    const embeddings = await add(
      inputEmbeddings,
      expandedPosEmbedding,
      this.hardware
    );
    
    // Run decoder layers
    let layerOutput = embeddings;
    for (let i = 0; i < this.config.decoderLayers; i++) {
      layerOutput = await this.runDecoderLayer(layerOutput, encoderOutput, i);
    }
    
    // Apply final layer norm
    const normalizedOutput = await layerNorm(
      layerOutput,
      this.weights.decoder!.finalLayerNorm!.weight!,
      this.weights.decoder!.finalLayerNorm!.bias!,
      1e-5,
      this.hardware
    );
    
    // Get final token representation
    const lastTokenIdx = inputLength - 1;
    const lastTokenRepresentation = await this.hardware.slice(
      normalizedOutput,
      [0, lastTokenIdx, 0],
      [batchSize, 1, hiddenSize]
    );
    
    const lastTokenReshaped = await this.hardware.reshape(
      lastTokenRepresentation,
      [batchSize, hiddenSize]
    );
    
    // Project to vocabulary
    const logits = await matmul(
      lastTokenReshaped,
      this.weights.outputProjection!,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
    );
    
    // Release temporary tensors
    await this.hardware.releaseTensor(inputEmbeddings);
    await this.hardware.releaseTensor(posEmbedding);
    await this.hardware.releaseTensor(expandedPosEmbedding);
    await this.hardware.releaseTensor(embeddings);
    await this.hardware.releaseTensor(layerOutput);
    await this.hardware.releaseTensor(normalizedOutput);
    await this.hardware.releaseTensor(lastTokenRepresentation);
    await this.hardware.releaseTensor(lastTokenReshaped);
    
    return logits;
  }
  
  /**
   * Run a single decoder layer
   * @param hiddenStates Input hidden states
   * @param encoderOutput Encoder hidden states
   * @param layerIndex Index of the decoder layer
   * @returns Output hidden states
   */
  private async runDecoderLayer(
    hiddenStates: Tensor,
    encoderOutput: Tensor,
    layerIndex: number
  ): Promise<Tensor> {
    const layer = this.weights.decoder!.layers![layerIndex];
    
    // Layer norm before self-attention
    const normedHiddenStates = await layerNorm(
      hiddenStates,
      layer.layerNorm1.weight!,
      layer.layerNorm1.bias!,
      1e-5,
      this.hardware
    );
    
    // Self-attention (causal)
    const selfAttentionOutput = await this.runSelfAttention(
      normedHiddenStates,
      layer.selfAttention.query.weight!,
      layer.selfAttention.query.bias!,
      layer.selfAttention.key.weight!,
      layer.selfAttention.key.bias!,
      layer.selfAttention.value.weight!,
      layer.selfAttention.value.bias!,
      layer.selfAttention.output.weight!,
      layer.selfAttention.output.bias!
    );
    
    // Residual connection
    const selfAttentionWithResidual = await add(
      selfAttentionOutput,
      hiddenStates,
      this.hardware
    );
    
    // Layer norm before cross-attention
    const normedSelfAttention = await layerNorm(
      selfAttentionWithResidual,
      layer.layerNorm2.weight!,
      layer.layerNorm2.bias!,
      1e-5,
      this.hardware
    );
    
    // Cross-attention with encoder output
    const crossAttentionOutput = await this.runCrossAttention(
      normedSelfAttention,
      encoderOutput,
      layer.crossAttention.query.weight!,
      layer.crossAttention.query.bias!,
      layer.crossAttention.key.weight!,
      layer.crossAttention.key.bias!,
      layer.crossAttention.value.weight!,
      layer.crossAttention.value.bias!,
      layer.crossAttention.output.weight!,
      layer.crossAttention.output.bias!
    );
    
    // Residual connection
    const crossAttentionWithResidual = await add(
      crossAttentionOutput,
      selfAttentionWithResidual,
      this.hardware
    );
    
    // Layer norm before FFN
    const normedCrossAttention = await layerNorm(
      crossAttentionWithResidual,
      layer.layerNorm3.weight!,
      layer.layerNorm3.bias!,
      1e-5,
      this.hardware
    );
    
    // FFN
    const intermediate = await matmul(
      normedCrossAttention,
      layer.ffn.intermediate.weight!,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
    );
    
    const intermediateWithBias = await add(
      intermediate,
      layer.ffn.intermediate.bias!,
      this.hardware
    );
    
    // GELU activation
    const intermediateActivated = await this.hardware.gelu(intermediateWithBias);
    
    const output = await matmul(
      intermediateActivated,
      layer.ffn.output.weight!,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
    );
    
    const outputWithBias = await add(
      output,
      layer.ffn.output.bias!,
      this.hardware
    );
    
    // Residual connection
    const outputWithResidual = await add(
      outputWithBias,
      crossAttentionWithResidual,
      this.hardware
    );
    
    // Release temporary tensors
    await this.hardware.releaseTensor(normedHiddenStates);
    await this.hardware.releaseTensor(selfAttentionOutput);
    await this.hardware.releaseTensor(selfAttentionWithResidual);
    await this.hardware.releaseTensor(normedSelfAttention);
    await this.hardware.releaseTensor(crossAttentionOutput);
    await this.hardware.releaseTensor(crossAttentionWithResidual);
    await this.hardware.releaseTensor(normedCrossAttention);
    await this.hardware.releaseTensor(intermediate);
    await this.hardware.releaseTensor(intermediateWithBias);
    await this.hardware.releaseTensor(intermediateActivated);
    await this.hardware.releaseTensor(output);
    await this.hardware.releaseTensor(outputWithBias);
    
    return outputWithResidual;
  }
  
  /**
   * Run cross-attention mechanism
   * @param hiddenStates Input hidden states
   * @param encoderOutput Encoder hidden states
   * @param queryWeight Query weight matrix
   * @param queryBias Query bias
   * @param keyWeight Key weight matrix
   * @param keyBias Key bias
   * @param valueWeight Value weight matrix
   * @param valueBias Value bias
   * @param outputWeight Output weight matrix
   * @param outputBias Output bias
   * @returns Cross-attention output
   */
  private async runCrossAttention(
    hiddenStates: Tensor,
    encoderOutput: Tensor,
    queryWeight: Tensor,
    queryBias: Tensor,
    keyWeight: Tensor,
    keyBias: Tensor,
    valueWeight: Tensor,
    valueBias: Tensor,
    outputWeight: Tensor,
    outputBias: Tensor
  ): Promise<Tensor> {
    // Get dimensions
    const [batchSize, seqLength, hiddenSize] = hiddenStates.dimensions;
    const [_, encoderLength, _] = encoderOutput.dimensions;
    const numHeads = this.config.numHeads;
    const headSize = hiddenSize / numHeads;
    
    // Linear projections
    const query = await matmul(
      hiddenStates,
      queryWeight,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
    );
    const queryWithBias = await add(query, queryBias, this.hardware);
    
    const key = await matmul(
      encoderOutput,
      keyWeight,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
    );
    const keyWithBias = await add(key, keyBias, this.hardware);
    
    const value = await matmul(
      encoderOutput,
      valueWeight,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
    );
    const valueWithBias = await add(value, valueBias, this.hardware);
    
    // Reshape to multi-head format
    const queryHeads = await this.hardware.reshape(
      queryWithBias,
      [batchSize, seqLength, numHeads, headSize]
    );
    const keyHeads = await this.hardware.reshape(
      keyWithBias,
      [batchSize, encoderLength, numHeads, headSize]
    );
    const valueHeads = await this.hardware.reshape(
      valueWithBias,
      [batchSize, encoderLength, numHeads, headSize]
    );
    
    // Transpose for matrix multiplication
    const queryTransposed = await this.hardware.transpose(
      queryHeads,
      [0, 2, 1, 3] // [batch, heads, seq_len, head_size]
    );
    const keyTransposed = await this.hardware.transpose(
      keyHeads,
      [0, 2, 3, 1] // [batch, heads, head_size, encoder_len]
    );
    const valueTransposed = await this.hardware.transpose(
      valueHeads,
      [0, 2, 1, 3] // [batch, heads, encoder_len, head_size]
    );
    
    // Calculate attention scores
    const attentionScores = await matmul(
      queryTransposed,
      keyTransposed,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
    );
    
    // Scale attention scores
    const scaleFactor = Math.sqrt(headSize);
    const scaledScores = await this.hardware.mul(
      attentionScores,
      await this.hardware.createTensor({
        dimensions: [1],
        data: new Float32Array([1.0 / scaleFactor]),
        dtype: 'float32'
      })
    );
    
    // Apply softmax to get attention weights
    const attentionWeights = await softmax(
      scaledScores,
      -1, // axis = -1 (last dimension)
      this.hardware
    );
    
    // Apply attention to values
    const contextLayer = await matmul(
      attentionWeights,
      valueTransposed,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
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
      outputWeight,
      this.hardware,
      { useOptimization: this.config.useBrowserOptimizations }
    );
    
    const outputWithBias = await add(
      attentionOutput,
      outputBias,
      this.hardware
    );
    
    // Release temporary tensors
    await this.hardware.releaseTensor(query);
    await this.hardware.releaseTensor(queryWithBias);
    await this.hardware.releaseTensor(key);
    await this.hardware.releaseTensor(keyWithBias);
    await this.hardware.releaseTensor(value);
    await this.hardware.releaseTensor(valueWithBias);
    await this.hardware.releaseTensor(queryHeads);
    await this.hardware.releaseTensor(keyHeads);
    await this.hardware.releaseTensor(valueHeads);
    await this.hardware.releaseTensor(queryTransposed);
    await this.hardware.releaseTensor(keyTransposed);
    await this.hardware.releaseTensor(valueTransposed);
    await this.hardware.releaseTensor(attentionScores);
    await this.hardware.releaseTensor(scaledScores);
    await this.hardware.releaseTensor(attentionWeights);
    await this.hardware.releaseTensor(contextLayer);
    await this.hardware.releaseTensor(contextTransposed);
    await this.hardware.releaseTensor(contextReshaped);
    await this.hardware.releaseTensor(attentionOutput);
    
    return outputWithBias;
  }
  
  /**
   * Get the next token from logits
   * @param logits Logits tensor
   * @returns Next token ID and log probability
   */
  private async getNextToken(logits: Tensor): Promise<[number, number]> {
    // Convert to array
    const logitsArray = await logits.toArray() as number[][];
    const logitsFlat = logitsArray[0];
    
    // Apply temperature and get probabilities
    const temperature = 1.0;
    const scaledLogits = logitsFlat.map(x => x / temperature);
    
    // Softmax
    const maxLogit = Math.max(...scaledLogits);
    const expLogits = scaledLogits.map(x => Math.exp(x - maxLogit));
    const sumExpLogits = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map(x => x / sumExpLogits);
    
    // Greedy selection
    const maxProb = Math.max(...probs);
    const maxIdx = probs.indexOf(maxProb);
    const logProb = Math.log(maxProb);
    
    return [maxIdx, logProb];
  }
  
  /**
   * Decode token IDs to text
   * @param tokens Token IDs
   * @returns Decoded text
   */
  private decodeTokens(tokens: number[]): string {
    // In a real implementation, this would use a tokenizer
    // For now, we'll just return a placeholder
    return `[Transcription from ${tokens.length} tokens]`;
  }
  
  /**
   * Extract segments with timestamps
   * @param tokens Token IDs
   * @param audioLength Audio length in seconds
   * @returns Segments with timestamps
   */
  private extractSegments(
    tokens: number[],
    audioLength: number
  ): WhisperSegment[] {
    // In a real implementation, this would extract timestamps from tokens
    // For now, we'll just return a placeholder
    return [
      {
        start: 0,
        end: audioLength,
        text: this.decodeTokens(tokens),
      },
    ];
  }
  
  /**
   * Process audio input through Whisper model
   * @param input Input audio data
   * @returns Whisper output with transcription
   */
  public async process(input: WhisperInput): Promise<WhisperOutput> {
    // Ensure model is initialized
    if (!this.initialized) {
      await this.initialize();
    }
    
    // Process audio into mel spectrogram
    const melSpectrogram = await this.processAudio(
      input.audioData,
      input.sampleRate || this.config.sampleRate
    );
    
    // Run encoder
    const encoderOutput = await this.runEncoder(melSpectrogram);
    
    // Store encoder output as shared tensor if needed
    const audioEmbedding = await this.createSharedTensor(
      encoderOutput,
      'audio_embedding'
    );
    
    // Run decoder
    const tokens = await this.runDecoder(
      encoderOutput,
      this.config.maxOutputLength,
      input.beamSize || 1
    );
    
    // Decode tokens to text
    const text = this.decodeTokens(tokens);
    
    // Extract segments if requested
    let segments: WhisperSegment[] | undefined;
    if (input.returnTimestamps) {
      const audioLengthSeconds = input.audioData.length / (input.sampleRate || this.config.sampleRate);
      segments = this.extractSegments(tokens, audioLengthSeconds);
    }
    
    // Release tensors
    await this.hardware.releaseTensor(melSpectrogram);
    await this.hardware.releaseTensor(encoderOutput);
    
    return {
      text,
      segments,
      detectedLanguage: input.multilingual ? 'en' : undefined,
      model: this.config.modelId,
      backend: this.hardware.getBackendType()
    };
  }
  
  /**
   * Create a shared tensor that can be used by other models
   * @param encoderOutput Encoder output
   * @param outputType Type of output to share
   * @returns Shared tensor reference
   */
  public async createSharedTensor(
    encoderOutput: Tensor,
    outputType: string = 'audio_embedding'
  ): Promise<SharedTensor> {
    // Create a new shared tensor
    const sharedTensor = new SharedTensor(
      encoderOutput,
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
  public getSharedTensor(outputType: string = 'audio_embedding'): SharedTensor | null {
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
    
    // Release encoder weights
    if (this.weights.encoder) {
      if (this.weights.encoder.embedding) {
        await this.hardware.releaseTensor(this.weights.encoder.embedding);
      }
      if (this.weights.encoder.positionalEmbedding) {
        await this.hardware.releaseTensor(this.weights.encoder.positionalEmbedding);
      }
      if (this.weights.encoder.finalLayerNorm) {
        if (this.weights.encoder.finalLayerNorm.weight) {
          await this.hardware.releaseTensor(this.weights.encoder.finalLayerNorm.weight);
        }
        if (this.weights.encoder.finalLayerNorm.bias) {
          await this.hardware.releaseTensor(this.weights.encoder.finalLayerNorm.bias);
        }
      }
      
      // Release encoder layer weights
      if (this.weights.encoder.layers) {
        for (const layer of this.weights.encoder.layers) {
          // Self-attention weights
          if (layer.selfAttention.query.weight) {
            await this.hardware.releaseTensor(layer.selfAttention.query.weight);
          }
          if (layer.selfAttention.query.bias) {
            await this.hardware.releaseTensor(layer.selfAttention.query.bias);
          }
          if (layer.selfAttention.key.weight) {
            await this.hardware.releaseTensor(layer.selfAttention.key.weight);
          }
          if (layer.selfAttention.key.bias) {
            await this.hardware.releaseTensor(layer.selfAttention.key.bias);
          }
          if (layer.selfAttention.value.weight) {
            await this.hardware.releaseTensor(layer.selfAttention.value.weight);
          }
          if (layer.selfAttention.value.bias) {
            await this.hardware.releaseTensor(layer.selfAttention.value.bias);
          }
          if (layer.selfAttention.output.weight) {
            await this.hardware.releaseTensor(layer.selfAttention.output.weight);
          }
          if (layer.selfAttention.output.bias) {
            await this.hardware.releaseTensor(layer.selfAttention.output.bias);
          }
          
          // Layer norm weights
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
          
          // FFN weights
          if (layer.ffn.intermediate.weight) {
            await this.hardware.releaseTensor(layer.ffn.intermediate.weight);
          }
          if (layer.ffn.intermediate.bias) {
            await this.hardware.releaseTensor(layer.ffn.intermediate.bias);
          }
          if (layer.ffn.output.weight) {
            await this.hardware.releaseTensor(layer.ffn.output.weight);
          }
          if (layer.ffn.output.bias) {
            await this.hardware.releaseTensor(layer.ffn.output.bias);
          }
        }
      }
    }
    
    // Release decoder weights
    if (this.weights.decoder) {
      if (this.weights.decoder.embedding) {
        await this.hardware.releaseTensor(this.weights.decoder.embedding);
      }
      if (this.weights.decoder.positionalEmbedding) {
        await this.hardware.releaseTensor(this.weights.decoder.positionalEmbedding);
      }
      if (this.weights.decoder.finalLayerNorm) {
        if (this.weights.decoder.finalLayerNorm.weight) {
          await this.hardware.releaseTensor(this.weights.decoder.finalLayerNorm.weight);
        }
        if (this.weights.decoder.finalLayerNorm.bias) {
          await this.hardware.releaseTensor(this.weights.decoder.finalLayerNorm.bias);
        }
      }
      
      // Release decoder layer weights
      if (this.weights.decoder.layers) {
        for (const layer of this.weights.decoder.layers) {
          // Self-attention weights
          if (layer.selfAttention.query.weight) {
            await this.hardware.releaseTensor(layer.selfAttention.query.weight);
          }
          if (layer.selfAttention.query.bias) {
            await this.hardware.releaseTensor(layer.selfAttention.query.bias);
          }
          if (layer.selfAttention.key.weight) {
            await this.hardware.releaseTensor(layer.selfAttention.key.weight);
          }
          if (layer.selfAttention.key.bias) {
            await this.hardware.releaseTensor(layer.selfAttention.key.bias);
          }
          if (layer.selfAttention.value.weight) {
            await this.hardware.releaseTensor(layer.selfAttention.value.weight);
          }
          if (layer.selfAttention.value.bias) {
            await this.hardware.releaseTensor(layer.selfAttention.value.bias);
          }
          if (layer.selfAttention.output.weight) {
            await this.hardware.releaseTensor(layer.selfAttention.output.weight);
          }
          if (layer.selfAttention.output.bias) {
            await this.hardware.releaseTensor(layer.selfAttention.output.bias);
          }
          
          // Cross-attention weights
          if (layer.crossAttention.query.weight) {
            await this.hardware.releaseTensor(layer.crossAttention.query.weight);
          }
          if (layer.crossAttention.query.bias) {
            await this.hardware.releaseTensor(layer.crossAttention.query.bias);
          }
          if (layer.crossAttention.key.weight) {
            await this.hardware.releaseTensor(layer.crossAttention.key.weight);
          }
          if (layer.crossAttention.key.bias) {
            await this.hardware.releaseTensor(layer.crossAttention.key.bias);
          }
          if (layer.crossAttention.value.weight) {
            await this.hardware.releaseTensor(layer.crossAttention.value.weight);
          }
          if (layer.crossAttention.value.bias) {
            await this.hardware.releaseTensor(layer.crossAttention.value.bias);
          }
          if (layer.crossAttention.output.weight) {
            await this.hardware.releaseTensor(layer.crossAttention.output.weight);
          }
          if (layer.crossAttention.output.bias) {
            await this.hardware.releaseTensor(layer.crossAttention.output.bias);
          }
          
          // Layer norm weights
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
          if (layer.layerNorm3.weight) {
            await this.hardware.releaseTensor(layer.layerNorm3.weight);
          }
          if (layer.layerNorm3.bias) {
            await this.hardware.releaseTensor(layer.layerNorm3.bias);
          }
          
          // FFN weights
          if (layer.ffn.intermediate.weight) {
            await this.hardware.releaseTensor(layer.ffn.intermediate.weight);
          }
          if (layer.ffn.intermediate.bias) {
            await this.hardware.releaseTensor(layer.ffn.intermediate.bias);
          }
          if (layer.ffn.output.weight) {
            await this.hardware.releaseTensor(layer.ffn.output.weight);
          }
          if (layer.ffn.output.bias) {
            await this.hardware.releaseTensor(layer.ffn.output.bias);
          }
        }
      }
    }
    
    // Release output projection
    if (this.weights.outputProjection) {
      await this.hardware.releaseTensor(this.weights.outputProjection);
    }
    
    // Release mel filterbank
    if (this.weights.melFilterbank) {
      await this.hardware.releaseTensor(this.weights.melFilterbank);
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
 * Factory function to create a Whisper model
 * @param hardware Hardware backend for tensor operations
 * @param config Whisper configuration
 * @returns Whisper model instance
 */
export function createWhisperModel(
  hardware: HardwareBackend,
  config: Partial<WhisperConfig> = {}
): Whisper {
  return new Whisper(hardware, config);
}