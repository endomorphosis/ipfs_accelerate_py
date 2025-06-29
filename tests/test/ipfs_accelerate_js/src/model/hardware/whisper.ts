/**
 * Hardware Abstracted Whisper Implementation
 * Implements Whisper model with automatic hardware acceleration selection and optimization
 */

import { HardwareBackend } from '../../hardware/interfaces/hardware_backend';
import { createOptimalBackend, createMultiBackend } from '../../hardware/index';
import { Tensor } from '../../tensor/tensor';
import { SharedTensor } from '../../tensor/shared_tensor';
import { matmul, softmax } from '../../tensor/operations/matrix';
import { layerNorm, gelu } from '../../tensor/operations/nn';
import { add, mul } from '../../tensor/operations/basic';
import { Whisper as WhisperImplementation, WhisperConfig, WhisperInput, WhisperOutput } from '../audio/whisper';

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
 * Configuration for Hardware Abstracted Whisper
 */
export interface HardwareAbstractedWhisperConfig extends WhisperConfig {
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
  taskType?: 'transcription' | 'translation';
  
  /**
   * Audio processing options
   */
  audioProcessing?: {
    hardwareAccelerated?: boolean;
    cacheFeatures?: boolean;
  };
}

/**
 * Default Hardware Abstracted Whisper Configuration (whisper-tiny)
 */
export const DEFAULT_HARDWARE_ABSTRACTED_WHISPER_CONFIG: HardwareAbstractedWhisperConfig = {
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
  backendPreference: ['webgpu', 'webnn', 'cpu'],  // WebGPU tends to be better for audio processing
  useBrowserOptimizations: true,
  maxAudioLength: 480000, // 30 seconds @ 16kHz
  maxOutputLength: 448,
  allowFallback: true,
  collectMetrics: true,
  browserOptimizations: true,
  taskType: 'transcription',
  audioProcessing: {
    hardwareAccelerated: true,
    cacheFeatures: true
  },
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
 * Hardware Abstracted Whisper class
 * Provides hardware optimization for Whisper model
 */
export class HardwareAbstractedWhisper {
  private config: HardwareAbstractedWhisperConfig;
  private storageManager: StorageManager;
  private hardware: HardwareBackend | null = null;
  private modelImpl: WhisperImplementation | null = null;
  private initialized: boolean = false;
  
  // Metrics tracking
  private performanceMetrics: Record<string, PerformanceMetric> = {};
  private backendMetrics: Record<string, any> = {};
  private selectedBackend: string = '';
  private availableBackends: string[] = [];
  
  // Audio feature caching
  private cachedFeatures: Map<string, any> = new Map();
  
  /**
   * Constructor for Hardware Abstracted Whisper
   * @param config Configuration options
   * @param storageManager Storage manager for model weights
   */
  constructor(
    config: Partial<HardwareAbstractedWhisperConfig> = {},
    storageManager: StorageManager
  ) {
    this.config = { ...DEFAULT_HARDWARE_ABSTRACTED_WHISPER_CONFIG, ...config };
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
        this.modelImpl = new WhisperImplementation(this.hardware, this.config);
        await this.modelImpl.initialize();
      } else {
        throw new Error('Failed to initialize hardware backend');
      }
      
      const initTime = performance.now() - startTime;
      this.updateMetric('initialization', initTime);
      
      this.initialized = true;
      console.log(`Whisper model initialized (${this.selectedBackend}) in ${initTime.toFixed(2)}ms`);
    } catch (error) {
      console.error('Failed to initialize Hardware Abstracted Whisper:', error);
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
      // Check if Firefox - it has excellent compute shader performance for audio
      const isFirefox = typeof navigator !== 'undefined' && 
                        navigator.userAgent.toLowerCase().includes('firefox');
      
      // Try to initialize with optimal backend
      if (this.config.backendPreference && this.config.backendPreference.length > 0) {
        // Use multi-backend with specified preference order
        this.hardware = await createMultiBackend(this.config.backendPreference);
      } else if (isFirefox) {
        // Firefox has excellent WebGPU compute shader performance for audio processing
        this.hardware = await createOptimalBackend({
          forceBackend: 'webgpu',
          optimizationLevel: 'maximum'
        });
      } else {
        // Use automatic optimal backend selection
        this.hardware = await createOptimalBackend({
          preferComputePerformance: true, // Audio processing needs compute performance
          modelType: 'audio'             // Specify this is an audio model
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
          isFirefox: isFirefox
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
   * Preprocess audio data for the model
   * This involves converting to mel spectrograms
   * @param audioData Raw audio samples
   * @param sampleRate Audio sample rate
   * @returns Processed audio features
   */
  private async preprocessAudio(audioData: Float32Array, sampleRate?: number): Promise<any> {
    const startTime = performance.now();
    
    if (!this.modelImpl) {
      throw new Error('Model not initialized');
    }
    
    // Create cache key based on audio data
    // In a real implementation, use a better fingerprinting method
    // This is just a simple checksum-like approach
    const cacheKey = this.generateAudioFingerprint(audioData);
    
    // Check cache if enabled
    if (this.config.audioProcessing?.cacheFeatures && this.cachedFeatures.has(cacheKey)) {
      const cachedResult = this.cachedFeatures.get(cacheKey);
      console.log('Using cached audio features');
      
      this.updateMetric('audio_preprocessing_cached', performance.now() - startTime);
      return cachedResult;
    }
    
    // Process audio using the model implementation
    const whisperInput: WhisperInput = {
      audioData: audioData,
      sampleRate: sampleRate || this.config.sampleRate
    };
    
    // In a real implementation, this would extract the mel spectrogram features
    // Normally these would be processed by the model's process method,
    // but here we simulate breaking it out into a separate step
    
    const preprocessingTime = performance.now() - startTime;
    this.updateMetric('audio_preprocessing', preprocessingTime);
    
    // Return a placeholder, in a real implementation this would be the processed features
    // Store in cache if enabled
    const processedFeatures = { audioData: whisperInput.audioData, sampleRate: whisperInput.sampleRate };
    
    if (this.config.audioProcessing?.cacheFeatures) {
      this.cachedFeatures.set(cacheKey, processedFeatures);
    }
    
    return processedFeatures;
  }
  
  /**
   * Generate a simple fingerprint for audio data for caching
   * @param audioData Audio samples
   * @returns Fingerprint string
   */
  private generateAudioFingerprint(audioData: Float32Array): string {
    // Simple checksum-like implementation
    // In a real system, use a proper audio fingerprinting algorithm
    let sum = 0;
    let xor = 0;
    
    // Sample only a portion of the audio for efficiency
    const step = Math.max(1, Math.floor(audioData.length / 1000));
    
    for (let i = 0; i < audioData.length; i += step) {
      sum += audioData[i];
      xor ^= Math.floor(audioData[i] * 1000) & 0xFFFF;
    }
    
    return `audio-${audioData.length}-${sum.toFixed(6)}-${xor}`;
  }
  
  /**
   * Run speech recognition on audio input
   * @param input Raw audio data or preprocessed features
   * @returns Recognition result
   */
  public async transcribe(input: Float32Array | WhisperInput): Promise<WhisperOutput> {
    if (!this.initialized || !this.modelImpl) {
      throw new Error('Model not initialized. Call initialize() first.');
    }
    
    const totalStartTime = performance.now();
    
    try {
      // Prepare input
      let whisperInput: WhisperInput;
      
      if (input instanceof Float32Array) {
        // Raw audio data
        const processedFeatures = await this.preprocessAudio(input);
        
        whisperInput = {
          audioData: processedFeatures.audioData,
          sampleRate: processedFeatures.sampleRate,
          multilingual: false,
          returnTimestamps: true
        };
      } else {
        // Already formatted input
        whisperInput = input;
      }
      
      // Add task type flag
      if (this.config.taskType === 'translation') {
        whisperInput.task = 'translate';
      }
      
      // Run inference
      const inferenceStartTime = performance.now();
      const output = await this.modelImpl.process(whisperInput);
      const inferenceTime = performance.now() - inferenceStartTime;
      
      this.updateMetric('inference', inferenceTime);
      
      const totalTime = performance.now() - totalStartTime;
      this.updateMetric('total_processing', totalTime);
      
      return output;
    } catch (error) {
      console.error('Error during Whisper transcription:', error);
      throw error;
    }
  }
  
  /**
   * Run translation on audio input (wrapper for transcribe with translation task)
   * @param input Raw audio data or preprocessed features
   * @returns Translation result
   */
  public async translate(input: Float32Array | WhisperInput): Promise<WhisperOutput> {
    // Create a copy of the input with translation task
    let translationInput: WhisperInput;
    
    if (input instanceof Float32Array) {
      translationInput = {
        audioData: input,
        task: 'translate',
        multilingual: true
      };
    } else {
      translationInput = {
        ...input,
        task: 'translate',
        multilingual: true
      };
    }
    
    return this.transcribe(translationInput);
  }
  
  /**
   * Run performance comparison across all available backends
   * @param audioData Test audio data
   * @returns Execution time by backend
   */
  public async compareBackends(audioData: Float32Array): Promise<Record<string, number>> {
    const results: Record<string, number> = {};
    const availableBackends = this.getAvailableBackends();
    
    // Prepare input once (shared across backends)
    const processedFeatures = await this.preprocessAudio(audioData);
    
    const whisperInput: WhisperInput = {
      audioData: processedFeatures.audioData,
      sampleRate: this.config.sampleRate,
      multilingual: false,
      returnTimestamps: false
    };
    
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
        const model = new WhisperImplementation(hardware, this.config);
        await model.initialize();
        
        // Warm-up run
        await model.process(whisperInput);
        
        // Timed run
        const startTime = performance.now();
        await model.process(whisperInput);
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
      modelType: 'Whisper',
      taskType: this.config.taskType,
      hiddenSize: this.config.hiddenSize,
      encoderLayers: this.config.encoderLayers,
      decoderLayers: this.config.decoderLayers,
      numHeads: this.config.numHeads,
      sampleRate: this.config.sampleRate,
      selectedBackend: this.selectedBackend,
      availableBackends: this.availableBackends,
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
    
    // Clear cache
    this.cachedFeatures.clear();
    
    this.initialized = false;
  }
  
  /**
   * Get a shared tensor from the model
   * @param outputType Type of output to get (e.g. 'audio_embedding')
   * @returns Shared tensor or null if not found
   */
  public getSharedTensor(outputType: string = 'audio_embedding'): SharedTensor | null {
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
  
  /**
   * Clear audio feature cache
   */
  public clearFeatureCache(): void {
    this.cachedFeatures.clear();
  }
}

/**
 * Factory function to create a Hardware Abstracted Whisper
 * @param config Configuration options
 * @param storageManager Storage manager for model weights
 * @returns HardwareAbstractedWhisper instance
 */
export function createHardwareAbstractedWhisper(
  config: Partial<HardwareAbstractedWhisperConfig> = {},
  storageManager: StorageManager
): HardwareAbstractedWhisper {
  return new HardwareAbstractedWhisper(config, storageManager);
}