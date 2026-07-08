/**
 * Hardware Abstracted Whisper Example
 * Demonstrates using the Hardware Abstraction Layer with Whisper model
 */

import { 
  createHardwareAbstractedWhisper, 
  createHardwareAbstractedBERT, 
  HardwareAbstractedWhisperConfig, 
  StorageManager, 
  IndexedDBStorageManager 
} from '../../../src/index';

/**
 * Audio processing utilities for Whisper
 */
export class AudioProcessor {
  private audioContext: AudioContext;
  
  constructor() {
    this.audioContext = new (window.AudioContext || window["webkitAudioContext"])();
  }
  
  /**
   * Process audio file into Float32Array at 16kHz
   * @param audioFile Audio file to process
   * @returns Float32Array of audio samples at 16kHz
   */
  async processAudioFile(audioFile: File): Promise<Float32Array> {
    const arrayBuffer = await audioFile.arrayBuffer();
    const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
    return this.resampleAndNormalize(audioBuffer);
  }
  
  /**
   * Process audio blob into Float32Array at 16kHz
   * @param blob Audio blob to process
   * @returns Float32Array of audio samples at 16kHz
   */
  async processAudioBlob(blob: Blob): Promise<Float32Array> {
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
    return this.resampleAndNormalize(audioBuffer);
  }
  
  /**
   * Record audio from microphone
   * @param durationMs Duration in milliseconds
   * @returns Float32Array of audio samples at 16kHz
   */
  async recordAudio(durationMs: number = 5000): Promise<Float32Array> {
    return new Promise(async (resolve, reject) => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const recorder = new MediaRecorder(stream);
        const chunks: BlobPart[] = [];
        
        recorder.ondataavailable = (e) => chunks.push(e.data);
        
        recorder.onstop = async () => {
          // Stop all tracks to release the microphone
          stream.getTracks().forEach(track => track.stop());
          
          const blob = new Blob(chunks, { type: 'audio/wav' });
          try {
            const audioData = await this.processAudioBlob(blob);
            resolve(audioData);
          } catch (error) {
            reject(error);
          }
        };
        
        // Start recording
        recorder.start();
        
        // Stop after duration
        setTimeout(() => {
          if (recorder.state === 'recording') {
            recorder.stop();
          }
        }, durationMs);
      } catch (error) {
        reject(error);
      }
    });
  }
  
  /**
   * Resample and normalize audio to 16kHz mono
   * @param audioBuffer Audio buffer to process
   * @returns Float32Array of audio samples at 16kHz
   */
  private resampleAndNormalize(audioBuffer: AudioBuffer): Float32Array {
    const numChannels = audioBuffer.numberOfChannels;
    const targetSampleRate = 16000;
    const originalSampleRate = audioBuffer.sampleRate;
    
    // Get all channel data and mix down to mono
    let monoData: Float32Array;
    if (numChannels === 1) {
      monoData = audioBuffer.getChannelData(0);
    } else {
      // Mix down to mono
      monoData = new Float32Array(audioBuffer.length);
      for (let i = 0; i < audioBuffer.length; i++) {
        let sum = 0;
        for (let channel = 0; channel < numChannels; channel++) {
          sum += audioBuffer.getChannelData(channel)[i];
        }
        monoData[i] = sum / numChannels;
      }
    }
    
    // If already at 16kHz, return as is
    if (originalSampleRate === targetSampleRate) {
      return monoData;
    }
    
    // Resample to 16kHz (simple resampling for demo purposes)
    // In a production app, use a proper resampling algorithm
    const resampleRatio = targetSampleRate / originalSampleRate;
    const newLength = Math.round(monoData.length * resampleRatio);
    const resampledData = new Float32Array(newLength);
    
    for (let i = 0; i < newLength; i++) {
      const originalIndex = Math.min(Math.floor(i / resampleRatio), monoData.length - 1);
      resampledData[i] = monoData[originalIndex];
    }
    
    return resampledData;
  }
  
  /**
   * Play audio from Float32Array
   * @param audioData Audio data to play
   * @param sampleRate Sample rate (default: 16000)
   */
  playAudio(audioData: Float32Array, sampleRate: number = 16000): void {
    // Create a new buffer with the specified sample rate
    const buffer = this.audioContext.createBuffer(1, audioData.length, sampleRate);
    
    // Copy the audio data to the buffer
    buffer.copyToChannel(audioData, 0);
    
    // Create a source node
    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    
    // Connect to the audio context destination
    source.connect(this.audioContext.destination);
    
    // Start playback
    source.start();
  }
}

/**
 * Whisper model manager
 */
export class WhisperManager {
  private model: any = null;
  private storageManager: StorageManager;
  private audioProcessor: AudioProcessor;
  private initialized: boolean = false;
  
  constructor() {
    this.audioProcessor = new AudioProcessor();
  }
  
  /**
   * Initialize the Whisper manager
   */
  async initialize(): Promise<void> {
    // Initialize storage manager
    this.storageManager = new IndexedDBStorageManager();
    await this.storageManager.initialize();
    this.initialized = true;
  }
  
  /**
   * Create and initialize a Whisper model
   * @param modelSize Model size (tiny, base, small, medium, large)
   * @param options Additional configuration options
   * @returns Performance metrics from initialization
   */
  async createModel(modelSize: string = 'tiny', options: Partial<HardwareAbstractedWhisperConfig> = {}): Promise<any> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    // Default config based on model size
    const config: Partial<HardwareAbstractedWhisperConfig> = {
      modelId: `openai/whisper-${modelSize}`,
      taskType: 'transcription',
      ...options
    };
    
    // Create and initialize the model
    this.model = createHardwareAbstractedWhisper(config, this.storageManager);
    await this.model.initialize();
    
    // Return backend info and model info
    return {
      backendInfo: this.model.getBackendMetrics(),
      modelInfo: this.model.getModelInfo()
    };
  }
  
  /**
   * Transcribe audio file
   * @param audioFile Audio file to transcribe
   * @returns Transcription result
   */
  async transcribeFile(audioFile: File): Promise<any> {
    if (!this.model) {
      throw new Error('Model not initialized. Call createModel() first.');
    }
    
    const audioData = await this.audioProcessor.processAudioFile(audioFile);
    return this.model.transcribe(audioData);
  }
  
  /**
   * Transcribe audio data
   * @param audioData Audio data to transcribe
   * @returns Transcription result
   */
  async transcribeAudio(audioData: Float32Array): Promise<any> {
    if (!this.model) {
      throw new Error('Model not initialized. Call createModel() first.');
    }
    
    return this.model.transcribe(audioData);
  }
  
  /**
   * Record and transcribe audio
   * @param durationMs Duration in milliseconds
   * @returns Transcription result
   */
  async recordAndTranscribe(durationMs: number = 5000): Promise<any> {
    if (!this.model) {
      throw new Error('Model not initialized. Call createModel() first.');
    }
    
    const audioData = await this.audioProcessor.recordAudio(durationMs);
    return this.model.transcribe(audioData);
  }
  
  /**
   * Translate audio file (from non-English to English)
   * @param audioFile Audio file to translate
   * @returns Translation result
   */
  async translateFile(audioFile: File): Promise<any> {
    if (!this.model) {
      throw new Error('Model not initialized. Call createModel() first.');
    }
    
    const audioData = await this.audioProcessor.processAudioFile(audioFile);
    return this.model.translate(audioData);
  }
  
  /**
   * Translate audio data (from non-English to English)
   * @param audioData Audio data to translate
   * @returns Translation result
   */
  async translateAudio(audioData: Float32Array): Promise<any> {
    if (!this.model) {
      throw new Error('Model not initialized. Call createModel() first.');
    }
    
    return this.model.translate(audioData);
  }
  
  /**
   * Record and translate audio
   * @param durationMs Duration in milliseconds
   * @returns Translation result
   */
  async recordAndTranslate(durationMs: number = 5000): Promise<any> {
    if (!this.model) {
      throw new Error('Model not initialized. Call createModel() first.');
    }
    
    const audioData = await this.audioProcessor.recordAudio(durationMs);
    return this.model.translate(audioData);
  }
  
  /**
   * Compare performance across all available backends
   * @param audioFile Audio file to use for testing
   * @returns Performance comparison results
   */
  async compareBackends(audioFile: File): Promise<Record<string, number>> {
    if (!this.model) {
      throw new Error('Model not initialized. Call createModel() first.');
    }
    
    const audioData = await this.audioProcessor.processAudioFile(audioFile);
    return this.model.compareBackends(audioData);
  }
  
  /**
   * Get performance metrics
   * @returns Performance metrics
   */
  getPerformanceMetrics(): any {
    if (!this.model) {
      throw new Error('Model not initialized. Call createModel() first.');
    }
    
    return this.model.getPerformanceMetrics();
  }
  
  /**
   * Get backend metrics
   * @returns Backend metrics
   */
  getBackendMetrics(): any {
    if (!this.model) {
      throw new Error('Model not initialized. Call createModel() first.');
    }
    
    return this.model.getBackendMetrics();
  }
  
  /**
   * Get model info
   * @returns Model info
   */
  getModelInfo(): any {
    if (!this.model) {
      throw new Error('Model not initialized. Call createModel() first.');
    }
    
    return this.model.getModelInfo();
  }
  
  /**
   * Get shared tensor (for multimodal integration)
   * @param outputType Output type (default: 'audio_embedding')
   * @returns Shared tensor
   */
  getSharedTensor(outputType: string = 'audio_embedding'): any {
    if (!this.model) {
      throw new Error('Model not initialized. Call createModel() first.');
    }
    
    return this.model.getSharedTensor(outputType);
  }
  
  /**
   * Clear audio feature cache
   */
  clearFeatureCache(): void {
    if (!this.model) {
      throw new Error('Model not initialized. Call createModel() first.');
    }
    
    this.model.clearFeatureCache();
  }
  
  /**
   * Release all resources
   */
  async dispose(): Promise<void> {
    if (this.model) {
      await this.model.dispose();
      this.model = null;
    }
  }
}

/**
 * Multimodal integration example with Whisper and BERT
 */
export class MultimodalManager {
  private whisperManager: WhisperManager;
  private bertModel: any = null;
  private storageManager: StorageManager;
  private initialized: boolean = false;
  
  constructor(whisperManager: WhisperManager) {
    this.whisperManager = whisperManager;
  }
  
  /**
   * Initialize the multimodal manager
   */
  async initialize(): Promise<void> {
    // Initialize storage manager if not already done
    if (!this.initialized) {
      this.storageManager = new IndexedDBStorageManager();
      await this.storageManager.initialize();
      this.initialized = true;
    }
  }
  
  /**
   * Initialize BERT model
   * @param config BERT configuration
   */
  async initializeBERT(config: any = {}): Promise<any> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    // Default config
    const bertConfig = {
      modelId: 'bert-base-uncased',
      taskType: 'embedding',
      ...config
    };
    
    // Create and initialize the BERT model
    this.bertModel = createHardwareAbstractedBERT(bertConfig, this.storageManager);
    await this.bertModel.initialize();
    
    return {
      backendInfo: this.bertModel.getBackendMetrics(),
      modelInfo: this.bertModel.getModelInfo()
    };
  }
  
  /**
   * Run multimodal analysis on audio file
   * @param audioFile Audio file to analyze
   * @returns Multimodal analysis result
   */
  async analyzeAudioFile(audioFile: File): Promise<any> {
    if (!this.bertModel) {
      throw new Error('BERT model not initialized. Call initializeBERT() first.');
    }
    
    // Transcribe audio with Whisper
    const transcriptionResult = await this.whisperManager.transcribeFile(audioFile);
    const text = transcriptionResult.text;
    
    // Get audio embedding
    const audioEmbedding = this.whisperManager.getSharedTensor('audio_embedding');
    
    // Process text with BERT
    const textEmbedding = await this.bertModel.predict(text);
    
    return {
      text,
      audioEmbedding,
      textEmbedding,
      // In a real app, you would pass these embeddings to a classifier
    };
  }
  
  /**
   * Process audio and text with both models
   * @param audioData Audio data
   * @param text Optional text to use instead of transcribing audio
   * @returns Processing results
   */
  async processMultimodal(audioData: Float32Array, text?: string): Promise<any> {
    if (!this.bertModel) {
      throw new Error('BERT model not initialized. Call initializeBERT() first.');
    }
    
    // Transcribe audio with Whisper if text not provided
    let transcribedText = text;
    if (!transcribedText) {
      const transcriptionResult = await this.whisperManager.transcribeAudio(audioData);
      transcribedText = transcriptionResult.text;
    }
    
    // Get audio embedding
    const audioEmbedding = this.whisperManager.getSharedTensor('audio_embedding');
    
    // Process text with BERT
    const textEmbedding = await this.bertModel.predict(transcribedText);
    
    return {
      text: transcribedText,
      audioEmbedding,
      textEmbedding
    };
  }
  
  /**
   * Release all resources
   */
  async dispose(): Promise<void> {
    if (this.bertModel) {
      await this.bertModel.dispose();
      this.bertModel = null;
    }
  }
}

/**
 * Example usage
 */
async function exampleUsage(): Promise<void> {
  // Initialize Whisper manager
  const whisperManager = new WhisperManager();
  await whisperManager.initialize();
  
  // Create Whisper model with custom options
  await whisperManager.createModel('tiny', {
    backendPreference: ['webgpu', 'webnn', 'cpu'],
    browserOptimizations: true,
    audioProcessing: {
      hardwareAccelerated: true,
      cacheFeatures: true
    },
    quantization: {
      enabled: true,
      bits: 8
    }
  });
  
  console.log('Whisper model info:', whisperManager.getModelInfo());
  console.log('Whisper backend info:', whisperManager.getBackendMetrics());
  
  // Record and transcribe audio
  try {
    console.log('Recording 5 seconds of audio...');
    const result = await whisperManager.recordAndTranscribe(5000);
    console.log('Transcription:', result.text);
    
    // Show performance metrics
    console.log('Performance metrics:', whisperManager.getPerformanceMetrics());
    
    // Initialize multimodal integration
    const multimodalManager = new MultimodalManager(whisperManager);
    await multimodalManager.initializeBERT();
    
    // Process the same audio with both models
    const audioData = await new AudioProcessor().recordAudio(5000);
    const multimodalResult = await multimodalManager.processMultimodal(audioData);
    
    console.log('Multimodal analysis:', multimodalResult);
    
    // Clean up
    await multimodalManager.dispose();
    await whisperManager.dispose();
  } catch (error) {
    console.error('Error:', error);
  }
}

// Export all for reuse
export { 
  WhisperManager, 
  MultimodalManager, 
  AudioProcessor,
  exampleUsage
};