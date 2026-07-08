/**
 * Example of using Hardware Abstracted BERT and Whisper models together
 * Demonstrates cross-model tensor sharing and multimodal operation
 */

import { HardwareAbstractionLayer, createHardwareAbstractionLayer } from '../../../src/hardware/hardware_abstraction_layer';
import { WebGPUBackend } from '../../../src/hardware/webgpu/backend';
import { WebNNBackend } from '../../../src/hardware/webnn/backend';
import { CPUBackend } from '../../../src/hardware/cpu/backend';
import { HardwareAbstractedBERT, createHardwareAbstractedBERT, StorageManager } from '../../../src/model/hardware/bert';
import { HardwareAbstractedWhisper, createHardwareAbstractedWhisper } from '../../../src/model/hardware/whisper';

/**
 * Simple in-memory storage manager implementation
 */
class InMemoryStorageManager implements StorageManager {
  private storage: Map<string, any> = new Map();
  
  async initialize(): Promise<void> {
    console.log('Initializing in-memory storage manager');
  }
  
  async getItem(key: string): Promise<any> {
    return this.storage.get(key);
  }
  
  async setItem(key: string, value: any): Promise<void> {
    this.storage.set(key, value);
  }
  
  async hasItem(key: string): Promise<boolean> {
    return this.storage.has(key);
  }
  
  async removeItem(key: string): Promise<void> {
    this.storage.delete(key);
  }
}

/**
 * Run multimodal example with BERT and Whisper
 */
async function runMultimodalExample() {
  try {
    // Initialize hardware and storage
    console.log('Setting up hardware and storage...');
    
    // Create HAL with all available backends
    const hal = await initializeHardware();
    
    // Create storage manager
    const storageManager = new InMemoryStorageManager();
    await storageManager.initialize();
    
    // Initialize BERT model
    console.log('\nInitializing BERT model...');
    const bertModel = createHardwareAbstractedBERT({
      modelId: 'bert-base-uncased',
      taskType: 'embedding',
      allowFallback: true,
      collectMetrics: true,
      browserOptimizations: true
    }, storageManager);
    
    await bertModel.initialize();
    console.log('BERT model initialized');
    console.log('BERT backend:', bertModel.getBackendMetrics().type);
    
    // Initialize Whisper model
    console.log('\nInitializing Whisper model...');
    const whisperModel = createHardwareAbstractedWhisper({
      modelId: 'openai/whisper-tiny',
      taskType: 'transcription',
      allowFallback: true,
      collectMetrics: true,
      browserOptimizations: true,
      audioProcessing: {
        hardwareAccelerated: true,
        cacheFeatures: true
      }
    }, storageManager);
    
    await whisperModel.initialize();
    console.log('Whisper model initialized');
    console.log('Whisper backend:', whisperModel.getBackendMetrics().type);
    
    // 1. First use case: Speech to text then semantic understanding
    console.log('\n=== Speech to Text with Semantic Analysis ===');
    
    // Create example audio data (in a real implementation, this would be from a microphone)
    const audioData = createExampleAudioData();
    
    // Transcribe audio using Whisper
    console.log('Transcribing audio with Whisper...');
    const transcriptionResult = await whisperModel.transcribe(audioData);
    const transcribedText = transcriptionResult.text;
    console.log(`Transcribed text: "${transcribedText}"`);
    
    // Now analyze the transcribed text using BERT
    console.log('Analyzing text with BERT...');
    const embeddingResult = await bertModel.predict(transcribedText);
    console.log('Generated BERT embeddings for transcribed text');
    console.log(`Embedding dimension: ${embeddingResult.length}`);
    
    // 2. Second use case: Combining audio and text embeddings
    console.log('\n=== Multimodal Embeddings (Audio + Text) ===');
    
    // Get shared tensors
    const audioEmbedding = whisperModel.getSharedTensor('audio_embedding');
    const textEmbedding = bertModel.getSharedTensor('text_embedding');
    
    console.log('Retrieved shared tensors:');
    console.log('- Audio embedding:', audioEmbedding ? 'Available' : 'Not available');
    console.log('- Text embedding:', textEmbedding ? 'Available' : 'Not available');
    
    if (audioEmbedding && textEmbedding) {
      // In a real implementation, these embeddings would be combined in a multimodal model
      console.log('Combining audio and text embeddings for multimodal understanding...');
      
      // Simulate a multimodal fusion operation
      const multimodalResult = await simulateMultimodalFusion(
        audioEmbedding, 
        textEmbedding,
        hal
      );
      
      console.log('Multimodal fusion completed');
      console.log(`Fused representation dimension: ${multimodalResult.dimensions[1]}`);
      
      // Release the multimodal tensor
      hal.releaseTensor(multimodalResult);
    } else {
      console.log('Shared tensors not available for multimodal fusion');
    }
    
    // Display performance metrics
    console.log('\n=== Performance Metrics ===');
    console.log('BERT metrics:');
    const bertMetrics = bertModel.getPerformanceMetrics();
    Object.entries(bertMetrics).forEach(([name, metric]) => {
      console.log(`- ${name}: avg=${metric.avg.toFixed(2)}ms, min=${metric.min.toFixed(2)}ms, max=${metric.max.toFixed(2)}ms`);
    });
    
    console.log('\nWhisper metrics:');
    const whisperMetrics = whisperModel.getPerformanceMetrics();
    Object.entries(whisperMetrics).forEach(([name, metric]) => {
      console.log(`- ${name}: avg=${metric.avg.toFixed(2)}ms, min=${metric.min.toFixed(2)}ms, max=${metric.max.toFixed(2)}ms`);
    });
    
    // Cleanup resources
    console.log('\nCleaning up resources...');
    await bertModel.dispose();
    await whisperModel.dispose();
    await hal.dispose();
    
    console.log('\nExample completed successfully');
    
  } catch (error) {
    console.error('Error in multimodal example:', error);
  }
}

/**
 * Initialize HAL with all available backends
 */
async function initializeHardware(): Promise<HardwareAbstractionLayer> {
  try {
    // Create backend instances
    const webgpuBackend = new WebGPUBackend();
    const cpuBackend = new CPUBackend();
    
    // Try to create WebNN backend, but don't fail if not available
    let webnnBackend;
    try {
      webnnBackend = new WebNNBackend();
      await webnnBackend.initialize();
    } catch (e) {
      console.log('WebNN not available, continuing without it');
    }
    
    // Create HAL with available backends
    const backends = [
      webgpuBackend,
      ...(webnnBackend ? [webnnBackend] : []),
      cpuBackend
    ];
    
    console.log(`Initializing HAL with ${backends.length} available backends:`);
    backends.forEach(backend => console.log(`- ${backend.type}`));
    
    // Create HAL
    const hal = createHardwareAbstractionLayer({
      backends,
      useBrowserOptimizations: true,
      enableTensorSharing: true,
      enableOperationFusion: true,
      memoryCacheSize: 1024 * 1024 * 200 // 200 MB cache
    });
    
    // Initialize HAL
    await hal.initialize();
    
    return hal;
  } catch (error) {
    console.error('Error initializing hardware:', error);
    throw error;
  }
}

/**
 * Create example audio data (simulated)
 */
function createExampleAudioData(): Float32Array {
  // Create 2 seconds of simulated audio at 16kHz
  const sampleRate = 16000;
  const duration = 2; // seconds
  const numSamples = sampleRate * duration;
  
  // Create a sine wave at 440 Hz (A4 note)
  const audioData = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    const time = i / sampleRate;
    audioData[i] = Math.sin(2 * Math.PI * 440 * time);
  }
  
  return audioData;
}

/**
 * Simulate multimodal fusion of audio and text embeddings
 */
async function simulateMultimodalFusion(
  audioEmbedding: any, 
  textEmbedding: any,
  hal: HardwareAbstractionLayer
): Promise<any> {
  // In a real implementation, this would perform a sophisticated fusion
  // of the audio and text embeddings. Here we'll just concatenate them
  // along the feature dimension as a simple example.
  
  // Get the tensors from the shared tensors
  const audioTensor = audioEmbedding.getTensor();
  const textTensor = textEmbedding.getTensor();
  
  // Get dimensions
  const audioDim = audioTensor.dimensions[audioTensor.dimensions.length - 1];
  const textDim = textTensor.dimensions[textTensor.dimensions.length - 1];
  
  // Create a new tensor with concatenated dimension
  const fusedDim = audioDim + textDim;
  const fusedTensor = await hal.createTensor({
    dimensions: [1, fusedDim],
    data: new Float32Array(fusedDim).fill(0),
    dtype: 'float32'
  });
  
  // In a real implementation, you would perform actual tensor operations here
  // For demonstration purposes, we'll just return the empty tensor
  return fusedTensor;
}

// Run the example
runMultimodalExample();

// Add DOM output for browser environment
if (typeof document !== 'undefined') {
  const outputElement = document.getElementById('output');
  if (outputElement) {
    // Override console.log to also output to the DOM
    const originalLog = console.log;
    const originalError = console.error;
    
    console.log = function(...args) {
      originalLog.apply(console, args);
      outputElement.innerHTML += args.join(' ') + '<br>';
    };
    
    console.error = function(...args) {
      originalError.apply(console, args);
      outputElement.innerHTML += '<span style="color:red">' + args.join(' ') + '</span><br>';
    };
  }
}