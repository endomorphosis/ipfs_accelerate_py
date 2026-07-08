/**
 * Hardware Abstraction Layer Multimodal Example
 * 
 * This example demonstrates how to use the Hardware Abstraction Layer (HAL)
 * with multiple hardware-abstracted models (BERT, ViT, CLIP, Whisper) together, 
 * showcasing cross-model tensor sharing for efficient multimodal processing.
 */

import { 
  createHardwareAbstractedModel, 
  StorageManager,
  HardwareAbstractedBERT,
  HardwareAbstractedWhisper,
  HardwareAbstractedCLIP,
  HardwareAbstractedViT,
  createHardwareAbstractedViT
} from '../src/model/hardware';

import { createHardwareAbstractionLayer, HardwareAbstractionLayer } from '../src/hardware/hardware_abstraction_layer';
import { createOptimalBackend, createMultiBackend } from '../src/hardware';
import { SharedTensor } from '../src/tensor/shared_tensor';

// Simple in-memory implementation of StorageManager for demonstration
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
 * Set up the Hardware Abstraction Layer and create all the models
 */
async function setupMultimodalSystem() {
  console.log('Setting up Hardware Abstraction Layer and multimodal models...');
  
  // Create storage manager
  const storageManager = new InMemoryStorageManager();
  await storageManager.initialize();
  
  // Initialize Hardware Abstraction Layer
  console.log('Initializing Hardware Abstraction Layer...');
  
  // Create a backend with multi-backend support
  const multiBackend = await createMultiBackend(['webgpu', 'webnn', 'cpu']);
  
  const hal = createHardwareAbstractionLayer({
    backends: [multiBackend],
    autoInitialize: true,
    useBrowserOptimizations: true,
    enableTensorSharing: true,  // Important for cross-model sharing
    enableOperationFusion: true
  });
  
  // Initialize backends
  await hal.initialize();
  console.log(`Hardware Abstraction Layer initialized with ${hal.getBackendType()} backend`);
  
  // Create hardware abstracted models
  console.log('Creating hardware abstracted models...');
  
  // BERT Model for text understanding
  const bert = createHardwareAbstractedModel('bert', {
    modelId: 'bert-base-uncased',
    backendPreference: ['webgpu', 'webnn', 'cpu'],
    allowFallback: true,
    taskType: 'embedding',
    enableTensorSharing: true
  }, storageManager) as HardwareAbstractedBERT;
  
  // Vision Transformer (ViT) for image understanding
  const vit = createHardwareAbstractedViT(hal, {
    modelId: 'vit-base-patch16-224',
    enableTensorSharing: true,
    prioritizeSpeed: true
  });
  
  // Whisper Model for audio processing
  const whisper = createHardwareAbstractedModel('whisper', {
    modelId: 'openai/whisper-tiny',
    backendPreference: ['webgpu', 'webnn', 'cpu'],
    allowFallback: true,
    taskType: 'transcription',
    enableTensorSharing: true
  }, storageManager) as HardwareAbstractedWhisper;
  
  // CLIP Model for multimodal understanding
  const clip = createHardwareAbstractedModel('clip', {
    modelId: 'openai/clip-vit-base-patch32',
    backendPreference: ['webgpu', 'webnn', 'cpu'],
    allowFallback: true,
    taskType: 'similarity',
    enableTensorSharing: true,
    useTextEmbedding: true,
    useImageEmbedding: true
  }, storageManager) as HardwareAbstractedCLIP;
  
  // Initialize models
  console.log('Initializing models (this may take a moment)...');
  try {
    await Promise.all([
      bert.initialize(),
      vit.initialize(),
      whisper.initialize(),
      clip.initialize()
    ]);
    console.log('All models initialized successfully');
  } catch (error) {
    console.error('Error initializing models:', error);
    throw error;
  }
  
  return { hal, bert, vit, whisper, clip, storageManager };
}

/**
 * Process a text input with BERT and use the shared embeddings
 */
async function processTextWithBERT(bert: HardwareAbstractedBERT): Promise<SharedTensor | null> {
  console.log('\nProcessing text with BERT...');
  
  const textInput = "A photograph of a cat sitting on a windowsill watching birds";
  console.log(`Text input: "${textInput}"`);
  
  try {
    const startTime = performance.now();
    const bertResult = await bert.predict(textInput);
    const processTime = performance.now() - startTime;
    
    console.log(`BERT processing complete in ${processTime.toFixed(2)}ms`);
    console.log(`Backend used: ${bert.getBackendMetrics().type}`);
    
    // Get shared tensor from BERT for later use with other models
    const bertSharedTensor = bert.getSharedTensor('text_embedding');
    console.log(`BERT shared tensor available: ${bertSharedTensor ? 'Yes' : 'No'}`);
    
    if (bertSharedTensor) {
      console.log(`BERT embedding shape: ${bertSharedTensor.tensor.dimensions.join(' × ')}`);
      return bertSharedTensor;
    }
    
    return null;
  } catch (error) {
    console.error('Error processing text with BERT:', error);
    return null;
  }
}

/**
 * Process an image with ViT and use the shared embeddings
 */
async function processImageWithViT(vit: HardwareAbstractedViT): Promise<SharedTensor | null> {
  console.log('\nProcessing image with ViT...');
  
  // Create a simulated image (in a real app, this would be actual image data)
  const imageSize = 224;
  const channels = 3;
  const imageData = new Float32Array(imageSize * imageSize * channels);
  
  // Simulate some image data (random pattern)
  for (let i = 0; i < imageData.length; i++) {
    imageData[i] = Math.random();
  }
  
  console.log(`Image input: ${imageSize}×${imageSize} RGB image`);
  
  try {
    const startTime = performance.now();
    const vitResult = await vit.process({
      imageData,
      width: imageSize,
      height: imageSize,
      isPreprocessed: true
    });
    const processTime = performance.now() - startTime;
    
    console.log(`ViT processing complete in ${processTime.toFixed(2)}ms`);
    console.log(`Backend used: ${vitResult.backend}`);
    console.log(`Top predicted class ID: ${vitResult.classId}`);
    
    // Get shared tensor from ViT for later use with other models
    const vitSharedTensor = vit.getSharedTensor('vision_embedding');
    console.log(`ViT shared tensor available: ${vitSharedTensor ? 'Yes' : 'No'}`);
    
    if (vitSharedTensor) {
      console.log(`ViT embedding shape: ${vitSharedTensor.tensor.dimensions.join(' × ')}`);
      return vitSharedTensor;
    }
    
    return null;
  } catch (error) {
    console.error('Error processing image with ViT:', error);
    return null;
  }
}

/**
 * Process audio with Whisper and use the shared embeddings
 */
async function processAudioWithWhisper(whisper: HardwareAbstractedWhisper): Promise<SharedTensor | null> {
  console.log('\nProcessing audio with Whisper...');
  
  // Create a simulated audio waveform (in a real app, this would be actual audio data)
  const audioSamples = 16000; // 1 second of audio at 16kHz
  const audioData = new Float32Array(audioSamples);
  
  // Simulate some audio data (simple sine wave)
  for (let i = 0; i < audioData.length; i++) {
    audioData[i] = Math.sin(i * 0.01);
  }
  
  console.log(`Audio input: ${audioSamples} samples (${audioSamples/16000}s at 16kHz)`);
  
  try {
    const startTime = performance.now();
    // In a real implementation, this would process the audio data
    const whisperResult = await whisper.transcribe(audioData);
    const processTime = performance.now() - startTime;
    
    console.log(`Whisper processing complete in ${processTime.toFixed(2)}ms`);
    console.log(`Backend used: ${whisper.getBackendMetrics().type}`);
    
    // Get shared tensor from Whisper for later use with other models
    const whisperSharedTensor = whisper.getSharedTensor('audio_embedding');
    console.log(`Whisper shared tensor available: ${whisperSharedTensor ? 'Yes' : 'No'}`);
    
    if (whisperSharedTensor) {
      console.log(`Whisper embedding shape: ${whisperSharedTensor.tensor.dimensions.join(' × ')}`);
      return whisperSharedTensor;
    }
    
    return null;
  } catch (error) {
    console.error('Error processing audio with Whisper:', error);
    return null;
  }
}

/**
 * Use CLIP with shared tensors from other models
 */
async function processWithCLIP(
  clip: HardwareAbstractedCLIP, 
  textEmbedding: SharedTensor | null, 
  imageEmbedding: SharedTensor | null
) {
  console.log('\nProcessing with CLIP...');
  
  if (!textEmbedding || !imageEmbedding) {
    console.log('Shared embeddings not available, CLIP will process from scratch...');
    
    // Create a simulated image (in a real app, this would be actual image data)
    const imageSize = 224;
    const channels = 3;
    const imageData = new Float32Array(imageSize * imageSize * channels);
    
    // Simulate some image data
    for (let i = 0; i < imageData.length; i++) {
      imageData[i] = Math.random();
    }
    
    const textInput = "A photograph of a cat sitting on a windowsill watching birds";
    
    const startTime = performance.now();
    const clipResult = await clip.processTextAndImage(textInput, {
      imageData,
      width: imageSize,
      height: imageSize
    });
    const processTime = performance.now() - startTime;
    
    console.log(`CLIP processing from scratch complete in ${processTime.toFixed(2)}ms`);
    console.log(`Text-image similarity score: ${clipResult.similarity.toFixed(4)}`);
    console.log(`Backend used: ${clip.getBackendMetrics().type}`);
    
    return clipResult;
  } else {
    console.log('Using shared embeddings with CLIP (optimized path)...');
    
    const startTime = performance.now();
    const clipResult = await clip.processWithSharedEmbeddings(textEmbedding, imageEmbedding);
    const processTime = performance.now() - startTime;
    
    console.log(`CLIP processing with shared tensors complete in ${processTime.toFixed(2)}ms`);
    console.log(`Text-image similarity score: ${clipResult.similarity.toFixed(4)}`);
    console.log(`Backend used: ${clip.getBackendMetrics().type}`);
    
    return clipResult;
  }
}

/**
 * Benchmark the performance difference between shared and non-shared tensor approaches
 */
async function benchmarkTensorSharing(
  bert: HardwareAbstractedBERT,
  vit: HardwareAbstractedViT,
  clip: HardwareAbstractedCLIP
) {
  console.log('\nBenchmarking tensor sharing performance...');
  
  const textInput = "A photograph of a majestic mountain landscape";
  
  // Create a consistent test image
  const imageSize = 224;
  const channels = 3;
  const imageData = new Float32Array(imageSize * imageSize * channels).fill(0.5);
  
  // Warm-up runs to ensure fairness
  await bert.predict(textInput);
  await vit.process({
    imageData,
    width: imageSize,
    height: imageSize,
    isPreprocessed: true
  });
  
  // Benchmark without tensor sharing
  console.log('\nRunning benchmark without tensor sharing...');
  const startWithout = performance.now();
  
  // Run BERT from scratch
  await bert.predict(textInput);
  
  // Run ViT from scratch
  await vit.process({
    imageData,
    width: imageSize,
    height: imageSize,
    isPreprocessed: true
  });
  
  // Run CLIP from scratch
  await clip.processTextAndImage(textInput, {
    imageData,
    width: imageSize,
    height: imageSize
  });
  
  const timeWithout = performance.now() - startWithout;
  console.log(`Without tensor sharing: ${timeWithout.toFixed(2)}ms`);
  
  // Benchmark with tensor sharing
  console.log('\nRunning benchmark with tensor sharing...');
  const startWith = performance.now();
  
  // Run BERT and get its embedding
  await bert.predict(textInput);
  const textEmbedding = bert.getSharedTensor('text_embedding');
  
  // Run ViT and get its embedding
  await vit.process({
    imageData,
    width: imageSize,
    height: imageSize,
    isPreprocessed: true
  });
  const imageEmbedding = vit.getSharedTensor('vision_embedding');
  
  // Use CLIP with shared embeddings
  if (textEmbedding && imageEmbedding) {
    await clip.processWithSharedEmbeddings(textEmbedding, imageEmbedding);
  }
  
  const timeWith = performance.now() - startWith;
  console.log(`With tensor sharing: ${timeWith.toFixed(2)}ms`);
  
  // Calculate improvement
  const improvement = ((timeWithout - timeWith) / timeWithout) * 100;
  console.log(`Performance improvement: ${improvement.toFixed(1)}%`);
  
  return {
    withoutSharing: timeWithout,
    withSharing: timeWith,
    improvement: improvement
  };
}

/**
 * Main demo function
 */
async function runHardwareAbstractionDemo() {
  console.log('==================================================');
  console.log('Hardware Abstraction Layer Multimodal Demo');
  console.log('==================================================');
  
  try {
    // Set up HAL and models
    const { hal, bert, vit, whisper, clip } = await setupMultimodalSystem();
    
    // Print model information
    console.log('\nModel Information:');
    console.log('BERT:', bert.getModelInfo());
    console.log('ViT Model ID:', 'vit-base-patch16-224');
    console.log('Whisper:', whisper.getModelInfo());
    console.log('CLIP:', clip.getModelInfo());
    
    // Process text with BERT and get embedding
    const textEmbedding = await processTextWithBERT(bert);
    
    // Process image with ViT and get embedding
    const imageEmbedding = await processImageWithViT(vit);
    
    // Process audio with Whisper and get embedding
    const audioEmbedding = await processAudioWithWhisper(whisper);
    
    // Use CLIP with shared tensors for optimized multimodal processing
    await processWithCLIP(clip, textEmbedding, imageEmbedding);
    
    // Benchmark tensor sharing performance
    await benchmarkTensorSharing(bert, vit, clip);
    
    // Compare backend performance
    console.log('\nComparing backend performance for BERT...');
    try {
      const testInput = "This is a test input for performance comparison.";
      const bertPerformance = await bert.compareBackends(testInput);
      console.log('BERT performance across backends:', bertPerformance);
    } catch (error) {
      console.log('Backend comparison not available in demo mode');
    }
    
    // Get performance metrics
    console.log('\nPerformance Metrics:');
    console.log('BERT Metrics:', JSON.stringify(bert.getPerformanceMetrics(), null, 2));
    
    // Clean up resources
    console.log('\nCleaning up resources...');
    await Promise.all([
      bert.dispose(),
      vit.dispose(),
      whisper.dispose(),
      clip.dispose()
    ]);
    
    hal.dispose();
    
    console.log('\nDemo completed successfully');
  } catch (error) {
    console.error('Error in hardware abstraction demo:', error);
  }
}

// Run the demo when this file is executed
if (typeof require !== 'undefined' && require.main === module) {
  runHardwareAbstractionDemo();
}