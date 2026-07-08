/**
 * Example of using Vision Transformer (ViT) with browser-specific optimizations
 */

import { WebGPUBackend } from '../../../src/hardware/webgpu/backend';
import { ViT, ViTConfig } from '../../../src/model/vision/vit';
import { BrowserType, detectBrowserType } from '../../../src/hardware/webgpu/browser_optimized_operations';

/**
 * Run ViT model with browser-specific optimizations
 */
async function runBrowserOptimizedViT() {
  try {
    // Initialize WebGPU backend
    console.log('Initializing WebGPU backend...');
    const backend = new WebGPUBackend();
    await backend.initialize();
    
    // Detect browser type
    const browserType = detectBrowserType();
    const browserNames = {
      [BrowserType.CHROME]: 'Google Chrome',
      [BrowserType.FIREFOX]: 'Mozilla Firefox',
      [BrowserType.SAFARI]: 'Apple Safari',
      [BrowserType.EDGE]: 'Microsoft Edge'
    };
    console.log(`Detected browser: ${browserNames[browserType] || 'Unknown'}`);
    
    // Configure ViT with browser-specific optimizations
    const config: ViTConfig = {
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
      backendPreference: ['webgpu', 'webnn', 'cpu'],
      useOptimizedOps: true,
      useBrowserOptimizations: true,  // Enable browser-specific optimizations
      browserType: browserType,       // Use detected browser type
      useOperationFusion: true,       // Enable operation fusion for better performance
      attentionDropout: 0.0
    };
    
    console.log('Creating ViT model with the following configuration:');
    console.log(JSON.stringify(config, null, 2));
    
    // Create ViT model
    const vit = new ViT(backend, config);
    
    // Initialize model (load weights)
    console.log('Initializing ViT model...');
    await vit.initialize();
    
    // Create a sample 224x224 RGB image (random data)
    const imageSize = 224;
    const imageData = new Float32Array(imageSize * imageSize * 3);
    
    // Fill with random pixel values (0-1 range)
    for (let i = 0; i < imageData.length; i++) {
      imageData[i] = Math.random();
    }
    
    // Run inference with browser-specific optimizations
    console.log('Running inference on sample image with browser-specific optimizations...');
    const startTime = performance.now();
    
    const output = await vit.process({
      imageData,
      width: imageSize,
      height: imageSize,
      isPreprocessed: true
    });
    
    const endTime = performance.now();
    
    console.log(`Inference completed in ${(endTime - startTime).toFixed(2)} ms`);
    console.log(`Model: ${output.model}`);
    console.log(`Backend: ${output.backend}`);
    
    // Display top 5 class probabilities
    console.log('Top 5 predicted classes:');
    const topClasses = getTopK(output.probabilities, 5);
    
    for (const [idx, prob] of topClasses) {
      console.log(`Class ${idx}: ${(prob * 100).toFixed(2)}%`);
    }
    
    // Cleanup
    await vit.dispose();
    await backend.dispose();
    
  } catch (error) {
    console.error('Error:', error);
  }
}

/**
 * Get top K indices and values from an array
 * @param arr Input array
 * @param k Number of top elements to return
 * @returns Array of [index, value] pairs sorted by value in descending order
 */
function getTopK(arr: number[], k: number): [number, number][] {
  return arr
    .map((value, index) => [index, value] as [number, number])
    .sort((a, b) => b[1] - a[1])
    .slice(0, k);
}

// Run the example
runBrowserOptimizedViT();

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