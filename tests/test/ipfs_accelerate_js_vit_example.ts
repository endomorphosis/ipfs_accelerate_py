/**
 * Example demonstrating the browser-optimized WebGPU accelerated ViT model
 * 
 * This example shows how to load and run inference with the Vision Transformer (ViT)
 * model using browser-specific WebGPU optimizations for maximum performance.
 */

import { WebGPUTensorSharing } from './ipfs_accelerate_js_webgpu_tensor_sharing.ts';
import { StorageManager } from './ipfs_accelerate_js_storage_manager.ts';
import { WebGPUOptimizedViT, ViTConfig } from './ipfs_accelerate_js_vit_optimized.ts';
import { getOptimizedShader, BrowserCapabilities } from './ipfs_accelerate_js_browser_optimized_shaders.ts';
import { TensorView, TensorDimensions } from './ipfs_accelerate_js_tensor.ts';

/**
 * Set up the WebGPU device and capabilities
 */
async function setupWebGPU(): Promise<GPUDevice | null> {
  if (!navigator.gpu) {
    console.error('WebGPU not supported in this browser');
    return null;
  }
  
  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance'
    });
    
    if (!adapter) {
      console.error('No WebGPU adapter found');
      return null;
    }
    
    const device = await adapter.requestDevice({
      requiredFeatures: ['shader-f16'],
      requiredLimits: {
        maxBufferSize: 1 << 30, // 1GB
        maxStorageBufferBindingSize: 1 << 30, // 1GB
      }
    });
    
    return device;
  } catch (error) {
    console.error('Error setting up WebGPU:', error);
    return null;
  }
}

/**
 * Load an image from a URL and convert it to a tensor
 */
async function loadImageAsTensor(
  imageUrl: string, 
  imageSize: number, 
  tensorSharing: WebGPUTensorSharing
): Promise<TensorView> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = async () => {
      // Create a canvas to resize and process the image
      const canvas = document.createElement('canvas');
      canvas.width = imageSize;
      canvas.height = imageSize;
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        reject(new Error('Could not get canvas context'));
        return;
      }
      
      // Draw and resize the image
      ctx.drawImage(img, 0, 0, imageSize, imageSize);
      
      // Get pixel data
      const imageData = ctx.getImageData(0, 0, imageSize, imageSize);
      const pixels = imageData.data;
      
      // Convert RGBA to RGB float array with normalization
      const rgbData = new Float32Array(imageSize * imageSize * 3);
      for (let i = 0; i < pixels.length / 4; i++) {
        // Normalize to [0, 1] and apply preprocessing
        rgbData[i * 3] = (pixels[i * 4] / 255.0 - 0.485) / 0.229;     // R, normalized with ImageNet mean/std
        rgbData[i * 3 + 1] = (pixels[i * 4 + 1] / 255.0 - 0.456) / 0.224; // G, normalized with ImageNet mean/std
        rgbData[i * 3 + 2] = (pixels[i * 4 + 2] / 255.0 - 0.406) / 0.225; // B, normalized with ImageNet mean/std
      }
      
      // Create tensor from the RGB data
      const tensor = await tensorSharing.createTensorFromData(
        rgbData,
        [1, imageSize, imageSize, 3],
        'float32'
      );
      
      resolve(tensor);
    };
    
    img.onerror = () => {
      reject(new Error('Failed to load image'));
    };
    
    img.src = imageUrl;
  });
}

/**
 * Basic ViT example with browser-optimized WebGPU acceleration
 */
export async function runBasicViTExample(imageUrl: string): Promise<string[]> {
  // Set up WebGPU
  const device = await setupWebGPU();
  if (!device) {
    throw new Error('WebGPU not supported or failed to initialize');
  }
  
  console.log('WebGPU device initialized');
  
  // Initialize tensor sharing system
  const tensorSharing = new WebGPUTensorSharing();
  await tensorSharing.initialize(device);
  
  // Get browser capabilities for logging
  const capabilities = await tensorSharing.getBrowserCapabilities();
  console.log('Browser capabilities detected:', capabilities);
  
  // Initialize storage manager
  const storageManager = new StorageManager('vit-models');
  await storageManager.initialize();
  
  // Check if model is already in storage, otherwise fetch and store
  const modelId = 'vit-base-patch16-224';
  const modelExists = await storageManager.modelExists(modelId);
  
  if (!modelExists) {
    console.log('Model not found in storage, downloading...');
    // In a real application, you would fetch the model weights here
    // and store them using storageManager.storeModelWeights()
    
    // For this example, we'll assume the model is already in storage
    throw new Error('Model not found in storage. Please download and store the model first.');
  }
  
  // Create ViT model configuration
  const vitConfig: ViTConfig = {
    imageSize: 224,
    patchSize: 16,
    numLayers: 12,
    hiddenSize: 768,
    numHeads: 12,
    mlpDim: 3072,
    numClasses: 1000,
    quantization: {
      enabled: true,
      bits: 8,
      blockSize: 32
    },
    useOptimizedAttention: true,
    modelId: modelId
  };
  
  // Initialize ViT model
  console.log('Initializing ViT model with browser-optimized WebGPU acceleration');
  const model = new WebGPUOptimizedViT(vitConfig, tensorSharing, storageManager);
  await model.initialize();
  
  // Show model info
  const modelInfo = model.getModelInfo();
  console.log('Model info:', modelInfo);
  
  // Load and process input image
  console.log('Loading input image:', imageUrl);
  const inputTensor = await loadImageAsTensor(imageUrl, vitConfig.imageSize, tensorSharing);
  
  // Run inference
  console.log('Running inference with browser-optimized WebGPU acceleration');
  console.time('inference');
  const probabilities = await model.predict(inputTensor);
  console.timeEnd('inference');
  
  // Get top 5 predictions
  const indices = Array.from(Array(probabilities.length).keys());
  const sortedIndices = indices.sort((a, b) => probabilities[b] - probabilities[a]).slice(0, 5);
  
  // Map indices to labels (using a placeholder function here)
  const labels = await getImageNetLabels();
  const topPredictions = sortedIndices.map(idx => ({
    label: labels[idx],
    probability: probabilities[idx]
  }));
  
  console.log('Top 5 predictions:', topPredictions);
  
  // Clean up resources
  await model.dispose();
  await tensorSharing.dispose();
  
  // Return top prediction labels
  return topPredictions.map(p => p.label);
}

/**
 * Compare browser-optimized WebGPU implementation with standard implementation
 */
export async function runPerformanceComparison(imageUrl: string): Promise<{
  optimizedTime: number;
  standardTime: number;
  speedup: number;
  browserInfo: BrowserCapabilities;
}> {
  // Set up WebGPU
  const device = await setupWebGPU();
  if (!device) {
    throw new Error('WebGPU not supported or failed to initialize');
  }
  
  // Initialize tensor sharing system with browser optimizations
  const optimizedTensorSharing = new WebGPUTensorSharing();
  await optimizedTensorSharing.initialize(device, {
    enableOptimizations: true,
    precompileShaders: true,
    optimizationLevel: 'maximum'
  });
  
  // Initialize tensor sharing system without browser optimizations
  const standardTensorSharing = new WebGPUTensorSharing();
  await standardTensorSharing.initialize(device, {
    enableOptimizations: false,
    precompileShaders: false,
    optimizationLevel: 'minimum'
  });
  
  // Get browser capabilities for reporting
  const browserCapabilities = await optimizedTensorSharing.getBrowserCapabilities();
  
  // Initialize storage manager
  const storageManager = new StorageManager('vit-models');
  await storageManager.initialize();
  
  // Create ViT model configuration
  const vitConfig: ViTConfig = {
    imageSize: 224,
    patchSize: 16,
    numLayers: 12,
    hiddenSize: 768,
    numHeads: 12,
    mlpDim: 3072,
    numClasses: 1000,
    quantization: {
      enabled: true,
      bits: 8
    },
    useOptimizedAttention: true,
    modelId: 'vit-base-patch16-224'
  };
  
  // Initialize optimized ViT model
  const optimizedModel = new WebGPUOptimizedViT(
    vitConfig, 
    optimizedTensorSharing, 
    storageManager
  );
  await optimizedModel.initialize();
  
  // Initialize standard ViT model 
  const standardModel = new WebGPUOptimizedViT(
    { ...vitConfig, useOptimizedAttention: false },
    standardTensorSharing,
    storageManager
  );
  await standardModel.initialize();
  
  // Load and process input image
  const optimizedInputTensor = await loadImageAsTensor(
    imageUrl, 
    vitConfig.imageSize, 
    optimizedTensorSharing
  );
  
  const standardInputTensor = await loadImageAsTensor(
    imageUrl, 
    vitConfig.imageSize, 
    standardTensorSharing
  );
  
  // Warm-up runs
  console.log('Performing warm-up runs...');
  await optimizedModel.predict(optimizedInputTensor);
  await standardModel.predict(standardInputTensor);
  
  // Number of iterations for timing
  const iterations = 10;
  
  // Time optimized model
  console.log(`Running ${iterations} iterations with browser-optimized implementation...`);
  const optimizedStartTime = performance.now();
  for (let i = 0; i < iterations; i++) {
    await optimizedModel.predict(optimizedInputTensor);
  }
  const optimizedEndTime = performance.now();
  const optimizedTime = (optimizedEndTime - optimizedStartTime) / iterations;
  
  // Time standard model
  console.log(`Running ${iterations} iterations with standard implementation...`);
  const standardStartTime = performance.now();
  for (let i = 0; i < iterations; i++) {
    await standardModel.predict(standardInputTensor);
  }
  const standardEndTime = performance.now();
  const standardTime = (standardEndTime - standardStartTime) / iterations;
  
  // Calculate speedup
  const speedup = standardTime / optimizedTime;
  
  // Clean up resources
  await optimizedModel.dispose();
  await standardModel.dispose();
  await optimizedTensorSharing.dispose();
  await standardTensorSharing.dispose();
  
  // Return performance results
  return {
    optimizedTime,
    standardTime,
    speedup,
    browserInfo: browserCapabilities
  };
}

/**
 * Create an interactive demonstration of browser-optimized WebGPU ViT model
 */
export async function createInteractiveBrowserOptimizedDemo(
  containerElement: HTMLElement
): Promise<{ 
  runInference: (imageUrl: string) => Promise<string[]>;
  detectBrowserCapabilities: () => Promise<BrowserCapabilities | null>;
  benchmarkPerformance: () => Promise<void>;
}> {
  // Create UI components
  containerElement.innerHTML = `
    <div class="browser-optimized-demo">
      <h2>Browser-Optimized WebGPU ViT Demo</h2>
      
      <div class="browser-info-section">
        <h3>Browser Capabilities</h3>
        <div id="browser-capabilities">
          <button id="detect-capabilities-btn">Detect Browser Capabilities</button>
          <div id="capabilities-result" class="capabilities-result"></div>
        </div>
      </div>
      
      <div class="model-section">
        <h3>ViT Model with Browser-Optimized WebGPU</h3>
        <div class="input-section">
          <input type="text" id="image-url" placeholder="Enter image URL" 
                 value="https://storage.googleapis.com/ipfs_accelerate_example_data/cat.jpg" />
          <button id="run-inference-btn">Run Inference</button>
        </div>
        <div class="preview-section">
          <div class="image-preview">
            <h4>Preview</h4>
            <img id="preview-image" src="" alt="Preview" />
          </div>
          <div class="results-section">
            <h4>Results</h4>
            <div id="inference-results"></div>
            <div id="inference-time"></div>
          </div>
        </div>
      </div>
      
      <div class="benchmark-section">
        <h3>Performance Benchmark</h3>
        <button id="run-benchmark-btn">Run Performance Benchmark</button>
        <div id="benchmark-results"></div>
        <div id="benchmark-chart" class="benchmark-chart"></div>
      </div>
    </div>
    
    <style>
      .browser-optimized-demo {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
      }
      
      .browser-info-section, .model-section, .benchmark-section {
        background: #f5f5f5;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
      }
      
      .capabilities-result {
        margin-top: 10px;
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 14px;
        background: #222;
        color: #fff;
        padding: 15px;
        border-radius: 4px;
      }
      
      .input-section {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
      }
      
      #image-url {
        flex: 1;
        padding: 8px;
        border: 1px solid #ccc;
        border-radius: 4px;
      }
      
      button {
        background: #0078d7;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        font-weight: bold;
      }
      
      button:hover {
        background: #0062ab;
      }
      
      .preview-section {
        display: flex;
        gap: 20px;
      }
      
      .image-preview, .results-section {
        flex: 1;
      }
      
      #preview-image {
        max-width: 100%;
        border-radius: 4px;
        border: 1px solid #ddd;
      }
      
      #inference-results {
        font-family: monospace;
        background: #f8f8f8;
        padding: 15px;
        border-radius: 4px;
        border: 1px solid #ddd;
      }
      
      #inference-time {
        margin-top: 10px;
        font-weight: bold;
      }
      
      .benchmark-chart {
        height: 300px;
        margin-top: 20px;
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 4px;
      }
    </style>
  `;
  
  // Get UI elements
  const detectCapabilitiesBtn = containerElement.querySelector('#detect-capabilities-btn') as HTMLButtonElement;
  const capabilitiesResult = containerElement.querySelector('#capabilities-result') as HTMLDivElement;
  const imageUrlInput = containerElement.querySelector('#image-url') as HTMLInputElement;
  const runInferenceBtn = containerElement.querySelector('#run-inference-btn') as HTMLButtonElement;
  const previewImage = containerElement.querySelector('#preview-image') as HTMLImageElement;
  const inferenceResults = containerElement.querySelector('#inference-results') as HTMLDivElement;
  const inferenceTime = containerElement.querySelector('#inference-time') as HTMLDivElement;
  const runBenchmarkBtn = containerElement.querySelector('#run-benchmark-btn') as HTMLButtonElement;
  const benchmarkResults = containerElement.querySelector('#benchmark-results') as HTMLDivElement;
  const benchmarkChart = containerElement.querySelector('#benchmark-chart') as HTMLDivElement;
  
  // Create WebGPU device and tensor sharing system
  let device: GPUDevice | null = null;
  let tensorSharing: WebGPUTensorSharing | null = null;
  let storageManager: StorageManager | null = null;
  let vitModel: WebGPUOptimizedViT | null = null;
  
  /**
   * Initialize WebGPU and model resources
   */
  async function initializeResources(): Promise<boolean> {
    try {
      // Set up WebGPU if not already set up
      if (!device) {
        device = await setupWebGPU();
        if (!device) {
          throw new Error('WebGPU not supported or failed to initialize');
        }
      }
      
      // Initialize tensor sharing if not already initialized
      if (!tensorSharing) {
        tensorSharing = new WebGPUTensorSharing();
        await tensorSharing.initialize(device, {
          enableOptimizations: true,
          precompileShaders: true,
          optimizationLevel: 'maximum'
        });
      }
      
      // Initialize storage manager if not already initialized
      if (!storageManager) {
        storageManager = new StorageManager('vit-models');
        await storageManager.initialize();
      }
      
      return true;
    } catch (error) {
      console.error('Error initializing resources:', error);
      return false;
    }
  }
  
  /**
   * Detect browser capabilities
   */
  async function detectBrowserCapabilities(): Promise<BrowserCapabilities | null> {
    if (!await initializeResources()) {
      return null;
    }
    
    try {
      const capabilities = await tensorSharing!.getBrowserCapabilities();
      
      // Format capabilities for display
      const formattedCapabilities = JSON.stringify(capabilities, null, 2);
      capabilitiesResult.textContent = formattedCapabilities;
      
      return capabilities;
    } catch (error) {
      console.error('Error detecting capabilities:', error);
      capabilitiesResult.textContent = 'Error detecting capabilities: ' + error;
      return null;
    }
  }
  
  /**
   * Load ViT model
   */
  async function loadModel(): Promise<boolean> {
    try {
      // Check if model is already loaded
      if (vitModel) {
        return true;
      }
      
      // Ensure resources are initialized
      if (!await initializeResources()) {
        return false;
      }
      
      // Create ViT model configuration
      const vitConfig: ViTConfig = {
        imageSize: 224,
        patchSize: 16,
        numLayers: 12,
        hiddenSize: 768,
        numHeads: 12,
        mlpDim: 3072,
        numClasses: 1000,
        quantization: {
          enabled: true,
          bits: 8,
          blockSize: 32
        },
        useOptimizedAttention: true,
        modelId: 'vit-base-patch16-224'
      };
      
      // Initialize ViT model
      console.log('Initializing ViT model with browser-optimized WebGPU acceleration');
      vitModel = new WebGPUOptimizedViT(vitConfig, tensorSharing!, storageManager!);
      await vitModel.initialize();
      
      return true;
    } catch (error) {
      console.error('Error loading model:', error);
      inferenceResults.textContent = 'Error loading model: ' + error;
      return false;
    }
  }
  
  /**
   * Run inference on an image
   */
  async function runInference(imageUrl: string): Promise<string[]> {
    try {
      // Update preview image
      previewImage.src = imageUrl;
      inferenceResults.textContent = 'Running inference...';
      inferenceTime.textContent = '';
      
      // Ensure model is loaded
      if (!await loadModel()) {
        throw new Error('Failed to load model');
      }
      
      // Load and process input image
      const inputTensor = await loadImageAsTensor(imageUrl, 224, tensorSharing!);
      
      // Run inference
      console.log('Running inference with browser-optimized WebGPU acceleration');
      const startTime = performance.now();
      const probabilities = await vitModel!.predict(inputTensor);
      const endTime = performance.now();
      const inferenceTimeMs = endTime - startTime;
      
      // Get top 5 predictions
      const indices = Array.from(Array(probabilities.length).keys());
      const sortedIndices = indices
        .sort((a, b) => probabilities[b] - probabilities[a])
        .slice(0, 5);
      
      // Map indices to labels
      const labels = await getImageNetLabels();
      const topPredictions = sortedIndices.map(idx => ({
        label: labels[idx],
        probability: probabilities[idx]
      }));
      
      // Display results
      inferenceResults.innerHTML = topPredictions
        .map((pred, i) => `
          <div class="prediction">
            <strong>${i + 1}. ${pred.label}</strong>: ${(pred.probability * 100).toFixed(2)}%
          </div>
        `)
        .join('');
      
      inferenceTime.textContent = `Inference time: ${inferenceTimeMs.toFixed(2)}ms`;
      
      return topPredictions.map(p => p.label);
    } catch (error) {
      console.error('Error running inference:', error);
      inferenceResults.textContent = 'Error running inference: ' + error;
      return [];
    }
  }
  
  /**
   * Run benchmark comparing optimized vs standard implementation
   */
  async function benchmarkPerformance(): Promise<void> {
    try {
      benchmarkResults.textContent = 'Running benchmark...';
      benchmarkChart.innerHTML = '';
      
      // Ensure resources are initialized
      if (!await initializeResources()) {
        throw new Error('Failed to initialize resources');
      }
      
      // Get image URL from input
      const imageUrl = imageUrlInput.value.trim();
      if (!imageUrl) {
        throw new Error('Please enter an image URL');
      }
      
      // Run performance comparison
      const results = await runPerformanceComparison(imageUrl);
      
      // Display results
      benchmarkResults.innerHTML = `
        <div class="benchmark-summary">
          <div><strong>Browser:</strong> ${results.browserInfo.browserType} ${results.browserInfo.browserVersion || ''}</div>
          <div><strong>GPU:</strong> ${results.browserInfo.gpuVendor} ${results.browserInfo.gpuModel || ''}</div>
          <div><strong>Optimized time:</strong> ${results.optimizedTime.toFixed(2)}ms</div>
          <div><strong>Standard time:</strong> ${results.standardTime.toFixed(2)}ms</div>
          <div><strong>Speedup:</strong> ${results.speedup.toFixed(2)}x</div>
        </div>
      `;
      
      // Create a simple bar chart
      const maxTime = Math.max(results.optimizedTime, results.standardTime);
      benchmarkChart.innerHTML = `
        <div class="chart-container">
          <div class="chart-label">Implementation</div>
          <div class="chart-bars">
            <div class="chart-bar-group">
              <div class="chart-bar-label">Optimized</div>
              <div class="chart-bar optimized-bar" style="width: ${(results.optimizedTime / maxTime * 100).toFixed(2)}%">
                ${results.optimizedTime.toFixed(2)}ms
              </div>
            </div>
            <div class="chart-bar-group">
              <div class="chart-bar-label">Standard</div>
              <div class="chart-bar standard-bar" style="width: ${(results.standardTime / maxTime * 100).toFixed(2)}%">
                ${results.standardTime.toFixed(2)}ms
              </div>
            </div>
          </div>
        </div>
        <style>
          .chart-container {
            display: flex;
            height: 100%;
            padding: 20px;
          }
          .chart-label {
            writing-mode: vertical-lr;
            transform: rotate(180deg);
            padding-right: 10px;
            font-weight: bold;
          }
          .chart-bars {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            justify-content: space-around;
          }
          .chart-bar-group {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
          }
          .chart-bar-label {
            width: 100px;
            font-weight: bold;
          }
          .chart-bar {
            height: 40px;
            display: flex;
            align-items: center;
            padding: 0 10px;
            color: white;
            font-weight: bold;
            border-radius: 4px;
            transition: width 1s ease-in-out;
          }
          .optimized-bar {
            background: #0078d7;
          }
          .standard-bar {
            background: #666;
          }
        </style>
      `;
    } catch (error) {
      console.error('Error running benchmark:', error);
      benchmarkResults.textContent = 'Error running benchmark: ' + error;
    }
  }
  
  // Wire up event handlers
  detectCapabilitiesBtn.addEventListener('click', () => {
    detectBrowserCapabilities();
  });
  
  runInferenceBtn.addEventListener('click', () => {
    const imageUrl = imageUrlInput.value.trim();
    if (imageUrl) {
      runInference(imageUrl);
    } else {
      inferenceResults.textContent = 'Please enter an image URL';
    }
  });
  
  runBenchmarkBtn.addEventListener('click', () => {
    benchmarkPerformance();
  });
  
  // Return API for external use
  return {
    runInference,
    detectBrowserCapabilities,
    benchmarkPerformance
  };
}

/**
 * Get ImageNet labels (placeholder implementation)
 */
async function getImageNetLabels(): Promise<string[]> {
  // In a real application, you would load the actual ImageNet labels
  return [
    'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark',
    'electric ray', 'stingray', 'rooster', 'hen', 'ostrich', 'brambling',
    'goldfinch', 'house finch', 'junco', 'indigo bunting', 'American robin',
    'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel', 'kite', 'bald eagle',
    'vulture', 'great grey owl', 'fire salamander', 'smooth newt', 'newt',
    'spotted salamander', 'axolotl', 'American bullfrog', 'tree frog', 'tailed frog',
    'loggerhead sea turtle', 'leatherback sea turtle', 'mud turtle', 'terrapin',
    'box turtle', 'banded gecko', 'green iguana', 'Carolina anole',
    'desert grassland whiptail lizard', 'agama', 'frilled-necked lizard',
    'alligator lizard', 'Gila monster', 'European green lizard', 'chameleon',
    'Komodo dragon', 'Nile crocodile', 'American alligator', 'triceratops',
    'worm snake', 'ring-necked snake', 'eastern hog-nosed snake', 'smooth green snake',
    'kingsnake', 'garter snake', 'water snake', 'vine snake', 'night snake',
    'boa constrictor', 'African rock python', 'Indian cobra', 'green mamba',
    'sea snake', 'Saharan horned viper', 'eastern diamondback rattlesnake',
    'sidewinder rattlesnake', 'trilobite', 'harvestman', 'scorpion', 'yellow garden spider',
    'barn spider', 'European garden spider', 'southern black widow', 'tarantula',
    'wolf spider', 'tick', 'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse',
    'prairie grouse', 'peacock', 'quail', 'partridge', 'grey parrot', 'macaw',
    'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater', 'hornbill',
    'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted merganser', 'goose',
    'black swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala',
    'wombat', 'jellyfish', 'sea anemone', 'brain coral', 'flatworm', 'nematode',
    'conch', 'snail', 'slug', 'sea slug', 'chiton', 'chambered nautilus',
    // Add more labels as needed or load them dynamically
  ];
}

// Example usage in browser:
// 
// document.addEventListener('DOMContentLoaded', async () => {
//   const demoContainer = document.getElementById('vit-demo-container');
//   if (demoContainer) {
//     const demo = await createInteractiveBrowserOptimizedDemo(demoContainer);
//     // You can programmatically interact with the demo through the API:
//     // await demo.detectBrowserCapabilities();
//     // await demo.runInference('https://example.com/cat.jpg');
//     // await demo.benchmarkPerformance();
//   }
// });