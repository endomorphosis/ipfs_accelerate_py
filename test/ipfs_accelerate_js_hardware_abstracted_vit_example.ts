/**
 * Example demonstrating the Hardware Abstraction Layer (HAL) accelerated ViT model
 * 
 * This example shows how to use the Hardware Abstraction Layer to automatically
 * select the optimal backend (WebGPU, WebNN, CPU) for ViT inference based on
 * available hardware capabilities.
 */

import { StorageManager } from './ipfs_accelerate_js_storage_manager';
import { HardwareAbstractedVIT, ViTConfig } from './ipfs_accelerate_js_vit_hardware_abstraction';
import { createHardwareAbstraction } from './ipfs_accelerate_js_hardware_abstraction';

/**
 * Load an image from a URL
 */
function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
    img.src = url;
  });
}

/**
 * Basic HAL-accelerated ViT example
 */
export async function runHardwareAbstractedViTExample(imageUrl: string): Promise<string[]> {
  console.log('Initializing Hardware Abstraction Layer for ViT inference');
  
  // First, detect available hardware capabilities
  const hal = await createHardwareAbstraction();
  const capabilities = hal.getCapabilities();
  console.log('Detected hardware capabilities:', capabilities);
  console.log('Available backends:', hal.getAvailableBackends());
  
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
  
  // Initialize HAL-accelerated ViT model
  console.log('Initializing ViT model with Hardware Abstraction Layer');
  const model = new HardwareAbstractedVIT(vitConfig, storageManager);
  await model.initialize();
  
  // Show model info
  const modelInfo = model.getModelInfo();
  console.log('Model info:', modelInfo);
  
  // Load input image
  console.log('Loading input image:', imageUrl);
  const inputImage = await loadImage(imageUrl);
  
  // Run inference
  console.log('Running inference with HAL-accelerated ViT');
  console.time('total_inference');
  const probabilities = await model.predict(inputImage);
  console.timeEnd('total_inference');
  
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
  
  // Return top prediction labels
  return topPredictions.map(p => p.label);
}

/**
 * Compare performance across available backends
 */
export async function runCrossBackendPerformanceComparison(imageUrl: string): Promise<{
  results: Record<string, {
    inferenceTime: number;
    supportLevel: string;
    topPrediction: string;
  }>;
  bestBackend: string;
}> {
  // Initialize storage manager
  const storageManager = new StorageManager('vit-models');
  await storageManager.initialize();
  
  // Create basic ViT model configuration
  const baseConfig: ViTConfig = {
    imageSize: 224,
    patchSize: 16,
    numLayers: 12,
    hiddenSize: 768,
    numHeads: 12,
    mlpDim: 3072,
    numClasses: 1000,
    useOptimizedAttention: true,
    modelId: 'vit-base-patch16-224'
  };
  
  // Get available hardware capabilities using HAL
  const hal = await createHardwareAbstraction();
  const availableBackends = hal.getAvailableBackends();
  
  // Load test image
  const inputImage = await loadImage(imageUrl);
  
  // Store results for each backend
  const results: Record<string, {
    inferenceTime: number;
    supportLevel: string;
    topPrediction: string;
  }> = {};
  
  // Define backends to test (including HAL's auto-selection)
  const backendsToTest = [...availableBackends, 'hal-auto'];
  
  // Test each backend
  for (const backendType of backendsToTest) {
    try {
      console.log(`Testing backend: ${backendType}`);
      
      // Create model configuration
      const modelConfig: ViTConfig = {
        ...baseConfig,
        // Enable quantization for WebGPU and CPU but not for WebNN
        quantization: backendType === 'webnn' ? undefined : {
          enabled: true,
          bits: 8,
          blockSize: 32
        }
      };
      
      // Create model
      const model = new HardwareAbstractedVIT(modelConfig, storageManager);
      
      // Initialize model
      await model.initialize();
      
      // Perform warm-up run
      console.log(`Performing warm-up run on ${backendType}`);
      await model.predict(inputImage);
      
      // Perform timed inference
      console.log(`Running timed inference on ${backendType}`);
      const startTime = performance.now();
      const probabilities = await model.predict(inputImage);
      const endTime = performance.now();
      const inferenceTime = endTime - startTime;
      
      // Get top prediction
      const indices = Array.from(Array(probabilities.length).keys());
      const topIdx = indices.reduce((a, b) => probabilities[a] > probabilities[b] ? a : b);
      const labels = await getImageNetLabels();
      const topPrediction = labels[topIdx];
      
      // Record results
      results[backendType] = {
        inferenceTime,
        supportLevel: 'full',
        topPrediction
      };
      
      // Clean up
      await model.dispose();
      
    } catch (error) {
      console.error(`Error testing backend ${backendType}:`, error);
      results[backendType] = {
        inferenceTime: -1,
        supportLevel: 'unsupported',
        topPrediction: 'N/A'
      };
    }
  }
  
  // Determine best backend based on inference time
  const backendsWithValidTimes = Object.entries(results)
    .filter(([_, data]) => data.inferenceTime > 0)
    .sort(([_, dataA], [__, dataB]) => dataA.inferenceTime - dataB.inferenceTime);
  
  const bestBackend = backendsWithValidTimes.length > 0 ? backendsWithValidTimes[0][0] : 'none';
  
  console.log('Performance comparison results:', results);
  console.log(`Best backend: ${bestBackend}`);
  
  return {
    results,
    bestBackend
  };
}

/**
 * Create an interactive demonstration of Hardware Abstracted ViT model
 */
export async function createInteractiveHALDemo(
  containerElement: HTMLElement
): Promise<{
  runInference: (imageUrl: string) => Promise<string[]>;
  detectCapabilities: () => Promise<any>;
  compareBackends: () => Promise<void>;
}> {
  // Create UI components
  containerElement.innerHTML = `
    <div class="hal-demo">
      <h2>Hardware Abstraction Layer ViT Demo</h2>
      
      <div class="hardware-info-section">
        <h3>Hardware Capabilities</h3>
        <div id="hardware-capabilities">
          <button id="detect-capabilities-btn">Detect Hardware Capabilities</button>
          <div id="capabilities-result" class="capabilities-result"></div>
          <div id="backend-list" class="backend-list"></div>
        </div>
      </div>
      
      <div class="model-section">
        <h3>ViT Model with Hardware Abstraction Layer</h3>
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
            <div id="backend-used"></div>
          </div>
        </div>
      </div>
      
      <div class="benchmark-section">
        <h3>Backend Comparison</h3>
        <button id="run-comparison-btn">Compare Available Backends</button>
        <div id="comparison-results"></div>
        <div id="comparison-chart" class="comparison-chart"></div>
      </div>
    </div>
    
    <style>
      .hal-demo {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
      }
      
      .hardware-info-section, .model-section, .benchmark-section {
        background: #f5f5f5;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
      }
      
      .capabilities-result, .backend-list {
        margin-top: 10px;
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 14px;
        background: #222;
        color: #fff;
        padding: 15px;
        border-radius: 4px;
      }
      
      .backend-list {
        margin-top: 10px;
        background: #333;
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
      
      #inference-time, #backend-used {
        margin-top: 10px;
        font-weight: bold;
      }
      
      .comparison-chart {
        height: 300px;
        margin-top: 20px;
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 4px;
      }
      
      .backend-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        margin-right: 8px;
        margin-bottom: 8px;
        font-weight: bold;
      }
      
      .backend-badge.webgpu {
        background: #4285f4;
        color: white;
      }
      
      .backend-badge.webnn {
        background: #0f9d58;
        color: white;
      }
      
      .backend-badge.cpu {
        background: #db4437;
        color: white;
      }
      
      .backend-badge.auto {
        background: #f4b400;
        color: black;
      }
    </style>
  `;
  
  // Get UI elements
  const detectCapabilitiesBtn = containerElement.querySelector('#detect-capabilities-btn') as HTMLButtonElement;
  const capabilitiesResult = containerElement.querySelector('#capabilities-result') as HTMLDivElement;
  const backendList = containerElement.querySelector('#backend-list') as HTMLDivElement;
  const imageUrlInput = containerElement.querySelector('#image-url') as HTMLInputElement;
  const runInferenceBtn = containerElement.querySelector('#run-inference-btn') as HTMLButtonElement;
  const previewImage = containerElement.querySelector('#preview-image') as HTMLImageElement;
  const inferenceResults = containerElement.querySelector('#inference-results') as HTMLDivElement;
  const inferenceTime = containerElement.querySelector('#inference-time') as HTMLDivElement;
  const backendUsed = containerElement.querySelector('#backend-used') as HTMLDivElement;
  const runComparisonBtn = containerElement.querySelector('#run-comparison-btn') as HTMLButtonElement;
  const comparisonResults = containerElement.querySelector('#comparison-results') as HTMLDivElement;
  const comparisonChart = containerElement.querySelector('#comparison-chart') as HTMLDivElement;
  
  // Create storage manager
  const storageManager = new StorageManager('vit-models');
  await storageManager.initialize();
  
  // Variable to store current model instance
  let currentModel: HardwareAbstractedVIT | null = null;
  
  /**
   * Detect hardware capabilities
   */
  async function detectCapabilities(): Promise<any> {
    // Initialize HAL
    const hal = await createHardwareAbstraction();
    const capabilities = hal.getCapabilities();
    const availableBackends = hal.getAvailableBackends();
    
    // Format capabilities for display
    capabilitiesResult.textContent = JSON.stringify(capabilities, null, 2);
    
    // Display available backends
    backendList.innerHTML = '<h4>Available Backends:</h4>';
    
    if (availableBackends.length === 0) {
      backendList.innerHTML += '<div>No hardware backends available</div>';
    } else {
      backendList.innerHTML += availableBackends.map(backend => {
        const backendClass = backend.replace(/[^a-z0-9]/gi, '').toLowerCase();
        return `<span class="backend-badge ${backendClass}">${backend}</span>`;
      }).join(' ');
      
      // Add HAL auto selection badge
      backendList.innerHTML += '<span class="backend-badge auto">HAL Auto-Selection</span>';
      
      // Best backend for vision
      const bestBackend = hal.getBestBackend('vision');
      if (bestBackend) {
        backendList.innerHTML += `<div>Best backend for vision: <strong>${bestBackend.type}</strong></div>`;
      }
    }
    
    return capabilities;
  }
  
  /**
   * Run inference with HAL-accelerated ViT model
   */
  async function runInference(imageUrl: string): Promise<string[]> {
    try {
      // Update preview image
      previewImage.src = imageUrl;
      inferenceResults.textContent = 'Running inference...';
      inferenceTime.textContent = '';
      backendUsed.textContent = '';
      
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
      
      // Check if we need to create a new model
      if (!currentModel) {
        // Initialize HAL-accelerated ViT model
        console.log('Initializing new ViT model with Hardware Abstraction Layer');
        currentModel = new HardwareAbstractedVIT(vitConfig, storageManager);
        await currentModel.initialize();
      }
      
      // Get model info
      const modelInfo = currentModel.getModelInfo();
      
      // Load input image
      const inputImage = await loadImage(imageUrl);
      
      // Run inference
      console.log('Running inference with HAL-accelerated ViT');
      const startTime = performance.now();
      const probabilities = await currentModel.predict(inputImage);
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
      backendUsed.textContent = `Backend: ${modelInfo.selectedBackend}`;
      
      return topPredictions.map(p => p.label);
      
    } catch (error) {
      console.error('Error running inference:', error);
      inferenceResults.textContent = 'Error running inference: ' + error;
      return [];
    }
  }
  
  /**
   * Compare performance across available backends
   */
  async function compareBackends(): Promise<void> {
    try {
      comparisonResults.textContent = 'Running comparison...';
      comparisonChart.innerHTML = '';
      
      // Get image URL
      const imageUrl = imageUrlInput.value.trim();
      if (!imageUrl) {
        throw new Error('Please enter an image URL');
      }
      
      // Run comparison
      const comparison = await runCrossBackendPerformanceComparison(imageUrl);
      
      // Display results
      comparisonResults.innerHTML = `
        <div class="comparison-summary">
          <h4>Results</h4>
          <table class="comparison-table">
            <thead>
              <tr>
                <th>Backend</th>
                <th>Inference Time (ms)</th>
                <th>Support Level</th>
                <th>Top Prediction</th>
              </tr>
            </thead>
            <tbody>
              ${Object.entries(comparison.results).map(([backend, data]) => `
                <tr class="${backend === comparison.bestBackend ? 'best-backend' : ''}">
                  <td>${backend}</td>
                  <td>${data.inferenceTime > 0 ? data.inferenceTime.toFixed(2) : 'N/A'}</td>
                  <td>${data.supportLevel}</td>
                  <td>${data.topPrediction}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
          <div class="best-backend-summary">
            <strong>Best backend: ${comparison.bestBackend}</strong>
          </div>
        </div>
      `;
      
      // Create chart
      comparisonChart.innerHTML = `
        <div class="chart-container">
          <div class="chart-label">Backend</div>
          <div class="chart-bars">
            ${Object.entries(comparison.results)
              .filter(([_, data]) => data.inferenceTime > 0)
              .sort(([_, dataA], [__, dataB]) => dataA.inferenceTime - dataB.inferenceTime)
              .map(([backend, data]) => {
                const backendClass = backend.replace(/[^a-z0-9]/gi, '').toLowerCase();
                // Calculate percentage of slowest time
                const maxTime = Math.max(...Object.values(comparison.results)
                  .filter(d => d.inferenceTime > 0)
                  .map(d => d.inferenceTime));
                const percentage = (data.inferenceTime / maxTime * 100).toFixed(2);
                
                return `
                  <div class="chart-bar-group">
                    <div class="chart-bar-label">${backend}</div>
                    <div class="chart-bar ${backendClass}" style="width: ${percentage}%">
                      ${data.inferenceTime.toFixed(2)}ms
                    </div>
                  </div>
                `;
              }).join('')}
          </div>
        </div>
        <style>
          .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
          }
          
          .comparison-table th, .comparison-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
          }
          
          .comparison-table th {
            background-color: #f2f2f2;
          }
          
          .comparison-table tr.best-backend {
            background-color: #e6f7ff;
            font-weight: bold;
          }
          
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
          
          .chart-bar.webgpu {
            background: #4285f4;
          }
          
          .chart-bar.webnn {
            background: #0f9d58;
          }
          
          .chart-bar.cpu {
            background: #db4437;
          }
          
          .chart-bar.halauto {
            background: #f4b400;
            color: black;
          }
        </style>
      `;
    } catch (error) {
      console.error('Error running comparison:', error);
      comparisonResults.textContent = 'Error running comparison: ' + error;
    }
  }
  
  // Wire up event handlers
  detectCapabilitiesBtn.addEventListener('click', () => {
    detectCapabilities();
  });
  
  runInferenceBtn.addEventListener('click', () => {
    const imageUrl = imageUrlInput.value.trim();
    if (imageUrl) {
      runInference(imageUrl);
    } else {
      inferenceResults.textContent = 'Please enter an image URL';
    }
  });
  
  runComparisonBtn.addEventListener('click', () => {
    compareBackends();
  });
  
  // Run initial detection
  await detectCapabilities();
  
  // Return API for external use
  return {
    runInference,
    detectCapabilities,
    compareBackends
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
    'Dungeness crab', 'rock crab', 'fiddler crab', 'red king crab', 'American lobster',
    'spiny lobster', 'crayfish', 'hermit crab', 'isopod', 'white stork', 'black stork',
    'spoonbill', 'flamingo', 'little blue heron', 'great egret', 'bittern', 
    'crane', 'limpkin', 'common gallinule', 'American coot', 'bustard',
    'ruddy turnstone', 'dunlin', 'common redshank', 'dowitcher', 'oystercatcher',
    'pelican', 'king penguin', 'albatross', 'grey whale', 'killer whale', 'dugong',
    'sea lion', 'Chihuahua', 'Japanese Chin', 'Maltese', 'Pekingese', 'Shih Tzu',
    'King Charles Spaniel', 'Papillon', 'toy terrier', 'Rhodesian Ridgeback',
    'Afghan Hound', 'Basset Hound', 'Beagle', 'Bloodhound', 'Bluetick Coonhound',
    'Black and Tan Coonhound', 'Treeing Walker Coonhound', 'English foxhound',
    'Redbone Coonhound', 'borzoi', 'Irish Wolfhound', 'Italian Greyhound',
    'Whippet', 'Ibizan Hound', 'Norwegian Elkhound', 'Otterhound', 'Saluki',
    'Scottish Deerhound', 'Weimaraner', 'Staffordshire Bull Terrier',
    'American Staffordshire Terrier', 'Bedlington Terrier', 'Border Terrier',
    'Kerry Blue Terrier', 'Irish Terrier', 'Norfolk Terrier', 'Norwich Terrier',
    'Yorkshire Terrier', 'Wire Fox Terrier', 'Lakeland Terrier', 'Sealyham Terrier',
    'Airedale Terrier', 'Cairn Terrier', 'Australian Terrier', 'Dandie Dinmont Terrier',
    'Boston Terrier', 'Miniature Schnauzer', 'Giant Schnauzer', 'Standard Schnauzer',
    'Scottish Terrier', 'Tibetan Terrier', 'Australian Silky Terrier',
    'Soft-coated Wheaten Terrier', 'West Highland White Terrier', 'Lhasa Apso',
    'Flat-Coated Retriever', 'Curly-coated Retriever', 'Golden Retriever',
    'Labrador Retriever', 'Chesapeake Bay Retriever', 'German Shorthaired Pointer',
    'Vizsla', 'English Setter', 'Irish Setter', 'Gordon Setter', 'Brittany',
    'Clumber Spaniel', 'English Springer Spaniel', 'Welsh Springer Spaniel',
    'Cocker Spaniels', 'Sussex Spaniel', 'Irish Water Spaniel', 'Kuvasz',
    'Schipperke', 'Groenendael', 'Malinois', 'Briard', 'Australian Kelpie',
    'Komondor', 'Old English Sheepdog', 'Shetland Sheepdog', 'collie',
    'Border Collie', 'Bouvier des Flandres', 'Rottweiler', 'German Shepherd Dog',
    'Dobermann', 'Miniature Pinscher', 'Greater Swiss Mountain Dog', 
    'Bernese Mountain Dog', 'Appenzeller Sennenhund', 'Entlebucher Sennenhund',
    'Boxer', 'Bullmastiff', 'Tibetan Mastiff', 'French Bulldog', 'Great Dane',
    'St. Bernard', 'husky', 'Alaskan Malamute', 'Siberian Husky', 'Dalmatian',
    'Affenpinscher', 'Basenji', 'pug', 'Leonberger', 'Newfoundland', 'Pyrenean Mountain Dog',
    'Samoyed', 'Pomeranian', 'Chow Chow', 'Keeshond', 'Brussels Griffon', 'Pembroke Welsh Corgi',
    'Cardigan Welsh Corgi', 'Toy Poodle', 'Miniature Poodle', 'Standard Poodle',
    'Mexican hairless dog', 'grey wolf', 'Alaskan tundra wolf', 'red wolf', 'coyote',
    'dingo', 'dhole', 'African wild dog', 'hyena', 'red fox', 'kit fox',
    'Arctic fox', 'grey fox', 'tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat',
    'Egyptian Mau', 'cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar', 'lion',
    'tiger', 'cheetah', 'brown bear', 'American black bear', 'polar bear', 'sloth bear',
    'mongoose', 'meerkat', 'tiger beetle', 'ladybug', 'ground beetle', 'longhorn beetle',
    'leaf beetle', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant',
    'grasshopper', 'cricket', 'stick insect', 'cockroach', 'mantis', 'cicada', 'leafhopper',
    'lacewing', 'dragonfly', 'damselfly', 'red admiral', 'ringlet', 'monarch butterfly',
    'small white butterfly', 'sulphur butterfly', 'gossamer-winged butterfly', 'starfish',
    'sea urchin', 'sea cucumber', 'cottontail rabbit', 'hare', 'Angora rabbit',
    'hamster', 'porcupine', 'fox squirrel', 'marmot', 'beaver', 'guinea pig',
    'common sorrel', 'zebra', 'pig', 'wild boar', 'warthog', 'hippopotamus',
    'ox', 'water buffalo', 'bison', 'ram', 'bighorn sheep', 'Alpine ibex', 'hartebeest',
    'impala', 'gazelle', 'dromedary', 'llama', 'weasel', 'mink', 'European polecat',
    'black-footed ferret', 'otter', 'skunk', 'badger', 'armadillo', 'three-toed sloth',
    'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'siamang', 'guenon', 'patas monkey',
    'baboon', 'macaque', 'langur', 'black-and-white colobus', 'proboscis monkey',
    'marmoset', 'white-headed capuchin', 'howler monkey', 'titi monkey',
    'Geoffroy\'s spider monkey', 'common squirrel monkey', 'ring-tailed lemur',
    'indri', 'Asian elephant', 'African bush elephant', 'red panda', 'giant panda',
    'snoek', 'eel', 'coho salmon', 'rock beauty', 'clownfish', 'sturgeon',
    'garfish', 'lionfish', 'pufferfish', 'abacus', 'abaya', 'academic gown',
    'accordion', 'acoustic guitar', 'aircraft carrier', 'airliner', 'airship',
    'altar', 'ambulance', 'amphibious vehicle', 'analog clock', 'apiary', 'apron',
    'waste container', 'assault rifle', 'backpack', 'bakery', 'balance beam',
    'balloon', 'ballpoint pen', 'Band-Aid', 'banjo', 'baluster', 'barbell',
    'barber chair', 'barbershop', 'barn', 'barometer', 'barrel', 'wheelbarrow',
    'baseball', 'basketball', 'bassinet', 'bassoon', 'swimming cap', 'bath towel',
    'bathtub', 'station wagon', 'lighthouse', 'beaker', 'military cap', 'beer bottle',
    'beer glass', 'bell-cot', 'bib', 'tandem bicycle', 'bikini', 'ring binder',
    'binoculars', 'birdhouse', 'boathouse', 'bobsleigh', 'bolo tie', 'poke bonnet',
    'bookcase', 'bookstore', 'bottle cap', 'bow', 'bow tie', 'brass', 'bra', 'breakwater',
    'breastplate', 'broom', 'bucket', 'buckle', 'bulletproof vest', 'high-speed train',
    'butcher shop', 'taxicab', 'cauldron', 'candle', 'cannon', 'canoe', 'can opener',
    'cardigan', 'car mirror', 'carousel', 'tool kit', 'carton', 'car wheel', 'automated teller machine',
    'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello', 'mobile phone',
    'chain', 'chain-link fence', 'chain mail', 'chainsaw', 'chest', 'chest of drawers',
    'chiffonier', 'chime', 'china cabinet', 'Christmas stocking', 'church', 'movie theater',
    'cleaver', 'cliff dwelling', 'cloak', 'clogs', 'cocktail shaker', 'coffee mug',
    'coffeemaker', 'coil', 'combination lock', 'computer keyboard', 'confectionery store',
    'container ship', 'convertible', 'corkscrew', 'cornet', 'cowboy boot', 'cowboy hat',
    'cradle', 'crane (machine)', 'crash helmet', 'crate', 'infant bed', 'Crock Pot',
    'croquet ball', 'crutch', 'cuirass', 'dam', 'desk', 'desktop computer', 'rotary dial telephone',
    'diaper', 'digital clock', 'digital watch', 'dining table', 'dishcloth', 'dishwasher',
    'disc brake', 'dock', 'dog sled', 'dome', 'doormat', 'drilling rig', 'drum', 'drumstick',
    'dumbbell', 'Dutch oven', 'electric fan', 'electric guitar', 'electric locomotive',
    'entertainment center', 'envelope', 'espresso machine', 'face powder', 'feather boa',
    'filing cabinet', 'fireboat', 'fire engine', 'fire screen sheet', 'flagpole', 'flute',
    'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen', 'four-poster bed',
    'freight car', 'French horn', 'frying pan', 'fur coat', 'garbage truck', 'gas mask',
    'gas pump', 'goblet', 'go-kart', 'golf ball', 'golf cart', 'gondola', 'gong', 'gown',
    'grand piano', 'greenhouse', 'grille', 'grocery store', 'guillotine', 'hair clip',
    'hair spray', 'half-track', 'hammer', 'hamper', 'hair dryer', 'hand-held computer',
    'handkerchief', 'hard disk drive', 'harmonica', 'harp', 'harvester', 'hatchet', 'holster',
    'home theater', 'honeycomb', 'hook', 'hoop skirt', 'horizontal bar', 'horse-drawn vehicle',
    'hourglass', 'iPod', 'clothes iron', 'jack-o\'-lantern', 'jeans', 'jeep', 'T-shirt',
    'jigsaw puzzle', 'pulled rickshaw', 'joystick', 'kimono', 'knee pad', 'knot', 'lab coat',
    'ladle', 'lampshade', 'laptop computer', 'lawn mower', 'lens cap', 'paper knife',
    'library', 'lifeboat', 'lighter', 'limousine', 'ocean liner', 'lipstick', 'slip-on shoe',
    'lotion', 'speaker', 'loupe', 'sawmill', 'magnetic compass', 'mail bag', 'mailbox',
    'tights', 'tank suit', 'manhole cover', 'maraca', 'marimba', 'mask', 'match',
    'maypole', 'maze', 'measuring cup', 'medicine chest', 'megalith', 'microphone',
    'microwave oven', 'military uniform', 'milk can', 'minibus', 'miniskirt', 'minivan',
    'missile', 'mitten', 'mixing bowl', 'mobile home', 'Model T', 'modem', 'monastery',
    'monitor', 'moped', 'mortar', 'square academic cap', 'mosque', 'mosquito net',
    'scooter', 'mountain bike', 'tent', 'computer mouse', 'mousetrap', 'moving van',
    'muzzle', 'nail', 'neck brace', 'necklace', 'nipple', 'notebook computer',
    'obelisk', 'oboe', 'ocarina', 'odometer', 'oil filter', 'organ', 'oscilloscope',
    'overskirt', 'bullock cart', 'oxygen mask', 'packet', 'paddle', 'paddle wheel',
    'padlock', 'paintbrush', 'pajamas', 'palace', 'pan flute', 'paper towel',
    'parachute', 'parallel bars', 'park bench', 'parking meter', 'passenger car',
    'patio', 'payphone', 'pedestal', 'pencil case', 'pencil sharpener', 'perfume',
    'Petri dish', 'photocopier', 'plectrum', 'Pickelhaube', 'picket fence', 'pickup truck',
    'pier', 'piggy bank', 'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel',
    'pirate ship', 'pitcher', 'hand plane', 'planetarium', 'plastic bag', 'plate rack',
    'plow', 'plunger', 'Polaroid camera', 'pole', 'police van', 'poncho',
    'billiard table', 'soda bottle', 'pot', 'potter\'s wheel', 'power drill',
    'prayer rug', 'printer', 'prison', 'projectile', 'projector', 'hockey puck',
    'punching bag', 'purse', 'quill', 'quilt', 'race car', 'racket', 'radiator',
    'radio', 'radio telescope', 'rain barrel', 'recreational vehicle', 'reel',
    'reflex camera', 'refrigerator', 'remote control', 'restaurant', 'revolver',
    'rifle', 'rocking chair', 'rotisserie', 'eraser', 'rugby ball', 'ruler',
    'running shoe', 'safe', 'safety pin', 'salt shaker', 'sandal', 'sarong',
    'saxophone', 'scabbard', 'weighing scale', 'school bus', 'schooner', 'scoreboard',
    'CRT screen', 'screw', 'screwdriver', 'seat belt', 'sewing machine', 'shield',
    'shoe store', 'shoji', 'shopping basket', 'shopping cart', 'shovel', 'shower cap',
    'shower curtain', 'ski', 'ski mask', 'sleeping bag', 'slide rule', 'sliding door',
    'slot machine', 'snorkel', 'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball',
    'sock', 'solar thermal collector', 'sombrero', 'soup bowl', 'space bar',
    'space heater', 'space shuttle', 'spatula', 'motorboat', 'spider web', 'spindle',
    'sports car', 'spotlight', 'stage', 'steam locomotive', 'through arch bridge',
    'steel drum', 'stethoscope', 'scarf', 'stone wall', 'stopwatch', 'stove',
    'strainer', 'tram', 'stretcher', 'couch', 'stupa', 'submarine', 'suit', 'sundial',
    'sunglass', 'sunglasses', 'sunscreen', 'suspension bridge', 'mop', 'sweatshirt',
    'swimsuit', 'swing', 'switch', 'syringe', 'table lamp', 'tank', 'tape player',
    'teapot', 'teddy bear', 'television', 'tennis ball', 'thatched roof', 'front curtain',
    'thimble', 'threshing machine', 'throne', 'tile roof', 'toaster', 'tobacco shop',
    'toilet seat', 'torch', 'totem pole', 'tow truck', 'toy store', 'tractor',
    'semi-trailer truck', 'tray', 'trench coat', 'tricycle', 'trimaran', 'tripod',
    'triumphal arch', 'trolleybus', 'trombone', 'tub', 'turnstile', 'typewriter keyboard',
    'umbrella', 'unicycle', 'upright piano', 'vacuum cleaner', 'vase', 'vault',
    'velvet', 'vending machine', 'vestment', 'viaduct', 'violin', 'volleyball',
    'waffle iron', 'wall clock', 'wallet', 'wardrobe', 'military aircraft', 'sink',
    'washing machine', 'water bottle', 'water jug', 'water tower', 'whiskey jug',
    'whistle', 'wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle',
    'wing', 'wok', 'wooden spoon', 'wool', 'split-rail fence', 'shipwreck', 'yawl',
    'yurt', 'website', 'comic book', 'crossword', 'traffic sign', 'traffic light',
    'dust jacket', 'menu', 'plate', 'guacamole', 'consommé', 'hot pot', 'trifle',
    'ice cream', 'ice pop', 'baguette', 'bagel', 'pretzel', 'cheeseburger', 'hot dog',
    'mashed potato', 'cabbage', 'broccoli', 'cauliflower', 'zucchini', 'spaghetti squash',
    'acorn squash', 'butternut squash', 'cucumber', 'artichoke', 'bell pepper',
    'cardoon', 'mushroom', 'Granny Smith apple', 'strawberry', 'orange', 'lemon',
    'fig', 'pineapple', 'banana', 'jackfruit', 'custard apple', 'pomegranate', 'hay',
    'carbonara', 'chocolate syrup', 'dough', 'meatloaf', 'pizza', 'pot pie', 'burrito',
    'red wine', 'espresso', 'cup', 'eggnog', 'alp', 'bubble', 'cliff', 'coral reef',
    'geyser', 'lakeshore', 'promontory', 'shoal', 'seashore', 'valley', 'volcano',
    'baseball player', 'bridegroom', 'scuba diver', 'rapeseed', 'daisy', 'yellow lady\'s slipper',
    'corn', 'acorn', 'rose hip', 'horse chestnut seed', 'coral fungus', 'agaric',
    'gyromitra', 'stinkhorn mushroom', 'earth star', 'hen-of-the-woods', 'bolete', 'ear',
    'toilet paper', 'cat'
  ];
}

// Example usage in browser:
// 
// document.addEventListener('DOMContentLoaded', async () => {
//   const demoContainer = document.getElementById('hal-vit-demo-container');
//   if (demoContainer) {
//     const demo = await createInteractiveHALDemo(demoContainer);
//     // You can programmatically interact with the demo through the API:
//     // await demo.detectCapabilities();
//     // await demo.runInference('https://example.com/cat.jpg');
//     // await demo.compareBackends();
//   }
// });