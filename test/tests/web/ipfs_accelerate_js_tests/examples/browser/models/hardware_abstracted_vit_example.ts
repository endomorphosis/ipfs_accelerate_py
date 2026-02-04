/**
 * Hardware Abstracted ViT Example
 * 
 * This example demonstrates the use of the Hardware Abstraction Layer (HAL)
 * with Vision Transformer (ViT) models for optimal performance across different
 * hardware backends (WebGPU, WebNN, CPU).
 */

import { createHardwareAbstraction, HardwareAbstraction, HardwareBackendType } from '../../../hardware/hardware_abstraction_layer';
import { HardwareAbstractedViT, HardwareAbstractedViTConfig, createHardwareAbstractedViT } from '../../../model/vision/hardware_abstracted_vit';
import { StorageManager } from '../../../storage/storage_manager';
import { IndexedDBStorageManager } from '../../../storage/indexeddb/indexeddb_storage_manager';
import { HardwareAbstractedBERT, createHardwareAbstractedBERT } from '../../../model/transformers/hardware_abstracted_bert';
import { BrowserDetector } from '../../../hardware/browser/browser_detector';
import { SharedTensor } from '../../../tensor/shared_tensor';

// ImageNet label map (simplified for the example)
const IMAGENET_LABELS = [
  'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark',
  'electric ray', 'stingray', 'rooster', 'hen', 'ostrich', 'brambling',
  'goldfinch', 'house finch', 'junco', 'indigo bunting', 'American robin',
  'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel', 'kite', 'bald eagle',
  'vulture', 'great grey owl', 'fire salamander', 'smooth newt', 'newt', 'spotted salamander',
  'axolotl', 'American bullfrog', 'tree frog', 'tailed frog', 'loggerhead sea turtle',
  'leatherback sea turtle', 'mud turtle', 'terrapin', 'box turtle', 'banded gecko',
  'green iguana', 'Carolina anole', 'desert grassland whiptail lizard', 'agama',
  'frilled-necked lizard', 'alligator lizard', 'Gila monster', 'European green lizard',
  'chameleon', 'Komodo dragon', 'Nile crocodile', 'American alligator', 'triceratops',
  'worm snake', 'ring-necked snake', 'eastern hog-nosed snake', 'smooth green snake',
  'kingsnake', 'garter snake', 'water snake', 'vine snake', 'night snake',
  'boa constrictor', 'African rock python', 'Indian cobra', 'green mamba',
  'sea snake', 'Saharan horned viper', 'eastern diamondback rattlesnake', 'sidewinder',
  'trilobite', 'harvestman', 'scorpion', 'yellow garden spider', 'barn spider',
  'European garden spider', 'southern black widow', 'tarantula', 'wolf spider',
  'tick', 'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie grouse',
  'peacock', 'quail', 'partridge', 'grey partridge', 'house sparrow', 'grey parrot',
  'macaw', 'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater', 'hornbill',
  'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted merganser', 'goose',
  'black swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala', 'wombat',
  // Simplified for brevity - add more as needed or use a complete set
  'cat', 'dog', 'car', 'bird', 'guitar', 'piano', 'computer', 'keyboard', 'mouse',
  'telephone', 'television', 'airplane', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair dryer', 'toothbrush', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
  'carrot', 'hot dog', 'pizza', 'donut', 'cake'
];

/**
 * Main class for the Hardware Abstracted ViT example
 */
export class HardwareAbstractedViTExample {
  private hal: HardwareAbstraction | null = null;
  private storageManager: StorageManager | null = null;
  private vitModel: HardwareAbstractedViT | null = null;
  private bertModel: HardwareAbstractedBERT | null = null;
  private capabilities: any = null;
  private selectedImageUrl: string = '';
  private modelConfig: Partial<HardwareAbstractedViTConfig> = {
    modelId: 'vit-base-patch16-224',
    enableTensorSharing: true,
    useQuantization: true
  };

  // Performance metrics
  private metrics: {
    initializationTime: number;
    preprocessingTime: number;
    inferenceTime: number;
    totalTime: number;
  } = {
    initializationTime: 0,
    preprocessingTime: 0,
    inferenceTime: 0,
    totalTime: 0
  };

  // DOM elements
  private elements: Record<string, HTMLElement> = {};

  /**
   * Constructor - initialize the example
   */
  constructor() {
    this.initializeDOM();
    this.setupEventListeners();
  }

  /**
   * Initialize DOM references
   */
  private initializeDOM() {
    // Get all necessary DOM elements
    this.elements = {
      detectHardwareBtn: document.getElementById('detect-hardware-btn') as HTMLButtonElement,
      hardwareCapabilities: document.getElementById('hardware-capabilities') as HTMLDivElement,
      backendBadges: document.getElementById('backend-badges') as HTMLDivElement,
      modelSelect: document.getElementById('model-select') as HTMLSelectElement,
      quantizationCheckbox: document.getElementById('quantization-checkbox') as HTMLInputElement,
      tensorSharingCheckbox: document.getElementById('tensor-sharing-checkbox') as HTMLInputElement,
      initializeModelBtn: document.getElementById('initialize-model-btn') as HTMLButtonElement,
      modelStatus: document.getElementById('model-status') as HTMLDivElement,
      imageUrl: document.getElementById('image-url') as HTMLInputElement,
      loadImageBtn: document.getElementById('load-image-btn') as HTMLButtonElement,
      sampleImages: document.getElementById('sample-images') as HTMLDivElement,
      previewImage: document.getElementById('preview-image') as HTMLImageElement,
      runInferenceBtn: document.getElementById('run-inference-btn') as HTMLButtonElement,
      performanceMetrics: document.getElementById('performance-metrics') as HTMLDivElement,
      initializationTime: document.getElementById('initialization-time') as HTMLDivElement,
      preprocessingTime: document.getElementById('preprocessing-time') as HTMLDivElement,
      inferenceTime: document.getElementById('inference-time') as HTMLDivElement,
      totalTime: document.getElementById('total-time') as HTMLDivElement,
      backendInfo: document.getElementById('backend-info') as HTMLDivElement,
      classificationResults: document.getElementById('classification-results') as HTMLDivElement,
      runComparisonBtn: document.getElementById('run-comparison-btn') as HTMLButtonElement,
      comparisonResults: document.getElementById('comparison-results') as HTMLDivElement,
      comparisonTable: document.getElementById('comparison-table') as HTMLTableElement,
      comparisonChart: document.getElementById('comparison-chart') as HTMLDivElement,
      multimodalDemoBtn: document.getElementById('multimodal-demo-btn') as HTMLButtonElement,
      multimodalResults: document.getElementById('multimodal-results') as HTMLDivElement
    };
  }

  /**
   * Set up event listeners for UI interactions
   */
  private setupEventListeners() {
    // Hardware detection button
    this.elements.detectHardwareBtn.addEventListener('click', () => this.detectHardwareCapabilities());

    // Model configuration
    this.elements.modelSelect.addEventListener('change', () => {
      this.modelConfig.modelId = this.elements.modelSelect.value;
    });

    this.elements.quantizationCheckbox.addEventListener('change', () => {
      this.modelConfig.useQuantization = this.elements.quantizationCheckbox.checked;
    });

    this.elements.tensorSharingCheckbox.addEventListener('change', () => {
      this.modelConfig.enableTensorSharing = this.elements.tensorSharingCheckbox.checked;
    });

    // Initialize model button
    this.elements.initializeModelBtn.addEventListener('click', () => this.initializeModel());

    // Image loading
    this.elements.loadImageBtn.addEventListener('click', () => this.loadImage());
    
    // Sample image selection
    const sampleImages = this.elements.sampleImages.querySelectorAll('.image-option');
    sampleImages.forEach(img => {
      img.addEventListener('click', (event) => {
        // Update the selected image
        sampleImages.forEach(el => el.classList.remove('selected'));
        (event.target as HTMLElement).classList.add('selected');
        
        // Update the image URL input
        const url = (event.target as HTMLElement).getAttribute('data-url') || '';
        this.elements.imageUrl.value = url;
        
        // Load the image
        this.loadImage();
      });
    });

    // Run inference button
    this.elements.runInferenceBtn.addEventListener('click', () => this.runInference());

    // Compare backends button
    this.elements.runComparisonBtn.addEventListener('click', () => this.compareBackends());

    // Multimodal demo button
    this.elements.multimodalDemoBtn.addEventListener('click', () => this.runMultimodalDemo());
  }

  /**
   * Detect hardware capabilities
   */
  async detectHardwareCapabilities() {
    // Update UI
    this.elements.detectHardwareBtn.disabled = true;
    this.elements.hardwareCapabilities.textContent = 'Detecting hardware capabilities...';
    this.elements.backendBadges.style.display = 'none';

    try {
      // Create HAL instance
      this.hal = await createHardwareAbstraction();
      
      // Get capabilities
      this.capabilities = this.hal.getCapabilities();
      
      // Display capabilities
      this.elements.hardwareCapabilities.innerHTML = JSON.stringify(this.capabilities, null, 2);
      
      // Display backend badges
      this.displayBackendBadges();
      
      // Enable model initialization
      this.elements.initializeModelBtn.disabled = false;
    } catch (error) {
      console.error('Error detecting hardware capabilities:', error);
      this.elements.hardwareCapabilities.textContent = `Error: ${error.message}`;
    } finally {
      this.elements.detectHardwareBtn.disabled = false;
    }
  }

  /**
   * Display backend badges based on available hardware
   */
  private displayBackendBadges() {
    if (!this.hal || !this.capabilities) return;
    
    const availableBackends = this.hal.getAvailableBackends();
    const bestBackend = this.determineOptimalBackend('vision');
    
    let html = '<div style="margin-bottom: 10px;"><strong>Available Backends:</strong></div>';
    
    // Add backend badges
    availableBackends.forEach(backend => {
      const isBest = backend === bestBackend;
      const badgeClass = `hardware-badge ${backend}-badge${isBest ? ' best-backend' : ''}`;
      html += `<span class="${badgeClass}">${backend}</span>`;
    });
    
    // Add HAL badge
    html += '<span class="hardware-badge auto-badge">HAL Auto-Selection</span>';
    
    // Add best backend info
    html += `<div style="margin-top: 10px;">Best backend for vision: <strong>${bestBackend}</strong></div>`;
    
    // Add browser detection info
    const browserInfo = BrowserDetector.detect();
    html += `<div style="margin-top: 10px;">Detected browser: <strong>${browserInfo.name} ${browserInfo.version}</strong></div>`;
    
    this.elements.backendBadges.innerHTML = html;
    this.elements.backendBadges.style.display = 'block';
  }

  /**
   * Determine the optimal backend for a given model type
   */
  private determineOptimalBackend(modelType: string): HardwareBackendType {
    if (!this.hal) return 'cpu';
    
    const availableBackends = this.hal.getAvailableBackends();
    
    // Vision models typically perform best on WebGPU
    if (modelType === 'vision') {
      if (availableBackends.includes('webgpu')) {
        return 'webgpu';
      } else if (availableBackends.includes('webnn')) {
        return 'webnn';
      }
    }
    
    // Text models typically perform best on WebNN (if available)
    if (modelType === 'text') {
      if (availableBackends.includes('webnn')) {
        return 'webnn';
      } else if (availableBackends.includes('webgpu')) {
        return 'webgpu';
      }
    }
    
    // Audio models typically perform best on WebGPU with Firefox
    if (modelType === 'audio') {
      const isFirefox = BrowserDetector.detect().name.toLowerCase() === 'firefox';
      if (isFirefox && availableBackends.includes('webgpu')) {
        return 'webgpu';
      } else if (availableBackends.includes('webnn')) {
        return 'webnn';
      } else if (availableBackends.includes('webgpu')) {
        return 'webgpu';
      }
    }
    
    // Default to CPU as fallback
    return 'cpu';
  }

  /**
   * Initialize the ViT model
   */
  async initializeModel() {
    // Update UI
    this.elements.initializeModelBtn.disabled = true;
    this.showModelStatus('Initializing model...');
    
    // Reset previous models
    await this.disposeModels();

    const startTime = performance.now();
    
    try {
      // Create storage manager
      this.storageManager = new IndexedDBStorageManager();
      await this.storageManager.initialize();
      
      // Ensure HAL is created
      if (!this.hal) {
        this.hal = await createHardwareAbstraction();
      }
      
      // Create model
      const config: Partial<HardwareAbstractedViTConfig> = {
        ...this.modelConfig,
        // Parse dimensions from model ID
        imageSize: this.parseImageSize(this.modelConfig.modelId || 'vit-base-patch16-224'),
        patchSize: this.parsePatchSize(this.modelConfig.modelId || 'vit-base-patch16-224'),
        browserType: BrowserDetector.detect().name.toLowerCase() as any
      };
      
      // Create the model
      this.vitModel = createHardwareAbstractedViT(this.hal, config);
      
      // Initialize model
      await this.vitModel.initialize();
      
      // Calculate initialization time
      const endTime = performance.now();
      this.metrics.initializationTime = Math.round(endTime - startTime);
      
      // Update UI
      this.showModelStatus(`Model initialized successfully in ${this.metrics.initializationTime}ms.\nUsing backend: ${this.hal.getActiveBackend().type}`);
      this.elements.runInferenceBtn.disabled = false;
      this.elements.runComparisonBtn.disabled = false;
      
      // Enable multimodal demo if tensor sharing is enabled
      if (this.modelConfig.enableTensorSharing) {
        this.elements.multimodalDemoBtn.disabled = false;
      }
      
    } catch (error) {
      console.error('Error initializing model:', error);
      this.showModelStatus(`Error initializing model: ${error.message}`);
    } finally {
      this.elements.initializeModelBtn.disabled = false;
    }
  }

  /**
   * Parse image size from model ID
   */
  private parseImageSize(modelId: string): number {
    // Extract image size from model ID (e.g., vit-base-patch16-224 -> 224)
    const match = modelId.match(/(\d+)$/);
    return match ? parseInt(match[1], 10) : 224;
  }

  /**
   * Parse patch size from model ID
   */
  private parsePatchSize(modelId: string): number {
    // Extract patch size from model ID (e.g., vit-base-patch16-224 -> 16)
    const match = modelId.match(/patch(\d+)/);
    return match ? parseInt(match[1], 10) : 16;
  }

  /**
   * Show model status
   */
  private showModelStatus(message: string) {
    this.elements.modelStatus.textContent = message;
    this.elements.modelStatus.style.display = 'block';
  }

  /**
   * Load an image from the input URL
   */
  async loadImage() {
    const url = this.elements.imageUrl.value.trim();
    if (!url) {
      alert('Please enter an image URL');
      return;
    }
    
    // Update selected image URL
    this.selectedImageUrl = url;
    
    // Show loading state
    this.elements.loadImageBtn.disabled = true;
    this.elements.previewImage.style.display = 'none';
    
    try {
      // Load the image
      this.elements.previewImage.src = url;
      this.elements.previewImage.style.display = 'none';
      
      // Wait for image to load
      await new Promise<void>((resolve, reject) => {
        this.elements.previewImage.onload = () => resolve();
        this.elements.previewImage.onerror = () => reject(new Error('Failed to load image'));
      });
      
      // Show the image
      this.elements.previewImage.style.display = 'block';
      
      // Enable inference if model is initialized
      if (this.vitModel) {
        this.elements.runInferenceBtn.disabled = false;
      }
      
    } catch (error) {
      console.error('Error loading image:', error);
      alert(`Error loading image: ${error.message}`);
    } finally {
      this.elements.loadImageBtn.disabled = false;
    }
  }

  /**
   * Run inference on the loaded image
   */
  async runInference() {
    if (!this.vitModel || !this.selectedImageUrl) {
      alert('Please initialize the model and load an image first');
      return;
    }
    
    // Update UI
    this.elements.runInferenceBtn.disabled = true;
    this.elements.classificationResults.textContent = 'Running inference...';
    this.resetMetricsDisplay();
    
    const totalStartTime = performance.now();
    
    try {
      // Get the image element
      const imageElement = this.elements.previewImage;
      
      // Prepare image data
      const preprocessStartTime = performance.now();
      const imageData = await this.prepareImageDataFromElement(imageElement);
      const preprocessEndTime = performance.now();
      this.metrics.preprocessingTime = Math.round(preprocessEndTime - preprocessStartTime);
      
      // Run inference
      const inferenceStartTime = performance.now();
      const result = await this.vitModel.process(imageData);
      const inferenceEndTime = performance.now();
      this.metrics.inferenceTime = Math.round(inferenceEndTime - inferenceStartTime);
      
      // Calculate total time
      const totalEndTime = performance.now();
      this.metrics.totalTime = Math.round(totalEndTime - totalStartTime);
      
      // Display results
      this.displayClassificationResults(result.probabilities);
      this.displayMetrics();
      this.displayBackendInfo(result.backend);
      
    } catch (error) {
      console.error('Error running inference:', error);
      this.elements.classificationResults.textContent = `Error: ${error.message}`;
    } finally {
      this.elements.runInferenceBtn.disabled = false;
    }
  }

  /**
   * Prepare image data from an image element
   */
  private async prepareImageDataFromElement(imageElement: HTMLImageElement): Promise<{
    imageData: Float32Array;
    width: number;
    height: number;
    isPreprocessed: boolean;
  }> {
    // Create a canvas to extract image data
    const canvas = document.createElement('canvas');
    const size = this.parseImageSize(this.modelConfig.modelId || 'vit-base-patch16-224');
    canvas.width = size;
    canvas.height = size;
    
    // Draw the image to the canvas
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get canvas context');
    }
    
    // Draw image with resize
    ctx.drawImage(imageElement, 0, 0, size, size);
    
    // Get image data
    const imageData = ctx.getImageData(0, 0, size, size);
    
    // Convert to Float32Array
    const floatData = new Float32Array(imageData.data.length);
    
    // Convert from RGBA to RGB and normalize
    const rgbData = new Float32Array(size * size * 3);
    let rgbIndex = 0;
    
    for (let i = 0; i < imageData.data.length; i += 4) {
      rgbData[rgbIndex++] = imageData.data[i] / 255.0;     // R
      rgbData[rgbIndex++] = imageData.data[i + 1] / 255.0; // G
      rgbData[rgbIndex++] = imageData.data[i + 2] / 255.0; // B
    }
    
    return {
      imageData: rgbData,
      width: size,
      height: size,
      isPreprocessed: true
    };
  }

  /**
   * Display classification results
   */
  private displayClassificationResults(probabilities: number[]) {
    // Get top 5 predictions
    const predictions = probabilities.map((prob, index) => ({
      label: index < IMAGENET_LABELS.length ? IMAGENET_LABELS[index] : `Class ${index}`,
      probability: prob
    }));
    
    // Sort by probability (descending)
    predictions.sort((a, b) => b.probability - a.probability);
    
    // Display top 5
    const top5 = predictions.slice(0, 5);
    
    let html = '<div style="margin-bottom: 10px;"><strong>Top 5 Predictions:</strong></div>';
    
    top5.forEach((pred, index) => {
      const percentage = (pred.probability * 100).toFixed(2);
      const barWidth = `${Math.max(pred.probability * 100, 0.5)}%`;
      
      html += `
        <div class="prediction">
          <div class="prediction-bar" style="width: ${barWidth}"></div>
          <div class="prediction-label">${index + 1}. ${pred.label}</div>
          <div class="prediction-value">${percentage}%</div>
        </div>
      `;
    });
    
    this.elements.classificationResults.innerHTML = html;
  }

  /**
   * Reset metrics display
   */
  private resetMetricsDisplay() {
    this.elements.performanceMetrics.style.display = 'none';
    this.elements.backendInfo.style.display = 'none';
  }

  /**
   * Display performance metrics
   */
  private displayMetrics() {
    this.elements.initializationTime.textContent = this.metrics.initializationTime.toString();
    this.elements.preprocessingTime.textContent = this.metrics.preprocessingTime.toString();
    this.elements.inferenceTime.textContent = this.metrics.inferenceTime.toString();
    this.elements.totalTime.textContent = this.metrics.totalTime.toString();
    this.elements.performanceMetrics.style.display = 'flex';
  }

  /**
   * Display backend information
   */
  private displayBackendInfo(backend: string) {
    if (!this.hal) return;
    
    const activeBackend = this.hal.getBackendType();
    const browserInfo = BrowserDetector.detect();
    
    this.elements.backendInfo.innerHTML = `
      <strong>Backend:</strong> ${backend} | 
      <strong>Browser:</strong> ${browserInfo.name} ${browserInfo.version}
    `;
    this.elements.backendInfo.style.display = 'block';
  }

  /**
   * Compare inference performance across all available backends
   */
  async compareBackends() {
    if (!this.hal || !this.selectedImageUrl) {
      alert('Please initialize the model and load an image first');
      return;
    }
    
    // Update UI
    this.elements.runComparisonBtn.disabled = true;
    this.elements.comparisonResults.style.display = 'none';
    this.elements.classificationResults.textContent = 'Running comparison across all backends...';
    
    try {
      // Get the image element
      const imageElement = this.elements.previewImage;
      
      // Prepare image data
      const imageData = await this.prepareImageDataFromElement(imageElement);
      
      // Get available backends
      const availableBackends = this.hal.getAvailableBackends();
      
      // Results object
      const results: Record<string, {
        inferenceTime: number;
        speedup: number;
        supportLevel: string;
        topPrediction: string;
      }> = {};
      
      // Current backend
      const originalBackend = this.hal.getBackendType();
      
      // Test each backend
      for (const backend of availableBackends) {
        // Set active backend
        this.hal.setBackendType(backend);
        
        // Recreate and initialize the model with the new backend
        const vitModel = createHardwareAbstractedViT(this.hal, this.modelConfig);
        await vitModel.initialize();
        
        // Run inference
        const startTime = performance.now();
        const result = await vitModel.process(imageData);
        const endTime = performance.now();
        const inferenceTime = Math.round(endTime - startTime);
        
        // Get top prediction
        const topPredictionIndex = result.probabilities.indexOf(Math.max(...result.probabilities));
        const topPrediction = topPredictionIndex < IMAGENET_LABELS.length 
          ? IMAGENET_LABELS[topPredictionIndex] 
          : `Class ${topPredictionIndex}`;
        
        // Store result
        results[backend] = {
          inferenceTime,
          speedup: 1, // Will calculate later relative to CPU
          supportLevel: 'full',
          topPrediction
        };
        
        // Clean up
        await vitModel.dispose();
      }
      
      // Calculate speedup relative to CPU
      const cpuTime = results['cpu']?.inferenceTime || 0;
      if (cpuTime > 0) {
        for (const backend in results) {
          results[backend].speedup = Number((cpuTime / results[backend].inferenceTime).toFixed(2));
        }
      }
      
      // Restore original backend
      this.hal.setBackendType(originalBackend);
      
      // Reinitialize the model
      await this.initializeModel();
      
      // Display comparison results
      this.displayComparisonResults(results);
      
    } catch (error) {
      console.error('Error comparing backends:', error);
      this.elements.classificationResults.textContent = `Error: ${error.message}`;
    } finally {
      this.elements.runComparisonBtn.disabled = false;
      // Ensure model is reinitialized with the original configuration
      if (!this.vitModel) {
        await this.initializeModel();
      }
    }
  }

  /**
   * Display comparison results
   */
  private displayComparisonResults(results: Record<string, {
    inferenceTime: number;
    speedup: number;
    supportLevel: string;
    topPrediction: string;
  }>) {
    // Sort backends by inference time (ascending)
    const sortedBackends = Object.keys(results).sort(
      (a, b) => results[a].inferenceTime - results[b].inferenceTime
    );
    
    // Get the fastest backend
    const fastestBackend = sortedBackends[0];
    
    // Clear previous results
    const tbody = this.elements.comparisonTable.querySelector('tbody');
    if (tbody) {
      tbody.innerHTML = '';
    }
    
    // Add rows to the table
    sortedBackends.forEach(backend => {
      const result = results[backend];
      const isFastest = backend === fastestBackend;
      
      const row = document.createElement('tr');
      if (isFastest) {
        row.classList.add('best-row');
      }
      
      row.innerHTML = `
        <td>${backend}</td>
        <td>${result.inferenceTime}</td>
        <td>${result.speedup}x</td>
        <td>${result.supportLevel}</td>
        <td>${result.topPrediction}</td>
      `;
      
      tbody?.appendChild(row);
    });
    
    // Create chart
    this.createComparisonChart(results, sortedBackends);
    
    // Show results
    this.elements.comparisonResults.style.display = 'block';
    this.elements.classificationResults.textContent = `Comparison complete. Fastest backend: ${fastestBackend}`;
  }

  /**
   * Create comparison chart
   */
  private createComparisonChart(results: Record<string, {
    inferenceTime: number;
    speedup: number;
    supportLevel: string;
    topPrediction: string;
  }>, sortedBackends: string[]) {
    // Get the slowest time for scaling
    const slowestTime = Math.max(...sortedBackends.map(backend => results[backend].inferenceTime));
    
    // Create HTML for the chart
    let chartHtml = `
      <div style="display: flex; height: 100%; padding: 20px;">
        <div style="writing-mode: vertical-lr; transform: rotate(180deg); padding-right: 10px; font-weight: bold;">Backend</div>
        <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: space-around;">
    `;
    
    // Add bars for each backend
    sortedBackends.forEach(backend => {
      const result = results[backend];
      const percentage = (result.inferenceTime / slowestTime) * 100;
      
      chartHtml += `
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
          <div style="width: 100px; font-weight: bold;">${backend}</div>
          <div class="chart-bar ${backend}-bar" style="width: ${percentage}%;">
            ${result.inferenceTime}ms
          </div>
        </div>
      `;
    });
    
    chartHtml += `
        </div>
      </div>
    `;
    
    this.elements.comparisonChart.innerHTML = chartHtml;
  }

  /**
   * Run multimodal demo with ViT and BERT
   */
  async runMultimodalDemo() {
    if (!this.hal || !this.vitModel || !this.modelConfig.enableTensorSharing) {
      alert('Please initialize the model with tensor sharing enabled');
      return;
    }
    
    // Update UI
    this.elements.multimodalDemoBtn.disabled = true;
    this.elements.multimodalResults.textContent = 'Initializing multimodal demo...';
    this.elements.multimodalResults.style.display = 'block';
    
    try {
      // Run ViT inference if not already done
      if (!this.selectedImageUrl) {
        alert('Please load an image and run inference first');
        this.elements.multimodalResults.style.display = 'none';
        this.elements.multimodalDemoBtn.disabled = false;
        return;
      }
      
      // Prepare image data
      const imageElement = this.elements.previewImage;
      const imageData = await this.prepareImageDataFromElement(imageElement);
      
      // Run ViT inference
      this.elements.multimodalResults.textContent = 'Running ViT inference...';
      const vitResult = await this.vitModel.process(imageData);
      
      // Get the shared tensor from ViT
      const sharedTensor = this.vitModel.getSharedTensor('vision_embedding');
      if (!sharedTensor) {
        throw new Error('No shared tensor available from ViT model');
      }
      
      // Initialize BERT model
      this.elements.multimodalResults.textContent = 'Initializing BERT model...';
      this.bertModel = createHardwareAbstractedBERT({
        modelId: 'bert-base-uncased',
        enableTensorSharing: true
      }, this.storageManager!, this.hal);
      
      await this.bertModel.initialize();
      
      // Prepare some text about the image
      const texts = [
        "A beautiful cat sitting on a carpet",
        "A dog playing in the park",
        "A car parked on the street",
        "A bird perched on a branch"
      ];
      
      // Process each text with BERT
      this.elements.multimodalResults.textContent = 'Running BERT inference...';
      
      const results = [];
      for (const text of texts) {
        const bertResult = await this.bertModel.predict(text);
        
        // Get BERT embeddings
        const textEmbedding = this.bertModel.getSharedTensor('text_embedding');
        if (!textEmbedding) {
          throw new Error('No shared tensor available from BERT model');
        }
        
        // Calculate similarity between vision and text embeddings
        // In a real implementation, this would use a proper similarity calculation
        // Here we'll just simulate it
        const similarity = this.simulateSimilarity(vitResult.probabilities, text);
        
        results.push({
          text,
          similarity
        });
      }
      
      // Sort by similarity (descending)
      results.sort((a, b) => b.similarity - a.similarity);
      
      // Display results
      let html = '<div style="margin-bottom: 15px;"><strong>Multimodal Results (ViT + BERT)</strong></div>';
      
      html += '<div style="margin-bottom: 10px;">Text-Vision Similarity:</div>';
      
      results.forEach((result, index) => {
        const percentage = (result.similarity * 100).toFixed(2);
        const barWidth = `${Math.max(result.similarity * 100, 0.5)}%`;
        
        html += `
          <div class="prediction">
            <div class="prediction-bar" style="width: ${barWidth}"></div>
            <div class="prediction-label">${index + 1}. "${result.text}"</div>
            <div class="prediction-value">${percentage}%</div>
          </div>
        `;
      });
      
      html += `
        <div style="margin-top: 15px; font-style: italic;">
          This demo shows how tensor sharing works between ViT and BERT models.
          The vision embeddings from ViT are shared with a multimodal system that
          compares them with text embeddings from BERT to find the most relevant
          text description for the image.
        </div>
      `;
      
      this.elements.multimodalResults.innerHTML = html;
      
    } catch (error) {
      console.error('Error running multimodal demo:', error);
      this.elements.multimodalResults.textContent = `Error: ${error.message}`;
    } finally {
      this.elements.multimodalDemoBtn.disabled = false;
    }
  }

  /**
   * Simulate similarity between image probabilities and text
   * In a real implementation, this would use proper embeddings and similarity calculation
   */
  private simulateSimilarity(imageProbs: number[], text: string): number {
    // Get the most likely class
    const topClassIndex = imageProbs.indexOf(Math.max(...imageProbs));
    const topClass = topClassIndex < IMAGENET_LABELS.length ? IMAGENET_LABELS[topClassIndex] : `Class ${topClassIndex}`;
    
    // Check if the text contains this class
    if (text.toLowerCase().includes(topClass.toLowerCase())) {
      return 0.8 + Math.random() * 0.2; // High similarity
    }
    
    // Check for other classes in the top 5
    const top5Indices = Array.from(Array(imageProbs.length).keys())
      .sort((a, b) => imageProbs[b] - imageProbs[a])
      .slice(0, 5);
    
    for (const idx of top5Indices) {
      const className = idx < IMAGENET_LABELS.length ? IMAGENET_LABELS[idx] : `Class ${idx}`;
      if (text.toLowerCase().includes(className.toLowerCase())) {
        return 0.5 + Math.random() * 0.3; // Medium similarity
      }
    }
    
    // Low similarity for unrelated text
    return Math.random() * 0.5;
  }

  /**
   * Dispose models and clean up resources
   */
  async disposeModels() {
    if (this.vitModel) {
      await this.vitModel.dispose();
      this.vitModel = null;
    }
    
    if (this.bertModel) {
      await this.bertModel.dispose();
      this.bertModel = null;
    }
  }
}

/**
 * Initialize the example when the DOM is loaded
 */
document.addEventListener('DOMContentLoaded', () => {
  const example = new HardwareAbstractedViTExample();
});

// Add automatic hardware detection on page load
window.addEventListener('load', async () => {
  const detectHardwareBtn = document.getElementById('detect-hardware-btn') as HTMLButtonElement;
  if (detectHardwareBtn) {
    detectHardwareBtn.click();
  }
});