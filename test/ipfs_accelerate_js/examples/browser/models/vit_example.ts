/**
 * Vision Transformer (ViT) Demo Example
 * 
 * This example demonstrates how to use the ViT model with WebGPU/WebNN
 * hardware acceleration in a browser environment.
 */

import { 
  createVitModel, 
  ViT, 
  ViTConfig,
  ViTInput, 
  ViTOutput 
} from '../../../src/model/vision/vit';
import { createWebGPUBackend } from '../../../src/hardware/webgpu/backend';
import { createWebNNBackend } from '../../../src/hardware/webnn/backend';
import { getBrowserHardwareCapabilities } from '../../../src/hardware/detection/hardware_detector';
import { TensorBackendType } from '../../../src/tensor';

// Define ImageNet classes (top 25 for brevity)
const IMAGENET_CLASSES = [
  'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark',
  'electric ray', 'stingray', 'rooster', 'hen', 'ostrich', 'brambling',
  'goldfinch', 'house finch', 'junco', 'indigo bunting', 'American robin',
  'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel', 'kite', 'bald eagle',
  'vulture', 'great grey owl', /* ...more classes would go here */ 
];

// Main application class
class VitDemo {
  private model: ViT | null = null;
  private hardware: any = null;
  private imageElement: HTMLImageElement | null = null;
  private selectedImageUrl: string | null = null;
  private isProcessing = false;
  private startTime = 0;
  
  constructor() {
    this.initialize();
  }
  
  async initialize() {
    try {
      // Set up event listeners
      this.setupEventListeners();
      
      // Detect hardware capabilities
      const capabilities = await getBrowserHardwareCapabilities();
      
      // Display backend info
      document.getElementById('backend-info')!.textContent = 
        `Browser: ${capabilities.browser}, WebGPU: ${capabilities.webgpu ? 'Supported' : 'Not Supported'}, WebNN: ${capabilities.webnn ? 'Supported' : 'Not Supported'}`;
      
      // Determine best backend
      let backendType: TensorBackendType = 'cpu';
      if (capabilities.webgpu) {
        backendType = 'webgpu';
        this.hardware = await createWebGPUBackend();
      } else if (capabilities.webnn) {
        backendType = 'webnn';
        this.hardware = await createWebNNBackend();
      } else {
        this.showError('WebGPU and WebNN are not supported in this browser. Falling back to CPU backend.');
        // In a real implementation, we would create a CPU backend
        // For this demo, we'll just show an error
        return;
      }
      
      await this.hardware.initialize();
      
      // Create model configuration
      const config: Partial<ViTConfig> = {
        modelId: 'google/vit-base-patch16-224',
        backendPreference: [backendType, 'cpu'],
        useOptimizedOps: true
      };
      
      // Create model
      this.model = createVitModel(this.hardware, config);
      
      // Initialize model in background
      this.updateProgress(10);
      await this.model.initialize();
      this.updateProgress(100);
      
      console.log('ViT model initialized');
    } catch (error) {
      console.error('Error initializing:', error);
      this.showError(`Error initializing: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  
  setupEventListeners() {
    // File upload handling
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    const selectFileButton = document.getElementById('select-file') as HTMLButtonElement;
    const dropArea = document.getElementById('drop-area') as HTMLDivElement;
    
    selectFileButton.addEventListener('click', () => {
      fileInput.click();
    });
    
    fileInput.addEventListener('change', (e: Event) => {
      const target = e.target as HTMLInputElement;
      if (target.files && target.files[0]) {
        this.handleSelectedFile(target.files[0]);
      }
    });
    
    // Drag and drop handling
    dropArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropArea.style.borderColor = '#2196F3';
    });
    
    dropArea.addEventListener('dragleave', () => {
      dropArea.style.borderColor = '#ccc';
    });
    
    dropArea.addEventListener('drop', (e) => {
      e.preventDefault();
      dropArea.style.borderColor = '#ccc';
      
      if (e.dataTransfer?.files.length) {
        this.handleSelectedFile(e.dataTransfer.files[0]);
      }
    });
    
    // Sample images
    const sampleImages = document.querySelectorAll('.sample-image');
    sampleImages.forEach((img) => {
      img.addEventListener('click', () => {
        const imgElement = img.querySelector('img') as HTMLImageElement;
        this.selectedImageUrl = imgElement.src;
        this.loadAndDisplayImage(imgElement.src);
      });
    });
    
    // Tab switching
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const tabId = (tab as HTMLElement).dataset.tab;
        this.switchTab(tabId as string);
      });
    });
  }
  
  switchTab(tabId: string) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
      tab.classList.remove('active');
    });
    document.querySelector(`.tab[data-tab="${tabId}"]`)?.classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
      content.classList.remove('active');
    });
    document.getElementById(`${tabId}-tab`)?.classList.add('active');
    
    // If sample tab is selected, hide the preview in the upload tab
    if (tabId === 'sample') {
      const preview = document.getElementById('preview') as HTMLImageElement;
      preview.style.display = 'none';
    }
  }
  
  handleSelectedFile(file: File) {
    if (!file.type.startsWith('image/')) {
      this.showError('Please select an image file.');
      return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
      this.selectedImageUrl = e.target?.result as string;
      this.loadAndDisplayImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  }
  
  loadAndDisplayImage(url: string) {
    const preview = document.getElementById('preview') as HTMLImageElement;
    
    // Switch to upload tab if we're on a different tab
    this.switchTab('upload');
    
    // Create new image element for processing
    this.imageElement = new Image();
    this.imageElement.crossOrigin = 'anonymous';
    
    this.imageElement.onload = () => {
      // Display the image
      preview.src = url;
      preview.style.display = 'block';
      
      // Process the image
      this.processImage();
    };
    
    this.imageElement.onerror = () => {
      this.showError('Error loading image. Try another image or check the URL.');
    };
    
    this.imageElement.src = url;
  }
  
  async processImage() {
    if (!this.model || !this.imageElement || this.isProcessing) {
      return;
    }
    
    try {
      this.isProcessing = true;
      
      // Show loading indicator
      document.getElementById('loading')!.style.display = 'block';
      document.getElementById('results')!.style.display = 'none';
      document.getElementById('attention-visualization')!.style.display = 'none';
      
      // Reset progress
      this.updateProgress(0);
      
      // Preprocess image to 224x224 (ViT standard input size)
      const imageData = this.preprocessImage(this.imageElement, 224, 224);
      this.updateProgress(30);
      
      // Prepare input
      const input: ViTInput = {
        imageData: imageData,
        width: 224,
        height: 224,
        isPreprocessed: false // We're providing raw pixel data (0-255)
      };
      
      // Record start time for performance measurement
      this.startTime = performance.now();
      
      // Run inference
      const result = await this.model.process(input);
      this.updateProgress(90);
      
      // Calculate elapsed time
      const elapsed = performance.now() - this.startTime;
      
      // Display results
      this.displayResults(result, elapsed);
      
      // Hide loading indicator
      document.getElementById('loading')!.style.display = 'none';
      document.getElementById('results')!.style.display = 'block';
      
      // Show attention visualization
      this.visualizeAttention();
      
      this.isProcessing = false;
    } catch (error) {
      console.error('Error processing image:', error);
      this.showError(`Error processing image: ${error instanceof Error ? error.message : String(error)}`);
      document.getElementById('loading')!.style.display = 'none';
      this.isProcessing = false;
    }
  }
  
  preprocessImage(image: HTMLImageElement, targetWidth: number, targetHeight: number): Uint8Array {
    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d')!;
    
    // Draw image with resize
    ctx.drawImage(image, 0, 0, targetWidth, targetHeight);
    
    // Get image data (RGBA format)
    const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);
    
    // Convert to RGB format (remove alpha channel)
    const rgbData = new Uint8Array(targetWidth * targetHeight * 3);
    for (let i = 0; i < imageData.data.length / 4; i++) {
      rgbData[i * 3] = imageData.data[i * 4]; // R
      rgbData[i * 3 + 1] = imageData.data[i * 4 + 1]; // G
      rgbData[i * 3 + 2] = imageData.data[i * 4 + 2]; // B
    }
    
    return rgbData;
  }
  
  displayResults(result: ViTOutput, elapsed: number) {
    const topPredictionsElement = document.getElementById('top-predictions')!;
    const performanceElement = document.getElementById('performance-info')!;
    
    // Clear previous results
    topPredictionsElement.innerHTML = '';
    
    // Get top 5 predictions
    const indices = Array.from({ length: result.probabilities.length }, (_, i) => i);
    const sortedIndices = indices.sort((a, b) => result.probabilities[b] - result.probabilities[a]);
    const top5 = sortedIndices.slice(0, 5);
    
    // Display top 5 predictions
    top5.forEach(idx => {
      const probability = result.probabilities[idx] * 100;
      const className = idx < IMAGENET_CLASSES.length 
        ? IMAGENET_CLASSES[idx] 
        : `Class ${idx}`;
      
      const resultItem = document.createElement('div');
      resultItem.className = 'result-item';
      
      const classLabel = document.createElement('div');
      classLabel.textContent = className;
      
      const probabilityLabel = document.createElement('div');
      probabilityLabel.textContent = `${probability.toFixed(2)}%`;
      
      const progressBar = document.createElement('div');
      progressBar.className = 'progress-bar';
      
      const progressFill = document.createElement('div');
      progressFill.className = 'progress-fill';
      progressFill.style.width = `${probability}%`;
      
      progressBar.appendChild(progressFill);
      
      resultItem.appendChild(classLabel);
      resultItem.appendChild(probabilityLabel);
      topPredictionsElement.appendChild(resultItem);
      topPredictionsElement.appendChild(progressBar);
    });
    
    // Display performance information
    performanceElement.textContent = `Inference time: ${elapsed.toFixed(2)}ms, Backend: ${result.backend}`;
  }
  
  visualizeAttention() {
    // In a real implementation, we would visualize attention weights
    // For this demo, we'll just show a placeholder
    document.getElementById('attention-visualization')!.style.display = 'block';
    
    const canvas = document.getElementById('attention-canvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;
    
    // Draw the original image
    if (this.imageElement) {
      ctx.drawImage(this.imageElement, 0, 0, canvas.width, canvas.height);
      
      // Overlay with simulated attention heatmap
      // In a real implementation, this would use actual attention weights
      ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
      ctx.fillRect(canvas.width / 4, canvas.height / 4, canvas.width / 2, canvas.height / 2);
    }
  }
  
  updateProgress(percent: number) {
    const progressElement = document.getElementById('progress') as HTMLElement;
    progressElement.style.width = `${percent}%`;
  }
  
  showError(message: string) {
    const errorElement = document.getElementById('error-message')!;
    errorElement.textContent = message;
    errorElement.style.display = 'block';
    
    // Hide after 5 seconds
    setTimeout(() => {
      errorElement.style.display = 'none';
    }, 5000);
  }
}

// Initialize the demo when the page loads
window.addEventListener('DOMContentLoaded', () => {
  new VitDemo();
});