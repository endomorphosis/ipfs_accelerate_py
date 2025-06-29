/**
 * Hardware Abstracted CLIP Example
 * 
 * This example demonstrates how to use the Hardware Abstracted CLIP model
 * with automatic hardware acceleration selection and optimization.
 */

import { createHardwareAbstractedCLIP } from '../../../src/model/hardware/clip';
import { createInMemoryStorageManager } from '../../../src/storage/in_memory_storage_manager';
import { createHardwareAbstractionLayer } from '../../../src/hardware/hardware_abstraction_layer';
import { createCPUBackend } from '../../../src/hardware/cpu/backend';
import { createWebGPUBackend } from '../../../src/hardware/webgpu/backend';
import { createWebNNBackend } from '../../../src/hardware/webnn/backend';

// Main demo function
async function runDemo() {
  // Display status
  const statusElement = document.getElementById('status');
  const resultsElement = document.getElementById('results');
  const imagePreviewElement = document.getElementById('image-preview') as HTMLImageElement;
  
  if (!statusElement || !resultsElement || !imagePreviewElement) {
    console.error('UI elements not found');
    return;
  }
  
  statusElement.textContent = 'Initializing CLIP model...';
  
  try {
    // Create backends
    const backends = [];
    
    // WebGPU backend
    try {
      const webgpuBackend = await createWebGPUBackend();
      backends.push(webgpuBackend);
      console.log('WebGPU backend created');
    } catch (e) {
      console.warn('WebGPU not available:', e);
    }
    
    // WebNN backend
    try {
      const webnnBackend = await createWebNNBackend();
      backends.push(webnnBackend);
      console.log('WebNN backend created');
    } catch (e) {
      console.warn('WebNN not available:', e);
    }
    
    // CPU backend (always available as fallback)
    const cpuBackend = await createCPUBackend();
    backends.push(cpuBackend);
    console.log('CPU backend created');
    
    // Create hardware abstraction layer
    const hal = createHardwareAbstractionLayer({
      backends,
      autoInitialize: true,
      useBrowserOptimizations: true,
      enableTensorSharing: true,
      enableOperationFusion: true
    });
    
    // Create storage manager
    const storageManager = createInMemoryStorageManager();
    await storageManager.initialize();
    
    // Create CLIP model
    const clip = createHardwareAbstractedCLIP({
      modelId: 'openai/clip-vit-base-patch32',
      imageSize: 224,
      taskType: 'similarity',
      allowFallback: true,
      collectMetrics: true,
      browserOptimizations: true
    }, storageManager);
    
    // Initialize model
    await clip.initialize();
    
    // Display backend information
    const backendInfo = clip.getBackendMetrics();
    statusElement.textContent = `Model initialized successfully using ${backendInfo.type} backend`;
    
    // Set up image selection
    const imageInput = document.getElementById('image-input') as HTMLInputElement;
    if (imageInput) {
      imageInput.addEventListener('change', async (event) => {
        const files = (event.target as HTMLInputElement).files;
        if (files && files.length > 0) {
          handleImageSelection(files[0], clip, imagePreviewElement, resultsElement);
        }
      });
    }
    
    // Set up text input
    const textInput = document.getElementById('text-input') as HTMLInputElement;
    const analyzeButton = document.getElementById('analyze-button');
    
    if (textInput && analyzeButton) {
      analyzeButton.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (text && imagePreviewElement.src) {
          await computeSimilarity(imagePreviewElement, text, clip, resultsElement);
        } else {
          alert('Please select an image and enter text to analyze');
        }
      });
    }
    
    // Set up zero-shot classification
    const classesInput = document.getElementById('classes-input') as HTMLInputElement;
    const classifyButton = document.getElementById('classify-button');
    
    if (classesInput && classifyButton) {
      classifyButton.addEventListener('click', async () => {
        const classesText = classesInput.value.trim();
        if (classesText && imagePreviewElement.src) {
          const classes = classesText.split(',').map(c => c.trim()).filter(c => c.length > 0);
          if (classes.length > 0) {
            await classifyImage(imagePreviewElement, classes, clip, resultsElement);
          } else {
            alert('Please enter comma-separated class names');
          }
        } else {
          alert('Please select an image and enter classes to classify');
        }
      });
    }
    
    // Set up performance benchmark
    const benchmarkButton = document.getElementById('benchmark-button');
    if (benchmarkButton) {
      benchmarkButton.addEventListener('click', async () => {
        if (imagePreviewElement.src) {
          await runBenchmark(imagePreviewElement, clip, resultsElement);
        } else {
          alert('Please select an image first');
        }
      });
    }
    
    console.log('CLIP demo initialized');
  } catch (error) {
    console.error('Failed to initialize CLIP model:', error);
    statusElement.textContent = `Error: ${error.message}`;
  }
}

// Handle image selection
async function handleImageSelection(
  file: File,
  clip: any,
  imagePreviewElement: HTMLImageElement,
  resultsElement: HTMLElement
) {
  // Clear previous results
  resultsElement.innerHTML = '<p>Loading image...</p>';
  
  try {
    // Load and display the image
    const imageUrl = URL.createObjectURL(file);
    imagePreviewElement.src = imageUrl;
    
    // Wait for image to load
    await new Promise<void>((resolve) => {
      imagePreviewElement.onload = () => {
        resolve();
      };
    });
    
    // Process image
    resultsElement.innerHTML = '<p>Image loaded successfully.</p>';
    
    // Get image encoding
    const startTime = performance.now();
    await clip.encodeImage(imagePreviewElement);
    const endTime = performance.now();
    
    resultsElement.innerHTML += `<p>Image encoded in ${(endTime - startTime).toFixed(2)}ms</p>`;
    
    // Show metrics
    const metrics = clip.getPerformanceMetrics();
    resultsElement.innerHTML += '<h3>Performance Metrics:</h3>';
    resultsElement.innerHTML += '<table><tr><th>Metric</th><th>Value</th></tr>';
    
    for (const [name, metric] of Object.entries(metrics)) {
      resultsElement.innerHTML += `<tr><td>${name}</td><td>${metric.avg.toFixed(2)}ms (min: ${metric.min.toFixed(2)}ms, max: ${metric.max.toFixed(2)}ms)</td></tr>`;
    }
    
    resultsElement.innerHTML += '</table>';
  } catch (error) {
    console.error('Error processing image:', error);
    resultsElement.innerHTML = `<p>Error: ${error.message}</p>`;
  }
}

// Compute similarity between image and text
async function computeSimilarity(
  imageElement: HTMLImageElement,
  text: string,
  clip: any,
  resultsElement: HTMLElement
) {
  resultsElement.innerHTML = '<p>Computing similarity...</p>';
  
  try {
    // Compute similarity
    const startTime = performance.now();
    const similarity = await clip.computeSimilarity(imageElement, text);
    const endTime = performance.now();
    
    // Display results
    resultsElement.innerHTML = `
      <h3>Similarity Results:</h3>
      <p>Image and text: "${text}"</p>
      <p>Similarity score: ${similarity.toFixed(4)}</p>
      <p>Computation time: ${(endTime - startTime).toFixed(2)}ms</p>
    `;
    
    // Show metrics
    const metrics = clip.getPerformanceMetrics();
    resultsElement.innerHTML += '<h3>Performance Metrics:</h3>';
    resultsElement.innerHTML += '<table><tr><th>Metric</th><th>Value</th></tr>';
    
    for (const [name, metric] of Object.entries(metrics)) {
      resultsElement.innerHTML += `<tr><td>${name}</td><td>${metric.avg.toFixed(2)}ms (min: ${metric.min.toFixed(2)}ms, max: ${metric.max.toFixed(2)}ms)</td></tr>`;
    }
    
    resultsElement.innerHTML += '</table>';
  } catch (error) {
    console.error('Error computing similarity:', error);
    resultsElement.innerHTML = `<p>Error: ${error.message}</p>`;
  }
}

// Classify image using zero-shot classification
async function classifyImage(
  imageElement: HTMLImageElement,
  classes: string[],
  clip: any,
  resultsElement: HTMLElement
) {
  resultsElement.innerHTML = '<p>Classifying image...</p>';
  
  try {
    // Classify image
    const startTime = performance.now();
    const classifications = await clip.classifyImage(imageElement, classes);
    const endTime = performance.now();
    
    // Display results
    resultsElement.innerHTML = `
      <h3>Classification Results:</h3>
      <p>Computation time: ${(endTime - startTime).toFixed(2)}ms</p>
      <table>
        <tr>
          <th>Class</th>
          <th>Score</th>
        </tr>
    `;
    
    for (const { label, score } of classifications) {
      resultsElement.innerHTML += `
        <tr>
          <td>${label}</td>
          <td>${score.toFixed(4)}</td>
        </tr>
      `;
    }
    
    resultsElement.innerHTML += '</table>';
    
    // Show metrics
    const metrics = clip.getPerformanceMetrics();
    resultsElement.innerHTML += '<h3>Performance Metrics:</h3>';
    resultsElement.innerHTML += '<table><tr><th>Metric</th><th>Value</th></tr>';
    
    for (const [name, metric] of Object.entries(metrics)) {
      resultsElement.innerHTML += `<tr><td>${name}</td><td>${metric.avg.toFixed(2)}ms (min: ${metric.min.toFixed(2)}ms, max: ${metric.max.toFixed(2)}ms)</td></tr>`;
    }
    
    resultsElement.innerHTML += '</table>';
  } catch (error) {
    console.error('Error classifying image:', error);
    resultsElement.innerHTML = `<p>Error: ${error.message}</p>`;
  }
}

// Run performance benchmark
async function runBenchmark(
  imageElement: HTMLImageElement,
  clip: any,
  resultsElement: HTMLElement
) {
  resultsElement.innerHTML = '<p>Running benchmark...</p>';
  
  try {
    // Prepare image input
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      throw new Error('Failed to get canvas context');
    }
    
    // Draw image to canvas
    ctx.drawImage(imageElement, 0, 0, 224, 224);
    
    // Get pixel data
    const imageData = ctx.getImageData(0, 0, 224, 224);
    
    // Convert to ClipImageInput
    const clipImageInput = {
      imageData: new Uint8Array(imageData.data),
      width: imageData.width,
      height: imageData.height
    };
    
    // Benchmark text
    const benchmarkText = 'a photo of a dog';
    
    // Run backend comparison
    const comparisonResults = await clip.compareBackends(clipImageInput, benchmarkText);
    
    // Display results
    resultsElement.innerHTML = `
      <h3>Backend Comparison:</h3>
      <table>
        <tr>
          <th>Backend</th>
          <th>Time (ms)</th>
        </tr>
    `;
    
    for (const [backend, time] of Object.entries(comparisonResults)) {
      resultsElement.innerHTML += `
        <tr>
          <td>${backend}</td>
          <td>${time < 0 ? 'Failed' : time.toFixed(2)}</td>
        </tr>
      `;
    }
    
    resultsElement.innerHTML += '</table>';
    
    // Show model info
    const modelInfo = clip.getModelInfo();
    resultsElement.innerHTML += '<h3>Model Information:</h3>';
    resultsElement.innerHTML += '<table><tr><th>Property</th><th>Value</th></tr>';
    
    for (const [property, value] of Object.entries(modelInfo)) {
      if (Array.isArray(value)) {
        resultsElement.innerHTML += `<tr><td>${property}</td><td>${value.join(', ')}</td></tr>`;
      } else {
        resultsElement.innerHTML += `<tr><td>${property}</td><td>${value}</td></tr>`;
      }
    }
    
    resultsElement.innerHTML += '</table>';
  } catch (error) {
    console.error('Error running benchmark:', error);
    resultsElement.innerHTML = `<p>Error: ${error.message}</p>`;
  }
}

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', runDemo);