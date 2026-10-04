/**
 * Example of using BERT with Hardware Abstraction Layer
 * Shows automatic backend selection and optimization
 */

import { HardwareAbstractionLayer, createHardwareAbstractionLayer } from '../../../src/hardware/hardware_abstraction_layer';
import { WebGPUBackend } from '../../../src/hardware/webgpu/backend';
import { WebNNBackend } from '../../../src/hardware/webnn/backend';
import { CPUBackend } from '../../../src/hardware/cpu/backend';
import { HardwareAbstractedBERT, createHardwareAbstractedBERT, StorageManager } from '../../../src/model/hardware/bert';
import { BrowserType } from '../../../src/hardware/webgpu/browser_optimized_operations';

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
 * Run BERT model with hardware abstraction layer
 */
async function runHardwareAbstractedBert() {
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
    
    console.log(`Initializing HAL with ${backends.length} backends:`);
    backends.forEach(backend => console.log(`- ${backend.type}`));
    
    // Create HAL
    const hal = createHardwareAbstractionLayer({
      backends,
      useBrowserOptimizations: true,
      enableTensorSharing: true,
      enableOperationFusion: true
    });
    
    // Initialize HAL
    await hal.initialize();
    
    // Storage manager for model weights
    const storageManager = new InMemoryStorageManager();
    await storageManager.initialize();
    
    // Create hardware abstracted BERT
    const model = createHardwareAbstractedBERT({
      modelId: 'bert-base-uncased',
      taskType: 'embedding',
      allowFallback: true,
      collectMetrics: true,
      browserOptimizations: true
    }, storageManager);
    
    // Initialize model
    console.log('Initializing hardware abstracted BERT model...');
    const initStart = performance.now();
    await model.initialize();
    const initTime = performance.now() - initStart;
    console.log(`Model initialized in ${initTime.toFixed(2)} ms`);
    
    // Get model info
    const modelInfo = model.getModelInfo();
    console.log('Model info:', modelInfo);
    
    // Get backend metrics
    const backendMetrics = model.getBackendMetrics();
    console.log('Using backend:', backendMetrics.type);
    
    // Example input text
    const inputText = "Hardware abstraction layers enable optimal performance across different devices and browsers.";
    
    // Run inference
    console.log(`\nProcessing text: "${inputText}"`);
    const inferenceStart = performance.now();
    const result = await model.predict(inputText);
    const inferenceTime = performance.now() - inferenceStart;
    console.log(`Inference completed in ${inferenceTime.toFixed(2)} ms`);
    
    // Display performance metrics
    const metrics = model.getPerformanceMetrics();
    console.log('\nPerformance metrics:');
    Object.entries(metrics).forEach(([name, metric]) => {
      console.log(`- ${name}: avg=${metric.avg.toFixed(2)}ms, min=${metric.min.toFixed(2)}ms, max=${metric.max.toFixed(2)}ms`);
    });
    
    // Compare performance across available backends
    console.log('\nRunning backend comparison...');
    const comparisonResults = await model.compareBackends(inputText);
    
    console.log('Backend comparison results:');
    Object.entries(comparisonResults).forEach(([backend, time]) => {
      if (time < 0) {
        console.log(`- ${backend}: Not available`);
      } else {
        console.log(`- ${backend}: ${time.toFixed(2)} ms`);
      }
    });
    
    // Cleanup
    await model.dispose();
    await hal.dispose();
    
    console.log('\nExample completed successfully');
    
  } catch (error) {
    console.error('Error:', error);
  }
}

// Run the example
runHardwareAbstractedBert();

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