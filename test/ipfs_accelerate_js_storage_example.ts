/**
 * WebNN Storage Integration Example
 * Demonstrates how to use the WebNN storage integration to cache and load models
 */

import { WebNNBackend } from './ipfs_accelerate_js_webnn_backend';
import { WebNNStorageIntegration } from './ipfs_accelerate_js_webnn_storage_integration';

/**
 * Example of using the WebNN storage integration
 */
export async function runStorageExample(): Promise<void> {
  console.log('Starting WebNN Storage Integration Example...');
  
  // Create WebNN backend
  const backend = new WebNNBackend({
    enableLogging: true,
    deviceType: 'gpu',
    powerPreference: 'high-performance'
  });
  
  // Create storage integration
  const storage = new WebNNStorageIntegration(backend, {
    enableLogging: true,
    enableModelCaching: true,
    dbName: 'webnn-example-storage'
  });
  
  // Initialize storage integration
  console.log('Initializing storage integration...');
  const initialized = await storage.initialize();
  
  if (!initialized) {
    console.error('Failed to initialize storage integration');
    return;
  }
  
  // Check if WebNN is supported
  if (!await backend.isSupported()) {
    console.error('WebNN is not supported in this browser');
    return;
  }
  
  console.log('WebNN and storage initialized successfully');
  
  // Create a simple model with weights
  const modelId = 'simple-mlp';
  const modelName = 'Simple MLP Model';
  
  // Check if model is already cached
  const isCached = await storage.isModelCached(modelId);
  console.log(`Model "${modelName}" is${isCached ? '' : ' not'} cached`);
  
  if (isCached) {
    // If cached, load it
    console.log('Loading model from cache...');
    const tensors = await storage.loadModel(modelId);
    
    if (tensors) {
      console.log(`Loaded ${tensors.size} tensors from cache`);
      
      // Example usage: Run matrix multiplication with loaded weights
      if (tensors.has('weights') && tensors.has('bias')) {
        const input = await backend.createTensor(
          new Float32Array([1, 2, 3, 4]),
          [1, 4],
          'float32'
        );
        
        const weights = tensors.get('weights')!;
        const bias = tensors.get('bias')!;
        
        const result1 = await backend.execute('matmul', {
          a: input,
          b: weights
        });
        
        const result2 = await backend.execute('add', {
          a: result1,
          b: bias
        });
        
        // Read result
        const output = await backend.readTensor(result2.tensor, result2.shape);
        console.log('Inference result:', Array.from(output as Float32Array));
      }
    } else {
      console.error('Failed to load model from cache');
    }
  } else {
    // If not cached, create and store it
    console.log('Creating and storing model...');
    
    // Create weights
    const weights = new Map();
    
    // Layer 1: 4x2 weight matrix
    weights.set('weights', {
      data: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
      shape: [4, 2],
      dataType: 'float32' as const
    });
    
    // Layer 1: 1x2 bias
    weights.set('bias', {
      data: new Float32Array([0.1, 0.2]),
      shape: [1, 2],
      dataType: 'float32' as const
    });
    
    // Store model
    const stored = await storage.storeModel(
      modelId,
      modelName,
      weights,
      {
        version: '1.0.0',
        framework: 'custom',
        description: 'A simple MLP model for demonstration'
      }
    );
    
    if (stored) {
      console.log('Model stored successfully');
      
      // Now load the stored model
      console.log('Loading model from cache...');
      const tensors = await storage.loadModel(modelId);
      
      if (tensors) {
        console.log(`Loaded ${tensors.size} tensors from cache`);
        
        // Example usage: Run matrix multiplication with loaded weights
        if (tensors.has('weights') && tensors.has('bias')) {
          const input = await backend.createTensor(
            new Float32Array([1, 2, 3, 4]),
            [1, 4],
            'float32'
          );
          
          const weights = tensors.get('weights')!;
          const bias = tensors.get('bias')!;
          
          const result1 = await backend.execute('matmul', {
            a: input,
            b: weights
          });
          
          const result2 = await backend.execute('add', {
            a: result1,
            b: bias
          });
          
          // Read result
          const output = await backend.readTensor(result2.tensor, result2.shape);
          console.log('Inference result:', Array.from(output as Float32Array));
        }
      } else {
        console.error('Failed to load model from cache');
      }
    } else {
      console.error('Failed to store model');
    }
  }
  
  // Show storage statistics
  const stats = await storage.getStorageStats();
  console.log('Storage statistics:', stats);
  
  // List all models
  const models = await storage.listModels();
  console.log('Available models:');
  for (const model of models) {
    console.log(`- ${model.name} (${model.id}, v${model.version}, ${formatBytes(model.size)})`);
  }
  
  console.log('Storage example completed');
}

/**
 * Format bytes into a human-readable string
 */
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Example for browser usage
if (typeof window !== 'undefined') {
  // Add a button to run the example
  window.addEventListener('DOMContentLoaded', () => {
    const button = document.createElement('button');
    button.textContent = 'Run WebNN Storage Example';
    button.style.padding = '10px';
    button.style.margin = '10px';
    button.style.fontSize = '16px';
    
    button.addEventListener('click', async () => {
      const output = document.createElement('pre');
      output.style.margin = '10px';
      output.style.padding = '10px';
      output.style.border = '1px solid #ccc';
      output.style.borderRadius = '5px';
      output.style.backgroundColor = '#f8f8f8';
      output.style.maxHeight = '400px';
      output.style.overflow = 'auto';
      
      document.body.appendChild(output);
      
      // Capture console.log output
      const originalLog = console.log;
      const originalError = console.error;
      
      console.log = (...args) => {
        originalLog(...args);
        output.textContent += args.map(arg => 
          typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
        ).join(' ') + '\n';
      };
      
      console.error = (...args) => {
        originalError(...args);
        output.textContent += 'ERROR: ' + args.map(arg => 
          typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
        ).join(' ') + '\n';
      };
      
      try {
        await runStorageExample();
      } catch (error) {
        console.error('Example failed:', error);
      } finally {
        // Restore original console functions
        console.log = originalLog;
        console.error = originalError;
      }
    });
    
    document.body.appendChild(button);
  });
}