/**
 * Hardware Abstraction Layer Example
 * Demonstrates how to use the hardware abstraction layer for executing operations
 * across different hardware backends (WebGPU, WebNN, CPU).
 * 
 * For detailed documentation, see HARDWARE_ABSTRACTION_LAYER_GUIDE.md
 */

import { createHardwareAbstraction, HardwareAbstraction } from './ipfs_accelerate_js_hardware_abstraction';

/**
 * Run matrix multiplication on the best available backend for the given model type
 */
async function runMatrixMultiplication(
  a: Float32Array,
  aShape: number[],
  b: Float32Array,
  bShape: number[],
  modelType: string = 'generic'
): Promise<{ result: Float32Array, shape: number[], backendType: string }> {
  console.log(`Running matrix multiplication for model type: ${modelType}`);
  
  // Create and initialize hardware abstraction layer
  const hal = await createHardwareAbstraction({
    // Customize backend order if needed
    backendOrder: ['webgpu', 'webnn', 'cpu'],
    // Enable auto fallback if preferred backend fails
    autoFallback: true,
    // Enable auto selection based on model type
    autoSelection: true
  });
  
  // Log available backends
  const availableBackends = hal.getAvailableBackends();
  console.log('Available backends:', availableBackends);
  
  // Get hardware capabilities
  const capabilities = hal.getCapabilities();
  console.log('Hardware capabilities:', capabilities);
  
  // Determine the best backend for the model type
  const bestBackend = hal.getBestBackend(modelType);
  console.log(`Best backend for ${modelType}:`, bestBackend?.type);
  
  try {
    // Execute the matrix multiplication operation
    const result = await hal.execute('matmul',
      { 
        a: { data: a, shape: aShape },
        b: { data: b, shape: bShape }
      },
      { modelType }
    );
    
    // Read the result data (this might involve copying from GPU memory)
    const resultData = new Float32Array(result.data);
    
    // Clean up resources
    hal.dispose();
    
    // Return the results and which backend was used
    return {
      result: resultData,
      shape: result.shape,
      backendType: bestBackend!.type
    };
  } catch (error) {
    console.error('Error executing operation:', error);
    hal.dispose();
    throw error;
  }
}

/**
 * Run a simple demonstration of the hardware abstraction layer
 */
async function runHardwareAbstractionDemo() {
  console.log('=== Hardware Abstraction Layer Demo ===');
  
  // Create sample matrices
  const matrixA = new Float32Array([1, 2, 3, 4, 5, 6]);
  const matrixAShape = [2, 3]; // 2 rows, 3 columns
  
  const matrixB = new Float32Array([7, 8, 9, 10, 11, 12, 13, 14, 15]);
  const matrixBShape = [3, 3]; // 3 rows, 3 columns
  
  // Test with different model types to see backend selection in action
  const modelTypes = ['vision', 'text', 'audio', 'generic'];
  
  for (const modelType of modelTypes) {
    try {
      console.log(`\nTesting with model type: ${modelType}`);
      
      const { result, shape, backendType } = await runMatrixMultiplication(
        matrixA,
        matrixAShape,
        matrixB,
        matrixBShape,
        modelType
      );
      
      console.log(`Operation executed on ${backendType} backend`);
      console.log('Result shape:', shape);
      console.log('Result data:', formatMatrix(result, shape));
    } catch (error) {
      console.error(`Error with ${modelType} model:`, error);
    }
  }
  
  console.log('\n=== Demo completed ===');
}

/**
 * Format a matrix for display
 */
function formatMatrix(data: Float32Array, shape: number[]): string {
  if (shape.length !== 2) {
    return `Data (non-matrix): ${data}`;
  }
  
  const rows = shape[0];
  const cols = shape[1];
  
  let result = '';
  for (let i = 0; i < rows; i++) {
    let row = '[ ';
    for (let j = 0; j < cols; j++) {
      row += data[i * cols + j].toFixed(1) + ' ';
    }
    row += ']';
    result += row + '\n';
  }
  
  return result;
}

// Execute the demo if this file is run directly
if (typeof window !== 'undefined') {
  // Browser environment
  window.addEventListener('DOMContentLoaded', () => {
    const demoButton = document.createElement('button');
    demoButton.textContent = 'Run Hardware Abstraction Demo';
    demoButton.addEventListener('click', runHardwareAbstractionDemo);
    document.body.appendChild(demoButton);
    
    const resultDiv = document.createElement('div');
    resultDiv.id = 'demo-results';
    resultDiv.style.whiteSpace = 'pre';
    resultDiv.style.fontFamily = 'monospace';
    document.body.appendChild(resultDiv);
    
    // Override console.log to display in the browser
    const originalLog = console.log;
    console.log = (...args) => {
      originalLog(...args);
      const resultDiv = document.getElementById('demo-results');
      if (resultDiv) {
        resultDiv.textContent += args.join(' ') + '\n';
      }
    };
  });
} else {
  // Node.js environment (or other non-browser JS environment)
  runHardwareAbstractionDemo();
}

// Export functions for use in other modules
export {
  runMatrixMultiplication,
  runHardwareAbstractionDemo
};