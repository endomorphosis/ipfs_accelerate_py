/**
 * WebGPU Tensor Example
 * Demonstrates using the WebGPU backend for tensor operations
 */

import { Tensor, ones, random } from '../tensor/tensor';
import { WebGPUBackend } from '../hardware/webgpu/backend';

/**
 * WebGPU Tensor Example
 * Runs basic tensor operations on the WebGPU backend
 */
export async function runWebGPUTensorExample() {
  // Check if WebGPU is available
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    console.error('WebGPU is not available in this environment');
    return {
      supported: false,
      error: 'WebGPU not available',
      results: null
    };
  }
  
  try {
    // Create and initialize WebGPU backend
    const backend = new WebGPUBackend();
    await backend.initialize();
    
    console.log('WebGPU backend initialized successfully');
    console.log('Capabilities:', backend.capabilities);
    
    // Create tensors with WebGPU backend
    const matrixSize = 1024; // Using 1024x1024 for benchmark
    
    console.log(`Creating ${matrixSize}x${matrixSize} matrices...`);
    
    // Create input matrices
    const matrixA = random(
      [matrixSize, matrixSize],
      -1,
      1,
      { backend: 'webgpu' }
    );
    
    const matrixB = random(
      [matrixSize, matrixSize],
      -1,
      1,
      { backend: 'webgpu' }
    );
    
    // Create tensors for testing basic operations
    const tensorA = random([1000], 0, 10, { backend: 'webgpu' });
    const tensorB = random([1000], 0, 10, { backend: 'webgpu' });
    
    console.log('Running tensor operations on WebGPU...');
    
    // Time the operations
    const results = {
      add: 0,
      multiply: 0,
      matmul: 0,
      relu: 0,
      sigmoid: 0,
      supported: true
    };
    
    // Test element-wise addition
    console.log('Testing element-wise addition...');
    const startAdd = performance.now();
    const addResult = await backend.add(tensorA, tensorB);
    results.add = performance.now() - startAdd;
    
    // Test element-wise multiplication
    console.log('Testing element-wise multiplication...');
    const startMul = performance.now();
    const mulResult = await backend.multiply(tensorA, tensorB);
    results.multiply = performance.now() - startMul;
    
    // Test matrix multiplication
    console.log('Testing matrix multiplication...');
    const startMatmul = performance.now();
    const matmulResult = await backend.matmul(matrixA, matrixB);
    results.matmul = performance.now() - startMatmul;
    
    // Test ReLU activation
    console.log('Testing ReLU activation...');
    const startRelu = performance.now();
    const reluResult = await backend.relu(tensorA);
    results.relu = performance.now() - startRelu;
    
    // Test sigmoid activation
    console.log('Testing sigmoid activation...');
    const startSigmoid = performance.now();
    const sigmoidResult = await backend.sigmoid(tensorA);
    results.sigmoid = performance.now() - startSigmoid;
    
    // Print results
    console.log('WebGPU Operation Benchmarks:');
    console.log(`- Element-wise addition: ${results.add.toFixed(2)}ms`);
    console.log(`- Element-wise multiplication: ${results.multiply.toFixed(2)}ms`);
    console.log(`- Matrix multiplication (${matrixSize}x${matrixSize}): ${results.matmul.toFixed(2)}ms`);
    console.log(`- ReLU activation: ${results.relu.toFixed(2)}ms`);
    console.log(`- Sigmoid activation: ${results.sigmoid.toFixed(2)}ms`);
    
    // Get buffer statistics
    const stats = backend.getStats();
    console.log('Buffer statistics:', stats);
    
    // Clean up
    backend.dispose();
    
    return {
      supported: true,
      results,
      stats
    };
  } catch (error) {
    console.error('WebGPU example failed:', error);
    return {
      supported: false,
      error: error instanceof Error ? error.message : String(error),
      results: null
    };
  }
}

// Run the example if in browser environment
if (typeof window !== 'undefined') {
  // Create output element
  const outputElement = document.createElement('div');
  outputElement.id = 'webgpu-output';
  document.body.appendChild(outputElement);
  
  // Run the example and update the output
  runWebGPUTensorExample().then(result => {
    if (result.supported) {
      outputElement.innerHTML = `
        <h2>WebGPU Tensor Operations</h2>
        <p>WebGPU is supported in this browser!</p>
        <h3>Performance Results:</h3>
        <ul>
          <li>Element-wise addition: ${result.results.add.toFixed(2)}ms</li>
          <li>Element-wise multiplication: ${result.results.multiply.toFixed(2)}ms</li>
          <li>Matrix multiplication (1024x1024): ${result.results.matmul.toFixed(2)}ms</li>
          <li>ReLU activation: ${result.results.relu.toFixed(2)}ms</li>
          <li>Sigmoid activation: ${result.results.sigmoid.toFixed(2)}ms</li>
        </ul>
        <h3>Buffer Statistics:</h3>
        <pre>${JSON.stringify(result.stats, null, 2)}</pre>
      `;
    } else {
      outputElement.innerHTML = `
        <h2>WebGPU Tensor Operations</h2>
        <p>WebGPU is not supported in this browser.</p>
        <p>Error: ${result.error}</p>
      `;
    }
  });
}