/**
 * Browser-Specific Layer Normalization Test
 * Tests the performance of browser-specific Layer Normalization operations
 */

import { WebGPUBackend } from '../src/hardware/webgpu/backend';
import { Tensor } from '../src/tensor/tensor';
import { 
  BrowserType, 
  detectBrowserType 
} from '../src/hardware/webgpu/browser_optimized_operations';
import {
  loadBrowserShader,
  ShaderType
} from '../src/hardware/webgpu/optimizations/browser_shader_loader';

/**
 * Generate a random tensor with specified shape
 */
function generateRandomTensor(shape: number[]): Tensor<number> {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = Array(size).fill(0).map(() => Math.random() * 2 - 1); // Range [-1, 1]
  return new Tensor<number>(shape, data, { dataType: 'float32' });
}

/**
 * Generate random gamma and beta parameters for Layer Normalization
 */
function generateLayerNormParameters(hiddenSize: number): { gamma: Tensor<number>, beta: Tensor<number> } {
  // For gamma, values around 1 are common
  const gammaData = Array(hiddenSize).fill(0).map(() => 0.9 + Math.random() * 0.2); // [0.9, 1.1]
  
  // For beta, values around 0 are common
  const betaData = Array(hiddenSize).fill(0).map(() => Math.random() * 0.2 - 0.1); // [-0.1, 0.1]
  
  return {
    gamma: new Tensor<number>([hiddenSize], gammaData, { dataType: 'float32' }),
    beta: new Tensor<number>([hiddenSize], betaData, { dataType: 'float32' })
  };
}

/**
 * CPU implementation of Layer Normalization
 * Normalizes across the last dimension of the input tensor
 */
function cpuLayerNorm(
  input: Tensor<number>, 
  gamma: Tensor<number>, 
  beta: Tensor<number>,
  epsilon: number = 1e-5
): Tensor<number> {
  const inputShape = input.shape;
  const hiddenSize = inputShape[inputShape.length - 1];
  const batchSize = input.size / hiddenSize;
  
  const output = new Float32Array(input.size);
  
  // Process each batch element independently
  for (let batch = 0; batch < batchSize; batch++) {
    const startIdx = batch * hiddenSize;
    
    // Calculate mean
    let mean = 0;
    for (let i = 0; i < hiddenSize; i++) {
      mean += input.data[startIdx + i];
    }
    mean /= hiddenSize;
    
    // Calculate variance
    let variance = 0;
    for (let i = 0; i < hiddenSize; i++) {
      const diff = input.data[startIdx + i] - mean;
      variance += diff * diff;
    }
    variance /= hiddenSize;
    
    // Apply normalization, scaling, and shifting
    const invStd = 1 / Math.sqrt(variance + epsilon);
    for (let i = 0; i < hiddenSize; i++) {
      const normalized = (input.data[startIdx + i] - mean) * invStd;
      output[startIdx + i] = normalized * gamma.data[i] + beta.data[i];
    }
  }
  
  return new Tensor<number>(inputShape, Array.from(output), { dataType: 'float32' });
}

/**
 * Check if two tensors are approximately equal (within epsilon)
 */
function areTensorsEqual(a: Tensor<number>, b: Tensor<number>, epsilon: number = 1e-5): boolean {
  if (a.shape.join(',') !== b.shape.join(',')) {
    console.error('Shape mismatch:', a.shape, 'vs', b.shape);
    return false;
  }
  
  let maxDifference = 0;
  let numDifferences = 0;
  
  for (let i = 0; i < a.data.length; i++) {
    const diff = Math.abs(a.data[i] - b.data[i]);
    if (diff > epsilon) {
      numDifferences++;
      maxDifference = Math.max(maxDifference, diff);
      
      // Print some sample differences for debugging
      if (numDifferences <= 5) {
        console.error(`Difference at index ${i}: ${a.data[i]} vs ${b.data[i]}, diff = ${diff}`);
      }
    }
  }
  
  if (numDifferences > 0) {
    console.error(`Found ${numDifferences} differences out of ${a.data.length} elements`);
    console.error(`Maximum difference: ${maxDifference}`);
    return false;
  }
  
  return true;
}

/**
 * Test Layer Normalization performance across browsers
 */
async function testLayerNormPerformance(
  backend: WebGPUBackend, 
  input: Tensor<number>,
  gamma: Tensor<number>,
  beta: Tensor<number>,
  epsilon: number = 1e-5
) {
  console.log('\n=== Testing Layer Normalization Performance ===');
  
  const actualBrowserType = detectBrowserType();
  const iterations = 10;
  const warmupIterations = 5;
  
  // Compute CPU reference result
  console.log('Computing CPU reference result...');
  const cpuResult = cpuLayerNorm(input, gamma, beta, epsilon);
  
  // Test each browser optimization
  const browserTypes = [
    { name: 'Generic (No Optimization)', type: null },
    { name: 'Chrome', type: BrowserType.CHROME },
    { name: 'Firefox', type: BrowserType.FIREFOX },
    { name: 'Safari', type: BrowserType.SAFARI },
    { name: 'Edge', type: BrowserType.EDGE }
  ];
  
  const results: Record<string, number> = {};
  
  // Get actual browser shader for later comparison
  await loadBrowserShader(ShaderType.LAYER_NORMALIZATION);
  
  // Test Layer Normalization implementations
  console.log('\nTesting Layer Normalization implementations:');
  for (const browser of browserTypes) {
    const name = browser.name;
    console.log(`Testing ${name} optimization...`);
    
    // Warmup
    for (let i = 0; i < warmupIterations; i++) {
      await backend.layerNorm(input, gamma, beta, {
        useBrowserOptimizations: browser.type !== null,
        browserType: browser.type,
        epsilon
      });
    }
    
    // Timed iterations
    const startTime = performance.now();
    let result;
    
    for (let i = 0; i < iterations; i++) {
      result = await backend.layerNorm(input, gamma, beta, {
        useBrowserOptimizations: browser.type !== null,
        browserType: browser.type,
        epsilon
      });
    }
    
    const endTime = performance.now();
    const avgTime = (endTime - startTime) / iterations;
    results[name] = avgTime;
    
    console.log(`${name}: ${avgTime.toFixed(2)} ms`);
    
    // Verify correctness
    const isCorrect = areTensorsEqual(cpuResult, result!, 1e-4);
    console.log(`${name} correctness: ${isCorrect ? 'PASSED' : 'FAILED'}`);
  }
  
  // Calculate speedups
  const baseTime = results['Generic (No Optimization)'];
  console.log('\nSpeedup compared to generic implementation:');
  
  for (const browser of browserTypes) {
    if (browser.type === null) continue;
    
    const speedup = baseTime / results[browser.name];
    console.log(`${browser.name}: ${speedup.toFixed(2)}x`);
    
    // Highlight current browser
    if (browser.type === actualBrowserType) {
      console.log(`Current browser (${browser.name}) speedup: ${speedup.toFixed(2)}x`);
    }
  }
  
  return results;
}

/**
 * Test performance with different hidden sizes
 */
async function testDifferentHiddenSizes(backend: WebGPUBackend) {
  console.log('\n=== Testing Different Hidden Sizes ===');
  
  const hiddenSizes = [256, 512, 768, 1024];
  const batchSize = 16;
  const sequenceLength = 32;
  
  for (const hiddenSize of hiddenSizes) {
    console.log(`\n--- Testing with hidden size ${hiddenSize} ---`);
    
    // Create input tensor [batch, seq_len, hidden]
    const input = generateRandomTensor([batchSize, sequenceLength, hiddenSize]);
    
    // Create gamma and beta parameters
    const { gamma, beta } = generateLayerNormParameters(hiddenSize);
    
    // Run performance test
    await testLayerNormPerformance(backend, input, gamma, beta);
  }
}

/**
 * Test fusion of Layer Normalization with other operations
 */
async function testLayerNormFusion(backend: WebGPUBackend, input: Tensor<number>, gamma: Tensor<number>, beta: Tensor<number>) {
  console.log('\n=== Testing Layer Normalization Fusion Performance ===');
  
  const actualBrowserType = detectBrowserType();
  const iterations = 10;
  const warmupIterations = 5;
  const epsilon = 1e-5;
  
  // Test common sequence: MatMul followed by LayerNorm (common in transformer models)
  console.log('Testing MatMul+LayerNorm fusion...');
  
  // Create weights for MatMul
  const inputShape = input.shape;
  const hiddenSize = inputShape[inputShape.length - 1];
  const weights = generateRandomTensor([hiddenSize, hiddenSize]);
  
  // Compute CPU reference result
  console.log('Computing CPU reference result...');
  
  // MatMul on CPU (very simplified - just for testing)
  const batchSize = input.size / hiddenSize;
  const cpuMatMulResult = new Float32Array(input.size);
  
  for (let batch = 0; batch < batchSize; batch++) {
    const startIdx = batch * hiddenSize;
    
    for (let i = 0; i < hiddenSize; i++) {
      let sum = 0;
      for (let j = 0; j < hiddenSize; j++) {
        sum += input.data[startIdx + j] * weights.data[j * hiddenSize + i];
      }
      cpuMatMulResult[startIdx + i] = sum;
    }
  }
  
  const cpuMatMulTensor = new Tensor<number>(inputShape, Array.from(cpuMatMulResult), { dataType: 'float32' });
  
  // LayerNorm on CPU
  const cpuLayerNormResult = cpuLayerNorm(cpuMatMulTensor, gamma, beta, epsilon);
  
  // Unfused execution (separate operations)
  console.log('Running unfused operations (separate MatMul and LayerNorm)...');
  
  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    const matMulResult = await backend.matmul(input, weights, {
      useBrowserOptimizations: true
    });
    await backend.layerNorm(matMulResult, gamma, beta, {
      useBrowserOptimizations: true,
      epsilon
    });
  }
  
  // Timed iterations
  const startTimeUnfused = performance.now();
  let unfusedResult;
  
  for (let i = 0; i < iterations; i++) {
    const matMulResult = await backend.matmul(input, weights, {
      useBrowserOptimizations: true
    });
    unfusedResult = await backend.layerNorm(matMulResult, gamma, beta, {
      useBrowserOptimizations: true,
      epsilon
    });
  }
  
  const endTimeUnfused = performance.now();
  const avgTimeUnfused = (endTimeUnfused - startTimeUnfused) / iterations;
  
  console.log(`Unfused execution: ${avgTimeUnfused.toFixed(2)} ms`);
  
  // Verify correctness
  const isUnfusedCorrect = areTensorsEqual(cpuLayerNormResult, unfusedResult!, 1e-4);
  console.log(`Unfused correctness: ${isUnfusedCorrect ? 'PASSED' : 'FAILED'}`);
  
  // Fused execution (executeOperations)
  console.log('Running fused operations (MatMul+LayerNorm combined)...');
  
  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    await backend.executeOperations(
      [input, weights, gamma, beta],
      ['matmul', 'layerNorm'],
      { 
        useBrowserOptimizations: true,
        useFusion: true,
        layerNormOptions: { epsilon }
      }
    );
  }
  
  // Timed iterations
  const startTimeFused = performance.now();
  let fusedResult;
  
  for (let i = 0; i < iterations; i++) {
    fusedResult = await backend.executeOperations(
      [input, weights, gamma, beta],
      ['matmul', 'layerNorm'],
      { 
        useBrowserOptimizations: true,
        useFusion: true,
        layerNormOptions: { epsilon }
      }
    );
  }
  
  const endTimeFused = performance.now();
  const avgTimeFused = (endTimeFused - startTimeFused) / iterations;
  
  console.log(`Fused execution: ${avgTimeFused.toFixed(2)} ms`);
  
  // Verify correctness
  const isFusedCorrect = areTensorsEqual(cpuLayerNormResult, fusedResult!, 1e-4);
  console.log(`Fused correctness: ${isFusedCorrect ? 'PASSED' : 'FAILED'}`);
  
  // Calculate speedup
  const fusionSpeedup = avgTimeUnfused / avgTimeFused;
  console.log(`Fusion speedup: ${fusionSpeedup.toFixed(2)}x`);
  
  return {
    unfusedTime: avgTimeUnfused,
    fusedTime: avgTimeFused,
    speedup: fusionSpeedup
  };
}

/**
 * Main function to run all tests
 */
async function main() {
  console.log('Browser-Specific Layer Normalization Test');
  console.log('========================================');
  
  const browserType = detectBrowserType();
  const browserNames = {
    [BrowserType.CHROME]: 'Google Chrome',
    [BrowserType.FIREFOX]: 'Mozilla Firefox',
    [BrowserType.SAFARI]: 'Apple Safari',
    [BrowserType.EDGE]: 'Microsoft Edge',
    [BrowserType.UNKNOWN]: 'Unknown Browser'
  };
  
  console.log(`Detected browser: ${browserNames[browserType]}`);
  
  try {
    // Initialize WebGPU backend
    console.log('Initializing WebGPU backend...');
    const backend = new WebGPUBackend();
    await backend.initialize();
    
    // Create test tensors for a typical transformer layer
    console.log('Creating test tensors...');
    const batchSize = 16;
    const sequenceLength = 32;
    const hiddenSize = 768; // Common size in BERT/ViT models
    
    // Input tensor [batch, seq_len, hidden]
    const input = generateRandomTensor([batchSize, sequenceLength, hiddenSize]);
    
    // Layer Normalization parameters
    const { gamma, beta } = generateLayerNormParameters(hiddenSize);
    
    // Run standard performance test
    await testLayerNormPerformance(backend, input, gamma, beta);
    
    // Test with different hidden sizes
    await testDifferentHiddenSizes(backend);
    
    // Test fusion with MatMul
    await testLayerNormFusion(backend, input, gamma, beta);
    
    // Cleanup
    await backend.dispose();
    
  } catch (error) {
    console.error('Error during tests:', error);
  }
}

// Check if WebGPU is available and run tests
if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
  console.log('WebGPU is available. Running tests...');
  main();
} else {
  console.error('WebGPU is not supported in this environment. Tests cannot run.');
}