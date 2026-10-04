/**
 * Browser-Specific Leaky ReLU Operation Test
 * Tests the performance of browser-specific Leaky ReLU operations
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
  const data = Array(size).fill(0).map(() => Math.random() * 4 - 2); // Range [-2, 2] for Leaky ReLU testing
  return new Tensor<number>(shape, data, { dataType: 'float32' });
}

/**
 * Compute Leaky ReLU on CPU for reference
 * LeakyReLU(x) = max(alpha * x, x) where alpha is a small number like 0.01
 */
function cpuLeakyRelu(input: Tensor<number>, alpha: number = 0.01): Tensor<number> {
  const output = new Float32Array(input.data.length);
  
  for (let i = 0; i < input.data.length; i++) {
    const x = input.data[i];
    output[i] = Math.max(alpha * x, x);
  }
  return new Tensor<number>(input.shape, Array.from(output), { dataType: 'float32' });
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
 * Test elementwise Leaky ReLU performance across browsers with different alpha values
 */
async function testLeakyReluPerformance(backend: WebGPUBackend, input: Tensor<number>) {
  console.log('\n=== Testing Elementwise Leaky ReLU Performance ===');
  
  const actualBrowserType = detectBrowserType();
  const iterations = 10;
  const warmupIterations = 5;
  
  // Test different alpha values
  const alphaValues = [0.01, 0.1, 0.2];
  
  for (const alpha of alphaValues) {
    console.log(`\n--- Testing with alpha = ${alpha} ---`);
    
    // Compute CPU reference result
    console.log('Computing CPU reference result...');
    const cpuResult = cpuLeakyRelu(input, alpha);
    
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
    await loadBrowserShader(ShaderType.ELEMENTWISE_LEAKY_RELU);
    
    // Test Leaky ReLU implementations
    console.log('\nTesting Leaky ReLU implementations:');
    for (const browser of browserTypes) {
      const name = browser.name;
      console.log(`Testing ${name} optimization...`);
      
      // Warmup
      for (let i = 0; i < warmupIterations; i++) {
        await backend.leakyRelu(input, {
          useBrowserOptimizations: browser.type !== null,
          browserType: browser.type,
          alpha
        });
      }
      
      // Timed iterations
      const startTime = performance.now();
      let result;
      
      for (let i = 0; i < iterations; i++) {
        result = await backend.leakyRelu(input, {
          useBrowserOptimizations: browser.type !== null,
          browserType: browser.type,
          alpha
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
  }
}

/**
 * Test fusion of Leaky ReLU with other operations for performance
 */
async function testLeakyReluFusion(backend: WebGPUBackend, inputA: Tensor<number>, inputB: Tensor<number>) {
  console.log('\n=== Testing Leaky ReLU Fusion Performance ===');
  
  const actualBrowserType = detectBrowserType();
  const iterations = 10;
  const warmupIterations = 5;
  const alpha = 0.01; // Standard alpha value for Leaky ReLU
  
  // Test sequences: add followed by Leaky ReLU (common in CNNs)
  console.log('Testing Add+LeakyReLU fusion...');
  
  // Compute CPU reference result
  console.log('Computing CPU reference result...');
  // Add on CPU
  const cpuAddResult = new Float32Array(inputA.data.length);
  for (let i = 0; i < inputA.data.length; i++) {
    cpuAddResult[i] = inputA.data[i] + inputB.data[i];
  }
  const cpuAddTensor = new Tensor<number>(inputA.shape, Array.from(cpuAddResult), { dataType: 'float32' });
  
  // Leaky ReLU on CPU
  const cpuLeakyReluResult = cpuLeakyRelu(cpuAddTensor, alpha);
  
  // Unfused execution (separate operations)
  console.log('Running unfused operations (separate add and Leaky ReLU)...');
  
  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    const addResult = await backend.add(inputA, inputB, {
      useBrowserOptimizations: true
    });
    await backend.leakyRelu(addResult, {
      useBrowserOptimizations: true,
      alpha
    });
  }
  
  // Timed iterations
  const startTimeUnfused = performance.now();
  let unfusedResult;
  
  for (let i = 0; i < iterations; i++) {
    const addResult = await backend.add(inputA, inputB, {
      useBrowserOptimizations: true
    });
    unfusedResult = await backend.leakyRelu(addResult, {
      useBrowserOptimizations: true,
      alpha
    });
  }
  
  const endTimeUnfused = performance.now();
  const avgTimeUnfused = (endTimeUnfused - startTimeUnfused) / iterations;
  
  console.log(`Unfused execution: ${avgTimeUnfused.toFixed(2)} ms`);
  
  // Verify correctness
  const isUnfusedCorrect = areTensorsEqual(cpuLeakyReluResult, unfusedResult!, 1e-4);
  console.log(`Unfused correctness: ${isUnfusedCorrect ? 'PASSED' : 'FAILED'}`);
  
  // Fused execution (executeOperations)
  console.log('Running fused operations (add+LeakyReLU combined)...');
  
  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    await backend.executeOperations(
      [inputA, inputB],
      ['add', 'leakyRelu'],
      { 
        useBrowserOptimizations: true,
        useFusion: true,
        leakyReluOptions: { alpha }
      }
    );
  }
  
  // Timed iterations
  const startTimeFused = performance.now();
  let fusedResult;
  
  for (let i = 0; i < iterations; i++) {
    fusedResult = await backend.executeOperations(
      [inputA, inputB],
      ['add', 'leakyRelu'],
      { 
        useBrowserOptimizations: true,
        useFusion: true,
        leakyReluOptions: { alpha }
      }
    );
  }
  
  const endTimeFused = performance.now();
  const avgTimeFused = (endTimeFused - startTimeFused) / iterations;
  
  console.log(`Fused execution: ${avgTimeFused.toFixed(2)} ms`);
  
  // Verify correctness
  const isFusedCorrect = areTensorsEqual(cpuLeakyReluResult, fusedResult!, 1e-4);
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
  console.log('Browser-Specific Leaky ReLU Operation Test');
  console.log('=========================================');
  
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
    
    // Create test tensors
    console.log('Creating test tensors...');
    const tensorSize = 1024 * 1024; // 1M elements
    const input = generateRandomTensor([tensorSize]);
    const inputA = generateRandomTensor([tensorSize]);
    const inputB = generateRandomTensor([tensorSize]);
    
    // Run tests
    await testLeakyReluPerformance(backend, input);
    await testLeakyReluFusion(backend, inputA, inputB);
    
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