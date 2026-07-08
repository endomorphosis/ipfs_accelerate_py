/**
 * Browser-Specific Elementwise Operation Test
 * Tests the performance of browser-specific elementwise operations
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
  const data = Array(size).fill(0).map(() => Math.random() * 2 - 1);
  return new Tensor<number>(shape, data, { dataType: 'float32' });
}

/**
 * Compute ReLU on CPU for reference
 */
function cpuRelu(input: Tensor<number>): Tensor<number> {
  const output = new Float32Array(input.data.length);
  for (let i = 0; i < input.data.length; i++) {
    output[i] = Math.max(0, input.data[i]);
  }
  return new Tensor<number>(input.shape, Array.from(output), { dataType: 'float32' });
}

/**
 * Compute elementwise addition on CPU for reference
 */
function cpuAdd(a: Tensor<number>, b: Tensor<number>): Tensor<number> {
  if (a.shape.join(',') !== b.shape.join(',')) {
    throw new Error('Shapes must match for addition');
  }
  
  const output = new Float32Array(a.data.length);
  for (let i = 0; i < a.data.length; i++) {
    output[i] = a.data[i] + b.data[i];
  }
  return new Tensor<number>(a.shape, Array.from(output), { dataType: 'float32' });
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
 * Test elementwise ReLU performance across browsers
 */
async function testReLUPerformance(backend: WebGPUBackend, input: Tensor<number>) {
  console.log('\n=== Testing Elementwise ReLU Performance ===');
  
  const actualBrowserType = detectBrowserType();
  const iterations = 10;
  const warmupIterations = 5;
  
  // Compute CPU reference result
  const cpuResult = cpuRelu(input);
  
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
  await loadBrowserShader(ShaderType.ELEMENTWISE_RELU);
  
  for (const browser of browserTypes) {
    const name = browser.name;
    console.log(`Testing ${name} optimization...`);
    
    // Warmup
    for (let i = 0; i < warmupIterations; i++) {
      await backend.relu(input, {
        useBrowserOptimizations: browser.type !== null,
        browserType: browser.type
      });
    }
    
    // Timed iterations
    const startTime = performance.now();
    let result;
    
    for (let i = 0; i < iterations; i++) {
      result = await backend.relu(input, {
        useBrowserOptimizations: browser.type !== null,
        browserType: browser.type
      });
    }
    
    const endTime = performance.now();
    const avgTime = (endTime - startTime) / iterations;
    results[name] = avgTime;
    
    console.log(`${name}: ${avgTime.toFixed(2)} ms`);
    
    // Verify correctness
    const isCorrect = areTensorsEqual(cpuResult, result!);
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
 * Test elementwise addition performance across browsers
 */
async function testAddPerformance(backend: WebGPUBackend, inputA: Tensor<number>, inputB: Tensor<number>) {
  console.log('\n=== Testing Elementwise Addition Performance ===');
  
  const actualBrowserType = detectBrowserType();
  const iterations = 10;
  const warmupIterations = 5;
  
  // Compute CPU reference result
  const cpuResult = cpuAdd(inputA, inputB);
  
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
  await loadBrowserShader(ShaderType.ELEMENTWISE_ADD);
  
  for (const browser of browserTypes) {
    const name = browser.name;
    console.log(`Testing ${name} optimization...`);
    
    // Warmup
    for (let i = 0; i < warmupIterations; i++) {
      await backend.add(inputA, inputB, {
        useBrowserOptimizations: browser.type !== null,
        browserType: browser.type
      });
    }
    
    // Timed iterations
    const startTime = performance.now();
    let result;
    
    for (let i = 0; i < iterations; i++) {
      result = await backend.add(inputA, inputB, {
        useBrowserOptimizations: browser.type !== null,
        browserType: browser.type
      });
    }
    
    const endTime = performance.now();
    const avgTime = (endTime - startTime) / iterations;
    results[name] = avgTime;
    
    console.log(`${name}: ${avgTime.toFixed(2)} ms`);
    
    // Verify correctness
    const isCorrect = areTensorsEqual(cpuResult, result!);
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
 * Test fusion of elementwise operations for performance
 */
async function testElementwiseFusion(backend: WebGPUBackend, inputA: Tensor<number>, inputB: Tensor<number>) {
  console.log('\n=== Testing Elementwise Operation Fusion ===');
  
  const actualBrowserType = detectBrowserType();
  const iterations = 10;
  const warmupIterations = 5;
  
  // Test sequences: add followed by relu
  console.log('Testing Add+ReLU fusion...');
  
  // Compute CPU reference result
  const cpuAddResult = cpuAdd(inputA, inputB);
  const cpuReluResult = cpuRelu(cpuAddResult);
  
  // Unfused execution (separate operations)
  console.log('Running unfused operations (separate add and relu)...');
  
  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    const addResult = await backend.add(inputA, inputB, {
      useBrowserOptimizations: true
    });
    await backend.relu(addResult, {
      useBrowserOptimizations: true
    });
  }
  
  // Timed iterations
  const startTimeUnfused = performance.now();
  let unfusedResult;
  
  for (let i = 0; i < iterations; i++) {
    const addResult = await backend.add(inputA, inputB, {
      useBrowserOptimizations: true
    });
    unfusedResult = await backend.relu(addResult, {
      useBrowserOptimizations: true
    });
  }
  
  const endTimeUnfused = performance.now();
  const avgTimeUnfused = (endTimeUnfused - startTimeUnfused) / iterations;
  
  console.log(`Unfused execution: ${avgTimeUnfused.toFixed(2)} ms`);
  
  // Verify correctness
  const isUnfusedCorrect = areTensorsEqual(cpuReluResult, unfusedResult!);
  console.log(`Unfused correctness: ${isUnfusedCorrect ? 'PASSED' : 'FAILED'}`);
  
  // Fused execution (executeOperations)
  console.log('Running fused operations (add+relu combined)...');
  
  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    await backend.executeOperations(
      [inputA, inputB],
      ['add', 'relu'],
      { 
        useBrowserOptimizations: true,
        useFusion: true
      }
    );
  }
  
  // Timed iterations
  const startTimeFused = performance.now();
  let fusedResult;
  
  for (let i = 0; i < iterations; i++) {
    fusedResult = await backend.executeOperations(
      [inputA, inputB],
      ['add', 'relu'],
      { 
        useBrowserOptimizations: true,
        useFusion: true
      }
    );
  }
  
  const endTimeFused = performance.now();
  const avgTimeFused = (endTimeFused - startTimeFused) / iterations;
  
  console.log(`Fused execution: ${avgTimeFused.toFixed(2)} ms`);
  
  // Verify correctness
  const isFusedCorrect = areTensorsEqual(cpuReluResult, fusedResult!);
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
  console.log('Browser-Specific Elementwise Operation Test');
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
    await testReLUPerformance(backend, input);
    await testAddPerformance(backend, inputA, inputB);
    await testElementwiseFusion(backend, inputA, inputB);
    
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