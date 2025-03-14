/**
 * Browser-Specific Advanced Elementwise Operation Test
 * Tests the performance of browser-specific tanh and sigmoid operations
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
  const data = Array(size).fill(0).map(() => Math.random() * 4 - 2); // Range [-2, 2] for tanh/sigmoid testing
  return new Tensor<number>(shape, data, { dataType: 'float32' });
}

/**
 * Compute tanh on CPU for reference
 */
function cpuTanh(input: Tensor<number>): Tensor<number> {
  const output = new Float32Array(input.data.length);
  for (let i = 0; i < input.data.length; i++) {
    // Standard tanh implementation
    output[i] = Math.tanh(input.data[i]);
  }
  return new Tensor<number>(input.shape, Array.from(output), { dataType: 'float32' });
}

/**
 * Compute sigmoid on CPU for reference
 */
function cpuSigmoid(input: Tensor<number>): Tensor<number> {
  const output = new Float32Array(input.data.length);
  for (let i = 0; i < input.data.length; i++) {
    // Standard sigmoid implementation with clamping
    const clampedValue = Math.max(-30, Math.min(30, input.data[i]));
    output[i] = 1 / (1 + Math.exp(-clampedValue));
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
 * Test elementwise tanh performance across browsers
 */
async function testTanhPerformance(backend: WebGPUBackend, input: Tensor<number>) {
  console.log('\n=== Testing Elementwise Tanh Performance ===');
  
  const actualBrowserType = detectBrowserType();
  const iterations = 10;
  const warmupIterations = 5;
  
  // Compute CPU reference result
  console.log('Computing CPU reference result...');
  const cpuResult = cpuTanh(input);
  
  // Test each browser optimization
  const browserTypes = [
    { name: 'Generic (No Optimization)', type: null },
    { name: 'Chrome', type: BrowserType.CHROME },
    { name: 'Firefox', type: BrowserType.FIREFOX },
    { name: 'Safari', type: BrowserType.SAFARI },
    { name: 'Edge', type: BrowserType.EDGE }
  ];
  
  const results: Record<string, number> = {};
  const fastMathResults: Record<string, number> = {};
  
  // Get actual browser shader for later comparison
  await loadBrowserShader(ShaderType.ELEMENTWISE_TANH);
  
  // Test standard tanh implementations
  console.log('\nTesting standard tanh implementations:');
  for (const browser of browserTypes) {
    const name = browser.name;
    console.log(`Testing ${name} optimization...`);
    
    // Warmup
    for (let i = 0; i < warmupIterations; i++) {
      await backend.tanh(input, {
        useBrowserOptimizations: browser.type !== null,
        browserType: browser.type,
        useFastMath: false
      });
    }
    
    // Timed iterations
    const startTime = performance.now();
    let result;
    
    for (let i = 0; i < iterations; i++) {
      result = await backend.tanh(input, {
        useBrowserOptimizations: browser.type !== null,
        browserType: browser.type,
        useFastMath: false
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
  
  // Test fast math tanh implementations
  console.log('\nTesting fast math tanh implementations:');
  for (const browser of browserTypes) {
    if (browser.type === null) continue; // Skip generic for fast math
    
    const name = `${browser.name} (Fast Math)`;
    console.log(`Testing ${name}...`);
    
    // Warmup
    for (let i = 0; i < warmupIterations; i++) {
      await backend.tanh(input, {
        useBrowserOptimizations: true,
        browserType: browser.type,
        useFastMath: true
      });
    }
    
    // Timed iterations
    const startTime = performance.now();
    let result;
    
    for (let i = 0; i < iterations; i++) {
      result = await backend.tanh(input, {
        useBrowserOptimizations: true,
        browserType: browser.type,
        useFastMath: true
      });
    }
    
    const endTime = performance.now();
    const avgTime = (endTime - startTime) / iterations;
    fastMathResults[browser.name] = avgTime;
    
    console.log(`${name}: ${avgTime.toFixed(2)} ms`);
    
    // Verify correctness with higher epsilon for fast math approximation
    const isCorrect = areTensorsEqual(cpuResult, result!, 1e-2);
    console.log(`${name} correctness: ${isCorrect ? 'PASSED' : 'FAILED (Expected for approximation)'}`);
  }
  
  // Calculate speedups for standard implementation
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
  
  // Calculate fast math speedup over standard implementation
  console.log('\nFast math speedup compared to standard implementation:');
  
  for (const browser of browserTypes) {
    if (browser.type === null) continue;
    
    const standardTime = results[browser.name];
    const fastTime = fastMathResults[browser.name];
    const speedup = standardTime / fastTime;
    
    console.log(`${browser.name}: ${speedup.toFixed(2)}x`);
    
    // Highlight current browser
    if (browser.type === actualBrowserType) {
      console.log(`Current browser (${browser.name}) fast math speedup: ${speedup.toFixed(2)}x`);
    }
  }
  
  return { standard: results, fastMath: fastMathResults };
}

/**
 * Test elementwise sigmoid performance across browsers
 */
async function testSigmoidPerformance(backend: WebGPUBackend, input: Tensor<number>) {
  console.log('\n=== Testing Elementwise Sigmoid Performance ===');
  
  const actualBrowserType = detectBrowserType();
  const iterations = 10;
  const warmupIterations = 5;
  
  // Compute CPU reference result
  console.log('Computing CPU reference result...');
  const cpuResult = cpuSigmoid(input);
  
  // Test each browser optimization
  const browserTypes = [
    { name: 'Generic (No Optimization)', type: null },
    { name: 'Chrome', type: BrowserType.CHROME },
    { name: 'Firefox', type: BrowserType.FIREFOX },
    { name: 'Safari', type: BrowserType.SAFARI },
    { name: 'Edge', type: BrowserType.EDGE }
  ];
  
  const results: Record<string, number> = {};
  const fastMathResults: Record<string, number> = {};
  
  // Get actual browser shader for later comparison
  await loadBrowserShader(ShaderType.ELEMENTWISE_SIGMOID);
  
  // Test standard sigmoid implementations
  console.log('\nTesting standard sigmoid implementations:');
  for (const browser of browserTypes) {
    const name = browser.name;
    console.log(`Testing ${name} optimization...`);
    
    // Warmup
    for (let i = 0; i < warmupIterations; i++) {
      await backend.sigmoid(input, {
        useBrowserOptimizations: browser.type !== null,
        browserType: browser.type,
        useFastMath: false
      });
    }
    
    // Timed iterations
    const startTime = performance.now();
    let result;
    
    for (let i = 0; i < iterations; i++) {
      result = await backend.sigmoid(input, {
        useBrowserOptimizations: browser.type !== null,
        browserType: browser.type,
        useFastMath: false
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
  
  // Test fast math sigmoid implementations
  console.log('\nTesting fast math sigmoid implementations:');
  for (const browser of browserTypes) {
    if (browser.type === null) continue; // Skip generic for fast math
    
    const name = `${browser.name} (Fast Math)`;
    console.log(`Testing ${name}...`);
    
    // Warmup
    for (let i = 0; i < warmupIterations; i++) {
      await backend.sigmoid(input, {
        useBrowserOptimizations: true,
        browserType: browser.type,
        useFastMath: true
      });
    }
    
    // Timed iterations
    const startTime = performance.now();
    let result;
    
    for (let i = 0; i < iterations; i++) {
      result = await backend.sigmoid(input, {
        useBrowserOptimizations: true,
        browserType: browser.type,
        useFastMath: true
      });
    }
    
    const endTime = performance.now();
    const avgTime = (endTime - startTime) / iterations;
    fastMathResults[browser.name] = avgTime;
    
    console.log(`${name}: ${avgTime.toFixed(2)} ms`);
    
    // Verify correctness with higher epsilon for fast math approximation
    const isCorrect = areTensorsEqual(cpuResult, result!, 1e-2);
    console.log(`${name} correctness: ${isCorrect ? 'PASSED' : 'FAILED (Expected for approximation)'}`);
  }
  
  // Calculate speedups for standard implementation
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
  
  // Calculate fast math speedup over standard implementation
  console.log('\nFast math speedup compared to standard implementation:');
  
  for (const browser of browserTypes) {
    if (browser.type === null) continue;
    
    const standardTime = results[browser.name];
    const fastTime = fastMathResults[browser.name];
    const speedup = standardTime / fastTime;
    
    console.log(`${browser.name}: ${speedup.toFixed(2)}x`);
    
    // Highlight current browser
    if (browser.type === actualBrowserType) {
      console.log(`Current browser (${browser.name}) fast math speedup: ${speedup.toFixed(2)}x`);
    }
  }
  
  return { standard: results, fastMath: fastMathResults };
}

/**
 * Test fusion of advanced elementwise operations for performance
 */
async function testAdvancedElementwiseFusion(backend: WebGPUBackend, inputA: Tensor<number>) {
  console.log('\n=== Testing Advanced Elementwise Operation Fusion ===');
  
  const actualBrowserType = detectBrowserType();
  const iterations = 10;
  const warmupIterations = 5;
  
  // Test sequences: tanh followed by sigmoid
  console.log('Testing Tanh+Sigmoid fusion...');
  
  // Compute CPU reference result
  const cpuTanhResult = cpuTanh(inputA);
  const cpuSigmoidResult = cpuSigmoid(cpuTanhResult);
  
  // Unfused execution (separate operations)
  console.log('Running unfused operations (separate tanh and sigmoid)...');
  
  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    const tanhResult = await backend.tanh(inputA, {
      useBrowserOptimizations: true
    });
    await backend.sigmoid(tanhResult, {
      useBrowserOptimizations: true
    });
  }
  
  // Timed iterations
  const startTimeUnfused = performance.now();
  let unfusedResult;
  
  for (let i = 0; i < iterations; i++) {
    const tanhResult = await backend.tanh(inputA, {
      useBrowserOptimizations: true
    });
    unfusedResult = await backend.sigmoid(tanhResult, {
      useBrowserOptimizations: true
    });
  }
  
  const endTimeUnfused = performance.now();
  const avgTimeUnfused = (endTimeUnfused - startTimeUnfused) / iterations;
  
  console.log(`Unfused execution: ${avgTimeUnfused.toFixed(2)} ms`);
  
  // Verify correctness
  const isUnfusedCorrect = areTensorsEqual(cpuSigmoidResult, unfusedResult!, 1e-4);
  console.log(`Unfused correctness: ${isUnfusedCorrect ? 'PASSED' : 'FAILED'}`);
  
  // Fused execution (executeOperations)
  console.log('Running fused operations (tanh+sigmoid combined)...');
  
  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    await backend.executeOperations(
      [inputA],
      ['tanh', 'sigmoid'],
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
      [inputA],
      ['tanh', 'sigmoid'],
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
  const isFusedCorrect = areTensorsEqual(cpuSigmoidResult, fusedResult!, 1e-4);
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
  console.log('Browser-Specific Advanced Elementwise Operation Test');
  console.log('==================================================');
  
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
    
    // Run tests
    await testTanhPerformance(backend, input);
    await testSigmoidPerformance(backend, input);
    await testAdvancedElementwiseFusion(backend, input);
    
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