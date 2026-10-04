/**
 * Browser-Specific SiLU Operation Test
 * Tests the performance of browser-specific SiLU operations
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
  const data = Array(size).fill(0).map(() => Math.random() * 4 - 2); // Range [-2, 2] for SiLU testing
  return new Tensor<number>(shape, data, { dataType: 'float32' });
}

/**
 * Compute SiLU on CPU for reference
 * SiLU(x) = x * sigmoid(x)
 */
function cpuSilu(input: Tensor<number>): Tensor<number> {
  const output = new Float32Array(input.data.length);
  
  for (let i = 0; i < input.data.length; i++) {
    const x = input.data[i];
    const sigmoid = 1 / (1 + Math.exp(-x));
    output[i] = x * sigmoid;
  }
  return new Tensor<number>(input.shape, Array.from(output), { dataType: 'float32' });
}

/**
 * Compute fast SiLU approximation on CPU for reference
 * FastSiLU(x) = x * fast_sigmoid(x), where fast_sigmoid(x) = x / (1 + abs(x)) * 0.5 + 0.5
 */
function cpuFastSilu(input: Tensor<number>): Tensor<number> {
  const output = new Float32Array(input.data.length);
  
  for (let i = 0; i < input.data.length; i++) {
    const x = input.data[i];
    const fastSigmoid = x / (1 + Math.abs(x)) * 0.5 + 0.5;
    output[i] = x * fastSigmoid;
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
 * Test elementwise SiLU performance across browsers
 */
async function testSiluPerformance(backend: WebGPUBackend, input: Tensor<number>) {
  console.log('\n=== Testing Elementwise SiLU Performance ===');
  
  const actualBrowserType = detectBrowserType();
  const iterations = 10;
  const warmupIterations = 5;
  
  // Compute CPU reference results
  console.log('Computing CPU reference results...');
  const cpuStandardResult = cpuSilu(input);
  const cpuFastResult = cpuFastSilu(input);
  
  // Test each browser optimization
  const browserTypes = [
    { name: 'Generic (No Optimization)', type: null },
    { name: 'Chrome', type: BrowserType.CHROME },
    { name: 'Firefox', type: BrowserType.FIREFOX },
    { name: 'Safari', type: BrowserType.SAFARI },
    { name: 'Edge', type: BrowserType.EDGE }
  ];
  
  const standardResults: Record<string, number> = {};
  const fastResults: Record<string, number> = {};
  
  // Get actual browser shader for later comparison
  await loadBrowserShader(ShaderType.ELEMENTWISE_SILU);
  
  // Test standard SiLU implementations
  console.log('\nTesting standard SiLU implementations:');
  for (const browser of browserTypes) {
    const name = browser.name;
    console.log(`Testing ${name} optimization...`);
    
    // Warmup
    for (let i = 0; i < warmupIterations; i++) {
      await backend.silu(input, {
        useBrowserOptimizations: browser.type !== null,
        browserType: browser.type,
        useFastMath: false
      });
    }
    
    // Timed iterations
    const startTime = performance.now();
    let result;
    
    for (let i = 0; i < iterations; i++) {
      result = await backend.silu(input, {
        useBrowserOptimizations: browser.type !== null,
        browserType: browser.type,
        useFastMath: false
      });
    }
    
    const endTime = performance.now();
    const avgTime = (endTime - startTime) / iterations;
    standardResults[name] = avgTime;
    
    console.log(`${name}: ${avgTime.toFixed(2)} ms`);
    
    // Verify correctness
    const isCorrect = areTensorsEqual(cpuStandardResult, result!, 1e-4);
    console.log(`${name} correctness: ${isCorrect ? 'PASSED' : 'FAILED'}`);
  }
  
  // Test fast SiLU implementations
  console.log('\nTesting fast SiLU implementations:');
  for (const browser of browserTypes) {
    if (browser.type === null) continue; // Skip generic for fast math
    
    const name = `${browser.name} (Fast Math)`;
    console.log(`Testing ${name}...`);
    
    // Warmup
    for (let i = 0; i < warmupIterations; i++) {
      await backend.silu(input, {
        useBrowserOptimizations: true,
        browserType: browser.type,
        useFastMath: true
      });
    }
    
    // Timed iterations
    const startTime = performance.now();
    let result;
    
    for (let i = 0; i < iterations; i++) {
      result = await backend.silu(input, {
        useBrowserOptimizations: true,
        browserType: browser.type,
        useFastMath: true
      });
    }
    
    const endTime = performance.now();
    const avgTime = (endTime - startTime) / iterations;
    fastResults[browser.name] = avgTime;
    
    console.log(`${name}: ${avgTime.toFixed(2)} ms`);
    
    // Verify correctness with higher epsilon for fast math approximation
    const isCorrect = areTensorsEqual(cpuFastResult, result!, 1e-2);
    console.log(`${name} correctness: ${isCorrect ? 'PASSED' : 'FAILED (Expected for approximation)'}`);
  }
  
  // Calculate speedups for standard implementation
  const baseTime = standardResults['Generic (No Optimization)'];
  console.log('\nSpeedup compared to generic implementation:');
  
  for (const browser of browserTypes) {
    if (browser.type === null) continue;
    
    const speedup = baseTime / standardResults[browser.name];
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
    
    const standardTime = standardResults[browser.name];
    const fastTime = fastResults[browser.name];
    const speedup = standardTime / fastTime;
    
    console.log(`${browser.name}: ${speedup.toFixed(2)}x`);
    
    // Highlight current browser
    if (browser.type === actualBrowserType) {
      console.log(`Current browser (${browser.name}) fast math speedup: ${speedup.toFixed(2)}x`);
    }
  }
  
  return { standard: standardResults, fast: fastResults };
}

/**
 * Test fusion of SiLU with other operations for performance
 */
async function testSiluFusion(backend: WebGPUBackend, inputA: Tensor<number>, inputB: Tensor<number>) {
  console.log('\n=== Testing SiLU Fusion Performance ===');
  
  const actualBrowserType = detectBrowserType();
  const iterations = 10;
  const warmupIterations = 5;
  
  // Test sequences: add followed by SiLU (common in transformer models)
  console.log('Testing Add+SiLU fusion...');
  
  // Compute CPU reference result
  console.log('Computing CPU reference result...');
  // Add on CPU
  const cpuAddResult = new Float32Array(inputA.data.length);
  for (let i = 0; i < inputA.data.length; i++) {
    cpuAddResult[i] = inputA.data[i] + inputB.data[i];
  }
  const cpuAddTensor = new Tensor<number>(inputA.shape, Array.from(cpuAddResult), { dataType: 'float32' });
  
  // SiLU on CPU
  const cpuSiluResult = cpuSilu(cpuAddTensor);
  
  // Unfused execution (separate operations)
  console.log('Running unfused operations (separate add and SiLU)...');
  
  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    const addResult = await backend.add(inputA, inputB, {
      useBrowserOptimizations: true
    });
    await backend.silu(addResult, {
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
    unfusedResult = await backend.silu(addResult, {
      useBrowserOptimizations: true
    });
  }
  
  const endTimeUnfused = performance.now();
  const avgTimeUnfused = (endTimeUnfused - startTimeUnfused) / iterations;
  
  console.log(`Unfused execution: ${avgTimeUnfused.toFixed(2)} ms`);
  
  // Verify correctness
  const isUnfusedCorrect = areTensorsEqual(cpuSiluResult, unfusedResult!, 1e-4);
  console.log(`Unfused correctness: ${isUnfusedCorrect ? 'PASSED' : 'FAILED'}`);
  
  // Fused execution (executeOperations)
  console.log('Running fused operations (add+SiLU combined)...');
  
  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    await backend.executeOperations(
      [inputA, inputB],
      ['add', 'silu'],
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
      ['add', 'silu'],
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
  const isFusedCorrect = areTensorsEqual(cpuSiluResult, fusedResult!, 1e-4);
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
  console.log('Browser-Specific SiLU Operation Test');
  console.log('===================================');
  
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
    await testSiluPerformance(backend, input);
    await testSiluFusion(backend, inputA, inputB);
    
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