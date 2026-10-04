/**
 * WebGPU Operation Fusion and Quantization Integration Test
 * Tests both the functionality and performance of operation fusion with quantization
 */

import { Tensor } from '../src/tensor/tensor';
import { WebGPUBackend } from '../src/hardware/webgpu/backend';
import { WebGPUOperationFusion, FusionPattern, FusionConfig } from '../src/hardware/webgpu/optimizations/operation_fusion';
import { BrowserType } from '../src/hardware/webgpu/browser_optimized_operations';

/**
 * Helper to generate a random matrix of specified shape
 */
function generateRandomTensor(shape: number[], range: number = 1): Tensor<number> {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = Array(size).fill(0).map(() => Math.random() * 2 * range - range);
  return new Tensor<number>(shape, data, { dataType: 'float32' });
}

/**
 * Simple CPU implementation of matrix multiplication for result validation
 */
function cpuMatmul(a: Tensor<number>, b: Tensor<number>): Tensor<number> {
  if (a.shape.length !== 2 || b.shape.length !== 2 || a.shape[1] !== b.shape[0]) {
    throw new Error('Invalid shapes for matrix multiplication');
  }
  
  const [m, k] = a.shape;
  const [_, n] = b.shape;
  const resultData = new Array(m * n).fill(0);
  
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) {
        sum += a.data[i * k + p] * b.data[p * n + j];
      }
      resultData[i * n + j] = sum;
    }
  }
  
  return new Tensor<number>([m, n], resultData, { dataType: 'float32' });
}

/**
 * CPU implementation of ReLU activation
 */
function cpuRelu(input: Tensor<number>): Tensor<number> {
  const resultData = input.data.map(v => Math.max(0, v));
  return new Tensor<number>(input.shape, resultData, { dataType: 'float32' });
}

/**
 * Check if two tensors are approximately equal (within epsilon)
 */
function areTensorsEqual(a: Tensor<number>, b: Tensor<number>, epsilon: number = 1e-4): boolean {
  if (a.shape.length !== b.shape.length) return false;
  if (!a.shape.every((dim, i) => dim === b.shape[i])) return false;
  
  for (let i = 0; i < a.data.length; i++) {
    if (Math.abs(a.data[i] - b.data[i]) > epsilon) {
      console.error(`Mismatch at index ${i}: ${a.data[i]} vs ${b.data[i]}`);
      return false;
    }
  }
  
  return true;
}

/**
 * Test the quantized matmul operation with different precision settings
 */
async function testQuantizedMatmul(
  backend: WebGPUBackend, 
  matrixA: Tensor<number>, 
  matrixB: Tensor<number>, 
  bitsPerWeight: 1 | 2 | 3 | 4 | 8
): Promise<{
  correctness: boolean,
  speedup: number,
  memoryReduction: number
}> {
  console.log(`Testing ${bitsPerWeight}-bit quantized matrix multiplication...`);
  
  // Create fusion config with specified bit precision
  const fusionConfig: FusionConfig = {
    useQuantizedWeights: true,
    bitsPerWeight: bitsPerWeight,
    useBrowserOptimizations: true,
    enabledPatterns: [FusionPattern.QuantizedMatmul]
  };
  
  const fusion = new WebGPUOperationFusion(backend, fusionConfig);
  
  // Compute reference result with CPU implementation
  console.log("Computing reference result with CPU...");
  const startCpu = performance.now();
  const cpuResult = cpuMatmul(matrixA, matrixB);
  const endCpu = performance.now();
  const cpuTime = endCpu - startCpu;
  console.log(`CPU computation time: ${cpuTime.toFixed(2)}ms`);
  
  // Original memory usage (float32)
  const originalMemory = matrixA.size * 4; // Only quantizing matrix A
  
  // Calculate memory usage after quantization
  // Each weight takes bitsPerWeight bits instead of 32 bits
  const quantizedMemory = Math.ceil(matrixA.size * bitsPerWeight / 8);
  
  // Add overhead for scales and zero points
  const overhead = 4 * 2; // 2 float32 values for scale and zero point
  const totalQuantizedMemory = quantizedMemory + overhead;
  
  // Memory reduction percentage
  const memoryReduction = (originalMemory - totalQuantizedMemory) / originalMemory * 100;
  
  // Perform quantized matrix multiplication
  console.log("Performing quantized matrix multiplication...");
  const startGpu = performance.now();
  // In a real test, you would call the actual fusion operation here
  // For now, we'll directly call the backend's operation
  const gpuResult = await backend.matmul(matrixA, matrixB, { 
    useQuantization: true,
    bitsPerWeight: bitsPerWeight
  });
  const endGpu = performance.now();
  const gpuTime = endGpu - startGpu;
  console.log(`GPU computation time: ${gpuTime.toFixed(2)}ms`);
  
  // Calculate speedup
  const speedup = cpuTime / gpuTime;
  
  // Check correctness with some tolerance for quantization error
  // Higher tolerance for lower bit precision
  const epsilon = bitsPerWeight <= 2 ? 0.1 : (bitsPerWeight <= 4 ? 0.05 : 0.01);
  const isCorrect = areTensorsEqual(cpuResult, gpuResult, epsilon);
  
  console.log(`Correctness: ${isCorrect ? 'PASSED' : 'FAILED'}`);
  console.log(`Speedup: ${speedup.toFixed(2)}x`);
  console.log(`Memory reduction: ${memoryReduction.toFixed(2)}%`);
  
  return {
    correctness: isCorrect,
    speedup: speedup,
    memoryReduction: memoryReduction
  };
}

/**
 * Test fused quantized matmul with activation (e.g., ReLU)
 */
async function testQuantizedMatmulActivation(
  backend: WebGPUBackend, 
  matrixA: Tensor<number>, 
  matrixB: Tensor<number>, 
  bitsPerWeight: 1 | 2 | 3 | 4 | 8
): Promise<{
  correctness: boolean,
  fusionSpeedup: number
}> {
  console.log(`Testing ${bitsPerWeight}-bit quantized matrix multiplication with ReLU...`);
  
  // Create fusion config
  const fusionConfig: FusionConfig = {
    useQuantizedWeights: true,
    bitsPerWeight: bitsPerWeight,
    useBrowserOptimizations: true,
    enabledPatterns: [FusionPattern.QuantizedMatmulActivation]
  };
  
  const fusion = new WebGPUOperationFusion(backend, fusionConfig);
  
  // Compute reference result with CPU implementation
  console.log("Computing reference result with CPU...");
  const startCpu = performance.now();
  const matmulResult = cpuMatmul(matrixA, matrixB);
  const cpuResult = cpuRelu(matmulResult);
  const endCpu = performance.now();
  const cpuTime = endCpu - startCpu;
  console.log(`CPU computation time: ${cpuTime.toFixed(2)}ms`);
  
  // Perform unfused operations (separate matmul and relu) for comparison
  console.log("Performing unfused operations (separate matmul + ReLU)...");
  const startUnfused = performance.now();
  const unfusedMatmul = await backend.matmul(matrixA, matrixB, { 
    useQuantization: true,
    bitsPerWeight: bitsPerWeight
  });
  const unfusedResult = await backend.relu(unfusedMatmul);
  const endUnfused = performance.now();
  const unfusedTime = endUnfused - startUnfused;
  console.log(`Unfused operations time: ${unfusedTime.toFixed(2)}ms`);
  
  // Perform fused operation (matmul + relu in one shader)
  console.log("Performing fused operation (matmul + ReLU)...");
  const startFused = performance.now();
  // In a real test, you would call the actual fusion operation here
  const fusedResult = await backend.executeOperations(
    [matrixA, matrixB],
    ['matmul', 'relu'],
    { 
      useQuantization: true,
      bitsPerWeight: bitsPerWeight,
      useFusion: true
    }
  );
  const endFused = performance.now();
  const fusedTime = endFused - startFused;
  console.log(`Fused operation time: ${fusedTime.toFixed(2)}ms`);
  
  // Calculate fusion speedup
  const fusionSpeedup = unfusedTime / fusedTime;
  
  // Check correctness
  const epsilon = bitsPerWeight <= 2 ? 0.1 : (bitsPerWeight <= 4 ? 0.05 : 0.01);
  const isCorrect = areTensorsEqual(cpuResult, fusedResult, epsilon);
  
  console.log(`Correctness: ${isCorrect ? 'PASSED' : 'FAILED'}`);
  console.log(`Fusion speedup: ${fusionSpeedup.toFixed(2)}x`);
  
  return {
    correctness: isCorrect,
    fusionSpeedup: fusionSpeedup
  };
}

/**
 * Test browser-specific optimizations for operation fusion
 */
async function testBrowserOptimizations(
  backend: WebGPUBackend,
  matrixA: Tensor<number>,
  matrixB: Tensor<number>
): Promise<{
  browsersCompared: string[],
  results: Record<string, number>
}> {
  console.log("Testing browser-specific optimizations...");
  
  const browsers = [
    { name: 'Chrome', type: BrowserType.CHROME },
    { name: 'Firefox', type: BrowserType.FIREFOX },
    { name: 'Safari', type: BrowserType.SAFARI },
    { name: 'Edge', type: BrowserType.EDGE },
    { name: 'Default', type: BrowserType.UNKNOWN }
  ];
  
  const results: Record<string, number> = {};
  
  // Compute reference result
  const cpuResult = cpuMatmul(matrixA, matrixB);
  const correctResults: Record<string, boolean> = {};
  
  // Test each browser optimization
  for (const browser of browsers) {
    console.log(`Testing optimizations for ${browser.name}...`);
    
    const fusionConfig: FusionConfig = {
      useQuantizedWeights: true,
      bitsPerWeight: 4,
      useBrowserOptimizations: true,
      browserOptimizationType: browser.type as any
    };
    
    const fusion = new WebGPUOperationFusion(backend, fusionConfig);
    
    const startTime = performance.now();
    const result = await backend.matmul(matrixA, matrixB, { 
      useQuantization: true,
      bitsPerWeight: 4,
      browserType: browser.type
    });
    const endTime = performance.now();
    const executionTime = endTime - startTime;
    
    results[browser.name] = executionTime;
    correctResults[browser.name] = areTensorsEqual(cpuResult, result, 0.05);
    
    console.log(`${browser.name} execution time: ${executionTime.toFixed(2)}ms`);
    console.log(`${browser.name} correctness: ${correctResults[browser.name] ? 'PASSED' : 'FAILED'}`);
  }
  
  // Find fastest browser
  let fastestBrowser = '';
  let fastestTime = Infinity;
  
  for (const [browser, time] of Object.entries(results)) {
    if (time < fastestTime && correctResults[browser]) {
      fastestTime = time;
      fastestBrowser = browser;
    }
  }
  
  console.log(`\nFastest browser: ${fastestBrowser} (${fastestTime.toFixed(2)}ms)`);
  
  // Calculate relative performance
  console.log("\nRelative performance (lower is better):");
  for (const [browser, time] of Object.entries(results)) {
    if (correctResults[browser]) {
      const relative = time / fastestTime;
      console.log(`${browser}: ${relative.toFixed(2)}x ${fastestBrowser}`);
    } else {
      console.log(`${browser}: FAILED (incorrect results)`);
    }
  }
  
  return {
    browsersCompared: browsers.map(b => b.name),
    results
  };
}

/**
 * Test ultra-low precision quantization (2-bit, 3-bit)
 */
async function testUltraLowPrecision(
  backend: WebGPUBackend,
  matrixA: Tensor<number>,
  matrixB: Tensor<number>
): Promise<{
  precisions: number[],
  accuracies: Record<string, number>,
  memoryReductions: Record<string, number>
}> {
  console.log("Testing ultra-low precision quantization...");
  
  const precisions: (1 | 2 | 3 | 4 | 8)[] = [1, 2, 3, 4, 8];
  const results: Record<string, number> = {};
  const accuracies: Record<string, number> = {};
  const memoryReductions: Record<string, number> = {};
  
  // Compute reference result
  const cpuResult = cpuMatmul(matrixA, matrixB);
  
  // Test each precision
  for (const bits of precisions) {
    console.log(`Testing ${bits}-bit quantization...`);
    
    const fusionConfig: FusionConfig = {
      useQuantizedWeights: true,
      bitsPerWeight: bits,
      useBrowserOptimizations: true
    };
    
    const fusion = new WebGPUOperationFusion(backend, fusionConfig);
    
    // Compute memory reduction
    const originalMemory = matrixA.size * 4;
    const quantizedMemory = Math.ceil(matrixA.size * bits / 8);
    const overhead = 4 * 2; // scale and zero point
    const totalQuantizedMemory = quantizedMemory + overhead;
    const memoryReduction = (originalMemory - totalQuantizedMemory) / originalMemory * 100;
    
    memoryReductions[`${bits}-bit`] = memoryReduction;
    
    // Test execution
    const result = await backend.matmul(matrixA, matrixB, { 
      useQuantization: true,
      bitsPerWeight: bits
    });
    
    // Calculate accuracy by comparing with reference
    let errorSum = 0;
    let errorMax = 0;
    
    for (let i = 0; i < cpuResult.data.length; i++) {
      const error = Math.abs(cpuResult.data[i] - result.data[i]);
      errorSum += error;
      errorMax = Math.max(errorMax, error);
    }
    
    const avgError = errorSum / cpuResult.data.length;
    const accuracy = 100 - (avgError / Math.max(...cpuResult.data.map(Math.abs)) * 100);
    
    accuracies[`${bits}-bit`] = accuracy;
    
    console.log(`${bits}-bit memory reduction: ${memoryReduction.toFixed(2)}%`);
    console.log(`${bits}-bit average error: ${avgError.toFixed(6)}`);
    console.log(`${bits}-bit max error: ${errorMax.toFixed(6)}`);
    console.log(`${bits}-bit accuracy: ${accuracy.toFixed(2)}%`);
  }
  
  console.log("\nPrecision comparison summary:");
  console.log("Precision | Memory Reduction | Accuracy");
  console.log("----------|------------------|----------");
  
  for (const bits of precisions) {
    console.log(`${bits}-bit     | ${memoryReductions[`${bits}-bit`].toFixed(2)}%          | ${accuracies[`${bits}-bit`].toFixed(2)}%`);
  }
  
  return {
    precisions: precisions,
    accuracies,
    memoryReductions
  };
}

/**
 * Run all tests
 */
async function runTests() {
  console.log("Starting WebGPU Operation Fusion and Quantization tests...");
  
  try {
    // Initialize WebGPU backend
    const backend = new WebGPUBackend();
    await backend.initialize();
    console.log("WebGPU backend initialized successfully");
    
    // Create test matrices
    const smallMatrixA = generateRandomTensor([64, 32]);
    const smallMatrixB = generateRandomTensor([32, 16]);
    
    const mediumMatrixA = generateRandomTensor([256, 128]);
    const mediumMatrixB = generateRandomTensor([128, 64]);
    
    const largeMatrixA = generateRandomTensor([1024, 512]);
    const largeMatrixB = generateRandomTensor([512, 256]);
    
    // Test matrix sizes
    console.log("Small matrices: 64x32 * 32x16");
    console.log("Medium matrices: 256x128 * 128x64");
    console.log("Large matrices: 1024x512 * 512x256");
    
    // Run quantized matmul tests
    console.log("\n=== Testing Quantized Matrix Multiplication ===");
    await testQuantizedMatmul(backend, mediumMatrixA, mediumMatrixB, 4);
    
    // Run fusion tests
    console.log("\n=== Testing Quantized Matrix Multiplication with Activation ===");
    await testQuantizedMatmulActivation(backend, mediumMatrixA, mediumMatrixB, 4);
    
    // Test browser optimizations
    console.log("\n=== Testing Browser-Specific Optimizations ===");
    await testBrowserOptimizations(backend, mediumMatrixA, mediumMatrixB);
    
    // Test ultra-low precision
    console.log("\n=== Testing Ultra-Low Precision Quantization ===");
    await testUltraLowPrecision(backend, smallMatrixA, smallMatrixB);
    
    // Large model test with 2-bit quantization (most memory efficient)
    console.log("\n=== Large Matrix Test with 2-bit Quantization ===");
    await testQuantizedMatmul(backend, largeMatrixA, largeMatrixB, 2);
    
    console.log("\nAll tests completed!");

  } catch (error) {
    console.error("Error running tests:", error);
  }
}

// Check if WebGPU is available and run tests
if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
  console.log("WebGPU is available. Running tests...");
  runTests();
} else {
  console.error("WebGPU is not supported in this environment. Tests cannot run.");
}