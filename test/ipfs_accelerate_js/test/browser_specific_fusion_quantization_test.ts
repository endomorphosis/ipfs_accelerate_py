/**
 * Browser-Specific Fusion and Quantization Integration Test
 * 
 * This test evaluates the combined performance of browser-specific optimizations
 * with operation fusion and quantization across different browsers.
 */

import { Tensor } from '../src/tensor/tensor';
import { WebGPUBackend } from '../src/hardware/webgpu/backend';
import { WebGPUOperationFusion, FusionPattern, FusionConfig } from '../src/hardware/webgpu/optimizations/operation_fusion';
import { BrowserType, detectBrowserType } from '../src/hardware/webgpu/browser_optimized_operations';
import {
  getOptimalWorkgroupSize,
  getOptimalTileSize
} from '../src/hardware/webgpu/optimizations/browser_specific_shaders';

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
 * CPU implementation of tanh activation
 */
function cpuTanh(input: Tensor<number>): Tensor<number> {
  const resultData = input.data.map(v => Math.tanh(v));
  return new Tensor<number>(input.shape, resultData, { dataType: 'float32' });
}

/**
 * CPU implementation of GELU activation (approximate)
 */
function cpuGelu(input: Tensor<number>): Tensor<number> {
  // GELU approximation: 0.5 * x * (1 + tanh((2/π)^0.5 * (x + 0.044715 * x^3)))
  const sqrt2OverPi = Math.sqrt(2 / Math.PI);
  const resultData = input.data.map(v => {
    const x3 = v * v * v;
    return 0.5 * v * (1 + Math.tanh(sqrt2OverPi * (v + 0.044715 * x3)));
  });
  return new Tensor<number>(input.shape, resultData, { dataType: 'float32' });
}

/**
 * Check if two tensors are approximately equal (within epsilon)
 */
function areTensorsEqual(a: Tensor<number>, b: Tensor<number>, epsilon: number = 1e-4): boolean {
  if (a.shape.length !== b.shape.length) return false;
  if (!a.shape.every((dim, i) => dim === b.shape[i])) return false;
  
  let maxDifference = 0;
  let sumDifference = 0;
  let diffCount = 0;
  
  for (let i = 0; i < a.data.length; i++) {
    const diff = Math.abs(a.data[i] - b.data[i]);
    if (diff > epsilon) {
      diffCount++;
      sumDifference += diff;
      maxDifference = Math.max(maxDifference, diff);
      
      // For debugging, print a few mismatches
      if (diffCount <= 5) {
        console.error(`Mismatch at index ${i}: ${a.data[i]} vs ${b.data[i]} (diff: ${diff})`);
      }
    }
  }
  
  if (diffCount > 0) {
    console.error(`Found ${diffCount} mismatches out of ${a.data.length} elements`);
    console.error(`Max difference: ${maxDifference}, Average difference: ${sumDifference / diffCount}`);
    return false;
  }
  
  return true;
}

/**
 * Test different bit-width quantized operations with browser-specific optimizations
 */
async function testBrowserSpecificQuantization(
  backend: WebGPUBackend,
  matrixA: Tensor<number>,
  matrixB: Tensor<number>
): Promise<{
  results: Record<string, Record<string, number>>,
  accuracies: Record<string, Record<string, number>>,
  correctness: Record<string, Record<string, boolean>>
}> {
  console.log("\n=== Testing Browser-Specific Quantization ===");
  
  const browserType = detectBrowserType();
  const browserNames = {
    [BrowserType.CHROME]: 'Chrome',
    [BrowserType.FIREFOX]: 'Firefox',
    [BrowserType.SAFARI]: 'Safari',
    [BrowserType.EDGE]: 'Edge',
    [BrowserType.UNKNOWN]: 'Unknown'
  };
  
  console.log(`Detected browser: ${browserNames[browserType]}`);
  console.log(`Matrix A shape: ${matrixA.shape.join('x')}`);
  console.log(`Matrix B shape: ${matrixB.shape.join('x')}`);
  
  // Get browser-specific optimal parameters
  const optimalWorkgroupSize = getOptimalWorkgroupSize(browserType, 'matmul');
  const matrixSize = matrixA.shape[0] * matrixA.shape[1];
  const optimalTileSize = getOptimalTileSize(browserType, matrixSize);
  
  console.log(`Using optimal workgroup size: ${optimalWorkgroupSize}`);
  console.log(`Using optimal tile size: ${optimalTileSize}`);
  
  // Bit widths to test
  const bitWidths: (1 | 2 | 3 | 4 | 8)[] = [8, 4, 3, 2, 1];
  
  // Run CPU reference implementation
  console.log("Computing reference result with CPU...");
  const cpuResult = cpuMatmul(matrixA, matrixB);
  
  // Results storage
  const results: Record<string, Record<string, number>> = {};
  const accuracies: Record<string, Record<string, number>> = {};
  const correctness: Record<string, Record<string, boolean>> = {};
  
  // Test each bit width
  for (const bits of bitWidths) {
    console.log(`\nTesting ${bits}-bit quantization:`);
    results[`${bits}-bit`] = {};
    accuracies[`${bits}-bit`] = {};
    correctness[`${bits}-bit`] = {};
    
    // Standard implementation (no browser optimizations)
    console.log("Running standard implementation (no browser optimizations)...");
    const startStandard = performance.now();
    const standardResult = await backend.matmul(matrixA, matrixB, {
      useQuantization: true,
      bitsPerWeight: bits,
      useBrowserOptimizations: false
    });
    const endStandard = performance.now();
    const standardTime = endStandard - startStandard;
    results[`${bits}-bit`]['standard'] = standardTime;
    
    // Calculate accuracy against CPU result
    const standardAccuracy = calculateAccuracy(cpuResult, standardResult);
    accuracies[`${bits}-bit`]['standard'] = standardAccuracy;
    
    // Check correctness with tolerance based on bit width
    const epsilon = bits <= 2 ? 0.1 : (bits <= 4 ? 0.05 : 0.01);
    const standardCorrect = areTensorsEqual(cpuResult, standardResult, epsilon);
    correctness[`${bits}-bit`]['standard'] = standardCorrect;
    
    console.log(`Standard implementation time: ${standardTime.toFixed(2)}ms`);
    console.log(`Standard implementation accuracy: ${standardAccuracy.toFixed(2)}%`);
    console.log(`Standard implementation correctness: ${standardCorrect ? 'PASSED' : 'FAILED'}`);
    
    // Browser-optimized implementation
    console.log("Running browser-optimized implementation...");
    const startOptimized = performance.now();
    const optimizedResult = await backend.matmul(matrixA, matrixB, {
      useQuantization: true,
      bitsPerWeight: bits,
      useBrowserOptimizations: true,
      browserType: browserType,
      workgroupSize: optimalWorkgroupSize,
      tileSize: optimalTileSize
    });
    const endOptimized = performance.now();
    const optimizedTime = endOptimized - startOptimized;
    results[`${bits}-bit`]['optimized'] = optimizedTime;
    
    // Calculate accuracy against CPU result
    const optimizedAccuracy = calculateAccuracy(cpuResult, optimizedResult);
    accuracies[`${bits}-bit`]['optimized'] = optimizedAccuracy;
    
    // Check correctness
    const optimizedCorrect = areTensorsEqual(cpuResult, optimizedResult, epsilon);
    correctness[`${bits}-bit`]['optimized'] = optimizedCorrect;
    
    console.log(`Browser-optimized implementation time: ${optimizedTime.toFixed(2)}ms`);
    console.log(`Browser-optimized implementation accuracy: ${optimizedAccuracy.toFixed(2)}%`);
    console.log(`Browser-optimized implementation correctness: ${optimizedCorrect ? 'PASSED' : 'FAILED'}`);
    
    // Calculate speedup
    const speedup = standardTime / optimizedTime;
    console.log(`Browser optimization speedup: ${speedup.toFixed(2)}x`);
  }
  
  // Print summary table
  console.log("\n=== Browser-Specific Quantization Summary ===");
  console.log("Bit-width | Standard (ms) | Optimized (ms) | Speedup | Accuracy | Correctness");
  console.log("----------|---------------|----------------|---------|----------|------------");
  
  for (const bits of bitWidths) {
    const standardTime = results[`${bits}-bit`]['standard'];
    const optimizedTime = results[`${bits}-bit`]['optimized'];
    const speedup = standardTime / optimizedTime;
    const accuracy = accuracies[`${bits}-bit`]['optimized'];
    const correct = correctness[`${bits}-bit`]['optimized'];
    
    console.log(`${bits}-bit     | ${standardTime.toFixed(2)}ms        | ${optimizedTime.toFixed(2)}ms         | ${speedup.toFixed(2)}x   | ${accuracy.toFixed(2)}%   | ${correct ? 'PASSED' : 'FAILED'}`);
  }
  
  return { results, accuracies, correctness };
}

/**
 * Calculate accuracy between reference and test tensor
 * Returns percentage (0-100) representing accuracy
 */
function calculateAccuracy(reference: Tensor<number>, test: Tensor<number>): number {
  let errorSum = 0;
  const refAbsMax = Math.max(...reference.data.map(Math.abs));
  
  for (let i = 0; i < reference.data.length; i++) {
    const error = Math.abs(reference.data[i] - test.data[i]);
    errorSum += error;
  }
  
  const avgError = errorSum / reference.data.length;
  // Calculate accuracy as percentage (100% minus normalized error)
  return 100 - (avgError / refAbsMax * 100);
}

/**
 * Test different browsers' fusion patterns on same hardware
 */
async function testFusionPatternsByBrowser(
  backend: WebGPUBackend,
  matrixA: Tensor<number>,
  matrixB: Tensor<number>,
  bitsPerWeight: 1 | 2 | 3 | 4 | 8 = 4
): Promise<{
  results: Record<string, Record<string, number>>
}> {
  console.log("\n=== Testing Fusion Patterns Across Browser Optimizations ===");
  
  const actualBrowser = detectBrowserType();
  const browserTypes = [
    BrowserType.CHROME,
    BrowserType.FIREFOX,
    BrowserType.SAFARI,
    BrowserType.EDGE
  ];
  
  const browserNames = {
    [BrowserType.CHROME]: 'Chrome',
    [BrowserType.FIREFOX]: 'Firefox',
    [BrowserType.SAFARI]: 'Safari',
    [BrowserType.EDGE]: 'Edge'
  };
  
  console.log(`Actual browser: ${browserNames[actualBrowser]}`);
  console.log(`Testing different browser optimizations with ${bitsPerWeight}-bit quantization`);
  
  // Fusion patterns to test
  const fusionPatterns = [
    { name: 'QuantizedMatmul', pattern: FusionPattern.QuantizedMatmul },
    { name: 'QuantizedMatmulActivation', pattern: FusionPattern.QuantizedMatmulActivation },
    { name: 'QuantizedAttention', pattern: FusionPattern.QuantizedAttention },
    { name: 'MatrixChain', pattern: FusionPattern.MatrixChain }
  ];
  
  // Results storage
  const results: Record<string, Record<string, number>> = {};
  for (const browser of browserTypes) {
    results[browserNames[browser]] = {};
  }
  
  // Prepare CPU reference result
  const cpuMatmulResult = cpuMatmul(matrixA, matrixB);
  const cpuReluResult = cpuRelu(cpuMatmulResult);
  
  // Test each browser optimization with each fusion pattern
  for (const browser of browserTypes) {
    console.log(`\nTesting ${browserNames[browser]} optimizations:`);
    
    for (const { name, pattern } of fusionPatterns) {
      if (pattern === FusionPattern.QuantizedMatmul) {
        console.log(`Testing ${name} pattern...`);
        
        const fusionConfig: FusionConfig = {
          useQuantizedWeights: true,
          bitsPerWeight,
          useBrowserOptimizations: true,
          browserOptimizationType: browser,
          enabledPatterns: [pattern]
        };
        
        const fusion = new WebGPUOperationFusion(backend, fusionConfig);
        
        const startTime = performance.now();
        // Execute the operation (in a real scenario, we would use fusion.execute())
        const result = await backend.matmul(matrixA, matrixB, { 
          useQuantization: true,
          bitsPerWeight,
          useBrowserOptimizations: true,
          browserType: browser
        });
        const endTime = performance.now();
        
        results[browserNames[browser]][name] = endTime - startTime;
        console.log(`${name} execution time: ${(endTime - startTime).toFixed(2)}ms`);
        
        // Check correctness (with higher tolerance for low-bit quantization)
        const epsilon = bitsPerWeight <= 2 ? 0.1 : (bitsPerWeight <= 4 ? 0.05 : 0.01);
        const correct = areTensorsEqual(cpuMatmulResult, result, epsilon);
        console.log(`${name} correctness: ${correct ? 'PASSED' : 'FAILED'}`);
        
      } else if (pattern === FusionPattern.QuantizedMatmulActivation) {
        console.log(`Testing ${name} pattern...`);
        
        const fusionConfig: FusionConfig = {
          useQuantizedWeights: true,
          bitsPerWeight,
          useBrowserOptimizations: true,
          browserOptimizationType: browser,
          enabledPatterns: [pattern]
        };
        
        const fusion = new WebGPUOperationFusion(backend, fusionConfig);
        
        const startTime = performance.now();
        // Execute the fused operation
        const result = await backend.executeOperations(
          [matrixA, matrixB],
          ['matmul', 'relu'],
          { 
            useQuantization: true,
            bitsPerWeight,
            useBrowserOptimizations: true,
            browserType: browser,
            useFusion: true
          }
        );
        const endTime = performance.now();
        
        results[browserNames[browser]][name] = endTime - startTime;
        console.log(`${name} execution time: ${(endTime - startTime).toFixed(2)}ms`);
        
        // Check correctness
        const epsilon = bitsPerWeight <= 2 ? 0.15 : (bitsPerWeight <= 4 ? 0.08 : 0.02);
        const correct = areTensorsEqual(cpuReluResult, result, epsilon);
        console.log(`${name} correctness: ${correct ? 'PASSED' : 'FAILED'}`);
      } else {
        // Skip other patterns for simplicity in this test
        console.log(`Skipping ${name} pattern (not fully implemented in test)`);
        results[browserNames[browser]][name] = 0;
      }
    }
  }
  
  // Print summary table
  console.log("\n=== Browser Optimization Fusion Pattern Summary ===");
  let headerRow = "Pattern               |";
  for (const browser of browserTypes) {
    headerRow += ` ${browserNames[browser]} (ms) |`;
  }
  console.log(headerRow);
  
  let separatorRow = "-----------------------|";
  for (const browser of browserTypes) {
    separatorRow += "--------------|";
  }
  console.log(separatorRow);
  
  for (const { name } of fusionPatterns) {
    let row = `${name.padEnd(22)}|`;
    for (const browser of browserTypes) {
      const time = results[browserNames[browser]][name];
      row += ` ${time > 0 ? time.toFixed(2) : 'N/A'} ms      |`;
    }
    console.log(row);
  }
  
  return { results };
}

/**
 * Test fusion pattern with different activations
 */
async function testActivationFusions(
  backend: WebGPUBackend,
  matrixA: Tensor<number>,
  matrixB: Tensor<number>,
  bitsPerWeight: 1 | 2 | 3 | 4 | 8 = 4
): Promise<{
  results: Record<string, Record<string, number>>,
  speedups: Record<string, number>
}> {
  console.log("\n=== Testing Different Activation Functions with Fusion ===");
  
  const browserType = detectBrowserType();
  const browserNames = {
    [BrowserType.CHROME]: 'Chrome',
    [BrowserType.FIREFOX]: 'Firefox',
    [BrowserType.SAFARI]: 'Safari',
    [BrowserType.EDGE]: 'Edge',
    [BrowserType.UNKNOWN]: 'Unknown'
  };
  
  console.log(`Detected browser: ${browserNames[browserType]}`);
  console.log(`Testing ${bitsPerWeight}-bit quantization with different activations`);
  
  // Activation functions to test
  const activations = ['relu', 'tanh', 'gelu'];
  
  // Results storage
  const results: Record<string, Record<string, number>> = {};
  const speedups: Record<string, number> = {};
  
  // Compute reference result
  const cpuMatmulResult = cpuMatmul(matrixA, matrixB);
  
  // Test each activation function
  for (const activation of activations) {
    console.log(`\nTesting ${activation} activation:`);
    results[activation] = {};
    
    // Compute CPU reference with activation
    let cpuActivationResult;
    if (activation === 'relu') {
      cpuActivationResult = cpuRelu(cpuMatmulResult);
    } else if (activation === 'tanh') {
      cpuActivationResult = cpuTanh(cpuMatmulResult);
    } else if (activation === 'gelu') {
      cpuActivationResult = cpuGelu(cpuMatmulResult);
    }
    
    // Test unfused operations (separate matmul and activation)
    console.log("Running unfused operations...");
    const startUnfused = performance.now();
    
    // First do matmul
    const unfusedMatmul = await backend.matmul(matrixA, matrixB, { 
      useQuantization: true,
      bitsPerWeight,
      useBrowserOptimizations: true,
      browserType
    });
    
    // Then do activation
    let unfusedResult;
    if (activation === 'relu') {
      unfusedResult = await backend.relu(unfusedMatmul);
    } else if (activation === 'tanh') {
      unfusedResult = await backend.tanh(unfusedMatmul);
    } else if (activation === 'gelu') {
      unfusedResult = await backend.gelu(unfusedMatmul);
    }
    
    const endUnfused = performance.now();
    const unfusedTime = endUnfused - startUnfused;
    results[activation]['unfused'] = unfusedTime;
    
    console.log(`Unfused operations time: ${unfusedTime.toFixed(2)}ms`);
    
    // Check correctness of unfused operations
    const unfusedEpsilon = bitsPerWeight <= 2 ? 0.15 : (bitsPerWeight <= 4 ? 0.08 : 0.02);
    const unfusedCorrect = areTensorsEqual(cpuActivationResult, unfusedResult, unfusedEpsilon);
    console.log(`Unfused operations correctness: ${unfusedCorrect ? 'PASSED' : 'FAILED'}`);
    
    // Test fused operations
    console.log("Running fused operations...");
    
    const fusionConfig: FusionConfig = {
      useQuantizedWeights: true,
      bitsPerWeight,
      useBrowserOptimizations: true,
      browserOptimizationType: browserType,
      enabledPatterns: [FusionPattern.QuantizedMatmulActivation]
    };
    
    const fusion = new WebGPUOperationFusion(backend, fusionConfig);
    
    const startFused = performance.now();
    const fusedResult = await backend.executeOperations(
      [matrixA, matrixB],
      ['matmul', activation],
      { 
        useQuantization: true,
        bitsPerWeight,
        useBrowserOptimizations: true,
        browserType,
        useFusion: true
      }
    );
    const endFused = performance.now();
    const fusedTime = endFused - startFused;
    results[activation]['fused'] = fusedTime;
    
    console.log(`Fused operations time: ${fusedTime.toFixed(2)}ms`);
    
    // Check correctness of fused operations
    const fusedEpsilon = bitsPerWeight <= 2 ? 0.15 : (bitsPerWeight <= 4 ? 0.08 : 0.02);
    const fusedCorrect = areTensorsEqual(cpuActivationResult, fusedResult, fusedEpsilon);
    console.log(`Fused operations correctness: ${fusedCorrect ? 'PASSED' : 'FAILED'}`);
    
    // Calculate speedup
    const speedup = unfusedTime / fusedTime;
    speedups[activation] = speedup;
    console.log(`Fusion speedup: ${speedup.toFixed(2)}x`);
  }
  
  // Print summary table
  console.log("\n=== Activation Fusion Summary ===");
  console.log("Activation | Unfused (ms) | Fused (ms) | Speedup");
  console.log("-----------|--------------|------------|--------");
  
  for (const activation of activations) {
    const unfusedTime = results[activation]['unfused'];
    const fusedTime = results[activation]['fused'];
    const speedup = speedups[activation];
    
    console.log(`${activation.padEnd(10)}| ${unfusedTime.toFixed(2)}ms${' '.repeat(10 - unfusedTime.toFixed(2).length)}| ${fusedTime.toFixed(2)}ms${' '.repeat(8 - fusedTime.toFixed(2).length)}| ${speedup.toFixed(2)}x`);
  }
  
  return { results, speedups };
}

/**
 * Run all tests
 */
async function runTests() {
  console.log("Starting Browser-Specific Fusion and Quantization tests...");
  
  try {
    // Initialize WebGPU backend
    const backend = new WebGPUBackend();
    await backend.initialize();
    console.log("WebGPU backend initialized successfully");
    
    // Create test matrices
    console.log("Generating test matrices...");
    
    // Small matrices for faster testing
    const smallMatrixA = generateRandomTensor([64, 32]);
    const smallMatrixB = generateRandomTensor([32, 64]);
    
    // Medium matrices for more realistic testing
    const mediumMatrixA = generateRandomTensor([256, 128]);
    const mediumMatrixB = generateRandomTensor([128, 64]);
    
    // Run the tests
    await testBrowserSpecificQuantization(backend, smallMatrixA, smallMatrixB);
    await testFusionPatternsByBrowser(backend, smallMatrixA, smallMatrixB, 4);
    await testActivationFusions(backend, smallMatrixA, smallMatrixB, 4);
    
    // Run additional test with medium matrices if time permits
    console.log("\n=== Running tests with medium-sized matrices ===");
    await testBrowserSpecificQuantization(backend, mediumMatrixA, mediumMatrixB);
    
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