/**
 * Tests for browser-optimized WebGPU operations
 */

import { WebGPUBackend } from '../src/hardware/webgpu/backend';
import { Tensor } from '../src/tensor/tensor';
import { 
  detectBrowserType,
  BrowserType,
  getBrowserCapabilities 
} from '../src/hardware/webgpu/browser_optimized_operations';

describe('Browser-optimized WebGPU operations', () => {
  let backend: WebGPUBackend;
  
  beforeAll(async () => {
    // Skip tests if WebGPU is not available in this environment
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
      return;
    }
    
    // Initialize backend
    backend = new WebGPUBackend();
    await backend.initialize();
  });
  
  afterAll(async () => {
    // Skip cleanup if WebGPU is not available
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
      return;
    }
    
    // Clean up resources
    if (backend) {
      backend.dispose();
    }
  });
  
  test('Should detect browser type', () => {
    // Skip test if running in Node environment
    if (typeof navigator === 'undefined') {
      return;
    }
    
    // Call the function
    const browserType = detectBrowserType();
    
    // Should return a valid browser type
    expect(Object.values(BrowserType)).toContain(browserType);
    console.log(`Detected browser: ${browserType}`);
  });
  
  test('Should retrieve browser capabilities when WebGPU is available', async () => {
    // Skip test if WebGPU is not supported
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
      return;
    }
    
    // Get browser capabilities
    const browserType = backend.getBrowserType();
    const capabilities = backend.getBrowserCapabilities();
    
    // Verify capabilities are retrieved
    expect(capabilities).not.toBeNull();
    expect(browserType).toBe(capabilities?.browserType);
    
    // Log capabilities for debugging
    console.log(`Browser capabilities:`, {
      browserType: capabilities?.browserType,
      version: capabilities?.version,
      performanceTier: capabilities?.performanceTier,
      hardware: capabilities?.hardware,
      optimalWorkgroupSizes: capabilities?.optimalWorkgroupSizes,
      optimalTileSizes: capabilities?.optimalTileSizes,
      optimizationFlags: capabilities?.optimizationFlags
    });
  });
  
  test('Should perform optimized matrix multiplication', async () => {
    // Skip test if WebGPU is not supported
    if (typeof navigator === 'undefined' || !('gpu' in navigator) || !backend.isInitialized()) {
      return;
    }
    
    // Create sample matrices
    const matrixA = new Tensor<number>(
      [4, 4],
      new Float32Array([
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
      ]),
      { dataType: 'float32' }
    );
    
    const matrixB = new Tensor<number>(
      [4, 4],
      new Float32Array([
        16, 15, 14, 13,
        12, 11, 10, 9,
        8, 7, 6, 5,
        4, 3, 2, 1
      ]),
      { dataType: 'float32' }
    );
    
    // Test with browser optimizations enabled
    backend.setBrowserOptimizationsEnabled(true);
    const resultWithOptimizations = await backend.matmul(matrixA, matrixB);
    
    // Test same operation with browser optimizations disabled
    backend.setBrowserOptimizationsEnabled(false);
    const resultWithoutOptimizations = await backend.matmul(matrixA, matrixB);
    
    // Re-enable optimizations for other tests
    backend.setBrowserOptimizationsEnabled(true);
    
    // Results should be the same regardless of optimization
    expect(resultWithOptimizations.shape).toEqual(resultWithoutOptimizations.shape);
    
    // Compare result values (allow small floating point difference)
    const resultDataWithOpt = resultWithOptimizations.getData() as Float32Array;
    const resultDataWithoutOpt = resultWithoutOptimizations.getData() as Float32Array;
    
    for (let i = 0; i < resultDataWithOpt.length; i++) {
      expect(Math.abs(resultDataWithOpt[i] - resultDataWithoutOpt[i])).toBeLessThan(1e-5);
    }
    
    // Expected result matrix after multiplication (for verification)
    const expectedResult = [
      80, 70, 60, 50,
      240, 214, 188, 162,
      400, 358, 316, 274,
      560, 502, 444, 386
    ];
    
    // Verify against expected matrix multiplication result
    const resultData = resultWithOptimizations.getData() as Float32Array;
    for (let i = 0; i < resultData.length; i++) {
      expect(Math.abs(resultData[i] - expectedResult[i])).toBeLessThan(1e-5);
    }
  });
  
  test('Should perform optimized batch matrix multiplication', async () => {
    // Skip test if WebGPU is not supported
    if (typeof navigator === 'undefined' || !('gpu' in navigator) || !backend.isInitialized()) {
      return;
    }
    
    // Create sample batch matrices (2 matrices in the batch)
    const batchA = new Tensor<number>(
      [2, 2, 3], // [batchSize, M, K]
      new Float32Array([
        // Batch item 0
        1, 2, 3,
        4, 5, 6,
        // Batch item 1
        7, 8, 9,
        10, 11, 12
      ]),
      { dataType: 'float32' }
    );
    
    const batchB = new Tensor<number>(
      [2, 3, 2], // [batchSize, K, N]
      new Float32Array([
        // Batch item 0
        1, 2,
        3, 4,
        5, 6,
        // Batch item 1
        7, 8,
        9, 10,
        11, 12
      ]),
      { dataType: 'float32' }
    );
    
    // Test with browser optimizations enabled
    backend.setBrowserOptimizationsEnabled(true);
    const resultWithOptimizations = await backend.batchMatmul(batchA, batchB);
    
    // Expected output shape [2, 2, 2]
    expect(resultWithOptimizations.shape).toEqual([2, 2, 2]);
    
    // Expected result after batch matrix multiplication
    const expectedResult = [
      // Batch item 0: [2x3] × [3x2]
      22, 28,
      49, 64,
      // Batch item 1: [2x3] × [3x2]
      202, 226,
      283, 316
    ];
    
    // Verify against expected batch matrix multiplication result
    const resultData = resultWithOptimizations.getData() as Float32Array;
    for (let i = 0; i < resultData.length; i++) {
      expect(Math.abs(resultData[i] - expectedResult[i])).toBeLessThan(1e-5);
    }
  });
  
  test('Should perform benchmarking and optimization', async () => {
    // Skip test if WebGPU is not supported
    if (typeof navigator === 'undefined' || !('gpu' in navigator) || !backend.isInitialized()) {
      return;
    }
    
    // Create larger matrices for benchmarking
    const SIZE = 128;
    const matrixA = new Tensor<number>(
      [SIZE, SIZE],
      new Float32Array(SIZE * SIZE).fill(1.0),
      { dataType: 'float32' }
    );
    
    const matrixB = new Tensor<number>(
      [SIZE, SIZE],
      new Float32Array(SIZE * SIZE).fill(1.0),
      { dataType: 'float32' }
    );
    
    // Get browser capabilities to log
    const browserType = backend.getBrowserType();
    const capabilities = backend.getBrowserCapabilities();
    
    // Output capabilities for reference
    console.log(`Running benchmark on ${browserType}, tier ${capabilities?.performanceTier}`);
    console.log(`Optimal workgroup size: ${capabilities?.optimalWorkgroupSizes.matmul}`);
    console.log(`Optimal tile size: ${capabilities?.optimalTileSizes.matmul}`);
    
    // Measure execution time with browser optimizations enabled
    const startWithOpt = performance.now();
    await backend.matmul(matrixA, matrixB);
    const endWithOpt = performance.now();
    
    // Measure execution time with browser optimizations disabled
    backend.setBrowserOptimizationsEnabled(false);
    const startWithoutOpt = performance.now();
    await backend.matmul(matrixA, matrixB);
    const endWithoutOpt = performance.now();
    
    // Re-enable optimizations
    backend.setBrowserOptimizationsEnabled(true);
    
    // Log performance metrics
    const timeWithOpt = endWithOpt - startWithOpt;
    const timeWithoutOpt = endWithoutOpt - startWithoutOpt;
    console.log(`Matrix multiplication (${SIZE}x${SIZE}) times:`);
    console.log(`- With browser optimizations: ${timeWithOpt.toFixed(2)}ms`);
    console.log(`- Without browser optimizations: ${timeWithoutOpt.toFixed(2)}ms`);
    console.log(`- Improvement: ${(timeWithoutOpt / timeWithOpt).toFixed(2)}x`);
    
    // We don't assert on performance numbers as they can vary significantly
    // between environments, but we expect the function to complete without errors
  });
});