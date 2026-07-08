/**
 * Example demonstrating WebGPU Matrix Operations
 * 
 * This example shows how to use WebGPU Matrix Operations with browser-specific optimizations.
 * It demonstrates matrix multiplication with different strategies and compares performance.
 */

import { BrowserOptimizedMatrixOperations, MatrixMultiplyStrategy } from './ipfs_accelerate_js_matrix_operations';
import { GPUBufferUtils } from './ipfs_accelerate_js_webgpu_backend';

/**
 * Main entry point for the example
 */
async function main() {
  console.log("WebGPU Matrix Operations Example");
  
  // Check if WebGPU is supported
  if (!navigator.gpu) {
    console.error("WebGPU is not supported in this browser");
    displayError("WebGPU is not supported in this browser. Try using Chrome 113+ or Edge 113+.");
    return;
  }
  
  try {
    // Request adapter and device
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("Couldn't request WebGPU adapter");
    }
    
    const device = await adapter.requestDevice();
    console.log("WebGPU device acquired");
    
    // Create buffer utils
    const bufferUtils = new GPUBufferUtils(device);
    
    // Detect browser
    const browserInfo = detectBrowser();
    console.log(`Detected browser: ${browserInfo.browserType} ${browserInfo.browserVersion}`);
    console.log(`GPU vendor: ${browserInfo.gpuVendor}`);
    
    // Create browser-optimized matrix operations
    const matrixOps = new BrowserOptimizedMatrixOperations(
      device, 
      bufferUtils, 
      browserInfo
    );
    
    await matrixOps.initialize();
    console.log("Matrix operations initialized");
    
    // Run matrix multiplications with different strategies
    await runMatrixMultiplicationTests(matrixOps);
    
    // Run browser optimization tests
    await runBrowserOptimizationTests(matrixOps);
    
    console.log("Example completed successfully");
  } catch (error) {
    console.error("Error running example:", error);
    displayError(`Error: ${error.message}`);
  }
}

/**
 * Run matrix multiplication tests with different strategies
 */
async function runMatrixMultiplicationTests(matrixOps: BrowserOptimizedMatrixOperations) {
  console.log("Running matrix multiplication tests...");
  
  // Define matrix sizes to test
  const sizes = [
    { name: "Small (32x32)", size: 32 },
    { name: "Medium (128x128)", size: 128 },
    { name: "Large (512x512)", size: 512 }
  ];
  
  const results = document.getElementById('results');
  results.innerHTML = '<h3>Matrix Multiplication Performance Tests</h3>';
  
  // Test each size
  for (const { name, size } of sizes) {
    results.innerHTML += `<h4>${name} Matrix Test (${size}x${size})</h4>`;
    const table = document.createElement('table');
    table.innerHTML = `
      <tr>
        <th>Strategy</th>
        <th>Execution Time (ms)</th>
        <th>Status</th>
      </tr>
    `;
    results.appendChild(table);
    
    // Create matrices
    const M = size;
    const N = size;
    const K = size;
    
    const matrixA = createRandomMatrix(M, K);
    const matrixB = createRandomMatrix(K, N);
    
    // Calculate result using CPU for verification
    const cpuResult = calculateMatrixMultiplicationCPU(matrixA, matrixB, M, N, K);
    
    // Test different strategies
    const strategies = [
      MatrixMultiplyStrategy.SIMPLE,
      MatrixMultiplyStrategy.TILED,
      MatrixMultiplyStrategy.MICRO_TILED,
      MatrixMultiplyStrategy.AUTO
    ];
    
    for (const strategy of strategies) {
      try {
        // Use WebGPUMatrixMultiplication directly for explicit strategy testing
        const startTime = performance.now();
        
        // For browser optimized version, call native matmul
        const result = await (matrixOps as any).matrixOps.matmul(
          matrixA, matrixB, M, N, K, strategy
        );
        
        const endTime = performance.now();
        const executionTime = endTime - startTime;
        
        // Verify result
        const isCorrect = verifyResult(result, cpuResult);
        
        // Add to table
        const row = document.createElement('tr');
        row.innerHTML = `
          <td>${strategy}</td>
          <td>${executionTime.toFixed(2)}</td>
          <td>${isCorrect ? '✅ Correct' : '❌ Error'}</td>
        `;
        table.appendChild(row);
        
        console.log(`Strategy ${strategy} took ${executionTime.toFixed(2)}ms for ${name}`);
      } catch (error) {
        console.error(`Error with strategy ${strategy} for ${name}:`, error);
        
        // Add error to table
        const row = document.createElement('tr');
        row.innerHTML = `
          <td>${strategy}</td>
          <td>-</td>
          <td>❌ Error: ${error.message}</td>
        `;
        table.appendChild(row);
      }
    }
  }
}

/**
 * Run browser optimization tests to show the impact of browser-specific optimizations
 */
async function runBrowserOptimizationTests(matrixOps: BrowserOptimizedMatrixOperations) {
  console.log("Running browser optimization tests...");
  
  const results = document.getElementById('results');
  results.innerHTML += '<h3>Browser Optimization Tests</h3>';
  results.innerHTML += '<p>This test compares the browser-optimized strategy selection with fixed strategies.</p>';
  
  // Use a 256x256 matrix for testing
  const M = 256;
  const N = 256;
  const K = 256;
  
  const matrixA = createRandomMatrix(M, K);
  const matrixB = createRandomMatrix(K, N);
  
  const table = document.createElement('table');
  table.innerHTML = `
    <tr>
      <th>Approach</th>
      <th>Execution Time (ms)</th>
      <th>Strategy Selected</th>
    </tr>
  `;
  results.appendChild(table);
  
  // Test browser-optimized approach
  try {
    const startTime = performance.now();
    
    // Use the browser-optimized matmul which selects strategy automatically
    const result = await matrixOps.matmul(matrixA, matrixB, M, N, K);
    
    const endTime = performance.now();
    const executionTime = endTime - startTime;
    
    // Get the strategy that was selected (approximate by checking execution time pattern)
    const strategies = [
      MatrixMultiplyStrategy.SIMPLE,
      MatrixMultiplyStrategy.TILED,
      MatrixMultiplyStrategy.MICRO_TILED
    ];
    
    // Run with each explicit strategy to match execution pattern
    const strategyTimes = [];
    
    for (const strategy of strategies) {
      const strategyStartTime = performance.now();
      await (matrixOps as any).matrixOps.matmul(matrixA, matrixB, M, N, K, strategy);
      const strategyEndTime = performance.now();
      strategyTimes.push({ 
        strategy, 
        time: strategyEndTime - strategyStartTime 
      });
    }
    
    // Find closest match
    strategyTimes.sort((a, b) => 
      Math.abs(a.time - executionTime) - Math.abs(b.time - executionTime)
    );
    
    const selectedStrategy = strategyTimes[0].strategy;
    
    // Add to table
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>Browser-Optimized</td>
      <td>${executionTime.toFixed(2)}</td>
      <td>${selectedStrategy}</td>
    `;
    table.appendChild(row);
    
    // Add rows for each fixed strategy for comparison
    for (const { strategy, time } of strategyTimes) {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>Fixed (${strategy})</td>
        <td>${time.toFixed(2)}</td>
        <td>-</td>
      `;
      table.appendChild(row);
    }
    
    console.log(`Browser-optimized approach took ${executionTime.toFixed(2)}ms, selected ${selectedStrategy}`);
  } catch (error) {
    console.error("Error in browser optimization test:", error);
    
    // Add error to table
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>Browser-Optimized</td>
      <td>-</td>
      <td>❌ Error: ${error.message}</td>
    `;
    table.appendChild(row);
  }
}

/**
 * Create a random matrix of specified dimensions
 */
function createRandomMatrix(rows: number, cols: number): Float32Array {
  const matrix = new Float32Array(rows * cols);
  for (let i = 0; i < rows * cols; i++) {
    matrix[i] = Math.random() * 2 - 1; // Values between -1 and 1
  }
  return matrix;
}

/**
 * Calculate matrix multiplication on CPU for verification
 */
function calculateMatrixMultiplicationCPU(
  matrixA: Float32Array,
  matrixB: Float32Array,
  M: number,
  N: number,
  K: number
): Float32Array {
  const result = new Float32Array(M * N);
  
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += matrixA[i * K + k] * matrixB[k * N + j];
      }
      result[i * N + j] = sum;
    }
  }
  
  return result;
}

/**
 * Verify that GPU result matches CPU result within tolerance
 */
function verifyResult(gpuResult: Float32Array, cpuResult: Float32Array): boolean {
  if (gpuResult.length !== cpuResult.length) {
    return false;
  }
  
  const tolerance = 1e-5;
  
  for (let i = 0; i < gpuResult.length; i++) {
    if (Math.abs(gpuResult[i] - cpuResult[i]) > tolerance) {
      console.error(`Result mismatch at index ${i}: GPU=${gpuResult[i]}, CPU=${cpuResult[i]}`);
      return false;
    }
  }
  
  return true;
}

/**
 * Detect browser type, version, and GPU vendor
 */
function detectBrowser(): { browserType: string, browserVersion: string, gpuVendor: string } {
  const userAgent = navigator.userAgent;
  let browserType = 'unknown';
  let browserVersion = 'unknown';
  
  // Extract browser info from user agent
  if (userAgent.indexOf('Chrome') !== -1) {
    browserType = 'chrome';
    const match = userAgent.match(/Chrome\/(\d+\.\d+)/);
    if (match) {
      browserVersion = match[1];
    }
  } else if (userAgent.indexOf('Firefox') !== -1) {
    browserType = 'firefox';
    const match = userAgent.match(/Firefox\/(\d+\.\d+)/);
    if (match) {
      browserVersion = match[1];
    }
  } else if (userAgent.indexOf('Safari') !== -1) {
    browserType = 'safari';
    const match = userAgent.match(/Version\/(\d+\.\d+)/);
    if (match) {
      browserVersion = match[1];
    }
  } else if (userAgent.indexOf('Edg') !== -1) {
    browserType = 'edge';
    const match = userAgent.match(/Edg\/(\d+\.\d+)/);
    if (match) {
      browserVersion = match[1];
    }
  }
  
  // For GPU vendor, we would normally query the WebGPU adapter info
  // For simplicity in this example, we'll use a placeholder
  let gpuVendor = 'unknown';
  
  // In a real implementation, we'd use:
  // const gpuVendor = adapter.adapter_info?.vendor || 'unknown';
  
  // For this example, attempt to roughly determine GPU vendor from user agent
  if (userAgent.indexOf('NVIDIA') !== -1) {
    gpuVendor = 'nvidia';
  } else if (userAgent.indexOf('AMD') !== -1 || userAgent.indexOf('ATI') !== -1) {
    gpuVendor = 'amd';
  } else if (userAgent.indexOf('Intel') !== -1) {
    gpuVendor = 'intel';
  } else if (userAgent.indexOf('Apple') !== -1) {
    gpuVendor = 'apple';
  }
  
  return { browserType, browserVersion, gpuVendor };
}

/**
 * Display error message
 */
function displayError(message: string) {
  const errorElement = document.getElementById('error');
  if (errorElement) {
    errorElement.textContent = message;
    errorElement.style.display = 'block';
  }
}

// Start the example when the page loads
window.addEventListener('load', main);