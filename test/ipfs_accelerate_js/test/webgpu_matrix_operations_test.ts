/**
 * WebGPU Matrix Operations Test
 * Tests the performance and correctness of specialized matrix operations
 */

import { WebGPUBackend } from '../src/hardware/webgpu/backend';
import { Tensor } from '../src/tensor/tensor';
import { MatrixOperationOptions } from '../src/hardware/webgpu/matrix_operations';

/**
 * Simple matrix multiplication
 * @param a Matrix A
 * @param b Matrix B
 * @returns Result matrix C = A × B
 */
function cpuMatmul(a: number[][], b: number[][]): number[][] {
  const m = a.length;
  const n = b[0].length;
  const p = b.length;
  
  // Initialize result matrix with zeros
  const result: number[][] = Array(m).fill(0).map(() => Array(n).fill(0));
  
  // Perform matrix multiplication
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < p; k++) {
        sum += a[i][k] * b[k][j];
      }
      result[i][j] = sum;
    }
  }
  
  return result;
}

/**
 * Check if two matrices are approximately equal (within epsilon)
 * @param a First matrix
 * @param b Second matrix
 * @param epsilon Maximum allowed difference
 * @returns Whether matrices are approximately equal
 */
function areMatricesEqual(a: number[][], b: number[][], epsilon: number = 1e-4): boolean {
  if (a.length !== b.length || a[0].length !== b[0].length) {
    return false;
  }
  
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[0].length; j++) {
      if (Math.abs(a[i][j] - b[i][j]) > epsilon) {
        console.error(`Mismatch at [${i}, ${j}]: ${a[i][j]} vs ${b[i][j]}`);
        return false;
      }
    }
  }
  
  return true;
}

/**
 * Convert 2D array to tensor
 * @param matrix Input matrix as 2D array
 * @returns Tensor representation
 */
function matrixToTensor(matrix: number[][]): Tensor<number> {
  const shape = [matrix.length, matrix[0].length];
  const data = new Array(shape[0] * shape[1]);
  
  // Flatten matrix into 1D array
  for (let i = 0; i < shape[0]; i++) {
    for (let j = 0; j < shape[1]; j++) {
      data[i * shape[1] + j] = matrix[i][j];
    }
  }
  
  return new Tensor<number>(shape, data, { dataType: 'float32' });
}

/**
 * Convert tensor to 2D array
 * @param tensor Input tensor
 * @returns 2D array representation
 */
function tensorToMatrix(tensor: Tensor<number>): number[][] {
  if (tensor.shape.length !== 2) {
    throw new Error('Tensor must be 2D to convert to matrix');
  }
  
  const rows = tensor.shape[0];
  const cols = tensor.shape[1];
  const result = Array(rows).fill(0).map(() => Array(cols).fill(0));
  
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[i][j] = tensor.data[i * cols + j];
    }
  }
  
  return result;
}

/**
 * Generate a random matrix of specified size
 * @param rows Number of rows
 * @param cols Number of columns
 * @returns Random matrix
 */
function generateRandomMatrix(rows: number, cols: number): number[][] {
  return Array(rows).fill(0).map(() => 
    Array(cols).fill(0).map(() => Math.random() * 2 - 1)
  );
}

/**
 * Run performance test comparing different matrix multiplication implementations
 * @param m Number of rows in A
 * @param k Number of columns in A / rows in B
 * @param n Number of columns in B
 */
async function runPerformanceTest(m: number, k: number, n: number) {
  console.log(`\nRunning matrix multiplication performance test: ${m}x${k} × ${k}x${n}`);
  
  // Generate random matrices
  const matrixA = generateRandomMatrix(m, k);
  const matrixB = generateRandomMatrix(k, n);
  
  // Run CPU implementation for comparison
  console.log('Running CPU implementation...');
  const startTimeCpu = performance.now();
  const cpuResult = cpuMatmul(matrixA, matrixB);
  const endTimeCpu = performance.now();
  const cpuTime = endTimeCpu - startTimeCpu;
  console.log(`CPU implementation time: ${cpuTime.toFixed(2)}ms`);
  
  // Convert to tensors
  const tensorA = matrixToTensor(matrixA);
  const tensorB = matrixToTensor(matrixB);
  
  // Initialize WebGPU backend
  const backend = new WebGPUBackend();
  await backend.initialize();
  
  // Test standard implementation
  console.log('Running standard WebGPU implementation...');
  const startTimeStandard = performance.now();
  const standardResult = await backend.matmul(tensorA, tensorB, { useSharedMemory: false });
  const endTimeStandard = performance.now();
  const standardTime = endTimeStandard - startTimeStandard;
  console.log(`Standard WebGPU implementation time: ${standardTime.toFixed(2)}ms`);
  
  // Test tiled implementation
  console.log('Running tiled WebGPU implementation...');
  const startTimeTiled = performance.now();
  const tiledResult = await backend.matmul(tensorA, tensorB, { 
    useSharedMemory: true, 
    tileSize: 16
  });
  const endTimeTiled = performance.now();
  const tiledTime = endTimeTiled - startTimeTiled;
  console.log(`Tiled WebGPU implementation time: ${tiledTime.toFixed(2)}ms`);
  
  // Test advanced tiled implementation
  console.log('Running advanced tiled WebGPU implementation...');
  const startTimeAdvanced = performance.now();
  const advancedResult = await backend.matmul(tensorA, tensorB, { 
    useSharedMemory: true, 
    tileSize: 16, 
    useLayoutOptimizations: true
  });
  const endTimeAdvanced = performance.now();
  const advancedTime = endTimeAdvanced - startTimeAdvanced;
  console.log(`Advanced WebGPU implementation time: ${advancedTime.toFixed(2)}ms`);
  
  // Convert results back to matrices
  const standardMatrix = tensorToMatrix(standardResult);
  const tiledMatrix = tensorToMatrix(tiledResult);
  const advancedMatrix = tensorToMatrix(advancedResult);
  
  // Verify correctness
  const standardCorrect = areMatricesEqual(cpuResult, standardMatrix);
  const tiledCorrect = areMatricesEqual(cpuResult, tiledMatrix);
  const advancedCorrect = areMatricesEqual(cpuResult, advancedMatrix);
  
  console.log(`\nCorrectness verification:`);
  console.log(`Standard implementation: ${standardCorrect ? 'PASSED' : 'FAILED'}`);
  console.log(`Tiled implementation: ${tiledCorrect ? 'PASSED' : 'FAILED'}`);
  console.log(`Advanced implementation: ${advancedCorrect ? 'PASSED' : 'FAILED'}`);
  
  // Calculate speedups
  const standardSpeedup = cpuTime / standardTime;
  const tiledSpeedup = cpuTime / tiledTime;
  const advancedSpeedup = cpuTime / advancedTime;
  
  console.log(`\nSpeedup over CPU implementation:`);
  console.log(`Standard WebGPU: ${standardSpeedup.toFixed(2)}x`);
  console.log(`Tiled WebGPU: ${tiledSpeedup.toFixed(2)}x`);
  console.log(`Advanced WebGPU: ${advancedSpeedup.toFixed(2)}x`);
  
  // Cleanup
  backend.dispose();
}

/**
 * Test batch matrix multiplication
 * @param batchSize Number of matrices in batch
 * @param m Number of rows in each A
 * @param k Number of columns in each A / rows in each B
 * @param n Number of columns in each B
 */
async function testBatchMatmul(batchSize: number, m: number, k: number, n: number) {
  console.log(`\nTesting batch matrix multiplication: ${batchSize} × (${m}x${k} × ${k}x${n})`);
  
  // Generate random batch of matrices
  const batchMatrixA = Array(batchSize).fill(0).map(() => generateRandomMatrix(m, k));
  const batchMatrixB = Array(batchSize).fill(0).map(() => generateRandomMatrix(k, n));
  
  // Calculate expected results using CPU implementation
  const expectedResults = batchMatrixA.map((matrixA, i) => cpuMatmul(matrixA, batchMatrixB[i]));
  
  // Convert to tensors
  const tensorA = new Tensor<number>(
    [batchSize, m, k],
    batchMatrixA.flat(2),
    { dataType: 'float32' }
  );
  
  const tensorB = new Tensor<number>(
    [batchSize, k, n],
    batchMatrixB.flat(2),
    { dataType: 'float32' }
  );
  
  // Initialize WebGPU backend
  const backend = new WebGPUBackend();
  await backend.initialize();
  
  // Run batch matrix multiplication
  console.log('Running WebGPU batch matrix multiplication...');
  const startTime = performance.now();
  const result = await backend.batchMatmul(tensorA, tensorB);
  const endTime = performance.now();
  console.log(`Batch matrix multiplication time: ${(endTime - startTime).toFixed(2)}ms`);
  
  // Verify results
  const resultData = result.data;
  let correct = true;
  
  for (let b = 0; b < batchSize; b++) {
    const resultMatrix: number[][] = [];
    for (let i = 0; i < m; i++) {
      resultMatrix[i] = [];
      for (let j = 0; j < n; j++) {
        resultMatrix[i][j] = resultData[b * m * n + i * n + j];
      }
    }
    
    const matrixCorrect = areMatricesEqual(expectedResults[b], resultMatrix);
    if (!matrixCorrect) {
      console.error(`Batch ${b}: Matrices don't match`);
      correct = false;
      break;
    }
  }
  
  console.log(`Batch matrix multiplication: ${correct ? 'PASSED' : 'FAILED'}`);
  
  // Cleanup
  backend.dispose();
}

/**
 * Test convolution operation
 * @param batchSize Batch size
 * @param inputHeight Input height
 * @param inputWidth Input width
 * @param inputChannels Input channels
 * @param filterHeight Filter height
 * @param filterWidth Filter width
 * @param outputChannels Output channels
 * @param strides Convolution strides
 * @param padding Padding mode
 */
async function testConv2d(
  batchSize: number,
  inputHeight: number,
  inputWidth: number,
  inputChannels: number,
  filterHeight: number,
  filterWidth: number,
  outputChannels: number,
  strides: [number, number] = [1, 1],
  padding: 'same' | 'valid' = 'valid'
) {
  console.log(`\nTesting 2D convolution:`);
  console.log(`Input shape: [${batchSize}, ${inputHeight}, ${inputWidth}, ${inputChannels}]`);
  console.log(`Filter shape: [${filterHeight}, ${filterWidth}, ${inputChannels}, ${outputChannels}]`);
  console.log(`Strides: [${strides[0]}, ${strides[1]}], Padding: ${padding}`);
  
  // Generate random input tensor data
  const inputSize = batchSize * inputHeight * inputWidth * inputChannels;
  const inputData = new Array(inputSize).fill(0).map(() => Math.random() * 2 - 1);
  
  // Generate random filter tensor data
  const filterSize = filterHeight * filterWidth * inputChannels * outputChannels;
  const filterData = new Array(filterSize).fill(0).map(() => Math.random() * 2 - 1);
  
  // Create tensors
  const inputTensor = new Tensor<number>(
    [batchSize, inputHeight, inputWidth, inputChannels],
    inputData,
    { dataType: 'float32' }
  );
  
  const filterTensor = new Tensor<number>(
    [filterHeight, filterWidth, inputChannels, outputChannels],
    filterData,
    { dataType: 'float32' }
  );
  
  // Initialize WebGPU backend
  const backend = new WebGPUBackend();
  await backend.initialize();
  
  // Run convolution
  console.log('Running WebGPU convolution...');
  const startTime = performance.now();
  const resultTensor = await backend.conv2d(inputTensor, filterTensor, strides, padding);
  const endTime = performance.now();
  console.log(`Convolution time: ${(endTime - startTime).toFixed(2)}ms`);
  
  // Print output shape
  console.log(`Output shape: [${resultTensor.shape.join(', ')}]`);
  
  // Cleanup
  backend.dispose();
  
  return resultTensor;
}

/**
 * Main function to run all tests
 */
async function main() {
  try {
    // Small matrices for correctness verification
    await runPerformanceTest(32, 32, 32);
    
    // Medium matrices to see performance difference
    await runPerformanceTest(128, 128, 128);
    
    // Large matrices to demonstrate advanced optimizations
    await runPerformanceTest(512, 512, 512);
    
    // Test batch matrix multiplication
    await testBatchMatmul(10, 32, 32, 32);
    
    // Test convolution (common CNN layer sizes)
    await testConv2d(1, 28, 28, 32, 3, 3, 64, [1, 1], 'same');
    
  } catch (error) {
    console.error('Error running tests:', error);
  }
}

// Check if WebGPU is available in the environment
if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
  main();
} else {
  console.error('WebGPU is not supported in this environment');
}