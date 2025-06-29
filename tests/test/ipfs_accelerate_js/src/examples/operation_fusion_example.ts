/**
 * Operation Fusion Example
 * Demonstrates how to use WebGPU operation fusion for improved performance
 */

import { createWebGPUBackend } from '../hardware/webgpu/backend';
import { Tensor } from '../tensor/tensor';
import { FusionOpType } from '../hardware/webgpu/optimizations/operation_fusion';

/**
 * Example of fusing operations for better performance
 */
async function operationFusionExample() {
  console.log("=== WebGPU Operation Fusion Example ===");
  
  // Check if WebGPU is available
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    console.error("WebGPU is not available in this environment");
    return;
  }
  
  try {
    // Initialize the WebGPU backend
    console.log("Initializing WebGPU backend...");
    const backend = await createWebGPUBackend();
    
    // Create sample matrices for matrix multiplication + activation
    // Matrix A: 128x256
    const matrixA = new Tensor<number>(
      [128, 256],
      new Float32Array(128 * 256).fill(0.1),
      { dataType: 'float32', backend: 'webgpu', device: backend.id }
    );
    
    // Matrix B: 256x128
    const matrixB = new Tensor<number>(
      [256, 128],
      new Float32Array(256 * 128).fill(0.2),
      { dataType: 'float32', backend: 'webgpu', device: backend.id }
    );
    
    // Run individual operations (non-fused)
    console.log("Running operations separately...");
    console.time("Non-fused operations");
    
    // Matrix multiplication
    const resultMatmul = await backend.matmul(matrixA, matrixB);
    
    // ReLU activation
    const resultActivation = await backend.relu(resultMatmul);
    
    console.timeEnd("Non-fused operations");
    
    // Run fused operations
    console.log("Running fused operations...");
    console.time("Fused operations");
    
    // Define operations to fuse
    const operations: FusionOpType[] = ['matmul', 'relu'];
    
    // Execute fused operations
    const resultFused = await backend.executeFusedOperations(
      [matrixA, matrixB],
      operations
    );
    
    console.timeEnd("Fused operations");
    
    // Verify results are similar
    const maxDiff = await verifyResults(resultActivation, resultFused);
    console.log(`Maximum difference between results: ${maxDiff}`);
    
    // Memory cleanup
    backend.dispose();
    console.log("Example completed successfully");
    
  } catch (error) {
    console.error("Error running example:", error);
  }
}

/**
 * Element-wise chain fusion example
 */
async function elementWiseChainExample() {
  console.log("\n=== Element-wise Chain Fusion Example ===");
  
  try {
    // Initialize the WebGPU backend
    const backend = await createWebGPUBackend();
    
    // Create sample tensors
    const shape = [1024, 1024];
    const size = shape[0] * shape[1];
    
    const tensorA = new Tensor<number>(
      shape,
      new Float32Array(size).fill(0.5),
      { dataType: 'float32', backend: 'webgpu', device: backend.id }
    );
    
    const tensorB = new Tensor<number>(
      shape,
      new Float32Array(size).fill(0.2),
      { dataType: 'float32', backend: 'webgpu', device: backend.id }
    );
    
    const tensorC = new Tensor<number>(
      shape,
      new Float32Array(size).fill(0.3),
      { dataType: 'float32', backend: 'webgpu', device: backend.id }
    );
    
    // Run individual operations (non-fused)
    console.log("Running operations separately...");
    console.time("Non-fused element-wise operations");
    
    // A + B
    const resultAdd = await backend.add(tensorA, tensorB);
    
    // (A + B) * C
    const resultMul = await backend.multiply(resultAdd, tensorC);
    
    console.timeEnd("Non-fused element-wise operations");
    
    // Run fused operations
    console.log("Running fused operations...");
    console.time("Fused element-wise operations");
    
    // Define operations to fuse
    const operations: FusionOpType[] = ['add', 'multiply'];
    
    // Execute fused operations
    const resultFused = await backend.executeFusedOperations(
      [tensorA, tensorB, tensorC],
      operations
    );
    
    console.timeEnd("Fused element-wise operations");
    
    // Verify results are similar
    const maxDiff = await verifyResults(resultMul, resultFused);
    console.log(`Maximum difference between results: ${maxDiff}`);
    
    // Memory cleanup
    backend.dispose();
    console.log("Example completed successfully");
    
  } catch (error) {
    console.error("Error running example:", error);
  }
}

/**
 * Binary + Unary fusion example (element-wise operation + activation)
 */
async function binaryUnaryExample() {
  console.log("\n=== Binary + Unary Fusion Example ===");
  
  try {
    // Initialize the WebGPU backend
    const backend = await createWebGPUBackend();
    
    // Create sample tensors
    const shape = [2048, 2048];
    const size = shape[0] * shape[1];
    
    const tensorA = new Tensor<number>(
      shape,
      new Float32Array(size).fill(0.1),
      { dataType: 'float32', backend: 'webgpu', device: backend.id }
    );
    
    const tensorB = new Tensor<number>(
      shape,
      new Float32Array(size).fill(-0.2),
      { dataType: 'float32', backend: 'webgpu', device: backend.id }
    );
    
    // Run individual operations (non-fused)
    console.log("Running operations separately...");
    console.time("Non-fused binary+unary operations");
    
    // A + B
    const resultAdd = await backend.add(tensorA, tensorB);
    
    // ReLU(A + B)
    const resultRelu = await backend.relu(resultAdd);
    
    console.timeEnd("Non-fused binary+unary operations");
    
    // Run fused operations
    console.log("Running fused operations...");
    console.time("Fused binary+unary operations");
    
    // Define operations to fuse
    const operations: FusionOpType[] = ['add', 'relu'];
    
    // Execute fused operations
    const resultFused = await backend.executeFusedOperations(
      [tensorA, tensorB],
      operations
    );
    
    console.timeEnd("Fused binary+unary operations");
    
    // Verify results are similar
    const maxDiff = await verifyResults(resultRelu, resultFused);
    console.log(`Maximum difference between results: ${maxDiff}`);
    
    // Memory cleanup
    backend.dispose();
    console.log("Example completed successfully");
    
  } catch (error) {
    console.error("Error running example:", error);
  }
}

/**
 * Custom fusion pattern
 */
async function customFusionPatternExample() {
  console.log("\n=== Custom Fusion Pattern Example ===");
  
  try {
    // Initialize the WebGPU backend
    const backend = await createWebGPUBackend();
    
    // Create a custom fusion pattern
    const customPattern: FusionOpType[] = ['multiply', 'add', 'sigmoid'];
    
    // Check if pattern can be fused
    const canFuse = backend.createFusionPattern(customPattern);
    console.log(`Can fuse custom pattern: ${canFuse}`);
    
    // If pattern can't be fused, end the example
    if (!canFuse) {
      console.log("Custom pattern cannot be fused. Example completed.");
      backend.dispose();
      return;
    }
    
    // Create sample tensors
    const shape = [1024, 1024];
    const size = shape[0] * shape[1];
    
    const tensorA = new Tensor<number>(
      shape,
      new Float32Array(size).fill(0.1),
      { dataType: 'float32', backend: 'webgpu', device: backend.id }
    );
    
    const tensorB = new Tensor<number>(
      shape,
      new Float32Array(size).fill(0.2),
      { dataType: 'float32', backend: 'webgpu', device: backend.id }
    );
    
    const tensorC = new Tensor<number>(
      shape,
      new Float32Array(size).fill(0.3),
      { dataType: 'float32', backend: 'webgpu', device: backend.id }
    );
    
    // Run individual operations (non-fused)
    console.log("Running operations separately...");
    console.time("Non-fused custom pattern");
    
    // A * B
    const resultMul = await backend.multiply(tensorA, tensorB);
    
    // (A * B) + C
    const resultAdd = await backend.add(resultMul, tensorC);
    
    // sigmoid((A * B) + C)
    const resultSigmoid = await backend.sigmoid(resultAdd);
    
    console.timeEnd("Non-fused custom pattern");
    
    // Run fused operations
    console.log("Running fused operations...");
    console.time("Fused custom pattern");
    
    // Execute fused operations
    const resultFused = await backend.executeFusedOperations(
      [tensorA, tensorB, tensorC],
      customPattern
    );
    
    console.timeEnd("Fused custom pattern");
    
    // Verify results are similar
    const maxDiff = await verifyResults(resultSigmoid, resultFused);
    console.log(`Maximum difference between results: ${maxDiff}`);
    
    // Memory cleanup
    backend.dispose();
    console.log("Example completed successfully");
    
  } catch (error) {
    console.error("Error running example:", error);
  }
}

/**
 * Verify that two tensors have similar values
 * @param tensor1 First tensor
 * @param tensor2 Second tensor
 * @returns Maximum absolute difference between tensor values
 */
async function verifyResults<T>(tensor1: Tensor<T>, tensor2: Tensor<T>): Promise<number> {
  // Ensure tensors have same shape
  if (tensor1.shape.join(',') !== tensor2.shape.join(',')) {
    throw new Error(`Tensors have different shapes: ${tensor1.shape} vs ${tensor2.shape}`);
  }
  
  // Get tensor data
  const data1 = tensor1.data as Float32Array;
  const data2 = tensor2.data as Float32Array;
  
  // Find maximum difference
  let maxDiff = 0;
  for (let i = 0; i < data1.length; i++) {
    const diff = Math.abs(data1[i] - data2[i]);
    maxDiff = Math.max(maxDiff, diff);
  }
  
  return maxDiff;
}

// Run the examples
async function runExamples() {
  await operationFusionExample();
  await elementWiseChainExample();
  await binaryUnaryExample();
  await customFusionPatternExample();
}

// Check if this is being run directly or imported
if (typeof require !== 'undefined' && require.main === module) {
  runExamples().catch(console.error);
}

// Export for importing
export {
  operationFusionExample,
  elementWiseChainExample,
  binaryUnaryExample,
  customFusionPatternExample
};