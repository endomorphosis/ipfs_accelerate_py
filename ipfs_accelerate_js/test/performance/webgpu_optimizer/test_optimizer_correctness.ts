/**
 * WebGPU Optimizer Correctness Tests
 * 
 * This file contains tests that validate the correctness of the WebGPU optimizer.
 * These tests compare the results of optimized operations with standard implementations
 * to ensure that the optimizations do not affect numerical accuracy.
 */

import { expect } from '@jest/globals';

// Import WebGPU backend and optimizer
import { WebGPUBackend } from '../../../src/hardware/webgpu/backend';
import { WebGPUOptimizer } from '../../../src/hardware/webgpu/optimizations/webgpu_optimizer';
import { TensorShape, DataType } from '../../../src/core/tensor_types';
import { Tensor } from '../../../src/core/tensor';
import { BrowserDetector } from '../../../src/browser/browser_detector';
import { BrowserType } from '../../../src/browser/browser_types';

// Import neural network layers
import { LinearLayer } from '../../../src/model/transformers/layers/linear';
import { LayerNormalization } from '../../../src/model/transformers/layers/normalization';
import { MultiHeadAttention } from '../../../src/model/transformers/layers/attention';

/**
 * Helper function to create random tensors
 */
function createRandomTensor(shape: number[], dataType: DataType = DataType.FLOAT32, seed: number = 0): Tensor {
  const dataSize = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(dataSize);
  
  // Use a seeded random number generator for reproducibility
  const random = (n: number) => {
    const x = Math.sin(n + seed) * 10000;
    return x - Math.floor(x);
  };
  
  for (let i = 0; i < dataSize; i++) {
    data[i] = (random(i) * 2 - 1); // Values between -1 and 1
  }
  
  return new Tensor(data, { shape, dataType });
}

/**
 * Helper function to check if two tensors are approximately equal
 */
function tensorsAreClose(a: Tensor, b: Tensor, epsilon: number = 1e-5): boolean {
  // Check shape equality
  if (a.getShape().length !== b.getShape().length) {
    return false;
  }
  
  for (let i = 0; i < a.getShape().length; i++) {
    if (a.getShape()[i] !== b.getShape()[i]) {
      return false;
    }
  }
  
  // Check data equality within epsilon
  const aData = a.getData();
  const bData = b.getData();
  
  if (aData.length !== bData.length) {
    return false;
  }
  
  let maxDiff = 0;
  let maxRelDiff = 0;
  
  for (let i = 0; i < aData.length; i++) {
    const diff = Math.abs(aData[i] - bData[i]);
    maxDiff = Math.max(maxDiff, diff);
    
    // Calculate relative difference for non-zero values
    if (Math.abs(aData[i]) > epsilon || Math.abs(bData[i]) > epsilon) {
      const relDiff = diff / Math.max(Math.abs(aData[i]), Math.abs(bData[i]));
      maxRelDiff = Math.max(maxRelDiff, relDiff);
    }
  }
  
  return maxDiff <= epsilon || maxRelDiff <= epsilon;
}

/**
 * Print tensor comparison information for debugging
 */
function printTensorComparison(a: Tensor, b: Tensor, name: string): void {
  const aData = a.getData();
  const bData = b.getData();
  
  let maxDiff = 0;
  let maxRelDiff = 0;
  let maxDiffIndex = 0;
  
  for (let i = 0; i < aData.length; i++) {
    const diff = Math.abs(aData[i] - bData[i]);
    if (diff > maxDiff) {
      maxDiff = diff;
      maxDiffIndex = i;
    }
    
    // Calculate relative difference for non-zero values
    if (Math.abs(aData[i]) > 1e-10 || Math.abs(bData[i]) > 1e-10) {
      const relDiff = diff / Math.max(Math.abs(aData[i]), Math.abs(bData[i]));
      maxRelDiff = Math.max(maxRelDiff, relDiff);
    }
  }
  
  console.log(`\nTensor comparison for ${name}:`);
  console.log(`  Shapes: ${a.getShape()} vs ${b.getShape()}`);
  console.log(`  Max absolute difference: ${maxDiff}`);
  console.log(`  Max relative difference: ${maxRelDiff}`);
  console.log(`  At index ${maxDiffIndex}: ${aData[maxDiffIndex]} vs ${bData[maxDiffIndex]}`);
  
  const numSamplesToShow = Math.min(5, aData.length);
  console.log(`  First ${numSamplesToShow} values:`);
  for (let i = 0; i < numSamplesToShow; i++) {
    console.log(`    [${i}] ${aData[i]} vs ${bData[i]}, diff: ${Math.abs(aData[i] - bData[i])}`);
  }
}

/**
 * Test matrix multiplication correctness
 */
describe('WebGPU Optimizer Matrix Multiplication Correctness', () => {
  // Define tolerance for numerical comparison
  const epsilon = 1e-4;
  
  test('Matrix multiplication produces correct results with optimization', async () => {
    // Create backend instances
    const optimizedBackend = new WebGPUBackend();
    const standardBackend = new WebGPUBackend();
    
    // Enable optimizations on one backend, disable on the other
    optimizedBackend.getOptimizer().setEnableOptimizations(true);
    standardBackend.getOptimizer().setEnableOptimizations(false);
    
    // Create test tensors
    const shapes = [
      { a: [32, 64], b: [64, 32] },
      { a: [128, 256], b: [256, 128] },
      { a: [512, 128], b: [128, 256] } // Non-square matrices
    ];
    
    for (const { a: shapeA, b: shapeB } of shapes) {
      // Create input tensors with the same seed for reproducibility
      const tensorA = createRandomTensor(shapeA, DataType.FLOAT32, 42);
      const tensorB = createRandomTensor(shapeB, DataType.FLOAT32, 43);
      
      // Run with optimized implementation
      const optimizedResult = await optimizedBackend.matmul(tensorA, tensorB);
      
      // Run with standard implementation
      const standardResult = await standardBackend.matmul(tensorA, tensorB);
      
      // Check if results are close
      const resultsMatch = tensorsAreClose(optimizedResult, standardResult, epsilon);
      
      if (!resultsMatch) {
        printTensorComparison(
          optimizedResult, 
          standardResult, 
          `MatMul [${shapeA}] x [${shapeB}]`
        );
      }
      
      expect(resultsMatch).toBe(true);
    }
  });
  
  test('Batch matrix multiplication produces correct results with optimization', async () => {
    // Create backend instances
    const optimizedBackend = new WebGPUBackend();
    const standardBackend = new WebGPUBackend();
    
    // Enable optimizations on one backend, disable on the other
    optimizedBackend.getOptimizer().setEnableOptimizations(true);
    standardBackend.getOptimizer().setEnableOptimizations(false);
    
    // Create test tensors
    const shapes = [
      { a: [4, 32, 64], b: [4, 64, 32] },
      { a: [8, 64, 32], b: [8, 32, 64] }
    ];
    
    for (const { a: shapeA, b: shapeB } of shapes) {
      // Create input tensors
      const tensorA = createRandomTensor(shapeA, DataType.FLOAT32, 44);
      const tensorB = createRandomTensor(shapeB, DataType.FLOAT32, 45);
      
      // Run with optimized implementation
      const optimizedResult = await optimizedBackend.batchMatMul(tensorA, tensorB);
      
      // Run with standard implementation
      const standardResult = await standardBackend.batchMatMul(tensorA, tensorB);
      
      // Check if results are close
      const resultsMatch = tensorsAreClose(optimizedResult, standardResult, epsilon);
      
      if (!resultsMatch) {
        printTensorComparison(
          optimizedResult, 
          standardResult, 
          `BatchMatMul [${shapeA}] x [${shapeB}]`
        );
      }
      
      expect(resultsMatch).toBe(true);
    }
  });
});

/**
 * Test element-wise operations correctness
 */
describe('WebGPU Optimizer Element-wise Operations Correctness', () => {
  // Define tolerance for numerical comparison
  const epsilon = 1e-5;
  
  test('Element-wise operations produce correct results with optimization', async () => {
    // Create backend instances
    const optimizedBackend = new WebGPUBackend();
    const standardBackend = new WebGPUBackend();
    
    // Enable optimizations on one backend, disable on the other
    optimizedBackend.getOptimizer().setEnableOptimizations(true);
    standardBackend.getOptimizer().setEnableOptimizations(false);
    
    // Define operations to test
    const operations = [
      {
        name: 'ReLU',
        fn: (backend: WebGPUBackend, tensor: Tensor) => backend.relu(tensor)
      },
      {
        name: 'Sigmoid',
        fn: (backend: WebGPUBackend, tensor: Tensor) => backend.sigmoid(tensor)
      },
      {
        name: 'Tanh',
        fn: (backend: WebGPUBackend, tensor: Tensor) => backend.tanh(tensor)
      },
      {
        name: 'GELU',
        fn: (backend: WebGPUBackend, tensor: Tensor) => backend.gelu(tensor)
      }
    ];
    
    // Create test tensors
    const shapes = [
      [32, 64],
      [128, 256],
      [8, 16, 32]
    ];
    
    for (const shape of shapes) {
      for (const operation of operations) {
        // Create input tensor
        const inputTensor = createRandomTensor(shape, DataType.FLOAT32, 46);
        
        // Run with optimized implementation
        const optimizedResult = await operation.fn(optimizedBackend, inputTensor);
        
        // Run with standard implementation
        const standardResult = await operation.fn(standardBackend, inputTensor);
        
        // Check if results are close
        const resultsMatch = tensorsAreClose(optimizedResult, standardResult, epsilon);
        
        if (!resultsMatch) {
          printTensorComparison(
            optimizedResult, 
            standardResult, 
            `${operation.name} [${shape}]`
          );
        }
        
        expect(resultsMatch).toBe(true);
      }
    }
  });
  
  test('Binary element-wise operations produce correct results with optimization', async () => {
    // Create backend instances
    const optimizedBackend = new WebGPUBackend();
    const standardBackend = new WebGPUBackend();
    
    // Enable optimizations on one backend, disable on the other
    optimizedBackend.getOptimizer().setEnableOptimizations(true);
    standardBackend.getOptimizer().setEnableOptimizations(false);
    
    // Define operations to test
    const operations = [
      {
        name: 'Add',
        fn: (backend: WebGPUBackend, a: Tensor, b: Tensor) => backend.add(a, b)
      },
      {
        name: 'Subtract',
        fn: (backend: WebGPUBackend, a: Tensor, b: Tensor) => backend.subtract(a, b)
      },
      {
        name: 'Multiply',
        fn: (backend: WebGPUBackend, a: Tensor, b: Tensor) => backend.multiply(a, b)
      },
      {
        name: 'Divide',
        fn: (backend: WebGPUBackend, a: Tensor, b: Tensor) => backend.divide(a, b)
      }
    ];
    
    // Create test tensors
    const shapes = [
      [32, 64],
      [128, 256],
      [8, 16, 32]
    ];
    
    for (const shape of shapes) {
      for (const operation of operations) {
        // Create input tensors
        const tensorA = createRandomTensor(shape, DataType.FLOAT32, 47);
        const tensorB = createRandomTensor(shape, DataType.FLOAT32, 48);
        
        // For division, ensure no zeros in the divisor
        if (operation.name === 'Divide') {
          const bData = tensorB.getData();
          for (let i = 0; i < bData.length; i++) {
            if (Math.abs(bData[i]) < 0.01) {
              bData[i] = 0.01 * (bData[i] < 0 ? -1 : 1);
            }
          }
          tensorB.setData(bData);
        }
        
        // Run with optimized implementation
        const optimizedResult = await operation.fn(optimizedBackend, tensorA, tensorB);
        
        // Run with standard implementation
        const standardResult = await operation.fn(standardBackend, tensorA, tensorB);
        
        // Check if results are close
        const resultsMatch = tensorsAreClose(optimizedResult, standardResult, epsilon);
        
        if (!resultsMatch) {
          printTensorComparison(
            optimizedResult, 
            standardResult, 
            `${operation.name} [${shape}]`
          );
        }
        
        expect(resultsMatch).toBe(true);
      }
    }
  });
});

/**
 * Test operation fusion correctness
 */
describe('WebGPU Optimizer Operation Fusion Correctness', () => {
  // Define tolerance for numerical comparison
  const epsilon = 1e-4;
  
  test('Linear + ReLU fusion produces correct results', async () => {
    // Create backend instances
    const optimizedBackend = new WebGPUBackend();
    const standardBackend = new WebGPUBackend();
    
    // Enable optimizations on one backend, disable on the other
    optimizedBackend.getOptimizer().setEnableOptimizations(true);
    optimizedBackend.getOptimizer().setEnableOperationFusion(true);
    standardBackend.getOptimizer().setEnableOptimizations(false);
    
    // Create test tensors
    const testCases = [
      { batchSize: 32, inputSize: 64, outputSize: 128 },
      { batchSize: 16, inputSize: 128, outputSize: 256 }
    ];
    
    for (const { batchSize, inputSize, outputSize } of testCases) {
      // Create input tensor
      const input = createRandomTensor([batchSize, inputSize], DataType.FLOAT32, 49);
      
      // Create weights and bias
      const weights = createRandomTensor([inputSize, outputSize], DataType.FLOAT32, 50);
      const bias = createRandomTensor([outputSize], DataType.FLOAT32, 51);
      
      // Create linear layers
      const linearLayerOptimized = new LinearLayer(inputSize, outputSize);
      linearLayerOptimized.setWeights(weights);
      linearLayerOptimized.setBias(bias);
      
      const linearLayerStandard = new LinearLayer(inputSize, outputSize);
      linearLayerStandard.setWeights(weights);
      linearLayerStandard.setBias(bias);
      
      // Run with optimized implementation (fusion should be applied)
      const linearOutputOptimized = await linearLayerOptimized.forward(input, optimizedBackend);
      const reluOutputOptimized = await optimizedBackend.relu(linearOutputOptimized);
      
      // Run with standard implementation (no fusion)
      const linearOutputStandard = await linearLayerStandard.forward(input, standardBackend);
      const reluOutputStandard = await standardBackend.relu(linearOutputStandard);
      
      // Check if results are close
      const resultsMatch = tensorsAreClose(reluOutputOptimized, reluOutputStandard, epsilon);
      
      if (!resultsMatch) {
        printTensorComparison(
          reluOutputOptimized, 
          reluOutputStandard, 
          `Linear+ReLU [${batchSize}, ${inputSize}] → [${batchSize}, ${outputSize}]`
        );
      }
      
      expect(resultsMatch).toBe(true);
    }
  });
  
  test('LayerNorm + GELU fusion produces correct results', async () => {
    // Create backend instances
    const optimizedBackend = new WebGPUBackend();
    const standardBackend = new WebGPUBackend();
    
    // Enable optimizations on one backend, disable on the other
    optimizedBackend.getOptimizer().setEnableOptimizations(true);
    optimizedBackend.getOptimizer().setEnableOperationFusion(true);
    standardBackend.getOptimizer().setEnableOptimizations(false);
    
    // Create test tensors
    const testCases = [
      { batchSize: 8, seqLength: 32, hiddenSize: 64 },
      { batchSize: 4, seqLength: 64, hiddenSize: 128 }
    ];
    
    for (const { batchSize, seqLength, hiddenSize } of testCases) {
      // Create input tensor
      const input = createRandomTensor([batchSize, seqLength, hiddenSize], DataType.FLOAT32, 52);
      
      // Create layer normalization parameters
      const gamma = createRandomTensor([hiddenSize], DataType.FLOAT32, 53);
      const beta = createRandomTensor([hiddenSize], DataType.FLOAT32, 54);
      
      // Create layer normalization
      const layerNormOptimized = new LayerNormalization(hiddenSize);
      layerNormOptimized.setGamma(gamma);
      layerNormOptimized.setBeta(beta);
      
      const layerNormStandard = new LayerNormalization(hiddenSize);
      layerNormStandard.setGamma(gamma);
      layerNormStandard.setBeta(beta);
      
      // Run with optimized implementation (fusion should be applied)
      const normOutputOptimized = await layerNormOptimized.forward(input, optimizedBackend);
      const geluOutputOptimized = await optimizedBackend.gelu(normOutputOptimized);
      
      // Run with standard implementation (no fusion)
      const normOutputStandard = await layerNormStandard.forward(input, standardBackend);
      const geluOutputStandard = await standardBackend.gelu(normOutputStandard);
      
      // Check if results are close
      const resultsMatch = tensorsAreClose(geluOutputOptimized, geluOutputStandard, epsilon);
      
      if (!resultsMatch) {
        printTensorComparison(
          geluOutputOptimized, 
          geluOutputStandard, 
          `LayerNorm+GELU [${batchSize}, ${seqLength}, ${hiddenSize}]`
        );
      }
      
      expect(resultsMatch).toBe(true);
    }
  });
  
  test('Multiple element-wise operations fusion produces correct results', async () => {
    // Create backend instances
    const optimizedBackend = new WebGPUBackend();
    const standardBackend = new WebGPUBackend();
    
    // Enable optimizations on one backend, disable on the other
    optimizedBackend.getOptimizer().setEnableOptimizations(true);
    optimizedBackend.getOptimizer().setEnableOperationFusion(true);
    standardBackend.getOptimizer().setEnableOptimizations(false);
    
    // Create test tensors
    const shapes = [
      [32, 64],
      [8, 16, 32]
    ];
    
    for (const shape of shapes) {
      // Create input tensors
      const tensorA = createRandomTensor(shape, DataType.FLOAT32, 55);
      const tensorB = createRandomTensor(shape, DataType.FLOAT32, 56);
      const tensorC = createRandomTensor(shape, DataType.FLOAT32, 57);
      
      // Run with optimized implementation (fusion should be applied)
      const mulOutputOptimized = await optimizedBackend.multiply(tensorA, tensorB);
      const addOutputOptimized = await optimizedBackend.add(mulOutputOptimized, tensorC);
      const tanhOutputOptimized = await optimizedBackend.tanh(addOutputOptimized);
      
      // Run with standard implementation (no fusion)
      const mulOutputStandard = await standardBackend.multiply(tensorA, tensorB);
      const addOutputStandard = await standardBackend.add(mulOutputStandard, tensorC);
      const tanhOutputStandard = await standardBackend.tanh(addOutputStandard);
      
      // Check if results are close
      const resultsMatch = tensorsAreClose(tanhOutputOptimized, tanhOutputStandard, epsilon);
      
      if (!resultsMatch) {
        printTensorComparison(
          tanhOutputOptimized, 
          tanhOutputStandard, 
          `Mul+Add+Tanh [${shape}]`
        );
      }
      
      expect(resultsMatch).toBe(true);
    }
  });
});

/**
 * Test neural network pattern recognition correctness
 */
describe('WebGPU Neural Network Pattern Recognition Correctness', () => {
  // Define tolerance for numerical comparison
  const epsilon = 1e-4;
  
  test('Multi-head attention with pattern recognition produces correct results', async () => {
    // Create backend instances
    const optimizedBackend = new WebGPUBackend();
    const standardBackend = new WebGPUBackend();
    
    // Enable optimizations on one backend, disable on the other
    optimizedBackend.getOptimizer().setEnableOptimizations(true);
    optimizedBackend.getOptimizer().setEnableOperationFusion(true);
    optimizedBackend.getOptimizer().setEnableNeuralNetworkPatternRecognition(true);
    standardBackend.getOptimizer().setEnableOptimizations(false);
    
    // Create test tensors
    const testCases = [
      { batchSize: 4, seqLength: 16, embedSize: 64, numHeads: 8 },
      { batchSize: 2, seqLength: 32, embedSize: 128, numHeads: 8 }
    ];
    
    for (const { batchSize, seqLength, embedSize, numHeads } of testCases) {
      // Create input tensor
      const input = createRandomTensor([batchSize, seqLength, embedSize], DataType.FLOAT32, 58);
      
      // Create attention layer
      const attentionOptimized = new MultiHeadAttention(embedSize, numHeads);
      const attentionStandard = new MultiHeadAttention(embedSize, numHeads);
      
      // Create and set weights
      const qWeight = createRandomTensor([embedSize, embedSize], DataType.FLOAT32, 59);
      const kWeight = createRandomTensor([embedSize, embedSize], DataType.FLOAT32, 60);
      const vWeight = createRandomTensor([embedSize, embedSize], DataType.FLOAT32, 61);
      const outWeight = createRandomTensor([embedSize, embedSize], DataType.FLOAT32, 62);
      
      attentionOptimized.setQKVWeights(qWeight, kWeight, vWeight);
      attentionOptimized.setOutputWeight(outWeight);
      
      attentionStandard.setQKVWeights(qWeight, kWeight, vWeight);
      attentionStandard.setOutputWeight(outWeight);
      
      // Run with optimized implementation (pattern recognition should be applied)
      const outputOptimized = await attentionOptimized.forward(input, input, input, null, optimizedBackend);
      
      // Run with standard implementation (no pattern recognition)
      const outputStandard = await attentionStandard.forward(input, input, input, null, standardBackend);
      
      // Check if results are close
      const resultsMatch = tensorsAreClose(outputOptimized, outputStandard, epsilon);
      
      if (!resultsMatch) {
        printTensorComparison(
          outputOptimized, 
          outputStandard, 
          `Multi-Head Attention [${batchSize}, ${seqLength}, ${embedSize}], Heads: ${numHeads}`
        );
      }
      
      expect(resultsMatch).toBe(true);
    }
  });
  
  test('Feed-forward network with pattern recognition produces correct results', async () => {
    // Create backend instances
    const optimizedBackend = new WebGPUBackend();
    const standardBackend = new WebGPUBackend();
    
    // Enable optimizations on one backend, disable on the other
    optimizedBackend.getOptimizer().setEnableOptimizations(true);
    optimizedBackend.getOptimizer().setEnableOperationFusion(true);
    optimizedBackend.getOptimizer().setEnableNeuralNetworkPatternRecognition(true);
    standardBackend.getOptimizer().setEnableOptimizations(false);
    
    // Create test tensors
    const testCases = [
      { batchSize: 8, seqLength: 16, hiddenSize: 64, ffnSize: 256 },
      { batchSize: 4, seqLength: 32, hiddenSize: 128, ffnSize: 512 }
    ];
    
    for (const { batchSize, seqLength, hiddenSize, ffnSize } of testCases) {
      // Create input tensor
      const input = createRandomTensor([batchSize, seqLength, hiddenSize], DataType.FLOAT32, 63);
      
      // Create FFN weights
      const fc1Weight = createRandomTensor([hiddenSize, ffnSize], DataType.FLOAT32, 64);
      const fc1Bias = createRandomTensor([ffnSize], DataType.FLOAT32, 65);
      const fc2Weight = createRandomTensor([ffnSize, hiddenSize], DataType.FLOAT32, 66);
      const fc2Bias = createRandomTensor([hiddenSize], DataType.FLOAT32, 67);
      
      // Create linear layers for optimized path
      const linear1Optimized = new LinearLayer(hiddenSize, ffnSize);
      linear1Optimized.setWeights(fc1Weight);
      linear1Optimized.setBias(fc1Bias);
      
      const linear2Optimized = new LinearLayer(ffnSize, hiddenSize);
      linear2Optimized.setWeights(fc2Weight);
      linear2Optimized.setBias(fc2Bias);
      
      // Create linear layers for standard path
      const linear1Standard = new LinearLayer(hiddenSize, ffnSize);
      linear1Standard.setWeights(fc1Weight);
      linear1Standard.setBias(fc1Bias);
      
      const linear2Standard = new LinearLayer(ffnSize, hiddenSize);
      linear2Standard.setWeights(fc2Weight);
      linear2Standard.setBias(fc2Bias);
      
      // Run with optimized implementation (pattern recognition should be applied)
      const hiddenOptimized = await linear1Optimized.forward(input, optimizedBackend);
      const activatedOptimized = await optimizedBackend.gelu(hiddenOptimized);
      const outputOptimized = await linear2Optimized.forward(activatedOptimized, optimizedBackend);
      
      // Run with standard implementation (no pattern recognition)
      const hiddenStandard = await linear1Standard.forward(input, standardBackend);
      const activatedStandard = await standardBackend.gelu(hiddenStandard);
      const outputStandard = await linear2Standard.forward(activatedStandard, standardBackend);
      
      // Check if results are close
      const resultsMatch = tensorsAreClose(outputOptimized, outputStandard, epsilon);
      
      if (!resultsMatch) {
        printTensorComparison(
          outputOptimized, 
          outputStandard, 
          `FFN [${batchSize}, ${seqLength}, ${hiddenSize}] → [${ffnSize}] → [${hiddenSize}]`
        );
      }
      
      expect(resultsMatch).toBe(true);
    }
  });
});

/**
 * Test memory layout optimization correctness
 */
describe('WebGPU Memory Layout Optimization Correctness', () => {
  // Define tolerance for numerical comparison
  const epsilon = 1e-4;
  
  test('Memory layout optimization produces correct results for matrix operations', async () => {
    // Create backend instances
    const optimizedBackend = new WebGPUBackend();
    const standardBackend = new WebGPUBackend();
    
    // Enable memory layout optimizations on one backend, disable on the other
    optimizedBackend.getOptimizer().setEnableOptimizations(true);
    optimizedBackend.getOptimizer().setEnableMemoryLayoutOptimizations(true);
    standardBackend.getOptimizer().setEnableOptimizations(false);
    
    // Define operations to test
    const operations = [
      {
        name: 'MatMul',
        fn: (backend: WebGPUBackend, a: Tensor, b: Tensor) => backend.matmul(a, b)
      },
      {
        name: 'Transpose',
        fn: (backend: WebGPUBackend, a: Tensor) => backend.transpose(a, [1, 0])
      }
    ];
    
    // Create test tensor shapes
    const shapes = [
      { a: [64, 128], b: [128, 64] },
      { a: [256, 128], b: [128, 256] },
      { a: [512, 64], b: [64, 128] } // Non-square
    ];
    
    for (const { a: shapeA, b: shapeB } of shapes) {
      for (const operation of operations) {
        // Skip transpose for certain shapes
        if (operation.name === 'Transpose' && shapeB) {
          continue;
        }
        
        // Create input tensors
        const tensorA = createRandomTensor(shapeA, DataType.FLOAT32, 68);
        const tensorB = shapeB ? createRandomTensor(shapeB, DataType.FLOAT32, 69) : null;
        
        // Run with optimized implementation
        let optimizedResult;
        if (tensorB) {
          optimizedResult = await operation.fn(optimizedBackend, tensorA, tensorB);
        } else {
          optimizedResult = await operation.fn(optimizedBackend, tensorA);
        }
        
        // Run with standard implementation
        let standardResult;
        if (tensorB) {
          standardResult = await operation.fn(standardBackend, tensorA, tensorB);
        } else {
          standardResult = await operation.fn(standardBackend, tensorA);
        }
        
        // Check if results are close
        const resultsMatch = tensorsAreClose(optimizedResult, standardResult, epsilon);
        
        if (!resultsMatch) {
          printTensorComparison(
            optimizedResult, 
            standardResult, 
            `${operation.name} [${shapeA}]${tensorB ? ` × [${shapeB}]` : ''}`
          );
        }
        
        expect(resultsMatch).toBe(true);
      }
    }
  });
  
  test('Memory layout optimization produces correct results for convolutions', async () => {
    // Create backend instances
    const optimizedBackend = new WebGPUBackend();
    const standardBackend = new WebGPUBackend();
    
    // Enable memory layout optimizations on one backend, disable on the other
    optimizedBackend.getOptimizer().setEnableOptimizations(true);
    optimizedBackend.getOptimizer().setEnableMemoryLayoutOptimizations(true);
    standardBackend.getOptimizer().setEnableOptimizations(false);
    
    // Create test tensor shapes [batch, channels, height, width]
    const shapes = [
      { input: [4, 3, 32, 32], filter: [16, 3, 3, 3] },
      { input: [2, 3, 64, 64], filter: [8, 3, 5, 5] }
    ];
    
    for (const { input: inputShape, filter: filterShape } of shapes) {
      // Create input tensors
      const inputTensor = createRandomTensor(inputShape, DataType.FLOAT32, 70);
      const filterTensor = createRandomTensor(filterShape, DataType.FLOAT32, 71);
      
      // Run with optimized implementation
      const optimizedResult = await optimizedBackend.conv2d(
        inputTensor, 
        filterTensor, 
        1, // stride
        1, // dilation
        'same' // padding
      );
      
      // Run with standard implementation
      const standardResult = await standardBackend.conv2d(
        inputTensor, 
        filterTensor, 
        1, // stride
        1, // dilation
        'same' // padding
      );
      
      // Check if results are close
      const resultsMatch = tensorsAreClose(optimizedResult, standardResult, epsilon);
      
      if (!resultsMatch) {
        printTensorComparison(
          optimizedResult, 
          standardResult, 
          `Conv2D Input:[${inputShape}], Filter:[${filterShape}]`
        );
      }
      
      expect(resultsMatch).toBe(true);
    }
  });
});

/**
 * Test browser-specific optimizations correctness
 */
describe('WebGPU Browser-Specific Optimizations Correctness', () => {
  // Define tolerance for numerical comparison
  const epsilon = 1e-4;
  
  test('Browser-specific optimizations produce correct results for matrix operations', async () => {
    // Create backend instances
    const optimizedBackend = new WebGPUBackend();
    const standardBackend = new WebGPUBackend();
    
    // Enable browser-specific optimizations on one backend, disable on the other
    optimizedBackend.getOptimizer().setEnableOptimizations(true);
    optimizedBackend.getOptimizer().setEnableBrowserSpecificOptimizations(true);
    standardBackend.getOptimizer().setEnableOptimizations(false);
    
    // Create test tensor shapes
    const shapes = [
      { a: [128, 128], b: [128, 128] },
      { a: [512, 512], b: [512, 128] }
    ];
    
    for (const { a: shapeA, b: shapeB } of shapes) {
      // Create input tensors
      const tensorA = createRandomTensor(shapeA, DataType.FLOAT32, 72);
      const tensorB = createRandomTensor(shapeB, DataType.FLOAT32, 73);
      
      // Run with optimized implementation
      const optimizedResult = await optimizedBackend.matmul(tensorA, tensorB);
      
      // Run with standard implementation
      const standardResult = await standardBackend.matmul(tensorA, tensorB);
      
      // Check if results are close
      const resultsMatch = tensorsAreClose(optimizedResult, standardResult, epsilon);
      
      if (!resultsMatch) {
        printTensorComparison(
          optimizedResult, 
          standardResult, 
          `Browser-Optimized MatMul [${shapeA}] × [${shapeB}]`
        );
      }
      
      expect(resultsMatch).toBe(true);
    }
  });
  
  test('Browser-specific optimizations produce correct results for reductions', async () => {
    // Create backend instances
    const optimizedBackend = new WebGPUBackend();
    const standardBackend = new WebGPUBackend();
    
    // Enable browser-specific optimizations on one backend, disable on the other
    optimizedBackend.getOptimizer().setEnableOptimizations(true);
    optimizedBackend.getOptimizer().setEnableBrowserSpecificOptimizations(true);
    standardBackend.getOptimizer().setEnableOptimizations(false);
    
    // Define operations to test
    const operations = [
      {
        name: 'Mean',
        fn: (backend: WebGPUBackend, a: Tensor, axis: number) => backend.mean(a, axis)
      },
      {
        name: 'Sum',
        fn: (backend: WebGPUBackend, a: Tensor, axis: number) => backend.sum(a, axis)
      },
      {
        name: 'Max',
        fn: (backend: WebGPUBackend, a: Tensor, axis: number) => backend.max(a, axis)
      }
    ];
    
    // Create test tensor shapes
    const shapes = [
      [64, 128],
      [32, 64, 16]
    ];
    
    for (const shape of shapes) {
      for (const operation of operations) {
        // Create input tensor
        const tensor = createRandomTensor(shape, DataType.FLOAT32, 74);
        
        // Test reduction along different axes
        for (let axis = 0; axis < shape.length; axis++) {
          // Run with optimized implementation
          const optimizedResult = await operation.fn(optimizedBackend, tensor, axis);
          
          // Run with standard implementation
          const standardResult = await operation.fn(standardBackend, tensor, axis);
          
          // Check if results are close
          const resultsMatch = tensorsAreClose(optimizedResult, standardResult, epsilon);
          
          if (!resultsMatch) {
            printTensorComparison(
              optimizedResult, 
              standardResult, 
              `Browser-Optimized ${operation.name} [${shape}] along axis ${axis}`
            );
          }
          
          expect(resultsMatch).toBe(true);
        }
      }
    }
  });
});

/**
 * Log test completion
 */
afterAll(() => {
  console.log('WebGPU Optimizer Correctness Tests completed');
});