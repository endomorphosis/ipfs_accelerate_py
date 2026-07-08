/**
 * Browser-Optimized WebGPU Tensor Sharing Examples
 * 
 * This file demonstrates how to use the WebGPU Tensor Sharing system
 * with browser-specific optimizations across different use cases.
 */

import { TensorSharingIntegration } from './ipfs_accelerate_js_tensor_sharing_integration';
import { WebGPUTensorSharing } from './ipfs_accelerate_js_webgpu_tensor_sharing';
import { StorageManager } from './ipfs_accelerate_js_storage_manager';
import { BrowserType, detectBrowserType } from './ipfs_accelerate_js_browser_optimized_shaders';

/**
 * Example 1: Basic setup with browser optimizations
 */
async function basicExample() {
  console.log('Running Basic Example with Browser Optimizations');
  
  // Create tensor sharing integration
  const tensorSharing = new TensorSharingIntegration();
  await tensorSharing.initialize();
  
  // Auto-detect browser type
  const browserType = detectBrowserType();
  console.log(`Detected browser: ${browserType}`);
  
  // Create WebGPU tensor sharing with browser optimizations
  const webgpuTensorSharing = new WebGPUTensorSharing(tensorSharing, {
    browserOptimizations: true,
    debug: true // Show optimization details
  });
  
  // Initialize
  const initialized = await webgpuTensorSharing.initialize();
  if (!initialized) {
    console.error('WebGPU initialization failed. Your browser may not support WebGPU.');
    return;
  }
  
  // Create example tensors
  const matrixA = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const matrixB = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
  
  // Register tensors with the tensor sharing system
  await tensorSharing.registerSharedTensor(
    'matrixA',
    [2, 4], // 2x4 matrix
    matrixA,
    'float32',
    'exampleModel',
    ['exampleModel']
  );
  
  await tensorSharing.registerSharedTensor(
    'matrixB',
    [4, 2], // 4x2 matrix
    matrixB,
    'float32',
    'exampleModel',
    ['exampleModel']
  );
  
  // Perform matrix multiplication with browser-optimized implementation
  console.log('Performing browser-optimized matrix multiplication');
  const resultTensorName = await webgpuTensorSharing.matmul(
    'matrixA', 'exampleModel',
    'matrixB', 'exampleModel',
    'resultMatrix', 'exampleModel'
  );
  
  // Get the result tensor
  const resultTensor = await tensorSharing.getSharedTensor(resultTensorName, 'exampleModel');
  console.log('Matrix multiplication result:', Array.from(resultTensor.getData()));
  
  // Perform element-wise operation with browser optimization
  console.log('Performing browser-optimized ReLU operation');
  const reluResultName = await webgpuTensorSharing.elementwise(
    resultTensorName, 'exampleModel',
    'reluResult', 'exampleModel',
    'relu'
  );
  
  // Get the result tensor
  const reluTensor = await tensorSharing.getSharedTensor(reluResultName, 'exampleModel');
  console.log('ReLU result:', Array.from(reluTensor.getData()));
  
  // Clean up
  webgpuTensorSharing.dispose();
  console.log('Basic example completed');
}

/**
 * Example 2: Matrix multiplication benchmark with browser optimizations
 */
async function matrixMultiplicationBenchmark() {
  console.log('Running Matrix Multiplication Benchmark with Browser Optimizations');
  
  // Create tensor sharing integration
  const tensorSharing = new TensorSharingIntegration();
  await tensorSharing.initialize();
  
  // Create WebGPU tensor sharing instances with and without optimizations
  const optimizedSharing = new WebGPUTensorSharing(tensorSharing, {
    browserOptimizations: true,
    debug: false
  });
  
  const standardSharing = new WebGPUTensorSharing(tensorSharing, {
    browserOptimizations: false,
    debug: false
  });
  
  // Initialize both instances
  await Promise.all([
    optimizedSharing.initialize(),
    standardSharing.initialize()
  ]);
  
  // Matrix sizes to benchmark
  const sizes = [
    [32, 32, 32],   // Small matrix: [M, K] x [K, N]
    [128, 128, 128], // Medium matrix
    [512, 512, 512]  // Large matrix
  ];
  
  // Number of runs for each benchmark
  const numRuns = 10;
  
  for (const [M, K, N] of sizes) {
    console.log(`\nBenchmarking ${M}x${K} × ${K}x${N} matrices`);
    
    // Create random matrices
    const matrixA = new Float32Array(M * K);
    const matrixB = new Float32Array(K * N);
    
    // Fill with random values
    for (let i = 0; i < M * K; i++) {
      matrixA[i] = Math.random() * 2 - 1;
    }
    
    for (let i = 0; i < K * N; i++) {
      matrixB[i] = Math.random() * 2 - 1;
    }
    
    // Register tensors
    await tensorSharing.registerSharedTensor(
      `matrixA_${M}_${K}`,
      [M, K],
      matrixA,
      'float32',
      'benchmarkModel',
      ['benchmarkModel']
    );
    
    await tensorSharing.registerSharedTensor(
      `matrixB_${K}_${N}`,
      [K, N],
      matrixB,
      'float32',
      'benchmarkModel',
      ['benchmarkModel']
    );
    
    // Run benchmarks
    console.log('Running standard implementation:');
    const standardTimes = [];
    for (let i = 0; i < numRuns; i++) {
      const startTime = performance.now();
      await standardSharing.matmul(
        `matrixA_${M}_${K}`, 'benchmarkModel',
        `matrixB_${K}_${N}`, 'benchmarkModel',
        `result_standard_${i}`, 'benchmarkModel'
      );
      const endTime = performance.now();
      standardTimes.push(endTime - startTime);
    }
    
    console.log('Running browser-optimized implementation:');
    const optimizedTimes = [];
    for (let i = 0; i < numRuns; i++) {
      const startTime = performance.now();
      await optimizedSharing.matmul(
        `matrixA_${M}_${K}`, 'benchmarkModel',
        `matrixB_${K}_${N}`, 'benchmarkModel',
        `result_optimized_${i}`, 'benchmarkModel'
      );
      const endTime = performance.now();
      optimizedTimes.push(endTime - startTime);
    }
    
    // Calculate average times (excluding first run for warm-up)
    const avgStandardTime = standardTimes.slice(1).reduce((a, b) => a + b, 0) / (numRuns - 1);
    const avgOptimizedTime = optimizedTimes.slice(1).reduce((a, b) => a + b, 0) / (numRuns - 1);
    const improvement = ((avgStandardTime - avgOptimizedTime) / avgStandardTime) * 100;
    
    console.log(`Average time (standard): ${avgStandardTime.toFixed(2)} ms`);
    console.log(`Average time (optimized): ${avgOptimizedTime.toFixed(2)} ms`);
    console.log(`Improvement: ${improvement.toFixed(2)}%`);
  }
  
  // Clean up
  optimizedSharing.dispose();
  standardSharing.dispose();
  console.log('Benchmark completed');
}

/**
 * Example 3: Multimodal model with browser optimizations
 */
async function multimodalExample() {
  console.log('Running Multimodal Example with Browser Optimizations');
  
  // Create tensor sharing integration
  const tensorSharing = new TensorSharingIntegration();
  await tensorSharing.initialize();
  
  // Create storage manager
  const storageManager = new StorageManager({
    databaseName: 'multimodal-example',
    storeName: 'tensors'
  });
  await storageManager.initialize();
  
  // Create WebGPU tensor sharing with browser optimizations
  const webgpuTensorSharing = new WebGPUTensorSharing(tensorSharing, {
    browserOptimizations: true,
    storageManager,
    debug: true
  });
  
  // Initialize
  await webgpuTensorSharing.initialize();
  
  // Simulate image processing with ViT
  console.log('Simulating ViT image processing...');
  const imageEmbeddingSize = 768;
  const sequenceLength = 197; // 196 patches + 1 cls token
  
  // Create random image embeddings
  const imageEmbeddings = new Float32Array(sequenceLength * imageEmbeddingSize);
  for (let i = 0; i < imageEmbeddings.length; i++) {
    imageEmbeddings[i] = Math.random() * 2 - 1;
  }
  
  // Register image embeddings
  await tensorSharing.registerSharedTensor(
    'image_embeddings',
    [sequenceLength, imageEmbeddingSize],
    imageEmbeddings,
    'float32',
    'vit-model',
    ['vit-model']
  );
  
  // Simulate text processing with BERT
  console.log('Simulating BERT text processing...');
  const textEmbeddingSize = 768;
  const textSequenceLength = 64;
  
  // Create random text embeddings
  const textEmbeddings = new Float32Array(textSequenceLength * textEmbeddingSize);
  for (let i = 0; i < textEmbeddings.length; i++) {
    textEmbeddings[i] = Math.random() * 2 - 1;
  }
  
  // Register text embeddings
  await tensorSharing.registerSharedTensor(
    'text_embeddings',
    [textSequenceLength, textEmbeddingSize],
    textEmbeddings,
    'float32',
    'bert-model',
    ['bert-model']
  );
  
  // Share embeddings with CLIP model
  await webgpuTensorSharing.shareTensorBetweenModels(
    'image_embeddings', 'vit-model', ['clip-model']
  );
  
  await webgpuTensorSharing.shareTensorBetweenModels(
    'text_embeddings', 'bert-model', ['clip-model']
  );
  
  // Extract CLS tokens
  await webgpuTensorSharing.createTensorView(
    'image_embeddings', 'clip-model',
    'image_cls_token', 'clip-model',
    0, // Start from first token (CLS)
    imageEmbeddingSize // Just one token
  );
  
  await webgpuTensorSharing.createTensorView(
    'text_embeddings', 'clip-model',
    'text_cls_token', 'clip-model',
    0, // Start from first token (CLS)
    textEmbeddingSize // Just one token
  );
  
  // Create projection matrices
  const projectionDim = 512;
  
  // Image projection matrix
  const imageProjection = new Float32Array(imageEmbeddingSize * projectionDim);
  for (let i = 0; i < imageProjection.length; i++) {
    imageProjection[i] = (Math.random() * 2 - 1) * 0.02; // Small random values
  }
  
  // Text projection matrix
  const textProjection = new Float32Array(textEmbeddingSize * projectionDim);
  for (let i = 0; i < textProjection.length; i++) {
    textProjection[i] = (Math.random() * 2 - 1) * 0.02; // Small random values
  }
  
  // Register projection matrices
  await tensorSharing.registerSharedTensor(
    'image_projection',
    [imageEmbeddingSize, projectionDim],
    imageProjection,
    'float32',
    'clip-model',
    ['clip-model']
  );
  
  await tensorSharing.registerSharedTensor(
    'text_projection',
    [textEmbeddingSize, projectionDim],
    textProjection,
    'float32',
    'clip-model',
    ['clip-model']
  );
  
  // Project embeddings to shared space with browser-optimized matrix multiplication
  console.log('Projecting embeddings to shared space...');
  await webgpuTensorSharing.matmul(
    'image_cls_token', 'clip-model',
    'image_projection', 'clip-model',
    'image_embedding', 'clip-model',
    true, false // Transpose first tensor
  );
  
  await webgpuTensorSharing.matmul(
    'text_cls_token', 'clip-model',
    'text_projection', 'clip-model',
    'text_embedding', 'clip-model',
    true, false // Transpose first tensor
  );
  
  // Create custom shader for L2 normalization
  const normalizeShader = `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    
    struct Params {
      length: u32
    }
    
    @group(0) @binding(2) var<uniform> params: Params;
    
    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      // Compute L2 norm
      var sum_squared = 0.0;
      for (var i = 0u; i < params.length; i = i + 1u) {
        let val = input[i];
        sum_squared = sum_squared + val * val;
      }
      
      let norm = sqrt(sum_squared);
      
      // Normalize
      for (var i = 0u; i < params.length; i = i + 1u) {
        output[i] = input[i] / norm;
      }
    }
  `;
  
  // Register custom shader
  await webgpuTensorSharing.createCustomShader(
    'normalize',
    normalizeShader,
    { input: 'float32' },
    { output: 'float32' }
  );
  
  // Normalize embeddings
  console.log('Normalizing embeddings with custom shader...');
  await webgpuTensorSharing.executeCustomShader(
    'normalize',
    { input: { tensorName: 'image_embedding', modelName: 'clip-model' } },
    { output: { tensorName: 'image_embedding_norm', shape: [1, projectionDim], dataType: 'float32' } },
    'clip-model'
  );
  
  await webgpuTensorSharing.executeCustomShader(
    'normalize',
    { input: { tensorName: 'text_embedding', modelName: 'clip-model' } },
    { output: { tensorName: 'text_embedding_norm', shape: [1, projectionDim], dataType: 'float32' } },
    'clip-model'
  );
  
  // Create shader for cosine similarity
  const similarityShader = `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> result: array<f32>;
    
    struct Params {
      length: u32
    }
    
    @group(0) @binding(3) var<uniform> params: Params;
    
    @compute @workgroup_size(1)
    fn main() {
      var dot_product = 0.0;
      
      for (var i = 0u; i < params.length; i = i + 1u) {
        dot_product = dot_product + a[i] * b[i];
      }
      
      result[0] = dot_product;
    }
  `;
  
  // Register similarity shader
  await webgpuTensorSharing.createCustomShader(
    'cosine_similarity',
    similarityShader,
    { a: 'float32', b: 'float32' },
    { result: 'float32' }
  );
  
  // Compute similarity
  console.log('Computing similarity score...');
  await webgpuTensorSharing.executeCustomShader(
    'cosine_similarity',
    {
      a: { tensorName: 'image_embedding_norm', modelName: 'clip-model' },
      b: { tensorName: 'text_embedding_norm', modelName: 'clip-model' }
    },
    {
      result: { tensorName: 'similarity_score', shape: [1], dataType: 'float32' }
    },
    'clip-model'
  );
  
  // Get similarity score
  const similarityTensor = await tensorSharing.getSharedTensor('similarity_score', 'clip-model');
  const similarityScore = similarityTensor.getData()[0];
  
  console.log(`Image-Text Similarity Score: ${similarityScore.toFixed(4)}`);
  
  // Check GPU memory usage
  const gpuMemoryUsage = webgpuTensorSharing.getGPUMemoryUsage();
  console.log(`GPU Memory Usage: ${(gpuMemoryUsage / (1024 * 1024)).toFixed(2)} MB`);
  
  // Optimize memory
  console.log('Optimizing memory usage...');
  await webgpuTensorSharing.optimizeMemoryUsage();
  
  // Check memory usage after optimization
  const gpuMemoryAfter = webgpuTensorSharing.getGPUMemoryUsage();
  console.log(`GPU Memory After Optimization: ${(gpuMemoryAfter / (1024 * 1024)).toFixed(2)} MB`);
  
  // Clean up
  webgpuTensorSharing.dispose();
  console.log('Multimodal example completed');
}

/**
 * Example 4: Quantization for memory efficiency
 */
async function quantizationExample() {
  console.log('Running Quantization Example with Browser Optimizations');
  
  // Create tensor sharing integration
  const tensorSharing = new TensorSharingIntegration();
  await tensorSharing.initialize();
  
  // Create WebGPU tensor sharing with browser optimizations
  const webgpuTensorSharing = new WebGPUTensorSharing(tensorSharing, {
    browserOptimizations: true,
    debug: true
  });
  
  // Initialize
  await webgpuTensorSharing.initialize();
  
  // Create a large tensor
  const size = 1000000; // 1M elements
  const largeTensor = new Float32Array(size);
  
  // Fill with random values
  for (let i = 0; i < size; i++) {
    largeTensor[i] = Math.random() * 2 - 1;
  }
  
  // Register tensor
  await tensorSharing.registerSharedTensor(
    'large_tensor',
    [size],
    largeTensor,
    'float32',
    'quantModel',
    ['quantModel']
  );
  
  // Get memory usage before quantization
  let gpuMemory = webgpuTensorSharing.getGPUMemoryUsage();
  console.log(`Memory usage before quantization: ${(gpuMemory / (1024 * 1024)).toFixed(2)} MB`);
  
  // Quantize to int8
  console.log('Quantizing tensor to int8...');
  const { tensorName: quantizedName, scaleTensorName } = await webgpuTensorSharing.quantize(
    'large_tensor', 'quantModel',
    'quantized_tensor', 'quantModel'
  );
  
  // Get memory usage after quantization
  gpuMemory = webgpuTensorSharing.getGPUMemoryUsage();
  console.log(`Memory usage after quantization: ${(gpuMemory / (1024 * 1024)).toFixed(2)} MB`);
  
  // Get scale value
  const scaleTensor = await tensorSharing.getSharedTensor(scaleTensorName, 'quantModel');
  const scale = scaleTensor.getData()[0];
  console.log(`Quantization scale: ${scale.toFixed(4)}`);
  
  // Dequantize back to float32
  console.log('Dequantizing tensor...');
  const dequantizedName = await webgpuTensorSharing.dequantize(
    quantizedName, 'quantModel',
    scaleTensorName, 'quantModel',
    'dequantized_tensor', 'quantModel'
  );
  
  // Calculate mean squared error
  const originalTensor = await tensorSharing.getSharedTensor('large_tensor', 'quantModel');
  const dequantizedTensor = await tensorSharing.getSharedTensor(dequantizedName, 'quantModel');
  
  const originalData = originalTensor.getData();
  const dequantizedData = dequantizedTensor.getData();
  
  let sumSquaredError = 0;
  for (let i = 0; i < size; i++) {
    const error = originalData[i] - dequantizedData[i];
    sumSquaredError += error * error;
  }
  
  const mse = sumSquaredError / size;
  console.log(`Mean squared error after quantization/dequantization: ${mse.toFixed(6)}`);
  
  // Clean up
  webgpuTensorSharing.dispose();
  console.log('Quantization example completed');
}

/**
 * Run all examples
 */
async function runExamples() {
  console.log('Starting WebGPU Tensor Sharing Examples with Browser Optimizations');
  console.log('============================================================');
  
  try {
    await basicExample();
    console.log('\n============================================================\n');
    
    await matrixMultiplicationBenchmark();
    console.log('\n============================================================\n');
    
    await multimodalExample();
    console.log('\n============================================================\n');
    
    await quantizationExample();
    console.log('\n============================================================\n');
    
    console.log('All examples completed successfully!');
  } catch (error) {
    console.error('Error running examples:', error);
  }
}

// Export examples
export {
  basicExample,
  matrixMultiplicationBenchmark,
  multimodalExample,
  quantizationExample,
  runExamples
};

// Run examples when in browser environment
if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', () => {
    const runButton = document.getElementById('run-examples');
    if (runButton) {
      runButton.addEventListener('click', runExamples);
    }
  });
}