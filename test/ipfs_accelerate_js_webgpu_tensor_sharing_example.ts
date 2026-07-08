/**
 * WebGPU Tensor Sharing Example
 * 
 * This example demonstrates how to use the WebGPU Tensor Sharing integration
 * to efficiently share tensors between models and accelerate operations with WebGPU
 */

import { TensorSharingIntegration } from './ipfs_accelerate_js_tensor_sharing_integration';
import { WebGPUTensorSharing } from './ipfs_accelerate_js_webgpu_tensor_sharing';
import { StorageManager } from './ipfs_accelerate_js_storage_manager';
import { WebGPUBackend } from './ipfs_accelerate_js_webgpu_backend';

/**
 * Run the WebGPU Tensor Sharing example
 */
async function runWebGPUTensorSharingExample() {
  console.log('Initializing WebGPU Tensor Sharing Example...');
  
  try {
    // Check if WebGPU is supported
    if (!navigator.gpu) {
      console.error('WebGPU is not supported in this browser. Please try a browser that supports WebGPU.');
      return;
    }
    
    // Step 1: Initialize components
    console.log('1. Creating storage manager...');
    const storageManager = new StorageManager({
      databaseName: 'webgpu-tensor-sharing-example',
      storeName: 'tensors'
    });
    await storageManager.initialize();
    
    console.log('2. Creating tensor sharing integration...');
    const tensorSharing = new TensorSharingIntegration();
    await tensorSharing.initialize();
    
    console.log('3. Creating WebGPU backend...');
    const webgpuBackend = new WebGPUBackend({
      powerPreference: 'high-performance',
      shaderCompilation: {
        precompile: true,
        cacheShaders: true,
        browserOptimizations: true
      }
    });
    
    console.log('4. Creating WebGPU tensor sharing integration...');
    const webgpuTensorSharing = new WebGPUTensorSharing(tensorSharing, {
      webgpuBackend,
      storageManager,
      enableZeroCopy: true,
      cachePipelines: true,
      debug: true
    });
    
    const initialized = await webgpuTensorSharing.initialize();
    if (!initialized) {
      console.error('Failed to initialize WebGPU Tensor Sharing. WebGPU may not be supported.');
      return;
    }
    
    console.log('Successfully initialized all components!');
    
    // Step 2: Create some example tensors for BERT and ViT models
    console.log('\nCreating example tensors for BERT and ViT models...');
    
    // BERT hidden states tensor (batch_size=1, sequence_length=128, hidden_size=768)
    const bertHiddenSize = 768;
    const bertSeqLength = 128;
    const bertHiddenStates = new Float32Array(bertSeqLength * bertHiddenSize);
    
    // Initialize with some data
    for (let i = 0; i < bertHiddenStates.length; i++) {
      bertHiddenStates[i] = Math.random() * 2 - 1; // Random values between -1 and 1
    }
    
    // Register tensor with tensor sharing system
    await tensorSharing.registerSharedTensor(
      'bert_hidden_states',
      [bertSeqLength, bertHiddenSize],
      bertHiddenStates,
      'float32',
      'bert-base-uncased',
      ['bert-base-uncased']
    );
    
    console.log('Created BERT hidden states tensor: [128, 768]');
    
    // ViT image embedding tensor (batch_size=1, sequence_length=197, hidden_size=768)
    // Note: 197 = 196 patches + 1 cls token for a 224x224 image with patch size 16
    const vitHiddenSize = 768;
    const vitSeqLength = 197;
    const vitEmbeddings = new Float32Array(vitSeqLength * vitHiddenSize);
    
    // Initialize with some data
    for (let i = 0; i < vitEmbeddings.length; i++) {
      vitEmbeddings[i] = Math.random() * 2 - 1;
    }
    
    // Register tensor with tensor sharing system
    await tensorSharing.registerSharedTensor(
      'vit_embeddings',
      [vitSeqLength, vitHiddenSize],
      vitEmbeddings,
      'float32',
      'vit-base-patch16-224',
      ['vit-base-patch16-224']
    );
    
    console.log('Created ViT embeddings tensor: [197, 768]');
    
    // CLIP projection matrices
    // Text projection matrix (hidden_size x projection_dim) - (768 x 512)
    const clipProjectionDim = 512;
    const clipTextProjection = new Float32Array(bertHiddenSize * clipProjectionDim);
    
    // Initialize with random weights
    for (let i = 0; i < clipTextProjection.length; i++) {
      clipTextProjection[i] = (Math.random() * 2 - 1) * 0.02; // Small random values
    }
    
    // Register tensor
    await tensorSharing.registerSharedTensor(
      'clip_text_projection',
      [bertHiddenSize, clipProjectionDim],
      clipTextProjection,
      'float32',
      'clip-vit-base-patch16',
      ['clip-vit-base-patch16']
    );
    
    // Vision projection matrix (hidden_size x projection_dim) - (768 x 512)
    const clipVisionProjection = new Float32Array(vitHiddenSize * clipProjectionDim);
    
    // Initialize with random weights
    for (let i = 0; i < clipVisionProjection.length; i++) {
      clipVisionProjection[i] = (Math.random() * 2 - 1) * 0.02;
    }
    
    // Register tensor
    await tensorSharing.registerSharedTensor(
      'clip_vision_projection',
      [vitHiddenSize, clipProjectionDim],
      clipVisionProjection,
      'float32',
      'clip-vit-base-patch16',
      ['clip-vit-base-patch16']
    );
    
    console.log('Created CLIP projection matrices: [768, 512]');
    
    // Step 3: Use WebGPU to accelerate operations
    console.log('\nPerforming WebGPU-accelerated operations...');
    
    // 3.1 Project BERT hidden states to CLIP embedding space
    console.log('1. Projecting BERT hidden states to CLIP embedding space...');
    
    // We'll use the CLS token (first token) as the text representation
    // First create a view of just the CLS token
    await tensorSharing.createTensorView(
      'bert_hidden_states',
      'bert_cls_token',
      0, // Starting from the first token
      bertHiddenSize, // Just one token
      'bert-base-uncased'
    );
    
    // Share with CLIP model
    await tensorSharing.shareTensorBetweenModels(
      'bert_cls_token',
      'bert-base-uncased',
      ['clip-vit-base-patch16']
    );
    
    // Project the CLS token to CLIP space using matrix multiplication
    await webgpuTensorSharing.matmul(
      'bert_cls_token', 'clip-vit-base-patch16',
      'clip_text_projection', 'clip-vit-base-patch16',
      'text_embedding', 'clip-vit-base-patch16',
      false, false // No transposes
    );
    
    console.log('Created text embedding in shared CLIP space: [1, 512]');
    
    // 3.2 Project ViT embeddings to CLIP embedding space
    console.log('2. Projecting ViT embeddings to CLIP embedding space...');
    
    // Extract CLS token from ViT (first token is CLS)
    await tensorSharing.createTensorView(
      'vit_embeddings',
      'vit_cls_token',
      0, // Starting from the first token (CLS)
      vitHiddenSize,
      'vit-base-patch16-224'
    );
    
    // Share with CLIP model
    await tensorSharing.shareTensorBetweenModels(
      'vit_cls_token',
      'vit-base-patch16-224',
      ['clip-vit-base-patch16']
    );
    
    // Project the CLS token to CLIP space
    await webgpuTensorSharing.matmul(
      'vit_cls_token', 'clip-vit-base-patch16',
      'clip_vision_projection', 'clip-vit-base-patch16',
      'vision_embedding', 'clip-vit-base-patch16',
      false, false
    );
    
    console.log('Created vision embedding in shared CLIP space: [1, 512]');
    
    // 3.3 Compute cosine similarity between text and vision embeddings
    console.log('3. Computing cosine similarity between text and vision embeddings...');
    
    // For cosine similarity, we'll need to normalize the embeddings first
    // Create a custom shader for L2 normalization
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
    
    // Register the custom shader
    await webgpuTensorSharing.createCustomShader(
      'normalize',
      normalizeShader,
      { input: 'float32' },
      { output: 'float32' }
    );
    
    // Normalize text embedding
    await webgpuTensorSharing.executeCustomShader(
      'normalize',
      { input: { tensorName: 'text_embedding', modelName: 'clip-vit-base-patch16' } },
      { output: { tensorName: 'text_embedding_normalized', shape: [1, clipProjectionDim], dataType: 'float32' } },
      'clip-vit-base-patch16',
      [1, 1, 1], // Workgroup size
      [1, 1, 1]  // Number of workgroups
    );
    
    // Normalize vision embedding
    await webgpuTensorSharing.executeCustomShader(
      'normalize',
      { input: { tensorName: 'vision_embedding', modelName: 'clip-vit-base-patch16' } },
      { output: { tensorName: 'vision_embedding_normalized', shape: [1, clipProjectionDim], dataType: 'float32' } },
      'clip-vit-base-patch16',
      [1, 1, 1],
      [1, 1, 1]
    );
    
    // Create a custom shader for dot product (cosine similarity with normalized vectors)
    const dotProductShader = `
      @group(0) @binding(0) var<storage, read> a: array<f32>;
      @group(0) @binding(1) var<storage, read> b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> result: array<f32>;
      
      struct Params {
        length: u32
      }
      
      @group(0) @binding(3) var<uniform> params: Params;
      
      @compute @workgroup_size(1)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        var dot_product = 0.0;
        for (var i = 0u; i < params.length; i = i + 1u) {
          dot_product = dot_product + a[i] * b[i];
        }
        
        result[0] = dot_product;
      }
    `;
    
    // Register the custom shader
    await webgpuTensorSharing.createCustomShader(
      'dot_product',
      dotProductShader,
      { a: 'float32', b: 'float32' },
      { result: 'float32' }
    );
    
    // Compute cosine similarity
    await webgpuTensorSharing.executeCustomShader(
      'dot_product',
      { 
        a: { tensorName: 'text_embedding_normalized', modelName: 'clip-vit-base-patch16' },
        b: { tensorName: 'vision_embedding_normalized', modelName: 'clip-vit-base-patch16' }
      },
      { 
        result: { tensorName: 'similarity_score', shape: [1], dataType: 'float32' }
      },
      'clip-vit-base-patch16',
      [1, 1, 1],
      [1, 1, 1]
    );
    
    console.log('Computed similarity score between text and vision embeddings');
    
    // Step 4: Read the results
    console.log('\nRetrieving and displaying results...');
    
    // Get normalized text embedding
    const textEmbedding = await tensorSharing.getSharedTensor(
      'text_embedding_normalized',
      'clip-vit-base-patch16'
    );
    
    // Get normalized vision embedding
    const visionEmbedding = await tensorSharing.getSharedTensor(
      'vision_embedding_normalized',
      'clip-vit-base-patch16'
    );
    
    // Get similarity score
    const similarityScore = await tensorSharing.getSharedTensor(
      'similarity_score',
      'clip-vit-base-patch16'
    );
    
    // Display the results
    console.log('\nResults:');
    console.log('Text embedding (first 5 values):', 
      Array.from(textEmbedding.getData().slice(0, 5))
        .map(v => v.toFixed(4))
        .join(', ')
    );
    
    console.log('Vision embedding (first 5 values):', 
      Array.from(visionEmbedding.getData().slice(0, 5))
        .map(v => v.toFixed(4))
        .join(', ')
    );
    
    console.log('Similarity score:', similarityScore.getData()[0].toFixed(4));
    
    // Step 5: Memory analysis and optimization
    console.log('\nMemory analysis and optimization...');
    
    // Get memory usage information
    const gpuMemoryUsage = webgpuTensorSharing.getGPUMemoryUsage();
    const cpuMemoryUsage = await tensorSharing.getTensorMemoryUsage();
    const modelMemoryUsage = await tensorSharing.getModelMemoryUsage();
    
    console.log('GPU memory usage:', (gpuMemoryUsage / (1024 * 1024)).toFixed(2), 'MB');
    console.log('CPU memory usage:', (cpuMemoryUsage / (1024 * 1024)).toFixed(2), 'MB');
    console.log('Memory usage by model:');
    
    for (const [model, usage] of Object.entries(modelMemoryUsage)) {
      console.log(`  ${model}: ${(usage / (1024 * 1024)).toFixed(2)} MB`);
    }
    
    // Optimize memory usage
    console.log('\nOptimizing memory usage...');
    await webgpuTensorSharing.optimizeMemoryUsage();
    
    const gpuMemoryAfter = webgpuTensorSharing.getGPUMemoryUsage();
    console.log('GPU memory after optimization:', (gpuMemoryAfter / (1024 * 1024)).toFixed(2), 'MB');
    
    // Step 6: Quantize tensors to save memory
    console.log('\nQuantizing tensors to int8 to save memory...');
    
    // Quantize BERT hidden states
    const { tensorName: bertQuantizedName, scaleTensorName: bertScaleName } = 
      await webgpuTensorSharing.quantize(
        'bert_hidden_states',
        'bert-base-uncased',
        'bert_hidden_states_quantized',
        'bert-base-uncased'
      );
    
    console.log(`Quantized BERT hidden states to int8 (${bertQuantizedName}, scale: ${bertScaleName})`);
    
    // Dequantize to verify
    await webgpuTensorSharing.dequantize(
      bertQuantizedName,
      'bert-base-uncased',
      bertScaleName,
      'bert-base-uncased',
      'bert_hidden_states_dequantized',
      'bert-base-uncased'
    );
    
    // Compare original and dequantized
    const originalTensor = await tensorSharing.getSharedTensor(
      'bert_hidden_states',
      'bert-base-uncased'
    );
    
    const dequantizedTensor = await tensorSharing.getSharedTensor(
      'bert_hidden_states_dequantized',
      'bert-base-uncased'
    );
    
    // Compute mean squared error
    const originalData = originalTensor.getData();
    const dequantizedData = dequantizedTensor.getData();
    
    let sumSquaredError = 0;
    for (let i = 0; i < originalData.length; i++) {
      const error = originalData[i] - dequantizedData[i];
      sumSquaredError += error * error;
    }
    
    const mse = sumSquaredError / originalData.length;
    console.log('Mean squared error after quantization/dequantization:', mse.toFixed(6));
    
    // Print tensors list before cleanup
    console.log('\nTensor list before cleanup:');
    const tensors = await tensorSharing.getRegisteredTensors();
    for (const tensor of tensors) {
      console.log(`  ${tensor}`);
    }
    
    // Final cleanup
    console.log('\nCleaning up resources...');
    webgpuTensorSharing.dispose();
    
    console.log('WebGPU Tensor Sharing Example completed successfully!');
  } catch (error) {
    console.error('Error in WebGPU Tensor Sharing Example:', error);
  }
}

// Run the example when the page loads
window.addEventListener('DOMContentLoaded', () => {
  const runButton = document.getElementById('run-example');
  if (runButton) {
    runButton.addEventListener('click', runWebGPUTensorSharingExample);
  } else {
    runWebGPUTensorSharingExample();
  }
});