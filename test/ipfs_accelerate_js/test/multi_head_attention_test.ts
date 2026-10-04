/**
 * Browser-Specific Multi-Head Attention Test
 * Tests the performance of browser-specific Multi-Head Attention operations
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
  const data = Array(size).fill(0).map(() => Math.random() * 2 - 1); // Range [-1, 1]
  return new Tensor<number>(shape, data, { dataType: 'float32' });
}

/**
 * Generate random projection matrices for attention
 */
function generateAttentionParameters(
  batchSize: number, 
  sequenceLength: number, 
  hiddenSize: number, 
  numHeads: number
): { 
  query: Tensor<number>, 
  key: Tensor<number>, 
  value: Tensor<number>,
  qkv_weights: Tensor<number>,
  output_weights: Tensor<number>
} {
  // Input shape: [batch_size, seq_len, hidden_size]
  const inputShape = [batchSize, sequenceLength, hiddenSize];
  
  // Create weights for QKV projections (3 projections combined)
  // Shape: [hidden_size, 3 * hidden_size]
  const qkvWeightsData = Array(hiddenSize * 3 * hiddenSize).fill(0)
    .map(() => (Math.random() * 0.2 - 0.1) / Math.sqrt(hiddenSize)); // Xavier init
  
  const qkvWeights = new Tensor<number>(
    [hiddenSize, 3 * hiddenSize], 
    qkvWeightsData, 
    { dataType: 'float32' }
  );
  
  // Create weights for output projection
  // Shape: [hidden_size, hidden_size]
  const outputWeightsData = Array(hiddenSize * hiddenSize).fill(0)
    .map(() => (Math.random() * 0.2 - 0.1) / Math.sqrt(hiddenSize)); // Xavier init
  
  const outputWeights = new Tensor<number>(
    [hiddenSize, hiddenSize], 
    outputWeightsData, 
    { dataType: 'float32' }
  );
  
  // Create inputs
  const query = generateRandomTensor(inputShape);
  const key = generateRandomTensor(inputShape);
  const value = generateRandomTensor(inputShape);
  
  return {
    query,
    key,
    value,
    qkv_weights: qkvWeights,
    output_weights: outputWeights
  };
}

/**
 * CPU implementation of Multi-Head Attention
 * Performs scaled dot-product attention with multiple heads
 */
function cpuMultiHeadAttention(
  query: Tensor<number>,
  key: Tensor<number>,
  value: Tensor<number>,
  numHeads: number,
  scale: number = 1.0,
  dropout: number = 0.0,
  useCausalMask: boolean = false
): Tensor<number> {
  const qShape = query.shape;
  const batchSize = qShape[0];
  const seqLen = qShape[1];
  const hiddenSize = qShape[2];
  const headDim = hiddenSize / numHeads;
  
  // Reshape query, key, value to [batch, seq_len, num_heads, head_dim]
  const reshapedQ = reshapeTensor(query, batchSize, seqLen, numHeads, headDim);
  const reshapedK = reshapeTensor(key, batchSize, seqLen, numHeads, headDim);
  const reshapedV = reshapeTensor(value, batchSize, seqLen, numHeads, headDim);
  
  // Compute attention scores: [batch, num_heads, seq_len, seq_len]
  const attentionScores = new Array(batchSize * numHeads * seqLen * seqLen).fill(0);
  
  // Compute dot product of query and key for each batch and head
  for (let b = 0; b < batchSize; b++) {
    for (let h = 0; h < numHeads; h++) {
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          let dotProduct = 0;
          
          for (let d = 0; d < headDim; d++) {
            // Calculate indices in reshapedQ and reshapedK
            const qIdx = ((b * seqLen + i) * numHeads + h) * headDim + d;
            const kIdx = ((b * seqLen + j) * numHeads + h) * headDim + d;
            
            dotProduct += reshapedQ[qIdx] * reshapedK[kIdx];
          }
          
          // Scale dot product
          dotProduct *= scale;
          
          // Apply causal mask if needed
          if (useCausalMask && j > i) {
            dotProduct = -1e9; // Mask out future positions
          }
          
          const attentionIdx = (((b * numHeads) + h) * seqLen + i) * seqLen + j;
          attentionScores[attentionIdx] = dotProduct;
        }
      }
    }
  }
  
  // Apply softmax to get attention weights
  const attentionWeights = applySoftmax(attentionScores, batchSize, numHeads, seqLen);
  
  // Apply dropout if needed
  if (dropout > 0) {
    applyDropout(attentionWeights, dropout);
  }
  
  // Compute weighted sum of values
  const output = new Array(batchSize * seqLen * hiddenSize).fill(0);
  
  for (let b = 0; b < batchSize; b++) {
    for (let h = 0; h < numHeads; h++) {
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          const attentionIdx = (((b * numHeads) + h) * seqLen + i) * seqLen + j;
          const weight = attentionWeights[attentionIdx];
          
          for (let d = 0; d < headDim; d++) {
            // Calculate indices
            const vIdx = ((b * seqLen + j) * numHeads + h) * headDim + d;
            const outIdx = ((b * seqLen + i) * numHeads + h) * headDim + d;
            
            output[outIdx] += weight * reshapedV[vIdx];
          }
        }
      }
    }
  }
  
  // Reshape output back to [batch, seq_len, hidden_size]
  return new Tensor<number>([batchSize, seqLen, hiddenSize], output, { dataType: 'float32' });
}

/**
 * Helper function to reshape tensor for attention computation
 */
function reshapeTensor(
  tensor: Tensor<number>, 
  batchSize: number, 
  seqLen: number, 
  numHeads: number, 
  headDim: number
): number[] {
  const result = new Array(batchSize * seqLen * numHeads * headDim).fill(0);
  const hiddenSize = numHeads * headDim;
  
  for (let b = 0; b < batchSize; b++) {
    for (let s = 0; s < seqLen; s++) {
      for (let h = 0; h < numHeads; h++) {
        for (let d = 0; d < headDim; d++) {
          // Input index: [batch, seq, hidden]
          const inputIdx = (b * seqLen + s) * hiddenSize + h * headDim + d;
          
          // Output index: [batch, seq, head, head_dim]
          const outputIdx = ((b * seqLen + s) * numHeads + h) * headDim + d;
          
          result[outputIdx] = tensor.data[inputIdx];
        }
      }
    }
  }
  
  return result;
}

/**
 * Apply softmax to attention scores
 */
function applySoftmax(
  scores: number[], 
  batchSize: number, 
  numHeads: number, 
  seqLen: number
): number[] {
  const result = new Array(scores.length).fill(0);
  
  for (let b = 0; b < batchSize; b++) {
    for (let h = 0; h < numHeads; h++) {
      for (let i = 0; i < seqLen; i++) {
        let max = -Infinity;
        
        // Find max for numerical stability
        for (let j = 0; j < seqLen; j++) {
          const idx = (((b * numHeads) + h) * seqLen + i) * seqLen + j;
          max = Math.max(max, scores[idx]);
        }
        
        // Compute exp and sum
        let sum = 0;
        for (let j = 0; j < seqLen; j++) {
          const idx = (((b * numHeads) + h) * seqLen + i) * seqLen + j;
          const expVal = Math.exp(scores[idx] - max);
          result[idx] = expVal;
          sum += expVal;
        }
        
        // Normalize
        for (let j = 0; j < seqLen; j++) {
          const idx = (((b * numHeads) + h) * seqLen + i) * seqLen + j;
          result[idx] /= sum;
        }
      }
    }
  }
  
  return result;
}

/**
 * Apply dropout to attention weights
 */
function applyDropout(weights: number[], dropout: number): void {
  for (let i = 0; i < weights.length; i++) {
    if (Math.random() < dropout) {
      weights[i] = 0;
    } else {
      weights[i] /= (1 - dropout); // Scale to maintain expected values
    }
  }
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
 * Test Multi-Head Attention performance across browsers
 */
async function testMultiHeadAttentionPerformance(
  backend: WebGPUBackend, 
  query: Tensor<number>,
  key: Tensor<number>,
  value: Tensor<number>,
  numHeads: number,
  scale: number = 1.0
) {
  console.log('\n=== Testing Multi-Head Attention Performance ===');
  
  const actualBrowserType = detectBrowserType();
  const iterations = 10;
  const warmupIterations = 5;
  
  // Compute CPU reference result
  console.log('Computing CPU reference result...');
  const cpuResult = cpuMultiHeadAttention(query, key, value, numHeads, scale);
  
  // Test each browser optimization
  const browserTypes = [
    { name: 'Generic (No Optimization)', type: null },
    { name: 'Chrome', type: BrowserType.CHROME },
    { name: 'Firefox', type: BrowserType.FIREFOX },
    { name: 'Safari', type: BrowserType.SAFARI },
    { name: 'Edge', type: BrowserType.EDGE }
  ];
  
  const results: Record<string, number> = {};
  
  // Get actual browser shader for later comparison
  await loadBrowserShader(ShaderType.MULTI_HEAD_ATTENTION);
  
  // Test Multi-Head Attention implementations
  console.log('\nTesting Multi-Head Attention implementations:');
  for (const browser of browserTypes) {
    const name = browser.name;
    console.log(`Testing ${name} optimization...`);
    
    // Warmup
    for (let i = 0; i < warmupIterations; i++) {
      await backend.multiHeadAttention(query, key, value, {
        useBrowserOptimizations: browser.type !== null,
        browserType: browser.type,
        numHeads,
        scale
      });
    }
    
    // Timed iterations
    const startTime = performance.now();
    let result;
    
    for (let i = 0; i < iterations; i++) {
      result = await backend.multiHeadAttention(query, key, value, {
        useBrowserOptimizations: browser.type !== null,
        browserType: browser.type,
        numHeads,
        scale
      });
    }
    
    const endTime = performance.now();
    const avgTime = (endTime - startTime) / iterations;
    results[name] = avgTime;
    
    console.log(`${name}: ${avgTime.toFixed(2)} ms`);
    
    // Verify correctness
    const isCorrect = areTensorsEqual(cpuResult, result!, 1e-4);
    console.log(`${name} correctness: ${isCorrect ? 'PASSED' : 'FAILED'}`);
  }
  
  // Calculate speedups
  const baseTime = results['Generic (No Optimization)'];
  console.log('\nSpeedup compared to generic implementation:');
  
  for (const browser of browserTypes) {
    if (browser.type === null) continue;
    
    const speedup = baseTime / results[browser.name];
    console.log(`${browser.name}: ${speedup.toFixed(2)}x`);
    
    // Highlight current browser
    if (browser.type === actualBrowserType) {
      console.log(`Current browser (${browser.name}) speedup: ${speedup.toFixed(2)}x`);
    }
  }
  
  return results;
}

/**
 * Test performance with different model sizes
 */
async function testDifferentModelSizes(backend: WebGPUBackend) {
  console.log('\n=== Testing Different Model Sizes ===');
  
  const modelConfigs = [
    { name: "Small", batchSize: 4, seqLen: 64, hiddenSize: 256, numHeads: 4 },
    { name: "Medium", batchSize: 2, seqLen: 128, hiddenSize: 512, numHeads: 8 },
    { name: "Large", batchSize: 1, seqLen: 256, hiddenSize: 768, numHeads: 12 },
    { name: "XL", batchSize: 1, seqLen: 384, hiddenSize: 1024, numHeads: 16 }
  ];
  
  for (const config of modelConfigs) {
    console.log(`\n--- Testing ${config.name} model (hidden: ${config.hiddenSize}, heads: ${config.numHeads}) ---`);
    
    // Create test tensors
    const { query, key, value } = generateAttentionParameters(
      config.batchSize, 
      config.seqLen, 
      config.hiddenSize, 
      config.numHeads
    );
    
    // Run performance test
    await testMultiHeadAttentionPerformance(
      backend, 
      query, 
      key, 
      value, 
      config.numHeads, 
      1.0 / Math.sqrt(config.hiddenSize / config.numHeads)
    );
  }
}

/**
 * Test fusion of attention with other operations
 */
async function testAttentionFusion(
  backend: WebGPUBackend, 
  input: Tensor<number>, 
  qkvWeights: Tensor<number>,
  outputWeights: Tensor<number>,
  numHeads: number
) {
  console.log('\n=== Testing Attention Fusion Performance ===');
  
  const actualBrowserType = detectBrowserType();
  const iterations = 10;
  const warmupIterations = 5;
  
  // Test common sequence: QKV projection -> Attention -> Output projection
  console.log('Testing full attention block with fusion...');
  
  // Shapes
  const batchSize = input.shape[0];
  const seqLen = input.shape[1];
  const hiddenSize = input.shape[2];
  const headDim = hiddenSize / numHeads;
  const scale = 1.0 / Math.sqrt(headDim);
  
  // Reference CPU computation (simplified)
  console.log('Computing CPU reference result (simplified)...');
  
  // Run unfused version with separate operations
  console.log('Running unfused operations (separate projections and attention)...');
  
  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    // QKV projections
    const projections = await backend.matmul(input, qkvWeights, {
      useBrowserOptimizations: true
    });
    
    // Split projections
    const q = await backend.slice(projections, 2, 0, hiddenSize);
    const k = await backend.slice(projections, 2, hiddenSize, 2 * hiddenSize);
    const v = await backend.slice(projections, 2, 2 * hiddenSize, 3 * hiddenSize);
    
    // Multi-head attention
    const attentionOutput = await backend.multiHeadAttention(q, k, v, {
      useBrowserOptimizations: true,
      numHeads,
      scale
    });
    
    // Output projection
    await backend.matmul(attentionOutput, outputWeights, {
      useBrowserOptimizations: true
    });
  }
  
  // Timed iterations
  const startTimeUnfused = performance.now();
  let unfusedResult;
  
  for (let i = 0; i < iterations; i++) {
    // QKV projections
    const projections = await backend.matmul(input, qkvWeights, {
      useBrowserOptimizations: true
    });
    
    // Split projections
    const q = await backend.slice(projections, 2, 0, hiddenSize);
    const k = await backend.slice(projections, 2, hiddenSize, 2 * hiddenSize);
    const v = await backend.slice(projections, 2, 2 * hiddenSize, 3 * hiddenSize);
    
    // Multi-head attention
    const attentionOutput = await backend.multiHeadAttention(q, k, v, {
      useBrowserOptimizations: true,
      numHeads,
      scale
    });
    
    // Output projection
    unfusedResult = await backend.matmul(attentionOutput, outputWeights, {
      useBrowserOptimizations: true
    });
  }
  
  const endTimeUnfused = performance.now();
  const avgTimeUnfused = (endTimeUnfused - startTimeUnfused) / iterations;
  
  console.log(`Unfused execution: ${avgTimeUnfused.toFixed(2)} ms`);
  
  // Fused execution
  console.log('Running fused operations (combined attention block)...');
  
  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    await backend.executeOperations(
      [input, qkvWeights, outputWeights],
      ['attentionBlock'],
      { 
        useBrowserOptimizations: true,
        useFusion: true,
        attentionOptions: { 
          numHeads,
          scale,
          causalMask: false
        }
      }
    );
  }
  
  // Timed iterations
  const startTimeFused = performance.now();
  let fusedResult;
  
  for (let i = 0; i < iterations; i++) {
    fusedResult = await backend.executeOperations(
      [input, qkvWeights, outputWeights],
      ['attentionBlock'],
      { 
        useBrowserOptimizations: true,
        useFusion: true,
        attentionOptions: { 
          numHeads,
          scale,
          causalMask: false
        }
      }
    );
  }
  
  const endTimeFused = performance.now();
  const avgTimeFused = (endTimeFused - startTimeFused) / iterations;
  
  console.log(`Fused execution: ${avgTimeFused.toFixed(2)} ms`);
  
  // Verify correctness
  const areTensorsRoughlyEqual = areTensorsEqual(unfusedResult!, fusedResult!, 1e-4);
  console.log(`Fusion correctness: ${areTensorsRoughlyEqual ? 'PASSED' : 'FAILED'}`);
  
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
  console.log('Browser-Specific Multi-Head Attention Test');
  console.log('========================================');
  
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
    
    // Create test tensors for a typical transformer layer
    console.log('Creating test tensors...');
    const batchSize = 2;
    const sequenceLength = 128;
    const hiddenSize = 512; // Common size in medium models
    const numHeads = 8;
    
    const { 
      query, 
      key, 
      value, 
      qkv_weights, 
      output_weights 
    } = generateAttentionParameters(
      batchSize, 
      sequenceLength, 
      hiddenSize, 
      numHeads
    );
    
    // Run standard performance test
    await testMultiHeadAttentionPerformance(
      backend, 
      query, 
      key, 
      value, 
      numHeads,
      1.0 / Math.sqrt(hiddenSize / numHeads) // Scale factor for attention
    );
    
    // Test with different model sizes
    await testDifferentModelSizes(backend);
    
    // Test fusion with projections (full attention block)
    await testAttentionFusion(backend, query, qkv_weights, output_weights, numHeads);
    
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