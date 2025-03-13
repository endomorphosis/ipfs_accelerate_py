/**
 * WebGPU Operation Fusion Tests
 * 
 * This file contains tests specifically focused on measuring the impact of
 * operation fusion optimizations in the WebGPU backend.
 */

import { expect } from '@jest/globals';
import { performance } from 'perf_hooks';

// Import WebGPU backend and optimizer
import { WebGPUBackend } from '../../../src/hardware/webgpu/backend';
import { WebGPUOptimizer } from '../../../src/hardware/webgpu/optimizations/webgpu_optimizer';
import { OperationFusion } from '../../../src/hardware/webgpu/optimizations/operation_fusion';
import { TensorShape, DataType } from '../../../src/core/tensor_types';
import { Tensor } from '../../../src/core/tensor';
import { BrowserDetector } from '../../../src/browser/browser_detector';
import { BrowserType } from '../../../src/browser/browser_types';

// Import neural network layers
import { LinearLayer } from '../../../src/model/transformers/layers/linear';
import { LayerNormalization } from '../../../src/model/transformers/layers/normalization';
import { MultiHeadAttention } from '../../../src/model/transformers/layers/attention';

/**
 * Interface for operation fusion test results
 */
interface FusionTestResult {
  operationName: string;
  fusionPattern: string;
  shapeInfo: string;
  unfusedTime: number;
  fusedTime: number;
  speedup: number;
  memorySavings: number;
  browser: string;
}

/**
 * Helper class for running operation fusion tests
 */
class OperationFusionTester {
  private results: FusionTestResult[] = [];
  private browser: BrowserType;
  
  constructor() {
    this.browser = BrowserDetector.detectBrowser();
  }
  
  /**
   * Runs a test comparing unfused vs fused operations
   */
  async runTest(
    operationName: string,
    fusionPattern: string,
    shapeInfo: string,
    unfusedFn: (backend: WebGPUBackend) => Promise<Tensor>,
    fusedFn: (backend: WebGPUBackend) => Promise<Tensor>,
    iterations: number = 10,
    warmupIterations: number = 3,
    collectMemory: boolean = true
  ): Promise<FusionTestResult> {
    console.log(`Running operation fusion test: ${operationName} (${fusionPattern})`);
    
    // Create WebGPU backend
    const backend = new WebGPUBackend();
    
    // Warm-up phase
    for (let i = 0; i < warmupIterations; i++) {
      await unfusedFn(backend);
      await fusedFn(backend);
    }
    
    // Memory before unfused operations
    let memoryBeforeUnfused = 0;
    if (collectMemory) {
      // Reset memory tracking
      backend.resetMemoryTracking();
      memoryBeforeUnfused = backend.getEstimatedMemoryUsage();
    }
    
    // Test unfused operations
    let unfusedTime = 0;
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await unfusedFn(backend);
      unfusedTime += performance.now() - start;
    }
    unfusedTime /= iterations;
    
    // Memory after unfused operations
    let memoryAfterUnfused = 0;
    if (collectMemory) {
      memoryAfterUnfused = backend.getEstimatedMemoryUsage();
    }
    
    // Memory consumption for unfused operations
    const unfusedMemory = memoryAfterUnfused - memoryBeforeUnfused;
    
    // Reset backend for fused operations
    await backend.resetBackend();
    
    // Memory before fused operations
    let memoryBeforeFused = 0;
    if (collectMemory) {
      // Reset memory tracking
      backend.resetMemoryTracking();
      memoryBeforeFused = backend.getEstimatedMemoryUsage();
    }
    
    // Test fused operations
    let fusedTime = 0;
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await fusedFn(backend);
      fusedTime += performance.now() - start;
    }
    fusedTime /= iterations;
    
    // Memory after fused operations
    let memoryAfterFused = 0;
    if (collectMemory) {
      memoryAfterFused = backend.getEstimatedMemoryUsage();
    }
    
    // Memory consumption for fused operations
    const fusedMemory = memoryAfterFused - memoryBeforeFused;
    
    // Calculate speedup and memory savings
    const speedup = unfusedTime / fusedTime;
    const memorySavings = collectMemory ? 
      (unfusedMemory - fusedMemory) / unfusedMemory : 0;
    
    // Create and store result
    const result: FusionTestResult = {
      operationName,
      fusionPattern,
      shapeInfo,
      unfusedTime,
      fusedTime,
      speedup,
      memorySavings,
      browser: this.browser
    };
    
    this.results.push(result);
    console.log(`Operation fusion test completed: ${operationName}`);
    console.log(`  Unfused: ${unfusedTime.toFixed(2)}ms`);
    console.log(`  Fused: ${fusedTime.toFixed(2)}ms (${speedup.toFixed(2)}x speedup)`);
    if (collectMemory) {
      console.log(`  Memory savings: ${(memorySavings * 100).toFixed(2)}%`);
    }
    
    return result;
  }
  
  /**
   * Returns all test results
   */
  getResults(): FusionTestResult[] {
    return this.results;
  }
  
  /**
   * Generates an HTML report of test results
   */
  generateHTMLReport(): string {
    let html = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>WebGPU Operation Fusion Test Results</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; }
          table { border-collapse: collapse; width: 100%; }
          th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
          th { background-color: #f2f2f2; }
          tr:nth-child(even) { background-color: #f9f9f9; }
          .significant-speedup { color: green; font-weight: bold; }
          .moderate-speedup { color: blue; }
          .minimal-speedup { color: black; }
          .fusion-pattern { font-family: monospace; background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; }
          .chart-container { margin-top: 30px; height: 400px; }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      </head>
      <body>
        <h1>WebGPU Operation Fusion Test Results</h1>
        <h2>Environment Information</h2>
        <p>Browser: ${this.browser}</p>
        
        <h2>Results Summary</h2>
        <table>
          <tr>
            <th>Operation</th>
            <th>Fusion Pattern</th>
            <th>Shape Info</th>
            <th>Unfused Time (ms)</th>
            <th>Fused Time (ms)</th>
            <th>Speedup</th>
            <th>Memory Savings</th>
          </tr>
    `;
    
    for (const result of this.results) {
      const speedupClass = result.speedup > 1.5 ? 'significant-speedup' : 
                          (result.speedup > 1.2 ? 'moderate-speedup' : 'minimal-speedup');
      
      html += `
        <tr>
          <td>${result.operationName}</td>
          <td class="fusion-pattern">${result.fusionPattern}</td>
          <td>${result.shapeInfo}</td>
          <td>${result.unfusedTime.toFixed(2)}</td>
          <td>${result.fusedTime.toFixed(2)}</td>
          <td class="${speedupClass}">${result.speedup.toFixed(2)}x</td>
          <td>${(result.memorySavings * 100).toFixed(2)}%</td>
        </tr>
      `;
    }
    
    html += `
        </table>
        
        <div class="chart-container">
          <canvas id="fusionChart"></canvas>
        </div>
        
        <script>
          const ctx = document.getElementById('fusionChart').getContext('2d');
          new Chart(ctx, {
            type: 'bar',
            data: {
              labels: ${JSON.stringify(this.results.map(r => `${r.operationName} (${r.fusionPattern})` ))},
              datasets: [
                {
                  label: 'Unfused Operations',
                  data: ${JSON.stringify(this.results.map(r => r.unfusedTime))},
                  backgroundColor: 'rgba(54, 162, 235, 0.6)',
                  borderColor: 'rgb(54, 162, 235)',
                  borderWidth: 1
                },
                {
                  label: 'Fused Operations',
                  data: ${JSON.stringify(this.results.map(r => r.fusedTime))},
                  backgroundColor: 'rgba(75, 192, 75, 0.6)',
                  borderColor: 'rgb(75, 192, 75)',
                  borderWidth: 1
                }
              ]
            },
            options: {
              scales: {
                y: {
                  beginAtZero: true,
                  title: {
                    display: true,
                    text: 'Time (ms)'
                  }
                },
                x: {
                  title: {
                    display: true,
                    text: 'Operation'
                  }
                }
              }
            }
          });
        </script>
        
        <div class="chart-container">
          <canvas id="speedupChart"></canvas>
        </div>
        
        <script>
          const ctxSpeedup = document.getElementById('speedupChart').getContext('2d');
          new Chart(ctxSpeedup, {
            type: 'bar',
            data: {
              labels: ${JSON.stringify(this.results.map(r => r.operationName))},
              datasets: [
                {
                  label: 'Speedup Factor',
                  data: ${JSON.stringify(this.results.map(r => r.speedup))},
                  backgroundColor: ${JSON.stringify(this.results.map(r => 
                    r.speedup > 1.5 ? 'rgba(75, 192, 75, 0.6)' : 
                    (r.speedup > 1.2 ? 'rgba(54, 162, 235, 0.6)' : 'rgba(201, 203, 207, 0.6)')
                  ))},
                  borderColor: ${JSON.stringify(this.results.map(r => 
                    r.speedup > 1.5 ? 'rgb(75, 192, 75)' : 
                    (r.speedup > 1.2 ? 'rgb(54, 162, 235)' : 'rgb(201, 203, 207)')
                  ))},
                  borderWidth: 1
                },
                {
                  label: 'Memory Savings (%)',
                  data: ${JSON.stringify(this.results.map(r => r.memorySavings * 100))},
                  backgroundColor: 'rgba(255, 159, 64, 0.6)',
                  borderColor: 'rgb(255, 159, 64)',
                  borderWidth: 1
                }
              ]
            },
            options: {
              scales: {
                y: {
                  beginAtZero: true,
                  title: {
                    display: true,
                    text: 'Factor / Percent'
                  }
                },
                x: {
                  title: {
                    display: true,
                    text: 'Operation'
                  }
                }
              }
            }
          });
        </script>
      </body>
      </html>
    `;
    
    return html;
  }
}

/**
 * Helper function to create random tensors
 */
function createRandomTensor(shape: number[], dataType: DataType = DataType.FLOAT32): Tensor {
  const dataSize = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(dataSize);
  for (let i = 0; i < dataSize; i++) {
    data[i] = Math.random() * 2 - 1; // Values between -1 and 1
  }
  return new Tensor(data, { shape, dataType });
}

/**
 * Test operation fusion patterns
 */
describe('WebGPU Operation Fusion Tests', () => {
  let tester: OperationFusionTester;
  
  beforeEach(() => {
    tester = new OperationFusionTester();
  });
  
  test('Linear + ReLU fusion', async () => {
    // Different input shapes to test
    const testCases = [
      { batchSize: 32, inputSize: 768, outputSize: 3072 },
      { batchSize: 16, inputSize: 512, outputSize: 2048 },
      { batchSize: 8, inputSize: 1024, outputSize: 4096 },
      { batchSize: 64, inputSize: 256, outputSize: 1024 }
    ];
    
    for (const { batchSize, inputSize, outputSize } of testCases) {
      // Create input tensor
      const input = createRandomTensor([batchSize, inputSize]);
      
      // Create weights and bias
      const weights = createRandomTensor([inputSize, outputSize]);
      const bias = createRandomTensor([outputSize]);
      
      // Create linear layer
      const linearLayer = new LinearLayer(inputSize, outputSize);
      linearLayer.setWeights(weights);
      linearLayer.setBias(bias);
      
      await tester.runTest(
        `Linear+ReLU (${batchSize}x${inputSize} → ${outputSize})`,
        'LinearActivation',
        `Input: ${batchSize}x${inputSize}, Output: ${outputSize}`,
        async (backend) => {
          // Unfused operations - separate linear and ReLU
          backend.getOptimizer().setEnableOptimizations(false); // Disable fusion
          const linearOutput = await linearLayer.forward(input, backend);
          const reluOutput = await backend.relu(linearOutput);
          return reluOutput;
        },
        async (backend) => {
          // Fused operations - should use LinearActivation fusion pattern
          backend.getOptimizer().setEnableOptimizations(true);
          backend.getOptimizer().setEnableOperationFusion(true);
          const linearOutput = await linearLayer.forward(input, backend);
          const reluOutput = await backend.relu(linearOutput);
          return reluOutput;
        },
        5, // Fewer iterations for large matrices
        2  // Fewer warmup iterations
      );
    }
    
    expect(tester.getResults().length).toBe(testCases.length);
  });
  
  test('LayerNorm + GELU fusion', async () => {
    // Different input shapes to test
    const testCases = [
      { batchSize: 32, seqLength: 64, hiddenSize: 768 },
      { batchSize: 16, seqLength: 128, hiddenSize: 512 },
      { batchSize: 8, seqLength: 256, hiddenSize: 1024 }
    ];
    
    for (const { batchSize, seqLength, hiddenSize } of testCases) {
      // Create input tensor
      const input = createRandomTensor([batchSize, seqLength, hiddenSize]);
      
      // Create layer normalization parameters
      const gamma = createRandomTensor([hiddenSize]);
      const beta = createRandomTensor([hiddenSize]);
      
      // Set all gamma values to 1 and beta to 0 for a standard layer norm
      const gammaData = new Float32Array(hiddenSize).fill(1.0);
      const betaData = new Float32Array(hiddenSize).fill(0.0);
      gamma.setData(gammaData);
      beta.setData(betaData);
      
      // Create layer normalization
      const layerNorm = new LayerNormalization(hiddenSize);
      layerNorm.setGamma(gamma);
      layerNorm.setBeta(beta);
      
      await tester.runTest(
        `LayerNorm+GELU (${batchSize}x${seqLength}x${hiddenSize})`,
        'NormActivation',
        `Batch: ${batchSize}, Seq: ${seqLength}, Hidden: ${hiddenSize}`,
        async (backend) => {
          // Unfused operations
          backend.getOptimizer().setEnableOptimizations(false); // Disable fusion
          const normOutput = await layerNorm.forward(input, backend);
          const geluOutput = await backend.gelu(normOutput);
          return geluOutput;
        },
        async (backend) => {
          // Fused operations - should use NormActivation fusion pattern
          backend.getOptimizer().setEnableOptimizations(true);
          backend.getOptimizer().setEnableOperationFusion(true);
          const normOutput = await layerNorm.forward(input, backend);
          const geluOutput = await backend.gelu(normOutput);
          return geluOutput;
        },
        5,
        2
      );
    }
    
    expect(tester.getResults().length).toBe(testCases.length);
  });
  
  test('Add + Activation fusion (ElementWiseActivation)', async () => {
    // Different activation functions to test
    const activations = [
      { name: 'ReLU', fn: (backend: WebGPUBackend, tensor: Tensor) => backend.relu(tensor) },
      { name: 'Tanh', fn: (backend: WebGPUBackend, tensor: Tensor) => backend.tanh(tensor) },
      { name: 'Sigmoid', fn: (backend: WebGPUBackend, tensor: Tensor) => backend.sigmoid(tensor) }
    ];
    
    // Different shapes to test
    const shapes = [
      [32, 768],   // Large vector
      [16, 32, 32] // Small 3D tensor
    ];
    
    for (const shape of shapes) {
      for (const activation of activations) {
        // Create input tensors
        const tensorA = createRandomTensor(shape);
        const tensorB = createRandomTensor(shape);
        
        await tester.runTest(
          `Add+${activation.name} (${shape.join('x')})`,
          'ElementWiseActivation',
          `Shape: ${shape.join('x')}`,
          async (backend) => {
            // Unfused operations
            backend.getOptimizer().setEnableOptimizations(false);
            const addOutput = await backend.add(tensorA, tensorB);
            const activationOutput = await activation.fn(backend, addOutput);
            return activationOutput;
          },
          async (backend) => {
            // Fused operations
            backend.getOptimizer().setEnableOptimizations(true);
            backend.getOptimizer().setEnableOperationFusion(true);
            const addOutput = await backend.add(tensorA, tensorB);
            const activationOutput = await activation.fn(backend, addOutput);
            return activationOutput;
          },
          10,
          3
        );
      }
    }
    
    expect(tester.getResults().length).toBe(shapes.length * activations.length);
  });
  
  test('Multiple element-wise operations fusion (ElementWiseChain)', async () => {
    // Different shapes to test
    const shapes = [
      [32, 768],   // Large vector
      [16, 16, 16] // Small 3D tensor
    ];
    
    for (const shape of shapes) {
      // Create input tensors
      const tensorA = createRandomTensor(shape);
      const tensorB = createRandomTensor(shape);
      const tensorC = createRandomTensor(shape);
      
      await tester.runTest(
        `Mul+Add+Tanh (${shape.join('x')})`,
        'ElementWiseChain',
        `Shape: ${shape.join('x')}`,
        async (backend) => {
          // Unfused operations - 3 separate element-wise operations
          backend.getOptimizer().setEnableOptimizations(false);
          const mulOutput = await backend.multiply(tensorA, tensorB);
          const addOutput = await backend.add(mulOutput, tensorC);
          const tanhOutput = await backend.tanh(addOutput);
          return tanhOutput;
        },
        async (backend) => {
          // Fused operations - should use ElementWiseChain fusion pattern
          backend.getOptimizer().setEnableOptimizations(true);
          backend.getOptimizer().setEnableOperationFusion(true);
          const mulOutput = await backend.multiply(tensorA, tensorB);
          const addOutput = await backend.add(mulOutput, tensorC);
          const tanhOutput = await backend.tanh(addOutput);
          return tanhOutput;
        },
        10,
        3
      );
    }
    
    expect(tester.getResults().length).toBe(shapes.length);
  });
  
  test('Self-attention fusion patterns (AttentionPattern)', async () => {
    // Define test cases
    const testCases = [
      { batchSize: 4, seqLength: 128, embedSize: 768, numHeads: 12 },
      { batchSize: 2, seqLength: 256, embedSize: 512, numHeads: 8 }
    ];
    
    for (const { batchSize, seqLength, embedSize, numHeads } of testCases) {
      // Create input tensor
      const input = createRandomTensor([batchSize, seqLength, embedSize]);
      
      // Create attention layer
      const headSize = embedSize / numHeads;
      const attention = new MultiHeadAttention(embedSize, numHeads);
      
      // Create QKV weights
      const qWeight = createRandomTensor([embedSize, embedSize]);
      const kWeight = createRandomTensor([embedSize, embedSize]);
      const vWeight = createRandomTensor([embedSize, embedSize]);
      const outWeight = createRandomTensor([embedSize, embedSize]);
      
      // Set weights
      attention.setQKVWeights(qWeight, kWeight, vWeight);
      attention.setOutputWeight(outWeight);
      
      await tester.runTest(
        `Self-Attention (${batchSize}x${seqLength}x${embedSize})`,
        'AttentionPattern',
        `Batch: ${batchSize}, Seq: ${seqLength}, Embed: ${embedSize}, Heads: ${numHeads}`,
        async (backend) => {
          // Unfused attention operations
          backend.getOptimizer().setEnableOptimizations(false);
          backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(false);
          return await attention.forward(input, input, input, null, backend);
        },
        async (backend) => {
          // Fused attention operations - should detect the attention pattern
          backend.getOptimizer().setEnableOptimizations(true);
          backend.getOptimizer().setEnableOperationFusion(true);
          backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(true);
          return await attention.forward(input, input, input, null, backend);
        },
        3, // Fewer iterations for complex operations
        1  // Fewer warmup iterations
      );
    }
    
    expect(tester.getResults().length).toBe(testCases.length);
  });
  
  test('Transformer FFN fusion pattern (FFNPattern)', async () => {
    // Define test cases
    const testCases = [
      { batchSize: 8, seqLength: 128, embedSize: 768, ffnSize: 3072 },
      { batchSize: 4, seqLength: 256, embedSize: 512, ffnSize: 2048 }
    ];
    
    for (const { batchSize, seqLength, embedSize, ffnSize } of testCases) {
      // Create input tensor
      const input = createRandomTensor([batchSize, seqLength, embedSize]);
      
      // Create FFN weights
      const fc1Weight = createRandomTensor([embedSize, ffnSize]);
      const fc1Bias = createRandomTensor([ffnSize]);
      const fc2Weight = createRandomTensor([ffnSize, embedSize]);
      const fc2Bias = createRandomTensor([embedSize]);
      
      // Create linear layers
      const linear1 = new LinearLayer(embedSize, ffnSize);
      linear1.setWeights(fc1Weight);
      linear1.setBias(fc1Bias);
      
      const linear2 = new LinearLayer(ffnSize, embedSize);
      linear2.setWeights(fc2Weight);
      linear2.setBias(fc2Bias);
      
      await tester.runTest(
        `Transformer FFN (${batchSize}x${seqLength}x${embedSize} → ${ffnSize})`,
        'FFNPattern',
        `Batch: ${batchSize}, Seq: ${seqLength}, Embed: ${embedSize}, FFN: ${ffnSize}`,
        async (backend) => {
          // Unfused FFN operations
          backend.getOptimizer().setEnableOptimizations(false);
          backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(false);
          
          // Standard FFN pattern: Linear → GELU → Linear
          const hidden = await linear1.forward(input, backend);
          const activated = await backend.gelu(hidden);
          const output = await linear2.forward(activated, backend);
          return output;
        },
        async (backend) => {
          // Fused FFN operations - should detect FFN pattern
          backend.getOptimizer().setEnableOptimizations(true);
          backend.getOptimizer().setEnableOperationFusion(true);
          backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(true);
          
          // The optimizer should recognize this pattern and apply fusion
          const hidden = await linear1.forward(input, backend);
          const activated = await backend.gelu(hidden);
          const output = await linear2.forward(activated, backend);
          return output;
        },
        3,
        1
      );
    }
    
    expect(tester.getResults().length).toBe(testCases.length);
  });
});

/**
 * Generate HTML report after all tests
 */
afterAll(() => {
  const tester = new OperationFusionTester();
  const htmlReport = tester.generateHTMLReport();
  
  // In a real environment, we would save this to a file
  console.log('All operation fusion tests completed');
  console.log(`Generated HTML report (${htmlReport.length} bytes)`);
});