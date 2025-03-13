/**
 * WebGPU Neural Network Pattern Recognition Tests
 * 
 * This file contains tests specifically focused on measuring the impact of
 * neural network pattern recognition optimizations in the WebGPU backend.
 */

import { expect } from '@jest/globals';
import { performance } from 'perf_hooks';

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
import { TransformerEncoderLayer } from '../../../src/model/transformers/layers/transformer';
import { TransformerDecoderLayer } from '../../../src/model/transformers/layers/transformer';

/**
 * Interface for neural network pattern recognition test results
 */
interface PatternRecognitionTestResult {
  networkName: string;
  patternName: string;
  networkConfig: string;
  standardTime: number;
  optimizedTime: number;
  speedup: number;
  memoryStandard?: number;
  memoryOptimized?: number;
  memorySavings?: number;
  patternsDetected: string[];
  browser: string;
}

/**
 * Helper class for running neural network pattern recognition tests
 */
class NeuralNetworkPatternTester {
  private results: PatternRecognitionTestResult[] = [];
  private browser: BrowserType;
  
  constructor() {
    this.browser = BrowserDetector.detectBrowser();
  }
  
  /**
   * Runs a test comparing standard vs pattern-optimized neural network execution
   */
  async runTest(
    networkName: string,
    patternName: string,
    networkConfig: string,
    standardFn: (backend: WebGPUBackend) => Promise<Tensor>,
    optimizedFn: (backend: WebGPUBackend) => Promise<Tensor>,
    iterations: number = 5,
    warmupIterations: number = 2,
    collectMemory: boolean = true
  ): Promise<PatternRecognitionTestResult> {
    console.log(`Running neural network pattern test: ${networkName} (${patternName})`);
    
    // Create WebGPU backend
    const backend = new WebGPUBackend();
    
    // Warm-up phase
    for (let i = 0; i < warmupIterations; i++) {
      // Allow backend to reset between iterations
      await backend.resetBackend();
      await standardFn(backend);
      
      await backend.resetBackend();
      await optimizedFn(backend);
    }
    
    // Memory before standard execution
    let memoryBeforeStandard = 0;
    if (collectMemory) {
      await backend.resetBackend();
      backend.resetMemoryTracking();
      memoryBeforeStandard = backend.getEstimatedMemoryUsage();
    }
    
    // Test standard execution
    let standardTime = 0;
    for (let i = 0; i < iterations; i++) {
      if (i > 0) await backend.resetBackend(); // Reset between iterations but keep first for memory tracking
      
      const start = performance.now();
      await standardFn(backend);
      standardTime += performance.now() - start;
    }
    standardTime /= iterations;
    
    // Memory after standard execution
    let memoryAfterStandard = 0;
    if (collectMemory) {
      memoryAfterStandard = backend.getEstimatedMemoryUsage();
    }
    
    // Memory consumption for standard execution
    const memoryStandard = collectMemory ? (memoryAfterStandard - memoryBeforeStandard) : undefined;
    
    // Reset backend before optimized execution
    await backend.resetBackend();
    
    // Memory before optimized execution
    let memoryBeforeOptimized = 0;
    if (collectMemory) {
      backend.resetMemoryTracking();
      memoryBeforeOptimized = backend.getEstimatedMemoryUsage();
    }
    
    // Reset detected patterns
    backend.getOptimizer().resetDetectedPatterns();
    
    // Test optimized execution
    let optimizedTime = 0;
    for (let i = 0; i < iterations; i++) {
      if (i > 0) await backend.resetBackend(); // Reset between iterations but keep first for pattern detection
      
      const start = performance.now();
      await optimizedFn(backend);
      optimizedTime += performance.now() - start;
    }
    optimizedTime /= iterations;
    
    // Memory after optimized execution
    let memoryAfterOptimized = 0;
    if (collectMemory) {
      memoryAfterOptimized = backend.getEstimatedMemoryUsage();
    }
    
    // Memory consumption for optimized execution
    const memoryOptimized = collectMemory ? (memoryAfterOptimized - memoryBeforeOptimized) : undefined;
    
    // Calculate speedup and memory savings
    const speedup = standardTime / optimizedTime;
    const memorySavings = (memoryStandard !== undefined && memoryOptimized !== undefined) ? 
      (memoryStandard - memoryOptimized) / memoryStandard : undefined;
    
    // Get detected patterns
    const patternsDetected = backend.getOptimizer().getDetectedPatterns() || [];
    
    // Create and store result
    const result: PatternRecognitionTestResult = {
      networkName,
      patternName,
      networkConfig,
      standardTime,
      optimizedTime,
      speedup,
      memoryStandard,
      memoryOptimized,
      memorySavings,
      patternsDetected,
      browser: this.browser
    };
    
    this.results.push(result);
    console.log(`Neural network pattern test completed: ${networkName}`);
    console.log(`  Standard execution: ${standardTime.toFixed(2)}ms`);
    console.log(`  Optimized execution: ${optimizedTime.toFixed(2)}ms (${speedup.toFixed(2)}x speedup)`);
    if (collectMemory && memorySavings !== undefined) {
      console.log(`  Memory savings: ${(memorySavings * 100).toFixed(2)}%`);
    }
    console.log(`  Detected patterns: ${patternsDetected.length > 0 ? patternsDetected.join(', ') : 'None'}`);
    
    return result;
  }
  
  /**
   * Returns all test results
   */
  getResults(): PatternRecognitionTestResult[] {
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
        <title>WebGPU Neural Network Pattern Recognition Test Results</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; }
          table { border-collapse: collapse; width: 100%; }
          th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
          th { background-color: #f2f2f2; }
          tr:nth-child(even) { background-color: #f9f9f9; }
          .significant-speedup { color: green; font-weight: bold; }
          .moderate-speedup { color: blue; }
          .minimal-speedup { color: black; }
          .pattern-list { font-family: monospace; background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; }
          .chart-container { margin-top: 30px; height: 400px; }
          .network-config { font-size: 0.9em; color: #555; }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      </head>
      <body>
        <h1>WebGPU Neural Network Pattern Recognition Test Results</h1>
        <h2>Environment Information</h2>
        <p>Browser: ${this.browser}</p>
        
        <h2>Results Summary</h2>
        <table>
          <tr>
            <th>Neural Network</th>
            <th>Pattern</th>
            <th>Configuration</th>
            <th>Standard Time (ms)</th>
            <th>Optimized Time (ms)</th>
            <th>Speedup</th>
            <th>Memory Savings</th>
            <th>Detected Patterns</th>
          </tr>
    `;
    
    for (const result of this.results) {
      const speedupClass = result.speedup > 1.5 ? 'significant-speedup' : 
                          (result.speedup > 1.2 ? 'moderate-speedup' : 'minimal-speedup');
      
      html += `
        <tr>
          <td>${result.networkName}</td>
          <td>${result.patternName}</td>
          <td class="network-config">${result.networkConfig}</td>
          <td>${result.standardTime.toFixed(2)}</td>
          <td>${result.optimizedTime.toFixed(2)}</td>
          <td class="${speedupClass}">${result.speedup.toFixed(2)}x</td>
          <td>${result.memorySavings !== undefined ? (result.memorySavings * 100).toFixed(2) + '%' : 'N/A'}</td>
          <td class="pattern-list">${result.patternsDetected.join('<br>') || 'None'}</td>
        </tr>
      `;
    }
    
    html += `
        </table>
        
        <div class="chart-container">
          <canvas id="speedupChart"></canvas>
        </div>
        
        <script>
          const ctx = document.getElementById('speedupChart').getContext('2d');
          new Chart(ctx, {
            type: 'bar',
            data: {
              labels: ${JSON.stringify(this.results.map(r => `${r.networkName} (${r.patternName})`))},
              datasets: [
                {
                  label: 'Standard Execution',
                  data: ${JSON.stringify(this.results.map(r => r.standardTime))},
                  backgroundColor: 'rgba(54, 162, 235, 0.6)',
                  borderColor: 'rgb(54, 162, 235)',
                  borderWidth: 1
                },
                {
                  label: 'Optimized Execution',
                  data: ${JSON.stringify(this.results.map(r => r.optimizedTime))},
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
                    text: 'Neural Network'
                  }
                }
              }
            }
          });
        </script>
        
        <div class="chart-container">
          <canvas id="speedupFactorChart"></canvas>
        </div>
        
        <script>
          const ctxSpeedup = document.getElementById('speedupFactorChart').getContext('2d');
          new Chart(ctxSpeedup, {
            type: 'bar',
            data: {
              labels: ${JSON.stringify(this.results.map(r => r.networkName))},
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
                  data: ${JSON.stringify(this.results.map(r => r.memorySavings !== undefined ? r.memorySavings * 100 : 0))},
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
                    text: 'Neural Network'
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
 * Test neural network pattern recognition
 */
describe('WebGPU Neural Network Pattern Recognition Tests', () => {
  let tester: NeuralNetworkPatternTester;
  
  beforeEach(() => {
    tester = new NeuralNetworkPatternTester();
  });
  
  test('Transformer Encoder Layer Pattern', async () => {
    // Test scenarios with different configurations
    const testCases = [
      { batchSize: 4, seqLength: 128, hiddenSize: 768, ffnSize: 3072, numHeads: 12 },
      { batchSize: 2, seqLength: 256, hiddenSize: 512, ffnSize: 2048, numHeads: 8 }
    ];
    
    for (const { batchSize, seqLength, hiddenSize, ffnSize, numHeads } of testCases) {
      // Create input tensor
      const input = createRandomTensor([batchSize, seqLength, hiddenSize]);
      
      // Create attention mask tensor
      const attentionMask = createRandomTensor([batchSize, 1, seqLength, seqLength]);
      
      // Create transformer encoder layer
      const encoderLayer = new TransformerEncoderLayer(
        hiddenSize,
        numHeads,
        ffnSize,
        0.1 // dropout rate
      );
      
      // Initialize parameters
      encoderLayer.initialize();
      
      await tester.runTest(
        'Transformer Encoder',
        'EncoderLayerPattern',
        `Batch: ${batchSize}, Seq: ${seqLength}, Hidden: ${hiddenSize}, FFN: ${ffnSize}, Heads: ${numHeads}`,
        async (backend) => {
          // Standard execution - neural network pattern recognition disabled
          backend.getOptimizer().setEnableOptimizations(false);
          backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(false);
          return await encoderLayer.forward(input, attentionMask, backend);
        },
        async (backend) => {
          // Optimized execution - neural network pattern recognition enabled
          backend.getOptimizer().setEnableOptimizations(true);
          backend.getOptimizer().setEnableOperationFusion(true);
          backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(true);
          return await encoderLayer.forward(input, attentionMask, backend);
        },
        3, // Fewer iterations for complex operations
        1  // Fewer warmup iterations
      );
    }
    
    expect(tester.getResults().length).toBe(testCases.length);
  });
  
  test('Transformer Decoder Layer Pattern', async () => {
    // Test scenarios with different configurations
    const testCases = [
      { batchSize: 4, seqLength: 64, hiddenSize: 768, ffnSize: 3072, numHeads: 12 }
    ];
    
    for (const { batchSize, seqLength, hiddenSize, ffnSize, numHeads } of testCases) {
      // Create input tensors
      const input = createRandomTensor([batchSize, seqLength, hiddenSize]);
      const encoderOutput = createRandomTensor([batchSize, seqLength * 2, hiddenSize]); // Encoder output typically has longer sequence length
      
      // Create attention masks
      const selfAttentionMask = createRandomTensor([batchSize, 1, seqLength, seqLength]);
      const crossAttentionMask = createRandomTensor([batchSize, 1, seqLength, seqLength * 2]);
      
      // Create transformer decoder layer
      const decoderLayer = new TransformerDecoderLayer(
        hiddenSize,
        numHeads,
        ffnSize,
        0.1 // dropout rate
      );
      
      // Initialize parameters
      decoderLayer.initialize();
      
      await tester.runTest(
        'Transformer Decoder',
        'DecoderLayerPattern',
        `Batch: ${batchSize}, Seq: ${seqLength}, Hidden: ${hiddenSize}, FFN: ${ffnSize}, Heads: ${numHeads}`,
        async (backend) => {
          // Standard execution - neural network pattern recognition disabled
          backend.getOptimizer().setEnableOptimizations(false);
          backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(false);
          return await decoderLayer.forward(input, encoderOutput, selfAttentionMask, crossAttentionMask, backend);
        },
        async (backend) => {
          // Optimized execution - neural network pattern recognition enabled
          backend.getOptimizer().setEnableOptimizations(true);
          backend.getOptimizer().setEnableOperationFusion(true);
          backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(true);
          return await decoderLayer.forward(input, encoderOutput, selfAttentionMask, crossAttentionMask, backend);
        },
        3, // Fewer iterations for complex operations
        1  // Fewer warmup iterations
      );
    }
    
    expect(tester.getResults().length).toBe(testCases.length);
  });
  
  test('Multi-Head Attention Pattern', async () => {
    // Test scenarios with different configurations
    const testCases = [
      { batchSize: 8, seqLength: 128, embedSize: 768, numHeads: 12 },
      { batchSize: 4, seqLength: 256, embedSize: 512, numHeads: 8 }
    ];
    
    for (const { batchSize, seqLength, embedSize, numHeads } of testCases) {
      // Create input tensor
      const input = createRandomTensor([batchSize, seqLength, embedSize]);
      
      // Create attention layer
      const attention = new MultiHeadAttention(embedSize, numHeads);
      
      // Create and set weights
      const qWeight = createRandomTensor([embedSize, embedSize]);
      const kWeight = createRandomTensor([embedSize, embedSize]);
      const vWeight = createRandomTensor([embedSize, embedSize]);
      const outWeight = createRandomTensor([embedSize, embedSize]);
      
      attention.setQKVWeights(qWeight, kWeight, vWeight);
      attention.setOutputWeight(outWeight);
      
      await tester.runTest(
        'Multi-Head Attention',
        'AttentionPattern',
        `Batch: ${batchSize}, Seq: ${seqLength}, Embed: ${embedSize}, Heads: ${numHeads}`,
        async (backend) => {
          // Standard execution - neural network pattern recognition disabled
          backend.getOptimizer().setEnableOptimizations(false);
          backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(false);
          return await attention.forward(input, input, input, null, backend);
        },
        async (backend) => {
          // Optimized execution - neural network pattern recognition enabled
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
  
  test('Feed-Forward Network Pattern', async () => {
    // Test scenarios with different configurations
    const testCases = [
      { batchSize: 16, seqLength: 128, hiddenSize: 768, ffnSize: 3072 },
      { batchSize: 8, seqLength: 256, hiddenSize: 512, ffnSize: 2048 }
    ];
    
    for (const { batchSize, seqLength, hiddenSize, ffnSize } of testCases) {
      // Create input tensor
      const input = createRandomTensor([batchSize, seqLength, hiddenSize]);
      
      // Create FFN weights
      const fc1Weight = createRandomTensor([hiddenSize, ffnSize]);
      const fc1Bias = createRandomTensor([ffnSize]);
      const fc2Weight = createRandomTensor([ffnSize, hiddenSize]);
      const fc2Bias = createRandomTensor([hiddenSize]);
      
      // Create linear layers
      const linear1 = new LinearLayer(hiddenSize, ffnSize);
      linear1.setWeights(fc1Weight);
      linear1.setBias(fc1Bias);
      
      const linear2 = new LinearLayer(ffnSize, hiddenSize);
      linear2.setWeights(fc2Weight);
      linear2.setBias(fc2Bias);
      
      await tester.runTest(
        'Feed-Forward Network',
        'FFNPattern',
        `Batch: ${batchSize}, Seq: ${seqLength}, Hidden: ${hiddenSize}, FFN: ${ffnSize}`,
        async (backend) => {
          // Standard execution - neural network pattern recognition disabled
          backend.getOptimizer().setEnableOptimizations(false);
          backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(false);
          
          // Standard FFN pattern: Linear → GELU → Linear
          const hidden = await linear1.forward(input, backend);
          const activated = await backend.gelu(hidden);
          const output = await linear2.forward(activated, backend);
          return output;
        },
        async (backend) => {
          // Optimized execution - neural network pattern recognition enabled
          backend.getOptimizer().setEnableOptimizations(true);
          backend.getOptimizer().setEnableOperationFusion(true);
          backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(true);
          
          // The optimizer should recognize this pattern and apply optimizations
          const hidden = await linear1.forward(input, backend);
          const activated = await backend.gelu(hidden);
          const output = await linear2.forward(activated, backend);
          return output;
        },
        3, // Fewer iterations for complex operations
        1  // Fewer warmup iterations
      );
    }
    
    expect(tester.getResults().length).toBe(testCases.length);
  });
  
  test('Residual Connection Pattern', async () => {
    // Test scenarios with different configurations
    const testCases = [
      { batchSize: 32, seqLength: 128, hiddenSize: 768 },
      { batchSize: 16, seqLength: 256, hiddenSize: 512 }
    ];
    
    for (const { batchSize, seqLength, hiddenSize } of testCases) {
      // Create input tensor
      const input = createRandomTensor([batchSize, seqLength, hiddenSize]);
      
      // Create layer normalization
      const layerNorm = new LayerNormalization(hiddenSize);
      const gamma = createRandomTensor([hiddenSize]);
      const beta = createRandomTensor([hiddenSize]);
      layerNorm.setGamma(gamma);
      layerNorm.setBeta(beta);
      
      // Create a simple sub-layer (linear transformation)
      const linear = new LinearLayer(hiddenSize, hiddenSize);
      const weight = createRandomTensor([hiddenSize, hiddenSize]);
      const bias = createRandomTensor([hiddenSize]);
      linear.setWeights(weight);
      linear.setBias(bias);
      
      await tester.runTest(
        'Residual Connection',
        'ResidualPattern',
        `Batch: ${batchSize}, Seq: ${seqLength}, Hidden: ${hiddenSize}`,
        async (backend) => {
          // Standard execution - neural network pattern recognition disabled
          backend.getOptimizer().setEnableOptimizations(false);
          backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(false);
          
          // Pre-norm residual connection pattern: LayerNorm → Sub-layer → Add
          const norm = await layerNorm.forward(input, backend);
          const sublayer = await linear.forward(norm, backend);
          const output = await backend.add(input, sublayer);
          return output;
        },
        async (backend) => {
          // Optimized execution - neural network pattern recognition enabled
          backend.getOptimizer().setEnableOptimizations(true);
          backend.getOptimizer().setEnableOperationFusion(true);
          backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(true);
          
          // The optimizer should recognize this pattern and apply optimizations
          const norm = await layerNorm.forward(input, backend);
          const sublayer = await linear.forward(norm, backend);
          const output = await backend.add(input, sublayer);
          return output;
        },
        5, // More iterations for faster operations
        2  // More warmup iterations
      );
    }
    
    expect(tester.getResults().length).toBe(testCases.length);
  });
});

/**
 * Generate HTML report after all tests
 */
afterAll(() => {
  const tester = new NeuralNetworkPatternTester();
  const htmlReport = tester.generateHTMLReport();
  
  // In a real environment, we would save this to a file
  console.log('All neural network pattern recognition tests completed');
  console.log(`Generated HTML report (${htmlReport.length} bytes)`);
});