/**
 * WebGPU Optimizer Benchmarking Suite
 * 
 * This file contains comprehensive benchmarks for the WebGPU optimization features,
 * including operation fusion, memory layout optimization, browser-specific optimizations,
 * and neural network pattern recognition.
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

// Import neural network operations for testing
import { LinearLayer } from '../../../src/model/transformers/layers/linear';
import { Activation } from '../../../src/model/transformers/layers/activation';
import { MultiHeadAttention } from '../../../src/model/transformers/layers/attention';
import { LayerNormalization } from '../../../src/model/transformers/layers/normalization';

/**
 * BenchmarkResult interface for storing benchmark results
 */
interface BenchmarkResult {
  name: string;
  optimizedTime: number;
  standardTime: number;
  speedup: number;
  memoryOptimized?: number;
  memoryStandard?: number;
  memorySavings?: number;
  browser: string;
  hardwareInfo: any;
}

/**
 * Helper class for running benchmarks
 */
class BenchmarkRunner {
  private results: BenchmarkResult[] = [];
  private backend: WebGPUBackend;
  private optimizer: WebGPUOptimizer;
  private browser: BrowserType;
  private hardwareInfo: any;
  
  constructor() {
    this.backend = new WebGPUBackend();
    this.optimizer = this.backend.getOptimizer();
    this.browser = BrowserDetector.detectBrowser();
    this.hardwareInfo = this.getHardwareInfo();
  }
  
  /**
   * Gathers hardware information for result context
   */
  private getHardwareInfo(): any {
    // In a real browser environment, we would get this information
    // For now, we'll return mock data
    return {
      gpu: 'Mock GPU',
      browser: this.browser,
      webGPUVersion: '1.0.0',
      osName: 'Mock OS',
      osVersion: '1.0'
    };
  }
  
  /**
   * Runs a benchmark comparing optimized vs standard implementation
   */
  async runBenchmark(
    name: string,
    optimizedFn: () => Promise<any>,
    standardFn: () => Promise<any>,
    iterations: number = 10,
    warmupIterations: number = 3,
    collectMemory: boolean = false
  ): Promise<BenchmarkResult> {
    console.log(`Running benchmark: ${name}`);
    
    // Warm-up phase
    for (let i = 0; i < warmupIterations; i++) {
      await optimizedFn();
      await standardFn();
    }
    
    // Optimized implementation timing
    const optimizedStartTime = performance.now();
    for (let i = 0; i < iterations; i++) {
      await optimizedFn();
    }
    const optimizedEndTime = performance.now();
    const optimizedTime = (optimizedEndTime - optimizedStartTime) / iterations;
    
    // Memory measurement for optimized (if enabled)
    let memoryOptimized = undefined;
    if (collectMemory) {
      // In browser we would use performance.memory
      // For now, estimate from backend tracked allocations
      memoryOptimized = this.backend.getEstimatedMemoryUsage();
    }
    
    // Standard implementation timing
    const standardStartTime = performance.now();
    for (let i = 0; i < iterations; i++) {
      await standardFn();
    }
    const standardEndTime = performance.now();
    const standardTime = (standardEndTime - standardStartTime) / iterations;
    
    // Memory measurement for standard (if enabled)
    let memoryStandard = undefined;
    if (collectMemory) {
      // In browser we would use performance.memory
      memoryStandard = this.backend.getEstimatedMemoryUsage();
    }
    
    // Calculate speedup and memory savings
    const speedup = standardTime / optimizedTime;
    let memorySavings = undefined;
    if (memoryOptimized !== undefined && memoryStandard !== undefined) {
      memorySavings = (memoryStandard - memoryOptimized) / memoryStandard;
    }
    
    // Create and store result
    const result: BenchmarkResult = {
      name,
      optimizedTime,
      standardTime,
      speedup,
      memoryOptimized,
      memoryStandard,
      memorySavings,
      browser: this.browser,
      hardwareInfo: this.hardwareInfo
    };
    
    this.results.push(result);
    console.log(`Benchmark ${name} completed: ${speedup.toFixed(2)}x speedup`);
    return result;
  }
  
  /**
   * Returns all benchmark results
   */
  getResults(): BenchmarkResult[] {
    return this.results;
  }
  
  /**
   * Generates an HTML report of benchmark results
   */
  generateHTMLReport(): string {
    let html = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>WebGPU Optimizer Benchmark Results</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; }
          table { border-collapse: collapse; width: 100%; }
          th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
          th { background-color: #f2f2f2; }
          tr:nth-child(even) { background-color: #f9f9f9; }
          .speedup-high { color: green; font-weight: bold; }
          .speedup-medium { color: blue; }
          .speedup-low { color: black; }
          .chart-container { margin-top: 30px; height: 400px; }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      </head>
      <body>
        <h1>WebGPU Optimizer Benchmark Results</h1>
        <h2>Environment Information</h2>
        <p>Browser: ${this.browser}</p>
        <p>Hardware: ${JSON.stringify(this.hardwareInfo)}</p>
        
        <h2>Results Summary</h2>
        <table>
          <tr>
            <th>Benchmark</th>
            <th>Optimized Time (ms)</th>
            <th>Standard Time (ms)</th>
            <th>Speedup</th>
            <th>Memory Savings</th>
          </tr>
    `;
    
    for (const result of this.results) {
      const speedupClass = result.speedup > 2 ? 'speedup-high' : 
                          (result.speedup > 1.5 ? 'speedup-medium' : 'speedup-low');
      html += `
        <tr>
          <td>${result.name}</td>
          <td>${result.optimizedTime.toFixed(2)}</td>
          <td>${result.standardTime.toFixed(2)}</td>
          <td class="${speedupClass}">${result.speedup.toFixed(2)}x</td>
          <td>${result.memorySavings !== undefined ? (result.memorySavings * 100).toFixed(2) + '%' : 'N/A'}</td>
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
              labels: ${JSON.stringify(this.results.map(r => r.name))},
              datasets: [{
                label: 'Speedup Factor (higher is better)',
                data: ${JSON.stringify(this.results.map(r => r.speedup))},
                backgroundColor: ${JSON.stringify(this.results.map(r => 
                  r.speedup > 2 ? 'rgba(75, 192, 75, 0.6)' : 
                  (r.speedup > 1.5 ? 'rgba(54, 162, 235, 0.6)' : 'rgba(201, 203, 207, 0.6)')
                ))},
                borderColor: ${JSON.stringify(this.results.map(r => 
                  r.speedup > 2 ? 'rgb(75, 192, 75)' : 
                  (r.speedup > 1.5 ? 'rgb(54, 162, 235)' : 'rgb(201, 203, 207)')
                ))},
                borderWidth: 1
              }]
            },
            options: {
              scales: {
                y: {
                  beginAtZero: true,
                  title: {
                    display: true,
                    text: 'Speedup Factor (x)'
                  }
                },
                x: {
                  title: {
                    display: true,
                    text: 'Benchmark'
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
 * Benchmark tests for matrix multiplication optimization
 */
describe('WebGPU Optimizer Matrix Multiplication Benchmarks', () => {
  let benchmarkRunner: BenchmarkRunner;
  
  beforeEach(async () => {
    benchmarkRunner = new BenchmarkRunner();
  });
  
  test('Small matrix multiplication (128x128)', async () => {
    const createTensors = () => {
      const a = new Tensor(new Float32Array(128 * 128), {shape: [128, 128], dataType: DataType.FLOAT32});
      const b = new Tensor(new Float32Array(128 * 128), {shape: [128, 128], dataType: DataType.FLOAT32});
      return { a, b };
    };
    
    const { a, b } = createTensors();
    const backend = new WebGPUBackend();
    
    await benchmarkRunner.runBenchmark(
      'MatMul 128x128',
      async () => {
        // Optimized path uses memory layout optimization and workgroup optimization
        backend.getOptimizer().setEnableOptimizations(true);
        const result = await backend.matmul(a, b);
        return result;
      },
      async () => {
        // Standard path with optimizations disabled
        backend.getOptimizer().setEnableOptimizations(false);
        const result = await backend.matmul(a, b);
        return result;
      },
      10
    );
    
    // We're just benchmarking, so no explicit assertion
    expect(true).toBe(true);
  });
  
  test('Medium matrix multiplication (512x512)', async () => {
    const createTensors = () => {
      const a = new Tensor(new Float32Array(512 * 512), {shape: [512, 512], dataType: DataType.FLOAT32});
      const b = new Tensor(new Float32Array(512 * 512), {shape: [512, 512], dataType: DataType.FLOAT32});
      return { a, b };
    };
    
    const { a, b } = createTensors();
    const backend = new WebGPUBackend();
    
    await benchmarkRunner.runBenchmark(
      'MatMul 512x512',
      async () => {
        backend.getOptimizer().setEnableOptimizations(true);
        const result = await backend.matmul(a, b);
        return result;
      },
      async () => {
        backend.getOptimizer().setEnableOptimizations(false);
        const result = await backend.matmul(a, b);
        return result;
      },
      10
    );
    
    expect(true).toBe(true);
  });
  
  test('Large matrix multiplication (1024x1024)', async () => {
    const createTensors = () => {
      const a = new Tensor(new Float32Array(1024 * 1024), {shape: [1024, 1024], dataType: DataType.FLOAT32});
      const b = new Tensor(new Float32Array(1024 * 1024), {shape: [1024, 1024], dataType: DataType.FLOAT32});
      return { a, b };
    };
    
    const { a, b } = createTensors();
    const backend = new WebGPUBackend();
    
    await benchmarkRunner.runBenchmark(
      'MatMul 1024x1024',
      async () => {
        backend.getOptimizer().setEnableOptimizations(true);
        const result = await backend.matmul(a, b);
        return result;
      },
      async () => {
        backend.getOptimizer().setEnableOptimizations(false);
        const result = await backend.matmul(a, b);
        return result;
      },
      5 // Fewer iterations for large matrices
    );
    
    expect(true).toBe(true);
  });
  
  test('Batch matrix multiplication (8 x 256x256)', async () => {
    const createTensors = () => {
      const a = new Tensor(new Float32Array(8 * 256 * 256), {shape: [8, 256, 256], dataType: DataType.FLOAT32});
      const b = new Tensor(new Float32Array(8 * 256 * 256), {shape: [8, 256, 256], dataType: DataType.FLOAT32});
      return { a, b };
    };
    
    const { a, b } = createTensors();
    const backend = new WebGPUBackend();
    
    await benchmarkRunner.runBenchmark(
      'Batch MatMul 8x256x256',
      async () => {
        backend.getOptimizer().setEnableOptimizations(true);
        const result = await backend.batchMatMul(a, b);
        return result;
      },
      async () => {
        backend.getOptimizer().setEnableOptimizations(false);
        const result = await backend.batchMatMul(a, b);
        return result;
      },
      10
    );
    
    expect(true).toBe(true);
  });
});

/**
 * Benchmark tests for operation fusion optimization
 */
describe('WebGPU Optimizer Operation Fusion Benchmarks', () => {
  let benchmarkRunner: BenchmarkRunner;
  
  beforeEach(async () => {
    benchmarkRunner = new BenchmarkRunner();
  });
  
  test('Linear + ReLU fusion (batch=32, in=768, out=3072)', async () => {
    const batchSize = 32;
    const inputSize = 768;
    const outputSize = 3072;
    
    const createTensors = () => {
      const input = new Tensor(new Float32Array(batchSize * inputSize), {shape: [batchSize, inputSize], dataType: DataType.FLOAT32});
      const weights = new Tensor(new Float32Array(inputSize * outputSize), {shape: [inputSize, outputSize], dataType: DataType.FLOAT32});
      const bias = new Tensor(new Float32Array(outputSize), {shape: [outputSize], dataType: DataType.FLOAT32});
      return { input, weights, bias };
    };
    
    const { input, weights, bias } = createTensors();
    const backend = new WebGPUBackend();
    const linearLayer = new LinearLayer(inputSize, outputSize);
    linearLayer.setWeights(weights);
    linearLayer.setBias(bias);
    
    await benchmarkRunner.runBenchmark(
      'Linear+ReLU Fusion',
      async () => {
        // Optimized path with operation fusion
        backend.getOptimizer().setEnableOptimizations(true);
        // The optimizer should automatically detect and fuse these operations
        const linearOutput = await linearLayer.forward(input, backend);
        const reluOutput = await backend.relu(linearOutput);
        return reluOutput;
      },
      async () => {
        // Standard path with optimizations disabled
        backend.getOptimizer().setEnableOptimizations(false);
        const linearOutput = await linearLayer.forward(input, backend);
        const reluOutput = await backend.relu(linearOutput);
        return reluOutput;
      },
      10,
      3,
      true // Collect memory usage
    );
    
    expect(true).toBe(true);
  });
  
  test('Add + Tanh fusion (size=128x768)', async () => {
    const createTensors = () => {
      const a = new Tensor(new Float32Array(128 * 768), {shape: [128, 768], dataType: DataType.FLOAT32});
      const b = new Tensor(new Float32Array(128 * 768), {shape: [128, 768], dataType: DataType.FLOAT32});
      return { a, b };
    };
    
    const { a, b } = createTensors();
    const backend = new WebGPUBackend();
    
    await benchmarkRunner.runBenchmark(
      'Add+Tanh Fusion',
      async () => {
        backend.getOptimizer().setEnableOptimizations(true);
        const added = await backend.add(a, b);
        const result = await backend.tanh(added);
        return result;
      },
      async () => {
        backend.getOptimizer().setEnableOptimizations(false);
        const added = await backend.add(a, b);
        const result = await backend.tanh(added);
        return result;
      },
      10
    );
    
    expect(true).toBe(true);
  });
  
  test('LayerNorm + GELU fusion (batch=32, dim=768)', async () => {
    const batchSize = 32;
    const dim = 768;
    
    const createTensors = () => {
      const input = new Tensor(new Float32Array(batchSize * dim), {shape: [batchSize, dim], dataType: DataType.FLOAT32});
      const gamma = new Tensor(new Float32Array(dim), {shape: [dim], dataType: DataType.FLOAT32});
      const beta = new Tensor(new Float32Array(dim), {shape: [dim], dataType: DataType.FLOAT32});
      // Initialize gamma to ones and beta to zeros for a standard layer norm
      const gammaData = new Float32Array(dim).fill(1.0);
      const betaData = new Float32Array(dim).fill(0.0);
      gamma.setData(gammaData);
      beta.setData(betaData);
      return { input, gamma, beta };
    };
    
    const { input, gamma, beta } = createTensors();
    const backend = new WebGPUBackend();
    const layerNorm = new LayerNormalization(dim);
    layerNorm.setGamma(gamma);
    layerNorm.setBeta(beta);
    
    await benchmarkRunner.runBenchmark(
      'LayerNorm+GELU Fusion',
      async () => {
        backend.getOptimizer().setEnableOptimizations(true);
        const normOutput = await layerNorm.forward(input, backend);
        const geluOutput = await backend.gelu(normOutput);
        return geluOutput;
      },
      async () => {
        backend.getOptimizer().setEnableOptimizations(false);
        const normOutput = await layerNorm.forward(input, backend);
        const geluOutput = await backend.gelu(normOutput);
        return geluOutput;
      },
      10,
      3,
      true
    );
    
    expect(true).toBe(true);
  });
});

/**
 * Benchmark tests for browser-specific optimizations
 */
describe('WebGPU Optimizer Browser-Specific Optimizations', () => {
  let benchmarkRunner: BenchmarkRunner;
  
  beforeEach(async () => {
    benchmarkRunner = new BenchmarkRunner();
  });
  
  test('Browser-optimized matrix multiplication workgroups', async () => {
    const createTensors = () => {
      const a = new Tensor(new Float32Array(512 * 512), {shape: [512, 512], dataType: DataType.FLOAT32});
      const b = new Tensor(new Float32Array(512 * 512), {shape: [512, 512], dataType: DataType.FLOAT32});
      return { a, b };
    };
    
    const { a, b } = createTensors();
    const backend = new WebGPUBackend();
    
    await benchmarkRunner.runBenchmark(
      'Browser-Optimized Workgroups',
      async () => {
        // Enable browser-specific workgroup optimizations
        backend.getOptimizer().setEnableOptimizations(true);
        backend.getOptimizer().setEnableBrowserSpecificOptimizations(true);
        const result = await backend.matmul(a, b);
        return result;
      },
      async () => {
        // Use standard workgroups
        backend.getOptimizer().setEnableOptimizations(true);
        backend.getOptimizer().setEnableBrowserSpecificOptimizations(false);
        const result = await backend.matmul(a, b);
        return result;
      },
      10
    );
    
    expect(true).toBe(true);
  });
  
  test('Browser-optimized memory layout for convolution', async () => {
    // Create tensors for convolution
    const batchSize = 16;
    const inputChannels = 3;
    const outputChannels = 16;
    const inputSize = 64;
    const kernelSize = 3;
    
    const createTensors = () => {
      const input = new Tensor(
        new Float32Array(batchSize * inputChannels * inputSize * inputSize),
        {shape: [batchSize, inputChannels, inputSize, inputSize], dataType: DataType.FLOAT32}
      );
      const filter = new Tensor(
        new Float32Array(outputChannels * inputChannels * kernelSize * kernelSize),
        {shape: [outputChannels, inputChannels, kernelSize, kernelSize], dataType: DataType.FLOAT32}
      );
      return { input, filter };
    };
    
    const { input, filter } = createTensors();
    const backend = new WebGPUBackend();
    
    await benchmarkRunner.runBenchmark(
      'Browser-Optimized Memory Layout',
      async () => {
        backend.getOptimizer().setEnableOptimizations(true);
        backend.getOptimizer().setEnableBrowserSpecificOptimizations(true);
        backend.getOptimizer().setEnableMemoryLayoutOptimizations(true);
        const result = await backend.conv2d(input, filter, 1, 1, 'same');
        return result;
      },
      async () => {
        backend.getOptimizer().setEnableOptimizations(true);
        backend.getOptimizer().setEnableBrowserSpecificOptimizations(false);
        backend.getOptimizer().setEnableMemoryLayoutOptimizations(false);
        const result = await backend.conv2d(input, filter, 1, 1, 'same');
        return result;
      },
      5
    );
    
    expect(true).toBe(true);
  });
});

/**
 * Benchmark tests for neural network pattern recognition optimization
 */
describe('WebGPU Neural Network Pattern Recognition Benchmarks', () => {
  let benchmarkRunner: BenchmarkRunner;
  
  beforeEach(async () => {
    benchmarkRunner = new BenchmarkRunner();
  });
  
  test('Self-attention pattern recognition', async () => {
    // Create tensors for self-attention
    const batchSize = 4;
    const seqLength = 128;
    const embedSize = 768;
    const numHeads = 12;
    const headSize = embedSize / numHeads;
    
    const createTensors = () => {
      const input = new Tensor(
        new Float32Array(batchSize * seqLength * embedSize),
        {shape: [batchSize, seqLength, embedSize], dataType: DataType.FLOAT32}
      );
      
      // Create Q, K, V projection weights
      const qWeight = new Tensor(
        new Float32Array(embedSize * embedSize),
        {shape: [embedSize, embedSize], dataType: DataType.FLOAT32}
      );
      const kWeight = new Tensor(
        new Float32Array(embedSize * embedSize),
        {shape: [embedSize, embedSize], dataType: DataType.FLOAT32}
      );
      const vWeight = new Tensor(
        new Float32Array(embedSize * embedSize),
        {shape: [embedSize, embedSize], dataType: DataType.FLOAT32}
      );
      const outWeight = new Tensor(
        new Float32Array(embedSize * embedSize),
        {shape: [embedSize, embedSize], dataType: DataType.FLOAT32}
      );
      
      return { input, qWeight, kWeight, vWeight, outWeight };
    };
    
    const { input, qWeight, kWeight, vWeight, outWeight } = createTensors();
    const backend = new WebGPUBackend();
    const attention = new MultiHeadAttention(embedSize, numHeads);
    attention.setQKVWeights(qWeight, kWeight, vWeight);
    attention.setOutputWeight(outWeight);
    
    await benchmarkRunner.runBenchmark(
      'Self-Attention Pattern Recognition',
      async () => {
        // With pattern recognition enabled
        backend.getOptimizer().setEnableOptimizations(true);
        backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(true);
        const result = await attention.forward(input, input, input, null, backend);
        return result;
      },
      async () => {
        // With pattern recognition disabled
        backend.getOptimizer().setEnableOptimizations(true);
        backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(false);
        const result = await attention.forward(input, input, input, null, backend);
        return result;
      },
      5,
      2,
      true
    );
    
    expect(true).toBe(true);
  });
  
  test('Transformer FFN pattern recognition', async () => {
    // Create tensors for FFN (Feed-Forward Network)
    const batchSize = 8;
    const seqLength = 128;
    const embedSize = 768;
    const ffnSize = 3072;
    
    const createTensors = () => {
      const input = new Tensor(
        new Float32Array(batchSize * seqLength * embedSize),
        {shape: [batchSize, seqLength, embedSize], dataType: DataType.FLOAT32}
      );
      
      // FFN weights
      const fc1Weight = new Tensor(
        new Float32Array(embedSize * ffnSize),
        {shape: [embedSize, ffnSize], dataType: DataType.FLOAT32}
      );
      const fc1Bias = new Tensor(
        new Float32Array(ffnSize),
        {shape: [ffnSize], dataType: DataType.FLOAT32}
      );
      const fc2Weight = new Tensor(
        new Float32Array(ffnSize * embedSize),
        {shape: [ffnSize, embedSize], dataType: DataType.FLOAT32}
      );
      const fc2Bias = new Tensor(
        new Float32Array(embedSize),
        {shape: [embedSize], dataType: DataType.FLOAT32}
      );
      
      return { input, fc1Weight, fc1Bias, fc2Weight, fc2Bias };
    };
    
    const { input, fc1Weight, fc1Bias, fc2Weight, fc2Bias } = createTensors();
    const backend = new WebGPUBackend();
    
    // Create linear layers
    const linear1 = new LinearLayer(embedSize, ffnSize);
    linear1.setWeights(fc1Weight);
    linear1.setBias(fc1Bias);
    
    const linear2 = new LinearLayer(ffnSize, embedSize);
    linear2.setWeights(fc2Weight);
    linear2.setBias(fc2Bias);
    
    await benchmarkRunner.runBenchmark(
      'Transformer FFN Pattern Recognition',
      async () => {
        // With pattern recognition enabled
        backend.getOptimizer().setEnableOptimizations(true);
        backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(true);
        
        // This is the standard FFN pattern: Linear -> GELU -> Linear
        const hidden = await linear1.forward(input, backend);
        const activated = await backend.gelu(hidden);
        const output = await linear2.forward(activated, backend);
        return output;
      },
      async () => {
        // With pattern recognition disabled
        backend.getOptimizer().setEnableOptimizations(true);
        backend.getOptimizer().setEnableNeuralNetworkPatternRecognition(false);
        
        const hidden = await linear1.forward(input, backend);
        const activated = await backend.gelu(hidden);
        const output = await linear2.forward(activated, backend);
        return output;
      },
      5,
      2,
      true
    );
    
    expect(true).toBe(true);
  });
});

/**
 * Generate HTML report after all tests
 */
afterAll(() => {
  const benchmarkRunner = new BenchmarkRunner();
  const htmlReport = benchmarkRunner.generateHTMLReport();
  
  // In a real environment, we would save this to a file
  // For now, just log summary
  console.log('All WebGPU optimizer benchmarks completed');
  console.log(`Generated HTML report (${htmlReport.length} bytes)`);
});