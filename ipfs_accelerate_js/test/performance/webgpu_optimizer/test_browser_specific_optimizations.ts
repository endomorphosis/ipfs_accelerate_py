/**
 * WebGPU Browser-Specific Optimizations Tests
 * 
 * This file contains tests specifically focused on measuring the impact of
 * browser-specific optimizations in the WebGPU backend.
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

/**
 * Interface for browser-specific optimization test results
 */
interface BrowserOptimizationTestResult {
  operationName: string;
  shape: number[];
  genericTime: number;
  browserOptimizedTime: number;
  speedup: number;
  browser: string;
  detectedOptimizations: string[];
}

/**
 * Helper class for running browser-specific optimization tests
 */
class BrowserSpecificOptimizationTester {
  private results: BrowserOptimizationTestResult[] = [];
  private browser: BrowserType;
  
  constructor() {
    this.browser = BrowserDetector.detectBrowser();
  }
  
  /**
   * Runs a test comparing generic vs browser-specific optimizations
   */
  async runTest(
    operationName: string,
    shape: number[],
    operationFn: (backend: WebGPUBackend, tensor: Tensor) => Promise<Tensor>,
    iterations: number = 10,
    warmupIterations: number = 3
  ): Promise<BrowserOptimizationTestResult> {
    console.log(`Running browser optimization test: ${operationName} with shape [${shape.join(', ')}]`);
    
    // Create a tensor with random data
    const dataSize = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(dataSize);
    for (let i = 0; i < dataSize; i++) {
      data[i] = Math.random();
    }
    
    const tensor = new Tensor(data, {
      shape: shape,
      dataType: DataType.FLOAT32
    });
    
    // Create WebGPU backend
    const backend = new WebGPUBackend();
    
    // Warm-up phase
    for (let i = 0; i < warmupIterations; i++) {
      // With generic optimizations
      backend.getOptimizer().setEnableOptimizations(true);
      backend.getOptimizer().setEnableBrowserSpecificOptimizations(false);
      await operationFn(backend, tensor);
      
      // With browser-specific optimizations
      backend.getOptimizer().setEnableOptimizations(true);
      backend.getOptimizer().setEnableBrowserSpecificOptimizations(true);
      await operationFn(backend, tensor);
    }
    
    // Test with generic optimizations
    let genericTime = 0;
    for (let i = 0; i < iterations; i++) {
      backend.getOptimizer().setEnableOptimizations(true);
      backend.getOptimizer().setEnableBrowserSpecificOptimizations(false);
      
      const start = performance.now();
      await operationFn(backend, tensor);
      genericTime += performance.now() - start;
    }
    genericTime /= iterations;
    
    // Test with browser-specific optimizations
    let browserOptimizedTime = 0;
    for (let i = 0; i < iterations; i++) {
      backend.getOptimizer().setEnableOptimizations(true);
      backend.getOptimizer().setEnableBrowserSpecificOptimizations(true);
      
      const start = performance.now();
      await operationFn(backend, tensor);
      browserOptimizedTime += performance.now() - start;
    }
    browserOptimizedTime /= iterations;
    
    // Calculate speedup
    const speedup = genericTime / browserOptimizedTime;
    
    // Get detected optimizations
    const detectedOptimizations = backend.getOptimizer().getAppliedBrowserOptimizations() || [];
    
    // Create and store result
    const result: BrowserOptimizationTestResult = {
      operationName,
      shape,
      genericTime,
      browserOptimizedTime,
      speedup,
      browser: this.browser,
      detectedOptimizations
    };
    
    this.results.push(result);
    console.log(`Browser optimization test completed: ${operationName}`);
    console.log(`  Generic: ${genericTime.toFixed(2)}ms`);
    console.log(`  Browser-optimized: ${browserOptimizedTime.toFixed(2)}ms (${speedup.toFixed(2)}x speedup)`);
    console.log(`  Applied optimizations: ${detectedOptimizations.join(', ') || 'none'}`);
    
    return result;
  }
  
  /**
   * Returns all test results
   */
  getResults(): BrowserOptimizationTestResult[] {
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
        <title>WebGPU Browser-Specific Optimization Test Results</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; }
          table { border-collapse: collapse; width: 100%; }
          th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
          th { background-color: #f2f2f2; }
          tr:nth-child(even) { background-color: #f9f9f9; }
          .significant-speedup { color: green; font-weight: bold; }
          .moderate-speedup { color: blue; }
          .minimal-speedup { color: black; }
          .optimizations-list { font-size: 0.9em; color: #555; }
          .chart-container { margin-top: 30px; height: 400px; }
          .browser-info { padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 20px; }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      </head>
      <body>
        <h1>WebGPU Browser-Specific Optimization Test Results</h1>
        
        <div class="browser-info">
          <h2>Browser Information</h2>
          <p><strong>Detected browser:</strong> ${this.browser}</p>
        </div>
        
        <h2>Results Summary</h2>
        <table>
          <tr>
            <th>Operation</th>
            <th>Shape</th>
            <th>Generic Time (ms)</th>
            <th>Browser-Optimized Time (ms)</th>
            <th>Speedup</th>
            <th>Applied Optimizations</th>
          </tr>
    `;
    
    for (const result of this.results) {
      const speedupClass = result.speedup > 1.5 ? 'significant-speedup' : 
                          (result.speedup > 1.2 ? 'moderate-speedup' : 'minimal-speedup');
      
      html += `
        <tr>
          <td>${result.operationName}</td>
          <td>[${result.shape.join(', ')}]</td>
          <td>${result.genericTime.toFixed(2)}</td>
          <td>${result.browserOptimizedTime.toFixed(2)}</td>
          <td class="${speedupClass}">${result.speedup.toFixed(2)}x</td>
          <td class="optimizations-list">${result.detectedOptimizations.join('<br>') || 'None detected'}</td>
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
              labels: ${JSON.stringify(this.results.map(r => `${r.operationName} [${r.shape.join('x')}]`))},
              datasets: [
                {
                  label: 'Generic Optimizations',
                  data: ${JSON.stringify(this.results.map(r => r.genericTime))},
                  backgroundColor: 'rgba(54, 162, 235, 0.6)',
                  borderColor: 'rgb(54, 162, 235)',
                  borderWidth: 1
                },
                {
                  label: 'Browser-Specific Optimizations',
                  data: ${JSON.stringify(this.results.map(r => r.browserOptimizedTime))},
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
          <canvas id="speedupFactorChart"></canvas>
        </div>
        
        <script>
          const ctxSpeedup = document.getElementById('speedupFactorChart').getContext('2d');
          new Chart(ctxSpeedup, {
            type: 'bar',
            data: {
              labels: ${JSON.stringify(this.results.map(r => `${r.operationName} [${r.shape.join('x')}]`))},
              datasets: [{
                label: 'Speedup Factor (higher is better)',
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
 * Test browser-specific optimizations for matrix operations
 */
describe('WebGPU Browser-Specific Optimization Tests', () => {
  let tester: BrowserSpecificOptimizationTester;
  
  beforeEach(() => {
    tester = new BrowserSpecificOptimizationTester();
  });
  
  test('Matrix multiplication workgroup optimization', async () => {
    // Different shapes to test
    const shapes = [
      [128, 128],   // Small square matrix
      [512, 512],   // Medium square matrix
      [1024, 1024], // Large square matrix
      [2048, 512],  // Tall matrix
      [512, 2048]   // Wide matrix
    ];
    
    for (const shape of shapes) {
      // Create second tensor for matrix multiplication
      const shape2 = [shape[1], shape[0]]; // For valid matrix multiplication
      
      await tester.runTest(
        'MatMul',
        shape,
        async (backend, tensorA) => {
          // Create tensor B
          const dataB = new Float32Array(shape2[0] * shape2[1]);
          for (let i = 0; i < dataB.length; i++) {
            dataB[i] = Math.random();
          }
          
          const tensorB = new Tensor(dataB, {
            shape: shape2,
            dataType: DataType.FLOAT32
          });
          
          return await backend.matmul(tensorA, tensorB);
        },
        5, // Fewer iterations for large matrices
        2  // Fewer warmup iterations
      );
    }
    
    expect(tester.getResults().length).toBe(shapes.length);
  });
  
  test('Convolution browser-specific optimization', async () => {
    // Different shapes to test [batch, channels, height, width]
    const shapes = [
      [16, 3, 32, 32],    // Small image batch
      [8, 3, 64, 64],     // Medium image batch
      [4, 3, 128, 128],   // Large image batch
      [1, 3, 224, 224]    // Single large image
    ];
    
    for (const shape of shapes) {
      // Create filter shape [outputChannels, inputChannels, kernelHeight, kernelWidth]
      const filterShape = [16, shape[1], 3, 3];
      
      await tester.runTest(
        'Conv2D',
        shape,
        async (backend, inputTensor) => {
          // Create filter tensor
          const filterData = new Float32Array(filterShape.reduce((a, b) => a * b, 1));
          for (let i = 0; i < filterData.length; i++) {
            filterData[i] = Math.random() * 0.1; // Small weights
          }
          
          const filterTensor = new Tensor(filterData, {
            shape: filterShape,
            dataType: DataType.FLOAT32
          });
          
          return await backend.conv2d(inputTensor, filterTensor, 1, 1, 'same');
        },
        3, // Fewer iterations for convolution
        1  // Fewer warmup iterations
      );
    }
    
    expect(tester.getResults().length).toBe(shapes.length);
  });
  
  test('Element-wise operations browser optimization', async () => {
    // Different shapes to test
    const shapes = [
      [128, 128],     // Small square
      [512, 512],     // Medium square
      [1024, 1024],   // Large square
      [1, 8192],      // Long vector
      [100, 100, 10]  // 3D tensor
    ];
    
    // Different element-wise operations to test
    const operations = [
      {
        name: 'ReLU',
        fn: async (backend: WebGPUBackend, tensor: Tensor) => await backend.relu(tensor)
      },
      {
        name: 'Tanh',
        fn: async (backend: WebGPUBackend, tensor: Tensor) => await backend.tanh(tensor)
      },
      {
        name: 'Sigmoid',
        fn: async (backend: WebGPUBackend, tensor: Tensor) => await backend.sigmoid(tensor)
      }
    ];
    
    for (const shape of shapes) {
      for (const op of operations) {
        await tester.runTest(
          op.name,
          shape,
          op.fn,
          10, // More iterations for fast operations
          3
        );
      }
    }
    
    expect(tester.getResults().length).toBe(shapes.length * operations.length);
  });
  
  test('Reduction operations browser optimization', async () => {
    // Different shapes to test
    const shapes = [
      [128, 128],     // Small square
      [512, 512],     // Medium square 
      [1024, 1024],   // Large square
      [100, 100, 10]  // 3D tensor
    ];
    
    // Different reduction operations to test
    const operations = [
      {
        name: 'Mean',
        fn: async (backend: WebGPUBackend, tensor: Tensor) => await backend.mean(tensor, -1)
      },
      {
        name: 'Sum',
        fn: async (backend: WebGPUBackend, tensor: Tensor) => await backend.sum(tensor, -1)
      },
      {
        name: 'Max',
        fn: async (backend: WebGPUBackend, tensor: Tensor) => await backend.max(tensor, -1)
      }
    ];
    
    for (const shape of shapes) {
      for (const op of operations) {
        await tester.runTest(
          op.name,
          shape,
          op.fn,
          5,
          2
        );
      }
    }
    
    expect(tester.getResults().length).toBe(shapes.length * operations.length);
  });
  
  test('Batch normalization browser optimization', async () => {
    // Different shapes to test [batch, features]
    const shapes = [
      [32, 128],     // Small features
      [16, 512],     // Medium features
      [8, 1024],     // Large features
      [64, 256]      // More batches
    ];
    
    for (const shape of shapes) {
      const featureSize = shape[1];
      
      await tester.runTest(
        'BatchNorm',
        shape,
        async (backend, tensor) => {
          // Create gamma, beta, mean, variance tensors
          const createParamTensor = (value: number = 1.0) => {
            const data = new Float32Array(featureSize).fill(value);
            return new Tensor(data, {
              shape: [featureSize],
              dataType: DataType.FLOAT32
            });
          };
          
          const gamma = createParamTensor(1.0);
          const beta = createParamTensor(0.0);
          const mean = createParamTensor(0.0);
          const variance = createParamTensor(1.0);
          
          return await backend.batchNorm(tensor, gamma, beta, mean, variance, 1e-5);
        },
        5,
        2
      );
    }
    
    expect(tester.getResults().length).toBe(shapes.length);
  });
});

/**
 * Generate HTML report after all tests
 */
afterAll(() => {
  const tester = new BrowserSpecificOptimizationTester();
  const htmlReport = tester.generateHTMLReport();
  
  // In a real environment, we would save this to a file
  console.log('All browser-specific optimization tests completed');
  console.log(`Generated HTML report (${htmlReport.length} bytes)`);
});