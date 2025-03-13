/**
 * WebGPU Memory Layout Optimization Tests
 * 
 * This file contains tests specifically focused on measuring the impact of
 * memory layout optimizations in the WebGPU backend.
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
 * Test result interface
 */
interface MemoryLayoutTestResult {
  operationName: string;
  shape: number[];
  rowMajorTime: number;
  columnMajorTime: number;
  optimalTime: number;
  optimalLayout: 'row-major' | 'column-major';
  speedupVsWorst: number;
  browser: string;
}

/**
 * Helper class for running memory layout optimization tests
 */
class MemoryLayoutOptimizationTester {
  private results: MemoryLayoutTestResult[] = [];
  private browser: BrowserType;
  
  constructor() {
    this.browser = BrowserDetector.detectBrowser();
  }
  
  /**
   * Runs a test comparing row-major, column-major, and auto-selected memory layouts
   */
  async runTest(
    operationName: string,
    shape: number[],
    operationFn: (backend: WebGPUBackend, tensor: Tensor) => Promise<Tensor>,
    iterations: number = 10
  ): Promise<MemoryLayoutTestResult> {
    console.log(`Running memory layout test: ${operationName} with shape [${shape.join(', ')}]`);
    
    // Create a tensor with random data
    const dataSize = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(dataSize);
    for (let i = 0; i < dataSize; i++) {
      data[i] = Math.random();
    }
    
    // Create tensors with different memory layouts
    const tensorRowMajor = new Tensor(data, {
      shape: shape,
      dataType: DataType.FLOAT32,
      memoryLayout: 'row-major'
    });
    
    const tensorColumnMajor = new Tensor(data, {
      shape: shape,
      dataType: DataType.FLOAT32,
      memoryLayout: 'column-major'
    });
    
    const tensorOptimal = new Tensor(data, {
      shape: shape,
      dataType: DataType.FLOAT32,
      // Let optimizer choose layout
    });
    
    // Create WebGPU backend
    const backend = new WebGPUBackend();
    
    // Test row-major layout
    let rowMajorTime = 0;
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await operationFn(backend, tensorRowMajor);
      rowMajorTime += performance.now() - start;
    }
    rowMajorTime /= iterations;
    
    // Test column-major layout
    let columnMajorTime = 0;
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await operationFn(backend, tensorColumnMajor);
      columnMajorTime += performance.now() - start;
    }
    columnMajorTime /= iterations;
    
    // Test optimizer-selected layout
    let optimalTime = 0;
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      // Enable memory layout optimization
      backend.getOptimizer().setEnableOptimizations(true);
      backend.getOptimizer().setEnableMemoryLayoutOptimizations(true);
      await operationFn(backend, tensorOptimal);
      optimalTime += performance.now() - start;
    }
    optimalTime /= iterations;
    
    // Determine which layout the optimizer should choose
    const optimalLayout: 'row-major' | 'column-major' = 
      rowMajorTime <= columnMajorTime ? 'row-major' : 'column-major';
    
    // Calculate speedup
    const worstTime = Math.max(rowMajorTime, columnMajorTime);
    const speedupVsWorst = worstTime / optimalTime;
    
    // Create and store result
    const result: MemoryLayoutTestResult = {
      operationName,
      shape,
      rowMajorTime,
      columnMajorTime,
      optimalTime,
      optimalLayout,
      speedupVsWorst,
      browser: this.browser
    };
    
    this.results.push(result);
    console.log(`Memory layout test completed: ${operationName}`);
    console.log(`  Row-major: ${rowMajorTime.toFixed(2)}ms`);
    console.log(`  Column-major: ${columnMajorTime.toFixed(2)}ms`);
    console.log(`  Optimal: ${optimalTime.toFixed(2)}ms (${speedupVsWorst.toFixed(2)}x vs worst)`);
    
    return result;
  }
  
  /**
   * Returns all test results
   */
  getResults(): MemoryLayoutTestResult[] {
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
        <title>WebGPU Memory Layout Optimization Test Results</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; }
          table { border-collapse: collapse; width: 100%; }
          th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
          th { background-color: #f2f2f2; }
          tr:nth-child(even) { background-color: #f9f9f9; }
          .optimal-row { background-color: #d4edda; }
          .suboptimal-row { background-color: #f8d7da; }
          .chart-container { margin-top: 30px; height: 400px; }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      </head>
      <body>
        <h1>WebGPU Memory Layout Optimization Test Results</h1>
        <h2>Environment Information</h2>
        <p>Browser: ${this.browser}</p>
        
        <h2>Results Summary</h2>
        <table>
          <tr>
            <th>Operation</th>
            <th>Shape</th>
            <th>Row-Major Time (ms)</th>
            <th>Column-Major Time (ms)</th>
            <th>Optimal Time (ms)</th>
            <th>Optimal Layout</th>
            <th>Speedup vs Worst</th>
          </tr>
    `;
    
    for (const result of this.results) {
      const rowClass = result.optimalTime <= Math.min(result.rowMajorTime, result.columnMajorTime) * 1.05 ? 
        'optimal-row' : 'suboptimal-row';
      
      html += `
        <tr class="${rowClass}">
          <td>${result.operationName}</td>
          <td>[${result.shape.join(', ')}]</td>
          <td>${result.rowMajorTime.toFixed(2)}</td>
          <td>${result.columnMajorTime.toFixed(2)}</td>
          <td>${result.optimalTime.toFixed(2)}</td>
          <td>${result.optimalLayout}</td>
          <td>${result.speedupVsWorst.toFixed(2)}x</td>
        </tr>
      `;
    }
    
    html += `
        </table>
        
        <div class="chart-container">
          <canvas id="layoutChart"></canvas>
        </div>
        
        <script>
          const ctx = document.getElementById('layoutChart').getContext('2d');
          new Chart(ctx, {
            type: 'bar',
            data: {
              labels: ${JSON.stringify(this.results.map(r => `${r.operationName} [${r.shape.join('x')}]`))},
              datasets: [
                {
                  label: 'Row Major',
                  data: ${JSON.stringify(this.results.map(r => r.rowMajorTime))},
                  backgroundColor: 'rgba(54, 162, 235, 0.6)',
                  borderColor: 'rgb(54, 162, 235)',
                  borderWidth: 1
                },
                {
                  label: 'Column Major',
                  data: ${JSON.stringify(this.results.map(r => r.columnMajorTime))},
                  backgroundColor: 'rgba(255, 99, 132, 0.6)',
                  borderColor: 'rgb(255, 99, 132)',
                  borderWidth: 1
                },
                {
                  label: 'Optimal (Auto-selected)',
                  data: ${JSON.stringify(this.results.map(r => r.optimalTime))},
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
      </body>
      </html>
    `;
    
    return html;
  }
}

/**
 * Test memory layout impact on matrix operations
 */
describe('WebGPU Memory Layout Optimization Tests', () => {
  let tester: MemoryLayoutOptimizationTester;
  
  beforeEach(() => {
    tester = new MemoryLayoutOptimizationTester();
  });
  
  test('Matrix multiplication memory layout optimization', async () => {
    // Define matrix shapes to test
    const shapes = [
      [128, 128],
      [256, 256],
      [512, 512],
      [1024, 1024],
      [2048, 128], // Tall matrix
      [128, 2048]  // Wide matrix
    ];
    
    for (const shape of shapes) {
      // Create second tensor for matrix multiplication
      const shape2 = [shape[1], shape[0]]; // For valid matrix multiplication
      
      await tester.runTest(
        'MatMul',
        shape,
        async (backend, tensorA) => {
          // Create tensor B with the same layout as A
          const dataB = new Float32Array(shape2[0] * shape2[1]);
          for (let i = 0; i < dataB.length; i++) {
            dataB[i] = Math.random();
          }
          
          const tensorB = new Tensor(dataB, {
            shape: shape2,
            dataType: DataType.FLOAT32,
            memoryLayout: tensorA.getMemoryLayout()
          });
          
          return await backend.matmul(tensorA, tensorB);
        },
        5 // Fewer iterations for large matrices
      );
    }
    
    expect(tester.getResults().length).toBe(shapes.length);
  });
  
  test('Transpose memory layout optimization', async () => {
    // Define matrix shapes to test
    const shapes = [
      [128, 128],
      [256, 256],
      [512, 512],
      [1024, 1024],
      [2048, 128], // Tall matrix
      [128, 2048]  // Wide matrix
    ];
    
    for (const shape of shapes) {
      await tester.runTest(
        'Transpose',
        shape,
        async (backend, tensor) => {
          return await backend.transpose(tensor, [1, 0]);
        },
        5
      );
    }
    
    expect(tester.getResults().length).toBe(shapes.length);
  });
  
  test('Convolution memory layout optimization', async () => {
    // Define convolution input shapes to test
    // [batch, channels, height, width]
    const shapes = [
      [16, 3, 32, 32],   // Small image
      [8, 3, 64, 64],    // Medium image
      [4, 3, 128, 128],  // Large image
      [32, 1, 28, 28],   // MNIST-like
      [16, 3, 224, 224]  // ImageNet-like
    ];
    
    for (const shape of shapes) {
      // Create filter for convolution
      const filterShape = [16, shape[1], 3, 3]; // [outputChannels, inputChannels, kernelHeight, kernelWidth]
      
      await tester.runTest(
        'Conv2D',
        shape,
        async (backend, inputTensor) => {
          // Create filter tensor with the same layout as input
          const filterData = new Float32Array(filterShape.reduce((a, b) => a * b, 1));
          for (let i = 0; i < filterData.length; i++) {
            filterData[i] = Math.random() * 0.1; // Small weights
          }
          
          const filterTensor = new Tensor(filterData, {
            shape: filterShape,
            dataType: DataType.FLOAT32,
            memoryLayout: inputTensor.getMemoryLayout()
          });
          
          return await backend.conv2d(inputTensor, filterTensor, 1, 1, 'same');
        },
        3 // Fewer iterations for convolution
      );
    }
    
    expect(tester.getResults().length).toBe(shapes.length);
  });
  
  test('Element-wise operations memory layout optimization', async () => {
    // Define shapes to test
    const shapes = [
      [128, 128],
      [256, 256],
      [512, 512],
      [1024, 1024],
      [1, 4096],   // Vector
      [4096, 1]    // Vector
    ];
    
    for (const shape of shapes) {
      await tester.runTest(
        'ReLU',
        shape,
        async (backend, tensor) => {
          return await backend.relu(tensor);
        },
        10
      );
    }
    
    expect(tester.getResults().length).toBe(shapes.length);
  });
  
  test('Batch matrix multiplication memory layout optimization', async () => {
    // Define batch matrix shapes to test
    // [batch, rows, cols]
    const shapes = [
      [8, 128, 128],
      [4, 256, 256],
      [2, 512, 512],
      [16, 64, 64],
      [32, 32, 32]
    ];
    
    for (const shape of shapes) {
      await tester.runTest(
        'BatchMatMul',
        shape,
        async (backend, tensorA) => {
          // Create tensor B with the same layout as A
          // Shape for B: [batch, cols, rows] for valid batch matmul
          const shapeB = [shape[0], shape[2], shape[1]];
          const dataB = new Float32Array(shapeB.reduce((a, b) => a * b, 1));
          for (let i = 0; i < dataB.length; i++) {
            dataB[i] = Math.random();
          }
          
          const tensorB = new Tensor(dataB, {
            shape: shapeB,
            dataType: DataType.FLOAT32,
            memoryLayout: tensorA.getMemoryLayout()
          });
          
          return await backend.batchMatMul(tensorA, tensorB);
        },
        5
      );
    }
    
    expect(tester.getResults().length).toBe(shapes.length);
  });
});

/**
 * Generate HTML report after all tests
 */
afterAll(() => {
  const tester = new MemoryLayoutOptimizationTester();
  const htmlReport = tester.generateHTMLReport();
  
  // In a real environment, we would save this to a file
  console.log('All memory layout optimization tests completed');
  console.log(`Generated HTML report (${htmlReport.length} bytes)`);
});