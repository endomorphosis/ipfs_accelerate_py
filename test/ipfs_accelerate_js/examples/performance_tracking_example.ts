/**
 * Example of using the WebGPU/WebNN Performance Tracking System
 * 
 * This example demonstrates how to:
 * - Use the Hardware Abstraction Layer with performance tracking
 * - View operation performance history
 * - Get recommended backends based on performance
 * - Export performance data for analysis
 */

import { 
  HardwareAbstractionLayer,
  BackendType,
  BackendSelectionCriteria
} from '../src/hardware/hardware_abstraction_layer';
import { Tensor } from '../src/tensor/tensor';

// Simulated hardware backends for demonstration
const mockWebGPUBackend = {
  id: 'webgpu-1',
  type: 'webgpu',
  isAvailable: true,
  isInitialized: () => true,
  initialize: async () => {},
  capabilities: {
    maxDimensions: 4,
    maxMatrixSize: 16384,
    supportedDataTypes: ['float32', 'float16', 'int32'],
    supportsAsync: true,
    supportedOperations: {
      basicArithmetic: true,
      matrixMultiplication: true,
      convolution: true,
      reduction: true,
      activation: true
    }
  },
  // Mock tensor operations with different performance characteristics
  allocateTensor: async () => {},
  releaseTensor: () => {},
  add: async (a: any, b: any) => a,
  subtract: async (a: any, b: any) => a,
  multiply: async (a: any, b: any) => a,
  divide: async (a: any, b: any) => a,
  matmul: async (a: any, b: any) => {
    // Simulate matmul taking 10ms
    await new Promise(resolve => setTimeout(resolve, 10));
    return new Tensor([a.dimensions[0], b.dimensions[1]], new Float32Array(a.dimensions[0] * b.dimensions[1]), 'float32');
  },
  transpose: async (a: any) => {
    // Simulate transpose taking 5ms
    await new Promise(resolve => setTimeout(resolve, 5));
    return new Tensor([a.dimensions[1], a.dimensions[0]], new Float32Array(a.dimensions[0] * a.dimensions[1]), 'float32');
  },
  relu: async (a: any) => a,
  sigmoid: async (a: any) => a,
  tanh: async (a: any) => a,
  softmax: async (a: any, axis: number) => a,
  reshape: async (a: any, newShape: number[]) => new Tensor(newShape, a.data, a.dtype),
  sync: async () => {},
  dispose: () => {}
};

const mockWebNNBackend = {
  id: 'webnn-1',
  type: 'webnn',
  isAvailable: true,
  isInitialized: () => true,
  initialize: async () => {},
  capabilities: {
    maxDimensions: 4,
    maxMatrixSize: 8192,
    supportedDataTypes: ['float32', 'int32'],
    supportsAsync: true,
    supportedOperations: {
      basicArithmetic: true,
      matrixMultiplication: true,
      convolution: true,
      reduction: true,
      activation: true
    }
  },
  // Mock tensor operations with different performance characteristics
  allocateTensor: async () => {},
  releaseTensor: () => {},
  add: async (a: any, b: any) => a,
  subtract: async (a: any, b: any) => a,
  multiply: async (a: any, b: any) => a,
  divide: async (a: any, b: any) => a,
  matmul: async (a: any, b: any) => {
    // Simulate matmul taking 5ms (faster than WebGPU)
    await new Promise(resolve => setTimeout(resolve, 5));
    return new Tensor([a.dimensions[0], b.dimensions[1]], new Float32Array(a.dimensions[0] * b.dimensions[1]), 'float32');
  },
  transpose: async (a: any) => {
    // Simulate transpose taking 3ms (faster than WebGPU)
    await new Promise(resolve => setTimeout(resolve, 3));
    return new Tensor([a.dimensions[1], a.dimensions[0]], new Float32Array(a.dimensions[0] * a.dimensions[1]), 'float32');
  },
  relu: async (a: any) => a,
  sigmoid: async (a: any) => a,
  tanh: async (a: any) => a,
  softmax: async (a: any, axis: number) => a,
  reshape: async (a: any, newShape: number[]) => new Tensor(newShape, a.data, a.dtype),
  sync: async () => {},
  dispose: () => {}
};

const mockCPUBackend = {
  id: 'cpu-1',
  type: 'cpu',
  isAvailable: true,
  isInitialized: () => true,
  initialize: async () => {},
  capabilities: {
    maxDimensions: 4,
    maxMatrixSize: 4096,
    supportedDataTypes: ['float32', 'int32', 'int8'],
    supportsAsync: false,
    supportedOperations: {
      basicArithmetic: true,
      matrixMultiplication: true,
      convolution: true,
      reduction: true,
      activation: true
    }
  },
  // Mock tensor operations with different performance characteristics
  allocateTensor: async () => {},
  releaseTensor: () => {},
  add: async (a: any, b: any) => a,
  subtract: async (a: any, b: any) => a,
  multiply: async (a: any, b: any) => a,
  divide: async (a: any, b: any) => a,
  matmul: async (a: any, b: any) => {
    // Simulate matmul taking 50ms (much slower than GPU)
    await new Promise(resolve => setTimeout(resolve, 50));
    return new Tensor([a.dimensions[0], b.dimensions[1]], new Float32Array(a.dimensions[0] * b.dimensions[1]), 'float32');
  },
  transpose: async (a: any) => {
    // Simulate transpose taking 20ms (much slower than GPU)
    await new Promise(resolve => setTimeout(resolve, 20));
    return new Tensor([a.dimensions[1], a.dimensions[0]], new Float32Array(a.dimensions[0] * a.dimensions[1]), 'float32');
  },
  relu: async (a: any) => a,
  sigmoid: async (a: any) => a,
  tanh: async (a: any) => a,
  softmax: async (a: any, axis: number) => a,
  reshape: async (a: any, newShape: number[]) => new Tensor(newShape, a.data, a.dtype),
  sync: async () => {},
  dispose: () => {}
};

async function runPerformanceTrackingDemo() {
  console.log('WebGPU/WebNN Performance Tracking Demo');
  console.log('=====================================\n');
  
  // Create Hardware Abstraction Layer with all backends
  const hal = new HardwareAbstractionLayer({
    backends: [
      mockWebGPUBackend,
      mockWebNNBackend,
      mockCPUBackend
    ],
    defaultBackend: 'webgpu',
    browserType: 'chrome'
  });
  
  // Initialize the HAL
  await hal.initialize();
  
  console.log('HAL initialized with backends:');
  console.log(' - WebGPU');
  console.log(' - WebNN');
  console.log(' - CPU\n');
  
  // Create test tensors
  console.log('Creating test tensors...');
  const tensorA = await hal.createTensor({
    dimensions: [1, 128],
    data: new Float32Array(128).fill(1),
    dtype: 'float32'
  });
  
  const tensorB = await hal.createTensor({
    dimensions: [128, 64],
    data: new Float32Array(128 * 64).fill(1),
    dtype: 'float32'
  });
  
  console.log(`Created tensorA with shape ${tensorA.dimensions}`);
  console.log(`Created tensorB with shape ${tensorB.dimensions}\n`);
  
  // Run operations multiple times with different backends
  console.log('Running matrix multiply operations on different backends...');
  
  // First run on WebGPU (default backend)
  console.log('\nRunning 10 operations on WebGPU:');
  for (let i = 0; i < 10; i++) {
    const result = await hal.matmul(tensorA, tensorB);
    console.log(`  - Run ${i+1}: Matrix multiply completed with shape ${result.dimensions}`);
  }
  
  // Switch to WebNN backend and run there
  console.log('\nRunning 10 operations on WebNN:');
  
  // Manually select WebNN backend using criteria
  const webnnCriteria: BackendSelectionCriteria = {
    modelType: 'text',
    backendPreference: ['webnn'],
    prioritizeSpeed: true
  };
  const webnnBackend = await hal.selectBackend(webnnCriteria);
  
  if (webnnBackend) {
    hal.setActiveBackend(webnnBackend);
    
    for (let i = 0; i < 10; i++) {
      const result = await hal.matmul(tensorA, tensorB);
      console.log(`  - Run ${i+1}: Matrix multiply completed with shape ${result.dimensions}`);
    }
  }
  
  // Switch to CPU backend
  console.log('\nRunning 5 operations on CPU (slower):');
  
  // Manually select CPU backend using criteria
  const cpuCriteria: BackendSelectionCriteria = {
    modelType: 'text',
    backendPreference: ['cpu']
  };
  const cpuBackend = await hal.selectBackend(cpuCriteria);
  
  if (cpuBackend) {
    hal.setActiveBackend(cpuBackend);
    
    for (let i = 0; i < 5; i++) {
      const result = await hal.matmul(tensorA, tensorB);
      console.log(`  - Run ${i+1}: Matrix multiply completed with shape ${result.dimensions}`);
    }
  }
  
  // Now run some transpose operations
  console.log('\nRunning transpose operations:');
  
  // Back to WebGPU for transpose
  const gpuCriteria: BackendSelectionCriteria = {
    modelType: 'text',
    backendPreference: ['webgpu']
  };
  const gpuBackend = await hal.selectBackend(gpuCriteria);
  
  if (gpuBackend) {
    hal.setActiveBackend(gpuBackend);
    
    for (let i = 0; i < 5; i++) {
      const result = await hal.transpose(tensorA);
      console.log(`  - WebGPU Run ${i+1}: Transpose completed with shape ${result.dimensions}`);
    }
  }
  
  // WebNN for transpose
  if (webnnBackend) {
    hal.setActiveBackend(webnnBackend);
    
    for (let i = 0; i < 5; i++) {
      const result = await hal.transpose(tensorA);
      console.log(`  - WebNN Run ${i+1}: Transpose completed with shape ${result.dimensions}`);
    }
  }
  
  // Get performance data
  console.log('\nPerformance Analysis:');
  
  // Get matmul performance history across backends
  console.log('\n1. Matrix Multiply Performance:');
  for (const backend of ['webgpu', 'webnn', 'cpu']) {
    const history = hal.getOperationPerformanceHistory('matmul');
    const filtered = history.filter(r => r.backendType === backend);
    
    if (filtered.length > 0) {
      const avgTime = filtered.reduce((sum, r) => sum + r.durationMs, 0) / filtered.length;
      console.log(`  - ${backend.toUpperCase()}: Average: ${avgTime.toFixed(2)}ms, Records: ${filtered.length}`);
    }
  }
  
  // Get recommended backend for matmul
  const recommendedMatmul = hal.getRecommendedBackendForOperation('matmul');
  console.log(`\n2. Recommended Backend for Matrix Multiply: ${recommendedMatmul}`);
  
  // Get recommended backend for transpose
  const recommendedTranspose = hal.getRecommendedBackendForOperation('transpose');
  console.log(`3. Recommended Backend for Transpose: ${recommendedTranspose}`);
  
  // Generate full performance report
  console.log('\n4. Full Performance Report:');
  const report = hal.generatePerformanceReport();
  
  console.log(`   - Total Operations: ${report.summary.totalOperations}`);
  console.log(`   - Total Backends: ${report.summary.totalBackends}`);
  console.log(`   - Total Executions: ${report.summary.totalExecutions}`);
  console.log(`   - Success Rate: ${(report.summary.successRate * 100).toFixed(2)}%`);
  
  console.log('\n5. Backend Recommendations:');
  for (const [operation, backend] of Object.entries(report.recommendations)) {
    console.log(`   - ${operation}: ${backend}`);
  }
  
  // Clean up
  await hal.releaseTensor(tensorA);
  await hal.releaseTensor(tensorB);
  hal.dispose();
  
  console.log('\nDemo completed successfully!');
}

// Run the demo
runPerformanceTrackingDemo().catch(error => {
  console.error('Demo failed with error:', error);
});

// If this file is run directly (not imported)
if (typeof require !== 'undefined' && require.main === module) {
  runPerformanceTrackingDemo();
}