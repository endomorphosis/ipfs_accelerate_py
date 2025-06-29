/**
 * Error Recovery System Example
 * 
 * This example demonstrates how to use the Error Recovery System to make hardware operations
 * more robust by automatically handling and recovering from various types of errors.
 */

import { 
  createHardwareAbstractionLayer, 
  HardwareAbstractionLayer 
} from '../src/hardware/hardware_abstraction_layer';
import { createErrorRecoveryManager } from '../src/hardware/error_recovery';
import { HardwareBackend } from '../src/hardware/interfaces/hardware_backend';
import { Tensor } from '../src/tensor/tensor';

// Example mock hardware backend that sometimes fails
class ExampleBackend implements HardwareBackend {
  readonly id: string;
  readonly type: string;
  readonly isAvailable: boolean = true;
  readonly capabilities: any = {
    maxDimensions: 4,
    maxMatrixSize: 1024,
    supportedDataTypes: ['float32', 'int32'],
    supportsAsync: true,
    supportedOperations: {
      basicArithmetic: true,
      matrixMultiplication: true,
      convolution: false,
      reduction: true,
      activation: true,
    }
  };
  
  private initialized: boolean = false;
  private failureRate: number = 0.5; // 50% chance of failure
  
  constructor(id: string, type: string, failureRate: number = 0.5) {
    this.id = id;
    this.type = type;
    this.failureRate = failureRate;
  }
  
  async initialize(): Promise<void> {
    console.log(`Initializing ${this.type} backend`);
    this.initialized = true;
  }
  
  isInitialized(): boolean {
    return this.initialized;
  }
  
  async allocateTensor<T>(tensor: Tensor<T>): Promise<void> {
    // Simulate tensor allocation
    console.log(`Allocating tensor with shape ${tensor.dimensions.join('x')}`);
  }
  
  releaseTensor<T>(tensor: Tensor<T>): void {
    // Simulate tensor release
    console.log(`Releasing tensor with shape ${tensor.dimensions.join('x')}`);
  }
  
  // Helper to simulate random failures
  private shouldFail(): boolean {
    return Math.random() < this.failureRate;
  }
  
  // Helper to create error based on operation
  private createError(operation: string): Error {
    const errorTypes = [
      `${this.type} backend failed to execute ${operation} operation: Out of memory`,
      `${this.type} backend failed to execute ${operation} operation: Invalid shape`,
      `${this.type} backend failed to execute ${operation} operation: Unsupported data type`,
      `${this.type} backend failed to execute ${operation} operation: Operation timed out`
    ];
    
    const errorMessage = errorTypes[Math.floor(Math.random() * errorTypes.length)];
    return new Error(errorMessage);
  }
  
  async matmul<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    // Simulate matrix multiplication with potential failure
    if (this.shouldFail()) {
      throw this.createError('matmul');
    }
    
    // Simulate successful operation
    console.log(`${this.type} backend: Executing matmul operation`);
    
    // Create a result tensor with appropriate shape
    const resultShape = [a.dimensions[0], b.dimensions[1]];
    return {
      dimensions: resultShape,
      data: [],
      dtype: a.dtype,
      dispose: () => {}
    } as Tensor<T>;
  }
  
  async transpose<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    // Simulate transpose with potential failure
    if (this.shouldFail()) {
      throw this.createError('transpose');
    }
    
    // Simulate successful operation
    console.log(`${this.type} backend: Executing transpose operation`);
    
    // Create a result tensor with transposed shape
    const resultShape = [...tensor.dimensions].reverse();
    return {
      dimensions: resultShape,
      data: [],
      dtype: tensor.dtype,
      dispose: () => {}
    } as Tensor<T>;
  }
  
  // Implement other required methods
  async add<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail()) throw this.createError('add');
    console.log(`${this.type} backend: Executing add operation`);
    return { dimensions: a.dimensions, data: [], dtype: a.dtype, dispose: () => {} } as Tensor<T>;
  }
  
  async subtract<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail()) throw this.createError('subtract');
    console.log(`${this.type} backend: Executing subtract operation`);
    return { dimensions: a.dimensions, data: [], dtype: a.dtype, dispose: () => {} } as Tensor<T>;
  }
  
  async multiply<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail()) throw this.createError('multiply');
    console.log(`${this.type} backend: Executing multiply operation`);
    return { dimensions: a.dimensions, data: [], dtype: a.dtype, dispose: () => {} } as Tensor<T>;
  }
  
  async divide<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail()) throw this.createError('divide');
    console.log(`${this.type} backend: Executing divide operation`);
    return { dimensions: a.dimensions, data: [], dtype: a.dtype, dispose: () => {} } as Tensor<T>;
  }
  
  async relu<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail()) throw this.createError('relu');
    console.log(`${this.type} backend: Executing relu operation`);
    return { dimensions: tensor.dimensions, data: [], dtype: tensor.dtype, dispose: () => {} } as Tensor<T>;
  }
  
  async sigmoid<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail()) throw this.createError('sigmoid');
    console.log(`${this.type} backend: Executing sigmoid operation`);
    return { dimensions: tensor.dimensions, data: [], dtype: tensor.dtype, dispose: () => {} } as Tensor<T>;
  }
  
  async tanh<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail()) throw this.createError('tanh');
    console.log(`${this.type} backend: Executing tanh operation`);
    return { dimensions: tensor.dimensions, data: [], dtype: tensor.dtype, dispose: () => {} } as Tensor<T>;
  }
  
  async softmax<T>(tensor: Tensor<T>, axis: number): Promise<Tensor<T>> {
    if (this.shouldFail()) throw this.createError('softmax');
    console.log(`${this.type} backend: Executing softmax operation with axis ${axis}`);
    return { dimensions: tensor.dimensions, data: [], dtype: tensor.dtype, dispose: () => {} } as Tensor<T>;
  }
  
  async reshape<T>(tensor: Tensor<T>, newShape: number[]): Promise<Tensor<T>> {
    if (this.shouldFail()) throw this.createError('reshape');
    console.log(`${this.type} backend: Executing reshape operation to ${newShape.join('x')}`);
    return { dimensions: newShape, data: [], dtype: tensor.dtype, dispose: () => {} } as Tensor<T>;
  }
  
  async sync(): Promise<void> {
    // Simulate synchronization
    console.log(`${this.type} backend: Synchronizing`);
  }
  
  dispose(): void {
    // Simulate disposal
    console.log(`${this.type} backend: Disposing resources`);
  }
}

// Helper function to create a tensor for the example
function createTensor<T>(dimensions: number[], data: T[], dtype: string = 'float32'): Tensor<T> {
  return {
    dimensions,
    data,
    dtype,
    dispose: () => { console.log(`Disposing tensor with shape ${dimensions.join('x')}`); }
  };
}

// Main example function
async function runErrorRecoveryExample() {
  console.log('IPFS Accelerate JS - Error Recovery System Example');
  console.log('================================================');
  
  // Create backends with different failure rates
  const backends = [
    new ExampleBackend('webgpu-1', 'webgpu', 0.8),  // High failure rate
    new ExampleBackend('webnn-1', 'webnn', 0.3),    // Medium failure rate
    new ExampleBackend('cpu-1', 'cpu', 0.1),        // Low failure rate
  ];
  
  // Create hardware abstraction layer
  const hal = createHardwareAbstractionLayer({
    backends,
    defaultBackend: 'webgpu',
    autoInitialize: true,
    useBrowserOptimizations: true,
    browserType: 'chrome',
    enableTensorSharing: true,
    enableOperationFusion: true
  });
  
  // Initialize the HAL
  await hal.initialize();
  
  // Get the performance tracker from the HAL
  // In a real application, this would be accessed directly via hal.performanceTracker
  // @ts-ignore - Accessing private property for demo purposes
  const performanceTracker = hal.performanceTracker;
  
  // Create error recovery manager
  const errorRecoveryManager = createErrorRecoveryManager(performanceTracker);
  
  // Create original operation functions
  const originalMatmul = async <T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> => {
    return hal.matmul(a, b);
  };
  
  const originalTranspose = async <T>(a: Tensor<T>): Promise<Tensor<T>> => {
    return hal.transpose(a);
  };
  
  // Create tensors for testing
  const tensorA = createTensor([2, 3], [1, 2, 3, 4, 5, 6]);
  const tensorB = createTensor([3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  
  // Protect the operations with error recovery
  const protectedMatmul = errorRecoveryManager.protect(
    originalMatmul,
    {
      operationName: 'matmul',
      backendType: hal.getBackendType() as any,
      // @ts-ignore - Accessing private property for demo purposes
      availableBackends: hal.backends,
      activeBackend: hal.getActiveBackend()!,
      performanceTracker,
      setActiveBackend: (backend) => hal.setActiveBackend(backend),
      browserType: 'chrome',
      useBrowserOptimizations: true
    }
  );
  
  const protectedTranspose = errorRecoveryManager.protect(
    originalTranspose,
    {
      operationName: 'transpose',
      backendType: hal.getBackendType() as any,
      // @ts-ignore - Accessing private property for demo purposes
      availableBackends: hal.backends,
      activeBackend: hal.getActiveBackend()!,
      performanceTracker,
      setActiveBackend: (backend) => hal.setActiveBackend(backend),
      browserType: 'chrome',
      useBrowserOptimizations: true
    }
  );
  
  console.log('\n1. Running operations without error recovery');
  console.log('-----------------------------------------');
  
  // Run multiple operations without error recovery
  for (let i = 0; i < 5; i++) {
    try {
      console.log(`\nIteration ${i + 1}:`);
      
      // Try matrix multiplication
      const result = await hal.matmul(tensorA, tensorB);
      console.log(`Matrix multiplication successful: Output shape ${result.dimensions.join('x')}`);
      
      // Try transpose
      const transposed = await hal.transpose(tensorA);
      console.log(`Transpose successful: Output shape ${transposed.dimensions.join('x')}`);
    } catch (error) {
      console.error(`Operation failed: ${(error as Error).message}`);
    }
  }
  
  console.log('\n2. Running operations with error recovery');
  console.log('---------------------------------------');
  
  // Run multiple operations with error recovery
  for (let i = 0; i < 5; i++) {
    try {
      console.log(`\nIteration ${i + 1}:`);
      
      // Try matrix multiplication with error recovery
      const result = await protectedMatmul(tensorA, tensorB);
      console.log(`Matrix multiplication successful: Output shape ${result.dimensions.join('x')}`);
      
      // Try transpose with error recovery
      const transposed = await protectedTranspose(tensorA);
      console.log(`Transpose successful: Output shape ${transposed.dimensions.join('x')}`);
    } catch (error) {
      console.error(`All recovery strategies failed: ${(error as Error).message}`);
    }
  }
  
  console.log('\n3. Generating recovery statistics');
  console.log('-------------------------------');
  
  // Generate and print recovery statistics
  const stats = errorRecoveryManager.getStrategySuccessRates();
  console.log('Recovery strategy success rates:');
  
  for (const [strategy, data] of Object.entries(stats)) {
    const successRate = data.attempts > 0 ? 
      `${(data.rate * 100).toFixed(1)}% (${data.successes}/${data.attempts})` : 
      'N/A (0 attempts)';
    
    console.log(`  ${strategy}: ${successRate}`);
  }
  
  console.log('\n4. Generating performance data');
  console.log('-----------------------------');
  
  // Generate and print performance data
  const performanceData = performanceTracker.exportPerformanceData();
  
  console.log('Operation summary:');
  console.log(`  Total operations: ${performanceData.summary.totalOperations}`);
  console.log(`  Total executions: ${performanceData.summary.totalExecutions}`);
  console.log(`  Success rate: ${(performanceData.summary.successRate * 100).toFixed(1)}%`);
  
  // Print backend recommendations based on performance
  const recommendations = performanceTracker.getAllRecommendations();
  
  console.log('\nRecommended backends based on performance:');
  for (const [operation, backend] of Object.entries(recommendations)) {
    console.log(`  ${operation}: ${backend}`);
  }
  
  // Cleanup
  hal.dispose();
  console.log('\nExample completed.');
}

// Run the example
runErrorRecoveryExample().catch(error => {
  console.error('Example failed:', error);
});

// Export for direct execution
export default runErrorRecoveryExample;