/**
 * Tests for Error Recovery System
 * 
 * This file includes comprehensive tests for the error recovery system, including:
 * - Error categorization
 * - Basic recovery strategies
 * - Strategy selection logic
 * - Performance tracking integration
 * - Recovery statistics and reporting
 * - Function protection mechanism
 */

import { 
  ErrorRecoveryManager, 
  ErrorCategory, 
  BackendSwitchStrategy,
  OperationFallbackStrategy,
  BrowserSpecificRecoveryStrategy,
  ParameterAdjustmentStrategy,
  RecoveryContext,
  ErrorRecoveryStrategy,
  RecoveryResult
} from '../../src/hardware/error_recovery';
import { PerformanceTracker } from '../../src/hardware/performance_tracking';
import { HardwareBackend } from '../../src/hardware/interfaces/hardware_backend';
import { Tensor } from '../../src/tensor/tensor';
import { BackendType } from '../../src/hardware/hardware_abstraction_layer';

// Mock Tensor class for testing
class MockTensor<T> implements Tensor<T> {
  dimensions: number[];
  data: T[];
  dtype: string;
  
  constructor(dimensions: number[], data: T[], dtype: string = 'float32') {
    this.dimensions = dimensions;
    this.data = data;
    this.dtype = dtype;
  }
  
  dispose(): void {}
}

// Mock Hardware Backend for testing
class MockHardwareBackend implements HardwareBackend {
  readonly id: string;
  readonly type: string;
  readonly isAvailable: boolean;
  private initialized: boolean = false;
  readonly shouldFail: boolean;
  readonly capabilities: any;
  
  constructor(id: string, type: string, isAvailable: boolean = true, shouldFail: boolean = false) {
    this.id = id;
    this.type = type;
    this.isAvailable = isAvailable;
    this.shouldFail = shouldFail;
    this.capabilities = {
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
  }
  
  async initialize(): Promise<void> {
    this.initialized = true;
  }
  
  isInitialized(): boolean {
    return this.initialized;
  }
  
  async allocateTensor<T>(tensor: Tensor<T>): Promise<void> {
    // Do nothing
  }
  
  releaseTensor<T>(tensor: Tensor<T>): void {
    // Do nothing
  }
  
  async add<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail) {
      throw new Error(`Backend ${this.type} failed to execute add operation`);
    }
    return new MockTensor([1], []) as Tensor<T>;
  }
  
  async subtract<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail) {
      throw new Error(`Backend ${this.type} failed to execute subtract operation`);
    }
    return new MockTensor([1], []) as Tensor<T>;
  }
  
  async multiply<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail) {
      throw new Error(`Backend ${this.type} failed to execute multiply operation`);
    }
    return new MockTensor([1], []) as Tensor<T>;
  }
  
  async divide<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail) {
      throw new Error(`Backend ${this.type} failed to execute divide operation`);
    }
    return new MockTensor([1], []) as Tensor<T>;
  }
  
  async matmul<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail) {
      throw new Error(`Backend ${this.type} failed to execute matmul operation`);
    }
    return new MockTensor([a.dimensions[0], b.dimensions[1]], []) as Tensor<T>;
  }
  
  async transpose<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail) {
      throw new Error(`Backend ${this.type} failed to execute transpose operation`);
    }
    
    // Simple transpose implementation for testing
    const newDimensions = [...tensor.dimensions].reverse();
    return new MockTensor(newDimensions, []) as Tensor<T>;
  }
  
  async relu<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail) {
      throw new Error(`Backend ${this.type} failed to execute relu operation`);
    }
    return new MockTensor(tensor.dimensions, []) as Tensor<T>;
  }
  
  async sigmoid<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail) {
      throw new Error(`Backend ${this.type} failed to execute sigmoid operation`);
    }
    return new MockTensor(tensor.dimensions, []) as Tensor<T>;
  }
  
  async tanh<T>(tensor: Tensor<T>): Promise<Tensor<T>> {
    if (this.shouldFail) {
      throw new Error(`Backend ${this.type} failed to execute tanh operation`);
    }
    return new MockTensor(tensor.dimensions, []) as Tensor<T>;
  }
  
  async softmax<T>(tensor: Tensor<T>, axis: number): Promise<Tensor<T>> {
    if (this.shouldFail) {
      throw new Error(`Backend ${this.type} failed to execute softmax operation`);
    }
    return new MockTensor(tensor.dimensions, []) as Tensor<T>;
  }
  
  async reshape<T>(tensor: Tensor<T>, newShape: number[]): Promise<Tensor<T>> {
    if (this.shouldFail) {
      throw new Error(`Backend ${this.type} failed to execute reshape operation`);
    }
    return new MockTensor(newShape, []) as Tensor<T>;
  }
  
  async sync(): Promise<void> {
    // Do nothing
  }
  
  dispose(): void {
    // Do nothing
  }
}

// Custom recovery strategy for testing
class TestRecoveryStrategy implements ErrorRecoveryStrategy {
  readonly name = 'test_strategy';
  readonly priority = 10;
  public canHandleCalled = false;
  public recoverCalled = false;
  public shouldSucceed = true;
  
  canHandle(error: Error, context: RecoveryContext): boolean {
    this.canHandleCalled = true;
    return true;
  }
  
  async recover(error: Error, context: RecoveryContext): Promise<RecoveryResult> {
    this.recoverCalled = true;
    
    if (this.shouldSucceed) {
      return {
        success: true,
        result: 'test_result',
        successfulStrategy: this.name,
        performance: {
          durationMs: 10
        }
      };
    } else {
      return {
        success: false,
        error: new Error('Test recovery strategy failed'),
        successfulStrategy: undefined
      };
    }
  }
}

describe('Error Recovery System', () => {
  let performanceTracker: PerformanceTracker;
  let errorRecoveryManager: ErrorRecoveryManager;
  let backends: Map<BackendType, HardwareBackend>;
  let activeBackend: HardwareBackend;
  
  beforeEach(() => {
    performanceTracker = new PerformanceTracker();
    
    // Create mock backends
    backends = new Map();
    backends.set('webgpu', new MockHardwareBackend('webgpu-1', 'webgpu', true, true)); // Will fail operations
    backends.set('webnn', new MockHardwareBackend('webnn-1', 'webnn', true, false)); // Will succeed operations
    backends.set('cpu', new MockHardwareBackend('cpu-1', 'cpu', true, false)); // Will succeed operations
    
    // Set active backend to one that will fail
    activeBackend = backends.get('webgpu')!;
    
    errorRecoveryManager = new ErrorRecoveryManager(performanceTracker);
  });
  
  describe('Error categorization', () => {
    test('should correctly categorize memory errors', () => {
      const error1 = new Error('Out of memory error occurred');
      const error2 = new Error('Memory allocation failed');
      const error3 = new RangeError('Buffer size limit exceeded');
      
      expect(errorRecoveryManager.categorizeError(error1)).toBe(ErrorCategory.MEMORY);
      expect(errorRecoveryManager.categorizeError(error2)).toBe(ErrorCategory.MEMORY);
      expect(errorRecoveryManager.categorizeError(error3)).toBe(ErrorCategory.MEMORY);
    });
    
    test('should correctly categorize execution errors', () => {
      const error1 = new Error('Invalid shape for operation');
      const error2 = new Error('Dimension mismatch between tensors');
      const error3 = new Error('Invalid input format');
      
      expect(errorRecoveryManager.categorizeError(error1)).toBe(ErrorCategory.EXECUTION);
      expect(errorRecoveryManager.categorizeError(error2)).toBe(ErrorCategory.EXECUTION);
      expect(errorRecoveryManager.categorizeError(error3)).toBe(ErrorCategory.EXECUTION);
    });
    
    test('should correctly categorize precision errors', () => {
      const error1 = new Error('Precision loss detected');
      const error2 = new Error('Numeric overflow occurred');
      const error3 = new Error('NaN values detected in output');
      
      expect(errorRecoveryManager.categorizeError(error1)).toBe(ErrorCategory.PRECISION);
      expect(errorRecoveryManager.categorizeError(error2)).toBe(ErrorCategory.PRECISION);
      expect(errorRecoveryManager.categorizeError(error3)).toBe(ErrorCategory.PRECISION);
    });
    
    test('should correctly categorize compatibility errors', () => {
      const error1 = new Error('Operation not supported on this backend');
      const error2 = new Error('Unsupported data type');
      const error3 = new Error('Browser compatibility issue detected');
      
      expect(errorRecoveryManager.categorizeError(error1)).toBe(ErrorCategory.COMPATIBILITY);
      expect(errorRecoveryManager.categorizeError(error2)).toBe(ErrorCategory.COMPATIBILITY);
      expect(errorRecoveryManager.categorizeError(error3)).toBe(ErrorCategory.COMPATIBILITY);
    });
    
    test('should default to unknown for uncategorized errors', () => {
      const error = new Error('Some random error message');
      
      expect(errorRecoveryManager.categorizeError(error)).toBe(ErrorCategory.UNKNOWN);
    });
  });
  
  describe('Recovery strategies', () => {
    describe('BackendSwitchStrategy', () => {
      test('should handle errors by switching to another backend', async () => {
        const strategy = new BackendSwitchStrategy();
        const error = new Error('Backend failed to execute operation');
        
        const context: RecoveryContext = {
          operationName: 'matmul',
          originalFn: async () => new MockTensor([1, 1], []),
          args: [new MockTensor([2, 3], []), new MockTensor([3, 4], [])],
          backendType: 'webgpu', // Currently using failing backend
          availableBackends: backends,
          activeBackend,
          performanceTracker,
          setActiveBackend: (backend: HardwareBackend) => {
            activeBackend = backend;
          },
          browserType: 'chrome',
          inputShapes: [[2, 3], [3, 4]]
        };
        
        // Check if strategy can handle this error
        expect(strategy.canHandle(error, context)).toBe(true);
        
        // Attempt recovery
        const result = await strategy.recover(error, context);
        
        // Should successfully recover using another backend
        expect(result.success).toBe(true);
        expect(result.result).toBeInstanceOf(MockTensor);
        expect(result.successfulStrategy).toBe('backend_switch');
        
        // Should have switched to a working backend
        expect(activeBackend.type).not.toBe('webgpu');
      });
      
      test('should return failure if all backends fail', async () => {
        // Create all failing backends
        backends.clear();
        backends.set('webgpu', new MockHardwareBackend('webgpu-1', 'webgpu', true, true));
        backends.set('webnn', new MockHardwareBackend('webnn-1', 'webnn', true, true));
        backends.set('cpu', new MockHardwareBackend('cpu-1', 'cpu', true, true));
        
        activeBackend = backends.get('webgpu')!;
        
        const strategy = new BackendSwitchStrategy();
        const error = new Error('Backend failed to execute operation');
        
        const context: RecoveryContext = {
          operationName: 'matmul',
          originalFn: async () => new MockTensor([1, 1], []),
          args: [new MockTensor([2, 3], []), new MockTensor([3, 4], [])],
          backendType: 'webgpu',
          availableBackends: backends,
          activeBackend,
          performanceTracker,
          setActiveBackend: (backend: HardwareBackend) => {
            activeBackend = backend;
          },
          browserType: 'chrome',
          inputShapes: [[2, 3], [3, 4]]
        };
        
        // Attempt recovery
        const result = await strategy.recover(error, context);
        
        // Should fail to recover
        expect(result.success).toBe(false);
        expect(result.error).toBeDefined();
      });
    });
    
    describe('OperationFallbackStrategy', () => {
      test('should check if operation has alternative implementation', () => {
        const strategy = new OperationFallbackStrategy();
        
        const context: RecoveryContext = {
          operationName: 'matmul',
          originalFn: async () => {},
          args: [],
          backendType: 'webgpu',
          availableBackends: backends,
          activeBackend,
          performanceTracker,
          setActiveBackend: () => {}
        };
        
        expect(strategy.canHandle(new Error('test'), context)).toBe(true);
        
        // Change to unsupported operation
        context.operationName = 'unsupported_op';
        expect(strategy.canHandle(new Error('test'), context)).toBe(false);
      });
      
      test('should attempt recovery with alternative implementation', async () => {
        const strategy = new OperationFallbackStrategy();
        const error = new Error('Backend failed to execute operation');
        
        const tensorA = new MockTensor([2, 3], []);
        const tensorB = new MockTensor([3, 4], []);
        
        const context: RecoveryContext = {
          operationName: 'matmul',
          originalFn: async () => new MockTensor([1, 1], []),
          args: [tensorA, tensorB],
          backendType: 'webgpu',
          availableBackends: backends,
          activeBackend: backends.get('webnn')!, // Use a working backend for the test
          performanceTracker,
          setActiveBackend: () => {},
          browserType: 'chrome',
          inputShapes: [[2, 3], [3, 4]]
        };
        
        // Attempt recovery
        const result = await strategy.recover(error, context);
        
        // Should successfully recover
        expect(result.success).toBe(true);
        expect(result.result).toBeInstanceOf(MockTensor);
        expect(result.successfulStrategy).toBe('operation_fallback');
      });
    });
    
    describe('BrowserSpecificRecoveryStrategy', () => {
      test('should only handle errors when browser optimizations are enabled', () => {
        const strategy = new BrowserSpecificRecoveryStrategy();
        const error = new Error('test error');
        
        const context: RecoveryContext = {
          operationName: 'matmul',
          originalFn: async () => {},
          args: [],
          backendType: 'webgpu',
          availableBackends: backends,
          activeBackend,
          performanceTracker,
          setActiveBackend: () => {},
          browserType: 'chrome',
          useBrowserOptimizations: true
        };
        
        expect(strategy.canHandle(error, context)).toBe(true);
        
        // Disable browser optimizations
        context.useBrowserOptimizations = false;
        expect(strategy.canHandle(error, context)).toBe(false);
        
        // Remove browser type
        context.useBrowserOptimizations = true;
        context.browserType = undefined;
        expect(strategy.canHandle(error, context)).toBe(false);
      });
    });
    
    describe('ParameterAdjustmentStrategy', () => {
      test('should handle specific operations that support parameter adjustment', () => {
        const strategy = new ParameterAdjustmentStrategy();
        const error = new Error('test error');
        
        const context: RecoveryContext = {
          operationName: 'matmul',
          originalFn: async () => {},
          args: [],
          backendType: 'webgpu',
          availableBackends: backends,
          activeBackend,
          performanceTracker,
          setActiveBackend: () => {}
        };
        
        expect(strategy.canHandle(error, context)).toBe(true);
        
        // Change to another adjustable operation
        context.operationName = 'softmax';
        expect(strategy.canHandle(error, context)).toBe(true);
        
        // Change to non-adjustable operation
        context.operationName = 'non_adjustable_op';
        expect(strategy.canHandle(error, context)).toBe(false);
      });
      
      test('should adjust matmul parameters', async () => {
        const strategy = new ParameterAdjustmentStrategy();
        const error = new Error('test error');
        
        let receivedArgs: any[] = [];
        
        const mockFn = async (...args: any[]) => {
          receivedArgs = args;
          return new MockTensor([2, 4], []);
        };
        
        const context: RecoveryContext = {
          operationName: 'matmul',
          originalFn: mockFn,
          args: [new MockTensor([2, 3], []), new MockTensor([3, 4], []), {}],
          backendType: 'webgpu',
          availableBackends: backends,
          activeBackend,
          performanceTracker,
          setActiveBackend: () => {}
        };
        
        // Attempt recovery
        const result = await strategy.recover(error, context);
        
        // Should successfully recover
        expect(result.success).toBe(true);
        expect(result.successfulStrategy).toBe('parameter_adjustment_matmul');
        
        // Should have modified the options parameter
        expect(receivedArgs[2]).toEqual({ useOptimization: false });
      });
      
      test('should adjust softmax parameters', async () => {
        const strategy = new ParameterAdjustmentStrategy();
        const error = new Error('test error');
        
        let receivedArgs: any[] = [];
        
        const mockFn = async (...args: any[]) => {
          receivedArgs = args;
          return new MockTensor([2, 3], []);
        };
        
        const context: RecoveryContext = {
          operationName: 'softmax',
          originalFn: mockFn,
          args: [new MockTensor([2, 3], [])], // Missing axis parameter
          backendType: 'webgpu',
          availableBackends: backends,
          activeBackend,
          performanceTracker,
          setActiveBackend: () => {}
        };
        
        // Attempt recovery
        const result = await strategy.recover(error, context);
        
        // Should successfully recover
        expect(result.success).toBe(true);
        expect(result.successfulStrategy).toBe('parameter_adjustment_softmax');
        
        // Should have added the axis parameter
        expect(receivedArgs[1]).toBe(1); // Last dimension index for a [2,3] tensor
      });
    });
  });
  
  describe('ErrorRecoveryManager', () => {
    test('should register default strategies', () => {
      const report = errorRecoveryManager.generateReport();
      const strategies = report.registeredStrategies;
      
      // Should have all default strategies
      expect(strategies.length).toBe(4);
      expect(strategies.some(s => s.name === 'backend_switch')).toBe(true);
      expect(strategies.some(s => s.name === 'operation_fallback')).toBe(true);
      expect(strategies.some(s => s.name === 'browser_specific')).toBe(true);
      expect(strategies.some(s => s.name === 'parameter_adjustment')).toBe(true);
      
      // Should be sorted by priority
      for (let i = 1; i < strategies.length; i++) {
        expect(strategies[i].priority).toBeGreaterThanOrEqual(strategies[i - 1].priority);
      }
    });
    
    test('should register additional strategies', () => {
      const testStrategy = new TestRecoveryStrategy();
      const manager = new ErrorRecoveryManager(performanceTracker, {
        additionalStrategies: [testStrategy]
      });
      
      const report = manager.generateReport();
      const strategies = report.registeredStrategies;
      
      // Should have all default strategies plus the test strategy
      expect(strategies.length).toBe(5);
      expect(strategies.some(s => s.name === 'test_strategy')).toBe(true);
    });
    
    test('should try strategies in order until one succeeds', async () => {
      const testStrategy = new TestRecoveryStrategy();
      testStrategy.shouldSucceed = true;
      
      const manager = new ErrorRecoveryManager(performanceTracker, {
        additionalStrategies: [testStrategy]
      });
      
      const error = new Error('Test error');
      const context: RecoveryContext = {
        operationName: 'matmul',
        originalFn: async () => {},
        args: [],
        backendType: 'webgpu',
        availableBackends: backends,
        activeBackend,
        performanceTracker,
        setActiveBackend: () => {}
      };
      
      // Spy on strategies
      const backendSwitchSpy = jest.spyOn(BackendSwitchStrategy.prototype, 'canHandle');
      
      // Attempt recovery
      const result = await manager.recoverFromError(error, context);
      
      // Should use the first strategy that can handle the error
      expect(backendSwitchSpy).toHaveBeenCalled();
      
      // Should have successfully recovered
      expect(result.success).toBe(true);
    });
    
    test('should protect a function with error recovery', async () => {
      // Create a function that will fail
      const originalFn = async (a: number, b: number): Promise<number> => {
        if (a < 0 || b < 0) {
          throw new Error('Negative values not allowed');
        }
        return a + b;
      };
      
      // Add a test strategy that will always succeed
      const testStrategy = new TestRecoveryStrategy();
      testStrategy.shouldSucceed = true;
      
      const manager = new ErrorRecoveryManager(performanceTracker, {
        additionalStrategies: [testStrategy]
      });
      
      // Create protected version of the function
      const protectedFn = manager.protect<[number, number], number>(
        originalFn,
        {
          operationName: 'add',
          backendType: 'webgpu',
          availableBackends: backends,
          activeBackend,
          performanceTracker,
          setActiveBackend: () => {}
        }
      );
      
      // Function should work normally with valid inputs
      expect(await protectedFn(2, 3)).toBe(5);
      
      // Function should recover from errors with invalid inputs
      const result = await protectedFn(-1, 2);
      
      // The test strategy returns 'test_result'
      expect(result).toBe('test_result');
      
      // The strategy should have been called
      expect(testStrategy.canHandleCalled).toBe(true);
      expect(testStrategy.recoverCalled).toBe(true);
    });
    
    test('should track recovery statistics', async () => {
      // Create strategies with different behaviors
      const successStrategy = new TestRecoveryStrategy();
      successStrategy.shouldSucceed = true;
      successStrategy.name = 'success_strategy';
      
      const failureStrategy = new TestRecoveryStrategy();
      failureStrategy.shouldSucceed = false;
      failureStrategy.name = 'failure_strategy';
      
      const manager = new ErrorRecoveryManager(performanceTracker, {
        additionalStrategies: [successStrategy, failureStrategy]
      });
      
      const error = new Error('Test error');
      const context: RecoveryContext = {
        operationName: 'test_op',
        originalFn: async () => {},
        args: [],
        backendType: 'webgpu',
        availableBackends: backends,
        activeBackend,
        performanceTracker,
        setActiveBackend: () => {}
      };
      
      // First try with success strategy
      await manager.recoverFromError(error, context);
      
      // Check statistics
      const stats = manager.getStrategySuccessRates();
      
      // Success strategy should have 1 attempt and 1 success
      expect(stats.success_strategy.attempts).toBe(1);
      expect(stats.success_strategy.successes).toBe(1);
      expect(stats.success_strategy.rate).toBe(1.0);
      
      // Failure strategy should not have been tried
      expect(stats.failure_strategy.attempts).toBe(0);
    });
  });
});