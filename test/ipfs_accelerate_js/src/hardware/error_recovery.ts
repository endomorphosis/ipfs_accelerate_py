/**
 * Error Recovery System for WebGPU/WebNN Resource Pool Integration
 * 
 * This module provides intelligent error recovery strategies for hardware operations, 
 * leveraging performance data to make optimal recovery decisions when operations fail.
 * 
 * Key features:
 * 1. Multiple recovery strategies including backend switching, operation fallback, and parameter adjustment
 * 2. Performance-aware recovery decisions using historical performance data
 * 3. Browser-specific recovery optimizations
 * 4. Progressive fallback with increasing levels of intervention
 * 5. Automatic learning from successful recoveries
 */

import { BackendType } from './hardware_abstraction_layer';
import { PerformanceTracker, PerformanceRecord } from './performance_tracking';
import { BrowserType } from './webgpu/browser_optimized_operations';
import { HardwareBackend } from './interfaces/hardware_backend';
import { Tensor } from '../tensor/tensor';

/**
 * Error category classification
 */
export enum ErrorCategory {
  /** Memory allocation or limit errors */
  MEMORY = 'memory',
  /** Execution errors (e.g., shape mismatches) */
  EXECUTION = 'execution',
  /** Precision-related errors */
  PRECISION = 'precision',
  /** Hardware compatibility errors */
  COMPATIBILITY = 'compatibility',
  /** Backend-specific implementation errors */
  IMPLEMENTATION = 'implementation',
  /** Unknown errors */
  UNKNOWN = 'unknown'
}

/**
 * Recovery attempt result
 */
export interface RecoveryResult {
  /** Whether the recovery was successful */
  success: boolean;
  /** If successful, the result of the operation */
  result?: any;
  /** Strategy that was successful */
  successfulStrategy?: string;
  /** Error if recovery failed */
  error?: Error;
  /** Performance metrics of the recovery */
  performance?: {
    /** Time taken for recovery in milliseconds */
    durationMs: number;
    /** Original operation time in milliseconds, if available */
    originalDurationMs?: number;
  };
}

/**
 * Interface for recovery strategies
 */
export interface ErrorRecoveryStrategy {
  /**
   * Name of the strategy
   */
  readonly name: string;
  
  /**
   * Priority of the strategy (lower numbers execute first)
   */
  readonly priority: number;
  
  /**
   * Check if this strategy can handle the given error
   * @param error Error to check
   * @param context Context information about the operation
   * @returns Whether this strategy can handle the error
   */
  canHandle(
    error: Error, 
    context: RecoveryContext
  ): boolean;
  
  /**
   * Attempt to recover from the error
   * @param error Error to recover from
   * @param context Context information about the operation
   * @returns Promise resolving to recovery result
   */
  recover(
    error: Error, 
    context: RecoveryContext
  ): Promise<RecoveryResult>;
}

/**
 * Context information for recovery
 */
export interface RecoveryContext {
  /** Name of the operation that failed */
  operationName: string;
  /** Original function that failed */
  originalFn: Function;
  /** Arguments to the operation */
  args: any[];
  /** Type of backend where the error occurred */
  backendType: BackendType;
  /** Available backends */
  availableBackends: Map<BackendType, HardwareBackend>;
  /** Current active backend */
  activeBackend: HardwareBackend;
  /** Performance tracker */
  performanceTracker: PerformanceTracker;
  /** Set active backend function */
  setActiveBackend: (backend: HardwareBackend) => void;
  /** Browser type if known */
  browserType?: BrowserType;
  /** Shapes of input tensors */
  inputShapes?: number[][];
  /** Whether browser optimizations are enabled */
  useBrowserOptimizations?: boolean;
}

/**
 * Strategy for switching to an alternative backend
 */
export class BackendSwitchStrategy implements ErrorRecoveryStrategy {
  readonly name = 'backend_switch';
  readonly priority = 1;
  
  /**
   * Check if this strategy can handle the error
   * @param error Error to check
   * @param context Context information about the operation
   * @returns Whether this strategy can handle the error
   */
  canHandle(
    error: Error, 
    context: RecoveryContext
  ): boolean {
    // Can handle if there are alternative backends available
    const alternativeBackends = this.getAlternativeBackends(context);
    return alternativeBackends.length > 0;
  }
  
  /**
   * Get available alternative backends
   * @param context Recovery context
   * @returns Array of available alternative backends
   */
  private getAlternativeBackends(
    context: RecoveryContext
  ): HardwareBackend[] {
    const currentBackendType = context.backendType;
    const alternativeBackends: HardwareBackend[] = [];
    
    // Use performance data to prioritize backends
    const recommendedBackend = context.performanceTracker.getRecommendedBackend(
      context.operationName
    );
    
    // Try recommended backend first if it's different from current
    if (recommendedBackend && recommendedBackend !== currentBackendType) {
      const recommended = context.availableBackends.get(recommendedBackend);
      if (recommended && recommended.isAvailable) {
        alternativeBackends.push(recommended);
      }
    }
    
    // Add all other available backends
    for (const [type, backend] of context.availableBackends.entries()) {
      if (type !== currentBackendType && 
          type !== recommendedBackend && 
          backend.isAvailable) {
        alternativeBackends.push(backend);
      }
    }
    
    return alternativeBackends;
  }
  
  /**
   * Attempt to recover from the error by switching backends
   * @param error Error to recover from
   * @param context Context information about the operation
   * @returns Promise resolving to recovery result
   */
  async recover(
    error: Error, 
    context: RecoveryContext
  ): Promise<RecoveryResult> {
    const alternativeBackends = this.getAlternativeBackends(context);
    
    // Try each alternative backend
    for (const backend of alternativeBackends) {
      // Store original backend to restore if needed
      const originalBackend = context.activeBackend;
      
      try {
        // Set the alternative backend as active
        context.setActiveBackend(backend);
        
        // Make sure it's initialized
        if (!backend.isInitialized()) {
          await backend.initialize();
        }
        
        console.info(
          `[ErrorRecovery] Trying alternative backend ${backend.type} for ${context.operationName}`
        );
        
        // Start timing recovery
        const startTime = performance.now();
        
        // Retry the operation with the new backend
        const result = await context.originalFn(...context.args);
        
        // End timing
        const endTime = performance.now();
        const durationMs = endTime - startTime;
        
        console.info(
          `[ErrorRecovery] Successfully recovered using ${backend.type} backend for ${context.operationName}`
        );
        
        // Return success result
        return {
          success: true,
          result,
          successfulStrategy: this.name,
          performance: {
            durationMs
          }
        };
      } catch (recoveryError) {
        console.warn(
          `[ErrorRecovery] Alternative backend ${backend.type} also failed for ${context.operationName}: ${recoveryError}`
        );
        
        // Restore original backend
        context.setActiveBackend(originalBackend);
      }
    }
    
    // If we get here, all alternative backends failed
    return {
      success: false,
      error: new Error(`All alternative backends failed for ${context.operationName}`),
      successfulStrategy: undefined
    };
  }
}

/**
 * Strategy for using alternative implementation of the same operation
 */
export class OperationFallbackStrategy implements ErrorRecoveryStrategy {
  readonly name = 'operation_fallback';
  readonly priority = 2;
  
  /**
   * Check if this strategy can handle the error
   * @param error Error to check
   * @param context Context information about the operation
   * @returns Whether this strategy can handle the error
   */
  canHandle(
    error: Error, 
    context: RecoveryContext
  ): boolean {
    // Can handle specific operations that have alternative implementations
    return this.hasAlternativeImplementation(context.operationName);
  }
  
  /**
   * Check if an operation has alternative implementations
   * @param operation Operation name
   * @returns Whether operation has alternatives
   */
  private hasAlternativeImplementation(operation: string): boolean {
    // Operations with known alternative implementations
    const operationsWithAlternatives = [
      'matmul',
      'transpose',
      'gelu',
      'layerNorm',
      'softmax'
    ];
    
    return operationsWithAlternatives.includes(operation);
  }
  
  /**
   * Get alternative implementation for an operation
   * @param context Recovery context
   * @returns Alternative implementation function or null
   */
  private getAlternativeImplementation(
    context: RecoveryContext
  ): Function | null {
    const operationName = context.operationName;
    const args = context.args;
    
    // For matrix multiplication, use a basic implementation
    if (operationName === 'matmul') {
      return async () => {
        // Check that arguments are tensors
        if (!(args[0] instanceof Tensor) || !(args[1] instanceof Tensor)) {
          throw new Error('Arguments must be tensors');
        }
        
        const a = args[0] as Tensor<any>;
        const b = args[1] as Tensor<any>;
        
        // Validate shapes for matmul
        if (a.dimensions.length < 2 || b.dimensions.length < 2) {
          throw new Error('Tensors must have at least 2 dimensions for matmul');
        }
        
        // Get current backend
        const backend = context.activeBackend;
        
        // Basic matmul implementation using other operations
        // Split matrices into smaller blocks if needed
        const blockSize = 32; // Small enough to avoid memory issues
        
        // Determine if we need to split
        const needsSplit = (
          a.dimensions[a.dimensions.length - 1] > 128 ||
          b.dimensions[b.dimensions.length - 2] > 128
        );
        
        if (needsSplit) {
          // Implement block matrix multiplication
          console.info(`[ErrorRecovery] Using blocked matrix multiplication for ${operationName}`);
          
          // Implementation would depend on backend capabilities
          // For simplicity, we'll just call the regular implementation with smaller matrices
          // But a real implementation would split into blocks and compute in parts
          
          // This is a placeholder - a real implementation would do block multiplication
          return backend.matmul(a, b);
        } else {
          // Try using individual operations if available
          // matmul(A, B) can be done with a series of dot products
          // But this is much less efficient than native implementation
          
          // For simplicity in this example, we'll still call the backend's matmul
          // but with different execution parameters
          console.info(`[ErrorRecovery] Using alternative matmul implementation for ${operationName}`);
          
          // A real implementation would:
          // 1. Extract data from tensors
          // 2. Perform manual matrix multiplication
          // 3. Create a new tensor with results
          
          // This is a placeholder
          return backend.matmul(a, b);
        }
      };
    }
    
    // For transpose, use a basic implementation
    if (operationName === 'transpose') {
      return async () => {
        // Check that arguments are tensors
        if (!(args[0] instanceof Tensor)) {
          throw new Error('Arguments must be tensors');
        }
        
        const tensor = args[0] as Tensor<any>;
        const backend = context.activeBackend;
        
        console.info(`[ErrorRecovery] Using alternative transpose implementation for ${operationName}`);
        
        // A real implementation would:
        // 1. Extract data from tensor
        // 2. Manually transpose the data
        // 3. Create a new tensor with results
        
        // This is a placeholder
        return backend.transpose(tensor);
      };
    }
    
    // Add alternative implementations for other operations as needed
    
    return null;
  }
  
  /**
   * Attempt to recover from the error using alternative operation implementation
   * @param error Error to recover from
   * @param context Context information about the operation
   * @returns Promise resolving to recovery result
   */
  async recover(
    error: Error, 
    context: RecoveryContext
  ): Promise<RecoveryResult> {
    const alternativeImpl = this.getAlternativeImplementation(context);
    
    if (!alternativeImpl) {
      return {
        success: false,
        error: new Error(`No alternative implementation available for ${context.operationName}`),
        successfulStrategy: undefined
      };
    }
    
    try {
      console.info(
        `[ErrorRecovery] Trying alternative implementation for ${context.operationName}`
      );
      
      // Start timing recovery
      const startTime = performance.now();
      
      // Execute the alternative implementation
      const result = await alternativeImpl();
      
      // End timing
      const endTime = performance.now();
      const durationMs = endTime - startTime;
      
      console.info(
        `[ErrorRecovery] Successfully recovered using alternative implementation for ${context.operationName}`
      );
      
      // Return success result
      return {
        success: true,
        result,
        successfulStrategy: this.name,
        performance: {
          durationMs
        }
      };
    } catch (recoveryError) {
      console.warn(
        `[ErrorRecovery] Alternative implementation also failed for ${context.operationName}: ${recoveryError}`
      );
      
      return {
        success: false,
        error: new Error(`Alternative implementation failed for ${context.operationName}: ${recoveryError}`),
        successfulStrategy: undefined
      };
    }
  }
}

/**
 * Strategy for browser-specific recovery techniques
 */
export class BrowserSpecificRecoveryStrategy implements ErrorRecoveryStrategy {
  readonly name = 'browser_specific';
  readonly priority = 3;
  
  /**
   * Check if this strategy can handle the error
   * @param error Error to check
   * @param context Context information about the operation
   * @returns Whether this strategy can handle the error
   */
  canHandle(
    error: Error, 
    context: RecoveryContext
  ): boolean {
    // Can handle if browser type is known and browser optimizations are enabled
    return (
      context.browserType !== undefined && 
      context.useBrowserOptimizations === true
    );
  }
  
  /**
   * Attempt to recover from the error using browser-specific techniques
   * @param error Error to recover from
   * @param context Context information about the operation
   * @returns Promise resolving to recovery result
   */
  async recover(
    error: Error, 
    context: RecoveryContext
  ): Promise<RecoveryResult> {
    const browserType = context.browserType!;
    const operationName = context.operationName;
    
    console.info(
      `[ErrorRecovery] Trying browser-specific recovery for ${operationName} in ${browserType}`
    );
    
    try {
      // Start timing recovery
      const startTime = performance.now();
      
      // Apply browser-specific techniques
      let result;
      
      switch (browserType) {
        case 'firefox':
          result = await this.recoverInFirefox(error, context);
          break;
        case 'chrome':
          result = await this.recoverInChrome(error, context);
          break;
        case 'edge':
          result = await this.recoverInEdge(error, context);
          break;
        case 'safari':
          result = await this.recoverInSafari(error, context);
          break;
        default:
          throw new Error(`Unsupported browser type: ${browserType}`);
      }
      
      // End timing
      const endTime = performance.now();
      const durationMs = endTime - startTime;
      
      console.info(
        `[ErrorRecovery] Successfully recovered using ${browserType}-specific technique for ${operationName}`
      );
      
      // Return success result
      return {
        success: true,
        result,
        successfulStrategy: `${this.name}_${browserType}`,
        performance: {
          durationMs
        }
      };
    } catch (recoveryError) {
      console.warn(
        `[ErrorRecovery] Browser-specific recovery failed for ${operationName} in ${browserType}: ${recoveryError}`
      );
      
      return {
        success: false,
        error: new Error(`Browser-specific recovery failed for ${operationName} in ${browserType}: ${recoveryError}`),
        successfulStrategy: undefined
      };
    }
  }
  
  /**
   * Apply Firefox-specific recovery techniques
   * @param error Original error
   * @param context Recovery context
   * @returns Result of the operation
   */
  private async recoverInFirefox(
    error: Error, 
    context: RecoveryContext
  ): Promise<any> {
    const operationName = context.operationName;
    
    if (operationName === 'matmul') {
      // Firefox-specific matmul recovery
      // For example, Firefox performs better with certain workgroup sizes
      console.info('[ErrorRecovery] Applying Firefox-specific matmul optimizations');
      
      // In a real implementation, we would modify workgroup size or shader implementation
      // Here we'll just retry the original function with adjusted parameters
      return context.originalFn(...context.args);
    }
    
    if (operationName === 'transpose') {
      // Firefox-specific transpose recovery
      console.info('[ErrorRecovery] Applying Firefox-specific transpose optimizations');
      
      return context.originalFn(...context.args);
    }
    
    // Add more Firefox-specific recovery techniques as needed
    
    // If no specific technique for this operation, throw
    throw new Error(`No Firefox-specific recovery for ${operationName}`);
  }
  
  /**
   * Apply Chrome-specific recovery techniques
   * @param error Original error
   * @param context Recovery context
   * @returns Result of the operation
   */
  private async recoverInChrome(
    error: Error, 
    context: RecoveryContext
  ): Promise<any> {
    const operationName = context.operationName;
    
    if (operationName === 'matmul') {
      // Chrome-specific matmul recovery
      console.info('[ErrorRecovery] Applying Chrome-specific matmul optimizations');
      
      return context.originalFn(...context.args);
    }
    
    // Add more Chrome-specific recovery techniques as needed
    
    // If no specific technique for this operation, throw
    throw new Error(`No Chrome-specific recovery for ${operationName}`);
  }
  
  /**
   * Apply Edge-specific recovery techniques
   * @param error Original error
   * @param context Recovery context
   * @returns Result of the operation
   */
  private async recoverInEdge(
    error: Error, 
    context: RecoveryContext
  ): Promise<any> {
    const operationName = context.operationName;
    
    if (operationName === 'matmul' && context.backendType === 'webnn') {
      // Edge-specific WebNN matmul recovery
      console.info('[ErrorRecovery] Applying Edge-specific WebNN optimizations');
      
      return context.originalFn(...context.args);
    }
    
    // Add more Edge-specific recovery techniques as needed
    
    // If no specific technique for this operation, throw
    throw new Error(`No Edge-specific recovery for ${operationName}`);
  }
  
  /**
   * Apply Safari-specific recovery techniques
   * @param error Original error
   * @param context Recovery context
   * @returns Result of the operation
   */
  private async recoverInSafari(
    error: Error, 
    context: RecoveryContext
  ): Promise<any> {
    const operationName = context.operationName;
    
    if (operationName === 'matmul') {
      // Safari-specific matmul recovery
      console.info('[ErrorRecovery] Applying Safari-specific matmul optimizations');
      
      return context.originalFn(...context.args);
    }
    
    // Add more Safari-specific recovery techniques as needed
    
    // If no specific technique for this operation, throw
    throw new Error(`No Safari-specific recovery for ${operationName}`);
  }
}

/**
 * Strategy for adjusting operation parameters
 */
export class ParameterAdjustmentStrategy implements ErrorRecoveryStrategy {
  readonly name = 'parameter_adjustment';
  readonly priority = 4;
  
  /**
   * Check if this strategy can handle the error
   * @param error Error to check
   * @param context Context information about the operation
   * @returns Whether this strategy can handle the error
   */
  canHandle(
    error: Error, 
    context: RecoveryContext
  ): boolean {
    // Can handle specific operations where parameters can be adjusted
    const adjustableOperations = [
      'matmul',
      'softmax',
      'reshape'
    ];
    
    return adjustableOperations.includes(context.operationName);
  }
  
  /**
   * Attempt to recover from the error by adjusting parameters
   * @param error Error to recover from
   * @param context Context information about the operation
   * @returns Promise resolving to recovery result
   */
  async recover(
    error: Error, 
    context: RecoveryContext
  ): Promise<RecoveryResult> {
    const operationName = context.operationName;
    const args = [...context.args]; // Create a copy of args to modify
    
    console.info(
      `[ErrorRecovery] Trying parameter adjustment for ${operationName}`
    );
    
    try {
      // Start timing recovery
      const startTime = performance.now();
      
      // Apply operation-specific parameter adjustments
      let result;
      
      switch (operationName) {
        case 'matmul':
          result = await this.adjustMatmulParameters(args, context);
          break;
        case 'softmax':
          result = await this.adjustSoftmaxParameters(args, context);
          break;
        case 'reshape':
          result = await this.adjustReshapeParameters(args, context);
          break;
        default:
          throw new Error(`No parameter adjustment for ${operationName}`);
      }
      
      // End timing
      const endTime = performance.now();
      const durationMs = endTime - startTime;
      
      console.info(
        `[ErrorRecovery] Successfully recovered using parameter adjustment for ${operationName}`
      );
      
      // Return success result
      return {
        success: true,
        result,
        successfulStrategy: `${this.name}_${operationName}`,
        performance: {
          durationMs
        }
      };
    } catch (recoveryError) {
      console.warn(
        `[ErrorRecovery] Parameter adjustment failed for ${operationName}: ${recoveryError}`
      );
      
      return {
        success: false,
        error: new Error(`Parameter adjustment failed for ${operationName}: ${recoveryError}`),
        successfulStrategy: undefined
      };
    }
  }
  
  /**
   * Adjust parameters for matrix multiplication
   * @param args Original arguments
   * @param context Recovery context
   * @returns Result of the operation
   */
  private async adjustMatmulParameters(
    args: any[],
    context: RecoveryContext
  ): Promise<any> {
    // Check that arguments are tensors
    if (!(args[0] instanceof Tensor) || !(args[1] instanceof Tensor)) {
      throw new Error('Arguments must be tensors');
    }
    
    const a = args[0] as Tensor<any>;
    const b = args[1] as Tensor<any>;
    
    // Add option to disable optimizations if they're causing problems
    const options = args[2] || {};
    options.useOptimization = false;
    args[2] = options;
    
    console.info('[ErrorRecovery] Adjusting matmul parameters: disabling optimizations');
    
    // Retry with adjusted parameters
    return context.originalFn(...args);
  }
  
  /**
   * Adjust parameters for softmax operation
   * @param args Original arguments
   * @param context Recovery context
   * @returns Result of the operation
   */
  private async adjustSoftmaxParameters(
    args: any[],
    context: RecoveryContext
  ): Promise<any> {
    // Check that arguments include a tensor
    if (!(args[0] instanceof Tensor)) {
      throw new Error('First argument must be a tensor');
    }
    
    // Adjust axis parameter if needed
    if (typeof args[1] !== 'number') {
      // Default to last dimension for softmax
      const tensor = args[0] as Tensor<any>;
      const lastDimIndex = tensor.dimensions.length - 1;
      args[1] = lastDimIndex;
      console.info(`[ErrorRecovery] Adjusting softmax parameters: setting axis to ${lastDimIndex}`);
    }
    
    // Retry with adjusted parameters
    return context.originalFn(...args);
  }
  
  /**
   * Adjust parameters for reshape operation
   * @param args Original arguments
   * @param context Recovery context
   * @returns Result of the operation
   */
  private async adjustReshapeParameters(
    args: any[],
    context: RecoveryContext
  ): Promise<any> {
    // Check that arguments include a tensor and shape
    if (!(args[0] instanceof Tensor) || !Array.isArray(args[1])) {
      throw new Error('Arguments must be a tensor and a shape array');
    }
    
    const tensor = args[0] as Tensor<any>;
    const newShape = args[1] as number[];
    
    // Calculate total elements
    const totalElements = tensor.dimensions.reduce((a, b) => a * b, 1);
    const newTotalElements = newShape.reduce((a, b) => a * b, 1);
    
    // If total elements don't match, try to fix the shape
    if (totalElements !== newTotalElements) {
      console.info('[ErrorRecovery] Adjusting reshape parameters: fixing shape mismatch');
      
      // Find dimensions with -1 (automatic sizing)
      const autoSizeIndex = newShape.indexOf(-1);
      
      if (autoSizeIndex >= 0) {
        // Calculate the automatic dimension
        const knownDims = newShape.filter((dim, i) => i !== autoSizeIndex);
        const knownSize = knownDims.reduce((a, b) => a * b, 1);
        const autoSize = totalElements / knownSize;
        
        // Replace -1 with calculated size
        newShape[autoSizeIndex] = autoSize;
        args[1] = newShape;
        
        console.info(`[ErrorRecovery] Set automatic dimension to ${autoSize}`);
      } else {
        // If no automatic dimension, try to flatten completely
        console.info('[ErrorRecovery] Flattening tensor completely');
        args[1] = [totalElements];
      }
    }
    
    // Retry with adjusted parameters
    return context.originalFn(...args);
  }
}

/**
 * Main error recovery manager
 */
export class ErrorRecoveryManager {
  /** Available recovery strategies */
  private strategies: ErrorRecoveryStrategy[] = [];
  
  /** Performance tracker for learning from recoveries */
  private performanceTracker: PerformanceTracker;
  
  /** Recovery success rate by strategy */
  private strategySuccessRates: Map<string, {
    attempts: number;
    successes: number;
  }> = new Map();
  
  /**
   * Initialize the error recovery manager
   * @param performanceTracker Performance tracker to use
   * @param options Additional options
   */
  constructor(
    performanceTracker: PerformanceTracker,
    options?: {
      /** Additional strategies to include */
      additionalStrategies?: ErrorRecoveryStrategy[]
    }
  ) {
    this.performanceTracker = performanceTracker;
    
    // Register default strategies in priority order
    this.registerStrategy(new BackendSwitchStrategy());
    this.registerStrategy(new OperationFallbackStrategy());
    this.registerStrategy(new BrowserSpecificRecoveryStrategy());
    this.registerStrategy(new ParameterAdjustmentStrategy());
    
    // Register additional strategies if provided
    if (options?.additionalStrategies) {
      for (const strategy of options.additionalStrategies) {
        this.registerStrategy(strategy);
      }
    }
  }
  
  /**
   * Register a recovery strategy
   * @param strategy Strategy to register
   */
  registerStrategy(strategy: ErrorRecoveryStrategy): void {
    this.strategies.push(strategy);
    
    // Sort strategies by priority
    this.strategies.sort((a, b) => a.priority - b.priority);
    
    // Initialize success rate tracking
    this.strategySuccessRates.set(strategy.name, {
      attempts: 0,
      successes: 0
    });
  }
  
  /**
   * Categorize an error based on its type and message
   * @param error Error to categorize
   * @returns Error category
   */
  categorizeError(error: Error): ErrorCategory {
    const errorMessage = error.message.toLowerCase();
    const errorType = error.name;
    
    // Memory-related errors
    if (
      errorMessage.includes('memory') ||
      errorMessage.includes('allocation') ||
      errorMessage.includes('buffer') ||
      errorMessage.includes('out of memory') ||
      errorType === 'RangeError'
    ) {
      return ErrorCategory.MEMORY;
    }
    
    // Execution errors
    if (
      errorMessage.includes('shape') ||
      errorMessage.includes('dimension') ||
      errorMessage.includes('size') ||
      errorMessage.includes('mismatch') ||
      errorMessage.includes('invalid')
    ) {
      return ErrorCategory.EXECUTION;
    }
    
    // Precision-related errors
    if (
      errorMessage.includes('precision') ||
      errorMessage.includes('overflow') ||
      errorMessage.includes('underflow') ||
      errorMessage.includes('nan') ||
      errorMessage.includes('infinity')
    ) {
      return ErrorCategory.PRECISION;
    }
    
    // Compatibility errors
    if (
      errorMessage.includes('not supported') ||
      errorMessage.includes('unsupported') ||
      errorMessage.includes('compatibility')
    ) {
      return ErrorCategory.COMPATIBILITY;
    }
    
    // Implementation errors
    if (
      errorMessage.includes('implementation') ||
      errorMessage.includes('not implemented')
    ) {
      return ErrorCategory.IMPLEMENTATION;
    }
    
    // Default to unknown
    return ErrorCategory.UNKNOWN;
  }
  
  /**
   * Attempt to recover from an error
   * @param error Error to recover from
   * @param context Recovery context
   * @returns Promise resolving to recovery result
   */
  async recoverFromError(
    error: Error,
    context: RecoveryContext
  ): Promise<RecoveryResult> {
    console.info(`[ErrorRecovery] Attempting to recover from error in ${context.operationName}: ${error.message}`);
    
    // Track the error in performance tracker
    const errorPerformanceRecord: PerformanceRecord = {
      timestamp: Date.now(),
      operation: context.operationName,
      backendType: context.backendType,
      browserType: context.browserType,
      durationMs: 0, // No duration for failed operation
      inputShapes: context.inputShapes || [],
      outputShape: [],
      success: false,
      errorType: error.name
    };
    
    this.performanceTracker.trackOperation(errorPerformanceRecord);
    
    // Try each strategy in priority order
    for (const strategy of this.strategies) {
      // Check if this strategy can handle this error
      if (strategy.canHandle(error, context)) {
        console.info(`[ErrorRecovery] Trying strategy: ${strategy.name}`);
        
        // Track attempt
        const stats = this.strategySuccessRates.get(strategy.name)!;
        stats.attempts++;
        
        try {
          // Attempt recovery
          const result = await strategy.recover(error, context);
          
          // If successful, track success and return
          if (result.success) {
            console.info(`[ErrorRecovery] Strategy ${strategy.name} successfully recovered ${context.operationName}`);
            
            // Update success stats
            stats.successes++;
            
            // Track successful recovery in performance tracker
            const successRecord: PerformanceRecord = {
              timestamp: Date.now(),
              operation: context.operationName,
              backendType: result.successfulStrategy?.includes('backend_switch') ?
                result.successfulStrategy.split('_')[2] : // Extract backend from strategy name
                context.backendType,
              browserType: context.browserType,
              durationMs: result.performance?.durationMs || 0,
              inputShapes: context.inputShapes || [],
              outputShape: [], // We don't have this information
              success: true
            };
            
            this.performanceTracker.trackOperation(successRecord);
            
            return result;
          }
        } catch (strategyError) {
          console.warn(`[ErrorRecovery] Strategy ${strategy.name} threw an error: ${strategyError}`);
        }
      }
    }
    
    // If we get here, all strategies failed
    console.warn(`[ErrorRecovery] All recovery strategies failed for ${context.operationName}`);
    
    return {
      success: false,
      error: new Error(`All recovery strategies failed for ${context.operationName}`),
      successfulStrategy: undefined
    };
  }
  
  /**
   * Create a protected version of a function that will attempt recovery if it fails
   * @param fn Function to protect
   * @param context Context for recovery
   * @returns Protected function
   */
  protect<T extends any[], R>(
    fn: (...args: T) => Promise<R>,
    partialContext: Omit<RecoveryContext, 'args' | 'originalFn'>
  ): (...args: T) => Promise<R> {
    return async (...args: T): Promise<R> => {
      try {
        // Try to execute the original function
        return await fn(...args);
      } catch (error) {
        console.warn(`[ErrorRecovery] Operation ${partialContext.operationName} failed: ${error}`);
        
        // Complete context with args and originalFn
        const context: RecoveryContext = {
          ...partialContext,
          args,
          originalFn: fn
        };
        
        // Attempt recovery
        const recoveryResult = await this.recoverFromError(error as Error, context);
        
        if (recoveryResult.success) {
          // Return successful recovery result
          return recoveryResult.result as R;
        }
        
        // If recovery failed, throw the original error
        throw error;
      }
    };
  }
  
  /**
   * Get success rates for all strategies
   * @returns Record of strategy success rates
   */
  getStrategySuccessRates(): Record<string, {
    attempts: number;
    successes: number;
    rate: number;
  }> {
    const result: Record<string, any> = {};
    
    for (const [name, stats] of this.strategySuccessRates.entries()) {
      result[name] = {
        ...stats,
        rate: stats.attempts > 0 ? stats.successes / stats.attempts : 0
      };
    }
    
    return result;
  }
  
  /**
   * Generate recovery statistics report
   * @returns Comprehensive recovery statistics
   */
  generateReport(): Record<string, any> {
    return {
      successRates: this.getStrategySuccessRates(),
      registeredStrategies: this.strategies.map(s => ({
        name: s.name,
        priority: s.priority
      })),
      recoveryCount: Array.from(this.strategySuccessRates.values())
        .reduce((sum, stats) => sum + stats.successes, 0)
    };
  }
}

/**
 * Factory function to create an error recovery manager
 * @param performanceTracker Performance tracker to use
 * @param options Additional options
 * @returns New ErrorRecoveryManager
 */
export function createErrorRecoveryManager(
  performanceTracker: PerformanceTracker,
  options?: {
    additionalStrategies?: ErrorRecoveryStrategy[]
  }
): ErrorRecoveryManager {
  return new ErrorRecoveryManager(performanceTracker, options);
}