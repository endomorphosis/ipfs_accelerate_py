# Error Recovery System API Reference

This document provides a detailed reference for the Error Recovery System API in the IPFS Accelerate JavaScript SDK.

## Core Components

### ErrorRecoveryManager

The central component that coordinates recovery strategies and handles error recovery attempts.

#### Creation

```typescript
import { createErrorRecoveryManager } from 'ipfs-accelerate';

const errorRecoveryManager = createErrorRecoveryManager(
  performanceTracker,
  options?: {
    additionalStrategies?: ErrorRecoveryStrategy[]
  }
);
```

#### Methods

| Method | Description |
|--------|-------------|
| `protect<T extends any[], R>(fn: (...args: T) => Promise<R>, context: RecoveryContext): (...args: T) => Promise<R>` | Creates a protected version of a function that will attempt recovery if it fails |
| `recoverFromError(error: Error, context: RecoveryContext): Promise<RecoveryResult>` | Attempts to recover from an error |
| `categorizeError(error: Error): ErrorCategory` | Categorizes an error based on its type and message |
| `registerStrategy(strategy: ErrorRecoveryStrategy): void` | Registers a new recovery strategy |
| `getStrategySuccessRates(): Record<string, { attempts: number; successes: number; rate: number; }>` | Gets success rates for all registered strategies |
| `generateReport(): Record<string, any>` | Generates a comprehensive report of recovery statistics |

### RecoveryContext

Context information provided to recovery strategies to help them understand and recover from errors.

```typescript
interface RecoveryContext {
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
```

### RecoveryResult

Result returned by recovery strategies, indicating whether recovery was successful.

```typescript
interface RecoveryResult {
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
```

### ErrorCategory

Enum for categorizing errors to help determine the most appropriate recovery actions.

```typescript
enum ErrorCategory {
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
```

## Recovery Strategies

### ErrorRecoveryStrategy Interface

Interface that all recovery strategies must implement.

```typescript
interface ErrorRecoveryStrategy {
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
```

### Built-in Recovery Strategies

The Error Recovery System includes several built-in strategies:

#### BackendSwitchStrategy

Switches to an alternative backend when an operation fails.

```typescript
// Automatically registered with the ErrorRecoveryManager
// Priority: 1 (highest priority, tried first)

// This strategy tries to execute the operation on different hardware backends,
// based on availability and performance history
```

#### OperationFallbackStrategy

Uses alternative implementations for operations.

```typescript
// Automatically registered with the ErrorRecoveryManager
// Priority: 2

// Supported operations:
// - matmul: Alternative matrix multiplication implementation
// - transpose: Alternative transpose implementation
// - gelu: Alternative GELU activation implementation
// - layerNorm: Alternative layer normalization implementation
// - softmax: Alternative softmax implementation
```

#### BrowserSpecificRecoveryStrategy

Applies browser-specific optimizations for different browsers.

```typescript
// Automatically registered with the ErrorRecoveryManager
// Priority: 3

// Applies optimizations specific to:
// - Firefox: Audio model optimizations with specialized workgroup sizes
// - Chrome: Vision model optimizations
// - Edge: WebNN optimizations
// - Safari: Compatibility fixes
```

#### ParameterAdjustmentStrategy

Adjusts operation parameters to resolve compatibility issues.

```typescript
// Automatically registered with the ErrorRecoveryManager
// Priority: 4

// Supported operations:
// - matmul: Adjusts optimization parameters
// - softmax: Adjusts axis parameter
// - reshape: Adjusts reshape parameters to fix dimension mismatches
```

## Creating Custom Recovery Strategies

You can extend the system with custom recovery strategies by implementing the `ErrorRecoveryStrategy` interface.

### Example: Custom Recovery Strategy

```typescript
import { ErrorRecoveryStrategy, RecoveryContext, RecoveryResult } from 'ipfs-accelerate';

class CustomRecoveryStrategy implements ErrorRecoveryStrategy {
  readonly name = 'custom_strategy';
  readonly priority = 10; // Higher priority = later execution

  canHandle(error: Error, context: RecoveryContext): boolean {
    // Determine if this strategy can handle the error
    return error.message.includes('specific error message');
  }

  async recover(error: Error, context: RecoveryContext): Promise<RecoveryResult> {
    // Implement custom recovery logic
    try {
      // Example: modify arguments and retry
      const modifiedArgs = context.args.map(arg => /* modify arg */);
      const result = await context.originalFn(...modifiedArgs);
      
      return {
        success: true,
        result,
        successfulStrategy: this.name,
        performance: {
          durationMs: 0 // Calculate actual duration
        }
      };
    } catch (recoveryError) {
      return {
        success: false,
        error: new Error(`Custom recovery failed: ${recoveryError}`),
        successfulStrategy: undefined
      };
    }
  }
}

// Register the custom strategy
const errorRecoveryManager = createErrorRecoveryManager(performanceTracker, {
  additionalStrategies: [new CustomRecoveryStrategy()]
});
```

## Integration with Hardware Abstraction Layer

Example of integrating the Error Recovery System with the Hardware Abstraction Layer:

```typescript
import { 
  createHardwareAbstractionLayer,
  createErrorRecoveryManager,
  BackendType
} from 'ipfs-accelerate';

// Create HAL
const hal = createHardwareAbstractionLayer({
  backends: availableBackends,
  defaultBackend: 'webgpu',
  autoInitialize: true
});

await hal.initialize();

// Create error recovery manager
const performanceTracker = hal.performanceTracker;
const errorRecoveryManager = createErrorRecoveryManager(performanceTracker);

// Protect critical operations
const operations = {
  matmul: hal.matmul.bind(hal),
  transpose: hal.transpose.bind(hal),
  relu: hal.relu.bind(hal),
  sigmoid: hal.sigmoid.bind(hal),
  // ... other operations
};

const protectedOperations = {};

// Create protected versions of all operations
for (const [name, op] of Object.entries(operations)) {
  protectedOperations[name] = errorRecoveryManager.protect(
    op,
    {
      operationName: name,
      backendType: hal.getBackendType() as BackendType,
      availableBackends: hal.backends,
      activeBackend: hal.getActiveBackend()!,
      performanceTracker,
      setActiveBackend: (backend) => hal.setActiveBackend(backend),
      browserType: 'chrome',
      useBrowserOptimizations: true
    }
  );
}

// Use protected operations
try {
  const result = await protectedOperations.matmul(tensorA, tensorB);
  // ...
} catch (error) {
  // All recovery attempts failed
  console.error('Operation failed despite recovery attempts:', error);
}
```

## Recovery Statistics and Analysis

The Error Recovery System provides statistics and analysis to help you understand how recovery is working:

```typescript
// Get recovery statistics
const stats = errorRecoveryManager.getStrategySuccessRates();
console.log('Recovery strategy success rates:', stats);
/*
{
  "backend_switch": {
    "attempts": 15,
    "successes": 12,
    "rate": 0.8
  },
  "operation_fallback": {
    "attempts": 8,
    "successes": 5,
    "rate": 0.625
  },
  "browser_specific": {
    "attempts": 6,
    "successes": 4,
    "rate": 0.6667
  },
  "parameter_adjustment": {
    "attempts": 4,
    "successes": 3,
    "rate": 0.75
  }
}
*/

// Generate comprehensive report
const report = errorRecoveryManager.generateReport();
console.log('Recovery report:', report);
/*
{
  "successRates": {
    // Same as getStrategySuccessRates()
  },
  "registeredStrategies": [
    {"name": "backend_switch", "priority": 1},
    {"name": "operation_fallback", "priority": 2},
    {"name": "browser_specific", "priority": 3},
    {"name": "parameter_adjustment", "priority": 4}
  ],
  "recoveryCount": 24 // Total successful recoveries
}
*/
```

## Performance Data Integration

The Error Recovery System leverages performance data from the Performance Tracking System to make intelligent recovery decisions:

```typescript
// Get performance data
const performanceData = performanceTracker.exportPerformanceData();

// Get backend recommendations based on performance
const recommendations = performanceTracker.getAllRecommendations();
```

This integration enables smart backend selection during recovery, prioritizing backends that have historically performed better for specific operations.

## Error Categorization

The system automatically categorizes errors to help determine the most appropriate recovery actions:

```typescript
// Memory errors
const memoryError = new Error('Out of memory error occurred');
errorRecoveryManager.categorizeError(memoryError); // ErrorCategory.MEMORY

// Execution errors
const executionError = new Error('Invalid shape for operation');
errorRecoveryManager.categorizeError(executionError); // ErrorCategory.EXECUTION

// Precision errors
const precisionError = new Error('Precision loss detected');
errorRecoveryManager.categorizeError(precisionError); // ErrorCategory.PRECISION

// Compatibility errors
const compatibilityError = new Error('Operation not supported on this backend');
errorRecoveryManager.categorizeError(compatibilityError); // ErrorCategory.COMPATIBILITY

// Implementation errors
const implementationError = new Error('Implementation not available');
errorRecoveryManager.categorizeError(implementationError); // ErrorCategory.IMPLEMENTATION

// Unknown errors
const unknownError = new Error('Something went wrong');
errorRecoveryManager.categorizeError(unknownError); // ErrorCategory.UNKNOWN
```

## Browser-Specific Recovery

The system includes specialized recovery techniques for different browsers:

```typescript
// Create a context with browser information
const context: RecoveryContext = {
  operationName: 'matmul',
  // ... other required fields
  browserType: 'firefox', // or 'chrome', 'edge', 'safari'
  useBrowserOptimizations: true
};

// Firefox optimizations
// - Audio model optimizations with specialized workgroup sizes
// - Compute shader optimizations for audio processing

// Chrome optimizations
// - Vision model optimizations
// - General WebGPU optimizations

// Edge optimizations
// - WebNN optimizations
// - Text model optimizations

// Safari optimizations
// - Compatibility fixes for WebGPU implementation
```

For more information on browser-specific optimizations, see the [Browser Optimizations Guide](../BROWSER_OPTIMIZATIONS.md).

## See Also

- [Error Recovery Guide](../ERROR_RECOVERY_GUIDE.md) - Overview and getting started guide
- [Performance Tracking API](./performance_tracking_api.md) - API for tracking hardware operation performance
- [Hardware Abstraction Layer API](./hal_api.md) - API for hardware abstraction layer
- [Example Code](../../examples/error_recovery_example.ts) - Complete example of error recovery system