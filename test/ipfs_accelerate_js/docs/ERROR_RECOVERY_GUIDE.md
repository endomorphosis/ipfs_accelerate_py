# Error Recovery System Guide

The Error Recovery System provides intelligent error handling for WebGPU/WebNN operations, making your machine learning applications more robust and resilient. By leveraging performance data and applying multiple recovery strategies, the system automatically handles common errors and ensures operations succeed whenever possible.

## Overview

The Error Recovery System is designed to:

1. **Detect and categorize errors** to apply appropriate recovery strategies
2. **Attempt multiple recovery approaches** in priority order
3. **Learn from recovery attempts** to improve future success rates
4. **Provide browser-specific optimizations** for better recovery rates
5. **Track statistics and performance data** to guide recovery decisions

## Key Components

### ErrorRecoveryManager

The central component that coordinates recovery strategies and handles error recovery attempts:

```typescript
import { createErrorRecoveryManager, ErrorRecoveryManager } from 'ipfs-accelerate';

// Create an error recovery manager
const errorRecoveryManager = createErrorRecoveryManager(performanceTracker);

// Use it to protect functions that might fail
const protectedFn = errorRecoveryManager.protect(originalFn, context);
```

### Recovery Strategies

The system includes several recovery strategies, applied in priority order:

1. **BackendSwitchStrategy**: Switches to an alternative backend (WebGPU, WebNN, CPU) when an operation fails
2. **OperationFallbackStrategy**: Uses alternative implementations for critical operations
3. **BrowserSpecificRecoveryStrategy**: Applies browser-specific optimizations for Chrome, Firefox, Edge, and Safari
4. **ParameterAdjustmentStrategy**: Adjusts operation parameters to resolve compatibility issues

### Error Categories

Errors are categorized to help determine the most appropriate recovery actions:

- `MEMORY`: Memory allocation or limit errors
- `EXECUTION`: Execution errors (e.g., shape mismatches)
- `PRECISION`: Precision-related errors (e.g., overflow)
- `COMPATIBILITY`: Hardware compatibility errors
- `IMPLEMENTATION`: Backend-specific implementation errors

## Integration with Hardware Abstraction Layer

The Error Recovery System integrates with the Hardware Abstraction Layer (HAL) and leverages the Performance Tracking System for data-driven recovery decisions.

### Getting Started

1. **Import the needed components**:

```typescript
import { 
  createHardwareAbstractionLayer,
  createErrorRecoveryManager,
  BackendType
} from 'ipfs-accelerate';
```

2. **Create the Hardware Abstraction Layer**:

```typescript
const hal = createHardwareAbstractionLayer({
  backends: availableBackends,
  defaultBackend: 'webgpu',
  autoInitialize: true
});

await hal.initialize();
```

3. **Create the Error Recovery Manager**:

```typescript
// Get performance tracker from HAL
const performanceTracker = hal.performanceTracker;

// Create error recovery manager
const errorRecoveryManager = createErrorRecoveryManager(performanceTracker);
```

4. **Protect operations with error recovery**:

```typescript
// Original operation function
const originalMatmul = async <T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>> => {
  return hal.matmul(a, b);
};

// Create protected version
const protectedMatmul = errorRecoveryManager.protect(
  originalMatmul,
  {
    operationName: 'matmul',
    backendType: hal.getBackendType() as BackendType,
    availableBackends: hal.backends,
    activeBackend: hal.getActiveBackend()!,
    performanceTracker,
    setActiveBackend: (backend) => hal.setActiveBackend(backend),
    browserType: 'chrome',
    useBrowserOptimizations: true
  }
);

// Use the protected function
try {
  const result = await protectedMatmul(tensorA, tensorB);
  console.log('Operation succeeded!');
} catch (error) {
  console.error('All recovery attempts failed:', error);
}
```

## Recovery Statistics and Reporting

The Error Recovery System tracks statistics about recovery attempts and success rates:

```typescript
// Get recovery statistics
const stats = errorRecoveryManager.getStrategySuccessRates();

// Generate comprehensive report
const report = errorRecoveryManager.generateReport();
```

## Browser-Specific Optimizations

The system includes specialized recovery techniques for different browsers:

- **Firefox**: Optimized for audio models with specialized workgroup sizes and compute shader settings
- **Chrome**: General optimizations with focus on vision models
- **Edge**: Specialized optimizations for WebNN operations
- **Safari**: Compatibility fixes for WebGPU implementation differences

## Custom Recovery Strategies

You can extend the system with custom recovery strategies:

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
    // ...

    return {
      success: true,
      result: recoveredResult,
      successfulStrategy: this.name
    };
  }
}

// Register the custom strategy
const errorRecoveryManager = createErrorRecoveryManager(performanceTracker, {
  additionalStrategies: [new CustomRecoveryStrategy()]
});
```

## Performance Considerations

- The Error Recovery System adds a small overhead to operations but significantly improves robustness
- Recovery attempts are prioritized by likelihood of success based on historical data
- Failed recovery attempts are tracked to avoid repeating unsuccessful strategies
- When a recovery strategy succeeds, that information is used to optimize future recovery attempts

## Example Use Cases

### High-Availability Applications

For applications requiring maximum uptime, wrap critical operations in error recovery:

```typescript
const criticalOperation = errorRecoveryManager.protect(
  originalOperation,
  recoveryContext
);
```

### Handling WebGPU Limitations in Different Browsers

The system automatically adapts to browser-specific limitations:

```typescript
// This will automatically apply browser-specific optimizations
const browserAwareOperation = errorRecoveryManager.protect(
  originalOperation,
  {
    ...recoveryContext,
    browserType: detectBrowserType(),
    useBrowserOptimizations: true
  }
);
```

### Fallback Chains for Memory-Intensive Operations

For operations that might encounter memory limits:

```typescript
// Will automatically try alternative backends if operation fails due to memory constraints
const memoryResilientOperation = errorRecoveryManager.protect(
  originalOperation,
  recoveryContext
);
```

## Conclusion

The Error Recovery System makes your machine learning applications more resilient by providing intelligent error handling with multiple recovery strategies. By leveraging performance data and browser-specific optimizations, the system can recover from many common errors, ensuring a smoother user experience even in challenging environments.

For more details, see the [full API documentation](./api/error_recovery_api.md) and the [example code](../examples/error_recovery_example.ts).