# IPFS Accelerate JavaScript SDK API Reference

This document provides a comprehensive reference for the IPFS Accelerate JavaScript SDK API.

## Table of Contents

- [Core Tensor API](#core-tensor-api)
  - [Tensor](#tensor)
  - [Creation Functions](#creation-functions)
  - [Tensor Operations](#tensor-operations)
- [SharedTensor API](#sharedtensor-api)
  - [SharedTensor](#sharedtensor)
  - [TensorSharingManager](#tensorsharingmanager)
- [Hardware Acceleration API](#hardware-acceleration-api)
  - [HardwareBackend Interface](#hardwarebackend-interface)
  - [Backend Creation](#backend-creation)
  - [WebGPU Backend](#webgpu-backend)
  - [WebNN Backend](#webnn-backend)
- [Hardware Detection API](#hardware-detection-api)
  - [General Hardware Detection](#general-hardware-detection)
  - [WebNN Features Detection](#webnn-features-detection)
- [Examples](#examples)

## Core Tensor API

### Tensor

The `Tensor` class is the core data structure representing n-dimensional arrays.

```typescript
class Tensor<T = number> {
  // Properties
  readonly shape: number[];          // Dimensions of the tensor
  readonly data: T[];                // Data array
  readonly dataType: string;         // Data type
  readonly backend: string;          // Backend used for operations
  readonly requiresGrad: boolean;    // Whether gradient computation is required
  readonly device: string;           // Device the tensor is stored on
  
  // Constructor
  constructor(shape: number[], data?: T[] | null, options?: TensorOptions);
  
  // Basic properties
  get size(): number;                // Total number of elements
  get rank(): number;                // Number of dimensions
  
  // Methods
  clone(): Tensor<T>;                // Create a copy
  zeros(): Tensor<T>;                // Create a tensor of same shape filled with zeros
  ones(): Tensor<T>;                 // Create a tensor of same shape filled with ones
  toString(): string;                // String representation
  get(...indices: number[]): T;      // Get value at specified indices
  set(value: T, ...indices: number[]): void; // Set value at specified indices
}

interface TensorOptions {
  dataType?: 'float32' | 'int32' | 'float64' | 'int64' | 'uint8' | 'bool';
  backend?: 'cpu' | 'webgpu' | 'webnn' | 'wasm';
  device?: string;
  requiresGrad?: boolean;
}
```

### Creation Functions

Helper functions for creating tensors:

```typescript
// Create tensor filled with zeros
function zeros<T>(shape: number[], options?: TensorOptions): Tensor<T>;

// Create tensor filled with ones
function ones<T>(shape: number[], options?: TensorOptions): Tensor<T>;

// Create tensor with range of values
function range(start: number, end: number, step?: number, options?: TensorOptions): Tensor<number>;

// Create tensor with random values
function random(shape: number[], min?: number, max?: number, options?: TensorOptions): Tensor<number>;
```

### Tensor Operations

Common operations available on tensors:

```typescript
// Matrix operations
function matmul<T>(a: Tensor<T>, b: Tensor<T>): Tensor<T>;
function transpose<T>(tensor: Tensor<T>): Tensor<T>;
function reshape<T>(tensor: Tensor<T>, newShape: number[]): Tensor<T>;

// Element-wise operations
function add<T>(a: Tensor<T>, b: Tensor<T>): Tensor<T>;
function subtract<T>(a: Tensor<T>, b: Tensor<T>): Tensor<T>;
function multiply<T>(a: Tensor<T>, b: Tensor<T>): Tensor<T>;
function divide<T>(a: Tensor<T>, b: Tensor<T>): Tensor<T>;

// Neural network operations
function relu<T>(tensor: Tensor<T>): Tensor<T>;
function sigmoid<T>(tensor: Tensor<T>): Tensor<T>;
function tanh<T>(tensor: Tensor<T>): Tensor<T>;
function softmax<T>(tensor: Tensor<T>, axis?: number): Tensor<T>;
```

## SharedTensor API

### SharedTensor

`SharedTensor` extends `Tensor` with reference counting for cross-model sharing:

```typescript
class SharedTensor<T = number> extends Tensor<T> {
  // Properties
  readonly refCount: number;          // Current reference count
  readonly producerModel: string;     // The model that created this tensor
  readonly consumerModels: string[];  // Models consuming this tensor
  readonly id: string;                // Unique ID for this shared tensor
  
  // Methods
  retain(): void;                     // Increment reference count
  release(): void;                    // Decrement reference count
  isShared(): boolean;                // Whether tensor is shared by multiple models
}
```

### TensorSharingManager

The `TensorSharingManager` manages tensors shared between models:

```typescript
class TensorSharingManager {
  // Constructor
  constructor(maxMemoryMB: number);
  
  // Methods
  registerSharedTensor(
    name: string,
    shape: number[],
    backend: string,
    producerModel: string,
    consumerModels: string[],
    dataType?: string
  ): SharedTensor;
  
  getSharedTensor(id: string): SharedTensor | null;
  releaseSharedTensor(id: string): void;
  optimizeMemoryUsage(): {
    memory_before: number;
    memory_after: number;
    memory_reduction_percent: number;
    tensors_released: number;
  };
  
  // Properties
  readonly activeTensors: SharedTensor[];
  readonly totalMemoryUsage: number;
}
```

## Hardware Acceleration API

### HardwareBackend Interface

The common interface for all hardware backends:

```typescript
interface HardwareBackend {
  // Properties
  readonly id: string;                // Unique identifier
  readonly type: string;              // Type (webgpu, webnn, etc.)
  readonly isAvailable: boolean;      // Whether available in current environment
  readonly capabilities: HardwareCapabilities; // Backend capabilities
  
  // Methods
  initialize(): Promise<void>;        // Initialize the backend
  isInitialized(): boolean;           // Check if initialized
  allocateTensor<T>(tensor: Tensor<T>): Promise<void>; // Allocate tensor on backend
  releaseTensor<T>(tensor: Tensor<T>): void; // Release tensor from backend
  
  // Tensor operations
  add<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>>;
  subtract<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>>;
  multiply<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>>;
  divide<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>>;
  matmul<T>(a: Tensor<T>, b: Tensor<T>): Promise<Tensor<T>>;
  transpose<T>(tensor: Tensor<T>): Promise<Tensor<T>>;
  relu<T>(tensor: Tensor<T>): Promise<Tensor<T>>;
  sigmoid<T>(tensor: Tensor<T>): Promise<Tensor<T>>;
  tanh<T>(tensor: Tensor<T>): Promise<Tensor<T>>;
  softmax<T>(tensor: Tensor<T>, axis?: number): Promise<Tensor<T>>;
  reshape<T>(tensor: Tensor<T>, newShape: number[]): Promise<Tensor<T>>;
  
  // Synchronization and cleanup
  sync(): Promise<void>;              // Synchronize backend execution
  dispose(): void;                    // Free all resources
}

interface HardwareCapabilities {
  maxDimensions: number;              // Maximum supported tensor dimensions
  maxMatrixSize: number;              // Maximum matrix size for efficient multiplication
  supportedDataTypes: string[];       // Supported data types
  availableMemory?: number;           // Available memory in bytes
  supportsAsync: boolean;             // Whether async execution is supported
  supportedOperations: {              // Supported operations
    basicArithmetic: boolean;
    matrixMultiplication: boolean;
    convolution: boolean;
    reduction: boolean;
    activation: boolean;
  };
}
```

### Backend Creation

Helper functions for creating hardware backends:

```typescript
// Create optimal backend based on hardware detection
function createOptimalBackend(preferences?: HardwarePreference): Promise<HardwareBackend>;

// Create WebGPU backend
function createWebGPUBackend(): Promise<WebGPUBackend>;

// Create WebNN backend
function createWebNNBackend(): Promise<WebNNBackend>;

// Create multi-backend with fallback
function createMultiBackend(backends?: ('webgpu' | 'webnn' | 'wasm' | 'cpu')[]): Promise<HardwareBackend>;

// Hardware preference options
interface HardwarePreference {
  backend?: string;                   // Preferred backend
  preferSpeed?: boolean;              // Whether to prefer speed over memory usage
  preferLowPower?: boolean;           // Whether to prefer low power consumption
  maxMemoryUsage?: number;            // Maximum memory usage in bytes
}
```

### WebGPU Backend

The `WebGPUBackend` class provides hardware acceleration using WebGPU:

```typescript
class WebGPUBackend implements HardwareBackend {
  // Implementation of HardwareBackend interface
  
  // Additional WebGPU-specific methods
  getStats(): {                       // Get buffer allocation statistics
    totalAllocated: number;
    currentlyAllocated: number;
    reused: number;
    created: number;
  };
  
  garbageCollect(maxBuffersPerSize?: number): void; // Clean up unused buffers
}
```

### WebNN Backend

The `WebNNBackend` class provides neural network acceleration using WebNN:

```typescript
class WebNNBackend implements HardwareBackend {
  // Implementation of HardwareBackend interface
  
  // Additional WebNN-specific properties and methods
  // (Graph management, operation compilation, etc.)
}
```

## Hardware Detection API

### General Hardware Detection

APIs for detecting hardware capabilities:

```typescript
// Detect hardware capabilities
function detectHardware(): Promise<DetectedHardware>;

// Optimize hardware selection based on detected hardware
function optimizeHardwareSelection(
  detected: DetectedHardware,
  preferences?: HardwarePreference
): string;

// Hardware detection result
interface DetectedHardware {
  hasWebGPU: boolean;                 // Whether WebGPU is available
  hasWebNN: boolean;                  // Whether WebNN is available
  hasWebGL: boolean;                  // Whether WebGL is available
  hasWasmSimd: boolean;               // Whether WASM SIMD is available
  webGPUDeviceName?: string;          // WebGPU device name
  webGPUCapabilities?: HardwareCapabilities; // WebGPU device capabilities
  webNNCapabilities?: HardwareCapabilities; // WebNN device capabilities
  browser: {                          // Browser information
    name: string;
    version: string;
    isMobile: boolean;
  };
  os: {                               // OS information
    name: string;
    version: string;
  };
  cpu: {                              // CPU information
    cores: number;
    architecture?: string;
  };
}
```

### WebNN Features Detection

APIs for detecting WebNN-specific features:

```typescript
// Detect WebNN features
function detectWebNNFeatures(): Promise<WebNNFeatures>;

// Check if the device has a neural processor
function hasNeuralProcessor(): boolean;

// Get optimal power preference for WebNN
function getOptimalPowerPreference(): 'default' | 'high-performance' | 'low-power';

// WebNN features detection result
interface WebNNFeatures {
  supported: boolean;                 // Whether WebNN is supported
  version?: string;                   // WebNN API version
  hardwareAccelerated: boolean;       // Whether hardware acceleration is available
  supportedModels: string[];          // Supported model formats
  supportedOperations: {              // Supported operations
    basic: boolean;
    conv2d: boolean;
    pool: boolean;
    normalization: boolean;
    recurrent: boolean;
    transformer: boolean;
  };
  browser: {                          // Browser information
    name: string;
    version: string;
  };
  accelerationType?: 'cpu' | 'gpu' | 'npu' | 'dsp' | 'unknown'; // Acceleration type
  preferredPowerPreference?: 'default' | 'high-performance' | 'low-power'; // Power preference
}
```

## Examples

For practical examples of using the API, see:

- [Basic Tensor Example](../../examples/tensor_matrix_example.ts)
- [Tensor Sharing Example](../../examples/tensor_sharing_example.ts)
- [WebGPU Example](../../examples/webgpu_tensor_example.ts)
- [WebNN Example](../../examples/webnn_tensor_example.ts)