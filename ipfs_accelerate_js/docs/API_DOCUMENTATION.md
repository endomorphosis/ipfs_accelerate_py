# IPFS Accelerate JavaScript SDK - API Documentation

This document provides detailed API documentation for the IPFS Accelerate JavaScript SDK, which has been fully migrated to TypeScript.

## Table of Contents

- [Core SDK](#core-sdk)
  - [initialize](#initialize)
  - [version](#version)
- [Hardware Abstraction](#hardware-abstraction)
  - [HardwareAbstraction](#hardwareabstraction)
  - [createHardwareAbstraction](#createhardwareabstraction)
- [Models](#models)
  - [createModel](#createmodel)
  - [BertModel](#bertmodel)
  - [VisionModel](#visionmodel)
  - [AudioModel](#audiomodel)
- [Tensor Operations](#tensor-operations)
  - [createTensor](#createtensor)
  - [zeros](#zeros)
  - [ones](#ones)
  - [Tensor](#tensor)
  - [TensorOperations](#tensoroperations)
- [React Integration](#react-integration)
  - [useModel](#usemodel)
  - [useHardwareInfo](#usehardwareinfo)
  - [ModelProcessor](#modelprocessor)
- [Resource Pool](#resource-pool)
  - [ResourcePool](#resourcepool)
  - [createResourcePool](#createresourcepool)
- [Browser Detection](#browser-detection)
  - [detectBrowser](#detectbrowser)
  - [getBrowserCapabilities](#getbrowsercapabilities)
- [Tensor Sharing](#tensor-sharing)
  - [shareTensor](#sharetensor)
  - [getTensor](#gettensor)
  - [releaseTensor](#releasetensor)
- [TypeScript Type Definitions](#typescript-type-definitions)

## Core SDK

### initialize

Initializes the SDK with the specified options.

```typescript
async function initialize(options: {
  logging?: boolean;
  preferredBackends?: string[];
  enableCache?: boolean;
  resourcePoolOptions?: any;
} = {}): Promise<{
  hardware: HardwareAbstraction;
  createModel: (name: string, options?: any) => Promise<any>;
  resourcePool: ResourcePool;
}>
```

**Parameters:**
- `options` (optional): Initialization options
  - `logging` (optional): Enable logging
  - `preferredBackends` (optional): Preferred order of hardware backends
  - `enableCache` (optional): Enable model weight caching
  - `resourcePoolOptions` (optional): Options for the resource pool

**Returns:**
- An object containing initialized SDK components:
  - `hardware`: The hardware abstraction layer
  - `createModel`: A function to create models
  - `resourcePool`: The resource pool

**Example:**
```javascript
const sdk = await initialize({
  logging: true,
  preferredBackends: ['webgpu', 'webnn', 'cpu'],
  enableCache: true
});
```

### version

The version of the SDK.

```typescript
const version: string
```

**Example:**
```javascript
import { version } from 'ipfs-accelerate-js';
console.log('SDK version:', version);
```

## Hardware Abstraction

### HardwareAbstraction

Class providing a unified interface to different hardware backends with proper TypeScript typing.

```typescript
export class HardwareAbstraction {
  constructor(preferences: Partial<HardwarePreferences> = {});
  async initialize(): Promise<boolean>;
  async getPreferredBackend(modelType: string): Promise<HardwareBackend | null>;
  async execute<T = any, U = any>(inputs: T, modelType: string): Promise<U>;
  async runModel<T = any, U = any>(model: Model, inputs: T): Promise<U>;
  dispose(): void;
}
```

#### Constructor

Creates a new HardwareAbstraction instance with the specified preferences.

```typescript
constructor(preferences: Partial<HardwarePreferences> = {})
```

**Parameters:**
- `preferences` (optional): Hardware preferences
  - `backendOrder` (optional): Preferred order of hardware backends
  - `modelPreferences` (optional): Model-specific backend preferences
  - `options` (optional): Additional backend-specific options

**Example:**
```typescript
const hardware = new HardwareAbstraction({
  backendOrder: ['webgpu', 'webnn', 'wasm', 'cpu'],
  modelPreferences: {
    'text': 'webgpu',
    'vision': 'webnn',
    'audio': 'webgpu'
  },
  options: {
    webgpu: {
      preferredAdapterType: 'high-performance'
    },
    webnn: {
      devicePreference: 'gpu'
    }
  }
});
```

#### Methods

##### initialize

Initializes the hardware abstraction layer and detects available backends.

```typescript
async initialize(): Promise<boolean>
```

**Returns:**
- `true` if at least one backend was successfully initialized, `false` otherwise

**Example:**
```typescript
const initialized = await hardware.initialize();
if (initialized) {
  console.log('Hardware acceleration initialized successfully');
} else {
  console.warn('Failed to initialize hardware acceleration');
}
```

##### getPreferredBackend

Gets the preferred backend for a specific model type based on configured preferences.

```typescript
async getPreferredBackend(modelType: string): Promise<HardwareBackend | null>
```

**Parameters:**
- `modelType`: The model type (e.g., 'text', 'vision', 'audio')

**Returns:**
- The preferred hardware backend for the specified model type, or null if no suitable backend is available

**Example:**
```typescript
const backend = await hardware.getPreferredBackend('text');
if (backend) {
  console.log(`Using backend: ${backend.getType()}`);
}
```

##### execute

Executes operations on the preferred backend for a given model type.

```typescript
async execute<T = any, U = any>(inputs: T, modelType: string): Promise<U>
```

**Type Parameters:**
- `T`: Input type
- `U`: Output type

**Parameters:**
- `inputs`: The inputs to execute
- `modelType`: The model type

**Returns:**
- The execution result

**Example:**
```typescript
const result = await hardware.execute({
  input: "Hello, world!"
}, 'text');
```

##### runModel

Runs a model with the specified inputs.

```typescript
async runModel<T = any, U = any>(model: Model, inputs: T): Promise<U>
```

**Type Parameters:**
- `T`: Input type
- `U`: Output type

**Parameters:**
- `model`: The model to run
- `inputs`: The inputs for the model

**Returns:**
- The model output

**Example:**
```typescript
const model = await createModel({
  id: 'bert-base-uncased',
  type: 'text'
}, hardware);

const result = await hardware.runModel(model, {
  input: "Hello, world!"
});
```

##### dispose

Cleans up resources and releases memory.

```typescript
dispose(): void
```

**Example:**
```typescript
// Clean up when done
hardware.dispose();
```

### createHardwareAbstraction

Factory function to create a hardware abstraction layer with the specified options.

```typescript
function createHardwareAbstraction(options: {
  backendOrder?: HardwareBackendType[];
  modelPreferences?: Record<string, HardwareBackendType>;
  webgpuOptions?: WebGPUOptions;
  webnnOptions?: WebNNOptions;
  wasmOptions?: WasmOptions;
  enableLogging?: boolean;
} = {}): Promise<HardwareAbstraction>
```

**Parameters:**
- `options` (optional): Hardware abstraction options
  - `backendOrder` (optional): Preferred order of hardware backends
  - `modelPreferences` (optional): Model-specific backend preferences
  - `webgpuOptions` (optional): WebGPU-specific options
  - `webnnOptions` (optional): WebNN-specific options
  - `wasmOptions` (optional): WebAssembly-specific options
  - `enableLogging` (optional): Enable logging

**Returns:**
- A promise that resolves to a hardware abstraction layer instance

**Example:**
```typescript
const hardware = await createHardwareAbstraction({
  backendOrder: ['webgpu', 'webnn', 'cpu'],
  enableLogging: true,
  webgpuOptions: {
    shaderPrecompilation: true
  }
});
```

## Models

### createModel

Creates a model with the specified name and options.

```typescript
async function createModel(
  modelName: string,
  hardware: HardwareAbstraction,
  options: ModelOptions = {}
): Promise<any>
```

**Parameters:**
- `modelName`: The name of the model to create
- `hardware`: The hardware abstraction layer
- `options` (optional): Model options

**Returns:**
- The created model

**Example:**
```javascript
const model = await createModel('bert-base-uncased', hardware, {
  preferredBackend: 'webgpu'
});
```

### BertModel

BERT model implementation.

#### Methods

##### load

Loads model weights.

```typescript
async load(): Promise<boolean>
```

**Returns:**
- `true` if loading was successful, `false` otherwise

##### predict

Runs inference on input text.

```typescript
async predict(input: BertInput): Promise<BertOutput>
```

**Parameters:**
- `input`: Input for the model
  - `input`: Input text or tokens
  - `tokenTypeIds` (optional): Token type IDs
  - `attentionMask` (optional): Attention mask

**Returns:**
- Model output
  - `lastHiddenState`: Last hidden state tensor
  - `pooledOutput`: Pooled output tensor
  - `hiddenStates` (optional): All hidden states
  - `attentions` (optional): All attention weights
  - `logits` (optional): Classification logits
  - `probabilities` (optional): Classification probabilities
  - `prediction` (optional): Classification prediction

##### isLoaded

Checks if the model is loaded.

```typescript
isLoaded(): boolean
```

**Returns:**
- `true` if the model is loaded, `false` otherwise

##### getType

Gets the model type.

```typescript
getType(): ModelType
```

**Returns:**
- The model type ('text')

##### getName

Gets the model name.

```typescript
getName(): string
```

**Returns:**
- The model name

##### getMetadata

Gets model metadata.

```typescript
getMetadata(): Record<string, any>
```

**Returns:**
- An object containing model metadata

##### dispose

Cleans up resources.

```typescript
dispose(): void
```

## Tensor Operations

### createTensor

Creates a tensor from the specified descriptor and data.

```typescript
function createTensor(
  descriptor: TensorDescriptor,
  data?: ArrayBufferView
): Tensor
```

**Parameters:**
- `descriptor`: Tensor descriptor
  - `dims`: Tensor dimensions
  - `dataType`: Tensor data type
  - `storage` (optional): Tensor storage location
  - `name` (optional): Tensor name
- `data` (optional): Tensor data

**Returns:**
- The created tensor

**Example:**
```javascript
const tensor = createTensor({
  dims: [1, 768],
  dataType: 'float32',
  name: 'embeddings'
}, new Float32Array(768).fill(0.5));
```

### zeros

Creates a tensor filled with zeros.

```typescript
function zeros(
  dims: number[],
  options: {
    dataType?: TensorDataType,
    storage?: TensorStorage,
    name?: string
  } = {}
): Tensor
```

**Parameters:**
- `dims`: Tensor dimensions
- `options` (optional): Tensor options
  - `dataType` (optional): Tensor data type
  - `storage` (optional): Tensor storage location
  - `name` (optional): Tensor name

**Returns:**
- A tensor filled with zeros

**Example:**
```javascript
const zeroTensor = zeros([1, 10], { dataType: 'float32' });
```

### ones

Creates a tensor filled with ones.

```typescript
function ones(
  dims: number[],
  options: {
    dataType?: TensorDataType,
    storage?: TensorStorage,
    name?: string
  } = {}
): Tensor
```

**Parameters:**
- `dims`: Tensor dimensions
- `options` (optional): Tensor options
  - `dataType` (optional): Tensor data type
  - `storage` (optional): Tensor storage location
  - `name` (optional): Tensor name

**Returns:**
- A tensor filled with ones

**Example:**
```javascript
const oneTensor = ones([1, 10], { dataType: 'float32' });
```

### Tensor

Class representing a tensor.

#### Methods

##### getDimensions

Gets tensor dimensions.

```typescript
getDimensions(): number[]
```

**Returns:**
- The tensor dimensions

##### getDataType

Gets tensor data type.

```typescript
getDataType(): TensorDataType
```

**Returns:**
- The tensor data type

##### getSize

Gets tensor size (total number of elements).

```typescript
getSize(): number
```

**Returns:**
- The total number of elements in the tensor

##### getData

Gets tensor data.

```typescript
getData<T extends ArrayBufferView>(): T
```

**Returns:**
- The tensor data as a typed array

##### getName

Gets tensor name.

```typescript
getName(): string
```

**Returns:**
- The tensor name

##### setName

Sets tensor name.

```typescript
setName(name: string): void
```

**Parameters:**
- `name`: The new tensor name

##### getGPUBuffer

Gets the WebGPU buffer for this tensor.

```typescript
getGPUBuffer(): GPUBuffer | null
```

**Returns:**
- The WebGPU buffer or null if not available

##### getWebNNOperand

Gets the WebNN operand for this tensor.

```typescript
getWebNNOperand(): any | null
```

**Returns:**
- The WebNN operand or null if not available

##### copyFromCPU

Copies data to the tensor from CPU.

```typescript
copyFromCPU(data: ArrayBufferView): void
```

**Parameters:**
- `data`: The data to copy

##### dispose

Cleans up resources.

```typescript
dispose(): void
```

## React Integration

### useModel

React hook for using AI models with hardware acceleration in React components.

```typescript
function useModel(options: UseModelOptions): UseModelResult
```

**Type Definitions:**
```typescript
interface UseModelOptions {
  modelId: string;
  modelType?: string;
  autoLoad?: boolean;
  autoHardwareSelection?: boolean;
  fallbackOrder?: HardwareBackendType[];
  config?: Record<string, any>;
}

interface UseModelResult {
  model: Model | null;
  status: 'idle' | 'loading' | 'loaded' | 'error';
  error: Error | null;
  loadModel: () => Promise<void>;
}
```

**Parameters:**
- `options`: Configuration options for the model hook
  - `modelId`: The model ID to load (e.g., 'bert-base-uncased')
  - `modelType` (optional): The model type (e.g., 'text', 'vision', 'audio')
  - `autoLoad` (optional): Whether to automatically load the model (default: true)
  - `autoHardwareSelection` (optional): Whether to automatically select hardware backend (default: true)
  - `fallbackOrder` (optional): Order of hardware backends to try
  - `config` (optional): Additional model-specific configuration

**Returns:**
- An object containing:
  - `model`: The loaded model, or null if not loaded
  - `status`: Current status ('idle', 'loading', 'loaded', 'error')
  - `error`: Error object if loading failed, null otherwise
  - `loadModel`: Function to manually load the model

**Example:**
```tsx
import React, { useState } from 'react';
import { useModel } from 'ipfs-accelerate/react';

function BertComponent() {
  const { model, status, error, loadModel } = useModel({
    modelId: 'bert-base-uncased',
    modelType: 'text',
    autoLoad: true
  });

  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (model && status === 'loaded') {
      const output = await model.execute({ input });
      setResult(output);
    }
  };

  if (status === 'loading') return <div>Loading model...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Enter text"
        />
        <button type="submit">Process</button>
      </form>
      {result && (
        <pre>{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
}
```

### useHardwareInfo

React hook for getting information about available hardware capabilities.

```typescript
function useHardwareInfo(): UseHardwareInfoResult
```

**Type Definitions:**
```typescript
interface UseHardwareInfoResult {
  capabilities: HardwareCapabilities | null;
  isReady: boolean;
  optimalBackend: string;
  error: Error | null;
}

interface HardwareCapabilities {
  browserName: string;
  browserVersion: string;
  platform: string;
  osVersion: string;
  isMobile: boolean;
  webgpuSupported: boolean;
  webgpuFeatures: string[];
  webnnSupported: boolean;
  webnnFeatures: string[];
  wasmSupported: boolean;
  wasmFeatures: string[];
  recommendedBackend: HardwareBackendType;
  memoryLimitMB: number;
}
```

**Returns:**
- An object containing:
  - `capabilities`: Hardware capabilities information, or null if not yet detected
  - `isReady`: Whether hardware detection is complete
  - `optimalBackend`: The optimal backend for the current hardware
  - `error`: Error object if detection failed, null otherwise

**Example:**
```tsx
import React from 'react';
import { useHardwareInfo } from 'ipfs-accelerate/react';

function HardwareInfoComponent() {
  const { capabilities, isReady, optimalBackend, error } = useHardwareInfo();

  if (!isReady) return <div>Detecting hardware capabilities...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      <h2>Hardware Capabilities</h2>
      <p>Optimal Backend: {optimalBackend}</p>
      <p>Browser: {capabilities.browserName} {capabilities.browserVersion}</p>
      <p>WebGPU Support: {capabilities.webgpuSupported ? '✅' : '❌'}</p>
      <p>WebNN Support: {capabilities.webnnSupported ? '✅' : '❌'}</p>
      <p>WASM Support: {capabilities.wasmSupported ? '✅' : '❌'}</p>
      <p>Memory Limit: {capabilities.memoryLimitMB}MB</p>
    </div>
  );
}
```

### ModelProcessor

React component for easy model processing with automatic state management.

```typescript
function ModelProcessor(props: ModelProcessorProps): React.ReactNode
```

**Type Definitions:**
```typescript
interface ModelProcessorProps {
  modelId: string;
  modelType?: string;
  input: any;
  onResult?: (result: any) => void;
  onError?: (error: Error) => void;
  children: (props: {
    result: any;
    loading: boolean;
    error: Error | null;
  }) => React.ReactNode;
}
```

**Parameters:**
- `props`: Component props
  - `modelId`: The model ID to load (e.g., 'bert-base-uncased')
  - `modelType` (optional): The model type (e.g., 'text', 'vision', 'audio')
  - `input`: Input data to process
  - `onResult` (optional): Callback function when processing completes
  - `onError` (optional): Callback function when an error occurs
  - `children`: Render prop function that receives processing state

**Example:**
```tsx
import React, { useState } from 'react';
import { ModelProcessor } from 'ipfs-accelerate/react';

function TextProcessorComponent() {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Enter text"
        />
        <button type="submit">Process</button>
      </form>

      <ModelProcessor
        modelId="bert-base-uncased"
        modelType="text"
        input={input ? { input } : null}
        onResult={result => console.log('Result:', result)}
        onError={error => console.error('Error:', error)}
      >
        {({ result, loading, error }) => (
          <div>
            {loading && <div>Processing...</div>}
            {error && <div>Error: {error.message}</div>}
            {result && (
              <pre>{JSON.stringify(result, null, 2)}</pre>
            )}
          </div>
        )}
      </ModelProcessor>
    </div>
  );
}
```

## Resource Pool

### createResourcePool

Creates a resource pool with the specified options.

```typescript
async function createResourcePool(
  options: ResourcePoolOptions = {}
): Promise<ResourcePool>
```

**Parameters:**
- `options` (optional): Resource pool options
  - `maxResourcesPerType` (optional): Maximum number of resources per type
  - `maxTotalResources` (optional): Maximum total resources
  - `idleTimeout` (optional): Resource idle timeout (ms)
  - `logging` (optional): Enable logging

**Returns:**
- A resource pool instance

**Example:**
```javascript
const pool = await createResourcePool({
  maxResourcesPerType: 4,
  maxTotalResources: 16,
  idleTimeout: 60000, // 1 minute
  logging: true
});
```

### ResourcePool

Class for managing browser resources.

#### Methods

##### initialize

Initializes the resource pool.

```typescript
async initialize(): Promise<boolean>
```

**Returns:**
- `true` if initialization was successful, `false` otherwise

##### registerFactory

Registers a resource factory.

```typescript
registerFactory<T>(resourceType: string, factory: (options?: any) => Promise<T>): void
```

**Parameters:**
- `resourceType`: The resource type
- `factory`: The factory function

##### registerCleanup

Registers a resource cleanup function.

```typescript
registerCleanup<T>(resourceType: string, cleanup: (resource: T) => void): void
```

**Parameters:**
- `resourceType`: The resource type
- `cleanup`: The cleanup function

##### acquireResource

Acquires a resource from the pool.

```typescript
async acquireResource<T>(resourceType: string, options: any = {}): Promise<T | null>
```

**Parameters:**
- `resourceType`: The resource type
- `options` (optional): Resource-specific options

**Returns:**
- The requested resource or null if unavailable

##### releaseResource

Releases a resource back to the pool.

```typescript
async releaseResource<T>(resource: T): Promise<void>
```

**Parameters:**
- `resource`: The resource to release

##### getStats

Gets statistics about resource usage.

```typescript
getStats(): ResourceStats
```

**Returns:**
- An object containing resource usage statistics

##### dispose

Cleans up all resources.

```typescript
dispose(): void
```

## Browser Detection

### detectBrowser

Detects current browser information.

```typescript
function detectBrowser(): BrowserInfo
```

**Returns:**
- An object containing browser information

**Example:**
```javascript
const browserInfo = detectBrowser();
console.log('Browser:', browserInfo.name);
console.log('Version:', browserInfo.version);
console.log('Mobile:', browserInfo.isMobile);
```

### getBrowserCapabilities

Gets comprehensive browser capabilities.

```typescript
async function getBrowserCapabilities(): Promise<{
  browser: BrowserInfo;
  webgpu: any | null;
  webnn: any | null;
}>
```

**Returns:**
- An object containing detailed browser capabilities

**Example:**
```javascript
const capabilities = await getBrowserCapabilities();
console.log('WebGPU support:', capabilities.webgpu?.supported);
console.log('WebNN support:', capabilities.webnn?.supported);
```