# IPFS Accelerate JS - Developer Guide

This document provides detailed information for developers who want to use, contribute to, or extend the IPFS Accelerate JavaScript SDK.

## Table of Contents

- [Architecture](#architecture)
- [Design Principles](#design-principles)
- [Development Setup](#development-setup)
- [Building the SDK](#building-the-sdk)
- [Testing](#testing)
- [Adding New Models](#adding-new-models)
- [Working with Hardware Backends](#working-with-hardware-backends)
- [Performance Optimization](#performance-optimization)
- [Contributing Guidelines](#contributing-guidelines)

## Architecture

The IPFS Accelerate JS SDK is designed with a modular architecture to support various hardware backends, models, and use cases:

```
                  ┌───────────────────────────────────────┐
                  │            IPFS Accelerate JS         │
                  └───────────────────────────────────────┘
                     │               │             │
          ┌──────────┘         ┌─────┘       ┌────┘
┌─────────▼─────────┐ ┌────────▼───────┐ ┌───▼────────────┐
│ Hardware Abstract.│ │  Model Layer   │ │ Resource Pool  │
└───────────────────┘ └────────────────┘ └────────────────┘
          │                  │                  │
    ┌─────┼───────┬────────┬┘           ┌──────┘
┌───▼───┐ ┌───▼───┐ ┌──────▼───┐  ┌─────▼─────┐
│WebGPU │ │WebNN  │ │ Models   │  │ Browser   │
│Backend│ │Backend│ │          │  │ Resources │
└───────┘ └───────┘ └──────────┘  └───────────┘
    │         │          │              │
    │         │          ▼              │
    │         │     ┌─────────┐         │
    │         │     │ Tensor  │         │
    │         │     │ Ops     │         │
    │         │     └─────────┘         │
    │         │          │              │
    └─────────┴──────────┴──────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Browser Rendering │
              └───────────────────┘
```

The key components are:

1. **Hardware Abstraction Layer**: Provides a unified interface to different hardware backends (WebGPU, WebNN, WebAssembly, CPU).

2. **Model Layer**: Implements various machine learning models (BERT, ViT, Whisper) with a consistent interface.

3. **Resource Pool**: Manages limited browser resources for efficient utilization.

4. **Tensor Operations**: Provides operations for working with tensors across different backends.

5. **Browser Integration**: Handles browser-specific optimizations and detection.

## Design Principles

The SDK follows these key design principles:

1. **Progressive Enhancement**: The SDK automatically falls back to less powerful backends when more powerful ones are not available.

2. **Type Safety**: Strong TypeScript typing for better developer experience and error prevention.

3. **Resource Efficiency**: Careful management of limited browser resources.

4. **Browser Compatibility**: Support for a wide range of browsers with different capabilities.

5. **Extensibility**: Easy addition of new models and backends.

## Development Setup

To set up the development environment:

1. Clone the repository:
```bash
git clone https://github.com/organization/ipfs-accelerate-js.git
cd ipfs-accelerate-js
```

2. Install dependencies:
```bash
npm install
```

3. Build the SDK:
```bash
npm run build
```

## Building the SDK

The SDK uses Rollup for bundling:

```bash
# Build a production version
npm run build

# Build a development version with source maps
npm run dev

# Watch for changes and rebuild
npm run dev:watch
```

The build process generates three types of outputs:

1. UMD build for direct browser usage (`dist/ipfs-accelerate.js`)
2. ESM build for modern bundlers (`dist/ipfs-accelerate.esm.js`)
3. TypeScript declaration files (`dist/types/`)

## Testing

The SDK uses Jest for testing:

```bash
# Run unit tests
npm test

# Run browser tests
npm run test:browser

# Run tests with coverage
npm run test:coverage
```

When adding new features, you should add corresponding tests in the `test/` directory. Unit tests should go in `test/unit/`, browser tests in `test/browser/`, and integration tests in `test/integration/`.

## Adding New Models

To add a new model to the SDK:

1. Create a new file in the appropriate directory:
   - Text models: `src/model/transformers/`
   - Vision models: `src/model/vision/`
   - Audio models: `src/model/audio/`

2. Implement the `IModel` interface:
```typescript
export class NewModel implements IModel<NewModelInput, NewModelOutput> {
  async load(): Promise<boolean> {
    // Load model weights
  }
  
  async predict(input: NewModelInput): Promise<NewModelOutput> {
    // Run inference
  }
  
  isLoaded(): boolean {
    // Check if model is loaded
  }
  
  getType(): ModelType {
    // Return model type
  }
  
  getName(): string {
    // Return model name
  }
  
  getMetadata(): Record<string, any> {
    // Return model metadata
  }
  
  dispose(): void {
    // Clean up resources
  }
}
```

3. Export the model from the directory's index.ts file.

4. Add a factory function to create the model:
```typescript
export async function createNewModel(
  modelName: string,
  hardware: HardwareAbstraction,
  options: ModelOptions = {}
): Promise<NewModel> {
  const model = new NewModel(modelName, hardware, options);
  await model.load();
  return model;
}
```

5. Update the main model factory function in `src/model/index.ts` to include the new model type.

## Working with Hardware Backends

The SDK is designed to work with different hardware backends:

### WebGPU Backend

The WebGPU backend uses the WebGPU API for hardware acceleration. To work with it directly:

```typescript
import { WebGPUBackend } from 'ipfs-accelerate-js';

// Create a WebGPU backend
const webgpu = new WebGPUBackend({
  logging: true,
  requiredFeatures: ['shader-f16']
});

// Initialize the backend
await webgpu.initialize();

// Use the backend
const device = webgpu.getDevice();
const adapter = webgpu.getAdapter();
```

### WebNN Backend

The WebNN backend uses the WebNN API for neural network acceleration. To work with it directly:

```typescript
import { WebNNBackend } from 'ipfs-accelerate-js';

// Create a WebNN backend
const webnn = new WebNNBackend({
  logging: true,
  devicePreference: 'gpu'
});

// Initialize the backend
await webnn.initialize();

// Use the backend
const context = webnn.getContext();
```

### Adding a New Backend

To add a new backend:

1. Create a new file in `src/hardware/backends/`.

2. Implement the `IBackend` interface:
```typescript
export class NewBackend implements IBackend {
  async initialize(): Promise<boolean> {
    // Initialize the backend
  }
  
  isInitialized(): boolean {
    // Check if the backend is initialized
  }
  
  isRealHardware(): boolean {
    // Check if the backend is using real hardware
  }
  
  getType(): HardwareBackendType {
    // Return the backend type
  }
  
  getCapabilities(): any {
    // Return the backend capabilities
  }
  
  getInfo(): any {
    // Return the backend information
  }
  
  dispose(): void {
    // Clean up resources
  }
}
```

3. Add detection functions:
```typescript
export async function isNewBackendSupported(): Promise<boolean> {
  // Check if the backend is supported
}

export async function getNewBackendInfo(): Promise<any> {
  // Get detailed backend information
}
```

4. Export the backend from the directory's index.ts file.

5. Update the hardware abstraction layer in `src/hardware/hardware_abstraction.ts` to include the new backend.

## Performance Optimization

To optimize the performance of the SDK:

1. **Reuse Resources**: Use the resource pool to manage and reuse expensive resources.

2. **Tensor Sharing**: Share tensors between models to reduce memory usage.

3. **Backend Selection**: Choose the most appropriate backend for each model type.

4. **Precompilation**: Precompile shaders and models for faster startup.

5. **Batch Processing**: Process multiple inputs in a batch for better throughput.

## Contributing Guidelines

When contributing to the SDK:

1. **Code Style**: Follow the existing code style and use the provided ESLint configuration.

2. **Documentation**: Add JSDoc comments to all public APIs and update the relevant documentation files.

3. **Testing**: Add tests for new features and ensure all tests pass before submitting a pull request.

4. **Pull Requests**: Keep pull requests focused on a single feature or bug fix.

5. **Versioning**: Follow semantic versioning (MAJOR.MINOR.PATCH).

6. **Backward Compatibility**: Maintain backward compatibility within the same major version.

For more details, see the [Contributing Guide](CONTRIBUTING.md).