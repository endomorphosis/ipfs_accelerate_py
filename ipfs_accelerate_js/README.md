# IPFS Accelerate JS SDK

JavaScript SDK for accelerating AI models in web browsers using WebGPU, WebNN, and IPFS optimization.

## Features

- 🚀 **Hardware Acceleration**: Utilize WebGPU and WebNN for fast model inference
- 🔄 **Automatic Fallbacks**: Seamlessly fall back to available hardware
- 🌐 **Browser Optimizations**: Browser-specific optimizations for best performance
- 📦 **Cross-Model Tensor Sharing**: Share tensors between models for memory efficiency
- 🔧 **Resource Pooling**: Manage resources efficiently across models
- 💾 **Model Storage**: Efficient storage and caching of model weights
- 📊 **Performance Monitoring**: Track and analyze performance

## Installation

```bash
npm install ipfs-accelerate-js
```

## Usage

### Basic Usage

```javascript
import { initialize } from 'ipfs-accelerate-js';

// Initialize the SDK
const sdk = await initialize();

// Load a model
const bertModel = await sdk.createModel('bert-base-uncased', {
  preferredBackend: 'webgpu'
});

// Run inference
const result = await bertModel.predict({
  text: "Hello, world!"
});

console.log(result);
```

### Hardware Selection

```javascript
import { createHardwareAbstraction } from 'ipfs-accelerate-js';

// Create hardware abstraction with preferences
const hardware = await createHardwareAbstraction({
  preferredBackends: ['webgpu', 'webnn', 'cpu'],
  logging: true
});

// Check capabilities
const capabilities = hardware.getCapabilities();
console.log('WebGPU supported:', capabilities.webgpu.supported);
console.log('WebNN supported:', capabilities.webnn.supported);

// Get best backend for model type
const bestBackend = hardware.getOptimalBackendForModel('vision');
console.log('Best backend for vision models:', bestBackend);
```

### Cross-Model Tensor Sharing

```javascript
import { initialize } from 'ipfs-accelerate-js';

const sdk = await initialize();

// Load two models that can share tensors
const bertEncoder = await sdk.createModel('bert-base-uncased-encoder');
const bertClassifier = await sdk.createModel('bert-base-uncased-classifier');

// Create shared context for tensor sharing
const sharedContext = sdk.createSharedContext();

// Run inference with shared tensors
const encodedText = await bertEncoder.predict(
  { text: "This movie was great!" },
  { context: sharedContext }
);

const classification = await bertClassifier.predict(
  { encoded: encodedText },
  { context: sharedContext }
);

console.log('Classification:', classification);
```

## Browser Compatibility

The SDK supports all major browsers with WebGPU or WebNN support:

| Browser | WebGPU | WebNN | Best For |
|---------|--------|-------|----------|
| Chrome  | ✅     | ⚠️    | Vision models |
| Edge    | ✅     | ✅    | Text models |
| Firefox | ✅     | ❌    | Audio models |
| Safari  | ✅     | ⚠️    | General use |

The SDK automatically detects the available capabilities and uses the best backend for each model type.

## Documentation

For detailed documentation, see:

- [API Reference](docs/api/README.md)
- [Examples](examples/README.md)
- [TypeScript Implementation](docs/TYPESCRIPT_IMPLEMENTATION_SUMMARY.md)

## Development

### Setup

```bash
git clone https://github.com/organization/ipfs-accelerate-js.git
cd ipfs-accelerate-js
npm install
```

### Build

```bash
npm run build
```

### Testing

```bash
npm test
```

### Examples

```bash
npm run examples
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

MIT License