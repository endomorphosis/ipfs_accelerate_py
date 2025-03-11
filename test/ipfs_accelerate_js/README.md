# IPFS Accelerate JavaScript SDK

A comprehensive toolkit for accelerating AI models in web browsers and Node.js environments using WebGPU, WebNN, and IPFS optimization.

## Status: Under Active Development

This JavaScript SDK is part of the migration effort to separate browser-specific WebGPU and WebNN implementations from the Python framework into a dedicated JavaScript implementation. The migration is planned to be completed in Q3 2025.

**Current Development Phase:** Planning & Initial Implementation (50% complete)

## Overview

The IPFS Accelerate JavaScript SDK provides a unified interface for leveraging hardware acceleration in web browsers, with support for WebGPU, WebNN, and WebAssembly backends. It enables efficient execution of AI models with browser-specific optimizations and advanced features like ultra-low precision quantization.

## Key Features

- **Web Hardware Acceleration**: Automatic detection and utilization of WebGPU, WebNN, and WebAssembly
- **IPFS Integration**: Optimized IPFS content loading and distribution in the browser
- **P2P Optimization**: Enhanced content distribution through peer-to-peer network optimization
- **IndexedDB Integration**: Built-in storage and analysis of acceleration results
- **Cross-Browser Support**: Works across Chrome, Firefox, Edge, and Safari with appropriate fallbacks
- **Browser-Specific Optimizations**: Special optimizations for different browsers (e.g., Firefox for audio models)
- **Ultra-Low Precision Framework**: Advanced quantization support from 8-bit down to 2-bit precision
- **Shader Precompilation**: Improved startup times through shader precompilation
- **React Integration**: Dedicated React hooks for easy integration

## Project Structure

```
ipfs_accelerate_js/
├── src/
│   ├── worker/                # Core model execution components
│   │   ├── webnn/             # WebNN backend implementation
│   │   ├── webgpu/            # WebGPU backend implementation
│   │   │   └── shaders/       # Browser-optimized WGSL shaders
│   │   └── wasm/              # WebAssembly backend implementation
│   ├── hardware/              # Hardware abstraction layer
│   │   └── backends/          # Hardware backend implementations
│   ├── model/                 # Model management
│   ├── quantization/          # Quantization framework
│   └── react/                 # React integration
├── examples/                  # Example implementations
├── dist/                      # Distribution files
└── test/                      # Test suite
```

## Development Status

This SDK is under active development with the following components implemented:

- ✅ WebGPU backend implementation
- ✅ WebNN backend implementation
- ✅ Hardware abstraction layer
- ✅ Model loader
- ✅ Quantization engine with 2-bit to 16-bit support
- ✅ Browser interface with capability detection
- ✅ React integration hooks and components
- ✅ Initial WGSL shader migration

## Installation (Not Yet Available)

The SDK is not yet available for installation as it's still under development. Once released, it will be installable via npm:

```bash
npm install ipfs-accelerate-js
```

## Example Usage

```javascript
import { createAccelerator } from 'ipfs-accelerate-js';

// Create an accelerator instance
const accelerator = await createAccelerator({
  autoDetectHardware: true,
  preferredBackend: 'webgpu',
  fallbackOrder: ['webgpu', 'webnn', 'wasm', 'cpu']
});

// Check hardware capabilities
const capabilities = accelerator.getCapabilities();
console.log('WebGPU support:', capabilities.webgpu.supported);

// Run a text model
const result = await accelerator.accelerate({
  modelId: 'bert-base-uncased',
  modelType: 'text',
  input: 'This is a test sentence.'
});

console.log(`Processing time: ${result.processingTime} ms`);
console.log(`Using hardware: ${result.hardware}`);
```

## React Integration Example

```jsx
import React, { useState } from 'react';
import { useModel, useHardwareInfo } from 'ipfs-accelerate-js/react';

function TextEmbeddingComponent() {
  // React hook for easy model loading
  const { model, status, error } = useModel({
    modelId: 'bert-base-uncased',
    autoHardwareSelection: true
  });

  // Hook for hardware information
  const { capabilities, isReady } = useHardwareInfo();

  const [input, setInput] = useState('');
  const [embedding, setEmbedding] = useState(null);

  async function generateEmbedding() {
    if (model && input) {
      const result = await model.getEmbeddings(input);
      setEmbedding(result);
    }
  }

  return (
    <div>
      <div>
        <h3>Hardware Status</h3>
        {isReady && (
          <ul>
            <li>WebGPU: {capabilities.webgpu.supported ? 'Yes' : 'No'}</li>
            <li>WebNN: {capabilities.webnn.supported ? 'Yes' : 'No'}</li>
            <li>Optimal backend: {capabilities.optimalBackend}</li>
          </ul>
        )}
      </div>

      <div>
        <h3>Text Embedding</h3>
        <input 
          value={input} 
          onChange={e => setInput(e.target.value)} 
          placeholder="Enter text to embed"
        />
        <button 
          onClick={generateEmbedding} 
          disabled={status !== 'loaded'}
        >
          Generate Embedding
        </button>
        {embedding && <p>Embedding generated: {embedding.length} dimensions</p>}
      </div>
    </div>
  );
}
```

## Browser Support

The SDK supports the following browsers:
- Chrome 113+ (best for vision models)
- Edge 113+ (best for WebNN models)
- Firefox 113+ (best for audio models)
- Safari 16.4+ (with some feature limitations)

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Run browser-specific tests
npm run test:browser

# Run example server
npm run examples
```

## Timeline

The development is following this timeline:

| Phase | Dates | Status |
|-------|-------|--------|
| Planning & Initial Implementation | March-May 2025 | 🔄 IN PROGRESS (50%) |
| Phase 1: Core Architecture | June-July 2025 | 📅 PLANNED |
| Phase 2: Browser Enhancement | July-August 2025 | 📅 PLANNED |
| Phase 3: Advanced Features | September-October 2025 | 📅 PLANNED |

## Documentation

For complete documentation, see the `docs/` directory.

## License

MIT

## Related Projects

- [IPFS Accelerate Python Framework](https://github.com/organization/ipfs-accelerate-py)
- [WebGPU Shader Collection](https://github.com/organization/webgpu-shaders)
- [WebNN Model Zoo](https://github.com/organization/webnn-models)