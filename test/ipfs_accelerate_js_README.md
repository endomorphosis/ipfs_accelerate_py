# IPFS Accelerate JavaScript SDK

A comprehensive toolkit for accelerating AI models in web browsers and Node.js environments using WebGPU, WebNN, and IPFS optimization.

## Overview

The IPFS Accelerate JavaScript SDK provides a unified interface for leveraging hardware acceleration in web browsers and optimizing content delivery through IPFS. It enables developers to run AI models efficiently using WebGPU, WebNN, and WebAssembly backends, with intelligent hardware selection and browser-specific optimizations.

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

## Installation

```bash
npm install ipfs-accelerate-js
```

Or include via CDN:

```html
<script src="https://cdn.jsdelivr.net/npm/ipfs-accelerate-js@0.4.0/dist/ipfs-accelerate.min.js"></script>
```

## Basic Usage

```javascript
import { WebAccelerator } from 'ipfs-accelerate-js';

// Create an accelerator instance
const accelerator = new WebAccelerator();
await accelerator.initialize();

// Check hardware capabilities
const capabilities = accelerator.getCapabilities();
console.log('WebGPU support:', capabilities.webgpu.supported);
console.log('WebNN support:', capabilities.webnn.supported);

// Run a text model
const textResult = await accelerator.accelerate({
  modelId: 'bert-base-uncased',
  modelType: 'text',
  input: 'This is a test sentence.',
  config: { autoSelectHardware: true }
});

console.log(`Processing time: ${textResult.processingTime} ms`);
console.log(`Using hardware: ${textResult.hardware}`);
```

## React Integration

```jsx
import React from 'react';
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

## Development Status

The IPFS Accelerate JavaScript SDK is currently in active development with the following timeline:

1. **Phase 1: Core Architecture** (June-July 2025)
   - Browser hardware detection
   - WebGPU, WebNN, and WebAssembly backends
   - Core acceleration interface
   - IndexedDB storage
   - React integration

2. **Phase 2: Browser Enhancement** (July-August 2025)
   - Browser-specific optimizations
   - Shader optimizations
   - Advanced hardware features
   - Advanced resource management

3. **Phase 3: Advanced Features** (August-September 2025)
   - Ultra-low precision framework
   - Distributed execution
   - P2P optimization
   - Performance benchmarking

## Documentation

For complete documentation, see:
- [API Reference](https://ipfsacceleratejs.org/docs/api)
- [Examples](https://ipfsacceleratejs.org/examples)
- [Browser Compatibility Matrix](https://ipfsacceleratejs.org/docs/compatibility)

## License

MIT

## Related Projects

- [IPFS Accelerate Python Framework](https://github.com/organization/ipfs-accelerate-py)
- [WebGPU Shader Collection](https://github.com/organization/webgpu-shaders)
- [WebNN Model Zoo](https://github.com/organization/webnn-models)