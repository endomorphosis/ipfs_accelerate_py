# IPFS Accelerate JavaScript SDK

A comprehensive toolkit for accelerating AI models in web browsers and Node.js environments using WebGPU, WebNN, and IPFS optimization.

## Overview

The IPFS Accelerate JavaScript SDK provides a unified interface for leveraging hardware acceleration in web browsers and optimizing content delivery through IPFS. It enables developers to run AI models efficiently using WebGPU, WebNN, and WebAssembly backends, with intelligent hardware selection and browser-specific optimizations.

## Key Features

- **Web Hardware Acceleration**: Automatic detection and utilization of WebGPU, WebNN, and WebAssembly
- **Comprehensive WebGPU & WebNN Backends**: Full implementation of both WebGPU and WebNN for optimal hardware acceleration
- **IPFS Integration**: Optimized IPFS content loading and distribution in the browser
- **P2P Optimization**: Enhanced content distribution through peer-to-peer network optimization
- **IndexedDB Integration**: Built-in storage and analysis of acceleration results
- **Cross-Browser Support**: Works across Chrome, Firefox, Edge, and Safari with appropriate fallbacks
- **Browser-Specific Optimizations**: Special optimizations for different browsers (e.g., Firefox for audio models, Edge for WebNN)
- **Ultra-Low Precision Framework**: Advanced quantization support from 8-bit down to 2-bit precision
- **Shader Precompilation**: Improved startup times through shader precompilation
- **Graph-Based Neural Network Execution**: Optimized execution of neural networks using WebNN graph operations
- **React Integration**: Dedicated React hooks for easy integration

## Installation

```bash
npm install ipfs-accelerate-js
```

Or include via CDN:

```html
<script src="https://cdn.jsdelivr.net/npm/ipfs-accelerate-js@0.5.0/dist/ipfs-accelerate.min.js"></script>
```

## Basic Usage

```javascript
import { createHardwareAbstraction, createViT } from 'ipfs-accelerate-js';

// Initialize hardware abstraction layer
const hardware = await createHardwareAbstraction();

// Create a Vision Transformer (ViT) model for image classification
const vit = await createViT('vit-base-patch16-224', hardware);

// Classify an image
const image = document.getElementById('image');
const predictions = await vit.classify(image);
console.log(predictions);
// [
//   { classId: 282, label: 'tiger cat', score: 0.92 },
//   { classId: 281, label: 'tabby cat', score: 0.05 },
//   ...
// ]

// Get image embeddings (useful for similarity search)
const embeddings = await vit.getEmbeddings(image);
console.log(`Embedding size: ${embeddings.length}`);

// Or use the legacy WebAccelerator API
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

## Advanced Usage with WebNN

```javascript
import { createHardwareAbstraction } from 'ipfs-accelerate-js';

async function demoWebNNAcceleration() {
  // Initialize hardware abstraction with WebNN preference
  const hardware = await createHardwareAbstraction({
    backendOrder: ['webnn', 'webgpu', 'wasm', 'cpu']
  });
  
  // Check if WebNN is available
  const hasWebNN = hardware.hasBackend('webnn');
  console.log(`WebNN available: ${hasWebNN}`);
  
  if (hasWebNN) {
    // Get WebNN backend
    const webnnBackend = hardware.getBackend('webnn');
    
    // Get capabilities
    const capabilities = await webnnBackend.getCapabilities();
    console.log('WebNN capabilities:', capabilities);
    
    // Create input tensors
    const inputTensor = await webnnBackend.createTensor(
      new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]), 
      [3, 3], 
      'float32'
    );
    
    // Create filter tensor for convolution
    const filterTensor = await webnnBackend.createTensor(
      new Float32Array([1, 1, 1, 0, 0, 0, -1, -1, -1]), 
      [1, 1, 3, 3], 
      'float32'
    );
    
    // Execute 2D convolution operation
    const convResult = await webnnBackend.execute('conv2d', {
      input: inputTensor,
      filter: filterTensor
    }, {
      padding: [1, 1, 1, 1],
      strides: [1, 1]
    });
    
    // Read results back to CPU
    const resultData = await webnnBackend.readTensor(
      convResult.tensor,
      convResult.shape,
      'float32'
    );
    
    console.log('Convolution result:', resultData);
    console.log('Result shape:', convResult.shape);
  }
}

// You can also use WebNN directly for specialized use cases
import { WebNNBackend } from 'ipfs-accelerate-js/webnn';

async function directWebNNUsage() {
  const webnn = new WebNNBackend({
    deviceType: 'gpu',
    powerPreference: 'high-performance',
    enableLogging: true
  });
  
  if (await webnn.isSupported()) {
    await webnn.initialize();
    // Use WebNN directly for specialized applications
  }
}
```

## React Integration

### Vision Model Integration

```jsx
import React, { useState, useCallback } from 'react';
import { useVisionModel, useHardwareInfo } from 'ipfs-accelerate-js/react';

function ImageClassificationComponent() {
  // React hook for ViT model
  const { model, status, error } = useVisionModel({
    modelId: 'vit-base-patch16-224',
    config: {
      returnEmbeddings: true,
      browserOptimizations: true
    }
  });

  // Hardware capabilities hook
  const { capabilities, isReady } = useHardwareInfo();
  
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [processingTime, setProcessingTime] = useState(null);
  
  const handleImageChange = useCallback((e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.src = event.target.result;
        img.onload = () => setSelectedImage(img);
      };
      reader.readAsDataURL(file);
    }
  }, []);
  
  const classifyImage = useCallback(async () => {
    if (model && selectedImage) {
      const startTime = performance.now();
      const result = await model.classify(selectedImage);
      const endTime = performance.now();
      
      setPredictions(result);
      setProcessingTime(endTime - startTime);
    }
  }, [model, selectedImage]);
  
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
        <h3>Image Classification</h3>
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleImageChange}
        />
        {selectedImage && (
          <div>
            <img 
              src={selectedImage.src} 
              alt="Selected" 
              style={{ maxWidth: '300px', maxHeight: '300px' }}
            />
            <button 
              onClick={classifyImage} 
              disabled={status !== 'loaded'}
            >
              Classify Image
            </button>
          </div>
        )}
        
        {predictions.length > 0 && (
          <div>
            <h4>Results (processed in {processingTime.toFixed(2)}ms)</h4>
            <ul>
              {predictions.map((pred, i) => (
                <li key={i}>
                  {pred.label}: {(pred.score * 100).toFixed(2)}%
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
```

### Text Model Integration

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

The SDK supports the following browsers with different hardware acceleration capabilities:

| Browser | Version | WebGPU | WebNN | Best Performance For |
|---------|---------|--------|-------|---------------------|
| Chrome | 121+ | ✅ | ✅ | Vision models, general WebGPU |
| Edge | 121+ | ✅ | ✅ | WebNN neural networks, text models |
| Firefox | 124+ | ✅ | ❌ | Audio models, compute shader performance |
| Safari | 17.4+ | ✅ | ❌ | Basic WebGPU support (experimental) |

### Browser-Specific Optimizations

- **Edge**: Best WebNN implementation with graph-based optimizations for neural networks
- **Firefox**: Superior compute shader performance (20-55% faster) for audio models
- **Chrome**: Good all-around WebGPU support with shader precompilation
- **Safari**: Experimental WebGPU support with some performance limitations

The SDK automatically detects the best backend based on browser capabilities, model type, and hardware. You can also manually specify backend preferences for specialized workloads.

## Development Status

The IPFS Accelerate JavaScript SDK is in active development with the following timeline:

1. **Phase 1: Core Architecture** ✅ COMPLETED (March 2025)
   - ✅ Browser hardware detection
   - ✅ WebGPU and WebNN backends
   - ✅ Core acceleration interface
   - ✅ React integration
   
2. **Phase 2: Browser Enhancement** 🔄 IN PROGRESS (April-May 2025)
   - ✅ Browser-specific optimizations
   - ✅ Shader optimizations for WebGPU
   - ✅ Graph-based optimizations for WebNN
   - 🔄 IndexedDB storage integration
   - 🔄 Advanced hardware features
   - 🔄 Advanced resource management

3. **Phase 3: Advanced Features** (May-June 2025)
   - 🔄 Ultra-low precision framework
   - 🔲 Distributed execution
   - 🔲 P2P optimization
   - 🔲 Performance benchmarking
   
### Current Version (0.5.0)

This release includes the following improvements:

- Complete implementation of WebNN backend with 4 core operations (matmul, elementwise, softmax, convolution)
- Graph-based computation with model caching for WebNN
- Device detection and simulation awareness
- Improved browser-specific optimizations for Edge browser
- Enhanced examples and documentation for WebNN usage

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