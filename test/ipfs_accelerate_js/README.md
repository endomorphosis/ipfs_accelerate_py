# IPFS Accelerate JS

Hardware-accelerated machine learning for the browser with WebGPU and WebNN.

[![npm version](https://img.shields.io/npm/v/ipfs-accelerate-js.svg)](https://www.npmjs.com/package/ipfs-accelerate-js)
[![license](https://img.shields.io/github/license/ipfs/ipfs-accelerate-js.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## Overview

IPFS Accelerate JS delivers high-performance AI models directly in the browser using WebGPU and WebNN for hardware acceleration. Run transformers, vision models, and audio models with GPU-accelerated inference - no server required.

### Key Features

- **Hardware Acceleration**: Uses WebGPU and WebNN for GPU-accelerated inference
- **Cross-Browser Compatibility**: Works in Chrome, Firefox, Edge, and Safari
- **Multiple Model Support**: Text (BERT), Vision (ViT), and Audio (Whisper) models
- **Cross-Model Tensor Sharing**: Share tensors between models for efficient multimodal applications
- **Memory Optimization**: Careful tensor management with proper cleanup
- **Browser-Specific Optimizations**: Tailored performance for different browsers

## Quick Start

### Installation

```bash
npm install ipfs-accelerate-js
```

### Basic Usage

```typescript
import { createVitModel, createWebGPUBackend } from 'ipfs-accelerate-js';

async function classifyImage(imageData, width, height) {
  // Create hardware backend
  const hardware = await createWebGPUBackend();
  await hardware.initialize();
  
  // Create ViT model
  const model = createVitModel(hardware, {
    modelId: 'google/vit-base-patch16-224'
  });
  await model.initialize();
  
  // Process image
  const result = await model.process({
    imageData,
    width,
    height
  });
  
  console.log(`Predicted class: ${result.classId}`);
  console.log(`Probability: ${result.probabilities[result.classId]}`);
  
  // Clean up resources
  await model.dispose();
}
```

## Models

### Vision Models

```typescript
import { createVitModel } from 'ipfs-accelerate-js';

// Create Vision Transformer (ViT) model
const vit = createVitModel(hardware, {
  modelId: 'google/vit-base-patch16-224',
  useOptimizedOps: true
});

// Process image
const result = await vit.process({
  imageData,
  width: 224,
  height: 224
});
```

### Text Models

```typescript
import { createBertModel } from 'ipfs-accelerate-js';

// Create BERT model
const bert = createBertModel(hardware, {
  modelId: 'bert-base-uncased',
  useOptimizedOps: true
});

// Tokenize and process text
const tokens = await bert.tokenize("Hello world");
const result = await bert.process(tokens);
```

### Audio Models

```typescript
import { createWhisperModel } from 'ipfs-accelerate-js';

// Create Whisper model
const whisper = createWhisperModel(hardware, {
  modelId: 'openai/whisper-tiny',
  useBrowserOptimizations: true
});

// Process audio
const result = await whisper.process({
  audioData,
  sampleRate: 16000
});

console.log(`Transcription: ${result.text}`);
```

## Cross-Model Tensor Sharing

Share tensors between models for efficient multimodal applications:

```typescript
// Process image with ViT
const vitResult = await vitModel.process(imageInput);
const visionEmbedding = vitModel.getSharedTensor('vision_embedding');

// Process text with BERT 
const bertResult = await bertModel.process(tokens);
const textEmbedding = bertModel.getSharedTensor('text_embedding');

// Use both embeddings in a multimodal application
// This avoids redundant computation and reduces memory usage
```

## Hardware Backends

Choose from multiple backends for hardware acceleration:

```typescript
import { 
  createWebGPUBackend, 
  createWebNNBackend,
  getBrowserHardwareCapabilities 
} from 'ipfs-accelerate-js';

// Detect available hardware capabilities
const capabilities = await getBrowserHardwareCapabilities();

// Create appropriate backend based on capabilities
let hardware;
if (capabilities.webgpu) {
  hardware = await createWebGPUBackend();
} else if (capabilities.webnn) {
  hardware = await createWebNNBackend();
} else {
  // Fallback to CPU backend
  hardware = await createCPUBackend();
}
```

## Browser Support

| Feature | Chrome | Firefox | Edge | Safari |
|---------|--------|---------|------|--------|
| WebGPU | 113+ | 118+ | 113+ | 17+ |
| WebNN | - | - | 113+ | - |
| CPU Fallback | ✓ | ✓ | ✓ | ✓ |

## Examples

### Browser-Based Image Classification

```html
<!DOCTYPE html>
<html>
<head>
  <title>ViT Demo</title>
  <script src="https://cdn.jsdelivr.net/npm/ipfs-accelerate-js/dist/ipfs-accelerate.min.js"></script>
</head>
<body>
  <input type="file" id="image-input" accept="image/*">
  <div id="result"></div>
  
  <script>
    const { createVitModel, createWebGPUBackend } = IPFSAccelerate;
    
    document.getElementById('image-input').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      const img = await createImageBitmap(file);
      
      // Create canvas to get image data
      const canvas = document.createElement('canvas');
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, 224, 224);
      const imageData = ctx.getImageData(0, 0, 224, 224).data;
      
      // Convert to RGB format (remove alpha channel)
      const rgbData = new Uint8Array(224 * 224 * 3);
      for (let i = 0; i < imageData.length / 4; i++) {
        rgbData[i * 3] = imageData[i * 4];
        rgbData[i * 3 + 1] = imageData[i * 4 + 1];
        rgbData[i * 3 + 2] = imageData[i * 4 + 2];
      }
      
      // Initialize hardware and model
      const hardware = await createWebGPUBackend();
      await hardware.initialize();
      
      const model = createVitModel(hardware, {
        modelId: 'google/vit-base-patch16-224'
      });
      await model.initialize();
      
      // Process image
      const result = await model.process({
        imageData: rgbData,
        width: 224,
        height: 224
      });
      
      // Display result
      document.getElementById('result').textContent = 
        `Top class: ${result.classId} (${result.probabilities[result.classId]})`;
      
      // Clean up
      await model.dispose();
    });
  </script>
</body>
</html>
```

## Advanced Usage

### Custom Configuration

```typescript
const vitModel = createVitModel(hardware, {
  modelId: 'google/vit-base-patch16-224',
  imageSize: 224,
  patchSize: 16,
  hiddenSize: 768,
  numLayers: 12,
  numHeads: 12,
  intermediateSize: 3072,
  backendPreference: ['webgpu', 'webnn', 'cpu'],
  useOptimizedOps: true
});
```

### Browser-Specific Optimizations

```typescript
import { getBrowserInfo } from 'ipfs-accelerate-js';

const browserInfo = getBrowserInfo();
let config = {};

if (browserInfo.name === 'chrome') {
  config = {
    useChromiumOptimizations: true,
    workgroupSize: 256
  };
} else if (browserInfo.name === 'firefox') {
  config = {
    useFirefoxOptimizations: true,
    workgroupSize: 128
  };
}

const model = createVitModel(hardware, {
  ...config,
  modelId: 'google/vit-base-patch16-224'
});
```

### Memory Management

```typescript
// Create model in a try-finally block to ensure cleanup
let model = null;
try {
  model = createVitModel(hardware, config);
  await model.initialize();
  
  // Use the model...
  const result = await model.process(input);
  
} finally {
  // Always clean up resources
  if (model) {
    await model.dispose();
  }
}
```

## Documentation

- [API Reference](./docs/api/README.md)
- [Models](./docs/models/README.md)
- [Hardware Backends](./docs/hardware/README.md)
- [Cross-Model Tensor Sharing](./docs/CROSS_MODEL_TENSOR_SHARING.md)
- [Performance Optimization](./docs/PERFORMANCE_OPTIMIZATION.md)
- [Examples](./examples/README.md)

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you use IPFS Accelerate JS in your research, please cite:

```bibtex
@software{ipfs_accelerate_js,
  author = {IPFS Accelerate Team},
  title = {IPFS Accelerate JS: Hardware-accelerated machine learning for the browser},
  url = {https://github.com/ipfs/ipfs-accelerate-js},
  year = {2025},
}
```