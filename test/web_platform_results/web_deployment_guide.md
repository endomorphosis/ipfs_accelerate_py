# Web Deployment Guide for IPFS Accelerate

This guide provides comprehensive instructions for deploying models using the WebNN and WebGPU backends provided by the IPFS Accelerate framework.

## Overview

The IPFS Accelerate Python framework supports two primary web deployment paths:

1. **WebNN Backend**: Deploy models using the Web Neural Network API standard
2. **WebGPU/transformers.js Backend**: Deploy models using WebGPU acceleration via transformers.js

Both approaches enable hardware-accelerated ML inference directly in modern web browsers.

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- ONNX Runtime 1.14+
- Node.js 14+ (for transformers.js)
- A modern web browser supporting WebNN or WebGPU

## Quick Start

```bash
# Clone the repository if you haven't already
git clone https://github.com/your-org/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install dependencies
pip install -r requirements.txt
npm install  # For web dependencies

# Export a BERT model for web deployment
python export_model_for_web.py --model bert-base-uncased --format both
```

## Export Process Overview

The export process involves these key steps:

1. **Model Loading**: Load the model from the Hugging Face Hub or local files
2. **Optimization**: Apply model-specific optimizations (quantization, pruning, etc.)
3. **ONNX Export**: Export to ONNX format (for WebNN)
4. **transformers.js Export**: Export to transformers.js format (for WebGPU)
5. **Validation**: Verify exported models match the original model's outputs

## WebNN Export

The WebNN export process converts models to ONNX format with optimizations for WebNN compatibility:

```python
from ipfs_accelerate_py.export import export_model_for_webnn

export_model_for_webnn(
    model_name="bert-base-uncased",
    output_dir="./web_models/webnn_bert",
    precision="fp16",  # Options: fp32, fp16, int8
    optimize=True,     # Apply ONNX Runtime optimizations
    simplify=True,     # Remove unused nodes
    operator_fusion=True  # Fuse operations where possible
)
```

### WebNN Export Options

| Option | Description | Default |
|--------|-------------|---------|
| `precision` | Model precision (fp32, fp16, int8) | "fp32" |
| `optimize` | Apply ONNX Runtime optimizations | True |
| `simplify` | Remove unused nodes | True |
| `operator_fusion` | Fuse operations for better performance | True |
| `dynamic_axes` | Use dynamic axes for variable-size inputs | True |
| `opset_version` | ONNX opset version | 15 |
| `validate` | Validate exported model matches original | True |

## WebGPU/transformers.js Export

The WebGPU export process prepares models for use with transformers.js:

```python
from ipfs_accelerate_py.export import export_model_for_webgpu

export_model_for_webgpu(
    model_name="bert-base-uncased",
    output_dir="./web_models/webgpu_bert",
    include_tokenizer=True,  # Export tokenizer with model
    quantized=False,         # Enable quantization
    optimize=True            # Apply transformers.js-specific optimizations
)
```

### WebGPU Export Options

| Option | Description | Default |
|--------|-------------|---------|
| `include_tokenizer` | Export tokenizer with model | True |
| `quantized` | Enable quantization | False |
| `optimize` | Apply transformers.js optimizations | True |
| `format` | Format type (safetensors, bin, onnx) | "safetensors" |
| `compression` | Compression type for files | "gzip" |
| `validate` | Validate exported model matches original | True |

## Model Compatibility

### WebNN Compatible Models

| Model Family | Compatibility | Notes |
|--------------|---------------|-------|
| BERT/DistilBERT | ✅ High | Excellent performance |
| T5 (small) | ✅ High | Good with fp16 precision |
| ViT | ✅ High | Works well on most hardware |
| ResNet | ✅ High | Excellent performance |
| GPT-2 (small) | ✅ Medium | Works with limitations |
| CLIP | ✅ Medium | Vision part works better than text |
| Whisper (tiny) | ✅ Medium | Limited to small variants |
| BLIP | ❌ Low | Complex architecture not well supported |
| LLaMa | ❌ Low | Too large for most browsers |

### WebGPU/transformers.js Compatible Models

| Model Family | Compatibility | Notes |
|--------------|---------------|-------|
| BERT/DistilBERT | ✅ High | Best performance |
| ViT | ✅ High | Excellent with WebGPU |
| ResNet | ✅ High | Very good performance |
| T5 (small) | ✅ Medium | Works but slower than WebNN |
| CLIP | ✅ Medium | Basic functionality works |
| GPT-2 (small) | ✅ Medium | Works with size limitations |
| Whisper (tiny) | ❌ Low | Limited support in transformers.js |
| LLaMa | ❌ Low | Too large for most browsers |

## Web Implementation Guide

### HTML Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IPFS Accelerate Web Demo</title>
    <!-- Load ONNX Runtime for WebNN -->
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <!-- Load transformers.js for WebGPU -->
    <script src="https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0"></script>
</head>
<body>
    <!-- Your interface here -->
    
    <script>
        // Your implementation here
    </script>
</body>
</html>
```

### WebNN Implementation

```javascript
// Check for WebNN support
const hasWebNN = navigator.ml && navigator.ml.getNeuralNetworkContext;

// Load the ONNX model
async function loadONNXModel() {
    try {
        // WebNN context (when available)
        let sessionOptions = {};
        
        if (hasWebNN) {
            // Use WebNN backend when available
            sessionOptions.executionProviders = ['webnn'];
        } else {
            // Fall back to WebAssembly
            sessionOptions.executionProviders = ['wasm'];
        }
        
        // Create inference session
        const session = await ort.InferenceSession.create(
            'models/model.onnx', 
            sessionOptions
        );
        
        return session;
    } catch (error) {
        console.error("Error loading model:", error);
        throw error;
    }
}

// Run inference
async function runInference(session, inputs) {
    try {
        // Create ONNX Tensors from inputs
        const feeds = {};
        for (const [name, data] of Object.entries(inputs)) {
            feeds[name] = new ort.Tensor(
                data.type, 
                data.data, 
                data.dims
            );
        }
        
        // Run inference
        const results = await session.run(feeds);
        return results;
    } catch (error) {
        console.error("Inference error:", error);
        throw error;
    }
}
```

### WebGPU/transformers.js Implementation

```javascript
// Import from transformers.js
const { pipeline, env } = window.transformers;

// Configure transformers.js to use WebGPU if available
env.useBrowserCache = true;
env.useWebGPU = true;

// Load model and create pipeline
async function loadTransformersModel() {
    try {
        // Create a pipeline for the specific task
        const model = await pipeline('text-classification', 'models/bert');
        return model;
    } catch (error) {
        console.error("Error loading model:", error);
        throw error;
    }
}

// Run inference
async function runInference(model, text) {
    try {
        // Run the model
        const result = await model(text);
        return result;
    } catch (error) {
        console.error("Inference error:", error);
        throw error;
    }
}
```

## Performance Optimization

### WebNN Optimizations

1. **Use Quantization**: Int8 quantization significantly improves performance:
   ```python
   export_model_for_webnn(model_name="...", precision="int8")
   ```

2. **Operator Fusion**: Fuse operations where possible:
   ```python
   export_model_for_webnn(model_name="...", operator_fusion=True)
   ```

3. **Reduce Model Size**: Use smaller/distilled models:
   ```python
   export_model_for_webnn(model_name="distilbert-base-uncased")
   ```

### WebGPU/transformers.js Optimizations

1. **Enable Caching**: Use browser caching for faster loading:
   ```javascript
   env.useBrowserCache = true;
   env.cacheVersion = '1.0'; // Update when model changes
   ```

2. **Use Batching**: Process multiple inputs at once:
   ```javascript
   const results = await model(['Text 1', 'Text 2', 'Text 3']);
   ```

3. **Enable Quantization**: Use 8-bit quantization:
   ```python
   export_model_for_webgpu(model_name="...", quantized=True)
   ```

## Advanced Deployment Techniques

### Progressive Loading

For larger models, implement progressive loading:

```javascript
// Load model components progressively
async function loadModelProgressively() {
    // Show loading progress UI
    updateLoadingUI(0);
    
    // Load tokenizer first (smaller)
    updateLoadingUI(10);
    const tokenizer = await loadTokenizer();
    updateLoadingUI(30);
    
    // Load model configuration
    const config = await loadConfig();
    updateLoadingUI(40);
    
    // Load model weights in chunks
    const totalChunks = 5;
    for (let i = 0; i < totalChunks; i++) {
        await loadModelChunk(i);
        updateLoadingUI(40 + (i + 1) * (60 / totalChunks));
    }
    
    // Finalize model
    const model = await finalizeModel(tokenizer, config);
    updateLoadingUI(100);
    
    return model;
}
```

### Web Workers

Move inference to a Web Worker to prevent UI blocking:

```javascript
// Create a worker
const worker = new Worker('model-worker.js');

// Send data to worker
worker.postMessage({
    action: 'run',
    input: 'Text to classify'
});

// Receive results
worker.onmessage = function(e) {
    if (e.data.action === 'result') {
        displayResults(e.data.result);
    }
};
```

### Service Workers

Cache models using Service Workers for offline use:

```javascript
// Register service worker
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js')
        .then(reg => console.log('Service Worker registered'))
        .catch(err => console.error('Service Worker error:', err));
}

// In sw.js
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open('model-cache-v1').then(cache => {
            return cache.addAll([
                '/models/model.onnx',
                '/models/tokenizer.json'
            ]);
        })
    );
});
```

## Troubleshooting

### Common Issues

1. **CORS Errors**: Set proper CORS headers on your server:
   ```
   Access-Control-Allow-Origin: *
   ```

2. **Memory Limitations**: If you encounter memory errors:
   - Use smaller models
   - Implement quantization
   - Process inputs in smaller chunks

3. **WebNN Not Available**: If WebNN is not available, the implementation will fall back to WASM. Check browser support using:
   ```javascript
   if (navigator.ml && navigator.ml.getNeuralNetworkContext) {
       console.log("WebNN is supported");
   } else {
       console.log("WebNN is not supported, using fallback");
   }
   ```

4. **WebGPU Not Available**: Check WebGPU support:
   ```javascript
   if (navigator.gpu) {
       console.log("WebGPU is supported");
   } else {
       console.log("WebGPU is not supported, using fallback");
   }
   ```

5. **Slow Loading**: Implement progressive loading or loading indicators to improve user experience.

## Browser Compatibility

| Browser | WebNN Support | WebGPU Support | Notes |
|---------|---------------|----------------|-------|
| Chrome 113+ | ✅ | ✅ | Best performance for both backends |
| Edge 113+ | ✅ | ✅ | Based on Chromium, similar to Chrome |
| Safari 16.4+ | ❌ (Polyfill) | ✅ | Native WebGPU, polyfill for WebNN |
| Firefox 115+ | ❌ (Polyfill) | ✅ (Experimental) | Enable WebGPU in about:config |
| Opera 99+ | ✅ | ✅ | Based on Chromium |
| Mobile Chrome | ✅ | ✅ | Performance varies by device |
| Mobile Safari | ❌ (Polyfill) | ✅ | iOS 16.4+ for WebGPU |

## Example Projects

Check out these example projects in the repository:

1. **Simple Text Classification**: `/examples/web/text_classification`
2. **Image Classification Demo**: `/examples/web/image_classification`
3. **Progressive Loading Example**: `/examples/web/progressive_loading`
4. **Web Worker Implementation**: `/examples/web/web_worker`
5. **Offline Capable Application**: `/examples/web/offline_app`

## Resources

- [Sample Web Implementation](/test/web_platform_results/sample_web_implementation.html)
- [WebNN API Documentation](https://www.w3.org/TR/webnn/)
- [WebGPU API Documentation](https://www.w3.org/TR/webgpu/)
- [transformers.js Documentation](https://huggingface.co/docs/transformers.js/index)
- [ONNX Runtime Web Documentation](https://onnxruntime.ai/docs/api/js/index.html)