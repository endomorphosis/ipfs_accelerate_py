# WebGPU and transformers.js Integration Guide

This guide explains how to use the WebGPU and transformers.js support added to the IPFS Accelerate Python framework for exporting models to run in browsers with GPU acceleration.

## What is WebGPU and transformers.js?

**WebGPU** is a modern web API for GPU programming in browsers, providing significantly improved performance over WebGL for compute workloads. It's designed for high-performance 3D graphics and computing.

**transformers.js** is a JavaScript port of the Hugging Face Transformers library that can leverage WebGPU for acceleration. It allows you to run Transformer models directly in web browsers with hardware acceleration.

## Export Process for transformers.js

The process involves:

1. Converting a PyTorch model to ONNX format as an intermediate step
2. Converting the ONNX model to transformers.js format
3. Setting up the transformers.js runtime in your web application

### Basic Export Example

```python
from model_export_capability import export_model
from transformers import AutoModel

# Load model
model = AutoModel.from_pretrained("bert-base-uncased")

# Export for transformers.js/WebGPU
success, message = export_model(
    model=model,
    model_id="bert-base-uncased",
    output_path="transformers_js_model_dir",
    export_format="webgpu"
)

if success:
    print(f"Success: {message}")
else:
    print(f"Failed: {message}")
```

## JavaScript Implementation with transformers.js

Below is an example of how to use the exported model in a web application with transformers.js:

```javascript
// Import transformers.js
import { pipeline, env } from '@xenova/transformers';

// Enable WebGPU acceleration
env.backends.onnx.wasm.numThreads = 1; // Disable WASM threading to prefer WebGPU
env.backends.onnx.webgpu.enabled = true; // Enable WebGPU backend

// Check if WebGPU is available
async function checkWebGPUSupport() {
  if (!navigator.gpu) {
    console.log('WebGPU is not supported in this browser');
    return false;
  }
  
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.log('Couldn\'t request WebGPU adapter');
      return false;
    }
    
    const device = await adapter.requestDevice();
    if (!device) {
      console.log('Couldn\'t request WebGPU device');
      return false;
    }
    
    console.log('WebGPU is supported in this browser');
    return true;
  } catch (e) {
    console.log('Error testing WebGPU support:', e);
    return false;
  }
}

// Initialize pipeline with webgpu acceleration
async function createPipeline(task, model) {
  console.log(`Loading ${model} for task: ${task}`);
  
  // Check WebGPU support
  const hasWebGPU = await checkWebGPUSupport();
  
  if (hasWebGPU) {
    console.log('Using WebGPU acceleration');
  } else {
    console.log('WebGPU not available, falling back to WASM');
    env.backends.onnx.webgpu.enabled = false;
  }
  
  // Create pipeline
  const pipe = await pipeline(task, model, {
    progress_callback: (progress) => {
      console.log(`Loading: ${Math.round(progress.progress * 100)}%`);
    }
  });
  
  return pipe;
}

// Run inference
async function runInference(pipe, input) {
  console.log('Running inference on input:', input);
  
  // Measure performance
  const startTime = performance.now();
  const result = await pipe(input);
  const inferenceTime = performance.now() - startTime;
  
  console.log(`Inference completed in ${inferenceTime.toFixed(2)}ms`);
  console.log('Result:', result);
  
  return { result, inferenceTime };
}

// Complete example with text classification
async function textClassificationExample() {
  try {
    // Create pipeline
    const classifier = await createPipeline(
      'text-classification',
      'distilbert-base-uncased-finetuned-sst-2-english'
    );
    
    // Run inference
    const text = "I love transformers.js, it's amazing!";
    const result = await runInference(classifier, text);
    
    document.getElementById('result').textContent = 
      `Sentiment: ${result.result[0].label} (${(result.result[0].score * 100).toFixed(2)}%)`;
      
  } catch (e) {
    console.error('Error:', e);
    document.getElementById('result').textContent = `Error: ${e.message}`;
  }
}

// Call the example when the page loads
window.addEventListener('DOMContentLoaded', textClassificationExample);
```

## Setting Up a Web Project with transformers.js

Here's how to set up a basic web project using transformers.js:

### 1. HTML Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>transformers.js Demo</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
    .result { margin-top: 20px; padding: 10px; border: 1px solid #ccc; }
    .loading { color: #888; }
  </style>
</head>
<body>
  <h1>transformers.js with WebGPU</h1>
  
  <div>
    <h2>Text Classification</h2>
    <input type="text" id="text-input" value="I love AI and machine learning!" />
    <button id="run-button">Run</button>
    
    <div class="result">
      <div id="loading" class="loading">Ready</div>
      <div id="result"></div>
    </div>
  </div>
  
  <script type="module" src="app.js"></script>
</body>
</html>
```

### 2. Install transformers.js

Using npm:

```bash
npm install @xenova/transformers
```

Or using a CDN:

```html
<script src="https://cdn.jsdelivr.net/npm/@xenova/transformers"></script>
```

### 3. Build Setup with Webpack/Vite

If you're using a build system like Webpack or Vite, configure it to handle WebAssembly:

**Webpack Config:**
```javascript
module.exports = {
  // ... other webpack config
  experiments: {
    asyncWebAssembly: true,
  },
};
```

**Vite Config:**
```javascript
// vite.config.js
export default {
  optimizeDeps: {
    exclude: ['@xenova/transformers']
  }
};
```

## Browser Compatibility

WebGPU support varies across browsers:

| Browser | WebGPU Support | transformers.js Support | Notes |
|---------|----------------|-------------------------|-------|
| Chrome  |  (v113+)     |  Fully supported      | Best performance and compatibility |
| Edge    |  (v113+)     |  Fully supported      | Based on Chromium, similar to Chrome |
| Firefox |   (behind flag) |   Limited           | Enable `dom.webgpu.enabled` in about:config |
| Safari  |  (v17+)      |  Supported in newest versions | Support improving in recent versions |

## Performance Considerations

- **Model Size**: Smaller models (under 100MB) work best with transformers.js
- **Precision**: FP16 models provide better performance while maintaining accuracy
- **First Inference**: The first inference is slower due to compilation overhead
- **Memory Usage**: Monitor memory usage in larger models to avoid browser crashes
- **Threading**: Disable WASM threading when using WebGPU for best performance
- **Caching**: Enable model caching to improve load times on subsequent visits

## Integration with IPFS Accelerate Test Framework

The test framework now includes WebGPU testing capabilities:

```python
# Test on WebGPU platform with transformers.js
def test_webgpu_platform():
    """Test the model on WebGPU platform with transformers.js"""
    model = TestHFBert()
    endpoint, processor, handler, queue, batch_size = model.init_webgpu()
    
    # Test inference
    result = handler("Example text for testing")
    
    # Check implementation type
    assert result["implementation_type"] in ["SIMULATED_WEBGPU_TRANSFORMERS_JS", "REAL_WEBGPU"]
    
    return result
```

## Model Support in transformers.js

transformers.js supports many (but not all) model architectures from Hugging Face:

| Architecture | Support | Notes |
|--------------|---------|-------|
| BERT         |      | Fully supported |
| DistilBERT   |      | Fully supported |
| RoBERTa      |      | Fully supported |
| T5           |      | Fully supported |
| BART         |      | Fully supported |
| ViT          |      | Fully supported |
| CLIP         |      | Fully supported |
| Whisper      |      | Supported with limitations |
| GPT-2        |      | Supported with limitations |
| LLaMA        | L     | Not yet supported |
| Stable Diffusion | L  | Not yet supported |

## Using Custom Models

To use your own custom models with transformers.js:

1. Export your model using the export_model function with format="webgpu"
2. Copy the exported files to your web server's static files directory
3. In JavaScript, specify the path to your model:

```javascript
// Use a custom local model
const classifier = await pipeline('text-classification', './models/my-custom-model');
```

## Troubleshooting WebGPU and transformers.js Issues

Common issues:

1. **WebGPU Not Available**: Check browser compatibility and ensure hardware supports WebGPU
2. **Slow First Inference**: This is normal - WebGPU compiles shaders on first run
3. **Out of Memory**: Reduce model size or use quantized models
4. **Model Not Supported**: Check if your model architecture is supported in transformers.js
5. **CORS Issues**: Ensure proper CORS headers when loading models from different domains

## Additional Resources

- [transformers.js Documentation](https://huggingface.co/docs/transformers.js)
- [WebGPU API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
- [transformers.js GitHub Repository](https://github.com/xenova/transformers.js)