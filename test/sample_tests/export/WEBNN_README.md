# WebNN Export and Integration Guide

This guide explains how to use the WebNN support added to the IPFS Accelerate Python framework for exporting models to WebNN format and running them in browsers.

## What is WebNN?

The Web Neural Network API (WebNN) is a web standard that provides hardware-accelerated neural network inference across different platforms and devices in web browsers. It allows you to run ML models efficiently in browsers by leveraging the underlying hardware acceleration capabilities.

## Export Process

The export process involves:

1. Converting a PyTorch model to ONNX format
2. Optimizing the ONNX model for web deployment
3. Setting up WebNN runtime in JavaScript

### Basic Export Example

```python
from model_export_capability import export_model
from transformers import AutoModel

# Load model
model = AutoModel.from_pretrained("bert-base-uncased")

# Export to WebNN format
success, message = export_model(
    model=model,
    model_id="bert-base-uncased",
    output_path="webnn_model_dir",
    export_format="webnn"
)

if success:
    print(f"Success: {message}")
else:
    print(f"Failed: {message}")
```

## WebNN JavaScript Implementation

Below is an example of how to use the exported model in a web application using WebNN:

```javascript
// WebNN inference code for BERT model
import * as ort from 'onnxruntime-web';

// Create WebNN execution context
async function createWebNNContext() {
  try {
    if ('ml' in navigator) {
      // Use WebNN API directly if available in browser
      const context = await navigator.ml.createContext({ type: 'gpu' });
      return context;
    } else {
      console.warn('WebNN API not available, falling back to wasm backend');
      return null;
    }
  } catch (e) {
    console.warn('Error creating WebNN context:', e);
    return null;
  }
}

// Load model
async function loadModel(modelPath) {
  try {
    // Try to create WebNN context
    const webnnContext = await createWebNNContext();
    
    // Set up execution providers
    const options = {
      executionProviders: ['webnn']
    };
    
    // If WebNN is available, use it as the preferred provider
    if (webnnContext) {
      options.executionProviders = ['webnn', 'wasm'];
      options.webnn = {
        context: webnnContext
      };
    } else {
      options.executionProviders = ['wasm'];
    }
    
    // Create session with specified options
    const session = await ort.InferenceSession.create(modelPath, options);
    return session;
  } catch (e) {
    console.error('Failed to load model:', e);
    throw e;
  }
}

// Run inference
async function runInference(session, inputData) {
  try {
    // Convert inputs to ONNX tensors
    const feeds = {};
    for (const [name, data] of Object.entries(inputData)) {
      feeds[name] = new ort.Tensor(
        name === 'input_ids' || name === 'attention_mask' ? 'int64' : 'float32',
        data,
        Array.isArray(data) ? [1, data.length] : data.shape
      );
    }
    
    // Measure inference time
    const startTime = performance.now();
    const results = await session.run(feeds);
    const inferenceTime = performance.now() - startTime;
    
    console.log(`Inference completed in ${inferenceTime.toFixed(2)}ms`);
    return { results, inferenceTime };
  } catch (e) {
    console.error('Inference failed:', e);
    throw e;
  }
}

// Full pipeline
async function webnnPipeline(inputData, modelPath) {
  // Load model
  const model = await loadModel(modelPath);
  
  // Run inference
  const result = await runInference(model, inputData);
  
  return result;
}

export { webnnPipeline };
```

## Testing WebNN Support in Your Browser

To test if your browser supports WebNN:

```javascript
async function checkWebNNSupport() {
  // Check if WebNN API is available
  if (!('ml' in navigator)) {
    console.log('WebNN API is not available in this browser');
    return false;
  }
  
  try {
    // Try to create a WebNN context
    const context = await navigator.ml.createContext();
    
    if (context) {
      // Try to get device information
      const deviceInfo = await context.getDeviceInfo();
      console.log('WebNN is supported on this browser');
      console.log('Device info:', deviceInfo);
      return true;
    }
  } catch (e) {
    console.log('Error testing WebNN support:', e);
  }
  
  return false;
}

// Check support
checkWebNNSupport().then(supported => {
  if (supported) {
    // WebNN is supported, proceed with using it
    console.log('WebNN is supported, you can use hardware acceleration');
  } else {
    // WebNN is not supported, fall back to WASM
    console.log('WebNN is not supported, falling back to WASM backend');
  }
});
```

## Browser Compatibility

WebNN support varies across browsers:

| Browser | WebNN Support | Notes |
|---------|---------------|-------|
| Chrome  |  (recent versions) | Best support via Chrome Origin Trial |
| Edge    |  (recent versions) | Based on Chromium, similar support |
| Safari  |   (partial) | Limited support in recent versions |
| Firefox | L (not yet) | No native support yet, fallback to WASM |

## Performance Considerations

- **Model Size**: Keep models under 50MB for best performance in browsers
- **Precision**: Use FP16 for WebNN to balance speed and accuracy
- **Operations**: Not all operations are supported in WebNN; stick to common ops
- **Memory Usage**: Monitor memory usage to avoid browser crashes
- **Batch Size**: Use batch size of 1 for most web applications
- **Threading**: WebNN runs on separate threads for UI responsiveness

## Integration with IPFS Accelerate Test Framework

The test framework now includes WebNN testing capabilities to verify your model's compatibility with WebNN:

```python
# Test on WebNN platform
def test_webnn_platform():
    """Test the model on WebNN platform"""
    model = TestHFBert()
    endpoint, processor, handler, queue, batch_size = model.init_webnn()
    
    # Test inference
    result = handler("Example text for testing")
    
    # Check implementation type
    assert result["implementation_type"] in ["SIMULATED_WEBNN", "REAL_WEBNN"]
    
    return result
```

This allows you to easily verify if your model will work with WebNN before deploying it to web environments.

## Troubleshooting WebNN Issues

Common issues when using WebNN:

1. **Browser Support**: Not all browsers support WebNN. Use feature detection.
2. **Operation Support**: Some operations may not be supported. Check console for errors.
3. **Memory Limitations**: Browsers have memory limits. Use smaller models.
4. **Precision Issues**: FP16 may cause precision loss. Test outputs carefully.
5. **CORS Issues**: When loading models from different domains, ensure CORS headers are set correctly.

For further assistance, check browser console logs for detailed error messages.