# IPFS Accelerate API Documentation

This document provides comprehensive API documentation for both the Python and TypeScript SDKs of IPFS Accelerate, enabling hardware-accelerated AI models across server environments and web browsers.

## Python SDK API

### Core API

```python
from ipfs_accelerate import create_accelerator, accelerate

# Create an accelerator with automatic hardware detection
accelerator = create_accelerator(auto_detect_hardware=True)

# Run inference with automatic hardware selection
result = accelerate(
    model_id="bert-base-uncased",
    model_type="text",
    input="This is a sample text for embedding."
)
```

### Hardware Abstraction Layer (HAL)

```python
from ipfs_accelerate.hardware import (
    HardwareAbstraction,
    detect_hardware_capabilities,
    get_optimal_backend
)

# Get hardware capabilities
capabilities = detect_hardware_capabilities()

# Create hardware abstraction layer with specific backends
hal = HardwareAbstraction(
    backend_order=["cuda", "openvino", "cpu"],
    model_preferences={"text": "cuda", "vision": "openvino"}
)

# Initialize hardware backends
hal.initialize()

# Get optimal backend for a specific model type
backend = get_optimal_backend("text", capabilities)

# Execute operation on specific backend
result = hal.execute(
    operation="matmul",
    inputs={"a": tensor_a, "b": tensor_b},
    backend="cuda"
)
```

### Resource Pool

```python
from ipfs_accelerate.resource_pool import ResourcePool

# Create a resource pool for efficient resource management
pool = ResourcePool(
    max_memory=8 * 1024 * 1024 * 1024,  # 8GB
    device_prefixes=["cuda", "cpu"],
    enable_memory_tracking=True
)

# Register a model with the pool
pool.register_model("bert-base-uncased", model_obj)

# Get a model from the pool (loaded if available, or loaded on demand)
model = pool.get_model("bert-base-uncased")

# Run inference using the model
result = model(inputs)
```

### Hardware Detection

```python
from ipfs_accelerate.hardware.detection import (
    detect_hardware,
    check_cuda_support,
    check_openvino_support,
    check_mps_support,
    get_ram_info
)

# Detect all hardware capabilities
hardware = detect_hardware()

# Check specific hardware support
cuda_info = check_cuda_support()
openvino_info = check_openvino_support()
mps_info = check_mps_support()
ram_info = get_ram_info()

print(f"CUDA available: {cuda_info['available']}")
print(f"CUDA devices: {cuda_info['device_count']}")
print(f"CUDA memory: {cuda_info['memory']}")
print(f"CUDA compute capability: {cuda_info['compute_capability']}")
```

### Model Loading

```python
from ipfs_accelerate.models import (
    load_model,
    list_available_models,
    get_model_info
)

# Get available models
models = list_available_models()

# Get model information
model_info = get_model_info("bert-base-uncased")

# Load a model with specific configuration
model = load_model(
    model_id="bert-base-uncased",
    model_type="text",
    device="cuda",
    quantization="int8",
    batch_size=8
)

# Run inference
result = model(inputs)
```

### Benchmark Tools

```python
from ipfs_accelerate.benchmark import (
    benchmark_model,
    compare_hardware,
    profile_memory,
    generate_report
)

# Benchmark a model on specific hardware
results = benchmark_model(
    model_id="bert-base-uncased",
    hardware=["cuda", "openvino", "cpu"],
    batch_sizes=[1, 2, 4, 8, 16],
    iterations=100,
    warmup=10
)

# Compare hardware performance
comparison = compare_hardware(
    model_id="bert-base-uncased",
    hardware=["cuda", "openvino", "cpu"],
    metric="throughput"
)

# Profile memory usage
memory_profile = profile_memory(
    model_id="bert-base-uncased",
    hardware="cuda",
    batch_sizes=[1, 8, 16]
)

# Generate benchmark report
report = generate_report(
    results=results,
    format="html",
    output_path="benchmark_report.html"
)
```

## TypeScript SDK API

The TypeScript SDK provides hardware-accelerated AI models directly in web browsers using WebGPU and WebNN.

### Core API

```typescript
import { createAccelerator } from 'ipfs-accelerate';

async function runInference() {
  // Create accelerator with automatic hardware detection
  const accelerator = await createAccelerator({
    autoDetectHardware: true
  });
  
  // Run inference
  const result = await accelerator.accelerate({
    modelId: 'bert-base-uncased',
    modelType: 'text',
    input: 'This is a sample text for embedding.'
  });
  
  console.log(result);
}
```

### Hardware Abstraction Layer (HAL)

```typescript
import { createHardwareAbstraction } from 'ipfs-accelerate/hardware';

async function runMatrixMultiplication() {
  // Create HAL with specific backend order
  const hal = await createHardwareAbstraction({
    backendOrder: ['webgpu', 'webnn', 'wasm', 'cpu']
  });
  
  // Create tensors
  const a = new Float32Array([1, 2, 3, 4]);
  const b = new Float32Array([5, 6, 7, 8]);
  
  const tensorA = await hal.getBackend('webgpu')?.createTensor(a, [2, 2]);
  const tensorB = await hal.getBackend('webgpu')?.createTensor(b, [2, 2]);
  
  // Execute matrix multiplication
  const result = await hal.execute('matmul', {
    a: tensorA,
    b: tensorB
  });
  
  // Get backend capabilities
  const capabilities = hal.getCapabilities();
  
  // Check available backends
  const backends = hal.getAvailableBackends();
  
  // Get the best backend for a specific model type
  const bestBackend = hal.getBestBackend('text');
}
```

### Hardware Detection

```typescript
import { detectHardwareCapabilities } from 'ipfs-accelerate/hardware';

async function checkHardware() {
  // Detect hardware capabilities
  const capabilities = await detectHardwareCapabilities();
  
  console.log(`Browser: ${capabilities.browserName} ${capabilities.browserVersion}`);
  console.log(`Platform: ${capabilities.platform} ${capabilities.osVersion}`);
  console.log(`Mobile device: ${capabilities.isMobile}`);
  console.log(`WebGPU supported: ${capabilities.webgpuSupported}`);
  console.log(`WebNN supported: ${capabilities.webnnSupported}`);
  console.log(`WebAssembly supported: ${capabilities.wasmSupported}`);
  console.log(`Recommended backend: ${capabilities.recommendedBackend}`);
  console.log(`Memory limit: ${capabilities.memoryLimitMB} MB`);
  
  // Check for browser-specific optimizations
  const { browserOptimizations } = capabilities;
  if (browserOptimizations.audioOptimized) {
    console.log('This browser has optimized audio model performance');
  }
  if (browserOptimizations.shaderPrecompilation) {
    console.log('This browser supports shader precompilation for faster startup');
  }
  if (browserOptimizations.parallelLoading) {
    console.log('This browser supports parallel model loading');
  }
}
```

### React Integration

```typescript
import React, { useState } from 'react';
import { useModel, useHardwareInfo } from 'ipfs-accelerate/react';

function BertEmbeddingComponent() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  
  // Use hardware detection hook
  const { capabilities, isReady, optimalBackend } = useHardwareInfo();
  
  // Use model hook
  const { model, status, error, loadModel } = useModel({
    modelId: 'bert-base-uncased',
    modelType: 'text',
    autoLoad: true,
    autoHardwareSelection: true
  });
  
  // Handle text input change
  const handleChange = (e) => {
    setText(e.target.value);
  };
  
  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (model && status === 'loaded') {
      try {
        const output = await model.execute(text);
        setResult(output);
      } catch (err) {
        console.error('Inference error:', err);
      }
    }
  };
  
  return (
    <div>
      <h2>BERT Text Embedding</h2>
      
      {/* Hardware info */}
      {isReady && (
        <div>
          <p>Running on: {optimalBackend}</p>
          <p>Browser: {capabilities.browserName} {capabilities.browserVersion}</p>
        </div>
      )}
      
      {/* Model status */}
      <div>
        Status: {status}
        {error && <p style={{ color: 'red' }}>Error: {error.message}</p>}
      </div>
      
      {/* Input form */}
      <form onSubmit={handleSubmit}>
        <textarea 
          value={text} 
          onChange={handleChange} 
          placeholder="Enter text to embed"
          rows={4}
          cols={50}
        />
        <button type="submit" disabled={status !== 'loaded'}>
          Generate Embedding
        </button>
      </form>
      
      {/* Results */}
      {result && (
        <div>
          <h3>Results</h3>
          <pre>
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
```

### WebGPU Operations

The WebGPU backend supports various operations for hardware-accelerated computation:

```typescript
// Matrix multiplication
const resultMatrix = await webgpuBackend.execute('matmul', {
  a: tensorA,
  b: tensorB
}, {
  transposeA: false,
  transposeB: false
});

// Element-wise operations
const resultRelu = await webgpuBackend.execute('elementwise', {
  input: tensor
}, {
  operation: 'relu'
});

const resultSigmoid = await webgpuBackend.execute('elementwise', {
  input: tensor
}, {
  operation: 'sigmoid'
});

const resultTanh = await webgpuBackend.execute('elementwise', {
  input: tensor
}, {
  operation: 'tanh'
});

// Softmax
const resultSoftmax = await webgpuBackend.execute('softmax', {
  input: tensor
}, {
  axis: -1
});

// Quantization (float32 to int8)
const resultQuantized = await webgpuBackend.execute('quantize', {
  input: tensor
});

// Dequantization (int8 to float32)
const resultDequantized = await webgpuBackend.execute('dequantize', {
  input: quantizedTensor,
  scale: scaleTensor
});
```

### WebNN Operations

The WebNN backend supports neural network operations through the WebNN API:

```typescript
// Create tensors on WebNN
const tensorA = await webnnBackend.createTensor(
  new Float32Array([1, 2, 3, 4]), [2, 2], 'float32'
);

const tensorB = await webnnBackend.createTensor(
  new Float32Array([5, 6, 7, 8]), [2, 2], 'float32'
);

// Matrix multiplication
const resultMatrix = await webnnBackend.execute('matmul', {
  a: tensorA,
  b: tensorB
}, {
  transposeA: false,
  transposeB: false
});

// Element-wise operations
const resultRelu = await webnnBackend.execute('elementwise', {
  input: tensor
}, {
  operation: 'relu'
});

const resultSigmoid = await webnnBackend.execute('elementwise', {
  input: tensor
}, {
  operation: 'sigmoid'
});

const resultTanh = await webnnBackend.execute('elementwise', {
  input: tensor
}, {
  operation: 'tanh'
});

// Softmax operation
const resultSoftmax = await webnnBackend.execute('softmax', {
  input: tensor
}, {
  axis: -1
});

// 2D Convolution operation
const filterTensor = await webnnBackend.createTensor(
  new Float32Array([1, 2, 1, 0, 1, 0, 1, 2, 1]), [1, 1, 3, 3], 'float32'
);

const resultConv = await webnnBackend.execute('conv2d', {
  input: tensor,
  filter: filterTensor
}, {
  padding: [1, 1, 1, 1],
  strides: [1, 1]
});

// Read tensor data back to CPU
const resultData = await webnnBackend.readTensor(
  resultMatrix.tensor, 
  resultMatrix.shape, 
  'float32'
);
```

### Advanced Configuration

```typescript
// Advanced accelerator options
const accelerator = await createAccelerator({
  // Hardware settings
  autoDetectHardware: true,
  backendOrder: ['webgpu', 'webnn', 'wasm', 'cpu'],
  modelPreferences: {
    text: 'webnn',
    vision: 'webgpu',
    audio: 'webgpu'
  },
  backendOptions: {
    webgpu: {
      powerPreference: 'high-performance',
      shaderCompilation: {
        precompile: true,
        browserOptimizations: true
      }
    }
  },
  
  // Memory optimizations
  enableTensorSharing: true,
  enableMemoryOptimization: true,
  
  // Browser optimizations
  enableBrowserOptimizations: true,
  
  // Storage settings
  storage: {
    enablePersistence: true,
    maxCacheSize: 1024 * 1024 * 1024 // 1GB
  },
  
  // P2P settings
  p2p: {
    enableModelSharing: false,
    enableTensorSharing: false
  }
});

// Advanced inference options
const result = await accelerator.accelerate({
  modelId: 'bert-base-uncased',
  modelType: 'text',
  input: 'This is a sample text for embedding.',
  backend: 'webgpu',
  modelOptions: {
    // Model-specific options
    quantized: true,
    shared: true
  },
  inferenceOptions: {
    batchSize: 4,
    precision: 'int8',
    streaming: false
  }
});
```

## Browser Compatibility

The TypeScript SDK has been tested and confirmed to work on the following browsers:

| Browser | Version | WebGPU | WebNN | Notes |
|---------|---------|--------|-------|-------|
| Chrome | 121+ | ✅ | ✅ | Full WebGPU support, good WebNN implementation |
| Edge | 121+ | ✅ | ✅ | Best WebNN support with superior performance |
| Firefox | 124+ | ✅ | ❌ | Best compute shader performance for audio models |
| Safari | 17.4+ | ✅ | ❌ | WebGPU support is still experimental |

Each backend has specific advantages based on hardware and browser:

- **WebGPU Backend**: Offers the best performance for compute-intensive operations, especially on Firefox for audio models with up to 55% better performance than other browsers.
- **WebNN Backend**: Provides optimized neural network operations on browsers that support it, with Edge offering the most complete and performant implementation.
- **Automatic Selection**: The SDK intelligently selects the optimal backend for each operation based on the browser, hardware, and model type.

### Simulation Validation Database Integration API

The framework includes a comprehensive database integration system for the Simulation Accuracy and Validation Framework, allowing storage, retrieval, and analysis of simulation validation data:

```python
from duckdb_api.simulation_validation.db_integration import SimulationValidationDBIntegration
from duckdb_api.simulation_validation.simulation_validation_framework import get_framework_instance

# Initialize database integration
db_integration = SimulationValidationDBIntegration(
    db_path="benchmark_db.duckdb"
)

# Initialize database schema
db_integration.initialize_database()

# Store various result types
db_integration.store_simulation_results(simulation_results)
db_integration.store_hardware_results(hardware_results)
db_integration.store_validation_results(validation_results)
db_integration.store_calibration_parameters(calibration_params)
db_integration.store_drift_detection_results(drift_results)

# Retrieve results by criteria
hw_results = db_integration.get_simulation_results_by_hardware("gpu_rtx3080")
model_results = db_integration.get_hardware_results_by_model("bert-base-uncased")
validation_results = db_integration.get_validation_results_by_criteria(
    hardware_id="gpu_rtx3080",
    model_id="bert-base-uncased",
    batch_size=16,
    precision="fp16"
)

# Get latest calibration parameters
latest_params = db_integration.get_latest_calibration_parameters()

# Get drift detection history
drift_history = db_integration.get_drift_detection_history(
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased"
)

# Get MAPE by hardware and model
mape_results = db_integration.get_mape_by_hardware_and_model()

# Analyze calibration effectiveness
effectiveness = db_integration.analyze_calibration_effectiveness(
    before_version="uncalibrated_v1.0",
    after_version="calibrated_v1.0"
)

# Export visualization data
db_integration.export_visualization_data(
    export_path="visualization_data.json",
    metrics=["throughput_items_per_second", "average_latency_ms"]
)

# Integrate with the framework
framework = get_framework_instance()
framework.set_db_integration(db_integration)

# Now use framework methods with automatic database integration
validation_results = framework.validate(simulation_results, hardware_results)
framework.store_validation_results(validation_results)
```

## Further Documentation

For more detailed information, please refer to the following resources:

- [IPFS Accelerate TypeScript Implementation Summary](TYPESCRIPT_IMPLEMENTATION_SUMMARY.md) - Detailed overview of the TypeScript implementation
- [SDK Documentation](SDK_DOCUMENTATION.md) - Comprehensive guide to using both Python and TypeScript SDKs
- [OpenAI API Enhancements (July 2025)](OPENAI_API_ENHANCEMENTS.md) - Comprehensive guide to the enhanced OpenAI API implementations in Python and TypeScript
- [IPFS WebNN/WebGPU SDK Guide](IPFS_WEBNN_WEBGPU_SDK_GUIDE.md) - Complete guide to using WebNN and WebGPU acceleration
- [Web Resource Pool Integration](WEB_RESOURCE_POOL_INTEGRATION.md) - Resource pool integration with browser-based WebNN/WebGPU acceleration
- [Real WebNN/WebGPU Benchmarking Guide](REAL_WEBNN_WEBGPU_BENCHMARKING_GUIDE.md) - Comprehensive guide for running real WebNN/WebGPU benchmarks
- [Simulation Validation Framework Documentation](SIMULATION_ACCURACY_VALIDATION_IMPLEMENTATION.md) - Complete guide to the Simulation Accuracy and Validation Framework
- [Database Integration Summary](duckdb_api/simulation_validation/db_integration_summary.md) - Detailed documentation on the database integration implementation
- [OpenAI Backend Usage](ipfs_accelerate_js/docs/api_backends/OPENAI_BACKEND_USAGE.md) - Detailed usage guide for the JavaScript OpenAI API implementation
- [OpenAI Mini Backend Usage](ipfs_accelerate_js/docs/api_backends/OPENAI_MINI_BACKEND_USAGE.md) - Usage guide for the lightweight JavaScript OpenAI API implementation