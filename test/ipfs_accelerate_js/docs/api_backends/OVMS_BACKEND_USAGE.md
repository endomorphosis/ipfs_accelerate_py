# OpenVINO Model Server (OVMS) Backend

The `OVMS` backend provides integration with OpenVINO Model Server, a high-performance, production-grade model serving solution optimized for inference on Intel hardware and beyond. OVMS enables deploying AI models at the edge or in the cloud with excellent performance on CPUs and Intel's specialized hardware accelerators.

This implementation extends the `BaseApiBackend` class, providing a standardized interface that works with the IPFS Accelerate API ecosystem. It can be used alongside other API backends through the unified API interface.

## Features

- Inference with OpenVINO-optimized models
- Multiple model version support with automatic version management
- Batch inference for high-throughput scenarios
- Comprehensive model management and configuration
- Server statistics and performance monitoring
- Detailed model metadata and shape information
- Advanced quantization configuration for INT8/INT4 inference
- Execution mode optimization (latency vs. throughput)
- Model status and health checking
- Model explanations for model interpretability (with supported models)
- Well-defined TypeScript types for improved developer experience

## Installation

The OVMS backend is included in the IPFS Accelerate JavaScript SDK. Ensure you have the `ipfs_accelerate_js` package installed.

```bash
npm install ipfs-accelerate
```

## Basic Usage

```typescript
import { OVMS } from 'ipfs_accelerate_js/api_backends';

// Initialize with API URL and model
const ovms = new OVMS({}, {
  ovms_api_url: 'http://localhost:9000',
  ovms_model: 'bert'
});

// Basic inference
async function runInference() {
  // Sample input data - format depends on your model's input requirements
  const inputData = [0, 1, 2, 3, 4];
  
  // Run inference with default model
  const result = await ovms.infer(undefined, inputData);
  console.log('Inference result:', result);
  
  // Or specify a different model
  const customResult = await ovms.infer('custom_model', inputData);
  console.log('Custom model result:', customResult);
}

// Batch inference for higher throughput
async function runBatchInference() {
  // Sample batch data - multiple inputs to process at once
  const batchData = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11]
  ];
  
  // Run batch inference
  const results = await ovms.batchInfer('bert', batchData);
  
  console.log(`Processed ${results.length} inputs in one batch`);
  console.log('Batch inference results:', results);
}
```

## Using with the API Backend Factory

The OVMS backend can be used with the API backend factory for easy creation and model compatibility detection:

```typescript
import { createApiBackend, findCompatibleBackend } from 'ipfs_accelerate_js/api_backends';

// Create backend by name
const ovms = createApiBackend('ovms', {}, {
  ovms_api_url: 'http://localhost:9000',
  ovms_model: 'bert'
});

// Find compatible backend for a model
const modelName = 'bert';
const backend = findCompatibleBackend(modelName);
if (backend) {
  console.log(`Found compatible backend: ${backend.constructor.name}`);
  
  // Run inference with the automatically selected backend
  const result = await backend.infer(modelName, [0, 1, 2, 3]);
  console.log('Inference result:', result);
}
```

## Advanced Usage

### Custom API Endpoint and Model Configuration

```typescript
// For self-hosted OVMS deployments with custom configuration
const customOVMS = new OVMS({}, {
  ovms_api_key: 'your_api_key', // Optional authentication for secure deployments
  ovms_model: 'bert',
  ovms_api_url: 'http://custom-ovms-server.com:9000',
  ovms_version: '2',  // Specific model version
  ovms_precision: 'FP16',  // Precision to use
  timeout: 60000  // 60 second timeout for long-running inferences
});

// Run inference with specific options
const result = await customOVMS.infer('bert', inputData, {
  version: '3',  // Override version for this request only
  precision: 'INT8',  // Use INT8 precision for this request only
  timeout: 120000  // 120 second timeout for this specific request
});
```

### Endpoint Handlers for Advanced Control

```typescript
// Create endpoint handlers for different models or endpoints
const ovms = new OVMS();

// Create endpoint handler for a specific model
const bertHandler = ovms.createEndpointHandler(
  'http://localhost:9000/v1/models/bert:predict',
  'bert'
);

// Create handler for a different model
const resnetHandler = ovms.createEndpointHandler(
  'http://localhost:9000/v1/models/resnet:predict', 
  'resnet'
);

// Use the handlers directly
const bertResult = await ovms.formatRequest(bertHandler, [0, 1, 2, 3]);
const resnetResult = await ovms.formatRequest(resnetHandler, imageArray);

// Testing if endpoints are available before using
const isBertAvailable = await ovms.testEndpoint(
  'http://localhost:9000/v1/models/bert:predict',
  'bert'
);

if (isBertAvailable) {
  console.log('BERT model is available for inference');
} else {
  console.log('BERT model is not available');
}
```

### Model Versions and Version Management

```typescript
// Get available model versions
const ovms = new OVMS();
const modelVersions = await ovms.getModelVersions('bert');
console.log(`Available versions: ${modelVersions.join(', ')}`);

// Run inference with a specific version
const result = await ovms.inferWithVersion('bert', '2', inputData);
console.log('Result from version 2:', result);

// Run inference with a different version
const newResult = await ovms.inferWithVersion('bert', '3', inputData);
console.log('Result from version 3:', newResult);

// Check which version is set as default
const modelInfo = await ovms.getModelInfo('bert');
console.log(`Current default version: ${modelInfo.version || 'latest'}`);
```

### Server and Model Information

```typescript
// Get detailed model information
const ovms = new OVMS();
const modelInfo = await ovms.getModelInfo('bert');
console.log(`Model name: ${modelInfo.name}`);
console.log(`Available versions: ${modelInfo.versions.join(', ')}`);
console.log(`Platform: ${modelInfo.platform}`);

// Input and output shapes are critical for understanding model requirements
if (modelInfo.inputs && modelInfo.inputs.length > 0) {
  console.log(`Input name: ${modelInfo.inputs[0].name}`);
  console.log(`Input datatype: ${modelInfo.inputs[0].datatype}`);
  console.log(`Input shape: [${modelInfo.inputs[0].shape.join(', ')}]`);
}

// Get detailed model status including loading state
const status = await ovms.getModelStatus('bert');
console.log(`Model state: ${status.state}`);
console.log(`Model health: ${status.health || 'Unknown'}`);

// Get comprehensive server statistics
const statistics = await ovms.getServerStatistics();
console.log(`Server uptime: ${statistics.server_uptime} seconds`);
console.log(`Server version: ${statistics.server_version}`);
console.log(`Active models: ${statistics.active_models}`);
console.log(`Total requests: ${statistics.total_requests}`);
console.log(`Requests/second: ${statistics.requests_per_second}`);
console.log(`Avg inference time: ${statistics.avg_inference_time}ms`);
console.log(`CPU usage: ${statistics.cpu_usage}%`);
console.log(`Memory usage: ${statistics.memory_usage}MB`);
```

### Performance Optimization

```typescript
// Reload a model (useful after updating model files)
await ovms.reloadModel('bert');

// Optimize for batch processing
await ovms.setModelConfig('bert', {
  batch_size: 32,        // Maximum batch size
  preferred_batch: 16,   // Preferred batch size for optimal performance
  instance_count: 2      // Number of model instances to run in parallel
});

// Set execution mode for latency-optimized inference (low latency)
await ovms.setExecutionMode('bert', 'latency');
console.log('Model optimized for lowest latency');

// Run latency-sensitive inference
const latencyResult = await ovms.infer('bert', inputData);

// Set execution mode for throughput-optimized inference (high throughput)
await ovms.setExecutionMode('bert', 'throughput');
console.log('Model optimized for maximum throughput');

// Run throughput-oriented batch inference
const throughputResults = await ovms.batchInfer('bert', batchData);
```

### Quantization for Memory and Performance Optimization

```typescript
// Enable INT8 quantization for a model
await ovms.setQuantization('bert', {
  enabled: true,
  method: 'int8',
  bits: 8
});
console.log('Model quantized to INT8 precision');

// Run inference with the quantized model
const quantizedResult = await ovms.infer('bert', inputData);

// Disable quantization if needed for higher accuracy
await ovms.setQuantization('bert', {
  enabled: false
});
console.log('Quantization disabled');
```

### Model Metadata and Explainability

```typescript
// Get comprehensive model metadata with input/output shapes
const metadata = await ovms.getModelMetadataWithShapes('bert');

// Log all inputs with their shapes
metadata.inputs.forEach(input => {
  console.log(`Input: ${input.name}, Type: ${input.datatype}, Shape: [${input.shape.join(', ')}]`);
});

// Log all outputs with their shapes
metadata.outputs.forEach(output => {
  console.log(`Output: ${output.name}, Type: ${output.datatype}, Shape: [${output.shape.join(', ')}]`);
});

// Get explanations for a prediction (if supported by model)
// Note: This requires a model that supports explainability
const explanation = await ovms.explainPrediction('bert', inputData);
console.log('Feature importance:', explanation.feature_importance);
console.log('Explanation factors:', explanation.factors);
```

## Chat Interface Support

While OVMS is primarily designed for inference rather than chat applications, the backend implements the standard chat interface required by BaseApiBackend to maintain API compatibility:

```typescript
// Using the chat interface (adapts to OVMS inference)
const messages = [
  { role: 'user', content: 'Hello world' }
];

// Chat completion (converts to inference)
const response = await ovms.chat(messages);
console.log('Response:', response.content);

// The content from the last user message is extracted and used as input
// for the OVMS model, and the result is returned as a chat response

// Streaming chat (simulates stream with single chunk)
for await (const chunk of ovms.streamChat(messages)) {
  console.log('Chunk:', chunk.content);
  if (chunk.done) break;
}

// Note: OVMS doesn't support true streaming, so this returns
// the entire response as a single chunk
```

## Handling Complex Input Formats

OVMS supports various input formats for different model types:

```typescript
// 1. Array input (e.g., for embeddings)
const arrayInput = [0, 1, 2, 3, 4];
const arrayResult = await ovms.infer('bert', arrayInput);

// 2. Object input (e.g., for structured data)
const objectInput = {
  data: [0, 1, 2, 3],
  mask: [1, 1, 1, 1]
};
const objectResult = await ovms.infer('bert', objectInput);

// 3. Nested array input (e.g., for images)
const imageInput = [
  [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
  [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
];
const imageResult = await ovms.infer('resnet', imageInput);

// 4. Pre-formatted OVMS input
const ovmsFormatInput = {
  instances: [
    {
      data: [0, 1, 2, 3],
      mask: [1, 1, 1, 1]
    }
  ]
};
const formattedResult = await ovms.infer('bert', ovmsFormatInput);
```

## Error Handling

The OVMS backend includes robust error handling with retries and informative error messages:

```typescript
// Handle potential inference errors
try {
  const result = await ovms.infer('bert', inputData);
  console.log('Inference successful:', result);
} catch (error) {
  if (error.status === 404) {
    console.error('Model not found. Check if the model is loaded on the server.');
  } else if (error.status === 400) {
    console.error('Bad request. Check your input format:', error.message);
  } else if (error.code === 'ECONNREFUSED') {
    console.error('Cannot connect to OVMS server. Ensure it\'s running.');
  } else {
    console.error('Inference error:', error);
  }
}
```

## API Reference

### Constructor

```typescript
new OVMS(resources?: ApiResources, metadata?: ApiMetadata)
```

**Parameters:**
- `resources`: Optional resources to pass to the backend
- `metadata`: Configuration options including:
  - `ovms_api_key` or `OVMS_API_KEY`: API key for secure OVMS deployments
  - `ovms_model` or `OVMS_MODEL`: Default model to use (default: 'model')
  - `ovms_api_url` or `OVMS_API_URL`: Base URL for the API (default: 'http://localhost:9000')
  - `ovms_version` or `OVMS_VERSION`: Model version to use (default: 'latest')
  - `ovms_precision` or `OVMS_PRECISION`: Precision to use (default: 'FP32')
  - `timeout` or `OVMS_TIMEOUT`: Request timeout in seconds (default: 30)

### Methods

#### `createEndpointHandler(endpointUrl?: string, model?: string): (data: OVMSRequestData) => Promise<OVMSResponse>`

Creates a handler for a specific endpoint.

**Parameters:**
- `endpointUrl`: Optional URL for the endpoint (default: constructed from apiUrl, model, and ':predict')
- `model`: Optional model name to use with this handler (default: this.modelName)

**Returns:** Function that accepts OVMSRequestData and returns a Promise resolving to OVMSResponse

#### `testEndpoint(endpointUrl?: string, model?: string): Promise<boolean>`

Tests if an endpoint is available and working.

**Parameters:**
- `endpointUrl`: Optional URL to test (default: constructed from apiUrl, model, and ':predict')
- `model`: Optional model name to use in the test (default: this.modelName)

**Returns:** Promise resolving to boolean indicating if the endpoint is available

#### `makePostRequestOVMS(endpointUrl: string, data: OVMSRequestData, options?: OVMSRequestOptions): Promise<OVMSResponse>`

Makes a POST request to the OVMS server.

**Parameters:**
- `endpointUrl`: URL for the request
- `data`: Request data in OVMS format
- `options`: Optional request options including version, timeout, and apiKey

**Returns:** Promise resolving to OVMSResponse

#### `formatRequest(handler: (data: OVMSRequestData) => Promise<OVMSResponse>, input: any): Promise<OVMSResponse>`

Formats input data for OVMS and sends the request using the provided handler.

**Parameters:**
- `handler`: Function to handle the formatted request
- `input`: Input data to format (can be array, object, or already formatted OVMS request data)

**Returns:** Promise resolving to OVMSResponse

#### `getModelInfo(model?: string): Promise<OVMSModelMetadata>`

Gets information about a model.

**Parameters:**
- `model`: Optional model name (default: this.modelName)

**Returns:** Promise resolving to OVMSModelMetadata

#### `getModelVersions(model?: string): Promise<string[]>`

Gets available versions for a model.

**Parameters:**
- `model`: Optional model name (default: this.modelName)

**Returns:** Promise resolving to array of version strings

#### `infer(model?: string, data: any, options?: OVMSRequestOptions): Promise<any>`

Runs inference with a model.

**Parameters:**
- `model`: Optional model name (default: this.modelName)
- `data`: Input data for inference
- `options`: Optional request options

**Returns:** Promise resolving to inference result (typically predictions array)

#### `batchInfer(model?: string, dataBatch: any[], options?: OVMSRequestOptions): Promise<any[]>`

Runs batch inference with a model.

**Parameters:**
- `model`: Optional model name (default: this.modelName)
- `dataBatch`: Array of input data for batch inference
- `options`: Optional request options

**Returns:** Promise resolving to array of inference results

#### `setModelConfig(model?: string, config: OVMSModelConfig): Promise<any>`

Sets configuration for a model.

**Parameters:**
- `model`: Optional model name (default: this.modelName)
- `config`: Configuration options including batch_size, instance_count, etc.

**Returns:** Promise resolving to configuration result

#### `setExecutionMode(model?: string, mode: 'latency' | 'throughput'): Promise<any>`

Sets execution mode for a model.

**Parameters:**
- `model`: Optional model name (default: this.modelName)
- `mode`: Execution mode ('latency' or 'throughput')

**Returns:** Promise resolving to execution mode result

#### `reloadModel(model?: string): Promise<any>`

Reloads a model.

**Parameters:**
- `model`: Optional model name (default: this.modelName)

**Returns:** Promise resolving to reload result

#### `getModelStatus(model?: string): Promise<any>`

Gets the status of a model.

**Parameters:**
- `model`: Optional model name (default: this.modelName)

**Returns:** Promise resolving to model status

#### `getServerStatistics(): Promise<OVMSServerStatistics>`

Gets server statistics.

**Returns:** Promise resolving to OVMSServerStatistics

#### `inferWithVersion(model?: string, version: string, data: any, options?: OVMSRequestOptions): Promise<any>`

Runs inference with a specific model version.

**Parameters:**
- `model`: Optional model name (default: this.modelName)
- `version`: Model version to use
- `data`: Input data for inference
- `options`: Optional request options

**Returns:** Promise resolving to inference result

#### `explainPrediction(model?: string, data: any, options?: OVMSRequestOptions): Promise<any>`

Gets explanation for a prediction.

**Parameters:**
- `model`: Optional model name (default: this.modelName)
- `data`: Input data for explanation
- `options`: Optional request options

**Returns:** Promise resolving to explanation result

#### `getModelMetadataWithShapes(model?: string): Promise<OVMSModelMetadata>`

Gets model metadata with input/output shapes.

**Parameters:**
- `model`: Optional model name (default: this.modelName)

**Returns:** Promise resolving to OVMSModelMetadata including input/output shapes

#### `setQuantization(model?: string, config: OVMSQuantizationConfig): Promise<any>`

Sets quantization for a model.

**Parameters:**
- `model`: Optional model name (default: this.modelName)
- `config`: Quantization configuration

**Returns:** Promise resolving to quantization result

#### `makePostRequest(data: any, apiKey?: string, options?: ApiRequestOptions): Promise<any>`

Makes a POST request to the OVMS server (BaseApiBackend implementation).

**Parameters:**
- `data`: Request data
- `apiKey`: Optional API key
- `options`: Optional request options

**Returns:** Promise resolving to response data

#### `makeStreamRequest(data: any, options?: ApiRequestOptions): AsyncGenerator<StreamChunk>`

Makes a streaming request to the OVMS server (BaseApiBackend implementation).

**Parameters:**
- `data`: Request data
- `options`: Optional request options

**Returns:** AsyncGenerator yielding StreamChunks

#### `chat(messages: Message[], options?: ApiRequestOptions): Promise<ChatCompletionResponse>`

Generates a chat completion from a list of messages (adapts to OVMS inference).

**Parameters:**
- `messages`: Array of message objects
- `options`: Optional request options

**Returns:** Promise resolving to ChatCompletionResponse

#### `streamChat(messages: Message[], options?: ApiRequestOptions): AsyncGenerator<StreamChunk>`

Generates a streaming chat completion (simulates streaming with single chunk).

**Parameters:**
- `messages`: Array of message objects
- `options`: Optional request options

**Returns:** AsyncGenerator yielding StreamChunks

#### `isCompatibleModel(model: string): boolean`

Checks if a model is compatible with OVMS.

**Parameters:**
- `model`: Model name to check

**Returns:** Boolean indicating if the model is compatible

## Interface Reference

### `OVMSRequestData`

Request data for OVMS API.

```typescript
interface OVMSRequestData {
  instances?: any[];         // Array of input instances for batch inference
  inputs?: any;              // Alternative format for inputs
  signature_name?: string;   // Optional signature name for models with multiple signatures
  [key: string]: any;        // Additional fields for model-specific requirements
}
```

### `OVMSResponse`

Response from OVMS API.

```typescript
interface OVMSResponse {
  predictions?: any[];       // Array of prediction results
  outputs?: any;             // Alternative format for outputs
  model_name?: string;       // Name of the model that generated the response
  model_version?: string;    // Version of the model that generated the response
  [key: string]: any;        // Additional fields in the response
}
```

### `OVMSModelMetadata`

Model metadata.

```typescript
interface OVMSModelMetadata {
  name: string;              // Model name
  versions: string[];        // Available model versions
  platform: string;          // Model platform (e.g., "openvino", "tensorflow")
  inputs: OVMSModelInput[];  // Model input specifications
  outputs: OVMSModelOutput[]; // Model output specifications
  [key: string]: any;        // Additional metadata fields
}
```

### `OVMSModelInput`

Model input information.

```typescript
interface OVMSModelInput {
  name: string;              // Input tensor name
  datatype: string;          // Data type (e.g., "FP32", "INT8")
  shape: number[];           // Input shape dimensions
  layout?: string;           // Optional tensor layout (e.g., "NHWC", "NCHW")
  [key: string]: any;        // Additional input specifications
}
```

### `OVMSModelOutput`

Model output information.

```typescript
interface OVMSModelOutput {
  name: string;              // Output tensor name
  datatype: string;          // Data type (e.g., "FP32", "INT8")
  shape: number[];           // Output shape dimensions
  layout?: string;           // Optional tensor layout
  [key: string]: any;        // Additional output specifications
}
```

### `OVMSRequestOptions`

Request options.

```typescript
interface OVMSRequestOptions extends ApiRequestOptions {
  version?: string;          // Model version to use
  shape?: number[];          // Optional shape override
  precision?: string;        // Precision to use (e.g., "FP32", "FP16", "INT8")
  config?: any;              // Additional configuration for the request
  [key: string]: any;        // Other request options
}
```

### `OVMSModelConfig`

Model configuration.

```typescript
interface OVMSModelConfig {
  batch_size?: number;       // Maximum batch size
  preferred_batch?: number;  // Preferred batch size for optimal performance
  instance_count?: number;   // Number of model instances to run in parallel
  execution_mode?: 'latency' | 'throughput'; // Optimization priority
  [key: string]: any;        // Additional configuration options
}
```

### `OVMSServerStatistics`

Server statistics.

```typescript
interface OVMSServerStatistics {
  server_uptime?: number;    // Server uptime in seconds
  server_version?: string;   // OVMS server version
  active_models?: number;    // Number of active models
  total_requests?: number;   // Total processed requests
  requests_per_second?: number; // Request throughput
  avg_inference_time?: number; // Average inference time in milliseconds
  cpu_usage?: number;        // CPU usage percentage
  memory_usage?: number;     // Memory usage in MB
  [key: string]: any;        // Additional statistics
}
```

### `OVMSModelStatistics`

Model statistics.

```typescript
interface OVMSModelStatistics {
  model: string;             // Model name
  statistics: {
    requests_processed?: number; // Number of processed requests
    tokens_generated?: number;   // For text models, number of tokens generated
    avg_inference_time?: number; // Average inference time in milliseconds
    throughput?: number;         // Inferences per second
    errors?: number;             // Number of errors
    [key: string]: any;          // Additional statistics
  };
  [key: string]: any;        // Additional model-specific statistics
}
```

### `OVMSQuantizationConfig`

Quantization configuration.

```typescript
interface OVMSQuantizationConfig {
  enabled: boolean;          // Whether quantization is enabled
  method?: string;           // Quantization method (e.g., "int8", "int4")
  bits?: number;             // Bit width for quantization
  [key: string]: any;        // Additional quantization parameters
}
```

## Common Base Backend Interface

As a class that extends the `BaseApiBackend`, the OVMS backend implements all the standardized methods from the base class, ensuring compatibility with the unified API ecosystem. This allows it to be used interchangeably with other backends like OpenAI, VLLM, or HuggingFace.

## OpenVINO Integration

This backend is specifically designed to work with models optimized for OpenVINO, Intel's toolkit for high-performance deep learning inference. OpenVINO Model Server provides:

1. Optimized inference on Intel CPUs, GPUs, and specialized hardware (including Intel's Neural Compute Stick 2, Vision Processing Units, and FPGAs)
2. Support for multiple frameworks (TensorFlow, PyTorch, ONNX, PaddlePaddle)
3. Advanced model management with versioning and A/B testing
4. High throughput and low latency inference with optimized kernels
5. Scalable deployment options from edge to cloud
6. Built-in model monitoring and metrics collection
7. INT8 and INT4 quantization for improved performance

### Supported Model Types

OVMS can serve various types of models, including but not limited to:

- Computer Vision (classification, detection, segmentation)
- Natural Language Processing (BERT, transformers)
- Speech Recognition and Processing
- Recommendation Systems
- Time Series Analysis

For more information on optimizing models for OpenVINO, see the [OpenVINO documentation](https://docs.openvino.ai/).

### Performance Benefits

Using OpenVINO with OVMS can provide significant performance improvements:

- **CPU Optimization**: Up to 10x speedup on Intel CPUs compared to non-optimized frameworks
- **Memory Efficiency**: Reduced memory footprint with quantization
- **Latency Reduction**: Optimized for real-time applications
- **Energy Efficiency**: Lower power consumption, ideal for edge deployments

## Model Conversion Tips

To use models with OVMS, they typically need to be converted to OpenVINO's IR (Intermediate Representation) format:

```bash
# Example: Converting a PyTorch model to OpenVINO IR
python -m mo --input_model model.onnx --output_dir ./openvino_model

# Example: Converting a TensorFlow model to OpenVINO IR
python -m mo --input_model model.pb --output_dir ./openvino_model
```

For detailed conversion instructions, refer to the [OpenVINO Model Optimizer Guide](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).