# VLLM Unified API Backend Usage Guide

The VLLM Unified API backend provides comprehensive integration with VLLM servers, enabling advanced text generation capabilities, batched inference, streaming responses, model management, and specialized features like LoRA adapters and quantization settings. This unified backend offers enhanced functionality over the standard VLLM backend, including Docker container management for seamless deployment and operation.

## Key Features

- **Standard Text Generation**: Simple API for text completion and chat-based generation
- **Batch Processing**: Efficient handling of multiple prompts in a single request
- **Streaming Generation**: Real-time token-by-token streaming for responsive UIs
- **Model Management**: API for querying model information and statistics
- **LoRA Adapters**: Support for listing and loading LoRA adapters
- **Quantization Control**: API for configuring model quantization settings
- **Comprehensive Error Handling**: Graceful handling of rate limits and server errors
- **Resource Pooling**: Efficient connection management and request queueing
- **Docker Container Management**: Built-in capabilities to start, stop, monitor, and manage VLLM containers
- **Automatic Container Startup**: Automatic container instantiation when making API requests
- **Container Health Monitoring**: Health checks and automatic recovery for Docker containers
- **High-Performance Inference**: Leverages VLLM's continuous batching and PagedAttention for efficient inference
- **Cross-Platform Compatibility**: Works in both Node.js and browser environments (with limitations)

## Basic Usage

### Installation

```bash
npm install ipfs-accelerate
```

### Initialization

```typescript
import { VllmUnified } from 'ipfs-accelerate/api_backends/vllm_unified';

// Simple initialization with defaults
const vllm = new VllmUnified();

// With custom settings
const vllm = new VllmUnified(
  {}, // resources (optional)
  {
    vllm_api_url: 'http://your-vllm-server:8000',
    vllm_model: 'meta-llama/Llama-2-7b-chat-hf',
    timeout: 60000 // milliseconds
  }
);
```

### Text Generation

```typescript
// Basic generation with a string prompt
const result = await vllm.makeRequest(
  'http://your-vllm-server:8000',
  'Tell me a short story about dragons',
  'llama-7b'
);
console.log(result.text);

// Generation with parameters
const resultWithParams = await vllm.makeRequest(
  'http://your-vllm-server:8000',
  {
    prompt: 'Tell me a short story about dragons',
    max_tokens: 200,
    temperature: 0.7,
    top_p: 0.95
  },
  'llama-7b'
);
console.log(resultWithParams.text);
```

### Chat Completion

```typescript
import { ChatMessage } from 'ipfs-accelerate/api_backends/types';
import { ChatGenerationOptions } from 'ipfs-accelerate/api_backends/vllm_unified/types';

// Define a conversation with multiple messages
const messages: ChatMessage[] = [
  { role: 'system', content: 'You are a helpful AI assistant specialized in science.' },
  { role: 'user', content: 'Tell me about dragons.' }
];

const chatOptions: ChatGenerationOptions = {
  model: 'llama-7b-chat',
  temperature: 0.7
};

const response = await vllm.chat(messages, chatOptions);
console.log(response.content);
```

### Streaming Generation

```typescript
// Stream text generation
const stream = vllm.streamGeneration(
  'http://your-vllm-server:8000',
  'Tell me a story about dragons',
  'llama-7b',
  { temperature: 0.7 }
);

for await (const chunk of stream) {
  process.stdout.write(chunk.text);
}
```

### Batch Processing

```typescript
// Process multiple prompts in parallel
const prompts = [
  'What is the capital of France?',
  'What is the capital of Italy?',
  'What is the capital of Germany?'
];

const batchResults = await vllm.processBatch(
  'http://your-vllm-server:8000',
  prompts,
  'llama-7b',
  { temperature: 0.1 }
);

batchResults.forEach((result, index) => {
  console.log(`Question ${index + 1}: ${prompts[index]}`);
  console.log(`Answer: ${result}`);
});
```

## Advanced Features

### Model Information and Statistics

```typescript
// Get model information
const modelInfo = await vllm.getModelInfo(
  'http://your-vllm-server:8000',
  'llama-7b'
);
console.log(`Model max sequence length: ${modelInfo.max_model_len}`);
console.log(`Model dtype: ${modelInfo.dtype}`);
console.log(`GPU memory utilization: ${modelInfo.gpu_memory_utilization * 100}%`);

// Get model statistics
const stats = await vllm.getModelStatistics(
  'http://your-vllm-server:8000',
  'llama-7b'
);
console.log(`Total requests processed: ${stats.statistics.requests_processed}`);
console.log(`Average generation time: ${stats.statistics.avg_generation_time}s`);
console.log(`Throughput: ${stats.statistics.throughput} tokens/second`);
```

### LoRA Adapters Management

LoRA (Low-Rank Adaptation) adapters allow for efficient fine-tuning of large language models. The VLLM Unified backend provides APIs to list and load these adapters:

```typescript
// List available LoRA adapters
const adapters = await vllm.listLoraAdapters(
  'http://your-vllm-server:8000'
);
console.log(`Available adapters: ${adapters.map(a => a.name).join(', ')}`);

// Load a LoRA adapter
const loadResult = await vllm.loadLoraAdapter(
  'http://your-vllm-server:8000',
  {
    adapter_name: 'MyAdapter',
    adapter_path: '/path/to/adapter',
    base_model: 'llama-7b'
  }
);
console.log(`Adapter loaded: ${loadResult.success}`);

// Generate text using a loaded LoRA adapter
const loraResult = await vllm.makeRequest(
  'http://your-vllm-server:8000',
  {
    prompt: 'Tell me a story about dragons',
    max_tokens: 200,
    lora_adapter: 'MyAdapter' // Specify the LoRA adapter to use
  },
  'llama-7b'
);
console.log(`Text with LoRA: ${loraResult.text}`);
```

### Quantization Control

Quantization reduces the precision of model weights to decrease memory usage and increase inference speed. The VLLM Unified backend allows configuring quantization settings:

```typescript
// Configure quantization
const quantizationResult = await vllm.setQuantization(
  'http://your-vllm-server:8000',
  'llama-7b',
  {
    enabled: true,
    method: 'awq',
    bits: 4
  }
);
console.log(`Quantization configuration: ${JSON.stringify(quantizationResult.quantization)}`);

// Generate text with quantized model
const quantizedResult = await vllm.makeRequest(
  'http://your-vllm-server:8000',
  'Tell me about machine learning',
  'llama-7b'
);
console.log(`Text from quantized model: ${quantizedResult.text}`);
```

## Advanced API Usage

### Custom Endpoint Handlers

The VLLM Unified backend allows creating custom endpoint handlers for reuse:

```typescript
// Create a custom endpoint handler
const handler = vllm.createVllmEndpointHandler(
  'http://your-vllm-server:8000',
  'llama-7b'
);

// Use the handler
const result = await handler({ prompt: 'Hello, world!' });
console.log(result.text);

// Create a handler with parameters
const paramHandler = vllm.createVllmEndpointHandlerWithParams(
  'http://your-vllm-server:8000',
  'llama-7b',
  { temperature: 0.7, top_p: 0.95 }
);

// The parameters will be automatically applied
const paramResult = await paramHandler({ prompt: 'Hello, world!' });
console.log(paramResult.text);
```

### Endpoint Multiplexing

For load balancing and high availability, you can create multiple endpoints:

```typescript
// Create multiple endpoints
const endpoint1 = vllm.createEndpoint({
  api_key: 'key1',
  max_concurrent_requests: 5,
  max_retries: 3
});

const endpoint2 = vllm.createEndpoint({
  api_key: 'key2',
  max_concurrent_requests: 10,
  max_retries: 5
});

// Make requests with specific endpoints
const result1 = await vllm.makeRequestWithEndpoint(
  endpoint1,
  'Hello, world!',
  'llama-7b'
);

const result2 = await vllm.makeRequestWithEndpoint(
  endpoint2,
  'How are you?',
  'llama-7b'
);

// Get endpoint statistics
const stats1 = vllm.getStats(endpoint1);
console.log(`Endpoint 1 requests: ${stats1.requests}`);
console.log(`Endpoint 1 success: ${stats1.success}`);
console.log(`Endpoint 1 errors: ${stats1.errors}`);

// Load balancing across multiple endpoints
const loadBalancer = vllm.createLoadBalancer([endpoint1, endpoint2], {
  strategy: 'round-robin', // or 'least-loaded'
  healthCheck: true
});

// Make a request through the load balancer
const balancedResult = await vllm.makeRequestWithLoadBalancer(
  loadBalancer,
  'Test prompt',
  'llama-7b'
);
```

### Circuit Breaker Pattern

The circuit breaker pattern prevents cascading failures by stopping requests when errors occur frequently:

```typescript
// Create an endpoint with circuit breaker configuration
const endpointWithCircuitBreaker = vllm.createEndpoint({
  api_key: 'my-key',
  max_concurrent_requests: 10,
  circuit_breaker: {
    failure_threshold: 5,        // Trip after 5 consecutive failures
    reset_timeout: 30000,        // 30 seconds timeout before resetting
    half_open_successful_calls: 2 // 2 successful calls to fully close
  }
});

// Circuit breaker events
vllm.on('circuit_open', (endpoint) => {
  console.log(`Circuit breaker opened for endpoint: ${endpoint.id}`);
});

vllm.on('circuit_half_open', (endpoint) => {
  console.log(`Circuit breaker half-opened for endpoint: ${endpoint.id}`);
});

vllm.on('circuit_closed', (endpoint) => {
  console.log(`Circuit breaker closed for endpoint: ${endpoint.id}`);
});
```

## Error Handling

The VLLM Unified API backend includes comprehensive error handling:

```typescript
try {
  const result = await vllm.makeRequest(
    'http://your-vllm-server:8000',
    'Tell me a story',
    'non-existent-model'
  );
} catch (error) {
  if (error.statusCode === 404) {
    console.error('Model not found');
  } else if (error.statusCode === 429) {
    console.error(`Rate limited. Retry after ${error.retryAfter} seconds`);
  } else if (error.isTransientError) {
    console.error('Temporary server error, retry later');
  } else {
    console.error(`Error: ${error.message}`);
  }
}
```

### Robust Error Handling Pattern

For production applications, implement a robust error handling pattern with exponential backoff:

```typescript
// Robust error handling with exponential backoff
const robustGenerateText = async (prompt, options = {}, retries = 3) => {
  try {
    return await vllm.makeRequest(
      'http://your-vllm-server:8000',
      prompt,
      'llama-7b',
      options
    );
  } catch (error) {
    if (
      (error.statusCode === 503 || error.statusCode === 429 || error.isTransientError) && 
      retries > 0
    ) {
      // Exponential backoff
      const delay = 1000 * Math.pow(2, 3 - retries);
      console.log(`Retrying after ${delay}ms (${retries} retries left)...`);
      await new Promise(resolve => setTimeout(resolve, delay));
      return robustGenerateText(prompt, options, retries - 1);
    }
    throw error;
  }
};

// Usage
try {
  const result = await robustGenerateText(
    'Tell me about quantum physics',
    { temperature: 0.7 }
  );
  console.log(result.text);
} catch (error) {
  console.error('Failed after multiple retries:', error);
}
```

## Docker Container Management

The VLLM Unified API backend includes built-in Docker container management capabilities, allowing you to seamlessly deploy, run, and manage VLLM containers directly from your application.

### Initializing with Container Support

```typescript
import { VllmUnified } from 'ipfs-accelerate/api_backends/vllm_unified';

// Initialize with container management enabled
const vllm = new VllmUnified({}, {
  vllm_api_url: 'http://localhost:8000',
  vllm_model: 'meta-llama/Llama-2-7b-chat-hf',
  vllm_container_enabled: true,
  vllm_container_image: 'vllm/vllm-openai:latest',
  vllm_container_gpu: true,
  vllm_tensor_parallel_size: 2,
  vllm_gpu_memory_utilization: 0.8
});
```

### Container Management Operations

```typescript
// Start the container explicitly
const started = await vllm.startContainer();
console.log(`Container started: ${started}`);

// Get current container status
const status = vllm.getContainerStatus();
console.log(`Container status: ${status}`);

// Get container logs
const logs = vllm.getContainerLogs();
logs.forEach(log => console.log(log));

// Get container configuration
const config = vllm.getContainerConfig();
console.log(`Container image: ${config?.image}`);
console.log(`GPU enabled: ${config?.gpu}`);
console.log(`Tensor parallel size: ${config?.tensor_parallel_size}`);

// Update container configuration
vllm.setContainerConfig({
  gpu_memory_utilization: 0.9,
  tensor_parallel_size: 4,
  max_model_len: 4096,
  quantization: 'awq'
});

// Restart the container
const restarted = await vllm.restartContainer();
console.log(`Container restarted: ${restarted}`);

// Stop the container
const stopped = await vllm.stopContainer();
console.log(`Container stopped: ${stopped}`);
```

### Container Metrics and Monitoring

```typescript
// Get container metrics
const metrics = await vllm.getContainerMetrics();
console.log(`CPU usage: ${metrics.cpu_usage}%`);
console.log(`Memory usage: ${metrics.memory_usage} MB`);
console.log(`GPU memory usage: ${metrics.gpu_memory_usage} MB`);
console.log(`Requests processed: ${metrics.requests_processed}`);
console.log(`Tokens generated: ${metrics.tokens_generated}`);
console.log(`Average latency: ${metrics.average_latency} ms`);
console.log(`Throughput: ${metrics.throughput} tokens/sec`);

// Set up container monitoring
vllm.startContainerMonitoring({
  interval: 5000, // Check every 5 seconds
  metrics: true,  // Collect metrics
  logs: true,     // Collect logs
  autoRestart: true // Automatically restart if unhealthy
});

// Container events
vllm.on('container_started', (containerId) => {
  console.log(`Container started: ${containerId}`);
});

vllm.on('container_stopped', (containerId) => {
  console.log(`Container stopped: ${containerId}`);
});

vllm.on('container_error', (error) => {
  console.error(`Container error: ${error.message}`);
});

vllm.on('container_unhealthy', (metrics) => {
  console.warn(`Container unhealthy: CPU ${metrics.cpu_usage}%, Memory ${metrics.memory_usage}MB`);
});

// Stop monitoring
vllm.stopContainerMonitoring();
```

### Automatic Container Management

The VLLM Unified backend automatically manages containers when you make API requests:

```typescript
// Container will start automatically when you make a request
const result = await vllm.makeRequest(
  'http://localhost:8000',
  'Tell me a story about dragons',
  'meta-llama/Llama-2-7b-chat-hf'
);

// Similarly, testing an endpoint will automatically start the container if needed
const isEndpointWorking = await vllm.testEndpoint();
```

### Loading Models into the Container

You can load models into the container for faster inference:

```typescript
// Load models into the container
const modelsLoaded = await vllm.loadModels([
  '/path/to/model1',
  '/path/to/model2'
]);
console.log(`Models loaded: ${modelsLoaded}`);

// Load models with specific options
const modelsLoadedWithOptions = await vllm.loadModels([
  {
    path: '/path/to/model1',
    quantization: 'awq',
    max_model_len: 4096
  },
  {
    path: '/path/to/model2',
    quantization: 'gptq',
    max_model_len: 8192
  }
]);
```

### Container Health Validation

Validate container capabilities to ensure everything is working correctly:

```typescript
// Validate container capabilities
const capabilities = await vllm.validateContainerCapabilities();
console.log(`Docker available: ${capabilities.docker_available}`);
console.log(`GPU support: ${capabilities.gpu_support}`);
console.log(`API accessible: ${capabilities.api_accessible}`);
console.log(`Model loaded: ${capabilities.model_loaded}`);

// Full system check
const systemCheck = await vllm.runSystemCheck();
console.log(`System check passed: ${systemCheck.passed}`);
console.log(`Issues found: ${systemCheck.issues.length}`);
systemCheck.issues.forEach(issue => console.log(`- ${issue.severity}: ${issue.message}`));

// Performance benchmark
const benchmark = await vllm.runContainerBenchmark({
  model: 'llama-7b',
  prompt_length: 100,
  output_length: 100,
  batch_size: 1,
  iterations: 5
});
console.log(`Throughput: ${benchmark.throughput} tokens/sec`);
console.log(`Latency: ${benchmark.latency} ms`);
```

### Enabling and Disabling Container Management

You can enable or disable container management at runtime:

```typescript
// Enable container mode
vllm.enableContainerMode({
  gpu: true,
  tensor_parallel_size: 4
});

// Disable container mode (will stop any running containers)
await vllm.disableContainerMode();
```

## Environment Variables and Configuration

The VLLM Unified API backend can be configured using either metadata during initialization or through environment variables:

### Environment Variables

#### General API Configuration
- `VLLM_API_URL`: Default API URL (e.g., `http://localhost:8000`)
- `VLLM_MODEL`: Default model to use (e.g., `meta-llama/Llama-2-7b-chat-hf`)
- `VLLM_API_KEY`: API key for authentication (if required)
- `VLLM_TIMEOUT`: Request timeout in milliseconds (default: 30000)

#### Container Management
- `VLLM_CONTAINER_ENABLED`: Enable container management (true/false)
- `VLLM_CONTAINER_IMAGE`: Docker image to use (e.g., `vllm/vllm-openai:latest`)
- `VLLM_CONTAINER_GPU`: Enable GPU support for container (true/false)
- `VLLM_CONFIG_PATH`: Path to store configuration files (default: `~/.vllm`)
- `VLLM_MODELS_PATH`: Path to store models (default: `~/.vllm/models`)
- `VLLM_API_PORT`: Port to expose for the API (default: 8000)
- `VLLM_TENSOR_PARALLEL_SIZE`: Number of GPUs for tensor parallelism (default: 1)
- `VLLM_GPU_MEMORY_UTILIZATION`: GPU memory utilization (0.0-1.0, default: 0.9)
- `VLLM_MAX_MODEL_LEN`: Maximum model context length (0 = use default)
- `VLLM_QUANTIZATION`: Quantization method (e.g., `awq`, `gptq`, `squeezellm`)
- `VLLM_TRUST_REMOTE_CODE`: Whether to trust remote code when loading models (true/false)

### Metadata Configuration Options

When instantiating the API backend, you can provide these settings via metadata:

```typescript
const vllm = new VllmUnified({}, {
  // General API Configuration
  vllm_api_url: 'http://your-vllm-server:8000',
  vllm_model: 'meta-llama/Llama-2-7b-chat-hf',
  vllmApiKey: 'your-api-key',  // Alternative camelCase syntax
  timeout: 60000,  // 60 seconds
  maxRetries: 3,
  maxConcurrentRequests: 10,
  queueSize: 100,
  
  // Container Management Configuration
  vllm_container_enabled: true,
  vllm_container_image: 'vllm/vllm-openai:latest',
  vllm_container_gpu: true,
  vllm_config_path: '/path/to/config',
  vllm_models_path: '/path/to/models',
  vllm_api_port: 8000,
  vllm_tensor_parallel_size: 2,
  vllm_gpu_memory_utilization: 0.8,
  vllm_max_model_len: 4096,
  vllm_quantization: 'awq',
  vllm_trust_remote_code: false,
  vllm_custom_args: '--additional-args'
});
```

Priority order:
1. Options provided directly in method calls
2. Metadata provided during initialization
3. Environment variables
4. Default values

## Compatibility and Performance

The VLLM Unified API backend is compatible with most transformer-based models and provides high-performance inference through the VLLM server.

### Compatible Model Families

- **Llama Family**: Llama, Llama 2, Llama 3, CodeLlama, etc.
- **Mistral Family**: Mistral, Mixtral models
- **Falcon Family**: Falcon models
- **MPT Models**: Mosaic's MPT models
- **OPT Models**: Meta's OPT series
- **BLOOM Models**: BigScience BLOOM models
- **Other Models**:
  - StableLM models
  - Pythia models
  - Qwen models
  - Claude-compatible models
  - Many other transformer-based models

### Performance and Features

VLLM provides significant performance advantages over standard inference engines:

- **Continuous Batching**: Efficiently handles multiple requests
- **PagedAttention**: Memory-efficient KV cache implementation
- **Tensor Parallelism**: Distributed inference across multiple GPUs
- **Quantization**: Support for various quantization methods (AWQ, SqueezeLLM, etc.)
- **Streaming**: Efficient token-by-token streaming for responsive UI

### Integration with Hardware Backends

The VLLM Unified backend can be easily integrated with hardware backends:

```typescript
import { VllmUnified } from 'ipfs-accelerate/api_backends/vllm_unified';
import { HardwareAbstractionLayer } from 'ipfs-accelerate/hardware';

// Create VLLM backend
const vllm = new VllmUnified();

// Create hardware abstraction layer
const hal = new HardwareAbstractionLayer();

// Register VLLM as a backend for large language models
hal.registerBackend('llm', vllm);

// Use through the hardware abstraction layer
const result = await hal.runInference('llm', {
  input: 'Hello, world!',
  model: 'llama-7b'
});
```

### Performance Benchmarking

The VLLM Unified backend includes tools for benchmarking performance:

```typescript
// Simple benchmark
const simpleBenchmark = await vllm.benchmark({
  prompt: 'What is artificial intelligence?',
  model: 'llama-7b',
  iterations: 5,
  warmup: 2
});
console.log(`Average latency: ${simpleBenchmark.avg_latency} ms`);
console.log(`Throughput: ${simpleBenchmark.throughput} tokens/sec`);

// Comprehensive benchmark with different configurations
const comprehensiveBenchmark = await vllm.runComprehensiveBenchmark({
  model: 'llama-7b',
  batch_sizes: [1, 2, 4, 8],
  prompt_lengths: [128, 256, 512],
  output_lengths: [128, 256],
  quantization: ['none', 'awq'],
  iterations: 3
});

// Save benchmark results
await vllm.saveBenchmarkResults(comprehensiveBenchmark, 'benchmark_results.json');

// Load and analyze benchmark results
const results = await vllm.loadBenchmarkResults('benchmark_results.json');
const analysis = vllm.analyzeBenchmarkResults(results);
console.log(`Optimal configuration: ${JSON.stringify(analysis.optimal_config)}`);
console.log(`Throughput improvement: ${analysis.throughput_improvement}x`);
```

For a complete list of compatible models and performance features, refer to the [VLLM project documentation](https://github.com/vllm-project/vllm).

## Advanced Use Cases

### Production-Ready LLM System

This example demonstrates a production-ready LLM system with:
- Automatic container management
- High availability and failover
- Rate limiting and request queuing
- Circuit breaker pattern
- Exponential backoff retries
- Detailed monitoring and metrics

```typescript
import { VllmUnified } from 'ipfs-accelerate/api_backends/vllm_unified';
import { EventEmitter } from 'events';

class ProductionLlmSystem extends EventEmitter {
  private vllm: VllmUnified;
  private active: boolean = false;
  private metrics: any = {
    requests: 0,
    success: 0,
    errors: 0,
    latency: []
  };

  constructor(config = {}) {
    super();
    
    // Initialize with container support and circuit breaker
    this.vllm = new VllmUnified({}, {
      vllm_container_enabled: true,
      vllm_container_gpu: true,
      vllm_api_url: 'http://localhost:8000',
      vllm_model: 'meta-llama/Llama-2-7b-chat-hf',
      maxRetries: 3,
      timeout: 60000,
      circuitBreaker: {
        failureThreshold: 5,
        resetTimeout: 30000
      }
    });
    
    // Set up event listeners
    this.vllm.on('container_started', this.handleContainerStarted.bind(this));
    this.vllm.on('container_error', this.handleContainerError.bind(this));
    this.vllm.on('circuit_open', this.handleCircuitOpen.bind(this));
    this.vllm.on('circuit_closed', this.handleCircuitClosed.bind(this));
  }
  
  // Start the system
  async start() {
    this.active = true;
    await this.vllm.startContainer();
    this.vllm.startContainerMonitoring({ interval: 5000 });
    
    // Validate container capabilities
    const capabilities = await this.vllm.validateContainerCapabilities();
    if (!capabilities.api_accessible) {
      throw new Error('API not accessible after container start');
    }
    
    this.emit('system_ready');
  }
  
  // Stop the system
  async stop() {
    this.active = false;
    this.vllm.stopContainerMonitoring();
    await this.vllm.stopContainer();
    this.emit('system_stopped');
  }
  
  // Generate text with robust error handling
  async generateText(prompt, options = {}) {
    if (!this.active) {
      throw new Error('System not active');
    }
    
    const startTime = Date.now();
    this.metrics.requests++;
    
    try {
      const result = await this.vllm.makeRequest(
        'http://localhost:8000',
        prompt,
        options.model || 'meta-llama/Llama-2-7b-chat-hf',
        options
      );
      
      // Record metrics
      this.metrics.success++;
      this.metrics.latency.push(Date.now() - startTime);
      
      return result;
    } catch (error) {
      this.metrics.errors++;
      this.emit('request_error', { prompt, error });
      throw error;
    }
  }
  
  // Chat completion with robust error handling
  async chat(messages, options = {}) {
    if (!this.active) {
      throw new Error('System not active');
    }
    
    const startTime = Date.now();
    this.metrics.requests++;
    
    try {
      const result = await this.vllm.chat(messages, options);
      
      // Record metrics
      this.metrics.success++;
      this.metrics.latency.push(Date.now() - startTime);
      
      return result;
    } catch (error) {
      this.metrics.errors++;
      this.emit('request_error', { messages, error });
      throw error;
    }
  }
  
  // Get system metrics
  getMetrics() {
    const avgLatency = this.metrics.latency.length > 0
      ? this.metrics.latency.reduce((a, b) => a + b, 0) / this.metrics.latency.length
      : 0;
    
    return {
      requests: this.metrics.requests,
      success: this.metrics.success,
      errors: this.metrics.errors,
      error_rate: this.metrics.requests > 0 
        ? (this.metrics.errors / this.metrics.requests) * 100 
        : 0,
      avg_latency: avgLatency,
      container_status: this.vllm.getContainerStatus(),
      active: this.active
    };
  }
  
  // Event handlers
  private handleContainerStarted(containerId) {
    this.emit('container_started', containerId);
  }
  
  private handleContainerError(error) {
    this.emit('container_error', error);
    if (this.active) {
      // Attempt recovery
      setTimeout(() => {
        this.vllm.restartContainer().catch(e => {
          this.emit('recovery_failed', e);
        });
      }, 5000);
    }
  }
  
  private handleCircuitOpen() {
    this.emit('circuit_open');
  }
  
  private handleCircuitClosed() {
    this.emit('circuit_closed');
  }
}

// Usage example
async function main() {
  const llmSystem = new ProductionLlmSystem();
  
  llmSystem.on('system_ready', () => {
    console.log('LLM system is ready');
  });
  
  llmSystem.on('request_error', ({ error }) => {
    console.error('Request error:', error.message);
  });
  
  try {
    await llmSystem.start();
    
    const result = await llmSystem.generateText(
      'Explain the concept of quantum computing in simple terms.',
      { temperature: 0.7, max_tokens: 200 }
    );
    console.log('Generated text:', result.text);
    
    const chatResult = await llmSystem.chat(
      [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What are the main challenges in AI safety?' }
      ],
      { temperature: 0.7, max_tokens: 300 }
    );
    console.log('Chat response:', chatResult.content);
    
    // Get metrics
    const metrics = llmSystem.getMetrics();
    console.log('System metrics:', metrics);
    
    // Stop the system
    await llmSystem.stop();
  } catch (error) {
    console.error('System error:', error);
    await llmSystem.stop();
  }
}

main().catch(console.error);
```

## Comparison with Python Implementation

The TypeScript VLLM Unified backend provides all the functionality of the Python implementation with additional features:

| Feature | Python Implementation | TypeScript Implementation |
|---------|----------------------|---------------------------|
| Basic Inference | ✅ | ✅ |
| Chat Completion | ✅ | ✅ |
| Batch Processing | ✅ | ✅ |
| Streaming | ✅ | ✅ |
| Queue Management | ✅ | ✅ Enhanced |
| Circuit Breaker | ✅ | ✅ Enhanced |
| Model Information | ✅ | ✅ |
| LoRA Support | ✅ | ✅ |
| Quantization | ✅ | ✅ |
| Performance Tracking | ✅ | ✅ Enhanced |
| Error Handling | ✅ | ✅ Enhanced |
| TypeScript Types | ❌ | ✅ |
| Hardware Integration | ✅ | ✅ Enhanced |
| Docker Container Management | ✅ | ✅ Enhanced |
| Automatic Container Startup | ❌ | ✅ |
| Container Health Monitoring | ❌ | ✅ |
| Production-Ready Examples | ❌ | ✅ |
| Load Balancing | ❌ | ✅ |
| Resource Pooling | ❌ | ✅ |
| Comprehensive Benchmarking | ❌ | ✅ |

The TypeScript implementation enhances the original Python features and adds new capabilities for enterprise deployment scenarios.

## Future Development and Contributing

The VLLM Unified backend is actively maintained as part of the IPFS Accelerate TypeScript SDK. Here are some areas for future enhancement:

1. **Advanced Streaming**: Improved WebSocket and HTTP/2 support for more efficient streaming
2. **Monitoring Dashboard**: Integration with dashboard visualization tools 
3. **Kubernetes Integration**: Native support for Kubernetes deployments
4. **Edge Deployment**: Optimized configurations for edge devices
5. **Advanced Auto-Scaling**: Automatic scaling based on queue length and latency metrics
6. **Multi-Model Serving**: Improved support for serving multiple models from a single container
7. **Cross-Platform Compatibility**: Enhanced support for browser environments 

To contribute to the VLLM Unified backend, please:

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Submit a pull request

For more information, see the project's contribution guidelines.

## Last Updated

This documentation was last updated on March 19, 2025.