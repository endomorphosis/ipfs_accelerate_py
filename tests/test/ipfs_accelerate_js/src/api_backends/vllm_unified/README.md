# VLLM Unified API Backend

This module provides a TypeScript implementation of the VLLM unified API backend with Docker container management capabilities, offering enhanced functionality for deploying and managing VLLM servers.

## Overview

The VLLM Unified backend extends the base VLLM backend with comprehensive Docker container management for simplified deployment and operation:

- **Container Management**: Automated Docker container lifecycle for VLLM servers
- **GPU Support**: NVIDIA GPU integration for accelerated inference
- **Configuration Management**: Extensive container and model configuration options
- **Health Monitoring**: Automatic health checks and container recovery
- **Error Handling**: Comprehensive error management with circuit breaking
- **Metrics Collection**: Performance statistics and resource utilization tracking

Plus all the standard VLLM features:
- Standard and streaming text generation
- Batch processing with metrics
- LoRA adapter support
- Quantization configuration
- Resource pooling

## Installation

```bash
npm install ipfs-accelerate-js
```

## Basic Usage

```typescript
import { VllmUnified } from 'ipfs-accelerate-js/api_backends/vllm_unified';

// Create a VLLM instance
const vllm = new VllmUnified({}, {
  vllm_api_url: 'http://localhost:8000',
  vllm_model: 'meta-llama/Llama-2-7b-chat-hf'
});

// Basic text generation
const result = await vllm.makeRequest(
  'Tell me a story about dragons',
  'meta-llama/Llama-2-7b-chat-hf'
);

// Chat completion
const chatResult = await vllm.chat(
  [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Tell me about dragons.' }
  ],
  { model: 'meta-llama/Llama-2-7b-chat-hf' }
);

// Streaming text generation
for await (const chunk of vllm.streamGeneration(
  'Tell me about quantum computing',
  'meta-llama/Llama-2-7b-chat-hf'
)) {
  console.log(chunk); // Process each chunk as it arrives
}
```

## Docker Container Management

The VllmUnified backend can automatically manage a Docker container running VLLM:

```typescript
import { VllmUnified } from 'ipfs-accelerate-js/api_backends/vllm_unified';

// Create a VLLM instance with container management
const vllm = new VllmUnified({}, {
  // Enable container mode
  vllm_container_enabled: true,
  
  // Container configuration
  vllm_container_image: 'vllm/vllm-openai:latest',
  vllm_container_gpu: true,
  vllm_model: 'meta-llama/Llama-2-7b-chat-hf',
  vllm_tensor_parallel_size: 2,
  vllm_gpu_memory_utilization: 0.8,
  vllm_quantization: 'awq',
  vllm_models_path: '/path/to/models',
  vllm_config_path: '/path/to/config',
  vllm_api_port: 8000
});

// Manually start the container if not auto-started
await vllm.startContainer();

// Generate text (automatically starts container if not running)
const result = await vllm.chat(
  [{ role: 'user', content: 'Tell me about quantum computing.' }]
);

// Get container status
const status = vllm.getContainerStatus(); // 'running', 'stopped', 'error', etc.

// Get container logs
const logs = vllm.getContainerLogs();

// Stop the container when done
await vllm.stopContainer();
```

## Advanced Container Configuration

```typescript
// Update container configuration
vllm.setContainerConfig({
  gpu_memory_utilization: 0.9,
  tensor_parallel_size: 4,
  max_model_len: 4096,
  quantization: 'awq'
});

// Restart with new configuration
await vllm.restartContainer();

// Validate container capabilities
const capabilities = await vllm.validateContainerCapabilities();
console.log(capabilities.gpu_support, capabilities.api_accessible);

// Load models into container
await vllm.loadModels([
  '/path/to/model1',
  '/path/to/model2'
]);
```

## Batch Processing with Metrics

```typescript
// Process a batch of prompts with detailed metrics
const [results, metrics] = await vllm.processBatchWithMetrics(
  ['Question 1', 'Question 2', 'Question 3'],
  'meta-llama/Llama-2-7b-chat-hf'
);

console.log(`Processed ${metrics.batch_size} prompts`);
console.log(`Total time: ${metrics.total_time_ms}ms`);
console.log(`Average time per item: ${metrics.average_time_per_item_ms}ms`);
console.log(`Tokens used: ${metrics.usage.total_tokens}`);
```

## Advanced Model Management

The VLLM Unified backend provides comprehensive model management capabilities:

```typescript
// Get model information
const modelInfo = await vllm.getModelInfo('meta-llama/Llama-2-7b-chat-hf');
console.log(`Model context length: ${modelInfo.max_model_len}`);
console.log(`Number of GPUs: ${modelInfo.num_gpu}`);

// Get model statistics
const stats = await vllm.getModelStatistics('meta-llama/Llama-2-7b-chat-hf');
console.log(`Requests processed: ${stats.statistics.requests_processed}`);
console.log(`Throughput: ${stats.statistics.throughput}`);

// Manage LoRA adapters
const adapters = await vllm.listLoraAdapters();
console.log(`Available LoRA adapters: ${adapters.length}`);

// Load a new LoRA adapter
const loaded = await vllm.loadLoraAdapter({
  adapter_name: 'my-adapter',
  adapter_path: '/path/to/adapter',
  base_model: 'meta-llama/Llama-2-7b-chat-hf'
});

// Configure model quantization
const quantized = await vllm.setQuantization(
  'meta-llama/Llama-2-7b-chat-hf',
  {
    enabled: true,
    method: 'awq',
    bits: 4
  }
);
```

## Configuration Options

The VllmUnified backend accepts the following configuration options:

| Option | Type | Description |
|--------|------|-------------|
| `vllm_api_url` | string | URL of the VLLM API server |
| `vllm_model` | string | Default model to use |
| `vllm_api_key` | string | API key for authentication |
| `vllm_container_enabled` | boolean | Enable container management |
| `vllm_container_image` | string | Docker image for VLLM container |
| `vllm_container_gpu` | boolean | Enable GPU support in container |
| `vllm_tensor_parallel_size` | number | Number of GPUs for tensor parallelism |
| `vllm_gpu_memory_utilization` | number | GPU memory utilization (0.0-1.0) |
| `vllm_max_model_len` | number | Maximum model context length |
| `vllm_quantization` | string | Quantization method (e.g., 'awq', 'gptq') |
| `vllm_models_path` | string | Path to model files on host |
| `vllm_config_path` | string | Path to configuration directory on host |
| `vllm_api_port` | number | Port for API server |
| `vllm_trust_remote_code` | boolean | Trust remote code when loading models |
| `vllm_custom_args` | string | Custom arguments for container |
| `vllm_extra_mounts` | string[] | Additional volume mounts for container |

## Files

- `types.ts`: TypeScript type definitions for the VLLM API and container management
- `vllm_unified.ts`: Main implementation of the VLLM Unified backend with container management
- `index.ts`: Module exports
- `README.md`: This documentation file

## Error Handling

The VLLM Unified backend provides comprehensive error handling with automatic retry and circuit breaking:

```typescript
try {
  const result = await vllm.chat(
    [{ role: 'user', content: 'Tell me about quantum computing.' }]
  );
} catch (error) {
  if (error.code === 'container_error') {
    console.error('Container error:', error.message);
  } else if (error.code === 'vllm_error') {
    console.error('VLLM API error:', error.message);
  } else if (error.code === 'rate_limit_error') {
    console.error('Rate limit exceeded, retry after:', error.retryAfter);
  } else if (error.code === 'timeout_error') {
    console.error('Request timed out');
  } else {
    console.error('Unknown error:', error);
  }
}
```

## Container Status Monitoring

The container status can be one of the following:
- `'stopped'`: Container is not running
- `'starting'`: Container is starting up
- `'running'`: Container is running and healthy
- `'stopping'`: Container is shutting down
- `'error'`: Container encountered an error

```typescript
// Get container status
const status = vllm.getContainerStatus();

// Check logs
const logs = vllm.getContainerLogs();
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.