# API Configuration Reference

This document provides a comprehensive reference for all configurable settings in the IPFS Accelerate Python Framework's API system. Use this guide to configure the advanced features according to your specific requirements.

## Table of Contents

1. [Global Configuration](#global-configuration)
2. [Queue System Configuration](#queue-system-configuration)
3. [Circuit Breaker Configuration](#circuit-breaker-configuration)
4. [Monitoring System Configuration](#monitoring-system-configuration)
5. [Request Batching Configuration](#request-batching-configuration)
6. [API Key Multiplexing Configuration](#api-key-multiplexing-configuration)
7. [API-Specific Configuration](#api-specific-configuration)
8. [Environment Variables](#environment-variables)
9. [Configuration Profiles](#configuration-profiles)

## Global Configuration

Global configuration parameters affect the overall behavior of the framework.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | bool | `False` | Enable verbose logging |
| `debug` | bool | `False` | Enable debug mode with additional logging |
| `log_level` | str | `"INFO"` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `log_file` | str | `None` | Path to log file (None for stdout) |
| `timeout` | float | `60.0` | Global timeout for requests in seconds |
| `default_api` | str | `"openai_api"` | Default API when none is specified |
| `use_local_fallback` | bool | `True` | Fall back to local models when API is unavailable |
| `environment` | str | `"production"` | Environment setting (production, staging, development) |

## Queue System Configuration

The queue system manages concurrent requests with priority-based processing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrent_requests` | int | `5` | Maximum number of concurrent requests |
| `queue_size` | int | `100` | Maximum queue size before rejecting new requests |
| `queue_processing` | bool | `True` | Enable/disable queue processing |
| `priority_levels` | int | `3` | Number of priority levels (typically 3: High, Normal, Low) |
| `queue_overflow_policy` | str | `"reject"` | Policy when queue is full ("reject", "replace_lowest") |
| `queue_check_interval` | float | `0.01` | Interval in seconds to check queue for new requests |
| `auto_scale_concurrency` | bool | `False` | Dynamically adjust max_concurrent_requests based on performance |
| `min_concurrent_requests` | int | `1` | Minimum concurrent requests when auto-scaling |
| `max_concurrent_factor` | float | `2.0` | Maximum auto-scale factor for concurrent requests |

## Circuit Breaker Configuration

The circuit breaker pattern prevents cascading failures and enables self-healing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `circuit_enabled` | bool | `True` | Enable/disable circuit breaker |
| `failure_threshold` | int | `5` | Number of failures before opening circuit |
| `reset_timeout` | float | `30.0` | Seconds to wait before attempting to half-open circuit |
| `circuit_state` | str | `"CLOSED"` | Initial circuit state (CLOSED, OPEN, HALF-OPEN) |
| `half_open_max_requests` | int | `1` | Maximum requests to allow when half-open |
| `failure_types` | list | `[ConnectionError, TimeoutError]` | Exception types to count as failures |
| `circuit_fallback_value` | any | `None` | Value to return when circuit is open (None to raise exception) |

## Monitoring System Configuration

The monitoring system tracks performance metrics, errors, and system health.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics_enabled` | bool | `True` | Enable/disable metrics collection |
| `detailed_latency` | bool | `False` | Store detailed latency for each request |
| `error_tracking` | bool | `True` | Track error types and frequencies |
| `latency_percentiles` | list | `[50, 90, 95, 99]` | Percentiles to calculate for latency reporting |
| `metrics_window_size` | int | `1000` | Number of requests to keep in metrics window |
| `metrics_reset_interval` | float | `3600.0` | Seconds before resetting metrics (0 for never) |
| `request_tracing` | bool | `False` | Enable request tracing with unique IDs |
| `tracing_headers` | bool | `False` | Include trace IDs in request headers |
| `record_request_content` | bool | `False` | Record request and response content in metrics |

## Request Batching Configuration

Request batching combines compatible requests for better throughput.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_batching` | bool | `True` | Enable/disable request batching |
| `max_batch_size` | int | `8` | Maximum number of requests in a batch |
| `batch_timeout` | float | `0.1` | Seconds to wait for batch completion |
| `min_batch_size` | int | `2` | Minimum batch size to process immediately |
| `batch_models` | list | `["all"]` | List of models that support batching ("all" for all models) |
| `preserve_order` | bool | `True` | Preserve request order in batch results |
| `max_batch_tokens` | int | `4096` | Maximum tokens in a batch for text generation |
| `token_padding` | int | `8` | Token padding between requests in a batch |

## API Key Multiplexing Configuration

API key multiplexing manages multiple keys with intelligent routing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `multiplexing_enabled` | bool | `False` | Enable/disable API key multiplexing |
| `default_strategy` | str | `"round-robin"` | Default selection strategy |
| `fallback_enabled` | bool | `True` | Enable fallback to alternative keys on failure |
| `max_key_attempts` | int | `3` | Maximum number of keys to try before failing |
| `key_switch_threshold` | float | `0.8` | Queue utilization threshold to switch keys |
| `key_rotation_interval` | float | `0.0` | Seconds between forced key rotation (0 for none) |
| `track_key_performance` | bool | `True` | Track and adjust based on key performance |
| `performance_window` | int | `100` | Number of requests in performance tracking window |

## API-Specific Configuration

These parameters are specific to individual API backends.

### OpenAI API

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `openai_api_version` | str | `None` | OpenAI API version to use |
| `openai_organization` | str | `None` | OpenAI organization ID |
| `openai_azure_endpoint` | str | `None` | Azure OpenAI endpoint URL |
| `openai_azure_deployment` | str | `None` | Azure OpenAI deployment name |
| `openai_model_map` | dict | `{}` | Model name mapping for OpenAI |

### Claude API

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `claude_api_version` | str | `None` | Claude API version |
| `claude_model_map` | dict | `{}` | Model name mapping for Claude |

### Groq API

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `groq_model_map` | dict | `{}` | Model name mapping for Groq |

### Ollama API

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ollama_host` | str | `"http://localhost:11434"` | Ollama host URL |
| `ollama_timeout` | float | `120.0` | Ollama request timeout |

### HuggingFace TGI and TEI

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hf_endpoint` | str | `None` | HuggingFace endpoint URL |
| `hf_model_map` | dict | `{}` | Model name mapping for HuggingFace |
| `hf_token_padding` | int | `8` | Token padding for batch processing |

## Environment Variables

These environment variables can be used to configure the API system.

| Environment Variable | Description |
|----------------------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_API_KEY_1`, `OPENAI_API_KEY_2`... | Multiple OpenAI API keys for multiplexing |
| `ANTHROPIC_API_KEY` | Claude API key |
| `ANTHROPIC_API_KEY_1`, `ANTHROPIC_API_KEY_2`... | Multiple Claude API keys for multiplexing |
| `GROQ_API_KEY` | Groq API key |
| `GROQ_API_KEY_1`, `GROQ_API_KEY_2`... | Multiple Groq API keys for multiplexing |
| `HF_API_TOKEN` | HuggingFace API token |
| `GOOGLE_API_KEY` | Google Gemini API key |
| `IPFS_ACCELERATE_LOG_LEVEL` | Override default log level |
| `IPFS_ACCELERATE_MAX_CONCURRENT` | Override default max_concurrent_requests |
| `IPFS_ACCELERATE_QUEUE_SIZE` | Override default queue_size |
| `IPFS_ACCELERATE_TIMEOUT` | Override default timeout |
| `IPFS_ACCELERATE_ENVIRONMENT` | Set environment (production, staging, development) |

## Configuration Profiles

Pre-defined configuration profiles for common scenarios.

### High-Performance Profile

Optimized for maximum throughput in stable environments.

```python
high_performance_config = {
    "max_concurrent_requests": 50,
    "queue_size": 1000,
    "queue_check_interval": 0.005,
    "auto_scale_concurrency": True,
    "max_concurrent_factor": 5.0,
    
    "circuit_enabled": True,
    "failure_threshold": 10,
    "reset_timeout": 15.0,
    
    "enable_batching": True,
    "max_batch_size": 32,
    "batch_timeout": 0.05,
    "min_batch_size": 4,
    
    "metrics_enabled": True,
    "detailed_latency": False,
    "metrics_window_size": 500,
    "metrics_reset_interval": 3600.0,
    
    "multiplexing_enabled": True,
    "default_strategy": "least-loaded",
    "key_switch_threshold": 0.9
}
```

### High-Reliability Profile

Optimized for maximum reliability with conservative settings.

```python
high_reliability_config = {
    "max_concurrent_requests": 8,
    "queue_size": 100,
    "queue_check_interval": 0.01,
    "auto_scale_concurrency": False,
    
    "circuit_enabled": True,
    "failure_threshold": 2,
    "reset_timeout": 120.0,
    "half_open_max_requests": 1,
    
    "enable_batching": False,
    
    "metrics_enabled": True,
    "detailed_latency": True,
    "error_tracking": True,
    "request_tracing": True,
    
    "multiplexing_enabled": True,
    "fallback_enabled": True,
    "max_key_attempts": 5,
    "track_key_performance": True
}
```

### Balanced Profile (Default)

Balanced performance and reliability for general use.

```python
balanced_config = {
    "max_concurrent_requests": 15,
    "queue_size": 200,
    
    "circuit_enabled": True,
    "failure_threshold": 5,
    "reset_timeout": 30.0,
    
    "enable_batching": True,
    "max_batch_size": 8,
    "batch_timeout": 0.1,
    
    "metrics_enabled": True,
    "metrics_window_size": 1000,
    
    "multiplexing_enabled": False  # Enable manually when needed
}
```

### Low-Resource Profile

Minimized resource usage for constrained environments.

```python
low_resource_config = {
    "max_concurrent_requests": 3,
    "queue_size": 50,
    "queue_check_interval": 0.05,
    
    "circuit_enabled": True,
    "failure_threshold": 3,
    "reset_timeout": 60.0,
    
    "enable_batching": True,
    "max_batch_size": 4,
    "batch_timeout": 0.2,
    
    "metrics_enabled": True,
    "detailed_latency": False,
    "metrics_window_size": 100,
    "metrics_reset_interval": 0.0,  # Never reset
    
    "multiplexing_enabled": False
}
```

## Usage Example

Apply configuration when initializing the accelerate client:

```python
from ipfs_accelerate import accelerate
from ipfs_accelerate.config import high_performance_config

# Initialize with high performance profile
client = accelerate(**high_performance_config)

# Override specific settings
client.max_concurrent_requests = 25
client.enable_batching = False

# Get OpenAI API client with this configuration
openai_client = client.get_api_client("openai_api")
```

Or apply configuration to a specific API client:

```python
from ipfs_accelerate import openai_api
from ipfs_accelerate.config import high_reliability_config

# Create client with high reliability profile
client = openai_api(config=high_reliability_config)
```