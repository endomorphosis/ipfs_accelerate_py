# Advanced API Features Documentation

This document describes the advanced features implemented in the API backends of the IPFS Accelerate Python framework, including the priority queue system, circuit breaker pattern, enhanced monitoring, request batching, and API key multiplexing capabilities.

## Advanced Features Overview

All API backends now include advanced features for improved reliability, performance, and monitoring:

### Thread-safe Request Queue
- Configurable queue size (default: 100)
- Configurable concurrency limits (default: 5)
- Queue monitoring with metrics
- Priority-based queue processing
- Queue overflow protection

### Exponential Backoff System
- Configurable retry attempts (default: 5)
- Configurable initial delay (default: 1s)
- Configurable backoff factor (default: 2)
- Configurable maximum delay (default: 16s)
- Error classification for retry decisions
- Rate limit detection and handling

### API Key Multiplexing
- Multiple API key support for each provider
- Round-robin key selection strategy
- Least-loaded key selection strategy
- Per-key usage metrics
- Thread-safe key management

## Implementation Status by API

| API Backend | Implementation | Queue & Backoff | Key Multiplexing | Real Integration | Testing Status |
|-------------|----------------|----------------|------------------|-----------------|----------------|
| OpenAI API  | REAL           | ✅             | ✅               | ✅              | PASS           |
| Claude API  | REAL           | ✅             | ✅               | ✅              | PASS           |
| Groq API    | REAL           | ✅             | ✅               | ✅              | PASS           |
| Ollama API  | REAL           | ✅             | ⚠️ (N/A)         | ✅              | PASS           |
| HF TGI API  | MOCK           | ✅             | ✅               | ⏳              | PARTIAL        |
| HF TEI API  | MOCK           | ✅             | ✅               | ⏳              | PARTIAL        |
| Gemini API  | MOCK           | ✅             | ✅               | ⏳              | PARTIAL        |
| LLVM API    | MOCK           | ✅             | ⚠️ (N/A)         | ⏳              | PARTIAL        |
| OVMS API    | REAL           | ✅             | ✅               | ✅              | PASS           |
| OPEA API    | MOCK           | ✅             | ✅               | ⏳              | PARTIAL        |
| S3 Kit API  | MOCK           | ✅             | ⚠️ (N/A)         | ⏳              | PARTIAL        |

## Test Suite

The queue and backoff system has been thoroughly tested with:

### API-specific Tests
- Tests with all supported models for each API
- Specific rate limit and error handling tests
- Authentication and credential management tests
- Response format validation

### Queue System Tests
- Queue size limit tests
- Concurrency limit tests
- Queue overflow tests
- Priority-based processing tests

### Backoff System Tests
- Retry logic tests
- Error classification tests
- Rate limit detection tests
- Connection error recovery tests

### API Key Multiplexing Tests
- Round-robin selection tests
- Least-loaded selection tests
- High concurrency tests
- Performance comparison tests

## New Testing Tools

### Comprehensive Ollama Tests
Tests the Ollama API with various queue and backoff configurations:

```bash
python test_ollama_backoff_comprehensive.py --model llama3 --host http://localhost:11434
```

### Enhanced API Multiplexing Tests
Tests API key multiplexing with different selection strategies:

```bash
python test_api_multiplexing_enhanced.py
```

### Combined Test Runner
Runs all queue and backoff tests with configurable options:

```bash
# Run all tests
python run_queue_backoff_tests.py

# Test specific APIs only
python run_queue_backoff_tests.py --apis openai groq claude

# Skip specific APIs
python run_queue_backoff_tests.py --skip-apis vllm opea ovms
```

## Usage Examples

### Basic Queue and Backoff Usage

```python
from ipfs_accelerate_py.api_backends import openai_api

# Create client with queue and backoff configuration
client = openai_api(
    resources={},
    metadata={
        "openai_api_key": "your-api-key",
        "max_concurrent_requests": 10,
        "queue_size": 50,
        "max_retries": 3,
        "backoff_factor": 1.5
    }
)

# Request automatically uses queue and backoff system
response = client.chat(
    model_name="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

### API Key Multiplexing Usage

```python
from ipfs_accelerate_py.api_key_multiplexing import ApiKeyMultiplexer

# Create multiplexer
multiplexer = ApiKeyMultiplexer()

# Add multiple keys
multiplexer.add_openai_key("key1", "sk-api-key-1", max_concurrent=5)
multiplexer.add_openai_key("key2", "sk-api-key-2", max_concurrent=5)
multiplexer.add_openai_key("key3", "sk-api-key-3", max_concurrent=5)

# Get client using round-robin strategy
client = multiplexer.get_openai_client(strategy="round-robin")
response = client.chat(
    model_name="gpt-4",
    messages=[{"role": "user", "content": "Hello, world!"}]
)

# Get client using least-loaded strategy
client = multiplexer.get_openai_client(strategy="least-loaded")
response = client.chat(
    model_name="gpt-4",
    messages=[{"role": "user", "content": "Hello again!"}]
)

# Get usage statistics
stats = multiplexer.get_usage_stats()
print(f"OpenAI key usage: {stats['openai']}")
```

## Environment Variables for API Keys

### Standard API Keys
- OpenAI API: `OPENAI_API_KEY`
- Claude API: `ANTHROPIC_API_KEY`
- Groq API: `GROQ_API_KEY`
- Hugging Face: `HF_API_TOKEN`
- Gemini API: `GOOGLE_API_KEY`

### Multiple API Keys for Multiplexing
- OpenAI keys: `OPENAI_API_KEY_1`, `OPENAI_API_KEY_2`, `OPENAI_API_KEY_3`
- Groq keys: `GROQ_API_KEY_1`, `GROQ_API_KEY_2`
- Claude keys: `ANTHROPIC_API_KEY_1`, `ANTHROPIC_API_KEY_2`
- Gemini keys: `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`

## Implementation Details

### Queue Processing

The queue system uses a background thread to process requests when the application reaches its concurrent request limit. This approach:

1. Prevents blocking the main thread
2. Ensures requests are processed in order
3. Maintains the API's response format and error handling
4. Provides proper cleanup of resources

### Backoff Strategy

The backoff mechanism implements a standard exponential backoff with customizable parameters:

1. Initial delay: How long to wait after the first failure
2. Backoff factor: How much to multiply the delay on each retry
3. Maximum delay: Cap on how long the delay can be
4. Respect for API guidance: Uses the `retry-after` header when provided

### API Key Multiplexing

The key multiplexing system provides several benefits:

1. Higher throughput by distributing requests across multiple API keys
2. Automatic fallback if one key hits rate limits
3. Load balancing based on queue depth
4. Usage tracking for billing and capacity planning
5. Thread-safe operation for concurrent applications

## Monitoring and Status

You can monitor the queue status during usage:

```python
# Check queue size
print(f"Queue size: {len(client.request_queue)}")

# Check current request count
print(f"Current requests: {client.current_requests}/{client.max_concurrent_requests}")

# When using multiplexing, get detailed usage stats
stats = multiplexer.get_usage_stats()
for key, data in stats["openai"].items():
    print(f"Key {key}: {data['usage']} requests, {data['queue_size']} queued")
```

## Extending to Other APIs

The queue and backoff implementations have been added to all API backends using the provided script:

```bash
python add_queue_backoff.py --api all
```

For individual APIs:

```bash
python add_queue_backoff.py --api [openai|groq|claude|gemini|etc]
```

## Troubleshooting

### API Errors Despite Retry

If you're still seeing API errors despite the retry mechanism:

1. Check your API key is valid
2. Increase the max_retries setting
3. Examine the logs for error details (messages include retry count)

### Queue Capacity Issues

If you see "Request queue is full" errors:

1. Increase the queue_size parameter
2. Consider implementing rate control on your side to prevent queue overflow
3. Implement request batching to reduce the number of API calls

## Future Improvements

1. **Priority-based Queue Processing**
   - Implement priority-based queue for critical requests
   - Allow priority boosting for certain request types
   - Add deadline-based scheduling

2. **Advanced Backoff Strategies**
   - Implement jitter for distributed systems
   - Add adaptive backoff based on response patterns
   - Implement circuit breaker pattern for persistent failures

3. **Enhanced Key Management**
   - Add key rotation based on usage quotas
   - Implement automatic key validation
   - Add cost estimation and budget tracking