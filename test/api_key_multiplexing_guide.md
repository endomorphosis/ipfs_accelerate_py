# API Key Multiplexing Guide

## Overview

The API Key Multiplexing feature allows you to use multiple API keys for OpenAI, Groq, Claude, and Gemini APIs, with the following benefits:

1. **Load Balancing**: Distribute requests across multiple API keys to avoid rate limits
2. **Request Queueing**: Each API key has its own separate request queue
3. **Efficient Resource Usage**: Automatically selects the best API key using different strategies
4. **Parallel Requests**: Run multiple concurrent requests across different API keys

This guide explains how to configure and use the API Key Multiplexing feature in the IPFS Accelerate Python framework.

## Configuration

### 1. API Key Setup in .env File

Create a `.env` file in your project directory with multiple API keys:

```
# OpenAI API Keys
OPENAI_API_KEY=sk-your-main-key               # Primary key
OPENAI_API_KEY_1=sk-your-first-key            # Key 1 for multiplexing
OPENAI_API_KEY_2=sk-your-second-key           # Key 2 for multiplexing
OPENAI_API_KEY_3=sk-your-third-key            # Key 3 for multiplexing

# Groq API Keys
GROQ_API_KEY=gsk-your-main-key                # Primary key
GROQ_API_KEY_1=gsk-your-first-key             # Key 1 for multiplexing
GROQ_API_KEY_2=gsk-your-second-key            # Key 2 for multiplexing

# Claude API Keys
CLAUDE_API_KEY=sk-ant-your-main-key           # Primary key
ANTHROPIC_API_KEY_1=sk-ant-your-first-key     # Key 1 for multiplexing
ANTHROPIC_API_KEY_2=sk-ant-your-second-key    # Key 2 for multiplexing

# Gemini API Keys
GEMINI_API_KEY=your-main-key                  # Primary key
GEMINI_API_KEY_1=your-first-key               # Key 1 for multiplexing
GEMINI_API_KEY_2=your-second-key              # Key 2 for multiplexing
```

The multiplexer will automatically detect and use these keys when you run the API tests.

### 2. Multiplexing Configuration

The API Key Multiplexer supports several configuration options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_concurrent_requests` | Maximum concurrent requests per API key | 5 |
| `queue_size` | Maximum queue size per API key | 100 |
| `initial_retry_delay` | Initial delay for backoff (seconds) | 1 |
| `backoff_factor` | Factor to multiply delay on each retry | 2 |
| `max_retries` | Maximum number of retries for failed requests | 5 |

## Usage

### Basic Usage in Code

```python
from test.api_key_multiplexing_example import ApiKeyMultiplexer

# Create the multiplexer
multiplexer = ApiKeyMultiplexer()

# Add API keys (can also be loaded automatically from environment)
multiplexer.add_openai_key("key1", "sk-your-first-key")
multiplexer.add_openai_key("key2", "sk-your-second-key")
multiplexer.add_groq_key("groq1", "gsk-your-first-key")

# Get an OpenAI client using round-robin strategy
openai_client = multiplexer.get_openai_client(strategy="round-robin")

# Use the client to make a request
response = openai_client("chat", messages=[{"role": "user", "content": "Hello"}])

# Get client statistics
stats = multiplexer.get_usage_stats()
print(stats)
```

### Selection Strategies

The multiplexer supports multiple strategies for selecting API keys:

1. **specific**: Use a specific key by name
   ```python
   client = multiplexer.get_openai_client(key_name="key1")
   ```

2. **round-robin**: Select the least recently used key (default)
   ```python
   client = multiplexer.get_openai_client(strategy="round-robin")
   ```

3. **least-loaded**: Select the key with the smallest request queue
   ```python
   client = multiplexer.get_openai_client(strategy="least-loaded")
   ```

## Integration with test_ipfs_accelerate

The API Key Multiplexing is automatically integrated with the `test_ipfs_accelerate` environment as the final test. It will:

1. Detect all available API keys in the environment
2. Create separate client instances for each key
3. Run concurrent requests across all available APIs
4. Report detailed statistics on queue usage and request distribution

The test results will be included in the standard test output JSON file.

## Running Standalone Tests

You can also run the API Key Multiplexing tests standalone:

```bash
python -m test.api_key_multiplexing_example
```

This will:
1. Load all API keys from your environment
2. Run concurrent requests across all available API providers
3. Print detailed statistics about request distribution

## Performance Considerations

- Each API key maintains its own separate queue and backoff timer
- The multiplexer uses thread locks to ensure thread safety
- Queue processing happens in background threads
- API keys with fewer requests are automatically favored by the least-loaded strategy

## Best Practices

1. Use multiple API keys from different accounts to maximize throughput
2. Configure different `max_concurrent_requests` values based on your rate limits
3. Use the least-loaded strategy for optimal throughput during heavy load
4. Monitor usage statistics to ensure even distribution across keys
5. Avoid using keys with different permission levels in the same multiplexer
6. Store your API keys securely in a `.env` file (never commit them to version control)

## Error Handling

The multiplexer provides detailed error information:

- Each API client has its own queue and backoff handling
- Errors from one API key won't affect requests to other keys
- Failed requests are tracked and reported in the statistics
- Rate limit errors trigger exponential backoff specific to that key