# API Enhancements: Queue, Backoff, and Environment Variables

This document explains the improved API implementation with request queuing, exponential backoff, and environment variable handling for API keys.

## Key Features

### 1. Request Queuing

The API implementation now includes a robust request queue system that:

- Limits concurrent API requests to prevent rate limiting (default: 5 concurrent requests)
- Automatically queues additional requests when at capacity
- Processes queued requests in FIFO order
- Includes configurable queue size (default: 100 requests)
- Provides thread-safe operation with proper locking
- Handles asynchronous request processing in a background thread

### 2. Exponential Backoff

All API requests now include automatic exponential backoff retry which:

- Automatically retries failed requests due to rate limiting or transient API errors
- Uses exponential delay between retries (1s, 2s, 4s, 8s, 16s by default)
- Respects `retry-after` headers from the API when available
- Provides configurable retry limits (default: 5 retries)
- Includes detailed logging of retry attempts

### 3. Environment Variable Handling

API keys can now be loaded from environment variables:

- Automatically checks for `OPENAI_API_KEY` environment variable
- Integrates with python-dotenv for loading from `.env` files
- Provides clear logging when keys are loaded from environment variables
- Maintains backward compatibility with existing key handling approaches

## Installation

1. Install the required dependency for environment variable loading:

```bash
pip install python-dotenv
```

2. Create a `.env` file in your project directory with your API keys:

```
OPENAI_API_KEY=your_openai_key_here
```

3. Ensure your `.gitignore` includes `.env` to avoid committing secrets:

```
.env
```

## Usage Examples

### Basic Usage

The enhancements are automatically enabled, so your existing code will work with improved reliability:

```python
from ipfs_accelerate_py.api_backends import openai_api

# API key will be loaded from environment variable if not provided
api = openai_api(resources={}, metadata={})

# Make a request - queue and backoff happen automatically
result = api.chat("gpt-4o", [{"role": "user", "content": "Hello"}])
```

### Configuring Queue and Backoff Settings

You can customize the queue and backoff settings:

```python
from ipfs_accelerate_py.api_backends import openai_api

api = openai_api(resources={}, metadata={})

# Configure queue settings
api.queue_enabled = True
api.max_concurrent_requests = 10  # Allow more concurrent requests
api.queue_size = 200  # Increase queue capacity

# Configure backoff settings
api.max_retries = 8
api.initial_retry_delay = 2  # Start with 2 second delay
api.backoff_factor = 3  # Use more aggressive backoff
api.max_retry_delay = 120  # Cap at 2 minutes

# Make requests - the enhanced settings will be used
result = api.chat("gpt-4o", [{"role": "user", "content": "Hello"}])
```

### Disabling Features

You can disable the queueing if needed:

```python
from ipfs_accelerate_py.api_backends import openai_api

api = openai_api(resources={}, metadata={})

# Disable queue (backoff will still be active)
api.queue_enabled = False

# Make requests
result = api.chat("gpt-4o", [{"role": "user", "content": "Hello"}])
```

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

### Environment Variables

The implementation checks for environment variables in this order:

1. Explicit API key provided in the constructor
2. API key provided in the `metadata` dictionary
3. API key from environment variable
4. Empty string (which will lead to authentication errors from the API)

## Testing

Comprehensive tests have been added to verify all new functionality:

- Queue functionality tests
- Backoff retry mechanism tests
- Environment variable handling tests

Run the tests with:

```bash
python -m test.apis.test_openai_api
```

## Extending to Other APIs

The queue and backoff implementations can be easily added to other API backends using:

```bash
python -m test.fix_openai_api_implementation
```

To apply similar enhancements to all API backends:

```bash
python -m test.add_queue_backoff --api all
```

## API Status Monitoring

You can monitor the queue status during usage:

```python
# Check queue size
print(f"Queue size: {len(api.request_queue)}")

# Check current request count
print(f"Current requests: {api.current_requests}/{api.max_concurrent_requests}")
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

## Support

For questions or issues with these enhancements, please file an issue in the repository or contact the development team.