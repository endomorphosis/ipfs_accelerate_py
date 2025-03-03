# API Error Handling Documentation

This document provides comprehensive guidance on error handling across all API backends in the IPFS Accelerate Python framework.

## Error Categories

The framework handles several categories of errors across all API backends, with standardized error handling patterns.

### 1. Connection Errors

Connection errors occur when the API client cannot establish or maintain a connection with the remote service.

#### Types of Connection Errors:
- **Network Connectivity Failures**
  - `ConnectionError`: Underlying connection was lost or refused
  - `DNSLookupError`: Failed to resolve service hostname
  - `TimeoutError`: Connection or read timeout
  - `SSLError`: SSL/TLS certificate verification failed
  
#### Handling Strategy:
```python
from ipfs_accelerate_py.api_backends import openai_api
import requests.exceptions

client = openai_api()

try:
    response = client.chat(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )
except requests.exceptions.ConnectionError as e:
    # Connection error - will automatically retry with backoff
    print(f"Connection error (will retry automatically): {str(e)}")
except requests.exceptions.Timeout as e:
    # Timeout error - will automatically retry with backoff
    print(f"Timeout error (will retry automatically): {str(e)}")
```

### 2. Authentication Errors

Authentication errors occur when API credentials are invalid, expired, or have insufficient permissions.

#### Types of Authentication Errors:
- **Credential Issues**
  - `AuthenticationError`: Invalid API key or token
  - `TokenExpiredError`: API token has expired
  - `PermissionDeniedError`: Insufficient permissions
  - `RateLimitExceededError`: API quota or rate limits exceeded
  
#### Handling Strategy:
```python
from ipfs_accelerate_py.api_backends import claude
import requests.exceptions

client = claude()

try:
    response = client.chat([{"role": "user", "content": "Hello"}])
except ValueError as e:
    error_info = e.args[0]
    if "authentication" in str(e).lower():
        # Authentication error - check your API key
        print(f"Authentication error: {str(e)}")
    elif "rate limit" in str(e).lower():
        # Rate limit error - will automatically retry with backoff
        print(f"Rate limit error (will retry automatically): {str(e)}")
```

### 3. Request Formatting Errors

Request formatting errors occur when the client sends invalid parameters or payloads.

#### Types of Request Formatting Errors:
- **Payload Validation Issues**
  - `InvalidRequestError`: Malformed request parameters
  - `ContextLengthExceededError`: Input exceeds model's context window
  - `InvalidModelError`: Requested model is not available
  - `ValidationError`: General parameter validation error
  
#### Handling Strategy:
```python
from ipfs_accelerate_py.api_backends import groq
import requests.exceptions

client = groq()

try:
    # Potentially too long context
    very_long_text = "..." * 10000  # Extremely long text
    response = client.chat(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": very_long_text}]
    )
except ValueError as e:
    if "context length" in str(e).lower():
        # Context length error - truncate input
        print(f"Context length exceeded: {str(e)}")
        # Implement truncation strategy
    elif "invalid model" in str(e).lower():
        # Invalid model error - use a different model
        print(f"Invalid model: {str(e)}")
        # Fallback to another model
```

### 4. Service Availability Errors

Service availability errors occur when the backend service is experiencing problems.

#### Types of Service Availability Errors:
- **Server-Side Issues**
  - `ServerError`: Internal server error (HTTP 5xx)
  - `ServiceUnavailableError`: Service is temporarily unavailable
  - `CircuitBreakerOpenError`: Circuit breaker has opened due to failures
  - `RegionOutageError`: Specific region is experiencing an outage
  
#### Handling Strategy:
```python
from ipfs_accelerate_py.api_backends import ollama
import requests.exceptions

client = ollama()

try:
    response = client.chat(
        model="llama3",
        messages=[{"role": "user", "content": "Hello"}]
    )
except Exception as e:
    if "circuit breaker is open" in str(e).lower():
        # Circuit breaker is open - service appears down
        print(f"Service appears to be down: {str(e)}")
        # Implement fallback to another service
    elif "503" in str(e) or "service unavailable" in str(e).lower():
        # Service is unavailable - will retry automatically
        print(f"Service unavailable (will retry automatically): {str(e)}")
```

### 5. Response Processing Errors

Response processing errors occur when handling the response from the API.

#### Types of Response Processing Errors:
- **Output Handling Issues**
  - `ResponseParsingError`: Failed to parse API response
  - `IncompleteResponseError`: Response was truncated or incomplete
  - `ContentFilterError`: Content was filtered by the API safety system
  - `UnexpectedResponseError`: Response did not match expected format
  
#### Handling Strategy:
```python
from ipfs_accelerate_py.api_backends import gemini
import json

client = gemini()

try:
    response = client.chat(
        messages=[{"role": "user", "content": "Generate a JSON response"}]
    )
    
    # Try to parse JSON from the response
    try:
        json_data = json.loads(response["text"])
    except json.JSONDecodeError:
        # Handle invalid JSON response
        print("Received invalid JSON response")
        # Implement fallback strategy
except Exception as e:
    if "content filtered" in str(e).lower():
        # Content was filtered
        print(f"Content filtered: {str(e)}")
        # Implement content modification strategy
```

## Built-in Error Handling Features

The API backends in the IPFS Accelerate Python framework include several built-in error handling features:

### 1. Exponential Backoff

All retryable errors trigger automatic exponential backoff:

```python
client = openai_api()

# Configure backoff settings
client.max_retries = 5                    # Maximum retry attempts (default: 5)
client.initial_retry_delay = 1            # Initial delay in seconds (default: 1)
client.backoff_factor = 2                 # Multiply delay by this factor each retry (default: 2)
client.max_retry_delay = 60               # Maximum delay cap in seconds (default: 60)
```

### 2. Circuit Breaker

The circuit breaker automatically protects against repeated failures:

```python
client = groq()

# Configure circuit breaker (if implemented)
if hasattr(client, "circuit_failure_threshold"):
    client.circuit_failure_threshold = 5  # Failures before opening circuit (default: 5)
    client.circuit_reset_timeout = 30     # Seconds before trying half-open (default: 30)
    client.circuit_success_threshold = 3  # Success count to close circuit (default: 3)
```

### 3. Error Classification

All errors are classified for appropriate handling:

```python
try:
    response = client.chat(...)
except Exception as e:
    error_info = getattr(e, "error_info", {})
    error_type = error_info.get("type", "unknown")
    should_retry = error_info.get("should_retry", False)
    
    if should_retry:
        # Error can be retried automatically
        print(f"Retryable error of type {error_type}")
    else:
        # Error requires intervention
        print(f"Non-retryable error of type {error_type}")
```

### 4. Error Metrics

All APIs track error metrics for monitoring:

```python
# After handling errors, check error metrics
if hasattr(client, "get_error_metrics"):
    metrics = client.get_error_metrics()
    print(f"Total errors: {metrics.get('total_errors', 0)}")
    print(f"Error rate: {metrics.get('error_rate', 0):.2f}%")
    
    # Error breakdown by type
    error_types = metrics.get('error_types', {})
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count}")
```

## Advanced Error Handling Strategies

### 1. API Key Multiplexing for Reliability

Use multiple API keys to handle authentication errors:

```python
from ipfs_accelerate_py.api_backends import openai_api
from ipfs_accelerate_py.api_key_multiplexing import ApiKeyMultiplexer

# Create multiplexer with multiple keys
multiplexer = ApiKeyMultiplexer()
multiplexer.add_openai_key("key1", "sk-key1...")
multiplexer.add_openai_key("key2", "sk-key2...")
multiplexer.add_openai_key("key3", "sk-key3...")

# Get client with automatic failover between keys
client = multiplexer.get_openai_client(strategy="least-loaded")

# Now use client as normal - if one key has issues, it will use another
response = client.chat(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### 2. Endpoint Failover for Service Availability

Use multiple endpoints for better availability:

```python
from ipfs_accelerate_py.api_backends import s3_kit

s3_client = s3_kit()

# Create endpoint multiplexer
class EndpointMultiplexer:
    def __init__(self, client):
        self.client = client
        self.endpoints = {}
        
    def add_endpoint(self, name, url, **config):
        self.endpoints[name] = {
            "url": url,
            "config": config,
            "handler": self.client.create_endpoint_handler(url, **config)
        }
        
    def execute(self, operation, **params):
        # Try primary endpoint
        try:
            return self.endpoints["primary"]["handler"](operation, **params)
        except Exception as e:
            # On failure, try backup endpoint
            print(f"Primary endpoint failed: {str(e)}")
            return self.endpoints["backup"]["handler"](operation, **params)

# Use multiplexer
multiplexer = EndpointMultiplexer(s3_client)
multiplexer.add_endpoint("primary", "https://primary-endpoint.com")
multiplexer.add_endpoint("backup", "https://backup-endpoint.com")

# Automatic failover between endpoints
result = multiplexer.execute("list_objects", bucket="my-bucket")
```

### 3. Content Filtering Error Handling

Handle content moderation errors:

```python
from ipfs_accelerate_py.api_backends import claude

client = claude()

def send_with_moderation_handling(messages):
    try:
        return client.chat(messages)
    except Exception as e:
        if "content policy" in str(e).lower() or "content filtered" in str(e).lower():
            # Content was filtered - try to modify the request
            print("Content moderation triggered, adjusting request...")
            
            # Strategy 1: Add a more explicit system message
            system_message = "Please provide a helpful, harmless, and ethical response."
            new_messages = [{"role": "system", "content": system_message}] + messages
            
            try:
                return client.chat(new_messages)
            except Exception as e2:
                print(f"Modified request also failed: {str(e2)}")
                return {"text": "Unable to generate response due to content policy."}
        else:
            # Other type of error
            raise e

# Use the wrapper function
response = send_with_moderation_handling([
    {"role": "user", "content": "Please explain this topic..."}
])
```

### 4. Context Length Error Handling

Handle input that exceeds context windows:

```python
from ipfs_accelerate_py.api_backends import groq
import tiktoken  # For token counting

client = groq()

def send_with_context_handling(model, messages, max_tokens=8192):
    # Count tokens in messages
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Approximate for most models
    
    # Count tokens in each message
    token_count = 0
    for msg in messages:
        token_count += len(encoding.encode(msg["content"]))
        token_count += 4  # Overhead per message
    
    # Add overhead for formatting
    token_count += 2
    
    if token_count > max_tokens - 500:  # Leave room for response
        print(f"Input too long ({token_count} tokens), truncating...")
        
        # Strategy: Keep system message, truncate user messages from oldest
        system_messages = [m for m in messages if m["role"] == "system"]
        user_messages = [m for m in messages if m["role"] != "system"]
        
        # Always keep the latest user message
        latest_user_message = user_messages.pop() if user_messages else None
        
        # Initialize with system messages
        truncated_messages = system_messages.copy()
        current_tokens = sum(len(encoding.encode(m["content"])) + 4 for m in truncated_messages)
        
        # Add as many user messages as fit, from newest to oldest
        user_messages.reverse()  # Start with most recent (except the very latest)
        for msg in user_messages:
            msg_tokens = len(encoding.encode(msg["content"])) + 4
            if current_tokens + msg_tokens < max_tokens - 500:
                truncated_messages.append(msg)
                current_tokens += msg_tokens
            else:
                # Can't fit this message
                break
                
        # Always add the latest user message, truncating if needed
        if latest_user_message:
            latest_content = latest_user_message["content"]
            latest_tokens = len(encoding.encode(latest_content)) + 4
            
            # If it doesn't fit, truncate it
            if current_tokens + latest_tokens > max_tokens - 500:
                # Calculate how many tokens we can use
                available_tokens = max_tokens - 500 - current_tokens
                if available_tokens > 20:  # Only if we have enough room for a meaningful message
                    # Truncate by token, which is approximate
                    truncated_content = encoding.decode(encoding.encode(latest_content)[:available_tokens])
                    latest_user_message["content"] = truncated_content + "... [truncated]"
            
            truncated_messages.append(latest_user_message)
        
        # Use truncated messages
        return client.chat(model=model, messages=truncated_messages)
    else:
        # No truncation needed
        return client.chat(model=model, messages=messages)

# Use the context handling wrapper
long_conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Very long message..." * 1000},
    # ... more messages
]

response = send_with_context_handling("llama3-70b-8192", long_conversation, max_tokens=8192)
```

## Error Reporting and Monitoring

All API backends include comprehensive error reporting and monitoring:

```python
# Configure error reporting
client.enable_error_reporting = True       # Enable detailed error reporting
client.error_log_size = 100                # Keep last 100 errors in memory
client.report_errors_to_console = True     # Print errors to console when they occur

# After running operations, check error reports
if hasattr(client, "get_error_report"):
    report = client.get_error_report()
    print(f"Total errors: {report['total_errors']}")
    
    # Show recent errors
    for error in report["recent_errors"][:5]:
        print(f"Error at {error['timestamp']}: {error['message']}")
        print(f"  Type: {error['type']}")
        print(f"  Request ID: {error['request_id']}")
        print(f"  Retried: {error['retried']}")
```

## Conclusion

This documentation provides a comprehensive guide to error handling across all API backends in the IPFS Accelerate Python framework. By following these patterns and using the built-in error handling features, you can build robust applications that gracefully handle failures and provide a reliable experience.

For API-specific error handling, refer to the individual API documentation files in the `apis/` directory.