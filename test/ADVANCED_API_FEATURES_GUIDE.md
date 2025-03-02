# Advanced API Features Guide

This comprehensive guide covers all advanced features implemented in the IPFS Accelerate Python Framework API system. These features work across all 11 supported API backends and provide enhanced reliability, performance, and monitoring capabilities.

## Table of Contents

1. [Priority Queue System](#priority-queue-system)
2. [Circuit Breaker Pattern](#circuit-breaker-pattern)
3. [Monitoring and Reporting](#monitoring-and-reporting)
4. [Request Batching](#request-batching)
5. [API Key Multiplexing](#api-key-multiplexing)
6. [Configuration Reference](#configuration-reference)
7. [Integration Patterns](#integration-patterns)

## Priority Queue System

The priority queue system provides a robust way to manage concurrent requests with different priority levels.

### Key Features

- **Three-tier priority levels**: HIGH, NORMAL, LOW
- **Thread-safe request queueing** with proper locking
- **Priority-based scheduling and processing**
- **Dynamic queue size configuration**
- **Queue status monitoring and metrics**

### Implementation

All API backends implement a consistent queue system with the following components:

```python
def __init__(self, resources=None, metadata=None):
    # Queue configuration
    self.max_concurrent_requests = 5
    self.queue_size = 100
    self.request_queue = []  # List-based queue
    self.active_requests = 0
    self.queue_lock = threading.RLock()
    self.queue_processing = True
    
    # Start queue processor
    self.queue_processor = threading.Thread(target=self._process_queue)
    self.queue_processor.daemon = True
    self.queue_processor.start()
```

### Queue Processing

The queue processor runs in a background thread and processes requests based on priority:

```python
def _process_queue(self):
    while self.queue_processing:
        try:
            # Process requests with proper concurrency management
            if not self.request_queue or self.active_requests >= self.max_concurrent_requests:
                time.sleep(0.01)
                continue
                
            # Sort queue by priority before processing
            self.request_queue.sort(key=lambda x: x[0])
            
            with self.queue_lock:
                if self.request_queue:
                    priority, future, func, args, kwargs = self.request_queue.pop(0)
                    self.active_requests += 1
                else:
                    continue
            
            # Process the request
            try:
                result = func(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                with self.queue_lock:
                    self.active_requests -= 1
                    
        except Exception as e:
            print(f"Error in queue processor: {e}")
            time.sleep(0.1)
```

### Usage Example

```python
# Submit a high-priority request
future = api_client._queue_request(func=api_client.generate_text, 
                                  priority=0,  # 0=HIGH, 1=NORMAL, 2=LOW
                                  text="Generate with high priority")

# Get the result
result = future.result()
```

## Circuit Breaker Pattern

The circuit breaker pattern prevents cascading failures and allows systems to recover from service outages.

### Key Features

- **Three-state machine**: CLOSED, OPEN, HALF-OPEN
- **Automatic service outage detection**
- **Self-healing capabilities** with configurable timeouts
- **Failure threshold configuration**
- **Fast-fail for unresponsive services**

### Implementation

All API backends implement a consistent circuit breaker with the following state machine:

```python
def __init__(self, resources=None, metadata=None):
    # Circuit breaker configuration
    self.circuit_state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
    self.failure_count = 0
    self.failure_threshold = 5
    self.reset_timeout = 30  # seconds
    self.last_failure_time = 0
    self.circuit_lock = threading.RLock()
```

### Circuit State Transitions

```python
def _check_circuit(self):
    """Check the circuit state before making a request"""
    with self.circuit_lock:
        current_time = time.time()
        
        # If OPEN, check if we should try HALF-OPEN
        if self.circuit_state == "OPEN":
            if current_time - self.last_failure_time > self.reset_timeout:
                self.circuit_state = "HALF-OPEN"
                return True
            return False
            
        # If HALF-OPEN or CLOSED, allow the request
        return True
        
def _on_success(self):
    """Handle successful request"""
    with self.circuit_lock:
        if self.circuit_state == "HALF-OPEN":
            # Reset on successful request in HALF-OPEN state
            self.circuit_state = "CLOSED"
            self.failure_count = 0
            
def _on_failure(self):
    """Handle failed request"""
    with self.circuit_lock:
        self.last_failure_time = time.time()
        
        if self.circuit_state == "HALF-OPEN":
            # Return to OPEN on failure in HALF-OPEN
            self.circuit_state = "OPEN"
        elif self.circuit_state == "CLOSED":
            # Increment failure count
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.circuit_state = "OPEN"
```

### Usage in Request Handling

```python
def make_api_request(self, *args, **kwargs):
    if not self._check_circuit():
        raise ServiceUnavailableError("Circuit breaker is OPEN")
        
    try:
        result = self._actual_api_call(*args, **kwargs)
        self._on_success()
        return result
    except Exception as e:
        self._on_failure()
        raise
```

## Monitoring and Reporting

Comprehensive monitoring and reporting capabilities across all API backends.

### Key Features

- **Comprehensive request statistics tracking**
- **Error classification and tracking by type**
- **Performance metrics by model and endpoint**
- **Queue and backoff metrics collection**
- **Detailed reporting capabilities**

### Implementation

```python
def __init__(self, resources=None, metadata=None):
    # Monitoring configuration
    self.metrics = {
        "requests": 0,
        "successes": 0,
        "failures": 0,
        "timeouts": 0,
        "retries": 0,
        "latency": [],
        "error_types": {}
    }
    self.metrics_lock = threading.RLock()
```

### Metrics Collection

```python
def _update_metrics(self, success=True, latency=None, error=None, retried=False):
    with self.metrics_lock:
        self.metrics["requests"] += 1
        
        if success:
            self.metrics["successes"] += 1
        else:
            self.metrics["failures"] += 1
            
        if latency is not None:
            self.metrics["latency"].append(latency)
            
        if retried:
            self.metrics["retries"] += 1
            
        if error is not None:
            error_type = type(error).__name__
            if error_type not in self.metrics["error_types"]:
                self.metrics["error_types"][error_type] = 0
            self.metrics["error_types"][error_type] += 1
```

### Reporting Interface

```python
def get_metrics_report(self):
    with self.metrics_lock:
        avg_latency = sum(self.metrics["latency"]) / len(self.metrics["latency"]) if self.metrics["latency"] else 0
        
        report = {
            "total_requests": self.metrics["requests"],
            "success_rate": self.metrics["successes"] / self.metrics["requests"] if self.metrics["requests"] else 0,
            "failure_rate": self.metrics["failures"] / self.metrics["requests"] if self.metrics["requests"] else 0,
            "retry_rate": self.metrics["retries"] / self.metrics["requests"] if self.metrics["requests"] else 0,
            "average_latency": avg_latency,
            "current_queue_size": len(self.request_queue),
            "active_requests": self.active_requests,
            "circuit_state": self.circuit_state,
            "error_breakdown": self.metrics["error_types"]
        }
        
        return report
```

## Request Batching

Automatic request combining for compatible operations to improve throughput.

### Key Features

- **Automatic request combining** for compatible models
- **Configurable batch size and timeout**
- **Model-specific batching strategies**
- **Batch queue management**
- **Optimized throughput for supported operations**

### Implementation

```python
def __init__(self, resources=None, metadata=None):
    # Batching configuration
    self.enable_batching = True
    self.max_batch_size = 8
    self.batch_timeout = 0.1  # seconds to wait for batch completion
    self.current_batch = {
        "requests": [],
        "created_at": None
    }
    self.batch_lock = threading.RLock()
```

### Batch Processing

```python
def _process_batch(self, batch):
    """Process a batch of requests together"""
    if not batch["requests"]:
        return
        
    # Extract inputs and futures
    inputs = [req["input"] for req in batch["requests"]]
    futures = [req["future"] for req in batch["requests"]]
    
    try:
        # Make the batched API call
        results = self._batch_api_call(inputs)
        
        # Set results for individual futures
        for i, future in enumerate(futures):
            if i < len(results):
                future.set_result(results[i])
            else:
                future.set_exception(Exception("Batch processing error: missing result"))
    except Exception as e:
        # Set the same exception for all futures in the batch
        for future in futures:
            future.set_exception(e)
```

### Batch Management

```python
def _add_to_batch(self, request_input, future):
    """Add a request to the current batch or create a new one"""
    with self.batch_lock:
        # If batch is empty, create a new one
        if not self.current_batch["requests"]:
            self.current_batch = {
                "requests": [],
                "created_at": time.time()
            }
            
        # Add request to batch
        self.current_batch["requests"].append({
            "input": request_input,
            "future": future
        })
        
        # Check if we should process the batch
        should_process = (
            len(self.current_batch["requests"]) >= self.max_batch_size or
            (time.time() - self.current_batch["created_at"] >= self.batch_timeout and
             len(self.current_batch["requests"]) > 0)
        )
        
        if should_process:
            batch_to_process = self.current_batch
            self.current_batch = {
                "requests": [],
                "created_at": None
            }
            return batch_to_process
            
        return None
```

## API Key Multiplexing

Support for using multiple API keys across services with intelligent routing.

### Key Features

- **Multiple API key management** for each provider
- **Per-key client instances** with separate queues
- **Intelligent routing strategies**:
  - Round-robin
  - Least-loaded
  - Specific key selection
- **Real-time usage statistics**
- **Automatic failover between keys**

### Implementation

```python
class ApiKeyMultiplexer:
    def __init__(self):
        # Initialize client dictionaries for each API type
        self.openai_clients = {}
        self.groq_clients = {}
        self.claude_clients = {}
        self.gemini_clients = {}
        
        # Initialize locks for thread safety
        self.openai_lock = threading.RLock()
        self.groq_lock = threading.RLock()
        self.claude_lock = threading.RLock()
        self.gemini_lock = threading.RLock()
```

### API Key Registration

```python
def add_openai_key(self, key_name, api_key, max_concurrent=5):
    """Add a new OpenAI API key with its own client instance"""
    with self.openai_lock:
        client = openai_api(
            resources={},
            metadata={"openai_api_key": api_key}
        )
        
        # Configure client settings
        client.max_concurrent_requests = max_concurrent
        
        # Store client in dictionary
        self.openai_clients[key_name] = {
            "client": client,
            "api_key": api_key,
            "usage": 0,
            "last_used": 0
        }
```

### Client Selection

```python
def get_openai_client(self, key_name=None, strategy="round-robin"):
    """Get an OpenAI client using the specified strategy"""
    with self.openai_lock:
        if len(self.openai_clients) == 0:
            raise ValueError("No OpenAI API keys registered")
            
        # Use specified key if provided
        if key_name and key_name in self.openai_clients:
            selected_key = key_name
        elif strategy == "round-robin":
            # Find least recently used client
            selected_key = min(
                self.openai_clients.keys(),
                key=lambda k: self.openai_clients[k]["last_used"]
            )
        elif strategy == "least-loaded":
            # Find client with smallest queue
            selected_key = min(
                self.openai_clients.keys(),
                key=lambda k: self.openai_clients[k]["client"].active_requests
            )
        else:
            # Default to first key
            selected_key = list(self.openai_clients.keys())[0]
            
        # Update usage statistics
        self.openai_clients[selected_key]["usage"] += 1
        self.openai_clients[selected_key]["last_used"] = time.time()
        
        return self.openai_clients[selected_key]["client"]
```

## Configuration Reference

All advanced features can be configured through the API client initialization or by directly setting attributes.

### Queue Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_concurrent_requests` | 5 | Maximum concurrent requests handled by a client |
| `queue_size` | 100 | Maximum queue size before rejecting new requests |
| `queue_processing` | True | Flag to enable/disable queue processing |

### Circuit Breaker Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `failure_threshold` | 5 | Number of consecutive failures before opening circuit |
| `reset_timeout` | 30 | Seconds to wait before attempting to half-open circuit |
| `circuit_state` | "CLOSED" | Current circuit state (CLOSED, OPEN, HALF-OPEN) |

### Monitoring Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metrics_enabled` | True | Enable/disable metrics collection |
| `detailed_latency` | False | Store detailed latency for each request |
| `error_tracking` | True | Track error types and frequencies |

### Batching Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_batching` | True | Enable/disable request batching |
| `max_batch_size` | 8 | Maximum number of requests in a batch |
| `batch_timeout` | 0.1 | Seconds to wait for batch completion |

### Key Multiplexing Configuration

| Parameter | Description |
|-----------|-------------|
| `key_name` | Unique identifier for the API key |
| `max_concurrent` | Maximum concurrent requests for this key |
| `strategy` | Key selection strategy (round-robin, least-loaded, specific) |

## Integration Patterns

### Basic Usage with All Features

```python
from ipfs_accelerate import accelerate

# Initialize with all advanced features
client = accelerate(
    # Queue configuration
    max_concurrent_requests=10,
    queue_size=200,
    
    # Circuit breaker configuration
    failure_threshold=3,
    reset_timeout=60,
    
    # Batching configuration
    enable_batching=True,
    max_batch_size=16,
    
    # Monitoring
    metrics_enabled=True,
    detailed_latency=True
)

# Get an OpenAI endpoint with advanced features enabled
openai_client = client.get_api_client("openai")

# Make a request with priority
response = openai_client.generate_text(
    "Generate a creative story about AI",
    priority=0,  # High priority
    model="gpt-3.5-turbo"
)

# Check metrics
metrics = openai_client.get_metrics_report()
print(f"Success rate: {metrics['success_rate']:.2f}")
```

### API Key Multiplexing Pattern

```python
from ipfs_accelerate import api_key_multiplexer

# Initialize multiplexer
multiplexer = api_key_multiplexer()

# Add multiple keys
multiplexer.add_openai_key("production", "sk-prod-key", max_concurrent=20)
multiplexer.add_openai_key("testing", "sk-test-key", max_concurrent=5)
multiplexer.add_openai_key("backup", "sk-backup-key", max_concurrent=10)

# Get clients with different strategies
client1 = multiplexer.get_openai_client(strategy="round-robin")
client2 = multiplexer.get_openai_client(strategy="least-loaded")
client3 = multiplexer.get_openai_client(key_name="production")

# Check usage across all keys
stats = multiplexer.get_usage_stats()
print(f"Production key usage: {stats['openai']['production']['usage']}")
```

### High-Performance Configuration

For high-throughput scenarios:

```python
# High performance configuration
client = accelerate(
    max_concurrent_requests=50,
    queue_size=1000,
    enable_batching=True,
    max_batch_size=32,
    batch_timeout=0.05,
    metrics_enabled=True
)
```

### High-Reliability Configuration

For maximum reliability:

```python
# High reliability configuration
client = accelerate(
    max_concurrent_requests=8,
    queue_size=100,
    failure_threshold=2,
    reset_timeout=120,
    enable_batching=False,
    detailed_latency=True,
    error_tracking=True
)
```