# Monitoring and Reporting Guide

This comprehensive guide covers the monitoring and reporting capabilities in the IPFS Accelerate Python Framework. These features provide detailed insights into API performance, error rates, and system health across all 11 supported API backends.

## Table of Contents

1. [Overview](#overview)
2. [Metrics Collection System](#metrics-collection-system)
3. [Request Tracing](#request-tracing)
4. [Error Classification and Tracking](#error-classification-and-tracking)
5. [Performance Metrics](#performance-metrics)
6. [Queue and Circuit Breaker Metrics](#queue-and-circuit-breaker-metrics)
7. [Visualization and Reporting](#visualization-and-reporting)
8. [Integration with External Monitoring](#integration-with-external-monitoring)
9. [Best Practices](#best-practices)

## Overview

The monitoring system provides comprehensive visibility into the operation of all API backends with the following capabilities:

- Real-time metrics collection for all API requests
- Detailed error tracking and classification
- Performance metrics by model and endpoint
- Queue and circuit breaker state monitoring
- Integration capabilities with external monitoring tools

All metrics are collected with minimal overhead and provide actionable insights into API performance and reliability.

## Metrics Collection System

### Core Metrics Structure

Each API backend maintains a metrics dictionary with the following structure:

```python
self.metrics = {
    "requests": 0,             # Total requests processed
    "successes": 0,            # Successful requests
    "failures": 0,             # Failed requests
    "timeouts": 0,             # Requests that timed out
    "retries": 0,              # Retried requests
    "latency": [],             # List of request latencies (ms)
    "token_counts": {          # Token counts for text generation
        "prompt": 0,
        "completion": 0
    },
    "error_types": {},         # Count of errors by type
    "models": {},              # Per-model metrics
    "timestamps": [],          # Request timestamps
    "queue_metrics": {         # Queue metrics
        "queue_time": [],      # Time spent in queue
        "queue_length": []     # Queue length at request time
    },
    "circuit_breaker": {       # Circuit breaker metrics
        "state_changes": [],   # Circuit state changes
        "open_time": 0         # Total time in OPEN state
    }
}
```

### Implementation

The metrics collection is implemented with thread-safe updates:

```python
def _update_metrics(self, success=True, latency=None, error=None, 
                   retried=False, model=None, prompt_tokens=0, 
                   completion_tokens=0):
    """Update metrics after a request completes"""
    with self.metrics_lock:
        # Basic counters
        self.metrics["requests"] += 1
        if success:
            self.metrics["successes"] += 1
        else:
            self.metrics["failures"] += 1
            
        # Latency tracking
        if latency is not None:
            self.metrics["latency"].append(latency)
            
        # Retry tracking
        if retried:
            self.metrics["retries"] += 1
            
        # Token counting
        self.metrics["token_counts"]["prompt"] += prompt_tokens
        self.metrics["token_counts"]["completion"] += completion_tokens
            
        # Error tracking
        if error is not None:
            error_type = type(error).__name__
            if error_type not in self.metrics["error_types"]:
                self.metrics["error_types"][error_type] = 0
            self.metrics["error_types"][error_type] += 1
            
        # Per-model tracking
        if model:
            if model not in self.metrics["models"]:
                self.metrics["models"][model] = {
                    "requests": 0,
                    "successes": 0,
                    "failures": 0,
                    "latency": []
                }
            self.metrics["models"][model]["requests"] += 1
            if success:
                self.metrics["models"][model]["successes"] += 1
            else:
                self.metrics["models"][model]["failures"] += 1
            if latency is not None:
                self.metrics["models"][model]["latency"].append(latency)
                
        # Timestamp tracking
        self.metrics["timestamps"].append(time.time())
```

### Usage

Metrics are automatically updated during API requests and can be accessed through the reporting interface:

```python
# Get overall metrics report
metrics_report = api_client.get_metrics_report()

# Get model-specific metrics
model_metrics = api_client.get_model_metrics("gpt-3.5-turbo")

# Get error breakdown
error_metrics = api_client.get_error_metrics()
```

## Request Tracing

All requests can be traced through the system with unique IDs.

### Trace Implementation

```python
def _generate_trace_id(self):
    """Generate a unique trace ID for request tracking"""
    return f"{uuid.uuid4()}"

def _process_with_tracing(self, func, *args, **kwargs):
    """Process a request with tracing"""
    trace_id = self._generate_trace_id()
    start_time = time.time()
    
    # Add trace to request context
    trace_context = {
        "trace_id": trace_id,
        "start_time": start_time,
        "queue_time": self._get_queue_time(trace_id)
    }
    
    if "context" not in kwargs:
        kwargs["context"] = {}
    kwargs["context"]["trace"] = trace_context
    
    # Execute the request
    try:
        result = func(*args, **kwargs)
        success = True
        error = None
    except Exception as e:
        success = False
        error = e
        raise
    finally:
        # Update metrics with trace information
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # ms
        
        self._update_metrics(
            success=success,
            latency=latency,
            error=error,
            model=kwargs.get("model"),
            trace_id=trace_id
        )
        
        # Store trace information if enabled
        if self.request_tracing:
            self._store_trace(trace_id, {
                "success": success,
                "latency": latency,
                "error": str(error) if error else None,
                "start_time": start_time,
                "end_time": end_time,
                "model": kwargs.get("model"),
                "queue_time": trace_context.get("queue_time")
            })
```

### Accessing Trace Data

```python
# Get trace information for a specific request
trace_data = api_client.get_trace("trace-uuid-here")

# Get recent traces
recent_traces = api_client.get_recent_traces(limit=10)
```

## Error Classification and Tracking

Errors are automatically classified and tracked for better troubleshooting.

### Error Categories

Errors are grouped into the following categories:

1. **Rate Limiting**: Errors related to API rate limits
2. **Authentication**: Errors with API keys or authentication
3. **Invalid Request**: Malformed requests or invalid parameters
4. **Service Unavailable**: Backend service unavailability
5. **Timeout**: Request timeouts
6. **Network**: Connection and network-related errors
7. **Unexpected**: Other unexpected errors

### Error Classification Implementation

```python
def _classify_error(self, error):
    """Classify an error into one of the standard categories"""
    error_type = type(error).__name__
    error_msg = str(error).lower()
    
    # Rate limiting errors
    if "rate limit" in error_msg or "too many requests" in error_msg:
        return "rate_limiting"
        
    # Authentication errors
    if "authentication" in error_msg or "unauthorized" in error_msg or "api key" in error_msg:
        return "authentication"
        
    # Invalid request errors
    if "invalid" in error_msg or "bad request" in error_msg:
        return "invalid_request"
        
    # Service unavailable errors
    if "unavailable" in error_msg or "server error" in error_msg:
        return "service_unavailable"
        
    # Timeout errors
    if "timeout" in error_msg or error_type == "TimeoutError":
        return "timeout"
        
    # Network errors
    if error_type in ["ConnectionError", "ConnectionRefusedError", "ConnectionResetError"]:
        return "network"
        
    # Default category
    return "unexpected"
```

### Error Metrics Reporting

```python
def get_error_metrics(self):
    """Get detailed error metrics"""
    with self.metrics_lock:
        # Calculate error rate
        total_requests = self.metrics["requests"]
        error_rate = self.metrics["failures"] / total_requests if total_requests > 0 else 0
        
        # Group errors by category
        categorized_errors = {}
        for error_type, count in self.metrics["error_types"].items():
            category = self._get_error_category(error_type)
            if category not in categorized_errors:
                categorized_errors[category] = 0
            categorized_errors[category] += count
            
        return {
            "error_rate": error_rate,
            "total_errors": self.metrics["failures"],
            "error_types": self.metrics["error_types"],
            "error_categories": categorized_errors,
            "most_common_error": max(self.metrics["error_types"].items(), 
                                    key=lambda x: x[1])[0] if self.metrics["error_types"] else None
        }
```

## Performance Metrics

Detailed performance metrics are collected for all API requests.

### Latency Metrics

```python
def get_latency_metrics(self):
    """Get detailed latency metrics"""
    with self.metrics_lock:
        latency_data = self.metrics["latency"]
        
        if not latency_data:
            return {
                "avg_latency": 0,
                "min_latency": 0,
                "max_latency": 0,
                "percentiles": {
                    "50": 0,
                    "90": 0,
                    "95": 0,
                    "99": 0
                }
            }
            
        latency_data.sort()
        
        return {
            "avg_latency": sum(latency_data) / len(latency_data),
            "min_latency": latency_data[0],
            "max_latency": latency_data[-1],
            "percentiles": {
                "50": latency_data[int(len(latency_data) * 0.5)],
                "90": latency_data[int(len(latency_data) * 0.9)],
                "95": latency_data[int(len(latency_data) * 0.95)],
                "99": latency_data[int(len(latency_data) * 0.99)]
            }
        }
```

### Throughput Metrics

```python
def get_throughput_metrics(self, time_window=60):
    """Get throughput metrics for a specific time window (in seconds)"""
    with self.metrics_lock:
        current_time = time.time()
        # Filter timestamps within the window
        recent_timestamps = [ts for ts in self.metrics["timestamps"] 
                            if current_time - ts <= time_window]
        
        # Calculate requests per second
        if recent_timestamps:
            requests_in_window = len(recent_timestamps)
            rps = requests_in_window / time_window
        else:
            rps = 0
            
        # Calculate token throughput if available
        token_throughput = {}
        if "token_counts" in self.metrics:
            prompt_tokens = self.metrics["token_counts"]["prompt"]
            completion_tokens = self.metrics["token_counts"]["completion"]
            total_tokens = prompt_tokens + completion_tokens
            
            if self.metrics["timestamps"]:
                elapsed_time = current_time - self.metrics["timestamps"][0]
                if elapsed_time > 0:
                    token_throughput = {
                        "tokens_per_second": total_tokens / elapsed_time,
                        "prompt_tokens_per_second": prompt_tokens / elapsed_time,
                        "completion_tokens_per_second": completion_tokens / elapsed_time
                    }
            
        return {
            "requests_per_second": rps,
            "total_requests_in_window": len(recent_timestamps),
            "time_window_seconds": time_window,
            "token_throughput": token_throughput
        }
```

### Model-Specific Metrics

```python
def get_model_metrics(self, model_name):
    """Get metrics for a specific model"""
    with self.metrics_lock:
        if model_name not in self.metrics["models"]:
            return {
                "requests": 0,
                "success_rate": 0,
                "avg_latency": 0
            }
            
        model_data = self.metrics["models"][model_name]
        total_requests = model_data["requests"]
        
        return {
            "requests": total_requests,
            "successes": model_data["successes"],
            "failures": model_data["failures"],
            "success_rate": model_data["successes"] / total_requests if total_requests > 0 else 0,
            "avg_latency": sum(model_data["latency"]) / len(model_data["latency"]) 
                          if model_data["latency"] else 0
        }
```

## Queue and Circuit Breaker Metrics

Metrics are also collected for the queue system and circuit breaker.

### Queue Metrics

```python
def get_queue_metrics(self):
    """Get metrics for the request queue"""
    with self.metrics_lock:
        queue_times = self.metrics["queue_metrics"]["queue_time"]
        queue_lengths = self.metrics["queue_metrics"]["queue_length"]
        
        avg_queue_time = sum(queue_times) / len(queue_times) if queue_times else 0
        avg_queue_length = sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0
        
        return {
            "current_queue_size": len(self.request_queue),
            "active_requests": self.active_requests,
            "avg_queue_time_ms": avg_queue_time,
            "avg_queue_length": avg_queue_length,
            "max_queue_length": max(queue_lengths) if queue_lengths else 0,
            "queue_rejection_count": self.metrics.get("queue_rejections", 0)
        }
```

### Circuit Breaker Metrics

```python
def get_circuit_breaker_metrics(self):
    """Get metrics for the circuit breaker"""
    with self.metrics_lock:
        state_changes = self.metrics["circuit_breaker"]["state_changes"]
        current_time = time.time()
        
        # Calculate time in each state
        state_time = {"CLOSED": 0, "OPEN": 0, "HALF-OPEN": 0}
        prev_state = "CLOSED"
        prev_time = self.metrics["timestamps"][0] if self.metrics["timestamps"] else current_time
        
        for state, timestamp in state_changes:
            state_time[prev_state] += timestamp - prev_time
            prev_state = state
            prev_time = timestamp
            
        # Add time since last state change
        state_time[prev_state] += current_time - prev_time
        
        return {
            "current_state": self.circuit_state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "open_circuit_count": sum(1 for state, _ in state_changes if state == "OPEN"),
            "state_time_seconds": state_time,
            "uptime_percentage": state_time["CLOSED"] / sum(state_time.values()) * 100 
                                if sum(state_time.values()) > 0 else 100
        }
```

## Visualization and Reporting

The metrics can be visualized and reported in various formats.

### Comprehensive Metrics Report

```python
def get_metrics_report(self):
    """Get a comprehensive metrics report"""
    with self.metrics_lock:
        # Basic metrics
        total_requests = self.metrics["requests"]
        success_rate = self.metrics["successes"] / total_requests if total_requests > 0 else 0
        
        # Latency metrics
        latency_metrics = self.get_latency_metrics()
        
        # Queue metrics
        queue_metrics = self.get_queue_metrics()
        
        # Circuit breaker metrics
        circuit_metrics = self.get_circuit_breaker_metrics()
        
        # Error metrics
        error_metrics = self.get_error_metrics()
        
        # Throughput metrics
        throughput = self.get_throughput_metrics()
        
        # Model metrics
        model_stats = {}
        for model in self.metrics["models"]:
            model_stats[model] = self.get_model_metrics(model)
            
        return {
            "timestamp": time.time(),
            "api_name": self.__class__.__name__,
            "uptime_seconds": time.time() - self.start_time,
            "total_requests": total_requests,
            "success_rate": success_rate,
            "latency": latency_metrics,
            "queue": queue_metrics,
            "circuit_breaker": circuit_metrics,
            "errors": error_metrics,
            "throughput": throughput,
            "models": model_stats
        }
```

### Report Visualization

The metrics report can be converted to various visualization formats:

```python
def get_ascii_report(self):
    """Generate an ASCII text report of metrics"""
    report = self.get_metrics_report()
    
    lines = [
        f"=== {report['api_name']} Metrics Report ===",
        f"Timestamp: {datetime.fromtimestamp(report['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}",
        f"Uptime: {report['uptime_seconds']:.2f} seconds",
        "",
        f"Total Requests: {report['total_requests']}",
        f"Success Rate: {report['success_rate'] * 100:.2f}%",
        f"Current Circuit State: {report['circuit_breaker']['current_state']}",
        f"Current Queue Size: {report['queue']['current_queue_size']}",
        "",
        "Latency (ms):",
        f"  Avg: {report['latency']['avg_latency']:.2f}",
        f"  P50: {report['latency']['percentiles']['50']:.2f}",
        f"  P95: {report['latency']['percentiles']['95']:.2f}",
        f"  P99: {report['latency']['percentiles']['99']:.2f}",
        "",
        "Throughput:",
        f"  Requests/sec: {report['throughput']['requests_per_second']:.2f}",
        "",
        "Errors:",
        f"  Error Rate: {report['errors']['error_rate'] * 100:.2f}%",
        f"  Most Common: {report['errors']['most_common_error'] or 'None'}"
    ]
    
    # Add model statistics
    if report['models']:
        lines.append("")
        lines.append("Model Statistics:")
        for model, stats in report['models'].items():
            lines.append(f"  {model}:")
            lines.append(f"    Requests: {stats['requests']}")
            lines.append(f"    Success Rate: {stats['success_rate'] * 100:.2f}%")
            lines.append(f"    Avg Latency: {stats['avg_latency']:.2f} ms")
    
    return "\n".join(lines)
```

### JSON Report

```python
def get_json_report(self):
    """Get metrics report as JSON string"""
    import json
    report = self.get_metrics_report()
    return json.dumps(report, indent=2)
```

## Integration with External Monitoring

The monitoring system can integrate with external monitoring tools.

### Prometheus Integration

```python
def get_prometheus_metrics(self):
    """Generate Prometheus-compatible metrics"""
    report = self.get_metrics_report()
    
    lines = [
        f'# HELP api_requests_total Total number of API requests',
        f'# TYPE api_requests_total counter',
        f'api_requests_total{{api="{report["api_name"]}"}} {report["total_requests"]}',
        
        f'# HELP api_success_rate Success rate of API requests',
        f'# TYPE api_success_rate gauge',
        f'api_success_rate{{api="{report["api_name"]}"}} {report["success_rate"]}',
        
        f'# HELP api_latency_milliseconds API request latency in milliseconds',
        f'# TYPE api_latency_milliseconds gauge',
        f'api_latency_milliseconds{{api="{report["api_name"]}",percentile="50"}} {report["latency"]["percentiles"]["50"]}',
        f'api_latency_milliseconds{{api="{report["api_name"]}",percentile="95"}} {report["latency"]["percentiles"]["95"]}',
        f'api_latency_milliseconds{{api="{report["api_name"]}",percentile="99"}} {report["latency"]["percentiles"]["99"]}',
        
        f'# HELP api_queue_size Current number of requests in queue',
        f'# TYPE api_queue_size gauge',
        f'api_queue_size{{api="{report["api_name"]}"}} {report["queue"]["current_queue_size"]}',
        
        f'# HELP api_active_requests Current number of active requests',
        f'# TYPE api_active_requests gauge',
        f'api_active_requests{{api="{report["api_name"]}"}} {report["queue"]["active_requests"]}',
        
        f'# HELP api_error_rate Rate of API request errors',
        f'# TYPE api_error_rate gauge',
        f'api_error_rate{{api="{report["api_name"]}"}} {report["errors"]["error_rate"]}',
        
        f'# HELP api_circuit_state Current circuit breaker state (0=CLOSED, 1=HALF-OPEN, 2=OPEN)',
        f'# TYPE api_circuit_state gauge',
        f'api_circuit_state{{api="{report["api_name"]}"}} {{"CLOSED": 0, "HALF-OPEN": 1, "OPEN": 2}[report["circuit_breaker"]["current_state"]]}',
        
        f'# HELP api_throughput_rps Requests per second',
        f'# TYPE api_throughput_rps gauge',
        f'api_throughput_rps{{api="{report["api_name"]}"}} {report["throughput"]["requests_per_second"]}'
    ]
    
    # Add model-specific metrics
    for model, stats in report["models"].items():
        lines.extend([
            f'# HELP api_model_requests_total Total requests for a specific model',
            f'# TYPE api_model_requests_total counter',
            f'api_model_requests_total{{api="{report["api_name"]}",model="{model}"}} {stats["requests"]}',
            
            f'# HELP api_model_success_rate Success rate for a specific model',
            f'# TYPE api_model_success_rate gauge',
            f'api_model_success_rate{{api="{report["api_name"]}",model="{model}"}} {stats["success_rate"]}',
            
            f'# HELP api_model_latency_milliseconds Average latency for a specific model',
            f'# TYPE api_model_latency_milliseconds gauge',
            f'api_model_latency_milliseconds{{api="{report["api_name"]}",model="{model}"}} {stats["avg_latency"]}'
        ])
    
    return "\n".join(lines)
```

### Grafana Dashboard Configuration

A sample Grafana dashboard configuration is provided for visualizing the metrics.

```python
def export_grafana_dashboard(self):
    """Export a Grafana dashboard configuration for this API"""
    # Implementation depends on specific Grafana version and needs
    # but would include panels for:
    # - Request rate and success rate
    # - Latency with percentiles
    # - Queue metrics
    # - Circuit breaker state
    # - Error rates by category
    # - Per-model statistics
    pass
```

## Best Practices

### Configuring Metrics Collection

For optimal performance, configure the metrics collection based on your needs:

```python
# High-detail configuration for debugging
client = api_client(
    metrics_enabled=True,
    detailed_latency=True,
    error_tracking=True,
    request_tracing=True,
    metrics_window_size=10000
)

# Low-overhead configuration for production
client = api_client(
    metrics_enabled=True,
    detailed_latency=False,
    error_tracking=True,
    request_tracing=False,
    metrics_window_size=1000,
    metrics_reset_interval=3600.0  # Reset hourly
)
```

### Regular Metrics Reset

For long-running applications, consider resetting metrics periodically:

```python
# Reset metrics every 24 hours
def reset_metrics_daily(client):
    while True:
        time.sleep(24 * 60 * 60)  # 24 hours
        client.reset_metrics()
        
threading.Thread(target=reset_metrics_daily, args=(api_client,), daemon=True).start()
```

### Monitoring Integration

For production systems, integrate with your monitoring stack:

```python
# Export Prometheus metrics on HTTP endpoint
def metrics_endpoint(client):
    from http.server import HTTPServer, BaseHTTPRequestHandler
    
    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(client.get_prometheus_metrics().encode())
    
    server = HTTPServer(('0.0.0.0', 8000), MetricsHandler)
    server.serve_forever()
    
threading.Thread(target=metrics_endpoint, args=(api_client,), daemon=True).start()
```

### Alert Configuration

Set up alerts based on specific metrics thresholds:

```python
def check_alerts(client, thresholds):
    while True:
        metrics = client.get_metrics_report()
        
        # Check error rate threshold
        if metrics['errors']['error_rate'] > thresholds['error_rate']:
            send_alert(f"High error rate: {metrics['errors']['error_rate']:.2%}")
            
        # Check latency threshold
        if metrics['latency']['percentiles']['95'] > thresholds['p95_latency']:
            send_alert(f"High P95 latency: {metrics['latency']['percentiles']['95']:.2f}ms")
            
        # Check queue size threshold
        if metrics['queue']['current_queue_size'] > thresholds['queue_size']:
            send_alert(f"Large queue size: {metrics['queue']['current_queue_size']}")
            
        time.sleep(60)  # Check every minute

thresholds = {
    'error_rate': 0.05,    # 5% error rate
    'p95_latency': 1000,   # 1000ms P95 latency
    'queue_size': 50       # 50 requests in queue
}

threading.Thread(target=check_alerts, args=(api_client, thresholds), daemon=True).start()
```