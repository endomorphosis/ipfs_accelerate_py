# S3 Kit Connection Multiplexing Guide

## Overview

The S3 Kit API in the IPFS Accelerate Python framework now supports advanced connection multiplexing, allowing applications to work with multiple S3-compatible storage endpoints simultaneously. This guide explains this feature and provides examples of its use.

## Key Features

- **Multiple Independent Endpoints**: Create separate handlers for AWS S3, MinIO, Ceph, and other S3-compatible services
- **Per-Endpoint Configuration**: Each endpoint has its own credentials, circuit breaker settings, and retry policy
- **Smart Routing Strategies**: Round-robin and least-loaded routing for optimal performance
- **Failover Support**: Automatic fallback to healthy endpoints when one endpoint fails
- **Thread-safe Implementation**: Safe for multi-threaded applications

## Basic Setup

```python
from ipfs_accelerate_py.api_backends import s3_kit
import threading
import time

# Initialize base S3 Kit instance
s3_kit_api = s3_kit(
    metadata={
        "s3cfg": {
            "accessKey": "default_access_key",  # Default fallback credentials
            "secretKey": "default_secret_key",
            "endpoint": "https://s3.amazonaws.com"
        }
    }
)

# Create the connection multiplexer
class S3EndpointMultiplexer:
    def __init__(self, s3_kit_instance):
        self.s3_kit = s3_kit_instance
        self.endpoint_handlers = {}
        self.endpoints_lock = threading.RLock()
        self.last_used = {}
        self.requests_per_endpoint = {}
        
    def add_endpoint(self, name, endpoint_url, access_key, secret_key, max_concurrent=5, 
                     circuit_breaker_threshold=5, retries=3):
        """Add a new S3 endpoint with its own configuration"""
        with self.endpoints_lock:
            handler = self.s3_kit.create_s3_endpoint_handler(
                endpoint_url=endpoint_url,
                access_key=access_key,
                secret_key=secret_key,
                max_concurrent=max_concurrent,
                circuit_breaker_threshold=circuit_breaker_threshold,
                retries=retries
            )
            self.endpoint_handlers[name] = handler
            self.last_used[name] = 0
            self.requests_per_endpoint[name] = 0
            return handler
    
    def get_endpoint(self, name=None, strategy="round-robin"):
        """Get an endpoint by name or using a selection strategy"""
        with self.endpoints_lock:
            if not self.endpoint_handlers:
                raise ValueError("No S3 endpoints have been added")
                
            # Return specific endpoint if requested
            if name and name in self.endpoint_handlers:
                self.last_used[name] = time.time()
                self.requests_per_endpoint[name] += 1
                return self.endpoint_handlers[name]
                
            # Apply selection strategy
            if strategy == "round-robin":
                # Choose least recently used endpoint
                selected = min(self.last_used.items(), key=lambda x: x[1])[0]
            elif strategy == "least-loaded":
                # Choose endpoint with fewest requests
                selected = min(self.requests_per_endpoint.items(), key=lambda x: x[1])[0]
            else:
                # Default to first endpoint
                selected = next(iter(self.endpoint_handlers.keys()))
                
            self.last_used[selected] = time.time()
            self.requests_per_endpoint[selected] += 1
            return self.endpoint_handlers[selected]
            
    # Convenience methods for common operations
    def upload_file(self, file_path, bucket, key, endpoint_name=None, strategy="round-robin"):
        handler = self.get_endpoint(endpoint_name, strategy)
        return handler("upload_file", file_path=file_path, bucket=bucket, key=key)
        
    def download_file(self, bucket, key, file_path, endpoint_name=None, strategy="round-robin"):
        handler = self.get_endpoint(endpoint_name, strategy)
        return handler("download_file", bucket=bucket, key=key, file_path=file_path)
        
    def list_objects(self, bucket, prefix=None, endpoint_name=None, strategy="round-robin"):
        handler = self.get_endpoint(endpoint_name, strategy)
        return handler("list_objects", bucket=bucket, prefix=prefix)
        
    def delete_object(self, bucket, key, endpoint_name=None, strategy="round-robin"):
        handler = self.get_endpoint(endpoint_name, strategy)
        return handler("delete_object", bucket=bucket, key=key)
        
    def head_object(self, bucket, key, endpoint_name=None, strategy="round-robin"):
        handler = self.get_endpoint(endpoint_name, strategy)
        return handler("head_object", bucket=bucket, key=key)
        
    def create_presigned_url(self, bucket, key, expiration=3600, endpoint_name=None, strategy="round-robin"):
        handler = self.get_endpoint(endpoint_name, strategy)
        return handler("create_presigned_url", bucket=bucket, key=key, expiration=expiration)
```

## Usage Examples

### Setting Up Multiple Endpoints

```python
# Create multiplexer instance
multiplexer = S3EndpointMultiplexer(s3_kit_api)

# Add AWS S3 endpoint
multiplexer.add_endpoint(
    name="aws-primary",
    endpoint_url="https://s3.us-east-1.amazonaws.com",
    access_key=os.environ.get("AWS_ACCESS_KEY"),
    secret_key=os.environ.get("AWS_SECRET_KEY"),
    max_concurrent=10
)

# Add AWS S3 secondary region
multiplexer.add_endpoint(
    name="aws-secondary",
    endpoint_url="https://s3.us-west-2.amazonaws.com",
    access_key=os.environ.get("AWS_ACCESS_KEY"),
    secret_key=os.environ.get("AWS_SECRET_KEY"),
    max_concurrent=10
)

# Add MinIO endpoint
multiplexer.add_endpoint(
    name="minio-local",
    endpoint_url="http://localhost:9000",
    access_key=os.environ.get("MINIO_ACCESS_KEY"),
    secret_key=os.environ.get("MINIO_SECRET_KEY"),
    max_concurrent=20,  # Higher concurrency for local endpoint
    circuit_breaker_threshold=10  # More lenient threshold for local endpoint
)

# Add Ceph endpoint
multiplexer.add_endpoint(
    name="ceph-storage",
    endpoint_url="https://ceph.example.com:7480",
    access_key=os.environ.get("CEPH_ACCESS_KEY"),
    secret_key=os.environ.get("CEPH_SECRET_KEY"),
    retries=5  # More retries for potentially less stable storage
)
```

### Using Routing Strategies

#### Round-Robin Load Balancing

Distributes requests evenly across all endpoints to balance load:

```python
# Upload files in round-robin fashion across all endpoints
for i in range(10):
    result = multiplexer.upload_file(
        file_path=f"file_{i}.txt",
        bucket="shared-bucket",
        key=f"data/file_{i}.txt",
        strategy="round-robin"
    )
```

#### Least-Loaded Endpoint Selection

Routes requests to the endpoint with the fewest active requests:

```python
# Use least-loaded strategy during high-traffic periods
for large_file in large_files:
    result = multiplexer.upload_file(
        file_path=large_file,
        bucket="media-bucket",
        key=f"videos/{os.path.basename(large_file)}",
        strategy="least-loaded"
    )
```

#### Direct Endpoint Selection

Send request to a specific endpoint when required:

```python
# Use local MinIO for development data
result = multiplexer.list_objects(
    bucket="dev-bucket",
    prefix="test-data/",
    endpoint_name="minio-local"
)

# Use AWS for production data
result = multiplexer.download_file(
    bucket="prod-bucket",
    key="reports/monthly.pdf",
    file_path="monthly_report.pdf",
    endpoint_name="aws-primary"
)
```

### Advanced Use Cases

#### Multi-Region Replication

Replicate files across multiple regions for redundancy:

```python
def replicate_to_all_endpoints(local_file, remote_key):
    """Upload a file to all configured endpoints for redundancy"""
    results = {}
    for endpoint_name in multiplexer.endpoint_handlers.keys():
        try:
            result = multiplexer.upload_file(
                file_path=local_file,
                bucket="backup-bucket",
                key=remote_key,
                endpoint_name=endpoint_name
            )
            results[endpoint_name] = {"success": True, "result": result}
        except Exception as e:
            results[endpoint_name] = {"success": False, "error": str(e)}
    return results

# Replicate important file to all endpoints
replication_results = replicate_to_all_endpoints(
    "critical_data.db", 
    "backups/critical_data_20250301.db"
)
```

#### Endpoint Health Checking

Implement health checks to prioritize healthy endpoints:

```python
def check_endpoint_health():
    """Check health of all endpoints and return only healthy ones"""
    healthy_endpoints = []
    for name in multiplexer.endpoint_handlers.keys():
        try:
            # Test endpoint with a simple list operation
            multiplexer.list_objects(
                bucket="health-check-bucket",
                endpoint_name=name
            )
            healthy_endpoints.append(name)
        except Exception as e:
            print(f"Endpoint {name} failed health check: {e}")
    return healthy_endpoints

# Get healthy endpoints and prefer them for routing
healthy_endpoints = check_endpoint_health()
if healthy_endpoints:
    # Choose a random healthy endpoint
    import random
    endpoint = random.choice(healthy_endpoints)
    result = multiplexer.download_file(
        bucket="important-bucket",
        key="critical-file.txt",
        file_path="local-copy.txt",
        endpoint_name=endpoint
    )
else:
    # Fall back to least-loaded strategy if no confirmed healthy endpoints
    result = multiplexer.download_file(
        bucket="important-bucket",
        key="critical-file.txt",
        file_path="local-copy.txt",
        strategy="least-loaded"
    )
```

### Error Handling

The circuit breaker pattern operates independently for each endpoint, so one failing endpoint won't affect others:

```python
try:
    # This will automatically use a different endpoint if the preferred one has its circuit breaker open
    result = multiplexer.download_file(
        bucket="shared-bucket",
        key="important-file.txt",
        file_path="local-copy.txt",
        endpoint_name="aws-primary"  # If this endpoint's circuit is open, will raise an exception
    )
except Exception as e:
    if "Circuit breaker is open" in str(e):
        # Fall back to secondary endpoint
        result = multiplexer.download_file(
            bucket="shared-bucket",
            key="important-file.txt",
            file_path="local-copy.txt",
            endpoint_name="aws-secondary"
        )
```

## Performance Considerations

- **Round-Robin**: Best for uniform workloads with similar endpoint performance
- **Least-Loaded**: Best for mixed workloads or endpoints with varying performance
- **Specific Endpoint**: Best when you need data locality or specific endpoint features

## Thread Safety

The S3 Kit multiplexer is fully thread-safe and can be used in multi-threaded applications:

```python
import concurrent.futures

# Concurrent operations across multiple endpoints
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    files_to_upload = [f"file_{i}.txt" for i in range(100)]
    
    for file_path in files_to_upload:
        # Submit each upload task
        future = executor.submit(
            multiplexer.upload_file,
            file_path=file_path,
            bucket="upload-bucket",
            key=f"batch/{file_path}",
            strategy="least-loaded"
        )
        futures.append(future)
    
    # Wait for all results
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            print(f"Upload successful: {result}")
        except Exception as e:
            print(f"Upload failed: {e}")
```

## Monitoring

To monitor endpoint usage statistics:

```python
def print_endpoint_stats():
    """Print usage statistics for all endpoints"""
    with multiplexer.endpoints_lock:
        print("=== Endpoint Usage Statistics ===")
        for name, count in multiplexer.requests_per_endpoint.items():
            print(f"Endpoint: {name}")
            print(f"  Requests: {count}")
            
            # Get handler metadata
            handler = multiplexer.endpoint_handlers[name]
            
            # Include endpoint info from handler metadata
            if hasattr(handler, "metadata"):
                print(f"  URL: {handler.metadata.get('endpoint_url', 'Unknown')}")
                
            print("---")
```

## Conclusion

The S3 Kit connection multiplexing feature provides a powerful way to work with multiple S3-compatible storage services simultaneously, improving reliability, performance, and flexibility. By using different routing strategies and endpoint-specific configurations, you can optimize your application's storage access patterns for various use cases.