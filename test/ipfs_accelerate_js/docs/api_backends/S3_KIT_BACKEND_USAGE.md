# S3 Kit API Backend

The S3 Kit API backend provides a TypeScript implementation for interacting with S3-compatible object storage services. This backend supports multiple endpoints, connection management, and a circuit breaker pattern for enhanced reliability.

## Overview

- Support for any S3-compatible object storage service (AWS S3, MinIO, Ceph, etc.)
- Multiple endpoint management with automatic failover and load balancing
- Circuit breaker pattern to prevent cascading failures
- Priority-based request queuing with configurable concurrency
- Exponential backoff for retries
- Comprehensive error handling

## Installation

The S3 Kit backend is included in the IPFS Accelerate JavaScript SDK. No additional installation is required.

## Configuration

### API Keys and Endpoints

To use the S3 Kit, you need to provide S3 credentials. You can provide them in any of the following ways:

1. Pass them directly in the metadata when creating the backend instance:
   ```typescript
   import { S3Kit } from 'ipfs_accelerate_js/api_backends';
   
   const s3Kit = new S3Kit({}, {
     s3cfg: {
       endpoint: 'https://s3.amazonaws.com',
       accessKey: 'your-access-key',
       secretKey: 'your-secret-key'
     }
   });
   ```

2. Set them as environment variables:
   ```bash
   # Node.js environment
   export S3_ENDPOINT=https://s3.amazonaws.com
   export S3_ACCESS_KEY=your-access-key
   export S3_SECRET_KEY=your-secret-key
   ```

### Custom Options

The S3 Kit backend supports several configuration options:

```typescript
import { S3Kit } from 'ipfs_accelerate_js/api_backends';

const s3Kit = new S3Kit({
  // Maximum concurrent requests (default: 10)
  max_concurrent_requests: 10,
  
  // Size limit for the request queue (default: 100)
  queue_size: 100,
  
  // Maximum retry attempts for failed requests (default: 3)
  max_retries: 3,
  
  // Initial delay before retrying a failed request in ms (default: 1000)
  initial_retry_delay: 1000,
  
  // Backoff factor for exponential retry delay (default: 2)
  backoff_factor: 2,
  
  // Default timeout for S3 requests in ms (default: 30000)
  default_timeout: 30000,
  
  // Selection strategy for endpoint multiplexing (default: 'round-robin')
  endpoint_selection_strategy: 'round-robin', // or 'least-loaded'
  
  // Circuit breaker configuration
  circuit_breaker: {
    // Threshold for consecutive failures before opening the breaker (default: 3)
    threshold: 3,
    
    // Timeout period in ms before attempting to half-open the breaker (default: 60000)
    timeout: 60000
  }
});
```

## Basic Usage

### Adding Endpoints

```typescript
import { S3Kit } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your S3 credentials
const s3Kit = new S3Kit();

// Add an endpoint with a unique name
const handler = s3Kit.addEndpoint(
  'minio-local',              // Endpoint name
  'http://localhost:9000',    // Endpoint URL
  'minioadmin',               // Access key
  'minioadmin',               // Secret key
  5,                          // Maximum concurrent requests (optional)
  3,                          // Circuit breaker threshold (optional)
  2                           // Maximum retries (optional)
);

// Add another endpoint for a different region or provider
s3Kit.addEndpoint(
  'aws-s3',
  'https://s3.amazonaws.com',
  'your-aws-access-key',
  'your-aws-secret-key'
);

// Get an endpoint by name
const endpoint = s3Kit.getEndpoint('minio-local');

// Get an endpoint using a strategy
const leastLoadedEndpoint = s3Kit.getEndpoint(undefined, 'least-loaded');
```

### Testing Endpoints

```typescript
import { S3Kit } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your S3 credentials
const s3Kit = new S3Kit();

// Test the default endpoint
const isAvailable = await s3Kit.testEndpoint();
console.log(`Default endpoint available: ${isAvailable}`);

// Test a specific endpoint
const customAvailable = await s3Kit.testS3Endpoint(
  'https://custom-endpoint.com',
  'custom-access-key',
  'custom-secret-key'
);
console.log(`Custom endpoint available: ${customAvailable}`);
```

### Uploading Files

```typescript
import { S3Kit } from 'ipfs_accelerate_js/api_backends';
import * as path from 'path';

// Create an instance with your S3 credentials
const s3Kit = new S3Kit();

// Upload a file to a bucket
const result = await s3Kit.uploadFile(
  '/path/to/local/file.txt',  // Local file path
  'my-bucket',                // S3 bucket name
  'path/in/bucket/file.txt',  // S3 object key
  { priority: 'HIGH' }        // Optional request options
);

console.log(`File uploaded: ${result.Location}`);
```

### Downloading Files

```typescript
import { S3Kit } from 'ipfs_accelerate_js/api_backends';
import * as path from 'path';

// Create an instance with your S3 credentials
const s3Kit = new S3Kit();

// Download a file from a bucket
const result = await s3Kit.downloadFile(
  'my-bucket',                // S3 bucket name
  'path/in/bucket/file.txt',  // S3 object key
  '/path/to/save/file.txt',   // Local file path to save
  { priority: 'NORMAL' }      // Optional request options
);

console.log(`File downloaded to: /path/to/save/file.txt`);
```

### Listing Objects

```typescript
import { S3Kit } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your S3 credentials
const s3Kit = new S3Kit();

// List objects in a bucket
const result = await s3Kit.listObjects(
  'my-bucket',                // S3 bucket name
  'path/in/bucket/',          // Optional prefix to filter objects
  {
    max_keys: 100,            // Maximum number of objects to list
    priority: 'LOW'           // Optional request priority
  }
);

console.log(`Found ${result.Contents.length} objects:`);
result.Contents.forEach(object => {
  console.log(`- ${object.Key} (${object.Size} bytes)`);
});
```

### Deleting Objects

```typescript
import { S3Kit } from 'ipfs_accelerate_js/api_backends';

// Create an instance with your S3 credentials
const s3Kit = new S3Kit();

// Delete an object from a bucket
const result = await s3Kit.deleteObject(
  'my-bucket',                // S3 bucket name
  'path/in/bucket/file.txt',  // S3 object key
  { priority: 'HIGH' }        // Optional request options
);

console.log(`Object deleted: ${result.DeleteMarker}`);
```

## Advanced Usage

### Priority-Based Request Queuing

The S3 Kit supports priority-based request queuing with three priority levels:

```typescript
// High priority request - processed first
await s3Kit.uploadFile(filePath, bucket, key, { priority: 'HIGH' });

// Normal priority request - processed after high priority
await s3Kit.downloadFile(bucket, key, filePath, { priority: 'NORMAL' });

// Low priority request - processed last
await s3Kit.listObjects(bucket, prefix, { priority: 'LOW' });
```

### Concurrent Operations

You can perform multiple operations concurrently:

```typescript
// Perform multiple operations concurrently
const operations = [
  s3Kit.listObjects('bucket-1'),
  s3Kit.listObjects('bucket-2'),
  s3Kit.listObjects('bucket-3')
];

const results = await Promise.all(operations);
```

### Error Handling

The S3 Kit includes robust error handling:

```typescript
try {
  await s3Kit.downloadFile('my-bucket', 'non-existent-key.txt', '/path/to/save.txt');
} catch (error) {
  console.error(`Error downloading file: ${error.message}`);
}
```

### Circuit Breaker Pattern

The S3 Kit implements a circuit breaker pattern to prevent cascading failures when an endpoint is experiencing issues:

1. **CLOSED state**: Normal operation, requests are processed
2. **OPEN state**: Too many failures have occurred, requests immediately fail
3. **HALF-OPEN state**: After a timeout period, the circuit breaker attempts to close by allowing a single request

This pattern improves reliability by:
- Preventing repeated calls to failing endpoints
- Allowing automatic recovery once the endpoint is healthy again
- Reducing load on struggling systems

### Endpoint Selection Strategies

The S3 Kit supports multiple endpoint selection strategies:

1. **Round-Robin**: Selects endpoints in a cyclical manner, ensuring equal distribution of requests
   ```typescript
   const endpoint = s3Kit.getEndpoint(undefined, 'round-robin');
   ```

2. **Least-Loaded**: Selects the endpoint with the fewest active requests
   ```typescript
   const endpoint = s3Kit.getEndpoint(undefined, 'least-loaded');
   ```

## Compatibility

The S3 Kit is compatible with any S3-compatible object storage service, including:

- Amazon S3
- MinIO
- Ceph Object Gateway
- Google Cloud Storage (with S3 compatibility)
- DigitalOcean Spaces
- Wasabi
- BackBlaze B2 (with S3 compatibility)

## Mock Mode

The S3 Kit includes a mock mode for testing without real S3 credentials. This is useful for development and testing:

```typescript
// When no S3 credentials are provided, the S3 Kit operates in mock mode
const mockS3Kit = new S3Kit();

// All operations work in mock mode, but don't actually interact with S3
const result = await mockS3Kit.listObjects('test-bucket');
console.log(result); // Returns mock data
```

## Error Handling

The S3 Kit throws standardized error objects with helpful properties. Common error types include:

- **Authentication errors**: When credentials are invalid or missing
- **Permission errors**: When access is denied to a bucket or object
- **Not found errors**: When a bucket or object doesn't exist
- **Rate limit errors**: When too many requests are made
- **Circuit open errors**: When the circuit breaker is open
- **Timeout errors**: When a request takes too long

```typescript
try {
  await s3Kit.downloadFile('bucket', 'key', '/path/to/save.txt');
} catch (error) {
  if (error.message.includes('not found')) {
    console.error('The specified object does not exist');
  } else if (error.message.includes('access denied')) {
    console.error('Permission denied for this operation');
  } else {
    console.error(`S3 operation failed: ${error.message}`);
  }
}
```

## Implementation Notes

The S3 Kit is designed to be a lightweight wrapper around S3-compatible APIs. In a production environment, it should be used with a full S3 client library like:

- AWS SDK for JavaScript (v3) for Node.js environments
- MinIO JavaScript Client for browser and Node.js environments
- AWS S3 REST API for direct interaction

For real production use of S3, consider importing these libraries directly rather than using the simplified mock implementation included in the S3 Kit.