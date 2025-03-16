// Tests for S3 Kit API Backend
import { S3Kit } from '../../src/api_backends/s3_kit';
import * as fs from 'fs';
import * as path from 'path';

// Mock fetch global
global.fetch = jest.fn();
const mockFetch = global.fetch as jest.Mock;

// Mock fs module
jest.mock('fs', () => ({
  existsSync: jest.fn().mockReturnValue(true),
  readFileSync: jest.fn().mockReturnValue(Buffer.from('mock file content')),
  writeFileSync: jest.fn()
}));

// Mock path module
jest.mock('path', () => ({
  join: jest.fn(),
  basename: jest.fn().mockReturnValue('test-file.txt')
}));

describe('S3 Kit API Backend', () => {
  let backend: S3Kit;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    mockFetch.mockReset();

    // Mock successful fetch response
    mockFetch.mockResolvedValue({
      status: 200,
      ok: true,
      json: async () => ({ success: true })
    });

    // Set up environment variables
    process.env.S3_ENDPOINT = 'https://mock-s3-endpoint.com';
    process.env.S3_ACCESS_KEY = 'mock-access-key';
    process.env.S3_SECRET_KEY = 'mock-secret-key';

    // Create backend instance
    backend = new S3Kit();
  });

  afterEach(() => {
    // Clean up environment variables
    delete process.env.S3_ENDPOINT;
    delete process.env.S3_ACCESS_KEY;
    delete process.env.S3_SECRET_KEY;
  });

  test('should initialize correctly', () => {
    expect(backend).toBeDefined();
    
    // Check default values
    // @ts-ignore - accessing private properties for testing
    expect(backend.maxConcurrentRequests).toBe(10);
    // @ts-ignore - accessing private properties for testing
    expect(backend.queueSize).toBe(100);
    // @ts-ignore - accessing private properties for testing
    expect(backend.maxRetries).toBe(3);
    // @ts-ignore - accessing private properties for testing
    expect(backend.endpointSelectionStrategy).toBe('round-robin');
  });

  test('should initialize with custom options', () => {
    const customBackend = new S3Kit({
      max_concurrent_requests: 20,
      queue_size: 200,
      max_retries: 5,
      initial_retry_delay: 2000,
      backoff_factor: 3,
      default_timeout: 60000,
      endpoint_selection_strategy: 'least-loaded',
      circuit_breaker: {
        threshold: 5,
        timeout: 120000
      }
    });

    expect(customBackend).toBeDefined();
    
    // Check custom values
    // @ts-ignore - accessing private properties for testing
    expect(customBackend.maxConcurrentRequests).toBe(20);
    // @ts-ignore - accessing private properties for testing
    expect(customBackend.queueSize).toBe(200);
    // @ts-ignore - accessing private properties for testing
    expect(customBackend.maxRetries).toBe(5);
    // @ts-ignore - accessing private properties for testing
    expect(customBackend.initialRetryDelay).toBe(2000);
    // @ts-ignore - accessing private properties for testing
    expect(customBackend.backoffFactor).toBe(3);
    // @ts-ignore - accessing private properties for testing
    expect(customBackend.defaultTimeout).toBe(60000);
    // @ts-ignore - accessing private properties for testing
    expect(customBackend.endpointSelectionStrategy).toBe('least-loaded');
    // @ts-ignore - accessing private properties for testing
    expect(customBackend.circuitBreakerThreshold).toBe(5);
    // @ts-ignore - accessing private properties for testing
    expect(customBackend.circuitBreakerTimeout).toBe(120000);
  });

  test('should get API key from environment variables', () => {
    const apiKey = backend.getApiKey();
    expect(apiKey).toBe('mock-access-key');
  });

  test('should get API key from metadata', () => {
    const customBackend = new S3Kit({}, {
      s3cfg: {
        accessKey: 'metadata-access-key',
        secretKey: 'metadata-secret-key',
        endpoint: 'https://metadata-endpoint.com'
      }
    });

    const apiKey = customBackend.getApiKey();
    expect(apiKey).toBe('metadata-access-key');
  });

  test('should get default S3 endpoint from environment variables', () => {
    // @ts-ignore - accessing private method for testing
    const endpoint = backend.getDefaultS3Endpoint();
    expect(endpoint).toBe('https://mock-s3-endpoint.com');
  });

  test('should get default S3 endpoint from metadata', () => {
    const customBackend = new S3Kit({}, {
      s3cfg: {
        accessKey: 'metadata-access-key',
        secretKey: 'metadata-secret-key',
        endpoint: 'https://metadata-endpoint.com'
      }
    });

    // @ts-ignore - accessing private method for testing
    const endpoint = customBackend.getDefaultS3Endpoint();
    expect(endpoint).toBe('https://metadata-endpoint.com');
  });

  test('should get default model (empty for S3)', () => {
    const model = backend.getDefaultModel();
    expect(model).toBe('');
  });

  test('should check model compatibility (always false for S3)', () => {
    expect(backend.isCompatibleModel('any-model')).toBe(false);
  });

  test('should add endpoint', () => {
    const handler = backend.addEndpoint(
      'test-endpoint',
      'https://test-endpoint.com',
      'test-access-key',
      'test-secret-key',
      10,
      5,
      3
    );

    expect(handler).toBeDefined();
    
    // @ts-ignore - accessing private properties for testing
    const endpoints = backend.endpoints;
    expect(endpoints.has('test-endpoint')).toBe(true);
    
    const endpoint = endpoints.get('test-endpoint');
    expect(endpoint).toBeDefined();
    
    // Check endpoint metadata
    // @ts-ignore - accessing private properties for testing
    expect(endpoint.metadata.endpoint_url).toBe('https://test-endpoint.com');
    // @ts-ignore - accessing private properties for testing
    expect(endpoint.metadata.access_key).toBe('test-access-key');
    // @ts-ignore - accessing private properties for testing
    expect(endpoint.metadata.secret_key).toBe('test-secret-key');
    // @ts-ignore - accessing private properties for testing
    expect(endpoint.metadata.max_concurrent).toBe(10);
    // @ts-ignore - accessing private properties for testing
    expect(endpoint.metadata.circuit_breaker_threshold).toBe(5);
    // @ts-ignore - accessing private properties for testing
    expect(endpoint.metadata.retries).toBe(3);
  });

  test('should throw error when adding endpoint without URL', () => {
    expect(() => {
      backend.addEndpoint(
        'test-endpoint',
        '',  // Empty URL
        'test-access-key',
        'test-secret-key'
      );
    }).toThrow('Endpoint URL is required');
  });

  test('should throw error when adding endpoint without access keys', () => {
    expect(() => {
      backend.addEndpoint(
        'test-endpoint',
        'https://test-endpoint.com',
        null,  // No access key
        'test-secret-key'
      );
    }).toThrow('Access key and secret key are required');

    expect(() => {
      backend.addEndpoint(
        'test-endpoint',
        'https://test-endpoint.com',
        'test-access-key',
        null  // No secret key
      );
    }).toThrow('Access key and secret key are required');
  });

  test('should get endpoint by name', () => {
    // Add test endpoints
    backend.addEndpoint(
      'endpoint1',
      'https://endpoint1.com',
      'access-key-1',
      'secret-key-1'
    );
    
    backend.addEndpoint(
      'endpoint2',
      'https://endpoint2.com',
      'access-key-2',
      'secret-key-2'
    );

    // Get endpoint by name
    const endpoint = backend.getEndpoint('endpoint2');
    expect(endpoint).toBeDefined();
    
    // @ts-ignore - accessing private properties for testing
    expect(endpoint.metadata.endpoint_url).toBe('https://endpoint2.com');
  });

  test('should get endpoint with round-robin strategy', () => {
    // Add test endpoints
    backend.addEndpoint(
      'endpoint1',
      'https://endpoint1.com',
      'access-key-1',
      'secret-key-1'
    );
    
    backend.addEndpoint(
      'endpoint2',
      'https://endpoint2.com',
      'access-key-2',
      'secret-key-2'
    );

    // Get endpoint with round-robin strategy
    // @ts-ignore - accessing private properties for testing
    backend.lastUsed.set('endpoint1', Date.now());
    // @ts-ignore - accessing private properties for testing
    backend.lastUsed.set('endpoint2', Date.now() - 1000);  // Set endpoint2 as used longer ago

    const endpoint = backend.getEndpoint(undefined, 'round-robin');
    expect(endpoint).toBeDefined();
    
    // Should select endpoint2 as it was used longer ago
    // @ts-ignore - accessing private properties for testing
    expect(endpoint.metadata.endpoint_url).toBe('https://endpoint2.com');
  });

  test('should get endpoint with least-loaded strategy', () => {
    // Add test endpoints
    backend.addEndpoint(
      'endpoint1',
      'https://endpoint1.com',
      'access-key-1',
      'secret-key-1'
    );
    
    backend.addEndpoint(
      'endpoint2',
      'https://endpoint2.com',
      'access-key-2',
      'secret-key-2'
    );

    // Get endpoint with least-loaded strategy
    // @ts-ignore - accessing private properties for testing
    backend.requestsPerEndpoint.set('endpoint1', 10);
    // @ts-ignore - accessing private properties for testing
    backend.requestsPerEndpoint.set('endpoint2', 5);  // Set endpoint2 as less loaded

    const endpoint = backend.getEndpoint(undefined, 'least-loaded');
    expect(endpoint).toBeDefined();
    
    // Should select endpoint2 as it has fewer requests
    // @ts-ignore - accessing private properties for testing
    expect(endpoint.metadata.endpoint_url).toBe('https://endpoint2.com');
  });

  test('should throw error when no endpoints are available', () => {
    // Create a new backend without default endpoint
    delete process.env.S3_ENDPOINT;
    const emptyBackend = new S3Kit();

    expect(() => {
      emptyBackend.getEndpoint();
    }).toThrow('No S3 endpoints have been added');
  });

  test('should test S3 endpoint successfully', async () => {
    mockFetch.mockResolvedValueOnce({
      status: 200,
      ok: true
    });

    const result = await backend.testS3Endpoint(
      'https://test-endpoint.com',
      'test-access-key',
      'test-secret-key'
    );

    expect(result).toBe(true);
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      'https://test-endpoint.com?location',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          'Authorization': expect.stringContaining('AWS test-access-key:')
        })
      })
    );
  });

  test('should test S3 endpoint with failure', async () => {
    mockFetch.mockResolvedValueOnce({
      status: 403,
      ok: false
    });

    const result = await backend.testS3Endpoint(
      'https://test-endpoint.com',
      'test-access-key',
      'test-secret-key'
    );

    expect(result).toBe(false);
  });

  test('should throw error when testing endpoint without credentials', async () => {
    await expect(
      backend.testS3Endpoint('https://test-endpoint.com', null, 'test-secret-key')
    ).rejects.toThrow('Access key and secret key are required');

    await expect(
      backend.testS3Endpoint('https://test-endpoint.com', 'test-access-key', null)
    ).rejects.toThrow('Access key and secret key are required');
  });

  test('should test default endpoint', async () => {
    mockFetch.mockResolvedValueOnce({
      status: 200,
      ok: true
    });

    const result = await backend.testEndpoint();

    expect(result).toBe(true);
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch).toHaveBeenCalledWith(
      'https://mock-s3-endpoint.com?location',
      expect.anything()
    );
  });

  test('should return false when default endpoint is not available', async () => {
    // Create a backend without default endpoint
    delete process.env.S3_ENDPOINT;
    const noEndpointBackend = new S3Kit();

    const result = await noEndpointBackend.testEndpoint();
    expect(result).toBe(false);
  });

  test('should queue and process upload file request', async () => {
    // Mock processQueue to resolve immediately
    // @ts-ignore - mocking private method for testing
    jest.spyOn(backend, 'processQueue').mockImplementation(async () => {
      // @ts-ignore - accessing private property for testing
      const request = backend.requestQueue[0];
      // @ts-ignore - mocking private method for testing
      const endpoint = backend.getEndpoint();
      const result = await endpoint.processRequest(request.operation, request.options);
      request.resolve(result);
    });

    const result = await backend.uploadFile(
      '/path/to/file.txt',
      'test-bucket',
      'test-key',
      { priority: 'HIGH' }
    );

    expect(result).toBeDefined();
    expect(result.Bucket).toBe('test-bucket');
    expect(result.Key).toBe('test-key');
    
    // @ts-ignore - verify private method was called
    expect(backend.processQueue).toHaveBeenCalled();
  });

  test('should queue and process download file request', async () => {
    // Mock processQueue to resolve immediately
    // @ts-ignore - mocking private method for testing
    jest.spyOn(backend, 'processQueue').mockImplementation(async () => {
      // @ts-ignore - accessing private property for testing
      const request = backend.requestQueue[0];
      // @ts-ignore - mocking private method for testing
      const endpoint = backend.getEndpoint();
      const result = await endpoint.processRequest(request.operation, request.options);
      request.resolve(result);
    });

    const result = await backend.downloadFile(
      'test-bucket',
      'test-key',
      '/path/to/save.txt',
      { priority: 'NORMAL' }
    );

    expect(result).toBeDefined();
    expect(result.Bucket).toBe('test-bucket');
    expect(result.Key).toBe('test-key');
    expect(result.Body).toBe('Mock file content');
    
    // @ts-ignore - verify private method was called
    expect(backend.processQueue).toHaveBeenCalled();
  });

  test('should queue and process list objects request', async () => {
    // Mock processQueue to resolve immediately
    // @ts-ignore - mocking private method for testing
    jest.spyOn(backend, 'processQueue').mockImplementation(async () => {
      // @ts-ignore - accessing private property for testing
      const request = backend.requestQueue[0];
      // @ts-ignore - mocking private method for testing
      const endpoint = backend.getEndpoint();
      const result = await endpoint.processRequest(request.operation, request.options);
      request.resolve(result);
    });

    const result = await backend.listObjects(
      'test-bucket',
      'test-prefix',
      { max_keys: 50, priority: 'LOW' }
    );

    expect(result).toBeDefined();
    expect(result.Name).toBe('test-bucket');
    expect(result.Prefix).toBe('test-prefix');
    expect(result.Contents).toHaveLength(2);
    expect(result.MaxKeys).toBe(50);
    
    // @ts-ignore - verify private method was called
    expect(backend.processQueue).toHaveBeenCalled();
  });

  test('should queue and process delete object request', async () => {
    // Mock processQueue to resolve immediately
    // @ts-ignore - mocking private method for testing
    jest.spyOn(backend, 'processQueue').mockImplementation(async () => {
      // @ts-ignore - accessing private property for testing
      const request = backend.requestQueue[0];
      // @ts-ignore - mocking private method for testing
      const endpoint = backend.getEndpoint();
      const result = await endpoint.processRequest(request.operation, request.options);
      request.resolve(result);
    });

    const result = await backend.deleteObject(
      'test-bucket',
      'test-key',
      { priority: 'HIGH' }
    );

    expect(result).toBeDefined();
    expect(result.DeleteMarker).toBe(true);
    expect(result.VersionId).toBe('mockVersionId');
    
    // @ts-ignore - verify private method was called
    expect(backend.processQueue).toHaveBeenCalled();
  });

  test('should create endpoint handler', () => {
    const handler = backend.createEndpointHandler('https://test-endpoint.com');
    expect(handler).toBeDefined();
    expect(typeof handler).toBe('function');
  });

  test('should throw error for unsupported API methods', async () => {
    await expect(
      backend.makePostRequest('url', { data: 'test' })
    ).rejects.toThrow('Method not applicable for S3 Kit');

    await expect(
      backend.makeStreamRequest('url', { data: 'test' })
    ).rejects.toThrow('Method not applicable for S3 Kit');

    await expect(
      backend.chat('model', [{ role: 'user', content: 'Hello' }])
    ).rejects.toThrow('Method not applicable for S3 Kit');

    // Need to wrap in a function since streamChat returns an AsyncGenerator
    await expect(async () => {
      for await (const chunk of backend.streamChat('model', [{ role: 'user', content: 'Hello' }])) {
        // This should not execute
      }
    }).rejects.toThrow('Method not applicable for S3 Kit');
  });

  test('should handle circuit breaker states correctly', async () => {
    // Add test endpoint
    const handler = backend.addEndpoint(
      'test-endpoint',
      'https://test-endpoint.com',
      'test-access-key',
      'test-secret-key',
      5,
      2,  // Set circuit breaker threshold to 2 for easier testing
      1
    );

    // Mock circuit breaker state
    // @ts-ignore - accessing private properties for testing
    handler.circuitState = 'OPEN';
    // @ts-ignore - accessing private properties for testing
    handler.lastStateChange = Date.now() - 10000;  // Set as changed 10 seconds ago
    
    // Mock processRequest to test circuit breaker
    // @ts-ignore - mocking private method for testing
    const processRequestSpy = jest.spyOn(handler, 'processRequest');
    
    // First call should transition to HALF_OPEN
    processRequestSpy.mockImplementationOnce(async () => {
      // @ts-ignore - accessing private properties for testing
      expect(handler.circuitState).toBe('HALF_OPEN');
      return { success: true };
    });
    
    // Test it
    await handler.processRequest('list_objects', { bucket: 'test-bucket' });
    
    expect(processRequestSpy).toHaveBeenCalledTimes(1);
    // @ts-ignore - accessing private properties for testing
    expect(handler.circuitState).toBe('CLOSED');  // Should close on success
  });

  test('should retry failed requests', async () => {
    // Mock queueRequest to track calls
    // @ts-ignore - mocking private method for testing
    const queueRequestSpy = jest.spyOn(backend, 'queueRequest');
    
    // Mock processQueue to simulate a failure and retry
    // @ts-ignore - mocking private method for testing
    jest.spyOn(backend, 'processQueue').mockImplementationOnce(async () => {
      // @ts-ignore - accessing private property for testing
      const request = backend.requestQueue[0];
      
      // Simulate failure on first attempt
      if (request.retryCount === 0) {
        request.retryCount++;
        
        // Simulate retry by calling queueRequest again after a delay
        setTimeout(() => {
          // @ts-ignore - accessing private method for testing
          backend.queueRequest(request);
        }, 10);
        
        request.reject(new Error('Simulated failure'));
      } else {
        // Success on retry
        // @ts-ignore - mocking private method for testing
        const endpoint = backend.getEndpoint();
        const result = await endpoint.processRequest(request.operation, request.options);
        request.resolve(result);
      }
    });
    
    // Configure Jest timers
    jest.useFakeTimers();

    // Start the operation
    const promise = backend.listObjects('test-bucket');
    
    // Fast-forward timer to trigger the retry
    jest.advanceTimersByTime(100);
    
    // Complete the operation
    const result = await promise;
    
    // Restore real timers
    jest.useRealTimers();
    
    // Verify results
    expect(result).toBeDefined();
    expect(result.Name).toBe('test-bucket');
    
    // Should have been called twice (original + retry)
    expect(queueRequestSpy).toHaveBeenCalledTimes(2);
  });

  test('should handle queue full condition', async () => {
    // Create backend with small queue size
    const smallQueueBackend = new S3Kit({
      queue_size: 1
    });
    
    // Fill the queue
    // @ts-ignore - accessing private property for testing
    smallQueueBackend.requestQueue = [{ id: 'test' }];
    
    // Try to queue another request
    await expect(
      smallQueueBackend.listObjects('test-bucket')
    ).rejects.toThrow('Request queue is full');
  });
});