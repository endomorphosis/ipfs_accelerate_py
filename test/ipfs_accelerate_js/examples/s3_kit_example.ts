/**
 * S3 Kit API Backend Example
 *
 * This example demonstrates how to use the S3 Kit API backend for various S3-compatible
 * object storage operations, including file upload, download, listing, and deletion.
 */

import { S3Kit } from '../src/api_backends/s3_kit';
import * as fs from 'fs';
import * as path from 'path';

// Create a new S3Kit instance
// You should set S3_ENDPOINT, S3_ACCESS_KEY, and S3_SECRET_KEY environment variables
// Or provide them in metadata when creating the instance
const s3Kit = new S3Kit({
  // Optional configuration
  max_concurrent_requests: 10,
  queue_size: 100,
  max_retries: 3,
  initial_retry_delay: 1000,
  backoff_factor: 2,
  endpoint_selection_strategy: 'round-robin',
  // Circuit breaker configuration
  circuit_breaker: {
    threshold: 3,  // Number of failures before opening the circuit
    timeout: 60000  // Time in ms before attempting to half-open the circuit
  }
}, {
  // Optional metadata
  s3cfg: {
    endpoint: process.env.S3_ENDPOINT,
    accessKey: process.env.S3_ACCESS_KEY,
    secretKey: process.env.S3_SECRET_KEY
  }
});

/**
 * Basic endpoint configuration example
 */
async function configureEndpointsExample() {
  console.log('Running endpoint configuration example...');
  
  try {
    // Add an additional endpoint
    const endpoint1 = s3Kit.addEndpoint(
      'minio-local',
      'http://localhost:9000',
      process.env.MINIO_ACCESS_KEY || 'minioadmin',
      process.env.MINIO_SECRET_KEY || 'minioadmin',
      5,  // Maximum concurrent requests
      3,  // Circuit breaker threshold
      2   // Maximum retries
    );
    
    console.log('Added MinIO local endpoint');
    
    // Add another endpoint (e.g., for a different region or provider)
    const endpoint2 = s3Kit.addEndpoint(
      'aws-s3',
      'https://s3.amazonaws.com',
      process.env.AWS_ACCESS_KEY || '',
      process.env.AWS_SECRET_KEY || '',
      10,  // Maximum concurrent requests
      5,   // Circuit breaker threshold
      3    // Maximum retries
    );
    
    console.log('Added AWS S3 endpoint');
    
    // Test the endpoints
    const minioWorking = await s3Kit.testS3Endpoint(
      'http://localhost:9000',
      process.env.MINIO_ACCESS_KEY || 'minioadmin',
      process.env.MINIO_SECRET_KEY || 'minioadmin'
    );
    
    console.log(`MinIO endpoint available: ${minioWorking}`);
    
    // Get endpoint by name
    const endpoint = s3Kit.getEndpoint('minio-local');
    console.log('Retrieved endpoint by name');
    
    // Get endpoint using a strategy
    const leastLoadedEndpoint = s3Kit.getEndpoint(undefined, 'least-loaded');
    console.log('Retrieved endpoint using least-loaded strategy');
  } catch (error) {
    console.error('Error in endpoint configuration example:', error);
  }
}

/**
 * File upload example
 */
async function uploadFileExample() {
  console.log('\nRunning file upload example...');
  
  try {
    // Create a sample file to upload
    const sampleFilePath = path.join(__dirname, 'sample-upload.txt');
    
    if (!fs.existsSync(sampleFilePath)) {
      fs.writeFileSync(sampleFilePath, 'This is a sample file for upload.\n');
      console.log('Created sample file at:', sampleFilePath);
    }
    
    // Upload the file to a bucket
    const bucket = 'test-bucket';
    const key = 'sample/upload.txt';
    
    const result = await s3Kit.uploadFile(
      sampleFilePath,
      bucket,
      key,
      { priority: 'HIGH' }  // High priority request
    );
    
    console.log('Upload successful:');
    console.log(`- Bucket: ${result.Bucket}`);
    console.log(`- Key: ${result.Key}`);
    console.log(`- ETag: ${result.ETag}`);
    console.log(`- Location: ${result.Location}`);
  } catch (error) {
    console.error('Error in file upload example:', error);
  }
}

/**
 * File download example
 */
async function downloadFileExample() {
  console.log('\nRunning file download example...');
  
  try {
    // Download a file from a bucket
    const bucket = 'test-bucket';
    const key = 'sample/upload.txt';
    const downloadPath = path.join(__dirname, 'sample-download.txt');
    
    const result = await s3Kit.downloadFile(
      bucket,
      key,
      downloadPath,
      { priority: 'NORMAL' }  // Normal priority request
    );
    
    console.log('Download successful:');
    console.log(`- Bucket: ${result.Bucket}`);
    console.log(`- Key: ${result.Key}`);
    console.log(`- Downloaded to: ${downloadPath}`);
    
    // In a real application, you would check if the file exists
    if (fs.existsSync(downloadPath)) {
      const content = fs.readFileSync(downloadPath, 'utf8');
      console.log('- File content:', content);
    }
  } catch (error) {
    console.error('Error in file download example:', error);
  }
}

/**
 * List objects example
 */
async function listObjectsExample() {
  console.log('\nRunning list objects example...');
  
  try {
    // List objects in a bucket
    const bucket = 'test-bucket';
    const prefix = 'sample/';  // Optional prefix to filter results
    
    const result = await s3Kit.listObjects(
      bucket,
      prefix,
      {
        max_keys: 100,  // Maximum number of objects to list
        priority: 'LOW'  // Low priority request
      }
    );
    
    console.log('List objects successful:');
    console.log(`- Bucket: ${result.Name}`);
    console.log(`- Prefix: ${result.Prefix}`);
    console.log(`- Max Keys: ${result.MaxKeys}`);
    console.log(`- Objects (${result.Contents.length}):`);
    
    result.Contents.forEach((object, index) => {
      console.log(`  ${index + 1}. ${object.Key} (${object.Size} bytes, Last Modified: ${object.LastModified})`);
    });
  } catch (error) {
    console.error('Error in list objects example:', error);
  }
}

/**
 * Delete object example
 */
async function deleteObjectExample() {
  console.log('\nRunning delete object example...');
  
  try {
    // Delete an object from a bucket
    const bucket = 'test-bucket';
    const key = 'sample/upload.txt';
    
    const result = await s3Kit.deleteObject(
      bucket,
      key,
      { priority: 'HIGH' }  // High priority request
    );
    
    console.log('Delete object successful:');
    console.log(`- Delete Marker: ${result.DeleteMarker}`);
    console.log(`- Version ID: ${result.VersionId}`);
  } catch (error) {
    console.error('Error in delete object example:', error);
  }
}

/**
 * Endpoint testing example
 */
async function testEndpointExample() {
  console.log('\nRunning endpoint testing example...');
  
  try {
    // Test the default endpoint
    const isAvailable = await s3Kit.testEndpoint();
    console.log(`Default S3 endpoint available: ${isAvailable}`);
    
    // Test a specific endpoint
    const customEndpointUrl = 'https://custom-s3-endpoint.com';
    const customAccessKey = 'custom-access-key';
    const customSecretKey = 'custom-secret-key';
    
    const customAvailable = await s3Kit.testS3Endpoint(
      customEndpointUrl,
      customAccessKey,
      customSecretKey
    );
    
    console.log(`Custom S3 endpoint available: ${customAvailable}`);
  } catch (error) {
    console.error('Error in endpoint testing example:', error);
  }
}

/**
 * Error handling example
 */
async function errorHandlingExample() {
  console.log('\nRunning error handling example...');
  
  try {
    // Try to list objects from a non-existent bucket
    await s3Kit.listObjects('non-existent-bucket');
  } catch (error) {
    console.log('Caught error from non-existent bucket:');
    console.log(`- Error message: ${error.message}`);
  }
  
  try {
    // Try to download a non-existent object
    await s3Kit.downloadFile(
      'test-bucket',
      'non-existent-key.txt',
      path.join(__dirname, 'non-existent-download.txt')
    );
  } catch (error) {
    console.log('Caught error from non-existent object:');
    console.log(`- Error message: ${error.message}`);
  }
}

/**
 * Concurrent operations example
 */
async function concurrentOperationsExample() {
  console.log('\nRunning concurrent operations example...');
  
  try {
    // Perform multiple operations concurrently
    const operations = [
      s3Kit.listObjects('test-bucket-1'),
      s3Kit.listObjects('test-bucket-2'),
      s3Kit.listObjects('test-bucket-3')
    ];
    
    console.log('Starting concurrent operations...');
    const results = await Promise.all(operations);
    
    console.log('All concurrent operations completed');
    console.log(`- Number of results: ${results.length}`);
    
    // In a real application, you would process the results
    results.forEach((result, index) => {
      console.log(`- Result ${index + 1}: Bucket: ${result.Name}, Objects: ${result.Contents.length}`);
    });
  } catch (error) {
    console.error('Error in concurrent operations example:', error);
  }
}

/**
 * Environment variables example
 */
function environmentVariablesExample() {
  console.log('\nEnvironment variables configuration:');
  console.log('To use S3 Kit, set the following environment variables:');
  console.log('- S3_ENDPOINT: Your S3-compatible endpoint URL');
  console.log('- S3_ACCESS_KEY: Your access key');
  console.log('- S3_SECRET_KEY: Your secret key');
  console.log('- Example: export S3_ENDPOINT=https://s3.amazonaws.com');
  console.log('- Or use a .env file with a library like dotenv');
}

/**
 * Run all examples
 */
async function runExamples() {
  // Check if required environment variables are set
  if (!process.env.S3_ENDPOINT || !process.env.S3_ACCESS_KEY || !process.env.S3_SECRET_KEY) {
    console.log('Required environment variables are not set. Using mock values for examples.');
    console.log('For real S3 operations, set S3_ENDPOINT, S3_ACCESS_KEY, and S3_SECRET_KEY.');
    console.log('Running examples with mock S3 operations...\n');
  }
  
  // Run all examples
  await configureEndpointsExample();
  await uploadFileExample();
  await downloadFileExample();
  await listObjectsExample();
  await deleteObjectExample();
  await testEndpointExample();
  await errorHandlingExample();
  await concurrentOperationsExample();
  environmentVariablesExample();
  
  console.log('\nAll examples completed.');
}

// Run examples if this file is executed directly
if (require.main === module) {
  runExamples().catch(error => {
    console.error('Error running examples:', error);
  });
}