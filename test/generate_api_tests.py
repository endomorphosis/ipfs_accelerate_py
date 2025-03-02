#!/usr/bin/env python
"""
Generate API Backend Test Files

This script creates test files for API backends that don't have them:
1. LLVM API test file
2. S3 Kit API test file
3. OPEA API test file (fixing the failing tests)

The generated files will follow the standardized test pattern for API backends.
"""

import os
import sys
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generate_api_tests")

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Template for API test files
TEST_FILE_TEMPLATE = """#!/usr/bin/env python
\"\"\"
Test script for {api_name_upper} API implementation.

This tests:
1. Basic API initialization and functionality
2. Queue system with concurrent requests
3. Backoff mechanism with simulated rate limits
4. Error handling scenarios
\"\"\"

import os
import sys
import time
import threading
import unittest
import logging
from unittest import mock
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("{api_name_lower}_api_test")

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the API backend
from ipfs_accelerate_py.api_backends.{api_name_lower} import {api_name_lower}

class {api_name_camel}APITest(unittest.TestCase):
    \"\"\"Test cases for {api_name_upper} API implementation\"\"\"
    
    def setUp(self):
        \"\"\"Set up test fixtures\"\"\"
        # Create API client with mock credentials
        self.api_client = {api_name_lower}(
            resources={{}},
            metadata={{
                "{api_name_lower}_api_key": "test_api_key_for_testing"
            }}
        )
        
        # Configure for testing
        self.api_client.max_retries = 3
        self.api_client.initial_retry_delay = 0.1
        self.api_client.max_concurrent_requests = 3
        self.api_client.queue_size = 10
    
    def test_init(self):
        \"\"\"Test API client initialization\"\"\"
        # Check if client was initialized properly
        self.assertIsNotNone(self.api_client)
        self.assertEqual(self.api_client.max_retries, 3)
        self.assertEqual(self.api_client.initial_retry_delay, 0.1)
        self.assertEqual(self.api_client.max_concurrent_requests, 3)
        self.assertEqual(self.api_client.queue_size, 10)
    
    @mock.patch('requests.post')
    def test_make_request(self, mock_post):
        \"\"\"Test the make_request method\"\"\"
        # Set up mock response
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {{"result": "success"}}
        mock_post.return_value = mock_response
        
        # Make a request
        result = self.api_client.make_request(
            endpoint_url="https://api.example.com/v1/endpoint",
            data={{"prompt": "Hello, world!"}},
            api_key="test_key",
            request_id="test_req_123"
        )
        
        # Verify the request was made correctly
        mock_post.assert_called_once()
        self.assertEqual(result, {{"result": "success"}})
    
    @mock.patch('requests.post')
    def test_backoff_mechanism(self, mock_post):
        \"\"\"Test exponential backoff for rate limits\"\"\"
        # Set up mock responses
        rate_limit_response = mock.Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {{"retry-after": "0.1"}}
        rate_limit_response.json.return_value = {{"error": {{"message": "Rate limit exceeded"}}}}
        
        success_response = mock.Mock()
        success_response.status_code = 200
        success_response.json.return_value = {{"result": "success after retry"}}
        
        # Configure mock to return rate limit first, then success
        mock_post.side_effect = [rate_limit_response, success_response]
        
        # Make a request
        result = self.api_client.make_request(
            endpoint_url="https://api.example.com/v1/endpoint",
            data={{"prompt": "Hello, world!"}},
            api_key="test_key",
            request_id="test_req_123"
        )
        
        # Verify the request was retried and returned the second response
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(result, {{"result": "success after retry"}})
    
    @mock.patch('requests.post')
    def test_queue_system(self, mock_post):
        \"\"\"Test queue system for concurrent requests\"\"\"
        # Configure mock to take time to respond
        def delayed_response(*args, **kwargs):
            time.sleep(0.2)  # Delay to ensure concurrent requests queue up
            response = mock.Mock()
            response.status_code = 200
            response.json.return_value = {{"result": "success"}}
            return response
            
        mock_post.side_effect = delayed_response
        
        # Number of concurrent requests (more than max_concurrent_requests)
        request_count = 10
        
        # Make concurrent requests
        with ThreadPoolExecutor(max_workers=request_count) as executor:
            futures = []
            for i in range(request_count):
                futures.append(executor.submit(
                    self.api_client.make_request,
                    endpoint_url=f"https://api.example.com/v1/endpoint",
                    data={{"prompt": f"Request {{i}}"}},
                    api_key="test_key",
                    request_id=f"test_req_{{i}}"
                ))
                
            # Wait for all requests to complete
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        
        # Verify all requests completed successfully
        self.assertEqual(len(results), request_count)
        self.assertTrue(all("result" in r for r in results))
        
        # Verify the correct number of actual requests were made
        self.assertEqual(mock_post.call_count, request_count)
    
    @mock.patch('requests.post')
    def test_unique_request_ids(self, mock_post):
        \"\"\"Test automatic generation of request IDs\"\"\"
        # Set up mock response
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {{"result": "success"}}
        mock_post.return_value = mock_response
        
        # Capture request headers to check request_id
        captured_headers = []
        def capture_headers(*args, **kwargs):
            captured_headers.append(kwargs.get("headers", {{}}))
            return mock_response
            
        mock_post.side_effect = capture_headers
        
        # Make multiple requests without explicit request_id
        for _ in range(3):
            self.api_client.make_request(
                endpoint_url="https://api.example.com/v1/endpoint",
                data={{"prompt": "Hello, world!"}},
                api_key="test_key"
            )
        
        # Verify each request got a unique ID
        request_ids = [h.get("X-Request-ID") for h in captured_headers]
        self.assertEqual(len(request_ids), 3)
        self.assertEqual(len(set(request_ids)), 3)  # All IDs should be unique
    
    @mock.patch('requests.post')
    def test_error_handling(self, mock_post):
        \"\"\"Test error handling for API errors\"\"\"
        # Set up mock response for various errors
        auth_error = mock.Mock()
        auth_error.status_code = 401
        auth_error.json.return_value = {{"error": {{"message": "Invalid API key"}}}}
        
        server_error = mock.Mock()
        server_error.status_code = 500
        server_error.json.return_value = {{"error": {{"message": "Internal server error"}}}}
        
        # Test authentication error
        mock_post.return_value = auth_error
        with self.assertRaises(ValueError):
            self.api_client.make_request(
                endpoint_url="https://api.example.com/v1/endpoint",
                data={{"prompt": "Hello, world!"}},
                api_key="invalid_key",
                request_id="test_req_auth_error"
            )
        
        # Test server error (should retry and eventually fail)
        mock_post.reset_mock()
        mock_post.return_value = server_error
        
        with self.assertRaises(ValueError):
            self.api_client.make_request(
                endpoint_url="https://api.example.com/v1/endpoint",
                data={{"prompt": "Hello, world!"}},
                api_key="test_key",
                request_id="test_req_server_error"
            )
        
        # Should have retried max_retries times
        self.assertEqual(mock_post.call_count, self.api_client.max_retries)

if __name__ == '__main__':
    unittest.main()
"""

def generate_test_file(api_name, output_dir):
    """Generate test file for the specified API"""
    api_name_lower = api_name.lower()
    api_name_upper = api_name.upper()
    api_name_camel = api_name.title().replace('_', '')
    
    # Create test file content
    test_content = TEST_FILE_TEMPLATE.format(
        api_name_lower=api_name_lower,
        api_name_upper=api_name_upper,
        api_name_camel=api_name_camel
    )
    
    # Write to file
    test_file_path = output_dir / f"test_{api_name_lower}.py"
    
    # Check if file already exists
    if test_file_path.exists():
        logger.warning(f"Test file {test_file_path} already exists. Creating a .new version instead.")
        test_file_path = output_dir / f"test_{api_name_lower}.py.new"
    
    with open(test_file_path, "w") as f:
        f.write(test_content)
    
    logger.info(f"✅ Generated test file: {test_file_path}")
    return test_file_path

def main():
    """Main function to generate missing test files"""
    parser = argparse.ArgumentParser(description="Generate missing test files for API backends")
    parser.add_argument("--api", choices=["llvm", "s3_kit", "opea", "all"], default="all",
                       help="Generate test for specific API or all missing tests")
    parser.add_argument("--output-dir", default=None,
                       help="Custom output directory for test files (default: ./)")
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent
    
    if not output_dir.exists():
        logger.error(f"Output directory {output_dir} does not exist")
        return 1
    
    # Determine which APIs to generate tests for
    if args.api == "all":
        apis_to_generate = ["llvm", "s3_kit", "opea"]
    else:
        apis_to_generate = [args.api]
    
    # Generate test files
    generated_files = []
    for api_name in apis_to_generate:
        test_file = generate_test_file(api_name, output_dir)
        generated_files.append((api_name, test_file))
    
    # Print summary
    logger.info("\n=== Test File Generation Summary ===")
    for api_name, file_path in generated_files:
        logger.info(f"{api_name}: Generated {file_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())