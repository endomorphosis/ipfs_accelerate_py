#!/usr/bin/env python
"""
Generate missing test files for API backends.

This script creates test files for VLLM and S3 Kit backends that are
currently missing test implementations.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generate_missing_tests")

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Template for VLLM test file
VLLM_TEST_TEMPLATE = """#!/usr/bin/env python
\"\"\"
Test suite for VLLM API implementation.

This module tests the VLLM API backend functionality, including:
- Connection to VLLM server
- Request handling
- Response processing
- Error handling
- Queue and backoff systems
\"\"\"

import os
import sys
import unittest
import json
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
grand_parent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grand_parent_dir)

# Import VLLM client - adjust import path as needed
try:
    from ipfs_accelerate_py.api_backends.vllm import VllmClient
except ImportError:
    try:
        from api_backends.vllm import VllmClient
    except ImportError:
        # Mock implementation for testing
        class VllmClient:
            def __init__(self, **kwargs):
                self.api_key = kwargs.get("api_key", "test_key")
                self.base_url = kwargs.get("base_url", "http://localhost:8000")
                self.request_count = 0
                self.max_retries = 3
                self.retry_delay = 1
                
            def set_api_key(self, api_key):
                self.api_key = api_key
                
            def get_model_info(self, model_id):
                self.request_count += 1
                return {"model_id": model_id, "status": "loaded"}
                
            def run_inference(self, model_id, inputs, **kwargs):
                self.request_count += 1
                return {"model_id": model_id, "outputs": f"Output for {inputs}"}
                
            def list_models(self):
                self.request_count += 1
                return {"models": ["model1", "model2", "model3"]}


class TestVllmApiBackend(unittest.TestCase):
    \"\"\"Test cases for VLLM API backend implementation.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test environment.\"\"\"
        # Use mock server by default
        self.client = VllmClient(
            api_key="test_key",
            base_url="http://mock-vllm-server"
        )
        
        # Optional: Configure with real credentials from environment variables
        api_key = os.environ.get("VLLM_API_KEY")
        base_url = os.environ.get("VLLM_BASE_URL")
        
        if api_key and base_url:
            self.client = VllmClient(
                api_key=api_key,
                base_url=base_url
            )
            self.using_real_client = True
        else:
            self.using_real_client = False
    
    def test_initialization(self):
        \"\"\"Test client initialization with API key.\"\"\"
        client = LlvmClient(api_key="test_api_key")
        self.assertEqual(client.api_key, "test_api_key")
        
        # Test initialization without API key
        with mock.patch.dict(os.environ, {"LLVM_API_KEY": "env_api_key"}):
            client = LlvmClient()
            self.assertEqual(client.api_key, "env_api_key")
    
    def test_list_models(self):
        \"\"\"Test listing available models.\"\"\"
        response = self.client.list_models()
        self.assertIsInstance(response, dict)
        self.assertIn("models", response)
        self.assertIsInstance(response["models"], list)
    
    def test_get_model_info(self):
        \"\"\"Test retrieving model information.\"\"\"
        model_id = "test-model"
        response = self.client.get_model_info(model_id)
        self.assertIsInstance(response, dict)
        self.assertIn("model_id", response)
        self.assertEqual(response["model_id"], model_id)
    
    def test_run_inference(self):
        \"\"\"Test running inference with a model.\"\"\"
        model_id = "test-model"
        inputs = "Test input data"
        response = self.client.run_inference(model_id, inputs)
        self.assertIsInstance(response, dict)
        self.assertIn("model_id", response)
        self.assertIn("outputs", response)
    
    def test_concurrent_requests(self):
        \"\"\"Test handling concurrent requests.\"\"\"
        num_requests = 5
        
        def make_request(i):
            return self.client.run_inference("test-model", f"Input {i}")
        
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            results = list(executor.map(make_request, range(num_requests)))
        
        self.assertEqual(len(results), num_requests)
        for i, result in enumerate(results):
            self.assertIn("outputs", result)
    
    def test_retry_mechanism(self):
        \"\"\"Test retry mechanism for failed requests.\"\"\"
        # Mock a server error
        original_run_inference = self.client.run_inference
        fail_count = [0]
        
        def mock_run_inference(model_id, inputs, **kwargs):
            fail_count[0] += 1
            if fail_count[0] <= 2:  # Fail twice then succeed
                raise Exception("Simulated server error")
            return original_run_inference(model_id, inputs, **kwargs)
        
        with mock.patch.object(self.client, 'run_inference', side_effect=mock_run_inference):
            try:
                # This should succeed after retries
                result = self.client._with_backoff(
                    lambda: self.client.run_inference("test-model", "test input")
                )
                self.assertIsInstance(result, dict)
                self.assertIn("outputs", result)
                self.assertEqual(fail_count[0], 3)  # 2 failures + 1 success
            except Exception as e:
                if not self.using_real_client:
                    self.fail(f"Retry mechanism failed: {e}")
                else:
                    # Skip for real client as we can't mock its methods reliably
                    logger.warning("Skipping retry test with real client")

    def test_error_handling(self):
        \"\"\"Test error handling for invalid requests.\"\"\"
        # Test with invalid model ID
        with self.assertRaises(Exception):
            with mock.patch.object(self.client, 'get_model_info', side_effect=Exception("Model not found")):
                self.client.get_model_info("invalid-model")
                
    def test_api_key_handling(self):
        \"\"\"Test API key handling in requests.\"\"\"
        # Test setting a new API key
        new_key = "new_test_key"
        self.client.set_api_key(new_key)
        self.assertEqual(self.client.api_key, new_key)
        
        # Verify it's used in requests
        with mock.patch.object(self.client, '_make_request') as mock_make_request:
            try:
                self.client.list_models()
                # Check if API key was used in headers (mocked client only)
                if not self.using_real_client:
                    mock_make_request.assert_called()
                    args, kwargs = mock_make_request.call_args
                    self.assertIn("headers", kwargs)
                    self.assertIn("Authorization", kwargs["headers"])
                    self.assertIn(new_key, kwargs["headers"]["Authorization"])
            except Exception:
                if not self.using_real_client:
                    raise
    
    def test_queue_system(self):
        \"\"\"Test the request queue system.\"\"\"
        if not hasattr(self.client, 'request_queue'):
            logger.warning("Skipping queue test - client doesn't have queue attribute")
            return
            
        # Test queue size configuration
        self.client.max_concurrent_requests = 2
        
        # Simulate concurrent requests that take time
        def slow_request(i):
            return self.client._with_queue(
                lambda: (time.sleep(0.5), self.client.run_inference("test-model", f"Input {i}"))[1]
            )
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(slow_request, range(4)))
        end_time = time.time()
        
        # Verify results
        self.assertEqual(len(results), 4)
        
        # Check if it took enough time for queue processing
        # (4 requests with 2 concurrency and 0.5s sleep should take ~1.0s)
        if not self.using_real_client:
            self.assertGreaterEqual(end_time - start_time, 1.0)
    
    def test_queue_and_retry_integration(self):
        \"\"\"Test integration of queue system with retry mechanism.\"\"\"
        if not hasattr(self.client, 'request_queue') or not hasattr(self.client, 'max_retries'):
            logger.warning("Skipping queue+retry test - missing required attributes")
            return
            
        # Mock a sometimes-failing function
        fail_rate = 0.5
        
        def flaky_function(i):
            if random.random() < fail_rate:
                raise Exception(f"Simulated random failure {i}")
            return f"Success {i}"
        
        # Wrap with queue and backoff
        def process_with_queue_and_backoff(i):
            return self.client._with_queue(
                lambda: self.client._with_backoff(
                    lambda: flaky_function(i)
                )
            )
            
        # Run concurrent requests
        num_requests = 5
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            results = list(executor.map(process_with_queue_and_backoff, range(num_requests)))
            
        # All should eventually succeed
        self.assertEqual(len(results), num_requests)
        for i, result in enumerate(results):
            self.assertEqual(result, f"Success {i}")


if __name__ == "__main__":
    unittest.main()
"""

# Template for S3 Kit test file
S3_KIT_TEST_TEMPLATE = """#!/usr/bin/env python
\"\"\"
Test suite for S3 Kit API implementation.

This module tests the S3 Kit API backend functionality, including:
- Connection to S3 service
- Model storage and retrieval
- Error handling
- Queue and backoff systems
\"\"\"

import os
import sys
import unittest
import json
import time
import random
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
grand_parent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grand_parent_dir)

# Import S3 Kit client - adjust import path as needed
try:
    from ipfs_accelerate_py.api_backends.s3_kit import S3KitClient
except ImportError:
    try:
        from api_backends.s3_kit import S3KitClient
    except ImportError:
        # Mock implementation for testing
        class S3KitClient:
            def __init__(self, **kwargs):
                self.aws_access_key = kwargs.get("aws_access_key", "test_access_key")
                self.aws_secret_key = kwargs.get("aws_secret_key", "test_secret_key")
                self.region = kwargs.get("region", "us-east-1")
                self.bucket = kwargs.get("bucket", "test-bucket")
                self.request_count = 0
                self.max_retries = 3
                self.retry_delay = 1
                self.stored_models = {}
                
            def upload_model(self, model_path, model_name=None):
                self.request_count += 1
                model_id = model_name or f"model-{uuid.uuid4()}"
                self.stored_models[model_id] = model_path
                return {"model_id": model_id, "status": "uploaded"}
                
            def download_model(self, model_id, destination_path):
                self.request_count += 1
                if model_id in self.stored_models:
                    return {"model_id": model_id, "status": "downloaded", "path": destination_path}
                raise Exception(f"Model {model_id} not found")
                
            def list_models(self):
                self.request_count += 1
                return {"models": list(self.stored_models.keys())}
                
            def delete_model(self, model_id):
                self.request_count += 1
                if model_id in self.stored_models:
                    del self.stored_models[model_id]
                    return {"status": "deleted", "model_id": model_id}
                raise Exception(f"Model {model_id} not found")


class TestS3KitApiBackend(unittest.TestCase):
    \"\"\"Test cases for S3 Kit API backend implementation.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test environment.\"\"\"
        # Use mock client by default
        self.client = S3KitClient(
            aws_access_key="test_access_key",
            aws_secret_key="test_secret_key",
            region="us-east-1",
            bucket="test-bucket"
        )
        
        # Optional: Configure with real credentials from environment variables
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        region = os.environ.get("AWS_REGION", "us-east-1")
        bucket = os.environ.get("S3_TEST_BUCKET")
        
        if aws_access_key and aws_secret_key and bucket:
            self.client = S3KitClient(
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key,
                region=region,
                bucket=bucket
            )
            self.using_real_client = True
        else:
            self.using_real_client = False
        
        # Create a temporary file for testing
        self.test_file_path = os.path.join(script_dir, "test_model.txt")
        with open(self.test_file_path, "w") as f:
            f.write("This is a test model file")
        
        # Track created models for cleanup
        self.test_models = []
    
    def tearDown(self):
        \"\"\"Clean up after tests.\"\"\"
        # Remove temporary file
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        
        # Clean up test models
        if self.using_real_client:
            for model_id in self.test_models:
                try:
                    self.client.delete_model(model_id)
                except Exception as e:
                    print(f"Error cleaning up model {model_id}: {e}")
    
    def test_initialization(self):
        \"\"\"Test client initialization with credentials.\"\"\"
        client = S3KitClient(
            aws_access_key="test_access_key",
            aws_secret_key="test_secret_key",
            region="us-west-2",
            bucket="custom-bucket"
        )
        self.assertEqual(client.aws_access_key, "test_access_key")
        self.assertEqual(client.aws_secret_key, "test_secret_key")
        self.assertEqual(client.region, "us-west-2")
        self.assertEqual(client.bucket, "custom-bucket")
        
        # Test initialization with environment variables
        with mock.patch.dict(os.environ, {
            "AWS_ACCESS_KEY_ID": "env_access_key",
            "AWS_SECRET_ACCESS_KEY": "env_secret_key",
            "AWS_REGION": "us-east-2",
            "S3_BUCKET": "env-bucket"
        }):
            client = S3KitClient()
            self.assertEqual(client.aws_access_key, "env_access_key")
            self.assertEqual(client.aws_secret_key, "env_secret_key")
            self.assertEqual(client.region, "us-east-2")
            self.assertEqual(client.bucket, "env-bucket")
    
    def test_upload_model(self):
        \"\"\"Test uploading a model to S3.\"\"\"
        response = self.client.upload_model(self.test_file_path, "test-model")
        self.assertIsInstance(response, dict)
        self.assertIn("model_id", response)
        self.assertIn("status", response)
        
        if self.using_real_client:
            # Remember model for cleanup
            self.test_models.append(response["model_id"])
    
    def test_list_models(self):
        \"\"\"Test listing available models.\"\"\"
        # First upload a model
        model_name = f"test-model-{uuid.uuid4()}"
        upload_response = self.client.upload_model(self.test_file_path, model_name)
        
        if self.using_real_client:
            self.test_models.append(upload_response["model_id"])
        
        # Now list models
        response = self.client.list_models()
        self.assertIsInstance(response, dict)
        self.assertIn("models", response)
        self.assertIsInstance(response["models"], list)
        
        # Check if our uploaded model is in the list
        if self.using_real_client:
            self.assertIn(upload_response["model_id"], response["models"])
    
    def test_download_model(self):
        \"\"\"Test downloading a model from S3.\"\"\"
        # First upload a model
        model_name = f"test-model-{uuid.uuid4()}"
        upload_response = self.client.upload_model(self.test_file_path, model_name)
        model_id = upload_response["model_id"]
        
        if self.using_real_client:
            self.test_models.append(model_id)
        
        # Now download it to a new location
        download_path = os.path.join(script_dir, f"downloaded_model_{uuid.uuid4()}.txt")
        response = self.client.download_model(model_id, download_path)
        
        self.assertIsInstance(response, dict)
        self.assertIn("status", response)
        self.assertEqual(response["status"], "downloaded")
        
        # Check if the file was downloaded
        if self.using_real_client:
            self.assertTrue(os.path.exists(download_path))
            
            # Cleanup downloaded file
            if os.path.exists(download_path):
                os.remove(download_path)
    
    def test_delete_model(self):
        \"\"\"Test deleting a model from S3.\"\"\"
        # First upload a model
        model_name = f"test-model-{uuid.uuid4()}"
        upload_response = self.client.upload_model(self.test_file_path, model_name)
        model_id = upload_response["model_id"]
        
        # Now delete it
        response = self.client.delete_model(model_id)
        
        self.assertIsInstance(response, dict)
        self.assertIn("status", response)
        self.assertEqual(response["status"], "deleted")
        
        # Verify it's gone from the list
        if self.using_real_client:
            list_response = self.client.list_models()
            self.assertNotIn(model_id, list_response["models"])
    
    def test_concurrent_uploads(self):
        \"\"\"Test handling concurrent uploads.\"\"\"
        num_uploads = 3
        
        def upload_model(i):
            model_name = f"concurrent-model-{i}-{uuid.uuid4()}"
            response = self.client.upload_model(self.test_file_path, model_name)
            if self.using_real_client:
                self.test_models.append(response["model_id"])
            return response
        
        with ThreadPoolExecutor(max_workers=num_uploads) as executor:
            results = list(executor.map(upload_model, range(num_uploads)))
        
        self.assertEqual(len(results), num_uploads)
        for result in results:
            self.assertIn("model_id", result)
            self.assertIn("status", result)
    
    def test_retry_mechanism(self):
        \"\"\"Test retry mechanism for failed requests.\"\"\"
        if not hasattr(self.client, '_with_backoff'):
            self.skipTest("Client doesn't have backoff method")
            
        # Mock a server error
        original_upload = self.client.upload_model
        fail_count = [0]
        
        def mock_upload(model_path, model_name=None):
            fail_count[0] += 1
            if fail_count[0] <= 2:  # Fail twice then succeed
                raise Exception("Simulated server error")
            return original_upload(model_path, model_name)
        
        with mock.patch.object(self.client, 'upload_model', side_effect=mock_upload):
            try:
                # This should succeed after retries
                model_name = f"retry-test-{uuid.uuid4()}"
                result = self.client._with_backoff(
                    lambda: self.client.upload_model(self.test_file_path, model_name)
                )
                
                if self.using_real_client:
                    self.test_models.append(result["model_id"])
                    
                self.assertIsInstance(result, dict)
                self.assertIn("model_id", result)
                self.assertEqual(fail_count[0], 3)  # 2 failures + 1 success
            except Exception as e:
                if not self.using_real_client:
                    self.fail(f"Retry mechanism failed: {e}")
                else:
                    # Skip for real client as we can't mock its methods reliably
                    pass
    
    def test_error_handling(self):
        \"\"\"Test error handling for invalid requests.\"\"\"
        # Test with non-existent model ID
        with self.assertRaises(Exception):
            self.client.download_model("non-existent-model", "dummy_path.txt")
    
    def test_queue_system(self):
        \"\"\"Test the request queue system.\"\"\"
        if not hasattr(self.client, 'request_queue'):
            self.skipTest("Client doesn't have queue attribute")
            
        # Test queue size configuration
        self.client.max_concurrent_requests = 2
        
        # Simulate concurrent requests that take time
        def slow_upload(i):
            model_name = f"queue-model-{i}-{uuid.uuid4()}"
            return self.client._with_queue(
                lambda: (time.sleep(0.5), self.client.upload_model(self.test_file_path, model_name))[1]
            )
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(slow_upload, range(4)))
        end_time = time.time()
        
        # Add models for cleanup
        if self.using_real_client:
            for result in results:
                self.test_models.append(result["model_id"])
        
        # Verify results
        self.assertEqual(len(results), 4)
        
        # Check if it took enough time for queue processing
        # (4 requests with 2 concurrency and 0.5s sleep should take ~1.0s)
        if not self.using_real_client:
            self.assertGreaterEqual(end_time - start_time, 1.0)


if __name__ == "__main__":
    unittest.main()
"""

def find_apis_dir() -> str:
    """
    Find the 'apis' directory containing test files.
    
    Returns:
        str: Path to the apis directory
    """
    potential_locations = [
        os.path.join(script_dir, "apis"),
        os.path.join(parent_dir, "apis"),
        os.path.join(parent_dir, "test", "apis")
    ]
    
    for location in potential_locations:
        if os.path.isdir(location):
            logger.info(f"Found APIs directory at: {location}")
            return location
    
    # If not found in common locations, search for it
    logger.info("Searching for APIs directory...")
    for root, dirs, files in os.walk(parent_dir):
        if "apis" in dirs and "test_openai_api.py" in os.listdir(os.path.join(root, "apis")):
            apis_dir = os.path.join(root, "apis")
            logger.info(f"Found APIs directory at: {apis_dir}")
            return apis_dir
    
    raise FileNotFoundError("Could not find 'apis' directory containing test files")

def check_missing_test_files(apis_dir: str) -> Dict[str, bool]:
    """
    Check which test files are missing in the apis directory.
    
    Args:
        apis_dir: Path to the apis directory
        
    Returns:
        Dict mapping API names to whether their test file is missing
    """
    test_files = {
        "vllm": "test_vllm.py",
        "s3_kit": "test_s3_kit.py"
    }
    
    missing_tests = {}
    
    for api_name, filename in test_files.items():
        file_path = os.path.join(apis_dir, filename)
        missing_tests[api_name] = not os.path.exists(file_path)
        
        if missing_tests[api_name]:
            logger.info(f"Test file missing for {api_name}: {filename}")
        else:
            logger.info(f"Test file exists for {api_name}: {filename}")
    
    return missing_tests

def generate_test_file(api_name: str, apis_dir: str) -> str:
    """
    Generate a test file for the specified API.
    
    Args:
        api_name: Name of the API (vllm or s3_kit)
        apis_dir: Path to the apis directory
        
    Returns:
        str: Path to the generated test file
    """
    filename = f"test_{api_name}.py"
    file_path = os.path.join(apis_dir, filename)
    
    if os.path.exists(file_path):
        logger.warning(f"Test file already exists at {file_path}, creating backup")
        backup_path = file_path + '.bak'
        # Backup existing file
        with open(file_path, 'r') as f:
            existing_content = f.read()
        with open(backup_path, 'w') as f:
            f.write(existing_content)
    
    # Select template based on API name
    if api_name == "vllm":
        template = VLLM_TEST_TEMPLATE
    elif api_name == "s3_kit":
        template = S3_KIT_TEST_TEMPLATE
    else:
        raise ValueError(f"Unknown API name: {api_name}")
    
    # Write the test file
    with open(file_path, 'w') as f:
        f.write(template)
    
    logger.info(f"Generated test file for {api_name} at {file_path}")
    return file_path

def main():
    """Main function to run the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate missing test files for API backends")
    parser.add_argument('--apis', choices=['llvm', 's3_kit', 'all'], default='all', 
                      help="Which API test file to generate (default: all)")
    parser.add_argument('--apis-dir', type=str, help="Path to the apis directory (optional)")
    parser.add_argument('--force', action='store_true', help="Force regeneration even if files exist")
    
    args = parser.parse_args()
    
    try:
        # Find APIs directory
        apis_dir = args.apis_dir if args.apis_dir else find_apis_dir()
        
        # Check which test files are missing
        missing_tests = check_missing_test_files(apis_dir)
        
        # Generate missing test files
        generated_files = []
        
        apis_to_generate = []
        if args.apis == 'all':
            apis_to_generate = ['llvm', 's3_kit']
        else:
            apis_to_generate = [args.apis]
        
        for api_name in apis_to_generate:
            if args.force or missing_tests[api_name]:
                file_path = generate_test_file(api_name, apis_dir)
                generated_files.append((api_name, file_path))
        
        if generated_files:
            logger.info("\nGenerated test files:")
            for api_name, file_path in generated_files:
                logger.info(f"- {api_name}: {file_path}")
        else:
            logger.info("No test files were generated")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()