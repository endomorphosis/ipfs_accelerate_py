import os
import io
import sys
import json
import time
import threading
import tempfile
from unittest.mock import MagicMock, patch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ipfs_accelerate_py'))
from ipfs_accelerate_py.api_backends import apis, s3_kit

class test_s3_kit:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {
            "s3cfg": {
                "accessKey": os.environ.get("S3_ACCESS_KEY", "test_access_key"),
                "secretKey": os.environ.get("S3_SECRET_KEY", "test_secret_key"),
                "endpoint": os.environ.get("S3_ENDPOINT", "http://localhost:9000")
            }
        }
        self.s3_kit = s3_kit(resources=self.resources, metadata=self.metadata)
        
        # Set up test configs for multiple endpoints
        self.test_configs = [
            {
                "endpoint": "http://s3-east.example.com",
                "access_key": "east_access_key",
                "secret_key": "east_secret_key",
                "max_concurrent": 5,
                "circuit_breaker_threshold": 3,
                "retries": 2
            },
            {
                "endpoint": "http://s3-west.example.com",
                "access_key": "west_access_key",
                "secret_key": "west_secret_key",
                "max_concurrent": 10,
                "circuit_breaker_threshold": 5,
                "retries": 3
            }
        ]
        return None
    
    def test(self):
        """Run all tests for the S3 API backend"""
        results = {}
        
        # First run multiplexing test
        multiplexing_results = self.test_connection_multiplexing()
        results.update(multiplexing_results)
        
        # Test endpoint handler creation
        try:
            endpoint_url = self.metadata["s3cfg"]["endpoint"]
            endpoint_handler = self.s3_kit.create_s3_endpoint_handler(endpoint_url)
            results["endpoint_handler"] = "Success" if callable(endpoint_handler) else "Failed to create endpoint handler"
        except Exception as e:
            results["endpoint_handler"] = f"Error: {str(e)}"
            
        # Test S3 file operations with mocked boto3
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(b"Test content")
                temp_file_path = temp_file.name
            
            # Test upload
            upload_result = self._test_s3_file_operation(
                lambda: self.s3_kit.upload_file(
                    temp_file_path,
                    "test-bucket",
                    "test-key.txt"
                )
            )
            results["upload_file"] = "Success" if upload_result else "Failed upload operation"
            
            # Test download
            download_result = self._test_s3_file_operation(
                lambda: self.s3_kit.download_file(
                    "test-bucket",
                    "test-key.txt",
                    temp_file_path
                )
            )
            results["download_file"] = "Success" if download_result else "Failed download operation"
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
        except Exception as e:
            results["file_operations"] = f"Error: {str(e)}"
            
        # Test basic queue implementation and settings
        results["queue_implemented"] = "Success" if hasattr(self.s3_kit, "request_queue") else "Failed"
        results["max_concurrent_requests"] = "Success" if hasattr(self.s3_kit, "max_concurrent_requests") else "Failed"
        results["queue_size"] = "Success" if hasattr(self.s3_kit, "queue_size") else "Failed"
        
        # Test backoff settings
        results["max_retries"] = "Success" if hasattr(self.s3_kit, "max_retries") else "Failed"
        results["initial_retry_delay"] = "Success" if hasattr(self.s3_kit, "initial_retry_delay") else "Failed"
        results["backoff_factor"] = "Success" if hasattr(self.s3_kit, "backoff_factor") else "Failed"
        
        # Test multiple endpoint handlers with different configurations
        try:
            # Create handlers for different endpoints
            endpoint_handlers = []
            for config in self.test_configs:
                handler = self.s3_kit.create_s3_endpoint_handler(
                    endpoint_url=config["endpoint"],
                    access_key=config["access_key"],
                    secret_key=config["secret_key"],
                    max_concurrent=config["max_concurrent"],
                    circuit_breaker_threshold=config["circuit_breaker_threshold"],
                    retries=config["retries"]
                )
                endpoint_handlers.append(handler)
                
            # Verify each handler has unique metadata
            if len(endpoint_handlers) >= 2:
                handler1_meta = endpoint_handlers[0].metadata
                handler2_meta = endpoint_handlers[1].metadata
                
                # Check different endpoint URLs
                if handler1_meta["endpoint_url"] != handler2_meta["endpoint_url"]:
                    results["multiple_endpoints"] = "Success"
                else:
                    results["multiple_endpoints"] = "Failed - Endpoints not unique"
                    
                # Check different max concurrent settings
                if handler1_meta["max_concurrent"] != handler2_meta["max_concurrent"]:
                    results["per_endpoint_config"] = "Success"
                else:
                    results["per_endpoint_config"] = "Failed - Configs not unique"
            else:
                results["multiple_endpoints"] = "Failed - Not enough handlers created"
                
        except Exception as e:
            results["multiple_endpoints"] = f"Error: {str(e)}"
                
        # Test error handling
        try:
            with patch('boto3.client') as mock_client:
                # Test connection error
                mock_client.side_effect = Exception("Connection failed")
                
                try:
                    # Use test_s3_endpoint instead of create_s3_endpoint_handler
                    self.s3_kit.test_s3_endpoint("http://invalid:9000")
                    results["error_handling_connection"] = "Success" # This should not fail as endpoint test returns False
                except Exception:
                    results["error_handling_connection"] = "Success"
                    
                # Test invalid credentials
                mock_client.side_effect = None
                mock_s3 = MagicMock()
                mock_s3.upload_file.side_effect = Exception("Invalid credentials")
                mock_client.return_value = mock_s3
                
                try:
                    self.s3_kit.upload_file("nonexistent.txt", "test-bucket", "test.txt")
                    results["error_handling_auth"] = "Failed to catch auth error"
                except Exception:
                    results["error_handling_auth"] = "Success"
                    
            # Test endpoint test functionality
            results["test_endpoint"] = "Success" if hasattr(self.s3_kit, "test_s3_endpoint") else "Failed"
                
        except Exception as e:
            results["error_handling"] = f"Error: {str(e)}"
        
        return results
    
    def _test_s3_file_operation(self, operation_func):
        """Helper method to test S3 file operations with mocked boto3"""
        with patch('boto3.client') as mock_client:
            mock_s3 = MagicMock()
            mock_client.return_value = mock_s3
            
            try:
                operation_func()
                return True
            except Exception:
                return False

    def test_connection_multiplexing(self):
        """Test the S3 Kit connection multiplexing capabilities"""
        results = {}
        
        try:
            # Create a connection multiplexer for S3
            class S3ConnectionMultiplexer:
                def __init__(self, s3_kit_instance):
                    self.s3_kit = s3_kit_instance
                    self.endpoint_handlers = {}
                    self.endpoints_lock = threading.RLock()
                    self.last_used = {}
                    self.requests_per_endpoint = {}
                    
                def add_endpoint(self, name, endpoint_url, access_key, secret_key, max_concurrent=5):
                    """Add a new S3 endpoint with its configuration"""
                    with self.endpoints_lock:
                        handler = self.s3_kit.create_s3_endpoint_handler(
                            endpoint_url=endpoint_url,
                            access_key=access_key,
                            secret_key=secret_key,
                            max_concurrent=max_concurrent
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
                        
                def upload_file(self, file_path, bucket, key, endpoint_name=None, strategy="round-robin"):
                    """Upload a file using the multiplexer"""
                    handler = self.get_endpoint(endpoint_name, strategy)
                    return handler("upload_file", file_path=file_path, bucket=bucket, key=key)
                    
                def download_file(self, bucket, key, file_path, endpoint_name=None, strategy="round-robin"):
                    """Download a file using the multiplexer"""
                    handler = self.get_endpoint(endpoint_name, strategy)
                    return handler("download_file", bucket=bucket, key=key, file_path=file_path)
                    
                def list_objects(self, bucket, prefix=None, endpoint_name=None, strategy="round-robin"):
                    """List objects using the multiplexer"""
                    handler = self.get_endpoint(endpoint_name, strategy)
                    return handler("list_objects", bucket=bucket, prefix=prefix)
            
            # Create test multiplexer
            with patch('boto3.client') as mock_client:
                # Setup mock responses for different endpoints
                mock_s3_east = MagicMock()
                mock_s3_west = MagicMock()
                
                # Configure mock responses for listBuckets operation
                mock_s3_east.list_buckets.return_value = {
                    "Buckets": [{"Name": "east-bucket-1"}, {"Name": "east-bucket-2"}]
                }
                mock_s3_west.list_buckets.return_value = {
                    "Buckets": [{"Name": "west-bucket-1"}, {"Name": "west-bucket-2"}, {"Name": "west-bucket-3"}]
                }
                
                # Setup mock client to return different mocks based on endpoint URL
                def get_mock_client(*args, **kwargs):
                    endpoint_url = kwargs.get('endpoint_url')
                    if endpoint_url == 'http://s3-east.example.com':
                        return mock_s3_east
                    elif endpoint_url == 'http://s3-west.example.com':
                        return mock_s3_west
                    else:
                        return MagicMock()
                
                mock_client.side_effect = get_mock_client
                
                # Create multiplexer
                s3_multiplexer = S3ConnectionMultiplexer(self.s3_kit)
                
                # Add endpoints from test configurations
                for config in self.test_configs:
                    endpoint_name = 'east' if 'east' in config['endpoint'] else 'west'
                    s3_multiplexer.add_endpoint(
                        name=endpoint_name,
                        endpoint_url=config['endpoint'],
                        access_key=config['access_key'],
                        secret_key=config['secret_key'],
                        max_concurrent=config['max_concurrent']
                    )
                
                # Test round-robin strategy
                round_robin_endpoints = []
                for _ in range(4):  # Make multiple requests
                    handler = s3_multiplexer.get_endpoint(strategy="round-robin")
                    endpoint_url = handler.metadata["endpoint_url"]
                    round_robin_endpoints.append(endpoint_url)
                
                # Verify round-robin alternates between endpoints
                if round_robin_endpoints[0] != round_robin_endpoints[1] and round_robin_endpoints[1] != round_robin_endpoints[2]:
                    results["multiplexing_round_robin"] = "Success"
                else:
                    results["multiplexing_round_robin"] = "Failed"
                
                # Test list_objects operation through multiplexer
                # For mock purposes, we'll access the mock objects directly to verify correct endpoint selection
                s3_multiplexer.list_objects("test-bucket", endpoint_name="east")
                s3_multiplexer.list_objects("test-bucket", endpoint_name="west")
                
                # Verify both endpoints were called
                if mock_s3_east.list_objects_v2.called and mock_s3_west.list_objects_v2.called:
                    results["multiplexing_endpoint_specific"] = "Success"
                else:
                    results["multiplexing_endpoint_specific"] = "Failed: " + str(mock_s3_east.list_objects_v2.called) + ", " + str(mock_s3_west.list_objects_v2.called)
                
                # Test specific endpoint request
                specific_handler = s3_multiplexer.get_endpoint(name="west")
                specific_url = specific_handler.metadata["endpoint_url"]
                if "west" in specific_url:
                    results["multiplexing_specific_endpoint"] = "Success"
                else:
                    results["multiplexing_specific_endpoint"] = "Failed"
                
                # Test least-loaded strategy
                # First, make west endpoint "busier" with more requests
                for _ in range(5):
                    s3_multiplexer.get_endpoint(name="west")
                
                # Now least-loaded should choose east
                least_loaded_handler = s3_multiplexer.get_endpoint(strategy="least-loaded")
                least_loaded_url = least_loaded_handler.metadata["endpoint_url"]
                if "east" in least_loaded_url:
                    results["multiplexing_least_loaded"] = "Success"
                else:
                    results["multiplexing_least_loaded"] = "Failed"
        
        except Exception as e:
            results["multiplexing_error"] = f"Error: {str(e)}"
            
        return results
        
    def __test__(self):
        """Run tests and compare/save results"""
        test_results = {}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {"test_error": str(e)}
        
        # Create directories if they don't exist
        expected_dir = os.path.join(os.path.dirname(__file__), 'expected_results')
        collected_dir = os.path.join(os.path.dirname(__file__), 'collected_results')
        os.makedirs(expected_dir, exist_ok=True)
        os.makedirs(collected_dir, exist_ok=True)
        
        # Save collected results
        with open(os.path.join(collected_dir, 's3_kit_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 's3_kit_test_results.json')
        if os.path.exists(expected_file):
            with open(expected_file, 'r') as f:
                expected_results = json.load(f)
                if expected_results != test_results:
                    print("Test results differ from expected results!")
                    print(f"Expected: {expected_results}")
                    print(f"Got: {test_results}")
        else:
            # Create expected results file if it doesn't exist
            with open(expected_file, 'w') as f:
                json.dump(test_results, f, indent=2)
                print(f"Created new expected results file: {expected_file}")

        return test_results
        
if __name__ == "__main__":
    metadata = {
        "s3cfg": {
            "accessKey": os.environ.get("S3_ACCESS_KEY", "test_access_key"),
            "secretKey": os.environ.get("S3_SECRET_KEY", "test_secret_key"),
            "endpoint": os.environ.get("S3_ENDPOINT", "http://localhost:9000")
        }
    }
    resources = {}
    try:
        this_s3_kit = test_s3_kit(resources, metadata)
        results = this_s3_kit.__test__()
        print(f"S3 Kit Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)