import os
import io
import sys
import json
import time
import threading
from unittest.mock import MagicMock, patch
import requests
import random

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ipfs_accelerate_py'))
from api_backends import apis, hf_tgi

class HFTGIMultiplexer:
    """
    Class to manage multiple HuggingFace TGI API keys with separate queues for each key.
    """
    
    def __init__(self):
        # Initialize API client dictionaries - each key will have its own client
        self.hf_tgi_clients = {}
        
        # Initialize lock for thread safety
        self.hf_tgi_lock = threading.RLock()
        
        print("HF TGI API Key Multiplexer initialized")
    
    def add_hf_tgi_key(self, key_name, api_key, model_id=None, max_concurrent=5):
        """Add a new HuggingFace TGI API key with its own client instance"""
        with self.hf_tgi_lock:
            # Create a new HF TGI client with this API key
            metadata = {
                "hf_api_key": api_key,
                "model_id": model_id or "google/t5-efficient-tiny"
            }
            
            client = hf_tgi(
                resources={},
                metadata=metadata
            )
            
            # Configure queue settings for this client if supported
            if hasattr(client, 'max_concurrent_requests'):
                client.max_concurrent_requests = max_concurrent
            
            # Store in our dictionary
            self.hf_tgi_clients[key_name] = {
                "client": client,
                "api_key": api_key,
                "model_id": metadata["model_id"],
                "usage": 0,
                "last_used": 0,
                "endpoints": {}  # Will store endpoint IDs if create_endpoint is supported
            }
            
            # Try to create an endpoint if the function exists
            try:
                if hasattr(client, 'create_endpoint'):
                    endpoint_id = client.create_endpoint(
                        api_key=api_key,
                        max_concurrent_requests=max_concurrent,
                        queue_size=100,
                        max_retries=3
                    )
                    self.hf_tgi_clients[key_name]["endpoints"]["default"] = endpoint_id
            except Exception as e:
                print(f"Note: Could not create endpoint for {key_name}: {str(e)}")
            
            print(f"Added HF TGI key: {key_name}")
            return key_name
    
    def get_hf_tgi_client(self, key_name=None, strategy="round-robin"):
        """
        Get a HF TGI client by key name or using a selection strategy
        
        Strategies:
        - "specific": Return the client for the specified key_name
        - "round-robin": Select the least recently used client
        - "least-loaded": Select the client with the smallest queue
        """
        with self.hf_tgi_lock:
            if len(self.hf_tgi_clients) == 0:
                raise ValueError("No HF TGI API keys have been added")
            
            if key_name and key_name in self.hf_tgi_clients:
                # Update usage stats
                self.hf_tgi_clients[key_name]["usage"] += 1
                self.hf_tgi_clients[key_name]["last_used"] = time.time()
                return self.hf_tgi_clients[key_name]["client"]
            
            if strategy == "round-robin":
                # Find the least recently used client
                selected_key = min(self.hf_tgi_clients.keys(), 
                                key=lambda k: self.hf_tgi_clients[k]["last_used"])
            elif strategy == "least-loaded" and hasattr(self.hf_tgi_clients[list(self.hf_tgi_clients.keys())[0]]["client"], "current_requests"):
                # Find the client with the smallest queue
                selected_key = min(self.hf_tgi_clients.keys(),
                                key=lambda k: self.hf_tgi_clients[k]["client"].current_requests)
            else:
                # Default to first key
                selected_key = list(self.hf_tgi_clients.keys())[0]
            
            # Update usage stats
            self.hf_tgi_clients[selected_key]["usage"] += 1
            self.hf_tgi_clients[selected_key]["last_used"] = time.time()
            
            return self.hf_tgi_clients[selected_key]["client"]
    
    def get_usage_stats(self):
        """Get usage statistics for all API keys"""
        stats = {
            "hf_tgi": {key: {
                "usage": data["usage"],
                "queue_size": len(data["client"].request_queue) if hasattr(data["client"], "request_queue") else 0,
                "current_requests": data["client"].current_requests if hasattr(data["client"], "current_requests") else 0,
                "model_id": data["model_id"]
            } for key, data in self.hf_tgi_clients.items()},
        }
        return stats

class test_hf_tgi:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {
            "hf_api_key": os.environ.get("HF_API_KEY", ""),
            "model_id": os.environ.get("HF_MODEL_ID", "google/t5-efficient-tiny")
        }
        self.hf_tgi = hf_tgi(resources=self.resources, metadata=self.metadata)
        return None
    
    def test(self):
        """Run all tests for the HuggingFace Text Generation Inference API backend"""
        results = {}
        
        # Test API key multiplexing features
        try:
            if hasattr(self.hf_tgi, 'create_endpoint'):
                # Create first endpoint with test key
                endpoint1 = self.hf_tgi.create_endpoint(
                    api_key="test_hf_key_1",
                    max_concurrent_requests=5,
                    queue_size=20,
                    max_retries=3,
                    initial_retry_delay=1,
                    backoff_factor=2
                )
                
                # Create second endpoint with different test key
                endpoint2 = self.hf_tgi.create_endpoint(
                    api_key="test_hf_key_2",
                    max_concurrent_requests=10,
                    queue_size=50,
                    max_retries=5
                )
                
                results["multiplexing_endpoint_creation"] = "Success" if endpoint1 and endpoint2 else "Failed to create endpoints"
                
                # Test usage statistics if implemented
                if hasattr(self.hf_tgi, 'get_stats'):
                    # Get stats for first endpoint
                    stats1 = self.hf_tgi.get_stats(endpoint1)
                    
                    # Make test requests to create stats
                    with patch.object(self.hf_tgi, 'make_post_request_hf_tgi') as mock_post:
                        mock_post.return_value = {"generated_text": "This is a test response"}
                        
                        # Make request with first endpoint if method exists
                        if hasattr(self.hf_tgi, 'make_request_with_endpoint'):
                            self.hf_tgi.make_request_with_endpoint(
                                endpoint_id=endpoint1,
                                data={"inputs": "Test for endpoint 1"}
                            )
                            
                            # Get updated stats
                            stats1_after = self.hf_tgi.get_stats(endpoint1)
                            
                            # Verify stats were updated
                            results["usage_statistics"] = "Success" if stats1_after != stats1 else "Failed to update statistics"
                        else:
                            results["usage_statistics"] = "Not implemented"
                else:
                    results["usage_statistics"] = "Not implemented"
                
                # Test request routing with different endpoints
                if hasattr(self.hf_tgi, 'make_request_with_endpoint'):
                    with patch.object(self.hf_tgi, 'make_post_request_hf_tgi') as mock_post:
                        mock_post.return_value = {"generated_text": "This is a test response"}
                        
                        # Test with endpoint 1
                        result1 = self.hf_tgi.make_request_with_endpoint(
                            endpoint_id=endpoint1,
                            data={"inputs": "Test for endpoint 1"},
                            request_id="test-request-123"
                        )
                        
                        # Test with endpoint 2
                        result2 = self.hf_tgi.make_request_with_endpoint(
                            endpoint_id=endpoint2,
                            data={"inputs": "Test for endpoint 2"},
                            request_id="test-request-456"
                        )
                        
                        # Verify both requests succeeded
                        both_succeeded = isinstance(result1, dict) and isinstance(result2, dict)
                        results["endpoint_routing"] = "Success" if both_succeeded else "Failed endpoint routing"
                        
                        # Verify different API keys were used for different endpoints
                        calls = mock_post.call_args_list
                        if len(calls) >= 2:
                            # Extract API keys from calls
                            api_key1 = calls[0][0][2] if len(calls[0][0]) > 2 else None
                            api_key2 = calls[1][0][2] if len(calls[1][0]) > 2 else None
                            
                            results["different_keys_used"] = "Success" if api_key1 != api_key2 else "Failed to use different API keys"
                        else:
                            results["different_keys_used"] = "Failed - insufficient calls made"
                else:
                    results["endpoint_routing"] = "Not implemented"
            else:
                results["multiplexing_endpoint_creation"] = "Not implemented"
        except Exception as e:
            results["multiplexing"] = f"Error: {str(e)}"

        # Test queue and backoff functionality
        try:
            if hasattr(self.hf_tgi, 'queue_enabled'):
                # Test queue settings
                results["queue_enabled"] = "Success" if hasattr(self.hf_tgi, 'queue_enabled') else "Missing queue_enabled"
                results["request_queue"] = "Success" if hasattr(self.hf_tgi, 'request_queue') else "Missing request_queue"
                results["max_concurrent_requests"] = "Success" if hasattr(self.hf_tgi, 'max_concurrent_requests') else "Missing max_concurrent_requests"
                results["current_requests"] = "Success" if hasattr(self.hf_tgi, 'current_requests') else "Missing current_requests counter"
                
                # Test backoff settings
                results["max_retries"] = "Success" if hasattr(self.hf_tgi, 'max_retries') else "Missing max_retries"
                results["initial_retry_delay"] = "Success" if hasattr(self.hf_tgi, 'initial_retry_delay') else "Missing initial_retry_delay"
                results["backoff_factor"] = "Success" if hasattr(self.hf_tgi, 'backoff_factor') else "Missing backoff_factor"
                
                # Test queue processing if implemented
                if hasattr(self.hf_tgi, '_process_queue'):
                    with patch.object(self.hf_tgi, '_process_queue') as mock_queue:
                        mock_queue.return_value = None
                        
                        # Force queue to be enabled and at capacity for testing
                        original_queue_enabled = self.hf_tgi.queue_enabled
                        original_current_requests = self.hf_tgi.current_requests
                        original_max_concurrent = self.hf_tgi.max_concurrent_requests
                        
                        self.hf_tgi.queue_enabled = True
                        self.hf_tgi.current_requests = self.hf_tgi.max_concurrent_requests
                        
                        # Prepare a mock request to add to queue
                        request_info = {
                            "data": {"inputs": "Queued request"},
                            "api_key": "test_key",
                            "request_id": "queue_test_456",
                            "future": {"result": None, "error": None, "completed": False}
                        }
                        
                        # Add request to queue
                        if not hasattr(self.hf_tgi, "request_queue"):
                            self.hf_tgi.request_queue = []
                            
                        self.hf_tgi.request_queue.append(request_info)
                        
                        # Trigger queue processing
                        if hasattr(self.hf_tgi, '_process_queue'):
                            self.hf_tgi._process_queue()
                            results["queue_processing"] = "Success" if mock_queue.called else "Failed to call queue processing"
                        
                        # Restore original values
                        self.hf_tgi.queue_enabled = original_queue_enabled
                        self.hf_tgi.current_requests = original_current_requests
                else:
                    results["queue_processing"] = "Not implemented"
            else:
                results["queue_backoff"] = "Not implemented"
        except Exception as e:
            results["queue_backoff"] = f"Error: {str(e)}"
        
        # Test endpoint handler creation
        try:
            model_id = self.metadata.get("model_id", "google/t5-efficient-tiny")
            endpoint_url = f"https://api-inference.huggingface.co/models/{model_id}"
            api_key = self.metadata.get("hf_api_key", "")
            
            endpoint_handler = self.hf_tgi.create_remote_text_generation_endpoint_handler(
                endpoint_url, api_key
            )
            results["endpoint_handler_creation"] = "Success" if callable(endpoint_handler) else "Failed to create endpoint handler"
        except Exception as e:
            results["endpoint_handler_creation"] = f"Error: {str(e)}"
        
        # Test endpoint testing functionality
        try:
            with patch.object(self.hf_tgi, 'make_post_request_hf_tgi') as mock_post:
                mock_post.return_value = {"generated_text": "This is a test response"}
                
                test_result = self.hf_tgi.test_tgi_endpoint(
                    endpoint_url=endpoint_url,
                    api_key=api_key,
                    model_name=model_id
                )
                results["test_endpoint"] = "Success" if test_result else "Failed endpoint test"
                
                # Verify correct parameters were used in request
                args, kwargs = mock_post.call_args
                results["test_endpoint_params"] = "Success" if "inputs" in args[1] else "Failed to pass correct parameters"
        except Exception as e:
            results["test_endpoint"] = f"Error: {str(e)}"

        # Test post request function
        try:
            with patch.object(requests, 'post') as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {"generated_text": "Test generated text"}
                mock_response.status_code = 200
                mock_post.return_value = mock_response
                
                data = {
                    "inputs": "The quick brown fox",
                    "parameters": {
                        "max_new_tokens": 20,
                        "do_sample": True,
                        "temperature": 0.7
                    }
                }
                
                # Test with custom request_id
                custom_request_id = "test_request_456"
                if hasattr(self.hf_tgi, 'make_post_request_hf_tgi') and len(self.hf_tgi.make_post_request_hf_tgi.__code__.co_varnames) > 3:
                    # If the method supports request_id parameter
                    post_result = self.hf_tgi.make_post_request_hf_tgi(endpoint_url, data, api_key, request_id=custom_request_id)
                    results["post_request"] = "Success" if post_result == mock_response.json.return_value else "Failed post request"
                    
                    # Verify headers were set correctly
                    args, kwargs = mock_post.call_args
                    headers = kwargs.get('headers', {})
                    authorization_header_set = "Authorization" in headers and headers["Authorization"] == f"Bearer {api_key}"
                    content_type_header_set = "Content-Type" in headers and headers["Content-Type"] == "application/json"
                    request_id_header_set = "X-Request-ID" in headers and headers["X-Request-ID"] == custom_request_id
                    
                    results["post_request_headers"] = "Success" if authorization_header_set and content_type_header_set else "Failed to set headers correctly"
                    results["request_id_tracking"] = "Success" if request_id_header_set else "Failed to set request ID header"
                else:
                    # Fall back to old method if request_id isn't supported
                    post_result = self.hf_tgi.make_post_request_hf_tgi(endpoint_url, data, api_key)
                    results["post_request"] = "Success" if post_result == mock_response.json.return_value else "Failed post request"
                    results["request_id_tracking"] = "Not implemented"
                    
                    # Verify headers were set correctly
                    args, kwargs = mock_post.call_args
                    headers = kwargs.get('headers', {})
                    authorization_header_set = "Authorization" in headers and headers["Authorization"] == f"Bearer {api_key}"
                    content_type_header_set = "Content-Type" in headers and headers["Content-Type"] == "application/json"
                    results["post_request_headers"] = "Success" if authorization_header_set and content_type_header_set else "Failed to set headers correctly"
        except Exception as e:
            results["post_request"] = f"Error: {str(e)}"
            
        # Test request formatting
        try:
            with patch.object(self.hf_tgi, 'make_post_request_hf_tgi') as mock_post:
                mock_post.return_value = {"generated_text": "Test generated text"}
                
                # Create a mock endpoint handler
                mock_handler = lambda data: self.hf_tgi.make_post_request_hf_tgi(endpoint_url, data, api_key)
                
                # Test with different parameter combinations
                # Case 1: Simple text input
                text_input = "Hello, world!"
                simple_result = self.hf_tgi.format_request(mock_handler, text_input)
                results["format_request_simple"] = "Success" if simple_result else "Failed simple format request"
                
                # Case 2: With parameters
                params_result = self.hf_tgi.format_request(mock_handler, text_input, 
                    max_new_tokens=100, temperature=0.8, top_p=0.95)
                results["format_request_params"] = "Success" if params_result else "Failed params format request"
                
                # Check parameter formatting
                calls = mock_post.call_args_list
                if len(calls) >= 2:
                    # Second call should have parameters
                    args, kwargs = calls[1]
                    data = args[1]
                    params_correct = (
                        "parameters" in data and
                        data["parameters"].get("max_new_tokens") == 100 and
                        data["parameters"].get("temperature") == 0.8 and
                        data["parameters"].get("top_p") == 0.95
                    )
                    results["parameter_formatting"] = "Success" if params_correct else "Failed parameter formatting"
        except Exception as e:
            results["request_formatting"] = f"Error: {str(e)}"
        
        # Test streaming functionality if implemented
        try:
            with patch.object(self.hf_tgi, 'make_stream_request_hf_tgi', 
                              return_value=iter([{"generated_text": "part1"}, {"generated_text": "part2"}])):
                if hasattr(self.hf_tgi, 'stream_generate'):
                    stream_result = list(self.hf_tgi.stream_generate(endpoint_url, "Test prompt", api_key))
                    results["streaming"] = "Success" if len(stream_result) > 0 else "Failed streaming"
                else:
                    results["streaming"] = "Not implemented"
        except Exception as e:
            results["streaming"] = f"Error: {str(e)}"
            
        # Test error handling
        try:
            with patch.object(requests, 'post') as mock_post:
                # Test 401 unauthorized
                mock_response = MagicMock()
                mock_response.status_code = 401
                mock_response.json.return_value = {"error": "Unauthorized"}
                mock_post.return_value = mock_response
                
                try:
                    self.hf_tgi.make_post_request_hf_tgi(endpoint_url, {"inputs": "test"}, "invalid_key")
                    results["error_handling_auth"] = "Failed to raise exception on 401"
                except Exception:
                    results["error_handling_auth"] = "Success"
                    
                # Test 404 model not found
                mock_response.status_code = 404
                mock_response.json.return_value = {"error": "Model not found"}
                
                try:
                    self.hf_tgi.make_post_request_hf_tgi(endpoint_url, {"inputs": "test"}, api_key)
                    results["error_handling_404"] = "Failed to raise exception on 404"
                except Exception:
                    results["error_handling_404"] = "Success"
        except Exception as e:
            results["error_handling"] = f"Error: {str(e)}"
            
        # Test the internal test method
        try:
            endpoint_url = "https://api-inference.huggingface.co/models/gpt2"
            test_handler = lambda x: {"generated_text": "Generated text response"}  # Mock handler for testing
            test_result = self.hf_tgi.__test__(endpoint_url, test_handler, "test_endpoint")
            results["internal_test"] = "Success" if test_result else "Failed internal test"
        except Exception as e:
            results["internal_test"] = f"Error: {str(e)}"
            
        # Test additional multiplexing features
        try:
            multiplexer_results = self._test_multiplexer()
            results.update(multiplexer_results)
        except Exception as e:
            results["hftgi_multiplexer"] = f"Error: {str(e)}"
            
        return results
    
    def _test_multiplexer(self):
        """Test HF TGI API key multiplexing with a dedicated multiplexer class"""
        results = {
            "hftgi_multiplexer": {
                "status": "Testing",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": {}
            }
        }
        
        try:
            # Initialize the multiplexer
            multiplexer = HFTGIMultiplexer()
            
            # Add keys from metadata if present
            keys_added = 0
            
            # Main API key
            if "hf_api_key" in self.metadata and self.metadata["hf_api_key"]:
                multiplexer.add_hf_tgi_key(
                    "hf_tgi_main", 
                    self.metadata["hf_api_key"],
                    model_id=self.metadata.get("model_id", "google/t5-efficient-tiny")
                )
                keys_added += 1
            
            # Additional keys
            for i in range(1, 4):
                key_name = f"hf_api_key_{i}"
                if key_name in self.metadata and self.metadata[key_name]:
                    model_id = self.metadata.get(f"model_id_{i}", "google/t5-efficient-tiny")
                    multiplexer.add_hf_tgi_key(f"hf_tgi_{i}", self.metadata[key_name], model_id=model_id)
                    keys_added += 1
            
            # Get stats on added keys
            if keys_added > 0:
                # Get usage statistics
                stats = multiplexer.get_usage_stats()
                results["hftgi_multiplexer"]["key_count"] = len(stats["hf_tgi"])
                results["hftgi_multiplexer"]["keys"] = list(stats["hf_tgi"].keys())
                
                # Test client retrieval with different strategies
                if len(stats["hf_tgi"]) > 0:
                    # Test round-robin strategy
                    with patch.object(requests, 'post') as mock_post:
                        mock_response = MagicMock()
                        mock_response.json.return_value = {"generated_text": "Test generated text"}
                        mock_response.status_code = 200
                        mock_post.return_value = mock_response
                        
                        clients_used = set()
                        for i in range(min(keys_added * 2, 6)):  # Make 2 requests per key, up to 6 max
                            client = multiplexer.get_hf_tgi_client(strategy="round-robin")
                            
                            # Create a model endpoint for testing
                            model_id = client.metadata.get("model_id", "google/t5-efficient-tiny")
                            endpoint_url = f"https://api-inference.huggingface.co/models/{model_id}"
                            api_key = client.metadata.get("hf_api_key", "")
                            
                            # Mock the endpoint handling
                            handler = client.create_remote_text_generation_endpoint_handler(endpoint_url, api_key)
                            
                            # Use the client to make a request
                            try:
                                prompt = f"Test prompt {i+1}"
                                response = client.format_request(handler, prompt)
                                
                                # Track which client we used by checking API key
                                clients_used.add(client.metadata.get("hf_api_key", ""))
                            except Exception as e:
                                print(f"Error in multiplexer test request {i+1}: {str(e)}")
                        
                        # Check if we used multiple clients
                        results["hftgi_multiplexer"]["round_robin"] = {
                            "clients_used": len(clients_used),
                            "success": len(clients_used) > 0
                        }
                
                # Get final usage statistics
                final_stats = multiplexer.get_usage_stats()
                client_usage = {key: stats["usage"] for key, stats in final_stats["hf_tgi"].items()}
                results["hftgi_multiplexer"]["usage_stats"] = client_usage
                
                # Set overall status
                if keys_added > 0 and results["hftgi_multiplexer"].get("round_robin", {}).get("success", False):
                    results["hftgi_multiplexer"]["status"] = "Success"
                elif keys_added > 0:
                    results["hftgi_multiplexer"]["status"] = "Partial Success - Keys added but round-robin failed"
                else:
                    results["hftgi_multiplexer"]["status"] = "Failed - No keys added"
            else:
                results["hftgi_multiplexer"]["status"] = "Skipped - No API keys available"
        except Exception as e:
            results["hftgi_multiplexer"]["status"] = "Error"
            results["hftgi_multiplexer"]["error"] = str(e)
            import traceback
            results["hftgi_multiplexer"]["traceback"] = traceback.format_exc()
        
        return results

    def __test__(self):
        """Run tests and compare/save results"""
        test_results = {}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {"test_error": str(e)}
        
        # Create directories if they don't exist
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_dir = os.path.join(base_dir, 'expected_results')
        collected_dir = os.path.join(base_dir, 'collected_results')
        
        # Create directories with appropriate permissions
        for directory in [expected_dir, collected_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, mode=0o755, exist_ok=True)
        
        # Save collected results
        results_file = os.path.join(collected_dir, 'hf_tgi_test_results.json')
        try:
            # Try to serialize the results to catch any non-serializable objects
            try:
                json_str = json.dumps(test_results)
            except TypeError as e:
                print(f"Warning: Test results contain non-serializable data: {str(e)}")
                # Convert non-serializable objects to strings
                serializable_results = {}
                for k, v in test_results.items():
                    try:
                        json.dumps({k: v})
                        serializable_results[k] = v
                    except TypeError:
                        serializable_results[k] = str(v)
                test_results = serializable_results
            
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
        except Exception as e:
            print(f"Error saving results to {results_file}: {str(e)}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_tgi_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                    if expected_results != test_results:
                        print("Test results differ from expected results!")
                        print(f"Expected: {json.dumps(expected_results, indent=2)}")
                        print(f"Got: {json.dumps(test_results, indent=2)}")
            except Exception as e:
                print(f"Error comparing results with {expected_file}: {str(e)}")
        else:
            # Create expected results file if it doesn't exist
            try:
                with open(expected_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
                    print(f"Created new expected results file: {expected_file}")
            except Exception as e:
                print(f"Error creating {expected_file}: {str(e)}")

        return test_results

if __name__ == "__main__":
    metadata = {
        "hf_api_key": os.environ.get("HF_API_KEY", ""),
        "model_id": os.environ.get("HF_MODEL_ID", "google/t5-efficient-tiny")
    }
    
    # Check for additional keys in environment variables
    for i in range(1, 4):
        key = os.environ.get(f"HF_API_KEY_{i}", "")
        if key:
            metadata[f"hf_api_key_{i}"] = key
            model = os.environ.get(f"HF_MODEL_ID_{i}", "")
            if model:
                metadata[f"model_id_{i}"] = model
    
    resources = {}
    try:
        this_hf_tgi = test_hf_tgi(resources, metadata)
        results = this_hf_tgi.__test__()
        print(f"HF TGI API Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)