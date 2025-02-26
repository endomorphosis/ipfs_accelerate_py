import os
import io
import sys
import json
from unittest.mock import MagicMock, patch
import requests

from ipfs_accelerate_py.api_backends import apis, hf_tgi

class test_hf_tgi:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {
            "hf_api_key": os.environ.get("HF_API_KEY", "")
        }
        self.hf_tgi = hf_tgi(resources=self.resources, metadata=self.metadata)
        return None
    
    def test(self):
        """Run all tests for the HuggingFace Text Generation Inference API backend"""
        results = {}
        
        # Test endpoint handler creation
        try:
            endpoint_url = "https://api-inference.huggingface.co/models/gpt2"
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
                    model_name="gpt2"
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
                
                post_result = self.hf_tgi.make_post_request_hf_tgi(endpoint_url, data, api_key)
                results["post_request"] = "Success" if post_result == mock_response.json.return_value else "Failed post request"
                
                # Verify headers were set correctly
                args, kwargs = mock_post.call_args
                headers = kwargs.get('headers', {})
                results["post_request_headers"] = "Success" if "Authorization" in headers else "Failed to set authorization header"
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
        with open(os.path.join(collected_dir, 'hf_tgi_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_tgi_test_results.json')
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
        "hf_api_key": os.environ.get("HF_API_KEY", "")
    }
    resources = {}
    try:
        this_hf_tgi = test_hf_tgi(resources, metadata)
        results = this_hf_tgi.__test__()
        print(f"HF TGI API Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)