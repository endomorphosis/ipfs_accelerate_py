import os
import io
import sys
import json
from unittest.mock import MagicMock, patch
import requests

from ipfs_accelerate_py.api_backends import apis, llvm

class test_llvm:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        self.llvm = llvm(resources=self.resources, metadata=self.metadata)
        return None
    
    def test(self):
        """Run all tests for the LLVM API backend"""
        results = {}
        
        # Test endpoint handler creation
        try:
            endpoint_url = "http://localhost:8090"
            endpoint_handler = self.llvm.create_llvm_endpoint_handler(endpoint_url)
            results["endpoint_handler"] = "Success" if callable(endpoint_handler) else "Failed to create endpoint handler"
        except Exception as e:
            results["endpoint_handler"] = f"Error: {str(e)}"
            
        # Test endpoint testing function
        try:
            with patch.object(self.llvm, 'make_post_request_llvm') as mock_post:
                mock_post.return_value = {
                    "result": "Test inference result",
                    "status": "success"
                }
                
                test_result = self.llvm.test_llvm_endpoint(endpoint_url)
                results["test_endpoint"] = "Success" if test_result else "Failed endpoint test"
                
                # Verify correct parameters were used
                args, kwargs = mock_post.call_args
                results["test_endpoint_params"] = "Success" if "input" in args[1] else "Failed to pass correct parameters"
        except Exception as e:
            results["test_endpoint"] = f"Error: {str(e)}"
            
        # Test post request function
        try:
            with patch.object(requests, 'post') as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "result": "Test inference result",
                    "status": "success",
                    "metrics": {
                        "inference_time": 0.123,
                        "throughput": 100.0
                    }
                }
                mock_response.status_code = 200
                mock_post.return_value = mock_response
                
                data = {
                    "input": "Test input data",
                    "parameters": {
                        "batch_size": 1,
                        "precision": "fp32"
                    }
                }
                
                post_result = self.llvm.make_post_request_llvm(endpoint_url, data)
                results["post_request"] = "Success" if "result" in post_result else "Failed post request"
                
                # Verify headers were set correctly
                args, kwargs = mock_post.call_args
                headers = kwargs.get('headers', {})
                content_type_set = headers.get("Content-Type") == "application/json"
                results["post_request_headers"] = "Success" if content_type_set else "Failed to set headers correctly"
        except Exception as e:
            results["post_request"] = f"Error: {str(e)}"
            
        # Test inference request formatting
        try:
            with patch.object(self.llvm, 'make_post_request_llvm') as mock_post:
                mock_post.return_value = {
                    "result": "Test inference result",
                    "status": "success"
                }
                
                # Create a mock endpoint handler
                mock_handler = lambda data: self.llvm.make_post_request_llvm(endpoint_url, data)
                
                # Test with different input types
                # Case 1: Simple input
                if hasattr(self.llvm, 'format_request'):
                    input_data = "Test input"
                    simple_result = self.llvm.format_request(mock_handler, input_data)
                    results["format_request_simple"] = "Success" if simple_result else "Failed simple format request"
                    
                    # Case 2: Batch input
                    batch_input = ["Input 1", "Input 2"]
                    batch_result = self.llvm.format_request(mock_handler, batch_input)
                    results["format_request_batch"] = "Success" if batch_result else "Failed batch format request"
                    
                    # Case 3: Input with parameters
                    params_input = {
                        "data": "Test input",
                        "parameters": {
                            "precision": "fp16",
                            "batch_size": 4
                        }
                    }
                    params_result = self.llvm.format_request(mock_handler, params_input)
                    results["format_request_params"] = "Success" if params_result else "Failed params format request"
                else:
                    results["format_request"] = "Method not implemented"
        except Exception as e:
            results["request_formatting"] = f"Error: {str(e)}"
            
        # Test error handling
        try:
            with patch.object(requests, 'post') as mock_post:
                # Test connection error
                mock_post.side_effect = requests.ConnectionError("Connection refused")
                
                try:
                    self.llvm.make_post_request_llvm(endpoint_url, {"input": "test"})
                    results["error_handling_connection"] = "Failed to catch connection error"
                except Exception:
                    results["error_handling_connection"] = "Success"
                    
                # Test invalid response
                mock_post.side_effect = None
                mock_response = MagicMock()
                mock_response.status_code = 500
                mock_response.json.return_value = {"error": "Internal server error"}
                mock_post.return_value = mock_response
                
                try:
                    self.llvm.make_post_request_llvm(endpoint_url, {"input": "test"})
                    results["error_handling_500"] = "Failed to raise exception on 500"
                except Exception:
                    results["error_handling_500"] = "Success"
                    
                # Test malformed response
                mock_response.status_code = 200
                mock_response.json.side_effect = ValueError("Invalid JSON")
                
                try:
                    self.llvm.make_post_request_llvm(endpoint_url, {"input": "test"})
                    results["error_handling_invalid_json"] = "Failed to handle invalid JSON"
                except Exception:
                    results["error_handling_invalid_json"] = "Success"
        except Exception as e:
            results["error_handling"] = f"Error: {str(e)}"
            
        # Test batch processing if implemented
        try:
            with patch.object(self.llvm, 'make_post_request_llvm') as mock_post:
                mock_post.return_value = {
                    "results": ["Result 1", "Result 2"],
                    "status": "success"
                }
                
                if hasattr(self.llvm, 'process_batch'):
                    batch_data = ["Input 1", "Input 2"]
                    batch_result = self.llvm.process_batch(batch_data)
                    results["batch_processing"] = "Success" if isinstance(batch_result, list) and len(batch_result) == 2 else "Failed batch processing"
                else:
                    results["batch_processing"] = "Not implemented"
        except Exception as e:
            results["batch_processing"] = f"Error: {str(e)}"
            
        # Test model information retrieval if implemented
        try:
            if hasattr(self.llvm, 'get_model_info'):
                with patch.object(requests, 'get') as mock_get:
                    mock_response = MagicMock()
                    mock_response.json.return_value = {
                        "name": "test_model",
                        "version": "1.0",
                        "precision": "fp32",
                        "backend": "llvm"
                    }
                    mock_response.status_code = 200
                    mock_get.return_value = mock_response
                    
                    model_info = self.llvm.get_model_info(endpoint_url)
                    results["model_info"] = "Success" if isinstance(model_info, dict) and "name" in model_info else "Failed model info retrieval"
            else:
                results["model_info"] = "Not implemented"
        except Exception as e:
            results["model_info"] = f"Error: {str(e)}"
        
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
        with open(os.path.join(collected_dir, 'llvm_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'llvm_test_results.json')
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
    metadata = {}
    resources = {}
    try:
        this_llvm = test_llvm(resources, metadata)
        results = this_llvm.__test__()
        print(f"LLVM API Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)