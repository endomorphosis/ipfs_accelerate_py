import os
import io
import sys
import json
from unittest.mock import MagicMock, patch
import requests

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from api_backends import apis, ovms

class test_ovms:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        self.ovms = ovms(resources=self.resources, metadata=self.metadata)
        return None
    
    def test(self):
        """Run all tests for the OpenVINO Model Server API backend"""
        results = {}
        
        # Test endpoint handler creation
        try:
            endpoint_url = "http://localhost:9000/v1/models/model_name:predict"
            endpoint_handler = self.ovms.create_ovms_endpoint_handler(endpoint_url)
            results["endpoint_handler"] = "Success" if callable(endpoint_handler) else "Failed to create endpoint handler"
        except Exception as e:
            results["endpoint_handler"] = f"Error: {str(e)}"
        
        # Test OVMS endpoint testing function
        try:
            with patch.object(self.ovms, 'make_post_request_ovms', return_value={"predictions": ["test_result"]}):
                test_result = self.ovms.test_ovms_endpoint(
                    endpoint_url=endpoint_url,
                    model_name="intel/test-model"
                )
                results["test_endpoint"] = "Success" if test_result else "Failed endpoint test"
        except Exception as e:
            results["test_endpoint"] = f"Error: {str(e)}"
            
        # Test post request function
        try:
            with patch('requests.post') as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {"predictions": ["test_result"]}
                mock_response.status_code = 200
                mock_post.return_value = mock_response
                
                # Create test data for the request
                data = {"instances": [{"data": [0, 1, 2, 3]}]}
                
                post_result = self.ovms.make_post_request_ovms(endpoint_url, data)
                results["post_request"] = "Success" if "predictions" in post_result else "Failed post request"
                
                # Verify headers were set correctly
                args, kwargs = mock_post.call_args
                headers = kwargs.get('headers', {})
                content_type_header_set = "Content-Type" in headers and headers["Content-Type"] == "application/json"
                results["post_request_headers"] = "Success" if content_type_header_set else "Failed to set headers correctly"
        except Exception as e:
            results["post_request"] = f"Error: {str(e)}"
            
        # Test request formatting
        try:
            with patch.object(self.ovms, 'make_post_request_ovms') as mock_post:
                mock_post.return_value = {"predictions": ["test_result"]}
                
                # Create a mock endpoint handler
                mock_handler = lambda data: self.ovms.make_post_request_ovms(endpoint_url, data)
                
                # Test with different data types
                # Case 1: Simple numeric input
                numeric_input = [1, 2, 3, 4]
                
                if hasattr(self.ovms, 'format_request'):
                    simple_result = self.ovms.format_request(mock_handler, numeric_input)
                    results["format_request_numeric"] = "Success" if simple_result else "Failed numeric format request"
                    
                    # Case 2: More complex input
                    complex_input = {"input": [1, 2, 3, 4], "options": {"precision": "float32"}}
                    complex_result = self.ovms.format_request(mock_handler, complex_input)
                    results["format_request_complex"] = "Success" if complex_result else "Failed complex format request"
                else:
                    results["format_request"] = "Method not implemented"
        except Exception as e:
            results["request_formatting"] = f"Error: {str(e)}"
            
        # Test model compatibility checker
        try:
            # Get API models registry from file path
            model_list_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'ipfs_accelerate_py', 'api_backends', 'model_list', 'ovms.json'
            )
            
            if os.path.exists(model_list_path):
                with open(model_list_path, 'r') as f:
                    model_list = json.load(f)
                    
                # Check if at least one model can be found
                if len(model_list) > 0:
                    results["model_list"] = "Success - Found models"
                    results["model_example"] = model_list[0]  # Example of first model in list
                else:
                    results["model_list"] = "Failed - No models in list"
            else:
                results["model_list"] = f"Failed - Model list file not found at {model_list_path}"
        except Exception as e:
            results["model_list"] = f"Error: {str(e)}"
            
        # Test error handling
        try:
            with patch('requests.post') as mock_post:
                # Test connection error
                mock_post.side_effect = requests.ConnectionError("Connection refused")
                
                try:
                    self.ovms.make_post_request_ovms(endpoint_url, {"instances": [{"data": [0, 1, 2, 3]}]})
                    results["error_handling_connection"] = "Failed to catch connection error"
                except Exception:
                    results["error_handling_connection"] = "Success"
                    
                # Reset mock to test other error types
                mock_post.side_effect = None
                mock_response = MagicMock()
                mock_response.status_code = 404
                mock_response.json.side_effect = ValueError("Invalid JSON")
                mock_post.return_value = mock_response
                
                try:
                    self.ovms.make_post_request_ovms(endpoint_url, {"instances": [{"data": [0, 1, 2, 3]}]})
                    results["error_handling_404"] = "Failed to raise exception on 404"
                except Exception:
                    results["error_handling_404"] = "Success"
        except Exception as e:
            results["error_handling"] = f"Error: {str(e)}"
            
        # Test the internal test method if available
        try:
            if hasattr(self.ovms, '__test__'):
                test_handler = lambda x: {"predictions": ["test_result"]}  # Mock handler for testing
                test_result = self.ovms.__test__(endpoint_url, test_handler, "test_model")
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
        with open(os.path.join(collected_dir, 'ovms_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'ovms_test_results.json')
        if os.path.exists(expected_file):
            with open(expected_file, 'r') as f:
                try:
                    expected_results = json.load(f)
                    if expected_results != test_results:
                        print("Test results differ from expected results!")
                        print(f"Expected: {json.dumps(expected_results, indent=2)}")
                        print(f"Got: {json.dumps(test_results, indent=2)}")
                except json.JSONDecodeError:
                    print(f"Warning: Expected results file {expected_file} contains invalid JSON.")
                    # Create a new expected results file with current results
                    with open(expected_file, 'w') as f_write:
                        json.dump(test_results, f_write, indent=2)
                        print(f"Created new expected results file: {expected_file}")
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
        this_ovms = test_ovms(resources, metadata)
        results = this_ovms.__test__()
        print(f"OVMS API Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        sys.exit(1)