import os
import io
import sys
import json
from unittest.mock import MagicMock, patch
import requests

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from api_backends import apis, hf_tei

class test_hf_tei:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {
            "hf_api_key": os.environ.get("HF_API_KEY", "")
        }
        self.hf_tei = hf_tei(resources=self.resources, metadata=self.metadata)
        return None
    
    def test(self):
        """Run all tests for the HuggingFace Text Embedding Inference API backend"""
        results = {}
        
        # Test basic endpoint functionality
        try:
            # Test the endpoint creation function
            endpoint_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
            api_key = self.metadata.get("hf_api_key", "")
            
            endpoint_handler = self.hf_tei.create_remote_text_embedding_endpoint_handler(
                endpoint_url, api_key
            )
            results["endpoint_handler"] = "Success" if callable(endpoint_handler) else "Failed to create endpoint handler"
        except Exception as e:
            results["endpoint_handler"] = str(e)
        
        # Test the endpoint testing function
        try:
            with patch.object(self.hf_tei, 'make_post_request_hf_tei') as mock_post:
                mock_post.return_value = [0.1, 0.2, 0.3] * 100  # Create a mock embedding vector
                
                test_result = self.hf_tei.test_hf_tei_endpoint(
                    endpoint_url=endpoint_url,
                    api_key=api_key,
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                results["test_endpoint"] = "Success" if test_result else "Failed endpoint test"
                
                # Verify correct parameters were used in request
                args, kwargs = mock_post.call_args
                results["test_endpoint_params"] = "Success" if "inputs" in args[1] else "Failed to pass correct parameters"
        except Exception as e:
            results["test_endpoint"] = str(e)
            
        # Test post request function
        try:
            with patch.object(requests, 'post') as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = [0.1, 0.2, 0.3] * 100  # Create a mock embedding vector
                mock_response.status_code = 200
                mock_post.return_value = mock_response
                
                data = {
                    "inputs": "Test input text for embedding"
                }
                
                post_result = self.hf_tei.make_post_request_hf_tei(endpoint_url, data, api_key)
                results["post_request"] = "Success" if isinstance(post_result, list) and len(post_result) > 0 else "Failed post request"
                
                # Verify headers were set correctly
                args, kwargs = mock_post.call_args
                headers = kwargs.get('headers', {})
                authorization_header_set = "Authorization" in headers and headers["Authorization"] == f"Bearer {api_key}"
                content_type_header_set = "Content-Type" in headers and headers["Content-Type"] == "application/json"
                results["post_request_headers"] = "Success" if authorization_header_set and content_type_header_set else "Failed to set headers correctly"
        except Exception as e:
            results["post_request"] = str(e)
            
        # Test request formatting
        try:
            with patch.object(self.hf_tei, 'make_post_request_hf_tei') as mock_post:
                mock_post.return_value = [0.1, 0.2, 0.3] * 100  # Create a mock embedding vector
                
                # Create a mock endpoint handler
                mock_handler = lambda data: self.hf_tei.make_post_request_hf_tei(endpoint_url, data, api_key)
                
                # Test with different input types
                # Case 1: Simple text input
                text_input = "Hello, world!"
                
                if hasattr(self.hf_tei, 'format_request'):
                    simple_result = self.hf_tei.format_request(mock_handler, text_input)
                    results["format_request_simple"] = "Success" if isinstance(simple_result, list) else "Failed simple format request"
                    
                    # Case 2: List of texts
                    list_input = ["Hello, world!", "Another text"]
                    list_result = self.hf_tei.format_request(mock_handler, list_input)
                    results["format_request_list"] = "Success" if isinstance(list_result, list) else "Failed list format request"
                else:
                    results["format_request"] = "Method not implemented"
        except Exception as e:
            results["request_formatting"] = str(e)
            
        # Test error handling
        try:
            with patch.object(requests, 'post') as mock_post:
                # Test 401 unauthorized
                mock_response = MagicMock()
                mock_response.status_code = 401
                mock_response.json.return_value = {"error": "Unauthorized"}
                mock_post.return_value = mock_response
                
                try:
                    self.hf_tei.make_post_request_hf_tei(endpoint_url, {"inputs": "test"}, "invalid_key")
                    results["error_handling_auth"] = "Failed to raise exception on 401"
                except Exception:
                    results["error_handling_auth"] = "Success"
                    
                # Test 404 model not found
                mock_response.status_code = 404
                mock_response.json.return_value = {"error": "Model not found"}
                
                try:
                    self.hf_tei.make_post_request_hf_tei(endpoint_url, {"inputs": "test"}, api_key)
                    results["error_handling_404"] = "Failed to raise exception on 404"
                except Exception:
                    results["error_handling_404"] = "Success"
        except Exception as e:
            results["error_handling"] = str(e)
            
        # Test embedding normalization if implemented
        try:
            if hasattr(self.hf_tei, 'normalize_embedding'):
                test_vector = [1.0, 2.0, 2.0]  # Vector with magnitude 3
                normalized = self.hf_tei.normalize_embedding(test_vector)
                # Check if the magnitude is close to 1
                magnitude = sum(x*x for x in normalized) ** 0.5
                results["normalization"] = "Success" if abs(magnitude - 1.0) < 0.0001 else f"Failed normalization, magnitude: {magnitude}"
            else:
                results["normalization"] = "Method not implemented"
        except Exception as e:
            results["normalization"] = str(e)
            
        # Test the internal test method
        try:
            test_handler = lambda x: [0.1, 0.2, 0.3]  # Mock handler for testing
            test_result = self.hf_tei.__test__(endpoint_url, test_handler, "test_endpoint")
            results["internal_test"] = "Success" if test_result else "Failed internal test"
        except Exception as e:
            results["internal_test"] = str(e)
            
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
        with open(os.path.join(collected_dir, 'hf_tei_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_tei_test_results.json')
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
        this_hf_tei = test_hf_tei(resources, metadata)
        results = this_hf_tei.__test__()
        print(f"HF TEI API Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)