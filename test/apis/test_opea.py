import os
import io
import sys
import json
from unittest.mock import MagicMock, patch
import requests

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ipfs_accelerate_py'))
from ipfs_accelerate_py.api_backends import apis, opea

class test_opea:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        self.opea = opea(resources=self.resources, metadata=self.metadata)
        return None
    
    def test(self):
        """Run all tests for the OpenAI Proxy API backend"""
        results = {}
        
        # Test endpoint handler creation
        try:
            endpoint_url = "http://localhost:8000/v1/chat/completions"
            endpoint_handler = self.opea.create_opea_endpoint_handler()
            results["endpoint_handler"] = "Success" if callable(endpoint_handler) else "Failed to create endpoint handler"
        except Exception as e:
            results["endpoint_handler"] = f"Error: {str(e)}"
            
        # Test endpoint testing function
        try:
            with patch.object(self.opea, 'make_post_request_opea') as mock_post:
                mock_post.return_value = {
                    "choices": [{
                        "message": {"content": "Test response", "role": "assistant"},
                        "finish_reason": "stop"
                    }]
                }
                
                test_result = self.opea.test_opea_endpoint(endpoint_url)
                results["test_endpoint"] = "Success" if test_result else "Failed endpoint test"
                
                # Verify correct parameters were used
                args, kwargs = mock_post.call_args
                messages_valid = isinstance(args[1].get("messages", []), list)
                results["test_endpoint_params"] = "Success" if messages_valid else "Failed to pass correct parameters"
        except Exception as e:
            results["test_endpoint"] = f"Error: {str(e)}"
            
        # Test post request function
        try:
            with patch.object(requests, 'post') as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "choices": [{
                        "message": {"content": "Test response", "role": "assistant"},
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30
                    }
                }
                mock_response.status_code = 200
                mock_post.return_value = mock_response
                
                data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello"}]
                }
                
                post_result = self.opea.make_post_request_opea(endpoint_url, data)
                results["post_request"] = "Success" if "choices" in post_result else "Failed post request"
                
                # Verify headers were set correctly
                args, kwargs = mock_post.call_args
                headers = kwargs.get('headers', {})
                content_type_set = headers.get("Content-Type") == "application/json"
                results["post_request_headers"] = "Success" if content_type_set else "Failed to set headers correctly"
        except Exception as e:
            results["post_request"] = f"Error: {str(e)}"
            
        # Test chat completion
        try:
            with patch.object(self.opea, 'make_post_request_opea') as mock_post:
                mock_post.return_value = {
                    "choices": [{
                        "message": {"content": "Test chat response", "role": "assistant"},
                        "finish_reason": "stop"
                    }]
                }
                
                # Test chat method if available
                messages = [{"role": "user", "content": "Hello"}]
                if hasattr(self.opea, 'chat'):
                    chat_result = self.opea.chat(messages)
                    results["chat_method"] = "Success" if chat_result and isinstance(chat_result, dict) else "Failed chat method"
                else:
                    results["chat_method"] = "Not implemented"
        except Exception as e:
            results["chat_method"] = f"Error: {str(e)}"
            
        # Test error handling
        try:
            with patch.object(requests, 'post') as mock_post:
                # Test connection error
                mock_post.side_effect = requests.ConnectionError("Connection refused")
                
                try:
                    self.opea.make_post_request_opea(endpoint_url, {"messages": [{"role": "user", "content": "test"}]})
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
                    self.opea.make_post_request_opea(endpoint_url, {"messages": [{"role": "user", "content": "test"}]})
                    results["error_handling_500"] = "Failed to raise exception on 500"
                except Exception:
                    results["error_handling_500"] = "Success"
        except Exception as e:
            results["error_handling"] = f"Error: {str(e)}"
            
        # Test streaming if implemented
        try:
            with patch.object(self.opea, 'make_stream_request_opea') as mock_stream:
                mock_stream.return_value = iter([
                    {"choices": [{"delta": {"content": "Test "}}]},
                    {"choices": [{"delta": {"content": "streaming "}}]},
                    {"choices": [{"delta": {"content": "response"}}]},
                    {"choices": [{"finish_reason": "stop"}]}
                ])
                
                if hasattr(self.opea, 'stream_chat'):
                    stream_result = list(self.opea.stream_chat([{"role": "user", "content": "Hello"}]))
                    results["streaming"] = "Success" if len(stream_result) > 0 else "Failed streaming"
                else:
                    results["streaming"] = "Not implemented"
        except Exception as e:
            results["streaming"] = f"Error: {str(e)}"
        
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
        with open(os.path.join(collected_dir, 'opea_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'opea_test_results.json')
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
        this_opea = test_opea(resources, metadata)
        results = this_opea.__test__()
        print(f"OPEA API Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)