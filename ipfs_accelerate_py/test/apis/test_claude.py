import os
import io
import sys
import json
from unittest.mock import MagicMock, patch
import requests

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from api_backends import apis, claude

class test_claude:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {
            "claude_api_key": os.environ.get("ANTHROPIC_API_KEY", "")
        }
        self.claude = claude(resources=self.resources, metadata=self.metadata)
        return None
    
    def test(self):
        """Run all tests for the Claude (Anthropic) API backend"""
        results = {}
        
        # Test endpoint handler creation
        try:
            endpoint_handler = self.claude.create_claude_endpoint_handler()
            results["endpoint_handler"] = "Success" if callable(endpoint_handler) else "Failed to create endpoint handler"
        except Exception as e:
            results["endpoint_handler"] = f"Error: {str(e)}"
            
        # Test endpoint testing function
        try:
            with patch.object(self.claude, 'make_post_request_claude') as mock_post:
                mock_post.return_value = {
                    "id": "test-id",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "This is a test response"}],
                    "model": "claude-3-opus-20240229",
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20
                    }
                }
                
                test_result = self.claude.test_claude_endpoint()
                results["test_endpoint"] = "Success" if test_result else "Failed endpoint test"
                
                # Verify correct parameters were used
                args, kwargs = mock_post.call_args
                results["test_endpoint_params"] = "Success" if "messages" in args[1] else "Failed to pass correct parameters"
        except Exception as e:
            results["test_endpoint"] = f"Error: {str(e)}"
            
        # Test post request function
        try:
            with patch.object(requests, 'post') as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "id": "test-id",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "This is a test response"}],
                    "model": "claude-3-opus-20240229",
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20
                    }
                }
                mock_response.status_code = 200
                mock_post.return_value = mock_response
                
                data = {
                    "model": "claude-3-opus-20240229",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
                
                post_result = self.claude.make_post_request_claude(data)
                results["post_request"] = "Success" if "content" in post_result else "Failed post request"
                
                # Verify headers were set correctly
                args, kwargs = mock_post.call_args
                headers = kwargs.get('headers', {})
                auth_header_set = headers.get("x-api-key") == self.metadata.get("claude_api_key")
                anthropic_header_set = "anthropic-version" in headers
                content_type_set = headers.get("Content-Type") == "application/json"
                results["post_request_headers"] = "Success" if auth_header_set and anthropic_header_set and content_type_set else "Failed to set headers correctly"
        except Exception as e:
            results["post_request"] = f"Error: {str(e)}"
            
        # Test chat method
        try:
            with patch.object(self.claude, 'make_post_request_claude') as mock_post:
                mock_post.return_value = {
                    "id": "test-id",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "This is a test chat response"}],
                    "model": "claude-3-opus-20240229",
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20
                    }
                }
                
                messages = [{"role": "user", "content": "Hello"}]
                chat_result = self.claude.chat(messages)
                results["chat_method"] = "Success" if chat_result and isinstance(chat_result, dict) else "Failed chat method"
        except Exception as e:
            results["chat_method"] = f"Error: {str(e)}"
            
        # Test streaming functionality if implemented
        try:
            with patch.object(self.claude, 'make_stream_request_claude') as mock_stream:
                mock_stream.return_value = iter([
                    {"type": "message_start", "message": {"id": "test-id", "role": "assistant", "content": []}},
                    {"type": "content_block_start", "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "delta": {"type": "text", "text": "This "}},
                    {"type": "content_block_delta", "delta": {"type": "text", "text": "is "}},
                    {"type": "content_block_delta", "delta": {"type": "text", "text": "a "}},
                    {"type": "content_block_delta", "delta": {"type": "text", "text": "test"}},
                    {"type": "content_block_stop"},
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}},
                    {"type": "message_stop"}
                ])
                
                if hasattr(self.claude, 'stream_chat'):
                    stream_result = list(self.claude.stream_chat([{"role": "user", "content": "Hello"}]))
                    results["streaming"] = "Success" if len(stream_result) > 0 else "Failed streaming"
                else:
                    results["streaming"] = "Not implemented"
        except Exception as e:
            results["streaming"] = f"Error: {str(e)}"
            
        # Test error handling
        try:
            with patch.object(requests, 'post') as mock_post:
                # Test invalid API key
                mock_response = MagicMock()
                mock_response.status_code = 401
                mock_response.json.return_value = {"error": {"type": "authentication_error", "message": "Invalid API key"}}
                mock_post.return_value = mock_response
                
                try:
                    self.claude.make_post_request_claude({"messages": [{"role": "user", "content": "test"}]})
                    results["error_handling_auth"] = "Failed to catch authentication error"
                except Exception:
                    results["error_handling_auth"] = "Success"
                    
                # Test rate limit error
                mock_response.status_code = 429
                mock_response.json.return_value = {"error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}}
                
                try:
                    self.claude.make_post_request_claude({"messages": [{"role": "user", "content": "test"}]})
                    results["error_handling_rate_limit"] = "Failed to catch rate limit error"
                except Exception:
                    results["error_handling_rate_limit"] = "Success"
                    
                # Test invalid request
                mock_response.status_code = 400
                mock_response.json.return_value = {"error": {"type": "invalid_request_error", "message": "Invalid request"}}
                
                try:
                    self.claude.make_post_request_claude({"invalid": "data"})
                    results["error_handling_400"] = "Failed to catch invalid request error"
                except Exception:
                    results["error_handling_400"] = "Success"
        except Exception as e:
            results["error_handling"] = f"Error: {str(e)}"
            
        # Test model compatibility if implemented
        try:
            if hasattr(self.claude, 'is_compatible_model'):
                compatible = self.claude.is_compatible_model("claude-3-opus-20240229")
                incompatible = self.claude.is_compatible_model("nonexistent-model")
                results["model_compatibility"] = "Success" if compatible and not incompatible else "Failed model compatibility check"
            else:
                results["model_compatibility"] = "Not implemented"
        except Exception as e:
            results["model_compatibility"] = f"Error: {str(e)}"
        
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
        with open(os.path.join(collected_dir, 'claude_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'claude_test_results.json')
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
        "claude_api_key": os.environ.get("ANTHROPIC_API_KEY", "")
    }
    resources = {}
    try:
        this_claude = test_claude(resources, metadata)
        results = this_claude.__test__()
        print(f"Claude API Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)