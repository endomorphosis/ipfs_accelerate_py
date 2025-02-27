import os
import io
import sys
import json
from unittest.mock import MagicMock, patch
import requests

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ipfs_accelerate_py'))
from api_backends import apis, gemini

class test_gemini:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {
            "gemini_api_key": os.environ.get("GEMINI_API_KEY", "")
        }
        self.gemini = gemini(resources=self.resources, metadata=self.metadata)
        return None
    
    def test(self):
        """Run all tests for the Google Gemini API backend"""
        results = {}
        
        # Test endpoint handler creation
        try:
            endpoint_handler = self.gemini.create_gemini_endpoint_handler()
            results["endpoint_handler"] = "Success" if callable(endpoint_handler) else "Failed to create endpoint handler"
        except Exception as e:
            results["endpoint_handler"] = f"Error: {str(e)}"
            
        # Test endpoint testing function
        try:
            with patch.object(self.gemini, 'make_post_request_gemini') as mock_post:
                mock_post.return_value = {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "This is a test response"}],
                                "role": "model"
                            },
                            "finishReason": "STOP",
                            "tokenCount": {
                                "totalTokens": 30,
                                "inputTokens": 10,
                                "outputTokens": 20
                            }
                        }
                    ]
                }
                
                test_result = self.gemini.test_gemini_endpoint()
                results["test_endpoint"] = "Success" if test_result else "Failed endpoint test"
                
                # Verify correct parameters were used
                args, kwargs = mock_post.call_args
                results["test_endpoint_params"] = "Success" if "contents" in args[1] else "Failed to pass correct parameters"
        except Exception as e:
            results["test_endpoint"] = f"Error: {str(e)}"
            
        # Test post request function
        try:
            with patch.object(requests, 'post') as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "This is a test response"}],
                                "role": "model"
                            },
                            "finishReason": "STOP",
                            "tokenCount": {
                                "totalTokens": 30,
                                "inputTokens": 10,
                                "outputTokens": 20
                            }
                        }
                    ]
                }
                mock_response.status_code = 200
                mock_post.return_value = mock_response
                
                data = {
                    "contents": [
                        {
                            "parts": [{"text": "Hello"}],
                            "role": "user"
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topP": 0.9,
                        "maxOutputTokens": 1000
                    }
                }
                
                post_result = self.gemini.make_post_request_gemini(data)
                results["post_request"] = "Success" if "candidates" in post_result else "Failed post request"
                
                # Verify headers were set correctly
                args, kwargs = mock_post.call_args
                headers = kwargs.get('headers', {})
                content_type_set = headers.get("Content-Type") == "application/json"
                results["post_request_headers"] = "Success" if content_type_set else "Failed to set headers correctly"
        except Exception as e:
            results["post_request"] = f"Error: {str(e)}"
            
        # Test chat method
        try:
            with patch.object(self.gemini, 'make_post_request_gemini') as mock_post:
                mock_post.return_value = {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "This is a test chat response"}],
                                "role": "model"
                            },
                            "finishReason": "STOP",
                            "tokenCount": {
                                "totalTokens": 30,
                                "inputTokens": 10,
                                "outputTokens": 20
                            }
                        }
                    ]
                }
                
                messages = [{"role": "user", "content": "Hello"}]
                chat_result = self.gemini.chat(messages)
                results["chat_method"] = "Success" if chat_result and isinstance(chat_result, dict) else "Failed chat method"
        except Exception as e:
            results["chat_method"] = f"Error: {str(e)}"
            
        # Test streaming functionality if implemented
        try:
            with patch.object(self.gemini, 'make_stream_request_gemini') as mock_stream:
                mock_stream.return_value = iter([
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [{"text": "This "}],
                                    "role": "model"
                                },
                                "finishReason": None
                            }
                        ]
                    },
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [{"text": "is "}],
                                    "role": "model"
                                },
                                "finishReason": None
                            }
                        ]
                    },
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [{"text": "a test"}],
                                    "role": "model"
                                },
                                "finishReason": "STOP"
                            }
                        ]
                    }
                ])
                
                if hasattr(self.gemini, 'stream_chat'):
                    stream_result = list(self.gemini.stream_chat([{"role": "user", "content": "Hello"}]))
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
                mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
                mock_post.return_value = mock_response
                
                try:
                    self.gemini.make_post_request_gemini({"contents": [{"parts": [{"text": "test"}], "role": "user"}]})
                    results["error_handling_auth"] = "Failed to catch authentication error"
                except Exception:
                    results["error_handling_auth"] = "Success"
                    
                # Test rate limit error
                mock_response.status_code = 429
                mock_response.json.return_value = {"error": {"message": "Resource exhausted"}}
                
                try:
                    self.gemini.make_post_request_gemini({"contents": [{"parts": [{"text": "test"}], "role": "user"}]})
                    results["error_handling_rate_limit"] = "Failed to catch rate limit error"
                except Exception:
                    results["error_handling_rate_limit"] = "Success"
                    
                # Test invalid request
                mock_response.status_code = 400
                mock_response.json.return_value = {"error": {"message": "Invalid request"}}
                
                try:
                    self.gemini.make_post_request_gemini({"invalid": "data"})
                    results["error_handling_400"] = "Failed to catch invalid request error"
                except Exception:
                    results["error_handling_400"] = "Success"
        except Exception as e:
            results["error_handling"] = f"Error: {str(e)}"
            
        # Test multi-modal handling if implemented
        try:
            with patch.object(self.gemini, 'make_post_request_gemini') as mock_post:
                mock_post.return_value = {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "This is a description of the image"}],
                                "role": "model"
                            },
                            "finishReason": "STOP"
                        }
                    ]
                }
                
                if hasattr(self.gemini, 'process_image'):
                    # Create a mock image
                    image_data = b"fake image data"
                    image_result = self.gemini.process_image(image_data, "What's in this image?")
                    results["image_processing"] = "Success" if image_result and isinstance(image_result, dict) else "Failed image processing"
                else:
                    results["image_processing"] = "Not implemented"
        except Exception as e:
            results["image_processing"] = f"Error: {str(e)}"
        
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
        with open(os.path.join(collected_dir, 'gemini_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'gemini_test_results.json')
        if os.path.exists(expected_file):
            with open(expected_file, 'r') as f:
                expected_results = json.load(f)
                
                # More detailed comparison of results
                all_match = True
                mismatches = []
                
                for key in set(expected_results.keys()) | set(test_results.keys()):
                    if key not in expected_results:
                        mismatches.append(f"Missing expected key: {key}")
                        all_match = False
                    elif key not in test_results:
                        mismatches.append(f"Missing actual key: {key}")
                        all_match = False
                    elif expected_results[key] != test_results[key]:
                        mismatches.append(f"Key '{key}' differs: Expected '{expected_results[key]}', got '{test_results[key]}'")
                        all_match = False
                
                if not all_match:
                    print("Test results differ from expected results!")
                    for mismatch in mismatches:
                        print(f"- {mismatch}")
                    print(f"\nComplete expected results: {json.dumps(expected_results, indent=2)}")
                    print(f"\nComplete actual results: {json.dumps(test_results, indent=2)}")
                else:
                    print("All test results match expected results.")
        else:
            # Create expected results file if it doesn't exist
            with open(expected_file, 'w') as f:
                json.dump(test_results, f, indent=2)
                print(f"Created new expected results file: {expected_file}")

        return test_results

if __name__ == "__main__":
    metadata = {
        "gemini_api_key": os.environ.get("GOOGLE_API_KEY", "")
    }
    resources = {}
    try:
        this_gemini = test_gemini(resources, metadata)
        results = this_gemini.__test__()
        print(f"Gemini API Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)