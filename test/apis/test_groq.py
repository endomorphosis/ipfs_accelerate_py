import os
import io
import sys
import json
from unittest.mock import MagicMock, patch
import requests

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ipfs_accelerate_py'))
from ipfs_accelerate_py.api_backends import apis, groq

class test_groq:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}}}}}}}}}}}}}}}}
        self.metadata = metadata if metadata else {}}}}}}}}}}}}}}}:
            "groq_api_key": os.environ.get("GROQ_API_KEY", "")
            }
            self.groq = groq(resources=self.resources, metadata=self.metadata)
        return None
    
    def test(self):
        """Run all tests for the Groq API backend"""
        results = {}}}}}}}}}}}}}}}}
        
        # Test endpoint handler creation
        try:
            endpoint_handler = self.groq.create_groq_endpoint_handler()
            results[]]]]],,,,,"endpoint_handler"] = "Success" if callable(endpoint_handler) else "Failed to create endpoint handler":,
        except Exception as e:
            results[]]]]],,,,,"endpoint_handler"] = f"Error: {}}}}}}}}}}}}}}}str(e)}"
            ,
        # Test the default endpoint with mocked responses
        try:
            endpoint_url = "https://api.groq.com/openai/v1/chat/completions"
            api_key = self.metadata.get("groq_api_key", "")
            model_name = "llama3-8b-8192"
            
            with patch.object(self.groq, 'make_post_request_groq') as mock_post:
                mock_post.return_value = {}}}}}}}}}}}}}}}
                "id": "test-id",
                "object": "chat.completion",
                "created": 1677825464,
                "choices": []]]]],,,,,
                {}}}}}}}}}}}}}}}
                "index": 0,
                "message": {}}}}}}}}}}}}}}}
                "role": "assistant",
                "content": "This is a test response from the mocked Groq API."
                },
                "finish_reason": "stop"
                }
                ],
                "usage": {}}}}}}}}}}}}}}}
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
                }
                }
                
                test_result = self.groq.test_groq_endpoint(endpoint_url, api_key, model_name)
                results[]]]]],,,,,"test_endpoint"] = "Success" if test_result else "Failed endpoint test"
                
                # Verify the test sent the right parameters
                args, kwargs = mock_post.call_args
                results[]]]]],,,,,"test_endpoint_params"] = "Success" if args[]]]]],,,,,0] == endpoint_url and "model" in args[]]]]],,,,,1] else "Failed to pass correct parameters":
        except Exception as e:
            results[]]]]],,,,,"test_endpoint"] = f"Error: {}}}}}}}}}}}}}}}str(e)}"
            
        # Test post request function
        try:
            with patch.object(requests, 'post') as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {}}}}}}}}}}}}}}}
                "id": "test-id",
                "object": "chat.completion",
                "created": 1677825464,
                "choices": []]]]],,,,,
                {}}}}}}}}}}}}}}}
                "index": 0,
                "message": {}}}}}}}}}}}}}}}
                "role": "assistant",
                "content": "This is a test response from the mocked Groq API."
                },
                "finish_reason": "stop"
                }
                ],
                "usage": {}}}}}}}}}}}}}}}
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
                }
                }
                mock_response.status_code = 200
                mock_post.return_value = mock_response
                
                data = {}}}}}}}}}}}}}}}
                "model": "llama3-8b-8192",
                "messages": []]]]],,,,,
                {}}}}}}}}}}}}}}}"role": "user", "content": "Test message"}
                ]
                }
                
                post_result = self.groq.make_post_request_groq(endpoint_url, data, api_key)
                results[]]]]],,,,,"post_request"] = "Success" if "choices" in post_result and post_result[]]]]],,,,,"choices"][]]]]],,,,,0][]]]]],,,,,"message"][]]]]],,,,,"content"] == "This is a test response from the mocked Groq API." else "Failed post request"
                
                # Verify headers were set correctly
                args, kwargs = mock_post.call_args
                headers = kwargs.get('headers', {}}}}}}}}}}}}}}}})
                authorization_header_set = "Authorization" in headers and headers[]]]]],,,,,"Authorization"] == f"Bearer {}}}}}}}}}}}}}}}api_key}"
                content_type_header_set = "Content-Type" in headers and headers[]]]]],,,,,"Content-Type"] == "application/json"
                results[]]]]],,,,,"post_request_headers"] = "Success" if authorization_header_set and content_type_header_set else "Failed to set headers correctly":
        except Exception as e:
            results[]]]]],,,,,"post_request"] = f"Error: {}}}}}}}}}}}}}}}str(e)}"
            
        # Test chat completion
        try:
            with patch.object(self.groq, 'make_post_request_groq') as mock_post:
                mock_post.return_value = {}}}}}}}}}}}}}}}
                "id": "test-id",
                "object": "chat.completion",
                "created": 1677825464,
                "choices": []]]]],,,,,
                {}}}}}}}}}}}}}}}
                "index": 0,
                "message": {}}}}}}}}}}}}}}}
                "role": "assistant",
                "content": "This is a test response from the mocked Groq API."
                },
                "finish_reason": "stop"
                }
                ],
                "usage": {}}}}}}}}}}}}}}}
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
                }
                }
                
                # Test chat method if available:
                if hasattr(self.groq, 'chat'):
                    messages = []]]]],,,,,{}}}}}}}}}}}}}}}"role": "user", "content": "Hello, Groq!"}]
                    chat_result = self.groq.chat(model_name, messages)
                    results[]]]]],,,,,"chat_method"] = "Success" if chat_result and "content" in chat_result else "Failed chat method":
                else:
                    results[]]]]],,,,,"chat_method"] = "Not implemented"
        except Exception as e:
            results[]]]]],,,,,"chat_method"] = f"Error: {}}}}}}}}}}}}}}}str(e)}"
            
        # Test streaming functionality if implemented::
        try:
            if hasattr(self.groq, 'stream_chat'):
                with patch.object(self.groq, 'make_stream_request_groq') as mock_stream:
                    mock_stream.return_value = iter([]]]]],,,,,
                    {}}}}}}}}}}}}}}}"choices": []]]]],,,,,{}}}}}}}}}}}}}}}"delta": {}}}}}}}}}}}}}}}"content": "This "}}]},
                    {}}}}}}}}}}}}}}}"choices": []]]]],,,,,{}}}}}}}}}}}}}}}"delta": {}}}}}}}}}}}}}}}"content": "is "}}]},
                    {}}}}}}}}}}}}}}}"choices": []]]]],,,,,{}}}}}}}}}}}}}}}"delta": {}}}}}}}}}}}}}}}"content": "a "}}]},
                    {}}}}}}}}}}}}}}}"choices": []]]]],,,,,{}}}}}}}}}}}}}}}"delta": {}}}}}}}}}}}}}}}"content": "test."}}]},
                    {}}}}}}}}}}}}}}}"choices": []]]]],,,,,{}}}}}}}}}}}}}}}"finish_reason": "stop"}]}
                    ])
                    
                    messages = []]]]],,,,,{}}}}}}}}}}}}}}}"role": "user", "content": "Hello, Groq!"}]
                    stream_result = list(self.groq.stream_chat(model_name, messages))
                    results[]]]]],,,,,"streaming"] = "Success" if len(stream_result) > 0 else "Failed streaming":
            else:
                results[]]]]],,,,,"streaming"] = "Not implemented"
        except Exception as e:
            results[]]]]],,,,,"streaming"] = f"Error: {}}}}}}}}}}}}}}}str(e)}"
            
        # Test error handling
        try:
            with patch.object(requests, 'post') as mock_post:
                # Test 401 unauthorized
                mock_response = MagicMock()
                mock_response.status_code = 401
                mock_response.json.return_value = {}}}}}}}}}}}}}}}"error": {}}}}}}}}}}}}}}}"message": "Invalid API key"}}
                mock_post.return_value = mock_response
                
                try:
                    self.groq.make_post_request_groq(endpoint_url, {}}}}}}}}}}}}}}}"model": model_name, "messages": []]]]],,,,,{}}}}}}}}}}}}}}}"role": "user", "content": "test"}]}, "invalid_key")
                    results[]]]]],,,,,"error_handling_auth"] = "Failed to raise exception on 401"
                except Exception:
                    results[]]]]],,,,,"error_handling_auth"] = "Success"
                    
                # Test 404 model not found
                    mock_response.status_code = 404
                    mock_response.json.return_value = {}}}}}}}}}}}}}}}"error": {}}}}}}}}}}}}}}}"message": "Model not found"}}
                
                try:
                    self.groq.make_post_request_groq(endpoint_url, {}}}}}}}}}}}}}}}"model": "nonexistent-model", "messages": []]]]],,,,,{}}}}}}}}}}}}}}}"role": "user", "content": "test"}]}, api_key)
                    results[]]]]],,,,,"error_handling_404"] = "Failed to raise exception on 404"
                except Exception:
                    results[]]]]],,,,,"error_handling_404"] = "Success"
                    
                # Test 429 rate limit exceeded
                    mock_response.status_code = 429
                    mock_response.json.return_value = {}}}}}}}}}}}}}}}"error": {}}}}}}}}}}}}}}}"message": "Rate limit exceeded"}}
                
                try:
                    self.groq.make_post_request_groq(endpoint_url, {}}}}}}}}}}}}}}}"model": model_name, "messages": []]]]],,,,,{}}}}}}}}}}}}}}}"role": "user", "content": "test"}]}, api_key)
                    results[]]]]],,,,,"error_handling_429"] = "Failed to raise exception on 429"
                except Exception:
                    results[]]]]],,,,,"error_handling_429"] = "Success"
        except Exception as e:
            results[]]]]],,,,,"error_handling"] = f"Error: {}}}}}}}}}}}}}}}str(e)}"
            
        # Test model compatibility checker if implemented::
        try:
            if hasattr(self.groq, 'is_compatible_model'):
                compatible = self.groq.is_compatible_model("llama3-8b-8192")
                incompatible = self.groq.is_compatible_model("nonexistent-model")
                results[]]]]],,,,,"model_compatibility"] = "Success" if compatible and not incompatible else "Failed model compatibility check":
            else:
                results[]]]]],,,,,"model_compatibility"] = "Not implemented"
        except Exception as e:
            results[]]]]],,,,,"model_compatibility"] = f"Error: {}}}}}}}}}}}}}}}str(e)}"
            
                return results

    def __test__(self):
        """Run tests and compare/save results"""
        test_results = {}}}}}}}}}}}}}}}}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {}}}}}}}}}}}}}}}"test_error": str(e)}
        
        # Create directories if they don't exist
            expected_dir = os.path.join(os.path.dirname(__file__), 'expected_results')
            collected_dir = os.path.join(os.path.dirname(__file__), 'collected_results')
            os.makedirs(expected_dir, exist_ok=True)
            os.makedirs(collected_dir, exist_ok=True)
        
        # Save collected results:
        with open(os.path.join(collected_dir, 'groq_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'groq_test_results.json'):
        if os.path.exists(expected_file):
            with open(expected_file, 'r') as f:
                expected_results = json.load(f)
                
                # More detailed comparison of results
                all_match = True
                mismatches = []]]]],,,,,]
                
                for key in set(expected_results.keys()) | set(test_results.keys()):
                    if key not in expected_results:
                        mismatches.append(f"Missing expected key: {}}}}}}}}}}}}}}}key}")
                        all_match = False
                    elif key not in test_results:
                        mismatches.append(f"Missing actual key: {}}}}}}}}}}}}}}}key}")
                        all_match = False
                    elif expected_results[]]]]],,,,,key] != test_results[]]]]],,,,,key]:
                        mismatches.append(f"Key '{}}}}}}}}}}}}}}}key}' differs: Expected '{}}}}}}}}}}}}}}}expected_results[]]]]],,,,,key]}', got '{}}}}}}}}}}}}}}}test_results[]]]]],,,,,key]}'")
                        all_match = False
                
                if not all_match:
                    print("Test results differ from expected results!")
                    for mismatch in mismatches:
                        print(f"- {}}}}}}}}}}}}}}}mismatch}")
                        print(f"\nComplete expected results: {}}}}}}}}}}}}}}}json.dumps(expected_results, indent=2)}")
                        print(f"\nComplete actual results: {}}}}}}}}}}}}}}}json.dumps(test_results, indent=2)}")
                else:
                    print("All test results match expected results.")
        else:
            # Create expected results file if it doesn't exist:
            with open(expected_file, 'w') as f:
                json.dump(test_results, f, indent=2)
                print(f"Created new expected results file: {}}}}}}}}}}}}}}}expected_file}")

            return test_results

if __name__ == "__main__":
    metadata = {}}}}}}}}}}}}}}}
    "groq_api_key": os.environ.get("GROQ_API_KEY", "")
    }
    resources = {}}}}}}}}}}}}}}}}
    try:
        test_groq_api = test_groq(resources, metadata)
        results = test_groq_api.__test__()
        print(f"Groq API Test Results: {}}}}}}}}}}}}}}}json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)