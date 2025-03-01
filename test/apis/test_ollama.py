import os
import io
import sys
import json
from unittest.mock import MagicMock, patch
import requests

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ipfs_accelerate_py'))
from api_backends import apis, ollama

class test_ollama:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {
            "ollama_api_url": os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api"),
            "ollama_model": os.environ.get("OLLAMA_MODEL", "llama2"),
            "timeout": int(os.environ.get("OLLAMA_TIMEOUT", "30"))
        }
        self.ollama = ollama(resources=self.resources, metadata=self.metadata)
        
        # Standard test prompts for consistency
        self.test_prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "Explain the concept of machine learning in simple terms.",
            "Write a short poem about nature.",
            "Translate 'hello world' to French.",
            "What is the capital of France?"
        ]
        return None
    
    def test(self):
        """Run all tests for the Ollama API backend"""
        results = {}
        
        # Test endpoint handler creation
        try:
            # Use the endpoint URL from metadata if provided, otherwise use default
            endpoint_url = self.metadata.get("ollama_api_url", "http://localhost:11434/api/generate")
            endpoint_handler = self.ollama.create_ollama_endpoint_handler(endpoint_url)
            results["endpoint_handler"] = "Success" if callable(endpoint_handler) else "Failed to create endpoint handler"
        except Exception as e:
            results["endpoint_handler"] = f"Error: {str(e)}"
            
        # Test endpoint testing function
        try:
            with patch.object(self.ollama, 'make_post_request_ollama') as mock_post:
                mock_post.return_value = {
                    "model": "llama2",
                    "response": "This is a test response",
                    "done": True
                }
                
                test_result = self.ollama.test_ollama_endpoint(endpoint_url)
                results["test_endpoint"] = "Success" if test_result else "Failed endpoint test"
                
                # Verify correct parameters were used
                args, kwargs = mock_post.call_args
                results["test_endpoint_params"] = "Success" if "prompt" in args[1] else "Failed to pass correct parameters"
        except Exception as e:
            results["test_endpoint"] = f"Error: {str(e)}"
            
        # Test post request function
        try:
            with patch.object(requests, 'post') as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "model": "llama2",
                    "response": "This is a test response",
                    "done": True,
                    "context": [1, 2, 3],
                    "total_duration": 123456789,
                    "load_duration": 12345678,
                    "prompt_eval_count": 10,
                    "prompt_eval_duration": 1234567,
                    "eval_count": 20,
                    "eval_duration": 2345678
                }
                mock_response.status_code = 200
                mock_post.return_value = mock_response
                
                data = {
                    "model": "llama2",
                    "prompt": "Hello, world!",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                }
                
                post_result = self.ollama.make_post_request_ollama(endpoint_url, data)
                results["post_request"] = "Success" if "response" in post_result else "Failed post request"
                
                # Verify headers were set correctly
                args, kwargs = mock_post.call_args
                headers = kwargs.get('headers', {})
                content_type_set = headers.get("Content-Type") == "application/json"
                results["post_request_headers"] = "Success" if content_type_set else "Failed to set headers correctly"
        except Exception as e:
            results["post_request"] = f"Error: {str(e)}"
            
        # Test chat completion
        try:
            with patch.object(self.ollama, 'make_post_request_ollama') as mock_post:
                mock_post.return_value = {
                    "model": self.metadata.get("ollama_model", "llama2"),
                    "response": "This is a chat response",
                    "done": True
                }
                
                if hasattr(self.ollama, 'chat'):
                    messages = [{"role": "user", "content": "Hello"}]
                    chat_result = self.ollama.chat(
                        self.metadata.get("ollama_model", "llama2"), 
                        messages
                    )
                    results["chat_method"] = "Success" if chat_result and isinstance(chat_result, dict) else "Failed chat method"
                    
                    # Test with system message if supported
                    if hasattr(self.ollama, 'chat_with_system'):
                        system_message = "You are a helpful assistant"
                        chat_system_result = self.ollama.chat_with_system(
                            self.metadata.get("ollama_model", "llama2"),
                            messages,
                            system_message
                        )
                        results["chat_with_system"] = "Success" if chat_system_result and isinstance(chat_system_result, dict) else "Failed chat with system"
                    else:
                        results["chat_with_system"] = "Not implemented"
                else:
                    results["chat_method"] = "Not implemented"
        except Exception as e:
            results["chat_method"] = f"Error: {str(e)}"
            
        # Test streaming functionality
        try:
            with patch.object(self.ollama, 'make_stream_request_ollama') as mock_stream:
                mock_stream.return_value = iter([
                    {"response": "This ", "done": False},
                    {"response": "is ", "done": False},
                    {"response": "a ", "done": False},
                    {"response": "test", "done": True}
                ])
                
                if hasattr(self.ollama, 'stream_chat'):
                    messages = [{"role": "user", "content": "Hello"}]
                    stream_result = list(self.ollama.stream_chat(
                        self.metadata.get("ollama_model", "llama2"),
                        messages
                    ))
                    results["streaming"] = "Success" if len(stream_result) > 0 else "Failed streaming"
                    
                    # Test streaming with options if supported
                    if hasattr(self.ollama, 'stream_chat_with_options'):
                        options = {
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "top_k": 40,
                            "num_predict": 100
                        }
                        stream_options_result = list(self.ollama.stream_chat_with_options(
                            self.metadata.get("ollama_model", "llama2"),
                            messages,
                            options
                        ))
                        results["streaming_with_options"] = "Success" if len(stream_options_result) > 0 else "Failed streaming with options"
                    else:
                        results["streaming_with_options"] = "Not implemented"
                else:
                    results["streaming"] = "Not implemented"
        except Exception as e:
            results["streaming"] = f"Error: {str(e)}"
            
        # Test error handling
        try:
            with patch.object(requests, 'post') as mock_post:
                # Test connection error
                mock_post.side_effect = requests.ConnectionError("Connection refused")
                
                try:
                    self.ollama.make_post_request_ollama(endpoint_url, {"prompt": "test"})
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
                    self.ollama.make_post_request_ollama(endpoint_url, {"prompt": "test"})
                    results["error_handling_500"] = "Failed to raise exception on 500"
                except Exception:
                    results["error_handling_500"] = "Success"
                    
                # Test model not found
                mock_response.status_code = 404
                mock_response.json.return_value = {"error": "Model not found"}
                
                try:
                    self.ollama.make_post_request_ollama(endpoint_url, {"model": "nonexistent", "prompt": "test"})
                    results["error_handling_404"] = "Failed to raise exception on 404"
                except Exception:
                    results["error_handling_404"] = "Success"
        except Exception as e:
            results["error_handling"] = f"Error: {str(e)}"
            
        # Test model list retrieval if implemented
        try:
            with patch.object(requests, 'get') as mock_get:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "models": [
                        {"name": "llama2", "size": 123456789, "modified_at": "2023-11-04T14:54:20Z", "digest": "sha256:1234567890"},
                        {"name": "codellama", "size": 234567890, "modified_at": "2023-11-05T15:54:20Z", "digest": "sha256:0987654321"}
                    ]
                }
                mock_response.status_code = 200
                mock_get.return_value = mock_response
                
                if hasattr(self.ollama, 'list_models'):
                    models = self.ollama.list_models()
                    results["model_list"] = "Success" if isinstance(models, list) and len(models) > 0 else "Failed model list retrieval"
                    
                    # Test model info if implemented
                    if hasattr(self.ollama, 'get_model_info'):
                        model_info = self.ollama.get_model_info(self.metadata.get("ollama_model", "llama2"))
                        results["model_info"] = "Success" if isinstance(model_info, dict) and "name" in model_info else "Failed model info retrieval"
                    else:
                        results["model_info"] = "Not implemented"
                else:
                    results["model_list"] = "Not implemented"
        except Exception as e:
            results["model_list"] = f"Error: {str(e)}"
            
        # Test model pulling if implemented
        try:
            with patch.object(requests, 'post') as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {"status": "success"}
                mock_response.status_code = 200
                mock_post.return_value = mock_response
                
                if hasattr(self.ollama, 'pull_model'):
                    pull_result = self.ollama.pull_model(self.metadata.get("ollama_model", "llama2"))
                    results["model_pull"] = "Success" if pull_result else "Failed model pull"
                else:
                    results["model_pull"] = "Not implemented"
        except Exception as e:
            results["model_pull"] = f"Error: {str(e)}"
        
        # Test embedding generation if implemented
        try:
            with patch.object(requests, 'post') as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dim embedding
                }
                mock_response.status_code = 200
                mock_post.return_value = mock_response
                
                if hasattr(self.ollama, 'generate_embeddings'):
                    embed_result = self.ollama.generate_embeddings(
                        self.metadata.get("ollama_model", "llama2"),
                        "This is a test sentence for embedding."
                    )
                    results["embeddings"] = "Success" if isinstance(embed_result, list) and len(embed_result) > 0 else "Failed embedding generation"
                    
                    # Test batch embedding if implemented
                    if hasattr(self.ollama, 'batch_embeddings'):
                        batch_result = self.ollama.batch_embeddings(
                            self.metadata.get("ollama_model", "llama2"),
                            self.test_prompts[:2]  # Use first two test prompts
                        )
                        results["batch_embeddings"] = "Success" if isinstance(batch_result, list) and len(batch_result) == 2 else "Failed batch embedding"
                    else:
                        results["batch_embeddings"] = "Not implemented"
                else:
                    results["embeddings"] = "Not implemented"
        except Exception as e:
            results["embeddings"] = f"Error: {str(e)}"
            
        # Test token counting if implemented
        try:
            with patch.object(requests, 'post') as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {"tokens": 42}
                mock_response.status_code = 200
                mock_post.return_value = mock_response
                
                if hasattr(self.ollama, 'count_tokens'):
                    token_count = self.ollama.count_tokens(
                        self.metadata.get("ollama_model", "llama2"),
                        "This is a test sentence for counting tokens."
                    )
                    results["token_count"] = "Success" if isinstance(token_count, int) and token_count > 0 else "Failed token counting"
                else:
                    results["token_count"] = "Not implemented"
        except Exception as e:
            results["token_count"] = f"Error: {str(e)}"
        
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
        results_file = os.path.join(collected_dir, 'ollama_test_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
        except Exception as e:
            print(f"Error saving results to {results_file}: {str(e)}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'ollama_test_results.json')
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
    metadata = {}
    resources = {}
    try:
        this_ollama = test_ollama(resources, metadata)
        results = this_ollama.__test__()
        print(f"Ollama API Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)