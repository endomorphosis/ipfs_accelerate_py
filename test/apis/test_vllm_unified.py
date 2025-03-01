import os
import sys
import json
import time
import argparse
import datetime
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ipfs_accelerate_py'))

# Import the test class
from apis.test_vllm import test_vllm

# Import for performance testing
import numpy as np
import requests

class TestVLLM(unittest.TestCase):
    """Unified test suite for VLLM API"""
    
    def setUp(self):
        """Set up test environment"""
        self.metadata = {
            "vllm_api_url": os.environ.get("VLLM_API_URL", "http://localhost:8000"),
            "vllm_model": os.environ.get("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
            "timeout": int(os.environ.get("VLLM_TIMEOUT", "30"))
        }
        self.resources = {}
        
        # Standard test inputs for consistency
        self.test_inputs = [
            "This is a simple text input",
            ["Input 1", "Input 2", "Input 3"],  # Batch of text inputs
            {"prompt": "Test input with parameters", "parameters": {"temperature": 0.7}},
            {"input": "Standard format input", "parameters": {"max_tokens": 100}}
        ]
    
    def test_standard_api(self):
        """Test the standard VLLM API functionality"""
        tester = test_vllm(self.resources, self.metadata)
        results = tester.test()
        
        # Verify critical tests passed
        self.assertIn("endpoint_handler", results)
        self.assertIn("test_endpoint", results)
        self.assertIn("post_request", results)
        
        # Save test results
        self._save_test_results(results, "vllm_test_results.json")
    
    def test_performance(self):
        """Test the performance of the VLLM API"""
        # Skip if performance tests disabled
        if os.environ.get("SKIP_PERFORMANCE_TESTS", "").lower() in ("true", "1", "yes"):
            self.skipTest("Performance tests disabled by environment variable")
        
        performance_results = {}
        
        # Create mock data for inference
        with patch.object(requests, 'post') as mock_post:
            # For single inference
            single_mock_response = MagicMock()
            single_mock_response.status_code = 200
            single_mock_response.json.return_value = {
                "text": "This is a test result for performance testing",
                "metadata": {
                    "finish_reason": "length",
                    "model": self.metadata.get("vllm_model"),
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 50,
                        "total_tokens": 60
                    }
                }
            }
            
            # For batch inference
            batch_mock_response = MagicMock()
            batch_mock_response.status_code = 200
            batch_mock_response.json.return_value = {
                "texts": [
                    "This is result 1 for performance testing",
                    "This is result 2 for performance testing",
                    "This is result 3 for performance testing"
                ],
                "metadata": {
                    "finish_reasons": ["length", "length", "length"],
                    "model": self.metadata.get("vllm_model"),
                    "usage": {
                        "prompt_tokens": 30,
                        "completion_tokens": 150,
                        "total_tokens": 180
                    }
                }
            }
            
            # For streaming inference
            stream_mock_response = MagicMock()
            stream_mock_response.status_code = 200
            stream_mock_response.iter_lines.return_value = [
                b'{"text": "This", "metadata": {"finish_reason": null, "is_streaming": true}}',
                b'{"text": "This is", "metadata": {"finish_reason": null, "is_streaming": true}}',
                b'{"text": "This is a", "metadata": {"finish_reason": null, "is_streaming": true}}',
                b'{"text": "This is a test", "metadata": {"finish_reason": "stop", "is_streaming": false}}'
            ]
            
            # Initialize test object
            tester = test_vllm(self.resources, self.metadata)
            
            # Set up endpoint URL
            endpoint_url = self.metadata.get("vllm_api_url", "http://localhost:8000")
            model_name = self.metadata.get("vllm_model", "meta-llama/Llama-2-7b-chat-hf")
            
            # Test single inference performance
            mock_post.return_value = single_mock_response
            
            start_time = time.time()
            for _ in range(10):  # Run 10 iterations for more stable measurement
                data = {"prompt": "Test input for performance measurement"}
                tester.vllm.make_post_request_vllm(endpoint_url, data)
            single_inference_time = (time.time() - start_time) / 10  # Average time
            performance_results["single_inference_time"] = f"{single_inference_time:.4f}s"
            
            # Test batch inference performance
            mock_post.return_value = batch_mock_response
            
            # Test batch processing if implemented
            if hasattr(tester.vllm, 'process_batch'):
                start_time = time.time()
                for _ in range(5):  # Run 5 iterations
                    batch_data = ["Input 1", "Input 2", "Input 3"]
                    tester.vllm.process_batch(endpoint_url, batch_data, model_name)
                batch_time = (time.time() - start_time) / 5  # Average time
                performance_results["batch_inference_time"] = f"{batch_time:.4f}s"
                
                # Calculate throughput
                inputs_per_second = 3 / batch_time  # 3 inputs in batch
                performance_results["inputs_per_second"] = f"{inputs_per_second:.2f}"
                
                # Speedup factor
                speedup = (single_inference_time * 3) / batch_time
                performance_results["batch_speedup_factor"] = f"{speedup:.2f}x"
            
            # Test streaming performance if implemented
            if hasattr(tester.vllm, 'stream_generation'):
                mock_post.return_value = stream_mock_response
                
                start_time = time.time()
                for _ in range(5):  # Run 5 iterations
                    stream_results = []
                    for chunk in tester.vllm.stream_generation(
                        endpoint_url=endpoint_url,
                        prompt="Test streaming performance",
                        model=model_name
                    ):
                        stream_results.append(chunk)
                stream_time = (time.time() - start_time) / 5  # Average time
                performance_results["streaming_time"] = f"{stream_time:.4f}s"
                
                # Compare with non-streaming
                streaming_overhead = stream_time / single_inference_time
                performance_results["streaming_overhead"] = f"{streaming_overhead:.2f}x"
            
            # Test different parameter settings
            parameter_tests = {
                "high_temperature": {"temperature": 0.9, "top_p": 0.95},
                "low_temperature": {"temperature": 0.1, "top_p": 0.1},
                "beam_search": {"use_beam_search": True, "n": 3},
                "greedy": {"temperature": 0.0}
            }
            
            param_times = {}
            for param_name, params in parameter_tests.items():
                start_time = time.time()
                for _ in range(5):
                    data = {"prompt": "Test input for parameter testing", **params}
                    tester.vllm.make_post_request_vllm(endpoint_url, data)
                param_time = (time.time() - start_time) / 5
                param_times[param_name] = param_time
            
            # Find fastest and slowest parameter settings
            fastest_param = min(param_times, key=param_times.get)
            slowest_param = max(param_times, key=param_times.get)
            
            performance_results["parameter_times"] = {
                k: f"{v:.4f}s" for k, v in param_times.items()
            }
            performance_results["fastest_parameter"] = fastest_param
            performance_results["parameter_speedup"] = f"{param_times[slowest_param] / param_times[fastest_param]:.2f}x"
        
        # Save performance results
        performance_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'collected_results', 
            f'vllm_performance_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        os.makedirs(os.path.dirname(performance_file), exist_ok=True)
        with open(performance_file, 'w') as f:
            json.dump(performance_results, f, indent=2)
        
        return performance_results
    
    def test_real_connection(self):
        """Test actual connection to VLLM server if available"""
        # Skip if real connection tests disabled
        if os.environ.get("SKIP_REAL_TESTS", "").lower() in ("true", "1", "yes"):
            self.skipTest("Real connection tests disabled by environment variable")
        
        connection_results = {}
        
        # Try to connect to the VLLM server
        try:
            # Check if VLLM server is running
            endpoint_url = self.metadata.get("vllm_api_url", "http://localhost:8000")
            model_name = self.metadata.get("vllm_model", "meta-llama/Llama-2-7b-chat-hf")
            
            # Check for server health
            try:
                response = requests.get(
                    f"{endpoint_url}/health", 
                    timeout=self.metadata.get("timeout", 30)
                )
                if response.status_code == 200:
                    connection_results["server_health"] = "Success"
                    try:
                        health_info = response.json()
                        connection_results["health_info"] = health_info
                    except:
                        connection_results["health_info"] = "Response not in JSON format"
                    
                    # Try to get model info
                    try:
                        model_response = requests.get(
                            f"{endpoint_url}/model", 
                            timeout=self.metadata.get("timeout", 30)
                        )
                        if model_response.status_code == 200:
                            connection_results["model_available"] = "Success"
                            try:
                                model_info = model_response.json()
                                connection_results["model_info"] = model_info
                            except:
                                connection_results["model_info"] = "Response not in JSON format"
                            
                            # Try a simple inference request
                            try:
                                infer_response = requests.post(
                                    f"{endpoint_url}/generate", 
                                    json={"prompt": "Test input for real connection test", "max_tokens": 20},
                                    headers={"Content-Type": "application/json"},
                                    timeout=self.metadata.get("timeout", 30)
                                )
                                
                                if infer_response.status_code == 200:
                                    connection_results["inference_test"] = "Success"
                                    try:
                                        infer_data = infer_response.json()
                                        if "text" in infer_data:
                                            connection_results["inference_response"] = infer_data["text"][:100] + "..." if len(infer_data["text"]) > 100 else infer_data["text"]
                                        elif "texts" in infer_data:
                                            connection_results["inference_response"] = infer_data["texts"][0][:100] + "..." if len(infer_data["texts"][0]) > 100 else infer_data["texts"][0]
                                        else:
                                            connection_results["inference_response"] = "Response format not recognized"
                                    except Exception:
                                        connection_results["inference_response"] = "Could not parse JSON response"
                                else:
                                    connection_results["inference_test"] = f"Failed with status {infer_response.status_code}"
                            except Exception as e:
                                connection_results["inference_test"] = f"Error: {str(e)}"
                        else:
                            connection_results["model_available"] = f"Failed with status {model_response.status_code}"
                    except Exception as e:
                        connection_results["model_check"] = f"Error: {str(e)}"
                else:
                    connection_results["server_health"] = f"Failed with status {response.status_code}"
            except requests.ConnectionError:
                connection_results["server_health"] = "Failed - Could not connect to VLLM server"
            except Exception as e:
                connection_results["server_health"] = f"Error: {str(e)}"
            
            # Try to test streaming if server is available
            if connection_results.get("inference_test") == "Success":
                try:
                    stream_response = requests.post(
                        f"{endpoint_url}/generate", 
                        json={"prompt": "Test streaming for real connection test", "max_tokens": 20, "stream": True},
                        headers={"Content-Type": "application/json"},
                        timeout=self.metadata.get("timeout", 30),
                        stream=True
                    )
                    
                    if stream_response.status_code == 200:
                        connection_results["streaming_test"] = "Success"
                        
                        # Collect up to 5 chunks to check streaming works
                        stream_chunks = []
                        try:
                            for i, line in enumerate(stream_response.iter_lines()):
                                if i >= 5:
                                    break
                                if line:
                                    try:
                                        chunk = json.loads(line)
                                        if "text" in chunk:
                                            stream_chunks.append(chunk["text"])
                                    except:
                                        pass
                            
                            connection_results["streaming_chunks"] = len(stream_chunks)
                            if stream_chunks:
                                connection_results["streaming_sample"] = stream_chunks[-1][:100]
                        except Exception as e:
                            connection_results["streaming_data"] = f"Error processing stream: {str(e)}"
                    else:
                        connection_results["streaming_test"] = f"Failed with status {stream_response.status_code}"
                except Exception as e:
                    connection_results["streaming_test"] = f"Error: {str(e)}"
            
            # Try to test batch processing if server is available
            if connection_results.get("inference_test") == "Success":
                try:
                    batch_response = requests.post(
                        f"{endpoint_url}/generate", 
                        json={"prompts": ["Test batch 1", "Test batch 2"], "max_tokens": 20},
                        headers={"Content-Type": "application/json"},
                        timeout=self.metadata.get("timeout", 30)
                    )
                    
                    if batch_response.status_code == 200:
                        connection_results["batch_test"] = "Success"
                        try:
                            batch_data = batch_response.json()
                            if "texts" in batch_data:
                                connection_results["batch_count"] = len(batch_data["texts"])
                                if batch_data["texts"]:
                                    connection_results["batch_sample"] = batch_data["texts"][0][:100]
                            else:
                                connection_results["batch_data"] = "Response does not contain 'texts' field"
                        except Exception as e:
                            connection_results["batch_data"] = f"Error parsing batch response: {str(e)}"
                    else:
                        connection_results["batch_test"] = f"Failed with status {batch_response.status_code}"
                except Exception as e:
                    connection_results["batch_test"] = f"Error: {str(e)}"
            
            # Check for LoRA adapters if server is available
            if connection_results.get("inference_test") == "Success":
                try:
                    lora_response = requests.get(
                        f"{endpoint_url}/lora_adapters", 
                        timeout=self.metadata.get("timeout", 30)
                    )
                    
                    if lora_response.status_code == 200:
                        connection_results["lora_check"] = "Success"
                        try:
                            lora_data = lora_response.json()
                            if "lora_adapters" in lora_data:
                                connection_results["lora_count"] = len(lora_data["lora_adapters"])
                                if lora_data["lora_adapters"]:
                                    connection_results["lora_sample"] = lora_data["lora_adapters"][0]
                            else:
                                connection_results["lora_data"] = "Response does not contain 'lora_adapters' field"
                        except Exception:
                            connection_results["lora_data"] = "Could not parse JSON response"
                    else:
                        # This endpoint might not exist in all VLLM servers
                        connection_results["lora_check"] = f"Not available (status {lora_response.status_code})"
                except Exception as e:
                    # Don't treat this as an error since not all VLLM servers support LoRA
                    connection_results["lora_check"] = "Not available"
        
        except Exception as e:
            connection_results["test_error"] = f"Error: {str(e)}"
        
        # Save connection results
        connection_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'collected_results', 
            f'vllm_connection_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        os.makedirs(os.path.dirname(connection_file), exist_ok=True)
        with open(connection_file, 'w') as f:
            json.dump(connection_results, f, indent=2)
        
        return connection_results
    
    def test_advanced_features(self):
        """Test advanced VLLM features if available"""
        # Skip if advanced tests disabled
        if os.environ.get("SKIP_ADVANCED_TESTS", "").lower() in ("true", "1", "yes"):
            self.skipTest("Advanced tests disabled by environment variable")
        
        advanced_results = {}
        
        # Initialize test object
        tester = test_vllm(self.resources, self.metadata)
        endpoint_url = self.metadata.get("vllm_api_url", "http://localhost:8000")
        model_name = self.metadata.get("vllm_model", "meta-llama/Llama-2-7b-chat-hf")
        
        # Test LoRA adapter features
        try:
            if hasattr(tester.vllm, 'list_lora_adapters'):
                with patch.object(requests, 'get') as mock_get:
                    mock_response = MagicMock()
                    mock_response.json.return_value = {
                        "lora_adapters": [
                            {
                                "id": "adapter1",
                                "name": "Test Adapter 1",
                                "base_model": model_name,
                                "size_mb": 12.5,
                                "active": True
                            },
                            {
                                "id": "adapter2",
                                "name": "Test Adapter 2",
                                "base_model": model_name,
                                "size_mb": 8.2,
                                "active": False
                            }
                        ]
                    }
                    mock_response.status_code = 200
                    mock_get.return_value = mock_response
                    
                    adapters = tester.vllm.list_lora_adapters(endpoint_url)
                    advanced_results["list_lora_adapters"] = "Success" if isinstance(adapters, list) and len(adapters) == 2 else "Failed to list LoRA adapters"
                
                # Test inference with LoRA adapter if available
                if hasattr(tester.vllm, 'generate_with_lora'):
                    with patch.object(requests, 'post') as mock_post:
                        mock_response = MagicMock()
                        mock_response.json.return_value = {
                            "text": "Test result with LoRA adapter",
                            "metadata": {
                                "finish_reason": "length",
                                "model": model_name,
                                "lora_adapter": "adapter1",
                                "usage": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 20,
                                    "total_tokens": 30
                                }
                            }
                        }
                        mock_response.status_code = 200
                        mock_post.return_value = mock_response
                        
                        lora_result = tester.vllm.generate_with_lora(
                            endpoint_url=endpoint_url,
                            prompt="Test with LoRA",
                            model=model_name,
                            adapter_id="adapter1"
                        )
                        
                        advanced_results["generate_with_lora"] = "Success" if isinstance(lora_result, dict) and "text" in lora_result else "Failed to generate with LoRA adapter"
            else:
                advanced_results["lora_features"] = "Not implemented"
        except Exception as e:
            advanced_results["lora_features"] = f"Error: {str(e)}"
        
        # Test quantization features
        try:
            if hasattr(tester.vllm, 'set_quantization'):
                with patch.object(requests, 'post') as mock_post:
                    mock_response = MagicMock()
                    mock_response.json.return_value = {
                        "success": True,
                        "message": "Quantization configuration updated",
                        "model": model_name,
                        "quantization": {
                            "enabled": True,
                            "method": "awq",
                            "bits": 4
                        }
                    }
                    mock_response.status_code = 200
                    mock_post.return_value = mock_response
                    
                    # Test all supported quantization methods
                    for method in ["awq", "gptq", "squeezellm"]:
                        for bits in [8, 4, 3]:
                            quant_config = {
                                "enabled": True,
                                "method": method,
                                "bits": bits
                            }
                            
                            quant_result = tester.vllm.set_quantization(endpoint_url, model_name, quant_config)
                            advanced_results[f"quantization_{method}_{bits}bit"] = "Success" if isinstance(quant_result, dict) and quant_result.get("success") else f"Failed to set {method} {bits}-bit quantization"
            else:
                advanced_results["quantization_features"] = "Not implemented"
        except Exception as e:
            advanced_results["quantization_features"] = f"Error: {str(e)}"
        
        # Test tensor parallelism if implemented
        try:
            if hasattr(tester.vllm, 'set_tensor_parallelism'):
                with patch.object(requests, 'post') as mock_post:
                    mock_response = MagicMock()
                    mock_response.json.return_value = {
                        "success": True,
                        "message": "Tensor parallelism configuration updated",
                        "model": model_name,
                        "tensor_parallel_size": 2
                    }
                    mock_response.status_code = 200
                    mock_post.return_value = mock_response
                    
                    tp_result = tester.vllm.set_tensor_parallelism(endpoint_url, model_name, tp_size=2)
                    advanced_results["tensor_parallelism"] = "Success" if isinstance(tp_result, dict) and tp_result.get("success") else "Failed to set tensor parallelism"
            else:
                advanced_results["tensor_parallelism"] = "Not implemented"
        except Exception as e:
            advanced_results["tensor_parallelism"] = f"Error: {str(e)}"
        
        # Test KV cache features if implemented
        try:
            if hasattr(tester.vllm, 'set_kv_cache_config'):
                with patch.object(requests, 'post') as mock_post:
                    mock_response = MagicMock()
                    mock_response.json.return_value = {
                        "success": True,
                        "message": "KV cache configuration updated",
                        "model": model_name,
                        "kv_cache_config": {
                            "block_size": 16,
                            "cache_mode": "auto",
                            "max_blocks_per_seq": 512
                        }
                    }
                    mock_response.status_code = 200
                    mock_post.return_value = mock_response
                    
                    cache_config = {
                        "block_size": 16,
                        "cache_mode": "auto",
                        "max_blocks_per_seq": 512
                    }
                    
                    kv_result = tester.vllm.set_kv_cache_config(endpoint_url, model_name, cache_config)
                    advanced_results["kv_cache_config"] = "Success" if isinstance(kv_result, dict) and kv_result.get("success") else "Failed to set KV cache configuration"
            else:
                advanced_results["kv_cache_features"] = "Not implemented"
        except Exception as e:
            advanced_results["kv_cache_features"] = f"Error: {str(e)}"
        
        # Test prompt caching if implemented
        try:
            if hasattr(tester.vllm, 'use_prompt_cache'):
                with patch.object(requests, 'post') as mock_post:
                    mock_response = MagicMock()
                    mock_response.json.return_value = {
                        "text": "Test result with prompt caching",
                        "metadata": {
                            "finish_reason": "length",
                            "model": model_name,
                            "cached_prompt": True,
                            "prompt_processing_time": 0.0015,  # Much faster with caching
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 20,
                                "total_tokens": 30
                            }
                        }
                    }
                    mock_response.status_code = 200
                    mock_post.return_value = mock_response
                    
                    cache_result = tester.vllm.use_prompt_cache(
                        endpoint_url=endpoint_url,
                        prompt="Test with prompt cache",
                        model=model_name,
                        cache_id="test_cache_123"
                    )
                    
                    advanced_results["prompt_caching"] = "Success" if isinstance(cache_result, dict) and cache_result.get("metadata", {}).get("cached_prompt") is True else "Failed to use prompt caching"
            else:
                advanced_results["prompt_caching"] = "Not implemented"
        except Exception as e:
            advanced_results["prompt_caching"] = f"Error: {str(e)}"
        
        # Save advanced features results
        advanced_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'collected_results', 
            f'vllm_advanced_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        os.makedirs(os.path.dirname(advanced_file), exist_ok=True)
        with open(advanced_file, 'w') as f:
            json.dump(advanced_results, f, indent=2)
        
        return advanced_results
    
    def _save_test_results(self, results, filename):
        """Save test results to file"""
        # Create directories if they don't exist
        base_dir = os.path.dirname(os.path.abspath(__file__))
        collected_dir = os.path.join(base_dir, 'collected_results')
        os.makedirs(collected_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(collected_dir, filename)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

def run_tests():
    """Run all tests or selected tests based on command line arguments"""
    parser = argparse.ArgumentParser(description='Test VLLM API')
    parser.add_argument('--standard', action='store_true', help='Run standard API tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--real', action='store_true', help='Run real connection tests')
    parser.add_argument('--advanced', action='store_true', help='Run advanced feature tests')
    parser.add_argument('--all', action='store_true', help='Run all tests (standard, performance, real, advanced)')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='Model to use for testing')
    parser.add_argument('--api-url', type=str, help='URL for VLLM API')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout in seconds for API requests')
    
    args = parser.parse_args()
    
    # Set environment variables from arguments
    if args.model:
        os.environ["VLLM_MODEL"] = args.model
    if args.api_url:
        os.environ["VLLM_API_URL"] = args.api_url
    if args.timeout:
        os.environ["VLLM_TIMEOUT"] = str(args.timeout)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add standard API tests
    if args.standard or args.all or (not args.standard and not args.performance and not args.real and not args.advanced):
        suite.addTest(TestVLLM('test_standard_api'))
    
    # Add performance tests
    if args.performance or args.all:
        suite.addTest(TestVLLM('test_performance'))
    
    # Add real connection tests
    if args.real or args.all:
        suite.addTest(TestVLLM('test_real_connection'))
    
    # Add advanced feature tests
    if args.advanced or args.all:
        suite.addTest(TestVLLM('test_advanced_features'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Save summary report
    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": os.environ.get("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
        "tests_run": len(result.failures) + len(result.errors) + result.testsRun - len(result.skipped),
        "success": result.wasSuccessful(),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped)
    }
    
    # Create summary file with timestamp
    summary_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'collected_results',
        f'vllm_summary_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests())