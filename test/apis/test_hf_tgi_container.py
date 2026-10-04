import os
import io
import sys
import json
import time
from unittest.mock import MagicMock, patch
import requests

sys.path.append()))))))os.path.join()))))))os.path.dirname()))))))os.path.dirname()))))))os.path.dirname()))))))__file__))), 'ipfs_accelerate_py'))
from ipfs_accelerate_py.api_backends import apis, hf_tgi

class test_hf_tgi_container:
    def __init__()))))))self, resources=None, metadata=None):
        self.resources = resources if resources else {}}}}}}}}
        self.metadata = metadata if metadata else {}}}}}}}:
            "hf_api_key": os.environ.get()))))))"HF_API_KEY", ""),
            "hf_container_url": os.environ.get()))))))"HF_CONTAINER_URL", "http://localhost:8080"),
            "docker_registry": os.environ.get()))))))"DOCKER_REGISTRY", "huggingface/text-generation-inference"),
            "container_tag": os.environ.get()))))))"CONTAINER_TAG", "latest"),
            "gpu_device": os.environ.get()))))))"GPU_DEVICE", "0"),
            "model_id": os.environ.get()))))))"HF_MODEL_ID", "google/t5-efficient-tiny")
            }
            self.hf_tgi = hf_tgi()))))))resources=self.resources, metadata=self.metadata)
        return None
    
    def test()))))))self):
        """Run all tests for the HuggingFace Text Generation Inference API with container orchestration"""
        results = {}}}}}}}}
        
        # Test container deployment
        try:
            container_url = self.metadata.get()))))))"hf_container_url")
            model_id = self.metadata.get()))))))"model_id")
            gpu_device = self.metadata.get()))))))"gpu_device")
            
            with patch()))))))'subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                
                deploy_result = self.deploy_tgi_container()))))))
                model_id=model_id,
                gpu_device=gpu_device
                )
                
                results[]],,"container_deployment"] = "Success" if deploy_result else "Failed to deploy container"
                ,
                # Verify correct docker command was used
                args, kwargs = mock_run.call_args
                cmd = args[]],,0],
                docker_present = "docker" in cmd and "run" in cmd
                model_present = model_id in ' '.join()))))))cmd)
                
                results[]],,"container_command"] = "Success" if docker_present and model_present else "Invalid container command":,
        except Exception as e:
            results[]],,"container_deployment"] = f"Error: {}}}}}}}str()))))))e)}"
            ,
        # Test endpoint connectivity
        try:
            container_url = self.metadata.get()))))))"hf_container_url")
            
            with patch.object()))))))requests, 'get') as mock_get:
                mock_response = MagicMock())))))))
                mock_response.status_code = 200
                mock_response.json.return_value = {}}}}}}}"status": "ok"}
                mock_get.return_value = mock_response
                
                connectivity_result = self.test_container_connectivity()))))))container_url)
                results[]],,"container_connectivity"] = "Success" if connectivity_result else "Failed connectivity test":,
        except Exception as e:
            results[]],,"container_connectivity"] = f"Error: {}}}}}}}str()))))))e)}"
            ,
        # Test endpoint handler creation
        try:
            container_url = self.metadata.get()))))))"hf_container_url")
            
            endpoint_handler = self.hf_tgi.create_remote_text_generation_endpoint_handler()))))))
            container_url, api_key=None  # Container doesn't need API key
            )
            results[]],,"endpoint_handler_creation"] = "Success" if callable()))))))endpoint_handler) else "Failed to create endpoint handler":,
        except Exception as e:
            results[]],,"endpoint_handler_creation"] = f"Error: {}}}}}}}str()))))))e)}"
            ,
        # Test generation with container
        try:
            with patch.object()))))))self.hf_tgi, 'make_post_request_hf_tgi') as mock_post:
                mock_post.return_value = {}}}}}}}"generated_text": "This is a test response from the container"}
                
                generation_result = self.generate_text()))))))
                prompt="Hello, world!",
                max_new_tokens=20,
                temperature=0.7
                )
                
                results[]],,"text_generation"] = "Success" if generation_result else "Failed text generation"
                ,
                # Verify correct parameters were used
                args, kwargs = mock_post.call_args
                data = args[]],,1],
                correct_params = ()))))))
                "inputs" in data and
                "parameters" in data and
                data[]],,"parameters"].get()))))))"max_new_tokens") == 20 and,
                data[]],,"parameters"].get()))))))"temperature") == 0.7,
                )
                
                results[]],,"generation_params"] = "Success" if correct_params else "Failed to set correct parameters":,
        except Exception as e:
            results[]],,"text_generation"] = f"Error: {}}}}}}}str()))))))e)}"
            ,
        # Test streaming generation with container
        try:
            with patch.object()))))))self.hf_tgi, 'make_stream_request_hf_tgi') as mock_stream:
                # Simulate streaming response
                mock_stream.return_value = iter()))))))[]],,
                {}}}}}}}"generated_text": "This"},
                {}}}}}}}"generated_text": "This is"},
                {}}}}}}}"generated_text": "This is a"},
                {}}}}}}}"generated_text": "This is a streaming"},
                {}}}}}}}"generated_text": "This is a streaming response"}
                ])
                
                stream_result = list()))))))self.stream_generate()))))))
                prompt="Hello, stream!",
                max_new_tokens=20
                ))
                
                streaming_success = len()))))))stream_result) > 1 and "streaming" in stream_result[]],,-1]
                results[]],,"streaming_generation"] = "Success" if streaming_success else "Failed streaming generation":
        except Exception as e:
            results[]],,"streaming_generation"] = f"Error: {}}}}}}}str()))))))e)}"
        
        # Test container health monitoring
        try:
            with patch.object()))))))requests, 'get') as mock_get:
                mock_response = MagicMock())))))))
                mock_response.status_code = 200
                mock_response.json.return_value = {}}}}}}}
                "model_id": self.metadata.get()))))))"model_id"),
                "model_sha": "abc123",
                "max_batch_size": 32,
                "max_input_length": 1024,
                "max_total_tokens": 2048,
                "waiting_served_ratio": 1.5,
                "max_waiting_tokens": 20,
                "max_batch_tokens": 1000,
                "requires_padding": True,
                "tokenizer_model_id": "gpt2"
                }
                mock_get.return_value = mock_response
                
                health_result = self.check_container_health()))))))self.metadata.get()))))))"hf_container_url"))
                results[]],,"health_check"] = "Success" if isinstance()))))))health_result, dict) and "model_id" in health_result else "Failed health check":
        except Exception as e:
            results[]],,"health_check"] = f"Error: {}}}}}}}str()))))))e)}"
        
        # Test container shutdown
        try:
            with patch()))))))'subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                
                shutdown_result = self.shutdown_container())))))))
                results[]],,"container_shutdown"] = "Success" if shutdown_result else "Failed to shut down container"
                
                # Verify correct docker command was used for shutdown
                args, kwargs = mock_run.call_args
                cmd = args[]],,0],
                docker_stop = "docker" in cmd and "stop" in cmd
                
                results[]],,"shutdown_command"] = "Success" if docker_stop else "Invalid shutdown command":
        except Exception as e:
            results[]],,"container_shutdown"] = f"Error: {}}}}}}}str()))))))e)}"
            
                    return results
    
    def deploy_tgi_container()))))))self, model_id, gpu_device="0", port=8080):
        """Deploy a TGI container for the specified model"""
        try:
            import subprocess
            
            # Get container configuration
            registry = self.metadata.get()))))))"docker_registry", "huggingface/text-generation-inference")
            tag = self.metadata.get()))))))"container_tag", "latest")
            
            # Prepare docker run command
            docker_cmd = []],,
            "docker", "run", "-d",
            "--name", f"tgi-{}}}}}}}model_id.replace()))))))'/', '-')}",
            "-p", f"{}}}}}}}port}:80",
            "--gpus", f"device={}}}}}}}gpu_device}",
            "--shm-size", "1g",  # Shared memory size for container
            "-e", f"MODEL_ID={}}}}}}}model_id}",
            "-e", "NUM_SHARD=1",  # Number of model shards ()))))))for multiple GPUs)
            "-e", "MAX_INPUT_LENGTH=1024",  # Maximum input length
            "-e", "MAX_TOTAL_TOKENS=2048",  # Maximum total tokens ()))))))input + output)
            "-e", "TRUST_REMOTE_CODE=true",  # Trust remote code from HF
            f"{}}}}}}}registry}:{}}}}}}}tag}"
            ]
            
            # Execute command
            result = subprocess.run()))))))docker_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print()))))))f"Error deploying container: {}}}}}}}result.stderr}")
            return False
            
            # Store container ID for later
            self.container_id = result.stdout.strip())))))))
            
            # Wait for container to start up
            time.sleep()))))))10)  # Give the container time to initialize
            
        return True
        except Exception as e:
            print()))))))f"Container deployment error: {}}}}}}}str()))))))e)}")
        return False
    
    def test_container_connectivity()))))))self, container_url):
        """Test if the TGI container is running and responding""":
        try:
            # Try to access the container health endpoint
            response = requests.get()))))))f"{}}}}}}}container_url}/health")
            
            if response.status_code == 200:
                health_data = response.json())))))))
                # Verify the expected model is loaded
                if "model_id" in health_data and health_data[]],,"model_id"] == self.metadata.get()))))))"model_id"):
                return True
            
            return False
        except Exception as e:
            print()))))))f"Container connectivity error: {}}}}}}}str()))))))e)}")
            return False
    
    def generate_text()))))))self, prompt, max_new_tokens=20, temperature=0.7, top_p=0.9):
        """Generate text using the TGI container"""
        try:
            container_url = self.metadata.get()))))))"hf_container_url")
            
            # Check if using T5 model and format prompt appropriately
            model_id = self.metadata.get()))))))"model_id", "")
            is_t5_model = "t5" in model_id.lower())))))))
            ::
            # T5 models typically expect a prefix like "translate English to German: " or "summarize: "
            if is_t5_model and not prompt.startswith()))))))"translate:") and not prompt.startswith()))))))"summarize:"):
                formatted_prompt = f"summarize: {}}}}}}}prompt}"
            else:
                formatted_prompt = prompt
            
            # Prepare request data
                data = {}}}}}}}
                "inputs": formatted_prompt,
                "parameters": {}}}}}}}
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "return_full_text": False
                }
                }
            
            # Make request to container
                response = self.hf_tgi.make_post_request_hf_tgi()))))))
                endpoint_url=container_url + "/generate",
                data=data,
                api_key=None  # No API key needed for local container
                )
            
                return response.get()))))))"generated_text", "")
        except Exception as e:
            print()))))))f"Text generation error: {}}}}}}}str()))))))e)}")
                return ""
    
    def stream_generate()))))))self, prompt, max_new_tokens=20, temperature=0.7):
        """Stream text generation using the TGI container"""
        try:
            container_url = self.metadata.get()))))))"hf_container_url")
            
            # Check if using T5 model and format prompt appropriately
            model_id = self.metadata.get()))))))"model_id", "")
            is_t5_model = "t5" in model_id.lower())))))))
            ::
            # T5 models typically expect a prefix like "translate English to German: " or "summarize: "
            if is_t5_model and not prompt.startswith()))))))"translate:") and not prompt.startswith()))))))"summarize:"):
                formatted_prompt = f"summarize: {}}}}}}}prompt}"
            else:
                formatted_prompt = prompt
                
            # Prepare request data
                data = {}}}}}}}
                "inputs": formatted_prompt,
                "parameters": {}}}}}}}
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "return_full_text": False,
                "stream": True
                }
                }
            
            # Make streaming request
            for chunk in self.hf_tgi.make_stream_request_hf_tgi())))))):
                endpoint_url=container_url + "/generate_stream",
                data=data,
                api_key=None  # No API key needed for local container
            ):
                yield chunk.get()))))))"generated_text", "")
        except Exception as e:
            print()))))))f"Stream generation error: {}}}}}}}str()))))))e)}")
            yield f"Error: {}}}}}}}str()))))))e)}"
    
    def check_container_health()))))))self, container_url):
        """Check the health of the TGI container"""
        try:
            response = requests.get()))))))f"{}}}}}}}container_url}/health")
            if response.status_code == 200:
            return response.json())))))))
        return {}}}}}}}"status": "error", "code": response.status_code}
        except Exception as e:
        return {}}}}}}}"status": "error", "message": str()))))))e)}
    
    def shutdown_container()))))))self):
        """Shut down the TGI container"""
        try:
            import subprocess
            
            # Get container ID
            container_id = getattr()))))))self, 'container_id', None)
            if not container_id:
                # Try to find container by name
                model_id = self.metadata.get()))))))"model_id", "gpt2")
                container_name = f"tgi-{}}}}}}}model_id.replace()))))))'/', '-')}"
                docker_cmd = []],,"docker", "stop", container_name]
            else:
                docker_cmd = []],,"docker", "stop", container_id]
            
            # Execute command
                result = subprocess.run()))))))docker_cmd, capture_output=True, text=True)
            
                return result.returncode == 0
        except Exception as e:
            print()))))))f"Container shutdown error: {}}}}}}}str()))))))e)}")
                return False
    
    def __test__()))))))self):
        """Run tests and compare/save results"""
        test_results = {}}}}}}}}
        try:
            test_results = self.test())))))))
        except Exception as e:
            test_results = {}}}}}}}"test_error": str()))))))e)}
        
        # Create directories if they don't exist
            base_dir = os.path.dirname()))))))os.path.abspath()))))))__file__))
            expected_dir = os.path.join()))))))base_dir, 'expected_results')
            collected_dir = os.path.join()))))))base_dir, 'collected_results')
        
        # Create directories with appropriate permissions:
        for directory in []],,expected_dir, collected_dir]:
            if not os.path.exists()))))))directory):
                os.makedirs()))))))directory, mode=0o755, exist_ok=True)
        
        # Save collected results
                results_file = os.path.join()))))))collected_dir, 'hf_tgi_container_test_results.json')
        try:
            # Try to serialize the results to catch any non-serializable objects
            try:
                json_str = json.dumps()))))))test_results)
            except TypeError as e:
                print()))))))f"Warning: Test results contain non-serializable data: {}}}}}}}str()))))))e)}")
                # Convert non-serializable objects to strings
                serializable_results = {}}}}}}}}
                for k, v in test_results.items()))))))):
                    try:
                        json.dumps())))))){}}}}}}}k: v})
                        serializable_results[]],,k] = v
                    except TypeError:
                        serializable_results[]],,k] = str()))))))v)
                        test_results = serializable_results
            
            with open()))))))results_file, 'w') as f:
                json.dump()))))))test_results, f, indent=2)
        except Exception as e:
            print()))))))f"Error saving results to {}}}}}}}results_file}: {}}}}}}}str()))))))e)}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join()))))))expected_dir, 'hf_tgi_container_test_results.json'):
        if os.path.exists()))))))expected_file):
            try:
                with open()))))))expected_file, 'r') as f:
                    expected_results = json.load()))))))f)
                    if expected_results != test_results:
                        print()))))))"Test results differ from expected results!")
                        print()))))))f"Expected: {}}}}}}}json.dumps()))))))expected_results, indent=2)}")
                        print()))))))f"Got: {}}}}}}}json.dumps()))))))test_results, indent=2)}")
            except Exception as e:
                print()))))))f"Error comparing results with {}}}}}}}expected_file}: {}}}}}}}str()))))))e)}")
        else:
            # Create expected results file if it doesn't exist:
            try:
                with open()))))))expected_file, 'w') as f:
                    json.dump()))))))test_results, f, indent=2)
                    print()))))))f"Created new expected results file: {}}}}}}}expected_file}")
            except Exception as e:
                print()))))))f"Error creating {}}}}}}}expected_file}: {}}}}}}}str()))))))e)}")

                    return test_results

if __name__ == "__main__":
    metadata = {}}}}}}}
    "hf_api_key": os.environ.get()))))))"HF_API_KEY", ""),
    "hf_container_url": os.environ.get()))))))"HF_CONTAINER_URL", "http://localhost:8080"),
    "docker_registry": os.environ.get()))))))"DOCKER_REGISTRY", "huggingface/text-generation-inference"),
    "container_tag": os.environ.get()))))))"CONTAINER_TAG", "latest"),
    "gpu_device": os.environ.get()))))))"GPU_DEVICE", "0"),
    "model_id": os.environ.get()))))))"HF_MODEL_ID", "gpt2")
    }
    resources = {}}}}}}}}
    try:
        this_hf_tgi_container = test_hf_tgi_container()))))))resources, metadata)
        results = this_hf_tgi_container.__test__())))))))
        print()))))))f"HF TGI Container API Test Results: {}}}}}}}json.dumps()))))))results, indent=2)}")
    except KeyboardInterrupt:
        print()))))))"Tests stopped by user.")
        # Try to clean up container if possible:
        try:
            this_hf_tgi_container.shutdown_container())))))))
        except:
            pass
            sys.exit()))))))1)