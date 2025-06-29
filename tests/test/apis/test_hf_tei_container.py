import os
import io
import sys
import json
import time
from unittest.mock import MagicMock, patch
import requests
import numpy as np

sys.path.append()))))os.path.join()))))os.path.dirname()))))os.path.dirname()))))os.path.dirname()))))__file__))), 'ipfs_accelerate_py'))
from ipfs_accelerate_py.api_backends import apis, hf_tei

class test_hf_tei_container:
    def __init__()))))self, resources=None, metadata=None):
        self.resources = resources if resources else {}}}}}}}
        self.metadata = metadata if metadata else {}}}}}}:
            "hf_api_key": os.environ.get()))))"HF_API_KEY", ""),
            "hf_container_url": os.environ.get()))))"HF_CONTAINER_URL", "http://localhost:8080"),
            "docker_registry": os.environ.get()))))"DOCKER_REGISTRY", "ghcr.io/huggingface/text-embeddings-inference"),
            "container_tag": os.environ.get()))))"CONTAINER_TAG", "latest"),
            "gpu_device": os.environ.get()))))"GPU_DEVICE", "0"),
            "model_id": os.environ.get()))))"HF_MODEL_ID", "BAAI/bge-small-en-v1.5")
            }
            self.hf_tei = hf_tei()))))resources=self.resources, metadata=self.metadata)
        return None
    
    def test()))))self):
        """Run all tests for the HuggingFace Text Embedding Inference with container deployment"""
        results = {}}}}}}}
        
        # Test container deployment
        try:
            container_url = self.metadata.get()))))"hf_container_url")
            model_id = self.metadata.get()))))"model_id")
            gpu_device = self.metadata.get()))))"gpu_device")
            
            with patch()))))'subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                
                deploy_result = self.deploy_tei_container()))))
                model_id=model_id,
                gpu_device=gpu_device
                )
                
                results[],"container_deployment"] = "Success" if deploy_result else "Failed to deploy container"
                ,
                # Verify correct docker command was used
                args, kwargs = mock_run.call_args
                cmd = args[],0],,
                docker_present = "docker" in cmd and "run" in cmd
                model_present = model_id in ' '.join()))))cmd)
                
                results[],"container_command"] = "Success" if docker_present and model_present else "Invalid container command":,
        except Exception as e:
            results[],"container_deployment"] = f"Error: {}}}}}}str()))))e)}"
            ,
        # Test endpoint connectivity
        try:
            container_url = self.metadata.get()))))"hf_container_url")
            
            with patch.object()))))requests, 'get') as mock_get:
                mock_response = MagicMock())))))
                mock_response.status_code = 200
                mock_response.json.return_value = {}}}}}}"status": "ok"}
                mock_get.return_value = mock_response
                
                connectivity_result = self.test_container_connectivity()))))container_url)
                results[],"container_connectivity"] = "Success" if connectivity_result else "Failed connectivity test":,
        except Exception as e:
            results[],"container_connectivity"] = f"Error: {}}}}}}str()))))e)}"
            ,
        # Test endpoint handler creation
        try:
            container_url = self.metadata.get()))))"hf_container_url")
            
            endpoint_handler = self.hf_tei.create_remote_text_embedding_endpoint_handler()))))
            container_url, api_key=None  # Container doesn't need API key
            )
            results[],"endpoint_handler_creation"] = "Success" if callable()))))endpoint_handler) else "Failed to create endpoint handler":,
        except Exception as e:
            results[],"endpoint_handler_creation"] = f"Error: {}}}}}}str()))))e)}"
            ,
        # Test embedding generation
        try:
            with patch.object()))))self.hf_tei, 'make_post_request_hf_tei') as mock_post:
                # Create a mock embedding with proper shape
                mock_post.return_value = [],0.1, 0.2, 0.3] * 10,0  # 300-dimensional vector
                ,
                embedding = self.generate_embedding()))))"Test text for embedding")
                
                results[],"embedding_generation"] = "Success" if isinstance()))))embedding, list) and len()))))embedding) > 0 else "Failed embedding generation"
                ,
                # Verify correct parameters were used
                args, kwargs = mock_post.call_args
                data = args[],1]
                ,,
                results[],"embedding_params"] = "Success" if "inputs" in data else "Failed to set correct parameters":,
        except Exception as e:
            results[],"embedding_generation"] = f"Error: {}}}}}}str()))))e)}"
            ,
        # Test batch embedding
        try:
            with patch.object()))))self.hf_tei, 'make_post_request_hf_tei') as mock_post:
                # Create a mock batch response ()))))list of embeddings)
                mock_post.return_value = [],[],0.1, 0.2, 0.3] * 10,0, [],0.4, 0.5, 0.6] * 100]
                ,
                batch_texts = [],"First test text", "Second test text"],
                batch_embeddings = self.generate_batch_embeddings()))))batch_texts)
                
                batch_success = ()))))
                isinstance()))))batch_embeddings, list) and
                len()))))batch_embeddings) == 2 and
                isinstance()))))batch_embeddings[],0],,, list) and
                len()))))batch_embeddings[],0],,) > 0
                )
                
                results[],"batch_embedding"] = "Success" if batch_success else "Failed batch embedding"
                ,
                # Verify correct parameters
                args, kwargs = mock_post.call_args
                data = args[],1]
                ,,
                results[],"batch_params"] = "Success" if "inputs" in data and isinstance()))))data[],"inputs"], list) else "Failed to set correct batch parameters":,
        except Exception as e:
            results[],"batch_embedding"] = f"Error: {}}}}}}str()))))e)}"
            ,
        # Test vector operations
        try:
            # Create two sample vectors
            v1 = [],0.1, 0.2, 0.3] * 10,
            v2 = [],0.4, 0.5, 0.6] * 10
            ,
            # Test vector normalization
            norm1 = self.normalize_vector()))))v1)
            norm2 = self.normalize_vector()))))v2)
            
            # Calculate cosine similarity
            similarity = self.calculate_similarity()))))norm1, norm2)
            
            results[],"vector_operations"] = "Success" if isinstance()))))similarity, float) and -1.0 <= similarity <= 1.0 else "Failed vector operations":,
        except Exception as e:
            results[],"vector_operations"] = f"Error: {}}}}}}str()))))e)}"
            ,
        # Test model information retrieval
        try:
            with patch.object()))))requests, 'get') as mock_get:
                mock_response = MagicMock())))))
                mock_response.status_code = 200
                mock_response.json.return_value = {}}}}}}
                "model_id": self.metadata.get()))))"model_id"),
                "model_sha": "abc123",
                "dim": 384,  # Typical embedding dimension
                "pooling": "mean",
                "normalized": True,
                "max_sequence_length": 512
                }
                mock_get.return_value = mock_response
                
                model_info = self.get_model_info()))))self.metadata.get()))))"hf_container_url"))
                results[],"model_info"] = "Success" if isinstance()))))model_info, dict) and "dim" in model_info else "Failed model info retrieval":,
        except Exception as e:
            results[],"model_info"] = f"Error: {}}}}}}str()))))e)}"
            ,
        # Test container shutdown
        try:
            with patch()))))'subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                
                shutdown_result = self.shutdown_container())))))
                results[],"container_shutdown"] = "Success" if shutdown_result else "Failed to shut down container"
                ,
                # Verify correct docker command was used for shutdown
                args, kwargs = mock_run.call_args
                cmd = args[],0],,
                docker_stop = "docker" in cmd and "stop" in cmd
                
                results[],"shutdown_command"] = "Success" if docker_stop else "Invalid shutdown command":,
        except Exception as e:
            results[],"container_shutdown"] = f"Error: {}}}}}}str()))))e)}"
            ,
                return results
    
    def deploy_tei_container()))))self, model_id, gpu_device="0", port=8080):
        """Deploy a TEI container for the specified model"""
        try:
            import subprocess
            
            # Get container configuration
            registry = self.metadata.get()))))"docker_registry", "ghcr.io/huggingface/text-embeddings-inference")
            tag = self.metadata.get()))))"container_tag", "latest")
            
            # Convert model_id to a safe container name
            container_name = f"tei-{}}}}}}model_id.replace()))))'/', '-').replace()))))'.', '-')}"
            
            # Prepare docker run command
            docker_cmd = [],
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{}}}}}}port}:80",
            "--gpus", f"device={}}}}}}gpu_device}",
            "--shm-size", "1g",  # Shared memory size for container
            "-e", f"MODEL_ID={}}}}}}model_id}",
            "-e", "TOKENIZERS_PARALLELISM=false",  # Avoid warnings
            "-e", "NUM_WORKER_THREADS=4",  # Worker threads
            "-e", "MAX_BATCH_SIZE=32",     # Maximum batch size
            "-e", "MAX_SEQUENCE_LENGTH=512", # Maximum sequence length
            f"{}}}}}}registry}:{}}}}}}tag}"
            ]
            
            # Execute command
            result = subprocess.run()))))docker_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print()))))f"Error deploying container: {}}}}}}result.stderr}")
            return False
            
            # Store container ID for later
            self.container_id = result.stdout.strip())))))
            self.container_name = container_name
            
            # Wait for container to start up
            time.sleep()))))10)  # Give the container time to initialize
            
        return True
        except Exception as e:
            print()))))f"Container deployment error: {}}}}}}str()))))e)}")
        return False
    
    def test_container_connectivity()))))self, container_url):
        """Test if the TEI container is running and responding""":
        try:
            # Try to access the container info endpoint
            response = requests.get()))))f"{}}}}}}container_url}/info")
            
            if response.status_code == 200:
                info_data = response.json())))))
                # Verify the expected model is loaded
                if "model_id" in info_data and info_data[],"model_id"] == self.metadata.get()))))"model_id"):
                return True
            
            # Alternative health check endpoint
                response = requests.get()))))f"{}}}}}}container_url}/health")
            return response.status_code == 200
            
        except Exception as e:
            print()))))f"Container connectivity error: {}}}}}}str()))))e)}")
            return False
    
    def generate_embedding()))))self, text):
        """Generate an embedding using the TEI container"""
        try:
            container_url = self.metadata.get()))))"hf_container_url")
            
            # Prepare request data
            data = {}}}}}}
            "inputs": text
            }
            
            # Make request to container
            response = self.hf_tei.make_post_request_hf_tei()))))
            endpoint_url=container_url + "/embed",
            data=data,
            api_key=None  # No API key needed for local container
            )
            
            # TEI returns the embedding vector directly
        return response
        except Exception as e:
            print()))))f"Embedding generation error: {}}}}}}str()))))e)}")
        return [],]
    
    def generate_batch_embeddings()))))self, texts):
        """Generate embeddings for a batch of texts"""
        try:
            container_url = self.metadata.get()))))"hf_container_url")
            
            # Prepare request data
            data = {}}}}}}
            "inputs": texts
            }
            
            # Make request to container
            response = self.hf_tei.make_post_request_hf_tei()))))
            endpoint_url=container_url + "/embed",
            data=data,
            api_key=None  # No API key needed for local container
            )
            
            # TEI returns list of embedding vectors for batch inputs
        return response
        except Exception as e:
            print()))))f"Batch embedding error: {}}}}}}str()))))e)}")
        return [],]
    
    def normalize_vector()))))self, vector):
        """Normalize an embedding vector to unit length"""
        try:
            # Convert to numpy array for easier operations
            vec = np.array()))))vector)
            
            # Calculate vector magnitude
            magnitude = np.linalg.norm()))))vec)
            
            # Avoid division by zero
            if magnitude > 0:
                normalized = vec / magnitude
            else:
                normalized = vec
                
            # Convert back to list
                return normalized.tolist())))))
        except Exception as e:
            print()))))f"Vector normalization error: {}}}}}}str()))))e)}")
                return vector
    
    def calculate_similarity()))))self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        try:
            # Convert to numpy arrays
            v1 = np.array()))))vec1)
            v2 = np.array()))))vec2)
            
            # Calculate dot product
            dot_product = np.dot()))))v1, v2)
            
            # Calculate magnitudes
            mag1 = np.linalg.norm()))))v1)
            mag2 = np.linalg.norm()))))v2)
            
            # Calculate cosine similarity
            if mag1 > 0 and mag2 > 0:
                similarity = dot_product / ()))))mag1 * mag2)
            else:
                similarity = 0.0
                
                return float()))))similarity)
        except Exception as e:
            print()))))f"Similarity calculation error: {}}}}}}str()))))e)}")
                return 0.0
    
    def get_model_info()))))self, container_url):
        """Get model information from the TEI container"""
        try:
            response = requests.get()))))f"{}}}}}}container_url}/info")
            if response.status_code == 200:
            return response.json())))))
        return {}}}}}}"status": "error", "code": response.status_code}
        except Exception as e:
        return {}}}}}}"status": "error", "message": str()))))e)}
    
    def shutdown_container()))))self):
        """Shut down the TEI container"""
        try:
            import subprocess
            
            # Check if we have container info
            container_id = getattr()))))self, 'container_id', None)
            container_name = getattr()))))self, 'container_name', None)
            :
            if not container_id and not container_name:
                # Try to find container by model ID
                model_id = self.metadata.get()))))"model_id", "")
                container_name = f"tei-{}}}}}}model_id.replace()))))'/', '-').replace()))))'.', '-')}"
            
            # Use container name if available, otherwise use ID:
            if container_name:
                docker_cmd = [],"docker", "stop", container_name]
            else:
                docker_cmd = [],"docker", "stop", container_id]
            
            # Execute command
                result = subprocess.run()))))docker_cmd, capture_output=True, text=True)
            
                return result.returncode == 0
        except Exception as e:
            print()))))f"Container shutdown error: {}}}}}}str()))))e)}")
                return False
    
    def test_advanced_features()))))self):
        """Test advanced features of the TEI container"""
        advanced_results = {}}}}}}}
        
        # Test multiple input formats
        try:
            with patch.object()))))self.hf_tei, 'make_post_request_hf_tei') as mock_post:
                # Mock response for embedding
                mock_post.return_value = [],0.1, 0.2, 0.3] * 10,0  # 300-dimensional vector
                ,
                container_url = self.metadata.get()))))"hf_container_url")
                
                # Test with dictionary input with additional options
                input_with_options = {}}}}}}
                "text": "Test sentence",
                "truncate": True,
                "normalize": True
                }
                
                # If format_options method exists, test it
                if hasattr()))))self.hf_tei, 'format_options'):
                    formatted_data = self.hf_tei.format_options()))))input_with_options)
                    advanced_results[],"format_options"] = "Success" if "inputs" in formatted_data else "Failed to format options":
                else:
                    advanced_results[],"format_options"] = "Method not implemented"
                    
                # Test with raw inputs ()))))direct text)
                if hasattr()))))self.hf_tei, 'raw_embed'):
                    raw_result = self.hf_tei.raw_embed()))))container_url + "/embed", "Raw text input", None)
                    advanced_results[],"raw_embed"] = "Success" if isinstance()))))raw_result, list) else "Failed raw embedding":
                else:
                    advanced_results[],"raw_embed"] = "Method not implemented"
        except Exception as e:
            advanced_results[],"input_format_tests"] = f"Error: {}}}}}}str()))))e)}"
            
        # Test pooling strategies if implemented::
        try:
            with patch.object()))))requests, 'post') as mock_post:
                mock_response = MagicMock())))))
                mock_response.status_code = 200
                mock_response.json.return_value = [],0.1, 0.2, 0.3] * 10,0
                mock_post.return_value = mock_response
                
                container_url = self.metadata.get()))))"hf_container_url")
                
                # Test with different pooling strategies if supported:
                if hasattr()))))self.hf_tei, 'embed_with_pooling'):
                    # Test CLS pooling
                    cls_result = self.hf_tei.embed_with_pooling()))))container_url, "Test text", None, pooling="cls")
                    advanced_results[],"cls_pooling"] = "Success" if isinstance()))))cls_result, list) else "Failed CLS pooling"
                    
                    # Test mean pooling
                    mean_result = self.hf_tei.embed_with_pooling()))))container_url, "Test text", None, pooling="mean")
                    advanced_results[],"mean_pooling"] = "Success" if isinstance()))))mean_result, list) else "Failed mean pooling":
                else:
                    advanced_results[],"pooling_strategies"] = "Method not implemented"
        except Exception as e:
            advanced_results[],"pooling_tests"] = f"Error: {}}}}}}str()))))e)}"
        
        # Test container health monitoring
        try:
            with patch.object()))))requests, 'get') as mock_get:
                mock_response = MagicMock())))))
                mock_response.status_code = 200
                mock_response.json.return_value = {}}}}}}"status": "ok", "load": 0.1}
                mock_get.return_value = mock_response
                
                container_url = self.metadata.get()))))"hf_container_url")
                
                # Test container health monitoring if implemented::
                if hasattr()))))self, 'check_container_health'):
                    health_status = self.check_container_health()))))container_url)
                    advanced_results[],"container_health"] = "Success" if health_status.get()))))"status") == "ok" else "Failed health check":
                else:
                    # Simple implementation for testing
                    def check_container_health()))))url):
                        response = requests.get()))))f"{}}}}}}url}/health")
                        if response.status_code == 200:
                        return response.json())))))
                    return {}}}}}}"status": "error", "code": response.status_code}
                    
                    health_status = check_container_health()))))container_url)
                    advanced_results[],"basic_health_check"] = "Success" if health_status.get()))))"status") == "ok" else "Failed basic health check":
        except Exception as e:
            advanced_results[],"health_monitoring"] = f"Error: {}}}}}}str()))))e)}"
            
                        return advanced_results
    
    def check_container_health()))))self, container_url):
        """Check the health status of the TEI container"""
        try:
            response = requests.get()))))f"{}}}}}}container_url}/health")
            if response.status_code == 200:
            return response.json())))))
        return {}}}}}}"status": "error", "code": response.status_code}
        except Exception as e:
        return {}}}}}}"status": "error", "message": str()))))e)}
    
    def __test__()))))self):
        """Run tests and compare/save results"""
        test_results = {}}}}}}}
        try:
            # Run standard tests
            test_results = self.test())))))
            
            # Run advanced tests
            advanced_results = self.test_advanced_features())))))
            
            # Merge results
            test_results.update()))))advanced_results)
        except Exception as e:
            test_results = {}}}}}}"test_error": str()))))e)}
        
        # Create directories if they don't exist
            base_dir = os.path.dirname()))))os.path.abspath()))))__file__))
            expected_dir = os.path.join()))))base_dir, 'expected_results')
            collected_dir = os.path.join()))))base_dir, 'collected_results')
        
        # Create directories with appropriate permissions:
        for directory in [],expected_dir, collected_dir]:
            if not os.path.exists()))))directory):
                os.makedirs()))))directory, mode=0o755, exist_ok=True)
        
        # Save collected results
                results_file = os.path.join()))))collected_dir, 'hf_tei_container_test_results.json')
        try:
            # Try to serialize the results to catch any non-serializable objects
            try:
                json_str = json.dumps()))))test_results)
            except TypeError as e:
                print()))))f"Warning: Test results contain non-serializable data: {}}}}}}str()))))e)}")
                # Convert non-serializable objects to strings
                serializable_results = {}}}}}}}
                for k, v in test_results.items()))))):
                    try:
                        json.dumps())))){}}}}}}k: v})
                        serializable_results[],k] = v
                    except TypeError:
                        serializable_results[],k] = str()))))v)
                        test_results = serializable_results
            
            with open()))))results_file, 'w') as f:
                json.dump()))))test_results, f, indent=2)
        except Exception as e:
            print()))))f"Error saving results to {}}}}}}results_file}: {}}}}}}str()))))e)}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join()))))expected_dir, 'hf_tei_container_test_results.json'):
        if os.path.exists()))))expected_file):
            try:
                with open()))))expected_file, 'r') as f:
                    expected_results = json.load()))))f)
                    if expected_results != test_results:
                        print()))))"Test results differ from expected results!")
                        print()))))f"Expected: {}}}}}}json.dumps()))))expected_results, indent=2)}")
                        print()))))f"Got: {}}}}}}json.dumps()))))test_results, indent=2)}")
            except Exception as e:
                print()))))f"Error comparing results with {}}}}}}expected_file}: {}}}}}}str()))))e)}")
        else:
            # Create expected results file if it doesn't exist:
            try:
                with open()))))expected_file, 'w') as f:
                    json.dump()))))test_results, f, indent=2)
                    print()))))f"Created new expected results file: {}}}}}}expected_file}")
            except Exception as e:
                print()))))f"Error creating {}}}}}}expected_file}: {}}}}}}str()))))e)}")

                    return test_results

if __name__ == "__main__":
    metadata = {}}}}}}
    "hf_api_key": os.environ.get()))))"HF_API_KEY", ""),
    "hf_container_url": os.environ.get()))))"HF_CONTAINER_URL", "http://localhost:8080"),
    "docker_registry": os.environ.get()))))"DOCKER_REGISTRY", "ghcr.io/huggingface/text-embeddings-inference"),
    "container_tag": os.environ.get()))))"CONTAINER_TAG", "latest"),
    "gpu_device": os.environ.get()))))"GPU_DEVICE", "0"),
    "model_id": os.environ.get()))))"HF_MODEL_ID", "BAAI/bge-small-en-v1.5")
    }
    resources = {}}}}}}}
    try:
        this_hf_tei_container = test_hf_tei_container()))))resources, metadata)
        results = this_hf_tei_container.__test__())))))
        print()))))f"HF TEI Container API Test Results: {}}}}}}json.dumps()))))results, indent=2)}")
    except KeyboardInterrupt:
        print()))))"Tests stopped by user.")
        # Try to clean up container if possible:
        try:
            this_hf_tei_container.shutdown_container())))))
        except:
            pass
            sys.exit()))))1)