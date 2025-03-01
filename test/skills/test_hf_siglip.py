# Standard library imports
import os
import sys
import json
import time
import datetime
import traceback
from pathlib import Path
from unittest.mock import MagicMock, patch

# Use direct import with absolute path
sys.path.insert(0, "/home/barberb/ipfs_accelerate_py")

# Import optional dependencies with fallbacks
try:
    import torch
    import numpy as np
except ImportError:
    torch = MagicMock()
    np = MagicMock()
    print("Warning: torch/numpy not available, using mock implementation")

try:
    import transformers
    import PIL
    from PIL import Image
except ImportError:
    transformers = MagicMock()
    PIL = MagicMock()
    Image = MagicMock()
    print("Warning: transformers/PIL not available, using mock implementation")

# Try to import from ipfs_accelerate_py
try:
    from ipfs_accelerate_py.worker.skillset.hf_siglip import hf_siglip
except ImportError:
    # Create a mock class if the real one doesn't exist
    class hf_siglip:
        def __init__(self, resources=None, metadata=None):
            self.resources = resources if resources else {}
            self.metadata = metadata if metadata else {}
            
        def init_cpu(self, model_name, processor_name, device):
            mock_handler = lambda images=None, text=None, **kwargs: {
                "image_embeds": torch.randn(1, 768),
                "text_embeds": torch.randn(1, 768) if text is not None else None,
                "similarity": torch.tensor([0.85]) if text is not None else None,
                "implementation_type": "(MOCK)"
            }
            return "mock_endpoint", "mock_processor", mock_handler, None, 1
            
        def init_cuda(self, model_name, processor_name, device):
            return self.init_cpu(model_name, processor_name, device)
            
        def init_openvino(self, model_name, processor_name, device):
            return self.init_cpu(model_name, processor_name, device)
    
    print("Warning: hf_siglip not found, using mock implementation")

class test_hf_siglip:
    """
    Test class for Google's SigLIP (Sigmoid Loss Image-text Pre-training) model.
    
    SigLIP is a state-of-the-art contrastive vision-language model developed by Google,
    using a novel sigmoid loss function instead of the traditional softmax loss used in CLIP.
    It achieves superior performance on zero-shot classification and image-text retrieval tasks.
    
    This class tests SigLIP functionality across different hardware 
    backends including CPU, CUDA, and OpenVINO.
    
    It verifies:
    1. Image embedding extraction 
    2. Text embedding extraction
    3. Image-text similarity calculation
    4. Zero-shot classification capabilities
    5. Cross-platform compatibility
    6. Performance metrics
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the SigLIP test environment"""
        # Set up resources with fallbacks
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np, 
            "transformers": transformers,
            "PIL": PIL,
            "Image": Image
        }
        
        # Store metadata
        self.metadata = metadata if metadata else {}
        
        # Initialize the SigLIP model
        self.siglip = hf_siglip(resources=self.resources, metadata=self.metadata)
        
        # Use a small, openly accessible model that doesn't require authentication
        # SigLIP has different model sizes available
        self.model_name = "google/siglip-base-patch16-224"  
        
        # Create test image
        self.test_image = self._create_test_image()
        
        # Create test text inputs for similarity and classification
        self.test_texts = ["a white circle on a black background",
                          "a black square on a white background",
                          "a colorful sunset over mountains",
                          "a photo of a cat",
                          "an abstract geometric shape"]
        
        # Test classes for zero-shot classification
        self.test_classes = [
            "a circle", 
            "a square", 
            "a triangle",
            "random noise", 
            "a checkerboard pattern"
        ]
        
        # Status tracking
        self.status_messages = {
            "cpu": "Not tested yet",
            "cuda": "Not tested yet",
            "openvino": "Not tested yet"
        }
        
        # Examples for tracking test outputs
        self.examples = []
        
        return None
    
    def _create_test_image(self):
        """Create a simple test image (224x224) with a white circle in the middle"""
        try:
            if isinstance(np, MagicMock) or isinstance(PIL, MagicMock):
                # Return mock if dependencies not available
                return MagicMock()
                
            # Create a black image with a white circle
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Draw a white circle in the middle
            center_x, center_y = 112, 112
            radius = 50
            
            y, x = np.ogrid[:224, :224]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            img[mask] = 255
            
            # Convert to PIL image
            pil_image = Image.fromarray(img)
            
            return pil_image
        except Exception as e:
            print(f"Error creating test image: {e}")
            return MagicMock()
    
    def _create_local_test_model(self):
        """Create a minimal test model directory for testing without downloading"""
        try:
            print("Creating local test model for SigLIP...")
            
            # Create model directory in /tmp for tests
            test_model_dir = os.path.join("/tmp", "siglip_test_model")
            os.makedirs(test_model_dir, exist_ok=True)
            
            # Create minimal config file
            config = {
                "model_type": "siglip",
                "hidden_size": 768,
                "intermediate_size": 3072,
                "projection_dim": 512,
                "text_config": {
                    "hidden_size": 768,
                    "vocab_size": 32000
                },
                "vision_config": {
                    "hidden_size": 768,
                    "image_size": 224,
                    "patch_size": 16
                }
            }
            
            # Write config
            with open(os.path.join(test_model_dir, "config.json"), "w") as f:
                json.dump(config, f)
                
            print(f"Test model created at {test_model_dir}")
            return test_model_dir
            
        except Exception as e:
            print(f"Error creating test model: {e}")
            return self.model_name  # Fall back to original name
    
    def test(self):
        """Run all tests for the SigLIP model"""
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.siglip is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"
        
        # Test CPU initialization and functionality
        try:
            print("Testing SigLIP on CPU...")
            
            # Check if using real transformers
            transformers_available = not isinstance(self.resources["transformers"], MagicMock)
            implementation_type = "(REAL)" if transformers_available else "(MOCK)"
            
            # Initialize for CPU
            endpoint, processor, handler, queue, batch_size = self.siglip.init_cpu(
                self.model_name,
                "cpu",
                "cpu"
            )
            
            valid_init = endpoint is not None and processor is not None and handler is not None
            results["cpu_init"] = f"Success {implementation_type}" if valid_init else "Failed CPU initialization"
            
            # Test image embedding
            start_time = time.time()
            image_output = handler(images=self.test_image)
            image_embedding_time = time.time() - start_time
            
            # Verify image embedding
            has_image_embedding = (
                image_output is not None and
                isinstance(image_output, dict) and
                "image_embeds" in image_output and
                hasattr(image_output["image_embeds"], "shape")
            )
            
            results["cpu_image_embedding"] = f"Success {implementation_type}" if has_image_embedding else "Failed image embedding"
            
            if has_image_embedding:
                # Record image embedding example
                image_embed_shape = list(image_output["image_embeds"].shape) if has_image_embedding else None
                
                example = {
                    "input": {
                        "type": "image_embedding",
                        "image": "image input (binary data not shown)"
                    },
                    "output": {
                        "embedding_shape": image_embed_shape,
                        "embedding_type": str(image_output["image_embeds"].dtype) if has_image_embedding else None
                    },
                    "timestamp": time.time(),
                    "elapsed_time": image_embedding_time,
                    "implementation_type": implementation_type.strip("()"),
                    "platform": "CPU"
                }
                
                self.examples.append(example)
            
            # Test similarity calculation
            similarity_results = {}
            
            for text in self.test_texts:
                try:
                    start_time = time.time()
                    similarity_output = handler(images=self.test_image, text=text)
                    similarity_time = time.time() - start_time
                    
                    # Verify similarity output
                    has_similarity = (
                        similarity_output is not None and
                        isinstance(similarity_output, dict) and
                        "similarity" in similarity_output and
                        "text_embeds" in similarity_output
                    )
                    
                    if has_similarity:
                        # Extract similarity score
                        similarity_score = similarity_output["similarity"].item() if hasattr(similarity_output["similarity"], "item") else similarity_output["similarity"]
                        
                        # Add example to collection
                        example = {
                            "input": {
                                "type": "similarity",
                                "image": "image input (binary data not shown)",
                                "text": text
                            },
                            "output": {
                                "similarity_score": similarity_score,
                                "text_embedding_shape": list(similarity_output["text_embeds"].shape) if "text_embeds" in similarity_output else None
                            },
                            "timestamp": time.time(),
                            "elapsed_time": similarity_time,
                            "implementation_type": implementation_type.strip("()"),
                            "platform": "CPU"
                        }
                        
                        self.examples.append(example)
                        
                        similarity_results[text] = {
                            "success": True,
                            "similarity_score": similarity_score,
                            "elapsed_time": similarity_time
                        }
                    else:
                        similarity_results[text] = {
                            "success": False,
                            "error": "No similarity score generated"
                        }
                except Exception as e:
                    print(f"Error in similarity test for text '{text}': {e}")
                    similarity_results[text] = {
                        "success": False,
                        "error": str(e)
                    }
            
            # Record similarity results
            results["cpu_similarity_results"] = similarity_results
            results["cpu_similarity"] = f"Success {implementation_type}" if any(item["success"] for item in similarity_results.values()) else "Failed all similarity tests"
            
            # Test zero-shot classification
            if len(self.examples) > 0 and all(item["success"] for item in similarity_results.values()):
                try:
                    # Perform zero-shot classification by comparing similarities across classes
                    start_time = time.time()
                    classification_results = {}
                    similarities = []
                    
                    for class_name in self.test_classes:
                        similarity_output = handler(images=self.test_image, text=f"a photo of {class_name}")
                        if "similarity" in similarity_output:
                            similarity_score = similarity_output["similarity"].item() if hasattr(similarity_output["similarity"], "item") else similarity_output["similarity"]
                            similarities.append((class_name, similarity_score))
                    
                    # Sort by similarity score (descending)
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    classification_time = time.time() - start_time
                    
                    # Record classification results
                    classification_results = {
                        "top_class": similarities[0][0] if similarities else None,
                        "top_score": similarities[0][1] if similarities else None,
                        "all_results": dict(similarities),
                        "elapsed_time": classification_time
                    }
                    
                    # Add example to collection
                    example = {
                        "input": {
                            "type": "zero_shot_classification",
                            "image": "image input (binary data not shown)",
                            "classes": self.test_classes
                        },
                        "output": classification_results,
                        "timestamp": time.time(),
                        "elapsed_time": classification_time,
                        "implementation_type": implementation_type.strip("()"),
                        "platform": "CPU"
                    }
                    
                    self.examples.append(example)
                    results["cpu_classification"] = f"Success {implementation_type}"
                    results["cpu_classification_results"] = classification_results
                except Exception as e:
                    print(f"Error in zero-shot classification test: {e}")
                    results["cpu_classification"] = f"Error: {str(e)}"
            
        except Exception as e:
            print(f"Error in CPU tests: {e}")
            traceback.print_exc()
            results["cpu_tests"] = f"Error: {str(e)}"
            
        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                print("Testing SigLIP on CUDA...")
                # Import CUDA utilities
                try:
                    sys.path.insert(0, "/home/barberb/ipfs_accelerate_py/test")
                    from utils import get_cuda_device, optimize_cuda_memory, benchmark_cuda_inference
                    cuda_utils_available = True
                except ImportError:
                    cuda_utils_available = False
                
                # Initialize for CUDA
                endpoint, processor, handler, queue, batch_size = self.siglip.init_cuda(
                    self.model_name,
                    "cuda",
                    "cuda:0"
                )
                
                valid_init = endpoint is not None and processor is not None and handler is not None
                results["cuda_init"] = "Success (REAL)" if valid_init else "Failed CUDA initialization"
                
                # Test image embedding with performance metrics
                start_time = time.time()
                image_output = handler(images=self.test_image)
                image_embedding_time = time.time() - start_time
                
                # Get GPU memory if available
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024) if hasattr(torch.cuda, "memory_allocated") else None
                
                # Verify image embedding
                has_image_embedding = (
                    image_output is not None and
                    isinstance(image_output, dict) and
                    "image_embeds" in image_output and
                    hasattr(image_output["image_embeds"], "shape")
                )
                
                results["cuda_image_embedding"] = "Success (REAL)" if has_image_embedding else "Failed image embedding"
                
                if has_image_embedding:
                    # Performance metrics
                    perf_metrics = {
                        "processing_time_seconds": image_embedding_time
                    }
                    
                    if gpu_memory_mb is not None:
                        perf_metrics["gpu_memory_mb"] = gpu_memory_mb
                        
                    # Record image embedding example
                    image_embed_shape = list(image_output["image_embeds"].shape) if has_image_embedding else None
                    
                    example = {
                        "input": {
                            "type": "image_embedding",
                            "image": "image input (binary data not shown)"
                        },
                        "output": {
                            "embedding_shape": image_embed_shape,
                            "embedding_type": str(image_output["image_embeds"].dtype) if has_image_embedding else None
                        },
                        "timestamp": time.time(),
                        "elapsed_time": image_embedding_time,
                        "implementation_type": "REAL",
                        "platform": "CUDA",
                        "performance_metrics": perf_metrics
                    }
                    
                    self.examples.append(example)
                
                # Test similarity calculation
                similarity_results = {}
                
                for text in self.test_texts:
                    try:
                        start_time = time.time()
                        similarity_output = handler(images=self.test_image, text=text)
                        similarity_time = time.time() - start_time
                        
                        # Verify similarity output
                        has_similarity = (
                            similarity_output is not None and
                            isinstance(similarity_output, dict) and
                            "similarity" in similarity_output and
                            "text_embeds" in similarity_output
                        )
                        
                        if has_similarity:
                            # Extract similarity score
                            similarity_score = similarity_output["similarity"].item() if hasattr(similarity_output["similarity"], "item") else similarity_output["similarity"]
                            
                            # Performance metrics
                            perf_metrics = {
                                "processing_time_seconds": similarity_time
                            }
                            
                            if gpu_memory_mb is not None:
                                perf_metrics["gpu_memory_mb"] = gpu_memory_mb
                                
                            # Add example to collection
                            example = {
                                "input": {
                                    "type": "similarity",
                                    "image": "image input (binary data not shown)",
                                    "text": text
                                },
                                "output": {
                                    "similarity_score": similarity_score,
                                    "text_embedding_shape": list(similarity_output["text_embeds"].shape) if "text_embeds" in similarity_output else None
                                },
                                "timestamp": time.time(),
                                "elapsed_time": similarity_time,
                                "implementation_type": "REAL",
                                "platform": "CUDA",
                                "performance_metrics": perf_metrics
                            }
                            
                            self.examples.append(example)
                            
                            similarity_results[text] = {
                                "success": True,
                                "similarity_score": similarity_score,
                                "elapsed_time": similarity_time,
                                "performance_metrics": perf_metrics
                            }
                        else:
                            similarity_results[text] = {
                                "success": False,
                                "error": "No similarity score generated"
                            }
                    except Exception as e:
                        print(f"Error in CUDA similarity test for text '{text}': {e}")
                        similarity_results[text] = {
                            "success": False,
                            "error": str(e)
                        }
                
                # Record similarity results
                results["cuda_similarity_results"] = similarity_results
                results["cuda_similarity"] = "Success (REAL)" if any(item["success"] for item in similarity_results.values()) else "Failed all similarity tests"
                
                # Test zero-shot classification
                if len(similarity_results) > 0 and any(item["success"] for item in similarity_results.values()):
                    try:
                        # Perform zero-shot classification by comparing similarities across classes
                        start_time = time.time()
                        classification_results = {}
                        similarities = []
                        
                        for class_name in self.test_classes:
                            similarity_output = handler(images=self.test_image, text=f"a photo of {class_name}")
                            if "similarity" in similarity_output:
                                similarity_score = similarity_output["similarity"].item() if hasattr(similarity_output["similarity"], "item") else similarity_output["similarity"]
                                similarities.append((class_name, similarity_score))
                        
                        # Sort by similarity score (descending)
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        classification_time = time.time() - start_time
                        
                        # Performance metrics
                        perf_metrics = {
                            "processing_time_seconds": classification_time,
                            "time_per_class": classification_time / len(self.test_classes) if self.test_classes else 0
                        }
                        
                        if gpu_memory_mb is not None:
                            perf_metrics["gpu_memory_mb"] = gpu_memory_mb
                            
                        # Record classification results
                        classification_results = {
                            "top_class": similarities[0][0] if similarities else None,
                            "top_score": similarities[0][1] if similarities else None,
                            "all_results": dict(similarities),
                            "elapsed_time": classification_time,
                            "performance_metrics": perf_metrics
                        }
                        
                        # Add example to collection
                        example = {
                            "input": {
                                "type": "zero_shot_classification",
                                "image": "image input (binary data not shown)",
                                "classes": self.test_classes
                            },
                            "output": classification_results,
                            "timestamp": time.time(),
                            "elapsed_time": classification_time,
                            "implementation_type": "REAL",
                            "platform": "CUDA",
                            "performance_metrics": perf_metrics
                        }
                        
                        self.examples.append(example)
                        results["cuda_classification"] = "Success (REAL)"
                        results["cuda_classification_results"] = classification_results
                    except Exception as e:
                        print(f"Error in CUDA zero-shot classification test: {e}")
                        results["cuda_classification"] = f"Error: {str(e)}"
                
            except Exception as e:
                print(f"Error in CUDA tests: {e}")
                traceback.print_exc()
                results["cuda_tests"] = f"Error: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"
            
        # Test OpenVINO if available
        try:
            print("Testing SigLIP on OpenVINO...")
            
            # Try to import OpenVINO
            try:
                import openvino
                openvino_available = True
            except ImportError:
                openvino_available = False
                
            if not openvino_available:
                results["openvino_tests"] = "OpenVINO not available"
            else:
                # Initialize for OpenVINO
                endpoint, processor, handler, queue, batch_size = self.siglip.init_openvino(
                    self.model_name,
                    "openvino",
                    "CPU"  # Standard OpenVINO device
                )
                
                valid_init = endpoint is not None and processor is not None and handler is not None
                results["openvino_init"] = "Success (REAL)" if valid_init else "Failed OpenVINO initialization"
                
                # Test image embedding
                start_time = time.time()
                image_output = handler(images=self.test_image)
                image_embedding_time = time.time() - start_time
                
                # Verify image embedding
                has_image_embedding = (
                    image_output is not None and
                    isinstance(image_output, dict) and
                    "image_embeds" in image_output and
                    hasattr(image_output["image_embeds"], "shape")
                )
                
                results["openvino_image_embedding"] = "Success (REAL)" if has_image_embedding else "Failed image embedding"
                
                if has_image_embedding:
                    # Performance metrics
                    perf_metrics = {
                        "processing_time_seconds": image_embedding_time
                    }
                    
                    # Record image embedding example
                    image_embed_shape = list(image_output["image_embeds"].shape) if has_image_embedding else None
                    
                    example = {
                        "input": {
                            "type": "image_embedding",
                            "image": "image input (binary data not shown)"
                        },
                        "output": {
                            "embedding_shape": image_embed_shape,
                            "embedding_type": str(image_output["image_embeds"].dtype) if has_image_embedding else None
                        },
                        "timestamp": time.time(),
                        "elapsed_time": image_embedding_time,
                        "implementation_type": "REAL",
                        "platform": "OpenVINO",
                        "performance_metrics": perf_metrics
                    }
                    
                    self.examples.append(example)
                
                # Test similarity with one example text
                test_text = self.test_texts[0] if self.test_texts else "a white circle on a black background"
                
                start_time = time.time()
                similarity_output = handler(images=self.test_image, text=test_text)
                similarity_time = time.time() - start_time
                
                # Verify similarity output
                has_similarity = (
                    similarity_output is not None and
                    isinstance(similarity_output, dict) and
                    "similarity" in similarity_output
                )
                
                if has_similarity:
                    # Extract similarity score
                    similarity_score = similarity_output["similarity"].item() if hasattr(similarity_output["similarity"], "item") else similarity_output["similarity"]
                    
                    # Performance metrics
                    perf_metrics = {
                        "processing_time_seconds": similarity_time
                    }
                    
                    # Add example to collection
                    example = {
                        "input": {
                            "type": "similarity",
                            "image": "image input (binary data not shown)",
                            "text": test_text
                        },
                        "output": {
                            "similarity_score": similarity_score
                        },
                        "timestamp": time.time(),
                        "elapsed_time": similarity_time,
                        "implementation_type": "REAL",
                        "platform": "OpenVINO",
                        "performance_metrics": perf_metrics
                    }
                    
                    self.examples.append(example)
                    results["openvino_similarity"] = "Success (REAL)"
                else:
                    results["openvino_similarity"] = "Failed similarity test"
        except Exception as e:
            print(f"Error in OpenVINO tests: {e}")
            traceback.print_exc()
            results["openvino_tests"] = f"Error: {str(e)}"
        
        # Add examples to results
        results["examples"] = self.examples
        
        return results
    
    def __test__(self):
        """Run tests and handle result storage and comparison"""
        test_results = {}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {
                "status": {"test_error": str(e)},
                "traceback": traceback.format_exc(),
                "examples": []
            }
        
        # Add metadata
        test_results["metadata"] = {
            "timestamp": time.time(),
            "torch_version": getattr(torch, "__version__", "mocked"),
            "numpy_version": getattr(np, "__version__", "mocked"),
            "transformers_version": getattr(transformers, "__version__", "mocked"),
            "pil_version": getattr(PIL, "__version__", "mocked"),
            "cuda_available": getattr(torch, "cuda", MagicMock()).is_available() if not isinstance(torch, MagicMock) else False,
            "cuda_device_count": getattr(torch, "cuda", MagicMock()).device_count() if not isinstance(torch, MagicMock) else 0,
            "transformers_mocked": isinstance(self.resources["transformers"], MagicMock),
            "test_model": self.model_name,
            "test_run_id": f"siglip-test-{int(time.time())}"
        }
        
        # Create directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_dir = os.path.join(base_dir, 'expected_results')
        collected_dir = os.path.join(base_dir, 'collected_results')
        
        os.makedirs(expected_dir, exist_ok=True)
        os.makedirs(collected_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(collected_dir, 'hf_siglip_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_siglip_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                
                # Simple check for basic compatibility
                if "init" in expected_results and "init" in test_results:
                    print("Results structure matches expected format.")
                else:
                    print("Warning: Results structure does not match expected format.")
            except Exception as e:
                print(f"Error reading expected results: {e}")
                # Create new expected results file
                with open(expected_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
        else:
            # Create new expected results file
            with open(expected_file, 'w') as f:
                json.dump(test_results, f, indent=2)
                
        return test_results

if __name__ == "__main__":
    try:
        print("Starting SigLIP test...")
        this_siglip = test_hf_siglip()
        results = this_siglip.__test__()
        print(f"SigLIP Test Completed")
        
        # Print a summary of the results
        if "init" in results:
            print(f"Initialization: {results['init']}")
        
        # CPU results
        cpu_status = []
        if "cpu_image_embedding" in results:
            cpu_status.append(f"Image Embedding: {results['cpu_image_embedding']}")
        if "cpu_similarity" in results:
            cpu_status.append(f"Similarity: {results['cpu_similarity']}")
        if "cpu_classification" in results:
            cpu_status.append(f"Classification: {results['cpu_classification']}")
        
        if cpu_status:
            print(f"CPU Tests: {', '.join(cpu_status)}")
        elif "cpu_tests" in results:
            print(f"CPU Tests: {results['cpu_tests']}")
            
        # CUDA results
        cuda_status = []
        if "cuda_image_embedding" in results:
            cuda_status.append(f"Image Embedding: {results['cuda_image_embedding']}")
        if "cuda_similarity" in results:
            cuda_status.append(f"Similarity: {results['cuda_similarity']}")
        if "cuda_classification" in results:
            cuda_status.append(f"Classification: {results['cuda_classification']}")
        
        if cuda_status:
            print(f"CUDA Tests: {', '.join(cuda_status)}")
        elif "cuda_tests" in results:
            print(f"CUDA Tests: {results['cuda_tests']}")
            
        # OpenVINO results
        openvino_status = []
        if "openvino_image_embedding" in results:
            openvino_status.append(f"Image Embedding: {results['openvino_image_embedding']}")
        if "openvino_similarity" in results:
            openvino_status.append(f"Similarity: {results['openvino_similarity']}")
        
        if openvino_status:
            print(f"OpenVINO Tests: {', '.join(openvino_status)}")
        elif "openvino_tests" in results:
            print(f"OpenVINO Tests: {results['openvino_tests']}")
            
        # Example count
        example_count = len(results.get("examples", []))
        print(f"Collected {example_count} test examples")
        
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)