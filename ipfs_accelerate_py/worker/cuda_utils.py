"""
CUDA Utilities for IPFS Accelerate Python Framework

This module provides common utilities for CUDA-based model inference
with optimized memory management and performance tracking.
"""

import os
import time
import traceback
from unittest.mock import MagicMock
import fcntl

# Try to import storage wrapper with comprehensive fallback
try:
    from ..common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from test.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        HAVE_STORAGE_WRAPPER = False
        def get_storage_wrapper(*args, **kwargs):
            return None

class cuda_utils:
    def __init__(self, resources=None, metadata=None):
        """
        Initialize CUDA utilities with resources
        
        Args:
            resources: Dictionary of resources (torch, etc.)
            metadata: Optional metadata
        """
        self.resources = resources
        self.metadata = metadata
        
        # Initialize storage wrapper
        self._storage = get_storage_wrapper() if HAVE_STORAGE_WRAPPER else None
        
        # Initialize core resources
        self.init()
        
        # Create lock directory for thread-safe operations
        os.makedirs(os.path.expanduser("~/.cache/ipfs_accelerate/locks"), exist_ok=True)
        
    def init(self):
        """Initialize required dependencies"""
        if "torch" not in list(self.resources.keys()):
            try:
                import torch
                self.torch = torch
            except ImportError:
                print("PyTorch not available, using mock")
                self.torch = MagicMock()
        else:
            self.torch = self.resources["torch"]

        if "transformers" not in list(self.resources.keys()):
            try:
                import transformers
                self.transformers = transformers
            except ImportError:
                print("Transformers not available, using mock")
                self.transformers = MagicMock()
        else:
            self.transformers = self.resources["transformers"]
            
        if "numpy" not in list(self.resources.keys()):
            try:
                import numpy as np
                self.np = np
            except ImportError:
                print("NumPy not available, using mock")
                self.np = MagicMock()
        else:
            self.np = self.resources["numpy"]

    def get_cuda_device(self, device_label="cuda:0"):
        """
        Get a valid CUDA device from label with proper error handling
        
        Args:
            device_label: String like "cuda:0" or "cuda:1"
            
        Returns:
            torch.device: CUDA device object, or None if not available
        """
        try:
            # Check if CUDA is available
            if not self.torch.cuda.is_available():
                print("CUDA is not available on this system")
                return None
                
            # Parse device parts
            parts = device_label.split(":")
            device_type = parts[0].lower()
            device_index = int(parts[1]) if len(parts) > 1 else 0
            
            # Validate device type
            if device_type != "cuda":
                print(f"Warning: Device type '{device_type}' is not CUDA, defaulting to 'cuda'")
                device_type = "cuda"
                
            # Validate device index
            cuda_device_count = self.torch.cuda.device_count()
            if device_index >= cuda_device_count:
                print(f"Warning: CUDA device index {device_index} out of range (0-{cuda_device_count-1}), using device 0")
                device_index = 0
                
            # Create device object
            device = self.torch.device(f"{device_type}:{device_index}")
            
            # Print device info
            device_name = self.torch.cuda.get_device_name(device_index)
            print(f"Using CUDA device: {device_name} (index {device_index})")
            
            # Get memory info
            if hasattr(self.torch.cuda, 'mem_get_info'):
                free_memory, total_memory = self.torch.cuda.mem_get_info(device_index)
                free_memory_gb = free_memory / (1024**3)
                total_memory_gb = total_memory / (1024**3)
                print(f"CUDA memory: {free_memory_gb:.2f}GB free / {total_memory_gb:.2f}GB total")
            
            return device
        except Exception as e:
            print(f"Error setting up CUDA device: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def optimize_cuda_memory(self, model, device, use_half_precision=True, max_memory=None):
        """
        Optimize CUDA memory usage for model inference
        
        Args:
            model: PyTorch model to optimize
            device: CUDA device to use
            use_half_precision: Whether to use FP16 precision
            max_memory: Maximum memory to use (in GB), None for auto
            
        Returns:
            model: Optimized model
        """
        try:
            # Check if we have a mock model
            if isinstance(model, MagicMock):
                print("Using mock model, skipping memory optimization")
                return model
                
            # Get available memory on device
            if max_memory is None and hasattr(self.torch.cuda, 'mem_get_info'):
                free_memory, total_memory = self.torch.cuda.mem_get_info(device.index)
                max_memory = free_memory * 0.9 / (1024**3)  # 90% of free memory in GB
                print(f"Auto-detected memory limit: {max_memory:.2f}GB")
            elif max_memory is None:
                # Default for when mem_get_info is not available
                max_memory = 4.0 
                print(f"Using default memory limit: {max_memory:.2f}GB")
            
            print(f"Optimizing model for CUDA memory usage (limit: {max_memory:.2f}GB)")
            
            # Convert to half precision if requested
            if use_half_precision:
                # Check if model supports half precision
                if hasattr(model, "half") and callable(model.half):
                    model = model.half()
                    print("Using half precision (FP16) for faster inference")
                else:
                    print("Model doesn't support half precision, using full precision")
            
            # Move model to CUDA
            if hasattr(model, "to") and callable(model.to):
                model = model.to(device)
                print(f"Model moved to {device}")
            else:
                print("Model doesn't support moving to device")
            
            # Set to evaluation mode
            if hasattr(model, "eval") and callable(model.eval):
                model.eval()
                print("Model set to evaluation mode")
            
            # Additional memory optimizations
            if hasattr(self.torch.cuda, 'empty_cache'):
                self.torch.cuda.empty_cache()
                
            return model
        except Exception as e:
            print(f"Error optimizing CUDA memory: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Return original model as fallback
            if hasattr(model, "to") and callable(model.to):
                return model.to(device)
            return model
    
    def cuda_batch_processor(self, model, inputs, batch_size=8, device=None, max_length=None):
        """
        Process inputs in batches for more efficient CUDA utilization
        
        Args:
            model: PyTorch model
            inputs: Input tensor or list of tensors
            batch_size: Size of batches to process
            device: CUDA device to use
            max_length: Maximum sequence length (for padded sequences)
            
        Returns:
            outputs: Processed outputs
        """
        try:
            # Check if we have a mock model
            if isinstance(model, MagicMock):
                print("Using mock model, returning mock output")
                return [self.torch.zeros((1, 10))]
            
            # Ensure inputs are in a list
            if not isinstance(inputs, list):
                inputs = [inputs]
                
            # Prepare batches
            batches = [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]
            outputs = []
            
            # Process each batch
            for i, batch in enumerate(batches):
                print(f"Processing batch {i+1}/{len(batches)} (size: {len(batch)})")
                
                # Move batch to CUDA
                if device is not None:
                    batch = [b.to(device) if hasattr(b, 'to') else b for b in batch]
                    
                # Process batch
                with self.torch.no_grad():  # Disable gradients for inference
                    # Choose processing method based on model and parameters
                    if hasattr(model, 'generate') and callable(model.generate):
                        # Language model generation
                        if max_length:
                            batch_output = model.generate(batch, max_length=max_length)
                        else:
                            batch_output = model.generate(batch)
                    else:
                        # Standard forward pass
                        batch_output = model(batch)
                    
                # Move results back to CPU to save GPU memory
                if isinstance(batch_output, self.torch.Tensor) and device is not None:
                    batch_output = batch_output.cpu()
                
                outputs.append(batch_output)
            
            # Combine outputs if possible
            if all(isinstance(o, self.torch.Tensor) for o in outputs):
                try:
                    return self.torch.cat(outputs, dim=0)
                except Exception as concat_error:
                    print(f"Error concatenating outputs: {concat_error}")
                    return outputs
            else:
                return outputs
        except Exception as e:
            print(f"Error in CUDA batch processing: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None

    class FileLock:
        """
        Simple file-based lock with timeout for thread-safe CUDA operations
        
        Usage:
            with FileLock("path/to/lock_file", timeout=60):
                # critical section
        """
        def __init__(self, lock_file, timeout=60):
            self.lock_file = lock_file
            self.timeout = timeout
            self.fd = None
        
        def __enter__(self):
            start_time = time.time()
            while True:
                try:
                    # Try to create and lock the file
                    self.fd = open(self.lock_file, 'w')
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except IOError:
                    # Check timeout
                    if time.time() - start_time > self.timeout:
                        raise TimeoutError(f"Could not acquire lock on {self.lock_file} within {self.timeout} seconds")
                    
                    # Wait and retry
                    time.sleep(1)
            return self
        
        def __exit__(self, *args):
            if self.fd:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                self.fd.close()
                try:
                    os.unlink(self.lock_file)
                except:
                    pass
    
    def benchmark_cuda_inference(self, model, inputs, iterations=5, device=None):
        """
        Benchmark model inference performance on CUDA
        
        Args:
            model: PyTorch model
            inputs: Input tensor or dictionary
            iterations: Number of iterations to run
            device: CUDA device to use
            
        Returns:
            dict: Performance metrics
        """
        # Check if we have a mock model
        if isinstance(model, MagicMock):
            print("Using mock model, returning mock benchmark results")
            return {
                "average_inference_time": 0.001,
                "iterations": iterations,
                "cuda_device": "Mock CUDA Device",
                "cuda_memory_used": 0.0
            }
        
        if device is None and self.torch.cuda.is_available():
            device = self.torch.device("cuda:0")
        
        # Ensure model is on CUDA
        if hasattr(model, 'to') and callable(model.to):
            model = model.to(device)
            
        if hasattr(model, 'eval') and callable(model.eval):
            model.eval()
        
        # Move inputs to CUDA if needed
        if isinstance(inputs, dict):
            cuda_inputs = {}
            for k, v in inputs.items():
                if hasattr(v, 'to') and callable(v.to):
                    cuda_inputs[k] = v.to(device)
                else:
                    cuda_inputs[k] = v
        elif hasattr(inputs, 'to') and callable(inputs.to):
            cuda_inputs = inputs.to(device)
        else:
            cuda_inputs = inputs
        
        # Warmup
        try:
            with self.torch.no_grad():
                if hasattr(model, 'generate') and callable(model.generate):
                    _ = model.generate(**cuda_inputs if isinstance(cuda_inputs, dict) else cuda_inputs)
                else:
                    _ = model(cuda_inputs)
        except Exception as warmup_error:
            print(f"Error during warmup: {warmup_error}")
            # Continue anyway
        
        # Get starting memory usage
        if hasattr(self.torch.cuda, 'memory_allocated'):
            memory_start = self.torch.cuda.memory_allocated(device) / (1024**2)  # MB
        else:
            memory_start = 0.0
        
        # Benchmark
        inference_times = []
        try:
            self.torch.cuda.synchronize(device)
            for i in range(iterations):
                start_time = time.time()
                with self.torch.no_grad():
                    if hasattr(model, 'generate') and callable(model.generate):
                        _ = model.generate(**cuda_inputs if isinstance(cuda_inputs, dict) else cuda_inputs)
                    else:
                        _ = model(cuda_inputs)
                self.torch.cuda.synchronize(device)
                end_time = time.time()
                inference_times.append(end_time - start_time)
                print(f"Iteration {i+1}/{iterations}: {inference_times[-1]:.4f}s")
        except Exception as bench_error:
            print(f"Error during benchmarking: {bench_error}")
            print(f"Traceback: {traceback.format_exc()}")
        
        # Get peak memory usage
        if hasattr(self.torch.cuda, 'max_memory_allocated'):
            memory_peak = self.torch.cuda.max_memory_allocated(device) / (1024**2)  # MB
            memory_used = memory_peak - memory_start
        else:
            memory_used = 0.0
        
        # Reset memory stats
        if hasattr(self.torch.cuda, 'reset_peak_memory_stats'):
            self.torch.cuda.reset_peak_memory_stats(device)
        
        # Clean up
        if hasattr(self.torch.cuda, 'empty_cache'):
            self.torch.cuda.empty_cache()
        
        # Calculate average and other statistics
        if inference_times:
            avg_time = sum(inference_times) / len(inference_times)
            min_time = min(inference_times)
            max_time = max(inference_times)
        else:
            avg_time = min_time = max_time = 0.0
        
        return {
            "average_inference_time": avg_time,
            "min_inference_time": min_time,
            "max_inference_time": max_time,
            "iterations": len(inference_times),
            "cuda_device": self.torch.cuda.get_device_name(device) if self.torch.cuda.is_available() else "N/A",
            "cuda_memory_used_mb": memory_used
        }
    
    def create_cuda_mock_implementation(self, model_type, shape_info=None):
        """
        Create a mock CUDA implementation for testing
        
        Args:
            model_type: Type of model to mock ('lm', 'embed', 'whisper', etc.)
            shape_info: Optional shape information for outputs
            
        Returns:
            tuple: Mock objects required for CUDA implementation
        """
        # Create mock device
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_device.index = 0
        
        # Create mock CUDA functions
        cuda_functions = {
            "is_available": MagicMock(return_value=True),
            "get_device_name": MagicMock(return_value="Mock CUDA Device"),
            "device_count": MagicMock(return_value=1),
            "current_device": MagicMock(return_value=0),
            "empty_cache": MagicMock(),
        }
        
        # Create appropriate mock objects based on model type
        if model_type == "lm":
            # Language model mocks
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.half.return_value = mock_model
            mock_model.eval.return_value = mock_model
            mock_model.generate.return_value = self.torch.tensor([[1, 2, 3, 4, 5]])
            
            # Handler that simulates CUDA acceleration
            def handler(prompt, max_new_tokens=100, temperature=0.7):
                return {
                    "text": f"(MOCK CUDA) Generated text for: {prompt[:20]}...",
                    "implementation_type": "MOCK",
                    "device": "cuda:0"
                }
                
            return None, MagicMock(), handler, None, 8
            
        elif model_type == "embed":
            # Embedding model mocks
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.half.return_value = mock_model
            mock_model.eval.return_value = mock_model
            
            # Handler that returns mock embeddings
            def handler(text):
                embed_dim = shape_info or 768
                # Create tensor with proper device info
                embedding = self.torch.zeros(embed_dim)
                embedding.requires_grad = False
                embedding._mock_device = "cuda:0"  # Simulate CUDA tensor
                return {
                    "embedding": embedding,
                    "implementation_type": "MOCK",
                    "device": "cuda:0"
                }
                
            return None, MagicMock(), handler, None, 16
            
        elif model_type in ["clip", "xclip"]:
            # Multimodal model mocks
            def handler(text=None, image=None):
                text_embed = self.torch.zeros(512)
                image_embed = self.torch.zeros(512)
                
                # Add mock device info
                text_embed._mock_device = "cuda:0"
                image_embed._mock_device = "cuda:0"
                
                return {
                    "text_embedding": text_embed,
                    "image_embedding": image_embed,
                    "similarity": self.torch.tensor([0.75]),
                    "implementation_type": "MOCK",
                    "device": "cuda:0"
                }
                
            return None, MagicMock(), handler, None, 8
        
        # Default catch-all mock
        return None, MagicMock(), lambda x: {"output": "(MOCK CUDA) Output", "implementation_type": "MOCK", "device": "cuda:0"}, None, 8

    def get_implementation_type(self, endpoint=None, tokenizer=None, prefix="", real_impl=None):
        """
        Determine the implementation type (REAL vs MOCK) based on component inspection
        
        Args:
            endpoint: Model endpoint to check
            tokenizer: Tokenizer to check
            prefix: Optional prefix for log messages
            real_impl: Override implementation type if provided
            
        Returns:
            bool: True if using real implementation, False if using mock
        """
        if real_impl is not None:
            # Use the provided implementation type if available
            print(f"{prefix}Using provided implementation type: {'REAL' if real_impl else 'MOCK'}")
            return real_impl
            
        # Check if we're using mock components
        is_mock = (
            isinstance(endpoint, MagicMock) or 
            isinstance(tokenizer, MagicMock) or
            not self.torch.cuda.is_available()
        )
        
        impl_type = "MOCK" if is_mock else "REAL"
        print(f"{prefix}Detected implementation type: {impl_type}")
        
        return not is_mock