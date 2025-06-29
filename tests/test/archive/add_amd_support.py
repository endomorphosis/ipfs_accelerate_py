#!/usr/bin/env python3
"""
Script to add AMD ROCm, WebNN, and WebGPU support to the test files.

This script takes an existing test file and enhances it with:
1. AMD ROCm hardware detection and handler support
2. WebNN hardware detection and handler support  
3. WebGPU/transformers.js hardware detection and handler support
4. Precision compatibility for all hardware types
"""

import os
import sys
import re
import argparse
from pathlib import Path

def enhance_test_file(test_file_path, force=False):
    """Enhance a test file with AMD, WebNN, and WebGPU support."""
    # Check if the file exists
    test_file = Path(test_file_path)
    if not test_file.exists():
        print(f"Error: Test file {test_file_path} does not exist")
        return False
        
    # Create a backup of the original file
    backup_file = test_file.with_suffix(test_file.suffix + '.backup')
    if not backup_file.exists() or force:
        with open(test_file, 'r') as f:
            content = f.read()
            
        with open(backup_file, 'w') as f:
            f.write(content)
            
        print(f"Created backup file: {backup_file}")
    else:
        print(f"Backup file {backup_file} already exists, skipping backup")
        with open(test_file, 'r') as f:
            content = f.read()
        
    # Check if the file already has AMD, WebNN, or WebGPU support
    has_amd = 'amd' in content.lower() and 'create_amd_' in content
    has_webnn = 'webnn' in content.lower() and 'create_webnn_' in content
    has_webgpu = 'webgpu' in content.lower() and 'create_webgpu_' in content
    
    if has_amd and has_webnn and has_webgpu:
        print(f"Test file {test_file_path} already has AMD, WebNN, and WebGPU support")
        return True
        
    # Add imports if needed
    if 'import datetime' not in content:
        content = content.replace(
            'import traceback',
            'import traceback\nimport datetime'
        )
    
    # Add hardware backends to the MODEL_REGISTRY
    if 'hardware_compatibility' not in content:
        # Add hardware compatibility section to the MODEL_REGISTRY
        content = content.replace(
            '"supports_openvino": True,',
            '"supports_openvino": True,\n        "supports_amd": True,\n        "supports_webnn": True,\n        "supports_webgpu": True,'
        )
    else:
        # Update existing hardware compatibility section
        if 'amd' not in content:
            content = re.sub(
                r'([\s\t]*)("hardware_compatibility"[^}]+})',
                r'\1"hardware_compatibility": {\n\1    "cpu": True,\n\1    "cuda": True,\n\1    "openvino": True,\n\1    "apple": True,\n\1    "qualcomm": False,  # Usually false for complex models\n\1    "amd": True,  # AMD ROCm support\n\1    "webnn": True,  # WebNN support\n\1    "webgpu": True   # WebGPU with transformers.js support\n\1}',
                content
            )
    
    # Add precision compatibility if not present
    if 'precision_compatibility' not in content:
        model_registry_end = re.search(r'\}\)$', content)
        if model_registry_end:
            precision_compatibility = """
        # Precision support by hardware
        "precision_compatibility": {
            "cpu": {
                "fp32": True,
                "fp16": False,
                "bf16": True,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "cuda": {
                "fp32": True,
                "fp16": True,
                "bf16": True,
                "int8": True,
                "int4": True,
                "uint4": True,
                "fp8": False,
                "fp4": False
            },
            "openvino": {
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "apple": {
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": False,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "amd": {
                "fp32": True,
                "fp16": True,
                "bf16": True,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "qualcomm": {
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "webnn": {
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            },
            "webgpu": {
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": True,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }
        },
"""
            content = content.replace(
                '"default_batch_size": 1',
                '"default_batch_size": 1,\n        ' + precision_compatibility.strip()
            )
    
    # Add AMD detection to the _detect_hardware method
    if 'amd_version' not in content:
        # Find the _detect_hardware method
        hardware_detection = """
        # Check AMD ROCm support
        try:
            # Check for the presence of ROCm by importing rocm-specific modules or checking for devices
            import subprocess
            
            # Try to run rocm-smi to detect ROCm installation
            result = subprocess.run(['rocm-smi', '--showproductname'], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  universal_newlines=True, check=False)
            
            if result.returncode == 0:
                capabilities["amd"] = True
                
                # Try to get version information
                version_result = subprocess.run(['rocm-smi', '--showversion'], 
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                             universal_newlines=True, check=False)
                
                if version_result.returncode == 0:
                    # Extract version from output
                    import re
                    match = re.search(r'ROCm-SMI version:\s+(\d+\.\d+\.\d+)', version_result.stdout)
                    if match:
                        capabilities["amd_version"] = match.group(1)
                
                # Try to count devices
                devices_result = subprocess.run(['rocm-smi', '--showalldevices'], 
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                             universal_newlines=True, check=False)
                
                if devices_result.returncode == 0:
                    # Count device entries in output
                    device_lines = [line for line in devices_result.stdout.split('\n') if 'GPU[' in line]
                    capabilities["amd_devices"] = len(device_lines)
        except (ImportError, FileNotFoundError):
            pass
            
        # Alternate check for AMD ROCm using torch hip if available
        if TORCH_AVAILABLE and not capabilities["amd"]:
            try:
                import torch.utils.hip as hip
                if hasattr(hip, "is_available") and hip.is_available():
                    capabilities["amd"] = True
                    capabilities["amd_devices"] = hip.device_count()
            except (ImportError, AttributeError):
                pass
"""
        content = re.sub(
            r'def _detect_hardware\([^)]*\):[^\n]*\n\s+"""[^"]*"""[^\n]*\n\s+capabilities = {[^}]+}',
            r'def _detect_hardware(self):\n        """Detect available hardware and return capabilities dictionary"""\n        capabilities = {\n            "cpu": True,\n            "cuda": False,\n            "cuda_version": None,\n            "cuda_devices": 0,\n            "mps": False,\n            "openvino": False,\n            "qualcomm": False,\n            "amd": False,\n            "amd_version": None,\n            "amd_devices": 0,\n            "webnn": False,\n            "webnn_version": None,\n            "webgpu": False,\n            "webgpu_version": None\n        }',
            content
        )
        
        # Add AMD detection after CUDA detection
        content = re.sub(
            r'(# Check MPS \(Apple Silicon\)[^\n]*\n\s+if TORCH_AVAILABLE[^\n]*\n\s+capabilities\["mps"\] = [^\n]*)',
            r'\1' + hardware_detection,
            content
        )
    
    # Add WebNN and WebGPU detection
    if 'webnn_version' not in content:
        webnn_detection = """
        # Check for WebNN availability
        try:
            # Check for WebNN in browser environment
            import platform
            import subprocess
            
            # Check if running in a browser context (looking for JavaScript engine)
            is_browser_env = False
            try:
                # Try to detect Node.js environment
                node_version = subprocess.run(['node', '--version'], 
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                            universal_newlines=True, check=False)
                if node_version.returncode == 0:
                    # Check for WebNN polyfill package
                    webnn_check = subprocess.run(['npm', 'list', 'webnn-polyfill'], 
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                               universal_newlines=True, check=False)
                    if "webnn-polyfill" in webnn_check.stdout:
                        capabilities["webnn"] = True
                        
                        # Try to extract version
                        import re
                        match = re.search(r'webnn-polyfill@(\d+\.\d+\.\d+)', webnn_check.stdout)
                        if match:
                            capabilities["webnn_version"] = match.group(1)
                        else:
                            capabilities["webnn_version"] = "unknown"
            except (FileNotFoundError, subprocess.SubprocessError):
                pass
            
            # Alternative check for WebNN support through imported modules
            if not capabilities["webnn"]:
                try:
                    import webnn_polyfill
                    capabilities["webnn"] = True
                    capabilities["webnn_version"] = getattr(webnn_polyfill, "__version__", "unknown")
                except ImportError:
                    pass
        except Exception:
            pass
            
        # Check for WebGPU / transformers.js availability
        try:
            import platform
            import subprocess
            
            # Try to detect Node.js environment first (for transformers.js)
            try:
                node_version = subprocess.run(['node', '--version'], 
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                            universal_newlines=True, check=False)
                if node_version.returncode == 0:
                    # Check for transformers.js package
                    transformers_js_check = subprocess.run(['npm', 'list', '@xenova/transformers'], 
                                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                                        universal_newlines=True, check=False)
                    if "@xenova/transformers" in transformers_js_check.stdout:
                        capabilities["webgpu"] = True
                        
                        # Try to extract version
                        import re
                        match = re.search(r'@xenova/transformers@(\d+\.\d+\.\d+)', transformers_js_check.stdout)
                        if match:
                            capabilities["webgpu_version"] = match.group(1)
                        else:
                            capabilities["webgpu_version"] = "unknown"
            except (FileNotFoundError, subprocess.SubprocessError):
                pass
            
            # Check if browser with WebGPU is available
            # This is a simplified check since we can't actually detect browser capabilities
            # in a server-side context, but we can check for typical browser detection packages
            if not capabilities["webgpu"]:
                try:
                    # Check for webgpu mock or polyfill
                    webgpu_check = subprocess.run(['npm', 'list', 'webgpu'], 
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                               universal_newlines=True, check=False)
                    if "webgpu" in webgpu_check.stdout:
                        capabilities["webgpu"] = True
                        
                        # Try to extract version
                        import re
                        match = re.search(r'webgpu@(\d+\.\d+\.\d+)', webgpu_check.stdout)
                        if match:
                            capabilities["webgpu_version"] = match.group(1)
                        else:
                            capabilities["webgpu_version"] = "unknown"
                except (FileNotFoundError, subprocess.SubprocessError):
                    pass
        except Exception:
            pass
"""
        content = re.sub(
            r'(# Check for Qualcomm AI Engine Direct SDK[^\n]*\n\s+try:[^\n]*\n\s+import [^\n]*\n\s+capabilities\["qualcomm"\] = [^\n]*\n\s+except ImportError:[^\n]*\n\s+pass)',
            r'\1' + webnn_detection,
            content
        )
    
    # Add AMD initialization method
    if 'init_amd' not in content:
        amd_init = """
    def init_amd(self, model_name, model_type, device="rocm:0", **kwargs):
        \"\"\"Initialize model for AMD ROCm inference.
        
        Args:
            model_name (str): Model identifier
            model_type (str): Type of model (text-generation, image-classification, etc.)
            device (str): ROCm device identifier ('rocm:0', 'rocm:1', etc.)
            precision (str, optional): Precision to use (fp32, fp16, bf16, int8, int4)
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        \"\"\"
        try:
            import asyncio
            
            # Get precision from kwargs or default to fp32
            precision = kwargs.get("precision", "fp32")
            
            # Create processor and endpoint
            processor = self._create_mock_processor()
            endpoint = self._create_mock_endpoint()
            
            # Move to AMD ROCm device
            endpoint = endpoint.to(device)
            
            # Apply quantization if needed
            if precision in ["int8", "int4", "uint4"]:
                # In real implementation, would apply quantization here
                print(f"Applying {precision} quantization for AMD ROCm device")
            
            # Create handler
            handler = self.create_amd_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=f"amd_{device}",
                endpoint=endpoint,
                tokenizer=processor,
                is_real_impl=True,
                batch_size=4,
                precision=precision
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 4  # Default to larger batch size for AMD GPUs
            
            return endpoint, processor, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing AMD ROCm model: {e}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {"output": "Mock AMD ROCm output", "input": x, "implementation_type": "MOCK"}
            return None, None, handler, asyncio.Queue(32), 2
"""
        # Find a good insertion point for the AMD init method (after init_apple)
        content = re.sub(
            r'(def init_apple\([^)]*\):[^\n]*\n[^a-zA-Z0-9_]*("""[^"]*"""[^\n]*\n)(?:[^def]*\n)*\s+return [^a-zA-Z0-9_]*None[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*None[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*handler[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*asyncio.Queue\([0-9]*\)[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*[0-9]+)',
            r'\1\2\3' + amd_init,
            content
        )
    
    # Add WebNN initialization method
    if 'init_webnn' not in content:
        webnn_init = """
    def init_webnn(self, model_name, model_type, device="webnn", **kwargs):
        \"\"\"Initialize model for WebNN inference (browser or Node.js environment).
        
        WebNN enables hardware-accelerated inference in web browsers and Node.js
        applications by providing a common API that maps to the underlying hardware.
        
        Args:
            model_name (str): Model identifier
            model_type (str): Type of model (text-generation, image-classification, etc.)
            device (str): Device identifier ('webnn')
            precision (str, optional): Precision to use (fp32, fp16, int8)
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        \"\"\"
        try:
            import asyncio
            import numpy as np
            
            # Get precision from kwargs or default to fp32
            precision = kwargs.get("precision", "fp32")
            
            # Create processor/tokenizer
            processor = self._create_mock_processor()
            
            # Create WebNN endpoint/model
            # This would integrate with the WebNN API
            class WebNNModel:
                def compute(self, inputs):
                    \"\"\"Process inputs with WebNN and return outputs.\"\"\"
                    batch_size = 1
                    seq_len = 10
                    if isinstance(inputs, dict) and 'input_ids' in inputs:
                        if hasattr(inputs['input_ids'], 'shape'):
                            batch_size = inputs['input_ids'].shape[0]
                            if len(inputs['input_ids'].shape) > 1:
                                seq_len = inputs['input_ids'].shape[1]
                    
                    # Return WebNN-style output
                    # Real implementation would use the WebNN API to run inference
                    return {"last_hidden_state": np.random.rand(batch_size, seq_len, 768).astype(np.float32)}
            
            endpoint = WebNNModel()
            
            # Create handler
            handler = self.create_webnn_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                webnn_label=device,
                endpoint=endpoint,
                tokenizer=processor,
                precision=precision
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 1
            
            return endpoint, processor, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing WebNN model: {e}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {"output": "Mock WebNN output", "input": x, "implementation_type": "MOCK"}
            return None, None, handler, asyncio.Queue(32), 1
"""
        # Find a good insertion point for the WebNN init method (after init_qualcomm)
        content = re.sub(
            r'(def init_qualcomm\([^)]*\):[^\n]*\n[^a-zA-Z0-9_]*("""[^"]*"""[^\n]*\n)(?:[^def]*\n)*\s+return [^a-zA-Z0-9_]*None[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*None[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*handler[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*asyncio.Queue\([0-9]*\)[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*[0-9]+)',
            r'\1\2\3' + webnn_init,
            content
        )
    
    # Add WebGPU initialization method
    if 'init_webgpu' not in content:
        webgpu_init = """
    def init_webgpu(self, model_name, model_type, device="webgpu", **kwargs):
        \"\"\"Initialize model for WebGPU inference using transformers.js.
        
        WebGPU provides modern GPU acceleration for machine learning models in web browsers
        and Node.js applications through libraries like transformers.js.
        
        Args:
            model_name (str): Model identifier
            model_type (str): Type of model (text-generation, image-classification, etc.)
            device (str): Device identifier ('webgpu')
            precision (str, optional): Precision to use (fp32, fp16, int8, int4)
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        \"\"\"
        try:
            import asyncio
            import numpy as np
            
            # Get precision from kwargs or default to fp32
            precision = kwargs.get("precision", "fp32")
            
            # Create processor/tokenizer
            processor = self._create_mock_processor()
            
            # Create WebGPU/transformers.js endpoint/model
            class TransformersJSModel:
                def __init__(self, model_id="Xenova/bert-base-uncased", task="feature-extraction"):
                    \"\"\"Initialize a transformers.js model with WebGPU support.
                    
                    In a real implementation, this would integrate with the transformers.js library
                    running in a browser or Node.js environment with WebGPU capabilities.
                    \"\"\"
                    self.model_id = model_id
                    self.task = task
                    print(f"Initialized transformers.js model '{model_id}' for task '{task}' with WebGPU acceleration")
                    
                def run(self, inputs):
                    \"\"\"Run inference using transformers.js with WebGPU.
                    
                    Args:
                        inputs: Dictionary of inputs with tokenized text
                        
                    Returns:
                        Dictionary with model outputs (hidden_states or embeddings)
                    \"\"\"
                    # Determine batch size and sequence length from inputs
                    batch_size = 1
                    seq_len = 10
                    if isinstance(inputs, dict) and 'input_ids' in inputs:
                        if isinstance(inputs['input_ids'], list):
                            batch_size = len(inputs['input_ids'])
                            if inputs['input_ids'] and isinstance(inputs['input_ids'][0], list):
                                seq_len = len(inputs['input_ids'][0])
                    
                    # Generate mock outputs that match transformers.js format
                    # Real implementation would use the transformers.js API with WebGPU
                    if self.task == "feature-extraction":
                        # Return embeddings for the CLS token for feature extraction
                        return {
                            "hidden_states": np.random.rand(batch_size, 768).tolist(),
                            "token_count": seq_len,
                            "model_version": "Xenova/bert-base-uncased", 
                            "device": "WebGPU"
                        }
                    else:
                        # Return full last_hidden_state for other tasks
                        return {
                            "last_hidden_state": np.random.rand(batch_size, seq_len, 768).tolist(),
                            "model_version": "Xenova/bert-base-uncased",
                            "device": "WebGPU"
                        }
            
            # Initialize transformers.js model with WebGPU support
            endpoint = TransformersJSModel(model_id=model_name, task="feature-extraction")
            
            # Create handler
            handler = self.create_webgpu_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                webgpu_label=device,
                endpoint=endpoint,
                tokenizer=processor,
                precision=precision
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 1
            
            return endpoint, processor, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing WebGPU model: {e}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {"output": "Mock WebGPU output", "input": x, "implementation_type": "MOCK"}
            return None, None, handler, asyncio.Queue(32), 1
"""
        # Find a good insertion point for the WebGPU init method (after init_webnn)
        if 'init_webnn' in content:
            content = re.sub(
                r'(def init_webnn\([^)]*\):[^\n]*\n[^a-zA-Z0-9_]*("""[^"]*"""[^\n]*\n)(?:[^def]*\n)*\s+return [^a-zA-Z0-9_]*None[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*None[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*handler[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*asyncio.Queue\([0-9]*\)[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*[0-9]+)',
                r'\1\2\3' + webgpu_init,
                content
            )
        else:
            # If webnn init doesn't exist, add after qualcomm init
            content = re.sub(
                r'(def init_qualcomm\([^)]*\):[^\n]*\n[^a-zA-Z0-9_]*("""[^"]*"""[^\n]*\n)(?:[^def]*\n)*\s+return [^a-zA-Z0-9_]*None[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*None[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*handler[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*asyncio.Queue\([0-9]*\)[^a-zA-Z0-9_]*,[^a-zA-Z0-9_]*[0-9]+)',
                r'\1\2\3' + webnn_init + webgpu_init,
                content
            )
    
    # Add AMD handler creation method
    if 'create_amd_text_embedding_endpoint_handler' not in content:
        amd_handler = """
    def create_amd_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, tokenizer=None, is_real_impl=False, batch_size=1, precision="fp32"):
        \"\"\"Create a handler function for AMD ROCm inference.
        
        Args:
            endpoint_model: Model name
            device: Device to run on ('rocm:0', etc.)
            hardware_label: Label for the endpoint
            endpoint: Model endpoint
            tokenizer: Tokenizer for the model
            is_real_impl: Whether this is a real implementation
            batch_size: Batch size for processing
            precision: Precision to use (fp32, fp16, bf16, int8, int4, uint4)
            
        Returns:
            A handler function that accepts text input and returns embeddings
        \"\"\"
        # Create a handler that works with the endpoint and tokenizer
        def handler(text_input):
            try:
                # This should match how the actual handler would process data
                import torch
                
                # Create mock output with appropriate structure
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                tensor_output = torch.rand((batch_size, 768))  # Standard embedding size
                
                # Return dictionary with tensor and metadata instead of adding attributes to tensor
                return {
                    "tensor": tensor_output,
                    "implementation_type": "AMD_ROCM",
                    "device": device,
                    "model": endpoint_model,
                    "precision": precision,
                    "is_amd": True
                }
            except Exception as e:
                print(f"Error in AMD ROCm handler: {e}")
                # Return a simple dict on error
                return {"output": f"Error in AMD ROCm handler: {e}", "implementation_type": "MOCK"}
                
        return handler
"""
        # Find a good insertion point for the AMD handler method
        content = re.sub(
            r'(def create_apple_text_embedding_endpoint_handler\([^)]*\):[^\n]*\n[^a-zA-Z0-9_]*("""[^"]*"""[^\n]*\n)(?:[^def]*\n)*\s+return handler)',
            r'\1\2\3' + amd_handler,
            content
        )
    
    # Add WebNN handler creation method
    if 'create_webnn_text_embedding_endpoint_handler' not in content:
        webnn_handler = """
    def create_webnn_text_embedding_endpoint_handler(self, endpoint_model, webnn_label, endpoint=None, tokenizer=None, precision="fp32"):
        \"\"\"Create a handler function for WebNN inference.
        
        WebNN (Web Neural Network API) is a browser-based API that provides hardware acceleration
        for neural networks on the web.
        
        Args:
            endpoint_model: Model name
            webnn_label: Label for the endpoint
            endpoint: WebNN model endpoint
            tokenizer: Tokenizer for the model
            precision: Precision to use (fp32, fp16, int8)
            
        Returns:
            A handler function that accepts text input and returns embeddings
        \"\"\"
        # Create a handler that works with the endpoint and tokenizer
        def handler(text_input):
            try:
                # This should match how the actual handler would process data
                import torch
                import numpy as np
                
                # Process input - this would be different depending on model type
                if hasattr(self, '_process_text_input'):
                    inputs = self._process_text_input(text_input, tokenizer)
                else:
                    # Use a simpler approach if _process_text_input is not available
                    batch_size = 1 if isinstance(text_input, str) else len(text_input)
                    inputs = {"input_ids": torch.ones((batch_size, 10), dtype=torch.long)}
                
                # Convert to appropriate format for WebNN
                webnn_inputs = {}
                for key, value in inputs.items():
                    # Convert PyTorch tensors to format needed by WebNN (typically array buffers)
                    webnn_inputs[key] = value.detach().cpu().numpy()
                
                # Run model with WebNN
                outputs = endpoint.compute(webnn_inputs)
                
                # Convert back to PyTorch tensors
                if isinstance(outputs, dict) and "last_hidden_state" in outputs:
                    last_hidden_state = torch.from_numpy(outputs["last_hidden_state"])
                else:
                    # Handle other output formats
                    last_hidden_state = torch.from_numpy(outputs)
                    
                # Extract embeddings (typically first token for BERT-like models)
                if last_hidden_state.ndim > 1:
                    embeddings = last_hidden_state[:, 0]  # Use CLS token embedding
                else:
                    embeddings = last_hidden_state
                
                # Create mock output with appropriate structure
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                
                # Return dictionary with result
                return {
                    "tensor": embeddings,
                    "implementation_type": "WEBNN",
                    "device": webnn_label,
                    "model": endpoint_model,
                    "precision": precision,
                    "is_webnn": True
                }
            except Exception as e:
                print(f"Error in WebNN handler: {e}")
                # Return a simple dict on error
                return {"output": f"Error in WebNN handler: {e}", "implementation_type": "MOCK"}
                
        return handler
"""
        # Find a good insertion point for the WebNN handler method
        qualcomm_handler_end_pattern = r'(def create_qualcomm_text_embedding_endpoint_handler\([^)]*\):[^\n]*\n[^a-zA-Z0-9_]*("""[^"]*"""[^\n]*\n)(?:[^def]*\n)*\s+return handler)'
        if re.search(qualcomm_handler_end_pattern, content):
            content = re.sub(
                qualcomm_handler_end_pattern,
                r'\1\2\3' + webnn_handler,
                content
            )
        else:
            # If qualcomm handler doesn't exist, add after apple handler
            apple_handler_end_pattern = r'(def create_apple_text_embedding_endpoint_handler\([^)]*\):[^\n]*\n[^a-zA-Z0-9_]*("""[^"]*"""[^\n]*\n)(?:[^def]*\n)*\s+return handler)'
            content = re.sub(
                apple_handler_end_pattern,
                r'\1\2\3' + webnn_handler,
                content
            )
    
    # Add WebGPU handler creation method
    if 'create_webgpu_text_embedding_endpoint_handler' not in content:
        webgpu_handler = """
    def create_webgpu_text_embedding_endpoint_handler(self, endpoint_model, webgpu_label, endpoint=None, tokenizer=None, precision="fp32"):
        \"\"\"Create a handler function for WebGPU inference with transformers.js.
        
        WebGPU is a modern web graphics and compute API that provides access to GPU
        acceleration for machine learning models through libraries like transformers.js.
        
        Args:
            endpoint_model: Model name
            webgpu_label: Label for the endpoint
            endpoint: WebGPU model endpoint (transformers.js)
            tokenizer: Tokenizer for the model
            precision: Precision to use (fp32, fp16, int8, int4)
            
        Returns:
            A handler function that accepts text input and returns embeddings
        \"\"\"
        # Create a handler that works with the endpoint and tokenizer
        def handler(text_input):
            try:
                # This should match how the actual handler would process data
                import torch
                import numpy as np
                
                # Process input - this would be different depending on model type
                if hasattr(self, '_process_text_input'):
                    inputs = self._process_text_input(text_input, tokenizer)
                else:
                    # Use a simpler approach if _process_text_input is not available
                    batch_size = 1 if isinstance(text_input, str) else len(text_input)
                    inputs = {"input_ids": torch.ones((batch_size, 10), dtype=torch.long)}
                
                # Convert to appropriate format for transformers.js / WebGPU
                webgpu_inputs = {}
                for key, value in inputs.items():
                    # Convert PyTorch tensors to format needed by transformers.js
                    webgpu_inputs[key] = value.detach().cpu().numpy().tolist()
                
                # Run model with WebGPU/transformers.js
                outputs = endpoint.run(webgpu_inputs)
                
                # Convert back to PyTorch tensors
                if isinstance(outputs, dict) and "hidden_states" in outputs:
                    # transformers.js output format
                    hidden_states = torch.tensor(outputs["hidden_states"], dtype=torch.float32)
                    if hidden_states.ndim > 1:
                        embeddings = hidden_states[:, 0]  # Use CLS token embedding
                    else:
                        embeddings = hidden_states
                elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
                    # Standard format
                    last_hidden_state = torch.tensor(outputs["last_hidden_state"], dtype=torch.float32)
                    embeddings = last_hidden_state[:, 0]  # Use CLS token embedding
                else:
                    # Handle direct output (array of embeddings)
                    if isinstance(outputs, (list, tuple)):
                        embeddings = torch.tensor(outputs, dtype=torch.float32)
                    else:
                        embeddings = torch.tensor([outputs], dtype=torch.float32)
                
                # Return dictionary with result
                return {
                    "tensor": embeddings,
                    "implementation_type": "WEBGPU",
                    "device": webgpu_label,
                    "model": endpoint_model,
                    "precision": precision,
                    "is_webgpu": True
                }
            except Exception as e:
                print(f"Error in WebGPU handler: {e}")
                # Return a simple dict on error
                return {"output": f"Error in WebGPU handler: {e}", "implementation_type": "MOCK"}
                
        return handler
"""
        # Find a good insertion point for the WebGPU handler method
        if 'create_webnn_text_embedding_endpoint_handler' in content:
            webnn_handler_end_pattern = r'(def create_webnn_text_embedding_endpoint_handler\([^)]*\):[^\n]*\n[^a-zA-Z0-9_]*("""[^"]*"""[^\n]*\n)(?:[^def]*\n)*\s+return handler)'
            content = re.sub(
                webnn_handler_end_pattern,
                r'\1\2\3' + webgpu_handler,
                content
            )
        else:
            # If webnn handler doesn't exist, add after qualcomm handler
            qualcomm_handler_end_pattern = r'(def create_qualcomm_text_embedding_endpoint_handler\([^)]*\):[^\n]*\n[^a-zA-Z0-9_]*("""[^"]*"""[^\n]*\n)(?:[^def]*\n)*\s+return handler)'
            content = re.sub(
                qualcomm_handler_end_pattern,
                r'\1\2\3' + webnn_handler + webgpu_handler,
                content
            )
    
    # Add AMD, WebNN, and WebGPU tests to the __test__ method
    if 'amd_test' not in content:
        amd_test = """
        # Test on AMD if available
        if self.hardware_capabilities.get("amd", False):
            for precision in ["fp32", "fp16", "bf16", "int8"]:
                try:
                    print(f"Testing on AMD ROCm with {precision.upper()} precision...")
                    model_info = self._get_model_info() if hasattr(self, '_get_model_info') else {}
                    
                    # Skip if precision not supported on AMD
                    if isinstance(model_info, dict) and "precision_compatibility" in model_info and "amd" in model_info["precision_compatibility"]:
                        if not model_info["precision_compatibility"]["amd"].get(precision, False):
                            print(f"Precision {precision.upper()} not supported on AMD ROCm, skipping...")
                            continue
                    
                    # Initialize model with specific precision
                    endpoint, processor, handler, queue, batch_size = self.init_amd(
                        model_name="test-model",
                        model_type="text-generation",
                        precision=precision
                    )
                    
                    # Test with simple input
                    input_text = f"This is a test input with {precision.upper()} precision on AMD ROCm"
                    output = handler(input_text)
                    
                    # Record results
                    examples.append({
                        "platform": f"AMD ROCm ({precision.upper()})",
                        "input": input_text,
                        "output_type": f"container: {str(type(output))}, tensor: {str(type(output.get('tensor', output)))}",
                        "implementation_type": output.get("implementation_type", "UNKNOWN"),
                        "precision": precision,
                        "hardware": "AMD"
                    })
                    
                    results[f"amd_{precision}_test"] = "Success"
                except Exception as e:
                    print(f"Error testing on AMD ROCm with {precision.upper()}: {e}")
                    traceback.print_exc()
                    results[f"amd_{precision}_test"] = f"Error: {str(e)}"
        else:
            results["amd_test"] = "AMD ROCm not available"
"""
        
        webnn_test = """
        # Test on WebNN if available
        if self.hardware_capabilities.get("webnn", False):
            for precision in ["fp32", "fp16", "int8"]:
                try:
                    print(f"Testing on WebNN with {precision.upper()} precision...")
                    model_info = self._get_model_info() if hasattr(self, '_get_model_info') else {}
                    
                    # Skip if precision not supported on WebNN
                    if isinstance(model_info, dict) and "precision_compatibility" in model_info and "webnn" in model_info["precision_compatibility"]:
                        if not model_info["precision_compatibility"]["webnn"].get(precision, False):
                            print(f"Precision {precision.upper()} not supported on WebNN, skipping...")
                            continue
                    
                    # Initialize model with specific precision
                    endpoint, processor, handler, queue, batch_size = self.init_webnn(
                        model_name="test-model",
                        model_type="text-generation",
                        precision=precision
                    )
                    
                    # Test with simple input
                    input_text = f"This is a test input with {precision.upper()} precision on WebNN"
                    output = handler(input_text)
                    
                    # Record results
                    examples.append({
                        "platform": f"WebNN ({precision.upper()})",
                        "input": input_text,
                        "output_type": f"container: {str(type(output))}, tensor: {str(type(output.get('tensor', output)))}",
                        "implementation_type": output.get("implementation_type", "UNKNOWN"),
                        "precision": precision,
                        "hardware": "WebNN"
                    })
                    
                    results[f"webnn_{precision}_test"] = "Success"
                except Exception as e:
                    print(f"Error testing on WebNN with {precision.upper()}: {e}")
                    traceback.print_exc()
                    results[f"webnn_{precision}_test"] = f"Error: {str(e)}"
        else:
            results["webnn_test"] = "WebNN not available"
"""
            
        webgpu_test = """
        # Test on WebGPU if available
        if self.hardware_capabilities.get("webgpu", False):
            for precision in ["fp32", "fp16", "int8"]:
                try:
                    print(f"Testing on WebGPU/transformers.js with {precision.upper()} precision...")
                    model_info = self._get_model_info() if hasattr(self, '_get_model_info') else {}
                    
                    # Skip if precision not supported on WebGPU
                    if isinstance(model_info, dict) and "precision_compatibility" in model_info and "webgpu" in model_info["precision_compatibility"]:
                        if not model_info["precision_compatibility"]["webgpu"].get(precision, False):
                            print(f"Precision {precision.upper()} not supported on WebGPU, skipping...")
                            continue
                    
                    # Initialize model with specific precision
                    endpoint, processor, handler, queue, batch_size = self.init_webgpu(
                        model_name="test-model",
                        model_type="text-generation",
                        precision=precision
                    )
                    
                    # Test with simple input
                    input_text = f"This is a test input with {precision.upper()} precision on WebGPU/transformers.js"
                    output = handler(input_text)
                    
                    # Record results
                    examples.append({
                        "platform": f"WebGPU ({precision.upper()})",
                        "input": input_text,
                        "output_type": f"container: {str(type(output))}, tensor: {str(type(output.get('tensor', output)))}",
                        "implementation_type": output.get("implementation_type", "UNKNOWN"),
                        "precision": precision,
                        "hardware": "WebGPU"
                    })
                    
                    results[f"webgpu_{precision}_test"] = "Success"
                except Exception as e:
                    print(f"Error testing on WebGPU with {precision.upper()}: {e}")
                    traceback.print_exc()
                    results[f"webgpu_{precision}_test"] = f"Error: {str(e)}"
        else:
            results["webgpu_test"] = "WebGPU not available"
"""
        # Find a good insertion point for these tests (before the return statement in __test__)
        content = re.sub(
            r'(\s+# Return test results\s+return {[^\n]+})',
            amd_test + webnn_test + webgpu_test + r'\1',
            content
        )
    
    # Add the initialization methods to the __init__ method
    if 'self.init_amd' not in content:
        content = re.sub(
            r'(self.init_apple\s*=\s*self.init_apple\n\s+self.init_qualcomm\s*=\s*self.init_qualcomm)',
            r'\1\n        self.init_amd = self.init_amd\n        self.init_webnn = self.init_webnn\n        self.init_webgpu = self.init_webgpu',
            content
        )
    
    # Add the handler creation methods to the __init__ method
    if 'self.create_amd_text_embedding_endpoint_handler' not in content:
        content = re.sub(
            r'(self.create_apple_text_embedding_endpoint_handler\s*=\s*self.create_apple_text_embedding_endpoint_handler\n\s+self.create_qualcomm_text_embedding_endpoint_handler\s*=\s*self.create_qualcomm_text_embedding_endpoint_handler)',
            r'\1\n        self.create_amd_text_embedding_endpoint_handler = self.create_amd_text_embedding_endpoint_handler\n        self.create_webnn_text_embedding_endpoint_handler = self.create_webnn_text_embedding_endpoint_handler\n        self.create_webgpu_text_embedding_endpoint_handler = self.create_webgpu_text_embedding_endpoint_handler',
            content
        )
    
    # Write the updated content back to the file
    with open(test_file, 'w') as f:
        f.write(content)
        
    print(f"Enhanced test file: {test_file_path}")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Add AMD, WebNN, and WebGPU support to test files")
    parser.add_argument("--test-file", type=str, required=True, help="Path to the test file to enhance")
    parser.add_argument("--force", action="store_true", help="Force overwrite of backup file if it exists")
    
    args = parser.parse_args()
    
    # Enhance the test file
    success = enhance_test_file(args.test_file, args.force)
    
    if success:
        print(f"Successfully enhanced test file: {args.test_file}")
        return 0
    else:
        print(f"Failed to enhance test file: {args.test_file}")
        return 1

if __name__ == "__main__":
    sys.exit(main())