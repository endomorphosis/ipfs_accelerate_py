#!/usr/bin/env python3
"""
Fix the indentation issues in the reference template.
"""

from ipfs_accelerate_py.worker.anyio_queue import AnyioQueue
import os
import re
import sys

def fix_template_indentation(template_path):
    """Fix indentation issues in the template file."""
    print(f"Fixing indentation in template: {template_path}")
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Fix all function replacement placeholders
    placeholders = [
        "{cpu_inference_code}", "{cpu_result_format}",
        "{cuda_inference_code}", "{cuda_result_format}",
        "{openvino_inference_code}", "{openvino_result_format}",
        "{apple_inference_code}", "{apple_result_format}",
        "{qualcomm_result_format}", "{mock_tokenize_output}",
        "{mock_forward_output}"
    ]
    
    for placeholder in placeholders:
        # Replace placeholder with properly indented code
        content = content.replace(placeholder, "# Code will be generated here")
    
    # Remove any unexpected indentation errors
    lines = content.splitlines()
    fixed_lines = []
    
    in_indented_block = False
    for line in lines:
        # Check for function definitions outside of class
        if re.match(r'^def\s+\w+', line) and not line.startswith('    def'):
            # This is a function outside of class, fix it
            fixed_lines.append(f"    {line}")
            in_indented_block = True
        # Check for continuing indentation in function outside class
        elif in_indented_block and line and not line.startswith('    ') and not line.startswith(')'):
            fixed_lines.append(f"    {line}")
        # Normal line
        else:
            fixed_lines.append(line)
            if line.strip() == '':
                in_indented_block = False
    
    # Write fixed content
    with open(template_path, 'w') as f:
        f.write('\n'.join(fixed_lines))
    
    print(f"Template fixed successfully")
    return True

def create_simple_template():
    """Create a simplified reference template."""
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                "templates", "simple_reference_template.py")
    
    print(f"Creating simplified template: {template_path}")
    
    template_content = """import asyncio
import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple

class hf_{model_type}:
    \"\"\"HuggingFace {model_type_upper} implementation.
    
    This class provides standardized interfaces for working with {model_type_upper} models
    across different hardware backends (CPU, CUDA, OpenVINO, Apple, Qualcomm).
    
    {model_description}
    \"\"\"
    
    def __init__(self, resources=None, metadata=None):
        \"\"\"Initialize the {model_type_upper} model.
        
        Args:
            resources (dict): Dictionary of shared resources (torch, transformers, etc.)
            metadata (dict): Configuration metadata
        \"\"\"
        self.resources = resources
        self.metadata = metadata
        
        # Initialization methods
        self.init = self.init
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init_apple = self.init_apple
        self.init_qualcomm = self.init_qualcomm
        
        # Test methods
        self.__test__ = self.__test__
        
        # Hardware-specific utilities
        self.snpe_utils = None
        return None
        
    def init(self):        
        if "torch" not in list(self.resources.keys()):
            import torch
            self.torch = torch
        else:
            self.torch = self.resources["torch"]

        if "transformers" not in list(self.resources.keys()):
            import transformers
            self.transformers = transformers
        else:
            self.transformers = self.resources["transformers"]
            
        if "numpy" not in list(self.resources.keys()):
            import numpy as np
            self.np = np
        else:
            self.np = self.resources["numpy"]

        return None
    
    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        \"\"\"Test function to validate endpoint functionality.\"\"\"
        test_input = "{test_input}"
        timestamp1 = time.time()
        test_batch = None
        
        try:
            # Run the model
            test_batch = endpoint_handler(test_input)
            print(test_batch)
            print("hf_{model_type} test passed")
            return True
        except Exception as e:
            print(e)
            print("hf_{model_type} test failed")
            return False
    
    def init_cpu(self, model_name, device, cpu_label):
        \"\"\"Initialize {model_type_upper} model for CPU inference.\"\"\"
        self.init()
        
        print(f"Loading {model_name} for CPU inference...")
        
        try:
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            model = {automodel_class}.from_pretrained(
                model_name,
                torch_dtype=self.torch.float32,
                device_map="cpu"
            )
            
            model.eval()
            
            # Create handler function
            handler = self.create_cpu_{task_type}_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=cpu_label,
                endpoint=model,
                tokenizer=tokenizer
            )
            
            return model, tokenizer, handler, AnyioQueue(32), 0
            
        except Exception as e:
            print(f"Error initializing CPU endpoint: {e}")
            return None, None, None, AnyioQueue(32), 0
    
    def init_cuda(self, model_name, device, cuda_label):
        \"\"\"Initialize {model_type_upper} model for CUDA inference.\"\"\"
        self.init()
        
        if not self.torch.cuda.is_available():
            print(f"CUDA not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", cuda_label.replace("cuda", "cpu"))
        
        print(f"Loading {model_name} for CUDA inference...")
        
        try:
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            model = {automodel_class}.from_pretrained(
                model_name,
                torch_dtype=self.torch.float16,
                device_map=device
            )
            
            model.eval()
            
            # Create handler function
            handler = self.create_cuda_{task_type}_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=cuda_label,
                endpoint=model,
                tokenizer=tokenizer
            )
            
            return model, tokenizer, handler, AnyioQueue(32), 0
            
        except Exception as e:
            print(f"Error initializing CUDA endpoint: {e}")
            return None, None, None, AnyioQueue(32), 0
    
    def init_openvino(self, model_name, device, openvino_label):
        \"\"\"Initialize {model_type_upper} model for OpenVINO inference.\"\"\"
        self.init()
        
        try:
            from optimum.intel import OVModelFor{task_class}
        except ImportError:
            print(f"OpenVINO not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", openvino_label.replace("openvino", "cpu"))
        
        print(f"Loading {model_name} for OpenVINO inference...")
        
        try:
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model_name)
            
            # Load model with OpenVINO optimization
            model = OVModelFor{task_class}.from_pretrained(
                model_name,
                device=device,
                export=True
            )
            
            # Create handler function
            handler = self.create_openvino_{task_type}_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=openvino_label,
                endpoint=model,
                tokenizer=tokenizer
            )
            
            return model, tokenizer, handler, AnyioQueue(32), 0
            
        except Exception as e:
            print(f"Error initializing OpenVINO endpoint: {e}")
            return None, None, None, AnyioQueue(32), 0
    
    def init_apple(self, model_name, device, apple_label):
        \"\"\"Initialize {model_type_upper} model for Apple Silicon (MPS) inference.\"\"\"
        self.init()
        
        if not (hasattr(self.torch, 'backends') and 
                hasattr(self.torch.backends, 'mps') and 
                self.torch.backends.mps.is_available()):
            print(f"Apple MPS not available, falling back to CPU")
            return self.init_cpu(model_name, "cpu", apple_label.replace("apple", "cpu"))
        
        print(f"Loading {model_name} for Apple Silicon inference...")
        
        try:
            # Load tokenizer
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            model = {automodel_class}.from_pretrained(
                model_name,
                torch_dtype=self.torch.float32,
                device_map="mps"
            )
            
            model.eval()
            
            # Create handler function
            handler = self.create_apple_{task_type}_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=apple_label,
                endpoint=model,
                tokenizer=tokenizer
            )
            
            return model, tokenizer, handler, AnyioQueue(32), 0
            
        except Exception as e:
            print(f"Error initializing Apple Silicon endpoint: {e}")
            return None, None, None, AnyioQueue(32), 0
    
    def init_qualcomm(self, model_name, device, qualcomm_label):
        \"\"\"Initialize {model_type_upper} model for Qualcomm inference.\"\"\"
        self.init()
        
        # For now, we create a mock implementation since Qualcomm SDK integration requires specific hardware
        print("Qualcomm implementation is a mock for now")
        return None, None, None, AnyioQueue(32), 0
    
    def create_cpu_{task_type}_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        \"\"\"Create handler function for CPU {task_type} endpoint.\"\"\"
        def handler(text, *args, **kwargs):
            try:
                # Convert single string to list for batch processing
                if isinstance(text, str):
                    batch = [text]
                else:
                    batch = text
                
                # Tokenize input
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                
                # Move inputs to the correct device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run inference with no gradient calculation
                with self.torch.no_grad():
                    outputs = endpoint(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
                
                return {
                    "success": True,
                    "embeddings": embeddings,
                    "device": device,
                    "hardware": hardware_label
                }
                
            except Exception as e:
                print(f"Error in CPU handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler
    
    def create_cuda_{task_type}_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        \"\"\"Create handler function for CUDA {task_type} endpoint.\"\"\"
        def handler(text, *args, **kwargs):
            try:
                # Convert single string to list for batch processing
                if isinstance(text, str):
                    batch = [text]
                else:
                    batch = text
                
                # Tokenize input
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                
                # Move inputs to the correct device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run inference with no gradient calculation
                with self.torch.no_grad():
                    outputs = endpoint(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
                
                return {
                    "success": True,
                    "embeddings": embeddings,
                    "device": device,
                    "hardware": hardware_label
                }
                
            except Exception as e:
                print(f"Error in CUDA handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler
    
    def create_openvino_{task_type}_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        \"\"\"Create handler function for OpenVINO {task_type} endpoint.\"\"\"
        def handler(text, *args, **kwargs):
            try:
                # Convert single string to list for batch processing
                if isinstance(text, str):
                    batch = [text]
                else:
                    batch = text
                
                # Tokenize input
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                
                # Run inference
                outputs = endpoint(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
                
                return {
                    "success": True,
                    "embeddings": embeddings,
                    "device": device,
                    "hardware": hardware_label
                }
                
            except Exception as e:
                print(f"Error in OpenVINO handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler
    
    def create_apple_{task_type}_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        \"\"\"Create handler function for Apple Silicon {task_type} endpoint.\"\"\"
        def handler(text, *args, **kwargs):
            try:
                # Convert single string to list for batch processing
                if isinstance(text, str):
                    batch = [text]
                else:
                    batch = text
                
                # Tokenize input
                inputs = tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                
                # Move inputs to the correct device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run inference with no gradient calculation
                with self.torch.no_grad():
                    outputs = endpoint(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
                
                return {
                    "success": True,
                    "embeddings": embeddings,
                    "device": device,
                    "hardware": hardware_label
                }
                
            except Exception as e:
                print(f"Error in Apple handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler
    
    def create_qualcomm_{task_type}_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint, tokenizer):
        \"\"\"Create handler function for Qualcomm {task_type} endpoint.\"\"\"
        def handler(text, *args, **kwargs):
            try:
                # This is a placeholder for Qualcomm implementation
                # Convert single string to list for batch processing
                if isinstance(text, str):
                    batch = [text]
                else:
                    batch = text
                
                return {
                    "success": True,
                    "device": device,
                    "hardware": hardware_label,
                    "embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(10)]
                }
                
            except Exception as e:
                print(f"Error in Qualcomm handler: {e}")
                return {"success": False, "error": str(e)}
        
        return handler
"""
    
    with open(template_path, 'w') as f:
        f.write(template_content)
    
    print(f"Simple template created successfully")
    return template_path

def main():
    """Main entry point."""
    # First try to fix the existing template
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "templates", "hf_reference_template.py")
    
    if not os.path.exists(template_path):
        print(f"Template not found: {template_path}")
        return 1
    
    # Create a backup of the original template
    backup_path = template_path + ".bak"
    try:
        with open(template_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
        print(f"Created backup: {backup_path}")
    except Exception as e:
        print(f"Error creating backup: {e}")
        return 1
    
    # Fix the template
    if not fix_template_indentation(template_path):
        print("Failed to fix template, creating simple template instead")
        simple_template_path = create_simple_template()
        print(f"Created simplified template: {simple_template_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())