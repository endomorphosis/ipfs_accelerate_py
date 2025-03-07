"""
ONNX Verification and Conversion Utility

This module provides utilities for verifying ONNX model availability and converting
PyTorch models to ONNX format when the original ONNX files are not available.
"""

import os
import logging
import hashlib
import json
import requests
from pathlib import Path
import torch
from typing import Dict, Optional, Union, Any, Tuple
from datetime import datetime

class OnnxVerificationError(Exception):
    """Base exception for ONNX verification errors."""
    pass

class OnnxConversionError(Exception):
    """Base exception for ONNX conversion errors."""
    pass

class OnnxVerifier:
    """Utility for verifying ONNX model availability before benchmarks."""
    
    def __init__(self, cache_dir: str = None, registry_path: str = None, 
                 max_retries: int = 3, timeout: int = 30):
        self.logger = logging.getLogger("OnnxVerifier")
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".ipfs_accelerate", "model_cache")
        self.registry_path = registry_path or os.path.join(self.cache_dir, "conversion_registry.json")
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize cache directory and registry
        os.makedirs(self.cache_dir, exist_ok=True)
        self._init_registry()
        
        # Initialize converter
        self.converter = PyTorchToOnnxConverter(cache_dir=self.cache_dir)
        
        self.logger.info(f"OnnxVerifier initialized with cache at {self.cache_dir}")
    
    def _init_registry(self):
        """Initialize or load the conversion registry."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    self.registry = json.load(f)
                self.logger.info(f"Loaded conversion registry with {len(self.registry)} entries")
            except Exception as e:
                self.logger.error(f"Error loading registry: {e}. Creating new registry.")
                self.registry = {}
        else:
            self.registry = {}
            self._save_registry()
            self.logger.info("Created new conversion registry")
    
    def _save_registry(self):
        """Save the conversion registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def verify_onnx_file(self, model_id: str, onnx_file_path: str) -> Tuple[bool, str]:
        """
        Verify if an ONNX file exists at the specified HuggingFace path.
        
        Args:
            model_id: HuggingFace model ID (e.g., "bert-base-uncased")
            onnx_file_path: Path to the ONNX file within the repository
            
        Returns:
            Tuple of (success, message)
        """
        self.logger.info(f"Verifying ONNX file for {model_id}: {onnx_file_path}")
        
        # Check if we have a valid cached conversion
        cache_key = f"{model_id}:{onnx_file_path}"
        if cache_key in self.registry and os.path.exists(self.registry[cache_key]["local_path"]):
            self.logger.info(f"Using cached conversion for {cache_key}")
            return True, self.registry[cache_key]["local_path"]
        
        # Check if the model exists on HuggingFace
        hf_url = f"https://huggingface.co/{model_id}/resolve/main/{onnx_file_path}"
        response = None
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Attempt {attempt+1}/{self.max_retries} to verify {hf_url}")
                response = requests.head(hf_url, timeout=self.timeout)
                
                if response.status_code == 200:
                    self.logger.info(f"ONNX file {hf_url} exists.")
                    return True, hf_url
                
                if response.status_code == 404:
                    self.logger.warning(f"ONNX file {hf_url} not found (404).")
                    break
                
                self.logger.warning(f"Received status code {response.status_code} for {hf_url}")
            except requests.RequestException as e:
                self.logger.warning(f"Request error for {hf_url}: {e}")
            
            # Only retry for certain errors
            if response and response.status_code not in [408, 429, 500, 502, 503, 504]:
                break
        
        self.logger.warning(f"ONNX file verification failed for {model_id}:{onnx_file_path}")
        return False, f"ONNX file verification failed after {self.max_retries} attempts"
    
    def get_onnx_model(self, model_id: str, onnx_file_path: str, 
                      conversion_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Get an ONNX model, using conversion from PyTorch if necessary.
        
        Args:
            model_id: HuggingFace model ID
            onnx_file_path: Path to the ONNX file within the repository
            conversion_config: Configuration for conversion if needed
            
        Returns:
            Path to the ONNX model file (either remote or local)
        """
        # First, try to verify if the ONNX file exists
        success, result = self.verify_onnx_file(model_id, onnx_file_path)
        if success:
            return result
        
        # If verification failed, try to convert from PyTorch
        self.logger.info(f"ONNX file not found, attempting conversion for {model_id}")
        
        try:
            local_path = self.converter.convert_from_pytorch(
                model_id=model_id,
                target_path=onnx_file_path,
                config=conversion_config
            )
            
            # Register the conversion in the registry
            cache_key = f"{model_id}:{onnx_file_path}"
            self.registry[cache_key] = {
                "model_id": model_id,
                "onnx_path": onnx_file_path,
                "local_path": local_path,
                "conversion_time": datetime.now().isoformat(),
                "conversion_config": conversion_config,
                "source": "pytorch_conversion"
            }
            self._save_registry()
            
            self.logger.info(f"Successfully converted {model_id} to ONNX at {local_path}")
            return local_path
            
        except Exception as e:
            self.logger.error(f"Failed to convert {model_id} to ONNX: {e}")
            raise OnnxConversionError(f"Failed to convert {model_id} to ONNX: {str(e)}")

class PyTorchToOnnxConverter:
    """Handles conversion from PyTorch models to ONNX format."""
    
    def __init__(self, cache_dir: str = None):
        self.logger = logging.getLogger("PyTorchToOnnxConverter")
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".ipfs_accelerate", "model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.logger.info(f"PyTorchToOnnxConverter initialized with cache at {self.cache_dir}")
    
    def convert_from_pytorch(self, model_id: str, target_path: str, 
                           config: Optional[Dict[str, Any]] = None) -> str:
        """
        Convert a PyTorch model to ONNX format.
        
        Args:
            model_id: HuggingFace model ID 
            target_path: Target path for the ONNX file
            config: Configuration for conversion
            
        Returns:
            Path to the converted ONNX file
        """
        try:
            # Import libraries only when needed to avoid dependencies when just verifying
            from transformers import AutoModel, AutoTokenizer, AutoConfig
            
            self.logger.info(f"Converting {model_id} to ONNX format")
            
            # Create a unique cache path based on model ID and target path
            model_hash = hashlib.md5(f"{model_id}:{target_path}".encode()).hexdigest()
            cache_subdir = os.path.join(self.cache_dir, model_hash)
            os.makedirs(cache_subdir, exist_ok=True)
            
            # Determine output path
            filename = os.path.basename(target_path)
            output_path = os.path.join(cache_subdir, filename)
            
            # Load model-specific configuration or use defaults
            config = config or {}
            model_type = config.get('model_type', self._detect_model_type(model_id))
            input_shapes = config.get('input_shapes', self._get_default_input_shapes(model_type))
            opset_version = config.get('opset_version', 12)
            
            # Load the PyTorch model
            self.logger.info(f"Loading PyTorch model {model_id}")
            model = self._load_pytorch_model(model_id, model_type)
            
            # Generate dummy input
            dummy_input = self._create_dummy_input(model_id, model_type, input_shapes)
            
            # Export to ONNX
            self.logger.info(f"Exporting {model_id} to ONNX with opset {opset_version}")
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=config.get('input_names', ['input']),
                output_names=config.get('output_names', ['output']),
                dynamic_axes=config.get('dynamic_axes', None)
            )
            
            # Verify the ONNX model
            self._verify_onnx_model(output_path)
            
            self.logger.info(f"Successfully exported {model_id} to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error converting {model_id} to ONNX: {e}")
            raise OnnxConversionError(f"Failed to convert {model_id} to ONNX: {str(e)}")
    
    def _detect_model_type(self, model_id: str) -> str:
        """Detect the model type based on model ID."""
        # This is a simplified detection logic
        model_id_lower = model_id.lower()
        
        if 'bert' in model_id_lower:
            return 'bert'
        elif 't5' in model_id_lower:
            return 't5'  
        elif 'gpt' in model_id_lower:
            return 'gpt'
        elif 'vit' in model_id_lower or 'vision' in model_id_lower:
            return 'vit'
        elif 'clip' in model_id_lower:
            return 'clip'
        elif 'whisper' in model_id_lower:
            return 'whisper'
        elif 'wav2vec' in model_id_lower:
            return 'wav2vec2'
        else:
            return 'unknown'
    
    def _get_default_input_shapes(self, model_type: str) -> Dict[str, Any]:
        """Get default input shapes based on model type."""
        if model_type == 'bert':
            return {'batch_size': 1, 'sequence_length': 128}
        elif model_type == 't5':
            return {'batch_size': 1, 'sequence_length': 128}
        elif model_type == 'gpt':
            return {'batch_size': 1, 'sequence_length': 128}
        elif model_type == 'vit':
            return {'batch_size': 1, 'channels': 3, 'height': 224, 'width': 224}
        elif model_type == 'clip':
            return {
                'vision': {'batch_size': 1, 'channels': 3, 'height': 224, 'width': 224},
                'text': {'batch_size': 1, 'sequence_length': 77}
            }
        elif model_type == 'whisper':
            return {'batch_size': 1, 'feature_size': 80, 'sequence_length': 3000}
        elif model_type == 'wav2vec2':
            return {'batch_size': 1, 'sequence_length': 16000}
        else:
            return {'batch_size': 1, 'sequence_length': 128}
    
    def _load_pytorch_model(self, model_id: str, model_type: str):
        """Load the appropriate PyTorch model based on model type."""
        try:
            from transformers import AutoModel, AutoTokenizer, AutoConfig
            from transformers import (
                BertModel, T5Model, GPT2Model, ViTModel, 
                CLIPModel, WhisperModel, Wav2Vec2Model
            )
            
            # Model-specific loading logic
            if model_type == 'bert':
                return BertModel.from_pretrained(model_id)
            elif model_type == 't5':
                return T5Model.from_pretrained(model_id)
            elif model_type == 'gpt':
                return GPT2Model.from_pretrained(model_id)
            elif model_type == 'vit':
                return ViTModel.from_pretrained(model_id)
            elif model_type == 'clip':
                return CLIPModel.from_pretrained(model_id)
            elif model_type == 'whisper':
                return WhisperModel.from_pretrained(model_id)
            elif model_type == 'wav2vec2':
                return Wav2Vec2Model.from_pretrained(model_id)
            else:
                # Generic loading as fallback
                return AutoModel.from_pretrained(model_id)
                
        except Exception as e:
            self.logger.error(f"Error loading PyTorch model {model_id}: {e}")
            raise OnnxConversionError(f"Failed to load PyTorch model {model_id}: {str(e)}")
    
    def _create_dummy_input(self, model_id: str, model_type: str, input_shapes: Dict[str, Any]):
        """Create dummy input tensors for the model."""
        try:
            if model_type == 'bert':
                batch_size = input_shapes.get('batch_size', 1)
                seq_length = input_shapes.get('sequence_length', 128)
                return {
                    'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
                    'attention_mask': torch.ones(batch_size, seq_length)
                }
            elif model_type == 't5':
                batch_size = input_shapes.get('batch_size', 1)
                seq_length = input_shapes.get('sequence_length', 128)
                return {
                    'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
                    'attention_mask': torch.ones(batch_size, seq_length)
                }
            elif model_type == 'gpt':
                batch_size = input_shapes.get('batch_size', 1)
                seq_length = input_shapes.get('sequence_length', 128)
                return torch.randint(0, 1000, (batch_size, seq_length))
            elif model_type == 'vit':
                batch_size = input_shapes.get('batch_size', 1)
                channels = input_shapes.get('channels', 3)
                height = input_shapes.get('height', 224)
                width = input_shapes.get('width', 224)
                return torch.rand(batch_size, channels, height, width)
            elif model_type == 'clip':
                # CLIP has multiple inputs (text and image)
                vision_shapes = input_shapes.get('vision', {})
                text_shapes = input_shapes.get('text', {})
                
                batch_size_vision = vision_shapes.get('batch_size', 1)
                channels = vision_shapes.get('channels', 3)
                height = vision_shapes.get('height', 224)
                width = vision_shapes.get('width', 224)
                
                batch_size_text = text_shapes.get('batch_size', 1)
                seq_length = text_shapes.get('sequence_length', 77)
                
                return {
                    'pixel_values': torch.rand(batch_size_vision, channels, height, width),
                    'input_ids': torch.randint(0, 1000, (batch_size_text, seq_length))
                }
            elif model_type == 'whisper':
                batch_size = input_shapes.get('batch_size', 1)
                feature_size = input_shapes.get('feature_size', 80)
                seq_length = input_shapes.get('sequence_length', 3000)
                return torch.rand(batch_size, feature_size, seq_length)
            elif model_type == 'wav2vec2':
                batch_size = input_shapes.get('batch_size', 1)
                seq_length = input_shapes.get('sequence_length', 16000)
                return torch.rand(batch_size, seq_length)
            else:
                # Generic fallback
                batch_size = input_shapes.get('batch_size', 1)
                seq_length = input_shapes.get('sequence_length', 128)
                return torch.randint(0, 1000, (batch_size, seq_length))
                
        except Exception as e:
            self.logger.error(f"Error creating dummy input for {model_id}: {e}")
            raise OnnxConversionError(f"Failed to create dummy input for {model_id}: {str(e)}")
    
    def _verify_onnx_model(self, onnx_path: str):
        """Verify that the ONNX model is valid."""
        try:
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            self.logger.info(f"ONNX model verification passed for {onnx_path}")
        except Exception as e:
            self.logger.error(f"ONNX model verification failed for {onnx_path}: {e}")
            raise OnnxConversionError(f"ONNX model verification failed: {str(e)}")

# Integration with benchmark system
def verify_and_get_onnx_model(model_id: str, onnx_path: str, conversion_config: Optional[Dict[str, Any]] = None) -> Tuple[str, bool]:
    """
    Helper function to get ONNX model path with fallback to conversion.
    For integration into benchmark runners.
    
    Args:
        model_id: HuggingFace model ID
        onnx_path: Path to the ONNX file
        conversion_config: Optional configuration for conversion
        
    Returns:
        Tuple of (model_path, was_converted)
    """
    verifier = OnnxVerifier()
    try:
        # First, verify if the ONNX file exists directly
        success, result = verifier.verify_onnx_file(model_id, onnx_path)
        if success:
            return result, False  # False indicates it wasn't converted
            
        # If not found, try conversion
        local_path = verifier.converter.convert_from_pytorch(
            model_id=model_id,
            target_path=onnx_path,
            config=conversion_config
        )
        
        # Register the conversion
        cache_key = f"{model_id}:{onnx_path}"
        verifier.registry[cache_key] = {
            "model_id": model_id,
            "onnx_path": onnx_path,
            "local_path": local_path,
            "conversion_time": datetime.now().isoformat(),
            "conversion_config": conversion_config,
            "source": "pytorch_conversion"
        }
        verifier._save_registry()
        
        return local_path, True  # True indicates it was converted
        
    except OnnxConversionError as e:
        logging.error(f"Error in ONNX verification/conversion: {e}")
        raise

# Example usage in benchmarking script
def benchmark_example_usage():
    """Example showing how to use the ONNX verification in a benchmark script."""
    model_id = "bert-base-uncased"
    onnx_path = "model.onnx"
    
    try:
        # Get the model path, with conversion if needed
        model_path, was_converted = verify_and_get_onnx_model(model_id, onnx_path)
        
        # Log whether the model was converted
        if was_converted:
            logging.info(f"Using converted ONNX model for {model_id} at {model_path}")
        else:
            logging.info(f"Using original ONNX model for {model_id} at {model_path}")
        
        # Continue with the benchmark using the model_path
        # ...
        
    except OnnxVerificationError as e:
        logging.error(f"ONNX verification failed: {e}")
    except OnnxConversionError as e:
        logging.error(f"ONNX conversion failed: {e}")