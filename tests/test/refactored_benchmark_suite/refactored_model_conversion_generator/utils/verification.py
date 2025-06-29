"""
Model verification utilities.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class ModelVerifier:
    """
    Utilities for verifying model files and conversions.
    """
    
    @staticmethod
    def verify_onnx_model(model_path: str) -> Tuple[bool, Optional[str]]:
        """
        Verify ONNX model.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Tuple of (valid, error_message)
        """
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"
                
            # Try to import onnx
            try:
                import onnx
            except ImportError:
                return False, "ONNX library not installed"
                
            # Load model
            try:
                model = onnx.load(model_path)
            except Exception as e:
                return False, f"Error loading ONNX model: {e}"
                
            # Check model
            try:
                onnx.checker.check_model(model)
            except Exception as e:
                return False, f"ONNX model validation failed: {e}"
                
            return True, None
            
        except Exception as e:
            return False, f"Error verifying ONNX model: {e}"
            
    @staticmethod
    def verify_pytorch_model(model_path: str) -> Tuple[bool, Optional[str]]:
        """
        Verify PyTorch model.
        
        Args:
            model_path: Path to PyTorch model
            
        Returns:
            Tuple of (valid, error_message)
        """
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"
                
            # Try to import torch
            try:
                import torch
            except ImportError:
                return False, "PyTorch library not installed"
                
            # Load model
            try:
                model = torch.load(model_path, map_location='cpu')
            except Exception as e:
                return False, f"Error loading PyTorch model: {e}"
                
            return True, None
            
        except Exception as e:
            return False, f"Error verifying PyTorch model: {e}"
            
    @staticmethod
    def verify_openvino_model(model_path: str) -> Tuple[bool, Optional[str]]:
        """
        Verify OpenVINO model.
        
        Args:
            model_path: Path to OpenVINO model (XML)
            
        Returns:
            Tuple of (valid, error_message)
        """
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"
                
            # Check if corresponding bin file exists
            bin_path = os.path.splitext(model_path)[0] + '.bin'
            if not os.path.exists(bin_path):
                return False, f"Binary file not found: {bin_path}"
                
            # Try to import OpenVINO
            try:
                try:
                    # Try newer OpenVINO API
                    from openvino.runtime import Core
                    ie = Core()
                except ImportError:
                    # Try older OpenVINO API
                    from openvino.inference_engine import IECore
                    ie = IECore()
            except ImportError:
                return False, "OpenVINO library not installed"
                
            # Load model
            try:
                network = ie.read_model(model=model_path)
            except Exception as e:
                return False, f"Error loading OpenVINO model: {e}"
                
            return True, None
            
        except Exception as e:
            return False, f"Error verifying OpenVINO model: {e}"
            
    @staticmethod
    def verify_webnn_model(model_path: str) -> Tuple[bool, Optional[str]]:
        """
        Verify WebNN model (JavaScript module).
        
        Args:
            model_path: Path to WebNN model (JavaScript)
            
        Returns:
            Tuple of (valid, error_message)
        """
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"
                
            # Check if file is a JavaScript file
            if not model_path.endswith('.js'):
                return False, f"Model file is not a JavaScript file: {model_path}"
                
            # Check file content
            with open(model_path, 'r') as f:
                content = f.read()
                
            # Check for WebNN-related code
            if 'navigator.ml' not in content or 'loadModel' not in content:
                return False, "File does not appear to be a WebNN model"
                
            return True, None
            
        except Exception as e:
            return False, f"Error verifying WebNN model: {e}"
            
    @staticmethod
    def verify_webgpu_model(model_path: str) -> Tuple[bool, Optional[str]]:
        """
        Verify WebGPU model (JavaScript module).
        
        Args:
            model_path: Path to WebGPU model (JavaScript)
            
        Returns:
            Tuple of (valid, error_message)
        """
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"
                
            # Check if file is a JavaScript file
            if not model_path.endswith('.js'):
                return False, f"Model file is not a JavaScript file: {model_path}"
                
            # Check file content
            with open(model_path, 'r') as f:
                content = f.read()
                
            # Check for WebGPU-related code
            if 'navigator.gpu' not in content or 'GPUBuffer' not in content:
                return False, "File does not appear to be a WebGPU model"
                
            return True, None
            
        except Exception as e:
            return False, f"Error verifying WebGPU model: {e}"
            
    @staticmethod
    def verify_model(model_path: str, model_format: str) -> Tuple[bool, Optional[str]]:
        """
        Verify model based on format.
        
        Args:
            model_path: Path to model
            model_format: Model format
            
        Returns:
            Tuple of (valid, error_message)
        """
        verification_methods = {
            'onnx': ModelVerifier.verify_onnx_model,
            'pytorch': ModelVerifier.verify_pytorch_model,
            'openvino': ModelVerifier.verify_openvino_model,
            'webnn': ModelVerifier.verify_webnn_model,
            'webgpu': ModelVerifier.verify_webgpu_model
        }
        
        if model_format in verification_methods:
            return verification_methods[model_format](model_path)
        else:
            # For unsupported formats, just check if file exists
            if os.path.exists(model_path):
                return True, None
            else:
                return False, f"Model file not found: {model_path}"
                
    @staticmethod
    def verify_conversion(source_path: str, target_path: str, 
                        source_format: str, target_format: str) -> Tuple[bool, Optional[str]]:
        """
        Verify model conversion.
        
        Args:
            source_path: Path to source model
            target_path: Path to converted model
            source_format: Source model format
            target_format: Target model format
            
        Returns:
            Tuple of (valid, error_message)
        """
        # Verify source model
        source_valid, source_error = ModelVerifier.verify_model(source_path, source_format)
        if not source_valid:
            return False, f"Source model is invalid: {source_error}"
            
        # Verify target model
        target_valid, target_error = ModelVerifier.verify_model(target_path, target_format)
        if not target_valid:
            return False, f"Target model is invalid: {target_error}"
            
        return True, None