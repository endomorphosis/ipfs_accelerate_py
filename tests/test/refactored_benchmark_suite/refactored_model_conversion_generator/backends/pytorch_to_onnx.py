"""
PyTorch to ONNX Converter

This module provides a converter for PyTorch models to ONNX format.
"""

import os
import logging
import tempfile
from typing import Dict, Any, Optional, List, Tuple
import json

from ..core.converter import ModelConverter, ConversionResult
from ..core.registry import register_converter

logger = logging.getLogger(__name__)

@register_converter(source_format='pytorch', target_format='onnx')
class PyTorchToOnnxConverter(ModelConverter):
    """
    Converter for PyTorch models to ONNX format.
    """
    
    def _get_source_format(self) -> str:
        """Get source format."""
        return 'pytorch'
        
    def _get_target_format(self) -> str:
        """Get target format."""
        return 'onnx'
        
    def _get_supported_model_types(self) -> List[str]:
        """Get supported model types."""
        return [
            'bert', 'gpt2', 'distilbert', 'roberta', 't5', 'vit', 'clip',
            'resnet', 'densenet', 'efficientnet', 'mobilenet', 'whisper'
        ]
        
    def _execute_conversion(self, model_path: str, output_path: str, 
                          model_type: Optional[str] = None, **kwargs) -> ConversionResult:
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            model_path: Path to the PyTorch model (.pt, .pth, .bin)
            output_path: Path to save the ONNX model
            model_type: Type of model (e.g., 'bert', 'vit')
            **kwargs: Additional conversion parameters
                - opset_version: ONNX opset version to use
                - input_shapes: Dictionary of input names to shapes
                - dynamic_axes: Dictionary of input/output names to dynamic axes
                - optimize: Whether to optimize the ONNX model
                
        Returns:
            ConversionResult with conversion details
        """
        try:
            # Import dependencies here to avoid requiring them for module import
            import torch
            import onnx
            
            # Get conversion options
            opset_version = kwargs.get('opset_version', 14)
            input_shapes = kwargs.get('input_shapes', None)
            dynamic_axes = kwargs.get('dynamic_axes', None)
            optimize = kwargs.get('optimize', True)
            
            # Load PyTorch model
            self.logger.info(f"Loading PyTorch model from {model_path}")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Get model loading function based on model type
            if model_type and model_type.lower() in ['bert', 'gpt2', 'distilbert', 'roberta', 't5']:
                # Handle transformer models
                return self._convert_transformers_model(
                    model_path, output_path, model_type, device, 
                    opset_version, input_shapes, dynamic_axes, optimize
                )
            
            elif model_type and model_type.lower() in ['vit', 'clip', 'resnet', 'densenet', 'efficientnet', 'mobilenet']:
                # Handle vision models
                return self._convert_vision_model(
                    model_path, output_path, model_type, device, 
                    opset_version, input_shapes, dynamic_axes, optimize
                )
                
            else:
                # Handle generic PyTorch model
                self.logger.warning(f"No specific handling for model type '{model_type}', using generic conversion")
                
                # Try to load with torch.load
                try:
                    model = torch.load(model_path, map_location=device)
                    if isinstance(model, dict) and 'model' in model:
                        model = model['model']
                except Exception as e:
                    self.logger.error(f"Error loading PyTorch model: {e}")
                    return ConversionResult(
                        success=False,
                        error=f"Error loading PyTorch model: {e}"
                    )
                    
                # Set model to evaluation mode
                if hasattr(model, 'eval'):
                    model.eval()
                    
                # Create dummy input if input_shapes not provided
                if not input_shapes:
                    self.logger.warning("No input shapes provided, using default dummy input")
                    dummy_input = torch.randn(1, 3, 224, 224, device=device)
                else:
                    # Create dummy inputs from input_shapes
                    dummy_inputs = {}
                    for name, shape in input_shapes.items():
                        dummy_inputs[name] = torch.randn(*shape, device=device)
                        
                    # If only one input, use it directly
                    if len(dummy_inputs) == 1:
                        dummy_input = list(dummy_inputs.values())[0]
                    else:
                        dummy_input = tuple(dummy_inputs.values())
                
                # Export to ONNX
                self.logger.info(f"Exporting PyTorch model to ONNX: {output_path}")
                torch.onnx.export(
                    model,
                    dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=list(input_shapes.keys()) if input_shapes else ['input'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes
                )
                
                # Verify and optimize ONNX model
                model_onnx = onnx.load(output_path)
                onnx.checker.check_model(model_onnx)
                
                if optimize:
                    self.logger.info("Optimizing ONNX model")
                    import onnxoptimizer
                    model_onnx = onnxoptimizer.optimize(model_onnx)
                    onnx.save(model_onnx, output_path)
                
                # Save model metadata
                metadata = {
                    'opset_version': opset_version,
                    'input_shapes': input_shapes,
                    'dynamic_axes': dynamic_axes,
                    'optimized': optimize,
                    'source_format': self.source_format,
                    'target_format': self.target_format,
                    'model_type': model_type
                }
                
                # Save metadata alongside model
                metadata_path = output_path + '.json'
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                return ConversionResult(
                    success=True,
                    output_path=output_path,
                    format=self.target_format,
                    metadata=metadata
                )
            
        except Exception as e:
            self.logger.error(f"Error converting PyTorch model to ONNX: {e}", exc_info=True)
            return ConversionResult(
                success=False,
                error=f"Error converting PyTorch model to ONNX: {e}"
            )
            
    def _convert_transformers_model(self, model_path: str, output_path: str, 
                                  model_type: str, device: str, opset_version: int,
                                  input_shapes: Optional[Dict[str, List[int]]] = None,
                                  dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                                  optimize: bool = True) -> ConversionResult:
        """
        Convert a transformers model to ONNX.
        
        Args:
            model_path: Path to the transformers model
            output_path: Path to save the ONNX model
            model_type: Transformer model type
            device: Device to use for conversion
            opset_version: ONNX opset version
            input_shapes: Dictionary of input names to shapes
            dynamic_axes: Dictionary of input/output names to dynamic axes
            optimize: Whether to optimize the ONNX model
            
        Returns:
            ConversionResult with conversion details
        """
        try:
            # Import transformers library
            from transformers import AutoModel, AutoConfig, AutoTokenizer
            import torch
            import onnx
            
            self.logger.info(f"Loading {model_type.upper()} model from {model_path}")
            
            # Load model configuration
            config = AutoConfig.from_pretrained(model_path)
            
            # Load model
            model = AutoModel.from_pretrained(model_path, config=config)
            model.eval()
            model.to(device)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Create sample input text
            text = "Hello, this is a sample input for ONNX conversion. The quick brown fox jumps over the lazy dog."
            
            # Tokenize input
            tokens = tokenizer(text, return_tensors="pt").to(device)
            
            # Set up dynamic axes for the ONNX model
            if dynamic_axes is None:
                dynamic_axes = {
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                    'token_type_ids': {0: 'batch_size', 1: 'sequence_length'} if 'token_type_ids' in tokens else None,
                    'output': {0: 'batch_size'}
                }
                # Remove None values
                dynamic_axes = {k: v for k, v in dynamic_axes.items() if v is not None}
            
            # Export to ONNX
            self.logger.info(f"Exporting {model_type.upper()} model to ONNX: {output_path}")
            
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    tuple(tokens.values()),
                    output_path,
                    input_names=list(tokens.keys()),
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    do_constant_folding=True
                )
            
            # Verify and optimize ONNX model
            model_onnx = onnx.load(output_path)
            onnx.checker.check_model(model_onnx)
            
            if optimize:
                self.logger.info("Optimizing ONNX model")
                import onnxoptimizer
                model_onnx = onnxoptimizer.optimize(model_onnx)
                onnx.save(model_onnx, output_path)
            
            # Save model metadata
            metadata = {
                'model_type': model_type,
                'opset_version': opset_version,
                'input_shapes': {k: list(v.shape) for k, v in tokens.items()},
                'dynamic_axes': dynamic_axes,
                'optimized': optimize,
                'source_format': self.source_format,
                'target_format': self.target_format
            }
            
            # Save metadata alongside model
            metadata_path = output_path + '.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return ConversionResult(
                success=True,
                output_path=output_path,
                format=self.target_format,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error converting transformer model to ONNX: {e}", exc_info=True)
            return ConversionResult(
                success=False,
                error=f"Error converting transformer model to ONNX: {e}"
            )
            
    def _convert_vision_model(self, model_path: str, output_path: str, 
                            model_type: str, device: str, opset_version: int,
                            input_shapes: Optional[Dict[str, List[int]]] = None,
                            dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                            optimize: bool = True) -> ConversionResult:
        """
        Convert a vision model to ONNX.
        
        Args:
            model_path: Path to the vision model
            output_path: Path to save the ONNX model
            model_type: Vision model type
            device: Device to use for conversion
            opset_version: ONNX opset version
            input_shapes: Dictionary of input names to shapes
            dynamic_axes: Dictionary of input/output names to dynamic axes
            optimize: Whether to optimize the ONNX model
            
        Returns:
            ConversionResult with conversion details
        """
        try:
            import torch
            import onnx
            
            self.logger.info(f"Loading {model_type.upper()} model from {model_path}")
            
            # Load appropriate vision model based on type
            if model_type.lower() == 'vit':
                from transformers import ViTModel, ViTConfig
                config = ViTConfig.from_pretrained(model_path)
                model = ViTModel.from_pretrained(model_path, config=config)
                
                # Default input shape for ViT (batch_size, channels, height, width)
                if input_shapes is None:
                    input_shapes = {'pixel_values': [1, 3, 224, 224]}
                    
            elif model_type.lower() == 'clip':
                from transformers import CLIPModel, CLIPConfig
                config = CLIPConfig.from_pretrained(model_path)
                model = CLIPModel.from_pretrained(model_path, config=config)
                
                # Default input shape for CLIP (batch_size, channels, height, width)
                if input_shapes is None:
                    input_shapes = {
                        'input_ids': [1, 77],
                        'pixel_values': [1, 3, 224, 224],
                        'attention_mask': [1, 77]
                    }
                    
            elif model_type.lower() in ['resnet', 'densenet', 'efficientnet', 'mobilenet']:
                # Try to load with torchvision
                try:
                    import torchvision.models as tv_models
                    model_fn = getattr(tv_models, model_type.lower() + '50', None)
                    if model_fn is None:
                        model_fn = getattr(tv_models, model_type.lower() + '18', None)
                        
                    if model_fn is not None:
                        model = model_fn(pretrained=True)
                    else:
                        # Fallback to torch.load
                        model = torch.load(model_path, map_location=device)
                except Exception as e:
                    self.logger.warning(f"Error loading with torchvision, falling back to torch.load: {e}")
                    model = torch.load(model_path, map_location=device)
                    
                # Default input shape for vision models (batch_size, channels, height, width)
                if input_shapes is None:
                    input_shapes = {'input': [1, 3, 224, 224]}
                    
            else:
                # Generic vision model
                model = torch.load(model_path, map_location=device)
                
                # Default input shape (batch_size, channels, height, width)
                if input_shapes is None:
                    input_shapes = {'input': [1, 3, 224, 224]}
            
            # Set model to evaluation mode
            model.eval()
            model.to(device)
            
            # Create dummy inputs
            dummy_inputs = {}
            for name, shape in input_shapes.items():
                dummy_inputs[name] = torch.randn(*shape, device=device)
                
            # If only one input, use it directly
            if len(dummy_inputs) == 1:
                dummy_input = list(dummy_inputs.values())[0]
            else:
                dummy_input = tuple(dummy_inputs.values())
            
            # Set up dynamic axes for the ONNX model
            if dynamic_axes is None:
                # Default dynamic axes for vision models
                dynamic_axes = {
                    'input': {0: 'batch_size'} if 'input' in input_shapes else None,
                    'pixel_values': {0: 'batch_size'} if 'pixel_values' in input_shapes else None,
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'} if 'input_ids' in input_shapes else None,
                    'attention_mask': {0: 'batch_size', 1: 'sequence_length'} if 'attention_mask' in input_shapes else None,
                    'output': {0: 'batch_size'}
                }
                # Remove None values
                dynamic_axes = {k: v for k, v in dynamic_axes.items() if v is not None}
            
            # Export to ONNX
            self.logger.info(f"Exporting {model_type.upper()} model to ONNX: {output_path}")
            
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    output_path,
                    input_names=list(input_shapes.keys()),
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    do_constant_folding=True
                )
            
            # Verify and optimize ONNX model
            model_onnx = onnx.load(output_path)
            onnx.checker.check_model(model_onnx)
            
            if optimize:
                self.logger.info("Optimizing ONNX model")
                import onnxoptimizer
                model_onnx = onnxoptimizer.optimize(model_onnx)
                onnx.save(model_onnx, output_path)
            
            # Save model metadata
            metadata = {
                'model_type': model_type,
                'opset_version': opset_version,
                'input_shapes': input_shapes,
                'dynamic_axes': dynamic_axes,
                'optimized': optimize,
                'source_format': self.source_format,
                'target_format': self.target_format
            }
            
            # Save metadata alongside model
            metadata_path = output_path + '.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return ConversionResult(
                success=True,
                output_path=output_path,
                format=self.target_format,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error converting vision model to ONNX: {e}", exc_info=True)
            return ConversionResult(
                success=False,
                error=f"Error converting vision model to ONNX: {e}"
            )
            
    def validate_model(self, model_path: str, model_type: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate a PyTorch model before conversion.
        
        Args:
            model_path: Path to the PyTorch model
            model_type: Type of model (e.g., 'bert', 'vit')
            
        Returns:
            Tuple of (valid, error_message)
        """
        try:
            import torch
            
            # Check if file exists
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"
                
            # Try to load model
            try:
                device = 'cpu'  # Use CPU for validation
                
                # Handle transformers models
                if model_type and model_type.lower() in ['bert', 'gpt2', 'distilbert', 'roberta', 't5']:
                    from transformers import AutoConfig
                    # Just try to load config to validate
                    _ = AutoConfig.from_pretrained(model_path)
                else:
                    # Try to load with torch.load
                    _ = torch.load(model_path, map_location=device)
                    
                return True, None
                
            except Exception as e:
                return False, f"Error loading PyTorch model: {e}"
                
        except ImportError as e:
            return False, f"Missing required dependencies: {e}"