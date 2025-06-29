"""
ONNX to OpenVINO Converter

This module provides a converter for ONNX models to OpenVINO IR format.
"""

import os
import logging
import tempfile
import subprocess
import platform
import json
from typing import Dict, Any, Optional, List, Tuple

from ..core.converter import ModelConverter, ConversionResult
from ..core.registry import register_converter

logger = logging.getLogger(__name__)

@register_converter(source_format='onnx', target_format='openvino')
class OnnxToOpenvinoConverter(ModelConverter):
    """
    Converter for ONNX models to OpenVINO IR format.
    """
    
    def _get_source_format(self) -> str:
        """Get source format."""
        return 'onnx'
        
    def _get_target_format(self) -> str:
        """Get target format."""
        return 'openvino'
        
    def _get_supported_model_types(self) -> List[str]:
        """Get supported model types."""
        return [
            'bert', 'gpt2', 'distilbert', 'roberta', 't5', 'vit', 'clip',
            'resnet', 'densenet', 'efficientnet', 'mobilenet', 'whisper'
        ]
        
    def _execute_conversion(self, model_path: str, output_path: str, 
                          model_type: Optional[str] = None, **kwargs) -> ConversionResult:
        """
        Convert ONNX model to OpenVINO IR format.
        
        Args:
            model_path: Path to the ONNX model
            output_path: Path to save the OpenVINO IR model (XML)
            model_type: Type of model (e.g., 'bert', 'vit')
            **kwargs: Additional conversion parameters
                - data_type: Data type for OpenVINO IR ('FP32', 'FP16', 'INT8')
                - input_shapes: Dictionary of input names to shapes
                - use_mo_api: Whether to use Model Optimizer API (if False, uses command line)
                
        Returns:
            ConversionResult with conversion details
        """
        try:
            # Get conversion options
            data_type = kwargs.get('data_type', 'FP32')
            input_shapes = kwargs.get('input_shapes', None)
            use_mo_api = kwargs.get('use_mo_api', True)
            
            # Ensure output path ends with .xml
            if not output_path.endswith('.xml'):
                output_dir = os.path.dirname(output_path)
                output_name = os.path.basename(output_path)
                output_path = os.path.join(output_dir, output_name + '.xml')
                
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load model metadata if available
            metadata = {}
            metadata_path = model_path + '.json'
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Error loading model metadata: {e}")
            
            # Use input shapes from metadata if not provided
            if input_shapes is None and 'input_shapes' in metadata:
                input_shapes = metadata.get('input_shapes')
                
            # Convert using Model Optimizer API or command line
            if use_mo_api:
                return self._convert_with_mo_api(
                    model_path, output_path, model_type, data_type, input_shapes, metadata
                )
            else:
                return self._convert_with_mo_cli(
                    model_path, output_path, model_type, data_type, input_shapes, metadata
                )
                
        except Exception as e:
            self.logger.error(f"Error converting ONNX model to OpenVINO: {e}", exc_info=True)
            return ConversionResult(
                success=False,
                error=f"Error converting ONNX model to OpenVINO: {e}"
            )
    
    def _convert_with_mo_api(self, model_path: str, output_path: str, 
                           model_type: Optional[str], data_type: str,
                           input_shapes: Optional[Dict[str, List[int]]], 
                           metadata: Dict[str, Any]) -> ConversionResult:
        """
        Convert ONNX model to OpenVINO IR using Model Optimizer API.
        
        Args:
            model_path: Path to the ONNX model
            output_path: Path to save the OpenVINO IR model (XML)
            model_type: Type of model (e.g., 'bert', 'vit')
            data_type: Data type for OpenVINO IR ('FP32', 'FP16', 'INT8')
            input_shapes: Dictionary of input names to shapes
            metadata: Model metadata
            
        Returns:
            ConversionResult with conversion details
        """
        try:
            # Import OpenVINO dependencies
            try:
                # For newer versions of OpenVINO
                from openvino.tools.mo import convert_model
                from openvino.runtime import Core
                new_api = True
            except ImportError:
                # For older versions of OpenVINO
                try:
                    from openvino.inference_engine import IECore
                    from mo.main import main as mo_main
                    new_api = False
                except ImportError:
                    return ConversionResult(
                        success=False,
                        error="OpenVINO Model Optimizer not found. Please install OpenVINO Runtime."
                    )
            
            # Log conversion details
            self.logger.info(f"Converting ONNX model to OpenVINO IR: {model_path} -> {output_path}")
            self.logger.info(f"Data type: {data_type}, Model type: {model_type}")
            
            # Prepare input shapes argument
            input_shape_str = None
            if input_shapes:
                # Format: "input1[1,3,224,224],input2[1,3]"
                shape_parts = []
                for name, shape in input_shapes.items():
                    shape_str = f"{name}[{','.join(map(str, shape))}]"
                    shape_parts.append(shape_str)
                input_shape_str = ','.join(shape_parts)
                self.logger.info(f"Input shapes: {input_shape_str}")
            
            # Convert model based on API version
            if new_api:
                # New API (OpenVINO 2022.1+)
                self.logger.info("Using new OpenVINO API for conversion")
                
                # Prepare conversion options
                conversion_args = {
                    'input_model': model_path,
                    'model_name': os.path.splitext(os.path.basename(output_path))[0],
                    'compress_to_fp16': data_type == 'FP16',
                }
                
                # Add input shapes if provided
                if input_shapes:
                    conversion_args['input'] = input_shape_str
                
                # Convert model
                ov_model = convert_model(**conversion_args)
                
                # Save model to output path
                output_dir = os.path.dirname(output_path)
                model_name = os.path.splitext(os.path.basename(output_path))[0]
                from openvino.runtime import serialize
                serialize(ov_model, os.path.join(output_dir, model_name + ".xml"))
                
                # Get model inputs and outputs
                input_info = {name: str(node.shape) for name, node in ov_model.inputs.items()}
                output_info = {name: str(node.shape) for name, node in ov_model.outputs.items()}
                
                # Save model metadata
                conversion_metadata = {
                    'data_type': data_type,
                    'input_shapes': input_shapes,
                    'source_format': self.source_format,
                    'target_format': self.target_format,
                    'model_type': model_type,
                    'input_info': input_info,
                    'output_info': output_info,
                    'original_metadata': metadata
                }
                
                # Save metadata alongside model
                metadata_path = output_path + '.json'
                with open(metadata_path, 'w') as f:
                    json.dump(conversion_metadata, f, indent=2)
                
                return ConversionResult(
                    success=True,
                    output_path=output_path,
                    format=self.target_format,
                    metadata=conversion_metadata
                )
                
            else:
                # Old API (OpenVINO 2021.4 and earlier)
                self.logger.info("Using legacy OpenVINO API for conversion")
                
                # Create temporary directory for conversion
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Prepare arguments for Model Optimizer
                    mo_args = [
                        '--input_model', model_path,
                        '--output_dir', os.path.dirname(output_path),
                        '--model_name', os.path.splitext(os.path.basename(output_path))[0]
                    ]
                    
                    # Add data type
                    if data_type == 'FP16':
                        mo_args.extend(['--compress_to_fp16'])
                    
                    # Add input shapes if provided
                    if input_shape_str:
                        mo_args.extend(['--input', input_shape_str])
                    
                    # Convert model with Model Optimizer
                    self.logger.info(f"Running Model Optimizer with args: {mo_args}")
                    exitcode = mo_main(mo_args)
                    
                    if exitcode != 0:
                        return ConversionResult(
                            success=False,
                            error=f"Model Optimizer failed with exit code {exitcode}"
                        )
                    
                    # Check if output files exist
                    if not os.path.exists(output_path):
                        return ConversionResult(
                            success=False,
                            error=f"Output file not found: {output_path}"
                        )
                    
                    # Save model metadata
                    conversion_metadata = {
                        'data_type': data_type,
                        'input_shapes': input_shapes,
                        'source_format': self.source_format,
                        'target_format': self.target_format,
                        'model_type': model_type,
                        'original_metadata': metadata
                    }
                    
                    # Try to get model inputs and outputs
                    try:
                        ie = IECore()
                        network = ie.read_network(model=output_path)
                        input_info = {name: str(info.shape) for name, info in network.input_info.items()}
                        output_info = {name: str(data.shape) for name, data in network.outputs.items()}
                        conversion_metadata['input_info'] = input_info
                        conversion_metadata['output_info'] = output_info
                    except Exception as e:
                        self.logger.warning(f"Error getting model info: {e}")
                    
                    # Save metadata alongside model
                    metadata_path = output_path + '.json'
                    with open(metadata_path, 'w') as f:
                        json.dump(conversion_metadata, f, indent=2)
                    
                    return ConversionResult(
                        success=True,
                        output_path=output_path,
                        format=self.target_format,
                        metadata=conversion_metadata
                    )
                    
        except Exception as e:
            self.logger.error(f"Error in Model Optimizer API conversion: {e}", exc_info=True)
            return ConversionResult(
                success=False,
                error=f"Error in Model Optimizer API conversion: {e}"
            )
    
    def _convert_with_mo_cli(self, model_path: str, output_path: str, 
                           model_type: Optional[str], data_type: str,
                           input_shapes: Optional[Dict[str, List[int]]], 
                           metadata: Dict[str, Any]) -> ConversionResult:
        """
        Convert ONNX model to OpenVINO IR using Model Optimizer command line.
        
        Args:
            model_path: Path to the ONNX model
            output_path: Path to save the OpenVINO IR model (XML)
            model_type: Type of model (e.g., 'bert', 'vit')
            data_type: Data type for OpenVINO IR ('FP32', 'FP16', 'INT8')
            input_shapes: Dictionary of input names to shapes
            metadata: Model metadata
            
        Returns:
            ConversionResult with conversion details
        """
        try:
            # Find mo executable
            mo_script = 'mo.py'  # Default script name
            # On Windows, look for the .exe or bat file
            if platform.system() == 'Windows':
                mo_script = 'mo.exe'
            
            # Try to find mo in PATH
            mo_path = None
            for path in os.environ['PATH'].split(os.pathsep):
                candidate = os.path.join(path, mo_script)
                if os.path.exists(candidate):
                    mo_path = candidate
                    break
                    
            # If not found, look in standard OpenVINO installation directories
            if mo_path is None:
                # Common OpenVINO installation directories
                openvino_dirs = [
                    os.path.expanduser('~/intel/openvino'),
                    '/opt/intel/openvino',
                    'C:\\Program Files (x86)\\Intel\\openvino',
                    os.path.expanduser('~/.local/lib/python*/site-packages/openvino/tools')
                ]
                
                for openvino_dir in openvino_dirs:
                    # Use glob to handle wildcards in path
                    import glob
                    for mo_candidate in glob.glob(os.path.join(openvino_dir, '**', mo_script), recursive=True):
                        if os.path.exists(mo_candidate):
                            mo_path = mo_candidate
                            break
                            
                    if mo_path:
                        break
            
            if mo_path is None:
                return ConversionResult(
                    success=False,
                    error="Model Optimizer not found. Please install OpenVINO Runtime and ensure mo script is in PATH."
                )
                
            # Log conversion details
            self.logger.info(f"Converting ONNX model to OpenVINO IR: {model_path} -> {output_path}")
            self.logger.info(f"Data type: {data_type}, Model type: {model_type}")
            
            # Prepare input shapes argument
            input_shape_str = None
            if input_shapes:
                # Format: "input1[1,3,224,224],input2[1,3]"
                shape_parts = []
                for name, shape in input_shapes.items():
                    shape_str = f"{name}[{','.join(map(str, shape))}]"
                    shape_parts.append(shape_str)
                input_shape_str = ','.join(shape_parts)
                self.logger.info(f"Input shapes: {input_shape_str}")
            
            # Prepare command line arguments
            cmd = [
                sys.executable if mo_path.endswith('.py') else mo_path,
                '--input_model', model_path,
                '--output_dir', os.path.dirname(output_path),
                '--model_name', os.path.splitext(os.path.basename(output_path))[0]
            ]
            
            # Add data type
            if data_type == 'FP16':
                cmd.extend(['--compress_to_fp16'])
            
            # Add input shapes if provided
            if input_shape_str:
                cmd.extend(['--input', input_shape_str])
            
            # Log command
            self.logger.info(f"Running: {' '.join(cmd)}")
            
            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check for errors
            if result.returncode != 0:
                self.logger.error(f"Model Optimizer failed: {result.stderr}")
                return ConversionResult(
                    success=False,
                    error=f"Model Optimizer failed: {result.stderr}"
                )
                
            # Check if output file exists
            if not os.path.exists(output_path):
                self.logger.error(f"Output file not found: {output_path}")
                return ConversionResult(
                    success=False,
                    error=f"Output file not found: {output_path}"
                )
                
            # Save model metadata
            conversion_metadata = {
                'data_type': data_type,
                'input_shapes': input_shapes,
                'source_format': self.source_format,
                'target_format': self.target_format,
                'model_type': model_type,
                'original_metadata': metadata
            }
            
            # Save metadata alongside model
            metadata_path = output_path + '.json'
            with open(metadata_path, 'w') as f:
                json.dump(conversion_metadata, f, indent=2)
            
            return ConversionResult(
                success=True,
                output_path=output_path,
                format=self.target_format,
                metadata=conversion_metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in Model Optimizer CLI conversion: {e}", exc_info=True)
            return ConversionResult(
                success=False,
                error=f"Error in Model Optimizer CLI conversion: {e}"
            )
            
    def validate_model(self, model_path: str, model_type: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate an ONNX model before conversion to OpenVINO.
        
        Args:
            model_path: Path to the ONNX model
            model_type: Type of model (e.g., 'bert', 'vit')
            
        Returns:
            Tuple of (valid, error_message)
        """
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"
                
            # Try to load ONNX model
            try:
                import onnx
                model = onnx.load(model_path)
                onnx.checker.check_model(model)
                return True, None
            except Exception as e:
                return False, f"Error validating ONNX model: {e}"
                
        except ImportError as e:
            return False, f"Missing required dependencies: {e}"