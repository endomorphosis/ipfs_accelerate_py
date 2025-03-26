"""
ONNX to WebNN Converter

This module provides a converter for ONNX models to WebNN format.
"""

import os
import logging
import json
import tempfile
import subprocess
import platform
from typing import Dict, Any, Optional, List, Tuple

from ..core.converter import ModelConverter, ConversionResult
from ..core.registry import register_converter

logger = logging.getLogger(__name__)

@register_converter(source_format='onnx', target_format='webnn')
class OnnxToWebNNConverter(ModelConverter):
    """
    Converter for ONNX models to WebNN format.
    """
    
    def _get_source_format(self) -> str:
        """Get source format."""
        return 'onnx'
        
    def _get_target_format(self) -> str:
        """Get target format."""
        return 'webnn'
        
    def _get_supported_model_types(self) -> List[str]:
        """Get supported model types."""
        return [
            'bert', 'vit', 'resnet', 'mobilenet', 'efficientnet', 'whisper'
        ]
        
    def _execute_conversion(self, model_path: str, output_path: str, 
                          model_type: Optional[str] = None, **kwargs) -> ConversionResult:
        """
        Convert ONNX model to WebNN format.
        
        Args:
            model_path: Path to the ONNX model
            output_path: Path to save the WebNN model
            model_type: Type of model (e.g., 'bert', 'vit')
            **kwargs: Additional conversion parameters
                - precision: Model precision ('default', 'float16', 'int8')
                - layout: Model layout ('default', 'nchw', 'nhwc')
                - optimization_level: Optimization level (0-3)
                - supported_ops: List of supported WebNN operations
                - browser_targets: List of target browsers ('chrome', 'firefox', 'safari', 'edge')
                
        Returns:
            ConversionResult with conversion details
        """
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get conversion options
            precision = kwargs.get('precision', 'default')
            layout = kwargs.get('layout', 'default')
            optimization_level = kwargs.get('optimization_level', 2)
            supported_ops = kwargs.get('supported_ops', None)
            browser_targets = kwargs.get('browser_targets', ['chrome', 'firefox', 'safari', 'edge'])
            
            # Load model metadata if available
            metadata = {}
            metadata_path = model_path + '.json'
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Error loading model metadata: {e}")
            
            # Create WebNN model
            return self._create_webnn_model(
                model_path, output_path, model_type, precision, layout,
                optimization_level, supported_ops, browser_targets, metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error converting ONNX model to WebNN: {e}", exc_info=True)
            return ConversionResult(
                success=False,
                error=f"Error converting ONNX model to WebNN: {e}"
            )
    
    def _create_webnn_model(self, model_path: str, output_path: str, model_type: Optional[str],
                         precision: str, layout: str, optimization_level: int,
                         supported_ops: Optional[List[str]], browser_targets: List[str],
                         original_metadata: Dict[str, Any]) -> ConversionResult:
        """
        Create WebNN model from ONNX model.
        
        Args:
            model_path: Path to the ONNX model
            output_path: Path to save the WebNN model
            model_type: Type of model (e.g., 'bert', 'vit')
            precision: Model precision ('default', 'float16', 'int8')
            layout: Model layout ('default', 'nchw', 'nhwc')
            optimization_level: Optimization level (0-3)
            supported_ops: List of supported WebNN operations
            browser_targets: List of target browsers
            original_metadata: Original model metadata
            
        Returns:
            ConversionResult with conversion details
        """
        try:
            # Check if ONNX model exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX model not found: {model_path}")
            
            # Check if we can import onnx
            try:
                import onnx
                import numpy as np
            except ImportError:
                raise ImportError("ONNX and NumPy libraries are required for WebNN conversion")
                
            # Load ONNX model
            self.logger.info(f"Loading ONNX model: {model_path}")
            model = onnx.load(model_path)
            
            # Validate model
            onnx.checker.check_model(model)
            
            # Create a JavaScript module that loads the model in WebNN
            js_code = self._generate_webnn_js_module(
                model_path, model, model_type, precision, layout, 
                optimization_level, supported_ops, browser_targets
            )
            
            # Save the JavaScript module
            with open(output_path, 'w') as f:
                f.write(js_code)
                
            # Save original ONNX model alongside WebNN model
            onnx_output_path = os.path.join(os.path.dirname(output_path), 
                                       os.path.basename(model_path))
            if model_path != onnx_output_path:
                import shutil
                shutil.copy(model_path, onnx_output_path)
                
            # Save JSON metadata for the model
            conversion_metadata = {
                'source_format': self.source_format,
                'target_format': self.target_format,
                'model_type': model_type,
                'precision': precision,
                'layout': layout,
                'optimization_level': optimization_level,
                'supported_ops': supported_ops,
                'browser_targets': browser_targets,
                'inputs': self._get_model_inputs(model),
                'outputs': self._get_model_outputs(model),
                'onnx_path': onnx_output_path,
                'original_metadata': original_metadata
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
            self.logger.error(f"Error creating WebNN model: {e}", exc_info=True)
            return ConversionResult(
                success=False,
                error=f"Error creating WebNN model: {e}"
            )
    
    def _generate_webnn_js_module(self, model_path: str, onnx_model, model_type: Optional[str],
                                precision: str, layout: str, optimization_level: int,
                                supported_ops: Optional[List[str]], 
                                browser_targets: List[str]) -> str:
        """
        Generate JavaScript module that loads the model in WebNN.
        
        Args:
            model_path: Path to the ONNX model
            onnx_model: Loaded ONNX model
            model_type: Type of model (e.g., 'bert', 'vit')
            precision: Model precision ('default', 'float16', 'int8')
            layout: Model layout ('default', 'nchw', 'nhwc')
            optimization_level: Optimization level (0-3)
            supported_ops: List of supported WebNN operations
            browser_targets: List of target browsers
            
        Returns:
            JavaScript module code
        """
        # Get model inputs and outputs
        inputs = self._get_model_inputs(onnx_model)
        outputs = self._get_model_outputs(onnx_model)
        
        # Create a JavaScript model loader for WebNN
        js_template = """
/**
 * WebNN model loader for {model_name}
 * Converted from ONNX format
 * Model type: {model_type}
 * Generated by IPFS Accelerate Model Conversion Generator
 */

// Model configuration
const modelConfig = {
  name: '{model_name}',
  type: '{model_type}',
  precision: '{precision}',
  layout: '{layout}',
  optimizationLevel: {optimization_level},
  browserTargets: {browser_targets},
  inputs: {inputs},
  outputs: {outputs}
};

/**
 * Load the model using WebNN API
 * @param {Object} options - Loading options
 * @param {string} options.modelPath - Path to the ONNX model file
 * @param {Object} options.context - WebNN context
 * @param {string} options.device - Device to use ('cpu', 'gpu')
 * @returns {Promise<Object>} - Loaded model
 */
export async function loadModel(options = {}) {
  const {{
    modelPath = '{model_path_basename}',
    context = null,
    device = 'gpu'
  }} = options;

  // Check if WebNN is supported
  if (!('ml' in navigator)) {
    throw new Error('WebNN is not supported in this browser');
  }

  // Get WebNN context if not provided
  const nnContext = context || (device === 'gpu' 
    ? navigator.ml.gpu() 
    : navigator.ml.cpu());

  try {
    console.log(`Loading model from ${{modelPath}} using WebNN on ${{device}}`);
    
    // Fetch the ONNX model
    const response = await fetch(modelPath);
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${{response.statusText}}`);
    }
    
    const modelBuffer = await response.arrayBuffer();
    
    // Create WebNN graph builder
    const builder = new MLGraphBuilder(nnContext);
    
    // Import ONNX model
    const model = await builder.importModel(modelBuffer);
    
    // Create inputs and outputs based on model
    const inputs = {inputs_creation};
    
    // Compile the WebNN graph
    const compiledModel = await builder.build({{
      inputs,
      {output_entry}
    }});
    
    return {
      model: compiledModel,
      config: modelConfig,
      async run(inputData) {
        // Create input tensors
        const nnInputs = {};
        for (const [name, info] of Object.entries(modelConfig.inputs)) {
          if (!inputData[name]) {
            throw new Error(`Input "${{name}}" is required but not provided`);
          }
          
          // Convert input data to appropriate format
          const inputArray = this._prepareInputData(inputData[name], info);
          nnInputs[name] = new MLTensor(inputArray, info.shape);
        }
        
        // Run inference
        const outputs = await this.model.compute(nnInputs);
        
        // Process outputs
        const result = {};
        for (const [name, output] of Object.entries(outputs)) {
          result[name] = {
            data: output.data,
            shape: output.shape
          };
        }
        
        return result;
      },
      
      _prepareInputData(data, info) {
        // Convert input data to the appropriate typed array
        if (data instanceof ArrayBuffer) {
          return new Float32Array(data);
        } else if (data instanceof Float32Array || data instanceof Float64Array || 
                  data instanceof Int32Array || data instanceof Uint8Array) {
          return data;
        } else if (Array.isArray(data)) {
          return new Float32Array(data);
        } else {
          throw new Error(`Unsupported input data type: ${{typeof data}}`);
        }
      }
    };
  } catch (error) {
    console.error('Error loading WebNN model:', error);
    throw error;
  }
}

/**
 * Check browser compatibility with this model
 * @returns {Object} - Compatibility status for each browser
 */
export function checkCompatibility() {
  const isWebNNSupported = 'ml' in navigator;
  const isGPUSupported = isWebNNSupported && 'gpu' in navigator.ml;
  const isCPUSupported = isWebNNSupported && 'cpu' in navigator.ml;
  
  // Check each browser
  const browsers = {
    chrome: {
      supported: isWebNNSupported,
      version: _getBrowserVersion('Chrome'),
      gpuAccelerated: isGPUSupported
    },
    firefox: {
      supported: isWebNNSupported,
      version: _getBrowserVersion('Firefox'),
      gpuAccelerated: isGPUSupported
    },
    safari: {
      supported: isWebNNSupported,
      version: _getBrowserVersion('Safari'),
      gpuAccelerated: isGPUSupported
    },
    edge: {
      supported: isWebNNSupported,
      version: _getBrowserVersion('Edg'),
      gpuAccelerated: isGPUSupported
    }
  };
  
  // Get current browser
  const currentBrowser = _getCurrentBrowser();
  
  return {
    isWebNNSupported,
    isGPUSupported,
    isCPUSupported,
    browsers,
    currentBrowser
  };
}

// Helper to get browser version
function _getBrowserVersion(browserName) {
  const userAgent = navigator.userAgent;
  const match = userAgent.match(new RegExp(`${{browserName}}/([\\d.]+)`));
  return match ? match[1] : null;
}

// Helper to get current browser
function _getCurrentBrowser() {
  const userAgent = navigator.userAgent;
  if (userAgent.indexOf('Chrome') > -1) return 'chrome';
  if (userAgent.indexOf('Firefox') > -1) return 'firefox';
  if (userAgent.indexOf('Safari') > -1 && userAgent.indexOf('Chrome') === -1) return 'safari';
  if (userAgent.indexOf('Edg') > -1) return 'edge';
  return 'unknown';
}

export default {
  loadModel,
  checkCompatibility,
  modelConfig
};
"""
        
        # Create inputs creation code
        inputs_creation_lines = []
        for name, info in inputs.items():
            shape_str = str(info['shape']).replace('(', '[').replace(')', ']').replace(',]', ']')
            inputs_creation_lines.append(f'"{name}": builder.input("{name}", {shape_str})')
        inputs_creation = ',\n      '.join(inputs_creation_lines)
        
        # Create output entry
        if len(outputs) == 1:
            # Single output
            output_name = list(outputs.keys())[0]
            output_entry = f'output: model.outputs[0]'
        else:
            # Multiple outputs
            output_entry_parts = []
            for i, name in enumerate(outputs.keys()):
                output_entry_parts.append(f'"{name}": model.outputs[{i}]')
            output_entry = f'outputs: {{{", ".join(output_entry_parts)}}}'
        
        # Format the template
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        formatted_js = js_template.format(
            model_name=model_name,
            model_type=model_type or 'unknown',
            precision=precision,
            layout=layout,
            optimization_level=optimization_level,
            browser_targets=json.dumps(browser_targets),
            inputs=json.dumps(inputs, indent=2),
            outputs=json.dumps(outputs, indent=2),
            model_path_basename=os.path.basename(model_path),
            inputs_creation=inputs_creation,
            output_entry=output_entry
        )
        
        return formatted_js
    
    def _get_model_inputs(self, model) -> Dict[str, Dict[str, Any]]:
        """
        Get model input information.
        
        Args:
            model: ONNX model
            
        Returns:
            Dictionary of input name to input information
        """
        inputs = {}
        for input_info in model.graph.input:
            name = input_info.name
            
            # Extract shape information
            shape = []
            if input_info.type.tensor_type.shape:
                for dim in input_info.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        # Dynamic dimension
                        shape.append(-1)
                    else:
                        shape.append(dim.dim_value)
            
            # Extract type information
            elem_type = input_info.type.tensor_type.elem_type
            type_name = self._get_onnx_type_name(elem_type)
            
            inputs[name] = {
                'shape': shape,
                'type': type_name
            }
            
        return inputs
    
    def _get_model_outputs(self, model) -> Dict[str, Dict[str, Any]]:
        """
        Get model output information.
        
        Args:
            model: ONNX model
            
        Returns:
            Dictionary of output name to output information
        """
        outputs = {}
        for output_info in model.graph.output:
            name = output_info.name
            
            # Extract shape information
            shape = []
            if output_info.type.tensor_type.shape:
                for dim in output_info.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        # Dynamic dimension
                        shape.append(-1)
                    else:
                        shape.append(dim.dim_value)
            
            # Extract type information
            elem_type = output_info.type.tensor_type.elem_type
            type_name = self._get_onnx_type_name(elem_type)
            
            outputs[name] = {
                'shape': shape,
                'type': type_name
            }
            
        return outputs
    
    def _get_onnx_type_name(self, elem_type: int) -> str:
        """
        Get ONNX type name from element type.
        
        Args:
            elem_type: ONNX element type
            
        Returns:
            Type name
        """
        try:
            from onnx import TensorProto
            type_map = {
                TensorProto.FLOAT: 'float32',
                TensorProto.UINT8: 'uint8',
                TensorProto.INT8: 'int8',
                TensorProto.UINT16: 'uint16',
                TensorProto.INT16: 'int16',
                TensorProto.INT32: 'int32',
                TensorProto.INT64: 'int64',
                TensorProto.BOOL: 'bool',
                TensorProto.FLOAT16: 'float16',
                TensorProto.DOUBLE: 'float64',
                TensorProto.COMPLEX64: 'complex64',
                TensorProto.COMPLEX128: 'complex128',
                TensorProto.STRING: 'string'
            }
            return type_map.get(elem_type, f'unknown_{elem_type}')
        except ImportError:
            # Fallback type mapping
            type_map = {
                1: 'float32',
                2: 'uint8',
                3: 'int8',
                4: 'uint16',
                5: 'int16',
                6: 'int32',
                7: 'int64',
                9: 'bool',
                10: 'float16',
                11: 'float64',
                14: 'complex64',
                15: 'complex128'
            }
            return type_map.get(elem_type, f'unknown_{elem_type}')
            
    def validate_model(self, model_path: str, model_type: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate an ONNX model before conversion to WebNN.
        
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
                
                # Check for unsupported operations
                unsupported_ops = self._check_unsupported_ops(model)
                if unsupported_ops:
                    return False, f"Model contains operations not supported by WebNN: {', '.join(unsupported_ops)}"
                
                return True, None
            except Exception as e:
                return False, f"Error validating ONNX model: {e}"
                
        except ImportError as e:
            return False, f"Missing required dependencies: {e}"
    
    def _check_unsupported_ops(self, model) -> List[str]:
        """
        Check for operations in the ONNX model that are not supported by WebNN.
        
        Args:
            model: ONNX model
            
        Returns:
            List of unsupported operations
        """
        # List of operations supported by WebNN
        # This is a simplified list and may not be complete
        supported_ops = {
            'Add', 'AveragePool', 'BatchNormalization', 'Concat', 'Conv', 'Gemm',
            'GlobalAveragePool', 'MatMul', 'MaxPool', 'Mul', 'Relu', 'Reshape',
            'Sigmoid', 'Softmax', 'Split', 'Tanh', 'Transpose'
        }
        
        # Collect all operations in the model
        model_ops = set()
        for node in model.graph.node:
            model_ops.add(node.op_type)
            
        # Find unsupported operations
        unsupported_ops = model_ops - supported_ops
        
        return list(unsupported_ops)