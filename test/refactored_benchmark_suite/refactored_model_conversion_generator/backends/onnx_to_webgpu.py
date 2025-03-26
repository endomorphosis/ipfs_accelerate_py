"""
ONNX to WebGPU Converter

This module provides a converter for ONNX models to WebGPU format.
"""

import os
import logging
import json
import tempfile
from typing import Dict, Any, Optional, List, Tuple

from ..core.converter import ModelConverter, ConversionResult
from ..core.registry import register_converter

logger = logging.getLogger(__name__)

@register_converter(source_format='onnx', target_format='webgpu')
class OnnxToWebGPUConverter(ModelConverter):
    """
    Converter for ONNX models to WebGPU format.
    """
    
    def _get_source_format(self) -> str:
        """Get source format."""
        return 'onnx'
        
    def _get_target_format(self) -> str:
        """Get target format."""
        return 'webgpu'
        
    def _get_supported_model_types(self) -> List[str]:
        """Get supported model types."""
        return [
            'bert', 'vit', 'resnet', 'mobilenet', 'efficientnet', 'whisper'
        ]
        
    def _execute_conversion(self, model_path: str, output_path: str, 
                          model_type: Optional[str] = None, **kwargs) -> ConversionResult:
        """
        Convert ONNX model to WebGPU format.
        
        Args:
            model_path: Path to the ONNX model
            output_path: Path to save the WebGPU model
            model_type: Type of model (e.g., 'bert', 'vit')
            **kwargs: Additional conversion parameters
                - precision: Model precision ('default', 'float16', 'int8', '4bit')
                - use_shader_cache: Whether to use shader caching
                - browser_targets: List of target browsers ('chrome', 'firefox', 'safari', 'edge')
                - wgsl_code_path: Optional path to save generated WGSL shader code
                
        Returns:
            ConversionResult with conversion details
        """
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get conversion options
            precision = kwargs.get('precision', 'default')
            use_shader_cache = kwargs.get('use_shader_cache', True)
            browser_targets = kwargs.get('browser_targets', ['chrome', 'firefox', 'safari', 'edge'])
            wgsl_code_path = kwargs.get('wgsl_code_path', None)
            
            # Validate precision option
            valid_precisions = ['default', 'float16', 'int8', '8bit', '4bit', '3bit', '2bit']
            if precision not in valid_precisions:
                logger.warning(f"Unsupported precision: {precision}. Using default instead.")
                precision = 'default'
                
            # Search for existing quantization work if using ultra-low precision
            if precision in ['4bit', '3bit', '2bit']:
                logger.info(f"Using ultra-low precision: {precision}")
                if model_type:
                    # For model-specific ultra-low precision, search for existing work
                    logger.info(f"Searching for existing {precision} quantization for {model_type} models")
            
            # Load model metadata if available
            metadata = {}
            metadata_path = model_path + '.json'
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Error loading model metadata: {e}")
            
            # Create WebGPU model
            return self._create_webgpu_model(
                model_path, output_path, model_type, precision, 
                use_shader_cache, browser_targets, wgsl_code_path, metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error converting ONNX model to WebGPU: {e}", exc_info=True)
            return ConversionResult(
                success=False,
                error=f"Error converting ONNX model to WebGPU: {e}"
            )
    
    def _create_webgpu_model(self, model_path: str, output_path: str, model_type: Optional[str],
                          precision: str, use_shader_cache: bool, browser_targets: List[str],
                          wgsl_code_path: Optional[str], original_metadata: Dict[str, Any]) -> ConversionResult:
        """
        Create WebGPU model from ONNX model.
        
        Args:
            model_path: Path to the ONNX model
            output_path: Path to save the WebGPU model
            model_type: Type of model (e.g., 'bert', 'vit')
            precision: Model precision ('default', 'float16', 'int8', '4bit')
            use_shader_cache: Whether to use shader caching
            browser_targets: List of target browsers
            wgsl_code_path: Optional path to save generated WGSL shader code
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
                raise ImportError("ONNX and NumPy libraries are required for WebGPU conversion")
                
            # Load ONNX model
            self.logger.info(f"Loading ONNX model: {model_path}")
            model = onnx.load(model_path)
            
            # Validate model
            onnx.checker.check_model(model)
            
            # Generate WGSL shader code for the model operations
            shader_code = self._generate_wgsl_shaders(model, model_type, precision)
            
            # Save WGSL shader code if path provided
            if wgsl_code_path:
                with open(wgsl_code_path, 'w') as f:
                    f.write(shader_code)
                self.logger.info(f"Saved WGSL shader code to {wgsl_code_path}")
            
            # Create a JavaScript module that loads the model in WebGPU
            js_code = self._generate_webgpu_js_module(
                model_path, model, model_type, precision, 
                use_shader_cache, browser_targets, shader_code
            )
            
            # Save the JavaScript module
            with open(output_path, 'w') as f:
                f.write(js_code)
                
            # Save original ONNX model alongside WebGPU model
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
                'use_shader_cache': use_shader_cache,
                'browser_targets': browser_targets,
                'wgsl_code_path': wgsl_code_path,
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
            self.logger.error(f"Error creating WebGPU model: {e}", exc_info=True)
            return ConversionResult(
                success=False,
                error=f"Error creating WebGPU model: {e}"
            )
    
    def _generate_wgsl_shaders(self, model, model_type: Optional[str], precision: str) -> str:
        """
        Generate WGSL shader code for the model operations.
        
        Args:
            model: ONNX model
            model_type: Type of model (e.g., 'bert', 'vit')
            precision: Model precision ('default', 'float16', 'int8', '4bit')
            
        Returns:
            WGSL shader code
        """
        # This is a simplified implementation that would generate WGSL shader code 
        # for common neural network operations like convolution, matmul, etc.
        # A real implementation would be much more complex and would analyze the 
        # ONNX graph to generate optimized WGSL shaders.
        
        # For demonstration purposes, we'll just return a template shader that includes
        # some common operations for the given model type and precision.
        
        # Define precision type
        precision_type = 'f32'
        if precision == 'float16':
            precision_type = 'f16'
        
        # Basic shader header
        shader_header = f"""// WebGPU WGSL Shaders for {model_type} model
// Generated by IPFS Accelerate Model Conversion Generator
// Precision: {precision}

struct Matrix {{
  size : vec2u,
  values : array<{precision_type}>,
}}

struct ComputeShaderInput {{
  @builtin(global_invocation_id) global_id : vec3u,
  @builtin(workgroup_id) workgroup_id : vec3u,
  @builtin(local_invocation_id) local_id : vec3u,
}};

"""
        
        # Generate specific shaders based on model type and operations found in the model
        operation_shaders = []
        
        # Extract all operations from the model
        operations = set()
        for node in model.graph.node:
            operations.add(node.op_type)
        
        # Generate shaders for common operations
        if 'MatMul' in operations:
            operation_shaders.append(self._generate_matmul_shader(precision_type))
            
        if 'Conv' in operations:
            operation_shaders.append(self._generate_conv_shader(precision_type))
            
        if any(op in operations for op in ['Relu', 'Sigmoid', 'Tanh']):
            operation_shaders.append(self._generate_activation_shaders(precision_type, operations))
            
        if 'Softmax' in operations:
            operation_shaders.append(self._generate_softmax_shader(precision_type))
            
        # Add model-type specific shaders
        if model_type == 'bert':
            operation_shaders.append(self._generate_bert_specific_shaders(precision_type))
        elif model_type == 'vit':
            operation_shaders.append(self._generate_vit_specific_shaders(precision_type))
            
        # Add special ultra-low precision quantization shaders if needed
        if precision in ['4bit', '3bit', '2bit']:
            operation_shaders.append(self._generate_ultra_low_precision_shaders(precision))
        
        # Combine all shaders
        return shader_header + "\n\n".join(operation_shaders)
    
    def _generate_matmul_shader(self, precision_type: str) -> str:
        """Generate MatMul shader."""
        return f"""@group(0) @binding(0) var<storage, read> inputA : Matrix;
@group(0) @binding(1) var<storage, read> inputB : Matrix;
@group(0) @binding(2) var<storage, read_write> output : Matrix;

@compute @workgroup_size(8, 8)
fn matmul(in: ComputeShaderInput) {{
  let row = in.global_id.x;
  let col = in.global_id.y;
  
  if (row >= output.size.x || col >= output.size.y) {{
    return;
  }}
  
  let M = output.size.x;
  let N = output.size.y;
  let K = inputA.size.y;  // Must equal inputB.size.x
  
  var sum : {precision_type} = 0.0;
  for (var k : u32 = 0; k < K; k = k + 1) {{
    let a_idx = row * K + k;
    let b_idx = k * N + col;
    sum = sum + inputA.values[a_idx] * inputB.values[b_idx];
  }}
  
  let out_idx = row * N + col;
  output.values[out_idx] = sum;
}}"""
    
    def _generate_conv_shader(self, precision_type: str) -> str:
        """Generate Convolution shader."""
        return f"""@group(0) @binding(0) var<storage, read> input : array<{precision_type}>;
@group(0) @binding(1) var<storage, read> weights : array<{precision_type}>;
@group(0) @binding(2) var<storage, read> bias : array<{precision_type}>;
@group(0) @binding(3) var<storage, read_write> output : array<{precision_type}>;
@group(0) @binding(4) var<uniform> dimensions : array<u32, 10>;  // [N, C, H, W, K, R, S, P, Q, stride]

@compute @workgroup_size(8, 8, 1)
fn conv2d(in: ComputeShaderInput) {{
  let n = in.global_id.z;
  let k = in.global_id.y;
  let p = in.global_id.x / dimensions[8]; // output width Q
  let q = in.global_id.x % dimensions[8];
  
  let N = dimensions[0]; // batch size
  let C = dimensions[1]; // input channels
  let H = dimensions[2]; // input height
  let W = dimensions[3]; // input width
  let K = dimensions[4]; // output channels
  let R = dimensions[5]; // filter height
  let S = dimensions[6]; // filter width
  let P = dimensions[7]; // output height
  let Q = dimensions[8]; // output width
  let stride = dimensions[9];
  
  if (n >= N || k >= K || p >= P || q >= Q) {{
    return;
  }}
  
  var sum : {precision_type} = bias[k];
  
  for (var c : u32 = 0; c < C; c = c + 1) {{
    for (var r : u32 = 0; r < R; r = r + 1) {{
      for (var s : u32 = 0; s < S; s = s + 1) {{
        let h = p * stride + r;
        let w = q * stride + s;
        
        if (h < H && w < W) {{
          let input_idx = n * C * H * W + c * H * W + h * W + w;
          let filter_idx = k * C * R * S + c * R * S + r * S + s;
          sum = sum + input[input_idx] * weights[filter_idx];
        }}
      }}
    }}
  }}
  
  let output_idx = n * K * P * Q + k * P * Q + p * Q + q;
  output[output_idx] = sum;
}}"""
    
    def _generate_activation_shaders(self, precision_type: str, operations: set) -> str:
        """Generate activation function shaders."""
        shaders = [f"""@group(0) @binding(0) var<storage, read> input : array<{precision_type}>;
@group(0) @binding(1) var<storage, read_write> output : array<{precision_type}>;
@group(0) @binding(2) var<uniform> length : u32;"""]
        
        if 'Relu' in operations:
            shaders.append(f"""
@compute @workgroup_size(256)
fn relu(in: ComputeShaderInput) {{
  let idx = in.global_id.x;
  if (idx >= length) {{
    return;
  }}
  
  output[idx] = max(input[idx], 0.0);
}}""")
        
        if 'Sigmoid' in operations:
            shaders.append(f"""
@compute @workgroup_size(256)
fn sigmoid(in: ComputeShaderInput) {{
  let idx = in.global_id.x;
  if (idx >= length) {{
    return;
  }}
  
  output[idx] = 1.0 / (1.0 + exp(-input[idx]));
}}""")
        
        if 'Tanh' in operations:
            shaders.append(f"""
@compute @workgroup_size(256)
fn tanh(in: ComputeShaderInput) {{
  let idx = in.global_id.x;
  if (idx >= length) {{
    return;
  }}
  
  output[idx] = tanh(input[idx]);
}}""")
        
        return "\n".join(shaders)
    
    def _generate_softmax_shader(self, precision_type: str) -> str:
        """Generate Softmax shader."""
        return f"""@group(0) @binding(0) var<storage, read> input : array<{precision_type}>;
@group(0) @binding(1) var<storage, read_write> output : array<{precision_type}>;
@group(0) @binding(2) var<uniform> dims : array<u32, 2>;  // [batch_size, seq_length]

@compute @workgroup_size(256)
fn softmax(in: ComputeShaderInput) {{
  let idx = in.global_id.x;
  let batch_size = dims[0];
  let seq_length = dims[1];
  
  if (idx >= batch_size) {{
    return;
  }}
  
  // Find max value for numerical stability
  var max_value : {precision_type} = -1000000.0;
  for (var i : u32 = 0; i < seq_length; i = i + 1) {{
    let val_idx = idx * seq_length + i;
    max_value = max(max_value, input[val_idx]);
  }}
  
  // Compute exp and sum
  var sum : {precision_type} = 0.0;
  for (var i : u32 = 0; i < seq_length; i = i + 1) {{
    let val_idx = idx * seq_length + i;
    let exp_val = exp(input[val_idx] - max_value);
    output[val_idx] = exp_val;  // Store temporarily
    sum = sum + exp_val;
  }}
  
  // Normalize
  for (var i : u32 = 0; i < seq_length; i = i + 1) {{
    let val_idx = idx * seq_length + i;
    output[val_idx] = output[val_idx] / sum;
  }}
}}"""
    
    def _generate_bert_specific_shaders(self, precision_type: str) -> str:
        """Generate BERT-specific shaders."""
        return f"""@group(0) @binding(0) var<storage, read> qkv : array<{precision_type}>;
@group(0) @binding(1) var<storage, read_write> attention : array<{precision_type}>;
@group(0) @binding(2) var<uniform> dims : array<u32, 4>;  // [batch_size, seq_length, num_heads, head_dim]

@compute @workgroup_size(8, 8)
fn self_attention(in: ComputeShaderInput) {{
  let batch_head = in.global_id.z;
  let batch_size = dims[0];
  let seq_length = dims[1];
  let num_heads = dims[2];
  let head_dim = dims[3];
  
  let batch_id = batch_head / num_heads;
  let head_id = batch_head % num_heads;
  
  let seq_i = in.global_id.x;
  let seq_j = in.global_id.y;
  
  if (batch_id >= batch_size || seq_i >= seq_length || seq_j >= seq_length) {{
    return;
  }}
  
  // Calculate indices into the QKV tensor
  let q_offset = batch_id * seq_length * num_heads * head_dim + seq_i * num_heads * head_dim + head_id * head_dim;
  let k_offset = batch_id * seq_length * num_heads * head_dim + seq_j * num_heads * head_dim + head_id * head_dim;
  
  // Compute dot product of Q and K
  var dot_product : {precision_type} = 0.0;
  for (var d : u32 = 0; d < head_dim; d = d + 1) {{
    dot_product = dot_product + qkv[q_offset + d] * qkv[k_offset + d];
  }}
  
  // Scale by sqrt(head_dim)
  dot_product = dot_product / sqrt({precision_type}(head_dim));
  
  // Store the result in the attention matrix
  let attn_idx = batch_id * num_heads * seq_length * seq_length + head_id * seq_length * seq_length + seq_i * seq_length + seq_j;
  attention[attn_idx] = dot_product;
}}"""
    
    def _generate_vit_specific_shaders(self, precision_type: str) -> str:
        """Generate ViT-specific shaders."""
        return f"""@group(0) @binding(0) var<storage, read> patches : array<{precision_type}>;
@group(0) @binding(1) var<storage, read> pos_embeddings : array<{precision_type}>;
@group(0) @binding(2) var<storage, read_write> embedded_patches : array<{precision_type}>;
@group(0) @binding(3) var<uniform> dims : array<u32, 3>;  // [batch_size, num_patches, embedding_dim]

@compute @workgroup_size(256)
fn patch_embedding(in: ComputeShaderInput) {{
  let idx = in.global_id.x;
  let batch_size = dims[0];
  let num_patches = dims[1];
  let embedding_dim = dims[2];
  
  let batch_id = idx / num_patches;
  let patch_id = idx % num_patches;
  
  if (batch_id >= batch_size || patch_id >= num_patches) {{
    return;
  }}
  
  let patch_offset = batch_id * num_patches * embedding_dim + patch_id * embedding_dim;
  
  // Add positional embedding to patch embedding
  for (var d : u32 = 0; d < embedding_dim; d = d + 1) {{
    embedded_patches[patch_offset + d] = patches[patch_offset + d] + pos_embeddings[patch_id * embedding_dim + d];
  }}
}}"""
    
    def _generate_ultra_low_precision_shaders(self, precision: str) -> str:
        """
        Generate ultra-low precision quantization shaders.
        
        Args:
            precision: Precision level ('4bit', '3bit', '2bit')
            
        Returns:
            WGSL shader code for the specified precision
        """
        if precision == '4bit':
            return self._generate_4bit_quantization_shaders()
        elif precision == '3bit':
            return self._generate_3bit_quantization_shaders()
        elif precision == '2bit':
            return self._generate_2bit_quantization_shaders()
        else:
            return self._generate_4bit_quantization_shaders()  # Default to 4-bit
            
    def _generate_4bit_quantization_shaders(self) -> str:
        """Generate 4-bit quantization shaders."""
        return """// 4-bit quantization support
struct QuantizationParams {
  scale : array<f32, 16>,
  zero_point : array<i32, 16>,
}

@group(0) @binding(0) var<storage, read> quantized_weights : array<u8>;
@group(0) @binding(1) var<uniform> quant_params : QuantizationParams;
@group(0) @binding(2) var<storage, read_write> dequantized_weights : array<f32>;
@group(0) @binding(3) var<uniform> length : u32;

@compute @workgroup_size(256)
fn dequantize_4bit(in: ComputeShaderInput) {
  let idx = in.global_id.x;
  if (idx >= length / 2) {
    return;
  }
  
  // Each byte contains two 4-bit values
  let byte = quantized_weights[idx];
  let value1 = byte & 0x0F;
  let value2 = (byte >> 4) & 0x0F;
  
  // Get group index for values (simplified for demonstration)
  let group1 = value1 % 16;
  let group2 = value2 % 16;
  
  // Dequantize using scale and zero point
  dequantized_weights[idx * 2] = f32(i32(value1) - quant_params.zero_point[group1]) * quant_params.scale[group1];
  dequantized_weights[idx * 2 + 1] = f32(i32(value2) - quant_params.zero_point[group2]) * quant_params.scale[group2];
}

@compute @workgroup_size(256)
fn quantize_4bit(in: ComputeShaderInput) {
  let idx = in.global_id.x;
  if (idx >= length / 2) {
    return;
  }
  
  // Get group index for values (simplified for demonstration)
  let group1 = idx % 16;
  let group2 = (idx + 1) % 16;
  
  // Quantize using scale and zero point
  let value1 = i32(dequantized_weights[idx * 2] / quant_params.scale[group1] + f32(quant_params.zero_point[group1]));
  let value2 = i32(dequantized_weights[idx * 2 + 1] / quant_params.scale[group2] + f32(quant_params.zero_point[group2]));
  
  // Clamp to 4-bit range
  let clamped1 = min(max(value1, 0), 15);
  let clamped2 = min(max(value2, 0), 15);
  
  // Pack two 4-bit values into one byte
  quantized_weights[idx] = u8(clamped1 | (clamped2 << 4));
}"""

    def _generate_3bit_quantization_shaders(self) -> str:
        """Generate 3-bit quantization shaders."""
        return """// 3-bit quantization support
struct Quantization3BitParams {
  scale : array<f32, 8>,
  zero_point : array<i32, 8>,
}

@group(0) @binding(0) var<storage, read> quantized_weights : array<u8>;
@group(0) @binding(1) var<uniform> quant_params : Quantization3BitParams;
@group(0) @binding(2) var<storage, read_write> dequantized_weights : array<f32>;
@group(0) @binding(3) var<uniform> length : u32;

@compute @workgroup_size(256)
fn dequantize_3bit(in: ComputeShaderInput) {
  let idx = in.global_id.x;
  if (idx >= length / 8) {
    return;
  }
  
  // Process 3 bytes at a time (8 3-bit values)
  let byte1 = quantized_weights[idx * 3];
  let byte2 = quantized_weights[idx * 3 + 1];
  let byte3 = quantized_weights[idx * 3 + 2];
  
  // Extract 3-bit values from the 3 bytes
  let val1 = byte1 & 0x07;
  let val2 = (byte1 >> 3) & 0x07;
  let val3 = ((byte1 >> 6) & 0x03) | ((byte2 & 0x01) << 2);
  let val4 = (byte2 >> 1) & 0x07;
  let val5 = (byte2 >> 4) & 0x07;
  let val6 = ((byte2 >> 7) & 0x01) | ((byte3 & 0x03) << 1);
  let val7 = (byte3 >> 2) & 0x07;
  let val8 = (byte3 >> 5) & 0x07;
  
  // Dequantize the values
  dequantized_weights[idx * 8] = f32(i32(val1) - quant_params.zero_point[val1 % 8]) * quant_params.scale[val1 % 8];
  dequantized_weights[idx * 8 + 1] = f32(i32(val2) - quant_params.zero_point[val2 % 8]) * quant_params.scale[val2 % 8];
  dequantized_weights[idx * 8 + 2] = f32(i32(val3) - quant_params.zero_point[val3 % 8]) * quant_params.scale[val3 % 8];
  dequantized_weights[idx * 8 + 3] = f32(i32(val4) - quant_params.zero_point[val4 % 8]) * quant_params.scale[val4 % 8];
  dequantized_weights[idx * 8 + 4] = f32(i32(val5) - quant_params.zero_point[val5 % 8]) * quant_params.scale[val5 % 8];
  dequantized_weights[idx * 8 + 5] = f32(i32(val6) - quant_params.zero_point[val6 % 8]) * quant_params.scale[val6 % 8];
  dequantized_weights[idx * 8 + 6] = f32(i32(val7) - quant_params.zero_point[val7 % 8]) * quant_params.scale[val7 % 8];
  dequantized_weights[idx * 8 + 7] = f32(i32(val8) - quant_params.zero_point[val8 % 8]) * quant_params.scale[val8 % 8];
}

// Function to pack 8 3-bit values into 3 bytes
@compute @workgroup_size(256)
fn quantize_3bit(in: ComputeShaderInput) {
  let idx = in.global_id.x;
  if (idx >= length / 8) {
    return;
  }
  
  // Quantize 8 values
  let group1 = idx % 8;
  let group2 = (idx + 1) % 8;
  let group3 = (idx + 2) % 8;
  let group4 = (idx + 3) % 8;
  let group5 = (idx + 4) % 8;
  let group6 = (idx + 5) % 8;
  let group7 = (idx + 6) % 8;
  let group8 = (idx + 7) % 8;
  
  // Quantize to 3-bit values (0-7)
  let q1 = i32(dequantized_weights[idx * 8] / quant_params.scale[group1] + f32(quant_params.zero_point[group1]));
  let q2 = i32(dequantized_weights[idx * 8 + 1] / quant_params.scale[group2] + f32(quant_params.zero_point[group2]));
  let q3 = i32(dequantized_weights[idx * 8 + 2] / quant_params.scale[group3] + f32(quant_params.zero_point[group3]));
  let q4 = i32(dequantized_weights[idx * 8 + 3] / quant_params.scale[group4] + f32(quant_params.zero_point[group4]));
  let q5 = i32(dequantized_weights[idx * 8 + 4] / quant_params.scale[group5] + f32(quant_params.zero_point[group5]));
  let q6 = i32(dequantized_weights[idx * 8 + 5] / quant_params.scale[group6] + f32(quant_params.zero_point[group6]));
  let q7 = i32(dequantized_weights[idx * 8 + 6] / quant_params.scale[group7] + f32(quant_params.zero_point[group7]));
  let q8 = i32(dequantized_weights[idx * 8 + 7] / quant_params.scale[group8] + f32(quant_params.zero_point[group8]));
  
  // Clamp to 3-bit range (0-7)
  let v1 = min(max(q1, 0), 7);
  let v2 = min(max(q2, 0), 7);
  let v3 = min(max(q3, 0), 7);
  let v4 = min(max(q4, 0), 7);
  let v5 = min(max(q5, 0), 7);
  let v6 = min(max(q6, 0), 7);
  let v7 = min(max(q7, 0), 7);
  let v8 = min(max(q8, 0), 7);
  
  // Pack the 8 3-bit values into 3 bytes
  // First byte: contains v1 (3 bits) + v2 (3 bits) + first 2 bits of v3
  quantized_weights[idx * 3] = u8(v1 | (v2 << 3) | ((v3 & 0x03) << 6));
  // Second byte: contains last 1 bit of v3 + v4 (3 bits) + v5 (3 bits) + first 1 bit of v6
  quantized_weights[idx * 3 + 1] = u8(((v3 & 0x04) >> 2) | (v4 << 1) | (v5 << 4) | ((v6 & 0x01) << 7));
  // Third byte: contains last 2 bits of v6 + v7 (3 bits) + v8 (3 bits)
  quantized_weights[idx * 3 + 2] = u8(((v6 & 0x06) >> 1) | (v7 << 2) | (v8 << 5));
}"""

    def _generate_2bit_quantization_shaders(self) -> str:
        """Generate 2-bit quantization shaders."""
        return """// 2-bit quantization support
struct Quantization2BitParams {
  scale : array<f32, 4>,
  zero_point : array<i32, 4>,
}

@group(0) @binding(0) var<storage, read> quantized_weights : array<u8>;
@group(0) @binding(1) var<uniform> quant_params : Quantization2BitParams;
@group(0) @binding(2) var<storage, read_write> dequantized_weights : array<f32>;
@group(0) @binding(3) var<uniform> length : u32;

@compute @workgroup_size(256)
fn dequantize_2bit(in: ComputeShaderInput) {
  let idx = in.global_id.x;
  if (idx >= length / 4) {
    return;
  }
  
  // Each byte contains four 2-bit values
  let byte = quantized_weights[idx];
  
  // Extract 2-bit values
  let val1 = byte & 0x03;
  let val2 = (byte >> 2) & 0x03;
  let val3 = (byte >> 4) & 0x03;
  let val4 = (byte >> 6) & 0x03;
  
  // Dequantize using scale and zero point
  dequantized_weights[idx * 4] = f32(i32(val1) - quant_params.zero_point[val1]) * quant_params.scale[val1];
  dequantized_weights[idx * 4 + 1] = f32(i32(val2) - quant_params.zero_point[val2]) * quant_params.scale[val2];
  dequantized_weights[idx * 4 + 2] = f32(i32(val3) - quant_params.zero_point[val3]) * quant_params.scale[val3];
  dequantized_weights[idx * 4 + 3] = f32(i32(val4) - quant_params.zero_point[val4]) * quant_params.scale[val4];
}

@compute @workgroup_size(256)
fn quantize_2bit(in: ComputeShaderInput) {
  let idx = in.global_id.x;
  if (idx >= length / 4) {
    return;
  }
  
  // Quantize four values to 2-bit each
  let q1 = i32(dequantized_weights[idx * 4] / quant_params.scale[0] + f32(quant_params.zero_point[0]));
  let q2 = i32(dequantized_weights[idx * 4 + 1] / quant_params.scale[1] + f32(quant_params.zero_point[1]));
  let q3 = i32(dequantized_weights[idx * 4 + 2] / quant_params.scale[2] + f32(quant_params.zero_point[2]));
  let q4 = i32(dequantized_weights[idx * 4 + 3] / quant_params.scale[3] + f32(quant_params.zero_point[3]));
  
  // Clamp to 2-bit range (0-3)
  let v1 = min(max(q1, 0), 3);
  let v2 = min(max(q2, 0), 3);
  let v3 = min(max(q3, 0), 3);
  let v4 = min(max(q4, 0), 3);
  
  // Pack four 2-bit values into one byte
  quantized_weights[idx] = u8(v1 | (v2 << 2) | (v3 << 4) | (v4 << 6));
}

// Mixed precision functions for handling different precision in different parts of the model
struct MixedPrecisionConfig {
  use_2bit : array<u32, 16>,  // Layer indices that use 2-bit
  use_3bit : array<u32, 16>,  // Layer indices that use 3-bit
  use_4bit : array<u32, 16>,  // Layer indices that use 4-bit
  use_8bit : array<u32, 16>,  // Layer indices that use 8-bit
  use_fp16 : array<u32, 16>,  // Layer indices that use fp16
  num_layers : u32,           // Total number of layers
}

@group(0) @binding(4) var<uniform> mixed_precision_config : MixedPrecisionConfig;

// Helper function to determine which precision to use for a given layer
fn get_precision_for_layer(layer_idx: u32) -> u32 {
  // Check 2-bit layers
  for (var i: u32 = 0; i < 16; i = i + 1) {
    if (i >= mixed_precision_config.num_layers) {
      break;
    }
    if (mixed_precision_config.use_2bit[i] == layer_idx) {
      return 2;
    }
  }
  
  // Check 3-bit layers
  for (var i: u32 = 0; i < 16; i = i + 1) {
    if (i >= mixed_precision_config.num_layers) {
      break;
    }
    if (mixed_precision_config.use_3bit[i] == layer_idx) {
      return 3;
    }
  }
  
  // Check 4-bit layers
  for (var i: u32 = 0; i < 16; i = i + 1) {
    if (i >= mixed_precision_config.num_layers) {
      break;
    }
    if (mixed_precision_config.use_4bit[i] == layer_idx) {
      return 4;
    }
  }
  
  // Check 8-bit layers
  for (var i: u32 = 0; i < 16; i = i + 1) {
    if (i >= mixed_precision_config.num_layers) {
      break;
    }
    if (mixed_precision_config.use_8bit[i] == layer_idx) {
      return 8;
    }
  }
  
  // Check fp16 layers
  for (var i: u32 = 0; i < 16; i = i + 1) {
    if (i >= mixed_precision_config.num_layers) {
      break;
    }
    if (mixed_precision_config.use_fp16[i] == layer_idx) {
      return 16;
    }
  }
  
  // Default to fp32
  return 32;
}"""
    
    def _generate_webgpu_js_module(self, model_path: str, onnx_model, model_type: Optional[str],
                                 precision: str, use_shader_cache: bool,
                                 browser_targets: List[str], shader_code: str) -> str:
        """
        Generate JavaScript module that loads the model in WebGPU.
        
        Args:
            model_path: Path to the ONNX model
            onnx_model: Loaded ONNX model
            model_type: Type of model (e.g., 'bert', 'vit')
            precision: Model precision ('default', 'float16', 'int8', '4bit')
            use_shader_cache: Whether to use shader caching
            browser_targets: List of target browsers
            shader_code: Generated WGSL shader code
            
        Returns:
            JavaScript module code
        """
        # Get model inputs and outputs
        inputs = self._get_model_inputs(onnx_model)
        outputs = self._get_model_outputs(onnx_model)
        
        # Create a JavaScript model loader for WebGPU
        js_template = """
/**
 * WebGPU model loader for {model_name}
 * Converted from ONNX format
 * Model type: {model_type}
 * Generated by IPFS Accelerate Model Conversion Generator
 */

// Model configuration
const modelConfig = {
  name: '{model_name}',
  type: '{model_type}',
  precision: '{precision}',
  useShaderCache: {use_shader_cache},
  browserTargets: {browser_targets},
  inputs: {inputs},
  outputs: {outputs}
};

// WGSL shader code
const shaderCode = `{shader_code}`;

/**
 * Load the model using WebGPU API
 * @param {Object} options - Loading options
 * @param {string} options.modelPath - Path to the ONNX model file
 * @param {Object} options.device - WebGPU device
 * @param {boolean} options.enableProfiling - Whether to enable performance profiling
 * @returns {Promise<Object>} - Loaded model
 */
export async function loadModel(options = {}) {
  const {{
    modelPath = '{model_path_basename}',
    device = null,
    enableProfiling = false
  }} = options;

  // Check if WebGPU is supported
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported in this browser');
  }

  try {
    // Get WebGPU adapter and device if not provided
    const gpuDevice = device || await initWebGPU();
    console.log(`Loading model from ${{modelPath}} using WebGPU`);
    
    // Fetch the ONNX model
    const response = await fetch(modelPath);
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${{response.statusText}}`);
    }
    
    const modelBuffer = await response.arrayBuffer();
    
    // Parse ONNX model
    const modelData = await parseOnnxModel(modelBuffer);
    
    // Create shader modules
    const shaderModules = await createShaderModules(gpuDevice, shaderCode);
    
    // Create compute pipelines
    const computePipelines = await createComputePipelines(gpuDevice, shaderModules, modelData);
    
    // Allocate buffers for weights, inputs, and outputs
    const buffers = await createBuffers(gpuDevice, modelData);
    
    // Create binding groups
    const bindGroups = await createBindGroups(gpuDevice, computePipelines, buffers);
    
    return {
      model: {
        modelData,
        device: gpuDevice,
        computePipelines,
        buffers,
        bindGroups,
        shaderModules
      },
      config: modelConfig,
      async run(inputData) {
        // Verify inputs
        for (const inputName of Object.keys(modelConfig.inputs)) {
          if (!inputData[inputName]) {
            throw new Error(`Input "${{inputName}}" is required but not provided`);
          }
        }
        
        // Start profiling if enabled
        let profilingData = null;
        if (enableProfiling) {
          profilingData = {
            startTime: performance.now(),
            stages: []
          };
        }
        
        // Upload input data to device
        await this._uploadInputData(inputData);
        
        if (enableProfiling) {
          profilingData.stages.push({
            name: 'uploadInputs',
            time: performance.now() - profilingData.startTime
          });
        }
        
        // Run computation
        await this._runComputation();
        
        if (enableProfiling) {
          profilingData.stages.push({
            name: 'computation',
            time: performance.now() - profilingData.startTime
          });
        }
        
        // Download output data
        const outputs = await this._downloadOutputs();
        
        if (enableProfiling) {
          profilingData.stages.push({
            name: 'downloadOutputs',
            time: performance.now() - profilingData.startTime
          });
          profilingData.totalTime = performance.now() - profilingData.startTime;
          outputs._profiling = profilingData;
        }
        
        return outputs;
      },
      
      async _uploadInputData(inputData) {
        const {{device, buffers}} = this.model;
        
        // For each input, upload data to the corresponding buffer
        for (const [name, info] of Object.entries(modelConfig.inputs)) {
          const inputBuffer = buffers.inputs[name];
          const data = inputData[name];
          
          // Convert input data to the appropriate typed array based on precision
          let typedArray;
          if (data instanceof Float32Array) {
            typedArray = data;
          } else if (Array.isArray(data)) {
            typedArray = new Float32Array(data);
          } else if (data instanceof ArrayBuffer) {
            typedArray = new Float32Array(data);
          } else {
            throw new Error(`Unsupported input data type for ${{name}}: ${{typeof data}}`);
          }
          
          // Write data to buffer
          device.queue.writeBuffer(inputBuffer, 0, typedArray);
        }
      },
      
      async _runComputation() {
        const {{device, computePipelines, bindGroups}} = this.model;
        
        // Create command encoder
        const commandEncoder = device.createCommandEncoder();
        
        // For each operation in the model, run the corresponding compute pass
        for (const [opName, pipeline] of Object.entries(computePipelines)) {
          const computePass = commandEncoder.beginComputePass();
          computePass.setPipeline(pipeline);
          
          // Set bind group for this operation
          computePass.setBindGroup(0, bindGroups[opName]);
          
          // Determine dispatch size based on operation
          const dispatch = this._getDispatchSize(opName);
          computePass.dispatchWorkgroups(dispatch.x, dispatch.y, dispatch.z);
          
          computePass.end();
        }
        
        // Submit command buffer and wait for completion
        const commandBuffer = commandEncoder.finish();
        device.queue.submit([commandBuffer]);
        await device.queue.onSubmittedWorkDone();
      },
      
      async _downloadOutputs() {
        const {{device, buffers}} = this.model;
        const results = {};
        
        // For each output, download data from the corresponding buffer
        for (const [name, info] of Object.entries(modelConfig.outputs)) {
          const outputBuffer = buffers.outputs[name];
          const size = this._getBufferSize(info.shape, info.type);
          
          // Create staging buffer for reading
          const stagingBuffer = device.createBuffer({
            size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
          });
          
          // Copy output to staging buffer
          const commandEncoder = device.createCommandEncoder();
          commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, size);
          device.queue.submit([commandEncoder.finish()]);
          
          // Map staging buffer and read data
          await stagingBuffer.mapAsync(GPUMapMode.READ);
          const mappedData = new Float32Array(stagingBuffer.getMappedRange().slice(0));
          stagingBuffer.unmap();
          
          results[name] = {
            data: mappedData,
            shape: info.shape
          };
        }
        
        return results;
      },
      
      _getDispatchSize(opName) {
        // Determine workgroup dispatch size based on operation
        // This is a simplified version and would be more complex in a real implementation
        
        const {{modelData}} = this.model;
        const op = modelData.operations[opName];
        
        switch (op.type) {
          case 'MatMul':
            // Dispatch based on output matrix dimensions
            const M = op.outputs[0].shape[0];
            const N = op.outputs[0].shape[1];
            return {
              x: Math.ceil(M / 8),
              y: Math.ceil(N / 8),
              z: 1
            };
            
          case 'Conv':
            // Dispatch based on output feature map dimensions
            const [N, K, P, Q] = op.outputs[0].shape;
            return {
              x: Math.ceil(P * Q / 8),
              y: Math.ceil(K / 8),
              z: N
            };
            
          default:
            // Default dispatch size for other operations
            return {
              x: Math.ceil(modelData.maxTensorSize / 256),
              y: 1,
              z: 1
            };
        }
      },
      
      _getBufferSize(shape, type) {
        // Calculate buffer size in bytes
        const elementCount = shape.reduce((a, b) => a * b, 1);
        
        // Determine element size based on type and precision
        let elementSize;
        if (modelConfig.precision === 'float16') {
          elementSize = 2;
        } else if (modelConfig.precision === 'int8' || modelConfig.precision === '4bit') {
          elementSize = 1;
        } else {
          elementSize = 4;  // Default to float32
        }
        
        return elementCount * elementSize;
      }
    };
  } catch (error) {
    console.error('Error loading WebGPU model:', error);
    throw error;
  }
}

/**
 * Initialize WebGPU device
 * @returns {Promise<GPUDevice>} - WebGPU device
 */
async function initWebGPU() {
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported');
  }
  
  // Request adapter
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance'
  });
  
  if (!adapter) {
    throw new Error('Failed to get GPU adapter');
  }
  
  // Request device
  const device = await adapter.requestDevice({
    requiredFeatures: ['shader-f16'],
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup
    }
  });
  
  return device;
}

/**
 * Parse ONNX model
 * @param {ArrayBuffer} buffer - ONNX model buffer
 * @returns {Object} - Parsed model data
 */
async function parseOnnxModel(buffer) {
  // This is a placeholder for actual ONNX parsing logic
  // In a real implementation, this would parse the ONNX buffer and extract
  // model structure, weights, operations, etc.
  
  // For demonstration, just return a simple model structure
  return {
    operations: {
      // Example operations would be extracted from the ONNX model
      op1: { type: 'MatMul', inputs: [...], outputs: [...] },
      op2: { type: 'Add', inputs: [...], outputs: [...] }
    },
    weights: {
      // Example weights would be extracted from the ONNX model
    },
    maxTensorSize: 1000000  // Example size
  };
}

/**
 * Create shader modules
 * @param {GPUDevice} device - WebGPU device
 * @param {string} shaderCode - WGSL shader code
 * @returns {Object} - Shader modules
 */
async function createShaderModules(device, shaderCode) {
  // Create shader module from the shader code
  const shaderModule = device.createShaderModule({
    code: shaderCode
  });
  
  // Return object with shader modules
  return {
    main: shaderModule
  };
}

/**
 * Create compute pipelines
 * @param {GPUDevice} device - WebGPU device
 * @param {Object} shaderModules - Shader modules
 * @param {Object} modelData - Model data
 * @returns {Object} - Compute pipelines
 */
async function createComputePipelines(device, shaderModules, modelData) {
  const pipelines = {};
  
  // For each operation in the model, create a compute pipeline
  for (const [opName, op] of Object.entries(modelData.operations)) {
    let entryPoint;
    
    // Determine entry point based on operation type
    switch (op.type) {
      case 'MatMul':
        entryPoint = 'matmul';
        break;
      case 'Conv':
        entryPoint = 'conv2d';
        break;
      case 'Relu':
        entryPoint = 'relu';
        break;
      case 'Sigmoid':
        entryPoint = 'sigmoid';
        break;
      case 'Softmax':
        entryPoint = 'softmax';
        break;
      default:
        console.warn(`No specific entry point for operation ${{op.type}}, using fallback`);
        entryPoint = 'defaultOp';
    }
    
    // Create pipeline layout
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [
        // Bind group layout would be specific to each operation
        device.createBindGroupLayout({
          entries: [
            // Example bind group entries
            {
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            {
              binding: 1,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'read-only-storage' }
            },
            {
              binding: 2,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' }
            }
          ]
        })
      ]
    });
    
    // Create compute pipeline
    pipelines[opName] = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModules.main,
        entryPoint
      }
    });
  }
  
  return pipelines;
}

/**
 * Create buffers for model weights, inputs, and outputs
 * @param {GPUDevice} device - WebGPU device
 * @param {Object} modelData - Model data
 * @returns {Object} - Buffers
 */
async function createBuffers(device, modelData) {
  const buffers = {
    weights: {},
    inputs: {},
    outputs: {}
  };
  
  // Create buffers for model weights
  for (const [name, weight] of Object.entries(modelData.weights)) {
    const size = weight.data.byteLength;
    const buffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    
    // Copy weight data to buffer
    new Float32Array(buffer.getMappedRange()).set(new Float32Array(weight.data));
    buffer.unmap();
    
    buffers.weights[name] = buffer;
  }
  
  // Create buffers for model inputs
  for (const [name, info] of Object.entries(modelConfig.inputs)) {
    const size = info.shape.reduce((a, b) => a * b, 1) * 4;  // Assuming float32
    const buffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    buffers.inputs[name] = buffer;
  }
  
  // Create buffers for model outputs
  for (const [name, info] of Object.entries(modelConfig.outputs)) {
    const size = info.shape.reduce((a, b) => a * b, 1) * 4;  // Assuming float32
    const buffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    
    buffers.outputs[name] = buffer;
  }
  
  return buffers;
}

/**
 * Create binding groups
 * @param {GPUDevice} device - WebGPU device
 * @param {Object} computePipelines - Compute pipelines
 * @param {Object} buffers - Buffers
 * @returns {Object} - Binding groups
 */
async function createBindGroups(device, computePipelines, buffers) {
  const bindGroups = {};
  
  // For each operation, create a bind group
  for (const [opName, pipeline] of Object.entries(computePipelines)) {
    // This is simplified and would be more complex in a real implementation
    // Each operation would have a specific bind group layout
    
    bindGroups[opName] = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        // Example bind group entries
        {
          binding: 0,
          resource: { buffer: buffers.inputs.input }
        },
        {
          binding: 1,
          resource: { buffer: buffers.weights.weight }
        },
        {
          binding: 2,
          resource: { buffer: buffers.outputs.output }
        }
      ]
    });
  }
  
  return bindGroups;
}

/**
 * Check browser compatibility with this model
 * @returns {Object} - Compatibility status for each browser
 */
export function checkCompatibility() {
  const isWebGPUSupported = 'gpu' in navigator;
  
  // Check each browser
  const browsers = {
    chrome: {
      supported: isWebGPUSupported && navigator.userAgent.includes('Chrome'),
      version: _getBrowserVersion('Chrome'),
      notes: 'Best support with Chrome 113+'
    },
    firefox: {
      supported: isWebGPUSupported && navigator.userAgent.includes('Firefox'),
      version: _getBrowserVersion('Firefox'),
      notes: 'Support varies, best with Firefox 118+'
    },
    safari: {
      supported: isWebGPUSupported && navigator.userAgent.includes('Safari') && !navigator.userAgent.includes('Chrome'),
      version: _getBrowserVersion('Safari'),
      notes: 'Support varies, best with Safari 17+'
    },
    edge: {
      supported: isWebGPUSupported && navigator.userAgent.includes('Edg'),
      version: _getBrowserVersion('Edg'),
      notes: 'Based on Chromium, similar to Chrome'
    }
  };
  
  // Get current browser
  const currentBrowser = _getCurrentBrowser();
  
  return {
    isWebGPUSupported,
    browsers,
    currentBrowser,
    precision: {
      supportsFP16: isWebGPUSupported && navigator.gpu?.features?.has('shader-f16'),
      supports4bit: modelConfig.precision === '4bit' && browsers[currentBrowser]?.supported
    }
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
  if (userAgent.indexOf('Chrome') > -1 && userAgent.indexOf('Edg') === -1) return 'chrome';
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
        
        # Format the template
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        formatted_js = js_template.format(
            model_name=model_name,
            model_type=model_type or 'unknown',
            precision=precision,
            use_shader_cache='true' if use_shader_cache else 'false',
            browser_targets=json.dumps(browser_targets),
            inputs=json.dumps(inputs, indent=2),
            outputs=json.dumps(outputs, indent=2),
            model_path_basename=os.path.basename(model_path),
            shader_code=shader_code.replace('`', '\\`')
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
        Validate an ONNX model before conversion to WebGPU.
        
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
                
                # WebGPU has potential limitations for large models
                model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
                if model_size > 500:
                    return False, f"Model size ({model_size:.2f} MB) may be too large for WebGPU"
                
                return True, None
            except Exception as e:
                return False, f"Error validating ONNX model: {e}"
                
        except ImportError as e:
            return False, f"Missing required dependencies: {e}"