/**
 * Converted from Python: webgpu_transformer_compute_shaders.py
 * Conversion date: 2025-03-11 04:09:34
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  compute_enabled: logger;
}

#!/usr/bin/env python3
"""
WebGPU Compute Shader Optimization for Transformer Models.

This module implements specialized compute shader optimizations for transformer models,
focusing on optimizing attention mechanisms, improving memory efficiency, and
enhancing the performance of common transformer operations like layer normalization.

Usage:
  # Import in other modules
  from fixed_web_platform.webgpu_transformer_compute_shaders import * as $1
"""

import * as $1
import * as $1
import * as $1
import * as $1
import ${$1} from "$1"

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("webgpu_transformer_compute")

# Constants for shader workgroup configurations
DEFAULT_WORKGROUP_SIZE = 256
ATTENTION_WORKGROUP_SIZE = 128
LAYERNORM_WORKGROUP_SIZE = 64
MLP_WORKGROUP_SIZE = 256
WARP_SIZE = 32  # GPU warp/wavefront size for alignment
MAX_SEQUENCE_LENGTH = 2048
MAX_HEADS = 32
MAX_HEAD_DIM = 128

class $1 extends $2 {
  """Implementation of WebGPU compute shaders for transformer models."""
  
}
  $1($2) {
    """
    Initialize WebGPU transformer compute shader optimizer.
    
  }
    Args:
      model_name: Name of the transformer model
      seq_length: Maximum sequence length
    """
    this.model_name = model_name
    this.seq_length = min(seq_length, MAX_SEQUENCE_LENGTH)
    this.hidden_size = 768  # Default hidden size
    this.num_heads = 12     # Default number of attention heads
    this.head_dim = 64      # Default head dimension
    this.compute_enabled = os.environ.get("WEBGPU_COMPUTE_SHADERS_ENABLED") == "1"
    this.shader_precompile = os.environ.get("WEBGPU_SHADER_PRECOMPILE_ENABLED") == "1"
    
    # Initialize performance metrics
    this.performance_metrics = {
      "compute_shader_config": {
        "workgroup_size": DEFAULT_WORKGROUP_SIZE,
        "attention_mechanism": ${$1},
        "layer_norm": ${$1},
        "mlp": ${$1}
      },
      }
      "attention_time_ms": 0.0,
      "layer_norm_time_ms": 0.0,
      "mlp_time_ms": 0.0,
      "total_compute_time_ms": 0.0,
      "memory_reduction_percent": 0.0
    }
    }
    
    logger.info(`$1`)
    
  def configure_for_model(self, $1: string, $1: Record<$2, $3> = null) -> Dict[str, Any]:
    """
    Configure compute shader settings based on model type.
    
    Args:
      model_type: Type of transformer model (bert, t5, llama, etc.)
      config: Optional configuration parameters
      
    Returns:
      Dictionary with compute shader configuration
    """
    if ($1) {
      logger.warning("WebGPU compute shaders !enabled, using default configuration")
      return this.performance_metrics
    
    }
    # Check if Flash Attention should be enabled
    # default to true, can be disabled via config
    enable_flash_attention = true
    if ($1) {
      enable_flash_attention = config["enable_flash_attention"]
    
    }
    # Apply model-specific optimizations
    if ($1) {
      # BERT-specific optimizations
      this.hidden_size = 768
      this.num_heads = 12
      this.head_dim = this.hidden_size // this.num_heads
      
    }
      if ($1) ${$1} else {
        this.performance_metrics["compute_shader_config"]["attention_mechanism"]["algorithm"] = "masked_self_attention"
        this.performance_metrics["memory_reduction_percent"] = 18.5
        
      }
      this.performance_metrics["compute_shader_config"]["attention_mechanism"]["kv_cache_enabled"] = false
      this.performance_metrics["compute_shader_config"]["layer_norm"]["algorithm"] = "optimized_layernorm"
      this.performance_metrics["compute_shader_config"]["mlp"]["algorithm"] = "fused_gelu"
      this.performance_metrics["compute_shader_config"]["optimized_for"] = "bert"
      
    elif ($1) {
      # T5-specific optimizations
      this.hidden_size = 512  # Default for t5-small
      this.num_heads = 8
      this.head_dim = this.hidden_size // this.num_heads
      
    }
      if ($1) ${$1} else {
        this.performance_metrics["compute_shader_config"]["attention_mechanism"]["algorithm"] = "cross_attention"
        this.performance_metrics["memory_reduction_percent"] = 22.0
        
      }
      this.performance_metrics["compute_shader_config"]["attention_mechanism"]["kv_cache_enabled"] = true
      this.performance_metrics["compute_shader_config"]["layer_norm"]["algorithm"] = "rms_norm"
      this.performance_metrics["compute_shader_config"]["mlp"]["algorithm"] = "fused_relu"
      this.performance_metrics["compute_shader_config"]["optimized_for"] = "t5"
      
    elif ($1) {
      # LLaMA-specific optimizations
      this.hidden_size = 4096  # Default for larger LLaMA models
      this.num_heads = 32
      this.head_dim = 128
      
    }
      if ($1) ${$1} else {
        this.performance_metrics["compute_shader_config"]["attention_mechanism"]["algorithm"] = "sliding_window"
        this.performance_metrics["compute_shader_config"]["attention_mechanism"]["window_size"] = 4096
        this.performance_metrics["memory_reduction_percent"] = 28.5
        
      }
      this.performance_metrics["compute_shader_config"]["attention_mechanism"]["kv_cache_enabled"] = true
      this.performance_metrics["compute_shader_config"]["layer_norm"]["algorithm"] = "rms_norm"
      this.performance_metrics["compute_shader_config"]["mlp"]["algorithm"] = "silu_gate"
      this.performance_metrics["compute_shader_config"]["optimized_for"] = "llama"
      
    elif ($1) {
      # GPT-style optimizations
      this.hidden_size = 768  # Default for smaller GPT models
      this.num_heads = 12
      this.head_dim = this.hidden_size // this.num_heads
      
    }
      if ($1) ${$1} else {
        this.performance_metrics["compute_shader_config"]["attention_mechanism"]["algorithm"] = "causal_attention"
        this.performance_metrics["memory_reduction_percent"] = 24.0
        
      }
      this.performance_metrics["compute_shader_config"]["attention_mechanism"]["kv_cache_enabled"] = true
      this.performance_metrics["compute_shader_config"]["layer_norm"]["algorithm"] = "layer_norm"
      this.performance_metrics["compute_shader_config"]["mlp"]["algorithm"] = "fused_gelu"
      this.performance_metrics["compute_shader_config"]["optimized_for"] = "gpt"
      
    elif ($1) {
      # Next-gen large model optimizations
      this.hidden_size = 4096  # Default for larger GPT3-type models
      this.num_heads = 32
      this.head_dim = 128
      
    }
      if ($1) ${$1} else ${$1} else {
      # Generic transformer optimizations
      }
      if ($1) ${$1} else {
        this.performance_metrics["compute_shader_config"]["attention_mechanism"]["algorithm"] = "standard_attention"
        this.performance_metrics["memory_reduction_percent"] = 15.0
        
      }
      this.performance_metrics["compute_shader_config"]["layer_norm"]["algorithm"] = "standard_layernorm"
      this.performance_metrics["compute_shader_config"]["mlp"]["algorithm"] = "standard_mlp"
      this.performance_metrics["compute_shader_config"]["optimized_for"] = "generic"
      
    # Apply custom configuration if provided
    if ($1) {
      for key, value in Object.entries($1):
        if ($1) {
          setattr(self, key, value)
        elif ($1) {
          this.performance_metrics["compute_shader_config"]["workgroup_size"] = value
        elif ($1) {
          subkey = key.replace("attention_", "")
          this.performance_metrics["compute_shader_config"]["attention_mechanism"][subkey] = value
        elif ($1) {
          subkey = key.replace("layernorm_", "")
          this.performance_metrics["compute_shader_config"]["layer_norm"][subkey] = value
        elif ($1) {
          subkey = key.replace("mlp_", "")
          this.performance_metrics["compute_shader_config"]["mlp"][subkey] = value
          
        }
    # Calculate aligned workgroup size (optimal for GPU architecture)
        }
    workgroup_size = this.performance_metrics["compute_shader_config"]["workgroup_size"]
        }
    aligned_size = (workgroup_size + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
        }
    this.performance_metrics["compute_shader_config"]["aligned_workgroup_size"] = aligned_size
        }
    
    }
    # Add scaling factor for attention computation
    attention_config = this.performance_metrics["compute_shader_config"]["attention_mechanism"]
    attention_config["scale_factor"] = 1.0 / (this.head_dim ** 0.5)
    
    # For Flash Attention, ensure block_size is set
    if ($1) {
      attention_config["block_size"] = min(64, this.seq_length)
    
    }
    # Log what we've configured
    if ($1) ${$1} else ${$1} (seq_length=${$1})")
      
    return this.performance_metrics
  
  $1($2): $3 {
    """
    Simulate attention mechanism with compute shaders.
    
  }
    Returns:
      Estimated processing time in milliseconds
    """
    if ($1) {
      # Basic simulation without compute optimization
      return 80.0 * (this.seq_length / 512.0) * (this.num_heads / 12.0)
    
    }
    start_time = time.time()
    
    # Get configuration
    attention_config = this.performance_metrics["compute_shader_config"]["attention_mechanism"]
    algorithm = attention_config["algorithm"]
    workgroup_size = attention_config["workgroup_size"]
    kv_cache_enabled = attention_config.get("kv_cache_enabled", false)
    
    # Determine efficiency factor based on attention algorithm
    if ($1) {
      # Flash Attention is significantly more efficient, especially for longer sequences
      # Start with base efficiency && adjust for sequence length
      efficiency_factor = 0.35  # 65% improvement baseline
      
    }
      # Flash Attention has better scaling for longer sequences
      if ($1) {
        seq_scaling = min(0.8, 0.35 + 0.1 * (1.0 - 512.0 / this.seq_length))
        efficiency_factor = min(seq_scaling, 0.25)  # Cap at 75% improvement
        
      }
      # For causal models, Flash Attention is even more efficient
      if ($1) {
        efficiency_factor *= 0.9  # Additional 10% improvement
        
      }
      # Block size affects efficiency
      block_size = attention_config.get("block_size", 64)
      if ($1) {
        block_efficiency = 1.0 - (0.05 * min(1.0, (block_size - 64) / 64.0))
        efficiency_factor *= block_efficiency  # Up to 5% additional improvement
        
      }
    elif ($1) {
      efficiency_factor = 0.45  # 55% improvement
      window_size = attention_config.get("window_size", 256)
      # Adjust for window size
      if ($1) {
        efficiency_factor *= (1.0 + 0.1 * (1.0 - min(1.0, window_size / this.seq_length)))
    elif ($1) {
      efficiency_factor = 0.60  # 40% improvement
    elif ($1) {
      efficiency_factor = 0.65  # 35% improvement
    elif ($1) ${$1} else {  # standard_attention
    }
      efficiency_factor = 0.80  # 20% improvement
    
    }
    # KV cache provides additional speedup for inference
      }
    if ($1) {
      # For Flash Attention, KV cache is already efficiently handled
      if ($1) {
        efficiency_factor *= 0.75  # Additional 25% improvement
      
      }
    # Simulate compute shader execution for attention mechanism
    }
    # In a real implementation, this would be a WebGPU compute shader
    }
    simulation_time = 0.001 * this.seq_length * efficiency_factor * (this.num_heads / 12.0)
    
    # Flash Attention has better scaling with larger head dimensions
    if ($1) {
      head_dim_factor = 1.0 - 0.2 * min(1.0, (this.head_dim - 64) / 64.0)  # Up to 20% additional improvement
      simulation_time *= head_dim_factor
    
    }
    time.sleep(simulation_time)  # Simulated time
    
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    
    # Calculate detailed performance metrics
    base_time = 50.0 * (this.seq_length / 512.0) * (this.num_heads / 12.0)
    optimized_time = base_time * efficiency_factor
    
    # Adjust based on head dimensions
    head_factor = (this.head_dim / 64.0)
    if ($1) {
      # Flash Attention scales better with head dimensions
      head_factor = (this.head_dim / 64.0) ** 0.8  # Sublinear scaling
    
    }
    processing_time = optimized_time * head_factor
    
    # For Flash Attention, we want to ensure the improvements are reflected properly
    if ($1) {
      # Calculate estimated speedup over standard attention
      standard_time = 50.0 * (this.seq_length / 512.0) * (this.num_heads / 12.0) * (this.head_dim / 64.0)
      estimated_speedup = standard_time / processing_time
      this.performance_metrics["estimated_speedup"] = estimated_speedup
      
    }
      # Log the performance characteristics
      logger.debug(`$1`)
    
    this.performance_metrics["attention_time_ms"] = processing_time
    return processing_time
  
  $1($2): $3 {
    """
    Simulate layer normalization with compute shaders.
    
  }
    Returns:
      Estimated processing time in milliseconds
    """
    if ($1) {
      # Basic simulation without compute optimization
      return 10.0 * (this.hidden_size / 768.0)
    
    }
    start_time = time.time()
    
    # Get configuration
    layernorm_config = this.performance_metrics["compute_shader_config"]["layer_norm"]
    algorithm = layernorm_config["algorithm"]
    workgroup_size = layernorm_config["workgroup_size"]
    
    # Determine efficiency factor based on layernorm algorithm
    if ($1) {
      efficiency_factor = 0.50  # 50% improvement
    elif ($1) ${$1} else {  # standard_layernorm
    }
      efficiency_factor = 0.75  # 25% improvement
    
    # Simulate compute shader execution for layer normalization
    # In a real implementation, this would be a WebGPU compute shader
    time.sleep(0.0005 * (this.hidden_size / 768.0) * efficiency_factor)  # Simulated time
    
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    
    # Calculate detailed performance metrics
    base_time = 5.0 * (this.hidden_size / 768.0)
    optimized_time = base_time * efficiency_factor
    
    this.performance_metrics["layer_norm_time_ms"] = optimized_time
    return optimized_time
  
  $1($2): $3 {
    """
    Simulate MLP computation with compute shaders.
    
  }
    Returns:
      Estimated processing time in milliseconds
    """
    if ($1) {
      # Basic simulation without compute optimization
      return 30.0 * (this.hidden_size / 768.0) * (this.seq_length / 512.0)
    
    }
    start_time = time.time()
    
    # Get configuration
    mlp_config = this.performance_metrics["compute_shader_config"]["mlp"]
    algorithm = mlp_config["algorithm"]
    workgroup_size = mlp_config["workgroup_size"]
    
    # Determine efficiency factor based on MLP algorithm
    if ($1) {
      efficiency_factor = 0.55  # 45% improvement
    elif ($1) {
      efficiency_factor = 0.60  # 40% improvement
    elif ($1) ${$1} else {  # standard_mlp
    }
      efficiency_factor = 0.75  # 25% improvement
    
    }
    # Simulate compute shader execution for MLP computation
    # In a real implementation, this would be a WebGPU compute shader
    time.sleep(0.001 * (this.hidden_size / 768.0) * (this.seq_length / 512.0) * efficiency_factor)  # Simulated time
    
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    
    # Calculate detailed performance metrics
    base_time = 20.0 * (this.hidden_size / 768.0) * (this.seq_length / 512.0)
    optimized_time = base_time * efficiency_factor
    
    this.performance_metrics["mlp_time_ms"] = optimized_time
    return optimized_time
  
  def process_transformer_layer(self, $1: number = 0) -> Dict[str, Any]:
    """
    Process a transformer layer with optimized compute shaders.
    
    Args:
      layer_idx: Index of the transformer layer
      
    Returns:
      Dictionary with performance metrics
    """
    # Simulate processing pipeline
    attention_time = this.simulate_attention_mechanism()
    layernorm_time = this.simulate_layer_normalization()
    mlp_time = this.simulate_mlp_computation()
    total_time = attention_time + layernorm_time + mlp_time
    
    # Update performance metrics
    this.performance_metrics["attention_time_ms"] = attention_time
    this.performance_metrics["layer_norm_time_ms"] = layernorm_time
    this.performance_metrics["mlp_time_ms"] = mlp_time
    this.performance_metrics["total_compute_time_ms"] = total_time
    
    # Calculate estimated speedup compared to non-compute shader implementation
    non_optimized_time = (80.0 * (this.seq_length / 512.0) * (this.num_heads / 12.0)) + \
              (10.0 * (this.hidden_size / 768.0)) + \
              (30.0 * (this.hidden_size / 768.0) * (this.seq_length / 512.0))
    
    speedup = non_optimized_time / total_time if total_time > 0 else 1.0
    this.performance_metrics["estimated_speedup"] = speedup
    
    logger.info(`$1`)
    return this.performance_metrics
  
  def generate_compute_shader_code(self, $1: string = "attention") -> Dict[str, str]:
    """
    Generate WebGPU compute shader code for a specific transformer component.
    
    Args:
      component: Component to generate code for ('attention', 'layernorm', 'mlp')
      
    Returns:
      Dictionary with shader code && metadata
    """
    shader_code = {
      "shader_code": "",
      "entry_point": "",
      "bind_groups": [],
      "metadata": {}
    }
    }
    
    if ($1) {
      # Get attention configuration
      config = this.performance_metrics["compute_shader_config"]["attention_mechanism"]
      algorithm = config["algorithm"]
      workgroup_size = config["workgroup_size"]
      
    }
      # Generate appropriate shader code based on algorithm
      if ($1) {
        shader_code["shader_code"] = this._generate_flash_attention_shader(workgroup_size)
        shader_code["entry_point"] = "main_flash_attention"
        # Add more metadata specific to Flash Attention
        shader_code["metadata"] = ${$1}
      elif ($1) {
        shader_code["shader_code"] = this._generate_sliding_window_attention_shader(workgroup_size)
        shader_code["entry_point"] = "main_sliding_window_attention"
        shader_code["metadata"] = ${$1}
      elif ($1) {
        shader_code["shader_code"] = this._generate_causal_attention_shader(workgroup_size)
        shader_code["entry_point"] = "main_causal_attention"
        shader_code["metadata"] = ${$1}
      } else {
        shader_code["shader_code"] = this._generate_standard_attention_shader(workgroup_size)
        shader_code["entry_point"] = "main_standard_attention"
        shader_code["metadata"] = ${$1}
      
      }
      # Add bind groups information
      }
      shader_code["bind_groups"] = [
      }
        ${$1},
        ${$1},
        ${$1},
        ${$1},
        ${$1}
      ]
      }
      
    elif ($1) {
      # Get layernorm configuration
      config = this.performance_metrics["compute_shader_config"]["layer_norm"]
      algorithm = config["algorithm"]
      workgroup_size = config["workgroup_size"]
      
    }
      # Generate appropriate shader code based on algorithm
      if ($1) ${$1} else {
        shader_code["shader_code"] = this._generate_layernorm_shader(workgroup_size)
        shader_code["entry_point"] = "main_layer_norm"
        
      }
      shader_code["metadata"] = ${$1}
      
      # Add bind groups information
      shader_code["bind_groups"] = [
        ${$1},
        ${$1},
        ${$1},
        ${$1},
        ${$1}
      ]
      
    elif ($1) {
      # Get MLP configuration
      config = this.performance_metrics["compute_shader_config"]["mlp"]
      algorithm = config["algorithm"]
      workgroup_size = config["workgroup_size"]
      
    }
      # Generate appropriate shader code based on algorithm
      if ($1) {
        shader_code["shader_code"] = this._generate_silu_gate_mlp_shader(workgroup_size)
        shader_code["entry_point"] = "main_silu_gate"
        shader_code["metadata"] = ${$1}
      elif ($1) {
        shader_code["shader_code"] = this._generate_fused_gelu_mlp_shader(workgroup_size)
        shader_code["entry_point"] = "main_fused_gelu"
        shader_code["metadata"] = ${$1}
      } else {
        shader_code["shader_code"] = this._generate_standard_mlp_shader(workgroup_size)
        shader_code["entry_point"] = "main_standard_mlp"
        shader_code["metadata"] = ${$1}
      
      }
      # Add bind groups information based on algorithm type
      }
      if ($1) {
        shader_code["bind_groups"] = [
          ${$1},
          ${$1},
          ${$1},
          ${$1},
          ${$1},
          ${$1},
          ${$1},
          ${$1},
          ${$1}
        ]
      } else {
        shader_code["bind_groups"] = [
          ${$1},
          ${$1},
          ${$1},
          ${$1},
          ${$1},
          ${$1},
          ${$1}
        ]
        
      }
    return shader_code
      }
  
      }
  $1($2): $3 {
    """Generate shader code for Flash Attention algorithm.
    
  }
    Flash Attention is a more efficient attention implementation that 
    saves memory by using tiling && avoids materializing the full 
    attention matrix.
    """
    # Create shader template for flash attention
    shader = `$1`
    // Flash Attention Compute Shader for WebGPU
    // Model: ${$1}
    // Configuration: seq_length=${$1}, hidden_size=${$1}, heads=${$1}, head_dim=${$1}
    
    struct Params {${$1}};
    
    @group(0) @binding(0) var<storage, read> input_q: array<f32>;
    @group(0) @binding(1) var<storage, read> input_k: array<f32>;
    @group(0) @binding(2) var<storage, read> input_v: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: Params;
    
    // Shared memory for tiles
    var<workgroup> tile_q: array<f32, ${$1}>;
    var<workgroup> tile_k: array<f32, ${$1}>;
    var<workgroup> tile_v: array<f32, ${$1}>;
    var<workgroup> tile_s: array<f32, ${$1}>;
    
    // Accumulators
    var<workgroup> tile_m: array<f32, ${$1}>; // Max values for numerical stability
    var<workgroup> tile_l: array<f32, ${$1}>; // Scaling factors
    var<workgroup> tile_o: array<f32, ${$1}>; // Output accumulators
    
    // Helper functions
    fn softmax_scale(x: f32, m: f32, l: f32) -> f32 {${$1}}
    
    @compute @workgroup_size(${$1}, 1, 1)
    fn main_flash_attention(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>,
      @builtin(workgroup_id) workgroup_id: vec3<u32>
    ) {{
      let seq_idx = global_id.x; // Token index in sequence
      let head_idx = global_id.y; // Attention head index
      let batch_idx = global_id.z; // Batch index
      
    }
      // Early exit if out of bounds
      if (seq_idx >= params.seq_length || head_idx >= params.num_heads || batch_idx >= params.batch_size) {${$1}}
      
      // Initialize accumulators for this token position
      var m_i = -1e30f; // Max value (-infinity)
      var l_i = 0.0f;   // Scaling factor
      var o_i = array<f32, ${$1}>();  // Output accumulator
      
      // Initialize output to zeros
      for (var d = 0u; d < params.head_dim; d++) {${$1}}
      
      // Load Q vector for current token into local memory
      let q_offset = (batch_idx * params.num_heads * params.seq_length + 
            head_idx * params.seq_length + 
            seq_idx) * params.head_dim;
      
      for (var d = 0u; d < params.head_dim; d++) {${$1}}
      
      // Process in blocks
      let num_blocks = (params.seq_length + params.block_size - 1u) / params.block_size;
      
      for (var block_idx = 0u; block_idx < num_blocks; block_idx++) {{
        let block_start = block_idx * params.block_size;
        let block_end = min(block_start + params.block_size, params.seq_length);
        
      }
        // Skip this block if using causal attention && current token comes before this block
        if (params.causal == 1u && seq_idx < block_start) {${$1}}
        
        // First, compute S = Q * K^T for this block
        
        // Step 1: Load K for this block into shared memory
        workgroupBarrier();
        if (local_id.x < block_end - block_start) {{
          let k_token_idx = block_start + local_id.x;
          let k_offset = (batch_idx * params.num_heads * params.seq_length + 
                head_idx * params.seq_length + 
                k_token_idx) * params.head_dim;
          
        }
          // Load K vector
          for (var d = 0u; d < params.head_dim; d++) {${$1}}
          
          // Also load V vector
          let v_offset = (batch_idx * params.num_heads * params.seq_length + 
                head_idx * params.seq_length + 
                k_token_idx) * params.head_dim;
          
          for (var d = 0u; d < params.head_dim; d++) {${$1}}
        }}
        workgroupBarrier();
        
        // Step 2: Compute attention scores for this block (Q * K^T)
        for (var j = 0u; j < block_end - block_start; j++) {{
          let k_token_idx = block_start + j;
          
        }
          // Skip if using causal attention && k_token_idx > seq_idx
          if (params.causal == 1u && k_token_idx > seq_idx) {${$1}}
          
          // Compute dot product of Q && K
          var score = 0.0f;
          for (var d = 0u; d < params.head_dim; d++) {${$1}}
          
          // Apply scaling
          score *= params.scale_factor;
          
          // Step 3: Update running maximum && scaling factor
          let m_ij = max(m_i, score);
          let l_ij = l_i * exp(m_i - m_ij) + exp(score - m_ij);
          
          // Step 4: Update the output using the online softmax algorithm
          for (var d = 0u; d < params.head_dim; d++) {${$1}}
          
          // Update running accumulators
          m_i = m_ij;
          l_i = l_ij;
        }}
      }}
      
      // Normalize the output
      if (l_i > 0.0) {{
        for (var d = 0u; d < params.head_dim; d++) {${$1}}
      }}
      }
      
      // Write the output
      let output_offset = (batch_idx * params.num_heads * params.seq_length + 
              head_idx * params.seq_length + 
              seq_idx) * params.head_dim;
      
      for (var d = 0u; d < params.head_dim; d++) {${$1}}
    }}
    """
    return shader
    
  $1($2): $3 {
    """Generate shader code for sliding window attention."""
    window_size = this.performance_metrics["compute_shader_config"]["attention_mechanism"].get("window_size", 256)
    
  }
    # Create shader template for sliding window attention
    shader = `$1`
    // Sliding Window Attention Compute Shader for WebGPU
    // Model: ${$1}
    // Configuration: seq_length=${$1}, hidden_size=${$1}, heads=${$1}, head_dim=${$1}
    
    struct Params {${$1}};
    
    @group(0) @binding(0) var<storage, read> input_q: array<f32>;
    @group(0) @binding(1) var<storage, read> input_k: array<f32>;
    @group(0) @binding(2) var<storage, read> input_v: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: Params;
    
    var<workgroup> tile_q: array<array<f32, ${$1}>, ${$1}>;
    var<workgroup> tile_k: array<array<f32, ${$1}>, ${$1}>;
    var<workgroup> tile_v: array<array<f32, ${$1}>, ${$1}>;
    
    @compute @workgroup_size(${$1}, 1, 1)
    fn main_sliding_window_attention(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>,
      @builtin(workgroup_id) workgroup_id: vec3<u32>
    ) {{
      let seq_pos = global_id.x;
      let head_idx = global_id.y;
      let batch_idx = global_id.z;
      
    }
      if (seq_pos >= params.seq_length || head_idx >= params.num_heads) {${$1}}
      
      // Sliding window attention implementation
      let window_start = max(0, i32(seq_pos) - i32(params.window_size) / 2);
      let window_end = min(params.seq_length, seq_pos + params.window_size / 2);
      
      // Initialize output accumulators
      var attn_scores: array<f32, ${$1}>;
      var max_score = -1e30; // Negative infinity for numerical stability
      var sum_exp = 0.0;
      
      // Load query vector for current position
      var q_vec: array<f32, ${$1}>;
      let q_offset = (batch_idx * params.num_heads * params.seq_length + head_idx * params.seq_length + seq_pos) * params.head_dim;
      
      for (var d = 0u; d < params.head_dim; d++) {${$1}}
      
      // Compute attention scores for tokens in the sliding window
      for (var j = u32(window_start); j < u32(window_end); j++) {{
        // Get the key vector for position j
        let k_offset = (batch_idx * params.num_heads * params.seq_length + head_idx * params.seq_length + j) * params.head_dim;
        
      }
        // Compute dot product
        var score = 0.0;
        for (var d = 0u; d < params.head_dim; d++) {${$1}}
        
        // Apply scaling
        score *= params.scale_factor;
        
        // Store score && track max for numerical stability
        attn_scores[j - u32(window_start)] = score;
        max_score = max(max_score, score);
      }}
      
      // Apply softmax to get attention weights
      for (var j = u32(window_start); j < u32(window_end); j++) {${$1}}
      
      // Normalize attention weights
      if (sum_exp > 0.0) {{
        for (var j = u32(window_start); j < u32(window_end); j++) {${$1}}
      }}
      }
      
      // Apply attention to values && accumulate
      var output_vec: array<f32, ${$1}>;
      for (var d = 0u; d < params.head_dim; d++) {${$1}}
      
      for (var j = u32(window_start); j < u32(window_end); j++) {{
        let v_offset = (batch_idx * params.num_heads * params.seq_length + head_idx * params.seq_length + j) * params.head_dim;
        
      }
        for (var d = 0u; d < params.head_dim; d++) {${$1}}
      }}
      
      // Store results
      let output_idx = (batch_idx * params.num_heads * params.seq_length + head_idx * params.seq_length + seq_pos) * params.head_dim;
      
      for (var d = 0u; d < params.head_dim; d++) {${$1}}
    }}
    """
    return shader
  
  $1($2): $3 {
    """Generate shader code for causal attention."""
    # Create shader template for causal attention
    shader = `$1`
    // Causal Attention Compute Shader for WebGPU
    // Model: ${$1}
    // Configuration: seq_length=${$1}, hidden_size=${$1}, heads=${$1}, head_dim=${$1}
    
  }
    struct Params {${$1}};
    
    @group(0) @binding(0) var<storage, read> input_q: array<f32>;
    @group(0) @binding(1) var<storage, read> input_k: array<f32>;
    @group(0) @binding(2) var<storage, read> input_v: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: Params;
    
    var<workgroup> tile_q: array<array<f32, ${$1}>, ${$1}>;
    var<workgroup> tile_k: array<array<f32, ${$1}>, ${$1}>;
    var<workgroup> tile_v: array<array<f32, ${$1}>, ${$1}>;
    
    @compute @workgroup_size(${$1}, 1, 1)
    fn main_causal_attention(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>,
      @builtin(workgroup_id) workgroup_id: vec3<u32>
    ) {{
      let seq_pos = global_id.x;
      let head_idx = global_id.y;
      let batch_idx = global_id.z;
      
    }
      if (seq_pos >= params.seq_length || head_idx >= params.num_heads) {${$1}}
      
      // Causal attention implementation (only attend to previous tokens)
      // ... compute causal attention ...
      
      // Store results
      let output_idx = batch_idx * params.seq_length * params.num_heads * params.head_dim +
            head_idx * params.seq_length * params.head_dim +
            seq_pos * params.head_dim;
      
      // Store results to output tensor
      // ...
    }}
    """
    return shader
  
  $1($2): $3 {
    """Generate shader code for standard attention."""
    # Create shader template for standard attention
    shader = `$1`
    // Standard Attention Compute Shader for WebGPU
    // Model: ${$1}
    // Configuration: seq_length=${$1}, hidden_size=${$1}, heads=${$1}, head_dim=${$1}
    
  }
    struct Params {${$1}};
    
    @group(0) @binding(0) var<storage, read> input_q: array<f32>;
    @group(0) @binding(1) var<storage, read> input_k: array<f32>;
    @group(0) @binding(2) var<storage, read> input_v: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: Params;
    
    var<workgroup> tile_q: array<array<f32, ${$1}>, ${$1}>;
    var<workgroup> tile_k: array<array<f32, ${$1}>, ${$1}>;
    var<workgroup> tile_v: array<array<f32, ${$1}>, ${$1}>;
    
    @compute @workgroup_size(${$1}, 1, 1)
    fn main_standard_attention(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>,
      @builtin(workgroup_id) workgroup_id: vec3<u32>
    ) {{
      let seq_pos = global_id.x;
      let head_idx = global_id.y;
      let batch_idx = global_id.z;
      
    }
      if (seq_pos >= params.seq_length || head_idx >= params.num_heads) {${$1}}
      
      // Standard attention implementation
      // ... compute standard attention ...
      
      // Store results
      let output_idx = batch_idx * params.seq_length * params.num_heads * params.head_dim +
            head_idx * params.seq_length * params.head_dim +
            seq_pos * params.head_dim;
      
      // Store results to output tensor
      // ...
    }}
    """
    return shader
  
  $1($2): $3 {
    """Generate shader code for layer normalization."""
    # Create shader template for layer normalization
    shader = `$1`
    // Layer Normalization Compute Shader for WebGPU
    // Model: ${$1}
    // Configuration: hidden_size=${$1}
    
  }
    struct Params {${$1}};
    
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> gamma: array<f32>;
    @group(0) @binding(2) var<storage, read> beta: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: Params;
    
    var<workgroup> partial_sum: array<f32, ${$1}>;
    var<workgroup> partial_sq_sum: array<f32, ${$1}>;
    
    @compute @workgroup_size(${$1}, 1, 1)
    fn main_layer_norm(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>,
      @builtin(workgroup_id) workgroup_id: vec3<u32>
    ) {{
      let token_idx = workgroup_id.x;
      let batch_idx = workgroup_id.y;
      let hidden_idx = local_id.x;
      
    }
      if (token_idx >= params.seq_length || batch_idx >= params.batch_size) {${$1}}
      
      // Layer normalization implementation
      // ... compute layer normalization ...
      
      // Store results
      let output_idx = batch_idx * params.seq_length * params.hidden_size +
            token_idx * params.hidden_size +
            hidden_idx;
      
      // Store results to output tensor
      // ...
    }}
    """
    return shader
  
  $1($2): $3 {
    """Generate shader code for RMS normalization."""
    # Create shader template for RMS normalization
    shader = `$1`
    // RMS Normalization Compute Shader for WebGPU
    // Model: ${$1}
    // Configuration: hidden_size=${$1}
    
  }
    struct Params {${$1}};
    
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;
    @group(0) @binding(3) var<uniform> params: Params;
    
    var<workgroup> partial_sq_sum: array<f32, ${$1}>;
    
    @compute @workgroup_size(${$1}, 1, 1)
    fn main_rms_norm(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>,
      @builtin(workgroup_id) workgroup_id: vec3<u32>
    ) {{
      let token_idx = workgroup_id.x;
      let batch_idx = workgroup_id.y;
      let hidden_idx = local_id.x;
      
    }
      if (token_idx >= params.seq_length || batch_idx >= params.batch_size) {${$1}}
      
      // RMS normalization implementation
      // ... compute RMS normalization ...
      
      // Store results
      let output_idx = batch_idx * params.seq_length * params.hidden_size +
            token_idx * params.hidden_size +
            hidden_idx;
      
      // Store results to output tensor
      // ...
    }}
    """
    return shader
  
  $1($2): $3 {
    """Generate shader code for standard MLP."""
    # Create shader template for standard MLP
    shader = `$1`
    // Standard MLP Compute Shader for WebGPU
    // Model: ${$1}
    // Configuration: hidden_size=${$1}
    
  }
    struct Params {${$1}};
    
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> fc1_weight: array<f32>;
    @group(0) @binding(2) var<storage, read> fc1_bias: array<f32>;
    @group(0) @binding(3) var<storage, read> fc2_weight: array<f32>;
    @group(0) @binding(4) var<storage, read> fc2_bias: array<f32>;
    @group(0) @binding(5) var<storage, read_write> output: array<f32>;
    @group(0) @binding(6) var<uniform> params: Params;
    
    @compute @workgroup_size(${$1}, 1, 1)
    fn main_standard_mlp(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>,
      @builtin(workgroup_id) workgroup_id: vec3<u32>
    ) {{
      let token_idx = global_id.x;
      let batch_idx = global_id.y;
      
    }
      if (token_idx >= params.seq_length || batch_idx >= params.batch_size) {${$1}}
      
      // Standard MLP implementation with activation
      // ... compute MLP with activation function ...
      
      // Store results
      let output_idx = batch_idx * params.seq_length * params.hidden_size +
            token_idx * params.hidden_size;
      
      // Store results to output tensor
      // ...
    }}
    """
    return shader
  
  $1($2): $3 {
    """Generate shader code for MLP with fused GELU activation."""
    # Create shader template for MLP with fused GELU
    shader = `$1`
    // MLP with Fused GELU Compute Shader for WebGPU
    // Model: ${$1}
    // Configuration: hidden_size=${$1}
    
  }
    struct Params {${$1}};
    
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> fc1_weight: array<f32>;
    @group(0) @binding(2) var<storage, read> fc1_bias: array<f32>;
    @group(0) @binding(3) var<storage, read> fc2_weight: array<f32>;
    @group(0) @binding(4) var<storage, read> fc2_bias: array<f32>;
    @group(0) @binding(5) var<storage, read_write> output: array<f32>;
    @group(0) @binding(6) var<uniform> params: Params;
    
    fn gelu(x: f32) -> f32 {${$1}}
    
    @compute @workgroup_size(${$1}, 1, 1)
    fn main_fused_gelu(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>,
      @builtin(workgroup_id) workgroup_id: vec3<u32>
    ) {{
      let token_idx = global_id.x;
      let batch_idx = global_id.y;
      
    }
      if (token_idx >= params.seq_length || batch_idx >= params.batch_size) {${$1}}
      
      // MLP with fused GELU implementation
      // ... compute MLP with fused GELU activation ...
      
      // Store results
      let output_idx = batch_idx * params.seq_length * params.hidden_size +
            token_idx * params.hidden_size;
      
      // Store results to output tensor
      // ...
    }}
    """
    return shader
  
  $1($2): $3 {
    """Generate shader code for MLP with SiLU gating."""
    # Create shader template for MLP with SiLU gating
    shader = `$1`
    // MLP with SiLU Gating Compute Shader for WebGPU
    // Model: ${$1}
    // Configuration: hidden_size=${$1}
    
  }
    struct Params {${$1}};
    
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> gate_weight: array<f32>;
    @group(0) @binding(2) var<storage, read> gate_bias: array<f32>;
    @group(0) @binding(3) var<storage, read> up_weight: array<f32>;
    @group(0) @binding(4) var<storage, read> up_bias: array<f32>;
    @group(0) @binding(5) var<storage, read> down_weight: array<f32>;
    @group(0) @binding(6) var<storage, read> down_bias: array<f32>;
    @group(0) @binding(7) var<storage, read_write> output: array<f32>;
    @group(0) @binding(8) var<uniform> params: Params;
    
    fn silu(x: f32) -> f32 {${$1}}
    
    @compute @workgroup_size(${$1}, 1, 1)
    fn main_silu_gate(
      @builtin(global_invocation_id) global_id: vec3<u32>,
      @builtin(local_invocation_id) local_id: vec3<u32>,
      @builtin(workgroup_id) workgroup_id: vec3<u32>
    ) {{
      let token_idx = global_id.x;
      let batch_idx = global_id.y;
      
    }
      if (token_idx >= params.seq_length || batch_idx >= params.batch_size) {${$1}}
      
      // MLP with SiLU gating implementation
      // ... compute MLP with SiLU gating ...
      
      // Store results
      let output_idx = batch_idx * params.seq_length * params.hidden_size +
            token_idx * params.hidden_size;
      
      // Store results to output tensor
      // ...
    }}
    """
    return shader


def setup_transformer_compute_shaders($1: string, $1: string = "bert", 
                  $1: number = 512,
                  $1: Record<$2, $3> = null) -> WebGPUTransformerComputeShaders:
  """
  Set up WebGPU compute shaders for transformer model processing.
  
  Args:
    model_name: Name of the model
    model_type: Type of transformer model (bert, t5, llama, gpt2)
    seq_length: Maximum sequence length
    config: Optional configuration parameters
    
  Returns:
    Configured WebGPUTransformerComputeShaders instance
  """
  # Create compute shader instance
  compute_shaders = WebGPUTransformerComputeShaders(model_name, seq_length)
  
  # Configure for specific model type
  compute_shaders.configure_for_model(model_type, config)
  
  return compute_shaders


def get_supported_transformer_models() -> List[str]:
  """
  Get list of transformer models with optimized compute shader support.
  
  Returns:
    List of supported model types
  """
  return ["bert", "t5", "llama", "llama2", "llama3", "gpt2", "gpt", "qwen2", "generic"]