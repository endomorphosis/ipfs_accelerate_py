#!/usr/bin/env python3
"""
Memory Optimization for Multi-Model Execution Support.

This module provides advanced memory optimization capabilities for concurrent
execution of multiple AI models, enabling efficient memory usage through operation-level
analysis, tensor reuse patterns, and allocation strategy optimization.

Key features:
1. Model operation memory profiling
2. Memory reuse pattern detection and optimization
3. Inter-model operation fusion for memory efficiency
4. Tensor lifetime analysis and optimization
5. Memory allocation/deallocation strategy optimization
6. Memory pressure simulation and mitigation
"""

import os
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("predictive_performance.memory_optimization")

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class ModelOperation:
    """
    Represents a single operation in a model's computation graph.
    
    This class tracks memory requirements, dependencies, and execution details
    for a specific operation in a model's computation graph, enabling memory
    optimization at the operation level.
    """
    
    def __init__(
        self,
        name: str,
        op_type: str,
        model_name: str,
        inputs: List[str],
        outputs: List[str],
        input_shapes: Optional[List[List[int]]] = None,
        output_shapes: Optional[List[List[int]]] = None,
        memory_read: int = 0,
        memory_write: int = 0,
        execution_time: float = 0.0,
        is_shared: bool = False
    ):
        """
        Initialize an operation in the computation graph.
        
        Args:
            name: Operation name/identifier
            op_type: Operation type (matmul, conv2d, add, etc.)
            model_name: Name of the model this operation belongs to
            inputs: List of input tensor names
            outputs: List of output tensor names
            input_shapes: Shapes of input tensors
            output_shapes: Shapes of output tensors
            memory_read: Memory read in bytes
            memory_write: Memory write in bytes
            execution_time: Estimated execution time in milliseconds
            is_shared: Whether this operation can be shared with other models
        """
        self.name = name
        self.op_type = op_type
        self.model_name = model_name
        self.inputs = inputs
        self.outputs = outputs
        self.input_shapes = input_shapes or []
        self.output_shapes = output_shapes or []
        self.memory_read = memory_read
        self.memory_write = memory_write
        self.execution_time = execution_time
        self.is_shared = is_shared
        
        # Computed attributes
        self.start_time = 0.0
        self.end_time = 0.0
        self.dependencies = set()
        self.dependents = set()
        self.execution_order = -1
        self.memory_peak = max(memory_read, memory_write)
        self.can_fuse_with = set()  # Operations this can be fused with
    
    def __repr__(self) -> str:
        return f"ModelOperation({self.name}, {self.op_type}, model={self.model_name})"
    
    def calculate_memory_usage(self) -> int:
        """Calculate total memory usage for this operation."""
        if self.input_shapes and self.output_shapes:
            # Calculate based on shapes
            input_size = sum(np.prod(shape) * 4 for shape in self.input_shapes)  # Assume float32 (4 bytes)
            output_size = sum(np.prod(shape) * 4 for shape in self.output_shapes)
            return input_size + output_size
        else:
            # Use predefined values
            return self.memory_read + self.memory_write
    
    def is_compatible_for_fusion(self, other: 'ModelOperation') -> bool:
        """Check if this operation can be fused with another operation."""
        # Operations can be fused if they're the same type and either:
        # 1. They're from the same model and one's output is the other's input
        # 2. They're from different models but have the same shapes and are marked as shareable
        
        # Same type check
        if self.op_type != other.op_type:
            return False
        
        # Same model, dependency relationship
        if self.model_name == other.model_name:
            return any(output in other.inputs for output in self.outputs) or \
                   any(output in self.inputs for output in other.outputs)
        
        # Different models but shareable
        if self.is_shared and other.is_shared:
            # Check shape compatibility 
            if len(self.input_shapes) != len(other.input_shapes) or \
               len(self.output_shapes) != len(other.output_shapes):
                return False
                
            # Check input shapes match
            for self_shape, other_shape in zip(self.input_shapes, other.input_shapes):
                if self_shape != other_shape:
                    return False
                    
            # Check output shapes match
            for self_shape, other_shape in zip(self.output_shapes, other.output_shapes):
                if self_shape != other_shape:
                    return False
                    
            return True
        
        return False


class Tensor:
    """
    Represents a tensor in memory during model execution.
    
    This class tracks the lifetime and usage patterns of a tensor during the
    execution of one or more models, enabling memory optimization through
    tensor reuse and allocation planning.
    """
    
    def __init__(
        self,
        name: str,
        shape: List[int],
        dtype: str = "float32",
        model_name: str = "",
        is_input: bool = False,
        is_output: bool = False,
        is_constant: bool = False,
        is_intermediate: bool = True
    ):
        """
        Initialize a tensor.
        
        Args:
            name: Tensor name/identifier
            shape: Tensor shape
            dtype: Data type (default: float32)
            model_name: Name of the model this tensor belongs to
            is_input: Whether this is a model input tensor
            is_output: Whether this is a model output tensor
            is_constant: Whether this is a constant tensor (weights)
            is_intermediate: Whether this is an intermediate tensor
        """
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.model_name = model_name
        self.is_input = is_input
        self.is_output = is_output
        self.is_constant = is_constant
        self.is_intermediate = is_intermediate
        
        # Memory tracking
        self.size_bytes = self._calculate_size_bytes()
        
        # Lifetime tracking
        self.created_by = ""  # Operation that created this tensor
        self.consumed_by = []  # Operations that consume this tensor
        self.first_use_time = float('inf')
        self.last_use_time = 0.0
        self.reused_count = 0  # Number of times this tensor is reused
        
        # Memory management
        self.can_reuse = is_intermediate and not is_output
        self.memory_address = None  # Simulated memory address
        self.deallocated = False
        self.shared_with = []  # Other tensors sharing this memory
    
    def __repr__(self) -> str:
        return f"Tensor({self.name}, {self.shape}, model={self.model_name})"
    
    def _calculate_size_bytes(self) -> int:
        """Calculate tensor size in bytes based on shape and dtype."""
        # Calculate number of elements
        num_elements = np.prod(self.shape)
        
        # Map dtype to byte size
        dtype_sizes = {
            "float32": 4,
            "float16": 2, 
            "int32": 4,
            "int16": 2,
            "int8": 1,
            "uint8": 1,
            "bool": 1
        }
        element_size = dtype_sizes.get(self.dtype, 4)  # Default to float32 (4 bytes)
        
        return int(num_elements * element_size)
    
    def update_lifetime(self, op_time: float, is_first: bool = False):
        """Update the lifetime of this tensor based on operation time."""
        if is_first:
            self.first_use_time = min(self.first_use_time, op_time)
        else:
            self.last_use_time = max(self.last_use_time, op_time)
    
    def is_alive_at(self, time_point: float) -> bool:
        """Check if this tensor is alive at the given time point."""
        return self.first_use_time <= time_point <= self.last_use_time
    
    def overlaps_with(self, other: 'Tensor') -> bool:
        """Check if this tensor's lifetime overlaps with another tensor."""
        return max(self.first_use_time, other.first_use_time) <= min(self.last_use_time, other.last_use_time)
    
    def can_share_memory_with(self, other: 'Tensor') -> bool:
        """Check if this tensor can share memory with another tensor."""
        # Can't share if either tensor can't be reused
        if not self.can_reuse or not other.can_reuse:
            return False
        
        # Can't share if they're from the same model (unless sharing is explicitly allowed)
        if self.model_name == other.model_name and self.model_name != "":
            if self not in other.shared_with and other not in self.shared_with:
                return False
        
        # Can't share if lifetimes overlap
        if self.overlaps_with(other):
            return False
        
        # Can share if sizes are compatible
        return self.size_bytes <= other.size_bytes


class MemoryOptimizer:
    """
    Advanced memory optimization for multi-model execution.
    
    This class performs detailed memory optimization for concurrent execution
    of multiple models, including operation-level profiling, tensor lifetime
    analysis, memory reuse pattern detection, and allocation strategy optimization.
    """
    
    def __init__(
        self,
        model_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        enable_operation_fusion: bool = True,
        enable_tensor_reuse: bool = True,
        enable_pool_allocation: bool = True,
        aggressive_optimization: bool = False,
        memory_limit: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize the memory optimizer.
        
        Args:
            model_profiles: Dictionary mapping model names to their operation profiles
            enable_operation_fusion: Whether to enable operation fusion
            enable_tensor_reuse: Whether to enable tensor memory reuse
            enable_pool_allocation: Whether to enable memory pool allocation
            aggressive_optimization: Whether to use aggressive optimization (higher memory savings but higher risk)
            memory_limit: Memory limit in bytes (None for unlimited)
            verbose: Whether to enable verbose logging
        """
        self.model_profiles = model_profiles or {}
        self.enable_operation_fusion = enable_operation_fusion
        self.enable_tensor_reuse = enable_tensor_reuse
        self.enable_pool_allocation = enable_pool_allocation
        self.aggressive_optimization = aggressive_optimization
        self.memory_limit = memory_limit
        
        # Set logging level
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Data structures for optimization
        self.operations: Dict[str, ModelOperation] = {}  # All operations
        self.tensors: Dict[str, Tensor] = {}  # All tensors
        self.execution_order: List[str] = []  # Operation execution order
        self.operation_fusions: List[Tuple[str, str]] = []  # Pairs of operations to fuse
        self.tensor_reuse_groups: List[List[str]] = []  # Groups of tensors that can reuse memory
        self.memory_pools: Dict[str, List[str]] = {}  # Memory pools for tensor allocation
        
        # Memory statistics
        self.original_peak_memory = 0
        self.optimized_peak_memory = 0
        self.memory_savings = 0
        self.memory_overhead = 0
        
        logger.info(f"Memory Optimizer initialized "
                    f"(fusion={'enabled' if enable_operation_fusion else 'disabled'}, "
                    f"reuse={'enabled' if enable_tensor_reuse else 'disabled'}, "
                    f"pool={'enabled' if enable_pool_allocation else 'disabled'}, "
                    f"aggressive={'yes' if aggressive_optimization else 'no'})")
    
    def load_model_profile(self, model_name: str, profile_path: Optional[str] = None) -> bool:
        """
        Load a model's memory profile.
        
        Args:
            model_name: Name of the model to load
            profile_path: Path to JSON profile (None to use predefined profiles)
            
        Returns:
            Success status
        """
        if profile_path:
            try:
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                self.model_profiles[model_name] = profile
                logger.info(f"Loaded profile for {model_name} from {profile_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading profile for {model_name}: {e}")
                return False
        else:
            # Use predefined profiles based on model type
            if model_name in self.model_profiles:
                logger.info(f"Using existing profile for {model_name}")
                return True
                
            # Extract model type and size from name
            model_type = self._extract_model_type(model_name)
            model_size = self._extract_model_size(model_name)
            
            # Generate profile based on model type and size
            profile = self._generate_model_profile(model_name, model_type, model_size)
            if profile:
                self.model_profiles[model_name] = profile
                logger.info(f"Generated profile for {model_name} ({model_type}, {model_size})")
                return True
            else:
                logger.error(f"Failed to generate profile for {model_name}")
                return False
    
    def _extract_model_type(self, model_name: str) -> str:
        """Extract model type from model name."""
        model_name = model_name.lower()
        
        # Common model type patterns
        if any(name in model_name for name in ["bert", "roberta", "distilbert"]):
            return "text_embedding"
        elif any(name in model_name for name in ["t5", "gpt", "llama", "opt"]):
            return "text_generation"
        elif any(name in model_name for name in ["vit", "resnet", "efficientnet"]):
            return "vision"
        elif any(name in model_name for name in ["whisper", "wav2vec", "hubert"]):
            return "audio"
        elif any(name in model_name for name in ["clip", "blip", "llava"]):
            return "multimodal"
        else:
            return "unknown"
    
    def _extract_model_size(self, model_name: str) -> str:
        """Extract model size from model name."""
        model_name = model_name.lower()
        
        # Look for size indicators
        if any(size in model_name for size in ["tiny", "mini", "small", "xs"]):
            return "small"
        elif any(size in model_name for size in ["large", "huge", "xl", "-l"]):
            return "large"
        elif any(size in model_name for size in ["medium", "mid", "base", "uncased"]):
            return "base"
        else:
            return "base"  # Default to base
    
    def _generate_model_profile(self, model_name: str, model_type: str, model_size: str) -> Optional[Dict[str, Any]]:
        """Generate a model profile based on model type and size."""
        # Profile templates for different model types and sizes
        profile_templates = {
            "text_embedding": {
                "small": {"ops": 15, "tensors": 30, "memory_mb": 120},
                "base": {"ops": 25, "tensors": 50, "memory_mb": 400},
                "large": {"ops": 40, "tensors": 80, "memory_mb": 1200}
            },
            "text_generation": {
                "small": {"ops": 25, "tensors": 50, "memory_mb": 300},
                "base": {"ops": 40, "tensors": 80, "memory_mb": 800},
                "large": {"ops": 60, "tensors": 120, "memory_mb": 3000}
            },
            "vision": {
                "small": {"ops": 20, "tensors": 40, "memory_mb": 200},
                "base": {"ops": 30, "tensors": 60, "memory_mb": 500},
                "large": {"ops": 45, "tensors": 90, "memory_mb": 1000}
            },
            "audio": {
                "small": {"ops": 30, "tensors": 60, "memory_mb": 300},
                "base": {"ops": 45, "tensors": 90, "memory_mb": 800},
                "large": {"ops": 60, "tensors": 120, "memory_mb": 1500}
            },
            "multimodal": {
                "small": {"ops": 35, "tensors": 70, "memory_mb": 400},
                "base": {"ops": 55, "tensors": 110, "memory_mb": 900},
                "large": {"ops": 80, "tensors": 160, "memory_mb": 2000}
            }
        }
        
        # Default to unknown if not found
        if model_type not in profile_templates:
            model_type = "text_embedding"
        
        if model_size not in profile_templates[model_type]:
            model_size = "base"
        
        template = profile_templates[model_type][model_size]
        
        # Generate a synthetic profile
        profile = {
            "model_name": model_name,
            "model_type": model_type,
            "model_size": model_size,
            "operations": self._generate_synthetic_operations(model_name, template["ops"], model_type),
            "tensors": self._generate_synthetic_tensors(model_name, template["tensors"], template["memory_mb"], model_type),
            "memory_mb": template["memory_mb"]
        }
        
        return profile
    
    def _generate_synthetic_operations(self, model_name: str, count: int, model_type: str) -> List[Dict[str, Any]]:
        """Generate synthetic operations for a model profile."""
        operations = []
        op_types = self._get_common_operations(model_type)
        
        for i in range(count):
            op_type = op_types[i % len(op_types)]
            op_name = f"{model_name}_{op_type}_{i}"
            
            # Generate inputs and outputs
            inputs = [f"{model_name}_tensor_{i*2}", f"{model_name}_tensor_{i*2+1}"]
            outputs = [f"{model_name}_tensor_{(i+1)*2}"]
            
            # Generate shapes based on operation type and model type
            input_shapes, output_shapes = self._generate_shapes_for_op(op_type, model_type)
            
            # Calculate memory usage
            memory_read = sum(np.prod(shape) * 4 for shape in input_shapes)
            memory_write = sum(np.prod(shape) * 4 for shape in output_shapes)
            
            # Determine if operation can be shared
            is_shared = op_type in ["embedding", "layer_norm", "attention"] and i < count // 3
            
            operations.append({
                "name": op_name,
                "op_type": op_type,
                "inputs": inputs,
                "outputs": outputs,
                "input_shapes": input_shapes,
                "output_shapes": output_shapes,
                "memory_read": memory_read,
                "memory_write": memory_write,
                "execution_time": i * 0.5,  # Synthetic execution time
                "is_shared": is_shared
            })
        
        return operations
    
    def _get_common_operations(self, model_type: str) -> List[str]:
        """Get common operations for a model type."""
        common_ops = ["matmul", "add", "layer_norm", "softmax"]
        
        if model_type == "text_embedding" or model_type == "text_generation":
            return common_ops + ["embedding", "attention", "dropout", "gelu"]
        elif model_type == "vision":
            return common_ops + ["conv2d", "max_pool", "batch_norm", "relu"]
        elif model_type == "audio":
            return common_ops + ["conv1d", "gru", "lstm", "mel_scale"]
        elif model_type == "multimodal":
            return common_ops + ["embedding", "attention", "conv2d", "cross_attention"]
        else:
            return common_ops
    
    def _generate_shapes_for_op(self, op_type: str, model_type: str) -> Tuple[List[List[int]], List[List[int]]]:
        """Generate input and output shapes for an operation."""
        batch_size = 1
        
        if model_type == "text_embedding" or model_type == "text_generation":
            seq_len = 128
            hidden_dim = 768
            
            if op_type == "matmul":
                return [[batch_size, seq_len, hidden_dim], [batch_size, hidden_dim, hidden_dim]], [[batch_size, seq_len, hidden_dim]]
            elif op_type == "embedding":
                return [[batch_size, seq_len]], [[batch_size, seq_len, hidden_dim]]
            elif op_type == "attention":
                return [[batch_size, seq_len, hidden_dim]], [[batch_size, seq_len, hidden_dim]]
            else:
                return [[batch_size, seq_len, hidden_dim]], [[batch_size, seq_len, hidden_dim]]
                
        elif model_type == "vision":
            img_size = 224
            channels = 3
            
            if op_type == "conv2d":
                return [[batch_size, channels, img_size, img_size]], [[batch_size, 64, img_size//2, img_size//2]]
            elif op_type == "max_pool":
                return [[batch_size, 64, img_size//2, img_size//2]], [[batch_size, 64, img_size//4, img_size//4]]
            else:
                return [[batch_size, 768]], [[batch_size, 768]]
                
        else:
            # Default shapes
            return [[batch_size, 512]], [[batch_size, 512]]
    
    def _generate_synthetic_tensors(self, model_name: str, count: int, memory_mb: int, model_type: str) -> List[Dict[str, Any]]:
        """Generate synthetic tensors for a model profile."""
        tensors = []
        
        # Calculate average tensor size
        avg_tensor_size = (memory_mb * 1024 * 1024) / count
        
        for i in range(count):
            tensor_name = f"{model_name}_tensor_{i}"
            
            # Generate tensor characteristics
            is_input = i < 2  # First few tensors are inputs
            is_output = i >= count - 2  # Last few tensors are outputs
            is_constant = i % 4 == 0 and not is_input and not is_output  # Some tensors are weights
            is_intermediate = not (is_input or is_output or is_constant)
            
            # Generate shape and size
            shape = self._generate_tensor_shape(model_type, i, is_input, is_output, is_constant)
            
            tensors.append({
                "name": tensor_name,
                "shape": shape,
                "dtype": "float32",
                "is_input": is_input,
                "is_output": is_output,
                "is_constant": is_constant,
                "is_intermediate": is_intermediate,
                "created_by": f"{model_name}_op_{max(0, i-2)}",
                "consumed_by": [f"{model_name}_op_{min(count-1, i+1)}"]
            })
        
        return tensors
    
    def _generate_tensor_shape(self, model_type: str, index: int, is_input: bool, is_output: bool, is_constant: bool) -> List[int]:
        """Generate a tensor shape based on model type and tensor characteristics."""
        batch_size = 1
        
        if model_type == "text_embedding" or model_type == "text_generation":
            seq_len = 128
            hidden_dim = 768
            
            if is_input:
                return [batch_size, seq_len]
            elif is_output:
                return [batch_size, seq_len, hidden_dim]
            elif is_constant:
                return [hidden_dim, hidden_dim]
            else:
                return [batch_size, seq_len, hidden_dim]
                
        elif model_type == "vision":
            img_size = 224
            channels = 3
            
            if is_input:
                return [batch_size, channels, img_size, img_size]
            elif is_output:
                return [batch_size, 1000]
            elif is_constant:
                return [64, channels, 3, 3]  # Typical conv kernel
            else:
                # Gradually decrease spatial dimensions
                stage = min(4, index // 4)
                feature_size = img_size // (2 ** stage)
                features = 64 * (2 ** stage)
                return [batch_size, features, feature_size, feature_size]
                
        elif model_type == "audio":
            seq_len = 300
            features = 80
            
            if is_input:
                return [batch_size, features, seq_len]
            elif is_output:
                return [batch_size, 1000]
            else:
                # Gradually change dimensions
                if index % 3 == 0:
                    return [batch_size, features, seq_len]
                elif index % 3 == 1:
                    return [batch_size, 256, seq_len // 2]
                else:
                    return [batch_size, 512]
        
        # Default case
        return [batch_size, 512]
    
    def build_computation_graph(self, model_names: List[str]) -> bool:
        """
        Build a unified computation graph from multiple models.
        
        Args:
            model_names: List of model names to include in the graph
            
        Returns:
            Success status
        """
        # Clear existing graph
        self.operations = {}
        self.tensors = {}
        self.execution_order = []
        
        success = True
        
        # Load model profiles if needed
        for model_name in model_names:
            if model_name not in self.model_profiles:
                if not self.load_model_profile(model_name):
                    logger.error(f"Failed to load profile for {model_name}")
                    success = False
        
        # Build operations and tensors
        for model_name in model_names:
            if model_name in self.model_profiles:
                profile = self.model_profiles[model_name]
                
                # Create operations
                for op_data in profile.get("operations", []):
                    op = ModelOperation(
                        name=op_data["name"],
                        op_type=op_data["op_type"],
                        model_name=model_name,
                        inputs=op_data["inputs"],
                        outputs=op_data["outputs"],
                        input_shapes=op_data.get("input_shapes", []),
                        output_shapes=op_data.get("output_shapes", []),
                        memory_read=op_data.get("memory_read", 0),
                        memory_write=op_data.get("memory_write", 0),
                        execution_time=op_data.get("execution_time", 0.0),
                        is_shared=op_data.get("is_shared", False)
                    )
                    self.operations[op.name] = op
                
                # Create tensors
                for tensor_data in profile.get("tensors", []):
                    tensor = Tensor(
                        name=tensor_data["name"],
                        shape=tensor_data["shape"],
                        dtype=tensor_data.get("dtype", "float32"),
                        model_name=model_name,
                        is_input=tensor_data.get("is_input", False),
                        is_output=tensor_data.get("is_output", False),
                        is_constant=tensor_data.get("is_constant", False),
                        is_intermediate=tensor_data.get("is_intermediate", True)
                    )
                    
                    # Set creator and consumers
                    tensor.created_by = tensor_data.get("created_by", "")
                    tensor.consumed_by = tensor_data.get("consumed_by", [])
                    
                    self.tensors[tensor.name] = tensor
        
        # Build dependencies between operations
        for op_name, operation in self.operations.items():
            # Add input tensors as dependencies
            for input_name in operation.inputs:
                if input_name in self.tensors:
                    tensor = self.tensors[input_name]
                    
                    # Add consuming operation
                    if op_name not in tensor.consumed_by:
                        tensor.consumed_by.append(op_name)
                    
                    # Add creator as dependency
                    if tensor.created_by and tensor.created_by in self.operations:
                        operation.dependencies.add(tensor.created_by)
                        self.operations[tensor.created_by].dependents.add(op_name)
            
            # Set operation as creator of output tensors
            for output_name in operation.outputs:
                if output_name in self.tensors:
                    self.tensors[output_name].created_by = op_name
        
        # Build initial execution order (topological sort)
        if self.operations:
            self._build_execution_order()
            
            # Calculate original peak memory
            self.original_peak_memory = self._calculate_peak_memory()
            logger.info(f"Built computation graph with {len(self.operations)} operations, "
                      f"{len(self.tensors)} tensors, peak memory: {self.original_peak_memory / (1024*1024):.2f} MB")
        else:
            logger.warning("No operations loaded, computation graph is empty")
            success = False
            
        return success
    
    def _build_execution_order(self):
        """Build execution order using topological sort."""
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node):
            if node in temp_visited:
                # Cycle detected
                logger.error(f"Cycle detected in computation graph at node {node}")
                return False
            
            if node in visited:
                return True
            
            temp_visited.add(node)
            
            # Visit dependencies
            for dep in self.operations[node].dependencies:
                if not visit(dep):
                    return False
            
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
            return True
        
        # Process all nodes
        for node in self.operations:
            if node not in visited:
                if not visit(node):
                    # Cycle detected
                    logger.error("Cannot build execution order due to cycles in graph")
                    return
        
        # Reverse the order (topological sort produces reverse order)
        self.execution_order = list(reversed(order))
        
        # Assign execution order to operations
        for i, op_name in enumerate(self.execution_order):
            self.operations[op_name].execution_order = i
    
    def _calculate_peak_memory(self) -> int:
        """Calculate peak memory usage based on current execution order."""
        # Simulate memory allocation during execution
        active_tensors = set()
        current_memory = 0
        peak_memory = 0
        
        for op_name in self.execution_order:
            operation = self.operations[op_name]
            
            # Add input tensors to active set if not already there
            for input_name in operation.inputs:
                if input_name in self.tensors and input_name not in active_tensors:
                    tensor = self.tensors[input_name]
                    active_tensors.add(input_name)
                    current_memory += tensor.size_bytes
            
            # Calculate peak at this operation
            peak_memory = max(peak_memory, current_memory)
            
            # Release output tensors from previous operations that are no longer needed
            for tensor_name in list(active_tensors):
                tensor = self.tensors[tensor_name]
                
                # Check if this is the last operation that uses this tensor
                # and the tensor is not an output tensor
                if tensor.consumed_by and op_name == tensor.consumed_by[-1] and not tensor.is_output:
                    active_tensors.remove(tensor_name)
                    current_memory -= tensor.size_bytes
            
            # Add output tensors to active set
            for output_name in operation.outputs:
                if output_name in self.tensors:
                    tensor = self.tensors[output_name]
                    active_tensors.add(output_name)
                    current_memory += tensor.size_bytes
            
            # Update peak memory
            peak_memory = max(peak_memory, current_memory)
        
        return peak_memory
    
    def analyze_tensor_lifetimes(self):
        """Analyze tensor lifetimes based on execution order."""
        if not self.execution_order:
            logger.error("Execution order not built, cannot analyze tensor lifetimes")
            return
        
        # Assign start and end times to operations
        current_time = 0
        for op_name in self.execution_order:
            operation = self.operations[op_name]
            
            # Set start time
            operation.start_time = current_time
            
            # Set end time (use execution time or default to 1.0)
            execution_time = operation.execution_time if operation.execution_time > 0 else 1.0
            current_time += execution_time
            operation.end_time = current_time
        
        # Assign lifetimes to tensors
        for tensor_name, tensor in self.tensors.items():
            # Set tensor creation time
            if tensor.created_by and tensor.created_by in self.operations:
                creator = self.operations[tensor.created_by]
                tensor.first_use_time = creator.end_time
            else:
                # Input tensors have first use time of 0
                tensor.first_use_time = 0
            
            # Set last use time
            if tensor.consumed_by:
                # Find last consumer operation
                last_consumer = None
                latest_time = 0
                
                for consumer_name in tensor.consumed_by:
                    if consumer_name in self.operations:
                        consumer = self.operations[consumer_name]
                        if consumer.end_time > latest_time:
                            latest_time = consumer.end_time
                            last_consumer = consumer
                
                if last_consumer:
                    tensor.last_use_time = last_consumer.end_time
            
            # Ensure last_use_time is not before first_use_time
            tensor.last_use_time = max(tensor.last_use_time, tensor.first_use_time)
            
            logger.debug(f"Tensor {tensor_name} lifetime: {tensor.first_use_time:.2f} - {tensor.last_use_time:.2f}")
    
    def identify_operation_fusion_opportunities(self) -> List[Tuple[str, str]]:
        """
        Identify operations that can be fused to reduce memory usage.
        
        Returns:
            List of operation pairs (op1, op2) that can be fused
        """
        if not self.enable_operation_fusion:
            return []
        
        fusion_opportunities = []
        
        # Build a map of operations by type for efficient lookup
        ops_by_type = {}
        for op_name, operation in self.operations.items():
            if operation.op_type not in ops_by_type:
                ops_by_type[operation.op_type] = []
            ops_by_type[operation.op_type].append(op_name)
        
        # Check each pair of operations of the same type
        for op_type, ops_of_type in ops_by_type.items():
            for i, op1_name in enumerate(ops_of_type):
                op1 = self.operations[op1_name]
                
                for j in range(i + 1, len(ops_of_type)):
                    op2_name = ops_of_type[j]
                    op2 = self.operations[op2_name]
                    
                    # Check if operations can be fused
                    if op1.is_compatible_for_fusion(op2):
                        # Operations are compatible for fusion
                        fusion_opportunities.append((op1_name, op2_name))
                        
                        # Update operations
                        op1.can_fuse_with.add(op2_name)
                        op2.can_fuse_with.add(op1_name)
                        
                        logger.debug(f"Found fusion opportunity: {op1_name} and {op2_name}")
        
        self.operation_fusions = fusion_opportunities
        logger.info(f"Identified {len(fusion_opportunities)} operation fusion opportunities")
        
        return fusion_opportunities
    
    def identify_tensor_reuse_opportunities(self) -> List[List[str]]:
        """
        Identify tensors that can reuse the same memory.
        
        Returns:
            List of tensor groups that can share memory
        """
        if not self.enable_tensor_reuse or not self.tensors:
            return []
        
        # Analyze tensor lifetimes if not already done
        if any(tensor.first_use_time == float('inf') for tensor in self.tensors.values()):
            self.analyze_tensor_lifetimes()
        
        # Sort tensors by size (largest first) for better memory reuse
        sorted_tensors = sorted(
            [t for t in self.tensors.values() if t.can_reuse],
            key=lambda t: t.size_bytes,
            reverse=True
        )
        
        # Create a compatibility graph
        compatibility_graph = {}
        for tensor in sorted_tensors:
            compatibility_graph[tensor.name] = []
        
        # Find compatible pairs
        for i, tensor1 in enumerate(sorted_tensors):
            for j in range(i + 1, len(sorted_tensors)):
                tensor2 = sorted_tensors[j]
                
                if tensor1.can_share_memory_with(tensor2):
                    compatibility_graph[tensor1.name].append(tensor2.name)
                    compatibility_graph[tensor2.name].append(tensor1.name)
        
        # Find tensor groups for reuse using graph coloring
        reuse_groups = []
        unassigned = set(t.name for t in sorted_tensors)
        
        while unassigned:
            # Start a new group with the largest unassigned tensor
            current_group = []
            
            # Get largest unassigned tensor
            largest_tensor = max(
                [t for t in sorted_tensors if t.name in unassigned],
                key=lambda t: t.size_bytes
            )
            
            current_group.append(largest_tensor.name)
            unassigned.remove(largest_tensor.name)
            
            # Try to add compatible tensors to the group
            for tensor_name in list(unassigned):
                # Check if this tensor is compatible with all tensors in the current group
                compatible = True
                for group_tensor in current_group:
                    if tensor_name not in compatibility_graph[group_tensor]:
                        compatible = False
                        break
                
                if compatible:
                    current_group.append(tensor_name)
                    unassigned.remove(tensor_name)
            
            # Add the group if it has at least 2 tensors
            if len(current_group) >= 2:
                reuse_groups.append(current_group)
                
                # Update tensor shared_with links
                for i, tensor1_name in enumerate(current_group):
                    tensor1 = self.tensors[tensor1_name]
                    for j in range(i + 1, len(current_group)):
                        tensor2_name = current_group[j]
                        tensor2 = self.tensors[tensor2_name]
                        
                        tensor1.shared_with.append(tensor2)
                        tensor2.shared_with.append(tensor1)
        
        self.tensor_reuse_groups = reuse_groups
        logger.info(f"Identified {len(reuse_groups)} tensor reuse groups with {sum(len(g) for g in reuse_groups)} tensors")
        
        return reuse_groups
    
    def optimize_memory_allocation(self) -> Dict[str, Any]:
        """
        Optimize memory allocation for all tensors.
        
        Returns:
            Dictionary with optimization results
        """
        if not self.tensors or not self.operations:
            logger.error("No tensors or operations loaded, cannot optimize memory allocation")
            return {"success": False, "error": "No data"}
        
        # Analyze tensor lifetimes if not already done
        if any(tensor.first_use_time == float('inf') for tensor in self.tensors.values()):
            self.analyze_tensor_lifetimes()
        
        # Identify operation fusions if not already done
        if self.enable_operation_fusion and not self.operation_fusions:
            self.identify_operation_fusion_opportunities()
        
        # Identify tensor reuse opportunities if not already done
        if self.enable_tensor_reuse and not self.tensor_reuse_groups:
            self.identify_tensor_reuse_opportunities()
        
        # Apply optimizations
        memory_before = self.original_peak_memory
        
        # Apply operation fusion
        if self.enable_operation_fusion:
            self._apply_operation_fusion()
        
        # Apply tensor reuse
        if self.enable_tensor_reuse:
            self._apply_tensor_reuse()
        
        # Apply memory pool allocation
        if self.enable_pool_allocation:
            self._apply_memory_pool_allocation()
        
        # Calculate optimized peak memory
        self.optimized_peak_memory = self._calculate_optimized_peak_memory()
        
        # Calculate memory savings
        self.memory_savings = memory_before - self.optimized_peak_memory
        
        # Check if we're within memory limit
        memory_limit_ok = True
        if self.memory_limit is not None:
            memory_limit_ok = self.optimized_peak_memory <= self.memory_limit
        
        # Prepare optimization results
        optimization_results = {
            "success": True,
            "original_peak_memory": memory_before,
            "optimized_peak_memory": self.optimized_peak_memory,
            "memory_savings": self.memory_savings,
            "memory_savings_percent": (self.memory_savings / memory_before * 100) if memory_before > 0 else 0,
            "memory_limit_ok": memory_limit_ok,
            "operation_fusions": len(self.operation_fusions),
            "tensor_reuse_groups": len(self.tensor_reuse_groups),
            "total_tensors_reused": sum(len(g) for g in self.tensor_reuse_groups),
            "memory_pools": len(self.memory_pools)
        }
        
        logger.info(f"Memory optimization complete: {optimization_results['memory_savings_percent']:.2f}% savings "
                  f"({memory_before / (1024*1024):.2f} MB -> {self.optimized_peak_memory / (1024*1024):.2f} MB)")
        
        return optimization_results
    
    def _apply_operation_fusion(self):
        """Apply operation fusion optimizations."""
        if not self.operation_fusions:
            return
        
        # For each fusion opportunity, simulate fusion by adjusting memory
        for op1_name, op2_name in self.operation_fusions:
            if op1_name in self.operations and op2_name in self.operations:
                op1 = self.operations[op1_name]
                op2 = self.operations[op2_name]
                
                # Find shared tensors between operations
                shared_outputs = set(op1.outputs) & set(op2.inputs)
                
                # Adjust memory for shared tensors (they won't need to be materialized)
                for tensor_name in shared_outputs:
                    if tensor_name in self.tensors:
                        tensor = self.tensors[tensor_name]
                        
                        # Mark tensor as handled in fusion
                        tensor.memory_address = -1  # Special marker for fusion
                        
                        logger.debug(f"Tensor {tensor_name} eliminated by fusion of {op1_name} and {op2_name}")
    
    def _apply_tensor_reuse(self):
        """Apply tensor memory reuse optimizations."""
        if not self.tensor_reuse_groups:
            return
        
        # Simulate memory addresses (0 is for unmapped memory)
        next_address = 1
        
        # Assign addresses to tensor reuse groups
        for group in self.tensor_reuse_groups:
            # Find largest tensor in group
            largest_tensor = max(
                [self.tensors[name] for name in group],
                key=lambda t: t.size_bytes
            )
            
            # Assign address to all tensors in group
            address = next_address
            next_address += largest_tensor.size_bytes
            
            for tensor_name in group:
                self.tensors[tensor_name].memory_address = address
                
                logger.debug(f"Tensor {tensor_name} assigned to shared address {address}")
    
    def _apply_memory_pool_allocation(self):
        """Apply memory pool allocation optimizations."""
        if not self.enable_pool_allocation:
            return
        
        # Group tensors by size for pool allocation
        size_groups = {}
        
        for tensor_name, tensor in self.tensors.items():
            # Skip tensors already handled by reuse groups
            if tensor.memory_address is not None:
                continue
            
            # Skip output tensors if not aggressive
            if tensor.is_output and not self.aggressive_optimization:
                continue
                
            # Skip constant tensors if not aggressive
            if tensor.is_constant and not self.aggressive_optimization:
                continue
            
            # Round up to nearest power of 2 for better pooling
            size = tensor.size_bytes
            pool_size = 1
            while pool_size < size:
                pool_size *= 2
            
            if pool_size not in size_groups:
                size_groups[pool_size] = []
            
            size_groups[pool_size].append(tensor_name)
        
        # Create memory pools
        self.memory_pools = {}
        
        for size, tensors in size_groups.items():
            if len(tensors) < 2:
                continue  # No benefit to pooling
            
            # Group tensors by lifetime to maximize reuse
            timeline = []
            
            for tensor_name in tensors:
                tensor = self.tensors[tensor_name]
                timeline.append((tensor.first_use_time, True, tensor_name))  # Allocation
                timeline.append((tensor.last_use_time, False, tensor_name))  # Deallocation
            
            # Sort by time
            timeline.sort()
            
            # Simulate pool allocation
            free_slots = []
            tensor_to_slot = {}
            
            for time, is_alloc, tensor_name in timeline:
                if is_alloc:
                    # Allocate tensor
                    if free_slots:
                        # Reuse a free slot
                        slot = free_slots.pop(0)
                    else:
                        # Create a new slot
                        slot = len(tensor_to_slot)
                    
                    tensor_to_slot[tensor_name] = slot
                else:
                    # Free tensor
                    if tensor_name in tensor_to_slot:
                        free_slots.append(tensor_to_slot[tensor_name])
                        del tensor_to_slot[tensor_name]
            
            # Create pool if there's reuse
            max_slot = max([tensor_to_slot[t] for t in tensor_to_slot] + [0])
            
            if max_slot < len(tensors) - 1:
                pool_name = f"pool_{size}"
                self.memory_pools[pool_name] = tensors
                
                # Assign tensors to slots
                for tensor_name in tensors:
                    tensor = self.tensors[tensor_name]
                    
                    if tensor_name in tensor_to_slot:
                        # Use the slot as an offset within the pool
                        tensor.memory_address = -(size * tensor_to_slot[tensor_name] + 2)  # Negative for pools
                
                logger.debug(f"Created memory pool {pool_name} with {len(tensors)} tensors and {max_slot + 1} slots")
    
    def _calculate_optimized_peak_memory(self) -> int:
        """Calculate optimized peak memory usage after applying optimizations."""
        # Simulate memory allocation during execution with optimizations
        active_tensors = set()
        current_memory = 0
        peak_memory = 0
        
        # Track memory usage by address for reused tensors
        address_memory = {}
        
        for op_name in self.execution_order:
            operation = self.operations[op_name]
            
            # Add input tensors to active set if not already there
            for input_name in operation.inputs:
                if input_name in self.tensors and input_name not in active_tensors:
                    tensor = self.tensors[input_name]
                    
                    # Skip if part of a fusion
                    if tensor.memory_address == -1:
                        continue
                    
                    active_tensors.add(input_name)
                    
                    # Check if tensor has a shared memory address
                    if tensor.memory_address is not None and tensor.memory_address > 0:
                        # Only count memory once per address
                        if tensor.memory_address not in address_memory:
                            address_memory[tensor.memory_address] = tensor.size_bytes
                            current_memory += tensor.size_bytes
                    elif tensor.memory_address is not None and tensor.memory_address < -1:
                        # Pool allocation
                        pool_slot = -(tensor.memory_address + 2)
                        if pool_slot not in address_memory:
                            address_memory[pool_slot] = tensor.size_bytes
                            current_memory += tensor.size_bytes
                    else:
                        # Normal allocation
                        current_memory += tensor.size_bytes
            
            # Calculate peak at this operation
            peak_memory = max(peak_memory, current_memory)
            
            # Release output tensors from previous operations that are no longer needed
            for tensor_name in list(active_tensors):
                tensor = self.tensors[tensor_name]
                
                # Check if this is the last operation that uses this tensor
                # and the tensor is not an output tensor
                if tensor.consumed_by and op_name == tensor.consumed_by[-1] and not tensor.is_output:
                    active_tensors.remove(tensor_name)
                    
                    # Only free memory if it's not shared or this is the last tensor with this address
                    if tensor.memory_address is not None and tensor.memory_address > 0:
                        # Check if any other active tensor uses this address
                        other_active_with_same_address = False
                        for other_name in active_tensors:
                            other = self.tensors[other_name]
                            if other.memory_address == tensor.memory_address:
                                other_active_with_same_address = True
                                break
                        
                        if not other_active_with_same_address:
                            current_memory -= address_memory.pop(tensor.memory_address, 0)
                    elif tensor.memory_address is not None and tensor.memory_address < -1:
                        # Pool allocation
                        pool_slot = -(tensor.memory_address + 2)
                        
                        # Check if any other active tensor uses this slot
                        other_active_with_same_slot = False
                        for other_name in active_tensors:
                            other = self.tensors[other_name]
                            if other.memory_address == tensor.memory_address:
                                other_active_with_same_slot = True
                                break
                        
                        if not other_active_with_same_slot:
                            current_memory -= address_memory.pop(pool_slot, 0)
                    elif tensor.memory_address is None:
                        # Normal allocation
                        current_memory -= tensor.size_bytes
            
            # Add output tensors to active set
            for output_name in operation.outputs:
                if output_name in self.tensors:
                    tensor = self.tensors[output_name]
                    
                    # Skip if part of a fusion
                    if tensor.memory_address == -1:
                        continue
                    
                    active_tensors.add(output_name)
                    
                    # Check if tensor has a shared memory address
                    if tensor.memory_address is not None and tensor.memory_address > 0:
                        # Only count memory once per address
                        if tensor.memory_address not in address_memory:
                            address_memory[tensor.memory_address] = tensor.size_bytes
                            current_memory += tensor.size_bytes
                    elif tensor.memory_address is not None and tensor.memory_address < -1:
                        # Pool allocation
                        pool_slot = -(tensor.memory_address + 2)
                        if pool_slot not in address_memory:
                            address_memory[pool_slot] = tensor.size_bytes
                            current_memory += tensor.size_bytes
                    else:
                        # Normal allocation
                        current_memory += tensor.size_bytes
            
            # Update peak memory
            peak_memory = max(peak_memory, current_memory)
        
        return peak_memory
    
    def generate_memory_plan(self) -> Dict[str, Any]:
        """
        Generate a detailed memory allocation plan.
        
        Returns:
            Dictionary with memory allocation plan
        """
        if not self.tensors or not self.optimized_peak_memory:
            logger.error("No optimized memory plan available")
            return {"success": False, "error": "No optimized plan"}
        
        # Build a memory plan with all the optimizations
        memory_plan = {
            "success": True,
            "peak_memory": self.optimized_peak_memory,
            "peak_memory_mb": self.optimized_peak_memory / (1024 * 1024),
            "operation_count": len(self.operations),
            "tensor_count": len(self.tensors),
            "execution_order": self.execution_order,
            "tensor_allocations": [],
            "memory_reuse_groups": [],
            "memory_pools": [],
            "operation_fusions": []
        }
        
        # Add tensor allocations
        for tensor_name, tensor in self.tensors.items():
            allocation = {
                "name": tensor_name,
                "model": tensor.model_name,
                "size_bytes": tensor.size_bytes,
                "size_mb": tensor.size_bytes / (1024 * 1024),
                "type": "input" if tensor.is_input else "output" if tensor.is_output else "constant" if tensor.is_constant else "intermediate",
                "first_use_time": tensor.first_use_time,
                "last_use_time": tensor.last_use_time,
                "lifetime": tensor.last_use_time - tensor.first_use_time,
                "memory_address": tensor.memory_address,
                "shared": len(tensor.shared_with) > 0,
                "part_of_fusion": tensor.memory_address == -1
            }
            
            memory_plan["tensor_allocations"].append(allocation)
        
        # Add memory reuse groups
        for i, group in enumerate(self.tensor_reuse_groups):
            group_info = {
                "group_id": i,
                "tensors": group,
                "models": list(set(self.tensors[name].model_name for name in group)),
                "total_size_bytes": max(self.tensors[name].size_bytes for name in group),
                "saved_bytes": sum(self.tensors[name].size_bytes for name in group) - 
                              max(self.tensors[name].size_bytes for name in group),
                "tensors_count": len(group)
            }
            
            memory_plan["memory_reuse_groups"].append(group_info)
        
        # Add memory pools
        for pool_name, pool_tensors in self.memory_pools.items():
            pool_info = {
                "name": pool_name,
                "tensors": pool_tensors,
                "slot_size_bytes": int(pool_name.split("_")[1]),
                "models": list(set(self.tensors[name].model_name for name in pool_tensors)),
                "tensors_count": len(pool_tensors)
            }
            
            memory_plan["memory_pools"].append(pool_info)
        
        # Add operation fusions
        for op1_name, op2_name in self.operation_fusions:
            fusion_info = {
                "operations": [op1_name, op2_name],
                "models": [self.operations[op1_name].model_name, self.operations[op2_name].model_name],
                "type": self.operations[op1_name].op_type,
                "eliminated_tensors": list(set(self.operations[op1_name].outputs) & set(self.operations[op2_name].inputs))
            }
            
            memory_plan["operation_fusions"].append(fusion_info)
        
        return memory_plan
    
    def optimize_models(self, model_names: List[str]) -> Dict[str, Any]:
        """
        Complete memory optimization pipeline for multiple models.
        
        Args:
            model_names: List of model names to optimize
            
        Returns:
            Dictionary with optimization results
        """
        # Build the computation graph
        graph_success = self.build_computation_graph(model_names)
        if not graph_success:
            return {"success": False, "error": "Failed to build computation graph"}
        
        # Analyze tensor lifetimes
        self.analyze_tensor_lifetimes()
        
        # Identify optimization opportunities
        if self.enable_operation_fusion:
            self.identify_operation_fusion_opportunities()
        
        if self.enable_tensor_reuse:
            self.identify_tensor_reuse_opportunities()
        
        # Apply optimizations
        optimization_results = self.optimize_memory_allocation()
        
        # Generate memory plan
        memory_plan = self.generate_memory_plan()
        
        # Combine results
        results = {
            "success": optimization_results["success"],
            "model_names": model_names,
            "model_count": len(model_names),
            "original_peak_memory_mb": self.original_peak_memory / (1024 * 1024),
            "optimized_peak_memory_mb": self.optimized_peak_memory / (1024 * 1024),
            "memory_savings_mb": self.memory_savings / (1024 * 1024),
            "memory_savings_percent": optimization_results["memory_savings_percent"],
            "memory_limit_ok": optimization_results["memory_limit_ok"],
            "optimization_summary": {
                "operation_fusions": optimization_results["operation_fusions"],
                "tensor_reuse_groups": optimization_results["tensor_reuse_groups"],
                "total_tensors_reused": optimization_results["total_tensors_reused"],
                "memory_pools": optimization_results["memory_pools"]
            },
            "memory_plan": memory_plan
        }
        
        logger.info(f"Completed memory optimization for {len(model_names)} models: "
                  f"{results['memory_savings_percent']:.2f}% savings "
                  f"({results['original_peak_memory_mb']:.2f} MB -> {results['optimized_peak_memory_mb']:.2f} MB)")
        
        return results


# Example usage
if __name__ == "__main__":
    # Configure detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    logger.info("Memory Optimization Module Demo")
    
    # Create optimizer
    optimizer = MemoryOptimizer(
        enable_operation_fusion=True,
        enable_tensor_reuse=True,
        enable_pool_allocation=True,
        aggressive_optimization=False,
        verbose=True
    )
    
    # Define models to optimize
    model_names = [
        "bert-base-uncased",
        "vit-base-patch16-224",
        "t5-small"
    ]
    
    # Run optimization
    results = optimizer.optimize_models(model_names)
    
    # Print results
    logger.info("\nOptimization Summary:")
    logger.info(f"Models: {', '.join(model_names)}")
    logger.info(f"Original peak memory: {results['original_peak_memory_mb']:.2f} MB")
    logger.info(f"Optimized peak memory: {results['optimized_peak_memory_mb']:.2f} MB")
    logger.info(f"Memory savings: {results['memory_savings_mb']:.2f} MB ({results['memory_savings_percent']:.2f}%)")
    logger.info(f"Operation fusions: {results['optimization_summary']['operation_fusions']}")
    logger.info(f"Tensor reuse groups: {results['optimization_summary']['tensor_reuse_groups']} "
               f"(with {results['optimization_summary']['total_tensors_reused']} tensors)")
    logger.info(f"Memory pools: {results['optimization_summary']['memory_pools']}")
    logger.info("Memory Optimization Module Demo completed")