#!/usr/bin/env python3
"""
Comprehensive Model-Hardware Compatibility System
Advanced compatibility rules and optimization recommendations
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompatibilityLevel(Enum):
    """Hardware compatibility levels."""
    OPTIMAL = "optimal"       # Best performance, recommended
    COMPATIBLE = "compatible" # Works well, minor limitations
    LIMITED = "limited"       # Works but with constraints
    UNSUPPORTED = "unsupported" # Not supported

class ModelFamily(Enum):
    """Model family types."""
    TRANSFORMER_ENCODER = "transformer_encoder"  # BERT, RoBERTa
    TRANSFORMER_DECODER = "transformer_decoder"  # GPT, LLaMA
    TRANSFORMER_ENCODER_DECODER = "transformer_encoder_decoder"  # T5, BART
    CONVOLUTIONAL = "convolutional"  # ResNet, EfficientNet
    DIFFUSION = "diffusion"  # Stable Diffusion, DALL-E
    AUDIO = "audio"  # Whisper, WaveNet
    MULTIMODAL = "multimodal"  # CLIP, FLAMINGO

@dataclass
class ModelRequirements:
    """Hardware requirements for a specific model."""
    min_memory_gb: Dict[str, float]  # Hardware -> memory requirement
    recommended_memory_gb: Dict[str, float]
    min_compute_tflops: float
    memory_bandwidth_gbps: float
    supported_precisions: List[str]
    batch_size_scaling: Dict[str, float]  # Hardware -> optimal batch size
    sequence_length_limits: Dict[str, int]  # Hardware -> max sequence
    special_requirements: List[str]
    optimization_features: List[str]

@dataclass
class HardwareCapabilities:
    """Hardware platform capabilities."""
    name: str
    memory_capacity_gb: float
    memory_bandwidth_gbps: float
    compute_tflops: float
    supported_precisions: List[str]
    optimization_features: List[str]
    power_efficiency: float
    cost_efficiency: float
    deployment_complexity: int  # 1-10 scale
    web_compatible: bool
    mobile_compatible: bool

@dataclass
class CompatibilityResult:
    """Model-hardware compatibility assessment."""
    model_name: str
    hardware_type: str
    compatibility_level: CompatibilityLevel
    confidence_score: float  # 0-100
    performance_score: float  # 0-100
    memory_utilization: float  # 0-1
    optimal_batch_size: int
    optimal_precision: str
    limitations: List[str]
    optimizations: List[str]
    estimated_performance: Dict[str, float]

@dataclass
class OptimizationRecommendation:
    """Hardware-specific optimization recommendation."""
    technique: str
    description: str
    expected_improvement: float  # Percentage
    implementation_difficulty: int  # 1-10 scale
    hardware_specific: bool
    prerequisites: List[str]

class ComprehensiveModelHardwareCompatibility:
    """Advanced model-hardware compatibility assessment system."""
    
    # Comprehensive model definitions
    MODEL_DEFINITIONS = {
        "bert-tiny": {
            "family": ModelFamily.TRANSFORMER_ENCODER,
            "parameters": 4.4e6,
            "model_size_mb": 17.6,
            "requirements": ModelRequirements(
                min_memory_gb={
                    "cpu": 0.5, "cuda": 0.3, "mps": 0.4, "webnn": 0.6, 
                    "webgpu": 0.8, "rocm": 0.3, "openvino": 0.4, "qualcomm": 0.5
                },
                recommended_memory_gb={
                    "cpu": 1.0, "cuda": 1.0, "mps": 1.0, "webnn": 1.5, 
                    "webgpu": 2.0, "rocm": 1.0, "openvino": 1.0, "qualcomm": 1.0
                },
                min_compute_tflops=0.1,
                memory_bandwidth_gbps=5.0,
                supported_precisions=["fp32", "fp16", "int8"],
                batch_size_scaling={
                    "cpu": 16, "cuda": 64, "mps": 32, "webnn": 8,
                    "webgpu": 4, "rocm": 64, "openvino": 32, "qualcomm": 8
                },
                sequence_length_limits={
                    "cpu": 2048, "cuda": 4096, "mps": 2048, "webnn": 1024,
                    "webgpu": 512, "rocm": 4096, "openvino": 2048, "qualcomm": 1024
                },
                special_requirements=[],
                optimization_features=["attention_optimization", "token_pruning"]
            )
        },
        "bert-base": {
            "family": ModelFamily.TRANSFORMER_ENCODER,
            "parameters": 110e6,
            "model_size_mb": 440,
            "requirements": ModelRequirements(
                min_memory_gb={
                    "cpu": 2.0, "cuda": 1.5, "mps": 1.8, "webnn": 3.0,
                    "webgpu": 4.0, "rocm": 1.5, "openvino": 2.0, "qualcomm": 2.5
                },
                recommended_memory_gb={
                    "cpu": 4.0, "cuda": 4.0, "mps": 4.0, "webnn": 6.0,
                    "webgpu": 8.0, "rocm": 4.0, "openvino": 4.0, "qualcomm": 4.0
                },
                min_compute_tflops=0.5,
                memory_bandwidth_gbps=25.0,
                supported_precisions=["fp32", "fp16", "int8"],
                batch_size_scaling={
                    "cpu": 8, "cuda": 32, "mps": 16, "webnn": 4,
                    "webgpu": 2, "rocm": 32, "openvino": 16, "qualcomm": 4
                },
                sequence_length_limits={
                    "cpu": 1024, "cuda": 2048, "mps": 1024, "webnn": 512,
                    "webgpu": 256, "rocm": 2048, "openvino": 1024, "qualcomm": 512
                },
                special_requirements=["attention_mechanism"],
                optimization_features=["mixed_precision", "gradient_checkpointing"]
            )
        },
        "gpt2-small": {
            "family": ModelFamily.TRANSFORMER_DECODER,
            "parameters": 124e6,
            "model_size_mb": 496,
            "requirements": ModelRequirements(
                min_memory_gb={
                    "cpu": 3.0, "cuda": 2.0, "mps": 2.5, "webnn": 4.0,
                    "webgpu": 6.0, "rocm": 2.0, "openvino": 3.0, "qualcomm": 3.5
                },
                recommended_memory_gb={
                    "cpu": 6.0, "cuda": 6.0, "mps": 6.0, "webnn": 8.0,
                    "webgpu": 12.0, "rocm": 6.0, "openvino": 6.0, "qualcomm": 6.0
                },
                min_compute_tflops=0.8,
                memory_bandwidth_gbps=40.0,
                supported_precisions=["fp32", "fp16", "int8"],
                batch_size_scaling={
                    "cpu": 4, "cuda": 16, "mps": 8, "webnn": 2,
                    "webgpu": 1, "rocm": 16, "openvino": 8, "qualcomm": 2
                },
                sequence_length_limits={
                    "cpu": 1024, "cuda": 2048, "mps": 1024, "webnn": 512,
                    "webgpu": 256, "rocm": 2048, "openvino": 1024, "qualcomm": 512
                },
                special_requirements=["autoregressive_generation", "kv_cache"],
                optimization_features=["kv_caching", "speculative_decoding"]
            )
        },
        "llama-7b": {
            "family": ModelFamily.TRANSFORMER_DECODER,
            "parameters": 6.7e9,
            "model_size_mb": 26800,
            "requirements": ModelRequirements(
                min_memory_gb={
                    "cpu": 32.0, "cuda": 16.0, "mps": 24.0, "webnn": float('inf'),
                    "webgpu": float('inf'), "rocm": 16.0, "openvino": 32.0, "qualcomm": float('inf')
                },
                recommended_memory_gb={
                    "cpu": 64.0, "cuda": 32.0, "mps": 48.0, "webnn": float('inf'),
                    "webgpu": float('inf'), "rocm": 32.0, "openvino": 64.0, "qualcomm": float('inf')
                },
                min_compute_tflops=5.0,
                memory_bandwidth_gbps=200.0,
                supported_precisions=["fp32", "fp16", "int8", "int4"],
                batch_size_scaling={
                    "cpu": 1, "cuda": 4, "mps": 2, "webnn": 0,
                    "webgpu": 0, "rocm": 4, "openvino": 1, "qualcomm": 0
                },
                sequence_length_limits={
                    "cpu": 2048, "cuda": 4096, "mps": 2048, "webnn": 0,
                    "webgpu": 0, "rocm": 4096, "openvino": 2048, "qualcomm": 0
                },
                special_requirements=["large_context", "rotary_embeddings"],
                optimization_features=["model_parallelism", "quantization", "pruning"]
            )
        },
        "stable-diffusion": {
            "family": ModelFamily.DIFFUSION,
            "parameters": 860e6,
            "model_size_mb": 3440,
            "requirements": ModelRequirements(
                min_memory_gb={
                    "cpu": 12.0, "cuda": 8.0, "mps": 10.0, "webnn": 16.0,
                    "webgpu": 20.0, "rocm": 8.0, "openvino": 12.0, "qualcomm": 16.0
                },
                recommended_memory_gb={
                    "cpu": 24.0, "cuda": 16.0, "mps": 20.0, "webnn": 32.0,
                    "webgpu": 40.0, "rocm": 16.0, "openvino": 24.0, "qualcomm": 32.0
                },
                min_compute_tflops=2.0,
                memory_bandwidth_gbps=100.0,
                supported_precisions=["fp32", "fp16"],
                batch_size_scaling={
                    "cpu": 1, "cuda": 4, "mps": 2, "webnn": 1,
                    "webgpu": 1, "rocm": 4, "openvino": 1, "qualcomm": 1
                },
                sequence_length_limits={
                    "cpu": 77, "cuda": 77, "mps": 77, "webnn": 77,
                    "webgpu": 77, "rocm": 77, "openvino": 77, "qualcomm": 77
                },
                special_requirements=["unet_architecture", "cross_attention"],
                optimization_features=["attention_slicing", "cpu_offloading"]
            )
        },
        "resnet-50": {
            "family": ModelFamily.CONVOLUTIONAL,
            "parameters": 25.6e6,
            "model_size_mb": 102.4,
            "requirements": ModelRequirements(
                min_memory_gb={
                    "cpu": 1.0, "cuda": 0.5, "mps": 0.8, "webnn": 1.5,
                    "webgpu": 2.0, "rocm": 0.5, "openvino": 1.0, "qualcomm": 1.2
                },
                recommended_memory_gb={
                    "cpu": 2.0, "cuda": 2.0, "mps": 2.0, "webnn": 3.0,
                    "webgpu": 4.0, "rocm": 2.0, "openvino": 2.0, "qualcomm": 2.5
                },
                min_compute_tflops=0.3,
                memory_bandwidth_gbps=15.0,
                supported_precisions=["fp32", "fp16", "int8"],
                batch_size_scaling={
                    "cpu": 32, "cuda": 128, "mps": 64, "webnn": 16,
                    "webgpu": 8, "rocm": 128, "openvino": 64, "qualcomm": 16
                },
                sequence_length_limits={
                    "cpu": 224, "cuda": 224, "mps": 224, "webnn": 224,
                    "webgpu": 224, "rocm": 224, "openvino": 224, "qualcomm": 224
                },
                special_requirements=["convolution_layers"],
                optimization_features=["tensor_cores", "winograd_convolution"]
            )
        },
        "whisper-base": {
            "family": ModelFamily.AUDIO,
            "parameters": 74e6,
            "model_size_mb": 296,
            "requirements": ModelRequirements(
                min_memory_gb={
                    "cpu": 1.5, "cuda": 1.0, "mps": 1.3, "webnn": 2.5,
                    "webgpu": 3.0, "rocm": 1.0, "openvino": 1.5, "qualcomm": 2.0
                },
                recommended_memory_gb={
                    "cpu": 3.0, "cuda": 3.0, "mps": 3.0, "webnn": 5.0,
                    "webgpu": 6.0, "rocm": 3.0, "openvino": 3.0, "qualcomm": 4.0
                },
                min_compute_tflops=0.4,
                memory_bandwidth_gbps=20.0,
                supported_precisions=["fp32", "fp16", "int8"],
                batch_size_scaling={
                    "cpu": 4, "cuda": 16, "mps": 8, "webnn": 2,
                    "webgpu": 1, "rocm": 16, "openvino": 8, "qualcomm": 2
                },
                sequence_length_limits={
                    "cpu": 1500, "cuda": 3000, "mps": 1500, "webnn": 750,
                    "webgpu": 375, "rocm": 3000, "openvino": 1500, "qualcomm": 750
                },
                special_requirements=["audio_processing", "mel_spectrogram"],
                optimization_features=["beam_search", "streaming_inference"]
            )
        }
    }
    
    # Hardware platform definitions
    HARDWARE_DEFINITIONS = {
        "cpu": HardwareCapabilities(
            name="Multi-core CPU",
            memory_capacity_gb=64.0,
            memory_bandwidth_gbps=51.2,
            compute_tflops=0.5,
            supported_precisions=["fp32", "int8"],
            optimization_features=["simd", "vectorization", "multithreading"],
            power_efficiency=0.85,
            cost_efficiency=0.95,
            deployment_complexity=2,
            web_compatible=False,
            mobile_compatible=False
        ),
        "cuda": HardwareCapabilities(
            name="NVIDIA GPU",
            memory_capacity_gb=24.0,
            memory_bandwidth_gbps=717.0,
            compute_tflops=83.0,
            supported_precisions=["fp32", "fp16", "int8", "int4"],
            optimization_features=["tensor_cores", "cuda_graphs", "cudnn"],
            power_efficiency=0.92,
            cost_efficiency=0.70,
            deployment_complexity=4,
            web_compatible=False,
            mobile_compatible=False
        ),
        "mps": HardwareCapabilities(
            name="Apple Silicon",
            memory_capacity_gb=32.0,
            memory_bandwidth_gbps=200.0,
            compute_tflops=10.4,
            supported_precisions=["fp32", "fp16"],
            optimization_features=["unified_memory", "neural_engine", "metal_shaders"],
            power_efficiency=0.95,
            cost_efficiency=0.80,
            deployment_complexity=3,
            web_compatible=False,
            mobile_compatible=True
        ),
        "rocm": HardwareCapabilities(
            name="AMD GPU",
            memory_capacity_gb=16.0,
            memory_bandwidth_gbps=624.0,
            compute_tflops=37.3,
            supported_precisions=["fp32", "fp16", "int8"],
            optimization_features=["rocblas", "miopen", "hip"],
            power_efficiency=0.88,
            cost_efficiency=0.75,
            deployment_complexity=5,
            web_compatible=False,
            mobile_compatible=False
        ),
        "webgpu": HardwareCapabilities(
            name="Web GPU",
            memory_capacity_gb=4.0,
            memory_bandwidth_gbps=25.6,
            compute_tflops=2.0,
            supported_precisions=["fp32", "fp16"],
            optimization_features=["compute_shaders", "buffer_ops"],
            power_efficiency=0.70,
            cost_efficiency=0.90,
            deployment_complexity=6,
            web_compatible=True,
            mobile_compatible=True
        ),
        "webnn": HardwareCapabilities(
            name="Web Neural Network",
            memory_capacity_gb=8.0,
            memory_bandwidth_gbps=102.4,
            compute_tflops=15.0,
            supported_precisions=["fp16", "int8", "int4"],
            optimization_features=["npu_acceleration", "ai_ops"],
            power_efficiency=0.93,
            cost_efficiency=0.85,
            deployment_complexity=7,
            web_compatible=True,
            mobile_compatible=True
        ),
        "openvino": HardwareCapabilities(
            name="Intel OpenVINO",
            memory_capacity_gb=64.0,
            memory_bandwidth_gbps=76.8,
            compute_tflops=1.2,
            supported_precisions=["fp32", "fp16", "int8"],
            optimization_features=["graph_optimization", "post_training_quantization"],
            power_efficiency=0.90,
            cost_efficiency=0.88,
            deployment_complexity=4,
            web_compatible=False,
            mobile_compatible=False
        ),
        "qualcomm": HardwareCapabilities(
            name="Qualcomm Hexagon",
            memory_capacity_gb=12.0,
            memory_bandwidth_gbps=51.2,
            compute_tflops=12.0,
            supported_precisions=["fp16", "int8", "int4"],
            optimization_features=["hexagon_nn", "snpe", "qti_acceleration"],
            power_efficiency=0.94,
            cost_efficiency=0.85,
            deployment_complexity=8,
            web_compatible=False,
            mobile_compatible=True
        )
    }
    
    def __init__(self):
        """Initialize comprehensive compatibility system."""
        logger.info("Initializing comprehensive model-hardware compatibility system...")
        logger.info(f"Loaded {len(self.MODEL_DEFINITIONS)} model definitions")
        logger.info(f"Loaded {len(self.HARDWARE_DEFINITIONS)} hardware platforms")
    
    def assess_compatibility(
        self, 
        model_name: str, 
        hardware_type: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> CompatibilityResult:
        """Assess model-hardware compatibility comprehensively."""
        
        if model_name not in self.MODEL_DEFINITIONS:
            return self._create_generic_compatibility_result(model_name, hardware_type)
        
        if hardware_type not in self.HARDWARE_DEFINITIONS:
            return self._create_unsupported_result(model_name, hardware_type)
        
        model_def = self.MODEL_DEFINITIONS[model_name]
        hardware_def = self.HARDWARE_DEFINITIONS[hardware_type]
        constraints = constraints or {}
        
        # Assess compatibility level
        compatibility_level = self._determine_compatibility_level(
            model_def, hardware_def, constraints
        )
        
        # Calculate performance scores
        performance_score = self._calculate_performance_score(model_def, hardware_def)
        confidence_score = self._calculate_confidence_score(model_def, hardware_def)
        
        # Memory utilization
        memory_utilization = self._calculate_memory_utilization(model_def, hardware_def)
        
        # Optimal configurations
        optimal_batch_size = self._determine_optimal_batch_size(model_def, hardware_def)
        optimal_precision = self._determine_optimal_precision(model_def, hardware_def)
        
        # Identify limitations and optimizations
        limitations = self._identify_limitations(model_def, hardware_def, constraints)
        optimizations = self._generate_optimizations(model_def, hardware_def)
        
        # Estimate performance metrics
        estimated_performance = self._estimate_performance_metrics(
            model_def, hardware_def, optimal_batch_size, optimal_precision
        )
        
        return CompatibilityResult(
            model_name=model_name,
            hardware_type=hardware_type,
            compatibility_level=compatibility_level,
            confidence_score=confidence_score,
            performance_score=performance_score,
            memory_utilization=memory_utilization,
            optimal_batch_size=optimal_batch_size,
            optimal_precision=optimal_precision,
            limitations=limitations,
            optimizations=optimizations,
            estimated_performance=estimated_performance
        )
    
    def _determine_compatibility_level(
        self, 
        model_def: Dict[str, Any], 
        hardware_def: HardwareCapabilities,
        constraints: Dict[str, Any]
    ) -> CompatibilityLevel:
        """Determine compatibility level based on requirements."""
        
        requirements = model_def["requirements"]
        
        # Check memory requirements
        min_memory = requirements.min_memory_gb.get(hardware_def.name.lower().split()[0], float('inf'))
        if min_memory == float('inf'):
            return CompatibilityLevel.UNSUPPORTED
        
        if hardware_def.memory_capacity_gb < min_memory:
            return CompatibilityLevel.UNSUPPORTED
        
        recommended_memory = requirements.recommended_memory_gb.get(hardware_def.name.lower().split()[0], float('inf'))
        
        # Check compute requirements
        if hardware_def.compute_tflops < requirements.min_compute_tflops:
            return CompatibilityLevel.LIMITED
        
        # Check precision support
        common_precisions = set(requirements.supported_precisions) & set(hardware_def.supported_precisions)
        if not common_precisions:
            return CompatibilityLevel.UNSUPPORTED
        
        # Check memory bandwidth
        if hardware_def.memory_bandwidth_gbps < requirements.memory_bandwidth_gbps * 0.5:
            return CompatibilityLevel.LIMITED
        
        # Check special requirements
        web_required = any("web" in req for req in requirements.special_requirements)
        if web_required and not hardware_def.web_compatible:
            return CompatibilityLevel.UNSUPPORTED
        
        # Determine final level
        if (hardware_def.memory_capacity_gb >= recommended_memory and
            hardware_def.compute_tflops >= requirements.min_compute_tflops * 2 and
            hardware_def.memory_bandwidth_gbps >= requirements.memory_bandwidth_gbps):
            return CompatibilityLevel.OPTIMAL
        elif (hardware_def.memory_capacity_gb >= min_memory * 1.5 and
              hardware_def.compute_tflops >= requirements.min_compute_tflops * 1.2):
            return CompatibilityLevel.COMPATIBLE
        else:
            return CompatibilityLevel.LIMITED
    
    def _calculate_performance_score(
        self, model_def: Dict[str, Any], hardware_def: HardwareCapabilities
    ) -> float:
        """Calculate expected performance score (0-100)."""
        
        requirements = model_def["requirements"]
        
        # Memory score (0-30 points)
        min_memory = requirements.min_memory_gb.get(hardware_def.name.lower().split()[0], float('inf'))
        if min_memory == float('inf'):
            memory_score = 0
        else:
            memory_ratio = hardware_def.memory_capacity_gb / min_memory
            memory_score = min(30, memory_ratio * 10)
        
        # Compute score (0-40 points)
        compute_ratio = hardware_def.compute_tflops / max(requirements.min_compute_tflops, 0.1)
        compute_score = min(40, compute_ratio * 10)
        
        # Bandwidth score (0-20 points)
        bandwidth_ratio = hardware_def.memory_bandwidth_gbps / max(requirements.memory_bandwidth_gbps, 1.0)
        bandwidth_score = min(20, bandwidth_ratio * 10)
        
        # Efficiency score (0-10 points)
        efficiency_score = hardware_def.power_efficiency * 10
        
        total_score = memory_score + compute_score + bandwidth_score + efficiency_score
        return min(100, total_score)
    
    def _calculate_confidence_score(
        self, model_def: Dict[str, Any], hardware_def: HardwareCapabilities
    ) -> float:
        """Calculate confidence in compatibility assessment (0-100)."""
        
        base_confidence = 70  # Base confidence level
        
        # Increase confidence for well-supported combinations
        requirements = model_def["requirements"]
        hardware_key = hardware_def.name.lower().split()[0]
        
        if hardware_key in requirements.min_memory_gb:
            base_confidence += 15  # Direct support
        
        if set(requirements.supported_precisions) & set(hardware_def.supported_precisions):
            base_confidence += 10  # Precision compatibility
        
        # Reduce confidence for complex requirements
        if len(requirements.special_requirements) > 2:
            base_confidence -= 10
        
        if hardware_def.deployment_complexity > 6:
            base_confidence -= 5
        
        return min(100, max(0, base_confidence))
    
    def _calculate_memory_utilization(
        self, model_def: Dict[str, Any], hardware_def: HardwareCapabilities
    ) -> float:
        """Calculate expected memory utilization (0-1)."""
        
        requirements = model_def["requirements"]
        hardware_key = hardware_def.name.lower().split()[0]
        
        min_memory = requirements.min_memory_gb.get(hardware_key, model_def["model_size_mb"] / 1024 * 2)
        
        # Add activation memory estimation
        activation_memory = min_memory * 0.3  # Rough estimate
        total_memory = min_memory + activation_memory
        
        utilization = total_memory / hardware_def.memory_capacity_gb
        return min(1.0, utilization)
    
    def _determine_optimal_batch_size(
        self, model_def: Dict[str, Any], hardware_def: HardwareCapabilities
    ) -> int:
        """Determine optimal batch size for the configuration."""
        
        requirements = model_def["requirements"]
        hardware_key = hardware_def.name.lower().split()[0]
        
        base_batch = requirements.batch_size_scaling.get(hardware_key, 1)
        
        # Adjust based on memory constraints
        memory_utilization = self._calculate_memory_utilization(model_def, hardware_def)
        if memory_utilization > 0.8:
            base_batch = max(1, base_batch // 2)
        elif memory_utilization < 0.4:
            base_batch = min(base_batch * 2, 64)
        
        return base_batch
    
    def _determine_optimal_precision(
        self, model_def: Dict[str, Any], hardware_def: HardwareCapabilities
    ) -> str:
        """Determine optimal precision for the configuration."""
        
        requirements = model_def["requirements"]
        common_precisions = set(requirements.supported_precisions) & set(hardware_def.supported_precisions)
        
        if not common_precisions:
            return "fp32"  # Fallback
        
        # Preference order based on performance and accuracy
        preference_order = ["int8", "fp16", "fp32", "int4"]
        
        for precision in preference_order:
            if precision in common_precisions:
                # Check if hardware has good support for this precision
                if precision == "int8" and "tensor_cores" in hardware_def.optimization_features:
                    return precision
                elif precision == "fp16" and hardware_def.compute_tflops > 5.0:
                    return precision
                elif precision == "fp32":
                    return precision
        
        return list(common_precisions)[0]  # First available
    
    def _identify_limitations(
        self, 
        model_def: Dict[str, Any], 
        hardware_def: HardwareCapabilities,
        constraints: Dict[str, Any]
    ) -> List[str]:
        """Identify potential limitations and constraints."""
        
        limitations = []
        requirements = model_def["requirements"]
        hardware_key = hardware_def.name.lower().split()[0]
        
        # Memory limitations
        min_memory = requirements.min_memory_gb.get(hardware_key, float('inf'))
        if hardware_def.memory_capacity_gb < min_memory * 1.5:
            limitations.append(f"Limited memory capacity ({hardware_def.memory_capacity_gb}GB available)")
        
        # Compute limitations
        if hardware_def.compute_tflops < requirements.min_compute_tflops * 1.5:
            limitations.append(f"Limited compute performance ({hardware_def.compute_tflops} TFLOPS)")
        
        # Bandwidth limitations
        if hardware_def.memory_bandwidth_gbps < requirements.memory_bandwidth_gbps:
            limitations.append("Memory bandwidth may be a bottleneck")
        
        # Precision limitations
        if "fp16" not in hardware_def.supported_precisions and "fp16" in requirements.supported_precisions:
            limitations.append("No FP16 support, limited optimization options")
        
        # Sequence length limitations
        max_seq = requirements.sequence_length_limits.get(hardware_key, 512)
        if max_seq < 1024:
            limitations.append(f"Limited sequence length support (max {max_seq} tokens)")
        
        # Deployment limitations
        if hardware_def.deployment_complexity > 6:
            limitations.append("Complex deployment and setup requirements")
        
        # Web compatibility limitations
        if hardware_def.web_compatible and model_def["model_size_mb"] > 1000:
            limitations.append("Large model size may cause loading issues in web environments")
        
        return limitations
    
    def _generate_optimizations(
        self, model_def: Dict[str, Any], hardware_def: HardwareCapabilities
    ) -> List[str]:
        """Generate optimization recommendations."""
        
        optimizations = []
        requirements = model_def["requirements"]
        
        # Precision optimizations
        if "fp16" in hardware_def.supported_precisions and "fp16" in requirements.supported_precisions:
            optimizations.append("Use FP16 precision for 30-50% speed improvement")
        
        if "int8" in hardware_def.supported_precisions:
            optimizations.append("Apply INT8 quantization for 2-4x speed improvement")
        
        # Hardware-specific optimizations
        if "tensor_cores" in hardware_def.optimization_features:
            optimizations.append("Enable Tensor Core acceleration for matrix operations")
        
        if "unified_memory" in hardware_def.optimization_features:
            optimizations.append("Utilize unified memory architecture for efficiency")
        
        # Model-specific optimizations
        if model_def["family"] == ModelFamily.TRANSFORMER_DECODER:
            optimizations.append("Implement KV caching for generation tasks")
            optimizations.append("Use speculative decoding for faster generation")
        
        if model_def["family"] == ModelFamily.DIFFUSION:
            optimizations.append("Enable attention slicing for memory efficiency")
            optimizations.append("Use CPU offloading for large batch sizes")
        
        # Memory optimizations
        memory_utilization = self._calculate_memory_utilization(model_def, hardware_def)
        if memory_utilization > 0.7:
            optimizations.append("Enable gradient checkpointing to reduce memory usage")
            optimizations.append("Use model sharding for very large models")
        
        # Batch optimizations
        optimal_batch = self._determine_optimal_batch_size(model_def, hardware_def)
        if optimal_batch > 1:
            optimizations.append(f"Use batch size {optimal_batch} for optimal throughput")
        
        return optimizations[:8]  # Limit to top 8 recommendations
    
    def _estimate_performance_metrics(
        self, 
        model_def: Dict[str, Any], 
        hardware_def: HardwareCapabilities,
        batch_size: int,
        precision: str
    ) -> Dict[str, float]:
        """Estimate performance metrics for the configuration."""
        
        # Simplified performance estimation
        base_inference_time = model_def["parameters"] / (hardware_def.compute_tflops * 1e12) * 1000
        
        # Precision adjustments
        precision_multipliers = {"fp32": 1.0, "fp16": 0.6, "int8": 0.35, "int4": 0.2}
        precision_factor = precision_multipliers.get(precision, 1.0)
        
        # Batch size scaling
        batch_efficiency = min(1.0, math.log(batch_size + 1) / math.log(17))  # Diminishing returns
        
        # Final metrics
        inference_time_ms = base_inference_time * precision_factor / batch_efficiency
        throughput = batch_size / (inference_time_ms / 1000) if inference_time_ms > 0 else 0
        
        memory_usage_mb = (model_def["model_size_mb"] * precision_factor + 
                          model_def["model_size_mb"] * batch_size * 0.1)
        
        power_consumption_w = (hardware_def.power_efficiency * 
                              min(hardware_def.compute_tflops, base_inference_time / 100) * 50)
        
        return {
            "inference_time_ms": inference_time_ms,
            "throughput_samples_per_sec": throughput,
            "memory_usage_mb": memory_usage_mb,
            "power_consumption_w": power_consumption_w,
            "efficiency_score": throughput / power_consumption_w if power_consumption_w > 0 else 0
        }
    
    def _create_generic_compatibility_result(
        self, model_name: str, hardware_type: str
    ) -> CompatibilityResult:
        """Create generic result for unknown models."""
        
        return CompatibilityResult(
            model_name=model_name,
            hardware_type=hardware_type,
            compatibility_level=CompatibilityLevel.COMPATIBLE,
            confidence_score=60.0,
            performance_score=70.0,
            memory_utilization=0.5,
            optimal_batch_size=4,
            optimal_precision="fp32",
            limitations=["Unknown model, using generic estimates"],
            optimizations=["Consider model profiling for accurate optimization"],
            estimated_performance={
                "inference_time_ms": 50.0,
                "throughput_samples_per_sec": 20.0,
                "memory_usage_mb": 1000.0,
                "power_consumption_w": 100.0,
                "efficiency_score": 0.2
            }
        )
    
    def _create_unsupported_result(
        self, model_name: str, hardware_type: str
    ) -> CompatibilityResult:
        """Create result for unsupported hardware."""
        
        return CompatibilityResult(
            model_name=model_name,
            hardware_type=hardware_type,
            compatibility_level=CompatibilityLevel.UNSUPPORTED,
            confidence_score=90.0,
            performance_score=0.0,
            memory_utilization=0.0,
            optimal_batch_size=1,
            optimal_precision="fp32",
            limitations=[f"Hardware type '{hardware_type}' not supported"],
            optimizations=["Use supported hardware platform"],
            estimated_performance={}
        )
    
    def get_hardware_recommendations(
        self, 
        model_name: str, 
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, CompatibilityResult]]:
        """Get hardware recommendations ranked by compatibility."""
        
        constraints = constraints or {}
        recommendations = []
        
        for hardware_type in self.HARDWARE_DEFINITIONS:
            # Apply constraints
            hardware_def = self.HARDWARE_DEFINITIONS[hardware_type]
            
            if constraints.get("max_power_w") and hardware_def.power_efficiency * 300 > constraints["max_power_w"]:
                continue
            
            if constraints.get("web_required") and not hardware_def.web_compatible:
                continue
                
            if constraints.get("mobile_required") and not hardware_def.mobile_compatible:
                continue
            
            result = self.assess_compatibility(model_name, hardware_type, constraints)
            recommendations.append((hardware_type, result))
        
        # Sort by compatibility level and performance score
        def sort_key(item):
            hardware_type, result = item
            level_scores = {
                CompatibilityLevel.OPTIMAL: 4,
                CompatibilityLevel.COMPATIBLE: 3,
                CompatibilityLevel.LIMITED: 2,
                CompatibilityLevel.UNSUPPORTED: 1
            }
            return (level_scores[result.compatibility_level], result.performance_score)
        
        recommendations.sort(key=sort_key, reverse=True)
        return recommendations

def run_comprehensive_compatibility_demo():
    """Run comprehensive compatibility demonstration."""
    print("🚀 Comprehensive Model-Hardware Compatibility Demo")
    print("=" * 60)
    
    compatibility = ComprehensiveModelHardwareCompatibility()
    
    # Test models
    test_models = ["bert-tiny", "bert-base", "gpt2-small", "llama-7b"]
    
    for model in test_models[:2]:  # Test first 2 models
        print(f"\n📊 Compatibility Analysis: {model}")
        print("-" * 40)
        
        recommendations = compatibility.get_hardware_recommendations(model)
        
        for i, (hardware, result) in enumerate(recommendations[:5], 1):
            print(f"  {i}. {hardware:12} - {result.compatibility_level.value.upper()}")
            print(f"     Performance: {result.performance_score:5.1f}/100")
            print(f"     Confidence:  {result.confidence_score:5.1f}/100")
            print(f"     Optimal Config: batch={result.optimal_batch_size}, {result.optimal_precision}")
            if result.limitations:
                print(f"     Limitations: {result.limitations[0]}")
            if result.optimizations:
                print(f"     Optimization: {result.optimizations[0]}")
            print()
    
    print("✅ Comprehensive compatibility analysis complete!")
    return True

if __name__ == "__main__":
    success = run_comprehensive_compatibility_demo()
    exit(0 if success else 1)