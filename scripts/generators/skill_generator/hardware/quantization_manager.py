#!/usr/bin/env python3
"""
Unified Quantization Interface for HuggingFace Skill Generator

This module provides a unified interface for different quantization methods
across various hardware backends (CUDA, ROCm, CPU, OpenVINO, QNN).

Supported quantization methods:
- bitsandbytes (8-bit, 4-bit) - CUDA
- GPTQ (Group-wise Post-Training Quantization) - Multi-backend
- AWQ (Activation-aware Weight Quantization) - Multi-backend
- GGUF (llama.cpp format) - CPU optimized
- OpenVINO INT8 - Intel hardware
- QNN 8-bit - Qualcomm hardware
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantizationMethod(Enum):
    """Supported quantization methods."""
    NONE = "none"
    BITSANDBYTES_8BIT = "bitsandbytes_8bit"
    BITSANDBYTES_4BIT = "bitsandbytes_4bit"
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF = "gguf"
    OPENVINO_INT8 = "openvino_int8"
    QNN_8BIT = "qnn_8bit"
    DYNAMIC_INT8 = "dynamic_int8"  # PyTorch dynamic quantization


class QuantizationManager:
    """
    Unified quantization manager for cross-platform model quantization.
    
    This manager automatically selects the best quantization method based on:
    - Available hardware
    - Model architecture
    - Performance requirements
    - Memory constraints
    """
    
    def __init__(self, hardware_type: str = "cpu", model_type: str = "encoder-only"):
        """
        Initialize the quantization manager.
        
        Args:
            hardware_type: Target hardware (cpu, cuda, rocm, mps, openvino, qnn)
            model_type: Model architecture type
        """
        self.hardware_type = hardware_type.lower()
        self.model_type = model_type
        self.available_methods = self._detect_available_methods()
        
        # Compatibility matrix: hardware -> supported methods
        self.hardware_compatibility = {
            "cpu": [
                QuantizationMethod.NONE,
                QuantizationMethod.DYNAMIC_INT8,
                QuantizationMethod.GGUF
            ],
            "cuda": [
                QuantizationMethod.NONE,
                QuantizationMethod.BITSANDBYTES_8BIT,
                QuantizationMethod.BITSANDBYTES_4BIT,
                QuantizationMethod.GPTQ,
                QuantizationMethod.AWQ,
                QuantizationMethod.DYNAMIC_INT8
            ],
            "rocm": [
                QuantizationMethod.NONE,
                QuantizationMethod.GPTQ,
                QuantizationMethod.AWQ,
                QuantizationMethod.DYNAMIC_INT8
            ],
            "mps": [
                QuantizationMethod.NONE,
                QuantizationMethod.DYNAMIC_INT8
            ],
            "openvino": [
                QuantizationMethod.NONE,
                QuantizationMethod.OPENVINO_INT8
            ],
            "qnn": [
                QuantizationMethod.NONE,
                QuantizationMethod.QNN_8BIT
            ]
        }
        
        logger.info(f"Quantization Manager initialized for {hardware_type} hardware")
        logger.info(f"Available quantization methods: {[m.value for m in self.available_methods]}")
    
    def _detect_available_methods(self) -> List[QuantizationMethod]:
        """
        Detect which quantization methods are available on this system.
        
        Returns:
            List of available QuantizationMethod enum values
        """
        available = [QuantizationMethod.NONE]
        
        # Check for bitsandbytes (CUDA only)
        try:
            import bitsandbytes as bnb
            if self.hardware_type == "cuda":
                available.extend([
                    QuantizationMethod.BITSANDBYTES_8BIT,
                    QuantizationMethod.BITSANDBYTES_4BIT
                ])
                logger.info("bitsandbytes detected")
        except ImportError:
            pass
        
        # Check for auto-gptq
        try:
            import auto_gptq
            available.append(QuantizationMethod.GPTQ)
            logger.info("auto-gptq detected")
        except ImportError:
            pass
        
        # Check for autoawq
        try:
            import awq
            available.append(QuantizationMethod.AWQ)
            logger.info("autoawq detected")
        except ImportError:
            pass
        
        # Check for llama-cpp-python (GGUF)
        try:
            import llama_cpp
            available.append(QuantizationMethod.GGUF)
            logger.info("llama-cpp-python detected")
        except ImportError:
            pass
        
        # Check for OpenVINO
        try:
            import openvino
            if self.hardware_type == "openvino":
                available.append(QuantizationMethod.OPENVINO_INT8)
                logger.info("OpenVINO detected")
        except ImportError:
            pass
        
        # Dynamic quantization is always available with PyTorch
        try:
            import torch
            available.append(QuantizationMethod.DYNAMIC_INT8)
        except ImportError:
            pass
        
        return available
    
    def get_recommended_method(
        self,
        memory_budget_gb: Optional[float] = None,
        speed_priority: bool = True
    ) -> QuantizationMethod:
        """
        Get the recommended quantization method for current hardware and constraints.
        
        Args:
            memory_budget_gb: Available memory budget in GB
            speed_priority: Whether to prioritize speed over quality
        
        Returns:
            Recommended QuantizationMethod
        """
        compatible_methods = self.hardware_compatibility.get(
            self.hardware_type,
            [QuantizationMethod.NONE]
        )
        
        # Filter to only available methods
        available_compatible = [
            m for m in compatible_methods 
            if m in self.available_methods
        ]
        
        if not available_compatible:
            return QuantizationMethod.NONE
        
        # Priority ranking based on hardware and requirements
        if self.hardware_type == "cuda":
            if memory_budget_gb and memory_budget_gb < 8:
                # Very tight memory budget
                if QuantizationMethod.BITSANDBYTES_4BIT in available_compatible:
                    return QuantizationMethod.BITSANDBYTES_4BIT
            elif memory_budget_gb and memory_budget_gb < 16:
                # Moderate memory budget
                if QuantizationMethod.BITSANDBYTES_8BIT in available_compatible:
                    return QuantizationMethod.BITSANDBYTES_8BIT
            
            # Good memory, prioritize speed
            if speed_priority:
                if QuantizationMethod.AWQ in available_compatible:
                    return QuantizationMethod.AWQ
                if QuantizationMethod.GPTQ in available_compatible:
                    return QuantizationMethod.GPTQ
        
        elif self.hardware_type == "cpu":
            if QuantizationMethod.GGUF in available_compatible:
                return QuantizationMethod.GGUF
            if QuantizationMethod.DYNAMIC_INT8 in available_compatible:
                return QuantizationMethod.DYNAMIC_INT8
        
        elif self.hardware_type == "openvino":
            if QuantizationMethod.OPENVINO_INT8 in available_compatible:
                return QuantizationMethod.OPENVINO_INT8
        
        elif self.hardware_type == "qnn":
            if QuantizationMethod.QNN_8BIT in available_compatible:
                return QuantizationMethod.QNN_8BIT
        
        # Default: no quantization
        return QuantizationMethod.NONE
    
    def get_quantization_config(
        self,
        method: QuantizationMethod
    ) -> Dict[str, Any]:
        """
        Get the configuration dictionary for a specific quantization method.
        
        Args:
            method: Quantization method to configure
        
        Returns:
            Configuration dictionary for the quantization method
        """
        configs = {
            QuantizationMethod.NONE: {},
            
            QuantizationMethod.BITSANDBYTES_8BIT: {
                "load_in_8bit": True,
                "llm_int8_threshold": 6.0,
                "llm_int8_has_fp16_weight": False
            },
            
            QuantizationMethod.BITSANDBYTES_4BIT: {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            },
            
            QuantizationMethod.GPTQ: {
                "bits": 4,
                "group_size": 128,
                "damp_percent": 0.01,
                "desc_act": False
            },
            
            QuantizationMethod.AWQ: {
                "bits": 4,
                "group_size": 128,
                "zero_point": True
            },
            
            QuantizationMethod.GGUF: {
                "n_ctx": 2048,
                "n_batch": 512,
                "n_threads": os.cpu_count()
            },
            
            QuantizationMethod.OPENVINO_INT8: {
                "quantization_level": "INT8",
                "quantization_preset": "PERFORMANCE"
            },
            
            QuantizationMethod.QNN_8BIT: {
                "quantization_overrides": "int8",
                "use_dsp": True
            },
            
            QuantizationMethod.DYNAMIC_INT8: {
                "dtype": "qint8",
                "qconfig": "default"
            }
        }
        
        return configs.get(method, {})
    
    def load_quantized_model(
        self,
        model_id: str,
        method: Optional[QuantizationMethod] = None,
        **kwargs
    ):
        """
        Load a model with the specified quantization method.
        
        Args:
            model_id: HuggingFace model identifier
            method: Quantization method to use (auto-detected if None)
            **kwargs: Additional arguments to pass to model loading
        
        Returns:
            Loaded and quantized model
        """
        if method is None:
            method = self.get_recommended_method()
        
        logger.info(f"Loading model {model_id} with {method.value} quantization")
        
        quant_config = self.get_quantization_config(method)
        
        # Merge quantization config with provided kwargs
        load_kwargs = {**quant_config, **kwargs}
        
        try:
            if method == QuantizationMethod.BITSANDBYTES_8BIT or \
               method == QuantizationMethod.BITSANDBYTES_4BIT:
                return self._load_bitsandbytes_model(model_id, load_kwargs)
            
            elif method == QuantizationMethod.GPTQ:
                return self._load_gptq_model(model_id, load_kwargs)
            
            elif method == QuantizationMethod.AWQ:
                return self._load_awq_model(model_id, load_kwargs)
            
            elif method == QuantizationMethod.GGUF:
                return self._load_gguf_model(model_id, load_kwargs)
            
            elif method == QuantizationMethod.OPENVINO_INT8:
                return self._load_openvino_model(model_id, load_kwargs)
            
            elif method == QuantizationMethod.QNN_8BIT:
                return self._load_qnn_model(model_id, load_kwargs)
            
            elif method == QuantizationMethod.DYNAMIC_INT8:
                return self._load_dynamic_int8_model(model_id, load_kwargs)
            
            else:
                # No quantization
                return self._load_standard_model(model_id, load_kwargs)
        
        except Exception as e:
            logger.error(f"Failed to load quantized model: {e}")
            logger.info("Falling back to standard model loading")
            return self._load_standard_model(model_id, kwargs)
    
    def _load_bitsandbytes_model(self, model_id: str, config: Dict) -> Any:
        """Load model with bitsandbytes quantization."""
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        
        bnb_config = BitsAndBytesConfig(**config)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        return model
    
    def _load_gptq_model(self, model_id: str, config: Dict) -> Any:
        """Load model with GPTQ quantization."""
        from auto_gptq import AutoGPTQForCausalLM
        
        model = AutoGPTQForCausalLM.from_quantized(
            model_id,
            use_safetensors=True,
            device_map="auto",
            **config
        )
        return model
    
    def _load_awq_model(self, model_id: str, config: Dict) -> Any:
        """Load model with AWQ quantization."""
        from awq import AutoAWQForCausalLM
        
        model = AutoAWQForCausalLM.from_quantized(
            model_id,
            fuse_layers=True,
            **config
        )
        return model
    
    def _load_gguf_model(self, model_id: str, config: Dict) -> Any:
        """Load model in GGUF format using llama-cpp-python."""
        from llama_cpp import Llama
        
        # Assume model_id is a path to GGUF file
        model = Llama(
            model_path=model_id,
            **config
        )
        return model
    
    def _load_openvino_model(self, model_id: str, config: Dict) -> Any:
        """Load model with OpenVINO quantization."""
        from openvino.runtime import Core
        from optimum.intel import OVModelForCausalLM
        
        model = OVModelForCausalLM.from_pretrained(
            model_id,
            export=True,
            **config
        )
        return model
    
    def _load_qnn_model(self, model_id: str, config: Dict) -> Any:
        """Load model with QNN quantization."""
        # QNN requires model conversion - placeholder for now
        logger.warning("QNN quantization requires model conversion")
        return self._load_standard_model(model_id, config)
    
    def _load_dynamic_int8_model(self, model_id: str, config: Dict) -> Any:
        """Load model with PyTorch dynamic INT8 quantization."""
        import torch
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(model_id)
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model
    
    def _load_standard_model(self, model_id: str, config: Dict) -> Any:
        """Load model without quantization."""
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained(model_id, **config)
        return model
    
    def estimate_memory_savings(self, method: QuantizationMethod, model_size_gb: float) -> float:
        """
        Estimate memory savings for a quantization method.
        
        Args:
            method: Quantization method
            model_size_gb: Original model size in GB
        
        Returns:
            Estimated memory savings ratio (0.0 to 1.0)
        """
        savings_ratios = {
            QuantizationMethod.NONE: 0.0,
            QuantizationMethod.BITSANDBYTES_8BIT: 0.5,  # 50% savings
            QuantizationMethod.BITSANDBYTES_4BIT: 0.75,  # 75% savings
            QuantizationMethod.GPTQ: 0.75,
            QuantizationMethod.AWQ: 0.75,
            QuantizationMethod.GGUF: 0.7,
            QuantizationMethod.OPENVINO_INT8: 0.75,
            QuantizationMethod.QNN_8BIT: 0.75,
            QuantizationMethod.DYNAMIC_INT8: 0.5
        }
        
        ratio = savings_ratios.get(method, 0.0)
        return model_size_gb * ratio
    
    def get_supported_methods_for_hardware(self, hardware: str) -> List[QuantizationMethod]:
        """
        Get all supported quantization methods for a specific hardware type.
        
        Args:
            hardware: Hardware type (cpu, cuda, rocm, etc.)
        
        Returns:
            List of supported QuantizationMethod values
        """
        return self.hardware_compatibility.get(hardware.lower(), [QuantizationMethod.NONE])


def create_quantization_config(
    hardware_type: str,
    model_type: str,
    memory_budget_gb: Optional[float] = None
) -> Dict[str, Any]:
    """
    Convenience function to create a quantization configuration.
    
    Args:
        hardware_type: Target hardware
        model_type: Model architecture type
        memory_budget_gb: Available memory budget
    
    Returns:
        Quantization configuration dictionary
    """
    manager = QuantizationManager(hardware_type, model_type)
    method = manager.get_recommended_method(memory_budget_gb)
    config = manager.get_quantization_config(method)
    
    return {
        "method": method.value,
        "config": config,
        "hardware": hardware_type,
        "estimated_savings_gb": manager.estimate_memory_savings(method, memory_budget_gb or 0)
    }


if __name__ == "__main__":
    # Example usage
    print("=== Quantization Manager Examples ===\n")
    
    # CUDA example
    print("CUDA Hardware:")
    cuda_manager = QuantizationManager("cuda", "decoder-only")
    method = cuda_manager.get_recommended_method(memory_budget_gb=8.0)
    print(f"  Recommended method: {method.value}")
    print(f"  Config: {cuda_manager.get_quantization_config(method)}")
    print()
    
    # CPU example
    print("CPU Hardware:")
    cpu_manager = QuantizationManager("cpu", "encoder-only")
    method = cpu_manager.get_recommended_method()
    print(f"  Recommended method: {method.value}")
    print(f"  Config: {cpu_manager.get_quantization_config(method)}")
    print()
    
    # OpenVINO example
    print("OpenVINO Hardware:")
    ov_manager = QuantizationManager("openvino", "encoder-only")
    method = ov_manager.get_recommended_method()
    print(f"  Recommended method: {method.value}")
    print(f"  Config: {ov_manager.get_quantization_config(method)}")
