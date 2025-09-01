#!/usr/bin/env python3
"""
Model-Hardware Compatibility Manager

This module provides enhanced rules for determining optimal hardware
for different model types, including memory requirements, performance
characteristics, and compatibility constraints.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)

class ModelSize(Enum):
    """Model size categories for memory estimation."""
    TINY = "tiny"       # < 50MB
    SMALL = "small"     # 50MB - 500MB
    MEDIUM = "medium"   # 500MB - 2GB
    LARGE = "large"     # 2GB - 10GB
    XLARGE = "xlarge"   # > 10GB

class PrecisionMode(Enum):
    """Supported precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"

@dataclass
class HardwareRequirements:
    """Hardware requirements for a specific configuration."""
    min_memory_gb: float
    recommended_memory_gb: float
    supported_precisions: List[PrecisionMode]
    performance_multiplier: float  # Relative to CPU baseline
    notes: Optional[str] = None

@dataclass
class ModelProfile:
    """Complete model profile with hardware compatibility."""
    model_family: str
    model_size: ModelSize
    memory_base_mb: float
    hardware_requirements: Dict[str, HardwareRequirements]
    web_compatible: bool = False
    mobile_compatible: bool = False
    edge_optimized: bool = False
    special_requirements: Optional[List[str]] = None

class ModelHardwareCompatibility:
    """
    Enhanced model-hardware compatibility engine.
    
    Provides sophisticated matching between models and hardware
    based on detailed requirements and performance characteristics.
    """
    
    def __init__(self):
        self.compatibility_rules = self._initialize_compatibility_rules()
        self.hardware_profiles = self._initialize_hardware_profiles()
    
    def _initialize_hardware_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Initialize hardware performance profiles."""
        return {
            "cpu": {
                "base_performance": 1.0,
                "memory_efficiency": 0.8,
                "supported_precisions": [PrecisionMode.FP32],
                "max_model_size_gb": 16,
                "parallel_capability": "thread_based",
                "web_compatible": False,
            },
            "cuda": {
                "base_performance": 10.0,
                "memory_efficiency": 0.9,
                "supported_precisions": [PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.INT8],
                "max_model_size_gb": 24,  # Typical consumer GPU
                "parallel_capability": "cuda_cores",
                "web_compatible": False,
                "requires_nvidia": True,
            },
            "rocm": {
                "base_performance": 8.0,
                "memory_efficiency": 0.9,
                "supported_precisions": [PrecisionMode.FP32, PrecisionMode.FP16],
                "max_model_size_gb": 16,
                "parallel_capability": "compute_units",
                "web_compatible": False,
                "requires_amd": True,
            },
            "mps": {
                "base_performance": 6.0,
                "memory_efficiency": 0.95,
                "supported_precisions": [PrecisionMode.FP32, PrecisionMode.FP16],
                "max_model_size_gb": 64,  # Unified memory
                "parallel_capability": "metal_performance_shaders",
                "web_compatible": False,
                "requires_apple_silicon": True,
            },
            "webnn": {
                "base_performance": 3.0,
                "memory_efficiency": 0.7,
                "supported_precisions": [PrecisionMode.FP32, PrecisionMode.INT8],
                "max_model_size_gb": 2,  # Browser limitations
                "parallel_capability": "webnn_api",
                "web_compatible": True,
                "browser_dependent": True,
            },
            "webgpu": {
                "base_performance": 4.0,
                "memory_efficiency": 0.75,
                "supported_precisions": [PrecisionMode.FP32, PrecisionMode.FP16],
                "max_model_size_gb": 4,  # Browser GPU memory
                "parallel_capability": "webgpu_shaders",
                "web_compatible": True,
                "browser_dependent": True,
            },
            "qualcomm": {
                "base_performance": 2.5,
                "memory_efficiency": 0.9,
                "supported_precisions": [PrecisionMode.INT8, PrecisionMode.INT4],
                "max_model_size_gb": 8,
                "parallel_capability": "hexagon_dsp",
                "web_compatible": False,
                "mobile_optimized": True,
            },
            "openvino": {
                "base_performance": 3.5,
                "memory_efficiency": 0.85,
                "supported_precisions": [PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.INT8],
                "max_model_size_gb": 12,
                "parallel_capability": "intel_optimization",
                "web_compatible": False,
                "intel_optimized": True,
            }
        }
    
    def _initialize_compatibility_rules(self) -> Dict[str, ModelProfile]:
        """Initialize comprehensive model compatibility rules."""
        
        rules = {}
        
        # BERT Family Models
        rules["bert"] = ModelProfile(
            model_family="bert",
            model_size=ModelSize.SMALL,
            memory_base_mb=440,  # bert-base-uncased
            web_compatible=True,
            hardware_requirements={
                "cpu": HardwareRequirements(
                    min_memory_gb=2, recommended_memory_gb=4,
                    supported_precisions=[PrecisionMode.FP32],
                    performance_multiplier=1.0,
                    notes="Good CPU performance for smaller sequences"
                ),
                "cuda": HardwareRequirements(
                    min_memory_gb=2, recommended_memory_gb=4,
                    supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.INT8],
                    performance_multiplier=8.0,
                    notes="Excellent GPU acceleration"
                ),
                "mps": HardwareRequirements(
                    min_memory_gb=2, recommended_memory_gb=4,
                    supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                    performance_multiplier=6.0,
                    notes="Great performance on Apple Silicon"
                ),
                "webnn": HardwareRequirements(
                    min_memory_gb=1, recommended_memory_gb=2,
                    supported_precisions=[PrecisionMode.FP32, PrecisionMode.INT8],
                    performance_multiplier=3.0,
                    notes="Works well in browsers with WebNN"
                ),
                "qualcomm": HardwareRequirements(
                    min_memory_gb=1, recommended_memory_gb=2,
                    supported_precisions=[PrecisionMode.INT8, PrecisionMode.INT4],
                    performance_multiplier=2.0,
                    notes="Mobile optimized with quantization"
                ),
            }
        )
        
        # GPT Family Models
        rules["gpt2"] = ModelProfile(
            model_family="gpt",
            model_size=ModelSize.MEDIUM,
            memory_base_mb=548,  # gpt2 base
            web_compatible=True,
            hardware_requirements={
                "cpu": HardwareRequirements(
                    min_memory_gb=4, recommended_memory_gb=8,
                    supported_precisions=[PrecisionMode.FP32],
                    performance_multiplier=1.0,
                    notes="Slow for generation, OK for embeddings"
                ),
                "cuda": HardwareRequirements(
                    min_memory_gb=2, recommended_memory_gb=4,
                    supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                    performance_multiplier=12.0,
                    notes="Excellent for text generation"
                ),
                "mps": HardwareRequirements(
                    min_memory_gb=2, recommended_memory_gb=4,
                    supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                    performance_multiplier=8.0,
                    notes="Very good performance on Apple Silicon"
                ),
            }
        )
        
        # LLaMA Family Models  
        rules["llama"] = ModelProfile(
            model_family="llama",
            model_size=ModelSize.LARGE,
            memory_base_mb=13000,  # 7B model
            web_compatible=False,
            hardware_requirements={
                "cpu": HardwareRequirements(
                    min_memory_gb=16, recommended_memory_gb=32,
                    supported_precisions=[PrecisionMode.FP32],
                    performance_multiplier=1.0,
                    notes="Very slow, requires large RAM"
                ),
                "cuda": HardwareRequirements(
                    min_memory_gb=8, recommended_memory_gb=16,
                    supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.INT8],
                    performance_multiplier=15.0,
                    notes="Requires high-end GPU for good performance"
                ),
                "mps": HardwareRequirements(
                    min_memory_gb=8, recommended_memory_gb=16,
                    supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                    performance_multiplier=10.0,
                    notes="Good with unified memory architecture"
                ),
            },
            special_requirements=["high_memory", "gpu_recommended"]
        )
        
        # Vision Transformer Models
        rules["vit"] = ModelProfile(
            model_family="vit",
            model_size=ModelSize.MEDIUM,
            memory_base_mb=632,  # vit-base-patch16-224
            web_compatible=True,
            hardware_requirements={
                "cpu": HardwareRequirements(
                    min_memory_gb=4, recommended_memory_gb=8,
                    supported_precisions=[PrecisionMode.FP32],
                    performance_multiplier=1.0,
                    notes="Adequate for inference"
                ),
                "cuda": HardwareRequirements(
                    min_memory_gb=2, recommended_memory_gb=4,
                    supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                    performance_multiplier=10.0,
                    notes="Excellent for batch processing"
                ),
                "mps": HardwareRequirements(
                    min_memory_gb=2, recommended_memory_gb=4,
                    supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                    performance_multiplier=7.0,
                    notes="Great performance on Apple Silicon"
                ),
                "webgpu": HardwareRequirements(
                    min_memory_gb=1, recommended_memory_gb=2,
                    supported_precisions=[PrecisionMode.FP32],
                    performance_multiplier=4.0,
                    notes="Good for web-based vision tasks"
                ),
            }
        )
        
        # CLIP Models
        rules["clip"] = ModelProfile(
            model_family="clip",
            model_size=ModelSize.MEDIUM,
            memory_base_mb=1200,  # clip-vit-base-patch32
            web_compatible=True,
            hardware_requirements={
                "cpu": HardwareRequirements(
                    min_memory_gb=4, recommended_memory_gb=8,
                    supported_precisions=[PrecisionMode.FP32],
                    performance_multiplier=1.0,
                    notes="Slower for image-text matching"
                ),
                "cuda": HardwareRequirements(
                    min_memory_gb=4, recommended_memory_gb=6,
                    supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                    performance_multiplier=12.0,
                    notes="Excellent for multimodal tasks"
                ),
                "mps": HardwareRequirements(
                    min_memory_gb=4, recommended_memory_gb=6,
                    supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                    performance_multiplier=8.0,
                    notes="Good multimodal performance"
                ),
                "webgpu": HardwareRequirements(
                    min_memory_gb=2, recommended_memory_gb=4,
                    supported_precisions=[PrecisionMode.FP32],
                    performance_multiplier=5.0,
                    notes="Decent for web applications"
                ),
            }
        )
        
        # Whisper Audio Models
        rules["whisper"] = ModelProfile(
            model_family="whisper",
            model_size=ModelSize.SMALL,
            memory_base_mb=244,  # whisper-base
            web_compatible=True,
            mobile_compatible=True,
            hardware_requirements={
                "cpu": HardwareRequirements(
                    min_memory_gb=2, recommended_memory_gb=4,
                    supported_precisions=[PrecisionMode.FP32],
                    performance_multiplier=1.0,
                    notes="Good for real-time transcription"
                ),
                "cuda": HardwareRequirements(
                    min_memory_gb=2, recommended_memory_gb=3,
                    supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                    performance_multiplier=8.0,
                    notes="Fast audio processing"
                ),
                "mps": HardwareRequirements(
                    min_memory_gb=2, recommended_memory_gb=3,
                    supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                    performance_multiplier=6.0,
                    notes="Excellent on Apple devices"
                ),
                "qualcomm": HardwareRequirements(
                    min_memory_gb=1, recommended_memory_gb=2,
                    supported_precisions=[PrecisionMode.INT8],
                    performance_multiplier=2.5,
                    notes="Optimized for mobile transcription"
                ),
            }
        )
        
        # T5 Models
        rules["t5"] = ModelProfile(
            model_family="t5",
            model_size=ModelSize.MEDIUM,
            memory_base_mb=892,  # t5-base
            web_compatible=True,
            hardware_requirements={
                "cpu": HardwareRequirements(
                    min_memory_gb=4, recommended_memory_gb=8,
                    supported_precisions=[PrecisionMode.FP32],
                    performance_multiplier=1.0,
                    notes="Slower for text-to-text generation"
                ),
                "cuda": HardwareRequirements(
                    min_memory_gb=3, recommended_memory_gb=6,
                    supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                    performance_multiplier=10.0,
                    notes="Great for text generation tasks"
                ),
                "mps": HardwareRequirements(
                    min_memory_gb=3, recommended_memory_gb=6,
                    supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                    performance_multiplier=7.0,
                    notes="Good performance on Apple Silicon"
                ),
            }
        )
        
        return rules
    
    def get_model_profile(self, model_name: str) -> Optional[ModelProfile]:
        """Get model profile by name or family."""
        # Direct lookup first
        if model_name in self.compatibility_rules:
            return self.compatibility_rules[model_name]
        
        # Try to match by family name
        model_lower = model_name.lower()
        for family_name, profile in self.compatibility_rules.items():
            if family_name in model_lower or model_lower.startswith(family_name):
                return profile
        
        # Pattern matching for common model names
        patterns = {
            r'bert.*': 'bert',
            r'gpt.*': 'gpt2', 
            r'llama.*': 'llama',
            r'vit.*': 'vit',
            r'clip.*': 'clip',
            r'whisper.*': 'whisper',
            r't5.*': 't5',
        }
        
        for pattern, family in patterns.items():
            if re.match(pattern, model_lower):
                if family in self.compatibility_rules:
                    return self.compatibility_rules[family]
        
        return None
    
    def get_optimal_hardware(self, model_name: str, available_hardware: List[str],
                           memory_limit_gb: Optional[float] = None,
                           prefer_web_compatible: bool = False,
                           prefer_mobile: bool = False) -> Dict[str, Any]:
        """
        Get optimal hardware recommendation for a model.
        
        Args:
            model_name: Name of the model
            available_hardware: List of available hardware types
            memory_limit_gb: Memory constraint
            prefer_web_compatible: Prefer web-compatible solutions
            prefer_mobile: Prefer mobile-compatible solutions
            
        Returns:
            Dict with recommendation details
        """
        profile = self.get_model_profile(model_name)
        
        if not profile:
            logger.warning(f"No profile found for model {model_name}, using defaults")
            return {
                "recommended_hardware": available_hardware[0] if available_hardware else "cpu",
                "confidence": "low",
                "reason": "No specific profile available",
                "fallback": True
            }
        
        # Filter hardware based on constraints
        candidate_hardware = []
        
        for hw in available_hardware:
            if hw not in profile.hardware_requirements:
                continue
                
            requirements = profile.hardware_requirements[hw]
            
            # Check memory constraint
            if memory_limit_gb and requirements.min_memory_gb > memory_limit_gb:
                continue
            
            # Check web compatibility
            if prefer_web_compatible and not profile.web_compatible:
                hw_profile = self.hardware_profiles.get(hw, {})
                if not hw_profile.get("web_compatible", False):
                    continue
            
            # Check mobile compatibility  
            if prefer_mobile and not profile.mobile_compatible:
                hw_profile = self.hardware_profiles.get(hw, {})
                if not hw_profile.get("mobile_optimized", False):
                    continue
            
            candidate_hardware.append((hw, requirements))
        
        if not candidate_hardware:
            # Fallback to CPU if available
            cpu_option = next((hw for hw in available_hardware if hw == "cpu"), None)
            return {
                "recommended_hardware": cpu_option or available_hardware[0],
                "confidence": "low",
                "reason": "No hardware meets all constraints",
                "fallback": True,
                "constraints_violated": True
            }
        
        # Score and rank hardware options
        scored_options = []
        
        for hw, requirements in candidate_hardware:
            hw_profile = self.hardware_profiles.get(hw, {})
            
            # Base performance score
            performance_score = requirements.performance_multiplier
            
            # Adjust for model size and hardware capability
            memory_efficiency = hw_profile.get("memory_efficiency", 0.8)
            memory_score = min(1.0, requirements.recommended_memory_gb / (requirements.min_memory_gb * 1.5))
            
            # Web/mobile bonuses
            web_bonus = 1.2 if prefer_web_compatible and hw_profile.get("web_compatible") else 1.0
            mobile_bonus = 1.2 if prefer_mobile and hw_profile.get("mobile_optimized") else 1.0
            
            # Calculate final score
            final_score = performance_score * memory_efficiency * memory_score * web_bonus * mobile_bonus
            
            scored_options.append({
                "hardware": hw,
                "score": final_score,
                "requirements": requirements,
                "hardware_profile": hw_profile
            })
        
        # Sort by score
        scored_options.sort(key=lambda x: x["score"], reverse=True)
        best_option = scored_options[0]
        
        return {
            "recommended_hardware": best_option["hardware"],
            "confidence": "high" if best_option["score"] > 5.0 else "medium",
            "performance_multiplier": best_option["requirements"].performance_multiplier,
            "memory_requirements": {
                "minimum_gb": best_option["requirements"].min_memory_gb,
                "recommended_gb": best_option["requirements"].recommended_memory_gb,
            },
            "supported_precisions": [p.value for p in best_option["requirements"].supported_precisions],
            "notes": best_option["requirements"].notes,
            "alternatives": [opt["hardware"] for opt in scored_options[1:3]],  # Top 2 alternatives
            "all_options": scored_options
        }
    
    def check_compatibility(self, model_name: str, hardware: str) -> Dict[str, Any]:
        """Check if a model is compatible with specific hardware."""
        profile = self.get_model_profile(model_name)
        
        if not profile:
            return {
                "compatible": True,  # Assume compatible if no specific rules
                "confidence": "unknown",
                "limitations": ["No specific compatibility data available"]
            }
        
        if hardware not in profile.hardware_requirements:
            return {
                "compatible": False,
                "confidence": "high",
                "reason": f"No support defined for {hardware}",
                "alternatives": list(profile.hardware_requirements.keys())
            }
        
        requirements = profile.hardware_requirements[hardware]
        hw_profile = self.hardware_profiles.get(hardware, {})
        
        # Check basic compatibility
        compatibility_issues = []
        
        # Memory check
        hw_max_memory = hw_profile.get("max_model_size_gb", float('inf'))
        if requirements.min_memory_gb > hw_max_memory:
            compatibility_issues.append(f"Model requires {requirements.min_memory_gb}GB but hardware limit is {hw_max_memory}GB")
        
        # Precision check
        hw_precisions = set(hw_profile.get("supported_precisions", [PrecisionMode.FP32]))
        model_precisions = set(requirements.supported_precisions)
        if not model_precisions.intersection(hw_precisions):
            compatibility_issues.append("No compatible precision modes")
        
        # Special requirements check
        if profile.special_requirements:
            for req in profile.special_requirements:
                if req == "gpu_recommended" and hardware == "cpu":
                    compatibility_issues.append("GPU strongly recommended for this model")
        
        is_compatible = len(compatibility_issues) == 0
        
        return {
            "compatible": is_compatible,
            "confidence": "high",
            "performance_rating": requirements.performance_multiplier,
            "memory_requirements": {
                "minimum_gb": requirements.min_memory_gb,
                "recommended_gb": requirements.recommended_memory_gb,
            },
            "supported_precisions": [p.value for p in requirements.supported_precisions],
            "issues": compatibility_issues,
            "notes": requirements.notes,
            "web_compatible": profile.web_compatible and hardware in ["webnn", "webgpu"],
            "mobile_compatible": profile.mobile_compatible and hardware == "qualcomm"
        }
    
    def get_model_families(self) -> List[str]:
        """Get list of supported model families."""
        return list(self.compatibility_rules.keys())
    
    def get_hardware_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get hardware capability profiles."""
        return self.hardware_profiles.copy()
    
    def estimate_inference_time(self, model_name: str, hardware: str, 
                              batch_size: int = 1, sequence_length: int = 512) -> Dict[str, Any]:
        """Estimate inference time for model on hardware."""
        profile = self.get_model_profile(model_name)
        hw_profile = self.hardware_profiles.get(hardware, {})
        
        if not profile or not hw_profile:
            return {
                "estimated_time_ms": None,
                "confidence": "unknown",
                "notes": "Insufficient data for estimation"
            }
        
        if hardware not in profile.hardware_requirements:
            return {
                "estimated_time_ms": None,
                "confidence": "low", 
                "notes": f"Model not optimized for {hardware}"
            }
        
        # Base estimation (very rough)
        base_time_ms = 100  # Base CPU inference time
        performance_multiplier = profile.hardware_requirements[hardware].performance_multiplier
        
        # Adjust for batch size (sub-linear scaling)
        batch_factor = batch_size ** 0.7
        
        # Adjust for sequence length (for text models)
        seq_factor = (sequence_length / 512) ** 1.2 if "bert" in model_name.lower() or "gpt" in model_name.lower() else 1.0
        
        estimated_time = base_time_ms * batch_factor * seq_factor / performance_multiplier
        
        return {
            "estimated_time_ms": round(estimated_time, 2),
            "confidence": "rough_estimate",
            "factors": {
                "batch_size": batch_size,
                "sequence_length": sequence_length if seq_factor > 1.0 else None,
                "performance_multiplier": performance_multiplier
            },
            "notes": "Rough estimate based on model family and hardware profiles"
        }

# Global instance
compatibility_manager = ModelHardwareCompatibility()

# Convenience functions
def get_optimal_hardware(model_name: str, available_hardware: List[str], **kwargs) -> Dict[str, Any]:
    """Get optimal hardware for a model."""
    return compatibility_manager.get_optimal_hardware(model_name, available_hardware, **kwargs)

def check_model_compatibility(model_name: str, hardware: str) -> Dict[str, Any]:
    """Check model-hardware compatibility."""
    return compatibility_manager.check_compatibility(model_name, hardware)

def get_supported_models() -> List[str]:
    """Get list of supported model families."""
    return compatibility_manager.get_model_families()

if __name__ == "__main__":
    # Demo the compatibility system
    print("🤖 Model-Hardware Compatibility Demo")
    print("=" * 50)
    
    # Test different scenarios
    test_scenarios = [
        ("bert-base-uncased", ["cpu", "cuda", "mps", "webnn"]),
        ("gpt2", ["cpu", "cuda", "mps"]), 
        ("whisper-base", ["cpu", "cuda", "qualcomm"]),
        ("clip-vit-base-patch32", ["cpu", "cuda", "webgpu"]),
    ]
    
    for model_name, hardware_list in test_scenarios:
        print(f"\n🔍 Testing {model_name}")
        result = get_optimal_hardware(model_name, hardware_list)
        print(f"  📌 Recommended: {result['recommended_hardware']}")
        print(f"  🎯 Confidence: {result['confidence']}")
        print(f"  💾 Memory: {result.get('memory_requirements', {}).get('recommended_gb', 'unknown')}GB")
        
        # Check specific compatibility
        for hw in hardware_list[:2]:  # Check top 2
            compat = check_model_compatibility(model_name, hw)
            status = "✅" if compat['compatible'] else "❌"
            print(f"    {status} {hw}: {compat.get('notes', 'No notes')}")