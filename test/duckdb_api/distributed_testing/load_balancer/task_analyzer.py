#!/usr/bin/env python3
"""
Distributed Testing Framework - Task Requirements Analyzer

This module implements the task requirements analysis system for the distributed
testing framework. It analyzes test tasks to determine their resource requirements,
hardware preferences, and optimal execution environment.

Key features:
- Analyzes test tasks to determine resource requirements (CPU, memory, GPU, etc.)
- Extracts hardware preferences based on model families and test types
- Calculates priority scores for different execution environments
- Predicts execution time based on task characteristics
- Identifies specialized hardware requirements for specific test types
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("task_analyzer")

# Model family constants for hardware preferences
MODEL_FAMILY_VISION = "vision"
MODEL_FAMILY_TEXT = "text"
MODEL_FAMILY_AUDIO = "audio"
MODEL_FAMILY_MULTIMODAL = "multimodal"

# Test type constants
TEST_TYPE_INFERENCE = "inference"
TEST_TYPE_TRAINING = "training"
TEST_TYPE_QUANTIZATION = "quantization"
TEST_TYPE_BENCHMARK = "benchmark"
TEST_TYPE_INTEGRATION = "integration"

# Default resource requirements by model family and size category
DEFAULT_REQUIREMENTS = {
    MODEL_FAMILY_VISION: {
        "small": {"cpu_cores": 2, "memory_gb": 2, "gpu_memory_gb": 2},
        "medium": {"cpu_cores": 4, "memory_gb": 4, "gpu_memory_gb": 4},
        "large": {"cpu_cores": 8, "memory_gb": 8, "gpu_memory_gb": 8}
    },
    MODEL_FAMILY_TEXT: {
        "small": {"cpu_cores": 2, "memory_gb": 2, "gpu_memory_gb": 2},
        "medium": {"cpu_cores": 4, "memory_gb": 6, "gpu_memory_gb": 6},
        "large": {"cpu_cores": 8, "memory_gb": 12, "gpu_memory_gb": 12}
    },
    MODEL_FAMILY_AUDIO: {
        "small": {"cpu_cores": 2, "memory_gb": 2, "gpu_memory_gb": 2},
        "medium": {"cpu_cores": 4, "memory_gb": 4, "gpu_memory_gb": 4},
        "large": {"cpu_cores": 8, "memory_gb": 8, "gpu_memory_gb": 8}
    },
    MODEL_FAMILY_MULTIMODAL: {
        "small": {"cpu_cores": 2, "memory_gb": 3, "gpu_memory_gb": 3},
        "medium": {"cpu_cores": 4, "memory_gb": 8, "gpu_memory_gb": 8},
        "large": {"cpu_cores": 8, "memory_gb": 16, "gpu_memory_gb": 16}
    }
}

# Hardware preference scores (higher is better)
HARDWARE_PREFERENCE_SCORES = {
    MODEL_FAMILY_VISION: {
        "cuda": 0.9,   # NVIDIA GPUs are great for vision models
        "rocm": 0.8,   # AMD GPUs are good for vision models
        "cpu": 0.3,    # CPU is usable but slow for vision models
        "mps": 0.7,    # Apple Silicon is good for vision models
        "openvino": 0.7,  # Intel OpenVINO is good for vision inference
        "hexagon": 0.6,   # Qualcomm DSP is good for mobile vision
    },
    MODEL_FAMILY_TEXT: {
        "cuda": 0.9,   # NVIDIA GPUs are great for text models
        "rocm": 0.7,   # AMD GPUs are good for text models
        "cpu": 0.5,    # CPU is decent for smaller text models
        "mps": 0.7,    # Apple Silicon is good for text models
        "openvino": 0.6,  # Intel OpenVINO is decent for text inference
        "hexagon": 0.5,   # Qualcomm DSP is okay for mobile text
    },
    MODEL_FAMILY_AUDIO: {
        "cuda": 0.8,   # NVIDIA GPUs are good for audio models
        "rocm": 0.7,   # AMD GPUs are decent for audio models
        "cpu": 0.6,    # CPU is reasonably good for audio models
        "mps": 0.8,    # Apple Silicon is great for audio models
        "openvino": 0.7,  # Intel OpenVINO is good for audio inference
        "hexagon": 0.7,   # Qualcomm DSP is good for mobile audio
    },
    MODEL_FAMILY_MULTIMODAL: {
        "cuda": 0.9,   # NVIDIA GPUs are great for multimodal models
        "rocm": 0.7,   # AMD GPUs are good for multimodal models
        "cpu": 0.3,    # CPU is usually too slow for multimodal models
        "mps": 0.7,    # Apple Silicon is good for multimodal models
        "openvino": 0.6,  # Intel OpenVINO is decent for multimodal inference
        "hexagon": 0.5,   # Qualcomm DSP is okay for mobile multimodal
    }
}

# Browser-specific preferences for WebNN/WebGPU resource pool integration
BROWSER_PREFERENCE_SCORES = {
    MODEL_FAMILY_VISION: {
        "chrome": 0.9,  # Chrome has excellent WebGPU support for vision
        "edge": 0.8,    # Edge is good for vision models
        "firefox": 0.7, # Firefox is decent for vision models
        "safari": 0.8,  # Safari has good vision support
    },
    MODEL_FAMILY_TEXT: {
        "chrome": 0.8,  # Chrome is good for text models
        "edge": 0.9,    # Edge with WebNN is excellent for text models
        "firefox": 0.7, # Firefox is decent for text models
        "safari": 0.7,  # Safari is decent for text models
    },
    MODEL_FAMILY_AUDIO: {
        "chrome": 0.7,  # Chrome is decent for audio models
        "edge": 0.8,    # Edge is good for audio models
        "firefox": 0.9, # Firefox excels at audio with compute shaders
        "safari": 0.7,  # Safari is decent for audio models
    },
    MODEL_FAMILY_MULTIMODAL: {
        "chrome": 0.8,  # Chrome is good for multimodal models
        "edge": 0.8,    # Edge is good for multimodal models
        "firefox": 0.7, # Firefox is decent for multimodal models
        "safari": 0.7,  # Safari is decent for multimodal models
    }
}

class TaskRequirementsAnalyzer:
    """Analyzes task requirements and resource needs."""
    
    def __init__(self, model_registry=None, result_aggregator=None):
        """Initialize the task analyzer.
        
        Args:
            model_registry: Optional model registry for detailed model information
            result_aggregator: Optional result aggregator for historical execution data
        """
        self.model_registry = model_registry
        self.result_aggregator = result_aggregator
        self.model_families_cache = {}  # Cache model family mappings
        self.execution_time_cache = {}  # Cache execution time predictions
        
    def analyze_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a task to determine its requirements and preferences.
        
        Args:
            task: Task definition including model, test type, and parameters
            
        Returns:
            Dictionary of task requirements and preferences
        """
        # Extract basic task information
        model_id = task.get('model_id')
        test_type = task.get('test_type', TEST_TYPE_INFERENCE)
        test_parameters = task.get('parameters', {})
        
        # Determine model family
        model_family = self._determine_model_family(model_id, test_parameters)
        
        # Determine model size category
        model_size = self._determine_model_size(model_id, test_parameters)
        
        # Get base resource requirements
        base_requirements = self._get_base_requirements(model_family, model_size, test_type)
        
        # Adjust for test type
        adjusted_requirements = self._adjust_for_test_type(base_requirements, test_type)
        
        # Add task-specific adjustments from parameters
        final_requirements = self._adjust_for_task_parameters(adjusted_requirements, test_parameters)
        
        # Determine hardware preferences
        hardware_preferences = self._determine_hardware_preferences(model_family, test_type, test_parameters)
        
        # Determine browser preferences if this is a web test
        browser_preferences = self._determine_browser_preferences(model_family, test_type, test_parameters)
        
        # Predict execution time
        predicted_execution_time = self._predict_execution_time(model_id, model_family, test_type, test_parameters)
        
        # Determine priority level
        priority_level = test_parameters.get('priority', 3)  # Default priority: medium (3)
        
        # Prepare and return the complete analysis
        analysis = {
            'model_id': model_id,
            'test_type': test_type,
            'model_family': model_family,
            'model_size': model_size,
            'resource_requirements': final_requirements,
            'hardware_preferences': hardware_preferences,
            'browser_preferences': browser_preferences if browser_preferences else None,
            'predicted_execution_time': predicted_execution_time,
            'priority_level': priority_level,
            'specialized_hardware_required': self._requires_specialized_hardware(model_id, test_type, test_parameters),
            'suggested_batch_size': self._suggest_batch_size(model_id, model_family, test_type, test_parameters),
        }
        
        logger.debug(f"Task analysis for {model_id}: {json.dumps(analysis, indent=2)}")
        return analysis
    
    def _determine_model_family(self, model_id: str, parameters: Dict[str, Any]) -> str:
        """Determine the model family based on model ID and parameters.
        
        Args:
            model_id: Model identifier
            parameters: Task parameters
            
        Returns:
            Model family (vision, text, audio, multimodal)
        """
        # Check cache first
        if model_id in self.model_families_cache:
            return self.model_families_cache[model_id]
        
        # Use explicit parameter if provided
        if 'model_family' in parameters:
            family = parameters['model_family']
            self.model_families_cache[model_id] = family
            return family
        
        # Try to determine from model_id
        model_id_lower = model_id.lower()
        
        # Vision models
        if any(x in model_id_lower for x in ['vit', 'resnet', 'efficientnet', 'yolo', 'detr', 
                                           'swin', 'dino', 'clip', 'segmentation']):
            family = MODEL_FAMILY_VISION
        
        # Text models
        elif any(x in model_id_lower for x in ['bert', 'gpt', 'llama', 't5', 'roberta', 
                                             'bard', 'xlm', 'bart', 'electra', 'deberta']):
            family = MODEL_FAMILY_TEXT
        
        # Audio models
        elif any(x in model_id_lower for x in ['wav2vec', 'whisper', 'hubert', 'clap', 
                                             'audio', 'speech', 'mms']):
            family = MODEL_FAMILY_AUDIO
        
        # Multimodal models
        elif any(x in model_id_lower for x in ['blip', 'flava', 'flamingo', 'pali', 
                                             'llava', 'qwen-vl', 'fuyu']):
            family = MODEL_FAMILY_MULTIMODAL
        
        # Fallback to registry if available
        elif self.model_registry:
            try:
                model_info = self.model_registry.get_model_info(model_id)
                if model_info and 'family' in model_info:
                    family = model_info['family']
                else:
                    family = MODEL_FAMILY_TEXT  # Default to text if not found
            except Exception as e:
                logger.warning(f"Error retrieving model info from registry: {e}")
                family = MODEL_FAMILY_TEXT  # Default to text
        
        else:
            # Default fallback
            family = MODEL_FAMILY_TEXT
        
        # Cache the result
        self.model_families_cache[model_id] = family
        return family
    
    def _determine_model_size(self, model_id: str, parameters: Dict[str, Any]) -> str:
        """Determine the model size category based on model ID and parameters.
        
        Args:
            model_id: Model identifier
            parameters: Task parameters
            
        Returns:
            Model size category (small, medium, large)
        """
        # Use explicit parameter if provided
        if 'model_size' in parameters:
            return parameters['model_size']
        
        # Use parameter size if provided
        if 'parameter_count' in parameters:
            param_count = parameters['parameter_count']
            if isinstance(param_count, str):
                param_count = param_count.lower()
                if 'b' in param_count:  # Billion
                    try:
                        count = float(param_count.replace('b', ''))
                        if count > 7:
                            return "large"
                        elif count > 1:
                            return "medium"
                        else:
                            return "small"
                    except ValueError:
                        pass
                elif 'm' in param_count:  # Million
                    try:
                        count = float(param_count.replace('m', ''))
                        if count > 500:
                            return "medium"
                        else:
                            return "small"
                    except ValueError:
                        pass
            elif isinstance(param_count, (int, float)):
                if param_count > 7_000_000_000:
                    return "large"
                elif param_count > 1_000_000_000:
                    return "medium"
                else:
                    return "small"
        
        # Try to determine from model_id
        model_id_lower = model_id.lower()
        
        # Check for size indicators in the model name
        if any(x in model_id_lower for x in ['large', '-l', '-xl', '-xxl', '13b', '33b', '70b', '7b']):
            return "large"
        elif any(x in model_id_lower for x in ['medium', '-m', '-base', '1b', '3b', '6b']):
            return "medium"
        elif any(x in model_id_lower for x in ['small', '-s', '-tiny', '-mini', '-nano']):
            return "small"
        
        # Fallback to registry if available
        if self.model_registry:
            try:
                model_info = self.model_registry.get_model_info(model_id)
                if model_info and 'size_category' in model_info:
                    return model_info['size_category']
            except Exception as e:
                logger.warning(f"Error retrieving model info from registry: {e}")
        
        # Default fallback based on specific model types
        if 'bert-base' in model_id_lower or 'distilbert' in model_id_lower:
            return "small"
        elif 'bert-large' in model_id_lower:
            return "medium"
        elif 'gpt2' in model_id_lower and not any(x in model_id_lower for x in ['medium', 'large', 'xl']):
            return "small"
        elif 'gpt2-medium' in model_id_lower:
            return "medium"
        elif 'gpt2-large' in model_id_lower or 'gpt2-xl' in model_id_lower:
            return "large"
        
        # Generic fallback
        return "medium"
    
    def _get_base_requirements(self, model_family: str, model_size: str, test_type: str) -> Dict[str, Any]:
        """Get base resource requirements for the given model family and size.
        
        Args:
            model_family: Model family (vision, text, audio, multimodal)
            model_size: Model size category (small, medium, large)
            test_type: Type of test being executed
            
        Returns:
            Dictionary of base resource requirements
        """
        # Get default requirements for model family and size
        if model_family in DEFAULT_REQUIREMENTS and model_size in DEFAULT_REQUIREMENTS[model_family]:
            return DEFAULT_REQUIREMENTS[model_family][model_size].copy()
        
        # Fallback to text model defaults
        if model_size in DEFAULT_REQUIREMENTS[MODEL_FAMILY_TEXT]:
            return DEFAULT_REQUIREMENTS[MODEL_FAMILY_TEXT][model_size].copy()
        
        # Final fallback
        return {
            "cpu_cores": 4,
            "memory_gb": 4,
            "gpu_memory_gb": 4,
            "disk_space_gb": 1,
            "network_bandwidth_mbps": 10,
        }
    
    def _adjust_for_test_type(self, base_requirements: Dict[str, Any], test_type: str) -> Dict[str, Any]:
        """Adjust resource requirements based on test type.
        
        Args:
            base_requirements: Base resource requirements
            test_type: Type of test being executed
            
        Returns:
            Adjusted resource requirements
        """
        adjusted = base_requirements.copy()
        
        # Adjust based on test type
        if test_type == TEST_TYPE_TRAINING:
            # Training needs more resources
            adjusted["cpu_cores"] = max(adjusted.get("cpu_cores", 4) * 2, 4)
            adjusted["memory_gb"] = max(adjusted.get("memory_gb", 4) * 2, 8)
            adjusted["gpu_memory_gb"] = max(adjusted.get("gpu_memory_gb", 4) * 2, 8)
            adjusted["disk_space_gb"] = max(adjusted.get("disk_space_gb", 1) * 3, 5)
        
        elif test_type == TEST_TYPE_QUANTIZATION:
            # Quantization needs more memory
            adjusted["memory_gb"] = max(adjusted.get("memory_gb", 4) * 1.5, 6)
            adjusted["gpu_memory_gb"] = max(adjusted.get("gpu_memory_gb", 4) * 1.5, 6)
        
        elif test_type == TEST_TYPE_BENCHMARK:
            # Benchmarks need cleaner environment and more precise measurement
            adjusted["dedicated_instance"] = True
            adjusted["isolated_execution"] = True
        
        # Add test type specific fields
        if test_type == TEST_TYPE_INFERENCE:
            adjusted["inference_optimized"] = True
        
        return adjusted
    
    def _adjust_for_task_parameters(self, base_requirements: Dict[str, Any], 
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust resource requirements based on task-specific parameters.
        
        Args:
            base_requirements: Base resource requirements
            parameters: Task parameters
            
        Returns:
            Adjusted resource requirements
        """
        adjusted = base_requirements.copy()
        
        # Apply explicit resource overrides if provided
        for resource_key in ["cpu_cores", "memory_gb", "gpu_memory_gb", "disk_space_gb", 
                          "network_bandwidth_mbps"]:
            if resource_key in parameters:
                adjusted[resource_key] = parameters[resource_key]
        
        # Adjust for batch size
        if "batch_size" in parameters:
            batch_size = parameters["batch_size"]
            if isinstance(batch_size, (int, float)) and batch_size > 1:
                # Scale memory requirements with batch size (sub-linear scaling)
                memory_scale = max(1.0, batch_size ** 0.7)  # Sub-linear scaling
                adjusted["memory_gb"] = max(adjusted.get("memory_gb", 4) * memory_scale, 2)
                adjusted["gpu_memory_gb"] = max(adjusted.get("gpu_memory_gb", 4) * memory_scale, 2)
        
        # Adjust for precision
        if "precision" in parameters:
            precision = parameters["precision"]
            if precision in ["float16", "fp16", "half"]:
                # Half precision reduces memory requirements
                adjusted["memory_gb"] = max(adjusted.get("memory_gb", 4) * 0.6, 2)
                adjusted["gpu_memory_gb"] = max(adjusted.get("gpu_memory_gb", 4) * 0.6, 2)
            elif precision in ["int8", "int4"]:
                # Quantized precision drastically reduces memory requirements
                quant_factor = 0.3 if precision == "int8" else 0.15
                adjusted["memory_gb"] = max(adjusted.get("memory_gb", 4) * quant_factor, 1)
                adjusted["gpu_memory_gb"] = max(adjusted.get("gpu_memory_gb", 4) * quant_factor, 1)
        
        # Adjust for sequence length / context window
        if "sequence_length" in parameters or "context_length" in parameters:
            seq_len = parameters.get("sequence_length", parameters.get("context_length", 512))
            if seq_len > 512:
                # Longer sequences need more memory (sub-linear scaling)
                memory_scale = max(1.0, (seq_len / 512) ** 0.8)  # Sub-linear scaling
                adjusted["memory_gb"] = max(adjusted.get("memory_gb", 4) * memory_scale, 2)
                adjusted["gpu_memory_gb"] = max(adjusted.get("gpu_memory_gb", 4) * memory_scale, 2)
        
        # Checks for specialized features
        if parameters.get("use_flash_attention", False):
            # Flash attention is more memory efficient
            adjusted["gpu_memory_gb"] = max(adjusted.get("gpu_memory_gb", 4) * 0.8, 1)
            adjusted["requires_cuda"] = True  # Flash attention requires CUDA
            
        if parameters.get("use_kv_cache", False):
            # KV cache increases memory usage for inference
            adjusted["memory_gb"] = max(adjusted.get("memory_gb", 4) * 1.2, 2)
            adjusted["gpu_memory_gb"] = max(adjusted.get("gpu_memory_gb", 4) * 1.2, 2)
            
        if parameters.get("gradient_checkpointing", False):
            # Gradient checkpointing trades compute for memory
            adjusted["memory_gb"] = max(adjusted.get("memory_gb", 4) * 0.7, 2)
            adjusted["gpu_memory_gb"] = max(adjusted.get("gpu_memory_gb", 4) * 0.7, 2)
            adjusted["cpu_cores"] = max(adjusted.get("cpu_cores", 4) * 1.3, 2)  # Need more compute
            
        if parameters.get("mixed_precision", False):
            # Mixed precision training
            adjusted["memory_gb"] = max(adjusted.get("memory_gb", 4) * 0.7, 2)
            adjusted["gpu_memory_gb"] = max(adjusted.get("gpu_memory_gb", 4) * 0.7, 2)
            adjusted["requires_cuda"] = True  # Mixed precision usually requires CUDA
        
        # Return the adjusted requirements
        return adjusted
    
    def _determine_hardware_preferences(self, model_family: str, test_type: str, 
                                      parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Determine hardware preferences for the task.
        
        Args:
            model_family: Model family (vision, text, audio, multimodal)
            test_type: Type of test being executed
            parameters: Task parameters
            
        Returns:
            Dictionary of hardware preferences
        """
        # Start with default preferences from the model family
        if model_family in HARDWARE_PREFERENCE_SCORES:
            preferences = HARDWARE_PREFERENCE_SCORES[model_family].copy()
        else:
            preferences = HARDWARE_PREFERENCE_SCORES[MODEL_FAMILY_TEXT].copy()  # Default to text
        
        # Adjust for test type
        if test_type == TEST_TYPE_TRAINING:
            # Training typically prefers CUDA/ROCm GPUs more strongly
            for backend in preferences:
                if backend in ["cuda", "rocm"]:
                    preferences[backend] = min(preferences[backend] * 1.2, 1.0)
                else:
                    preferences[backend] = preferences[backend] * 0.8
        
        elif test_type == TEST_TYPE_QUANTIZATION:
            # Quantization works well on most hardware, slight preference for CUDA
            for backend in preferences:
                if backend == "cuda":
                    preferences[backend] = min(preferences[backend] * 1.1, 1.0)
        
        # Adjust for specific parameters
        if parameters.get("use_flash_attention", False) or parameters.get("mixed_precision", False):
            # These features require CUDA
            preferences = {k: (v if k == "cuda" else 0.1) for k, v in preferences.items()}
        
        # If precision is specified, adjust for hardware compatibility
        if "precision" in parameters:
            precision = parameters["precision"]
            if precision in ["int8", "int4"]:
                # Quantized models work well on CPU and specialized hardware
                for backend in preferences:
                    if backend in ["cpu", "openvino", "hexagon"]:
                        preferences[backend] = min(preferences[backend] * 1.2, 1.0)
        
        # For specific hardware requirements in parameters
        if "required_hardware" in parameters:
            required_hw = parameters["required_hardware"]
            if isinstance(required_hw, str):
                required_hw = [required_hw]
            # Set all non-required hardware to very low preference
            preferences = {k: (0.9 if k in required_hw else 0.1) for k, v in preferences.items()}
        
        # Build preference list sorted by score (higher is better)
        sorted_preferences = sorted(
            [(backend, score) for backend, score in preferences.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create the final hardware preferences
        result = {
            "preference_scores": preferences,
            "priority_list": [backend for backend, _ in sorted_preferences]
        }
        
        if "required_hardware" in parameters:
            result["required_hardware"] = parameters["required_hardware"]
            
        return result
    
    def _determine_browser_preferences(self, model_family: str, test_type: str, 
                                     parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Determine browser preferences for web tests.
        
        Args:
            model_family: Model family (vision, text, audio, multimodal)
            test_type: Type of test being executed
            parameters: Task parameters
            
        Returns:
            Dictionary of browser preferences or None if not applicable
        """
        # Check if this is a web test
        is_web_test = parameters.get("platform", "").lower() in ["web", "browser", "webnn", "webgpu"]
        
        if not is_web_test:
            return None
        
        # Start with default preferences from the model family
        if model_family in BROWSER_PREFERENCE_SCORES:
            preferences = BROWSER_PREFERENCE_SCORES[model_family].copy()
        else:
            preferences = BROWSER_PREFERENCE_SCORES[MODEL_FAMILY_TEXT].copy()  # Default to text
        
        # Check for specific browser requirements
        if "required_browser" in parameters:
            required_browser = parameters["required_browser"]
            if isinstance(required_browser, str):
                required_browser = [required_browser]
            # Set all non-required browsers to very low preference
            preferences = {k: (0.9 if k in required_browser else 0.1) for k, v in preferences.items()}
        
        # Adjust for specific web backends
        web_backend = parameters.get("web_backend", "").lower()
        if web_backend == "webnn":
            # WebNN works best in Edge
            preferences["edge"] = min(preferences.get("edge", 0.5) * 1.3, 1.0)
        elif web_backend == "webgpu":
            # WebGPU works well in Chrome
            preferences["chrome"] = min(preferences.get("chrome", 0.5) * 1.2, 1.0)
        elif web_backend == "webgl":
            # WebGL works well in Firefox
            preferences["firefox"] = min(preferences.get("firefox", 0.5) * 1.2, 1.0)
        
        # Build preference list sorted by score (higher is better)
        sorted_preferences = sorted(
            [(browser, score) for browser, score in preferences.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create the final browser preferences
        result = {
            "preference_scores": preferences,
            "priority_list": [browser for browser, _ in sorted_preferences]
        }
        
        if "required_browser" in parameters:
            result["required_browser"] = parameters["required_browser"]
            
        return result
    
    def _predict_execution_time(self, model_id: str, model_family: str, test_type: str,
                              parameters: Dict[str, Any]) -> float:
        """Predict the execution time for the task.
        
        Args:
            model_id: Model identifier
            model_family: Model family (vision, text, audio, multimodal)
            test_type: Type of test being executed
            parameters: Task parameters
            
        Returns:
            Predicted execution time in seconds
        """
        # Generate a cache key for this prediction
        cache_key = f"{model_id}_{test_type}_{hash(str(sorted(parameters.items())))}"
        
        # Check cache first
        if cache_key in self.execution_time_cache:
            return self.execution_time_cache[cache_key]
        
        # If we have access to result aggregator, use historical data
        if self.result_aggregator:
            try:
                # Get historical execution times for similar tasks
                historical_data = self.result_aggregator.get_execution_times(
                    model_id=model_id,
                    test_type=test_type,
                    time_range="7d"  # Last 7 days
                )
                
                if historical_data and len(historical_data) > 0:
                    # Use median of historical execution times as prediction
                    median_time = historical_data["median"]
                    self.execution_time_cache[cache_key] = median_time
                    return median_time
            except Exception as e:
                logger.warning(f"Error retrieving historical execution data: {e}")
        
        # Fallback to estimation based on model family, type, and parameters
        base_time = 60.0  # Default 1 minute
        
        # Adjust for model family
        family_multipliers = {
            MODEL_FAMILY_VISION: 1.2,
            MODEL_FAMILY_TEXT: 1.0,
            MODEL_FAMILY_AUDIO: 1.5,
            MODEL_FAMILY_MULTIMODAL: 2.0
        }
        base_time *= family_multipliers.get(model_family, 1.0)
        
        # Adjust for test type
        type_multipliers = {
            TEST_TYPE_INFERENCE: 1.0,
            TEST_TYPE_TRAINING: 5.0,
            TEST_TYPE_QUANTIZATION: 3.0,
            TEST_TYPE_BENCHMARK: 2.5,
            TEST_TYPE_INTEGRATION: 2.0
        }
        base_time *= type_multipliers.get(test_type, 1.0)
        
        # Adjust for model size
        model_size = self._determine_model_size(model_id, parameters)
        size_multipliers = {
            "small": 0.5,
            "medium": 1.0,
            "large": 3.0
        }
        base_time *= size_multipliers.get(model_size, 1.0)
        
        # Adjust for batch size
        batch_size = parameters.get("batch_size", 1)
        if isinstance(batch_size, (int, float)) and batch_size > 1:
            # Sub-linear scaling with batch size
            base_time *= max(1.0, batch_size ** 0.7)
        
        # Adjust for sequence length
        seq_len = parameters.get("sequence_length", parameters.get("context_length", 512))
        if seq_len > 512:
            # Sub-linear scaling with sequence length
            base_time *= max(1.0, (seq_len / 512) ** 0.8)
        
        # Store in cache and return
        self.execution_time_cache[cache_key] = base_time
        return base_time
    
    def _requires_specialized_hardware(self, model_id: str, test_type: str, 
                                     parameters: Dict[str, Any]) -> bool:
        """Determine if the task requires specialized hardware.
        
        Args:
            model_id: Model identifier
            test_type: Type of test being executed
            parameters: Task parameters
            
        Returns:
            True if the task requires specialized hardware, False otherwise
        """
        # Check for explicit requirements
        if "required_hardware" in parameters:
            return True
            
        # Training generally requires specialized hardware
        if test_type == TEST_TYPE_TRAINING:
            return True
            
        # Check for specialized features
        if any(parameters.get(feature, False) for feature in [
            "use_flash_attention",
            "mixed_precision",
            "tensor_cores",
            "tpu_accelerated",
            "gpu_required"
        ]):
            return True
            
        # Large models often require specialized hardware
        model_size = self._determine_model_size(model_id, parameters)
        if model_size == "large":
            return True
            
        # Check precision requirements
        precision = parameters.get("precision", "float32")
        if precision in ["bfloat16"]:  # bfloat16 is only available on certain hardware
            return True
            
        return False
    
    def _suggest_batch_size(self, model_id: str, model_family: str, test_type: str,
                          parameters: Dict[str, Any]) -> int:
        """Suggest an optimal batch size for the task.
        
        Args:
            model_id: Model identifier
            model_family: Model family (vision, text, audio, multimodal)
            test_type: Type of test being executed
            parameters: Task parameters
            
        Returns:
            Suggested batch size
        """
        # If batch size is explicitly specified, use it
        if "batch_size" in parameters:
            return parameters["batch_size"]
        
        # Default batch sizes by model family and test type
        default_batch_sizes = {
            MODEL_FAMILY_VISION: {
                TEST_TYPE_INFERENCE: 16,
                TEST_TYPE_TRAINING: 32,
                TEST_TYPE_BENCHMARK: 64,
                "default": 16
            },
            MODEL_FAMILY_TEXT: {
                TEST_TYPE_INFERENCE: 8,
                TEST_TYPE_TRAINING: 16,
                TEST_TYPE_BENCHMARK: 32,
                "default": 8
            },
            MODEL_FAMILY_AUDIO: {
                TEST_TYPE_INFERENCE: 4,
                TEST_TYPE_TRAINING: 8,
                TEST_TYPE_BENCHMARK: 16,
                "default": 4
            },
            MODEL_FAMILY_MULTIMODAL: {
                TEST_TYPE_INFERENCE: 2,
                TEST_TYPE_TRAINING: 4,
                TEST_TYPE_BENCHMARK: 8,
                "default": 2
            }
        }
        
        # Get default batch size for model family and test type
        family_defaults = default_batch_sizes.get(model_family, default_batch_sizes[MODEL_FAMILY_TEXT])
        batch_size = family_defaults.get(test_type, family_defaults.get("default", 8))
        
        # Adjust for model size
        model_size = self._determine_model_size(model_id, parameters)
        size_multipliers = {
            "small": 2.0,
            "medium": 1.0,
            "large": 0.25
        }
        batch_size = max(1, int(batch_size * size_multipliers.get(model_size, 1.0)))
        
        # Special adjustments
        if parameters.get("precision", "") in ["int8", "int4"]:
            # Quantized models can use larger batch sizes
            batch_size = max(1, int(batch_size * 2.0))
            
        if parameters.get("memory_optimized", False):
            # Memory-optimized setups can use larger batch sizes
            batch_size = max(1, int(batch_size * 1.5))
            
        return batch_size


def analyze_task_cli() -> None:
    """CLI entry point for analyzing task requirements."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Analyze task requirements")
    parser.add_argument('--model', type=str, required=True, help="Model ID")
    parser.add_argument('--test-type', type=str, default="inference", 
                      help="Test type (inference, training, quantization, benchmark)")
    parser.add_argument('--parameters', type=str, default="{}", 
                      help="Task parameters in JSON format")
    parser.add_argument('--output', type=str, help="Output file (default: stdout)")
    args = parser.parse_args()
    
    # Parse parameters
    try:
        parameters = json.loads(args.parameters)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in parameters", file=sys.stderr)
        sys.exit(1)
    
    # Create analyzer and task
    analyzer = TaskRequirementsAnalyzer()
    task = {
        "model_id": args.model,
        "test_type": args.test_type,
        "parameters": parameters
    }
    
    # Analyze task
    analysis = analyzer.analyze_task(task)
    
    # Output
    output_json = json.dumps(analysis, indent=2)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
        print(f"Analysis saved to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    analyze_task_cli()