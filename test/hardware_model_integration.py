#!/usr/bin/env python
"""
Hardware-aware model classification utility.
This module integrates hardware detection with model family classification.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple

# Import the required components
from hardware_detection import detect_available_hardware, HardwareDetector
from model_family_classifier import classify_model, ModelFamilyClassifier

logger = logging.getLogger(__name__)

class HardwareAwareModelClassifier:
    """
    Integrates hardware detection with model family classification
    to provide optimal hardware selection and template recommendations.
    """
    
    def __init__(self, 
                 hardware_cache_path: Optional[str] = None,
                 model_db_path: Optional[str] = None,
                 force_refresh: bool = False):
        """
        Initialize the hardware-aware model classifier.
        
        Args:
            hardware_cache_path: Optional path to hardware detection cache
            model_db_path: Optional path to model database
            force_refresh: Force refresh of hardware detection cache
        """
        self.hardware_detector = HardwareDetector(
            cache_file=hardware_cache_path,
            force_refresh=force_refresh
        )
        self.model_classifier = ModelFamilyClassifier(model_db_path=model_db_path)
        self.hardware_info = self.hardware_detector.get_available_hardware()
        self.hardware_details = self.hardware_detector.get_hardware_details()
        
        # Log available hardware
        available_hw = [hw for hw, available in self.hardware_info.items() if available]
        logger.info(f"Hardware-aware model classifier initialized. Available hardware: {', '.join(available_hw)}")
        logger.info(f"Best available hardware: {self.hardware_detector.get_best_available_hardware()}")
    
    def classify_model(self, 
                       model_name: str, 
                       model_class: Optional[str] = None,
                       tasks: Optional[List[str]] = None,
                       methods: Optional[List[str]] = None,
                       hw_compat_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify a model with hardware awareness.
        
        Args:
            model_name: Name of the model to classify
            model_class: Optional model class name
            tasks: Optional list of model tasks
            methods: Optional list of model methods
            hw_compat_override: Optional hardware compatibility overrides
            
        Returns:
            Dictionary with classification results and hardware recommendations
        """
        # Generate hardware compatibility profile if not provided
        hw_compatibility = hw_compat_override or self._create_hw_compatibility_profile(model_name)
        
        # Classify the model using both model info and hardware info
        classification = self.model_classifier.classify_model(
            model_name=model_name,
            model_class=model_class,
            tasks=tasks, 
            methods=methods,
            hw_compatibility=hw_compatibility
        )
        
        # Add hardware-specific information
        classification["hardware_profile"] = hw_compatibility
        classification["recommended_hardware"] = self._determine_optimal_hardware(
            classification["family"], 
            classification.get("subfamily")
        )
        
        # Add template recommendation
        classification["recommended_template"] = self.model_classifier.get_template_for_family(
            classification["family"],
            classification.get("subfamily")
        )
        
        # Add resource requirements based on model family and hardware
        classification["resource_requirements"] = self._determine_resource_requirements(
            classification["family"],
            classification.get("subfamily"),
            classification["recommended_hardware"]
        )
        
        return classification
    
    def _create_hw_compatibility_profile(self, model_name: str) -> Dict[str, Any]:
        """
        Create a hardware compatibility profile for a model.
        
        In a production system, this would use a database or test results.
        For this implementation, we use a simple heuristic based on model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with hardware compatibility information
        """
        normalized_name = model_name.lower()
        
        # Default profile - generally compatible with everything
        profile = {
            "cuda": {"compatible": True, "memory_usage": {"peak": 500}},
            "mps": {"compatible": True},
            "rocm": {"compatible": True},
            "openvino": {"compatible": True},
            "webnn": {"compatible": True},
            "webgpu": {"compatible": True}
        }
        
        # Adjust for known model types
        
        # Large language models (e.g., llama, gpt-*, falcon, etc.)
        if any(name in normalized_name for name in ["llama", "gpt", "falcon", "mixtral", "phi", "gemma"]):
            profile["cuda"]["memory_usage"]["peak"] = 5000  # Increased memory requirement
            profile["mps"]["compatible"] = False  # Often too large for MPS
            profile["openvino"]["compatible"] = False  # Often not optimized for OpenVINO
            profile["webnn"]["compatible"] = False  # Too large for WebNN
            profile["webgpu"]["compatible"] = False  # Too large for WebGPU
        
        # Vision-language models
        elif any(name in normalized_name for name in ["llava", "blip", "pali"]):
            profile["cuda"]["memory_usage"]["peak"] = 7000
            profile["mps"]["compatible"] = False
            profile["openvino"]["compatible"] = False
            profile["webnn"]["compatible"] = False
            profile["webgpu"]["compatible"] = False
        
        # Audio models
        elif any(name in normalized_name for name in ["whisper", "wav2vec", "hubert", "clap"]):
            profile["cuda"]["memory_usage"]["peak"] = 2000
            profile["webnn"]["compatible"] = False
            profile["audio_incompatible"] = True
        
        # Vision models
        elif any(name in normalized_name for name in ["vit", "resnet", "swin", "deit", "clip"]):
            profile["cuda"]["memory_usage"]["peak"] = 1500
            profile["openvino"]["compatible"] = True  # Good for OpenVINO
        
        # Add system availability
        for hw_type, available in self.hardware_info.items():
            if hw_type in profile:
                profile[hw_type]["system_available"] = available
                
                # If system doesn't have this hardware, model can't effectively use it
                if not available:
                    profile[hw_type]["effective_compatibility"] = False
                else:
                    profile[hw_type]["effective_compatibility"] = profile[hw_type].get("compatible", False)
        
        return profile
    
    def _determine_optimal_hardware(self, family: str, subfamily: Optional[str] = None) -> str:
        """
        Determine the optimal hardware for a model family and subfamily.
        
        Args:
            family: Model family
            subfamily: Optional model subfamily
            
        Returns:
            String with recommended hardware type
        """
        # Get best available hardware
        best_hw = self.hardware_detector.get_best_available_hardware()
        
        # Model family specific recommendations
        if family == "text_generation":
            # Text generation models benefit from CUDA if available
            if self.hardware_info.get("cuda", False):
                return "cuda"
            elif self.hardware_info.get("rocm", False):
                return "rocm"
            elif self.hardware_info.get("mps", False) and subfamily != "causal_lm":
                # MPS for smaller text generation models
                return "mps"
            else:
                return "cpu"
        
        elif family == "vision":
            # Vision models work well with OpenVINO
            if self.hardware_info.get("openvino", False):
                return "openvino"
            # Fallback to GPU
            elif self.hardware_info.get("cuda", False):
                return "cuda"
            elif self.hardware_info.get("mps", False):
                return "mps"
            else:
                return "cpu"
        
        elif family == "audio":
            # Audio models often need CUDA
            if self.hardware_info.get("cuda", False):
                return "cuda"
            elif self.hardware_info.get("mps", False):
                return "mps"
            else:
                return "cpu"
        
        elif family == "multimodal":
            # Multimodal models almost always need CUDA
            if self.hardware_info.get("cuda", False):
                return "cuda"
            else:
                return "cpu"  # Often slow on CPU
        
        # Default to best available
        return best_hw
    
    def _determine_resource_requirements(self, family: str, subfamily: Optional[str] = None, hardware: str = "cpu") -> Dict[str, Any]:
        """
        Determine resource requirements for a model family on specific hardware.
        
        Args:
            family: Model family
            subfamily: Optional model subfamily
            hardware: Hardware type
            
        Returns:
            Dictionary with resource requirements
        """
        # Default requirements
        requirements = {
            "min_memory_mb": 2000,
            "recommended_memory_mb": 4000,
            "cpu_cores": 2,
            "disk_space_mb": 500,
            "batch_size": 1
        }
        
        # Adjust based on model family
        if family == "text_generation":
            if subfamily == "causal_lm":
                # Large language models need more resources
                requirements["min_memory_mb"] = 8000
                requirements["recommended_memory_mb"] = 16000
                requirements["cpu_cores"] = 4
                requirements["disk_space_mb"] = 5000
            else:
                requirements["min_memory_mb"] = 4000
                requirements["recommended_memory_mb"] = 8000
        
        elif family == "vision":
            requirements["min_memory_mb"] = 4000
            requirements["recommended_memory_mb"] = 8000
            requirements["batch_size"] = 4 if hardware in ["cuda", "rocm"] else 1
        
        elif family == "audio":
            requirements["min_memory_mb"] = 4000
            requirements["recommended_memory_mb"] = 8000
            # Audio often needs more disk for temp files
            requirements["disk_space_mb"] = 1000
        
        elif family == "multimodal":
            requirements["min_memory_mb"] = 12000
            requirements["recommended_memory_mb"] = 24000
            requirements["cpu_cores"] = 8
            requirements["disk_space_mb"] = 10000
        
        # Adjust based on hardware
        if hardware == "cuda" or hardware == "rocm":
            # GPU can use smaller batches more efficiently
            requirements["batch_size"] = max(4, requirements["batch_size"])
        elif hardware == "cpu":
            # CPU might need more memory
            requirements["min_memory_mb"] *= 1.5
            requirements["recommended_memory_mb"] *= 1.5
            # But can't handle as large batches
            requirements["batch_size"] = 1
        
        return requirements
    
    def get_compatible_models_for_hardware(self, models: List[str], hardware_type: str) -> List[str]:
        """
        Find models compatible with specific hardware.
        
        Args:
            models: List of model names
            hardware_type: Hardware type to check compatibility for
            
        Returns:
            List of compatible model names
        """
        compatible_models = []
        
        for model_name in models:
            # Create hardware compatibility profile
            hw_compat = self._create_hw_compatibility_profile(model_name)
            
            # Check if model is compatible with specified hardware
            if (hardware_type in hw_compat and 
                hw_compat[hardware_type].get("compatible", False) and
                hw_compat[hardware_type].get("system_available", False)):
                compatible_models.append(model_name)
        
        return compatible_models
    
    def recommend_model_for_task(self, task: str, hardware_constraints: List[str] = None) -> Dict[str, Any]:
        """
        Recommend a model for a specific task with hardware constraints.
        
        Args:
            task: Task name (e.g., "text-generation", "image-classification")
            hardware_constraints: Optional list of required hardware types
            
        Returns:
            Dictionary with model recommendation
        """
        # Map task to model family
        task_to_family = {
            "text-generation": "text_generation",
            "summarization": "text_generation",
            "translation": "text_generation",
            "fill-mask": "embedding",
            "sentence-similarity": "embedding",
            "token-classification": "embedding",
            "image-classification": "vision",
            "object-detection": "vision",
            "image-segmentation": "vision",
            "depth-estimation": "vision",
            "automatic-speech-recognition": "audio",
            "audio-classification": "audio",
            "text-to-audio": "audio",
            "image-to-text": "multimodal",
            "visual-question-answering": "multimodal",
            "text-to-image": "multimodal"
        }
        
        family = task_to_family.get(task)
        if not family:
            return {
                "error": f"Unknown task: {task}",
                "recommendations": []
            }
        
        # Sample models for each family (in a real system, this would be from a database)
        family_to_models = {
            "embedding": ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
            "text_generation": ["gpt2", "t5-small", "facebook/opt-125m"],
            "vision": ["google/vit-base-patch16-224", "microsoft/resnet-50", "facebook/deit-base-patch16-224"],
            "audio": ["facebook/wav2vec2-base", "openai/whisper-small", "facebook/hubert-base-ls960"],
            "multimodal": ["llava-hf/llava-1.5-7b-hf", "Salesforce/blip-image-captioning-base", "openai/clip-vit-base-patch32"]
        }
        
        candidate_models = family_to_models.get(family, [])
        
        # Filter by hardware constraints
        if hardware_constraints:
            compatible_models = []
            for model in candidate_models:
                is_compatible = True
                hw_compat = self._create_hw_compatibility_profile(model)
                
                for hw_type in hardware_constraints:
                    if (hw_type not in hw_compat or 
                        not hw_compat[hw_type].get("compatible", False) or
                        not hw_compat[hw_type].get("system_available", False)):
                        is_compatible = False
                        break
                
                if is_compatible:
                    compatible_models.append(model)
            
            candidate_models = compatible_models
        
        # Recommend best model by getting detail for each
        recommendations = []
        for model in candidate_models:
            classification = self.classify_model(model)
            recommendations.append({
                "model_name": model,
                "family": classification["family"],
                "subfamily": classification.get("subfamily"),
                "recommended_hardware": classification["recommended_hardware"],
                "resource_requirements": classification["resource_requirements"],
                "confidence": classification.get("confidence", 0)
            })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return {
            "task": task,
            "family": family,
            "hardware_constraints": hardware_constraints,
            "recommendations": recommendations
        }
    
    def get_optimal_resource_pool_config(self, models: List[str]) -> Dict[str, Any]:
        """
        Generate optimal resource pool configuration for a set of models.
        
        Args:
            models: List of model names
            
        Returns:
            Dictionary with resource pool configuration
        """
        # Classify all models
        model_classifications = [self.classify_model(model) for model in models]
        
        # Determine maximum resource requirements
        max_memory_mb = 0
        max_disk_space_mb = 0
        max_cpu_cores = 2
        recommended_timeout_mins = 10  # Default timeout
        
        for classification in model_classifications:
            requirements = classification.get("resource_requirements", {})
            max_memory_mb = max(max_memory_mb, requirements.get("recommended_memory_mb", 0))
            max_disk_space_mb = max(max_disk_space_mb, requirements.get("disk_space_mb", 0))
            max_cpu_cores = max(max_cpu_cores, requirements.get("cpu_cores", 0))
            
            # Large models need longer timeouts
            if classification.get("family") in ["text_generation", "multimodal"]:
                recommended_timeout_mins = max(recommended_timeout_mins, 30)
        
        # Check if we need low memory mode
        system_memory_mb = 0
        try:
            import psutil
            system_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        except ImportError:
            # Fallback - assume 8GB
            system_memory_mb = 8192
        
        low_memory_mode = system_memory_mb < (max_memory_mb * 1.5)
        
        return {
            "models": models,
            "resource_requirements": {
                "max_memory_mb": max_memory_mb,
                "max_disk_space_mb": max_disk_space_mb,
                "max_cpu_cores": max_cpu_cores
            },
            "resource_pool_config": {
                "low_memory_mode": low_memory_mode,
                "recommended_timeout_mins": recommended_timeout_mins,
                "aggressive_cleanup": low_memory_mode,
                "batch_processing": not low_memory_mode and system_memory_mb > (max_memory_mb * 2)
            },
            "hardware_recommendations": {
                "preferred_hardware": self.hardware_detector.get_best_available_hardware(),
                "torch_device": self.hardware_detector.get_torch_device(),
                "available_hardware": [hw for hw, available in self.hardware_info.items() if available]
            }
        }

def get_hardware_aware_model_classification(model_name: str, 
                                           model_class: Optional[str] = None,
                                           tasks: Optional[List[str]] = None,
                                           hw_cache_path: Optional[str] = None,
                                           model_db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to get hardware-aware model classification.
    
    Args:
        model_name: Name of the model to classify
        model_class: Optional model class name
        tasks: Optional list of model tasks
        hw_cache_path: Optional path to hardware detection cache
        model_db_path: Optional path to model database
        
    Returns:
        Dictionary with classification results and hardware recommendations
    """
    classifier = HardwareAwareModelClassifier(
        hardware_cache_path=hw_cache_path,
        model_db_path=model_db_path
    )
    
    return classifier.classify_model(
        model_name=model_name,
        model_class=model_class,
        tasks=tasks
    )

if __name__ == "__main__":
    import argparse
    
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hardware-aware model classification")
    parser.add_argument("--model", type=str, help="Model name to classify")
    parser.add_argument("--task", type=str, help="Task to recommend a model for")
    parser.add_argument("--hw", type=str, nargs="+", help="Hardware constraints for task recommendation")
    parser.add_argument("--hw-cache", type=str, help="Path to hardware detection cache")
    parser.add_argument("--model-db", type=str, help="Path to model database")
    parser.add_argument("--resource-config", type=str, nargs="+", help="Models to generate resource pool config for")
    args = parser.parse_args()
    
    # Create classifier
    classifier = HardwareAwareModelClassifier(
        hardware_cache_path=args.hw_cache,
        model_db_path=args.model_db
    )
    
    # Handle model classification
    if args.model:
        classification = classifier.classify_model(args.model)
        
        print(f"\n=== Hardware-Aware Classification for {args.model} ===")
        print(f"Family: {classification['family']}")
        print(f"Subfamily: {classification.get('subfamily')}")
        print(f"Recommended Hardware: {classification['recommended_hardware']}")
        print(f"Recommended Template: {classification['recommended_template']}")
        
        print("\nResource Requirements:")
        for key, value in classification["resource_requirements"].items():
            print(f"  {key}: {value}")
        
        print("\nHardware Compatibility:")
        hw_profile = classification.get("hardware_profile", {})
        for hw_type, details in hw_profile.items():
            if isinstance(details, dict) and "compatible" in details:
                status = "✅" if details.get("compatible", False) else "❌"
                system_status = "✅" if details.get("system_available", False) else "❌"
                print(f"  {hw_type}: {status} (System: {system_status})")
    
    # Handle task recommendation
    elif args.task:
        recommendation = classifier.recommend_model_for_task(args.task, args.hw)
        
        if "error" in recommendation:
            print(f"Error: {recommendation['error']}")
        else:
            print(f"\n=== Model Recommendations for Task: {args.task} ===")
            if args.hw:
                print(f"Hardware Constraints: {', '.join(args.hw)}")
            
            if not recommendation["recommendations"]:
                print("No compatible models found for the specified constraints.")
            else:
                for i, model in enumerate(recommendation["recommendations"], 1):
                    print(f"\n{i}. {model['model_name']}")
                    print(f"   Family: {model['family']}")
                    print(f"   Optimal Hardware: {model['recommended_hardware']}")
                    print(f"   Memory Required: {model['resource_requirements']['recommended_memory_mb']} MB")
    
    # Handle resource pool configuration
    elif args.resource_config:
        config = classifier.get_optimal_resource_pool_config(args.resource_config)
        
        print(f"\n=== Resource Pool Configuration for {len(args.resource_config)} Models ===")
        print(f"Max Memory Required: {config['resource_requirements']['max_memory_mb']} MB")
        print(f"Low Memory Mode: {'Enabled' if config['resource_pool_config']['low_memory_mode'] else 'Disabled'}")
        print(f"Recommended Timeout: {config['resource_pool_config']['recommended_timeout_mins']} minutes")
        print(f"Batch Processing: {'Enabled' if config['resource_pool_config']['batch_processing'] else 'Disabled'}")
        
        print("\nHardware Recommendations:")
        print(f"  Preferred Hardware: {config['hardware_recommendations']['preferred_hardware']}")
        print(f"  PyTorch Device: {config['hardware_recommendations']['torch_device']}")
        print(f"  Available Hardware: {', '.join(config['hardware_recommendations']['available_hardware'])}")
    
    else:
        # If no specific action, show hardware capabilities
        print("\n=== Hardware Capabilities ===")
        available_hw = [hw for hw, available in classifier.hardware_info.items() if available]
        print(f"Available Hardware: {', '.join(available_hw)}")
        print(f"Best Available Hardware: {classifier.hardware_detector.get_best_available_hardware()}")
        print(f"PyTorch Device: {classifier.hardware_detector.get_torch_device()}")
        
        # Show CUDA devices if available
        if classifier.hardware_info.get("cuda", False):
            cuda_details = classifier.hardware_details.get("cuda", {})
            print(f"\nCUDA Devices: {cuda_details.get('device_count', 0)}")
            for device in cuda_details.get("devices", []):
                print(f"  {device.get('name')}")
        
        print("\nUse --model MODEL_NAME to classify a specific model")
        print("Use --task TASK_NAME to get model recommendations for a task")
        print("Use --resource-config MODEL1 MODEL2 ... to generate resource pool configuration")