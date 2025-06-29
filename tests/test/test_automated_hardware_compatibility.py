#!/usr/bin/env python
"""
Automated hardware compatibility testing for model families.

This script automatically tests compatibility between different hardware platforms
and model families, creating a comprehensive compatibility matrix.
It integrates with the hardware detection and model family classification systems
to perform real-world compatibility testing across available hardware platforms.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import concurrent.futures

# Configure logging
logging.basicConfig())))level=logging.INFO, 
format='%())))asctime)s - %())))name)s - %())))levelname)s - %())))message)s')
logger = logging.getLogger())))__name__)

# Try to import required components with graceful degradation
try:
    from resource_pool import get_global_resource_pool
    RESOURCE_POOL_AVAILABLE = True
except ImportError as e:
    logger.error())))f"Resource pool not available: {}}}}}}}}}}}}}}}}e}")
    RESOURCE_POOL_AVAILABLE = False

try:
    from generators.hardware.hardware_detection import ())))
    HardwareDetector, detect_available_hardware,
    CPU, CUDA, ROCM, MPS, OPENVINO, WEBNN, WEBGPU, QUALCOMM
    )
    HARDWARE_DETECTION_AVAILABLE = True
except ImportError as e:
    logger.error())))f"Hardware detection not available: {}}}}}}}}}}}}}}}}e}")
    HARDWARE_DETECTION_AVAILABLE = False
    # Define fallback constants
    CPU, CUDA, ROCM, MPS, OPENVINO, WEBNN, WEBGPU, QUALCOMM = "cpu", "cuda", "rocm", "mps", "openvino", "webnn", "webgpu", "qualcomm"

try:
    from model_family_classifier import classify_model, ModelFamilyClassifier
    MODEL_CLASSIFIER_AVAILABLE = True
except ImportError as e:
    logger.warning())))f"Model family classifier not available: {}}}}}}}}}}}}}}}}e}")
    MODEL_CLASSIFIER_AVAILABLE = False

# Define test model families and representative models
    DEFAULT_TEST_MODELS = {}}}}}}}}}}}}}}}}
    "embedding": []]]]],,,,,
    "prajjwal1/bert-tiny",
    "distilbert-base-uncased"
    ],
    "text_generation": []]]]],,,,,
    "gpt2",
    "google/t5-efficient-tiny"
    ],
    "vision": []]]]],,,,,
    "google/vit-base-patch16-224",
    "facebook/dinov2-small"
    ],
    "audio": []]]]],,,,,
    "openai/whisper-tiny",
    "facebook/wav2vec2-base"
    ],
    "multimodal": []]]]],,,,,
    "openai/clip-vit-base-patch32",
    "vinvino02/glpn-tiny"
    ]
    }

class HardwareCompatibilityTester:
    """
    Tests model compatibility across different hardware platforms.
    
    This class handles automated testing of model compatibility with different
    hardware backends, creating a comprehensive compatibility matrix.
    """
    
    def __init__())))self, 
    output_dir: str = "./hardware_compatibility_results",
    hw_cache_path: Optional[]]]]],,,,,str] = None,
    model_db_path: Optional[]]]]],,,,,str] = None,
    timeout: float = 0.1,
                 test_models: Optional[]]]]],,,,,Dict[]]]]],,,,,str, List[]]]]],,,,,str]]] = None):
                     """
                     Initialize the hardware compatibility tester.
        
        Args:
            output_dir: Directory for test results
            hw_cache_path: Optional path to hardware detection cache
            model_db_path: Optional path to model database
            timeout: Resource cleanup timeout in minutes
            test_models: Optional dictionary mapping model families to test models
            """
            self.output_dir = output_dir
            self.hw_cache_path = hw_cache_path
            self.model_db_path = model_db_path
            self.timeout = timeout
            self.test_models = test_models or DEFAULT_TEST_MODELS
        
        # Create output directory
            os.makedirs())))output_dir, exist_ok=True)
        
        # Initialize results storage
            self.results = {}}}}}}}}}}}}}}}}
            "timestamp": datetime.now())))).isoformat())))),
            "available_hardware": {}}}}}}}}}}}}}}}}},
            "compatibility_matrix": {}}}}}}}}}}}}}}}}},
            "model_family_compatibility": {}}}}}}}}}}}}}}}}},
            "hardware_platform_capabilities": {}}}}}}}}}}}}}}}}},
            "detailed_results": {}}}}}}}}}}}}}}}}},
            "errors": {}}}}}}}}}}}}}}}}}
            }
        
        # Check for required components
        if not RESOURCE_POOL_AVAILABLE:
            raise ImportError())))"Resource pool is required for compatibility testing")
            
        if not HARDWARE_DETECTION_AVAILABLE:
            logger.warning())))"Hardware detection not available. Using limited testing capabilities.")
            
        if not MODEL_CLASSIFIER_AVAILABLE:
            logger.warning())))"Model classifier not available. Using predefined model families.")
    
    def detect_available_hardware())))self) -> Dict[]]]]],,,,,str, bool]:
        """
        Detect available hardware platforms.
        
        Returns:
            Dictionary of hardware platforms and their availability
            """
            logger.info())))"Detecting available hardware platforms...")
        
        if HARDWARE_DETECTION_AVAILABLE:
            # Use hardware detection module for comprehensive detection
            detector = HardwareDetector())))cache_file=self.hw_cache_path)
            hardware_info = detector.get_available_hardware()))))
            best_hardware = detector.get_best_available_hardware()))))
            device_with_index = detector.get_device_with_index()))))
            
            # Store detection results
            self.results[]]]]],,,,,"available_hardware"] = {}}}}}}}}}}}}}}}}
            "platforms": hardware_info,
            "best_available": best_hardware,
            "torch_device": device_with_index
            }
            
            # Get detailed hardware information
            hardware_details = detector.get_hardware_details()))))
            self.results[]]]]],,,,,"hardware_details"] = hardware_details
            
            # Create more readable summary of available platforms
            available_platforms = {}}}}}}}}}}}}}}}}
            platform: available for platform, available in hardware_info.items()))))
            if platform in []]]]],,,,,CPU, CUDA, ROCM, MPS, OPENVINO, WEBNN, WEBGPU, QUALCOMM]
            }
            :
                logger.info())))f"Available hardware platforms: {}}}}}}}}}}}}}}}}[]]]]],,,,,p for p, a in available_platforms.items())))) if a]}")
            return available_platforms:
        else:
            # Fallback to basic detection using torch
            try:
                import torch
                cuda_available = torch.cuda.is_available()))))
                mps_available = hasattr())))torch.backends, "mps") and torch.backends.mps.is_available()))))
                
                available_platforms = {}}}}}}}}}}}}}}}}
                CPU: True,
                CUDA: cuda_available,
                MPS: mps_available,
                ROCM: False,
                OPENVINO: False,
                WEBNN: False,
                WEBGPU: False,
                QUALCOMM: False
                }
                
                # Store basic detection results
                self.results[]]]]],,,,,"available_hardware"] = {}}}}}}}}}}}}}}}}
                "platforms": available_platforms,
                    "best_available": "cuda" if cuda_available else "mps" if mps_available else "cpu",:
                        "torch_device": "cuda" if cuda_available else "mps" if mps_available else "cpu"
                        }
                :
                    logger.info())))f"Available hardware platforms ())))basic detection): {}}}}}}}}}}}}}}}}[]]]]],,,,,p for p, a in available_platforms.items())))) if a]}")
                return available_platforms:
            except ImportError:
                logger.warning())))"PyTorch not available. Assuming only CPU is available.")
                available_platforms = {}}}}}}}}}}}}}}}}
                CPU: True,
                CUDA: False,
                MPS: False,
                ROCM: False,
                OPENVINO: False,
                WEBNN: False,
                WEBGPU: False,
                QUALCOMM: False
                }
                
                # Store minimal detection results
                self.results[]]]]],,,,,"available_hardware"] = {}}}}}}}}}}}}}}}}
                "platforms": available_platforms,
                "best_available": "cpu",
                "torch_device": "cpu"
                }
                
                logger.info())))"Available hardware platforms ())))minimal detection): CPU only")
                    return available_platforms
    
                    def test_model_on_hardware())))self,
                    model_name: str,
                               hardware_platform: str) -> Dict[]]]]],,,,,str, Any]:
                                   """
                                   Test a model on a specific hardware platform.
        
        Args:
            model_name: Name of the model to test
            hardware_platform: Hardware platform to test on
            
        Returns:
            Dictionary with test results
            """
            logger.info())))f"Testing {}}}}}}}}}}}}}}}}model_name} on {}}}}}}}}}}}}}}}}hardware_platform}...")
        
        # Initialize result dictionary
            result = {}}}}}}}}}}}}}}}}
            "model_name": model_name,
            "hardware_platform": hardware_platform,
            "success": False,
            "error": None,
            "load_time_ms": None,
            "inference_time_ms": None,
            "memory_usage_mb": None,
            "hardware_details": {}}}}}}}}}}}}}}}}},
            "model_details": {}}}}}}}}}}}}}}}}}
            }
        
        # Get resource pool
            pool = get_global_resource_pool()))))
        
        # Get model family if classifier is available
        model_family = "default":
        if MODEL_CLASSIFIER_AVAILABLE:
            try:
                # Classify model
                classification = classify_model())))model_name=model_name)
                model_family = classification.get())))"family", "default")
                result[]]]]],,,,,"model_details"] = {}}}}}}}}}}}}}}}}
                "family": model_family,
                "subfamily": classification.get())))"subfamily"),
                "confidence": classification.get())))"confidence", 0)
                }
                logger.debug())))f"Model {}}}}}}}}}}}}}}}}model_name} classified as {}}}}}}}}}}}}}}}}model_family}")
            except Exception as e:
                logger.warning())))f"Error classifying model {}}}}}}}}}}}}}}}}model_name}: {}}}}}}}}}}}}}}}}e}")
                # Try to infer model family from model name
                if "bert" in model_name.lower())))) or "roberta" in model_name.lower())))):
                    model_family = "embedding"
                elif "gpt" in model_name.lower())))) or "t5" in model_name.lower())))):
                    model_family = "text_generation"
                elif "vit" in model_name.lower())))) or "resnet" in model_name.lower())))):
                    model_family = "vision"
                elif "whisper" in model_name.lower())))) or "wav2vec" in model_name.lower())))):
                    model_family = "audio"
                elif "clip" in model_name.lower())))) or "blip" in model_name.lower())))):
                    model_family = "multimodal"
        else:
            # Try to infer model family from model name
            if "bert" in model_name.lower())))) or "roberta" in model_name.lower())))):
                model_family = "embedding"
            elif "gpt" in model_name.lower())))) or "t5" in model_name.lower())))):
                model_family = "text_generation"
            elif "vit" in model_name.lower())))) or "resnet" in model_name.lower())))):
                model_family = "vision"
            elif "whisper" in model_name.lower())))) or "wav2vec" in model_name.lower())))):
                model_family = "audio"
            elif "clip" in model_name.lower())))) or "blip" in model_name.lower())))):
                model_family = "multimodal"
            
                result[]]]]],,,,,"model_details"] = {}}}}}}}}}}}}}}}}
                "family": model_family,
                "inferred_from_name": True
                }
        
        # Create hardware preferences
                hardware_preferences = {}}}}}}}}}}}}}}}}"device": hardware_platform}
        
        # Measure load time
                load_start_time = time.time()))))
        
        try:
            # Define model constructor for resource pool
            def create_model())))):
                try:
                    # Make sure torch is loaded
                    torch = pool.get_resource())))"torch", constructor=lambda: __import__())))"torch"))
                    if torch is None:
                    raise ImportError())))"PyTorch not available")
                    
                    # Make sure transformers is loaded
                    transformers = pool.get_resource())))"transformers", constructor=lambda: __import__())))"transformers"))
                    if transformers is None:
                    raise ImportError())))"Transformers not available")
                    
                    # Select appropriate model class based on model family
                    if model_family == "text_generation":
                        try:
                            from transformers import AutoModelForCausalLM
                            logger.debug())))f"Using AutoModelForCausalLM for {}}}}}}}}}}}}}}}}model_name}")
                        return AutoModelForCausalLM.from_pretrained())))model_name)
                        except Exception as e:
                            logger.warning())))f"Error loading with AutoModelForCausalLM: {}}}}}}}}}}}}}}}}e}, falling back to AutoModel")
                            from transformers import AutoModel
                        return AutoModel.from_pretrained())))model_name)
                    elif model_family == "vision":
                        try:
                            from transformers import AutoModelForImageClassification
                            logger.debug())))f"Using AutoModelForImageClassification for {}}}}}}}}}}}}}}}}model_name}")
                        return AutoModelForImageClassification.from_pretrained())))model_name)
                        except Exception as e:
                            logger.warning())))f"Error loading with AutoModelForImageClassification: {}}}}}}}}}}}}}}}}e}, falling back to AutoModel")
                            from transformers import AutoModel
                        return AutoModel.from_pretrained())))model_name)
                    elif model_family == "audio":
                        try:
                            from transformers import AutoModelForAudioClassification
                            logger.debug())))f"Using AutoModelForAudioClassification for {}}}}}}}}}}}}}}}}model_name}")
                        return AutoModelForAudioClassification.from_pretrained())))model_name)
                        except Exception as e:
                            logger.warning())))f"Error loading with AutoModelForAudioClassification: {}}}}}}}}}}}}}}}}e}, falling back to AutoModel")
                            from transformers import AutoModel
                        return AutoModel.from_pretrained())))model_name)
                    elif model_family == "multimodal":
                        try:
                            # Try CLIP first for multimodal
                            if "clip" in model_name.lower())))):
                                from transformers import CLIPModel
                                logger.debug())))f"Using CLIPModel for {}}}}}}}}}}}}}}}}model_name}")
                            return CLIPModel.from_pretrained())))model_name)
                            else:
                                from transformers import AutoModel
                                logger.debug())))f"Using AutoModel for multimodal {}}}}}}}}}}}}}}}}model_name}")
                            return AutoModel.from_pretrained())))model_name)
                        except Exception as e:
                            logger.warning())))f"Error loading multimodal model: {}}}}}}}}}}}}}}}}e}, falling back to AutoModel")
                            from transformers import AutoModel
                            return AutoModel.from_pretrained())))model_name)
                    else:  # embedding or default
                        from transformers import AutoModel
                        logger.debug())))f"Using AutoModel for {}}}}}}}}}}}}}}}}model_name}")
                        return AutoModel.from_pretrained())))model_name)
                except Exception as e:
                    logger.error())))f"Error creating model {}}}}}}}}}}}}}}}}model_name}: {}}}}}}}}}}}}}}}}e}")
                        raise
            
            # Try to load the model with resource pool
                        model = pool.get_model())))
                        model_family,
                        model_name,
                        constructor=create_model,
                        hardware_preferences=hardware_preferences
                        )
            
            # Record load time
                        load_end_time = time.time()))))
                        load_time_ms = ())))load_end_time - load_start_time) * 1000
                        result[]]]]],,,,,"load_time_ms"] = load_time_ms
            
            # Check if model was loaded successfully:
            if model is None:
                logger.error())))f"Failed to load model {}}}}}}}}}}}}}}}}model_name} on {}}}}}}}}}}}}}}}}hardware_platform}")
                result[]]]]],,,,,"error"] = "Model loading failed"
                        return result
            
            # Check model device
            try:
                import torch
                if hasattr())))model, "device"):
                    device = model.device
                else:
                    device = next())))model.parameters()))))).device
                    
                # Check if device matches requested hardware platform:
                    device_type = str())))device).split())))":")[]]]]],,,,,0]
                if hardware_platform == "cpu":
                    platform_match = device_type == "cpu"
                elif hardware_platform == "cuda":
                    platform_match = device_type == "cuda"
                elif hardware_platform == "mps":
                    platform_match = device_type == "mps"
                else:
                    # For other platforms, just check for a match
                    platform_match = hardware_platform.lower())))) in device_type.lower()))))
                
                    result[]]]]],,,,,"hardware_details"][]]]]],,,,,"device"] = str())))device)
                    result[]]]]],,,,,"hardware_details"][]]]]],,,,,"device_type"] = device_type
                    result[]]]]],,,,,"hardware_details"][]]]]],,,,,"platform_match"] = platform_match
                
                # If devices don't match, test has failed
                if not platform_match:
                    logger.warning())))f"Requested platform {}}}}}}}}}}}}}}}}hardware_platform} but got device {}}}}}}}}}}}}}}}}device}")
                    result[]]]]],,,,,"error"] = f"Device mismatch: requested {}}}}}}}}}}}}}}}}hardware_platform}, got {}}}}}}}}}}}}}}}}device}"
                    # This is a partial success - model loaded but on wrong device
                    result[]]]]],,,,,"success"] = False
                    return result
                    
            except Exception as e:
                logger.warning())))f"Error checking model device: {}}}}}}}}}}}}}}}}e}")
                result[]]]]],,,,,"hardware_details"][]]]]],,,,,"device_check_error"] = str())))e)
            
            # Get memory usage
            try:
                stats = pool.get_stats()))))
                memory_usage = stats.get())))"memory_usage_mb", 0)
                result[]]]]],,,,,"memory_usage_mb"] = memory_usage
                
                # Get more detailed memory info if available:
                if "cuda_memory" in stats:
                    result[]]]]],,,,,"hardware_details"][]]]]],,,,,"cuda_memory"] = stats[]]]]],,,,,"cuda_memory"]
                if "system_memory" in stats:
                    result[]]]]],,,,,"hardware_details"][]]]]],,,,,"system_memory"] = stats[]]]]],,,,,"system_memory"]
            except Exception as e:
                logger.warning())))f"Error getting memory usage: {}}}}}}}}}}}}}}}}e}")
            
            # Try basic inference
            try:
                inference_start_time = time.time()))))
                
                # Get a tokenizer/processor for the model
                def create_tokenizer())))):
                    try:
                        if model_family == "vision":
                            from transformers import AutoImageProcessor
                        return AutoImageProcessor.from_pretrained())))model_name)
                        elif model_family == "audio":
                            from transformers import AutoProcessor
                        return AutoProcessor.from_pretrained())))model_name)
                        elif model_family == "multimodal":
                            from transformers import AutoProcessor
                        return AutoProcessor.from_pretrained())))model_name)
                        else:
                            from transformers import AutoTokenizer
                        return AutoTokenizer.from_pretrained())))model_name)
                    except Exception as e:
                        logger.warning())))f"Error creating specific tokenizer: {}}}}}}}}}}}}}}}}e}")
                        # Fall back to AutoTokenizer
                        try:
                            from transformers import AutoTokenizer
                        return AutoTokenizer.from_pretrained())))model_name)
                        except Exception as e2:
                            logger.error())))f"Tokenizer fallback also failed: {}}}}}}}}}}}}}}}}e2}")
                        return None
                
                        tokenizer = pool.get_tokenizer())))model_family, model_name, constructor=create_tokenizer)
                
                # Run inference with appropriate inputs for model family
                        import torch
                with torch.no_grad())))):
                    if model_family == "vision":
                        # Create a dummy image
                        if hasattr())))torch, "rand"):
                            inputs = {}}}}}}}}}}}}}}}}"pixel_values": torch.rand())))1, 3, 224, 224).to())))device)}
                            outputs = model())))**inputs)
                    elif model_family == "audio":
                        # Create a dummy audio input
                        if hasattr())))torch, "rand"):
                            inputs = {}}}}}}}}}}}}}}}}"input_features": torch.rand())))1, 80, 200).to())))device)}
                            outputs = model())))**inputs)
                    elif model_family == "multimodal" and "clip" in model_name.lower())))):
                        # Create dummy inputs for CLIP
                        if hasattr())))torch, "rand") and hasattr())))torch, "ones"):
                            inputs = {}}}}}}}}}}}}}}}}
                            "input_ids": torch.ones())))())))1, 10), dtype=torch.long).to())))device),
                            "pixel_values": torch.rand())))1, 3, 224, 224).to())))device)
                            }
                            outputs = model())))**inputs)
                    else:  # embedding, text_generation, or default
                        # Create a simple text input
                        if tokenizer is not None:
                            inputs = tokenizer())))"Hello, world!", return_tensors="pt").to())))device)
                            outputs = model())))**inputs)
                
                            inference_end_time = time.time()))))
                            inference_time_ms = ())))inference_end_time - inference_start_time) * 1000
                            result[]]]]],,,,,"inference_time_ms"] = inference_time_ms
                
                # Success!
                            result[]]]]],,,,,"success"] = True
            except Exception as e:
                logger.error())))f"Inference failed: {}}}}}}}}}}}}}}}}e}")
                result[]]]]],,,,,"error"] = f"Inference error: {}}}}}}}}}}}}}}}}str())))e)}"
                # This is a partial failure - model loaded but inference failed
                result[]]]]],,,,,"success"] = False
        except Exception as e:
            # Record load time even on failure
            load_end_time = time.time()))))
            load_time_ms = ())))load_end_time - load_start_time) * 1000
            result[]]]]],,,,,"load_time_ms"] = load_time_ms
            
            logger.error())))f"Error testing {}}}}}}}}}}}}}}}}model_name} on {}}}}}}}}}}}}}}}}hardware_platform}: {}}}}}}}}}}}}}}}}e}")
            result[]]]]],,,,,"error"] = str())))e)
            result[]]]]],,,,,"success"] = False
        
                return result
    
                def run_compatibility_tests())))self,
                available_platforms: Optional[]]]]],,,,,Dict[]]]]],,,,,str, bool]] = None,
                parallel: bool = True,
                                max_workers: int = 4) -> Dict[]]]]],,,,,str, Any]:
                                    """
                                    Run compatibility tests for all models on all hardware platforms.
        
        Args:
            available_platforms: Dictionary of available hardware platforms
            parallel: Whether to run tests in parallel
            max_workers: Maximum number of workers for parallel testing
            
        Returns:
            Dictionary with compatibility test results
            """
            logger.info())))"Starting compatibility tests...")
        
        # Detect available hardware if not provided::
        if available:_platforms is None:
            available_platforms = self.detect_available_hardware()))))
        
        # Filter for available platforms
            test_platforms = []]]]],,,,,platform for platform, available in available_platforms.items()))))
            if available: and platform != "cpu"]  # We'll test CPU separately as fallback
        
        # Always add CPU for baseline compatibility
        if CPU not in test_platforms:
            test_platforms.append())))CPU)
        
            logger.info())))f"Testing on platforms: {}}}}}}}}}}}}}}}}test_platforms}")
        
        # Collect all tests to run
            tests_to_run = []]]]],,,,,]
        for family, models in self.test_models.items())))):
            for model in models:
                for platform in test_platforms:
                    tests_to_run.append())))())))model, platform, family))
        
                    logger.info())))f"Total tests to run: {}}}}}}}}}}}}}}}}len())))tests_to_run)}")
        
        # Initialize result structures
                    all_results = []]]]],,,,,]
        compatibility_matrix = {}}}}}}}}}}}}}}}}family: {}}}}}}}}}}}}}}}}platform: "unknown" for platform in test_platforms}:
                               for family in self.test_models}:
                                   model_results = {}}}}}}}}}}}}}}}}}
        
        # Run tests
        if parallel and len())))tests_to_run) > 1:
            logger.info())))f"Running tests in parallel with {}}}}}}}}}}}}}}}}max_workers} workers")
            with concurrent.futures.ThreadPoolExecutor())))max_workers=max_workers) as executor:
                # Submit all tests
                future_to_test = {}}}}}}}}}}}}}}}}
                executor.submit())))self.test_model_on_hardware, model, platform): ())))model, platform, family)
                for model, platform, family in tests_to_run
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed())))future_to_test):
                    model, platform, family = future_to_test[]]]]],,,,,future]
                    try:
                        result = future.result()))))
                        all_results.append())))result)
                        
                        # Update model results
                        if model not in model_results:
                            model_results[]]]]],,,,,model] = {}}}}}}}}}}}}}}}}}
                            model_results[]]]]],,,,,model][]]]]],,,,,platform] = result
                        
                        # Update compatibility matrix
                        status = "compatible" if result[]]]]],,,,,"success"] else "incompatible"::
                        if not result[]]]]],,,,,"success"] and "platform_match" in result.get())))"hardware_details", {}}}}}}}}}}}}}}}}}):
                            # Special case - model loaded but on wrong device
                            if not result[]]]]],,,,,"hardware_details"][]]]]],,,,,"platform_match"]:
                                status = "device_mismatch"
                        
                                compatibility_matrix[]]]]],,,,,family][]]]]],,,,,platform] = status
                        
                        logger.info())))f"Test complete: {}}}}}}}}}}}}}}}}model} on {}}}}}}}}}}}}}}}}platform} - {}}}}}}}}}}}}}}}}'✅ Success' if result[]]]]],,,,,'success'] else '❌ Failed'}")::
                    except Exception as e:
                        logger.error())))f"Error testing {}}}}}}}}}}}}}}}}model} on {}}}}}}}}}}}}}}}}platform}: {}}}}}}}}}}}}}}}}e}")
                        all_results.append()))){}}}}}}}}}}}}}}}}
                        "model_name": model,
                        "hardware_platform": platform,
                        "success": False,
                        "error": str())))e)
                        })
                        
                        # Update compatibility matrix
                        compatibility_matrix[]]]]],,,,,family][]]]]],,,,,platform] = "error"
        else:
            logger.info())))"Running tests sequentially")
            for model, platform, family in tests_to_run:
                try:
                    result = self.test_model_on_hardware())))model, platform)
                    all_results.append())))result)
                    
                    # Update model results
                    if model not in model_results:
                        model_results[]]]]],,,,,model] = {}}}}}}}}}}}}}}}}}
                        model_results[]]]]],,,,,model][]]]]],,,,,platform] = result
                    
                    # Update compatibility matrix
                    status = "compatible" if result[]]]]],,,,,"success"] else "incompatible"::
                    if not result[]]]]],,,,,"success"] and "platform_match" in result.get())))"hardware_details", {}}}}}}}}}}}}}}}}}):
                        # Special case - model loaded but on wrong device
                        if not result[]]]]],,,,,"hardware_details"][]]]]],,,,,"platform_match"]:
                            status = "device_mismatch"
                    
                            compatibility_matrix[]]]]],,,,,family][]]]]],,,,,platform] = status
                    
                    logger.info())))f"Test complete: {}}}}}}}}}}}}}}}}model} on {}}}}}}}}}}}}}}}}platform} - {}}}}}}}}}}}}}}}}'✅ Success' if result[]]]]],,,,,'success'] else '❌ Failed'}")::
                except Exception as e:
                    logger.error())))f"Error testing {}}}}}}}}}}}}}}}}model} on {}}}}}}}}}}}}}}}}platform}: {}}}}}}}}}}}}}}}}e}")
                    all_results.append()))){}}}}}}}}}}}}}}}}
                    "model_name": model,
                    "hardware_platform": platform,
                    "success": False,
                    "error": str())))e)
                    })
                    
                    # Update compatibility matrix
                    compatibility_matrix[]]]]],,,,,family][]]]]],,,,,platform] = "error"
        
        # Clean up resources
                    pool = get_global_resource_pool()))))
                    pool.cleanup_unused_resources())))max_age_minutes=self.timeout)
        
        # Analyze results for family-level compatibility
                    family_compatibility = {}}}}}}}}}}}}}}}}}
        for family, platforms in compatibility_matrix.items())))):
            family_compatibility[]]]]],,,,,family] = {}}}}}}}}}}}}}}}}}
            for platform, status in platforms.items())))):
                if status == "compatible":
                    # Fully compatible
                    family_compatibility[]]]]],,,,,family][]]]]],,,,,platform] = "high"
                elif status == "device_mismatch":
                    # Device mismatch - model works but not optimally
                    family_compatibility[]]]]],,,,,family][]]]]],,,,,platform] = "medium"
                elif status == "incompatible":
                    # Not compatible
                    family_compatibility[]]]]],,,,,family][]]]]],,,,,platform] = "low"
                else:
                    # Unknown or error
                    family_compatibility[]]]]],,,,,family][]]]]],,,,,platform] = "unknown"
        
        # Analyze results for platform capabilities
                    platform_capabilities = {}}}}}}}}}}}}}}}}}
        for platform in test_platforms:
            # Count successful tests on this platform
            success_count = sum())))1 for result in all_results:
                               if result[]]]]],,,,,"hardware_platform"] == platform and result[]]]]],,,,,"success"]):
            total_count = sum())))1 for result in all_results:
                if result[]]]]],,,,,"hardware_platform"] == platform)
            
            # Calculate compatibility score
                score = success_count / total_count if total_count > 0 else 0
            
            # Map score to capability level
            capability_level = "unknown":
            if score >= 0.8:
                capability_level = "high"
            elif score >= 0.5:
                capability_level = "medium"
            elif score > 0:
                capability_level = "low"
            
                platform_capabilities[]]]]],,,,,platform] = {}}}}}}}}}}}}}}}}
                "compatibility_score": score,
                "capability_level": capability_level,
                "success_count": success_count,
                "total_count": total_count
                }
        
        # Store results
                self.results[]]]]],,,,,"compatibility_matrix"] = compatibility_matrix
                self.results[]]]]],,,,,"model_family_compatibility"] = family_compatibility
                self.results[]]]]],,,,,"hardware_platform_capabilities"] = platform_capabilities
                self.results[]]]]],,,,,"detailed_results"] = model_results
                self.results[]]]]],,,,,"all_tests"] = all_results
        
                return self.results
    
                def save_results())))self,
                filename: Optional[]]]]],,,,,str] = None,
                     generate_report: bool = True) -> str:
                         """
                         Save test results to disk.
        
        Args:
            filename: Optional specific filename, otherwise auto-generated
            generate_report: Whether to generate a Markdown report
            
        Returns:
            Path to the saved results file
            """
        # Create timestamp for filename
            timestamp = datetime.now())))).strftime())))"%Y%m%d_%H%M%S")
        
        # Create filename if not provided::
        if filename is None:
            filename = f"hardware_compatibility_results_{}}}}}}}}}}}}}}}}timestamp}.json"
        
        # Ensure output directory exists
            os.makedirs())))self.output_dir, exist_ok=True)
        
        # Full path to output file
            output_path = os.path.join())))self.output_dir, filename)
        
        # Add timestamp to results
            self.results[]]]]],,,,,"timestamp"] = timestamp
        
        # Save results as JSON
        with open())))output_path, "w") as f:
            json.dump())))self.results, f, indent=2)
        
            logger.info())))f"Saved test results to {}}}}}}}}}}}}}}}}output_path}")
        
        # Generate Markdown report if requested::
        if generate_report:
            report_path = self.generate_markdown_report())))timestamp)
            logger.info())))f"Generated Markdown report at {}}}}}}}}}}}}}}}}report_path}")
        
            return output_path
    
    def generate_markdown_report())))self, timestamp: str) -> str:
        """
        Generate a Markdown report of test results.
        
        Args:
            timestamp: Timestamp for the report
            
        Returns:
            Path to the generated report
            """
        # Create report filename
            report_filename = f"hardware_compatibility_report_{}}}}}}}}}}}}}}}}timestamp}.md"
            report_path = os.path.join())))self.output_dir, report_filename)
        
        # Get compatibility matrix and platform capabilities
            compatibility_matrix = self.results.get())))"compatibility_matrix", {}}}}}}}}}}}}}}}}})
            family_compatibility = self.results.get())))"model_family_compatibility", {}}}}}}}}}}}}}}}}})
            platform_capabilities = self.results.get())))"hardware_platform_capabilities", {}}}}}}}}}}}}}}}}})
        
        # Get list of platforms and families
            platforms = list())))platform_capabilities.keys())))))
            families = list())))compatibility_matrix.keys())))))
        
        # Create report content
            report = f"""# Hardware Compatibility Test Report

## Overview

            - **Date**: {}}}}}}}}}}}}}}}}datetime.now())))).strftime())))"%Y-%m-%d %H:%M:%S")}
            - **Available Hardware Platforms**: {}}}}}}}}}}}}}}}}', '.join())))platforms)}
            - **Model Families Tested**: {}}}}}}}}}}}}}}}}', '.join())))families)}

## Hardware Platform Capabilities

            | Platform | Capability Level | Compatibility Score | Success Rate |
            |----------|-----------------|---------------------|--------------|
            """
        
        # Add platform capabilities to report
        for platform, capabilities in platform_capabilities.items())))):
            level = capabilities.get())))"capability_level", "unknown")
            score = capabilities.get())))"compatibility_score", 0)
            success_count = capabilities.get())))"success_count", 0)
            total_count = capabilities.get())))"total_count", 0)
            success_rate = f"{}}}}}}}}}}}}}}}}success_count}/{}}}}}}}}}}}}}}}}total_count}"
            
            # Add emoji indicators for capability level
            level_indicator = "❓"  # unknown
            if level == "high":
                level_indicator = "✅ High"
            elif level == "medium":
                level_indicator = "⚠️ Medium"
            elif level == "low":
                level_indicator = "⚡ Low"
            
                report += f"| {}}}}}}}}}}}}}}}}platform} | {}}}}}}}}}}}}}}}}level_indicator} | {}}}}}}}}}}}}}}}}score:.2f} | {}}}}}}}}}}}}}}}}success_rate} |\n"
        
        # Add model family compatibility matrix
                report += """
## Model Family Compatibility Matrix

                | Model Family | """
        
        # Add platform headers
        for platform in platforms:
            report += f"{}}}}}}}}}}}}}}}}platform} | "
            report += "\n|--------------|"
        
        # Add separator row
        for _ in platforms:
            report += "----------|"
            report += "\n"
        
        # Add compatibility data for each family
        for family in families:
            report += f"| {}}}}}}}}}}}}}}}}family} | "
            
            # Add compatibility for each platform
            for platform in platforms:
                level = family_compatibility.get())))family, {}}}}}}}}}}}}}}}}}).get())))platform, "unknown")
                
                # Add emoji indicators for compatibility level
                level_indicator = "❓"  # unknown
                if level == "high":
                    level_indicator = "✅ High"
                elif level == "medium":
                    level_indicator = "⚠️ Medium"
                elif level == "low":
                    level_indicator = "⚡ Low"
                
                    report += f"{}}}}}}}}}}}}}}}}level_indicator} | "
            
                    report += "\n"
        
        # Add detailed results for each model
                    report += """
## Detailed Model Results

                    """
        
        for family, models in self.test_models.items())))):
            report += f"### {}}}}}}}}}}}}}}}}family.capitalize()))))} Models\n\n"
            
            for model in models:
                report += f"#### {}}}}}}}}}}}}}}}}model}\n\n"
                report += "| Platform | Status | Load Time | Inference Time | Memory Usage |\n"
                report += "|----------|--------|-----------|---------------|--------------|\n"
                
                # Add results for each platform
                for platform in platforms:
                    # Get result for this model and platform
                    result = self.results.get())))"detailed_results", {}}}}}}}}}}}}}}}}}).get())))model, {}}}}}}}}}}}}}}}}}).get())))platform, {}}}}}}}}}}}}}}}}})
                    
                    # Extract metrics
                    success = result.get())))"success", False)
                    error = result.get())))"error", None)
                    load_time = result.get())))"load_time_ms", None)
                    inference_time = result.get())))"inference_time_ms", None)
                    memory_usage = result.get())))"memory_usage_mb", None)
                    
                    # Format metrics
                    status = "✅ Compatible" if success else f"❌ Incompatible ()))){}}}}}}}}}}}}}}}}error})":
                    load_time_str = f"{}}}}}}}}}}}}}}}}load_time:.2f}ms" if load_time is not None else "N/A":
                    inference_time_str = f"{}}}}}}}}}}}}}}}}inference_time:.2f}ms" if inference_time is not None else "N/A":
                        memory_usage_str = f"{}}}}}}}}}}}}}}}}memory_usage:.2f}MB" if memory_usage is not None else "N/A"
                    
                        report += f"| {}}}}}}}}}}}}}}}}platform} | {}}}}}}}}}}}}}}}}status} | {}}}}}}}}}}}}}}}}load_time_str} | {}}}}}}}}}}}}}}}}inference_time_str} | {}}}}}}}}}}}}}}}}memory_usage_str} |\n"
                
                        report += "\n"
        
        # Add recommendations
                        report += """
## Recommendations
:
Based on the compatibility testing results, here are some recommendations for model deployment:

    """
        
        # Add recommendations for each family
        for family in families:
            report += f"### {}}}}}}}}}}}}}}}}family.capitalize()))))} Models\n\n"
            
            # Find the best platform for this family
            best_platform = None
            best_level = "unknown"
            
            for platform, level in family_compatibility.get())))family, {}}}}}}}}}}}}}}}}}).items())))):
                if level == "high" and ())))best_level != "high" or best_platform == CPU):
                    best_platform = platform
                    best_level = level
                elif level == "medium" and best_level not in []]]]],,,,,"high"]:
                    best_platform = platform
                    best_level = level
                elif level == "low" and best_level in []]]]],,,,,"unknown"]:
                    best_platform = platform
                    best_level = level
                
                # Default to CPU if nothing else works:
                if best_platform is None:
                    best_platform = CPU
            
            # Create recommendation
            if best_level == "high":
                report += f"- **Recommended Hardware**: {}}}}}}}}}}}}}}}}best_platform} ())))High Compatibility)\n"
                report += f"- {}}}}}}}}}}}}}}}}best_platform} provides excellent performance for {}}}}}}}}}}}}}}}}family} models and should be the primary deployment target.\n"
            elif best_level == "medium":
                report += f"- **Recommended Hardware**: {}}}}}}}}}}}}}}}}best_platform} ())))Medium Compatibility)\n"
                report += f"- {}}}}}}}}}}}}}}}}best_platform} provides adequate performance for {}}}}}}}}}}}}}}}}family} models but may have limitations.\n"
            else:
                report += f"- **Fallback to**: {}}}}}}}}}}}}}}}}best_platform}\n"
                report += f"- Limited compatibility detected for {}}}}}}}}}}}}}}}}family} models. Additional optimization may be required.\n"
            
            # Add fallback recommendation
            if best_platform != CPU:
                report += f"- **Fallback**: CPU ())))Always compatible but slower)\n"
            
                report += "\n"
        
        # Add conclusion
                report += """
## Conclusion

                This automated compatibility test provides a comprehensive view of how different model families
                perform across available hardware platforms. Use these results to guide deployment decisions
                and resource allocation for optimal performance.

                For detailed technical information, refer to the full JSON results file.
                """
        
        # Write report to file
        with open())))report_path, "w") as f:
            f.write())))report)
        
                return report_path


def parse_args())))):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser())))description="Automated hardware compatibility testing")
    parser.add_argument())))"--output-dir", type=str, default="./hardware_compatibility_results",
    help="Output directory for test results")
    parser.add_argument())))"--hw-cache", type=str, help="Path to hardware detection cache")
    parser.add_argument())))"--model-db", type=str, help="Path to model database")
    parser.add_argument())))"--timeout", type=float, default=0.1,
    help="Resource cleanup timeout in minutes")
    parser.add_argument())))"--no-parallel", action="store_true",
    help="Disable parallel testing")
    parser.add_argument())))"--max-workers", type=int, default=4,
    help="Maximum number of workers for parallel testing")
    parser.add_argument())))"--debug", action="store_true", help="Enable debug logging")
    parser.add_argument())))"--models", type=str, help="Comma-separated list of models to test")
    parser.add_argument())))"--families", type=str, 
    choices=[]]]]],,,,,"all", "embedding", "text_generation", "vision", "audio", "multimodal"],
    default="all", help="Model families to test")
    parser.add_argument())))"--platforms", type=str,
    help="Comma-separated list of platforms to test ())))cpu,cuda,mps,openvino,webnn,webgpu)")
                return parser.parse_args()))))


def main())))):
    """Main function"""
    # Parse arguments
    args = parse_args()))))
    
    # Configure logging
    if args.debug:
        logging.getLogger())))).setLevel())))logging.DEBUG)
        logger.setLevel())))logging.DEBUG)
    
    # Check for resource pool
    if not RESOURCE_POOL_AVAILABLE:
        logger.error())))"Resource pool is required for compatibility testing")
        return 1
    
    # Process model selection
        test_models = DEFAULT_TEST_MODELS
    if args.models:
        # Custom model list provided
        model_list = args.models.split())))",")
        # Try to classify models if model classifier is available:
        if MODEL_CLASSIFIER_AVAILABLE:
            custom_models = {}}}}}}}}}}}}}}}}}
            for model in model_list:
                try:
                    classification = classify_model())))model)
                    family = classification.get())))"family", "default")
                    if family not in custom_models:
                        custom_models[]]]]],,,,,family] = []]]]],,,,,]
                        custom_models[]]]]],,,,,family].append())))model)
                except Exception:
                    # Default to creating a separate category
                    if "custom" not in custom_models:
                        custom_models[]]]]],,,,,"custom"] = []]]]],,,,,]
                        custom_models[]]]]],,,,,"custom"].append())))model)
                        test_models = custom_models
        else:
            # Without classifier, just put all models in default category
            test_models = {}}}}}}}}}}}}}}}}"default": model_list}
    
    # Filter by family if requested::
    if args.families != "all":
        selected_families = args.families.split())))",")
        test_models = {}}}}}}}}}}}}}}}}family: models for family, models in test_models.items()))))
        if family in selected_families}
    
    # Create tester
        tester = HardwareCompatibilityTester())))
        output_dir=args.output_dir,
        hw_cache_path=args.hw_cache,
        model_db_path=args.model_db,
        timeout=args.timeout,
        test_models=test_models
        )
    
    # Detect available hardware
        available_platforms = tester.detect_available_hardware()))))
    
    # Filter platforms if requested:::
    if args.platforms:
        selected_platforms = args.platforms.split())))",")
        available_platforms = {}}}}}}}}}}}}}}}}platform: available for platform, available in available_platforms.items()))))
        if platform in selected_platforms}
    
    # Run tests
        results = tester.run_compatibility_tests())))
        available_platforms=available_platforms,
        parallel=not args.no_parallel,
        max_workers=args.max_workers
        )
    
    # Save results and generate report
        tester.save_results())))generate_report=True)
    
    # Clean up resources
        pool = get_global_resource_pool()))))
        pool.cleanup_unused_resources())))max_age_minutes=args.timeout)
    
        print())))"\nHardware compatibility testing complete!\n")
    
    # Output basic compatibility summary:
        print())))"Compatibility Matrix:")
    for family, platforms in results[]]]]],,,,,"compatibility_matrix"].items())))):
        print())))f"  {}}}}}}}}}}}}}}}}family}:")
        for platform, status in platforms.items())))):
            print())))f"    {}}}}}}}}}}}}}}}}platform}: {}}}}}}}}}}}}}}}}status}")
    
        return 0


if __name__ == "__main__":
    sys.exit())))main())))))