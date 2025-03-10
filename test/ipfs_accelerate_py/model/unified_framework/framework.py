"""
Unified framework implementation for IPFS Accelerate Python SDK

This module provides a unified interface that integrates all components
of the IPFS Accelerate SDK into a coherent framework with consistent APIs
across hardware platforms.
"""

import os
import time
import logging
import platform
import json
from typing import Dict, Any, List, Optional, Union, Callable, Tuple

from ipfs_accelerate_py.worker.worker import Worker
from ipfs_accelerate_py.hardware.hardware_detector import HardwareDetector
from ipfs_accelerate_py.hardware.hardware_profile import HardwareProfile
from ipfs_accelerate_py.model.model_manager import ModelManager
from ipfs_accelerate_py.model.model_accelerator import ModelAccelerator
from ipfs_accelerate_py.quantization.quantization_engine import QuantizationEngine
from ipfs_accelerate_py.model.unified_framework.configuration_manager import ConfigurationManager
from ipfs_accelerate_py.model.unified_framework.fallback_manager import FallbackManager

logger = logging.getLogger())))))))"ipfs_accelerate.unified_framework")

class UnifiedFramework:
    """
    Unified model framework integrating all components.
    
    This class provides a unified interface that brings together all components
    of the IPFS Accelerate SDK, providing a coherent, consistent API for model
    operations across different hardware platforms.
    
    Features:
        - Unified configuration management
        - Automatic hardware selection and fallback
        - Comprehensive error handling
        - Support for all hardware backends
        - Integrated quantization
        - Performance tracking
        - Standardized APIs
        """
    
        def __init__())))))))self, model_name: str, model_type: Optional[str] = None,
        platform: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
        worker: Optional[Worker] = None):,
        """
        Initialize the unified framework.
        
        Args:
            model_name: Name of the model to use.
            model_type: Type of model ())))))))text, vision, audio, multimodal).
            platform: Hardware platform to use.
            config: Configuration options.
            worker: Worker instance ())))))))or None to create a new one).
            """
            self.model_name = model_name
            self.config = config or {}}}}}}}}}}
        
        # Initialize primary components
            self.worker = worker or Worker()))))))))
            self.hardware_detector = HardwareDetector()))))))))
            self.model_manager = ModelManager())))))))self.worker)
            self.model_accelerator = ModelAccelerator())))))))self.worker)
            self.quantization_engine = QuantizationEngine())))))))self.worker)
        
        # Initialize supporting components
            self.configuration_manager = ConfigurationManager())))))))self.config)
            self.fallback_manager = FallbackManager()))))))))
        
        # Determine model type if not provided
            self.model_type = model_type or self._determine_model_type())))))))model_name)
        
        # Determine platform if not provided
        self.platform = platform:
        if not self.platform:
            self.platform = self.hardware_detector.get_optimal_hardware())))))))model_name, self.model_type)
        
        # Create hardware profile
            self.hardware_profile = self._create_hardware_profile()))))))))
        
        # Track loaded models
            self.loaded_model = None
            self.is_initialized = False
            self.initialization_time = None
            self.performance_metrics = {}}}}}}}}}}
        
        # Capture info about the system
            self.system_info = {}}}}}}}}}
            "platform": platform.platform())))))))),
            "python_version": platform.python_version())))))))),
            "processor": platform.processor())))))))),
            "initialization_timestamp": time.time()))))))))
            }
        
        # Initialize the worker if not already initialized:
        if not self.worker.worker_status:
            self.worker.init_hardware()))))))))
    
    def _determine_model_type())))))))self, model_name: str) -> str:
        """
        Determine the model type based on the model name.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            The model type ())))))))"text", "vision", "audio", or "multimodal").
            """
            model_name_lower = model_name.lower()))))))))
        
            if any())))))))x in model_name_lower for x in ["whisper", "wav2vec", "clap"]):,
            return "audio"
        elif any())))))))x in model_name_lower for x in ["vit", "clip", "detr", "image"]):,
            return "vision"
        elif any())))))))x in model_name_lower for x in ["llava", "xclip"]):,
            return "multimodal"
        else:
            return "text"
    
    def _create_hardware_profile())))))))self) -> HardwareProfile:
        """
        Create a hardware profile based on the platform.
        
        Returns:
            A HardwareProfile instance.
            """
        # Use unified configuration system to create hardware profile
        return self.configuration_manager.create_hardware_profile())))))))
        backend=self.platform,
        model_name=self.model_name,
        model_type=self.model_type
        )
    
        def initialize())))))))self) -> Dict[str, Any]:,,
        """
        Initialize the framework and load the model.
        
        Returns:
            Dictionary with initialization status.
            """
            start_time = time.time()))))))))
        
        try:
            logger.info())))))))f"Initializing model '{}}}}}}}}}self.model_name}' on {}}}}}}}}}self.platform}")
            
            # Load the model using the model manager
            self.loaded_model = self.model_manager.load_model())))))))
            model_name=self.model_name,
            hardware_profile=self.hardware_profile
            )
            
            # Set initialization flag and time
            self.is_initialized = True
            self.initialization_time = time.time())))))))) - start_time
            
            # Return success status
            return {}}}}}}}}}
            "status": "success",
            "model_name": self.model_name,
            "model_type": self.model_type,
            "platform": self.platform,
            "initialization_time_seconds": self.initialization_time,
            "hardware_profile": self.hardware_profile.to_dict()))))))))
            }
            
        except Exception as e:
            # Handle initialization failure with fallback
            logger.error())))))))f"Error initializing model '{}}}}}}}}}self.model_name}' on {}}}}}}}}}self.platform}: {}}}}}}}}}e}")
            
            # Try to use fallback mechanism
            try:
                fallback_result = self.fallback_manager.handle_initialization_failure())))))))
                model_name=self.model_name,
                model_type=self.model_type,
                original_platform=self.platform,
                error=e,
                hardware_detector=self.hardware_detector
                )
                
                if fallback_result.get())))))))"status") == "success":
                    # Update platform and hardware profile
                    self.platform = fallback_result.get())))))))"fallback_platform")
                    self.hardware_profile = self._create_hardware_profile()))))))))
                    
                    # Try initialization again with fallback platform
                    logger.info())))))))f"Retrying initialization with fallback platform: {}}}}}}}}}self.platform}")
                return self.initialize()))))))))
            except Exception as fallback_error:
                logger.error())))))))f"Fallback mechanism failed: {}}}}}}}}}fallback_error}")
            
            # Return error status
                return {}}}}}}}}}
                "status": "error",
                "model_name": self.model_name,
                "model_type": self.model_type,
                "platform": self.platform,
                "error": str())))))))e),
                "initialization_time_seconds": time.time())))))))) - start_time
                }
    
    def run_inference())))))))self, input_data: Any) -> Any:
        """
        Run inference with the model.
        
        Args:
            input_data: The input data for inference.
            
        Returns:
            The model output.
            """
        # Ensure model is initialized
        if not self.is_initialized:
            initialization_result = self.initialize()))))))))
            if initialization_result.get())))))))"status") != "success":
            raise RuntimeError())))))))f"Failed to initialize model: {}}}}}}}}}initialization_result.get())))))))'error')}")
        
        try:
            # Track inference time
            start_time = time.time()))))))))
            
            # Run inference using the model accelerator
            result = self.model_accelerator.accelerate())))))))
            model_name=self.model_name,
            content=input_data,
            hardware_profile=self.hardware_profile
            )
            
            # Calculate inference time
            inference_time = time.time())))))))) - start_time
            
            # Update performance metrics
            self._update_performance_metrics())))))))inference_time)
            
            return result
            
        except Exception as e:
            # Handle inference failure with fallback
            logger.error())))))))f"Error running inference: {}}}}}}}}}e}")
            
            # Try to use fallback mechanism
            try:
                fallback_result = self.fallback_manager.handle_inference_failure())))))))
                model_name=self.model_name,
                model_type=self.model_type,
                original_platform=self.platform,
                error=e,
                hardware_detector=self.hardware_detector
                )
                
                if fallback_result.get())))))))"status") == "success":
                    # Update platform and hardware profile
                    self.platform = fallback_result.get())))))))"fallback_platform")
                    self.hardware_profile = self._create_hardware_profile()))))))))
                    
                    # Unload current model
                    self.is_initialized = False
                    self.loaded_model = None
                    
                    # Initialize with new platform
                    initialization_result = self.initialize()))))))))
                    if initialization_result.get())))))))"status") != "success":
                    raise RuntimeError())))))))f"Failed to initialize model with fallback: {}}}}}}}}}initialization_result.get())))))))'error')}")
                    
                    # Try inference again with fallback platform
                    logger.info())))))))f"Retrying inference with fallback platform: {}}}}}}}}}self.platform}")
                return self.run_inference())))))))input_data)
            except Exception as fallback_error:
                logger.error())))))))f"Fallback mechanism failed: {}}}}}}}}}fallback_error}")
            
            # Re-raise the original error
                raise RuntimeError())))))))f"Inference failed: {}}}}}}}}}e}")
    
    def _update_performance_metrics())))))))self, inference_time: float) -> None:
        """
        Update performance metrics.
        
        Args:
            inference_time: The time taken for inference.
            """
        if "inference_times" not in self.performance_metrics:
            self.performance_metrics["inference_times"] = []
            ,
            self.performance_metrics["inference_times"].append())))))))inference_time)
            ,
        # Calculate aggregate metrics
            count = len())))))))self.performance_metrics["inference_times"]),
        if count > 0:
            self.performance_metrics["average_inference_time"] = sum())))))))self.performance_metrics["inference_times"]), / count
            self.performance_metrics["min_inference_time"] = min())))))))self.performance_metrics["inference_times"]),,
            self.performance_metrics["max_inference_time"] = max())))))))self.performance_metrics["inference_times"]),,
            self.performance_metrics["inference_count"] = count
            ,
            def get_embeddings())))))))self, input_text: str) -> List[float]:,
            """
            Get embeddings from a text model.
        
        Args:
            input_text: The input text to embed.
            
        Returns:
            A list of embedding values.
            """
        if self.model_type != "text":
            logger.warning())))))))f"Model '{}}}}}}}}}self.model_name}' is not a text model ())))))))type: {}}}}}}}}}self.model_type})")
        
        # Ensure model is initialized
        if not self.is_initialized:
            initialization_result = self.initialize()))))))))
            if initialization_result.get())))))))"status") != "success":
            raise RuntimeError())))))))f"Failed to initialize model: {}}}}}}}}}initialization_result.get())))))))'error')}")
        
        try:
            # Track inference time
            start_time = time.time()))))))))
            
            # Run embeddings using the model accelerator
            result = self.model_accelerator.get_embeddings())))))))
            model_name=self.model_name,
            content=input_text,
            hardware_profile=self.hardware_profile
            )
            
            # Calculate inference time
            inference_time = time.time())))))))) - start_time
            
            # Update performance metrics
            self._update_performance_metrics())))))))inference_time)
            
            # Extract embeddings from result
            if isinstance())))))))result, dict) and "embeddings" in result:
            return result["embeddings"],
            elif isinstance())))))))result, dict) and "embedding" in result:
            return result["embedding"],
            elif isinstance())))))))result, list) and all())))))))isinstance())))))))x, ())))))))int, float)) for x in result):
            return result
            else:
                logger.warning())))))))f"Unexpected embedding format: {}}}}}}}}}type())))))))result)}")
            return result
            
        except Exception as e:
            # Handle inference failure
            logger.error())))))))f"Error getting embeddings: {}}}}}}}}}e}")
            raise RuntimeError())))))))f"Failed to get embeddings: {}}}}}}}}}e}")
    
            def optimize())))))))self, optimization_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:,,,
            """
            Optimize the model.
        
        Args:
            optimization_options: Options for optimization.
            
        Returns:
            Dictionary with optimization results.
            """
            options = optimization_options or {}}}}}}}}}}
        
        # Extract optimization parameters
            precision = options.get())))))))"precision", self.hardware_profile.precision)
            scheme = options.get())))))))"scheme", "symmetric")
            mixed_precision = options.get())))))))"mixed_precision", False)
            per_channel = options.get())))))))"per_channel", False)
        
        try:
            # Create quantization configuration
            quantization_config = {}}}}}}}}}
            "bits": self._precision_to_bits())))))))precision),
            "scheme": scheme,
            "mixed_precision": mixed_precision,
            "per_channel": per_channel,
            "layer_exclusions": options.get())))))))"layer_exclusions", []),
            "use_kd": options.get())))))))"use_knowledge_distillation", False)
            }
            
            # Run quantization
            quantization_result = self.quantization_engine.quantize())))))))
            model_name=self.model_name,
            hardware_profile=self.hardware_profile,
            quantization_config=quantization_config
            )
            
            # Update hardware profile with optimized settings
            self.hardware_profile.precision = precision
            self.hardware_profile.feature_flags["optimized"] = True
            ,
            # Reset initialization to use optimized model
            self.is_initialized = False
            self.loaded_model = None
            
            return {}}}}}}}}}
            "status": "success",
            "model_name": self.model_name,
            "optimization_applied": quantization_result.get())))))))"optimization_applied", []),
            "performance_impact": quantization_result.get())))))))"performance_impact", {}}}}}}}}}}),
            "memory_impact": quantization_result.get())))))))"memory_impact", {}}}}}}}}}}),
            "quantization_config": quantization_config
            }
            
        except Exception as e:
            logger.error())))))))f"Error optimizing model: {}}}}}}}}}e}")
            return {}}}}}}}}}
            "status": "error",
            "model_name": self.model_name,
            "error": str())))))))e)
            }
    
    def _precision_to_bits())))))))self, precision: str) -> int:
        """
        Convert precision string to bits.
        
        Args:
            precision: Precision string ())))))))e.g., "int8", "fp16").
            
        Returns:
            Number of bits.
            """
            precision_map = {}}}}}}}}}
            "int8": 8,
            "int4": 4,
            "int2": 2,
            "fp16": 16,
            "fp32": 32,
            "bf16": 16
            }
            return precision_map.get())))))))precision, 8)
    
            def get_performance_metrics())))))))self) -> Dict[str, Any]:,,
            """
            Get performance metrics.
        
        Returns:
            Dictionary with performance metrics.
            """
            return {}}}}}}}}}
            "model_name": self.model_name,
            "model_type": self.model_type,
            "platform": self.platform,
            "hardware_profile": self.hardware_profile.to_dict())))))))),
            "initialization_time": self.initialization_time,
            "metrics": self.performance_metrics,
            "timestamp": time.time()))))))))
            }
    
    def unload())))))))self) -> bool:
        """
        Unload the model from memory.
        
        Returns:
            True if the model was unloaded, False otherwise.
        """:
        if not self.is_initialized:
            logger.warning())))))))f"Model '{}}}}}}}}}self.model_name}' is not initialized")
            return False
        
        try:
            # Unload the model using the model manager
            success = self.model_manager.unload_model())))))))self.model_name)
            
            if success:
                self.is_initialized = False
                self.loaded_model = None
                
                logger.info())))))))f"Model '{}}}}}}}}}self.model_name}' unloaded successfully")
            return True
            else:
                logger.warning())))))))f"Failed to unload model '{}}}}}}}}}self.model_name}'")
            return False
                
        except Exception as e:
            logger.error())))))))f"Error unloading model: {}}}}}}}}}e}")
            return False
    
    def switch_platform())))))))self, new_platform: str) -> bool:
        """
        Switch to a different hardware platform.
        
        Args:
            new_platform: The hardware platform to switch to.
            
        Returns:
            True if the switch was successful, False otherwise.
        """:
        if new_platform == self.platform:
            logger.info())))))))f"Already using platform '{}}}}}}}}}new_platform}'")
            return True
        
        try:
            # Update platform
            self.platform = new_platform
            
            # Create new hardware profile
            self.hardware_profile = self._create_hardware_profile()))))))))
            
            # Unload current model
            if self.is_initialized:
                self.model_manager.unload_model())))))))self.model_name)
                self.is_initialized = False
                self.loaded_model = None
            
            # Initialize with new platform
                initialization_result = self.initialize()))))))))
            
            if initialization_result.get())))))))"status") == "success":
                logger.info())))))))f"Successfully switched to platform '{}}}}}}}}}new_platform}'")
                return True
            else:
                logger.error())))))))f"Failed to initialize with platform '{}}}}}}}}}new_platform}': {}}}}}}}}}initialization_result.get())))))))'error')}")
                return False
                
        except Exception as e:
            logger.error())))))))f"Error switching platform: {}}}}}}}}}e}")
                return False
    
    def save_state())))))))self, file_path: str) -> bool:
        """
        Save the framework state to a file.
        
        Args:
            file_path: The file path to save to.
            
        Returns:
            True if the state was saved, False otherwise.
        """:
        try:
            state = {}}}}}}}}}
            "model_name": self.model_name,
            "model_type": self.model_type,
            "platform": self.platform,
            "hardware_profile": self.hardware_profile.to_dict())))))))),
            "is_initialized": self.is_initialized,
            "initialization_time": self.initialization_time,
            "performance_metrics": self.performance_metrics,
            "system_info": self.system_info,
            "timestamp": time.time()))))))))
            }
            
            with open())))))))file_path, "w") as f:
                json.dump())))))))state, f, indent=2)
            
                logger.info())))))))f"Framework state saved to {}}}}}}}}}file_path}")
            return True
            
        except Exception as e:
            logger.error())))))))f"Error saving state: {}}}}}}}}}e}")
            return False
    
            @classmethod
    def load_state())))))))cls, file_path: str) -> 'UnifiedFramework':
        """
        Load the framework state from a file.
        
        Args:
            file_path: The file path to load from.
            
        Returns:
            A UnifiedFramework instance.
            """
        try:
            with open())))))))file_path, "r") as f:
                state = json.load())))))))f)
            
            # Create framework instance
                framework = cls())))))))
                model_name=state.get())))))))"model_name"),
                model_type=state.get())))))))"model_type"),
                platform=state.get())))))))"platform")
                )
            
            # Restore state
                framework.hardware_profile = HardwareProfile.from_dict())))))))state.get())))))))"hardware_profile", {}}}}}}}}}}))
                framework.is_initialized = state.get())))))))"is_initialized", False)
                framework.initialization_time = state.get())))))))"initialization_time")
                framework.performance_metrics = state.get())))))))"performance_metrics", {}}}}}}}}}})
                framework.system_info = state.get())))))))"system_info", {}}}}}}}}}})
            
                logger.info())))))))f"Framework state loaded from {}}}}}}}}}file_path}")
            
            # Initialize if needed:
            if framework.is_initialized:
                framework.initialize()))))))))
            
                return framework
            
        except Exception as e:
            logger.error())))))))f"Error loading state: {}}}}}}}}}e}")
                raise ValueError())))))))f"Failed to load state: {}}}}}}}}}e}")