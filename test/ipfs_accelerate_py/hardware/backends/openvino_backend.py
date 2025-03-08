"""
OpenVINO backend implementation for IPFS Accelerate SDK.

This module provides OpenVINO-specific functionality for model acceleration.
"""

import os
import logging
import random
import time
from typing import Dict, Any, List, Optional, Union, Tuple
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ipfs_accelerate.hardware.openvino")

# OpenVINO device map for readable device types
DEVICE_MAP = {
    "CPU": "cpu",
    "GPU": "gpu",
    "MYRIAD": "vpu",
    "HDDL": "vpu",
    "GNA": "gna",
    "HETERO": "hetero",
    "MULTI": "multi",
    "AUTO": "auto"
}

class OpenVINOBackend:
    """
    OpenVINO backend for model acceleration.
    
    This class provides functionality for running models with Intel OpenVINO on various
    hardware including CPU, Intel GPUs, and VPUs.
    """
    
    def __init__(self, config=None):
        """
        Initialize OpenVINO backend.
        
        Args:
            config: Configuration instance (optional)
        """
        self.config = config or {}
        self.models = {}
        self._available_devices = []
        self._device_info = {}
        self._compiler_info = {}
        self._core = None
        self._model_cache = {}
        self._cache_dir = self.config.get("cache_dir", os.path.expanduser("~/.cache/ipfs_accelerate/openvino"))
        
        # Create cache directory if it doesn't exist
        os.makedirs(self._cache_dir, exist_ok=True)
        
        # Check if OpenVINO is available
        self._check_availability()
    
    def _check_availability(self) -> bool:
        """
        Check if OpenVINO is available and collect device information.
        
        Returns:
            True if OpenVINO is available, False otherwise.
        """
        try:
            import openvino
            
            # Store version
            self._version = openvino.__version__
            
            # Try to initialize OpenVINO Core
            try:
                from openvino.runtime import Core
                core = Core()
                self._core = core
                
                # Get available devices
                available_devices = core.available_devices
                self._available_devices = available_devices
                
                # Collect information about each device
                for device in available_devices:
                    try:
                        device_type = device.split('.')[0]
                        readable_type = DEVICE_MAP.get(device_type, "unknown")
                        
                        # Get full device info
                        try:
                            full_device_name = core.get_property(device, "FULL_DEVICE_NAME")
                        except Exception:
                            full_device_name = f"Unknown {device_type} device"
                        
                        device_info = {
                            "device_name": device,
                            "device_type": readable_type,
                            "full_name": full_device_name,
                            "supports_fp32": True,  # All devices support FP32
                            "supports_fp16": device_type in ["GPU", "CPU", "MYRIAD", "HDDL"],  # Most devices support FP16
                            "supports_int8": device_type in ["GPU", "CPU"],  # Only some devices support INT8
                        }
                        
                        # Add additional properties for specific device types
                        if device_type == "CPU":
                            try:
                                cpu_threads = core.get_property(device, "CPU_THREADS_NUM")
                                device_info["cpu_threads"] = cpu_threads
                            except:
                                pass
                        elif device_type == "GPU":
                            try:
                                gpu_device_name = core.get_property(device, "DEVICE_ARCHITECTURE")
                                device_info["architecture"] = gpu_device_name
                            except:
                                pass
                        
                        self._device_info[device] = device_info
                    except Exception as e:
                        logger.warning(f"Could not get detailed info for OpenVINO device {device}: {str(e)}")
                
                # Try to get compiler info
                try:
                    self._compiler_info = {
                        "optimization_capabilities": core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
                    }
                except:
                    pass
                
                self._available = True
                logger.info(f"OpenVINO {self._version} is available with devices: {', '.join(available_devices)}")
                return True
            except Exception as e:
                self._available = False
                logger.warning(f"Failed to initialize OpenVINO Core: {str(e)}")
                return False
        except ImportError:
            self._available = False
            logger.warning("OpenVINO is not installed")
            return False
    
    def is_available(self) -> bool:
        """
        Check if OpenVINO is available.
        
        Returns:
            True if OpenVINO is available, False otherwise.
        """
        return getattr(self, '_available', False)
    
    def get_device_info(self, device_name: str = "CPU") -> Dict[str, Any]:
        """
        Get OpenVINO device information.
        
        Args:
            device_name: Device name to get information for.
            
        Returns:
            Dictionary with device information.
        """
        if not self.is_available():
            return {"available": False, "message": "OpenVINO is not available"}
        
        if device_name not in self._device_info:
            logger.warning(f"Device {device_name} not found")
            return {"available": False, "message": f"Device {device_name} not found"}
        
        return self._device_info[device_name]
    
    def get_all_devices(self) -> List[Dict[str, Any]]:
        """
        Get information about all available OpenVINO devices.
        
        Returns:
            List of dictionaries with device information.
        """
        if not self.is_available():
            return []
        
        return [self._device_info[device] for device in self._available_devices]
        
    def _apply_fp16_transformations(self, model):
        """
        Apply FP16 precision transformations to the model.
        
        Args:
            model: OpenVINO model to transform
            
        Returns:
            Transformed model with FP16 precision
        """
        try:
            import openvino as ov
            from openvino.runtime import set_batch
            from openvino.runtime.passes import Manager, GraphRewrite, Matcher, WrapType
            
            # Create a pass manager
            pass_manager = Manager()
            
            # Apply FP16 conversion transformation
            pass_manager.register_pass("convert_fp32_to_fp16", GraphRewrite())
            
            # Apply the transformations
            pass_manager.run_passes(model)
            
            logger.info("Applied FP16 transformations to the model")
            return model
        except Exception as e:
            logger.warning(f"Failed to apply FP16 transformations: {str(e)}")
            return model  # Return original model if transformation fails
    
    def _apply_int8_transformations(self, model, calibration_dataset=None):
        """
        Apply INT8 precision transformations to the model.
        
        For full INT8 quantization, a calibration dataset is recommended.
        This function supports both basic INT8 compatibility without
        calibration data and advanced INT8 quantization with calibration.
        
        Args:
            model: OpenVINO model to transform
            calibration_dataset: Optional calibration data for advanced quantization
            
        Returns:
            Transformed model with INT8 precision optimizations
        """
        try:
            import openvino as ov
            from openvino.runtime import Core, Type
            
            # If no calibration data is provided, apply basic transformations
            if calibration_dataset is None:
                logger.info("No calibration data provided, applying basic INT8 compatibility.")
                
                try:
                    # Create a pass manager for model transformations
                    from openvino.runtime.passes import Manager, ConstantFolding
                    pass_manager = Manager()
                    
                    # Add basic transformations that help with INT8 compatibility
                    pass_manager.register_pass(ConstantFolding())
                    
                    # Apply the passes
                    pass_manager.run_passes(model)
                    
                    logger.info("Applied basic INT8 compatibility transformations to the model")
                    return model
                except Exception as e:
                    logger.warning(f"Failed to apply basic INT8 transformations: {str(e)}")
                    logger.warning("Falling back to original model")
                    return model
                
            # Advanced INT8 quantization with calibration data
            else:
                logger.info("Applying advanced INT8 quantization with calibration data")
                
                try:
                    # Check for NNCF API first (newer approach)
                    try:
                        from openvino.tools import pot
                        from openvino.tools.pot.api import Metric, DataLoader
                        from openvino.tools.pot.engines.ie_engine import IEEngine
                        from openvino.tools.pot.graph import load_model, save_model
                        from openvino.tools.pot.algorithms.quantization import DefaultQuantization
                        
                        # Custom calibration data loader
                        class CalibrationLoader(DataLoader):
                            def __init__(self, data):
                                self.data = data
                                self.indices = list(range(len(data)))
                                
                            def __len__(self):
                                return len(self.data)
                                
                            def __getitem__(self, index):
                                return self.data[index]
                        
                        # Advanced quantization parameters
                        quantization_params = {
                            'target_device': 'ANY',  # Can work on any device
                            'preset': 'mixed',  # Use mixed precision for better accuracy/performance balance
                            'stat_subset_size': min(300, len(calibration_dataset)),
                            'stat_subset_seed': 42,  # For reproducibility
                            'use_layerwise_tuning': True,  # Enable per-layer optimization
                            'inplace_statistics': True,  # Compute statistics in-place
                            'granularity': 'channel'  # Apply channel-wise quantization
                        }
                        
                        # Configure quantization algorithm
                        algorithm = [{
                            'name': 'DefaultQuantization',
                            'params': quantization_params
                        }]
                        
                        # Create data loader
                        data_loader = CalibrationLoader(calibration_dataset)
                        
                        # Create engine for quantization
                        engine = IEEngine(config={"device": "CPU"}, data_loader=data_loader)
                        
                        # Create quantization algorithm
                        algo = DefaultQuantization(preset=algorithm)
                        
                        # Apply quantization
                        quantized_model = algo.run(model, data_loader)
                        
                        logger.info("Applied advanced INT8 quantization with NNCF/POT API")
                        return quantized_model
                        
                    except (ImportError, AttributeError) as e:
                        # Try legacy POT API
                        logger.info(f"New POT API not available, trying legacy POT: {str(e)}")
                        try:
                            from openvino.tools.pot import DataLoader, IEEngine
                            from openvino.tools.pot.algorithms.quantization import DefaultQuantization
                            from openvino.tools.pot.graph import load_model, save_model
                            
                            # Get default quantization parameters
                            ignored_scopes = []  # Layers to skip during quantization
                            preset = [
                                {
                                    'name': 'DefaultQuantization',
                                    'params': {
                                        'target_device': 'CPU',  # Target hardware device
                                        'preset': 'performance',  # performance or accuracy focus
                                        'stat_subset_size': min(300, len(calibration_dataset)),  # Num samples from calibration dataset
                                        'ignored_scope': ignored_scopes
                                    }
                                }
                            ]
                            
                            # Create a custom data loader for the calibration dataset
                            class CalibrationLoader(DataLoader):
                                def __init__(self, data):
                                    self.data = data
                                    self.index = 0
                                    
                                def __len__(self):
                                    return len(self.data)
                                    
                                def __getitem__(self, index):
                                    return self.data[index]
                            
                            # Create data loader
                            data_loader = CalibrationLoader(calibration_dataset)
                            
                            # Create engine for quantization
                            engine = IEEngine(config={"device": "CPU"}, data_loader=data_loader)
                            
                            # Create quantization algorithm
                            algo = DefaultQuantization(preset=preset)
                            
                            # Apply quantization
                            quantized_model = algo.run(model, data_loader)
                            
                            logger.info("Applied advanced INT8 quantization with legacy POT API")
                            return quantized_model
                        except (ImportError, AttributeError, Exception) as e:
                            logger.warning(f"POT API not available: {str(e)}")
                            # Fall back to simplified approach
                            raise ImportError("POT API not available")
                        
                    except Exception as e:
                        logger.warning(f"Failed to apply quantization with POT API: {str(e)}")
                        # Fall back to simplified approach
                        raise ImportError("Quantization failed with POT API")
                        
                except (ImportError, Exception):
                    # Fallback for older OpenVINO versions or when POT is not available
                    logger.warning("openvino.tools.pot not available, falling back to nGraph quantization")
                    
                    # Use simplified quantization approach
                    try:
                        from openvino.runtime import Core, Model, PartialShape
                        
                        # Set model precision to INT8 for compatible layers
                        for node in model.get_ops():
                            # Skip specific node types not suitable for INT8
                            if node.get_type_name() in ["Const", "Result", "Parameter"]:
                                continue
                                
                            for output_idx in range(len(node.outputs())):
                                node.set_output_type(output_idx, Type.i8, False)
                                
                        logger.info("Applied simplified INT8 transformations via type conversion")
                        return model
                    except Exception as e:
                        logger.warning(f"Failed to apply simplified INT8 transformations: {str(e)}")
                        logger.warning("Returning original model without INT8 transformations")
                        return model
                    
        except Exception as e:
            logger.warning(f"Failed to apply INT8 transformations: {str(e)}")
            return model  # Return original model if transformation fails
            
    def _apply_mixed_precision_transformations(self, model, config=None):
        """
        Apply mixed precision transformations to the model.
        
        This enables different precision formats for different parts of the model
        based on their sensitivity to quantization.
        
        Args:
            model: OpenVINO model to transform
            config: Configuration for mixed precision
            
        Returns:
            Transformed model with mixed precision
        """
        try:
            import openvino as ov
            from openvino.runtime import Core, Type
            
            config = config or {}
            logger.info("Applying mixed precision transformations")
            
            # Try to use NNCF for advanced mixed precision transformations
            try:
                # Check if nncf is available
                import nncf
                from openvino.tools import pot
                from openvino.runtime.passes import Manager, GraphRewrite
                
                # Get precision configuration for different layer types
                precision_config = config.get("precision_config", {
                    # Attention layers are more sensitive to precision loss
                    "attention": "FP16",
                    # Matrix multiplication operations
                    "matmul": "INT8",
                    # Default precision for other layers
                    "default": "INT8"
                })
                
                # Create a pass manager
                pass_manager = Manager()
                
                # Set precision for different node types
                for node in model.get_ops():
                    node_type = node.get_type_name()
                    node_name = node.get_friendly_name()
                    
                    # Apply different precision based on layer type
                    if "matmul" in node_type.lower() and precision_config.get("matmul") == "INT8":
                        for output_idx in range(len(node.outputs())):
                            node.set_output_type(output_idx, Type.i8, False)
                            
                    elif any(attn_name in node_name.lower() for attn_name in 
                             ["attention", "self_attn", "mha"]) and precision_config.get("attention") == "FP16":
                        for output_idx in range(len(node.outputs())):
                            node.set_output_type(output_idx, Type.f16, False)
                    
                    elif precision_config.get("default") == "INT8":
                        # Default to INT8 for compatible operations
                        if node_type not in ["Const", "Result", "Parameter"]:
                            for output_idx in range(len(node.outputs())):
                                node.set_output_type(output_idx, Type.i8, False)
                    
                    elif precision_config.get("default") == "FP16":
                        # Default to FP16
                        for output_idx in range(len(node.outputs())):
                            node.set_output_type(output_idx, Type.f16, False)
                
                # Apply constant folding for optimization
                from openvino.runtime.passes import ConstantFolding
                pass_manager.register_pass(ConstantFolding())
                
                # Run the passes
                pass_manager.run_passes(model)
                
                logger.info("Applied mixed precision transformations with layer-specific settings")
                return model
                
            except ImportError:
                logger.warning("Advanced mixed precision libraries not available")
                
                # Fallback to basic mixed precision implementation
                # For simple approach, just apply INT8 to most layers but keep sensitive ones in FP16
                sensitive_op_types = [
                    "MatMul", "Softmax", "LayerNorm", "GRUCell", "LSTMCell", "RNNCell"
                ]
                
                for node in model.get_ops():
                    node_type = node.get_type_name()
                    
                    # Skip constant and parameter nodes
                    if node_type in ["Const", "Result", "Parameter"]:
                        continue
                        
                    # Keep sensitive operations in FP16 
                    if node_type in sensitive_op_types:
                        for output_idx in range(len(node.outputs())):
                            node.set_output_type(output_idx, Type.f16, False)
                    else:
                        # Set other operations to INT8
                        for output_idx in range(len(node.outputs())):
                            node.set_output_type(output_idx, Type.i8, False)
                            
                logger.info("Applied basic mixed precision transformations (FP16 for sensitive ops, INT8 for others)")
                return model
                
        except Exception as e:
            logger.warning(f"Failed to apply mixed precision transformations: {str(e)}")
            return model  # Return original model if transformation fails
            
    def _generate_dummy_calibration_data(self, model_info, num_samples=10):
        """
        Generate simple dummy calibration data for INT8 quantization.
        
        For real-world usage, this should be replaced with actual
        representative data for the model being quantized.
        
        Args:
            model_info: Dictionary with model information including inputs shape
            num_samples: Number of samples to generate
            
        Returns:
            List of dictionaries with input data
        """
        try:
            import numpy as np
            
            if not model_info or "inputs_info" not in model_info:
                logger.warning("No model info provided for calibration data generation")
                return None
                
            inputs_info = model_info["inputs_info"]
            
            # Create dummy calibration dataset
            calibration_dataset = []
            
            for _ in range(num_samples):
                sample = {}
                
                for input_name, input_shape in inputs_info.items():
                    # Create random data with appropriate shape
                    input_type = "float32"  # Default type
                    
                    # For input_ids or similar, use integer data
                    if "ids" in input_name.lower() or "token" in input_name.lower() or "index" in input_name.lower():
                        input_type = "int32"
                        # Generate random integers
                        sample[input_name] = np.random.randint(0, 1000, size=input_shape).astype("int32")
                    elif "mask" in input_name.lower():
                        # For masks, generate 0s and 1s
                        sample[input_name] = np.random.randint(0, 2, size=input_shape).astype("int32")
                    else:
                        # For other inputs, generate floats
                        sample[input_name] = np.random.rand(*input_shape).astype("float32")
                
                calibration_dataset.append(sample)
                
            logger.info(f"Generated {num_samples} dummy calibration samples for INT8 quantization")
            return calibration_dataset
            
        except Exception as e:
            logger.warning(f"Failed to generate calibration data: {str(e)}")
            return None
            
    def _get_cached_model_path(self, model_name: str, precision: str, device: str) -> Optional[str]:
        """
        Get path to cached model if it exists.
        
        Args:
            model_name: Name of the model
            precision: Precision format (FP32, FP16, INT8)
            device: Target device
            
        Returns:
            Path to cached model if it exists, None otherwise
        """
        cache_key = f"{model_name}_{precision}_{device}"
        cache_path = os.path.join(self._cache_dir, cache_key)
        
        if os.path.exists(cache_path) and os.path.isdir(cache_path):
            xml_file = os.path.join(cache_path, "model.xml")
            bin_file = os.path.join(cache_path, "model.bin")
            
            if os.path.exists(xml_file) and os.path.exists(bin_file):
                logger.info(f"Found cached model: {cache_path}")
                return xml_file
                
        return None
        
    def _cache_model(self, model, model_name: str, precision: str, device: str) -> str:
        """
        Cache a model for future use.
        
        Args:
            model: OpenVINO model to cache
            model_name: Name of the model
            precision: Precision format (FP32, FP16, INT8)
            device: Target device
            
        Returns:
            Path to cached model
        """
        try:
            import openvino as ov
            
            cache_key = f"{model_name}_{precision}_{device}"
            cache_path = os.path.join(self._cache_dir, cache_key)
            
            # Create cache directory if it doesn't exist
            os.makedirs(cache_path, exist_ok=True)
            
            # Save model to cache
            xml_path = os.path.join(cache_path, "model.xml")
            ov.save_model(model, xml_path, {"compress_to_fp16": precision == "FP16"})
            
            logger.info(f"Cached model at: {cache_path}")
            return xml_path
        except Exception as e:
            logger.warning(f"Failed to cache model: {str(e)}")
            return None
    
    def load_model(self, model_name: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Load a model with OpenVINO.
        
        Args:
            model_name: Name of the model.
            config: Configuration options.
            
        Returns:
            Dictionary with load result.
        """
        if not self.is_available():
            logger.error("OpenVINO is not available")
            return {"status": "error", "message": "OpenVINO is not available"}
        
        # Get device from config or use default
        config = config or {}
        device = config.get("device", "CPU")
        
        if device not in self._available_devices:
            if "AUTO" in self._available_devices:
                device = "AUTO"
                logger.info(f"Requested device {device} not found, using AUTO instead")
            else:
                logger.error(f"Device {device} not found")
                return {"status": "error", "message": f"Device {device} not found"}
        
        model_key = f"{model_name}_{device}"
        if model_key in self.models:
            logger.info(f"Model {model_name} already loaded on OpenVINO {device}")
            return {
                "status": "success",
                "model_name": model_name,
                "device": device,
                "already_loaded": True
            }
        
        # Check if we should use optimum.intel integration
        use_optimum = config.get("use_optimum", True)  # Default to using optimum if available
        model_type = config.get("model_type", "unknown")
        
        # Check if this looks like a HuggingFace model
        is_hf_model = False
        if "/" in model_name or model_name in [
            "bert-base-uncased", "bert-large-uncased", "roberta-base", "t5-small", "t5-base",
            "gpt2", "gpt2-medium", "vit-base-patch16-224", "clip-vit-base-patch32"
        ]:
            is_hf_model = True
        
        # Try to use optimum.intel if this is a HuggingFace model
        if is_hf_model and use_optimum:
            # Check if optimum.intel is available
            optimum_info = self.get_optimum_integration()
            if optimum_info.get("available", False):
                logger.info(f"Using optimum.intel integration for HuggingFace model {model_name}")
                result = self.load_model_with_optimum(model_name, config)
                # If optimum loading succeeded, return the result
                if result.get("status") == "success":
                    logger.info(f"Successfully loaded model {model_name} with optimum.intel")
                    return result
                # Otherwise, log a warning and continue with standard loading
                else:
                    logger.warning(f"Failed to load with optimum.intel: {result.get('message')}")
                    logger.warning("Falling back to standard OpenVINO loading")

        try:
            import openvino as ov
            
            # Get model path from config
            model_path = config.get("model_path")
            if not model_path:
                logger.error("Model path not provided")
                return {"status": "error", "message": "Model path not provided"}
            
            # Check if model file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                return {"status": "error", "message": f"Model file not found at {model_path}"}
            
            # Get precision and other configuration options
            model_format = config.get("model_format", "IR")  # IR is OpenVINO's default format
            precision = config.get("precision", "FP32")
            
            # Check for mixed precision configuration
            mixed_precision = config.get("mixed_precision", False)
            
            # Check for multi-device configuration
            multi_device = config.get("multi_device", False)
            device_priorities = config.get("device_priorities", None)
            
            # Additional configuration for inference
            inference_config = {}
            
            # Set number of CPU threads if provided and device is CPU or contains CPU
            if "CPU" in device and "cpu_threads" in config:
                inference_config["CPU_THREADS_NUM"] = config["cpu_threads"]
            
            # Set up cache directory for compiled models
            cache_dir = config.get("cache_dir")
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                inference_config["CACHE_DIR"] = cache_dir
            
            # Enable or disable dynamic shapes
            dynamic_shapes = config.get("dynamic_shapes", True)
            if dynamic_shapes:
                inference_config["ENABLE_DYNAMIC_SHAPES"] = "YES"
            
            # Add performance hints if provided
            if "performance_hint" in config:
                inference_config["PERFORMANCE_HINT"] = config["performance_hint"]
                
            # Enable model caching if requested
            model_caching = config.get("model_caching", True)
            if model_caching:
                if "CACHE_DIR" not in inference_config:
                    cache_dir = os.path.join(self._cache_dir, "compiled_models")
                    os.makedirs(cache_dir, exist_ok=True)
                    inference_config["CACHE_DIR"] = cache_dir
                    
                # Set unique model name for caching
                cache_key = f"{model_name}_{precision}_{device}".replace("/", "_").replace(":", "_")
                inference_config["MODEL_CACHE_KEY"] = cache_key
                
            # Handle GPU-specific configurations
            if "GPU" in device:
                # Enable FP16 compute if not explicitly disabled
                inference_config["GPU_FP16_ENABLE"] = "YES" if config.get("gpu_fp16_enable", True) else "NO"
                
                # Set preferred GPU optimizations (modern is a good default for newer GPUs)
                if "gpu_optimize" in config:
                    inference_config["GPU_OPTIMIZE"] = config["gpu_optimize"]
                
            # Create a compiled model
            logger.info(f"Loading model {model_name} from {model_path} on OpenVINO {device}")
            
            # Set up device based on multi-device configuration
            target_device = device
            if multi_device:
                logger.info("Using multi-device configuration")
                if device_priorities:
                    # Format: "MULTI:<dev1>,<dev2>,..." or "MULTI:CPU(1.0),GPU.0(1.5)"
                    # Higher number means higher priority
                    target_device = f"MULTI:{','.join(device_priorities)}"
                    logger.info(f"Using device priorities: {target_device}")
                else:
                    # Infer best devices based on availability
                    available_priorities = []
                    
                    # Add available devices with reasonable priorities
                    if "GPU" in self._available_devices:
                        available_priorities.append("GPU(1.5)")  # GPU highest priority for compute
                    
                    if "CPU" in self._available_devices:
                        available_priorities.append("CPU(1.0)")  # CPU backup
                    
                    # Add other available devices with lower priority
                    for dev in self._available_devices:
                        if dev not in ["GPU", "CPU", "MULTI", "AUTO"] and dev not in available_priorities:
                            available_priorities.append(f"{dev}(0.8)")
                    
                    if available_priorities:
                        target_device = f"MULTI:{','.join(available_priorities)}"
                        logger.info(f"Inferred device configuration: {target_device}")
                    else:
                        logger.warning("No suitable devices found for multi-device, falling back to original device")
                        target_device = device
            
            # Load model using OpenVINO Runtime Core
            try:
                if model_format == "IR":
                    # Load IR model directly
                    ov_model = self._core.read_model(model_path)
                    
                    # Apply precision transformations
                    if mixed_precision:
                        # Apply mixed precision transformation
                        mixed_precision_config = config.get("mixed_precision_config", {})
                        ov_model = self._apply_mixed_precision_transformations(ov_model, mixed_precision_config)
                        logger.info("Applied mixed precision transformations to the model")
                        
                    elif precision == "FP16":
                        ov_model = self._apply_fp16_transformations(ov_model)
                        
                    elif precision == "INT8":
                        # For INT8, check if calibration data is provided
                        calibration_data = config.get("calibration_data")
                        
                        # If no calibration data but we have a loaded model, try to generate some
                        if calibration_data is None and model_key in self.models:
                            model_info = self.models[model_key]
                            calibration_data = self._generate_dummy_calibration_data(
                                model_info,
                                num_samples=config.get("calibration_samples", 10)
                            )
                            
                        # Apply INT8 transformations with calibration data (if available)
                        ov_model = self._apply_int8_transformations(ov_model, calibration_data)
                    
                    # Check if we should precompile for specific shapes
                    if config.get("precompile_shapes", False) and "input_shapes" in config:
                        input_shapes = config["input_shapes"]
                        logger.info(f"Precompiling model for specific shapes: {input_shapes}")
                        
                        # Set input shapes for precompilation
                        for input_name, shape in input_shapes.items():
                            if input_name in ov_model.inputs:
                                try:
                                    from openvino.runtime import PartialShape
                                    ov_model.reshape({input_name: PartialShape(shape)})
                                except Exception as e:
                                    logger.warning(f"Failed to reshape model for input {input_name}: {e}")
                    
                    # Compile model for target device
                    compiled_model = self._core.compile_model(ov_model, target_device, inference_config)
                    
                elif model_format == "ONNX":
                    # Load ONNX model directly
                    ov_model = self._core.read_model(model_path)
                    
                    # Apply precision transformations
                    if mixed_precision:
                        # Apply mixed precision transformation
                        mixed_precision_config = config.get("mixed_precision_config", {})
                        ov_model = self._apply_mixed_precision_transformations(ov_model, mixed_precision_config)
                        logger.info("Applied mixed precision transformations to the model")
                        
                    elif precision == "FP16":
                        ov_model = self._apply_fp16_transformations(ov_model)
                        
                    elif precision == "INT8":
                        # For INT8, check if calibration data is provided
                        calibration_data = config.get("calibration_data")
                        
                        # If no calibration data and model already loaded, try to generate some
                        if calibration_data is None and model_key in self.models:
                            model_info = self.models[model_key]
                            calibration_data = self._generate_dummy_calibration_data(
                                model_info,
                                num_samples=config.get("calibration_samples", 10)
                            )
                            
                        # Apply INT8 transformations with calibration data (if available)
                        ov_model = self._apply_int8_transformations(ov_model, calibration_data)
                    
                    # Check if we should precompile for specific shapes
                    if config.get("precompile_shapes", False) and "input_shapes" in config:
                        input_shapes = config["input_shapes"]
                        logger.info(f"Precompiling model for specific shapes: {input_shapes}")
                        
                        # Set input shapes for precompilation
                        for input_name, shape in input_shapes.items():
                            if input_name in ov_model.inputs:
                                try:
                                    from openvino.runtime import PartialShape
                                    ov_model.reshape({input_name: PartialShape(shape)})
                                except Exception as e:
                                    logger.warning(f"Failed to reshape model for input {input_name}: {e}")
                    
                    # Compile model for target device
                    compiled_model = self._core.compile_model(ov_model, target_device, inference_config)
                    
                else:
                    logger.error(f"Unsupported model format: {model_format}")
                    return {"status": "error", "message": f"Unsupported model format: {model_format}"}
                
                # Create infer request for model inference
                infer_request = compiled_model.create_infer_request()
                
                # Store model information and objects
                self.models[model_key] = {
                    "name": model_name,
                    "device": device,
                    "model_path": model_path,
                    "model_format": model_format,
                    "precision": precision,
                    "loaded": True,
                    "config": config,
                    "ov_model": ov_model,
                    "compiled_model": compiled_model,
                    "infer_request": infer_request,
                    "inputs_info": {input_name: input_port.get_shape() for input_name, input_port in ov_model.inputs.items()},
                    "outputs_info": {output_name: output_port.get_shape() for output_name, output_port in ov_model.outputs.items()},
                    "load_time": time.time()
                }
                
                logger.info(f"Model {model_name} successfully loaded on {device}")
                logger.debug(f"Model inputs: {self.models[model_key]['inputs_info']}")
                logger.debug(f"Model outputs: {self.models[model_key]['outputs_info']}")
                
                return {
                    "status": "success",
                    "model_name": model_name,
                    "device": device,
                    "model_format": model_format,
                    "precision": precision,
                    "inputs_info": self.models[model_key]['inputs_info'],
                    "outputs_info": self.models[model_key]['outputs_info']
                }
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                return {"status": "error", "message": f"Failed to load model: {str(e)}"}
                
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            return {"status": "error", "message": f"Failed to import required modules: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error during model loading: {str(e)}")
            return {"status": "error", "message": f"Unexpected error during model loading: {str(e)}"}
    
    def unload_model(self, model_name: str, device: str = "CPU") -> Dict[str, Any]:
        """
        Unload a model from OpenVINO.
        
        Args:
            model_name: Name of the model.
            device: Device name.
            
        Returns:
            Dictionary with unload result.
        """
        if not self.is_available():
            logger.error("OpenVINO is not available")
            return {"status": "error", "message": "OpenVINO is not available"}
        
        model_key = f"{model_name}_{device}"
        if model_key not in self.models:
            logger.warning(f"Model {model_name} not loaded on OpenVINO {device}")
            return {"status": "error", "message": f"Model {model_name} not loaded on OpenVINO {device}"}
        
        try:
            logger.info(f"Unloading model {model_name} from OpenVINO {device}")
            
            # Get model info
            model_info = self.models[model_key]
            
            # Delete all references to OpenVINO objects for garbage collection
            model_info.pop("ov_model", None)
            model_info.pop("compiled_model", None)
            model_info.pop("infer_request", None)
            
            # Remove model information
            del self.models[model_key]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            return {
                "status": "success",
                "model_name": model_name,
                "device": device
            }
        except Exception as e:
            logger.error(f"Error unloading model: {str(e)}")
            return {"status": "error", "message": f"Error unloading model: {str(e)}"}
    
    def run_inference(self, model_name: str, content: Any, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run inference with OpenVINO.
        
        Args:
            model_name: Name of the model.
            content: Input content for inference.
            config: Configuration options.
            
        Returns:
            Dictionary with inference result.
        """
        if not self.is_available():
            logger.error("OpenVINO is not available")
            return {"status": "error", "message": "OpenVINO is not available"}
        
        # Get device from config or use default
        config = config or {}
        device = config.get("device", "CPU")
        
        if device not in self._available_devices:
            if "AUTO" in self._available_devices:
                device = "AUTO"
                logger.info(f"Requested device {device} not found, using AUTO instead")
            else:
                logger.error(f"Device {device} not found")
                return {"status": "error", "message": f"Device {device} not found"}
        
        model_key = f"{model_name}_{device}"
        if model_key not in self.models:
            logger.warning(f"Model {model_name} not loaded on OpenVINO {device}, loading now")
            load_result = self.load_model(model_name, config)
            if load_result.get("status") != "success":
                return load_result
        
        # Get model info
        model_info = self.models[model_key]
        
        # Check if this is an optimum.intel model
        if model_info.get("optimum_integration", False):
            # Run inference with optimum.intel model
            return self._run_optimum_inference(model_name, content, config)
        
        try:
            import numpy as np
            
            infer_request = model_info.get("infer_request")
            
            if not infer_request:
                logger.error(f"Model {model_name} does not have a valid inference request")
                return {"status": "error", "message": "Invalid inference request"}
            
            # Get model inputs info
            inputs_info = model_info["inputs_info"]
            
            # Process input data based on content type
            try:
                # Measure start time for performance metrics
                start_time = time.time()
                
                # Memory before inference
                memory_before = self._get_memory_usage()
                
                # Prepare input data
                input_data = self._prepare_input_data(content, inputs_info, config)
                
                # Set input data for inference
                for input_name, input_tensor in input_data.items():
                    infer_request.set_input_tensor(input_name, input_tensor)
                
                # Start async inference
                infer_request.start_async()
                # Wait for inference to complete
                infer_request.wait()
                
                # Get inference results
                results = {}
                for output_name in model_info["outputs_info"].keys():
                    results[output_name] = infer_request.get_output_tensor(output_name).data
                
                # Measure end time
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000  # ms
                
                # Memory after inference
                memory_after = self._get_memory_usage()
                memory_usage = memory_after - memory_before
                
                # Post-process results if needed
                processed_results = self._postprocess_results(results, config.get("model_type", "unknown"))
                
                # Calculate performance metrics
                throughput = 1000 / inference_time  # items per second
                
                # Add model-type specific metrics
                if config.get("model_type") == "text":
                    batch_size = config.get("batch_size", 1)
                    seq_length = config.get("sequence_length", 128)
                    throughput = (batch_size * 1000) / inference_time
                    
                return {
                    "status": "success",
                    "model_name": model_name,
                    "device": device,
                    "latency_ms": inference_time,
                    "throughput_items_per_sec": throughput,
                    "memory_usage_mb": memory_usage,
                    "results": processed_results,
                    "execution_order": config.get("execution_order", 0),  # For batched execution
                    "batch_size": config.get("batch_size", 1)
                }
                
            except Exception as e:
                logger.error(f"Error during inference: {str(e)}")
                return {"status": "error", "message": f"Error during inference: {str(e)}"}
                
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            return {"status": "error", "message": f"Failed to import required modules: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error during inference: {str(e)}")
            return {"status": "error", "message": f"Unexpected error during inference: {str(e)}"}
            
    def _run_optimum_inference(self, model_name: str, content: Any, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run inference with an optimum.intel model.
        
        Args:
            model_name: Name of the model.
            content: Input content for inference (text, image, etc.).
            config: Configuration options.
            
        Returns:
            Dictionary with inference result.
        """
        # Get device from config or use default
        config = config or {}
        device = config.get("device", "CPU")
        model_key = f"{model_name}_{device}"
        
        # Get model info
        model_info = self.models[model_key]
        ov_model = model_info.get("ov_model")
        processor = model_info.get("processor")
        model_type = model_info.get("ov_model_type", "unknown")
        
        if not ov_model:
            logger.error("Optimum.intel model not found")
            return {"status": "error", "message": "Optimum.intel model not found"}
        
        try:
            import torch
            import numpy as np
            
            # Measure start time for performance metrics
            start_time = time.time()
            
            # Memory before inference
            memory_before = self._get_memory_usage()
            
            # Process inputs based on model type
            try:
                # Prepare inputs
                if model_type in ["masked", "sequence-classification", "token-classification", "causal", "seq2seq"]:
                    # Text models use a tokenizer
                    if processor is None:
                        from transformers import AutoTokenizer
                        processor = AutoTokenizer.from_pretrained(model_name)
                        self.models[model_key]["processor"] = processor  # Cache for future use
                    
                    # Process based on content type
                    if isinstance(content, dict) and "input_ids" in content:
                        # Content is already tokenized
                        inputs = content
                    elif isinstance(content, str):
                        # Text content needs tokenization
                        inputs = processor(content, return_tensors="pt")
                    else:
                        # Unknown content format
                        logger.error(f"Unsupported content format for text model: {type(content)}")
                        return {"status": "error", "message": f"Unsupported content format for text model: {type(content)}"}
                
                elif model_type in ["image-classification", "object-detection", "image-segmentation"]:
                    # Image models use an image processor
                    if processor is None:
                        from transformers import AutoImageProcessor
                        processor = AutoImageProcessor.from_pretrained(model_name)
                        self.models[model_key]["processor"] = processor  # Cache for future use
                    
                    # Process based on content type
                    if isinstance(content, dict) and "pixel_values" in content:
                        # Content is already processed
                        inputs = content
                    elif isinstance(content, np.ndarray):
                        # Raw image data
                        inputs = processor(content, return_tensors="pt")
                    else:
                        # Try to process as PIL image or path
                        try:
                            inputs = processor(content, return_tensors="pt")
                        except Exception as e:
                            logger.error(f"Failed to process image input: {str(e)}")
                            return {"status": "error", "message": f"Failed to process image input: {str(e)}"}
                
                elif model_type in ["audio-classification", "automatic-speech-recognition"]:
                    # Audio models use a feature extractor
                    if processor is None:
                        from transformers import AutoFeatureExtractor
                        processor = AutoFeatureExtractor.from_pretrained(model_name)
                        self.models[model_key]["processor"] = processor  # Cache for future use
                    
                    # Process based on content type
                    if isinstance(content, dict) and "input_features" in content:
                        # Content is already processed
                        inputs = content
                    elif isinstance(content, np.ndarray):
                        # Raw audio data
                        inputs = processor(content, return_tensors="pt")
                    else:
                        # Try to process as audio file path
                        try:
                            inputs = processor(content, return_tensors="pt")
                        except Exception as e:
                            logger.error(f"Failed to process audio input: {str(e)}")
                            return {"status": "error", "message": f"Failed to process audio input: {str(e)}"}
                
                else:
                    # For other model types, try a generic approach
                    if isinstance(content, dict):
                        # Assume the content is already in the right format
                        inputs = content
                    else:
                        logger.error(f"Unsupported model type for content: {model_type}")
                        return {"status": "error", "message": f"Unsupported model type for content: {model_type}"}
                
                # Run inference with optimum.intel model
                with torch.no_grad():
                    outputs = ov_model(**inputs)
                
                # Measure end time
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000  # ms
                
                # Memory after inference
                memory_after = self._get_memory_usage()
                memory_usage = memory_after - memory_before
                
                # Process outputs based on model type
                processed_outputs = {}
                
                # Extract relevant outputs based on model type
                if hasattr(outputs, "logits"):
                    processed_outputs["logits"] = outputs.logits.cpu().numpy()
                
                if hasattr(outputs, "last_hidden_state"):
                    processed_outputs["last_hidden_state"] = outputs.last_hidden_state.cpu().numpy()
                
                if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    processed_outputs["hidden_states"] = [hs.cpu().numpy() for hs in outputs.hidden_states]
                
                # Post-process results based on model type (custom for different model families)
                if model_type == "sequence-classification":
                    # Get predicted class
                    if "logits" in processed_outputs:
                        import numpy as np
                        logits = processed_outputs["logits"]
                        predictions = np.argmax(logits, axis=-1)
                        processed_outputs["predictions"] = predictions
                
                elif model_type == "token-classification":
                    # Get token predictions
                    if "logits" in processed_outputs:
                        import numpy as np
                        logits = processed_outputs["logits"]
                        predictions = np.argmax(logits, axis=-1)
                        processed_outputs["predictions"] = predictions
                
                elif model_type in ["causal", "seq2seq"]:
                    # For text generation, extract the generated IDs
                    if hasattr(outputs, "sequences"):
                        processed_outputs["sequences"] = outputs.sequences.cpu().numpy()
                        
                        # Try to decode the sequences if tokenizer is available
                        if processor is not None and hasattr(processor, "batch_decode"):
                            try:
                                decoded_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
                                processed_outputs["generated_text"] = decoded_text
                            except Exception as e:
                                logger.warning(f"Failed to decode sequences: {str(e)}")
                
                # Calculate performance metrics
                throughput = 1000 / inference_time  # items per second
                
                # Add model-type specific metrics
                if model_type in ["masked", "sequence-classification", "token-classification", "causal", "seq2seq"]:
                    batch_size = inputs.get("input_ids", []).shape[0] if "input_ids" in inputs else 1
                    seq_length = inputs.get("input_ids", []).shape[1] if "input_ids" in inputs else 0
                    throughput = (batch_size * 1000) / inference_time
                
                return {
                    "status": "success",
                    "model_name": model_name,
                    "device": device,
                    "model_type": model_type,
                    "latency_ms": inference_time,
                    "throughput_items_per_sec": throughput,
                    "memory_usage_mb": memory_usage,
                    "results": processed_outputs,
                    "execution_order": config.get("execution_order", 0),
                    "batch_size": config.get("batch_size", 1),
                    "optimum_integration": True
                }
            
            except Exception as e:
                logger.error(f"Error during optimum.intel inference: {str(e)}")
                return {"status": "error", "message": f"Error during optimum.intel inference: {str(e)}"}
        
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            return {"status": "error", "message": f"Failed to import required modules: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error during optimum.intel inference: {str(e)}")
            return {"status": "error", "message": f"Unexpected error during optimum.intel inference: {str(e)}"}
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
        except ImportError:
            logger.warning("psutil not installed, cannot measure memory usage")
            return 0.0
        except Exception as e:
            logger.warning(f"Error measuring memory usage: {str(e)}")
            return 0.0
    
    def _prepare_input_data(self, content: Any, inputs_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input data for the model.
        
        Args:
            content: Input content for inference
            inputs_info: Model input information
            config: Configuration options
            
        Returns:
            Dictionary mapping input names to prepared tensors
        """
        try:
            import numpy as np
            model_type = config.get("model_type", "unknown")
            
            # Handle different content types based on model type
            if isinstance(content, dict):
                # Content is already in the format {input_name: tensor}
                prepared_inputs = {}
                
                # Validate and prepare each input tensor
                for input_name, tensor in content.items():
                    if input_name in inputs_info:
                        # Convert to numpy array if needed
                        if not isinstance(tensor, np.ndarray):
                            if hasattr(tensor, "numpy"):  # PyTorch tensor or similar
                                tensor = tensor.numpy()
                            else:
                                tensor = np.array(tensor)
                        
                        # Reshape if needed
                        shape = inputs_info[input_name]
                        if tensor.shape != shape and len(shape) > 0 and -1 not in shape:
                            logger.warning(f"Reshaping input tensor {input_name} from {tensor.shape} to {shape}")
                            tensor = tensor.reshape(shape)
                        
                        prepared_inputs[input_name] = tensor
                    else:
                        logger.warning(f"Input name {input_name} not found in model inputs, skipping")
                
                return prepared_inputs
            elif isinstance(content, np.ndarray):
                # Single numpy array, use the first input
                if len(inputs_info) == 1:
                    input_name = list(inputs_info.keys())[0]
                    shape = inputs_info[input_name]
                    
                    # Reshape if needed
                    if content.shape != shape and len(shape) > 0 and -1 not in shape:
                        logger.warning(f"Reshaping input tensor from {content.shape} to {shape}")
                        content = content.reshape(shape)
                    
                    return {input_name: content}
                else:
                    logger.error(f"Cannot map single input to multiple model inputs: {list(inputs_info.keys())}")
                    raise ValueError("Cannot map single input to multiple model inputs")
            else:
                # Handle based on model type
                if model_type == "text":
                    return self._prepare_text_input(content, inputs_info, config)
                elif model_type == "vision":
                    return self._prepare_vision_input(content, inputs_info, config)
                elif model_type == "audio":
                    return self._prepare_audio_input(content, inputs_info, config)
                elif model_type == "multimodal":
                    return self._prepare_multimodal_input(content, inputs_info, config)
                else:
                    logger.error(f"Unsupported model type: {model_type}")
                    raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            logger.error(f"Error preparing input data: {str(e)}")
            raise
    
    def _prepare_text_input(self, content: Any, inputs_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare text input data for the model.
        
        Args:
            content: Text input content
            inputs_info: Model input information
            config: Configuration options
            
        Returns:
            Dictionary mapping input names to prepared tensors
        """
        try:
            import numpy as np
            
            # Basic handling for text models (simplified)
            # In a real implementation, this would use tokenizers and handle various text models
            
            # If content is already tokenized
            if isinstance(content, dict) and "input_ids" in content:
                prepared_inputs = {}
                
                for key, value in content.items():
                    if key in inputs_info:
                        if hasattr(value, "numpy"):
                            value = value.numpy()
                        elif not isinstance(value, np.ndarray):
                            value = np.array(value)
                        
                        prepared_inputs[key] = value
                
                return prepared_inputs
            
            # Default simple handling for raw text
            logger.warning("Using simplified text processing - for production use, integrate with proper tokenization")
            
            # Get the first input name
            input_name = list(inputs_info.keys())[0]
            
            # Create dummy input for demonstration
            # In real implementation, use proper tokenization
            shape = inputs_info[input_name]
            batch_size = shape[0] if shape[0] != -1 else 1
            seq_length = shape[1] if len(shape) > 1 and shape[1] != -1 else 128
            
            # Create dummy input ids (this should be replaced with actual tokenization)
            input_ids = np.zeros((batch_size, seq_length), dtype=np.int64)
            
            # For demo purposes only
            if isinstance(content, str):
                # Just a dummy conversion of characters to IDs (not realistic)
                # This should be replaced with proper tokenization
                for i, char in enumerate(content[:min(len(content), seq_length)]):
                    input_ids[0, i] = ord(char) % 30000
            
            attention_mask = np.ones((batch_size, seq_length), dtype=np.int64)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        except Exception as e:
            logger.error(f"Error preparing text input: {str(e)}")
            raise
    
    def _prepare_vision_input(self, content: Any, inputs_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare vision input data for the model.
        
        Args:
            content: Vision input content (image path, PIL image, numpy array)
            inputs_info: Model input information
            config: Configuration options
            
        Returns:
            Dictionary mapping input names to prepared tensors
        """
        try:
            import numpy as np
            
            # Get the first input name
            input_name = list(inputs_info.keys())[0]
            shape = inputs_info[input_name]
            
            # Determine expected input shape
            batch_size = shape[0] if shape[0] != -1 else 1
            channels = shape[1] if len(shape) > 3 else 3  # Default to 3 channels (RGB)
            height = shape[2] if len(shape) > 3 else 224  # Default height
            width = shape[3] if len(shape) > 3 else 224   # Default width
            
            # Handle PIL Image
            if hasattr(content, "convert") and callable(getattr(content, "convert")):
                # Convert PIL Image to numpy array
                content = content.convert("RGB")
                img_array = np.array(content)
                # Transpose from HWC to CHW format
                img_array = img_array.transpose((2, 0, 1))
                # Add batch dimension if needed
                if len(img_array.shape) == 3:
                    img_array = np.expand_dims(img_array, axis=0)
                
                # Normalize if needed
                if config.get("normalize", True):
                    img_array = img_array / 255.0
                    
                    # Apply ImageNet normalization if specified
                    if config.get("imagenet_norm", False):
                        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
                        std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
                        img_array = (img_array - mean) / std
                
                # Resize if needed
                if img_array.shape[2] != height or img_array.shape[3] != width:
                    logger.warning(f"Image shape mismatch: expected ({height}, {width}), got ({img_array.shape[2]}, {img_array.shape[3]})")
                    # For proper implementation, use a resize function here
                
                return {input_name: img_array}
                
            # Handle numpy array
            elif isinstance(content, np.ndarray):
                img_array = content
                
                # Handle different formats
                if len(img_array.shape) == 3:  # HWC format
                    # Convert HWC to CHW
                    img_array = img_array.transpose((2, 0, 1))
                    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                elif len(img_array.shape) == 4:  # BHWC or BCHW format
                    if img_array.shape[3] == 3 or img_array.shape[3] == 1:  # BHWC
                        img_array = img_array.transpose((0, 3, 1, 2))  # Convert to BCHW
                
                # Apply normalization if needed
                if config.get("normalize", True):
                    img_array = img_array / 255.0
                    
                    # Apply ImageNet normalization if specified
                    if config.get("imagenet_norm", False):
                        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
                        std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
                        img_array = (img_array - mean) / std
                
                return {input_name: img_array}
                
            # Handle file path
            elif isinstance(content, str) and os.path.exists(content):
                try:
                    from PIL import Image
                    image = Image.open(content).convert("RGB")
                    img_array = np.array(image)
                    # Transpose from HWC to CHW format
                    img_array = img_array.transpose((2, 0, 1))
                    # Add batch dimension
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Apply normalization if needed
                    if config.get("normalize", True):
                        img_array = img_array / 255.0
                        
                        # Apply ImageNet normalization if specified
                        if config.get("imagenet_norm", False):
                            mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
                            std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
                            img_array = (img_array - mean) / std
                    
                    return {input_name: img_array}
                except ImportError:
                    logger.error("PIL not installed, cannot load image from path")
                    raise
                except Exception as e:
                    logger.error(f"Error loading image from path: {str(e)}")
                    raise
            else:
                logger.error(f"Unsupported vision input type: {type(content)}")
                raise ValueError(f"Unsupported vision input type: {type(content)}")
        except Exception as e:
            logger.error(f"Error preparing vision input: {str(e)}")
            raise
    
    def _prepare_audio_input(self, content: Any, inputs_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare audio input data for the model.
        
        Args:
            content: Audio input content (file path, numpy array with audio samples, or dict with processed features)
            inputs_info: Model input information
            config: Configuration options
            
        Returns:
            Dictionary mapping input names to prepared tensors
        """
        try:
            import numpy as np
            
            # Get the first input name
            input_name = list(inputs_info.keys())[0]
            shape = inputs_info[input_name]
            
            # Handle different audio input formats
            if isinstance(content, dict):
                # Already processed features
                return self._prepare_processed_audio_features(content, inputs_info, config)
            elif isinstance(content, np.ndarray):
                # Raw audio samples (1D array)
                return self._prepare_raw_audio_samples(content, inputs_info, config)
            elif isinstance(content, str) and os.path.exists(content):
                # Audio file path
                return self._prepare_audio_from_file(content, inputs_info, config)
            else:
                logger.error(f"Unsupported audio content type: {type(content)}")
                raise ValueError(f"Unsupported audio content type: {type(content)}")
                
        except ImportError as e:
            logger.error(f"Required audio processing libraries not available: {e}")
            logger.warning("Falling back to dummy audio tensor")
            
            # Create dummy audio tensor as fallback
            dummy_audio = np.zeros(shape, dtype=np.float32)
            return {input_name: dummy_audio}
        except Exception as e:
            logger.error(f"Error preparing audio input: {e}")
            raise
            
    def _prepare_processed_audio_features(self, content: Dict[str, Any], inputs_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Process already extracted audio features."""
        import numpy as np
        
        prepared_inputs = {}
        
        # Process each input in the content dictionary
        for key, value in content.items():
            if key in inputs_info:
                # Convert to numpy if needed
                if not isinstance(value, np.ndarray):
                    if hasattr(value, "numpy"):  # PyTorch tensor or similar
                        value = value.numpy()
                    else:
                        value = np.array(value)
                
                # Reshape if needed to match expected shape
                expected_shape = inputs_info[key]
                if value.shape != expected_shape and not any(dim == -1 for dim in expected_shape):
                    logger.info(f"Reshaping audio input {key} from {value.shape} to {expected_shape}")
                    value = self._reshape_to_match(value, expected_shape)
                
                prepared_inputs[key] = value
            else:
                # Check if this is a renamed input (common with feature extractors)
                # Common mappings between HF and ONNX/OpenVINO models
                alternate_names = {
                    "input_features": ["input_values", "inputs", "audio_input"],
                    "attention_mask": ["mask", "input_mask"],
                }
                
                # Try to find matching input name
                matched = False
                for ov_name, alt_names in alternate_names.items():
                    if key in alt_names and ov_name in inputs_info:
                        # Found matching alternate name
                        if not isinstance(value, np.ndarray):
                            if hasattr(value, "numpy"):
                                value = value.numpy()
                            else:
                                value = np.array(value)
                                
                        # Reshape if needed
                        expected_shape = inputs_info[ov_name]
                        if value.shape != expected_shape and not any(dim == -1 for dim in expected_shape):
                            logger.info(f"Reshaping audio input {key} (as {ov_name}) from {value.shape} to {expected_shape}")
                            value = self._reshape_to_match(value, expected_shape)
                            
                        prepared_inputs[ov_name] = value
                        matched = True
                        break
                
                if not matched:
                    logger.warning(f"Input name {key} not found in model inputs, skipping")
        
        return prepared_inputs
        
    def _prepare_raw_audio_samples(self, samples: np.ndarray, inputs_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw audio samples into model features."""
        import numpy as np
        
        # Get model configuration
        sample_rate = config.get("sample_rate", 16000)  # Default to 16kHz
        feature_size = config.get("feature_size", 80)   # Default feature size
        feature_type = config.get("feature_type", "log_mel_spectrogram")
        normalize = config.get("normalize", True)       # Whether to normalize features
        
        # Get the first input name
        input_name = list(inputs_info.keys())[0]
        expected_shape = inputs_info[input_name]
        
        try:
            # Try to import librosa for feature extraction
            import librosa
            
            # Resample if needed
            if config.get("original_sample_rate") and config.get("original_sample_rate") != sample_rate:
                samples = librosa.resample(
                    samples, 
                    orig_sr=config.get("original_sample_rate"), 
                    target_sr=sample_rate
                )
            
            # Extract features based on feature_type
            if feature_type == "log_mel_spectrogram":
                # Extract log mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=samples,
                    sr=sample_rate,
                    n_mels=feature_size,
                    n_fft=config.get("n_fft", 1024),
                    hop_length=config.get("hop_length", 512)
                )
                
                # Convert to log scale
                log_mel = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Normalize if requested
                if normalize:
                    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
                
                # Reshape to match expected input shape
                features = self._reshape_to_match(log_mel, expected_shape)
                
            elif feature_type == "mfcc":
                # Extract MFCCs
                mfcc = librosa.feature.mfcc(
                    y=samples,
                    sr=sample_rate,
                    n_mfcc=feature_size
                )
                
                # Normalize if requested
                if normalize:
                    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
                
                # Reshape to match expected input shape
                features = self._reshape_to_match(mfcc, expected_shape)
                
            else:
                # For unknown feature types, use raw samples and try to reshape
                logger.warning(f"Unknown audio feature type: {feature_type}. Using raw samples.")
                features = self._reshape_to_match(samples, expected_shape)
            
            return {input_name: features}
            
        except ImportError:
            logger.warning("librosa not available for audio processing. Using raw samples.")
            
            # Try to use the raw samples directly, reshaping as needed
            try:
                features = self._reshape_to_match(samples, expected_shape)
                return {input_name: features}
            except Exception as e:
                logger.error(f"Failed to prepare raw audio samples: {e}")
                # Fall back to zeros
                dummy_audio = np.zeros(expected_shape, dtype=np.float32)
                return {input_name: dummy_audio}
        
    def _prepare_audio_from_file(self, file_path: str, inputs_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Load audio from file and process it for the model."""
        import numpy as np
        
        try:
            # Try to import libraries for audio loading and processing
            import librosa
            
            # Load audio file with librosa
            logger.info(f"Loading audio file: {file_path}")
            sample_rate = config.get("sample_rate", 16000)  # Default to 16kHz
            audio, orig_sr = librosa.load(file_path, sr=sample_rate, mono=True)
            
            # Use the raw samples processing function
            config["original_sample_rate"] = orig_sr
            return self._prepare_raw_audio_samples(audio, inputs_info, config)
            
        except ImportError:
            logger.warning("librosa not available for audio file loading")
            
            # Try alternative methods
            try:
                import scipy.io.wavfile
                
                # Try to load with scipy
                sr, audio = scipy.io.wavfile.read(file_path)
                
                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                # Convert to float32 and normalize if int type
                if audio.dtype.kind in 'iu':  # integer type
                    max_value = np.iinfo(audio.dtype).max
                    audio = audio.astype(np.float32) / max_value
                
                # Set original sample rate in config
                config["original_sample_rate"] = sr
                
                # Process with raw samples function
                return self._prepare_raw_audio_samples(audio, inputs_info, config)
                
            except (ImportError, Exception) as e:
                logger.error(f"Failed to load audio file: {e}")
                
                # Fall back to zeros
                input_name = list(inputs_info.keys())[0]
                expected_shape = inputs_info[input_name]
                dummy_audio = np.zeros(expected_shape, dtype=np.float32)
                return {input_name: dummy_audio}
                
    def _reshape_to_match(self, data: np.ndarray, target_shape: List[int]) -> np.ndarray:
        """Reshape data to match target shape, handling dynamic dimensions."""
        import numpy as np
        
        # If shapes already match, return as is
        if data.shape == tuple(target_shape):
            return data
            
        # Filter out dynamic dimensions (-1) from target shape
        static_dims = [(i, dim) for i, dim in enumerate(target_shape) if dim != -1]
        
        # Start with the original data shape
        new_shape = list(data.shape)
        
        # Expand dimensions if needed
        while len(new_shape) < len(target_shape):
            new_shape = [1] + new_shape
            
        # Set static dimensions to match target
        for i, dim in static_dims:
            if i < len(new_shape):
                new_shape[i] = dim
        
        # If first dim is batch and is 1 in target but not in data, add batch dim
        if target_shape[0] == 1 and (len(data.shape) < len(target_shape) or data.shape[0] != 1):
            data = np.expand_dims(data, axis=0)
            new_shape[0] = 1
            
        # Reshape data to new shape
        try:
            return data.reshape(new_shape)
        except ValueError:
            # If direct reshape fails, try more flexible approach
            logger.warning(f"Direct reshape from {data.shape} to {new_shape} failed. Trying flexible approach.")
            
            # For audio models, common shapes:
            # [batch, sequence] or [batch, channels, sequence] or [batch, feature, sequence]
            if len(target_shape) == 2:
                # Target is [batch, sequence]
                if len(data.shape) == 1:
                    # 1D array, add batch dimension
                    return np.expand_dims(data, axis=0)
                elif len(data.shape) == 2:
                    # Already 2D, ensure batch dim is correct
                    if target_shape[0] != -1 and data.shape[0] != target_shape[0]:
                        # Reshape to match batch size
                        return np.reshape(data, target_shape)
                    return data
            elif len(target_shape) == 3:
                # Target is [batch, channels/features, sequence]
                if len(data.shape) == 2:
                    # 2D array [features, time] - add batch dimension
                    return np.expand_dims(data, axis=0)
                elif len(data.shape) == 3:
                    # Already 3D, check dimensions
                    return data.reshape(target_shape)
            
            # Last resort: try to flatten and then reshape
            try:
                flattened = data.flatten()
                target_size = np.prod([dim for dim in target_shape if dim > 0])
                
                # Pad or truncate to match size
                if len(flattened) < target_size:
                    padded = np.zeros(target_size, dtype=data.dtype)
                    padded[:len(flattened)] = flattened
                    flattened = padded
                elif len(flattened) > target_size:
                    flattened = flattened[:target_size]
                    
                # Now reshape to target_shape
                return flattened.reshape(target_shape)
            except Exception as e:
                logger.error(f"Failed to reshape data: {e}")
                # Return original data
                return data
    
    def _prepare_multimodal_input(self, content: Any, inputs_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare multimodal input data for the model.
        
        Args:
            content: Multimodal input content
            inputs_info: Model input information
            config: Configuration options
            
        Returns:
            Dictionary mapping input names to prepared tensors
        """
        # This is a placeholder - in a real implementation this would process multimodal data
        logger.warning("Multimodal input processing is not fully implemented")
        
        try:
            import numpy as np
            
            # Check if content is a dictionary with separate parts
            if isinstance(content, dict):
                prepared_inputs = {}
                
                # Handle text parts
                if "text" in content:
                    text_inputs = self._prepare_text_input(content["text"], inputs_info, config)
                    prepared_inputs.update(text_inputs)
                
                # Handle image parts
                if "image" in content:
                    image_inputs = self._prepare_vision_input(content["image"], inputs_info, config)
                    prepared_inputs.update(image_inputs)
                
                # Handle audio parts
                if "audio" in content:
                    audio_inputs = self._prepare_audio_input(content["audio"], inputs_info, config)
                    prepared_inputs.update(audio_inputs)
                
                return prepared_inputs
            else:
                logger.error("Multimodal input must be a dictionary with separate modalities")
                raise ValueError("Multimodal input must be a dictionary with separate modalities")
        except Exception as e:
            logger.error(f"Error preparing multimodal input: {str(e)}")
            raise
    
    def _postprocess_results(self, results: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """
        Post-process model output.
        
        Args:
            results: Raw model output
            model_type: Type of model
            
        Returns:
            Post-processed results
        """
        try:
            # Default is to return raw results
            processed_results = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in results.items()}
            
            # Model-specific post-processing
            if model_type == "text":
                # Text-specific processing would go here (e.g., decoding output IDs)
                pass
            elif model_type == "vision":
                # Vision-specific processing would go here (e.g., applying softmax)
                pass
            elif model_type == "audio":
                # Audio-specific processing would go here
                pass
            elif model_type == "multimodal":
                # Multimodal-specific processing would go here
                pass
            
            return processed_results
        except Exception as e:
            logger.error(f"Error post-processing results: {str(e)}")
            return results  # Return raw results in case of error
    
    def get_optimum_integration(self) -> Dict[str, Any]:
        """
        Check for optimum.intel integration for HuggingFace models.
        
        Returns:
            Dictionary with optimum integration status.
        """
        result = {
            "available": False,
            "version": None,
            "supported_models": []
        }
        
        try:
            # Try to import optimum.intel
            optimum_intel_spec = importlib.util.find_spec("optimum.intel")
            if optimum_intel_spec is not None:
                # optimum.intel is available
                result["available"] = True
                
                # Try to get version
                try:
                    import optimum.intel
                    result["version"] = optimum.intel.__version__
                except (ImportError, AttributeError):
                    pass
                
                # Check for specific optimum.intel functionality and model types
                model_types = [
                    ("SequenceClassification", "OVModelForSequenceClassification"),
                    ("TokenClassification", "OVModelForTokenClassification"),
                    ("QuestionAnswering", "OVModelForQuestionAnswering"),
                    ("CausalLM", "OVModelForCausalLM"),
                    ("Seq2SeqLM", "OVModelForSeq2SeqLM"),
                    ("MaskedLM", "OVModelForMaskedLM"),
                    ("Vision", "OVModelForImageClassification"),
                    ("FeatureExtraction", "OVModelForFeatureExtraction"),
                    ("ImageSegmentation", "OVModelForImageSegmentation"),
                    ("AudioClassification", "OVModelForAudioClassification"),
                    ("SpeechSeq2Seq", "OVModelForSpeechSeq2Seq"),
                    ("MultipleChoice", "OVModelForMultipleChoice")
                ]
                
                for model_type, class_name in model_types:
                    try:
                        # Dynamically import the class
                        model_class = getattr(
                            __import__("optimum.intel", fromlist=[class_name]), 
                            class_name
                        )
                        
                        # Store model type and class info
                        model_info = {
                            "type": model_type,
                            "class_name": class_name,
                            "available": True
                        }
                        
                        # Add to supported models
                        result["supported_models"].append(model_info)
                        
                        # Also set the legacy field
                        legacy_key = f"{model_type.lower()}_available"
                        result[legacy_key] = True
                        
                    except (ImportError, AttributeError) as e:
                        # Model type not available
                        legacy_key = f"{model_type.lower()}_available"
                        result[legacy_key] = False
                
                # Check for additional features
                try:
                    from optimum.intel import OVQuantizer
                    result["quantization_support"] = True
                except ImportError:
                    result["quantization_support"] = False
                
                try:
                    from optimum.intel import OVTrainingArguments
                    result["training_support"] = True
                except ImportError:
                    result["training_support"] = False
                    
                # Get supported OpenVINO version
                try:
                    from optimum.intel.utils.import_utils import check_if_transformers_greater
                    result["requires_transformers"] = check_if_transformers_greater()
                except ImportError:
                    pass
                    
                # Check for config options
                try:
                    from optimum.intel import OVConfig
                    result["config_support"] = True
                    
                    # Get default config
                    default_config = OVConfig.from_dict({})
                    result["default_config"] = {
                        "compression": default_config.compression if hasattr(default_config, "compression") else None,
                        "optimization_level": default_config.optimization_level if hasattr(default_config, "optimization_level") else None
                    }
                except ImportError:
                    result["config_support"] = False
                
        except ImportError:
            pass
        
        return result
        
    def load_model_with_optimum(self, model_name: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Load a model using optimum.intel integration.
        
        This method provides enhanced integration with optimum.intel for HuggingFace models,
        providing better performance and compatibility than the standard approach.
        
        Args:
            model_name: Name of the model to load.
            config: Configuration options.
            
        Returns:
            Dictionary with load result.
        """
        if not self.is_available():
            logger.error("OpenVINO is not available")
            return {"status": "error", "message": "OpenVINO is not available"}
            
        # Check if optimum.intel is available
        optimum_info = self.get_optimum_integration()
        if not optimum_info.get("available", False):
            logger.error("optimum.intel is not available")
            return {"status": "error", "message": "optimum.intel is not available"}
            
        config = config or {}
        device = config.get("device", "CPU")
        precision = config.get("precision", "FP32")
        model_type = config.get("model_type", "text")
        
        try:
            import optimum.intel
            from transformers import AutoConfig
            
            # Get model configuration to determine model type
            logger.info(f"Loading model configuration for {model_name}")
            model_config = AutoConfig.from_pretrained(model_name)
            model_config_dict = model_config.to_dict()
            
            # Find appropriate OV model class based on model type
            model_class = None
            ov_model_type = None
            
            # Try to determine model task from config
            task_mapping = {
                "seq2seq": "OVModelForSeq2SeqLM",
                "causal": "OVModelForCausalLM",
                "masked": "OVModelForMaskedLM",
                "sequence-classification": "OVModelForSequenceClassification",
                "token-classification": "OVModelForTokenClassification",
                "question-answering": "OVModelForQuestionAnswering",
                "image-classification": "OVModelForImageClassification",
                "audio-classification": "OVModelForAudioClassification",
                "feature-extraction": "OVModelForFeatureExtraction"
            }
            
            # Try to determine task from config
            task = None
            
            # Check config keys
            if "architectures" in model_config_dict:
                arch = model_config_dict["architectures"][0] if model_config_dict["architectures"] else None
                
                if arch:
                    arch_lower = arch.lower()
                    
                    if "seq2seq" in arch_lower or "t5" in arch_lower:
                        task = "seq2seq"
                    elif "causal" in arch_lower or "gpt" in arch_lower or "llama" in arch_lower:
                        task = "causal"
                    elif "masked" in arch_lower or "bert" in arch_lower:
                        task = "masked"
                    elif "classification" in arch_lower:
                        if "token" in arch_lower:
                            task = "token-classification"
                        else:
                            task = "sequence-classification"
                    elif "questionanswering" in arch_lower:
                        task = "question-answering"
                    elif "vision" in arch_lower or "vit" in arch_lower:
                        task = "image-classification"
                    elif "audio" in arch_lower or "wav2vec" in arch_lower:
                        task = "audio-classification"
            
            # If task not determined from architecture, try to infer from model type
            if not task:
                model_name_lower = model_name.lower()
                
                if "t5" in model_name_lower:
                    task = "seq2seq"
                elif "gpt" in model_name_lower or "llama" in model_name_lower:
                    task = "causal"
                elif "bert" in model_name_lower:
                    task = "masked"
                elif "vit" in model_name_lower:
                    task = "image-classification"
                elif "wav2vec" in model_name_lower:
                    task = "audio-classification"
                elif model_type == "text":
                    task = "masked"  # Default for text
                elif model_type == "vision":
                    task = "image-classification"  # Default for vision
                elif model_type == "audio":
                    task = "audio-classification"  # Default for audio
            
            # Get model class based on task
            if task and task in task_mapping:
                class_name = task_mapping[task]
                
                try:
                    model_class = getattr(optimum.intel, class_name)
                    ov_model_type = task
                    logger.info(f"Using {class_name} for model {model_name}")
                except (AttributeError, ImportError):
                    logger.warning(f"{class_name} not found in optimum.intel")
            
            # If no task identified or class not found, try available models from optimum info
            if not model_class:
                for model_info in optimum_info.get("supported_models", []):
                    if model_info.get("available"):
                        try:
                            model_class = getattr(optimum.intel, model_info["class_name"])
                            ov_model_type = model_info["type"]
                            logger.info(f"Using {model_info['class_name']} as fallback for model {model_name}")
                            break
                        except (AttributeError, ImportError):
                            continue
            
            # If no model class found, return error
            if not model_class:
                logger.error(f"No suitable optimum.intel model class found for {model_name}")
                return {"status": "error", "message": f"No suitable optimum.intel model class found for {model_name}"}
            
            # Create OpenVINO config
            ov_config = {}
            
            # Set device
            ov_config["device"] = device
            
            # Handle precision
            if precision == "FP16":
                ov_config["enable_fp16"] = True
            elif precision == "INT8":
                ov_config["enable_int8"] = True
            
            try:
                # Try to import OVConfig for advanced configuration
                from optimum.intel import OVConfig
                
                # Create optimum.intel config
                optimum_config = OVConfig(
                    compression=config.get("compression", None),
                    optimization_level=config.get("optimization_level", None),
                    enable_int8=True if precision == "INT8" else False,
                    enable_fp16=True if precision == "FP16" else False,
                    device=device
                )
                
                logger.info(f"Loading model {model_name} with optimum.intel")
                
                # Load model with optimum.intel
                ov_model = model_class.from_pretrained(
                    model_name,
                    ov_config=optimum_config,
                    export=True,  # Export to OpenVINO IR format
                    trust_remote_code=config.get("trust_remote_code", True)
                )
                
            except (ImportError, AttributeError):
                # Fallback for older versions or when OVConfig is not available
                logger.info(f"Loading model {model_name} with optimum.intel (legacy mode)")
                
                load_kwargs = {
                    "from_transformers": True,
                    "use_io_binding": True,
                    "trust_remote_code": config.get("trust_remote_code", True)
                }
                
                # Add precision settings
                if precision == "INT8":
                    load_kwargs["load_in_8bit"] = True
                
                ov_model = model_class.from_pretrained(
                    model_name,
                    **load_kwargs
                )
            
            # Store model in registry
            model_key = f"{model_name}_{device}"
            
            # Get processor/tokenizer
            try:
                if "vision" in ov_model_type.lower():
                    from transformers import AutoImageProcessor
                    processor = AutoImageProcessor.from_pretrained(model_name)
                elif "audio" in ov_model_type.lower():
                    from transformers import AutoFeatureExtractor
                    processor = AutoFeatureExtractor.from_pretrained(model_name)
                else:
                    from transformers import AutoTokenizer
                    processor = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                logger.warning(f"Failed to load processor/tokenizer: {e}")
                processor = None
            
            # Store model
            self.models[model_key] = {
                "name": model_name,
                "device": device,
                "model_path": model_name,
                "model_format": "optimum.intel",
                "precision": precision,
                "loaded": True,
                "config": config,
                "ov_model": ov_model,
                "processor": processor,
                "ov_model_type": ov_model_type,
                "optimum_integration": True,
                "load_time": time.time()
            }
            
            logger.info(f"Model {model_name} successfully loaded with optimum.intel on {device}")
            
            return {
                "status": "success",
                "model_name": model_name,
                "device": device,
                "model_format": "optimum.intel",
                "precision": precision,
                "ov_model_type": ov_model_type
            }
            
        except Exception as e:
            logger.error(f"Failed to load model with optimum.intel: {str(e)}")
            return {"status": "error", "message": f"Failed to load model with optimum.intel: {str(e)}"}
    
    def run_huggingface_inference(self, model_name_or_path: str, inputs: Any, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run inference with a HuggingFace model loaded with optimum.intel.
        
        Args:
            model_name_or_path: HuggingFace model name or path
            inputs: Input data for the model (text, tokenized inputs, etc.)
            config: Additional configuration options
            
        Returns:
            Dictionary with inference result
        """
        if not self.is_available():
            logger.error("OpenVINO is not available")
            return {"status": "error", "message": "OpenVINO is not available"}
        
        config = config or {}
        device = config.get("device", "CPU")
        
        # Check if model is loaded
        model_key = f"{model_name_or_path}_{device}"
        if model_key not in self.models:
            logger.warning(f"Model {model_name_or_path} not loaded on OpenVINO {device}, loading now")
            
            # Need model_type for loading
            model_type = config.get("model_type")
            if not model_type:
                logger.error("model_type is required for loading HuggingFace model")
                return {"status": "error", "message": "model_type is required for loading HuggingFace model"}
                
            load_result = self.load_huggingface_model(model_name_or_path, model_type, device, config)
            if load_result.get("status") != "success":
                return load_result
        
        model_info = self.models[model_key]
        model = model_info.get("model")
        tokenizer = model_info.get("tokenizer")
        model_type = model_info.get("model_type")
        
        if not model:
            logger.error(f"Model {model_name_or_path} is loaded but model object is missing")
            return {"status": "error", "message": "Model object is missing"}
        
        try:
            import torch
            import numpy as np
            
            # Measure start time for performance metrics
            start_time = time.time()
            
            # Memory before inference
            memory_before = self._get_memory_usage()
            
            # Process input based on model type and input format
            if model_type in ["sequence_classification", "token_classification", "question_answering"]:
                if isinstance(inputs, str):
                    # Simple text input
                    model_inputs = tokenizer(inputs, return_tensors="pt")
                elif isinstance(inputs, list) and all(isinstance(s, str) for s in inputs):
                    # Batch of text inputs
                    model_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
                elif isinstance(inputs, dict):
                    # Already tokenized inputs
                    model_inputs = inputs
                else:
                    logger.error(f"Unsupported input format for model type {model_type}")
                    return {"status": "error", "message": f"Unsupported input format for model type {model_type}"}
                
                # Run inference
                with torch.no_grad():
                    outputs = model(**model_inputs)
                
            elif model_type in ["causal_lm", "seq2seq_lm"]:
                # Generation parameters
                max_length = config.get("max_length", 50)
                min_length = config.get("min_length", 0)
                num_beams = config.get("num_beams", 1)
                temperature = config.get("temperature", 1.0)
                top_k = config.get("top_k", 50)
                top_p = config.get("top_p", 1.0)
                
                if isinstance(inputs, str):
                    # Simple text input
                    model_inputs = tokenizer(inputs, return_tensors="pt")
                elif isinstance(inputs, list) and all(isinstance(s, str) for s in inputs):
                    # Batch of text inputs
                    model_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
                elif isinstance(inputs, dict):
                    # Already tokenized inputs
                    model_inputs = inputs
                else:
                    logger.error(f"Unsupported input format for model type {model_type}")
                    return {"status": "error", "message": f"Unsupported input format for model type {model_type}"}
                
                # Run generation
                generate_kwargs = {
                    "max_length": max_length,
                    "min_length": min_length,
                    "num_beams": num_beams,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p
                }
                
                # Add specific generation parameters from config
                for key, value in config.items():
                    if key.startswith("generation_"):
                        param_name = key.replace("generation_", "")
                        generate_kwargs[param_name] = value
                
                # Run inference
                with torch.no_grad():
                    outputs = model.generate(**model_inputs, **generate_kwargs)
                
                # Process outputs
                if model_type == "seq2seq_lm":
                    # For Seq2Seq models, we need to decode the outputs
                    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    outputs = {"generated_text": decoded_outputs}
                else:
                    # For CausalLM models, we need to decode the outputs
                    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    outputs = {"generated_text": decoded_outputs}
                
            elif model_type == "vision":
                # Process image input
                # This would need proper image preprocessing based on the model
                if hasattr(inputs, "pixel_values"):
                    # Already preprocessed inputs
                    model_inputs = inputs
                else:
                    logger.error("Vision models require preprocessed inputs with pixel_values")
                    return {"status": "error", "message": "Vision models require preprocessed inputs with pixel_values"}
                
                # Run inference
                with torch.no_grad():
                    outputs = model(**model_inputs)
                
            elif model_type == "feature_extraction":
                if isinstance(inputs, str):
                    # Simple text input
                    model_inputs = tokenizer(inputs, return_tensors="pt")
                elif isinstance(inputs, list) and all(isinstance(s, str) for s in inputs):
                    # Batch of text inputs
                    model_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
                elif isinstance(inputs, dict):
                    # Already tokenized inputs
                    model_inputs = inputs
                else:
                    logger.error(f"Unsupported input format for model type {model_type}")
                    return {"status": "error", "message": f"Unsupported input format for model type {model_type}"}
                
                # Run inference
                with torch.no_grad():
                    outputs = model(**model_inputs)
            else:
                logger.error(f"Unsupported model type for inference: {model_type}")
                return {"status": "error", "message": f"Unsupported model type for inference: {model_type}"}
            
            # Measure end time
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # ms
            
            # Memory after inference
            memory_after = self._get_memory_usage()
            memory_usage = memory_after - memory_before
            
            # Process outputs to native Python types for JSON serialization
            processed_outputs = {}
            for key, value in outputs.items():
                if hasattr(value, "numpy"):
                    processed_outputs[key] = value.numpy().tolist()
                elif isinstance(value, torch.Tensor):
                    processed_outputs[key] = value.detach().cpu().numpy().tolist()
                elif isinstance(value, np.ndarray):
                    processed_outputs[key] = value.tolist()
                else:
                    processed_outputs[key] = value
            
            # Calculate performance metrics
            throughput = 1000 / inference_time  # items per second based on batch size 1
            batch_size = config.get("batch_size", 1)  # Default to 1 if not specified
            if batch_size > 1:
                throughput = (batch_size * 1000) / inference_time
            
            return {
                "status": "success",
                "model_name": model_name_or_path,
                "device": device,
                "model_type": model_type,
                "latency_ms": inference_time,
                "throughput_items_per_sec": throughput,
                "memory_usage_mb": memory_usage,
                "outputs": processed_outputs,
                "batch_size": batch_size
            }
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            return {"status": "error", "message": f"Failed to import required modules: {str(e)}"}
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            return {"status": "error", "message": f"Error during inference: {str(e)}"}
    
    def load_huggingface_model(self, model_name_or_path: str, model_type: str, device: str = "CPU", config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Load a HuggingFace model with optimum.intel integration.
        
        Args:
            model_name_or_path: HuggingFace model name or path
            model_type: Type of model (sequence_classification, causal_lm, seq2seq_lm, etc.)
            device: OpenVINO device to use
            config: Additional configuration options
            
        Returns:
            Dictionary with load result
        """
        if not self.is_available():
            logger.error("OpenVINO is not available")
            return {"status": "error", "message": "OpenVINO is not available"}
        
        # Check optimum.intel integration
        optimum_integration = self.get_optimum_integration()
        if not optimum_integration.get("available", False):
            logger.error("optimum.intel integration is not available")
            return {"status": "error", "message": "optimum.intel integration is not available"}
        
        config = config or {}
        
        # Validate device
        if device not in self._available_devices:
            if "AUTO" in self._available_devices:
                device = "AUTO"
                logger.info(f"Requested device {device} not found, using AUTO instead")
            else:
                logger.error(f"Device {device} not found")
                return {"status": "error", "message": f"Device {device} not found"}
        
        # Check if model is already loaded
        model_key = f"{model_name_or_path}_{device}"
        if model_key in self.models:
            logger.info(f"Model {model_name_or_path} already loaded on OpenVINO {device}")
            return {
                "status": "success",
                "model_name": model_name_or_path,
                "device": device,
                "already_loaded": True
            }
        
        # Load model with optimum.intel
        logger.info(f"Using optimum.intel integration for HuggingFace model {model_name_or_path}")
        
        try:
            import torch
            import numpy as np
            import transformers
            from transformers import AutoTokenizer, AutoConfig

            # Load model configuration
            logger.info(f"Loading model configuration for {model_name_or_path}")
            model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=config.get("trust_remote_code", False))
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=config.get("trust_remote_code", False))
            
            # Map model type to optimum.intel model class
            model_class_map = {}
            
            # Try to import all available optimum model classes
            try:
                from optimum.intel import OVModelForSequenceClassification
                model_class_map["sequence_classification"] = OVModelForSequenceClassification
            except ImportError:
                pass
                
            try:
                from optimum.intel import OVModelForTokenClassification
                model_class_map["token_classification"] = OVModelForTokenClassification
            except ImportError:
                pass
                
            try:
                from optimum.intel import OVModelForQuestionAnswering
                model_class_map["question_answering"] = OVModelForQuestionAnswering
            except ImportError:
                pass
                
            try:
                from optimum.intel import OVModelForCausalLM
                model_class_map["causal_lm"] = OVModelForCausalLM
            except ImportError:
                pass
                
            try:
                from optimum.intel import OVModelForSeq2SeqLM
                model_class_map["seq2seq_lm"] = OVModelForSeq2SeqLM
            except ImportError:
                pass
                
            try:
                from optimum.intel import OVModelForMaskedLM
                model_class_map["masked_lm"] = OVModelForMaskedLM
            except ImportError:
                pass
                
            try:
                from optimum.intel import OVModelForImageClassification
                model_class_map["vision"] = OVModelForImageClassification
            except ImportError:
                pass
                
            try:
                from optimum.intel import OVModelForFeatureExtraction
                model_class_map["feature_extraction"] = OVModelForFeatureExtraction
            except ImportError:
                pass
            
            # Check if model type is supported
            if model_type not in model_class_map:
                # Try to infer model type from config
                if hasattr(model_config, "architectures") and model_config.architectures:
                    arch = model_config.architectures[0]
                    if "MaskedLM" in arch:
                        inferred_type = "masked_lm"
                    elif "CausalLM" in arch:
                        inferred_type = "causal_lm"
                    elif "Seq2SeqLM" in arch:
                        inferred_type = "seq2seq_lm"
                    elif "SequenceClassification" in arch:
                        inferred_type = "sequence_classification"
                    elif "TokenClassification" in arch:
                        inferred_type = "token_classification"
                    elif "QuestionAnswering" in arch:
                        inferred_type = "question_answering"
                    elif "ImageClassification" in arch:
                        inferred_type = "vision"
                    else:
                        inferred_type = "feature_extraction"
                        
                    if inferred_type in model_class_map:
                        logger.info(f"Inferred model type {inferred_type} from architecture {arch}")
                        model_type = inferred_type
                
                # If still not supported, try to map to a similar supported type
                if model_type not in model_class_map:
                    if "bert" in model_name_or_path.lower():
                        if "masked_lm" in model_class_map:
                            logger.info(f"Using masked_lm for BERT model {model_name_or_path}")
                            model_type = "masked_lm"
                        elif "feature_extraction" in model_class_map:
                            logger.info(f"Using feature_extraction for BERT model {model_name_or_path}")
                            model_type = "feature_extraction"
            
            # Check if model type is supported now
            if model_type not in model_class_map:
                logger.error(f"Unsupported model type: {model_type}")
                logger.error(f"Supported types: {list(model_class_map.keys())}")
                return {"status": "error", "message": f"Unsupported model type: {model_type}"}
            
            # Get the appropriate model class
            model_class = model_class_map[model_type]
            logger.info(f"Using {model_class.__name__} for model {model_name_or_path}")
            
            # Set loading parameters
            load_kwargs = {
                "device": device,
                "trust_remote_code": config.get("trust_remote_code", False)
            }
            
            # Add precision if specified
            precision = config.get("precision")
            if precision:
                if precision == "FP16":
                    load_kwargs["load_in_8bit"] = False
                    load_kwargs["load_in_4bit"] = False
                    # Some specific FP16 handling based on optimum.intel version
                elif precision == "INT8":
                    load_kwargs["load_in_8bit"] = True
                    load_kwargs["load_in_4bit"] = False
                elif precision == "INT4":
                    load_kwargs["load_in_8bit"] = False
                    load_kwargs["load_in_4bit"] = True
            
            # Try to load model with optimum.intel
            logger.info(f"Loading model {model_name_or_path} with optimum.intel")
            try:
                start_time = time.time()
                model = model_class.from_pretrained(
                    model_name_or_path,
                    **load_kwargs
                )
                load_time = time.time() - start_time
            except Exception as e:
                logger.error(f"Failed to load model with optimum.intel: {str(e)}")
                logger.warning(f"Failed to load with optimum.intel: {str(e)}")
                logger.warning("Falling back to standard OpenVINO loading")
                
                # If optimum.intel fails, we need to go through the PyTorch->ONNX->OpenVINO path
                # Since we have already loaded the PyTorch model and tokenizer, we can try to export
                # it to ONNX and then convert to OpenVINO IR format
                try:
                    logger.info(f"Converting {model_name_or_path} to ONNX and then to OpenVINO IR format")
                    
                    # Import PyTorch and transformers for direct loading
                    from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSequenceClassification
                    
                    # Map model type to transformers model class
                    transformers_model_map = {
                        "masked_lm": AutoModelForMaskedLM,
                        "causal_lm": AutoModelForCausalLM,
                        "sequence_classification": AutoModelForSequenceClassification,
                        # Add more mappings as needed
                    }
                    
                    # Use appropriate model class or default to the most likely one
                    if model_type in transformers_model_map:
                        pt_model_class = transformers_model_map[model_type]
                    elif "bert" in model_name_or_path.lower():
                        pt_model_class = AutoModelForMaskedLM
                    else:
                        pt_model_class = AutoModelForMaskedLM  # Default fallback
                    
                    # Load PyTorch model
                    pt_model = pt_model_class.from_pretrained(model_name_or_path, trust_remote_code=config.get("trust_remote_code", False))
                    
                    # Create a sample input for the model
                    sample_text = "This is a sample input for ONNX conversion."
                    encoded_input = tokenizer(sample_text, return_tensors="pt")
                    
                    # Create output directory for ONNX and OpenVINO IR files
                    import tempfile
                    tmp_dir = tempfile.mkdtemp(prefix="openvino_conversion_")
                    onnx_path = os.path.join(tmp_dir, f"{model_name_or_path.replace('/', '_')}.onnx")
                    ir_path = os.path.join(tmp_dir, f"{model_name_or_path.replace('/', '_')}.xml")
                    
                    # Export to ONNX
                    logger.info(f"Exporting {model_name_or_path} to ONNX format at {onnx_path}")
                    torch.onnx.export(
                        pt_model,
                        tuple(encoded_input.values()),
                        onnx_path,
                        input_names=list(encoded_input.keys()),
                        output_names=["output"],
                        dynamic_axes={name: {0: "batch_size"} for name in encoded_input.keys()},
                        opset_version=12
                    )
                    
                    # Convert ONNX to OpenVINO IR
                    logger.info(f"Converting ONNX model to OpenVINO IR format at {ir_path}")
                    conversion_result = self.convert_from_onnx(
                        onnx_path,
                        ir_path,
                        {
                            "precision": precision or "FP32",
                            "input_shapes": {name: list(tensor.shape) for name, tensor in encoded_input.items()}
                        }
                    )
                    
                    if conversion_result.get("status") != "success":
                        logger.error(f"Failed to convert ONNX model to OpenVINO IR: {conversion_result.get('message', 'Unknown error')}")
                        return {"status": "error", "message": f"Failed to convert ONNX model to OpenVINO IR: {conversion_result.get('message', 'Unknown error')}"}
                    
                    # Now load the converted model
                    return self.load_model(
                        ir_path,
                        {
                            "device": device,
                            "model_format": "IR",
                            "model_type": model_type,
                            "precision": precision or "FP32",
                            "original_model": model_name_or_path
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed in fallback conversion path: {str(e)}")
                    return {"status": "error", "message": f"Failed in fallback conversion path: {str(e)}"}
            
            # Store model information
            self.models[model_key] = {
                "name": model_name_or_path,
                "device": device,
                "model_type": model_type,
                "tokenizer": tokenizer,
                "model": model,
                "loaded": True,
                "load_time": load_time,
                "config": config,
                "optimum_model": True
            }
            
            # Get model information
            model_info = {
                "model_type": model_type,
                "device": device,
                "load_time_sec": load_time,
                "tokenizer_type": type(tokenizer).__name__,
                "model_class": type(model).__name__
            }
            
            return {
                "status": "success",
                "model_name": model_name_or_path,
                "device": device,
                "model_type": model_type,
                "model_info": model_info,
                "load_time_sec": load_time
            }
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            return {"status": "error", "message": f"Failed to import required modules: {str(e)}"}
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {str(e)}")
            return {"status": "error", "message": f"Failed to load HuggingFace model: {str(e)}"}
    
    def convert_from_pytorch(self, model, example_inputs, output_path, config=None) -> Dict[str, Any]:
        """
        Convert PyTorch model to OpenVINO format via ONNX.
        
        Args:
            model: PyTorch model to convert.
            example_inputs: Example inputs for tracing.
            output_path: Path to save converted model.
            config: Configuration options.
            
        Returns:
            Dictionary with conversion result.
        """
        if not self.is_available():
            logger.error("OpenVINO is not available")
            return {"status": "error", "message": "OpenVINO is not available"}
        
        config = config or {}
        precision = config.get("precision", "FP32")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Set ONNX path for intermediate conversion
        onnx_path = output_path.replace(".xml", ".onnx")
        if output_path.endswith(".xml"):
            onnx_path = output_path.replace(".xml", ".onnx")
        else:
            onnx_path = output_path + ".onnx"
        
        logger.info(f"Converting PyTorch model to OpenVINO format via ONNX with precision {precision}")
        logger.info(f"ONNX intermediate path: {onnx_path}")
        logger.info(f"Final output path: {output_path}")
        
        try:
            import torch
            import openvino as ov
            
            # Step 1: Convert PyTorch model to ONNX
            start_time = time.time()
            
            # Set ONNX export parameters
            dynamic_axes = config.get("dynamic_axes")
            input_names = config.get("input_names")
            output_names = config.get("output_names")
            
            # If input/output names not provided, try to infer them
            if input_names is None:
                # Try to infer input names
                if isinstance(example_inputs, dict):
                    input_names = list(example_inputs.keys())
                else:
                    input_names = ["input"]
                    
            if output_names is None:
                # Use default output names
                output_names = ["output"]
            
            # Put model in evaluation mode
            model.eval()
            
            # Determine ONNX export API based on PyTorch version
            logger.info("Exporting PyTorch model to ONNX...")
            
            if hasattr(torch.onnx, "export"):
                # Standard export API
                torch.onnx.export(model, 
                                 example_inputs,
                                 onnx_path,
                                 export_params=True,
                                 opset_version=config.get("opset_version", 13),
                                 do_constant_folding=True,
                                 input_names=input_names,
                                 output_names=output_names,
                                 dynamic_axes=dynamic_axes)
            else:
                logger.error("torch.onnx.export not found - pytorch installation may be incomplete")
                return {"status": "error", "message": "torch.onnx.export not found"}
            
            # Verify ONNX file was created
            if not os.path.exists(onnx_path):
                logger.error("ONNX export failed - file not created")
                return {"status": "error", "message": "ONNX export failed - file not created"}
            
            onnx_export_time = time.time() - start_time
            logger.info(f"PyTorch to ONNX conversion completed in {onnx_export_time:.2f} seconds")
            
            # Step 2: Convert ONNX to OpenVINO IR
            ov_result = self.convert_from_onnx(onnx_path, output_path, config)
            
            # Check if OpenVINO conversion was successful
            if ov_result.get("status") != "success":
                logger.error(f"ONNX to OpenVINO conversion failed: {ov_result.get('message')}")
                return ov_result
            
            # Add additional information to result
            total_time = time.time() - start_time
            result = {
                "status": "success",
                "output_path": ov_result.get("output_path"),
                "precision": precision,
                "message": "Model converted successfully",
                "pytorch_to_onnx_time_sec": onnx_export_time,
                "total_conversion_time_sec": total_time,
                "model_size_mb": ov_result.get("model_size_mb"),
                "inputs": ov_result.get("inputs"),
                "outputs": ov_result.get("outputs")
            }
            
            # Keep or delete ONNX file based on config
            if not config.get("keep_onnx", False):
                try:
                    os.remove(onnx_path)
                    logger.info(f"Removed intermediate ONNX file: {onnx_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove intermediate ONNX file: {str(e)}")
            
            return result
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            return {"status": "error", "message": f"Failed to import required modules: {str(e)}"}
        except Exception as e:
            logger.error(f"Failed to convert model: {str(e)}")
            return {"status": "error", "message": f"Failed to convert model: {str(e)}"}
    
    def convert_from_onnx(self, onnx_path, output_path, config=None) -> Dict[str, Any]:
        """
        Convert ONNX model to OpenVINO format.
        
        Args:
            onnx_path: Path to ONNX model.
            output_path: Path to save converted model.
            config: Configuration options.
            
        Returns:
            Dictionary with conversion result.
        """
        if not self.is_available():
            logger.error("OpenVINO is not available")
            return {"status": "error", "message": "OpenVINO is not available"}
        
        # Verify the ONNX file exists
        if not os.path.exists(onnx_path):
            logger.error(f"ONNX file not found at {onnx_path}")
            return {"status": "error", "message": f"ONNX file not found at {onnx_path}"}
        
        config = config or {}
        precision = config.get("precision", "FP32")
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logger.info(f"Converting ONNX model from {onnx_path} to OpenVINO IR format with precision {precision}")
        logger.info(f"Output path: {output_path}")
        
        try:
            import openvino as ov
            
            # Read the ONNX model
            start_time = time.time()
            
            # Set conversion parameters
            conversion_params = {}
            
            # Specify input shapes if provided
            if "input_shapes" in config:
                conversion_params["input"] = config["input_shapes"]
            
            # Set model layout if provided
            if "layout" in config:
                conversion_params["layout"] = config["layout"]
            
            # Enable transformations for better performance
            conversion_params["static_shape"] = not config.get("dynamic_shapes", True)
            
            # Convert to OpenVINO IR using the Core API
            ov_model = self._core.read_model(onnx_path)
            
            # Apply precision-specific optimizations
            if precision == "FP16":
                ov_model = self._apply_fp16_transformations(ov_model)
            elif precision == "INT8":
                ov_model = self._apply_int8_transformations(ov_model)
            
            # Save the model to disk
            xml_path = output_path
            if not xml_path.endswith(".xml"):
                xml_path += ".xml"
                
            bin_path = xml_path.replace(".xml", ".bin")
            
            # Save the model
            # The save_model has a different API depending on OpenVINO version
            try:
                # Newer versions use positional arguments
                if precision == "FP16":
                    ov.save_model(ov_model, xml_path, True)  # compress_to_fp16=True
                else:
                    ov.save_model(ov_model, xml_path)
            except TypeError:
                # Try the older API with keyword arguments
                save_params = {}
                if precision == "FP16":
                    save_params["compress_to_fp16"] = True
                
                # Try different call patterns based on API version
                try:
                    ov.save_model(model=ov_model, model_path=xml_path, **save_params)
                except TypeError:
                    try:
                        ov.save_model(model=ov_model, path=xml_path, **save_params)
                    except TypeError:
                        # Last resort: try without parameters
                        ov.save_model(ov_model, xml_path)
            
            # Verify model files were created
            if not os.path.exists(xml_path) or not os.path.exists(bin_path):
                logger.error("Failed to save model files")
                return {"status": "error", "message": "Failed to save model files"}
                
            # Get model info
            model_size_mb = os.path.getsize(bin_path) / (1024 * 1024)
            conversion_time = time.time() - start_time
            
            logger.info(f"Successfully converted ONNX model to OpenVINO IR in {conversion_time:.2f} seconds")
            logger.info(f"Model size: {model_size_mb:.2f} MB")
            
            return {
                "status": "success",
                "output_path": xml_path,
                "precision": precision,
                "message": "Model converted successfully",
                "model_size_mb": model_size_mb,
                "conversion_time_sec": conversion_time,
                "inputs": {name: port.get_shape() for name, port in ov_model.inputs.items()},
                "outputs": {name: port.get_shape() for name, port in ov_model.outputs.items()}
            }
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            return {"status": "error", "message": f"Failed to import required modules: {str(e)}"}
        except Exception as e:
            logger.error(f"Failed to convert model: {str(e)}")
            return {"status": "error", "message": f"Failed to convert model: {str(e)}"}