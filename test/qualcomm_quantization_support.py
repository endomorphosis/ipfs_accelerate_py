#!/usr/bin/env python3

import os
import sys
import json
import time
import argparse
import importlib.util
from typing import Dict, List, Any, Optional, Union, Tuple
import traceback

# Try to import necessary packages
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: NumPy not found. This is required for quantization.")

# Configure paths
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import local modules
try:
    from test_ipfs_accelerate import QualcommTestHandler, TestResultsDBHandler
    HAS_TEST_MODULES = True
except ImportError:
    HAS_TEST_MODULES = False
    print("Warning: Could not import QualcommTestHandler. Make sure test_ipfs_accelerate.py is in the path.")

# Try importing quality models modules
try:
    from centralized_hardware_detection import hardware_detection
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    print("Warning: Could not import hardware_detection module.")

# Define quantization methods
QUANTIZATION_METHODS = {
    "dynamic": "Dynamic quantization (qint8)",
    "static": "Static quantization with calibration data (qint8)",
    "weight_only": "Weight-only quantization (keeps activations in fp32)",
    "int8": "Full INT8 quantization",
    "int4": "Ultra-low precision INT4 quantization",
    "mixed": "Mixed precision (different parts of the model at different precisions)"
}

class QualcommQuantization:
    """
    Implements quantization support for Qualcomm AI Engine.
    
    This class enables various quantization methods for models running on Qualcomm
    hardware, with a focus on power efficiency and performance for mobile/edge deployment.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the Qualcomm quantization handler.
        
        Args:
            db_path: Path to DuckDB database for storing results
        """
        self.db_path = db_path
        self.qualcomm_handler = None
        self.db_handler = None
        self.mock_mode = False
        
        # Initialize handlers
        self._init_handlers()
        
    def _init_handlers(self):
        """Initialize Qualcomm handler and database handler."""
        if not HAS_TEST_MODULES:
            print("Error: QualcommTestHandler could not be imported.")
            return

        # Initialize Qualcomm test handler
        self.qualcomm_handler = QualcommTestHandler()
        print(f"Initialized Qualcomm handler (available: {self.qualcomm_handler.is_available()})")
        
        # Set mock mode if real hardware isn't available
        if not self.qualcomm_handler.is_available():
            self.mock_mode = os.environ.get("QUALCOMM_MOCK", "1") == "1"
            self.qualcomm_handler.mock_mode = self.mock_mode
            print(f"No Qualcomm hardware detected. Mock mode: {self.mock_mode}")
        
        # Initialize database handler
        if self.db_path:
            self.db_handler = TestResultsDBHandler(self.db_path)
            print(f"Initialized database handler with path: {self.db_path}")
        
    def is_available(self) -> bool:
        """Check if Qualcomm quantization is available."""
        return (self.qualcomm_handler is not None and 
                (self.qualcomm_handler.is_available() or self.mock_mode))
    
    def list_quantization_methods(self) -> Dict[str, str]:
        """List available quantization methods with descriptions."""
        return QUANTIZATION_METHODS
    
    def get_supported_methods(self) -> Dict[str, bool]:
        """Get quantization methods supported by the current Qualcomm configuration."""
        if not self.is_available():
            return {method: False for method in QUANTIZATION_METHODS}
        
        # Check SDK capabilities - different SDKs support different methods
        sdk_type = self.qualcomm_handler.sdk_type
        supported = {
            "dynamic": True,  # All SDKs support dynamic quantization
            "static": True,   # All SDKs support static quantization
            "weight_only": True, # All SDKs support weight-only quantization
            "int8": True,     # All SDKs support INT8
            "int4": sdk_type == "QNN" and hasattr(self.qualcomm_handler, "_convert_model_qnn"),  # Only QNN SDK supports INT4
            "mixed": sdk_type == "QNN" and hasattr(self.qualcomm_handler, "_convert_model_qnn")  # Only QNN SDK supports mixed precision
        }
        
        # In mock mode, support everything
        if self.mock_mode:
            supported = {method: True for method in QUANTIZATION_METHODS}
            
        return supported
    
    def quantize_model(self, 
                       model_path: str, 
                       output_path: str, 
                       method: str = "dynamic", 
                       model_type: str = "text",
                       calibration_data: Any = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Quantize a model using the specified method.
        
        Args:
            model_path: Path to input model (ONNX or PyTorch)
            output_path: Path for converted model
            method: Quantization method (dynamic, static, weight_only, int8, int4, mixed)
            model_type: Type of model (text, vision, audio, llm)
            calibration_data: Calibration data for static quantization
            **kwargs: Additional parameters for quantization
            
        Returns:
            dict: Quantization results
        """
        if not self.is_available():
            return {"error": "Qualcomm quantization not available"}
        
        # Check if method is supported
        supported_methods = self.get_supported_methods()
        if method not in supported_methods:
            return {"error": f"Quantization method '{method}' not recognized. Available methods: {list(supported_methods.keys())}"}
        
        if not supported_methods[method]:
            return {"error": f"Quantization method '{method}' not supported by current Qualcomm configuration"}
            
        # Validate model type
        valid_model_types = ["text", "vision", "audio", "llm"]
        if model_type not in valid_model_types:
            return {"error": f"Invalid model type: {model_type}. Must be one of: {valid_model_types}"}
        
        # Start timing
        start_time = time.time()
        
        # Apply quantization
        try:
            # Set conversion parameters based on quantization method
            conversion_params = self._get_conversion_params(method, model_type, calibration_data, **kwargs)
            
            # Add quantization to parameters
            conversion_params["quantization"] = True
            conversion_params["quantization_method"] = method
            
            # Convert and quantize model
            if self.mock_mode:
                # In mock mode, simulate quantization
                result = self._mock_quantize_model(model_path, output_path, method, model_type, conversion_params)
            else:
                # Real quantization with appropriate SDK
                if self.qualcomm_handler.sdk_type == "QNN":
                    result = self._quantize_model_qnn(model_path, output_path, method, model_type, conversion_params)
                elif self.qualcomm_handler.sdk_type == "QTI":
                    result = self._quantize_model_qti(model_path, output_path, method, model_type, conversion_params)
                else:
                    return {"error": f"Unsupported SDK type: {self.qualcomm_handler.sdk_type}"}
            
            # Calculate metrics and add to result
            quantization_time = time.time() - start_time
            result["quantization_time"] = quantization_time
            
            # Add power efficiency metrics
            power_metrics = self._estimate_power_efficiency(model_type, method)
            result["power_efficiency_metrics"] = power_metrics
            
            # Add device info
            result["device_info"] = self.qualcomm_handler.get_device_info()
            
            # Store results in database if available
            if self.db_handler and hasattr(self.db_handler, "is_available") and self.db_handler.is_available():
                self._store_quantization_results(result, model_path, output_path, method, model_type)
            
            return result
            
        except Exception as e:
            error_result = {
                "error": f"Error during quantization: {str(e)}",
                "traceback": traceback.format_exc(),
                "method": method,
                "model_type": model_type
            }
            print(f"Error quantizing model: {str(e)}")
            print(traceback.format_exc())
            return error_result
    
    def _get_conversion_params(self, method: str, model_type: str, calibration_data: Any, **kwargs) -> Dict[str, Any]:
        """Get conversion parameters based on quantization method and model type."""
        # Base parameters for all methods
        params = {
            "model_type": model_type
        }
        
        # Method-specific parameters
        if method == "dynamic":
            params["dynamic_quantization"] = True
            params["quantization_dtype"] = "qint8"
        elif method == "static":
            params["static_quantization"] = True
            params["quantization_dtype"] = "qint8"
            if calibration_data is not None:
                params["calibration_data"] = calibration_data
        elif method == "weight_only":
            params["weight_only_quantization"] = True
            params["quantization_dtype"] = "qint8"
            params["keep_fp32_activations"] = True
        elif method == "int8":
            params["int8_quantization"] = True
            params["quantization_dtype"] = "qint8"
        elif method == "int4":
            params["int4_quantization"] = True
            params["quantization_dtype"] = "qint4"
        elif method == "mixed":
            params["mixed_precision"] = True
            # Default mixed precision configuration
            mixed_config = {
                "weights": "int4",
                "activations": "int8",
                "attention": "int8",
                "output": "fp16"
            }
            # Override with user-provided config if available
            if "mixed_config" in kwargs:
                mixed_config.update(kwargs["mixed_config"])
            params["mixed_precision_config"] = mixed_config
        
        # Add model-type specific optimizations
        if model_type == "text":
            params["optimize_text_models"] = True
        elif model_type == "vision":
            params["input_layout"] = "NCHW"
            params["optimize_vision_models"] = True
        elif model_type == "audio":
            params["optimize_audio_models"] = True
        elif model_type == "llm":
            params["optimize_llm"] = True
            params["enable_kv_cache"] = True
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key != "mixed_config":  # Already handled above
                params[key] = value
                
        return params
    
    def _mock_quantize_model(self, model_path: str, output_path: str, method: str, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation for testing without real hardware."""
        print(f"Mock Qualcomm: Quantizing {model_path} to {output_path} using {method} method for {model_type} model")
        
        # Simulate model size reduction based on quantization method
        size_reduction_map = {
            "dynamic": 0.25,    # 4x reduction
            "static": 0.22,     # 4.5x reduction
            "weight_only": 0.30, # 3.3x reduction
            "int8": 0.25,       # 4x reduction
            "int4": 0.12,       # 8x reduction
            "mixed": 0.18       # 5.5x reduction
        }
        
        # Simulate latency improvement
        latency_improvement_map = {
            "dynamic": 0.85,    # 15% faster
            "static": 0.75,     # 25% faster
            "weight_only": 0.80, # 20% faster
            "int8": 0.70,       # 30% faster
            "int4": 0.65,       # 35% faster
            "mixed": 0.72       # 28% faster
        }
        
        # Create mock result
        result = {
            "status": "success",
            "input_path": model_path,
            "output_path": output_path,
            "model_type": model_type,
            "quantization_method": method,
            "params": params,
            "mock_mode": True,
            "size_reduction_ratio": 1.0 / size_reduction_map.get(method, 0.25),
            "latency_improvement_ratio": 1.0 / latency_improvement_map.get(method, 0.85),
            "sdk_type": self.qualcomm_handler.sdk_type or "MOCK_SDK"
        }
        
        return result
    
    def _quantize_model_qnn(self, model_path: str, output_path: str, method: str, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Quantize model using QNN SDK."""
        # This will be implemented with real QNN SDK
        import qnn_wrapper
        
        # Add QNN-specific parameters
        qnn_params = params.copy()
        
        # Method-specific QNN parameters
        if method == "int4":
            qnn_params["enable_low_precision"] = True
            qnn_params["weight_precision"] = "int4"
        elif method == "mixed":
            qnn_params["enable_mixed_precision"] = True
            mixed_config = qnn_params.get("mixed_precision_config", {})
            qnn_params["weight_bitwidth"] = 4 if mixed_config.get("weights") == "int4" else 8
            qnn_params["activation_bitwidth"] = 8 if mixed_config.get("activations") == "int8" else 16
            
        # Ensure model_path and output_path are set correctly
        qnn_params["input_model"] = model_path
        qnn_params["output_model"] = output_path
        
        # Convert and quantize model
        qnn_result = qnn_wrapper.convert_model(**qnn_params)
        
        # Extract relevant metrics
        orig_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        new_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        
        # Create result structure
        result = {
            "status": "success" if qnn_result else "failure",
            "input_path": model_path,
            "output_path": output_path,
            "model_type": model_type,
            "quantization_method": method,
            "params": params,
            "sdk_type": "QNN",
            "original_size": orig_size,
            "quantized_size": new_size
        }
        
        # Calculate size reduction if sizes are available
        if orig_size > 0 and new_size > 0:
            result["size_reduction_ratio"] = orig_size / new_size
            
        return result
    
    def _quantize_model_qti(self, model_path: str, output_path: str, method: str, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Quantize model using QTI SDK."""
        # This will be implemented with real QTI SDK
        from qti.aisw import dlc_utils
        
        # Add QTI-specific parameters
        qti_params = params.copy()
        
        # Method-specific QTI parameters
        if method == "int8":
            qti_params["quantization"] = "symmetric_8bit"
        elif method == "weight_only":
            qti_params["quantization"] = "weight_only_8bit"
        elif method == "dynamic":
            qti_params["quantization"] = "dynamic_8bit"
        elif method == "static":
            qti_params["quantization"] = "symmetric_8bit"
            if "calibration_data" in qti_params:
                qti_params["calibration_dataset"] = qti_params.pop("calibration_data")
        else:
            # INT4 and mixed precision may not be supported by QTI
            return {"error": f"Quantization method '{method}' not supported by QTI SDK"}
        
        # Ensure model_path and output_path are set correctly
        qti_params["input_model"] = model_path
        qti_params["output_model"] = output_path
        
        # Convert and quantize model
        qti_result = dlc_utils.convert_onnx_to_dlc(**qti_params)
        
        # Extract relevant metrics
        orig_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        new_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        
        # Create result structure
        result = {
            "status": "success" if qti_result else "failure",
            "input_path": model_path,
            "output_path": output_path,
            "model_type": model_type,
            "quantization_method": method,
            "params": params,
            "sdk_type": "QTI",
            "original_size": orig_size,
            "quantized_size": new_size
        }
        
        # Calculate size reduction if sizes are available
        if orig_size > 0 and new_size > 0:
            result["size_reduction_ratio"] = orig_size / new_size
            
        return result
    
    def _estimate_power_efficiency(self, model_type: str, method: str) -> Dict[str, float]:
        """Estimate power efficiency metrics based on model type and quantization method."""
        # Base power efficiency metrics by model type
        base_metrics = {
            "text": {
                "power_consumption_mw": 400.0,
                "energy_efficiency_items_per_joule": 150.0,
                "battery_impact_percent_per_hour": 2.5
            },
            "vision": {
                "power_consumption_mw": 550.0,
                "energy_efficiency_items_per_joule": 80.0,
                "battery_impact_percent_per_hour": 3.0
            },
            "audio": {
                "power_consumption_mw": 500.0,
                "energy_efficiency_items_per_joule": 65.0,
                "battery_impact_percent_per_hour": 2.8
            },
            "llm": {
                "power_consumption_mw": 650.0,
                "energy_efficiency_items_per_joule": 35.0,
                "battery_impact_percent_per_hour": 4.0
            }
        }
        
        # Improvement factors by quantization method
        improvement_factors = {
            "dynamic": {
                "power_factor": 0.85,    # 15% power reduction
                "efficiency_factor": 1.15, # 15% efficiency improvement
                "battery_factor": 0.85    # 15% battery impact reduction
            },
            "static": {
                "power_factor": 0.80,    # 20% power reduction
                "efficiency_factor": 1.25, # 25% efficiency improvement
                "battery_factor": 0.80    # 20% battery impact reduction
            },
            "weight_only": {
                "power_factor": 0.90,    # 10% power reduction
                "efficiency_factor": 1.10, # 10% efficiency improvement
                "battery_factor": 0.90    # 10% battery impact reduction
            },
            "int8": {
                "power_factor": 0.75,    # 25% power reduction
                "efficiency_factor": 1.30, # 30% efficiency improvement
                "battery_factor": 0.75    # 25% battery impact reduction
            },
            "int4": {
                "power_factor": 0.65,    # 35% power reduction
                "efficiency_factor": 1.50, # 50% efficiency improvement
                "battery_factor": 0.65    # 35% battery impact reduction
            },
            "mixed": {
                "power_factor": 0.70,    # 30% power reduction
                "efficiency_factor": 1.40, # 40% efficiency improvement
                "battery_factor": 0.70    # 30% battery impact reduction
            }
        }
        
        # Get base metrics for model type
        metrics = base_metrics.get(model_type, base_metrics["text"]).copy()
        
        # Apply improvement factors
        factors = improvement_factors.get(method, improvement_factors["dynamic"])
        metrics["power_consumption_mw"] *= factors["power_factor"]
        metrics["energy_efficiency_items_per_joule"] *= factors["efficiency_factor"]
        metrics["battery_impact_percent_per_hour"] *= factors["battery_factor"]
        
        # Add additional derived metrics
        metrics["power_reduction_percent"] = (1 - factors["power_factor"]) * 100
        metrics["efficiency_improvement_percent"] = (factors["efficiency_factor"] - 1) * 100
        metrics["battery_savings_percent"] = (1 - factors["battery_factor"]) * 100
        
        # Add thermal metrics
        thermal_improvement = (1 - factors["power_factor"]) * 1.5  # Thermal improvement is greater than power reduction
        metrics["estimated_thermal_reduction_percent"] = thermal_improvement * 100
        metrics["thermal_throttling_risk"] = "Low" if thermal_improvement > 0.3 else "Medium" if thermal_improvement > 0.15 else "High"
        
        return metrics
    
    def _store_quantization_results(self, 
                                   result: Dict[str, Any], 
                                   model_path: str, 
                                   output_path: str, 
                                   method: str, 
                                   model_type: str) -> bool:
        """Store quantization results in the database."""
        if not self.db_handler or not hasattr(self.db_handler, "api") or not self.db_handler.api:
            return False
            
        try:
            # Extract key values
            original_size = result.get("original_size", 0)
            quantized_size = result.get("quantized_size", 0)
            reduction_ratio = result.get("size_reduction_ratio", 0)
            power_metrics = result.get("power_efficiency_metrics", {})
            
            # Create database entry
            query = """
            INSERT INTO model_conversion_metrics (
                model_name, source_format, target_format, hardware_target, 
                conversion_success, conversion_time, file_size_before, file_size_after,
                precision, optimization_level, error_message, timestamp,
                power_consumption_mw, energy_efficiency_items_per_joule,
                battery_impact_percent_per_hour, thermal_throttling_risk,
                quantization_method, model_type, sdk_type, sdk_version,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP,
                     ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            # Determine formats
            source_format = os.path.splitext(model_path)[1].lstrip(".") if model_path else "unknown"
            target_format = os.path.splitext(output_path)[1].lstrip(".") if output_path else "qnn"
            
            # Extract device info
            device_info = result.get("device_info", {})
            sdk_type = device_info.get("sdk_type", result.get("sdk_type", "unknown"))
            sdk_version = device_info.get("sdk_version", "unknown")
            
            # Prepare parameters
            params = [
                os.path.basename(model_path),                # model_name
                source_format,                               # source_format
                target_format,                               # target_format
                "qualcomm",                                  # hardware_target
                result.get("status") == "success",           # conversion_success
                result.get("quantization_time", 0),          # conversion_time
                original_size,                               # file_size_before
                quantized_size,                              # file_size_after
                method,                                      # precision
                1,                                           # optimization_level
                result.get("error", ""),                     # error_message
                power_metrics.get("power_consumption_mw", 0),             # power_consumption_mw
                power_metrics.get("energy_efficiency_items_per_joule", 0), # energy_efficiency_items_per_joule
                power_metrics.get("battery_impact_percent_per_hour", 0),   # battery_impact_percent_per_hour
                power_metrics.get("thermal_throttling_risk", "Unknown"),   # thermal_throttling_risk
                method,                                      # quantization_method
                model_type,                                  # model_type
                sdk_type,                                    # sdk_type
                sdk_version,                                 # sdk_version
                json.dumps(result)                           # metadata
            ]
            
            # Execute the query
            self.db_handler.api.execute_query(query, params)
            print(f"Stored quantization results in database for {os.path.basename(model_path)}")
            return True
        except Exception as e:
            print(f"Error storing quantization results in database: {e}")
            print(traceback.format_exc())
            return False
    
    def benchmark_quantized_model(self, 
                                 model_path: str, 
                                 inputs: Any = None,
                                 model_type: str = None,
                                 **kwargs) -> Dict[str, Any]:
        """
        Benchmark a quantized model for performance and power efficiency.
        
        Args:
            model_path: Path to the quantized model
            inputs: Input data for benchmarking
            model_type: Type of model (text, vision, audio, llm)
            **kwargs: Additional parameters for benchmarking
            
        Returns:
            dict: Benchmark results
        """
        if not self.is_available():
            return {"error": "Qualcomm quantization not available"}
            
        # Create sample inputs if not provided
        if inputs is None and HAS_NUMPY and model_type:
            inputs = self._create_sample_input(model_type)
            
        if model_type is None:
            # Try to infer model type from path
            model_type = self._infer_model_type_from_path(model_path)
            
        # Run benchmark with power monitoring
        try:
            if self.mock_mode:
                # Mock benchmark
                benchmark_result = self._mock_benchmark(model_path, model_type)
            else:
                # Real benchmark
                benchmark_result = self.qualcomm_handler.run_inference(
                    model_path=model_path,
                    input_data=inputs,
                    monitor_metrics=True,
                    model_type=model_type,
                    **kwargs
                )
                
            return benchmark_result
        
        except Exception as e:
            error_result = {
                "error": f"Error benchmarking model: {str(e)}",
                "traceback": traceback.format_exc(),
                "model_path": model_path,
                "model_type": model_type
            }
            print(f"Error benchmarking model: {str(e)}")
            print(traceback.format_exc())
            return error_result
    
    def _create_sample_input(self, model_type: str) -> Any:
        """Create appropriate sample input based on model type."""
        if not HAS_NUMPY:
            return None
            
        if model_type == "vision":
            # Image tensor for vision models (batch_size, channels, height, width)
            return np.random.randn(1, 3, 224, 224).astype(np.float32)
        elif model_type == "audio":
            # Audio waveform for audio models (batch_size, samples)
            return np.random.randn(1, 16000).astype(np.float32)  # 1 second at 16kHz
        elif model_type == "llm":
            # Text prompt for language models
            return "This is a longer sample text for testing language models with the Qualcomm AI Engine. This text will be used for benchmarking inference performance on mobile hardware."
        else:
            # Simple text for embedding models
            return "This is a sample text for testing Qualcomm endpoint"
    
    def _infer_model_type_from_path(self, model_path: str) -> str:
        """Infer model type from model path."""
        model_path = str(model_path).lower()
        
        # Check model path for indicators
        if any(x in model_path for x in ["vit", "clip", "vision", "image", "resnet", "detr", "vgg"]):
            return "vision"
        elif any(x in model_path for x in ["whisper", "wav2vec", "clap", "audio", "speech", "voice"]):
            return "audio"
        elif any(x in model_path for x in ["llava", "llama", "gpt", "llm", "falcon", "mistral", "phi"]):
            return "llm"
        elif any(x in model_path for x in ["bert", "roberta", "text", "embed", "sentence", "bge"]):
            return "text"
        
        # Default to text if no indicators found
        return "text"
    
    def _mock_benchmark(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Mock benchmark for testing without real hardware."""
        print(f"Mock Qualcomm: Benchmarking {model_path} (type: {model_type})")
        
        # Generate mock benchmark results
        latency_ms = {
            "text": 5.0,
            "vision": 15.0,
            "audio": 25.0,
            "llm": 40.0
        }.get(model_type, 10.0)
        
        throughput = {
            "text": 120.0,  # tokens/second
            "vision": 50.0,  # images/second
            "audio": 8.0,    # seconds of audio/second
            "llm": 20.0      # tokens/second
        }.get(model_type, 50.0)
        
        throughput_units = {
            "text": "tokens/second",
            "vision": "images/second",
            "audio": "seconds of audio/second",
            "llm": "tokens/second"
        }.get(model_type, "samples/second")
        
        # Generate mock power metrics
        power_metrics = {
            "power_consumption_mw": {
                "text": 350.0,
                "vision": 450.0,
                "audio": 400.0,
                "llm": 550.0
            }.get(model_type, 400.0),
            
            "energy_consumption_mj": {
                "text": 35.0,
                "vision": 67.5,
                "audio": 100.0,
                "llm": 220.0
            }.get(model_type, 50.0),
            
            "temperature_celsius": {
                "text": 38.0,
                "vision": 42.0,
                "audio": 41.0,
                "llm": 45.0
            }.get(model_type, 40.0),
            
            "monitoring_duration_ms": 1000.0,
            
            "average_power_mw": {
                "text": 350.0,
                "vision": 450.0,
                "audio": 400.0,
                "llm": 550.0
            }.get(model_type, 400.0),
            
            "peak_power_mw": {
                "text": 420.0,
                "vision": 540.0,
                "audio": 480.0,
                "llm": 660.0
            }.get(model_type, 480.0),
            
            "idle_power_mw": {
                "text": 140.0,
                "vision": 180.0,
                "audio": 160.0,
                "llm": 220.0
            }.get(model_type, 160.0),
            
            "energy_efficiency_items_per_joule": {
                "text": 150.0,
                "vision": 80.0,
                "audio": 65.0,
                "llm": 35.0
            }.get(model_type, 100.0),
            
            "thermal_throttling_detected": False,
            
            "battery_impact_percent_per_hour": {
                "text": 2.5,
                "vision": 3.0,
                "audio": 2.8,
                "llm": 4.0
            }.get(model_type, 3.0),
            
            "model_type": model_type
        }
        
        # Create mock output
        import numpy as np
        if model_type == "vision":
            mock_output = np.random.randn(1, 1000)  # Classification logits
        elif model_type == "text":
            mock_output = np.random.randn(1, 768)  # Embedding vector
        elif model_type == "audio":
            mock_output = np.random.randn(1, 128, 20)  # Audio features
        elif model_type == "llm":
            # Generate a small token sequence
            mock_output = np.random.randint(0, 50000, size=(1, 10))
        else:
            mock_output = np.random.randn(1, 768)  # Default embedding
        
        # Generate complete result
        benchmark_result = {
            "status": "success",
            "output": mock_output,
            "metrics": power_metrics,
            "device_info": {
                "device_name": "Mock Qualcomm Device",
                "sdk_type": self.qualcomm_handler.sdk_type or "MOCK_SDK",
                "sdk_version": self.qualcomm_handler.sdk_version or "unknown",
                "mock_mode": True,
                "has_power_metrics": True,
                "model_type": model_type
            },
            "sdk_type": self.qualcomm_handler.sdk_type or "MOCK_SDK",
            "model_type": model_type,
            "throughput": throughput,
            "throughput_units": throughput_units,
            "latency_ms": latency_ms,
            "mock_mode": True
        }
        
        return benchmark_result
    
    def compare_quantization_methods(self, 
                                     model_path: str, 
                                     output_dir: str, 
                                     model_type: str = None,
                                     methods: List[str] = None) -> Dict[str, Any]:
        """
        Compare different quantization methods for a given model.
        
        Args:
            model_path: Path to input model
            output_dir: Directory for saving quantized models
            model_type: Type of model (text, vision, audio, llm)
            methods: List of quantization methods to compare (if None, tests all supported methods)
            
        Returns:
            dict: Comparison results
        """
        if not self.is_available():
            return {"error": "Qualcomm quantization not available"}
            
        # Infer model type if not provided
        if model_type is None:
            model_type = self._infer_model_type_from_path(model_path)
            
        # Get supported methods if not provided
        supported_methods = self.get_supported_methods()
        if methods is None:
            methods = [method for method, supported in supported_methods.items() if supported]
        else:
            # Filter out unsupported methods
            methods = [method for method in methods if method in supported_methods and supported_methods[method]]
            
        if not methods:
            return {"error": "No supported quantization methods available"}
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results dictionary
        comparison_results = {
            "model_path": model_path,
            "model_type": model_type,
            "output_dir": output_dir,
            "methods_compared": methods,
            "results": {},
            "summary": {},
            "power_comparison": {},
            "size_comparison": {},
            "latency_comparison": {}
        }
        
        # Create sample inputs for benchmarking
        sample_input = self._create_sample_input(model_type)
        
        # Test each method
        for method in methods:
            print(f"Testing quantization method: {method}")
            
            # Set output path for quantized model
            output_path = os.path.join(output_dir, f"{os.path.basename(model_path)}.{method}.qnn")
            
            # Quantize model
            quant_result = self.quantize_model(
                model_path=model_path,
                output_path=output_path,
                method=method,
                model_type=model_type
            )
            
            # Skip failed quantizations
            if "error" in quant_result:
                comparison_results["results"][method] = {
                    "status": "error",
                    "error": quant_result["error"]
                }
                continue
                
            # Benchmark quantized model
            benchmark_result = self.benchmark_quantized_model(
                model_path=output_path,
                inputs=sample_input,
                model_type=model_type
            )
            
            # Store combined results
            comparison_results["results"][method] = {
                "quantization": quant_result,
                "benchmark": benchmark_result
            }
            
            # Extract key metrics for comparison
            size_reduction = quant_result.get("size_reduction_ratio", 1.0)
            latency_ms = benchmark_result.get("latency_ms", 0.0)
            power_metrics = benchmark_result.get("metrics", {})
            
            # Store in comparison tables
            comparison_results["power_comparison"][method] = {
                "power_consumption_mw": power_metrics.get("power_consumption_mw", 0.0),
                "energy_efficiency_items_per_joule": power_metrics.get("energy_efficiency_items_per_joule", 0.0),
                "battery_impact_percent_per_hour": power_metrics.get("battery_impact_percent_per_hour", 0.0)
            }
            
            comparison_results["size_comparison"][method] = {
                "size_reduction_ratio": size_reduction,
                "size_reduction_percent": (1 - 1/size_reduction) * 100 if size_reduction > 0 else 0
            }
            
            comparison_results["latency_comparison"][method] = {
                "latency_ms": latency_ms,
                "throughput": benchmark_result.get("throughput", 0.0),
                "throughput_units": benchmark_result.get("throughput_units", "items/second")
            }
        
        # Generate summary with best method for each metric
        best_power_method = min(comparison_results["power_comparison"].items(), 
                               key=lambda x: x[1]["power_consumption_mw"], 
                               default=(None, {}))[0]
                               
        best_efficiency_method = max(comparison_results["power_comparison"].items(), 
                                   key=lambda x: x[1]["energy_efficiency_items_per_joule"], 
                                   default=(None, {}))[0]
                                   
        best_battery_method = min(comparison_results["power_comparison"].items(), 
                                key=lambda x: x[1]["battery_impact_percent_per_hour"], 
                                default=(None, {}))[0]
                                
        best_size_method = max(comparison_results["size_comparison"].items(), 
                             key=lambda x: x[1]["size_reduction_ratio"], 
                             default=(None, {}))[0]
                             
        best_latency_method = min(comparison_results["latency_comparison"].items(), 
                                key=lambda x: x[1]["latency_ms"] if x[1]["latency_ms"] > 0 else float('inf'), 
                                default=(None, {}))[0]
                                
        best_throughput_method = max(comparison_results["latency_comparison"].items(), 
                                   key=lambda x: x[1]["throughput"], 
                                   default=(None, {}))[0]
        
        # Create summary
        comparison_results["summary"] = {
            "best_power_efficiency": best_power_method,
            "best_energy_efficiency": best_efficiency_method,
            "best_battery_life": best_battery_method,
            "best_size_reduction": best_size_method,
            "best_latency": best_latency_method,
            "best_throughput": best_throughput_method,
            "overall_recommendation": self._get_overall_recommendation(
                comparison_results, model_type,
                [best_power_method, best_efficiency_method, best_battery_method, 
                 best_size_method, best_latency_method, best_throughput_method]
            )
        }
        
        return comparison_results
    
    def _get_overall_recommendation(self, 
                                   comparison_results: Dict[str, Any], 
                                   model_type: str,
                                   best_methods: List[str]) -> Dict[str, Any]:
        """Get overall recommendation based on comparison results."""
        # Count method occurrences in best_methods
        method_counts = {}
        for method in best_methods:
            if method is not None:
                method_counts[method] = method_counts.get(method, 0) + 1
        
        # Get most common method
        most_common_method = max(method_counts.items(), key=lambda x: x[1], default=(None, 0))[0] if method_counts else None
        
        # Model type specific recommendations
        model_specific_recommendations = {
            "text": {
                "primary_metric": "energy_efficiency_items_per_joule",
                "recommended_method": "int8" if "int8" in comparison_results["results"] else "dynamic",
                "rationale": "Text embedding models benefit most from energy efficiency optimizations."
            },
            "vision": {
                "primary_metric": "throughput",
                "recommended_method": "int8" if "int8" in comparison_results["results"] else "static",
                "rationale": "Vision models typically need to process multiple frames, so throughput is critical."
            },
            "audio": {
                "primary_metric": "battery_impact_percent_per_hour",
                "recommended_method": "mixed" if "mixed" in comparison_results["results"] else "int8",
                "rationale": "Audio processing is often long-running, so battery impact is most important."
            },
            "llm": {
                "primary_metric": "latency_ms",
                "recommended_method": "int4" if "int4" in comparison_results["results"] else "mixed",
                "rationale": "Language models benefit most from aggressive quantization to reduce memory usage and improve latency."
            }
        }
        
        model_rec = model_specific_recommendations.get(model_type, {
            "primary_metric": "energy_efficiency_items_per_joule",
            "recommended_method": "int8",
            "rationale": "General recommendation based on balance of performance and efficiency."
        })
        
        # Find best method for primary metric
        primary_metric = model_rec["primary_metric"]
        
        # Determine best method based on primary metric
        if primary_metric == "energy_efficiency_items_per_joule":
            best_for_primary = summary.get("best_energy_efficiency")
        elif primary_metric == "throughput":
            best_for_primary = summary.get("best_throughput")
        elif primary_metric == "battery_impact_percent_per_hour":
            best_for_primary = summary.get("best_battery_life")
        elif primary_metric == "latency_ms":
            best_for_primary = summary.get("best_latency")
        else:
            best_for_primary = most_common_method
        
        # Combine recommendations
        overall_rec = model_rec.copy()
        overall_rec["most_common_best_method"] = most_common_method
        overall_rec["best_for_primary_metric"] = best_for_primary
        
        # Final recommendation logic
        if best_for_primary in comparison_results["results"]:
            overall_rec["final_recommendation"] = best_for_primary
        elif most_common_method in comparison_results["results"]:
            overall_rec["final_recommendation"] = most_common_method
        else:
            overall_rec["final_recommendation"] = model_rec["recommended_method"]
            
        # Check if final recommendation is valid
        if overall_rec["final_recommendation"] not in comparison_results["results"]:
            # Fall back to first successful method
            for method, result in comparison_results["results"].items():
                if result.get("status") != "error":
                    overall_rec["final_recommendation"] = method
                    overall_rec["rationale"] += " (Fallback recommendation based on available methods.)"
                    break
        
        return overall_rec
    
    def generate_report(self, 
                       comparison_results: Dict[str, Any], 
                       output_path: str = None) -> str:
        """
        Generate a comprehensive report of quantization comparison results.
        
        Args:
            comparison_results: Results from compare_quantization_methods
            output_path: Path to save the report (if None, returns the report as a string)
            
        Returns:
            str: Report content
        """
        # Extract key information
        model_path = comparison_results.get("model_path", "Unknown")
        model_type = comparison_results.get("model_type", "Unknown")
        methods = comparison_results.get("methods_compared", [])
        results = comparison_results.get("results", {})
        summary = comparison_results.get("summary", {})
        power_comparison = comparison_results.get("power_comparison", {})
        size_comparison = comparison_results.get("size_comparison", {})
        latency_comparison = comparison_results.get("latency_comparison", {})
        
        # Generate report header
        report = f"""# Qualcomm AI Engine Quantization Comparison Report

## Overview

- **Model:** {os.path.basename(model_path)}
- **Model Type:** {model_type}
- **Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Methods Compared:** {", ".join(methods)}
- **SDK Type:** {self.qualcomm_handler.sdk_type or "Unknown"}
- **SDK Version:** {self.qualcomm_handler.sdk_version or "Unknown"}

## Summary of Recommendations

- **Overall Recommendation:** {summary.get("overall_recommendation", {}).get("final_recommendation", "Unknown")}
- **Rationale:** {summary.get("overall_recommendation", {}).get("rationale", "Unknown")}
- **Best Power Efficiency:** {summary.get("best_power_efficiency", "Unknown")}
- **Best Energy Efficiency:** {summary.get("best_energy_efficiency", "Unknown")}
- **Best Battery Life:** {summary.get("best_battery_life", "Unknown")}
- **Best Size Reduction:** {summary.get("best_size_reduction", "Unknown")}
- **Best Latency:** {summary.get("best_latency", "Unknown")}
- **Best Throughput:** {summary.get("best_throughput", "Unknown")}

## Comparison Tables

### Power and Energy Efficiency

| Method | Power Consumption (mW) | Energy Efficiency (items/J) | Battery Impact (%/hour) |
|--------|------------------------|----------------------------|-------------------------|
"""
        
        # Add power comparison table
        for method, metrics in sorted(power_comparison.items()):
            report += f"| {method} | {metrics.get('power_consumption_mw', 0):.2f} | {metrics.get('energy_efficiency_items_per_joule', 0):.2f} | {metrics.get('battery_impact_percent_per_hour', 0):.2f} |\n"
        
        # Add size comparison table
        report += """
### Model Size

| Method | Size Reduction Ratio | Size Reduction (%) |
|--------|---------------------|-------------------|
"""
        
        for method, metrics in sorted(size_comparison.items()):
            report += f"| {method} | {metrics.get('size_reduction_ratio', 0):.2f}x | {metrics.get('size_reduction_percent', 0):.2f}% |\n"
        
        # Add latency comparison table
        report += """
### Performance

| Method | Latency (ms) | Throughput | Units |
|--------|-------------|------------|-------|
"""
        
        for method, metrics in sorted(latency_comparison.items()):
            report += f"| {method} | {metrics.get('latency_ms', 0):.2f} | {metrics.get('throughput', 0):.2f} | {metrics.get('throughput_units', '')} |\n"
        
        # Add detailed results for each method
        report += """
## Detailed Results by Method

"""
        
        for method, result in sorted(results.items()):
            if result.get("status") == "error":
                report += f"### {method}\n\n- **Status:** Error\n- **Error:** {result.get('error', 'Unknown error')}\n\n"
                continue
                
            quantization = result.get("quantization", {})
            benchmark = result.get("benchmark", {})
            
            report += f"### {method}\n\n"
            
            # Quantization details
            report += "#### Quantization\n\n"
            report += f"- **Status:** {quantization.get('status', 'Unknown')}\n"
            report += f"- **Size Reduction:** {quantization.get('size_reduction_ratio', 0):.2f}x\n"
            if "original_size" in quantization and "quantized_size" in quantization:
                orig_mb = quantization["original_size"] / (1024 * 1024)
                quant_mb = quantization["quantized_size"] / (1024 * 1024)
                report += f"- **Original Size:** {orig_mb:.2f} MB\n"
                report += f"- **Quantized Size:** {quant_mb:.2f} MB\n"
            
            # Benchmark details
            report += "\n#### Performance\n\n"
            report += f"- **Latency:** {benchmark.get('latency_ms', 0):.2f} ms\n"
            report += f"- **Throughput:** {benchmark.get('throughput', 0):.2f} {benchmark.get('throughput_units', 'items/second')}\n"
            
            # Power metrics
            metrics = benchmark.get("metrics", {})
            if metrics:
                report += "\n#### Power Metrics\n\n"
                report += f"- **Power Consumption:** {metrics.get('power_consumption_mw', 0):.2f} mW\n"
                report += f"- **Energy Efficiency:** {metrics.get('energy_efficiency_items_per_joule', 0):.2f} items/joule\n"
                report += f"- **Battery Impact:** {metrics.get('battery_impact_percent_per_hour', 0):.2f}% per hour\n"
                report += f"- **Thermal Throttling Detected:** {metrics.get('thermal_throttling_detected', False)}\n"
            
            report += "\n"
        
        # Add recommendations
        report += """
## Recommendations for Mobile Deployment

"""
        
        overall_rec = summary.get("overall_recommendation", {})
        final_method = overall_rec.get("final_recommendation", methods[0] if methods else "dynamic")
        
        report += f"### Recommended Method: {final_method}\n\n"
        report += f"- **Rationale:** {overall_rec.get('rationale', 'No rationale provided')}\n"
        report += f"- **Primary Metric for {model_type.capitalize()} Models:** {overall_rec.get('primary_metric', 'Unknown')}\n\n"
        
        # Add method-specific recommendations
        report += """
### Model-Type Specific Considerations

- **Text Models:** Typically benefit most from energy efficiency optimizations. INT8 is a good balance.
- **Vision Models:** Throughput is critical for most vision applications. Static INT8 or mixed precision recommended.
- **Audio Models:** Battery impact is important for long-running audio processing. Mixed precision works well.
- **LLM Models:** Memory constraints are critical. INT4 or mixed precision is recommended.

### Implementation Code

To implement the recommended quantization method:

```python
from qualcomm_quantization_support import QualcommQuantization

# Initialize the quantization module
qquant = QualcommQuantization()

# Apply the recommended quantization
result = qquant.quantize_model(
    model_path="path/to/model",
    output_path="path/to/output",
    method="{final_method}",
    model_type="{model_type}"
)

# Run inference with the quantized model
inference_result = qquant.benchmark_quantized_model(
    model_path="path/to/output", 
    model_type="{model_type}"
)
```
"""
        
        # Save report if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            print(f"Report saved to {output_path}")
        
        return report

def main():
    """Command-line interface for Qualcomm quantization support."""
    parser = argparse.ArgumentParser(description="Qualcomm AI Engine Quantization Support")
    
    # Command groups
    command_group = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List quantization methods
    list_parser = command_group.add_parser("list", help="List available quantization methods")
    
    # Quantize model
    quantize_parser = command_group.add_parser("quantize", help="Quantize a model for Qualcomm AI Engine")
    quantize_parser.add_argument("--model-path", required=True, help="Path to input model (ONNX or PyTorch)")
    quantize_parser.add_argument("--output-path", required=True, help="Path for converted model")
    quantize_parser.add_argument("--method", default="dynamic", help="Quantization method")
    quantize_parser.add_argument("--model-type", default="text", help="Model type (text, vision, audio, llm)")
    quantize_parser.add_argument("--calibration-data", help="Path to calibration data for static quantization")
    quantize_parser.add_argument("--params", help="JSON string with additional parameters")
    
    # Benchmark quantized model
    benchmark_parser = command_group.add_parser("benchmark", help="Benchmark a quantized model")
    benchmark_parser.add_argument("--model-path", required=True, help="Path to quantized model")
    benchmark_parser.add_argument("--model-type", help="Model type (text, vision, audio, llm)")
    
    # Compare quantization methods
    compare_parser = command_group.add_parser("compare", help="Compare quantization methods")
    compare_parser.add_argument("--model-path", required=True, help="Path to input model")
    compare_parser.add_argument("--output-dir", required=True, help="Directory for saving quantized models")
    compare_parser.add_argument("--model-type", help="Model type (text, vision, audio, llm)")
    compare_parser.add_argument("--methods", help="Comma-separated list of methods to compare")
    compare_parser.add_argument("--report-path", help="Path to save the comparison report")
    
    # Common options
    parser.add_argument("--db-path", help="Path to DuckDB database")
    parser.add_argument("--mock", action="store_true", help="Force mock mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set environment variables if needed
    if args.mock:
        os.environ["QUALCOMM_MOCK"] = "1"
    
    # Create quantization handler
    qquant = QualcommQuantization(db_path=args.db_path)
    
    # Check availability
    if not qquant.is_available():
        print("Error: Qualcomm AI Engine not available and mock mode disabled.")
        return 1
    
    # Process commands
    if args.command == "list":
        methods = qquant.list_quantization_methods()
        supported = qquant.get_supported_methods()
        
        print("\nAvailable Qualcomm AI Engine Quantization Methods:\n")
        for method, description in sorted(methods.items()):
            support_status = "✅ Supported" if supported.get(method, False) else "❌ Not supported"
            print(f"- {method}: {description} [{support_status}]")
            
        print(f"\nSDK Type: {qquant.qualcomm_handler.sdk_type or 'Unknown'}")
        print(f"SDK Version: {qquant.qualcomm_handler.sdk_version or 'Unknown'}")
        print(f"Mock Mode: {qquant.mock_mode}")
        
    elif args.command == "quantize":
        # Parse additional parameters if provided
        params = {}
        if args.params:
            try:
                params = json.loads(args.params)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON format in --params: {args.params}")
                return 1
        
        # Quantize model
        result = qquant.quantize_model(
            model_path=args.model_path,
            output_path=args.output_path,
            method=args.method,
            model_type=args.model_type,
            calibration_data=args.calibration_data,
            **params
        )
        
        # Print results
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1
            
        print(f"\nQuantization completed successfully:")
        print(f"- Input: {args.model_path}")
        print(f"- Output: {args.output_path}")
        print(f"- Method: {args.method}")
        print(f"- Status: {result.get('status', 'Unknown')}")
        
        if "size_reduction_ratio" in result:
            print(f"- Size Reduction: {result['size_reduction_ratio']:.2f}x")
            
        # Print power efficiency metrics
        if "power_efficiency_metrics" in result:
            metrics = result["power_efficiency_metrics"]
            print("\nEstimated Power Efficiency Metrics:")
            print(f"- Power Consumption: {metrics.get('power_consumption_mw', 0):.2f} mW")
            print(f"- Energy Efficiency: {metrics.get('energy_efficiency_items_per_joule', 0):.2f} items/joule")
            print(f"- Battery Impact: {metrics.get('battery_impact_percent_per_hour', 0):.2f}% per hour")
            print(f"- Power Reduction: {metrics.get('power_reduction_percent', 0):.2f}%")
            print(f"- Efficiency Improvement: {metrics.get('efficiency_improvement_percent', 0):.2f}%")
            print(f"- Thermal Reduction: {metrics.get('estimated_thermal_reduction_percent', 0):.2f}%")
            print(f"- Thermal Throttling Risk: {metrics.get('thermal_throttling_risk', 'Unknown')}")
            
    elif args.command == "benchmark":
        # Benchmark quantized model
        result = qquant.benchmark_quantized_model(
            model_path=args.model_path,
            model_type=args.model_type
        )
        
        # Print results
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1
            
        print(f"\nBenchmark completed successfully:")
        print(f"- Model: {args.model_path}")
        print(f"- Model Type: {args.model_type or 'Auto-detected'}")
        print(f"- Status: {result.get('status', 'Unknown')}")
        
        # Print performance metrics
        print("\nPerformance Metrics:")
        print(f"- Latency: {result.get('latency_ms', 0):.2f} ms")
        print(f"- Throughput: {result.get('throughput', 0):.2f} {result.get('throughput_units', 'items/second')}")
        
        # Print power metrics
        if "metrics" in result:
            metrics = result["metrics"]
            print("\nPower and Thermal Metrics:")
            print(f"- Power Consumption: {metrics.get('power_consumption_mw', 0):.2f} mW")
            print(f"- Average Power: {metrics.get('average_power_mw', 0):.2f} mW")
            print(f"- Peak Power: {metrics.get('peak_power_mw', 0):.2f} mW")
            print(f"- Temperature: {metrics.get('temperature_celsius', 0):.2f}°C")
            print(f"- Energy Efficiency: {metrics.get('energy_efficiency_items_per_joule', 0):.2f} items/joule")
            print(f"- Battery Impact: {metrics.get('battery_impact_percent_per_hour', 0):.2f}% per hour")
            print(f"- Thermal Throttling Detected: {metrics.get('thermal_throttling_detected', False)}")
            
    elif args.command == "compare":
        # Parse methods list if provided
        methods = None
        if args.methods:
            methods = [m.strip() for m in args.methods.split(",")]
            
        # Compare quantization methods
        result = qquant.compare_quantization_methods(
            model_path=args.model_path,
            output_dir=args.output_dir,
            model_type=args.model_type,
            methods=methods
        )
        
        # Print results
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1
            
        # Generate report
        report_path = args.report_path or os.path.join(args.output_dir, "quantization_comparison_report.md")
        report = qquant.generate_report(result, report_path)
        
        # Print summary
        summary = result.get("summary", {})
        recommendation = summary.get("overall_recommendation", {})
        
        print(f"\nQuantization Comparison Completed Successfully:")
        print(f"- Model: {args.model_path}")
        print(f"- Model Type: {args.model_type or 'Auto-detected'}")
        print(f"- Methods Compared: {', '.join(result.get('methods_compared', []))}")
        print(f"- Output Directory: {args.output_dir}")
        print(f"- Report: {report_path}")
        
        print("\nSummary of Recommendations:")
        print(f"- Overall Recommendation: {recommendation.get('final_recommendation', 'Unknown')}")
        print(f"- Rationale: {recommendation.get('rationale', 'Unknown')}")
        print(f"- Best Power Efficiency: {summary.get('best_power_efficiency', 'Unknown')}")
        print(f"- Best Energy Efficiency: {summary.get('best_energy_efficiency', 'Unknown')}")
        print(f"- Best Battery Life: {summary.get('best_battery_life', 'Unknown')}")
        print(f"- Best Size Reduction: {summary.get('best_size_reduction', 'Unknown')}")
        print(f"- Best Latency: {summary.get('best_latency', 'Unknown')}")
        print(f"- Best Throughput: {summary.get('best_throughput', 'Unknown')}")
        
    else:
        parser.print_help()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
