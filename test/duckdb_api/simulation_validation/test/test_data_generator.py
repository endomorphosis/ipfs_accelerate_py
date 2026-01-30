#!/usr/bin/env python3
"""
Test data generator for the Simulation Accuracy and Validation Framework.

This module provides utilities for generating realistic test data for simulation results,
hardware results, validation results, calibration records, and drift detection results.
It supports various data generation scenarios including:
- Baseline accurate simulation data
- Simulation data with controlled error rates
- Time series data with trends
- Data with seasonal patterns
- Outliers and anomalies
- Hardware-specific patterns

The generated data can be used for both testing and demonstration purposes.
"""

import os
import sys
import uuid
import random
import datetime
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path

# Add parent directory to path for module imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required data structures
from data.duckdb.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult,
    CalibrationRecord,
    DriftDetectionResult
)


class TestDataGenerator:
    """Generator for realistic test data for the Simulation Validation Framework."""
    
    # Standard metrics for different hardware types
    BASE_METRICS = {
        "gpu": {
            "throughput_items_per_second": {
                "rtx3080": 90.0,
                "rtx4090": 150.0,
                "a100": 180.0
            },
            "average_latency_ms": {
                "rtx3080": 17.0,
                "rtx4090": 10.0,
                "a100": 8.5
            },
            "peak_memory_mb": {
                "rtx3080": 2200,
                "rtx4090": 2800,
                "a100": 3500
            }
        },
        "cpu": {
            "throughput_items_per_second": {
                "intel_xeon": 40.0,
                "amd_epyc": 45.0,
                "intel_core_i9": 35.0
            },
            "average_latency_ms": {
                "intel_xeon": 35.0,
                "amd_epyc": 32.0,
                "intel_core_i9": 38.0
            },
            "peak_memory_mb": {
                "intel_xeon": 1800,
                "amd_epyc": 1700,
                "intel_core_i9": 1900
            }
        },
        "webgpu": {
            "throughput_items_per_second": {
                "chrome": 60.0,
                "firefox": 55.0,
                "safari": 50.0
            },
            "average_latency_ms": {
                "chrome": 25.0,
                "firefox": 27.0,
                "safari": 29.0
            },
            "peak_memory_mb": {
                "chrome": 1500,
                "firefox": 1600,
                "safari": 1700
            }
        }
    }
    
    # Standard model configurations
    MODEL_CONFIGS = {
        "bert-base-uncased": {
            "hidden_size": 768,
            "num_layers": 12,
            "batch_sizes": [1, 8, 16, 32],
            "precisions": ["fp32", "fp16", "int8"]
        },
        "vit-base-patch16-224": {
            "hidden_size": 768,
            "num_layers": 12,
            "batch_sizes": [1, 16, 32, 64],
            "precisions": ["fp32", "fp16", "int8"]
        },
        "whisper-small": {
            "hidden_size": 768,
            "num_layers": 12,
            "batch_sizes": [1, 4, 8, 16],
            "precisions": ["fp32", "fp16"]
        },
        "llama-7b": {
            "hidden_size": 4096,
            "num_layers": 32,
            "batch_sizes": [1, 2, 4, 8],
            "precisions": ["fp32", "fp16", "int8"]
        }
    }
    
    # Hardware details
    HARDWARE_DETAILS = {
        "gpu_rtx3080": {
            "name": "NVIDIA RTX 3080",
            "compute_capability": "8.6",
            "vram_gb": 10,
            "test_environment": {
                "os": "Linux",
                "cuda_version": "11.4",
                "driver_version": "470.82.01"
            }
        },
        "gpu_rtx4090": {
            "name": "NVIDIA RTX 4090",
            "compute_capability": "8.9",
            "vram_gb": 24,
            "test_environment": {
                "os": "Linux",
                "cuda_version": "12.1",
                "driver_version": "530.30.02"
            }
        },
        "gpu_a100": {
            "name": "NVIDIA A100",
            "compute_capability": "8.0",
            "vram_gb": 40,
            "test_environment": {
                "os": "Linux",
                "cuda_version": "11.8",
                "driver_version": "510.39.01"
            }
        },
        "cpu_intel_xeon": {
            "name": "Intel Xeon",
            "cores": 24,
            "threads": 48,
            "test_environment": {
                "os": "Linux",
                "memory_gb": 64,
                "compiler": "GCC 11.2"
            }
        },
        "cpu_amd_epyc": {
            "name": "AMD EPYC",
            "cores": 32,
            "threads": 64,
            "test_environment": {
                "os": "Linux",
                "memory_gb": 128,
                "compiler": "GCC 11.2"
            }
        },
        "cpu_intel_core_i9": {
            "name": "Intel Core i9",
            "cores": 16,
            "threads": 24,
            "test_environment": {
                "os": "Linux",
                "memory_gb": 32,
                "compiler": "GCC 11.2"
            }
        },
        "webgpu_chrome": {
            "name": "Chrome WebGPU",
            "browser_version": "120.0.6099.109",
            "test_environment": {
                "os": "Windows 10",
                "js_engine": "V8"
            }
        },
        "webgpu_firefox": {
            "name": "Firefox WebGPU",
            "browser_version": "113.0",
            "test_environment": {
                "os": "Windows 10",
                "js_engine": "SpiderMonkey"
            }
        },
        "webgpu_safari": {
            "name": "Safari WebGPU",
            "browser_version": "17.0",
            "test_environment": {
                "os": "macOS Sonoma",
                "js_engine": "JavaScriptCore"
            }
        }
    }
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the test data generator with an optional random seed.
        
        Args:
            seed: Optional random seed for reproducible data generation
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Keep track of generated data
        self.simulation_results = []
        self.hardware_results = []
        self.validation_results = []
        self.calibration_records = []
        self.drift_detection_results = []
    
    def _get_metric_noise(self, metric_name: str, error_level: str = "low") -> float:
        """Generate appropriate level of noise for a given metric.
        
        Args:
            metric_name: Name of the metric to generate noise for
            error_level: Level of error ("none", "low", "medium", "high")
            
        Returns:
            A multiplicative noise factor for the metric
        """
        if error_level == "none":
            return 1.0
        
        # Define noise levels
        noise_levels = {
            "low": {
                "throughput_items_per_second": (0.97, 1.03),
                "average_latency_ms": (0.95, 1.05),
                "peak_memory_mb": (0.96, 1.04)
            },
            "medium": {
                "throughput_items_per_second": (0.90, 1.10),
                "average_latency_ms": (0.85, 1.15),
                "peak_memory_mb": (0.88, 1.12)
            },
            "high": {
                "throughput_items_per_second": (0.75, 1.25),
                "average_latency_ms": (0.70, 1.30),
                "peak_memory_mb": (0.75, 1.25)
            }
        }
        
        # Use default if metric not specified
        if metric_name not in noise_levels[error_level]:
            return random.uniform(0.9, 1.1)
        
        # Return random value within range
        low, high = noise_levels[error_level][metric_name]
        return random.uniform(low, high)
    
    def _add_trend(self, base_value: float, day_idx: int, trend_type: str, strength: float) -> float:
        """Add a trend component to a time series value.
        
        Args:
            base_value: Base value to modify
            day_idx: Day index (0-based) for time series
            trend_type: Type of trend ("linear", "exponential", "step", "none")
            strength: Strength of the trend effect
            
        Returns:
            Modified value with trend applied
        """
        if trend_type == "none":
            return base_value
        
        if trend_type == "linear":
            return base_value * (1 + strength * 0.01 * day_idx)
        
        if trend_type == "exponential":
            return base_value * (1 + strength * 0.01) ** day_idx
        
        if trend_type == "step":
            step_point = 10  # Day at which step occurs
            if day_idx >= step_point:
                return base_value * (1 + strength * 0.1)
        
        return base_value
    
    def _add_seasonality(self, base_value: float, day_idx: int, pattern: str, amplitude: float) -> float:
        """Add a seasonal component to a time series value.
        
        Args:
            base_value: Base value to modify
            day_idx: Day index (0-based) for time series
            pattern: Type of seasonal pattern ("none", "weekly", "monthly", "sinusoidal")
            amplitude: Amplitude of the seasonal effect
            
        Returns:
            Modified value with seasonality applied
        """
        if pattern == "none":
            return base_value
        
        if pattern == "weekly":
            # Weekend effect (days 5 and 6 in a week)
            day_of_week = day_idx % 7
            if day_of_week >= 5:  # Weekend
                return base_value * (1 - amplitude * 0.1)
            return base_value
        
        if pattern == "monthly":
            # End of month effect
            day_of_month = day_idx % 30
            if day_of_month >= 27:  # End of month
                return base_value * (1 + amplitude * 0.1)
            return base_value
        
        if pattern == "sinusoidal":
            # Smooth sinusoidal pattern
            return base_value * (1 + amplitude * 0.1 * np.sin(2 * np.pi * day_idx / 14))
        
        return base_value
    
    def _add_outliers(self, base_value: float, outlier_prob: float, outlier_magnitude: float) -> float:
        """Add potential outliers to a value.
        
        Args:
            base_value: Base value to modify
            outlier_prob: Probability of an outlier (0-1)
            outlier_magnitude: Magnitude of the outlier effect
            
        Returns:
            Potentially modified value with outlier
        """
        if random.random() < outlier_prob:
            # Generate outlier
            direction = 1 if random.random() > 0.5 else -1
            return base_value * (1 + direction * outlier_magnitude)
        return base_value
    
    def generate_hardware_result(
        self,
        model_id: str, 
        hardware_id: str,
        batch_size: Optional[int] = None,
        precision: Optional[str] = None,
        timestamp: Optional[str] = None,
        time_index: Optional[int] = None,
        metrics_override: Optional[Dict[str, float]] = None
    ) -> HardwareResult:
        """Generate a realistic hardware result.
        
        Args:
            model_id: ID of the model
            hardware_id: ID of the hardware
            batch_size: Batch size used for the test
            precision: Precision used for the test
            timestamp: ISO format timestamp
            time_index: Index for time series data (for generating patterns)
            metrics_override: Optional dictionary to override default metrics
            
        Returns:
            HardwareResult object with realistic data
        """
        # Extract hardware type from ID
        hardware_type = hardware_id.split('_')[0]  # e.g., 'gpu' from 'gpu_rtx3080'
        hardware_model = "_".join(hardware_id.split('_')[1:])  # e.g., 'rtx3080'
        
        # Generate or use provided timestamp
        if timestamp is None:
            if time_index is not None:
                base_date = datetime.datetime(2025, 3, 1)
                current_date = base_date + datetime.timedelta(days=time_index)
                timestamp = current_date.isoformat()
            else:
                timestamp = datetime.datetime.now().isoformat()
        
        # Use appropriate batch size if not specified
        if batch_size is None:
            batch_sizes = self.MODEL_CONFIGS.get(model_id, {}).get("batch_sizes", [1, 8, 16, 32])
            batch_size = random.choice(batch_sizes)
        
        # Use appropriate precision if not specified
        if precision is None:
            precisions = self.MODEL_CONFIGS.get(model_id, {}).get("precisions", ["fp32", "fp16"])
            precision = random.choice(precisions)
        
        # Generate metrics
        metrics = {}
        
        for metric_name in ["throughput_items_per_second", "average_latency_ms", "peak_memory_mb"]:
            base_value = self.BASE_METRICS.get(hardware_type, {}).get(metric_name, {}).get(hardware_model, 0)
            
            if base_value == 0:
                # Use a reasonable default if not found
                if metric_name == "throughput_items_per_second":
                    base_value = 50.0
                elif metric_name == "average_latency_ms":
                    base_value = 25.0
                elif metric_name == "peak_memory_mb":
                    base_value = 2000
            
            # Adjust for batch size
            if metric_name == "throughput_items_per_second":
                # Throughput increases with batch size but not linearly
                batch_factor = (batch_size / 8) ** 0.8
                base_value *= batch_factor
            elif metric_name == "average_latency_ms":
                # Latency increases slightly with batch size
                batch_factor = 1 + 0.1 * np.log2(batch_size / 8) if batch_size > 8 else 1
                base_value *= batch_factor
            elif metric_name == "peak_memory_mb":
                # Memory increases with batch size almost linearly
                batch_factor = 0.5 + 0.5 * (batch_size / 8)
                base_value *= batch_factor
            
            # Adjust for precision
            if precision == "fp16":
                if metric_name == "throughput_items_per_second":
                    base_value *= 1.5  # Higher throughput with fp16
                elif metric_name == "average_latency_ms":
                    base_value *= 0.7  # Lower latency with fp16
                elif metric_name == "peak_memory_mb":
                    base_value *= 0.6  # Lower memory with fp16
            elif precision == "int8":
                if metric_name == "throughput_items_per_second":
                    base_value *= 2.2  # Even higher throughput with int8
                elif metric_name == "average_latency_ms":
                    base_value *= 0.5  # Even lower latency with int8
                elif metric_name == "peak_memory_mb":
                    base_value *= 0.35  # Even lower memory with int8
            
            # Add time-based patterns if time_index is provided
            if time_index is not None:
                # Add trend - slight improvement over time
                base_value = self._add_trend(base_value, time_index, "linear", 0.5)
                
                # Add weekly seasonality
                base_value = self._add_seasonality(base_value, time_index, "weekly", 0.2)
                
                # Add occasional outliers
                base_value = self._add_outliers(base_value, 0.05, 0.2)
            
            # Add some random variation
            base_value *= random.uniform(0.98, 1.02)
            
            # Round to appropriate precision
            if metric_name == "throughput_items_per_second" or metric_name == "average_latency_ms":
                base_value = round(base_value, 2)
            else:
                base_value = round(base_value)
            
            metrics[metric_name] = base_value
        
        # Override with provided metrics if any
        if metrics_override:
            for key, value in metrics_override.items():
                metrics[key] = value
        
        # Get hardware details
        hardware_details = self.HARDWARE_DETAILS.get(hardware_id, {})
        
        # Create hardware result
        result = HardwareResult(
            model_id=model_id,
            hardware_id=hardware_id,
            batch_size=batch_size,
            precision=precision,
            timestamp=timestamp,
            metrics=metrics,
            hardware_details=hardware_details.get("name", {}),
            test_environment=hardware_details.get("test_environment", {})
        )
        
        # Store result
        self.hardware_results.append(result)
        
        return result
    
    def generate_simulation_result(
        self,
        model_id: str,
        hardware_id: str, 
        batch_size: Optional[int] = None,
        precision: Optional[str] = None,
        timestamp: Optional[str] = None,
        time_index: Optional[int] = None,
        simulation_version: str = "sim_v1.0",
        error_level: str = "low",
        metrics_override: Optional[Dict[str, float]] = None,
        matching_hardware_result: Optional[HardwareResult] = None
    ) -> SimulationResult:
        """Generate a realistic simulation result, optionally matched to a hardware result.
        
        Args:
            model_id: ID of the model
            hardware_id: ID of the hardware being simulated
            batch_size: Batch size used for the simulation
            precision: Precision used for the simulation
            timestamp: ISO format timestamp
            time_index: Index for time series data (for generating patterns)
            simulation_version: Version of the simulation software
            error_level: Level of simulation error ("none", "low", "medium", "high")
            metrics_override: Optional dictionary to override default metrics
            matching_hardware_result: Optional matching hardware result to base this on
            
        Returns:
            SimulationResult object with realistic data
        """
        # Use matching hardware result if provided
        if matching_hardware_result:
            hardware_id = matching_hardware_result.hardware_id
            batch_size = matching_hardware_result.batch_size
            precision = matching_hardware_result.precision
            timestamp = matching_hardware_result.timestamp
            
            # Generate simulation metrics based on hardware metrics with appropriate error
            metrics = {}
            for metric_name, hw_value in matching_hardware_result.metrics.items():
                noise_factor = self._get_metric_noise(metric_name, error_level)
                metrics[metric_name] = round(hw_value * noise_factor, 2)
        else:
            # Generate a hardware result to use as a base
            hw_result = self.generate_hardware_result(
                model_id=model_id,
                hardware_id=hardware_id,
                batch_size=batch_size,
                precision=precision,
                timestamp=timestamp,
                time_index=time_index
            )
            
            # Generate simulation metrics based on hardware metrics with appropriate error
            metrics = {}
            for metric_name, hw_value in hw_result.metrics.items():
                noise_factor = self._get_metric_noise(metric_name, error_level)
                metrics[metric_name] = round(hw_value * noise_factor, 2)
        
        # Override with provided metrics if any
        if metrics_override:
            for key, value in metrics_override.items():
                metrics[key] = value
        
        # Extract hardware type for simulation parameters
        hardware_type = hardware_id.split('_')[0]
        
        # Create simulation parameters
        simulation_params = {
            "model_params": self.MODEL_CONFIGS.get(model_id, {})
        }
        
        # Add hardware-specific parameters
        if hardware_type == "gpu":
            simulation_params["hardware_params"] = {
                "gpu_compute_capability": self.HARDWARE_DETAILS.get(hardware_id, {}).get("compute_capability", "8.0"),
                "gpu_memory": self.HARDWARE_DETAILS.get(hardware_id, {}).get("vram_gb", 8) * 1024  # Convert to MB
            }
        elif hardware_type == "cpu":
            simulation_params["hardware_params"] = {
                "cpu_cores": self.HARDWARE_DETAILS.get(hardware_id, {}).get("cores", 16),
                "cpu_threads": self.HARDWARE_DETAILS.get(hardware_id, {}).get("threads", 32)
            }
        elif hardware_type == "webgpu":
            simulation_params["hardware_params"] = {
                "browser": hardware_id.split('_')[1],
                "js_engine": self.HARDWARE_DETAILS.get(hardware_id, {}).get("test_environment", {}).get("js_engine", "V8")
            }
        
        # Create simulation result
        result = SimulationResult(
            model_id=model_id,
            hardware_id=hardware_id,
            batch_size=batch_size,
            precision=precision,
            timestamp=timestamp,
            simulation_version=simulation_version,
            metrics=metrics,
            simulation_params=simulation_params
        )
        
        # Store result
        self.simulation_results.append(result)
        
        return result
    
    def generate_validation_result(
        self,
        simulation_result: Optional[SimulationResult] = None,
        hardware_result: Optional[HardwareResult] = None,
        validation_timestamp: Optional[str] = None,
        validation_version: str = "v1.0"
    ) -> ValidationResult:
        """Generate a validation result comparing simulation and hardware results.
        
        Args:
            simulation_result: SimulationResult to validate
            hardware_result: HardwareResult to compare against
            validation_timestamp: ISO format timestamp for validation
            validation_version: Version of validation software
            
        Returns:
            ValidationResult object with comparison metrics
        """
        # Generate matching sim and hardware results if not provided
        if simulation_result is None and hardware_result is None:
            # Generate hardware result
            model_id = random.choice(list(self.MODEL_CONFIGS.keys()))
            hardware_id = random.choice(list(self.HARDWARE_DETAILS.keys()))
            
            hardware_result = self.generate_hardware_result(
                model_id=model_id,
                hardware_id=hardware_id
            )
            
            # Generate matching simulation result
            simulation_result = self.generate_simulation_result(
                model_id=model_id,
                hardware_id=hardware_id,
                matching_hardware_result=hardware_result
            )
        elif simulation_result is None:
            # Generate matching simulation result
            simulation_result = self.generate_simulation_result(
                model_id=hardware_result.model_id,
                hardware_id=hardware_result.hardware_id,
                matching_hardware_result=hardware_result
            )
        elif hardware_result is None:
            # Generate matching hardware result
            hardware_result = self.generate_hardware_result(
                model_id=simulation_result.model_id,
                hardware_id=simulation_result.hardware_id,
                batch_size=simulation_result.batch_size,
                precision=simulation_result.precision,
                timestamp=simulation_result.timestamp
            )
        
        # Use hardware timestamp if validation timestamp not provided
        if validation_timestamp is None:
            validation_timestamp = hardware_result.timestamp
        
        # Create metrics comparison
        metrics_comparison = {}
        for metric in ["throughput_items_per_second", "average_latency_ms", "peak_memory_mb"]:
            if metric in hardware_result.metrics and metric in simulation_result.metrics:
                hw_value = hardware_result.metrics[metric]
                sim_value = simulation_result.metrics[metric]
                abs_error = abs(sim_value - hw_value)
                rel_error = abs_error / hw_value if hw_value != 0 else 0
                mape = rel_error * 100
                
                metrics_comparison[metric] = {
                    "simulation_value": sim_value,
                    "hardware_value": hw_value,
                    "absolute_error": round(abs_error, 3),
                    "relative_error": round(rel_error, 4),
                    "mape": round(mape, 2)
                }
        
        # Calculate overall MAPE
        mape_values = [comparison["mape"] for comparison in metrics_comparison.values()]
        overall_mape = sum(mape_values) / len(mape_values) if mape_values else 0
        
        # Create validation result
        result = ValidationResult(
            simulation_result=simulation_result,
            hardware_result=hardware_result,
            metrics_comparison=metrics_comparison,
            validation_timestamp=validation_timestamp,
            validation_version=validation_version,
            overall_accuracy_score=round(overall_mape, 2)
        )
        
        # Store result
        self.validation_results.append(result)
        
        return result
    
    def generate_calibration_record(
        self,
        id: Optional[str] = None,
        hardware_type: Optional[str] = None,
        model_type: Optional[str] = None,
        validation_results_before: Optional[List[ValidationResult]] = None,
        validation_results_after: Optional[List[ValidationResult]] = None,
        timestamp: Optional[str] = None,
        calibration_version: str = "v1.0"
    ) -> CalibrationRecord:
        """Generate a calibration record with before/after validation results.
        
        Args:
            id: Optional ID for the calibration record
            hardware_type: Type of hardware (e.g., "gpu_rtx3080")
            model_type: Type of model (e.g., "bert-base-uncased")
            validation_results_before: Validation results before calibration
            validation_results_after: Validation results after calibration
            timestamp: ISO format timestamp for calibration
            calibration_version: Version of calibration software
            
        Returns:
            CalibrationRecord object with comparison metrics
        """
        # Generate ID if not provided
        if id is None:
            id = f"cal_{uuid.uuid4().hex[:8]}"
        
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
        
        # Select hardware and model if not provided
        if hardware_type is None:
            hardware_type = random.choice(list(self.HARDWARE_DETAILS.keys()))
        
        if model_type is None:
            model_type = random.choice(list(self.MODEL_CONFIGS.keys()))
        
        # Generate validation results before calibration if not provided
        if validation_results_before is None:
            validation_results_before = []
            # Generate 3 results with medium errors
            for i in range(3):
                hw_result = self.generate_hardware_result(
                    model_id=model_type,
                    hardware_id=hardware_type,
                    time_index=i
                )
                
                sim_result = self.generate_simulation_result(
                    model_id=model_type,
                    hardware_id=hardware_type,
                    matching_hardware_result=hw_result,
                    error_level="medium"
                )
                
                val_result = self.generate_validation_result(
                    simulation_result=sim_result,
                    hardware_result=hw_result
                )
                
                validation_results_before.append(val_result)
        
        # Create correction factors based on before results
        correction_factors = {}
        for metric in ["throughput_items_per_second", "average_latency_ms", "peak_memory_mb"]:
            # Calculate average ratio of hardware to simulation values
            ratios = []
            for val_result in validation_results_before:
                comparison = val_result.metrics_comparison.get(metric, {})
                if "hardware_value" in comparison and "simulation_value" in comparison:
                    hw_val = comparison["hardware_value"]
                    sim_val = comparison["simulation_value"]
                    if sim_val != 0:
                        ratios.append(hw_val / sim_val)
            
            # Set correction factor
            if ratios:
                correction_factors[metric] = round(sum(ratios) / len(ratios), 3)
            else:
                correction_factors[metric] = 1.0
        
        # Generate validation results after calibration if not provided
        if validation_results_after is None:
            validation_results_after = []
            # Generate 3 results with low errors (improved after calibration)
            for i in range(3):
                hw_result = self.generate_hardware_result(
                    model_id=model_type,
                    hardware_id=hardware_type,
                    time_index=10 + i  # Later in time
                )
                
                # Create simulation result with metrics adjusted by correction factors
                sim_metrics = {}
                for metric, value in hw_result.metrics.items():
                    if metric in correction_factors:
                        # Apply correction and add small error
                        sim_metrics[metric] = value / correction_factors[metric] * random.uniform(0.97, 1.03)
                
                sim_result = self.generate_simulation_result(
                    model_id=model_type,
                    hardware_id=hardware_type,
                    matching_hardware_result=hw_result,
                    error_level="low",
                    metrics_override=sim_metrics
                )
                
                val_result = self.generate_validation_result(
                    simulation_result=sim_result,
                    hardware_result=hw_result
                )
                
                validation_results_after.append(val_result)
        
        # Calculate improvement metrics
        improvement_metrics = {}
        
        # Calculate per-metric improvements
        for metric in ["throughput_items_per_second", "average_latency_ms", "peak_memory_mb"]:
            before_mapes = []
            after_mapes = []
            
            for val_result in validation_results_before:
                if metric in val_result.metrics_comparison:
                    before_mapes.append(val_result.metrics_comparison[metric]["mape"])
            
            for val_result in validation_results_after:
                if metric in val_result.metrics_comparison:
                    after_mapes.append(val_result.metrics_comparison[metric]["mape"])
            
            if before_mapes and after_mapes:
                before_avg = sum(before_mapes) / len(before_mapes)
                after_avg = sum(after_mapes) / len(after_mapes)
                abs_improvement = before_avg - after_avg
                rel_improvement_pct = (abs_improvement / before_avg) * 100 if before_avg > 0 else 0
                
                improvement_metrics[metric] = {
                    "before_mape": round(before_avg, 2),
                    "after_mape": round(after_avg, 2),
                    "absolute_improvement": round(abs_improvement, 2),
                    "relative_improvement_pct": round(rel_improvement_pct, 2)
                }
        
        # Calculate overall improvement
        all_before_mapes = []
        all_after_mapes = []
        
        for val_result in validation_results_before:
            all_before_mapes.append(val_result.overall_accuracy_score)
        
        for val_result in validation_results_after:
            all_after_mapes.append(val_result.overall_accuracy_score)
        
        if all_before_mapes and all_after_mapes:
            before_avg = sum(all_before_mapes) / len(all_before_mapes)
            after_avg = sum(all_after_mapes) / len(all_after_mapes)
            abs_improvement = before_avg - after_avg
            rel_improvement_pct = (abs_improvement / before_avg) * 100 if before_avg > 0 else 0
            
            improvement_metrics["overall"] = {
                "before_mape": round(before_avg, 2),
                "after_mape": round(after_avg, 2),
                "absolute_improvement": round(abs_improvement, 2),
                "relative_improvement_pct": round(rel_improvement_pct, 2)
            }
        
        # Create calibration record
        result = CalibrationRecord(
            id=id,
            timestamp=timestamp,
            hardware_type=hardware_type,
            model_type=model_type,
            previous_parameters={
                "correction_factors": {k: 1.0 for k in correction_factors}
            },
            updated_parameters={
                "correction_factors": correction_factors
            },
            validation_results_before=validation_results_before,
            validation_results_after=validation_results_after,
            improvement_metrics=improvement_metrics,
            calibration_version=calibration_version
        )
        
        # Store result
        self.calibration_records.append(result)
        
        return result
    
    def generate_drift_detection_result(
        self,
        id: Optional[str] = None,
        hardware_type: Optional[str] = None,
        model_type: Optional[str] = None,
        is_significant: Optional[bool] = None,
        historical_window_start: Optional[str] = None,
        historical_window_end: Optional[str] = None,
        new_window_start: Optional[str] = None,
        new_window_end: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> DriftDetectionResult:
        """Generate a drift detection result.
        
        Args:
            id: Optional ID for the drift detection record
            hardware_type: Type of hardware
            model_type: Type of model
            is_significant: Whether drift is significant
            historical_window_start: Start of historical window
            historical_window_end: End of historical window
            new_window_start: Start of new window
            new_window_end: End of new window
            timestamp: ISO format timestamp for drift detection
            
        Returns:
            DriftDetectionResult object with drift metrics
        """
        # Generate ID if not provided
        if id is None:
            id = f"drift_{uuid.uuid4().hex[:8]}"
        
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
        
        # Select hardware and model if not provided
        if hardware_type is None:
            hardware_type = random.choice(list(self.HARDWARE_DETAILS.keys()))
        
        if model_type is None:
            model_type = random.choice(list(self.MODEL_CONFIGS.keys()))
        
        # Set significance if not provided
        if is_significant is None:
            is_significant = random.choice([True, False])
        
        # Generate window dates if not provided
        if historical_window_start is None:
            base_date = datetime.datetime(2025, 3, 1)
            historical_window_start = base_date.isoformat()
        
        if historical_window_end is None:
            base_date = datetime.datetime.fromisoformat(historical_window_start)
            historical_window_end = (base_date + datetime.timedelta(days=10)).isoformat()
        
        if new_window_start is None:
            historical_end_date = datetime.datetime.fromisoformat(historical_window_end)
            new_window_start = (historical_end_date + datetime.timedelta(days=1)).isoformat()
        
        if new_window_end is None:
            new_start_date = datetime.datetime.fromisoformat(new_window_start)
            new_window_end = (new_start_date + datetime.timedelta(days=10)).isoformat()
        
        # Generate drift metrics
        drift_metrics = {}
        for metric in ["throughput_items_per_second", "average_latency_ms", "peak_memory_mb"]:
            # Determine if drift is detected for this metric
            if is_significant:
                # At least one metric should show drift if significant
                drift_detected = random.random() < 0.7
            else:
                # Less likely to show drift if not significant
                drift_detected = random.random() < 0.3
            
            # Generate p-value
            if drift_detected:
                p_value = random.uniform(0.001, 0.049)  # Below 0.05 threshold
                mean_change_pct = random.uniform(10.1, 25.0)  # Above 10% threshold
            else:
                p_value = random.uniform(0.051, 0.3)  # Above 0.05 threshold
                mean_change_pct = random.uniform(2.0, 9.9)  # Below 10% threshold
            
            drift_metrics[metric] = {
                "p_value": round(p_value, 3),
                "drift_detected": drift_detected,
                "mean_change_pct": round(mean_change_pct, 1)
            }
        
        # Create drift detection result
        result = DriftDetectionResult(
            id=id,
            timestamp=timestamp,
            hardware_type=hardware_type,
            model_type=model_type,
            drift_metrics=drift_metrics,
            is_significant=is_significant,
            historical_window_start=historical_window_start,
            historical_window_end=historical_window_end,
            new_window_start=new_window_start,
            new_window_end=new_window_end,
            thresholds_used={
                "p_value": 0.05,
                "mean_change_pct": 10.0
            }
        )
        
        # Store result
        self.drift_detection_results.append(result)
        
        return result
    
    def generate_time_series_data(
        self,
        model_id: str,
        hardware_id: str,
        num_days: int = 30,
        start_date: Optional[datetime.datetime] = None,
        trend_type: str = "linear",
        trend_strength: float = 1.0,
        seasonality_pattern: str = "weekly",
        seasonality_amplitude: float = 0.5,
        outlier_probability: float = 0.05,
        outlier_magnitude: float = 0.2,
        error_level: str = "low",
        error_growth: bool = False
    ) -> Tuple[List[HardwareResult], List[SimulationResult], List[ValidationResult]]:
        """Generate a time series of matching hardware and simulation results.
        
        Args:
            model_id: ID of the model
            hardware_id: ID of the hardware
            num_days: Number of days in the time series
            start_date: Starting date for the time series
            trend_type: Type of trend ("linear", "exponential", "step", "none")
            trend_strength: Strength of the trend effect
            seasonality_pattern: Type of seasonal pattern ("none", "weekly", "monthly", "sinusoidal")
            seasonality_amplitude: Amplitude of the seasonal effect
            outlier_probability: Probability of an outlier (0-1)
            outlier_magnitude: Magnitude of the outlier effect
            error_level: Starting error level for simulation ("none", "low", "medium", "high")
            error_growth: Whether simulation error grows over time
            
        Returns:
            Tuple of (hardware_results, simulation_results, validation_results)
        """
        # Use provided start date or default to March 1, 2025
        if start_date is None:
            start_date = datetime.datetime(2025, 3, 1)
        
        hw_results = []
        sim_results = []
        val_results = []
        
        for day in range(num_days):
            # Calculate current date
            current_date = start_date + datetime.timedelta(days=day)
            
            # Generate hardware result with time patterns
            hw_result = self.generate_hardware_result(
                model_id=model_id,
                hardware_id=hardware_id,
                timestamp=current_date.isoformat(),
                time_index=day
            )
            hw_results.append(hw_result)
            
            # Determine error level for this day
            current_error_level = error_level
            if error_growth:
                # Gradually increase error level over time
                if day > num_days * 0.7:  # Last 30% of days
                    current_error_level = "high"
                elif day > num_days * 0.4:  # Middle 30% of days
                    current_error_level = "medium"
            
            # Generate matching simulation result with appropriate error
            sim_result = self.generate_simulation_result(
                model_id=model_id,
                hardware_id=hardware_id,
                matching_hardware_result=hw_result,
                error_level=current_error_level
            )
            sim_results.append(sim_result)
            
            # Generate validation result
            val_result = self.generate_validation_result(
                simulation_result=sim_result,
                hardware_result=hw_result
            )
            val_results.append(val_result)
        
        return hw_results, sim_results, val_results
    
    def generate_drift_scenario(
        self,
        model_id: str,
        hardware_id: str,
        num_days_before: int = 15,
        num_days_after: int = 15,
        start_date: Optional[datetime.datetime] = None,
        drift_magnitude: float = 0.2,
        drift_direction: str = "positive",
        affected_metrics: Optional[List[str]] = None
    ) -> Tuple[List[HardwareResult], List[SimulationResult], List[ValidationResult], DriftDetectionResult]:
        """Generate a scenario with a drift in simulation accuracy.
        
        Args:
            model_id: ID of the model
            hardware_id: ID of the hardware
            num_days_before: Number of days before drift
            num_days_after: Number of days after drift
            start_date: Starting date for the time series
            drift_magnitude: Magnitude of the drift (0-1)
            drift_direction: Direction of drift ("positive" or "negative")
            affected_metrics: List of metrics affected by drift
            
        Returns:
            Tuple of (hardware_results, simulation_results, validation_results, drift_detection)
        """
        # Use provided start date or default to March 1, 2025
        if start_date is None:
            start_date = datetime.datetime(2025, 3, 1)
        
        # Default affected metrics if not provided
        if affected_metrics is None:
            affected_metrics = ["throughput_items_per_second", "peak_memory_mb"]
        
        # Generate before-drift data with good accuracy
        hw_before, sim_before, val_before = self.generate_time_series_data(
            model_id=model_id,
            hardware_id=hardware_id,
            num_days=num_days_before,
            start_date=start_date,
            error_level="low"
        )
        
        # Calculate drift date
        drift_date = start_date + datetime.timedelta(days=num_days_before)
        
        # Generate after-drift data with drift in simulation
        hw_after = []
        sim_after = []
        val_after = []
        
        for day in range(num_days_after):
            current_date = drift_date + datetime.timedelta(days=day)
            
            # Generate hardware result
            hw_result = self.generate_hardware_result(
                model_id=model_id,
                hardware_id=hardware_id,
                timestamp=current_date.isoformat(),
                time_index=num_days_before + day
            )
            hw_after.append(hw_result)
            
            # Generate simulation result with drift
            # Start with base simulation with low error
            sim_result = self.generate_simulation_result(
                model_id=model_id,
                hardware_id=hardware_id,
                matching_hardware_result=hw_result,
                error_level="low"
            )
            
            # Apply drift to affected metrics
            for metric in affected_metrics:
                if metric in sim_result.metrics:
                    original_value = sim_result.metrics[metric]
                    
                    # Calculate drift factor
                    if drift_direction == "positive":
                        drift_factor = 1 + drift_magnitude
                    else:
                        drift_factor = 1 - drift_magnitude
                    
                    # Apply drift
                    sim_result.metrics[metric] = round(original_value * drift_factor, 2)
            
            sim_after.append(sim_result)
            
            # Generate validation result
            val_result = self.generate_validation_result(
                simulation_result=sim_result,
                hardware_result=hw_result
            )
            val_after.append(val_result)
        
        # Combine data
        hw_results = hw_before + hw_after
        sim_results = sim_before + sim_after
        val_results = val_before + val_after
        
        # Generate drift detection result
        drift_detection = self.generate_drift_detection_result(
            hardware_type=hardware_id,
            model_type=model_id,
            is_significant=True,
            historical_window_start=start_date.isoformat(),
            historical_window_end=(drift_date - datetime.timedelta(days=1)).isoformat(),
            new_window_start=drift_date.isoformat(),
            new_window_end=(drift_date + datetime.timedelta(days=num_days_after-1)).isoformat(),
            timestamp=(drift_date + datetime.timedelta(days=num_days_after)).isoformat()
        )
        
        return hw_results, sim_results, val_results, drift_detection
    
    def generate_calibration_scenario(
        self,
        model_id: str,
        hardware_id: str,
        num_days_before: int = 10,
        num_days_after: int = 10,
        start_date: Optional[datetime.datetime] = None
    ) -> Tuple[List[HardwareResult], List[SimulationResult], List[ValidationResult], CalibrationRecord]:
        """Generate a scenario with a calibration that improves simulation accuracy.
        
        Args:
            model_id: ID of the model
            hardware_id: ID of the hardware
            num_days_before: Number of days before calibration
            num_days_after: Number of days after calibration
            start_date: Starting date for the time series
            
        Returns:
            Tuple of (hardware_results, simulation_results, validation_results, calibration_record)
        """
        # Use provided start date or default to March 1, 2025
        if start_date is None:
            start_date = datetime.datetime(2025, 3, 1)
        
        # Generate before-calibration data with medium error
        hw_before, sim_before, val_before = self.generate_time_series_data(
            model_id=model_id,
            hardware_id=hardware_id,
            num_days=num_days_before,
            start_date=start_date,
            error_level="medium"
        )
        
        # Calculate calibration date
        calibration_date = start_date + datetime.timedelta(days=num_days_before)
        
        # Generate after-calibration data with improved accuracy
        hw_after, sim_after, val_after = self.generate_time_series_data(
            model_id=model_id,
            hardware_id=hardware_id,
            num_days=num_days_after,
            start_date=calibration_date,
            error_level="low"
        )
        
        # Create calibration record
        calibration_record = self.generate_calibration_record(
            hardware_type=hardware_id,
            model_type=model_id,
            validation_results_before=val_before,
            validation_results_after=val_after,
            timestamp=calibration_date.isoformat()
        )
        
        # Combine data
        hw_results = hw_before + hw_after
        sim_results = sim_before + sim_after
        val_results = val_before + val_after
        
        return hw_results, sim_results, val_results, calibration_record
    
    def generate_complete_dataset(
        self,
        num_models: int = 3,
        num_hardware_types: int = 3,
        days_per_series: int = 30,
        include_calibrations: bool = True,
        include_drifts: bool = True,
        random_seed: Optional[int] = None
    ) -> Dict[str, List]:
        """Generate a complete dataset with multiple models, hardware types, and scenarios.
        
        Args:
            num_models: Number of different models to include
            num_hardware_types: Number of different hardware types to include
            days_per_series: Number of days per time series
            include_calibrations: Whether to include calibration scenarios
            include_drifts: Whether to include drift scenarios
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with lists of all generated results
        """
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Reset data storage
        self.simulation_results = []
        self.hardware_results = []
        self.validation_results = []
        self.calibration_records = []
        self.drift_detection_results = []
        
        # Select models and hardware types
        available_models = list(self.MODEL_CONFIGS.keys())
        available_hardware = list(self.HARDWARE_DETAILS.keys())
        
        selected_models = random.sample(available_models, min(num_models, len(available_models)))
        selected_hardware = random.sample(available_hardware, min(num_hardware_types, len(available_hardware)))
        
        # Generate baseline data for all combinations
        for model_id in selected_models:
            for hardware_id in selected_hardware:
                # Generate time series with good accuracy
                start_date = datetime.datetime(2025, 3, 1)
                self.generate_time_series_data(
                    model_id=model_id,
                    hardware_id=hardware_id,
                    num_days=days_per_series,
                    start_date=start_date
                )
        
        # Generate calibration scenarios (one per hardware type)
        if include_calibrations:
            for hardware_id in selected_hardware:
                model_id = random.choice(selected_models)
                start_date = datetime.datetime(2025, 4, 1)  # Later date
                
                self.generate_calibration_scenario(
                    model_id=model_id,
                    hardware_id=hardware_id,
                    start_date=start_date
                )
        
        # Generate drift scenarios (one per model)
        if include_drifts:
            for model_id in selected_models:
                hardware_id = random.choice(selected_hardware)
                start_date = datetime.datetime(2025, 5, 1)  # Even later date
                
                self.generate_drift_scenario(
                    model_id=model_id,
                    hardware_id=hardware_id,
                    start_date=start_date
                )
        
        # Return all generated data
        return {
            "simulation_results": self.simulation_results,
            "hardware_results": self.hardware_results,
            "validation_results": self.validation_results,
            "calibration_records": self.calibration_records,
            "drift_detection_results": self.drift_detection_results
        }
    
    def save_dataset_to_json(self, dataset: Dict[str, List], output_path: str) -> str:
        """Save a generated dataset to a JSON file.
        
        Args:
            dataset: Generated dataset dictionary
            output_path: Path to save the JSON file
            
        Returns:
            Path to the saved JSON file
        """
        # Prepare JSON-serializable version of the dataset
        serializable_dataset = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metadata": {
                "simulation_results_count": len(dataset["simulation_results"]),
                "hardware_results_count": len(dataset["hardware_results"]),
                "validation_results_count": len(dataset["validation_results"]),
                "calibration_records_count": len(dataset["calibration_records"]),
                "drift_detection_results_count": len(dataset["drift_detection_results"])
            },
            "data": {
                "simulation_results": [result.to_dict() for result in dataset["simulation_results"]],
                "hardware_results": [result.to_dict() for result in dataset["hardware_results"]],
                "validation_results": [result.to_dict() for result in dataset["validation_results"]],
                "calibration_records": [record.to_dict() for record in dataset["calibration_records"]],
                "drift_detection_results": [result.to_dict() for result in dataset["drift_detection_results"]]
            }
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(serializable_dataset, f, indent=2)
        
        return output_path