#!/usr/bin/env python
"""
Qualcomm Neural Network (QNN) hardware detection and support module.

This module provides capabilities for:
1. Detecting Qualcomm AI Engine availability
2. Analyzing device specifications
3. Testing power and thermal monitoring for edge devices
4. Optimizing model deployment for mobile devices

Implementation progress: 60% complete (March 2025)
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock QNN SDK import - in a real implementation, this would import the actual SDK
class MockQNNSDK:
    def __init__(self, version: str = "2.10"):
        self.version = version
        self.available = True
        self.devices = [
            {"name": "Snapdragon 8 Gen 3", "compute_units": 16, "cores": 8, "memory": 8192, "dtype_support": ["fp32", "fp16", "int8", "int4"]},
            {"name": "Snapdragon 8 Gen 2", "compute_units": 12, "cores": 8, "memory": 6144, "dtype_support": ["fp32", "fp16", "int8"]},
            {"name": "Snapdragon 7+ Gen 2", "compute_units": 8, "cores": 8, "memory": 4096, "dtype_support": ["fp32", "fp16", "int8"]}
        ]
        self.current_device = None
        logger.info(f"Initialized QNN SDK version {version}")
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """List all available QNN devices"""
        return self.devices
    
    def select_device(self, device_name: str) -> bool:
        """Select a specific device for operations"""
        for device in self.devices:
            if device["name"] == device_name:
                self.current_device = device
                logger.info(f"Selected device: {device_name}")
                return True
        logger.error(f"Device not found: {device_name}")
        return False
    
    def get_device_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently selected device"""
        return self.current_device
    
    def test_device(self) -> Dict[str, Any]:
        """Run a basic test on the current device"""
        if not self.current_device:
            return {"success": False, "error": "No device selected"}
        
        # Simulate a basic test
        return {
            "success": True,
            "device": self.current_device["name"],
            "test_time_ms": 102.3,
            "operations_per_second": 5.2e9
        }

# Use mock for now, will be replaced with actual SDK in production
try:
    # Try to import actual QNN SDK if available
    pass  # In real implementation: from qnn_sdk import QNNSDK
    qnn_sdk = MockQNNSDK()  # Would be real SDK in production
    QNN_AVAILABLE = True
except ImportError:
    qnn_sdk = MockQNNSDK()  # Use mock for development/testing
    QNN_AVAILABLE = True  # Mock always available


class QNNCapabilityDetector:
    """Detects and validates QNN hardware capabilities"""
    
    def __init__(self):
        self.sdk = qnn_sdk
        self.devices = self.sdk.list_devices()
        self.selected_device = None
        self.default_model_path = "models/test_model.onnx"
        self.capability_cache = {}
        
    def is_available(self) -> bool:
        """Check if QNN SDK and hardware are available"""
        return QNN_AVAILABLE and len(self.devices) > 0
    
    def get_devices(self) -> List[Dict[str, Any]]:
        """Get list of available devices"""
        return self.devices
    
    def select_device(self, device_name: str = None) -> bool:
        """Select a specific device by name, or first available if None"""
        if device_name:
            if self.sdk.select_device(device_name):
                self.selected_device = self.sdk.get_device_info()
                return True
            return False
        
        # Select first available device if none specified
        if self.devices:
            if self.sdk.select_device(self.devices[0]["name"]):
                self.selected_device = self.sdk.get_device_info()
                return True
        return False
    
    def get_capability_summary(self) -> Dict[str, Any]:
        """Get a summary of capabilities for the selected device"""
        if not self.selected_device:
            if not self.select_device():
                return {"error": "No device available"}
        
        # Return cached results if available
        if "capability_summary" in self.capability_cache:
            return self.capability_cache["capability_summary"]
        
        # Generate capability summary
        summary = {
            "device_name": self.selected_device["name"],
            "compute_units": self.selected_device["compute_units"],
            "memory_mb": self.selected_device["memory"],
            "precision_support": self.selected_device["dtype_support"],
            "sdk_version": self.sdk.version,
            "recommended_models": self._get_recommended_models(),
            "estimated_performance": self._estimate_performance()
        }
        
        self.capability_cache["capability_summary"] = summary
        return summary
    
    def _get_recommended_models(self) -> List[str]:
        """Get list of recommended models for this device"""
        if not self.selected_device:
            return []
        
        # Base recommendations on device capabilities
        memory_mb = self.selected_device["memory"]
        precision = self.selected_device["dtype_support"]
        
        # Simple recommendation logic based on memory and precision
        recommendations = []
        
        # All devices can run these models
        recommendations.extend([
            "bert-tiny", 
            "bert-mini",
            "distilbert-base-uncased",
            "mobilevit-small",
            "whisper-tiny"
        ])
        
        # For devices with >4GB memory
        if memory_mb >= 4096:
            recommendations.extend([
                "bert-base-uncased",
                "t5-small",
                "vit-base",
                "whisper-small"
            ])
        
        # For high-end devices with >6GB memory
        if memory_mb >= 6144:
            recommendations.extend([
                "opt-350m",
                "llama-7b-4bit",  # Quantized version
                "t5-base",
                "clip-vit-base"
            ])
        
        # For devices with int4 support (advanced quantization)
        if "int4" in precision:
            recommendations.extend([
                "llama-7b-int4",
                "llama-13b-int4",
                "vicuna-7b-int4"
            ])
            
        return recommendations
    
    def _estimate_performance(self) -> Dict[str, float]:
        """Estimate performance for common model types"""
        if not self.selected_device:
            return {}
        
        # Simple linear model based on compute units and memory
        compute_units = self.selected_device["compute_units"]
        memory_mb = self.selected_device["memory"]
        
        # Coefficients derived from benchmarks (would be calibrated with real data)
        cu_factor = 0.8
        mem_factor = 0.2
        base_performance = {
            "bert_base_latency_ms": 25.0,
            "bert_base_throughput_items_per_sec": 40.0,
            "whisper_tiny_latency_ms": 150.0,
            "whisper_tiny_throughput_items_per_sec": 6.5,
            "vit_base_latency_ms": 45.0,
            "vit_base_throughput_items_per_sec": 22.0
        }
        
        # Apply scaling factors
        performance_estimate = {}
        for metric, base_value in base_performance.items():
            if "latency" in metric:
                # Lower latency is better, so inverse scaling
                scaled_value = base_value / (
                    cu_factor * compute_units / 12 + 
                    mem_factor * memory_mb / 6144
                )
            else:
                # Higher throughput is better, direct scaling
                scaled_value = base_value * (
                    cu_factor * compute_units / 12 + 
                    mem_factor * memory_mb / 6144
                )
            performance_estimate[metric] = round(scaled_value, 2)
            
        return performance_estimate
        
    def test_model_compatibility(self, model_path: str) -> Dict[str, Any]:
        """Test if a specific model is compatible with the selected device"""
        if not self.selected_device:
            if not self.select_device():
                return {"compatible": False, "error": "No device available"}
        
        # In real implementation, this would analyze the model file
        # For now, simulate based on file size if the file exists
        if os.path.exists(model_path):
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            memory_mb = self.selected_device["memory"]
            
            # Simple compatibility check based on size
            compatible = file_size_mb * 3 < memory_mb  # Assume 3x size needed for inference
            
            return {
                "compatible": compatible,
                "model_size_mb": round(file_size_mb, 2),
                "device_memory_mb": memory_mb,
                "reason": "Sufficient memory" if compatible else "Insufficient memory",
                "supported_precisions": self.selected_device["dtype_support"]
            }
        else:
            # Simulate compatibility based on model path name
            if "tiny" in model_path or "mini" in model_path or "small" in model_path:
                return {
                    "compatible": True,
                    "simulated": True,
                    "reason": "Small model variants are typically compatible",
                    "supported_precisions": self.selected_device["dtype_support"]
                }
            elif "base" in model_path:
                return {
                    "compatible": self.selected_device["memory"] >= 4096,
                    "simulated": True,
                    "reason": "Base models require at least 4GB memory",
                    "supported_precisions": self.selected_device["dtype_support"]
                }
            elif "large" in model_path:
                return {
                    "compatible": self.selected_device["memory"] >= 8192,
                    "simulated": True,
                    "reason": "Large models require at least 8GB memory",
                    "supported_precisions": self.selected_device["dtype_support"]
                }
            else:
                return {
                    "compatible": True,
                    "simulated": True,
                    "reason": "Compatibility assumed; actual test recommended",
                    "supported_precisions": self.selected_device["dtype_support"]
                }


class QNNPowerMonitor:
    """Monitor power and thermal impacts for QNN deployments"""
    
    def __init__(self, device_name: str = None):
        self.detector = QNNCapabilityDetector()
        if device_name:
            self.detector.select_device(device_name)
        else:
            self.detector.select_device()
        
        self.monitoring_active = False
        self.monitoring_data = []
        self.start_time = 0
        self.base_power_level = self._estimate_base_power()
    
    def _estimate_base_power(self) -> float:
        """Estimate base power level of the device when idle"""
        # In real implementation, this would use device-specific power APIs
        # For now, return simulated values based on device type
        if not self.detector.selected_device:
            return 0.0
        
        device_name = self.detector.selected_device["name"]
        if "8 Gen 3" in device_name:
            return 0.8  # Watts
        elif "8 Gen 2" in device_name:
            return 1.0  # Watts
        elif "7+" in device_name:
            return 0.7  # Watts
        else:
            return 0.5  # Watts
    
    def start_monitoring(self) -> bool:
        """Start monitoring power and thermal metrics"""
        if self.monitoring_active:
            return True  # Already monitoring
        
        if not self.detector.selected_device:
            logger.error("No device selected for monitoring")
            return False
        
        self.monitoring_active = True
        self.monitoring_data = []
        self.start_time = time.time()
        logger.info(f"Started power monitoring for {self.detector.selected_device['name']}")
        return True
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return summary stats"""
        if not self.monitoring_active:
            return {"error": "Monitoring not active"}
        
        duration = time.time() - self.start_time
        self.monitoring_active = False
        
        # In a real implementation, this would collect and aggregate actual readings
        # For now, simulate readings based on device type and duration
        
        # Generate simulated monitoring data points
        sample_count = min(int(duration * 10), 100)  # 10 samples per second, max 100
        
        device_name = self.detector.selected_device["name"]
        # Parameters for simulation based on device
        if "8 Gen 3" in device_name:
            base_power = 0.8
            power_variance = 0.3
            base_temp = 32.0
            temp_variance = 5.0
            temp_rise_factor = 0.5
        elif "8 Gen 2" in device_name:
            base_power = 1.0
            power_variance = 0.4
            base_temp = 34.0
            temp_variance = 6.0
            temp_rise_factor = 0.6
        else:
            base_power = 0.7
            power_variance = 0.2
            base_temp = 30.0
            temp_variance = 4.0
            temp_rise_factor = 0.4
        
        # Generate simulated power and temperature readings
        import random
        for i in range(sample_count):
            rel_time = i / max(1, sample_count - 1)  # 0 to 1
            
            # Power tends to start high and then stabilize
            power_factor = 1.0 + (0.5 * (1.0 - rel_time))
            power_watts = base_power * power_factor + random.uniform(-power_variance, power_variance)
            
            # Temperature tends to rise over time
            temp_rise = base_temp + (temp_rise_factor * rel_time * 15)  # Up to 15 degrees rise
            temp_celsius = temp_rise + random.uniform(-temp_variance, temp_variance)
            
            self.monitoring_data.append({
                "timestamp": self.start_time + (rel_time * duration),
                "power_watts": max(0.1, power_watts),  # Ensure positive power
                "soc_temp_celsius": max(20, temp_celsius),  # Reasonable temperature range
                "battery_temp_celsius": max(20, temp_celsius - 3 + random.uniform(-1, 1)),  # Battery temp follows SOC
                "throttling_detected": temp_celsius > 45  # Throttling threshold
            })
        
        # Compute summary statistics
        avg_power = sum(d["power_watts"] for d in self.monitoring_data) / len(self.monitoring_data)
        max_power = max(d["power_watts"] for d in self.monitoring_data)
        avg_soc_temp = sum(d["soc_temp_celsius"] for d in self.monitoring_data) / len(self.monitoring_data)
        max_soc_temp = max(d["soc_temp_celsius"] for d in self.monitoring_data)
        throttling_points = sum(1 for d in self.monitoring_data if d["throttling_detected"])
        
        # Estimated battery impact (simplified model)
        battery_impact_percent = (avg_power / 3.5) * 100  # Assuming 3.5W is full device power
        
        summary = {
            "device_name": device_name,
            "duration_seconds": duration,
            "average_power_watts": round(avg_power, 2),
            "peak_power_watts": round(max_power, 2),
            "average_soc_temp_celsius": round(avg_soc_temp, 2),
            "peak_soc_temp_celsius": round(max_soc_temp, 2),
            "thermal_throttling_detected": throttling_points > 0,
            "thermal_throttling_duration_seconds": throttling_points / 10,  # Assuming 10 samples per second
            "estimated_battery_impact_percent": round(battery_impact_percent, 2),
            "sample_count": len(self.monitoring_data),
            "power_efficiency_score": round(100 - battery_impact_percent, 2)  # Higher is better
        }
        
        logger.info(f"Completed power monitoring: avg={avg_power:.2f}W, max={max_power:.2f}W, impact={battery_impact_percent:.2f}%")
        return summary
    
    def get_monitoring_data(self) -> List[Dict[str, Any]]:
        """Get the raw monitoring data points"""
        return self.monitoring_data
    
    def estimate_battery_life(self, avg_power_watts: float, battery_capacity_mah: int = 5000, 
                             battery_voltage: float = 3.85) -> Dict[str, Any]:
        """
        Estimate battery life impact 
        
        Args:
            avg_power_watts: Average power consumption in watts
            battery_capacity_mah: Battery capacity in mAh (default: 5000mAh, typical flagship)
            battery_voltage: Battery voltage in volts (default: 3.85V, typical Li-ion)
        
        Returns:
            Dict with battery life estimates
        """
        # Calculate battery energy in watt-hours
        battery_wh = (battery_capacity_mah / 1000) * battery_voltage
        
        # Estimate battery life in hours at this power level
        hours = battery_wh / avg_power_watts if avg_power_watts > 0 else 0
        
        # Estimate percentage of battery used per hour
        percent_per_hour = (avg_power_watts / battery_wh) * 100 if battery_wh > 0 else 0
        
        # Compare to baseline power to get impact
        base_power_impact = self.base_power_level
        incremental_power = max(0, avg_power_watts - base_power_impact)
        incremental_percent = (incremental_power / avg_power_watts) * 100 if avg_power_watts > 0 else 0
        
        return {
            "battery_capacity_mah": battery_capacity_mah,
            "battery_energy_wh": round(battery_wh, 2),
            "estimated_runtime_hours": round(hours, 2),
            "battery_percent_per_hour": round(percent_per_hour, 2),
            "incremental_power_watts": round(incremental_power, 2),
            "incremental_percent": round(incremental_percent, 2),
            "efficiency_score": round(100 - min(100, incremental_percent), 2)  # Higher is better
        }


class QNNModelOptimizer:
    """Optimize models for QNN deployment on mobile/edge devices"""
    
    def __init__(self, device_name: str = None):
        self.detector = QNNCapabilityDetector()
        if device_name:
            self.detector.select_device(device_name)
        else:
            self.detector.select_device()
        
        self.supported_optimizations = {
            "quantization": ["fp16", "int8", "int4"], 
            "pruning": ["magnitude", "structured"],
            "distillation": ["vanilla", "progressive"],
            "compression": ["weight_sharing", "huffman"],
            "memory": ["kv_cache_optimization", "activation_checkpointing"]
        }
    
    def get_supported_optimizations(self) -> Dict[str, List[str]]:
        """Get supported optimization techniques for the current device"""
        if not self.detector.selected_device:
            return {}
        
        # Filter supported optimizations based on device capabilities
        result = dict(self.supported_optimizations)
        
        # Only include int4 quantization if supported by device
        if "int4" not in self.detector.selected_device["dtype_support"]:
            result["quantization"] = [q for q in result["quantization"] if q != "int4"]
            
        return result
    
    def recommend_optimizations(self, model_path: str) -> Dict[str, Any]:
        """Recommend optimizations for a specific model on the current device"""
        # Check model compatibility first
        compatibility = self.detector.test_model_compatibility(model_path)
        if not compatibility.get("compatible", False):
            return {
                "compatible": False,
                "reason": compatibility.get("reason", "Model incompatible with device"),
                "recommendations": ["Consider a smaller model variant"]
            }
        
        # Base recommendations on model name and device capabilities
        model_filename = os.path.basename(model_path)
        optimizations = []
        details = {}
        
        # Default optimization for all models
        optimizations.append("quantization:fp16")
        details["quantization"] = {
            "recommended": "fp16",
            "reason": "Good balance of accuracy and performance",
            "estimated_speedup": 1.8,
            "estimated_size_reduction": "50%"
        }
        
        # Model-specific optimizations
        if "llama" in model_filename.lower() or "opt" in model_filename.lower() or "gpt" in model_filename.lower():
            # Large language model optimizations
            if "int8" in self.detector.selected_device["dtype_support"]:
                optimizations.append("quantization:int8")
                details["quantization"]["recommended"] = "int8"
                details["quantization"]["estimated_speedup"] = 3.2
                details["quantization"]["estimated_size_reduction"] = "75%"
            
            optimizations.append("memory:kv_cache_optimization")
            details["memory"] = {
                "recommended": "kv_cache_optimization",
                "reason": "Critical for LLM inference efficiency",
                "estimated_memory_reduction": "40%"
            }
            
            if "large" in model_filename.lower():
                optimizations.append("pruning:magnitude")
                details["pruning"] = {
                    "recommended": "magnitude",
                    "reason": "Reduce model size with minimal accuracy impact",
                    "estimated_speedup": 1.4,
                    "estimated_size_reduction": "30%",
                    "sparsity_target": "30%"
                }
        
        elif "whisper" in model_filename.lower() or "wav2vec" in model_filename.lower():
            # Audio model optimizations
            optimizations.append("pruning:structured")
            details["pruning"] = {
                "recommended": "structured",
                "reason": "Maintain performance on hardware accelerators",
                "estimated_speedup": 1.5,
                "estimated_size_reduction": "35%",
                "sparsity_target": "40%"
            }
        
        elif "vit" in model_filename.lower() or "clip" in model_filename.lower():
            # Vision model optimizations
            if "int8" in self.detector.selected_device["dtype_support"]:
                optimizations.append("quantization:int8")
                details["quantization"]["recommended"] = "int8"
                details["quantization"]["estimated_speedup"] = 2.8
                details["quantization"]["estimated_size_reduction"] = "75%"
                
            optimizations.append("compression:weight_sharing")
            details["compression"] = {
                "recommended": "weight_sharing",
                "reason": "Effective for transformer attention layers",
                "estimated_speedup": 1.2,
                "estimated_size_reduction": "25%"
            }
        
        # Power efficiency recommendations for all models
        power_score = self._estimate_power_efficiency(model_filename, optimizations)
        
        return {
            "compatible": True,
            "recommended_optimizations": optimizations,
            "optimization_details": details,
            "estimated_power_efficiency_score": power_score,
            "device": self.detector.selected_device["name"],
            "estimated_memory_reduction": self._estimate_memory_impact(optimizations)
        }
    
    def _estimate_power_efficiency(self, model_name: str, optimizations: List[str]) -> float:
        """Estimate power efficiency score (0-100, higher is better)"""
        # Base score for the model type
        if "tiny" in model_name or "mini" in model_name:
            base_score = 85
        elif "small" in model_name:
            base_score = 75
        elif "base" in model_name:
            base_score = 65
        elif "large" in model_name:
            base_score = 45
        else:
            base_score = 60
        
        # Adjust based on optimizations
        for opt in optimizations:
            if "quantization:fp16" in opt:
                base_score += 5
            elif "quantization:int8" in opt:
                base_score += 10
            elif "quantization:int4" in opt:
                base_score += 15
            elif "pruning:" in opt:
                base_score += 5
            elif "kv_cache_optimization" in opt:
                base_score += 8
            elif "compression:" in opt:
                base_score += 5
        
        # Limit to 0-100 range
        return min(100, max(0, base_score))
    
    def _estimate_memory_impact(self, optimizations: List[str]) -> str:
        """Estimate memory reduction from optimizations"""
        total_reduction = 0
        
        for opt in optimizations:
            if "quantization:fp16" in opt:
                total_reduction += 0.5  # 50% reduction
            elif "quantization:int8" in opt:
                total_reduction += 0.75  # 75% reduction
            elif "quantization:int4" in opt:
                total_reduction += 0.875  # 87.5% reduction
            elif "pruning:" in opt:
                total_reduction += 0.3  # 30% reduction
            elif "kv_cache_optimization" in opt:
                total_reduction += 0.4  # 40% reduction
            elif "compression:" in opt:
                total_reduction += 0.25  # 25% reduction
        
        # Cap at 95% maximum reduction and convert to percentage string
        effective_reduction = min(0.95, total_reduction)
        return f"{int(effective_reduction * 100)}%"
    
    def simulate_optimization(self, model_path: str, optimizations: List[str]) -> Dict[str, Any]:
        """Simulate applying optimizations to a model"""
        # In a real implementation, this would apply actual optimizations
        # For now, simulate the results
        
        model_filename = os.path.basename(model_path)
        original_size = os.path.getsize(model_path) if os.path.exists(model_path) else 100 * 1024 * 1024  # 100MB default
        
        # Calculate size reduction based on optimizations
        size_reduction = 0
        for opt in optimizations:
            if "quantization:fp16" in opt:
                size_reduction += 0.5  # 50% reduction
            elif "quantization:int8" in opt:
                size_reduction += 0.75  # 75% reduction
            elif "quantization:int4" in opt:
                size_reduction += 0.875  # 87.5% reduction
            elif "pruning:" in opt:
                size_reduction += 0.3  # 30% reduction
            elif "compression:" in opt:
                size_reduction += 0.25  # 25% reduction
        
        # Cap at 95% maximum reduction
        effective_reduction = min(0.95, size_reduction)
        optimized_size = original_size * (1 - effective_reduction)
        
        # Simulate performance impact
        speedup = 1.0
        for opt in optimizations:
            if "quantization:fp16" in opt:
                speedup *= 1.8
            elif "quantization:int8" in opt:
                speedup *= 3.2
            elif "quantization:int4" in opt:
                speedup *= 4.0
            elif "pruning:" in opt:
                speedup *= 1.4
            elif "kv_cache_optimization" in opt:
                speedup *= 1.5
            elif "compression:" in opt:
                speedup *= 1.2
        
        # Cap at reasonable speedup
        effective_speedup = min(10.0, speedup)
        
        # Generate simulated benchmark results
        latency_reduction = 1.0 - (1.0 / effective_speedup)
        base_latency = 20.0  # ms
        if "large" in model_filename.lower():
            base_latency = 100.0
        elif "base" in model_filename.lower():
            base_latency = 50.0
        elif "small" in model_filename.lower():
            base_latency = 25.0
            
        optimized_latency = base_latency * (1.0 - latency_reduction)
        
        # Estimate power efficiency
        power_efficiency = self._estimate_power_efficiency(model_filename, optimizations)
        
        return {
            "model": model_filename,
            "original_size_bytes": original_size,
            "optimized_size_bytes": int(optimized_size),
            "size_reduction_percent": round(effective_reduction * 100, 2),
            "original_latency_ms": base_latency,
            "optimized_latency_ms": round(optimized_latency, 2),
            "speedup_factor": round(effective_speedup, 2),
            "power_efficiency_score": power_efficiency,
            "optimizations_applied": optimizations,
            "device": self.detector.selected_device["name"] if self.detector.selected_device else "Unknown"
        }


# Main functionality for command-line usage
def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QNN hardware detection and optimization")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # detect command
    detect_parser = subparsers.add_parser("detect", help="Detect QNN capabilities")
    detect_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # power command
    power_parser = subparsers.add_parser("power", help="Test power consumption")
    power_parser.add_argument("--device", help="Specific device to test")
    power_parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds")
    power_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Recommend model optimizations")
    optimize_parser.add_argument("--model", required=True, help="Path to model file")
    optimize_parser.add_argument("--device", help="Specific device to target")
    optimize_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    args = parser.parse_args()
    
    if args.command == "detect":
        detector = QNNCapabilityDetector()
        if detector.is_available():
            detector.select_device()
            result = detector.get_capability_summary()
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"QNN SDK Version: {result['sdk_version']}")
                print(f"Device: {result['device_name']}")
                print(f"Compute Units: {result['compute_units']}")
                print(f"Memory: {result['memory_mb']} MB")
                print(f"Precision Support: {', '.join(result['precision_support'])}")
                print("\nRecommended Models:")
                for model in result['recommended_models']:
                    print(f"  - {model}")
        else:
            print("QNN hardware not detected")
    
    elif args.command == "power":
        monitor = QNNPowerMonitor(args.device)
        print(f"Starting power monitoring for {args.duration} seconds...")
        monitor.start_monitoring()
        time.sleep(args.duration)
        results = monitor.stop_monitoring()
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print("\nPower Monitoring Results:")
            print(f"Device: {results['device_name']}")
            print(f"Duration: {results['duration_seconds']:.2f} seconds")
            print(f"Average Power: {results['average_power_watts']} W")
            print(f"Peak Power: {results['peak_power_watts']} W")
            print(f"Battery Impact: {results['estimated_battery_impact_percent']}%")
            print(f"Thermal Throttling: {'Yes' if results['thermal_throttling_detected'] else 'No'}")
            if results['thermal_throttling_detected']:
                print(f"Throttling Duration: {results['thermal_throttling_duration_seconds']:.2f} seconds")
            print(f"Power Efficiency Score: {results['power_efficiency_score']}/100")
    
    elif args.command == "optimize":
        optimizer = QNNModelOptimizer(args.device)
        recommendations = optimizer.recommend_optimizations(args.model)
        
        if args.json:
            print(json.dumps(recommendations, indent=2))
        else:
            print(f"\nModel Optimization Recommendations for {os.path.basename(args.model)}")
            print(f"Target Device: {recommendations['device']}")
            print(f"Compatible: {'Yes' if recommendations['compatible'] else 'No'}")
            
            if recommendations['compatible']:
                print("\nRecommended Optimizations:")
                for opt in recommendations['recommended_optimizations']:
                    print(f"  - {opt}")
                
                print(f"\nEstimated Memory Reduction: {recommendations['estimated_memory_reduction']}")
                print(f"Estimated Power Efficiency Score: {recommendations['estimated_power_efficiency_score']}/100")
                
                print("\nDetailed Recommendations:")
                for category, details in recommendations['optimization_details'].items():
                    print(f"  {category.title()}:")
                    for key, value in details.items():
                        print(f"    - {key}: {value}")
            else:
                print(f"Reason: {recommendations['reason']}")
                print("\nSuggestions:")
                for suggestion in recommendations.get('recommendations', []):
                    print(f"  - {suggestion}")

if __name__ == "__main__":
    main()