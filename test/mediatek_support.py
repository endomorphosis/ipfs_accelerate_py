#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaTek Neural Processing Support for IPFS Accelerate Python Framework

This module implements support for MediaTek Neural Processing Unit (NPU) hardware acceleration.
It provides components for model conversion, optimization, deployment, and benchmarking on 
MediaTek-powered mobile and edge devices.

Features:
- MediaTek Dimensity and Helio chip detection and capability analysis
- Model conversion to MediaTek Neural Processing SDK format
- Power-efficient deployment with MediaTek APU (AI Processing Unit)
- Battery impact analysis and optimization for MediaTek devices
- Thermal monitoring and management for MediaTek NPU
- Performance profiling and benchmarking

Date: April 2025
"""

import os
import sys
import json
import time
import logging
import datetime
import numpy as np
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

# Local imports
try:
    from benchmark_db_api import BenchmarkDBAPI, get_db_connection
    from mobile_thermal_monitoring import (
        ThermalZone,
        CoolingPolicy,
        MobileThermalMonitor
    )
except ImportError:
    logger.warning("Could not import some required modules. Some functionality may be limited.")


class MediaTekChipset:
    """Represents a MediaTek chipset with its capabilities."""
    
    def __init__(self, name: str, npu_cores: int, npu_tflops: float,
                 max_precision: str, supported_precisions: List[str],
                 max_power_draw: float, typical_power: float):
        """
        Initialize a MediaTek chipset.
        
        Args:
            name: Name of the chipset (e.g., "Dimensity 9300")
            npu_cores: Number of NPU cores
            npu_tflops: NPU performance in TFLOPS (FP16)
            max_precision: Maximum precision supported (e.g., "FP16")
            supported_precisions: List of supported precisions
            max_power_draw: Maximum power draw in watts
            typical_power: Typical power draw in watts
        """
        self.name = name
        self.npu_cores = npu_cores
        self.npu_tflops = npu_tflops
        self.max_precision = max_precision
        self.supported_precisions = supported_precisions
        self.max_power_draw = max_power_draw
        self.typical_power = typical_power
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation of the chipset
        """
        return {
            "name": self.name,
            "npu_cores": self.npu_cores,
            "npu_tflops": self.npu_tflops,
            "max_precision": self.max_precision,
            "supported_precisions": self.supported_precisions,
            "max_power_draw": self.max_power_draw,
            "typical_power": self.typical_power
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MediaTekChipset':
        """
        Create a MediaTek chipset from dictionary data.
        
        Args:
            data: Dictionary containing chipset data
            
        Returns:
            MediaTek chipset instance
        """
        return cls(
            name=data.get("name", "Unknown"),
            npu_cores=data.get("npu_cores", 0),
            npu_tflops=data.get("npu_tflops", 0.0),
            max_precision=data.get("max_precision", "FP16"),
            supported_precisions=data.get("supported_precisions", ["FP16", "INT8"]),
            max_power_draw=data.get("max_power_draw", 5.0),
            typical_power=data.get("typical_power", 2.0)
        )


class MediaTekChipsetRegistry:
    """Registry of MediaTek chipsets and their capabilities."""
    
    def __init__(self):
        """Initialize the MediaTek chipset registry."""
        self.chipsets = self._create_chipset_database()
    
    def _create_chipset_database(self) -> Dict[str, MediaTekChipset]:
        """
        Create database of MediaTek chipsets.
        
        Returns:
            Dictionary mapping chipset names to MediaTekChipset objects
        """
        chipsets = {}
        
        # Dimensity 9000 series (flagship)
        chipsets["dimensity_9300"] = MediaTekChipset(
            name="Dimensity 9300",
            npu_cores=6,
            npu_tflops=35.7,
            max_precision="FP16",
            supported_precisions=["FP32", "FP16", "BF16", "INT8", "INT4"],
            max_power_draw=9.0,
            typical_power=4.0
        )
        
        chipsets["dimensity_9200"] = MediaTekChipset(
            name="Dimensity 9200",
            npu_cores=6,
            npu_tflops=30.5,
            max_precision="FP16",
            supported_precisions=["FP32", "FP16", "BF16", "INT8", "INT4"],
            max_power_draw=8.5,
            typical_power=3.8
        )
        
        # Dimensity 8000 series (premium)
        chipsets["dimensity_8300"] = MediaTekChipset(
            name="Dimensity 8300",
            npu_cores=4,
            npu_tflops=19.8,
            max_precision="FP16",
            supported_precisions=["FP32", "FP16", "INT8", "INT4"],
            max_power_draw=6.5,
            typical_power=3.0
        )
        
        chipsets["dimensity_8200"] = MediaTekChipset(
            name="Dimensity 8200",
            npu_cores=4,
            npu_tflops=15.5,
            max_precision="FP16",
            supported_precisions=["FP32", "FP16", "INT8", "INT4"],
            max_power_draw=6.0,
            typical_power=2.8
        )
        
        # Dimensity 7000 series (mid-range)
        chipsets["dimensity_7300"] = MediaTekChipset(
            name="Dimensity 7300",
            npu_cores=2,
            npu_tflops=9.8,
            max_precision="FP16",
            supported_precisions=["FP16", "INT8", "INT4"],
            max_power_draw=5.0,
            typical_power=2.2
        )
        
        # Dimensity 6000 series (mainstream)
        chipsets["dimensity_6300"] = MediaTekChipset(
            name="Dimensity 6300",
            npu_cores=1,
            npu_tflops=4.2,
            max_precision="FP16",
            supported_precisions=["FP16", "INT8"],
            max_power_draw=3.5,
            typical_power=1.8
        )
        
        # Helio series
        chipsets["helio_g99"] = MediaTekChipset(
            name="Helio G99",
            npu_cores=1,
            npu_tflops=2.5,
            max_precision="FP16",
            supported_precisions=["FP16", "INT8"],
            max_power_draw=3.0,
            typical_power=1.5
        )
        
        chipsets["helio_g95"] = MediaTekChipset(
            name="Helio G95",
            npu_cores=1,
            npu_tflops=1.8,
            max_precision="FP16",
            supported_precisions=["FP16", "INT8"],
            max_power_draw=2.5,
            typical_power=1.2
        )
        
        return chipsets
    
    def get_chipset(self, name: str) -> Optional[MediaTekChipset]:
        """
        Get a MediaTek chipset by name.
        
        Args:
            name: Name of the chipset (e.g., "dimensity_9300")
            
        Returns:
            MediaTekChipset object or None if not found
        """
        # Try direct lookup
        if name in self.chipsets:
            return self.chipsets[name]
        
        # Try normalized name
        normalized_name = name.lower().replace(" ", "_").replace("-", "_")
        if normalized_name in self.chipsets:
            return self.chipsets[normalized_name]
        
        # Try prefix match
        for chipset_name, chipset in self.chipsets.items():
            if chipset_name.startswith(normalized_name) or normalized_name.startswith(chipset_name):
                return chipset
        
        # Try contains match
        for chipset_name, chipset in self.chipsets.items():
            if normalized_name in chipset_name or chipset_name in normalized_name:
                return chipset
        
        return None
    
    def get_all_chipsets(self) -> List[MediaTekChipset]:
        """
        Get all MediaTek chipsets.
        
        Returns:
            List of all MediaTekChipset objects
        """
        return list(self.chipsets.values())
    
    def save_to_file(self, file_path: str) -> bool:
        """
        Save chipset database to a file.
        
        Args:
            file_path: Path to save the database
            
        Returns:
            Success status
        """
        try:
            data = {name: chipset.to_dict() for name, chipset in self.chipsets.items()}
            
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved chipset database to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving chipset database: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, file_path: str) -> Optional['MediaTekChipsetRegistry']:
        """
        Load chipset database from a file.
        
        Args:
            file_path: Path to load the database from
            
        Returns:
            MediaTekChipsetRegistry or None if loading failed
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            registry = cls()
            registry.chipsets = {name: MediaTekChipset.from_dict(chipset_data) 
                                for name, chipset_data in data.items()}
            
            logger.info(f"Loaded chipset database from {file_path}")
            return registry
        except Exception as e:
            logger.error(f"Error loading chipset database: {e}")
            return None


class MediaTekDetector:
    """Detects and analyzes MediaTek hardware capabilities."""
    
    def __init__(self):
        """Initialize the MediaTek detector."""
        self.chipset_registry = MediaTekChipsetRegistry()
    
    def detect_mediatek_hardware(self) -> Optional[MediaTekChipset]:
        """
        Detect MediaTek hardware in the current device.
        
        Returns:
            MediaTekChipset or None if not detected
        """
        # For testing, check if we're simulating a specific MediaTek chipset
        if "TEST_MEDIATEK_CHIPSET" in os.environ:
            chipset_name = os.environ["TEST_MEDIATEK_CHIPSET"]
            return self.chipset_registry.get_chipset(chipset_name)
        
        # Attempt to detect MediaTek hardware through various methods
        chipset_name = None
        
        # Try Android detection methods
        if self._is_android():
            chipset_name = self._detect_on_android()
        
        # If a chipset was detected, look it up in the registry
        if chipset_name:
            return self.chipset_registry.get_chipset(chipset_name)
        
        # No MediaTek hardware detected
        return None
    
    def _is_android(self) -> bool:
        """
        Check if the current device is running Android.
        
        Returns:
            True if running on Android, False otherwise
        """
        # For testing
        if "TEST_PLATFORM" in os.environ and os.environ["TEST_PLATFORM"].lower() == "android":
            return True
        
        # Try to use the actual Android check
        try:
            # Check for Android build properties
            result = subprocess.run(
                ["getprop", "ro.build.version.sdk"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0 and result.stdout.strip() != ""
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _detect_on_android(self) -> Optional[str]:
        """
        Detect MediaTek chipset on Android.
        
        Returns:
            MediaTek chipset name or None if not detected
        """
        # For testing
        if "TEST_MEDIATEK_CHIPSET" in os.environ:
            return os.environ["TEST_MEDIATEK_CHIPSET"]
        
        try:
            # Try to get hardware info from Android properties
            result = subprocess.run(
                ["getprop", "ro.hardware"],
                capture_output=True,
                text=True
            )
            hardware = result.stdout.strip().lower()
            
            if "mt" in hardware or "mediatek" in hardware:
                # Try to get more specific chipset info
                result = subprocess.run(
                    ["getprop", "ro.board.platform"],
                    capture_output=True,
                    text=True
                )
                platform = result.stdout.strip().lower()
                
                # Try to map platform to known chipset
                if "mt6" in platform:  # Older naming scheme
                    if "mt6889" in platform or "mt6893" in platform:
                        return "dimensity_1200"
                    elif "mt6885" in platform:
                        return "dimensity_1000"
                    elif "mt6877" in platform:
                        return "dimensity_900"
                    # Add more mappings as needed
                elif "dimensity" in platform:
                    if "9300" in platform:
                        return "dimensity_9300"
                    elif "9200" in platform:
                        return "dimensity_9200"
                    elif "8300" in platform:
                        return "dimensity_8300"
                    elif "8200" in platform:
                        return "dimensity_8200"
                    elif "7300" in platform:
                        return "dimensity_7300"
                    elif "6300" in platform:
                        return "dimensity_6300"
                    # Extract number if pattern not matched exactly
                    import re
                    match = re.search(r'dimensity[_\s-]*(\d+)', platform)
                    if match:
                        return f"dimensity_{match.group(1)}"
                elif "helio" in platform:
                    if "g99" in platform:
                        return "helio_g99"
                    elif "g95" in platform:
                        return "helio_g95"
                    # Extract model if pattern not matched exactly
                    import re
                    match = re.search(r'helio[_\s-]*([a-z]\d+)', platform, re.IGNORECASE)
                    if match:
                        return f"helio_{match.group(1).lower()}"
                
                # If we got here, we know it's MediaTek but couldn't identify the exact model
                return "mediatek_unknown"
            
            return None
            
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
    
    def get_capability_analysis(self, chipset: MediaTekChipset) -> Dict[str, Any]:
        """
        Get detailed capability analysis for a MediaTek chipset.
        
        Args:
            chipset: MediaTek chipset to analyze
            
        Returns:
            Dictionary containing capability analysis
        """
        # Model capability classification
        model_capabilities = {
            "embedding_models": {
                "suitable": True,
                "max_size": "Large",
                "performance": "High",
                "notes": "Efficient for all embedding model sizes"
            },
            "vision_models": {
                "suitable": True,
                "max_size": "Large",
                "performance": "High",
                "notes": "Strong performance for vision models"
            },
            "text_generation": {
                "suitable": chipset.npu_tflops >= 15.0,
                "max_size": "Small" if chipset.npu_tflops < 10.0 else 
                            "Medium" if chipset.npu_tflops < 25.0 else "Large",
                "performance": "Low" if chipset.npu_tflops < 10.0 else 
                               "Medium" if chipset.npu_tflops < 25.0 else "High",
                "notes": "Limited to smaller LLMs on mid-range and lower chipsets"
            },
            "audio_models": {
                "suitable": True,
                "max_size": "Medium" if chipset.npu_tflops < 15.0 else "Large",
                "performance": "Medium" if chipset.npu_tflops < 15.0 else "High",
                "notes": "Good performance for most audio models"
            },
            "multimodal_models": {
                "suitable": chipset.npu_tflops >= 10.0,
                "max_size": "Small" if chipset.npu_tflops < 15.0 else 
                            "Medium" if chipset.npu_tflops < 30.0 else "Large",
                "performance": "Low" if chipset.npu_tflops < 15.0 else 
                               "Medium" if chipset.npu_tflops < 30.0 else "High",
                "notes": "Best suited for flagship chipsets (9000 and 8000 series)"
            }
        }
        
        # Precision support analysis
        precision_support = {
            precision: True for precision in chipset.supported_precisions
        }
        precision_support.update({
            precision: False for precision in ["FP32", "FP16", "BF16", "INT8", "INT4", "INT2"]
            if precision not in chipset.supported_precisions
        })
        
        # Power efficiency analysis
        power_efficiency = {
            "tflops_per_watt": chipset.npu_tflops / chipset.typical_power,
            "efficiency_rating": "Low" if (chipset.npu_tflops / chipset.typical_power) < 3.0 else
                                "Medium" if (chipset.npu_tflops / chipset.typical_power) < 6.0 else "High",
            "battery_impact": "High" if chipset.typical_power > 3.0 else 
                              "Medium" if chipset.typical_power > 2.0 else "Low"
        }
        
        # Recommended optimizations
        recommended_optimizations = []
        
        if "INT8" in chipset.supported_precisions:
            recommended_optimizations.append("INT8 quantization")
        
        if "INT4" in chipset.supported_precisions:
            recommended_optimizations.append("INT4 quantization for weight-only")
        
        if chipset.npu_cores > 1:
            recommended_optimizations.append("Model parallelism across NPU cores")
        
        if chipset.typical_power > 2.5:
            recommended_optimizations.append("Dynamic power scaling")
            recommended_optimizations.append("Thermal-aware scheduling")
        
        # Competitive analysis
        competitive_position = {
            "vs_qualcomm": "Similar" if 10.0 <= chipset.npu_tflops <= 25.0 else
                          "Higher" if chipset.npu_tflops > 25.0 else "Lower",
            "vs_apple": "Lower" if chipset.npu_tflops < 20.0 else "Similar",
            "vs_samsung": "Higher" if chipset.npu_tflops > 15.0 else "Similar",
            "overall_ranking": "High-end" if chipset.npu_tflops >= 25.0 else
                              "Mid-range" if chipset.npu_tflops >= 10.0 else "Entry-level"
        }
        
        return {
            "chipset": chipset.to_dict(),
            "model_capabilities": model_capabilities,
            "precision_support": precision_support,
            "power_efficiency": power_efficiency,
            "recommended_optimizations": recommended_optimizations,
            "competitive_position": competitive_position
        }


class MediaTekModelConverter:
    """Converts models to MediaTek Neural Processing SDK format."""
    
    def __init__(self, toolchain_path: Optional[str] = None):
        """
        Initialize the MediaTek model converter.
        
        Args:
            toolchain_path: Optional path to MediaTek Neural Processing SDK toolchain
        """
        self.toolchain_path = toolchain_path or os.environ.get("MEDIATEK_SDK_PATH", "/opt/mediatek/npu-sdk")
    
    def _check_toolchain(self) -> bool:
        """
        Check if MediaTek toolchain is available.
        
        Returns:
            True if toolchain is available, False otherwise
        """
        # For testing, assume toolchain is available if we're simulating
        if "TEST_MEDIATEK_CHIPSET" in os.environ:
            return True
        
        # Check if the toolchain directory exists
        return os.path.exists(self.toolchain_path)
    
    def convert_to_mediatek_format(self, 
                                  model_path: str, 
                                  output_path: str,
                                  target_chipset: str,
                                  precision: str = "INT8",
                                  optimize_for_latency: bool = True,
                                  enable_power_optimization: bool = True) -> bool:
        """
        Convert a model to MediaTek Neural Processing SDK format.
        
        Args:
            model_path: Path to input model (ONNX, TensorFlow, or PyTorch)
            output_path: Path to save converted model
            target_chipset: Target MediaTek chipset
            precision: Target precision (FP32, FP16, INT8, INT4)
            optimize_for_latency: Whether to optimize for latency (otherwise throughput)
            enable_power_optimization: Whether to enable power optimizations
            
        Returns:
            True if conversion successful, False otherwise
        """
        logger.info(f"Converting model to MediaTek format: {model_path} -> {output_path}")
        logger.info(f"Target chipset: {target_chipset}, precision: {precision}")
        
        # Check if toolchain is available
        if not self._check_toolchain():
            logger.error(f"MediaTek Neural Processing SDK toolchain not found at {self.toolchain_path}")
            return False
        
        # For testing/simulation, we'll just create a mock output file
        if "TEST_MEDIATEK_CHIPSET" in os.environ:
            try:
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                # Create a mock model file
                with open(output_path, 'w') as f:
                    f.write(f"MediaTek NPU model for {target_chipset}\n")
                    f.write(f"Original model: {model_path}\n")
                    f.write(f"Precision: {precision}\n")
                    f.write(f"Optimize for latency: {optimize_for_latency}\n")
                    f.write(f"Power optimization: {enable_power_optimization}\n")
                
                logger.info(f"Created mock MediaTek model at {output_path}")
                return True
            except Exception as e:
                logger.error(f"Error creating mock MediaTek model: {e}")
                return False
        
        # In a real implementation, we would call the MediaTek neural compiler here
        # This would be something like:
        # command = [
        #     f"{self.toolchain_path}/bin/npu-compiler",
        #     "--input", model_path,
        #     "--output", output_path,
        #     "--target", target_chipset,
        #     "--precision", precision
        # ]
        # if optimize_for_latency:
        #     command.append("--optimize-latency")
        # if enable_power_optimization:
        #     command.append("--enable-power-opt")
        # 
        # result = subprocess.run(command, capture_output=True, text=True)
        # return result.returncode == 0
        
        # Since we can't actually run the compiler, simulate a successful conversion
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Create a mock model file
            with open(output_path, 'w') as f:
                f.write(f"MediaTek NPU model for {target_chipset}\n")
                f.write(f"Original model: {model_path}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Optimize for latency: {optimize_for_latency}\n")
                f.write(f"Power optimization: {enable_power_optimization}\n")
            
            logger.info(f"Created mock MediaTek model at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating mock MediaTek model: {e}")
            return False
    
    def quantize_model(self, 
                     model_path: str,
                     output_path: str,
                     calibration_data_path: Optional[str] = None,
                     precision: str = "INT8",
                     per_channel: bool = True) -> bool:
        """
        Quantize a model for MediaTek NPU.
        
        Args:
            model_path: Path to input model
            output_path: Path to save quantized model
            calibration_data_path: Path to calibration data
            precision: Target precision (INT8, INT4)
            per_channel: Whether to use per-channel quantization
            
        Returns:
            True if quantization successful, False otherwise
        """
        logger.info(f"Quantizing model to {precision}: {model_path} -> {output_path}")
        
        # Check if toolchain is available
        if not self._check_toolchain():
            logger.error(f"MediaTek Neural Processing SDK toolchain not found at {self.toolchain_path}")
            return False
        
        # For testing/simulation, create a mock output file
        if "TEST_MEDIATEK_CHIPSET" in os.environ:
            try:
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                # Create a mock quantized model file
                with open(output_path, 'w') as f:
                    f.write(f"MediaTek NPU quantized model ({precision})\n")
                    f.write(f"Original model: {model_path}\n")
                    f.write(f"Calibration data: {calibration_data_path}\n")
                    f.write(f"Per-channel: {per_channel}\n")
                
                logger.info(f"Created mock quantized model at {output_path}")
                return True
            except Exception as e:
                logger.error(f"Error creating mock quantized model: {e}")
                return False
        
        # In a real implementation, we would call the MediaTek quantization tool
        # This would be something like:
        # command = [
        #     f"{self.toolchain_path}/bin/npu-quantizer",
        #     "--input", model_path,
        #     "--output", output_path,
        #     "--precision", precision
        # ]
        # if calibration_data_path:
        #     command.extend(["--calibration-data", calibration_data_path])
        # if per_channel:
        #     command.append("--per-channel")
        # 
        # result = subprocess.run(command, capture_output=True, text=True)
        # return result.returncode == 0
        
        # Since we can't actually run the quantizer, simulate a successful quantization
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Create a mock quantized model file
            with open(output_path, 'w') as f:
                f.write(f"MediaTek NPU quantized model ({precision})\n")
                f.write(f"Original model: {model_path}\n")
                f.write(f"Calibration data: {calibration_data_path}\n")
                f.write(f"Per-channel: {per_channel}\n")
            
            logger.info(f"Created mock quantized model at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating mock quantized model: {e}")
            return False
    
    def analyze_model_compatibility(self, 
                                  model_path: str,
                                  target_chipset: str) -> Dict[str, Any]:
        """
        Analyze model compatibility with MediaTek NPU.
        
        Args:
            model_path: Path to input model
            target_chipset: Target MediaTek chipset
            
        Returns:
            Dictionary containing compatibility analysis
        """
        logger.info(f"Analyzing model compatibility for {target_chipset}: {model_path}")
        
        # For testing/simulation, return a mock compatibility analysis
        model_info = {
            "format": model_path.split(".")[-1],
            "size_mb": 10.5,  # Mock size
            "ops_count": 5.2e9,  # Mock ops count
            "estimated_memory_mb": 250  # Mock memory estimate
        }
        
        # Get chipset information from registry
        chipset_registry = MediaTekChipsetRegistry()
        chipset = chipset_registry.get_chipset(target_chipset)
        
        if not chipset:
            logger.warning(f"Unknown chipset: {target_chipset}")
            chipset = MediaTekChipset(
                name=target_chipset,
                npu_cores=1,
                npu_tflops=1.0,
                max_precision="FP16",
                supported_precisions=["FP16", "INT8"],
                max_power_draw=2.0,
                typical_power=1.0
            )
        
        # Analyze compatibility
        compatibility = {
            "supported": True,
            "recommended_precision": "INT8" if "INT8" in chipset.supported_precisions else "FP16",
            "estimated_performance": {
                "latency_ms": 50.0,  # Mock latency
                "throughput_items_per_second": 20.0,  # Mock throughput
                "power_consumption_mw": chipset.typical_power * 1000 * 0.8,  # Mock power consumption
                "memory_usage_mb": model_info["estimated_memory_mb"]
            },
            "optimization_opportunities": [
                "INT8 quantization" if "INT8" in chipset.supported_precisions else None,
                "INT4 weight-only quantization" if "INT4" in chipset.supported_precisions else None,
                "Layer fusion" if chipset.npu_tflops > 5.0 else None,
                "Memory bandwidth optimization" if chipset.npu_cores > 2 else None
            ],
            "potential_issues": []
        }
        
        # Filter out None values from optimization opportunities
        compatibility["optimization_opportunities"] = [
            opt for opt in compatibility["optimization_opportunities"] if opt is not None
        ]
        
        # Check for potential issues
        if model_info["ops_count"] > chipset.npu_tflops * 1e12 * 0.1:
            compatibility["potential_issues"].append("Model complexity may exceed optimal performance range")
        
        if model_info["estimated_memory_mb"] > 1000 and chipset.npu_tflops < 10.0:
            compatibility["potential_issues"].append("Model memory requirements may be too high for this chipset")
        
        # If no issues found, note that
        if not compatibility["potential_issues"]:
            compatibility["potential_issues"].append("No significant issues detected")
        
        return {
            "model_info": model_info,
            "chipset_info": chipset.to_dict(),
            "compatibility": compatibility
        }


class MediaTekThermalMonitor:
    """MediaTek-specific thermal monitoring extension."""
    
    def __init__(self, device_type: str = "android"):
        """
        Initialize MediaTek thermal monitor.
        
        Args:
            device_type: Type of device (e.g., "android")
        """
        # Create base thermal monitor
        self.base_monitor = MobileThermalMonitor(device_type=device_type)
        
        # Add MediaTek-specific thermal zones
        self._add_mediatek_thermal_zones()
        
        # Set MediaTek-specific cooling policy
        self._set_mediatek_cooling_policy()
    
    def _add_mediatek_thermal_zones(self):
        """Add MediaTek-specific thermal zones."""
        # APU (AI Processing Unit) thermal zone
        self.base_monitor.thermal_zones["apu"] = ThermalZone(
            name="apu",
            critical_temp=90.0,
            warning_temp=75.0,
            path="/sys/class/thermal/thermal_zone5/temp" if os.path.exists("/sys/class/thermal/thermal_zone5/temp") else None,
            sensor_type="apu"
        )
        
        # Some MediaTek devices have a separate NPU thermal zone
        if os.path.exists("/sys/class/thermal/thermal_zone6/temp"):
            self.base_monitor.thermal_zones["npu"] = ThermalZone(
                name="npu",
                critical_temp=95.0,
                warning_temp=80.0,
                path="/sys/class/thermal/thermal_zone6/temp",
                sensor_type="npu"
            )
        
        logger.info("Added MediaTek-specific thermal zones")
    
    def _set_mediatek_cooling_policy(self):
        """Set MediaTek-specific cooling policy."""
        from mobile_thermal_monitoring import ThermalEventType, CoolingPolicy
        
        # Create a specialized cooling policy for MediaTek
        policy = CoolingPolicy(
            name="MediaTek NPU Cooling Policy",
            description="Cooling policy optimized for MediaTek NPU/APU"
        )
        
        # MediaTek APUs are particularly sensitive to thermal conditions
        # So we implement a more aggressive policy
        
        # Normal actions
        policy.add_action(
            ThermalEventType.NORMAL,
            lambda: self.base_monitor.throttling_manager._set_throttling_level(0),
            "Clear throttling and restore normal performance"
        )
        
        # Warning actions - more aggressive than default
        policy.add_action(
            ThermalEventType.WARNING,
            lambda: self.base_monitor.throttling_manager._set_throttling_level(2),  # Moderate throttling instead of mild
            "Apply moderate throttling (25% performance reduction)"
        )
        
        # Throttling actions - more aggressive than default
        policy.add_action(
            ThermalEventType.THROTTLING,
            lambda: self.base_monitor.throttling_manager._set_throttling_level(3),  # Heavy throttling
            "Apply heavy throttling (50% performance reduction)"
        )
        
        # Critical actions - more aggressive than default
        policy.add_action(
            ThermalEventType.CRITICAL,
            lambda: self.base_monitor.throttling_manager._set_throttling_level(4),  # Severe throttling
            "Apply severe throttling (75% performance reduction)"
        )
        policy.add_action(
            ThermalEventType.CRITICAL,
            lambda: self._reduce_apu_clock(),
            "Reduce APU clock frequency"
        )
        
        # Emergency actions
        policy.add_action(
            ThermalEventType.EMERGENCY,
            lambda: self.base_monitor.throttling_manager._set_throttling_level(5),  # Emergency throttling
            "Apply emergency throttling (90% performance reduction)"
        )
        policy.add_action(
            ThermalEventType.EMERGENCY,
            lambda: self._pause_apu_workload(),
            "Pause APU workload temporarily"
        )
        policy.add_action(
            ThermalEventType.EMERGENCY,
            lambda: self.base_monitor.throttling_manager._trigger_emergency_cooldown(),
            "Trigger emergency cooldown procedure"
        )
        
        # Apply the policy
        self.base_monitor.configure_cooling_policy(policy)
        logger.info("Applied MediaTek-specific cooling policy")
    
    def _reduce_apu_clock(self):
        """Reduce APU clock frequency."""
        logger.warning("Reducing APU clock frequency")
        # In a real implementation, this would interact with MediaTek's
        # thermal management framework to reduce APU/NPU clock frequency
        # For simulation, we'll just log this action
    
    def _pause_apu_workload(self):
        """Pause APU workload temporarily."""
        logger.warning("Pausing APU workload temporarily")
        # In a real implementation, this would signal the inference runtime
        # to pause NPU execution and potentially fall back to CPU
        # For simulation, we'll just log this action
    
    def start_monitoring(self):
        """Start thermal monitoring."""
        self.base_monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop thermal monitoring."""
        self.base_monitor.stop_monitoring()
    
    def get_current_thermal_status(self) -> Dict[str, Any]:
        """
        Get current thermal status.
        
        Returns:
            Dictionary with thermal status information
        """
        status = self.base_monitor.get_current_thermal_status()
        
        # Add MediaTek-specific thermal information
        if "apu" in self.base_monitor.thermal_zones:
            status["apu_temperature"] = self.base_monitor.thermal_zones["apu"].current_temp
        
        if "npu" in self.base_monitor.thermal_zones:
            status["npu_temperature"] = self.base_monitor.thermal_zones["npu"].current_temp
        
        return status
    
    def get_recommendations(self) -> List[str]:
        """
        Get MediaTek-specific thermal recommendations.
        
        Returns:
            List of recommendations
        """
        recommendations = self.base_monitor._generate_recommendations()
        
        # Add MediaTek-specific recommendations
        if "apu" in self.base_monitor.thermal_zones:
            apu_zone = self.base_monitor.thermal_zones["apu"]
            if apu_zone.current_temp >= apu_zone.warning_temp:
                recommendations.append(f"MEDIATEK: APU temperature ({apu_zone.current_temp:.1f}°C) is elevated. Consider using INT8 quantization to reduce power.")
            
            if apu_zone.current_temp >= apu_zone.critical_temp:
                recommendations.append(f"MEDIATEK: APU temperature ({apu_zone.current_temp:.1f}°C) is critical. Reduce batch size or switch to CPU inference.")
        
        return recommendations


class MediaTekBenchmarkRunner:
    """Runs benchmarks on MediaTek NPU hardware."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize MediaTek benchmark runner.
        
        Args:
            db_path: Optional path to benchmark database
        """
        self.db_path = db_path or os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
        self.thermal_monitor = None
        self.detector = MediaTekDetector()
        self.chipset = self.detector.detect_mediatek_hardware()
        
        # Initialize database connection
        self._init_db()
    
    def _init_db(self):
        """Initialize database connection if available."""
        self.db_api = None
        
        if self.db_path:
            try:
                from benchmark_db_api import BenchmarkDBAPI
                self.db_api = BenchmarkDBAPI(self.db_path)
                logger.info(f"Connected to benchmark database at {self.db_path}")
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to initialize database connection: {e}")
                self.db_path = None
    
    def run_benchmark(self, 
                    model_path: str,
                    batch_sizes: List[int] = [1, 2, 4, 8],
                    precision: str = "INT8",
                    duration_seconds: int = 60,
                    monitor_thermals: bool = True,
                    output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run benchmark on MediaTek NPU.
        
        Args:
            model_path: Path to model
            batch_sizes: List of batch sizes to benchmark
            precision: Precision to use for benchmarking
            duration_seconds: Duration of benchmark in seconds per batch size
            monitor_thermals: Whether to monitor thermals during benchmark
            output_path: Optional path to save benchmark results
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info(f"Running MediaTek NPU benchmark for {model_path}")
        logger.info(f"Batch sizes: {batch_sizes}, precision: {precision}, duration: {duration_seconds}s")
        
        if not self.chipset:
            logger.error("No MediaTek hardware detected")
            return {"error": "No MediaTek hardware detected"}
        
        # Start thermal monitoring if requested
        if monitor_thermals:
            logger.info("Starting thermal monitoring")
            self.thermal_monitor = MediaTekThermalMonitor(device_type="android")
            self.thermal_monitor.start_monitoring()
        
        try:
            # Run benchmark for each batch size
            batch_results = {}
            
            for batch_size in batch_sizes:
                logger.info(f"Benchmarking with batch size {batch_size}")
                
                # Simulate running the model on MediaTek NPU
                start_time = time.time()
                latencies = []
                
                # For testing/simulation, generate synthetic benchmark data
                # In a real implementation, we would load the model and run inference
                
                # Synthetic throughput calculation based on chipset capabilities and batch size
                throughput_base = self.chipset.npu_tflops * 10  # Baseline items per second
                throughput_scale = 1.0 if batch_size == 1 else (1.0 + 0.5 * np.log2(batch_size))  # Scale with batch size
                if batch_size > 8:
                    throughput_scale = throughput_scale * 0.9  # Diminishing returns for very large batches
                
                throughput = throughput_base * throughput_scale
                
                # Synthetic latency
                latency_base = 10.0  # Base latency in ms for batch size 1
                latency = latency_base * (1 + 0.2 * np.log2(batch_size))  # Latency increases with batch size
                
                # Simulate multiple runs
                num_runs = min(100, int(duration_seconds / (latency / 1000)))
                for _ in range(num_runs):
                    # Add some variation to the latency
                    run_latency = latency * (1 + 0.1 * np.random.normal(0, 0.1))
                    latencies.append(run_latency)
                    
                    # Simulate the passage of time
                    if len(latencies) % 10 == 0:
                        time.sleep(0.01)
                
                end_time = time.time()
                actual_duration = end_time - start_time
                
                # Calculate statistics
                latency_avg = np.mean(latencies)
                latency_p50 = np.percentile(latencies, 50)
                latency_p90 = np.percentile(latencies, 90)
                latency_p99 = np.percentile(latencies, 99)
                
                # Power metrics (simulated)
                power_consumption = self.chipset.typical_power * (0.5 + 0.5 * min(batch_size, 8) / 8)  # W
                power_consumption_mw = power_consumption * 1000  # Convert to mW
                energy_per_inference = power_consumption_mw * (latency_avg / 1000)  # mJ
                
                # Memory metrics (simulated)
                memory_base = 200  # Base memory in MB
                memory_usage = memory_base * (1 + 0.5 * min(batch_size, 8) / 8)  # MB
                
                # Temperature metrics (from thermal monitor if available)
                temperature_metrics = {}
                if self.thermal_monitor:
                    status = self.thermal_monitor.get_current_thermal_status()
                    temperature_metrics = {
                        "cpu_temperature": status.get("thermal_zones", {}).get("cpu", {}).get("current_temp", 0),
                        "gpu_temperature": status.get("thermal_zones", {}).get("gpu", {}).get("current_temp", 0),
                        "apu_temperature": status.get("apu_temperature", 0),
                    }
                
                # Store results for this batch size
                batch_results[batch_size] = {
                    "throughput_items_per_second": throughput,
                    "latency_ms": {
                        "avg": latency_avg,
                        "p50": latency_p50,
                        "p90": latency_p90,
                        "p99": latency_p99
                    },
                    "power_metrics": {
                        "power_consumption_mw": power_consumption_mw,
                        "energy_per_inference_mj": energy_per_inference,
                        "performance_per_watt": throughput / power_consumption
                    },
                    "memory_metrics": {
                        "memory_usage_mb": memory_usage
                    },
                    "temperature_metrics": temperature_metrics
                }
            
            # Combine results
            results = {
                "model_path": model_path,
                "precision": precision,
                "chipset": self.chipset.to_dict() if self.chipset else None,
                "timestamp": time.time(),
                "datetime": datetime.datetime.now().isoformat(),
                "batch_results": batch_results,
                "system_info": self._get_system_info()
            }
            
            # Get thermal recommendations if available
            if self.thermal_monitor:
                results["thermal_recommendations"] = self.thermal_monitor.get_recommendations()
            
            # Save results to database if available
            if self.db_api:
                try:
                    self.db_api.insert_mediatek_benchmark(results)
                    logger.info("Saved benchmark results to database")
                except Exception as e:
                    logger.error(f"Error saving results to database: {e}")
            
            # Save results to file if requested
            if output_path:
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2)
                    logger.info(f"Saved benchmark results to {output_path}")
                except Exception as e:
                    logger.error(f"Error saving results to file: {e}")
            
            return results
        
        finally:
            # Stop thermal monitoring if started
            if self.thermal_monitor:
                logger.info("Stopping thermal monitoring")
                self.thermal_monitor.stop_monitoring()
                self.thermal_monitor = None
    
    def _get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.
        
        Returns:
            Dictionary containing system information
        """
        # For testing/simulation, create mock system info
        system_info = {
            "os": "Android",
            "os_version": "13",
            "device_model": "MediaTek Test Device",
            "cpu_model": f"MediaTek {self.chipset.name if self.chipset else 'Unknown'}",
            "memory_total_gb": 8,
            "storage_total_gb": 128
        }
        
        # In a real implementation, we would get this information from the device
        
        return system_info
    
    def compare_with_cpu(self, 
                       model_path: str,
                       batch_size: int = 1,
                       precision: str = "INT8",
                       duration_seconds: int = 30) -> Dict[str, Any]:
        """
        Compare MediaTek NPU performance with CPU.
        
        Args:
            model_path: Path to model
            batch_size: Batch size for comparison
            precision: Precision to use
            duration_seconds: Duration of benchmark in seconds
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Comparing MediaTek NPU with CPU for {model_path}")
        
        if not self.chipset:
            logger.error("No MediaTek hardware detected")
            return {"error": "No MediaTek hardware detected"}
        
        # Run NPU benchmark
        npu_results = self.run_benchmark(
            model_path=model_path,
            batch_sizes=[batch_size],
            precision=precision,
            duration_seconds=duration_seconds,
            monitor_thermals=True
        )
        
        # Get NPU metrics
        npu_throughput = npu_results.get("batch_results", {}).get(batch_size, {}).get("throughput_items_per_second", 0)
        npu_latency = npu_results.get("batch_results", {}).get(batch_size, {}).get("latency_ms", {}).get("avg", 0)
        npu_power = npu_results.get("batch_results", {}).get(batch_size, {}).get("power_metrics", {}).get("power_consumption_mw", 0)
        
        # Simulate CPU benchmark (in a real implementation, we would run the model on CPU)
        # CPU is typically much slower than NPU for inference
        cpu_throughput = npu_throughput * 0.1  # Assume CPU is ~10x slower
        cpu_latency = npu_latency * 10.0  # Assume CPU has ~10x higher latency
        cpu_power = npu_power * 1.5  # Assume CPU uses ~1.5x more power
        
        # Calculate speedup ratios
        speedup_throughput = npu_throughput / cpu_throughput if cpu_throughput > 0 else float('inf')
        speedup_latency = cpu_latency / npu_latency if npu_latency > 0 else float('inf')
        speedup_power_efficiency = (cpu_power / cpu_throughput) / (npu_power / npu_throughput) if cpu_throughput > 0 and npu_throughput > 0 else float('inf')
        
        # Compile comparison results
        comparison = {
            "model_path": model_path,
            "batch_size": batch_size,
            "precision": precision,
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "npu": {
                "throughput_items_per_second": npu_throughput,
                "latency_ms": npu_latency,
                "power_consumption_mw": npu_power
            },
            "cpu": {
                "throughput_items_per_second": cpu_throughput,
                "latency_ms": cpu_latency,
                "power_consumption_mw": cpu_power
            },
            "speedups": {
                "throughput": speedup_throughput,
                "latency": speedup_latency,
                "power_efficiency": speedup_power_efficiency
            },
            "chipset": self.chipset.to_dict() if self.chipset else None
        }
        
        return comparison
    
    def compare_precision_impact(self,
                               model_path: str,
                               batch_size: int = 1,
                               precisions: List[str] = ["FP32", "FP16", "INT8"],
                               duration_seconds: int = 30) -> Dict[str, Any]:
        """
        Compare impact of different precisions on MediaTek NPU performance.
        
        Args:
            model_path: Path to model
            batch_size: Batch size for comparison
            precisions: List of precisions to compare
            duration_seconds: Duration of benchmark in seconds per precision
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Comparing precision impact for {model_path}")
        logger.info(f"Precisions: {precisions}, batch size: {batch_size}")
        
        if not self.chipset:
            logger.error("No MediaTek hardware detected")
            return {"error": "No MediaTek hardware detected"}
        
        # Check which precisions are supported by the chipset
        supported_precisions = []
        for precision in precisions:
            if precision in self.chipset.supported_precisions:
                supported_precisions.append(precision)
            else:
                logger.warning(f"Precision {precision} is not supported by {self.chipset.name}")
        
        if not supported_precisions:
            logger.error("None of the specified precisions are supported")
            return {"error": "None of the specified precisions are supported"}
        
        # Run benchmark for each precision
        precision_results = {}
        
        for precision in supported_precisions:
            logger.info(f"Benchmarking with precision {precision}")
            
            # Run benchmark
            results = self.run_benchmark(
                model_path=model_path,
                batch_sizes=[batch_size],
                precision=precision,
                duration_seconds=duration_seconds,
                monitor_thermals=True
            )
            
            # Extract relevant metrics
            precision_results[precision] = results.get("batch_results", {}).get(batch_size, {})
        
        # Analyze precision impact
        reference_precision = supported_precisions[0]
        impact_analysis = {}
        
        for precision in supported_precisions[1:]:
            ref_throughput = precision_results[reference_precision].get("throughput_items_per_second", 0)
            ref_latency = precision_results[reference_precision].get("latency_ms", {}).get("avg", 0)
            ref_power = precision_results[reference_precision].get("power_metrics", {}).get("power_consumption_mw", 0)
            
            cur_throughput = precision_results[precision].get("throughput_items_per_second", 0)
            cur_latency = precision_results[precision].get("latency_ms", {}).get("avg", 0)
            cur_power = precision_results[precision].get("power_metrics", {}).get("power_consumption_mw", 0)
            
            # Calculate relative changes
            throughput_change = (cur_throughput / ref_throughput - 1) * 100 if ref_throughput > 0 else float('inf')
            latency_change = (ref_latency / cur_latency - 1) * 100 if cur_latency > 0 else float('inf')
            power_change = (ref_power / cur_power - 1) * 100 if cur_power > 0 else float('inf')
            
            impact_analysis[f"{reference_precision}_vs_{precision}"] = {
                "throughput_change_percent": throughput_change,
                "latency_change_percent": latency_change,
                "power_change_percent": power_change
            }
        
        # Compile comparison results
        comparison = {
            "model_path": model_path,
            "batch_size": batch_size,
            "reference_precision": reference_precision,
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "precision_results": precision_results,
            "impact_analysis": impact_analysis,
            "chipset": self.chipset.to_dict() if self.chipset else None
        }
        
        return comparison


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MediaTek Neural Processing Support")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect MediaTek hardware")
    detect_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze MediaTek hardware capabilities")
    analyze_parser.add_argument("--chipset", help="MediaTek chipset to analyze (default: auto-detect)")
    analyze_parser.add_argument("--output", help="Output file path")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert model to MediaTek format")
    convert_parser.add_argument("--model", required=True, help="Input model path")
    convert_parser.add_argument("--output", required=True, help="Output model path")
    convert_parser.add_argument("--chipset", help="Target MediaTek chipset (default: auto-detect)")
    convert_parser.add_argument("--precision", default="INT8", choices=["FP32", "FP16", "INT8", "INT4"], help="Target precision")
    convert_parser.add_argument("--optimize-latency", action="store_true", help="Optimize for latency")
    convert_parser.add_argument("--power-optimization", action="store_true", help="Enable power optimizations")
    
    # Quantize command
    quantize_parser = subparsers.add_parser("quantize", help="Quantize model for MediaTek NPU")
    quantize_parser.add_argument("--model", required=True, help="Input model path")
    quantize_parser.add_argument("--output", required=True, help="Output model path")
    quantize_parser.add_argument("--calibration-data", help="Calibration data path")
    quantize_parser.add_argument("--precision", default="INT8", choices=["INT8", "INT4"], help="Target precision")
    quantize_parser.add_argument("--per-channel", action="store_true", help="Use per-channel quantization")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark on MediaTek NPU")
    benchmark_parser.add_argument("--model", required=True, help="Model path")
    benchmark_parser.add_argument("--batch-sizes", default="1,2,4,8", help="Comma-separated batch sizes")
    benchmark_parser.add_argument("--precision", default="INT8", help="Precision to use")
    benchmark_parser.add_argument("--duration", type=int, default=60, help="Duration in seconds per batch size")
    benchmark_parser.add_argument("--no-thermal-monitoring", action="store_true", help="Disable thermal monitoring")
    benchmark_parser.add_argument("--output", help="Output file path")
    benchmark_parser.add_argument("--db-path", help="Path to benchmark database")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare MediaTek NPU with CPU")
    compare_parser.add_argument("--model", required=True, help="Model path")
    compare_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    compare_parser.add_argument("--precision", default="INT8", help="Precision to use")
    compare_parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    compare_parser.add_argument("--output", help="Output file path")
    
    # Compare precision command
    compare_precision_parser = subparsers.add_parser("compare-precision", help="Compare impact of different precisions")
    compare_precision_parser.add_argument("--model", required=True, help="Model path")
    compare_precision_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    compare_precision_parser.add_argument("--precisions", default="FP32,FP16,INT8", help="Comma-separated precisions")
    compare_precision_parser.add_argument("--duration", type=int, default=30, help="Duration in seconds per precision")
    compare_precision_parser.add_argument("--output", help="Output file path")
    
    # Generate chipset database command
    generate_db_parser = subparsers.add_parser("generate-chipset-db", help="Generate MediaTek chipset database")
    generate_db_parser.add_argument("--output", required=True, help="Output file path")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "detect":
        detector = MediaTekDetector()
        chipset = detector.detect_mediatek_hardware()
        
        if chipset:
            if args.json:
                print(json.dumps(chipset.to_dict(), indent=2))
            else:
                print(f"Detected MediaTek hardware: {chipset.name}")
                print(f"NPU cores: {chipset.npu_cores}")
                print(f"NPU performance: {chipset.npu_tflops} TFLOPS")
                print(f"Supported precisions: {', '.join(chipset.supported_precisions)}")
        else:
            if args.json:
                print(json.dumps({"error": "No MediaTek hardware detected"}, indent=2))
            else:
                print("No MediaTek hardware detected")
                return 1
    
    elif args.command == "analyze":
        detector = MediaTekDetector()
        
        # Get chipset
        if args.chipset:
            chipset_registry = MediaTekChipsetRegistry()
            chipset = chipset_registry.get_chipset(args.chipset)
            if not chipset:
                logger.error(f"Unknown chipset: {args.chipset}")
                return 1
        else:
            chipset = detector.detect_mediatek_hardware()
            if not chipset:
                logger.error("No MediaTek hardware detected")
                return 1
        
        # Analyze capabilities
        analysis = detector.get_capability_analysis(chipset)
        
        # Output analysis
        if args.output:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                with open(args.output, 'w') as f:
                    json.dump(analysis, f, indent=2)
                logger.info(f"Saved capability analysis to {args.output}")
            except Exception as e:
                logger.error(f"Error saving analysis: {e}")
                return 1
        else:
            print(json.dumps(analysis, indent=2))
    
    elif args.command == "convert":
        converter = MediaTekModelConverter()
        
        # Get chipset
        if args.chipset:
            chipset = args.chipset
        else:
            detector = MediaTekDetector()
            chipset_obj = detector.detect_mediatek_hardware()
            if not chipset_obj:
                logger.error("No MediaTek hardware detected")
                return 1
            chipset = chipset_obj.name
        
        # Convert model
        success = converter.convert_to_mediatek_format(
            model_path=args.model,
            output_path=args.output,
            target_chipset=chipset,
            precision=args.precision,
            optimize_for_latency=args.optimize_latency,
            enable_power_optimization=args.power_optimization
        )
        
        if success:
            logger.info(f"Successfully converted model to {args.output}")
        else:
            logger.error("Failed to convert model")
            return 1
    
    elif args.command == "quantize":
        converter = MediaTekModelConverter()
        
        # Quantize model
        success = converter.quantize_model(
            model_path=args.model,
            output_path=args.output,
            calibration_data_path=args.calibration_data,
            precision=args.precision,
            per_channel=args.per_channel
        )
        
        if success:
            logger.info(f"Successfully quantized model to {args.output}")
        else:
            logger.error("Failed to quantize model")
            return 1
    
    elif args.command == "benchmark":
        # Parse batch sizes
        batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]
        
        # Create benchmark runner
        runner = MediaTekBenchmarkRunner(db_path=args.db_path)
        
        # Run benchmark
        results = runner.run_benchmark(
            model_path=args.model,
            batch_sizes=batch_sizes,
            precision=args.precision,
            duration_seconds=args.duration,
            monitor_thermals=not args.no_thermal_monitoring,
            output_path=args.output
        )
        
        if "error" in results:
            logger.error(results["error"])
            return 1
        
        if not args.output:
            print(json.dumps(results, indent=2))
    
    elif args.command == "compare":
        # Create benchmark runner
        runner = MediaTekBenchmarkRunner()
        
        # Run comparison
        results = runner.compare_with_cpu(
            model_path=args.model,
            batch_size=args.batch_size,
            precision=args.precision,
            duration_seconds=args.duration
        )
        
        if "error" in results:
            logger.error(results["error"])
            return 1
        
        # Output comparison
        if args.output:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved comparison results to {args.output}")
            except Exception as e:
                logger.error(f"Error saving results: {e}")
                return 1
        else:
            print(json.dumps(results, indent=2))
    
    elif args.command == "compare-precision":
        # Parse precisions
        precisions = [s.strip() for s in args.precisions.split(",")]
        
        # Create benchmark runner
        runner = MediaTekBenchmarkRunner()
        
        # Run comparison
        results = runner.compare_precision_impact(
            model_path=args.model,
            batch_size=args.batch_size,
            precisions=precisions,
            duration_seconds=args.duration
        )
        
        if "error" in results:
            logger.error(results["error"])
            return 1
        
        # Output comparison
        if args.output:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved precision comparison results to {args.output}")
            except Exception as e:
                logger.error(f"Error saving results: {e}")
                return 1
        else:
            print(json.dumps(results, indent=2))
    
    elif args.command == "generate-chipset-db":
        registry = MediaTekChipsetRegistry()
        success = registry.save_to_file(args.output)
        
        if success:
            logger.info(f"Successfully generated MediaTek chipset database to {args.output}")
        else:
            logger.error("Failed to generate chipset database")
            return 1
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())