#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware detection and information example.

This script demonstrates how to use the hardware abstraction layer
to detect and get information about available hardware.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import hardware
from hardware.cpu import CPUBackend
from hardware.cuda import CUDABackend
from hardware.mps import MPSBackend
from hardware.rocm import ROCmBackend
from hardware.openvino import OpenVINOBackend
from hardware.webnn import WebNNBackend
from hardware.webgpu import WebGPUBackend

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def detect_all_hardware():
    """Detect all available hardware platforms."""
    logger.info("Detecting available hardware...")
    
    available = hardware.get_available_hardware()
    logger.info(f"Available hardware: {available}")
    
    return available

def get_all_hardware_info():
    """Get detailed information about all hardware."""
    logger.info("Getting hardware information...")
    
    info = hardware.get_hardware_info()
    
    # Pretty print hardware info
    logger.info(json.dumps(info, indent=2, default=str))
    
    return info

def test_hardware_initialization(hw_name):
    """Test initializing a specific hardware platform."""
    logger.info(f"Testing initialization of {hw_name}...")
    
    try:
        device = hardware.initialize_hardware(hw_name)
        if device:
            logger.info(f"Successfully initialized {hw_name}: {device}")
            return True
        else:
            logger.warning(f"Failed to initialize {hw_name}")
            return False
    except Exception as e:
        logger.error(f"Error initializing {hw_name}: {e}")
        return False

def test_all_hardware():
    """Test all available hardware platforms."""
    logger.info("Testing all available hardware...")
    
    available = detect_all_hardware()
    results = {}
    
    for hw in available:
        results[hw] = test_hardware_initialization(hw)
    
    logger.info("Hardware test results:")
    for hw, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {hw}: {status}")
    
    return results

def main():
    """Main entry point."""
    logger.info("Hardware Abstraction Layer Example")
    logger.info("---------------------------------")
    
    # Detect available hardware
    available = detect_all_hardware()
    
    # Get hardware information
    info = get_all_hardware_info()
    
    # Test initialization
    test_all_hardware()
    
    # Demonstrate direct backend usage
    logger.info("\nDemonstrating direct backend usage:")
    
    # CPU backend (always available)
    cpu_backend = CPUBackend()
    cpu_device = cpu_backend.initialize()
    logger.info(f"Initialized CPU directly: {cpu_device}")
    cpu_backend.cleanup()
    
    # CUDA backend (if available)
    if CUDABackend.is_available():
        cuda_backend = CUDABackend()
        cuda_device = cuda_backend.initialize()
        logger.info(f"Initialized CUDA directly: {cuda_device}")
        cuda_backend.cleanup()
    
    # MPS backend (if available)
    if MPSBackend.is_available():
        mps_backend = MPSBackend()
        mps_device = mps_backend.initialize()
        logger.info(f"Initialized MPS directly: {mps_device}")
        mps_backend.cleanup()
    
    logger.info("\nHardware detection complete!")

if __name__ == "__main__":
    main()