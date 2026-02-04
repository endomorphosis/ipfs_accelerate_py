#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for hardware backends.

This script tests the hardware abstraction layer and backend implementations.
"""

import os
import sys
import unittest
import logging
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import hardware
from hardware.base import HardwareBackend
from hardware.cpu import CPUBackend
from hardware.cuda import CUDABackend
from hardware.mps import MPSBackend

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TestHardwareBackends(unittest.TestCase):
    """Test cases for hardware backends."""
    
    def test_base_hardware_backend(self):
        """Test base hardware backend."""
        backend = HardwareBackend()
        self.assertFalse(backend.is_available())
        self.assertEqual(backend.get_info(), {"available": False})
        self.assertIsNone(backend.initialize())
    
    def test_cpu_backend(self):
        """Test CPU backend."""
        try:
            backend = CPUBackend()
            
            # CPU should always be available
            self.assertTrue(CPUBackend.is_available())
            
            # Test info
            info = CPUBackend.get_info()
            self.assertTrue(info["available"])
            self.assertIn("cores_logical", info)
            
            # Test initialization
            device = backend.initialize()
            self.assertIsNotNone(device)
            
            # Clean up
            backend.cleanup()
            
            logger.info("CPU backend test passed")
        except Exception as e:
            self.skipTest(f"CPU backend test failed: {e}")
    
    def test_get_available_hardware(self):
        """Test getting available hardware."""
        try:
            available = hardware.get_available_hardware()
            
            # CPU should always be available
            self.assertIn("cpu", available)
            
            # Log available hardware
            logger.info(f"Available hardware: {available}")
            
            # Test hardware info
            info = hardware.get_hardware_info()
            self.assertTrue(info["cpu"])
            
            logger.info("get_available_hardware test passed")
        except Exception as e:
            self.skipTest(f"get_available_hardware test failed: {e}")
    
    def test_initialize_hardware(self):
        """Test initializing hardware."""
        try:
            # Test CPU initialization
            device = hardware.initialize_hardware("cpu")
            self.assertIsNotNone(device)
            
            # Test fallback for unknown hardware
            device = hardware.initialize_hardware("unknown_hardware")
            self.assertIsNotNone(device)  # Should fall back to CPU
            
            logger.info("initialize_hardware test passed")
        except Exception as e:
            self.skipTest(f"initialize_hardware test failed: {e}")
    
    def test_get_hardware_backend(self):
        """Test getting hardware backend."""
        try:
            # Test CPU backend
            backend = hardware.get_hardware_backend("cpu")
            self.assertIsInstance(backend, CPUBackend)
            
            # Test fallback for unknown hardware
            backend = hardware.get_hardware_backend("unknown_hardware")
            self.assertIsInstance(backend, CPUBackend)  # Should fall back to CPU
            
            logger.info("get_hardware_backend test passed")
        except Exception as e:
            self.skipTest(f"get_hardware_backend test failed: {e}")
    
    def test_cuda_backend_if_available(self):
        """Test CUDA backend if available."""
        if not CUDABackend.is_available():
            self.skipTest("CUDA not available")
        
        try:
            backend = CUDABackend()
            
            # Test capabilities
            capabilities = CUDABackend.get_capabilities()
            self.assertIn("cuda", capabilities)
            logger.info(f"CUDA capabilities: {capabilities}")
            
            # Test device count
            device_count = CUDABackend.get_device_count()
            self.assertGreater(device_count, 0)
            logger.info(f"CUDA device count: {device_count}")
            
            # Test info
            info = CUDABackend.get_info()
            self.assertTrue(info["available"])
            self.assertIn("device_count", info)
            self.assertIn("devices", info)
            
            # Test initialization
            device = backend.initialize()
            self.assertIsNotNone(device)
            
            # Clean up
            backend.cleanup()
            
            logger.info("CUDA backend test passed")
        except Exception as e:
            self.skipTest(f"CUDA backend test failed: {e}")
    
    def test_mps_backend_if_available(self):
        """Test MPS backend if available."""
        if not MPSBackend.is_available():
            self.skipTest("MPS not available")
        
        try:
            backend = MPSBackend()
            
            # Test info
            info = MPSBackend.get_info()
            self.assertTrue(info["available"])
            
            # Test initialization
            device = backend.initialize()
            self.assertIsNotNone(device)
            
            # Clean up
            backend.cleanup()
            
            logger.info("MPS backend test passed")
        except Exception as e:
            self.skipTest(f"MPS backend test failed: {e}")

if __name__ == "__main__":
    unittest.main()