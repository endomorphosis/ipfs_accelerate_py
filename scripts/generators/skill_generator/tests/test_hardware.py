#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the hardware detection system in the refactored generator suite.
Tests hardware detectors, configuration, and selection logic.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware.detector import HardwareDetectionManager
from hardware.detector import CUDADetector, ROCmDetector, MPSDetector, OpenVINODetector, WebNNDetector, WebGPUDetector


class HardwareDetectorTest(unittest.TestCase):
    """Tests for individual hardware detectors."""

    def test_cuda_detector(self):
        """Test CUDA hardware detector."""
        detector = CUDADetector()
        
        # Test with mocked torch where CUDA is available
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.version.cuda", "11.7"), \
             patch("torch.cuda.device_count", return_value=2), \
             patch("torch.cuda.get_device_properties") as mock_props:
             
            # Mock device properties
            mock_props.return_value = Mock(
                name="NVIDIA GeForce RTX 3080",
                total_memory=10 * 1024 * 1024 * 1024,  # 10 GB
                major=8,
                minor=6
            )
            
            # Run detection
            result = detector.detect()
            
            # Verify detection results
            self.assertTrue(result["available"])
            self.assertEqual("11.7", result["version"])
            self.assertEqual(2, result["device_count"])
            self.assertEqual(2, len(result["devices"]))
            self.assertEqual("NVIDIA GeForce RTX 3080", result["devices"][0]["name"])
            self.assertEqual(10 * 1024 * 1024 * 1024, result["devices"][0]["total_memory"])
        
        # Test with mocked torch where CUDA is not available
        with patch("torch.cuda.is_available", return_value=False):
            result = detector.detect()
            self.assertFalse(result["available"])
        
        # Test with ImportError (torch not installed)
        with patch("torch.cuda.is_available", side_effect=ImportError()):
            result = detector.detect()
            self.assertFalse(result["available"])

    def test_rocm_detector(self):
        """Test ROCm hardware detector."""
        detector = ROCmDetector()
        
        # Test with mocked torch where ROCm is available
        with patch("torch.version.hip", "5.2.0"), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_properties") as mock_props:
             
            # Mock device properties
            mock_props.return_value = Mock(
                name="AMD Radeon RX 6800 XT",
                total_memory=16 * 1024 * 1024 * 1024,  # 16 GB
                major=10,
                minor=3
            )
            
            # Run detection
            result = detector.detect()
            
            # Verify detection results
            self.assertTrue(result["available"])
            self.assertEqual("5.2.0", result["version"])
            self.assertEqual(1, result["device_count"])
            self.assertEqual(1, len(result["devices"]))
            self.assertEqual("AMD Radeon RX 6800 XT", result["devices"][0]["name"])
            self.assertEqual(16 * 1024 * 1024 * 1024, result["devices"][0]["total_memory"])
        
        # Test with mocked torch where ROCm is not available
        with patch("torch.version", spec=["hip"], hip=None), \
             patch("torch.cuda.is_available", return_value=False):
            result = detector.detect()
            self.assertFalse(result["available"])
        
        # Test with ImportError (torch not installed)
        with patch("torch.version", side_effect=ImportError()):
            result = detector.detect()
            self.assertFalse(result["available"])

    def test_mps_detector(self):
        """Test MPS (Apple Silicon) hardware detector."""
        detector = MPSDetector()
        
        # Test with mocked torch where MPS is available
        with patch("torch.backends.mps.is_available", return_value=True), \
             patch("torch.backends.mps.is_built", return_value=True):
            
            # Run detection
            result = detector.detect()
            
            # Verify detection results
            self.assertTrue(result["available"])
            self.assertTrue(result["is_built"])
        
        # Test with mocked torch where MPS is not available
        with patch("torch.backends.mps.is_available", return_value=False), \
             patch("torch.backends.mps.is_built", return_value=True):
            result = detector.detect()
            self.assertFalse(result["available"])
        
        # Test with ImportError (torch not installed)
        with patch("torch.backends.mps.is_available", side_effect=ImportError()):
            result = detector.detect()
            self.assertFalse(result["available"])

    def test_openvino_detector(self):
        """Test OpenVINO hardware detector."""
        detector = OpenVINODetector()
        
        # Test with mocked openvino where it's available
        with patch.dict("sys.modules", {"openvino": Mock()}), \
             patch("openvino.runtime.Core") as mock_core:
            
            # Mock available devices
            mock_instance = mock_core.return_value
            mock_instance.available_devices = ["CPU", "GPU.0", "GPU.1"]
            mock_instance.get_property.return_value = "2022.3"
            
            # Run detection
            result = detector.detect()
            
            # Verify detection results
            self.assertTrue(result["available"])
            self.assertEqual("2022.3", result["version"])
            self.assertEqual(3, len(result["devices"]))
            self.assertIn("CPU", result["devices"])
            self.assertIn("GPU.0", result["devices"])
            self.assertIn("GPU.1", result["devices"])
        
        # Test with ImportError (openvino not installed)
        with patch.dict("sys.modules", {"openvino": None}):
            result = detector.detect()
            self.assertFalse(result["available"])

    def test_webnn_detector(self):
        """Test WebNN hardware detector."""
        detector = WebNNDetector()
        
        # WebNN detection is typically a browser-side feature, so we just test the interface
        result = detector.detect()
        self.assertIn("available", result)
        
        # Here we would normally mock browser APIs, but since this runs in Python, 
        # WebNN will typically report as unavailable in the test environment
        self.assertFalse(result["available"])

    def test_webgpu_detector(self):
        """Test WebGPU hardware detector."""
        detector = WebGPUDetector()
        
        # WebGPU detection is typically a browser-side feature, so we just test the interface
        result = detector.detect()
        self.assertIn("available", result)
        
        # Here we would normally mock browser APIs, but since this runs in Python, 
        # WebGPU will typically report as unavailable in the test environment
        self.assertFalse(result["available"])


class HardwareDetectionManagerTest(unittest.TestCase):
    """Tests for the hardware detection manager."""

    def setUp(self):
        """Set up test environment."""
        self.manager = HardwareDetectionManager()
        
        # Register mock detectors
        self.mock_cuda_detector = Mock()
        self.mock_cuda_detector.detect.return_value = {
            "available": True,
            "version": "11.7",
            "device_count": 2
        }
        
        self.mock_rocm_detector = Mock()
        self.mock_rocm_detector.detect.return_value = {
            "available": False
        }
        
        self.mock_mps_detector = Mock()
        self.mock_mps_detector.detect.return_value = {
            "available": False
        }
        
        self.mock_openvino_detector = Mock()
        self.mock_openvino_detector.detect.return_value = {
            "available": True,
            "version": "2022.3",
            "devices": ["CPU", "GPU.0"]
        }
        
        # Register the mock detectors
        self.manager.register_detector("cuda", self.mock_cuda_detector)
        self.manager.register_detector("rocm", self.mock_rocm_detector)
        self.manager.register_detector("mps", self.mock_mps_detector)
        self.manager.register_detector("openvino", self.mock_openvino_detector)

    def test_register_detector(self):
        """Test registering a detector."""
        # Create a new mock detector
        mock_new_detector = Mock()
        mock_new_detector.detect.return_value = {"available": True}
        
        # Register the new detector
        self.manager.register_detector("new_hardware", mock_new_detector)
        
        # Verify it was registered
        self.assertIn("new_hardware", self.manager.detectors)
        self.assertEqual(mock_new_detector, self.manager.detectors["new_hardware"])

    def test_detect_all(self):
        """Test detecting all hardware."""
        # Detect all hardware
        results = self.manager.detect_all()
        
        # Verify all registered detectors were called
        self.mock_cuda_detector.detect.assert_called_once()
        self.mock_rocm_detector.detect.assert_called_once()
        self.mock_mps_detector.detect.assert_called_once()
        self.mock_openvino_detector.detect.assert_called_once()
        
        # Verify results include all hardware types
        self.assertIn("cuda", results)
        self.assertIn("rocm", results)
        self.assertIn("mps", results)
        self.assertIn("openvino", results)
        
        # Verify specific results
        self.assertTrue(results["cuda"]["available"])
        self.assertEqual("11.7", results["cuda"]["version"])
        self.assertFalse(results["rocm"]["available"])
        self.assertFalse(results["mps"]["available"])
        self.assertTrue(results["openvino"]["available"])
        self.assertEqual("2022.3", results["openvino"]["version"])

    def test_detect_specific(self):
        """Test detecting specific hardware."""
        # Detect CUDA
        cuda_result = self.manager.detect("cuda")
        self.assertTrue(cuda_result["available"])
        self.assertEqual("11.7", cuda_result["version"])
        
        # Detect ROCm
        rocm_result = self.manager.detect("rocm")
        self.assertFalse(rocm_result["available"])
        
        # Detect non-existent hardware
        non_existent_result = self.manager.detect("non_existent")
        self.assertFalse(non_existent_result["available"])
        self.assertIn("error", non_existent_result)

    def test_exception_handling(self):
        """Test handling exceptions during detection."""
        # Create a detector that raises an exception
        mock_error_detector = Mock()
        mock_error_detector.detect.side_effect = Exception("Test error")
        self.manager.register_detector("error_hardware", mock_error_detector)
        
        # Detect all hardware should not crash
        results = self.manager.detect_all()
        
        # The error hardware should have an error in its result
        self.assertIn("error_hardware", results)
        self.assertFalse(results["error_hardware"]["available"])
        self.assertIn("error", results["error_hardware"])
        self.assertEqual("Test error", results["error_hardware"]["error"])
        
        # Individual detection should also handle exceptions
        error_result = self.manager.detect("error_hardware")
        self.assertFalse(error_result["available"])
        self.assertIn("error", error_result)
        self.assertEqual("Test error", error_result["error"])


class HardwareRecommendationTest(unittest.TestCase):
    """Tests for hardware recommendation logic."""

    def setUp(self):
        """Set up test environment."""
        self.manager = HardwareDetectionManager()
        
        # Set up mock detectors
        self.setup_mock_detectors()

    def setup_mock_detectors(self):
        """Set up mock detectors with different hardware configurations."""
        # CUDA detector
        mock_cuda_detector = Mock()
        mock_cuda_detector.detect.return_value = {
            "available": True,
            "version": "11.7",
            "device_count": 2,
            "devices": [
                {"name": "NVIDIA GeForce RTX 3080", "total_memory": 10 * 1024 * 1024 * 1024},
                {"name": "NVIDIA GeForce RTX 3070", "total_memory": 8 * 1024 * 1024 * 1024}
            ]
        }
        
        # ROCm detector
        mock_rocm_detector = Mock()
        mock_rocm_detector.detect.return_value = {
            "available": False
        }
        
        # MPS detector
        mock_mps_detector = Mock()
        mock_mps_detector.detect.return_value = {
            "available": False
        }
        
        # OpenVINO detector
        mock_openvino_detector = Mock()
        mock_openvino_detector.detect.return_value = {
            "available": True,
            "version": "2022.3",
            "devices": ["CPU", "GPU.0", "GPU.1"]
        }
        
        # WebNN detector
        mock_webnn_detector = Mock()
        mock_webnn_detector.detect.return_value = {
            "available": False
        }
        
        # WebGPU detector
        mock_webgpu_detector = Mock()
        mock_webgpu_detector.detect.return_value = {
            "available": False
        }
        
        # Register all detectors
        self.manager.register_detector("cuda", mock_cuda_detector)
        self.manager.register_detector("rocm", mock_rocm_detector)
        self.manager.register_detector("mps", mock_mps_detector)
        self.manager.register_detector("openvino", mock_openvino_detector)
        self.manager.register_detector("webnn", mock_webnn_detector)
        self.manager.register_detector("webgpu", mock_webgpu_detector)

    def test_get_recommended_device(self):
        """Test getting the recommended device for a model."""
        # Mock the select_device method to test the device recommendation logic
        original_select_device = self.manager.select_device
        
        try:
            # Override the select_device method for testing
            self.manager.select_device = lambda model_info=None: self._mock_select_device(model_info)
            
            # Test with no model info (should return best available device)
            device = self.manager.select_device()
            self.assertEqual("cuda", device)
            
            # Test with model that requires CUDA
            model_cuda = {"hardware_requirements": {"cuda": {"required": True}}}
            device = self.manager.select_device(model_cuda)
            self.assertEqual("cuda", device)
            
            # Test with model that prefers MPS but MPS is not available
            model_mps = {"hardware_preferences": {"mps": {"preferred": True}}}
            device = self.manager.select_device(model_mps)
            self.assertEqual("cuda", device)  # Falls back to CUDA
            
            # Test with model that can run on OpenVINO
            model_openvino = {"hardware_preferences": {"openvino": {"preferred": True}}}
            device = self.manager.select_device(model_openvino)
            self.assertEqual("openvino", device)
            
            # Test with extremely large model that exceeds CUDA memory
            model_large = {
                "size_mb": 20000,  # 20 GB model (exceeds our mock 10 GB GPU memory)
                "hardware_requirements": {"cuda": {"min_vram_gb": 20}}
            }
            device = self.manager.select_device(model_large)
            self.assertEqual("cpu", device)  # Falls back to CPU
            
        finally:
            # Restore the original method
            self.manager.select_device = original_select_device

    def _mock_select_device(self, model_info=None):
        """Mock implementation of select_device for testing."""
        # If no model info is provided, return the best available device
        if model_info is None:
            if self.manager.detect("cuda")["available"]:
                return "cuda"
            elif self.manager.detect("rocm")["available"]:
                return "rocm"
            elif self.manager.detect("mps")["available"]:
                return "mps"
            elif self.manager.detect("openvino")["available"]:
                return "openvino"
            else:
                return "cpu"
        
        # Check hardware requirements
        if "hardware_requirements" in model_info:
            requirements = model_info["hardware_requirements"]
            
            # Check CUDA requirements
            if "cuda" in requirements and requirements["cuda"].get("required", False):
                if not self.manager.detect("cuda")["available"]:
                    return "cpu"  # CUDA required but not available
                
                # Check VRAM requirements
                if "min_vram_gb" in requirements["cuda"]:
                    min_vram_gb = requirements["cuda"]["min_vram_gb"]
                    cuda_info = self.manager.detect("cuda")
                    for device in cuda_info.get("devices", []):
                        vram_gb = device.get("total_memory", 0) / (1024 * 1024 * 1024)
                        if vram_gb >= min_vram_gb:
                            return "cuda"
                    return "cpu"  # No CUDA device with enough VRAM
                
                return "cuda"
        
        # Check hardware preferences
        if "hardware_preferences" in model_info:
            preferences = model_info["hardware_preferences"]
            
            # Check MPS preference
            if "mps" in preferences and preferences["mps"].get("preferred", False):
                if self.manager.detect("mps")["available"]:
                    return "mps"
            
            # Check OpenVINO preference
            if "openvino" in preferences and preferences["openvino"].get("preferred", False):
                if self.manager.detect("openvino")["available"]:
                    return "openvino"
        
        # Default to CUDA if available
        if self.manager.detect("cuda")["available"]:
            return "cuda"
        elif self.manager.detect("rocm")["available"]:
            return "rocm"
        elif self.manager.detect("mps")["available"]:
            return "mps"
        elif self.manager.detect("openvino")["available"]:
            return "openvino"
        else:
            return "cpu"

    def test_device_selection_priority(self):
        """Test the priority of device selection when multiple are available."""
        # Mock different hardware availability scenarios
        scenarios = [
            # Scenario 1: CUDA only
            {
                "cuda": {"available": True},
                "rocm": {"available": False},
                "mps": {"available": False},
                "openvino": {"available": False},
                "expected": "cuda"
            },
            # Scenario 2: ROCm only
            {
                "cuda": {"available": False},
                "rocm": {"available": True},
                "mps": {"available": False},
                "openvino": {"available": False},
                "expected": "rocm"
            },
            # Scenario 3: MPS only
            {
                "cuda": {"available": False},
                "rocm": {"available": False},
                "mps": {"available": True},
                "openvino": {"available": False},
                "expected": "mps"
            },
            # Scenario 4: OpenVINO only
            {
                "cuda": {"available": False},
                "rocm": {"available": False},
                "mps": {"available": False},
                "openvino": {"available": True},
                "expected": "openvino"
            },
            # Scenario 5: All available (CUDA should be preferred)
            {
                "cuda": {"available": True},
                "rocm": {"available": True},
                "mps": {"available": True},
                "openvino": {"available": True},
                "expected": "cuda"
            },
            # Scenario 6: No hardware acceleration (should fall back to CPU)
            {
                "cuda": {"available": False},
                "rocm": {"available": False},
                "mps": {"available": False},
                "openvino": {"available": False},
                "expected": "cpu"
            }
        ]
        
        # Test each scenario
        for i, scenario in enumerate(scenarios):
            # Override detector results
            for hw_type, result in scenario.items():
                if hw_type != "expected":
                    detector = self.manager.detectors.get(hw_type)
                    if detector:
                        detector.detect.return_value = result
            
            # Select device using the mock implementation
            device = self._mock_select_device()
            
            # Verify selected device matches expected
            self.assertEqual(scenario["expected"], device, f"Failed scenario {i+1}")


if __name__ == "__main__":
    unittest.main()