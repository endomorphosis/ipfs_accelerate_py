#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for MediaTek Neural Processing Support

This script implements tests for the MediaTek Neural Processing support module.
It validates the core functionality of chip detection, model conversion, thermal
monitoring, and benchmarking.

Date: April 2025
"""

import os
import sys
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append())))))))))str())))))))))Path())))))))))__file__).resolve())))))))))).parent))

# Import MediaTek support components
try:
    from mediatek_support import ())))))))))
    MediaTekChipset,
    MediaTekChipsetRegistry,
    MediaTekDetector,
    MediaTekModelConverter,
    MediaTekThermalMonitor,
    MediaTekBenchmarkRunner
    )
except ImportError:
    print())))))))))"Error: mediatek_support module could not be imported.")
    sys.exit())))))))))1)

class TestMediaTekChipset())))))))))unittest.TestCase):
    """Tests for the MediaTekChipset class."""
    
    def test_chipset_initialization())))))))))self):
        """Test initializing a MediaTek chipset."""
        chipset = MediaTekChipset())))))))))
        name="Dimensity 9300",
        npu_cores=6,
        npu_tflops=35.7,
        max_precision="FP16",
        supported_precisions=[],"FP32", "FP16", "BF16", "INT8", "INT4"],
        max_power_draw=9.0,
        typical_power=4.0
        )
        
        self.assertEqual())))))))))chipset.name, "Dimensity 9300")
        self.assertEqual())))))))))chipset.npu_cores, 6)
        self.assertAlmostEqual())))))))))chipset.npu_tflops, 35.7)
        self.assertEqual())))))))))chipset.max_precision, "FP16")
        self.assertEqual())))))))))chipset.supported_precisions, [],"FP32", "FP16", "BF16", "INT8", "INT4"]),,
        self.assertEqual())))))))))chipset.max_power_draw, 9.0)
        self.assertEqual())))))))))chipset.typical_power, 4.0)
    
    def test_to_dict())))))))))self):
        """Test converting chipset to dictionary."""
        chipset = MediaTekChipset())))))))))
        name="Dimensity 9300",
        npu_cores=6,
        npu_tflops=35.7,
        max_precision="FP16",
        supported_precisions=[],"FP32", "FP16", "BF16", "INT8", "INT4"],
        max_power_draw=9.0,
        typical_power=4.0
        )
        
        chipset_dict = chipset.to_dict()))))))))))
        
        self.assertEqual())))))))))chipset_dict[],"name"], "Dimensity 9300"),
        self.assertEqual())))))))))chipset_dict[],"npu_cores"], 6),
        self.assertAlmostEqual())))))))))chipset_dict[],"npu_tflops"], 35.7),
        self.assertEqual())))))))))chipset_dict[],"max_precision"], "FP16"),
        self.assertEqual())))))))))chipset_dict[],"supported_precisions"], [],"FP32", "FP16", "BF16", "INT8", "INT4"]),,,
        self.assertEqual())))))))))chipset_dict[],"max_power_draw"], 9.0),
        self.assertEqual())))))))))chipset_dict[],"typical_power"], 4.0)
        ,
    def test_from_dict())))))))))self):
        """Test creating chipset from dictionary."""
        chipset_dict = {}}}}}}}}}}}}}}}
        "name": "Dimensity 9300",
        "npu_cores": 6,
        "npu_tflops": 35.7,
        "max_precision": "FP16",
        "supported_precisions": [],"FP32", "FP16", "BF16", "INT8", "INT4"],
        "max_power_draw": 9.0,
        "typical_power": 4.0
        }
        
        chipset = MediaTekChipset.from_dict())))))))))chipset_dict)
        
        self.assertEqual())))))))))chipset.name, "Dimensity 9300")
        self.assertEqual())))))))))chipset.npu_cores, 6)
        self.assertAlmostEqual())))))))))chipset.npu_tflops, 35.7)
        self.assertEqual())))))))))chipset.max_precision, "FP16")
        self.assertEqual())))))))))chipset.supported_precisions, [],"FP32", "FP16", "BF16", "INT8", "INT4"]),,
        self.assertEqual())))))))))chipset.max_power_draw, 9.0)
        self.assertEqual())))))))))chipset.typical_power, 4.0)


class TestMediaTekChipsetRegistry())))))))))unittest.TestCase):
    """Tests for the MediaTekChipsetRegistry class."""
    
    def setUp())))))))))self):
        """Set up test fixtures."""
        self.registry = MediaTekChipsetRegistry()))))))))))
    
    def test_create_chipset_database())))))))))self):
        """Test creating chipset database."""
        chipsets = self.registry.chipsets
        
        # Check that the chipset database contains expected entries
        self.assertIn())))))))))"dimensity_9300", chipsets)
        self.assertIn())))))))))"dimensity_8300", chipsets)
        self.assertIn())))))))))"dimensity_7300", chipsets)
        self.assertIn())))))))))"dimensity_6300", chipsets)
        self.assertIn())))))))))"helio_g99", chipsets)
        
        # Check that a specific chipset has the correct attributes
        dimensity_9300 = chipsets[],"dimensity_9300"],
        self.assertEqual())))))))))dimensity_9300.name, "Dimensity 9300")
        self.assertEqual())))))))))dimensity_9300.npu_cores, 6)
        self.assertGreater())))))))))dimensity_9300.npu_tflops, 30.0)
        self.assertIn())))))))))"FP16", dimensity_9300.supported_precisions)
        self.assertIn())))))))))"INT8", dimensity_9300.supported_precisions)
    
    def test_get_chipset())))))))))self):
        """Test getting chipset by name."""
        # Test exact match
        chipset = self.registry.get_chipset())))))))))"dimensity_9300")
        self.assertIsNotNone())))))))))chipset)
        self.assertEqual())))))))))chipset.name, "Dimensity 9300")
        
        # Test normalized name
        chipset = self.registry.get_chipset())))))))))"Dimensity 9300")
        self.assertIsNotNone())))))))))chipset)
        self.assertEqual())))))))))chipset.name, "Dimensity 9300")
        
        # Test prefix match
        chipset = self.registry.get_chipset())))))))))"dimensity_9")
        self.assertIsNotNone())))))))))chipset)
        self.assertTrue())))))))))chipset.name.startswith())))))))))"Dimensity 9"))
        
        # Test contains match
        chipset = self.registry.get_chipset())))))))))"9300")
        self.assertIsNotNone())))))))))chipset)
        self.assertEqual())))))))))chipset.name, "Dimensity 9300")
        
        # Test non-existent chipset
        chipset = self.registry.get_chipset())))))))))"non_existent_chipset")
        self.assertIsNone())))))))))chipset)
    
    def test_get_all_chipsets())))))))))self):
        """Test getting all chipsets."""
        chipsets = self.registry.get_all_chipsets()))))))))))
        
        # Check that the list contains multiple chipsets
        self.assertGreater())))))))))len())))))))))chipsets), 5)
        
        # Check that all returned items are MediaTekChipset objects
        for chipset in chipsets:
            self.assertIsInstance())))))))))chipset, MediaTekChipset)
    
    def test_save_and_load_from_file())))))))))self):
        """Test saving and loading chipset database to/from file."""
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile())))))))))suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save registry to file
            success = self.registry.save_to_file())))))))))temp_path)
            self.assertTrue())))))))))success)
            
            # Load registry from file
            loaded_registry = MediaTekChipsetRegistry.load_from_file())))))))))temp_path)
            self.assertIsNotNone())))))))))loaded_registry)
            
            # Compare original and loaded registries
            original_chipsets = self.registry.chipsets
            loaded_chipsets = loaded_registry.chipsets
            
            self.assertEqual())))))))))len())))))))))original_chipsets), len())))))))))loaded_chipsets))
            
            for chipset_name, chipset in original_chipsets.items())))))))))):
                self.assertIn())))))))))chipset_name, loaded_chipsets)
                loaded_chipset = loaded_chipsets[],chipset_name]
                ,
                self.assertEqual())))))))))chipset.name, loaded_chipset.name)
                self.assertEqual())))))))))chipset.npu_cores, loaded_chipset.npu_cores)
                self.assertEqual())))))))))chipset.npu_tflops, loaded_chipset.npu_tflops)
                self.assertEqual())))))))))chipset.max_precision, loaded_chipset.max_precision)
                self.assertEqual())))))))))chipset.supported_precisions, loaded_chipset.supported_precisions)
                self.assertEqual())))))))))chipset.max_power_draw, loaded_chipset.max_power_draw)
                self.assertEqual())))))))))chipset.typical_power, loaded_chipset.typical_power)
        
        finally:
            # Clean up temporary file
            if os.path.exists())))))))))temp_path):
                os.unlink())))))))))temp_path)


class TestMediaTekDetector())))))))))unittest.TestCase):
    """Tests for the MediaTekDetector class."""
    
    def setUp())))))))))self):
        """Set up test fixtures."""
        self.detector = MediaTekDetector()))))))))))
    
    def tearDown())))))))))self):
        """Clean up test fixtures."""
        # Clear environment variables
        for var in [],"TEST_MEDIATEK_CHIPSET", "TEST_PLATFORM"]:,
            if var in os.environ:
                del os.environ[],var]
                ,
    def test_detect_mediatek_hardware_with_env_var())))))))))self):
        """Test detecting MediaTek hardware with environment variable."""
        # Set environment variable to simulate a MediaTek chipset
        os.environ[],"TEST_MEDIATEK_CHIPSET"] = "dimensity_9300"
        ,    ,
        # Detect hardware
        chipset = self.detector.detect_mediatek_hardware()))))))))))
        
        # Check that the hardware was detected
        self.assertIsNotNone())))))))))chipset)
        self.assertEqual())))))))))chipset.name, "Dimensity 9300")
    
    def test_detect_mediatek_hardware_on_android())))))))))self):
        """Test detecting MediaTek hardware on Android."""
        # Set environment variables to simulate Android with MediaTek
        os.environ[],"TEST_PLATFORM"] = "android",
        os.environ[],"TEST_MEDIATEK_CHIPSET"] = "dimensity_8300"
        ,
        # Mock the Android detection methods
        with patch.object())))))))))self.detector, '_is_android', return_value=True):
            with patch.object())))))))))self.detector, '_detect_on_android', return_value="dimensity_8300"):
                # Detect hardware
                chipset = self.detector.detect_mediatek_hardware()))))))))))
                
                # Check that the hardware was detected
                self.assertIsNotNone())))))))))chipset)
                self.assertEqual())))))))))chipset.name, "Dimensity 8300")
    
    def test_detect_mediatek_hardware_no_hardware())))))))))self):
        """Test detecting MediaTek hardware when none is present."""
        # Mock the Android detection methods to return None
        with patch.object())))))))))self.detector, '_is_android', return_value=False):
            # Detect hardware
            chipset = self.detector.detect_mediatek_hardware()))))))))))
            
            # Check that no hardware was detected
            self.assertIsNone())))))))))chipset)
    
    def test_get_capability_analysis())))))))))self):
        """Test getting capability analysis for a chipset."""
        # Get a chipset to analyze
        chipset = self.detector.chipset_registry.get_chipset())))))))))"dimensity_9300")
        self.assertIsNotNone())))))))))chipset)
        
        # Get capability analysis
        analysis = self.detector.get_capability_analysis())))))))))chipset)
        
        # Check that the analysis contains expected sections
        self.assertIn())))))))))"chipset", analysis)
        self.assertIn())))))))))"model_capabilities", analysis)
        self.assertIn())))))))))"precision_support", analysis)
        self.assertIn())))))))))"power_efficiency", analysis)
        self.assertIn())))))))))"recommended_optimizations", analysis)
        self.assertIn())))))))))"competitive_position", analysis)
        
        # Check some specific attributes
        self.assertEqual())))))))))analysis[],"chipset"][],"name"], "Dimensity 9300"),,
        self.assertTrue())))))))))analysis[],"model_capabilities"][],"embedding_models"][],"suitable"]),
        self.assertTrue())))))))))analysis[],"model_capabilities"][],"vision_models"][],"suitable"]),
        self.assertTrue())))))))))analysis[],"model_capabilities"][],"text_generation"][],"suitable"]),
        self.assertTrue())))))))))analysis[],"precision_support"][],"FP16"]),
        self.assertTrue())))))))))analysis[],"precision_support"][],"INT8"])
        ,
        # Flagship chipsets should be suitable for all model types
        for model_type, capability in analysis[],"model_capabilities"].items())))))))))):,
        self.assertTrue())))))))))capability[],"suitable"])

        ,
class TestMediaTekModelConverter())))))))))unittest.TestCase):
    """Tests for the MediaTekModelConverter class."""
    
    def setUp())))))))))self):
        """Set up test fixtures."""
        self.converter = MediaTekModelConverter()))))))))))
        
        # Set environment variable to simulate a MediaTek chipset
        os.environ[],"TEST_MEDIATEK_CHIPSET"] = "dimensity_9300"
        ,
    def tearDown())))))))))self):
        """Clean up test fixtures."""
        # Clear environment variables
        if "TEST_MEDIATEK_CHIPSET" in os.environ:
            del os.environ[],"TEST_MEDIATEK_CHIPSET"]
            ,,
    def test_check_toolchain())))))))))self):
        """Test checking toolchain availability."""
        # With TEST_MEDIATEK_CHIPSET set, toolchain should be considered available
        self.assertTrue())))))))))self.converter._check_toolchain())))))))))))
        
        # When not simulating, it should check if the toolchain path exists:
        with patch.dict())))))))))'os.environ', {}}}}}}}}}}}}}}}}, clear=True):
            with patch())))))))))'os.path.exists', return_value=False):
                self.assertFalse())))))))))self.converter._check_toolchain())))))))))))
            
            with patch())))))))))'os.path.exists', return_value=True):
                self.assertTrue())))))))))self.converter._check_toolchain())))))))))))
    
    def test_convert_to_mediatek_format())))))))))self):
        """Test converting model to MediaTek format."""
        # Create temporary files for testing
        with tempfile.NamedTemporaryFile())))))))))suffix=".onnx", delete=False) as input_file, \
             tempfile.NamedTemporaryFile())))))))))suffix=".npu", delete=False) as output_file:
                 input_path = input_file.name
                 output_path = output_file.name
        
        try:
            # Convert model
            success = self.converter.convert_to_mediatek_format())))))))))
            model_path=input_path,
            output_path=output_path,
            target_chipset="dimensity_9300",
            precision="INT8",
            optimize_for_latency=True,
            enable_power_optimization=True
            )
            
            # Check that conversion was successful
            self.assertTrue())))))))))success)
            
            # Check that output file was created
            self.assertTrue())))))))))os.path.exists())))))))))output_path))
            
            # Check file contents
            with open())))))))))output_path, 'r') as f:
                content = f.read()))))))))))
                self.assertIn())))))))))"MediaTek NPU model for dimensity_9300", content)
                self.assertIn())))))))))f"Original model: {}}}}}}}}}}}}}}}input_path}", content)
                self.assertIn())))))))))"Precision: INT8", content)
                self.assertIn())))))))))"Optimize for latency: True", content)
                self.assertIn())))))))))"Power optimization: True", content)
        
        finally:
            # Clean up temporary files
            for path in [],input_path, output_path]:,,
                if os.path.exists())))))))))path):
                    os.unlink())))))))))path)
    
    def test_quantize_model())))))))))self):
        """Test quantizing model for MediaTek NPU."""
        # Create temporary files for testing
        with tempfile.NamedTemporaryFile())))))))))suffix=".onnx", delete=False) as input_file, \
             tempfile.NamedTemporaryFile())))))))))suffix=".npu", delete=False) as output_file:
                 input_path = input_file.name
                 output_path = output_file.name
        
        try:
            # Quantize model
            success = self.converter.quantize_model())))))))))
            model_path=input_path,
            output_path=output_path,
            calibration_data_path=None,
            precision="INT8",
            per_channel=True
            )
            
            # Check that quantization was successful
            self.assertTrue())))))))))success)
            
            # Check that output file was created
            self.assertTrue())))))))))os.path.exists())))))))))output_path))
            
            # Check file contents
            with open())))))))))output_path, 'r') as f:
                content = f.read()))))))))))
                self.assertIn())))))))))"MediaTek NPU quantized model ())))))))))INT8)", content)
                self.assertIn())))))))))f"Original model: {}}}}}}}}}}}}}}}input_path}", content)
                self.assertIn())))))))))"Per-channel: True", content)
        
        finally:
            # Clean up temporary files
            for path in [],input_path, output_path]:,,
                if os.path.exists())))))))))path):
                    os.unlink())))))))))path)
    
    def test_analyze_model_compatibility())))))))))self):
        """Test analyzing model compatibility with MediaTek NPU."""
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile())))))))))suffix=".onnx", delete=False) as input_file:
            input_path = input_file.name
        
        try:
            # Analyze compatibility
            analysis = self.converter.analyze_model_compatibility())))))))))
            model_path=input_path,
            target_chipset="dimensity_9300"
            )
            
            # Check that analysis contains expected sections
            self.assertIn())))))))))"model_info", analysis)
            self.assertIn())))))))))"chipset_info", analysis)
            self.assertIn())))))))))"compatibility", analysis)
            
            # Check some specific attributes
            self.assertEqual())))))))))analysis[],"chipset_info"][],"name"], "Dimensity 9300"),,
            self.assertTrue())))))))))analysis[],"compatibility"][],"supported"]),
            self.assertIn())))))))))"recommended_precision", analysis[],"compatibility"]),,,
            self.assertIn())))))))))"estimated_performance", analysis[],"compatibility"]),,,
            self.assertIn())))))))))"optimization_opportunities", analysis[],"compatibility"]),,,
            self.assertIn())))))))))"potential_issues", analysis[],"compatibility"]),,,
        
        finally:
            # Clean up temporary file
            if os.path.exists())))))))))input_path):
                os.unlink())))))))))input_path)


class TestMediaTekThermalMonitor())))))))))unittest.TestCase):
    """Tests for the MediaTekThermalMonitor class."""
    
    @patch())))))))))'mobile_thermal_monitoring.MobileThermalMonitor')
    def setUp())))))))))self, mock_base_monitor):
        """Set up test fixtures."""
        # Create a mock for the base monitor
        self.mock_base_monitor = MagicMock()))))))))))
        mock_base_monitor.return_value = self.mock_base_monitor
        
        # Set up thermal zones dictionary
        self.mock_base_monitor.thermal_zones = {}}}}}}}}}}}}}}}}
        
        # Set up throttling manager
        self.mock_throttling_manager = MagicMock()))))))))))
        self.mock_base_monitor.throttling_manager = self.mock_throttling_manager
        
        # Create thermal monitor
        from mobile_thermal_monitoring import ThermalZone
        with patch())))))))))'os.path.exists', return_value=False), \
             patch())))))))))'mobile_thermal_monitoring.ThermalZone', side_effect=lambda **kwargs: MagicMock())))))))))**kwargs)):
                 self.thermal_monitor = MediaTekThermalMonitor())))))))))device_type="android")
    
    def test_initialization())))))))))self):
        """Test initializing MediaTek thermal monitor."""
        # Check that the base monitor was initialized
        self.assertIsNotNone())))))))))self.thermal_monitor.base_monitor)
        
        # Check that MediaTek-specific thermal zones were added
        self.assertIn())))))))))"apu", self.mock_base_monitor.thermal_zones)
        
        # Check that MediaTek-specific cooling policy was set
        self.mock_base_monitor.configure_cooling_policy.assert_called_once()))))))))))
    
    def test_start_stop_monitoring())))))))))self):
        """Test starting and stopping thermal monitoring."""
        # Start monitoring
        self.thermal_monitor.start_monitoring()))))))))))
        self.mock_base_monitor.start_monitoring.assert_called_once()))))))))))
        
        # Stop monitoring
        self.thermal_monitor.stop_monitoring()))))))))))
        self.mock_base_monitor.stop_monitoring.assert_called_once()))))))))))
    
    def test_get_current_thermal_status())))))))))self):
        """Test getting current thermal status."""
        # Mock base monitor's get_current_thermal_status method
        status = {}}}}}}}}}}}}}}}
        "device_type": "android",
        "overall_status": "NORMAL",
        "thermal_zones": {}}}}}}}}}}}}}}}
        "cpu": {}}}}}}}}}}}}}}}"current_temp": 60.0},
        "gpu": {}}}}}}}}}}}}}}}"current_temp": 55.0}
        }
        }
        self.mock_base_monitor.get_current_thermal_status.return_value = status
        
        # Set up APU thermal zone
        self.mock_base_monitor.thermal_zones[],"apu"] = MagicMock())))))))))),
        self.mock_base_monitor.thermal_zones[],"apu"].current_temp = 65.0
        ,
        # Get thermal status
        thermal_status = self.thermal_monitor.get_current_thermal_status()))))))))))
        
        # Check that base status was returned with MediaTek-specific additions
        self.assertEqual())))))))))thermal_status[],"device_type"], "android"),
        self.assertEqual())))))))))thermal_status[],"overall_status"], "NORMAL"),
        self.assertIn())))))))))"apu_temperature", thermal_status)
        self.assertEqual())))))))))thermal_status[],"apu_temperature"], 65.0)
        ,
    def test_get_recommendations())))))))))self):
        """Test getting MediaTek-specific thermal recommendations."""
        # Mock base monitor's _generate_recommendations method
        base_recommendations = [],"STATUS OK: All thermal zones within normal operating temperatures."],
        self.mock_base_monitor._generate_recommendations.return_value = base_recommendations
        
        # Set up APU thermal zone with elevated temperature
        from mobile_thermal_monitoring import ThermalZone
        apu_zone = MagicMock()))))))))))
        apu_zone.current_temp = 80.0
        apu_zone.warning_temp = 75.0
        apu_zone.critical_temp = 90.0
        self.mock_base_monitor.thermal_zones[],"apu"] = apu_zone
        ,
        # Get recommendations
        recommendations = self.thermal_monitor.get_recommendations()))))))))))
        
        # Check that base recommendations were returned with MediaTek-specific additions
        self.assertEqual())))))))))len())))))))))recommendations), 2)  # Base recommendation + MediaTek-specific
        self.assertEqual())))))))))recommendations[],0], "STATUS OK: All thermal zones within normal operating temperatures."),
        self.assertIn())))))))))"MEDIATEK: APU temperature", recommendations[],1]),
        self.assertIn())))))))))"is elevated", recommendations[],1]),


class TestMediaTekBenchmarkRunner())))))))))unittest.TestCase):
    """Tests for the MediaTekBenchmarkRunner class."""
    
    def setUp())))))))))self):
        """Set up test fixtures."""
        # Set environment variable to simulate a MediaTek chipset
        os.environ[],"TEST_MEDIATEK_CHIPSET"] = "dimensity_9300"
        ,    ,
        # Create benchmark runner
        with patch())))))))))'mediatek_support.MediaTekDetector.detect_mediatek_hardware') as mock_detect:
            # Mock the detect_mediatek_hardware method to return a chipset
            chipset = MediaTekChipset())))))))))
            name="Dimensity 9300",
            npu_cores=6,
            npu_tflops=35.7,
            max_precision="FP16",
            supported_precisions=[],"FP32", "FP16", "BF16", "INT8", "INT4"],
            max_power_draw=9.0,
            typical_power=4.0
            )
            mock_detect.return_value = chipset
            
            self.benchmark_runner = MediaTekBenchmarkRunner()))))))))))
    
    def tearDown())))))))))self):
        """Clean up test fixtures."""
        # Clear environment variables
        if "TEST_MEDIATEK_CHIPSET" in os.environ:
            del os.environ[],"TEST_MEDIATEK_CHIPSET"]
            ,,
    def test_initialization())))))))))self):
        """Test initializing MediaTek benchmark runner."""
        # Check that chipset was detected
        self.assertIsNotNone())))))))))self.benchmark_runner.chipset)
        self.assertEqual())))))))))self.benchmark_runner.chipset.name, "Dimensity 9300")
    
        @patch())))))))))'mediatek_support.MediaTekThermalMonitor')
    def test_run_benchmark())))))))))self, mock_thermal_monitor):
        """Test running benchmark on MediaTek NPU."""
        # Mock thermal monitor
        mock_thermal_monitor_instance = MagicMock()))))))))))
        mock_thermal_monitor.return_value = mock_thermal_monitor_instance
        mock_thermal_monitor_instance.get_current_thermal_status.return_value = {}}}}}}}}}}}}}}}
        "thermal_zones": {}}}}}}}}}}}}}}}
        "cpu": {}}}}}}}}}}}}}}}"current_temp": 60.0},
        "gpu": {}}}}}}}}}}}}}}}"current_temp": 55.0}
        },
        "apu_temperature": 65.0
        }
        mock_thermal_monitor_instance.get_recommendations.return_value = [],
        "MEDIATEK: APU temperature ())))))))))65.0°C) is elevated. Consider using INT8 quantization to reduce power."
        ]
        
        # Create temporary files for testing
        with tempfile.NamedTemporaryFile())))))))))suffix=".onnx", delete=False) as input_file, \
             tempfile.NamedTemporaryFile())))))))))suffix=".json", delete=False) as output_file:
                 input_path = input_file.name
                 output_path = output_file.name
        
        try:
            # Run benchmark
            results = self.benchmark_runner.run_benchmark())))))))))
            model_path=input_path,
            batch_sizes=[],1, 2, 4],
            precision="INT8",
            duration_seconds=1,  # Short duration for testing
            monitor_thermals=True,
            output_path=output_path
            )
            
            # Check that benchmark results contain expected sections
            self.assertIn())))))))))"model_path", results)
            self.assertEqual())))))))))results[],"model_path"], input_path)
            self.assertEqual())))))))))results[],"precision"], "INT8")
            self.assertIn())))))))))"chipset", results)
            self.assertIn())))))))))"batch_results", results)
            self.assertIn())))))))))"system_info", results)
            self.assertIn())))))))))"thermal_recommendations", results)
            
            # Check batch results
            batch_results = results[],"batch_results"]
            self.assertEqual())))))))))len())))))))))batch_results), 3)  # 3 batch sizes
            for batch_size in [],1, 2, 4]:
                self.assertIn())))))))))batch_size, batch_results)
                self.assertIn())))))))))"throughput_items_per_second", batch_results[],batch_size])
                self.assertIn())))))))))"latency_ms", batch_results[],batch_size])
                self.assertIn())))))))))"power_metrics", batch_results[],batch_size])
                self.assertIn())))))))))"memory_metrics", batch_results[],batch_size])
                self.assertIn())))))))))"temperature_metrics", batch_results[],batch_size])
            
            # Check that thermal monitor was used correctly
                mock_thermal_monitor.assert_called_once_with())))))))))device_type="android")
                mock_thermal_monitor_instance.start_monitoring.assert_called_once()))))))))))
                mock_thermal_monitor_instance.stop_monitoring.assert_called_once()))))))))))
            
            # Check that results were saved to file
                self.assertTrue())))))))))os.path.exists())))))))))output_path))
            with open())))))))))output_path, 'r') as f:
                saved_results = json.load())))))))))f)
                self.assertEqual())))))))))saved_results[],"model_path"], input_path)
                self.assertEqual())))))))))saved_results[],"precision"], "INT8")
                self.assertEqual())))))))))len())))))))))saved_results[],"batch_results"]), 3)
        
        finally:
            # Clean up temporary files
            for path in [],input_path, output_path]:,,
                if os.path.exists())))))))))path):
                    os.unlink())))))))))path)
    
    def test_compare_with_cpu())))))))))self):
        """Test comparing MediaTek NPU performance with CPU."""
        # Mock run_benchmark method to return predictable results
        with patch.object())))))))))self.benchmark_runner, 'run_benchmark') as mock_run_benchmark:
            # Set up mock results
            mock_results = {}}}}}}}}}}}}}}}
            "batch_results": {}}}}}}}}}}}}}}}
            1: {}}}}}}}}}}}}}}}
            "throughput_items_per_second": 100.0,
            "latency_ms": {}}}}}}}}}}}}}}}
            "avg": 10.0
            },
            "power_metrics": {}}}}}}}}}}}}}}}
            "power_consumption_mw": 2000.0
            }
            }
            }
            }
            mock_run_benchmark.return_value = mock_results
            
            # Create a temporary file for testing
            with tempfile.NamedTemporaryFile())))))))))suffix=".onnx", delete=False) as input_file:
                input_path = input_file.name
            
            try:
                # Run comparison
                comparison = self.benchmark_runner.compare_with_cpu())))))))))
                model_path=input_path,
                batch_size=1,
                precision="INT8",
                duration_seconds=1  # Short duration for testing
                )
                
                # Check that comparison contains expected sections
                self.assertIn())))))))))"model_path", comparison)
                self.assertEqual())))))))))comparison[],"model_path"], input_path)
                self.assertEqual())))))))))comparison[],"batch_size"], 1)
                self.assertEqual())))))))))comparison[],"precision"], "INT8")
                self.assertIn())))))))))"npu", comparison)
                self.assertIn())))))))))"cpu", comparison)
                self.assertIn())))))))))"speedups", comparison)
                
                # Check NPU and CPU metrics
                self.assertEqual())))))))))comparison[],"npu"][],"throughput_items_per_second"], 100.0)
                self.assertEqual())))))))))comparison[],"npu"][],"latency_ms"], 10.0)
                self.assertEqual())))))))))comparison[],"npu"][],"power_consumption_mw"], 2000.0)
                
                # CPU metrics should be derived from NPU metrics
                self.assertEqual())))))))))comparison[],"cpu"][],"throughput_items_per_second"], 10.0)  # 10x slower
                self.assertEqual())))))))))comparison[],"cpu"][],"latency_ms"], 100.0)  # 10x higher
                self.assertEqual())))))))))comparison[],"cpu"][],"power_consumption_mw"], 3000.0)  # 1.5x more power
                
                # Check speedups
                self.assertEqual())))))))))comparison[],"speedups"][],"throughput"], 10.0)
                self.assertEqual())))))))))comparison[],"speedups"][],"latency"], 10.0)
                self.assertEqual())))))))))comparison[],"speedups"][],"power_efficiency"], 15.0)
            
            finally:
                # Clean up temporary file
                if os.path.exists())))))))))input_path):
                    os.unlink())))))))))input_path)
    
    def test_compare_precision_impact())))))))))self):
        """Test comparing impact of different precisions on MediaTek NPU performance."""
        # Mock run_benchmark method to return predictable results
        with patch.object())))))))))self.benchmark_runner, 'run_benchmark') as mock_run_benchmark:
            # Set up mock results for different precisions
            def mock_run_benchmark_side_effect())))))))))model_path, batch_sizes, precision, **kwargs):
                base_throughput = 100.0
                base_latency = 10.0
                base_power = 2000.0
                
                if precision == "FP32":
                    throughput_factor = 0.5
                    latency_factor = 2.0
                    power_factor = 1.5
                elif precision == "FP16":
                    throughput_factor = 1.0
                    latency_factor = 1.0
                    power_factor = 1.0
                elif precision == "INT8":
                    throughput_factor = 2.0
                    latency_factor = 0.5
                    power_factor = 0.8
                else:
                    throughput_factor = 1.0
                    latency_factor = 1.0
                    power_factor = 1.0
                
                    batch_size = batch_sizes[],0]
                    return {}}}}}}}}}}}}}}}
                    "batch_results": {}}}}}}}}}}}}}}}
                    batch_size: {}}}}}}}}}}}}}}}
                    "throughput_items_per_second": base_throughput * throughput_factor,
                    "latency_ms": {}}}}}}}}}}}}}}}
                    "avg": base_latency * latency_factor
                    },
                    "power_metrics": {}}}}}}}}}}}}}}}
                    "power_consumption_mw": base_power * power_factor
                    }
                    }
                    }
                    }
            
                    mock_run_benchmark.side_effect = mock_run_benchmark_side_effect
            
            # Create a temporary file for testing
            with tempfile.NamedTemporaryFile())))))))))suffix=".onnx", delete=False) as input_file:
                input_path = input_file.name
            
            try:
                # Run comparison
                comparison = self.benchmark_runner.compare_precision_impact())))))))))
                model_path=input_path,
                batch_size=1,
                precisions=[],"FP16", "INT8"],
                duration_seconds=1  # Short duration for testing
                )
                
                # Check that comparison contains expected sections
                self.assertIn())))))))))"model_path", comparison)
                self.assertEqual())))))))))comparison[],"model_path"], input_path)
                self.assertEqual())))))))))comparison[],"batch_size"], 1)
                self.assertEqual())))))))))comparison[],"reference_precision"], "FP16")
                self.assertIn())))))))))"precision_results", comparison)
                self.assertIn())))))))))"impact_analysis", comparison)
                
                # Check precision results
                precision_results = comparison[],"precision_results"]
                self.assertIn())))))))))"FP16", precision_results)
                self.assertIn())))))))))"INT8", precision_results)
                
                # Check impact analysis
                impact_analysis = comparison[],"impact_analysis"]
                self.assertIn())))))))))"FP16_vs_INT8", impact_analysis)
                
                # Check specific impact
                fp16_vs_int8 = impact_analysis[],"FP16_vs_INT8"]
                self.assertIn())))))))))"throughput_change_percent", fp16_vs_int8)
                self.assertIn())))))))))"latency_change_percent", fp16_vs_int8)
                self.assertIn())))))))))"power_change_percent", fp16_vs_int8)
                
                # INT8 should show improvement over FP16
                self.assertGreater())))))))))fp16_vs_int8[],"throughput_change_percent"], 0)  # Better throughput
                self.assertGreater())))))))))fp16_vs_int8[],"latency_change_percent"], 0)  # Better latency
                self.assertGreater())))))))))fp16_vs_int8[],"power_change_percent"], 0)  # Better power efficiency
            
            finally:
                # Clean up temporary file
                if os.path.exists())))))))))input_path):
                    os.unlink())))))))))input_path)


if __name__ == "__main__":
    unittest.main()))))))))))