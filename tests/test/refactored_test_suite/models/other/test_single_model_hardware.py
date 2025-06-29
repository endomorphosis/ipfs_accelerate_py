"""Migrated to refactored test suite on 2025-03-21

Test a single model across multiple hardware platforms.

This script focuses on testing a single model across all hardware platforms
to ensure it works correctly on all platforms, with detailed reporting.
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

from refactored_test_suite.hardware_test import HardwareTest
from refactored_test_suite.model_test import ModelTest

# Hardware platforms to test
ALL_HARDWARE_PLATFORMS = ["cpu", "cuda", "mps", "openvino", "rocm", "webnn", "webgpu"]

class TestSingleModelHardware(ModelTest):
    """Test a single model across multiple hardware platforms."""
    
    def setUp(self):
        super().setUp()
        self.model_name = "prajjwal1/bert-tiny"  # Default model
        self.model_path = None
        self.output_dir = "hardware_test_results"
        self.hardware_platforms = ALL_HARDWARE_PLATFORMS
        self.test_results = {}
        
        # Extend hardware detection from base class
        self.detect_extended_hardware()
    
    def detect_extended_hardware(self):
        """Detect additional hardware platforms beyond base class detection."""
        # Start with what the base class detected
        available = {"webgpu": self.has_webgpu, "webnn": self.has_webnn, "cpu": True}
        
        # Check for PyTorch-based platforms
        try:
            import torch
            
            # Check CUDA
            if "cuda" in self.hardware_platforms:
                available["cuda"] = torch.cuda.is_available()
                if available["cuda"]:
                    self.logger.info(f"CUDA is available with {torch.cuda.device_count()} devices")
            
            # Check MPS (Apple Silicon)
            if "mps" in self.hardware_platforms:
                if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_available"):
                    available["mps"] = torch.backends.mps.is_available()
                    if available["mps"]:
                        self.logger.info(f"MPS (Apple Silicon) is available")
                else:
                    available["mps"] = False
                    
            # Check ROCm (AMD)
            if "rocm" in self.hardware_platforms:
                if torch.cuda.is_available() and hasattr(torch.version, "hip"):
                    available["rocm"] = True
                    self.logger.info(f"ROCm (AMD) is available")
                else:
                    available["rocm"] = False
        except ImportError:
            # PyTorch not available
            self.logger.warning("PyTorch not available, CUDA/MPS/ROCm support cannot be detected")
            for platform in ["cuda", "mps", "rocm"]:
                if platform in self.hardware_platforms:
                    available[platform] = False
                    
        # Check OpenVINO
        if "openvino" in self.hardware_platforms:
            try:
                import openvino
                available["openvino"] = True
                self.logger.info(f"OpenVINO is available (version {openvino.__version__})")
            except ImportError:
                available["openvino"] = False
        
        # Store the extended hardware availability
        self.available_hardware = available
    
    def load_model_test_module(self, model_file):
        """
        Load a model test module from a file.
        
        Args:
            model_file: Path to the model test file
            
        Returns:
            Imported module or None if an error occurred
        """
        try:
            import importlib.util
            
            # Get absolute path
            model_file = Path(model_file).absolute()
            
            # Import module
            module_name = os.path.basename(model_file).replace('.py', '')
            spec = importlib.util.spec_from_file_location(module_name, model_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return module
        except Exception as e:
            self.logger.error(f"Error loading module {model_file}: {e}")
            traceback.print_exc()
            return None
    
    def find_test_class(self, module):
        """
        Find the test class in the module.
        
        Args:
            module: Imported module
            
        Returns:
            Test class or None if not found
        """
        if not module:
            return None
        
        # Look for classes that match naming patterns for test classes
        test_class_patterns = ["Test", "TestBase"]
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            if isinstance(attr, type) and any(pattern in attr_name for pattern in test_class_patterns):
                return attr
        
        return None
    
    def test_model_on_platform(self, platform):
        """
        Test a model on a specific platform.
        
        Args:
            platform: Hardware platform to test on
            
        Returns:
            Test results dictionary
        """
        if not self.model_path:
            self.skipTest("No model path specified")
            
        self.logger.info(f"Testing {self.model_name} on {platform}...")
        start_time = time.time()
        
        results = {
            "model": self.model_name,
            "platform": platform,
            "timestamp": datetime.datetime.now().isoformat(),
            "success": False,
            "execution_time": 0
        }
        
        try:
            # Skip if platform is not available (except CPU which is always available)
            if platform != "cpu" and not self.available_hardware.get(platform, False):
                self.logger.warning(f"Platform {platform} not available, skipping test")
                results["success"] = False
                results["error"] = f"Platform {platform} not available"
                results["is_skipped"] = True
                return results
            
            # Load module and find test class
            module = self.load_model_test_module(self.model_path)
            TestClass = self.find_test_class(module)
            
            if not TestClass:
                results["error"] = "Could not find test class in module"
                return results
                
            # Create test instance
            test_instance = TestClass(model_id=self.model_name)
            
            # Run test for the platform
            platform_results = test_instance.run_test(platform)
            
            # Update results
            results["success"] = platform_results.get("success", False)
            results["platform_results"] = platform_results
            results["implementation_type"] = platform_results.get("implementation_type", "UNKNOWN")
            results["is_mock"] = "MOCK" in results.get("implementation_type", "")
            
            # Extract additional information if available
            if "execution_time" in platform_results:
                results["execution_time"] = platform_results["execution_time"]
                
            if "error" in platform_results:
                results["error"] = platform_results["error"]
                
            # Save examples if available
            if hasattr(test_instance, "examples") and test_instance.examples:
                results["examples"] = test_instance.examples
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
            self.logger.error(f"Error testing {self.model_name} on {platform}: {e}")
        
        # Calculate execution time
        results["total_execution_time"] = time.time() - start_time
            
        # Save results if output directory is provided
        if self.output_dir:
            output_dir = Path(self.output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            output_file = output_dir / f"{self.model_name.replace('/', '_')}_{platform}_test.json"
            
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {output_file}")
        
        return results
    
    def run_tests_on_all_platforms(self):
        """Run tests on all specified platforms."""
        results = {}
        
        for platform in self.hardware_platforms:
            result = self.test_model_on_platform(platform)
            results[platform] = result
            
            if result["success"]:
                self.logger.info(f"✅ {platform} test passed")
                
                # Check if implementation is mocked
                if result.get("is_mock", True):
                    self.logger.warning(f"⚠️ {platform} implementation is mocked!")
                else:
                    self.logger.info(f"💯 {platform} implementation is real")
            else:
                self.logger.error(f"❌ {platform} test failed: {result.get('error', 'Unknown error')}")
        
        return results
    
    def generate_report(self, results):
        """Generate a markdown report of the test results."""
        if not self.output_dir:
            return None
            
        output_dir = Path(self.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        report_file = output_dir / f"summary_{self.model_name.replace('/', '_')}.md"
        
        with open(report_file, "w") as f:
            f.write(f"# Hardware Test Report for {self.model_name}\n\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write("## Results Summary\n\n")
            f.write("| Platform | Status | Implementation Type | Execution Time |\n")
            f.write("|----------|--------|---------------------|---------------|\n")
            
            for platform, result in results.items():
                if result["success"]:
                    status = "✅ Passed"
                else:
                    status = "❌ Failed"
                
                impl_type = result.get("implementation_type", "UNKNOWN")
                exec_time = f"{result.get('execution_time', 0):.3f} sec"
                
                f.write(f"| {platform} | {status} | {impl_type} | {exec_time} |\n")
            
            f.write("\n")
            
            # Implementation issues
            failures = [(platform, result) for platform, result in results.items()
                      if not result["success"]]
            
            if failures:
                f.write("## Implementation Issues\n\n")
                for platform, result in failures:
                    f.write(f"### {platform.upper()}\n\n")
                    f.write(f"**Error**: {result.get('error', 'Unknown error')}\n\n")
                    
                    if "traceback" in result:
                        f.write("**Traceback**:\n")
                        f.write("```\n")
                        f.write(result["traceback"])
                        f.write("```\n\n")
                
                f.write("\n")
            
            # Mock implementations
            mocks = [(platform, result) for platform, result in results.items()
                   if result["success"] and result.get("is_mock", True)]
            
            if mocks:
                f.write("## Mock Implementations\n\n")
                for platform, result in mocks:
                    f.write(f"- **{platform}**: {result.get('implementation_type', 'UNKNOWN')}\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if failures:
                f.write("### Fix Implementation Issues\n\n")
                for platform, _ in failures:
                    f.write(f"- Fix {self.model_name} implementation on {platform}\n")
                f.write("\n")
            
            if mocks:
                f.write("### Replace Mock Implementations\n\n")
                for platform, _ in mocks:
                    f.write(f"- Replace mock implementation of {self.model_name} on {platform}\n")
                f.write("\n")
            
            if not failures and not mocks:
                f.write("All implementations are working correctly and are not mocks! 🎉\n\n")
        
        return report_file
    
    def test_model_on_all_platforms(self):
        """Run the model tests on all available platforms."""
        # This is the main test method for the unittest framework
        if not self.model_path:
            self.skipTest("No model path specified")
            
        # Run tests on all platforms
        results = self.run_tests_on_all_platforms()
        
        # Generate report
        report_file = self.generate_report(results)
        
        # Verify if all tests passed
        failures = [(platform, result) for platform, result in results.items()
                  if not result["success"] and not result.get("is_skipped", False)]
        
        # Assert that there are no failures
        self.assertEqual(len(failures), 0, f"Some tests failed: {', '.join(p for p, _ in failures)}")
        
        # Return results for further analysis
        return results
    
    def test_hardware_detection(self):
        """Test the hardware detection capabilities."""
        # Skip this test if running the parent class directly
        if self.__class__ == TestSingleModelHardware:
            self.skipTest("This is a base class test")
            
        # Verify that our hardware detection works
        self.assertIsNotNone(self.available_hardware)
        self.assertTrue(isinstance(self.available_hardware, dict))
        
        # CPU should always be available
        self.assertTrue(self.available_hardware.get("cpu", False))
        
        # WebGPU and WebNN detection should come from the base class
        self.assertEqual(self.available_hardware.get("webgpu", None), self.has_webgpu)
        self.assertEqual(self.available_hardware.get("webnn", None), self.has_webnn)
    
    def set_model_info(self, model_path, model_name, output_dir=None):
        """Set model information for testing."""
        self.model_path = model_path
        self.model_name = model_name or self._infer_model_name(model_path)
        self.output_dir = output_dir or "hardware_test_results"
    
    def _infer_model_name(self, model_path):
        """Infer model name from the test file path."""
        if not model_path:
            return "unknown_model"
            
        model_file = Path(model_path)
        model_type = model_file.stem.replace("test_hf_", "")
        
        # Use a default model for each type
        default_models = {
            "bert": "prajjwal1/bert-tiny",
            "t5": "google/t5-efficient-tiny",
            "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "clip": "openai/clip-vit-base-patch32",
            "vit": "facebook/deit-tiny-patch16-224",
            "clap": "laion/clap-htsat-unfused",
            "whisper": "openai/whisper-tiny",
            "wav2vec2": "facebook/wav2vec2-base",
            "llava": "llava-hf/llava-1.5-7b-hf",
            "llava_next": "llava-hf/llava-v1.6-mistral-7b",
            "xclip": "microsoft/xclip-base-patch32",
            "qwen2": "Qwen/Qwen2-0.5B-Instruct",
            "detr": "facebook/detr-resnet-50"
        }
        
        return default_models.get(model_type, f"unknown_{model_type}_model")

    def test_model_loading(self):
        # Test basic model loading
        if not hasattr(self, 'model_id') or not self.model_id:
            self.skipTest("No model_id specified")
        
        try:
            # Import the appropriate library
            if 'bert' in self.model_id.lower() or 'gpt' in self.model_id.lower() or 't5' in self.model_id.lower():
                import transformers
                model = transformers.AutoModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'clip' in self.model_id.lower():
                import transformers
                model = transformers.CLIPModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'whisper' in self.model_id.lower():
                import transformers
                model = transformers.WhisperModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'wav2vec2' in self.model_id.lower():
                import transformers
                model = transformers.Wav2Vec2Model.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            else:
                # Generic loading
                try:
                    import transformers
                    model = transformers.AutoModel.from_pretrained(self.model_id)
                    self.assertIsNotNone(model, "Model loading failed")
                except:
                    self.skipTest(f"Could not load model {self.model_id} with AutoModel")
        except Exception as e:
            self.fail(f"Model loading failed: {e}")

    def detect_preferred_device(self):
        # Detect available hardware and choose the preferred device
        try:
            import torch
        
            # Check for CUDA
            if torch.cuda.is_available():
                return "cuda"
        
            # Check for MPS (Apple Silicon)
            if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
                return "mps"
        
            # Fallback to CPU
            return "cpu"
        except ImportError:
            return "cpu"