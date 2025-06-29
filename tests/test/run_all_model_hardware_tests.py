#!/usr/bin/env python3
"""
Comprehensive model hardware test runner.

This script tests all 13 key model classes across all hardware platforms,
with proper error handling and reporting.
"""

import os
import sys
import json
import logging
import argparse
import datetime
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig())
level=logging.INFO,
format='%())asctime)s - %())levelname)s - %())message)s'
)
logger = logging.getLogger())__name__)

# Define the 13 high priority model classes
KEY_MODELS = {}}}}}}}}}}}}
"bert": "bert-base-uncased",
"t5": "t5-small",
"llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
"clip": "openai/clip-vit-base-patch32",
"vit": "google/vit-base-patch16-224",
"clap": "laion/clap-htsat-unfused",
"whisper": "openai/whisper-tiny",
"wav2vec2": "facebook/wav2vec2-base",
"llava": "llava-hf/llava-1.5-7b-hf",
"llava_next": "llava-hf/llava-v1.6-mistral-7b",
"xclip": "microsoft/xclip-base-patch32",
"qwen2": "Qwen/Qwen2-0.5B-Instruct",
"detr": "facebook/detr-resnet-50"
}

# Smaller versions for testing
SMALL_VERSIONS = {}}}}}}}}}}}}
"bert": "prajjwal1/bert-tiny",
"t5": "google/t5-efficient-tiny",
"vit": "facebook/deit-tiny-patch16-224",
"whisper": "openai/whisper-tiny",
"llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
"qwen2": "Qwen/Qwen2-0.5B-Instruct"
}

# All hardware platforms to test
ALL_HARDWARE_PLATFORMS = []],,"cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]
,
class ModelHardwareTester:
    """Tests all key models across all hardware platforms."""
    
    def __init__())self, 
    output_dir: str = "./hardware_test_results",
    use_small_models: bool = True,
    hardware_platforms: list = None,
                models_dir: str = None):
                    """
                    Initialize the tester.
        
        Args:
            output_dir: Directory to save test results
            use_small_models: Use smaller model variants when available
            hardware_platforms: List of hardware platforms to test, or None for all
            models_dir: Directory containing model test files
            """
            self.output_dir = Path())output_dir)
            self.output_dir.mkdir())exist_ok=True, parents=True)
        
            self.use_small_models = use_small_models
            self.hardware_platforms = hardware_platforms or ALL_HARDWARE_PLATFORMS
        
        # Try to find model files directory
        if models_dir:
            self.models_dir = Path())models_dir)
        else:
            # Try common locations
            possible_dirs = []],,
            "./updated_models",
            "./key_models_hardware_fixes",
            "./modality_tests"
            ]
            
            for dir_path in possible_dirs:
                if os.path.isdir())dir_path):
                    self.models_dir = Path())dir_path)
                    logger.info())f"Using models directory: {}}}}}}}}}}}}self.models_dir}")
                break
            else:
                # If no directory found, use current directory
                self.models_dir = Path())".")
                logger.warning())f"No models directory found, using current directory")
        
        # Set up results tracking
                self.timestamp = datetime.datetime.now())).strftime())"%Y%m%d_%H%M%S")
                self.results = {}}}}}}}}}}}}
                "timestamp": self.timestamp,
                "models_tested": {}}}}}}}}}}}}},
                "hardware_platforms": self.hardware_platforms,
                "test_results": {}}}}}}}}}}}}},
                "summary": {}}}}}}}}}}}}}
                }
        
        # Detect available hardware
                self.available_hardware = self._detect_hardware()))
    
    def _detect_hardware())self):
        """Detect available hardware platforms."""
        logger.info())"Detecting available hardware platforms...")
        
        available = {}}}}}}}}}}}}"cpu": True}  # CPU is always available
        
        # Check for CUDA ())NVIDIA) support
        try:
            import torch
            available[]],,"cuda"] = torch.cuda.is_available()))
            if available[]],,"cuda"]:
                logger.info())f"CUDA is available with {}}}}}}}}}}}}torch.cuda.device_count()))} devices")
            
            # Check for MPS ())Apple Silicon) support
            if hasattr())torch.backends, "mps") and hasattr())torch.backends.mps, "is_available"):
                available[]],,"mps"] = torch.backends.mps.is_available()))
                if available[]],,"mps"]:
                    logger.info())"MPS ())Apple Silicon) is available")
            else:
                available[]],,"mps"] = False
            
            # Check for ROCm ())AMD) support
            if torch.cuda.is_available())) and hasattr())torch.version, "hip"):
                available[]],,"rocm"] = True
                logger.info())"ROCm ())AMD) is available")
            else:
                available[]],,"rocm"] = False
        except ImportError:
            logger.warning())"PyTorch not available, CUDA/MPS/ROCm cannot be detected")
            available[]],,"cuda"] = False
            available[]],,"mps"] = False
            available[]],,"rocm"] = False
        
        # Check for OpenVINO
        try:
            import openvino
            available[]],,"openvino"] = True
            logger.info())f"OpenVINO is available ())version {}}}}}}}}}}}}openvino.__version__})")
        except ImportError:
            available[]],,"openvino"] = False
        
        # WebNN and WebGPU can be simulated, so mark as available
        # In real browser tests, these would be conditionally available
            available[]],,"webnn"] = True
            available[]],,"webgpu"] = True
            logger.info())"WebNN and WebGPU will be tested in simulation mode")
        
        # Filter to include only requested platforms
        return {}}}}}}}}}}}}hw: available.get())hw, False) for hw in self.hardware_platforms}:
    def get_model_files())self):
        """Find all model test files."""
        model_files = {}}}}}}}}}}}}}
        
        for model_key in KEY_MODELS.keys())):
            filename = f"test_hf_{}}}}}}}}}}}}model_key}.py"
            filepath = self.models_dir / filename
            
            if filepath.exists())):
                model_files[]],,model_key] = filepath
                logger.debug())f"Found model file for {}}}}}}}}}}}}model_key}: {}}}}}}}}}}}}filepath}")
            else:
                logger.warning())f"Model file for {}}}}}}}}}}}}model_key} not found at {}}}}}}}}}}}}filepath}")
        
                logger.info())f"Found {}}}}}}}}}}}}len())model_files)} model test files out of {}}}}}}}}}}}}len())KEY_MODELS)} key models")
                return model_files
    
    def run_test())self, model_key, model_file, platform):
        """Run a test for a specific model on a specific platform."""
        logger.info())f"Testing {}}}}}}}}}}}}model_key} on {}}}}}}}}}}}}platform} platform...")
        
        # Use smaller model variant if available and requested:
        if self.use_small_models and model_key in SMALL_VERSIONS:
            model_name = SMALL_VERSIONS[]],,model_key]
        else:
            model_name = KEY_MODELS[]],,model_key]
        
        # Create output directory for this run
            run_dir = self.output_dir / f"run_{}}}}}}}}}}}}self.timestamp}"
            run_dir.mkdir())exist_ok=True)
        
        # Prepare command to run the test
            cmd = []],,
            sys.executable,
            "run_hardware_tests.py",
            "--models", model_key,
            "--platforms", platform,
            "--output-dir", str())run_dir),
            "--models-dir", str())self.models_dir)
            ]
        
        # Add model name if not default:
        if model_name != KEY_MODELS[]],,model_key]:
            cmd.extend())[]],,"--model-names", model_name])
        
        # Run the command
        try:
            logger.debug())f"Running: {}}}}}}}}}}}}' '.join())cmd)}")
            result = subprocess.run())cmd, check=False, capture_output=True, text=True)
            
            # Check for test result file
            result_file = run_dir / f"{}}}}}}}}}}}}model_key}_{}}}}}}}}}}}}platform.lower()))}_test.json"
            
            if result_file.exists())):
                with open())result_file, "r") as f:
                    test_result = json.load())f)
                
                # Store result
                return {}}}}}}}}}}}}
                "status": "success",
                "result_file": str())result_file),
                "test_result": test_result
                }
            else:
                # Test failed to produce output file
                return {}}}}}}}}}}}}
                "status": "error",
                "error": "No test result file produced",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
                }
        except Exception as e:
            # Test execution failed
                return {}}}}}}}}}}}}
                "status": "error",
                "error": str())e),
                "exception_type": type())e).__name__
                }
    
    def analyze_result())self, model_key, platform, test_result):
        """Analyze a test result to determine success status."""
        if test_result[]],,"status"] == "error":
        return {}}}}}}}}}}}}
        "success": False,
        "error": test_result.get())"error", "Unknown error"),
        "details": test_result
        }
        
        # Check if test ran successfully
        result_data = test_result.get())"test_result", {}}}}}}}}}}}}})
        :
        if "results" in result_data and platform in result_data[]],,"results"]:
            platform_result = result_data[]],,"results"][]],,platform]
            
            # Check success flag
            success = platform_result.get())"success", False)
            
            # Check for implementation type ())mock vs real)
            impl_type = platform_result.get())"implementation_type", "UNKNOWN")
            is_mock = "MOCK" in impl_type
            
            return {}}}}}}}}}}}}
            "success": success,
            "implementation_type": impl_type,
            "is_mock": is_mock,
            "execution_time": platform_result.get())"execution_time", 0),
            "details": platform_result
            }
        else:
            return {}}}}}}}}}}}}
            "success": False,
            "error": "No platform results found in test output",
            "details": result_data
            }
    
    def run_all_tests())self):
        """Run tests for all models on all platforms."""
        logger.info())"Starting comprehensive hardware testing for all key models...")
        
        # Get all model files
        model_files = self.get_model_files()))
        
        # Initialize results structure
        all_results = {}}}}}}}}}}}}}
        summary = {}}}}}}}}}}}}
        "total_tests": 0,
        "successful_tests": 0,
        "failed_tests": 0,
        "mock_implementations": 0,
        "real_implementations": 0,
        "by_platform": {}}}}}}}}}}}}},
        "by_model": {}}}}}}}}}}}}}
        }
        
        # Initialize platform and model summaries
        for platform in self.hardware_platforms:
            summary[]],,"by_platform"][]],,platform] = {}}}}}}}}}}}}
            "total": 0,
            "success": 0,
            "failure": 0,
            "mock": 0,
            "real": 0
            }
        
        for model_key in KEY_MODELS.keys())):
            summary[]],,"by_model"][]],,model_key] = {}}}}}}}}}}}}
            "total": 0,
            "success": 0,
            "failure": 0,
            "mock": 0,
            "real": 0
            }
        
        # Run tests for each model on each platform
        for model_key, model_file in model_files.items())):
            all_results[]],,model_key] = {}}}}}}}}}}}}}
            
            for platform in self.hardware_platforms:
                # Skip if hardware not available:
                if platform != "cpu" and not self.available_hardware.get())platform, False):
                    logger.info())f"Skipping {}}}}}}}}}}}}model_key} on {}}}}}}}}}}}}platform} ())hardware not available)")
                continue
                
                # Run the test
                test_result = self.run_test())model_key, model_file, platform)
                all_results[]],,model_key][]],,platform] = test_result
                
                # Analyze the result
                analysis = self.analyze_result())model_key, platform, test_result)
                all_results[]],,model_key][]],,platform][]],,"analysis"] = analysis
                
                # Update summary
                summary[]],,"total_tests"] += 1
                summary[]],,"by_platform"][]],,platform][]],,"total"] += 1
                summary[]],,"by_model"][]],,model_key][]],,"total"] += 1
                
                if analysis[]],,"success"]:
                    summary[]],,"successful_tests"] += 1
                    summary[]],,"by_platform"][]],,platform][]],,"success"] += 1
                    summary[]],,"by_model"][]],,model_key][]],,"success"] += 1
                    
                    if analysis.get())"is_mock", True):
                        summary[]],,"mock_implementations"] += 1
                        summary[]],,"by_platform"][]],,platform][]],,"mock"] += 1
                        summary[]],,"by_model"][]],,model_key][]],,"mock"] += 1
                    else:
                        summary[]],,"real_implementations"] += 1
                        summary[]],,"by_platform"][]],,platform][]],,"real"] += 1
                        summary[]],,"by_model"][]],,model_key][]],,"real"] += 1
                else:
                    summary[]],,"failed_tests"] += 1
                    summary[]],,"by_platform"][]],,platform][]],,"failure"] += 1
                    summary[]],,"by_model"][]],,model_key][]],,"failure"] += 1
        
        # Store results
                    self.results[]],,"test_results"] = all_results
                    self.results[]],,"summary"] = summary
        
        # Generate report
                    report_path = self.generate_report()))
        
                    logger.info())f"Completed comprehensive hardware testing.")
                    logger.info())f"Total tests: {}}}}}}}}}}}}summary[]],,'total_tests']}")
                    logger.info())f"Successful tests: {}}}}}}}}}}}}summary[]],,'successful_tests']}")
                    logger.info())f"Failed tests: {}}}}}}}}}}}}summary[]],,'failed_tests']}")
                    logger.info())f"Mock implementations: {}}}}}}}}}}}}summary[]],,'mock_implementations']}")
                    logger.info())f"Real implementations: {}}}}}}}}}}}}summary[]],,'real_implementations']}")
                    logger.info())f"Report saved to: {}}}}}}}}}}}}report_path}")
        
                        return self.results
    
    def generate_report())self):
        """Generate a report of the test results."""
        logger.info())"Generating hardware test report...")
        
        # Create report file
        report_file = self.output_dir / f"hardware_test_report_{}}}}}}}}}}}}self.timestamp}.md"
        
        with open())report_file, "w") as f:
            # Header
            f.write())"# Key Models Hardware Test Report\n\n")
            f.write())f"Generated: {}}}}}}}}}}}}datetime.datetime.now())).strftime())'%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            summary = self.results[]],,"summary"]
            f.write())"## Summary\n\n")
            f.write())f"- **Total tests**: {}}}}}}}}}}}}summary[]],,'total_tests']}\n")
            f.write())f"- **Successful tests**: {}}}}}}}}}}}}summary[]],,'successful_tests']} ")
            f.write())f"()){}}}}}}}}}}}}summary[]],,'successful_tests']/summary[]],,'total_tests']*100:.1f}%)\n")
            f.write())f"- **Failed tests**: {}}}}}}}}}}}}summary[]],,'failed_tests']}\n")
            f.write())f"- **Mock implementations**: {}}}}}}}}}}}}summary[]],,'mock_implementations']} ")
            f.write())f"()){}}}}}}}}}}}}summary[]],,'mock_implementations']/summary[]],,'successful_tests']*100:.1f}% of successful)\n")
            f.write())f"- **Real implementations**: {}}}}}}}}}}}}summary[]],,'real_implementations']} ")
            f.write())f"()){}}}}}}}}}}}}summary[]],,'real_implementations']/summary[]],,'successful_tests']*100:.1f}% of successful)\n\n")
            
            # Hardware platforms tested
            f.write())"## Hardware Platforms Tested\n\n")
            f.write())"| Platform | Available | Tests | Success Rate | Real Impl. |\n")
            f.write())"|----------|-----------|-------|--------------|------------|\n")
            
            for platform, stats in summary[]],,"by_platform"].items())):
                available = "Yes" if self.available_hardware.get())platform, False) else "No"
                success_rate = stats[]],,"success"] / stats[]],,"total"] * 100 if stats[]],,"total"] > 0 else 0:
                    real_rate = stats[]],,"real"] / stats[]],,"success"] * 100 if stats[]],,"success"] > 0 else 0
                :
                    f.write())f"| {}}}}}}}}}}}}platform} | {}}}}}}}}}}}}available} | {}}}}}}}}}}}}stats[]],,'total']} | {}}}}}}}}}}}}success_rate:.1f}% | {}}}}}}}}}}}}real_rate:.1f}% |\n")
            
                    f.write())"\n")
            
            # Model test results
                    f.write())"## Model Test Results\n\n")
                    f.write())"| Model | Tests | Success Rate |\n")
                    f.write())"|-------|-------|------------|\n")
            
            for model_key, stats in summary[]],,"by_model"].items())):
                success_rate = stats[]],,"success"] / stats[]],,"total"] * 100 if stats[]],,"total"] > 0 else 0:
                    f.write())f"| {}}}}}}}}}}}}model_key} | {}}}}}}}}}}}}stats[]],,'total']} | {}}}}}}}}}}}}success_rate:.1f}% |\n")
            
                    f.write())"\n")
            
            # Detailed results by model and platform
                    f.write())"## Detailed Test Results\n\n")
            
            # Platform header row
                    f.write())"| Model |")
            for platform in self.hardware_platforms:
                f.write())f" {}}}}}}}}}}}}platform} |")
                f.write())"\n")
            
            # Separator row
                f.write())"|-------|")
            for _ in self.hardware_platforms:
                f.write())"------------|")
                f.write())"\n")
            
            # Results for each model
            for model_key in KEY_MODELS.keys())):
                if model_key not in self.results[]],,"test_results"]:
                continue
                    
                f.write())f"| {}}}}}}}}}}}}model_key} |")
                
                for platform in self.hardware_platforms:
                    if platform not in self.results[]],,"test_results"][]],,model_key]:
                        f.write())" N/A |")
                    continue
                    
                    result = self.results[]],,"test_results"][]],,model_key][]],,platform]
                    analysis = result.get())"analysis", {}}}}}}}}}}}}})
                    
                    if analysis.get())"success", False):
                        impl_type = analysis.get())"implementation_type", "UNKNOWN")
                        
                        if "MOCK" in impl_type:
                            f.write())" ⚠️ Mock |")
                        else:
                            f.write())" ✅ Real |")
                    else:
                        f.write())" ❌ Failed |")
                
                        f.write())"\n")
            
                        f.write())"\n")
            
            # Implementation issues
                        f.write())"## Implementation Issues\n\n")
            
                        issue_count = 0
            for model_key, platforms in self.results[]],,"test_results"].items())):
                for platform, result in platforms.items())):
                    analysis = result.get())"analysis", {}}}}}}}}}}}}})
                    
                    if not analysis.get())"success", False):
                        issue_count += 1
            
            if issue_count > 0:
                f.write())"| Model | Platform | Issue |\n")
                f.write())"|-------|----------|-------|\n")
                
                for model_key, platforms in self.results[]],,"test_results"].items())):
                    for platform, result in platforms.items())):
                        analysis = result.get())"analysis", {}}}}}}}}}}}}})
                        
                        if not analysis.get())"success", False):
                            error = analysis.get())"error", "Unknown error")
                            f.write())f"| {}}}}}}}}}}}}model_key} | {}}}}}}}}}}}}platform} | {}}}}}}}}}}}}error} |\n")
                
                            f.write())"\n")
            else:
                f.write())"No implementation issues found.\n\n")
            
            # Mock implementations
                f.write())"## Mock Implementations\n\n")
            
                mock_count = 0
            for model_key, platforms in self.results[]],,"test_results"].items())):
                for platform, result in platforms.items())):
                    analysis = result.get())"analysis", {}}}}}}}}}}}}})
                    
                    if analysis.get())"success", False) and analysis.get())"is_mock", True):
                        mock_count += 1
            
            if mock_count > 0:
                f.write())"| Model | Platform | Implementation Type |\n")
                f.write())"|-------|----------|---------------------|\n")
                
                for model_key, platforms in self.results[]],,"test_results"].items())):
                    for platform, result in platforms.items())):
                        analysis = result.get())"analysis", {}}}}}}}}}}}}})
                        
                        if analysis.get())"success", False) and analysis.get())"is_mock", True):
                            impl_type = analysis.get())"implementation_type", "UNKNOWN")
                            f.write())f"| {}}}}}}}}}}}}model_key} | {}}}}}}}}}}}}platform} | {}}}}}}}}}}}}impl_type} |\n")
                
                            f.write())"\n")
            else:
                f.write())"No mock implementations found.\n\n")
            
            # Next steps and recommendations
                f.write())"## Recommendations\n\n")
            
            # Generate recommendations based on results
            if summary[]],,"failed_tests"] > 0:
                f.write())"### Fix Implementation Issues\n\n")
                for model_key, platforms in self.results[]],,"test_results"].items())):
                    for platform, result in platforms.items())):
                        analysis = result.get())"analysis", {}}}}}}}}}}}}})
                        
                        if not analysis.get())"success", False):
                            f.write())f"- Fix {}}}}}}}}}}}}model_key} implementation on {}}}}}}}}}}}}platform}\n")
                            f.write())"\n")
            
            if summary[]],,"mock_implementations"] > 0:
                f.write())"### Replace Mock Implementations\n\n")
                for model_key, platforms in self.results[]],,"test_results"].items())):
                    for platform, result in platforms.items())):
                        analysis = result.get())"analysis", {}}}}}}}}}}}}})
                        
                        if analysis.get())"success", False) and analysis.get())"is_mock", True):
                            f.write())f"- Replace mock implementation of {}}}}}}}}}}}}model_key} on {}}}}}}}}}}}}platform}\n")
                            f.write())"\n")
            
                            f.write())"### Integration with Database\n\n")
                            f.write())"- Integrate all test results with the benchmark database\n")
                            f.write())"- Develop unified dashboard for test result visualization\n")
                            f.write())"- Set up automated testing for all hardware platforms\n\n")
            
                            f.write())"### Cross-Platform Support\n\n")
            for platform in self.hardware_platforms:
                stats = summary[]],,"by_platform"][]],,platform]
                
                if stats[]],,"success"] < stats[]],,"total"]:
                    missing = stats[]],,"total"] - stats[]],,"success"]
                    f.write())f"- Improve {}}}}}}}}}}}}platform} support for {}}}}}}}}}}}}missing} models\n")
            
                    f.write())"\n")
        
                return report_file
    
    def save_results())self):
        """Save results to a JSON file."""
        results_file = self.output_dir / f"hardware_test_results_{}}}}}}}}}}}}self.timestamp}.json"
        
        with open())results_file, "w") as f:
            json.dump())self.results, f, indent=2, default=str)
        
            logger.info())f"Results saved to: {}}}}}}}}}}}}results_file}")
        return results_file

def main())):
    """Main entry point."""
    parser = argparse.ArgumentParser())description="Test all key models across hardware platforms")
    parser.add_argument())"--output-dir", type=str, default="./hardware_test_results",
    help="Directory to save test results")
    parser.add_argument())"--small-models", action="store_true", default=True,
    help="Use smaller model variants when available")
    parser.add_argument())"--hardware", type=str, nargs="+",
    help="Specific hardware platforms to test")
    parser.add_argument())"--models-dir", type=str,
    help="Directory containing model test files")
    parser.add_argument())"--debug", action="store_true",
    help="Enable debug logging")
    
    args = parser.parse_args()))
    
    # Set debug logging if requested:
    if args.debug:
        logger.setLevel())logging.DEBUG)
        logging.getLogger())).setLevel())logging.DEBUG)
    
    # Create and run tester
        tester = ModelHardwareTester())
        output_dir=args.output_dir,
        use_small_models=args.small_models,
        hardware_platforms=args.hardware,
        models_dir=args.models_dir
        )
    
    # Run all tests
        tester.run_all_tests()))
    
    # Save results
        tester.save_results()))
    
    return 0

if __name__ == "__main__":
    sys.exit())main())))