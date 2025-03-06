#!/usr/bin/env python3
"""
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hardware platforms to test
ALL_HARDWARE_PLATFORMS = ["cpu", "cuda", "mps", "openvino", "rocm", "webnn", "webgpu"]

def detect_hardware(platforms=None):
    """
    Detect which hardware platforms are available.
    
    Args:
        platforms: List of platforms to check, or None for all
        
    Returns:
        Dictionary of platform availability
    """
    check_platforms = platforms or ALL_HARDWARE_PLATFORMS
    available = {"cpu": True}  # CPU is always available
    
    # Check for PyTorch-based platforms
    try:
        import torch
        
        # Check CUDA
        if "cuda" in check_platforms:
            available["cuda"] = torch.cuda.is_available()
            if available["cuda"]:
                logger.info(f"CUDA is available with {torch.cuda.device_count()} devices")
        
        # Check MPS (Apple Silicon)
        if "mps" in check_platforms:
            if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_available"):
                available["mps"] = torch.backends.mps.is_available()
                if available["mps"]:
                    logger.info(f"MPS (Apple Silicon) is available")
            else:
                available["mps"] = False
        
        # Check ROCm (AMD)
        if "rocm" in check_platforms:
            if torch.cuda.is_available() and hasattr(torch.version, "hip"):
                available["rocm"] = True
                logger.info(f"ROCm (AMD) is available")
            else:
                available["rocm"] = False
    except ImportError:
        # PyTorch not available
        logger.warning("PyTorch not available, CUDA/MPS/ROCm support cannot be detected")
        for platform in ["cuda", "mps", "rocm"]:
            if platform in check_platforms:
                available[platform] = False
    
    # Check OpenVINO
    if "openvino" in check_platforms:
        try:
            import openvino
            available["openvino"] = True
            logger.info(f"OpenVINO is available (version {openvino.__version__})")
        except ImportError:
            available["openvino"] = False
    
    # Web platforms - always enable for simulation
    if "webnn" in check_platforms:
        available["webnn"] = True
        logger.info("WebNN will be tested in simulation mode")
    
    if "webgpu" in check_platforms:
        available["webgpu"] = True
        logger.info("WebGPU will be tested in simulation mode")
    
    return available

def load_model_test_module(model_file):
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
        logger.error(f"Error loading module {model_file}: {e}")
        traceback.print_exc()
        return None

def find_test_class(module):
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

def test_model_on_platform(model_path, model_name, platform, output_dir=None):
    """
    Test a model on a specific platform.
    
    Args:
        model_path: Path to the model test file
        model_name: Name of the model to test
        platform: Hardware platform to test on
        output_dir: Directory to save results (optional)
        
    Returns:
        Test results dictionary
    """
    logger.info(f"Testing {model_name} on {platform}...")
    start_time = time.time()
    
    results = {
        "model": model_name,
        "platform": platform,
        "timestamp": datetime.datetime.now().isoformat(),
        "success": False,
        "execution_time": 0
    }
    
    try:
        # Load module and find test class
        module = load_model_test_module(model_path)
        TestClass = find_test_class(module)
        
        if not TestClass:
            results["error"] = "Could not find test class in module"
            return results
        
        # Create test instance
        test_instance = TestClass(model_id=model_name)
        
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
        logger.error(f"Error testing {model_name} on {platform}: {e}")
    
    # Calculate execution time
    results["total_execution_time"] = time.time() - start_time
    
    # Save results if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        output_file = output_dir / f"{model_name.replace('/', '_')}_{platform}_test.json"
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
    
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test a model across hardware platforms")
    parser.add_argument("--model-file", type=str, required=True,
                      help="Path to the model test file")
    parser.add_argument("--model-name", type=str,
                      help="Name or ID of the model to test")
    parser.add_argument("--platforms", type=str, nargs="+", default=ALL_HARDWARE_PLATFORMS,
                      help="Hardware platforms to test")
    parser.add_argument("--output-dir", type=str, default="hardware_test_results",
                      help="Directory to save test results")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    # Check if model file exists
    model_file = Path(args.model_file)
    if not model_file.exists():
        logger.error(f"Model file not found: {model_file}")
        return 1
    
    # Try to infer model name from filename if not provided
    model_name = args.model_name
    if not model_name:
        # Extract model type from filename (e.g., test_hf_bert.py -> bert)
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
        
        model_name = default_models.get(model_type)
        if not model_name:
            logger.error(f"Could not infer default model name for {model_type}")
            return 1
        
        logger.info(f"Using default model name for {model_type}: {model_name}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Detect available hardware
    available_hardware = detect_hardware(args.platforms)
    
    # Run tests on all specified platforms
    results = {}
    
    for platform in args.platforms:
        if platform != "cpu" and not available_hardware.get(platform, False):
            logger.warning(f"Platform {platform} not available, skipping test")
            continue
        
        result = test_model_on_platform(model_file, model_name, platform, output_dir)
        results[platform] = result
        
        if result["success"]:
            logger.info(f"✅ {platform} test passed")
            
            # Check if implementation is mocked
            if result.get("is_mock", True):
                logger.warning(f"⚠️ {platform} implementation is mocked!")
            else:
                logger.info(f"💯 {platform} implementation is real")
        else:
            logger.error(f"❌ {platform} test failed: {result.get('error', 'Unknown error')}")
    
    # Generate summary report
    report_file = output_dir / f"summary_{model_name.replace('/', '_')}.md"
    
    with open(report_file, "w") as f:
        f.write(f"# Hardware Test Report for {model_name}\n\n")
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
                f.write(f"- Fix {model_name} implementation on {platform}\n")
            f.write("\n")
        
        if mocks:
            f.write("### Replace Mock Implementations\n\n")
            for platform, _ in mocks:
                f.write(f"- Replace mock implementation of {model_name} on {platform}\n")
            f.write("\n")
        
        if not failures and not mocks:
            f.write("All implementations are working correctly and are not mocks! 🎉\n\n")
    
    logger.info(f"Report saved to: {report_file}")
    
    # Check overall success
    if failures:
        logger.warning(f"Some tests failed ({len(failures)}/{len(results)})")
        return 1
    else:
        logger.info(f"All tests passed ({len(results)}/{len(results)})")
        return 0

if __name__ == "__main__":
    sys.exit(main())