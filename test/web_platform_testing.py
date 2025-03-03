#!/usr/bin/env python3
"""
Web Platform Testing for IPFS Accelerate Python

This module implements comprehensive testing for WebNN and WebGPU capabilities
across different model modalities (text, vision, audio, multimodal).
"""

import os
import sys
import json
import time
import argparse
import importlib
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple, Union

# Add the parent directory to sys.path to import modules correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import required packages
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed, some functionality will be limited")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: NumPy not installed, some functionality will be limited")

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: Transformers not installed, some functionality will be limited")

try:
    import onnx
    import onnxruntime
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: ONNX libraries not installed, WebNN export will be simulated")

# Define modality types for categorization
MODALITY_TYPES = {
    "text": ["bert", "gpt", "t5", "llama", "roberta", "distilbert", "mistral", "phi"],
    "vision": ["vit", "resnet", "detr", "convnext", "swin", "sam"],
    "audio": ["whisper", "wav2vec", "clap", "hubert", "speecht5"],
    "multimodal": ["clip", "llava", "blip", "flava", "git", "pix2struct"]
}

class WebPlatformTesting:
    """
    Main class for testing WebNN and WebGPU capabilities across different models.
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize web platform testing framework.
        
        Args:
            resources: Dictionary of shared resources
            metadata: Configuration metadata
        """
        self.resources = resources or {}
        self.metadata = metadata or {}
        
        # Define web platforms to test
        self.web_platforms = ["webnn", "webgpu"]
        
        # Import skill test modules
        self.skill_modules = self._import_skill_modules()
        
        # Setup paths for results
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.test_dir, "web_platform_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Performance tracking
        self.performance_metrics = {}
        
    def _import_skill_modules(self):
        """Import all skill test modules from the skills folder."""
        skills_dir = os.path.join(self.test_dir, "skills")
        skill_modules = {}
        
        if not os.path.exists(skills_dir):
            print(f"Warning: Skills directory not found at {skills_dir}")
            return skill_modules
            
        for filename in os.listdir(skills_dir):
            if filename.startswith("test_hf_") and filename.endswith(".py"):
                module_name = filename[:-3]  # Remove .py extension
                try:
                    module = importlib.import_module(f"test.skills.{module_name}")
                    skill_modules[module_name] = module
                except ImportError as e:
                    print(f"Error importing {module_name}: {e}")
                    
        return skill_modules
        
    def detect_model_modality(self, model_name: str) -> str:
        """Detect model modality based on name patterns.
        
        Args:
            model_name: The model name to categorize
            
        Returns:
            String modality: "text", "vision", "audio", "multimodal", or "unknown"
        """
        model_name_lower = model_name.lower()
        
        for modality, patterns in MODALITY_TYPES.items():
            for pattern in patterns:
                if pattern.lower() in model_name_lower:
                    return modality
                    
        return "unknown"
        
    def test_model_on_web_platform(self, 
                                  model_name: str, 
                                  platform: str,
                                  timeout: int = 300) -> Dict[str, Any]:
        """Test a specific model on a web platform.
        
        Args:
            model_name: Name of the model to test
            platform: Web platform to test on ("webnn" or "webgpu")
            timeout: Maximum time in seconds for the test
            
        Returns:
            Dictionary with test results
        """
        if platform not in self.web_platforms:
            return {
                "success": False,
                "error": f"Unsupported platform: {platform}",
                "model": model_name
            }
            
        # Clean up model name for module lookup
        module_name = model_name
        if model_name.startswith("test_"):
            module_name = model_name
        elif not model_name.startswith("test_hf_"):
            module_name = f"test_hf_{model_name}"
            
        # Get the test module
        if module_name not in self.skill_modules:
            return {
                "success": False,
                "error": f"Test module not found for {model_name}",
                "model": model_name
            }
            
        module = self.skill_modules[module_name]
        
        # Get the test class
        test_class = None
        for attr_name in dir(module):
            if attr_name.startswith("Test") and not attr_name.startswith("TestCase"):
                test_class = getattr(module, attr_name)
                break
                
        if test_class is None:
            return {
                "success": False,
                "error": f"Test class not found in module {module_name}",
                "model": model_name
            }
            
        # Initialize the test instance
        try:
            test_instance = test_class()
        except Exception as e:
            return {
                "success": False,
                "error": f"Error initializing test class: {str(e)}",
                "model": model_name,
                "traceback": traceback.format_exc()
            }
            
        # Detect modality
        modality = self.detect_model_modality(model_name)
        
        # Run the appropriate test method based on the platform
        start_time = time.time()
        try:
            if platform == "webnn":
                if hasattr(test_instance, "test_webnn"):
                    result = test_instance.test_webnn()
                elif hasattr(test_instance, "init_webnn"):
                    # Initialize for WebNN
                    endpoint, processor, handler, queue, batch_size = test_instance.init_webnn()
                    
                    # Run inference with a basic input (based on modality)
                    if modality == "text":
                        test_input = "The quick brown fox jumps over the lazy dog."
                    elif modality == "vision":
                        test_input = "test.jpg"
                    elif modality == "audio":
                        test_input = "test.mp3"
                    elif modality == "multimodal":
                        test_input = {"image": "test.jpg", "text": "What is this?"}
                    else:
                        test_input = "Example input"
                        
                    inference_result = handler(test_input)
                    
                    # Check implementation type
                    if isinstance(inference_result, dict) and "implementation_type" in inference_result:
                        implementation_type = inference_result["implementation_type"]
                    else:
                        implementation_type = "UNKNOWN"
                        
                    result = {
                        "success": True,
                        "implementation_type": implementation_type,
                        "model": model_name
                    }
                else:
                    result = {
                        "success": False,
                        "error": "No WebNN test method found in test class",
                        "model": model_name
                    }
            elif platform == "webgpu":
                if hasattr(test_instance, "test_webgpu"):
                    result = test_instance.test_webgpu()
                elif hasattr(test_instance, "init_webgpu"):
                    # Initialize for WebGPU
                    endpoint, processor, handler, queue, batch_size = test_instance.init_webgpu()
                    
                    # Run inference with a basic input (based on modality)
                    if modality == "text":
                        test_input = "The quick brown fox jumps over the lazy dog."
                    elif modality == "vision":
                        test_input = "test.jpg"
                    elif modality == "audio":
                        test_input = "test.mp3"
                    elif modality == "multimodal":
                        test_input = {"image": "test.jpg", "text": "What is this?"}
                    else:
                        test_input = "Example input"
                        
                    inference_result = handler(test_input)
                    
                    # Check implementation type
                    if isinstance(inference_result, dict) and "implementation_type" in inference_result:
                        implementation_type = inference_result["implementation_type"]
                    else:
                        implementation_type = "UNKNOWN"
                        
                    result = {
                        "success": True,
                        "implementation_type": implementation_type,
                        "model": model_name
                    }
                else:
                    result = {
                        "success": False,
                        "error": "No WebGPU test method found in test class",
                        "model": model_name
                    }
        except Exception as e:
            result = {
                "success": False,
                "error": f"Error running test: {str(e)}",
                "model": model_name,
                "traceback": traceback.format_exc()
            }
            
        # Add timing information
        end_time = time.time()
        result["execution_time"] = end_time - start_time
        result["timeout"] = timeout
        result["timed_out"] = (end_time - start_time) >= timeout
        result["modality"] = modality
        result["timestamp"] = datetime.now().isoformat()
        
        return result
        
    def test_models_on_web_platform(self,
                                   models: List[str],
                                   platform: str,
                                   parallel: bool = False,
                                   max_workers: int = 4,
                                   timeout: int = 300) -> Dict[str, Any]:
        """Test multiple models on a web platform.
        
        Args:
            models: List of model names to test
            platform: Web platform to test on ("webnn" or "webgpu")
            parallel: Whether to run tests in parallel
            max_workers: Maximum number of parallel workers
            timeout: Maximum time in seconds for each test
            
        Returns:
            Dictionary with test results for all models
        """
        results = {}
        start_time = time.time()
        
        if parallel and len(models) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_model = {
                    executor.submit(self.test_model_on_web_platform, model, platform, timeout): model
                    for model in models
                }
                
                for future in as_completed(future_to_model):
                    model = future_to_model[future]
                    try:
                        result = future.result()
                        results[model] = result
                    except Exception as e:
                        results[model] = {
                            "success": False,
                            "error": f"Thread execution error: {str(e)}",
                            "model": model,
                            "traceback": traceback.format_exc()
                        }
        else:
            for model in models:
                results[model] = self.test_model_on_web_platform(model, platform, timeout)
                
        # Add summary information
        end_time = time.time()
        total_time = end_time - start_time
        success_count = sum(1 for r in results.values() if r.get("success", False))
        
        summary = {
            "platform": platform,
            "total_models": len(models),
            "successful_models": success_count,
            "success_rate": success_count / len(models) if models else 0,
            "total_execution_time": total_time,
            "average_execution_time": total_time / len(models) if models else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        return {"results": results, "summary": summary}
        
    def compare_web_platforms(self,
                             models: List[str],
                             parallel: bool = False,
                             max_workers: int = 4,
                             timeout: int = 300) -> Dict[str, Any]:
        """Compare WebNN and WebGPU performance for a set of models.
        
        Args:
            models: List of model names to test
            parallel: Whether to run tests in parallel
            max_workers: Maximum number of parallel workers
            timeout: Maximum time in seconds for each test
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        # Test on WebNN
        webnn_results = self.test_models_on_web_platform(
            models=models,
            platform="webnn",
            parallel=parallel,
            max_workers=max_workers,
            timeout=timeout
        )
        
        # Test on WebGPU
        webgpu_results = self.test_models_on_web_platform(
            models=models,
            platform="webgpu",
            parallel=parallel,
            max_workers=max_workers,
            timeout=timeout
        )
        
        # Compare results model by model
        for model in models:
            webnn_result = webnn_results["results"].get(model, {})
            webgpu_result = webgpu_results["results"].get(model, {})
            
            model_comparison = {
                "model": model,
                "modality": webnn_result.get("modality") or webgpu_result.get("modality") or "unknown",
                "webnn": {
                    "success": webnn_result.get("success", False),
                    "execution_time": webnn_result.get("execution_time"),
                    "implementation_type": webnn_result.get("implementation_type")
                },
                "webgpu": {
                    "success": webgpu_result.get("success", False),
                    "execution_time": webgpu_result.get("execution_time"),
                    "implementation_type": webgpu_result.get("implementation_type")
                }
            }
            
            # Add performance comparison if both succeeded
            if webnn_result.get("success", False) and webgpu_result.get("success", False):
                webnn_time = webnn_result.get("execution_time", 0)
                webgpu_time = webgpu_result.get("execution_time", 0)
                
                if webnn_time > 0 and webgpu_time > 0:
                    speedup = webnn_time / webgpu_time
                    model_comparison["speedup"] = speedup
                    model_comparison["faster_platform"] = "webgpu" if speedup > 1 else "webnn"
                
            comparison[model] = model_comparison
            
        # Calculate overall statistics
        models_with_both = [m for m in models if 
                           comparison[m]["webnn"]["success"] and 
                           comparison[m]["webgpu"]["success"]]
        
        avg_speedup = sum(comparison[m].get("speedup", 1) for m in models_with_both) / len(models_with_both) if models_with_both else 0
        
        summary = {
            "total_models": len(models),
            "models_with_both_platforms": len(models_with_both),
            "webnn_only_success": sum(1 for m in models if 
                                    comparison[m]["webnn"]["success"] and 
                                    not comparison[m]["webgpu"]["success"]),
            "webgpu_only_success": sum(1 for m in models if 
                                     not comparison[m]["webnn"]["success"] and 
                                     comparison[m]["webgpu"]["success"]),
            "average_speedup": avg_speedup,
            "faster_platform_overall": "webgpu" if avg_speedup > 1 else "webnn",
            "timestamp": datetime.now().isoformat()
        }
        
        return {"comparisons": comparison, "summary": summary}
        
    def generate_report(self, 
                       report_data: Dict[str, Any],
                       report_type: str,
                       output_format: str = "json") -> str:
        """Generate a report for web platform testing results.
        
        Args:
            report_data: Data to include in the report
            report_type: Type of report ("single", "multi", "comparison")
            output_format: Format for the report ("json", "md")
            
        Returns:
            Path to the generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == "json":
            # JSON report
            filename = f"web_platform_{report_type}_{timestamp}.json"
            file_path = os.path.join(self.results_dir, filename)
            
            with open(file_path, "w") as f:
                json.dump(report_data, f, indent=2)
                
        elif output_format == "md":
            # Markdown report
            filename = f"web_platform_{report_type}_{timestamp}.md"
            file_path = os.path.join(self.results_dir, filename)
            
            with open(file_path, "w") as f:
                if report_type == "comparison":
                    self._write_comparison_markdown(f, report_data)
                elif report_type == "single":
                    self._write_single_platform_markdown(f, report_data)
                elif report_type == "multi":
                    self._write_multi_platform_markdown(f, report_data)
                    
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        print(f"Report generated at: {file_path}")
        return file_path
        
    def _write_comparison_markdown(self, file, data):
        """Write a comparison report in markdown format."""
        file.write("# Web Platform Comparison Report\n\n")
        file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write summary
        file.write("## Summary\n\n")
        summary = data["summary"]
        file.write(f"- Total models tested: {summary['total_models']}\n")
        file.write(f"- Models with both platforms working: {summary['models_with_both_platforms']}\n")
        file.write(f"- WebNN only successes: {summary['webnn_only_success']}\n")
        file.write(f"- WebGPU only successes: {summary['webgpu_only_success']}\n")
        file.write(f"- Average speedup (WebGPU vs WebNN): {summary['average_speedup']:.2f}x\n")
        file.write(f"- Faster platform overall: {summary['faster_platform_overall'].upper()}\n\n")
        
        # Write comparison table
        file.write("## Detailed Comparison\n\n")
        file.write("| Model | Modality | WebNN Success | WebGPU Success | WebNN Time (s) | WebGPU Time (s) | Speedup | Faster Platform |\n")
        file.write("|-------|----------|--------------|----------------|----------------|-----------------|---------|----------------|\n")
        
        for model, comp in data["comparisons"].items():
            webnn = comp["webnn"]
            webgpu = comp["webgpu"]
            
            webnn_success = "✅" if webnn["success"] else "❌"
            webgpu_success = "✅" if webgpu["success"] else "❌"
            
            webnn_time = f"{webnn['execution_time']:.3f}" if webnn["execution_time"] else "N/A"
            webgpu_time = f"{webgpu['execution_time']:.3f}" if webgpu["execution_time"] else "N/A"
            
            speedup = f"{comp.get('speedup', 'N/A'):.2f}x" if "speedup" in comp else "N/A"
            faster = comp.get("faster_platform", "N/A").upper() if "faster_platform" in comp else "N/A"
            
            file.write(f"| {model} | {comp['modality']} | {webnn_success} | {webgpu_success} | {webnn_time} | {webgpu_time} | {speedup} | {faster} |\n")
            
        # Write implementation types
        file.write("\n## Implementation Types\n\n")
        file.write("| Model | WebNN Implementation | WebGPU Implementation |\n")
        file.write("|-------|----------------------|----------------------|\n")
        
        for model, comp in data["comparisons"].items():
            webnn_impl = comp["webnn"].get("implementation_type", "N/A")
            webgpu_impl = comp["webgpu"].get("implementation_type", "N/A")
            
            file.write(f"| {model} | {webnn_impl} | {webgpu_impl} |\n")
        
    def _write_single_platform_markdown(self, file, data):
        """Write a single platform report in markdown format."""
        platform = data["summary"]["platform"]
        file.write(f"# {platform.upper()} Platform Testing Report\n\n")
        file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write summary
        file.write("## Summary\n\n")
        summary = data["summary"]
        file.write(f"- Total models tested: {summary['total_models']}\n")
        file.write(f"- Successful models: {summary['successful_models']}\n")
        file.write(f"- Success rate: {summary['success_rate']*100:.2f}%\n")
        file.write(f"- Total execution time: {summary['total_execution_time']:.2f}s\n")
        file.write(f"- Average execution time: {summary['average_execution_time']:.2f}s\n\n")
        
        # Write results table
        file.write("## Detailed Results\n\n")
        file.write("| Model | Modality | Success | Time (s) | Implementation Type |\n")
        file.write("|-------|----------|---------|----------|---------------------|\n")
        
        for model, result in data["results"].items():
            success = "✅" if result.get("success", False) else "❌"
            execution_time = f"{result.get('execution_time', 0):.3f}"
            impl_type = result.get("implementation_type", "N/A")
            modality = result.get("modality", "unknown")
            
            file.write(f"| {model} | {modality} | {success} | {execution_time} | {impl_type} |\n")
            
        # Write error section if there are failures
        failures = {m: r for m, r in data["results"].items() if not r.get("success", False)}
        if failures:
            file.write("\n## Errors\n\n")
            
            for model, result in failures.items():
                file.write(f"### {model}\n\n")
                file.write(f"Error: {result.get('error', 'Unknown error')}\n\n")
                
                if "traceback" in result:
                    file.write("```\n")
                    file.write(result["traceback"])
                    file.write("```\n\n")
        
    def _write_multi_platform_markdown(self, file, data):
        """Write a multi-platform report in markdown format."""
        file.write("# Web Platforms Testing Report\n\n")
        file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Loop through platforms
        for platform, platform_data in data.items():
            file.write(f"## {platform.upper()} Platform\n\n")
            
            # Write summary
            summary = platform_data["summary"]
            file.write("### Summary\n\n")
            file.write(f"- Total models tested: {summary['total_models']}\n")
            file.write(f"- Successful models: {summary['successful_models']}\n")
            file.write(f"- Success rate: {summary['success_rate']*100:.2f}%\n")
            file.write(f"- Total execution time: {summary['total_execution_time']:.2f}s\n")
            file.write(f"- Average execution time: {summary['average_execution_time']:.2f}s\n\n")
            
            # Write results table
            file.write("### Detailed Results\n\n")
            file.write("| Model | Modality | Success | Time (s) | Implementation Type |\n")
            file.write("|-------|----------|---------|----------|---------------------|\n")
            
            for model, result in platform_data["results"].items():
                success = "✅" if result.get("success", False) else "❌"
                execution_time = f"{result.get('execution_time', 0):.3f}"
                impl_type = result.get("implementation_type", "N/A")
                modality = result.get("modality", "unknown")
                
                file.write(f"| {model} | {modality} | {success} | {execution_time} | {impl_type} |\n")
                
            file.write("\n")
            
    def get_all_available_models(self) -> List[str]:
        """Get all available models for testing.
        
        Returns:
            List of model names that have test modules
        """
        models = []
        for module_name in self.skill_modules:
            if module_name.startswith("test_hf_"):
                model_name = module_name[8:]  # Remove "test_hf_" prefix
                models.append(model_name)
        return models
        
    def get_models_by_modality(self, modality: str) -> List[str]:
        """Get available models for a specific modality.
        
        Args:
            modality: The modality to filter by
            
        Returns:
            List of model names for the specified modality
        """
        all_models = self.get_all_available_models()
        return [m for m in all_models if self.detect_model_modality(m) == modality]
    
    def run_web_compatibility_report(self, 
                                   modality: Optional[str] = None,
                                   limit: int = 5,
                                   parallel: bool = False,
                                   output_format: str = "md") -> str:
        """Generate a comprehensive web compatibility report.
        
        Args:
            modality: Filter models by modality (None for all)
            limit: Maximum number of models to test per modality
            parallel: Whether to run tests in parallel
            output_format: Output format ("json" or "md")
            
        Returns:
            Path to the generated report
        """
        # Get models to test
        if modality and modality != "all":
            models = self.get_models_by_modality(modality)[:limit]
        else:
            # Get representative models from each modality
            models = []
            for m in ["text", "vision", "audio", "multimodal"]:
                modality_models = self.get_models_by_modality(m)[:limit]
                models.extend(modality_models)
        
        # Run comparison
        comparison_results = self.compare_web_platforms(
            models=models,
            parallel=parallel
        )
        
        # Generate report
        report_path = self.generate_report(
            report_data=comparison_results,
            report_type="comparison",
            output_format=output_format
        )
        
        return report_path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Web Platform Testing Tool")
    
    # Main operation modes
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test-model", type=str, help="Test a specific model")
    group.add_argument("--test-modality", type=str, choices=["text", "vision", "audio", "multimodal", "all"],
                     help="Test models from a specific modality")
    group.add_argument("--compare", action="store_true", help="Compare WebNN and WebGPU performance")
    
    # Platform selection
    parser.add_argument("--platform", type=str, choices=["webnn", "webgpu", "both"],
                      default="both", help="Web platform to test")
    
    # Test parameters
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of models to test")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--timeout", type=int, default=300, help="Test timeout in seconds")
    
    # Output options
    parser.add_argument("--output-format", type=str, choices=["json", "md"], 
                      default="md", help="Output format for reports")
    parser.add_argument("--output-dir", type=str, help="Custom output directory for reports")
    
    # List available models
    parser.add_argument("--list-models", action="store_true", help="List available models for testing")
    parser.add_argument("--list-by-modality", action="store_true", help="List models grouped by modality")
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Create testing framework
    tester = WebPlatformTesting()
    
    # Set custom output directory if specified
    if args.output_dir:
        tester.results_dir = args.output_dir
        os.makedirs(tester.results_dir, exist_ok=True)
    
    # Handle listing options
    if args.list_models:
        models = tester.get_all_available_models()
        print(f"Available models ({len(models)}):")
        for model in sorted(models):
            print(f"- {model}")
        return
        
    if args.list_by_modality:
        print("Models by modality:")
        for modality in ["text", "vision", "audio", "multimodal", "unknown"]:
            modality_models = tester.get_models_by_modality(modality)
            print(f"\n{modality.upper()} ({len(modality_models)}):")
            for model in sorted(modality_models):
                print(f"- {model}")
        return
    
    # Handle test operations
    if args.test_model:
        model = args.test_model
        if args.platform == "both" or args.platform == "webnn":
            print(f"Testing {model} on WebNN...")
            result = tester.test_model_on_web_platform(model, "webnn", args.timeout)
            tester.generate_report(
                {"results": {model: result}, 
                 "summary": {
                    "platform": "webnn",
                    "total_models": 1,
                    "successful_models": 1 if result.get("success", False) else 0,
                    "success_rate": 1 if result.get("success", False) else 0,
                    "total_execution_time": result.get("execution_time", 0),
                    "average_execution_time": result.get("execution_time", 0),
                    "timestamp": datetime.now().isoformat()
                 }
                }, 
                "single", 
                args.output_format
            )
            
        if args.platform == "both" or args.platform == "webgpu":
            print(f"Testing {model} on WebGPU...")
            result = tester.test_model_on_web_platform(model, "webgpu", args.timeout)
            tester.generate_report(
                {"results": {model: result}, 
                 "summary": {
                    "platform": "webgpu",
                    "total_models": 1,
                    "successful_models": 1 if result.get("success", False) else 0,
                    "success_rate": 1 if result.get("success", False) else 0,
                    "total_execution_time": result.get("execution_time", 0),
                    "average_execution_time": result.get("execution_time", 0),
                    "timestamp": datetime.now().isoformat()
                 }
                }, 
                "single", 
                args.output_format
            )
            
    elif args.test_modality:
        modality = args.test_modality
        
        # Get models for the modality
        if modality == "all":
            # Get representative models from each modality
            models = []
            for m in ["text", "vision", "audio", "multimodal"]:
                modality_models = tester.get_models_by_modality(m)[:args.limit]
                models.extend(modality_models)
        else:
            models = tester.get_models_by_modality(modality)[:args.limit]
            
        # Run tests on selected platform(s)
        if args.platform == "both" or args.platform == "webnn":
            print(f"Testing {len(models)} {modality} models on WebNN...")
            results = tester.test_models_on_web_platform(
                models=models,
                platform="webnn",
                parallel=args.parallel,
                timeout=args.timeout
            )
            tester.generate_report(results, "single", args.output_format)
            
        if args.platform == "both" or args.platform == "webgpu":
            print(f"Testing {len(models)} {modality} models on WebGPU...")
            results = tester.test_models_on_web_platform(
                models=models,
                platform="webgpu",
                parallel=args.parallel,
                timeout=args.timeout
            )
            tester.generate_report(results, "single", args.output_format)
            
    elif args.compare:
        # Determine models to compare
        if args.test_modality and args.test_modality != "all":
            models = tester.get_models_by_modality(args.test_modality)[:args.limit]
        else:
            # Get representative models from each modality
            models = []
            for modality in ["text", "vision", "audio", "multimodal"]:
                modality_models = tester.get_models_by_modality(modality)[:args.limit]
                models.extend(modality_models)
                
        # Run comparison
        print(f"Comparing WebNN and WebGPU performance on {len(models)} models...")
        comparison_results = tester.compare_web_platforms(
            models=models,
            parallel=args.parallel,
            timeout=args.timeout
        )
        
        # Generate report
        tester.generate_report(comparison_results, "comparison", args.output_format)
        
    else:
        # Default action: run comprehensive compatibility report
        print("Generating comprehensive web compatibility report...")
        tester.run_web_compatibility_report(
            modality=args.test_modality,
            limit=args.limit,
            parallel=args.parallel,
            output_format=args.output_format
        )

if __name__ == "__main__":
    main()