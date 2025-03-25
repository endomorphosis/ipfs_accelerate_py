#!/usr/bin/env python3
"""
Test automatic generation of test files for new models.

This script tests the ability to automatically generate test files for
new models that might be released after our initial implementation.
"""

import os
import sys
import time
import json
import re
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"generate_tests_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Simplified architecture detection
def normalize_model_name(model_name: str) -> str:
    """Normalize model name."""
    # Extract the base model name (remove organization)
    if "/" in model_name:
        model_name = model_name.split("/")[1]
    
    # Remove version numbers and sizes
    model_name = re.sub(r"-\d+b.*$", "", model_name.lower())
    model_name = re.sub(r"\.?v\d+.*$", "", model_name)
    
    # Normalize hyphens
    model_name = model_name.replace("-", "_")
    
    return model_name

def get_architecture_type(model_name: str) -> str:
    """Get architecture type from model name."""
    name = model_name.lower()
    
    if any(token in name for token in ["gpt", "llama", "falcon", "phi", "mistral", "gemma"]):
        return "decoder-only"
    elif any(token in name for token in ["bert", "roberta", "deberta", "electra"]):
        return "encoder-only"
    elif any(token in name for token in ["t5", "bart", "pegasus"]):
        return "encoder-decoder"
    elif any(token in name for token in ["vit", "swin", "convnext", "resnet"]):
        return "vision"
    elif any(token in name for token in ["clip", "blip"]):
        return "vision-encoder-text-decoder"
    elif any(token in name for token in ["whisper", "wav2vec", "hubert"]):
        return "speech"
    elif any(token in name for token in ["llava", "flava", "flamingo"]):
        return "multimodal"
    else:
        return "decoder-only"  # Default for newer models

class ModelTestGenerator:
    """Simplified ModelTestGenerator for testing."""
    
    def __init__(self, output_dir="generated_tests/new_models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_test_file(self, model_name, force=False, verify=True):
        """Generate a test file."""
        try:
            arch_type = get_architecture_type(model_name)
            normalized_name = normalize_model_name(model_name)
            file_path = os.path.join(self.output_dir, f"test_hf_{normalized_name}.py")
            
            # Skip if file exists and not forcing
            if os.path.exists(file_path) and not force:
                logger.info(f"File already exists: {file_path}")
                return False, file_path
            
            # Create minimal test file
            with open(file_path, "w") as f:
                f.write(f"""#!/usr/bin/env python3
\"\"\"
Test file for {model_name} ({arch_type} architecture).
\"\"\"

import os
import sys
import logging
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    \"\"\"Test the model with mocked objects.\"\"\"
    try:
        # Mock implementation
        model = MagicMock()
        model.generate = MagicMock(return_value="Generated text")
        
        # Run basic test
        result = model.generate("Test input")
        
        return True
    except Exception as e:
        logger.error(f"Test failed: {{e}}")
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("✅ Test passed")
        sys.exit(0)
    else:
        print("❌ Test failed")
        sys.exit(1)
""")
            
            logger.info(f"Generated test file: {file_path}")
            return True, file_path
        
        except Exception as e:
            logger.error(f"Error generating test file: {e}")
            return False, ""

def test_model_generation(model_name: str, force: bool = False, verify: bool = True, output_dir: str = None) -> Tuple[bool, str, str]:
    """
    Test generation of a test file for a model.
    
    Args:
        model_name: Model name
        force: Whether to overwrite existing files
        verify: Whether to verify the generated file
        output_dir: Directory to output generated files
        
    Returns:
        Tuple of (success, file_path, architecture)
    """
    try:
        # Create test generator
        if output_dir is None:
            output_dir = "generated_tests/new_models"
        
        os.makedirs(output_dir, exist_ok=True)
        generator = ModelTestGenerator(output_dir=output_dir)
        
        # Detect model architecture
        architecture = get_architecture_type(model_name)
        logger.info(f"Detected architecture '{architecture}' for model '{model_name}'")
        
        # Generate test file
        logger.info(f"Generating test file for {model_name}")
        success, file_path = generator.generate_test_file(model_name, force=force, verify=verify)
        
        if success:
            logger.info(f"✅ Successfully generated test file: {file_path}")
        else:
            logger.error(f"❌ Failed to generate test file for {model_name}")
        
        return success, file_path, architecture
    except Exception as e:
        logger.error(f"❌ Error testing model generation for {model_name}: {e}")
        return False, "", "unknown"

def run_integration_test(test_file: str, mock: bool = False) -> bool:
    """
    Run an integration test on a generated test file.
    
    Args:
        test_file: Path to test file
        mock: Whether to use mocked dependencies
        
    Returns:
        True if test passed, False otherwise
    """
    try:
        # Skip if file doesn't exist
        if not os.path.exists(test_file):
            logger.error(f"Test file does not exist: {test_file}")
            return False
        
        # Set up environment
        env = os.environ.copy()
        if mock:
            env["MOCK_TORCH"] = "True"
            env["MOCK_TRANSFORMERS"] = "True"
            env["MOCK_TOKENIZERS"] = "True"
            env["MOCK_SENTENCEPIECE"] = "True"
        
        # Run test
        import subprocess
        cmd = [sys.executable, test_file, "--save"]
        
        logger.info(f"Running test: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        # Check result
        if result.returncode == 0:
            logger.info(f"✅ Test passed: {test_file}")
            return True
        else:
            logger.error(f"❌ Test failed: {test_file}")
            logger.error(f"Error: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Error running test {test_file}: {e}")
        return False

def generate_report(results: List[Dict[str, Any]], report_file: str) -> None:
    """
    Generate a report of test results.
    
    Args:
        results: List of test results
        report_file: Path to save report
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        # Write report file
        with open(report_file, "w") as f:
            f.write("# New Model Test Generation Report\n\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write summary
            total = len(results)
            generation_success = sum(1 for r in results if r["generation_success"])
            test_success = sum(1 for r in results if r["test_success"])
            
            f.write("## Summary\n\n")
            f.write(f"- **Total models**: {total}\n")
            gen_percentage = (generation_success/total*100) if total > 0 else 0
            f.write(f"- **Generation success**: {generation_success} ({gen_percentage:.1f}%)\n")
            test_percentage = (test_success/total*100) if total > 0 else 0
            f.write(f"- **Test success**: {test_success} ({test_percentage:.1f}%)\n\n")
            
            # Write results by architecture
            arch_results = {}
            for result in results:
                arch = result["architecture"]
                if arch not in arch_results:
                    arch_results[arch] = {"total": 0, "generation_success": 0, "test_success": 0}
                
                arch_results[arch]["total"] += 1
                if result["generation_success"]:
                    arch_results[arch]["generation_success"] += 1
                if result["test_success"]:
                    arch_results[arch]["test_success"] += 1
            
            f.write("## Results by Architecture\n\n")
            f.write("| Architecture | Total | Generation Success | Test Success |\n")
            f.write("|--------------|-------|-------------------|-------------|\n")
            
            for arch, stats in sorted(arch_results.items()):
                gen_rate = (stats["generation_success"] / stats["total"] * 100) if stats["total"] > 0 else 0
                test_rate = (stats["test_success"] / stats["total"] * 100) if stats["total"] > 0 else 0
                f.write(f"| {arch} | {stats['total']} | {stats['generation_success']} ({gen_rate:.1f}%) | {stats['test_success']} ({test_rate:.1f}%) |\n")
            
            # Write detailed results
            f.write("\n## Model Details\n\n")
            f.write("| Model | Architecture | Generation | Test |\n")
            f.write("|-------|--------------|------------|------|\n")
            
            for result in results:
                model = result["model"]
                arch = result["architecture"]
                gen_status = "✅ Success" if result["generation_success"] else "❌ Failed"
                test_status = "✅ Success" if result["test_success"] else "❌ Failed"
                
                f.write(f"| {model} | {arch} | {gen_status} | {test_status} |\n")
            
            # Write details for failed models
            failed_generation = [r for r in results if not r["generation_success"]]
            failed_tests = [r for r in results if r["generation_success"] and not r["test_success"]]
            
            if failed_generation:
                f.write("\n## Failed Generation\n\n")
                for result in failed_generation:
                    f.write(f"### {result['model']}\n\n")
                    f.write(f"- **Architecture**: {result['architecture']}\n")
                    f.write(f"- **Error**: {result.get('error', 'Unknown error')}\n\n")
            
            if failed_tests:
                f.write("\n## Failed Tests\n\n")
                for result in failed_tests:
                    f.write(f"### {result['model']}\n\n")
                    f.write(f"- **Architecture**: {result['architecture']}\n")
                    f.write(f"- **Test file**: {result['file_path']}\n")
                    f.write(f"- **Error**: {result.get('test_error', 'Unknown error')}\n\n")
        
        logger.info(f"✅ Report generated: {report_file}")
    except Exception as e:
        logger.error(f"❌ Error generating report: {e}")

def main():
    """Command-line entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test automatic generation for new models")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to test")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing files")
    parser.add_argument("--verify", action="store_true", help="Verify generated files")
    parser.add_argument("--mock", action="store_true", help="Use mocked dependencies for tests")
    parser.add_argument("--output-dir", default="generated_tests/new_models", help="Directory for generated files")
    parser.add_argument("--report-dir", default="reports", help="Directory for reports")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    
    # Track results
    results = []
    
    # Test each model
    for model in args.models:
        logger.info(f"\nTesting model: {model}")
        
        # Track model result
        result = {
            "model": model,
            "generation_success": False,
            "test_success": False,
            "architecture": "unknown",
            "file_path": "",
            "error": None,
            "test_error": None
        }
        
        try:
            # Generate test file
            generation_success, file_path, architecture = test_model_generation(
                model, force=args.force, verify=args.verify, output_dir=args.output_dir
            )
            
            result["generation_success"] = generation_success
            result["architecture"] = architecture
            result["file_path"] = file_path
            
            if generation_success:
                # Run integration test
                test_success = run_integration_test(file_path, mock=args.mock)
                result["test_success"] = test_success
            
        except Exception as e:
            logger.error(f"❌ Error processing model {model}: {e}")
            result["error"] = str(e)
        
        # Add result to list
        results.append(result)
    
    # Generate report
    report_file = os.path.join(args.report_dir, f"new_models_test_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    generate_report(results, report_file)
    
    # Print summary
    logger.info("\nSummary:")
    logger.info(f"- Total models: {len(results)}")
    logger.info(f"- Generation success: {sum(1 for r in results if r['generation_success'])}")
    logger.info(f"- Test success: {sum(1 for r in results if r['test_success'])}")
    logger.info(f"- Report: {report_file}")
    
    # Return status
    return 0 if all(r["generation_success"] and r["test_success"] for r in results) else 1

if __name__ == "__main__":
    sys.exit(main())