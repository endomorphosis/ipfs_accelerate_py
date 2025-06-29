#!/usr/bin/env python3
"""
Hugging Face Test Automation Integration Script

This script integrates all the components of our test automation framework:
1. Test generation
2. Indentation fixing
3. Architecture-aware template selection
4. Verification and validation
5. Reporting and metrics collection

Usage:
    python test_integration.py [--generate] [--fix] [--verify] [--report]
"""

import os
import sys
import glob
import json
import argparse
import logging
import subprocess
import importlib.util
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"test_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(os.path.dirname(ROOT_DIR), "models")
TEST_DIR = os.path.join(os.path.dirname(ROOT_DIR), "test")
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")
FIXED_TESTS_DIR = os.path.join(ROOT_DIR, "fixed_tests")
COLLECTED_RESULTS_DIR = os.path.join(FIXED_TESTS_DIR, "collected_results")

# Ensure output directories exist
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(FIXED_TESTS_DIR, exist_ok=True)
os.makedirs(COLLECTED_RESULTS_DIR, exist_ok=True)

# Import fix modules if available
def import_module_from_file(module_name, file_path):
    """Import a module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            logger.warning(f"Could not find module spec for {file_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.warning(f"Could not import {module_name} from {file_path}: {e}")
        return None

# Import test generator
test_generator_path = os.path.join(ROOT_DIR, "regenerate_tests_with_fixes.py")
test_generator = import_module_from_file("regenerate_tests_with_fixes", test_generator_path)

# Import indentation fixer
indentation_fixer_path = os.path.join(ROOT_DIR, "complete_indentation_fix.py")
indentation_fixer = import_module_from_file("complete_indentation_fix", indentation_fixer_path)

# Architecture types
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "roberta", "albert", "distilbert", "electra", "camembert", "xlm-roberta"],
    "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "bloom", "llama", "mistral", "falcon", "phi"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2"],
    "vision-encoder-text-decoder": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct"]
}

def get_architecture_type(model_type):
    """Determine architecture type from model type."""
    for arch_type, models in ARCHITECTURE_TYPES.items():
        for model in models:
            if model.lower() in model_type.lower():
                return arch_type
    return "encoder-only"  # Default

def list_model_types():
    """List all known model types and their architectures."""
    model_types = []
    
    for arch_type, models in ARCHITECTURE_TYPES.items():
        for model in models:
            model_types.append((model, arch_type))
    
    return model_types

def find_test_files(pattern="test_hf_*.py"):
    """Find test files matching pattern."""
    return glob.glob(os.path.join(ROOT_DIR, pattern))

def generate_test_file(model_type, force=False, verify=True):
    """Generate a test file for a specific model type."""
    if not test_generator:
        logger.error("Test generator module not found")
        return False
        
    file_path = os.path.join(ROOT_DIR, f"test_hf_{model_type}.py")
    
    # Use test_generator to regenerate the file
    try:
        if hasattr(test_generator, "regenerate_test_file"):
            success = test_generator.regenerate_test_file(file_path, force=force, verify=verify)
        else:
            # Direct subprocess call
            cmd = [sys.executable, test_generator_path, "--single", model_type]
            if force:
                cmd.append("--force")
            if verify:
                cmd.append("--verify")
                
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0
            
            if not success:
                logger.error(f"Failed to generate test file: {result.stderr}")
        
        return success
    except Exception as e:
        logger.error(f"Error generating test file for {model_type}: {e}")
        return False

def fix_test_file(file_path, verify=True):
    """Fix indentation issues in a test file."""
    if not indentation_fixer:
        logger.error("Indentation fixer module not found")
        return False
        
    # Use indentation_fixer to fix the file
    try:
        if hasattr(indentation_fixer, "fix_class_method_indentation"):
            success = indentation_fixer.fix_class_method_indentation(file_path, backup=True)
            
            if success and verify:
                # Verify syntax
                syntax_valid = indentation_fixer.verify_python_syntax(file_path)
                return syntax_valid
            
            return success
        else:
            # Direct subprocess call
            cmd = [sys.executable, indentation_fixer_path, file_path]
            if verify:
                cmd.append("--verify")
                
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0
            
            if not success:
                logger.error(f"Failed to fix indentation: {result.stderr}")
            
            return success
    except Exception as e:
        logger.error(f"Error fixing indentation in {file_path}: {e}")
        return False

def verify_test_file(file_path):
    """Verify syntax of a test file."""
    try:
        # Use Python's built-in compile function to check syntax
        with open(file_path, 'r') as f:
            code = f.read()
        
        compile(code, file_path, 'exec')
        logger.info(f"✅ {file_path}: Syntax is valid")
        return True
    except SyntaxError as e:
        logger.error(f"❌ {file_path}: Syntax error at line {e.lineno}")
        logger.error(f"   {e.text.strip() if e.text else ''}")
        logger.error(f"   {'^'.rjust(e.offset) if e.offset else ''}")
        return False
    except Exception as e:
        logger.error(f"❌ {file_path}: Error verifying syntax: {e}")
        return False

def run_test_file(file_path, output_dir=COLLECTED_RESULTS_DIR):
    """Run a test file and collect results."""
    try:
        # Extract model type from filename
        filename = os.path.basename(file_path)
        if not filename.startswith("test_hf_"):
            logger.warning(f"Invalid filename: {filename}, should start with 'test_hf_'")
            return False
            
        model_type = filename[8:].replace(".py", "")
        
        # Create command
        cmd = [sys.executable, file_path, "--save", "--output-dir", output_dir]
        
        # Run test
        logger.info(f"Running test: {file_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check result
        if result.returncode == 0:
            logger.info(f"✅ {file_path}: Test passed")
            return True
        else:
            logger.error(f"❌ {file_path}: Test failed")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"Error running test {file_path}: {e}")
        return False

def generate_report(output_dir=COLLECTED_RESULTS_DIR):
    """Generate a comprehensive report of test results."""
    try:
        # Find all result files
        result_files = glob.glob(os.path.join(output_dir, "*.json"))
        
        if not result_files:
            logger.warning(f"No result files found in {output_dir}")
            return False
        
        # Collect results
        results = {}
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract model info
                if "metadata" in data and "model" in data["metadata"]:
                    model_id = data["metadata"]["model"]
                    
                    # Skip if this is a summary file
                    if "summary" in os.path.basename(file_path):
                        continue
                    
                    # Add to results
                    if model_id not in results:
                        results[model_id] = []
                    
                    results[model_id].append({
                        "file": os.path.basename(file_path),
                        "success": any(r.get("pipeline_success", False) for r in data.get("results", {}).values()),
                        "performance": data.get("performance", {}),
                        "hardware": data.get("hardware", {}),
                        "timestamp": data.get("metadata", {}).get("timestamp", "")
                    })
            except Exception as e:
                logger.warning(f"Error processing result file {file_path}: {e}")
        
        # Generate summary report
        summary = {
            "total_models": len(results),
            "successful_models": sum(1 for model_results in results.values() 
                                    if any(r["success"] for r in model_results)),
            "total_tests": sum(len(model_results) for model_results in results.values()),
            "successful_tests": sum(sum(1 for r in model_results if r["success"]) 
                                   for model_results in results.values()),
            "models": {model: {"success": any(r["success"] for r in model_results)} 
                      for model, model_results in results.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate coverage percentage
        if summary["total_models"] > 0:
            summary["coverage_percentage"] = (summary["successful_models"] / summary["total_models"]) * 100
        else:
            summary["coverage_percentage"] = 0
        
        # Write summary to file
        summary_path = os.path.join(output_dir, f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Write markdown report
        report_path = os.path.join(ROOT_DIR, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(report_path, 'w') as f:
            f.write("# Hugging Face Models Test Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Models**: {summary['total_models']}\n")
            f.write(f"- **Successful Models**: {summary['successful_models']} ({summary['coverage_percentage']:.1f}%)\n")
            f.write(f"- **Total Tests**: {summary['total_tests']}\n")
            f.write(f"- **Successful Tests**: {summary['successful_tests']} ({summary['successful_tests']/summary['total_tests']*100 if summary['total_tests'] else 0:.1f}%)\n\n")
            
            f.write("## Model Results\n\n")
            f.write("| Model | Status | Tests |\n")
            f.write("|-------|--------|-------|\n")
            
            for model, model_info in summary["models"].items():
                status = "✅ Pass" if model_info["success"] else "❌ Fail"
                tests_count = len(results[model])
                tests_pass = sum(1 for r in results[model] if r["success"])
                f.write(f"| {model} | {status} | {tests_pass}/{tests_count} |\n")
        
        logger.info(f"✅ Generated summary report: {summary_path}")
        logger.info(f"✅ Generated markdown report: {report_path}")
        
        # Print summary
        print("\n# Hugging Face Models Test Report")
        print(f"\nTotal Models: {summary['total_models']}")
        print(f"Successful Models: {summary['successful_models']} ({summary['coverage_percentage']:.1f}%)")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful Tests: {summary['successful_tests']} ({summary['successful_tests']/summary['total_tests']*100 if summary['total_tests'] else 0:.1f}%)")
        
        return True
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return False

def full_integration(models=None, verify=True, force=False, run_tests=False, report=True):
    """
    Run the full integration process.
    
    Args:
        models: List of model types to process (None for all)
        verify: Whether to verify syntax after each operation
        force: Whether to force regeneration of existing files
        run_tests: Whether to run the generated tests
        report: Whether to generate a final report
    
    Returns:
        Success status (bool)
    """
    if models is None:
        # Default to core models if none specified
        models = ["bert", "gpt2", "t5", "vit"]
    
    logger.info(f"Running full integration for models: {models}")
    
    # Track results
    results = {
        "generated": [],
        "fixed": [],
        "verified": [],
        "tested": [],
        "failed": []
    }
    
    # Process each model
    for model_type in models:
        file_path = os.path.join(ROOT_DIR, f"test_hf_{model_type}.py")
        logger.info(f"Processing model: {model_type}")
        
        # Generate test file
        if generate_test_file(model_type, force=force, verify=False):
            results["generated"].append(model_type)
            logger.info(f"✅ Generated test file for {model_type}")
        else:
            results["failed"].append((model_type, "generation"))
            logger.error(f"❌ Failed to generate test file for {model_type}")
            continue
        
        # Fix indentation
        if fix_test_file(file_path, verify=False):
            results["fixed"].append(model_type)
            logger.info(f"✅ Fixed indentation for {model_type}")
        else:
            results["failed"].append((model_type, "indentation"))
            logger.error(f"❌ Failed to fix indentation for {model_type}")
            continue
        
        # Verify syntax
        if verify and verify_test_file(file_path):
            results["verified"].append(model_type)
            logger.info(f"✅ Verified syntax for {model_type}")
        else:
            results["failed"].append((model_type, "verification"))
            logger.error(f"❌ Failed to verify syntax for {model_type}")
            continue
        
        # Run test
        if run_tests and run_test_file(file_path):
            results["tested"].append(model_type)
            logger.info(f"✅ Test passed for {model_type}")
        elif run_tests:
            results["failed"].append((model_type, "testing"))
            logger.error(f"❌ Test failed for {model_type}")
            continue
    
    # Generate report if requested
    if report and run_tests:
        generate_report()
    
    # Print summary
    print("\n# Integration Summary")
    print(f"\nGenerated: {len(results['generated'])}/{len(models)}")
    print(f"Fixed: {len(results['fixed'])}/{len(models)}")
    print(f"Verified: {len(results['verified'])}/{len(models)}")
    if run_tests:
        print(f"Tested: {len(results['tested'])}/{len(models)}")
    print(f"Failed: {len(results['failed'])}")
    
    if results['failed']:
        print("\nFailed models:")
        for model, stage in results['failed']:
            print(f"  - {model}: {stage}")
    
    return len(results['failed']) == 0

def main():
    parser = argparse.ArgumentParser(description="Test Integration Script")
    
    # Action arguments
    action_group = parser.add_argument_group("Actions")
    action_group.add_argument("--generate", action="store_true", help="Generate test files")
    action_group.add_argument("--fix", action="store_true", help="Fix indentation in test files")
    action_group.add_argument("--verify", action="store_true", help="Verify syntax of test files")
    action_group.add_argument("--run", action="store_true", help="Run test files")
    action_group.add_argument("--report", action="store_true", help="Generate report")
    action_group.add_argument("--all", action="store_true", help="Run full integration")
    
    # Model selection
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument("--models", type=str, help="Comma-separated list of model types")
    model_group.add_argument("--core", action="store_true", help="Use core models (bert, gpt2, t5, vit)")
    model_group.add_argument("--arch", type=str, choices=ARCHITECTURE_TYPES.keys(), 
                            help="Select models by architecture type")
    
    # Other options
    parser.add_argument("--force", action="store_true", help="Force operations even if files exist")
    parser.add_argument("--list", action="store_true", help="List available model types")
    
    args = parser.parse_args()
    
    # List model types if requested
    if args.list:
        model_types = list_model_types()
        print("\nAvailable Model Types:")
        for model, arch in model_types:
            print(f"  - {model} ({arch})")
        return 0
    
    # Determine models to process
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.core:
        models = ["bert", "gpt2", "t5", "vit"]
    elif args.arch:
        models = [model for model, arch in list_model_types() if arch == args.arch]
    
    # Run requested actions
    if args.all:
        # Run full integration
        success = full_integration(
            models=models, 
            verify=True, 
            force=args.force, 
            run_tests=True, 
            report=True
        )
        return 0 if success else 1
    
    # Run individual actions
    if args.generate:
        if not models:
            logger.error("No models specified. Use --models, --core, or --arch")
            return 1
            
        for model in models:
            generate_test_file(model, force=args.force, verify=args.verify)
    
    if args.fix:
        files = find_test_files()
        if not files:
            logger.error("No test files found")
            return 1
            
        for file in files:
            if models is None or any(f"test_hf_{model}.py" in file for model in models):
                fix_test_file(file, verify=args.verify)
    
    if args.verify:
        files = find_test_files()
        if not files:
            logger.error("No test files found")
            return 1
            
        for file in files:
            if models is None or any(f"test_hf_{model}.py" in file for model in models):
                verify_test_file(file)
    
    if args.run:
        files = find_test_files()
        if not files:
            logger.error("No test files found")
            return 1
            
        for file in files:
            if models is None or any(f"test_hf_{model}.py" in file for model in models):
                run_test_file(file)
    
    if args.report:
        generate_report()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())