#!/usr/bin/env python3
"""
Unified test runner for all Hugging Face model families.

This script provides a centralized interface for testing different model families,
generating reports, and summarizing results across model architectures.
"""

import os
import sys
import json
import time
import datetime
import argparse
import logging
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = CURRENT_DIR / "collected_results"
SUMMARY_FILE = CURRENT_DIR / "test_summary.json"
REPORT_FILE = CURRENT_DIR / "test_report.md"

# Map of model categories to test modules
MODEL_FAMILIES = {
    "bert": {
        "module": "test_simplified",
        "description": "BERT-family masked language models",
        "default_model": "bert-base-uncased",
        "class": "TestSimpleModel",
        "status": "complete"
    },
    "gpt2": {
        "module": "test_simplified",
        "description": "GPT-2 causal language models",
        "default_model": "gpt2",
        "class": "TestSimpleModel",
        "status": "complete"
    },
    "t5": {
        "module": "test_simplified",
        "description": "T5 encoder-decoder models",
        "default_model": "t5-small",
        "class": "TestSimpleModel",
        "status": "complete"
    },
    "clip": {
        "module": "test_simplified",
        "description": "CLIP vision-language models",
        "default_model": "openai/clip-vit-base-patch32",
        "class": "TestSimpleModel",
        "status": "complete"
    },
    "llama": {
        "module": "test_simplified",
        "description": "LLaMA causal language models",
        "default_model": "meta-llama/Llama-2-7b-hf",
        "class": "TestSimpleModel",
        "status": "complete"
    },
    "whisper": {
        "module": "test_simplified",
        "description": "Whisper speech recognition models",
        "default_model": "openai/whisper-tiny",
        "class": "TestSimpleModel",
        "status": "complete"
    },
    "wav2vec2": {
        "module": "test_simplified",
        "description": "Wav2Vec2 speech models",
        "default_model": "facebook/wav2vec2-base",
        "class": "TestSimpleModel",
        "status": "complete"
    }
}

def get_available_model_families():
    """Get list of available model families that have implemented test modules."""
    available_families = {}
    
    for family_id, family_info in MODEL_FAMILIES.items():
        module_path = CURRENT_DIR / f"{family_info['module']}.py"
        if module_path.exists():
            available_families[family_id] = family_info
    
    return available_families

def import_model_tester(family_id):
    """Import the testing module for a specific model family."""
    family_info = MODEL_FAMILIES.get(family_id)
    if not family_info:
        raise ValueError(f"Unknown model family: {family_id}")
    
    try:
        # Add CWD to Python path to ensure local imports work
        sys.path.insert(0, str(CURRENT_DIR))
        
        module_name = family_info["module"]
        logger.info(f"Importing module: {module_name}")
        
        module = importlib.import_module(module_name)
        
        # Check if the module has necessary functions
        if not hasattr(module, "get_available_models"):
            logger.warning(f"Module {module_name} doesn't have get_available_models function")
        
        if not hasattr(module, family_info["class"]):
            logger.warning(f"Module {module_name} doesn't have {family_info['class']} class")
            
        # Check if the class has run_tests method
        class_obj = getattr(module, family_info["class"])
        instance = class_obj()
        if not hasattr(instance, "run_tests"):
            logger.warning(f"Class {family_info['class']} doesn't have run_tests method")
        
        return module
    except ImportError as e:
        logger.error(f"Failed to import module {family_info['module']}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error in import_model_tester: {e}")
        return None

def run_model_test(family_id, model_id=None, all_models=False, all_hardware=False, save=True):
    """Run tests for a specific model or all models in a family."""
    family_info = MODEL_FAMILIES.get(family_id)
    if not family_info:
        logger.error(f"Unknown model family: {family_id}")
        return None
    
    # Import the module
    module = import_model_tester(family_id)
    if not module:
        return None
    
    results = {}
    
    # Test a specific model
    if model_id and not all_models:
        logger.info(f"Testing model {model_id} from family {family_id}")
        
        # Get the test class
        tester_class = getattr(module, family_info["class"])
        tester = tester_class(model_id)
        
        # Run tests
        model_results = tester.run_tests(all_hardware=all_hardware)
        
        # Save results
        if save:
            output_path = module.save_results(model_id, model_results, output_dir=RESULTS_DIR)
            logger.info(f"Saved results to {output_path}")
        
        results[model_id] = {
            "success": any(r.get("pipeline_success", False) for r in model_results["results"].values() 
                          if r.get("pipeline_success") is not False)
        }
    
    # Test all models in the family
    elif all_models:
        if hasattr(module, "test_all_models"):
            logger.info(f"Testing all models in family {family_id}")
            results = module.test_all_models(output_dir=RESULTS_DIR, all_hardware=all_hardware)
        else:
            logger.error(f"Module {family_info['module']} doesn't support testing all models")
    
    return results

def generate_summary_report(results_by_family):
    """Generate a summary report of test results across all families."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate success rates
    total_models = 0
    successful_models = 0
    family_stats = {}
    
    for family_id, family_results in results_by_family.items():
        family_total = len(family_results)
        family_success = sum(1 for r in family_results.values() if r.get("success", False))
        
        total_models += family_total
        successful_models += family_success
        
        family_stats[family_id] = {
            "total": family_total,
            "successful": family_success,
            "success_rate": (family_success / family_total) * 100 if family_total > 0 else 0
        }
    
    # Create summary data
    summary = {
        "timestamp": timestamp,
        "total_models": total_models,
        "successful_models": successful_models,
        "success_rate": (successful_models / total_models) * 100 if total_models > 0 else 0,
        "by_family": family_stats
    }
    
    # Save JSON summary
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Generate markdown report
    with open(REPORT_FILE, "w") as f:
        f.write("# Hugging Face Model Test Report\n\n")
        f.write(f"*Generated: {timestamp}*\n\n")
        
        f.write("## Overall Results\n\n")
        f.write(f"- **Total Models Tested**: {total_models}\n")
        f.write(f"- **Successfully Tested**: {successful_models}\n")
        f.write(f"- **Success Rate**: {summary['success_rate']:.1f}%\n\n")
        
        f.write("## Results by Model Family\n\n")
        f.write("| Family | Description | Models Tested | Success Rate |\n")
        f.write("|--------|-------------|---------------|-------------|\n")
        
        for family_id, stats in family_stats.items():
            family_info = MODEL_FAMILIES.get(family_id, {})
            description = family_info.get("description", "Unknown")
            f.write(f"| {family_id} | {description} | {stats['total']} | {stats['success_rate']:.1f}% |\n")
        
        f.write("\n## Detailed Results\n\n")
        for family_id, family_results in results_by_family.items():
            f.write(f"### {family_id.upper()} Models\n\n")
            
            f.write("| Model | Success | Notes |\n")
            f.write("|-------|---------|-------|\n")
            
            for model_id, result in family_results.items():
                success = "✅" if result.get("success", False) else "❌"
                notes = result.get("notes", "")
                f.write(f"| {model_id} | {success} | {notes} |\n")
            
            f.write("\n")
    
    logger.info(f"Generated summary report: {REPORT_FILE}")
    return summary

def main():
    """Command-line entry point."""
    global RESULTS_DIR
    
    parser = argparse.ArgumentParser(description="Test Hugging Face models")
    
    # Model family selection
    family_group = parser.add_mutually_exclusive_group()
    family_group.add_argument("--family", type=str, help="Model family to test")
    family_group.add_argument("--all-families", action="store_true", help="Test all model families")
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, help="Specific model to test")
    model_group.add_argument("--all-models", action="store_true", help="Test all models in the family")
    
    # Testing options
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    parser.add_argument("--cpu-only", action="store_true", help="Test only on CPU")
    
    # Output options
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR), help="Directory for output files")
    
    # List options
    parser.add_argument("--list-families", action="store_true", help="List all available model families")
    parser.add_argument("--list-models", type=str, help="List all available models in a family")
    
    args = parser.parse_args()
    
    # Update output directory
    RESULTS_DIR = Path(args.output_dir)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # List model families if requested
    if args.list_families:
        available_families = get_available_model_families()
        
        print("\nAvailable Model Families:")
        for family_id, family_info in MODEL_FAMILIES.items():
            status = "✅ Available" if family_id in available_families else "⏳ Planned"
            print(f"  - {family_id}: {family_info['description']} [{status}]")
        return
    
    # List models in a family if requested
    if args.list_models:
        family_id = args.list_models
        
        if family_id not in MODEL_FAMILIES:
            print(f"Unknown model family: {family_id}")
            return
        
        module = import_model_tester(family_id)
        if not module or not hasattr(module, "get_available_models"):
            print(f"Cannot list models for family: {family_id}")
            return
        
        models = module.get_available_models()
        family_info = MODEL_FAMILIES[family_id]
        
        print(f"\nAvailable Models in {family_id.upper()} Family:")
        print(f"Description: {family_info['description']}")
        print(f"Models ({len(models)}):")
        
        for model_id in models:
            print(f"  - {model_id}")
        return
    
    # Override hardware if CPU only
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Test a specific family
    if args.family:
        family_id = args.family
        
        if family_id not in MODEL_FAMILIES:
            print(f"Unknown model family: {family_id}")
            return
        
        model_id = args.model or MODEL_FAMILIES[family_id]["default_model"]
        results = {family_id: run_model_test(
            family_id=family_id,
            model_id=model_id,
            all_models=args.all_models,
            all_hardware=args.all_hardware,
            save=not args.no_save
        )}
    
    # Test all families
    elif args.all_families:
        available_families = get_available_model_families()
        results = {}
        
        for family_id in available_families:
            logger.info(f"Testing family: {family_id}")
            
            family_results = run_model_test(
                family_id=family_id,
                all_models=args.all_models,
                all_hardware=args.all_hardware,
                save=not args.no_save
            )
            
            if family_results:
                results[family_id] = family_results
    
    # Default: test BERT with default model
    else:
        default_family = "bert"
        default_model = MODEL_FAMILIES[default_family]["default_model"]
        
        print(f"No specific family requested, testing default model: {default_model}")
        results = {default_family: run_model_test(
            family_id=default_family,
            model_id=default_model,
            all_hardware=args.all_hardware,
            save=not args.no_save
        )}
    
    # Generate summary if we ran multiple tests
    if args.all_models or args.all_families:
        generate_summary_report(results)

if __name__ == "__main__":
    main()