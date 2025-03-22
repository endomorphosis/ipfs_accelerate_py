#!/usr/bin/env python3
"""
Script to generate test files for missing models identified in
the HF_MODEL_COVERAGE_ROADMAP.md that are not yet implemented.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure enhanced_generator is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path is set
try:
    from enhanced_generator import (
        MODEL_REGISTRY, 
        generate_test, 
        get_model_architecture,
        validate_generated_file
    )
except ImportError as e:
    logger.error(f"Failed to import from enhanced_generator: {e}")
    logger.error("Make sure to update enhanced_generator.py first with update_model_registry.py")
    sys.exit(1)

# Define priority model lists from the roadmap
CRITICAL_PRIORITY_MODELS = [
    "gpt-j", 
    "flan-t5", 
    "xlm-roberta", 
    "vision-text-dual-encoder"
]

HIGH_PRIORITY_MODELS = [
    "codellama", 
    "gpt-neo", 
    "gpt-neox", 
    "qwen2", 
    "qwen3",
    "longt5", 
    "pegasus-x", 
    "luke", 
    "mpnet",
    "fuyu", 
    "kosmos-2", 
    "llava-next", 
    "video-llava",
    "bark", 
    "mobilenet-v2",
    "blip-2", 
    "chinese-clip", 
    "clipseg"
]

# Medium priority models are the remaining models in MODEL_REGISTRY
# that are not in CRITICAL_PRIORITY_MODELS or HIGH_PRIORITY_MODELS

def get_missing_models() -> Dict[str, List[str]]:
    """
    Identify missing models by comparing the roadmap priorities
    against what's currently implemented.
    """
    # Get lists of model names from the roadmap
    implemented_models = set(MODEL_REGISTRY.keys())
    
    missing_critical = [model for model in CRITICAL_PRIORITY_MODELS 
                        if model.replace('-', '_') not in implemented_models 
                        and model not in implemented_models]
    
    missing_high = [model for model in HIGH_PRIORITY_MODELS 
                    if model.replace('-', '_') not in implemented_models 
                    and model not in implemented_models]
    
    # Get remaining models from the roadmap (medium priority)
    with open('skills/HF_MODEL_COVERAGE_ROADMAP.md', 'r') as f:
        roadmap_content = f.read()
    
    # Extract all models marked as implemented
    all_implemented = set()
    for line in roadmap_content.splitlines():
        if line.startswith('- [x]'):
            model_name = line.split('- [x]')[1].strip().split(' ')[0].lower()
            all_implemented.add(model_name)
    
    # Identify medium priority models not in critical or high
    medium_priority = all_implemented - set(CRITICAL_PRIORITY_MODELS) - set(HIGH_PRIORITY_MODELS)
    missing_medium = [model for model in medium_priority 
                     if model.replace('-', '_') not in implemented_models 
                     and model not in implemented_models]
    
    return {
        "critical": missing_critical,
        "high": missing_high,
        "medium": missing_medium
    }

def generate_missing_model_tests(output_dir: str, priority: str = "all") -> Dict[str, Any]:
    """
    Generate test files for missing models.
    
    Args:
        output_dir: Directory to output the generated files
        priority: Priority level to generate ("critical", "high", "medium", or "all")
        
    Returns:
        Dictionary with status information about generated files
    """
    missing_models = get_missing_models()
    results = {
        "generated": 0,
        "failed": 0,
        "skipped": 0,
        "details": {}
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which priority levels to process
    priorities_to_process = ["critical", "high", "medium"] if priority == "all" else [priority]
    
    for current_priority in priorities_to_process:
        if current_priority not in missing_models:
            logger.warning(f"No models found for priority: {current_priority}")
            continue
            
        models = missing_models[current_priority]
        logger.info(f"Processing {len(models)} {current_priority} priority models")
        
        for model_type in models:
            # Try with both original and normalized names
            model_variants = [model_type]
            if "-" in model_type:
                model_variants.append(model_type.replace("-", "_"))
            
            success = False
            error_msg = ""
            
            for variant in model_variants:
                try:
                    logger.info(f"Generating test for {variant} (original: {model_type})")
                    result = generate_test(variant, output_dir)
                    
                    # Validate the generated file
                    is_valid, validation_msg = validate_generated_file(result["file_path"])
                    if is_valid:
                        logger.info(f"Successfully generated and validated test: {result['file_path']}")
                        results["generated"] += 1
                        results["details"][model_type] = {
                            "status": "success", 
                            "file_path": result["file_path"],
                            "architecture": result["architecture"]
                        }
                        success = True
                        break
                    else:
                        error_msg = f"Validation failed: {validation_msg}"
                except Exception as e:
                    error_msg = f"Generation failed: {str(e)}"
                    continue
            
            if not success:
                logger.error(f"Failed to generate test for {model_type}: {error_msg}")
                results["failed"] += 1
                results["details"][model_type] = {
                    "status": "failed", 
                    "error": error_msg
                }
    
    # Generate a report
    report_path = os.path.join(output_dir, "generation_report.md")
    with open(report_path, "w") as f:
        f.write(f"# Model Test Generation Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Generated:** {results['generated']}\n")
        f.write(f"- **Failed:** {results['failed']}\n")
        f.write(f"- **Skipped:** {results['skipped']}\n\n")
        
        f.write(f"## Details\n\n")
        for model_type, details in results["details"].items():
            if details["status"] == "success":
                f.write(f"### ✅ {model_type}\n")
                f.write(f"- Architecture: {details['architecture']}\n")
                f.write(f"- File path: {details['file_path']}\n\n")
            else:
                f.write(f"### ❌ {model_type}\n")
                f.write(f"- Error: {details['error']}\n\n")
    
    logger.info(f"Generation report written to: {report_path}")
    return results

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate test files for missing models")
    parser.add_argument("--output-dir", default="missing_model_tests", 
                        help="Directory to output the generated files")
    parser.add_argument("--priority", choices=["critical", "high", "medium", "all"], 
                        default="all", help="Priority level to generate")
    args = parser.parse_args()
    
    results = generate_missing_model_tests(args.output_dir, args.priority)
    
    logger.info(f"Summary:")
    logger.info(f"  Generated: {results['generated']}")
    logger.info(f"  Failed: {results['failed']}")
    logger.info(f"  Skipped: {results['skipped']}")
    
    return 0 if results["failed"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())