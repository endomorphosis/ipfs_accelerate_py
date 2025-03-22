#!/usr/bin/env python3
"""
Generate metadata for HuggingFace model tests, focusing on naming conventions.

This script extracts and documents:
1. The mapping of hyphenated model names to valid Python identifiers
2. Class name capitalization patterns for different models
3. Architecture types for different models

Usage:
    python generate_model_metadata.py [--output OUTPUT_FILE]
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import helpers from fix_hyphenated_models
try:
    from fix_hyphenated_models import (
        to_valid_identifier,
        get_class_name_capitalization,
        get_upper_case_name,
        get_architecture_type,
        HYPHENATED_MODEL_MAPS,
        ARCHITECTURE_TYPES,
        CLASS_NAME_FIXES
    )
except ImportError:
    logger.error("Could not import fix_hyphenated_models.py. Make sure it exists in the same directory.")
    sys.exit(1)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = CURRENT_DIR

def find_all_model_names() -> List[str]:
    """Find all model names from architecture types."""
    all_models = []
    for arch_type, models in ARCHITECTURE_TYPES.items():
        all_models.extend(models)
    
    # Remove duplicates while preserving order
    unique_models = []
    seen = set()
    for model in all_models:
        if model not in seen:
            unique_models.append(model)
            seen.add(model)
    
    return sorted(unique_models)

def find_hyphenated_model_names() -> List[str]:
    """Find all hyphenated model names."""
    all_models = find_all_model_names()
    return [model for model in all_models if "-" in model]

def get_all_transformers_classes() -> List[str]:
    """Get a list of all valid transformers model classes."""
    # This would ideally introspect the transformers library
    # For now, we'll return a static list of known classes
    return [
        "BertForMaskedLM",
        "GPT2LMHeadModel", 
        "T5ForConditionalGeneration",
        "ViTForImageClassification",
        "GPTJForCausalLM",
        "GPTNeoForCausalLM",
        "GPTNeoXForCausalLM",
        "XLMRobertaForMaskedLM"
    ]

def generate_model_mappings() -> Dict:
    """Generate a comprehensive mapping of model names to identifiers and classes."""
    mappings = {}
    
    # Get all models
    all_models = find_all_model_names()
    
    for model in all_models:
        valid_id = to_valid_identifier(model)
        class_name = get_class_name_capitalization(model)
        upper_name = get_upper_case_name(model)
        arch_type = get_architecture_type(model)
        
        mappings[model] = {
            "valid_identifier": valid_id,
            "class_name": class_name,
            "upper_name": upper_name,
            "architecture_type": arch_type,
            "is_hyphenated": "-" in model,
            "test_file_name": f"test_hf_{valid_id}.py",
            "test_class_name": f"Test{class_name}Models"
        }
    
    return mappings

def generate_metadata() -> Dict:
    """Generate comprehensive metadata about model naming conventions."""
    metadata = {
        "model_mappings": generate_model_mappings(),
        "architecture_types": ARCHITECTURE_TYPES,
        "hyphenated_models": find_hyphenated_model_names(),
        "class_name_fixes": CLASS_NAME_FIXES,
        "summary": {
            "total_models": len(find_all_model_names()),
            "hyphenated_models": len(find_hyphenated_model_names()),
            "architecture_types": len(ARCHITECTURE_TYPES)
        }
    }
    
    return metadata

def generate_markdown_report(metadata: Dict) -> str:
    """Generate a markdown report from metadata."""
    report = "# HuggingFace Model Naming Conventions\n\n"
    
    # Summary section
    report += "## Summary\n\n"
    report += f"- Total models: {metadata['summary']['total_models']}\n"
    report += f"- Hyphenated models: {metadata['summary']['hyphenated_models']}\n"
    report += f"- Architecture types: {metadata['summary']['architecture_types']}\n\n"
    
    # Architecture types section
    report += "## Architecture Types\n\n"
    for arch_type, models in metadata['architecture_types'].items():
        report += f"### {arch_type}\n\n"
        for model in models:
            report += f"- {model}\n"
        report += "\n"
    
    # Hyphenated models section
    report += "## Hyphenated Model Mappings\n\n"
    report += "These models require special handling in Python code due to hyphens in their names.\n\n"
    report += "| Original Name | Valid Identifier | Class Name | Test File Name | Test Class Name |\n"
    report += "|---------------|-----------------|------------|----------------|----------------|\n"
    
    for model in metadata['hyphenated_models']:
        mapping = metadata['model_mappings'][model]
        report += f"| {model} | {mapping['valid_identifier']} | {mapping['class_name']} | {mapping['test_file_name']} | {mapping['test_class_name']} |\n"
    
    # Class name fixes section
    report += "\n## Class Name Capitalization Fixes\n\n"
    report += "These class names require specific capitalization patterns.\n\n"
    report += "| Incorrect | Correct |\n"
    report += "|-----------|--------|\n"
    
    for incorrect, correct in metadata['class_name_fixes'].items():
        report += f"| {incorrect} | {correct} |\n"
    
    return report

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate model metadata and naming conventions")
    parser.add_argument("--output", type=str, default="model_naming_conventions.md",
                       help="Output file for the markdown report")
    parser.add_argument("--json", type=str, default="model_metadata.json",
                       help="Output file for the JSON metadata")
    
    args = parser.parse_args()
    
    # Generate metadata
    logger.info("Generating model metadata...")
    metadata = generate_metadata()
    
    # Generate and save markdown report
    report = generate_markdown_report(metadata)
    with open(os.path.join(OUTPUT_DIR, args.output), 'w') as f:
        f.write(report)
    logger.info(f"Markdown report saved to {args.output}")
    
    # Save JSON metadata
    with open(os.path.join(OUTPUT_DIR, args.json), 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"JSON metadata saved to {args.json}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())