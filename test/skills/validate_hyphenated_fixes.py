#!/usr/bin/env python3
"""
Validate fixes for hyphenated model names in HuggingFace test files.

This script checks that:
1. All hyphenated model names have corresponding valid test files
2. The test files have valid Python syntax
3. The naming conventions and class naming are consistent

Usage:
    python validate_hyphenated_fixes.py [--report] [--fix-issues]
"""

import os
import sys
import re
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the fix_hyphenated_models module
try:
    from fix_hyphenated_models import (
        to_valid_identifier, 
        get_class_name_capitalization,
        find_hyphenated_models,
        check_file_syntax,
        create_test_file
    )
except ImportError:
    logger.error("Could not import fix_hyphenated_models.py. Make sure it exists in the same directory.")
    sys.exit(1)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
FIXED_TESTS_DIR = CURRENT_DIR / "fixed_tests"

def get_test_file_path(model_name: str) -> Path:
    """Get the expected test file path for a model name."""
    valid_id = to_valid_identifier(model_name)
    return FIXED_TESTS_DIR / f"test_hf_{valid_id}.py"

def check_test_file_exists(model_name: str) -> bool:
    """Check if a test file exists for a model name."""
    test_file = get_test_file_path(model_name)
    return test_file.exists()

def validate_test_file_content(model_name: str) -> Tuple[bool, List[str]]:
    """Validate the content of a test file for syntax and naming consistency."""
    test_file = get_test_file_path(model_name)
    if not test_file.exists():
        return False, [f"Test file does not exist: {test_file}"]
    
    # Read the file content
    with open(test_file, 'r') as f:
        content = f.read()
    
    # List of validation issues
    issues = []
    
    # Check for valid Python syntax
    syntax_valid, error = check_file_syntax(content, str(test_file))
    if not syntax_valid:
        issues.append(f"Syntax error: {error}")
    
    # Get the expected identifiers for the model
    valid_id = to_valid_identifier(model_name)
    class_name = get_class_name_capitalization(model_name)
    
    # Check for class declaration
    test_class_name = f"Test{class_name}Models"
    if test_class_name not in content:
        issues.append(f"Missing or incorrect test class name: expected {test_class_name}")
    
    # Check for model registry
    registry_pattern = rf"({valid_id.upper()}_MODELS_REGISTRY|{model_name.replace('-', '_').upper()}_MODELS_REGISTRY)"
    if not re.search(registry_pattern, content):
        issues.append(f"Missing or incorrect model registry: expected {valid_id.upper()}_MODELS_REGISTRY")
    
    # Check for valid identifiers (no hyphens in variable names)
    if "-" in model_name:
        if model_name in content:
            # It's okay for the model name to appear in strings or comments
            if re.search(rf"[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.*{model_name}", content):
                issues.append(f"Found hyphenated model name in variable assignment: {model_name}")
            
            if re.search(rf"def\s+\w+_{model_name}", content):
                issues.append(f"Found hyphenated model name in function definition: {model_name}")
    
    return len(issues) == 0, issues

def validate_all_hyphenated_models() -> Dict[str, Dict]:
    """Validate all hyphenated models and return validation results."""
    results = {}
    hyphenated_models = find_hyphenated_models()
    
    for model_name in hyphenated_models:
        logger.info(f"Validating {model_name}...")
        
        # Check if test file exists
        file_exists = check_test_file_exists(model_name)
        
        # Validate content if file exists
        if file_exists:
            valid, issues = validate_test_file_content(model_name)
        else:
            valid = False
            issues = [f"Test file does not exist"]
        
        results[model_name] = {
            "exists": file_exists,
            "valid": valid,
            "issues": issues
        }
    
    return results

def generate_validation_report(results: Dict[str, Dict]) -> str:
    """Generate a validation report from the results."""
    report = "# Hyphenated Model Validation Report\n\n"
    
    # Summary stats
    total = len(results)
    existing = sum(1 for r in results.values() if r["exists"])
    valid = sum(1 for r in results.values() if r["valid"])
    
    report += f"## Summary\n\n"
    report += f"- Total hyphenated models: {total}\n"
    report += f"- Existing test files: {existing}/{total} ({existing/total*100:.1f}%)\n"
    report += f"- Valid test files: {valid}/{total} ({valid/total*100:.1f}%)\n\n"
    
    # Categorize models
    all_valid = []
    missing = []
    invalid = []
    
    for model_name, result in results.items():
        if result["valid"]:
            all_valid.append(model_name)
        elif not result["exists"]:
            missing.append(model_name)
        else:
            invalid.append(model_name)
    
    # List all valid models
    report += f"## Valid Models ({len(all_valid)})\n\n"
    for model in sorted(all_valid):
        report += f"- {model} ✅\n"
    
    # List missing models
    report += f"\n## Missing Models ({len(missing)})\n\n"
    for model in sorted(missing):
        valid_id = to_valid_identifier(model)
        report += f"- {model} ❌ (expected file: test_hf_{valid_id}.py)\n"
    
    # List invalid models with issues
    report += f"\n## Invalid Models ({len(invalid)})\n\n"
    for model in sorted(invalid):
        report += f"### {model} ❌\n\n"
        report += f"File: test_hf_{to_valid_identifier(model)}.py\n\n"
        report += "Issues:\n"
        for issue in results[model]["issues"]:
            report += f"- {issue}\n"
        report += "\n"
    
    return report

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate fixes for hyphenated model names")
    parser.add_argument("--report", action="store_true", help="Generate a validation report")
    parser.add_argument("--fix-issues", action="store_true", help="Attempt to fix validation issues")
    parser.add_argument("--output-file", type=str, default="hyphenated_model_validation.md", 
                      help="Output file for validation report")
    
    args = parser.parse_args()
    
    # Validate all hyphenated models
    logger.info("Validating all hyphenated models...")
    results = validate_all_hyphenated_models()
    
    # Count validation results
    total = len(results)
    valid = sum(1 for r in results.values() if r["valid"])
    
    logger.info(f"Validation complete: {valid}/{total} models valid")
    
    # Generate and save report if requested
    if args.report:
        report = generate_validation_report(results)
        with open(args.output_file, 'w') as f:
            f.write(report)
        logger.info(f"Validation report saved to {args.output_file}")
    
    # Fix issues if requested
    if args.fix_issues:
        logger.info("Attempting to fix issues...")
        
        # Find models with issues
        models_to_fix = []
        for model_name, result in results.items():
            if not result["valid"]:
                models_to_fix.append(model_name)
        
        # Try to fix each model
        fixed_count = 0
        for model_name in models_to_fix:
            logger.info(f"Fixing {model_name}...")
            success, error = create_test_file(model_name, FIXED_TESTS_DIR)
            if success:
                fixed_count += 1
                logger.info(f"Successfully fixed {model_name}")
            else:
                logger.error(f"Failed to fix {model_name}: {error}")
        
        logger.info(f"Fixed {fixed_count}/{len(models_to_fix)} models with issues")
    
    # Return non-zero exit code if any models are invalid
    return 0 if valid == total else 1

if __name__ == "__main__":
    sys.exit(main())