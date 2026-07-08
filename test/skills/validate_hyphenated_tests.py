#!/usr/bin/env python3
"""
Validate test files for hyphenated model names.

This script checks the syntax of generated test files for hyphenated model names.
It verifies that the files compile correctly, contain the necessary model name references,
and follow the naming conventions.
"""

import os
import sys
import re
import logging
from pathlib import Path
import argparse
import traceback
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
FIXED_TESTS_DIR = CURRENT_DIR / "fixed_tests"
VALIDATION_REPORT_PATH = CURRENT_DIR / "hyphenated_validation_report.md"

def check_file_syntax(file_path: Path) -> Tuple[bool, str]:
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        compile(content, file_path, 'exec')
        return True, "Syntax OK"
    except SyntaxError as e:
        error_message = f"Syntax error on line {e.lineno}: {e.msg}"
        if hasattr(e, 'text') and e.text:
            error_message += f"\n{e.text}"
            if hasattr(e, 'offset') and e.offset:
                error_message += "\n" + " " * (e.offset - 1) + "^"
        
        return False, error_message
    except Exception as e:
        return False, str(e)

def validate_hyphenated_model_file(file_path: Path) -> Dict:
    """Validate a test file for a hyphenated model name."""
    result = {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "syntax_valid": False,
        "model_class_found": False,
        "registry_variable_found": False,
        "errors": []
    }
    
    try:
        # Check syntax
        syntax_valid, error_message = check_file_syntax(file_path)
        result["syntax_valid"] = syntax_valid
        
        if not syntax_valid:
            result["errors"].append(f"Syntax error: {error_message}")
            return result
        
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract model name from file name (test_hf_gpt_j.py -> gpt_j)
        match = re.search(r'test_hf_(.+?)\.py', file_path.name)
        if match:
            model_name = match.group(1)
            
            # Check for model class
            model_class_pattern = rf'Test([A-Za-z0-9]+)Models'
            model_class_match = re.search(model_class_pattern, content)
            
            if model_class_match:
                result["model_class_found"] = True
                result["model_class_name"] = model_class_match.group(0)
            else:
                result["errors"].append(f"Model class not found (Test*Models)")
            
            # Check for registry variable
            registry_var_pattern = r'([A-Za-z0-9_]+)_MODELS_REGISTRY'
            registry_var_match = re.search(registry_var_pattern, content)
            
            if registry_var_match:
                result["registry_variable_found"] = True
                result["registry_variable_name"] = registry_var_match.group(0)
            else:
                result["errors"].append(f"Registry variable not found (*_MODELS_REGISTRY)")
            
            # Additional validations can be added here
            
        else:
            result["errors"].append(f"Could not extract model name from file name: {file_path.name}")
        
    except Exception as e:
        result["errors"].append(f"Error validating file: {str(e)}")
        logger.error(f"Error validating {file_path}: {traceback.format_exc()}")
    
    return result

def find_hyphenated_model_files() -> List[Path]:
    """Find all test files for hyphenated model names."""
    if not FIXED_TESTS_DIR.exists():
        logger.warning(f"Fixed tests directory not found: {FIXED_TESTS_DIR}")
        return []
    
    # Pattern matches test_hf_model_name.py where model_name contains an underscore (from hyphen)
    pattern = r'test_hf_[a-z0-9]+_[a-z0-9_]+\.py'
    
    files = []
    for file_path in FIXED_TESTS_DIR.glob("test_hf_*.py"):
        if re.match(pattern, file_path.name):
            files.append(file_path)
    
    return files

def validate_all_files() -> List[Dict]:
    """Validate all files for hyphenated model names."""
    files = find_hyphenated_model_files()
    logger.info(f"Found {len(files)} files for hyphenated model names")
    
    results = []
    for file_path in files:
        logger.info(f"Validating {file_path.name}")
        result = validate_hyphenated_model_file(file_path)
        results.append(result)
        
        # Log result
        if result["syntax_valid"] and result["model_class_found"] and result["registry_variable_found"]:
            logger.info(f"✅ {file_path.name} is valid")
        else:
            logger.warning(f"❌ {file_path.name} has issues: {result['errors']}")
    
    return results

def generate_validation_report(results: List[Dict]) -> str:
    """Generate a validation report for all files."""
    if not results:
        return "# Hyphenated Model Validation Report\n\nNo files found for validation."
    
    # Count statistics
    total_files = len(results)
    valid_files = sum(1 for r in results if r["syntax_valid"] and r["model_class_found"] and r["registry_variable_found"])
    syntax_errors = sum(1 for r in results if not r["syntax_valid"])
    missing_class = sum(1 for r in results if not r["model_class_found"])
    missing_registry = sum(1 for r in results if not r["registry_variable_found"])
    
    import datetime
    # Generate report
    report = [
        "# Hyphenated Model Validation Report",
        "",
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
        "",
        "## Summary",
        "",
        f"- Total Files: {total_files}",
        f"- Valid Files: {valid_files} ({valid_files/total_files*100:.1f}%)",
        f"- Files with Syntax Errors: {syntax_errors}",
        f"- Files Missing Model Class: {missing_class}",
        f"- Files Missing Registry Variable: {missing_registry}",
        "",
        "## Detailed Results",
        ""
    ]
    
    # Add detailed results
    for result in results:
        file_name = result["file_name"]
        status = "✅ Valid" if (result["syntax_valid"] and result["model_class_found"] and result["registry_variable_found"]) else "❌ Invalid"
        
        report.append(f"### {file_name} - {status}")
        report.append("")
        
        report.append(f"- Syntax Valid: {'✅' if result['syntax_valid'] else '❌'}")
        report.append(f"- Model Class Found: {'✅' if result['model_class_found'] else '❌'}")
        if result.get("model_class_name"):
            report.append(f"  - Class Name: `{result['model_class_name']}`")
        
        report.append(f"- Registry Variable Found: {'✅' if result['registry_variable_found'] else '❌'}")
        if result.get("registry_variable_name"):
            report.append(f"  - Variable Name: `{result['registry_variable_name']}`")
        
        if result["errors"]:
            report.append("")
            report.append("**Errors:**")
            for error in result["errors"]:
                report.append(f"- {error}")
        
        report.append("")
    
    return "\n".join(report)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate test files for hyphenated model names")
    parser.add_argument("--file", type=str, help="Validate a specific file")
    parser.add_argument("--report", action="store_true", help="Generate a markdown report")
    parser.add_argument("--report-path", type=Path, default=VALIDATION_REPORT_PATH, help="Path for the report file")
    
    args = parser.parse_args()
    
    if args.file:
        # Validate specific file
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return 1
        
        result = validate_hyphenated_model_file(file_path)
        
        if result["syntax_valid"] and result["model_class_found"] and result["registry_variable_found"]:
            logger.info(f"✅ {file_path.name} is valid")
            return 0
        else:
            logger.error(f"❌ {file_path.name} has issues: {result['errors']}")
            return 1
    
    else:
        # Validate all files
        results = validate_all_files()
        
        # Generate report if requested
        if args.report:
            report = generate_validation_report(results)
            with open(args.report_path, 'w') as f:
                f.write(report)
            logger.info(f"Report written to {args.report_path}")
        
        # Return success if all files are valid
        valid_files = sum(1 for r in results if r["syntax_valid"] and r["model_class_found"] and r["registry_variable_found"])
        return 0 if valid_files == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())