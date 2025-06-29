#!/usr/bin/env python3
"""
Test validator for HuggingFace model tests.

This module validates that test files comply with the ModelTest pattern.
"""

import os
import sys
import ast
import glob
import logging
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTestVisitor(ast.NodeVisitor):
    """AST visitor to check for ModelTest pattern compliance."""
    
    def __init__(self):
        self.imports = []
        self.classes = []
        self.methods = {}
        self.inherits_model_test = False
        self.model_test_import = None
    
    def visit_Import(self, node):
        for name in node.names:
            self.imports.append(name.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            for name in node.names:
                self.imports.append(f"{node.module}.{name.name}")
                # Check for ModelTest import
                if "ModelTest" in name.name:
                    self.model_test_import = f"{node.module}.{name.name}"
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        methods = []
        
        # Check for inheritance from ModelTest
        for base in node.bases:
            if isinstance(base, ast.Name) and "ModelTest" in base.id:
                self.inherits_model_test = True
            elif isinstance(base, ast.Attribute) and "ModelTest" in base.attr:
                self.inherits_model_test = True
        
        # Record methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)
        
        self.methods[node.name] = methods
        self.generic_visit(node)
    
    def check_required_methods(self, required_methods=None):
        """Check if the class implements required methods."""
        if required_methods is None:
            required_methods = ["get_default_model_id", "run_all_tests"]
        
        missing_methods = []
        for class_name, methods in self.methods.items():
            for required in required_methods:
                if required not in methods:
                    missing_methods.append(f"{class_name}.{required}")
        
        return missing_methods


def validate_syntax(file_path: str) -> bool:
    """
    Validate Python syntax of a file.
    
    Args:
        file_path: Path to file to validate
        
    Returns:
        True if syntax is valid, False otherwise
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", file_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {file_path}: Syntax is valid")
            return True
        else:
            logger.error(f"❌ {file_path}: Syntax error")
            logger.error(result.stderr)
            return False
    except Exception as e:
        logger.error(f"❌ {file_path}: Error validating syntax: {e}")
        return False


def validate_model_test_pattern(file_path: str) -> Dict[str, Any]:
    """
    Validate that a file follows the ModelTest pattern.
    
    Args:
        file_path: Path to file to validate
        
    Returns:
        Dict with validation results
    """
    results = {
        "follows_pattern": False,
        "imports_model_test": False,
        "inherits_model_test": False,
        "has_required_methods": False,
        "missing_methods": [],
        "errors": []
    }
    
    try:
        # Read the file
        with open(file_path, "r") as f:
            content = f.read()
        
        # Parse the AST
        try:
            tree = ast.parse(content)
            visitor = ModelTestVisitor()
            visitor.visit(tree)
            
            # Check imports
            model_test_imports = [imp for imp in visitor.imports if "ModelTest" in imp]
            results["imports_model_test"] = len(model_test_imports) > 0
            results["model_test_import"] = visitor.model_test_import
            
            # Check inheritance
            results["inherits_model_test"] = visitor.inherits_model_test
            
            # Check required methods
            missing_methods = visitor.check_required_methods()
            results["missing_methods"] = missing_methods
            results["has_required_methods"] = len(missing_methods) == 0
            
            # Overall result
            results["follows_pattern"] = (
                results["imports_model_test"] and 
                results["inherits_model_test"] and 
                results["has_required_methods"]
            )
            
            if results["follows_pattern"]:
                logger.info(f"✅ {file_path}: Follows ModelTest pattern")
            else:
                logger.warning(f"⚠️ {file_path}: Does not follow ModelTest pattern")
                if not results["imports_model_test"]:
                    logger.warning(f"  - Missing ModelTest import")
                if not results["inherits_model_test"]:
                    logger.warning(f"  - Does not inherit from ModelTest")
                if not results["has_required_methods"]:
                    logger.warning(f"  - Missing required methods: {', '.join(missing_methods)}")
        except SyntaxError as e:
            results["errors"].append(f"Syntax error: {e}")
            logger.error(f"❌ {file_path}: Syntax error in AST parsing: {e}")
    except Exception as e:
        results["errors"].append(f"Error reading file: {e}")
        logger.error(f"❌ {file_path}: Error reading file: {e}")
    
    return results


def validate_test_file(file_path: str) -> Dict[str, Any]:
    """
    Validate a test file for syntax and ModelTest pattern compliance.
    
    Args:
        file_path: Path to file to validate
        
    Returns:
        Dict with validation results
    """
    results = {
        "file_path": file_path,
        "syntax_valid": False,
        "pattern_valid": False,
        "pattern_details": {},
        "errors": []
    }
    
    # Check syntax
    results["syntax_valid"] = validate_syntax(file_path)
    
    # If syntax is valid, check ModelTest pattern
    if results["syntax_valid"]:
        pattern_results = validate_model_test_pattern(file_path)
        results["pattern_valid"] = pattern_results["follows_pattern"]
        results["pattern_details"] = pattern_results
    
    # Overall result
    results["valid"] = results["syntax_valid"] and results["pattern_valid"]
    
    return results


def validate_test_files(directory: str, pattern: str = "test_hf_*.py") -> Dict[str, Any]:
    """
    Validate all test files in a directory.
    
    Args:
        directory: Directory to search for test files
        pattern: Glob pattern for test files
        
    Returns:
        Dict with validation results
    """
    # Find test files
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    
    if not files:
        logger.warning(f"No files found matching pattern: {search_pattern}")
        return {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "files": []
        }
    
    # Validate each file
    results = []
    for file_path in files:
        result = validate_test_file(file_path)
        results.append(result)
    
    # Calculate statistics
    total = len(results)
    valid = sum(1 for r in results if r["valid"])
    
    return {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "files": results
    }


def generate_validation_report(results: Dict[str, Any], output_file: str = "validation_report.md") -> None:
    """
    Generate a validation report.
    
    Args:
        results: Validation results
        output_file: Path to save report
    """
    with open(output_file, "w") as f:
        f.write("# Test File Validation Report\n\n")
        
        # Write summary
        f.write("## Summary\n\n")
        f.write(f"- **Total files**: {results['total']}\n")
        percentage = results['valid']/results['total']*100 if results['total'] > 0 else 0
        f.write(f"- **Valid files**: {results['valid']} ({percentage:.1f}%)\n")
        f.write(f"- **Invalid files**: {results['invalid']}\n\n")
        
        # Write details for each file
        f.write("## File Details\n\n")
        f.write("| File | Syntax Valid | Pattern Valid | Missing Methods |\n")
        f.write("|------|-------------|---------------|----------------|\n")
        
        for result in sorted(results["files"], key=lambda r: os.path.basename(r["file_path"])):
            file_name = os.path.basename(result["file_path"])
            syntax_valid = "✅" if result["syntax_valid"] else "❌"
            pattern_valid = "✅" if result["pattern_valid"] else "❌"
            
            missing_methods = result.get("pattern_details", {}).get("missing_methods", [])
            missing_str = ", ".join(missing_methods) if missing_methods else "-"
            
            f.write(f"| {file_name} | {syntax_valid} | {pattern_valid} | {missing_str} |\n")
        
        # Write invalid files section
        invalid_files = [r for r in results["files"] if not r["valid"]]
        if invalid_files:
            f.write("\n## Invalid Files\n\n")
            
            for result in invalid_files:
                file_name = os.path.basename(result["file_path"])
                f.write(f"### {file_name}\n\n")
                
                if not result["syntax_valid"]:
                    f.write("- ❌ **Syntax invalid**\n")
                    for error in result.get("errors", []):
                        f.write(f"  - {error}\n")
                
                if result["syntax_valid"] and not result["pattern_valid"]:
                    f.write("- ❌ **ModelTest pattern invalid**\n")
                    pattern_details = result.get("pattern_details", {})
                    
                    if not pattern_details.get("imports_model_test", False):
                        f.write("  - Missing ModelTest import\n")
                    
                    if not pattern_details.get("inherits_model_test", False):
                        f.write("  - Does not inherit from ModelTest\n")
                    
                    missing_methods = pattern_details.get("missing_methods", [])
                    if missing_methods:
                        f.write("  - Missing required methods:\n")
                        for method in missing_methods:
                            f.write(f"    - {method}\n")
                
                f.write("\n")
    
    logger.info(f"Validation report written to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate HuggingFace model test files")
    parser.add_argument("--directory", "-d", default="./", help="Directory to search for test files")
    parser.add_argument("--pattern", "-p", default="test_hf_*.py", help="Glob pattern for test files")
    parser.add_argument("--report", "-r", default="validation_report.md", help="Path to save validation report")
    
    args = parser.parse_args()
    
    # Validate test files
    results = validate_test_files(args.directory, args.pattern)
    
    # Generate report
    generate_validation_report(results, args.report)
    
    # Return success if all files are valid
    sys.exit(0 if results["invalid"] == 0 else 1)