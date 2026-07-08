#!/usr/bin/env python3
"""
Batch validation script for checking ModelTest compliance across all test files.

This script validates all test files in the specified directory (or subdirectories)
and generates a report on compliance with the ModelTest standard.
"""

import os
import sys
import ast
import logging
import argparse
from pathlib import Path
import datetime
import json
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Required methods for ModelTest subclasses
REQUIRED_METHODS = ['test_model_loading', 'load_model', 'verify_model_output', 'detect_preferred_device']

def is_modeltest_subclass(file_path):
    """Check if the file contains a class that inherits from ModelTest."""
    try:
        with open(file_path, "r") as f:
            content = f.read()
            
        tree = ast.parse(content)
        
        # Track imports to find ModelTest
        imports = []
        model_test_import = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and 'model_test' in node.module:
                    for name in node.names:
                        if name.name == 'ModelTest':
                            model_test_import = f"{node.module}.{name.name}"
                            imports.append(model_test_import)
                            
            elif isinstance(node, ast.Import):
                for name in node.names:
                    if 'model_test' in name.name:
                        imports.append(name.name)
                        if name.name.endswith('ModelTest'):
                            model_test_import = name.name
        
        if not model_test_import:
            return False, "ModelTest not imported", None, []
        
        # Find classes that inherit from ModelTest
        valid_classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'ModelTest':
                        valid_classes.append(node.name)
                    elif isinstance(base, ast.Attribute) and base.attr == 'ModelTest':
                        valid_classes.append(node.name)
        
        if not valid_classes:
            return False, "No class inherits from ModelTest", None, []
            
        # Check for required methods
        class_methods = set()
        missing_methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in valid_classes:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_methods.add(item.name)
        
        for method in REQUIRED_METHODS:
            if method not in class_methods:
                missing_methods.append(method)
        
        if missing_methods:
            return False, f"Missing required methods: {', '.join(missing_methods)}", valid_classes[0], missing_methods
            
        return True, f"Valid ModelTest subclass: {valid_classes[0]}", valid_classes[0], []
        
    except Exception as e:
        return False, f"Error processing file: {str(e)}", None, []

def classify_model_type(file_path):
    """Determine the model type based on file name or content."""
    file_name = os.path.basename(file_path).lower()
    
    # Extract content for more accurate classification if needed
    try:
        with open(file_path, "r") as f:
            content = f.read()
    except:
        content = ""
    
    # Classify by file name
    if any(name in file_name for name in ['bert', 'gpt', 't5', 'gemma', 'llama', 'llm', 'text']):
        return 'text'
    elif any(name in file_name for name in ['vit', 'vision', 'image', 'clip', 'resnet', 'convnext']):
        return 'vision'
    elif any(name in file_name for name in ['whisper', 'audio', 'wav2vec', 'speech', 'hubert']):
        return 'audio'
    elif any(name in file_name for name in ['vqa', 'multimodal', 'blip', 'videoqa']):
        return 'multimodal'
    elif any(name in file_name for name in ['api', 'claude', 'openai', 'groq']):
        return 'api'
    
    # If not identified by name, check content
    if content:
        if 'image' in content and ('classification' in content or 'vision' in content):
            return 'vision'
        elif 'text' in content and ('generation' in content or 'tokenizer' in content):
            return 'text'
        elif 'audio' in content or 'speech' in content:
            return 'audio'
    
    # Default category
    return 'other'

def find_test_files(root_dir):
    """Find all Python test files in the directory tree."""
    test_files = []
    for path in Path(root_dir).rglob("*.py"):
        # Include files that start with test_ or end with _test.py
        if (path.name.startswith("test_") or path.name.endswith("_test.py")) and path.is_file():
            test_files.append(str(path))
    return test_files

def validate_directory(directory, verbose=False):
    """Validate all test files in the directory."""
    test_files = find_test_files(directory)
    
    results = {
        "valid": [], 
        "invalid": [], 
        "total": len(test_files),
        "by_type": {
            "text": {"valid": 0, "invalid": 0},
            "vision": {"valid": 0, "invalid": 0},
            "audio": {"valid": 0, "invalid": 0},
            "multimodal": {"valid": 0, "invalid": 0},
            "api": {"valid": 0, "invalid": 0},
            "other": {"valid": 0, "invalid": 0}
        },
        "missing_methods": Counter()
    }
    
    for file_path in test_files:
        valid, message, class_name, missing = is_modeltest_subclass(file_path)
        relative_path = os.path.relpath(file_path, directory)
        model_type = classify_model_type(file_path)
        
        if valid:
            results["valid"].append({
                "path": relative_path, 
                "class": class_name, 
                "message": message,
                "type": model_type
            })
            results["by_type"][model_type]["valid"] += 1
        else:
            results["invalid"].append({
                "path": relative_path, 
                "message": message,
                "type": model_type,
                "missing": missing
            })
            results["by_type"][model_type]["invalid"] += 1
            
            # Track missing methods
            for method in missing:
                results["missing_methods"][method] += 1
    
    return results

def print_results(results, verbose=False):
    """Print validation results in a readable format."""
    print(f"\nValidation Results for {results['total']} test files:")
    print(f"Valid: {len(results['valid'])} files")
    print(f"Invalid: {len(results['invalid'])} files")
    
    # Calculate compliance percentage
    compliance = (len(results["valid"]) / results["total"]) * 100 if results["total"] > 0 else 0
    print(f"Overall Compliance: {compliance:.1f}% ({len(results['valid'])}/{results['total']} files)")
    
    # Print type breakdown
    print("\n--- Compliance by Model Type ---")
    for model_type, counts in results["by_type"].items():
        total = counts["valid"] + counts["invalid"]
        if total > 0:
            type_compliance = (counts["valid"] / total) * 100
            print(f"{model_type.title()}: {type_compliance:.1f}% ({counts['valid']}/{total})")
    
    # Print missing methods statistics
    print("\n--- Missing Methods ---")
    for method, count in results["missing_methods"].most_common():
        print(f"{method}: {count} files")
    
    if verbose:
        print("\n--- Valid Files ---")
        for file_info in results["valid"]:
            print(f"✅ {file_info['path']} - {file_info['class']} ({file_info['type']})")
        
        print("\n--- Invalid Files ---")
        for file_info in results["invalid"]:
            missing_str = f" Missing: {', '.join(file_info['missing'])}" if file_info['missing'] else ""
            print(f"❌ {file_info['path']} ({file_info['type']}) - {file_info['message']}{missing_str}")

def generate_report(results, output_file=None):
    """Generate a Markdown report from validation results."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    compliance = (len(results["valid"]) / results["total"]) * 100 if results["total"] > 0 else 0
    
    report = f"# ModelTest Compliance Report\n\n"
    report += f"Generated: {timestamp}\n\n"
    report += f"## Summary\n\n"
    report += f"- Total test files: {results['total']}\n"
    report += f"- Compliant files: {len(results['valid'])}\n"
    report += f"- Non-compliant files: {len(results['invalid'])}\n"
    report += f"- **Compliance rate: {compliance:.1f}%**\n\n"
    
    report += f"## Compliance by Model Type\n\n"
    for model_type, counts in results["by_type"].items():
        total = counts["valid"] + counts["invalid"]
        if total > 0:
            type_compliance = (counts["valid"] / total) * 100
            report += f"- {model_type.title()}: {type_compliance:.1f}% ({counts['valid']}/{total})\n"
    
    report += f"\n## Missing Methods\n\n"
    for method, count in results["missing_methods"].most_common():
        report += f"- {method}: {count} files\n"
    
    report += f"\n## Compliant Files\n\n"
    for file_info in results["valid"]:
        report += f"- ✅ `{file_info['path']}` - `{file_info['class']}` ({file_info['type']})\n"
    
    report += f"\n## Non-Compliant Files\n\n"
    
    # Group non-compliant files by type
    non_compliant_by_type = {}
    for file_info in results["invalid"]:
        model_type = file_info['type']
        if model_type not in non_compliant_by_type:
            non_compliant_by_type[model_type] = []
        non_compliant_by_type[model_type].append(file_info)
    
    # List non-compliant files by type
    for model_type, files in non_compliant_by_type.items():
        report += f"### {model_type.title()} Models\n\n"
        for file_info in files:
            missing_str = f" Missing: {', '.join(file_info['missing'])}" if file_info['missing'] else ""
            report += f"- ❌ `{file_info['path']}` - {file_info['message']}{missing_str}\n"
        report += "\n"
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_file}")
    
    return report

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate test files for ModelTest compliance")
    parser.add_argument("--path", type=str, help="Directory or file to validate")
    parser.add_argument("--report", type=str, help="Path to save the Markdown report")
    parser.add_argument("--json", type=str, help="Path to save the results as JSON")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    # Determine the path to validate
    if args.path:
        path = args.path
        
        # Handle both file and directory paths
        if os.path.isfile(path):
            # Process a single file
            logger.info(f"Validating test file: {path}")
            valid, message, class_name, missing = is_modeltest_subclass(path)
            file_name = os.path.basename(path)
            
            if valid:
                print(f"✅ {file_name} - {class_name}: {message}")
                return 0
            else:
                missing_str = f" Missing: {', '.join(missing)}" if missing else ""
                print(f"❌ {file_name} - {message}{missing_str}")
                return 1
        elif os.path.isdir(path):
            # Process a directory
            directory = path
        else:
            logger.error(f"Path not found: {path}")
            sys.exit(1)
    else:
        directory = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)
    
    logger.info(f"Validating test files in {directory}")
    results = validate_directory(directory, args.verbose)
    print_results(results, args.verbose)
    
    if args.report:
        generate_report(results, args.report)
    
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"JSON results saved to {args.json}")
    
    # Return non-zero exit code if any invalid files
    return 1 if results["invalid"] else 0

if __name__ == "__main__":
    sys.exit(main())