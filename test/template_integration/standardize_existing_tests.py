#!/usr/bin/env python3
"""
Tool to standardize existing test files to follow the ModelTest base class pattern.

This script analyzes existing test files and refactors them to follow the 
standardized testing pattern with ModelTest base class and required methods.
"""

import os
import sys
import ast
import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Tuple, Union

# Configure logging
log_filename = f"standardize_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

# Import template utilities if available
try:
    from template_integration.validate_test_files import TestFileValidator
except ImportError:
    logger.warning("Could not import TestFileValidator, will use internal implementation")
    TestFileValidator = None

# Required methods for standard ModelTest class
REQUIRED_METHODS = [
    "setUp",
    "test_model_loading"
]

# Templates for required methods
SETUP_TEMPLATE = """
def setUp(self):
    # Set up resources for each test method
    super().setUp()
    self.model_id = "{model_id}"
    
    # Configure hardware preference
    self.preferred_device = self.detect_preferred_device()
"""

MODEL_LOADING_TEMPLATE = """
def test_model_loading(self):
    # Test basic model loading
    if not hasattr(self, 'model_id') or not self.model_id:
        self.skipTest("No model_id specified")
        
    try:
        # Import the appropriate library
        if 'bert' in self.model_id.lower() or 'gpt' in self.model_id.lower() or 't5' in self.model_id.lower():
            import transformers
            model = transformers.AutoModel.from_pretrained(self.model_id)
            self.assertIsNotNone(model, "Model loading failed")
        elif 'clip' in self.model_id.lower():
            import transformers
            model = transformers.CLIPModel.from_pretrained(self.model_id)
            self.assertIsNotNone(model, "Model loading failed")
        elif 'whisper' in self.model_id.lower():
            import transformers
            model = transformers.WhisperModel.from_pretrained(self.model_id)
            self.assertIsNotNone(model, "Model loading failed")
        elif 'wav2vec2' in self.model_id.lower():
            import transformers
            model = transformers.Wav2Vec2Model.from_pretrained(self.model_id)
            self.assertIsNotNone(model, "Model loading failed")
        else:
            # Generic loading
            try:
                import transformers
                model = transformers.AutoModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            except:
                self.skipTest(f"Could not load model {self.model_id} with AutoModel")
    except Exception as e:
        self.fail(f"Model loading failed: {e}")
"""

DEVICE_TEMPLATE = """
def detect_preferred_device(self):
    # Detect available hardware and choose the preferred device
    try:
        import torch
        
        # Check for CUDA
        if torch.cuda.is_available():
            return "cuda"
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
            return "mps"
        
        # Fallback to CPU
        return "cpu"
    except ImportError:
        return "cpu"
"""

class TestFileStandardizer:
    """Standardizes test files to follow the ModelTest pattern."""
    
    def __init__(self, base_class="ModelTest", required_methods=None):
        """Initialize the standardizer."""
        self.base_class = base_class
        self.required_methods = required_methods or REQUIRED_METHODS
        self.validator = TestFileValidator() if TestFileValidator else None
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a test file to understand its structure.
        
        Args:
            file_path: Path to the test file
            
        Returns:
            Dictionary with file analysis results
        """
        result = {
            "file_path": file_path,
            "valid": False,
            "test_classes": [],
            "imports": [],
            "model_ids": [],
            "methods": {},
            "inherits_from_model_test": False,
            "has_required_methods": False,
            "changes_needed": [],
            "valid_syntax": False
        }
        
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            result["error"] = f"Error reading file: {e}"
            return result
        
        # Parse AST to check syntax
        try:
            tree = ast.parse(content)
            result["valid_syntax"] = True
        except SyntaxError as e:
            result["error"] = f"Syntax error on line {e.lineno}: {e.msg}"
            return result
        
        # Find imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    result["imports"].append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                for name in node.names:
                    result["imports"].append(f"{module}.{name.name}")
        
        # Find all class definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name.startswith('Test'):
                    class_info = {
                        "name": node.name,
                        "bases": [base.id if isinstance(base, ast.Name) else 
                                 base.attr if isinstance(base, ast.Attribute) else 
                                 str(base) for base in node.bases],
                        "methods": [],
                        "model_id": None,
                        "device_detection": False
                    }
                    
                    # Check methods
                    for sub_node in ast.walk(node):
                        if isinstance(sub_node, ast.FunctionDef):
                            class_info["methods"].append(sub_node.name)
                            result["methods"][sub_node.name] = True
                            
                            # Look for model_id assignments
                            if sub_node.name == 'setUp':
                                for setup_node in ast.walk(sub_node):
                                    if (isinstance(setup_node, ast.Assign) and 
                                        len(setup_node.targets) == 1 and 
                                        isinstance(setup_node.targets[0], ast.Attribute)):
                                        
                                        attr = setup_node.targets[0]
                                        if (isinstance(attr.value, ast.Name) and 
                                            attr.value.id == 'self' and 
                                            attr.attr == 'model_id'):
                                            
                                            # Found self.model_id assignment
                                            if isinstance(setup_node.value, ast.Constant):
                                                class_info["model_id"] = setup_node.value.value
                                                result["model_ids"].append(setup_node.value.value)
                                            elif isinstance(setup_node.value, ast.Str):  # For Python < 3.8
                                                class_info["model_id"] = setup_node.value.s
                                                result["model_ids"].append(setup_node.value.s)
                    
                    # Check for device detection method
                    if "detect_preferred_device" in class_info["methods"]:
                        class_info["device_detection"] = True
                    
                    result["test_classes"].append(class_info)
        
        # Determine if the file has ModelTest base class
        for class_info in result["test_classes"]:
            if self.base_class in class_info["bases"]:
                result["inherits_from_model_test"] = True
                break
        
        # Check if all required methods are present
        missing_methods = []
        for method in self.required_methods:
            if method not in result["methods"]:
                missing_methods.append(method)
        
        if not missing_methods:
            result["has_required_methods"] = True
        else:
            result["changes_needed"].append(f"Add missing methods: {', '.join(missing_methods)}")
        
        # Determine if the file is valid
        result["valid"] = result["inherits_from_model_test"] and result["has_required_methods"]
        
        # Add necessary changes
        if not result["inherits_from_model_test"]:
            result["changes_needed"].append(f"Change class to inherit from {self.base_class}")
        
        if not result["model_ids"] and not result["valid"]:
            result["changes_needed"].append("Add model_id assignment in setUp method")
        
        return result
    
    def standardize_file(self, file_path: str, output_path: str = None, backup: bool = True) -> Dict[str, Any]:
        """
        Standardize a test file to follow the ModelTest pattern.
        
        Args:
            file_path: Path to the test file
            output_path: Path to write the standardized file (defaults to overwrite original)
            backup: Whether to create a backup of the original file
            
        Returns:
            Dictionary with standardization results
        """
        # Create output path if not specified
        if not output_path:
            output_path = file_path
        
        # Analyze the file
        analysis = self.analyze_file(file_path)
        
        # If file is already valid, return early
        if analysis["valid"]:
            logger.info(f"File {file_path} is already standardized")
            return {
                "file_path": file_path,
                "output_path": output_path,
                "standardized": False,
                "message": "File already follows standards"
            }
        
        # Create a backup if requested
        if backup and output_path == file_path:
            backup_path = f"{file_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {
                "file_path": file_path,
                "output_path": output_path,
                "standardized": False,
                "error": f"Error reading file: {e}"
            }
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {
                "file_path": file_path,
                "output_path": output_path,
                "standardized": False,
                "error": f"Syntax error on line {e.lineno}: {e.msg}"
            }
        
        # Determine needed modifications
        changes = []
        
        # Check if we need to add ModelTest import
        model_test_import_needed = False
        if not analysis["inherits_from_model_test"]:
            model_test_import_needed = True
        
        # Check if we need to add method implementations
        methods_to_add = {}
        for method in self.required_methods:
            if method not in analysis["methods"]:
                methods_to_add[method] = True
        
        # Create modified content
        modified_content = content
        
        # Add ModelTest import if needed
        if model_test_import_needed:
            # Find a good place to add the import - after the last import
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    imports.append(node)
            
            if imports:
                last_import = imports[-1]
                last_import_end = last_import.end_lineno if hasattr(last_import, 'end_lineno') else last_import.lineno
                
                # Split content and add import after the last import
                lines = modified_content.split('\n')
                lines.insert(last_import_end, "\nfrom refactored_test_suite.model_test import ModelTest")
                modified_content = '\n'.join(lines)
                
                changes.append("Added ModelTest import")
            else:
                # No imports found, add at the top
                modified_content = "from refactored_test_suite.model_test import ModelTest\n\n" + modified_content
                changes.append("Added ModelTest import at the top")
        
        # Modify class inheritance
        if not analysis["inherits_from_model_test"]:
            # Find test classes
            class_defs = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                    class_defs.append(node)
            
            # Modify each test class
            for class_def in class_defs:
                # Get class definition line
                class_line = content.split('\n')[class_def.lineno - 1]
                
                # Replace the base class(es)
                new_class_line = class_line
                
                # If there are existing base classes, add ModelTest to them
                if '(' in class_line and ')' in class_line:
                    if class_line.strip().endswith('):'):
                        # Empty base class parentheses
                        new_class_line = class_line.replace('():', f'({self.base_class}):', 1)
                    else:
                        # Existing base classes
                        open_paren = class_line.find('(')
                        close_paren = class_line.rfind(')')
                        
                        existing_bases = class_line[open_paren+1:close_paren].strip()
                        if existing_bases:
                            # Replace existing bases with ModelTest
                            new_class_line = class_line[:open_paren+1] + self.base_class + class_line[close_paren:]
                        else:
                            # Empty parentheses
                            new_class_line = class_line[:open_paren+1] + self.base_class + class_line[close_paren:]
                else:
                    # No base classes, add ModelTest
                    new_class_line = class_line.replace(':', f'({self.base_class}):', 1)
                
                # Replace the class definition line
                modified_content = modified_content.replace(class_line, new_class_line)
                changes.append(f"Modified class {class_def.name} to inherit from {self.base_class}")
        
        # Add missing methods
        if methods_to_add:
            # Find test classes
            class_defs = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                    class_defs.append(node)
            
            # Add methods to the first test class
            if class_defs:
                class_def = class_defs[0]
                
                # Get indentation
                class_line = content.split('\n')[class_def.lineno - 1]
                indent = ' ' * (len(class_line) - len(class_line.lstrip()))
                method_indent = indent + '    '
                
                # Find the end of the class
                class_end_line = len(content.split('\n'))
                for i in range(class_def.lineno, len(content.split('\n'))):
                    line = content.split('\n')[i]
                    # If we find a line that has the same or less indentation than the class
                    # and it's not a blank line, it's outside the class
                    if line.strip() and len(line) - len(line.lstrip()) <= len(indent):
                        class_end_line = i
                        break
                
                # Model ID for the setUp method
                model_id = "None"
                if analysis["model_ids"]:
                    model_id = analysis["model_ids"][0]
                elif "bert" in file_path.lower():
                    model_id = "bert-base-uncased"
                elif "gpt" in file_path.lower():
                    model_id = "gpt2"
                elif "clip" in file_path.lower():
                    model_id = "openai/clip-vit-base-patch32"
                elif "whisper" in file_path.lower():
                    model_id = "openai/whisper-tiny"
                elif "wav2vec" in file_path.lower():
                    model_id = "facebook/wav2vec2-base-960h"
                elif "t5" in file_path.lower():
                    model_id = "t5-base"
                elif "vit" in file_path.lower():
                    model_id = "google/vit-base-patch16-224"
                
                # Methods to add
                methods_content = ""
                if "setUp" in methods_to_add:
                    setup_method = SETUP_TEMPLATE.format(model_id=model_id)
                    methods_content += "\n" + "\n".join([method_indent + line if line.strip() else line 
                                                     for line in setup_method.split('\n')]) + "\n"
                    changes.append("Added setUp method")
                
                if "test_model_loading" in methods_to_add:
                    model_loading_method = MODEL_LOADING_TEMPLATE
                    methods_content += "\n" + "\n".join([method_indent + line if line.strip() else line 
                                                      for line in model_loading_method.split('\n')]) + "\n"
                    changes.append("Added test_model_loading method")
                
                # Check if we need to add detect_preferred_device
                if "detect_preferred_device" not in analysis["methods"]:
                    device_method = DEVICE_TEMPLATE
                    methods_content += "\n" + "\n".join([method_indent + line if line.strip() else line 
                                                     for line in device_method.split('\n')]) + "\n"
                    changes.append("Added detect_preferred_device method")
                
                # Split content and add methods at the end of the class
                lines = modified_content.split('\n')
                lines.insert(class_end_line, methods_content)
                modified_content = '\n'.join(lines)
        
        # Write the modified content
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            logger.info(f"Wrote standardized file to {output_path}")
        except Exception as e:
            return {
                "file_path": file_path,
                "output_path": output_path,
                "standardized": False,
                "error": f"Error writing file: {e}"
            }
        
        # Validate the standardized file
        if self.validator:
            validation = self.validator.validate_file(output_path)
            if not validation["overall_valid"]:
                logger.warning(f"Standardized file still has validation issues: {validation['errors']}")
        
        return {
            "file_path": file_path,
            "output_path": output_path,
            "standardized": True,
            "changes": changes,
            "message": f"Successfully standardized with {len(changes)} changes"
        }
    
    def batch_standardize(
        self, 
        directory: str, 
        output_directory: Optional[str] = None,
        backup: bool = True,
        recursive: bool = True,
        pattern: str = "test_*.py",
        skip_valid: bool = True
    ) -> Dict[str, Any]:
        """
        Standardize all test files in a directory.
        
        Args:
            directory: Directory to scan for test files
            output_directory: Directory to write standardized files (defaults to same as input)
            backup: Whether to create backups of original files
            recursive: Whether to scan subdirectories
            pattern: File name pattern to match
            skip_valid: Whether to skip files that are already valid
            
        Returns:
            Dictionary with standardization results
        """
        results = {
            "directory": directory,
            "output_directory": output_directory or directory,
            "pattern": pattern,
            "timestamp": datetime.now().isoformat(),
            "files": {},
            "summary": {
                "total_files": 0,
                "standardized_files": 0,
                "skipped_files": 0,
                "failed_files": 0
            }
        }
        
        # Get all test files
        if recursive:
            file_paths = list(Path(directory).rglob(pattern))
        else:
            file_paths = list(Path(directory).glob(pattern))
        
        # Sort file paths for consistent output
        file_paths.sort()
        
        # Initialize counters
        total_files = 0
        standardized_files = 0
        skipped_files = 0
        failed_files = 0
        
        # Process each file
        for file_path in file_paths:
            total_files += 1
            
            # Determine output path
            if output_directory:
                rel_path = os.path.relpath(file_path, directory)
                output_path = os.path.join(output_directory, rel_path)
            else:
                output_path = str(file_path)
            
            # Analyze file
            analysis = self.analyze_file(str(file_path))
            
            # Skip valid files if requested
            if analysis["valid"] and skip_valid:
                logger.info(f"Skipping valid file: {file_path}")
                results["files"][str(file_path)] = {
                    "standardized": False,
                    "skipped": True,
                    "valid": True,
                    "message": "File already follows standards"
                }
                skipped_files += 1
                continue
            
            # Standardize file
            try:
                result = self.standardize_file(str(file_path), output_path, backup)
                results["files"][str(file_path)] = result
                
                if result.get("standardized"):
                    standardized_files += 1
                    logger.info(f"Standardized: {file_path}")
                else:
                    skipped_files += 1
                    logger.info(f"Skipped: {file_path} - {result.get('message')}")
                
            except Exception as e:
                logger.error(f"Error standardizing {file_path}: {e}")
                results["files"][str(file_path)] = {
                    "standardized": False,
                    "error": str(e)
                }
                failed_files += 1
        
        # Update summary
        results["summary"]["total_files"] = total_files
        results["summary"]["standardized_files"] = standardized_files
        results["summary"]["skipped_files"] = skipped_files
        results["summary"]["failed_files"] = failed_files
        
        # Calculate percentages
        if total_files > 0:
            results["summary"]["standardized_percentage"] = (standardized_files / total_files) * 100
        else:
            results["summary"]["standardized_percentage"] = 0
        
        # Log summary
        logger.info(f"Standardization complete: {standardized_files}/{total_files} files standardized "
                   f"({results['summary']['standardized_percentage']:.1f}%)")
        
        return results

def generate_report(results: Dict[str, Any], output_file: str) -> None:
    """Generate a human-readable standardization report."""
    with open(output_file, 'w') as f:
        f.write("# Test File Standardization Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write summary
        f.write("## Summary\n\n")
        f.write(f"- Directory: `{results['directory']}`\n")
        f.write(f"- Output Directory: `{results['output_directory']}`\n")
        f.write(f"- Pattern: `{results['pattern']}`\n")
        f.write(f"- Total files: {results['summary']['total_files']}\n")
        f.write(f"- Standardized files: {results['summary']['standardized_files']} ")
        if results['summary']['total_files'] > 0:
            f.write(f"({results['summary']['standardized_percentage']:.1f}%)\n")
        else:
            f.write("(0%)\n")
        f.write(f"- Skipped files: {results['summary']['skipped_files']}\n")
        f.write(f"- Failed files: {results['summary']['failed_files']}\n\n")
        
        # Write standardized files
        if results['summary']['standardized_files'] > 0:
            f.write("## Standardized Files\n\n")
            
            for file_path, file_result in results["files"].items():
                if file_result.get("standardized"):
                    f.write(f"### {os.path.basename(file_path)}\n\n")
                    f.write(f"- Path: `{file_path}`\n")
                    f.write(f"- Output: `{file_result.get('output_path', file_path)}`\n")
                    f.write("\n**Changes:**\n\n")
                    for change in file_result.get("changes", []):
                        f.write(f"- {change}\n")
                    f.write("\n")
            
            f.write("\n")
        
        # Write skipped files
        if results['summary']['skipped_files'] > 0:
            f.write("## Skipped Files\n\n")
            
            for file_path, file_result in results["files"].items():
                if file_result.get("skipped"):
                    f.write(f"- `{file_path}`: {file_result.get('message', 'Skipped')}\n")
            
            f.write("\n")
        
        # Write failed files
        if results['summary']['failed_files'] > 0:
            f.write("## Failed Files\n\n")
            
            for file_path, file_result in results["files"].items():
                if file_result.get("error"):
                    f.write(f"- `{file_path}`: {file_result.get('error', 'Failed')}\n")
            
            f.write("\n")

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Standardize test files to follow ModelTest pattern")
    
    # Target specification
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--file", type=str, help="Standardize a single test file")
    target_group.add_argument("--directory", type=str, help="Standardize all test files in a directory")
    
    # Directory options
    directory_group = parser.add_argument_group("Directory Options")
    directory_group.add_argument("--pattern", type=str, default="test_*.py",
                               help="File name pattern to match (default: test_*.py)")
    directory_group.add_argument("--no-recursive", action="store_true",
                               help="Don't recursively scan subdirectories")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output", type=str, help="Output file or directory")
    output_group.add_argument("--no-backup", action="store_true", help="Don't create backup files")
    output_group.add_argument("--report", type=str, help="Generate human-readable report file")
    output_group.add_argument("--overwrite-valid", action="store_true", help="Overwrite files that are already valid")
    
    # Advanced options
    advanced_group = parser.add_argument_group("Advanced Options")
    advanced_group.add_argument("--base-class", type=str, default="ModelTest",
                              help="Base class for test classes (default: ModelTest)")
    
    # Other options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create standardizer
    standardizer = TestFileStandardizer(base_class=args.base_class)
    
    # Standardize target
    if args.file:
        # Standardize single file
        result = standardizer.standardize_file(
            args.file,
            args.output,
            not args.no_backup
        )
        
        # Print result
        if result["standardized"]:
            print(f"✅ File standardized: {args.file}")
            for change in result.get("changes", []):
                print(f"  ✓ {change}")
        else:
            print(f"⚠️ File not standardized: {args.file}")
            if "error" in result:
                print(f"  ✗ {result['error']}")
            else:
                print(f"  ℹ️ {result.get('message', 'No changes needed')}")
        
        # Generate report if requested
        if args.report:
            results = {
                "directory": os.path.dirname(args.file),
                "output_directory": os.path.dirname(args.output or args.file),
                "pattern": os.path.basename(args.file),
                "timestamp": datetime.now().isoformat(),
                "files": {args.file: result},
                "summary": {
                    "total_files": 1,
                    "standardized_files": 1 if result["standardized"] else 0,
                    "skipped_files": 0 if result["standardized"] else 1,
                    "failed_files": 0,
                    "standardized_percentage": 100 if result["standardized"] else 0
                }
            }
            generate_report(results, args.report)
            print(f"Report written to {args.report}")
        
        # Return appropriate exit code
        return 0 if result["standardized"] or "skipped" in result else 1
    
    elif args.directory:
        # Standardize directory
        results = standardizer.batch_standardize(
            args.directory,
            args.output,
            not args.no_backup,
            not args.no_recursive,
            args.pattern,
            not args.overwrite_valid
        )
        
        # Print summary
        print("\nStandardization Summary:")
        print(f"Total files: {results['summary']['total_files']}")
        print(f"Standardized files: {results['summary']['standardized_files']} ", end="")
        if results['summary']['total_files'] > 0:
            print(f"({results['summary']['standardized_percentage']:.1f}%)")
        else:
            print("(0%)")
        print(f"Skipped files: {results['summary']['skipped_files']}")
        print(f"Failed files: {results['summary']['failed_files']}")
        
        # Generate report if requested
        if args.report:
            generate_report(results, args.report)
            print(f"Report written to {args.report}")
        
        # Return appropriate exit code
        return 0 if results['summary']['failed_files'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())