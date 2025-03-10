#!/usr/bin/env python3
"""
Template Validator Integration Module

This module provides a unified interface for validating templates across different generators.
It supports validating syntax, imports, class structure, hardware compatibility, and more.
"""

import os
import re
import ast
import logging
import importlib.util
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TemplateValidator:
    """Validator for template-based code generation."""
    
    def __init__(self, generator_type: str = "generic"):
        """Initialize the validator for a specific generator type."""
        self.generator_type = generator_type
        self.errors = []
        
    def validate_syntax(self, content: str) -> bool:
        """
        Validate Python syntax in template content.
        
        Args:
            content: Template string to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            ast.parse(content)
            return True
        except SyntaxError as e:
            self.errors.append(f"Syntax error at line {e.lineno}, column {e.offset}: {e.msg}")
            return False
            
    def validate_hardware_compatibility(self, content: str) -> bool:
        """
        Validate hardware compatibility in template content.
        
        Args:
            content: Template string to validate
            
        Returns:
            True if hardware compatibility is valid, False otherwise
        """
        # Key patterns to look for in a fully hardware-compatible template
        patterns = [
            r"self\.has_cuda",
            r"self\.has_mps",
            r"self\.has_rocm",
            r"self\.has_openvino",
            r"self\.has_qualcomm",
            r"self\.has_webnn",
            r"self\.has_webgpu",
        ]
        
        # Count how many patterns are found
        found_patterns = 0
        for pattern in patterns:
            if re.search(pattern, content):
                found_patterns += 1
                
        # We should find at least 4 patterns in a hardware-compatible template
        hardware_compatible = found_patterns >= 4
        
        if not hardware_compatible:
            self.errors.append(f"Hardware compatibility is incomplete. Found {found_patterns}/7 hardware platforms.")
            
        return hardware_compatible
        
    def validate_resource_pool(self, content: str) -> bool:
        """
        Validate resource pool compatibility in template content.
        
        Args:
            content: Template string to validate
            
        Returns:
            True if resource pool compatibility is valid, False otherwise
        """
        # Check for resource pool patterns
        has_resource_pool = "resource_pool" in content.lower()
        
        return has_resource_pool
        
    def validate_classes(self, content: str) -> bool:
        """
        Validate class structure in template content.
        
        Args:
            content: Template string to validate
            
        Returns:
            True if class structure is valid, False otherwise
        """
        try:
            # Parse the content
            tree = ast.parse(content)
            
            # Look for class definitions
            class_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            # Check if we have at least one class
            if not class_nodes:
                self.errors.append("No class definitions found in template")
                return False
                
            # Check class methods
            for class_node in class_nodes:
                # Look for methods
                methods = [node for node in class_node.body if isinstance(node, ast.FunctionDef)]
                
                # Check if we have basic methods (init, run, get_model, etc.)
                method_names = [method.name for method in methods]
                
                # Basic methods that should be present in a test class
                required_methods = ["__init__", "run"]
                
                for method in required_methods:
                    if method not in method_names:
                        self.errors.append(f"Required method '{method}' not found in class '{class_node.name}'")
                        return False
            
            return True
        except SyntaxError:
            # Syntax validation is handled separately
            return False
            
    def validate_imports(self, content: str) -> bool:
        """
        Validate import statements in template content.
        
        Args:
            content: Template string to validate
            
        Returns:
            True if imports are valid, False otherwise
        """
        try:
            # Parse the content
            tree = ast.parse(content)
            
            # Look for import statements
            import_nodes = [node for node in ast.walk(tree) 
                           if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)]
            
            # Check if we have at least basic imports
            if not import_nodes:
                self.errors.append("No import statements found in template")
                return False
                
            # Basic imports that should be present
            required_imports = ["torch", "logging", "os"]
            found_imports = []
            
            for node in import_nodes:
                if isinstance(node, ast.Import):
                    for name in node.names:
                        found_imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    found_imports.append(node.module)
            
            # Check for required imports
            for imp in required_imports:
                if not any(imp in found for found in found_imports):
                    self.errors.append(f"Required import '{imp}' not found in template")
                    return False
            
            return True
        except SyntaxError:
            # Syntax validation is handled separately
            return False
            
    def validate_indentation(self, content: str, strict: bool = False) -> bool:
        """
        Validate consistent indentation in template content.
        
        Args:
            content: Template string to validate
            strict: If True, enforce strict indentation rules
            
        Returns:
            True if indentation is consistent, False otherwise
        """
        lines = content.split('\n')
        indentation_sizes = []
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue
                
            # Calculate indentation size (number of spaces at beginning)
            indentation = len(line) - len(line.lstrip())
            
            # Only track non-zero indentation levels
            if indentation > 0:
                indentation_sizes.append((i+1, indentation))
        
        # Check for consistent indentation step (4 spaces or 2 spaces)
        if indentation_sizes:
            # Count occurrences of each indentation size
            from collections import Counter
            indentation_counts = Counter([size for _, size in indentation_sizes])
            
            # DEBUG: Print indentation counts
            unique_indentations = sorted(indentation_counts.items())
            indentation_debug = ', '.join([f"{size} spaces ({count} lines)" for size, count in unique_indentations])
            
            # Check for 4-space indentation
            if all(size % 4 == 0 for size in indentation_counts):
                return True
                
            # Check for 2-space indentation
            if all(size % 2 == 0 for size in indentation_counts):
                return True
                
            # For templates, we'll allow some inconsistency in multiline strings
            # (they often have mixed indentation because of the way they're embedded)
            if not strict and "template" in self.generator_type.lower():
                # Just warn about it but don't fail validation
                self.errors.append(f"Warning: Mixed indentation found in template: {indentation_debug}")
                return True
                
            # If neither consistent, report error with details
            inconsistent_lines = []
            for line_num, size in indentation_sizes:
                if size % 4 != 0 and size % 2 != 0:
                    inconsistent_lines.append(f"Line {line_num} has {size} spaces")
                    if len(inconsistent_lines) >= 5:
                        inconsistent_lines.append("... and more")
                        break
                        
            inconsistent_examples = "; ".join(inconsistent_lines)
            self.errors.append(f"Inconsistent indentation found in template: {indentation_debug}. Examples: {inconsistent_examples}")
            return False
        
        return True
            
    def validate_all(self, content: str, 
                    validate_hardware: bool = True, 
                    check_resource_pool: bool = False,
                    strict_indentation: bool = False) -> Tuple[bool, List[str]]:
        """
        Run all validation checks on template content.
        
        Args:
            content: Template string to validate
            validate_hardware: Whether to validate hardware compatibility
            check_resource_pool: Whether to check for resource pool compatibility
            strict_indentation: Whether to enforce strict indentation rules
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        self.errors = []  # Reset errors
        
        # Run all validations
        syntax_valid = self.validate_syntax(content)
        
        # Only continue validation if syntax is valid
        if syntax_valid:
            imports_valid = self.validate_imports(content)
            classes_valid = self.validate_classes(content)
            indentation_valid = self.validate_indentation(content, strict=strict_indentation)
            
            # Optional validations
            hardware_valid = True
            if validate_hardware:
                hardware_valid = self.validate_hardware_compatibility(content)
                
            resource_pool_valid = True
            if check_resource_pool:
                resource_pool_valid = self.validate_resource_pool(content)
                
            # Overall validity
            is_valid = (syntax_valid and imports_valid and classes_valid and 
                        indentation_valid and hardware_valid and
                        (not check_resource_pool or resource_pool_valid))
        else:
            is_valid = False
        
        return is_valid, self.errors
        
    def validate_file(self, file_path: str, 
                     validate_hardware: bool = True,
                     check_resource_pool: bool = False,
                     strict_indentation: bool = False) -> Tuple[bool, List[str]]:
        """
        Validate a template file.
        
        Args:
            file_path: Path to template file
            validate_hardware: Whether to validate hardware compatibility
            check_resource_pool: Whether to check for resource pool compatibility
            strict_indentation: Whether to enforce strict indentation rules
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            return self.validate_all(content, validate_hardware, check_resource_pool, strict_indentation)
        except Exception as e:
            return False, [f"Error reading or validating file: {str(e)}"]


def validate_template_for_generator(template_content: str, 
                                  generator_type: str = "generic",
                                  **kwargs) -> Tuple[bool, List[str]]:
    """
    Validate a template string for a specific generator.
    
    Args:
        template_content: Template string to validate
        generator_type: Type of generator (merged_test_generator, etc.)
        **kwargs: Additional validator options
            - validate_hardware: Whether to validate hardware compatibility (default: True)
            - check_resource_pool: Whether to check for resource pool compatibility (default: False)
            - strict_indentation: Whether to enforce strict indentation rules (default: False)
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    validator = TemplateValidator(generator_type)
    return validator.validate_all(template_content, **kwargs)
    
    
def validate_template_file_for_generator(file_path: str,
                                       generator_type: str = "generic",
                                       **kwargs) -> Tuple[bool, List[str]]:
    """
    Validate a template file for a specific generator.
    
    Args:
        file_path: Path to template file
        generator_type: Type of generator (merged_test_generator, etc.)
        **kwargs: Additional validator options
            - validate_hardware: Whether to validate hardware compatibility (default: True)
            - check_resource_pool: Whether to check for resource pool compatibility (default: False)
            - strict_indentation: Whether to enforce strict indentation rules (default: False)
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    validator = TemplateValidator(generator_type)
    return validator.validate_file(file_path, **kwargs)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Template Validator")
    parser.add_argument("--file", type=str, help="Template file to validate")
    parser.add_argument("--generator", type=str, default="generic", help="Generator type")
    parser.add_argument("--hardware", action="store_true", help="Validate hardware compatibility")
    parser.add_argument("--resource-pool", action="store_true", help="Check resource pool compatibility")
    parser.add_argument("--strict-indentation", action="store_true", help="Enforce strict indentation rules")
    parser.add_argument("--content", type=str, help="Template content to validate directly")
    
    args = parser.parse_args()
    
    if args.file:
        is_valid, errors = validate_template_file_for_generator(
            args.file, 
            args.generator,
            validate_hardware=args.hardware,
            check_resource_pool=args.resource_pool,
            strict_indentation=args.strict_indentation
        )
        
        if is_valid:
            print(f"✅ Template is valid for {args.generator} generator.")
        else:
            print(f"❌ Template has errors for {args.generator} generator:")
            for error in errors:
                print(f"  - {error}")
    elif args.content:
        is_valid, errors = validate_template_for_generator(
            args.content,
            args.generator,
            validate_hardware=args.hardware,
            check_resource_pool=args.resource_pool,
            strict_indentation=args.strict_indentation
        )
        
        if is_valid:
            print(f"✅ Template content is valid for {args.generator} generator.")
        else:
            print(f"❌ Template content has errors for {args.generator} generator:")
            for error in errors:
                print(f"  - {error}")
    else:
        parser.print_help()