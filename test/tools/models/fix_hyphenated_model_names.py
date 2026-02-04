#!/usr/bin/env python3
"""
Script to fix hyphenated model names in Python test files.
This script replaces hyphenated model names with underscore versions in variable names.
"""

import os
import re
import sys
import argparse
from typing import List, Dict, Tuple

class HyphenatedModelNameFixer:
    """Fixes hyphenated model names in Python test files."""
    
    def __init__(self, files_to_fix: List[str] = None, test_dir: str = "skills"):
        """Initialize the fixer.
        
        Args:
            files_to_fix: List of specific files to fix
            test_dir: Directory containing test files if no specific files are provided
        """
        self.files_to_fix = files_to_fix or []
        self.test_dir = test_dir
        self.results = {"fixed": [], "skipped": [], "errors": []}
        
    def find_test_files(self) -> List[str]:
        """Find all Python test files in the specified directory.
        
        Returns:
            List of paths to test files
        """
        test_files = []
        for root, _, files in os.walk(self.test_dir):
            for file in files:
                if file.endswith('.py') and file.startswith('test_hf_'):
                    # Look specifically for files that might have hyphenated model names
                    if '-' in file or self._potentially_has_hyphenated_model(os.path.join(root, file)):
                        test_files.append(os.path.join(root, file))
        return test_files
    
    def _potentially_has_hyphenated_model(self, file_path: str) -> bool:
        """Check if file potentially has hyphenated model names.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file potentially has hyphenated model names
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Look for common patterns that indicate hyphenated model names
            patterns = [
                r"[A-Z]+-[A-Z]+_MODELS_REGISTRY",  # MODEL-NAME_MODELS_REGISTRY
                r"[A-Za-z]+-[A-Za-z]+(Model|LMHeadModel|ForImageClassification)",  # Model-nameModel
                r"Test[A-Za-z]+-[A-Za-z]+",  # TestModel-name
                r'"[a-z]+-[a-z]+"',  # "model-name"
            ]
            
            for pattern in patterns:
                if re.search(pattern, content):
                    return True
            
            return False
                    
        except Exception:
            # If we can't read the file, skip it
            return False
    
    def fix_file(self, file_path: str) -> Tuple[bool, str]:
        """Fix hyphenated model names in a single file.
        
        Args:
            file_path: Path to the file to fix
            
        Returns:
            Tuple of (success, message)
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Extract base model name from filename
            base_name = os.path.basename(file_path)
            match = re.search(r'test_hf_([a-zA-Z0-9_-]+)\.py', base_name)
            
            if not match:
                return False, "Could not extract model name from filename"
            
            model_name = match.group(1)
            
            # Skip if no hyphen in model name
            if '-' not in model_name and '-' not in content:
                return False, "No hyphenated names found"
            
            # Replace model name patterns
            original_content = content
            
            # Case 1: Variable names with MODEL-NAME_MODELS_REGISTRY
            if '-' in model_name:
                hyphenated_pattern = model_name.replace('-', '-').upper() + "_MODELS_REGISTRY"
                underscore_pattern = model_name.replace('-', '_').upper() + "_MODELS_REGISTRY"
                content = content.replace(hyphenated_pattern, underscore_pattern)
            
            # Case 2: Class names like TestModel-nameModels
            if '-' in model_name:
                # Convert model-name to ModelName for class names
                model_name_parts = model_name.split('-')
                pascal_model_name = ''.join(part.capitalize() for part in model_name_parts)
                
                hyphenated_class = f"Test{model_name.capitalize()}-"
                fixed_class = f"Test{pascal_model_name}"
                content = re.sub(r'Test[A-Za-z]+-[A-Za-z]+', fixed_class, content)
            
            # Case 3: Model class references like MODEL-NAMELMHeadModel
            if '-' in model_name:
                model_upper = model_name.upper()
                model_upper_parts = model_upper.split('-')
                
                for suffix in ["LMHeadModel", "Model", "ForImageClassification", "ForCausalLM"]:
                    hyphenated_ref = f"{model_upper.replace('-', '-')}{suffix}"
                    fixed_ref = f"{pascal_model_name}{suffix}"
                    content = content.replace(hyphenated_ref, fixed_ref)
            
            # Case 4: Fix filename references with hyphens
            if '-' in model_name:
                hyphenated_filename = f"hf_{model_name.lower()}_"
                underscore_filename = f"hf_{model_name.lower().replace('-', '_')}_"
                content = content.replace(hyphenated_filename, underscore_filename)
            
            # If no changes made, skip
            if content == original_content:
                return False, "No changes needed"
            
            # Write fixed content
            with open(file_path, 'w') as file:
                file.write(content)
            
            return True, "Fixed hyphenated model names"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def fix_all_files(self) -> Dict:
        """Fix hyphenated model names in all files.
        
        Returns:
            Dict with results
        """
        files_to_process = self.files_to_fix or self.find_test_files()
        print(f"Found {len(files_to_process)} files to check")
        
        for file_path in files_to_process:
            success, message = self.fix_file(file_path)
            
            if success:
                self.results["fixed"].append((file_path, message))
            elif "Error" in message:
                self.results["errors"].append((file_path, message))
            else:
                self.results["skipped"].append((file_path, message))
        
        return self.results
    
    def print_report(self) -> None:
        """Print a report of the fix results."""
        fixed_count = len(self.results["fixed"])
        skipped_count = len(self.results["skipped"])
        error_count = len(self.results["errors"])
        total = fixed_count + skipped_count + error_count
        
        print("\n=== HYPHENATED MODEL NAME FIX REPORT ===")
        print(f"Total files processed: {total}")
        print(f"Fixed: {fixed_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Errors: {error_count}")
        
        if fixed_count > 0:
            print("\nFiles fixed:")
            for file_path, message in self.results["fixed"]:
                print(f"  - {os.path.basename(file_path)}: {message}")
        
        if error_count > 0:
            print("\nFiles with errors:")
            for file_path, message in self.results["errors"]:
                print(f"  - {os.path.basename(file_path)}: {message}")
        
        print("\nFix completed.")

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Fix hyphenated model names in Python test files")
    parser.add_argument("--dir", type=str, default="skills",
                        help="Directory containing test files")
    parser.add_argument("--files", nargs='+', help="Specific files to fix")
    args = parser.parse_args()
    
    fixer = HyphenatedModelNameFixer(files_to_fix=args.files, test_dir=args.dir)
    fixer.fix_all_files()
    fixer.print_report()
    
    # Return non-zero exit code if there are errors
    if len(fixer.results["errors"]) > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())