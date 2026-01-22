#!/usr/bin/env python3
"""
Test Migration Tool

This script migrates tests from the existing structure to the refactored test suite.
It reads the migration plan and transforms tests to use the new base classes.
"""

import os
import sys
import json
import re
import ast
import argparse
import shutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Base classes and import mappings
BASE_CLASS_MAPPING = {
    'model': 'ModelTest',
    'hardware': 'HardwareTest',
    'browser': 'BrowserTest',
    'api': 'APITest',
    'unit': 'BaseTest',
    'integration': 'BaseTest',
    'resource_pool': 'BaseTest',
}

# Import path mappings
IMPORT_MAPPING = {
    'ModelTest': 'from refactored_test_suite.model_test import ModelTest',
    'HardwareTest': 'from refactored_test_suite.hardware_test import HardwareTest',
    'BrowserTest': 'from refactored_test_suite.browser_test import BrowserTest',
    'APITest': 'from refactored_test_suite.api_test import APITest',
    'BaseTest': 'from refactored_test_suite.base_test import BaseTest',
}

# Target directory for tests
REFACTORED_DIR = 'refactored_test_suite'

# Test categories for classification
TEST_CATEGORIES = {
    'model': ['bert', 'vit', 'gpt', 'llama', 'model', 'transformer'],
    'hardware': ['hardware', 'webgpu', 'webnn', 'device', 'platform'],
    'browser': ['browser', 'firefox', 'chrome', 'safari', 'edge'],
    'api': ['api', 'endpoint', 'service', 'server', 'client'],
    'resource_pool': ['resource', 'pool', 'allocation', 'bridge'],
    'integration': ['integration', 'connect', 'e2e'],
}

# Model subcategories for further classification
MODEL_SUBCATEGORIES = {
    'text': ['bert', 'gpt', 'llama', 't5', 'roberta', 'text', 'language'],
    'vision': ['vit', 'clip', 'image', 'vision', 'yolo', 'segmentation'],
    'audio': ['audio', 'wav', 'whisper', 'speech', 'voice'],
}


class TestFileMigrator:
    """Class for migrating an individual test file."""
    
    def __init__(self, source_path: str, migration_plan: Dict[str, Any]):
        """Initialize the migrator with a source file path."""
        self.source_path = source_path
        self.migration_plan = migration_plan
        self.source_filename = os.path.basename(source_path)
        self.target_path = self._determine_target_path()
        self.category = self._determine_category()
        self.base_class = BASE_CLASS_MAPPING.get(self.category, 'BaseTest')
    
    def _determine_category(self) -> str:
        """Determine the category of the test file based on its name and path."""
        filename = self.source_filename.lower()
        source_path = self.source_path.lower()
        
        # Check path first for more specific categorization
        for category, keywords in TEST_CATEGORIES.items():
            for keyword in keywords:
                if keyword in source_path:
                    return category
        
        # Fall back to checking filename
        for category, keywords in TEST_CATEGORIES.items():
            for keyword in keywords:
                if keyword in filename:
                    return category
        
        # Default to unit test if no category matches
        return 'unit'
    
    def _determine_target_path(self) -> str:
        """Determine the target path based on the migration plan."""
        relative_path = os.path.relpath(self.source_path)
        
        # Check if the file has a specific mapping in the migration plan
        if relative_path in self.migration_plan.get('migration_targets', {}):
            return os.path.join(REFACTORED_DIR, self.migration_plan['migration_targets'][relative_path])
        
        # Determine category from filename
        filename = self.source_filename.lower()
        
        # Determine main category
        category = None
        for cat, keywords in TEST_CATEGORIES.items():
            for keyword in keywords:
                if keyword in filename:
                    category = cat
                    break
            if category:
                break
        
        if not category:
            category = 'unit'  # Default category
        
        # For model tests, determine subcategory
        if category == 'model':
            subcategory = None
            for subcat, keywords in MODEL_SUBCATEGORIES.items():
                for keyword in keywords:
                    if keyword in filename:
                        subcategory = subcat
                        break
                if subcategory:
                    break
            
            if not subcategory:
                subcategory = 'other'  # Default subcategory
            
            return os.path.join(REFACTORED_DIR, 'models', subcategory, self.source_filename)
        
        # For hardware tests, determine subcategory
        elif category == 'hardware':
            if 'webgpu' in filename:
                return os.path.join(REFACTORED_DIR, 'hardware', 'webgpu', self.source_filename)
            elif 'webnn' in filename:
                return os.path.join(REFACTORED_DIR, 'hardware', 'webnn', self.source_filename)
            else:
                return os.path.join(REFACTORED_DIR, 'hardware', 'platform', self.source_filename)
        
        # Other categories go directly to their folder
        return os.path.join(REFACTORED_DIR, category, self.source_filename)
    
    def migrate(self) -> Tuple[bool, str]:
        """
        Migrate the test file to the new structure.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Read source file
            with open(self.source_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse AST
            has_syntax_error = False
            try:
                tree = ast.parse(source_code)
            except SyntaxError as e:
                # Instead of failing, just fall back to basic file copy for files with syntax errors
                has_syntax_error = True
                tree = None
                print(f"Warning: Syntax error in {self.source_path}: {str(e)} - Will copy file without transformation")
            
            # Transform the code or just copy if there's a syntax error
            if has_syntax_error:
                transformed_code = source_code
                
                # Add a comment at the top
                migration_note = f"# WARNING: This file had syntax errors and was copied without transformation\n# Migrated on {datetime.now().strftime('%Y-%m-%d')}\n\n"
                transformed_code = migration_note + transformed_code
            else:
                transformed_code = self._transform_code(source_code, tree)
            
            # Ensure target directory exists
            os.makedirs(os.path.dirname(self.target_path), exist_ok=True)
            
            # Write to target file
            with open(self.target_path, 'w', encoding='utf-8') as f:
                f.write(transformed_code)
            
            if has_syntax_error:
                return True, f"Copied (with syntax errors) {self.source_path} to {self.target_path}"
            else:
                return True, f"Migrated {self.source_path} to {self.target_path}"
        
        except Exception as e:
            return False, f"Error migrating {self.source_path}: {str(e)}"
    
    def _transform_code(self, source_code: str, tree: ast.AST) -> str:
        """Transform the source code to use the new structure."""
        # Find test classes
        test_classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it's a test class (either name starts with "Test" or inherits from unittest.TestCase)
                if node.name.startswith('Test') or any(base.id == 'TestCase' for base in node.bases if isinstance(base, ast.Name)):
                    test_classes.append(node)
        
        if not test_classes:
            # No test classes found, just copy the file
            return source_code
        
        # Add import for base class
        import_statement = IMPORT_MAPPING[self.base_class]
        
        # Simple string replacement for base class
        # This is a basic approach - for production, you'd want to use a more robust method
        modified_code = source_code
        
        # Replace unittest.TestCase base class with the new base class
        test_case_patterns = [
            r'class\s+(\w+)\(unittest\.TestCase\)',
            r'class\s+(\w+)\(TestCase\)',
        ]
        
        for pattern in test_case_patterns:
            modified_code = re.sub(pattern, f'class \\1({self.base_class})', modified_code)
        
        # Add import at the top of the file if it doesn't already exist
        if import_statement not in modified_code:
            # Find the position after existing imports
            import_pos = 0
            for i, line in enumerate(modified_code.split('\n')):
                if line.startswith('import ') or line.startswith('from '):
                    import_pos = i + 1
            
            # Insert the import
            lines = modified_code.split('\n')
            lines.insert(import_pos, import_statement)
            modified_code = '\n'.join(lines)
        
        # Add migration note as a docstring at the top of the file
        migration_note = f'"""Migrated to refactored test suite on {datetime.now().strftime("%Y-%m-%d")}\n\n"""'
        
        # Add the note after the shebang line if it exists
        if modified_code.startswith('#!'):
            # Find the end of the shebang line
            shebang_end = modified_code.find('\n') + 1
            modified_code = modified_code[:shebang_end] + '\n' + migration_note + '\n' + modified_code[shebang_end:]
        else:
            modified_code = migration_note + '\n' + modified_code
        
        return modified_code


def load_migration_plan(plan_path: str) -> Dict[str, Any]:
    """Load the migration plan from a JSON file."""
    with open(plan_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_test_files(directory: str, pattern: str = '**/test_*.py') -> List[str]:
    """Find all test files in a directory."""
    import glob
    return [f for f in glob.glob(os.path.join(directory, pattern), recursive=True)
            if os.path.isfile(f)]


def migrate_files(file_paths: List[str], migration_plan: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """
    Migrate multiple test files.
    
    Args:
        file_paths: List of file paths to migrate
        migration_plan: Migration plan dictionary
        dry_run: If True, don't actually migrate, just report what would be done
    
    Returns:
        Dictionary with migration results
    """
    results = {
        'success': [],
        'failure': [],
        'skipped': []
    }
    
    for source_path in file_paths:
        # Skip files already in the refactored directory
        if source_path.startswith(REFACTORED_DIR):
            results['skipped'].append((source_path, 'Already in refactored directory'))
            continue
        
        # Create migrator for this file
        migrator = TestFileMigrator(source_path, migration_plan)
        
        if dry_run:
            print(f"Would migrate {source_path} to {migrator.target_path} (category: {migrator.category}, base class: {migrator.base_class})")
            results['success'].append((source_path, migrator.target_path))
        else:
            success, message = migrator.migrate()
            if success:
                results['success'].append((source_path, migrator.target_path))
                print(f"✅ {message}")
            else:
                results['failure'].append((source_path, message))
                print(f"❌ {message}")
    
    return results


def generate_migration_report(results: Dict[str, Any], output_path: str) -> None:
    """Generate a report of the migration results."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Test Migration Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total files processed: {len(results['success']) + len(results['failure']) + len(results['skipped'])}\n")
        f.write(f"- Successfully migrated: {len(results['success'])}\n")
        f.write(f"- Failed to migrate: {len(results['failure'])}\n")
        f.write(f"- Skipped: {len(results['skipped'])}\n\n")
        
        if results['success']:
            f.write("## Successful Migrations\n\n")
            for source, target in results['success']:
                f.write(f"- {source} → {target}\n")
            f.write("\n")
        
        if results['failure']:
            f.write("## Failed Migrations\n\n")
            for source, error in results['failure']:
                f.write(f"- {source}: {error}\n")
            f.write("\n")
        
        if results['skipped']:
            f.write("## Skipped Files\n\n")
            for source, reason in results['skipped']:
                f.write(f"- {source}: {reason}\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description='Migrate test files to refactored structure')
    parser.add_argument('--plan', type=str, default='refactored_test_suite/migration_plan.json',
                      help='Path to the migration plan JSON file')
    parser.add_argument('--files', type=str, nargs='+',
                      help='Specific test files to migrate')
    parser.add_argument('--dir', type=str, default='.',
                      help='Directory to search for test files')
    parser.add_argument('--pattern', type=str, default='**/test_*.py',
                      help='Glob pattern for finding test files')
    parser.add_argument('--limit', type=int, default=0,
                      help='Limit the number of files to migrate (0 for no limit)')
    parser.add_argument('--dry-run', action='store_true',
                      help='Do not actually migrate, just print what would be done')
    parser.add_argument('--report', type=str, default='refactored_test_suite/migration_report.md',
                      help='Path to save the migration report')
    
    args = parser.parse_args()
    
    # Load migration plan
    print(f"Loading migration plan from {args.plan}")
    migration_plan = load_migration_plan(args.plan)
    
    # Find test files
    if args.files:
        file_paths = args.files
    else:
        print(f"Finding test files in {args.dir} matching pattern {args.pattern}")
        file_paths = find_test_files(args.dir, args.pattern)
    
    # Apply limit if specified
    if args.limit > 0:
        file_paths = file_paths[:args.limit]
    
    print(f"Found {len(file_paths)} test files to process")
    
    # Migrate files
    print("Starting migration...")
    results = migrate_files(file_paths, migration_plan, args.dry_run)
    
    # Generate report
    print(f"Generating migration report to {args.report}")
    generate_migration_report(results, args.report)
    
    # Print summary
    print("\nMigration complete!")
    print(f"Successfully migrated: {len(results['success'])}")
    print(f"Failed to migrate: {len(results['failure'])}")
    print(f"Skipped: {len(results['skipped'])}")
    print(f"See {args.report} for details")


if __name__ == '__main__':
    main()