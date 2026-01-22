#!/usr/bin/env python3
"""
Updates import statements in Python files to use the new package structure.
This script replaces old import paths with new package-based imports.
"""

import os
import re
import sys
from pathlib import Path

# Mapping of old imports to new imports
IMPORT_MAPPINGS = {
    # Direct imports to package imports
    "from benchmark_db_api import": "from duckdb_api.core.benchmark_db_api import",
    "import benchmark_db_api": "import duckdb_api.core.benchmark_db_api as benchmark_db_api",
    "from benchmark_db_query import": "from duckdb_api.core.benchmark_db_query import",
    "import benchmark_db_query": "import duckdb_api.core.benchmark_db_query as benchmark_db_query",
    "from merged_test_generator import": "from generators.test_generators.merged_test_generator import",
    "import merged_test_generator": "import generators.test_generators.merged_test_generator as merged_test_generator",
    "from fixed_merged_test_generator_clean import": "from generators.test_generators.fixed_merged_test_generator_clean import",
    "import fixed_merged_test_generator_clean": "import generators.test_generators.fixed_merged_test_generator_clean as fixed_merged_test_generator_clean",
    "from simple_test_generator import": "from generators.test_generators.simple_test_generator import",
    "import simple_test_generator": "import generators.test_generators.simple_test_generator as simple_test_generator",
    "from resource_pool import": "from generators.utils.resource_pool import",
    "import resource_pool": "import generators.utils.resource_pool as resource_pool",
    "from test_generator_with_resource_pool import": "from generators.utils.test_generator_with_resource_pool import",
    "import test_generator_with_resource_pool": "import generators.utils.test_generator_with_resource_pool as test_generator_with_resource_pool",
    "from hardware_detection import": "from generators.hardware.hardware_detection import",
    "import hardware_detection": "import generators.hardware.hardware_detection as hardware_detection",
    "from template_database import": "from generators.templates.template_database import",
    "import template_database": "import generators.templates.template_database as template_database",
}

def update_imports_in_file(file_path: str) -> int:
    """
    Updates import statements in a Python file.
    
    Args:
        file_path: Path to the Python file to update
        
    Returns:
        Number of replacements made
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for old_import, new_import in IMPORT_MAPPINGS.items():
            content = content.replace(old_import, new_import)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return sum(content.count(new_import) for old_import, new_import in IMPORT_MAPPINGS.items())
        
        return 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def process_directory(directory: str) -> int:
    """
    Processes all Python files in a directory recursively.
    
    Args:
        directory: Directory to process
        
    Returns:
        Total number of replacements made
    """
    total_replacements = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                replacements = update_imports_in_file(file_path)
                
                if replacements > 0:
                    print(f"Updated {replacements} imports in {file_path}")
                    total_replacements += replacements
    
    return total_replacements

def main():
    """Main function."""
    project_root = Path(__file__).parent.parent  # Go up one directory from test
    
    directories = [
        project_root / "generators",
        project_root / "duckdb_api",
    ]
    
    total_replacements = 0
    
    for directory in directories:
        if not directory.exists():
            print(f"Warning: Directory {directory} does not exist.")
            continue
        
        print(f"Processing directory: {directory}")
        replacements = process_directory(directory)
        total_replacements += replacements
    
    print(f"\nTotal import statements updated: {total_replacements}")

if __name__ == "__main__":
    main()