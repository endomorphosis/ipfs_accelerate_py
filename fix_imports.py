#!/usr/bin/env python
"""
Script to fix import statements in Python files after directory reorganization.
This script updates import statements to reflect the new module structure.
"""

import os
import re
from pathlib import Path

# Mapping of old import paths to new ones
IMPORT_MAPPING = {
    # Generators
    r"(?:from|import)\s+(test\.test_ipfs_accelerate)": r"\1 -> generators.models.test_ipfs_accelerate",
    r"(?:from|import)\s+(test\.integration_test_suite)": r"\1 -> generators.test_runners.integration_test_suite",
    
    # DuckDB API
    r"(?:from|import)\s+(test\.scripts\.benchmark_regression_detector)": r"\1 -> duckdb_api.analysis.benchmark_regression_detector",
    r"(?:from|import)\s+(test\.duckdb_api\.core\.run_benchmark_with_db)": r"\1 -> duckdb_api.core.run_benchmark_with_db",
    r"(?:from|import)\s+(test\.scripts\.ci_benchmark_integrator)": r"\1 -> duckdb_api.scripts.ci_benchmark_integrator",
    r"(?:from|import)\s+(test\.scripts\.benchmark_db\.create_benchmark_schema)": r"\1 -> duckdb_api.scripts.create_benchmark_schema",
    r"(?:from|import)\s+(test\.test_and_benchmark)": r"\1 -> generators.test_generators.test_and_benchmark",
    r"(?:from|import)\s+(test\.scripts\.benchmark_db_query)": r"\1 -> duckdb_api.core.benchmark_db_query",
    
    # Fixed Web Platform
    r"(?:from|import)\s+(test\.archive\.web_platform_test_runner)": r"\1 -> fixed_web_platform.web_platform_test_runner",
    
    # Predictive Performance
    r"(?:from|import)\s+(test\.archive\.hardware_model_predictor)": r"\1 -> predictive_performance.hardware_model_predictor",
    r"(?:from|import)\s+(test\.archive\.model_performance_predictor)": r"\1 -> predictive_performance.model_performance_predictor",
}

def fix_imports_in_file(file_path):
    """
    Fix import statements in a single file.
    
    Args:
        file_path: Path to the Python file to update
    
    Returns:
        int: Number of imports fixed
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    original_content = content
    num_fixes = 0
    
    # Check and fix each import pattern
    for old_pattern, new_info in IMPORT_MAPPING.items():
        old_import, new_import = new_info.split(' -> ')
        
        # Use regex to find and replace import statements
        pattern = old_pattern
        replacement = lambda m: m.group(0).replace(old_import, new_import)
        
        # Apply the replacement
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            num_fixes += len(re.findall(pattern, content))
            content = new_content
    
    # Write back the modified file if any changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return num_fixes

def fix_imports_in_directory(directory_path):
    """
    Fix import statements in all Python files in a directory and its subdirectories.
    
    Args:
        directory_path: Path to the directory to process
    
    Returns:
        tuple: (files_processed, files_modified, imports_fixed)
    """
    files_processed = 0
    files_modified = 0
    total_imports_fixed = 0
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                files_processed += 1
                
                imports_fixed = fix_imports_in_file(file_path)
                if imports_fixed > 0:
                    files_modified += 1
                    total_imports_fixed += imports_fixed
                    print(f"Fixed {imports_fixed} imports in {file_path}")
    
    return files_processed, files_modified, total_imports_fixed

def main():
    """Run the import fixer on the reorganized directories."""
    project_root = Path.cwd().parent  # Go up one level from current directory
    
    # Directories to process
    directories = [
        project_root / "duckdb_api",
        project_root / "generators",
        project_root / "fixed_web_platform",
        project_root / "predictive_performance"
    ]
    
    total_files_processed = 0
    total_files_modified = 0
    total_imports_fixed = 0
    
    for directory in directories:
        if not directory.exists():
            print(f"Directory not found: {directory}")
            continue
        
        print(f"Processing {directory}...")
        files_processed, files_modified, imports_fixed = fix_imports_in_directory(directory)
        
        total_files_processed += files_processed
        total_files_modified += files_modified
        total_imports_fixed += imports_fixed
    
    # Print summary
    print("\n" + "="*50)
    print("Import Fix Summary")
    print("-"*50)
    print(f"Processed {total_files_processed} Python files")
    print(f"Modified {total_files_modified} files")
    print(f"Fixed {total_imports_fixed} import statements")

if __name__ == "__main__":
    main()