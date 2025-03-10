#!/usr/bin/env python3
"""
This script updates path references in CI/CD workflow files to reflect directory reorganization.
It updates paths from test/ to either generators/ or duckdb_api/ as appropriate.
"""

import re
import os
import glob

# Define mappings for file paths
PATH_MAPPINGS = []]],,,
    # test/scripts -> duckdb_api/scripts
(r'test/scripts/', r'duckdb_api/scripts/'),
(r'duckdb_api/scripts/benchmark_db_query\.py', r'duckdb_api/core/benchmark_db_query.py'),
(r'duckdb_api/scripts/benchmark_regression_detector\.py', r'duckdb_api/analysis/benchmark_regression_detector.py'),
    
    # test/ Python files -> duckdb_api/core/
(r'test/run_benchmark_with_db\.py', r'duckdb_api/core/run_benchmark_with_db.py'),
(r'test/benchmark_db_query\.py', r'duckdb_api/core/benchmark_db_query.py'),
(r'test/benchmark_regression_detector\.py', r'duckdb_api/analysis/benchmark_regression_detector.py'),
(r'test/hardware_model_predictor\.py', r'predictive_performance/hardware_model_predictor.py'),
(r'test/model_performance_predictor\.py', r'predictive_performance/model_performance_predictor.py'),
(r'test/create_benchmark_schema\.py', r'duckdb_api/schema/create_benchmark_schema.py'),
(r'test/ci_benchmark_integrator\.py', r'duckdb_api/scripts/ci_benchmark_integrator.py'),
    
    # Generators
(r'test/test_ipfs_accelerate\.py', r'generators/models/test_ipfs_accelerate.py'),
(r'test/generate_compatibility_matrix\.py', r'duckdb_api/visualization/generate_compatibility_matrix.py'),
(r'test/generate_enhanced_compatibility_matrix\.py', r'duckdb_api/visualization/generate_enhanced_compatibility_matrix.py'),
(r'test/integration_test_suite\.py', r'generators/test_runners/integration_test_suite.py'),
(r'test/web_platform_test_runner\.py', r'fixed_web_platform/web_platform_test_runner.py'),
    
    # Requirement files
(r'test/requirements_api\.txt', r'requirements_api.txt'),
(r'test/requirements\.txt', r'requirements.txt'),
    
    # Simple path removals
(r'cd test\n', r''),
(r'cd test', r''),
]

# Path cleanup for artifact references
ARTIFACT_PATH_REPLACEMENTS = []]],,,
(r'path: test/integration_test_results', r'path: integration_test_results'),
]

# Define the CI/CD workflow files to update
CI_CD_FILES = []]],,,
'.github/workflows/benchmark_db_ci.yml',
'.github/workflows/update_compatibility_matrix.yml',
'.github/workflows/test_and_benchmark.yml',
'.github/workflows/integration_tests.yml',
'.github/workflows/test_results_integration.yml',
]

def update_file(file_path):
    """Update path references in a single file."""
    if not os.path.exists(file_path):
        print(f"\1{file_path}\3")
    return
    
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
        original_content = content
        changes_made = 0
    
    # Apply mappings
    for old_pattern, new_pattern in PATH_MAPPINGS:
        new_content = re.sub(old_pattern, new_pattern, content)
        if new_content != content:
            changes_made += len(re.findall(old_pattern, content))
            content = new_content
    
    # Apply artifact path replacements
    for old_pattern, new_pattern in ARTIFACT_PATH_REPLACEMENTS:
        new_content = re.sub(old_pattern, new_pattern, content)
        if new_content != content:
            changes_made += len(re.findall(old_pattern, content))
            content = new_content
    
    # Write updated content if changes were made:
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
            print(f"\1{file_path}\3")
    else:
        print(f"\1{file_path}\3")

def main():
    """Update path references in all CI/CD workflow files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    total_files = 0
    total_changes = 0
    
    for rel_path in CI_CD_FILES:
        full_path = os.path.join(base_dir, rel_path)
        update_file(full_path)
        total_files += 1
    
        print(f"\nProcessed {total_files} files")

if __name__ == "__main__":
    main()