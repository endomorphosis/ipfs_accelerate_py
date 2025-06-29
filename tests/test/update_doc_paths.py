#!/usr/bin/env python3
"""
Documentation Path Updater

This script automatically updates file paths in documentation files (.md) to reflect
the new directory structure after the reorganization of files from test/ to
generators/ and duckdb_api/ directories.

Usage:
    python update_doc_paths.py [],--file path/to/file.md] [],--dir path/to/dir] [],--dry-run],
    """

    import argparse
    import os
    import re
    import sys
    from pathlib import Path

# Mapping of path patterns to their new locations
    PATH_MAPPINGS = [],
    # Generator-related files
    (r'python\s+test/([],^/]+_generator\.py)', r'python generators/\1'),
    (r'python\s+test/(merged_test_generator\.py)', r'python generators/test_generators/\1'),
    (r'python\s+test/(integrated_skillset_generator\.py)', r'python generators/\1'),
    (r'python\s+test/(template_[],^\.]+\.py)', r'python generators/templates/\1'),
    (r'python\s+test/(web_platform_test_runner\.py)', r'python generators/runners/web/\1'),
    (r'python\s+test/(verify_[],^\.]+\.py)', r'python generators/validators/\1'),
    (r'python\s+test/(run_model_benchmarks\.py)', r'python generators/benchmark_generators/\1'),
    (r'python\s+test/(test_web_platform_integration\.py)', r'python generators/web/\1'),
    
    # Database-related files
    (r'python\s+test/(benchmark_db_[],^\.]+\.py)', r'python duckdb_api/core/\1'),
    (r'python\s+test/(run_comprehensive_benchmark_timing\.py)', r'python duckdb_api/visualization/\1'),
    (r'python\s+test/(run_time_series_performance\.py)', r'python duckdb_api/utils/\1'),
    (r'python\s+test/(time_series_performance\.py)', r'python duckdb_api/core/\1'),
    (r'python\s+test/scripts/(benchmark_db_query\.py)', r'python duckdb_api/core/\1'),
    (r'python\s+test/scripts/(ci_benchmark_timing_report\.py)', r'python duckdb_api/ci/\1'),
    (r'python\s+test/examples/(run_benchmark_timing_example\.py)', r'python duckdb_api/examples/\1'),
    (r'python\s+test/(run_web_platform_tests_with_db\.py)', r'python duckdb_api/web/\1'),
    
    # Import statements
    (r'from\s+time_series_performance\s+import', r'from duckdb_api.core.time_series_performance import'),
    (r'from\s+benchmark_db_[],^\s]+\s+import', lambda match: f'from duckdb_api.core.{match.group(0).split(" ")[],1]} import'),
    
    # Schema files
    (r'test/db_schema/([],^\.]+\.sql)', r'duckdb_api/schema/\1'),
    
    # Absolute paths
    (r'/home/barberb/ipfs_accelerate_py/test/([],^/]+_generator\.py)', r'/home/barberb/ipfs_accelerate_py/generators/\1'),
    (r'/home/barberb/ipfs_accelerate_py/test/(merged_test_generator\.py)', r'/home/barberb/ipfs_accelerate_py/generators/test_generators/\1'),
    (r'/home/barberb/ipfs_accelerate_py/test/(integrated_skillset_generator\.py)', r'/home/barberb/ipfs_accelerate_py/generators/\1'),
    (r'/home/barberb/ipfs_accelerate_py/test/(benchmark_db_[],^\.]+\.py)', r'/home/barberb/ipfs_accelerate_py/duckdb_api/core/\1'),
    (r'/home/barberb/ipfs_accelerate_py/test/(fixed_benchmark_db_query\.py)', r'/home/barberb/ipfs_accelerate_py/duckdb_api/core/benchmark_db_query.py'),
    
    # Simple command invocations (without python prefix)
    (r'fixed_benchmark_db_query\.py', r'duckdb_api/core/benchmark_db_query.py'),
    ]

def update_file_paths(file_path, dry_run=False):
    """Update file paths in a documentation file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
            original_content = content
            changes_made = 0
        
        for pattern, replacement in PATH_MAPPINGS:
            new_content, count = re.subn(pattern, replacement, content)
            if count > 0:
                content = new_content
                changes_made += count
        
        if changes_made > 0:
            print(f"\1{file_path}\3")
            
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    print(f"\1{file_path}\3")
            else:
                print(f"\1{file_path}\3")
        else:
            print(f"\1{file_path}\3")
        
                return changes_made > 0
        
    except Exception as e:
        print(f"\1{e}\3")
                return False

def process_directory(directory, dry_run=False):
    """Process all markdown files in the directory."""
    updated_files = 0
    total_files = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.md'):
                file_path = os.path.join(root, file)
                if update_file_paths(file_path, dry_run):
                    updated_files += 1
                    total_files += 1
    
                    print(f"\nSummary: Updated {updated_files} out of {total_files} files.")
                return updated_files, total_files

def main():
    parser = argparse.ArgumentParser(description='Update file paths in documentation files')
    parser.add_argument('--file', help='Path to a specific file to update')
    parser.add_argument('--dir', help='Path to a directory containing files to update')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying them')
    
    args = parser.parse_args()
    
    if args.file:
        update_file_paths(args.file, args.dry_run)
    elif args.dir:
        process_directory(args.dir, args.dry_run)
    else:
        # Default to current directory if no file or directory is specified
        current_dir = os.getcwd():
            print(f"\1{current_dir}\3")
            process_directory(current_dir, args.dry_run)

if __name__ == "__main__":
    main()