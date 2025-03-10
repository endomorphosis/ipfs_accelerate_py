#!/usr/bin/env python
"""
Script to migrate files for CI/CD workflow compatibility.
This script will move files identified by verify_cicd_paths.py to their required locations
and create the necessary directory structure.

Usage: python migrate_files_for_ci.py
"""

import os
import shutil
import json
from pathlib import Path

# Migration mapping from verification results
MIGRATION_MAP = [
    {
        "source": "../test/scripts/benchmark_regression_detector.py",
        "target": "../duckdb_api/analysis/benchmark_regression_detector.py"
    },
    {
        "source": "../test/duckdb_api/core/run_benchmark_with_db.py",
        "target": "../duckdb_api/core/run_benchmark_with_db.py" 
    },
    {
        "source": "../test/scripts/ci_benchmark_integrator.py",
        "target": "../duckdb_api/scripts/ci_benchmark_integrator.py"
    },
    {
        "source": "../test/scripts/benchmark_db/create_benchmark_schema.py", 
        "target": "../duckdb_api/scripts/create_benchmark_schema.py"
    },
    {
        "source": "../generators/generate_compatibility_matrix.py",
        "target": "../duckdb_api/visualization/generate_compatibility_matrix.py"
    },
    {
        "source": "../generators/generate_enhanced_compatibility_matrix.py",
        "target": "../duckdb_api/visualization/generate_enhanced_compatibility_matrix.py"
    },
    {
        "source": "../test/archive/web_platform_test_runner.py",
        "target": "../fixed_web_platform/web_platform_test_runner.py"
    },
    {
        "source": "../test/test_ipfs_accelerate.py",
        "target": "../generators/models/test_ipfs_accelerate.py"
    },
    {
        "source": "../test/integration_test_suite.py",
        "target": "../generators/test_runners/integration_test_suite.py"
    },
    {
        "source": "../test/archive/hardware_model_predictor.py",
        "target": "../predictive_performance/hardware_model_predictor.py"
    },
    {
        "source": "../test/archive/model_performance_predictor.py",
        "target": "../predictive_performance/model_performance_predictor.py"
    }
]

def migrate_files(dry_run=True):
    """
    Migrate files according to the migration map.
    
    Args:
        dry_run (bool): If True, only print what would be done without making changes.
    """
    project_root = Path.cwd()
    
    # Prepare a summary
    successful = []
    failed = []
    
    for item in MIGRATION_MAP:
        source_path = project_root / item["source"]
        target_path = project_root / item["target"]
        
        # Check if source exists
        if not source_path.exists():
            print(f"❌ Source not found: {item['source']}")
            failed.append({"source": item["source"], "target": item["target"], "reason": "Source not found"})
            continue
        
        # Check if target already exists
        if target_path.exists():
            print(f"⚠️ Target already exists: {item['target']}")
            failed.append({"source": item["source"], "target": item["target"], "reason": "Target already exists"})
            continue
        
        # Create target directory if it doesn't exist
        target_dir = target_path.parent
        if not target_dir.exists():
            if dry_run:
                print(f"Would create directory: {target_dir}")
            else:
                print(f"Creating directory: {target_dir}")
                target_dir.mkdir(parents=True, exist_ok=True)
        
        # Move the file
        if dry_run:
            print(f"Would move: {item['source']} -> {item['target']}")
            successful.append({"source": item["source"], "target": item["target"]})
        else:
            try:
                print(f"Moving: {item['source']} -> {item['target']}")
                shutil.copy2(source_path, target_path)
                successful.append({"source": item["source"], "target": item["target"]})
            except Exception as e:
                print(f"❌ Error moving file: {str(e)}")
                failed.append({"source": item["source"], "target": item["target"], "reason": str(e)})
    
    # Print summary
    print("\n" + "="*50)
    print(f"Migration Summary (Dry Run: {dry_run})")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    # Generate migration report
    report = {
        "dry_run": dry_run,
        "successful": successful,
        "failed": failed
    }
    
    report_path = project_root / "migration_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Migration report saved to: {report_path}")
    
    return len(failed) == 0

def main():
    print("CI/CD File Migration Tool")
    print("="*50)
    print("This tool will migrate files to match CI/CD workflow expectations.")
    print("Default mode is dry-run (no changes made).")
    
    choice = input("Do you want to run in dry-run mode? [Y/n]: ").strip().lower()
    dry_run = choice != 'n'
    
    if not dry_run:
        confirm = input("⚠️ This will make changes to your filesystem. Are you sure? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Operation cancelled.")
            return
    
    success = migrate_files(dry_run)
    
    if dry_run:
        print("\nRun again with dry-run=False to perform the actual migration.")
    
    return 0 if success else 1

if __name__ == "__main__":
    main()