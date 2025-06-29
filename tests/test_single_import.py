#!/usr/bin/env python
"""
Simple script to test importing a single module.
"""

import importlib.util
import sys
from pathlib import Path

def test_import(module_path):
    """
    Attempt to import a single module and print dependency errors.
    
    Args:
        module_path (str): Path to the module to import
    """
    print(f"Attempting to import: {module_path}")
    try:
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        if spec is None:
            print(f"Error: Could not find module specification for {module_path}")
            return
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["test_module"] = module
        
        # Attempt to execute the module
        spec.loader.exec_module(module)
        
        print(f"Successfully imported {module_path}")
        
    except ImportError as e:
        print(f"Import Error: {str(e)}")
        print("This likely indicates a missing dependency.")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        
if __name__ == "__main__":
    project_root = Path.cwd().parent
    
    # Test DuckDB API
    duckdb_file = project_root / "duckdb_api" / "analysis" / "benchmark_regression_detector.py"
    if duckdb_file.exists():
        test_import(duckdb_file)
    else:
        print(f"File not found: {duckdb_file}")
        
    # Test integration test suite
    integration_file = project_root / "generators" / "test_runners" / "integration_test_suite.py"
    if integration_file.exists():
        test_import(integration_file)
    else:
        print(f"File not found: {integration_file}")
        
    # Test IPFS accelerate
    ipfs_file = project_root / "generators" / "models" / "test_ipfs_accelerate.py"
    if ipfs_file.exists():
        test_import(ipfs_file)
    else:
        print(f"File not found: {ipfs_file}")