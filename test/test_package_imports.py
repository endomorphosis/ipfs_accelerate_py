#!/usr/bin/env python3
"""
Tests that Python package imports work correctly for the reorganized code structure.
This script tests imports with different PYTHONPATH configurations.
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test package imports with various configurations."""
    project_root = Path(__file__).parent.parent.absolute()
    print(f"Project root: {project_root}")
    
    # Absolute path to simplify imports
    sys.path.insert(0, str(project_root))
    
    print("\n==== Testing root package imports ====")
    try:
        import generators
        print(f"✅ Imported generators package: {generators.__file__}")
    except ImportError as e:
        print(f"❌ Failed to import generators package: {str(e)}")
    
    try:
        import duckdb_api
        print(f"✅ Imported duckdb_api package: {duckdb_api.__file__}")
    except ImportError as e:
        print(f"❌ Failed to import duckdb_api package: {str(e)}")
    
    print("\n==== Testing subpackage imports ====")
    try:
        from generators.test_generators import simple_test_generator
        print(f"✅ Imported simple_test_generator: {simple_test_generator.__file__}")
    except ImportError as e:
        print(f"❌ Failed to import simple_test_generator: {str(e)}")
    
    try:
        from generators.utils import utils
        print(f"✅ Imported utils: {utils.__file__}")
    except ImportError as e:
        print(f"❌ Failed to import utils: {str(e)}")
        
    try:
        from duckdb_api.core import benchmark_db_api
        print(f"✅ Imported benchmark_db_api: {benchmark_db_api.__file__}")
    except ImportError as e:
        print(f"❌ Failed to import benchmark_db_api: {str(e)}")
        
    try:
        from duckdb_api.utils import cleanup_stale_reports
        print(f"✅ Imported cleanup_stale_reports: {cleanup_stale_reports.__file__}")
    except ImportError as e:
        print(f"❌ Failed to import cleanup_stale_reports: {str(e)}")
    
    print("\n==== Testing deep imports ====")
    try:
        from generators.templates.model_templates import text_model_template
        print(f"✅ Imported text_model_template module")
    except ImportError as e:
        print(f"❌ Failed to import text_model_template: {str(e)}")
    
    try:
        from duckdb_api.schema.creation import create_benchmark_schema
        print(f"✅ Imported create_benchmark_schema module")
    except ImportError as e:
        print(f"❌ Failed to import create_benchmark_schema: {str(e)}")
    
    print("\n==== Import test summary ====")
    print("If you see any failures, check that:")
    print("1. The module exists in the correct location")
    print("2. All directories have proper __init__.py files")
    print("3. Your PYTHONPATH includes the project root")
    print(f"   Current PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
if __name__ == "__main__":
    test_imports()