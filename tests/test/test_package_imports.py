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
    print(f"\1{project_root}\3")
    
    # Absolute path to simplify imports
    sys.path.insert(0, str(project_root))
    
    print("\n==== Testing root package imports ====")
    try:
        import generators
        print(f"\1{generators.__file__}\3")
    except ImportError as e:
        print(f"\1{str(e)}\3")
    
    try:
        import duckdb_api
        print(f"\1{duckdb_api.__file__}\3")
    except ImportError as e:
        print(f"\1{str(e)}\3")
    
        print("\n==== Testing subpackage imports ====")
    try:
        from generators.test_generators import generators.test_generators.simple_test_generator as simple_test_generator
        print(f"\1{simple_test_generator.__file__}\3")
    except ImportError as e:
        print(f"\1{str(e)}\3")
    
    try:
        from generators.utils import utils
        print(f"\1{utils.__file__}\3")
    except ImportError as e:
        print(f"\1{str(e)}\3")
        
    try:
        from duckdb_api.core import duckdb_api.core.benchmark_db_api as benchmark_db_api
        print(f"\1{benchmark_db_api.__file__}\3")
    except ImportError as e:
        print(f"\1{str(e)}\3")
        
    try:
        from duckdb_api.utils import cleanup_stale_reports
        print(f"\1{cleanup_stale_reports.__file__}\3")
    except ImportError as e:
        print(f"\1{str(e)}\3")
    
        print("\n==== Testing deep imports ====")
    try:
        from generators.templates.model_templates import text_model_template
        print(f"✅ Imported text_model_template module")
    except ImportError as e:
        print(f"\1{str(e)}\3")
    
    try:
        from duckdb_api.schema.creation import create_benchmark_schema
        print(f"✅ Imported create_benchmark_schema module")
    except ImportError as e:
        print(f"\1{str(e)}\3")
    
        print("\n==== Import test summary ====")
        print("If you see any failures, check that:")
        print("1. The module exists in the correct location")
        print("2. All directories have proper __init__.py files")
        print("3. Your PYTHONPATH includes the project root")
        print(f"\1{os.environ.get('PYTHONPATH', 'Not set')}\3")
    
if __name__ == "__main__":
    test_imports()