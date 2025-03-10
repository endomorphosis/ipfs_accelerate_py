#\!/usr/bin/env python3
"""
Test the import functionality of migrated modules
"""

import sys
import os
import traceback

def test_generator_imports():
    """Test imports from generators package"""
    print("Testing generators imports...")
    try:
        # Add the parent directory to the Python path so we can import generators
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Test importing from generators
        from generators import utils
        print("✅ Successfully imported generators.utils")
        
        from generators.test_generators import sample_test_generator
        print("✅ Successfully imported generators.test_generators.sample_test_generator")
        
        from generators.models import skill_hf_bert
        print("✅ Successfully imported generators.models.skill_hf_bert")
        
        from generators.templates.model_templates import template_bert
        print("✅ Successfully imported generators.templates.model_templates.template_bert")
        
        return True
    except Exception as e:
        print(f"❌ Error importing generators modules: {e}")
        traceback.print_exc()
        return False

def test_duckdb_api_imports():
    """Test imports from duckdb_api package"""
    print("\nTesting duckdb_api imports...")
    try:
        # Test importing from duckdb_api
        from duckdb_api import benchmark_db_api
        print("✅ Successfully imported duckdb_api.benchmark_db_api")
        
        from duckdb_api.schema import check_database_schema
        print("✅ Successfully imported duckdb_api.schema.check_database_schema")
        
        from duckdb_api.core import benchmark_db_query
        print("✅ Successfully imported duckdb_api.core.benchmark_db_query")
        
        from duckdb_api.utils import cleanup_stale_reports
        print("✅ Successfully imported duckdb_api.utils.cleanup_stale_reports")
        
        return True
    except Exception as e:
        print(f"❌ Error importing duckdb_api modules: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing module imports for reorganized structure...")
    
    generators_success = test_generator_imports()
    duckdb_api_success = test_duckdb_api_imports()
    
    if generators_success and duckdb_api_success:
        print("\n✅ All import tests passed successfully\!")
        sys.exit(0)
    else:
        print("\n❌ Some import tests failed.")
        sys.exit(1)
