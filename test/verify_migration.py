#\!/usr/bin/env python3
"""
Comprehensive test script to verify the functionality of migrated modules.
This script attempts to import and perform basic testing of key components.
"""

import os
import sys
import traceback

def setup_paths():
    """Add the necessary paths to sys.path for imports to work"""
    # Add the parent directory to sys.path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    return parent_dir

def test_generators_imports():
    """Test importing key generator modules"""
    success_count = 0
    total_count = 0
    
    print("\n=== Testing Generators Package Imports ===")
    modules_to_test = [
        "generators.config",
        "generators.utils",
        "generators.test_generators.sample_test_generator",
        "generators.test_generators.qualified_test_generator",
        "generators.models.skill_hf_bert",
        "generators.templates.model_templates.template_bert",
        "generators.templates.template_selection",
        "generators.skill_generators.skill_generator",
    ]
    
    for module_name in modules_to_test:
        total_count += 1
        try:
            print(f"Importing {module_name}...", end="")
            __import__(module_name)
            print(" ✅")
            success_count += 1
        except Exception as e:
            print(f" ❌ - {str(e)}")
    
    print(f"\nGenerator imports: {success_count}/{total_count} successful ({success_count/total_count*100:.1f}%)")
    return success_count, total_count

def test_duckdb_api_imports():
    """Test importing key duckdb_api modules"""
    success_count = 0
    total_count = 0
    
    print("\n=== Testing DuckDB API Package Imports ===")
    modules_to_test = [
        "duckdb_api.schema",
        "duckdb_api.core.benchmark_db_query",
        "duckdb_api.migration",
        "duckdb_api.utils", 
        "duckdb_api.visualization",
    ]
    
    for module_name in modules_to_test:
        total_count += 1
        try:
            print(f"Importing {module_name}...", end="")
            __import__(module_name)
            print(" ✅")
            success_count += 1
        except Exception as e:
            print(f" ❌ - {str(e)}")
    
    print(f"\nDuckDB API imports: {success_count}/{total_count} successful ({success_count/total_count*100:.1f}%)")
    return success_count, total_count

def test_class_instantiation():
    """Try to instantiate key classes from migrated modules"""
    success_count = 0
    total_count = 0
    
    print("\n=== Testing Class Instantiation ===")
    
    # Test instantiating classes
    classes_to_test = [
        {
            "name": "generators.templates.model_templates.template_bert.BertTemplate",
            "args": [],
            "kwargs": {}
        },
        {
            "name": "generators.test_generators.qualified_test_generator.QualifiedTestGenerator",
            "args": [],
            "kwargs": {}
        }
    ]
    
    for class_info in classes_to_test:
        total_count += 1
        class_name = class_info["name"]
        args = class_info["args"]
        kwargs = class_info["kwargs"]
        
        try:
            print(f"Instantiating {class_name}...", end="")
            
            # Import the module and get the class
            module_name, class_attr = class_name.rsplit(".", 1)
            try:
                module = __import__(module_name, fromlist=[class_attr])
                class_obj = getattr(module, class_attr)
                
                # Try to instantiate it
                instance = class_obj(*args, **kwargs)
                print(" ✅")
                success_count += 1
            except (ImportError, AttributeError) as e:
                print(f" ❌ - Import error: {str(e)}")
            except Exception as e:
                print(f" ❌ - Instantiation error: {str(e)}")
        except Exception as e:
            print(f" ❌ - Unexpected error: {str(e)}")
    
    print(f"\nClass instantiation: {success_count}/{total_count} successful ({success_count/total_count*100 if total_count > 0 else 0:.1f}%)")
    return success_count, total_count

def run_tests():
    """Run all verification tests"""
    parent_dir = setup_paths()
    print(f"Parent directory: {parent_dir}")
    print(f"Python path: {sys.path}")
    
    # Run tests
    gen_success, gen_total = test_generators_imports()
    db_success, db_total = test_duckdb_api_imports()
    class_success, class_total = test_class_instantiation()
    
    # Calculate total scores
    total_success = gen_success + db_success + class_success
    total_tests = gen_total + db_total + class_total
    
    print("\n=== Migration Verification Summary ===")
    print(f"Generator imports: {gen_success}/{gen_total} successful ({gen_success/gen_total*100 if gen_total > 0 else 0:.1f}%)")
    print(f"DuckDB API imports: {db_success}/{db_total} successful ({db_success/db_total*100 if db_total > 0 else 0:.1f}%)")
    print(f"Class instantiation: {class_success}/{class_total} successful ({class_success/class_total*100 if class_total > 0 else 0:.1f}%)")
    print(f"Overall: {total_success}/{total_tests} successful ({total_success/total_tests*100 if total_tests > 0 else 0:.1f}%)")
    
    if total_success == total_tests:
        print("\n✅ All tests passed successfully\!")
        return 0
    else:
        print("\n⚠️ Some tests failed. See details above.")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
