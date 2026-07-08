#!/usr/bin/env python3
"""
Quick script to verify the migration of key files to the new structure.
"""

import os
import sys
import importlib

def setup_paths():
    """Add the necessary paths to sys.path for imports to work"""
    # Add the parent directory to sys.path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    return parent_dir

def test_imports():
    """Test importing key modules"""
    success_count = 0
    total_count = 0
    
    print("\n=== Testing Imports ===")
    modules_to_test = [
        "generators.test_generators.fixed_merged_test_generator_clean",
        "generators.utils.test_generator_with_resource_pool", 
        "generators.utils.resource_pool"
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
    
    print(f"\nImports: {success_count}/{total_count} successful ({success_count/total_count*100 if total_count > 0 else 0:.1f}%)")
    return success_count, total_count

def main():
    """Run verification tests"""
    parent_dir = setup_paths()
    print(f"Parent directory: {parent_dir}")
    
    # Run tests
    import_success, import_total = test_imports()
    
    # Report
    print("\n=== Verification Summary ===")
    print(f"Imports: {import_success}/{import_total} successful ({import_success/import_total*100 if import_total > 0 else 0:.1f}%)")
    
    if import_success == import_total:
        print("\n✅ All tests passed successfully!")
        return 0
    else:
        print("\n⚠️ Some tests failed. See details above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())