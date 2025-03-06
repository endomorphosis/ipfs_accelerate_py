#!/usr/bin/env python
"""
Simple test script to explore the ipfs_accelerate_py package
"""

import sys
import importlib
import inspect
import pkgutil

def explore_module(module_name):
    """Explore the contents of a module"""
    print(f"\n=== Exploring module: {module_name} ===")
    
    try:
        # Import the module
        module = importlib.import_module(module_name)
        
        # Check if it's a package
        is_package = hasattr(module, '__path__')
        print(f"Is a package: {is_package}")
        
        # Print module path
        if hasattr(module, '__file__'):
            print(f"Module path: {module.__file__}")
        else:
            print("Module has no __file__ attribute")
        
        # List module attributes
        print("\nAttributes:")
        for name in dir(module):
            if not name.startswith('__'):
                try:
                    attr = getattr(module, name)
                    attr_type = type(attr).__name__
                    attr_module = getattr(attr, '__module__', 'None')
                    
                    if inspect.ismodule(attr):
                        print(f"  {name}: Module from {attr_module}")
                    elif inspect.isclass(attr):
                        print(f"  {name}: Class from {attr_module}")
                    elif inspect.isfunction(attr):
                        print(f"  {name}: Function from {attr_module}")
                    elif inspect.ismethod(attr):
                        print(f"  {name}: Method from {attr_module}")
                    else:
                        print(f"  {name}: {attr_type} from {attr_module}")
                except Exception as e:
                    print(f"  {name}: Error getting attribute: {e}")
        
        # List submodules if it's a package
        if is_package:
            print("\nSubmodules:")
            for _, name, is_pkg in pkgutil.iter_modules(module.__path__):
                print(f"  {name} {'(package)' if is_pkg else '(module)'}")
        
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")
    except Exception as e:
        print(f"Error exploring {module_name}: {e}")

def main():
    """Main function"""
    # Explore the main package
    explore_module('ipfs_accelerate_py')
    
    # Explore submodules
    potential_submodules = [
        'ipfs_accelerate_py.api_backends',
        'ipfs_accelerate_py.ipfs_accelerate',
        'ipfs_accelerate_py.config',
        'ipfs_accelerate_py.backends'
    ]
    
    for module_name in potential_submodules:
        try:
            explore_module(module_name)
        except Exception as e:
            print(f"Error exploring {module_name}: {e}")

if __name__ == "__main__":
    main()