#!/usr/bin/env python

import os
import time
import json
import importlib
import argparse
import sys

def print_module_details(module_name):
    """Print details about a Python module"""
    try:
        module = importlib.import_module(module_name)
        print(f"Module {module_name} is available")
        print(f"Path: {getattr(module, '__file__', 'No file path')}")
        
        # Print attributes
        for attr_name in dir(module):
            if not attr_name.startswith('__'):
                try:
                    attr = getattr(module, attr_name)
                    attr_type = type(attr).__name__
                    print(f"  {attr_name}: {attr_type}")
                except:
                    print(f"  {attr_name}: [Error getting attribute]")
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")

def test_ipfs_accelerate():
    """Test basic functionality of the ipfs_accelerate_py package"""
    print("\n=== Testing ipfs_accelerate_py ===")
    
    # Try to import the main module
    try:
        import ipfs_accelerate_py
        print("Successfully imported ipfs_accelerate_py")
        print(f"Package path: {ipfs_accelerate_py.__file__}")
        
        # Try to access main components
        if hasattr(ipfs_accelerate_py, 'ipfs_accelerate'):
            print("Found ipfs_accelerate module")
            ipfs_module = ipfs_accelerate_py.ipfs_accelerate
            
            # Try to find load_checkpoint_and_dispatch function
            if hasattr(ipfs_module, 'load_checkpoint_and_dispatch'):
                print("Found load_checkpoint_and_dispatch function")
            else:
                print("Missing load_checkpoint_and_dispatch function")
        else:
            print("Missing ipfs_accelerate module")
            
        # Check for expected attributes
        attributes = ['backends', 'config', 'ipfs_accelerate', 'load_checkpoint_and_dispatch']
        for attr in attributes:
            has_attr = hasattr(ipfs_accelerate_py, attr)
            print(f"Has {attr}: {has_attr}")
            
        # Explore the backends module
        if hasattr(ipfs_accelerate_py, 'backends'):
            print("\nExploring backends:")
            print_module_details("ipfs_accelerate_py.backends")
            
        # Explore the config module
        if hasattr(ipfs_accelerate_py, 'config'):
            print("\nExploring config:")
            print_module_details("ipfs_accelerate_py.config")
            
    except ImportError as e:
        print(f"Failed to import ipfs_accelerate_py: {e}")
    except Exception as e:
        print(f"Error testing ipfs_accelerate_py: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test the IPFS Accelerate Python package")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")
    args = parser.parse_args()
    
    # Set up verbose printing
    global verbose
    verbose = args.verbose
    
    # Print Python environment info
    print("=== Python Environment ===")
    print(f"Python version: {sys.version}")
    print(f"System platform: {sys.platform}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Run basic test
    test_ipfs_accelerate()
    
    print("\nTest complete")

if __name__ == "__main__":
    main()