#!/usr/bin/env python3
"""
Test script to simulate missing dependencies and verify mock detection.
This script temporarily modifies sys.modules to simulate missing dependencies,
then runs the test file to check if mock detection works correctly.
"""

import sys
import importlib
import os
from unittest.mock import MagicMock

def simulate_missing_dependency(module_name):
    """Simulate a missing dependency by adding a mock to sys.modules."""
    if module_name in sys.modules:
        print(f"Temporarily removing {module_name} from sys.modules")
        sys.modules[module_name] = MagicMock()
        sys.modules[module_name].__name__ = module_name
        return True
    else:
        print(f"Module {module_name} was not imported yet")
        sys.modules[module_name] = MagicMock()
        sys.modules[module_name].__name__ = module_name
        return False

def run_test_with_mocks(test_file, mock_modules=None):
    """Run a test file with specified modules mocked."""
    if mock_modules is None:
        mock_modules = ["torch", "transformers", "tokenizers"]
    
    # Store original modules if they exist
    originals = {}
    for module in mock_modules:
        if module in sys.modules:
            originals[module] = sys.modules[module]
            
    # Install mocks
    for module in mock_modules:
        simulate_missing_dependency(module)
    
    # Print status of mocked dependencies
    print("\nCurrent dependency status:")
    for module in mock_modules:
        print(f"  - {module}: MOCKED")
    print()
    
    try:
        # Run the test file as a subprocess to ensure clean environment
        result = os.system(f"python {test_file} --cpu-only")
        return result == 0
    finally:
        # Restore original modules
        for module, original in originals.items():
            sys.modules[module] = original

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test mock detection by simulating missing dependencies")
    parser.add_argument("--test-file", type=str, default="skills/fixed_tests/test_hf_bert.py", 
                      help="Path to test file to run with mocked dependencies")
    parser.add_argument("--mock", type=str, nargs='+', default=["torch", "transformers", "tokenizers"],
                      help="List of modules to mock")
    
    args = parser.parse_args()
    
    print(f"Running {args.test_file} with mocked dependencies: {', '.join(args.mock)}")
    success = run_test_with_mocks(args.test_file, args.mock)
    
    if success:
        print("✅ Test completed successfully")
    else:
        print("❌ Test failed")