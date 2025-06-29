#!/usr/bin/env python3
"""
Directly run tests with intentionally mocked imports to test our detection code.
"""

import sys
import importlib
from unittest.mock import MagicMock

# Block imports before they happen
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['tokenizers'] = MagicMock()
sys.modules['sentencepiece'] = MagicMock()

# Now run the tests
if __name__ == "__main__":
    import os
    import importlib.util
    
    # Print mocked modules
    print("RUNNING WITH MOCKED DEPENDENCIES:")
    print("  - torch: MOCKED")
    print("  - transformers: MOCKED")
    print("  - tokenizers: MOCKED")
    print("  - sentencepiece: MOCKED")
    print("\n" + "="*50 + "\n")
    
    # Load the test file as a module
    test_path = os.path.join("skills", "fixed_tests", "test_hf_bert.py")
    spec = importlib.util.spec_from_file_location("test_hf_bert", test_path)
    test_module = importlib.util.module_from_spec(spec)
    
    # Run the test module's main function
    sys.argv = [test_path, "--cpu-only"]
    spec.loader.exec_module(test_module)