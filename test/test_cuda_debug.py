#!/usr/bin/env python3

import os
import sys
import time

# This script is used to debug the CUDA availability issue
print("==== TORCH CUDA DEBUG ====")

# Add path to import 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import torch normally
try:
    import torch
    print(f"torch imported normally: {torch.__version__}")
    print(f"cuda available (torch.cuda.is_available()): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"Error importing torch: {e}")

# Try importing specifically the way test_hf_llama.py does it
print("\n==== IMPORT LIKE IN TEST_HF_LLAMA.PY ====")
try:
    from unittest.mock import MagicMock, patch
    import numpy as np
    
    try:
        import torch as llama_torch
    except ImportError:
        llama_torch = MagicMock()
        print("Warning: torch not available, using mock implementation")
    
    try:
        import transformers as llama_transformers
    except ImportError:
        llama_transformers = MagicMock()
        print("Warning: transformers not available, using mock implementation")
    
    print(f"llama_torch type: {type(llama_torch)}")
    print(f"Is llama_torch a MagicMock? {isinstance(llama_torch, MagicMock)}")
    
    if not isinstance(llama_torch, MagicMock):
        print(f"llama_torch version: {llama_torch.__version__}")
        print(f"cuda available via llama_torch: {llama_torch.cuda.is_available()}")
        if llama_torch.cuda.is_available():
            print(f"llama_torch.cuda.device_count(): {llama_torch.cuda.device_count()}")
            print(f"llama_torch.cuda.get_device_name(0): {llama_torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"Error in llama import test: {e}")

print("\n==== TESTING TORCH OPERATIONS ====")
try:
    if not isinstance(llama_torch, MagicMock) and llama_torch.cuda.is_available():
        # Create a tensor and move it to CUDA
        x = llama_torch.tensor([1.0, 2.0, 3.0])
        print(f"x.device before: {x.device}")
        x = x.cuda()
        print(f"x.device after: {x.device}")
        
        # Test some CUDA operations
        y = x * 2
        print(f"y = x * 2: {y}")
    else:
        print("Skipping tensor operations - torch not available or CUDA not available")
except Exception as e:
    print(f"Error in tensor operations: {e}")