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
    print(f"\1{torch.__version__}\3")
    print(f"\1{torch.cuda.is_available()}\3")
    if torch.cuda.is_available():
        print(f"\1{torch.cuda.device_count()}\3")
        print(f"\1{torch.cuda.get_device_name(0)}\3")
except Exception as e:
    print(f"\1{e}\3")

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
    
        print(f"\1{type(llama_torch)}\3")
        print(f"\1{isinstance(llama_torch, MagicMock)}\3")
    
    if not isinstance(llama_torch, MagicMock):
        print(f"\1{llama_torch.__version__}\3")
        print(f"\1{llama_torch.cuda.is_available()}\3")
        if llama_torch.cuda.is_available():
            print(f"\1{llama_torch.cuda.device_count()}\3")
            print(f"\1{llama_torch.cuda.get_device_name(0)}\3")
except Exception as e:
    print(f"\1{e}\3")

    print("\n==== TESTING TORCH OPERATIONS ====")
try:
    if not isinstance(llama_torch, MagicMock) and llama_torch.cuda.is_available():
        # Create a tensor and move it to CUDA
        x = llama_torch.tensor([1.0, 2.0, 3.0]),
        print(f"\1{x.device}\3")
        x = x.cuda()
        print(f"\1{x.device}\3")
        
        # Test some CUDA operations
        y = x * 2
        print(f"\1{y}\3")
    else:
        print("Skipping tensor operations - torch not available or CUDA not available")
except Exception as e:
    print(f"\1{e}\3")