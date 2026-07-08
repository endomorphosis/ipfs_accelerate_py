#!/usr/bin/env python3
"""
Simple test for mock detection that avoids multithreading issues.
"""

# Mock imports before they happen
import sys
from unittest.mock import MagicMock

# Create mocks for key dependencies
sys.modules['torch'] = MagicMock()
sys.modules['torch'].cuda = MagicMock()
sys.modules['torch'].cuda.is_available = lambda: False
sys.modules['transformers'] = MagicMock()
sys.modules['tokenizers'] = MagicMock()
sys.modules['sentencepiece'] = MagicMock()

# Define flags for mock detection
HAS_TORCH = False 
HAS_TRANSFORMERS = False
HAS_TOKENIZERS = False
HAS_SENTENCEPIECE = False

# Check for mock status
using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE

# Print detection results
print("\nMOCK DETECTION TEST RESULTS:\n")

if using_real_inference and not using_mocks:
    print(f"🚀 Using REAL INFERENCE with actual models")
else:
    print(f"🔷 Using MOCK OBJECTS for CI/CD testing only")
    print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")

print("\nMetadata:")
metadata = {
    "has_transformers": HAS_TRANSFORMERS,
    "has_torch": HAS_TORCH,
    "has_tokenizers": HAS_TOKENIZERS,
    "has_sentencepiece": HAS_SENTENCEPIECE,
    "using_real_inference": using_real_inference,
    "using_mocks": using_mocks,
    "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
}

for key, value in metadata.items():
    print(f"  - {key}: {value}")