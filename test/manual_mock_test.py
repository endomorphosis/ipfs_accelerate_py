#!/usr/bin/env python3
"""
Manual test for mock detection to verify the emoji indicators.
This simulates both real and mock scenarios in a single script.
"""

import os
import sys
import contextlib
import io

print("\n===== REAL DEPENDENCIES TEST =====")
# Test with real dependencies
HAS_TRANSFORMERS = True
HAS_TORCH = True
HAS_TOKENIZERS = True
HAS_SENTENCEPIECE = True

# Apply detection logic
using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE

print("\nTEST RESULTS SUMMARY:")

# Display appropriate indicator
if using_real_inference and not using_mocks:
    print(f"🚀 Using REAL INFERENCE with actual models")
else:
    print(f"🔷 Using MOCK OBJECTS for CI/CD testing only")
    print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")

print("\n===== MOCK DEPENDENCIES TEST =====")
# Test with mock dependencies
HAS_TRANSFORMERS = False  # Mock transformers
HAS_TORCH = True
HAS_TOKENIZERS = True
HAS_SENTENCEPIECE = True

# Apply detection logic
using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE

print("\nTEST RESULTS SUMMARY:")

# Display appropriate indicator
if using_real_inference and not using_mocks:
    print(f"🚀 Using REAL INFERENCE with actual models")
else:
    print(f"🔷 Using MOCK OBJECTS for CI/CD testing only")
    print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")

print("\n===== PARTIAL MOCK DEPENDENCIES TEST =====")
# Test with partial mock dependencies
HAS_TRANSFORMERS = True
HAS_TORCH = True
HAS_TOKENIZERS = False  # Mock tokenizers 
HAS_SENTENCEPIECE = True

# Apply detection logic
using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE

print("\nTEST RESULTS SUMMARY:")

# Display appropriate indicator
if using_real_inference and not using_mocks:
    print(f"🚀 Using REAL INFERENCE with actual models")
else:
    print(f"🔷 Using MOCK OBJECTS for CI/CD testing only")
    print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")

print("\n===== VERIFICATION SUMMARY =====")
print("\nThis test verifies that our mock detection logic correctly identifies:")
print("1. When all dependencies are available (🚀 REAL INFERENCE indicator)")
print("2. When core dependencies are missing (🔷 MOCK OBJECTS indicator)")
print("3. When support dependencies are missing (🔷 MOCK OBJECTS indicator)")
print("\nNote: In all test files, we check for:")
print("- Core: transformers, torch")
print("- Support: tokenizers, sentencepiece")
print("- If ANY of these are missing, we show the mock objects indicator")