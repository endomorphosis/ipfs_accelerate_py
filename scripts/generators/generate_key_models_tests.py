#!/usr/bin/env python3
"""
Generate Key Models Tests

This script generates tests for all key models using the minimal generator.
It focuses on the 13 high-priority models from CLAUDE.md.
"""

import os
import sys
import subprocess
from pathlib import Path

# Current directory
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Key models from CLAUDE.md
KEY_MODELS = [
    "bert-base-uncased",
    "t5-small",
    "opt-125m",  # llama family
    "clip-vit-base-patch32",
    "vit-base-patch16-224",
    "clap-htsat-flan-t5",  # clap example
    "whisper-tiny",
    "wav2vec2-base",
    "llava-v1.5-7b",
    "llava-next-vicuna-7b",
    "xclip-base-patch32",
    "qwen2-7b",
    "detr-resnet-50"
]

# Make sure minimal_generator.py exists
GENERATOR_PATH = CURRENT_DIR / "minimal_generator.py"
if not GENERATOR_PATH.exists():
    print("Error: minimal_generator.py not found. Create it first.")
    sys.exit(1)

# Create output directory
OUTPUT_DIR = CURRENT_DIR / "generated_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate tests for each model
success_count = 0
for model in KEY_MODELS:
    try:
        print(f"Generating test for: {model}")
        cmd = [
            sys.executable,
            str(GENERATOR_PATH),
            "--generate", model,
            "--platform", "all",
            "--output-dir", str(OUTPUT_DIR)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully generated test for {model}")
            success_count += 1
        else:
            print(f"❌ Failed to generate test for {model}: {result.stderr}")
    except Exception as e:
        print(f"❌ Error generating test for {model}: {str(e)}")

print(f"\nGenerated {success_count} of {len(KEY_MODELS)} test files.")
print(f"Output directory: {OUTPUT_DIR}")

if success_count == len(KEY_MODELS):
    print("\nAll key model tests generated successfully!")
else:
    print(f"\nWARNING: {len(KEY_MODELS) - success_count} models failed.")