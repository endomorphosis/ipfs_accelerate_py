#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script to demonstrate the HuggingFace model test generator suite.
This script shows how to use the TestGeneratorSuite class to generate test files.
"""

import os
import sys
import logging
from pathlib import Path
from test_generator_suite import TestGeneratorSuite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_generator_example")

def main():
    """Run the example test generator."""
    
    # Initialize the generator
    generator = TestGeneratorSuite()
    
    # Create output directory
    output_dir = Path(__file__).parent / "generated_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # List of models to generate
    models = [
        "bert",      # Encoder-only
        "gpt2",      # Decoder-only
        "t5",        # Encoder-decoder
        "vit",       # Vision
        "wav2vec2",  # Speech
        "clip",      # Vision-text
    ]
    
    # Generate tests for each model
    results = []
    for model in models:
        logger.info(f"Generating test for {model}...")
        output_file = output_dir / f"test_hf_{model}.py"
        result = generator.generate_test(model, str(output_file))
        results.append(result)
        
        if result["success"]:
            logger.info(f"✅ Successfully generated test for {model}")
        else:
            logger.error(f"❌ Failed to generate test for {model}: {result.get('error')}")
    
    # Print summary
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    
    print("\n" + "="*50)
    print(f"Test Generation Summary: {successful}/{total} successful")
    print("="*50)
    
    for i, (model, result) in enumerate(zip(models, results)):
        status = "✅ Success" if result["success"] else f"❌ Failed: {result.get('error')}"
        print(f"{i+1}. {model}: {status}")
    
    print("\nGenerated files:")
    for path in sorted(output_dir.glob("test_*.py")):
        print(f"- {path.relative_to(path.parent.parent)}")
    
    return 0 if successful == total else 1

if __name__ == "__main__":
    sys.exit(main())