#!/usr/bin/env python3
"""
Script to verify the implementation of 13 key model tests.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Define the 13 key models
KEY_MODELS = [
    "bert",
    "clap",
    "clip",
    "detr",
    "llama",
    "llava",
    "llava_next",
    "qwen2",
    "t5",
    "vit",
    "wav2vec2",
    "whisper",
    "xclip"
]

# Base directories
SKILLS_DIR = Path("./skills")
MODALITY_TESTS_DIR = Path("./modality_tests")
FIXED_GENERATED_TESTS_DIR = Path("./fixed_generated_tests")
NEW_GENERATED_TESTS_DIR = Path("./new_generated_tests")

def check_key_model_tests():
    """Check if test files for key models exist and report their status."""
    results = {}
    missing_models = []
    
    for model in KEY_MODELS:
        # Construct potential filenames
        test_filename = f"test_hf_{model}.py"
        test_files = {
            "skills": SKILLS_DIR / test_filename,
            "modality_tests": MODALITY_TESTS_DIR / test_filename,
            "fixed_generated_tests": FIXED_GENERATED_TESTS_DIR / test_filename,
            "new_generated_tests": NEW_GENERATED_TESTS_DIR / test_filename
        }
        
        # Check if any of the files exist
        found = False
        locations = []
        
        for location, file_path in test_files.items():
            if file_path.exists():
                found = True
                locations.append(location)
        
        results[model] = {
            "found": found,
            "locations": locations
        }
        
        if not found:
            missing_models.append(model)
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"key_models_report_{timestamp}.md"
    json_file = f"key_models_status_{timestamp}.json"
    
    # Write JSON status file
    with open(json_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_models": len(KEY_MODELS),
            "found_models": len(KEY_MODELS) - len(missing_models),
            "missing_models": len(missing_models),
            "coverage_percentage": (len(KEY_MODELS) - len(missing_models)) / len(KEY_MODELS) * 100,
            "results": results
        }, f, indent=2)
    
    # Write markdown report
    with open(report_file, "w") as f:
        f.write("# Key Model Test Implementation Report\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Key Models**: {len(KEY_MODELS)}\n")
        f.write(f"- **Models Found**: {len(KEY_MODELS) - len(missing_models)}\n")
        f.write(f"- **Models Missing**: {len(missing_models)}\n")
        f.write(f"- **Coverage**: {(len(KEY_MODELS) - len(missing_models)) / len(KEY_MODELS) * 100:.1f}%\n\n")
        
        f.write("## Key Model Status\n\n")
        f.write("| Model | Status | Locations |\n")
        f.write("|-------|--------|----------|\n")
        
        for model in KEY_MODELS:
            info = results[model]
            status = "✅ Found" if info["found"] else "❌ Missing"
            locations = ", ".join(info["locations"]) if info["found"] else "None"
            f.write(f"| {model} | {status} | {locations} |\n")
        
        if missing_models:
            f.write("\n## Missing Models\n\n")
            f.write("The following key models need implementation:\n\n")
            for model in missing_models:
                f.write(f"- {model}\n")
        
        f.write("\n## Next Steps\n\n")
        if missing_models:
            f.write("1. Create test implementations for the missing models using the merged test generator\n")
            f.write("2. Verify all test files are working correctly\n")
            f.write("3. Run performance benchmarks for each key model\n")
        else:
            f.write("All key models have test implementations! Next steps:\n")
            f.write("1. Verify all test files are working correctly\n")
            f.write("2. Run performance benchmarks for each key model\n")
            f.write("3. Ensure test compatibility across all hardware platforms\n")
    
    print(f"Report generated: {report_file}")
    print(f"Status JSON: {json_file}")
    
    return results, missing_models

if __name__ == "__main__":
    results, missing_models = check_key_model_tests()
    
    # Print summary to console
    print("\nKey Model Test Implementation Summary:")
    print(f"Total Key Models: {len(KEY_MODELS)}")
    print(f"Models Found: {len(KEY_MODELS) - len(missing_models)}")
    print(f"Models Missing: {len(missing_models)}")
    print(f"Coverage: {(len(KEY_MODELS) - len(missing_models)) / len(KEY_MODELS) * 100:.1f}%")
    
    if missing_models:
        print("\nMissing Models:")
        for model in missing_models:
            print(f"- {model}")
    else:
        print("\nAll key models have test implementations!")