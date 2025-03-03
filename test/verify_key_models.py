#!/usr/bin/env python3
"""
Script to verify the implementation of 13 key model tests.

This script checks for the existence of test files for key models
and optionally validates their web platform implementation types.
"""

import os
import sys
import json
import argparse
import re
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
KEY_MODELS_HARDWARE_FIXES_DIR = Path("./key_models_hardware_fixes")

def check_implementation_type(file_path, platform):
    """
    Check if the implementation type in the file matches the expected platform.
    
    Args:
        file_path: Path to the test file
        platform: Platform to check (webnn or webgpu)
        
    Returns:
        Tuple of (has_implementation, implementation_type)
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for expected implementation type
        if platform.lower() == "webnn":
            # Check for REAL_WEBNN implementation type
            if "REAL_WEBNN" in content:
                return True, "REAL_WEBNN"
            elif "MOCK_WEBNN" in content:
                return True, "MOCK_WEBNN"
            elif "SIMULATED_WEBNN" in content:
                return True, "SIMULATED_WEBNN"
        elif platform.lower() == "webgpu":
            # Check for REAL_WEBGPU implementation type
            if "REAL_WEBGPU" in content:
                return True, "REAL_WEBGPU"
            elif "MOCK_WEBGPU" in content:
                return True, "MOCK_WEBGPU"
            elif "SIMULATED_WEBGPU" in content:
                return True, "SIMULATED_WEBGPU"
        
        # Check if file has a create_[platform]_handler method
        platform_handler_pattern = rf"def\s+create_{platform.lower()}_handler"
        if re.search(platform_handler_pattern, content):
            return True, "HANDLER_FOUND"
        
        return False, "NOT_FOUND"
    except Exception as e:
        print(f"Error checking implementation type in {file_path}: {e}")
        return False, f"ERROR: {str(e)}"

def check_key_model_tests(platform=None):
    """
    Check if test files for key models exist and report their status.
    
    Args:
        platform: Optional platform to check implementation types for
    """
    results = {}
    missing_models = []
    
    for model in KEY_MODELS:
        # Construct potential filenames
        test_filename = f"test_hf_{model}.py"
        test_files = {
            "skills": SKILLS_DIR / test_filename,
            "modality_tests": MODALITY_TESTS_DIR / test_filename,
            "fixed_generated_tests": FIXED_GENERATED_TESTS_DIR / test_filename,
            "new_generated_tests": NEW_GENERATED_TESTS_DIR / test_filename,
            "key_models_hardware_fixes": KEY_MODELS_HARDWARE_FIXES_DIR / test_filename
        }
        
        # Check if any of the files exist
        found = False
        locations = []
        implementation_types = {}
        
        for location, file_path in test_files.items():
            if file_path.exists():
                found = True
                locations.append(location)
                
                # Check implementation type if platform is specified
                if platform:
                    has_impl, impl_type = check_implementation_type(file_path, platform)
                    implementation_types[location] = impl_type
        
        results[model] = {
            "found": found,
            "locations": locations,
        }
        
        # Add implementation types if platform was specified
        if platform:
            results[model]["implementation_types"] = implementation_types
            results[model]["has_implementation"] = any(
                impl_type.startswith("REAL") for impl_type in implementation_types.values()
            )
        
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
            "platform": platform,
            "results": results
        }, f, indent=2)
    
    # Write markdown report
    with open(report_file, "w") as f:
        f.write("# Key Model Test Implementation Report\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        if platform:
            f.write(f"**Platform:** {platform.upper()}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Key Models**: {len(KEY_MODELS)}\n")
        f.write(f"- **Models Found**: {len(KEY_MODELS) - len(missing_models)}\n")
        f.write(f"- **Models Missing**: {len(missing_models)}\n")
        f.write(f"- **Coverage**: {(len(KEY_MODELS) - len(missing_models)) / len(KEY_MODELS) * 100:.1f}%\n\n")
        
        f.write("## Key Model Status\n\n")
        
        if platform:
            f.write("| Model | Status | Locations | Implementation |\n")
            f.write("|-------|--------|-----------|----------------|\n")
            
            for model in KEY_MODELS:
                info = results[model]
                status = "✅ Found" if info["found"] else "❌ Missing"
                locations = ", ".join(info["locations"]) if info["found"] else "None"
                
                # Format implementation types
                if info["found"] and platform:
                    impl_types = []
                    for location, impl_type in info.get("implementation_types", {}).items():
                        if impl_type.startswith("REAL"):
                            impl_types.append(f"✅ {impl_type}")
                        elif impl_type.startswith("MOCK") or impl_type.startswith("SIMULATED"):
                            impl_types.append(f"⚠️ {impl_type}")
                        elif impl_type == "HANDLER_FOUND":
                            impl_types.append(f"ℹ️ Handler found")
                        else:
                            impl_types.append(f"❌ {impl_type}")
                    
                    implementation = "<br>".join(impl_types) if impl_types else "❌ Not implemented"
                else:
                    implementation = "N/A"
                
                f.write(f"| {model} | {status} | {locations} | {implementation} |\n")
        else:
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
        
        if platform:
            f.write("\n## Platform Implementation Summary\n\n")
            implemented_count = sum(1 for info in results.values() 
                                   if info["found"] and info.get("has_implementation", False))
            f.write(f"- **Models with {platform.upper()} Implementation**: {implemented_count}/{len(KEY_MODELS)}\n")
            f.write(f"- **Implementation Coverage**: {implemented_count/len(KEY_MODELS)*100:.1f}%\n\n")
            
            f.write("### Models With Proper Implementation\n\n")
            for model in KEY_MODELS:
                info = results[model]
                if info["found"] and info.get("has_implementation", False):
                    f.write(f"- {model}\n")
            
            f.write("\n### Models Needing Implementation\n\n")
            for model in KEY_MODELS:
                info = results[model]
                if info["found"] and not info.get("has_implementation", False):
                    f.write(f"- {model} (Found in: {', '.join(info['locations'])})\n")
        
        f.write("\n## Next Steps\n\n")
        if missing_models:
            f.write("1. Create test implementations for the missing models using the merged test generator\n")
            f.write("2. Verify all test files are working correctly\n")
            f.write("3. Run performance benchmarks for each key model\n")
        elif platform and implemented_count < len(KEY_MODELS):
            f.write(f"1. Add {platform.upper()} implementation to all key models\n")
            f.write(f"2. Fix implementation types to consistently use REAL_{platform.upper()}\n")
            f.write("3. Run the fix_test_files.py script to standardize implementation types\n")
        else:
            f.write("All key models have test implementations! Next steps:\n")
            f.write("1. Verify all test files are working correctly\n")
            f.write("2. Run performance benchmarks for each key model\n")
            f.write("3. Ensure test compatibility across all hardware platforms\n")
    
    print(f"Report generated: {report_file}")
    print(f"Status JSON: {json_file}")
    
    return results, missing_models

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Verify key model implementations")
    parser.add_argument("--platform", choices=["webnn", "webgpu"], 
                        help="Check implementation types for a specific platform")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    results, missing_models = check_key_model_tests(args.platform)
    
    # Print summary to console
    print("\nKey Model Test Implementation Summary:")
    print(f"Total Key Models: {len(KEY_MODELS)}")
    print(f"Models Found: {len(KEY_MODELS) - len(missing_models)}")
    print(f"Models Missing: {len(missing_models)}")
    print(f"Coverage: {(len(KEY_MODELS) - len(missing_models)) / len(KEY_MODELS) * 100:.1f}%")
    
    if args.platform:
        implemented_count = sum(1 for info in results.values() 
                               if info["found"] and info.get("has_implementation", False))
        print(f"\n{args.platform.upper()} Implementation Coverage: {implemented_count}/{len(KEY_MODELS)} ({implemented_count/len(KEY_MODELS)*100:.1f}%)")
    
    if missing_models:
        print("\nMissing Models:")
        for model in missing_models:
            print(f"- {model}")
    else:
        print("\nAll key models have test implementations!")