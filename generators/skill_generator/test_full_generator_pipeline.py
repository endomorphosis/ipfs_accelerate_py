#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the full test generation pipeline with specialized templates.

This script verifies that the test generator suite correctly handles
specialized model architectures and hardware platforms, with a focus on:
1. The three specialized templates: text-to-image, protein-folding, and video-processing
2. ROCm hardware support
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

# Import the test generator suite
from test_generator_suite import TestGeneratorSuite

def test_architecture_detection():
    """Test architecture detection for specialized templates."""
    from generators.architecture_detector import get_architecture_type
    
    # Test cases for each specialized architecture
    test_cases = {
        "text-to-image": [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "kandinsky-community/kandinsky-2-1"
        ],
        "protein-folding": [
            "facebook/esm2_t33_650M_UR50D",
            "facebook/esm1b_t33_650M_UR50S",
            "facebook/esmfold_v1"
        ],
        "video-processing": [
            "MCG-NJU/videomae-base-finetuned-kinetics",
            "google/vivit-b-16x2",
            "facebook/timesformer-base-finetuned-k400"
        ]
    }
    
    # Run tests and report results
    success_count = 0
    total_count = sum(len(models) for models in test_cases.values())
    
    print("\nTesting architecture detection for specialized templates:")
    for expected_arch, models in test_cases.items():
        print(f"\n{expected_arch.upper()} models:")
        for model in models:
            detected_arch = get_architecture_type(model)
            status = "✓" if detected_arch == expected_arch else "✗"
            print(f"  {status} {model:50} → {detected_arch}")
            if detected_arch == expected_arch:
                success_count += 1
    
    accuracy = (success_count / total_count) * 100 if total_count > 0 else 0
    print(f"\nAccuracy: {success_count}/{total_count} ({accuracy:.2f}%)")
    
    return success_count == total_count

def test_template_loading():
    """Test template loading for specialized templates."""
    generator = TestGeneratorSuite()
    
    # Check if specialized templates are loaded
    specialized_templates = [
        "text_to_image",
        "text_to_image_template",
        "protein_folding",
        "protein_folding_template",
        "video_processing",
        "video_processing_template"
    ]
    
    print("\nChecking template registry for specialized templates:")
    found_templates = []
    for template_name in specialized_templates:
        if template_name in generator.template_registry:
            print(f"  ✓ Found template: {template_name}")
            found_templates.append(template_name)
        else:
            print(f"  ✗ Missing template: {template_name}")
    
    expected_count = 3  # We expect at least one version of each specialized template
    success = len(set([t.split('_')[0] for t in found_templates])) >= expected_count
    
    print(f"\nFound {len(found_templates)} specialized templates (expected at least {expected_count})")
    return success

def test_template_selection():
    """Test template selection for specialized architectures."""
    generator = TestGeneratorSuite()
    
    # Test cases with model info
    test_cases = [
        {
            "model_type": "stable-diffusion-v1-5",
            "model_info": {
                "architecture": "text-to-image",
                "task": "text-to-image",
                "name": "stable-diffusion-v1-5"
            }
        },
        {
            "model_type": "esm2",
            "model_info": {
                "architecture": "protein-folding",
                "task": "fill-mask",
                "name": "esm2"
            }
        },
        {
            "model_type": "videomae",
            "model_info": {
                "architecture": "video-processing",
                "task": "video-classification",
                "name": "videomae"
            }
        }
    ]
    
    print("\nTesting template selection for specialized architectures:")
    success_count = 0
    
    for test_case in test_cases:
        model_type = test_case["model_type"]
        model_info = test_case["model_info"]
        
        template = generator.get_template_for_model(model_info)
        
        # Check if template contains the expected architecture type
        expected_arch = model_info["architecture"].replace("-", "_")
        if template and (expected_arch in template or expected_arch.split("_")[0] in template):
            print(f"  ✓ {model_type:20} → Selected appropriate template ({expected_arch})")
            success_count += 1
        else:
            print(f"  ✗ {model_type:20} → Failed to select appropriate template")
            print(f"      Expected: {expected_arch}")
            print(f"      Template contains: {template[:100]}...")
    
    print(f"\nSuccessfully selected {success_count}/{len(test_cases)} templates")
    return success_count == len(test_cases)

def test_hardware_detection():
    """Test hardware detection especially for ROCm."""
    generator = TestGeneratorSuite()
    hardware_info = generator.detect_hardware()
    
    print("\nDetected hardware:")
    for hw_name, hw_props in hardware_info.items():
        status = "Available" if hw_props.get("available", False) else "Not available"
        device_name = hw_props.get("name", "Unknown")
        print(f"  {hw_name:10}: {status:15} - {device_name}")
    
    # Check if ROCm detection is included
    if "rocm" in hardware_info:
        print("\nROCm hardware detection is implemented!")
        return True
    else:
        print("\nFailed to find ROCm hardware detection!")
        return False

def test_full_generation():
    """Test full test generation for specialized architectures."""
    generator = TestGeneratorSuite()
    output_dir = Path("./generated_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Test models for specialized architectures
    test_models = [
        "stable-diffusion",  # text-to-image
        "esm",               # protein-folding
        "videomae"           # video-processing
    ]
    
    print("\nGenerating test files for specialized architectures:")
    results = []
    for model_type in test_models:
        output_path = output_dir / f"test_hf_{model_type.replace('-', '_')}.py"
        print(f"Generating test for {model_type} → {output_path}")
        
        result = generator.generate_test(model_type, str(output_path))
        results.append(result)
        
        if result["success"]:
            print(f"  ✓ Successfully generated test for {model_type}")
            # Check if the file exists and contains the expected architecture
            if output_path.exists():
                with open(output_path, 'r') as f:
                    content = f.read()
                architecture = result.get("architecture", "unknown")
                if architecture.replace("-", "_") in content or architecture.split("-")[0] in content:
                    print(f"    ✓ File contains expected architecture type: {architecture}")
                else:
                    print(f"    ✗ File does not contain expected architecture type: {architecture}")
                    result["success"] = False
        else:
            print(f"  ✗ Failed to generate test for {model_type}: {result.get('error', 'Unknown error')}")
    
    success_count = sum(1 for result in results if result["success"])
    print(f"\nSuccessfully generated {success_count}/{len(test_models)} test files")
    
    return success_count == len(test_models)

def run_all_tests():
    """Run all test functions and report overall results."""
    test_functions = [
        test_architecture_detection,
        test_template_loading,
        test_template_selection,
        test_hardware_detection,
        test_full_generation
    ]
    
    print("=" * 80)
    print("Running tests for specialized templates and ROCm support")
    print("=" * 80)
    
    results = []
    for i, test_func in enumerate(test_functions, 1):
        print(f"\n{i}. Running: {test_func.__name__}")
        print("-" * 80)
        try:
            success = test_func()
            results.append(success)
            status = "PASSED" if success else "FAILED"
            print(f"\nTest {i} {status}")
        except Exception as e:
            results.append(False)
            print(f"\nTest {i} ERROR: {e}")
    
    # Overall results
    success_count = sum(results)
    total_count = len(results)
    
    print("\n" + "=" * 80)
    print(f"OVERALL RESULTS: {success_count}/{total_count} tests passed")
    print("=" * 80)
    
    for i, (test_func, success) in enumerate(zip(test_functions, results), 1):
        status = "PASSED" if success else "FAILED"
        print(f"{i}. {test_func.__name__:30} - {status}")
    
    return success_count == total_count

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)