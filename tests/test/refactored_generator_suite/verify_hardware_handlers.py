#!/usr/bin/env python3
"""
Verify that all generated models have the required hardware handlers.

This script checks that all generated model implementations in the generated_skillsets
directory have handlers for the following hardware backends:
- CPU
- CUDA
- ROCm
- OpenVINO
- MPS (Apple)
- QNN (Qualcomm)
"""

import os
import glob
import re
import json
from collections import defaultdict

# Hardware backends to check for
REQUIRED_HANDLERS = [
    "init_cpu",
    "init_cuda",
    "init_openvino",
    "init_apple",
    "init_qualcomm",
    "init_rocm",
    "create_cpu_.*_endpoint_handler",
    "create_cuda_.*_endpoint_handler",
    "create_openvino_.*_endpoint_handler",
    "create_apple_.*_endpoint_handler",
    "create_qualcomm_.*_endpoint_handler",
    "create_rocm_.*_endpoint_handler",
]

def check_file(file_path):
    """Check if a file contains all required hardware handlers."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    results = {}
    for handler in REQUIRED_HANDLERS:
        pattern = r'def\s+' + handler
        match = re.search(pattern, content)
        results[handler] = match is not None
    
    return results

def main():
    """Main entry point."""
    # Find all generated model files
    model_files = glob.glob('./generated_skillsets/hf_*.py')
    
    # Skip summary file
    model_files = [f for f in model_files if 'summary' not in f]
    
    print(f"Found {len(model_files)} model files to check")
    
    # Check each file
    results = {}
    architectures = defaultdict(int)
    
    for file_path in model_files:
        model_name = os.path.basename(file_path).replace('hf_', '').replace('.py', '')
        results[model_name] = check_file(file_path)
        
        # Try to extract architecture
        with open(file_path, 'r') as f:
            content = f.read()
            arch_match = re.search(r'architecture_type\s*=\s*[\'"]([^\'"]+)[\'"]', content)
            if arch_match:
                architectures[arch_match.group(1)] += 1
    
    # Calculate statistics
    total_models = len(results)
    handler_stats = defaultdict(int)
    
    for model, handlers in results.items():
        for handler, has_handler in handlers.items():
            if has_handler:
                handler_stats[handler] += 1
    
    # Print summary
    print("\n=== HARDWARE HANDLER COVERAGE SUMMARY ===")
    print(f"Total models checked: {total_models}")
    print("\nHardware handler coverage:")
    
    for handler in REQUIRED_HANDLERS:
        count = handler_stats[handler]
        percentage = (count / total_models) * 100
        print(f"  {handler}: {count}/{total_models} ({percentage:.2f}%)")
    
    print("\nArchitecture distribution:")
    for arch, count in architectures.items():
        print(f"  {arch}: {count}")
    
    # Create detailed report
    report = {
        "total_models": total_models,
        "handler_stats": {h: handler_stats[h] for h in REQUIRED_HANDLERS},
        "architectures": dict(architectures),
        "model_details": results
    }
    
    with open('./hardware_coverage_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nDetailed report written to hardware_coverage_report.json")

if __name__ == "__main__":
    main()