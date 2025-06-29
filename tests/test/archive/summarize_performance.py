#!/usr/bin/env python3
"""
Script to summarize performance results for all tested models.
"""

import os
import json

def summarize_performance():
    """
    Read all performance test results and print a summary table.
    """
    performance_dir = os.path.join(os.path.dirname(__file__), "performance_results")
    files = os.listdir(performance_dir)
    
    print("\nPERFORMANCE SUMMARY FOR TESTED MODELS:")
    print("-" * 80)
    print(f"{'Model':<15} | {'Exec Time':<12} | {'CPU':<10} | {'CUDA':<10} | {'OpenVINO':<10}")
    print("-" * 80)
    
    for file in files:
        if file.endswith('.json'):
            with open(os.path.join(performance_dir, file), 'r') as f:
                data = json.load(f)
                
                # Extract model name and execution time
                model = data.get('metadata', {}).get('model', 'unknown')
                exec_time = data.get('metadata', {}).get('execution_time', 0)
                
                # Extract implementation types for each platform
                platforms = {}
                results = data.get('results', {})
                
                # Look for CPU implementation type
                for platform in ['cpu', 'cuda', 'openvino']:
                    impl = 'Unknown'
                    
                    # Check for direct platform status keys first
                    if platform + "_init" in results and isinstance(results[platform + "_init"], str):
                        if "Success" in results[platform + "_init"] and "(" in results[platform + "_init"] and ")" in results[platform + "_init"]:
                            impl = results[platform + "_init"].split("(")[1].split(")")[0]
                    
                    # If not found, check examples for implementation type
                    if impl == 'Unknown' and 'examples' in results:
                        for example in results['examples']:
                            if example.get('platform', '').lower() == platform.lower():
                                example_impl = example.get('implementation_type', '')
                                if example_impl and example_impl.startswith('(') and example_impl.endswith(')'):
                                    impl = example_impl[1:-1]
                                    break
                    
                    # Check metadata if still not found
                    if impl == 'Unknown' and 'metadata' in results and 'platform_status' in results['metadata']:
                        platform_status = results['metadata']['platform_status']
                        if platform in platform_status:
                            status = platform_status[platform]
                            if "REAL" in status:
                                impl = "REAL"
                            elif "MOCK" in status:
                                impl = "MOCK"
                                
                    platforms[platform] = impl
                
                print(f"{model:<15} | {exec_time:<12.2f}s | {platforms.get('cpu', 'Unknown'):<10} | {platforms.get('cuda', 'Unknown'):<10} | {platforms.get('openvino', 'Unknown'):<10}")
    
    print("-" * 80)

if __name__ == "__main__":
    summarize_performance()