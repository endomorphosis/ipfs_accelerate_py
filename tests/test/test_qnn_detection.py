#!/usr/bin/env python
"""
Test the QNN detection in the hardware_detection module.
"""

import os
import sys
from pathlib import Path

# Add the repository root to the Python path
sys.path.append(str(Path(__file__).parent))

from generators.hardware.hardware_detection import HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_QNN, HAS_WEBNN, HAS_WEBGPU
from hardware_detection.capabilities import detect_all_hardware, detect_qnn

def main():
    """Test QNN detection"""
    print("Hardware detection flags:")
    print(f"\1{HAS_CUDA}\3")
    print(f"\1{HAS_ROCM}\3")
    print(f"\1{HAS_OPENVINO}\3")
    print(f"\1{HAS_MPS}\3")
    print(f"\1{HAS_QNN}\3")
    print(f"\1{HAS_WEBNN}\3")
    print(f"\1{HAS_WEBGPU}\3")
    
    print("\nQNN detection results:")
    qnn_results = detect_qnn()
    
    for key, value in qnn_results.items():
        if key != "detailed_info":  # Skip detailed info for brevity
        print(f"\1{value}\3")
    
        if qnn_results["detected"]:,
        print("\nQNN is detected!")
        if qnn_results.get("detailed_info"):
            info = qnn_results["detailed_info"],
            print(f"\1{info.get('device_name', 'Unknown')}\3")
            print(f"  Memory: {info.get('memory_mb', 'Unknown')} MB")
            print(f"\1{', '.join(info.get('precision_support', []))}\3"),
            print(f"\1{info.get('sdk_version', 'Unknown')}\3")
            print(f"  Recommended Models: {len(info.get('recommended_models', []))} models"),
    else:
        print("\nQNN is not detected.")
        
            return 0

if __name__ == "__main__":
    sys.exit(main())