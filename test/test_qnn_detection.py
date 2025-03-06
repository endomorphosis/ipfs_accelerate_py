#!/usr/bin/env python
"""
Test the QNN detection in the hardware_detection module.
"""

import os
import sys
from pathlib import Path

# Add the repository root to the Python path
sys.path.append(str(Path(__file__).parent))

from hardware_detection import HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_QNN, HAS_WEBNN, HAS_WEBGPU
from hardware_detection.capabilities import detect_all_hardware, detect_qnn

def main():
    """Test QNN detection"""
    print("Hardware detection flags:")
    print(f"HAS_CUDA = {HAS_CUDA}")
    print(f"HAS_ROCM = {HAS_ROCM}")
    print(f"HAS_OPENVINO = {HAS_OPENVINO}")
    print(f"HAS_MPS = {HAS_MPS}")
    print(f"HAS_QNN = {HAS_QNN}")
    print(f"HAS_WEBNN = {HAS_WEBNN}")
    print(f"HAS_WEBGPU = {HAS_WEBGPU}")
    
    print("\nQNN detection results:")
    qnn_results = detect_qnn()
    
    for key, value in qnn_results.items():
        if key != "detailed_info":  # Skip detailed info for brevity
            print(f"  {key}: {value}")
    
    if qnn_results["detected"]:
        print("\nQNN is detected!")
        if qnn_results.get("detailed_info"):
            info = qnn_results["detailed_info"]
            print(f"  Device: {info.get('device_name', 'Unknown')}")
            print(f"  Memory: {info.get('memory_mb', 'Unknown')} MB")
            print(f"  Precision: {', '.join(info.get('precision_support', []))}")
            print(f"  SDK Version: {info.get('sdk_version', 'Unknown')}")
            print(f"  Recommended Models: {len(info.get('recommended_models', []))} models")
    else:
        print("\nQNN is not detected.")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())