#!/usr/bin/env python
"""
Simple test for enhanced hardware detection with simulation detection.

This script tests the hardware_detection_updates.py module to verify
that it correctly detects and reports simulated hardware platforms.
"""

import os
import sys
import json
import logging
from pprint import pprint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our enhanced hardware detection
try:
    from hardware_detection_updates import detect_hardware_with_simulation_check
except ImportError as e:
    logger.error(f"Failed to import scripts.generators.hardware.hardware_detection as hardware_detection_updates: {str(e)}")
    sys.exit(1)

def main():
    """Run the enhanced hardware detection test"""
    print("Testing enhanced hardware detection with simulation detection...\n")
    
    # Test without simulation
    print("Running hardware detection without simulation...")
    hardware_info = detect_hardware_with_simulation_check()
    
    # Display available hardware
    print("\nAvailable hardware:")
    for hw_type, available in hardware_info["hardware"].items():
        if available:
            simulated = hw_type in hardware_info.get("simulated_hardware", [])
            if simulated:
                print(f"  {hw_type}: Available (SIMULATED)")
            else:
                print(f"  {hw_type}: Available")
        else:
            print(f"  {hw_type}: Not available")
    
    # Display simulation status
    simulated_hardware = hardware_info.get("simulated_hardware", [])
    if simulated_hardware:
        print("\nSimulated hardware:")
        for hw_type in simulated_hardware:
            print(f"  {hw_type}: Simulated")
            
        # Display simulation warning
        if "simulation_warning" in hardware_info:
            print(f"\nWarning: {hardware_info['simulation_warning']}")
    else:
        print("\nNo hardware is being simulated.")
    
    # Test with simulation
    print("\n\nTesting with WEBGPU_SIMULATION=1...\n")
    
    # Set simulation environment variable
    os.environ["WEBGPU_SIMULATION"] = "1"
    
    # Run detection again
    hardware_info = detect_hardware_with_simulation_check()
    
    # Display available hardware
    print("\nAvailable hardware with WEBGPU_SIMULATION=1:")
    for hw_type, available in hardware_info["hardware"].items():
        if available:
            simulated = hw_type in hardware_info.get("simulated_hardware", [])
            if simulated:
                print(f"  {hw_type}: Available (SIMULATED)")
            else:
                print(f"  {hw_type}: Available")
        else:
            print(f"  {hw_type}: Not available")
    
    # Display simulation status
    simulated_hardware = hardware_info.get("simulated_hardware", [])
    if simulated_hardware:
        print("\nSimulated hardware:")
        for hw_type in simulated_hardware:
            print(f"  {hw_type}: Simulated")
            
        # Display simulation warning
        if "simulation_warning" in hardware_info:
            print(f"\nWarning: {hardware_info['simulation_warning']}")
    else:
        print("\nNo hardware is being simulated.")
    
    # Reset environment variable
    os.environ.pop("WEBGPU_SIMULATION", None)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())