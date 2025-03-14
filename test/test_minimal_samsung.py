#!/usr/bin/env python3
"""
Ultra-minimal Samsung NPU Support Test

This script tests only the core Samsung chipset and detector functionality
without any external dependencies. It's designed to verify the most basic
functionality of the Samsung NPU support.
"""

import os
import sys
from pathlib import Path

# Configure the path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Set simulation mode
os.environ["TEST_SAMSUNG_CHIPSET"] = "exynos_2400"

# Try to import the minimal required components
try:
    from samsung_support import SamsungChipset, SamsungChipsetRegistry, SamsungDetector
    print("Successfully imported Samsung NPU support module")
except ImportError as e:
    print(f"Error importing Samsung NPU support: {e}")
    sys.exit(1)

# Test SamsungChipset
def test_chipset():
    print("\n=== Testing SamsungChipset ===")
    chipset = SamsungChipset(
        name="Exynos 2400 (Test)",
        npu_cores=8,
        npu_tops=34.4,
        max_precision="FP16",
        supported_precisions=["FP32", "FP16", "BF16", "INT8", "INT4"],
        max_power_draw=8.5,
        typical_power=3.5
    )
    
    print(f"Created chipset: {chipset.name}")
    print(f"  NPU Cores: {chipset.npu_cores}")
    print(f"  NPU Performance: {chipset.npu_tops} TOPS")
    print(f"  Supported precisions: {chipset.supported_precisions}")
    
    # Test to_dict method
    chipset_dict = chipset.to_dict()
    print(f"  Dict conversion successful: {bool(chipset_dict)}")
    
    # Test from_dict method
    reconstructed = SamsungChipset.from_dict(chipset_dict)
    print(f"  Reconstruction successful: {reconstructed.name == chipset.name}")
    
    return True

# Test SamsungChipsetRegistry
def test_registry():
    print("\n=== Testing SamsungChipsetRegistry ===")
    registry = SamsungChipsetRegistry()
    
    chipsets = registry.get_all_chipsets()
    print(f"Registry contains {len(chipsets)} chipsets")
    
    # Test retrieval
    exynos_2400 = registry.get_chipset("exynos_2400")
    if exynos_2400:
        print(f"Found Exynos 2400: {exynos_2400.name}")
        print(f"  NPU Cores: {exynos_2400.npu_cores}")
        print(f"  NPU Performance: {exynos_2400.npu_tops} TOPS")
        return True
    else:
        print("Failed to find Exynos 2400 in registry")
        return False

# Test SamsungDetector
def test_detector():
    print("\n=== Testing SamsungDetector ===")
    detector = SamsungDetector()
    
    # Detect hardware
    chipset = detector.detect_samsung_hardware()
    
    if chipset:
        print(f"Samsung NPU detected: {chipset.name}")
        print(f"  NPU Cores: {chipset.npu_cores}")
        print(f"  NPU Performance: {chipset.npu_tops} TOPS")
        print(f"  Max Precision: {chipset.max_precision}")
        print(f"  Supported precisions: {', '.join(chipset.supported_precisions)}")
        
        # Test capability analysis
        try:
            analysis = detector.get_capability_analysis(chipset)
            print("\nCapability analysis:")
            print(f"  Power efficiency: {analysis['power_efficiency']['efficiency_rating']}")
            print(f"  TOPS per watt: {analysis['power_efficiency']['tops_per_watt']:.2f}")
            
            # Test model capabilities
            model_types = list(analysis["model_capabilities"].keys())
            print(f"  Analyzed model types: {', '.join(model_types[:3])}...")
            
            return True
        except Exception as e:
            print(f"Error in capability analysis: {e}")
            return False
    else:
        print("No Samsung NPU detected (this should not happen in simulation mode)")
        return False

# Run tests
def main():
    print("=== SAMSUNG NPU SUPPORT ULTRA-MINIMAL TEST ===")
    print(f"Simulation mode: TEST_SAMSUNG_CHIPSET={os.environ.get('TEST_SAMSUNG_CHIPSET')}")
    
    success = True
    
    # Run tests
    if not test_chipset():
        print("SamsungChipset test failed")
        success = False
        
    if not test_registry():
        print("SamsungChipsetRegistry test failed")
        success = False
        
    if not test_detector():
        print("SamsungDetector test failed")
        success = False
    
    # Print summary
    if success:
        print("\n=== ALL TESTS PASSED ===")
        return 0
    else:
        print("\n=== SOME TESTS FAILED ===")
        return 1

if __name__ == "__main__":
    sys.exit(main())