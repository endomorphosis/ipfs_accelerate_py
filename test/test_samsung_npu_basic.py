#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Samsung NPU Support Test

This script provides a simple test of the Samsung NPU support functionality.
It allows testing both the standalone Samsung NPU detection/simulation and
the integration with the centralized hardware detection system.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

# Import Samsung support
try:
    from samsung_support import (
        SamsungDetector,
        SamsungChipset,
        SamsungBenchmarkRunner,
        SamsungModelConverter,
        SAMSUNG_NPU_AVAILABLE,
        SAMSUNG_NPU_SIMULATION_MODE
    )
    print(f"Samsung NPU support loaded: Available={SAMSUNG_NPU_AVAILABLE}, Simulation={SAMSUNG_NPU_SIMULATION_MODE}")
except ImportError:
    print("Could not import Samsung NPU support module")
    sys.exit(1)

# Try to import centralized hardware detection
try:
    from centralized_hardware_detection.hardware_detection import (
        HardwareManager
    )
    CENTRALIZED_HARDWARE_AVAILABLE = True
    print("Centralized hardware detection available")
except (ImportError, ModuleNotFoundError):
    CENTRALIZED_HARDWARE_AVAILABLE = False
    print("Centralized hardware detection not available")


def test_standalone_detection(simulation=False):
    """Test standalone Samsung NPU detection."""
    print("\n=== Testing standalone Samsung NPU detection ===")
    
    # Enable simulation if requested
    if simulation:
        os.environ["TEST_SAMSUNG_CHIPSET"] = "exynos_2400"
        print("Enabled Samsung NPU simulation mode")
    
    # Create detector
    detector = SamsungDetector()
    
    # Detect hardware
    chipset = detector.detect_samsung_hardware()
    
    if chipset:
        print(f"Samsung NPU detected: {chipset.name}")
        print(f"  NPU Cores: {chipset.npu_cores}")
        print(f"  NPU Performance: {chipset.npu_tops} TOPS")
        print(f"  Supported precisions: {chipset.supported_precisions}")
        
        # Get capability analysis
        analysis = detector.get_capability_analysis(chipset)
        
        # Display model capability summary
        print("\nModel Capability Summary:")
        for model_type, capability in analysis["model_capabilities"].items():
            print(f"  {model_type}: {'Suitable' if capability['suitable'] else 'Not suitable'}, "
                  f"Max size: {capability['max_size']}, Performance: {capability['performance']}")
        
        # Display competitive position
        competitive = analysis["competitive_position"]
        print("\nCompetitive Position:")
        print(f"  vs Qualcomm: {competitive['vs_qualcomm']}")
        print(f"  vs MediaTek: {competitive['vs_mediatek']}")
        print(f"  vs Apple: {competitive['vs_apple']}")
        print(f"  Overall: {competitive['overall_ranking']}")
        
        # Display top 3 recommendations
        print("\nTop Recommendations:")
        for i, recommendation in enumerate(analysis["recommended_optimizations"][:3], 1):
            print(f"  {i}. {recommendation}")
    else:
        print("No Samsung NPU detected")


def test_centralized_detection(simulation=False):
    """Test centralized hardware detection with Samsung NPU."""
    if not CENTRALIZED_HARDWARE_AVAILABLE:
        print("\n=== Skipping centralized hardware detection test (not available) ===")
        return
    
    print("\n=== Testing centralized hardware detection with Samsung NPU ===")
    
    # Enable simulation if requested
    if simulation:
        os.environ["TEST_SAMSUNG_CHIPSET"] = "exynos_2400"
        print("Enabled Samsung NPU simulation mode")
    
    # Create hardware manager
    hardware_manager = HardwareManager()
    
    # Get capabilities
    capabilities = hardware_manager.get_capabilities()
    
    # Check for Samsung NPU
    if capabilities.get("samsung_npu", False):
        print("Samsung NPU detected via centralized hardware detection")
        print(f"Simulation mode: {capabilities.get('samsung_npu_simulation', False)}")
        
        # Check model compatibility for different model types
        model_types = ["bert-base-uncased", "vit-base-patch16-224", "whisper-tiny", "llama-7b"]
        
        print("\nModel compatibility by model type:")
        for model_type in model_types:
            compatibility = hardware_manager.get_model_hardware_compatibility(model_type)
            print(f"  {model_type}: Samsung NPU compatible = {compatibility.get('samsung_npu', False)}")
    else:
        print("No Samsung NPU detected via centralized hardware detection")


def test_model_compatibility(simulation=False):
    """Test model compatibility analysis."""
    print("\n=== Testing model compatibility analysis ===")
    
    # Enable simulation if requested
    if simulation:
        os.environ["TEST_SAMSUNG_CHIPSET"] = "exynos_2400"
        print("Enabled Samsung NPU simulation mode")
    
    # Create model converter
    converter = SamsungModelConverter()
    
    # Create a temporary model file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        model_path = tmp.name
    
    try:
        # Test model compatibility for different chipsets
        chipsets = ["exynos_2400", "exynos_2200", "exynos_1380", "exynos_850"]
        
        for chipset_name in chipsets:
            print(f"\nAnalyzing compatibility for {chipset_name}:")
            result = converter.analyze_model_compatibility(model_path, chipset_name)
            
            # Print compatibility summary
            compatibility = result["compatibility"]
            chipset_info = result["chipset_info"]
            
            print(f"  Chipset: {chipset_info['name']} ({chipset_info['npu_tops']} TOPS)")
            print(f"  Supported: {compatibility['supported']}")
            print(f"  Recommended precision: {compatibility['recommended_precision']}")
            
            # Estimated performance
            perf = compatibility["estimated_performance"]
            print(f"  Estimated latency: {perf['latency_ms']:.1f} ms")
            print(f"  Estimated throughput: {perf['throughput_items_per_second']:.1f} items/sec")
            
            # Issues
            issues = compatibility["potential_issues"]
            if issues and issues[0] != "No significant issues detected":
                print(f"  Potential issues: {issues[0]}")
            
            # Top optimizations
            opts = compatibility["optimization_opportunities"][:2]
            if opts:
                print(f"  Top optimizations: {', '.join(opts)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(model_path):
            os.unlink(model_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Basic Samsung NPU Support Test")
    parser.add_argument("--simulation", action="store_true", help="Enable simulation mode")
    parser.add_argument("--standalone", action="store_true", help="Test standalone detection only")
    parser.add_argument("--centralized", action="store_true", help="Test centralized detection only")
    parser.add_argument("--compatibility", action="store_true", help="Test model compatibility only")
    
    args = parser.parse_args()
    
    # If no specific tests selected, run all tests
    run_all = not (args.standalone or args.centralized or args.compatibility)
    
    if run_all or args.standalone:
        test_standalone_detection(args.simulation)
    
    if run_all or args.centralized:
        test_centralized_detection(args.simulation)
    
    if run_all or args.compatibility:
        test_model_compatibility(args.simulation)