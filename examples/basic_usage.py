#!/usr/bin/env python3
"""
Basic Usage Example for IPFS Accelerate Python

This example demonstrates the core functionality of the IPFS Accelerate Python
framework, including hardware detection, model optimization recommendations,
and basic usage patterns.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Demonstrate basic usage of IPFS Accelerate Python."""

    print("🚀 IPFS Accelerate Python - Basic Usage Example")
    print("=" * 60)

    # Step 1: Import and detect hardware
    print("\n🔍 Step 1: Hardware Detection")
    try:
        from hardware_detection import HardwareDetector
        from utils.safe_imports import print_dependency_status

        # Create detector instance
        detector = HardwareDetector()

        # Get all available hardware
        available = detector.get_available_hardware()
        print(f"📊 Available hardware: {list(available.keys())}")

        # Show which hardware is actually available
        for hw_type, is_available in available.items():
            icon = "✅" if is_available else "❌"
            print(f"  {icon} {hw_type.upper()}")

        # Get best hardware for general use
        best_hardware = detector.get_best_available_hardware()
        print(f"🎯 Best available hardware: {best_hardware}")

    except ImportError as e:
        print(f"❌ Error importing hardware detection: {e}")
        print("💡 Tip: Make sure you're running from the project root directory")
        return False

    # Step 2: Model compatibility checking
    print("\n🤖 Step 2: Model Compatibility Checking")
    try:
        from utils.model_compatibility import get_optimal_hardware, check_model_compatibility

        # List of example models to test
        test_models = ["bert-base-uncased", "gpt2", "whisper-base", "clip-vit-base-patch32"]

        available_hardware_list = [hw for hw, avail in available.items() if avail]

        print(f"🧪 Testing models with available hardware: {available_hardware_list}")

        for model_name in test_models:
            print(f"\n  📝 Testing: {model_name}")

            # Get optimal hardware recommendation
            recommendation = get_optimal_hardware(model_name, available_hardware_list)

            recommended_hw = recommendation.get("recommended_hardware", "unknown")
            confidence = recommendation.get("confidence", "unknown")
            performance = recommendation.get("performance_multiplier", 1.0)

            print(f"    🎯 Recommended: {recommended_hw} (confidence: {confidence})")
            print(f"    ⚡ Performance boost: {performance:.1f}x over CPU")

            # Check memory requirements
            memory_req = recommendation.get("memory_requirements", {})
            if memory_req:
                min_mem = memory_req.get("minimum_gb", "unknown")
                rec_mem = memory_req.get("recommended_gb", "unknown")
                print(f"    💾 Memory: {min_mem}GB min, {rec_mem}GB recommended")

            # Show alternatives
            alternatives = recommendation.get("alternatives", [])
            if alternatives:
                print(f"    🔄 Alternatives: {', '.join(alternatives)}")

    except ImportError as e:
        print(f"❌ Error importing model compatibility: {e}")
        print("💡 Model compatibility features may not be available")

    # Step 3: Dependency status
    print("\n📦 Step 3: Dependency Status")
    try:
        print_dependency_status()
    except Exception as e:
        print(f"❌ Error checking dependencies: {e}")

    # Step 4: Hardware benchmarking (if available)
    print("\n⚡ Step 4: Performance Information")
    try:
        # Get all hardware details
        all_details = detector.get_hardware_details()

        for hw_type in available_hardware_list:
            try:
                details = all_details.get(hw_type, {})
                if details and details.get("available"):
                    print(f"  🖥️  {hw_type.upper()}:")

                    # Show relevant details
                    if "device_count" in details:
                        print(f"    - Device count: {details['device_count']}")
                    if "memory_total_gb" in details:
                        print(f"    - Memory: {details['memory_total_gb']:.1f}GB")
                    if "compute_capability" in details:
                        print(f"    - Compute capability: {details['compute_capability']}")
                    if "version" in details:
                        print(f"    - Version: {details['version']}")

            except Exception as e:
                print(f"  ⚠️  {hw_type}: Error getting details - {e}")

    except Exception as e:
        print(f"❌ Error getting performance info: {e}")

    # Step 5: Practical usage tips
    print("\n💡 Step 5: Usage Tips")
    print("Based on your system, here are some recommendations:")

    # Give specific advice based on detected hardware
    if available.get("cuda", False):
        print("  🚀 You have CUDA support! Great for:")
        print("    - Large language models (GPT, LLaMA)")
        print("    - Image processing (Vision Transformers)")
        print("    - Fast training and fine-tuning")

    elif available.get("mps", False):
        print("  🍎 You have Apple Silicon MPS! Excellent for:")
        print("    - Efficient inference with unified memory")
        print("    - Good balance of performance and power efficiency")
        print("    - Most transformer models work well")

    elif available.get("webnn", False) or available.get("webgpu", False):
        print("  🌐 You have web acceleration support! Good for:")
        print("    - Browser-based ML applications")
        print("    - Lightweight models (BERT, small Vision Transformers)")
        print("    - Real-time applications")

    else:
        print("  🖥️  CPU-only setup detected. Best for:")
        print("    - Small to medium models")
        print("    - Development and testing")
        print("    - Models optimized for CPU (quantized versions)")

    print("\n🎯 Next Steps:")
    print("  1. Try running the comprehensive test suite: python run_all_tests.py")
    print("  2. Explore other examples in the examples/ directory")
    print("  3. Check out the documentation in TESTING_README.md")
    print("  4. Consider the model compatibility guide for optimal performance")

    print("\n✅ Basic usage example completed successfully!")
    return True


def quick_hardware_test():
    """Quick hardware functionality test."""
    print("\n🧪 Quick Hardware Test")
    print("-" * 30)

    try:
        from hardware_detection import HardwareDetector

        detector = HardwareDetector()

        # Test basic detection
        cpu_available = detector.is_available("cpu")
        print(f"CPU available: {'✅' if cpu_available else '❌'}")

        # Test GPU detection
        gpu_types = ["cuda", "rocm", "mps"]
        gpu_found = False
        for gpu_type in gpu_types:
            if detector.is_available(gpu_type):
                print(f"{gpu_type.upper()} available: ✅")
                gpu_found = True
                break

        if not gpu_found:
            print("GPU acceleration: ❌ (CPU only)")

        # Test web acceleration
        web_types = ["webnn", "webgpu"]
        web_available = any(detector.is_available(web_type) for web_type in web_types)
        print(f"Web acceleration: {'✅' if web_available else '❌'}")

        print("🎉 Hardware test completed!")
        return True

    except Exception as e:
        print(f"❌ Hardware test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IPFS Accelerate Python Basic Usage Example")
    parser.add_argument("--quick", action="store_true", help="Run quick hardware test only")

    args = parser.parse_args()

    if args.quick:
        success = quick_hardware_test()
    else:
        success = main()

    sys.exit(0 if success else 1)
