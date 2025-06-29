#!/usr/bin/env python3
"""
Demonstration script for using the enhanced model registry with AMD and precision support

This script shows how to use the enhanced model registry to:
    1. Detect available hardware
    2. Check precision compatibility
    3. Get dependencies for models on specific hardware and precision types
    4. Simulate running tests across hardware and precision combinations

Usage:
    python demonstrate_amd_precision.py [--model MODEL_NAME] [--hardware HARDWARE1,HARDWARE2] [--precision PRECISION1,PRECISION2],
    """

    import os
    import sys
    import argparse
    import time
    from pathlib import Path

# Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model_registry_with_amd_precision import MODEL_REGISTRY, detect_hardware
except ImportError:
    print("Error importing model_registry_with_amd_precision")
    print("Please run this script from the test/sample_tests directory")
    sys.exit(1)

def is_precision_supported(model_name, hardware, precision):
    """Check if a specific precision is supported for a model on given hardware.""":
    if model_name not in MODEL_REGISTRY:
        return False
        
    # Get model info
        model_info = MODEL_REGISTRY[model_name]
        ,,,
    # Check hardware compatibility
        if not model_info["hardware_compatibility"].get(hardware, False):,
    return False
        
    # Check precision compatibility
    return model_info["precision_compatibility"][hardware].get(precision, False)
    ,
def get_dependencies(model_name, hardware, precision):
    """Get all dependencies for a model on specific hardware and precision."""
    if model_name not in MODEL_REGISTRY:
    return []
    ,
    model_info = MODEL_REGISTRY[model_name]
    ,,,
    # Start with base dependencies
    dependencies = list(model_info["dependencies"]["pip"])
    ,
    # Add hardware-specific dependencies
    if hardware in model_info["dependencies"]["optional"]:,
    dependencies.extend(model_info["dependencies"]["optional"][hardware])
    ,
    # Add precision-specific dependencies
    if precision in model_info["dependencies"]["precision"]:,
    dependencies.extend(model_info["dependencies"]["precision"][precision])
    ,
    return dependencies

def run_precision_tests(model_name, hardware_types=None, precision_types=None):
    """Simulate running tests for a model across hardware and precision types."""
    model_info = MODEL_REGISTRY.get(model_name)
    if not model_info:
        print(f"Model {model_name} not found in registry")
    return
    
    # Set defaults
    hardware_types = hardware_types or ["cpu", "cuda", "amd", "openvino", "apple", "qualcomm"],
    precision_types = precision_types or ["fp32", "fp16", "bf16", "int8", "int4", "uint4", "fp8", "fp4"]
    ,
    # Get hardware capabilities
    hardware_capabilities = detect_hardware()
    
    # Track results
    results = {}
    
    # Test each hardware and precision combination if available:
    for hardware in hardware_types:
        # Skip if hardware not available:
        if hardware != "cpu" and not hardware_capabilities.get(hardware, False):
            results[f"{hardware}_test"] = "Hardware not available",
            print(f"⚠️  {hardware.upper()}: Not available on this system")
        continue
            
        for precision in precision_types:
            # Skip if precision not supported on this hardware:
            if not model_info["precision_compatibility"][hardware].get(precision, False):,
            results[f"{hardware}_{precision}_test"] = "Precision not supported",
        continue
                
            # Run test with specific hardware and precision
            try:
                print(f"🧪 Testing {model_name} on {hardware.upper()} with {precision.upper()} precision...")
                
                # Simulate model loading and testing
                time.sleep(0.2)  # Simulate loading model
                
                # Get dependencies for this combination
                deps = get_dependencies(model_name, hardware, precision)
                
                # Print configuration details
                print(f"\1{'...' if len(deps) > 3 else ''}\3"):,
                print(f"\1{hardware.upper()}\3")
                print(f"\1{precision.upper()}\3")
                print(f"  └─ Status: Success")
                
                results[f"{hardware}_{precision}_test"] = "Success",
            except Exception as e:
                results[f"{hardware}_{precision}_test"] = f"\1{str(e)}\3",
                print(f"\1{str(e)}\3")
    
                return results

def print_hardware_summary(hardware_capabilities):
    """Print a summary of available hardware."""
    print("\n🖥️  Hardware Detection Results:")
    print("=" * 40)
    
    # CPU (always available)
    print("✅ CPU: Available")
    
    # CUDA
    if hardware_capabilities.get("cuda", False):
        print(f"✅ CUDA: Available")
        print(f"\1{hardware_capabilities.get('cuda_version', 'Unknown')}\3")
        print(f"\1{hardware_capabilities.get('cuda_devices', 0)}\3")
    else:
        print("❌ CUDA: Not available")
    
    # AMD ROCm
    if hardware_capabilities.get("amd", False):
        print(f"✅ AMD ROCm: Available")
        print(f"\1{hardware_capabilities.get('amd_version', 'Unknown')}\3")
        print(f"\1{hardware_capabilities.get('amd_devices', 0)}\3")
    else:
        print("❌ AMD ROCm: Not available")
    
    # Apple Silicon
    if hardware_capabilities.get("mps", False):
        print(f"✅ Apple Silicon: Available")
    else:
        print("❌ Apple Silicon: Not available")
    
    # OpenVINO
    if hardware_capabilities.get("openvino", False):
        print(f"✅ OpenVINO: Available")
    else:
        print("❌ OpenVINO: Not available")
    
    # Qualcomm
    if hardware_capabilities.get("qualcomm", False):
        print(f"✅ Qualcomm AI: Available")
    else:
        print("❌ Qualcomm AI: Not available")

def print_model_summary(model_name):
    """Print a summary of the model's capabilities."""
    if model_name not in MODEL_REGISTRY:
        print(f"Model {model_name} not found in registry")
    return
    
    model_info = MODEL_REGISTRY[model_name]
    ,,,
    print(f"\1{model_name}\3")
    print("=" * 40)
    print(f"\1{model_info['description']}\3"),
    print(f"\1{model_info['embedding_dim']}\3"),
    print(f"\1{model_info['sequence_length']}\3"),
    print(f"\1{model_info['default_batch_size']}\3")
    ,
    # Print hardware compatibility
    print("\nHardware Compatibility:")
    for hw, supported in model_info["hardware_compatibility"].items():,
        status = "✅ Supported" if supported else "❌ Not Supported":
            print(f"\1{status}\3")
    
    # Print precision compatibility for each hardware
            print("\nPrecision Support by Hardware:")
            for hw, precisions in model_info["precision_compatibility"].items():,
            if model_info["hardware_compatibility"].get(hw, False):,
            print(f"  {hw.upper()}:")
            for precision, supported in precisions.items():
                status = "✅" if supported else "❌":
                    print(f"\1{status}\3")

def main():
    """Main function to demonstrate model registry usage."""
    parser = argparse.ArgumentParser(description="Demonstrate enhanced model registry")
    parser.add_argument("--model", type=str, default="bert", help="Model name to check")
    parser.add_argument("--hardware", type=str, default="cpu,cuda,amd,openvino,apple,qualcomm", 
    help="Hardware types to check (comma-separated)")
    parser.add_argument("--precision", type=str, default="fp32,fp16,bf16,int8,int4,uint4", 
    help="Precision types to check (comma-separated)")
    
    args = parser.parse_args()
    
    # Get hardware capabilities
    hardware_capabilities = detect_hardware()
    
    # Print hardware summary
    print_hardware_summary(hardware_capabilities)
    
    # Print model information
    print_model_summary(args.model)
    
    # Parse hardware and precision arguments
    hardware_types = [hw.strip() for hw in args.hardware.split(",")]:,
    precision_types = [p.strip() for p in args.precision.split(",")]:,
    # Run tests
    print("\n🧪 Running Simulated Tests:")
    print("=" * 40)
    results = run_precision_tests(args.model, hardware_types, precision_types)
    
    # Print summary
    print("\n📋 Test Summary:")
    print("=" * 40)
    success_count = sum(1 for result in results.values() if "Success" in result)
    not_available = sum(1 for result in results.values() if "not available" in result)
    not_supported = sum(1 for result in results.values() if "not supported" in result)
    error_count = sum(1 for result in results.values() if "Error" in result)
    :
        print(f"\1{len(results)}\3")
        print(f"\1{success_count}\3")
        print(f"\1{not_available}\3")
        print(f"\1{not_supported}\3")
        print(f"\1{error_count}\3")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())