#!/usr/bin/env python3
"""
Test script for real Modular integration.

This script tests the actual Modular backend implementation with real
Mojo compiler detection and MAX Engine integration.
"""

import asyncio
import sys
import os
import tempfile
import json
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backends.modular_backend import ModularBackend, ModularEnvironment


async def test_environment_detection():
    """Test environment detection capabilities."""
    print("🔍 Testing Modular Environment Detection...")
    
    env = ModularEnvironment()
    
    print(f"  ✅ Mojo Available: {env.mojo_available}")
    print(f"  ✅ MAX Available: {env.max_available}")
    print(f"  ✅ Modular Version: {env.modular_version}")
    print(f"  ✅ Detected Devices: {len(env.devices)}")
    
    for device in env.devices:
        print(f"    - {device['type']}: {device['name']}")
        if 'cores' in device:
            print(f"      Cores: {device['cores']}")
        if 'memory_mb' in device:
            print(f"      Memory: {device['memory_mb']} MB")
        if 'simd_width' in device:
            print(f"      SIMD Width: {device['simd_width']}")
    
    return env


async def test_backend_initialization():
    """Test backend initialization."""
    print("\n🚀 Testing Backend Initialization...")
    
    backend = ModularBackend()
    await backend.initialize()
    
    # Get hardware info
    hw_info = backend.get_hardware_info()
    print(f"  ✅ Hardware Info: {json.dumps(hw_info, indent=2)}")
    
    # Health check
    health = await backend.health_check()
    print(f"  ✅ Health Status: {health['status']}")
    
    for component, status in health['components'].items():
        print(f"    - {component}: {status['status']}")
        if 'version' in status:
            print(f"      Version: {status['version']}")
        if 'path' in status:
            print(f"      Path: {status['path']}")
    
    return backend


async def test_mojo_compilation():
    """Test Mojo compilation."""
    print("\n🔧 Testing Mojo Compilation...")
    
    backend = ModularBackend()
    await backend.initialize()
    
    # Create a test model file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
# Test model for Mojo compilation
import numpy as np

class TestModel:
    def __init__(self):
        self.weights = np.random.randn(10, 10)
    
    def forward(self, x):
        return np.dot(x, self.weights)
""")
        test_model_path = f.name
    
    try:
        # Test compilation
        result = await backend.compile_mojo(
            model_id="test_model", 
            model_path=test_model_path,
            optimization_level="O2"
        )
        
        print(f"  ✅ Compilation Success: {result['success']}")
        print(f"  ✅ Compilation Time: {result.get('compilation_time', 'N/A')} seconds")
        print(f"  ✅ Optimizations: {result.get('optimizations_applied', [])}")
        
        if not result['success']:
            print(f"  ❌ Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"  ✅ Compiled Path: {result.get('compiled_path', 'N/A')}")
        
        return result
        
    finally:
        # Cleanup
        if os.path.exists(test_model_path):
            os.unlink(test_model_path)


async def test_max_deployment():
    """Test MAX Engine deployment."""
    print("\n🚀 Testing MAX Deployment...")
    
    backend = ModularBackend()
    await backend.initialize()
    
    # Test deployment
    result = await backend.deploy_max(
        model_id="test_model",
        target_hardware="cpu"
    )
    
    print(f"  ✅ Deployment Success: {result['success']}")
    print(f"  ✅ Endpoint URL: {result.get('endpoint_url', 'N/A')}")
    print(f"  ✅ Target Hardware: {result.get('target_hardware', 'N/A')}")
    print(f"  ✅ Allocated Resources: {result.get('allocated_resources', {})}")
    
    if not result['success']:
        print(f"  ❌ Error: {result.get('error', 'Unknown error')}")
    
    return result


async def test_benchmarking():
    """Test performance benchmarking."""
    print("\n📊 Testing Performance Benchmarking...")
    
    backend = ModularBackend()
    await backend.initialize()
    
    # Test benchmarking
    result = await backend.benchmark_modular(
        model_id="test_model",
        workload_type="inference"
    )
    
    print(f"  ✅ Benchmark Success: {result['success']}")
    print(f"  ✅ Benchmark Duration: {result.get('benchmark_duration', 'N/A')} seconds")
    
    if result['success']:
        benchmarks = result.get('benchmark_results', {})
        print(f"  ✅ Throughput: {benchmarks.get('throughput_tokens_per_sec', 'N/A')} tokens/sec")
        print(f"  ✅ Latency P50: {benchmarks.get('latency_ms', {}).get('p50', 'N/A')} ms")
        print(f"  ✅ Memory Usage: {benchmarks.get('memory_usage_mb', 'N/A')} MB")
        
        comparison = result.get('comparison_vs_baseline', {})
        print(f"  ✅ Performance vs Baseline:")
        for metric, value in comparison.items():
            print(f"    - {metric}: {value}")
    
    return result


async def main():
    """Main test function."""
    print("🎯 Real Modular Integration Test Suite")
    print("=" * 50)
    
    try:
        # Test environment detection
        env = await test_environment_detection()
        
        # Test backend initialization
        backend = await test_backend_initialization()
        
        # Test Mojo compilation
        await test_mojo_compilation()
        
        # Test MAX deployment
        await test_max_deployment()
        
        # Test benchmarking
        await test_benchmarking()
        
        print("\n🎉 All tests completed!")
        
        # Summary
        print("\n📋 Summary:")
        print(f"  - Mojo Available: {'✅' if env.mojo_available else '❌'}")
        print(f"  - MAX Available: {'✅' if env.max_available else '❌'}")
        print(f"  - Devices Detected: {len(env.devices)}")
        print(f"  - Modular Version: {env.modular_version or 'Not installed'}")
        
        if env.mojo_available or env.max_available:
            print("\n🚀 Real Modular integration is working!")
        else:
            print("\n⚠️  Modular SDK not installed - using mock implementations")
            print("   Install Modular SDK: https://developer.modular.com/download")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
