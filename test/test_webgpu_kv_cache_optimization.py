#!/usr/bin/env python3
"""
Test script for WebGPU KV-Cache optimization implementation.

This script tests the memory-efficient Key-Value cache management system
for large language models in WebGPU environments, verifying functionality
of key features:
- 4-bit quantized KV cache
- Sliding window approach for memory-constrained environments
- Dynamic cache pruning

Usage:
    python test_webgpu_kv_cache_optimization.py
"""

import os
import sys
import time
import argparse
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_kv_cache")

# Import the KV cache optimization module
try:
    from fixed_web_platform.webgpu_kv_cache_optimization import (
        WebGPUKVCacheManager,
        setup_kv_cache_for_llm,
        generate_kv_cache_shaders
    )
except ImportError:
    logger.error("Failed to import WebGPU KV cache optimization module.")
    logger.error("Make sure the module exists at fixed_web_platform/webgpu_kv_cache_optimization.py")
    sys.exit(1)

def test_kv_cache_basic_functionality():
    """Test basic functionality of the KV cache system."""
    logger.info("Testing basic KV cache functionality...")
    
    # Create a KV cache manager
    kv_manager = WebGPUKVCacheManager(
        max_seq_length=512,
        head_dim=64,
        max_memory_mb=500,
        enable_quantization=False,  # Disable quantization for this test
        sliding_window=False
    )
    
    # Initialize a cache
    cache_id = kv_manager.initialize_cache(
        batch_size=1,
        num_heads=8,
        model_name="test_model"
    )
    
    # Generate some test data
    batch_size = 1
    num_heads = 8
    head_dim = 64
    
    test_keys = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
    test_values = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
    
    # Update cache with test data
    result = kv_manager.update_cache(cache_id, test_keys, test_values, position=0)
    assert result["success"], "Failed to update KV cache"
    assert result["position"] == 0, f"Expected position 0, got {result['position']}"
    
    # Retrieve values from cache
    entries = kv_manager.get_cache_entries(cache_id, positions=[0])
    assert entries["found"], "Failed to retrieve cache entries"
    
    # Check that retrieved values match the originals (within float precision)
    retrieved_keys = entries["keys"]
    retrieved_values = entries["values"]
    
    assert retrieved_keys.shape == (batch_size, num_heads, 1, head_dim), f"Unexpected key shape: {retrieved_keys.shape}"
    assert retrieved_values.shape == (batch_size, num_heads, 1, head_dim), f"Unexpected value shape: {retrieved_values.shape}"
    
    # Check reconstruction accuracy (should be perfect without quantization)
    key_error = np.abs(retrieved_keys[:, :, 0, :] - test_keys).mean()
    value_error = np.abs(retrieved_values[:, :, 0, :] - test_values).mean()
    
    assert key_error < 1e-5, f"Key reconstruction error too high: {key_error}"
    assert value_error < 1e-5, f"Value reconstruction error too high: {value_error}"
    
    # Test cache clear
    clear_result = kv_manager.clear_cache(cache_id)
    assert clear_result["success"], "Failed to clear cache"
    
    # Verify cache is cleared
    stats = kv_manager.get_cache_statistics()
    assert stats["num_caches"] == 0, f"Expected 0 caches after clearing, got {stats['num_caches']}"
    
    logger.info("Basic KV cache functionality test passed!")
    return True

def test_kv_cache_sliding_window():
    """Test sliding window functionality of the KV cache system."""
    logger.info("Testing KV cache sliding window functionality...")
    
    # Create a KV cache manager with sliding window enabled
    max_seq_length = 128
    window_size = 32
    
    kv_manager = WebGPUKVCacheManager(
        max_seq_length=max_seq_length,
        head_dim=64,
        max_memory_mb=200,
        enable_quantization=False,
        sliding_window=True,
        window_size=window_size
    )
    
    # Initialize a cache
    cache_id = kv_manager.initialize_cache(
        batch_size=1,
        num_heads=8,
        model_name="test_model_sliding_window"
    )
    
    # Generate some test data
    batch_size = 1
    num_heads = 8
    head_dim = 64
    
    # Test sequence that's longer than the window size
    test_seq_length = window_size * 2
    
    # Add keys and values for each position
    for pos in range(test_seq_length):
        test_keys = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
        test_values = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
        
        result = kv_manager.update_cache(cache_id, test_keys, test_values, position=pos)
        assert result["success"], f"Failed to update KV cache at position {pos}"
    
    # Check cache statistics
    stats = kv_manager.get_cache_statistics(cache_id)
    assert stats["current_length"] <= window_size, f"Cache length {stats['current_length']} exceeds window size {window_size}"
    
    # After adding more tokens than the window size, the first ones should be overwritten
    # So trying to access early positions should fail or return newer values
    entries_start = kv_manager.get_cache_entries(cache_id, positions=[0])
    entries_end = kv_manager.get_cache_entries(cache_id, positions=[test_seq_length - 1])
    
    if entries_start["found"]:
        # If found, it means the position 0 maps to a newer position due to circular buffer
        assert 0 in entries_start["positions"], "Position mapping error in sliding window"
    
    assert entries_end["found"], "Should be able to retrieve the most recent position"
    
    # Clear the cache
    kv_manager.clear_cache(cache_id)
    
    logger.info("KV cache sliding window test passed!")
    return True

def test_kv_cache_quantization():
    """Test 4-bit quantization in the KV cache system."""
    logger.info("Testing KV cache 4-bit quantization...")
    
    # Skip this test if quantization is not available
    try:
        from fixed_web_platform.webgpu_quantization import WebGPUQuantizer
    except ImportError:
        logger.warning("Skipping quantization test - WebGPUQuantizer not available")
        return False
    
    # Create a KV cache manager with 4-bit quantization
    kv_manager = WebGPUKVCacheManager(
        max_seq_length=512,
        head_dim=64,
        max_memory_mb=500,
        enable_quantization=True,
        sliding_window=False
    )
    
    # Only proceed if quantization is actually enabled
    if not kv_manager.enable_quantization:
        logger.warning("Skipping quantization test - quantization not available")
        return False
    
    # Initialize a cache
    cache_id = kv_manager.initialize_cache(
        batch_size=1,
        num_heads=8,
        model_name="test_model_quantized"
    )
    
    # Generate some test data
    batch_size = 1
    num_heads = 8
    head_dim = 64
    
    # Use controlled data to test quantization accuracy
    # Create tensor with values from -1 to 1 to test full quantization range
    range_tensor = np.linspace(-1, 1, head_dim, dtype=np.float32)
    test_keys = np.tile(range_tensor, (batch_size, num_heads, 1))
    test_values = np.tile(range_tensor, (batch_size, num_heads, 1))
    
    # Update cache with test data
    result = kv_manager.update_cache(cache_id, test_keys, test_values, position=0)
    assert result["success"], "Failed to update KV cache with quantized data"
    
    # Retrieve quantized values from cache
    entries = kv_manager.get_cache_entries(cache_id, positions=[0])
    assert entries["found"], "Failed to retrieve quantized cache entries"
    
    # Check reconstruction accuracy (should be lower with 4-bit quantization)
    retrieved_keys = entries["keys"]
    retrieved_values = entries["values"]
    
    key_error = np.abs(retrieved_keys[:, :, 0, :] - test_keys).mean()
    value_error = np.abs(retrieved_values[:, :, 0, :] - test_values).mean()
    
    # Since we're using 4-bit quantization, some error is expected
    assert key_error < 0.1, f"Key quantization error too high: {key_error}"
    assert value_error < 0.1, f"Value quantization error too high: {value_error}"
    
    # Test memory reduction
    stats = kv_manager.get_cache_statistics(cache_id)
    expected_memory_reduction = 0.75  # 4-bit should be 75% smaller than 32-bit
    
    # Compare with a non-quantized version to verify memory savings
    kv_manager_fp32 = WebGPUKVCacheManager(
        max_seq_length=512,
        head_dim=64,
        max_memory_mb=500,
        enable_quantization=False,
        sliding_window=False
    )
    
    cache_id_fp32 = kv_manager_fp32.initialize_cache(
        batch_size=1,
        num_heads=8,
        model_name="test_model_fp32"
    )
    
    stats_fp32 = kv_manager_fp32.get_cache_statistics(cache_id_fp32)
    
    # Check memory usage difference (should be close to 4:1 ratio)
    memory_ratio = stats["memory_mb"] / stats_fp32["memory_mb"]
    assert memory_ratio < 0.5, f"Memory reduction not significant: {memory_ratio:.2f}, expected ~0.25"
    
    logger.info(f"KV cache 4-bit quantization test passed! Memory ratio: {memory_ratio:.2f}")
    return True

def test_kv_cache_pruning():
    """Test dynamic pruning of the KV cache."""
    logger.info("Testing KV cache dynamic pruning...")
    
    # Create a KV cache manager with pruning enabled
    kv_manager = WebGPUKVCacheManager(
        max_seq_length=128,
        head_dim=64,
        max_memory_mb=200,
        enable_quantization=False,
        sliding_window=False,
        enable_pruning=True
    )
    
    # Initialize a cache
    cache_id = kv_manager.initialize_cache(
        batch_size=1,
        num_heads=8,
        model_name="test_model_pruning"
    )
    
    # Generate some test data
    batch_size = 1
    num_heads = 8
    head_dim = 64
    
    # Add keys and values for 32 positions
    num_positions = 32
    for pos in range(num_positions):
        test_keys = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
        test_values = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
        
        kv_manager.update_cache(cache_id, test_keys, test_values, position=pos)
    
    # Verify all positions are cached
    stats_before = kv_manager.get_cache_statistics(cache_id)
    assert stats_before["current_length"] == num_positions, f"Expected {num_positions} positions, got {stats_before['current_length']}"
    
    # Perform pruning
    pruning_result = kv_manager.prune_cache(cache_id, strategy="least_used")
    assert pruning_result["success"], "Pruning failed"
    
    # Verify cache was reduced
    stats_after = kv_manager.get_cache_statistics(cache_id)
    assert stats_after["current_length"] < num_positions, f"Expected reduced length after pruning, got {stats_after['current_length']}"
    assert stats_after["current_length"] == pruning_result["tokens_kept"], "Inconsistent token count after pruning"
    
    # Try different pruning strategies
    # First, reset the cache
    kv_manager.clear_cache(cache_id)
    cache_id = kv_manager.initialize_cache(
        batch_size=1,
        num_heads=8,
        model_name="test_model_pruning"
    )
    
    # Add keys and values for positions
    for pos in range(num_positions):
        test_keys = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
        test_values = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
        kv_manager.update_cache(cache_id, test_keys, test_values, position=pos)
    
    # Add extra accesses to certain positions
    special_positions = [5, 10, 15]
    for pos in special_positions:
        # Access these positions multiple times
        for _ in range(5):  # Access 5 times each
            kv_manager.get_cache_entries(cache_id, positions=[pos])
    
    # Prune using least_used strategy
    result_least_used = kv_manager.prune_cache(cache_id, strategy="least_used")
    assert result_least_used["success"], "least_used pruning failed"
    
    # Verify special positions are still in cache
    entries = kv_manager.get_cache_entries(cache_id, positions=special_positions)
    assert entries["found"], "Frequently used positions were incorrectly pruned"
    
    logger.info("KV cache dynamic pruning test passed!")
    return True

def test_shader_generation():
    """Test shader code generation for KV cache operations."""
    logger.info("Testing KV cache shader generation...")
    
    # Generate shaders with different configurations
    shader_configs = [
        {"seq_length": 512, "num_heads": 8, "head_dim": 64, "use_4bit": True, "causal": True},
        {"seq_length": 2048, "num_heads": 32, "head_dim": 128, "use_4bit": True, "causal": True},
        {"seq_length": 512, "num_heads": 8, "head_dim": 64, "use_4bit": False, "causal": False},
    ]
    
    for i, config in enumerate(shader_configs):
        logger.info(f"Testing shader configuration {i+1}: {config}")
        
        # Generate shaders
        shaders = generate_kv_cache_shaders(**config)
        
        # Verify expected shader components exist
        assert "kv_access" in shaders, "Missing kv_access shader"
        assert "kv_update" in shaders, "Missing kv_update shader"
        
        # Check basic content
        for shader_type, shader_data in shaders.items():
            assert "shader_code" in shader_data, f"Missing shader code in {shader_type}"
            assert "entry_point" in shader_data, f"Missing entry point in {shader_type}"
            assert "workgroup_size" in shader_data, f"Missing workgroup size in {shader_type}"
            assert "configuration" in shader_data, f"Missing configuration in {shader_type}"
            
            # Verify configuration matches input
            shader_config = shader_data["configuration"]
            for key, value in config.items():
                assert shader_config[key] == value, f"Configuration mismatch for {key}: expected {value}, got {shader_config[key]}"
            
            # Check if shader code contains type-specific bindings
            if config["use_4bit"]:
                assert "u8" in shader_data["shader_code"], f"4-bit shader should use u8 type but it's missing in {shader_type}"
            else:
                assert "f32" in shader_data["shader_code"], f"Full precision shader should use f32 type in {shader_type}"
    
    logger.info("KV cache shader generation test passed!")
    return True

def test_setup_function():
    """Test the setup_kv_cache_for_llm convenience function."""
    logger.info("Testing KV cache setup function...")
    
    # Test with various configurations
    test_configs = [
        {"model_name": "llama-7b", "max_seq_length": 2048, "head_dim": 128, "num_heads": 32, 
         "enable_quantization": False, "sliding_window": True, "window_size": 512},
        
        {"model_name": "qwen2-7b", "max_seq_length": 1024, "head_dim": 128, "num_heads": 32,
         "enable_quantization": True, "sliding_window": False, "window_size": None},
         
        {"model_name": "falcon-7b", "max_seq_length": 4096, "head_dim": 64, "num_heads": 64,
         "enable_quantization": True, "sliding_window": True, "window_size": 2048}
    ]
    
    for config in test_configs:
        # Set up KV cache
        kv_manager, cache_id = setup_kv_cache_for_llm(**config)
        
        # Verify KV cache manager was created
        assert isinstance(kv_manager, WebGPUKVCacheManager), "setup_kv_cache_for_llm did not return a WebGPUKVCacheManager"
        assert cache_id is not None, "setup_kv_cache_for_llm did not return a valid cache ID"
        
        # Verify configuration was applied
        stats = kv_manager.get_cache_statistics(cache_id)
        assert stats["batch_size"] == 1, f"Expected batch_size=1, got {stats['batch_size']}"
        assert stats["num_heads"] == config["num_heads"], f"Expected num_heads={config['num_heads']}, got {stats['num_heads']}"
        assert stats["head_dim"] == config["head_dim"], f"Expected head_dim={config['head_dim']}, got {stats['head_dim']}"
        
        # Check sliding window configuration
        if config["sliding_window"]:
            win_size = config["window_size"] or (config["max_seq_length"] // 4)
            assert stats["sliding_window"], "Sliding window not enabled"
            assert stats["window_size"] == win_size, f"Expected window_size={win_size}, got {stats['window_size']}"
    
    logger.info("KV cache setup function test passed!")
    return True

def test_large_model_memory_efficiency(model_size_gb=7):
    """Test memory efficiency for large models like 7B parameter LLMs."""
    logger.info(f"Testing memory efficiency for {model_size_gb}B parameter model...")
    
    # Simulate approximate KV cache memory requirements for a large model
    # 7B model typical config: ~32 layers, 32 heads, head_dim=128
    num_layers = 32
    num_heads = 32
    head_dim = 128
    seq_length = 2048
    batch_size = 1
    
    # Memory required for full-precision KV cache (per layer)
    # KV cache: 2 (K+V) * batch_size * num_heads * seq_length * head_dim * 4 bytes (float32)
    memory_per_layer_mb = 2 * batch_size * num_heads * seq_length * head_dim * 4 / (1024 * 1024)
    total_memory_mb = memory_per_layer_mb * num_layers
    
    logger.info(f"Estimated KV cache memory for {seq_length} tokens: {total_memory_mb:.2f}MB (full precision)")
    
    # Test different optimization strategies
    strategies = [
        {"name": "Full precision", "quantization": False, "sliding_window": False, "window_size": None},
        {"name": "4-bit quantization", "quantization": True, "sliding_window": False, "window_size": None},
        {"name": "Sliding window (1024)", "quantization": False, "sliding_window": True, "window_size": 1024},
        {"name": "Sliding window (512)", "quantization": False, "sliding_window": True, "window_size": 512},
        {"name": "Combined optimizations", "quantization": True, "sliding_window": True, "window_size": 512}
    ]
    
    results = []
    for strategy in strategies:
        # Create KV cache manager with this strategy
        kv_manager = WebGPUKVCacheManager(
            max_seq_length=seq_length,
            head_dim=head_dim,
            max_memory_mb=total_memory_mb * 2,  # Set high to avoid automatic restrictions
            enable_quantization=strategy["quantization"],
            sliding_window=strategy["sliding_window"],
            window_size=strategy["window_size"]
        )
        
        # Initialize cache
        cache_id = kv_manager.initialize_cache(
            batch_size=batch_size,
            num_heads=num_heads,
            model_name=f"llama-{model_size_gb}b"
        )
        
        # Get memory usage statistics
        stats = kv_manager.get_cache_statistics(cache_id)
        memory_mb = stats["memory_mb"]
        
        # Calculate reduction percentage
        reduction_percent = (1 - memory_mb / total_memory_mb) * 100
        
        results.append({
            "strategy": strategy["name"],
            "memory_mb": memory_mb,
            "reduction_percent": reduction_percent
        })
        
        logger.info(f"Strategy: {strategy['name']}")
        logger.info(f"  Memory usage: {memory_mb:.2f}MB")
        logger.info(f"  Reduction: {reduction_percent:.2f}%")
    
    # Verify that the combined strategy has the lowest memory usage
    memory_usages = [r["memory_mb"] for r in results]
    min_memory = min(memory_usages)
    min_strategy_idx = memory_usages.index(min_memory)
    
    assert results[min_strategy_idx]["strategy"] == "Combined optimizations", \
        f"Expected 'Combined optimizations' to have lowest memory, but got {results[min_strategy_idx]['strategy']}"
    
    # Verify 4-bit quantization achieves ~75% reduction
    quant_result = next(r for r in results if r["strategy"] == "4-bit quantization")
    assert quant_result["reduction_percent"] > 70, \
        f"4-bit quantization achieved only {quant_result['reduction_percent']:.2f}% reduction, expected >70%"
    
    logger.info("Large model memory efficiency test passed!")
    return results

def run_integration_test(seq_length=512, num_heads=8, head_dim=64):
    """Run an integration test simulating realistic KV cache usage during LLM inference."""
    logger.info("Running KV cache integration test...")
    
    # Create KV cache manager with all optimizations
    kv_manager = WebGPUKVCacheManager(
        max_seq_length=seq_length,
        head_dim=head_dim,
        max_memory_mb=500,
        enable_quantization=True,
        sliding_window=True,
        window_size=256,
        enable_pruning=True
    )
    
    # Initialize cache
    cache_id = kv_manager.initialize_cache(
        batch_size=1,
        num_heads=num_heads,
        model_name="test_integration"
    )
    
    # Simulate autoregressive generation
    batch_size = 1
    input_length = 32  # Initial input length
    total_length = 128  # Target sequence length
    
    logger.info(f"Simulating autoregressive generation from {input_length} to {total_length} tokens...")
    
    # First, add initial input to KV cache
    for pos in range(input_length):
        keys = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
        values = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
        kv_manager.update_cache(cache_id, keys, values, position=pos)
    
    # Then simulate autoregressive generation, adding one token at a time
    for pos in range(input_length, total_length):
        # First, retrieve the KV cache for previous tokens
        # In a real implementation, this would be used for attention computation
        prev_positions = list(range(max(0, pos-16), pos))  # Get recent positions for attention
        entries = kv_manager.get_cache_entries(cache_id, positions=prev_positions)
        
        if not entries["found"]:
            logger.error(f"Failed to retrieve cache entries at position {pos}")
            return False
        
        # Generate new KV for the current position
        keys = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
        values = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
        
        # Update the cache
        kv_manager.update_cache(cache_id, keys, values, position=pos)
        
        # Every 32 tokens, report status and conditionally prune
        if pos % 32 == 0 and pos > input_length:
            stats = kv_manager.get_cache_statistics(cache_id)
            logger.info(f"Position {pos}: Cache size {stats['current_length']} tokens, Memory: {stats['memory_mb']:.2f}MB")
            
            # Simulate pruning decision (e.g., when memory usage is high)
            if stats["current_length"] > 96:
                logger.info("Pruning KV cache...")
                pruning_result = kv_manager.prune_cache(cache_id, strategy="least_used")
                if pruning_result["success"]:
                    logger.info(f"Pruned {pruning_result['tokens_pruned']} tokens, kept {pruning_result['tokens_kept']}")
    
    # Report final statistics
    final_stats = kv_manager.get_cache_statistics(cache_id)
    logger.info(f"Final cache size: {final_stats['current_length']} tokens")
    logger.info(f"Final memory usage: {final_stats['memory_mb']:.2f}MB")
    logger.info(f"KV cache integration test completed successfully!")
    
    return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test WebGPU KV cache optimizations")
    parser.add_argument("--test", choices=["all", "basic", "sliding_window", "quantization", 
                                          "pruning", "shader", "setup", "memory", "integration"],
                       default="all", help="Which test to run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def main():
    """Main function to run tests."""
    args = parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("WebGPU KV Cache Optimization Tests")
    print("==================================")
    
    test_functions = {
        "basic": test_kv_cache_basic_functionality,
        "sliding_window": test_kv_cache_sliding_window,
        "quantization": test_kv_cache_quantization,
        "pruning": test_kv_cache_pruning,
        "shader": test_shader_generation,
        "setup": test_setup_function,
        "memory": test_large_model_memory_efficiency,
        "integration": run_integration_test
    }
    
    # Run selected test or all tests
    if args.test == "all":
        print("\nRunning all tests...\n")
        success = True
        for test_name, test_func in test_functions.items():
            print(f"\n--- Running {test_name} test ---")
            try:
                result = test_func()
                if not result:
                    print(f"❌ {test_name} test failed or was skipped")
                    success = False
                else:
                    print(f"✅ {test_name} test passed")
            except Exception as e:
                print(f"❌ {test_name} test failed with error: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                success = False
        
        if success:
            print("\n🎉 All tests passed successfully!")
        else:
            print("\n⚠️ Some tests failed or were skipped")
            sys.exit(1)
    else:
        # Run individual test
        print(f"\nRunning {args.test} test...\n")
        try:
            result = test_functions[args.test]()
            if result:
                print(f"\n✅ {args.test} test passed successfully!")
            else:
                print(f"\n❌ {args.test} test failed or was skipped")
                sys.exit(1)
        except Exception as e:
            print(f"\n❌ {args.test} test failed with error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()