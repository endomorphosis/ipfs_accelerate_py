#!/usr/bin/env python3
"""
Test Script for WebGPU/WebNN Resource Pool Integration

This script demonstrates the functionality of the WebGPU/WebNN Resource Pool Integration,
including fault tolerance features, connection pooling, browser-aware load balancing,
cross-browser model sharding, and performance history tracking.

Usage:
    python run_test_webgpu_resource_pool.py [--models MODEL_LIST] [--fault-tolerance]
                                         [--test-sharding] [--recovery-tests]
                                         [--concurrent-models] [--fault-injection]
                                         [--stress-test] [--duration SECONDS]
                                         [--test-state-management] [--sync-interval SECONDS]
"""

import argparse
import anyio
import json
import logging
import random
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

# Import the resource pool integration
from resource_pool_bridge import ResourcePoolBridgeIntegration, ModelProxy, FaultTolerantModelProxy
from model_sharding import ShardedModelExecution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("webgpu_resource_pool_test.log")
    ]
)
logger = logging.getLogger(__name__)

# Sample model configurations for testing
SAMPLE_MODELS = {
    "bert": {
        "name": "bert-base-uncased",
        "type": "text_embedding",
        "input_example": "This is a sample text for embedding",
        "hardware_preferences": {"priority_list": ["webgpu", "cpu"]}
    },
    "vit": {
        "name": "vit-base-patch16-224",
        "type": "vision",
        "input_example": {"image_data": "simulated_image_data", "width": 224, "height": 224},
        "hardware_preferences": {"priority_list": ["webgpu", "cpu"]}
    },
    "whisper": {
        "name": "whisper-small",
        "type": "audio",
        "input_example": {"audio_data": "simulated_audio_data", "sample_rate": 16000},
        "hardware_preferences": {"priority_list": ["webgpu", "cpu"]}
    },
    "llama": {
        "name": "llama-7b",
        "type": "large_language_model",
        "input_example": "Write a short poem about technology",
        "hardware_preferences": {"priority_list": ["webgpu", "webnn", "cpu"]}
    }
}

async def test_basic_functionality(integration):
    """Test basic functionality of the resource pool integration."""
    logger.info("Testing basic functionality")
    
    # Get a model
    model = await integration.get_model(
        model_type="text_embedding",
        model_name="bert-base-uncased",
        hardware_preferences={"priority_list": ["webgpu", "cpu"]},
        fault_tolerance={
            "recovery_timeout": 30,
            "state_persistence": True,
            "failover_strategy": "immediate"
        }
    )
    
    if not model:
        logger.error("Failed to get model")
        return False
    
    # Run inference
    start_time = time.time()
    result = await model("This is a sample text for embedding")
    duration = time.time() - start_time
    
    logger.info(f"Inference result: {result}")
    logger.info(f"Inference duration: {duration:.3f} seconds")
    
    # Get model info
    info = await model.get_info()
    logger.info(f"Model info: {info}")
    
    return True

async def test_concurrent_models(integration, model_list):
    """Test concurrent model execution."""
    logger.info("Testing concurrent model execution")
    
    # Get models
    models = []
    for model_name in model_list:
        if model_name not in SAMPLE_MODELS:
            logger.warning(f"Model {model_name} not found in sample models")
            continue
            
        model_config = SAMPLE_MODELS[model_name]
        
        model = await integration.get_model(
            model_type=model_config["type"],
            model_name=model_config["name"],
            hardware_preferences=model_config["hardware_preferences"],
            fault_tolerance={
                "recovery_timeout": 30,
                "state_persistence": True,
                "failover_strategy": "immediate"
            }
        )
        
        if model:
            models.append((model_name, model, model_config))
            logger.info(f"Got model {model_name}")
        else:
            logger.error(f"Failed to get model {model_name}")
    
    if not models:
        logger.error("No models were created")
        return False
    
    # Run inference on all models concurrently
    tasks = []
    
    for model_name, model, model_config in models:
        task = model(model_config["input_example"])
        tasks.append((model_name, task))
    
    # Wait for all inference tasks
    results = {}
    
    for model_name, task in tasks:
        try:
            result = await task
            results[model_name] = result
            logger.info(f"Inference completed for {model_name}")
        except Exception as e:
            logger.error(f"Inference failed for {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    # Log results
    for model_name, result in results.items():
        logger.info(f"Result for {model_name}: {result}")
    
    return True

async def test_fault_tolerance(integration, model_list):
    """Test fault tolerance features."""
    logger.info("Testing fault tolerance features")
    
    # Get a model with fault tolerance
    model_name = model_list[0] if model_list else "bert"
    model_config = SAMPLE_MODELS[model_name]
    
    model = await integration.get_model(
        model_type=model_config["type"],
        model_name=model_config["name"],
        hardware_preferences=model_config["hardware_preferences"],
        fault_tolerance={
            "recovery_timeout": 30,
            "state_persistence": True,
            "failover_strategy": "immediate"
        }
    )
    
    if not model:
        logger.error(f"Failed to get model {model_name}")
        return False
    
    logger.info(f"Got model {model_name}")
    
    # Run inference once to establish baseline
    try:
        result = await model(model_config["input_example"])
        logger.info(f"Baseline inference completed for {model_name}: {result}")
    except Exception as e:
        logger.error(f"Baseline inference failed for {model_name}: {str(e)}")
        return False
    
    # Simulate browser crash by changing browser_id to an invalid value
    original_browser_id = model.browser_id
    logger.info(f"Simulating browser crash by changing browser_id from {original_browser_id} to 'crashed-browser'")
    model.browser_id = "crashed-browser"
    
    # Run inference again - should trigger recovery
    try:
        result = await model(model_config["input_example"])
        logger.info(f"Recovery inference completed for {model_name}: {result}")
        logger.info(f"Model was recovered with new browser_id: {model.browser_id}")
        
        if model.browser_id != original_browser_id and model.browser_id != "crashed-browser":
            logger.info("✅ Fault tolerance recovery successful")
            return True
        else:
            logger.error("❌ Fault tolerance recovery failed - browser ID did not change")
            return False
    except Exception as e:
        logger.error(f"Recovery inference failed for {model_name}: {str(e)}")
        return False

async def test_model_sharding(integration, model_list):
    """Test cross-browser model sharding."""
    logger.info("Testing cross-browser model sharding")
    
    # Create sharded model execution
    try:
        sharded_execution = ShardedModelExecution(
            model_name="llama-13b",
            sharding_strategy="layer_balanced",
            num_shards=3,
            fault_tolerance_level="high",
            recovery_strategy="coordinated",
            connection_pool=integration.connection_pool
        )
        
        # Initialize sharded execution
        await sharded_execution.initialize()
        
        logger.info("Sharded model initialized successfully")
        
        # Run inference on sharded model
        result = await sharded_execution.run_inference("Write a short story about artificial intelligence")
        
        logger.info(f"Sharded model inference result: {result}")
        
        return True
    except Exception as e:
        logger.error(f"Error in sharded model execution: {str(e)}")
        return False

async def test_sharding_recovery(integration, model_list):
    """Test recovery in sharded model execution."""
    logger.info("Testing recovery in sharded model execution")
    
    # Create sharded model execution
    try:
        sharded_execution = ShardedModelExecution(
            model_name="llama-13b",
            sharding_strategy="layer_balanced",
            num_shards=3,
            fault_tolerance_level="high",
            recovery_strategy="retry_failed_shards",
            connection_pool=integration.connection_pool
        )
        
        # Initialize sharded execution
        await sharded_execution.initialize()
        
        logger.info("Sharded model initialized successfully")
        
        # Simulate shard failure by modifying an internal browser assignment
        # This is a bit hacky but works for the test
        shard_id = list(sharded_execution.sharded_model_manager.sharded_models[sharded_execution.sharded_model_id]["shards"].keys())[0]
        original_browser_id = sharded_execution.sharded_model_manager.sharded_models[sharded_execution.sharded_model_id]["shards"][shard_id]["browser_id"]
        
        logger.info(f"Simulating failure for shard {shard_id} by changing browser_id from {original_browser_id} to 'crashed-browser'")
        sharded_execution.sharded_model_manager.sharded_models[sharded_execution.sharded_model_id]["shards"][shard_id]["browser_id"] = "crashed-browser"
        
        # Run inference on sharded model - should trigger recovery
        result = await sharded_execution.run_inference("Write a short story about artificial intelligence")
        
        logger.info(f"Sharded model inference result after recovery: {result}")
        
        # Check if recovery happened
        current_browser_id = sharded_execution.sharded_model_manager.sharded_models[sharded_execution.sharded_model_id]["shards"][shard_id]["browser_id"]
        
        if current_browser_id != original_browser_id and current_browser_id != "crashed-browser":
            logger.info(f"✅ Shard recovery successful, new browser_id: {current_browser_id}")
            return True
        else:
            logger.error("❌ Shard recovery failed - browser ID did not change")
            return False
    except Exception as e:
        logger.error(f"Error in sharded model recovery test: {str(e)}")
        return False

async def test_performance_history(integration):
    """Test performance history tracking and analysis."""
    logger.info("Testing performance history tracking and analysis")
    
    # Simulate some performance data
    for _ in range(10):
        # Record a simulated operation
        await integration.performance_tracker.record_operation_performance(
            browser_id=f"chrome-{random.randint(1, 3)}",
            model_id=f"model-{random.randint(1, 5)}",
            model_type=random.choice(["text_embedding", "vision", "audio"]),
            operation_type="inference",
            latency=random.uniform(50, 500),
            success=random.random() > 0.2,
            metadata={"batch_size": random.randint(1, 8)}
        )
    
    # Get performance history
    history = await integration.get_performance_history(
        model_type="text_embedding",
        time_range="7d",
        metrics=["latency", "success_rate", "sample_count"]
    )
    
    logger.info(f"Performance history: {json.dumps(history, indent=2)}")
    
    # Analyze trends
    recommendations = await integration.analyze_performance_trends(history)
    
    logger.info(f"Performance recommendations: {json.dumps(recommendations, indent=2)}")
    
    # Apply optimizations
    success = await integration.apply_performance_optimizations(recommendations)
    
    logger.info(f"Applied optimizations: {success}")
    
    return True

async def test_stress(integration, model_list, duration, fault_injection):
    """Run a stress test with high concurrency and optional fault injection."""
    logger.info(f"Running stress test for {duration} seconds with fault injection: {fault_injection}")
    
    # Track results
    total_operations = 0
    successful_operations = 0
    failed_operations = 0
    fault_recovery_success = 0
    fault_recovery_failure = 0
    
    # Create models
    models = []
    for model_name in model_list:
        if model_name not in SAMPLE_MODELS:
            continue
            
        model_config = SAMPLE_MODELS[model_name]
        
        model = await integration.get_model(
            model_type=model_config["type"],
            model_name=model_config["name"],
            hardware_preferences=model_config["hardware_preferences"],
            fault_tolerance={
                "recovery_timeout": 30,
                "state_persistence": True,
                "failover_strategy": "immediate"
            }
        )
        
        if model:
            models.append((model_name, model, model_config))
            logger.info(f"Created model {model_name} for stress test")
        else:
            logger.error(f"Failed to create model {model_name} for stress test")
    
    if not models:
        logger.error("No models were created for stress test")
        return False
    
    # Run operations for the specified duration
    start_time = time.time()
    end_time = start_time + duration
    
    while time.time() < end_time:
        # Select a random model
        model_name, model, model_config = random.choice(models)
        
        try:
            # Inject fault randomly if enabled
            if fault_injection and random.random() < 0.3:
                original_browser_id = model.browser_id
                logger.info(f"Injecting fault for model {model_name} by changing browser_id to 'crashed-browser'")
                model.browser_id = "crashed-browser"
                
                # Run inference - should trigger recovery
                result = await model(model_config["input_example"])
                
                # Check if recovery happened
                if model.browser_id != original_browser_id and model.browser_id != "crashed-browser":
                    logger.info(f"Fault recovery successful for model {model_name}, new browser_id: {model.browser_id}")
                    fault_recovery_success += 1
                else:
                    logger.warning(f"Fault recovery failed for model {model_name}")
                    fault_recovery_failure += 1
            else:
                # Normal operation
                result = await model(model_config["input_example"])
            
            # Operation completed successfully
            successful_operations += 1
            
        except Exception as e:
            logger.error(f"Operation failed for model {model_name}: {str(e)}")
            failed_operations += 1
        
        total_operations += 1
        
        # Brief pause to avoid flooding
        await anyio.sleep(0.1)
    
    # Log results
    elapsed = time.time() - start_time
    operations_per_second = total_operations / elapsed
    
    logger.info(f"Stress test completed:")
    logger.info(f"- Duration: {elapsed:.2f} seconds")
    logger.info(f"- Total operations: {total_operations}")
    logger.info(f"- Successful operations: {successful_operations}")
    logger.info(f"- Failed operations: {failed_operations}")
    logger.info(f"- Operations per second: {operations_per_second:.2f}")
    
    if fault_injection:
        logger.info(f"- Fault injections with successful recovery: {fault_recovery_success}")
        logger.info(f"- Fault injections with failed recovery: {fault_recovery_failure}")
        logger.info(f"- Recovery success rate: {fault_recovery_success / (fault_recovery_success + fault_recovery_failure) * 100:.2f}%")
    
    return True

async def test_state_management(integration, sync_interval):
    """Test transaction-based state management."""
    logger.info(f"Testing transaction-based state management with sync interval: {sync_interval}")
    
    # Check if state manager is available
    if not hasattr(integration, 'state_manager') or not integration.state_manager:
        logger.error("State manager not available")
        return False
    
    # Set custom sync interval
    integration.state_manager.sync_interval = sync_interval
    
    # Test browser registration
    browser_id = f"test-browser-{uuid.uuid4().hex[:8]}"
    browser_type = "chrome"
    capabilities = {"webgpu": True, "webnn": False, "compute_shaders": True}
    
    success = await integration.state_manager.register_browser(
        browser_id=browser_id,
        browser_type=browser_type,
        capabilities=capabilities
    )
    
    if not success:
        logger.error("Failed to register browser")
        return False
    
    logger.info(f"Registered browser {browser_id}")
    
    # Test model registration
    model_id = f"test-model-{uuid.uuid4().hex[:8]}"
    model_name = "bert-test"
    model_type = "text_embedding"
    
    success = await integration.state_manager.register_model(
        model_id=model_id,
        model_name=model_name,
        model_type=model_type,
        browser_id=browser_id
    )
    
    if not success:
        logger.error("Failed to register model")
        return False
    
    logger.info(f"Registered model {model_id} on browser {browser_id}")
    
    # Test operation tracking
    operation_id = f"test-op-{uuid.uuid4().hex[:8]}"
    
    await integration.state_manager.record_operation(
        operation_id=operation_id,
        model_id=model_id,
        operation_type="inference",
        start_time=datetime.now().isoformat(),
        status="started",
        metadata={"test": True}
    )
    
    logger.info(f"Recorded operation {operation_id}")
    
    # Complete operation
    await integration.state_manager.complete_operation(
        operation_id=operation_id,
        status="completed",
        end_time=datetime.now().isoformat(),
        result={"success": True}
    )
    
    logger.info(f"Completed operation {operation_id}")
    
    # Test browser reassignment
    new_browser_id = f"test-browser-{uuid.uuid4().hex[:8]}"
    
    success = await integration.state_manager.register_browser(
        browser_id=new_browser_id,
        browser_type="edge",
        capabilities={"webgpu": True, "webnn": True, "compute_shaders": False}
    )
    
    if not success:
        logger.error("Failed to register new browser")
        return False
    
    logger.info(f"Registered new browser {new_browser_id}")
    
    # Update model browser
    success = await integration.state_manager.update_model_browser(
        model_id=model_id,
        browser_id=new_browser_id
    )
    
    if not success:
        logger.error("Failed to update model browser")
        return False
    
    logger.info(f"Updated model {model_id} to use browser {new_browser_id}")
    
    # Verify state
    model_state = integration.state_manager.get_model_state(model_id)
    
    if not model_state:
        logger.error("Failed to get model state")
        return False
    
    if model_state.get("browser_id") != new_browser_id:
        logger.error(f"Model state has incorrect browser ID: {model_state.get('browser_id')}")
        return False
    
    logger.info(f"Model state verified, browser ID: {model_state.get('browser_id')}")
    
    # Force state sync
    await integration.state_manager._sync_state()
    await integration.state_manager._update_checksums()
    await integration.state_manager._verify_state_consistency()
    
    logger.info("Forced state synchronization")
    
    # Simulate state corruption
    logger.info("Simulating state corruption...")
    integration.state_manager.state["models"][model_id]["browser_id"] = "corrupted-browser"
    
    # Force verification - should detect inconsistency
    await integration.state_manager._update_checksums()
    await integration.state_manager._verify_state_consistency()
    
    logger.info("State consistency verification completed")
    
    return True

async def main():
    """Main entry point for the test script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test WebGPU/WebNN Resource Pool Integration")
    
    parser.add_argument("--models", default="bert,vit,whisper", help="Comma-separated list of models to test")
    parser.add_argument("--fault-tolerance", action="store_true", help="Test fault tolerance features")
    parser.add_argument("--test-sharding", action="store_true", help="Test cross-browser model sharding")
    parser.add_argument("--recovery-tests", action="store_true", help="Test recovery mechanisms")
    parser.add_argument("--concurrent-models", action="store_true", help="Test concurrent model execution")
    parser.add_argument("--fault-injection", action="store_true", help="Test with fault injection")
    parser.add_argument("--stress-test", action="store_true", help="Run stress test with high concurrency")
    parser.add_argument("--duration", type=int, default=60, help="Duration of stress test in seconds")
    parser.add_argument("--test-state-management", action="store_true", help="Test transaction-based state management")
    parser.add_argument("--sync-interval", type=int, default=5, help="Sync interval for state management in seconds")
    
    args = parser.parse_args()
    
    # Parse model list
    model_list = args.models.split(",")
    
    logger.info("Starting WebGPU/WebNN Resource Pool Integration test")
    logger.info(f"Models: {model_list}")
    
    # Create resource pool integration
    integration = ResourcePoolBridgeIntegration(
        max_connections=4,
        browser_preferences={
            'audio': 'firefox',
            'vision': 'chrome',
            'text_embedding': 'edge'
        },
        adaptive_scaling=True,
        enable_fault_tolerance=True,
        recovery_strategy="progressive",
        state_sync_interval=args.sync_interval,
        redundancy_factor=2
    )
    
    # Initialize integration
    await integration.initialize()
    
    # Setup signal handlers for graceful shutdown
    should_exit = False
    
    def shutdown_handler(signum, frame):
        nonlocal should_exit
        logger.info(f"Received signal {signum}, initiating shutdown")
        should_exit = True
    
    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, shutdown_handler)
    
    # Run tests based on arguments
    test_results = {}
    
    try:
        # Always run basic functionality test
        test_results["basic"] = await test_basic_functionality(integration)
        
        # Run selected tests
        if args.concurrent_models:
            test_results["concurrent_models"] = await test_concurrent_models(integration, model_list)
        
        if args.fault_tolerance:
            test_results["fault_tolerance"] = await test_fault_tolerance(integration, model_list)
        
        if args.test_sharding:
            test_results["sharding"] = await test_model_sharding(integration, model_list)
            
            if args.recovery_tests:
                test_results["sharding_recovery"] = await test_sharding_recovery(integration, model_list)
        
        if args.recovery_tests and not args.test_sharding:
            test_results["fault_tolerance"] = await test_fault_tolerance(integration, model_list)
        
        if args.test_state_management:
            test_results["state_management"] = await test_state_management(integration, args.sync_interval)
        
        # Performance history tracking
        test_results["performance_history"] = await test_performance_history(integration)
        
        # Run stress test last
        if args.stress_test:
            test_results["stress_test"] = await test_stress(integration, model_list, args.duration, args.fault_injection)
        
    except Exception as e:
        logger.error(f"Error in test execution: {str(e)}")
        
    # Print test results
    logger.info("\n=== Test Results ===")
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    success_count = sum(1 for result in test_results.values() if result)
    total_count = len(test_results)
    
    logger.info(f"\nSummary: {success_count}/{total_count} tests passed")
    
    # Clean up
    logger.info("Tests completed, shutting down")

if __name__ == "__main__":
    anyio.run(main())