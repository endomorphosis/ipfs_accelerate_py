#!/usr/bin/env python3
"""
Advanced Fault-Tolerant Cross-Browser Model Sharding Test Suite

This script provides comprehensive testing for fault-tolerant cross-browser model sharding,
focusing on validating enterprise-grade fault tolerance capabilities and recovery mechanisms.

Usage:
    python test_fault_tolerant_cross_browser_model_sharding.py --model bert-base-uncased --shards 3 --type layer --fault-tolerance-level high
    python test_fault_tolerant_cross_browser_model_sharding.py --model whisper-tiny --shards 2 --type component --model-type audio --fail-browser firefox
    python test_fault_tolerant_cross_browser_model_sharding.py --model vit-base-patch16-224 --shards 3 --type layer --model-type vision --comprehensive
"""

import os
import sys
import json
import time
import argparse
import anyio
import logging
import random
from pathlib import Path
import datetime
import traceback
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

# Import required modules
try:
    from fixed_web_platform.cross_browser_model_sharding import ModelShardingManager
    SHARDING_AVAILABLE = True
except ImportError as e:
    logger.error(f"CrossBrowserModelSharding not available: {e}")
    SHARDING_AVAILABLE = False

# Import resource pool bridge extensions if available
try:
    from fixed_web_platform.resource_pool_bridge_integration import ResourcePoolBridgeIntegration
    RESOURCE_POOL_AVAILABLE = True
except ImportError as e:
    logger.error(f"ResourcePoolBridgeIntegration not available: {e}")
    RESOURCE_POOL_AVAILABLE = False

# Import browser performance history if available
try:
    from fixed_web_platform.browser_performance_history import BrowserPerformanceHistory
    PERFORMANCE_HISTORY_AVAILABLE = True
except ImportError as e:
    logger.error(f"BrowserPerformanceHistory not available: {e}")
    PERFORMANCE_HISTORY_AVAILABLE = False

# Import distributed testing framework if available
try:
    from distributed_testing.test_coordinator import TestCoordinator
    DISTRIBUTED_TESTING_AVAILABLE = True
except ImportError as e:
    logger.error(f"TestCoordinator not available: {e}")
    DISTRIBUTED_TESTING_AVAILABLE = False

# Model family mapping for test inputs
MODEL_FAMILY_MAP = {
    "text": ["bert-base-uncased", "roberta-base", "gpt2", "t5-small"],
    "vision": ["vit-base-patch16-224", "resnet-50", "deit-base-distilled-patch16-224"],
    "audio": ["whisper-tiny", "wav2vec2-base", "hubert-base"],
    "multimodal": ["clip-vit-base-patch32", "flava-base", "blip-base"]
}

# Define sharding strategies
SHARDING_STRATEGIES = ["layer", "attention_feedforward", "component"]

# Define fault tolerance levels
FAULT_TOLERANCE_LEVELS = ["none", "low", "medium", "high", "critical"]

# Define recovery strategies
RECOVERY_STRATEGIES = ["simple", "progressive", "parallel", "coordinated"]

def get_model_input(model_type: str, sequence_length: int = 10) -> Dict[str, Any]:
    """Get appropriate test input based on model type"""
    if model_type == "text" or model_type == "text_embedding":
        # Generate appropriate sequence length for text models
        return {
            'input_ids': [101] + [2000 + i for i in range(sequence_length)] + [102],
            'attention_mask': [1] * (sequence_length + 2)
        }
    elif model_type == "vision":
        # Create a small image tensor (batch_size x channels x height x width)
        return {'pixel_values': [[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(1)]}
    elif model_type == "audio":
        # Create a small audio tensor (batch_size x time x features)
        return {'input_features': [[[0.1 for _ in range(80)] for _ in range(3000)]]}
    elif model_type == "multimodal":
        # Create combined text and image input
        return {
            'input_ids': [101] + [2000 + i for i in range(sequence_length)] + [102],
            'attention_mask': [1] * (sequence_length + 2),
            'pixel_values': [[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(1)]
        }
    else:
        # Generic input for unknown model types
        return {'inputs': [0.0 for _ in range(10)]}

async def setup_resource_pool(args) -> Optional[ResourcePoolBridgeIntegration]:
    """Set up resource pool integration if available"""
    if not RESOURCE_POOL_AVAILABLE:
        logger.warning("ResourcePoolBridgeIntegration not available, skipping integration")
        return None

    try:
        # Configure browser preferences based on model type
        browser_preferences = {}
        if args.model_type == "audio":
            browser_preferences["audio"] = "firefox"
        elif args.model_type == "vision":
            browser_preferences["vision"] = "chrome"
        elif args.model_type == "text" or args.model_type == "text_embedding":
            browser_preferences["text_embedding"] = "edge"
        
        # Create resource pool with fault tolerance options
        pool = ResourcePoolBridgeIntegration(
            max_connections=args.max_connections,
            browser_preferences=browser_preferences,
            adaptive_scaling=args.adaptive_scaling,
            enable_fault_tolerance=args.fault_tolerance,
            fault_tolerance_options={
                "level": args.fault_tolerance_level,
                "recovery_strategy": args.recovery_strategy,
                "checkpoint_interval": args.checkpoint_interval,
                "max_recovery_attempts": args.max_retries,
                "browser_health_check_interval": args.health_check_interval
            }
        )

        # Initialize the resource pool
        await pool.initialize()
        logger.info(f"Resource pool initialized with {args.max_connections} max connections")
        return pool
    except Exception as e:
        logger.error(f"Error setting up resource pool: {e}")
        return None

async def setup_performance_history(args) -> Optional[BrowserPerformanceHistory]:
    """Set up browser performance history if available"""
    if not PERFORMANCE_HISTORY_AVAILABLE or not args.use_performance_history:
        return None

    try:
        # Create performance history tracker with database integration
        history = BrowserPerformanceHistory(
            db_path=args.db_path if args.db_path else None,
            max_entries=args.max_history_entries,
            update_interval=args.history_update_interval
        )
        await history.initialize()
        logger.info(f"Browser performance history tracker initialized")
        return history
    except Exception as e:
        logger.error(f"Error setting up performance history: {e}")
        return None

async def simulate_browser_failure(manager, browser_index, args) -> Dict[str, Any]:
    """Simulate different types of browser failures"""
    if not args.fault_tolerance:
        logger.warning("Fault tolerance not enabled, skipping failure simulation")
        return {"simulated": False, "reason": "fault_tolerance_disabled"}

    if browser_index >= args.shards:
        logger.warning(f"Invalid browser index {browser_index}, must be < {args.shards}")
        return {"simulated": False, "reason": "invalid_browser_index"}

    logger.info(f"Simulating failure for browser {browser_index}")
    failure_type = args.failure_type if args.failure_type else random.choice(
        ["connection_lost", "browser_crash", "memory_pressure", "timeout"]
    )
    
    try:
        # Get the browser allocation to identify which browser to fail
        metrics = manager.get_metrics()
        if not metrics or "browser_allocation" not in metrics:
            logger.warning("Cannot get browser allocation from metrics")
            return {"simulated": False, "reason": "no_metrics_available"}
            
        # Get the browser type for this shard
        browser_allocation = metrics["browser_allocation"]
        shard_info = browser_allocation.get(str(browser_index))
        if not shard_info:
            logger.warning(f"No browser info for shard {browser_index}")
            return {"simulated": False, "reason": "no_browser_info"}
            
        browser_type = shard_info.get("browser", "unknown")
        
        start_time = time.time()
        
        # Apply appropriate failure based on failure_type
        if failure_type == "connection_lost":
            # Simulate connection loss by calling internal failure handler
            await manager._handle_connection_failure(browser_index)
            
        elif failure_type == "browser_crash":
            # Simulate browser crash by invalidating the browser instance
            await manager._simulate_browser_crash(browser_index)
            
        elif failure_type == "memory_pressure":
            # Simulate memory pressure
            await manager._simulate_memory_pressure(browser_index, level=0.85)
            
        elif failure_type == "timeout":
            # Simulate operation timeout
            await manager._simulate_operation_timeout(browser_index)
            
        elapsed_time = time.time() - start_time
        
        logger.info(f"Simulated {failure_type} failure for browser {browser_index} ({browser_type}) in {elapsed_time:.2f}s")
        
        # If delay is specified, wait before continuing
        if args.failure_delay > 0:
            logger.info(f"Waiting {args.failure_delay}s before continuing")
            await anyio.sleep(args.failure_delay)
            
        return {
            "simulated": True,
            "browser_index": browser_index,
            "browser_type": browser_type,
            "failure_type": failure_type,
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        logger.error(f"Error simulating browser failure: {e}")
        return {"simulated": False, "reason": str(e)}

async def test_model_sharding(args) -> Dict[str, Any]:
    """Comprehensive test for fault-tolerant cross-browser model sharding"""
    if not SHARDING_AVAILABLE:
        logger.error("Cannot test model sharding: Cross-browser model sharding not available")
        return {"status": "error", "reason": "sharding_not_available"}
    
    # Track overall test results
    test_results = {
        "model_name": args.model,
        "model_type": args.model_type,
        "shard_count": args.shards,
        "shard_type": args.type,
        "fault_tolerance": {
            "enabled": args.fault_tolerance,
            "level": args.fault_tolerance_level,
            "recovery_strategy": args.recovery_strategy
        },
        "test_parameters": vars(args),
        "start_time": datetime.datetime.now().isoformat(),
        "status": "initialized",
        "phases": {}
    }
    
    try:
        # Phase 1: Setup
        phase_start = time.time()
        test_results["phases"]["setup"] = {"status": "running"}
        
        # Set environment variables for optimizations
        if args.compute_shaders:
            os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
            logger.info("Enabled compute shader optimization")
        
        if args.shader_precompile:
            os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1"
            logger.info("Enabled shader precompilation")
        
        if args.parallel_loading:
            os.environ["WEB_PARALLEL_LOADING_ENABLED"] = "1"
            logger.info("Enabled parallel model loading")
            
        # Setup resource pool if integration enabled
        resource_pool = None
        if args.resource_pool_integration:
            resource_pool = await setup_resource_pool(args)
            if not resource_pool:
                logger.warning("Could not set up resource pool, continuing without integration")
                
        # Setup performance history if enabled
        performance_history = None
        if args.use_performance_history:
            performance_history = await setup_performance_history(args)
            if not performance_history:
                logger.warning("Could not set up performance history, continuing without it")
                
        # Create model sharding manager with appropriate options
        manager_args = {
            "model_name": args.model,
            "num_shards": args.shards,
            "shard_type": args.type,
            "model_type": args.model_type,
            "enable_fault_tolerance": args.fault_tolerance,
            "fault_tolerance_level": args.fault_tolerance_level,
            "recovery_strategy": args.recovery_strategy,
            "max_retries": args.max_retries,
            "timeout": args.timeout,
            "enable_ipfs": not args.disable_ipfs,
            "db_path": args.db_path
        }
        
        # Add resource pool integration if available
        if resource_pool:
            manager_args["resource_pool"] = resource_pool
            
        # Add performance history if available
        if performance_history:
            manager_args["performance_history"] = performance_history
            
        # Create the model sharding manager
        manager = ModelShardingManager(**manager_args)
        
        # Complete setup phase
        test_results["phases"]["setup"]["duration"] = time.time() - phase_start
        test_results["phases"]["setup"]["status"] = "completed"
        logger.info(f"Setup phase completed in {test_results['phases']['setup']['duration']:.2f}s")
        
        # Phase 2: Initialization
        phase_start = time.time()
        test_results["phases"]["initialization"] = {"status": "running"}
        
        # Initialize sharding with timeout protection
        logger.info(f"Initializing sharding for {args.model} with {args.shards} shards")
        logger.info(f"Using shard type: {args.type}, model type: {args.model_type}")
        
        try:
            # Use asyncio.wait_for to add timeout protection
            initialized = await # TODO: Replace with anyio.fail_after - asyncio.wait_for(
                manager.initialize_sharding(),
                timeout=args.timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Initialization timeout after {args.timeout}s")
            test_results["phases"]["initialization"]["status"] = "failed"
            test_results["phases"]["initialization"]["reason"] = "timeout"
            test_results["status"] = "failed"
            return test_results
        
        if not initialized:
            logger.error("Failed to initialize model sharding")
            test_results["phases"]["initialization"]["status"] = "failed"
            test_results["phases"]["initialization"]["reason"] = "initialization_failed"
            test_results["status"] = "failed"
            return test_results
        
        logger.info("✅ Model sharding initialized successfully")
        test_results["phases"]["initialization"]["duration"] = time.time() - phase_start
        test_results["phases"]["initialization"]["status"] = "completed"
        
        # Get and store initialization metrics
        init_metrics = manager.get_initialization_metrics()
        test_results["initialization_metrics"] = init_metrics
        logger.info(f"Initialization completed in {test_results['phases']['initialization']['duration']:.2f}s")
        
        # Phase 3: Preliminary Inference (pre-failure)
        phase_start = time.time()
        test_results["phases"]["pre_failure_inference"] = {"status": "running"}
        
        # Get model input based on model type
        sample_input = get_model_input(args.model_type, sequence_length=args.sequence_length)
        
        # Run initial inference with timeout protection
        logger.info(f"Running pre-failure inference for {args.model}")
        try:
            # Use asyncio.wait_for to add timeout protection
            start_time = time.time()
            result = await # TODO: Replace with anyio.fail_after - asyncio.wait_for(
                manager.run_inference_sharded(sample_input),
                timeout=args.timeout
            )
            execution_time = time.time() - start_time
        except asyncio.TimeoutError:
            logger.error(f"Pre-failure inference timeout after {args.timeout}s")
            test_results["phases"]["pre_failure_inference"]["status"] = "failed"
            test_results["phases"]["pre_failure_inference"]["reason"] = "timeout"
            test_results["status"] = "failed"
            return test_results
        
        # Check inference result
        if 'error' in result:
            logger.error(f"❌ Pre-failure inference error: {result['error']}")
            test_results["phases"]["pre_failure_inference"]["status"] = "failed"
            test_results["phases"]["pre_failure_inference"]["reason"] = result['error']
            test_results["status"] = "failed"
            return test_results
        
        logger.info(f"✅ Pre-failure inference successful in {execution_time:.2f}s")
        test_results["phases"]["pre_failure_inference"]["duration"] = time.time() - phase_start
        test_results["phases"]["pre_failure_inference"]["status"] = "completed"
        test_results["phases"]["pre_failure_inference"]["execution_time"] = execution_time
        test_results["phases"]["pre_failure_inference"]["result"] = result
        
        # Get and store pre-failure metrics
        pre_failure_metrics = manager.get_metrics()
        test_results["pre_failure_metrics"] = pre_failure_metrics
        logger.info(f"Pre-failure inference completed in {test_results['phases']['pre_failure_inference']['duration']:.2f}s")
        
        # Phase 4: Failure Simulation (if enabled)
        if args.fault_tolerance and args.simulate_failure:
            phase_start = time.time()
            test_results["phases"]["failure_simulation"] = {"status": "running"}
            
            # Determine which browser/shard to fail
            browser_to_fail = args.fail_shard if args.fail_shard is not None else random.randint(0, args.shards - 1)
            
            # Simulate browser failure
            failure_result = await simulate_browser_failure(manager, browser_to_fail, args)
            test_results["phases"]["failure_simulation"]["result"] = failure_result
            
            if not failure_result["simulated"]:
                logger.warning(f"Failed to simulate browser failure: {failure_result.get('reason', 'unknown')}")
                test_results["phases"]["failure_simulation"]["status"] = "warning"
            else:
                logger.info(f"Successfully simulated failure for browser {browser_to_fail}")
                test_results["phases"]["failure_simulation"]["status"] = "completed"
                
            test_results["phases"]["failure_simulation"]["duration"] = time.time() - phase_start
            logger.info(f"Failure simulation completed in {test_results['phases']['failure_simulation']['duration']:.2f}s")
            
        # Phase 5: Post-Failure Inference
        if args.fault_tolerance and args.simulate_failure:
            phase_start = time.time()
            test_results["phases"]["post_failure_inference"] = {"status": "running"}
            
            # Run post-failure inference with timeout protection
            logger.info(f"Running post-failure inference for {args.model}")
            try:
                # Use asyncio.wait_for to add timeout protection
                start_time = time.time()
                result = await # TODO: Replace with anyio.fail_after - asyncio.wait_for(
                    manager.run_inference_sharded(sample_input),
                    timeout=args.timeout
                )
                execution_time = time.time() - start_time
            except asyncio.TimeoutError:
                logger.error(f"Post-failure inference timeout after {args.timeout}s")
                test_results["phases"]["post_failure_inference"]["status"] = "failed"
                test_results["phases"]["post_failure_inference"]["reason"] = "timeout"
                test_results["status"] = "failed"
                return test_results
            
            # Check inference result
            if 'error' in result:
                logger.error(f"❌ Post-failure inference error: {result['error']}")
                test_results["phases"]["post_failure_inference"]["status"] = "failed"
                test_results["phases"]["post_failure_inference"]["reason"] = result['error']
                test_results["status"] = "failed"
                return test_results
            
            logger.info(f"✅ Post-failure inference successful in {execution_time:.2f}s")
            test_results["phases"]["post_failure_inference"]["duration"] = time.time() - phase_start
            test_results["phases"]["post_failure_inference"]["status"] = "completed"
            test_results["phases"]["post_failure_inference"]["execution_time"] = execution_time
            test_results["phases"]["post_failure_inference"]["result"] = result
            
            # Get and store post-failure metrics
            post_failure_metrics = manager.get_metrics()
            test_results["post_failure_metrics"] = post_failure_metrics
            logger.info(f"Post-failure inference completed in {test_results['phases']['post_failure_inference']['duration']:.2f}s")
            
            # Extract and analyze recovery metrics
            if "recovery_metrics" in post_failure_metrics:
                recovery_metrics = post_failure_metrics["recovery_metrics"]
                test_results["recovery_metrics"] = recovery_metrics
                logger.info(f"Recovery metrics: {json.dumps(recovery_metrics, indent=2)}")
            
        # Phase 6: Performance Testing (if enabled)
        if args.performance_test:
            phase_start = time.time()
            test_results["phases"]["performance_testing"] = {"status": "running"}
            
            # Run multiple iterations for performance testing
            logger.info(f"Running performance test with {args.iterations} iterations")
            performance_results = []
            
            for i in range(args.iterations):
                try:
                    start_time = time.time()
                    result = await # TODO: Replace with anyio.fail_after - asyncio.wait_for(
                        manager.run_inference_sharded(sample_input),
                        timeout=args.timeout
                    )
                    iteration_time = time.time() - start_time
                    
                    if 'error' not in result:
                        performance_results.append({
                            "iteration": i + 1,
                            "execution_time": iteration_time,
                            "metrics": result.get("metrics", {})
                        })
                        logger.info(f"Iteration {i+1}/{args.iterations}: {iteration_time:.3f}s")
                    else:
                        logger.error(f"Error in iteration {i+1}: {result['error']}")
                except asyncio.TimeoutError:
                    logger.error(f"Timeout in iteration {i+1}")
                except Exception as e:
                    logger.error(f"Error in iteration {i+1}: {e}")
                    
                # Wait between iterations if specified
                if args.iteration_delay > 0 and i < args.iterations - 1:
                    await anyio.sleep(args.iteration_delay)
            
            # Calculate performance statistics
            if performance_results:
                execution_times = [r["execution_time"] for r in performance_results]
                avg_time = sum(execution_times) / len(execution_times)
                min_time = min(execution_times)
                max_time = max(execution_times)
                
                test_results["phases"]["performance_testing"]["statistics"] = {
                    "avg_execution_time": avg_time,
                    "min_execution_time": min_time,
                    "max_execution_time": max_time,
                    "std_dev": (sum((t - avg_time) ** 2 for t in execution_times) / len(execution_times)) ** 0.5,
                    "successful_iterations": len(performance_results),
                    "total_iterations": args.iterations
                }
                
                test_results["phases"]["performance_testing"]["iterations"] = performance_results
                logger.info(f"Performance test completed: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")
            
            test_results["phases"]["performance_testing"]["duration"] = time.time() - phase_start
            test_results["phases"]["performance_testing"]["status"] = "completed"
            
        # Phase 7: Stress Testing (if enabled)
        if args.stress_test:
            phase_start = time.time()
            test_results["phases"]["stress_testing"] = {"status": "running"}
            
            # Run concurrent requests for stress testing
            logger.info(f"Running stress test with {args.concurrent_requests} concurrent requests for {args.stress_duration}s")
            
            async def run_single_stress_request(request_id):
                try:
                    start_time = time.time()
                    result = await manager.run_inference_sharded(sample_input)
                    execution_time = time.time() - start_time
                    return {
                        "request_id": request_id,
                        "success": 'error' not in result,
                        "execution_time": execution_time,
                        "error": result.get('error', None)
                    }
                except Exception as e:
                    return {
                        "request_id": request_id,
                        "success": False,
                        "error": str(e)
                    }
            
            # Function to generate concurrent requests
            async def generate_requests():
                stress_results = []
                start_time = time.time()
                request_count = 0
                
                while time.time() - start_time < args.stress_duration:
                    # Create a batch of concurrent requests
                    batch_size = min(args.concurrent_requests, 10)  # Limit batch size to avoid overwhelming system
                    batch_tasks = []
                    
                    for i in range(batch_size):
                        request_id = request_count + i + 1
                        task = # TODO: Replace with task group - asyncio.create_task(run_single_stress_request(request_id))
                        batch_tasks.append(task)
                    
                    # Wait for all tasks in batch to complete
                    batch_results = await # TODO: Replace with task group - asyncio.gather(*batch_tasks, return_exceptions=True)
                    stress_results.extend([r for r in batch_results if not isinstance(r, Exception)])
                    
                    request_count += batch_size
                    logger.info(f"Completed {request_count} stress test requests")
                    
                    # Check if we've reached the duration limit
                    if time.time() - start_time >= args.stress_duration:
                        break
                    
                    # Short delay between batches to prevent system overload
                    await anyio.sleep(0.1)
                
                return stress_results
            
            # Run stress test
            stress_results = await generate_requests()
            
            # Calculate stress test statistics
            if stress_results:
                successful_requests = [r for r in stress_results if r.get("success", False)]
                failed_requests = [r for r in stress_results if not r.get("success", False)]
                
                if successful_requests:
                    execution_times = [r.get("execution_time", 0) for r in successful_requests]
                    avg_time = sum(execution_times) / len(execution_times)
                    min_time = min(execution_times)
                    max_time = max(execution_times)
                else:
                    avg_time = min_time = max_time = 0
                
                test_results["phases"]["stress_testing"]["statistics"] = {
                    "total_requests": len(stress_results),
                    "successful_requests": len(successful_requests),
                    "failed_requests": len(failed_requests),
                    "success_rate": len(successful_requests) / max(1, len(stress_results)),
                    "avg_execution_time": avg_time,
                    "min_execution_time": min_time,
                    "max_execution_time": max_time
                }
                
                logger.info(f"Stress test: {len(successful_requests)}/{len(stress_results)} successful requests, avg time: {avg_time:.3f}s")
            
            test_results["phases"]["stress_testing"]["duration"] = time.time() - phase_start
            test_results["phases"]["stress_testing"]["status"] = "completed"
        
        # Phase 8: Cleanup
        phase_start = time.time()
        test_results["phases"]["cleanup"] = {"status": "running"}
        
        # Get final metrics
        final_metrics = manager.get_metrics()
        test_results["final_metrics"] = final_metrics
        
        # Close manager
        await manager.close()
        logger.info("Model sharding manager closed")
        
        # Clean up resource pool if used
        if resource_pool:
            await resource_pool.close()
            logger.info("Resource pool closed")
        
        test_results["phases"]["cleanup"]["duration"] = time.time() - phase_start
        test_results["phases"]["cleanup"]["status"] = "completed"
        
        # Calculate overall test results
        test_results["end_time"] = datetime.datetime.now().isoformat()
        test_results["total_duration"] = sum(phase["duration"] for phase in test_results["phases"].values() if "duration" in phase)
        test_results["status"] = "completed"
        
        logger.info(f"Test completed successfully in {test_results['total_duration']:.2f}s")
        
        return test_results
        
    except Exception as e:
        logger.error(f"Error testing model sharding: {e}")
        traceback.print_exc()
        
        # Record the error in test results
        test_results["status"] = "error"
        test_results["error"] = str(e)
        test_results["error_traceback"] = traceback.format_exc()
        test_results["end_time"] = datetime.datetime.now().isoformat()
        
        return test_results

async def save_test_results(results: Dict[str, Any], args) -> None:
    """Save test results to file"""
    if not args.output:
        return
    
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Add timestamp to results
        results["saved_at"] = datetime.datetime.now().isoformat()
        
        # Write results to file
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Test results saved to {args.output}")
    except Exception as e:
        logger.error(f"Error saving test results: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Advanced Fault-Tolerant Cross-Browser Model Sharding Test")
    
    # Model selection options
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                       help="Model name to shard")
    parser.add_argument("--model-type", type=str, default="text",
                       choices=["text", "vision", "audio", "multimodal", "text_embedding"],
                       help="Type of model")
    parser.add_argument("--list-models", type=str, choices=["text", "vision", "audio", "multimodal", "text_embedding"],
                       help="List supported models for a model type")
    
    # Sharding options
    parser.add_argument("--shards", type=int, default=3,
                       help="Number of shards to create")
    parser.add_argument("--type", type=str, default="layer",
                       choices=SHARDING_STRATEGIES,
                       help="Type of sharding to use")
    parser.add_argument("--sequence-length", type=int, default=10,
                       help="Sequence length for text inputs")
    
    # Fault tolerance options
    parser.add_argument("--fault-tolerance", action="store_true",
                       help="Enable fault tolerance features")
    parser.add_argument("--fault-tolerance-level", type=str, default="medium",
                       choices=FAULT_TOLERANCE_LEVELS,
                       help="Fault tolerance level")
    parser.add_argument("--recovery-strategy", type=str, default="progressive",
                       choices=RECOVERY_STRATEGIES,
                       help="Recovery strategy to use")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retry attempts for recovery")
    parser.add_argument("--checkpoint-interval", type=int, default=60,
                       help="Interval in seconds between state checkpoints")
    parser.add_argument("--health-check-interval", type=int, default=30,
                       help="Interval in seconds between browser health checks")
    
    # Failure simulation options
    parser.add_argument("--simulate-failure", action="store_true",
                       help="Simulate browser failure during test")
    parser.add_argument("--fail-shard", type=int,
                       help="Specific shard to fail (default: random)")
    parser.add_argument("--failure-type", type=str,
                       choices=["connection_lost", "browser_crash", "memory_pressure", "timeout"],
                       help="Type of failure to simulate (default: random)")
    parser.add_argument("--failure-delay", type=float, default=0,
                       help="Delay in seconds after failure before continuing")
    parser.add_argument("--cascade-failures", action="store_true",
                       help="Simulate cascading failures")
    
    # Performance testing options
    parser.add_argument("--performance-test", action="store_true",
                       help="Run performance tests with multiple iterations")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of iterations for performance testing")
    parser.add_argument("--iteration-delay", type=float, default=0.5,
                       help="Delay in seconds between iterations")
    
    # Stress testing options
    parser.add_argument("--stress-test", action="store_true",
                       help="Run stress test with concurrent requests")
    parser.add_argument("--concurrent-requests", type=int, default=10,
                       help="Number of concurrent requests for stress testing")
    parser.add_argument("--stress-duration", type=int, default=60,
                       help="Duration in seconds for stress testing")
    
    # Resource pool options
    parser.add_argument("--resource-pool-integration", action="store_true",
                       help="Enable integration with resource pool")
    parser.add_argument("--max-connections", type=int, default=4,
                       help="Maximum number of browser connections")
    parser.add_argument("--adaptive-scaling", action="store_true",
                       help="Enable adaptive scaling of browser resources")
    
    # Performance history options
    parser.add_argument("--use-performance-history", action="store_true",
                       help="Enable browser performance history tracking")
    parser.add_argument("--max-history-entries", type=int, default=1000,
                       help="Maximum number of history entries to keep")
    parser.add_argument("--history-update-interval", type=int, default=60,
                       help="Interval in seconds for history updates")
    
    # General options
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout in seconds for initialization and inference")
    parser.add_argument("--db-path", type=str,
                       help="Path to DuckDB database for storing results")
    parser.add_argument("--compute-shaders", action="store_true",
                       help="Enable compute shader optimization for audio models")
    parser.add_argument("--shader-precompile", action="store_true",
                       help="Enable shader precompilation for faster startup")
    parser.add_argument("--parallel-loading", action="store_true",
                       help="Enable parallel model loading for multimodal models")
    parser.add_argument("--disable-ipfs", action="store_true",
                       help="Disable IPFS acceleration")
    parser.add_argument("--all-optimizations", action="store_true",
                       help="Enable all optimizations")
    
    # Output options
    parser.add_argument("--output", type=str,
                       help="Path to output file for test results (JSON)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    # Comprehensive test mode
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive test suite with all validations")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # List models if requested
    if args.list_models:
        models = MODEL_FAMILY_MAP.get(args.list_models, [])
        print(f"Supported models for {args.list_models}:")
        for model in models:
            print(f"  - {model}")
        return 0
    
    # Configure for comprehensive testing
    if args.comprehensive:
        args.fault_tolerance = True
        args.simulate_failure = True
        args.performance_test = True
        args.resource_pool_integration = True
        args.use_performance_history = True
        args.all_optimizations = True
        
        logger.info("Running in comprehensive test mode")
    
    # Handle all optimizations flag
    if args.all_optimizations:
        args.compute_shaders = True
        args.shader_precompile = True
        args.parallel_loading = True
    
    # Set browser-specific optimizations based on model type
    if args.model_type == "audio" and not args.all_optimizations:
        args.compute_shaders = True
        os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
        logger.info("Enabled Firefox compute shader optimizations for audio model")
    
    if args.model_type == "vision" and not args.all_optimizations:
        args.shader_precompile = True
        logger.info("Enabled shader precompilation for vision model")
    
    if args.model_type == "multimodal" and not args.all_optimizations:
        args.parallel_loading = True
        logger.info("Enabled parallel loading for multimodal model")
    
    # Print test configuration
    logger.info(f"Testing {args.model} ({args.model_type}) with {args.shards} shards using {args.type} sharding strategy")
    if args.fault_tolerance:
        logger.info(f"Fault tolerance level: {args.fault_tolerance_level}, recovery strategy: {args.recovery_strategy}")
    
    try:
        # Run the test and get results
        test_results = anyio.run(test_model_sharding(args))
        
        # Save results if output path specified
        if args.output:
            anyio.run(save_test_results(test_results, args))
        
        # Determine exit code based on test status
        if test_results["status"] == "completed":
            logger.info("Test completed successfully")
            return 0
        elif test_results["status"] == "failed":
            logger.error(f"Test failed: {test_results.get('reason', 'unknown reason')}")
            return 1
        else:
            logger.error(f"Test error: {test_results.get('error', 'unknown error')}")
            return 2
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())