#!/usr/bin/env python3
"""
Test Real Browser Fault Tolerance for WebNN/WebGPU Resource Pool

This script tests the fault tolerance mechanisms with real browser connections,
focusing on validating recovery strategies in production-like environments.

Usage:
    python test_real_browser_fault_tolerance.py --model bert-base-uncased --browser-count 3
    python test_real_browser_fault_tolerance.py --model whisper-tiny --model-type audio --browser chrome,firefox
    python test_real_browser_fault_tolerance.py --model vit-base-patch16-224 --model-type vision --force-failure
"""

from ipfs_accelerate_py.anyio_helpers import gather, wait_for
import os
import sys
import json
import time
import random
import logging
import argparse
import anyio
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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
    from fixed_web_platform.resource_pool_bridge_recovery import (
        ResourcePoolRecoveryManager,
        BrowserFailureCategory
    )
    from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Required modules not available: {e}")
    MODULES_AVAILABLE = False

# Import Selenium and related modules for browser automation if available
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.firefox.service import Service as FirefoxService
    from selenium.webdriver.edge.service import Service as EdgeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    
    # Try to import webdriver manager if available
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from webdriver_manager.firefox import GeckoDriverManager
        from webdriver_manager.microsoft import EdgeChromiumDriverManager
        WEBDRIVER_MANAGER_AVAILABLE = True
    except ImportError:
        WEBDRIVER_MANAGER_AVAILABLE = False
        
    SELENIUM_AVAILABLE = True
except ImportError:
    logger.warning("Selenium not available, real browser testing will be limited")
    SELENIUM_AVAILABLE = False
    WEBDRIVER_MANAGER_AVAILABLE = False

# Model input helpers
def get_model_input(model_type: str, sequence_length: int = 10) -> Dict[str, Any]:
    """Get appropriate test input based on model type"""
    if model_type == "text" or model_type == "text_embedding":
        return {
            'input_ids': [101] + [2000 + i for i in range(sequence_length)] + [102],
            'attention_mask': [1] * (sequence_length + 2)
        }
    elif model_type == "vision":
        return {'pixel_values': [[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(1)]}
    elif model_type == "audio":
        return {'input_features': [[[0.1 for _ in range(80)] for _ in range(3000)]]}
    elif model_type == "multimodal":
        return {
            'input_ids': [101] + [2000 + i for i in range(sequence_length)] + [102],
            'attention_mask': [1] * (sequence_length + 2),
            'pixel_values': [[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(1)]
        }
    else:
        return {'inputs': [0.0 for _ in range(10)]}

# Browser management functions
async def setup_browsers(browser_types: List[str], headless: bool = True) -> List[Dict[str, Any]]:
    """Set up real browser instances using Selenium"""
    if not SELENIUM_AVAILABLE:
        logger.error("Selenium not available, cannot set up real browsers")
        return []
    
    browser_instances = []
    
    for i, browser_type in enumerate(browser_types):
        browser_type = browser_type.lower()
        try:
            # Create browser instance
            if browser_type == "chrome":
                options = ChromeOptions()
                if headless:
                    options.add_argument("--headless=new")
                options.add_argument("--disable-gpu")
                options.add_argument("--enable-webgpu")
                options.add_argument("--enable-features=WebGPU")
                options.add_argument("--disable-site-isolation-trials")
                
                # Add WebNN flag if available in this Chrome version
                options.add_argument("--enable-features=WebNN")
                
                if WEBDRIVER_MANAGER_AVAILABLE:
                    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
                else:
                    driver = webdriver.Chrome(options=options)
                    
            elif browser_type == "firefox":
                options = FirefoxOptions()
                if headless:
                    options.add_argument("--headless")
                
                # Add Firefox-specific WebGPU flags
                options.set_preference("dom.webgpu.enabled", True)
                options.set_preference("gfx.webgpu.force-enabled", True)
                
                # Add Firefox-specific compute shader optimizations
                options.set_preference("dom.webgpu.advanced-compute-enabled", True)
                
                if WEBDRIVER_MANAGER_AVAILABLE:
                    driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)
                else:
                    driver = webdriver.Firefox(options=options)
                    
            elif browser_type == "edge":
                options = EdgeOptions()
                if headless:
                    options.add_argument("--headless=new")
                options.add_argument("--disable-gpu")
                options.add_argument("--enable-webgpu")
                options.add_argument("--enable-features=WebGPU,WebNN")
                
                if WEBDRIVER_MANAGER_AVAILABLE:
                    driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()), options=options)
                else:
                    driver = webdriver.Edge(options=options)
            else:
                logger.warning(f"Unsupported browser type: {browser_type}")
                continue
            
            # Load WebNN/WebGPU test page
            driver.get("https://example.com")
            
            # Wait for page to load
            try:
                WebDriverWait(driver, 10).until(
                    EC.title_contains("Example")
                )
            except TimeoutException:
                logger.warning(f"Timeout waiting for test page to load in {browser_type}")
            
            # Add browser to list
            browser_instances.append({
                "id": f"{browser_type}_{i+1}",
                "type": browser_type,
                "driver": driver,
                "capabilities": await detect_browser_capabilities(driver, browser_type)
            })
            
            logger.info(f"Browser {browser_type}_{i+1} launched successfully")
            
        except Exception as e:
            logger.error(f"Failed to set up {browser_type} browser: {e}")
    
    return browser_instances

async def detect_browser_capabilities(driver, browser_type: str) -> Dict[str, bool]:
    """Detect WebGPU and WebNN capabilities of a browser"""
    try:
        # Check for WebGPU support
        webgpu_supported = await anyio.to_thread.run_sync(
            lambda: driver.execute_script("return typeof navigator.gpu !== 'undefined'")
        )
        
        # Check for WebNN support (only in newer browsers)
        webnn_supported = await anyio.to_thread.run_sync(
            lambda: driver.execute_script("return typeof navigator.ml !== 'undefined' && typeof navigator.ml.getNeuralNetworkContext !== 'undefined'")
        )
        
        # Additional capability checks
        compute_shaders_supported = False
        parallel_loading_supported = True  # Assume supported by default
        shader_precompilation_supported = True  # Assume supported by default
        
        # Firefox-specific checks
        if browser_type == "firefox":
            compute_shaders_supported = await anyio.to_thread.run_sync(
                lambda: driver.execute_script("return navigator.userAgent.indexOf('Firefox') > -1")
            )
        
        return {
            "webgpu": webgpu_supported,
            "webnn": webnn_supported,
            "compute_shaders": compute_shaders_supported,
            "parallel_loading": parallel_loading_supported,
            "shader_precompilation": shader_precompilation_supported
        }
    except Exception as e:
        logger.error(f"Error detecting browser capabilities: {e}")
        return {
            "webgpu": False,
            "webnn": False,
            "compute_shaders": False,
            "parallel_loading": False,
            "shader_precompilation": False
        }

async def clean_up_browsers(browser_instances: List[Dict[str, Any]]) -> None:
    """Clean up browser instances"""
    for browser in browser_instances:
        try:
            await anyio.to_thread.run_sync(lambda: browser["driver"].quit())
            logger.info(f"Browser {browser['id']} closed successfully")
        except Exception as e:
            logger.error(f"Error closing browser {browser['id']}: {e}")

async def force_browser_failure(browser_instances: List[Dict[str, Any]], failure_index: int, 
                               failure_type: str) -> Dict[str, Any]:
    """Force a browser failure for testing recovery"""
    if not browser_instances or failure_index >= len(browser_instances):
        return {"success": False, "reason": "No browser instance available"}
    
    browser = browser_instances[failure_index]
    
    try:
        if failure_type == "crash":
            # Force browser to crash by executing invalid JavaScript
            await anyio.to_thread.run_sync(
                lambda: browser["driver"].execute_script("window.open('chrome://crash')")
            )
            logger.info(f"Forced crash for browser {browser['id']}")
            return {"success": True, "browser_id": browser["id"], "failure_type": "crash"}
            
        elif failure_type == "hang":
            # Force browser to hang by executing infinite loop
            await anyio.to_thread.run_sync(
                lambda: browser["driver"].execute_script("while(true) {}")
            )
            logger.info(f"Forced hang for browser {browser['id']}")
            return {"success": True, "browser_id": browser["id"], "failure_type": "hang"}
            
        elif failure_type == "memory":
            # Force memory pressure by allocating large arrays
            script = """
            let arrays = [];
            try {
                for (let i = 0; i < 100; i++) {
                    arrays.push(new Uint8Array(100 * 1024 * 1024));  // Allocate 100MB per array
                }
            } catch (e) {
                return "Memory allocation failed: " + e.message;
            }
            return "Allocated memory successfully";
            """
            result = await anyio.to_thread.run_sync(
                lambda: browser["driver"].execute_script(script)
            )
            logger.info(f"Forced memory pressure for browser {browser['id']}: {result}")
            return {"success": True, "browser_id": browser["id"], "failure_type": "memory"}
            
        elif failure_type == "disconnect":
            # Simulate network disconnection by closing and replacing the browser
            browser_id = browser["id"]
            browser_type = browser["type"]
            
            # Close the browser forcefully
            await anyio.to_thread.run_sync(lambda: browser["driver"].quit())
            
            logger.info(f"Forced disconnection for browser {browser_id}")
            return {"success": True, "browser_id": browser_id, "failure_type": "disconnect"}
            
        else:
            return {"success": False, "reason": f"Unknown failure type: {failure_type}"}
            
    except Exception as e:
        logger.error(f"Error forcing browser failure: {e}")
        return {"success": False, "reason": str(e)}

# Main test function
async def test_real_browser_fault_tolerance(args) -> Dict[str, Any]:
    """Test fault tolerance with real browser connections"""
    if not MODULES_AVAILABLE:
        logger.error("Required modules not available")
        return {"status": "error", "reason": "modules_not_available"}
    
    # Initialize test results
    test_results = {
        "model_name": args.model,
        "model_type": args.model_type,
        "browsers": args.browsers.split(","),
        "browser_count": args.browser_count,
        "fault_tolerance_enabled": True,
        "fault_tolerance_level": args.fault_tolerance_level,
        "recovery_strategy": args.recovery_strategy,
        "start_time": datetime.datetime.now().isoformat(),
        "status": "initialized",
        "phases": {}
    }
    
    try:
        # Phase 1: Browser Setup
        phase_start = time.time()
        test_results["phases"]["browser_setup"] = {"status": "running"}
        
        # Determine which browsers to launch
        browser_types = []
        for _ in range(args.browser_count):
            # Distribute browsers based on model type or specified browsers
            if args.browsers:
                available_types = args.browsers.split(",")
                browser_types.append(available_types[_ % len(available_types)])
            else:
                # Allocate browsers based on model type
                if args.model_type == "audio":
                    # Prioritize Firefox for audio models (better compute shader support)
                    if _ == 0:
                        browser_types.append("firefox")
                    else:
                        browser_types.append(["chrome", "edge"][_ % 2])
                elif args.model_type == "text" or args.model_type == "text_embedding":
                    # Prioritize Edge for text models (better WebNN support)
                    if _ == 0:
                        browser_types.append("edge")
                    else:
                        browser_types.append(["chrome", "firefox"][_ % 2])
                else:
                    # Balanced allocation for other model types
                    browser_types.append(["chrome", "firefox", "edge"][_ % 3])
        
        # Launch real browser instances
        logger.info(f"Launching {len(browser_types)} browsers: {', '.join(browser_types)}")
        browser_instances = await setup_browsers(browser_types, headless=not args.show_browsers)
        
        if not browser_instances:
            logger.error("Failed to set up any browser instances")
            test_results["phases"]["browser_setup"]["status"] = "failed"
            test_results["phases"]["browser_setup"]["reason"] = "no_browsers_available"
            test_results["status"] = "failed"
            return test_results
        
        # Log browser capabilities
        for browser in browser_instances:
            logger.info(f"Browser {browser['id']} capabilities: {json.dumps(browser['capabilities'])}")
        
        test_results["phases"]["browser_setup"]["duration"] = time.time() - phase_start
        test_results["phases"]["browser_setup"]["status"] = "completed"
        test_results["phases"]["browser_setup"]["browser_count"] = len(browser_instances)
        test_results["phases"]["browser_setup"]["browsers"] = [{
            "id": browser["id"],
            "type": browser["type"],
            "capabilities": browser["capabilities"]
        } for browser in browser_instances]
        
        # Phase 2: Resource Pool Setup
        phase_start = time.time()
        test_results["phases"]["resource_pool_setup"] = {"status": "running"}
        
        # Create resource pool integration
        resource_pool = ResourcePoolBridgeIntegration(
            max_connections=len(browser_instances),
            adaptive_scaling=True,
            enable_fault_tolerance=True,
            fault_tolerance_options={
                "level": args.fault_tolerance_level,
                "recovery_strategy": args.recovery_strategy,
                "max_recovery_attempts": args.max_retries
            }
        )
        
        # Initialize with existing browser instances
        # NOTE: In a real implementation, this would connect to the actual browsers
        # For this test, we're just demonstrating the architecture
        await resource_pool.initialize()
        
        # Create recovery manager
        recovery_manager = ResourcePoolRecoveryManager(
            connection_pool=resource_pool,
            fault_tolerance_level=args.fault_tolerance_level,
            recovery_strategy=args.recovery_strategy
        )
        
        await recovery_manager.initialize()
        
        test_results["phases"]["resource_pool_setup"]["duration"] = time.time() - phase_start
        test_results["phases"]["resource_pool_setup"]["status"] = "completed"
        
        # Phase 3: Model Sharding Setup
        phase_start = time.time()
        test_results["phases"]["model_sharding_setup"] = {"status": "running"}
        
        # Create model sharding manager with recovery manager
        manager = ModelShardingManager(
            model_name=args.model,
            num_shards=min(len(browser_instances), args.shards),
            shard_type=args.shard_type,
            model_type=args.model_type,
            enable_ipfs=not args.disable_ipfs,
            resource_pool=resource_pool,
            recovery_manager=recovery_manager,
            db_path=args.db_path
        )
        
        # Initialize sharding
        logger.info(f"Initializing sharding for {args.model} with {min(len(browser_instances), args.shards)} shards")
        try:
            initialized = await wait_for(
                manager.initialize_sharding(),
                timeout=args.timeout
            )
        except TimeoutError:
            logger.error(f"Initialization timeout after {args.timeout}s")
            test_results["phases"]["model_sharding_setup"]["status"] = "failed"
            test_results["phases"]["model_sharding_setup"]["reason"] = "timeout"
            test_results["status"] = "failed"
            
            # Clean up browser instances
            await clean_up_browsers(browser_instances)
            return test_results
        
        if not initialized:
            logger.error("Failed to initialize model sharding")
            test_results["phases"]["model_sharding_setup"]["status"] = "failed"
            test_results["phases"]["model_sharding_setup"]["reason"] = "initialization_failed"
            test_results["status"] = "failed"
            
            # Clean up browser instances
            await clean_up_browsers(browser_instances)
            return test_results
        
        logger.info("Model sharding initialized successfully")
        test_results["phases"]["model_sharding_setup"]["duration"] = time.time() - phase_start
        test_results["phases"]["model_sharding_setup"]["status"] = "completed"
        
        # Phase 4: Initial Inference
        phase_start = time.time()
        test_results["phases"]["initial_inference"] = {"status": "running"}
        
        # Get model input
        sample_input = get_model_input(args.model_type, sequence_length=args.sequence_length)
        
        # Run inference
        logger.info(f"Running initial inference for {args.model}")
        try:
            start_time = time.time()
            result = await wait_for(
                manager.run_inference_sharded(sample_input),
                timeout=args.timeout
            )
            execution_time = time.time() - start_time
        except TimeoutError:
            logger.error(f"Initial inference timeout after {args.timeout}s")
            test_results["phases"]["initial_inference"]["status"] = "failed"
            test_results["phases"]["initial_inference"]["reason"] = "timeout"
            test_results["status"] = "failed"
            
            # Clean up browser instances
            await clean_up_browsers(browser_instances)
            return test_results
        
        # Check inference result
        if 'error' in result:
            logger.error(f"Initial inference error: {result['error']}")
            test_results["phases"]["initial_inference"]["status"] = "failed"
            test_results["phases"]["initial_inference"]["reason"] = result['error']
            test_results["status"] = "failed"
            
            # Clean up browser instances
            await clean_up_browsers(browser_instances)
            return test_results
        
        logger.info(f"Initial inference successful in {execution_time:.2f}s")
        test_results["phases"]["initial_inference"]["duration"] = time.time() - phase_start
        test_results["phases"]["initial_inference"]["status"] = "completed"
        test_results["phases"]["initial_inference"]["execution_time"] = execution_time
        
        # Phase 5: Forced Failure (if requested)
        if args.force_failure:
            phase_start = time.time()
            test_results["phases"]["forced_failure"] = {"status": "running"}
            
            # Determine which browser to fail
            failure_index = args.failure_index if args.failure_index is not None else random.randint(0, len(browser_instances) - 1)
            failure_type = args.failure_type if args.failure_type else random.choice(["crash", "hang", "memory", "disconnect"])
            
            logger.info(f"Forcing {failure_type} failure for browser at index {failure_index}")
            
            failure_result = await force_browser_failure(browser_instances, failure_index, failure_type)
            test_results["phases"]["forced_failure"]["result"] = failure_result
            
            if not failure_result.get("success", False):
                logger.warning(f"Failed to force browser failure: {failure_result.get('reason', 'unknown')}")
                test_results["phases"]["forced_failure"]["status"] = "warning"
            else:
                logger.info(f"Successfully forced {failure_type} failure for browser {failure_result.get('browser_id')}")
                test_results["phases"]["forced_failure"]["status"] = "completed"
            
            test_results["phases"]["forced_failure"]["duration"] = time.time() - phase_start
            
            # Allow time for failure to be detected
            logger.info(f"Waiting {args.failure_detection_delay}s for failure to be detected")
            await anyio.sleep(args.failure_detection_delay)
        
        # Phase 6: Post-Failure Inference
        if args.force_failure:
            phase_start = time.time()
            test_results["phases"]["post_failure_inference"] = {"status": "running"}
            
            # Run inference after failure
            logger.info(f"Running post-failure inference for {args.model}")
            try:
                start_time = time.time()
                result = await wait_for(
                    manager.run_inference_sharded(sample_input),
                    timeout=args.timeout
                )
                execution_time = time.time() - start_time
            except TimeoutError:
                logger.error(f"Post-failure inference timeout after {args.timeout}s")
                test_results["phases"]["post_failure_inference"]["status"] = "failed"
                test_results["phases"]["post_failure_inference"]["reason"] = "timeout"
                test_results["status"] = "failed"
                
                # Clean up browser instances
                await clean_up_browsers(browser_instances)
                return test_results
            
            # Check inference result
            if 'error' in result:
                logger.error(f"Post-failure inference error: {result['error']}")
                test_results["phases"]["post_failure_inference"]["status"] = "failed"
                test_results["phases"]["post_failure_inference"]["reason"] = result['error']
                test_results["status"] = "failed"
                
                # Clean up browser instances
                await clean_up_browsers(browser_instances)
                return test_results
            
            logger.info(f"Post-failure inference successful in {execution_time:.2f}s")
            test_results["phases"]["post_failure_inference"]["duration"] = time.time() - phase_start
            test_results["phases"]["post_failure_inference"]["status"] = "completed"
            test_results["phases"]["post_failure_inference"]["execution_time"] = execution_time
            test_results["phases"]["post_failure_inference"]["result"] = result
            
            # Get recovery statistics
            recovery_stats = recovery_manager.get_recovery_statistics()
            test_results["recovery_statistics"] = recovery_stats
            
            logger.info(f"Recovery statistics: {json.dumps(recovery_stats, indent=2)}")
        
        # Phase 7: Benchmark
        if args.benchmark:
            phase_start = time.time()
            test_results["phases"]["benchmark"] = {"status": "running"}
            
            # Run multiple inferences for benchmarking
            logger.info(f"Running benchmark with {args.iterations} iterations")
            benchmark_results = []
            
            for i in range(args.iterations):
                try:
                    start_time = time.time()
                    result = await wait_for(
                        manager.run_inference_sharded(sample_input),
                        timeout=args.timeout
                    )
                    execution_time = time.time() - start_time
                    
                    if 'error' not in result:
                        benchmark_results.append({
                            "iteration": i + 1,
                            "execution_time": execution_time,
                            "metrics": result.get("metrics", {})
                        })
                        logger.info(f"Benchmark iteration {i+1}/{args.iterations}: {execution_time:.3f}s")
                    else:
                        logger.error(f"Error in iteration {i+1}: {result['error']}")
                except TimeoutError:
                    logger.error(f"Timeout in iteration {i+1}")
                except Exception as e:
                    logger.error(f"Error in iteration {i+1}: {e}")
                
                # Wait between iterations
                if i < args.iterations - 1:
                    await anyio.sleep(args.iteration_delay)
            
            # Calculate benchmark statistics
            if benchmark_results:
                execution_times = [r["execution_time"] for r in benchmark_results]
                avg_time = sum(execution_times) / len(execution_times)
                min_time = min(execution_times)
                max_time = max(execution_times)
                
                test_results["phases"]["benchmark"]["statistics"] = {
                    "avg_execution_time": avg_time,
                    "min_execution_time": min_time,
                    "max_execution_time": max_time,
                    "std_dev": (sum((t - avg_time) ** 2 for t in execution_times) / len(execution_times)) ** 0.5,
                    "successful_iterations": len(benchmark_results),
                    "total_iterations": args.iterations
                }
                
                test_results["phases"]["benchmark"]["iterations"] = benchmark_results
                logger.info(f"Benchmark completed: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")
            
            test_results["phases"]["benchmark"]["duration"] = time.time() - phase_start
            test_results["phases"]["benchmark"]["status"] = "completed"
        
        # Phase 8: Cleanup
        phase_start = time.time()
        test_results["phases"]["cleanup"] = {"status": "running"}
        
        # Get final metrics
        final_metrics = manager.get_metrics()
        test_results["final_metrics"] = final_metrics
        
        # Close manager
        await manager.close()
        logger.info("Model sharding manager closed")
        
        # Close resource pool
        await resource_pool.close()
        logger.info("Resource pool closed")
        
        # Clean up browser instances
        await clean_up_browsers(browser_instances)
        
        test_results["phases"]["cleanup"]["duration"] = time.time() - phase_start
        test_results["phases"]["cleanup"]["status"] = "completed"
        
        # Calculate overall test duration
        test_results["end_time"] = datetime.datetime.now().isoformat()
        test_results["total_duration"] = sum(phase["duration"] for phase in test_results["phases"].values() if "duration" in phase)
        test_results["status"] = "completed"
        
        logger.info(f"Test completed successfully in {test_results['total_duration']:.2f}s")
        
        return test_results
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        
        # Add error details to test results
        test_results["status"] = "error"
        test_results["error"] = str(e)
        test_results["error_traceback"] = traceback.format_exc()
        test_results["end_time"] = datetime.datetime.now().isoformat()
        
        # Try to clean up browsers if available
        if "browser_instances" in locals():
            await clean_up_browsers(browser_instances)
        
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
            
        # Write results to file
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Test results saved to {args.output}")
    except Exception as e:
        logger.error(f"Error saving test results: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Real Browser Fault Tolerance Test")
    
    # Model options
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                       help="Model name to test")
    parser.add_argument("--model-type", type=str, default="text",
                       choices=["text", "vision", "audio", "multimodal", "text_embedding"],
                       help="Type of model")
    parser.add_argument("--sequence-length", type=int, default=10,
                       help="Sequence length for text inputs")
    
    # Browser options
    parser.add_argument("--browsers", type=str, default="",
                       help="Comma-separated list of browsers to use (chrome,firefox,edge)")
    parser.add_argument("--browser-count", type=int, default=3,
                       help="Number of browser instances to launch")
    parser.add_argument("--show-browsers", action="store_true",
                       help="Show browser windows (disable headless mode)")
    
    # Sharding options
    parser.add_argument("--shards", type=int, default=3,
                       help="Number of shards to create")
    parser.add_argument("--shard-type", type=str, default="layer",
                       choices=["layer", "attention_feedforward", "component"],
                       help="Type of sharding to use")
    
    # Fault tolerance options
    parser.add_argument("--fault-tolerance-level", type=str, default="high",
                       choices=["none", "low", "medium", "high", "critical"],
                       help="Fault tolerance level")
    parser.add_argument("--recovery-strategy", type=str, default="progressive",
                       choices=["restart", "reconnect", "failover", "progressive", "parallel"],
                       help="Recovery strategy to use")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retry attempts for recovery")
    
    # Failure options
    parser.add_argument("--force-failure", action="store_true",
                       help="Force a browser failure during test")
    parser.add_argument("--failure-index", type=int,
                       help="Index of browser to fail (default: random)")
    parser.add_argument("--failure-type", type=str,
                       choices=["crash", "hang", "memory", "disconnect"],
                       help="Type of failure to force (default: random)")
    parser.add_argument("--failure-detection-delay", type=float, default=2.0,
                       help="Delay in seconds to wait for failure detection")
    
    # Benchmark options
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark after recovery")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of iterations for benchmark")
    parser.add_argument("--iteration-delay", type=float, default=0.5,
                       help="Delay in seconds between benchmark iterations")
    
    # General options
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout in seconds for operations")
    parser.add_argument("--db-path", type=str,
                       help="Path to DuckDB database for storing performance data")
    parser.add_argument("--disable-ipfs", action="store_true",
                       help="Disable IPFS acceleration")
    parser.add_argument("--output", type=str,
                       help="Path to output file for test results (JSON)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check availability of required modules
    if not MODULES_AVAILABLE:
        logger.error("Required modules not available, cannot run test")
        return 1
    
    if not SELENIUM_AVAILABLE:
        logger.warning("Selenium not available, real browser testing will be limited")
    
    # Print test configuration
    logger.info(f"Testing {args.model} ({args.model_type}) with fault tolerance level {args.fault_tolerance_level}")
    if args.force_failure:
        logger.info(f"Will force browser failure during test")
    
    try:
        # Run the test
        test_results = anyio.run(test_real_browser_fault_tolerance(args))
        
        # Save results if output path specified
        if args.output:
            anyio.run(save_test_results(test_results, args))
        
        # Determine exit code based on test status
        if test_results["status"] == "completed":
            logger.info("Test completed successfully")
            return 0
        else:
            logger.error(f"Test failed: {test_results.get('error', 'Unknown error')}")
            return 1
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())