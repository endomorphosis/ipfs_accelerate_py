#!/usr/bin/env python3
"""
Example of using the Load Balancer Resource Pool Bridge

This script demonstrates how to use the Load Balancer Resource Pool Bridge
to integrate the Distributed Testing Framework with WebGPU/WebNN Resource Pool.

Usage:
    python resource_pool_load_balancer_example.py
"""

import anyio
import logging
import os
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

try:
    from test.tests.distributed.distributed_testing.examples.load_balancer_resource_pool_bridge import (
        LoadBalancerResourcePoolBridge, 
        create_bridge
    )
    from data.duckdb.distributed_testing.load_balancer.models import TestRequirements
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please make sure the required modules are installed")
    sys.exit(1)


async def run_example():
    """Run the example."""
    logger.info("Starting Load Balancer Resource Pool Bridge example")
    
    # Create bridge
    bridge_config = {
        "db_path": "test_resource_pool.db",
        "max_browsers_per_worker": 3,
        "enable_fault_tolerance": True,
        "recovery_strategy": "progressive",
        "browser_preferences": {
            'audio': 'firefox',
            'vision': 'chrome',
            'text_embedding': 'edge'
        }
    }
    
    bridge = create_bridge(bridge_config)
    
    try:
        # Start bridge
        await bridge.start()
        
        # Register workers
        worker_ids = ["worker-1", "worker-2", "worker-3"]
        for worker_id in worker_ids:
            await bridge.register_worker(worker_id)
            logger.info(f"Registered worker: {worker_id}")
        
        # Define test callback
        def test_callback(test_id, result):
            logger.info(f"Test {test_id} completed with status: {result.get('status')}")
            logger.info(f"Execution time: {result.get('execution_time', 0):.2f}s")
        
        # Register callback
        bridge.register_test_execution_callback(test_callback)
        
        # Submit tests
        test_configs = [
            {
                "test_id": "test-vision-1",
                "model_id": "vit-base-patch16-224",
                "model_type": "vision",
                "priority": 2,
                "browser": "chrome"
            },
            {
                "test_id": "test-audio-1",
                "model_id": "whisper-tiny",
                "model_type": "audio",
                "priority": 3,
                "browser": "firefox"
            },
            {
                "test_id": "test-text-1",
                "model_id": "bert-base-uncased",
                "model_type": "text_embedding",
                "priority": 1,
                "browser": "edge"
            },
            {
                "test_id": "test-sharded-1",
                "model_id": "llama-13b",
                "model_type": "large_language_model",
                "priority": 3,
                "browser": "chrome",
                "requires_sharding": True,
                "num_shards": 3
            }
        ]
        
        # Submit tests
        for config in test_configs:
            # Create test requirements
            browser_requirements = {"preferred": config.get("browser")}
            
            # Add sharding requirements if needed
            sharding_requirements = None
            if config.get("requires_sharding"):
                sharding_requirements = {
                    "strategy": "layer_balanced",
                    "num_shards": config.get("num_shards", 2),
                    "fault_tolerance_level": "medium"
                }
            
            test_requirements = TestRequirements(
                test_id=config["test_id"],
                model_id=config["model_id"],
                model_type=config["model_type"],
                minimum_memory=2.0,
                priority=config["priority"],
                browser_requirements=browser_requirements,
                requires_sharding=config.get("requires_sharding", False),
                sharding_requirements=sharding_requirements
            )
            
            # Submit test
            test_id = await bridge.submit_test(test_requirements)
            logger.info(f"Submitted test: {test_id}")
        
        # Wait for tests to complete
        logger.info("Waiting for tests to complete...")
        await anyio.sleep(10)
        
        # Analyze system performance
        logger.info("Analyzing system performance...")
        analysis = await bridge.analyze_system_performance()
        
        # Log performance analysis
        logger.info("System performance analysis:")
        worker_analysis = analysis.get("worker_analysis", {})
        for worker_id, worker_data in worker_analysis.items():
            recommendations = worker_data.get("recommendations", {})
            browser_prefs = recommendations.get("browser_preferences", {})
            if browser_prefs:
                logger.info(f"Worker {worker_id} browser preference recommendations:")
                for model_type, browser in browser_prefs.items():
                    logger.info(f"  - {model_type}: {browser}")
            else:
                logger.info(f"Worker {worker_id}: No browser preference recommendations")
        
        # Apply recommendations
        logger.info("Applying optimization recommendations...")
        results = await bridge.apply_optimization_recommendations(analysis)
        logger.info(f"Applied optimizations: {len(results['applied'])} applied, {len(results['failed'])} failed")
        
        # Wait a bit more for any remaining operations
        await anyio.sleep(5)
        
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
    finally:
        # Stop bridge
        await bridge.stop()
        logger.info("Load Balancer Resource Pool Bridge example completed")


def main():
    """Main function to run the example."""
    try:
        anyio.run(run_example())
    except KeyboardInterrupt:
        logger.info("Example interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")


if __name__ == "__main__":
    main()