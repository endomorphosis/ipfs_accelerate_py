#!/usr/bin/env python3
"""
Dynamic Resource Management Integration Test

This script tests the integration of the Dynamic Resource Management (DRM) system
with the Coordinator server and Cloud Provider Manager. It verifies that the
coordinator can properly scale worker nodes up and down based on workload patterns.

Usage:
    python test_drm_integration.py
"""

import os
import sys
import json
import time
import uuid
import argparse
import logging
import threading
import anyio
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("drm_integration_test")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import required modules
from data.duckdb.distributed_testing.dynamic_resource_manager import DynamicResourceManager, ScalingDecision
from data.duckdb.distributed_testing.resource_performance_predictor import ResourcePerformancePredictor
from data.duckdb.distributed_testing.cloud_provider_manager import CloudProviderManager


class MockCloudProvider:
    """Mock cloud provider for testing DRM integration."""
    
    def __init__(self, name):
        """Initialize mock cloud provider."""
        self.name = name
        self.workers = {}
        self.worker_counter = 0
    
    def create_worker(self, config):
        """Create a mock worker node."""
        worker_id = f"{self.name}-worker-{self.worker_counter}"
        self.worker_counter += 1
        
        logger.info(f"Creating worker {worker_id} with config: {json.dumps(config, indent=2)}")
        
        # Create mock worker data
        self.workers[worker_id] = {
            "id": worker_id,
            "status": "running",
            "config": config,
            "creation_time": time.time()
        }
        
        return {
            "worker_id": worker_id,
            "status": "running",
            "provider": self.name
        }
    
    def terminate_worker(self, worker_id):
        """Terminate a mock worker node."""
        if worker_id in self.workers:
            self.workers[worker_id]["status"] = "terminated"
            logger.info(f"Terminated worker {worker_id}")
            return True
        
        logger.warning(f"Worker {worker_id} not found")
        return False
    
    def get_worker_status(self, worker_id):
        """Get mock worker status."""
        if worker_id in self.workers:
            return {
                "worker_id": worker_id,
                "status": self.workers[worker_id]["status"],
                "provider": self.name
            }
        
        return None
    
    def get_available_resources(self):
        """Get available resources."""
        return {
            "provider": self.name,
            "max_workers": 10,
            "active_workers": len([w for w in self.workers.values() if w["status"] == "running"])
        }


def test_drm_integration():
    """Test DRM integration with CloudProviderManager."""
    logger.info("Starting DRM integration test")
    
    # Initialize components
    drm = DynamicResourceManager(
        target_utilization=0.7,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
        evaluation_window=5,  # 5 seconds for testing
        scale_up_cooldown=5,  # 5 seconds for testing
        scale_down_cooldown=5  # 5 seconds for testing
    )
    
    cloud_provider_manager = CloudProviderManager()
    
    # Add mock cloud providers
    mock_aws = MockCloudProvider("aws")
    mock_gcp = MockCloudProvider("gcp")
    mock_docker = MockCloudProvider("docker_local")
    
    cloud_provider_manager.add_provider("aws", mock_aws)
    cloud_provider_manager.add_provider("gcp", mock_gcp)
    cloud_provider_manager.add_provider("docker_local", mock_docker)
    
    # Test CloudProviderManager.get_preferred_provider
    provider_name = cloud_provider_manager.get_preferred_provider({"gpu": True})
    logger.info(f"Preferred provider for GPU workload: {provider_name}")
    
    provider_name = cloud_provider_manager.get_preferred_provider({"min_memory_gb": 64})
    logger.info(f"Preferred provider for high memory workload: {provider_name}")
    
    provider_name = cloud_provider_manager.get_preferred_provider({"min_cpu_cores": 32})
    logger.info(f"Preferred provider for high CPU workload: {provider_name}")
    
    # Register some workers
    drm.register_worker(
        "worker-1", 
        {
            "cpu": {"cores": 8, "available_cores": 2},
            "memory": {"total_mb": 16384, "available_mb": 4096},
            "gpu": {"devices": 1, "memory_mb": 8192, "available_memory_mb": 2048}
        }
    )
    
    drm.register_worker(
        "worker-2", 
        {
            "cpu": {"cores": 4, "available_cores": 1},
            "memory": {"total_mb": 8192, "available_mb": 2048}
        }
    )
    
    # Get utilization
    utilization = drm.get_worker_utilization()
    logger.info(f"Initial utilization: {utilization['overall']['overall']:.1%}")
    
    # Test scaling decision - should trigger scale up
    scaling_decision = drm.evaluate_scaling()
    logger.info(f"Scaling decision: {scaling_decision.action} - {scaling_decision.reason}")
    
    # Process scaling decision with cloud provider manager
    if scaling_decision.action == "scale_up":
        for i in range(scaling_decision.count):
            worker_result = cloud_provider_manager.create_worker(
                provider=scaling_decision.provider or "aws",
                resources=scaling_decision.resource_requirements,
                worker_type=scaling_decision.worker_type,
                coordinator_url="ws://localhost:8080",
                api_key="test_api_key"
            )
            
            if worker_result:
                # Register worker with DRM
                worker_id = worker_result["worker_id"]
                
                # Create simulated resources based on worker type
                if scaling_decision.worker_type == "gpu":
                    resources = {
                        "cpu": {"cores": 16, "available_cores": 16},
                        "memory": {"total_mb": 32768, "available_mb": 32768},
                        "gpu": {"devices": 2, "memory_mb": 16384, "available_memory_mb": 16384}
                    }
                elif scaling_decision.worker_type == "memory":
                    resources = {
                        "cpu": {"cores": 8, "available_cores": 8},
                        "memory": {"total_mb": 65536, "available_mb": 65536}
                    }
                else:  # default or cpu
                    resources = {
                        "cpu": {"cores": 8, "available_cores": 8},
                        "memory": {"total_mb": 16384, "available_mb": 16384}
                    }
                
                drm.register_worker(worker_id, resources)
                logger.info(f"Registered new worker {worker_id}")
    
    # Simulate busy workers by setting low available resources
    for worker_id in ["worker-1", "worker-2"]:
        if worker_id in drm.worker_resources:
            drm.worker_resources[worker_id]["resources"]["cpu"]["available_cores"] = 0
            drm.worker_resources[worker_id]["resources"]["memory"]["available_mb"] = 0
            if "gpu" in drm.worker_resources[worker_id]["resources"]:
                drm.worker_resources[worker_id]["resources"]["gpu"]["available_memory_mb"] = 0
    
    # Wait a bit for cooldown
    logger.info("Waiting for cooldown...")
    time.sleep(6)
    
    # Get updated utilization
    utilization = drm.get_worker_utilization()
    logger.info(f"Updated utilization: {utilization['overall']['overall']:.1%}")
    
    # Test scaling decision again
    scaling_decision = drm.evaluate_scaling()
    logger.info(f"Second scaling decision: {scaling_decision.action} - {scaling_decision.reason}")
    
    # Process scaling decision
    if scaling_decision.action == "scale_up":
        for i in range(scaling_decision.count):
            worker_result = cloud_provider_manager.create_worker(
                provider=scaling_decision.provider or "aws",
                resources=scaling_decision.resource_requirements,
                worker_type=scaling_decision.worker_type
            )
            
            if worker_result:
                # Register worker with DRM
                worker_id = worker_result["worker_id"]
                resources = {
                    "cpu": {"cores": 8, "available_cores": 8},
                    "memory": {"total_mb": 16384, "available_mb": 16384}
                }
                drm.register_worker(worker_id, resources)
                logger.info(f"Registered new worker {worker_id}")
    
    # Now simulate low utilization by setting high available resources for all workers
    for worker_id in drm.worker_resources:
        resources = drm.worker_resources[worker_id]["resources"]
        resources["cpu"]["available_cores"] = resources["cpu"]["cores"]
        resources["memory"]["available_mb"] = resources["memory"]["total_mb"]
        if "gpu" in resources:
            resources["gpu"]["available_memory_mb"] = resources["gpu"]["memory_mb"]
    
    # Wait for cooldown
    logger.info("Waiting for scale-down cooldown...")
    time.sleep(6)
    
    # Test scaling decision - should trigger scale down
    utilization = drm.get_worker_utilization()
    logger.info(f"Low utilization: {utilization['overall']['overall']:.1%}")
    
    scaling_decision = drm.evaluate_scaling()
    logger.info(f"Scale-down decision: {scaling_decision.action} - {scaling_decision.reason}")
    
    # Process scaling decision
    if scaling_decision.action == "scale_down":
        logger.info(f"Scaling down {len(scaling_decision.worker_ids)} workers: {scaling_decision.worker_ids}")
        
        for worker_id in scaling_decision.worker_ids:
            # First find the provider for this worker
            # In a real system, this would be tracked
            for provider_name, provider in [("aws", mock_aws), ("gcp", mock_gcp), ("docker_local", mock_docker)]:
                if worker_id in provider.workers:
                    success = cloud_provider_manager.terminate_worker(provider_name, worker_id)
                    
                    if success:
                        # Deregister worker
                        drm.deregister_worker(worker_id)
                        logger.info(f"Terminated and deregistered worker {worker_id}")
                    break
    
    # Get final worker count
    worker_count = len(drm.worker_resources)
    logger.info(f"Final worker count: {worker_count}")
    
    logger.info("DRM integration test completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Resource Management Integration Test")
    args = parser.parse_args()
    
    # Run test
    test_drm_integration()