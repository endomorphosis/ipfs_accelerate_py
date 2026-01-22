#!/usr/bin/env python3
"""
Test Hardware Integration with Coordinator

This script tests the integration between the hardware capability detector
and the coordinator component of the distributed testing framework.

Usage:
    python run_test_hardware_integration.py --db-path ./test_db.duckdb
"""

import os
import sys
import json
import logging
import asyncio
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_hardware_integration")

# Import necessary modules
from hardware_capability_detector import (
    HardwareCapabilityDetector,
    HardwareType,
    HardwareVendor,
    PrecisionType,
    CapabilityScore,
    HardwareCapability,
    WorkerHardwareCapabilities
)

from coordinator_hardware_integration import CoordinatorHardwareIntegration


class DummyRequest:
    """Dummy request object for testing."""
    
    def __init__(self, data=None, headers=None):
        self.data = data or {}
        self.headers = headers or {}
    
    async def json(self):
        return self.data


class DummyResponse:
    """Dummy response object for testing."""
    
    def __init__(self, data=None, status=200):
        self.data = data or {}
        self.status = status
    
    async def json(self):
        return self.data


class DummyCoordinator:
    """Dummy coordinator for testing."""
    
    def __init__(self, db_path=None):
        self.db_path = db_path
        self.workers = {}
        self.worker_connections = {}
        self.tasks = {}
        self.pending_tasks = set()
        self.running_tasks = {}
    
    async def handle_worker_register(self, request):
        """Dummy worker registration handler."""
        try:
            # Get data from request
            data = await request.json()
            
            # Generate worker ID
            import uuid
            worker_id = data.get("worker_id", f"worker-{uuid.uuid4().hex[:8]}")
            
            # Create worker entry
            self.workers[worker_id] = {
                "worker_id": worker_id,
                "hostname": data.get("hostname", "unknown"),
                "status": "active",
                "registered": datetime.now().isoformat(),
                "last_heartbeat": datetime.now().isoformat()
            }
            
            # Simulate worker connection
            self.worker_connections[worker_id] = True
            
            # Return success response
            return DummyResponse({
                "status": "success",
                "worker_id": worker_id,
                "message": "Worker registered successfully"
            })
        
        except Exception as e:
            logger.error(f"Error in dummy worker registration: {str(e)}")
            return DummyResponse({"error": str(e)}, status=500)
    
    async def _can_worker_handle_task(self, worker_id, task):
        """Dummy handler for checking if worker can handle task."""
        # Basic check: worker exists and is active
        if worker_id not in self.workers:
            return False
        
        return self.workers[worker_id].get("status") == "active"
    
    async def _assign_pending_tasks(self):
        """Dummy handler for assigning pending tasks."""
        tasks_assigned = 0
        
        # Get list of pending tasks
        pending_task_ids = list(self.pending_tasks)
        
        # Get available workers
        available_workers = []
        for worker_id, worker in self.workers.items():
            if (worker.get("status") == "active" and 
                worker_id in self.worker_connections):
                available_workers.append(worker_id)
        
        # If no workers available, return
        if not available_workers:
            return 0
        
        # Assign tasks in a round-robin fashion
        for task_id in pending_task_ids:
            # Skip if task doesn't exist
            if task_id not in self.tasks:
                continue
            
            # Find an available worker
            for worker_id in available_workers:
                # Assign task to worker
                self.running_tasks[task_id] = worker_id
                self.pending_tasks.remove(task_id)
                tasks_assigned += 1
                break
        
        return tasks_assigned


async def test_hardware_integration(db_path=None, enable_browser_detection=False):
    """Test hardware integration with coordinator."""
    logger.info("Testing hardware integration with coordinator")
    
    # Create dummy coordinator
    coordinator = DummyCoordinator(db_path)
    
    # Create hardware integration
    hardware_integration = CoordinatorHardwareIntegration(
        coordinator=coordinator,
        db_path=db_path,
        enable_browser_detection=enable_browser_detection
    )
    
    # Initialize integration
    await hardware_integration.initialize()
    
    # Create hardware capability detector
    detector = HardwareCapabilityDetector(
        db_path=db_path,
        enable_browser_detection=enable_browser_detection
    )
    
    # Detect hardware capabilities
    capabilities = detector.detect_all_capabilities()
    
    # Print capabilities summary
    logger.info(f"Detected {len(capabilities.hardware_capabilities)} hardware capabilities")
    logger.info(f"OS: {capabilities.os_type} {capabilities.os_version}")
    logger.info(f"CPU Count: {capabilities.cpu_count}")
    logger.info(f"Total Memory: {capabilities.total_memory_gb:.2f} GB")
    
    # Print detected hardware
    for hw in capabilities.hardware_capabilities:
        hw_type = hw.hardware_type.name if isinstance(hw.hardware_type, Enum) else hw.hardware_type
        vendor = hw.vendor.name if isinstance(hw.vendor, Enum) else hw.vendor
        logger.info(f"  {hw_type} - {vendor} - {hw.model}")
    
    # Create dummy worker registration request with hardware capabilities
    worker_data = {
        "hostname": capabilities.hostname,
        "worker_id": "test-worker-1",
        "capabilities": {
            "hardware": [
                hw.hardware_type.value if isinstance(hw.hardware_type, Enum) else hw.hardware_type
                for hw in capabilities.hardware_capabilities
            ]
        },
        "hardware_capabilities": {
            "hostname": capabilities.hostname,
            "os_type": capabilities.os_type,
            "os_version": capabilities.os_version,
            "cpu_count": capabilities.cpu_count,
            "total_memory_gb": capabilities.total_memory_gb,
            "hardware_capabilities": [
                {
                    "hardware_type": hw.hardware_type.value if isinstance(hw.hardware_type, Enum) else hw.hardware_type,
                    "vendor": hw.vendor.value if isinstance(hw.vendor, Enum) else hw.vendor,
                    "model": hw.model,
                    "version": hw.version,
                    "driver_version": hw.driver_version,
                    "compute_units": hw.compute_units,
                    "cores": hw.cores,
                    "memory_gb": hw.memory_gb,
                    "supported_precisions": [
                        p.value if isinstance(p, Enum) else p
                        for p in hw.supported_precisions
                    ],
                    "capabilities": hw.capabilities,
                    "scores": {
                        k: v.value if isinstance(v, Enum) else v
                        for k, v in hw.scores.items()
                    }
                }
                for hw in capabilities.hardware_capabilities
            ]
        }
    }
    request = DummyRequest(worker_data, {"X-API-Key": "dummy-api-key"})
    
    # Register worker with hardware capabilities
    logger.info("Registering worker with hardware capabilities")
    response = await hardware_integration._enhanced_worker_register_handler(request)
    
    if response.status == 200:
        logger.info("Worker registered successfully")
    else:
        logger.error(f"Worker registration failed: {await response.json()}")
        return
    
    # Verify worker in coordinator
    worker_id = "test-worker-1"
    if worker_id in coordinator.workers:
        logger.info(f"Worker {worker_id} found in coordinator")
    else:
        logger.error(f"Worker {worker_id} not found in coordinator")
        return
    
    # Test hardware requirement checks
    logger.info("Testing hardware requirement checks")
    
    # Create task with basic hardware requirements
    task_basic = {
        "task_id": "task-basic",
        "type": "test",
        "requirements": {
            "hardware": {
                "required_types": ["cpu"]
            }
        }
    }
    
    # Create task with advanced hardware requirements
    task_advanced = {
        "task_id": "task-advanced",
        "type": "test",
        "requirements": {
            "hardware": {
                "required_types": ["gpu"],
                "min_memory_gb": 8.0,
                "precision_types": ["fp16"]
            }
        }
    }
    
    # Create task with impossible hardware requirements
    task_impossible = {
        "task_id": "task-impossible",
        "type": "test",
        "requirements": {
            "hardware": {
                "required_types": ["tpu"],
                "min_memory_gb": 128.0,
                "precision_types": ["fp4"]
            }
        }
    }
    
    # Add tasks to coordinator
    coordinator.tasks[task_basic["task_id"]] = task_basic
    coordinator.tasks[task_advanced["task_id"]] = task_advanced
    coordinator.tasks[task_impossible["task_id"]] = task_impossible
    
    # Add tasks to pending tasks
    coordinator.pending_tasks.add(task_basic["task_id"])
    coordinator.pending_tasks.add(task_advanced["task_id"])
    coordinator.pending_tasks.add(task_impossible["task_id"])
    
    # Check if worker can handle tasks
    can_handle_basic = await hardware_integration._enhanced_can_worker_handle_task(worker_id, task_basic)
    can_handle_advanced = await hardware_integration._enhanced_can_worker_handle_task(worker_id, task_advanced)
    can_handle_impossible = await hardware_integration._enhanced_can_worker_handle_task(worker_id, task_impossible)
    
    logger.info(f"Can handle basic task: {can_handle_basic}")
    logger.info(f"Can handle advanced task: {can_handle_advanced}")
    logger.info(f"Can handle impossible task: {can_handle_impossible}")
    
    # Test task assignment
    logger.info("Testing task assignment")
    tasks_assigned = await hardware_integration._enhanced_assign_pending_tasks()
    
    logger.info(f"Tasks assigned: {tasks_assigned}")
    logger.info(f"Running tasks: {coordinator.running_tasks}")
    
    # Final summary
    logger.info("\nTest Summary:")
    logger.info(f"Worker registered: {'Yes' if worker_id in coordinator.workers else 'No'}")
    logger.info(f"Basic task handled: {'Yes' if can_handle_basic else 'No'}")
    logger.info(f"Advanced task handled: {'Yes' if can_handle_advanced else 'No'}")
    logger.info(f"Impossible task handled: {'Yes' if can_handle_impossible else 'No'}")
    
    # Get database statistics if available
    if db_path and os.path.exists(db_path):
        try:
            import duckdb
            conn = duckdb.connect(db_path)
            
            # Get worker count
            worker_count = conn.execute("SELECT COUNT(*) FROM worker_hardware").fetchone()[0]
            
            # Get hardware count
            hw_count = conn.execute("SELECT COUNT(*) FROM hardware_capabilities").fetchone()[0]
            
            logger.info(f"Database statistics:")
            logger.info(f"  Workers in database: {worker_count}")
            logger.info(f"  Hardware capabilities in database: {hw_count}")
            
            conn.close()
        except Exception as e:
            logger.error(f"Error getting database statistics: {str(e)}")
    
    logger.info("Hardware integration test completed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test hardware integration with coordinator")
    parser.add_argument("--db-path", help="Path to DuckDB database")
    parser.add_argument("--enable-browser-detection", action="store_true", help="Enable browser detection")
    
    args = parser.parse_args()
    
    asyncio.run(test_hardware_integration(
        db_path=args.db_path,
        enable_browser_detection=args.enable_browser_detection
    ))


if __name__ == "__main__":
    main()