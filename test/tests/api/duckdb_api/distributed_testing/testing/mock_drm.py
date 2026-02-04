#!/usr/bin/env python3
"""
Mock Dynamic Resource Manager for Testing

This module provides a mock implementation of the DynamicResourceManager
for testing the real-time dashboard and visualization components without
needing a real DRM instance.
"""

import random
import time
import datetime
import logging
import math
from typing import Dict, List, Any, Optional, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class ScalingDecision:
    """Mock scaling decision class."""
    
    def __init__(self, action, count, reason):
        """Initialize scaling decision."""
        self.action = action
        self.count = count
        self.reason = reason
        self.worker_ids = [f"worker-{i}" for i in range(count)]
        self.timestamp = datetime.datetime.now()

class MockDynamicResourceManager:
    """
    Mock Dynamic Resource Manager for testing visualizations.
    
    Generates simulated resource utilization data, scaling decisions,
    and other metrics needed for testing visualization components.
    """
    
    def __init__(self):
        """Initialize the mock DRM."""
        # Configuration
        self.scale_up_threshold = 0.80
        self.scale_down_threshold = 0.30
        self.target_utilization = 0.60
        
        # Resource state
        self.total_workers = 5
        self.active_tasks = 3
        self.pending_tasks = 1
        
        # Latest scaling decision
        self.last_scaling_decision = None
        
        # Worker data
        self.workers = {
            f"worker-{i}": self._create_mock_worker(i)
            for i in range(self.total_workers)
        }
        
        # Performance metrics
        self.performance_metrics = {
            "task_throughput": random.uniform(5.0, 15.0),
            "allocation_time": random.uniform(50.0, 150.0),
            "resource_efficiency": random.uniform(0.6, 0.9) * 100
        }
        
        # Simulation parameters
        self.last_update = time.time()
        self.update_count = 0
        self.cycle_period = 300  # Seconds for one complete cycle
        
        logger.info("Mock DynamicResourceManager initialized")
    
    def _create_mock_worker(self, worker_index: int) -> Dict[str, Any]:
        """Create a mock worker with random initial state."""
        # Generate utilization based on worker index for variety
        base_cpu = 0.3 + (worker_index * 0.1) % 0.6
        base_memory = 0.2 + (worker_index * 0.15) % 0.6
        base_gpu = 0.1 + (worker_index * 0.2) % 0.8 if worker_index % 2 == 0 else 0.0
        
        return {
            "utilization": {
                "cpu": base_cpu,
                "memory": base_memory,
                "gpu": base_gpu,
                "overall": (base_cpu + base_memory + (base_gpu if base_gpu > 0 else 0)) / 3
            },
            "tasks": max(0, min(3, worker_index % 4)),
            "resources": {
                "cpu": {
                    "cores": 8,
                    "available_cores": math.ceil(8 * (1 - base_cpu)),
                },
                "memory": {
                    "total_mb": 16384,
                    "available_mb": math.ceil(16384 * (1 - base_memory)),
                },
                "gpu": {
                    "memory_mb": 8192 if worker_index % 2 == 0 else 0,
                    "available_memory_mb": math.ceil(8192 * (1 - base_gpu)) if worker_index % 2 == 0 else 0,
                }
            },
            "health": "healthy",
            "status": "active",
            "last_heartbeat": datetime.datetime.now().isoformat()
        }
    
    def _update_simulation(self):
        """Update simulation state."""
        current_time = time.time()
        elapsed = current_time - self.last_update
        self.last_update = current_time
        self.update_count += 1
        
        # Use a sinusoidal pattern to simulate cyclical load
        # Calculate phase based on time
        phase = (current_time % self.cycle_period) / self.cycle_period * 2 * math.pi
        
        # Base load factor varies between 0.2 and 0.8
        base_load = 0.5 + 0.3 * math.sin(phase)
        
        # Add some random variation
        noise = random.uniform(-0.1, 0.1)
        load_factor = max(0.1, min(0.9, base_load + noise))
        
        # Update active and pending tasks
        self.active_tasks = max(0, min(self.total_workers * 3, 
                                      int(self.total_workers * load_factor * 2 + random.randint(-1, 1))))
        self.pending_tasks = max(0, min(10, 
                                      int(load_factor * 5 - self.total_workers + random.randint(-1, 1))))
        
        # Update worker utilization
        for worker_id, worker in self.workers.items():
            # Worker-specific load factor (some workers busier than others)
            worker_load = max(0.1, min(0.95, load_factor * random.uniform(0.8, 1.2)))
            
            # Update utilization
            cpu_util = max(0.05, min(0.95, worker_load * random.uniform(0.9, 1.1)))
            memory_util = max(0.05, min(0.95, worker_load * random.uniform(0.8, 1.2)))
            gpu_util = 0.0
            
            # Only update GPU for workers with GPU
            if worker["resources"]["gpu"]["memory_mb"] > 0:
                gpu_util = max(0.0, min(0.95, worker_load * random.uniform(0.7, 1.3)))
            
            overall_util = (cpu_util + memory_util + (gpu_util if gpu_util > 0 else 0)) / 3 if gpu_util > 0 else (cpu_util + memory_util) / 2
            
            worker["utilization"] = {
                "cpu": cpu_util,
                "memory": memory_util,
                "gpu": gpu_util,
                "overall": overall_util
            }
            
            # Update available resources
            worker["resources"]["cpu"]["available_cores"] = math.ceil(
                worker["resources"]["cpu"]["cores"] * (1 - cpu_util)
            )
            worker["resources"]["memory"]["available_mb"] = math.ceil(
                worker["resources"]["memory"]["total_mb"] * (1 - memory_util)
            )
            if worker["resources"]["gpu"]["memory_mb"] > 0:
                worker["resources"]["gpu"]["available_memory_mb"] = math.ceil(
                    worker["resources"]["gpu"]["memory_mb"] * (1 - gpu_util)
                )
            
            # Update tasks
            worker["tasks"] = max(0, min(4, int(worker_load * 3 + random.randint(-1, 1))))
            
            # Update heartbeat
            worker["last_heartbeat"] = datetime.datetime.now().isoformat()
        
        # Generate scaling decisions occasionally
        if self.update_count % 10 == 0:
            self._generate_scaling_decision(load_factor)
        
        # Update performance metrics
        self.performance_metrics["task_throughput"] = max(1.0, min(30.0, 
                                                             15.0 * load_factor + random.uniform(-2.0, 2.0)))
        self.performance_metrics["allocation_time"] = max(10.0, min(300.0, 
                                                            100.0 + 100.0 * load_factor + random.uniform(-20.0, 20.0)))
        self.performance_metrics["resource_efficiency"] = max(30.0, min(95.0, 
                                                            70.0 + 20.0 * math.sin(phase + math.pi/4) + random.uniform(-5.0, 5.0)))
        
    def _generate_scaling_decision(self, load_factor):
        """Generate a mock scaling decision based on the current load."""
        # Calculate overall utilization
        utilizations = [worker["utilization"]["overall"] for worker in self.workers.values()]
        overall_util = sum(utilizations) / len(utilizations) if utilizations else 0.5
        
        # Decide on scaling action
        if overall_util > self.scale_up_threshold and random.random() < 0.7:
            # Scale up
            scale_count = random.randint(1, 3)
            reason = "High overall utilization"
            action = "scale_up"
            
            # Actually increase worker count
            for i in range(self.total_workers, self.total_workers + scale_count):
                self.workers[f"worker-{i}"] = self._create_mock_worker(i)
            
            self.total_workers += scale_count
            
        elif overall_util < self.scale_down_threshold and self.total_workers > 3 and random.random() < 0.7:
            # Scale down
            scale_count = random.randint(1, min(2, self.total_workers - 3))
            reason = "Low overall utilization"
            action = "scale_down"
            
            # Actually decrease worker count
            for i in range(self.total_workers - scale_count, self.total_workers):
                if f"worker-{i}" in self.workers:
                    del self.workers[f"worker-{i}"]
            
            self.total_workers -= scale_count
            
        else:
            # Maintain
            scale_count = 0
            reason = "Utilization within target range"
            action = "maintain"
        
        # Create and store scaling decision
        self.last_scaling_decision = ScalingDecision(action, scale_count, reason)
    
    def get_worker_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about workers and their resource utilization.
        
        Returns:
            Dictionary of worker statistics
        """
        # Update simulation state
        self._update_simulation()
        
        # Calculate overall utilization
        cpu_values = [w["utilization"]["cpu"] for w in self.workers.values()]
        memory_values = [w["utilization"]["memory"] for w in self.workers.values()]
        gpu_values = [w["utilization"]["gpu"] for w in self.workers.values() if w["utilization"]["gpu"] > 0]
        
        overall_utilization = {
            "cpu": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            "memory": sum(memory_values) / len(memory_values) if memory_values else 0,
            "gpu": sum(gpu_values) / len(gpu_values) if gpu_values else 0,
            "overall": sum([w["utilization"]["overall"] for w in self.workers.values()]) / len(self.workers) if self.workers else 0
        }
        
        return {
            "total_workers": self.total_workers,
            "active_workers": self.total_workers,
            "active_tasks": self.active_tasks,
            "pending_tasks": self.pending_tasks,
            "resource_reservations": self.active_tasks,
            "overall_utilization": overall_utilization,
            "workers": self.workers
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the DRM system.
        
        Returns:
            Dictionary of performance metrics
        """
        return self.performance_metrics