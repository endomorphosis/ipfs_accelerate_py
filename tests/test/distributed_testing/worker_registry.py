#!/usr/bin/env python3
"""
Worker Registry for Distributed Testing Framework

This module provides worker registration, health tracking, and management
for distributed test execution.
"""

import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkerRegistry:
    """
    Registry for tracking and managing distributed workers
    """
    
    def __init__(self, registry_id: str = "default"):
        """
        Initialize worker registry
        
        Args:
            registry_id: Unique identifier for this registry
        """
        self.registry_id = registry_id
        self.workers = {}  # worker_id -> worker_info
        self.worker_health = {}  # worker_id -> health_info
        self.worker_capabilities = {}  # capability -> [worker_ids]
        
        # Worker state change history
        self.state_changes = []
        
        # Create logger
        self.logger = logging.getLogger(f"worker_registry.{registry_id}")
        self.logger.info(f"Worker registry {registry_id} initialized")
    
    async def register(self, worker_id: str, worker_info: Dict[str, Any]) -> bool:
        """
        Register a worker with the registry
        
        Args:
            worker_id: Unique identifier for the worker
            worker_info: Information about the worker
            
        Returns:
            Registration success
        """
        # Record current time
        current_time = time.time()
        
        # Add or update worker
        is_new = worker_id not in self.workers
        
        # Store worker info
        self.workers[worker_id] = {
            **worker_info,
            "registration_time": current_time,
            "last_updated": current_time
        }
        
        # Initialize health tracking
        self.worker_health[worker_id] = {
            "last_heartbeat": current_time,
            "health_status": "healthy",
            "health_checks": []
        }
        
        # Register capabilities
        capabilities = worker_info.get("capabilities", [])
        for capability in capabilities:
            if capability not in self.worker_capabilities:
                self.worker_capabilities[capability] = set()
            self.worker_capabilities[capability].add(worker_id)
        
        # Record state change
        self.state_changes.append({
            "worker_id": worker_id,
            "action": "register" if is_new else "update",
            "timestamp": current_time,
            "status": worker_info.get("status", "unknown")
        })
        
        self.logger.info(f"Worker {worker_id} {'registered' if is_new else 'updated'}")
        return True
    
    async def deregister(self, worker_id: str) -> bool:
        """
        Deregister a worker from the registry
        
        Args:
            worker_id: Worker to deregister
            
        Returns:
            Deregistration success
        """
        if worker_id not in self.workers:
            self.logger.warning(f"Worker {worker_id} not found for deregistration")
            return False
        
        # Get worker info before removal
        worker_info = self.workers[worker_id]
        
        # Remove from workers
        del self.workers[worker_id]
        
        # Remove from health tracking
        if worker_id in self.worker_health:
            del self.worker_health[worker_id]
        
        # Remove from capabilities
        capabilities = worker_info.get("capabilities", [])
        for capability in capabilities:
            if capability in self.worker_capabilities and worker_id in self.worker_capabilities[capability]:
                self.worker_capabilities[capability].remove(worker_id)
                
                # Clean up empty capability sets
                if not self.worker_capabilities[capability]:
                    del self.worker_capabilities[capability]
        
        # Record state change
        self.state_changes.append({
            "worker_id": worker_id,
            "action": "deregister",
            "timestamp": time.time(),
            "status": "deregistered"
        })
        
        self.logger.info(f"Worker {worker_id} deregistered")
        return True
    
    async def update_worker_status(self, worker_id: str, status: str) -> bool:
        """
        Update worker status
        
        Args:
            worker_id: Worker to update
            status: New status
            
        Returns:
            Update success
        """
        if worker_id not in self.workers:
            self.logger.warning(f"Worker {worker_id} not found for status update")
            return False
        
        # Update status
        self.workers[worker_id]["status"] = status
        self.workers[worker_id]["last_updated"] = time.time()
        
        # Record state change
        self.state_changes.append({
            "worker_id": worker_id,
            "action": "status_update",
            "timestamp": time.time(),
            "status": status
        })
        
        self.logger.info(f"Worker {worker_id} status updated to {status}")
        return True
    
    async def record_heartbeat(self, worker_id: str, health_info: Dict[str, Any] = None) -> bool:
        """
        Record worker heartbeat
        
        Args:
            worker_id: Worker to update
            health_info: Optional health information
            
        Returns:
            Heartbeat recording success
        """
        if worker_id not in self.workers:
            self.logger.warning(f"Worker {worker_id} not found for heartbeat")
            return False
        
        current_time = time.time()
        
        # Update health tracking
        if worker_id in self.worker_health:
            self.worker_health[worker_id]["last_heartbeat"] = current_time
            
            if health_info:
                # Record health check
                health_check = {
                    "timestamp": current_time,
                    **health_info
                }
                self.worker_health[worker_id]["health_checks"].append(health_check)
                
                # Keep only last 10 health checks
                if len(self.worker_health[worker_id]["health_checks"]) > 10:
                    self.worker_health[worker_id]["health_checks"] = self.worker_health[worker_id]["health_checks"][-10:]
                
                # Update health status
                if "status" in health_info:
                    self.worker_health[worker_id]["health_status"] = health_info["status"]
        
        # Update worker last updated time
        self.workers[worker_id]["last_updated"] = current_time
        
        return True
    
    async def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get worker information
        
        Args:
            worker_id: Worker to retrieve
            
        Returns:
            Worker information or None if not found
        """
        if worker_id not in self.workers:
            return None
        
        worker_info = dict(self.workers[worker_id])
        
        # Add health information
        if worker_id in self.worker_health:
            worker_info["health"] = dict(self.worker_health[worker_id])
            
            # Calculate time since last heartbeat
            current_time = time.time()
            last_heartbeat = self.worker_health[worker_id]["last_heartbeat"]
            worker_info["health"]["time_since_heartbeat"] = current_time - last_heartbeat
        
        return worker_info
    
    async def get_workers_by_capability(self, capability: str) -> List[str]:
        """
        Get workers with a specific capability
        
        Args:
            capability: Capability to match
            
        Returns:
            List of worker IDs with the capability
        """
        if capability not in self.worker_capabilities:
            return []
        
        return list(self.worker_capabilities[capability])
    
    async def get_all_workers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all workers
        
        Returns:
            Dictionary of worker_id -> worker_info
        """
        worker_dict = {}
        
        for worker_id in self.workers:
            worker_info = await self.get_worker(worker_id)
            if worker_info:
                worker_dict[worker_id] = worker_info
        
        return worker_dict
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the worker registry
        
        Returns:
            Dictionary with registry statistics
        """
        current_time = time.time()
        
        # Count workers by status
        status_counts = {}
        for worker_id, worker_info in self.workers.items():
            status = worker_info.get("status", "unknown")
            if status not in status_counts:
                status_counts[status] = 0
            status_counts[status] += 1
        
        # Count capabilities
        capability_counts = {
            capability: len(worker_ids)
            for capability, worker_ids in self.worker_capabilities.items()
        }
        
        # Generate stats
        stats = {
            "registry_id": self.registry_id,
            "total_workers": len(self.workers),
            "status_distribution": status_counts,
            "capability_distribution": capability_counts,
            "state_changes_count": len(self.state_changes),
            "last_state_change": self.state_changes[-1] if self.state_changes else None
        }
        
        return stats