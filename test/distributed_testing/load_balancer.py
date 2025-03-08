#!/usr/bin/env python3
"""
Distributed Testing Framework - Load Balancer

This module implements adaptive load balancing for the distributed testing framework.
It monitors worker performance in real-time and redistributes tasks for optimal utilization.

Usage:
    Import this module in coordinator.py to enable adaptive load balancing.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdaptiveLoadBalancer:
    """Adaptive load balancer for distributed testing framework."""
    
    def __init__(
        self, 
        coordinator,
        check_interval: int = 30,
        utilization_threshold_high: float = 0.85,
        utilization_threshold_low: float = 0.2,
        performance_window: int = 5,
        enable_task_migration: bool = True,
        max_simultaneous_migrations: int = 2
    ):
        """
        Initialize the adaptive load balancer.
        
        Args:
            coordinator: Reference to the coordinator instance
            check_interval: Interval for load balance checks in seconds
            utilization_threshold_high: Threshold for high utilization (0.0-1.0)
            utilization_threshold_low: Threshold for low utilization (0.0-1.0)
            performance_window: Window size for performance measurements in minutes
            enable_task_migration: Whether to enable task migration
            max_simultaneous_migrations: Maximum number of simultaneous task migrations
        """
        self.coordinator = coordinator
        self.check_interval = check_interval
        self.utilization_threshold_high = utilization_threshold_high
        self.utilization_threshold_low = utilization_threshold_low
        self.performance_window = performance_window
        self.enable_task_migration = enable_task_migration
        self.max_simultaneous_migrations = max_simultaneous_migrations
        
        # Performance measurements
        self.worker_performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Current migrations
        self.active_migrations: Dict[str, Dict[str, Any]] = {}  # task_id -> migration info
        
        # Migration history
        self.migration_history: List[Dict[str, Any]] = []
        
        logger.info("Adaptive load balancer initialized")
    
    async def start_balancing(self):
        """Start the load balancing loop."""
        logger.info("Starting adaptive load balancing")
        
        while True:
            try:
                # Update performance metrics
                await self.update_performance_metrics()
                
                # Check for load imbalance
                if await self.detect_load_imbalance():
                    # Balance load if imbalance detected
                    await self.balance_load()
                
                # Clean up completed migrations
                await self.cleanup_migrations()
                
            except Exception as e:
                logger.error(f"Error in load balancing loop: {str(e)}")
            
            # Sleep until next check
            await asyncio.sleep(self.check_interval)
    
    async def update_performance_metrics(self):
        """Update performance metrics for all workers."""
        try:
            now = datetime.now()
            
            # Collect current metrics for all active workers
            for worker_id, worker in self.coordinator.workers.items():
                # Skip offline workers
                if worker.get("status") == "offline":
                    continue
                
                # Get worker hardware metrics
                hardware_metrics = worker.get("hardware_metrics", {})
                
                # Calculate overall utilization
                cpu_percent = hardware_metrics.get("cpu_percent", 0)
                memory_percent = hardware_metrics.get("memory_percent", 0)
                
                # If GPU metrics are available, include them
                gpu_utilization = 0
                if "gpu" in hardware_metrics:
                    gpu_metrics = hardware_metrics["gpu"]
                    if isinstance(gpu_metrics, list) and len(gpu_metrics) > 0:
                        # Average utilization across GPUs
                        gpu_utils = [gpu.get("memory_utilization_percent", 0) for gpu in gpu_metrics]
                        gpu_utilization = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
                    elif isinstance(gpu_metrics, dict):
                        gpu_utilization = gpu_metrics.get("memory_utilization_percent", 0)
                
                # Calculate combined utilization (weighted average)
                # Weight CPU and memory equally, and GPU if it's used
                has_gpu = gpu_utilization > 0
                if has_gpu:
                    utilization = (cpu_percent + memory_percent + gpu_utilization) / 3
                else:
                    utilization = (cpu_percent + memory_percent) / 2
                
                # Normalize to 0.0-1.0 range
                utilization = utilization / 100
                
                # Count running tasks for this worker
                running_tasks = sum(1 for task_id, w_id in self.coordinator.running_tasks.items() if w_id == worker_id)
                
                # Create performance record
                performance = {
                    "timestamp": now.isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "gpu_utilization": gpu_utilization if has_gpu else None,
                    "utilization": utilization,
                    "running_tasks": running_tasks,
                    "has_gpu": has_gpu
                }
                
                # Add to history
                if worker_id not in self.worker_performance_history:
                    self.worker_performance_history[worker_id] = []
                
                self.worker_performance_history[worker_id].append(performance)
                
                # Limit history to performance window (e.g., last 5 minutes)
                cutoff_time = now - timedelta(minutes=self.performance_window)
                self.worker_performance_history[worker_id] = [
                    p for p in self.worker_performance_history[worker_id]
                    if datetime.fromisoformat(p["timestamp"]) >= cutoff_time
                ]
            
            # Log overall system utilization
            await self._log_system_utilization()
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    async def _log_system_utilization(self):
        """Log overall system utilization."""
        if not self.worker_performance_history:
            return
        
        # Calculate average utilization across all workers
        total_utilization = 0.0
        total_workers = 0
        
        for worker_id, history in self.worker_performance_history.items():
            if history:
                # Get latest performance record
                latest = history[-1]
                total_utilization += latest["utilization"]
                total_workers += 1
        
        if total_workers > 0:
            avg_utilization = total_utilization / total_workers
            logger.debug(f"System utilization: {avg_utilization:.2%} across {total_workers} workers")
    
    async def detect_load_imbalance(self) -> bool:
        """
        Detect if there is a load imbalance in the system.
        
        Returns:
            True if imbalance detected, False otherwise
        """
        if not self.worker_performance_history:
            return False
        
        # Get current utilization for all active workers
        worker_utilization = {}
        
        for worker_id, history in self.worker_performance_history.items():
            # Skip workers with no history
            if not history:
                continue
            
            # Skip offline workers
            worker = self.coordinator.workers.get(worker_id)
            if not worker or worker.get("status") == "offline":
                continue
            
            # Get average utilization over the last few records
            recent_history = history[-min(5, len(history)):]
            avg_utilization = sum(p["utilization"] for p in recent_history) / len(recent_history)
            
            worker_utilization[worker_id] = avg_utilization
        
        # Need at least 2 workers to detect imbalance
        if len(worker_utilization) < 2:
            return False
        
        # Find highest and lowest utilization
        max_util_worker = max(worker_utilization.items(), key=lambda x: x[1])
        min_util_worker = min(worker_utilization.items(), key=lambda x: x[1])
        
        max_worker_id, max_util = max_util_worker
        min_worker_id, min_util = min_util_worker
        
        # Check if there's a significant imbalance
        imbalance_detected = (max_util > self.utilization_threshold_high and 
                             min_util < self.utilization_threshold_low and
                             max_util - min_util > 0.4)  # At least 40% difference
        
        if imbalance_detected:
            logger.info(f"Load imbalance detected: Worker {max_worker_id} at {max_util:.2%}, "
                     f"Worker {min_worker_id} at {min_util:.2%}")
        
        return imbalance_detected
    
    async def balance_load(self):
        """Balance load by redistributing tasks."""
        # Skip if task migration is disabled
        if not self.enable_task_migration:
            logger.info("Task migration is disabled, skipping load balancing")
            return
        
        # Skip if too many active migrations
        if len(self.active_migrations) >= self.max_simultaneous_migrations:
            logger.info(f"Too many active migrations ({len(self.active_migrations)}), skipping load balancing")
            return
        
        try:
            # Get worker utilization
            worker_utilization = {}
            
            for worker_id, history in self.worker_performance_history.items():
                # Skip workers with no history
                if not history:
                    continue
                
                # Skip offline workers
                worker = self.coordinator.workers.get(worker_id)
                if not worker or worker.get("status") == "offline":
                    continue
                
                # Get latest utilization
                latest = history[-1]
                worker_utilization[worker_id] = latest["utilization"]
            
            # Identify overloaded and underloaded workers
            overloaded_workers = [
                (worker_id, util) for worker_id, util in worker_utilization.items()
                if util > self.utilization_threshold_high
            ]
            
            underloaded_workers = [
                (worker_id, util) for worker_id, util in worker_utilization.items()
                if util < self.utilization_threshold_low
            ]
            
            # Sort overloaded workers by utilization (highest first)
            overloaded_workers.sort(key=lambda x: x[1], reverse=True)
            
            # Sort underloaded workers by utilization (lowest first)
            underloaded_workers.sort(key=lambda x: x[1])
            
            if not overloaded_workers or not underloaded_workers:
                logger.info("No workers suitable for load balancing")
                return
            
            # Attempt to migrate tasks from overloaded to underloaded workers
            migrations_initiated = 0
            
            for overloaded_id, _ in overloaded_workers:
                # Stop if we've reached maximum simultaneous migrations
                if migrations_initiated >= self.max_simultaneous_migrations:
                    break
                
                # Find tasks that can be migrated from this worker
                migratable_tasks = await self._find_migratable_tasks(overloaded_id)
                
                if not migratable_tasks:
                    logger.info(f"No migratable tasks found for overloaded worker {overloaded_id}")
                    continue
                
                for underloaded_id, _ in underloaded_workers:
                    # Skip if this would exceed max migrations
                    if migrations_initiated >= self.max_simultaneous_migrations:
                        break
                    
                    # Check if we can migrate a task to this worker
                    for task_id, task in migratable_tasks.items():
                        # Skip tasks that are already being migrated
                        if task_id in self.active_migrations:
                            continue
                        
                        # Check if worker can handle this task
                        if await self._can_worker_handle_task(underloaded_id, task):
                            # Initiate migration
                            success = await self._migrate_task(task_id, overloaded_id, underloaded_id)
                            
                            if success:
                                migrations_initiated += 1
                                logger.info(f"Initiated migration of task {task_id} from worker {overloaded_id} to {underloaded_id}")
                                
                                # Check if we've reached the limit
                                if migrations_initiated >= self.max_simultaneous_migrations:
                                    break
            
            if migrations_initiated > 0:
                logger.info(f"Initiated {migrations_initiated} task migrations for load balancing")
            else:
                logger.info("No suitable task migrations found for load balancing")
            
        except Exception as e:
            logger.error(f"Error balancing load: {str(e)}")
    
    async def _find_migratable_tasks(self, worker_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Find tasks that can be migrated from a worker.
        
        Args:
            worker_id: Worker ID to find migratable tasks for
            
        Returns:
            Dictionary of migratable tasks (task_id -> task info)
        """
        migratable_tasks = {}
        
        # Find all tasks assigned to this worker
        for task_id, assigned_worker_id in self.coordinator.running_tasks.items():
            if assigned_worker_id != worker_id:
                continue
            
            # Skip if task doesn't exist
            if task_id not in self.coordinator.tasks:
                continue
            
            task = self.coordinator.tasks[task_id]
            
            # Skip tasks that are almost complete
            # This would require task progress reporting, which we might not have
            # For now, skip tasks that have been running for a long time (assumption that they're almost done)
            if "started" in task:
                try:
                    started = datetime.fromisoformat(task["started"])
                    running_time = (datetime.now() - started).total_seconds()
                    
                    # Skip tasks that have been running for more than 5 minutes
                    # This is a simple heuristic and might need adjustment
                    if running_time > 300:  # 5 minutes
                        continue
                except (ValueError, TypeError):
                    pass
            
            # Add task to migratable tasks
            migratable_tasks[task_id] = task
        
        return migratable_tasks
    
    async def _can_worker_handle_task(self, worker_id: str, task: Dict[str, Any]) -> bool:
        """
        Check if a worker can handle a task.
        
        Args:
            worker_id: Worker ID to check
            task: Task to check
            
        Returns:
            True if the worker can handle the task, False otherwise
        """
        # Skip if worker doesn't exist
        if worker_id not in self.coordinator.workers:
            return False
        
        worker = self.coordinator.workers[worker_id]
        
        # Skip inactive workers
        if worker.get("status") != "active":
            return False
        
        # Check task requirements against worker capabilities
        task_requirements = task.get("requirements", {})
        worker_capabilities = worker.get("capabilities", {})
        
        # Check required hardware
        required_hardware = task_requirements.get("hardware", [])
        if required_hardware:
            worker_hardware = worker_capabilities.get("hardware", [])
            if not all(hw in worker_hardware for hw in required_hardware):
                return False
        
        # Check memory requirements
        min_memory_gb = task_requirements.get("min_memory_gb", 0)
        if min_memory_gb > 0:
            worker_memory_gb = worker_capabilities.get("memory", {}).get("total_gb", 0)
            if worker_memory_gb < min_memory_gb:
                return False
        
        # Check CUDA compute capability
        min_cuda_compute = task_requirements.get("min_cuda_compute", 0)
        if min_cuda_compute > 0:
            worker_cuda_compute = float(worker_capabilities.get("gpu", {}).get("cuda_compute", 0))
            if worker_cuda_compute < min_cuda_compute:
                return False
        
        return True
    
    async def _migrate_task(self, task_id: str, source_worker_id: str, target_worker_id: str) -> bool:
        """
        Migrate a task from one worker to another.
        
        Args:
            task_id: Task ID to migrate
            source_worker_id: Source worker ID
            target_worker_id: Target worker ID
            
        Returns:
            True if migration was initiated successfully, False otherwise
        """
        # Skip if either worker doesn't exist
        if source_worker_id not in self.coordinator.workers or target_worker_id not in self.coordinator.workers:
            logger.warning(f"Cannot migrate task {task_id}: Worker does not exist")
            return False
        
        # Skip if the task doesn't exist
        if task_id not in self.coordinator.tasks:
            logger.warning(f"Cannot migrate task {task_id}: Task does not exist")
            return False
        
        # Get task
        task = self.coordinator.tasks[task_id]
        
        try:
            # Step 1: Mark task as "migrating"
            task["status"] = "migrating"
            task["migration"] = {
                "source_worker_id": source_worker_id,
                "target_worker_id": target_worker_id,
                "start_time": datetime.now().isoformat()
            }
            
            # Step 2: Cancel task on source worker
            if source_worker_id in self.coordinator.worker_connections:
                try:
                    await self.coordinator.worker_connections[source_worker_id].send_json({
                        "type": "cancel_task",
                        "task_id": task_id,
                        "reason": "migration"
                    })
                    logger.info(f"Sent cancellation request for task {task_id} to worker {source_worker_id}")
                except Exception as e:
                    logger.error(f"Error sending cancellation request to worker {source_worker_id}: {str(e)}")
                    return False
            
            # Step 3: Add migration to active migrations
            self.active_migrations[task_id] = {
                "task_id": task_id,
                "source_worker_id": source_worker_id,
                "target_worker_id": target_worker_id,
                "start_time": datetime.now().isoformat(),
                "status": "cancelling"
            }
            
            # Migration initiated successfully
            return True
            
        except Exception as e:
            logger.error(f"Error initiating migration for task {task_id}: {str(e)}")
            return False
    
    async def handle_task_cancelled_for_migration(self, task_id: str, source_worker_id: str):
        """
        Handle task cancellation for migration.
        
        Args:
            task_id: Task ID
            source_worker_id: Source worker ID
        """
        # Skip if this task is not being migrated
        if task_id not in self.active_migrations:
            logger.warning(f"Task {task_id} cancellation received but not in active migrations")
            return
        
        # Get migration info
        migration = self.active_migrations[task_id]
        
        # Skip if source worker doesn't match
        if migration["source_worker_id"] != source_worker_id:
            logger.warning(f"Task {task_id} cancellation received from unexpected worker {source_worker_id}")
            return
        
        try:
            # Update migration status
            migration["status"] = "assigning"
            migration["cancel_time"] = datetime.now().isoformat()
            
            # Get task
            if task_id not in self.coordinator.tasks:
                logger.warning(f"Task {task_id} not found for migration")
                return
            
            task = self.coordinator.tasks[task_id]
            
            # Update task status to pending (so it can be assigned again)
            task["status"] = "pending"
            if "started" in task:
                del task["started"]
            if "worker_id" in task:
                del task["worker_id"]
            
            # Add to pending tasks
            self.coordinator.pending_tasks.add(task_id)
            
            # Remove from running tasks
            if task_id in self.coordinator.running_tasks:
                del self.coordinator.running_tasks[task_id]
            
            # Try to assign task to target worker
            target_worker_id = migration["target_worker_id"]
            
            # Add this back to the task so it can be used by the task scheduler
            task["preferred_worker_id"] = target_worker_id
            
            # Update database
            self.coordinator.db.execute(
                """
                UPDATE distributed_tasks
                SET status = 'pending', worker_id = NULL, start_time = NULL
                WHERE task_id = ?
                """,
                (task_id,)
            )
            
            logger.info(f"Task {task_id} cancelled on source worker, now pending reassignment to target worker")
            
            # Assign pending tasks (will include our migrated task)
            await self.coordinator._assign_pending_tasks()
            
            # Update migration status
            migration["status"] = "assigned"
            migration["assign_time"] = datetime.now().isoformat()
            
            # Check if assignment was successful
            if task_id in self.coordinator.running_tasks:
                actual_worker_id = self.coordinator.running_tasks[task_id]
                migration["actual_worker_id"] = actual_worker_id
                
                # Check if assigned to expected worker
                if actual_worker_id == target_worker_id:
                    logger.info(f"Task {task_id} successfully migrated to target worker {target_worker_id}")
                else:
                    logger.warning(f"Task {task_id} assigned to different worker {actual_worker_id} than target {target_worker_id}")
            else:
                logger.warning(f"Task {task_id} not assigned to any worker after migration")
                migration["status"] = "failed"
            
        except Exception as e:
            logger.error(f"Error handling task {task_id} cancellation for migration: {str(e)}")
            
            # Mark migration as failed
            if task_id in self.active_migrations:
                self.active_migrations[task_id]["status"] = "failed"
                self.active_migrations[task_id]["error"] = str(e)
    
    async def cleanup_migrations(self):
        """Clean up completed migrations."""
        now = datetime.now()
        
        # Identify completed migrations
        completed_migrations = []
        
        for task_id, migration in list(self.active_migrations.items()):
            # Skip recent migrations (less than 60 seconds old)
            try:
                start_time = datetime.fromisoformat(migration["start_time"])
                age = (now - start_time).total_seconds()
                
                if age < 60:
                    continue
            except (ValueError, TypeError, KeyError):
                pass
            
            # Check if migration is complete
            status = migration.get("status", "")
            
            if status in ["assigned", "failed"]:
                # Migration is complete, move to history
                migration["end_time"] = now.isoformat()
                self.migration_history.append(migration)
                completed_migrations.append(task_id)
            
            # Also clean up very old migrations (more than 10 minutes old)
            try:
                start_time = datetime.fromisoformat(migration["start_time"])
                age = (now - start_time).total_seconds()
                
                if age > 600:  # 10 minutes
                    logger.warning(f"Cleaning up stale migration for task {task_id} (age: {age:.1f}s)")
                    migration["end_time"] = now.isoformat()
                    migration["status"] = "timeout"
                    self.migration_history.append(migration)
                    completed_migrations.append(task_id)
            except (ValueError, TypeError, KeyError):
                pass
        
        # Remove completed migrations from active migrations
        for task_id in completed_migrations:
            if task_id in self.active_migrations:
                del self.active_migrations[task_id]
        
        # Limit migration history to last 100 entries
        if len(self.migration_history) > 100:
            self.migration_history = self.migration_history[-100:]
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the load balancer.
        
        Returns:
            Statistics about the load balancer
        """
        now = datetime.now()
        
        # Calculate system-wide utilization
        total_utilization = 0.0
        worker_utils = []
        
        for worker_id, history in self.worker_performance_history.items():
            if history:
                # Get latest performance record
                latest = history[-1]
                util = latest["utilization"]
                total_utilization += util
                worker_utils.append(util)
        
        # Calculate stats
        avg_utilization = total_utilization / len(worker_utils) if worker_utils else 0
        min_utilization = min(worker_utils) if worker_utils else 0
        max_utilization = max(worker_utils) if worker_utils else 0
        utilization_stdev = (sum((u - avg_utilization) ** 2 for u in worker_utils) / len(worker_utils)) ** 0.5 if worker_utils else 0
        
        # Count migrations in different time windows
        migrations_last_hour = 0
        migrations_last_day = 0
        
        for migration in self.migration_history:
            try:
                end_time = datetime.fromisoformat(migration.get("end_time", "1970-01-01T00:00:00"))
                age = (now - end_time).total_seconds()
                
                if age <= 3600:  # 1 hour
                    migrations_last_hour += 1
                
                if age <= 86400:  # 1 day
                    migrations_last_day += 1
            except (ValueError, TypeError):
                pass
        
        # Build stats
        stats = {
            "system_utilization": {
                "average": avg_utilization,
                "min": min_utilization,
                "max": max_utilization,
                "std_dev": utilization_stdev,
                "imbalance_score": max_utilization - min_utilization if worker_utils else 0,
            },
            "active_workers": len(worker_utils),
            "migrations": {
                "active": len(self.active_migrations),
                "last_hour": migrations_last_hour,
                "last_day": migrations_last_day,
                "total_history": len(self.migration_history),
            },
            "config": {
                "check_interval": self.check_interval,
                "utilization_threshold_high": self.utilization_threshold_high,
                "utilization_threshold_low": self.utilization_threshold_low,
                "enable_task_migration": self.enable_task_migration,
                "max_simultaneous_migrations": self.max_simultaneous_migrations,
            }
        }
        
        return stats