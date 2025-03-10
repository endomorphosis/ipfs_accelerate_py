#!/usr/bin/env python3
"""
Distributed Testing Framework - Health Monitoring Module

This module provides health monitoring functionality for the distributed testing framework.
It tracks worker health, monitors task execution, and implements auto-recovery mechanisms.

Usage:
    Import this module in coordinator.py to enable health monitoring features.
    """

    import asyncio
    import logging
    import time
    from datetime import datetime, timedelta
    from typing import Dict, List, Optional, Any, Set

# Configure logging
    logging.basicConfig()))))))))))))))))))))
    level=logging.INFO,
    format='%()))))))))))))))))))))asctime)s - %()))))))))))))))))))))name)s - %()))))))))))))))))))))levelname)s - %()))))))))))))))))))))message)s'
    )
    logger = logging.getLogger()))))))))))))))))))))__name__)

class HealthMonitor:
    """Health monitor for distributed testing framework."""
    
    def __init__()))))))))))))))))))))
    self,
    coordinator,
    heartbeat_timeout: int = 30,
    task_timeout_multiplier: float = 3.0,
    auto_recovery: bool = True,
    check_interval: int = 10
    ):
        """
        Initialize the health monitor.
        
        Args:
            coordinator: Reference to the coordinator instance
            heartbeat_timeout: Timeout for worker heartbeats in seconds
            task_timeout_multiplier: Multiplier for task timeout based on estimated execution time
            auto_recovery: Whether to enable automatic recovery
            check_interval: Interval for health checks in seconds
            """
            self.coordinator = coordinator
            self.heartbeat_timeout = heartbeat_timeout
            self.task_timeout_multiplier = task_timeout_multiplier
            self.auto_recovery = auto_recovery
            self.check_interval = check_interval
        
        # Task timeout tracking
            self.task_timeout_estimates: Dict[str, float] = {}}}}}}}}
            ,
        # Worker health metrics
            self.worker_health_metrics: Dict[str, Dict[str, Any]] = {}}}}}}}}
            ,,
        # Task health metrics
            self.task_health_metrics: Dict[str, Dict[str, Any]] = {}}}}}}}}
            ,,
        # Recovery tracking
            self.recovery_attempts: Dict[str, int] = {}}}}}}}}  # worker_id -> attempts,
            self.task_recovery_attempts: Dict[str, int] = {}}}}}}}}  # task_id -> attempts
            ,
            logger.info()))))))))))))))))))))"Health monitor initialized")
    
    async def start_monitoring()))))))))))))))))))))self):
        """Start the health monitoring loop."""
        logger.info()))))))))))))))))))))"Starting health monitoring loop")
        
        while True:
            try:
                # Check worker health
                await self.check_worker_health())))))))))))))))))))))
                
                # Check task health
                await self.check_task_health())))))))))))))))))))))
                
                # Update health metrics in database
                await self.update_health_metrics_in_db())))))))))))))))))))))
                
            except Exception as e:
                logger.error()))))))))))))))))))))f"Error in health monitoring loop: {}}}}}}}str()))))))))))))))))))))e)}")
            
            # Sleep until next check
                await asyncio.sleep()))))))))))))))))))))self.check_interval)
    
    async def check_worker_health()))))))))))))))))))))self):
        """Check health of all workers."""
        now = datetime.now())))))))))))))))))))))
        
        for worker_id, worker in list()))))))))))))))))))))self.coordinator.workers.items())))))))))))))))))))))):
            # Skip offline workers
            if worker.get()))))))))))))))))))))"status") == "offline":
            continue
            
            # Check heartbeat timestamp
            last_heartbeat_str = worker.get()))))))))))))))))))))"last_heartbeat", "1970-01-01T00:00:00")
            try:
                last_heartbeat = datetime.fromisoformat()))))))))))))))))))))last_heartbeat_str)
                heartbeat_age = ()))))))))))))))))))))now - last_heartbeat).total_seconds())))))))))))))))))))))
            except ()))))))))))))))))))))ValueError, TypeError):
                logger.warning()))))))))))))))))))))f"Invalid heartbeat timestamp for worker {}}}}}}}worker_id}: {}}}}}}}last_heartbeat_str}")
                heartbeat_age = float()))))))))))))))))))))'inf')
            
            # Update health metrics
                self.worker_health_metrics[worker_id] = {}}}}}}},
                "worker_id": worker_id,
                "last_heartbeat": last_heartbeat_str,
                "heartbeat_age_seconds": heartbeat_age,
                "status": worker.get()))))))))))))))))))))"status", "unknown"),
                "check_time": now.isoformat()))))))))))))))))))))),
                }
            
            # Check if heartbeat is too old:
            if heartbeat_age > self.heartbeat_timeout:
                logger.warning()))))))))))))))))))))f"Worker {}}}}}}}worker_id} heartbeat is {}}}}}}}heartbeat_age:.1f} seconds old ()))))))))))))))))))))timeout: {}}}}}}}self.heartbeat_timeout}s)")
                
                # Mark worker as offline
                worker["status"] = "offline"
                ,
                # Update database
                self.coordinator.db.execute()))))))))))))))))))))
                """
                UPDATE worker_nodes
                SET status = 'offline', last_heartbeat = ?
                WHERE worker_id = ?
                """,
                ()))))))))))))))))))))now, worker_id)
                )
                
                # Handle failed tasks
                await self.handle_worker_failure()))))))))))))))))))))worker_id)
                
                # Attempt recovery if enabled:
                if self.auto_recovery:
                    await self.attempt_worker_recovery()))))))))))))))))))))worker_id)
    
    async def check_task_health()))))))))))))))))))))self):
        """Check health of all running tasks."""
        now = datetime.now())))))))))))))))))))))
        
        for task_id, worker_id in list()))))))))))))))))))))self.coordinator.running_tasks.items())))))))))))))))))))))):
            # Skip if task no longer exists::
            if task_id not in self.coordinator.tasks:
            continue
            
            task = self.coordinator.tasks[task_id],
            ,
            # Calculate task running time
            if "started" in task:
                try:
                    started = datetime.fromisoformat()))))))))))))))))))))task["started"],),
                    running_time = ()))))))))))))))))))))now - started).total_seconds())))))))))))))))))))))
                except ()))))))))))))))))))))ValueError, TypeError):
                    logger.warning()))))))))))))))))))))f"Invalid start time for task {}}}}}}}task_id}: {}}}}}}}task.get()))))))))))))))))))))'started')}")
                    running_time = 0
            else:
                running_time = 0
            
            # Get timeout for this task
                timeout = self.get_task_timeout()))))))))))))))))))))task_id, task)
            
            # Update health metrics
                self.task_health_metrics[task_id], = {}}}}}}},
                "task_id": task_id,
                "worker_id": worker_id,
                "running_time_seconds": running_time,
                "timeout_seconds": timeout,
                "status": task.get()))))))))))))))))))))"status", "unknown"),
                "check_time": now.isoformat()))))))))))))))))))))),
                }
            
            # Check if task has timed out:
            if running_time > timeout:
                logger.warning()))))))))))))))))))))f"Task {}}}}}}}task_id} has been running for {}}}}}}}running_time:.1f} seconds on worker {}}}}}}}worker_id} ()))))))))))))))))))))timeout: {}}}}}}}timeout}s)")
                
                # Handle task timeout
                await self.handle_task_timeout()))))))))))))))))))))task_id, worker_id)
    
                def get_task_timeout()))))))))))))))))))))self, task_id: str, task: Dict[str, Any]) -> float:,
                """
                Get timeout for a task.
        
        Args:
            task_id: Task ID
            task: Task information
            
        Returns:
            Timeout in seconds
            """
        # Check if task has explicit timeout in config
        task_config = task.get()))))))))))))))))))))"config", {}}}}}}}}):
        if isinstance()))))))))))))))))))))task_config, dict) and "timeout_seconds" in task_config:
            return float()))))))))))))))))))))task_config["timeout_seconds"])
            ,
        # Check if we have an estimated timeout for this task type
            task_type = task.get()))))))))))))))))))))"type", "unknown")
            estimated_timeout = self.task_timeout_estimates.get()))))))))))))))))))))f"{}}}}}}}task_type}", None)
        :
        if estimated_timeout is not None:
            # Apply multiplier for safety margin
            return estimated_timeout * self.task_timeout_multiplier
        
        # Default timeouts based on task type
            default_timeouts = {}}}}}}}
            "benchmark": 600,  # 10 minutes
            "test": 300,       # 5 minutes
            "custom": 300,     # 5 minutes
            }
        
            return default_timeouts.get()))))))))))))))))))))task_type, 600)  # Default to 10 minutes
    
    async def handle_worker_failure()))))))))))))))))))))self, worker_id: str):
        """
        Handle worker failure.
        
        Args:
            worker_id: Worker ID
            """
            logger.info()))))))))))))))))))))f"Handling failure of worker {}}}}}}}worker_id}")
        
        # Find all tasks assigned to this worker
        for task_id, assigned_worker_id in list()))))))))))))))))))))self.coordinator.running_tasks.items())))))))))))))))))))))):
            if assigned_worker_id == worker_id:
                logger.info()))))))))))))))))))))f"Marking task {}}}}}}}task_id} as failed due to worker failure")
                
                # Update task status
                await self.coordinator._update_task_status()))))))))))))))))))))task_id, "failed", {}}}}}}}
                "error": f"Worker {}}}}}}}worker_id} failed or disconnected"
                })
                
                # Update failure metrics in database
                self.coordinator.db.execute()))))))))))))))))))))
                """
                INSERT INTO task_execution_history ()))))))))))))))))))))
                task_id, worker_id, attempt, status, start_time, end_time,
                execution_time_seconds, error_message, hardware_metrics
                ) VALUES ()))))))))))))))))))))?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ()))))))))))))))))))))
                task_id, worker_id,
                self.coordinator.tasks[task_id],.get()))))))))))))))))))))"attempts", 1),
                "failed",
                datetime.fromisoformat()))))))))))))))))))))self.coordinator.tasks[task_id],.get()))))))))))))))))))))"started", datetime.now()))))))))))))))))))))).isoformat()))))))))))))))))))))))),
                datetime.now()))))))))))))))))))))),
                0,
                f"Worker {}}}}}}}worker_id} failed or disconnected",
                "{}}}}}}}}"
                )
                )
                
                # If auto-recovery is enabled, requeue the task
                if self.auto_recovery:
                    await self.requeue_task()))))))))))))))))))))task_id)
    
    async def handle_task_timeout()))))))))))))))))))))self, task_id: str, worker_id: str):
        """
        Handle task timeout.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID
            """
            logger.info()))))))))))))))))))))f"Handling timeout of task {}}}}}}}task_id} on worker {}}}}}}}worker_id}")
        
        # Update task status
            await self.coordinator._update_task_status()))))))))))))))))))))task_id, "failed", {}}}}}}}
            "error": "Task execution timed out"
            })
        
        # Update timeout metrics in database
            self.coordinator.db.execute()))))))))))))))))))))
            """
            INSERT INTO task_execution_history ()))))))))))))))))))))
            task_id, worker_id, attempt, status, start_time, end_time,
            execution_time_seconds, error_message, hardware_metrics
            ) VALUES ()))))))))))))))))))))?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ()))))))))))))))))))))
            task_id, worker_id,
            self.coordinator.tasks[task_id],.get()))))))))))))))))))))"attempts", 1),
            "failed",
            datetime.fromisoformat()))))))))))))))))))))self.coordinator.tasks[task_id],.get()))))))))))))))))))))"started", datetime.now()))))))))))))))))))))).isoformat()))))))))))))))))))))))),
            datetime.now()))))))))))))))))))))),
            0,
            "Task execution timed out",
            "{}}}}}}}}"
            )
            )
        
        # If auto-recovery is enabled, requeue the task
        if self.auto_recovery:
            await self.requeue_task()))))))))))))))))))))task_id)
    
    async def attempt_worker_recovery()))))))))))))))))))))self, worker_id: str):
        """
        Attempt to recover a failed worker.
        
        Args:
            worker_id: Worker ID
            """
        # Increment recovery attempts
            self.recovery_attempts[worker_id] = self.recovery_attempts.get()))))))))))))))))))))worker_id, 0) + 1
            ,
        # Log recovery attempt
            logger.info()))))))))))))))))))))f"Attempting recovery of worker {}}}}}}}worker_id} ()))))))))))))))))))))attempt {}}}}}}}self.recovery_attempts[worker_id]})")
            ,
        # For now, we rely on worker's auto-reconnect functionality
        # In the future, we could implement more sophisticated recovery mechanisms
        # such as sending signals to restart the worker process
        
        # Update worker status in database
            self.coordinator.db.execute()))))))))))))))))))))
            """
            UPDATE worker_nodes
            SET status = 'recovery_pending'
            WHERE worker_id = ?
            """,
            ()))))))))))))))))))))worker_id,)
            )
        
        # Update worker status in memory
        if worker_id in self.coordinator.workers:
            self.coordinator.workers[worker_id]["status"] = "recovery_pending"
            ,
    async def requeue_task()))))))))))))))))))))self, task_id: str):
        """
        Requeue a failed task.
        
        Args:
            task_id: Task ID
            """
        # Skip if task no longer exists::
        if task_id not in self.coordinator.tasks:
            return
        
        # Get task retry policy
            task = self.coordinator.tasks[task_id],
            task_config = task.get()))))))))))))))))))))"config", {}}}}}}}})
            retry_policy = task_config.get()))))))))))))))))))))"retry_policy", {}}}}}}}})
        
        # Get max retries from policy ()))))))))))))))))))))default to 3)
            max_retries = retry_policy.get()))))))))))))))))))))"max_retries", 3)
        
        # Increment task recovery attempts
            self.task_recovery_attempts[task_id], = self.task_recovery_attempts.get()))))))))))))))))))))task_id, 0) + 1
        
        # Check if max retries exceeded:
        if self.task_recovery_attempts[task_id], > max_retries:
            logger.warning()))))))))))))))))))))f"Max retries ())))))))))))))))))))){}}}}}}}max_retries}) exceeded for task {}}}}}}}task_id}")
            return
        
        # Log requeue attempt
            logger.info()))))))))))))))))))))f"Requeuing task {}}}}}}}task_id} ()))))))))))))))))))))attempt {}}}}}}}self.task_recovery_attempts[task_id],}/{}}}}}}}max_retries})")
        
        # Update task in database
            self.coordinator.db.execute()))))))))))))))))))))
            """
            UPDATE distributed_tasks
            SET status = 'pending', worker_id = NULL, start_time = NULL
            WHERE task_id = ?
            """,
            ()))))))))))))))))))))task_id,)
            )
        
        # Update task in memory
            task["status"] = "pending",
        if "started" in task:
            del task["started"],
        if "worker_id" in task:
            del task["worker_id"]
            ,
        # Add to pending tasks
            self.coordinator.pending_tasks.add()))))))))))))))))))))task_id)
        
        # Remove from running tasks
        if task_id in self.coordinator.running_tasks:
            del self.coordinator.running_tasks[task_id],
        
        # Try to assign task to a new worker
            await self.coordinator._assign_pending_tasks())))))))))))))))))))))
    
    async def update_health_metrics_in_db()))))))))))))))))))))self):
        """Update health metrics in the database."""
        now = datetime.now())))))))))))))))))))))
        
        try:
            # Update worker health metrics in database
            for worker_id, metrics in self.worker_health_metrics.items()))))))))))))))))))))):
                # Check if worker health metrics table exists
                self.coordinator.db.execute()))))))))))))))))))))
                """
                CREATE TABLE IF NOT EXISTS worker_health_metrics ()))))))))))))))))))))
                id INTEGER PRIMARY KEY,
                worker_id VARCHAR,
                check_time TIMESTAMP,
                heartbeat_age_seconds FLOAT,
                status VARCHAR,
                metrics JSON
                )
                """
                )
                
                # Insert metrics
                self.coordinator.db.execute()))))))))))))))))))))
                """
                INSERT INTO worker_health_metrics ()))))))))))))))))))))
                worker_id, check_time, heartbeat_age_seconds, status, metrics
                ) VALUES ()))))))))))))))))))))?, ?, ?, ?, ?)
                """,
                ()))))))))))))))))))))
                worker_id,
                now,
                metrics.get()))))))))))))))))))))"heartbeat_age_seconds", 0),
                metrics.get()))))))))))))))))))))"status", "unknown"),
                "{}}}}}}}}"  # Empty JSON for now
                )
                )
            
            # Update task health metrics in database:
            for task_id, metrics in self.task_health_metrics.items()))))))))))))))))))))):
                # Check if task health metrics table exists
                self.coordinator.db.execute()))))))))))))))))))))
                """
                CREATE TABLE IF NOT EXISTS task_health_metrics ()))))))))))))))))))))
                id INTEGER PRIMARY KEY,
                task_id VARCHAR,
                worker_id VARCHAR,
                check_time TIMESTAMP,
                running_time_seconds FLOAT,
                timeout_seconds FLOAT,
                status VARCHAR,
                metrics JSON
                )
                """
                )
                
                # Insert metrics
                self.coordinator.db.execute()))))))))))))))))))))
                """
                INSERT INTO task_health_metrics ()))))))))))))))))))))
                task_id, worker_id, check_time, running_time_seconds,
                timeout_seconds, status, metrics
                ) VALUES ()))))))))))))))))))))?, ?, ?, ?, ?, ?, ?)
                """,
                ()))))))))))))))))))))
                task_id,
                metrics.get()))))))))))))))))))))"worker_id", "unknown"),
                now,
                metrics.get()))))))))))))))))))))"running_time_seconds", 0),
                metrics.get()))))))))))))))))))))"timeout_seconds", 0),
                metrics.get()))))))))))))))))))))"status", "unknown"),
                "{}}}}}}}}"  # Empty JSON for now
                )
                )
            :
        except Exception as e:
            logger.error()))))))))))))))))))))f"Error updating health metrics in database: {}}}}}}}str()))))))))))))))))))))e)}")
    
    def update_task_timeout_estimate()))))))))))))))))))))self, task_type: str, execution_time: float):
        """
        Update task timeout estimate based on actual execution time.
        
        Args:
            task_type: Task type
            execution_time: Actual execution time in seconds
            """
        # Only update if execution time is reasonable:
            if execution_time <= 0 or execution_time > 3600 * 24:  # 24 hours max
        return
        
        key = f"{}}}}}}}task_type}"
        
        if key not in self.task_timeout_estimates:
            # First estimate
            self.task_timeout_estimates[key], = execution_time,
        else:
            # Exponential moving average ()))))))))))))))))))))alpha = 0.3)
            alpha = 0.3
            current = self.task_timeout_estimates[key],
            self.task_timeout_estimates[key], = ()))))))))))))))))))))1 - alpha) * current + alpha * execution_time
        
            logger.debug()))))))))))))))))))))f"Updated timeout estimate for {}}}}}}}task_type}: {}}}}}}}self.task_timeout_estimates[key],:.2f}s")
    
            def get_worker_health_summary()))))))))))))))))))))self) -> Dict[str, Any]:,,
            """
            Get summary of worker health.
        
        Returns:
            Summary of worker health
            """
            total_workers = len()))))))))))))))))))))self.coordinator.workers)
            active_workers = sum()))))))))))))))))))))1 for w in self.coordinator.workers.values()))))))))))))))))))))) if w.get()))))))))))))))))))))"status") == "active")
            idle_workers = sum()))))))))))))))))))))1 for w in self.coordinator.workers.values()))))))))))))))))))))) if w.get()))))))))))))))))))))"status") == "idle")
            offline_workers = sum()))))))))))))))))))))1 for w in self.coordinator.workers.values()))))))))))))))))))))) if w.get()))))))))))))))))))))"status") == "offline")
        
        summary = {}}}}}}}:
            "total_workers": total_workers,
            "active_workers": active_workers,
            "idle_workers": idle_workers,
            "offline_workers": offline_workers,
            "active_percentage": ()))))))))))))))))))))active_workers / total_workers * 100) if total_workers > 0 else 0,:
                "recovery_attempts": sum()))))))))))))))))))))self.recovery_attempts.values())))))))))))))))))))))),
                "health_check_interval": self.check_interval,
                "heartbeat_timeout": self.heartbeat_timeout,
                "auto_recovery_enabled": self.auto_recovery,
                }
        
            return summary
    
            def get_task_health_summary()))))))))))))))))))))self) -> Dict[str, Any]:,,
            """
            Get summary of task health.
        
        Returns:
            Summary of task health
            """
            total_tasks = len()))))))))))))))))))))self.coordinator.tasks)
            pending_tasks = len()))))))))))))))))))))self.coordinator.pending_tasks)
            running_tasks = len()))))))))))))))))))))self.coordinator.running_tasks)
            completed_tasks = len()))))))))))))))))))))self.coordinator.completed_tasks)
            failed_tasks = len()))))))))))))))))))))self.coordinator.failed_tasks)
        
            summary = {}}}}}}}
            "total_tasks": total_tasks,
            "pending_tasks": pending_tasks,
            "running_tasks": running_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "completion_rate": ()))))))))))))))))))))completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,::
            "failure_rate": ()))))))))))))))))))))failed_tasks / total_tasks * 100) if total_tasks > 0 else 0,::
                "task_recovery_attempts": sum()))))))))))))))))))))self.task_recovery_attempts.values())))))))))))))))))))))),
                "timeout_estimates": dict()))))))))))))))))))))self.task_timeout_estimates),
                }
        
                return summary