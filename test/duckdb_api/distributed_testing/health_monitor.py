#!/usr/bin/env python3
"""
Distributed Testing Framework - Health Monitor

This module implements the health monitoring system for the distributed testing
framework. It's responsible for:

- Worker node health monitoring and status tracking
- Failure detection and handling
- Automatic recovery mechanisms
- Resource monitoring and alerting
- Performance anomaly detection
- Task timeout monitoring
- Worker connection status monitoring

Usage:
    This module is used by the coordinator server to monitor the health of
    worker nodes and take appropriate actions when issues are detected.
"""

import os
import sys
import json
import time
import logging
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("health_monitor")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Worker status constants
WORKER_STATUS_REGISTERED = "registered"
WORKER_STATUS_ACTIVE = "active"
WORKER_STATUS_BUSY = "busy"
WORKER_STATUS_UNAVAILABLE = "unavailable"
WORKER_STATUS_DISCONNECTED = "disconnected"

# Health status constants
HEALTH_STATUS_HEALTHY = "healthy"
HEALTH_STATUS_WARNING = "warning"
HEALTH_STATUS_CRITICAL = "critical"
HEALTH_STATUS_UNKNOWN = "unknown"

# Default thresholds
DEFAULT_CPU_WARNING_THRESHOLD = 80.0
DEFAULT_CPU_CRITICAL_THRESHOLD = 95.0
DEFAULT_MEMORY_WARNING_THRESHOLD = 85.0
DEFAULT_MEMORY_CRITICAL_THRESHOLD = 95.0
DEFAULT_HEARTBEAT_WARNING_INTERVAL = 60  # seconds
DEFAULT_HEARTBEAT_CRITICAL_INTERVAL = 180  # seconds
DEFAULT_TASK_SUCCESS_WARNING_THRESHOLD = 0.7  # 70% success rate
DEFAULT_TASK_SUCCESS_CRITICAL_THRESHOLD = 0.5  # 50% success rate
DEFAULT_HEALTH_CHECK_INTERVAL = 30  # seconds

class HealthMonitor:
    """Health monitoring system for the distributed testing framework."""
    
    def __init__(self, worker_manager=None, task_scheduler=None, db_manager=None):
        """Initialize the health monitor.
        
        Args:
            worker_manager: Worker manager instance
            task_scheduler: Task scheduler instance
            db_manager: Database manager instance
        """
        self.worker_manager = worker_manager
        self.task_scheduler = task_scheduler
        self.db_manager = db_manager
        
        # Worker health status and metrics
        self.worker_health = {}  # worker_id -> health status dict
        self.health_history = {}  # worker_id -> list of historical health status
        self.task_timeout_checks = {}  # task_id -> last check time
        
        # Performance metrics for anomaly detection
        self.performance_baselines = {}  # worker_id -> baseline metrics
        self.anomaly_history = {}  # worker_id -> list of anomalies
        
        # Auto-recovery tracking
        self.recovery_attempts = {}  # worker_id -> list of recovery attempts
        self.recovery_success = {}  # worker_id -> success count
        
        # Configuration
        self.config = {
            "cpu_warning_threshold": DEFAULT_CPU_WARNING_THRESHOLD,
            "cpu_critical_threshold": DEFAULT_CPU_CRITICAL_THRESHOLD,
            "memory_warning_threshold": DEFAULT_MEMORY_WARNING_THRESHOLD,
            "memory_critical_threshold": DEFAULT_MEMORY_CRITICAL_THRESHOLD,
            "heartbeat_warning_interval": DEFAULT_HEARTBEAT_WARNING_INTERVAL,
            "heartbeat_critical_interval": DEFAULT_HEARTBEAT_CRITICAL_INTERVAL,
            "task_success_warning_threshold": DEFAULT_TASK_SUCCESS_WARNING_THRESHOLD,
            "task_success_critical_threshold": DEFAULT_TASK_SUCCESS_CRITICAL_THRESHOLD,
            "health_history_size": 10,  # How many historical data points to keep
            "health_check_interval": DEFAULT_HEALTH_CHECK_INTERVAL,
            "auto_recovery_enabled": True,  # Enable automatic recovery attempts
            "max_recovery_attempts": 3,  # Maximum number of recovery attempts per worker
            "recovery_backoff_factor": 2.0,  # Backoff factor for recovery attempts
            "recovery_cooldown_period": 600,  # Seconds to wait after max attempts
            "performance_anomaly_detection_enabled": True,  # Enable anomaly detection
            "worker_quarantine_enabled": True,  # Enable quarantining unreliable workers
            "worker_quarantine_threshold": 3,  # Failed recovery attempts before quarantine
            "worker_probation_period": 1800,  # Seconds to keep worker in quarantine
            "alert_on_critical": True,  # Generate alerts for critical health issues
            "alert_on_warning": False,  # Generate alerts for warning health issues
            "alert_recovery_threshold": 3,  # Health checks to pass before recovery
        }
        
        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_stop_event = threading.Event()
        
        # Alert subscribers
        self.alert_subscribers = []
        
        logger.info("Health monitor initialized")
    
    def configure(self, config_updates: Dict[str, Any]):
        """Update the health monitor configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        self.config.update(config_updates)
        logger.info(f"Health monitor configuration updated: {config_updates}")
    
    def start_monitoring(self):
        """Start the health monitoring thread."""
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread already running")
            return
            
        self.monitoring_stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Health monitor thread started")
    
    def stop_monitoring(self):
        """Stop the health monitoring thread."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread not running")
            return
            
        self.monitoring_stop_event.set()
        self.monitoring_thread.join(timeout=5.0)
        if self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread did not stop gracefully")
            
        logger.info("Health monitor thread stopped")
    
    def _monitoring_loop(self):
        """Health monitoring thread function."""
        while not self.monitoring_stop_event.is_set():
            try:
                # Check worker health
                self._check_all_workers_health()
                
                # Check for task timeouts
                if self.task_scheduler:
                    self._check_task_timeouts()
                    
                # Check for performance anomalies
                if self.config["performance_anomaly_detection_enabled"]:
                    self._detect_performance_anomalies()
                    
                # Process any automatic recovery actions
                if self.config["auto_recovery_enabled"]:
                    self._process_auto_recovery()
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                
            # Wait for next interval
            self.monitoring_stop_event.wait(self.config["health_check_interval"])
    
    def _check_all_workers_health(self):
        """Check health status for all known workers."""
        if not self.worker_manager:
            return
            
        # Get all workers from worker manager
        workers = []
        if hasattr(self.worker_manager, "workers"):
            workers = list(self.worker_manager.workers.values())
        elif hasattr(self.worker_manager, "get_all_workers"):
            workers = self.worker_manager.get_all_workers()
            
        for worker in workers:
            worker_id = worker.get("worker_id")
            if not worker_id:
                continue
                
            self._check_worker_health(worker_id, worker)
    
    def _check_worker_health(self, worker_id: str, worker: Dict[str, Any]):
        """Check the health status of a specific worker.
        
        Args:
            worker_id: ID of the worker
            worker: Worker information
        """
        # Initialize worker health if not exists
        if worker_id not in self.worker_health:
            self.worker_health[worker_id] = {
                "status": HEALTH_STATUS_UNKNOWN,
                "last_update": datetime.now(),
                "issues": [],
                "metrics": {},
                "check_count": 0,
                "consecutive_healthy": 0,
                "consecutive_unhealthy": 0
            }
            
        # Initialize health history if not exists
        if worker_id not in self.health_history:
            self.health_history[worker_id] = []
            
        health_status = self.worker_health[worker_id]
        previous_status = health_status["status"]
        issues = []
        
        # Check connection status
        connection_status = worker.get("status")
        if connection_status in [WORKER_STATUS_UNAVAILABLE, WORKER_STATUS_DISCONNECTED]:
            issues.append(f"Worker is {connection_status}")
            
        # Check heartbeat
        last_heartbeat = worker.get("last_heartbeat")
        if last_heartbeat:
            time_since_heartbeat = (datetime.now() - last_heartbeat).total_seconds()
            
            if time_since_heartbeat > self.config["heartbeat_critical_interval"]:
                issues.append(f"No heartbeat for {time_since_heartbeat:.1f}s (critical)")
            elif time_since_heartbeat > self.config["heartbeat_warning_interval"]:
                issues.append(f"No heartbeat for {time_since_heartbeat:.1f}s (warning)")
                
        # Check resource metrics
        if "capabilities" in worker:
            metrics = worker.get("hardware_metrics", {})
            
            # CPU usage
            if "cpu_percent" in metrics:
                cpu_percent = metrics["cpu_percent"]
                
                if cpu_percent > self.config["cpu_critical_threshold"]:
                    issues.append(f"CPU usage at {cpu_percent:.1f}% (critical)")
                elif cpu_percent > self.config["cpu_warning_threshold"]:
                    issues.append(f"CPU usage at {cpu_percent:.1f}% (warning)")
                    
            # Memory usage
            if "memory_percent" in metrics:
                memory_percent = metrics["memory_percent"]
                
                if memory_percent > self.config["memory_critical_threshold"]:
                    issues.append(f"Memory usage at {memory_percent:.1f}% (critical)")
                elif memory_percent > self.config["memory_warning_threshold"]:
                    issues.append(f"Memory usage at {memory_percent:.1f}% (warning)")
                    
            # Store metrics for tracking
            health_status["metrics"] = metrics
        
        # Check task success rate
        if hasattr(self.task_scheduler, "worker_performance"):
            performance = self.task_scheduler.worker_performance.get(worker_id, {})
            if "success_rate" in performance:
                success_rate = performance["success_rate"]
                
                if success_rate < self.config["task_success_critical_threshold"]:
                    issues.append(f"Task success rate at {success_rate*100:.1f}% (critical)")
                elif success_rate < self.config["task_success_warning_threshold"]:
                    issues.append(f"Task success rate at {success_rate*100:.1f}% (warning)")
        
        # Determine overall health status
        if any("critical" in issue.lower() for issue in issues):
            new_status = HEALTH_STATUS_CRITICAL
        elif issues:
            new_status = HEALTH_STATUS_WARNING
        else:
            new_status = HEALTH_STATUS_HEALTHY
            
        # Update health status
        health_status["status"] = new_status
        health_status["last_update"] = datetime.now()
        health_status["issues"] = issues
        health_status["check_count"] += 1
        
        # Update consecutive counts
        if new_status == HEALTH_STATUS_HEALTHY:
            health_status["consecutive_healthy"] += 1
            health_status["consecutive_unhealthy"] = 0
        else:
            health_status["consecutive_healthy"] = 0
            health_status["consecutive_unhealthy"] += 1
            
        # Add to history
        self.health_history[worker_id].append({
            "timestamp": datetime.now(),
            "status": new_status,
            "issues": issues.copy()
        })
        
        # Limit history size
        max_history = self.config["health_history_size"]
        if len(self.health_history[worker_id]) > max_history:
            self.health_history[worker_id] = self.health_history[worker_id][-max_history:]
            
        # Log status changes
        if new_status != previous_status:
            if new_status == HEALTH_STATUS_HEALTHY:
                logger.info(f"Worker {worker_id} health status improved to {new_status}")
            else:
                level = logging.WARNING if new_status == HEALTH_STATUS_WARNING else logging.ERROR
                logger.log(level, f"Worker {worker_id} health status degraded to {new_status}: {', '.join(issues)}")
                
            # Generate alert if configured
            if new_status == HEALTH_STATUS_CRITICAL and self.config["alert_on_critical"]:
                self._generate_alert(worker_id, new_status, issues)
            elif new_status == HEALTH_STATUS_WARNING and self.config["alert_on_warning"]:
                self._generate_alert(worker_id, new_status, issues)
    
    def _check_task_timeouts(self):
        """Check for task timeouts and take appropriate actions."""
        if not self.task_scheduler:
            return
            
        # Get running tasks from task scheduler
        if hasattr(self.task_scheduler, "running_tasks"):
            running_tasks = self.task_scheduler.running_tasks.copy()
        else:
            running_tasks = {}
            
        # Database access for task details
        if not self.db_manager:
            return
            
        # Check each running task
        for task_id, worker_id in running_tasks.items():
            # Get task details
            task = self.db_manager.get_task(task_id)
            if not task:
                continue
                
            # Skip if no start time
            if "start_time" not in task:
                continue
                
            start_time = task["start_time"]
            timeout_seconds = task.get("timeout_seconds", 3600)  # Default 1 hour
            
            # Calculate time running
            time_running = (datetime.now() - start_time).total_seconds()
            
            # Check for timeout
            if time_running > timeout_seconds:
                logger.warning(f"Task {task_id} on worker {worker_id} has timed out after {time_running:.1f}s")
                
                # Let task scheduler handle the timeout
                if hasattr(self.task_scheduler, "check_timeouts"):
                    self.task_scheduler.check_timeouts()
    
    def _detect_performance_anomalies(self):
        """Detect performance anomalies for workers."""
        if not self.task_scheduler or not hasattr(self.task_scheduler, "worker_performance"):
            return
            
        for worker_id, performance in self.task_scheduler.worker_performance.items():
            # Skip if no task types data
            if "task_types" not in performance:
                continue
                
            # Initialize anomaly tracking if not exists
            if worker_id not in self.anomaly_history:
                self.anomaly_history[worker_id] = []
                
            # Check task types for anomalies
            for task_type, type_perf in performance["task_types"].items():
                # Skip if not enough data
                if type_perf.get("task_count", 0) < 5:
                    continue
                    
                # Check execution time trend
                if "execution_times" in type_perf and len(type_perf["execution_times"]) >= 5:
                    times = type_perf["execution_times"][-5:]  # Last 5 times
                    
                    # Calculate z-scores to detect outliers
                    if len(times) >= 3:
                        mean = statistics.mean(times)
                        stdev = statistics.stdev(times) if len(times) > 1 else 0
                        
                        if stdev > 0:
                            z_scores = [(t - mean) / stdev for t in times]
                            
                            # Check for extreme outliers (z-score > 3)
                            outliers = [i for i, z in enumerate(z_scores) if abs(z) > 3]
                            
                            if outliers:
                                # Found anomalies
                                for i in outliers:
                                    anomaly = {
                                        "timestamp": datetime.now(),
                                        "worker_id": worker_id,
                                        "task_type": task_type,
                                        "metric": "execution_time",
                                        "value": times[i],
                                        "mean": mean,
                                        "stdev": stdev,
                                        "z_score": z_scores[i]
                                    }
                                    
                                    self.anomaly_history[worker_id].append(anomaly)
                                    
                                    logger.warning(f"Performance anomaly detected for worker {worker_id}, "
                                                  f"task type {task_type}: execution time {times[i]:.2f}s "
                                                  f"(z-score: {z_scores[i]:.2f})")
                
                # Check success rate drop
                if "success_rates" in type_perf and len(type_perf["success_rates"]) >= 2:
                    rates = type_perf["success_rates"]
                    current_rate = rates[-1]
                    previous_rate = rates[-2]
                    
                    # Detect significant drop (>20% decrease)
                    if previous_rate > 0 and current_rate < previous_rate * 0.8:
                        anomaly = {
                            "timestamp": datetime.now(),
                            "worker_id": worker_id,
                            "task_type": task_type,
                            "metric": "success_rate",
                            "value": current_rate,
                            "previous": previous_rate,
                            "drop_percent": (1 - current_rate / previous_rate) * 100
                        }
                        
                        self.anomaly_history[worker_id].append(anomaly)
                        
                        logger.warning(f"Success rate anomaly detected for worker {worker_id}, "
                                      f"task type {task_type}: dropped from {previous_rate*100:.1f}% "
                                      f"to {current_rate*100:.1f}%")
            
            # Limit anomaly history size
            max_history = self.config["health_history_size"] * 2
            if len(self.anomaly_history[worker_id]) > max_history:
                self.anomaly_history[worker_id] = self.anomaly_history[worker_id][-max_history:]
    
    def _process_auto_recovery(self):
        """Process automatic recovery actions for unhealthy workers."""
        if not self.config["auto_recovery_enabled"]:
            return
            
        for worker_id, health_status in self.worker_health.items():
            # Skip healthy workers
            if health_status["status"] == HEALTH_STATUS_HEALTHY:
                continue
                
            # Skip workers that aren't consistently unhealthy
            min_unhealthy = 3  # Require at least 3 consecutive unhealthy checks
            if health_status["consecutive_unhealthy"] < min_unhealthy:
                continue
                
            # Initialize recovery tracking if not exists
            if worker_id not in self.recovery_attempts:
                self.recovery_attempts[worker_id] = []
                
            # Check if in cooldown period after max attempts
            if self.recovery_attempts[worker_id]:
                last_attempt = self.recovery_attempts[worker_id][-1]["timestamp"]
                time_since_last = (datetime.now() - last_attempt).total_seconds()
                
                if len(self.recovery_attempts[worker_id]) >= self.config["max_recovery_attempts"]:
                    cooldown = self.config["recovery_cooldown_period"]
                    if time_since_last < cooldown:
                        # Still in cooldown
                        continue
                        
                    # Reset attempts after cooldown
                    if time_since_last >= cooldown:
                        self.recovery_attempts[worker_id] = []
                else:
                    # Check backoff for subsequent attempts
                    backoff = self.config["recovery_backoff_factor"]
                    min_interval = self.config["health_check_interval"] * 2
                    attempt_count = len(self.recovery_attempts[worker_id])
                    
                    required_interval = min_interval * (backoff ** (attempt_count - 1))
                    if time_since_last < required_interval:
                        # Not enough time between attempts
                        continue
            
            # Check if worker should be quarantined
            if self.config["worker_quarantine_enabled"]:
                failed_attempts = sum(1 for a in self.recovery_attempts[worker_id] if not a["success"])
                if failed_attempts >= self.config["worker_quarantine_threshold"]:
                    # Quarantine worker
                    self._quarantine_worker(worker_id)
                    continue
            
            # Perform recovery action
            recovery_result = self._recover_worker(worker_id, health_status["issues"])
            
            # Record attempt
            self.recovery_attempts[worker_id].append({
                "timestamp": datetime.now(),
                "issues": health_status["issues"],
                "action": recovery_result["action"],
                "success": recovery_result["success"],
                "details": recovery_result["details"]
            })
            
            # Update success tracking
            if worker_id not in self.recovery_success:
                self.recovery_success[worker_id] = 0
                
            if recovery_result["success"]:
                self.recovery_success[worker_id] += 1
    
    def _recover_worker(self, worker_id: str, issues: List[str]) -> Dict[str, Any]:
        """Attempt to recover an unhealthy worker.
        
        Args:
            worker_id: ID of the worker
            issues: List of issues affecting the worker
            
        Returns:
            Dict containing recovery attempt results
        """
        logger.info(f"Attempting to recover worker {worker_id} with issues: {issues}")
        
        # Determine recovery action based on issues
        disconnected = any("disconnected" in issue.lower() for issue in issues)
        unavailable = any("unavailable" in issue.lower() for issue in issues)
        heartbeat = any("heartbeat" in issue.lower() for issue in issues)
        high_cpu = any("cpu usage" in issue.lower() for issue in issues)
        high_memory = any("memory usage" in issue.lower() for issue in issues)
        low_success = any("success rate" in issue.lower() for issue in issues)
        
        result = {
            "action": "none",
            "success": False,
            "details": "No recovery action taken"
        }
        
        if disconnected or unavailable or heartbeat:
            # Connection issues - try to reconnect or restart
            if hasattr(self.worker_manager, "reconnect_worker"):
                try:
                    reconnect_result = self.worker_manager.reconnect_worker(worker_id)
                    result = {
                        "action": "reconnect",
                        "success": reconnect_result,
                        "details": "Attempted to reconnect worker"
                    }
                except Exception as e:
                    result = {
                        "action": "reconnect",
                        "success": False,
                        "details": f"Error reconnecting worker: {str(e)}"
                    }
        elif high_cpu or high_memory:
            # Resource issues - try to reduce workload
            if hasattr(self.task_scheduler, "redistribute_tasks"):
                try:
                    redistribute_result = self.task_scheduler.redistribute_tasks(worker_id)
                    result = {
                        "action": "redistribute",
                        "success": redistribute_result > 0,  # Success if any tasks redistributed
                        "details": f"Redistributed {redistribute_result} tasks"
                    }
                except Exception as e:
                    result = {
                        "action": "redistribute",
                        "success": False,
                        "details": f"Error redistributing tasks: {str(e)}"
                    }
        elif low_success:
            # Performance issues - try to reset worker state
            if hasattr(self.worker_manager, "reset_worker"):
                try:
                    reset_result = self.worker_manager.reset_worker(worker_id)
                    result = {
                        "action": "reset",
                        "success": reset_result,
                        "details": "Attempted to reset worker state"
                    }
                except Exception as e:
                    result = {
                        "action": "reset",
                        "success": False,
                        "details": f"Error resetting worker: {str(e)}"
                    }
        
        # Log result
        if result["success"]:
            logger.info(f"Recovery of worker {worker_id} successful with action '{result['action']}'")
        else:
            logger.warning(f"Recovery of worker {worker_id} failed with action '{result['action']}': {result['details']}")
            
        return result
    
    def _quarantine_worker(self, worker_id: str):
        """Quarantine a worker that has failed multiple recovery attempts.
        
        Args:
            worker_id: ID of the worker to quarantine
        """
        logger.warning(f"Quarantining worker {worker_id} due to failed recovery attempts")
        
        # Mark as unavailable in worker manager
        if self.worker_manager and hasattr(self.worker_manager, "update_worker_status"):
            self.worker_manager.update_worker_status(worker_id, WORKER_STATUS_UNAVAILABLE)
            
        # Add to database blacklist if available
        if self.db_manager and hasattr(self.db_manager, "quarantine_worker"):
            probation_end = datetime.now() + timedelta(seconds=self.config["worker_probation_period"])
            self.db_manager.quarantine_worker(worker_id, probation_end)
            
        # Generate alert
        self._generate_alert(
            worker_id, 
            "quarantined",
            [f"Worker quarantined after {self.config['worker_quarantine_threshold']} failed recovery attempts"]
        )
    
    def _generate_alert(self, worker_id: str, status: str, issues: List[str]):
        """Generate an alert for an issue with a worker.
        
        Args:
            worker_id: ID of the affected worker
            status: Status of the worker
            issues: List of issues affecting the worker
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "worker_id": worker_id,
            "status": status,
            "issues": issues,
            "id": f"alert_{uuid.uuid4().hex}"
        }
        
        # Log alert
        logger.warning(f"ALERT: Worker {worker_id} is {status}: {'; '.join(issues)}")
        
        # Send to subscribers
        for subscriber in self.alert_subscribers:
            try:
                subscriber(alert)
            except Exception as e:
                logger.error(f"Error sending alert to subscriber: {e}")
        
        # Store in database if available
        if self.db_manager and hasattr(self.db_manager, "store_alert"):
            self.db_manager.store_alert(alert)
    
    def get_worker_health(self, worker_id: str = None) -> Dict[str, Any]:
        """Get the health status of a worker or all workers.
        
        Args:
            worker_id: Optional worker ID to get health for (all workers if None)
            
        Returns:
            Dict containing health status information
        """
        if worker_id:
            return self.worker_health.get(worker_id, {})
        else:
            return self.worker_health
    
    def get_worker_anomalies(self, worker_id: str = None, 
                            limit: int = 10) -> Dict[str, Any]:
        """Get performance anomalies for a worker or all workers.
        
        Args:
            worker_id: Optional worker ID to get anomalies for (all workers if None)
            limit: Maximum number of anomalies to return per worker
            
        Returns:
            Dict containing anomaly information
        """
        if worker_id:
            return {
                worker_id: self.anomaly_history.get(worker_id, [])[-limit:]
            }
        else:
            return {
                w_id: anomalies[-limit:] 
                for w_id, anomalies in self.anomaly_history.items()
            }
    
    def get_recovery_history(self, worker_id: str = None, 
                           limit: int = 10) -> Dict[str, Any]:
        """Get recovery attempt history for a worker or all workers.
        
        Args:
            worker_id: Optional worker ID to get history for (all workers if None)
            limit: Maximum number of attempts to return per worker
            
        Returns:
            Dict containing recovery attempt history
        """
        if worker_id:
            return {
                worker_id: self.recovery_attempts.get(worker_id, [])[-limit:]
            }
        else:
            return {
                w_id: attempts[-limit:] 
                for w_id, attempts in self.recovery_attempts.items()
            }
    
    def subscribe_to_alerts(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to alerts generated by the health monitor.
        
        Args:
            callback: Function to call with alert information
        """
        self.alert_subscribers.append(callback)
        logger.info(f"Added alert subscriber, total subscribers: {len(self.alert_subscribers)}")
    
    def unsubscribe_from_alerts(self, callback: Callable[[Dict[str, Any]], None]):
        """Unsubscribe from alerts.
        
        Args:
            callback: Function to remove from subscribers
        """
        if callback in self.alert_subscribers:
            self.alert_subscribers.remove(callback)
            logger.info(f"Removed alert subscriber, remaining subscribers: {len(self.alert_subscribers)}")
    
    def reset_worker_health(self, worker_id: str):
        """Reset health tracking for a worker.
        
        Args:
            worker_id: ID of the worker
        """
        if worker_id in self.worker_health:
            del self.worker_health[worker_id]
            
        if worker_id in self.health_history:
            del self.health_history[worker_id]
            
        if worker_id in self.anomaly_history:
            del self.anomaly_history[worker_id]
            
        if worker_id in self.recovery_attempts:
            del self.recovery_attempts[worker_id]
            
        if worker_id in self.recovery_success:
            del self.recovery_success[worker_id]
            
        logger.info(f"Reset health tracking for worker {worker_id}")
            
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate a comprehensive health report for all workers.
        
        Returns:
            Dict containing the health report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "workers": {},
            "system": {
                "total_workers": len(self.worker_health),
                "healthy_workers": sum(1 for status in self.worker_health.values() 
                                     if status["status"] == HEALTH_STATUS_HEALTHY),
                "warning_workers": sum(1 for status in self.worker_health.values() 
                                     if status["status"] == HEALTH_STATUS_WARNING),
                "critical_workers": sum(1 for status in self.worker_health.values() 
                                      if status["status"] == HEALTH_STATUS_CRITICAL),
                "quarantined_workers": 0  # Will be updated if db_manager available
            },
            "recovery": {
                "total_attempts": sum(len(attempts) for attempts in self.recovery_attempts.values()),
                "successful_attempts": sum(self.recovery_success.get(w_id, 0) for w_id in self.recovery_attempts),
                "quarantine_threshold": self.config["worker_quarantine_threshold"]
            }
        }
        
        # Get quarantined workers count from db if available
        if self.db_manager and hasattr(self.db_manager, "get_quarantined_workers_count"):
            report["system"]["quarantined_workers"] = self.db_manager.get_quarantined_workers_count()
        
        # Add worker details
        for worker_id, health_status in self.worker_health.items():
            worker_report = {
                "status": health_status["status"],
                "issues": health_status["issues"],
                "last_update": health_status["last_update"].isoformat(),
                "consecutive_healthy": health_status["consecutive_healthy"],
                "consecutive_unhealthy": health_status["consecutive_unhealthy"],
                "recovery_attempts": len(self.recovery_attempts.get(worker_id, [])),
                "successful_recoveries": self.recovery_success.get(worker_id, 0),
                "recent_anomalies": len(self.anomaly_history.get(worker_id, [])),
                "metrics": health_status.get("metrics", {})
            }
            
            report["workers"][worker_id] = worker_report
            
        return report