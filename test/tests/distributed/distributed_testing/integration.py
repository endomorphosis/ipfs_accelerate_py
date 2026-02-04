"""
Distributed Testing Framework Integration Module.

This module integrates all components of the Distributed Testing Framework:
1. ML-based anomaly detection
2. Prometheus/Grafana monitoring
3. Advanced scheduling algorithms
4. Dynamic resource management

It provides a unified API for managing the entire framework.
"""

import os
import logging
import threading
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union

# Import framework components
from .ml_anomaly_detection import MLAnomalyDetection
from .prometheus_grafana_integration import PrometheusGrafanaIntegration
from .advanced_scheduling import AdvancedScheduler, Task, Worker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dtf_integration")


class DistributedTestingFramework:
    """
    Main integration class for the Distributed Testing Framework.
    
    This class ties together all the components of the framework:
    - Scheduling and resource allocation
    - Monitoring and metrics collection
    - Anomaly detection and prediction
    - External system integration
    """
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        monitoring_config: Optional[Dict[str, Any]] = None,
        ml_config: Optional[Dict[str, Any]] = None,
        coordinator_id: Optional[str] = None,
        data_dir: str = "data",
    ):
        """
        Initialize the Distributed Testing Framework.
        
        Args:
            config_file: Path to configuration file
            scheduler_config: Configuration for the advanced scheduler
            monitoring_config: Configuration for Prometheus/Grafana integration
            ml_config: Configuration for ML anomaly detection
            coordinator_id: Unique identifier for this coordinator
            data_dir: Directory for data storage
        """
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Override with provided configs
        if scheduler_config:
            self.config["scheduler"] = {**self.config.get("scheduler", {}), **scheduler_config}
        if monitoring_config:
            self.config["monitoring"] = {**self.config.get("monitoring", {}), **monitoring_config}
        if ml_config:
            self.config["ml"] = {**self.config.get("ml", {}), **ml_config}
            
        # Initialize coordinator information
        self.coordinator_id = coordinator_id or self.config.get("coordinator_id", f"coordinator-{int(time.time())}")
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize scheduler
        self.scheduler = AdvancedScheduler(**self.config.get("scheduler", {}))
        
        # Initialize monitoring integration
        self.monitoring = PrometheusGrafanaIntegration(**self.config.get("monitoring", {}))
        
        # Shared state
        self.running = False
        self.threads = []
        self.last_metrics_update = 0
        self.metrics_interval = self.config.get("metrics_interval", 30)  # seconds
        
        # Initialize metrics storage
        self.metrics = {
            "tasks": {},
            "workers": {},
            "resources": {},
            "system": {},
        }
        
        logger.info(f"Initialized Distributed Testing Framework (Coordinator: {self.coordinator_id})")
        
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file if provided."""
        config = {
            # Default configuration
            "scheduler": {
                "algorithm": "adaptive",
                "fairness_window": 100,
                "resource_match_weight": 0.7,
                "user_fair_share_enabled": True,
                "adaptive_interval": 50,
            },
            "monitoring": {
                "prometheus_port": 8000,
                "metrics_collection_interval": 30,
                "anomaly_detection_interval": 300,
            },
            "ml": {
                "algorithms": ["isolation_forest", "dbscan", "threshold"],
                "forecasting": ["arima", "prophet"],
                "visualization": True,
            },
            "metrics_interval": 30,
            "scheduling_interval": 5,
        }
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Merge configurations
                self._deep_merge_configs(config, file_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
                
        return config
    
    def _deep_merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge configurations."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_configs(base[key], value)
            else:
                base[key] = value
    
    def start(self) -> bool:
        """
        Start the Distributed Testing Framework.
        
        Returns:
            True if started successfully
        """
        if self.running:
            logger.warning("Framework already running")
            return False
            
        self.running = True
        
        # Start monitoring integration
        self.monitoring.start()
        
        # Start background threads
        
        # Thread for scheduling tasks
        scheduling_thread = threading.Thread(
            target=self._scheduling_loop,
            daemon=True
        )
        scheduling_thread.start()
        self.threads.append(scheduling_thread)
        
        # Thread for collecting metrics
        metrics_thread = threading.Thread(
            target=self._metrics_loop,
            daemon=True
        )
        metrics_thread.start()
        self.threads.append(metrics_thread)
        
        logger.info(f"Started Distributed Testing Framework (Coordinator: {self.coordinator_id})")
        return True
    
    def stop(self) -> bool:
        """
        Stop the Distributed Testing Framework.
        
        Returns:
            True if stopped successfully
        """
        if not self.running:
            logger.warning("Framework not running")
            return False
            
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5)
            
        # Stop monitoring integration
        self.monitoring.stop()
        
        logger.info(f"Stopped Distributed Testing Framework (Coordinator: {self.coordinator_id})")
        return True
    
    def _scheduling_loop(self) -> None:
        """Background thread for task scheduling."""
        scheduling_interval = self.config.get("scheduling_interval", 5)  # seconds
        
        while self.running:
            try:
                # Schedule tasks
                assignments = self.scheduler.schedule_tasks()
                
                # Log assignments
                if assignments:
                    logger.info(f"Scheduled {len(assignments)} tasks")
                    
                    # In a real implementation, this would notify workers or
                    # update a database with the task assignments
                    
                    # For now, just update metrics
                    self._update_assignment_metrics(assignments)
                    
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                
            # Sleep until next scheduling cycle
            time.sleep(scheduling_interval)
    
    def _metrics_loop(self) -> None:
        """Background thread for collecting and updating metrics."""
        while self.running:
            try:
                # Collect metrics
                self._collect_metrics()
                
                # Update monitoring integration
                self.monitoring.update_metrics_from_data(self.metrics)
                
                # Check for anomalies
                self._check_anomalies()
                
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                
            # Sleep until next metrics cycle
            time.sleep(self.metrics_interval)
    
    def _collect_metrics(self) -> None:
        """Collect metrics from the framework."""
        # Get current time
        now = time.time()
        self.last_metrics_update = now
        
        # Collect task metrics
        task_stats = self.scheduler.get_task_queue_stats()
        worker_stats = self.scheduler.get_worker_stats()
        algorithm_stats = self.scheduler.get_algorithm_performance()
        
        # Update metrics
        self.metrics["tasks"] = task_stats
        self.metrics["workers"] = worker_stats
        self.metrics["scheduler"] = {
            "algorithm": self.scheduler.current_best_algorithm or self.scheduler.algorithm,
            "performance": algorithm_stats,
        }
        
        # Collect system metrics
        self.metrics["system"] = {
            "timestamp": now,
            "coordinator_id": self.coordinator_id,
            "uptime": time.time() - self.start_time if hasattr(self, "start_time") else 0,
        }
        
    def _update_assignment_metrics(self, assignments: List[Tuple[str, str]]) -> None:
        """Update metrics based on task assignments."""
        # Keep track of assignments by worker type, task type, etc.
        for task_id, worker_id in assignments:
            if task_id in self.scheduler.tasks and worker_id in self.scheduler.workers:
                task = self.scheduler.tasks[task_id]
                worker = self.scheduler.workers[worker_id]
                
                # Track the assignment
                if "assignments" not in self.metrics:
                    self.metrics["assignments"] = {
                        "by_worker_type": {},
                        "by_task_type": {},
                        "recent": [],
                    }
                    
                # By worker type
                worker_type = worker.worker_type
                if worker_type not in self.metrics["assignments"]["by_worker_type"]:
                    self.metrics["assignments"]["by_worker_type"][worker_type] = 0
                self.metrics["assignments"]["by_worker_type"][worker_type] += 1
                
                # By task type
                task_type = task.task_type
                if task_type not in self.metrics["assignments"]["by_task_type"]:
                    self.metrics["assignments"]["by_task_type"][task_type] = 0
                self.metrics["assignments"]["by_task_type"][task_type] += 1
                
                # Recent assignments (keep last 100)
                self.metrics["assignments"]["recent"].append({
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "task_type": task_type,
                    "worker_type": worker_type,
                    "timestamp": time.time(),
                })
                
                # Limit to last 100
                if len(self.metrics["assignments"]["recent"]) > 100:
                    self.metrics["assignments"]["recent"] = self.metrics["assignments"]["recent"][-100:]
    
    def _check_anomalies(self) -> None:
        """Check for anomalies in metrics."""
        # Get detected anomalies from monitoring
        anomalies = self.monitoring.get_detected_anomalies()
        
        # In a real implementation, this would take action based on anomalies,
        # such as adjusting scheduling parameters, alerting administrators, etc.
        
        # For now, just log significant anomalies
        for (metric_name, algorithm), anomaly_info in anomalies.items():
            severity = anomaly_info.get("severity", 0)
            if severity > 70:  # Only report high severity anomalies
                logger.warning(f"High severity anomaly detected in {metric_name} "
                             f"using {algorithm}: severity={severity}")
    
    # Task management methods
    
    def add_task(self, task_data: Dict[str, Any]) -> Optional[str]:
        """
        Add a task to the framework.
        
        Args:
            task_data: Dictionary containing task information
            
        Returns:
            Task ID if added successfully, None otherwise
        """
        try:
            # Create Task object from dictionary
            task = Task(
                task_id=task_data.get("task_id", f"task-{int(time.time() * 1000)}"),
                task_type=task_data.get("task_type", "test"),
                user_id=task_data.get("user_id", "default"),
                priority=task_data.get("priority", 0),
                estimated_duration=task_data.get("estimated_duration", 0.0),
                required_resources=task_data.get("required_resources", {}),
                dependencies=task_data.get("dependencies", []),
                metadata=task_data.get("metadata", {}),
                submission_time=task_data.get("submission_time", time.time()),
                deadline=task_data.get("deadline"),
            )
            
            # Add to scheduler
            success = self.scheduler.add_task(task)
            if success:
                return task.task_id
                
            return None
        except Exception as e:
            logger.error(f"Error adding task: {e}")
            return None
    
    def add_worker(self, worker_data: Dict[str, Any]) -> Optional[str]:
        """
        Add or update a worker in the framework.
        
        Args:
            worker_data: Dictionary containing worker information
            
        Returns:
            Worker ID if added successfully, None otherwise
        """
        try:
            # Create Worker object from dictionary
            worker = Worker(
                worker_id=worker_data.get("worker_id", f"worker-{int(time.time() * 1000)}"),
                worker_type=worker_data.get("worker_type", "default"),
                capabilities=worker_data.get("capabilities", {}),
                status=worker_data.get("status", "idle"),
                current_task=worker_data.get("current_task"),
                performance_metrics=worker_data.get("performance_metrics", {}),
                metadata=worker_data.get("metadata", {}),
            )
            
            # Add to scheduler
            success = self.scheduler.add_worker(worker)
            if success:
                return worker.worker_id
                
            return None
        except Exception as e:
            logger.error(f"Error adding worker: {e}")
            return None
    
    def update_worker_status(self, worker_id: str, status: str) -> bool:
        """
        Update a worker's status.
        
        Args:
            worker_id: Worker ID to update
            status: New status
            
        Returns:
            True if updated successfully
        """
        return self.scheduler.update_worker_status(worker_id, status)
    
    def complete_task(self, worker_id: str, success: bool, result: Any = None) -> Optional[str]:
        """
        Mark a task as completed by a worker.
        
        Args:
            worker_id: ID of worker that completed the task
            success: Whether the task was completed successfully
            result: Result data or error message
            
        Returns:
            Task ID that was completed, or None if not found
        """
        return self.scheduler.complete_task(worker_id, success, result)
    
    # Metrics and monitoring methods
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduling statistics."""
        return {
            "tasks": self.scheduler.get_task_queue_stats(),
            "workers": self.scheduler.get_worker_stats(),
            "algorithms": self.scheduler.get_algorithm_performance(),
        }
    
    def get_detected_anomalies(self) -> Dict[str, Any]:
        """Get detected anomalies."""
        return self.monitoring.get_detected_anomalies()
    
    def get_forecasts(self) -> Dict[str, Any]:
        """Get metric forecasts."""
        return self.monitoring.get_forecasts()
    
    # Health check and diagnostics
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the framework."""
        health = {
            "status": "healthy",
            "coordinator_id": self.coordinator_id,
            "uptime": time.time() - self.start_time if hasattr(self, "start_time") else 0,
            "components": {
                "scheduler": "running" if self.running else "stopped",
                "monitoring": "running" if self.monitoring.running else "stopped",
            },
            "metrics_age": time.time() - self.last_metrics_update,
            "task_counts": {
                "pending": len(self.scheduler.task_queue),
                "running": len(self.scheduler.running_tasks),
            },
            "worker_counts": {
                "total": len(self.scheduler.workers),
                "available": len(self.scheduler.available_workers),
            },
        }
        
        # Check if metrics are stale
        if health["metrics_age"] > self.metrics_interval * 3:
            health["status"] = "degraded"
            health["issues"] = ["Stale metrics"]
            
        return health
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific task.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            Task status dictionary or None if not found
        """
        if task_id in self.scheduler.tasks:
            return self.scheduler.tasks[task_id].to_dict()
        return None
    
    def get_worker_status(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific worker.
        
        Args:
            worker_id: Worker ID to check
            
        Returns:
            Worker status dictionary or None if not found
        """
        if worker_id in self.scheduler.workers:
            return self.scheduler.workers[worker_id].to_dict()
        return None


# Helper function to create and start the framework
def create_distributed_testing_framework(
    config_file: Optional[str] = None,
    coordinator_id: Optional[str] = None,
    data_dir: str = "data",
    scheduler_config: Optional[Dict[str, Any]] = None,
    monitoring_config: Optional[Dict[str, Any]] = None,
    ml_config: Optional[Dict[str, Any]] = None,
) -> DistributedTestingFramework:
    """
    Create and start a Distributed Testing Framework instance.
    
    Args:
        config_file: Path to configuration file
        coordinator_id: Unique identifier for this coordinator
        data_dir: Directory for data storage
        scheduler_config: Configuration for the advanced scheduler
        monitoring_config: Configuration for Prometheus/Grafana integration
        ml_config: Configuration for ML anomaly detection
        
    Returns:
        Running DistributedTestingFramework instance
    """
    # Create framework instance
    framework = DistributedTestingFramework(
        config_file=config_file,
        coordinator_id=coordinator_id,
        data_dir=data_dir,
        scheduler_config=scheduler_config,
        monitoring_config=monitoring_config,
        ml_config=ml_config,
    )
    
    # Start the framework
    framework.start()
    
    return framework


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Distributed Testing Framework")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--coordinator-id", help="Unique identifier for this coordinator")
    parser.add_argument("--data-dir", default="data", help="Directory for data storage")
    parser.add_argument("--prometheus-port", type=int, default=8000, help="Port for Prometheus metrics")
    
    args = parser.parse_args()
    
    # Create monitoring config from CLI args
    monitoring_config = {
        "prometheus_port": args.prometheus_port
    }
    
    # Create and start framework
    framework = create_distributed_testing_framework(
        config_file=args.config,
        coordinator_id=args.coordinator_id,
        data_dir=args.data_dir,
        monitoring_config=monitoring_config,
    )
    
    # Keep running until interrupted
    try:
        while framework.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping framework...")
        framework.stop()