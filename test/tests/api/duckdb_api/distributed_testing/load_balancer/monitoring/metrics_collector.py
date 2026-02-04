#!/usr/bin/env python3
"""
Metrics Collector for Load Balancer Monitoring Dashboard

This module collects metrics from the load balancer and coordinator
for visualization in the monitoring dashboard.
"""

import os
import sys
import time
import json
import logging
import threading
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from pathlib import Path

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("metrics_collector")

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDB not available. Metrics will not be stored in database.")
    DUCKDB_AVAILABLE = False

class MetricType(Enum):
    """Types of metrics that can be collected."""
    SYSTEM = "system"
    WORKER = "worker"
    TASK = "task"

class MetricsCollector:
    """
    Collects and stores metrics from the distributed testing system.
    
    This class gathers metrics from the load balancer and coordinator,
    processes them, and stores them in a time-series database for
    visualization in the monitoring dashboard.
    """
    
    def __init__(self, 
                 db_path: str = "metrics.duckdb",
                 collection_interval: float = 1.0,
                 retention_days: int = 7):
        """
        Initialize the metrics collector.
        
        Args:
            db_path: Path to the DuckDB database
            collection_interval: Interval in seconds between metric collections
            retention_days: Number of days to retain historical metrics
        """
        self.db_path = db_path
        self.collection_interval = collection_interval
        self.retention_days = retention_days
        
        # Initialize metrics sources
        self._load_balancer = None
        self._coordinator = None
        
        # Initialize database connection
        self.conn = None
        if DUCKDB_AVAILABLE:
            try:
                self.conn = duckdb.connect(self.db_path)
                self._create_schema()
                logger.info(f"Connected to metrics database: {self.db_path}")
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                self.conn = None
        
        # Initialize collection thread
        self._stop_collection = threading.Event()
        self.collection_thread = None
        
        # Metrics cache
        self.system_metrics_cache = {}
        self.worker_metrics_cache = {}
        self.task_metrics_cache = {}
        
        # Callback registry for real-time updates
        self._callbacks = {
            MetricType.SYSTEM: [],
            MetricType.WORKER: [],
            MetricType.TASK: []
        }
        
        # Last collection timestamp
        self.last_collection = None
    
    def set_sources(self, load_balancer=None, coordinator=None):
        """
        Set the metrics sources.
        
        Args:
            load_balancer: Load balancer instance
            coordinator: Coordinator instance
        """
        self._load_balancer = load_balancer
        self._coordinator = coordinator
        logger.info("Metrics sources configured")
    
    def _create_schema(self):
        """Create the database schema for storing metrics."""
        if not self.conn:
            return
        
        try:
            # Create system metrics table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                timestamp TIMESTAMP,
                metric_name VARCHAR,
                metric_value FLOAT,
                PRIMARY KEY (timestamp, metric_name)
            )
            """)
            
            # Create worker metrics table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS worker_metrics (
                timestamp TIMESTAMP,
                worker_id VARCHAR,
                metric_name VARCHAR,
                metric_value FLOAT,
                PRIMARY KEY (timestamp, worker_id, metric_name)
            )
            """)
            
            # Create task metrics table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS task_metrics (
                timestamp TIMESTAMP,
                task_id VARCHAR,
                metric_name VARCHAR,
                metric_value FLOAT,
                PRIMARY KEY (timestamp, task_id, metric_name)
            )
            """)
            
            # Create task metadata table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS task_metadata (
                task_id VARCHAR PRIMARY KEY,
                model_id VARCHAR,
                model_family VARCHAR,
                test_type VARCHAR,
                priority INTEGER,
                creation_time TIMESTAMP
            )
            """)
            
            # Create worker metadata table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS worker_metadata (
                worker_id VARCHAR PRIMARY KEY,
                hostname VARCHAR,
                worker_type VARCHAR,
                capabilities JSON,
                registration_time TIMESTAMP
            )
            """)
            
            logger.info("Database schema created")
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
    
    def start(self):
        """Start the metrics collection process."""
        if self.collection_thread and self.collection_thread.is_alive():
            logger.warning("Metrics collection already running")
            return
        
        self._stop_collection.clear()
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self.collection_thread.start()
        
        logger.info(f"Metrics collection started with interval {self.collection_interval}s")
    
    def stop(self):
        """Stop the metrics collection process."""
        if not self.collection_thread or not self.collection_thread.is_alive():
            logger.warning("Metrics collection not running")
            return
        
        self._stop_collection.set()
        self.collection_thread.join(timeout=5)
        
        if self.conn:
            self.conn.close()
            self.conn = None
        
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Background loop for collecting metrics at regular intervals."""
        while not self._stop_collection.is_set():
            try:
                # Collect metrics
                self._collect_metrics()
                
                # Store metrics
                self._store_metrics()
                
                # Clean up old metrics
                self._cleanup_old_metrics()
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
            
            # Wait for next collection interval
            self._stop_collection.wait(self.collection_interval)
    
    def _collect_metrics(self):
        """Collect metrics from sources and process them."""
        timestamp = datetime.datetime.now()
        self.last_collection = timestamp
        
        # Collect system metrics
        system_metrics = self._collect_system_metrics()
        if system_metrics:
            self.system_metrics_cache = {
                'timestamp': timestamp,
                'metrics': system_metrics
            }
            self._notify_callbacks(MetricType.SYSTEM, timestamp, system_metrics)
        
        # Collect worker metrics
        worker_metrics = self._collect_worker_metrics()
        if worker_metrics:
            self.worker_metrics_cache = {
                'timestamp': timestamp,
                'metrics': worker_metrics
            }
            self._notify_callbacks(MetricType.WORKER, timestamp, worker_metrics)
        
        # Collect task metrics
        task_metrics = self._collect_task_metrics()
        if task_metrics:
            self.task_metrics_cache = {
                'timestamp': timestamp,
                'metrics': task_metrics
            }
            self._notify_callbacks(MetricType.TASK, timestamp, task_metrics)
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """
        Collect system-level metrics.
        
        Returns:
            Dictionary of metric name to value
        """
        metrics = {}
        
        # Collect from load balancer if available
        if self._load_balancer:
            lb = self._load_balancer
            
            # Worker metrics
            total_workers = 0
            active_workers = 0
            
            if hasattr(lb, 'workers'):
                total_workers = len(lb.workers)
                active_workers = sum(1 for w in lb.workers.values() if w.get('status') == 'active')
            
            metrics['total_workers'] = float(total_workers)
            metrics['active_workers'] = float(active_workers)
            metrics['inactive_workers'] = float(total_workers - active_workers)
            metrics['worker_utilization'] = float(active_workers / max(1, total_workers)) * 100
            
            # Task metrics
            if hasattr(lb, 'task_queue'):
                metrics['queued_tasks'] = float(len(lb.task_queue))
            else:
                metrics['queued_tasks'] = 0.0
                
            if hasattr(lb, 'running_tasks'):
                metrics['running_tasks'] = float(len(lb.running_tasks))
            else:
                metrics['running_tasks'] = 0.0
            
            # Calculate throughput (tasks per minute)
            if hasattr(lb, '_task_completion_times'):
                # Count tasks completed in the last minute
                now = time.time()
                minute_ago = now - 60
                recent_completions = [t for t in lb._task_completion_times if t > minute_ago]
                metrics['throughput'] = float(len(recent_completions))
            else:
                metrics['throughput'] = 0.0
            
            # Calculate average queue time
            if hasattr(lb, '_task_queue_times'):
                if lb._task_queue_times:
                    avg_queue_time = sum(lb._task_queue_times) / len(lb._task_queue_times)
                    metrics['avg_queue_time'] = float(avg_queue_time)
                else:
                    metrics['avg_queue_time'] = 0.0
            else:
                metrics['avg_queue_time'] = 0.0
        
        # Collect from coordinator if available
        if self._coordinator:
            coordinator = self._coordinator
            
            # Task status counts
            task_statuses = {}
            for task in coordinator.tasks.values():
                status = task.get('status', 'unknown')
                task_statuses[status] = task_statuses.get(status, 0) + 1
            
            for status, count in task_statuses.items():
                metrics[f'tasks_{status}'] = float(count)
            
            # Calculate total completed tasks
            metrics['completed_tasks'] = float(task_statuses.get('completed', 0))
            metrics['failed_tasks'] = float(task_statuses.get('failed', 0))
            
            # Calculate success rate
            total_finished = task_statuses.get('completed', 0) + task_statuses.get('failed', 0)
            if total_finished > 0:
                success_rate = task_statuses.get('completed', 0) / total_finished
                metrics['success_rate'] = float(success_rate) * 100
            else:
                metrics['success_rate'] = 100.0
        
        return metrics
    
    def _collect_worker_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Collect worker-level metrics.
        
        Returns:
            Dictionary of worker ID to metric dictionary
        """
        metrics = {}
        
        # Collect from coordinator if available
        if self._coordinator:
            coordinator = self._coordinator
            
            for worker_id, worker in coordinator.workers.items():
                worker_metrics = {}
                
                # Basic metrics
                worker_metrics['status'] = 1.0 if worker.get('status') == 'active' else 0.0
                
                # Load metrics
                metrics_data = worker.get('metrics', {})
                worker_metrics['cpu_utilization'] = float(metrics_data.get('cpu_utilization', 0.0))
                worker_metrics['memory_utilization'] = float(metrics_data.get('memory_utilization', 0.0))
                worker_metrics['gpu_utilization'] = float(metrics_data.get('gpu_utilization', 0.0))
                worker_metrics['disk_utilization'] = float(metrics_data.get('disk_utilization', 0.0))
                
                # Task metrics
                tasks = worker.get('tasks', [])
                worker_metrics['active_tasks'] = float(len(tasks))
                
                # Save worker metadata if needed
                self._store_worker_metadata(worker_id, worker)
                
                # Store in metrics dict
                metrics[worker_id] = worker_metrics
        
        # Collect from load balancer if available
        if self._load_balancer:
            lb = self._load_balancer
            
            # Get per-worker load information
            if hasattr(lb, 'worker_loads'):
                for worker_id, load in lb.worker_loads.items():
                    if worker_id not in metrics:
                        metrics[worker_id] = {}
                    
                    # Add load metrics
                    if hasattr(load, 'cpu_utilization'):
                        metrics[worker_id]['cpu_utilization'] = float(load.cpu_utilization)
                    if hasattr(load, 'memory_utilization'):
                        metrics[worker_id]['memory_utilization'] = float(load.memory_utilization)
                    if hasattr(load, 'gpu_utilization'):
                        metrics[worker_id]['gpu_utilization'] = float(load.gpu_utilization)
                    if hasattr(load, 'active_tests'):
                        metrics[worker_id]['active_tasks'] = float(load.active_tests)
                    if hasattr(load, 'queue_depth'):
                        metrics[worker_id]['queue_depth'] = float(load.queue_depth)
            
            # Get worker assignment efficiency
            if hasattr(lb, '_worker_assignment_stats'):
                for worker_id, stats in lb._worker_assignment_stats.items():
                    if worker_id not in metrics:
                        metrics[worker_id] = {}
                    
                    # Calculate assignment efficiency
                    total_assignments = stats.get('total_assignments', 0)
                    optimal_assignments = stats.get('optimal_assignments', 0)
                    
                    if total_assignments > 0:
                        efficiency = optimal_assignments / total_assignments
                        metrics[worker_id]['assignment_efficiency'] = float(efficiency) * 100
                    else:
                        metrics[worker_id]['assignment_efficiency'] = 100.0
        
        return metrics
    
    def _collect_task_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Collect task-level metrics.
        
        Returns:
            Dictionary of task ID to metric dictionary
        """
        metrics = {}
        
        # Collect from coordinator if available
        if self._coordinator:
            coordinator = self._coordinator
            
            for task_id, task in coordinator.tasks.items():
                task_metrics = {}
                
                # Basic metrics
                status = task.get('status', 'unknown')
                task_metrics['status_queued'] = 1.0 if status == 'queued' else 0.0
                task_metrics['status_assigned'] = 1.0 if status == 'assigned' else 0.0
                task_metrics['status_running'] = 1.0 if status == 'running' else 0.0
                task_metrics['status_completed'] = 1.0 if status == 'completed' else 0.0
                task_metrics['status_failed'] = 1.0 if status == 'failed' else 0.0
                
                # Timing metrics
                create_time = task.get('create_time')
                start_time = task.get('start_time')
                end_time = task.get('end_time')
                
                if create_time and start_time:
                    queue_time = (start_time - create_time).total_seconds()
                    task_metrics['queue_time'] = float(queue_time)
                
                if start_time and end_time:
                    processing_time = (end_time - start_time).total_seconds()
                    task_metrics['processing_time'] = float(processing_time)
                
                if create_time and end_time:
                    total_time = (end_time - create_time).total_seconds()
                    task_metrics['total_time'] = float(total_time)
                
                # Priority
                if 'priority' in task:
                    task_metrics['priority'] = float(task['priority'])
                
                # Result metrics
                result = task.get('result', {})
                if isinstance(result, dict):
                    task_metrics['success'] = 1.0 if result.get('success', False) else 0.0
                    
                    # Extract numeric metrics from result
                    metrics_data = result.get('metrics', {})
                    if isinstance(metrics_data, dict):
                        for key, value in metrics_data.items():
                            if isinstance(value, (int, float)):
                                task_metrics[f'result_{key}'] = float(value)
                
                # Save task metadata
                self._store_task_metadata(task_id, task)
                
                # Store in metrics dict
                metrics[task_id] = task_metrics
        
        # Collect from load balancer if available
        if self._load_balancer:
            lb = self._load_balancer
            
            # Get task assignment information
            if hasattr(lb, 'assignments'):
                for task_id, assignment in lb.assignments.items():
                    if task_id not in metrics:
                        metrics[task_id] = {}
                    
                    # Add assignment metrics
                    if hasattr(assignment, 'worker_id'):
                        metrics[task_id]['has_worker'] = 1.0
                    else:
                        metrics[task_id]['has_worker'] = 0.0
                    
                    # Add timing metrics
                    if hasattr(assignment, 'assigned_at') and assignment.assigned_at:
                        metrics[task_id]['has_assignment_time'] = 1.0
                    else:
                        metrics[task_id]['has_assignment_time'] = 0.0
                    
                    if hasattr(assignment, 'started_at') and assignment.started_at:
                        metrics[task_id]['has_start_time'] = 1.0
                    else:
                        metrics[task_id]['has_start_time'] = 0.0
                    
                    if hasattr(assignment, 'completed_at') and assignment.completed_at:
                        metrics[task_id]['has_completion_time'] = 1.0
                    else:
                        metrics[task_id]['has_completion_time'] = 0.0
                    
                    if hasattr(assignment, 'execution_time'):
                        metrics[task_id]['execution_time'] = float(assignment.execution_time)
        
        return metrics
    
    def _store_worker_metadata(self, worker_id: str, worker: Dict[str, Any]):
        """
        Store worker metadata in the database.
        
        Args:
            worker_id: Worker ID
            worker: Worker data dictionary
        """
        if not self.conn:
            return
        
        try:
            # Extract metadata
            hostname = worker.get('hostname', f"host-{worker_id}")
            
            # Determine worker type
            capabilities = worker.get('capabilities', {})
            hardware = capabilities.get('hardware', {})
            
            worker_type = "cpu"
            if hardware.get('gpu', {}).get('available', False):
                worker_type = "gpu"
            elif hardware.get('tpu', {}).get('available', False):
                worker_type = "tpu"
            
            # Store capabilities as JSON
            capabilities_json = json.dumps(capabilities)
            
            # Registration time
            registration_time = worker.get('registration_time', datetime.datetime.now())
            
            # Check if metadata already exists
            result = self.conn.execute("""
            SELECT worker_id FROM worker_metadata WHERE worker_id = ?
            """, [worker_id]).fetchone()
            
            if result:
                # Update existing metadata
                self.conn.execute("""
                UPDATE worker_metadata
                SET hostname = ?, worker_type = ?, capabilities = ?, registration_time = ?
                WHERE worker_id = ?
                """, [hostname, worker_type, capabilities_json, registration_time, worker_id])
            else:
                # Insert new metadata
                self.conn.execute("""
                INSERT INTO worker_metadata (worker_id, hostname, worker_type, capabilities, registration_time)
                VALUES (?, ?, ?, ?, ?)
                """, [worker_id, hostname, worker_type, capabilities_json, registration_time])
        except Exception as e:
            logger.error(f"Error storing worker metadata: {e}")
    
    def _store_task_metadata(self, task_id: str, task: Dict[str, Any]):
        """
        Store task metadata in the database.
        
        Args:
            task_id: Task ID
            task: Task data dictionary
        """
        if not self.conn:
            return
        
        try:
            # Extract metadata
            config = task.get('config', {})
            model = config.get('model', {})
            
            model_id = model.get('model_id', '')
            model_family = model.get('model_family', '')
            test_type = config.get('test_type', '')
            priority = config.get('priority', 3)
            
            # Creation time
            creation_time = task.get('create_time', datetime.datetime.now())
            
            # Check if metadata already exists
            result = self.conn.execute("""
            SELECT task_id FROM task_metadata WHERE task_id = ?
            """, [task_id]).fetchone()
            
            if result:
                # Update existing metadata
                self.conn.execute("""
                UPDATE task_metadata
                SET model_id = ?, model_family = ?, test_type = ?, priority = ?, creation_time = ?
                WHERE task_id = ?
                """, [model_id, model_family, test_type, priority, creation_time, task_id])
            else:
                # Insert new metadata
                self.conn.execute("""
                INSERT INTO task_metadata (task_id, model_id, model_family, test_type, priority, creation_time)
                VALUES (?, ?, ?, ?, ?, ?)
                """, [task_id, model_id, model_family, test_type, priority, creation_time])
        except Exception as e:
            logger.error(f"Error storing task metadata: {e}")
    
    def _store_metrics(self):
        """Store collected metrics in the database."""
        if not self.conn:
            return
        
        try:
            # Store system metrics
            if self.system_metrics_cache:
                timestamp = self.system_metrics_cache['timestamp']
                metrics = self.system_metrics_cache['metrics']
                
                for metric_name, metric_value in metrics.items():
                    self.conn.execute("""
                    INSERT INTO system_metrics (timestamp, metric_name, metric_value)
                    VALUES (?, ?, ?)
                    """, [timestamp, metric_name, metric_value])
            
            # Store worker metrics
            if self.worker_metrics_cache:
                timestamp = self.worker_metrics_cache['timestamp']
                workers_metrics = self.worker_metrics_cache['metrics']
                
                for worker_id, metrics in workers_metrics.items():
                    for metric_name, metric_value in metrics.items():
                        self.conn.execute("""
                        INSERT INTO worker_metrics (timestamp, worker_id, metric_name, metric_value)
                        VALUES (?, ?, ?, ?)
                        """, [timestamp, worker_id, metric_name, metric_value])
            
            # Store task metrics
            if self.task_metrics_cache:
                timestamp = self.task_metrics_cache['timestamp']
                tasks_metrics = self.task_metrics_cache['metrics']
                
                for task_id, metrics in tasks_metrics.items():
                    for metric_name, metric_value in metrics.items():
                        self.conn.execute("""
                        INSERT INTO task_metrics (timestamp, task_id, metric_name, metric_value)
                        VALUES (?, ?, ?, ?)
                        """, [timestamp, task_id, metric_name, metric_value])
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than the retention period."""
        if not self.conn:
            return
        
        try:
            # Calculate retention threshold
            threshold = datetime.datetime.now() - datetime.timedelta(days=self.retention_days)
            
            # Delete old system metrics
            self.conn.execute("""
            DELETE FROM system_metrics WHERE timestamp < ?
            """, [threshold])
            
            # Delete old worker metrics
            self.conn.execute("""
            DELETE FROM worker_metrics WHERE timestamp < ?
            """, [threshold])
            
            # Delete old task metrics
            self.conn.execute("""
            DELETE FROM task_metrics WHERE timestamp < ?
            """, [threshold])
            
            # Delete old task metadata
            self.conn.execute("""
            DELETE FROM task_metadata 
            WHERE task_id NOT IN (
                SELECT DISTINCT task_id FROM task_metrics
            )
            """)
            
            # Delete old worker metadata
            self.conn.execute("""
            DELETE FROM worker_metadata 
            WHERE worker_id NOT IN (
                SELECT DISTINCT worker_id FROM worker_metrics
            )
            """)
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")
    
    def register_callback(self, metric_type: MetricType, callback: callable):
        """
        Register a callback for real-time metric updates.
        
        Args:
            metric_type: Type of metrics (system, worker, or task)
            callback: Function to call with metrics updates
        """
        if metric_type in self._callbacks:
            self._callbacks[metric_type].append(callback)
            logger.debug(f"Registered callback for {metric_type.value} metrics")
        else:
            logger.warning(f"Invalid metric type: {metric_type}")
    
    def unregister_callback(self, metric_type: MetricType, callback: callable):
        """
        Unregister a callback for real-time metric updates.
        
        Args:
            metric_type: Type of metrics (system, worker, or task)
            callback: Function to unregister
        """
        if metric_type in self._callbacks and callback in self._callbacks[metric_type]:
            self._callbacks[metric_type].remove(callback)
            logger.debug(f"Unregistered callback for {metric_type.value} metrics")
    
    def _notify_callbacks(self, metric_type: MetricType, timestamp, metrics):
        """
        Notify callbacks about metric updates.
        
        Args:
            metric_type: Type of metrics (system, worker, or task)
            timestamp: Timestamp of the metrics
            metrics: Metrics data
        """
        if metric_type in self._callbacks:
            for callback in self._callbacks[metric_type]:
                try:
                    callback(timestamp, metrics)
                except Exception as e:
                    logger.error(f"Error in metrics callback: {e}")
    
    def get_current_system_metrics(self) -> Dict[str, float]:
        """
        Get the most recent system metrics.
        
        Returns:
            Dictionary of metric name to value
        """
        if self.system_metrics_cache:
            return self.system_metrics_cache['metrics']
        return {}
    
    def get_current_worker_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get the most recent worker metrics.
        
        Returns:
            Dictionary of worker ID to metrics dictionary
        """
        if self.worker_metrics_cache:
            return self.worker_metrics_cache['metrics']
        return {}
    
    def get_current_task_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get the most recent task metrics.
        
        Returns:
            Dictionary of task ID to metrics dictionary
        """
        if self.task_metrics_cache:
            return self.task_metrics_cache['metrics']
        return {}
    
    def get_worker_metadata(self, worker_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get worker metadata.
        
        Args:
            worker_id: Specific worker ID or None for all workers
            
        Returns:
            Dictionary of worker ID to metadata dictionary
        """
        if not self.conn:
            return {}
        
        try:
            if worker_id:
                # Get specific worker
                result = self.conn.execute("""
                SELECT worker_id, hostname, worker_type, capabilities, registration_time
                FROM worker_metadata
                WHERE worker_id = ?
                """, [worker_id]).fetchone()
                
                if result:
                    return {
                        worker_id: {
                            'hostname': result[1],
                            'worker_type': result[2],
                            'capabilities': json.loads(result[3]),
                            'registration_time': result[4]
                        }
                    }
                return {}
            else:
                # Get all workers
                results = self.conn.execute("""
                SELECT worker_id, hostname, worker_type, capabilities, registration_time
                FROM worker_metadata
                """).fetchall()
                
                metadata = {}
                for row in results:
                    w_id, hostname, worker_type, capabilities, registration_time = row
                    metadata[w_id] = {
                        'hostname': hostname,
                        'worker_type': worker_type,
                        'capabilities': json.loads(capabilities),
                        'registration_time': registration_time
                    }
                
                return metadata
        except Exception as e:
            logger.error(f"Error getting worker metadata: {e}")
            return {}
    
    def get_task_metadata(self, task_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get task metadata.
        
        Args:
            task_id: Specific task ID or None for all tasks
            
        Returns:
            Dictionary of task ID to metadata dictionary
        """
        if not self.conn:
            return {}
        
        try:
            if task_id:
                # Get specific task
                result = self.conn.execute("""
                SELECT task_id, model_id, model_family, test_type, priority, creation_time
                FROM task_metadata
                WHERE task_id = ?
                """, [task_id]).fetchone()
                
                if result:
                    return {
                        task_id: {
                            'model_id': result[1],
                            'model_family': result[2],
                            'test_type': result[3],
                            'priority': result[4],
                            'creation_time': result[5]
                        }
                    }
                return {}
            else:
                # Get all tasks
                results = self.conn.execute("""
                SELECT task_id, model_id, model_family, test_type, priority, creation_time
                FROM task_metadata
                """).fetchall()
                
                metadata = {}
                for row in results:
                    t_id, model_id, model_family, test_type, priority, creation_time = row
                    metadata[t_id] = {
                        'model_id': model_id,
                        'model_family': model_family,
                        'test_type': test_type,
                        'priority': priority,
                        'creation_time': creation_time
                    }
                
                return metadata
        except Exception as e:
            logger.error(f"Error getting task metadata: {e}")
            return {}
    
    def get_historical_system_metrics(self, 
                                     metric_names: List[str], 
                                     start_time: Optional[datetime.datetime] = None,
                                     end_time: Optional[datetime.datetime] = None,
                                     interval: str = "1m") -> Dict[str, List[Tuple[datetime.datetime, float]]]:
        """
        Get historical system metrics.
        
        Args:
            metric_names: List of metric names to retrieve
            start_time: Start time for historical data (default: 1 hour ago)
            end_time: End time for historical data (default: now)
            interval: Aggregation interval (e.g., "1m", "5m", "1h")
            
        Returns:
            Dictionary of metric name to list of (timestamp, value) tuples
        """
        if not self.conn:
            return {}
        
        try:
            # Set default time range if not provided
            if not end_time:
                end_time = datetime.datetime.now()
            if not start_time:
                start_time = end_time - datetime.timedelta(hours=1)
            
            # Parse interval
            interval_seconds = self._parse_interval(interval)
            
            # Prepare result dictionary
            result = {metric: [] for metric in metric_names}
            
            # Prepare query parameters
            params = []
            metric_placeholders = []
            
            for metric in metric_names:
                metric_placeholders.append("?")
                params.append(metric)
            
            metric_list = ", ".join(metric_placeholders)
            params.extend([start_time, end_time])
            
            # Execute query
            query = f"""
            SELECT 
                metric_name, 
                time_bucket(INTERVAL '{interval_seconds} seconds', timestamp) as bucket,
                AVG(metric_value) as avg_value
            FROM system_metrics
            WHERE metric_name IN ({metric_list})
            AND timestamp >= ?
            AND timestamp <= ?
            GROUP BY metric_name, bucket
            ORDER BY metric_name, bucket
            """
            
            results = self.conn.execute(query, params).fetchall()
            
            # Process results
            for row in results:
                metric_name, bucket, avg_value = row
                if metric_name in result:
                    result[metric_name].append((bucket, avg_value))
            
            return result
        except Exception as e:
            logger.error(f"Error getting historical system metrics: {e}")
            return {}
    
    def get_historical_worker_metrics(self,
                                     worker_id: str,
                                     metric_names: List[str],
                                     start_time: Optional[datetime.datetime] = None,
                                     end_time: Optional[datetime.datetime] = None,
                                     interval: str = "1m") -> Dict[str, List[Tuple[datetime.datetime, float]]]:
        """
        Get historical worker metrics.
        
        Args:
            worker_id: Worker ID
            metric_names: List of metric names to retrieve
            start_time: Start time for historical data (default: 1 hour ago)
            end_time: End time for historical data (default: now)
            interval: Aggregation interval (e.g., "1m", "5m", "1h")
            
        Returns:
            Dictionary of metric name to list of (timestamp, value) tuples
        """
        if not self.conn:
            return {}
        
        try:
            # Set default time range if not provided
            if not end_time:
                end_time = datetime.datetime.now()
            if not start_time:
                start_time = end_time - datetime.timedelta(hours=1)
            
            # Parse interval
            interval_seconds = self._parse_interval(interval)
            
            # Prepare result dictionary
            result = {metric: [] for metric in metric_names}
            
            # Prepare query parameters
            params = [worker_id]
            metric_placeholders = []
            
            for metric in metric_names:
                metric_placeholders.append("?")
                params.append(metric)
            
            metric_list = ", ".join(metric_placeholders)
            params.extend([start_time, end_time])
            
            # Execute query
            query = f"""
            SELECT 
                metric_name, 
                time_bucket(INTERVAL '{interval_seconds} seconds', timestamp) as bucket,
                AVG(metric_value) as avg_value
            FROM worker_metrics
            WHERE worker_id = ?
            AND metric_name IN ({metric_list})
            AND timestamp >= ?
            AND timestamp <= ?
            GROUP BY metric_name, bucket
            ORDER BY metric_name, bucket
            """
            
            results = self.conn.execute(query, params).fetchall()
            
            # Process results
            for row in results:
                metric_name, bucket, avg_value = row
                if metric_name in result:
                    result[metric_name].append((bucket, avg_value))
            
            return result
        except Exception as e:
            logger.error(f"Error getting historical worker metrics: {e}")
            return {}
    
    def _parse_interval(self, interval: str) -> int:
        """
        Parse time interval string to seconds.
        
        Args:
            interval: Interval string (e.g., "1m", "5m", "1h")
            
        Returns:
            Interval in seconds
        """
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit == 's':
            return value
        elif unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        elif unit == 'd':
            return value * 86400
        else:
            # Default to seconds
            return value
    
    def get_worker_performance_score(self, worker_id: str) -> float:
        """
        Calculate a performance score for a worker based on historical metrics.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            Performance score (0-100)
        """
        if not self.conn:
            return 50.0  # Default middle score
        
        try:
            # Get recent task completions
            results = self.conn.execute("""
            SELECT AVG(metric_value) as avg_execution_time, COUNT(*) as task_count
            FROM task_metrics
            WHERE task_id IN (
                SELECT DISTINCT task_id FROM task_metrics
                WHERE metric_name = 'status_completed'
                AND metric_value = 1.0
            )
            AND task_id IN (
                SELECT t.task_id
                FROM task_metrics t
                JOIN task_metrics w ON t.task_id = w.task_id
                WHERE t.metric_name = 'worker_id'
                AND t.metric_value = ?
                AND w.metric_name = 'processing_time'
            )
            AND metric_name = 'processing_time'
            AND timestamp > ?
            """, [worker_id, datetime.datetime.now() - datetime.timedelta(hours=24)]).fetchone()
            
            if results and results[1] > 0:
                avg_execution_time = results[0]
                task_count = results[1]
                
                # Get system-wide average
                system_avg = self.conn.execute("""
                SELECT AVG(metric_value) as system_avg
                FROM task_metrics
                WHERE task_id IN (
                    SELECT DISTINCT task_id FROM task_metrics
                    WHERE metric_name = 'status_completed'
                    AND metric_value = 1.0
                )
                AND metric_name = 'processing_time'
                AND timestamp > ?
                """, [datetime.datetime.now() - datetime.timedelta(hours=24)]).fetchone()
                
                if system_avg and system_avg[0]:
                    system_avg_time = system_avg[0]
                    
                    # Calculate performance score (lower execution time is better)
                    # Also consider number of tasks completed (more is better)
                    time_score = min(100, max(0, 100 * (system_avg_time / max(0.1, avg_execution_time))))
                    
                    # Get average tasks completed per worker
                    avg_tasks = self.conn.execute("""
                    SELECT AVG(task_count) as avg_tasks
                    FROM (
                        SELECT COUNT(*) as task_count
                        FROM task_metrics
                        WHERE task_id IN (
                            SELECT DISTINCT task_id FROM task_metrics
                            WHERE metric_name = 'status_completed'
                            AND metric_value = 1.0
                        )
                        AND task_id IN (
                            SELECT t.task_id
                            FROM task_metrics t
                            JOIN task_metrics w ON t.task_id = w.task_id
                            WHERE t.metric_name = 'worker_id'
                            AND w.metric_name = 'processing_time'
                        )
                        GROUP BY metric_value
                    )
                    """).fetchone()
                    
                    if avg_tasks and avg_tasks[0]:
                        avg_tasks_per_worker = avg_tasks[0]
                        task_count_score = min(100, max(0, 100 * (task_count / max(1, avg_tasks_per_worker))))
                    else:
                        task_count_score = 50.0
                    
                    # Combine scores (70% time efficiency, 30% task count)
                    return 0.7 * time_score + 0.3 * task_count_score
            
            # If no data available, return middle score
            return 50.0
        except Exception as e:
            logger.error(f"Error calculating worker performance score: {e}")
            return 50.0
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the metrics data.
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if not self.conn:
            return anomalies
        
        try:
            # Check for overloaded workers
            worker_metrics = self.get_current_worker_metrics()
            
            for worker_id, metrics in worker_metrics.items():
                # Check CPU utilization
                if metrics.get('cpu_utilization', 0) > 90:
                    anomalies.append({
                        'type': 'worker_overload',
                        'worker_id': worker_id,
                        'metric': 'cpu_utilization',
                        'value': metrics['cpu_utilization'],
                        'threshold': 90,
                        'severity': 'high',
                        'message': f"Worker {worker_id} CPU utilization is {metrics['cpu_utilization']}%, above 90% threshold"
                    })
                
                # Check memory utilization
                if metrics.get('memory_utilization', 0) > 90:
                    anomalies.append({
                        'type': 'worker_overload',
                        'worker_id': worker_id,
                        'metric': 'memory_utilization',
                        'value': metrics['memory_utilization'],
                        'threshold': 90,
                        'severity': 'high',
                        'message': f"Worker {worker_id} memory utilization is {metrics['memory_utilization']}%, above 90% threshold"
                    })
                
                # Check GPU utilization
                if metrics.get('gpu_utilization', 0) > 95:
                    anomalies.append({
                        'type': 'worker_overload',
                        'worker_id': worker_id,
                        'metric': 'gpu_utilization',
                        'value': metrics['gpu_utilization'],
                        'threshold': 95,
                        'severity': 'medium',
                        'message': f"Worker {worker_id} GPU utilization is {metrics['gpu_utilization']}%, above 95% threshold"
                    })
                
                # Check queue depth
                if metrics.get('queue_depth', 0) > 10:
                    anomalies.append({
                        'type': 'worker_queue_depth',
                        'worker_id': worker_id,
                        'metric': 'queue_depth',
                        'value': metrics['queue_depth'],
                        'threshold': 10,
                        'severity': 'medium',
                        'message': f"Worker {worker_id} has {metrics['queue_depth']} tasks in queue, above 10 threshold"
                    })
            
            # Check system metrics
            system_metrics = self.get_current_system_metrics()
            
            # Check queue time
            if system_metrics.get('avg_queue_time', 0) > 30:
                anomalies.append({
                    'type': 'system_high_queue_time',
                    'metric': 'avg_queue_time',
                    'value': system_metrics['avg_queue_time'],
                    'threshold': 30,
                    'severity': 'high',
                    'message': f"Average queue time is {system_metrics['avg_queue_time']}s, above 30s threshold"
                })
            
            # Check throughput
            if system_metrics.get('throughput', 0) < 1 and system_metrics.get('queued_tasks', 0) > 5:
                anomalies.append({
                    'type': 'system_low_throughput',
                    'metric': 'throughput',
                    'value': system_metrics['throughput'],
                    'threshold': 1,
                    'severity': 'high',
                    'message': f"System throughput is {system_metrics['throughput']} tasks/minute with {system_metrics.get('queued_tasks', 0)} tasks queued"
                })
            
            # Check success rate
            if system_metrics.get('success_rate', 100) < 80:
                anomalies.append({
                    'type': 'system_low_success_rate',
                    'metric': 'success_rate',
                    'value': system_metrics['success_rate'],
                    'threshold': 80,
                    'severity': 'high',
                    'message': f"System success rate is {system_metrics['success_rate']}%, below 80% threshold"
                })
            
            # Check for stalled tasks
            # Get tasks that have been running for too long
            results = self.conn.execute("""
            SELECT t.task_id, tm.model_id, tm.model_family
            FROM task_metrics t
            JOIN task_metadata tm ON t.task_id = tm.task_id
            WHERE t.metric_name = 'status_running'
            AND t.metric_value = 1.0
            AND t.timestamp < ?
            AND t.task_id NOT IN (
                SELECT task_id FROM task_metrics
                WHERE metric_name IN ('status_completed', 'status_failed')
                AND metric_value = 1.0
            )
            """, [datetime.datetime.now() - datetime.timedelta(minutes=30)]).fetchall()
            
            for row in results:
                task_id, model_id, model_family = row
                anomalies.append({
                    'type': 'task_stalled',
                    'task_id': task_id,
                    'model_id': model_id,
                    'model_family': model_family,
                    'severity': 'high',
                    'message': f"Task {task_id} ({model_id}) has been running for more than 30 minutes"
                })
            
            return anomalies
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return anomalies


if __name__ == "__main__":
    # Example usage
    collector = MetricsCollector(db_path="metrics.duckdb")
    
    # Start metrics collection
    collector.start()
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop metrics collection
        collector.stop()