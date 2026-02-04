#!/usr/bin/env python3
"""
Hardware Utilization Monitor

This module provides real-time hardware utilization monitoring for the Distributed Testing Framework.
It tracks CPU, memory, GPU, and other hardware resource usage during task execution and provides
this information to the coordinator for more informed scheduling decisions.

Features:
- Real-time monitoring of CPU, memory, GPU, and other hardware resources
- Historical tracking of hardware utilization
- Database integration for persistent storage of utilization metrics
- Threshold-based alerts for overutilization
- Integration with the coordinator for hardware-aware scheduling

Usage:
    monitor = HardwareUtilizationMonitor(db_path="./hardware_db.duckdb")
    monitor.start_monitoring()
    # ... execute tasks ...
    metrics = monitor.get_current_metrics()
    monitor.stop_monitoring()
"""

import os
import sys
import time
import json
import uuid
import logging
import threading
import anyio
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

def _is_test_mode() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI") or "pytest" in sys.modules)


def _log_optional_dependency(message: str) -> None:
    if _is_test_mode():
        logging.debug(message)
    else:
        logging.info(message)


# Try to import required packages with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    _log_optional_dependency("psutil not available. CPU and memory monitoring will be limited")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    _log_optional_dependency("GPUtil not available. GPU monitoring will be limited")

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    _log_optional_dependency("duckdb not available. Metrics storage will be disabled")

# Import hardware capability detector for integration
from hardware_capability_detector import (
    HardwareCapabilityDetector,
    HardwareType,
    HardwareVendor,
    PrecisionType,
    CapabilityScore,
    HardwareCapability,
    WorkerHardwareCapabilities
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("hardware_utilization_monitor")


@dataclass
class ResourceUtilization:
    """Data class for resource utilization metrics."""
    timestamp: datetime
    # CPU metrics
    cpu_percent: float = 0.0
    cpu_times: Dict[str, float] = field(default_factory=dict)
    cpu_count: int = 0
    cpu_freq: Dict[str, float] = field(default_factory=dict)
    # Memory metrics
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    swap_percent: float = 0.0
    swap_used_gb: float = 0.0
    # GPU metrics
    gpu_utilization: List[Dict[str, Any]] = field(default_factory=list)
    # Disk metrics
    disk_percent: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    # Network metrics
    net_sent_mb: float = 0.0
    net_recv_mb: float = 0.0
    # Additional metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    # Task information (if monitoring during task execution)
    task_id: Optional[str] = None


@dataclass
class TaskResourceUsage:
    """Data class for tracking resource usage during task execution."""
    task_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    metrics: List[ResourceUtilization] = field(default_factory=list)
    # Peak usage metrics
    peak_cpu_percent: float = 0.0
    peak_memory_percent: float = 0.0
    peak_gpu_percent: float = 0.0
    # Average usage metrics
    avg_cpu_percent: float = 0.0
    avg_memory_percent: float = 0.0
    avg_gpu_percent: float = 0.0
    # Cumulative metrics
    total_disk_read_mb: float = 0.0
    total_disk_write_mb: float = 0.0
    total_net_sent_mb: float = 0.0
    total_net_recv_mb: float = 0.0
    # Completion status
    completed: bool = False
    success: bool = False
    error: Optional[str] = None


@dataclass
class HardwareAlert:
    """Data class for hardware utilization alerts."""
    timestamp: datetime
    resource_type: str
    severity: str
    message: str
    metric_value: float
    threshold: float
    task_id: Optional[str] = None


class MonitoringLevel(Enum):
    """Monitoring detail level."""
    BASIC = "basic"      # CPU, memory only, less frequent
    STANDARD = "standard"  # Default level, all resources, moderate frequency
    DETAILED = "detailed"  # All resources with more metrics, high frequency
    INTENSIVE = "intensive"  # Maximum level, all metrics, highest frequency


class HardwareUtilizationMonitor:
    """
    Monitor for hardware resource utilization.
    
    This class provides real-time monitoring of hardware resources with configurable
    monitoring levels, alert thresholds, and database integration.
    """
    
    def __init__(
        self,
        worker_id: Optional[str] = None,
        db_path: Optional[str] = None,
        monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD,
        interval_seconds: float = 1.0,
        alert_thresholds: Optional[Dict[str, float]] = None,
        hardware_detector: Optional[HardwareCapabilityDetector] = None
    ):
        """
        Initialize the hardware utilization monitor.
        
        Args:
            worker_id: Worker ID (auto-generated if not provided)
            db_path: Path to DuckDB database for metrics storage
            monitoring_level: Level of detail for monitoring
            interval_seconds: Interval between measurements in seconds
            alert_thresholds: Dictionary of {metric_name: threshold_value}
            hardware_detector: Optional hardware capability detector instance
        """
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.db_path = db_path
        self.monitoring_level = monitoring_level
        self.interval_seconds = interval_seconds
        self.alert_thresholds = alert_thresholds or {
            "cpu_percent": 90.0,
            "memory_percent": 90.0,
            "gpu_percent": 95.0,
            "disk_percent": 95.0
        }
        
        # Hardware detector for integration
        self.hardware_detector = hardware_detector
        
        # Database connection
        self.db_connection = None
        if db_path and DUCKDB_AVAILABLE:
            self._init_database()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.current_metrics = None
        
        # Resource tracking variables
        self.baseline_metrics = None
        self.prev_disk_io = None
        self.prev_net_io = None
        
        # Task tracking
        self.task_resources = {}
        self.active_task_id = None
        
        # Alert tracking
        self.alerts = []
        self.alert_callbacks = []
        
        # Historical metrics
        self.historical_metrics = []
        self.max_historical_metrics = 1000  # Limit to prevent memory issues
        
        # Initialize hardware capability detection if not provided
        if not self.hardware_detector and db_path and DUCKDB_AVAILABLE:
            self.hardware_detector = HardwareCapabilityDetector(
                worker_id=self.worker_id,
                db_path=self.db_path
            )
    
    def _init_database(self):
        """Initialize database connection and tables."""
        try:
            # Connect to database
            self.db_connection = duckdb.connect(self.db_path)
            
            # Create necessary tables
            self._create_tables()
            
            logger.info(f"Database connection established to {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            self.db_connection = None
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        if not self.db_connection:
            return
        
        try:
            # Create resource_utilization table
            self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS resource_utilization (
                    id BIGINT PRIMARY KEY,
                    worker_id VARCHAR,
                    timestamp TIMESTAMP,
                    cpu_percent FLOAT,
                    memory_percent FLOAT,
                    memory_used_gb FLOAT,
                    memory_available_gb FLOAT,
                    swap_percent FLOAT,
                    gpu_utilization JSON,
                    disk_percent FLOAT,
                    disk_read_mb FLOAT,
                    disk_write_mb FLOAT,
                    net_sent_mb FLOAT,
                    net_recv_mb FLOAT,
                    metrics JSON,
                    task_id VARCHAR
                )
            """)
            
            # Create task_resource_usage table
            self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS task_resource_usage (
                    id BIGINT PRIMARY KEY,
                    task_id VARCHAR,
                    worker_id VARCHAR,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    peak_cpu_percent FLOAT,
                    peak_memory_percent FLOAT,
                    peak_gpu_percent FLOAT,
                    avg_cpu_percent FLOAT,
                    avg_memory_percent FLOAT,
                    avg_gpu_percent FLOAT,
                    total_disk_read_mb FLOAT,
                    total_disk_write_mb FLOAT,
                    total_net_sent_mb FLOAT,
                    total_net_recv_mb FLOAT,
                    completed BOOLEAN,
                    success BOOLEAN,
                    error VARCHAR
                )
            """)
            
            # Create hardware_alerts table
            self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS hardware_alerts (
                    id BIGINT PRIMARY KEY,
                    worker_id VARCHAR,
                    timestamp TIMESTAMP,
                    resource_type VARCHAR,
                    severity VARCHAR,
                    message VARCHAR,
                    metric_value FLOAT,
                    threshold FLOAT,
                    task_id VARCHAR
                )
            """)
            
            # Ensure sequences exist for id assignment on insert
            self.db_connection.execute("CREATE SEQUENCE IF NOT EXISTS resource_utilization_id_seq")
            self.db_connection.execute("CREATE SEQUENCE IF NOT EXISTS task_resource_usage_id_seq")
            self.db_connection.execute("CREATE SEQUENCE IF NOT EXISTS hardware_alerts_id_seq")

            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create database tables: {str(e)}")
    
    def start_monitoring(self):
        """Start monitoring hardware utilization."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        # Take baseline measurements
        self.baseline_metrics = self._collect_metrics()
        
        # Set monitoring state
        self.monitoring_active = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Hardware utilization monitoring started with {self.monitoring_level.value} level")
    
    def stop_monitoring(self):
        """Stop monitoring hardware utilization."""
        if not self.monitoring_active:
            logger.warning("Monitoring is not active")
            return
        
        # Set monitoring state
        self.monitoring_active = False
        
        # Wait for thread to terminate
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        logger.info("Hardware utilization monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop, runs in a separate thread."""
        try:
            while self.monitoring_active:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Update current metrics
                self.current_metrics = metrics
                
                # Add to historical metrics
                self._add_to_historical_metrics(metrics)
                
                # Store in database if connected
                self._store_metrics(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Update task metrics if task is active
                if self.active_task_id:
                    self._update_task_metrics(metrics)
                
                # Sleep for interval
                time.sleep(self.interval_seconds)
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
            self.monitoring_active = False
    
    def _collect_metrics(self) -> ResourceUtilization:
        """
        Collect hardware utilization metrics.
        
        Returns:
            ResourceUtilization object with current metrics
        """
        now = datetime.now()
        metrics = ResourceUtilization(timestamp=now, task_id=self.active_task_id)
        
        # Collect CPU metrics if psutil is available
        if PSUTIL_AVAILABLE:
            metrics.cpu_percent = psutil.cpu_percent(interval=None)
            metrics.cpu_count = psutil.cpu_count(logical=True)
            
            if self.monitoring_level in [MonitoringLevel.DETAILED, MonitoringLevel.INTENSIVE]:
                # Collect detailed CPU metrics
                cpu_times = psutil.cpu_times_percent()
                metrics.cpu_times = {
                    "user": cpu_times.user,
                    "system": cpu_times.system,
                    "idle": cpu_times.idle
                }
                
                try:
                    cpu_freq = psutil.cpu_freq()
                    if cpu_freq:
                        metrics.cpu_freq = {
                            "current": cpu_freq.current,
                            "min": cpu_freq.min if hasattr(cpu_freq, 'min') else None,
                            "max": cpu_freq.max if hasattr(cpu_freq, 'max') else None
                        }
                except Exception as e:
                    logger.debug(f"Failed to get CPU frequency info: {str(e)}")
            
            # Collect memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_used_gb = memory.used / (1024 ** 3)
            metrics.memory_available_gb = memory.available / (1024 ** 3)
            
            swap = psutil.swap_memory()
            metrics.swap_percent = swap.percent
            metrics.swap_used_gb = swap.used / (1024 ** 3)
            
            # Collect disk metrics
            disk = psutil.disk_usage('/')
            metrics.disk_percent = disk.percent
            
            # Collect disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if disk_io and self.prev_disk_io:
                time_diff = (now - self.baseline_metrics.timestamp).total_seconds()
                metrics.disk_read_mb = (disk_io.read_bytes - self.prev_disk_io.read_bytes) / (1024 ** 2)
                metrics.disk_write_mb = (disk_io.write_bytes - self.prev_disk_io.write_bytes) / (1024 ** 2)
            self.prev_disk_io = disk_io
            
            # Collect network metrics
            net_io = psutil.net_io_counters()
            if net_io and self.prev_net_io:
                time_diff = (now - self.baseline_metrics.timestamp).total_seconds()
                metrics.net_sent_mb = (net_io.bytes_sent - self.prev_net_io.bytes_sent) / (1024 ** 2)
                metrics.net_recv_mb = (net_io.bytes_recv - self.prev_net_io.bytes_recv) / (1024 ** 2)
            self.prev_net_io = net_io
        
        # Collect GPU metrics if GPUtil is available
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_metrics = {
                        "id": i,
                        "name": gpu.name,
                        "load": gpu.load * 100.0,  # Convert to percentage
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100.0 if gpu.memoryTotal > 0 else 0.0,
                        "temperature": gpu.temperature
                    }
                    metrics.gpu_utilization.append(gpu_metrics)
            except Exception as e:
                logger.debug(f"Failed to get GPU metrics: {str(e)}")
        
        # Collect additional metrics for intensive monitoring
        if self.monitoring_level == MonitoringLevel.INTENSIVE:
            if PSUTIL_AVAILABLE:
                try:
                    # Process information
                    process_count = len(psutil.pids())
                    metrics.metrics["process_count"] = process_count
                    
                    # System load average (Linux-only)
                    if hasattr(psutil, 'getloadavg'):
                        load1, load5, load15 = psutil.getloadavg()
                        metrics.metrics["load_avg_1min"] = load1
                        metrics.metrics["load_avg_5min"] = load5
                        metrics.metrics["load_avg_15min"] = load15
                    
                    # Per-CPU utilization
                    if hasattr(psutil, 'cpu_percent') and self.cpu_count > 1:
                        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
                        metrics.metrics["per_cpu_percent"] = per_cpu
                except Exception as e:
                    logger.debug(f"Failed to get intensive metrics: {str(e)}")
        
        return metrics
    
    def _add_to_historical_metrics(self, metrics: ResourceUtilization):
        """Add metrics to historical record, with limit to prevent memory issues."""
        self.historical_metrics.append(metrics)
        
        # Trim list if it gets too large
        if len(self.historical_metrics) > self.max_historical_metrics:
            # Remove oldest metrics
            self.historical_metrics = self.historical_metrics[-self.max_historical_metrics:]
    
    def _store_metrics(self, metrics: ResourceUtilization):
        """Store metrics in database."""
        if not self.db_connection:
            return
        
        try:
            # Convert GPU metrics to JSON string
            gpu_json = json.dumps(metrics.gpu_utilization)
            metrics_json = json.dumps(metrics.metrics)
            
            # Insert into database
            self.db_connection.execute("""
                INSERT INTO resource_utilization (
                    id, worker_id, timestamp, cpu_percent, memory_percent, memory_used_gb,
                    memory_available_gb, swap_percent, gpu_utilization, disk_percent,
                    disk_read_mb, disk_write_mb, net_sent_mb, net_recv_mb, metrics, task_id
                ) VALUES (nextval('resource_utilization_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                self.worker_id, metrics.timestamp, metrics.cpu_percent,
                metrics.memory_percent, metrics.memory_used_gb, metrics.memory_available_gb,
                metrics.swap_percent, gpu_json, metrics.disk_percent,
                metrics.disk_read_mb, metrics.disk_write_mb, metrics.net_sent_mb,
                metrics.net_recv_mb, metrics_json, metrics.task_id
            ])
        except Exception as e:
            logger.warning(f"Failed to store metrics in database: {str(e)}")
    
    def _check_alerts(self, metrics: ResourceUtilization):
        """Check for threshold-based alerts."""
        alerts = []
        
        # Check CPU utilization
        if metrics.cpu_percent >= self.alert_thresholds.get("cpu_percent", 90.0):
            alerts.append(HardwareAlert(
                timestamp=metrics.timestamp,
                resource_type="cpu",
                severity="warning" if metrics.cpu_percent < 95.0 else "critical",
                message=f"CPU utilization at {metrics.cpu_percent:.1f}%",
                metric_value=metrics.cpu_percent,
                threshold=self.alert_thresholds.get("cpu_percent", 90.0),
                task_id=metrics.task_id
            ))
        
        # Check memory utilization
        if metrics.memory_percent >= self.alert_thresholds.get("memory_percent", 90.0):
            alerts.append(HardwareAlert(
                timestamp=metrics.timestamp,
                resource_type="memory",
                severity="warning" if metrics.memory_percent < 95.0 else "critical",
                message=f"Memory utilization at {metrics.memory_percent:.1f}%",
                metric_value=metrics.memory_percent,
                threshold=self.alert_thresholds.get("memory_percent", 90.0),
                task_id=metrics.task_id
            ))
        
        # Check GPU utilization
        for gpu_info in metrics.gpu_utilization:
            if gpu_info.get("load", 0.0) >= self.alert_thresholds.get("gpu_percent", 95.0):
                alerts.append(HardwareAlert(
                    timestamp=metrics.timestamp,
                    resource_type="gpu",
                    severity="warning" if gpu_info.get("load", 0.0) < 98.0 else "critical",
                    message=f"GPU {gpu_info.get('name', 'Unknown')} utilization at {gpu_info.get('load', 0.0):.1f}%",
                    metric_value=gpu_info.get("load", 0.0),
                    threshold=self.alert_thresholds.get("gpu_percent", 95.0),
                    task_id=metrics.task_id
                ))
            
            # Check GPU memory
            if gpu_info.get("memory_percent", 0.0) >= self.alert_thresholds.get("gpu_memory_percent", 90.0):
                alerts.append(HardwareAlert(
                    timestamp=metrics.timestamp,
                    resource_type="gpu_memory",
                    severity="warning" if gpu_info.get("memory_percent", 0.0) < 95.0 else "critical",
                    message=f"GPU {gpu_info.get('name', 'Unknown')} memory at {gpu_info.get('memory_percent', 0.0):.1f}%",
                    metric_value=gpu_info.get("memory_percent", 0.0),
                    threshold=self.alert_thresholds.get("gpu_memory_percent", 90.0),
                    task_id=metrics.task_id
                ))
        
        # Check disk utilization
        if metrics.disk_percent >= self.alert_thresholds.get("disk_percent", 95.0):
            alerts.append(HardwareAlert(
                timestamp=metrics.timestamp,
                resource_type="disk",
                severity="warning" if metrics.disk_percent < 98.0 else "critical",
                message=f"Disk utilization at {metrics.disk_percent:.1f}%",
                metric_value=metrics.disk_percent,
                threshold=self.alert_thresholds.get("disk_percent", 95.0),
                task_id=metrics.task_id
            ))
        
        # Store alerts
        for alert in alerts:
            self.alerts.append(alert)
            
            # Store in database
            self._store_alert(alert)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.warning(f"Error in alert callback: {str(e)}")
            
            # Log alert
            log_level = logging.WARNING if alert.severity == "warning" else logging.ERROR
            if _is_test_mode():
                log_level = logging.DEBUG
            logger.log(log_level, f"Hardware Alert: {alert.message}")
    
    def _store_alert(self, alert: HardwareAlert):
        """Store alert in database."""
        if not self.db_connection:
            return
        
        try:
            self.db_connection.execute("""
                INSERT INTO hardware_alerts (
                    id, worker_id, timestamp, resource_type, severity, message,
                    metric_value, threshold, task_id
                ) VALUES (nextval('hardware_alerts_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                self.worker_id, alert.timestamp, alert.resource_type,
                alert.severity, alert.message, alert.metric_value,
                alert.threshold, alert.task_id
            ])
        except Exception as e:
            logger.warning(f"Failed to store alert in database: {str(e)}")
    
    def start_task_monitoring(self, task_id: str):
        """
        Start monitoring resources for a specific task.
        
        Args:
            task_id: ID of the task to monitor
        """
        if not self.monitoring_active:
            self.start_monitoring()
        
        self.active_task_id = task_id
        
        # Initialize task resource tracking
        self.task_resources[task_id] = TaskResourceUsage(
            task_id=task_id,
            start_time=datetime.now()
        )
        
        logger.info(f"Started monitoring resources for task {task_id}")
    
    def stop_task_monitoring(self, task_id: str, success: bool = True, error: Optional[str] = None):
        """
        Stop monitoring resources for a specific task.
        
        Args:
            task_id: ID of the task to stop monitoring
            success: Whether the task completed successfully
            error: Error message if the task failed
        """
        if task_id not in self.task_resources:
            logger.warning(f"Task {task_id} not being monitored")
            return
        
        # Set active task to None if it matches
        if self.active_task_id == task_id:
            self.active_task_id = None
        
        # Update task resource usage
        task_usage = self.task_resources[task_id]
        task_usage.end_time = datetime.now()
        task_usage.completed = True
        task_usage.success = success
        task_usage.error = error
        
        # Store in database
        self._store_task_usage(task_id)
        
        logger.info(f"Stopped monitoring resources for task {task_id}")
        
        # Return task resource usage for potential reporting
        return task_usage
    
    def _update_task_metrics(self, metrics: ResourceUtilization):
        """Update task resource usage with new metrics."""
        if not self.active_task_id or self.active_task_id not in self.task_resources:
            return
        
        task_usage = self.task_resources[self.active_task_id]
        
        # Add metrics to list
        task_usage.metrics.append(metrics)
        
        # Update peak values
        task_usage.peak_cpu_percent = max(task_usage.peak_cpu_percent, metrics.cpu_percent)
        task_usage.peak_memory_percent = max(task_usage.peak_memory_percent, metrics.memory_percent)
        
        # Update peak GPU values
        for gpu_info in metrics.gpu_utilization:
            task_usage.peak_gpu_percent = max(task_usage.peak_gpu_percent, gpu_info.get("load", 0.0))
        
        # Update cumulative metrics
        task_usage.total_disk_read_mb += metrics.disk_read_mb
        task_usage.total_disk_write_mb += metrics.disk_write_mb
        task_usage.total_net_sent_mb += metrics.net_sent_mb
        task_usage.total_net_recv_mb += metrics.net_recv_mb
        
        # Calculate averages
        if task_usage.metrics:
            task_usage.avg_cpu_percent = sum(m.cpu_percent for m in task_usage.metrics) / len(task_usage.metrics)
            task_usage.avg_memory_percent = sum(m.memory_percent for m in task_usage.metrics) / len(task_usage.metrics)
            
            # Calculate average GPU utilization
            gpu_values = []
            for m in task_usage.metrics:
                for gpu_info in m.gpu_utilization:
                    gpu_values.append(gpu_info.get("load", 0.0))
            if gpu_values:
                task_usage.avg_gpu_percent = sum(gpu_values) / len(gpu_values)
    
    def _store_task_usage(self, task_id: str):
        """Store task resource usage in database."""
        if not self.db_connection or task_id not in self.task_resources:
            return
        
        task_usage = self.task_resources[task_id]
        
        try:
            self.db_connection.execute("""
                INSERT INTO task_resource_usage (
                    id, task_id, worker_id, start_time, end_time, peak_cpu_percent,
                    peak_memory_percent, peak_gpu_percent, avg_cpu_percent,
                    avg_memory_percent, avg_gpu_percent, total_disk_read_mb,
                    total_disk_write_mb, total_net_sent_mb, total_net_recv_mb,
                    completed, success, error
                ) VALUES (nextval('task_resource_usage_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                task_usage.task_id, self.worker_id, task_usage.start_time,
                task_usage.end_time, task_usage.peak_cpu_percent,
                task_usage.peak_memory_percent, task_usage.peak_gpu_percent,
                task_usage.avg_cpu_percent, task_usage.avg_memory_percent,
                task_usage.avg_gpu_percent, task_usage.total_disk_read_mb,
                task_usage.total_disk_write_mb, task_usage.total_net_sent_mb,
                task_usage.total_net_recv_mb, task_usage.completed,
                task_usage.success, task_usage.error
            ])
        except Exception as e:
            logger.warning(f"Failed to store task usage in database: {str(e)}")
    
    def get_current_metrics(self) -> Optional[ResourceUtilization]:
        """
        Get the current hardware utilization metrics.
        
        Returns:
            ResourceUtilization object with current metrics or None if monitoring is not active
        """
        if not self.monitoring_active:
            logger.debug("Monitoring is not active")
        
        return self.current_metrics
    
    def get_task_metrics(self, task_id: str) -> Optional[TaskResourceUsage]:
        """
        Get resource usage for a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            TaskResourceUsage object or None if task not found
        """
        return self.task_resources.get(task_id)
    
    def get_alerts(self, resource_type: Optional[str] = None, severity: Optional[str] = None,
                 task_id: Optional[str] = None) -> List[HardwareAlert]:
        """
        Get hardware alerts with optional filtering.
        
        Args:
            resource_type: Filter by resource type
            severity: Filter by severity
            task_id: Filter by task ID
            
        Returns:
            List of hardware alerts
        """
        filtered_alerts = self.alerts
        
        if resource_type:
            filtered_alerts = [a for a in filtered_alerts if a.resource_type == resource_type]
        
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
        
        if task_id:
            filtered_alerts = [a for a in filtered_alerts if a.task_id == task_id]
        
        return filtered_alerts
    
    def register_alert_callback(self, callback):
        """
        Register a callback function for hardware alerts.
        
        Args:
            callback: Function that takes a HardwareAlert as parameter
        """
        self.alert_callbacks.append(callback)
    
    def get_historical_metrics(self, 
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None,
                             task_id: Optional[str] = None) -> List[ResourceUtilization]:
        """
        Get historical metrics with optional filtering.
        
        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            task_id: Filter by task ID
            
        Returns:
            List of resource utilization metrics
        """
        filtered_metrics = self.historical_metrics
        
        if start_time:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= start_time]
        
        if end_time:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp <= end_time]
        
        if task_id:
            filtered_metrics = [m for m in filtered_metrics if m.task_id == task_id]
        
        return filtered_metrics
    
    def get_resource_utilization_from_db(self,
                                       start_time: Optional[datetime] = None,
                                       end_time: Optional[datetime] = None,
                                       task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get resource utilization metrics from database with optional filtering.
        
        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            task_id: Filter by task ID
            
        Returns:
            List of resource utilization metrics as dictionaries
        """
        if not self.db_connection:
            logger.warning("No database connection")
            return []
        
        try:
            query = """
                SELECT * FROM resource_utilization
                WHERE worker_id = ?
            """
            params = [self.worker_id]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            if task_id:
                query += " AND task_id = ?"
                params.append(task_id)
            
            query += " ORDER BY timestamp"
            
            # Execute query
            results = self.db_connection.execute(query, params).fetchall()
            
            # Convert to dictionaries
            metrics = []
            for row in results:
                metric = {
                    "id": row[0],
                    "worker_id": row[1],
                    "timestamp": row[2],
                    "cpu_percent": row[3],
                    "memory_percent": row[4],
                    "memory_used_gb": row[5],
                    "memory_available_gb": row[6],
                    "swap_percent": row[7],
                    "gpu_utilization": json.loads(row[8]) if row[8] else [],
                    "disk_percent": row[9],
                    "disk_read_mb": row[10],
                    "disk_write_mb": row[11],
                    "net_sent_mb": row[12],
                    "net_recv_mb": row[13],
                    "metrics": json.loads(row[14]) if row[14] else {},
                    "task_id": row[15]
                }
                metrics.append(metric)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to query resource utilization from database: {str(e)}")
            return []
    
    def get_task_usage_from_db(self,
                             task_id: Optional[str] = None,
                             completed: Optional[bool] = None,
                             success: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Get task resource usage from database with optional filtering.
        
        Args:
            task_id: Filter by task ID
            completed: Filter by completion status
            success: Filter by success status
            
        Returns:
            List of task resource usage as dictionaries
        """
        if not self.db_connection:
            logger.warning("No database connection")
            return []
        
        try:
            query = """
                SELECT * FROM task_resource_usage
                WHERE worker_id = ?
            """
            params = [self.worker_id]
            
            if task_id:
                query += " AND task_id = ?"
                params.append(task_id)
            
            if completed is not None:
                query += " AND completed = ?"
                params.append(completed)
            
            if success is not None:
                query += " AND success = ?"
                params.append(success)
            
            query += " ORDER BY start_time DESC"
            
            # Execute query
            results = self.db_connection.execute(query, params).fetchall()
            
            # Convert to dictionaries
            task_usages = []
            for row in results:
                task_usage = {
                    "id": row[0],
                    "task_id": row[1],
                    "worker_id": row[2],
                    "start_time": row[3],
                    "end_time": row[4],
                    "peak_cpu_percent": row[5],
                    "peak_memory_percent": row[6],
                    "peak_gpu_percent": row[7],
                    "avg_cpu_percent": row[8],
                    "avg_memory_percent": row[9],
                    "avg_gpu_percent": row[10],
                    "total_disk_read_mb": row[11],
                    "total_disk_write_mb": row[12],
                    "total_net_sent_mb": row[13],
                    "total_net_recv_mb": row[14],
                    "completed": row[15],
                    "success": row[16],
                    "error": row[17]
                }
                task_usages.append(task_usage)
            
            return task_usages
        
        except Exception as e:
            logger.error(f"Failed to query task usage from database: {str(e)}")
            return []
    
    def export_metrics_to_json(self, file_path: str):
        """
        Export metrics to a JSON file.
        
        Args:
            file_path: Path to output file
        """
        try:
            # Prepare data for serialization
            data = {
                "worker_id": self.worker_id,
                "monitoring_level": self.monitoring_level.value,
                "interval_seconds": self.interval_seconds,
                "export_time": datetime.now().isoformat(),
                "current_metrics": self._serialize_metrics(self.current_metrics) if self.current_metrics else None,
                "historical_metrics": [self._serialize_metrics(m) for m in self.historical_metrics],
                "alerts": [self._serialize_alert(a) for a in self.alerts],
                "task_resources": {
                    task_id: self._serialize_task_usage(usage)
                    for task_id, usage in self.task_resources.items()
                }
            }
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported metrics to {file_path}")
        
        except Exception as e:
            logger.error(f"Failed to export metrics to JSON: {str(e)}")
    
    def _serialize_metrics(self, metrics: ResourceUtilization) -> Dict[str, Any]:
        """Convert ResourceUtilization object to serializable dictionary."""
        if not metrics:
            return None
        
        return {
            "timestamp": metrics.timestamp.isoformat(),
            "cpu_percent": metrics.cpu_percent,
            "cpu_times": metrics.cpu_times,
            "cpu_count": metrics.cpu_count,
            "cpu_freq": metrics.cpu_freq,
            "memory_percent": metrics.memory_percent,
            "memory_used_gb": metrics.memory_used_gb,
            "memory_available_gb": metrics.memory_available_gb,
            "swap_percent": metrics.swap_percent,
            "swap_used_gb": metrics.swap_used_gb,
            "gpu_utilization": metrics.gpu_utilization,
            "disk_percent": metrics.disk_percent,
            "disk_read_mb": metrics.disk_read_mb,
            "disk_write_mb": metrics.disk_write_mb,
            "net_sent_mb": metrics.net_sent_mb,
            "net_recv_mb": metrics.net_recv_mb,
            "metrics": metrics.metrics,
            "task_id": metrics.task_id
        }
    
    def _serialize_alert(self, alert: HardwareAlert) -> Dict[str, Any]:
        """Convert HardwareAlert object to serializable dictionary."""
        return {
            "timestamp": alert.timestamp.isoformat(),
            "resource_type": alert.resource_type,
            "severity": alert.severity,
            "message": alert.message,
            "metric_value": alert.metric_value,
            "threshold": alert.threshold,
            "task_id": alert.task_id
        }
    
    def _serialize_task_usage(self, usage: TaskResourceUsage) -> Dict[str, Any]:
        """Convert TaskResourceUsage object to serializable dictionary."""
        return {
            "task_id": usage.task_id,
            "start_time": usage.start_time.isoformat(),
            "end_time": usage.end_time.isoformat() if usage.end_time else None,
            "metrics": [self._serialize_metrics(m) for m in usage.metrics],
            "peak_cpu_percent": usage.peak_cpu_percent,
            "peak_memory_percent": usage.peak_memory_percent,
            "peak_gpu_percent": usage.peak_gpu_percent,
            "avg_cpu_percent": usage.avg_cpu_percent,
            "avg_memory_percent": usage.avg_memory_percent,
            "avg_gpu_percent": usage.avg_gpu_percent,
            "total_disk_read_mb": usage.total_disk_read_mb,
            "total_disk_write_mb": usage.total_disk_write_mb,
            "total_net_sent_mb": usage.total_net_sent_mb,
            "total_net_recv_mb": usage.total_net_recv_mb,
            "completed": usage.completed,
            "success": usage.success,
            "error": usage.error
        }
    
    def generate_utilization_report(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive utilization report.
        
        Args:
            task_id: Optional task ID to filter report for a specific task
            
        Returns:
            Dictionary with utilization report data
        """
        report = {
            "worker_id": self.worker_id,
            "report_time": datetime.now().isoformat(),
            "monitoring_level": self.monitoring_level.value,
            "summary": {},
            "alerts": [],
            "tasks": []
        }
        
        # Current metrics
        if self.current_metrics:
            report["current_metrics"] = self._serialize_metrics(self.current_metrics)
        
        # Summary statistics
        if self.historical_metrics:
            # CPU stats
            cpu_values = [m.cpu_percent for m in self.historical_metrics]
            report["summary"]["cpu"] = {
                "min": min(cpu_values),
                "max": max(cpu_values),
                "avg": sum(cpu_values) / len(cpu_values),
                "current": self.current_metrics.cpu_percent if self.current_metrics else None
            }
            
            # Memory stats
            memory_values = [m.memory_percent for m in self.historical_metrics]
            report["summary"]["memory"] = {
                "min": min(memory_values),
                "max": max(memory_values),
                "avg": sum(memory_values) / len(memory_values),
                "current": self.current_metrics.memory_percent if self.current_metrics else None
            }
            
            # GPU stats if available
            gpu_values = []
            for metric in self.historical_metrics:
                for gpu_info in metric.gpu_utilization:
                    gpu_values.append(gpu_info.get("load", 0.0))
            
            if gpu_values:
                report["summary"]["gpu"] = {
                    "min": min(gpu_values),
                    "max": max(gpu_values),
                    "avg": sum(gpu_values) / len(gpu_values),
                    "current": self.current_metrics.gpu_utilization[0].get("load", 0.0) if self.current_metrics and self.current_metrics.gpu_utilization else None
                }
            
            # Network stats
            net_sent_values = [m.net_sent_mb for m in self.historical_metrics if m.net_sent_mb is not None]
            net_recv_values = [m.net_recv_mb for m in self.historical_metrics if m.net_recv_mb is not None]
            
            if net_sent_values and net_recv_values:
                report["summary"]["network"] = {
                    "sent": {
                        "min": min(net_sent_values),
                        "max": max(net_sent_values),
                        "avg": sum(net_sent_values) / len(net_sent_values),
                        "total": sum(net_sent_values)
                    },
                    "received": {
                        "min": min(net_recv_values),
                        "max": max(net_recv_values),
                        "avg": sum(net_recv_values) / len(net_recv_values),
                        "total": sum(net_recv_values)
                    }
                }
        
        # Alerts
        filtered_alerts = self.alerts
        if task_id:
            filtered_alerts = [a for a in filtered_alerts if a.task_id == task_id]
        
        report["alerts"] = [self._serialize_alert(a) for a in filtered_alerts]
        
        # Task usage
        if task_id:
            # Get specific task
            if task_id in self.task_resources:
                task_usage = self.task_resources[task_id]
                report["tasks"].append(self._serialize_task_usage(task_usage))
        else:
            # Get all tasks
            for task_id, task_usage in self.task_resources.items():
                report["tasks"].append(self._serialize_task_usage(task_usage))
        
        return report
    
    def generate_html_report(self, file_path: str, task_id: Optional[str] = None):
        """
        Generate an HTML report of hardware utilization.
        
        Args:
            file_path: Path to output HTML file
            task_id: Optional task ID to filter report for a specific task
        """
        try:
            # Generate report data
            report_data = self.generate_utilization_report(task_id)
            
            # Create HTML
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Hardware Utilization Report - Worker {report_data["worker_id"]}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .section {{
            margin-bottom: 30px;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .warning {{
            color: #e67e22;
        }}
        .critical {{
            color: #e74c3c;
        }}
        .success {{
            color: #2ecc71;
        }}
        .failure {{
            color: #e74c3c;
        }}
    </style>
</head>
<body>
    <h1>Hardware Utilization Report</h1>
    <p>Worker ID: {report_data["worker_id"]}</p>
    <p>Report Time: {report_data["report_time"]}</p>
    <p>Monitoring Level: {report_data["monitoring_level"]}</p>
    
    <div class="section">
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Resource</th>
                <th>Min</th>
                <th>Max</th>
                <th>Average</th>
                <th>Current</th>
            </tr>
"""
            
            # Add summary rows
            for resource, stats in report_data.get("summary", {}).items():
                current_value = stats.get("current")
                if current_value is not None:
                    current_str = f"{current_value:.1f}%"
                else:
                    current_str = "N/A"
                
                html += f"""
            <tr>
                <td>{resource.upper()}</td>
                <td>{stats.get("min", 0.0):.1f}%</td>
                <td>{stats.get("max", 0.0):.1f}%</td>
                <td>{stats.get("avg", 0.0):.1f}%</td>
                <td>{current_str}</td>
            </tr>
"""
            
            html += """
        </table>
    </div>
    
    <div class="section">
        <h2>Alerts</h2>
"""
            
            # Add alerts
            if report_data.get("alerts"):
                html += """
        <table>
            <tr>
                <th>Time</th>
                <th>Resource</th>
                <th>Severity</th>
                <th>Message</th>
                <th>Value</th>
                <th>Threshold</th>
                <th>Task</th>
            </tr>
"""
                
                for alert in report_data["alerts"]:
                    severity_class = "warning" if alert["severity"] == "warning" else "critical"
                    
                    html += f"""
            <tr>
                <td>{alert["timestamp"]}</td>
                <td>{alert["resource_type"]}</td>
                <td class="{severity_class}">{alert["severity"]}</td>
                <td>{alert["message"]}</td>
                <td>{alert["metric_value"]:.1f}%</td>
                <td>{alert["threshold"]:.1f}%</td>
                <td>{alert["task_id"] or "N/A"}</td>
            </tr>
"""
                
                html += """
        </table>
"""
            else:
                html += """
        <p>No alerts recorded.</p>
"""
            
            html += """
    </div>
    
    <div class="section">
        <h2>Task Usage</h2>
"""
            
            # Add task usage
            if report_data.get("tasks"):
                html += """
        <table>
            <tr>
                <th>Task ID</th>
                <th>Start Time</th>
                <th>End Time</th>
                <th>Status</th>
                <th>Peak CPU</th>
                <th>Peak Memory</th>
                <th>Peak GPU</th>
                <th>Avg CPU</th>
                <th>Avg Memory</th>
                <th>Avg GPU</th>
                <th>Disk I/O (MB)</th>
                <th>Network I/O (MB)</th>
            </tr>
"""
                
                for task in report_data["tasks"]:
                    status_class = "success" if task.get("success") else "failure"
                    status_text = "Completed" if task.get("completed") else "Running"
                    status_text = "Success" if task.get("completed") and task.get("success") else status_text
                    status_text = "Failed" if task.get("completed") and not task.get("success") else status_text
                    
                    html += f"""
            <tr>
                <td>{task["task_id"]}</td>
                <td>{task["start_time"]}</td>
                <td>{task["end_time"] or "Running"}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{task["peak_cpu_percent"]:.1f}%</td>
                <td>{task["peak_memory_percent"]:.1f}%</td>
                <td>{task["peak_gpu_percent"]:.1f}%</td>
                <td>{task["avg_cpu_percent"]:.1f}%</td>
                <td>{task["avg_memory_percent"]:.1f}%</td>
                <td>{task["avg_gpu_percent"]:.1f}%</td>
                <td>{task["total_disk_read_mb"]:.1f} / {task["total_disk_write_mb"]:.1f}</td>
                <td>{task["total_net_sent_mb"]:.1f} / {task["total_net_recv_mb"]:.1f}</td>
            </tr>
"""
                
                html += """
        </table>
"""
            else:
                html += """
        <p>No task usage recorded.</p>
"""
            
            html += """
    </div>
</body>
</html>
"""
            
            # Write to file
            with open(file_path, 'w') as f:
                f.write(html)
            
            logger.info(f"Generated HTML report at {file_path}")
        
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {str(e)}")


# Simple example usage
def main():
    """Simple example of using the hardware utilization monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hardware Utilization Monitor")
    parser.add_argument("--db-path", help="Path to DuckDB database")
    parser.add_argument("--duration", type=int, default=10, help="Monitoring duration in seconds")
    parser.add_argument("--level", default="standard", choices=["basic", "standard", "detailed", "intensive"],
                       help="Monitoring detail level")
    parser.add_argument("--task-id", help="Optional task ID for monitoring")
    parser.add_argument("--export-json", help="Export metrics to JSON file")
    parser.add_argument("--generate-html", help="Generate HTML report")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = HardwareUtilizationMonitor(
        db_path=args.db_path,
        monitoring_level=MonitoringLevel(args.level),
        interval_seconds=0.5  # Faster interval for demo
    )
    
    # Register alert callback
    def alert_callback(alert):
        print(f"ALERT: {alert.severity.upper()} - {alert.message}")
    
    monitor.register_alert_callback(alert_callback)
    
    # Start monitoring
    if args.task_id:
        monitor.start_task_monitoring(args.task_id)
    else:
        monitor.start_monitoring()
    
    print(f"Monitoring hardware utilization for {args.duration} seconds...")
    
    try:
        # Generate some load for testing
        for i in range(args.duration):
            # CPU load
            if i % 3 == 0:
                print(f"Generating CPU load ({i}/{args.duration})...")
                _ = [i*i for i in range(10000000)]
            
            # Memory load
            if i % 4 == 0:
                print(f"Generating memory load ({i}/{args.duration})...")
                big_list = [0] * 1000000
            
            # Sleep
            time.sleep(1)
            
            # Print current metrics
            current = monitor.get_current_metrics()
            if current:
                print(f"CPU: {current.cpu_percent:.1f}%, Memory: {current.memory_percent:.1f}%")
                
                # Print GPU metrics if available
                for gpu in current.gpu_utilization:
                    print(f"GPU {gpu.get('id', 0)}: {gpu.get('load', 0.0):.1f}%, Memory: {gpu.get('memory_percent', 0.0):.1f}%")
    
    finally:
        # Stop monitoring
        if args.task_id:
            task_usage = monitor.stop_task_monitoring(args.task_id)
            print(f"Task resource usage:")
            print(f"  Peak CPU: {task_usage.peak_cpu_percent:.1f}%")
            print(f"  Peak Memory: {task_usage.peak_memory_percent:.1f}%")
            print(f"  Peak GPU: {task_usage.peak_gpu_percent:.1f}%")
            print(f"  Avg CPU: {task_usage.avg_cpu_percent:.1f}%")
            print(f"  Avg Memory: {task_usage.avg_memory_percent:.1f}%")
        else:
            monitor.stop_monitoring()
        
        # Export metrics if requested
        if args.export_json:
            monitor.export_metrics_to_json(args.export_json)
            print(f"Exported metrics to {args.export_json}")
        
        # Generate HTML report if requested
        if args.generate_html:
            monitor.generate_html_report(args.generate_html, args.task_id)
            print(f"Generated HTML report at {args.generate_html}")


if __name__ == "__main__":
    main()