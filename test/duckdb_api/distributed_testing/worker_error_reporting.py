#!/usr/bin/env python3
"""
Worker Error Reporting System

This module enhances error reporting capabilities for worker nodes in the Distributed Testing Framework,
providing comprehensive error context and categorization for more effective error handling.
"""

import os
import sys
import json
import time
import uuid
import psutil
import socket
import platform
import logging
import traceback
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from duckdb_api.distributed_testing.distributed_error_handler import ErrorCategory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("worker_error_reporting")

class EnhancedErrorReporter:
    """Enhanced error reporting for worker nodes."""
    
    def __init__(self, worker_id: str, capabilities: Dict[str, Any]):
        """Initialize the enhanced error reporter.
        
        Args:
            worker_id: ID of the worker node
            capabilities: Hardware capabilities of the worker
        """
        self.worker_id = worker_id
        self.capabilities = capabilities
        self.error_history = []
        self.max_error_history = 100
        self.hardware_type = self._determine_primary_hardware_type()
        
        logger.info(f"Enhanced error reporting initialized for worker {worker_id}")
        logger.info(f"Primary hardware type: {self.hardware_type}")
    
    def _determine_primary_hardware_type(self) -> str:
        """Determine the primary hardware type for the worker.
        
        Returns:
            Primary hardware type string
        """
        hardware_types = self.capabilities.get("hardware_types", [])
        
        # Define priority order for hardware types
        priority_order = ["cuda", "rocm", "mps", "webgpu", "webnn", "cpu"]
        
        # Find the highest priority hardware type that is available
        for hardware in priority_order:
            if hardware in hardware_types:
                return hardware
                
        return "cpu"  # Default to CPU if no other type is found
    
    def categorize_error(self, error: Dict[str, Any]) -> ErrorCategory:
        """Categorize an error based on type and context.
        
        Args:
            error: Error information dictionary
            
        Returns:
            ErrorCategory indicating the type of error
        """
        error_type = error.get("type", "")
        error_message = error.get("message", "")
        traceback_str = error.get("traceback", "")
        
        # Resource allocation errors
        if any(term in error_message.lower() for term in 
               ["out of memory", "no space", "disk full", "resource exhausted", 
                "cuda out of memory", "gpu memory"]):
            return ErrorCategory.RESOURCE_ALLOCATION_ERROR
            
        # Resource cleanup errors
        if any(term in error_message.lower() for term in 
               ["resource leak", "unable to free", "could not release", 
                "cleanup failed", "resource not released"]):
            return ErrorCategory.RESOURCE_CLEANUP_ERROR
            
        # Network connection errors
        if any(term in error_message.lower() for term in 
               ["connection refused", "connection reset", "no route to host", 
                "name resolution", "network unreachable", "socket error", 
                "websocket", "connection error"]):
            return ErrorCategory.NETWORK_CONNECTION_ERROR
            
        # Network timeout errors
        if any(term in error_message.lower() for term in 
               ["timeout", "timed out", "deadline exceeded", "request timed out"]):
            return ErrorCategory.NETWORK_TIMEOUT_ERROR
            
        # Hardware availability errors
        if any(term in error_message.lower() for term in 
               ["device not found", "no cuda device", "no gpu available", 
                "hardware unavailable", "device unavailable"]):
            return ErrorCategory.HARDWARE_AVAILABILITY_ERROR
            
        # Hardware capability errors
        if any(term in error_message.lower() for term in 
               ["insufficient capability", "compute capability", "not supported", 
                "feature unavailable", "operation not supported"]):
            return ErrorCategory.HARDWARE_CAPABILITY_ERROR
            
        # Hardware performance errors
        if any(term in error_message.lower() for term in 
               ["thermal throttling", "overheating", "performance degradation", 
                "slow execution", "performance error"]):
            return ErrorCategory.HARDWARE_PERFORMANCE_ERROR
            
        # Worker disconnection errors
        if any(term in error_message.lower() for term in 
               ["disconnected", "connection lost", "connection closed", 
                "worker disconnected", "session expired"]):
            return ErrorCategory.WORKER_DISCONNECTION_ERROR
            
        # Worker crash errors
        if any(term in error_message.lower() for term in 
               ["crashed", "segmentation fault", "bus error", "access violation", 
                "illegal instruction", "fatal error", "worker crashed"]):
            return ErrorCategory.WORKER_CRASH_ERROR
            
        # Worker overload errors
        if any(term in error_message.lower() for term in 
               ["overloaded", "too many requests", "load too high", 
                "worker overload", "resource contention"]):
            return ErrorCategory.WORKER_OVERLOAD_ERROR
            
        # Test execution errors
        if any(term in error_message.lower() for term in 
               ["assertion failed", "test failed", "test error", "execution error", 
                "benchmark error", "test execution"]):
            return ErrorCategory.TEST_EXECUTION_ERROR
            
        # Test dependency errors
        if any(term in error_message.lower() for term in 
               ["dependency missing", "import error", "module not found", 
                "cannot import", "dependency error", "no module named"]):
            return ErrorCategory.TEST_DEPENDENCY_ERROR
            
        # Test configuration errors
        if any(term in error_message.lower() for term in 
               ["invalid configuration", "config error", "parameter error", 
                "invalid parameter", "configuration error"]):
            return ErrorCategory.TEST_CONFIGURATION_ERROR
            
        # Unknown errors
        return ErrorCategory.UNKNOWN_ERROR
    
    def collect_system_context(self) -> Dict[str, Any]:
        """Collect system context information for error reporting.
        
        Returns:
            Dictionary with system context
        """
        context = {
            "timestamp": datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "hardware_type": self.hardware_type
        }
        
        # Add hardware metrics if psutil is available
        try:
            import psutil
            
            # CPU metrics
            cpu_metrics = {}
            cpu_metrics["percent"] = psutil.cpu_percent(interval=0.1)
            cpu_metrics["count"] = psutil.cpu_count()
            cpu_metrics["physical_count"] = psutil.cpu_count(logical=False)
            cpu_frequencies = psutil.cpu_freq()
            if cpu_frequencies:
                cpu_metrics["frequency_mhz"] = cpu_frequencies.current
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_metrics = {
                "total_gb": round(memory.total / (1024 ** 3), 2),
                "available_gb": round(memory.available / (1024 ** 3), 2),
                "used_percent": memory.percent
            }
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_metrics = {
                "total_gb": round(disk.total / (1024 ** 3), 2),
                "free_gb": round(disk.free / (1024 ** 3), 2),
                "used_percent": disk.percent
            }
            
            # Network metrics
            net_io = psutil.net_io_counters()
            network_metrics = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "error_in": net_io.errin,
                "error_out": net_io.errout,
                "drop_in": net_io.dropin,
                "drop_out": net_io.dropout
            }
            
            # Add to context
            context["metrics"] = {
                "cpu": cpu_metrics,
                "memory": memory_metrics,
                "disk": disk_metrics,
                "network": network_metrics
            }
        except Exception as e:
            logger.warning(f"Error collecting system metrics: {e}")
            context["metrics"] = {"error": str(e)}
        
        # Add GPU metrics if CUDA is available
        try:
            if "cuda" in self.capabilities.get("hardware_types", []):
                import torch
                
                if torch.cuda.is_available():
                    gpu_metrics = {}
                    gpu_count = torch.cuda.device_count()
                    gpu_metrics["count"] = gpu_count
                    
                    devices = []
                    for i in range(gpu_count):
                        device_metrics = {}
                        device_metrics["name"] = torch.cuda.get_device_name(i)
                        device_metrics["index"] = i
                        
                        # Get compute capability
                        major, minor = torch.cuda.get_device_capability(i)
                        device_metrics["compute_capability"] = f"{major}.{minor}"
                        
                        # Get memory information
                        try:
                            mem_info = torch.cuda.get_device_properties(i).total_memory
                            free_mem = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
                            used_mem = torch.cuda.memory_allocated(i)
                            
                            device_metrics["total_memory_gb"] = round(mem_info / (1024 ** 3), 2)
                            device_metrics["free_memory_gb"] = round(free_mem / (1024 ** 3), 2)
                            device_metrics["used_memory_gb"] = round(used_mem / (1024 ** 3), 2)
                            device_metrics["memory_utilization"] = round((used_mem / mem_info) * 100, 2)
                        except Exception as e:
                            device_metrics["memory_error"] = str(e)
                            
                        devices.append(device_metrics)
                    
                    gpu_metrics["devices"] = devices
                    context["gpu_metrics"] = gpu_metrics
        except Exception as e:
            logger.warning(f"Error collecting GPU metrics: {e}")
            context["gpu_metrics"] = {"error": str(e)}
            
        return context
    
    def create_enhanced_error_report(self, error_type: str, message: str, 
                                     task_id: Optional[str] = None) -> Dict[str, Any]:
        """Create an enhanced error report with comprehensive context.
        
        Args:
            error_type: Type of error
            message: Error message
            task_id: Optional ID of the task that generated the error
            
        Returns:
            Dictionary with enhanced error information
        """
        # Get the exception traceback if available
        exc_traceback = traceback.format_exc() if sys.exc_info()[0] else ""
        
        # Create basic error information
        error_info = {
            "type": error_type,
            "message": message,
            "traceback": exc_traceback,
            "timestamp": datetime.now().isoformat(),
            "worker_id": self.worker_id,
            "task_id": task_id
        }
        
        # Categorize the error
        error_category = self.categorize_error(error_info)
        error_info["error_category"] = error_category
        
        # Add system context
        error_info["system_context"] = self.collect_system_context()
        
        # Add hardware context
        error_info["hardware_context"] = {
            "hardware_type": self.hardware_type,
            "hardware_types": self.capabilities.get("hardware_types", []),
            "hardware_status": {
                # Add any hardware-specific status information here
                "overheating": self._check_for_overheating(),
                "memory_pressure": self._check_memory_pressure(),
                "throttling": self._check_for_throttling()
            }
        }
        
        # Track error in history
        self._add_to_error_history(error_info)
        
        # Add error frequency information
        error_info["error_frequency"] = self._analyze_error_frequency(error_type, message)
        
        logger.info(f"Created enhanced error report for {error_type} error (category: {error_category})")
        return error_info
    
    def _add_to_error_history(self, error_info: Dict[str, Any]):
        """Add an error to the history.
        
        Args:
            error_info: Error information dictionary
        """
        # Add to history, keeping only the last N errors
        self.error_history.append({
            "type": error_info["type"],
            "message": error_info["message"],
            "error_category": error_info["error_category"],
            "timestamp": error_info["timestamp"],
            "task_id": error_info.get("task_id")
        })
        
        # Trim error history if needed
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
    
    def _analyze_error_frequency(self, error_type: str, message: str) -> Dict[str, Any]:
        """Analyze the frequency of similar errors in the history.
        
        Args:
            error_type: Type of error
            message: Error message
            
        Returns:
            Dictionary with error frequency information
        """
        # Count similar errors in the last N hours
        one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
        six_hours_ago = (datetime.now() - timedelta(hours=6)).isoformat()
        twenty_four_hours_ago = (datetime.now() - timedelta(hours=24)).isoformat()
        
        # Count by type
        count_same_type_1h = sum(1 for e in self.error_history 
                               if e["type"] == error_type and e["timestamp"] >= one_hour_ago)
        count_same_type_6h = sum(1 for e in self.error_history 
                               if e["type"] == error_type and e["timestamp"] >= six_hours_ago)
        count_same_type_24h = sum(1 for e in self.error_history 
                                if e["type"] == error_type and e["timestamp"] >= twenty_four_hours_ago)
        
        # Count by similarity of message (simple substring match)
        count_similar_message_1h = sum(1 for e in self.error_history 
                                     if message in e["message"] and e["timestamp"] >= one_hour_ago)
        count_similar_message_6h = sum(1 for e in self.error_history 
                                     if message in e["message"] and e["timestamp"] >= six_hours_ago)
        count_similar_message_24h = sum(1 for e in self.error_history 
                                      if message in e["message"] and e["timestamp"] >= twenty_four_hours_ago)
        
        return {
            "same_type": {
                "last_1h": count_same_type_1h,
                "last_6h": count_same_type_6h,
                "last_24h": count_same_type_24h
            },
            "similar_message": {
                "last_1h": count_similar_message_1h,
                "last_6h": count_similar_message_6h,
                "last_24h": count_similar_message_24h
            },
            "total_errors": len(self.error_history),
            "recurring": count_similar_message_1h > 2  # Flag if error is recurring
        }
    
    def _check_for_overheating(self) -> bool:
        """Check if the system is experiencing overheating issues.
        
        Returns:
            True if overheating is detected, False otherwise
        """
        try:
            # Check CPU temperatures if sensors are available
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Examine all temperature sensors
                    for name, entries in temps.items():
                        for entry in entries:
                            # Consider high temperature if above 85°C
                            if entry.current > 85:
                                logger.warning(f"Overheating detected: {name} at {entry.current}°C")
                                return True
            
            # Check GPU temperatures if available
            if "cuda" in self.capabilities.get("hardware_types", []):
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        if gpu.temperature > 85:
                            logger.warning(f"GPU overheating detected: {gpu.name} at {gpu.temperature}°C")
                            return True
                except ImportError:
                    pass
                
            return False
        except Exception as e:
            logger.warning(f"Error checking for overheating: {e}")
            return False
    
    def _check_memory_pressure(self) -> bool:
        """Check if the system is experiencing memory pressure.
        
        Returns:
            True if memory pressure is detected, False otherwise
        """
        try:
            # Check system memory
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning(f"Memory pressure detected: {memory.percent}% used")
                return True
                
            # Check GPU memory if available
            if "cuda" in self.capabilities.get("hardware_types", []):
                try:
                    import torch
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            mem_info = torch.cuda.get_device_properties(i).total_memory
                            used_mem = torch.cuda.memory_allocated(i)
                            usage_percent = (used_mem / mem_info) * 100
                            
                            if usage_percent > 90:
                                logger.warning(f"GPU memory pressure detected: Device {i} at {usage_percent:.1f}% used")
                                return True
                except ImportError:
                    pass
                    
            return False
        except Exception as e:
            logger.warning(f"Error checking memory pressure: {e}")
            return False
    
    def _check_for_throttling(self) -> bool:
        """Check if the system is experiencing thermal throttling.
        
        Returns:
            True if throttling is detected, False otherwise
        """
        try:
            # Check CPU frequencies
            if hasattr(psutil, "cpu_freq"):
                freq = psutil.cpu_freq()
                if freq and hasattr(freq, "current") and hasattr(freq, "max"):
                    # If current frequency is significantly lower than max frequency
                    if freq.current < (freq.max * 0.7):
                        logger.warning(f"CPU throttling detected: {freq.current}MHz vs {freq.max}MHz max")
                        return True
            
            # For GPUs, would need vendor-specific tools to detect throttling accurately
            return False
        except Exception as e:
            logger.warning(f"Error checking for throttling: {e}")
            return False

# Function to integrate with an existing worker client
def integrate_error_reporting(worker):
    """Integrate enhanced error reporting with a worker client.
    
    Args:
        worker: Worker client instance
        
    Returns:
        Worker client with enhanced error reporting
    """
    # Create error reporter instance
    worker.error_reporter = EnhancedErrorReporter(
        worker_id=worker.worker_id,
        capabilities=worker.capabilities
    )
    
    # Store the original _report_task_error method
    original_report_task_error = worker._report_task_error
    
    # Override task error reporting
    async def enhanced_report_task_error(task_id, error):
        """Enhanced task error reporting.
        
        Args:
            task_id: ID of the task
            error: Error message
            
        Returns:
            True if reporting is successful, False otherwise
        """
        # Create enhanced error report
        if isinstance(error, dict):
            enhanced_error = worker.error_reporter.create_enhanced_error_report(
                error_type=error.get("type", "TaskError"),
                message=error.get("message", str(error)),
                task_id=task_id
            )
        else:
            enhanced_error = worker.error_reporter.create_enhanced_error_report(
                error_type="TaskError",
                message=str(error),
                task_id=task_id
            )
        
        # Call original method with enhanced error
        return await original_report_task_error(task_id, enhanced_error)
    
    # Replace the method
    worker._report_task_error = enhanced_report_task_error
    
    # Add method to create enhanced error reports for other scenarios
    worker.create_enhanced_error_report = lambda error_type, message, task_id=None: \
        worker.error_reporter.create_enhanced_error_report(error_type, message, task_id)
    
    logger.info(f"Enhanced error reporting integrated with worker {worker.worker_id}")
    
    return worker