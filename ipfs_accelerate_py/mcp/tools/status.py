"""
Status Monitoring Tools for IPFS Accelerate MCP Server

This module provides MCP tools for monitoring server status and performance.
"""

import os
import time
import uuid
import logging
import platform
import psutil
from typing import Dict, List, Any, Optional

logger = logging.getLogger("ipfs_accelerate_mcp.tools.status")

# Server start time
SERVER_START_TIME = time.time()

# In-memory storage for monitoring sessions (in a real implementation, this would be persistent)
MONITORING_SESSIONS = {}

def register_tools(mcp):
    """Register status-related tools with the MCP server"""
    
    @mcp.tool()
    def get_server_status() -> Dict[str, Any]:
        """
        Get server status
        
        This tool returns the current status of the MCP server.
        
        Returns:
            Dictionary with server status
        """
        try:
            # Calculate uptime
            uptime_seconds = time.time() - SERVER_START_TIME
            
            # Get server information
            server_config = mcp.access_resource("server_config")
            
            return {
                "status": "running",
                "version": "0.1.0",  # This would be dynamic in a real implementation
                "uptime_seconds": uptime_seconds,
                "host": server_config.get("host", "localhost"),
                "port": server_config.get("port", 8080),
                "system": platform.system(),
                "python_version": platform.python_version()
            }
        except Exception as e:
            return {
                "error": f"Error getting server status: {str(e)}"
            }
    
    @mcp.tool()
    def get_performance_metrics() -> Dict[str, Any]:
        """
        Get performance metrics
        
        This tool returns the current performance metrics of the system.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024 ** 3)
            memory_total_gb = memory.total / (1024 ** 3)
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024 ** 3)
            disk_total_gb = disk.total / (1024 ** 3)
            
            # Get network info (simplified)
            network_stats = psutil.net_io_counters()
            
            return {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_gb": memory_used_gb,
                "memory_total_gb": memory_total_gb,
                "disk_percent": disk_percent,
                "disk_used_gb": disk_used_gb,
                "disk_total_gb": disk_total_gb,
                "network_bytes_sent": network_stats.bytes_sent,
                "network_bytes_recv": network_stats.bytes_recv
            }
        except Exception as e:
            return {
                "error": f"Error getting performance metrics: {str(e)}"
            }
    
    @mcp.tool()
    def start_session(session_name: str = "") -> Dict[str, Any]:
        """
        Start a monitoring session
        
        This tool starts a new monitoring session to track operations.
        
        Args:
            session_name: Optional name for the session
            
        Returns:
            Dictionary with session information
        """
        try:
            # Generate a unique ID for this session
            session_id = str(uuid.uuid4())
            
            # Create the session
            session = {
                "id": session_id,
                "name": session_name if session_name else f"Session-{session_id[:8]}",
                "start_time": time.time(),
                "operations": 0,
                "total_processing_time": 0.0,
                "metrics": []  # Will store periodic metrics
            }
            
            # Store initial metrics
            session["metrics"].append(get_performance_metrics())
            
            # Store the session
            MONITORING_SESSIONS[session_id] = session
            
            logger.info(f"Started monitoring session {session_id}: {session_name}")
            
            return {
                "id": session_id,
                "name": session["name"],
                "start_time": session["start_time"]
            }
        except Exception as e:
            return {
                "error": f"Error starting session: {str(e)}"
            }
    
    @mcp.tool()
    def end_session(session_id: str) -> Dict[str, Any]:
        """
        End a monitoring session
        
        This tool ends a monitoring session and returns a summary.
        
        Args:
            session_id: ID of the session to end
            
        Returns:
            Dictionary with session summary
        """
        try:
            # Check if the session exists
            if session_id not in MONITORING_SESSIONS:
                return {
                    "error": f"Session '{session_id}' not found."
                }
            
            # Get the session
            session = MONITORING_SESSIONS[session_id]
            
            # Calculate session duration
            end_time = time.time()
            duration_seconds = end_time - session["start_time"]
            
            # Calculate average metrics over the session
            avg_cpu_percent = sum(m.get("cpu_percent", 0) for m in session["metrics"]) / len(session["metrics"]) if session["metrics"] else 0
            avg_memory_percent = sum(m.get("memory_percent", 0) for m in session["metrics"]) / len(session["metrics"]) if session["metrics"] else 0
            
            # Add final metrics
            session["metrics"].append(get_performance_metrics())
            
            # Prepare session summary
            summary = {
                "id": session_id,
                "name": session["name"],
                "start_time": session["start_time"],
                "end_time": end_time,
                "duration_seconds": duration_seconds,
                "operations": session["operations"],
                "avg_processing_time": session["total_processing_time"] / session["operations"] if session["operations"] > 0 else 0,
                "avg_cpu_percent": avg_cpu_percent,
                "avg_memory_percent": avg_memory_percent
            }
            
            # Update the session
            session["end_time"] = end_time
            session["summary"] = summary
            
            logger.info(f"Ended monitoring session {session_id}: {duration_seconds:.2f}s, {session['operations']} operations")
            
            # In a real implementation, we might persist the session data
            # For this mock, we'll just return the summary and keep the session in memory
            
            return summary
        except Exception as e:
            return {
                "error": f"Error ending session: {str(e)}"
            }
    
    @mcp.tool()
    def log_operation(session_id: str, 
                     operation_type: str,
                     processing_time: float,
                     details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Log an operation to a monitoring session
        
        This tool logs an operation to a monitoring session.
        
        Args:
            session_id: ID of the session
            operation_type: Type of operation (e.g., "inference", "endpoint_creation")
            processing_time: Time taken to process the operation
            details: Optional details about the operation
            
        Returns:
            Dictionary with the log status
        """
        try:
            # Check if the session exists
            if session_id not in MONITORING_SESSIONS:
                return {
                    "error": f"Session '{session_id}' not found."
                }
            
            # Get the session
            session = MONITORING_SESSIONS[session_id]
            
            # Update operation statistics
            session["operations"] += 1
            session["total_processing_time"] += processing_time
            
            # Add current metrics if we haven't added any in a while
            current_time = time.time()
            if not session["metrics"] or (current_time - session["metrics"][-1]["timestamp"]) > 60:
                session["metrics"].append(get_performance_metrics())
            
            # Calculate average processing time
            avg_processing_time = session["total_processing_time"] / session["operations"]
            
            logger.info(f"Logged operation to session {session_id}: {operation_type} in {processing_time:.3f}s")
            
            return {
                "status": "success",
                "session_id": session_id,
                "operation_type": operation_type,
                "processing_time": processing_time,
                "session_stats": {
                    "operations": session["operations"],
                    "total_processing_time": session["total_processing_time"],
                    "avg_processing_time": avg_processing_time
                }
            }
        except Exception as e:
            return {
                "error": f"Error logging operation: {str(e)}"
            }
    
    @mcp.tool()
    def get_session(session_id: str) -> Dict[str, Any]:
        """
        Get a monitoring session
        
        This tool gets information about a monitoring session.
        
        Args:
            session_id: ID of the session to get
            
        Returns:
            Dictionary with the session information
        """
        try:
            # Check if the session exists
            if session_id not in MONITORING_SESSIONS:
                return {
                    "error": f"Session '{session_id}' not found."
                }
            
            # Return the session without the full metrics history (which could be large)
            session = MONITORING_SESSIONS[session_id]
            session_info = {k: v for k, v in session.items() if k != "metrics"}
            session_info["latest_metrics"] = session["metrics"][-1] if session["metrics"] else None
            session_info["metrics_count"] = len(session["metrics"])
            
            return session_info
        except Exception as e:
            return {
                "error": f"Error getting session: {str(e)}"
            }
