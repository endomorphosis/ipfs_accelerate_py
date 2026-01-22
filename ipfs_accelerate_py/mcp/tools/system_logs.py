"""
MCP tool for accessing system logs.
"""

import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def register_tools(mcp_server):
    """Register system logs tools with the MCP server."""
    
    @mcp_server.tool()
    def get_system_logs(
        lines: int = 100,
        since: Optional[str] = None,
        level: Optional[str] = None,
        service: str = "ipfs-accelerate"
    ) -> Dict[str, Any]:
        """
        Get system logs from the IPFS Accelerate service.
        
        Args:
            lines: Number of log lines to retrieve (default: 100)
            since: Time period to get logs from (e.g., "1 hour ago", "30 minutes ago")
            level: Filter by log level (INFO, WARNING, ERROR, CRITICAL, DEBUG)
            service: Service name to get logs from (default: ipfs-accelerate)
            
        Returns:
            Dictionary with logs array and metadata
        """
        from ipfs_accelerate_py.logs import get_system_logs as get_logs
        
        try:
            logs = get_logs(
                service=service,
                lines=lines,
                since=since,
                level=level
            )
            
            return {
                "success": True,
                "logs": logs,
                "total": len(logs),
                "service": service,
                "filters": {
                    "lines": lines,
                    "since": since,
                    "level": level
                }
            }
        except Exception as e:
            logger.error(f"Failed to get system logs: {e}")
            return {
                "success": False,
                "error": str(e),
                "logs": [],
                "total": 0
            }
    
    @mcp_server.tool()
    def get_recent_errors(hours: int = 24, service: str = "ipfs-accelerate") -> Dict[str, Any]:
        """
        Get recent error logs from the IPFS Accelerate service.
        
        Args:
            hours: Number of hours to look back (default: 24)
            service: Service name to get logs from (default: ipfs-accelerate)
            
        Returns:
            Dictionary with error logs and metadata
        """
        from ipfs_accelerate_py.logs import SystemLogs
        
        try:
            logs_manager = SystemLogs(service)
            errors = logs_manager.get_recent_errors(hours=hours)
            
            return {
                "success": True,
                "errors": errors,
                "total": len(errors),
                "service": service,
                "time_period": f"Last {hours} hours"
            }
        except Exception as e:
            logger.error(f"Failed to get recent errors: {e}")
            return {
                "success": False,
                "error": str(e),
                "errors": [],
                "total": 0
            }
    
    @mcp_server.tool()
    def get_log_stats(service: str = "ipfs-accelerate") -> Dict[str, Any]:
        """
        Get statistics about system logs.
        
        Args:
            service: Service name to get stats for (default: ipfs-accelerate)
            
        Returns:
            Dictionary with log statistics (counts by level, etc.)
        """
        from ipfs_accelerate_py.logs import SystemLogs
        
        try:
            logs_manager = SystemLogs(service)
            stats = logs_manager.get_stats()
            
            return {
                "success": True,
                "stats": stats,
                "service": service
            }
        except Exception as e:
            logger.error(f"Failed to get log stats: {e}")
            return {
                "success": False,
                "error": str(e),
                "stats": {}
            }
    
    logger.info("System logs MCP tools registered")
