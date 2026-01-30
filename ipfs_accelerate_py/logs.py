"""
System logs access module for IPFS Accelerate.

This module provides access to system logs through multiple interfaces:
- Python package import
- CLI tool
- MCP server tool
"""

import os
import sys
import json
import subprocess
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

try:
    from .common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from test.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        HAVE_STORAGE_WRAPPER = False

logger = logging.getLogger(__name__)


class SystemLogs:
    """Access and manage system logs for IPFS Accelerate services."""
    
    def __init__(self, service_name: str = "ipfs-accelerate"):
        """
        Initialize system logs access.
        
        Args:
            service_name: Name of the systemd service to query logs for
        """
        self.service_name = service_name
        
        # Initialize storage wrapper
        if HAVE_STORAGE_WRAPPER:
            try:
                self._storage = get_storage_wrapper(auto_detect_ci=True)
            except Exception:
                self._storage = None
        else:
            self._storage = None
    
    def get_logs(
        self,
        lines: int = 100,
        since: Optional[str] = None,
        level: Optional[str] = None,
        follow: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get system logs.
        
        Args:
            lines: Number of lines to retrieve (default: 100)
            since: Time period (e.g., "1 hour ago", "30 minutes ago")
            level: Filter by log level (INFO, WARNING, ERROR, etc.)
            follow: Whether to follow logs in real-time (for CLI mode)
            
        Returns:
            List of log entries with timestamp, level, and message
        """
        # Ensure level is a string or None, handle unexpected types
        if level is not None and not isinstance(level, str):
            logger.warning(f"Invalid level type: {type(level)}, converting to string")
            if isinstance(level, (list, tuple)) and len(level) > 0:
                level = str(level[0])
            else:
                level = str(level)
        
        try:
            # Build journalctl command
            cmd = ["journalctl", "-u", self.service_name, "--no-pager"]
            
            if since:
                cmd.extend(["--since", since])
            
            if lines and not follow:
                cmd.extend(["-n", str(lines)])
            
            if follow:
                cmd.append("-f")
            
            # Add JSON output for structured parsing
            if not follow:
                cmd.append("--output=json")
            
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=not follow,
                text=True,
                check=True
            )
            
            if follow:
                # In follow mode, output goes to stdout directly
                return []
            
            # Parse JSON output
            logs = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    message = entry.get('MESSAGE', '')
                    # Handle case where MESSAGE might be a list or other type
                    if isinstance(message, list):
                        message = ' '.join(str(m) for m in message)
                    elif not isinstance(message, str):
                        message = str(message)
                    
                    log_entry = {
                        'timestamp': self._format_timestamp(entry.get('__REALTIME_TIMESTAMP')),
                        'level': self._extract_level(message),
                        'message': message,
                        'unit': entry.get('_SYSTEMD_UNIT', ''),
                        'pid': entry.get('_PID', '')
                    }
                    
                    # Filter by level if specified
                    if level and isinstance(level, str) and log_entry['level'] != level.upper():
                        continue
                    
                    logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
            
            return logs
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get logs: {e}")
            return []
        except FileNotFoundError:
            logger.error("journalctl not found - systemd not available")
            return self._get_fallback_logs(lines)
    
    def get_recent_errors(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent error logs.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of error log entries
        """
        since = f"{hours} hours ago"
        all_logs = self.get_logs(lines=1000, since=since)
        return [log for log in all_logs if log['level'] in ['ERROR', 'CRITICAL']]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get log statistics.
        
        Returns:
            Dictionary with log counts by level
        """
        logs = self.get_logs(lines=1000, since="1 hour ago")
        
        stats = {
            'total': len(logs),
            'by_level': {
                'INFO': 0,
                'WARNING': 0,
                'ERROR': 0,
                'CRITICAL': 0,
                'DEBUG': 0
            },
            'time_range': '1 hour'
        }
        
        for log in logs:
            level = log.get('level', 'INFO')
            if level in stats['by_level']:
                stats['by_level'][level] += 1
        
        return stats
    
    def _format_timestamp(self, timestamp_micro: Optional[str]) -> str:
        """Format microsecond timestamp to readable string."""
        if not timestamp_micro:
            return datetime.now().isoformat()
        
        try:
            timestamp_sec = int(timestamp_micro) / 1000000
            dt = datetime.fromtimestamp(timestamp_sec)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            return datetime.now().isoformat()
    
    def _extract_level(self, message) -> str:
        """Extract log level from message."""
        # Handle case where message might be a list
        if isinstance(message, list):
            message = ' '.join(str(m) for m in message)
        elif not isinstance(message, str):
            message = str(message)
        
        message_upper = message.upper()
        for level in ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']:
            if level in message_upper:
                return level
        return 'INFO'
    
    def _get_fallback_logs(self, lines: int) -> List[Dict[str, Any]]:
        """Get fallback logs when journalctl is not available."""
        # Try to read from log files
        log_paths = [
            '/var/log/ipfs-accelerate.log',
            os.path.expanduser('~/.ipfs-accelerate/logs/service.log'),
            '/tmp/ipfs-accelerate.log'
        ]
        
        for log_path in log_paths:
            if os.path.exists(log_path):
                try:
                    # Try distributed storage first
                    if self._storage and self._storage.is_distributed:
                        try:
                            cached_data = self._storage.read_file(log_path)
                            if cached_data:
                                log_lines = cached_data.decode('utf-8').split('\n')[-lines:]
                            else:
                                with open(log_path, 'r') as f:
                                    content = f.read()
                                    log_lines = content.split('\n')[-lines:]
                                # Cache logs (temporary, not pinned)
                                self._storage.write_file(content.encode('utf-8'), log_path, pin=False)
                        except Exception:
                            # Fallback to local filesystem
                            with open(log_path, 'r') as f:
                                log_lines = f.readlines()[-lines:]
                    else:
                        with open(log_path, 'r') as f:
                            log_lines = f.readlines()[-lines:]
                    
                    logs = []
                    for line in log_lines:
                        logs.append({
                            'timestamp': datetime.now().isoformat(),
                            'level': 'INFO',
                            'message': line.strip(),
                            'unit': self.service_name,
                            'pid': ''
                        })
                    return logs
                except Exception as e:
                    logger.debug(f"Failed to read log file {log_path}: {e}")
        
        return [{
            'timestamp': datetime.now().isoformat(),
            'level': 'INFO',
            'message': 'No logs available',
            'unit': self.service_name,
            'pid': ''
        }]


def get_system_logs(
    service: str = "ipfs-accelerate",
    lines: int = 100,
    since: Optional[str] = None,
    level: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to get system logs.
    
    Args:
        service: Service name
        lines: Number of lines
        since: Time period
        level: Log level filter
        
    Returns:
        List of log entries
    """
    logs = SystemLogs(service)
    return logs.get_logs(lines=lines, since=since, level=level)


def main():
    """CLI entry point for system logs."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Access IPFS Accelerate system logs'
    )
    parser.add_argument(
        '--service',
        default='ipfs-accelerate',
        help='Service name (default: ipfs-accelerate)'
    )
    parser.add_argument(
        '--lines', '-n',
        type=int,
        default=100,
        help='Number of lines to display (default: 100)'
    )
    parser.add_argument(
        '--since',
        help='Show logs since time (e.g., "1 hour ago", "30 minutes ago")'
    )
    parser.add_argument(
        '--level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Filter by log level'
    )
    parser.add_argument(
        '--follow', '-f',
        action='store_true',
        help='Follow log output'
    )
    parser.add_argument(
        '--errors',
        action='store_true',
        help='Show only errors from last 24 hours'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show log statistics'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output in JSON format'
    )
    
    args = parser.parse_args()
    
    logs_manager = SystemLogs(args.service)
    
    if args.stats:
        stats = logs_manager.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"\n📊 Log Statistics (last hour):")
            print(f"Total logs: {stats['total']}")
            print(f"\nBy level:")
            for level, count in stats['by_level'].items():
                if count > 0:
                    print(f"  {level}: {count}")
        return
    
    if args.errors:
        logs = logs_manager.get_recent_errors()
        if args.json:
            print(json.dumps(logs, indent=2))
        else:
            print(f"\n🚨 Recent Errors (last 24 hours): {len(logs)} found\n")
            for log in logs:
                print(f"[{log['timestamp']}] {log['level']}: {log['message']}")
        return
    
    logs = logs_manager.get_logs(
        lines=args.lines,
        since=args.since,
        level=args.level,
        follow=args.follow
    )
    
    if args.json:
        print(json.dumps(logs, indent=2))
    else:
        if not args.follow:
            print(f"\n📝 System Logs ({len(logs)} entries):\n")
        for log in logs:
            level_emoji = {
                'ERROR': '❌',
                'CRITICAL': '🔥',
                'WARNING': '⚠️',
                'INFO': 'ℹ️',
                'DEBUG': '🔍'
            }.get(log['level'], '📝')
            print(f"{level_emoji} [{log['timestamp']}] {log['level']}: {log['message']}")


if __name__ == '__main__':
    main()
