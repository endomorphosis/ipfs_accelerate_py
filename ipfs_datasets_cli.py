#!/usr/bin/env python3
"""
IPFS Datasets CLI - Enhanced MCP Dashboard for Dataset Processing

This module provides a comprehensive CLI for dataset processing with an enhanced
MCP dashboard that supports IPFS, Hugging Face, Parquet, and CAR files.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import signal
import subprocess
import time
import webbrowser
from typing import Dict, Any, List, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ipfs_datasets_cli")

class IPFSDatasetsCLI:
    """Enhanced CLI for dataset processing with MCP dashboard."""
    
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = 8899
        self.log_file = Path.home() / '.ipfs_datasets' / 'mcp_dashboard.log'
        self.log_file.parent.mkdir(exist_ok=True)
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal, cleaning up...")
            self.cleanup()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up...")
        
    def run_mcp_start(self, args):
        """Start enhanced MCP server dashboard."""
        logger.info("Starting IPFS Datasets MCP Dashboard...")
        
        try:
            if args.blocking:
                # Run in blocking mode for debugging
                return self._start_dashboard_blocking(args)
            else:
                # Run in background with readiness check
                return self._start_dashboard_background(args)
                
        except KeyboardInterrupt:
            logger.info("MCP dashboard stopped by user")
            return 0
        except Exception as e:
            logger.error(f"Error starting MCP dashboard: {e}")
            return 1
            
    def _start_dashboard_blocking(self, args):
        """Start dashboard in blocking mode."""
        from ipfs_accelerate_py.enhanced_mcp_dashboard import EnhancedMCPDashboard
        
        dashboard = EnhancedMCPDashboard(
            host=args.host or self.host,
            port=args.port or self.port
        )
        
        try:
            dashboard.run(debug=args.debug)
        except KeyboardInterrupt:
            logger.info("Dashboard stopped by user")
            
        return 0
        
    def _start_dashboard_background(self, args):
        """Start dashboard in background with readiness check."""
        import subprocess
        import time
        import requests
        
        host = args.host or self.host
        port = args.port or self.port
        
        # Create command to run dashboard
        cmd = [
            sys.executable, '-c',
            f"""
from ipfs_accelerate_py.enhanced_mcp_dashboard import EnhancedMCPDashboard
dashboard = EnhancedMCPDashboard(host='{host}', port={port})
dashboard.run(debug={args.debug})
"""
        ]
        
        # Start process with logging
        with open(self.log_file, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent
            )
            
        # Wait briefly for startup
        time.sleep(2)
        
        # Check readiness
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = requests.get(f'http://{host}:{port}/api/mcp/status', timeout=2)
                if response.status_code == 200:
                    print(f"✅ MCP Dashboard started successfully!")
                    print(f"📊 Dashboard: http://{host}:{port}/mcp")
                    print(f"🔌 API Status: http://{host}:{port}/api/mcp/status")
                    return 0
            except requests.RequestException:
                pass
                
            time.sleep(1)
            
        # If readiness check failed
        if process.poll() is not None:
            print(f"❌ Dashboard process exited immediately (code: {process.returncode})")
        else:
            print(f"⚠️  Dashboard may still be starting...")
            
        print(f"📝 Check logs: ipfs-datasets mcp logs")
        print(f"🔧 Or try: ipfs-datasets mcp start --blocking")
        
        return 0
        
    def run_mcp_status(self, args):
        """Check MCP dashboard status."""
        import requests
        
        try:
            response = requests.get(f'http://{self.host}:{self.port}/api/mcp/status', timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                print("✅ MCP Dashboard is running")
                print(f"📊 Dashboard: http://{self.host}:{self.port}/mcp")
                
                if args.json:
                    print(json.dumps(status_data, indent=2))
                else:
                    print(f"🔧 Tools available: {status_data.get('tools_available', 'Unknown')}")
                    print(f"📈 Status: {status_data.get('status', 'Unknown')}")
                    
                return 0
            else:
                print(f"❌ Dashboard returned status {response.status_code}")
                return 1
                
        except requests.RequestException as e:
            print("❌ MCP Dashboard is not running or not accessible")
            print(f"🔧 Start with: ipfs-datasets mcp start")
            return 1
            
    def run_mcp_logs(self, args):
        """View dashboard logs."""
        if not self.log_file.exists():
            print(f"❌ Log file not found: {self.log_file}")
            print("🔧 Start dashboard first: ipfs-datasets mcp start")
            return 1
            
        try:
            if args.follow:
                # Use tail -f equivalent
                subprocess.run(['tail', f'-n{args.lines}', '-f', str(self.log_file)])
            else:
                # Show last N lines
                subprocess.run(['tail', f'-n{args.lines}', str(self.log_file)])
                
        except KeyboardInterrupt:
            pass
        except FileNotFoundError:
            # Fallback to Python implementation
            self._tail_logs(args.lines, args.follow)
            
        return 0
        
    def _tail_logs(self, lines, follow):
        """Python implementation of log tailing."""
        try:
            with open(self.log_file, 'r') as f:
                # Get last N lines
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
                for line in recent_lines:
                    print(line.rstrip())
                    
                if follow:
                    # Simple follow mode
                    f.seek(0, 2)  # Go to end
                    while True:
                        line = f.readline()
                        if line:
                            print(line.rstrip())
                        else:
                            time.sleep(0.1)
                            
        except KeyboardInterrupt:
            pass

def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="IPFS Datasets CLI - Enhanced MCP Dashboard for Dataset Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ipfs-datasets mcp start                    # Start dashboard in background
  ipfs-datasets mcp start --blocking         # Start in foreground for debugging
  ipfs-datasets mcp status                   # Check if dashboard is running
  ipfs-datasets mcp logs --follow            # Tail dashboard logs
        """
    )
    
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # MCP command group
    mcp_parser = subparsers.add_parser('mcp', help='MCP dashboard management')
    mcp_subparsers = mcp_parser.add_subparsers(dest='mcp_command', help='MCP commands')
    
    # MCP start command
    start_parser = mcp_subparsers.add_parser('start', help='Start MCP dashboard')
    start_parser.add_argument('--host', help='Host to bind to (default: 127.0.0.1)')
    start_parser.add_argument('--port', type=int, help='Port to bind to (default: 8899)')
    start_parser.add_argument('--blocking', action='store_true', help='Run in foreground for debugging')
    start_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # MCP status command
    status_parser = mcp_subparsers.add_parser('status', help='Check dashboard status')
    
    # MCP logs command
    logs_parser = mcp_subparsers.add_parser('logs', help='View dashboard logs')
    logs_parser.add_argument('--lines', type=int, default=50, help='Number of lines to show (default: 50)')
    logs_parser.add_argument('--follow', action='store_true', help='Follow logs in real-time')
    
    return parser

def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    cli = IPFSDatasetsCLI()
    cli.setup_signal_handlers()
    
    try:
        if args.command == 'mcp':
            if args.mcp_command == 'start':
                return cli.run_mcp_start(args)
            elif args.mcp_command == 'status':
                return cli.run_mcp_status(args)
            elif args.mcp_command == 'logs':
                return cli.run_mcp_logs(args)
            else:
                parser.print_help()
                return 1
        else:
            parser.print_help()
            return 1
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())