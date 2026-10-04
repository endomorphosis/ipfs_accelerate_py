#!/usr/bin/env python3
"""
Test script for MCP Dashboard with SDK improvements

This script starts the MCP dashboard and opens it in a browser to demonstrate
the JavaScript SDK integration.
"""

import os
import sys
import time
import webbrowser
from threading import Thread

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def start_dashboard():
    """Start the MCP dashboard server."""
    try:
        from ipfs_accelerate_py.mcp_dashboard import MCPDashboard
        
        print("🚀 Starting MCP Dashboard with SDK Integration...")
        print("=" * 60)
        
        # Create dashboard instance with unified registry
        dashboard = MCPDashboard(
            port=8899,
            host='127.0.0.1',
            use_unified_registry=True
        )
        
        print("✅ Dashboard initialized")
        print("📍 URL: http://127.0.0.1:8899")
        print("🎮 SDK Playground: http://127.0.0.1:8899 (click 'SDK Playground' tab)")
        print("🔧 MCP Tools: http://127.0.0.1:8899 (click 'MCP Tools' tab)")
        print()
        print("Features to test:")
        print("  1. SDK Playground - Interactive SDK examples")
        print("  2. SDK Statistics - Real-time call tracking")
        print("  3. Tool Execution - All tools use SDK")
        print("  4. Batch Operations - Parallel execution demo")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Open browser after a delay
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://127.0.0.1:8899')
        
        Thread(target=open_browser, daemon=True).start()
        
        # Start the dashboard
        dashboard.run()
        
    except ImportError as e:
        print(f"❌ Error: Failed to import dashboard: {e}")
        print("Make sure Flask and Flask-CORS are installed:")
        print("  pip install flask flask-cors")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        start_dashboard()
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped")
        sys.exit(0)
