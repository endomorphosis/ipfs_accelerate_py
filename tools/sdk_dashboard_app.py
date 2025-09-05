#!/usr/bin/env python3
"""
SDK Dashboard Application

A Flask application that serves the Kitchen Sink AI Testing Interface
using only the JavaScript SDK with JSON-RPC communication.
"""

import os
import sys
import threading
import time
from pathlib import Path

from flask import Flask, render_template, send_from_directory
from flask_cors import CORS

# Import our JSON-RPC server
from mcp_jsonrpc_server import MCPJSONRPCServer
import uvicorn

class SDKDashboardApp:
    """SDK-based Dashboard Application."""
    
    def __init__(self, dashboard_port=8080, jsonrpc_port=8000):
        """Initialize the SDK Dashboard App."""
        self.dashboard_port = dashboard_port
        self.jsonrpc_port = jsonrpc_port
        
        # Create Flask app for dashboard
        self.dashboard_app = Flask(__name__)
        CORS(self.dashboard_app)
        self.dashboard_app.config['SECRET_KEY'] = 'sdk-dashboard-2025'
        
        # Setup dashboard routes
        self._setup_dashboard_routes()
        
        # Create JSON-RPC server
        self.jsonrpc_server = MCPJSONRPCServer()
        
        print(f"✅ SDK Dashboard initialized")
        print(f"📊 Dashboard will run on http://localhost:{dashboard_port}")
        print(f"🔗 JSON-RPC server will run on http://localhost:{jsonrpc_port}")
    
    def _setup_dashboard_routes(self):
        """Setup routes for the dashboard."""
        
        @self.dashboard_app.route('/')
        def dashboard():
            """Serve the SDK dashboard."""
            return render_template('sdk_dashboard.html')
        
        @self.dashboard_app.route('/static/<path:filename>')
        def static_files(filename):
            """Serve static files."""
            return send_from_directory('static', filename)
        
        @self.dashboard_app.route('/health')
        def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "dashboard_port": self.dashboard_port,
                "jsonrpc_port": self.jsonrpc_port,
                "endpoints": {
                    "dashboard": f"http://localhost:{self.dashboard_port}",
                    "jsonrpc": f"http://localhost:{self.jsonrpc_port}/jsonrpc"
                }
            }
    
    def run_jsonrpc_server(self):
        """Run the JSON-RPC server in a separate thread."""
        print(f"🚀 Starting JSON-RPC server on port {self.jsonrpc_port}...")
        uvicorn.run(
            self.jsonrpc_server.app,
            host="127.0.0.1",
            port=self.jsonrpc_port,
            log_level="info"
        )
    
    def run_dashboard(self):
        """Run the dashboard server."""
        print(f"🚀 Starting Dashboard server on port {self.dashboard_port}...")
        self.dashboard_app.run(
            host="127.0.0.1",
            port=self.dashboard_port,
            debug=False,
            use_reloader=False
        )
    
    def run(self):
        """Run both servers."""
        print("🔧 Starting SDK Dashboard Application...")
        
        # Start JSON-RPC server in background thread
        jsonrpc_thread = threading.Thread(
            target=self.run_jsonrpc_server,
            daemon=True
        )
        jsonrpc_thread.start()
        
        # Give JSON-RPC server time to start
        print("⏳ Waiting for JSON-RPC server to start...")
        time.sleep(3)
        
        # Start dashboard server (blocking)
        print("✅ JSON-RPC server started, now starting dashboard...")
        self.run_dashboard()

def create_app():
    """Create and return the dashboard Flask app."""
    app = SDKDashboardApp()
    return app.dashboard_app

if __name__ == "__main__":
    app = SDKDashboardApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Shutting down SDK Dashboard Application...")
    except Exception as e:
        print(f"❌ Error running SDK Dashboard: {e}")
        sys.exit(1)