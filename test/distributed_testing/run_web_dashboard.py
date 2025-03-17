#!/usr/bin/env python3
"""
Run Script for Result Aggregator Web Dashboard

This script runs the Result Aggregator Web Dashboard with default settings.

Usage:
    python run_web_dashboard.py [--port PORT] [--db-path DB_PATH] [--debug] [--update-interval SECONDS]
"""

import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Import the web dashboard module
    from result_aggregator.web_dashboard import main
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Result Aggregator Web Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the web server on")
    parser.add_argument("--db-path", default="./test_results.duckdb", help="Path to DuckDB database")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--enable-ml", action="store_true", default=True, help="Enable machine learning features")
    parser.add_argument("--enable-visualization", action="store_true", default=True, help="Enable visualization features")
    parser.add_argument("--update-interval", type=int, default=5, help="Interval in seconds for WebSocket real-time monitoring updates")
    
    args = parser.parse_args()
    
    # Ensure the database directory exists
    db_dir = os.path.dirname(args.db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Run the web dashboard
    sys.argv = [sys.argv[0]]  # Reset args for the main function
    if args.port:
        sys.argv.extend(["--port", str(args.port)])
    if args.db_path:
        sys.argv.extend(["--db-path", args.db_path])
    if args.debug:
        sys.argv.append("--debug")
    if args.enable_ml:
        sys.argv.append("--enable-ml")
    if args.enable_visualization:
        sys.argv.append("--enable-visualization")
    if args.update_interval:
        sys.argv.extend(["--update-interval", str(args.update_interval)])
    
    print(f"Starting Result Aggregator Web Dashboard")
    print(f"Database path: {args.db_path}")
    print(f"Port: {args.port}")
    print(f"Debug mode: {args.debug}")
    print(f"ML features: {args.enable_ml}")
    print(f"Visualization features: {args.enable_visualization}")
    print(f"WebSocket update interval: {args.update_interval} seconds")
    print("Access the dashboard at http://localhost:{0}".format(args.port))
    print("Use Ctrl+C to stop the server")
    
    main()
    
except ImportError as e:
    print(f"Error importing web dashboard module: {e}")
    print("Make sure you have installed all required dependencies:")
    print("pip install flask flask-cors flask-socketio duckdb numpy pandas matplotlib scikit-learn")
    print("\nNote: flask-socketio is required for WebSocket real-time updates.")
    sys.exit(1)
except Exception as e:
    print(f"Error starting web dashboard: {e}")
    sys.exit(1)