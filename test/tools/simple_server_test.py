#!/usr/bin/env python3
"""
Simple Kitchen Sink Server Test
"""

import sys
import os
import time
import threading
import subprocess

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def start_server():
    """Start the Kitchen Sink server in a separate process."""
    try:
        from kitchen_sink_app import create_app
        kitchen_sink = create_app()
        app = kitchen_sink.app  # Get the actual Flask app
        print("✅ Flask app created successfully")
        print("🚀 Starting server on http://127.0.0.1:8080")
        app.run(host='127.0.0.1', port=8080, debug=False, threaded=True)
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Kitchen Sink Server Test...")
    start_server()