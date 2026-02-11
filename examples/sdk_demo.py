#!/usr/bin/env python3
"""
SDK Dashboard Quick Demo

This script demonstrates the complete JavaScript SDK and JSON-RPC MCP server.
Run this to see the system in action!
"""

import os
import sys
import time
import webbrowser
import threading
from pathlib import Path

def print_banner():
    """Print the demo banner."""
    print("🚀" + "="*70 + "🚀")
    print("🎯 JAVASCRIPT SDK & JSON-RPC MCP SERVER DEMO 🎯")
    print("🚀" + "="*70 + "🚀")
    print()
    print("✅ Features Demonstrated:")
    print("  📱 Complete JavaScript SDK for MCP communication")
    print("  🔗 JSON-RPC 2.0 protocol implementation")  
    print("  🌐 Kitchen Sink dashboard using ONLY the SDK")
    print("  🤖 Browser automation testing framework")
    print("  🎯 28+ AI inference methods available")
    print()

def start_demo():
    """Start the demo."""
    print_banner()
    
    print("🔧 Starting SDK Dashboard with JSON-RPC server...")
    print("⏳ Please wait while servers initialize...")
    print()
    
    # Import and start the SDK dashboard
    try:
        from sdk_dashboard_app import SDKDashboardApp
        
        app = SDKDashboardApp()
        
        print("📊 Dashboard URL: http://localhost:8080")
        print("🔗 JSON-RPC URL: http://localhost:8000")
        print()
        print("🎯 Available Features:")
        print("  • Text Generation (GPT-style)")
        print("  • Text Classification (Sentiment Analysis)")
        print("  • Text Embeddings (Vector Generation)")
        print("  • Audio Processing (Transcription)")
        print("  • Vision Models (Classification & Generation)")
        print("  • Model Recommendations (AI-powered)")
        print("  • Model Manager (Search & Browse)")
        print("  • Specialized Tools (Code Generation)")
        print()
        print("💡 The dashboard uses ONLY the JavaScript SDK")
        print("💡 All communication is via JSON-RPC 2.0")
        print()
        print("🌐 Opening dashboard in browser...")
        
        # Start the application in a thread to allow opening browser
        def run_app():
            app.run()
        
        app_thread = threading.Thread(target=run_app, daemon=True)
        app_thread.start()
        
        # Wait a bit for servers to start
        time.sleep(5)
        
        # Open browser
        try:
            webbrowser.open("http://localhost:8080")
        except Exception as e:
            print(f"⚠️ Could not auto-open browser: {e}")
            print("📌 Please manually open: http://localhost:8080")
        
        print()
        print("🎉 Demo is running! Press Ctrl+C to stop.")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Demo stopped. Thank you!")
            
    except ImportError as e:
        print(f"❌ Failed to import SDK dashboard: {e}")
        print("📦 Please make sure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_demo()