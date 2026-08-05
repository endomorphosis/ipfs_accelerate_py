#!/usr/bin/env python3
"""
Kitchen Sink AI Testing Interface Demo

This script demonstrates the complete kitchen sink testing interface
that allows testing different AI model inference types with model selection.
"""

import sys
import os
import time
import webbrowser
import subprocess
from threading import Thread

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))


def start_kitchen_sink_server():
    """Start the kitchen sink Flask server."""
    print("🚀 Starting Kitchen Sink AI Testing Interface...")

    # Import and create the app
    from kitchen_sink_app import create_app

    try:
        kitchen_sink = create_app()
        print("✅ Kitchen Sink App created successfully")

        # Run the server
        print("🌐 Starting web server on http://127.0.0.1:8090")
        kitchen_sink.run(host="127.0.0.1", port=8090, debug=False)

    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

    return True


def open_browser():
    """Open the browser to the kitchen sink interface."""
    time.sleep(2)  # Wait for server to start
    try:
        webbrowser.open("http://127.0.0.1:8090")
        print("🌐 Opened browser to Kitchen Sink interface")
    except:
        print("⚠️ Could not open browser automatically")
        print("📝 Please open http://127.0.0.1:8090 in your browser")


def demo_features():
    """Print demo features and instructions."""
    print("\n" + "=" * 60)
    print("🧪 KITCHEN SINK AI MODEL TESTING INTERFACE")
    print("=" * 60)
    print()

    print("🎯 FEATURES:")
    print("• Multi-tab interface for different AI inference types")
    print("• Model selection with autocomplete from model manager")
    print("• Real-time inference testing and results display")
    print("• Performance feedback collection for bandit learning")
    print("• Model recommendation based on task requirements")
    print()

    print("📋 AVAILABLE TABS:")
    print("1. 📝 Text Generation (Causal LM)")
    print("   - GPT-style text generation")
    print("   - Temperature and length controls")
    print("   - Hardware selection")
    print()

    print("2. 🏷️ Text Classification")
    print("   - Sentiment analysis and classification")
    print("   - Confidence scores and class probabilities")
    print("   - Visual score bars")
    print()

    print("3. 🔢 Text Embeddings")
    print("   - Vector representations of text")
    print("   - Dimensionality display")
    print("   - Normalization options")
    print()

    print("4. 🎯 Model Recommendations")
    print("   - AI-powered model selection")
    print("   - Task-specific recommendations")
    print("   - Confidence scoring")
    print()

    print("5. 📊 Model Manager")
    print("   - Browse available models")
    print("   - Search and filter capabilities")
    print("   - Model metadata display")
    print()

    print("🔧 HOW TO USE:")
    print("1. Select a tab for the type of inference you want to test")
    print("2. Choose a model using the autocomplete field (or leave empty for auto-selection)")
    print("3. Enter your input text and adjust parameters")
    print("4. Click the action button to run inference")
    print("5. View results and provide feedback to improve recommendations")
    print()

    print("💡 TIPS:")
    print("• Leave model field empty to use automatic model selection")
    print("• Use the feedback buttons to help improve future recommendations")
    print("• Check the Model Manager tab to see all available models")
    print("• Model autocomplete searches by name, ID, and architecture")
    print()


def main():
    """Main demo function."""
    print("🎬 Kitchen Sink AI Testing Interface Demo")
    print("=========================================")

    # Show demo features
    demo_features()

    # Start the browser in a separate thread
    browser_thread = Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Start the server (this will block)
    try:
        start_kitchen_sink_server()
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

    print("✅ Demo completed")


if __name__ == "__main__":
    main()
