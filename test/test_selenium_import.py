#\!/usr/bin/env python3
import sys
print(f"Python version: {sys.version}")

try:
    import websocket
    print(f"websocket imported, dir: {dir(websocket)}")
except ImportError as e:
    print(f"Error importing websocket: {e}")

try:
    import websocket_client
    print(f"websocket_client imported")
except ImportError as e:
    print(f"Error importing websocket_client: {e}")

try:
    from selenium import webdriver
    print(f"selenium imported successfully")
except ImportError as e:
    print(f"Error importing selenium webdriver: {e}")
