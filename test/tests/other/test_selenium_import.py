#\!/usr/bin/env python3
import sys
print(f"\1{sys.version}\3")

try:
    import websocket
    print(f"\1{dir(websocket)}\3")
except ImportError as e:
    print(f"\1{e}\3")

try:
    import websocket_client
    print(f"websocket_client imported")
except ImportError as e:
    print(f"\1{e}\3")

try:
    from selenium import webdriver
    print(f"selenium imported successfully")
except ImportError as e:
    print(f"\1{e}\3")