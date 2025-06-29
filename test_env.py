#!/usr/bin/env python3
print("Python is working!")
print("Testing imports...")

try:
    import flask
    print("✅ Flask available")
except ImportError as e:
    print(f"❌ Flask not available: {e}")

try:
    import fastapi
    print("✅ FastAPI available")
except ImportError as e:
    print(f"❌ FastAPI not available: {e}")

try:
    import uvicorn
    print("✅ Uvicorn available")
except ImportError as e:
    print(f"❌ Uvicorn not available: {e}")

print("Environment test complete!")
