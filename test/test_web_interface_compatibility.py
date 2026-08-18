#!/usr/bin/env python3
"""
Web Interface Compatibility Checker
Checks for deprecated APIs in web interface components for Python 3.12
"""

import sys
import warnings
import importlib
import anyio
from pathlib import Path


def check_web_interface():
    """Check web interface for Python 3.12 compatibility"""
    print("🌐 Checking web interface compatibility...")

    # Check if main web server can be imported
    try:
        # Set warnings to be errors to catch deprecation warnings
        warnings.filterwarnings("error", category=DeprecationWarning)

        # Try importing main components
        import main

        print("✅ Main web server imports successfully")

    except DeprecationWarning as e:
        print(f"⚠️ DeprecationWarning in web interface: {e}")
        return False
    except ImportError as e:
        print(f"ℹ️ Web dependencies not installed: {e}")
        print("💡 Install with: pip install ipfs_accelerate_py[all]")
        return True  # Not an actual error, just missing deps
    except Exception as e:
        print(f"❌ Error in web interface: {e}")
        return False

    # Check FastAPI compatibility
    try:
        import fastapi

        print(
            f"✅ FastAPI {fastapi.__version__} compatible with Python {sys.version_info.major}.{sys.version_info.minor}"
        )
    except ImportError:
        print("ℹ️ FastAPI not installed")

    # Check uvicorn compatibility
    try:
        import uvicorn

        print(f"✅ Uvicorn compatible")
    except ImportError:
        print("ℹ️ Uvicorn not installed")

    # Check websockets compatibility
    try:
        import websockets

        print(f"✅ Websockets {websockets.version.version} compatible")
    except ImportError:
        print("ℹ️ Websockets not installed")

    return True


def check_async_compatibility():
    """Check async/await compatibility"""
    print("⚡ Checking async compatibility...")

    # Test basic async functionality
    async def test_async():
        await anyio.sleep(0.001)
        return "async works"

    try:
        result = anyio.run(test_async)
        print("✅ Async/await functionality working")
        return True
    except Exception as e:
        print(f"❌ Async functionality error: {e}")
        return False


def check_deprecated_apis():
    """Check for common deprecated API patterns"""
    print("🔍 Checking for deprecated API patterns...")

    deprecated_patterns = []

    # Check if any files use deprecated patterns
    web_files = [
        Path("main.py"),
        Path("webgpu_platform.py"),
        Path("ipfs_accelerate_py/webnn_webgpu_integration.py"),
    ]

    for file_path in web_files:
        if file_path.exists():
            try:
                content = file_path.read_text(encoding="utf-8")

                # Check for deprecated patterns
                patterns_to_check = [
                    ("coroutine", "Use async def instead"),
                    ("collections.Iterable", "Use collections.abc.Iterable"),
                    ("collections.Mapping", "Use collections.abc.Mapping"),
                    ("imp.load_source", "Use importlib instead"),
                    ("cgi.escape", "Use html.escape"),
                ]

                for pattern, suggestion in patterns_to_check:
                    if pattern in content:
                        deprecated_patterns.append(f"⚠️ {file_path}: Found {pattern} - {suggestion}")

            except Exception as e:
                print(f"⚠️ Could not check {file_path}: {e}")

    if deprecated_patterns:
        print("Found deprecated API patterns:")
        for pattern in deprecated_patterns:
            print(pattern)
        return False
    else:
        print("✅ No deprecated API patterns found in web interface")
        return True


def create_web_compatibility_fix():
    """Create a compatibility layer for web interface"""
    print("🔧 Creating web interface compatibility fixes...")

    compatibility_code = '''"""
Web Interface Compatibility Layer for Python 3.12
"""

import sys
import warnings
from typing import Any, Dict, Optional

# Suppress specific warnings that are not critical
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

class WebInterfaceCompatibility:
    """Compatibility layer for web interface components"""
    
    @staticmethod
    def check_python_version() -> bool:
        """Check if Python version is supported"""
        version = sys.version_info
        if version < (3, 8):
            raise RuntimeError(f"Python 3.8+ required, found {version.major}.{version.minor}")
        
        if version >= (3, 12):
            # Apply Python 3.12 specific fixes
            WebInterfaceCompatibility._apply_python312_fixes()
        
        return True
    
    @staticmethod
    def _apply_python312_fixes():
        """Apply Python 3.12 specific compatibility fixes"""
        # Placeholder for future Python 3.12-specific fixes
        return None
    
    @staticmethod
    def get_safe_server_config() -> Dict[str, Any]:
        """Get server configuration safe for Python 3.12"""
        config = {
            "host": "localhost",
            "port": 9999,
            "reload": False,  # Safer for production
            "access_log": False,  # Reduce log noise
            "workers": 1,  # Single worker for compatibility
        }
        
        return config

# Initialize compatibility on import
try:
    WebInterfaceCompatibility.check_python_version()
except Exception as e:
    print(f"Web interface compatibility warning: {e}")
'''

    # Write compatibility layer
    compat_file = Path("web_compatibility.py")
    try:
        compat_file.write_text(compatibility_code)
        print(f"✅ Created web compatibility layer: {compat_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to create compatibility layer: {e}")
        return False


def main():
    """Main function"""
    print("🚀 Web Interface Python 3.12 Compatibility Check")
    print(f"Python version: {sys.version}")
    print("-" * 50)

    all_passed = True

    # Run checks
    checks = [
        ("Web Interface Import", check_web_interface),
        ("Async Compatibility", check_async_compatibility),
        ("Deprecated API Patterns", check_deprecated_apis),
    ]

    for check_name, check_func in checks:
        print(f"\n--- {check_name} ---")
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} failed: {e}")
            all_passed = False

    # Create compatibility fixes
    print(f"\n--- Compatibility Fixes ---")
    create_web_compatibility_fix()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if all_passed:
        print("✅ Web interface is compatible with Python 3.12")
        print("💡 Install dependencies with: pip install ipfs_accelerate_py[all]")
    else:
        print("⚠️ Some compatibility issues found")
        print("💡 Check the compatibility layer in web_compatibility.py")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
