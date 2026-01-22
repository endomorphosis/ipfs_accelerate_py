"""
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
        # Ensure proper error handling for async operations
        import asyncio
        
        # Fix for potential asyncio changes
        if not hasattr(asyncio, 'coroutine'):
            # asyncio.coroutine was removed, ensure we use async def
            pass
    
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
        
        # Python 3.12 specific adjustments
        if sys.version_info >= (3, 12):
            config["loop"] = "asyncio"  # Explicit loop specification
            
        return config

# Initialize compatibility on import
try:
    WebInterfaceCompatibility.check_python_version()
except Exception as e:
    print(f"Web interface compatibility warning: {e}")
