"""
Unified HuggingFace Model Server

A production-ready model server for HuggingFace models with:
- Automatic skill discovery and registration
- OpenAI-compatible API endpoints
- Intelligent hardware selection
- Multi-model serving with load balancing
- Request batching and caching
- Circuit breaker pattern
- Health checks and Prometheus metrics
- Complete async/await support
"""

__version__ = "0.1.0"
__author__ = "IPFS Accelerate Team"

from .config import ServerConfig

# Lazy import server to avoid dependency issues
def create_server(config=None):
    """Create and return HFModelServer instance."""
    from .server import HFModelServer
    return HFModelServer(config)

__all__ = ["create_server", "ServerConfig"]
