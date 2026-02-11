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

from .server import HFModelServer
from .config import ServerConfig

__all__ = ["HFModelServer", "ServerConfig"]
