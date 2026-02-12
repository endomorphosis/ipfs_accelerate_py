"""
Authentication and authorization package.
"""

from .api_keys import APIKeyManager, APIKey
from .rate_limiter import RateLimiter

# Lazy import for middleware to avoid FastAPI dependency
def get_auth_middleware():
    """Lazy import for AuthMiddleware."""
    from .middleware import AuthMiddleware
    return AuthMiddleware

__all__ = [
    "APIKeyManager",
    "APIKey",
    "RateLimiter",
    "get_auth_middleware",
]
