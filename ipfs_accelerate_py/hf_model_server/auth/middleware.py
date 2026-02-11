"""
Authentication middleware for FastAPI.
"""

import logging
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from .api_keys import APIKeyManager

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


class AuthMiddleware:
    """Authentication middleware."""
    
    def __init__(self, api_key_manager: APIKeyManager, enabled: bool = True):
        """
        Initialize auth middleware.
        
        Args:
            api_key_manager: APIKeyManager instance
            enabled: Whether authentication is enabled
        """
        self.api_key_manager = api_key_manager
        self.enabled = enabled
    
    async def verify_request(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = None
    ):
        """
        Verify request authentication.
        
        Args:
            request: FastAPI request
            credentials: HTTP authorization credentials
            
        Raises:
            HTTPException: If authentication fails
        """
        if not self.enabled:
            return None
        
        # Extract API key
        api_key = None
        if credentials:
            api_key = credentials.credentials
        elif "x-api-key" in request.headers:
            api_key = request.headers["x-api-key"]
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Validate key
        key_obj = self.api_key_manager.validate_key(api_key)
        if not key_obj:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Store in request state
        request.state.api_key = key_obj
        logger.debug(f"Request authenticated with key: {key_obj.key_id}")
        
        return key_obj
