"""
API key management.
"""

import secrets
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class APIKey:
    """API key with metadata."""
    key_id: str
    key_hash: str
    name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    rate_limit: int = 100  # requests per minute
    allowed_models: Optional[List[str]] = None  # None = all models
    metadata: Dict = field(default_factory=dict)


class APIKeyManager:
    """Manages API keys for authentication."""
    
    def __init__(self):
        """Initialize API key manager."""
        self._keys: Dict[str, APIKey] = {}
    
    def generate_key(
        self,
        name: str,
        rate_limit: int = 100,
        allowed_models: Optional[List[str]] = None
    ) -> tuple[str, APIKey]:
        """
        Generate new API key.
        
        Args:
            name: Name/description for the key
            rate_limit: Rate limit (requests per minute)
            allowed_models: List of allowed model IDs (None = all)
            
        Returns:
            Tuple of (api_key_string, APIKey object)
        """
        # Generate random key
        key_string = f"hf_{secrets.token_urlsafe(32)}"
        
        # Hash for storage
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        key_id = key_hash[:16]
        
        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            rate_limit=rate_limit,
            allowed_models=allowed_models,
        )
        
        self._keys[key_hash] = api_key
        logger.info(f"Generated API key: {key_id} ({name})")
        
        return key_string, api_key
    
    def validate_key(self, key_string: str) -> Optional[APIKey]:
        """
        Validate API key.
        
        Args:
            key_string: API key string
            
        Returns:
            APIKey if valid, None otherwise
        """
        if not key_string:
            return None
        
        # Hash and look up
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        api_key = self._keys.get(key_hash)
        
        if api_key and api_key.is_active:
            # Update last used
            api_key.last_used_at = datetime.utcnow()
            return api_key
        
        return None
    
    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke API key.
        
        Args:
            key_id: Key ID to revoke
            
        Returns:
            True if revoked, False if not found
        """
        for api_key in self._keys.values():
            if api_key.key_id == key_id:
                api_key.is_active = False
                logger.info(f"Revoked API key: {key_id}")
                return True
        return False
    
    def list_keys(self, include_inactive: bool = False) -> List[APIKey]:
        """
        List all API keys.
        
        Args:
            include_inactive: Include inactive keys
            
        Returns:
            List of APIKey objects
        """
        if include_inactive:
            return list(self._keys.values())
        else:
            return [k for k in self._keys.values() if k.is_active]
    
    def get_key_by_id(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID."""
        for api_key in self._keys.values():
            if api_key.key_id == key_id:
                return api_key
        return None
