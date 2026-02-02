"""
Rate limiting for API requests.
"""

import time
import logging
from collections import defaultdict
from typing import Dict, Tuple
import asyncio

from .api_keys import APIKey

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, enabled: bool = True):
        """
        Initialize rate limiter.
        
        Args:
            enabled: Whether rate limiting is enabled
        """
        self.enabled = enabled
        # key_id -> (count, window_start)
        self._counts: Dict[str, Tuple[int, float]] = defaultdict(lambda: (0, time.time()))
        self._lock = asyncio.Lock()
    
    async def check_limit(self, api_key: APIKey) -> Tuple[bool, int]:
        """
        Check if request is within rate limit.
        
        Args:
            api_key: APIKey object
            
        Returns:
            Tuple of (allowed, remaining_requests)
        """
        if not self.enabled:
            return True, api_key.rate_limit
        
        async with self._lock:
            key_id = api_key.key_id
            count, window_start = self._counts[key_id]
            current_time = time.time()
            
            # Reset window if needed (60 seconds)
            if current_time - window_start >= 60:
                count = 0
                window_start = current_time
            
            # Check limit
            if count >= api_key.rate_limit:
                remaining = 0
                allowed = False
            else:
                count += 1
                remaining = api_key.rate_limit - count
                allowed = True
            
            # Update counts
            self._counts[key_id] = (count, window_start)
            
            if not allowed:
                logger.warning(f"Rate limit exceeded for key: {key_id}")
            
            return allowed, remaining
    
    def get_headers(self, api_key: APIKey) -> Dict[str, str]:
        """Get rate limit headers."""
        count, window_start = self._counts.get(api_key.key_id, (0, time.time()))
        remaining = max(0, api_key.rate_limit - count)
        reset_time = int(window_start + 60)
        
        return {
            "X-RateLimit-Limit": str(api_key.rate_limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
        }
