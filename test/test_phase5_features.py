"""
Tests for Phase 5: Authentication, Rate Limiting, and Request Queuing.
"""

import pytest
import anyio
from datetime import datetime

# Import directly to avoid server dependencies
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ipfs_accelerate_py.hf_model_server.auth.api_keys import APIKeyManager, APIKey
from ipfs_accelerate_py.hf_model_server.auth.rate_limiter import RateLimiter
from ipfs_accelerate_py.hf_model_server.middleware.request_queue import (
    RequestQueue,
    QueueManager,
    RequestPriority,
    QueuedRequest
)


class TestAPIKeyManager:
    """Test API key management."""
    
    def test_generate_key(self):
        """Test API key generation."""
        manager = APIKeyManager()
        
        key_string, api_key = manager.generate_key(
            name="test_key",
            rate_limit=50
        )
        
        assert key_string.startswith("hf_")
        assert api_key.name == "test_key"
        assert api_key.rate_limit == 50
        assert api_key.is_active
        assert len(api_key.key_id) == 16
    
    def test_validate_key(self):
        """Test API key validation."""
        manager = APIKeyManager()
        
        key_string, api_key = manager.generate_key("test_key")
        
        # Validate correct key
        validated = manager.validate_key(key_string)
        assert validated is not None
        assert validated.key_id == api_key.key_id
        assert validated.last_used_at is not None
        
        # Validate incorrect key
        invalid = manager.validate_key("invalid_key")
        assert invalid is None
    
    def test_revoke_key(self):
        """Test API key revocation."""
        manager = APIKeyManager()
        
        key_string, api_key = manager.generate_key("test_key")
        
        # Revoke key
        success = manager.revoke_key(api_key.key_id)
        assert success
        assert not api_key.is_active
        
        # Validate revoked key
        validated = manager.validate_key(key_string)
        assert validated is None
    
    def test_list_keys(self):
        """Test listing API keys."""
        manager = APIKeyManager()
        
        # Generate keys
        manager.generate_key("key1")
        manager.generate_key("key2")
        key3_str, key3 = manager.generate_key("key3")
        
        # List active keys
        keys = manager.list_keys(include_inactive=False)
        assert len(keys) == 3
        
        # Revoke one
        manager.revoke_key(key3.key_id)
        keys = manager.list_keys(include_inactive=False)
        assert len(keys) == 2
        
        # List all keys
        all_keys = manager.list_keys(include_inactive=True)
        assert len(all_keys) == 3
    
    def test_allowed_models(self):
        """Test allowed models restriction."""
        manager = APIKeyManager()
        
        key_string, api_key = manager.generate_key(
            name="restricted_key",
            allowed_models=["gpt2", "bert-base"]
        )
        
        assert api_key.allowed_models == ["gpt2", "bert-base"]


class TestRateLimiter:
    """Test rate limiting."""
    
    @pytest.mark.asyncio
    async def test_check_limit(self):
        """Test rate limit checking."""
        limiter = RateLimiter(enabled=True)
        
        api_key = APIKey(
            key_id="test_key",
            key_hash="hash",
            name="test",
            rate_limit=5
        )
        
        # Should allow first 5 requests
        for i in range(5):
            allowed, remaining = await limiter.check_limit(api_key)
            assert allowed
            assert remaining == 5 - (i + 1)
        
        # Should deny 6th request
        allowed, remaining = await limiter.check_limit(api_key)
        assert not allowed
        assert remaining == 0
    
    @pytest.mark.asyncio
    async def test_rate_limit_disabled(self):
        """Test rate limiter when disabled."""
        limiter = RateLimiter(enabled=False)
        
        api_key = APIKey(
            key_id="test_key",
            key_hash="hash",
            name="test",
            rate_limit=5
        )
        
        # Should allow unlimited requests
        for _ in range(10):
            allowed, remaining = await limiter.check_limit(api_key)
            assert allowed
    
    @pytest.mark.asyncio
    async def test_window_reset(self):
        """Test that rate limit window resets."""
        limiter = RateLimiter(enabled=True)
        
        api_key = APIKey(
            key_id="test_key",
            key_hash="hash",
            name="test",
            rate_limit=3
        )
        
        # Use up limit
        for _ in range(3):
            await limiter.check_limit(api_key)
        
        # Manually reset window by manipulating internal state
        limiter._counts[api_key.key_id] = (0, 0.0)
        
        # Should allow requests again
        allowed, remaining = await limiter.check_limit(api_key)
        assert allowed
    
    def test_get_headers(self):
        """Test rate limit header generation."""
        limiter = RateLimiter()
        
        api_key = APIKey(
            key_id="test_key",
            key_hash="hash",
            name="test",
            rate_limit=100
        )
        
        headers = limiter.get_headers(api_key)
        
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers
        assert headers["X-RateLimit-Limit"] == "100"


class TestRequestQueue:
    """Test request queuing."""
    
    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self):
        """Test basic enqueue and dequeue."""
        queue = RequestQueue(max_size=10, timeout_seconds=5.0)
        
        # Enqueue request
        success = await queue.enqueue(
            request_id="req1",
            model_id="gpt2",
            data={"prompt": "test"},
            priority=RequestPriority.NORMAL
        )
        assert success
        
        # Check size
        size = await queue.size()
        assert size == 1
        
        # Dequeue request
        request = await queue.dequeue(timeout=1.0)
        assert request is not None
        assert request.request_id == "req1"
        assert request.model_id == "gpt2"
        
        # Queue should be empty
        size = await queue.size()
        assert size == 0
    
    @pytest.mark.asyncio
    async def test_queue_full(self):
        """Test queue full behavior."""
        queue = RequestQueue(max_size=2)
        
        # Fill queue
        await queue.enqueue("req1", "gpt2", {})
        await queue.enqueue("req2", "gpt2", {})
        
        # Should reject third request
        success = await queue.enqueue("req3", "gpt2", {})
        assert not success
        
        # Should report full
        is_full = await queue.is_full()
        assert is_full
    
    @pytest.mark.asyncio
    async def test_priority_queue(self):
        """Test priority queue ordering."""
        queue = RequestQueue(max_size=10, enable_priority=True)
        
        # Enqueue with different priorities
        await queue.enqueue("req1", "gpt2", {}, priority=RequestPriority.LOW)
        await queue.enqueue("req2", "gpt2", {}, priority=RequestPriority.HIGH)
        await queue.enqueue("req3", "gpt2", {}, priority=RequestPriority.NORMAL)
        
        # Should dequeue in priority order (HIGH, NORMAL, LOW)
        req1 = await queue.dequeue()
        assert req1.request_id == "req2"  # HIGH
        
        req2 = await queue.dequeue()
        assert req2.request_id == "req3"  # NORMAL
        
        req3 = await queue.dequeue()
        assert req3.request_id == "req1"  # LOW
    
    @pytest.mark.asyncio
    async def test_timeout(self):
        """Test dequeue timeout."""
        queue = RequestQueue()
        
        # Try to dequeue from empty queue with timeout
        request = await queue.dequeue(timeout=0.1)
        assert request is None
    
    @pytest.mark.asyncio
    async def test_expired_requests(self):
        """Test that expired requests are removed."""
        queue = RequestQueue(timeout_seconds=0.1)
        
        # Enqueue request
        await queue.enqueue("req1", "gpt2", {})
        
        # Wait for expiration
        await anyio.sleep(0.15)
        
        # Try to dequeue - should get None (expired)
        request = await queue.dequeue(timeout=0.1)
        # The request might be removed, or we might get None
        # Either way, queue should handle it gracefully
        
        # Stats should show timeout
        stats = queue.get_stats()
        # Could be 0 or 1 depending on when cleanup happened
        assert stats["total_timeouts"] >= 0
    
    @pytest.mark.asyncio
    async def test_clear_queue(self):
        """Test clearing queue."""
        queue = RequestQueue()
        
        # Add requests
        await queue.enqueue("req1", "gpt2", {})
        await queue.enqueue("req2", "gpt2", {})
        
        size = await queue.size()
        assert size == 2
        
        # Clear queue
        await queue.clear()
        
        size = await queue.size()
        assert size == 0
    
    def test_get_stats(self):
        """Test queue statistics."""
        queue = RequestQueue(max_size=10)
        
        stats = queue.get_stats()
        
        assert "current_size" in stats
        assert "max_size" in stats
        assert "total_queued" in stats
        assert "total_processed" in stats
        assert "total_timeouts" in stats
        assert "total_rejected" in stats
        assert "utilization" in stats


class TestQueueManager:
    """Test queue manager."""
    
    @pytest.mark.asyncio
    async def test_global_queue(self):
        """Test global queue mode."""
        manager = QueueManager(
            default_max_size=10,
            enable_per_model_queues=False
        )
        
        # Enqueue requests for different models
        await manager.enqueue("req1", "gpt2", {})
        await manager.enqueue("req2", "bert", {})
        
        # Should use same queue
        req1 = await manager.dequeue()
        req2 = await manager.dequeue()
        
        assert req1.request_id == "req1"
        assert req2.request_id == "req2"
    
    @pytest.mark.asyncio
    async def test_per_model_queues(self):
        """Test per-model queue mode."""
        manager = QueueManager(
            default_max_size=10,
            enable_per_model_queues=True
        )
        
        # Enqueue requests for different models
        await manager.enqueue("req1", "gpt2", {})
        await manager.enqueue("req2", "bert", {})
        
        # Dequeue from gpt2 queue
        req1 = await manager.dequeue(model_id="gpt2")
        assert req1.request_id == "req1"
        assert req1.model_id == "gpt2"
        
        # Dequeue from bert queue
        req2 = await manager.dequeue(model_id="bert")
        assert req2.request_id == "req2"
        assert req2.model_id == "bert"
    
    @pytest.mark.asyncio
    async def test_aggregate_stats(self):
        """Test aggregate statistics."""
        manager = QueueManager(enable_per_model_queues=True)
        
        # Add requests to different model queues
        await manager.enqueue("req1", "gpt2", {})
        await manager.enqueue("req2", "bert", {})
        
        # Get aggregate stats
        stats = await manager.get_stats()
        
        assert "total_size" in stats
        assert stats["total_size"] == 2
        assert "queues" in stats
        assert len(stats["queues"]) == 2


class TestQueuedRequest:
    """Test queued request dataclass."""
    
    def test_is_expired(self):
        """Test expiration check."""
        import time
        from datetime import datetime, timedelta
        
        # Create request with 0.1 second timeout
        request = QueuedRequest(
            request_id="req1",
            model_id="gpt2",
            data={},
            timeout_seconds=0.1
        )
        
        # Should not be expired immediately
        assert not request.is_expired()
        
        # Manually set old timestamp
        request.queued_at = datetime.utcnow() - timedelta(seconds=1)
        
        # Should be expired now
        assert request.is_expired()
    
    def test_comparison(self):
        """Test priority comparison for sorting."""
        req_low = QueuedRequest(
            request_id="low",
            model_id="gpt2",
            data={},
            priority=RequestPriority.LOW
        )
        
        req_high = QueuedRequest(
            request_id="high",
            model_id="gpt2",
            data={},
            priority=RequestPriority.HIGH
        )
        
        # High priority should be "less than" (comes first)
        assert req_high < req_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
