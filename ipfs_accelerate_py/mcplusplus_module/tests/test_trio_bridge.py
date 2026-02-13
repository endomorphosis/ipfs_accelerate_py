"""
Tests for MCP++ Trio bridge functionality

This module tests the Trio bridge utilities that allow running Trio code
in different async contexts.

Module: ipfs_accelerate_py.mcplusplus_module.tests.test_trio_bridge
"""

import pytest
import trio

from ipfs_accelerate_py.mcplusplus_module.trio import (
    run_in_trio,
    is_trio_context,
    require_trio,
)


class TestTrioBridge:
    """Tests for Trio bridge utilities."""

    def test_is_trio_context_outside(self):
        """Test is_trio_context returns False outside Trio."""
        assert not is_trio_context()

    @pytest.mark.trio
    async def test_is_trio_context_inside(self):
        """Test is_trio_context returns True inside Trio."""
        assert is_trio_context()

    def test_require_trio_outside(self):
        """Test require_trio raises outside Trio."""
        with pytest.raises(RuntimeError, match="requires a Trio event loop"):
            require_trio()

    @pytest.mark.trio
    async def test_require_trio_inside(self):
        """Test require_trio passes inside Trio."""
        require_trio()  # Should not raise

    @pytest.mark.trio
    async def test_run_in_trio_already_in_trio(self):
        """Test run_in_trio when already in Trio context."""
        async def trio_func(x):
            assert is_trio_context()
            return x * 2

        result = await run_in_trio(trio_func, 21)
        assert result == 42

    @pytest.mark.trio
    async def test_run_in_trio_sync_function(self):
        """Test run_in_trio with a synchronous function."""
        def sync_func(x, y):
            return x + y

        result = await run_in_trio(sync_func, 10, 32)
        assert result == 42


class TestTrioIntegration:
    """Integration tests for Trio functionality."""

    @pytest.mark.trio
    async def test_trio_nursery(self):
        """Test that Trio nurseries work correctly."""
        results = []

        async def worker(n):
            await trio.sleep(0.01)
            results.append(n)

        async with trio.open_nursery() as nursery:
            for i in range(5):
                nursery.start_soon(worker, i)

        assert len(results) == 5
        assert set(results) == {0, 1, 2, 3, 4}

    @pytest.mark.trio
    async def test_trio_cancel_scope(self):
        """Test Trio cancel scopes."""
        results = []

        async def long_task():
            for i in range(10):
                await trio.sleep(0.1)
                results.append(i)

        with trio.move_on_after(0.15) as cancel_scope:
            await long_task()

        assert cancel_scope.cancelled_caught
        assert len(results) < 10  # Should have been cancelled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
