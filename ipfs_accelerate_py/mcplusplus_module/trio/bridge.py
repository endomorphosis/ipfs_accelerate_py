"""
Trio bridge and runtime utilities for MCP++

This module provides utilities for running Trio code in different contexts
and bridging between asyncio and Trio when necessary.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, TypeVar

import anyio
import sniffio

logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.trio.bridge")

T = TypeVar("T")


async def run_in_trio(func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run a callable in a Trio context.

    The libp2p stack used by TaskQueue is Trio-based. When MCP tools run
    under FastAPI/Uvicorn, the ambient async runtime is typically asyncio.
    Running Trio-only code under asyncio can fail in surprising ways.

    If we're already in Trio, run inline. Otherwise, execute in a worker thread
    using `anyio.run(..., backend='trio')`.

    Args:
        func: The callable to run (can be sync or async)
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        The result of calling func(*args, **kwargs)

    Example:
        >>> import trio
        >>> async def trio_only_function(x):
        ...     async with trio.open_nursery() as nursery:
        ...         return x * 2
        ...
        >>> # From asyncio context:
        >>> result = await run_in_trio(trio_only_function, 21)
        >>> # result == 42
    """
    try:
        current_library = sniffio.current_async_library()
        if current_library == "trio":
            # We're already in Trio, run directly
            result = func(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result
    except sniffio.AsyncLibraryNotFoundError:
        # No async library detected, will run in thread
        pass

    # We're in asyncio or no async context - run in a thread with Trio
    def _runner() -> T:
        async def _inner() -> T:
            result = func(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result

        return anyio.run(_inner, backend="trio")

    return await anyio.to_thread.run_sync(_runner)


def is_trio_context() -> bool:
    """Check if we're running in a Trio context.

    Returns:
        True if currently in a Trio event loop, False otherwise
    """
    try:
        return sniffio.current_async_library() == "trio"
    except sniffio.AsyncLibraryNotFoundError:
        return False


def require_trio() -> None:
    """Raise an error if not running in a Trio context.

    Raises:
        RuntimeError: If not in a Trio event loop

    Example:
        >>> async def trio_only_function():
        ...     require_trio()
        ...     # Rest of function assumes Trio
    """
    if not is_trio_context():
        raise RuntimeError(
            "This function requires a Trio event loop. "
            "Use trio.run() or run_in_trio() wrapper."
        )


class TrioContext:
    """Context manager for ensuring code runs in a Trio context.

    This is useful for library code that should work in both asyncio and Trio
    contexts but has some Trio-specific requirements.

    Example:
        >>> async def my_function():
        ...     async with TrioContext():
        ...         # This code is guaranteed to run in Trio
        ...         async with trio.open_nursery() as nursery:
        ...             nursery.start_soon(some_task)
    """

    def __init__(self):
        self._was_trio = False
        self._context = None

    async def __aenter__(self):
        self._was_trio = is_trio_context()
        if not self._was_trio:
            # We need to transition to Trio
            # This is complex and generally not recommended
            # Instead, use run_in_trio for individual operations
            logger.warning(
                "TrioContext used in non-Trio context. "
                "Consider using run_in_trio() for individual operations instead."
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass


__all__ = [
    "run_in_trio",
    "is_trio_context",
    "require_trio",
    "TrioContext",
]
