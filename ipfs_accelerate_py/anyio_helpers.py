"""Small AnyIO-based helpers used during the asyncio -> AnyIO migration.

These mirror a subset of the common asyncio APIs used throughout the repo
(e.g. gather(), wait_for()) so that bulk mechanical refactors can land
without forcing every callsite to be rewritten manually in one pass.

They intentionally depend only on AnyIO.
"""

from ipfs_accelerate_py.anyio_helpers import gather, wait_for
from __future__ import annotations

from typing import Any, Awaitable, List, Sequence, TypeVar

import anyio

T = TypeVar("T")


async def wait_for(awaitable: Awaitable[T], timeout: float) -> T:
    """Await an awaitable with a timeout.

    This is an AnyIO equivalent of asyncio.wait_for(awaitable, timeout=...).
    """
    with anyio.fail_after(timeout):
        return await awaitable


async def gather(*awaitables: Awaitable[T], return_exceptions: bool = False) -> List[Any]:
    """Run awaitables concurrently and collect results.

    Similar to asyncio.gather(*aws, return_exceptions=...).

    Notes:
    - When return_exceptions is False, the first exception cancels siblings.
    - When return_exceptions is True, exceptions are captured in the results list.
    """

    results: List[Any] = [None] * len(awaitables)

    async def _run_one(index: int, aw: Awaitable[T]) -> None:
        try:
            results[index] = await aw
        except BaseException as exc:  # noqa: BLE001
            if return_exceptions:
                results[index] = exc
            else:
                raise

    async with anyio.create_task_group() as tg:
        for i, aw in enumerate(awaitables):
            tg.start_soon(_run_one, i, aw)

    return results
