from __future__ import annotations

import threading
from typing import Generic, TypeVar

import anyio

T = TypeVar("T")


class AnyioQueue(Generic[T]):
    """A small queue-like adapter backed by AnyIO.

    This is intentionally minimal; it supports the subset used across this repo:
    `put`, `put_nowait`, `get`, `get_nowait`, `qsize`, `empty`, and `task_done`.

    Note: size is approximate but consistent for in-process usage.
    """

    def __init__(self, maxsize: int = 0):
        # anyio streams use `max_buffer_size`; 0 means unbuffered.
        self._send, self._recv = anyio.create_memory_object_stream(maxsize)
        self._maxsize = maxsize
        self._size = 0
        self._size_lock = threading.Lock()

    async def put(self, item: T) -> None:
        await self._send.send(item)
        with self._size_lock:
            self._size += 1

    def put_nowait(self, item: T) -> None:
        self._send.send_nowait(item)
        with self._size_lock:
            self._size += 1

    async def get(self) -> T:
        item = await self._recv.receive()
        with self._size_lock:
            if self._size > 0:
                self._size -= 1
        return item

    def get_nowait(self) -> T:
        item = self._recv.receive_nowait()
        with self._size_lock:
            if self._size > 0:
                self._size -= 1
        return item

    def qsize(self) -> int:
        return self._size

    def empty(self) -> bool:
        return self._size <= 0

    def task_done(self) -> None:
        # Provided for compatibility with queue-like interfaces.
        return None

    async def aclose(self) -> None:
        await self._send.aclose()
        await self._recv.aclose()
