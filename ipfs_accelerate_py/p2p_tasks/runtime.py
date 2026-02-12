"""Runtime helpers for running the TaskQueue libp2p service in-process.

The TaskQueue service (`serve_task_queue`) is Trio-based and long-lived.
This module provides a small thread-based runner so other components (e.g. MCP)
can start the service and later stop it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskQueueP2PServiceHandle:
    thread: threading.Thread
    started: threading.Event


class TaskQueueP2PServiceRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._trio_token = None
        self._cancel_scope = None
        self._last_error: Optional[BaseException] = None

    @property
    def started(self) -> bool:
        return self._started.is_set()

    @property
    def running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    @property
    def last_error(self) -> str:
        return str(self._last_error) if self._last_error else ""

    def start(self, *, queue_path: str, listen_port: Optional[int] = None, accelerate_instance: object | None = None) -> TaskQueueP2PServiceHandle:
        """Start the TaskQueue p2p service in a background thread.

        Safe to call multiple times; subsequent calls return the existing handle.
        """

        with self._lock:
            if self._thread and self._thread.is_alive():
                return TaskQueueP2PServiceHandle(thread=self._thread, started=self._started)

            self._started.clear()
            self._last_error = None

            def _runner() -> None:
                try:
                    import trio

                    async def _main() -> None:
                        self._trio_token = trio.lowlevel.current_trio_token()
                        with trio.CancelScope() as scope:
                            self._cancel_scope = scope
                            self._started.set()
                            from .service import serve_task_queue

                            await serve_task_queue(
                                queue_path=str(queue_path),
                                listen_port=int(listen_port) if listen_port is not None else None,
                                accelerate_instance=accelerate_instance,
                            )

                    trio.run(_main)
                except BaseException as exc:
                    self._last_error = exc
                    # If the service died before marking started, unblock waiters.
                    self._started.set()

            t = threading.Thread(
                target=_runner,
                name="ipfs_accelerate_py_taskqueue_p2p_service",
                daemon=True,
            )
            self._thread = t
            t.start()
            return TaskQueueP2PServiceHandle(thread=t, started=self._started)

    def stop(self, *, timeout_s: float = 2.0) -> bool:
        """Best-effort stop of the background service."""

        with self._lock:
            if not self._thread:
                return True

            if self._trio_token is not None and self._cancel_scope is not None:
                try:
                    import trio

                    def _cancel() -> None:
                        try:
                            self._cancel_scope.cancel()
                        except Exception:
                            pass

                    trio.from_thread.run_sync(_cancel, trio_token=self._trio_token)
                except Exception:
                    pass

            try:
                self._thread.join(timeout=max(0.1, float(timeout_s)))
            except Exception:
                pass

            return not self._thread.is_alive()
