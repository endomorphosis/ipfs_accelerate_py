"""Peer-to-peer TaskQueue transport + worker helpers.

This package is intended to be reusable by long-running services (e.g. systemd)
that want to participate in the distributed TaskQueue network.

It is intentionally dependency-minimal and uses py-libp2p for transport.
"""

from .protocol import PROTOCOL_V1, auth_ok, get_shared_token
from .task_queue import TaskQueue, default_queue_path
from .service import serve_task_queue
from .client import (
    RemoteQueue,
    submit_task,
    submit_task_with_info,
    get_task,
    wait_task,
    get_capabilities,
    get_capabilities_sync,
    call_tool,
    call_tool_sync,
    cache_get,
    cache_get_sync,
    cache_has,
    cache_has_sync,
    cache_set,
    cache_set_sync,
)
from .worker import run_worker

__all__ = [
    "PROTOCOL_V1",
    "auth_ok",
    "get_shared_token",
    "TaskQueue",
    "default_queue_path",
    "serve_task_queue",
    "RemoteQueue",
    "submit_task",
    "submit_task_with_info",
    "get_task",
    "wait_task",
    "get_capabilities",
    "get_capabilities_sync",
    "call_tool",
    "call_tool_sync",
    "cache_get",
    "cache_get_sync",
    "cache_has",
    "cache_has_sync",
    "cache_set",
    "cache_set_sync",
    "run_worker",
]
