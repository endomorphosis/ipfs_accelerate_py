"""Peer-to-peer TaskQueue transport + worker helpers.

This package is intended to be reusable by long-running services (e.g. systemd)
that want to participate in the distributed TaskQueue network.

It is intentionally dependency-minimal and uses py-libp2p for transport.
"""

from .protocol import PROTOCOL_V1, auth_ok, get_shared_token
from .task_queue import TaskQueue, default_queue_path
from .service import serve_task_queue
from .client import RemoteQueue, submit_task, submit_task_with_info, get_task, wait_task
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
    "run_worker",
]
