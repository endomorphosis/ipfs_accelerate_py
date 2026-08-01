"""Lazy public surface for peer-to-peer TaskQueue helpers.

Importing a dependency-light leaf such as :mod:`p2p_tasks.task_types` must not
initialize the network service, worker, MCP bridge, or GitHub cache.  Public
compatibility exports are therefore resolved only when explicitly requested.
"""

from __future__ import annotations

import importlib
from typing import Final

_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "PROTOCOL_V1": ("protocol", "PROTOCOL_V1"),
    "auth_ok": ("protocol", "auth_ok"),
    "get_shared_token": ("protocol", "get_shared_token"),
    "PeerTrustLevel": ("peer_trust", "PeerTrustLevel"),
    "baseline_max_claim_priority": (
        "peer_trust",
        "baseline_max_claim_priority",
    ),
    "resolve_peer_trust_level": ("peer_trust", "resolve_peer_trust_level"),
    "trust_tiers_enabled": ("peer_trust", "trust_tiers_enabled"),
    "TaskQueue": ("task_queue", "TaskQueue"),
    "default_queue_path": ("task_queue", "default_queue_path"),
    "serve_task_queue": ("service", "serve_task_queue"),
    "RemoteQueue": ("client", "RemoteQueue"),
    "submit_task": ("client", "submit_task"),
    "submit_task_with_info": ("client", "submit_task_with_info"),
    "submit_task_sync": ("client", "submit_task_sync"),
    "submit_task_with_info_sync": ("client", "submit_task_with_info_sync"),
    "submit_docker_hub_task": ("client", "submit_docker_hub_task"),
    "submit_docker_hub_task_sync": (
        "client",
        "submit_docker_hub_task_sync",
    ),
    "submit_docker_github_task": ("client", "submit_docker_github_task"),
    "submit_docker_github_task_sync": (
        "client",
        "submit_docker_github_task_sync",
    ),
    "claim_next": ("client", "claim_next"),
    "claim_next_sync": ("client", "claim_next_sync"),
    "heartbeat": ("client", "heartbeat"),
    "heartbeat_sync": ("client", "heartbeat_sync"),
    "list_tasks": ("client", "list_tasks"),
    "list_tasks_sync": ("client", "list_tasks_sync"),
    "complete_task": ("client", "complete_task"),
    "complete_task_sync": ("client", "complete_task_sync"),
    "get_task": ("client", "get_task"),
    "wait_task": ("client", "wait_task"),
    "get_capabilities": ("client", "get_capabilities"),
    "get_capabilities_sync": ("client", "get_capabilities_sync"),
    "call_tool": ("client", "call_tool"),
    "call_tool_sync": ("client", "call_tool_sync"),
    "cache_get": ("client", "cache_get"),
    "cache_get_sync": ("client", "cache_get_sync"),
    "cache_has": ("client", "cache_has"),
    "cache_has_sync": ("client", "cache_has_sync"),
    "cache_set": ("client", "cache_set"),
    "cache_set_sync": ("client", "cache_set_sync"),
    "run_worker": ("worker", "run_worker"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Resolve one compatibility export without importing unrelated stacks."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
