"""Last observed completion blocks. Never a board and never authority."""

from __future__ import annotations

import threading
from typing import Any, Mapping

COMPLETION_BLOCK_KEYS: tuple[str, ...] = (
    "recovery_plan_delta_outstanding",
    "similarity_not_resolution",
    "cold_execution_required",
    "qualification_incomplete",
    "observations_not_preserved",
    "negative_memory_blocks",
    "world_root_cas_not_completion",
    "boundary_contract_required",
    "incompatible_identity",
    "undeclared_cst_transform",
    "unbounded_procedure",
    "rollout_not_required",
    "merge_nomination_outstanding",
)
_LAST = threading.local()


def empty_completion_blocks() -> dict[str, Any]:
    payload: dict[str, Any] = {key: False for key in COMPLETION_BLOCK_KEYS}
    payload["accepted_as_authority"] = False
    payload["completion_authority"] = False
    payload["blocks_completion"] = False
    payload["reason"] = ""
    return payload


def publish_completion_blocks(**flags: bool) -> dict[str, Any]:
    """Record the current block bits for ops and step handoffs."""

    payload = empty_completion_blocks()
    for key in COMPLETION_BLOCK_KEYS:
        payload[key] = bool(flags.get(key))
    for key in COMPLETION_BLOCK_KEYS:
        if payload[key]:
            payload["reason"] = key
            payload["blocks_completion"] = True
            break
    _LAST.value = dict(payload)
    return payload


def last_completion_blocks() -> dict[str, Any]:
    value = getattr(_LAST, "value", None)
    if isinstance(value, Mapping):
        payload = empty_completion_blocks()
        payload.update(dict(value))
        payload["accepted_as_authority"] = False
        payload["completion_authority"] = False
        payload["blocks_completion"] = any(
            payload.get(key) is True for key in COMPLETION_BLOCK_KEYS
        )
        if payload["blocks_completion"] and not payload.get("reason"):
            for key in COMPLETION_BLOCK_KEYS:
                if payload.get(key) is True:
                    payload["reason"] = key
                    break
        return payload
    return empty_completion_blocks()


def clear_completion_blocks() -> dict[str, Any]:
    """Drop the thread-local last blocks. Never a completion."""

    payload = empty_completion_blocks()
    _LAST.value = dict(payload)
    return payload


def completion_is_blocked(source: Mapping[str, Any] | None = None) -> bool:
    """Conjunctive idle/success predicate. Queue-empty is not this bit."""

    if last_completion_blocks().get("blocks_completion") is True:
        return True
    try:
        from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.cst_kit_commit import (
            merge_handoff_outstanding,
        )
    except Exception:
        merge_handoff_outstanding = None
    if merge_handoff_outstanding is not None and merge_handoff_outstanding():
        return True
    if not isinstance(source, Mapping):
        return False
    if source.get("blocks_completion") is True:
        return True
    return any(source.get(key) is True for key in COMPLETION_BLOCK_KEYS)


def attach_completion_blocks(result: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Stamp a daemon/runtime result with the conjunctive block bits.

    Source-unchanged is not board success. Never completion authority.
    """

    payload = dict(result or {})
    blocks = last_completion_blocks()
    payload["completion_blocks"] = blocks
    payload["completion_authority"] = False
    payload["accepted_as_authority"] = False
    blocked = blocks.get("blocks_completion") is True
    payload["blocks_completion"] = blocked
    if blocked:
        payload["blocked"] = True
        if not str(payload.get("reason") or "").strip():
            payload["reason"] = str(blocks.get("reason") or "completion_blocked")
    return payload
