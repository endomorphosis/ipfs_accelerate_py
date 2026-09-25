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
        return payload
    return empty_completion_blocks()
