"""SAWM W1 world-root CAS and VFS outbox on the existing board owner.

A publication request cannot change the current root or complete a task.
``cas_completed`` and ``completion_authority`` on a world-root or outbox
record are not board success. Missing payload is fail-open. TypeSafe is
never this owner.
"""

from __future__ import annotations

from typing import Any, Mapping

WORLD_ROOT_CAS_NOT_COMPLETION = "world_root_cas_not_completion"
PUBLICATION_REQUEST_ONLY = "publication_request_only"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _record(state: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in (
        "world_root_publication",
        "vfs_outbox",
        "semantic_world_root",
        "publication_request",
    ):
        nested = state.get(key)
        if isinstance(nested, Mapping):
            return {**state, **nested}
    return state


def claims_world_root_cas(state: Mapping[str, Any] | None) -> bool:
    payload = _mapping(state)
    return bool(
        payload.get("world_root_publication")
        or payload.get("vfs_outbox")
        or payload.get("semantic_world_root_cid")
        or payload.get("publication_request")
        or payload.get("semantic_world_root")
        or payload.get("world_root")
    )


def world_root_cas_view(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Inspect a wake payload. Never completes a task."""

    payload = _mapping(state)
    claimed = claims_world_root_cas(payload)
    merged = _record(payload)
    illegal = bool(
        merged.get("cas_completed") is True
        or merged.get("completion_authority") is True
        or merged.get("changes_current_root") is True
        or (
            merged.get("admitted") is True
            and merged.get("proposal_only") is not True
            and claimed
        )
    )
    blocks = bool(claimed and illegal)
    return {
        "accepted_as_authority": False,
        "completes_task": False,
        "claimed": claimed,
        "proposal_only": merged.get("proposal_only") is True or not illegal,
        "blocks_completion": blocks,
        "reason_code": (
            WORLD_ROOT_CAS_NOT_COMPLETION
            if blocks
            else (PUBLICATION_REQUEST_ONLY if claimed else "")
        ),
    }


def world_root_cas_blocks_completion(state: Mapping[str, Any] | None) -> bool:
    return bool(world_root_cas_view(state)["blocks_completion"])


__all__ = [
    "PUBLICATION_REQUEST_ONLY",
    "WORLD_ROOT_CAS_NOT_COMPLETION",
    "claims_world_root_cas",
    "world_root_cas_blocks_completion",
    "world_root_cas_view",
]
