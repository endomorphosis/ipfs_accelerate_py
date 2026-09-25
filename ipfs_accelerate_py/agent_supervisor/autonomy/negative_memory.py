"""SPAR W4 / SAWM negative memory on the existing board owner.

A retained negative episode cannot become reuse, a passing wave, or board
success. Exact identity does not override it. Missing payload is fail-open:
negative memory is not required on every wake. TypeSafe is never this owner.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

NEGATIVE_MEMORY_BLOCKS_REUSE = "negative_memory_blocks"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
        return tuple(out)
    return ()


def _query_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    found: list[str] = []
    for key in ("key_cid", "transition_cid", "query_key_cid"):
        text = str(payload.get(key) or "").strip()
        if text and text not in found:
            found.append(text)
    nested = payload.get("accepted_transition")
    if isinstance(nested, Mapping):
        cid = str(nested.get("transition_cid") or nested.get("cid") or "").strip()
        if cid and cid not in found:
            found.append(cid)
    query = payload.get("query_transition")
    if isinstance(query, Mapping):
        cid = str(query.get("transition_cid") or query.get("cid") or "").strip()
        if cid and cid not in found:
            found.append(cid)
    return tuple(found)


def claims_negative_memory(state: Mapping[str, Any] | None) -> bool:
    payload = _mapping(state)
    nested = payload.get("negative_memory")
    if isinstance(nested, Mapping) or payload.get("negative_memory") is True:
        return True
    return bool(
        payload.get("negative_episode") is True
        or payload.get("retained_negative_cids")
        or (
            isinstance(payload.get("accepted_transition"), Mapping)
            and payload["accepted_transition"].get("negative_episode") is True
        )
    )


def negative_memory_view(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Inspect a wake payload. Never completes a task."""

    payload = _mapping(state)
    nested = payload.get("negative_memory")
    extra = nested if isinstance(nested, Mapping) else {}
    merged = {**payload, **dict(extra)}
    claimed = claims_negative_memory(payload)
    retained = _ids(merged.get("retained_negative_cids"))
    query = _query_ids(merged)
    explicit = (
        payload.get("negative_memory") is True
        or payload.get("negative_episode") is True
        or (
            isinstance(payload.get("accepted_transition"), Mapping)
            and payload["accepted_transition"].get("negative_episode") is True
        )
    )
    hit = bool(retained and query and set(retained) & set(query))
    unmatched = bool(retained and query and not hit)
    blocks = bool(claimed and (explicit or hit or (retained and not query)))
    if unmatched and not explicit:
        blocks = False
    return {
        "accepted_as_authority": False,
        "completes_task": False,
        "claimed": claimed,
        "retained_negative_cids": retained,
        "blocks_completion": blocks,
        "reason_code": NEGATIVE_MEMORY_BLOCKS_REUSE if blocks else "",
    }


def negative_memory_blocks_completion(state: Mapping[str, Any] | None) -> bool:
    return bool(negative_memory_view(state)["blocks_completion"])


__all__ = [
    "NEGATIVE_MEMORY_BLOCKS_REUSE",
    "claims_negative_memory",
    "negative_memory_blocks_completion",
    "negative_memory_view",
]
