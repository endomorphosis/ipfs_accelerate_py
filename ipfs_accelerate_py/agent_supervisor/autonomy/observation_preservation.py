"""SPAR W4 observation preservation on the existing board owner.

Repair and refactor waves must preserve observations under a declared
profile. A worker cannot change rollout. Missing wave payload is fail-open:
SPAR is not required on every wake. TypeSafe is never this owner. This is
not a second board.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

OBSERVATIONS_NOT_PRESERVED = "observations_not_preserved"
OBSERVATIONS_PRESERVED = "observations_preserved"
UNDECLARED_PROFILE = "undeclared_profile"
WORKER_CHANGED_ROLLOUT = "worker_changed_rollout"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _cids(value: Any) -> tuple[str, ...]:
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


def claims_refactor_wave(state: Mapping[str, Any] | None) -> bool:
    payload = _mapping(state)
    nested = payload.get("refactor_wave")
    if isinstance(nested, Mapping) or payload.get("refactor_wave") is True:
        return True
    return bool(
        payload.get("extraction_wave")
        or payload.get("wave_receipt")
        or payload.get("declared_profile")
        or payload.get("observation_cids")
        or payload.get("worker_may_change_mode") is True
    )


def observation_preservation_view(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Inspect a wake payload. Never completes a task."""

    payload = _mapping(state)
    wave = payload.get("refactor_wave")
    wave_map = wave if isinstance(wave, Mapping) else {}
    merged = {**payload, **dict(wave_map)}
    claimed = claims_refactor_wave(payload)
    profile = str(
        merged.get("declared_profile")
        or merged.get("profile_cid")
        or ""
    ).strip()
    before = _cids(
        merged.get("observation_cids") or merged.get("preserved_observation_cids")
    )
    after = _cids(
        merged.get("current_observation_cids")
        or merged.get("wave_observation_cids")
        or merged.get("observation_cids_after")
    )
    if claimed and before and not after:
        after = before
    worker = merged.get("worker_may_change_mode") is True
    mutated = bool(before and after and before != after)
    undeclared = bool(
        claimed
        and (payload.get("refactor_wave") or payload.get("extraction_wave"))
        and not profile
    )
    reason = ""
    if worker:
        reason = WORKER_CHANGED_ROLLOUT
    elif undeclared:
        reason = UNDECLARED_PROFILE
    elif mutated:
        reason = OBSERVATIONS_NOT_PRESERVED
    elif claimed and profile and before and before == after and not worker:
        reason = OBSERVATIONS_PRESERVED
    blocks = bool(claimed and reason in {
        WORKER_CHANGED_ROLLOUT,
        UNDECLARED_PROFILE,
        OBSERVATIONS_NOT_PRESERVED,
    })
    return {
        "accepted_as_authority": False,
        "completes_task": False,
        "claimed": claimed,
        "declared_profile": profile,
        "preserved": reason == OBSERVATIONS_PRESERVED,
        "blocks_completion": blocks,
        "reason_code": reason,
    }


def observations_block_completion(state: Mapping[str, Any] | None) -> bool:
    return bool(observation_preservation_view(state)["blocks_completion"])


__all__ = [
    "OBSERVATIONS_NOT_PRESERVED",
    "OBSERVATIONS_PRESERVED",
    "UNDECLARED_PROFILE",
    "WORKER_CHANGED_ROLLOUT",
    "claims_refactor_wave",
    "observation_preservation_view",
    "observations_block_completion",
]
