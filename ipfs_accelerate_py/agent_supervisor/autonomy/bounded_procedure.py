"""SPAR W4 bounded procedures and fixed-point reanalysis on the existing owner.

Verified episodes may support a bounded procedure. An unbounded, unverified,
or skip-reanalysis procedure cannot complete a task. Missing payload is
fail-open. TypeSafe is never this owner.
"""

from __future__ import annotations

from typing import Any, Mapping

UNBOUNDED_PROCEDURE = "unbounded_procedure"
UNVERIFIED_EPISODE = "unverified_episode"
REANALYSIS_REQUIRED = "fixed_point_reanalysis_required"
BOUNDED_PROCEDURE = "bounded_procedure"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _record(state: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("bounded_procedure", "procedure", "verified_episode"):
        nested = state.get(key)
        if isinstance(nested, Mapping):
            return {**state, **nested}
    return state


def claims_bounded_procedure(state: Mapping[str, Any] | None) -> bool:
    payload = _mapping(state)
    if any(
        isinstance(payload.get(key), Mapping)
        for key in ("bounded_procedure", "procedure", "verified_episode")
    ):
        return True
    return bool(
        payload.get("unbounded") is True
        or payload.get("unbounded_procedure") is True
        or payload.get("skip_reanalysis") is True
        or payload.get("fixed_point_reanalysis") is True
        or payload.get("reanalysis") is True
        or payload.get("bounded") is True
    )


def bounded_procedure_view(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Inspect a wake payload. Never completes a task."""

    payload = _mapping(state)
    claimed = claims_bounded_procedure(payload)
    merged = _record(payload)
    unbounded = (
        merged.get("unbounded") is True
        or merged.get("unbounded_procedure") is True
        or merged.get("bound_exhausted") is True
    )
    verified = merged.get("verified_episode") is True or merged.get("verified") is True
    skip = (
        merged.get("skip_reanalysis") is True
        or merged.get("reanalysis") is False
        or merged.get("fixed_point_reanalysis") is False
    )
    claims_fixed = (
        merged.get("fixed_point") is True
        or merged.get("fixed_point_reanalysis") is True
        or merged.get("reanalysis") is True
        or skip
    )
    bounded = merged.get("bounded") is True
    reason = ""
    if claimed and unbounded:
        reason = UNBOUNDED_PROCEDURE
    elif claimed and not verified and (
        bounded
        or merged.get("procedure")
        or payload.get("procedure")
        or payload.get("bounded_procedure")
    ):
        reason = UNVERIFIED_EPISODE
    elif claimed and claims_fixed and skip:
        reason = REANALYSIS_REQUIRED
    elif claimed and bounded and verified and not skip:
        reason = BOUNDED_PROCEDURE
    elif claimed:
        reason = UNBOUNDED_PROCEDURE
    blocks = bool(
        claimed
        and reason
        in {UNBOUNDED_PROCEDURE, UNVERIFIED_EPISODE, REANALYSIS_REQUIRED}
    )
    return {
        "accepted_as_authority": False,
        "completes_task": False,
        "claimed": claimed,
        "bounded": reason == BOUNDED_PROCEDURE,
        "blocks_completion": blocks,
        "reason_code": reason,
    }


def unbounded_procedure_blocks_completion(state: Mapping[str, Any] | None) -> bool:
    return bool(bounded_procedure_view(state)["blocks_completion"])


__all__ = [
    "BOUNDED_PROCEDURE",
    "REANALYSIS_REQUIRED",
    "UNBOUNDED_PROCEDURE",
    "UNVERIFIED_EPISODE",
    "bounded_procedure_view",
    "claims_bounded_procedure",
    "unbounded_procedure_blocks_completion",
]
