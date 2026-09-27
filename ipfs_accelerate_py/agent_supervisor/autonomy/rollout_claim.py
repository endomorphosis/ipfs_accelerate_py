"""H.2 / SPAR rollout claims on the existing board owner.

Required cannot be inferred from bootstrap, shadow, or an unobserved mode.
A wake that claims guarded/required above the observed SPAR mode cannot
complete a task. Missing claim is fail-open. TypeSafe is never this owner.
"""

from __future__ import annotations

from typing import Any, Mapping

from .qualification import observe_rollout

ROLLOUT_NOT_REQUIRED = "rollout_not_required"
ROLLOUT_RESPECTED = "rollout_respected"
_RANK = {"off": 0, "shadow": 1, "guarded": 2, "required": 3}
_CLAIMS = frozenset(_RANK)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _claimed_level(state: Mapping[str, Any]) -> str:
    nested = state.get("rollout_claim")
    payload = nested if isinstance(nested, Mapping) else state
    if payload.get("rollout_required") is True or payload.get("required_rollout") is True:
        return "required"
    text = str(payload.get("claimed_rollout") or payload.get("rollout") or "").strip()
    if text in _CLAIMS:
        return text
    return ""


def claims_rollout(state: Mapping[str, Any] | None) -> bool:
    payload = _mapping(state)
    if isinstance(payload.get("rollout_claim"), Mapping):
        return True
    return bool(_claimed_level(payload))


def rollout_claim_view(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Inspect a wake payload. Never completes a task. Never infers required."""

    payload = _mapping(state)
    claimed = claims_rollout(payload)
    want = _claimed_level(payload)
    observed = observe_rollout(payload) or "off"
    overclaimed = bool(
        claimed
        and want
        and _RANK.get(want, 0) > _RANK.get(observed, 0)
    )
    reason = ""
    if overclaimed:
        reason = ROLLOUT_NOT_REQUIRED
    elif claimed and want:
        reason = ROLLOUT_RESPECTED
    return {
        "accepted_as_authority": False,
        "completes_task": False,
        "claimed": claimed,
        "claimed_rollout": want,
        "observed_rollout": observed if claimed else "",
        "respected": reason == ROLLOUT_RESPECTED,
        "blocks_completion": overclaimed,
        "reason_code": reason,
    }


def rollout_claim_blocks_completion(state: Mapping[str, Any] | None) -> bool:
    return bool(rollout_claim_view(state)["blocks_completion"])


__all__ = [
    "ROLLOUT_NOT_REQUIRED",
    "ROLLOUT_RESPECTED",
    "claims_rollout",
    "rollout_claim_blocks_completion",
    "rollout_claim_view",
]
