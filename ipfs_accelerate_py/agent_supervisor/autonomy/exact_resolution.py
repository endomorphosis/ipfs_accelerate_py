"""SAWM W1 on the existing board owner.

Similarity may nominate. It cannot complete a task, idle a board, or
substitute for ``ProgramWorldReuseGate`` exact identity. Missing world
payload is fail-open: SAWM is not required on every wake. TypeSafe is
never this owner.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_receipts import (
    ANN_REASON_CODES,
)

SIMILARITY_NOT_RESOLUTION = "similarity_not_resolution"
EXACT_RESOLUTION_SATISFIED = "exact_resolution_satisfied"
ACCEPTED_TRANSITION_REUSABLE = "accepted_transition_reusable"
TRANSITION_NOT_ACCEPTED = "transition_not_accepted"
SIMILAR_TRANSITION_NOT_EXACT = "similar_transition_not_exact"
_ANN_MARKERS = frozenset({"score", "ann_score", "nearest", "similarity"})
_TRANSITION_KEYS = (
    "accepted_transition",
    "pre_action_post",
    "query_transition",
    "similar_transition",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _candidates(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return ()
    return tuple(value)


def claims_world_resolution(state: Mapping[str, Any] | None) -> bool:
    payload = _mapping(state)
    return bool(
        payload.get("reuse_decision")
        or payload.get("reuse_query")
        or payload.get("similarity_candidates")
        or payload.get("program_world_reuse")
        or payload.get("ann_candidates")
        or any(key in payload for key in _TRANSITION_KEYS)
        or payload.get("pre_cid")
        or payload.get("post_cid")
    )


def _decision_payload(state: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("reuse_decision", "program_world_reuse"):
        nested = state.get(key)
        if isinstance(nested, Mapping):
            return nested
    return state


def _ann_shaped(candidates: Sequence[Any]) -> bool:
    return any(
        isinstance(item, Mapping) and (set(item) & _ANN_MARKERS) for item in candidates
    )


def _triple(source: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(source.get("pre_cid") or source.get("pre") or "").strip(),
        str(source.get("action_cid") or source.get("action") or "").strip(),
        str(source.get("post_cid") or source.get("post") or "").strip(),
    )


def _transition_record(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("accepted_transition", "pre_action_post", "transition"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            return nested
    return payload


def accepted_transition_reusable(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Exact accepted pre/action/post may become reusable state. Similarity may not."""

    payload = _mapping(state)
    accepted = _transition_record(payload)
    query = payload.get("query_transition")
    query_map = query if isinstance(query, Mapping) else accepted
    pre, action, post = _triple(accepted)
    q_pre, q_action, q_post = _triple(query_map)
    complete = bool(pre and action and post and q_pre and q_action and q_post)
    exact = complete and (pre, action, post) == (q_pre, q_action, q_post)
    is_accepted = accepted.get("accepted") is True or payload.get("accepted") is True
    similar = (
        payload.get("similar_transition") is True
        or str(accepted.get("evidence_class") or "").strip()
        in {"vector_candidate", "model_hypothesis", "similarity"}
    )
    if similar and not exact:
        return {
            "claimed": True,
            "exact": False,
            "reusable": False,
            "reason_code": SIMILAR_TRANSITION_NOT_EXACT,
        }
    if complete and not is_accepted:
        return {
            "claimed": True,
            "exact": exact,
            "reusable": False,
            "reason_code": TRANSITION_NOT_ACCEPTED,
        }
    if exact and is_accepted:
        return {
            "claimed": True,
            "exact": True,
            "reusable": True,
            "reason_code": ACCEPTED_TRANSITION_REUSABLE,
        }
    if complete and not exact:
        return {
            "claimed": True,
            "exact": False,
            "reusable": False,
            "reason_code": SIMILARITY_NOT_RESOLUTION,
        }
    return {"claimed": False, "exact": False, "reusable": False, "reason_code": ""}


def exact_resolution_view(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Inspect a wake payload. Never completes a task."""

    payload = _mapping(state)
    claimed = claims_world_resolution(payload)
    decision = _decision_payload(payload)
    candidates = _candidates(
        payload.get("similarity_candidates") or payload.get("ann_candidates")
    )
    exact = decision.get("exact_match", payload.get("exact_match"))
    verdict = str(
        decision.get("verdict") or decision.get("decision") or ""
    ).strip().lower()
    reason = str(decision.get("reason_code") or "").strip()
    nominated = bool(candidates) or reason in ANN_REASON_CODES or reason == (
        "similarity_is_not_reuse"
    )
    exact_hit = exact is True and verdict in {"", "reuse"}
    transition = accepted_transition_reusable(payload)
    if transition["claimed"]:
        claimed = True
        if transition["reusable"]:
            exact_hit = True
            reason = transition["reason_code"]
        else:
            exact_hit = False
            reason = transition["reason_code"] or reason
    blocks = claimed and not exact_hit
    if nominated and not exact_hit:
        blocks = True
        reason = reason or SIMILARITY_NOT_RESOLUTION
    return {
        "accepted_as_authority": False,
        "completes_task": False,
        "claimed": claimed,
        "nominated": nominated or bool(payload.get("similar_transition")),
        "ann_shaped": _ann_shaped(candidates),
        "exact_match": exact_hit,
        "transition_reusable": bool(transition.get("reusable")),
        "blocks_completion": blocks,
        "reason_code": (
            reason
            if exact_hit and reason == ACCEPTED_TRANSITION_REUSABLE
            else (
                EXACT_RESOLUTION_SATISFIED
                if exact_hit
                else (reason or SIMILARITY_NOT_RESOLUTION if blocks else "")
            )
        ),
    }


def similarity_blocks_completion(state: Mapping[str, Any] | None) -> bool:
    return bool(exact_resolution_view(state)["blocks_completion"])


__all__ = [
    "ACCEPTED_TRANSITION_REUSABLE",
    "EXACT_RESOLUTION_SATISFIED",
    "SIMILAR_TRANSITION_NOT_EXACT",
    "SIMILARITY_NOT_RESOLUTION",
    "TRANSITION_NOT_ACCEPTED",
    "accepted_transition_reusable",
    "claims_world_resolution",
    "exact_resolution_view",
    "similarity_blocks_completion",
]
