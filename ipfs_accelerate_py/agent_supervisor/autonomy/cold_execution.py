"""PCTDD W3 on the existing board owner.

Hash memo may nominate reuse. Cold execution is the reference. Repeated work
is avoided only when current inputs, effects, evidence, and publication
checks all permit it. A memo hit without those four checks cannot complete a
task or idle a board. Missing proof-reuse payload is fail-open: PCTDD is not
required on every wake. TypeSafe is never this owner.

This is not a second board. ``ReuseDecision`` SKIP remains the proof-reuse
owner; this module only refuses to treat memo-only hits as board success.
"""

from __future__ import annotations

from typing import Any, Mapping

COLD_EXECUTION_REQUIRED = "cold_execution_required"
MEMO_REUSE_PERMITTED = "memo_reuse_permitted"
COLLECTION_SEED_NOT_PASS = "collection_seed_not_pass"
TEARDOWN_REQUIRED = "teardown_required"
TEARDOWN_FAILED = "teardown_failed"
DEFINITION_IS_NOT_CALL = "fixture_definition_is_not_call"
_PROOF_DECISION_KEYS = ("proof_reuse_decision", "test_reuse_decision")
_STAGES = frozenset(
    {
        "collection_seed",
        "fixture_definition",
        "fixture_instance",
        "execution_key",
        "phase_receipt",
    }
)
_REUSE_KINDS = frozenset(
    {"fixture_definition", "fixture_instance", "call", "whole_item"}
)
_PERMIT_ALIASES = (
    ("current_inputs", "inputs_current"),
    ("current_effects", "effects_current"),
    ("current_evidence", "evidence_current"),
    ("publication_checked", "published"),
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _action(value: Any) -> str:
    text = str(getattr(value, "value", value) or "").strip().lower()
    return text


def _decision_payload(state: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in _PROOF_DECISION_KEYS:
        nested = state.get(key)
        if isinstance(nested, Mapping):
            return nested
    return {}


def claims_proof_reuse(state: Mapping[str, Any] | None) -> bool:
    payload = _mapping(state)
    if any(payload.get(key) for key in _PROOF_DECISION_KEYS):
        return True
    return bool(
        payload.get("hash_memo")
        or payload.get("memo_hit")
        or payload.get("proof_cache_hit")
        or payload.get("cold_execution") is True
        or payload.get("pctdd_stage")
        or payload.get("reuse_kind")
        or payload.get("setup_ran") is True
        or payload.get("teardown_failed") is True
    )


def _flag(source: Mapping[str, Any], *names: str) -> bool:
    return any(source.get(name) is True for name in names)


def _stage(payload: Mapping[str, Any]) -> str:
    text = str(payload.get("pctdd_stage") or payload.get("stage") or "").strip()
    return text if text in _STAGES else ""


def _reuse_kind(payload: Mapping[str, Any]) -> str:
    text = str(payload.get("reuse_kind") or "").strip()
    return text if text in _REUSE_KINDS else ""


def current_reuse_permitted(state: Mapping[str, Any] | None) -> bool:
    """True only when all four current PCTDD checks are explicit True."""

    payload = _mapping(state)
    decision = _decision_payload(payload)
    merged = {**payload, **dict(decision)}
    return all(_flag(merged, *names) for names in _PERMIT_ALIASES)


def cold_execution_view(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Inspect a wake payload. Never completes a task."""

    payload = _mapping(state)
    claimed = claims_proof_reuse(payload)
    decision = _decision_payload(payload)
    action = _action(decision.get("action") or payload.get("action"))
    reason = str(
        getattr(decision.get("reason_code"), "value", decision.get("reason_code"))
        or payload.get("reason_code")
        or ""
    ).strip()
    memo = (
        _flag(
            {**payload, **dict(decision)},
            "hash_memo",
            "memo_hit",
            "proof_cache_hit",
        )
        or reason in {"proof_cache_hit", "hash_memo"}
        or action == "skip"
    )
    cold = payload.get("cold_execution") is True or action == "run"
    merged = {**payload, **dict(decision)}
    permitted = current_reuse_permitted(payload)
    stage = _stage(merged)
    kind = _reuse_kind(merged)
    setup_ran = merged.get("setup_ran") is True
    teardown_ran = merged.get("teardown_ran") is True
    teardown_failed = merged.get("teardown_failed") is True
    whole_item = (
        merged.get("whole_item_pass") is True
        or kind == "whole_item"
        or action == "skip"
    )
    lifecycle = ""
    if stage == "collection_seed" and (memo or whole_item or permitted):
        lifecycle = COLLECTION_SEED_NOT_PASS
        permitted = False
    elif kind == "fixture_definition" and whole_item:
        lifecycle = DEFINITION_IS_NOT_CALL
        permitted = False
    elif teardown_failed:
        lifecycle = TEARDOWN_FAILED
        permitted = False
    elif setup_ran and not teardown_ran:
        lifecycle = TEARDOWN_REQUIRED
        permitted = False
    reuse_ok = bool(permitted and memo and not lifecycle)
    blocks = bool(claimed and (lifecycle or (memo and not permitted)))
    return {
        "accepted_as_authority": False,
        "completes_task": False,
        "claimed": claimed,
        "memo_hit": memo,
        "stage": stage,
        "reuse_kind": kind,
        "cold_reference": bool(cold and not blocks),
        "reuse_permitted": reuse_ok,
        "blocks_completion": blocks,
        "reason_code": (
            lifecycle
            or (
                MEMO_REUSE_PERMITTED
                if reuse_ok
                else (COLD_EXECUTION_REQUIRED if blocks else "")
            )
        ),
    }


def memo_blocks_completion(state: Mapping[str, Any] | None) -> bool:
    return bool(cold_execution_view(state)["blocks_completion"])


__all__ = [
    "COLD_EXECUTION_REQUIRED",
    "COLLECTION_SEED_NOT_PASS",
    "DEFINITION_IS_NOT_CALL",
    "MEMO_REUSE_PERMITTED",
    "TEARDOWN_FAILED",
    "TEARDOWN_REQUIRED",
    "claims_proof_reuse",
    "cold_execution_view",
    "current_reuse_permitted",
    "memo_blocks_completion",
]
