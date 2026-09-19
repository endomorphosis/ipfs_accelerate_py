"""SAWM-016 exact state/transition/procedure/proof reuse gates."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_receipts import (
    OPERATIONAL_ACCEPTANCE_AUTHORITIES,
    ProgramWorldAdmissionError,
    ProgramWorldReuseDecision,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_reuse import (
    ADAPTER_ID,
    CachedReuseEvidence,
    NegativeMemoryRecord,
    ProgramWorldReuseGate,
    ProgramWorldReuseKey,
    REASON_BINDING_MISMATCH,
    REASON_EXACT_HIT,
    REASON_NEGATIVE_MEMORY,
    REASON_PROOF_CACHE,
    REASON_RAW_FALLBACK,
    REASON_SIMILARITY,
    REASON_STALE,
    REASON_TYPED_UNAVAILABLE,
    REASON_UNSAFE_ABSTRACTION,
    evaluate_program_world_reuse,
    explain_program_world_reuse,
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _key(**overrides: object) -> ProgramWorldReuseKey:
    fields: dict[str, object] = {
        "state_cid": _cid("state"),
        "goal_cid": _cid("goal"),
        "policy_cid": _cid("policy"),
        "environment_cid": _cid("env"),
        "toolchain_cid": _cid("toolchain"),
        "procedure_revision_cid": _cid("procedure"),
        "obligation_cids": (_cid("obligation"),),
        "selection_cids": (_cid("selection"),),
        "validation_dependency_cids": (_cid("validation"),),
    }
    fields.update(overrides)
    return ProgramWorldReuseKey(**fields)  # type: ignore[arg-type]


def _gate_with(evidence: CachedReuseEvidence, generation: int = 3) -> ProgramWorldReuseGate:
    gate = ProgramWorldReuseGate(current_generation=generation)
    gate.remember(evidence)
    return gate


def test_exact_hit_is_proposal_until_supervisor_admission() -> None:
    key = _key()
    gate = _gate_with(CachedReuseEvidence(key=key, generation=3))
    decision = evaluate_program_world_reuse(key, gate=gate)
    assert isinstance(decision, ProgramWorldReuseDecision)
    assert decision.verdict == "reuse"
    assert decision.exact_match is True
    assert decision.reason_code == REASON_EXACT_HIT
    assert decision.proposal_only is True
    assert decision.admitted is False
    assert decision.ann_authoritative is False
    assert decision.operational_acceptance_authorities == OPERATIONAL_ACCEPTANCE_AUTHORITIES


def test_single_binding_mismatch_rejects_closed() -> None:
    cached = _key()
    query = _key(goal_cid=_cid("other-goal"))
    gate = _gate_with(CachedReuseEvidence(key=cached, generation=3))
    decision = evaluate_program_world_reuse(query, gate=gate)
    assert decision.verdict == "reject"
    assert decision.reason_code == REASON_BINDING_MISMATCH
    assert decision.exact_match is False
    assert decision.admitted is False
    explanation = explain_program_world_reuse(query, gate=gate)
    assert explanation["mismatches"] == ["goal_cid"]


def test_stale_evidence_is_revoked_and_remembered() -> None:
    key = _key()
    gate = _gate_with(CachedReuseEvidence(key=key, generation=2), generation=3)
    decision = evaluate_program_world_reuse(key, gate=gate)
    assert decision.verdict == "reject"
    assert decision.reason_code == REASON_STALE
    second = evaluate_program_world_reuse(key, gate=gate)
    assert second.reason_code == REASON_NEGATIVE_MEMORY


def test_unsafe_abstraction_fails_closed() -> None:
    key = _key()
    gate = _gate_with(
        CachedReuseEvidence(key=key, generation=3, abstraction_safe=False)
    )
    decision = evaluate_program_world_reuse(key, gate=gate)
    assert decision.verdict == "reject"
    assert decision.reason_code == REASON_UNSAFE_ABSTRACTION
    assert decision.admitted is False


def test_proof_cache_fresh_admission_is_exact_and_not_self_admitted() -> None:
    key = _key()
    gate = _gate_with(
        CachedReuseEvidence(key=key, generation=3, proof_cid=_cid("proof"))
    )
    decision = evaluate_program_world_reuse(
        key,
        gate=gate,
        admission_authority="ipfs_accelerate_py.agent_supervisor.validation.validation_runtime",
        admission_evidence_cid=_cid("validation-evidence"),
    )
    assert decision.verdict == "reuse"
    assert decision.reason_code == REASON_PROOF_CACHE
    assert decision.exact_match is True
    assert decision.admitted is True
    assert decision.proposal_only is False


def test_similarity_only_candidates_cannot_admit_reuse() -> None:
    key = _key()
    gate = _gate_with(CachedReuseEvidence(key=key, generation=3))
    decision = evaluate_program_world_reuse(
        key,
        gate=gate,
        similarity_candidates=({"score": 0.99, "nearest": True},),
    )
    assert decision.verdict == "reject"
    assert decision.reason_code in {REASON_SIMILARITY, "ann_not_authoritative"}
    assert decision.exact_match is False
    assert decision.admitted is False


def test_typed_unavailable_falls_back_closed() -> None:
    key = _key()
    gate = _gate_with(CachedReuseEvidence(key=key, generation=3))
    decision = evaluate_program_world_reuse(key, gate=gate, typed_available=False)
    assert decision.verdict == "unavailable"
    assert decision.reason_code == REASON_TYPED_UNAVAILABLE
    assert decision.fallback is True
    assert decision.admitted is False


def test_raw_source_without_proof_is_not_reuse() -> None:
    key = _key()
    gate = _gate_with(
        CachedReuseEvidence(key=key, generation=3, raw_source=True, proof_cid=None)
    )
    decision = evaluate_program_world_reuse(key, gate=gate)
    assert decision.verdict == "unavailable"
    assert decision.reason_code == REASON_RAW_FALLBACK
    assert decision.fallback is True


def test_negative_memory_invalidation_blocks_later_exact_hits() -> None:
    key = _key()
    gate = ProgramWorldReuseGate(current_generation=3)
    gate.invalidate(
        NegativeMemoryRecord(
            key_cid=key.key_cid,
            reason_code="operator_revoked",
            generation=3,
        )
    )
    gate.remember(CachedReuseEvidence(key=key, generation=3))
    decision = evaluate_program_world_reuse(key, gate=gate)
    assert decision.verdict == "reject"
    assert decision.reason_code == REASON_NEGATIVE_MEMORY
    explanation = explain_program_world_reuse(key, gate=gate)
    assert explanation["negative_memory"] is True


def test_reuse_gate_cannot_self_admit() -> None:
    key = _key()
    gate = _gate_with(CachedReuseEvidence(key=key, generation=3))
    with pytest.raises(ProgramWorldAdmissionError, match="cannot admit"):
        evaluate_program_world_reuse(
            key,
            gate=gate,
            admission_authority=ADAPTER_ID,
            admission_evidence_cid=_cid("self"),
        )
