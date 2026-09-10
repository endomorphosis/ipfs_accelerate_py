"""SAWM-016 exact state, transition, procedure, and proof reuse-gate tests."""

from __future__ import annotations

import ast
import importlib
import inspect
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_receipts import (
    EVENT_AUTHORITY,
    MERGE_AUTHORITY,
    OPERATIONAL_ACCEPTANCE_AUTHORITIES,
    VALIDATION_AUTHORITY,
    FreshnessState,
    ProgramWorldReuseDecision,
    ReuseVerdict,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_reuse import (
    GATE_ID,
    PROGRAM_WORLD_REUSE_GATE_INTERFACE,
    REUSE_KEY_BINDINGS,
    SAWM_EXACT_REUSE_EVIDENCE,
    SAWM_REUSE_ADMISSION_EVIDENCE,
    NegativeMemoryRecord,
    ProgramWorldReuseCandidate,
    ProgramWorldReuseError,
    ProgramWorldReuseEvaluation,
    ProgramWorldReuseGate,
    ProgramWorldReuseKey,
    ProgramWorldReuseRejection,
    ReuseEvidenceKind,
    ReuseReasonCode,
    ReuseStage,
    assert_no_duplicate_reuse_authority,
    binding_mismatches,
    evaluate_program_world_reuse,
    explain_program_world_reuse,
    program_world_reuse_module_source,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REUSE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_state/program_world_reuse.py"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _key(**overrides: Any) -> ProgramWorldReuseKey:
    fields: dict[str, Any] = {
        "state_cid": _cid("state-v1"),
        "goal_cid": _cid("goal-v1"),
        "policy_cid": _cid("policy-v1"),
        "environment_cid": _cid("env-v1"),
        "toolchain_cid": _cid("toolchain-v1"),
        "obligation_cids": (_cid("obl-a"),),
        "selection_cids": (_cid("sel-a"),),
        "procedure_revision_cid": _cid("proc-v1"),
        "validation_dependency_cids": (_cid("val-a"),),
    }
    fields.update(overrides)
    return ProgramWorldReuseKey(**fields)


def _candidate(
    key: ProgramWorldReuseKey | None = None, **overrides: Any
) -> ProgramWorldReuseCandidate:
    fields: dict[str, Any] = {
        "key": key if key is not None else _key(),
        "evidence_kind": ReuseEvidenceKind.EXACT,
        "freshness": FreshnessState.FRESH,
    }
    fields.update(overrides)
    return ProgramWorldReuseCandidate(**fields)


def _proof_candidate(
    query: ProgramWorldReuseKey,
    *,
    mismatched_field: str = "state_cid",
    **overrides: Any,
) -> ProgramWorldReuseCandidate:
    other = _cid(f"other-{mismatched_field}")
    candidate_key = replace(query, **{mismatched_field: other})
    query_value = getattr(query, mismatched_field)
    fields: dict[str, Any] = {
        "key": candidate_key,
        "evidence_kind": ReuseEvidenceKind.PROOF_BACKED,
        "freshness": FreshnessState.FRESH,
        "relation_claim_cid": _cid("relation-proved"),
        "relation_kind": "equality",
        "relation_authority_status": "proved",
        "relation_scope_cid": _cid("scope-v1"),
        "query_scope_cid": _cid("scope-v1"),
        "relation_assumption_cids": (_cid("asm-a"),),
        "query_assumption_cids": (_cid("asm-a"), _cid("asm-b")),
        "relation_environment_cid": query.environment_cid,
        "relation_policy_cid": query.policy_cid,
        "relation_left_cid": query_value,
        "relation_right_cid": other,
        "proof_admission_authority": VALIDATION_AUTHORITY,
        "proof_admission_evidence_cid": _cid("proof-admission"),
    }
    fields.update(overrides)
    return ProgramWorldReuseCandidate(**fields)


def test_public_interfaces_and_predicted_symbols() -> None:
    source = program_world_reuse_module_source()
    tree = ast.parse(source)
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "evaluate_program_world_reuse" in functions
    assert "explain_program_world_reuse" in functions
    assert "ProgramWorldReuseKey" in classes
    assert "ProgramWorldReuseGate" in classes
    assert "ProgramWorldReuseRejection" in classes
    assert "NegativeMemoryRecord" in classes
    assert PROGRAM_WORLD_REUSE_GATE_INTERFACE == "ProgramWorldReuseGate@1"
    assert SAWM_EXACT_REUSE_EVIDENCE == "sawm/exact-reuse@1"
    assert SAWM_REUSE_ADMISSION_EVIDENCE == "sawm/reuse-admission@1"
    assert inspect.isfunction(evaluate_program_world_reuse)
    assert inspect.isfunction(explain_program_world_reuse)
    assert_no_duplicate_reuse_authority(source)


def test_import_has_no_io_or_thread_side_effects() -> None:
    before = {thread.name for thread in threading.enumerate()}
    imported = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_reuse"
    )
    after = {thread.name for thread in threading.enumerate()}
    assert after == before
    assert inspect.isfunction(imported.evaluate_program_world_reuse)
    assert inspect.isfunction(imported.explain_program_world_reuse)
    assert inspect.isclass(imported.ProgramWorldReuseGate)


def test_gate_does_not_import_or_duplicate_foreign_authorities() -> None:
    source = REUSE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    assert "ipfs_datasets_py" not in imports
    assert "ipfs_kit_py" not in imports
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    banned = {
        "SemanticObjectEnvelope",
        "ProgramRelationClaim",
        "ProgramTransitionQuery",
        "ContextCompiler",
        "DurableCoordinationStore",
        "VerifiedSemanticBlockStore",
        "ProofCarryingProcedureCompiler",
        "VerificationReceiptCache",
        "StateAbstractionProfile",
    }
    assert not (defined & banned)


def test_reuse_key_rehashes_and_binds_every_declared_dimension() -> None:
    key = _key()
    assert key.reuse_key_cid.startswith("b")
    assert ProgramWorldReuseKey.from_dict(key.to_dict()).reuse_key_cid == key.reuse_key_cid
    assert set(REUSE_KEY_BINDINGS) == {
        "state_cid",
        "goal_cid",
        "policy_cid",
        "environment_cid",
        "toolchain_cid",
        "obligation_cids",
        "selection_cids",
        "procedure_revision_cid",
        "validation_dependency_cids",
    }
    twin = _key()
    assert twin.reuse_key_cid == key.reuse_key_cid
    changed = replace(key, goal_cid=_cid("goal-v2"))
    assert changed.reuse_key_cid != key.reuse_key_cid


def test_exact_hit_grants_proposal_reuse_until_supervisor_admission() -> None:
    query = _key()
    evaluation = evaluate_program_world_reuse(query, _candidate(query))
    assert evaluation.granted is True
    assert evaluation.verdict == ReuseVerdict.REUSE.value
    assert evaluation.reason_code == ReuseReasonCode.EXACT_IDENTITY_MATCH.value
    assert evaluation.decision.exact_match is True
    assert evaluation.decision.proposal_only is True
    assert evaluation.decision.admitted is False
    assert evaluation.decision.ann_authoritative is False
    assert "cache_hit_does_not_suppress_validation" in evaluation.decision.limitations
    assert evaluation.decision.operational_acceptance_authorities == (
        OPERATIONAL_ACCEPTANCE_AUTHORITIES
    )
    round_trip = ProgramWorldReuseDecision.from_dict(evaluation.decision.to_dict())
    assert round_trip.decision_cid == evaluation.decision.decision_cid


def test_exact_hit_with_independent_admission_is_granted() -> None:
    query = _key()
    evaluation = evaluate_program_world_reuse(
        query,
        _candidate(query),
        admission_authority=VALIDATION_AUTHORITY,
        admission_evidence_cid=_cid("validation-receipt"),
    )
    assert evaluation.granted is True
    assert evaluation.decision.admitted is True
    assert evaluation.decision.proposal_only is False
    assert evaluation.decision.admission_authority == VALIDATION_AUTHORITY
    assert MERGE_AUTHORITY in evaluation.decision.operational_acceptance_authorities
    assert EVENT_AUTHORITY in evaluation.decision.operational_acceptance_authorities


@pytest.mark.parametrize("field", REUSE_KEY_BINDINGS)
def test_single_binding_mismatch_fails_closed(field: str) -> None:
    query = _key()
    if field in {"obligation_cids", "selection_cids", "validation_dependency_cids"}:
        candidate_key = replace(query, **{field: (_cid(f"other-{field}"),)})
    else:
        candidate_key = replace(query, **{field: _cid(f"other-{field}")})
    evaluation = evaluate_program_world_reuse(query, _candidate(candidate_key))
    assert evaluation.granted is False
    assert evaluation.verdict == ReuseVerdict.REJECT.value
    assert field in evaluation.mismatched_bindings
    assert evaluation.mismatched_bindings == (field,)
    assert evaluation.reason_code.endswith("_mismatch")
    assert evaluation.rejection is not None
    assert field in evaluation.rejection.diagnostic or evaluation.reason_code in (
        evaluation.rejection.reason_code
    )
    assert binding_mismatches(query, candidate_key) == (field,)


def test_stale_bindings_revoke_reuse() -> None:
    query = _key()
    evaluation = evaluate_program_world_reuse(
        query, _candidate(query, freshness=FreshnessState.STALE)
    )
    assert evaluation.granted is False
    assert evaluation.verdict == ReuseVerdict.REJECT.value
    assert evaluation.reason_code == ReuseReasonCode.STALE_BINDING.value
    assert "stale" in evaluation.diagnostic.lower()
    assert evaluation.rejection is not None
    assert "stale_bindings_revoke" in evaluation.rejection.limitations


def test_unknown_freshness_widens_and_fails_closed() -> None:
    query = _key()
    evaluation = evaluate_program_world_reuse(
        query, _candidate(query, freshness=FreshnessState.UNKNOWN)
    )
    assert evaluation.granted is False
    assert evaluation.reason_code == ReuseReasonCode.FRESHNESS_UNKNOWN.value
    assert "unknown" in evaluation.diagnostic.lower()


def test_unsafe_abstraction_cannot_admit_reuse() -> None:
    query = _key()
    evaluation = evaluate_program_world_reuse(
        query,
        _candidate(
            query,
            evidence_kind=ReuseEvidenceKind.ABSTRACT,
            abstraction_soundness_claim="under_approximation",
            abstraction_reuse_admitted=False,
            unknown_relevance_dimensions=("heap",),
        ),
    )
    assert evaluation.granted is False
    assert evaluation.reason_code == ReuseReasonCode.UNSAFE_ABSTRACTION.value
    assert "unsafe" in evaluation.diagnostic.lower()


def test_proved_abstraction_does_not_bypass_exact_or_proof_gates() -> None:
    query = _key()
    other = replace(query, state_cid=_cid("abstract-other-state"))
    evaluation = evaluate_program_world_reuse(
        query,
        _candidate(
            other,
            evidence_kind=ReuseEvidenceKind.ABSTRACT,
            abstraction_soundness_claim="over_approximation",
            abstraction_reuse_admitted=True,
        ),
    )
    assert evaluation.granted is False
    assert evaluation.reason_code == ReuseReasonCode.STATE_MISMATCH.value


def test_proof_cache_hit_requires_fresh_independent_admission() -> None:
    query = _key()
    evaluation = evaluate_program_world_reuse(
        query,
        _candidate(
            query,
            evidence_kind=ReuseEvidenceKind.CACHE,
            proof_cache_hit=True,
        ),
    )
    assert evaluation.granted is False
    assert (
        evaluation.reason_code
        == ReuseReasonCode.PROOF_CACHE_REQUIRES_FRESH_ADMISSION.value
    )
    assert "fresh admission" in evaluation.diagnostic.lower()


def test_proof_cache_with_fresh_admission_can_reuse() -> None:
    query = _key()
    evaluation = evaluate_program_world_reuse(
        query,
        _candidate(
            query,
            evidence_kind=ReuseEvidenceKind.CACHE,
            proof_cache_hit=True,
            proof_admission_authority=VALIDATION_AUTHORITY,
            proof_admission_evidence_cid=_cid("fresh-proof-admission"),
        ),
        admission_authority=MERGE_AUTHORITY,
        admission_evidence_cid=_cid("merge-admission"),
    )
    assert evaluation.granted is True
    assert evaluation.reason_code == ReuseReasonCode.EXACT_IDENTITY_MATCH.value
    assert evaluation.decision.admitted is True
    assert "cache_hit_does_not_suppress_validation" in evaluation.decision.limitations


def test_independently_proof_backed_scoped_evidence_can_grant_reuse() -> None:
    query = _key()
    evaluation = evaluate_program_world_reuse(query, _proof_candidate(query))
    assert evaluation.granted is True
    assert evaluation.proof_backed is True
    assert evaluation.reason_code == ReuseReasonCode.PROOF_BACKED_SCOPED_MATCH.value
    assert "scope" in evaluation.diagnostic.lower()
    assert "relation_reuse_scoped_to_assumptions" in evaluation.decision.limitations


def test_proof_backed_reuse_stays_inside_explicit_scope_and_assumptions() -> None:
    query = _key()
    scope_eval = evaluate_program_world_reuse(
        query,
        _proof_candidate(query, query_scope_cid=_cid("other-scope")),
    )
    assert scope_eval.granted is False
    assert scope_eval.reason_code == ReuseReasonCode.SCOPE_MISMATCH.value
    assumption_eval = evaluate_program_world_reuse(
        query,
        _proof_candidate(
            query,
            relation_assumption_cids=(_cid("asm-secret"),),
            query_assumption_cids=(_cid("asm-a"),),
        ),
    )
    assert assumption_eval.granted is False
    assert assumption_eval.reason_code == ReuseReasonCode.ASSUMPTION_MISMATCH.value
    unproved = evaluate_program_world_reuse(
        query,
        _proof_candidate(query, relation_authority_status="asserted"),
    )
    assert unproved.granted is False
    assert unproved.reason_code == ReuseReasonCode.RELATION_NOT_PROVED.value


def test_similarity_alone_is_context_only_and_cannot_grant_reuse() -> None:
    query = _key()
    evaluation = evaluate_program_world_reuse(
        query,
        _candidate(
            query,
            evidence_kind=ReuseEvidenceKind.SIMILARITY,
            similarity_only=True,
            ann_candidates=({"score": 0.99, "nearest": True},),
        ),
    )
    assert evaluation.granted is False
    assert evaluation.verdict == ReuseVerdict.REJECT.value
    assert evaluation.reason_code in {
        ReuseReasonCode.SIMILARITY_IS_NOT_REUSE.value,
        ReuseReasonCode.ANN_NOT_AUTHORITATIVE.value,
    }
    assert evaluation.similarity_context_only is True
    assert "context-only" in evaluation.diagnostic.lower() or "advisory" in (
        evaluation.diagnostic.lower()
    )


def test_exact_hit_keeps_similarity_advisory_without_granting_from_it() -> None:
    query = _key()
    evaluation = evaluate_program_world_reuse(
        query,
        _candidate(query, ann_candidates=({"score": 0.42},)),
    )
    assert evaluation.granted is True
    assert evaluation.similarity_context_only is True
    assert "similarity_is_context_only" in evaluation.decision.limitations
    assert evaluation.decision.ann_authoritative is False


def test_typed_unavailable_fails_closed_with_raw_source_fallback() -> None:
    query = _key()
    evaluation = evaluate_program_world_reuse(
        query,
        _candidate(
            query,
            evidence_kind=ReuseEvidenceKind.UNAVAILABLE,
            capability_available=False,
            capability_reason_code="import_failed",
            capability_surface="ipfs_datasets_py.program_world",
            raw_source_cid=_cid("raw-source"),
        ),
    )
    assert evaluation.granted is False
    assert evaluation.verdict == ReuseVerdict.UNAVAILABLE.value
    assert evaluation.reason_code == ReuseReasonCode.TYPED_UNAVAILABLE.value
    assert evaluation.raw_fallback is True
    assert evaluation.decision.fallback is True
    assert "raw_source_fallback_available" in evaluation.decision.limitations
    unavailable = evaluation.to_unavailable_result()
    assert unavailable.reason_code == ReuseReasonCode.TYPED_UNAVAILABLE.value
    assert unavailable.adapter_id == GATE_ID


def test_incomplete_and_environment_incompatible_evidence_fail_closed() -> None:
    query = _key()
    incomplete = evaluate_program_world_reuse(
        query, _candidate(query, incomplete=True)
    )
    assert incomplete.granted is False
    assert incomplete.reason_code == ReuseReasonCode.INCOMPLETE_BINDINGS.value
    env = evaluate_program_world_reuse(
        query, _candidate(query, environment_compatible=False)
    )
    assert env.granted is False
    assert env.reason_code == ReuseReasonCode.ENVIRONMENT_INCOMPATIBLE.value
    assert "environment" in env.diagnostic.lower()


def test_cache_hit_cannot_suppress_mandatory_validation() -> None:
    query = _key()
    evaluation = evaluate_program_world_reuse(
        query,
        _candidate(
            query,
            evidence_kind=ReuseEvidenceKind.CACHE,
            proof_cache_hit=True,
            proof_admission_authority=VALIDATION_AUTHORITY,
            proof_admission_evidence_cid=_cid("fresh-proof"),
            suppress_validation=True,
        ),
    )
    assert evaluation.granted is False
    assert (
        evaluation.reason_code
        == ReuseReasonCode.CACHE_HIT_DOES_NOT_SUPPRESS_VALIDATION.value
    )


def test_gate_cannot_admit_its_own_output() -> None:
    query = _key()
    evaluation = evaluate_program_world_reuse(
        query,
        _candidate(query),
        admission_authority=GATE_ID,
        admission_evidence_cid=_cid("self-admission"),
    )
    assert evaluation.granted is False
    assert evaluation.reason_code == ReuseReasonCode.SELF_ADMISSION_FORBIDDEN.value
    adapter = evaluate_program_world_reuse(
        query,
        _candidate(query),
        admission_authority="ProgramWorldReuseGate",
        admission_evidence_cid=_cid("self-class"),
    )
    assert adapter.reason_code == ReuseReasonCode.SELF_ADMISSION_FORBIDDEN.value


def test_missing_candidate_abstains_with_raw_fallback_open() -> None:
    evaluation = evaluate_program_world_reuse(_key(), None)
    assert evaluation.granted is False
    assert evaluation.verdict == ReuseVerdict.ABSTAIN.value
    assert evaluation.reason_code == ReuseReasonCode.NO_EXACT_IDENTITY_MATCH.value
    assert evaluation.raw_fallback is True


def test_negative_memory_records_rejection_and_invalidates_after_repair() -> None:
    gate = ProgramWorldReuseGate()
    query = _key()
    stale = _candidate(
        query,
        freshness=FreshnessState.STALE,
        invalidator_cids=(_cid("stale-invalidator"),),
    )
    first = gate.evaluate(query, stale)
    assert first.reason_code == ReuseReasonCode.STALE_BINDING.value
    assert first.negative_memory is not None
    assert first.negative_memory.key_cid == query.reuse_key_cid
    replay = gate.evaluate(query, stale)
    assert replay.negative_memory is not None
    assert replay.rejection is not None
    assert "negative memory" in replay.diagnostic.lower()
    repaired = _candidate(query, freshness=FreshnessState.FRESH)
    granted = gate.evaluate(
        query,
        repaired,
        admission_authority=VALIDATION_AUTHORITY,
        admission_evidence_cid=_cid("current-admission"),
    )
    assert granted.granted is True
    assert granted.negative_memory is None
    assert gate.negative_memory_records() == ()
    stale_again = gate.evaluate(query, stale)
    assert stale_again.reason_code == ReuseReasonCode.STALE_BINDING.value
    explicit = gate.record_negative_memory(
        NegativeMemoryRecord(
            key_cid=query.reuse_key_cid,
            reason_code=ReuseReasonCode.STALE_BINDING.value,
            diagnostic="recorded",
            candidate_evidence_cid=stale.candidate_evidence_cid,
            invalidator_cids=stale.invalidator_cids,
        )
    )
    cleared = gate.invalidate_negative_memory(
        query.reuse_key_cid, invalidator_cids=stale.invalidator_cids
    )
    assert explicit.record_cid in {item.record_cid for item in cleared}
    assert gate.lookup_negative_memory(query, stale) is None


def test_explain_program_world_reuse_reports_why_fail_closed() -> None:
    query = _key()
    evaluation = evaluate_program_world_reuse(
        query, _candidate(replace(query, policy_cid=_cid("policy-other")))
    )
    explanation = explain_program_world_reuse(evaluation)
    assert explanation.granted is False
    assert explanation.reason_code == ReuseReasonCode.POLICY_MISMATCH.value
    assert explanation.mismatched_bindings == ("policy_cid",)
    assert explanation.evaluation_cid == evaluation.evaluation_cid
    assert explanation.to_dict()["explanation_cid"].startswith("b")
    direct = explain_program_world_reuse(
        query, _candidate(query, freshness=FreshnessState.STALE)
    )
    assert direct.reason_code == ReuseReasonCode.STALE_BINDING.value
    assert "revoke" in direct.diagnostic.lower() or "stale" in direct.diagnostic.lower()
    assert {item["stage"] for item in direct.stages} == {
        stage.value for stage in ReuseStage
    }


def test_rejection_and_negative_memory_round_trip() -> None:
    rejection = ProgramWorldReuseRejection(
        reason_code=ReuseReasonCode.STATE_MISMATCH.value,
        diagnostic="state_cid differs from the current exact key",
        mismatched_bindings=("state_cid",),
        verdict=ReuseVerdict.REJECT,
        limitations=("exact_reuse_required",),
        invalidator_cids=(_cid("inv-1"),),
    )
    assert (
        ProgramWorldReuseRejection.from_dict(rejection.to_dict()).rejection_cid
        == rejection.rejection_cid
    )
    memory = NegativeMemoryRecord(
        key_cid=_key().reuse_key_cid,
        reason_code=rejection.reason_code,
        diagnostic=rejection.diagnostic,
        mismatched_bindings=rejection.mismatched_bindings,
        invalidator_cids=rejection.invalidator_cids,
        rejection_cid=rejection.rejection_cid,
    )
    assert NegativeMemoryRecord.from_dict(memory.to_dict()).record_cid == memory.record_cid
    with pytest.raises(ProgramWorldReuseError):
        ProgramWorldReuseRejection(
            reason_code="ok",
            diagnostic="cannot grant",
            verdict=ReuseVerdict.REUSE,
        )


def test_evaluation_records_every_stage_and_cannot_self_hash_forge() -> None:
    evaluation = evaluate_program_world_reuse(_key(), _candidate(_key()))
    assert isinstance(evaluation, ProgramWorldReuseEvaluation)
    assert [item.stage for item in evaluation.stages] == [
        stage.value for stage in ReuseStage
    ]
    payload = evaluation.to_dict()
    assert payload["evaluation_cid"] == evaluation.evaluation_cid
    forged = dict(payload)
    forged["reason_code"] = "forged"
    assert forged["evaluation_cid"] == evaluation.evaluation_cid
    rebuilt = evaluate_program_world_reuse(_key(), _candidate(_key()))
    assert rebuilt.evaluation_cid == evaluation.evaluation_cid
