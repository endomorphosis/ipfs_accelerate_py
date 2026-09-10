"""SAWM-016 exact state, transition, procedure, and proof reuse-gate tests."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_receipts import (
    ADAPTER_ID,
    EVENT_AUTHORITY,
    MERGE_AUTHORITY,
    OPERATIONAL_ACCEPTANCE_AUTHORITIES,
    VALIDATION_AUTHORITY,
    ProgramWorldAdmissionError,
    ProgramWorldReuseDecision,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_reuse import (
    BINDING_FIELDS,
    BINDING_MISMATCH_REASON,
    EMPTY_COLLECTION_BINDING_TEXT,
    NEGATIVE_MEMORY_RECORD_INTERFACE,
    PROGRAM_WORLD_REUSE_GATE_INTERFACE,
    PROGRAM_WORLD_REUSE_KEY_INTERFACE,
    PROGRAM_WORLD_REUSE_REJECTION_INTERFACE,
    REUSE_GATE_ID,
    SAWM_EXACT_REUSE_EVIDENCE,
    SAWM_REUSE_ADMISSION_EVIDENCE,
    UNBOUND_BINDING_TEXT,
    AbstractionKind,
    BindingComparison,
    NegativeMemoryRecord,
    NegativeMemoryStatus,
    ProgramWorldReuseCandidate,
    ProgramWorldReuseError,
    ProgramWorldReuseEvaluation,
    ProgramWorldReuseExplanation,
    ProgramWorldReuseGate,
    ProgramWorldReuseKey,
    ProgramWorldReuseRejection,
    ReuseEvidenceKind,
    ReuseGateDisposition,
    ReuseReasonCode,
    evaluate_program_world_reuse,
    explain_program_world_reuse,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REUSE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_state/program_world_reuse.py"
)
ADAPTER_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_state/program_world_adapters.py"
)
_OPT_OUTS = {
    "IPFS_DATASETS_AUTO_INSTALL": "0",
    "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
    "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
    "IPFS_KIT_AUTO_INSTALL_DEPS": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _key(**overrides: Any) -> ProgramWorldReuseKey:
    fields: dict[str, Any] = {
        "state_cid": _cid("state-v1"),
        "goal_cid": _cid("goal-v1"),
        "policy_cid": _cid("policy-v1"),
        "environment_cid": _cid("env-v1"),
        "toolchain_cid": _cid("toolchain-v1"),
        "procedure_revision_cid": _cid("procedure-v1"),
        "obligation_cids": (_cid("obl-a"), _cid("obl-b")),
        "selection_cids": (_cid("sel-a"),),
        "validation_dependency_cids": (_cid("val-a"),),
        "scope_cid": _cid("scope-v1"),
        "assumption_cids": (_cid("asm-a"),),
    }
    fields.update(overrides)
    return ProgramWorldReuseKey(**fields)


def _candidate(
    key: ProgramWorldReuseKey | None = None, **overrides: Any
) -> ProgramWorldReuseCandidate:
    fields: dict[str, Any] = {"key": key if key is not None else _key()}
    fields.update(overrides)
    return ProgramWorldReuseCandidate(**fields)


def _proof(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "independently_proof_backed": True,
        "proof_freshly_admitted": True,
        "proof_receipt_cid": _cid("proof-receipt"),
        "proof_authority": VALIDATION_AUTHORITY,
    }
    fields.update(overrides)
    return fields


def _names(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    return functions, classes


# ---------------------------------------------------------------------------
# Interfaces, symbols, cold import, authority boundary
# ---------------------------------------------------------------------------


def test_public_interfaces_and_predicted_symbols_are_present() -> None:
    functions, classes = _names(REUSE_PATH)
    assert "evaluate_program_world_reuse" in functions
    assert "explain_program_world_reuse" in functions
    assert "ProgramWorldReuseKey" in classes
    assert "ProgramWorldReuseGate" in classes
    assert "ProgramWorldReuseRejection" in classes
    assert "NegativeMemoryRecord" in classes
    assert PROGRAM_WORLD_REUSE_GATE_INTERFACE == "ProgramWorldReuseGate@1"
    assert PROGRAM_WORLD_REUSE_KEY_INTERFACE == "ProgramWorldReuseKey@1"
    assert PROGRAM_WORLD_REUSE_REJECTION_INTERFACE == "ProgramWorldReuseRejection@1"
    assert NEGATIVE_MEMORY_RECORD_INTERFACE == "NegativeMemoryRecord@1"
    assert SAWM_EXACT_REUSE_EVIDENCE == "sawm/exact-reuse@1"
    assert SAWM_REUSE_ADMISSION_EVIDENCE == "sawm/reuse-admission@1"
    assert inspect.isfunction(evaluate_program_world_reuse)
    assert inspect.isfunction(explain_program_world_reuse)
    assert OPERATIONAL_ACCEPTANCE_AUTHORITIES == (
        VALIDATION_AUTHORITY,
        MERGE_AUTHORITY,
        EVENT_AUTHORITY,
    )


def test_adapters_still_do_not_define_the_reuse_gate() -> None:
    _functions, classes = _names(ADAPTER_PATH)
    assert "ProgramWorldReuseGate" not in classes
    assert "evaluate_program_world_reuse" not in _functions


def test_import_has_no_io_or_thread_side_effects() -> None:
    before = {thread.name for thread in threading.enumerate()}
    imported = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_reuse"
    )
    after = {thread.name for thread in threading.enumerate()}
    assert after == before
    assert inspect.isfunction(imported.evaluate_program_world_reuse)
    assert inspect.isclass(imported.ProgramWorldReuseGate)


def test_cold_import_loads_neither_datasets_program_world_nor_kit_store() -> None:
    script = r"""
import json
import os
import sys
import threading

effects = []

def forbidden(name):
    def call(*args, **kwargs):
        effects.append(name)
        raise AssertionError(f"forbidden import side effect: {name}")
    return call

os.system = forbidden("os.system")
_orig_start = threading.Thread.start

def _thread_start(self, *args, **kwargs):
    effects.append("threading.Thread.start")
    raise AssertionError("forbidden import side effect: threading.Thread.start")

threading.Thread.start = _thread_start

import ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_reuse as reuse

identity_loaded = any(
    name.startswith(
        "ipfs_datasets_py.logic.software_contracts.semantic_state.program_"
    )
    for name in sys.modules
)
kit_loaded = any("semantic_world_store" in name for name in sys.modules)
cache_loaded = any(
    "verification.receipt_cache" in name or "proof_test_reuse_current_tree_gate" in name
    for name in sys.modules
)
print(json.dumps({
    "effects": effects,
    "identity_loaded": identity_loaded,
    "kit_loaded": kit_loaded,
    "cache_loaded": cache_loaded,
    "interface": reuse.PROGRAM_WORLD_REUSE_GATE_INTERFACE,
    "evaluate": reuse.evaluate_program_world_reuse.__name__,
    "explain": reuse.explain_program_world_reuse.__name__,
}))
"""
    env = dict(os.environ)
    env.update(_OPT_OUTS)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO_ROOT),
            str(REPO_ROOT / "ipfs_datasets_py"),
            str(REPO_ROOT / "ipfs_kit_py"),
            env.get("PYTHONPATH", ""),
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["effects"] == []
    assert payload["identity_loaded"] is False
    assert payload["kit_loaded"] is False
    assert payload["cache_loaded"] is False
    assert payload["interface"] == "ProgramWorldReuseGate@1"
    assert payload["evaluate"] == "evaluate_program_world_reuse"
    assert payload["explain"] == "explain_program_world_reuse"


def test_module_does_not_duplicate_landed_authorities() -> None:
    source = REUSE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
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
        "VerificationReceiptCache",
        "ProofTestReuseCurrentTreeGate",
    }
    assert not (defined & banned)
    assert "pickle" not in source
    assert "subprocess" not in source


# ---------------------------------------------------------------------------
# Exact hit and single-binding-mismatch matrix
# ---------------------------------------------------------------------------


def test_exact_hit_grants_proposal_reuse_until_supervisor_admission() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(current, current)
    assert evaluation.granted is True
    assert evaluation.verdict == ReuseGateDisposition.REUSE.value
    assert evaluation.reason_code == ReuseReasonCode.EXACT_IDENTITY_MATCH.value
    assert evaluation.decision.verdict == "reuse"
    assert evaluation.decision.exact_match is True
    assert evaluation.decision.proposal_only is True
    assert evaluation.decision.admitted is False
    assert evaluation.decision.ann_authoritative is False
    assert evaluation.rejection is None
    assert evaluation.mismatched_bindings == ()
    assert "similarity_is_not_reuse" in evaluation.limitations
    assert "reuse_is_proposal_until_supervisor_admission" in evaluation.limitations
    assert "gate_cannot_admit_own_output" in evaluation.limitations
    round_trip = ProgramWorldReuseEvaluation.from_dict(evaluation.to_dict())
    assert round_trip.evaluation_cid == evaluation.evaluation_cid
    assert ProgramWorldReuseKey.from_dict(current.to_dict()).key_cid == current.key_cid


def test_supervisor_admission_marks_reuse_admitted_without_self_approval() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(
        current,
        current,
        admission_authority=VALIDATION_AUTHORITY,
        admission_evidence_cid=_cid("validation-receipt"),
    )
    assert evaluation.decision.admitted is True
    assert evaluation.decision.proposal_only is False
    assert evaluation.decision.admission_authority == VALIDATION_AUTHORITY
    assert evaluation.decision.operational_acceptance_authorities == (
        OPERATIONAL_ACCEPTANCE_AUTHORITIES
    )


def test_self_admission_is_rejected() -> None:
    current = _key()
    with pytest.raises(ProgramWorldAdmissionError, match="cannot admit"):
        evaluate_program_world_reuse(
            current,
            current,
            admission_authority=REUSE_GATE_ID,
            admission_evidence_cid=_cid("forged"),
        )
    with pytest.raises(ProgramWorldAdmissionError, match="cannot admit"):
        evaluate_program_world_reuse(
            current,
            current,
            admission_authority=ADAPTER_ID,
            admission_evidence_cid=_cid("forged"),
        )


@pytest.mark.parametrize("field", BINDING_FIELDS)
def test_single_binding_mismatch_matrix_rejects(field: str) -> None:
    current = _key()
    if field in {"obligation_cids", "selection_cids", "validation_dependency_cids"}:
        replacement = (_cid(f"{field}-other"),)
    else:
        replacement = _cid(f"{field}-other")
    candidate_key = current.replace(**{field: replacement})
    evaluation = evaluate_program_world_reuse(current, candidate_key)
    assert evaluation.granted is False
    assert evaluation.verdict == ReuseGateDisposition.REJECT.value
    assert evaluation.reason_code == BINDING_MISMATCH_REASON[field]
    assert evaluation.mismatched_bindings == (field,)
    assert evaluation.rejection is not None
    assert evaluation.rejection.reason_code == BINDING_MISMATCH_REASON[field]
    assert field in evaluation.rejection.diagnostic
    explanation = explain_program_world_reuse(evaluation)
    compared = {item.field: item for item in explanation.bindings}
    assert compared[field].equal is False
    for other in BINDING_FIELDS:
        if other == field:
            continue
        assert compared[other].equal is True


def test_multiple_binding_mismatches_use_generic_reason() -> None:
    current = _key()
    candidate = current.replace(state_cid=_cid("state-other"), goal_cid=_cid("goal-other"))
    evaluation = evaluate_program_world_reuse(current, candidate)
    assert evaluation.reason_code == ReuseReasonCode.BINDING_MISMATCH.value
    assert evaluation.mismatched_bindings == ("goal_cid", "state_cid")


def test_no_candidate_abstains_fail_closed() -> None:
    evaluation = evaluate_program_world_reuse(_key(), None)
    assert evaluation.verdict == ReuseGateDisposition.ABSTAIN.value
    assert evaluation.reason_code == ReuseReasonCode.NO_EXACT_IDENTITY_MATCH.value
    assert evaluation.granted is False
    assert evaluation.decision.verdict == "abstain"
    assert evaluation.decision.exact_match is False


# ---------------------------------------------------------------------------
# Stale / freshness, unsafe abstraction, environment, incomplete
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("freshness", "reason"),
    [
        ("stale", ReuseReasonCode.STALE_BINDING.value),
        ("unknown", ReuseReasonCode.FRESHNESS_UNKNOWN.value),
        ("unavailable", ReuseReasonCode.FRESHNESS_UNAVAILABLE.value),
    ],
)
def test_stale_and_unknown_freshness_revokes_reuse(freshness: str, reason: str) -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(
        current, _candidate(current, freshness=freshness)
    )
    assert evaluation.granted is False
    assert evaluation.reason_code == reason
    assert evaluation.verdict == ReuseGateDisposition.REJECT.value
    explanation = explain_program_world_reuse(evaluation)
    assert "freshness" in explanation.diagnostic.lower() or "stale" in explanation.diagnostic.lower()


def test_incomplete_procedure_binding_fails_closed() -> None:
    current = _key(procedure_revision_cid=None)
    assert current.complete is False
    evaluation = evaluate_program_world_reuse(current, current)
    assert evaluation.reason_code == ReuseReasonCode.INCOMPLETE_BINDINGS.value
    assert evaluation.granted is False
    assert evaluation.verdict == ReuseGateDisposition.REJECT.value
    assert evaluation.mismatched_bindings == ("procedure_revision_cid",)
    assert evaluation.rejection is not None
    assert evaluation.rejection.reason_code == ReuseReasonCode.INCOMPLETE_BINDINGS.value
    explanation = explain_program_world_reuse(evaluation)
    assert "complete" in explanation.diagnostic.lower() or "procedure" in explanation.diagnostic.lower()
    compared = {item.field: item for item in explanation.bindings}
    assert compared["procedure_revision_cid"].current_value == UNBOUND_BINDING_TEXT
    assert compared["procedure_revision_cid"].candidate_value == UNBOUND_BINDING_TEXT
    assert compared["procedure_revision_cid"].reason == "incomplete"
    assert tuple(item.field for item in explanation.bindings) == BINDING_FIELDS


def test_incomplete_candidate_procedure_binding_fails_closed() -> None:
    current = _key()
    candidate = _key(procedure_revision_cid=None)
    evaluation = evaluate_program_world_reuse(current, candidate)
    assert evaluation.reason_code == ReuseReasonCode.INCOMPLETE_BINDINGS.value
    assert evaluation.granted is False
    assert evaluation.mismatched_bindings == ("procedure_revision_cid",)
    compared = {item.field: item for item in evaluation.explanation.bindings}
    assert compared["procedure_revision_cid"].current_value == current.procedure_revision_cid
    assert compared["procedure_revision_cid"].candidate_value == UNBOUND_BINDING_TEXT
    assert compared["procedure_revision_cid"].reason == "incomplete"


def test_empty_collections_remain_complete_and_reusable() -> None:
    current = _key(
        obligation_cids=(),
        selection_cids=(),
        validation_dependency_cids=(),
    )
    assert current.complete is True
    evaluation = evaluate_program_world_reuse(current, current)
    assert evaluation.granted is True
    assert evaluation.reason_code == ReuseReasonCode.EXACT_IDENTITY_MATCH.value
    compared = {item.field: item for item in evaluation.explanation.bindings}
    assert compared["obligation_cids"].current_value == EMPTY_COLLECTION_BINDING_TEXT
    assert compared["obligation_cids"].equal is True


def test_unsafe_abstraction_rejects_without_raw_source() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(
        current,
        _candidate(
            current,
            evidence_kind=ReuseEvidenceKind.ABSTRACT_STATE,
            abstraction_kind=AbstractionKind.ABSTRACT,
            abstraction_safe=False,
        ),
    )
    assert evaluation.verdict == ReuseGateDisposition.REJECT.value
    assert evaluation.reason_code == ReuseReasonCode.UNSAFE_ABSTRACTION.value
    assert evaluation.raw_source_cid is None
    assert evaluation.decision.fallback is False


def test_unsafe_abstraction_falls_back_to_raw_source() -> None:
    current = _key()
    raw = _cid("raw-source")
    evaluation = evaluate_program_world_reuse(
        current,
        _candidate(
            current,
            evidence_kind=ReuseEvidenceKind.ABSTRACT_STATE,
            abstraction_kind=AbstractionKind.ABSTRACT,
            abstraction_safe=False,
            raw_source_cid=raw,
        ),
    )
    assert evaluation.verdict == ReuseGateDisposition.RAW_FALLBACK.value
    assert evaluation.reason_code == ReuseReasonCode.UNSAFE_ABSTRACTION.value
    assert evaluation.raw_source_cid == raw
    assert evaluation.decision.verdict == "reject"
    assert evaluation.decision.fallback is True
    assert "raw_source_fallback" in evaluation.limitations
    explanation = explain_program_world_reuse(evaluation)
    assert explanation.raw_source_fallback is True
    assert explanation.raw_source_cid == raw


def test_safe_abstract_state_may_reuse_when_bindings_match() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(
        current,
        _candidate(
            current,
            evidence_kind=ReuseEvidenceKind.ABSTRACT_STATE,
            abstraction_kind=AbstractionKind.ABSTRACT,
            abstraction_safe=True,
        ),
    )
    assert evaluation.granted is True
    assert evaluation.reason_code == ReuseReasonCode.EXACT_IDENTITY_MATCH.value


def test_environment_incompatible_fails_closed_even_on_exact_key() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(
        current, _candidate(current, environment_compatible=False)
    )
    assert evaluation.reason_code == ReuseReasonCode.ENVIRONMENT_INCOMPATIBLE.value
    assert evaluation.granted is False
    assert evaluation.mismatched_bindings == ("environment_cid",)


def test_simulated_evidence_cannot_be_reused_as_live() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(
        current, _candidate(current, simulated=True)
    )
    assert evaluation.reason_code == ReuseReasonCode.SIMULATED_NOT_LIVE.value
    assert evaluation.granted is False


# ---------------------------------------------------------------------------
# Proof-cache fresh admission and proof-backed scoped identity
# ---------------------------------------------------------------------------


def test_proof_cache_hit_without_fresh_admission_is_rejected() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(
        current,
        _candidate(
            current,
            evidence_kind=ReuseEvidenceKind.PROOF_CACHE,
            proof_cache_hit=True,
            proof_freshly_admitted=False,
            proof_receipt_cid=_cid("cached-proof"),
            proof_authority=VALIDATION_AUTHORITY,
        ),
    )
    assert evaluation.granted is False
    assert (
        evaluation.reason_code
        == ReuseReasonCode.PROOF_CACHE_REQUIRES_FRESH_ADMISSION.value
    )
    assert "proof_cache_is_not_authority" in evaluation.limitations
    assert "cache_hit_never_suppresses_current_admission" in evaluation.limitations
    explanation = explain_program_world_reuse(evaluation)
    assert explanation.proof_cache_hit is True
    assert explanation.proof_freshly_admitted is False


def test_proof_cache_hit_with_fresh_independent_admission_reuses() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(
        current,
        _candidate(
            current,
            evidence_kind=ReuseEvidenceKind.PROOF_CACHE,
            **_proof(),
        ),
    )
    assert evaluation.granted is True
    assert evaluation.reason_code == ReuseReasonCode.EXACT_IDENTITY_MATCH.value
    assert evaluation.explanation.proof_cache_hit is True
    assert evaluation.explanation.proof_freshly_admitted is True
    assert evaluation.decision.exact_match is True


def test_proof_cache_cannot_self_admit() -> None:
    current = _key()
    with pytest.raises(ProgramWorldAdmissionError, match="cannot admit or prove"):
        evaluate_program_world_reuse(
            current,
            _candidate(
                current,
                evidence_kind=ReuseEvidenceKind.PROOF_CACHE,
                independently_proof_backed=True,
                proof_freshly_admitted=True,
                proof_receipt_cid=_cid("self-proof"),
                proof_authority=PROGRAM_WORLD_REUSE_GATE_INTERFACE,
            ),
        )


def test_proof_backed_scoped_state_equality_can_grant_reuse() -> None:
    current = _key()
    related = current.replace(state_cid=_cid("state-equivalent"))
    evaluation = evaluate_program_world_reuse(
        current,
        _candidate(
            related,
            evidence_kind=ReuseEvidenceKind.PROOF_BACKED_SCOPE,
            scoped_identity_proved=True,
            relation_claim_cid=_cid("eq-claim"),
            **_proof(),
        ),
    )
    assert evaluation.granted is True
    assert evaluation.reason_code == ReuseReasonCode.PROOF_BACKED_SCOPED_IDENTITY.value
    assert evaluation.decision.exact_match is True
    assert evaluation.decision.state_cid == current.state_cid
    assert "relation_reuse_requires_explicit_scope" in evaluation.limitations


def test_proof_backed_reuse_fails_when_scope_or_assumptions_differ() -> None:
    current = _key()
    scoped = current.replace(scope_cid=_cid("scope-other"))
    evaluation = evaluate_program_world_reuse(
        current,
        _candidate(
            scoped,
            evidence_kind=ReuseEvidenceKind.PROOF_BACKED_SCOPE,
            scoped_identity_proved=True,
            relation_claim_cid=_cid("eq-claim"),
            **_proof(),
        ),
    )
    assert evaluation.reason_code == ReuseReasonCode.RELATION_SCOPE_MISMATCH.value
    assumed = current.replace(assumption_cids=(_cid("asm-other"),))
    evaluation = evaluate_program_world_reuse(
        current,
        _candidate(
            assumed,
            evidence_kind=ReuseEvidenceKind.PROOF_BACKED_SCOPE,
            scoped_identity_proved=True,
            relation_claim_cid=_cid("eq-claim"),
            **_proof(),
        ),
    )
    assert evaluation.reason_code == ReuseReasonCode.SCOPE_ASSUMPTION_MISMATCH.value


def test_proof_backed_state_mismatch_without_scoped_proof_is_rejected() -> None:
    current = _key()
    related = current.replace(state_cid=_cid("state-other"))
    evaluation = evaluate_program_world_reuse(
        current,
        _candidate(related, **_proof()),
    )
    assert evaluation.reason_code == ReuseReasonCode.STATE_MISMATCH.value
    assert evaluation.granted is False


def test_incomplete_proof_backing_fails_closed() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(
        current,
        _candidate(
            current,
            independently_proof_backed=True,
            proof_freshly_admitted=True,
            proof_authority=VALIDATION_AUTHORITY,
        ),
    )
    assert evaluation.reason_code == ReuseReasonCode.PROOF_BACKING_INCOMPLETE.value


# ---------------------------------------------------------------------------
# Similarity-only rejection
# ---------------------------------------------------------------------------


def test_similarity_only_is_context_only_and_never_reuse() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(
        current,
        _candidate(
            current,
            evidence_kind=ReuseEvidenceKind.SIMILARITY,
            similarity_only=True,
        ),
    )
    assert evaluation.granted is False
    assert evaluation.reason_code == ReuseReasonCode.SIMILARITY_IS_NOT_REUSE.value
    assert evaluation.decision.verdict == "reject"
    assert evaluation.decision.exact_match is False
    explanation = explain_program_world_reuse(evaluation)
    assert explanation.similarity_authoritative is False
    assert "context-only" in explanation.diagnostic.lower() or "cannot grant" in explanation.diagnostic.lower()


def test_ann_identity_fields_are_rejected_on_payloads() -> None:
    with pytest.raises(ProgramWorldReuseError, match="ANN/score"):
        ProgramWorldReuseKey.from_mapping(
            {
                "state_cid": _cid("state-v1"),
                "goal_cid": _cid("goal-v1"),
                "policy_cid": _cid("policy-v1"),
                "environment_cid": _cid("env-v1"),
                "toolchain_cid": _cid("toolchain-v1"),
                "procedure_revision_cid": _cid("procedure-v1"),
                "similarity": 0.99,
            }
        )


def test_explanation_forbids_authoritative_similarity() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(current, current)
    with pytest.raises(ProgramWorldReuseError, match="similarity cannot be authoritative"):
        ProgramWorldReuseExplanation(
            verdict=evaluation.verdict,
            reason_code=evaluation.reason_code,
            diagnostic=evaluation.explanation.diagnostic,
            current_key_cid=current.key_cid,
            bindings=evaluation.explanation.bindings,
            similarity_authoritative=True,
            limitations=evaluation.limitations,
        )


# ---------------------------------------------------------------------------
# Typed unavailable
# ---------------------------------------------------------------------------


def test_typed_unavailable_when_capability_missing() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(
        current,
        _candidate(
            current,
            capability_available=False,
            capability_reason_code="import_failed",
            capability_surface="datasets",
        ),
    )
    assert evaluation.verdict == ReuseGateDisposition.UNAVAILABLE.value
    assert evaluation.reason_code == ReuseReasonCode.IMPORT_FAILED.value
    assert evaluation.decision.verdict == "unavailable"
    assert evaluation.decision.fallback is True
    assert "typed_unavailable" in evaluation.limitations


def test_typed_unavailable_uses_generic_code_for_unknown_surface_reason() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(
        current,
        _candidate(
            current,
            capability_available=False,
            capability_reason_code="backend_missing",
        ),
    )
    assert evaluation.reason_code == ReuseReasonCode.CAPABILITY_UNAVAILABLE.value


# ---------------------------------------------------------------------------
# Negative memory record and invalidation
# ---------------------------------------------------------------------------


def test_negative_memory_records_rejection_and_blocks_retry() -> None:
    gate = ProgramWorldReuseGate()
    current = _key()
    stale = _candidate(current, freshness="stale")
    first = gate.evaluate(current, stale, remember=True)
    assert first.reason_code == ReuseReasonCode.STALE_BINDING.value
    assert first.negative_memory is not None
    assert first.negative_memory.active is True
    assert first.explanation.negative_memory_status == NegativeMemoryStatus.RECORDED.value
    memory = NegativeMemoryRecord.from_dict(first.negative_memory.to_dict())
    assert memory.memory_cid == first.negative_memory.memory_cid

    second = gate.evaluate(current, stale, remember=True)
    assert second.reason_code == ReuseReasonCode.NEGATIVE_MEMORY_HIT.value
    assert second.explanation.negative_memory_status == NegativeMemoryStatus.HIT.value
    assert second.negative_memory is not None
    assert second.negative_memory.memory_cid == first.negative_memory.memory_cid


def test_negative_memory_is_invalidated_when_bindings_change() -> None:
    gate = ProgramWorldReuseGate()
    current = _key()
    stale = _candidate(current, freshness="stale")
    first = gate.evaluate(current, stale, remember=True)
    assert first.negative_memory is not None
    changed = current.replace(environment_cid=_cid("env-v2"))
    invalidated = gate.invalidate_negative_memory(changed, reason="binding_changed")
    assert len(invalidated) == 1
    assert invalidated[0].invalidated is True
    assert invalidated[0].invalidation_reason == "binding_changed"
    assert all(not item.active for item in gate.negative_memory)
    retry = gate.evaluate(changed, _candidate(changed, freshness="stale"), remember=True)
    assert retry.reason_code == ReuseReasonCode.STALE_BINDING.value
    assert retry.explanation.negative_memory_status == NegativeMemoryStatus.RECORDED.value


def test_fresh_candidate_does_not_hit_stale_negative_memory() -> None:
    gate = ProgramWorldReuseGate()
    current = _key()
    gate.evaluate(current, _candidate(current, freshness="stale"), remember=True)
    fresh = gate.evaluate(current, _candidate(current, freshness="fresh"))
    assert fresh.granted is True
    assert fresh.reason_code == ReuseReasonCode.EXACT_IDENTITY_MATCH.value
    assert all(not item.active for item in gate.negative_memory)


def test_negative_memory_round_trip_and_active_flag() -> None:
    current = _key()
    record = NegativeMemoryRecord(
        key_cid=current.key_cid,
        candidate_cid=_cid("candidate"),
        reason_code=ReuseReasonCode.STALE_BINDING.value,
        recorded_key=current,
        mismatched_bindings=("environment_cid",),
        scope_cid=current.scope_cid,
    )
    assert record.active is True
    invalidated = record.invalidate("binding_changed")
    assert invalidated.invalidated is True
    assert invalidated.active is False
    rebuilt = NegativeMemoryRecord.from_dict(invalidated.to_dict())
    assert rebuilt.memory_cid == invalidated.memory_cid


# ---------------------------------------------------------------------------
# Explanation completeness and record contracts
# ---------------------------------------------------------------------------


def test_explain_enumerates_every_binding_and_round_trips() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(current, current)
    explanation = explain_program_world_reuse(evaluation)
    assert tuple(item.field for item in explanation.bindings) == BINDING_FIELDS
    assert all(item.equal for item in explanation.bindings)
    rebuilt = ProgramWorldReuseExplanation.from_dict(explanation.to_dict())
    assert rebuilt.explanation_cid == explanation.explanation_cid
    again = explain_program_world_reuse(evaluation.to_dict())
    assert again.explanation_cid == explanation.explanation_cid


def test_rejection_record_round_trip() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(
        current, current.replace(goal_cid=_cid("goal-other"))
    )
    assert evaluation.rejection is not None
    rebuilt = ProgramWorldReuseRejection.from_dict(evaluation.rejection.to_dict())
    assert rebuilt.rejection_cid == evaluation.rejection.rejection_cid
    assert rebuilt.mismatched_bindings == ("goal_cid",)


def test_decision_never_sets_ann_authoritative() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(current, current)
    assert evaluation.decision.ann_authoritative is False
    payload = evaluation.decision.to_dict()
    assert payload["ann_authoritative"] is False
    rebuilt = ProgramWorldReuseDecision.from_dict(payload)
    assert rebuilt.decision_cid == evaluation.decision.decision_cid


def test_key_identity_is_order_insensitive_for_collections() -> None:
    first = _key(
        obligation_cids=(_cid("obl-b"), _cid("obl-a")),
        selection_cids=(_cid("sel-a"),),
    )
    second = _key(
        obligation_cids=(_cid("obl-a"), _cid("obl-b")),
        selection_cids=(_cid("sel-a"),),
    )
    assert first.key_cid == second.key_cid
    assert first.obligation_cids == (_cid("obl-a"), _cid("obl-b"))


def test_duplicate_collection_cids_are_rejected() -> None:
    with pytest.raises(ProgramWorldReuseError, match="duplicates"):
        _key(obligation_cids=(_cid("obl-a"), _cid("obl-a")))


def test_unknown_reason_code_is_rejected() -> None:
    current = _key()
    with pytest.raises(ProgramWorldReuseError, match="reason_code"):
        ProgramWorldReuseRejection(
            reason_code="feels_similar",
            diagnostic="similarity trap",
            current_key_cid=current.key_cid,
        )


def test_gate_evaluate_accepts_mapping_inputs() -> None:
    current = _key()
    evaluation = evaluate_program_world_reuse(
        current.to_dict(),
        {
            "state_cid": current.state_cid,
            "goal_cid": current.goal_cid,
            "policy_cid": current.policy_cid,
            "environment_cid": current.environment_cid,
            "toolchain_cid": current.toolchain_cid,
            "procedure_revision_cid": current.procedure_revision_cid,
            "obligation_cids": current.obligation_cids,
            "selection_cids": current.selection_cids,
            "validation_dependency_cids": current.validation_dependency_cids,
            "scope_cid": current.scope_cid,
            "assumption_cids": current.assumption_cids,
        },
    )
    assert evaluation.granted is True


def test_binding_comparison_schema_is_closed() -> None:
    item = BindingComparison(
        field="state_cid",
        equal=True,
        current_value=_cid("state-v1"),
        candidate_value=_cid("state-v1"),
        reason="equal",
    )
    rebuilt = BindingComparison.from_dict(item.to_dict())
    assert rebuilt.field == "state_cid"
    with pytest.raises(ProgramWorldReuseError):
        BindingComparison.from_dict({**item.to_dict(), "score": 1})
