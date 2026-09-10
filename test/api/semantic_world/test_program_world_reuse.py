"""SAWM-016 exact state, transition, procedure, and proof reuse-gate tests."""

from __future__ import annotations

import ast
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
    EVENT_AUTHORITY,
    MERGE_AUTHORITY,
    OPERATIONAL_ACCEPTANCE_AUTHORITIES,
    PROGRAM_WORLD_REUSE_DECISION_INTERFACE,
    VALIDATION_AUTHORITY,
    ProgramWorldAdmissionError,
    ProgramWorldReuseDecision,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_reuse import (
    DATASETS_ABSTRACTION_AUTHORITY,
    DATASETS_RELATION_AUTHORITY,
    EXACT_BINDING_FIELDS,
    GATE_ID,
    GATE_OWNED_AUTHORITIES,
    LOCATED_LANDED_AUTHORITIES,
    NEGATIVE_MEMORY_RECORD_INTERFACE,
    PROGRAM_WORLD_REUSE_GATE_INTERFACE,
    PROGRAM_WORLD_REUSE_KEY_INTERFACE,
    PROGRAM_WORLD_REUSE_REJECTION_INTERFACE,
    PROOF_BACKED_SUBSTITUTABLE_BINDINGS,
    RECEIPT_CACHE_AUTHORITY,
    SAWM_EXACT_REUSE_EVIDENCE,
    SAWM_REUSE_ADMISSION_EVIDENCE,
    TEST_PROOF_CACHE_AUTHORITY,
    NegativeMemoryRecord,
    ProgramWorldReuseError,
    ProgramWorldReuseEvaluation,
    ProgramWorldReuseExplanation,
    ProgramWorldReuseGate,
    ProgramWorldReuseKey,
    ProgramWorldReuseRejection,
    ProgramWorldReuseRequest,
    ReuseEvidenceKind,
    ReuseReasonCode,
    assert_no_duplicate_reuse_authority,
    evaluate_program_world_reuse,
    explain_program_world_reuse,
    load_program_world_reuse_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REUSE_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/semantic_state/program_world_reuse.py"
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
        "obligation_cids": (_cid("obligation-a"),),
        "selection_cids": (_cid("selection-a"),),
        "procedure_revision_cid": _cid("procedure-rev-v1"),
        "validation_dependency_cids": (_cid("validation-dep-a"),),
    }
    fields.update(overrides)
    return ProgramWorldReuseKey(**fields)


def _request(**overrides: Any) -> ProgramWorldReuseRequest:
    key = overrides.pop("key", None) or _key()
    if "candidate_key" not in overrides:
        overrides["candidate_key"] = key
    return ProgramWorldReuseRequest(key=key, **overrides)


def _proof_backed_request(**overrides: Any) -> ProgramWorldReuseRequest:
    key = overrides.pop("key", None) or _key()
    candidate = overrides.pop("candidate_key", key.replace(state_cid=_cid("state-equiv")))
    fields: dict[str, Any] = {
        "key": key,
        "candidate_key": candidate,
        "evidence_kind": ReuseEvidenceKind.PROOF_BACKED,
        "independently_proof_backed": True,
        "proof_admitted": True,
        "proof_admission_authority": DATASETS_RELATION_AUTHORITY,
        "proof_admission_evidence_cid": _cid("proof-admission-v1"),
        "proof_cache_hit": True,
        "proof_cache_fresh": True,
        "relation_claim_cid": _cid("relation-eq"),
        "relation_scope_cid": _cid("scope-v1"),
        "relation_kind": "equality",
        "relation_status": "proved",
        "relation_environment_cid": key.environment_cid,
        "relation_policy_cid": key.policy_cid,
        "relation_subject_cids": (key.state_cid, candidate.state_cid),
        "relation_assumptions_hold": True,
    }
    fields.update(overrides)
    return ProgramWorldReuseRequest(**fields)


# ---------------------------------------------------------------------------
# Interfaces, symbols, AST boundary
# ---------------------------------------------------------------------------


def test_public_interfaces_and_predicted_symbols() -> None:
    tree = ast.parse(REUSE_PATH.read_text(encoding="utf-8"))
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
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
    assert PROGRAM_WORLD_REUSE_DECISION_INTERFACE == "ProgramWorldReuseDecision@1"
    assert SAWM_EXACT_REUSE_EVIDENCE == "sawm/exact-reuse@1"
    assert SAWM_REUSE_ADMISSION_EVIDENCE == "sawm/reuse-admission@1"
    assert GATE_ID.startswith("ipfs-accelerate.semantic-state.program-world-reuse")
    assert RECEIPT_CACHE_AUTHORITY in LOCATED_LANDED_AUTHORITIES
    assert TEST_PROOF_CACHE_AUTHORITY in LOCATED_LANDED_AUTHORITIES
    assert_no_duplicate_reuse_authority()


def test_adapters_still_do_not_define_the_reuse_gate() -> None:
    tree = ast.parse(ADAPTER_PATH.read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "ProgramWorldReuseGate" not in defined
    assert "evaluate_program_world_reuse" not in defined


def test_cold_import_has_no_io_thread_or_datasets_kit_side_effects() -> None:
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
proof_cache_loaded = any(
    name.endswith("verification.receipt_cache") or name.endswith("proof.test_proof_cache")
    for name in sys.modules
)
print(json.dumps({
    "effects": effects,
    "identity_loaded": identity_loaded,
    "kit_loaded": kit_loaded,
    "proof_cache_loaded": proof_cache_loaded,
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
    assert payload["proof_cache_loaded"] is False
    assert payload["interface"] == "ProgramWorldReuseGate@1"
    assert payload["evaluate"] == "evaluate_program_world_reuse"
    assert payload["explain"] == "explain_program_world_reuse"


def test_import_does_not_start_threads() -> None:
    before = {thread.name for thread in threading.enumerate()}
    import ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_reuse as reuse

    after = {thread.name for thread in threading.enumerate()}
    assert after == before
    assert reuse.PROGRAM_WORLD_REUSE_GATE_INTERFACE == "ProgramWorldReuseGate@1"


# ---------------------------------------------------------------------------
# Exact hit and binding-mismatch matrix
# ---------------------------------------------------------------------------


def test_exact_hit_is_proposal_until_supervisor_admission() -> None:
    evaluation = evaluate_program_world_reuse(_request())
    assert isinstance(evaluation, ProgramWorldReuseEvaluation)
    assert evaluation.granted is True
    assert evaluation.decision.verdict == "reuse"
    assert evaluation.decision.exact_match is True
    assert evaluation.decision.proposal_only is True
    assert evaluation.decision.admitted is False
    assert evaluation.decision.ann_authoritative is False
    assert evaluation.decision.reason_code == ReuseReasonCode.EXACT_IDENTITY_MATCH.value
    assert evaluation.decision.operational_acceptance_authorities == (
        OPERATIONAL_ACCEPTANCE_AUTHORITIES
    )
    assert "mandatory_validation_required" in evaluation.decision.limitations
    assert "cache_hit_never_suppresses_current_admission" in evaluation.decision.limitations
    assert evaluation.rejection is None
    round_trip = ProgramWorldReuseKey.from_dict(evaluation.key.to_dict())
    assert round_trip.key_cid == evaluation.key.key_cid
    decision_round_trip = ProgramWorldReuseDecision.from_dict(evaluation.decision.to_dict())
    assert decision_round_trip.decision_cid == evaluation.decision.decision_cid


def test_exact_hit_can_be_admitted_only_by_supervisor_authorities() -> None:
    evaluation = evaluate_program_world_reuse(
        _request(
            admission_authority=VALIDATION_AUTHORITY,
            admission_evidence_cid=_cid("validation-admission"),
        )
    )
    assert evaluation.decision.admitted is True
    assert evaluation.decision.proposal_only is False
    assert evaluation.decision.admission_authority == VALIDATION_AUTHORITY
    assert MERGE_AUTHORITY in evaluation.decision.operational_acceptance_authorities
    assert EVENT_AUTHORITY in evaluation.decision.operational_acceptance_authorities


def test_gate_cannot_admit_its_own_reuse_decision() -> None:
    with pytest.raises(ProgramWorldAdmissionError, match="cannot admit"):
        _request(admission_authority=GATE_ID, admission_evidence_cid=_cid("self"))
    with pytest.raises(ProgramWorldAdmissionError, match="cannot admit"):
        evaluate_program_world_reuse(
            _request(
                admission_authority="ProgramWorldReuseGate",
                admission_evidence_cid=_cid("self"),
            )
        )
    for owned in GATE_OWNED_AUTHORITIES:
        with pytest.raises(ProgramWorldAdmissionError):
            ProgramWorldReuseRequest(
                key=_key(),
                candidate_key=_key(),
                admission_authority=owned,
                admission_evidence_cid=_cid("self"),
            )


@pytest.mark.parametrize("field", EXACT_BINDING_FIELDS)
def test_single_binding_mismatch_refuses_reuse(field: str) -> None:
    key = _key()
    if field in {"obligation_cids", "selection_cids", "validation_dependency_cids"}:
        replacement = (_cid(f"other-{field}"),)
    elif field == "procedure_revision_cid":
        replacement = _cid("procedure-rev-other")
    else:
        replacement = _cid(f"other-{field}")
    candidate = key.replace(**{field: replacement})
    evaluation = evaluate_program_world_reuse(_request(key=key, candidate_key=candidate))
    assert evaluation.granted is False
    assert evaluation.decision.verdict == "reject"
    assert evaluation.decision.reason_code == ReuseReasonCode.BINDING_MISMATCH.value
    assert evaluation.decision.exact_match is False
    assert field in evaluation.rejection.mismatched_bindings
    assert field in evaluation.explanation.mismatched_bindings
    assert field in evaluation.explanation.why


def test_key_cid_changes_when_any_binding_changes() -> None:
    base = _key()
    seen = {base.key_cid}
    for field in EXACT_BINDING_FIELDS:
        if field in {"obligation_cids", "selection_cids", "validation_dependency_cids"}:
            other = base.replace(**{field: (_cid(f"alt-{field}"),)})
        elif field == "procedure_revision_cid":
            other = base.replace(procedure_revision_cid=_cid("alt-procedure"))
        else:
            other = base.replace(**{field: _cid(f"alt-{field}")})
        assert other.key_cid not in seen
        seen.add(other.key_cid)


# ---------------------------------------------------------------------------
# Stale / unsafe / similarity / unavailable / raw-source / proof-cache
# ---------------------------------------------------------------------------


def test_stale_freshness_revokes_even_an_exact_hit() -> None:
    evaluation = evaluate_program_world_reuse(_request(freshness="stale"))
    assert evaluation.decision.verdict == "reject"
    assert evaluation.decision.reason_code == ReuseReasonCode.STALE_EVIDENCE.value
    assert "stale_authoritative_reuse_forbidden" in evaluation.decision.limitations
    assert "freshness" in evaluation.explanation.why


def test_unknown_freshness_fails_closed() -> None:
    evaluation = evaluate_program_world_reuse(_request(freshness="unknown"))
    assert evaluation.decision.verdict == "reject"
    assert evaluation.decision.reason_code == ReuseReasonCode.STALE_EVIDENCE.value


def test_stale_world_root_revokes_reuse() -> None:
    evaluation = evaluate_program_world_reuse(
        _request(
            current_world_root_cid=_cid("root-current"),
            candidate_world_root_cid=_cid("root-historical"),
        )
    )
    assert evaluation.decision.reason_code == ReuseReasonCode.STALE_EVIDENCE.value
    assert evaluation.granted is False


def test_unsafe_abstraction_fails_closed_and_offers_raw_source_fallback() -> None:
    evaluation = evaluate_program_world_reuse(
        _request(
            abstraction_profile_cid=_cid("profile-interval"),
            candidate_abstraction_profile_cid=_cid("profile-interval"),
            abstraction_sound=False,
            unproved_omission=True,
            raw_source_available=True,
            source_cid=_cid("raw-source"),
        )
    )
    assert evaluation.decision.verdict == "reject"
    assert evaluation.decision.reason_code == ReuseReasonCode.UNSAFE_ABSTRACTION.value
    assert evaluation.decision.fallback is True
    assert evaluation.rejection.raw_source_fallback is True
    assert evaluation.explanation.raw_source_fallback is True
    assert "unsafe_abstraction_reuse_forbidden" in evaluation.decision.limitations


def test_unavailable_abstraction_dimensions_fail_closed() -> None:
    evaluation = evaluate_program_world_reuse(
        _request(
            abstraction_profile_cid=_cid("profile"),
            candidate_abstraction_profile_cid=_cid("profile"),
            abstraction_sound=True,
            abstraction_soundness_authority=DATASETS_ABSTRACTION_AUTHORITY,
            unavailable_dimensions=("native_heap",),
            raw_source_available=True,
            source_cid=_cid("raw-source"),
        )
    )
    assert evaluation.decision.reason_code == ReuseReasonCode.UNSAFE_ABSTRACTION.value
    assert "unknown_relevance_widens" in evaluation.decision.limitations


def test_sound_abstraction_certified_by_datasets_can_reuse_exactly() -> None:
    evaluation = evaluate_program_world_reuse(
        _request(
            abstraction_profile_cid=_cid("profile-interval"),
            candidate_abstraction_profile_cid=_cid("profile-interval"),
            abstraction_sound=True,
            abstraction_soundness_authority=DATASETS_ABSTRACTION_AUTHORITY,
        )
    )
    assert evaluation.granted is True
    assert evaluation.decision.exact_match is True


def test_gate_cannot_certify_abstraction_soundness() -> None:
    with pytest.raises(ProgramWorldAdmissionError, match="abstraction"):
        _request(
            abstraction_profile_cid=_cid("profile"),
            abstraction_sound=True,
            abstraction_soundness_authority=GATE_ID,
        )


def test_similarity_only_is_context_only_and_cannot_grant_reuse() -> None:
    key = _key()
    evaluation = evaluate_program_world_reuse(
        _request(
            key=key,
            candidate_key=key.replace(state_cid=_cid("nearby-state")),
            evidence_kind=ReuseEvidenceKind.SIMILARITY,
            ann_candidates=({"score": 0.99, "nearest": True},),
            similarity_score=0.99,
        )
    )
    assert evaluation.decision.verdict == "reject"
    assert evaluation.decision.reason_code == ReuseReasonCode.SIMILARITY_IS_NOT_REUSE.value
    assert evaluation.decision.exact_match is False
    assert evaluation.decision.admitted is False
    assert evaluation.explanation.context_only is True
    assert "similarity_is_context_only" in evaluation.decision.limitations


def test_similarity_present_does_not_block_current_exact_hit() -> None:
    evaluation = evaluate_program_world_reuse(
        _request(
            ann_candidates=({"score": 0.42},),
            similarity_score=0.42,
        )
    )
    assert evaluation.granted is True
    assert evaluation.decision.exact_match is True
    assert evaluation.explanation.context_only is True
    assert "similarity_is_context_only" in evaluation.decision.limitations
    assert evaluation.decision.ann_authoritative is False


def test_typed_unavailable_surfaces_fail_closed_without_simulation() -> None:
    evaluation = evaluate_program_world_reuse(
        _request(
            unavailable_surfaces=("proof_cache",),
            evidence_kind=ReuseEvidenceKind.UNAVAILABLE,
            raw_source_available=True,
        )
    )
    assert evaluation.decision.verdict == "unavailable"
    assert evaluation.decision.fallback is True
    assert evaluation.decision.reason_code == ReuseReasonCode.TYPED_UNAVAILABLE.value
    assert "proof_cache_unavailable" in evaluation.decision.limitations
    assert evaluation.granted is False


@pytest.mark.parametrize(
    "surface,limitation",
    [
        ("datasets_identity", "datasets_identity_surface_unavailable"),
        ("kit_verified_store", "kit_verified_store_unavailable"),
        ("procedure_authority", "procedure_authority_unavailable"),
    ],
)
def test_each_typed_unavailable_surface_is_named(surface: str, limitation: str) -> None:
    evaluation = evaluate_program_world_reuse(
        _request(unavailable_surfaces=(surface,), evidence_kind="unavailable")
    )
    assert evaluation.decision.verdict == "unavailable"
    assert limitation in evaluation.decision.limitations


def test_raw_source_fallback_is_recorded_when_bindings_are_incomplete() -> None:
    evaluation = evaluate_program_world_reuse(
        _request(
            complete=False,
            raw_source_available=True,
            source_cid=_cid("raw-source"),
            evidence_kind=ReuseEvidenceKind.RAW_SOURCE,
        )
    )
    assert evaluation.decision.verdict == "reject"
    assert evaluation.decision.reason_code == ReuseReasonCode.INCOMPLETE_BINDING.value
    assert evaluation.rejection.raw_source_fallback is True
    assert evaluation.explanation.raw_source_fallback is True


def test_proof_cache_hit_without_fresh_independent_admission_cannot_close_a_mismatch() -> None:
    key = _key()
    candidate = key.replace(state_cid=_cid("proved-equivalent-state"))
    evaluation = evaluate_program_world_reuse(
        _request(
            key=key,
            candidate_key=candidate,
            evidence_kind=ReuseEvidenceKind.PROOF_BACKED,
            independently_proof_backed=True,
            proof_cache_hit=True,
            proof_cache_fresh=False,
            proof_admitted=False,
            relation_kind="equality",
            relation_status="proved",
            relation_claim_cid=_cid("relation"),
            relation_scope_cid=_cid("scope"),
            relation_subject_cids=(key.state_cid, candidate.state_cid),
        )
    )
    assert evaluation.granted is False
    assert (
        evaluation.decision.reason_code
        == ReuseReasonCode.PROOF_CACHE_REQUIRES_FRESH_ADMISSION.value
    )
    assert "cache_hit_never_suppresses_current_admission" in evaluation.decision.limitations


def test_independently_proof_backed_scoped_equality_can_grant_reuse() -> None:
    evaluation = evaluate_program_world_reuse(_proof_backed_request())
    assert evaluation.granted is True
    assert evaluation.decision.exact_match is True
    assert evaluation.explanation.exact_match is False
    assert evaluation.explanation.independently_proof_backed is True
    assert (
        evaluation.decision.reason_code
        == ReuseReasonCode.INDEPENDENTLY_PROOF_BACKED.value
    )
    assert evaluation.decision.proposal_only is True
    assert "relation_reuse_scoped_to_assumptions" in evaluation.decision.limitations


def test_proof_backed_reuse_still_requires_supervisor_admission() -> None:
    evaluation = evaluate_program_world_reuse(
        _proof_backed_request(
            admission_authority=VALIDATION_AUTHORITY,
            admission_evidence_cid=_cid("supervisor-admission"),
        )
    )
    assert evaluation.decision.admitted is True
    assert evaluation.decision.proposal_only is False


def test_proof_backed_reuse_cannot_escape_relation_scope() -> None:
    key = _key()
    candidate = key.replace(state_cid=_cid("outside-scope-state"))
    evaluation = evaluate_program_world_reuse(
        _proof_backed_request(
            key=key,
            candidate_key=candidate,
            relation_subject_cids=(key.state_cid, _cid("some-other-state")),
        )
    )
    assert evaluation.granted is False
    assert evaluation.decision.reason_code == ReuseReasonCode.RELATION_SCOPE_EXCEEDED.value


def test_proof_backed_reuse_cannot_substitute_environment_or_policy() -> None:
    key = _key()
    candidate = key.replace(environment_cid=_cid("env-other"))
    evaluation = evaluate_program_world_reuse(
        _proof_backed_request(key=key, candidate_key=candidate)
    )
    assert evaluation.decision.reason_code == ReuseReasonCode.BINDING_MISMATCH.value
    assert "environment_cid" in evaluation.rejection.mismatched_bindings
    assert "environment_cid" not in PROOF_BACKED_SUBSTITUTABLE_BINDINGS


def test_contradictory_relation_abstains() -> None:
    evaluation = evaluate_program_world_reuse(
        _proof_backed_request(relation_kind="contradiction", relation_status="proved")
    )
    assert evaluation.decision.verdict == "abstain"
    assert evaluation.decision.reason_code == ReuseReasonCode.CONTRADICTORY_PREMISES.value
    assert "no_ex_falso_reuse" in evaluation.decision.limitations


def test_environment_incompatible_flag_fails_closed() -> None:
    evaluation = evaluate_program_world_reuse(_request(environment_compatible=False))
    assert evaluation.decision.reason_code == ReuseReasonCode.ENVIRONMENT_INCOMPATIBLE.value
    assert evaluation.granted is False


def test_no_candidate_abstains_and_keeps_raw_source_available() -> None:
    evaluation = evaluate_program_world_reuse(
        _request(candidate_key=None, raw_source_available=True, source_cid=_cid("raw"))
    )
    assert evaluation.decision.verdict == "abstain"
    assert evaluation.decision.reason_code == ReuseReasonCode.NO_EXACT_IDENTITY_MATCH.value
    assert evaluation.decision.fallback is True


def test_gate_cannot_be_its_own_proof_authority() -> None:
    with pytest.raises(ProgramWorldAdmissionError, match="cannot validate"):
        _proof_backed_request(proof_admission_authority=GATE_ID)


def test_unknown_proof_authority_is_rejected() -> None:
    with pytest.raises(ProgramWorldReuseError, match="landed"):
        _proof_backed_request(proof_admission_authority="ann.similarity.oracle")


# ---------------------------------------------------------------------------
# Negative memory
# ---------------------------------------------------------------------------


def test_negative_memory_records_scoped_rejection_and_replays() -> None:
    gate = ProgramWorldReuseGate()
    key = _key()
    candidate = key.replace(state_cid=_cid("mismatch-state"))
    first = evaluate_program_world_reuse(
        _request(key=key, candidate_key=candidate), gate=gate
    )
    assert first.decision.reason_code == ReuseReasonCode.BINDING_MISMATCH.value
    assert first.negative_memory is not None
    assert first.negative_memory.active is True
    replay = evaluate_program_world_reuse(
        _request(key=key, candidate_key=candidate), gate=gate
    )
    assert replay.decision.reason_code == ReuseReasonCode.NEGATIVE_MEMORY_INVALIDATION.value
    assert replay.negative_memory is not None
    assert replay.explanation.negative_memory_cid == replay.negative_memory.record_cid
    assert NegativeMemoryRecord.from_dict(replay.negative_memory.to_dict()).record_cid == (
        replay.negative_memory.record_cid
    )


def test_negative_memory_does_not_block_a_different_candidate() -> None:
    gate = ProgramWorldReuseGate()
    key = _key()
    evaluate_program_world_reuse(
        _request(key=key, candidate_key=key.replace(state_cid=_cid("bad"))),
        gate=gate,
    )
    hit = evaluate_program_world_reuse(_request(key=key, candidate_key=key), gate=gate)
    assert hit.granted is True


def test_negative_memory_invalidation_allows_a_now_fresh_exact_hit() -> None:
    gate = ProgramWorldReuseGate()
    key = _key()
    stale = evaluate_program_world_reuse(
        _request(
            key=key,
            candidate_key=key,
            freshness="stale",
            current_world_root_cid=_cid("root-old"),
            candidate_world_root_cid=_cid("root-old"),
        ),
        gate=gate,
    )
    assert stale.decision.reason_code == ReuseReasonCode.STALE_EVIDENCE.value
    assert stale.negative_memory is not None
    invalidated = gate.invalidate_negative_memory(key_cid=key.key_cid)
    assert invalidated
    assert all(not record.active for record in invalidated)
    fresh = evaluate_program_world_reuse(
        _request(
            key=key,
            candidate_key=key,
            freshness="fresh",
            current_world_root_cid=_cid("root-current"),
            candidate_world_root_cid=_cid("root-current"),
        ),
        gate=gate,
    )
    assert fresh.granted is True


def test_stale_negative_memory_lifts_when_freshness_and_root_are_current() -> None:
    gate = ProgramWorldReuseGate()
    key = _key()
    evaluate_program_world_reuse(
        _request(
            key=key,
            candidate_key=key,
            freshness="stale",
            current_world_root_cid=_cid("root-now"),
            candidate_world_root_cid=_cid("root-then"),
        ),
        gate=gate,
    )
    fresh = evaluate_program_world_reuse(
        _request(
            key=key,
            candidate_key=key,
            freshness="fresh",
            current_world_root_cid=_cid("root-now"),
            candidate_world_root_cid=_cid("root-now"),
        ),
        gate=gate,
    )
    assert fresh.granted is True


def test_new_independent_proof_evidence_lifts_prior_proof_rejection() -> None:
    gate = ProgramWorldReuseGate()
    key = _key()
    candidate = key.replace(state_cid=_cid("state-equiv"))
    first = evaluate_program_world_reuse(
        _proof_backed_request(
            key=key,
            candidate_key=candidate,
            proof_cache_fresh=False,
            proof_admitted=False,
            independently_proof_backed=True,
            proof_admission_authority=RECEIPT_CACHE_AUTHORITY,
            proof_admission_evidence_cid=None,
        ),
        gate=gate,
    )
    assert first.granted is False
    second = evaluate_program_world_reuse(
        _proof_backed_request(
            key=key,
            candidate_key=candidate,
            proof_admission_authority=TEST_PROOF_CACHE_AUTHORITY,
            proof_admission_evidence_cid=_cid("fresh-proof-v2"),
        ),
        gate=gate,
    )
    assert second.granted is True


# ---------------------------------------------------------------------------
# Explanation and factories
# ---------------------------------------------------------------------------


def test_explain_program_world_reuse_round_trips_evaluation_and_decision() -> None:
    evaluation = evaluate_program_world_reuse(_request())
    from_eval = explain_program_world_reuse(evaluation)
    assert from_eval.explanation_cid == evaluation.explanation.explanation_cid
    from_decision = explain_program_world_reuse(evaluation.decision)
    assert from_decision.verdict == "reuse"
    assert from_decision.exact_match is True
    from_request = explain_program_world_reuse(_request())
    assert from_request.reason_code == ReuseReasonCode.EXACT_IDENTITY_MATCH.value
    rebuilt = ProgramWorldReuseExplanation.from_dict(from_eval.to_dict())
    assert rebuilt.explanation_cid == from_eval.explanation_cid


def test_explain_rejection_names_the_fail_closed_reason() -> None:
    evaluation = evaluate_program_world_reuse(_request(freshness="stale"))
    explanation = explain_program_world_reuse(evaluation)
    assert explanation.verdict == "reject"
    assert "stale" in explanation.why.lower()
    assert explanation.required_admission is True


def test_load_program_world_reuse_gate_returns_isolated_memory() -> None:
    first = load_program_world_reuse_gate()
    second = load_program_world_reuse_gate()
    key = _key()
    evaluate_program_world_reuse(
        _request(key=key, candidate_key=key.replace(goal_cid=_cid("other-goal"))),
        gate=first,
    )
    assert first.negative_memory_records()
    assert second.negative_memory_records() == ()


def test_keyword_evaluate_path_matches_request_object() -> None:
    key = _key()
    via_kwargs = evaluate_program_world_reuse(
        state_cid=key.state_cid,
        goal_cid=key.goal_cid,
        policy_cid=key.policy_cid,
        environment_cid=key.environment_cid,
        toolchain_cid=key.toolchain_cid,
        obligation_cids=key.obligation_cids,
        selection_cids=key.selection_cids,
        procedure_revision_cid=key.procedure_revision_cid,
        validation_dependency_cids=key.validation_dependency_cids,
        candidate_key=key,
    )
    via_request = evaluate_program_world_reuse(_request(key=key, candidate_key=key))
    assert via_kwargs.decision.reason_code == via_request.decision.reason_code
    assert via_kwargs.key.key_cid == via_request.key.key_cid


def test_rejection_record_round_trip() -> None:
    evaluation = evaluate_program_world_reuse(_request(freshness="stale"))
    assert evaluation.rejection is not None
    cloned = ProgramWorldReuseRejection.from_dict(evaluation.rejection.to_dict())
    assert cloned.rejection_cid == evaluation.rejection.rejection_cid


def test_ann_identity_fields_cannot_enter_reuse_key_payload() -> None:
    key = _key()
    payload = key.identity_payload()
    for forbidden in ("score", "similarity", "embedding", "ann_score", "nearest"):
        assert forbidden not in payload


def test_freshness_unavailable_is_typed_unavailable() -> None:
    evaluation = evaluate_program_world_reuse(_request(freshness="unavailable"))
    assert evaluation.decision.verdict == "unavailable"
    assert evaluation.decision.reason_code == ReuseReasonCode.TYPED_UNAVAILABLE.value
