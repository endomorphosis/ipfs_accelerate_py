"""Integration evidence for GoalDirectedProofTactician@1 (FVT-G036 / FVT-027).

Proves the acceptance subset:

* exact cache keys include tree / target / assumptions / provider / version /
  policy / bounds;
* model drafts and cache hits cannot bypass independent validation;
* proof-carrying execution is resumable from a durable checkpoint;
* ZKP binds an existing trusted receipt without increasing its assurance; and
* legal compatibility remains intact when legal evidence is in scope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.proof.goal_directed_tactician import (
    GOAL_DIRECTED_PROOF_TACTICIAN_INTERFACE,
    GOAL_DIRECTED_PROOF_TACTICIAN_SCHEMA,
    TACTICIAN_CACHE_KEY_SCHEMA,
    TACTICIAN_CHECKPOINT_SCHEMA,
    TACTICIAN_RESULT_SCHEMA,
    TACTICIAN_ZKP_BINDING_SCHEMA,
    AdmissionDecision,
    ExactTacticianCacheKey,
    GoalDirectedProofTactician,
    GoalDirectedTacticianError,
    GoalDirectedTacticianRequest,
    GoalDirectedTacticianResult,
    PhaseStatus,
    TacticianCheckpoint,
    TacticianPhase,
    TacticianStopReason,
    UtilityAuthority,
    UtilityRole,
    ZkpReceiptBinding,
    bind_zkp_to_trusted_receipt,
    build_exact_tactician_cache_key,
    claims_authority,
    create_goal_directed_proof_tactician,
    default_utility_bindings,
    reject_authority_bypass,
    run_goal_directed_tactician,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bounds(**overrides: Any) -> dict[str, Any]:
    payload = {
        "wall_time_ms": 30_000,
        "memory_bytes": 256 * 1024 * 1024,
        "max_steps": 64,
        "portfolio_width": 2,
    }
    payload.update(overrides)
    return payload


def _request(**overrides: Any) -> GoalDirectedTacticianRequest:
    payload: dict[str, Any] = {
        "tree_id": "tree:repo@abc123",
        "target_id": "goal:lease-safety",
        "assumption_ids": ("assumption:dep-ready", "assumption:bounds-ok"),
        "provider_id": "provider:leanstral",
        "provider_version": "1.2.3",
        "policy_id": "policy:fvt-tactician",
        "bounds": _bounds(),
        "formal_goal_id": "formal-goal:lease-safety",
        "obligation_id": "obl:lease-safety",
        "corpus_id": "corpus:proof-tactician",
        "corpus_version": "2026.07",
        "toolchain_id": "toolchain:locked@1",
        "required_assurance": AssuranceLevel.KERNEL_VERIFIED,
        "require_legal_compatibility": False,
        "enable_zkp": True,
    }
    payload.update(overrides)
    return GoalDirectedTacticianRequest(**payload)


def _kernel_ok(context: dict[str, Any]) -> dict[str, Any]:
    del context
    return {
        "status": "ok",
        "reason_code": "kernel_checked",
        "assurance": AssuranceLevel.KERNEL_VERIFIED.value,
        "receipt_id": "receipt:kernel:lease-safety",
        "independently_validated": True,
    }


def _prove_ok(context: dict[str, Any]) -> dict[str, Any]:
    del context
    return {
        "status": "ok",
        "reason_code": "independent_prove",
        "assurance": AssuranceLevel.KERNEL_VERIFIED.value,
        "receipt_id": "receipt:prove:lease-safety",
        "independently_validated": True,
    }


# ---------------------------------------------------------------------------
# Interface / composition
# ---------------------------------------------------------------------------


def test_interface_identity_and_default_utilities() -> None:
    assert GOAL_DIRECTED_PROOF_TACTICIAN_INTERFACE == "GoalDirectedProofTactician@1"
    assert GoalDirectedProofTactician.interface == (
        GOAL_DIRECTED_PROOF_TACTICIAN_INTERFACE
    )
    assert GoalDirectedProofTactician.schema == GOAL_DIRECTED_PROOF_TACTICIAN_SCHEMA
    roles = {item.role for item in default_utility_bindings()}
    assert UtilityRole.FORMALIZATION in roles
    assert UtilityRole.RETRIEVAL in roles
    assert UtilityRole.PROOF_SCHEDULER in roles
    assert UtilityRole.PROOF_CARRYING_PLANNER in roles
    assert UtilityRole.HAMMER in roles
    assert UtilityRole.KERNEL in roles
    assert UtilityRole.LEANSTRAL in roles
    assert UtilityRole.SYMAI in roles
    assert UtilityRole.AUTOENCODER in roles
    assert UtilityRole.LEGAL_ADAPTER in roles
    assert UtilityRole.CACHE in roles
    assert UtilityRole.CORPUS in roles
    assert UtilityRole.ZKP_BINDING in roles
    assert UtilityRole.SUPERVISOR_ADMISSION in roles
    # Guidance utilities must never be authority-bearing.
    for item in default_utility_bindings():
        if item.role in {
            UtilityRole.HAMMER,
            UtilityRole.LEANSTRAL,
            UtilityRole.SYMAI,
            UtilityRole.AUTOENCODER,
            UtilityRole.RETRIEVAL,
        }:
            assert item.authority is UtilityAuthority.GUIDANCE


def test_exact_cache_key_includes_tree_target_assumptions_provider_version_policy_bounds() -> None:
    key = build_exact_tactician_cache_key(
        tree_id="tree:repo@abc123",
        target_id="goal:lease-safety",
        assumption_ids=("assumption:dep-ready", "assumption:bounds-ok"),
        provider_id="provider:leanstral",
        provider_version="1.2.3",
        policy_id="policy:fvt-tactician",
        bounds=_bounds(),
        corpus_id="corpus:proof-tactician",
        corpus_version="2026.07",
    )
    payload = key.to_dict()
    assert payload["schema"] == TACTICIAN_CACHE_KEY_SCHEMA
    assert payload["tree_id"] == "tree:repo@abc123"
    assert payload["target_id"] == "goal:lease-safety"
    assert payload["assumption_ids"] == [
        "assumption:dep-ready",
        "assumption:bounds-ok",
    ]
    assert payload["provider_id"] == "provider:leanstral"
    assert payload["provider_version"] == "1.2.3"
    assert payload["policy_id"] == "policy:fvt-tactician"
    assert payload["bounds"]["wall_time_ms"] == 30_000
    assert payload["bound_digest"].startswith("sha256:")
    assert payload["assumptions_digest"].startswith("sha256:")
    assert payload["key_id"].startswith("tactician-cache-key:sha256:")

    # Changing any required component must invalidate the key.
    shifted = build_exact_tactician_cache_key(
        tree_id="tree:repo@abc123",
        target_id="goal:lease-safety",
        assumption_ids=("assumption:dep-ready", "assumption:bounds-ok"),
        provider_id="provider:leanstral",
        provider_version="1.2.4",  # version change
        policy_id="policy:fvt-tactician",
        bounds=_bounds(),
    )
    assert shifted.key_id != key.key_id

    tree_shift = build_exact_tactician_cache_key(
        tree_id="tree:repo@def456",
        target_id="goal:lease-safety",
        assumption_ids=("assumption:dep-ready", "assumption:bounds-ok"),
        provider_id="provider:leanstral",
        provider_version="1.2.3",
        policy_id="policy:fvt-tactician",
        bounds=_bounds(),
    )
    assert tree_shift.key_id != key.key_id

    assumption_shift = build_exact_tactician_cache_key(
        tree_id="tree:repo@abc123",
        target_id="goal:lease-safety",
        assumption_ids=("assumption:dep-ready",),
        provider_id="provider:leanstral",
        provider_version="1.2.3",
        policy_id="policy:fvt-tactician",
        bounds=_bounds(),
    )
    assert assumption_shift.key_id != key.key_id

    bounds_shift = build_exact_tactician_cache_key(
        tree_id="tree:repo@abc123",
        target_id="goal:lease-safety",
        assumption_ids=("assumption:dep-ready", "assumption:bounds-ok"),
        provider_id="provider:leanstral",
        provider_version="1.2.3",
        policy_id="policy:fvt-tactician",
        bounds=_bounds(wall_time_ms=10_000),
    )
    assert bounds_shift.key_id != key.key_id

    # Round-trip and projection into formal verification cache key surface.
    restored = ExactTacticianCacheKey.from_dict(payload)
    assert restored.matches(key)
    proof_key = key.to_proof_cache_key()
    assert proof_key.candidate_tree == key.tree_id
    assert proof_key.policy == key.policy_id


def test_empty_bounds_rejected() -> None:
    with pytest.raises(GoalDirectedTacticianError, match="bounds"):
        build_exact_tactician_cache_key(
            tree_id="tree:x",
            target_id="goal:y",
            provider_id="provider:z",
            provider_version="1",
            policy_id="policy:p",
            bounds={},
        )


# ---------------------------------------------------------------------------
# Model / cache cannot bypass validation
# ---------------------------------------------------------------------------


def test_model_draft_claiming_authority_cannot_bypass_validation() -> None:
    draft = {
        "lemma": "x >= 0",
        "verified": True,
        "assurance": "kernel_verified",
    }
    assert claims_authority(draft) is True
    with pytest.raises(GoalDirectedTacticianError, match="cannot bypass"):
        reject_authority_bypass(
            draft,
            source="model_draft",
            independently_validated=False,
        )
    # After independent validation the same payload may be retained for audit.
    reject_authority_bypass(
        draft,
        source="model_draft",
        independently_validated=True,
    )

    result = GoalDirectedProofTactician(kernel=_kernel_ok).run(
        _request(model_draft=draft)
    )
    # Model draft with authority claims is rejected before kernel path can admit.
    assert result.stop_reason is TacticianStopReason.MODEL_BYPASS_REJECTED
    assert not result.admitted
    assert not result.independently_validated
    assert any(
        phase.phase is TacticianPhase.VALIDATE
        and phase.status is PhaseStatus.REJECTED
        for phase in result.phases
    )


def test_cache_hit_claiming_authority_without_validation_is_rejected() -> None:
    store: dict[str, dict[str, Any]] = {}

    def lookup(key: ExactTacticianCacheKey) -> dict[str, Any] | None:
        return store.get(key.key_id)

    def put(key: ExactTacticianCacheKey, payload: dict[str, Any]) -> None:
        store[key.key_id] = dict(payload)

    # Poison the cache with an unvalidated authority claim.
    req = _request()
    key = req.cache_key()
    store[key.key_id] = {
        "receipt_id": "receipt:forged",
        "authoritative_assurance": AssuranceLevel.KERNEL_VERIFIED.value,
        "verified": True,
        "independently_validated": False,
    }

    result = GoalDirectedProofTactician(
        cache_lookup=lookup,
        cache_store=put,
        # No kernel / prove path that would independently validate.
    ).run(req)
    assert result.stop_reason is TacticianStopReason.CACHE_BYPASS_REJECTED
    assert not result.admitted
    assert not result.independently_validated


def test_validated_cache_hit_is_admissible() -> None:
    store: dict[str, dict[str, Any]] = {}

    def lookup(key: ExactTacticianCacheKey) -> dict[str, Any] | None:
        return store.get(key.key_id)

    def put(key: ExactTacticianCacheKey, payload: dict[str, Any]) -> None:
        store[key.key_id] = dict(payload)

    req = _request()
    key = req.cache_key()
    store[key.key_id] = {
        "receipt_id": "receipt:validated-cache",
        "authoritative_assurance": AssuranceLevel.KERNEL_VERIFIED.value,
        "independently_validated": True,
    }

    result = GoalDirectedProofTactician(
        cache_lookup=lookup,
        cache_store=put,
    ).run(req)
    assert result.stop_reason is TacticianStopReason.ADMITTED
    assert result.admitted
    assert result.independently_validated
    assert result.receipt_id == "receipt:validated-cache"
    assert result.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert result.cache_key.matches(key)


# ---------------------------------------------------------------------------
# Formalization gate / happy path / ZKP
# ---------------------------------------------------------------------------


def test_prose_cannot_bypass_formalization() -> None:
    result = GoalDirectedProofTactician(kernel=_kernel_ok).run(
        _request(formal_goal_id="")
    )
    assert result.stop_reason is TacticianStopReason.FORMALIZATION_REQUIRED
    assert not result.admitted
    assert any(
        phase.phase is TacticianPhase.FORMALIZE
        and phase.status is PhaseStatus.REJECTED
        for phase in result.phases
    )


def test_happy_path_admits_with_kernel_and_zkp_without_assurance_increase(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "tactician_checkpoint.json"
    result = GoalDirectedProofTactician(
        kernel=_kernel_ok,
        checkpoint_dir=tmp_path,
    ).run(_request(), checkpoint_path=checkpoint_path)

    assert result.stop_reason is TacticianStopReason.ADMITTED
    assert result.admitted
    assert result.independently_validated
    assert result.legal_compatible
    assert result.resumable
    assert result.receipt_id == "receipt:kernel:lease-safety"
    assert result.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert result.admission.decision is AdmissionDecision.ADMITTED
    assert result.zkp_binding is not None
    assert result.zkp_binding.receipt_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert result.zkp_binding.bound_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert result.zkp_binding.assurance_increased is False
    assert result.zkp_binding.to_dict()["schema"] == TACTICIAN_ZKP_BINDING_SCHEMA
    # Exact cache key present on the result with all required fields.
    key = result.cache_key.to_dict()
    for field_name in (
        "tree_id",
        "target_id",
        "assumption_ids",
        "provider_id",
        "provider_version",
        "policy_id",
        "bounds",
    ):
        assert field_name in key and key[field_name]
    # Composed utilities recorded.
    assert len(result.utilities) >= 10
    # Phases include the orchestration spine.
    phase_names = {phase.phase for phase in result.phases}
    assert TacticianPhase.BUILD_CACHE_KEY in phase_names
    assert TacticianPhase.FORMALIZE in phase_names
    assert TacticianPhase.VALIDATE in phase_names
    assert TacticianPhase.ZKP_BIND in phase_names
    assert TacticianPhase.ADMISSION in phase_names
    assert checkpoint_path.is_file()
    wire = result.to_dict()
    assert wire["schema"] == TACTICIAN_RESULT_SCHEMA
    assert wire["interface"] == GOAL_DIRECTED_PROOF_TACTICIAN_INTERFACE
    restored = GoalDirectedTacticianResult.from_dict(wire)
    assert restored.admitted
    assert restored.zkp_binding is not None
    assert restored.zkp_binding.assurance_increased is False


def test_zkp_binding_rejects_assurance_increase() -> None:
    with pytest.raises(GoalDirectedTacticianError, match="must not increase"):
        ZkpReceiptBinding(
            receipt_id="receipt:x",
            receipt_assurance=AssuranceLevel.KERNEL_VERIFIED,
            bound_assurance=AssuranceLevel.ATTESTED,  # illegal increase
            circuit_id="circuit:x",
            backend_id="backend:x",
            verification_key_id="vk:x",
        )


def test_zkp_requires_trusted_receipt() -> None:
    with pytest.raises(GoalDirectedTacticianError, match="trusted"):
        bind_zkp_to_trusted_receipt(
            receipt_id="receipt:untrusted",
            receipt_assurance=AssuranceLevel.CANDIDATE,
            circuit_id="circuit:x",
            backend_id="backend:x",
            verification_key_id="vk:x",
        )

    binding = bind_zkp_to_trusted_receipt(
        receipt_id="receipt:ok",
        receipt_assurance=AssuranceLevel.KERNEL_VERIFIED,
        circuit_id="circuit:x",
        backend_id="backend:x",
        verification_key_id="vk:x",
    )
    assert binding.assurance_increased is False
    assert binding.bound_assurance is AssuranceLevel.KERNEL_VERIFIED


# ---------------------------------------------------------------------------
# Resumable proof-carrying execution
# ---------------------------------------------------------------------------


def test_proof_carrying_execution_is_resumable(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "resume.json"
    first = GoalDirectedProofTactician(kernel=_kernel_ok).run(
        _request(),
        checkpoint_path=checkpoint_path,
    )
    assert first.admitted
    assert first.checkpoint is not None
    assert first.checkpoint.resumable
    assert first.checkpoint.to_dict()["schema"] == TACTICIAN_CHECKPOINT_SCHEMA
    assert checkpoint_path.is_file()

    # Resume from disk with the same exact cache key reuses checkpoint phases.
    second = GoalDirectedProofTactician(kernel=_kernel_ok).run(
        _request(),
        checkpoint_path=checkpoint_path,
    )
    assert second.admitted
    assert any(
        phase.phase is TacticianPhase.LOAD_CHECKPOINT
        and phase.status is PhaseStatus.RESUMED
        for phase in second.phases
    )

    # Mismatched key (different tree) must fail closed.
    mismatch = GoalDirectedProofTactician(kernel=_kernel_ok).run(
        _request(tree_id="tree:repo@other"),
        checkpoint=TacticianCheckpoint.load(checkpoint_path),
    )
    assert mismatch.stop_reason is TacticianStopReason.CHECKPOINT_MISMATCH
    assert not mismatch.admitted

    # Atomic write round-trip.
    loaded = TacticianCheckpoint.load(checkpoint_path)
    assert loaded.cache_key.matches(first.cache_key)
    assert loaded.independently_validated is True


def test_run_functional_entry_and_factory() -> None:
    tactician = create_goal_directed_proof_tactician(kernel=_kernel_ok)
    assert isinstance(tactician, GoalDirectedProofTactician)
    result = run_goal_directed_tactician(
        _request().to_dict(),
        kernel=_kernel_ok,
        prove=_prove_ok,
    )
    assert result.admitted
    assert result.independently_validated


# ---------------------------------------------------------------------------
# Legal compatibility
# ---------------------------------------------------------------------------


def test_legal_compatibility_remains_intact_when_required() -> None:
    def legal_ok(context: dict[str, Any]) -> dict[str, Any]:
        assert context.get("require_legal_compatibility") is True
        return {
            "status": "ok",
            "reason_code": "legal_constraints_compatible",
            "legal_compatible": True,
            "obligations": ["legal:privacy-retention"],
        }

    result = GoalDirectedProofTactician(
        kernel=_kernel_ok,
        legal=legal_ok,
    ).run(_request(require_legal_compatibility=True))
    assert result.admitted
    assert result.legal_compatible is True
    assert any(
        phase.phase is TacticianPhase.LEGAL and phase.status is PhaseStatus.OK
        for phase in result.phases
    )


def test_legal_incompatibility_blocks_admission() -> None:
    def legal_bad(context: dict[str, Any]) -> dict[str, Any]:
        del context
        return {
            "status": "ok",
            "reason_code": "conflicting_legal_constraint",
            "legal_compatible": False,
        }

    result = GoalDirectedProofTactician(
        kernel=_kernel_ok,
        legal=legal_bad,
    ).run(_request(require_legal_compatibility=True))
    assert result.stop_reason is TacticianStopReason.LEGAL_INCOMPATIBLE
    assert not result.admitted
    assert result.legal_compatible is False


def test_guidance_utilities_never_grant_admission_alone() -> None:
    def guidance_only(context: dict[str, Any]) -> dict[str, Any]:
        del context
        return {
            "status": "ok",
            "reason_code": "guidance_complete",
            "verified": True,  # must not matter
            "assurance": "kernel_verified",
        }

    result = GoalDirectedProofTactician(
        guidance=guidance_only,
        # no kernel / prove
    ).run(_request())
    assert not result.admitted
    assert result.stop_reason in {
        TacticianStopReason.VALIDATION_FAILED,
        TacticianStopReason.MODEL_BYPASS_REJECTED,
        TacticianStopReason.CACHE_BYPASS_REJECTED,
    }
    assert not result.independently_validated


def test_request_cache_key_round_trip() -> None:
    req = _request()
    wire = req.to_dict()
    restored = GoalDirectedTacticianRequest.from_dict(wire)
    assert restored.cache_key().matches(req.cache_key())
    assert restored.provider_version == "1.2.3"
    assert restored.assumption_ids == (
        "assumption:dep-ready",
        "assumption:bounds-ok",
    )


def test_cancellation_is_fail_closed() -> None:
    result = GoalDirectedProofTactician(kernel=_kernel_ok).run(
        _request(),
        cancelled=True,
    )
    assert result.stop_reason is TacticianStopReason.CANCELLED
    assert not result.admitted
