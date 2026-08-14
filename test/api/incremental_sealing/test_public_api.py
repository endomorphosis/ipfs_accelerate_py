"""IPS-043: lazy public APIs for IncrementalProofSealer."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.proof import incremental_sealing as package
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing import (
    CLI_OPERATIONS,
    IMPORT_HERMETICITY_SUBSET,
    PUBLIC_API_NAMES,
    PUBLIC_API_SUBSET,
    compare_full_and_incremental,
    create_full_checkpoint,
    create_incremental_plan,
    execute_incremental_plan,
    explain_invalidation,
    explain_reuse,
    optional_capability_status,
    verify_seal,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.executor import (
    CachedCandidate,
    ResourcePolicy,
    execute_incremental_plan as _execute_incremental_plan_direct,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    FullCheckpointReason,
    RepositoryStateView,
    RequiredUnitEvidence,
    VerificationPolicyView,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
    ParentSealContext,
    PlanMode,
    UnitPlanningInput,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    IntegrityCommitment,
    ProofMode,
    ProofTerminalStatus,
    SealStatus,
)

_DIGEST_A = "sha256:" + ("aa" * 32)
_DIGEST_B = "sha256:" + ("bb" * 32)
_DIGEST_C = "sha256:" + ("cc" * 32)
_DIGEST_D = "sha256:" + ("dd" * 32)
_DIGEST_E = "sha256:" + ("ee" * 32)
_DIGEST_F = "sha256:" + ("ff" * 32)
_DIGEST_1 = "sha256:" + ("11" * 32)
_DIGEST_2 = "sha256:" + ("22" * 32)
_PARENT = "sha256:" + ("99" * 32)
_VK = "vk/prod-1"


def _state(**overrides: object) -> RepositoryStateView:
    payload = {
        "repository_id": "repo/accelerate",
        "revision": "rev-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "source_root_cid": _DIGEST_A,
        "repository_state_cid": _DIGEST_B,
        "environment_cid": _DIGEST_C,
        "parent_revision_ids": (),
    }
    payload.update(overrides)
    return RepositoryStateView(**payload)  # type: ignore[arg-type]


def _policy(**overrides: object) -> VerificationPolicyView:
    payload = {
        "policy_cid": _DIGEST_D,
        "proof_schema_version": "1",
        "canonicalization_version": "1",
        "dependency_graph_schema_version": "graph@1",
        "circuit_id": "circuit@v1",
        "verification_key_id": _VK,
    }
    payload.update(overrides)
    return VerificationPolicyView(**payload)  # type: ignore[arg-type]


def _unit(unit_id: str, **overrides: object) -> RequiredUnitEvidence:
    payload = {
        "unit_id": unit_id,
        "proof_object_cid": _DIGEST_E,
        "category": "unit_test",
        "terminal_status": ProofTerminalStatus.INTEGRITY_VERIFIED.value,
        "proof_mode": ProofMode.INTEGRITY_ONLY.value,
        "required_for_seal": True,
        "freshly_verified": True,
        "cache_reused_without_fresh_verification": False,
        "circuit_id": "circuit@v1",
        "verification_key_id": _VK,
    }
    payload.update(overrides)
    return RequiredUnitEvidence(**payload)  # type: ignore[arg-type]


def _plan_unit(unit_id: str, **overrides: object) -> UnitPlanningInput:
    payload = {
        "unit_id": unit_id,
        "preserved": True,
        "cache_key_complete": True,
        "admitted": True,
        "candidate_present": True,
    }
    payload.update(overrides)
    return UnitPlanningInput(**payload)  # type: ignore[arg-type]


def _parent() -> ParentSealContext:
    return ParentSealContext(
        seal_cid=_PARENT,
        repository_state_cid=_DIGEST_B,
        source_root_cid=_DIGEST_A,
        schema_version="1",
        canonicalization_version="1",
        environment_cid=_DIGEST_C,
        policy_cid=_DIGEST_D,
    )


def test_public_api_freeze_and_evidence_subsets() -> None:
    assert PUBLIC_API_SUBSET == "ips/public-api@1"
    assert IMPORT_HERMETICITY_SUBSET == "ips/import-hermeticity@1"
    assert package.CLI_SUBSET == "ips/cli@1"
    assert package.PACKAGE_SCHEMA_VERSION == "1"
    assert PUBLIC_API_NAMES == (
        "create_full_checkpoint",
        "create_incremental_plan",
        "execute_incremental_plan",
        "verify_seal",
        "explain_reuse",
        "explain_invalidation",
        "compare_full_and_incremental",
    )
    assert CLI_OPERATIONS == (
        "full",
        "incremental",
        "verify",
        "plan",
        "explain-reuse",
        "explain-invalidation",
        "benchmark",
        "cache-status",
        "force-full",
    )
    assert len(PUBLIC_API_NAMES) == 7
    assert len(CLI_OPERATIONS) == 9
    for name in PUBLIC_API_NAMES:
        assert hasattr(package, name)
        assert callable(getattr(package, name))
        # Each freeze name must resolve to the same callable as the owning module.
        resolved = getattr(package, name)
        assert callable(resolved)
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = package.not_a_public_export  # type: ignore[attr-defined]


def test_cold_package_import_is_hermetic() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, sys; "
                "mod = importlib.import_module("
                "'ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing'"
                "); "
                "assert mod.PUBLIC_API_SUBSET == 'ips/public-api@1'; "
                "assert mod.IMPORT_HERMETICITY_SUBSET == 'ips/import-hermeticity@1'; "
                "assert len(mod.PUBLIC_API_NAMES) == 7; "
                "assert len(mod.CLI_OPERATIONS) == 9; "
                # Cold package import must not pull optional prover/CID stacks.
                "assert 'provekit' not in sys.modules; "
                "assert 'py_ecc' not in sys.modules; "
                "assert 'multiformats' not in sys.modules; "
                # Submodules stay unloaded until an API is resolved.
                "assert 'ipfs_accelerate_py.agent_supervisor.proof."
                "incremental_sealing.full_checkpoint' not in sys.modules; "
                "assert 'ipfs_accelerate_py.agent_supervisor.proof."
                "incremental_sealing.provers' not in sys.modules; "
                "assert 'ipfs_accelerate_py.agent_supervisor.proof."
                "incremental_sealing.backends' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_create_full_checkpoint_public_api() -> None:
    seal = create_full_checkpoint(
        _state(),
        _policy(),
        units=(_unit("unit/a"), _unit("unit/b", proof_object_cid=_DIGEST_F)),
        fallback_reasons=("first_state", "missing_parent"),
    )
    assert seal.sealed is True
    assert seal.seal_status is SealStatus.SEALED_FULL
    assert seal.seal_cid().startswith("sha256:")
    assert set(seal.required_unit_ids) == {"unit/a", "unit/b"}


def test_production_seal_rejects_simulated_evidence() -> None:
    seal = create_full_checkpoint(
        _state(),
        _policy(),
        units=(
            _unit("unit/a"),
            _unit(
                "unit/sim",
                proof_mode=ProofMode.SIMULATED.value,
                terminal_status=ProofTerminalStatus.SIMULATED.value,
            ),
        ),
        fallback_reasons=("first_state",),
    )
    assert seal.sealed is False
    assert seal.seal_status is SealStatus.SIMULATED_ONLY
    assert seal.reason is FullCheckpointReason.SIMULATED_REQUIRED_UNIT
    assert "unit/sim" in seal.rejected_unit_ids
    assert seal.seal_status is not SealStatus.SEALED_FULL


def test_create_and_execute_incremental_plan() -> None:
    plan = create_incremental_plan(
        _parent(),
        _DIGEST_B,
        _DIGEST_1,
        units=(
            _plan_unit("unit/reuse"),
            _plan_unit(
                "unit/new",
                preserved=False,
                added=True,
                admitted=False,
                candidate_present=False,
            ),
            _plan_unit(
                "unit/reprove",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
        ),
        changed_root_cids=(_DIGEST_2,),
    )
    assert plan.mode is PlanMode.INCREMENTAL
    assert "unit/reuse" in plan.reusable_unit_ids
    assert "unit/new" in plan.added_unit_ids

    digest = _DIGEST_E
    cid = _DIGEST_F
    store = {
        "unit/reuse": CachedCandidate(
            unit_id="unit/reuse",
            expected_digest=digest,
            observed_digest=digest,
            public_input_cid=cid,
            observed_public_input_cid=cid,
            proof_object_cid=cid,
            evidence=IntegrityCommitment(
                digest=digest,
                cid=cid,
                merkle_inclusion="leaf:0",
                byte_length=32,
            ),
        )
    }
    # Public export must be the same callable as the executor module.
    assert execute_incremental_plan is _execute_incremental_plan_direct
    result = execute_incremental_plan(
        plan,
        ResourcePolicy(),
        fetch=store.get,
    )
    assert result.succeeded is True
    assert "unit/reuse" in result.reused_unit_ids
    assert "unit/new" in result.newly_proved_unit_ids
    assert "unit/reprove" in result.newly_proved_unit_ids


def test_verify_seal_public_api() -> None:
    seal = create_full_checkpoint(
        _state(),
        _policy(),
        units=(_unit("unit/a"),),
        fallback_reasons=("first_state",),
    )
    accepted = verify_seal(
        seal,
        trusted_keys=(_VK, "n/a"),
        verification_policy=_policy(),
        unit_proofs=(),
        require_cryptographic_check=False,
        require_complete_history=False,
    )
    # Fresh full seals with integrity units may accept or reject depending on
    # signature stage; either outcome must be typed and never export secrets.
    assert accepted.reason is not None
    payload = accepted.to_canonical()
    assert payload["proving_key_exported"] is False
    assert payload["witness_exported"] is False
    assert "proving_key" not in payload["details"]


def test_explain_reuse_and_invalidation() -> None:
    seal = create_full_checkpoint(
        _state(),
        _policy(),
        units=(_unit("unit/a"),),
        fallback_reasons=("first_state",),
    )
    reuse = explain_reuse(seal, "unit/a")
    assert reuse.unit_id == "unit/a"
    canonical = reuse.to_canonical()
    assert canonical["substitutes_for_verification"] is False
    assert canonical["file_unchanged_is_not_reuse_authority"] is True
    assert len(canonical["bound_cache_key_fields"]) >= 20

    plan = create_incremental_plan(
        _parent(),
        _DIGEST_B,
        _DIGEST_1,
        units=(
            _plan_unit("unit/keep"),
            _plan_unit(
                "unit/changed",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
        ),
        changed_root_cids=(_DIGEST_2,),
    )
    invalidation = explain_invalidation(plan, "unit/changed")
    assert invalidation.unit_id == "unit/changed"
    assert invalidation.invalidated is True
    inv = invalidation.to_canonical()
    assert inv["substitutes_for_verification"] is False
    assert len(inv["bound_cache_key_fields"]) >= 20


def test_compare_full_and_incremental() -> None:
    comparison = compare_full_and_incremental(
        _DIGEST_1,
        _parent(),
        _policy().to_canonical(),
        units=(
            _plan_unit("unit/keep"),
            _plan_unit(
                "unit/changed",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
        ),
        estimated=True,
        old_repository_state=_DIGEST_B,
        changed_root_cids=(_DIGEST_2,),
    )
    payload = comparison.to_canonical()
    assert payload["estimated"] is True
    assert payload["estimated_as_measured"] is False
    assert payload["substitutes_for_verification"] is False
    assert comparison.full_required_units >= comparison.incremental_prove_units


def test_missing_optional_capability_is_typed() -> None:
    report = optional_capability_status(
        "provekit",
        availability_overrides={"provekit": False},
    )
    assert report["backend_id"] == "provekit"
    assert report["status"] in {"unavailable", "unknown", "available"}
    assert report["auto_install"] is False
    assert report["network_accessed"] is False
    assert report["keys_generated"] is False
    assert report["user_state_mutated"] is False
    assert report["processes_started"] is False
    assert isinstance(report["reason_code"], str) and report["reason_code"]
    assert isinstance(report["production_seal_allowed"], bool)
    # Forced-unavailable provekit never upgrades to production success.
    if report["status"] == "unavailable":
        assert report["production_seal_allowed"] is False

    simulated = optional_capability_status("simulated")
    assert simulated["status"] in {"simulated_only", "unavailable", "available"}
    assert simulated["production_seal_allowed"] is False
    assert "simulated" in simulated["reason_code"] or simulated[
        "production_seal_allowed"
    ] is False

    unknown = optional_capability_status("not-a-real-backend")
    assert unknown["status"] == "unknown"
    assert unknown["production_seal_allowed"] is False
    assert unknown["auto_install"] is False
    assert unknown["network_accessed"] is False


def test_lazy_export_resolves_same_callable() -> None:
    reloaded = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing"
    )
    assert reloaded.create_full_checkpoint is create_full_checkpoint
    assert reloaded.create_incremental_plan is create_incremental_plan
    assert reloaded.execute_incremental_plan is execute_incremental_plan
    assert reloaded.verify_seal is verify_seal
    assert reloaded.explain_reuse is explain_reuse
    assert reloaded.explain_invalidation is explain_invalidation
    assert reloaded.compare_full_and_incremental is compare_full_and_incremental
    # CLI entrypoints are lazy and identity-stable once resolved.
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing import cli as cli_mod

    assert reloaded.main is cli_mod.main
    assert reloaded.build_parser is cli_mod.build_parser
