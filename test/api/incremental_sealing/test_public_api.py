"""IPS-043: public API freeze, simulated rejection, and cold-import hermeticity."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.proof import incremental_sealing as package
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing import (
    PUBLIC_API_NAMES,
    PUBLIC_API_SUBSET,
    compare_full_and_incremental,
    create_full_checkpoint,
    create_incremental_plan,
    execute_incremental_plan,
    explain_invalidation,
    explain_reuse,
    report_optional_capabilities,
    verify_seal,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.executor import (
    CachedCandidate,
    ExecutionReasonCode,
    FreshProof,
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
_PARENT = "sha256:" + ("99" * 32)
_VK = "vk/prod-1"
_POLICY = _DIGEST_D
_TRUSTED = (_VK, "n/a")


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
        "policy_cid": _POLICY,
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
        environment_cid=_DIGEST_C,
        policy_cid=_POLICY,
    )


def test_public_api_freeze_exports_all_seven_apis() -> None:
    assert PUBLIC_API_SUBSET == "ips/public-api@1"
    assert PUBLIC_API_NAMES == (
        "create_full_checkpoint",
        "create_incremental_plan",
        "execute_incremental_plan",
        "verify_seal",
        "explain_reuse",
        "explain_invalidation",
        "compare_full_and_incremental",
    )
    for name in PUBLIC_API_NAMES:
        assert hasattr(package, name)
        assert callable(getattr(package, name))
    assert package.create_full_checkpoint is create_full_checkpoint
    assert package.create_incremental_plan is create_incremental_plan
    assert package.execute_incremental_plan is execute_incremental_plan
    assert package.verify_seal is verify_seal
    assert package.explain_reuse is explain_reuse
    assert package.explain_invalidation is explain_invalidation
    assert package.compare_full_and_incremental is compare_full_and_incremental


def test_create_full_checkpoint_public_api() -> None:
    seal = create_full_checkpoint(
        _state(),
        _policy(),
        units=(
            _unit("unit/a"),
            _unit(
                "unit/b",
                category="static_analysis",
                proof_object_cid=_DIGEST_F,
            ),
        ),
        expected_unit_ids=("unit/a", "unit/b"),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
    )
    assert seal.sealed is True
    assert seal.seal_status is SealStatus.SEALED_FULL
    assert seal.required_unit_ids == ("unit/a", "unit/b")


def test_create_incremental_plan_public_api() -> None:
    plan = create_incremental_plan(
        _parent(),
        _DIGEST_B,
        _DIGEST_F,
        units=(
            _plan_unit("unit/reuse"),
            _plan_unit(
                "unit/reprove",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
        ),
    )
    assert plan.mode is PlanMode.INCREMENTAL
    assert "unit/reuse" in plan.reusable_unit_ids
    assert "unit/reprove" in plan.invalidated_unit_ids


def test_execute_incremental_plan_public_api_rejects_simulated() -> None:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.admission import (
        EvidenceCandidate,
    )
    from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
        IntegrityCommitment,
    )

    plan = create_incremental_plan(
        _parent(),
        _DIGEST_B,
        _DIGEST_F,
        units=(
            _plan_unit("unit/reuse"),
            _plan_unit(
                "unit/new",
                preserved=False,
                added=True,
                admitted=False,
                candidate_present=False,
            ),
        ),
    )
    digest = "sha256:" + ("11" * 32)
    cid = "sha256:" + ("22" * 32)

    def fetch(unit_id: str) -> CachedCandidate | None:
        if unit_id != "unit/reuse":
            return None
        return CachedCandidate(
            unit_id,
            digest,
            digest,
            cid,
            cid,
            simulated=True,
        )

    def prove(unit):  # noqa: ANN001
        return FreshProof(
            unit.unit_id,
            EvidenceCandidate(
                evidence=IntegrityCommitment(
                    digest=digest,
                    cid=cid,
                    merkle_inclusion="leaf:0",
                    byte_length=32,
                ),
                proof_system_id="integrity",
                public_input_cid=cid,
                proof_unit_id=unit.unit_id,
                expected_digest=digest,
                observed_digest=digest,
                observed_public_input_cid=cid,
                proof_mode=ProofMode.INTEGRITY_ONLY,
                terminal_status=ProofTerminalStatus.INTEGRITY_VERIFIED,
            ),
            digest,
            simulated=True,
            status="simulated",
        )

    result = execute_incremental_plan(plan, fetch=fetch, prove=prove)
    assert result.succeeded is False
    assert ExecutionReasonCode.SIMULATED_FORBIDDEN.value in result.reason_codes
    assert result.may_aggregate is False


def test_verify_seal_public_api() -> None:
    seal = create_full_checkpoint(
        _state(),
        _policy(),
        units=(_unit("unit/a"),),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
    )
    result = verify_seal(seal, _TRUSTED, _policy())
    assert result.accepted is True
    assert result.reason.value == "accepted"


def test_explain_reuse_and_invalidation_public_apis() -> None:
    seal = create_full_checkpoint(
        _state(),
        _policy(),
        units=(_unit("unit/a"),),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
    )
    reuse = explain_reuse(seal, "unit/a")
    assert reuse.unit_id == "unit/a"
    assert reuse.freshly_verified is True
    assert reuse.to_canonical()["substitutes_for_verification"] is False

    plan = create_incremental_plan(
        _parent(),
        _DIGEST_B,
        _DIGEST_F,
        units=(
            _plan_unit(
                "unit/changed",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
        ),
    )
    invalidation = explain_invalidation(plan, "unit/changed")
    assert invalidation.unit_id == "unit/changed"
    assert invalidation.invalidated is True
    assert invalidation.to_canonical()["substitutes_for_verification"] is False


def test_compare_full_and_incremental_public_api() -> None:
    comparison = compare_full_and_incremental(
        _state(repository_state_cid=_DIGEST_F),
        _parent(),
        _policy(),
        units=(
            _plan_unit("unit/reuse"),
            _plan_unit(
                "unit/reprove",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
        ),
        old_repository_state=_DIGEST_B,
        estimated=True,
    )
    assert comparison.full_required_units >= 1
    assert comparison.incremental_prove_units >= 1
    assert comparison.estimated is True
    assert comparison.cost_comparison is not None


def test_production_seals_reject_simulated_evidence() -> None:
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
    )
    assert seal.sealed is False
    assert seal.seal_status is SealStatus.SIMULATED_ONLY
    assert seal.reason is FullCheckpointReason.SIMULATED_REQUIRED_UNIT
    assert "unit/sim" in seal.rejected_unit_ids

    # Verification of a simulated_only seal must not accept it as production.
    verification = verify_seal(seal, _TRUSTED, _policy())
    assert verification.accepted is False


def test_missing_optional_capabilities_are_typed() -> None:
    report = report_optional_capabilities()
    assert report["evidence_subset"] == PUBLIC_API_SUBSET
    assert report["kit_store"]["status"] == "unavailable"
    assert report["kit_store"]["reason_code"] == "optional_kit_not_injected"
    assert report["datasets_semantic"]["status"] == "unavailable"
    backends = report["backends"]
    assert isinstance(backends, dict)
    assert "simulated" in backends
    assert backends["simulated"]["status"] == "simulated_only"
    assert backends["simulated"]["production_seal_allowed"] is False
    # ProveKit absence is typed, never an uncaught import failure.
    if "provekit" in backends:
        assert backends["provekit"]["status"] in {
            "unavailable",
            "available",
            "unknown",
        }


def test_cold_public_api_import_is_hermetic() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, os, socket, sys\n"
                "before_modules = set(sys.modules)\n"
                "opened = []\n"
                "real_open = open\n"
                "def guarded_open(*args, **kwargs):\n"
                "    path = str(args[0]) if args else ''\n"
                "    # Allow interpreter/bootstrap reads only under sys.prefix / this process.\n"
                "    opened.append(path)\n"
                "    return real_open(*args, **kwargs)\n"
                "connect_calls = []\n"
                "real_connect = socket.socket.connect\n"
                "def guarded_connect(self, address):\n"
                "    connect_calls.append(address)\n"
                "    raise AssertionError('network connect during cold import')\n"
                "socket.socket.connect = guarded_connect\n"
                "mod = importlib.import_module(\n"
                "    'ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing'\n"
                ")\n"
                "assert mod.PUBLIC_API_SUBSET == 'ips/public-api@1'\n"
                "assert mod.PUBLIC_API_NAMES[0] == 'create_full_checkpoint'\n"
                "# Cold package import must not load heavy sealing submodules.\n"
                "heavy = [\n"
                "    'ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint',\n"
                "    'ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.executor',\n"
                "    'ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.backends',\n"
                "    'ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.trust',\n"
                "    'ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.sealer',\n"
                "]\n"
                "loaded_heavy = [name for name in heavy if name in sys.modules]\n"
                "assert loaded_heavy == [], loaded_heavy\n"
                "assert connect_calls == []\n"
                "print('ok')\n"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "ok" in completed.stdout


def test_lazy_attribute_resolution_matches_submodule_exports() -> None:
    # Force lazy resolution of each public API on the live package module.
    for name in PUBLIC_API_NAMES:
        value = getattr(package, name)
        assert callable(value)
    assert "create_full_checkpoint" in package.__all__
    assert "compare_full_and_incremental" in package.__all__
    # Submodule identity for the facade functions.
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing import (
        full_checkpoint as full_mod,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing import (
        explanations as expl_mod,
    )

    assert package.create_full_checkpoint is full_mod.create_full_checkpoint
    assert package.explain_reuse is expl_mod.explain_reuse
    assert package.compare_full_and_incremental is expl_mod.compare_full_and_incremental


def test_unknown_public_attribute_fails_closed() -> None:
    with pytest.raises(AttributeError):
        getattr(package, "generate_proving_key")
    with pytest.raises(AttributeError):
        getattr(package, "download_circuit_setup")
