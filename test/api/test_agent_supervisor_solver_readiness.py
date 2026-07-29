"""SCA-621 / SCA-G180 solver readiness tests for SCAEV180PROOFREADY.

Proves exact, reproducible DCEC/Z3/TDFOL/CEC/Hammer availability, unsupported
status for missing backends, non-authoritative solver SAT until kernel
reconstruction, and readiness identities in trust-aware proof-cache keys.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.solver_readiness import (
    REQUIRED_SOLVER_FAMILIES,
    SCAEV180PROOFREADY,
    SCAEV180PROOFREADY_COVERAGE,
    SCAEV180PROOFREADY_EVIDENCE,
    SolverAuthority,
    SolverBackendFamily,
    SolverBackendReadiness,
    SolverReadinessProbe,
    SolverReadinessProbeConfig,
    SolverReadinessStatus,
    probe_solver_readiness,
    production_compose_supported_obligation,
    register_supported_backends,
    solver_cache_key_material,
)


class FakeDiscovery:
    def __init__(
        self,
        *,
        modules: set[str] | None = None,
        executables: dict[str, str] | None = None,
        versions: dict[str, str] | None = None,
    ) -> None:
        self.modules = modules or set()
        self.executables = executables or {}
        self.versions = versions or {"ipfs_datasets_py": "9.9.9-test"}

    def find_spec(self, module: str) -> object | None:
        if module in self.modules:
            return SimpleNamespace(origin=f"/python/{module.replace('.', '/')}.py")
        return None

    def which(self, name: str) -> str | None:
        return self.executables.get(name)

    def version(self, distribution: str) -> str:
        return self.versions.get(distribution, "")


ALL_MODULES = {
    "ipfs_datasets_py.logic.CEC.dcec_wrapper",
    "ipfs_datasets_py.logic.CEC.shadow_prover_wrapper",
    "ipfs_datasets_py.logic.external_provers.smt.z3_prover_bridge",
    "z3",
    "ipfs_datasets_py.logic.TDFOL.tdfol_prover",
    "ipfs_datasets_py.logic.TDFOL.tdfol_core",
    "ipfs_datasets_py.logic.hammers.premise_selection",
    "ipfs_datasets_py.logic.hammers.reconstruction",
    "ipfs_datasets_py.logic.hammers.portfolio",
}


def _probe(
    discovery: FakeDiscovery,
    **kwargs: Any,
) -> SolverReadinessProbe:
    return SolverReadinessProbe(
        kwargs.pop("config", None),
        find_spec=discovery.find_spec,
        which=discovery.which,
        distribution_version=discovery.version,
        **kwargs,
    )


def test_scaev180proofready_evidence_term_is_declared_and_receipted() -> None:
    assert SCAEV180PROOFREADY == "SCAEV180PROOFREADY"
    assert SCAEV180PROOFREADY_EVIDENCE == SCAEV180PROOFREADY
    assert "exact-reproducible-dcec-z3-tdfol-cec-hammer-availability" in (
        SCAEV180PROOFREADY_COVERAGE
    )
    assert "solver-output-non-authoritative-until-kernel-reconstruction" in (
        SCAEV180PROOFREADY_COVERAGE
    )
    assert "sat-or-capability-never-promoted-to-proof" in SCAEV180PROOFREADY_COVERAGE

    discovery = FakeDiscovery(
        modules=ALL_MODULES,
        executables={"z3": "/opt/z3"},
    )
    report = _probe(discovery).probe()
    payload = report.to_dict()

    assert SCAEV180PROOFREADY in payload["evidence"]["requirement_ids"]
    assert payload["evidence"]["coverage"] == list(SCAEV180PROOFREADY_COVERAGE)
    assert payload["proof_attempted"] is False
    assert payload["proof_success"] is False
    assert json.loads(json.dumps(payload)) == payload


def test_report_covers_every_required_family_exactly() -> None:
    discovery = FakeDiscovery(modules=ALL_MODULES, executables={"z3": "/usr/bin/z3"})
    report = _probe(discovery).probe()

    assert tuple(item.family for item in report.backends) == REQUIRED_SOLVER_FAMILIES
    assert set(report.by_family) == set(REQUIRED_SOLVER_FAMILIES)
    for family in REQUIRED_SOLVER_FAMILIES:
        backend = report.backend(family)
        assert backend.supported is True
        assert backend.status is SolverReadinessStatus.RECONSTRUCTION_READY
        assert backend.proof_success is False
        assert backend.authority is SolverAuthority.NON_AUTHORITATIVE
        assert backend.capability_revision.startswith("solver-capability:sha256:")


def test_missing_backends_are_typed_unsupported_with_proof_success_false() -> None:
    discovery = FakeDiscovery()
    report = probe_solver_readiness(
        find_spec=discovery.find_spec,
        which=discovery.which,
        distribution_version=discovery.version,
    )

    assert report.supported_backends == ()
    for backend in report.backends:
        assert backend.status is SolverReadinessStatus.UNSUPPORTED
        assert backend.supported is False
        assert backend.proof_success is False
        assert backend.reason_code == "backend_unavailable"
        assert "unsupported" in backend.reason
        payload = backend.to_dict()
        assert payload["proof_success"] is False
        assert SCAEV180PROOFREADY in payload["evidence"]["requirement_ids"]

    dcec = report.backend(SolverBackendFamily.DCEC)
    assert "DCEC" not in dcec.reason or "dcec" in dcec.reason
    z3 = report.backend("z3")
    assert z3.status is SolverReadinessStatus.UNSUPPORTED


def test_partial_z3_install_stays_unsupported() -> None:
    # Binding without bridge and without CLI remains unsupported.
    discovery = FakeDiscovery(modules={"z3"})
    z3 = _probe(discovery).probe_family(SolverBackendFamily.Z3)
    assert z3.supported is False
    assert z3.status is SolverReadinessStatus.UNSUPPORTED

    # CLI alone without modules remains unsupported.
    discovery = FakeDiscovery(executables={"z3": "/usr/bin/z3"})
    z3 = _probe(discovery).probe_family(SolverBackendFamily.Z3)
    assert z3.supported is False


def test_readiness_identities_enter_proof_cache_key_material() -> None:
    discovery = FakeDiscovery(modules=ALL_MODULES, executables={"z3": "/opt/z3"})
    report = _probe(discovery).probe()
    material = solver_cache_key_material(report)

    assert material["report_identity"] == report.report_identity
    assert len(material["backends"]) == len(REQUIRED_SOLVER_FAMILIES)
    for item in material["backends"]:
        assert "capability_revision" in item
        assert "readiness_identity" in item
        assert item["authority"] == SolverAuthority.NON_AUTHORITATIVE.value
    assert SCAEV180PROOFREADY in material["evidence"]["requirement_ids"]

    single = solver_cache_key_material(report.backend(SolverBackendFamily.HAMMER))
    assert single["backend"]["family"] == "hammer"
    assert single["backend"]["supported"] is True


def test_solver_sat_never_promoted_without_kernel_reconstruction() -> None:
    discovery = FakeDiscovery(modules=ALL_MODULES, executables={"z3": "/opt/z3"})
    readiness = _probe(discovery).probe_family(SolverBackendFamily.TDFOL)

    composition = production_compose_supported_obligation(
        obligation_id="obligation:tdfol-1",
        readiness=readiness,
        kernel_reconstructed=False,
        solver_result_sat=True,
        proof_attempted=True,
    )
    payload = composition.to_dict()

    assert composition.admitted is False
    assert composition.proof_success is False
    assert composition.authority is SolverAuthority.NON_AUTHORITATIVE
    assert composition.reason_code == "solver_sat_non_authoritative"
    assert SCAEV180PROOFREADY in payload["evidence"]["requirement_ids"]
    assert payload["proof_cache_key_material"]["trust_aware_proof_cache"] is True


def test_production_compose_admits_only_after_kernel_reconstruction() -> None:
    discovery = FakeDiscovery(modules=ALL_MODULES, executables={"z3": "/opt/z3"})
    readiness = _probe(discovery).probe_family(SolverBackendFamily.HAMMER)

    blocked = production_compose_supported_obligation(
        obligation_id="obligation:hammer-1",
        readiness=readiness,
        kernel_reconstructed=True,
        kernel_receipt_id="",
    )
    assert blocked.admitted is False
    assert blocked.reason_code == "kernel_receipt_missing"
    assert blocked.proof_success is False

    admitted = production_compose_supported_obligation(
        obligation_id="obligation:hammer-1",
        readiness=readiness,
        kernel_reconstructed=True,
        kernel_receipt_id="kernel-receipt:sha256:abc",
        solver_result_sat=True,
    )
    assert admitted.admitted is True
    assert admitted.proof_success is True
    assert admitted.authority is SolverAuthority.KERNEL_AUTHORITATIVE
    assert admitted.reason_code == "kernel_reconstructed_admitted"
    assert admitted.proof_cache_key_material["solver"]["readiness_identity"] == (
        readiness.readiness_identity
    )
    assert admitted.proof_cache_key_material["kernel_receipt_id"] == (
        "kernel-receipt:sha256:abc"
    )


def test_unsupported_backend_cannot_be_production_composed() -> None:
    discovery = FakeDiscovery()
    readiness = _probe(discovery).probe_family(SolverBackendFamily.DCEC)
    composition = production_compose_supported_obligation(
        obligation_id="obligation:dcec-missing",
        readiness=readiness,
        kernel_reconstructed=True,
        kernel_receipt_id="kernel-receipt:ignored",
    )
    assert composition.admitted is False
    assert composition.proof_success is False
    assert composition.reason_code == "backend_unsupported"


def test_register_supported_backends_excludes_unsupported() -> None:
    discovery = FakeDiscovery(
        modules={
            "ipfs_datasets_py.logic.TDFOL.tdfol_prover",
            "ipfs_datasets_py.logic.TDFOL.tdfol_core",
            "ipfs_datasets_py.logic.hammers.premise_selection",
            "ipfs_datasets_py.logic.hammers.reconstruction",
            "ipfs_datasets_py.logic.hammers.portfolio",
        }
    )
    report = _probe(discovery).probe()
    supported = register_supported_backends(report)
    families = {item.family for item in supported}
    assert SolverBackendFamily.TDFOL in families
    assert SolverBackendFamily.HAMMER in families
    assert SolverBackendFamily.DCEC not in families
    assert SolverBackendFamily.Z3 not in families
    assert SolverBackendFamily.CEC not in families


def test_readiness_receipt_forbids_kernel_authority_and_proof_success() -> None:
    with pytest.raises(ValueError, match="kernel authority"):
        SolverBackendReadiness(
            family=SolverBackendFamily.Z3,
            status=SolverReadinessStatus.RECONSTRUCTION_READY,
            provider_id="z3",
            capability_revision="solver-capability:sha256:x",
            package_version="1",
            observations=(),
            reconstruction_compatible=True,
            reason_code="x",
            reason="x",
            supported=True,
            authority=SolverAuthority.KERNEL_AUTHORITATIVE,
        )


def test_self_test_failure_keeps_backend_unsupported() -> None:
    discovery = FakeDiscovery(modules=ALL_MODULES, executables={"z3": "/opt/z3"})

    def fail(_family: SolverBackendFamily) -> bool:
        return False

    probe = _probe(
        discovery,
        config=SolverReadinessProbeConfig(require_self_test=True),
        self_test=fail,
    )
    readiness = probe.probe_family(SolverBackendFamily.CEC)
    assert readiness.supported is False
    assert readiness.reason_code == "self_test_failed"
    assert readiness.proof_success is False


def test_probe_is_reproducible_for_identical_discovery() -> None:
    discovery = FakeDiscovery(modules=ALL_MODULES, executables={"z3": "/opt/z3"})
    first = _probe(discovery).probe().to_dict()
    second = _probe(discovery).probe().to_dict()
    # Drop report_identity which is derived from payload including itself only
    # after first serialization path; compare backends and evidence.
    assert first["backends"] == second["backends"]
    assert first["evidence"] == second["evidence"]
    assert first["supported_families"] == second["supported_families"]


def test_from_dict_round_trip() -> None:
    discovery = FakeDiscovery(modules=ALL_MODULES, executables={"z3": "/bin/z3"})
    original = _probe(discovery).probe_family(SolverBackendFamily.DCEC)
    restored = SolverBackendReadiness.from_dict(original.to_dict())
    assert restored.family is original.family
    assert restored.status is original.status
    assert restored.capability_revision == original.capability_revision
    assert restored.readiness_identity == original.readiness_identity
