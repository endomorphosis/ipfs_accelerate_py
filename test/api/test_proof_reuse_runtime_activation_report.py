"""PTR-149: live capability reporting for runtime activation.

Availability must derive from composed typed services and bounded non-mutating
probes.  Reports never install packages or import optional stacks merely to
claim readiness.  Native Groth16 readiness is always separate from
test-certificate authority; knowledge-of-axioms can never satisfy the latter.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.testing.proof_reuse.config import ProofReuseMode
from ipfs_accelerate_py.testing.proof_reuse.reporting import (
    PROOF_REUSE_RUNTIME_ACTIVATION_REPORT_INTERFACE,
    PROOF_REUSE_RUNTIME_ACTIVATION_REPORT_SCHEMA,
    ProofReuseRuntimeActivationReport,
    proof_reuse_runtime_activation_report,
    proof_reuse_runtime_activation_report_from_root,
)
from ipfs_accelerate_py.testing.proof_reuse.services import (
    LIVE_TYPED_SERVICE_PROBE_INTERFACE,
    NATIVE_GROTH16_READINESS_PROBE_INTERFACE,
    PROOF_REUSE_GROTH16_CIRCUIT_REF_ENV,
    TEST_CERTIFICATE_AUTHORITY_PROBE_INTERFACE,
    compose_default_proof_reuse_services,
    live_runtime_activation_inventory,
    probe_live_typed_services,
    probe_native_groth16_readiness,
    probe_test_certificate_authority,
    proof_reuse_dependency_plan,
)


class _RecordingInstaller:
    """Installer that records probe calls and never mutates the environment."""

    def __init__(self, *, ready: bool = False, installed: bool = False) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._ready = ready
        self._installed = installed

    def ensure_groth16_native_backend(self, *, consent: bool = False) -> Any:
        self.calls.append(("ensure_groth16_native_backend", consent))
        return SimpleNamespace(
            available=bool(self._installed and consent is False),
            reason_code="native_present" if self._installed else "native_absent",
        )

    def inspect_groth16_runtime(self) -> dict[str, Any]:
        self.calls.append(("inspect_groth16_runtime", None))
        return {
            "interface": "Groth16RuntimeCapabilityStatus@1",
            "ready": self._ready,
            "readiness_scope": "generic_native_or_endpoint_capability",
            "action": "RUN" if self._ready else "DEFERRED",
            "test_certificate_authority_ready": False,
            "test_certificate_authority_reason": (
                "test_pass_circuit_provider_unavailable"
            ),
            "skip_authority": False,
            "network_attempted": False,
            "process_started": False,
            "trusted_setup_attempted": False,
        }

    def install(self, dependency: Any) -> bool:
        self.calls.append(("install", getattr(dependency, "module_name", dependency)))
        raise AssertionError("activation report must never install packages")


def test_report_symbols_and_schema_are_stable() -> None:
    assert (
        PROOF_REUSE_RUNTIME_ACTIVATION_REPORT_INTERFACE
        == "ProofReuseRuntimeActivationReport@1"
    )
    assert PROOF_REUSE_RUNTIME_ACTIVATION_REPORT_SCHEMA.endswith(
        "proof-reuse-runtime-activation-report@1"
    )


def test_live_report_derives_from_composed_default_services(tmp_path: Path) -> None:
    installer = _RecordingInstaller(ready=False, installed=False)
    services = compose_default_proof_reuse_services(
        mode=ProofReuseMode.WRITE,
        root_path=tmp_path,
        cache_root=tmp_path / "cache",
        installer=installer,
        environ={
            "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
            "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
        },
    )
    report = proof_reuse_runtime_activation_report(
        services=services,
        installer=installer,
        environ={
            "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
            "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
        },
    )
    assert isinstance(report, ProofReuseRuntimeActivationReport)
    assert report.interface == PROOF_REUSE_RUNTIME_ACTIVATION_REPORT_INTERFACE
    assert report.live is True
    assert report.network_attempted is False
    assert report.install_attempted is False
    assert report.import_for_readiness is False
    assert report.prove_attempted is False
    assert report.composition["interface"] == LIVE_TYPED_SERVICE_PROBE_INTERFACE
    assert report.native_groth16["interface"] == NATIVE_GROTH16_READINESS_PROBE_INTERFACE
    assert (
        report.test_certificate_authority["interface"]
        == TEST_CERTIFICATE_AUTHORITY_PROBE_INTERFACE
    )
    # Default composition exposes issuer/lookup/store/revalidator handles.
    handles = report.composition["handles"]
    assert handles["issuer"]["present"] is True
    assert handles["lookup"]["present"] is True
    assert handles["store"]["present"] is True
    assert handles["candidate_store"]["present"] is True
    assert handles["revalidator"]["present"] is True
    # Cold hard-coded plan is not the live source of truth.
    cold = proof_reuse_dependency_plan({})
    assert cold["runtime_activation_live_probe_required"] is True
    assert cold["runtime_activation"].get("live_probe") is not True
    assert "install" not in {name for name, _ in installer.calls}


def test_report_never_installs_or_proves(tmp_path: Path) -> None:
    installer = _RecordingInstaller()
    report = proof_reuse_runtime_activation_report(
        mode=ProofReuseMode.READ,
        root_path=tmp_path,
        cache_root=tmp_path / "cache",
        installer=installer,
        environ={
            "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
            "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
        },
        compose_if_missing=True,
    )
    assert report.install_attempted is False
    assert report.prove_attempted is False
    assert report.network_attempted is False
    for name, arg in installer.calls:
        assert name != "install"
        if name == "ensure_groth16_native_backend":
            assert arg is False  # consent denied — non-mutating


def test_native_groth16_separated_from_test_certificate_authority(
    tmp_path: Path,
) -> None:
    installer = _RecordingInstaller(ready=True, installed=True)
    env = {
        "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
        "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
        PROOF_REUSE_GROTH16_CIRCUIT_REF_ENV: "knowledge_of_axioms@v2",
        "GROTH16_BACKEND_ARTIFACTS_ROOT": str(tmp_path / "missing-artifacts"),
    }
    native = probe_native_groth16_readiness(installer=installer, environ=env)
    certificate = probe_test_certificate_authority(
        installer=installer,
        environ=env,
        artifacts_root=tmp_path / "missing-artifacts",
    )
    assert native["ready"] is True
    assert native["knowledge_of_axioms_circuit"] is True
    assert certificate["ready"] is False
    assert certificate["knowledge_of_axioms_rejected"] is True
    assert "knowledge_of_axioms" in certificate["reason_code"]

    report = proof_reuse_runtime_activation_report(
        mode=ProofReuseMode.OFF,
        root_path=tmp_path,
        cache_root=tmp_path / "cache",
        installer=installer,
        environ=env,
        artifacts_root=tmp_path / "missing-artifacts",
    )
    assert report.native_groth16_ready is True
    assert report.test_certificate_authority_ready is False
    assert report.knowledge_of_axioms_cannot_satisfy_test_certificate_authority is True


def test_knowledge_of_axioms_never_satisfies_certificate_even_with_keys(
    tmp_path: Path,
) -> None:
    # Co-located generic keys under knowledge_of_axioms still cannot be authority.
    artifacts = tmp_path / "artifacts" / "v5"
    artifacts.mkdir(parents=True)
    (artifacts / "proving_key.bin").write_bytes(b"pk-bytes")
    (artifacts / "verifying_key.bin").write_bytes(b"vk-bytes")
    env = {
        PROOF_REUSE_GROTH16_CIRCUIT_REF_ENV: "knowledge_of_axioms@v4",
        "GROTH16_BACKEND_ARTIFACTS_ROOT": str(tmp_path / "artifacts"),
        "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
        "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
    }
    certificate = probe_test_certificate_authority(
        environ=env,
        artifacts_root=tmp_path / "artifacts",
    )
    assert certificate["ready"] is False
    assert (
        certificate["reason_code"]
        == "knowledge_of_axioms_cannot_satisfy_test_certificate_authority"
    )


def test_hostile_self_pinned_v4_manifest_cannot_mark_certificate_ready(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts" / "v5"
    artifacts.mkdir(parents=True)
    (artifacts / "proving_key.bin").write_bytes(b"real-pk-material")
    (artifacts / "verifying_key.bin").write_bytes(b"real-vk-material")
    manifest = tmp_path / "attacker-manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    env = {
        "GROTH16_BACKEND_ARTIFACTS_ROOT": str(tmp_path / "artifacts"),
        "IPFS_TEST_PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST": str(manifest),
        "IPFS_TEST_PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_SHA256": (
            hashlib.sha256(manifest.read_bytes()).hexdigest()
        ),
        "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
        "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
        # No knowledge_of_axioms circuit binding.
    }
    certificate = probe_test_certificate_authority(
        environ=env,
        artifacts_root=tmp_path / "artifacts",
    )
    assert certificate["ready"] is False
    assert certificate["artifact_bindings"]["provenance_ready"] is False
    assert certificate["reason_code"] == "artifact_manifest_unapproved"
    assert certificate["knowledge_of_axioms_circuit"] is False


def test_report_round_trip_dict(tmp_path: Path) -> None:
    report = proof_reuse_runtime_activation_report_from_root(
        tmp_path,
        mode=ProofReuseMode.OFF,
        installer=_RecordingInstaller(),
        environ={
            "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
            "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
        },
    )
    rebuilt = ProofReuseRuntimeActivationReport.from_dict(report.to_dict())
    assert rebuilt.to_dict() == report.to_dict()
    assert rebuilt.live is True


def test_live_inventory_differs_from_cold_hardcoded_plan(tmp_path: Path) -> None:
    cold = proof_reuse_dependency_plan({})["runtime_activation"]
    services = compose_default_proof_reuse_services(
        mode=ProofReuseMode.WRITE,
        root_path=tmp_path,
        cache_root=tmp_path / "cache",
        installer=lambda _dep: False,
        environ={
            "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
            "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
        },
    )
    live = live_runtime_activation_inventory(
        services,
        environ={
            "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
            "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
        },
    )
    assert live["live_probe"] is True
    assert cold.get("live_probe") is not True
    # Live inventory reflects actual composition (issuer present by default).
    assert live["deferred_certificate_issuer_configured"] is True
    assert cold["deferred_certificate_issuer_configured"] is False


def test_probe_live_typed_services_without_services_is_fail_closed() -> None:
    probe = probe_live_typed_services(None)
    assert probe["required_handles_present"] is False
    assert probe["ordinary_default_composition_usable"] is False
    assert probe["import_for_readiness"] is False


def test_module_exports_remain_import_safe() -> None:
    reporting = importlib.import_module(
        "ipfs_accelerate_py.testing.proof_reuse.reporting"
    )
    assert callable(reporting.proof_reuse_runtime_activation_report)
    services = importlib.import_module(
        "ipfs_accelerate_py.testing.proof_reuse.services"
    )
    assert callable(services.probe_test_certificate_authority)
    assert callable(services.live_runtime_activation_inventory)


def test_activation_gap_when_reviewed_keys_absent(tmp_path: Path) -> None:
    """Missing reviewed v4 keys/manifest is an explicit gap, not warm-skip authority."""

    installer = _RecordingInstaller(ready=False, installed=False)
    report = proof_reuse_runtime_activation_report(
        mode=ProofReuseMode.WRITE,
        root_path=tmp_path,
        cache_root=tmp_path / "cache",
        installer=installer,
        environ={
            "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
            "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
        },
        artifacts_root=tmp_path / "missing-artifacts",
    )
    assert report.test_certificate_authority_ready is False
    assert report.activation_gap_present is True
    assert report.activation_gap["present"] is True
    assert report.activation_gap["warm_skip_authorized"] is False
    assert report.activation_gap["closeout_authorized"] is False
    assert report.activation_gap["tests_continue"] is True
    assert report.ordinary_warm_skip_path_complete is False
    assert "activation_gap" in " ".join(report.activation_blocker_codes) or any(
        "activation_gap" in code for code in report.activation_blocker_codes
    )


def test_unmanifested_native_binary_cannot_satisfy_certificate_authority(
    tmp_path: Path,
) -> None:
    installer = _RecordingInstaller(ready=True, installed=True)
    env = {
        "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
        "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
        "GROTH16_BACKEND_ARTIFACTS_ROOT": str(tmp_path / "missing-artifacts"),
    }
    certificate = probe_test_certificate_authority(
        installer=installer,
        environ=env,
        artifacts_root=tmp_path / "missing-artifacts",
    )
    assert certificate["ready"] is False
    assert certificate.get("unmanifested_native_binary_rejected") is True

    report = proof_reuse_runtime_activation_report(
        mode=ProofReuseMode.OFF,
        root_path=tmp_path,
        cache_root=tmp_path / "cache",
        installer=installer,
        environ=env,
        artifacts_root=tmp_path / "missing-artifacts",
    )
    assert report.native_groth16_installed is True or report.native_groth16_ready is True
    assert report.test_certificate_authority_ready is False
    assert (
        report.unmanifested_native_binary_cannot_satisfy_test_certificate_authority
        is True
    )
    assert report.ordinary_warm_skip_path_complete is False
    assert report.activation_gap_present is True
