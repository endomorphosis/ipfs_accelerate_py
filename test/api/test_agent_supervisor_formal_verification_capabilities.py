from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor import (
    FORMAL_VERIFICATION_CAPABILITY_REPORT_VERSION,
    FORMAL_VERIFICATION_CAPABILITY_SCHEMA_VERSION,
    CapabilityDimension,
    CapabilityHealth,
    FormalVerificationCapabilityProbe,
    FormalVerificationProbeConfig,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_capabilities import (
    _find_spec_without_import,
)


EXPECTED_PROVIDERS = {
    "hammer",
    "tdfol",
    "external_provers",
    "lean",
    "leanstral",
    "frame_logic",
    "knowledge_graphs",
    "zkp_backends",
}


class FakeDiscovery:
    def __init__(
        self,
        *,
        modules: set[str] | None = None,
        executables: dict[str, str] | None = None,
    ) -> None:
        self.modules = modules or set()
        self.executables = executables or {}
        self.package_calls: list[str] = []
        self.executable_calls: list[str] = []

    def find_spec(self, module: str) -> object | None:
        self.package_calls.append(module)
        if module in self.modules:
            return SimpleNamespace(origin=f"/python/{module.replace('.', '/')}.py")
        return None

    def which(self, executable: str) -> str | None:
        self.executable_calls.append(executable)
        return self.executables.get(executable)

    @staticmethod
    def version(distribution: str) -> str:
        return f"{distribution}-test-version"


class MutableClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _all_modules() -> set[str]:
    return {
        "ipfs_datasets_py.logic.hammers",
        "ipfs_datasets_py.logic.hammers.learned_selector",
        "ipfs_datasets_py.logic.TDFOL",
        "ipfs_datasets_py.logic.external_provers",
        "ipfs_datasets_py.logic.external_provers.interactive.lean_prover_bridge",
        "ipfs_datasets_py.logic.modal.leanstral",
        "ipfs_datasets_py.logic.flogic",
        "ipfs_datasets_py.knowledge_graphs",
        "ipfs_datasets_py.logic.zkp.backends",
        "ipfs_datasets_py.logic.zkp.circuits",
        "spacy",
        "en_core_web_sm",
        "transformers",
        "torch",
        "networkx",
        "z3",
        "cvc5",
        "symai",
        "py_ecc",
    }


def _artifact_directories(tmp_path: Path) -> tuple[Path, Path, Path]:
    leanstral = tmp_path / "leanstral-model"
    leanstral.mkdir()
    (leanstral / "config.json").write_text("{}", encoding="utf-8")
    (leanstral / "model.safetensors").write_bytes(b"weights")

    groth16 = tmp_path / "groth16"
    groth16.mkdir()
    (groth16 / "proving_key.bin").write_bytes(b"pk")
    (groth16 / "verifying_key.bin").write_bytes(b"vk")

    provekit = tmp_path / "provekit"
    provekit.mkdir()
    (provekit / "manifest.json").write_text("{}", encoding="utf-8")
    (provekit / "circuit.pkp").write_bytes(b"prover")
    (provekit / "circuit.pkv").write_bytes(b"verifier")
    return leanstral, groth16, provekit


def test_report_is_versioned_and_covers_every_formal_logic_provider(
    tmp_path: Path,
) -> None:
    leanstral, groth16, provekit = _artifact_directories(tmp_path)
    discovery = FakeDiscovery(
        modules=_all_modules(),
        executables={
            name: f"/usr/bin/{name}"
            for name in (
                "eprover",
                "vampire",
                "z3",
                "cvc5",
                "coqtop",
                "lean",
                "runergo",
                "groth16",
                "provekit-cli",
            )
        },
    )
    probe = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(
            leanstral_model_path=str(leanstral),
            groth16_artifacts_path=str(groth16),
            provekit_artifacts_path=str(provekit),
        ),
        find_spec=discovery.find_spec,
        which=discovery.which,
        distribution_version=discovery.version,
        environ={},
    )

    report = probe.probe()
    payload = report.to_dict()

    assert report.schema_version == FORMAL_VERIFICATION_CAPABILITY_SCHEMA_VERSION
    assert report.report_version == FORMAL_VERIFICATION_CAPABILITY_REPORT_VERSION
    assert set(report.capabilities) == EXPECTED_PROVIDERS
    assert set(payload["providers"]) == EXPECTED_PROVIDERS
    assert payload["bounded"] is True
    assert report.probe_count <= probe.config.max_checks
    assert json.loads(json.dumps(payload)) == payload


def test_each_health_dimension_is_serialized_separately(tmp_path: Path) -> None:
    leanstral, groth16, provekit = _artifact_directories(tmp_path)
    discovery = FakeDiscovery(
        modules=_all_modules(),
        executables={
            "lean": "/opt/lean",
            "eprover": "/opt/eprover",
            "runergo": "/opt/runergo",
            "groth16": "/opt/groth16",
            "provekit-cli": "/opt/provekit-cli",
        },
    )
    report = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(
            leanstral_model_path=str(leanstral),
            groth16_artifacts_path=str(groth16),
            provekit_artifacts_path=str(provekit),
        ),
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()

    expected_dimensions = {dimension.value for dimension in CapabilityDimension}
    for provider in report.providers:
        assert set(provider.to_dict()["health"]) == expected_dimensions
        assert provider.provider_health

    assert report.provider("hammer").executable_health
    assert report.provider("external_provers").package_health
    assert report.provider("leanstral").model_health
    assert report.provider("zkp_backends").circuit_health
    assert report.provider("knowledge_graphs").optional_dependency_health


def test_missing_optional_packages_models_bindings_and_binaries_are_explicit() -> None:
    discovery = FakeDiscovery()
    report = FormalVerificationCapabilityProbe(
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()

    tdfol = report.provider("tdfol")
    assert tdfol.status is CapabilityHealth.UNAVAILABLE
    assert "not importable" in tdfol.reason
    assert "spaCy is not installed" in tdfol.optional_dependency_health[0].reason
    assert "model weights" in tdfol.model_health[0].reason

    external = report.provider("external_provers")
    bindings = {check.name: check for check in external.package_health}
    assert bindings["Z3 Python binding"].status is CapabilityHealth.UNAVAILABLE
    assert "bindings are not installed" in bindings["Z3 Python binding"].reason
    assert bindings["CVC5 Python binding"].status is CapabilityHealth.UNAVAILABLE

    lean = report.provider("lean")
    assert lean.status is CapabilityHealth.UNAVAILABLE
    assert "executable" in lean.executable_health[0].reason

    frame = report.provider("frame_logic")
    assert "simulation mode" in frame.executable_health[0].reason

    leanstral = report.provider("leanstral")
    assert "model weights are not configured" in leanstral.model_health[0].reason


def test_core_provider_degrades_when_only_optional_nlp_health_is_missing() -> None:
    modules = {
        "ipfs_datasets_py.logic.TDFOL",
        "ipfs_datasets_py.knowledge_graphs",
    }
    discovery = FakeDiscovery(modules=modules)
    report = FormalVerificationCapabilityProbe(
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()

    assert report.provider("tdfol").status is CapabilityHealth.DEGRADED
    assert report.provider("knowledge_graphs").status is CapabilityHealth.DEGRADED
    assert "optional" in report.provider("tdfol").reason
    assert "NLP/model" in report.provider("knowledge_graphs").reason


def test_python_discovery_failures_do_not_escape_or_import_providers() -> None:
    calls: list[str] = []

    def broken_find_spec(module: str) -> None:
        calls.append(module)
        raise RuntimeError("broken import hook")

    report = FormalVerificationCapabilityProbe(
        find_spec=broken_find_spec,
        which=lambda _name: None,
        environ={},
    ).probe()

    assert calls
    assert set(report.capabilities) == EXPECTED_PROVIDERS
    assert all(
        provider.status in {CapabilityHealth.DEGRADED, CapabilityHealth.UNAVAILABLE}
        for provider in report.providers
    )
    assert "failed safely" in report.provider("hammer").package_health[0].reason


def test_default_dotted_package_discovery_does_not_execute_parent_packages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "side_effect_provider"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise RuntimeError('package initializer executed')\n",
        encoding="utf-8",
    )
    (package / "backend.py").write_text("CAPABLE = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("side_effect_provider", None)

    spec = _find_spec_without_import("side_effect_provider.backend")

    assert spec is not None
    assert spec.origin and spec.origin.endswith("backend.py")
    assert "side_effect_provider" not in sys.modules


def test_probe_is_cached_until_forced_or_ttl_expires() -> None:
    discovery = FakeDiscovery()
    clock = MutableClock()
    probe = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(cache_ttl_seconds=10),
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
        monotonic=clock,
        wall_clock=clock,
    )

    first = probe.probe()
    package_calls = len(discovery.package_calls)
    second = probe.probe()
    assert second is first
    assert len(discovery.package_calls) == package_calls

    forced = probe.probe(force_refresh=True)
    assert forced is not first
    assert len(discovery.package_calls) > package_calls

    calls_after_force = len(discovery.package_calls)
    clock.value += 11
    expired = probe.probe()
    assert expired is not forced
    assert len(discovery.package_calls) > calls_after_force

    probe.clear_cache()
    assert probe.probe() is not expired


def test_probe_check_budget_bounds_discovery_work() -> None:
    discovery = FakeDiscovery(modules=_all_modules())
    probe = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(max_checks=1, timeout_seconds=5),
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    )

    report = probe.probe()

    assert report.probe_count == 1
    assert len(discovery.package_calls) + len(discovery.executable_calls) == 1
    limited = [
        check
        for provider in report.providers
        for check in provider.checks
        if check.metadata.get("probe_limited")
    ]
    assert limited
    assert all("limit" in check.reason for check in limited)


def test_probe_wall_clock_budget_bounds_stalled_discovery() -> None:
    def stalled_find_spec(_module: str) -> None:
        time.sleep(0.25)

    probe = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(timeout_seconds=0.03),
        find_spec=stalled_find_spec,
        which=lambda _name: None,
        environ={},
    )

    started = time.monotonic()
    report = probe.probe()
    elapsed = time.monotonic() - started

    assert elapsed < 0.15
    assert report.duration_seconds < 0.15
    assert report.provider("hammer").status is CapabilityHealth.UNAVAILABLE
    assert "timed out safely" in report.provider("hammer").package_health[0].reason


def test_capability_availability_never_becomes_proof_success(
    tmp_path: Path,
) -> None:
    leanstral, groth16, provekit = _artifact_directories(tmp_path)
    discovery = FakeDiscovery(
        modules=_all_modules(),
        executables={
            "lean": "/opt/lean",
            "eprover": "/opt/eprover",
            "runergo": "/opt/runergo",
            "groth16": "/opt/groth16",
            "provekit-cli": "/opt/provekit-cli",
        },
    )
    report = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(
            leanstral_model_path=str(leanstral),
            groth16_artifacts_path=str(groth16),
            provekit_artifacts_path=str(provekit),
        ),
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()

    assert any(provider.available for provider in report.providers)
    assert report.proof_attempted is False
    assert report.proof_success is False
    assert all(provider.proof_success is False for provider in report.providers)
    assert all(
        check.proof_attempted is False and check.proof_success is False
        for provider in report.providers
        for check in provider.checks
    )
    assert report.to_dict()["proof_success"] is False
    assert report.to_dict()["providers"]["lean"]["proof_success"] is False


def test_zkp_backend_health_requires_executable_and_circuit_artifacts(
    tmp_path: Path,
) -> None:
    discovery = FakeDiscovery(
        modules={
            "ipfs_datasets_py.logic.zkp.backends",
            "ipfs_datasets_py.logic.zkp.circuits",
        },
        executables={"groth16": "/opt/groth16"},
    )
    report = FormalVerificationCapabilityProbe(
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()
    zkp = report.provider("zkp_backends")
    provider_checks = {check.name: check for check in zkp.provider_health}

    assert zkp.status is CapabilityHealth.DEGRADED
    assert provider_checks["simulated ZKP backend"].status is CapabilityHealth.DEGRADED
    assert provider_checks["Groth16 backend"].status is CapabilityHealth.UNAVAILABLE
    assert "non-cryptographic" in provider_checks["simulated ZKP backend"].reason
    assert any(
        "not configured" in check.reason for check in zkp.circuit_health
    )

    artifact_dir = tmp_path / "groth16"
    artifact_dir.mkdir()
    (artifact_dir / "proving_key.bin").write_bytes(b"pk")
    (artifact_dir / "verifying_key.bin").write_bytes(b"vk")
    ready = FormalVerificationCapabilityProbe(
        FormalVerificationProbeConfig(groth16_artifacts_path=str(artifact_dir)),
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()

    assert ready.provider("zkp_backends").status is CapabilityHealth.AVAILABLE


def test_invalid_probe_limits_fail_fast() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        FormalVerificationProbeConfig(timeout_seconds=0)
    with pytest.raises(ValueError, match="max_checks"):
        FormalVerificationProbeConfig(max_checks=0)
    with pytest.raises(ValueError, match="cache_ttl_seconds"):
        FormalVerificationProbeConfig(cache_ttl_seconds=-1)
