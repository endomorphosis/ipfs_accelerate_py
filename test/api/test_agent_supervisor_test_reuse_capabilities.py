from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.integrations.test_reuse_capabilities import (
    TEST_REUSE_CACHE_MODULE,
    TEST_REUSE_CAPABILITY_REPORT_SCHEMA,
    _find_spec_without_import,
    probe_test_reuse_capabilities,
)
from ipfs_accelerate_py.agent_supervisor.integrations.test_reuse_capabilities import (
    TestReuseCapabilityName as CapabilityName,
)
from ipfs_accelerate_py.agent_supervisor.integrations.test_reuse_capabilities import (
    TestReuseCapabilityProbe as CapabilityProbe,
)
from ipfs_accelerate_py.agent_supervisor.integrations.test_reuse_capabilities import (
    TestReuseCapabilityProbeConfig as CapabilityProbeConfig,
)
from ipfs_accelerate_py.agent_supervisor.integrations.test_reuse_capabilities import (
    TestReuseCapabilityStatus as CapabilityStatus,
)

ALL_MODULES = {
    "multiformats",
    "ipfs_datasets_py.logic.zkp",
    "ipfs_datasets_py.logic.zkp.backends.groth16",
    "ipfs_datasets_py.logic.zkp.backends.provekit",
    TEST_REUSE_CACHE_MODULE,
    "ipfshttpclient",
    "ipfs_datasets_py.logic.zkp.zkp_verifier",
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


class FailsOnMappingAccess(Mapping[str, Any]):
    """Injected mapping that proves construction does not inspect providers."""

    def __getitem__(self, key: str) -> Any:
        raise AssertionError(f"mapping accessed during construction: {key}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("mapping iterated during construction")

    def __len__(self) -> int:
        raise AssertionError("mapping sized during construction")


class SlowMapping(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        time.sleep(0.2)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


class HostileDiscoveryResult:
    """Provider value that must never be interpreted on the probe thread."""

    def __bool__(self) -> bool:
        raise AssertionError("provider truth conversion escaped the probe boundary")

    def __str__(self) -> str:
        raise AssertionError("provider string conversion escaped the probe boundary")


def _available_config(tmp_path: Path) -> CapabilityProbeConfig:
    groth16 = tmp_path / "groth16"
    provekit = tmp_path / "provekit"
    cache = tmp_path / "cache"
    for directory in (groth16, provekit, cache):
        directory.mkdir()
    verifier_key = tmp_path / "verifier.key"
    verifier_circuit = tmp_path / "verifier.circuit"
    verifier_key.write_bytes(b"key")
    verifier_circuit.write_bytes(b"circuit")
    return CapabilityProbeConfig(
        groth16_artifacts_path=str(groth16),
        provekit_artifacts_path=str(provekit),
        cache_path=str(cache),
        local_verifier_key_path=str(verifier_key),
        local_verifier_circuit_path=str(verifier_circuit),
    )


def test_construction_is_lazy() -> None:
    calls: list[str] = []

    def unexpected(value: str) -> None:
        calls.append(value)
        raise AssertionError("discovery must not run during construction")

    CapabilityProbe(
        find_spec=unexpected,
        which=unexpected,
        path_is_file=unexpected,
        path_is_dir=unexpected,
        environ={},
        backend_registry=FailsOnMappingAccess(),
        capability_metadata=FailsOnMappingAccess(),
    )

    assert calls == []


def test_off_mode_performs_zero_discovery_and_is_non_blocking() -> None:
    calls: list[str] = []

    def unexpected(value: str) -> None:
        calls.append(value)
        raise AssertionError("disabled probing must not inspect optional providers")

    report = CapabilityProbe(
        find_spec=unexpected,
        which=unexpected,
        path_is_file=unexpected,
        path_is_dir=unexpected,
        environ={"IPFS_TEST_PROOF_REUSE_MODE": "OFF"},
    ).probe()

    assert calls == []
    assert report.probe_count == 0
    assert tuple(fact.capability_id for fact in report.capabilities) == tuple(CapabilityName)
    assert {fact.status for fact in report.capabilities} == {CapabilityStatus.DISABLED}
    assert all(
        fact.optional and fact.non_blocking and not fact.blocking and fact.test_action == "run"
        for fact in report.capabilities
    )
    assert report.unavailable_is_non_blocking
    payload = report.to_dict()
    assert payload["schema_version"] == TEST_REUSE_CAPABILITY_REPORT_SCHEMA
    assert payload["lazy"] is True
    assert payload["bounded"] is True
    assert payload["side_effect_free"] is True
    assert payload["network_attempted"] is False
    assert payload["daemon_started"] is False
    assert payload["cache_created"] is False
    assert json.loads(json.dumps(payload)) == payload


def test_all_cold_prerequisites_are_available_and_deterministic(
    tmp_path: Path,
) -> None:
    config = _available_config(tmp_path)
    registry = {
        "groth16": {"api_version": 1, "available": True},
        "provekit": {"api_version": "1", "available": True},
    }

    def make_report() -> object:
        discovery = FakeDiscovery(
            modules=ALL_MODULES,
            executables={
                "groth16": "/opt/bin/groth16",
                "provekit-cli": "/opt/bin/provekit-cli",
            },
        )
        return CapabilityProbe(
            config,
            find_spec=discovery.find_spec,
            which=discovery.which,
            environ={},
            backend_registry=registry,
        ).probe()

    first = make_report()
    second = make_report()

    assert all(fact.status is CapabilityStatus.AVAILABLE for fact in first.capabilities)
    assert first.to_dict() == second.to_dict()
    assert first.probe_count <= config.max_checks


def test_missing_optional_capabilities_fall_through_to_test_execution() -> None:
    discovery = FakeDiscovery()
    report = CapabilityProbe(
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()

    assert {fact.status for fact in report.capabilities} == {CapabilityStatus.MISSING}
    assert report.all_optional
    assert report.unavailable_is_non_blocking
    assert all(fact.test_action == "run" for fact in report.capabilities)
    assert report.capability("local_verifier").reason_code == ("local_verifier_not_configured")


def test_cache_probe_targets_live_certificate_store() -> None:
    discovery = FakeDiscovery(modules={TEST_REUSE_CACHE_MODULE})

    report = CapabilityProbe(
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()

    cache = report.capability("cache")
    assert cache.status is CapabilityStatus.AVAILABLE
    assert [item.subject for item in cache.evidence] == [
        TEST_REUSE_CACHE_MODULE
    ]
    assert TEST_REUSE_CACHE_MODULE in discovery.package_calls
    assert not any(
        "mcp_contract_proof_cache" in module
        for module in discovery.package_calls
    )


def test_metadata_distinguishes_every_status_without_provider_discovery() -> None:
    discovery = FakeDiscovery()
    metadata = {
        "multiformats": {"available": True},
        "datasets_zk": {"enabled": False},
        "groth16": {"schema_version": "TestReuseCapability@2"},
        "provekit": {},
    }
    report = CapabilityProbe(
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
        capability_metadata=metadata,
    ).probe()

    assert report.capability("multiformats").status is (CapabilityStatus.AVAILABLE)
    assert report.capability("datasets_zk").status is (CapabilityStatus.DISABLED)
    assert report.capability("groth16").status is (CapabilityStatus.INCOMPATIBLE)
    assert report.capability("provekit").status is (CapabilityStatus.UNKNOWN)
    assert report.capability("cache").status is CapabilityStatus.MISSING
    assert "multiformats" not in discovery.package_calls
    assert "ipfs_datasets_py.logic.zkp" not in discovery.package_calls
    assert "groth16" not in discovery.executable_calls
    assert "provekit-cli" not in discovery.executable_calls
    assert report.unavailable_is_non_blocking


def test_registry_incompatibility_and_absence_short_circuit_backend_probes() -> None:
    discovery = FakeDiscovery(modules=ALL_MODULES)
    report = CapabilityProbe(
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
        backend_registry={"groth16": {"api_version": 2}},
    ).probe()

    assert report.capability("groth16").status is (CapabilityStatus.INCOMPATIBLE)
    assert report.capability("provekit").status is CapabilityStatus.MISSING
    assert "ipfs_datasets_py.logic.zkp.backends.groth16" not in (discovery.package_calls)
    assert "ipfs_datasets_py.logic.zkp.backends.provekit" not in (discovery.package_calls)
    assert "groth16" not in discovery.executable_calls
    assert "provekit-cli" not in discovery.executable_calls


def test_probe_check_limit_is_strict_and_remaining_facts_are_unknown() -> None:
    discovery = FakeDiscovery(modules=ALL_MODULES)
    config = CapabilityProbeConfig(max_checks=1)
    report = CapabilityProbe(
        config,
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()

    assert report.probe_count == 1
    assert report.capability("multiformats").status is (CapabilityStatus.AVAILABLE)
    assert report.capability("datasets_zk").status is (CapabilityStatus.UNKNOWN)
    assert all(
        fact.status
        in {
            CapabilityStatus.AVAILABLE,
            CapabilityStatus.UNKNOWN,
            CapabilityStatus.MISSING,
        }
        for fact in report.capabilities
    )
    assert report.unavailable_is_non_blocking


def test_slow_discovery_is_time_bounded_and_reported_unknown() -> None:
    def slow_find_spec(_module: str) -> None:
        time.sleep(0.2)

    started = time.monotonic()
    report = CapabilityProbe(
        CapabilityProbeConfig(timeout_seconds=0.02),
        find_spec=slow_find_spec,
        which=lambda _name: None,
        environ={},
    ).probe()
    elapsed = time.monotonic() - started

    assert elapsed < 0.15
    assert report.probe_count <= report.capabilities.__len__() * 7
    assert report.capability("multiformats").status is (CapabilityStatus.UNKNOWN)
    assert report.capability("multiformats").reason_code == "probe_timed_out"
    assert report.unavailable_is_non_blocking


def test_broken_discovery_hooks_are_isolated_as_unknown() -> None:
    def broken_find_spec(_module: str) -> None:
        raise RuntimeError("unstable third-party finder")

    report = CapabilityProbe(
        find_spec=broken_find_spec,
        which=lambda _name: None,
        environ={},
    ).probe()

    fact = report.capability("multiformats")
    assert fact.status is CapabilityStatus.UNKNOWN
    assert fact.reason_code == "probe_failed"
    assert "unstable" not in json.dumps(report.to_dict())
    assert report.unavailable_is_non_blocking


@pytest.mark.parametrize("hook", ("which", "path_is_file", "path_is_dir"))
def test_malformed_discovery_results_are_isolated_as_unknown(hook: str) -> None:
    hostile = HostileDiscoveryResult()
    kwargs: dict[str, Any] = {
        "find_spec": lambda _module: None,
        "which": lambda _name: None,
        "path_is_file": lambda _path: False,
        "path_is_dir": lambda _path: False,
        "environ": {},
    }
    config = CapabilityProbeConfig()
    if hook == "which":
        kwargs["which"] = lambda _name: hostile
    elif hook == "path_is_file":
        kwargs["environ"] = {"IPFS_DATASETS_GROTH16_BINARY": "/configured/groth16"}
        kwargs["path_is_file"] = lambda _path: hostile
    else:
        config = CapabilityProbeConfig(groth16_artifacts_path="/configured/groth16")
        kwargs["path_is_dir"] = lambda _path: hostile

    report = CapabilityProbe(config, **kwargs).probe()

    fact = report.capability("groth16")
    assert fact.status is CapabilityStatus.UNKNOWN
    assert fact.reason_code == "probe_failed"
    assert fact.test_action == "run"
    assert report.unavailable_is_non_blocking


def test_broken_provider_mappings_are_isolated_as_unknown() -> None:
    metadata_report = CapabilityProbe(
        find_spec=lambda _module: None,
        which=lambda _name: None,
        environ={},
        capability_metadata=FailsOnMappingAccess(),
    ).probe()

    assert {fact.status for fact in metadata_report.capabilities} == {
        CapabilityStatus.UNKNOWN
    }
    assert {fact.reason_code for fact in metadata_report.capabilities} == {"probe_failed"}
    assert metadata_report.unavailable_is_non_blocking

    registry_report = CapabilityProbe(
        find_spec=lambda _module: None,
        which=lambda _name: None,
        environ={},
        backend_registry=FailsOnMappingAccess(),
    ).probe()

    assert registry_report.capability("groth16").status is CapabilityStatus.UNKNOWN
    assert registry_report.capability("provekit").status is CapabilityStatus.UNKNOWN
    assert registry_report.capability("groth16").reason_code == "probe_failed"
    assert registry_report.capability("provekit").reason_code == "probe_failed"
    assert registry_report.unavailable_is_non_blocking


def test_slow_provider_metadata_is_time_bounded_and_unknown() -> None:
    started = time.monotonic()
    report = CapabilityProbe(
        CapabilityProbeConfig(timeout_seconds=0.02),
        find_spec=lambda _module: None,
        which=lambda _name: None,
        environ={},
        capability_metadata=SlowMapping(),
    ).probe()
    elapsed = time.monotonic() - started

    assert elapsed < 0.15
    assert report.probe_count == 1
    assert {fact.status for fact in report.capabilities} == {CapabilityStatus.UNKNOWN}
    assert report.capability("multiformats").reason_code == "probe_timed_out"
    assert all(
        fact.reason_code in {"probe_timed_out", "probe_budget_exhausted"}
        for fact in report.capabilities
    )
    assert report.unavailable_is_non_blocking


def test_slow_configuration_mapping_is_time_bounded_and_unknown() -> None:
    started = time.monotonic()
    report = CapabilityProbe(
        CapabilityProbeConfig(timeout_seconds=0.02),
        find_spec=lambda _module: None,
        which=lambda _name: None,
        environ=SlowMapping(),
    ).probe()
    elapsed = time.monotonic() - started

    assert elapsed < 0.15
    assert report.probe_count == 1
    assert {fact.status for fact in report.capabilities} == {CapabilityStatus.UNKNOWN}
    assert {fact.reason_code for fact in report.capabilities} == {"probe_timed_out"}
    assert report.unavailable_is_non_blocking


def test_local_verifier_requires_module_key_and_circuit(tmp_path: Path) -> None:
    key = tmp_path / "verifier.key"
    key.write_bytes(b"key")
    discovery = FakeDiscovery(modules=ALL_MODULES)
    report = CapabilityProbe(
        CapabilityProbeConfig(local_verifier_key_path=str(key)),
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()

    fact = report.capability("local_verifier")
    assert fact.status is CapabilityStatus.MISSING
    assert fact.test_action == "run"
    assert "ipfs_datasets_py.logic.zkp.zkp_verifier" not in discovery.package_calls


def test_default_dotted_discovery_does_not_import_parent_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "reuse_side_effect_provider"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise RuntimeError('initializer executed')\n",
        encoding="utf-8",
    )
    (package / "backend.py").write_text("AVAILABLE = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("reuse_side_effect_provider", None)

    spec = _find_spec_without_import("reuse_side_effect_provider.backend")

    assert spec is not None
    assert spec.origin and spec.origin.endswith("backend.py")
    assert "reuse_side_effect_provider" not in sys.modules


@pytest.mark.parametrize(
    "kwargs",
    (
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
        {"max_checks": 0},
        {"max_checks": True},
        {"disabled_capabilities": frozenset({"not-a-capability"})},
        {"cache_path": "   "},
    ),
)
def test_invalid_probe_bounds_and_configuration_are_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CapabilityProbeConfig(**kwargs)


def test_convenience_function_returns_an_uncached_typed_snapshot() -> None:
    discovery = FakeDiscovery()
    first = probe_test_reuse_capabilities(
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={"IPFS_TEST_PROOF_REUSE_MODE": "off"},
    )
    second = probe_test_reuse_capabilities(
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={"IPFS_TEST_PROOF_REUSE_MODE": "off"},
    )

    assert first is not second
    assert first.to_dict() == second.to_dict()
    assert discovery.package_calls == []
    assert discovery.executable_calls == []
