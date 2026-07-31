"""LPR-032 tests for lazy datasets doctor Logic capability probes."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations import (
    ipfs_datasets_doctor_logic as doctor_logic,
)
from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_doctor_logic import (
    DatasetsDoctorLogicCapability,
    DatasetsDoctorLogicDiagnosticCode,
    DatasetsDoctorLogicStatus,
    IpfsDatasetsDoctorLogic,
    get_default_doctor_logic,
    module_path_available,
    probe_datasets_doctor_logic_capabilities,
)


def test_cold_construction_is_lazy_and_side_effect_free() -> None:
    original_home = os.environ.get("HOME")
    original_prefix = sys.prefix
    facade = get_default_doctor_logic()
    assert facade.last_report is None
    declaration = facade.capability_declaration()
    assert not declaration.available
    assert (
        declaration.reason_code
        == DatasetsDoctorLogicDiagnosticCode.LAZY_NOT_PROBED.value
    )
    assert os.environ.get("HOME") == original_home
    assert sys.prefix == original_prefix


def test_probe_loads_exact_supervisor_capabilities_without_install_or_network() -> None:
    original_home = os.environ.get("HOME")
    original_prefix = sys.prefix
    report = probe_datasets_doctor_logic_capabilities()
    assert report.install_attempted is False
    assert report.network_attempted is False
    assert report.target_import_attempted is False
    assert report.process_global_mutation is False
    assert os.environ.get("HOME") == original_home
    assert sys.prefix == original_prefix

    cache_cap = report.get("doctor.proof_cache")
    assert cache_cap is not None
    assert cache_cap.available
    assert cache_cap.proof_authority
    assert "lookup" in cache_cap.operations
    assert "revalidate_for_render" in cache_cap.operations
    assert "invalidate_semantic_root" in cache_cap.operations

    formal = report.get("doctor.formal_verification_cache")
    assert formal is not None
    assert formal.available

    identity = report.get("doctor.content_identity_bridge")
    assert identity is not None
    assert identity.available
    assert identity.interface_version == "ContentIdentityBridge@1"

    isolation = report.get("doctor.import_isolation")
    assert isolation is not None
    assert isolation.available
    assert isolation.details.get("import_isolation") == doctor_logic.IMPORT_ISOLATION_HARDENED

    # Capability declarations never claim semantic/completion authority.
    for cap in report.capabilities:
        assert cap.semantic_authority is False
        assert cap.completion_authority is False
        assert cap.candidate_authoritative is False


def test_probe_report_round_trip_dict() -> None:
    report = probe_datasets_doctor_logic_capabilities()
    payload = report.to_dict()
    assert payload["schema"] == doctor_logic.DOCTOR_LOGIC_CAPABILITY_REPORT_SCHEMA
    assert payload["install_attempted"] is False
    assert payload["network_attempted"] is False
    assert "doctor.proof_cache" in payload["available_ids"]


def test_facade_probe_is_idempotent_and_open_cache(tmp_path: Path) -> None:
    facade = IpfsDatasetsDoctorLogic()
    first = facade.probe()
    second = facade.probe()
    assert first is second

    forced = facade.probe(force=True)
    assert forced is not first

    gate = facade.open_doctor_proof_cache(tmp_path)
    from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import (
        DoctorProofCacheGate,
    )

    assert isinstance(gate, DoctorProofCacheGate)


def test_load_module_after_capability_probe() -> None:
    facade = IpfsDatasetsDoctorLogic()
    module = facade.load_module("doctor.proof_cache")
    assert hasattr(module, "DoctorProofCacheGate")
    assert hasattr(module, "DoctorCacheAuditReceipt")


def test_missing_capability_fails_closed() -> None:
    facade = IpfsDatasetsDoctorLogic()
    cap = facade.ensure("doctor.nonexistent.capability")
    assert not cap.available
    assert (
        cap.diagnostic is not None
        and cap.diagnostic.code
        is DatasetsDoctorLogicDiagnosticCode.MODULE_PATH_UNAVAILABLE
    )
    with pytest.raises(LookupError):
        facade.load_module("doctor.nonexistent.capability")


def test_injected_importer_timeout_is_diagnostic() -> None:
    def slow_importer(name: str):
        raise TimeoutError(f"import of {name} exceeded budget")

    # Use a non-optional required-symbol path so timeout is exercised.
    report = probe_datasets_doctor_logic_capabilities(
        importer=slow_importer,
        timeout_seconds=1.0,
        allow_optional_path_only=False,
    )
    # At least one capability should surface as timed_out or unavailable via timeout.
    statuses = {cap.status for cap in report.capabilities}
    assert (
        DatasetsDoctorLogicStatus.TIMED_OUT in statuses
        or DatasetsDoctorLogicStatus.UNAVAILABLE in statuses
    )
    timed = [
        cap
        for cap in report.capabilities
        if cap.diagnostic
        and cap.diagnostic.code is DatasetsDoctorLogicDiagnosticCode.PROBE_TIMED_OUT
    ]
    # Isolation is synthetic and may still be available.
    assert timed or any(
        cap.status is DatasetsDoctorLogicStatus.TIMED_OUT for cap in report.capabilities
    )


def test_injected_importer_missing_symbols_is_incompatible() -> None:
    empty = types.ModuleType("empty_doctor_logic_stub")

    def importer(name: str):
        return empty

    report = probe_datasets_doctor_logic_capabilities(
        importer=importer,
        allow_optional_path_only=False,
    )
    cache_cap = report.get("doctor.proof_cache")
    assert cache_cap is not None
    assert not cache_cap.available
    assert cache_cap.status is DatasetsDoctorLogicStatus.INCOMPATIBLE
    assert (
        cache_cap.diagnostic is not None
        and cache_cap.diagnostic.code
        is DatasetsDoctorLogicDiagnosticCode.REQUIRED_SYMBOL_MISSING
    )


def test_process_global_mutation_during_import_is_unsafe() -> None:
    def mutator(name: str):
        # Leave HOME mutated so the probe guard fails closed.
        os.environ["HOME"] = "/tmp/doctor-logic-forbidden-home"
        return types.ModuleType(name)

    original_home = os.environ.get("HOME")
    try:
        report = probe_datasets_doctor_logic_capabilities(
            importer=mutator,
            allow_optional_path_only=False,
        )
        unsafe = [
            cap
            for cap in report.capabilities
            if cap.status is DatasetsDoctorLogicStatus.UNSAFE
            or (
                cap.diagnostic is not None
                and cap.diagnostic.code
                is DatasetsDoctorLogicDiagnosticCode.PROCESS_GLOBAL_MUTATION_FORBIDDEN
            )
        ]
        assert unsafe
        # HOME must be restored even when the importer mutates it.
        assert os.environ.get("HOME") == original_home
    finally:
        if original_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = original_home


def test_capability_rejects_authority_claims() -> None:
    with pytest.raises(ValueError):
        DatasetsDoctorLogicCapability(
            capability_id="bad",
            status=DatasetsDoctorLogicStatus.AVAILABLE,
            module_paths=("some.module",),
            semantic_authority=True,
        )
    with pytest.raises(ValueError):
        DatasetsDoctorLogicCapability(
            capability_id="bad2",
            status=DatasetsDoctorLogicStatus.AVAILABLE,
            module_paths=("some.module",),
            candidate_authoritative=True,
        )


def test_module_path_available_for_supervisor_modules() -> None:
    assert module_path_available(
        "ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache"
    )
    assert module_path_available(
        "ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache"
    )
    assert not module_path_available(
        "ipfs_accelerate_py.agent_supervisor.definitely_missing_module_xyz"
    )


def test_isolated_hammer_loader_accessible_without_unsafe_mutation() -> None:
    original_home = os.environ.get("HOME")
    original_prefix = sys.prefix
    facade = IpfsDatasetsDoctorLogic()
    loader = facade.get_isolated_hammer_loader()
    report = loader.isolation_report()
    assert report.get("mutates_home") is False
    assert report.get("mutates_sys_prefix") is False
    assert report.get("concurrency_safe") is True
    assert os.environ.get("HOME") == original_home
    assert sys.prefix == original_prefix


def test_isolation_report_summary() -> None:
    facade = IpfsDatasetsDoctorLogic()
    summary = facade.isolation_report()
    assert summary["install_attempted"] is False
    assert summary["network_attempted"] is False
    assert summary["target_import_attempted"] is False
    assert summary["process_global_mutation"] is False
    assert summary["probe_isolation"] is not None
    assert summary["probe_isolation"]["available"] is True


def test_importlib_reload_preserves_public_surface() -> None:
    reloaded = importlib.reload(doctor_logic)
    report = reloaded.probe_datasets_doctor_logic_capabilities()
    assert report.get("doctor.proof_cache") is not None
    assert report.get("doctor.proof_cache").available
