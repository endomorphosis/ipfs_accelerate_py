"""Focused tests for the operator-owned SAWM R2 controls.

These tests inspect and render controls only. They do not open the live
authority database, start Quack, probe a provider, or launch a supervisor.
"""

from __future__ import annotations

import contextlib
import copy
import ctypes
import errno
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

_SUCCESSOR_CONTROL_KEYS_NEWEST_FIRST = (
    "post_m47_clean_shutdown_restart_successor_materialization",
    "ignored_python_cache_preservation_and_recovery_successor_materialization",
    "legacy_no_delta_rescue_recovery_successor_materialization",
    "failed_pre_authoritative_m44_validation_successor_materialization",
    "post_m43_hardened_procfs_user_manager_restart_successor_materialization",
    "dead_attempt_lifecycle_recovery_restart_successor_materialization",
    "failed_pre_authoritative_m41_evidence_projection_successor_materialization",
    "failed_pre_authoritative_m40_validation_successor_materialization",
    "failed_pre_authoritative_m39_successor_materialization",
    "committed_m38_evidence_reconciliation_successor_materialization",
    "pre_authoritative_custody_restart_successor_materialization",
    "post_reboot_generation_restart_successor_materialization",
    "operator_task_binding_correction_successor_materialization",
    "immutable_authority_identity_normalization_successor_materialization",
    "json_emission_normalization_successor_materialization",
    "live_preflight_contract_successor_materialization",
    "live_preflight_plan_anchor_successor_materialization",
    "detached_coordinator_pid_recovery_successor_materialization",
    "stopped_owner_restart_source_seal_successor_materialization",
    "committed_evidence_verification_successor_materialization",
    "live_claim_admission_recovery_successor_materialization",
    "dead_owner_parallel_resume_successor_materialization",
    "automatic_stall_recovery_successor_materialization",
    "native_duckdb_preload_successor_materialization",
    "multi_lane_sidecar_reopen_successor_materialization",
    "multi_lane_successor_materialization",
    "live_preflight_receipt_compatibility_successor_materialization",
    "generation_realization_successor_materialization",
    "test_isolation_successor_materialization",
    "live_catalog_inventory_successor_materialization",
    "portal_completion_persistence_successor_materialization",
    "source_binding_successor_materialization",
    "accepted_source_retry_successor_materialization",
    "runtime_root_rebind_successor_materialization",
    "stale_owner_restart_successor_materialization",
    "quack_refresh_successor_materialization",
    "declared_output_retry_successor_materialization",
    "live_provider_retry_successor_materialization",
    "live_projection_successor_materialization",
    "live_recovery_successor_materialization",
    "source_repair_successor_materialization",
)


def _historical_successor_controls_at(
    target_key: str,
    scheduler: Mapping[str, object],
    migration: Mapping[str, object] | None = None,
    seal: Mapping[str, object] | None = None,
) -> tuple[
    dict[str, object],
    dict[str, object] | None,
    dict[str, object] | None,
]:
    """Return isolated controls with every successor newer than target removed."""
    assert target_key in _SUCCESSOR_CONTROL_KEYS_NEWEST_FIRST
    target_index = _SUCCESSOR_CONTROL_KEYS_NEWEST_FIRST.index(target_key)
    newer_keys = _SUCCESSOR_CONTROL_KEYS_NEWEST_FIRST[:target_index]
    historical_scheduler = copy.deepcopy(dict(scheduler))
    historical_migration = (
        copy.deepcopy(dict(migration)) if migration is not None else None
    )
    historical_seal = copy.deepcopy(dict(seal)) if seal is not None else None
    for newer_key in newer_keys:
        historical_scheduler.pop(newer_key, None)
        if historical_migration is not None:
            historical_migration.pop(newer_key, None)
        if historical_seal is not None:
            historical_seal.pop(f"{newer_key}_cid", None)
    assert not any(key in historical_scheduler for key in newer_keys)
    if historical_migration is not None:
        assert not any(key in historical_migration for key in newer_keys)
    if historical_seal is not None:
        assert not any(f"{key}_cid" in historical_seal for key in newer_keys)
    return historical_scheduler, historical_migration, historical_seal


def test_historical_successor_controls_include_m40() -> None:
    """Historical fixtures must remove M40 from every protected surface."""

    m40_key = "failed_pre_authoritative_m39_successor_materialization"
    m39_key = "committed_m38_evidence_reconciliation_successor_materialization"
    scheduler = {m40_key: {"revision": "M40"}, m39_key: {"revision": "M39"}}
    migration = copy.deepcopy(scheduler)
    seal = {
        f"{m40_key}_cid": "sha256:" + "4" * 64,
        f"{m39_key}_cid": "sha256:" + "3" * 64,
    }

    current, current_migration, current_seal = _historical_successor_controls_at(
        m40_key, scheduler, migration, seal
    )
    assert current[m40_key] == scheduler[m40_key]
    assert current_migration is not None
    assert current_migration[m40_key] == migration[m40_key]
    assert current_seal is not None
    assert current_seal[f"{m40_key}_cid"] == seal[f"{m40_key}_cid"]

    historical, historical_migration, historical_seal = (
        _historical_successor_controls_at(m39_key, scheduler, migration, seal)
    )
    assert m40_key not in historical
    assert historical_migration is not None and m40_key not in historical_migration
    assert historical_seal is not None
    assert f"{m40_key}_cid" not in historical_seal


def test_historical_successor_controls_include_m41_before_m40() -> None:
    """M40 fixtures remove M41 while current M41 remains presence-first."""

    m41_key = (
        "failed_pre_authoritative_m40_validation_successor_materialization"
    )
    m40_key = "failed_pre_authoritative_m39_successor_materialization"
    scheduler = {m41_key: {"revision": "M41"}, m40_key: {"revision": "M40"}}
    migration = copy.deepcopy(scheduler)
    seal = {
        f"{m41_key}_cid": "sha256:" + "5" * 64,
        f"{m40_key}_cid": "sha256:" + "4" * 64,
    }

    current, current_migration, current_seal = _historical_successor_controls_at(
        m41_key, scheduler, migration, seal
    )
    assert current[m41_key] == scheduler[m41_key]
    assert current_migration is not None
    assert current_migration[m41_key] == migration[m41_key]
    assert current_seal is not None
    assert current_seal[f"{m41_key}_cid"] == seal[f"{m41_key}_cid"]

    historical, historical_migration, historical_seal = (
        _historical_successor_controls_at(m40_key, scheduler, migration, seal)
    )
    assert m41_key not in historical
    assert historical_migration is not None and m41_key not in historical_migration
    assert historical_seal is not None
    assert f"{m41_key}_cid" not in historical_seal


def test_historical_successor_controls_include_m42_before_m41() -> None:
    """M41 fixtures remove M42 while current M42 remains presence-first."""

    m42_key = (
        "failed_pre_authoritative_m41_evidence_projection_"
        "successor_materialization"
    )
    m41_key = "failed_pre_authoritative_m40_validation_successor_materialization"
    scheduler = {m42_key: {"revision": "M42"}, m41_key: {"revision": "M41"}}
    migration = copy.deepcopy(scheduler)
    seal = {
        f"{m42_key}_cid": "sha256:" + "6" * 64,
        f"{m41_key}_cid": "sha256:" + "5" * 64,
    }

    current, current_migration, current_seal = _historical_successor_controls_at(
        m42_key, scheduler, migration, seal
    )
    assert current[m42_key] == scheduler[m42_key]
    assert current_migration is not None
    assert current_migration[m42_key] == migration[m42_key]
    assert current_seal is not None
    assert current_seal[f"{m42_key}_cid"] == seal[f"{m42_key}_cid"]

    historical, historical_migration, historical_seal = (
        _historical_successor_controls_at(m41_key, scheduler, migration, seal)
    )
    assert m42_key not in historical
    assert historical_migration is not None and m42_key not in historical_migration
    assert historical_seal is not None
    assert f"{m42_key}_cid" not in historical_seal


def test_historical_successor_controls_include_m43_before_m42() -> None:
    """M42 fixtures remove M43 while current M43 remains presence-first."""

    m43_key = "dead_attempt_lifecycle_recovery_restart_successor_materialization"
    m42_key = (
        "failed_pre_authoritative_m41_evidence_projection_"
        "successor_materialization"
    )
    scheduler = {m43_key: {"revision": "M43"}, m42_key: {"revision": "M42"}}
    migration = copy.deepcopy(scheduler)
    seal = {
        # Deliberately non-current fixture: only relative key removal matters.
        f"{m43_key}_cid": "sha256:SYNTHETIC_UNSEALED_M43_AUTHORITY_CID",
        f"{m42_key}_cid": "sha256:" + "6" * 64,
    }

    current, current_migration, current_seal = _historical_successor_controls_at(
        m43_key, scheduler, migration, seal
    )
    assert current[m43_key] == scheduler[m43_key]
    assert current_migration is not None
    assert current_migration[m43_key] == migration[m43_key]
    assert current_seal is not None
    assert current_seal[f"{m43_key}_cid"] == seal[f"{m43_key}_cid"]

    historical, historical_migration, historical_seal = (
        _historical_successor_controls_at(m42_key, scheduler, migration, seal)
    )
    assert m43_key not in historical
    assert historical_migration is not None and m43_key not in historical_migration
    assert historical_seal is not None
    assert f"{m43_key}_cid" not in historical_seal


def test_historical_successor_controls_include_m44_before_m43() -> None:
    """M43 fixtures remove M44 while current M44 remains presence-first."""

    m44_key = (
        "post_m43_hardened_procfs_user_manager_restart_"
        "successor_materialization"
    )
    m43_key = "dead_attempt_lifecycle_recovery_restart_successor_materialization"
    scheduler = {m44_key: {"revision": "M44"}, m43_key: {"revision": "M43"}}
    migration = copy.deepcopy(scheduler)
    seal = {
        f"{m44_key}_cid": "sha256:SYNTHETIC_UNSEALED_M44_AUTHORITY_CID",
        f"{m43_key}_cid": "sha256:" + "7" * 64,
    }

    current, current_migration, current_seal = _historical_successor_controls_at(
        m44_key, scheduler, migration, seal
    )
    assert current[m44_key] == scheduler[m44_key]
    assert current_migration is not None
    assert current_migration[m44_key] == migration[m44_key]
    assert current_seal is not None
    assert current_seal[f"{m44_key}_cid"] == seal[f"{m44_key}_cid"]

    historical, historical_migration, historical_seal = (
        _historical_successor_controls_at(m43_key, scheduler, migration, seal)
    )
    assert m44_key not in historical
    assert historical_migration is not None and m44_key not in historical_migration
    assert historical_seal is not None
    assert f"{m44_key}_cid" not in historical_seal


def test_historical_successor_controls_include_m45_before_m44() -> None:
    """M44 fixtures remove M45 while current M45 remains presence-first."""

    m45_key = "failed_pre_authoritative_m44_validation_successor_materialization"
    m44_key = (
        "post_m43_hardened_procfs_user_manager_restart_"
        "successor_materialization"
    )
    scheduler = {m45_key: {"revision": "M45"}, m44_key: {"revision": "M44"}}
    migration = copy.deepcopy(scheduler)
    seal = {
        f"{m45_key}_cid": "sha256:SYNTHETIC_UNSEALED_M45_AUTHORITY_CID",
        f"{m44_key}_cid": "sha256:" + "8" * 64,
    }

    current, current_migration, current_seal = _historical_successor_controls_at(
        m45_key, scheduler, migration, seal
    )
    assert current[m45_key] == scheduler[m45_key]
    assert current_migration is not None
    assert current_migration[m45_key] == migration[m45_key]
    assert current_seal is not None
    assert current_seal[f"{m45_key}_cid"] == seal[f"{m45_key}_cid"]

    historical, historical_migration, historical_seal = (
        _historical_successor_controls_at(m44_key, scheduler, migration, seal)
    )
    assert m45_key not in historical
    assert historical_migration is not None and m45_key not in historical_migration
    assert historical_seal is not None
    assert f"{m45_key}_cid" not in historical_seal


def test_historical_successor_controls_include_m46_before_m45() -> None:
    """M45 fixtures remove M46 while current M46 remains presence-first."""

    m46_key = "legacy_no_delta_rescue_recovery_successor_materialization"
    m45_key = "failed_pre_authoritative_m44_validation_successor_materialization"
    scheduler = {m46_key: {"revision": "M46"}, m45_key: {"revision": "M45"}}
    migration = copy.deepcopy(scheduler)
    seal = {
        f"{m46_key}_cid": "sha256:" + "a" * 64,
        f"{m45_key}_cid": "sha256:SYNTHETIC_UNSEALED_M45_AUTHORITY_CID",
    }

    current, current_migration, current_seal = _historical_successor_controls_at(
        m46_key, scheduler, migration, seal
    )
    assert current[m46_key] == scheduler[m46_key]
    assert current_migration is not None
    assert current_migration[m46_key] == migration[m46_key]
    assert current_seal is not None
    assert current_seal[f"{m46_key}_cid"] == seal[f"{m46_key}_cid"]

    historical, historical_migration, historical_seal = (
        _historical_successor_controls_at(m45_key, scheduler, migration, seal)
    )
    assert m46_key not in historical
    assert historical_migration is not None and m46_key not in historical_migration
    assert historical_seal is not None
    assert f"{m46_key}_cid" not in historical_seal


def test_historical_successor_controls_include_m47_before_m46() -> None:
    """M46 fixtures remove M47 while current M47 remains presence-first."""

    m47_key = (
        "ignored_python_cache_preservation_and_recovery_"
        "successor_materialization"
    )
    m46_key = "legacy_no_delta_rescue_recovery_successor_materialization"
    scheduler = {m47_key: {"revision": "M47"}, m46_key: {"revision": "M46"}}
    migration = copy.deepcopy(scheduler)
    seal = {
        f"{m47_key}_cid": "sha256:" + "b" * 64,
        f"{m46_key}_cid": "sha256:" + "a" * 64,
    }

    current, current_migration, current_seal = _historical_successor_controls_at(
        m47_key, scheduler, migration, seal
    )
    assert current[m47_key] == scheduler[m47_key]
    assert current_migration is not None
    assert current_migration[m47_key] == migration[m47_key]
    assert current_seal is not None
    assert current_seal[f"{m47_key}_cid"] == seal[f"{m47_key}_cid"]

    historical, historical_migration, historical_seal = (
        _historical_successor_controls_at(m46_key, scheduler, migration, seal)
    )
    assert m47_key not in historical
    assert historical_migration is not None and m47_key not in historical_migration
    assert historical_seal is not None
    assert f"{m47_key}_cid" not in historical_seal


def test_historical_successor_controls_include_m48_before_m47() -> None:
    """M47 fixtures remove M48 while current M48 remains presence-first."""

    m48_key = "post_m47_clean_shutdown_restart_successor_materialization"
    m47_key = (
        "ignored_python_cache_preservation_and_recovery_"
        "successor_materialization"
    )
    scheduler = {m48_key: {"revision": "M48"}, m47_key: {"revision": "M47"}}
    migration = copy.deepcopy(scheduler)
    seal = {
        f"{m48_key}_cid": "sha256:" + "c" * 64,
        f"{m47_key}_cid": "sha256:" + "b" * 64,
    }

    current, current_migration, current_seal = _historical_successor_controls_at(
        m48_key, scheduler, migration, seal
    )
    assert current[m48_key] == scheduler[m48_key]
    assert current_migration is not None
    assert current_migration[m48_key] == migration[m48_key]
    assert current_seal is not None
    assert current_seal[f"{m48_key}_cid"] == seal[f"{m48_key}_cid"]

    historical, historical_migration, historical_seal = (
        _historical_successor_controls_at(m47_key, scheduler, migration, seal)
    )
    assert m48_key not in historical
    assert historical_migration is not None and m48_key not in historical_migration
    assert historical_seal is not None
    assert f"{m48_key}_cid" not in historical_seal


def _load(relative: str, name: str) -> ModuleType:
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reidentify_extension_projection(pin: dict[str, object]) -> None:
    body = {key: value for key, value in pin.items() if key != "projection_id"}
    pin["projection_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_static_board_gate_is_valid() -> None:
    validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_validator_test",
    )
    report = validator.validate_program(REPO_ROOT)
    assert report["valid"] is True, report["errors"]
    assert report["task_count"] == 45
    assert report["goal_count"] == 29
    assert report["markdown_completion_is_authority"] is False


def test_dependency_gate_qualifies_the_exact_isolated_launch_stack() -> None:
    validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_validator_test",
    )
    report = validator.validate_dependencies(REPO_ROOT, cold_import=True)
    assert report["valid"] is True, report["errors"]
    assert report["database_opened"] is True
    assert report["network_required"] is False

    checks = {item["name"]: item for item in report["checks"]}
    for name in (
        "sealed_launch_toolchain_declaration",
        "accepted_control_plane_mode_closure",
        "exact_immutable_validation_runtime_closure",
        "independent_native_dependency_authorization",
        "quack_httpfs_projection_pins",
        "isolated_launch_toolchain",
        "recomputed_native_dependency_pin",
        "isolated_duckdb_quack_httpfs_load",
        "cold_import_side_effects",
    ):
        assert checks[name]["passed"] is True, checks[name]["detail"]
    mode_closure = checks["accepted_control_plane_mode_closure"]["detail"]
    assert mode_closure["argv_flags"] == ["-I", "-S", "-B"]
    assert mode_closure["file_count"] >= 852
    assert mode_closure["errors"] == []
    assert all(
        int(mode, 8) & 0o022 == 0
        for mode in mode_closure["mode_counts"]
    )
    isolated = checks["isolated_duckdb_quack_httpfs_load"]["detail"]
    assert isolated["python_executable"] == "/usr/bin/python3.12"
    assert isolated["argv_flags"] == ["-I", "-S", "-B"]
    assert isolated["ambient_specs"] == {
        "duckdb": False,
        "_duckdb": False,
        "pytest": False,
    }
    assert isolated["native_module_origin"].startswith("/proc/self/fd/")
    assert [row[0] for row in isolated["extension_rows"]] == ["httpfs", "quack"]
    assert isolated["extension_path_reporting"] == (
        "normalized_after_exact_runtime_path_verification"
    )
    assert all(
        row[4].startswith("$ISOLATED_HOME/.duckdb/extensions/")
        for row in isolated["extension_rows"]
    )
    assert isolated["settings"] == [False, False, False, False]
    assert isolated["select_42"] == 42
    runtime = checks["exact_immutable_validation_runtime_closure"]["detail"]
    assert runtime["python_executable"] == "/usr/bin/python3.12"
    assert runtime["pythonpath_entries"] == [
        "/opt/ipfs-accelerate-aseh-validation-9b3ba6caebcf/site-packages-py-multihash",
        "/opt/ipfs-accelerate-aseh-validation-9b3ba6caebcf/site-packages",
        "/opt/ipfs-accelerate-legal-validation-7ffe92439767/site-packages",
    ]
    assert runtime["payloads"]["delta_deployment"]["manifest_entry_count"] == 4526
    assert runtime["payloads"]["base_deployment"]["manifest_entry_count"] == 28058
    assert runtime["import_probe"]["module_count"] == 39
    dependency = runtime["project_dependency_preflight"]
    assert dependency["valid"] is True
    assert dependency["passed"] is True
    assert dependency["reason"] == (
        "approved_validation_environment_satisfies_project_dependencies"
    )
    assert dependency["path_sensitive"] is True
    assert dependency["validation_command_count"] == 46
    assert dependency["requirements_count"] == 43
    assert dependency["missing_count"] == 0
    assert dependency["incompatible_count"] == 0
    assert dependency["invalid_count"] == 0
    assert dependency["failure"] is None


def test_m29_nested_source_authority_overlay_is_presence_first() -> None:
    validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_validator_m29_nested_overlay_test",
    )
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m29_nested_overlay_test",
    )
    scheduler = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "committed_evidence_verification_successor_materialization"
    scheduler, migration, seal = _historical_successor_controls_at(
        key, scheduler, migration, seal
    )
    assert migration is not None
    assert seal is not None
    expected = (
        materializer._expected_m29_committed_evidence_verification_authority()
    )
    scheduler[key] = copy.deepcopy(expected)
    migration[key] = copy.deepcopy(expected)
    seal[f"{key}_cid"] = materializer._identity(expected)

    effective, errors = validator._effective_nested_source_authorities(
        seal["source_authorities"], scheduler, migration, seal
    )
    assert errors == []
    assert effective["ipfs_datasets_py"]["gitlink_commit"] == (
        "b9f5b86199c03e427fd51fcea302479880421ff8"
    )
    assert effective["ipfs_datasets_py"]["tree"] == (
        "52c0c7be05a51956ba5aa2b6f85d38e03588f3b1"
    )
    assert effective["ipfs_kit_py"]["gitlink_commit"] == (
        "fc9248073e9f67ac59ca607c7736746907b08037"
    )
    assert effective["ipfs_kit_py"]["tree"] == (
        "b26e05db1b199e7e491b45b686a0845fabbabadb"
    )

    partial_seal = copy.deepcopy(seal)
    partial_seal.pop(f"{key}_cid")
    _effective, partial_errors = validator._effective_nested_source_authorities(
        seal["source_authorities"], scheduler, migration, partial_seal
    )
    assert partial_errors == ["active M29 nested-source authority is partial"]

    mismatch = copy.deepcopy(migration)
    mismatch[key]["current_kit_tree"] = "0" * 40
    _effective, mismatch_errors = validator._effective_nested_source_authorities(
        seal["source_authorities"], scheduler, mismatch, seal
    )
    assert mismatch_errors == ["active M29 nested-source authority differs"]

    bad_cid = copy.deepcopy(seal)
    bad_cid[f"{key}_cid"] = "sha256:" + "0" * 64
    _effective, cid_errors = validator._effective_nested_source_authorities(
        seal["source_authorities"], scheduler, migration, bad_cid
    )
    assert cid_errors == ["active M29 nested-source authority CID differs"]

    null_scheduler = copy.deepcopy(scheduler)
    null_scheduler[key] = None
    _effective, null_errors = validator._effective_nested_source_authorities(
        seal["source_authorities"], null_scheduler, migration, seal
    )
    assert null_errors == ["active M29 nested-source authority differs"]

    historical_scheduler, historical_migration, historical_seal = (
        _historical_successor_controls_at(key, scheduler, migration, seal)
    )
    historical_scheduler.pop(key)
    assert historical_migration is not None
    historical_migration.pop(key)
    assert historical_seal is not None
    historical_seal.pop(f"{key}_cid")
    _effective, historical_errors = validator._effective_nested_source_authorities(
        seal["source_authorities"],
        historical_scheduler,
        historical_migration,
        historical_seal,
    )
    assert historical_errors == []


def test_m28_nested_source_authority_overlay_is_presence_first() -> None:
    validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_validator_m18_nested_overlay_test",
    )
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m28_nested_overlay_test",
    )
    scheduler = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    m29_key = "committed_evidence_verification_successor_materialization"
    m28_key = "live_claim_admission_recovery_successor_materialization"
    scheduler, migration, seal = _historical_successor_controls_at(
        m28_key, scheduler, migration, seal
    )
    assert migration is not None
    assert seal is not None
    expected_m28 = (
        materializer._expected_m28_live_claim_admission_recovery_authority()
    )
    scheduler[m28_key] = copy.deepcopy(expected_m28)
    migration[m28_key] = copy.deepcopy(expected_m28)
    seal[f"{m28_key}_cid"] = materializer._identity(expected_m28)
    historical = {
        item["package"]: item
        for item in seal["source_authorities"]
    }
    effective, errors = validator._effective_nested_source_authorities(
        seal["source_authorities"], scheduler, migration, seal
    )
    assert errors == []
    assert historical["ipfs_datasets_py"]["gitlink_commit"] == (
        "1ab21f7a630aa9db1dd5e3257ca900ffd184faf2"
    )
    assert historical["ipfs_datasets_py"]["tree"] == (
        "b45f817a185d508af20d482e449bf85ee2500a37"
    )
    assert effective["ipfs_datasets_py"]["gitlink_commit"] == (
        "b9f5b86199c03e427fd51fcea302479880421ff8"
    )
    assert effective["ipfs_datasets_py"]["tree"] == (
        "52c0c7be05a51956ba5aa2b6f85d38e03588f3b1"
    )
    assert effective["ipfs_kit_py"]["gitlink_commit"] == (
        "fc9248073e9f67ac59ca607c7736746907b08037"
    )
    assert effective["ipfs_kit_py"]["tree"] == (
        "b26e05db1b199e7e491b45b686a0845fabbabadb"
    )

    m27_scheduler = copy.deepcopy(scheduler)
    m27_migration = copy.deepcopy(migration)
    m27_seal = copy.deepcopy(seal)
    partial_m28_seal = copy.deepcopy(seal)
    partial_m28_seal.pop(f"{m28_key}_cid")
    _effective, partial_m28_errors = (
        validator._effective_nested_source_authorities(
            seal["source_authorities"],
            scheduler,
            migration,
            partial_m28_seal,
        )
    )
    assert partial_m28_errors == ["active M28 nested-source authority is partial"]
    mismatched_m28 = copy.deepcopy(migration)
    mismatched_m28[m28_key]["current_kit_tree"] = "0" * 40
    _effective, mismatch_m28_errors = (
        validator._effective_nested_source_authorities(
            seal["source_authorities"], scheduler, mismatched_m28, seal
        )
    )
    assert mismatch_m28_errors == ["active M28 nested-source authority differs"]
    null_m28 = copy.deepcopy(scheduler)
    null_m28[m28_key] = None
    _effective, null_m28_errors = validator._effective_nested_source_authorities(
        seal["source_authorities"], null_m28, migration, seal
    )
    assert null_m28_errors == ["active M28 nested-source authority differs"]

    m27_scheduler.pop(m28_key)
    m27_migration.pop(m28_key)
    m27_seal.pop(f"{m28_key}_cid")
    m27_effective, m27_errors = validator._effective_nested_source_authorities(
        seal["source_authorities"], m27_scheduler, m27_migration, m27_seal
    )
    assert m27_errors == []
    assert m27_effective["ipfs_kit_py"]["gitlink_commit"] == (
        "f30b58d4340ac6670a9da4691b6d08e5eb5948c9"
    )
    assert m27_effective["ipfs_kit_py"]["tree"] == (
        "4d2246774f1ec70cfc529f40d55b5acbf5ea66f6"
    )

    historical_scheduler = copy.deepcopy(scheduler)
    historical_migration = copy.deepcopy(migration)
    historical_seal = copy.deepcopy(seal)
    for successor_key in (
        "live_claim_admission_recovery_successor_materialization",
        "dead_owner_parallel_resume_successor_materialization",
        "automatic_stall_recovery_successor_materialization",
        "native_duckdb_preload_successor_materialization",
        "multi_lane_sidecar_reopen_successor_materialization",
        "multi_lane_successor_materialization",
        "live_preflight_receipt_compatibility_successor_materialization",
        "generation_realization_successor_materialization",
        "test_isolation_successor_materialization",
        "live_catalog_inventory_successor_materialization",
    ):
        historical_scheduler.pop(successor_key)
        historical_migration.pop(successor_key)
        historical_seal.pop(f"{successor_key}_cid")

    key = "portal_completion_persistence_successor_materialization"
    partial_seal = copy.deepcopy(historical_seal)
    partial_seal.pop(f"{key}_cid")
    _effective, partial_errors = validator._effective_nested_source_authorities(
        seal["source_authorities"],
        historical_scheduler,
        historical_migration,
        partial_seal,
    )
    assert partial_errors == ["active M18 nested-source authority is partial"]

    mismatched = copy.deepcopy(historical_migration)
    mismatched[key]["runtime_datasets_tree"] = "0" * 40
    _effective, mismatch_errors = validator._effective_nested_source_authorities(
        seal["source_authorities"],
        historical_scheduler,
        mismatched,
        historical_seal,
    )
    assert mismatch_errors == ["active M18 nested-source authority differs"]

    null_scheduler = copy.deepcopy(historical_scheduler)
    null_scheduler[key] = None
    _effective, null_errors = validator._effective_nested_source_authorities(
        seal["source_authorities"],
        null_scheduler,
        historical_migration,
        historical_seal,
    )
    assert null_errors == ["active M18 nested-source authority differs"]


def test_m6_validation_runtime_tamper_and_absolute_launcher_fail_closed() -> None:
    validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_validation_runtime_tamper_test",
    )
    config = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    seal = json.loads(
        (REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json")
        .read_text(encoding="utf-8")
    )
    exact = validator._validation_runtime_closure(
        REPO_ROOT,
        config,
        seal,
        rehash_payloads=False,
        probe_imports=False,
        probe_dependencies=False,
    )
    assert exact["valid"] is True, exact["errors"]

    tampered = copy.deepcopy(config)
    tampered["validation_runtime"]["delta_deployment"]["artifacts"][0][
        "sha256"
    ] = "sha256:" + ("00" * 32)
    rejected = validator._validation_runtime_closure(
        REPO_ROOT,
        tampered,
        seal,
        rehash_payloads=False,
        probe_imports=False,
        probe_dependencies=False,
    )
    assert rejected["valid"] is False
    assert any("validation_runtime" in error for error in rejected["errors"])

    _historical, operational, command_errors = (
        validator._historical_and_operational_validation_commands(REPO_ROOT)
    )
    assert command_errors == []
    assert len(operational) == 46
    assert all(
        command.startswith("PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. python ")
        for command in operational
    )
    from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
        ValidationRuntimeError,
        validation_shell_command,
    )

    validation_shell_command(operational[0])
    with pytest.raises(
        ValidationRuntimeError,
        match="sealed python or python3 launcher",
    ):
        validation_shell_command(
            operational[0].replace(
                " python ", " /home/barberb/.local/bin/python ", 1
            )
        )


def test_dependency_gate_rejects_projection_identity_drift_from_native() -> None:
    validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_native_projection_binding_test",
    )
    config = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    seal = json.loads(
        (REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json")
        .read_text(encoding="utf-8")
    )

    projections, errors = validator._validate_extension_projection_pins(
        seal,
        config,
    )
    assert errors == []
    assert projections["quack"]["engine_version"] == "v1.5.5"
    assert projections["quack"]["platform"] == "linux_arm64"
    assert projections["httpfs"]["engine_version"] == "v1.5.5"
    assert projections["httpfs"]["platform"] == "linux_arm64"

    for field, value in (
        ("engine_version", "v9.9.9"),
        ("platform", "linux_amd64"),
    ):
        wrong = copy.deepcopy(seal)
        pin = wrong["configured_board_quack_projection"]["pin"]
        pin[field] = value
        _reidentify_extension_projection(pin)
        _, drift_errors = validator._validate_extension_projection_pins(
            wrong,
            config,
        )
        assert (
            "configured-board extension engine/platform differs from "
            "native DuckDB/toolchain"
        ) in drift_errors

    wrong_httpfs = copy.deepcopy(seal)
    wrong_httpfs_config = copy.deepcopy(config)
    httpfs_pin = wrong_httpfs["httpfs_extension_pin"]
    wrong_parent = Path(httpfs_pin["path"]).parent.parent / "linux_amd64"
    httpfs_pin["path"] = str(wrong_parent / "httpfs.duckdb_extension")
    httpfs_pin["info_path"] = str(
        wrong_parent / "httpfs.duckdb_extension.info"
    )
    wrong_httpfs_config["quack_owner"]["pinned_httpfs_extension"] = copy.deepcopy(
        httpfs_pin
    )
    _, httpfs_errors = validator._validate_extension_projection_pins(
        wrong_httpfs,
        wrong_httpfs_config,
    )
    assert (
        "httpfs extension path engine/platform differs from "
        "native DuckDB/toolchain"
    ) in httpfs_errors


def test_rendered_population_has_exact_operator_frontier() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_test",
    )
    population = materializer.build_population(REPO_ROOT)
    tasks = population["taskboard"]
    goals = population["objectives"]
    assert len(tasks) == 45
    assert len(goals) == 29
    assert [task["task_id"] for task in tasks] == [
        f"SAWM-{index:03d}" for index in range(45)
    ]
    assert tasks[0]["status"] == "ready"
    assert all(task["status"] == "todo" for task in tasks[1:])
    assert tasks[1]["depends_on"] == [tasks[0]["task_cid"]]
    assert all(str(goal["goal_cid"]).startswith("sha256:") for goal in goals)
    assert population["plan_revision"] == "SAWM-PLAN-R2"


def test_nonroot_goal_objective_projection_is_optional_and_closed() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_optional_goal_objective_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_optional_goal_objective_test",
    )
    goals = materializer.build_population(REPO_ROOT)["objectives"]
    root = next(goal for goal in goals if goal["goal_id"] == "SAWM-G000")
    children = [goal for goal in goals if goal["goal_id"] != "SAWM-G000"]

    assert root["objective_id"]
    assert children
    assert all("objective_id" not in goal for goal in children)
    assert all(str(goal.get("objective_id") or "") == "" for goal in children)

    source = Path(operator.__file__).read_text(encoding="utf-8")
    assert 'str(expected.get("objective_id") or "")' in source
    assert 'expected["objective_id"]' not in source


def test_managed_daemon_uses_the_supervisors_explicit_board_lock(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        board_scoped_checkout_mutation_lock_path,
        board_scoped_protected_path_maintenance_lock_path,
        checkout_mutation_lock_path,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        parse_args as parse_daemon_args,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    namespace = "semantic-addressed-world-model-v1"
    daemon = object.__new__(PortalImplementationDaemon)
    daemon.repo_root = tmp_path
    daemon.board_namespace = namespace
    supervisor = object.__new__(PortalImplementationSupervisor)
    supervisor.board_namespace = namespace
    supervisor.config = SimpleNamespace(repo_root=tmp_path)

    expected = board_scoped_checkout_mutation_lock_path(tmp_path, namespace)
    assert daemon._repo_merge_lock_path() == expected
    assert supervisor._repo_merge_lock_path() == expected
    maintenance = daemon._protected_path_maintenance_lock_path()
    assert maintenance == (
        board_scoped_protected_path_maintenance_lock_path(tmp_path, namespace)
    )
    assert expected != checkout_mutation_lock_path(tmp_path)
    assert maintenance != checkout_mutation_lock_path(tmp_path)
    assert expected != board_scoped_checkout_mutation_lock_path(
        tmp_path,
        "independent-board-v1",
    )
    assert parse_daemon_args(
        ["--board-namespace", namespace, "--once"]
    ).board_namespace == namespace

    daemon.board_namespace = ""
    assert daemon._repo_merge_lock_path() == checkout_mutation_lock_path(
        tmp_path
    )


def test_preserved_definition_cids_rehash_from_the_prior_source_binding() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_identity_test",
    )
    population = materializer.build_population(REPO_ROOT)
    prior = population["migration_inventory"]

    for record in (*population["objectives"], *population["taskboard"]):
        assert record["definition"]["source_binding_cid"] == prior[
            "definition_source_binding_cid"
        ]
        assert materializer._identity(record["definition"]) == record["definition_cid"]
    assert materializer._identity(population["program_definition"]) == population[
        "program_definition_cid"
    ]


def test_materialization_rejects_a_dirty_or_uncommitted_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_dirty_test",
    )
    population = materializer.build_population(REPO_ROOT)
    real_git = materializer._git

    def dirty_git(root: Path, *args: str, **kwargs: object) -> str:
        if args[:2] == ("status", "--porcelain=v1"):
            return " M scripts/materialize_semantic_addressed_world_model_program.py"
        return real_git(root, *args, **kwargs)

    monkeypatch.setattr(materializer, "_git", dirty_git)
    with pytest.raises(materializer.MaterializationError, match="clean committed"):
        materializer._assert_committed_clean_source(REPO_ROOT, population)

    monkeypatch.setattr(materializer, "_git", real_git)
    stale_population = copy.deepcopy(population)
    stale_population["source_binding"]["head"] = "0" * 40
    with pytest.raises(
        materializer.MaterializationError,
        match="source binding changed",
    ):
        materializer._assert_source_binding_matches(
            REPO_ROOT,
            stale_population,
        )


def test_materialization_refuses_preexisting_successor_sidecars(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_fresh_sidecars_test",
    )
    target = tmp_path / "control.duckdb"
    (tmp_path / "control.execution.duckdb").touch(mode=0o600)
    with pytest.raises(
        materializer.MaterializationError,
        match="execution/coordination authority is not fresh",
    ):
        materializer._assert_fresh_successor_operational_state(target)


def test_append_only_source_migration_rehearsal_verifies_exactly() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_migration_test",
    )
    population = materializer.build_population(REPO_ROOT)
    migration = population["migration_inventory"]
    config = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    prior = materializer._verify_prior_store(REPO_ROOT, config, population)
    dependency = materializer._validator_report(
        REPO_ROOT, "scripts/validate_semantic_addressed_world_model_dependencies.py"
    )
    board = materializer._validator_report(
        REPO_ROOT, "scripts/validate_semantic_addressed_world_model_board.py"
    )
    validation_digest = materializer._identity(
        {
            "dependency": dependency,
            "board": board,
            "program_definition_cid": population["program_definition_cid"],
        }
    )

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    with tempfile.TemporaryDirectory(prefix="sawm-r2-test-", dir="/tmp") as directory:
        stage = Path(directory) / "control.duckdb"
        shutil.copy2(Path(prior["database_path"]), stage)
        source = DatabaseTaskSource(
            stage,
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
            owner_id="sawm-r2-source-migrator",
        )
        try:
            operator = source.get_task("SAWM-000")
            assert operator is not None
            migration_body = materializer._migration_body(
                population, config, validation_digest
            )
            assert migration_body["preworker_launch_failure"] == migration[
                "preworker_launch_failure"
            ]
            assert migration_body["preprovider_task_failure"] == migration[
                "preprovider_task_failure"
            ]
            recovery_receipt = materializer._task_recovery_receipt(population)
            assert migration_body["nonterminal_task_recovery_receipt"] == (
                recovery_receipt
            )
            migration_digest = materializer._identity(migration_body)
            source.plans.append_revision(
                plan_cid=str(population["plan_root_cid"]),
                expected_revision=int(migration["prior_plan_revision"]),
                body={
                    "current_source_binding_cid": population["source_binding"][
                        "source_binding_cid"
                    ],
                    "source_migration_revision": migration["migration_revision"],
                    "source_migration_digest": migration_digest,
                    "supersession_mode": materializer._M6_SUPERSESSION_MODE,
                },
                delta=materializer._migration_plan_delta(population),
            )
            source.record_evidence(
                task_cid=operator.task_cid,
                evidence_kind="operator_control_plane_source_migration",
                digest=migration_digest,
                body=migration_body,
            )
            operational_event_ids = {}
            for alias in materializer._operational_task_aliases():
                prior_task = source.intent.get_task(alias)
                assert prior_task is not None
                expected_status = "blocked" if alias == "SAWM-001" else "todo"
                expected_revision = 5 if alias == "SAWM-001" else 1
                assert (prior_task["status"], prior_task["revision"]) == (
                    expected_status,
                    expected_revision,
                )
                prior_validations = tuple(
                    tuple(str(part) for part in item.get("argv") or ())
                    for item in prior_task["validations"]
                )
                operational_validations, replacement_count = (
                    materializer._operational_validation_commands(
                        prior_validations,
                        task_alias=alias,
                    )
                )
                operational_receipt = materializer._operational_validation_receipt(
                    population,
                    task_alias=alias,
                    task_cid=prior_task["task_cid"],
                    expected_status=expected_status,
                    expected_revision=expected_revision,
                    prior_validations=prior_validations,
                    operational_validations=operational_validations,
                    replacement_count=replacement_count,
                )
                upsert = source.intent.upsert_task(
                    task_cid=prior_task["task_cid"],
                    task_alias=prior_task["task_alias"],
                    goal_cid=prior_task["goal_cid"],
                    ordinal=prior_task["ordinal"],
                    status=prior_task["status"],
                    priority=prior_task["priority"],
                    plan_cid=prior_task["plan_cid"],
                    objective_id=prior_task["objective_id"],
                    body={
                        **dict(prior_task["body"]),
                        "operational_validation_revision": operational_receipt,
                    },
                    identity=dict(prior_task["identity"]),
                    expected_revision=expected_revision,
                    dependencies=list(prior_task["dependencies"]),
                    outputs=[
                        dict(item["effect"]) for item in prior_task["outputs"]
                    ],
                    acceptance=[
                        dict(item["evidence_policy"])
                        for item in prior_task["acceptance"]
                    ],
                    validations=[
                        {**dict(prior["policy"]), "argv": list(argv)}
                        for prior, argv in zip(
                            prior_task["validations"],
                            operational_validations,
                            strict=True,
                        )
                    ],
                )
                operational_event_ids[alias] = upsert.event_id
            candidate = source.get_task("SAWM-001")
            assert candidate is not None
            assert (candidate.status, candidate.revision) == ("blocked", 6)
            recovery = source.compare_and_set_status(
                candidate.task_cid,
                expected_revision=6,
                status="todo",
                receipt=recovery_receipt,
            )
            assert recovery.changed is True
            assert recovery.previous_status == "blocked"
            assert (recovery.task.status, recovery.task.revision) == ("todo", 7)
        finally:
            source.close()

        verified = materializer._verify_store(
            stage,
            population,
            require_operator_complete=True,
            require_migration=True,
            migration_config=config,
            expected_validation_digest=validation_digest,
        )
        assert verified["event_watermark"] == migration["prior_event_watermark"] + 47
        assert verified["migration_event_watermark"] == 123
        assert verified["operational_validation_first_event_watermark"] == 124
        assert verified["operational_validation_last_event_watermark"] == 167
        assert verified["target_event_watermark"] == 168
        assert verified["projection_cid"] == materializer._M6_EXPECTED_PROJECTION_CID
        assert verified["statuses"]["SAWM-001"] == "todo"
        assert verified["revisions"]["SAWM-001"] == 7
        assert all(verified["revisions"][f"SAWM-{index:03d}"] == 2 for index in range(2, 45))
        assert verified["accepted_definition_changes"] == 0
        assert verified["accepted_completion_changes"] == 0
        assert verified["nonterminal_task_status_recovery_changes"] == 1
        assert verified["task_recovery_receipt"] == recovery_receipt
        assert verified["projection_matches_events"] is True
        assert materializer._store_sha256(Path(prior["database_path"])) == migration[
            "prior_control_store_sha256"
        ]
        stale_lock = stage.parent / ".migration-receipt.json.publish.lock"
        stale_lock.touch(mode=0o600)
        receipt = materializer._ensure_migration_receipt(
            stage.parent, stage, population, verified, validation_digest
        )
        receipt_path = stage.parent / "migration-receipt.json"
        assert receipt_path.is_file()
        assert receipt["schema"] == "sawm/non-authoritative-migration-receipt@3"
        assert receipt["migration_event_watermark"] == 123
        assert receipt["operational_validation_first_event_watermark"] == 124
        assert receipt["operational_validation_last_event_watermark"] == 167
        assert receipt["target_event_watermark"] == 168
        assert receipt["migration_projection_cid"] != receipt["projection_cid"]
        assert receipt["projection_cid"] == materializer._M6_EXPECTED_PROJECTION_CID
        assert receipt["task_recovery_event_id"] == verified[
            "task_recovery_event_id"
        ]
        assert receipt["task_recovery_receipt"] == recovery_receipt
        assert receipt["nonterminal_task_status_recovery_changes"] == 1
        assert not (stage.parent / "control.execution.duckdb").exists()
        assert not (stage.parent / "control.coordination.duckdb").exists()
        assert materializer._ensure_migration_receipt(
            stage.parent, stage, population, verified, validation_digest
        ) == receipt
        receipt_path.unlink()
        assert materializer._ensure_migration_receipt(
            stage.parent, stage, population, verified, validation_digest
        ) == receipt


def test_m7_source_only_migration_rehearsal_and_tamper_gates() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m7_rehearsal_test",
    )
    population = materializer.build_population(REPO_ROOT)
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    validation_digest = materializer._m7_validation_digest(
        REPO_ROOT,
        population,
    )
    authority = config["source_repair_materialization"]
    prior_path = REPO_ROOT / authority["prior_store_id"]
    prior_hash = materializer._store_sha256(prior_path)
    prior_append_surface_digest = materializer._append_surface_digest(prior_path)
    assert prior_append_surface_digest == authority["prior_append_surface_digest"]

    class AdapterRow:
        def __init__(self, columns: tuple[str, ...], values: tuple[object, ...]):
            self.columns = columns
            self.values = values

        def __getitem__(self, index: int) -> object:
            return self.values[index]

        def __iter__(self):
            return iter(self.columns)

    adapter_row = AdapterRow(("left", "right"), ("value-left", "value-right"))
    assert tuple(adapter_row) == ("left", "right")
    assert adapter_row != ("value-left", "value-right")
    assert materializer._positional_rows([adapter_row], 2) == [
        ("value-left", "value-right")
    ]

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    with tempfile.TemporaryDirectory(
        prefix="sawm-r2-m7-test-",
        dir="/tmp",
    ) as directory:
        stage = Path(directory) / "control.duckdb"
        shutil.copy2(prior_path, stage)
        before_semantic = materializer._semantic_authority_digest(stage)
        before_frozen = materializer._frozen_base_authority_digest(stage)
        source = DatabaseTaskSource(
            stage,
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
            owner_id="sawm-r2-source-migrator",
        )
        try:
            operator = source.get_task("SAWM-000")
            assert operator is not None
            body = materializer._m7_migration_body(
                population,
                config,
                validation_digest,
            )
            failure = materializer._M7_PREPUBLICATION_MATERIALIZATION_FAILURE
            assert body["schema"] == (
                "sawm/operator-control-plane-source-migration@5"
            )
            assert body["prepublication_materialization_failure"] == failure
            assert body["prepublication_materialization_failure_cid"] == (
                materializer._identity(failure)
            )
            assert failure["target_published"] is False
            assert failure["implementation_provider_invoked"] is False
            digest = materializer._identity(body)
            delta = materializer._m7_migration_plan_delta(population, config)
            source.plans.append_revision(
                plan_cid=str(population["plan_root_cid"]),
                expected_revision=7,
                body={
                    "current_source_binding_cid": population["source_binding"][
                        "source_binding_cid"
                    ],
                    "source_migration_revision": materializer._M7_MIGRATION_REVISION,
                    "source_migration_digest": digest,
                    "supersession_mode": materializer._M7_SUPERSESSION_MODE,
                },
                delta=delta,
            )
            source.record_evidence(
                task_cid=operator.task_cid,
                evidence_kind="operator_control_plane_source_migration",
                digest=digest,
                body=body,
            )
        finally:
            source.close()

        verified = materializer._verify_m7_store(
            stage,
            population,
            config,
            validation_digest,
        )
        assert verified["event_watermark"] == 170
        assert verified["projection_cid"] == (
            materializer._M7_EXPECTED_PROJECTION_CID
        )
        assert verified["task_revision_changes"] == 0
        assert verified["task_status_changes"] == 0
        assert verified["accepted_definition_changes"] == 0
        assert verified["accepted_completion_changes"] == 0
        assert verified["projection_matches_events"] is True
        assert materializer._semantic_authority_digest(stage) == before_semantic
        assert materializer._frozen_base_authority_digest(stage) == before_frozen
        assert before_semantic == authority["prior_semantic_authority_digest"]
        assert before_frozen == authority["prior_frozen_base_authority_digest"]
        assert materializer._store_sha256(prior_path) == prior_hash

        import duckdb

        forged_evidence = Path(directory) / "forged-evidence.duckdb"
        shutil.copy2(stage, forged_evidence)
        connection = duckdb.connect(str(forged_evidence))
        try:
            connection.execute(
                "INSERT INTO evidence_nodes "
                "SELECT ?, parent_evidence_id, task_cid, evidence_kind, digest, "
                "created_at, body_json FROM evidence_nodes LIMIT 1",
                ["sha256:" + "1" * 64],
            )
        finally:
            connection.close()
        with pytest.raises(
            materializer.MigrationRequired,
            match="append-surface",
        ):
            materializer._verify_m7_store(
                forged_evidence,
                population,
                config,
                validation_digest,
            )

        forged_plan = Path(directory) / "forged-plan.duckdb"
        shutil.copy2(stage, forged_plan)
        connection = duckdb.connect(str(forged_plan))
        try:
            connection.execute(
                "UPDATE plans SET body_json = ?",
                ['{"forged":true}'],
            )
        finally:
            connection.close()
        with pytest.raises(
            materializer.MigrationRequired,
            match="plan body",
        ):
            materializer._verify_m7_store(
                forged_plan,
                population,
                config,
                validation_digest,
            )

        forged_state = Path(directory) / "forged-state.duckdb"
        shutil.copy2(stage, forged_state)
        connection = duckdb.connect(str(forged_state))
        try:
            connection.execute(
                "UPDATE state_servers SET status = 'running' WHERE generation = 8"
            )
        finally:
            connection.close()
        with pytest.raises(
            materializer.MigrationRequired,
            match="frozen base authority",
        ):
            materializer._verify_m7_store(
                forged_state,
                population,
                config,
                validation_digest,
            )


def test_m7_controls_remain_immutable_historical_authority() -> None:
    dependency_validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m7_source_repair_test",
    )
    board_validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m7_source_repair_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    assert dependency_validator._m7_source_repair_errors(
        config,
        seal,
        migration,
    ) == []
    assert board_validator._m7_migration_errors(config, seal, migration) == []

    changed_authority = copy.deepcopy(config)
    changed_authority["source_repair_materialization"][
        "accepted_completion_changes"
    ] = 1
    assert dependency_validator._m7_source_repair_errors(
        changed_authority,
        seal,
        migration,
    )
    drifted_seal = copy.deepcopy(seal)
    drifted_seal["source_repair_materialization_cid"] = "sha256:" + "0" * 64
    assert dependency_validator._m7_source_repair_errors(
        config,
        drifted_seal,
        migration,
    )

    # Active M8 runtime movement cannot invalidate the frozen M7 authority.
    stale_runtime = copy.deepcopy(config)
    stale_runtime["runtime_paths"]["root"] = (
        "data/agent_supervisor/semantic_addressed_world_model/run-r2-m6"
    )
    assert dependency_validator._m7_source_repair_errors(
        stale_runtime,
        seal,
        migration,
    ) == []


def test_m8_controls_and_live_comparator_fail_closed() -> None:
    dependency_validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m8_source_repair_test",
    )
    board_validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m8_source_repair_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m8_live_comparator_test",
    )
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m8_dispatch_fail_closed_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    expected_cid = (
        "sha256:01935058ca682411743904513524b3eab41369b057151db529af14d46c5c0963"
    )
    authority = config["source_repair_successor_materialization"]
    assert authority == migration["source_repair_successor_materialization"]
    assert "sha256:" + hashlib.sha256(
        json.dumps(
            authority,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest() == expected_cid
    assert seal["source_repair_successor_materialization_cid"] == expected_cid
    assert dependency_validator._m8_source_repair_errors(
        config,
        seal,
        migration,
    ) == []
    assert board_validator._m8_migration_errors(config, seal, migration) == []

    # Exercise the historical M8 selector in isolation. Newer keys intentionally
    # have precedence, including fail-closed malformed handling.
    malformed_successor, _historical_migration, _historical_seal = (
        _historical_successor_controls_at(
            "source_repair_successor_materialization",
            config,
        )
    )
    malformed_successor["source_repair_successor_materialization"] = []
    with pytest.raises(
        operator.OperatorError,
        match="active source-only successor authority is invalid",
    ):
        operator._active_source_repair_materialization(malformed_successor)
    with pytest.raises(
        materializer.MaterializationError,
        match="M8 source-only successor authority is invalid",
    ):
        materializer._m8_successor_configured(malformed_successor)

    changed_authority = copy.deepcopy(config)
    changed_authority["source_repair_successor_materialization"][
        "accepted_completion_changes"
    ] = 1
    assert dependency_validator._m8_source_repair_errors(
        changed_authority,
        seal,
        migration,
    )

    changed_failure_config = copy.deepcopy(config)
    changed_failure_inventory = copy.deepcopy(migration)
    for controls in (changed_failure_config, changed_failure_inventory):
        controls["source_repair_successor_materialization"][
            "live_preflight_failure"
        ]["implementation_provider_invoked"] = True
    assert dependency_validator._m8_source_repair_errors(
        changed_failure_config,
        seal,
        changed_failure_inventory,
    )

    stale_runtime = copy.deepcopy(config)
    stale_runtime.pop("live_recovery_successor_materialization")
    stale_runtime["runtime_paths"]["root"] = (
        "data/agent_supervisor/semantic_addressed_world_model/run-r2-m7"
    )
    assert dependency_validator._m8_source_repair_errors(
        stale_runtime,
        seal,
        migration,
    )

    drifted_seal = copy.deepcopy(seal)
    drifted_seal["source_repair_successor_materialization_cid"] = (
        "sha256:" + "0" * 64
    )
    assert dependency_validator._m8_source_repair_errors(
        config,
        drifted_seal,
        migration,
    )

    source = Path(operator.__file__).read_text(encoding="utf-8")
    assert "materializer._verify_m6_task_projection(live, population)" in source
    assert "materializer._semantic_authority_digest_on(connection)" in source
    assert (
        'successor_key = "source_repair_successor_materialization"' in source
    )
    assert (
        'expected_event_cursor = int(preflight_contract["target_event_watermark"])'
        in source
    )
    assert 'str(expected.get("objective_id") or "")' in source
    assert 'expected["objective_id"]' not in source
    assert "checked = materializer.check_materialized(REPO_ROOT, config_path)" in source


def test_m8_materializer_control_rehearsal_and_tamper_gate() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m8_control_rehearsal_test",
    )
    population = materializer.build_population(REPO_ROOT)
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    authority = materializer._m8_source_repair_authority(population, config)
    validation_digest = materializer._m7_validation_digest(
        REPO_ROOT,
        population,
    )
    body = materializer._m8_migration_body(
        population,
        config,
        validation_digest,
    )
    delta = materializer._m8_migration_plan_delta(population, config)

    assert materializer._identity(authority) == (
        "sha256:01935058ca682411743904513524b3eab41369b057151db529af14d46c5c0963"
    )
    assert body["schema"] == "sawm/operator-control-plane-source-migration@6"
    assert body["migration_revision"] == "SAWM-R2-M8"
    assert body["supersession_reason"] == (
        "optional_nonroot_goal_objective_projection_repair"
    )
    assert body["live_preflight_failure"] == authority["live_preflight_failure"]
    assert body["accepted_goal_definitions_rewritten"] is False
    assert body["accepted_completion_changes"] == 0
    assert body["implementation_provider_invocations"] == 0
    assert body["worker_self_approval"] is False
    assert delta["kind"] == "optional_nonroot_goal_objective_projection_repair"
    assert delta["accepted_definition_changes"] == 0
    assert delta["accepted_completion_changes"] == 0
    assert delta["worker_self_approval"] is False

    prior_path = REPO_ROOT / authority["prior_store_id"]
    prior_hash = materializer._store_sha256(prior_path)
    materializer._assert_m8_prior_publication_anchor(
        REPO_ROOT,
        authority,
        prior_path.resolve(),
    )
    with tempfile.TemporaryDirectory(
        prefix="sawm-r2-m8-test-",
        dir="/tmp",
    ) as directory:
        stage = Path(directory) / "control.duckdb"
        shutil.copy2(prior_path, stage)
        before_semantic = materializer._semantic_authority_digest(stage)
        before_frozen = materializer._frozen_base_authority_digest(stage)
        before_append = materializer._append_surface_digest(stage)

        from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
            DatabaseTaskSource,
        )

        source = DatabaseTaskSource(
            stage,
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
            owner_id="sawm-r2-source-migrator",
        )
        try:
            operator_task = source.get_task("SAWM-000")
            assert operator_task is not None
            source.plans.append_revision(
                plan_cid=str(population["plan_root_cid"]),
                expected_revision=8,
                body={
                    "current_source_binding_cid": population["source_binding"][
                        "source_binding_cid"
                    ],
                    "source_migration_revision": materializer._M8_MIGRATION_REVISION,
                    "source_migration_digest": materializer._identity(body),
                    "supersession_mode": materializer._M8_SUPERSESSION_MODE,
                },
                delta=delta,
            )
            source.record_evidence(
                task_cid=operator_task.task_cid,
                evidence_kind="operator_control_plane_source_migration",
                digest=materializer._identity(body),
                body=body,
            )
        finally:
            source.close()

        verified = materializer._verify_m8_store(
            stage,
            population,
            config,
            validation_digest,
        )
        assert verified["event_watermark"] == 172
        assert verified["projection_cid"] == materializer._M8_EXPECTED_PROJECTION_CID
        assert verified["projection_matches_events"] is True
        assert verified["task_revision_changes"] == 0
        assert verified["task_status_changes"] == 0
        assert verified["accepted_definition_changes"] == 0
        assert verified["accepted_completion_changes"] == 0
        assert materializer._semantic_authority_digest(stage) == before_semantic
        assert materializer._frozen_base_authority_digest(stage) == before_frozen
        assert before_semantic == authority["prior_semantic_authority_digest"]
        assert before_frozen == authority["prior_frozen_base_authority_digest"]
        assert before_append == authority["prior_append_surface_digest"]
        assert materializer._store_sha256(prior_path) == prior_hash

    changed_config = copy.deepcopy(config)
    changed_config["source_repair_successor_materialization"][
        "accepted_completion_changes"
    ] = 1
    with pytest.raises(
        materializer.MaterializationError,
        match="differs across scheduler and inventory",
    ):
        materializer._m8_source_repair_authority(population, changed_config)


def test_scheduler_keeps_ducklake_non_authoritative() -> None:
    config = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    assert config["max_lanes"] == 4
    assert config["database_program"]["authority_mode"] == "quack"
    assert config["database_program"]["failover_policy"] == "fail_closed"
    ducklake = config["ducklake_history_projection"]
    assert ducklake["load_local_only"] is True
    assert ducklake["install_or_network_forbidden"] is True
    assert ducklake["authority"] is False
    assert ducklake["scheduling_prerequisite"] is False
    assert ducklake["completion_prerequisite"] is False


def test_m8_nofollow_receipts_and_ducklake_retry_are_idempotent(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m8_advisory_retry_test",
    )
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "receipt.json"
    link.symlink_to(target.name)
    with pytest.raises(
        materializer.MigrationRequired,
        match="no-follow regular file",
    ):
        materializer._load_nofollow_json(
            link,
            root=tmp_path,
            noun="test receipt",
        )

    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    config["ducklake_history_projection"].update(
        {
            "catalog_path": "ducklake/history.ducklake",
            "data_path": "ducklake/data",
            "receipt_path": "ducklake-history-receipt.json",
        }
    )
    record = {
        "program_definition_cid": "sha256:" + "1" * 64,
        "projection_cid": "baguqeera" + "a" * 52,
    }
    first = materializer._ducklake_projection(tmp_path, config, record)
    second = materializer._ducklake_projection(tmp_path, config, record)
    assert second == first
    assert first["authority"] is False
    assert first["scheduling_prerequisite"] is False
    assert first["completion_prerequisite"] is False
    unhashed = dict(first)
    claimed = unhashed.pop("receipt_cid")
    assert claimed == materializer._identity(unhashed)

    receipt_path = tmp_path / "ducklake-history-receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        materializer.MigrationRequired,
        match="DuckLake history receipt differs",
    ):
        materializer._ducklake_projection(tmp_path, config, record)

    if first["status"] == "available":
        import duckdb

        connection = duckdb.connect(":memory:")
        try:
            connection.execute("SET autoinstall_known_extensions = false")
            connection.execute("SET autoload_known_extensions = false")
            connection.execute("LOAD ducklake")
            catalog = tmp_path / "ducklake/history.ducklake"
            data = tmp_path / "ducklake/data"

            def literal(value: Path) -> str:
                return "'" + str(value).replace("'", "''") + "'"

            connection.execute(
                "ATTACH "
                + literal(Path("ducklake:" + str(catalog)))
                + " AS sawm_history_check (DATA_PATH "
                + literal(data)
                + ")"
            )
            count = connection.execute(
                "SELECT COUNT(*) FROM sawm_history_check.control_history "
                "WHERE program_definition_cid = ? AND projection_cid = ?",
                [record["program_definition_cid"], record["projection_cid"]],
            ).fetchone()[0]
            assert int(count) == 1
        finally:
            connection.close()


def test_m6_migration_preserves_m1_through_m5_and_preprovider_evidence() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_chain_test",
    )
    population = materializer.build_population(REPO_ROOT)
    migration = population["migration_inventory"]
    history = migration["migration_history"]

    assert migration["migration_revision"] == "SAWM-R2-M6"
    assert migration["migration_kind"] == (
        "bounded_validation_runtime_and_operational_board_command_recovery"
    )
    assert migration["supersession_reason"] == (
        "source_authority_revision_and_settled_preprovider_task_requeue"
    )
    assert migration["prior_plan_revision"] == 6
    assert migration["target_plan_revision"] == 7
    assert migration["prior_event_watermark"] == 121
    assert len(history) == 5
    m1 = history[0]
    m2 = history[1]
    m3 = history[2]
    m4 = history[3]
    m5 = history[4]
    assert m1["migration_revision"] == "SAWM-R2-M1"
    assert m2["migration_revision"] == "SAWM-R2-M2"
    assert m3["migration_revision"] == "SAWM-R2-M3"
    assert m4["migration_revision"] == "SAWM-R2-M4"
    assert m5["migration_revision"] == "SAWM-R2-M5"
    assert all(entry["schema"] == "sawm/source-migration-history-entry@1" for entry in history[:3])
    assert m4["schema"] == "sawm/source-migration-history-entry@2"
    assert m5["schema"] == "sawm/source-migration-history-entry@3"
    assert materializer._store_sha256(REPO_ROOT / m1["prior_store_id"]) == m1[
        "prior_control_store_sha256"
    ]
    assert materializer._store_sha256(REPO_ROOT / m1["target_store_id"]) == m1[
        "target_control_store_sha256"
    ]
    assert materializer._store_sha256(REPO_ROOT / m2["target_store_id"]) == m2[
        "target_control_store_sha256"
    ]
    assert materializer._store_sha256(REPO_ROOT / m3["target_store_id"]) == m3[
        "target_control_store_sha256"
    ]
    assert materializer._store_sha256(REPO_ROOT / m4["target_store_id"]) == m4[
        "target_control_store_sha256"
    ]
    assert materializer._store_sha256(REPO_ROOT / m5["target_store_id"]) == m5[
        "target_control_store_sha256"
    ]
    assert m5["target_control_store_sha256"] == migration[
        "prior_control_store_sha256"
    ]
    assert m5["target_event_prefix_sha256"] == migration[
        "prior_event_prefix_sha256"
    ]
    assert m5["prior_event_watermark"] == 116
    assert m5["migration_event_watermark"] == 118
    assert m5["materialization_event_watermark"] == 119
    assert m5["target_event_watermark"] == 121
    assert m5["materialization_projection_cid"] == migration[
        "prior_materialization_projection_cid"
    ]
    assert m5["projection_cid"] == migration["prior_projection_cid"]
    assert m5["migration_receipt_cid"] == migration[
        "prior_materialization_receipt_cid"
    ]
    for entry in history:
        materializer._verify_receipt_anchor(
            REPO_ROOT / entry["migration_receipt_path"],
            entry["migration_receipt_cid"],
        )
    failure = migration["preworker_launch_failure"]
    assert set(failure) == {
        "accepted_control_plane_admitted",
        "command",
        "configuration_root",
        "control_plane_admission_cid",
        "control_plane_archive_sha256",
        "control_plane_capsule_id",
        "coordinator_log_path",
        "coordinator_log_sha256",
        "coordinator_pid",
        "coordinator_pid_projection_after_failure",
        "credential_handoff_retired",
        "error_payload",
        "error_payload_cid",
        "exit_code",
        "failure_time_authority",
        "implementation_provider_invoked",
        "outer_launch_command",
        "outer_launch_exit_code",
        "owner_generation",
        "owner_process_birth_id",
        "owner_server_id",
        "phase",
        "provider_capability_probed",
        "schema",
        "source_head",
        "source_tree",
        "store_id",
        "task_claimed",
        "task_state_changed",
        "worker_started",
    }
    assert failure["schema"] == "sawm/pre-worker-launch-failure@2"
    assert failure["phase"] == "detached_coordinator_provider_entry_module_preflight"
    assert failure["outer_launch_exit_code"] == 0
    assert failure["exit_code"] == 2
    assert failure["accepted_control_plane_admitted"] is True
    assert failure["provider_capability_probed"] is True
    assert failure["coordinator_pid_projection_after_failure"] == "absent"
    assert failure["worker_started"] is False
    assert failure["task_claimed"] is False
    assert failure["task_state_changed"] is False
    assert failure["implementation_provider_invoked"] is False
    assert failure["credential_handoff_retired"] is True
    assert failure["failure_time_authority"] == "unavailable"
    preprovider = migration["preprovider_task_failure"]
    assert preprovider["schema"] == "sawm/pre-provider-task-failure@2"
    assert preprovider["task_alias"] == "SAWM-001"
    assert preprovider["task_status"] == "in_progress"
    assert preprovider["canonical_task_status"] == "blocked"
    assert preprovider["canonical_task_revision"] == 5
    assert preprovider["canonical_event_watermark"] == 121
    assert preprovider["settlement_closed"] is True
    assert preprovider["canonical_projection_cid"] == migration[
        "prior_projection_cid"
    ]
    assert preprovider["lifecycle_started"] is True
    assert preprovider["worktree_setup_occurred"] is True
    assert preprovider["provider_dispatched"] is False
    assert preprovider["implementation_provider_invoked"] is False
    assert preprovider["effect_claim_recorded"] is False
    assert preprovider["implementation_commit_created"] is False
    assert preprovider["merge_attempted"] is False
    assert preprovider["task_completed"] is False
    assert preprovider["owner_status"] == "stopped"
    verified_failure = materializer._verify_preprovider_failure_artifacts(
        REPO_ROOT, migration
    )
    assert verified_failure["provider_invocation_count"] == 0
    assert verified_failure["effect_claim_count"] == 0


def test_m6_operational_revisions_and_requeue_are_fail_closed() -> None:
    validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m6_recovery_paths_test",
    )
    dependency_validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m6_recovery_paths_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    recovery_controls = {
        "ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py",
        "test/api/test_agent_supervisor_database_coordination.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
        "test/api/test_agent_supervisor_database_portal_bridge.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
        "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
        "test/api/test_agent_supervisor_project_dependency_preflight.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/provider_command_binding.py",
        "test/api/test_agent_supervisor_provider_command_binding.py",
    }

    assert recovery_controls.issubset(validator.CONTROL_RELATIVE_PATHS)
    assert recovery_controls.issubset(dependency_validator.CONTROL_PATHS)
    assert tuple(config["protected_paths"]) == validator.CONTROL_RELATIVE_PATHS
    assert recovery_controls.issubset(
        config["configured_board_live_capsule"]["control_paths"]
    )
    assert recovery_controls.issubset(migration["bounded_control_plane_repair_paths"])
    assert validator._m6_migration_errors(config, seal, migration) == []
    assert dependency_validator._m6_source_migration_errors(
        config, seal, migration
    ) == []

    collapsed_m5_watermarks = copy.deepcopy(migration)
    collapsed_m5_watermarks["migration_history"][4][
        "materialization_event_watermark"
    ] = 121
    assert validator._m6_migration_errors(
        config, seal, collapsed_m5_watermarks
    )
    assert dependency_validator._m6_source_migration_errors(
        config, seal, collapsed_m5_watermarks
    )

    definition_rewrite = copy.deepcopy(config)
    definition_rewrite["prior_materialization"]["operational_validation_requeue"][
        "accepted_definition_changes"
    ] = 1
    assert validator._m6_migration_errors(definition_rewrite, seal, migration)
    assert dependency_validator._m6_source_migration_errors(
        definition_rewrite, seal, migration
    )

    drifted_seal_binding = copy.deepcopy(seal)
    drifted_seal_binding["source_migration"]["program_definition_cid"] = (
        "sha256:" + "0" * 64
    )
    assert validator._m6_migration_errors(config, drifted_seal_binding, migration)
    assert dependency_validator._m6_source_migration_errors(
        config, drifted_seal_binding, migration
    )


def test_live_owner_identity_requires_exact_canonical_replica_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_identity_test",
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        _schema_fingerprint_digest,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state

    raw_schema_fingerprint = (
        "baguqeerayrmgjy3yusfmd7c3zcojf6l2wg7l46dnyvlm4homwfipaf5r2nha"
    )
    identity = {
        "server_id": "server:sawm-test",
        "store_id": "store:sawm-test",
        "database_uuid": "database:sawm-test",
        "process_birth_id": "birth:sawm-test",
        "listen_uri": "quack:127.0.0.1:45247",
        "extension_fingerprint": "sha256:" + ("ab" * 32),
        "schema_revision": 1,
        "schema_fingerprint": _schema_fingerprint_digest(raw_schema_fingerprint),
        "generation": 2,
        "fence_epoch": 2,
        "revision": 0,
        "credential_generation": 2,
        "secret_handle": "env://SAWM_QUACK_TOKEN",
    }

    class Result:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self.rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return self.rows

    class Connection:
        state_present = True

        def execute(
            self, sql: str, parameters: list[object] | None = None
        ) -> Result:
            del parameters
            if "FROM state_servers" in sql:
                rows = [
                    (
                        identity["server_id"], identity["store_id"],
                        identity["database_uuid"], identity["process_birth_id"],
                        identity["listen_uri"], identity["extension_fingerprint"],
                        1, 2, "ready", 1,
                    )
                ] if self.state_present else []
                return Result(rows)
            if "FROM store_generations" in sql:
                return Result([(2, 1, 2, 0, identity["database_uuid"], identity["process_birth_id"])])
            if "FROM credentials" in sql:
                return Result([(
                    "cred:server:sawm-test:2", identity["secret_handle"],
                    2, "quack-auth", None, 0,
                )])
            if "FROM control_plane_metadata" in sql:
                return Result([
                    ("database_uuid", identity["database_uuid"]),
                    ("schema_fingerprint", raw_schema_fingerprint),
                    ("schema_version", "1"),
                ])
            raise AssertionError(sql)

        def close(self) -> None:
            return None

    connection = Connection()
    monkeypatch.setattr(
        duckdb_state,
        "open_quack_transport_connection",
        lambda _uri, *, token: connection,
    )
    observed = operator._remote_owner_identity(
        identity["listen_uri"], "opaque-test-token", identity
    )
    assert observed["canonical_rows_verified"] is True
    connection.state_present = False
    with pytest.raises(operator.OperatorError, match="missing or ambiguous"):
        operator._remote_owner_identity(
            identity["listen_uri"], "opaque-test-token", identity
        )


def test_board_gate_binds_dependency_validator_native_authorization_and_extensions() -> None:
    validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_dependency_binding_test",
    )
    config = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    seal = json.loads(
        (REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json")
        .read_text(encoding="utf-8")
    )

    assert validator._configured_board_dependency_errors(REPO_ROOT, config, seal) == []

    wrong_validator = copy.deepcopy(config)
    wrong_validator["dependency_validator_path"] = "scripts/other_validator.py"
    assert any(
        "dependency_validator_path" in error
        for error in validator._configured_board_dependency_errors(
            REPO_ROOT, wrong_validator, seal
        )
    )

    wrong_httpfs = copy.deepcopy(config)
    wrong_httpfs["quack_owner"]["pinned_httpfs_extension"]["version"] = "stale"
    assert any(
        "scheduler httpfs pin differs" in error
        for error in validator._configured_board_dependency_errors(
            REPO_ROOT, wrong_httpfs, seal
        )
    )

    wrong_projection = copy.deepcopy(seal)
    wrong_projection["configured_board_quack_projection"]["pin"][
        "projection_id"
    ] = "sha256:" + ("11" * 32)
    assert any(
        "Quack projection is invalid" in error
        for error in validator._configured_board_dependency_errors(
            REPO_ROOT, config, wrong_projection
        )
    )

    for field, value in (
        ("engine_version", "v9.9.9"),
        ("platform", "linux_amd64"),
    ):
        wrong_native_binding = copy.deepcopy(seal)
        projection_pin = wrong_native_binding[
            "configured_board_quack_projection"
        ]["pin"]
        projection_pin[field] = value
        _reidentify_extension_projection(projection_pin)
        assert any(
            "extension engine/platform differs from native DuckDB/toolchain"
            in error
            for error in validator._configured_board_dependency_errors(
                REPO_ROOT,
                config,
                wrong_native_binding,
            )
        )

    wrong_httpfs_root = copy.deepcopy(seal)
    wrong_httpfs_config = copy.deepcopy(config)
    httpfs_pin = wrong_httpfs_root["httpfs_extension_pin"]
    wrong_parent = Path(httpfs_pin["path"]).parent.parent / "linux_amd64"
    httpfs_pin["path"] = str(wrong_parent / "httpfs.duckdb_extension")
    httpfs_pin["info_path"] = str(
        wrong_parent / "httpfs.duckdb_extension.info"
    )
    wrong_httpfs_config["quack_owner"]["pinned_httpfs_extension"] = copy.deepcopy(
        httpfs_pin
    )
    assert any(
        "httpfs extension path engine/platform differs from native DuckDB/toolchain"
        in error
        for error in validator._configured_board_dependency_errors(
            REPO_ROOT,
            wrong_httpfs_config,
            wrong_httpfs_root,
        )
    )

    wrong_authorization = copy.deepcopy(seal)
    wrong_authorization["configured_board_native_dependency"]["acceptance"][
        "authorization_id"
    ] = "sha256:" + ("00" * 32)
    assert any(
        "authorization does not bind" in error
        for error in validator._configured_board_dependency_errors(
            REPO_ROOT, config, wrong_authorization
        )
    )


def _synthetic_extension_pin(
    root: Path,
    name: str,
    version: str,
    payload: bytes,
) -> dict[str, object]:
    extension_root = root / "v1.5.5" / "linux_arm64"
    extension_root.mkdir(parents=True, exist_ok=True)
    extension = extension_root / f"{name}.duckdb_extension"
    info = extension_root / f"{name}.duckdb_extension.info"
    extension.write_bytes(payload)
    info_payload = f"metadata:{name}".encode()
    info.write_bytes(info_payload)
    pin: dict[str, object] = {
        "path": str(extension),
        "info_path": str(info),
        "version": version,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "info_sha256": hashlib.sha256(info_payload).hexdigest(),
        "info_size": len(info_payload),
        "network_install_allowed": False,
        "unsigned_extension_allowed": False,
    }
    if name == "quack":
        pin["service_external_access_limitation"] = (
            "canonical_writer_sealed; "
            "pinned_extension_preloaded_only_in_locked_read_only_loopback_replica"
        )
    return pin


def _install_synthetic_quack_native_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    preload_error: Exception | None = None,
) -> tuple[int, int]:
    """Install a non-native exact bootstrap double with a real held fd."""

    # These tests model the operator's sanitized process birth.  Host shells
    # may carry OpenFOAM or other native-loader overrides, which production
    # correctly rejects before touching the sealed dependency.  Remove every
    # loader override through monkeypatch so the synthetic positive path is
    # explicit and the hostile-loader test below remains independent.
    for name in tuple(os.environ):
        if name.startswith("LD_"):
            monkeypatch.delenv(name, raising=False)

    from ipfs_accelerate_py import agent_implementation_route as native_route
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        configured_board_scheduler as scheduler,
    )

    for alias in ("_duckdb", "duckdb"):
        monkeypatch.delitem(sys.modules, alias, raising=False)
    descriptor, writer = os.pipe()
    executable = f"/proc/self/fd/{descriptor}"
    module = SimpleNamespace(__file__=executable, __version__="1.5.5")
    launch = SimpleNamespace(
        descriptor=SimpleNamespace(descriptor=descriptor),
        pin=SimpleNamespace(
            dependency_id="sha256:" + "1" * 64,
            distribution_version="1.5.5",
        ),
    )
    board = SimpleNamespace()
    snapshot = SimpleNamespace()

    monkeypatch.setattr(
        scheduler,
        "load_configured_board",
        lambda *_args, **_kwargs: events.append("load_board") or board,
    )
    monkeypatch.setattr(
        scheduler,
        "_configured_board_dependency_seal_snapshot",
        lambda value: events.append("snapshot") or snapshot
        if value is board
        else (_ for _ in ()).throw(AssertionError("wrong board")),
    )

    def seal(value: object, *, dependency_seal_snapshot: object) -> object:
        assert value is board
        assert dependency_seal_snapshot is snapshot
        events.append("seal")
        return launch

    monkeypatch.setattr(
        scheduler,
        "_seal_configured_board_native_dependency",
        seal,
    )

    def preload(value: object) -> object:
        assert value is launch
        events.append("preload")
        if preload_error is not None:
            raise preload_error
        monkeypatch.setitem(sys.modules, "_duckdb", module)
        monkeypatch.setitem(sys.modules, "duckdb", module)
        return module

    def verify(value: object) -> str:
        assert value is launch
        os.fstat(descriptor)
        events.append("verify_fd")
        return executable

    monkeypatch.setattr(
        native_route,
        "preload_agent_supervisor_native_dependency",
        preload,
    )
    monkeypatch.setattr(
        native_route,
        "verify_agent_supervisor_native_dependency_sealed_fd",
        verify,
    )
    return descriptor, writer


def test_quack_start_preloads_exact_native_before_validation_and_holds_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_quack_native_order_test",
    )
    events: list[str] = []
    descriptor, writer = _install_synthetic_quack_native_bootstrap(
        monkeypatch,
        events,
    )

    class Transport:
        def __init__(self, owner: object) -> None:
            assert owner == "test"
            events.append("transport")

        def prepare_extension_custody(self) -> None:
            os.fstat(descriptor)
            events.append("custody")

        def stop(self) -> None:
            events.append("transport_stop")

    monkeypatch.setattr(operator, "_SawmQuackTransport", Transport)

    def validate(config: object, config_path: Path) -> None:
        assert config == {"quack_owner": "test"}
        assert config_path == REPO_ROOT / "config/test-quack.json"
        os.fstat(descriptor)
        events.append("validate")

    def start(config: object, *, transport: object) -> int:
        assert config == {"quack_owner": "test"}
        assert isinstance(transport, Transport)
        os.fstat(descriptor)
        events.append("start")
        return 17

    monkeypatch.setattr(operator, "_validate_offline_quack_start", validate)
    monkeypatch.setattr(operator, "_start_quack", start)
    try:
        assert operator._run_quack_start(
            {"quack_owner": "test"},
            REPO_ROOT / "config/test-quack.json",
        ) == 17
        assert events == [
            "load_board",
            "snapshot",
            "seal",
            "preload",
            "verify_fd",
            "transport",
            "custody",
            "validate",
            "start",
            "verify_fd",
        ]
        with pytest.raises(OSError):
            os.fstat(descriptor)
    finally:
        os.close(writer)


def test_quack_start_reports_custody_quota_before_validation_or_owner_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_quack_custody_quota_test",
    )
    events: list[str] = []
    descriptor, writer = _install_synthetic_quack_native_bootstrap(
        monkeypatch,
        events,
    )

    class Transport:
        def __init__(self, owner: object) -> None:
            assert owner == {"authority": "test"}
            events.append("transport")

        def prepare_extension_custody(self) -> None:
            events.append("custody")
            raise operator.QuackExtensionCustodyBlocker(
                operation="inotify_add_watch",
                errno_number=errno.ENOSPC,
            )

        def stop(self) -> None:
            events.append("transport_stop")

    monkeypatch.setattr(operator, "_SawmQuackTransport", Transport)
    monkeypatch.setattr(
        operator,
        "_validate_offline_quack_start",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("custody quota must block before offline database validation")
        ),
    )
    monkeypatch.setattr(
        operator,
        "_start_quack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("custody quota must block before owner/server start")
        ),
    )
    try:
        with pytest.raises(operator.QuackExtensionCustodyBlocker) as raised:
            operator._run_quack_start(
                {"quack_owner": {"authority": "test"}},
                REPO_ROOT / "config/test-quack.json",
            )
        report = raised.value.as_dict()
        assert report == {
            "schema": "sawm/quack-startup-capability-blocker@1",
            "valid": False,
            "terminal": "typed_external_capability",
            "phase": "pre_authoritative_mutation_extension_custody",
            "reason_code": "inotify_watch_quota_exhausted",
            "operation": "inotify_add_watch",
            "errno_number": errno.ENOSPC,
            "errno_name": "ENOSPC",
            "retryable": True,
            "retry_requires_changed_resource_evidence": True,
            "authoritative_database_opened": False,
            "owner_marker_created": False,
            "store_generation_changes": 0,
            "credential_changes": 0,
            "token_handoff_created": False,
            "task_status_changes": 0,
            "completion_changes": 0,
            "recovery": {
                "automatic_process_termination": False,
                "automatic_kernel_limit_change": False,
                "guidance": [
                    "release or fence stale watcher consumers through their own authority",
                    "or have an operator explicitly increase the inotify resource limit",
                    "retry only after the available-capacity evidence changes",
                ],
            },
        }
        assert events == [
            "load_board",
            "snapshot",
            "seal",
            "preload",
            "verify_fd",
            "transport",
            "custody",
            "transport_stop",
            "verify_fd",
        ]
    finally:
        os.close(writer)


def test_quack_custody_blocker_main_output_is_typed_and_credential_free(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_quack_custody_output_test",
    )
    secret = "not-for-operator-output"
    monkeypatch.setenv("SAWM_TEST_QUACK_TOKEN", secret)
    monkeypatch.setattr(operator, "_config", lambda _path: {"quack_owner": {}})
    monkeypatch.setattr(
        operator,
        "_run_quack_start",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            operator.QuackExtensionCustodyBlocker(
                operation="inotify_add_watch",
                errno_number=errno.ENOSPC,
            )
        ),
    )

    assert operator.main(["--config", "unused.json", "quack-start"]) == 2
    rendered = capsys.readouterr().out
    payload = json.loads(rendered)
    assert payload["schema"] == "sawm/quack-startup-capability-blocker@1"
    assert payload["reason_code"] == "inotify_watch_quota_exhausted"
    assert payload["authoritative_database_opened"] is False
    assert secret not in rendered


def test_quack_start_closes_native_fd_when_owner_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_quack_native_owner_failure_test",
    )
    events: list[str] = []
    descriptor, writer = _install_synthetic_quack_native_bootstrap(
        monkeypatch,
        events,
    )

    class Transport:
        def __init__(self, owner: object) -> None:
            assert owner == "test"
            events.append("transport")

        def prepare_extension_custody(self) -> None:
            events.append("custody")

        def stop(self) -> None:
            events.append("transport_stop")

    monkeypatch.setattr(operator, "_SawmQuackTransport", Transport)
    monkeypatch.setattr(
        operator,
        "_validate_offline_quack_start",
        lambda *_args: events.append("validate"),
    )

    def fail_owner(_config: object, *, transport: object) -> int:
        assert isinstance(transport, Transport)
        os.fstat(descriptor)
        events.append("start")
        raise RuntimeError("owner failed")

    monkeypatch.setattr(operator, "_start_quack", fail_owner)
    try:
        with pytest.raises(RuntimeError, match="owner failed"):
            operator._run_quack_start(
                {"quack_owner": "test"},
                REPO_ROOT / "config/test-quack.json",
            )
        assert events[-6:] == [
            "transport",
            "custody",
            "validate",
            "start",
            "transport_stop",
            "verify_fd",
        ]
        with pytest.raises(OSError):
            os.fstat(descriptor)
    finally:
        os.close(writer)


def test_quack_start_preload_failure_never_reaches_validation_or_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_quack_native_preload_failure_test",
    )
    events: list[str] = []
    descriptor, writer = _install_synthetic_quack_native_bootstrap(
        monkeypatch,
        events,
        preload_error=ValueError("wrong native runtime"),
    )
    monkeypatch.setattr(
        operator,
        "_validate_offline_quack_start",
        lambda *_args: events.append("forbidden_validation"),
    )
    monkeypatch.setattr(
        operator,
        "_start_quack",
        lambda *_args: events.append("forbidden_start"),
    )
    try:
        with pytest.raises(operator.OperatorError, match="failed closed"):
            operator._run_quack_start({}, REPO_ROOT / "config/test-quack.json")
        assert events == ["load_board", "snapshot", "seal", "preload"]
        with pytest.raises(OSError):
            os.fstat(descriptor)
    finally:
        os.close(writer)


def test_quack_start_rejects_preloaded_ambient_duckdb_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_quack_ambient_native_test",
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        configured_board_scheduler as scheduler,
    )

    monkeypatch.setitem(sys.modules, "duckdb", ModuleType("duckdb"))
    monkeypatch.setattr(
        scheduler,
        "load_configured_board",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("board loading must not precede ambient rejection")
        ),
    )
    with pytest.raises(operator.OperatorError, match="preloaded ambient"):
        operator._run_quack_start({}, REPO_ROOT / "config/test-quack.json")


def test_quack_start_rejects_ambient_loader_environment_and_closes_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_quack_ambient_loader_test",
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        configured_board_scheduler as scheduler,
    )

    for alias in ("_duckdb", "duckdb"):
        monkeypatch.delitem(sys.modules, alias, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/hostile-native-loader")
    descriptors_before = set(Path("/proc/self/fd").iterdir())
    captured: dict[str, object] = {}
    original_seal = scheduler._seal_configured_board_native_dependency

    def capture_seal(
        board: object,
        *,
        dependency_seal_snapshot: object,
    ) -> object:
        launch = original_seal(
            board,
            dependency_seal_snapshot=dependency_seal_snapshot,
        )
        captured["launch"] = launch
        return launch

    monkeypatch.setattr(
        scheduler,
        "_seal_configured_board_native_dependency",
        capture_seal,
    )
    monkeypatch.setattr(
        operator,
        "_validate_offline_quack_start",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("ambient loader state must prevent validation")
        ),
    )
    monkeypatch.setattr(
        operator,
        "_start_quack",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("ambient loader state must prevent owner start")
        ),
    )

    with pytest.raises(operator.OperatorError, match="failed closed"):
        operator._run_quack_start({}, operator.CONFIG_PATH)
    # Loader state is an external process-birth precondition. The Python
    # bootstrap refuses it; it never removes LD_* in process, imports DuckDB,
    # or leaves a native descriptor behind.
    assert captured == {}
    assert set(Path("/proc/self/fd").iterdir()) == descriptors_before
    assert os.environ["LD_LIBRARY_PATH"] == "/tmp/hostile-native-loader"
    assert "_duckdb" not in sys.modules
    assert "duckdb" not in sys.modules


def test_operator_loads_and_resolves_exact_extension_names(
    tmp_path: Path,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_extension_mapping_test",
    )
    httpfs = _synthetic_extension_pin(tmp_path, "httpfs", "httpfs-v1", b"httpfs")
    quack = _synthetic_extension_pin(tmp_path, "quack", "quack-v1", b"quack")
    transport = operator._SawmQuackTransport(
        {"pinned_httpfs_extension": httpfs, "pinned_extension": quack}
    )

    class Result:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self._rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return self._rows

    class Connection:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self.rows = rows
            self.statements: list[str] = []

        def execute(self, sql: str) -> Result:
            self.statements.append(sql)
            return Result(self.rows if "duckdb_extensions()" in sql else [])

    transport._ensure_extension_projection()
    install_paths = transport._sealed_extension_set.install_paths
    exact_rows = [
        ("httpfs", str(install_paths["httpfs"]), httpfs["version"]),
        ("quack", str(install_paths["quack"]), quack["version"]),
    ]
    exact = Connection(exact_rows)
    try:
        transport._load_reviewed_extensions(exact)
        assert exact.statements[:2] == ["LOAD httpfs", "LOAD quack"]

        swapped = Connection(
            [
                ("httpfs", exact_rows[1][1], quack["version"]),
                ("quack", exact_rows[0][1], httpfs["version"]),
            ]
        )
        with pytest.raises(operator.OperatorError, match="differs from the reviewed pin"):
            transport._load_reviewed_extensions(swapped)

        duplicate_name = Connection(
            [
                ("httpfs", exact_rows[0][1], httpfs["version"]),
                ("httpfs", exact_rows[1][1], quack["version"]),
            ]
        )
        with pytest.raises(operator.OperatorError, match="missing or ambiguous"):
            transport._load_reviewed_extensions(duplicate_name)

        wrong_version = Connection(
            [
                exact_rows[0],
                ("quack", exact_rows[1][1], "stale-version"),
            ]
        )
        with pytest.raises(operator.OperatorError, match="differs from the reviewed pin"):
            transport._load_reviewed_extensions(wrong_version)
    finally:
        transport.stop()


def test_operator_types_inotify_quota_when_preparing_extension_custody() -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_extension_custody_errno_test",
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_extension_projection import (
        ConfiguredBoardExtensionProjectionError,
    )

    class Seal:
        closed = False

        def verify(self) -> None:
            return None

        def _open_watch(self) -> int:
            ctypes.set_errno(errno.ENOSPC)
            raise ConfiguredBoardExtensionProjectionError(
                "sealed extension set race detector could not bind custody"
            )

        def close(self) -> None:
            self.closed = True

    seal = Seal()
    transport = operator._SawmQuackTransport({})
    transport._sealed_extension_set = seal
    try:
        with pytest.raises(operator.QuackExtensionCustodyBlocker) as raised:
            transport.prepare_extension_custody()
        assert raised.value.operation == "inotify_add_watch"
        assert raised.value.errno_number == errno.ENOSPC
        assert raised.value.reason_code == "inotify_watch_quota_exhausted"
        assert transport._extension_custody_watch is None
    finally:
        transport.stop()
    assert seal.closed is True


def test_operator_reuses_one_prepared_watch_without_nested_load_guard(
    tmp_path: Path,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_extension_custody_reuse_test",
    )
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    httpfs_path = tmp_path / "httpfs.duckdb_extension"
    quack_path = tmp_path / "quack.duckdb_extension"

    class Seal:
        open_calls = 0
        fallback_guard_calls = 0
        closed_after_watch = False
        install_paths = {"httpfs": httpfs_path, "quack": quack_path}

        def verify(self) -> None:
            os.fstat(read_fd)

        def _open_watch(self) -> int:
            self.open_calls += 1
            return read_fd

        @staticmethod
        def _watch_changed(descriptor: int) -> bool:
            try:
                return bool(os.read(descriptor, 64 * 1024))
            except BlockingIOError:
                return False

        @contextlib.contextmanager
        def load_guard(self):
            self.fallback_guard_calls += 1
            yield

        def close(self) -> None:
            with pytest.raises(OSError):
                os.fstat(read_fd)
            self.closed_after_watch = True

    class Result:
        def fetchall(self) -> list[tuple[str, str, str]]:
            return [
                ("httpfs", str(httpfs_path), "httpfs-v1"),
                ("quack", str(quack_path), "quack-v1"),
            ]

    class Connection:
        statements: list[str]

        def __init__(self) -> None:
            self.statements = []

        def execute(self, sql: str) -> Result:
            self.statements.append(sql)
            return Result()

    seal = Seal()
    transport = operator._SawmQuackTransport(
        {
            "pinned_httpfs_extension": {"version": "httpfs-v1"},
            "pinned_extension": {"version": "quack-v1"},
        }
    )
    transport._sealed_extension_set = seal
    connection = Connection()
    try:
        transport.prepare_extension_custody()
        transport._load_reviewed_extensions(connection)
        transport._load_reviewed_extensions(connection)
        assert seal.open_calls == 1
        assert seal.fallback_guard_calls == 0
        assert connection.statements.count("LOAD httpfs") == 2
        assert connection.statements.count("LOAD quack") == 2
        assert transport._extension_custody_watch == read_fd
    finally:
        transport.stop()
        os.close(write_fd)
    assert seal.closed_after_watch is True


def test_operator_prepared_watch_rejects_a_custody_event_during_load(
    tmp_path: Path,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_extension_custody_event_test",
    )
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    httpfs_path = tmp_path / "httpfs.duckdb_extension"
    quack_path = tmp_path / "quack.duckdb_extension"

    class Seal:
        fallback_guard_calls = 0
        install_paths = {"httpfs": httpfs_path, "quack": quack_path}

        def verify(self) -> None:
            os.fstat(read_fd)

        def _open_watch(self) -> int:
            return read_fd

        @staticmethod
        def _watch_changed(descriptor: int) -> bool:
            try:
                return bool(os.read(descriptor, 64 * 1024))
            except BlockingIOError:
                return False

        @contextlib.contextmanager
        def load_guard(self):
            self.fallback_guard_calls += 1
            yield

        def close(self) -> None:
            with pytest.raises(OSError):
                os.fstat(read_fd)

    class Result:
        def fetchall(self) -> list[tuple[str, str, str]]:
            return [
                ("httpfs", str(httpfs_path), "httpfs-v1"),
                ("quack", str(quack_path), "quack-v1"),
            ]

    class TamperingConnection:
        def execute(self, sql: str) -> Result:
            if sql == "LOAD quack":
                os.write(write_fd, b"custody-event")
            return Result()

    seal = Seal()
    transport = operator._SawmQuackTransport(
        {
            "pinned_httpfs_extension": {"version": "httpfs-v1"},
            "pinned_extension": {"version": "quack-v1"},
        }
    )
    transport._sealed_extension_set = seal
    try:
        transport.prepare_extension_custody()
        with pytest.raises(operator.OperatorError, match="custody changed during"):
            transport._load_reviewed_extensions(TamperingConnection())
        assert transport._extension_custody_poisoned is True
        assert seal.fallback_guard_calls == 0
    finally:
        transport.stop()
        os.close(write_fd)


def test_operator_rechecks_extension_bytes_after_native_load(tmp_path: Path) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_extension_tamper_test",
    )
    httpfs = _synthetic_extension_pin(tmp_path, "httpfs", "httpfs-v1", b"httpfs")
    quack = _synthetic_extension_pin(tmp_path, "quack", "quack-v1", b"quack")
    transport = operator._SawmQuackTransport(
        {"pinned_httpfs_extension": httpfs, "pinned_extension": quack}
    )
    transport._ensure_extension_projection()
    install_paths = transport._sealed_extension_set.install_paths

    class Result:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self._rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return self._rows

    class TamperingConnection:
        def execute(self, sql: str) -> Result:
            if sql == "LOAD quack":
                projected = install_paths["quack"]
                metadata = projected.with_name(f"{projected.name}.info")
                os.chmod(metadata, 0o600)
                metadata.write_bytes(b"tamper")
            rows = [
                ("httpfs", str(install_paths["httpfs"]), httpfs["version"]),
                ("quack", str(install_paths["quack"]), quack["version"]),
            ]
            return Result(rows if "duckdb_extensions()" in sql else [])

    try:
        with pytest.raises(operator.OperatorError, match="custody|bytes drifted"):
            transport._load_reviewed_extensions(TamperingConnection())
    finally:
        transport.stop()


def test_sawm_disables_board_extension_access_at_connection_birth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("duckdb")
    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        BOARD_EXTENSION_INSTALL_POLICY_ENV,
        open_board_control_plane,
    )

    monkeypatch.setenv(BOARD_EXTENSION_INSTALL_POLICY_ENV, "disabled")
    plane = open_board_control_plane(
        tmp_path,
        root=tmp_path / "control",
    )
    try:
        settings = plane._conn().execute(
            "SELECT current_setting('autoinstall_known_extensions'), "
            "current_setting('autoload_known_extensions'), "
            "current_setting('enable_external_access'), "
            "current_setting('allow_unsigned_extensions'), "
            "current_setting('lock_configuration')"
        ).fetchone()
        assert tuple(settings[index] for index in range(len(settings))) == (
            False,
            False,
            False,
            False,
            True,
        )
        with pytest.raises(Exception, match="configuration has been lock"):
            plane._conn().execute("SET autoinstall_known_extensions=true")
        assert plane.quack_loaded is False
        assert plane.ducklake_loaded is False
    finally:
        plane.close()


def test_load_only_ducklake_remains_root_confined_after_policy_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("duckdb")
    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        BOARD_EXTENSION_INSTALL_POLICY_ENV,
        open_board_control_plane,
    )

    root = tmp_path / "control"
    monkeypatch.setenv(BOARD_EXTENSION_INSTALL_POLICY_ENV, "load_only")
    plane = open_board_control_plane(tmp_path, root=root)
    try:
        if not plane.ducklake_loaded:
            pytest.skip("the current interpreter has no admitted DuckLake extension")
        assert plane.ducklake_attached is True
        assert plane.backend == (
            "ducklake+quack" if plane.quack_loaded else "hermetic-duckdb"
        )
        plane._project_relations_to_lake()
        settings = plane._conn().execute(
            "SELECT current_setting('enable_external_access'), "
            "current_setting('lock_configuration'), "
            "current_setting('allowed_directories')"
        ).fetchone()
        observed = tuple(settings[index] for index in range(len(settings)))
        assert observed[0:2] == (False, True)
        assert str(root.resolve()) + os.sep in observed[2]
        assert all(
            Path(item).resolve(strict=False).is_relative_to(root.resolve())
            for item in observed[2]
        )
        with pytest.raises(Exception, match="disabled|not allowed"):
            plane._conn().execute("SELECT * FROM read_text('/etc/hosts')")
    finally:
        plane.close()


def test_board_control_plane_closes_connection_when_finalization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import board_control_plane

    class Connection:
        closed = False

        def close(self) -> None:
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(
        board_control_plane,
        "_open_extension_capable_connection",
        lambda *_args, **_kwargs: connection,
    )
    monkeypatch.setattr(
        board_control_plane,
        "_initialize_open_board_control_plane",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("lock failed")),
    )

    with pytest.raises(RuntimeError, match="lock failed"):
        board_control_plane.open_board_control_plane(
            tmp_path,
            root=tmp_path / "control",
        )
    assert connection.closed is True


def test_operator_stop_closes_shared_extension_custody_on_quack_stop_failure() -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_stop_cleanup_test",
    )
    transport = operator._SawmQuackTransport({})
    custody_read, custody_write = os.pipe()

    class Replica:
        closed = False

        def execute(self, _sql: str, _parameters: object = None) -> None:
            raise RuntimeError("quack stop failed")

        def close(self) -> None:
            self.closed = True

    class Seal:
        close_count = 0
        custody_closed_first = False

        def close(self) -> None:
            with pytest.raises(OSError):
                os.fstat(custody_read)
            self.custody_closed_first = True
            self.close_count += 1

    replica = Replica()
    seal = Seal()
    transport._serve_uri = "quack:127.0.0.1:45123"
    transport._replica_connection = replica
    transport._sealed_extension_set = seal
    transport._extension_custody_watch = custody_read

    try:
        with pytest.raises(RuntimeError, match="quack stop failed"):
            transport.stop()
        assert replica.closed is True
        assert seal.close_count == 1
        assert seal.custody_closed_first is True
        assert transport._sealed_extension_set is None
        assert transport._extension_custody_watch is None
    finally:
        os.close(custody_write)


def test_m10_controls_and_live_projection_comparator_fail_closed() -> None:
    dependency_validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m10_projection_test",
    )
    board_validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m10_projection_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m10_projection_test",
    )
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m10_projection_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "live_projection_successor_materialization"
    cid_key = "live_projection_successor_materialization_cid"
    expected_cid = (
        "sha256:f27878def0ee9b406d0dbaac669728278e4f375cb267ee7f76330faaf9b14f10"
    )
    authority = config[key]
    assert authority == migration[key]
    assert "sha256:" + hashlib.sha256(
        json.dumps(
            authority,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest() == expected_cid
    assert seal[cid_key] == expected_cid
    assert dependency_validator._m10_live_projection_errors(
        config,
        seal,
        migration,
    ) == []
    assert board_validator._m10_migration_errors(config, seal, migration) == []
    # M9 is still checked as immutable history, but its old active paths no
    # longer override the key-present M10 generation.
    assert dependency_validator._m9_live_recovery_errors(
        config,
        seal,
        migration,
    ) == []
    historical_config, historical_migration, historical_seal = (
        _historical_successor_controls_at(key, config, migration, seal)
    )
    assert historical_migration is not None
    assert historical_seal is not None
    assert operator._active_source_repair_materialization(
        historical_config
    ) == authority

    assert authority["live_task_projection"] == {
        "schema": "sawm/live-task-projection-expectation@1",
        "task_count": 45,
        "task_revision_count": 1,
        "operator_task_alias": "SAWM-000",
        "operator_status": "completed",
        "operator_revision": 2,
        "candidate_task_alias": "SAWM-001",
        "candidate_task_cid": (
            "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
        ),
        "candidate_status": "retrying",
        "candidate_revision": 10,
        "candidate_completion_receipt": {
            "operation": "operator_control_plane_repair",
            "settlement_id": (
                "baguqeeransrfearh5ojrnru6ls43mn7xsx6mlhndkhokrn4k7wlhowxjnshq"
            ),
        },
        "remaining_task_status": "todo",
        "remaining_task_revision": 2,
    }
    assert authority["task_revision_changes"] == 0
    assert authority["task_status_changes"] == 0
    assert authority["coordination_semantic_changes"] == 0
    assert authority["coordination_sidecar_copied_unchanged"] is True
    assert authority["provider_strategy"]["route_changed"] is False
    repair_paths = set(authority["bounded_control_plane_repair_paths"])
    assert len(repair_paths) == 9
    assert not any("todo_daemon" in path for path in repair_paths)
    assert "test/api/test_agent_supervisor_database_portal_bridge.py" not in repair_paths

    malformed = copy.deepcopy(historical_config)
    malformed[key] = []
    with pytest.raises(
        operator.OperatorError,
        match="active M10 live-projection successor authority is invalid",
    ):
        operator._active_source_repair_materialization(malformed)
    with pytest.raises(
        materializer.MaterializationError,
        match="M10 live projection authority is invalid",
    ):
        materializer._m10_successor_configured(malformed)

    partial_inventory = copy.deepcopy(historical_migration)
    partial_inventory.pop(key)
    assert any(
        "M10" in error
        for error in board_validator._active_successor_migration_errors(
            historical_config,
            historical_seal,
            partial_inventory,
        )
    )
    changed_config = copy.deepcopy(config)
    changed_migration = copy.deepcopy(migration)
    for control in (changed_config, changed_migration):
        control[key]["task_status_changes"] = 1
    assert dependency_validator._m10_live_projection_errors(
        changed_config,
        seal,
        changed_migration,
    )
    changed_config = copy.deepcopy(historical_config)
    changed_config["provider"]["fallback_model_id"] = "changed"
    assert dependency_validator._m10_live_projection_errors(
        changed_config,
        historical_seal,
        historical_migration,
    )

    operator_source = Path(operator.__file__).read_text(encoding="utf-8")
    materializer_source = Path(materializer.__file__).read_text(encoding="utf-8")
    assert operator_source.index(key) < operator_source.index(
        "live_recovery_successor_materialization"
    )
    assert "_verify_m9_head_task_projection" in operator_source
    assert (
        'active_source_repair["target_semantic_authority_digest"]'
        in operator_source
    )
    assert "_require_active_final_pair_marker" in operator_source
    assert "def _m10_successor_configured" in materializer_source
    assert "_verify_m9_live_task_projection" in materializer_source
    assert 'candidate.status != "retrying"' in materializer_source
    assert "candidate.revision != 10" in materializer_source


def test_m10_projection_identity_binds_the_exact_m9_task_head() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m10_projection_identity_test",
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    population = materializer.build_population(REPO_ROOT)
    material = {
        "objectives": 1,
        "goals": [
            {
                "goal_cid": str(goal["goal_cid"]),
                "status": str(goal["status"]),
                "revision": 1,
            }
            for goal in sorted(
                population["objectives"], key=lambda item: str(item["goal_cid"])
            )
        ],
        "plans": [
            {
                "plan_cid": str(population["plan_root_cid"]),
                "status": "active",
                "revision": 11,
            }
        ],
        "tasks": [
            {
                "task_cid": str(task["task_cid"]),
                "status": (
                    "completed"
                    if task["task_id"] == "SAWM-000"
                    else "retrying"
                    if task["task_id"] == "SAWM-001"
                    else "todo"
                ),
                "revision": (
                    2
                    if task["task_id"] == "SAWM-000"
                    else 10
                    if task["task_id"] == "SAWM-001"
                    else 2
                ),
            }
            for task in sorted(
                population["taskboard"], key=lambda item: str(item["task_cid"])
            )
        ],
        "dependency_count": 136,
        "event_watermark": 179,
    }
    assert content_identity(material) == (
        "baguqeerareq2bngq3hffyk5vidym2ukeleg5gehpaxhqvvdjayn7ucxplcaq"
    )


def test_m10_live_projection_comparators_accept_only_m9_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m10_live_comparator_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m10_live_comparator_test",
    )
    population = materializer.build_population(REPO_ROOT)
    rearm_receipt = materializer._m9_task_rearm_receipt()

    candidate = SimpleNamespace(
        task_cid=(
            "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
        ),
        status="retrying",
        revision=10,
        body={"completion_receipt": rearm_receipt},
    )
    candidate_source = SimpleNamespace(get_task=lambda _task: candidate)
    monkeypatch.setattr(
        materializer,
        "_verify_m6_task_projection",
        lambda *_args, **_kwargs: ({}, {}, {}),
    )
    assert materializer._verify_m9_live_task_projection(
        candidate_source,
        population,
    ) == ({}, {}, {})
    candidate.status = "todo"
    with pytest.raises(
        materializer.MigrationRequired,
        match="frozen M9 task rearm projection differs",
    ):
        materializer._verify_m9_live_task_projection(
            candidate_source,
            population,
        )
    candidate.status = "retrying"

    tasks: dict[str, SimpleNamespace] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        body: dict[str, object] = {}
        if alias == "SAWM-001":
            body["completion_receipt"] = rearm_receipt
        tasks[str(expected["task_cid"])] = SimpleNamespace(
            status=(
                "completed"
                if alias == "SAWM-000"
                else "retrying"
                if alias == "SAWM-001"
                else "todo"
            ),
            revision=(
                2
                if alias == "SAWM-000"
                else 10
                if alias == "SAWM-001"
                else 2
            ),
            body=body,
        )

    task_revision_rows = [
        (
            candidate.task_cid,
            10,
            "retrying",
            json.dumps({"completion_receipt": rearm_receipt}),
        )
    ]

    class Result:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self.rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return self.rows

        def fetchone(self) -> tuple[object, ...]:
            return self.rows[0]

    class Connection:
        def execute(self, statement: str) -> Result:
            if "FROM task_revisions" in statement:
                return Result(task_revision_rows)
            if "FROM completion_receipts" in statement:
                return Result([(1,)])
            raise AssertionError(f"unexpected comparator query: {statement}")

    class ConnectionContext:
        def __enter__(self) -> Connection:
            return Connection()

        def __exit__(self, *_args: object) -> None:
            return None

    source = SimpleNamespace(
        get_task=lambda task_cid: tasks.get(task_cid),
        intent=SimpleNamespace(
            _connection=lambda *, write=False: ConnectionContext()
        ),
    )

    def historical_projection_mismatch(*_args: object, **_kwargs: object) -> None:
        raise materializer.MigrationRequired(
            "frozen M6 task status/revision projection differs"
        )

    monkeypatch.setattr(
        materializer,
        "_verify_m6_task_projection",
        historical_projection_mismatch,
    )
    statuses, revisions, _receipts = operator._verify_m9_head_task_projection(
        source,
        population,
        materializer,
    )
    assert statuses["SAWM-001"] == "retrying"
    assert revisions["SAWM-001"] == 10
    tasks[candidate.task_cid].revision = 7
    with pytest.raises(
        materializer.MigrationRequired,
        match="M9-head task status/revision differs: SAWM-001",
    ):
        operator._verify_m9_head_task_projection(
            source,
            population,
            materializer,
        )


def _build_m9_rehearsal_pair(
    materializer: ModuleType,
    *,
    control: Path,
    coordination: Path,
    prior_control: Path,
    prior_coordination: Path,
    population: dict[str, object],
    config: dict[str, object],
    validation_digest: str,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinator,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources import intent_repository
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    shutil.copyfile(prior_control, control)
    shutil.copyfile(prior_coordination, coordination)
    # The historical M9 verifier requires the plan row and its append event to
    # carry the same second-resolution timestamp.  The repository samples the
    # clock separately for those two records, so freeze it for this disposable
    # rehearsal and restore the production clock before returning.
    original_clock = intent_repository._utc_iso
    frozen_now = original_clock()
    intent_repository._utc_iso = lambda _moment=None: frozen_now
    try:
        source = DatabaseTaskSource(
            control,
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
            owner_id="sawm-r2-runtime-recovery-migrator",
        )
        try:
            operator = source.get_task("SAWM-000")
            assert operator is not None
            body = materializer._m9_migration_body(
                population,
                config,
                validation_digest,
            )
            digest = materializer._identity(body)
            source.plans.append_revision(
                plan_cid=str(population["plan_root_cid"]),
                expected_revision=9,
                body={
                    "current_source_binding_cid": population["source_binding"][
                        "source_binding_cid"
                    ],
                    "source_migration_revision": materializer._M9_MIGRATION_REVISION,
                    "source_migration_digest": digest,
                    "supersession_mode": materializer._M9_SUPERSESSION_MODE,
                },
                delta=materializer._m9_migration_plan_delta(population, config),
            )
            source.record_evidence(
                task_cid=operator.task_cid,
                evidence_kind="operator_control_plane_runtime_recovery",
                digest=digest,
                body=body,
            )
            task_cas = source.compare_and_set_status(
                materializer._M8_LIVE_FAILURE_RECEIPT["task_cid"],
                9,
                "retrying",
                materializer._m9_task_rearm_receipt(),
            )
        finally:
            source.close()
    finally:
        intent_repository._utc_iso = original_clock
    coordinator = DatabaseCoordinator(coordination).open()
    try:
        rearm = coordinator.rearm_failed_task(
            failure_receipt=materializer._M8_LIVE_FAILURE_RECEIPT,
            control_task_observation=task_cas.to_dict(),
            now_ms=materializer._M9_COORDINATION_REARM_OBSERVED_AT_MS,
        )
    finally:
        coordinator.close()
    assert rearm["ready"] is True
    assert rearm["replayed"] is False


def test_m9_exact_control_and_coordination_rehearsal(tmp_path: Path) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m9_exact_rehearsal_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._m9_live_recovery_authority(population, config)
    assert materializer._identity(authority) == (
        "sha256:e7834d12a6150a7d4dd1f90dcb5369d73a6d8620a38bf6baa0cbbb3da17bebb9"
    )
    frozen = materializer._verify_frozen_m8_live_authority(
        REPO_ROOT,
        authority,
    )
    assert frozen["append_surface_digest"] == authority[
        "prior_append_surface_digest"
    ]
    malformed = copy.deepcopy(config)
    malformed["live_recovery_successor_materialization"] = []
    with pytest.raises(
        materializer.MaterializationError,
        match="M9 live recovery authority is invalid",
    ):
        materializer._m9_successor_configured(malformed)

    prior_control = REPO_ROOT / authority["prior_store_id"]
    prior_coordination = REPO_ROOT / authority["prior_coordination_store_id"]
    control = tmp_path / "control.duckdb"
    coordination = tmp_path / "control.coordination.duckdb"
    validation_digest = "sha256:m9-focused-rehearsal"
    before_prior = (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    )

    symlink_root = tmp_path / "symlink-predecessor"
    expected_parent = (
        symlink_root
        / "data/agent_supervisor/semantic_addressed_world_model/run-r2-m8"
    )
    expected_parent.mkdir(parents=True)
    relocated = symlink_root / "relocated-control.duckdb"
    shutil.copyfile(prior_control, relocated)
    (expected_parent / "control.duckdb").symlink_to(relocated)
    with pytest.raises(
        materializer.MigrationRequired,
        match="no-follow regular file",
    ):
        materializer._verify_frozen_m8_live_authority(
            symlink_root,
            authority,
        )
    with pytest.raises(
        materializer.MigrationRequired,
        match="no-follow regular file",
    ):
        materializer._assert_m9_prior_publication_anchor(
            symlink_root,
            authority,
        )

    _build_m9_rehearsal_pair(
        materializer,
        control=control,
        coordination=coordination,
        prior_control=prior_control,
        prior_coordination=prior_coordination,
        population=population,
        config=config,
        validation_digest=validation_digest,
    )
    before_verify = (
        materializer._store_sha256(control),
        materializer._store_sha256(coordination),
    )
    report = materializer._verify_m9_store_pair_copy(
        control,
        coordination,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    assert report["projection_cid"] == materializer._M9_EXPECTED_PROJECTION_CID
    assert report["event_watermark"] == 177
    assert report["coordination_projection_digest"] == (
        materializer._M9_COORDINATION_PROJECTION_DIGEST
    )
    assert report["coordination_event_count"] == 36
    assert report["task_revision_changes"] == 1
    assert before_verify == (
        materializer._store_sha256(control),
        materializer._store_sha256(coordination),
    )
    assert before_prior == (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    )

    import duckdb

    connection = duckdb.connect(str(control))
    try:
        connection.execute(
            "UPDATE goals SET title = title || '-tampered' "
            "WHERE goal_cid = (SELECT MIN(goal_cid) FROM goals)"
        )
    finally:
        connection.close()
    with pytest.raises(
        materializer.MigrationRequired,
        match="non-authorized control table",
    ):
        materializer._verify_m9_store_pair_copy(
            control,
            coordination,
            prior_control,
            prior_coordination,
            population,
            config,
            validation_digest,
        )


def test_m9_receipt_is_last_single_link_pair_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m9_receipt_recovery_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._m9_live_recovery_authority(population, config)
    prior_control = tmp_path / "prior-control.duckdb"
    prior_coordination = tmp_path / "prior-control.coordination.duckdb"
    shutil.copyfile(REPO_ROOT / authority["prior_store_id"], prior_control)
    shutil.copyfile(
        REPO_ROOT / authority["prior_coordination_store_id"],
        prior_coordination,
    )
    target = tmp_path / "target"
    target.mkdir()
    control = target / "control.duckdb"
    coordination = target / "control.coordination.duckdb"
    validation_digest = "sha256:m9-receipt-recovery"
    _build_m9_rehearsal_pair(
        materializer,
        control=control,
        coordination=coordination,
        prior_control=prior_control,
        prior_coordination=prior_coordination,
        population=population,
        config=config,
        validation_digest=validation_digest,
    )
    verified = materializer._verify_m9_store_pair(
        tmp_path,
        control,
        coordination,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_committed_clean_source",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m9_source_delta",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m9_prior_publication_anchor",
        lambda *_args, **_kwargs: (prior_control, prior_coordination),
    )
    receipt = materializer._ensure_m9_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    )
    receipt_path = target / "migration-receipt.json"
    assert receipt["receipt_is_final_pair_commit_marker"] is True
    assert receipt_path.stat().st_nlink == 1
    assert control.stat().st_nlink == coordination.stat().st_nlink == 1
    assert not list(target.glob(".migration-receipt.json.*.tmp"))
    assert materializer._ensure_m9_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    ) == receipt

    receipt_path.unlink()
    pending = target / ".migration-receipt.json.999999.tmp"
    pending.write_bytes(materializer._canonical(receipt) + b"\n")
    os.link(pending, receipt_path)
    assert receipt_path.stat().st_nlink == 2
    recovered = materializer._ensure_m9_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    )
    assert recovered == receipt
    assert receipt_path.stat().st_nlink == 1
    assert not pending.exists()

    alias = target / "mutable-control-alias.duckdb"
    os.link(control, alias)
    with pytest.raises(
        materializer.MigrationRequired,
        match="exactly 1 link",
    ):
        materializer._ensure_m9_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    alias.unlink()
    execution = target / "control.execution.duckdb"
    execution.write_bytes(b"not-copied")
    with pytest.raises(
        materializer.MigrationRequired,
        match="execution sidecar exists",
    ):
        materializer._ensure_m9_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    execution.unlink()

    # Reproduce a sidecar appearing after full pair verification but before
    # the receipt lock is acquired.  The final commit-input gate must refuse
    # to publish the marker.
    receipt_path.unlink()
    original_pair_verifier = materializer._verify_m9_store_pair

    def inject_execution_sidecar(*args: object, **kwargs: object) -> object:
        result = original_pair_verifier(*args, **kwargs)
        execution.write_bytes(b"appeared-during-verification")
        return result

    monkeypatch.setattr(
        materializer,
        "_verify_m9_store_pair",
        inject_execution_sidecar,
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="execution sidecar exists at receipt commit",
    ):
        materializer._ensure_m9_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    assert not receipt_path.exists()
    execution.unlink()
    monkeypatch.setattr(
        materializer,
        "_verify_m9_store_pair",
        original_pair_verifier,
    )

    # If the store changes after the last pre-link recheck, the post-link
    # check removes only the marker created by this invocation and retains the
    # pending receipt bytes for crash/audit recovery.
    original_commit_gate = materializer._assert_m9_receipt_commit_inputs
    commit_gate_calls = 0

    def mutate_after_prelink_gate(*args: object, **kwargs: object) -> None:
        nonlocal commit_gate_calls
        original_commit_gate(*args, **kwargs)
        commit_gate_calls += 1
        if commit_gate_calls == 2:
            with control.open("ab") as handle:
                handle.write(b"post-gate-mutation")
                handle.flush()
                os.fsync(handle.fileno())

    monkeypatch.setattr(
        materializer,
        "_assert_m9_receipt_commit_inputs",
        mutate_after_prelink_gate,
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="store pair changed at receipt commit",
    ):
        materializer._ensure_m9_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    assert not receipt_path.exists()
    assert len(list(target.glob(".migration-receipt.json.*.tmp"))) == 1


def test_m11_controls_and_provider_retry_authority_fail_closed() -> None:
    dependency_validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m11_provider_retry_test",
    )
    board_validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m11_provider_retry_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m11_provider_retry_test",
    )
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m11_provider_retry_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "live_provider_retry_successor_materialization"
    cid_key = "live_provider_retry_successor_materialization_cid"
    expected_cid = (
        "sha256:7f8404735adae0fb7ed890cda97dc674b78ae915efa193207e98abd95eb5193e"
    )
    failure_cid = (
        "sha256:a1fae6ef892b02a19b1155a0a5ed7c0eefa8f8255f3949e25d5a8b8b9d721676"
    )
    authority = config[key]

    assert authority == migration[key]
    assert "sha256:" + hashlib.sha256(
        json.dumps(
            authority,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest() == expected_cid
    assert seal[cid_key] == expected_cid
    assert materializer._expected_m11_live_provider_retry_authority() == authority
    assert materializer._identity(authority) == expected_cid
    population = materializer.build_population(REPO_ROOT)
    assert materializer._m11_live_provider_retry_authority(
        population,
        config,
    ) == authority
    assert materializer._m11_successor_configured(config) is True
    assert dependency_validator._m11_provider_retry_errors(
        config,
        seal,
        migration,
    ) == []
    assert board_validator._m11_migration_errors(config, seal, migration) == []
    assert dependency_validator._m10_live_projection_errors(
        config,
        seal,
        migration,
    ) == []
    historical_config, historical_migration, historical_seal = (
        _historical_successor_controls_at(key, config, migration, seal)
    )
    assert historical_migration is not None
    assert historical_seal is not None
    assert operator._active_source_repair_materialization(historical_config) == authority

    assert authority["schema"] == "sawm/provider-launch-repair-authorization@1"
    assert authority["migration_revision"] == "SAWM-R2-M11"
    assert authority["prior_event_watermark"] == 181
    assert authority["target_plan_revision"] == 12
    assert authority["target_event_watermark"] == 184
    assert authority["target_generation"] == 13
    assert authority["target_projection_cid"] == (
        "baguqeeratdygaminyfax543hh5bik3kj5admm5filgtu37a3jfnyc6snwnxa"
    )
    assert authority["prior_coordination_wal_sha256"] == (
        "938dbc7028ec438889019c352b5a2529cc8c0f0fd18a11f0eb157a6772fc5321"
    )
    assert authority["prior_coordination_wal_size"] == 24_746
    assert authority["prior_coordination_event_count"] == 55
    assert authority["target_coordination_event_count"] == 56
    assert authority["coordination_wal_replayed_on_copy"] is True
    assert authority["prior_coordination_store_mutated"] is False
    assert authority["task_rearm"]["from_status"] == "blocked"
    assert authority["task_rearm"]["from_revision"] == 12
    assert authority["task_rearm"]["to_status"] == "retrying"
    assert authority["task_rearm"]["to_revision"] == 13
    assert authority["task_rearm"]["settlement_id"] == (
        "baguqeera7cwinhpjgl2etuitwiuhs4ts6lix5txyb2pjtsbfz2npfyryznma"
    )
    failure = authority["live_implementation_failure"]
    assert authority["live_implementation_failure_cid"] == failure_cid
    assert materializer._identity(failure) == failure_cid
    assert failure["provider_execution_observed"] is True
    assert failure["provider_execution_accounting_mismatch"] is True
    assert failure["settlement_provider_invocation_count"] == 0
    assert authority["implementation_provider_invocations_observed"] == 1
    assert authority["settlement_provider_invocation_count"] == 0
    for field in (
        "effect_claim_changes",
        "implementation_commit_changes",
        "merge_attempt_changes",
        "accepted_definition_changes",
        "accepted_completion_changes",
    ):
        assert authority[field] == 0
    assert authority["execution_sidecar_copied"] is False
    assert authority["worker_self_approval"] is False
    assert authority["provider_strategy"] == {
        "capability_probe_required": True,
        "prior_route": "grok-4.6_then_codex_on_independently_verified_quota",
        "provider_result_is_completion_authority": False,
        "route_changed": False,
        "target_route": "grok-4.6_then_codex_on_independently_verified_quota",
    }

    repair_paths = set(authority["bounded_control_plane_repair_paths"])
    assert len(repair_paths) == 11
    assert authority["runner_repair"]["runner_path"] in repair_paths
    assert authority["runner_repair"]["focused_test_path"] in repair_paths
    assert repair_paths.issubset(dependency_validator.CONTROL_PATHS)
    assert repair_paths.issubset(board_validator.CONTROL_RELATIVE_PATHS)
    assert repair_paths.issubset(
        set(population["source_binding"]["control_sha256"])
    )
    assert repair_paths.issubset(set(config["protected_paths"]))
    assert repair_paths.issubset(
        set(config["configured_board_live_capsule"]["control_paths"])
    )

    malformed = copy.deepcopy(historical_config)
    malformed[key] = []
    with pytest.raises(
        operator.OperatorError,
        match="active M11 provider-retry successor authority is invalid",
    ):
        operator._active_source_repair_materialization(malformed)
    with pytest.raises(
        materializer.MaterializationError,
        match="M11 live provider retry authority is invalid",
    ):
        materializer._m11_successor_configured(malformed)
    assert any(
        "M11" in error
        for error in board_validator._active_successor_migration_errors(
            malformed,
            historical_seal,
            historical_migration,
        )
    )

    partial_inventory = copy.deepcopy(historical_migration)
    partial_inventory.pop(key)
    assert any(
        "M11" in error
        for error in board_validator._active_successor_migration_errors(
            historical_config,
            historical_seal,
            partial_inventory,
        )
    )
    partial_seal = copy.deepcopy(historical_seal)
    partial_seal.pop(cid_key)
    assert any(
        "M11" in error
        for error in board_validator._active_successor_migration_errors(
            historical_config,
            partial_seal,
            historical_migration,
        )
    )

    def rejected(mutator: object) -> None:
        changed_config = copy.deepcopy(config)
        changed_migration = copy.deepcopy(migration)
        mutator(changed_config[key])
        mutator(changed_migration[key])
        assert dependency_validator._m11_provider_retry_errors(
            changed_config,
            seal,
            changed_migration,
        )

    mutators = (
        lambda item: item.__setitem__("prior_coordination_wal_size", 24_745),
        lambda item: item.__setitem__("target_event_watermark", 183),
        lambda item: item["task_rearm"].__setitem__("to_revision", 14),
        lambda item: item["runner_repair"].__setitem__(
            "provider_execution_observed", False
        ),
        lambda item: item.__setitem__(
            "implementation_provider_invocations_observed", 0
        ),
        lambda item: item.__setitem__("execution_sidecar_copied", True),
        lambda item: item["provider_strategy"].__setitem__(
            "route_changed", True
        ),
        lambda item: item.__setitem__("worker_self_approval", True),
        lambda item: item["bounded_control_plane_repair_paths"].pop(),
    )
    for mutator in mutators:
        rejected(mutator)

    operator_source = Path(operator.__file__).read_text(encoding="utf-8")
    materializer_source = Path(materializer.__file__).read_text(encoding="utf-8")
    assert operator_source.index(key) < operator_source.index(
        "live_projection_successor_materialization"
    )
    assert "_verify_m11_head_task_projection" in operator_source
    assert "_require_m11_final_pair_marker" in operator_source
    assert materializer_source.index("def _m11_successor_configured") < (
        materializer_source.index("def _m10_target_paths")
    )
    assert "def _assert_m11_source_delta" in materializer_source


def test_m11_live_task_comparator_requires_exact_append_only_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m11_comparator_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m11_comparator_test",
    )
    population = materializer.build_population(REPO_ROOT)
    candidate_cid = (
        "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
    )
    tasks: dict[str, SimpleNamespace] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        body: dict[str, object] = {}
        if alias == "SAWM-001":
            body["completion_receipt"] = dict(operator._M11_REARM_RECEIPT)
        tasks[str(expected["task_cid"])] = SimpleNamespace(
            status=(
                "completed"
                if alias == "SAWM-000"
                else "retrying"
                if alias == "SAWM-001"
                else "todo"
            ),
            revision=(
                2
                if alias == "SAWM-000"
                else 13
                if alias == "SAWM-001"
                else 2
            ),
            body=body,
        )
    history = [
        (candidate_cid, 10, "retrying", operator._M9_REARM_RECEIPT),
        (candidate_cid, 11, "in_progress", operator._M10_CLAIM_RECEIPT),
        (candidate_cid, 12, "blocked", operator._M10_FAILURE_RECEIPT),
        (candidate_cid, 13, "retrying", operator._M11_REARM_RECEIPT),
    ]

    class Result:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self.rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return self.rows

        def fetchone(self) -> tuple[object, ...]:
            return self.rows[0]

    class Connection:
        def execute(self, statement: str) -> Result:
            if "FROM task_revisions" in statement:
                return Result(
                    [
                        (task_cid, revision, status, json.dumps({
                            "completion_receipt": dict(receipt)
                        }))
                        for task_cid, revision, status, receipt in history
                    ]
                )
            if "FROM completion_receipts" in statement:
                return Result([(1,)])
            raise AssertionError(f"unexpected comparator query: {statement}")

    class ConnectionContext:
        def __enter__(self) -> Connection:
            return Connection()

        def __exit__(self, *_args: object) -> None:
            return None

    source = SimpleNamespace(
        get_task=lambda task_cid: tasks.get(task_cid),
        intent=SimpleNamespace(
            _connection=lambda *, write=False: ConnectionContext()
        ),
    )
    monkeypatch.setattr(
        materializer,
        "_verify_m6_task_projection",
        lambda *_args, **_kwargs: ({}, {}, {}),
    )
    statuses, revisions, _receipts = operator._verify_m11_head_task_projection(
        source,
        population,
        materializer,
    )
    assert statuses["SAWM-001"] == "retrying"
    assert revisions["SAWM-001"] == 13

    history.pop(1)
    with pytest.raises(
        materializer.MigrationRequired,
        match="M11-head task revision/completion history differs",
    ):
        operator._verify_m11_head_task_projection(
            source,
            population,
            materializer,
        )
    history.insert(
        1,
        (candidate_cid, 11, "in_progress", operator._M10_CLAIM_RECEIPT),
    )
    tasks[candidate_cid].revision = 12
    with pytest.raises(
        materializer.MigrationRequired,
        match="M11-head task status/revision differs: SAWM-001",
    ):
        operator._verify_m11_head_task_projection(
            source,
            population,
            materializer,
        )


def test_m11_prior_base_wal_replay_is_disposable_and_tamper_closed(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m11_wal_replay_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._m11_live_provider_retry_authority(
        population,
        config,
    )
    copied_root = tmp_path / "disposable-authority"
    copied_paths: list[Path] = []
    for field in (
        "prior_store_id",
        "prior_coordination_store_id",
        "prior_coordination_wal_id",
        "prior_materialization_receipt_path",
        "prior_owner_status_path",
    ):
        relative = Path(authority[field])
        source = REPO_ROOT / relative
        target = copied_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied_paths.append(target)
    source_hashes = {
        field: materializer._store_sha256(REPO_ROOT / authority[field])
        for field in (
            "prior_store_id",
            "prior_coordination_store_id",
            "prior_coordination_wal_id",
        )
    }

    _control, _coordination, _wal, report = (
        materializer._assert_m11_prior_publication_anchor(
            copied_root,
            authority,
            population,
        )
    )
    assert report["coordination_event_count"] == 55
    assert report["coordination_projection_digest"] == (
        "sha256:25b8bd03cfe9a684a2626a6c5e6c955b295ba6073248c7a1cf2cd83d7550576d"
    )
    assert {
        field: materializer._store_sha256(REPO_ROOT / authority[field])
        for field in source_hashes
    } == source_hashes
    assert all(path.is_file() for path in copied_paths)

    copied_wal = copied_root / authority["prior_coordination_wal_id"]
    copied_wal.write_bytes(copied_wal.read_bytes() + b"tamper")
    with pytest.raises(
        materializer.MigrationRequired,
        match="control/coordination/WAL bytes differ",
    ):
        materializer._assert_m11_prior_publication_anchor(
            copied_root,
            authority,
            population,
        )


def _build_m11_rehearsal_pair(
    materializer: ModuleType,
    *,
    control: Path,
    coordination: Path,
    prior_control: Path,
    prior_coordination: Path,
    population: dict[str, object],
    config: dict[str, object],
    validation_digest: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build the exact M11 append and rearm only on disposable stores."""

    import duckdb
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinator,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources import intent_repository
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    monkeypatch.setattr(
        intent_repository,
        "_utc_iso",
        lambda _moment=None: materializer._M11_CONTROL_RECORDED_AT,
    )
    control.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prior_control, control)
    shutil.copyfile(prior_coordination, coordination)
    prior_wal = prior_coordination.with_name(prior_coordination.name + ".wal")
    target_wal = coordination.with_name(coordination.name + ".wal")
    shutil.copyfile(prior_wal, target_wal)

    source = DatabaseTaskSource(
        control,
        install_schema=False,
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        owner_id="sawm-r2-provider-retry-migrator",
    )
    try:
        operator = source.get_task("SAWM-000")
        assert operator is not None
        body = materializer._m11_migration_body(
            population,
            config,
            validation_digest,
        )
        digest = materializer._identity(body)
        source.plans.append_revision(
            plan_cid=str(population["plan_root_cid"]),
            expected_revision=11,
            body={
                "current_source_binding_cid": population["source_binding"][
                    "source_binding_cid"
                ],
                "source_migration_revision": materializer._M11_MIGRATION_REVISION,
                "source_migration_digest": digest,
                "supersession_mode": materializer._M11_SUPERSESSION_MODE,
            },
            delta=materializer._m11_migration_plan_delta(population, config),
        )
        source.record_evidence(
            task_cid=operator.task_cid,
            evidence_kind="operator_control_plane_source_migration",
            digest=digest,
            body=body,
        )
        task_cas = source.compare_and_set_status(
            materializer._M10_LIVE_FAILURE_RECEIPT["task_cid"],
            12,
            "retrying",
            materializer._m11_task_rearm_receipt(),
        )
    finally:
        source.close()

    coordinator = DatabaseCoordinator(coordination).open()
    try:
        rearm = coordinator.rearm_failed_task(
            failure_receipt=materializer._M10_LIVE_FAILURE_RECEIPT,
            control_task_observation=task_cas.to_dict(),
            now_ms=materializer._M11_COORDINATION_REARM_OBSERVED_AT_MS,
        )
    finally:
        coordinator.close()
    assert rearm["ready"] is True
    assert rearm["replayed"] is False

    # Seal the disposable target's replayed coordination state into its base;
    # the frozen predecessor base+WAL remain byte-for-byte untouched.
    checkpoint = duckdb.connect(str(coordination))
    try:
        checkpoint.execute("CHECKPOINT")
    finally:
        checkpoint.close()
    assert not os.path.lexists(target_wal)


def test_m11_pair_receipt_last_rehearsal_is_idempotent_and_tamper_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m11_pair_receipt_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._m11_live_provider_retry_authority(
        population,
        config,
    )
    validation_digest = "sha256:m11-disposable-pair-receipt"

    # Recreate the exact predecessor publication layout under a disposable
    # root so every anchor and receipt path is exercised without a live write.
    for field in (
        "prior_store_id",
        "prior_coordination_store_id",
        "prior_coordination_wal_id",
        "prior_materialization_receipt_path",
        "prior_owner_status_path",
    ):
        source_path = REPO_ROOT / authority[field]
        copied_path = tmp_path / authority[field]
        copied_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, copied_path)
    prior_control = tmp_path / authority["prior_store_id"]
    prior_coordination = tmp_path / authority["prior_coordination_store_id"]
    control = tmp_path / authority["target_store_id"]
    coordination = tmp_path / authority["target_coordination_store_id"]
    original_paths = tuple(
        REPO_ROOT / authority[field]
        for field in (
            "prior_store_id",
            "prior_coordination_store_id",
            "prior_coordination_wal_id",
            "prior_materialization_receipt_path",
            "prior_owner_status_path",
        )
    )
    original_hashes = {
        path: materializer._store_sha256(path) for path in original_paths
    }

    _build_m11_rehearsal_pair(
        materializer,
        control=control,
        coordination=coordination,
        prior_control=prior_control,
        prior_coordination=prior_coordination,
        population=population,
        config=config,
        validation_digest=validation_digest,
        monkeypatch=monkeypatch,
    )
    target_hashes = (
        materializer._store_sha256(control),
        materializer._store_sha256(coordination),
    )
    verified = materializer._verify_m11_store_pair(
        tmp_path,
        control,
        coordination,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    assert verified["projection_cid"] == materializer._M11_EXPECTED_PROJECTION_CID
    assert verified["event_watermark"] == 184
    assert verified["coordination_event_count"] == 56
    assert verified["task_revision_changes"] == 1
    assert verified["task_status_changes"] == 1
    assert target_hashes == (
        materializer._store_sha256(control),
        materializer._store_sha256(coordination),
    )

    assert materializer._verify_m11_store_pair(
        tmp_path,
        control,
        coordination,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    ) == verified
    anchored = materializer._assert_m11_prior_publication_anchor(
        tmp_path,
        authority,
        population,
    )

    # The pair and predecessor were each checked above through their real
    # disposable replay paths. Receipt-only fault cases should exercise the
    # linearization protocol without repeating that expensive replay.
    monkeypatch.setattr(
        materializer,
        "_verify_m11_store_pair",
        lambda *_args, **_kwargs: dict(verified),
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m11_prior_publication_anchor",
        lambda *_args, **_kwargs: anchored,
    )

    # Exercise the exact committed source-delta closure independently of the
    # dirty test checkout, then retain that fail-closed fake for receipt work.
    changed_paths = list(authority["bounded_control_plane_repair_paths"])

    def exact_git(_root: Path, *args: str, **_kwargs: object) -> str:
        if args[:2] == ("merge-base", "--is-ancestor"):
            return ""
        if args and args[0] == "diff":
            return "\n".join(f"M\t{path}" for path in changed_paths)
        raise AssertionError(f"unexpected M11 source-delta git call: {args}")

    monkeypatch.setattr(materializer, "_git", exact_git)
    delta_population = copy.deepcopy(population)
    delta_population["source_binding"]["head"] = "1" * 40
    materializer._assert_m11_source_delta(
        tmp_path,
        delta_population,
        authority,
    )
    changed_paths.append("unexpected.py")
    with pytest.raises(
        materializer.MaterializationError,
        match="exact repair paths",
    ):
        materializer._assert_m11_source_delta(
            tmp_path,
            delta_population,
            authority,
        )
    changed_paths.pop()
    monkeypatch.setattr(
        materializer,
        "_assert_m11_source_delta",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_committed_clean_source",
        lambda *_args, **_kwargs: None,
    )

    receipt_path = control.parent / "migration-receipt.json"
    expected = materializer._expected_m11_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    )
    malformed_pending = control.parent / ".migration-receipt.json.bad.tmp"
    malformed_pending.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        materializer.MigrationRequired,
        match="pending M11 receipt temporary differs",
    ):
        materializer._ensure_m11_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    assert not receipt_path.exists()
    malformed_pending.unlink()

    # An exact orphan from a crash before hardlink publication is recovered;
    # an ambiguous or unrelated hardlink is never accepted.
    pending = control.parent / ".migration-receipt.json.999999.tmp"
    pending.write_bytes(materializer._canonical(expected) + b"\n")
    receipt = materializer._ensure_m11_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    )
    assert receipt == expected
    assert receipt["schema"] == "sawm/non-authoritative-migration-receipt@9"
    assert receipt["receipt_is_final_pair_commit_marker"] is True
    assert receipt_path.stat().st_nlink == 1
    assert not pending.exists()
    assert not list(control.parent.glob(".migration-receipt.json.*.tmp"))
    assert target_hashes == (
        materializer._store_sha256(control),
        materializer._store_sha256(coordination),
    )

    assert materializer._ensure_m11_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    ) == receipt
    assert materializer._verify_existing_m11_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    ) == receipt

    crash_alias = control.parent / ".migration-receipt.json.123.tmp"
    os.link(receipt_path, crash_alias)
    assert receipt_path.stat().st_nlink == 2
    assert materializer._ensure_m11_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    ) == receipt
    assert receipt_path.stat().st_nlink == 1
    assert not crash_alias.exists()

    unrelated_alias = control.parent / "unrelated-receipt-hardlink.json"
    os.link(receipt_path, unrelated_alias)
    with pytest.raises(
        materializer.MigrationRequired,
        match="ambiguous pending hardlink",
    ):
        materializer._ensure_m11_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    unrelated_alias.unlink()

    execution_sidecar = control.with_name("control.execution.duckdb")
    execution_sidecar.write_bytes(b"forbidden")
    with pytest.raises(
        materializer.MigrationRequired,
        match="mutable sidecar appeared before final receipt",
    ):
        materializer._ensure_m11_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    execution_sidecar.unlink()

    pristine_receipt = receipt_path.read_bytes()
    tampered = json.loads(pristine_receipt)
    tampered["worker_self_approval"] = True
    receipt_path.write_bytes(materializer._canonical(tampered) + b"\n")
    with pytest.raises(
        materializer.MigrationRequired,
        match="final pair marker differs",
    ):
        materializer._verify_existing_m11_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    receipt_path.write_bytes(pristine_receipt)
    assert original_hashes == {
        path: materializer._store_sha256(path) for path in original_paths
    }
    assert target_hashes == (
        materializer._store_sha256(control),
        materializer._store_sha256(coordination),
    )


class _M39OneRowResult:
    def __init__(self, row: object | None) -> None:
        self._row = row

    def fetchone(self) -> object | None:
        return self._row


class _M39CommittedEventConnection:
    def __init__(self, evidence: object, event: object) -> None:
        self.evidence = evidence
        self.event = event

    def execute(self, query: str, _parameters: object = None) -> _M39OneRowResult:
        del _parameters
        if "FROM evidence_nodes" in query:
            return _M39OneRowResult(self.evidence)
        if "FROM domain_events" in query:
            return _M39OneRowResult(self.event)
        raise AssertionError(f"unexpected query: {query}")


def test_m39_reconstructs_the_exact_committed_m38_event_not_current_controls() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m39_historical_event_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    historical = materializer._m39_historical_m38_authority()
    body, evidence, event = materializer._m39_expected_historical_m38_rows(config)

    assert materializer._identity(historical) == materializer._M39_M38_AUTHORITY_CID
    assert materializer._identity(body) == materializer._M39_M38_EVIDENCE_DIGEST
    assert body["current_source_head"] == materializer._M39_M38_SOURCE_HEAD
    assert body["current_source_tree"] == materializer._M39_M38_SOURCE_TREE
    assert body["authorization_cid"] == materializer._M39_M38_AUTHORITY_CID
    assert evidence[0] == materializer._M39_M38_EVIDENCE_ID
    assert event[0] == materializer._M39_M38_EVENT_ID
    assert materializer._identity(
        materializer._expected_m38_pre_authoritative_custody_restart_authority()
    ) != materializer._M39_M38_AUTHORITY_CID


def test_m39_authority_binds_c1_repair_lineage_and_closed_deltas() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m39_authority_test",
    )
    authority = (
        materializer._expected_m39_committed_m38_evidence_reconciliation_authority()
    )
    reference = materializer._m39_authority_reference()
    chain = authority["source_chain"]

    assert authority["schema"] == (
        "sawm/committed-m38-evidence-reconciliation-successor-authorization@1"
    )
    assert reference == {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M39",
        "authority_cid": materializer._identity(authority),
    }
    assert chain["base_control_commit"] == materializer._M39_BASE_CONTROL_COMMIT
    assert chain["json_comparison_repair_commit"] == (
        "146653af91fe3846cb98e49a54ae1173e3a3dc66"
    )
    assert chain["canonical_envelope_repair_commit"] == (
        "32d2966c4944157d664748c536cfa167f7ae38f5"
    )
    assert chain["materializer_repair_commit"] == (
        "b581305f42ad4eda6b3d749680e79107c1c150b3"
    )
    assert chain["final_control_parent"] == chain["materializer_repair_commit"]
    assert chain["bounded_repair_commit_count"] == 3
    assert authority["exact_changes"]["evidence_node_changes"] == 1
    assert authority["exact_changes"]["evidence_event_changes"] == 1
    assert authority["exact_changes"]["validation_event_changes"] == 0
    assert authority["exact_changes"]["accepted_completion_changes"] == 0


def test_m39_verifies_full_historical_row_and_closed_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m39_committed_event_verifier_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    _body, evidence, event = materializer._m39_expected_historical_m38_rows(config)
    projection = {
        "evidence_node_count": 48,
        "evidence_event_count": 37,
        "validation_event_count": 11,
        "passed_validation_event_count": 11,
        "validation_evidence_node_count": 11,
    }
    monkeypatch.setattr(
        materializer,
        "_m38_evidence_projection_from_events",
        lambda *_args, **_kwargs: projection,
    )
    result = materializer._verify_m39_committed_m38_event(
        _M39CommittedEventConnection(evidence, event), config
    )
    assert result["prior_m38_event_id"] == materializer._M39_M38_EVENT_ID
    assert result["prior_m38_evidence_id"] == materializer._M39_M38_EVIDENCE_ID
    assert result["prior_evidence_node_count"] == 48
    assert result["prior_evidence_event_count"] == 37
    assert result["prior_validation_event_count"] == 11

    tampered_event = tuple(event[:9]) + ("{}",)
    with pytest.raises(
        materializer.MigrationRequired,
        match="committed M38 event/evidence rows differ",
    ):
        materializer._verify_m39_committed_m38_event(
            _M39CommittedEventConnection(evidence, tampered_event), config
        )


@pytest.mark.parametrize("decoded_side", ["evidence", "event"])
def test_m39_normalizes_named_quack_rows_and_json_representations(
    monkeypatch: pytest.MonkeyPatch,
    decoded_side: str,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        f"sawm_materializer_m39_named_rows_{decoded_side}_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    _body, evidence, event = materializer._m39_expected_historical_m38_rows(config)
    evidence_columns = (
        "evidence_id",
        "parent_evidence_id",
        "task_cid",
        "evidence_kind",
        "digest",
        "created_at",
        "body_json",
    )
    event_columns = (
        "event_id",
        "stream_id",
        "sequence",
        "global_sequence",
        "event_type",
        "task_cid",
        "attempt_id",
        "session_id",
        "recorded_at",
        "body_json",
    )
    evidence_row = dict(reversed(tuple(zip(evidence_columns, evidence))))
    event_row = dict(reversed(tuple(zip(event_columns, event))))
    if decoded_side == "evidence":
        evidence_row["body_json"] = json.loads(evidence[-1])
        event_row["body_json"] = event[-1].encode("utf-8")
    else:
        evidence_row["body_json"] = evidence[-1].encode("utf-8")
        event_row["body_json"] = json.loads(event[-1])
    monkeypatch.setattr(
        materializer,
        "_m38_evidence_projection_from_events",
        lambda *_args, **_kwargs: {
            "evidence_node_count": 48,
            "evidence_event_count": 37,
            "validation_event_count": 11,
            "passed_validation_event_count": 11,
            "validation_evidence_node_count": 11,
        },
    )

    result = materializer._verify_m39_committed_m38_event(
        _M39CommittedEventConnection(evidence_row, event_row), config
    )
    assert result["committed_m38_event_and_evidence_verified"] is True

    evidence_row.pop("digest")
    with pytest.raises(
        materializer.MigrationRequired,
        match="committed M38 event/evidence rows differ",
    ):
        materializer._verify_m39_committed_m38_event(
            _M39CommittedEventConnection(evidence_row, event_row), config
        )


def test_m39_receipt_publication_is_last_idempotent_and_nofollow(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m39_receipt_test",
    )
    control = tmp_path / "control.duckdb"
    control.write_bytes(b"not-opened")
    expected = {"schema": "test/m39-receipt@1", "receipt_cid": "sha256:test"}

    first = materializer._ensure_m39_source_successor_receipt(
        tmp_path, control, expected
    )
    second = materializer._ensure_m39_source_successor_receipt(
        tmp_path, control, expected
    )
    assert first == second == expected
    assert json.loads(
        (tmp_path / "m39-source-successor-receipt.json").read_text(
            encoding="utf-8"
        )
    ) == expected
    assert stat.S_IMODE(
        (tmp_path / "m39-source-successor-receipt.json").stat().st_mode
    ) == 0o600
    for revision in ("m37", "m38", "m39"):
        lock_path = tmp_path / f".{revision}-source-successor-receipt.publish.lock"
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600

    (tmp_path / "m39-source-successor-receipt.json").chmod(0o640)
    with pytest.raises(materializer.MigrationRequired, match="mode is unsafe"):
        materializer._ensure_m39_source_successor_receipt(
            tmp_path, control, expected
        )

    (tmp_path / "m39-source-successor-receipt.json").unlink()
    (tmp_path / "m39-source-successor-receipt.json").symlink_to(control)
    with pytest.raises(materializer.MigrationRequired):
        materializer._ensure_m39_source_successor_receipt(
            tmp_path, control, expected
        )


def test_m39_dispatch_precedes_m38_and_requires_receipt_last() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m39_dispatch_test",
    )
    check_source = inspect.getsource(materializer.check_materialized)
    materialize_source = inspect.getsource(materializer.materialize)
    m39 = "_m39_successor_configured_on_any_surface"
    m38 = "_m38_successor_configured_on_any_surface"

    assert check_source.index(m39) < check_source.index(m38)
    assert materialize_source.index(m39) < materialize_source.index(m38)
    core = inspect.getsource(materializer._materialize_m39)
    assert core.index("_verify_m39_live_materialization") < core.index(
        "_ensure_m39_source_successor_receipt"
    )
    assert "event_cursor == _M39_PRIOR_EVENT_WATERMARK" in core
    assert "event_cursor != _M39_TARGET_EVENT_WATERMARK" in core


def test_m48_current_stale_owner_recovery_requires_generation_35_successor() -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m39_stale_recovery_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )

    with pytest.raises(
        operator.OperatorError,
        match=(
            "M48 binds a cleanly stopped generation-34 owner; use the sealed "
            "generation-35 quack-start path instead of stale-owner recovery"
        ),
    ):
        operator._recover_stale_quack(config)


def test_m39_live_verifier_requires_exact_plus_one_evidence_only_delta() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m39_delta_test",
    )
    source = inspect.getsource(materializer._verify_m39_live_materialization)

    assert "_verify_m39_committed_m38_event" in source
    assert source.count("_m39_exact_row_matches") == 2
    assert "_verify_m38_evidence_projection" in source
    assert "_M39_TARGET_EVIDENCE_NODE_COUNT" in source
    assert "_M39_TARGET_EVIDENCE_EVENT_COUNT" in source
    assert 'prior["prior_evidence_node_count"] + 1' in source
    assert 'prior["prior_evidence_event_count"] + 1' in source
    assert 'prior["prior_validation_event_count"]' in source
    assert 'prior["prior_passed_validation_event_count"]' in source
    assert "_M39_LIVE_SERVER_ID" in source
    assert "_M39_LIVE_PROCESS_BIRTH_ID" in source


def test_m39_restart_rows_use_sealed_historical_stopped_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m39_restart_authority_test",
    )
    authority = (
        materializer._expected_m39_committed_m38_evidence_reconciliation_authority()
    )
    observed: list[object] = []

    def inspect_restart(
        source: object, identity: object, historical: object
    ) -> dict[str, bool]:
        observed.extend((source, identity, historical))
        return {"generation_29_30_restart_rows_verified": True}

    monkeypatch.setattr(
        materializer, "_inspect_m37_generation_restart_rows", inspect_restart
    )
    source = object()
    identity = {"server_id": materializer._M39_LIVE_SERVER_ID}

    assert materializer._inspect_m39_generation_restart_rows(
        source, identity, authority
    ) == {"generation_29_30_restart_rows_verified": True}
    historical = authority["historical_m38_authority"]
    assert observed == [source, identity, historical]
    assert historical["stopped_owner"]["generation"] == 29
    assert materializer._identity(historical) == materializer._M39_M38_AUTHORITY_CID
    for m39_path in (
        materializer._verify_m39_live_materialization,
        materializer._materialize_m39,
    ):
        source_text = inspect.getsource(m39_path)
        assert "_inspect_m39_generation_restart_rows" in source_text
        assert "_inspect_m37_generation_restart_rows" not in source_text

    malformed = dict(authority)
    malformed.pop("historical_m38_authority")
    with pytest.raises(
        materializer.MigrationRequired,
        match="historical M38 generation-restart authority differs",
    ):
        materializer._inspect_m39_generation_restart_rows(
            source, identity, malformed
        )


def test_m40_authority_seals_failed_m39_and_exact_helper_repair() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m40_authority_test",
    )
    authority = (
        materializer
        ._expected_m40_failed_pre_authoritative_m39_successor_authority()
    )
    reference = materializer._m40_authority_reference()
    failed = authority["failed_m39_pre_authoritative_attempt"]
    repair = authority["accepted_restart_helper_repair"]
    chain = authority["source_chain"]

    assert reference == {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M40",
        "authority_cid": materializer._M40_AUTHORITY_CID,
    }
    assert materializer._identity(authority) == (
        "sha256:377b6e7269a7025f236642f12aaf92264582f42a1efa4899eb4d6e64b0e41db2"
    )
    assert materializer._identity(authority["prior_m39_authority"]) == (
        materializer._M40_M39_AUTHORITY_CID
    )
    assert failed["error"] == "KeyError: 'stopped_owner'"
    assert failed["quack_mutation_request_created"] is False
    assert failed["event_292_rows_created"] == 0
    assert failed["m39_receipt_created"] is False
    assert failed["task_revision_changes"] == 0
    assert failed["accepted_completion_changes"] == 0
    assert repair["repair_parent"] == materializer._M40_M39_FINAL_CONTROL_COMMIT
    assert repair["repair_commit"] == materializer._M40_RESTART_HELPER_REPAIR_COMMIT
    assert repair["blob_oids"] == dict(
        materializer._M40_RESTART_HELPER_REPAIR_BLOBS
    )
    assert repair["routes_through_sealed_historical_m38_authority"] is True
    assert chain["final_control_parent"] == (
        materializer._M40_RESTART_HELPER_REPAIR_COMMIT
    )
    assert chain["final_control_commit_is_current_head"] is True
    assert chain["final_control_commit_count"] == 1
    assert authority["target_event_watermark"] == 292
    assert authority["target_projection_cid"] == materializer._M39_TARGET_PROJECTION_CID
    assert authority["target_authority"]["evidence_kind"] == (
        "operator_control_plane_failed_pre_authoritative_m39_successor"
    )
    materializer._validated_m40_live_preflight_contract(authority)


def test_m40_restart_rows_route_through_repaired_m39_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m40_restart_authority_test",
    )
    authority = (
        materializer
        ._expected_m40_failed_pre_authoritative_m39_successor_authority()
    )
    observed: list[object] = []

    def inspect_restart(
        source: object, identity: object, prior_m39: object
    ) -> dict[str, bool]:
        observed.extend((source, identity, prior_m39))
        return {"generation_29_30_restart_rows_verified": True}

    monkeypatch.setattr(
        materializer, "_inspect_m39_generation_restart_rows", inspect_restart
    )
    source = object()
    identity = {"server_id": materializer._M40_LIVE_SERVER_ID}
    assert materializer._inspect_m40_generation_restart_rows(
        source, identity, authority
    ) == {"generation_29_30_restart_rows_verified": True}
    assert observed == [source, identity, authority["prior_m39_authority"]]

    malformed = dict(authority)
    malformed["prior_m39_authority"] = {}
    with pytest.raises(
        materializer.MigrationRequired,
        match="historical M39 restart authority differs",
    ):
        materializer._inspect_m40_generation_restart_rows(
            source, identity, malformed
        )


def test_m40_receipt_publication_is_last_idempotent_and_exclusive(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m40_receipt_test",
    )
    control = tmp_path / "control.duckdb"
    control.write_bytes(b"test")
    expected = {"schema": "test/m40-receipt@1", "receipt_cid": "sha256:test"}

    first = materializer._ensure_m40_source_successor_receipt(
        tmp_path, control, expected
    )
    second = materializer._ensure_m40_source_successor_receipt(
        tmp_path, control, expected
    )
    assert first == second == expected
    path = tmp_path / "m40-source-successor-receipt.json"
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for revision in ("m37", "m38", "m39", "m40"):
        assert (
            tmp_path / f".{revision}-source-successor-receipt.publish.lock"
        ).exists()

    path.unlink()
    (tmp_path / "m39-source-successor-receipt.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="M39 receipt unexpectedly exists",
    ):
        materializer._ensure_m40_source_successor_receipt(
            tmp_path, control, expected
        )


def test_m40_dispatch_precedes_m39_and_receipt_follows_live_verification() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m40_dispatch_test",
    )
    check_source = inspect.getsource(materializer.check_materialized)
    materialize_source = inspect.getsource(materializer.materialize)
    m40 = "_m40_successor_configured_on_any_surface"
    m39 = "_m39_successor_configured_on_any_surface"
    assert check_source.index(m40) < check_source.index(m39)
    assert materialize_source.index(m40) < materialize_source.index(m39)
    main_source = inspect.getsource(materializer.main)
    assert main_source.index(
        "failed_pre_authoritative_m39_successor_materialization"
    ) < main_source.index(
        "committed_m38_evidence_reconciliation_successor_materialization"
    )
    core = inspect.getsource(materializer._materialize_m40)
    assert core.index("_verify_m40_live_materialization") < core.index(
        "_ensure_m40_source_successor_receipt"
    )
    assert "snapshot.event_cursor == _M40_PRIOR_EVENT_WATERMARK" in core
    assert "snapshot.event_cursor != _M40_TARGET_EVENT_WATERMARK" in core
    assert "target_event_exists is not None" in core
    assert "m39-source-successor-receipt.json" not in core.split(
        "for revision", 1
    )[0]


def test_m40_operator_selects_newest_authority_by_key_presence() -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m40_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    historical_config, _historical_migration, _historical_seal = (
        _historical_successor_controls_at(
            "failed_pre_authoritative_m39_successor_materialization",
            config,
        )
    )
    historical_config["database_program"]["store_generation"] = "30"
    active = operator._active_source_repair_materialization(historical_config)
    assert active["migration_revision"] == "SAWM-R2-M40"
    assert active["target_event_watermark"] == 292
    assert active["failed_m39_pre_authoritative_attempt"][
        "quack_mutation_request_created"
    ] is False
    configured = inspect.getsource(operator._successor_materialization_configured)
    assert configured.index("_M40_SUCCESSOR_KEY") < configured.index(
        "_M39_SUCCESSOR_KEY"
    )
    normalized = inspect.getsource(operator._normalized_live_preflight_contract)
    assert normalized.index("_M40_MIGRATION_REVISION") < normalized.index(
        '"SAWM-R2-M39"'
    )
    preflight = inspect.getsource(operator._live_preflight)
    assert "m40_active = active_revision == _M40_MIGRATION_REVISION" in preflight
    assert preflight.index("m40_active,") < preflight.index("m39_active,")
    assert preflight.index("_verify_m40_live_materialization") < preflight.index(
        "_verify_m39_live_materialization"
    )
    assert preflight.index("_expected_m40_source_successor_receipt") < (
        preflight.index("_expected_m39_source_successor_receipt")
    )
    assert preflight.index("_M40_SUCCESSOR_KEY in config") < preflight.index(
        "_M39_SUCCESSOR_KEY in config"
    )
    plan_binding_gate = (
        'live_plan_body.get("current_source_binding_cid")\n'
        '                    != preserved_plan_anchor["plan_source_binding_cid"]'
    )
    assert preflight.count(plan_binding_gate) == 1
    assert (
        'preserved_plan_anchor["plan_source_binding_cid"]\n'
        '                    != preserved_plan_anchor["plan_source_binding_cid"]'
    ) not in preflight
    marker = inspect.getsource(operator._require_active_final_pair_marker)
    assert marker.index("_M40_SUCCESSOR_KEY") < marker.index("_M39_SUCCESSOR_KEY")


def test_m42_authority_seals_failed_m41_projection_and_exact_repair() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m42_authority_test",
    )
    authority = (
        materializer
        ._expected_m42_failed_pre_authoritative_m41_evidence_projection_successor_authority()
    )
    reference = materializer._m42_authority_reference()
    failed = authority["failed_m41_pre_authoritative_materialization"]
    legacy = authority["exact_legacy_projection_authority"]
    repair = authority["accepted_projection_repair"]
    hardening = authority["final_control_hardening"]

    assert reference == {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M42",
        "authority_cid": materializer._M42_AUTHORITY_CID,
    }
    assert materializer._identity(authority) == materializer._M42_AUTHORITY_CID
    assert materializer._identity(authority["prior_m41_authority"]) == (
        materializer._M42_M41_AUTHORITY_CID
    )
    assert failed["materializer_invoked"] is True
    assert failed["authenticated_live_quack_read_opened"] is True
    assert failed["quack_mutation_request_created"] is False
    assert failed["record_evidence_reached"] is False
    assert failed["event_watermark_before"] == failed["event_watermark_after"] == 291
    assert failed["event_292_rows_created"] == 0
    assert failed["m41_receipt_created"] is False
    assert legacy["manifest_cid"] == (
        materializer._M42_LEGACY_PROJECTION_MANIFEST_CID
    )
    assert legacy["evidence_node_count"] == 48
    assert legacy["evidence_event_count"] == 37
    assert legacy["validation_event_count"] == 11
    assert legacy["evidence_refresh_overlay_count"] == 9
    assert legacy["compact_validation_evidence_overlay_count"] == 1
    assert legacy["validation_attempt_overlay_count"] == 1
    assert repair["repair_parent"] == materializer._M42_M41_FINAL_CONTROL_COMMIT
    assert repair["repair_commit"] == materializer._M42_PROJECTION_REPAIR_COMMIT
    assert repair["blob_oids"] == dict(materializer._M42_PROJECTION_REPAIR_BLOBS)
    assert hardening["helper_does_not_establish_target_body_authority"] is True
    assert hardening["production_db_derived_expected_row_forbidden"] is True
    assert hardening["target_event_session_id"] == "session:intent"
    assert authority["target_event_watermark"] == 292
    assert authority["target_projection_derivation"][
        "event_body_bound_by_event_prefix_not_projection"
    ] is True
    materializer._validated_m42_live_preflight_contract(authority)


def test_m42_restart_rows_route_through_exact_m41_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m42_restart_authority_test",
    )
    authority = (
        materializer
        ._expected_m42_failed_pre_authoritative_m41_evidence_projection_successor_authority()
    )
    observed: list[object] = []

    def inspect_restart(
        source: object, identity: object, prior_m41: object
    ) -> dict[str, bool]:
        observed.extend((source, identity, prior_m41))
        return {"generation_29_30_restart_rows_verified": True}

    monkeypatch.setattr(
        materializer, "_inspect_m41_generation_restart_rows", inspect_restart
    )
    source = object()
    identity = {"server_id": materializer._M42_LIVE_SERVER_ID}
    assert materializer._inspect_m42_generation_restart_rows(
        source, identity, authority
    ) == {"generation_29_30_restart_rows_verified": True}
    assert observed == [source, identity, authority["prior_m41_authority"]]

    malformed = dict(authority)
    malformed["prior_m41_authority"] = {}
    with pytest.raises(
        materializer.MigrationRequired,
        match="historical M41 restart authority differs",
    ):
        materializer._inspect_m42_generation_restart_rows(
            source, identity, malformed
        )


def test_m42_receipt_publication_is_last_idempotent_and_exclusive(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m42_receipt_test",
    )
    control = tmp_path / "control.duckdb"
    control.write_bytes(b"test")
    expected = {"schema": "test/m42-receipt@1", "receipt_cid": "sha256:test"}

    first = materializer._ensure_m42_source_successor_receipt(
        tmp_path, control, expected
    )
    second = materializer._ensure_m42_source_successor_receipt(
        tmp_path, control, expected
    )
    assert first == second == expected
    path = tmp_path / "m42-source-successor-receipt.json"
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for revision in ("m37", "m38", "m39", "m40", "m41", "m42"):
        assert (
            tmp_path / f".{revision}-source-successor-receipt.publish.lock"
        ).exists()

    path.unlink()
    (tmp_path / "m41-source-successor-receipt.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="M41 receipt unexpectedly exists",
    ):
        materializer._ensure_m42_source_successor_receipt(
            tmp_path, control, expected
        )


def test_m48_authority_binds_m47_final_control_and_generation_35() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m48_authority_test",
    )
    authority = (
        materializer
        ._expected_m48_post_m47_clean_shutdown_restart_successor_authority()
    )
    contract = materializer._validated_m48_live_preflight_contract(authority)
    reference = materializer._m48_authority_reference()

    assert materializer._M48_AUTHORITY_CID == (
        "sha256:a7b262b976b38eb94646a9d73c595ff49ab9b1f89fa528443745ed964be025f6"
    )
    assert materializer._identity(authority) == materializer._M48_AUTHORITY_CID
    assert len(materializer._canonical(authority)) == 17_315
    assert reference == {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M48",
        "authority_cid": materializer._M48_AUTHORITY_CID,
    }
    assert authority["stopped_owner"]["generation"] == 34
    assert authority["stopped_owner"]["target_generation"] == 35
    assert authority["target_event_watermark"] == 305
    assert authority["target_projection_cid"] == (
        "baguqeerar773puawanjlnneg2heko27svsonv5dfg5pioayyqemfu73ljpwa"
    )
    assert contract["prior_event_watermark"] == 304
    assert contract["target_event_watermark"] == 305
    assert contract["generation_restart_authorized"] is True

    receipt = authority["preserved_m47_receipt"]
    assert receipt == {
        "schema": "sawm/preserved-source-successor-receipt@1",
        "path": (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27/"
            "m47-source-successor-receipt.json"
        ),
        "sha256": (
            "e8ef92a302f38d37d5704d67d1beb9575091c16c2ee61d6f680fd34aa4cfe731"
        ),
        "size": 8_088,
        "mode": "0600",
        "receipt_cid": (
            "sha256:52d8954b5c31a07deafd6077cc9786a3de170017660bca0746dffae08c128919"
        ),
        "created_or_rewritten": False,
    }
    preserved = authority["preserved_m47_materialization"]
    assert preserved["event_watermark"] == 304
    assert preserved["event_id"] == (
        "baguqeeranmbjd63u4wungeeox327cskqjtyypsavnwjwdlh3pqj3zbmtkeua"
    )
    assert preserved["evidence_id"] == (
        "baguqeeraqy7idops72utponoumzonzd3hbo4f4tqmg74of223pcbvb2mouua"
    )
    assert preserved["migration_digest"] == (
        "sha256:bceaad94e642ea09ba9ef1d4567b809db1ef6e5f371b3ff5b86273a1b3efa1b8"
    )
    assert preserved["source_binding_cid"] == (
        "sha256:2f87cf1b9620144aca641b49acbbe39d12a19917318bd351e5210c81e206dae1"
    )
    chain = authority["source_chain"]
    assert chain["m47_final_control_commit"] == (
        "ffec7b3c57cd75843dbedfb260263a98c4104d36"
    )
    assert chain["m47_final_control_parent"] == (
        "7fe0f09615c6d07ba72d3f6334b6bb4f6a512141"
    )
    assert chain["m47_final_control_tree"] == (
        "100f37e1da44aa5aec079c922b962d1ac9fdc3db"
    )
    assert chain["m47_final_control_blobs"] == dict(
        materializer._M48_M47_FINAL_CONTROL_BLOBS
    )
    assert len(chain["m47_final_control_blobs"]) == 9
    assert chain["m47_final_control_modes"] == dict(
        materializer._M48_M47_FINAL_CONTROL_MODES
    )
    assert chain["final_control_parent"] == chain["m47_final_control_commit"]
    assert chain["ordinary_repair_commit_count"] == 0
    assert chain["final_control_commit_count"] == 1
    assert authority["ordinary_source_changes"] == 0

    shutdown = authority["failed_test_induced_shutdown_observation"]
    assert shutdown["test_suite_completed"] is False
    assert shutdown["test_suite_exit_code"] == 2
    assert shutdown["passed_tests_before_interruption"] == 21
    assert shutdown["operational_history_only"] is True
    assert shutdown["validation_authority"] is False
    assert shutdown["task_completion_authority"] is False
    assert shutdown["worker_self_approval"] is False


def test_m48_preflight_and_surface_declarations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m48_tamper_test",
    )
    authority = (
        materializer
        ._expected_m48_post_m47_clean_shutdown_restart_successor_authority()
    )
    for section, field, value in (
        ("preserved_m47_receipt", "receipt_cid", "sha256:" + "0" * 64),
        ("preserved_m47_materialization", "event_id", "bad-event"),
        ("failed_test_induced_shutdown_observation", "validation_authority", True),
        ("live_preflight_contract", "target_event_watermark", 304),
        ("source_chain", "ordinary_repair_commit_count", 1),
        ("exact_changes", "task_status_changes", 1),
    ):
        changed = copy.deepcopy(authority)
        changed[section][field] = value
        with pytest.raises(
            materializer.MaterializationError,
            match="M48 live preflight contract differs",
        ):
            materializer._validated_m48_live_preflight_contract(changed)

    key = materializer._M48_SUPERSESSION_REASON
    with monkeypatch.context() as scoped:
        controls = iter(({}, {}))
        scoped.setattr(materializer, "_load_json", lambda _path: next(controls))
        with pytest.raises(
            materializer.MaterializationError,
            match="M48 successor authority is only partially declared",
        ):
            materializer._m48_successor_configured_on_any_surface(
                REPO_ROOT, {key: materializer._m48_authority_reference()}
            )

    with monkeypatch.context() as scoped:
        bad = {key: {"migration_revision": "tampered"}}
        controls = iter((bad, {f"{key}_cid": materializer._M48_AUTHORITY_CID}))
        scoped.setattr(materializer, "_load_json", lambda _path: next(controls))
        with pytest.raises(
            materializer.MaterializationError,
            match="M48 successor authority differs across controls",
        ):
            materializer._m48_successor_configured_on_any_surface(
                REPO_ROOT, {key: materializer._m48_authority_reference()}
            )


def test_m48_source_delta_requires_direct_nine_control_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m48_source_delta_test",
    )
    authority = (
        materializer
        ._expected_m48_post_m47_clean_shutdown_restart_successor_authority()
    )
    current = "f" * 40
    current_tree = "e" * 40
    population = {
        "source_binding": {
            "head": current,
            "tree": current_tree,
            "datasets_gitlink": authority["current_datasets_gitlink"],
            "kit_gitlink": authority["current_kit_gitlink"],
        }
    }

    def exact_git(root: Path, *args: str) -> str:
        if args[:4] == ("rev-list", "--parents", "-n", "1"):
            commit = args[4]
            parent = (
                materializer._M48_M47_FINAL_CONTROL_PARENT
                if commit == materializer._M48_M47_FINAL_CONTROL_COMMIT
                else materializer._M48_M47_FINAL_CONTROL_COMMIT
            )
            return f"{commit} {parent}"
        if args[0] == "rev-parse":
            expression = args[1]
            if expression == f"{materializer._M48_M47_FINAL_CONTROL_COMMIT}^{{tree}}":
                return materializer._M48_M47_FINAL_CONTROL_TREE
            if expression == f"{current}^{{tree}}":
                return current_tree
            if expression.endswith("^{tree}"):
                return (
                    authority["current_datasets_tree"]
                    if root.name == "ipfs_datasets_py"
                    else authority["current_kit_tree"]
                )
            _commit, path = expression.split(":", 1)
            if path == "ipfs_datasets_py":
                return authority["current_datasets_gitlink"]
            if path == "ipfs_kit_py":
                return authority["current_kit_gitlink"]
            return materializer._M48_M47_FINAL_CONTROL_BLOBS[path]
        if args[0] == "ls-tree":
            path = args[3]
            oid = materializer._M48_M47_FINAL_CONTROL_BLOBS[path]
            mode = materializer._M48_M47_FINAL_CONTROL_MODES[path]
            return f"{mode} blob {oid}\t{path}"
        raise AssertionError((root, args))

    monkeypatch.setattr(materializer, "_git", exact_git)
    monkeypatch.setattr(
        materializer,
        "_m27_name_status",
        lambda _root, _old, _new: {
            path: "M" for path in materializer._M48_OPERATOR_CONTROL_PATHS
        },
    )
    monkeypatch.setattr(
        materializer, "_assert_m43_current_control_modes", lambda *_args: None
    )
    materializer._assert_m48_source_delta(REPO_ROOT, population, authority)

    tamperers = (
        lambda value: value["operator_control_paths"].append("ordinary.py"),
        lambda value: value["source_chain"]["m47_final_control_blobs"].__setitem__(
            next(iter(materializer._M48_M47_FINAL_CONTROL_BLOBS)), "0" * 40
        ),
        lambda value: value["source_chain"]["m47_final_control_modes"].__setitem__(
            next(iter(materializer._M48_M47_FINAL_CONTROL_MODES)), "100755"
        ),
        lambda value: value["source_chain"].__setitem__(
            "final_control_parent", "0" * 40
        ),
        lambda value: value.__setitem__("ordinary_source_changes", 1),
    )
    for tamper in tamperers:
        changed = copy.deepcopy(authority)
        tamper(changed)
        with pytest.raises(
            materializer.MaterializationError,
            match="M48 exact source-only control chain differs",
        ):
            materializer._assert_m48_source_delta(REPO_ROOT, population, changed)


def test_m48_dependency_and_board_validators_are_exact_with_mocked_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m48_validator_test",
    )
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m48_validator_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m48_validator_test",
    )
    key = materializer._M48_SUPERSESSION_REASON
    reference = materializer._m48_authority_reference()
    scheduler = {key: reference}
    migration = {key: reference}
    seal = {f"{key}_cid": materializer._M48_AUTHORITY_CID}

    fake_spec = SimpleNamespace(
        loader=SimpleNamespace(exec_module=lambda _module: None)
    )
    monkeypatch.setattr(
        dependencies.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: fake_spec,
    )
    monkeypatch.setattr(
        dependencies.importlib.util,
        "module_from_spec",
        lambda _spec: materializer,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m48_historical_m47_controls",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        dependencies,
        "_m48_source_chain_errors",
        lambda *_args, **_kwargs: [],
    )
    assert (
        dependencies._m48_post_m47_clean_shutdown_restart_successor_errors(
            scheduler,
            seal,
            migration,
            root=REPO_ROOT,
            require_active_runtime=False,
        )
        == []
    )

    partial_errors = (
        dependencies._m48_post_m47_clean_shutdown_restart_successor_errors(
            scheduler,
            {},
            migration,
            root=REPO_ROOT,
            require_active_runtime=False,
        )
    )
    assert "M48 successor authority is only partially declared" in partial_errors
    tampered = {key: {**reference, "migration_revision": "SAWM-R2-M47"}}
    tampered_errors = (
        dependencies._m48_post_m47_clean_shutdown_restart_successor_errors(
            tampered,
            seal,
            migration,
            root=REPO_ROOT,
            require_active_runtime=False,
        )
    )
    assert "M48 successor reference differs" in tampered_errors

    monkeypatch.setattr(board, "_dependency_validator_module", lambda _root: dependencies)
    assert (
        board._m48_migration_errors(
            scheduler, seal, migration, require_active_runtime=False
        )
        == []
    )


def test_m48_presence_masks_m47_across_dispatchers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m48_presence_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m48_presence_test",
    )
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m48_presence_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m48_presence_test",
    )
    m48_key = materializer._M48_SUPERSESSION_REASON
    m47_key = materializer._M47_SUPERSESSION_REASON

    materializer_calls: list[str] = []
    monkeypatch.setattr(materializer, "_load_json", lambda _path: {m48_key: None})
    monkeypatch.setattr(
        materializer,
        "_m48_successor_configured_on_any_surface",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        materializer,
        "_m47_successor_configured_on_any_surface",
        lambda *_args, **_kwargs: pytest.fail("M47 materializer route must stay masked"),
    )
    monkeypatch.setattr(
        materializer,
        "_check_m48_materialized",
        lambda *_args, **_kwargs: materializer_calls.append("check") or {"m48": True},
    )
    monkeypatch.setattr(
        materializer,
        "_materialize_m48",
        lambda *_args, **_kwargs: materializer_calls.append("materialize")
        or {"m48": True},
    )
    assert materializer.check_materialized(REPO_ROOT) == {"m48": True}
    assert materializer.materialize(REPO_ROOT) == {"m48": True}
    assert materializer_calls == ["check", "materialize"]

    board_calls: list[str] = []
    monkeypatch.setattr(
        board,
        "_m48_migration_errors",
        lambda *_args, **_kwargs: board_calls.append("M48") or ["m48-invalid"],
    )
    monkeypatch.setattr(
        board,
        "_m47_migration_errors",
        lambda *_args, **_kwargs: pytest.fail("M47 board route must stay masked"),
    )
    errors = board._active_successor_migration_errors(
        {m48_key: None, m47_key: {}},
        {f"{m48_key}_cid": "bad", f"{m47_key}_cid": "historical"},
        {m48_key: None, m47_key: {}},
    )
    assert errors == ["m48-invalid"]
    assert board_calls == ["M48"]

    monkeypatch.setattr(
        operator,
        "_require_m48_source_successor_marker",
        lambda *_args, **_kwargs: MappingProxyType({"selected": "M48"}),
    )
    monkeypatch.setattr(
        operator,
        "_require_m47_source_successor_marker",
        lambda *_args, **_kwargs: pytest.fail("M47 facade route must stay masked"),
    )
    selected = operator._require_active_final_pair_marker(
        {m48_key: None, m47_key: {}}, {}, object(), checked={}
    )
    assert selected == {"selected": "M48"}

    authority = (
        materializer
        ._expected_m48_post_m47_clean_shutdown_restart_successor_authority()
    )
    reference = materializer._m48_authority_reference()
    fake_materializer = SimpleNamespace(
        _M48_AUTHORITY_CID=materializer._M48_AUTHORITY_CID,
        _M48_OPERATOR_CONTROL_PATHS=materializer._M48_OPERATOR_CONTROL_PATHS,
        _expected_m48_post_m47_clean_shutdown_restart_successor_authority=(
            lambda: authority
        ),
        _validated_m48_live_preflight_contract=lambda value: value[
            "live_preflight_contract"
        ],
        _assert_m48_historical_m47_controls=lambda *_args: None,
        _m48_authority_reference=lambda: reference,
        _identity=lambda _value: materializer._M48_AUTHORITY_CID,
        _canonical=lambda _value: b"x" * 17_315,
    )
    fake_spec = SimpleNamespace(
        loader=SimpleNamespace(exec_module=lambda _module: None)
    )
    monkeypatch.setattr(
        dependencies.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: fake_spec,
    )
    monkeypatch.setattr(
        dependencies.importlib.util,
        "module_from_spec",
        lambda _spec: fake_materializer,
    )
    dependency_calls: list[str] = []
    monkeypatch.setattr(
        dependencies,
        "_m48_source_chain_errors",
        lambda *_args, **_kwargs: dependency_calls.append("M48") or [],
    )
    monkeypatch.setattr(
        dependencies,
        "_m47_source_chain_errors",
        lambda *_args, **_kwargs: pytest.fail("M47 dependency route must stay masked"),
    )
    source_authorities = [
        {"package": "ipfs_datasets_py"},
        {"package": "ipfs_kit_py"},
    ]
    scheduler = {m48_key: reference, m47_key: {}}
    migration = copy.deepcopy(scheduler)
    seal = {
        f"{m48_key}_cid": materializer._M48_AUTHORITY_CID,
        f"{m47_key}_cid": materializer._M47_AUTHORITY_CID,
    }
    effective, dependency_errors = dependencies._effective_nested_source_authorities(
        source_authorities, scheduler, migration, seal
    )
    assert dependency_errors == []
    assert dependency_calls == ["M48"]
    assert effective["ipfs_datasets_py"]["head"] == (
        authority["current_datasets_gitlink"]
    )
    assert effective["ipfs_kit_py"]["head"] == authority["current_kit_gitlink"]

    for function in (materializer.check_materialized, materializer.materialize):
        source = inspect.getsource(function)
        assert source.index("_m48_successor_configured_on_any_surface") < (
            source.index("_m47_successor_configured_on_any_surface")
        )
    for function in (
        operator._active_source_repair_materialization,
        operator._require_active_final_pair_marker,
        operator._validate_offline_quack_start,
        operator._normalized_live_preflight_contract,
        operator._live_preflight,
    ):
        source = inspect.getsource(function)
        assert source.index("M48") < source.index("M47")
    board_source = inspect.getsource(board.validate_program)
    assert board_source.index("_M48_SUCCESSOR_KEY") < board_source.index(
        "_M47_SUCCESSOR_KEY"
    )
    dependency_source = inspect.getsource(dependencies.validate_dependencies)
    assert dependency_source.index("m48_key = _M48_SUCCESSOR_KEY") < (
        dependency_source.index("m47_key = _M47_SUCCESSOR_KEY")
    )
    assert dependency_source.index("if any(m48_presence)") < (
        dependency_source.index("elif any(m47_presence)")
    )
    assert dependency_source.index("m48_declared = _m48_successor_declared") < (
        dependency_source.index("m47_declared = _m47_successor_declared")
    )


def test_m47_authority_binds_exact_repair_and_generation_34_transition() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m47_authority_test",
    )
    authority = (
        materializer
        ._expected_m47_ignored_python_cache_preservation_and_recovery_successor_authority()
    )
    reference = materializer._m47_authority_reference()
    contract = materializer._validated_m47_live_preflight_contract(authority)

    assert reference == {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M47",
        "authority_cid": materializer._M47_AUTHORITY_CID,
    }
    assert materializer._identity(authority) == materializer._M47_AUTHORITY_CID
    assert len(materializer._canonical(authority)) == 18_795
    assert authority["prior_authority"]["event_watermark"] == 303
    assert authority["prior_authority"]["event_prefix_sha256"] == (
        "dfb96eb98b763d00deca5a93f70fbf0a21ed6c576d5c31f26f1bdc2551b261d3"
    )
    assert authority["stopped_owner"]["generation"] == 33
    assert authority["stopped_owner"]["target_generation"] == 34
    assert authority["preserved_m46_receipt"]["receipt_cid"] == (
        "sha256:3f9d79c33306ada7b3074609c9fd1e43beee7e474a9fd8b4caacdd65966513c2"
    )
    assert authority["preserved_m46_receipt"]["created_or_rewritten"] is False
    assert authority["preserved_m46_materialization"]["event_watermark"] == 303
    assert authority["preserved_m46_materialization"]["m44_receipt_absent"]
    repair = authority["accepted_ignored_python_cache_preservation_repair"]
    assert repair["repair_parent"] == materializer._M47_M46_FINAL_CONTROL_COMMIT
    assert repair["repair_commit"] == materializer._M47_REPAIR_COMMIT
    assert repair["repair_tree"] == materializer._M47_REPAIR_TREE
    assert repair["binary_diff_sha256"] == materializer._M47_REPAIR_DIFF_SHA256
    assert repair["changed_paths"] == sorted(materializer._M47_REPAIR_BLOBS)
    assert repair["current_tag_cpython_cache_required"] is True
    assert repair["tracked_source_binding_required"] is True
    assert repair["raw_bytes_copied_to_receipt"] is False
    assert repair["recovery_classifier_executes_payloads"] is False
    assert repair["admitted_as_declared_output"] is False
    assert repair["cache_observation_authoritative"] is False
    assert repair["unsafe_ignored_artifacts_fail_closed"] is True
    assert repair["task_completion_authority"] is False
    assert contract["prior_generation"] == 33
    assert contract["target_generation"] == 34
    assert contract["prior_event_watermark"] == 303
    assert contract["target_event_watermark"] == 304
    assert contract["m46_receipt_must_be_preserved"] is True
    assert contract["m44_receipt_must_remain_absent"] is True
    assert authority["target_projection_cid"] == (
        "baguqeeraadvcue2olr4ies6ctvwaf4rzgafn5myeyomldgeaazkq53cco5wq"
    )
    assert authority["exact_changes"]["event_suffix_length"] == 1
    assert authority["exact_changes"]["task_status_changes"] == 0
    assert authority["exact_changes"]["accepted_completion_changes"] == 0

    for function in (materializer.check_materialized, materializer.materialize):
        source = inspect.getsource(function)
        assert source.index("_m47_successor_configured_on_any_surface") < (
            source.index("_m46_successor_configured_on_any_surface")
        )
    source_gate = inspect.getsource(materializer._assert_m47_source_delta)
    assert "_M47_M46_FINAL_CONTROL_COMMIT" in source_gate
    assert "_M47_REPAIR_COMMIT" in source_gate
    assert "_M47_REPAIR_DIFF_SHA256" in source_gate
    assert "_m27_name_status" in source_gate
    core = inspect.getsource(materializer._materialize_m47)
    assert core.index("_verify_m47_live_materialization") < core.index(
        "_ensure_m47_source_successor_receipt"
    )
    assert "_ensure_m46_source_successor_receipt" not in core


def test_m47_preflight_contract_rejects_tampering() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m47_tamper_test",
    )
    changes = (
        ("preserved_m46_receipt", "receipt_cid", "sha256:" + "0" * 64),
        ("preserved_m46_materialization", "event_id", "baguqeera" + "a" * 52),
        (
            "accepted_ignored_python_cache_preservation_repair",
            "repair_commit",
            "0" * 40,
        ),
        ("live_preflight_contract", "target_event_watermark", 303),
        ("exact_changes", "accepted_completion_changes", 1),
        ("preservation", "worker_self_approval", True),
    )
    for section, field, value in changes:
        authority = copy.deepcopy(
            materializer
            ._expected_m47_ignored_python_cache_preservation_and_recovery_successor_authority()
        )
        authority[section][field] = value
        with pytest.raises(
            materializer.MaterializationError,
            match="M47 live preflight contract differs",
        ):
            materializer._validated_m47_live_preflight_contract(authority)


def test_m47_receipt_publication_preserves_m46_and_m44_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m47_receipt_test",
    )
    control = tmp_path / "control.duckdb"
    control.write_bytes(b"test")
    preserved = {"receipt_cid": materializer._M47_M46_RECEIPT_CID}
    monkeypatch.setattr(
        materializer,
        "_verify_m47_preserved_m46_receipt",
        lambda *_args, **_kwargs: preserved,
    )
    expected = {"schema": "test/m47-receipt@1", "receipt_cid": "sha256:test"}

    first = materializer._ensure_m47_source_successor_receipt(
        tmp_path, control, expected
    )
    second = materializer._ensure_m47_source_successor_receipt(
        tmp_path, control, expected
    )
    assert first == second == expected
    path = tmp_path / "m47-source-successor-receipt.json"
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for revision in ("m44", "m45", "m46", "m47"):
        assert (
            tmp_path / f".{revision}-source-successor-receipt.publish.lock"
        ).exists()

    (tmp_path / "m44-source-successor-receipt.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="M44 receipt unexpectedly exists",
    ):
        materializer._ensure_m47_source_successor_receipt(
            tmp_path, control, expected
        )


def test_m47_actual_protected_surfaces_bind_generation_34(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m47_protected_surfaces_test",
    )
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m47_protected_surfaces_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m47_protected_surfaces_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m47_protected_surfaces_test",
    )
    live_scheduler = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    live_seal = json.loads(
        (
            REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    live_migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    key = materializer._M47_SUPERSESSION_REASON
    scheduler, migration, seal = _historical_successor_controls_at(
        key, live_scheduler, live_migration, live_seal
    )
    assert seal is not None and migration is not None
    scheduler["database_program"]["store_generation"] = "34"
    reference = materializer._m47_authority_reference()

    assert scheduler[key] == migration[key] == reference
    assert seal[f"{key}_cid"] == materializer._M47_AUTHORITY_CID
    assert scheduler["database_program"]["store_generation"] == "34"
    assert scheduler["database_program"]["store_id"] == materializer._M47_STORE_ID
    assert scheduler["quack_owner"]["store_id"] == materializer._M47_STORE_ID
    assert scheduler["runtime_paths"]["root"] == materializer._M47_RUNTIME_ROOT

    # This isolated M47 fixture bypasses M47's obsolete current-HEAD source gate.
    monkeypatch.setattr(
        dependencies, "_m47_source_chain_errors", lambda *_args, **_kwargs: []
    )
    assert (
        dependencies
        ._m47_ignored_python_cache_preservation_and_recovery_successor_errors(
            scheduler,
            seal,
            migration,
            root=REPO_ROOT,
            require_active_runtime=True,
        )
        == []
    )

    monkeypatch.setattr(
        board,
        "_dependency_validator_module",
        lambda _root: dependencies,
    )
    assert board._m47_migration_errors(scheduler, seal, migration) == []

    selected = operator._active_source_repair_materialization(scheduler)
    assert selected["migration_revision"] == "SAWM-R2-M47"
    assert selected["target_generation"] == 34
    assert selected["runtime_binding"]["store_generation"] == 34
    assert selected["runtime_binding"]["target_event_watermark"] == 304


def test_m47_presence_masks_m46_across_all_dispatchers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m47_presence_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m47_presence_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m47_presence_test",
    )
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m47_presence_test",
    )
    m47_key = operator._M47_SUCCESSOR_KEY
    m46_key = operator._M46_SUCCESSOR_KEY

    with monkeypatch.context() as scoped:
        scoped.setattr(materializer, "_load_json", lambda _path: {})
        with pytest.raises(
            materializer.MaterializationError,
            match="M47 successor authority is only partially declared",
        ):
            materializer._m47_successor_configured_on_any_surface(
                REPO_ROOT, {m47_key: None}
            )

    materializer_calls: list[str] = []
    monkeypatch.setattr(materializer, "_load_json", lambda _path: {m47_key: None})
    monkeypatch.setattr(
        materializer,
        "_m47_successor_configured_on_any_surface",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        materializer,
        "_m46_successor_configured_on_any_surface",
        lambda *_args, **_kwargs: pytest.fail("M46 materializer route must stay masked"),
    )
    monkeypatch.setattr(
        materializer,
        "_check_m47_materialized",
        lambda *_args, **_kwargs: materializer_calls.append("check") or {"m47": True},
    )
    monkeypatch.setattr(
        materializer,
        "_materialize_m47",
        lambda *_args, **_kwargs: materializer_calls.append("materialize")
        or {"m47": True},
    )
    assert materializer.check_materialized(REPO_ROOT) == {"m47": True}
    assert materializer.materialize(REPO_ROOT) == {"m47": True}
    assert materializer_calls == ["check", "materialize"]

    board_calls: list[str] = []
    monkeypatch.setattr(
        board,
        "_m47_migration_errors",
        lambda *_args, **_kwargs: board_calls.append("M47") or ["m47-invalid"],
    )
    monkeypatch.setattr(
        board,
        "_m46_migration_errors",
        lambda *_args, **_kwargs: pytest.fail("M46 board route must stay masked"),
    )
    errors = board._active_successor_migration_errors(
        {m47_key: None, m46_key: {}},
        {f"{m47_key}_cid": "bad", f"{m46_key}_cid": "historical"},
        {m47_key: None, m46_key: {}},
    )
    assert errors == ["m47-invalid"]
    assert board_calls == ["M47"]

    monkeypatch.setattr(
        operator,
        "_require_m47_source_successor_marker",
        lambda *_args, **_kwargs: MappingProxyType({"selected": "M47"}),
    )
    monkeypatch.setattr(
        operator,
        "_require_m46_source_successor_marker",
        lambda *_args, **_kwargs: pytest.fail(
            "M46 operator marker must stay masked"
        ),
    )
    selected = operator._require_active_final_pair_marker(
        {m47_key: None, m46_key: {}}, {}, object(), checked={}
    )
    assert selected == {"selected": "M47"}

    scheduler = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    dependency_calls: list[str] = []
    monkeypatch.setattr(
        dependencies,
        "_m47_source_chain_errors",
        lambda *_args, **_kwargs: dependency_calls.append("M47") or [],
    )
    monkeypatch.setattr(
        dependencies,
        "_m46_source_chain_errors",
        lambda *_args, **_kwargs: pytest.fail(
            "M46 dependency route must stay masked"
        ),
    )
    effective, dependency_errors = dependencies._effective_nested_source_authorities(
        seal["source_authorities"], scheduler, migration, seal
    )
    assert dependency_errors == []
    assert dependency_calls == ["M47"]
    assert effective["ipfs_datasets_py"]["head"] == (
        "b9f5b86199c03e427fd51fcea302479880421ff8"
    )
    assert effective["ipfs_kit_py"]["head"] == (
        "fc9248073e9f67ac59ca607c7736746907b08037"
    )


def test_m47_source_delta_rejects_path_blob_mode_diff_and_parent_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m47_source_delta_negative_test",
    )
    authority = (
        materializer
        ._expected_m47_ignored_python_cache_preservation_and_recovery_successor_authority()
    )
    unsealed_population = {
        "source_binding": {
            "head": materializer._M47_REPAIR_COMMIT,
            "tree": materializer._M47_REPAIR_TREE,
        }
    }
    with pytest.raises(
        materializer.MaterializationError,
        match="M47 exact repair/control source chain differs",
    ):
        materializer._assert_m47_source_delta(
            REPO_ROOT, unsealed_population, authority
        )

    current = "f" * 40
    current_tree = "e" * 40
    population = {
        "source_binding": {
            "head": current,
            "tree": current_tree,
            "datasets_gitlink": authority["current_datasets_gitlink"],
            "kit_gitlink": authority["current_kit_gitlink"],
        }
    }

    def fake_git(root: Path, *args: str) -> str:
        if args[:4] == ("rev-list", "--parents", "-n", "1"):
            commit = args[4]
            parents = {
                materializer._M47_M46_FINAL_CONTROL_COMMIT: (
                    materializer._M46_REPAIR_COMMIT
                ),
                materializer._M47_REPAIR_COMMIT: (
                    materializer._M47_M46_FINAL_CONTROL_COMMIT
                ),
                current: materializer._M47_REPAIR_COMMIT,
            }
            return f"{commit} {parents[commit]}"
        if args[0] == "rev-parse":
            expression = args[1]
            trees = {
                f"{materializer._M47_M46_FINAL_CONTROL_COMMIT}^{{tree}}": (
                    materializer._M47_M46_FINAL_CONTROL_TREE
                ),
                f"{materializer._M47_REPAIR_COMMIT}^{{tree}}": (
                    materializer._M47_REPAIR_TREE
                ),
                f"{current}^{{tree}}": current_tree,
            }
            if expression in trees:
                return trees[expression]
            if expression.endswith("^{tree}"):
                if Path(root).name == "ipfs_datasets_py":
                    return authority["current_datasets_tree"]
                if Path(root).name == "ipfs_kit_py":
                    return authority["current_kit_tree"]
            commit, path = expression.split(":", 1)
            if path == "ipfs_datasets_py":
                return authority["current_datasets_gitlink"]
            if path == "ipfs_kit_py":
                return authority["current_kit_gitlink"]
            blobs = (
                materializer._M47_M46_FINAL_CONTROL_BLOBS
                if commit == materializer._M47_M46_FINAL_CONTROL_COMMIT
                else materializer._M47_REPAIR_BLOBS
            )
            return blobs[path]
        if args[0] == "ls-tree":
            commit, path = args[1], args[3]
            blobs = (
                materializer._M47_M46_FINAL_CONTROL_BLOBS
                if commit == materializer._M47_M46_FINAL_CONTROL_COMMIT
                else materializer._M47_REPAIR_BLOBS
            )
            modes = (
                materializer._M47_M46_FINAL_CONTROL_MODES
                if commit == materializer._M47_M46_FINAL_CONTROL_COMMIT
                else materializer._M47_REPAIR_MODES
            )
            return f"{modes[path]} blob {blobs[path]}\t{path}"
        raise AssertionError((root, args))

    monkeypatch.setattr(materializer, "_git", fake_git)
    monkeypatch.setattr(
        materializer,
        "_m27_name_status",
        lambda _root, old, _new: (
            {path: "M" for path in materializer._M47_REPAIR_BLOBS}
            if old == materializer._M47_M46_FINAL_CONTROL_COMMIT
            else {path: "M" for path in materializer._M47_OPERATOR_CONTROL_PATHS}
        ),
    )
    monkeypatch.setattr(
        materializer, "_assert_m43_current_control_modes", lambda *_args: None
    )
    monkeypatch.setattr(
        materializer.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=b"repair"),
    )
    monkeypatch.setattr(
        materializer.hashlib,
        "sha256",
        lambda _value=b"": SimpleNamespace(
            hexdigest=lambda: materializer._M47_REPAIR_DIFF_SHA256
        ),
    )
    materializer._assert_m47_source_delta(REPO_ROOT, population, authority)

    tamperers = (
        lambda value: value["operator_control_paths"].pop(),
        lambda value: value["source_chain"]["repair_blobs"].__setitem__(
            next(iter(materializer._M47_REPAIR_BLOBS)), "0" * 40
        ),
        lambda value: value["source_chain"]["repair_modes"].__setitem__(
            next(iter(materializer._M47_REPAIR_MODES)), "100755"
        ),
        lambda value: value["source_chain"].__setitem__(
            "repair_diff_sha256", "0" * 64
        ),
        lambda value: value["source_chain"].__setitem__(
            "final_control_parent", "0" * 40
        ),
    )
    for tamper in tamperers:
        changed = copy.deepcopy(authority)
        tamper(changed)
        with pytest.raises(
            materializer.MaterializationError,
            match="M47 exact repair/control source chain differs",
        ):
            materializer._assert_m47_source_delta(REPO_ROOT, population, changed)


def test_m46_authority_binds_exact_repair_and_generation_33_transition() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m46_authority_test",
    )
    authority = (
        materializer
        ._expected_m46_legacy_no_delta_rescue_recovery_successor_authority()
    )
    reference = materializer._m46_authority_reference()
    contract = materializer._validated_m46_live_preflight_contract(authority)

    assert reference == {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M46",
        "authority_cid": (
            "sha256:47ed315018541ef5c4c759c1c486cae2a411821ed4e9902877b08439b79b4455"
        ),
    }
    assert materializer._identity(authority) == materializer._M46_AUTHORITY_CID
    assert len(materializer._canonical(authority)) == 19_365
    assert authority["prior_authority"]["event_watermark"] == 302
    assert authority["prior_authority"]["event_prefix_sha256"] == (
        "481af01b013ab87cebe2f1870eb0788f33661ad800e0af8d43905125a37309b6"
    )
    assert authority["stopped_owner"]["generation"] == 32
    assert authority["stopped_owner"]["target_generation"] == 33
    assert authority["preserved_m45_receipt"]["receipt_cid"] == (
        "sha256:46508d2540469d9cbc3ab0cc5220db0bf0cf0cb3cfdce6eed2ff8c8f4da71a97"
    )
    assert authority["preserved_m45_receipt"]["created_or_rewritten"] is False
    assert authority["preserved_m45_materialization"]["event_watermark"] == 302
    assert authority["preserved_m45_materialization"]["m44_receipt_absent"]
    repair = authority["accepted_legacy_no_delta_rescue_repair"]
    assert repair["repair_parent"] == materializer._M46_M45_FINAL_CONTROL_COMMIT
    assert repair["repair_commit"] == materializer._M46_REPAIR_COMMIT
    assert repair["repair_tree"] == materializer._M46_REPAIR_TREE
    assert repair["binary_diff_sha256"] == materializer._M46_REPAIR_DIFF_SHA256
    assert repair["changed_paths"] == sorted(materializer._M46_REPAIR_BLOBS)
    assert repair["status_empty_or_nested_gitlink_only_required"] is True
    assert repair["attestation_metadata_validated_exactly"] is True
    assert repair["task_completion_authority"] is False
    assert contract["prior_generation"] == 32
    assert contract["target_generation"] == 33
    assert contract["prior_event_watermark"] == 302
    assert contract["target_event_watermark"] == 303
    assert contract["m45_receipt_must_be_preserved"] is True
    assert contract["m44_receipt_must_remain_absent"] is True
    assert authority["target_projection_cid"] == (
        "baguqeera3gwgex35vb2k2c2u2vq5qq6d65immvntj3nl2fqfzjpts6jvrfta"
    )
    assert authority["exact_changes"]["event_suffix_length"] == 1
    assert authority["exact_changes"]["task_status_changes"] == 0
    assert authority["exact_changes"]["accepted_completion_changes"] == 0

    for function in (materializer.check_materialized, materializer.materialize):
        source = inspect.getsource(function)
        assert source.index("_m46_successor_configured_on_any_surface") < (
            source.index("_m45_successor_configured_on_any_surface")
        )
    source_gate = inspect.getsource(materializer._assert_m46_source_delta)
    assert "_M46_M45_FINAL_CONTROL_COMMIT" in source_gate
    assert "_M46_REPAIR_COMMIT" in source_gate
    assert "_M46_REPAIR_DIFF_SHA256" in source_gate
    assert "_m27_name_status" in source_gate
    core = inspect.getsource(materializer._materialize_m46)
    assert core.index("_verify_m46_live_materialization") < core.index(
        "_ensure_m46_source_successor_receipt"
    )
    assert "_ensure_m45_source_successor_receipt" not in core


def test_m46_preflight_contract_rejects_tampering() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m46_tamper_test",
    )
    changes = (
        ("preserved_m45_receipt", "receipt_cid", "sha256:" + "0" * 64),
        ("preserved_m45_materialization", "event_id", "baguqeera" + "a" * 52),
        ("accepted_legacy_no_delta_rescue_repair", "repair_commit", "0" * 40),
        ("live_preflight_contract", "target_event_watermark", 302),
        ("exact_changes", "accepted_completion_changes", 1),
        ("preservation", "worker_self_approval", True),
    )
    for section, field, value in changes:
        authority = copy.deepcopy(
            materializer
            ._expected_m46_legacy_no_delta_rescue_recovery_successor_authority()
        )
        authority[section][field] = value
        with pytest.raises(
            materializer.MaterializationError,
            match="M46 live preflight contract differs",
        ):
            materializer._validated_m46_live_preflight_contract(authority)


def test_m46_receipt_publication_preserves_m45_and_m44_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m46_receipt_test",
    )
    control = tmp_path / "control.duckdb"
    control.write_bytes(b"test")
    preserved = {"receipt_cid": materializer._M46_M45_RECEIPT_CID}
    monkeypatch.setattr(
        materializer,
        "_verify_m46_preserved_m45_receipt",
        lambda *_args, **_kwargs: preserved,
    )
    expected = {"schema": "test/m46-receipt@1", "receipt_cid": "sha256:test"}

    first = materializer._ensure_m46_source_successor_receipt(
        tmp_path, control, expected
    )
    second = materializer._ensure_m46_source_successor_receipt(
        tmp_path, control, expected
    )
    assert first == second == expected
    path = tmp_path / "m46-source-successor-receipt.json"
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for revision in ("m44", "m45", "m46"):
        assert (
            tmp_path / f".{revision}-source-successor-receipt.publish.lock"
        ).exists()

    (tmp_path / "m44-source-successor-receipt.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="M44 receipt unexpectedly exists",
    ):
        materializer._ensure_m46_source_successor_receipt(
            tmp_path, control, expected
        )


def test_m46_actual_protected_surfaces_bind_generation_33(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m46_protected_surfaces_test",
    )
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m46_protected_surfaces_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m46_protected_surfaces_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m46_protected_surfaces_test",
    )
    live_scheduler = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    live_seal = json.loads(
        (
            REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    live_migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    key = materializer._M46_SUPERSESSION_REASON
    scheduler, migration, seal = _historical_successor_controls_at(
        key, live_scheduler, live_migration, live_seal
    )
    assert seal is not None and migration is not None
    scheduler["database_program"]["store_generation"] = "33"
    reference = materializer._m46_authority_reference()

    assert scheduler[key] == migration[key] == reference
    assert seal[f"{key}_cid"] == materializer._M46_AUTHORITY_CID
    assert scheduler["database_program"]["store_id"] == materializer._M46_STORE_ID
    assert scheduler["quack_owner"]["store_id"] == materializer._M46_STORE_ID
    assert scheduler["runtime_paths"]["root"] == materializer._M46_RUNTIME_ROOT

    # The final-control child is intentionally not committed in this fixture.
    # Suppress only that Git-head check while exercising all protected values.
    monkeypatch.setattr(
        dependencies, "_m46_source_chain_errors", lambda *_args, **_kwargs: []
    )
    assert dependencies._m46_legacy_no_delta_rescue_recovery_successor_errors(
        scheduler,
        seal,
        migration,
        root=REPO_ROOT,
        require_active_runtime=True,
    ) == []

    monkeypatch.setattr(
        board,
        "_dependency_validator_module",
        lambda _root: dependencies,
    )
    assert board._m46_migration_errors(scheduler, seal, migration) == []

    selected = operator._active_source_repair_materialization(scheduler)
    assert selected["migration_revision"] == "SAWM-R2-M46"
    assert selected["target_generation"] == 33
    assert selected["runtime_binding"]["store_generation"] == 33
    assert selected["runtime_binding"]["target_event_watermark"] == 303


def test_m46_presence_masks_m45_across_all_dispatchers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m46_presence_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m46_presence_test",
    )
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m46_presence_test",
    )
    m46_key = operator._M46_SUCCESSOR_KEY
    m45_key = operator._M45_SUCCESSOR_KEY

    board_calls: list[str] = []
    monkeypatch.setattr(
        board,
        "_m46_migration_errors",
        lambda *_args, **_kwargs: board_calls.append("M46") or ["m46-invalid"],
    )
    monkeypatch.setattr(
        board,
        "_m45_migration_errors",
        lambda *_args, **_kwargs: pytest.fail("M45 board route must stay masked"),
    )
    errors = board._active_successor_migration_errors(
        {m46_key: None, m45_key: {}},
        {f"{m46_key}_cid": "bad", f"{m45_key}_cid": "historical"},
        {m46_key: None, m45_key: {}},
    )
    assert errors == ["m46-invalid"]
    assert board_calls == ["M46"]

    monkeypatch.setattr(
        operator,
        "_require_m46_source_successor_marker",
        lambda *_args, **_kwargs: MappingProxyType({"selected": "M46"}),
    )
    monkeypatch.setattr(
        operator,
        "_require_m45_source_successor_marker",
        lambda *_args, **_kwargs: pytest.fail(
            "M45 operator marker must stay masked"
        ),
    )
    selected = operator._require_active_final_pair_marker(
        {m46_key: None, m45_key: {}}, {}, object(), checked={}
    )
    assert selected == {"selected": "M46"}

    live_scheduler = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    live_migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    live_seal = json.loads(
        (
            REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    scheduler, migration, seal = _historical_successor_controls_at(
        m46_key, live_scheduler, live_migration, live_seal
    )
    assert seal is not None and migration is not None
    dependency_calls: list[str] = []
    monkeypatch.setattr(
        dependencies,
        "_m46_source_chain_errors",
        lambda *_args, **_kwargs: dependency_calls.append("M46") or [],
    )
    monkeypatch.setattr(
        dependencies,
        "_m45_source_chain_errors",
        lambda *_args, **_kwargs: pytest.fail(
            "M45 dependency route must stay masked"
        ),
    )
    effective, dependency_errors = dependencies._effective_nested_source_authorities(
        seal["source_authorities"], scheduler, migration, seal
    )
    assert dependency_errors == []
    assert dependency_calls == ["M46"]
    assert effective["ipfs_datasets_py"]["head"] == (
        "b9f5b86199c03e427fd51fcea302479880421ff8"
    )
    assert effective["ipfs_kit_py"]["head"] == (
        "fc9248073e9f67ac59ca607c7736746907b08037"
    )


def test_m46_source_delta_rejects_unsealed_or_tampered_children() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m46_source_delta_negative_test",
    )
    authority = (
        materializer
        ._expected_m46_legacy_no_delta_rescue_recovery_successor_authority()
    )
    unsealed_population = {
        "source_binding": {
            "head": materializer._M46_REPAIR_COMMIT,
            "tree": materializer._M46_REPAIR_TREE,
        }
    }
    with pytest.raises(
        materializer.MaterializationError,
        match="M46 exact repair/control source chain differs",
    ):
        materializer._assert_m46_source_delta(
            REPO_ROOT, unsealed_population, authority
        )

    tampered = copy.deepcopy(authority)
    tampered["source_chain"]["repair_commit"] = "0" * 40
    nonfinal_population = {
        "source_binding": {
            "head": materializer._M45_TEST_REPAIR_COMMIT,
            "tree": materializer._M45_TEST_REPAIR_TREE,
        }
    }
    with pytest.raises(
        materializer.MaterializationError,
        match="M46 exact repair/control source chain differs",
    ):
        materializer._assert_m46_source_delta(
            REPO_ROOT, nonfinal_population, tampered
        )


def test_m45_authority_binds_exact_repair_and_unchanged_m44_transition() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m45_authority_test",
    )
    authority = (
        materializer
        ._expected_m45_failed_pre_authoritative_m44_validation_successor_authority()
    )
    reference = materializer._m45_authority_reference()
    contract = materializer._validated_m45_live_preflight_contract(authority)
    m44 = (
        materializer
        ._expected_m44_post_m43_hardened_procfs_user_manager_restart_successor_authority()
    )

    assert reference == {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M45",
        "authority_cid": materializer._M45_AUTHORITY_CID,
    }
    assert materializer._identity(authority) == materializer._M45_AUTHORITY_CID
    assert len(materializer._canonical(authority)) == 14_585
    assert materializer._identity(m44) == materializer._M44_AUTHORITY_CID
    assert authority["preserved_m44_authority"]["canonical_body_size"] == 20_272
    repair = authority["accepted_test_expectation_correction"]
    assert repair["repair_parent"] == materializer._M45_M44_FINAL_CONTROL_COMMIT
    assert repair["repair_commit"] == materializer._M45_TEST_REPAIR_COMMIT
    assert repair["repair_tree"] == materializer._M45_TEST_REPAIR_TREE
    assert repair["repair_blob_oid"] == materializer._M45_TEST_REPAIR_BLOB
    assert repair["binary_diff_sha256"] == materializer._M45_TEST_REPAIR_DIFF_SHA256
    assert repair["operator_behavior_changed"] is False
    assert repair["validation_weakened"] is False
    assert contract["prior_generation"] == 31
    assert contract["target_generation"] == 32
    assert contract["prior_event_watermark"] == 301
    assert contract["target_event_watermark"] == 302
    assert contract["m44_receipt_must_be_absent_before_publication"] is True
    assert authority["delegated_m44_materialization"][
        "m44_transition_semantics_preserved_exactly"
    ] is True
    assert authority["exact_changes"]["test_expectation_changes"] == 1
    assert authority["exact_changes"]["task_status_changes"] == 0
    assert authority["exact_changes"]["accepted_completion_changes"] == 0
    assert authority["preservation"]["m44_receipt_absent_before_publication"]

    for function in (materializer.check_materialized, materializer.materialize):
        source = inspect.getsource(function)
        assert source.index("_m45_successor_configured_on_any_surface") < (
            source.index("_m44_successor_configured_on_any_surface")
        )
    source_gate = inspect.getsource(materializer._assert_m45_source_delta)
    assert "_M45_TEST_REPAIR_COMMIT" in source_gate
    assert "_M45_TEST_REPAIR_DIFF_SHA256" in source_gate
    assert '"git", "show"' in source_gate
    assert ".read_text(" not in source_gate
    core = inspect.getsource(materializer._materialize_m45)
    assert "_m44_migration_body" in core
    assert core.index("_verify_m44_live_materialization") < core.index(
        "_ensure_m45_source_successor_receipt"
    )
    assert "_ensure_m44_source_successor_receipt" not in core
    publisher = inspect.getsource(materializer._ensure_m45_source_successor_receipt)
    assert publisher.index("_verify_m44_preserved_m43_receipt") < (
        publisher.index("os.replace")
    )
    assert publisher.index("os.path.lexists(m44_path)") < publisher.index(
        "os.replace"
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("preserved_m44_authority", "authority_cid", "sha256:" + "0" * 64),
        ("accepted_test_expectation_correction", "repair_commit", "0" * 40),
        ("accepted_test_expectation_correction", "binary_diff_sha256", "0" * 64),
        ("delegated_m44_materialization", "target_generation", 31),
        ("live_preflight_contract", "target_event_watermark", 301),
        ("exact_changes", "accepted_completion_changes", 1),
        ("preservation", "worker_self_approval", True),
    ),
)
def test_m45_preflight_contract_rejects_tampering(
    section: str,
    field: str,
    value: object,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        f"sawm_materializer_m45_tamper_{section}_{field}",
    )
    authority = copy.deepcopy(
        materializer
        ._expected_m45_failed_pre_authoritative_m44_validation_successor_authority()
    )
    authority[section][field] = value
    with pytest.raises(
        materializer.MaterializationError,
        match="M45 live preflight contract differs",
    ):
        materializer._validated_m45_live_preflight_contract(authority)


def test_m45_presence_masks_m44_in_board_and_operator_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m45_presence_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m45_presence_test",
    )
    m45_key = operator._M45_SUCCESSOR_KEY
    m44_key = operator._M44_SUCCESSOR_KEY
    calls: list[str] = []
    monkeypatch.setattr(
        board,
        "_m45_migration_errors",
        lambda *_args, **_kwargs: calls.append("board-m45") or ["m45-invalid"],
    )
    monkeypatch.setattr(
        board,
        "_m44_migration_errors",
        lambda *_args, **_kwargs: pytest.fail("M44 must stay historical"),
    )
    errors = board._active_successor_migration_errors(
        {m45_key: None, m44_key: {}},
        {f"{m45_key}_cid": "bad", f"{m44_key}_cid": "historical"},
        {m45_key: None, m44_key: {}},
    )
    assert errors == ["m45-invalid"]
    assert calls == ["board-m45"]

    monkeypatch.setattr(
        operator,
        "_require_m45_source_successor_marker",
        lambda *_args, **_kwargs: MappingProxyType({"selected": "M45"}),
    )
    monkeypatch.setattr(
        operator,
        "_require_m44_source_successor_marker",
        lambda *_args, **_kwargs: pytest.fail("M44 marker must stay historical"),
    )
    selected = operator._require_active_final_pair_marker(
        {m45_key: None, m44_key: {}}, {}, object(), checked={}
    )
    assert selected == {"selected": "M45"}


def test_m45_dependency_validator_keeps_m44_historical() -> None:
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m45_historical_m44_test",
    )
    source = inspect.getsource(
        dependencies._m45_failed_pre_authoritative_m44_validation_successor_errors
    )
    assert "_assert_m45_historical_m44_controls" in source
    assert "_m44_source_chain_errors(" not in source
    assert "_assert_m44_source_delta" not in source


def test_m44_authority_binds_stopped_m43_head_and_exact_repair() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m44_authority_test",
    )
    authority = (
        materializer
        ._expected_m44_post_m43_hardened_procfs_user_manager_restart_successor_authority()
    )
    reference = materializer._m44_authority_reference()
    contract = materializer._validated_m44_live_preflight_contract(authority)

    assert reference == {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M44",
        "authority_cid": materializer._M44_AUTHORITY_CID,
    }
    assert materializer._identity(authority) == materializer._M44_AUTHORITY_CID
    assert authority["schema"] == (
        "sawm/post-m43-hardened-procfs-user-manager-restart-authorization@1"
    )
    assert authority["prior_authority"]["event_watermark"] == 301
    assert authority["stopped_owner"]["generation"] == 31
    assert authority["stopped_owner"]["target_generation"] == 32
    assert authority["preserved_m43_receipt"]["receipt_cid"] == (
        "sha256:eea44e0f2aae970c1580c59e7b3310904fad480f249b3b1f5df5cd887fc1f050"
    )
    assert authority["preserved_m43_receipt"]["created_or_rewritten"] is False
    assert authority["preserved_m43_materialization"]["event_watermark"] == 297
    assert authority["post_m43_operational_suffix"]["events"] == (
        materializer._m44_post_m43_operational_suffix()["events"]
    )
    assert authority["post_m43_operational_suffix"]["event_count"] == 4
    repair = authority["accepted_user_manager_procfs_repair"]
    assert repair["repair_parent"] == materializer._M44_REPAIR_PARENT
    assert repair["repair_commit"] == materializer._M44_REPAIR_COMMIT
    assert repair["blob_oids"] == dict(materializer._M44_REPAIR_BLOBS)
    assert repair["path_modes"] == dict(materializer._M44_REPAIR_MODES)
    assert repair["near_misses_fail_closed"] is True
    assert repair["validation_weakened"] is False
    assert repair["authority_weakened"] is False
    assert contract["prior_event_watermark"] == 301
    assert contract["target_event_watermark"] == 302
    assert contract["prior_generation"] == 31
    assert contract["target_generation"] == 32
    assert contract["events_297_through_301_must_be_preserved"] is True
    assert contract["event_302_must_be_absent_before_append"] is True
    assert authority["exact_changes"]["task_revision_changes"] == 0
    assert authority["exact_changes"]["task_status_changes"] == 0
    assert authority["exact_changes"]["worker_self_approval"] is False
    for function in (materializer.check_materialized, materializer.materialize):
        source = inspect.getsource(function)
        assert source.index("_m44_successor_configured_on_any_surface") < (
            source.index("_m43_successor_configured_on_any_surface")
        )
    core = inspect.getsource(materializer._materialize_m44)
    assert core.index("_verify_m44_preserved_m43_receipt") < core.index(
        "source.record_evidence"
    )
    assert core.index("_verify_m44_live_materialization") < core.index(
        "_ensure_m44_source_successor_receipt"
    )
    assert "snapshot.event_cursor == _M44_PRIOR_EVENT_WATERMARK" in core
    assert "snapshot.event_cursor != _M44_TARGET_EVENT_WATERMARK" in core
    publisher = inspect.getsource(materializer._ensure_m44_source_successor_receipt)
    assert publisher.index("_verify_m44_preserved_m43_receipt") < (
        publisher.index("os.replace")
    )
    assert publisher.count("_verify_m44_preserved_m43_receipt") >= 3
    historical_publisher = inspect.getsource(
        materializer._ensure_m29_source_successor_receipt
    )
    assert "_verify_m44_preserved_m43_receipt" not in historical_publisher
    verifier = inspect.getsource(materializer._verify_m44_live_materialization)
    assert "_verify_m44_preserved_m43_materialization" in verifier
    assert "_m44_operational_suffix_on" in verifier
    assert "_inspect_m44_generation_restart_rows" in verifier


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("preserved_m43_receipt", "receipt_cid", "sha256:" + "0" * 64),
        ("preserved_m43_materialization", "event_id", "baguqeerawrong"),
        ("post_m43_operational_suffix", "event_count", 3),
        ("accepted_user_manager_procfs_repair", "repair_commit", "0" * 40),
        ("live_preflight_contract", "target_generation", 31),
        ("exact_changes", "task_status_changes", 1),
        ("preservation", "worker_self_approval", True),
    ),
)
def test_m44_preflight_contract_rejects_anchor_tampering(
    section: str,
    field: str,
    value: object,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        f"sawm_materializer_m44_tamper_{section}_{field}",
    )
    authority = copy.deepcopy(
        materializer
        ._expected_m44_post_m43_hardened_procfs_user_manager_restart_successor_authority()
    )
    authority[section][field] = value
    with pytest.raises(
        materializer.MaterializationError,
        match="M44 live preflight contract differs",
    ):
        materializer._validated_m44_live_preflight_contract(authority)


def test_m44_presence_masks_m43_in_board_and_operator_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m44_presence_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m44_presence_test",
    )
    m44_key = operator._M44_SUCCESSOR_KEY
    m43_key = operator._M43_SUCCESSOR_KEY
    calls: list[str] = []

    monkeypatch.setattr(
        board,
        "_m44_migration_errors",
        lambda *_args, **_kwargs: calls.append("board-m44") or ["m44-invalid"],
    )
    monkeypatch.setattr(
        board,
        "_m43_migration_errors",
        lambda *_args, **_kwargs: calls.append("board-m43") or [],
    )
    errors = board._active_successor_migration_errors(
        {m44_key: None, m43_key: {}},
        {f"{m44_key}_cid": "bad", f"{m43_key}_cid": "historical"},
        {m44_key: None, m43_key: {}},
    )
    assert errors == ["m44-invalid"]
    assert calls == ["board-m44"]

    monkeypatch.setattr(
        operator,
        "_require_m44_source_successor_marker",
        lambda *_args, **_kwargs: MappingProxyType({"selected": "M44"}),
    )
    monkeypatch.setattr(
        operator,
        "_require_m43_source_successor_marker",
        lambda *_args, **_kwargs: pytest.fail("M43 marker must stay historical"),
    )
    selected = operator._require_active_final_pair_marker(
        {m44_key: None, m43_key: {}},
        {},
        object(),
        checked={},
    )
    assert selected == {"selected": "M44"}


def test_m44_dependency_validator_does_not_reenter_active_m43_source_gate() -> None:
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m44_historical_m43_test",
    )
    source = inspect.getsource(
        dependencies
        ._m44_post_m43_hardened_procfs_user_manager_restart_successor_errors
    )
    assert "_assert_m44_historical_m43_controls" in source
    assert "_m43_source_chain_errors(" not in source
    assert "_assert_m43_source_delta" not in source


def test_m44_dependency_source_chain_rejects_repair_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m44_source_chain_test",
    )
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m44_source_chain_test",
    )
    authority = (
        materializer
        ._expected_m44_post_m43_hardened_procfs_user_manager_restart_successor_authority()
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m44_source_delta",
        lambda *_args, **_kwargs: None,
    )
    assert dependencies._m44_source_chain_errors(
        REPO_ROOT, materializer, authority
    ) == []

    tampered = copy.deepcopy(authority)
    tampered["source_chain"]["repair_commit"] = "0" * 40
    errors = dependencies._m44_source_chain_errors(
        REPO_ROOT, materializer, tampered
    )
    assert len(errors) == 1
    assert "M44 exact repair/control source chain differs" in errors[0]


def test_m43_authority_binds_stopped_generation_and_preserves_m42() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m43_authority_test",
    )
    authority = (
        materializer
        ._expected_m43_dead_attempt_lifecycle_recovery_restart_authority()
    )
    reference = materializer._m43_authority_reference()
    prior = authority["prior_authority"]
    stopped = authority["stopped_owner"]
    artifacts = authority["stopped_prestart_artifacts"]
    receipt = authority["preserved_m42_receipt"]
    repair = authority["accepted_lifecycle_repair"]
    failed_validation = authority["failed_pre_authoritative_control_validation"]
    fixture_repair = authority["accepted_historical_fixture_repair"]
    failed_reseal = authority["failed_pre_authoritative_reseal_validation"]
    verifier_repair = authority[
        "accepted_bounded_materializer_verifier_repair"
    ]
    revision3 = authority["prior_revision3_authority"]
    validation_attempts = authority["revision4_validation_attempts"]
    successor_evidence_repair = authority[
        "accepted_successor_evidence_verifier_repair"
    ]
    prior_control = authority["prior_control_authorization"]
    revision2_control = revision3["prior_control_authorization"]
    source_chain = authority["source_chain"]
    changes = authority["exact_changes"]

    assert reference == {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M43",
        "authority_cid": (
            "sha256:6ddc11cb9e37da82023e5a89124298f532cfc943cc347fa5ea67ff13bcd1eb43"
        ),
    }
    assert authority["schema"].endswith("authorization@4")
    assert authority["authorization_revision"] == 4
    assert authority["control_recorded_at"] == "2026-09-01T11:00:00Z"
    assert authority["authorization_amended_at"] == "2026-09-01T11:00:00Z"
    assert authority["prior_authorization_cid"] == (
        "sha256:1c1c16ed284a0176a2244bddd79847934ba6a369e9ab46c674a109853d2d67d8"
    )
    assert prior_control == {
        "authorization_cid": authority["prior_authorization_cid"],
        "authorization_revision": 3,
        "prior_authorization_cid": (
            "sha256:f0db2f708316ad8ef58cb78886b5df74872d80c6147a1ed6e1d51faeeac35049"
        ),
        "control_recorded_at": "2026-09-01T10:00:00Z",
        "control_commit": "8c22185c66f2250732071e789e69a0cd42cf9a7d",
        "control_tree": "754e49e85310f96d6709f515328c27ce7bc9b43f",
        "superseded_before_authoritative_event_297": True,
        "event_297_appended": False,
        "receipt_published": False,
    }
    assert materializer._identity(revision3) == authority[
        "prior_authorization_cid"
    ]
    assert revision3 == materializer._expected_m43_revision3_authority()
    assert revision3["schema"].endswith("authorization@3")
    assert revision3["authorization_revision"] == 3
    assert revision3["control_recorded_at"] == "2026-09-01T10:00:00Z"
    assert revision3["authorization_amended_at"] == "2026-09-01T10:00:00Z"
    assert revision3["prior_authorization_cid"] == (
        "sha256:f0db2f708316ad8ef58cb78886b5df74872d80c6147a1ed6e1d51faeeac35049"
    )
    assert revision2_control == {
        "authorization_cid": revision3["prior_authorization_cid"],
        "authorization_revision": 2,
        "prior_authorization_cid": materializer._M43_PRE_AUTHORITATIVE_AUTHORITY_CID,
        "control_recorded_at": "2026-09-01T09:00:00Z",
        "control_commit": "d693f82660603adc56f3c09429d3d578225d7fb9",
        "control_tree": "6cc740dbfcac0d9a8004d3a261410a2b63553ea9",
        "superseded_before_event_297": True,
        "event_297_appended": False,
        "receipt_published": False,
    }
    assert authority["authorization_amendment_paths"] == sorted(
        materializer._M43_OPERATOR_CONTROL_PATHS
    )
    assert authority["target_generation"] == 31
    assert authority["target_event_watermark"] == 297
    assert authority["target_projection_cid"] == (
        "baguqeerazspjonqzwhd5e2jmnpl4lacaasfmtkziur4awfrihibnh6mxkpoa"
    )
    assert prior == {
        **prior,
        "event_watermark": 296,
        "event_prefix_sha256": (
            "89c64a4f018c2a3cfdce675eb8fb27913674e76995d64d89cabec42dd2967b70"
        ),
        "projection_cid": (
            "baguqeerasguaepwupk3d5vme3cqsbicujihwvxnemnoenvdtnu3uwrscbt6q"
        ),
        "semantic_authority_digest": (
            "sha256:a9f7e45d543cd983b36d475f3145344c956c547524bf33adfdc630de2bda7ae0"
        ),
    }
    assert stopped["generation"] == 30
    assert stopped["target_generation"] == 31
    assert stopped["status"] == "stopped"
    assert stopped["revision"] == 2
    assert stopped["database_uuid"] == "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4"
    assert artifacts["control_sha256"] == (
        "6798563648545b3fc05f1b7638ad2d0448c743d3a788bf78208a5d28a76a95f7"
    )
    assert artifacts["coordination_sha256"] == (
        "ddbdf352e6a41452c6584cfa06fc760b90a94f1ff6473ff2c5eeb93de7551785"
    )
    assert artifacts["status_sha256"] == (
        "3f8c1227e7bc29c3057d238e550880cdfb6144a3687a73dea12e3ca063148a4c"
    )
    assert all(
        artifacts[name] is True
        for name in (
            "owner_marker_absent",
            "stop_control_absent",
            "token_handoff_absent",
            "control_wal_absent",
            "coordination_wal_absent",
        )
    )
    assert receipt["sha256"] == (
        "ff9a24d339cf06eacb3573cd2825e0648a558efe5ec9539c0c4f489002ca609d"
    )
    assert receipt["size"] == 14_112
    assert receipt["receipt_cid"] == (
        "sha256:31565b6bfc8e071f4278acc88fd3500ca5c4d25eee63d0131b16ceca3e7a9169"
    )
    assert receipt["created_or_rewritten"] is False
    assert set(repair["changed_paths"]) == {
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py",
        "test/api/test_agent_supervisor_database_portal_bridge.py",
        "test/api/test_agent_supervisor_configured_board_live_capsule.py",
    }
    assert repair["repair_parent"] == (
        "a8bce148b793dcd15ac742df3a29e5773a178f28"
    )
    assert repair["repair_commit"] == (
        "7e3fa1170edac23149e0d1f38f5ff6b5f5ddb571"
    )
    assert repair["repair_tree"] == (
        "df40d6f879753c8c2ca00f35fd28054a29fd600a"
    )
    assert repair["blob_oids"] == {
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py": (
            "97f44d032063a8a98cfca277dd123c998244f076"
        ),
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py": (
            "51e577170d975f056e3f97a9ded55d210bee8792"
        ),
        "test/api/test_agent_supervisor_database_portal_bridge.py": (
            "07fb7dd738256126c96e1370dd5a60501a13647b"
        ),
        "test/api/test_agent_supervisor_configured_board_live_capsule.py": (
            "4638112c6c2da24cb5912914192f332f2903bd6b"
        ),
    }
    assert repair["path_modes"] == {
        path: "100644" for path in repair["changed_paths"]
    }
    assert source_chain["initial_control_parent"] == repair["repair_commit"]
    assert source_chain["initial_control_commit"] == (
        "21c2a72f0e9d23d86ac990d4320cf1d80a05a044"
    )
    assert source_chain["initial_control_tree"] == (
        "3efca4759e083b1615b6ac092e0b40054f233452"
    )
    assert source_chain["initial_control_blobs"] == {
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json": (
            "2c3cdbe4a307b9309ba3a1e09e78805a2e738437"
        ),
        "config/semantic_addressed_world_model_dependencies.seal.json": (
            "f18b018ba2879ee8899737d830db6528840a3cd8"
        ),
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md": (
            "25047b1171688d4973dbd3adad2ad921c5550fe8"
        ),
        "docs/architecture/semantic_addressed_world_model_inventory/"
        "prior_materialization_migration.json": (
            "6bf831d39995cdc84828f12535efe41f728c2cb9"
        ),
        "scripts/materialize_semantic_addressed_world_model_program.py": (
            "4bd399f7cfdc88b6f9b5b5a407b7d41eb069322e"
        ),
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py": (
            "ea243c8e888869960ce2805601604d4aa51f472b"
        ),
        "scripts/validate_semantic_addressed_world_model_board.py": (
            "db26d8bf21f687c944baa83192a824331f1e1892"
        ),
        "scripts/validate_semantic_addressed_world_model_dependencies.py": (
            "f1c654a3205370ea71704c752a3fce2e3198c332"
        ),
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py": (
            "f3676283ea67f2aab68e667d5b2f407a7106444f"
        ),
    }
    assert source_chain["initial_control_modes"] == {
        path: "100644" for path in source_chain["initial_control_blobs"]
    }
    assert failed_validation == {
        **failed_validation,
        "authority_cid": materializer._M43_PRE_AUTHORITATIVE_AUTHORITY_CID,
        "control_commit": "f2f4bafe8952b6fe25d93c1dc145c0298234e2c2",
        "control_tree": "4c49141b84d714954ca5410dc7fbc807f82be4eb",
        "collected_tests": 280,
        "passed_tests": 271,
        "failed_tests": 9,
        "failed_test_ids": list(materializer._M43_FAILED_TEST_IDS),
        "production_runtime_defect": False,
        "materializer_invoked": False,
        "quack_started": False,
        "authenticated_mutation_request_created": False,
        "event_297_rows_created": 0,
        "m43_receipt_created": False,
        "accepted_completion_changes": 0,
        "worker_self_approval": False,
    }
    assert failed_validation["control_blobs"] == {
        path: materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_BLOBS[path]
        for path in materializer._M43_OPERATOR_CONTROL_PATHS
    }
    assert all(
        failed_validation[name] == 0
        for name in (
            "evidence_node_changes",
            "evidence_event_changes",
            "validation_event_changes",
            "store_generation_row_changes",
            "state_server_row_changes",
            "credential_row_changes",
            "server_epoch_row_changes",
            "capability_snapshot_row_changes",
            "task_revision_changes",
            "task_status_changes",
            "goal_revision_changes",
            "plan_revision_changes",
            "effect_claim_changes",
            "merge_attempt_changes",
            "implementation_provider_invocations",
            "coordination_semantic_changes",
            "accepted_completion_changes",
        )
    )
    assert fixture_repair == {
        **fixture_repair,
        "repair_parent": failed_validation["control_commit"],
        "repair_commit": "7361c38dfdeaa6578876f23c431593942accc74f",
        "repair_tree": "a25f9f9e1b2b617012a46799b369b3ed09a97728",
        "changed_paths": [
            "test/api/semantic_world/"
            "test_semantic_addressed_world_model_board.py"
        ],
        "focused_former_failures_replayed": 9,
        "focused_former_failures_passed": 9,
        "focused_replay_is_full_suite": False,
        "production_code_changes": 0,
        "fixture_collection_or_selection_logic_changed": False,
        "validation_weakened": False,
        "authority_weakened": False,
        "worker_self_approval": False,
    }
    assert fixture_repair["blob_oids"] == {
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py": (
            "cf9e119747ff01835a2cf70565052cc77d6e402a"
        )
    }
    assert failed_reseal == {
        **failed_reseal,
        "authority_cid": revision3["prior_authorization_cid"],
        "control_commit": revision2_control["control_commit"],
        "control_tree": revision2_control["control_tree"],
        "observed_failure_count": 2,
        "authoritative_materializer_invoked": False,
        "authoritative_quack_started": False,
        "authenticated_mutation_request_created": False,
        "event_297_rows_created": 0,
        "m43_receipt_created": False,
        "accepted_completion_changes": 0,
        "worker_self_approval": False,
    }
    assert failed_reseal["isolated_live_rehearsal"] == {
        **failed_reseal["isolated_live_rehearsal"],
        "source_head": revision2_control["control_commit"],
        "source_tree": revision2_control["control_tree"],
        "prestart_admission_valid": True,
        "disposable_generation_31_started": True,
        "disposable_generation_31_ready": True,
        "disposable_generation_31_stopped": True,
        "event_head_before": 296,
        "event_head_after": 296,
        "m43_receipt_created": False,
        "failure_kind": "quack_duckdb_row_mapping_name_iteration",
        "failure_site": "_m43_operational_suffix_on",
        "exception_class": "JSONDecodeError",
        "authoritative_runtime_used": False,
    }
    assert failed_reseal["full_suite_validation"] == {
        **failed_reseal["full_suite_validation"],
        "collected_tests": 280,
        "passed_tests": 279,
        "failed_tests": 1,
        "failure_kind": "task_revision_event_timestamp_source_mismatch",
        "error": "MigrationRequired: operator task-recovery event differs",
        "retained_log_sha256": (
            "1a1fd7acb5c7b09057d5971434d1bba0fc4007c744706037eebec10e7d155ce4"
        ),
        "retained_log_size": 383_803,
        "retained_log_mode": "0600",
        "retained_log_is_authority": False,
    }
    assert failed_reseal["full_suite_validation"]["failed_test_ids"] == [
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py::"
        "test_append_only_source_migration_rehearsal_verifies_exactly"
    ]
    assert failed_reseal["full_suite_validation"]["retained_log_path"] == (
        "/tmp/m43-semantic-world-pytest.NAHHWQ.log"
    )
    assert all(
        failed_reseal[name] == 0
        for name in (
            "evidence_node_changes",
            "evidence_event_changes",
            "validation_event_changes",
            "store_generation_row_changes",
            "state_server_row_changes",
            "credential_row_changes",
            "server_epoch_row_changes",
            "capability_snapshot_row_changes",
            "task_revision_changes",
            "task_status_changes",
            "goal_revision_changes",
            "plan_revision_changes",
            "coordination_semantic_changes",
            "effect_claim_changes",
            "merge_attempt_changes",
            "implementation_provider_invocations",
            "accepted_completion_changes",
        )
    )
    assert verifier_repair == {
        **verifier_repair,
        "repair_parent": revision2_control["control_commit"],
        "repair_commit": "c5ca423d74e27fe9156c4ce9b476851f565cb7cd",
        "repair_tree": "502eeb8cb5be1ea926df40975486823879e8c3b9",
        "repair_class_count": 2,
        "repair_site_count": 5,
        "expected_values_changed": False,
        "lifecycle_checks_weakened": False,
        "validation_weakened": False,
        "authority_weakened": False,
        "database_mutations": 0,
        "event_append_changes": 0,
        "accepted_completion_changes": 0,
        "m43_duckdbrow_regression_passed": True,
        "historical_rehearsal_passed": True,
        "worker_self_approval": False,
    }
    assert verifier_repair["blob_oids"] == {
        "scripts/materialize_semantic_addressed_world_model_program.py": (
            "775c02038fbd75f0eab3f3c3fdb0ef9fdeb78132"
        ),
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py": (
            "8a98a5818b88c3195d5ed8a75133a19a93c92940"
        ),
    }
    assert verifier_repair["path_modes"] == {
        path: "100644" for path in verifier_repair["blob_oids"]
    }
    assert verifier_repair["repair_classes"] == [
        "duckdb_positional_row_normalization",
        "task_revision_timestamp_source_binding",
    ]
    assert verifier_repair["repair_sites"] == [
        "m43_operational_suffix_rows",
        "m43_prestart_event_type_counts",
        "m43_live_preappend_event_type_counts",
        "m43_live_postappend_event_type_counts",
        "m6_recovery_revision_recorded_at",
    ]
    assert validation_attempts == materializer._m43_revision4_validation_attempts()
    assert validation_attempts["schema"] == (
        "sawm/m43-revision-4-validation-attempts@1"
    )
    assert validation_attempts["worker_self_approval"] is False
    full_suite, disposable = validation_attempts["attempts"]
    assert full_suite == {
        **full_suite,
        "ordinal": 1,
        "kind": "authoritative_current_tree_full_suite",
        "source_commit": "8c22185c66f2250732071e789e69a0cd42cf9a7d",
        "source_tree": "754e49e85310f96d6709f515328c27ce7bc9b43f",
        "result": "passed",
        "collected": 280,
        "passed": 280,
        "failed": 0,
        "duration_seconds": "546.92",
        "python": "3.12.3",
        "pytest": "9.1.1",
        "completed_at_from_log_mtime": "2026-09-01T10:28:56.012371110Z",
        "authoritative_state_changes": 0,
        "task_status_changes": 0,
        "goal_revision_changes": 0,
        "provider_invocation_changes": 0,
        "merge_attempt_changes": 0,
    }
    assert full_suite["retained_log"] == {
        "authority": "non_authoritative_retained_validation_log",
        "path": "/tmp/m43-semantic-world-pytest.SFTF0v.log",
        "sha256": (
            "c2004ef842570e2f2345caff606efed078842e71725a1d942e2e93ed4a80034d"
        ),
        "size_bytes": 2_674_942,
        "mode": "0600",
    }
    assert disposable == {
        **disposable,
        "ordinal": 2,
        "kind": "fresh_disposable_clone_partial_append_rehearsal",
        "disposable": True,
        "authoritative": False,
        "source_commit": "8c22185c66f2250732071e789e69a0cd42cf9a7d",
        "source_tree": "754e49e85310f96d6709f515328c27ce7bc9b43f",
        "source_binding_cid": (
            "sha256:baa52e8efe025b0c3e8cbafc2ac0bdf3f64cc630e4e12cecdd12259adda88f67"
        ),
        "authorization_cid": authority["prior_authorization_cid"],
        "validation_digest": (
            "sha256:583d4480426d92035848a83d730c8ca0ab867f1f8f1ac7cd332240313a97d2d1"
        ),
        "result": "failed_after_authenticated_append_before_receipt",
        "typed_error": (
            "MigrationRequired: M42 exact evidence projection membership differs"
        ),
        "successful_append_observed_at": "2026-09-01T10:34:22.338Z",
        "stopped_at": "2026-09-01T10:34:42Z",
        "post_stop_lifecycle": "stopped",
        "post_stop_generation": 31,
        "event_changes": 1,
        "evidence_node_changes": 1,
        "task_status_changes": 0,
        "task_revision_changes": 0,
        "goal_revision_changes": 0,
        "accepted_completion_changes": 0,
        "provider_invocation_changes": 0,
        "merge_attempt_changes": 0,
        "owner_marker_present_after_stop": False,
        "credential_handoff_present_after_stop": False,
    }
    assert disposable["post_control_store"] == {
        "sha256": (
            "22fe8e92031317551c0b036d1c2a49ccc927c9536a987e57ac3ca7e6bb0720b8"
        ),
        "size_bytes": 43_528_192,
        "mode": "0664",
    }
    assert disposable["post_status"] == {
        "sha256": (
            "c339f4d048ea54cf1e3a47df0095860b31665da2ce07040ca9109c659988f765"
        ),
        "size_bytes": 2_257,
        "mode": "0600",
        "lifecycle": "stopped",
        "identity_status": "stopped",
        "generation": 31,
    }
    assert disposable["mutation_result"] == {
        "operation": "evidence_record@1",
        "ok": True,
        "rowcounts": [0, 1, 1],
        "sha256": (
            "03adb834891d0e72a410d3641bb21501c98ae59d090b1ff176d73d66bd1f3a07"
        ),
        "size_bytes": 2_285,
        "mode": "0600",
    }
    assert disposable["partial_append"] == {
        "event_watermark_before": 296,
        "event_watermark_after": 297,
        "evidence_count_before": 49,
        "evidence_count_after": 50,
        "event_id": (
            "baguqeerach3itwdabnwmogqhgwqxadvbu536nfgm2frptrk3oo7kqdjm3moa"
        ),
        "evidence_id": (
            "baguqeeray4agdk7scb6l43obebi535l5estjwswf5eifgawz5clhfgyvapsq"
        ),
        "event_rehashed": True,
        "evidence_rehashed": True,
        "event_evidence_binding_verified": True,
        "m43_receipt_present": False,
    }
    assert validation_attempts["authoritative_anchors_after_both_attempts"] == {
        "unchanged": True,
        "control_store": {
            "sha256": (
                "6798563648545b3fc05f1b7638ad2d0448c743d3a788bf78208a5d28a76a95f7"
            ),
            "size_bytes": 43_528_192,
        },
        "coordination_store": {
            "sha256": (
                "ddbdf352e6a41452c6584cfa06fc760b90a94f1ff6473ff2c5eeb93de7551785"
            ),
            "size_bytes": 16_789_504,
        },
        "m42_receipt": {
            "sha256": (
                "ff9a24d339cf06eacb3573cd2825e0648a558efe5ec9539c0c4f489002ca609d"
            ),
            "size_bytes": 14_112,
        },
        "stopped_status": {
            "sha256": (
                "3f8c1227e7bc29c3057d238e550880cdfb6144a3687a73dea12e3ca063148a4c"
            ),
            "size_bytes": 2_408,
        },
        "authoritative_event_watermark": 296,
        "authoritative_evidence_count": 49,
        "authoritative_generation": 30,
    }
    assert successor_evidence_repair == (
        materializer._m43_successor_evidence_verifier_repair()
    )
    assert successor_evidence_repair == {
        **successor_evidence_repair,
        "schema": "sawm/bounded-successor-evidence-verifier-repair@1",
        "repair_parent": prior_control["control_commit"],
        "repair_commit": "93c806cecb6c4929acdc8ccd2702f11d72585ffc",
        "repair_tree": "9d852330f8c100450889f055e9e58b94e4b35294",
        "failure_kind": "post_append_predecessor_whole_table_scope_mismatch",
        "permitted_successor_evidence_row_count": 1,
        "successor_row_is_caller_bound": True,
        "successor_row_content_identity_rehashed": True,
        "successor_row_canonical_body_required": True,
        "successor_row_collision_rejected": True,
        "unlisted_extra_evidence_rejected": True,
        "wrong_successor_evidence_rejected": True,
        "preappend_default_behavior_changed": False,
        "m42_return_fields_changed": False,
        "m42_return_counts_changed": False,
        "receipt_schema_weakened": False,
        "authority_weakened": False,
        "database_mutations": 0,
        "event_append_changes": 0,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_completion_changes": 0,
        "worker_self_approval": False,
    }
    assert successor_evidence_repair["blob_oids"] == {
        "scripts/materialize_semantic_addressed_world_model_program.py": (
            "db4da6dcbf979972b0bc6564534ae15d6f2708b5"
        ),
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py": (
            "2b412f977c1cfe19acbb15c96b73b35a44306252"
        ),
    }
    assert successor_evidence_repair["changed_paths"] == sorted(
        successor_evidence_repair["blob_oids"]
    )
    assert successor_evidence_repair["path_modes"] == {
        path: "100644" for path in successor_evidence_repair["blob_oids"]
    }
    assert source_chain["failed_pre_authoritative_control_parent"] == (
        source_chain["initial_control_commit"]
    )
    assert source_chain["failed_pre_authoritative_control_commit"] == (
        failed_validation["control_commit"]
    )
    assert source_chain["failed_pre_authoritative_control_tree"] == (
        failed_validation["control_tree"]
    )
    assert source_chain["historical_fixture_repair_parent"] == (
        failed_validation["control_commit"]
    )
    assert source_chain["historical_fixture_repair_commit"] == (
        fixture_repair["repair_commit"]
    )
    assert source_chain["historical_fixture_repair_tree"] == (
        fixture_repair["repair_tree"]
    )
    assert source_chain["prior_final_control_parent"] == fixture_repair[
        "repair_commit"
    ]
    assert source_chain["prior_final_control_commit"] == revision2_control[
        "control_commit"
    ]
    assert source_chain["prior_final_control_tree"] == revision2_control[
        "control_tree"
    ]
    assert source_chain["prior_final_control_blobs"] == dict(
        materializer._M43_PRIOR_FINAL_CONTROL_BLOBS
    )
    assert source_chain["prior_final_control_modes"] == dict(
        materializer._M43_PRIOR_FINAL_CONTROL_MODES
    )
    assert source_chain["bounded_materializer_verifier_repair_parent"] == (
        revision2_control["control_commit"]
    )
    assert source_chain["bounded_materializer_verifier_repair_commit"] == (
        verifier_repair["repair_commit"]
    )
    assert source_chain["bounded_materializer_verifier_repair_tree"] == (
        verifier_repair["repair_tree"]
    )
    assert source_chain["bounded_materializer_verifier_repair_blobs"] == dict(
        materializer._M43_VERIFIER_REPAIR_BLOBS
    )
    assert source_chain["bounded_materializer_verifier_repair_modes"] == dict(
        materializer._M43_VERIFIER_REPAIR_MODES
    )
    assert source_chain["revision3_final_control_parent"] == verifier_repair[
        "repair_commit"
    ]
    assert source_chain["revision3_final_control_commit"] == prior_control[
        "control_commit"
    ]
    assert source_chain["revision3_final_control_tree"] == prior_control[
        "control_tree"
    ]
    assert source_chain["revision3_final_control_blobs"] == {
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json": (
            "176952af29363272dd4d597c1e8961301abafe21"
        ),
        "config/semantic_addressed_world_model_dependencies.seal.json": (
            "3fec38ba8c0cd960cfb3379158ee03961b12218b"
        ),
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md": (
            "547538605c1e6f43b11082e7f9a1593ab240bbba"
        ),
        "docs/architecture/semantic_addressed_world_model_inventory/"
        "prior_materialization_migration.json": (
            "7f0a2e3c17cd90861f5e7cb924a3ebf236b7e1d4"
        ),
        "scripts/materialize_semantic_addressed_world_model_program.py": (
            "8a87b56ecadeed8a2b780bcee0ab3f26fed62780"
        ),
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py": (
            "128c70ea5193d2fb5c652606884034c3aafdad8f"
        ),
        "scripts/validate_semantic_addressed_world_model_board.py": (
            "5e9ad70bba6d67aa2985bc4907149c4986538511"
        ),
        "scripts/validate_semantic_addressed_world_model_dependencies.py": (
            "456f7da65356d929a580b1117e06ce6686ff58d7"
        ),
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py": (
            "a01f3d160ea379c3d3abe1b7b4e237281b123b12"
        ),
    }
    assert source_chain["revision3_final_control_modes"] == {
        path: "100644"
        for path in source_chain["revision3_final_control_blobs"]
    }
    assert source_chain["successor_evidence_verifier_repair_parent"] == (
        prior_control["control_commit"]
    )
    assert source_chain["successor_evidence_verifier_repair_commit"] == (
        successor_evidence_repair["repair_commit"]
    )
    assert source_chain["successor_evidence_verifier_repair_tree"] == (
        successor_evidence_repair["repair_tree"]
    )
    assert source_chain["successor_evidence_verifier_repair_blobs"] == (
        successor_evidence_repair["blob_oids"]
    )
    assert source_chain["successor_evidence_verifier_repair_modes"] == (
        successor_evidence_repair["path_modes"]
    )
    assert source_chain["final_reseal_parent"] == successor_evidence_repair[
        "repair_commit"
    ]
    assert source_chain["failed_pre_authoritative_control_commit_count"] == 2
    assert source_chain["historical_fixture_repair_commit_count"] == 1
    assert source_chain["bounded_materializer_verifier_repair_commit_count"] == 1
    assert source_chain["revision3_final_control_commit_count"] == 1
    assert source_chain["successor_evidence_verifier_repair_commit_count"] == 1
    assert source_chain["failed_disposable_partial_append_count"] == 1
    assert source_chain["final_reseal_commit_count"] == 3
    assert source_chain["final_control_commit_count"] == 4
    assert not any(
        name in authority
        for name in (
            "final_control_commit",
            "final_control_tree",
            "final_control_blobs",
        )
    )
    hardening = authority["final_control_hardening"]
    assert hardening["operational_suffix_malformed_rows_are_typed_conflicts"] is True
    assert hardening["event_count_malformed_rows_are_typed_conflicts"] is True
    assert hardening["event_count_non_integer_values_are_typed_conflicts"] is True
    assert hardening["pending_authority_rejected_by_operator_facade"] is True
    assert hardening["current_commit_identity_embedded_in_authority"] is False
    assert hardening["current_tree_identity_embedded_in_authority"] is False
    assert hardening["current_blob_identities_embedded_in_authority"] is False
    assert hardening["schema"].endswith("hardening@2")
    assert hardening[
        "exact_successor_evidence_is_the_only_postappend_allowance"
    ] is True
    assert hardening["m42_return_contract_preserved"] is True
    assert hardening["unlisted_physical_evidence_still_fails_closed"] is True
    assert hardening["authority_weakened"] is False
    assert 'restart-source-seal@4' in inspect.getsource(
        materializer._m43_migration_body
    )
    assert 'restart-receipt@4' in inspect.getsource(
        materializer._expected_m43_source_successor_receipt
    )
    assert not any(
        name in source_chain
        for name in (
            "final_control_commit",
            "final_control_tree",
            "final_control_blobs",
        )
    )
    assert authority["ordinary_source_changes"] == len(repair["changed_paths"])
    assert repair["repair_path_set_finalized"] is True
    assert repair["newest_first_attempt_selection"] is True
    assert repair["prepared_receipt_precedes_lifecycle_cas"] is True
    assert repair["marker_retirement_is_no_replace"] is True
    assert changes["task_status_changes"] == 0
    assert changes["accepted_completion_changes"] == 0
    preservation = authority["preservation"]
    assert preservation["revision3_authority_preserved_exactly"] is True
    assert preservation["revision3_full_suite_evidence_preserved"] is True
    assert preservation[
        "disposable_partial_append_preserved_as_non_authoritative"
    ] is True
    assert preservation["authoritative_event_297_remained_absent"] is True
    assert preservation[
        "successor_evidence_verifier_repair_changed_no_task_authority"
    ] is True
    assert materializer._validated_m43_live_preflight_contract(authority) == (
        authority["live_preflight_contract"]
    )
    assert materializer._identity(authority["prior_m42_authority"]) == (
        materializer._M43_M42_AUTHORITY_CID
    )
    assert materializer._identity(authority) == reference["authority_cid"]

    tampered_prior = copy.deepcopy(authority)
    tampered_prior["prior_control_authorization"]["event_297_appended"] = True
    with pytest.raises(
        materializer.MaterializationError, match="M43 live preflight contract differs"
    ):
        materializer._validated_m43_live_preflight_contract(tampered_prior)

    tampered_revision3 = copy.deepcopy(authority)
    tampered_revision3["prior_revision3_authority"]["authorized"] = False
    with pytest.raises(
        materializer.MaterializationError, match="M43 live preflight contract differs"
    ):
        materializer._validated_m43_live_preflight_contract(tampered_revision3)

    tampered_validation_attempt = copy.deepcopy(authority)
    tampered_validation_attempt["revision4_validation_attempts"]["attempts"][
        0
    ]["passed"] = 279
    with pytest.raises(
        materializer.MaterializationError, match="M43 live preflight contract differs"
    ):
        materializer._validated_m43_live_preflight_contract(
            tampered_validation_attempt
        )

    tampered_disposable_authority = copy.deepcopy(authority)
    tampered_disposable_authority["revision4_validation_attempts"][
        "authoritative_anchors_after_both_attempts"
    ]["authoritative_event_watermark"] = 297
    with pytest.raises(
        materializer.MaterializationError, match="M43 live preflight contract differs"
    ):
        materializer._validated_m43_live_preflight_contract(
            tampered_disposable_authority
        )

    tampered_successor_repair = copy.deepcopy(authority)
    tampered_successor_repair[
        "accepted_successor_evidence_verifier_repair"
    ]["unlisted_extra_evidence_rejected"] = False
    with pytest.raises(
        materializer.MaterializationError, match="M43 live preflight contract differs"
    ):
        materializer._validated_m43_live_preflight_contract(
            tampered_successor_repair
        )

    tampered_repair = copy.deepcopy(authority)
    tampered_repair["accepted_historical_fixture_repair"]["blob_oids"] = {}
    with pytest.raises(
        materializer.MaterializationError, match="M43 live preflight contract differs"
    ):
        materializer._validated_m43_live_preflight_contract(tampered_repair)

    tampered_zero_effect = copy.deepcopy(authority)
    tampered_zero_effect["failed_pre_authoritative_control_validation"][
        "event_297_rows_created"
    ] = 1
    with pytest.raises(
        materializer.MaterializationError, match="M43 live preflight contract differs"
    ):
        materializer._validated_m43_live_preflight_contract(tampered_zero_effect)

    tampered_count = copy.deepcopy(authority)
    tampered_count["source_chain"]["final_control_commit_count"] = 1
    with pytest.raises(
        materializer.MaterializationError, match="M43 live preflight contract differs"
    ):
        materializer._validated_m43_live_preflight_contract(tampered_count)

    tampered_revision3_chain = copy.deepcopy(authority)
    tampered_revision3_chain["source_chain"][
        "revision3_final_control_commit"
    ] = "0" * 40
    with pytest.raises(
        materializer.MaterializationError, match="M43 live preflight contract differs"
    ):
        materializer._validated_m43_live_preflight_contract(
            tampered_revision3_chain
        )

    tampered_reseal_failure = copy.deepcopy(authority)
    tampered_reseal_failure["failed_pre_authoritative_reseal_validation"][
        "event_297_rows_created"
    ] = 1
    with pytest.raises(
        materializer.MaterializationError, match="M43 live preflight contract differs"
    ):
        materializer._validated_m43_live_preflight_contract(
            tampered_reseal_failure
        )

    tampered_verifier_repair = copy.deepcopy(authority)
    tampered_verifier_repair[
        "accepted_bounded_materializer_verifier_repair"
    ]["validation_weakened"] = True
    with pytest.raises(
        materializer.MaterializationError, match="M43 live preflight contract differs"
    ):
        materializer._validated_m43_live_preflight_contract(
            tampered_verifier_repair
        )

    tampered_hardening = copy.deepcopy(authority)
    tampered_hardening["final_control_hardening"][
        "pending_authority_rejected_by_operator_facade"
    ] = False
    with pytest.raises(
        materializer.MaterializationError, match="M43 live preflight contract differs"
    ):
        materializer._validated_m43_live_preflight_contract(tampered_hardening)


def test_m43_current_seal_rejects_a_synthetic_unsealed_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m43_synthetic_unsealed_test",
    )
    authority = (
        materializer
        ._expected_m43_dead_attempt_lifecycle_recovery_restart_authority()
    )
    monkeypatch.setattr(
        materializer,
        "_M43_AUTHORITY_CID",
        "sha256:PENDING_M43_FINAL_CONTROL_AUTHORITY_CID",
    )
    with pytest.raises(
        materializer.MaterializationError,
        match="M43 final control identities are not resealed",
    ):
        materializer._assert_m43_source_delta(
            REPO_ROOT, {"source_binding": {"head": "unused"}}, authority
        )

    control_path = sorted(materializer._M43_OPERATOR_CONTROL_PATHS)[0]
    monkeypatch.setattr(
        materializer,
        "_git",
        lambda *_args: f"100755 blob {'0' * 40}\t{control_path}",
    )
    with pytest.raises(
        materializer.MaterializationError,
        match="M43 final-control mode differs",
    ):
        materializer._assert_m43_current_control_modes(
            REPO_ROOT, "synthetic-current", (control_path,)
        )


def test_m43_dispatch_is_newest_and_receipt_follows_live_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m43_dispatch_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m43_dispatch_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    config, _migration, _seal = _historical_successor_controls_at(
        "dead_attempt_lifecycle_recovery_restart_successor_materialization",
        config,
    )
    config["database_program"]["store_generation"] = "31"
    active = operator._active_source_repair_materialization(config)
    assert active["migration_revision"] == "SAWM-R2-M43"
    assert active["target_generation"] == 31
    assert active["target_event_watermark"] == 297

    for function in (materializer.check_materialized, materializer.materialize):
        source = inspect.getsource(function)
        assert source.index("_m43_successor_configured_on_any_surface") < (
            source.index("_m42_successor_configured_on_any_surface")
        )
    main_source = inspect.getsource(materializer.main)
    assert main_source.index(
        "dead_attempt_lifecycle_recovery_restart_successor_materialization"
    ) < main_source.index(
        "failed_pre_authoritative_m41_evidence_projection_successor_materialization"
    )
    core = inspect.getsource(materializer._materialize_m43)
    assert core.index("_verify_m43_preserved_m42_receipt") < core.index(
        "source.record_evidence"
    )
    assert core.index("_verify_m43_live_materialization") < core.index(
        "_ensure_m43_source_successor_receipt"
    )
    assert "snapshot.event_cursor == _M43_PRIOR_EVENT_WATERMARK" in core
    assert "snapshot.event_cursor != _M43_TARGET_EVENT_WATERMARK" in core
    live_verifier = inspect.getsource(
        materializer._verify_m43_live_materialization
    )
    assert (
        "expected_successor_evidence_row=expected_evidence" in live_verifier
    )

    configured = inspect.getsource(operator._successor_materialization_configured)
    assert configured.index("_M43_SUCCESSOR_KEY") < configured.index(
        "_M42_SUCCESSOR_KEY"
    )
    normalized = inspect.getsource(operator._normalized_live_preflight_contract)
    assert normalized.index("_M43_MIGRATION_REVISION") < normalized.index(
        "_M42_MIGRATION_REVISION"
    )
    preflight = inspect.getsource(operator._live_preflight)
    assert "m43_active = active_revision == _M43_MIGRATION_REVISION" in preflight
    assert preflight.index("m43_active,") < preflight.index("m42_active,")
    assert preflight.index("_verify_m43_live_materialization") < preflight.index(
        "_verify_m42_live_materialization"
    )
    marker = inspect.getsource(operator._require_active_final_pair_marker)
    assert marker.index("_M43_SUCCESSOR_KEY") < marker.index("_M42_SUCCESSOR_KEY")
    offline = inspect.getsource(operator._validate_offline_quack_start)
    assert offline.index("_M43_SUCCESSOR_KEY") < offline.index(
        "_M42_SUCCESSOR_KEY"
    )

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBRow,
    )

    event_body = json.dumps(
        {
            "body": {
                "receipt": {
                    "attempt_id": "attempt-1",
                    "claim_id": "claim-1",
                    "lease_id": "lease-1",
                    "operation": "attempt_lifecycle_recovered",
                },
                "revision": 11,
                "status": "in_progress",
                "task_alias": "SAWM-006",
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    quack_row = DuckDBRow(
        ("event_id", "global_sequence", "body_json"),
        ("event-293", 293, event_body),
    )
    cursor = SimpleNamespace(fetchall=lambda: [quack_row])
    connection = SimpleNamespace(execute=lambda *_args: cursor)
    assert materializer._m43_operational_suffix_on(connection) == [
        {
            "attempt_id": "attempt-1",
            "claim_id": "claim-1",
            "event_id": "event-293",
            "global_sequence": 293,
            "lease_id": "lease-1",
            "operation": "attempt_lifecycle_recovered",
            "revision": 11,
            "status": "in_progress",
            "task_alias": "SAWM-006",
        }
    ]
    count_row = DuckDBRow(
        ("evidence_event_count", "validation_event_count"),
        (18, 1),
    )
    count_cursor = SimpleNamespace(fetchone=lambda: count_row)
    count_connection = SimpleNamespace(execute=lambda *_args: count_cursor)
    assert materializer._m43_event_type_counts_on(
        count_connection, 296
    ) == (18, 1)
    short_row = DuckDBRow(("event_id", "global_sequence"), ("event-293", 293))
    short_cursor = SimpleNamespace(fetchall=lambda: [short_row])
    short_connection = SimpleNamespace(execute=lambda *_args: short_cursor)
    with pytest.raises(
        materializer.MigrationRequired,
        match="M43 post-M42 event row is malformed",
    ):
        materializer._m43_operational_suffix_on(short_connection)
    invalid_json_row = DuckDBRow(
        ("event_id", "global_sequence", "body_json"),
        ("event-293", 293, "not-json"),
    )
    invalid_json_cursor = SimpleNamespace(fetchall=lambda: [invalid_json_row])
    invalid_json_connection = SimpleNamespace(
        execute=lambda *_args: invalid_json_cursor
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="M43 post-M42 event row is malformed",
    ):
        materializer._m43_operational_suffix_on(invalid_json_connection)
    invalid_sequence_row = DuckDBRow(
        ("event_id", "global_sequence", "body_json"),
        ("event-293", "not-an-integer", event_body),
    )
    invalid_sequence_cursor = SimpleNamespace(
        fetchall=lambda: [invalid_sequence_row]
    )
    invalid_sequence_connection = SimpleNamespace(
        execute=lambda *_args: invalid_sequence_cursor
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="M43 post-M42 event row is malformed",
    ):
        materializer._m43_operational_suffix_on(invalid_sequence_connection)
    invalid_revision_body = json.dumps(
        {
            "body": {
                "receipt": {
                    "attempt_id": "attempt-1",
                    "claim_id": "claim-1",
                    "lease_id": "lease-1",
                    "operation": "attempt_lifecycle_recovered",
                },
                "revision": "not-an-integer",
                "status": "in_progress",
                "task_alias": "SAWM-006",
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    invalid_revision_row = DuckDBRow(
        ("event_id", "global_sequence", "body_json"),
        ("event-293", 293, invalid_revision_body),
    )
    invalid_revision_cursor = SimpleNamespace(
        fetchall=lambda: [invalid_revision_row]
    )
    invalid_revision_connection = SimpleNamespace(
        execute=lambda *_args: invalid_revision_cursor
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="M43 post-M42 event row is malformed",
    ):
        materializer._m43_operational_suffix_on(invalid_revision_connection)
    short_count_row = DuckDBRow(("evidence_event_count",), (18,))
    short_count_cursor = SimpleNamespace(fetchone=lambda: short_count_row)
    short_count_connection = SimpleNamespace(
        execute=lambda *_args: short_count_cursor
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="M43 event-type count row is malformed",
    ):
        materializer._m43_event_type_counts_on(short_count_connection, 296)
    non_integer_count_row = DuckDBRow(
        ("evidence_event_count", "validation_event_count"),
        (18, "1"),
    )
    non_integer_cursor = SimpleNamespace(fetchone=lambda: non_integer_count_row)
    non_integer_connection = SimpleNamespace(
        execute=lambda *_args: non_integer_cursor
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="M43 event-type count row differs",
    ):
        materializer._m43_event_type_counts_on(non_integer_connection, 296)
    assert "_m43_event_type_counts_on(" in inspect.getsource(
        materializer._materialize_m43
    )
    assert "_m43_event_type_counts_on(" in inspect.getsource(
        materializer._verify_m43_live_materialization
    )
    legacy_verifier = inspect.getsource(materializer._verify_store)
    assert "recovery_revision_rows = _positional_rows(" in legacy_verifier
    assert (
        '"recorded_at": str(recovery_revision_rows[0][1])'
        in legacy_verifier
    )
    assert '"recorded_at": recovery_event["recorded_at"]' not in legacy_verifier

    monkeypatch.setattr(
        operator,
        "_M43_AUTHORITY_CID",
        "sha256:PENDING_M43_FINAL_CONTROL_AUTHORITY_CID",
    )
    with pytest.raises(
        operator.OperatorError,
        match="active M43 lifecycle-recovery restart authority is invalid",
    ):
        operator._active_source_repair_materialization(config)


def test_m42_dispatch_precedes_m41_and_receipt_follows_live_verification() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m42_dispatch_test",
    )
    check_source = inspect.getsource(materializer.check_materialized)
    materialize_source = inspect.getsource(materializer.materialize)
    m42 = "_m42_successor_configured_on_any_surface"
    m41 = "_m41_successor_configured_on_any_surface"
    assert check_source.index(m42) < check_source.index(m41)
    assert materialize_source.index(m42) < materialize_source.index(m41)
    main_source = inspect.getsource(materializer.main)
    assert main_source.index(
        "failed_pre_authoritative_m41_evidence_projection_successor_materialization"
    ) < main_source.index(
        "failed_pre_authoritative_m40_validation_successor_materialization"
    )
    core = inspect.getsource(materializer._materialize_m42)
    assert core.index("_verify_m42_live_materialization") < core.index(
        "_ensure_m42_source_successor_receipt"
    )
    assert "snapshot.event_cursor == _M42_PRIOR_EVENT_WATERMARK" in core
    assert "snapshot.event_cursor != _M42_TARGET_EVENT_WATERMARK" in core
    assert "_verify_m42_exact_legacy_projection(connection)" in core
    verifier = inspect.getsource(materializer._verify_m42_live_materialization)
    assert "expected_target_evidence_row=expected_evidence" in verifier
    assert "expected_target_evidence_row=evidence" not in verifier
    assert verifier.index("expected_evidence = (") < verifier.index(
        "evidence = connection.execute("
    )
    assert 'for revision in ("m37", "m38", "m39", "m40", "m41")' in core


def test_m42_operator_selects_newest_authority_by_key_presence() -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m42_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    m42_key = (
        "failed_pre_authoritative_m41_evidence_projection_"
        "successor_materialization"
    )
    historical, _, _ = _historical_successor_controls_at(m42_key, config)
    historical["database_program"]["store_generation"] = "30"
    active = operator._active_source_repair_materialization(historical)
    assert active["migration_revision"] == "SAWM-R2-M42"
    assert active["target_event_watermark"] == 292
    assert active["failed_m41_pre_authoritative_materialization"][
        "quack_mutation_request_created"
    ] is False
    configured = inspect.getsource(operator._successor_materialization_configured)
    assert configured.index("_M43_SUCCESSOR_KEY") < configured.index(
        "_M42_SUCCESSOR_KEY"
    )
    assert configured.index("_M42_SUCCESSOR_KEY") < configured.index(
        "_M41_SUCCESSOR_KEY"
    )
    normalized = inspect.getsource(operator._normalized_live_preflight_contract)
    assert normalized.index("_M42_MIGRATION_REVISION") < normalized.index(
        "_M41_MIGRATION_REVISION"
    )
    preflight = inspect.getsource(operator._live_preflight)
    assert "m42_active = active_revision == _M42_MIGRATION_REVISION" in preflight
    assert preflight.index("m42_active,") < preflight.index("m41_active,")
    assert preflight.index("_verify_m42_live_materialization") < preflight.index(
        "_verify_m41_live_materialization"
    )
    assert preflight.index("_expected_m42_source_successor_receipt") < (
        preflight.index("_expected_m41_source_successor_receipt")
    )
    marker = inspect.getsource(operator._require_active_final_pair_marker)
    assert marker.index("_M43_SUCCESSOR_KEY") < marker.index("_M42_SUCCESSOR_KEY")
    assert marker.index("_M42_SUCCESSOR_KEY") < marker.index("_M41_SUCCESSOR_KEY")


def test_m41_authority_seals_failed_m40_validation_and_exact_repair() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m41_authority_test",
    )
    authority = (
        materializer
        ._expected_m41_failed_pre_authoritative_m40_validation_successor_authority()
    )
    reference = materializer._m41_authority_reference()
    failed = authority["failed_m40_pre_authoritative_validation"]
    repair = authority["accepted_historical_test_helper_repair"]
    chain = authority["source_chain"]

    assert reference == {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M41",
        "authority_cid": materializer._M41_AUTHORITY_CID,
    }
    assert materializer._identity(authority) == (
        "sha256:25ad5550b59024c8da9b4821fba2d7b1b49d2a17a781e5d4bd2cbe58cb7b0233"
    )
    assert materializer._identity(authority["prior_m40_authority"]) == (
        materializer._M41_M40_AUTHORITY_CID
    )
    assert failed["phase"] == "sealed_semantic_world_test_suite"
    assert failed["tests_passed"] == 285
    assert failed["tests_failed"] == 22
    assert len(failed["failed_test_node_ids"]) == 22
    assert failed["materializer_invoked"] is False
    assert failed["quack_mutation_request_created"] is False
    assert failed["event_292_rows_created"] == 0
    assert failed["m40_receipt_created"] is False
    assert repair["repair_parent"] == materializer._M41_M40_FINAL_CONTROL_COMMIT
    assert repair["repair_commit"] == (
        materializer._M41_HISTORICAL_TEST_HELPER_REPAIR_COMMIT
    )
    assert repair["blob_oids"] == dict(
        materializer._M41_HISTORICAL_TEST_HELPER_REPAIR_BLOBS
    )
    assert repair["last_failed_rerun_passed"] == 22
    assert repair["direct_regression_passed"] is True
    assert repair["production_selection_changed"] is False
    assert chain["final_control_parent"] == (
        materializer._M41_HISTORICAL_TEST_HELPER_REPAIR_COMMIT
    )
    assert chain["final_control_commit_is_current_head"] is True
    assert chain["final_control_commit_count"] == 1
    assert authority["target_event_watermark"] == 292
    assert authority["target_projection_cid"] == (
        materializer._M40_TARGET_PROJECTION_CID
    )
    assert authority["target_authority"]["evidence_kind"] == (
        "operator_control_plane_failed_pre_authoritative_m40_validation_successor"
    )
    assert authority["live_preflight_contract"][
        "m40_receipt_must_be_absent"
    ] is True
    materializer._validated_m41_live_preflight_contract(authority)


def test_m41_restart_rows_route_through_exact_m40_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m41_restart_authority_test",
    )
    authority = (
        materializer
        ._expected_m41_failed_pre_authoritative_m40_validation_successor_authority()
    )
    observed: list[object] = []

    def inspect_restart(
        source: object, identity: object, prior_m40: object
    ) -> dict[str, bool]:
        observed.extend((source, identity, prior_m40))
        return {"generation_29_30_restart_rows_verified": True}

    monkeypatch.setattr(
        materializer, "_inspect_m40_generation_restart_rows", inspect_restart
    )
    source = object()
    identity = {"server_id": materializer._M41_LIVE_SERVER_ID}
    assert materializer._inspect_m41_generation_restart_rows(
        source, identity, authority
    ) == {"generation_29_30_restart_rows_verified": True}
    assert observed == [source, identity, authority["prior_m40_authority"]]

    malformed = dict(authority)
    malformed["prior_m40_authority"] = {}
    with pytest.raises(
        materializer.MigrationRequired,
        match="historical M40 restart authority differs",
    ):
        materializer._inspect_m41_generation_restart_rows(
            source, identity, malformed
        )


def test_m41_receipt_publication_is_last_idempotent_and_exclusive(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m41_receipt_test",
    )
    control = tmp_path / "control.duckdb"
    control.write_bytes(b"test")
    expected = {"schema": "test/m41-receipt@1", "receipt_cid": "sha256:test"}

    first = materializer._ensure_m41_source_successor_receipt(
        tmp_path, control, expected
    )
    second = materializer._ensure_m41_source_successor_receipt(
        tmp_path, control, expected
    )
    assert first == second == expected
    path = tmp_path / "m41-source-successor-receipt.json"
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for revision in ("m37", "m38", "m39", "m40", "m41"):
        assert (
            tmp_path / f".{revision}-source-successor-receipt.publish.lock"
        ).exists()

    path.unlink()
    (tmp_path / "m40-source-successor-receipt.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="M40 receipt unexpectedly exists",
    ):
        materializer._ensure_m41_source_successor_receipt(
            tmp_path, control, expected
        )


def test_m41_dispatch_precedes_m40_and_receipt_follows_live_verification() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m41_dispatch_test",
    )
    check_source = inspect.getsource(materializer.check_materialized)
    materialize_source = inspect.getsource(materializer.materialize)
    m41 = "_m41_successor_configured_on_any_surface"
    m40 = "_m40_successor_configured_on_any_surface"
    assert check_source.index(m41) < check_source.index(m40)
    assert materialize_source.index(m41) < materialize_source.index(m40)
    main_source = inspect.getsource(materializer.main)
    assert main_source.index(
        "failed_pre_authoritative_m40_validation_successor_materialization"
    ) < main_source.index(
        "failed_pre_authoritative_m39_successor_materialization"
    )
    core = inspect.getsource(materializer._materialize_m41)
    assert core.index("_verify_m41_live_materialization") < core.index(
        "_ensure_m41_source_successor_receipt"
    )
    assert "snapshot.event_cursor == _M41_PRIOR_EVENT_WATERMARK" in core
    assert "snapshot.event_cursor != _M41_TARGET_EVENT_WATERMARK" in core
    assert "target_event_exists is not None" in core
    assert 'for revision in ("m37", "m38", "m39", "m40")' in core


def test_m41_operator_selects_newest_authority_by_key_presence() -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m41_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    historical_config, _historical_migration, _historical_seal = (
        _historical_successor_controls_at(
            "failed_pre_authoritative_m40_validation_successor_materialization",
            config,
        )
    )
    historical_config["database_program"]["store_generation"] = "30"
    active = operator._active_source_repair_materialization(historical_config)
    assert active["migration_revision"] == "SAWM-R2-M41"
    assert active["target_event_watermark"] == 292
    assert active["failed_m40_pre_authoritative_validation"][
        "materializer_invoked"
    ] is False
    configured = inspect.getsource(operator._successor_materialization_configured)
    assert configured.index("_M41_SUCCESSOR_KEY") < configured.index(
        "_M40_SUCCESSOR_KEY"
    )
    normalized = inspect.getsource(operator._normalized_live_preflight_contract)
    assert normalized.index("_M41_MIGRATION_REVISION") < normalized.index(
        "_M40_MIGRATION_REVISION"
    )
    preflight = inspect.getsource(operator._live_preflight)
    assert "m41_active = active_revision == _M41_MIGRATION_REVISION" in preflight
    assert preflight.index("m41_active,") < preflight.index("m40_active,")
    assert preflight.index("_verify_m41_live_materialization") < preflight.index(
        "_verify_m40_live_materialization"
    )
    assert preflight.index("_expected_m41_source_successor_receipt") < (
        preflight.index("_expected_m40_source_successor_receipt")
    )
    assert preflight.index("_M41_SUCCESSOR_KEY in config") < preflight.index(
        "_M40_SUCCESSOR_KEY in config"
    )
    marker = inspect.getsource(operator._require_active_final_pair_marker)
    assert marker.index("_M41_SUCCESSOR_KEY") < marker.index("_M40_SUCCESSOR_KEY")


def test_m21_generation_realization_authority_is_exact_and_presence_first() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m21_authority_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m21_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "generation_realization_successor_materialization"
    expected = materializer._expected_m21_generation_realization_authority()
    historical_config, _historical_migration, _historical_seal = (
        _historical_successor_controls_at(key, config)
    )

    assert config[key] == expected
    assert inventory[key] == expected
    assert seal[f"{key}_cid"] == materializer._identity(expected)
    assert materializer._m21_successor_configured(config) is True
    assert dict(
        operator._active_source_repair_materialization(historical_config)
    ) == expected
    assert operator._successor_materialization_configured(historical_config) is True
    assert expected["target_generation"] == 21
    assert expected["target_plan_revision"] == 22
    assert expected["target_event_watermark"] == 231
    assert expected["generation_mismatch"] == {
        "sealed_expected_generation": 21,
        "observed_realized_generation": 20,
        "typed_preflight_error": (
            "live Quack generation 20 differs from the sealed generation 21"
        ),
    }
    bound_marker = dict(operator._m21_receipt_authority_fields(expected))
    assert operator._m21_receipt_has_exact_authority_fields(
        bound_marker,
        expected,
    )
    for field, invalid in {
        "prior_lifecycle_artifacts_preserved": False,
        "prior_read_replica_sha256": "0" * 64,
        "prior_stopped_status_projection_sha256": "0" * 64,
        "prior_server_id": "server:forged",
        "prior_process_birth_id": "birth:forged",
        "prior_generation": 19,
        "generation_mismatch": {
            **expected["generation_mismatch"],
            "observed_realized_generation": 19,
        },
    }.items():
        tampered_marker = copy.deepcopy(bound_marker)
        tampered_marker[field] = invalid
        assert not operator._m21_receipt_has_exact_authority_fields(
            tampered_marker,
            expected,
        )

    malformed = copy.deepcopy(expected)
    malformed["target_generation"] = 22
    partial = {
        key: malformed,
        "test_isolation_successor_materialization": config[
            "test_isolation_successor_materialization"
        ],
    }
    with pytest.raises(
        materializer.MaterializationError,
        match="M21 generation-realization authority is invalid",
    ):
        materializer._m21_successor_configured(partial)
    with pytest.raises(
        operator.OperatorError,
        match="active M21 generation-realization successor authority is invalid",
    ):
        operator._active_source_repair_materialization(partial)


def test_m21_historical_authority_retains_generation_21_runtime_binding() -> None:
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    authority = config["generation_realization_successor_materialization"]
    runtime_root = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m21"
    control = f"{runtime_root}/control.duckdb"
    coordination = f"{runtime_root}/control.coordination.duckdb"

    assert authority["target_runtime_root"] == runtime_root
    assert authority["target_store_id"] == control
    assert authority["target_coordination_store_id"] == coordination
    assert authority["target_generation"] == 21
    assert authority["target_plan_revision"] == 22
    assert authority["target_event_watermark"] == 231
    assert authority["target_quack_port"] == 24064
    assert config[
        "live_preflight_receipt_compatibility_successor_materialization"
    ]["prior_store_id"] == control


def test_m21_stopped_m20_anchor_is_verified_without_mutation() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m21_anchor_test",
    )
    authority = materializer._expected_m21_generation_realization_authority()
    population = materializer.build_population(REPO_ROOT)
    anchored_paths = tuple(
        REPO_ROOT / str(authority[key])
        for key in (
            "prior_store_id",
            "prior_coordination_store_id",
            "prior_read_replica_path",
            "prior_stopped_status_projection_path",
            "prior_migration_receipt_path",
        )
    )
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in anchored_paths
    }

    control, coordination = materializer._assert_m21_prior_anchor(
        REPO_ROOT,
        authority,
        population,
    )

    assert control == anchored_paths[0].resolve()
    assert coordination == anchored_paths[1].resolve()
    assert before == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in anchored_paths
    }


def test_m21_private_stage_appends_only_plan_and_operator_evidence(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m21_private_stage_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    authority = materializer._expected_m21_generation_realization_authority()
    population = materializer.build_population(REPO_ROOT)
    prior_control, prior_coordination = materializer._assert_m21_prior_anchor(
        REPO_ROOT,
        authority,
        population,
    )
    prior_hashes = (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    )
    stage_dir = tmp_path / "m21-stage"
    stage_dir.mkdir()
    validation_digest = materializer._identity(
        {"schema": "sawm/test-validation-digest@1", "valid": True}
    )

    staged = materializer._stage_m21_store_pair(
        REPO_ROOT,
        stage_dir,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    verified = staged["verified"]

    assert verified["valid"] is True
    assert verified["event_watermark"] == 231
    assert verified["task_revision_changes"] == 0
    assert verified["task_status_changes"] == 0
    assert verified["goal_changes"] == 0
    assert verified["accepted_definition_changes"] == 0
    assert verified["accepted_completion_changes"] == 0
    assert verified["coordination_semantic_changes"] == 0
    assert materializer._store_sha256(staged["stage_coordination"]) == prior_hashes[1]
    assert prior_hashes == (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    )
    assert len(materializer._M21_RECEIPT_KEYS) == 92
    assert {
        "receipt_cid",
        "generation_realization_successor_materialization_cid",
        "accepted_control_plane_repair",
        "ordinary_source_changes",
        "worker_self_approval",
    } <= materializer._M21_RECEIPT_KEYS


def test_m29_committed_evidence_verification_authority_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m29_authority_test",
    )
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m29_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "committed_evidence_verification_successor_materialization"
    expected = (
        materializer._expected_m29_committed_evidence_verification_authority()
    )
    config[key] = copy.deepcopy(expected)
    inventory[key] = copy.deepcopy(expected)
    seal[f"{key}_cid"] = materializer._identity(expected)

    assert expected["schema"] == (
        "sawm/committed-evidence-verification-successor-materialization-"
        "authorization@1"
    )
    assert expected["migration_revision"] == "SAWM-R2-M29"
    assert expected["migration_kind"] == key
    assert expected["runtime_binding"]["run_id"] == "run-r2-m27"
    assert expected["runtime_binding"]["store_generation"] == 27
    assert expected["runtime_binding"]["quack_port"] == 24_070
    assert expected["runtime_binding"]["prior_event_watermark"] == 273
    assert expected["runtime_binding"]["target_event_watermark"] == 274
    assert expected["prior_authority"]["receipt_absent"] is True
    assert expected["prior_authority"]["failure_stage"] == (
        "post_append_verification_before_receipt_publication"
    )
    assert expected["prior_authority"]["event_id"] == (
        "baguqeeragd2blhdw2gdjwwqovgwuoxlaulnto35lxtlycappubgqz4fm3lia"
    )
    assert expected["prior_authority"]["evidence_id"] == (
        "baguqeeraumvois7bdkb7dk27zfbqecpudpc5htrxskz4x2jgau56ivalqvla"
    )
    assert expected["prior_authority"]["plan_source_binding_cid"] == (
        "sha256:83e28e01de41699d5b2312ead03e7f33d9989d809d97924ea0a230af2c038856"
    )
    assert expected["prior_authority"]["plan_migration_digest"] == (
        "sha256:43eb5eb9b3f05c6ffe00c918901524e61c4a94a8d7334a3a3261ef95870cc73b"
    )
    assert expected["target_authority"]["event_watermark"] == 274
    assert expected["source_chain"] == {
        "m28_control_commit": "d7e2a4ba9bc7eef32ffad131ffd092ff11f934c4",
        "verifier_repair_commit": "bb79ffc69b199a735e672072a8cf534417918393",
        "verifier_repair_parent": "d7e2a4ba9bc7eef32ffad131ffd092ff11f934c4",
        "verifier_repair_tree": "c1e0ebea9b82b635d70eaf0b61044118dc05c052",
        "verifier_repair_blobs": {
            "scripts/materialize_semantic_addressed_world_model_program.py": (
                "d88e511bbee55a577d046760de4311c3aabdb188"
            ),
            "scripts/ops/agent_supervisor/semantic_addressed_world_model.py": (
                "d7356bdf08f75e97494f2fc721cf8fb34894bd35"
            ),
            "test/api/semantic_world/"
            "test_semantic_addressed_world_model_board.py": (
                "a1f2167768f6289ae6feff336ccd30a50821d13f"
            ),
        },
        "final_control_commit_count": 1,
    }
    assert expected["exact_changes"] == {
        "event_suffix_length": 1,
        "evidence_node_changes": 1,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "plan_revision_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "coordination_semantic_changes": 0,
        "sidecar_changes": 0,
        "store_generation_row_changes": 0,
        "state_server_row_changes": 0,
        "credential_row_changes": 0,
        "implementation_provider_invocations": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
    }
    assert len(expected["operator_control_paths"]) == 9
    assert expected["preservation"][
        "failed_m28_post_append_attempt_preserved"
    ] is True
    assert expected["preservation"]["m28_receipt_created_or_rewritten"] is False
    assert expected["preservation"]["same_live_owner"] is True
    assert expected["preservation"]["same_store_generation"] is True
    assert expected["preservation"]["worker_self_approval"] is False
    repair = expected["historical_transition_repair"]
    assert repair["task_alias"] == "SAWM-012"
    assert repair["actual_configured_board_admission_cid"] == ""
    assert repair["accepted_completion_changed"] is False
    assert repair["task_completion_authority"] is False

    monkeypatch.setattr(
        dependency,
        "_m29_source_chain_errors",
        lambda *_args, **_kwargs: [],
    )
    assert dependency._m29_committed_evidence_verification_successor_errors(
        config,
        seal,
        inventory,
        root=REPO_ROOT,
        require_active_runtime=False,
    ) == []

    changed = copy.deepcopy(config)
    changed[key]["exact_changes"]["accepted_completion_changes"] = 1
    errors = dependency._m29_committed_evidence_verification_successor_errors(
        changed, seal, inventory, root=REPO_ROOT
    )
    assert any("differs across controls" in error for error in errors)


def test_m29_presence_masks_m28_and_keeps_every_predecessor_historical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m29_presence_test",
    )
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m29_presence_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m29_presence_test",
    )
    key = "committed_evidence_verification_successor_materialization"
    authority = (
        materializer._expected_m29_committed_evidence_verification_authority()
    )
    scheduler = {key: authority}
    migration = {key: authority}
    seal = {f"{key}_cid": materializer._identity(authority)}
    assert dependency._m29_successor_declared(scheduler, {}, {}) is True
    assert dependency._m29_successor_declared({}, {}, migration) is True
    assert dependency._m29_successor_declared({}, seal, {}) is True
    assert dependency._m29_successor_declared({}, {}, {}) is False

    monkeypatch.setattr(
        board,
        "_m29_migration_errors",
        lambda *_args, **_kwargs: ["M29 active"],
    )
    historical_calls: list[bool] = []

    def historical(*_args: object, **kwargs: object) -> list[str]:
        historical_calls.append(kwargs.get("require_active_runtime") is False)
        return []

    for name in (
        "_m28_migration_errors", "_m27_migration_errors",
        "_m26_migration_errors", "_m25_migration_errors",
        "_m24_migration_errors", "_m23_migration_errors",
        "_m22_migration_errors", "_m21_migration_errors",
        "_m20_migration_errors", "_m19_migration_errors",
        "_m18_migration_errors", "_m17_migration_errors",
        "_m16_migration_errors",
    ):
        monkeypatch.setattr(board, name, historical)
    assert board._active_successor_migration_errors(
        scheduler, seal, migration
    ) == ["M29 active"]
    assert historical_calls == [True] * 13

    partial = {key: authority}
    errors = board._active_successor_migration_errors(partial, {}, {})
    assert "M29 active" in errors
    assert any("only partially declared" in error for error in errors)


    # Live M38 event bodies are Quack envelopes; verification uses inner JSON.
def test_m38_authority_pins_custody_restart_and_preserves_m37() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_authority_test",
    )
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m38_authority_test",
    )
    key = "pre_authoritative_custody_restart_successor_materialization"
    authority = (
        materializer._expected_m38_pre_authoritative_custody_restart_authority()
    )
    identity_state = dependencies._m38_source_chain_identity_state(
        materializer, authority
    )
    reference = dependencies._m38_authority_reference_for_source_state(
        materializer, authority
    )
    scheduler = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    assert scheduler[key] == reference == migration[key]
    expected_authority_cid = (
        "sha256:PENDING_M38_AUTHORITY_CID"
        if identity_state == "placeholder"
        else materializer._identity(authority)
    )
    if identity_state == "sealed":
        assert expected_authority_cid == (
            "sha256:664af21f470ada7e4d4bf02313df473ed345c539b122f30036db8c8ee171a8eb"
        )
    assert seal[f"{key}_cid"] == reference["authority_cid"] == expected_authority_cid
    assert reference["schema"] == "sawm/operator-control-authority-reference@1"
    assert reference["migration_revision"] == "SAWM-R2-M38"
    assert authority["schema"] == (
        "sawm/pre-authoritative-custody-restart-successor-authorization@1"
    )
    assert authority["superseded_m37_authority"]["authority_cid"] == (
        "sha256:c776180b7e65de98d5de235765db60148f7693148512b335260ddb772563a795"
    )
    assert authority["prior_authority"]["control_store_sha256"] == (
        "ea5b66208455f398502e8ad939566a5957f5bc65f35ed5afe8cc3998be66eb41"
    )
    if identity_state == "placeholder":
        assert materializer._m38_source_identities_pending() is True
    else:
        assert identity_state == "sealed"
        assert materializer._m38_source_identities_pending() is False


def test_m38_placeholder_controls_fail_closed_at_launch_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_placeholder_rejection_test",
    )
    key = "pre_authoritative_custody_restart_successor_materialization"
    pending = {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M38",
        "authority_cid": "sha256:PENDING_M38_AUTHORITY_CID",
    }
    monkeypatch.setattr(
        materializer,
        "_M38_INITIAL_CONTROL_COMMIT",
        "PENDING_M38_INITIAL_CONTROL_COMMIT",
    )
    monkeypatch.setattr(
        materializer,
        "_M38_INITIAL_CONTROL_TREE",
        "PENDING_M38_INITIAL_CONTROL_TREE",
    )
    monkeypatch.setattr(
        materializer,
        "_M38_INITIAL_CONTROL_BLOBS",
        {
            path: f"PENDING_M38_INITIAL_CONTROL_BLOB_{index}"
            for index, path in enumerate(
                sorted(materializer._M38_OPERATOR_CONTROL_PATHS), start=1
            )
        },
    )
    assert materializer._m38_successor_configured({key: pending}) is True
    with pytest.raises(
        materializer.MaterializationError,
        match="M38 source identities are not resealed",
    ):
        materializer._assert_m38_source_delta(
            REPO_ROOT,
            {"source_binding": {"head": "0" * 40, "tree": "0" * 40}},
            materializer._expected_m38_pre_authoritative_custody_restart_authority(),
        )
    m37_key = "post_reboot_generation_restart_successor_materialization"
    m37_reference = {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M37",
        "authority_cid": (
            "sha256:c776180b7e65de98d5de235765db60148f7693148512b335260ddb772563a795"
        ),
    }
    monkeypatch.setattr(
        materializer,
        "_load_json",
        lambda *_args: {
            f"{key}_cid": pending["authority_cid"],
            f"{m37_key}_cid": m37_reference["authority_cid"],
        },
    )
    with pytest.raises(
        materializer.MaterializationError,
        match="M38 source identities are not resealed",
    ):
        materializer._m38_source_binding_authority(
            REPO_ROOT,
            {
                "migration_inventory": {
                    key: pending,
                    m37_key: m37_reference,
                },
                "source_binding": {"head": "0" * 40},
            },
            {
                key: pending,
                m37_key: m37_reference,
            },
        )


@pytest.mark.parametrize("identity_case", ("mixed", "arbitrary", "all_zero"))
def test_m38_rejects_nonexact_initial_control_identity_states(
    monkeypatch: pytest.MonkeyPatch,
    identity_case: str,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        f"sawm_materializer_m38_{identity_case}_identity_test",
    )
    paths = sorted(materializer._M38_OPERATOR_CONTROL_PATHS)
    exact_pending = {
        path: f"PENDING_M38_INITIAL_CONTROL_BLOB_{index}"
        for index, path in enumerate(paths, start=1)
    }
    if identity_case == "mixed":
        commit = "PENDING_M38_INITIAL_CONTROL_COMMIT"
        tree = "a" * 40
        blobs = exact_pending
    elif identity_case == "arbitrary":
        commit = "PENDING_M38_NOT_AN_AUTHORIZED_SENTINEL"
        tree = "PENDING_M38_INITIAL_CONTROL_TREE"
        blobs = exact_pending
    else:
        commit = tree = "0" * 40
        blobs = {path: "0" * 40 for path in paths}
    monkeypatch.setattr(materializer, "_M38_INITIAL_CONTROL_COMMIT", commit)
    monkeypatch.setattr(materializer, "_M38_INITIAL_CONTROL_TREE", tree)
    monkeypatch.setattr(materializer, "_M38_INITIAL_CONTROL_BLOBS", blobs)
    with pytest.raises(
        materializer.MaterializationError,
        match="M38 initial-control identities mix placeholder and sealed values",
    ):
        materializer._m38_source_identities_pending()


def test_m38_dependency_validator_rejects_noncanonical_pending_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_noncanonical_dependency_identity_test",
    )
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m38_noncanonical_identity_test",
    )
    authority = copy.deepcopy(
        materializer._expected_m38_pre_authoritative_custody_restart_authority()
    )
    paths = sorted(materializer._M38_OPERATOR_CONTROL_PATHS)
    arbitrary_commit = "PENDING_M38_INITIAL_UNAUTHORIZED_COMMIT"
    arbitrary_tree = "PENDING_M38_INITIAL_UNAUTHORIZED_TREE"
    arbitrary_blobs = {
        path: f"PENDING_M38_INITIAL_UNAUTHORIZED_BLOB_{index}"
        for index, path in enumerate(paths, start=1)
    }
    chain = authority["source_chain"]
    chain["initial_control_commit"] = arbitrary_commit
    chain["initial_control_tree"] = arbitrary_tree
    chain["initial_control_blobs"] = arbitrary_blobs
    monkeypatch.setattr(
        materializer, "_M38_INITIAL_CONTROL_COMMIT", arbitrary_commit
    )
    monkeypatch.setattr(materializer, "_M38_INITIAL_CONTROL_TREE", arbitrary_tree)
    monkeypatch.setattr(materializer, "_M38_INITIAL_CONTROL_BLOBS", arbitrary_blobs)
    with pytest.raises(
        RuntimeError,
        match="M38 initial-control identities mix placeholder/sealed values",
    ):
        dependencies._m38_source_chain_identity_state(materializer, authority)


def test_m38_preserves_the_exact_historical_m37_triplet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_historical_m37_triplet_test",
    )
    scheduler = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "post_reboot_generation_restart_successor_materialization"
    materializer._assert_m38_historical_m37_triplet(
        scheduler, migration, seal
    )

    for surface, field in (
        (scheduler, key),
        (migration, key),
        (seal, f"{key}_cid"),
    ):
        altered = copy.deepcopy(surface)
        del altered[field]
        arguments = (
            (altered, migration, seal)
            if surface is scheduler
            else (scheduler, altered, seal)
            if surface is migration
            else (scheduler, migration, altered)
        )
        with pytest.raises(
            materializer.MaterializationError,
            match="M38 historical M37 authority triplet differs",
        ):
            materializer._assert_m38_historical_m37_triplet(*arguments)

    original = materializer._expected_m37_post_reboot_generation_restart_authority

    def altered_m37_authority() -> dict[str, object]:
        authority = copy.deepcopy(original())
        authority["authorized"] = False
        return authority

    monkeypatch.setattr(
        materializer,
        "_expected_m37_post_reboot_generation_restart_authority",
        altered_m37_authority,
    )
    with pytest.raises(
        materializer.MaterializationError,
        match="M38 historical M37 authority triplet differs",
    ):
        materializer._assert_m38_historical_m37_triplet(
            scheduler, migration, seal
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "state_root_mode",
        "lane_mode",
        "lane_owner",
        "lane_symlink",
        "pid_symlink",
    ),
)
def test_m38_physical_prestart_evidence_rejects_directory_and_symlink_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        f"sawm_materializer_m38_physical_{mutation}_test",
    )
    root = tmp_path.resolve()
    runtime = root / "runtime"
    state_root = runtime / "state"
    lane_dir = state_root / "lane-0"
    lane_dir.mkdir(parents=True)
    os.chmod(state_root, 0o700)
    os.chmod(lane_dir, 0o775)
    control = runtime / "control.duckdb"
    replica = runtime / "control.read-replica.duckdb"
    control.write_bytes(b"control")
    replica.write_bytes(b"replica")
    os.chmod(control, 0o664)
    os.chmod(replica, 0o600)
    pid_path = lane_dir / "sawm_lane_0_supervisor.pid"
    pid_payload = b"123\n"
    pid_path.write_bytes(pid_payload)
    os.chmod(pid_path, 0o664)

    control_stat = os.lstat(control)
    replica_stat = os.lstat(replica)
    pid_stat = os.lstat(pid_path)
    monkeypatch.setattr(materializer, "_M38_RUNTIME_ROOT", "runtime")
    monkeypatch.setattr(
        materializer, "_M38_PRIOR_CONTROL_SHA256", hashlib.sha256(b"control").hexdigest()
    )
    monkeypatch.setattr(materializer, "_M38_PRIOR_CONTROL_SIZE", len(b"control"))
    monkeypatch.setattr(
        materializer, "_M38_PRIOR_CONTROL_MTIME_NS", control_stat.st_mtime_ns
    )
    monkeypatch.setattr(
        materializer, "_M38_PRIOR_CONTROL_CTIME_NS", control_stat.st_ctime_ns
    )
    monkeypatch.setattr(
        materializer, "_M38_READ_REPLICA_SHA256", hashlib.sha256(b"replica").hexdigest()
    )
    monkeypatch.setattr(materializer, "_M38_READ_REPLICA_SIZE", len(b"replica"))
    monkeypatch.setattr(
        materializer, "_M38_READ_REPLICA_MTIME_NS", replica_stat.st_mtime_ns
    )
    monkeypatch.setattr(
        materializer, "_M38_READ_REPLICA_CTIME_NS", replica_stat.st_ctime_ns
    )
    expected_uid = pid_stat.st_uid + (1 if mutation == "lane_owner" else 0)
    evidence = {
        "lane-0": {
            "path": "runtime/state/lane-0/sawm_lane_0_supervisor.pid",
            "pid": 123,
            "inode": pid_stat.st_ino,
            "size": len(pid_payload),
            "sha256": hashlib.sha256(pid_payload).hexdigest(),
            "mtime_ns": pid_stat.st_mtime_ns,
            "ctime_ns": pid_stat.st_ctime_ns,
            "mode": 0o664,
            "uid": expected_uid,
            "gid": pid_stat.st_gid,
            "link_count": 1,
            "device": pid_stat.st_dev,
            "owner_liveness": "dead",
            "liveness_probe": "kill_pid_0_process_lookup_error",
        }
    }
    monkeypatch.setattr(materializer, "_m38_lane_pid_evidence", lambda: evidence)

    if mutation == "state_root_mode":
        os.chmod(state_root, 0o755)
    elif mutation == "lane_mode":
        os.chmod(lane_dir, 0o700)
    elif mutation == "lane_symlink":
        shutil.rmtree(lane_dir)
        external_lane = root / "external-lane"
        external_lane.mkdir()
        lane_dir.symlink_to(external_lane, target_is_directory=True)
    elif mutation == "pid_symlink":
        pid_path.unlink()
        external_pid = root / "external.pid"
        external_pid.write_bytes(pid_payload)
        pid_path.symlink_to(external_pid)

    monkeypatch.setattr(
        materializer.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    with pytest.raises(materializer.MigrationRequired):
        materializer._verify_m38_failed_attempt_physical_evidence(root, control)


def test_m38_prestart_rejects_a_read_replica_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_replica_wal_test",
    )
    root = tmp_path.resolve()
    runtime = root / "runtime"
    runtime.mkdir()
    control = runtime / "control.duckdb"
    coordination = runtime / "control.coordination.duckdb"
    replica_wal = runtime / "control.read-replica.duckdb.wal"
    replica_wal.write_bytes(b"unexpected WAL")
    authority = (
        materializer._expected_m38_pre_authoritative_custody_restart_authority()
    )
    stopped = authority["stopped_owner"]
    status = {
        "lifecycle": "stopped",
        "recovered_stale_owner": True,
        "recovery_stopped_at": materializer._M38_PRIOR_STOPPED_AT,
        "identity": {
            "status": "stopped",
            "server_id": stopped["server_id"],
            "process_birth_id": stopped["process_birth_id"],
            "database_uuid": stopped["database_uuid"],
            "store_id": stopped["store_id"],
            "listen_uri": stopped["listen_uri"],
            "extension_fingerprint": materializer._M38_EXTENSION_FINGERPRINT,
            "generation": materializer._M38_PRIOR_GENERATION,
        },
    }
    monkeypatch.setattr(materializer, "build_population", lambda _root: {})
    monkeypatch.setattr(
        materializer, "_assert_committed_clean_source", lambda *_args: None
    )
    monkeypatch.setattr(
        materializer, "_m38_source_binding_authority", lambda *_args: authority
    )
    monkeypatch.setattr(materializer, "_assert_m38_source_delta", lambda *_args: None)
    monkeypatch.setattr(
        materializer,
        "_m38_target_paths",
        lambda *_args: (control, coordination),
    )
    monkeypatch.setattr(materializer, "_assert_offline", lambda *_args: None)

    def sealed_identity(path: Path, **_kwargs: object) -> tuple[str, int]:
        if path == control:
            return (
                materializer._M38_PRIOR_CONTROL_SHA256,
                materializer._M38_PRIOR_CONTROL_SIZE,
            )
        if path == coordination:
            return (
                materializer._M38_PRIOR_COORDINATION_SHA256,
                materializer._M38_PRIOR_COORDINATION_SIZE,
            )
        if path.name == "quack-state-server.status.json":
            return (
                materializer._M37_STOPPED_STATUS_SHA256,
                materializer._M37_STOPPED_STATUS_SIZE,
            )
        if path.name == "quack-stale-owner-recovery-receipt.json":
            return (
                materializer._M37_RECOVERY_RECEIPT_SHA256,
                materializer._M37_RECOVERY_RECEIPT_SIZE,
            )
        return (
            materializer._M37_M36_RECEIPT_SHA256,
            materializer._M37_M36_RECEIPT_SIZE,
        )

    monkeypatch.setattr(materializer, "_stable_regular_sha256", sealed_identity)
    monkeypatch.setattr(
        materializer,
        "_verify_m37_preserved_receipts",
        lambda *_args: {
            "m36_historical_anchor_verified": True,
            "m36_receipt_and_event_prefix_verified": True,
            "stale_owner_recovery_receipt_verified": True,
        },
    )
    monkeypatch.setattr(
        materializer,
        "_verify_m38_failed_attempt_physical_evidence",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        materializer, "_load_nofollow_json", lambda *_args, **_kwargs: (status, "")
    )
    monkeypatch.setattr(
        materializer,
        "_inspect_m38_stopped_projection",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("WAL must fail before database inspection")
        ),
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="M38 stopped owner status differs",
    ):
        materializer._check_m38_prestart_admission(root, {})


class _M38ProjectionResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def fetchall(self) -> list[object]:
        return self._rows

    def fetchone(self) -> object | None:
        return self._rows[0] if self._rows else None


class _M38ProjectionConnection:
    def __init__(
        self,
        events: list[object],
        evidence_nodes: list[object],
    ) -> None:
        self._events = events
        self._evidence_nodes = evidence_nodes

    def execute(
        self, query: str, _parameters: object = None
    ) -> _M38ProjectionResult:
        if "FROM domain_events" in query:
            parameters = list(_parameters or ())
            watermark = int(parameters[0]) if parameters else 2**63 - 1
            return _M38ProjectionResult(
                [
                    row
                    for row in self._events
                    if int(
                        row["global_sequence"]
                        if isinstance(row, dict)
                        else row[3]
                    )
                    <= watermark
                ]
            )
        if "FROM evidence_nodes" in query:
            return _M38ProjectionResult(
                sorted(
                    self._evidence_nodes,
                    key=lambda row: str(
                        row["evidence_id"] if isinstance(row, dict) else row[0]
                    ),
                )
            )
        raise AssertionError(f"unexpected query: {query}")


def _m38_evidence_id(inner: dict[str, object]) -> str:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    return content_identity(
        {
            "task_cid": inner["task_cid"],
            "evidence_kind": inner["evidence_kind"],
            "digest": inner["digest"],
            "body": inner["body"],
        }
    )


def _m38_projection_event(
    materializer: object,
    *,
    sequence: int,
    event_type: str,
    inner: dict[str, object],
    attempt_id: str = "",
    event_recorded_at: str | None = None,
) -> tuple[object, ...]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    subject_field = (
        "evidence_id"
        if event_type == "intent.evidence_recorded"
        else "result_id"
    )
    inner_time_field = (
        "created_at"
        if event_type == "intent.evidence_recorded"
        else "recorded_at"
    )
    inner_recorded_at = str(inner[inner_time_field])
    recorded_at = (
        inner_recorded_at
        if event_recorded_at is None
        else str(event_recorded_at)
    )
    envelope = {
        "schema": "ipfs_accelerate_py/agent-supervisor/intent-event@1",
        "event_type": event_type,
        "subject_id": inner[subject_field],
        "body": inner,
        "recorded_at": recorded_at,
        "owner_id": "m38-test-owner",
    }
    event_id = content_identity(
        {
            "stream_id": "stream:intent",
            "sequence": sequence,
            "global_sequence": sequence,
            "event_type": event_type,
            "body": envelope,
        }
    )
    return (
        event_id,
        "stream:intent",
        sequence,
        sequence,
        event_type,
        inner["task_cid"],
        attempt_id,
        "session:m38-test",
        recorded_at,
        materializer._canonical(envelope).decode("utf-8"),
    )


def _m38_validation_projection(
    materializer: object,
    *,
    task_cid: str,
    outcome: str,
    digest: str,
    recorded_at: str,
    attempt_id: str,
    argv: list[str],
) -> tuple[dict[str, object], tuple[str, ...] | None]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    run_id = content_identity(
        {
            "task_cid": task_cid,
            "attempt_id": attempt_id,
            "argv": argv,
            "recorded_at": recorded_at,
        }
    )
    result_id = content_identity(
        {
            "run_id": run_id,
            "outcome": outcome,
            "evidence_digest": digest,
        }
    )
    inner: dict[str, object] = {
        "result_id": result_id,
        "run_id": run_id,
        "task_cid": task_cid,
        "outcome": outcome,
        "evidence_digest": digest,
        "argv": argv,
        "body": {"bounded": True},
        "recorded_at": recorded_at,
        "revision": 0,
    }
    if outcome != "passed":
        return inner, None
    evidence_id = content_identity(
        {
            "task_cid": task_cid,
            "evidence_kind": "validation",
            "digest": digest,
            "run_id": run_id,
        }
    )
    row = (
        evidence_id,
        "",
        task_cid,
        "validation",
        digest,
        recorded_at,
        materializer._canonical(
            {
                "run_id": run_id,
                "result_id": result_id,
                "argv": argv,
                "outcome": "passed",
            }
        ).decode("utf-8"),
    )
    return inner, row


def test_m38_evidence_projection_derives_validation_and_folds_last_write() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_complete_evidence_projection_test",
    )
    initial = {
        "evidence_id": "",
        "parent_evidence_id": "",
        "task_cid": "task:expected",
        "evidence_kind": "operator_control",
        "digest": "sha256:expected",
        "body": {"bounded": True},
        "created_at": "2026-09-01T01:10:00Z",
        "revision": 0,
    }
    initial["evidence_id"] = _m38_evidence_id(initial)
    refreshed = {**initial, "created_at": "2026-09-01T01:11:00Z"}
    passed, validation_row = _m38_validation_projection(
        materializer,
        task_cid="task:validated",
        outcome="passed",
        digest="sha256:passed",
        recorded_at="2026-09-01T01:12:00Z",
        attempt_id="attempt:1",
        argv=["python", "-m", "pytest"],
    )
    failed, failed_row = _m38_validation_projection(
        materializer,
        task_cid="task:validated",
        outcome="failed",
        digest="sha256:failed",
        recorded_at="2026-09-01T01:13:00Z",
        attempt_id="attempt:2",
        argv=["python", "-m", "pytest", "failed"],
    )
    assert validation_row is not None
    assert failed_row is None
    expected_evidence_row = (
        refreshed["evidence_id"],
        refreshed["parent_evidence_id"],
        refreshed["task_cid"],
        refreshed["evidence_kind"],
        refreshed["digest"],
        refreshed["created_at"],
        materializer._canonical(refreshed["body"]).decode("utf-8"),
    )
    events = [
        _m38_projection_event(
            materializer,
            sequence=1,
            event_type="intent.evidence_recorded",
            inner=initial,
        ),
        _m38_projection_event(
            materializer,
            sequence=2,
            event_type="intent.evidence_recorded",
            inner=refreshed,
        ),
        _m38_projection_event(
            materializer,
            sequence=3,
            event_type="intent.validation_recorded",
            inner=passed,
            attempt_id="attempt:1",
        ),
        _m38_projection_event(
            materializer,
            sequence=4,
            event_type="intent.validation_recorded",
            inner=failed,
            attempt_id="attempt:2",
        ),
    ]

    result = materializer._verify_m38_evidence_projection(
        _M38ProjectionConnection(
            events, [expected_evidence_row, validation_row]
        ),
        watermark=4,
    )

    assert result["evidence_node_count"] == 2
    assert result["evidence_event_count"] == 2
    assert result["evidence_recorded_node_count"] == 1
    assert result["validation_event_count"] == 2
    assert result["passed_validation_event_count"] == 1
    assert result["validation_evidence_node_count"] == 1
    assert result["complete_evidence_projection_verified"] is True


def test_m38_evidence_projection_binds_split_evidence_and_event_times() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_split_evidence_event_times_test",
    )
    inner = {
        "evidence_id": "",
        "parent_evidence_id": "",
        "task_cid": "task:split-evidence-time",
        "evidence_kind": "operator_control",
        "digest": "sha256:split-evidence-time",
        "body": {"bounded": True},
        "created_at": "2026-09-01T01:10:00Z",
        "revision": 0,
    }
    inner["evidence_id"] = _m38_evidence_id(inner)
    evidence_row = (
        inner["evidence_id"],
        inner["parent_evidence_id"],
        inner["task_cid"],
        inner["evidence_kind"],
        inner["digest"],
        inner["created_at"],
        materializer._canonical(inner["body"]).decode("utf-8"),
    )
    event = _m38_projection_event(
        materializer,
        sequence=1,
        event_type="intent.evidence_recorded",
        inner=inner,
        event_recorded_at="2026-09-01T01:10:01Z",
    )

    result = materializer._verify_m38_evidence_projection(
        _M38ProjectionConnection([event], [evidence_row]), watermark=1
    )

    assert result["evidence_node_count"] == 1
    assert event[8] != inner["created_at"]

    outer_row_tamper = list(event)
    outer_row_tamper[8] = "2026-09-01T01:10:02Z"
    with pytest.raises(
        materializer.MigrationRequired,
        match="M38 evidence event envelope differs",
    ):
        materializer._verify_m38_evidence_projection(
            _M38ProjectionConnection([tuple(outer_row_tamper)], [evidence_row]),
            watermark=1,
        )

    inner_time_tamper = {
        **inner,
        "created_at": "2026-09-01T01:10:02Z",
    }
    inner_tampered_event = _m38_projection_event(
        materializer,
        sequence=1,
        event_type="intent.evidence_recorded",
        inner=inner_time_tamper,
        event_recorded_at="2026-09-01T01:10:01Z",
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="M38 evidence-node/event projection conflicts",
    ):
        materializer._verify_m38_evidence_projection(
            _M38ProjectionConnection([inner_tampered_event], [evidence_row]),
            watermark=1,
        )


def test_m38_evidence_projection_binds_split_validation_and_event_times() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_split_validation_event_times_test",
    )
    inner, evidence_row = _m38_validation_projection(
        materializer,
        task_cid="task:split-validation-time",
        outcome="passed",
        digest="sha256:split-validation-time",
        recorded_at="2026-09-01T01:11:00Z",
        attempt_id="attempt:split-validation-time",
        argv=["python", "-m", "pytest"],
    )
    assert evidence_row is not None
    event = _m38_projection_event(
        materializer,
        sequence=1,
        event_type="intent.validation_recorded",
        inner=inner,
        attempt_id="attempt:split-validation-time",
        event_recorded_at="2026-09-01T01:11:01Z",
    )

    result = materializer._verify_m38_evidence_projection(
        _M38ProjectionConnection([event], [evidence_row]), watermark=1
    )

    assert result["validation_event_count"] == 1
    assert result["passed_validation_event_count"] == 1
    assert event[8] != inner["recorded_at"]

    outer_row_tamper = list(event)
    outer_row_tamper[8] = "2026-09-01T01:11:02Z"
    with pytest.raises(
        materializer.MigrationRequired,
        match="M38 evidence event envelope differs",
    ):
        materializer._verify_m38_evidence_projection(
            _M38ProjectionConnection([tuple(outer_row_tamper)], [evidence_row]),
            watermark=1,
        )

    inner_time_tamper = {
        **inner,
        "recorded_at": "2026-09-01T01:11:02Z",
    }
    inner_tampered_event = _m38_projection_event(
        materializer,
        sequence=1,
        event_type="intent.validation_recorded",
        inner=inner_time_tamper,
        attempt_id="attempt:split-validation-time",
        event_recorded_at="2026-09-01T01:11:01Z",
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="M38 validation event envelope differs",
    ):
        materializer._verify_m38_evidence_projection(
            _M38ProjectionConnection([inner_tampered_event], [evidence_row]),
            watermark=1,
        )


def _install_synthetic_m42_evidence_overlay(
    monkeypatch: pytest.MonkeyPatch,
    materializer: ModuleType,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    columns = (
        "evidence_id",
        "parent_evidence_id",
        "task_cid",
        "evidence_kind",
        "digest",
        "created_at",
        "body_json",
    )
    refresh_source = (
        "evidence:refresh",
        "",
        "task:refresh",
        "operator_control",
        "sha256:refresh",
        "2026-09-01T01:20:00Z",
        "{}",
    )
    refresh_target = refresh_source[0:5] + (
        "2026-09-01T01:20:01Z",
    ) + refresh_source[6:]
    full_body = {
        "argv": ["python", "-m", "pytest"],
        "outcome": "passed",
        "result_id": "result:compact",
        "run_id": "run:compact",
    }
    compact_body = {
        "result_id": "result:compact",
        "run_id": "run:compact",
    }
    compact_source = (
        "evidence:compact",
        "",
        "task:compact",
        "validation",
        "sha256:compact",
        "2026-09-01T01:21:00Z",
        materializer._canonical(full_body).decode("utf-8"),
    )
    compact_target = compact_source[0:6] + (
        materializer._canonical(compact_body).decode("utf-8"),
    )
    row_id = lambda row: materializer._m42_projection_row_identity(
        "sawm/evidence-projection-row@1", columns, row
    )
    monkeypatch.setattr(materializer, "_M42_LEGACY_PROJECTION_WATERMARK", 1)
    monkeypatch.setattr(materializer, "_M42_LEGACY_EVIDENCE_NODE_COUNT", 2)
    monkeypatch.setattr(
        materializer,
        "_M42_LEGACY_EVIDENCE_REFRESHED_AT",
        refresh_target[5],
    )
    monkeypatch.setattr(
        materializer,
        "_M42_LEGACY_EVIDENCE_REFRESH_ROWS",
        MappingProxyType(
            {
                refresh_source[0]: MappingProxyType(
                    {
                        "source_created_at": refresh_source[5],
                        "source_row_cid": row_id(refresh_source),
                        "target_row_cid": row_id(refresh_target),
                    }
                )
            }
        ),
    )
    monkeypatch.setattr(
        materializer,
        "_M42_LEGACY_COMPACT_VALIDATION_EVIDENCE",
        MappingProxyType(
            {
                "evidence_id": compact_source[0],
                "source_body_cid": materializer._identity(full_body),
                "target_body": MappingProxyType(compact_body),
                "target_body_cid": materializer._identity(compact_body),
                "source_row_cid": row_id(compact_source),
                "target_row_cid": row_id(compact_target),
            }
        ),
    )
    monkeypatch.setattr(
        materializer,
        "_M42_LEGACY_PROJECTION_MANIFEST_CID",
        materializer._identity(materializer._m42_legacy_projection_manifest()),
    )
    return [refresh_source, compact_source], [refresh_target, compact_target]


def test_m42_exact_legacy_evidence_overlay_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m42_exact_legacy_evidence_overlay_test",
    )
    source, target = _install_synthetic_m42_evidence_overlay(
        monkeypatch, materializer
    )

    assert materializer._m42_apply_exact_legacy_evidence_overlay(
        source, watermark=1
    ) == sorted(target)

    with pytest.raises(
        materializer.MigrationRequired,
        match="M42 legacy evidence source count differs",
    ):
        materializer._m42_apply_exact_legacy_evidence_overlay(
            [*source, source[0]], watermark=1
        )

    partial = [
        ("evidence:other", *source[0][1:]),
        source[1],
    ]
    with pytest.raises(
        materializer.MigrationRequired,
        match="M42 legacy evidence overlay is partial",
    ):
        materializer._m42_apply_exact_legacy_evidence_overlay(
            partial, watermark=1
        )

    wrong_timestamp = [
        source[0][0:5] + ("2026-09-01T01:20:02Z",) + source[0][6:],
        source[1],
    ]
    with pytest.raises(
        materializer.MigrationRequired,
        match="M42 legacy evidence refresh source row differs",
    ):
        materializer._m42_apply_exact_legacy_evidence_overlay(
            wrong_timestamp, watermark=1
        )

    wrong_kind = [
        source[0][0:3] + ("validation",) + source[0][4:],
        source[1],
    ]
    with pytest.raises(
        materializer.MigrationRequired,
        match="M42 legacy evidence refresh source row differs",
    ):
        materializer._m42_apply_exact_legacy_evidence_overlay(
            wrong_kind, watermark=1
        )

    wrong_body = [
        source[0],
        source[1][0:6] + ('{"run_id":"run:other"}',),
    ]
    with pytest.raises(
        materializer.MigrationRequired,
        match="M42 compact validation evidence source row differs",
    ):
        materializer._m42_apply_exact_legacy_evidence_overlay(
            wrong_body, watermark=1
        )

    with pytest.raises(
        materializer.MigrationRequired,
        match="M42 legacy evidence overlay watermark differs",
    ):
        materializer._m42_apply_exact_legacy_evidence_overlay(
            source, watermark=2
        )


class _M42ValidationProjectionConnection:
    def __init__(
        self,
        event: tuple[object, ...],
        runs: list[tuple[object, ...]],
        results: list[tuple[object, ...]],
    ) -> None:
        self.event = event
        self.runs = runs
        self.results = results

    def execute(
        self, query: str, _parameters: object = None
    ) -> _M38ProjectionResult:
        if "FROM domain_events" in query:
            return _M38ProjectionResult(
                [(self.event[3], self.event[6], self.event[9])]
            )
        if "FROM validation_runs" in query:
            return _M38ProjectionResult(sorted(self.runs))
        if "FROM validation_results" in query:
            return _M38ProjectionResult(sorted(self.results))
        raise AssertionError(f"unexpected query: {query}")


def test_m42_validation_tables_bind_the_exact_attempt_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m42_exact_validation_table_overlay_test",
    )
    attempt_id = "attempt:source"
    inner, _evidence = _m38_validation_projection(
        materializer,
        task_cid="task:validation-overlay",
        outcome="passed",
        digest="sha256:validation-overlay",
        recorded_at="2026-09-01T01:22:00Z",
        attempt_id=attempt_id,
        argv=["python", "-m", "pytest"],
    )
    event = _m38_projection_event(
        materializer,
        sequence=1,
        event_type="intent.validation_recorded",
        inner=inner,
        attempt_id=attempt_id,
        event_recorded_at="2026-09-01T01:22:01Z",
    )
    run_columns = (
        "run_id",
        "task_cid",
        "attempt_id",
        "started_at",
        "finished_at",
        "status",
        "command_digest",
        "body_json",
    )
    source_run = (
        inner["run_id"],
        inner["task_cid"],
        attempt_id,
        inner["recorded_at"],
        inner["recorded_at"],
        inner["outcome"],
        content_identity({"argv": inner["argv"]}),
        materializer._canonical(
            {"argv": inner["argv"], **inner["body"]}
        ).decode("utf-8"),
    )
    target_run = source_run[0:2] + ("",) + source_run[3:]
    result = (
        inner["result_id"],
        inner["run_id"],
        inner["task_cid"],
        0,
        inner["outcome"],
        inner["evidence_digest"],
        materializer._canonical(inner["body"]).decode("utf-8"),
    )
    row_id = lambda row: materializer._m42_projection_row_identity(
        "sawm/validation-run-projection-row@1", run_columns, row
    )
    monkeypatch.setattr(materializer, "_M42_LEGACY_PROJECTION_WATERMARK", 1)
    monkeypatch.setattr(materializer, "_M42_LEGACY_VALIDATION_EVENT_COUNT", 1)
    monkeypatch.setattr(
        materializer,
        "_M42_LEGACY_VALIDATION_RUN_OVERLAY",
        MappingProxyType(
            {
                "global_sequence": 1,
                "run_id": source_run[0],
                "source_attempt_id": attempt_id,
                "target_attempt_id": "",
                "source_row_cid": row_id(source_run),
                "target_row_cid": row_id(target_run),
            }
        ),
    )
    monkeypatch.setattr(
        materializer, "_M42_LEGACY_PROJECTION_MANIFEST_CID", "sha256:test"
    )
    connection = _M42ValidationProjectionConnection(
        event, [target_run], [result]
    )
    runs_digest = materializer._identity(
        {
            "schema": materializer._M42_LEGACY_VALIDATION_PROJECTION_SCHEMA,
            "manifest_cid": "sha256:test",
            "event_watermark": 1,
            "table": "validation_runs",
            "row_count": 1,
            "rows": [target_run],
        }
    )
    results_digest = materializer._identity(
        {
            "schema": materializer._M42_LEGACY_VALIDATION_PROJECTION_SCHEMA,
            "manifest_cid": "sha256:test",
            "event_watermark": 1,
            "table": "validation_results",
            "row_count": 1,
            "rows": [result],
        }
    )
    monkeypatch.setattr(
        materializer, "_M42_LEGACY_VALIDATION_RUNS_DIGEST", runs_digest
    )
    monkeypatch.setattr(
        materializer, "_M42_LEGACY_VALIDATION_RESULTS_DIGEST", results_digest
    )

    verified = materializer._m42_legacy_validation_table_projection(connection)

    assert verified["validation_run_count"] == 1
    assert verified["validation_result_count"] == 1
    assert verified["legacy_validation_attempt_overlay_count"] == 1

    connection.runs = [source_run]
    with pytest.raises(
        materializer.MigrationRequired,
        match="M42 exact validation table projection differs",
    ):
        materializer._m42_legacy_validation_table_projection(connection)


def test_m42_production_legacy_projection_manifest_is_closed() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m42_production_legacy_manifest_test",
    )
    manifest = materializer._m42_legacy_projection_manifest()

    assert materializer._identity(manifest) == (
        materializer._M42_LEGACY_PROJECTION_MANIFEST_CID
    )
    assert manifest["event_watermark"] == 291
    assert manifest["evidence_node_count"] == 48
    assert len(manifest["evidence_refresh_rows"]) == 9
    assert len(
        {
            row["evidence_id"]
            for row in manifest["evidence_refresh_rows"]
        }
    ) == 9
    assert all(
        row["source_row_cid"] != row["target_row_cid"]
        for row in manifest["evidence_refresh_rows"]
    )
    assert manifest["compact_validation_evidence"]["source_row_cid"] != (
        manifest["compact_validation_evidence"]["target_row_cid"]
    )
    assert manifest["validation_run_overlay"]["global_sequence"] == 106
    assert manifest["validation_run_overlay"]["source_attempt_id"]
    assert manifest["validation_run_overlay"]["target_attempt_id"] == ""
    assert materializer._M42_LEGACY_EVIDENCE_PROJECTION_DIGEST.startswith(
        "sha256:"
    )
    assert materializer._M42_LEGACY_VALIDATION_RUNS_DIGEST.startswith(
        "sha256:"
    )
    assert materializer._M42_LEGACY_VALIDATION_RESULTS_DIGEST.startswith(
        "sha256:"
    )

    strict = inspect.getsource(materializer._verify_m38_evidence_projection)
    m42 = inspect.getsource(materializer._verify_m42_exact_legacy_projection)
    assert "_m42_apply_exact_legacy_evidence_overlay" not in strict
    assert "_m42_apply_exact_legacy_evidence_overlay" in m42


class _M42ExactTargetConnection:
    def __init__(
        self,
        evidence_rows: list[tuple[object, ...]],
        target_event: tuple[object, ...] | None,
    ) -> None:
        self.evidence_rows = evidence_rows
        self.target_event = target_event

    def execute(
        self, query: str, _parameters: object = None
    ) -> _M38ProjectionResult:
        if "FROM evidence_nodes ORDER BY evidence_id" in query:
            return _M38ProjectionResult(sorted(self.evidence_rows))
        if "FROM domain_events" in query and "global_sequence=?" in query:
            return _M38ProjectionResult(
                [] if self.target_event is None else [self.target_event]
            )
        raise AssertionError(f"unexpected query: {query}")


def _synthetic_m42_exact_target(
    monkeypatch: pytest.MonkeyPatch,
    materializer: ModuleType,
) -> tuple[
    tuple[object, ...],
    tuple[object, ...],
    tuple[object, ...],
]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    prior = (
        "evidence:legacy",
        "",
        "task:legacy",
        "operator_control",
        "sha256:legacy",
        "2026-09-01T05:59:59Z",
        "{}",
    )
    target_body = {"bounded": True, "source": "M42"}
    target_digest = materializer._identity(target_body)
    target_id = content_identity(
        {
            "task_cid": materializer._M42_OPERATOR_TASK_CID,
            "evidence_kind": materializer._M42_EVIDENCE_KIND,
            "digest": target_digest,
            "body": target_body,
        }
    )
    target = (
        target_id,
        "",
        materializer._M42_OPERATOR_TASK_CID,
        materializer._M42_EVIDENCE_KIND,
        target_digest,
        materializer._M42_CONTROL_RECORDED_AT,
        materializer._canonical(target_body).decode("utf-8"),
    )
    envelope = {
        "schema": "ipfs_accelerate_py/agent-supervisor/intent-event@1",
        "event_type": "intent.evidence_recorded",
        "subject_id": target_id,
        "body": {
            "evidence_id": target_id,
            "parent_evidence_id": "",
            "task_cid": target[2],
            "evidence_kind": target[3],
            "digest": target_digest,
            "body": target_body,
            "created_at": target[5],
            "revision": 0,
        },
        "recorded_at": target[5],
        "owner_id": "sawm-r2-m42-live-source-sealer",
    }
    event_id = content_identity(
        {
            "stream_id": "stream:intent",
            "sequence": 292,
            "global_sequence": 292,
            "event_type": "intent.evidence_recorded",
            "body": envelope,
        }
    )
    event = (
        event_id,
        "stream:intent",
        292,
        292,
        "intent.evidence_recorded",
        target[2],
        "",
        "session:intent",
        target[5],
        materializer._canonical(envelope).decode("utf-8"),
    )
    projection = {
        "rows": [prior],
        "evidence_node_count": 1,
        "evidence_event_count": 1,
        "validation_event_count": 1,
        "passed_validation_event_count": 1,
        "validation_evidence_node_count": 1,
    }
    monkeypatch.setattr(
        materializer,
        "_m38_evidence_projection_from_events",
        lambda _connection, *, watermark: dict(projection),
    )
    monkeypatch.setattr(
        materializer,
        "_m42_apply_exact_legacy_evidence_overlay",
        lambda rows, *, watermark: list(rows),
    )
    monkeypatch.setattr(materializer, "_M42_LEGACY_EVIDENCE_NODE_COUNT", 1)
    monkeypatch.setattr(materializer, "_M42_LEGACY_EVIDENCE_EVENT_COUNT", 1)
    monkeypatch.setattr(materializer, "_M42_LEGACY_VALIDATION_EVENT_COUNT", 1)
    monkeypatch.setattr(
        materializer, "_M42_LEGACY_PASSED_VALIDATION_EVENT_COUNT", 1
    )
    monkeypatch.setattr(
        materializer, "_M42_LEGACY_VALIDATION_EVIDENCE_NODE_COUNT", 1
    )
    monkeypatch.setattr(
        materializer, "_M42_LEGACY_EVIDENCE_REFRESH_ROWS", MappingProxyType({})
    )
    monkeypatch.setattr(
        materializer, "_M42_LEGACY_PROJECTION_MANIFEST_CID", "sha256:test"
    )
    evidence_digest = materializer._identity(
        {
            "schema": materializer._M42_LEGACY_EVIDENCE_PROJECTION_SCHEMA,
            "manifest_cid": "sha256:test",
            "event_watermark": materializer._M42_LEGACY_PROJECTION_WATERMARK,
            "evidence_node_count": 1,
            "evidence_event_count": 1,
            "validation_event_count": 1,
            "passed_validation_event_count": 1,
            "validation_evidence_node_count": 1,
            "rows": [prior],
        }
    )
    monkeypatch.setattr(
        materializer, "_M42_LEGACY_EVIDENCE_PROJECTION_DIGEST", evidence_digest
    )
    monkeypatch.setattr(
        materializer,
        "_m42_legacy_validation_table_projection",
        lambda _connection: {
            "validation_run_count": 1,
            "validation_result_count": 1,
            "validation_runs_digest": "sha256:runs",
            "validation_results_digest": "sha256:results",
            "legacy_validation_attempt_overlay_count": 1,
            "complete_validation_table_projection_verified": True,
        },
    )
    return prior, target, event


def test_m42_exact_target_requires_event_identity_session_and_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m42_exact_target_test",
    )
    prior, target, event = _synthetic_m42_exact_target(monkeypatch, materializer)
    connection = _M42ExactTargetConnection([prior, target], event)

    verified = materializer._verify_m42_exact_legacy_projection(
        connection, expected_target_evidence_row=target
    )
    assert verified["legacy_event_watermark"] == 291
    assert verified["prior_evidence_node_count"] == 1
    assert verified["total_evidence_node_count"] == 2
    assert verified["additional_evidence_node_count"] == 1

    with pytest.raises(
        materializer.MigrationRequired, match="lacks event 292"
    ):
        materializer._verify_m42_exact_legacy_projection(
            _M42ExactTargetConnection([prior, target], None),
            expected_target_evidence_row=target,
        )

    bad_id = ("not-a-cid",) + target[1:]
    with pytest.raises(
        materializer.MigrationRequired, match="target evidence identity differs"
    ):
        materializer._verify_m42_exact_legacy_projection(
            connection, expected_target_evidence_row=bad_id
        )

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    successor_body = {"bounded": True, "source": "M43"}
    successor_digest = materializer._identity(successor_body)
    successor_id = content_identity(
        {
            "task_cid": "task:successor",
            "evidence_kind": "operator_control_successor",
            "digest": successor_digest,
            "body": successor_body,
        }
    )
    successor = (
        successor_id,
        "",
        "task:successor",
        "operator_control_successor",
        successor_digest,
        "2026-09-01T10:00:00Z",
        materializer._canonical(successor_body).decode("utf-8"),
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="exact evidence projection membership differs",
    ):
        materializer._verify_m42_exact_legacy_projection(
            _M42ExactTargetConnection([prior, target, successor], event),
            expected_target_evidence_row=target,
        )
    successor_verified = materializer._verify_m42_exact_legacy_projection(
        _M42ExactTargetConnection([prior, target, successor], event),
        expected_target_evidence_row=target,
        expected_successor_evidence_row=successor,
    )
    assert successor_verified["total_evidence_node_count"] == 2
    assert successor_verified["additional_evidence_node_count"] == 1
    with pytest.raises(
        materializer.MigrationRequired,
        match="successor evidence identity differs",
    ):
        materializer._verify_m42_exact_legacy_projection(
            _M42ExactTargetConnection([prior, target, successor], event),
            expected_target_evidence_row=target,
            expected_successor_evidence_row=("not-a-cid",) + successor[1:],
        )
    wrong_successor = successor[:4] + ("sha256:wrong",) + successor[5:]
    with pytest.raises(
        materializer.MigrationRequired,
        match="exact evidence projection rows differ",
    ):
        materializer._verify_m42_exact_legacy_projection(
            _M42ExactTargetConnection(
                [prior, target, wrong_successor], event
            ),
            expected_target_evidence_row=target,
            expected_successor_evidence_row=successor,
        )

    arbitrary_body = json.loads(str(target[6]))
    arbitrary_id = content_identity(
        {
            "task_cid": "task:arbitrary",
            "evidence_kind": target[3],
            "digest": target[4],
            "body": arbitrary_body,
        }
    )
    arbitrary = (arbitrary_id, "", "task:arbitrary") + target[3:]
    with pytest.raises(
        materializer.MigrationRequired, match="target evidence identity differs"
    ):
        materializer._verify_m42_exact_legacy_projection(
            connection, expected_target_evidence_row=arbitrary
        )

    wrong_session = event[:7] + ("session:forged",) + event[8:]
    with pytest.raises(
        materializer.MigrationRequired,
        match="target event/evidence binding differs",
    ):
        materializer._verify_m42_exact_legacy_projection(
            _M42ExactTargetConnection([prior, target], wrong_session),
            expected_target_evidence_row=target,
        )

    extra = ("evidence:unexpected",) + prior[1:]
    with pytest.raises(
        materializer.MigrationRequired,
        match="exact evidence projection membership differs",
    ):
        materializer._verify_m42_exact_legacy_projection(
            _M42ExactTargetConnection([prior, target, extra], event),
            expected_target_evidence_row=target,
        )

    with pytest.raises(
        materializer.MigrationRequired,
        match="target evidence row must be one tuple",
    ):
        materializer._verify_m42_exact_legacy_projection(
            connection,
            expected_target_evidence_row=[target, target],  # type: ignore[arg-type]
        )


def test_m42_exact_target_requires_canonical_event_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m42_canonical_target_event_test",
    )
    prior, target, event = _synthetic_m42_exact_target(monkeypatch, materializer)
    noncanonical = event[:-1] + (
        json.dumps(json.loads(str(event[-1])), indent=2, sort_keys=False),
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="target event projection body is not canonical",
    ):
        materializer._verify_m42_exact_legacy_projection(
            _M42ExactTargetConnection([prior, target], noncanonical),
            expected_target_evidence_row=target,
        )


def test_m38_evidence_projection_normalizes_named_quack_rows() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_named_quack_projection_test",
    )
    inner = {
        "evidence_id": "",
        "parent_evidence_id": "",
        "task_cid": "task:expected",
        "evidence_kind": "operator_control",
        "digest": "sha256:expected",
        "body": {"bounded": True},
        "created_at": "2026-09-01T01:10:00Z",
        "revision": 0,
    }
    inner["evidence_id"] = _m38_evidence_id(inner)
    event = _m38_projection_event(
        materializer,
        sequence=1,
        event_type="intent.evidence_recorded",
        inner=inner,
    )
    event_columns = (
        "event_id",
        "stream_id",
        "sequence",
        "global_sequence",
        "event_type",
        "task_cid",
        "attempt_id",
        "session_id",
        "recorded_at",
        "body_json",
    )
    event_row = dict(zip(event_columns, event, strict=True))
    event_row["body_json"] = json.loads(str(event_row["body_json"]))
    evidence_row = {
        "evidence_id": inner["evidence_id"],
        "parent_evidence_id": "",
        "task_cid": inner["task_cid"],
        "evidence_kind": inner["evidence_kind"],
        "digest": inner["digest"],
        "created_at": inner["created_at"],
        "body_json": dict(inner["body"]),
    }

    result = materializer._verify_m38_evidence_projection(
        _M38ProjectionConnection([event_row], [evidence_row]), watermark=1
    )

    assert result["evidence_node_count"] == 1
    assert result["evidence_event_count"] == 1
    assert result["validation_event_count"] == 0


def test_m38_evidence_projection_matches_observed_36_plus_11_shape() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_observed_evidence_shape_test",
    )
    events: list[tuple[object, ...]] = []
    evidence_nodes: list[tuple[object, ...]] = []
    for index in range(36):
        inner = {
            "evidence_id": "",
            "parent_evidence_id": "",
            "task_cid": f"task:observed:{index}",
            "evidence_kind": "operator_control",
            "digest": f"sha256:observed:{index}",
            "body": {"index": index},
            "created_at": f"2026-09-01T01:10:{index:02d}Z",
            "revision": 0,
        }
        inner["evidence_id"] = _m38_evidence_id(inner)
        events.append(
            _m38_projection_event(
                materializer,
                sequence=index + 1,
                event_type="intent.evidence_recorded",
                inner=inner,
            )
        )
        evidence_nodes.append(
            (
                inner["evidence_id"],
                "",
                inner["task_cid"],
                inner["evidence_kind"],
                inner["digest"],
                inner["created_at"],
                materializer._canonical(inner["body"]).decode("utf-8"),
            )
        )
    for index in range(11):
        inner, row = _m38_validation_projection(
            materializer,
            task_cid=f"task:validated:{index}",
            outcome="passed",
            digest=f"sha256:validation:{index}",
            recorded_at=f"2026-09-01T01:11:{index:02d}Z",
            attempt_id=f"attempt:{index}",
            argv=["python", "-m", "pytest", str(index)],
        )
        assert row is not None
        events.append(
            _m38_projection_event(
                materializer,
                sequence=37 + index,
                event_type="intent.validation_recorded",
                inner=inner,
                attempt_id=f"attempt:{index}",
            )
        )
        evidence_nodes.append(row)

    prior = materializer._verify_m38_evidence_projection(
        _M38ProjectionConnection(events, evidence_nodes), watermark=47
    )
    assert prior["evidence_node_count"] == 47
    assert prior["evidence_event_count"] == 36
    assert prior["validation_event_count"] == 11
    assert prior["passed_validation_event_count"] == 11
    assert prior["validation_evidence_node_count"] == 11

    successor = {
        "evidence_id": "",
        "parent_evidence_id": "",
        "task_cid": "task:m38-operator",
        "evidence_kind": "operator_control",
        "digest": "sha256:m38-successor",
        "body": {"migration_revision": "SAWM-R2-M38"},
        "created_at": "2026-09-01T01:12:00Z",
        "revision": 0,
    }
    successor["evidence_id"] = _m38_evidence_id(successor)
    events.append(
        _m38_projection_event(
            materializer,
            sequence=48,
            event_type="intent.evidence_recorded",
            inner=successor,
        )
    )
    evidence_nodes.append(
        (
            successor["evidence_id"],
            "",
            successor["task_cid"],
            successor["evidence_kind"],
            successor["digest"],
            successor["created_at"],
            materializer._canonical(successor["body"]).decode("utf-8"),
        )
    )
    target = materializer._verify_m38_evidence_projection(
        _M38ProjectionConnection(events, evidence_nodes), watermark=48
    )

    assert target["evidence_node_count"] == prior["evidence_node_count"] + 1
    assert target["evidence_event_count"] == prior["evidence_event_count"] + 1
    assert target["validation_event_count"] == prior["validation_event_count"]
    assert target["passed_validation_event_count"] == prior[
        "passed_validation_event_count"
    ]


def test_m38_evidence_projection_rejects_conflicting_repeated_identity() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_conflicting_evidence_event_test",
    )
    initial = {
        "evidence_id": "",
        "parent_evidence_id": "",
        "task_cid": "task:expected",
        "evidence_kind": "operator_control",
        "digest": "sha256:expected",
        "body": {"bounded": True},
        "created_at": "2026-09-01T01:10:00Z",
        "revision": 0,
    }
    initial["evidence_id"] = _m38_evidence_id(initial)
    conflicting = {**initial, "parent_evidence_id": "evidence:other-parent"}
    events = [
        _m38_projection_event(
            materializer,
            sequence=1,
            event_type="intent.evidence_recorded",
            inner=initial,
        ),
        _m38_projection_event(
            materializer,
            sequence=2,
            event_type="intent.evidence_recorded",
            inner=conflicting,
        ),
    ]

    with pytest.raises(
        materializer.MigrationRequired,
        match="M38 evidence event identity-bound fields conflict",
    ):
        materializer._verify_m38_evidence_projection(
            _M38ProjectionConnection(events, []), watermark=2
        )


def test_m38_evidence_projection_rejects_tampered_evidence_identity() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_tampered_evidence_identity_test",
    )
    inner = {
        "evidence_id": "",
        "parent_evidence_id": "",
        "task_cid": "task:expected",
        "evidence_kind": "operator_control",
        "digest": "sha256:expected",
        "body": {"bounded": True},
        "created_at": "2026-09-01T01:10:00Z",
        "revision": 0,
    }
    inner["evidence_id"] = _m38_evidence_id(inner)
    inner["digest"] = "sha256:tampered"
    event = _m38_projection_event(
        materializer,
        sequence=1,
        event_type="intent.evidence_recorded",
        inner=inner,
    )

    with pytest.raises(
        materializer.MigrationRequired,
        match="M38 evidence event envelope differs",
    ):
        materializer._verify_m38_evidence_projection(
            _M38ProjectionConnection([event], []), watermark=1
        )


def test_m38_evidence_projection_rejects_malformed_validation_identity() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_malformed_validation_event_test",
    )
    validation, _row = _m38_validation_projection(
        materializer,
        task_cid="task:validated",
        outcome="passed",
        digest="sha256:passed",
        recorded_at="2026-09-01T01:12:00Z",
        attempt_id="attempt:1",
        argv=["python", "-m", "pytest"],
    )
    validation["run_id"] = "forged:run"
    event = _m38_projection_event(
        materializer,
        sequence=1,
        event_type="intent.validation_recorded",
        inner=validation,
        attempt_id="attempt:1",
    )

    with pytest.raises(
        materializer.MigrationRequired,
        match="M38 validation event envelope differs",
    ):
        materializer._verify_m38_evidence_projection(
            _M38ProjectionConnection([event], []), watermark=1
        )


@pytest.mark.parametrize(
    ("column", "value", "error"),
    [
        (0, "forged:event", "M38 evidence event envelope differs"),
        (1, "stream:foreign", "M38 evidence event envelope differs"),
        (2, 2, "M38 evidence event sequence is invalid"),
        (7, "", "M38 evidence event envelope differs"),
        (8, "2026-09-01T01:10:01Z", "M38 evidence event envelope differs"),
    ],
)
def test_m38_evidence_projection_rejects_malformed_event_row_binding(
    column: int,
    value: object,
    error: str,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        f"sawm_materializer_m38_malformed_event_row_{column}_test",
    )
    inner = {
        "evidence_id": "",
        "parent_evidence_id": "",
        "task_cid": "task:expected",
        "evidence_kind": "operator_control",
        "digest": "sha256:expected",
        "body": {"bounded": True},
        "created_at": "2026-09-01T01:10:00Z",
        "revision": 0,
    }
    inner["evidence_id"] = _m38_evidence_id(inner)
    event = list(
        _m38_projection_event(
            materializer,
            sequence=1,
            event_type="intent.evidence_recorded",
            inner=inner,
        )
    )
    event[column] = value

    with pytest.raises(materializer.MigrationRequired, match=error):
        materializer._verify_m38_evidence_projection(
            _M38ProjectionConnection([tuple(event)], []), watermark=2
        )


def test_m38_evidence_projection_allows_independent_outer_and_inner_recorded_at() -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_split_recorded_at_test",
    )
    inner = {
        "evidence_id": "",
        "parent_evidence_id": "",
        "task_cid": "task:expected",
        "evidence_kind": "operator_control",
        "digest": "sha256:expected",
        "body": {"bounded": True},
        "created_at": "2026-09-01T01:10:00Z",
        "revision": 0,
    }
    inner["evidence_id"] = _m38_evidence_id(inner)
    event = list(
        _m38_projection_event(
            materializer,
            sequence=1,
            event_type="intent.evidence_recorded",
            inner=inner,
        )
    )
    envelope = json.loads(str(event[9]))
    envelope["recorded_at"] = "2026-09-01T01:10:01Z"
    event[8] = envelope["recorded_at"]
    event[9] = materializer._canonical(envelope).decode("utf-8")
    event[0] = content_identity(
        {
            "stream_id": "stream:intent",
            "sequence": 1,
            "global_sequence": 1,
            "event_type": "intent.evidence_recorded",
            "body": envelope,
        }
    )

    evidence_row = (
        inner["evidence_id"],
        inner["parent_evidence_id"],
        inner["task_cid"],
        inner["evidence_kind"],
        inner["digest"],
        inner["created_at"],
        materializer._canonical(inner["body"]).decode("utf-8"),
    )
    verified = materializer._verify_m38_evidence_projection(
        _M38ProjectionConnection([tuple(event)], [evidence_row]), watermark=1
    )

    assert verified["evidence_node_count"] == 1
    assert event[8] != inner["created_at"]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing", "M38 evidence-node projection is missing rows"),
        ("extra", "M38 evidence-node projection has extra rows"),
        ("duplicate", "M38 evidence-node identities are duplicate"),
        ("conflicting", "M38 evidence-node/event projection conflicts"),
    ],
)
def test_m38_evidence_projection_rejects_nonexact_rows(
    mutation: str,
    error: str,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        f"sawm_materializer_m38_{mutation}_evidence_projection_test",
    )
    inner = {
        "evidence_id": "",
        "parent_evidence_id": "",
        "task_cid": "task:expected",
        "evidence_kind": "operator_control",
        "digest": "sha256:expected",
        "body": {"bounded": True},
        "created_at": "2026-09-01T01:10:00Z",
        "revision": 0,
    }
    inner["evidence_id"] = _m38_evidence_id(inner)
    expected_row = (
        inner["evidence_id"],
        inner["parent_evidence_id"],
        inner["task_cid"],
        inner["evidence_kind"],
        inner["digest"],
        inner["created_at"],
        materializer._canonical(inner["body"]).decode("utf-8"),
    )
    orphan_row = (
        "evidence:orphan",
        "",
        "task:orphan",
        "unbound",
        "sha256:orphan",
        inner["created_at"],
        "{}",
    )
    conflicting_row = (
        *expected_row[0:4],
        "sha256:conflicting",
        *expected_row[5:],
    )
    actual_rows = {
        "missing": [],
        "extra": [expected_row, orphan_row],
        "duplicate": [expected_row, expected_row],
        "conflicting": [conflicting_row],
    }[mutation]
    event = _m38_projection_event(
        materializer,
        sequence=1,
        event_type="intent.evidence_recorded",
        inner=inner,
    )

    with pytest.raises(
        materializer.MigrationRequired,
        match=error,
    ):
        materializer._verify_m38_evidence_projection(
            _M38ProjectionConnection([event], actual_rows), watermark=1
        )


def test_m38_target_increment_uses_the_full_event_derived_projection() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_full_projection_increment_test",
    )
    live = inspect.getsource(materializer._verify_m38_live_materialization)

    assert "prior_evidence_event_count" not in live
    assert "prior_evidence_projection[\"evidence_node_count\"] + 1" in live
    assert "prior_evidence_projection[\"validation_event_count\"]" in live
    assert "prior_evidence_projection[\"passed_validation_event_count\"]" in live


def test_m38_presence_masks_m37_and_keeps_all_history_nonactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m38_presence_test",
    )
    key = "pre_authoritative_custody_restart_successor_materialization"
    reference = {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M38",
        "authority_cid": "sha256:PENDING_M38_AUTHORITY_CID",
    }
    scheduler = {key: reference}
    migration = {key: reference}
    seal = {f"{key}_cid": "sha256:PENDING_M38_AUTHORITY_CID"}
    monkeypatch.setattr(
        board, "_m38_migration_errors", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        board,
        "_m37_migration_errors",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("M37 must remain masked while M38 is complete")
        ),
    )
    historical_calls: list[bool] = []

    def historical(*_args: object, **kwargs: object) -> list[str]:
        historical_calls.append(kwargs.get("require_active_runtime") is False)
        return []

    for name in (
        "_m36_migration_errors", "_m35_migration_errors",
        "_m34_migration_errors", "_m33_migration_errors",
        "_m32_migration_errors", "_m31_migration_errors",
        "_m30_migration_errors", "_m29_migration_errors",
        "_m28_migration_errors", "_m27_migration_errors",
        "_m26_migration_errors", "_m25_migration_errors",
        "_m24_migration_errors", "_m23_migration_errors",
        "_m22_migration_errors", "_m21_migration_errors",
        "_m20_migration_errors", "_m19_migration_errors",
        "_m18_migration_errors", "_m17_migration_errors",
        "_m16_migration_errors",
    ):
        monkeypatch.setattr(board, name, historical)
    assert board._active_successor_migration_errors(
        scheduler, seal, migration
    ) == []
    assert historical_calls == [True] * 21


@pytest.mark.parametrize(
    "presence",
    (
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ),
)
def test_m38_partial_three_surface_declarations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    presence: tuple[bool, bool, bool],
) -> None:
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m38_partial_surface_test_" + "".join(map(str, presence)),
    )
    key = "pre_authoritative_custody_restart_successor_materialization"
    reference = {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M38",
        "authority_cid": "sha256:PENDING_M38_AUTHORITY_CID",
    }
    scheduler = {key: reference} if presence[0] else {}
    migration = {key: reference} if presence[1] else {}
    seal = {f"{key}_cid": reference["authority_cid"]} if presence[2] else {}
    monkeypatch.setattr(board, "_m38_migration_errors", lambda *_args: [])
    errors = board._active_successor_migration_errors(
        scheduler, seal, migration
    )
    assert errors == ["M38 custody restart authority is only partially declared"]


def test_operator_executes_m38_prestart_before_m37(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m38_runtime_precedence_test",
    )
    calls: list[str] = []
    admitted = {
        "valid": True,
        "action": "admitted_pre_authoritative_custody_restart_to_generation_30",
        "database_path": str((operator.REPO_ROOT / operator._M38_STORE_ID).resolve()),
        "coordination_path": str(
            (operator.REPO_ROOT / operator._M38_COORDINATION_STORE_ID).resolve()
        ),
        "prior_generation": operator._M38_PRIOR_GENERATION,
        "target_generation": operator._M38_GENERATION,
        "prior_event_watermark": operator._M38_PRIOR_EVENT_WATERMARK,
        "prior_projection_cid": operator._M38_PRIOR_PROJECTION_CID,
        "failed_m37_attempt_prestart_evidence_bound": True,
        "post_failure_checkpoint_bytes_verified": True,
        "read_replica_bytes_verified": True,
        "stale_lane_pid_evidence_verified": True,
        "target_projection_recomputed": True,
        "pre_authoritative_custody_required": True,
        "prestart_authorization_consumed": False,
    }

    class Materializer:
        @staticmethod
        def build_population(_root: Path) -> dict[str, object]:
            return {}

        @staticmethod
        def _assert_committed_clean_source(*_args: object) -> None:
            return None

        @staticmethod
        def _check_m38_prestart_admission(
            _root: Path, _config: Mapping[str, object]
        ) -> dict[str, object]:
            calls.append("m38")
            return admitted

        @staticmethod
        def _check_m37_prestart_admission(*_args: object) -> dict[str, object]:
            calls.append("m37")
            raise AssertionError("M37 must not run when M38 is declared")

    monkeypatch.setattr(operator, "_validator", lambda *_args: {"valid": True})
    monkeypatch.setattr(operator, "_materializer", Materializer)
    marker = {"migration_revision": "SAWM-R2-M38"}
    monkeypatch.setattr(
        operator, "_active_source_repair_materialization", lambda _config: marker
    )
    result = operator._validate_offline_quack_start(
        {
            operator._M38_SUCCESSOR_KEY: {},
            operator._M37_SUCCESSOR_KEY: {},
        }
    )
    assert calls == ["m38"]
    assert result["prior_authority"] == marker
    assert result["store"] == admitted


def test_m38_prestart_live_verification_and_receipt_controls_are_closed() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m38_closed_controls_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m38_closed_controls_test",
    )
    prestart = inspect.getsource(materializer._check_m38_prestart_admission)
    live = inspect.getsource(materializer._verify_m38_live_materialization)
    receipt = inspect.getsource(materializer._expected_m38_source_successor_receipt)
    dispatch = inspect.getsource(materializer.materialize)
    check_dispatch = inspect.getsource(materializer.check_materialized)
    marker = inspect.getsource(operator._require_m38_source_successor_marker)
    offline_start = inspect.getsource(operator._validate_offline_quack_start)

    assert "_assert_offline(control)" in prestart
    assert "if os.path.lexists(receipt_path):" in prestart
    assert "_verify_m38_failed_attempt_physical_evidence" in prestart
    assert 'control.parent / "control.read-replica.duckdb.wal"' in prestart
    assert '"admitted_pre_authoritative_custody_restart_to_generation_30"' in prestart
    assert "read_only=False" not in prestart
    assert '"authoritative": False' in receipt
    assert '"superseded_m37_authority_cid": _M38_M37_AUTHORITY_CID' in receipt
    assert dispatch.index(
        "if _m38_successor_configured_on_any_surface(root, config):"
    ) < dispatch.index(
        "if _m37_successor_configured_on_any_surface(root, config):"
    )
    assert check_dispatch.index(
        "if _m38_successor_configured_on_any_surface(root, config):"
    ) < check_dispatch.index(
        "if _m37_successor_configured_on_any_surface(root, config):"
    )
    assert "M38 marker requires exact live materializer verification" in marker
    assert offline_start.index("if _M38_SUCCESSOR_KEY in config:") < (
        offline_start.index("if _M37_SUCCESSOR_KEY in config:")
    )
    assert "_check_m38_prestart_admission" in offline_start
    assert '"pre_authoritative_custody_repair_source_verified": True' in live
    assert "_verify_m38_evidence_projection" in live
    assert 'prior_evidence_projection["evidence_node_count"] + 1' in live
    assert 'prior_evidence_projection["validation_event_count"]' in live
    assert '"failed_m37_attempt_authority_bound"' in receipt
    assert '"sidecars_preserved"' not in receipt
    assert '"coordination_and_historical_receipt_bytes_preserved": True' in receipt
    assert (
        '"sealed_prestart_contract_requires_post_failure_read_replica_identity": True'
        in receipt
    )
    assert '"read_replica_rebuilt_non_authoritatively_on_start": True' in receipt
    assert "record_completion" not in inspect.getsource(materializer._materialize_m38)


def test_m37_authority_pins_reboot_recovery_generation_and_source_chain() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m37_authority_test",
    )
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m37_authority_test",
    )
    key = "post_reboot_generation_restart_successor_materialization"
    authority = (
        materializer._expected_m37_post_reboot_generation_restart_authority()
    )
    identity_state = dependencies._m37_source_chain_identity_state(
        materializer, authority
    )
    reference = dependencies._m37_authority_reference_for_source_state(
        materializer, authority
    )
    scheduler = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    assert scheduler[key] == reference == migration[key]
    expected_authority_cid = (
        "sha256:PENDING_M37_AUTHORITY_CID"
        if identity_state == "placeholder"
        else materializer._identity(authority)
    )
    if identity_state == "sealed":
        assert expected_authority_cid == (
            "sha256:c776180b7e65de98d5de235765db60148f7693148512b335260ddb772563a795"
        )
    assert seal[f"{key}_cid"] == reference["authority_cid"] == expected_authority_cid
    assert reference["schema"] == "sawm/operator-control-authority-reference@1"
    assert reference["migration_revision"] == "SAWM-R2-M37"

    assert authority["schema"] == (
        "sawm/post-reboot-generation-restart-successor-authorization@1"
    )
    assert authority["supersession_mode"] == (
        "generation_bearing_post_reboot_restart_source_seal"
    )
    assert authority["control_recorded_at"] == "2026-09-01T00:10:00Z"
    assert authority["target_generation"] == 30
    assert authority["runtime_binding"]["prior_event_watermark"] == 290
    assert authority["target_event_watermark"] == 291
    assert authority["target_projection_cid"] == (
        "baguqeeravycbuo73fyu5mpad55qi5duk3la53lubqeu7nu6kjtnahehjtnsq"
    )
    prior = authority["prior_authority"]
    assert prior["schema"] == "sawm/current-operational-head@1"
    assert prior["event_watermark"] == 290
    assert prior["event_prefix_sha256"] == (
        "27ec7ecda1536d1bf7ed4b7e709b3b7284efcc84fbfb804b63db437ed31bd22c"
    )
    assert prior["projection_cid"] == (
        "baguqeerahwerrrfx6cx6ukpljlp2r4i32lkac3bnhq5ej2f3hozg7cnt6shq"
    )
    assert prior["semantic_authority_digest"] == (
        "sha256:f51d9cb949538441218254297e279fa2bf5884e1bdc9d20693cf8213db841dde"
    )
    assert prior["expected_task_heads"]["SAWM-006"] == {
        "status": "in_progress",
        "revision": 9,
    }
    assert prior["expected_task_heads"]["SAWM-008"] == {
        "status": "in_progress",
        "revision": 11,
    }
    m36_anchor = authority["m36_historical_anchor"]
    assert m36_anchor["migration_revision"] == "SAWM-R2-M36"
    assert m36_anchor["event_watermark"] == 286
    assert m36_anchor["event_prefix_sha256"] == (
        "d415613fd55f337c0142ce09ce770e73b4956006fab77cab50fbf19ea5bd9b00"
    )
    assert m36_anchor["receipt_cid"] == (
        "sha256:b37bb7a32eefdb431e6bad9faec723d58b7cf1dc69fdc52bac43621da6047c1e"
    )
    suffix = authority["post_m36_operational_suffix"]
    assert suffix["from_event_exclusive"] == 286
    assert suffix["to_event_inclusive"] == 290
    assert [event["global_sequence"] for event in suffix["events"]] == [
        287, 288, 289, 290,
    ]
    assert [event["task_alias"] for event in suffix["events"]] == [
        "SAWM-006", "SAWM-008", "SAWM-008", "SAWM-006",
    ]
    assert [
        (
            event["global_sequence"], event["event_id"],
            event["previous_status"], event["status"], event["revision"],
            event["operation"], event["attempt_id"], event["claim_id"],
            event["lease_id"], event["recorded_at"],
        )
        for event in suffix["events"]
    ] == [
        (
            287,
            "baguqeeraeetkh52dorimsks4rwzcnogldijz3xq4qv4ooz3xrwdquuwfpsrq",
            "in_progress", "retrying", 8,
            "automatic_expired_attempt_requeue",
            "attempt:64d8abb8c2244a7a94f3e7413e14e5fc",
            "claim:1ca15c7f26e941b9a1a948fa0c1e4d33",
            "lease:111faf11f21147b6bc0e82c9859692eb",
            "2026-08-31T21:29:42Z",
        ),
        (
            288,
            "baguqeerai5nsslqqy4nottsoihkgnzvjuc43ug6rzbzazxdv226e2dvx5xxa",
            "in_progress", "retrying", 10,
            "automatic_expired_attempt_requeue",
            "attempt:0b3c42d98c034946b05b9254bdfaa848",
            "claim:d397d36afc9045aaaa9a79dbe613963a",
            "lease:1cee87cc686642a185716677b0aab898",
            "2026-08-31T21:29:50Z",
        ),
        (
            289,
            "baguqeeraw2au4vfnlmjqv7pwf5n3a7r7qbkvofenehr5pfoa7telr3626mtq",
            "retrying", "in_progress", 11, "database_claim",
            "attempt:a4e1157f12af486ea3db76868789b6d1",
            "claim:698a3495a57744488e936f1b9e4c742e",
            "lease:fcddc47abf8c4701a7a17036820dec59",
            "2026-08-31T21:30:06Z",
        ),
        (
            290,
            "baguqeeraqyqkvcpzuyih65paddgdg3g6sgxot2ocecszo2u444fovy37fxoq",
            "retrying", "in_progress", 9, "database_claim",
            "attempt:8b2f351da39e4ad4b677f068d0bdb159",
            "claim:e0d33190d99444239d7faa14e9149566",
            "lease:15938d40060c4a50ac751ede8150385c",
            "2026-08-31T21:30:25Z",
        ),
    ]
    assert suffix["expired_attempt_provider_invocation_count"] == 0
    assert suffix["expired_attempt_effect_claim_count"] == 0
    assert suffix["accepted_completion_changes"] == 0
    stopped = authority["stopped_owner"]
    assert (stopped["generation"], stopped["target_generation"]) == (29, 30)
    assert stopped["status"] == "stopped"
    assert stopped["stopped_at"] == "2026-09-01T00:00:51Z"
    recovery = authority["stale_owner_recovery"]
    assert recovery["receipt_cid"] == (
        "baguqeerayoxp2turydlpaytskth6jevi23wdp3iiheubcsu2q5hs3fk55dma"
    )
    assert recovery["owner_liveness"] == "dead"
    assert recovery["database_bookkeeping_settled"] is True
    assert recovery["task_completion_authority"] is False
    assert authority["accepted_control_plane_repair"]["authority_weakened"] is False
    assert authority["accepted_control_plane_repair"]["receipts_rewritten"] is False
    assert authority["exact_changes"] == {
        "event_suffix_length": 1,
        "evidence_node_changes": 1,
        "store_generation_row_changes": 1,
        "state_server_row_changes": 1,
        "server_epoch_row_changes": 1,
        "capability_snapshot_row_changes": 1,
        "credential_row_changes": 1,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "goal_revision_changes": 0,
        "plan_revision_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "coordination_semantic_changes": 0,
        "sidecar_semantic_changes": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "implementation_provider_invocations": 0,
        "merge_attempt_changes": 0,
    }
    assert authority["preservation"]["m36_historical_anchor_preserved"] is True
    assert authority["preservation"]["post_m36_operational_suffix_preserved"] is True
    assert authority["preservation"]["generation_29_preserved_stopped"] is True
    assert authority["preservation"]["task_heads_preserved"] is True
    assert authority["preservation"]["plan_head_preserved"] is True
    assert authority["preservation"]["worker_self_approval"] is False
    assert dict(materializer._validated_m37_live_preflight_contract(authority)) == (
        authority["live_preflight_contract"]
    )

    chain = authority["source_chain"]
    assert chain["base_control_commit"] == (
        "a3db1cde328c5aeba86896d4f6813821251ceb7e"
    )
    assert chain["base_control_tree"] == (
        "fba8c205656afda738ac5f14c2841fb1da452ea0"
    )
    if identity_state == "sealed":
        assert chain["initial_control_commit"] == (
            "fb6672403850bc4b473db8e5e759176e3028adc0"
        )
        assert chain["initial_control_tree"] == (
            "bb9132b3dd333ad11c60efac45186e1fc6cdb11a"
        )
    assert len(authority["operator_control_paths"]) == 9
    assert set(chain["initial_control_blobs"]) == set(
        authority["operator_control_paths"]
    )

    final = copy.deepcopy(authority)
    final_chain = final["source_chain"]
    final_chain["initial_control_commit"] = "1" * 40
    final_chain["initial_control_tree"] = "2" * 40
    final_chain["final_reseal_parent"] = "1" * 40
    final_chain["initial_control_blobs"] = {
        path: f"{index + 3:x}" * 40
        for index, path in enumerate(authority["operator_control_paths"])
    }
    fake_materializer = SimpleNamespace(
        _M37_INITIAL_CONTROL_COMMIT="1" * 40,
        _M37_INITIAL_CONTROL_TREE="2" * 40,
        _M37_INITIAL_CONTROL_BLOBS=final_chain["initial_control_blobs"],
    )
    assert dependencies._m37_source_chain_identity_state(
        fake_materializer, final
    ) == "sealed"
    mixed = copy.deepcopy(final)
    first_path = next(iter(mixed["source_chain"]["initial_control_blobs"]))
    mixed["source_chain"]["initial_control_blobs"][first_path] = (
        "PENDING_M37_MIXED_BLOB"
    )
    mixed_materializer = SimpleNamespace(
        _M37_INITIAL_CONTROL_COMMIT="1" * 40,
        _M37_INITIAL_CONTROL_TREE="2" * 40,
        _M37_INITIAL_CONTROL_BLOBS=mixed["source_chain"]["initial_control_blobs"],
    )
    with pytest.raises(RuntimeError, match="mix placeholder and sealed"):
        dependencies._m37_source_chain_identity_state(mixed_materializer, mixed)
    noncanonical = copy.deepcopy(authority)
    noncanonical_chain = noncanonical["source_chain"]
    noncanonical_chain["initial_control_commit"] = "0" * 40
    noncanonical_chain["initial_control_tree"] = "0" * 40
    noncanonical_chain["final_reseal_parent"] = "0" * 40
    noncanonical_chain["initial_control_blobs"] = {
        path: "0" * 40 for path in authority["operator_control_paths"]
    }
    noncanonical_materializer = SimpleNamespace(
        _M37_INITIAL_CONTROL_COMMIT="0" * 40,
        _M37_INITIAL_CONTROL_TREE="0" * 40,
        _M37_INITIAL_CONTROL_BLOBS=noncanonical_chain[
            "initial_control_blobs"
        ],
    )
    with pytest.raises(RuntimeError, match="noncanonical placeholder"):
        dependencies._m37_source_chain_identity_state(
            noncanonical_materializer, noncanonical
        )


def test_m37_placeholder_controls_fail_closed_at_launch_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m37_placeholder_rejection_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m37_placeholder_rejection_test",
    )
    key = "post_reboot_generation_restart_successor_materialization"
    pending = {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M37",
        "authority_cid": "sha256:PENDING_M37_AUTHORITY_CID",
    }
    with pytest.raises(
        materializer.MaterializationError,
        match="M37 post-reboot generation restart authority is invalid",
    ):
        materializer._m37_successor_configured({key: pending})
    monkeypatch.setattr(
        materializer,
        "_M37_INITIAL_CONTROL_COMMIT",
        "PENDING_M37_INITIAL_CONTROL_COMMIT",
    )
    monkeypatch.setattr(
        materializer,
        "_M37_INITIAL_CONTROL_TREE",
        "PENDING_M37_INITIAL_CONTROL_TREE",
    )
    monkeypatch.setattr(
        materializer,
        "_M37_INITIAL_CONTROL_BLOBS",
        MappingProxyType(
            {
                path: f"PENDING_M37_INITIAL_BLOB_{index:02d}"
                for index, path in enumerate(
                    sorted(materializer._M37_OPERATOR_CONTROL_PATHS), start=1
                )
            }
        ),
    )
    with pytest.raises(
        materializer.MaterializationError,
        match="M37 source identities are not resealed",
    ):
        materializer._assert_m37_source_delta(
            REPO_ROOT,
            {},
            materializer._expected_m37_post_reboot_generation_restart_authority(),
        )
    with pytest.raises(
        operator.OperatorError,
        match="active M37 generation restart authority is invalid",
    ):
        operator._active_source_repair_materialization({key: pending})


def test_m37_presence_masks_m36_and_keeps_all_history_nonactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m37_presence_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m37_presence_test",
    )
    key = "post_reboot_generation_restart_successor_materialization"
    reference = {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M37",
        "authority_cid": "sha256:PENDING_M37_AUTHORITY_CID",
    }
    scheduler = {key: reference}
    migration = {key: reference}
    seal = {f"{key}_cid": "sha256:PENDING_M37_AUTHORITY_CID"}
    assert dependency._m37_successor_declared(scheduler, {}, {}) is True
    assert dependency._m37_successor_declared({}, seal, {}) is True
    assert dependency._m37_successor_declared({}, {}, migration) is True
    assert dependency._m37_successor_declared({}, {}, {}) is False

    monkeypatch.setattr(
        board, "_m37_migration_errors", lambda *_args, **_kwargs: ["M37 active"]
    )
    historical_calls: list[bool] = []

    def historical(*_args: object, **kwargs: object) -> list[str]:
        historical_calls.append(kwargs.get("require_active_runtime") is False)
        return []

    for name in (
        "_m36_migration_errors", "_m35_migration_errors",
        "_m34_migration_errors", "_m33_migration_errors",
        "_m32_migration_errors", "_m31_migration_errors",
        "_m30_migration_errors", "_m29_migration_errors",
        "_m28_migration_errors", "_m27_migration_errors",
        "_m26_migration_errors", "_m25_migration_errors",
        "_m24_migration_errors", "_m23_migration_errors",
        "_m22_migration_errors", "_m21_migration_errors",
        "_m20_migration_errors", "_m19_migration_errors",
        "_m18_migration_errors", "_m17_migration_errors",
        "_m16_migration_errors",
    ):
        monkeypatch.setattr(board, name, historical)
    assert board._active_successor_migration_errors(
        scheduler, seal, migration
    ) == ["M37 active"]
    assert historical_calls == [True] * 21

    historical_calls.clear()
    partial_errors = board._active_successor_migration_errors(scheduler, {}, {})
    assert "M37 active" in partial_errors
    assert any("partially declared" in error for error in partial_errors)
    assert historical_calls == [True] * 21


def test_m37_validates_m36_source_at_preserved_historical_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m37_historical_m36_head_test",
    )
    scheduler = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    observed: list[str | None] = []

    def source_chain(
        _root: Path,
        _materializer: object,
        _authority: Mapping[str, object],
        *,
        current_head: str | None = None,
    ) -> list[str]:
        observed.append(current_head)
        return []

    monkeypatch.setattr(dependencies, "_m36_source_chain_errors", source_chain)
    assert dependencies._m36_operator_task_binding_correction_successor_errors(
        scheduler,
        seal,
        migration,
        root=REPO_ROOT,
        require_active_runtime=False,
    ) == []
    assert observed == ["a3db1cde328c5aeba86896d4f6813821251ceb7e"]


def test_m37_prestart_live_verification_and_receipt_controls_are_closed() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m37_closed_controls_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m37_closed_controls_test",
    )
    prestart = inspect.getsource(materializer._check_m37_prestart_admission)
    live = inspect.getsource(materializer._verify_m37_live_materialization)
    receipt = inspect.getsource(materializer._expected_m37_source_successor_receipt)
    publish = inspect.getsource(materializer._ensure_m37_source_successor_receipt)
    materialize = inspect.getsource(materializer._materialize_m37)
    check = inspect.getsource(materializer._check_m37_materialized)
    dispatch = inspect.getsource(materializer.materialize)
    check_dispatch = inspect.getsource(materializer.check_materialized)
    cli_dispatch = inspect.getsource(materializer.main)
    marker = inspect.getsource(operator._require_m37_source_successor_marker)
    offline_start = inspect.getsource(operator._validate_offline_quack_start)

    assert prestart.index("_assert_offline(control)") < prestart.index(
        "if os.path.lexists(receipt_path):"
    )
    assert prestart.index("if os.path.lexists(receipt_path):") < prestart.index(
        "duckdb.connect(str(control), read_only=True)"
    )
    assert "_verify_m37_preserved_receipts" in prestart
    assert "_M37_PRIOR_EVENT_PREFIX_SHA256" in prestart
    assert '"m36_historical_anchor_verified"' in prestart
    assert '"post_m36_operational_suffix_verified": True' in prestart
    assert '"prestart_authorization_consumed": False' in prestart
    assert "read_only=False" not in prestart

    assert "_inspect_m37_live_projection" in live
    assert "_inspect_m37_generation_restart_rows" in live
    assert "with source.intent._connection(write=False)" in live
    assert "_M37_PRIOR_EVENT_PREFIX_SHA256" in live
    assert '"full_event_and_evidence_body_verified": True' in live
    assert '"m36_historical_anchor_verified"' in live
    assert '"post_m36_operational_suffix_verified": True' in live
    assert '"accepted_completion_changes": 0' in live
    assert '"worker_self_approval": False' in live

    assert '"authoritative": False' in receipt
    assert '"control_database_is_authority": True' in receipt
    assert '"target_generation": _M37_TARGET_GENERATION' in receipt
    assert '"m36_receipt_cid": _M37_M36_RECEIPT_CID' in receipt
    assert '"stale_owner_recovery_receipt_cid": _M37_RECOVERY_RECEIPT_CID' in receipt
    assert '"m36_historical_anchor_verified"' in receipt
    assert '"post_m36_operational_suffix_verified"' in receipt
    assert 'result["receipt_cid"] = _identity(result)' in receipt
    assert "fcntl.LOCK_EX" in publish
    assert publish.index("if os.path.lexists(path):") < publish.index(
        "os.replace(temporary, path)"
    )
    assert 'raise MigrationRequired("M37 source successor receipt differs")' in publish

    assert materialize.index("if os.path.lexists(receipt_path):") < (
        materialize.index("source.record_evidence(")
    )
    assert materialize.index("_inspect_m37_live_projection(") < (
        materialize.index("source.record_evidence(")
    )
    assert materialize.index("source.record_evidence(") < materialize.index(
        "_verify_m37_live_materialization("
    )
    assert materialize.index("_verify_m37_live_materialization(") < (
        materialize.index("_ensure_m37_source_successor_receipt(")
    )
    assert "preserved_after != preserved" in materialize
    assert "coordination_before" in materialize
    assert "record_completion" not in materialize
    assert "record_plan" not in materialize
    assert check.index("if not os.path.lexists(receipt_path):") < check.index(
        "_verify_m37_live_materialization("
    )
    assert check.index("_verify_m37_live_materialization(") < check.index(
        "if observed != expected:"
    )
    assert dispatch.index(
        "if _m37_successor_configured_on_any_surface(root, config):"
    ) < dispatch.index(
        "if _m36_successor_configured_on_any_surface(root, config):"
    )
    assert check_dispatch.index(
        "if _m37_successor_configured_on_any_surface(root, config):"
    ) < check_dispatch.index(
        "if _m36_successor_configured_on_any_surface(root, config):"
    )
    assert (
        '"post_reboot_generation_restart_successor_materialization"'
        in cli_dispatch
    )

    assert "_require_m36_source_successor_marker" in marker
    assert "m36_claimed != materializer._identity(m36_unhashed)" in marker
    assert "checked is None" in marker
    assert 'checked.get("m36_historical_anchor_verified") is not True' in marker
    assert 'checked.get("post_m36_operational_suffix_verified") is not True' in marker
    assert 'observed.get("worker_self_approval") is not False' in marker
    assert offline_start.index("if _M38_SUCCESSOR_KEY in config:") < (
        offline_start.index("if _M37_SUCCESSOR_KEY in config:")
    )
    assert offline_start.index("if _M37_SUCCESSOR_KEY in config:") < (
        offline_start.index(
            'if "operator_task_binding_correction_successor_materialization" in config:'
        )
    )
    assert "_check_m37_prestart_admission" in offline_start
    assert '"prestart_authorization_consumed") is not False' in offline_start


def test_m36_authority_corrects_operator_task_binding_and_preserves_m35_failure() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m36_authority_test",
    )
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m36_authority_test",
    )
    key = "operator_task_binding_correction_successor_materialization"
    authority = (
        materializer._expected_m36_operator_task_binding_correction_authority()
    )
    reference = materializer._m36_authority_reference()
    scheduler = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json").read_text(encoding="utf-8")
    )
    migration = json.loads(
        (REPO_ROOT / "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json").read_text(encoding="utf-8")
    )
    seal = json.loads(
        (REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json").read_text(encoding="utf-8")
    )
    assert scheduler[key] == reference == migration[key]
    assert seal[f"{key}_cid"] == reference["authority_cid"]
    assert reference == {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M36",
        "authority_cid": materializer._identity(authority),
    }
    assert authority["supersession_mode"] == (
        "append_only_operator_task_binding_correction"
    )
    assert authority["control_recorded_at"] == "2026-08-31T20:50:00Z"
    assert authority["target_generation"] == 29
    assert authority["runtime_binding"]["prior_event_watermark"] == 285
    assert authority["target_event_watermark"] == 286
    assert authority["target_projection_cid"] == (
        "baguqeeravzrhagxizn7o45ukuevzkhreb7dzpabd4g4kmyci2if5rxr32dda"
    )
    assert authority["prior_authority"]["migration_revision"] == "SAWM-R2-M34"
    assert authority["prior_authority"]["m34_receipt_cid"] == (
        "sha256:759b71e0d0fa73a1ac81bb98fb1b96c93f3150b09667cd19fc05330b34b50b01"
    )
    superseded = authority["superseded_source_authority"]
    assert superseded == {
        "migration_revision": "SAWM-R2-M35",
        "authority_cid": "sha256:be555c52e0ddf0737b1f3796ab21618e11226c7fae14d176c50d0cb4edfaf5d6",
        "base_control_commit": "4dfe1c4c81ffd65f6a2d5c5cdc38b1cd33f1f443",
        "initial_control_commit": "27c5e1e5925228757fc02378eb9d3b2a8addc60e",
        "initial_control_tree": "75f9c888b22911d47e941fb03b771c8d8f92dd99",
        "final_control_commit": "5bbf2dec97458585ee95034c986057a1552b805d",
        "final_control_tree": "954736c103dd644d7896e032c9676ed38d989831",
        "materialized": False,
        "receipt_created": False,
    }
    failed = authority["failed_m35_append"]
    assert failed["failure_kind"] == "operator_task_cid_not_found"
    assert failed["sealed_target_task_cid"] == (
        "sha256:308a38585461080c06bf51f36a5b9cff75c4bf5a6e88ddcbccbca73198a51d1d"
    )
    assert failed["sealed_target_task_present"] is False
    assert failed["m35_event_appended"] is False
    assert failed["m35_receipt_created"] is False
    assert failed["event_watermark_before"] == failed["event_watermark_after"] == 285
    target = authority["target_authority"]
    assert target["operator_task_alias"] == "SAWM-000"
    assert target["operator_task_cid"] == (
        "sha256:8b8f43dd51ea4d8467af0e5cae4100478f16666d36c6f4fad49c23fd8e43a3d6"
    )
    assert (target["operator_task_status"], target["operator_task_revision"]) == (
        "completed", 2,
    )
    chain = authority["source_chain"]
    assert chain["base_control_commit"] == (
        "5bbf2dec97458585ee95034c986057a1552b805d"
    )
    assert chain["base_control_tree"] == "954736c103dd644d7896e032c9676ed38d989831"
    assert len(authority["operator_control_paths"]) == 9
    assert chain["initial_control_commit"] == (
        "e208bc490b8d12f6d86b286d98d1ac63bb4e62be"
    )
    assert chain["initial_control_tree"] == (
        "53fb95b9fd8e4f6f776c0380b0f1f959ddd58326"
    )
    assert chain["initial_control_blobs"] == {
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json": "ef87cc2eca909c0067ddb3367db3a94f90772c73",
        "config/semantic_addressed_world_model_dependencies.seal.json": "085a698e10b7f81f7c3f03fb8ebfc55177ca1b88",
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md": "f6be8bee8e7675437c9ab67bdbe8d3a44c2470bf",
        "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json": "af7deb1f4f3a004eab59a4a55109a7ad175ff33e",
        "scripts/materialize_semantic_addressed_world_model_program.py": "2c10c15764112afbc2f7e286f753386b13c7e06c",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py": "7f9cfa6c757f9f91c57db1129b293fa8f048b79b",
        "scripts/validate_semantic_addressed_world_model_board.py": "8bdeb08a0f7bb059bc9b6f18363d3b2d4076d916",
        "scripts/validate_semantic_addressed_world_model_dependencies.py": "1f36af05e5d5bfc25b607551e406907d8e2b5953",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py": "5e067d5ea1ee48a0ddbcb8e6eb0896c53c33be55",
    }
    assert dependencies._m36_source_chain_identity_state(
        materializer, authority
    ) == "sealed"
    assert dict(materializer._validated_m36_live_preflight_contract(authority)) == (
        authority["live_preflight_contract"]
    )

    final = copy.deepcopy(authority)
    final_chain = final["source_chain"]
    final_chain["initial_control_commit"] = "1" * 40
    final_chain["initial_control_tree"] = "2" * 40
    final_chain["final_reseal_parent"] = "1" * 40
    final_chain["initial_control_blobs"] = {
        path: f"{index + 3:x}" * 40
        for index, path in enumerate(authority["operator_control_paths"])
    }
    fake_materializer = SimpleNamespace(
        _M36_INITIAL_CONTROL_COMMIT="1" * 40,
        _M36_INITIAL_CONTROL_TREE="2" * 40,
        _M36_INITIAL_CONTROL_BLOBS=final_chain["initial_control_blobs"],
    )
    assert dependencies._m36_source_chain_identity_state(
        fake_materializer, final
    ) == "sealed"
    mixed = copy.deepcopy(final)
    first_path = next(iter(mixed["source_chain"]["initial_control_blobs"]))
    mixed["source_chain"]["initial_control_blobs"][first_path] = "0" * 40
    mixed_materializer = SimpleNamespace(
        _M36_INITIAL_CONTROL_COMMIT="1" * 40,
        _M36_INITIAL_CONTROL_TREE="2" * 40,
        _M36_INITIAL_CONTROL_BLOBS=mixed["source_chain"]["initial_control_blobs"],
    )
    with pytest.raises(RuntimeError, match="mix placeholder and sealed"):
        dependencies._m36_source_chain_identity_state(mixed_materializer, mixed)


def test_m36_precedence_history_preappend_resolution_and_no_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m36_presence_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m36_presence_test",
    )
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m36_history_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m36_presence_test",
    )
    scheduler = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json").read_text(encoding="utf-8")
    )
    migration = json.loads(
        (REPO_ROOT / "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json").read_text(encoding="utf-8")
    )
    seal = json.loads(
        (REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json").read_text(encoding="utf-8")
    )
    key = "operator_task_binding_correction_successor_materialization"
    scheduler, historical_migration, historical_seal = (
        _historical_successor_controls_at(key, scheduler, migration, seal)
    )
    assert historical_migration is not None
    assert historical_seal is not None
    migration = historical_migration
    seal = historical_seal
    # M37 advances the same runtime store from generation 29 to 30.  Removing
    # only its authority key is intentionally insufficient to synthesize M36:
    # restore M36's exact sealed runtime generation as well.
    scheduler["database_program"]["store_generation"] = "29"
    selected = operator._active_source_repair_materialization(scheduler)
    assert selected["migration_revision"] == "SAWM-R2-M36"
    selection_source = inspect.getsource(operator._active_source_repair_materialization)
    assert selection_source.index("if m36_key in config:") < selection_source.index("if m35_key in config:")
    assert dependencies._m36_successor_declared({key: None}, {}, {}) is True

    monkeypatch.setattr(board, "_m36_migration_errors", lambda *_a, **_k: ["M36 active"])
    historical_calls: list[bool] = []

    def historical(*_args: object, **kwargs: object) -> list[str]:
        historical_calls.append(kwargs.get("require_active_runtime") is False)
        return []

    for name in (
        "_m35_migration_errors", "_m34_migration_errors", "_m33_migration_errors",
        "_m32_migration_errors", "_m31_migration_errors", "_m30_migration_errors",
        "_m29_migration_errors", "_m28_migration_errors", "_m27_migration_errors",
        "_m26_migration_errors", "_m25_migration_errors", "_m24_migration_errors",
        "_m23_migration_errors", "_m22_migration_errors", "_m21_migration_errors",
        "_m20_migration_errors", "_m19_migration_errors", "_m18_migration_errors",
        "_m17_migration_errors", "_m16_migration_errors",
    ):
        monkeypatch.setattr(board, name, historical)
    assert board._active_successor_migration_errors({key: None}, {}, {}) == [
        "M36 active",
        "M36 operator-task binding authority is only partially declared",
    ]
    assert historical_calls == [True] * 20

    observed: list[str | None] = []

    def source_chain(
        _root: Path,
        _materializer: object,
        _authority: Mapping[str, object],
        *,
        current_head: str | None = None,
    ) -> list[str]:
        observed.append(current_head)
        return []

    monkeypatch.setattr(dependencies, "_m35_source_chain_errors", source_chain)
    assert dependencies._m35_immutable_authority_identity_normalization_successor_errors(
        scheduler, seal, migration, root=REPO_ROOT, require_active_runtime=False
    ) == []
    assert observed == ["5bbf2dec97458585ee95034c986057a1552b805d"]

    authority = materializer._expected_m36_operator_task_binding_correction_authority()
    assert authority["live_owner"]["generation_restart_authorized"] is False
    assert authority["preservation"]["generation_restart"] is False
    assert authority["preservation"]["failed_m35_append_preserved"] is True
    materialize_source = inspect.getsource(materializer._materialize_m36)
    assert materialize_source.index("if os.path.lexists(receipt_path):") < materialize_source.index("_require_m36_operator_task_binding(source)")
    assert materialize_source.index("_require_m36_operator_task_binding(source)") < materialize_source.index("source.record_evidence(")
    assert "record_completion" not in materialize_source
    assert "record_generation_start" not in materialize_source
    marker_source = inspect.getsource(operator._require_m36_source_successor_marker)
    assert "m34-source-successor-receipt.json" in marker_source
    assert "m35-source-successor-receipt.json" in marker_source
    assert "os.path.lexists" in marker_source
    assert "m36-source-successor-receipt.json" in marker_source
    assert "checked is None" in marker_source
    assert "write_text" not in marker_source
    no_restart = inspect.getsource(operator._validate_offline_quack_start)
    assert "M36 requires the exact live generation-29 owner" in no_restart


def test_m35_authority_and_placeholder_or_sealed_chain_are_exact() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m35_authority_test",
    )
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m35_authority_test",
    )
    key = "immutable_authority_identity_normalization_successor_materialization"
    authority = (
        materializer._expected_m35_immutable_authority_identity_normalization_authority()
    )
    reference = materializer._m35_authority_reference()
    scheduler = json.loads((REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json").read_text(encoding="utf-8"))
    migration = json.loads((REPO_ROOT / "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json").read_text(encoding="utf-8"))
    seal = json.loads((REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json").read_text(encoding="utf-8"))
    assert scheduler[key] == reference == migration[key]
    assert seal[f"{key}_cid"] == reference["authority_cid"]
    assert reference == {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M35",
        "authority_cid": materializer._identity(authority),
    }
    assert authority["supersession_mode"] == (
        "append_only_immutable_authority_identity_normalization"
    )
    assert authority["control_recorded_at"] == "2026-08-31T20:20:00Z"
    assert authority["runtime_binding"]["run_id"] == "run-r2-m27"
    assert authority["target_generation"] == 29
    assert authority["target_quack_port"] == 24_070
    assert authority["runtime_binding"]["prior_event_watermark"] == 285
    assert authority["target_event_watermark"] == 286
    assert authority["target_projection_cid"] == (
        "baguqeeravzrhagxizn7o45ukuevzkhreb7dzpabd4g4kmyci2if5rxr32dda"
    )
    prior = authority["prior_authority"]
    assert prior["event_prefix_sha256"] == (
        "79797c1e1593fa880a4ac088796e1ca713dc276df8ad118c12aa0c1ab49f1c7f"
    )
    assert prior["projection_cid"] == (
        "baguqeeragsizyo6v4izu7qfvjbj5l5bjkuycw2xyaf3vhlzx2nai7xyrvd4q"
    )
    assert prior["m34_receipt_sha256"] == (
        "3ec2d6100998f430a388e4c32cbcdda2feb53f070b683b10bfef28ccfa85a872"
    )
    assert prior["m34_receipt_cid"] == (
        "sha256:759b71e0d0fa73a1ac81bb98fb1b96c93f3150b09667cd19fc05330b34b50b01"
    )
    chain = authority["source_chain"]
    assert chain["base_control_commit"] == "4dfe1c4c81ffd65f6a2d5c5cdc38b1cd33f1f443"
    assert chain["base_control_tree"] == "894d9a4d206faf4e59328111023ec44fcf26f96e"
    assert len(authority["operator_control_paths"]) == 9
    assert dependencies._m35_source_chain_identity_state(
        materializer, authority
    ) == "sealed"
    assert chain["initial_control_commit"] == (
        "27c5e1e5925228757fc02378eb9d3b2a8addc60e"
    )
    assert chain["initial_control_tree"] == (
        "75f9c888b22911d47e941fb03b771c8d8f92dd99"
    )
    assert chain["initial_control_blobs"] == {
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json": "21ab73ae195f9514a713d83e8442d29db9c27547",
        "config/semantic_addressed_world_model_dependencies.seal.json": "f30dee923eeeba60b391c1af0982b525055fa441",
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md": "da11d8b4d5f5eed7e5d955b51deded59abc34059",
        "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json": "560f6f039b8934ef6b25d32758db5f7374b5d872",
        "scripts/materialize_semantic_addressed_world_model_program.py": "159db3cdedb34b0d406b8c1d9b7bd8b71eb3b83f",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py": "99129c72bb1ffc63e5624649e64982dfcb46c5f9",
        "scripts/validate_semantic_addressed_world_model_board.py": "41ad96cb9a5cb9ae412f3124c6f4a3e6aeaf19d6",
        "scripts/validate_semantic_addressed_world_model_dependencies.py": "b33b9a61c7f930154d65f7b97cf6b73e0855bc4c",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py": "345f964cdb11e80e9840a0d779aea345c3746fd9",
    }
    assert dict(materializer._validated_m35_live_preflight_contract(authority)) == (
        authority["live_preflight_contract"]
    )

    final = copy.deepcopy(authority)
    final_chain = final["source_chain"]
    final_chain["initial_control_commit"] = "1" * 40
    final_chain["initial_control_tree"] = "2" * 40
    final_chain["final_reseal_parent"] = "1" * 40
    final_chain["initial_control_blobs"] = {
        path: f"{index + 3:x}" * 40
        for index, path in enumerate(authority["operator_control_paths"])
    }
    fake_materializer = SimpleNamespace(
        _M35_INITIAL_CONTROL_COMMIT="1" * 40,
        _M35_INITIAL_CONTROL_TREE="2" * 40,
        _M35_INITIAL_CONTROL_BLOBS=final_chain["initial_control_blobs"],
    )
    assert dependencies._m35_source_chain_identity_state(
        fake_materializer, final
    ) == "sealed"
    mixed = copy.deepcopy(final)
    first_path = next(iter(mixed["source_chain"]["initial_control_blobs"]))
    mixed["source_chain"]["initial_control_blobs"][first_path] = "0" * 40
    mixed_materializer = SimpleNamespace(
        _M35_INITIAL_CONTROL_COMMIT="1" * 40,
        _M35_INITIAL_CONTROL_TREE="2" * 40,
        _M35_INITIAL_CONTROL_BLOBS=mixed["source_chain"]["initial_control_blobs"],
    )
    with pytest.raises(RuntimeError, match="mix placeholder and sealed"):
        dependencies._m35_source_chain_identity_state(mixed_materializer, mixed)


def test_m35_operator_normalizes_immutable_authority_at_receipt_boundary() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m35_boundary_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m35_boundary_test",
    )
    authority = (
        materializer._expected_m35_immutable_authority_identity_normalization_authority()
    )
    population = {
        "program_definition_cid": "sha256:program",
        "source_binding": {"source_binding_cid": "sha256:source"},
    }
    verified = {
        "migration_digest": "sha256:migration",
        "migration_evidence_id": "sha256:evidence",
        "migration_evidence_event_id": 286,
        "target_event_prefix_sha256": "sha256:prefix",
        "semantic_authority_digest": "sha256:semantic",
        "live_server_id": "server:test",
        "live_process_birth_id": "birth:test",
        "live_started_at": "2026-08-31T17:47:26Z",
    }
    immutable = MappingProxyType(authority)
    assert materializer._expected_m35_source_successor_receipt(
        population, immutable, "sha256:validation", verified
    ) == materializer._expected_m35_source_successor_receipt(
        population, dict(immutable), "sha256:validation", verified
    )
    source = inspect.getsource(operator._live_preflight)
    assert "receipt_authority = dict(active_source_repair)" in source
    m35_branch = source[source.index("if m35_active:"):source.index("elif m34_active:")]
    assert "_expected_m35_source_successor_receipt" in m35_branch
    assert "receipt_authority" in m35_branch


def test_m35_presence_history_and_no_restart_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m35_presence_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m35_presence_test",
    )
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m35_history_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m35_presence_test",
    )
    scheduler = json.loads((REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json").read_text(encoding="utf-8"))
    migration = json.loads((REPO_ROOT / "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json").read_text(encoding="utf-8"))
    seal = json.loads((REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json").read_text(encoding="utf-8"))
    key = "immutable_authority_identity_normalization_successor_materialization"
    scheduler, migration, seal = _historical_successor_controls_at(
        key, scheduler, migration, seal
    )
    assert migration is not None and seal is not None
    # M37 is the only successor that advances this store to generation 30.
    # Removing its authority key must also restore M35's sealed generation.
    scheduler["database_program"]["store_generation"] = "29"
    selected = operator._active_source_repair_materialization(scheduler)
    assert selected["migration_revision"] == "SAWM-R2-M35"
    selection_source = inspect.getsource(operator._active_source_repair_materialization)
    assert selection_source.index("if m35_key in config:") < selection_source.index("if m34_key in config:")
    assert dependencies._m35_successor_declared({key: None}, {}, {}) is True

    monkeypatch.setattr(board, "_m35_migration_errors", lambda *_a, **_k: ["M35 active"])
    historical_calls: list[bool] = []
    def historical(*_args: object, **kwargs: object) -> list[str]:
        historical_calls.append(kwargs.get("require_active_runtime") is False)
        return []
    for name in (
        "_m34_migration_errors", "_m33_migration_errors", "_m32_migration_errors",
        "_m31_migration_errors", "_m30_migration_errors", "_m29_migration_errors",
        "_m28_migration_errors", "_m27_migration_errors", "_m26_migration_errors",
        "_m25_migration_errors", "_m24_migration_errors", "_m23_migration_errors",
        "_m22_migration_errors", "_m21_migration_errors", "_m20_migration_errors",
        "_m19_migration_errors", "_m18_migration_errors", "_m17_migration_errors",
        "_m16_migration_errors",
    ):
        monkeypatch.setattr(board, name, historical)
    assert board._active_successor_migration_errors({key: None}, {}, {}) == [
        "M35 active",
        "M35 immutable-authority identity authority is only partially declared",
    ]
    assert historical_calls == [True] * 19

    observed: list[str | None] = []
    def source_chain(
        _root: Path, _materializer: object, _authority: Mapping[str, object],
        *, current_head: str | None = None,
    ) -> list[str]:
        observed.append(current_head)
        return []
    monkeypatch.setattr(dependencies, "_m34_source_chain_errors", source_chain)
    assert dependencies._m34_json_emission_normalization_successor_errors(
        scheduler, seal, migration, root=REPO_ROOT, require_active_runtime=False
    ) == []
    assert observed == ["4dfe1c4c81ffd65f6a2d5c5cdc38b1cd33f1f443"]

    authority = materializer._expected_m35_immutable_authority_identity_normalization_authority()
    assert authority["live_owner"]["generation_restart_authorized"] is False
    assert authority["preservation"]["generation_restart"] is False
    assert authority["preservation"]["m34_receipt_preserved"] is True
    materialize_source = inspect.getsource(materializer._materialize_m35)
    marker_source = inspect.getsource(operator._require_m35_source_successor_marker)
    assert "record_evidence" in materialize_source
    assert "record_completion" not in materialize_source
    assert "record_generation_start" not in materialize_source
    assert "m34-source-successor-receipt.json" in marker_source
    assert "m35-source-successor-receipt.json" in marker_source
    assert "checked is None" in marker_source
    assert "write_text" not in marker_source


def test_m34_authority_and_recursive_json_emission_are_exact(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m34_authority_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m34_json_test",
    )
    key = "json_emission_normalization_successor_materialization"
    authority = materializer._expected_m34_json_emission_normalization_authority()
    reference = materializer._m34_authority_reference()
    scheduler = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json")
        .read_text(encoding="utf-8")
    )
    assert scheduler[key] == reference == migration[key]
    assert seal[f"{key}_cid"] == reference["authority_cid"]
    assert reference == {
        "schema": "sawm/operator-control-authority-reference@1",
        "migration_revision": "SAWM-R2-M34",
        "authority_cid": materializer._identity(authority),
    }
    assert authority["runtime_binding"]["run_id"] == "run-r2-m27"
    assert authority["target_generation"] == 29
    assert authority["target_quack_port"] == 24_070
    assert authority["runtime_binding"]["prior_event_watermark"] == 284
    assert authority["target_event_watermark"] == 285
    assert authority["prior_authority"] == {
        **authority["prior_authority"],
        "migration_revision": "SAWM-R2-M33",
        "event_prefix_sha256": (
            "22687fc6b6f5c082b1c30fcc4a1669d661bc8a67fe39e52e453b56d8eb3ee236"
        ),
        "projection_cid": (
            "baguqeera5wkenkpg5zpndh5whgwrqkvpq2e7qz6xv6rflrajynrpqf7dtmla"
        ),
        "m33_receipt_sha256": (
            "cb5040ce01d739240d0e29ce89f0874f9dd56302a4d0836acf4c076f56f4a682"
        ),
        "m33_receipt_cid": (
            "sha256:ae8270d95f6b5d199a6dc20ba63b6fe5cb7f0fe4f8a30044b03af796a72dcf64"
        ),
    }
    assert authority["target_projection_cid"] == (
        "baguqeeragsizyo6v4izu7qfvjbj5l5bjkuycw2xyaf3vhlzx2nai7xyrvd4q"
    )
    source_chain = authority["source_chain"]
    assert source_chain["base_control_commit"] == (
        "a5aa77fe58cd706ed8a1a2ae9d3f1e652f28b7f9"
    )
    assert source_chain["base_control_tree"] == (
        "8fe38992e84c291a0cab6d2afc6cda5bb4f08ab8"
    )
    assert source_chain["initial_control_commit"] == (
        "e342b63f3f143bb85ed4744e5391c4f8e7c961cd"
    )
    assert source_chain["initial_control_tree"] == (
        "30f0f07fa41947647461ccbd44d6dc4008848259"
    )
    assert source_chain["final_reseal_parent"] == source_chain[
        "initial_control_commit"
    ]
    assert len(authority["operator_control_paths"]) == 9
    assert set(source_chain["initial_control_blobs"]) == set(
        authority["operator_control_paths"]
    )
    assert source_chain["initial_control_blobs"] == {
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json": "71de45675fe8f1d134aa42af3f7fc782fbe9388c",
        "config/semantic_addressed_world_model_dependencies.seal.json": "68dc298dccd48352273bd3d7bf96368a56e4b561",
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md": "b3f998a018f2928df7b80b584e8ea88072f5046a",
        "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json": "73831143dd7d320dfe6a8a63c3331eedd387ad72",
        "scripts/materialize_semantic_addressed_world_model_program.py": "fa41e431afb1b8e26220c9715f8242c06096cc3a",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py": "1a9bc64217aa43fec874dd18754773931d7b8c9d",
        "scripts/validate_semantic_addressed_world_model_board.py": "91f7630233c696c180e7ab237aa5716274a884b2",
        "scripts/validate_semantic_addressed_world_model_dependencies.py": "0b38fd8d3a996139e0eb728e8437f5cbc820b871",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py": "d7c58109e86878017dcb629dc1104c1d5d6af428",
    }
    assert dict(materializer._validated_m34_live_preflight_contract(authority)) == (
        authority["live_preflight_contract"]
    )

    nested = MappingProxyType(
        {
            "valid": True,
            "nested": MappingProxyType(
                {"items": (MappingProxyType({"value": 7}), {"value": 8})}
            ),
        }
    )
    assert operator._emit(nested) == 0
    assert json.loads(capsys.readouterr().out) == {
        "valid": True,
        "nested": {"items": [{"value": 7}, {"value": 8}]},
    }

    secret = "m34-quack-token-value"
    monkeypatch.setenv("SAWM_TEST_QUACK_TOKEN", secret)
    assert operator._emit(
        MappingProxyType(
            {
                "valid": False,
                f"key-{secret}": (MappingProxyType({"value": f"uses-{secret}"}),),
            }
        )
    ) == 2
    rendered = capsys.readouterr().out
    assert secret not in rendered
    assert json.loads(rendered) == {
        "key-<redacted-quack-token>": [
            {"value": "uses-<redacted-quack-token>"}
        ],
        "valid": False,
    }
    for invalid in (
        MappingProxyType({1: "non-string-key"}),
        MappingProxyType({"valid": True, "value": float("nan")}),
        MappingProxyType({"valid": True, "value": object()}),
        MappingProxyType({secret: 1, "<redacted-quack-token>": 2}),
    ):
        with pytest.raises(operator.OperatorError):
            operator._emit(invalid)
        assert capsys.readouterr().out == ""


def test_m34_presence_masks_m33_and_m33_uses_historical_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m34_presence_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m34_presence_test",
    )
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m34_historical_m33_head_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m34_presence_test",
    )
    scheduler = json.loads((REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json").read_text(encoding="utf-8"))
    migration = json.loads((REPO_ROOT / "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json").read_text(encoding="utf-8"))
    seal = json.loads((REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json").read_text(encoding="utf-8"))
    scheduler, migration, seal = _historical_successor_controls_at(
        "json_emission_normalization_successor_materialization",
        scheduler,
        migration,
        seal,
    )
    assert migration is not None and seal is not None
    # M37 is the only successor that advances this store to generation 30.
    # Removing it restores M34's exact live-owner generation.
    scheduler["database_program"]["store_generation"] = "29"

    selected = operator._active_source_repair_materialization(scheduler)
    assert selected["migration_revision"] == "SAWM-R2-M34"
    assert dict(selected) == materializer._expected_m34_json_emission_normalization_authority()
    selection_source = inspect.getsource(operator._active_source_repair_materialization)
    assert selection_source.index("if m34_key in config:") < selection_source.index("if m33_key in config:")
    key = "json_emission_normalization_successor_materialization"
    assert dependencies._m34_successor_declared({key: None}, {}, {}) is True

    monkeypatch.setattr(board, "_m34_migration_errors", lambda *_args, **_kwargs: ["M34 active"])
    historical_calls: list[bool] = []

    def historical(*_args: object, **kwargs: object) -> list[str]:
        historical_calls.append(kwargs.get("require_active_runtime") is False)
        return []

    for name in (
        "_m33_migration_errors", "_m32_migration_errors", "_m31_migration_errors",
        "_m30_migration_errors", "_m29_migration_errors", "_m28_migration_errors",
        "_m27_migration_errors", "_m26_migration_errors", "_m25_migration_errors",
        "_m24_migration_errors", "_m23_migration_errors", "_m22_migration_errors",
        "_m21_migration_errors", "_m20_migration_errors", "_m19_migration_errors",
        "_m18_migration_errors", "_m17_migration_errors", "_m16_migration_errors",
    ):
        monkeypatch.setattr(board, name, historical)
    assert board._active_successor_migration_errors(
        {key: None}, {}, {}
    ) == [
        "M34 active",
        "M34 JSON-emission normalization authority is only partially declared",
    ]
    assert historical_calls == [True] * 18

    observed: list[str | None] = []

    def source_chain(
        _root: Path, _materializer: object, _authority: Mapping[str, object],
        *, current_head: str | None = None,
    ) -> list[str]:
        observed.append(current_head)
        return []

    monkeypatch.setattr(dependencies, "_m33_source_chain_errors", source_chain)
    assert dependencies._m33_live_preflight_contract_successor_errors(
        scheduler, seal, migration, root=REPO_ROOT, require_active_runtime=False
    ) == []
    assert observed == ["a5aa77fe58cd706ed8a1a2ae9d3f1e652f28b7f9"]


def test_m34_is_evidence_only_without_restart_or_synthetic_marker() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m34_append_only_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m34_marker_test",
    )
    authority = materializer._expected_m34_json_emission_normalization_authority()
    assert authority["live_owner"]["generation_restart_authorized"] is False
    assert authority["preservation"]["generation_restart"] is False
    assert authority["preservation"]["same_live_owner"] is True
    assert authority["preservation"]["m33_receipt_preserved"] is True
    assert authority["exact_changes"]["event_suffix_length"] == 1
    assert authority["exact_changes"]["owner_generation_changes"] == 0
    assert authority["exact_changes"]["task_revision_changes"] == 0
    assert authority["exact_changes"]["accepted_completion_changes"] == 0

    materialize_source = inspect.getsource(materializer._materialize_m34)
    marker_source = inspect.getsource(operator._require_m34_source_successor_marker)
    assert "record_evidence" in materialize_source
    assert "record_completion" not in materialize_source
    assert "record_generation_start" not in materialize_source
    assert "m33-source-successor-receipt.json" in marker_source
    assert "m34-source-successor-receipt.json" in marker_source
    assert "checked is None" in marker_source
    assert "write_text" not in marker_source
    for required in (
        "m33_receipt_cid",
        "m33_event_prefix_verified",
        "normalized_live_preflight_contract_verified",
        "json_emission_normalization_verified",
        "recursive_json_emission_normalization_verified",
        "full_event_and_evidence_body_verified",
    ):
        assert required in marker_source


def test_m33_authority_and_normalized_preflight_contract_are_exact() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m33_authority_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m33_contract_test",
    )
    key = "live_preflight_contract_successor_materialization"
    authority = materializer._expected_m33_live_preflight_contract_authority()
    reference = materializer._m33_authority_reference()
    scheduler = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    assert scheduler[key] == reference == migration[key]
    assert seal[f"{key}_cid"] == reference["authority_cid"]
    assert reference["authority_cid"] == materializer._identity(authority)
    assert authority["migration_revision"] == "SAWM-R2-M33"
    assert authority["target_generation"] == 29
    assert authority["runtime_binding"]["prior_event_watermark"] == 283
    assert authority["target_event_watermark"] == 284
    assert authority["source_chain"]["base_control_commit"] == (
        "0de00631fa0fdd2195dc6d532050887b83ce72f7"
    )
    assert authority["source_chain"]["base_control_tree"] == (
        "c624481b51808d6e5f7fc0e801328592b210aeb3"
    )
    assert authority["source_chain"]["initial_control_commit"] == (
        "b0526d4085b231f1eca0cf638743e8d455debc08"
    )
    assert authority["source_chain"]["initial_control_tree"] == (
        "61054b8cdf793a6c9ed4f326215e60b9799a4a6d"
    )
    assert set(authority["source_chain"]["initial_control_blobs"]) == set(
        authority["operator_control_paths"]
    )
    assert len(authority["operator_control_paths"]) == 9
    contract = authority["live_preflight_contract"]
    assert set(contract) == {
        "schema", "migration_revision", "successor_class", "target_store_id",
        "database_uuid", "target_generation", "target_event_watermark",
        "target_plan_revision", "target_projection_cid",
        "semantic_authority_digest", "preserved_plan_anchor",
        "expected_task_heads",
    }
    assert contract["schema"] == "sawm/live-preflight-contract@1"
    assert contract["successor_class"] == "evidence_only_post_m27"
    assert contract["preserved_plan_anchor"] == authority["preserved_plan_anchor"]
    assert contract["expected_task_heads"] == authority["expected_task_heads"]
    assert dict(materializer._validated_m33_live_preflight_contract(authority)) == contract
    assert dict(operator._normalized_live_preflight_contract(authority, materializer)) == contract


@pytest.mark.parametrize(
    ("paths", "invalid"),
    (
        ((("live_preflight_contract", "migration_revision"), ("migration_revision",)),
         "SAWM-R2-M33-drift"),
        ((("live_preflight_contract", "database_uuid"),
          ("runtime_binding", "database_uuid"), ("live_owner", "database_uuid")),
         "database:m33-drift"),
        ((("live_preflight_contract", "target_generation"), ("target_generation",),
          ("runtime_binding", "store_generation"), ("live_owner", "generation")), 30),
        ((("live_preflight_contract", "target_event_watermark"),
          ("target_event_watermark",), ("runtime_binding", "target_event_watermark"),
          ("target_authority", "event_watermark")), 285),
        ((("live_preflight_contract", "target_plan_revision"),
          ("target_plan_revision",), ("runtime_binding", "plan_revision"),
          ("target_authority", "plan_revision")), 29),
        ((("live_preflight_contract", "target_projection_cid"),
          ("target_projection_cid",), ("target_authority", "projection_cid")),
         "baguqeera-m33-drift"),
        ((("live_preflight_contract", "semantic_authority_digest"),
          ("prior_authority", "semantic_authority_digest")), "sha256:" + "0" * 64),
    ),
)
def test_m33_normalized_contract_rejects_coordinated_drift(
    paths: tuple[tuple[str, ...], ...], invalid: object
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m33_contract_drift_test_" + str(len(paths)),
    )
    authority = copy.deepcopy(materializer._expected_m33_live_preflight_contract_authority())
    for path in paths:
        cursor = authority
        for component in path[:-1]:
            cursor = cursor[component]
        cursor[path[-1]] = copy.deepcopy(invalid)
    with pytest.raises(
        materializer.MaterializationError,
        match="M33 normalized live-preflight contract differs",
    ):
        materializer._validated_m33_live_preflight_contract(authority)


def test_m33_presence_masks_m32_and_m32_uses_historical_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m33_presence_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m33_presence_test",
    )
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m33_historical_m32_head_test",
    )
    scheduler = json.loads((REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json").read_text(encoding="utf-8"))
    migration = json.loads((REPO_ROOT / "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json").read_text(encoding="utf-8"))
    seal = json.loads((REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json").read_text(encoding="utf-8"))
    scheduler, migration, seal = _historical_successor_controls_at(
        "live_preflight_contract_successor_materialization",
        scheduler,
        migration,
        seal,
    )
    assert migration is not None and seal is not None
    # M37 is the only successor that advances this store to generation 30.
    # Removing it restores M33's exact live-owner generation.
    scheduler["database_program"]["store_generation"] = "29"
    selected = operator._active_source_repair_materialization(scheduler)
    assert selected["migration_revision"] == "SAWM-R2-M33"
    assert dict(selected) == materializer._expected_m33_live_preflight_contract_authority()
    selection_source = inspect.getsource(operator._active_source_repair_materialization)
    assert selection_source.index("if m33_key in config:") < selection_source.index("if m32_key in config:")
    observed: list[str | None] = []

    def source_chain(
        _root: Path, _materializer: object, _authority: Mapping[str, object],
        *, current_head: str | None = None,
    ) -> list[str]:
        observed.append(current_head)
        return []

    monkeypatch.setattr(dependencies, "_m32_source_chain_errors", source_chain)
    assert dependencies._m32_live_preflight_plan_anchor_successor_errors(
        scheduler, seal, migration, root=REPO_ROOT, require_active_runtime=False
    ) == []
    assert observed == [authority_head := materializer._expected_m33_live_preflight_contract_authority()["source_chain"]["base_control_commit"]]
    assert authority_head == "0de00631fa0fdd2195dc6d532050887b83ce72f7"


def test_m33_preflight_and_marker_are_contract_driven_and_append_only() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m33_contract_route_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m33_contract_route_test",
    )
    live_source = inspect.getsource(operator._live_preflight)
    recovery_source = inspect.getsource(operator._recover_stale_quack)
    offline_source = inspect.getsource(operator._validate_offline_quack_start)
    marker_source = inspect.getsource(operator._require_m33_source_successor_marker)
    materialize_source = inspect.getsource(materializer._materialize_m33)
    assert "_normalized_live_preflight_contract" in live_source
    assert 'preflight_contract["database_uuid"]' in live_source
    assert 'active_source_repair["prior_database_uuid"]' not in live_source
    assert "_normalized_live_preflight_contract" in recovery_source
    assert "M33 requires the exact live generation-29 owner" in offline_source
    assert "_require_m32_source_successor_marker" in marker_source
    assert "m32-source-successor-receipt.json" in marker_source
    assert "m33_source_successor_receipt_cid" in marker_source
    assert "_M33_PRIOR_EVENT_WATERMARK" in materialize_source
    assert "_M33_TARGET_EVENT_WATERMARK" in materialize_source
    assert "record_evidence" in materialize_source
    assert "record_completion" not in materialize_source


def test_m32_authority_and_preserved_plan_anchor_are_exact() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m32_authority_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m32_plan_anchor_test",
    )
    key = "live_preflight_plan_anchor_successor_materialization"
    authority = materializer._expected_m32_live_preflight_plan_anchor_authority()
    reference = materializer._m32_authority_reference()
    scheduler = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json")
        .read_text(encoding="utf-8")
    )
    assert scheduler[key] == reference == migration[key]
    assert reference["authority_cid"] == materializer._identity(authority)
    assert seal[f"{key}_cid"] == reference["authority_cid"]
    assert authority["migration_revision"] == "SAWM-R2-M32"
    assert authority["target_generation"] == 29
    assert authority["prior_authority"]["event_watermark"] == 282
    assert authority["target_event_watermark"] == 283
    assert authority["target_projection_cid"] == (
        "baguqeerab3m6k3ea4ulaouojsdazccvepipcfryyblps7676ymqrtbyc5tiq"
    )
    assert authority["source_chain"]["initial_control_commit"] == (
        "547485ffacd636c6046b00ed23dc0f4c53de4315"
    )
    assert authority["source_chain"]["initial_control_tree"] == (
        "10e153a78f5709e8b841567ab4475f60eff05db3"
    )
    assert authority["expected_task_heads"] == materializer._m30_expected_task_heads()
    anchor = operator._preserved_m27_plan_anchor(
        scheduler, authority, materializer
    )
    assert dict(anchor) == {
        "plan_source_binding_cid": (
            "sha256:83e28e01de41699d5b2312ead03e7f33d9989d809d97924ea0a230af2c038856"
        ),
        "plan_migration_digest": (
            "sha256:43eb5eb9b3f05c6ffe00c918901524e61c4a94a8d7334a3a3261ef95870cc73b"
        ),
    }
    live_source = inspect.getsource(operator._live_preflight)
    assert "_preserved_m27_plan_anchor" in live_source
    assert 'active_source_repair["prior_authority"]' not in live_source


def test_m32_preserved_plan_anchor_fails_closed_on_historical_m29_drift() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m32_anchor_drift_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m32_anchor_drift_test",
    )
    config = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    key = "committed_evidence_verification_successor_materialization"
    config[key]["prior_authority"]["plan_source_binding_cid"] = "sha256:" + "0" * 64
    with pytest.raises(operator.OperatorError, match="preserved M29 plan authority differs"):
        operator._preserved_m27_plan_anchor(
            config,
            materializer._expected_m32_live_preflight_plan_anchor_authority(),
            materializer,
        )


def test_m32_validates_m31_source_chain_at_its_preserved_historical_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m32_historical_m31_head_test",
    )
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m32_historical_m31_head_test",
    )
    scheduler = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json")
        .read_text(encoding="utf-8")
    )
    observed: list[str | None] = []

    def source_chain(
        _root: Path,
        _materializer: object,
        _authority: Mapping[str, object],
        *,
        current_head: str | None = None,
    ) -> list[str]:
        observed.append(current_head)
        return []

    monkeypatch.setattr(dependencies, "_m31_source_chain_errors", source_chain)
    assert dependencies._m31_detached_coordinator_pid_recovery_successor_errors(
        scheduler,
        seal,
        migration,
        root=REPO_ROOT,
        require_active_runtime=False,
    ) == []
    assert observed == [
        materializer._expected_m32_live_preflight_plan_anchor_authority()[
            "source_chain"
        ]["base_control_commit"]
    ]


def test_m31_authority_pins_dead_pid_event_281_and_generation_29() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m31_authority_test",
    )
    key = "detached_coordinator_pid_recovery_successor_materialization"
    authority = (
        materializer._expected_m31_detached_coordinator_pid_recovery_authority()
    )
    scheduler = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json")
        .read_text(encoding="utf-8")
    )
    assert scheduler[key] == authority == migration[key]
    assert seal[f"{key}_cid"] == materializer._identity(authority)
    # M31 remains immutable history while M48 is the current generation-35
    # restart authority.
    assert scheduler["database_program"]["store_generation"] == "35"
    assert authority["migration_revision"] == "SAWM-R2-M31"
    assert authority["prior_authority"]["event_watermark"] == 281
    assert authority["target_event_watermark"] == 282
    assert authority["target_generation"] == 29
    assert authority["target_projection_cid"] == (
        "baguqeerakjradc5sa5dmflygtfh2birrd5onygnt6q2pkvoomsxrspi22jaa"
    )
    assert authority["stopped_owner"]["generation"] == 28
    assert authority["stopped_owner"]["stopped_at"] == "2026-08-31T16:38:00Z"
    stale = authority["failed_launch_evidence"]["stale_pid_projection"]
    assert stale == {
        "path": (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27/"
            "state/configured-board-master.pid"
        ),
        "pid": 3_554_888,
        "content_sha256": (
            "592c926b10dfc688ab07af087bb761228c61a1f8f2829c7f577464322eacca46"
        ),
        "mode": 0o664,
        "uid": 1000,
        "gid": 1000,
        "link_count": 1,
        "inode": 97_255_434,
        "pid_observed_dead": True,
        "completion_authority": False,
    }
    assert authority["accepted_source_repair"]["changed_paths"] == sorted(
        authority["accepted_source_repair"]["blob_oids"]
    )
    chain = authority["source_chain"]
    assert chain["initial_control_commit"] == (
        "07aed87e3ebc4ef5667541435fd04f2d62a39b25"
    )
    assert chain["initial_control_tree"] == (
        "c9fe9a657d2d0e1e05172688dbc44dea6f699265"
    )
    assert chain["final_reseal_parent"] == chain["initial_control_commit"]
    assert set(chain["initial_control_blobs"]) == set(
        authority["operator_control_paths"]
    )
    body = materializer._m31_migration_body(
        materializer.build_population(REPO_ROOT),
        scheduler,
        "sha256:" + "1" * 64,
    )
    assert body["exact_changes"] == authority["exact_changes"]
    assert body["preservation"] == authority["preservation"]


def test_m31_presence_masks_m30_and_keeps_history_nonactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m31_presence_test",
    )
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m31_presence_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m31_presence_test",
    )
    key = "detached_coordinator_pid_recovery_successor_materialization"
    authority = (
        materializer._expected_m31_detached_coordinator_pid_recovery_authority()
    )
    scheduler = {key: authority}
    migration = {key: authority}
    seal = {f"{key}_cid": materializer._identity(authority)}
    assert dependency._m31_successor_declared(scheduler, {}, {}) is True
    assert dependency._m31_successor_declared({}, seal, {}) is True
    assert dependency._m31_successor_declared({}, {}, migration) is True
    monkeypatch.setattr(
        board, "_m31_migration_errors", lambda *_args, **_kwargs: ["M31 active"]
    )
    historical_calls: list[bool] = []

    def historical(*_args: object, **kwargs: object) -> list[str]:
        historical_calls.append(kwargs.get("require_active_runtime") is False)
        return []

    for name in (
        "_m30_migration_errors", "_m29_migration_errors", "_m28_migration_errors",
        "_m27_migration_errors", "_m26_migration_errors", "_m25_migration_errors",
        "_m24_migration_errors", "_m23_migration_errors", "_m22_migration_errors",
        "_m21_migration_errors", "_m20_migration_errors", "_m19_migration_errors",
        "_m18_migration_errors", "_m17_migration_errors", "_m16_migration_errors",
    ):
        monkeypatch.setattr(board, name, historical)
    assert board._active_successor_migration_errors(
        scheduler, seal, migration
    ) == ["M31 active"]
    assert historical_calls == [True] * 15
    partial_errors = board._active_successor_migration_errors(scheduler, {}, {})
    assert "M31 active" in partial_errors
    assert any("partially declared" in error for error in partial_errors)


def test_m31_pid_and_receipt_controls_are_stable_and_serializable() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m31_stable_pid_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m31_reservation_test",
    )
    prestart_source = inspect.getsource(materializer._check_m31_prestart_admission)
    ensure_source = inspect.getsource(materializer._ensure_m31_source_successor_receipt)
    live_source = inspect.getsource(operator._live_preflight)
    main_source = inspect.getsource(operator.main)
    assert "_read_stable_regular_bytes" in prestart_source
    assert "with serialized_lock_update(pid_path):" in prestart_source
    assert "pid_path.resolve().read_bytes" not in prestart_source
    assert 'identity.get("stopped_at")' not in prestart_source
    assert "generation,started_at,stopped_at,status,revision FROM state_servers" in (
        prestart_source
    )
    assert "return dict(observed)" in ensure_source
    assert "return dict(expected)" in ensure_source
    assert live_source.index("before_token_handoff_retirement()") < live_source.index(
        "retire_token_handoff("
    )
    assert "_reserve_detached_coordinator_pid" in main_source
    assert "coordinator_pid_reservation=(" in main_source
    assert 'coordinator_pid_reservation.state == "reserved"' in main_source


def test_m30_marker_uses_checked_prefix_and_m31_composes_receipt_chain() -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m30_m31_chain_test",
    )
    m30_source = inspect.getsource(operator._require_m30_source_successor_marker)
    m31_source = inspect.getsource(operator._require_m31_source_successor_marker)
    assert 'checked.get("prior_event_prefix_verified") is not True' in m30_source
    assert 'observed.get("prior_event_prefix_sha256")' not in m30_source
    assert 'observed.get("prior_event_prefix_verified")' not in m30_source
    assert "_require_m29_source_successor_marker" in m30_source
    assert "_require_m30_source_successor_marker" in m31_source
    assert '"m27_final_pair_receipt_cid"' in m31_source
    assert '"m29_source_successor_receipt_cid"' in m31_source
    assert '"m30_source_successor_receipt_cid"' in m31_source
    assert '"m31_source_successor_receipt_cid"' in m31_source


def test_m30_authority_pins_stopped_event_280_and_generation_28() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m30_authority_test",
    )
    key = "stopped_owner_restart_source_seal_successor_materialization"
    authority = (
        materializer._expected_m30_stopped_owner_restart_source_seal_authority()
    )
    scheduler = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json")
        .read_text(encoding="utf-8")
    )
    assert scheduler[key] == authority == migration[key]
    assert seal[f"{key}_cid"] == materializer._identity(authority)
    # M30 remains immutable history while the current M48 owner is generation 35.
    assert scheduler["database_program"]["store_generation"] == "35"
    assert authority["schema"].endswith("authorization@2")
    assert authority["authorization_revision"] == 2
    assert authority["control_recorded_at"] == "2026-08-31T15:59:48Z"
    assert authority["authorization_amended_at"] == "2026-08-31T15:59:48Z"
    assert authority["prior_authorization_cid"] == (
        "sha256:ef37e79ce07cbb18d16259feeb85c3176661ebc64ec3cb108c3d7b5624e294fe"
    )
    assert authority["prior_authority"]["event_watermark"] == 280
    assert authority["prior_authority"]["event_prefix_sha256"] == (
        "633332aa54c610819c55ba0bcd9c4fa4b745bb5c4414e063a009a084df73d909"
    )
    assert authority["prior_authority"]["projection_cid"] == (
        "baguqeeragk42f56z3bz6ofrjcfipippglefbblgnj7qtpl4zrg735w3ytzka"
    )
    assert authority["prior_authority"]["semantic_authority_digest"] == (
        "sha256:395168339f24de03f6d6f91cc0ca0365df2ffa8d71a80ac9fecab6e282817163"
    )
    assert authority["prior_authority"]["program_definition_cid"] == (
        "sha256:f581af1f2234c127231b47bb1bb8d42910b984dcd1bece9a6303949ac1ba0b72"
    )
    assert authority["prior_authority"]["plan_root_cid"] == (
        "sha256:d9481937430405ff6a512e779b14b7ce676de45d277c65d3763ebe49445ba914"
    )
    assert authority["prior_authority"]["source_binding_cid"] == (
        "sha256:60c69c7e554a3d037a441f8a9232b8e9eecbdc13fd53dc55cb79e9942738256e"
    )
    assert authority["prior_authority"]["control_store_sha256"] == (
        "4e1c0cc5bd0aba06c4cc1267290baf22aafa10a25d9fc76ac14ea3c00a00d465"
    )
    assert authority["prior_authority"]["control_store_size"] == 43_528_192
    assert authority["prior_authority"]["coordination_store_sha256"] == (
        "0a93e9993e66b0671bc155a425f0536bb41a71affe2993d8128ba685bcba65cd"
    )
    assert authority["prior_authority"]["coordination_store_size"] == 17_051_648
    assert authority["prior_authority"]["stopped_status_sha256"] == (
        "bf44e8b996fc16bb2aed34f908e325301f9e4e32bd3c022052bb7c291683db36"
    )
    assert authority["prior_authority"]["m29_receipt_sha256"] == (
        "6159a3e51d3850015692c4c2d1bbd7135ceab8ac500f185287b2f22616772f05"
    )
    assert authority["target_generation"] == 28
    assert authority["target_event_watermark"] == 281
    assert authority["target_projection_cid"] == (
        "baguqeeragvf7yhfecuccjs3fqqg7azmwukg63sbgkbivnczfv6uljqd4js4q"
    )
    assert authority["stopped_owner"]["stopped_at"] == "2026-08-31T04:54:18Z"
    assert authority["expected_task_heads"]["SAWM-006"] == {
        "status": "in_progress", "revision": 7,
    }
    assert authority["expected_task_heads"]["SAWM-008"] == {
        "status": "in_progress", "revision": 9,
    }
    assert authority["expected_task_heads"]["SAWM-015"] == {
        "status": "completed", "revision": 7,
    }
    repair = authority["accepted_source_repair"]
    assert repair["repair_commit"] == (
        "b1c0226a5e95c36656b300034cd9f78b8a70201e"
    )
    assert repair["repair_parent"] == (
        "64acbcc2afb7f51f33eb3ba398f6013caa54f6de"
    )
    assert repair["repair_tree"] == (
        "69cdfa11e4ec5ffee6040cef582b7c006865ec5b"
    )
    assert repair["changed_paths"] == sorted(repair["blob_oids"])
    chain = authority["source_chain"]
    assert chain["initial_control_commit"] == (
        "8233b47ba4c05470235ec832e95a70fdce13316d"
    )
    assert chain["initial_control_tree"] == (
        "e7b322fa25b22369f17748417cea100a624fb000"
    )
    assert chain["initial_authorization_cid"] == (
        "sha256:ef37e79ce07cbb18d16259feeb85c3176661ebc64ec3cb108c3d7b5624e294fe"
    )
    assert chain["cursor_normalization_repair_commit"] == (
        "95505a7eec81a5eedd859e7efb97539d759c918f"
    )
    assert chain["cursor_normalization_repair_parent"] == (
        "8233b47ba4c05470235ec832e95a70fdce13316d"
    )
    assert chain["cursor_normalization_repair_tree"] == (
        "1c1d7a9920bbbdb16bb08c05c2cb7fd1c11c0ffa"
    )
    assert chain["final_control_commit_count"] == 2
    assert chain["intermediary_repair_commit_count"] == 1
    assert chain["final_reseal_commit_count"] == 1
    cursor_repair = authority["bounded_materializer_repair"]
    assert cursor_repair["normalization"] == "integer_index_to_tuple"
    assert cursor_repair["row_widths"] == {
        "state_servers": 14,
        "store_generations": 9,
        "credentials": 8,
        "server_epochs": 5,
        "capability_snapshots": 9,
    }
    assert cursor_repair["event_append_changes"] == 0
    assert cursor_repair["lifecycle_checks_weakened"] is False
    prior_control = authority["prior_control_authorization"]
    assert prior_control == {
        "authorization_cid": authority["prior_authorization_cid"],
        "control_commit": chain["initial_control_commit"],
        "control_tree": chain["initial_control_tree"],
        "control_recorded_at": "2026-08-31T05:05:00Z",
        "superseded_before_event_281": True,
        "event_281_appended": False,
        "receipt_published": False,
    }
    assert authority["authorization_amendment_paths"] == sorted(
        materializer._M30_AMENDMENT_PATHS
    )
    assert set(authority["bounded_control_plane_repair_paths"]) == (
        set(authority["operator_control_paths"])
        | set(repair["changed_paths"])
    )


def test_m30_presence_masks_m29_and_keeps_history_nonactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m30_presence_test",
    )
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m30_presence_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m30_presence_test",
    )
    key = "stopped_owner_restart_source_seal_successor_materialization"
    authority = (
        materializer._expected_m30_stopped_owner_restart_source_seal_authority()
    )
    scheduler = {key: authority}
    migration = {key: authority}
    seal = {f"{key}_cid": materializer._identity(authority)}
    assert dependency._m30_successor_declared(scheduler, {}, {}) is True
    assert dependency._m30_successor_declared({}, seal, {}) is True
    assert dependency._m30_successor_declared({}, {}, migration) is True
    monkeypatch.setattr(
        board, "_m30_migration_errors", lambda *_args, **_kwargs: ["M30 active"]
    )
    historical_calls: list[bool] = []

    def historical(*_args: object, **kwargs: object) -> list[str]:
        historical_calls.append(kwargs.get("require_active_runtime") is False)
        return []

    for name in (
        "_m29_migration_errors", "_m28_migration_errors", "_m27_migration_errors",
        "_m26_migration_errors", "_m25_migration_errors", "_m24_migration_errors",
        "_m23_migration_errors", "_m22_migration_errors", "_m21_migration_errors",
        "_m20_migration_errors", "_m19_migration_errors", "_m18_migration_errors",
        "_m17_migration_errors", "_m16_migration_errors",
    ):
        monkeypatch.setattr(board, name, historical)
    assert board._active_successor_migration_errors(
        scheduler, seal, migration
    ) == ["M30 active"]
    assert historical_calls == [True] * 14
    partial_errors = board._active_successor_migration_errors(scheduler, {}, {})
    assert "M30 active" in partial_errors
    assert any("partially declared" in error for error in partial_errors)


def test_m30_quack_start_uses_prestart_admission_not_post_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m30_prestart_test",
    )
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m30_prestart_test",
    )
    config = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    key = "stopped_owner_restart_source_seal_successor_materialization"
    config, _historical_migration, _historical_seal = (
        _historical_successor_controls_at(key, config)
    )
    config["database_program"]["store_generation"] = "28"
    authority = (
        materializer._expected_m30_stopped_owner_restart_source_seal_authority()
    )
    fake = SimpleNamespace(
        build_population=lambda _root: {},
        _assert_committed_clean_source=lambda _root, _population: None,
        _expected_m30_stopped_owner_restart_source_seal_authority=lambda: authority,
        _check_m30_prestart_admission=lambda _root, _config: {
            "valid": True,
            "prior_generation": 27,
            "target_generation": 28,
            "prior_event_watermark": 280,
        },
        check_materialized=lambda *_args: pytest.fail(
            "M30 prestart must not require its future live receipt"
        ),
    )
    monkeypatch.setattr(operator, "_materializer", lambda: fake)
    monkeypatch.setattr(operator, "_validator", lambda *_args: {"valid": True})
    report = operator._validate_offline_quack_start(config)
    assert report["store"]["target_generation"] == 28
    assert report["prior_authority"] == authority


def test_m30_prestart_rejects_stale_stop_and_token_handoff_controls() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m30_prestart_artifacts_test",
    )
    source = inspect.getsource(materializer._check_m30_prestart_admission)
    assert 'status_path.with_name("quack-state-server.stop")' in source
    assert '"env___SAWM_QUACK_TOKEN.quack-token"' in source
    assert "os.path.lexists(stop_path)" in source
    assert "os.path.lexists(token_handoff_path)" in source


def test_m30_live_verifier_rechecks_event_280_prefix() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m30_prefix_test",
    )
    source = inspect.getsource(materializer._verify_m30_live_materialization)
    assert "prior_prefix = _event_prefix_digest" in source
    assert "_M30_PRIOR_EVENT_WATERMARK" in source
    assert "_M30_PRIOR_EVENT_PREFIX_SHA256" in source
    assert '"prior_event_prefix_verified": True' in source

    projection_source = inspect.getsource(materializer._inspect_m30_live_projection)
    assert "snapshot.dependency_count != 136" in projection_source
    assert "snapshot.plan_count != 1" in projection_source
    assert 'snapshot.plan_root_cid != str(population["plan_root_cid"])' in (
        projection_source
    )


def test_m30_restart_verifier_pins_full_lifecycle_rows() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m30_lifecycle_rows_test",
    )
    source = inspect.getsource(materializer._inspect_m30_generation_restart_rows)
    for required in (
        "schema_revision,fence_epoch,revision",
        "rotated_at,revoked_at,revision",
        "server_id,epoch,fence_epoch,started_at,ended_at",
        "snapshot_id,server_id,profile_id,duckdb_version,extension_name",
        "identity.get(\"extension_fingerprint\") != _M30_EXTENSION_FINGERPRINT",
        'int(identity.get("revision") or 0) != 0',
        'int(identity.get("credential_generation") or 0) != 28',
    ):
        assert required in source


def test_m30_restart_verifier_accepts_positional_duckdb_rows() -> None:
    """The live Quack facade returns Mapping rows, not plain tuples."""

    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m30_duckdb_row_test",
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBRow,
    )

    authority = (
        materializer._expected_m30_stopped_owner_restart_source_seal_authority()
    )
    stopped = authority["stopped_owner"]
    server_id = "server:m30-positional-row-test"
    birth_id = "birth:m30-positional-row-test"
    started_at = "2026-08-31T16:00:00Z"
    startup_epoch = 1_788_192_000
    fingerprint = materializer._M30_EXTENSION_FINGERPRINT
    identity = {
        "server_id": server_id,
        "store_id": materializer._M30_STORE_ID,
        "database_uuid": materializer._M30_DATABASE_UUID,
        "process_birth_id": birth_id,
        "listen_uri": "quack:127.0.0.1:24070",
        "extension_fingerprint": fingerprint,
        "schema_revision": 1,
        "generation": 28,
        "fence_epoch": 28,
        "started_at": started_at,
        "status": "ready",
        "revision": 0,
        "credential_generation": 28,
        "startup_epoch": startup_epoch,
    }
    capability_body = json.dumps(
        {
            "status": "compatible",
            "profile_id": "agent-supervisor-duckdb-quack-1.5",
            "extension_fingerprint": fingerprint,
        },
        sort_keys=True,
    )
    rows = {
        "state_servers": [
            (
                stopped["server_id"], stopped["store_id"],
                stopped["database_uuid"], stopped["process_birth_id"],
                stopped["listen_uri"], fingerprint, 1, 27,
                stopped["started_at"], stopped["stopped_at"], "stopped", 2,
                "", "{}",
            ),
            (
                server_id, materializer._M30_STORE_ID,
                materializer._M30_DATABASE_UUID, birth_id,
                "quack:127.0.0.1:24070", fingerprint, 1, 28, started_at,
                None, "ready", 1, "", "{}",
            ),
        ],
        "store_generations": [
            (
                27, 1, 27, 0, materializer._M30_DATABASE_UUID,
                materializer._M30_PRIOR_PROCESS_BIRTH_ID,
                materializer._M30_PRIOR_STARTED_AT, "", "{}",
            ),
            (
                28, 1, 28, 0, materializer._M30_DATABASE_UUID, birth_id,
                started_at, "", "{}",
            ),
        ],
        "credentials": [
            (
                f"cred:{materializer._M30_PRIOR_SERVER_ID}:27",
                "env://SAWM_QUACK_TOKEN", 27, "quack-auth",
                materializer._M30_PRIOR_STARTED_AT, None, None, 0,
            ),
            (
                f"cred:{server_id}:28", "env://SAWM_QUACK_TOKEN", 28,
                "quack-auth", started_at, None, None, 0,
            ),
        ],
        "server_epochs": [(server_id, startup_epoch, 28, started_at, None)],
        "capability_snapshots": [
            (
                f"cap:{server_id}:28", server_id,
                "agent-supervisor-duckdb-quack-1.5", "1.5.5", "quack",
                fingerprint, "compatible", started_at, capability_body,
            )
        ],
    }

    class Cursor:
        def __init__(self, values: list[tuple[object, ...]]) -> None:
            self.values = values

        def fetchall(self) -> list[DuckDBRow]:
            return [
                DuckDBRow(
                    (f"column_{index}" for index in range(len(value))), value
                )
                for value in self.values
            ]

        def fetchone(self) -> DuckDBRow | None:
            values = self.fetchall()
            return values[0] if values else None

    class Connection:
        def execute(
            self, statement: str, parameters: object = None
        ) -> Cursor:
            del parameters
            normalized = " ".join(statement.split())
            if normalized.startswith("SELECT COUNT(*) FROM "):
                return Cursor([(28,)])
            for table, values in rows.items():
                if f" FROM {table} " in f" {normalized} ":
                    return Cursor(values)
            raise AssertionError(f"unexpected statement: {normalized}")

    @contextlib.contextmanager
    def connection(*, write: bool = False) -> object:
        assert write is False
        yield Connection()

    source = SimpleNamespace(
        intent=SimpleNamespace(_connection=connection)
    )
    verified = materializer._inspect_m30_generation_restart_rows(
        source, identity, authority
    )
    assert verified == {
        "generation_27_28_restart_rows_verified": True,
        "prior_owner_generation": 27,
        "live_owner_generation": 28,
        "live_server_id": server_id,
        "live_process_birth_id": birth_id,
        "live_started_at": started_at,
    }
    poisoned = list(rows["capability_snapshots"][0])
    poisoned[6] = "incompatible"
    rows["capability_snapshots"][0] = tuple(poisoned)
    with pytest.raises(
        materializer.MigrationRequired,
        match="M30 exact generation-27/28 restart rows differ",
    ):
        materializer._inspect_m30_generation_restart_rows(
            source, identity, authority
        )


def test_m29_provider_environment_guard_rejects_quack_token_substrings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m29_provider_token_guard_test",
    )
    source = inspect.getsource(operator._live_preflight)
    assert "discovery.token in str(value)" in source
    assert "for value in provider_environment.values()" in source
    assert "discovery.token in provider_environment.values()" not in source

    token = "sealed-quack-owner-token"
    provider_environment = {
        "PATH": "/usr/bin",
        "UNRELATED": f"prefix-{token}-suffix",
    }
    assert token not in provider_environment.values()
    assert any(token in str(value) for value in provider_environment.values())

    target_store = "run/control.duckdb"
    endpoint = "quack:127.0.0.1:24070"
    database_uuid = "database:m29-provider-token-test"
    live_identity = {
        "server_id": "server:m29-provider-token-test",
        "store_id": target_store,
        "database_uuid": database_uuid,
        "process_birth_id": "birth:m29-provider-token-test",
        "listen_uri": endpoint,
        "extension_fingerprint": "sha256:" + "1" * 64,
        "schema_revision": 1,
        "schema_fingerprint": "sha256:" + "2" * 64,
        "generation": 27,
        "credential_generation": 27,
        "secret_handle": "env://SAWM_QUACK_TOKEN",
    }
    state_dir = tmp_path / "quack-owner"
    state_dir.mkdir()
    status_path = state_dir / "status.json"
    status_path.write_text(
        json.dumps({"identity": live_identity}),
        encoding="utf-8",
    )
    discovery = SimpleNamespace(
        uri=endpoint,
        token=token,
        status_path=str(status_path),
        source="status_file",
        reason="ready",
    )
    population = {
        "program_definition_cid": "sha256:" + "3" * 64,
        "repository_tree_id": "tree:m29-provider-token-test",
        "plan_root_cid": "plan:m29-provider-token-test",
        "source_binding": {"source_binding_cid": "source:m29-provider-token-test"},
        "objectives": [],
    }
    authority = {
        "migration_revision": "SAWM-R2-M29-test",
        "target_generation": 27,
        "target_event_watermark": 274,
        "target_plan_revision": 28,
        "target_store_id": target_store,
        "prior_database_uuid": database_uuid,
        "prior_semantic_authority_digest": "semantic:m29-provider-token-test",
    }
    config = {
        "database_program": {
            "store_id": target_store,
            "store_generation": "27",
            "quack_endpoint": endpoint,
        },
        "quack_owner": {"state_dir": "quack-owner"},
        "provider": {
            "primary_model_id": "grok-test",
            "fallback_model_id": "codex-test",
            "fallback_reasoning_effort": "medium",
            "fallback_trigger": "primary_quota_exhausted",
        },
    }

    class Materializer:
        class MigrationRequired(RuntimeError):
            pass

        class MaterializationError(RuntimeError):
            pass

        @staticmethod
        def build_population(_root: Path) -> dict[str, object]:
            return population

        @staticmethod
        def _identity(_value: object) -> str:
            return "sha256:" + "4" * 64

        @staticmethod
        def _verify_m6_task_projection(
            _live: object, _population: object
        ) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
            return {"SAWM-000": "completed"}, {}, {}

        @staticmethod
        def _semantic_authority_digest_on(_connection: object) -> str:
            return "semantic:m29-provider-token-test"

    class Live:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.intent = SimpleNamespace(
                _connection=lambda **_kwargs: contextlib.nullcontext(object())
            )
            self.plans = {
                population["plan_root_cid"]: {
                    "plan_cid": population["plan_root_cid"],
                    "revision": 28,
                    "body": {
                        "current_source_binding_cid": population[
                            "source_binding"
                        ]["source_binding_cid"],
                        "source_migration_revision": authority[
                            "migration_revision"
                        ],
                    },
                }
            }

        @staticmethod
        def snapshot() -> SimpleNamespace:
            return SimpleNamespace(
                to_dict=lambda: {
                    "task_count": 45,
                    "goal_count": 29,
                    "dependency_count": 136,
                    "plan_count": 1,
                    "event_cursor": 274,
                    "projection_cid": "projection:m29-provider-token-test",
                    "plan_root_cid": population["plan_root_cid"],
                }
            )

        @staticmethod
        def close() -> None:
            return None

    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
        quack_state_server,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        database_task_source,
        duckdb_state,
    )
    from ipfs_accelerate_py import llm_router

    monkeypatch.setattr(operator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(operator, "_validator", lambda *_args: {"valid": True})
    monkeypatch.setattr(operator, "_materializer", lambda: Materializer())
    monkeypatch.setattr(
        operator,
        "_active_source_repair_materialization",
        lambda _config: authority,
    )
    monkeypatch.setattr(
        operator,
        "_require_active_final_pair_marker",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        operator,
        "_expected_live_projection_cid",
        lambda *_args: "projection:m29-provider-token-test",
    )
    monkeypatch.setattr(
        operator,
        "_remote_owner_identity",
        lambda *_args: {
            **live_identity,
            "canonical_rows_verified": True,
            "live": True,
        },
    )
    monkeypatch.setattr(
        duckdb_state,
        "discover_live_quack_endpoint",
        lambda _store: discovery,
    )
    monkeypatch.setattr(database_task_source, "DatabaseTaskSource", Live)
    monkeypatch.setattr(
        quack_state_server,
        "retire_token_handoff",
        lambda **_kwargs: {"retired": True},
    )
    monkeypatch.setattr(
        multi_supervisor_runner,
        "DatabaseProgramConfig",
        SimpleNamespace(from_mapping=lambda _mapping: object()),
    )
    monkeypatch.setattr(
        multi_supervisor_runner,
        "provider_subprocess_environment",
        lambda *_args, **_kwargs: provider_environment,
    )

    provider_probe_called = False

    def provider_probe(**_kwargs: object) -> object:
        nonlocal provider_probe_called
        provider_probe_called = True
        raise AssertionError("provider probe received an owner credential")

    monkeypatch.setattr(
        llm_router,
        "probe_grok_codex_agent_route_readiness",
        provider_probe,
    )
    for name in (
        duckdb_state.QUACK_ENDPOINT_ENV,
        duckdb_state.QUACK_MUTATION_BINDING_ENV,
        duckdb_state.QUACK_STORE_ID_ENV,
        duckdb_state.QUACK_TOKEN_ENV,
        "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION",
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
        "SAWM_QUACK_TOKEN",
    ):
        monkeypatch.setenv(name, "test-placeholder")

    with pytest.raises(
        operator.OperatorError,
        match="provider probe environment retained owner credential",
    ):
        operator._live_preflight(
            config,
            probe_provider=True,
            retire_provider_token_handoff=True,
        )
    assert provider_probe_called is False


def test_m28_historical_source_chain_uses_m29_control_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m28_historical_head_test",
    )
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m28_historical_head_test",
    )
    m29_key = "committed_evidence_verification_successor_materialization"
    m28_key = "live_claim_admission_recovery_successor_materialization"
    m29 = materializer._expected_m29_committed_evidence_verification_authority()
    m28 = materializer._expected_m28_live_claim_admission_recovery_authority()
    scheduler = {m29_key: m29, m28_key: m28}
    migration = {m29_key: m29, m28_key: m28}
    seal = {
        f"{m29_key}_cid": materializer._identity(m29),
        f"{m28_key}_cid": materializer._identity(m28),
    }
    observed: dict[str, object] = {}

    def source_chain(
        _root: Path,
        _materializer: object,
        _authority: object,
        *,
        current_head: str | None = None,
    ) -> list[str]:
        observed["current_head"] = current_head
        return []

    monkeypatch.setattr(dependency, "_m28_source_chain_errors", source_chain)
    assert dependency._m28_live_claim_admission_recovery_successor_errors(
        scheduler,
        seal,
        migration,
        root=REPO_ROOT,
        require_active_runtime=False,
    ) == []
    assert observed["current_head"] == (
        "d7e2a4ba9bc7eef32ffad131ffd092ff11f934c4"
    )


def test_m29_historical_source_chain_uses_m30_control_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m29_historical_head_test",
    )
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m29_historical_head_test",
    )
    m30_key = "stopped_owner_restart_source_seal_successor_materialization"
    m29_key = "committed_evidence_verification_successor_materialization"
    m30 = materializer._expected_m30_stopped_owner_restart_source_seal_authority()
    m29 = materializer._expected_m29_committed_evidence_verification_authority()
    scheduler = {m30_key: m30, m29_key: m29}
    migration = {m30_key: m30, m29_key: m29}
    seal = {
        f"{m30_key}_cid": materializer._identity(m30),
        f"{m29_key}_cid": materializer._identity(m29),
    }
    observed: dict[str, object] = {}

    def source_chain(
        _root: Path,
        _materializer: object,
        _authority: object,
        *,
        current_head: str | None = None,
    ) -> list[str]:
        observed["current_head"] = current_head
        return []

    monkeypatch.setattr(dependency, "_m29_source_chain_errors", source_chain)
    assert dependency._m29_committed_evidence_verification_successor_errors(
        scheduler,
        seal,
        migration,
        root=REPO_ROOT,
        require_active_runtime=False,
    ) == []
    assert observed["current_head"] == (
        "f8e0cb5c8887474267b354e03363bde0c4a08597"
    )


def test_m29_source_chain_history_rebinds_exact_m30_control_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m29_history_rebind_test",
    )
    control_head = "f8e0cb5c8887474267b354e03363bde0c4a08597"
    captured: dict[str, object] = {}

    class Materializer:
        @staticmethod
        def build_population(_root: Path) -> dict[str, object]:
            return {
                "source_binding": {
                    "head": "f" * 40,
                    "tree": "e" * 40,
                    "datasets_gitlink": "d" * 40,
                    "kit_gitlink": "c" * 40,
                }
            }

        @staticmethod
        def _assert_m29_source_delta(
            _root: Path,
            population: Mapping[str, object],
            _authority: Mapping[str, object],
        ) -> None:
            captured.update(population["source_binding"])

    def git(_root: Path, *args: str) -> str:
        ref = args[-1]
        if ref.endswith("^{tree}"):
            return "1" * 40
        if ref.endswith(":ipfs_datasets_py"):
            return "2" * 40
        if ref.endswith(":ipfs_kit_py"):
            return "3" * 40
        raise AssertionError(ref)

    monkeypatch.setattr(dependency, "_git", git)
    assert dependency._m29_source_chain_errors(
        REPO_ROOT,
        Materializer(),
        {},
        current_head=control_head,
    ) == []
    assert captured == {
        "head": control_head,
        "tree": "1" * 40,
        "datasets_gitlink": "2" * 40,
        "kit_gitlink": "3" * 40,
    }


def test_m28_source_chain_history_rebinds_exact_m29_control_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m28_history_rebind_test",
    )
    control_head = "d7e2a4ba9bc7eef32ffad131ffd092ff11f934c4"
    captured: dict[str, object] = {}

    class Materializer:
        @staticmethod
        def build_population(_root: Path) -> dict[str, object]:
            return {
                "source_binding": {
                    "head": "f" * 40,
                    "tree": "e" * 40,
                    "datasets_gitlink": "d" * 40,
                    "kit_gitlink": "c" * 40,
                }
            }

        @staticmethod
        def _assert_m28_source_delta(
            _root: Path,
            population: Mapping[str, object],
            _authority: Mapping[str, object],
        ) -> None:
            captured.update(population["source_binding"])

    def git(_root: Path, *args: str) -> str:
        ref = args[-1]
        if ref.endswith("^{tree}"):
            return "1" * 40
        if ref.endswith(":ipfs_datasets_py"):
            return "2" * 40
        if ref.endswith(":ipfs_kit_py"):
            return "3" * 40
        raise AssertionError(ref)

    monkeypatch.setattr(dependency, "_git", git)
    assert dependency._m28_source_chain_errors(
        REPO_ROOT,
        Materializer(),
        {},
        current_head=control_head,
    ) == []
    assert captured == {
        "head": control_head,
        "tree": "1" * 40,
        "datasets_gitlink": "2" * 40,
        "kit_gitlink": "3" * 40,
    }


def test_m28_live_claim_admission_recovery_authority_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m28_authority_test",
    )
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m28_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "live_claim_admission_recovery_successor_materialization"
    expected = (
        materializer._expected_m28_live_claim_admission_recovery_authority()
    )
    config, inventory, seal = _historical_successor_controls_at(
        key, config, inventory, seal
    )
    assert inventory is not None and seal is not None
    config["database_program"]["store_generation"] = "27"
    # Exercise the three sealed declaration surfaces without requiring this
    # unit test to run only after the operator has published the M28 control
    # commit.  The static gates validate the real files once any M28 surface
    # is present.
    config[key] = copy.deepcopy(expected)
    inventory[key] = copy.deepcopy(expected)
    seal[f"{key}_cid"] = materializer._identity(expected)

    assert config[key] == inventory[key] == expected
    assert seal[f"{key}_cid"] == materializer._identity(expected)
    assert expected["schema"] == (
        "sawm/live-claim-admission-recovery-successor-materialization-"
        "authorization@1"
    )
    assert expected["migration_revision"] == "SAWM-R2-M28"
    assert expected["prior_authority"]["coordination_event_count"] == 3_019
    assert expected["prior_authority"]["coordination_projection_digest"] == (
        "sha256:7abbd22e48ed99b31fb02190c6406b631f3341de22c9904af3d2b5b10d9509c0"
    )
    assert expected["runtime_binding"] == {
        "run_id": "run-r2-m27",
        "runtime_root": (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        ),
        "store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m27/control.duckdb"
        ),
        "coordination_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m27/control.coordination.duckdb"
        ),
        "worktree_root": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m27/worktrees"
        ),
        "store_generation": 27,
        "quack_port": 24_070,
        "quack_endpoint": "quack:127.0.0.1:24070",
        "database_uuid": "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
        "server_id": "server:ff5834df-4af2-4cb0-a4a4-7dbc4f258457",
        "process_birth_id": "birth:7157839e6e5bc6ce351f41c7d9cd6c94",
        "plan_revision": 28,
        "prior_event_watermark": 272,
        "target_event_watermark": 273,
    }
    repair = expected["historical_transition_repair"]
    assert repair["task_alias"] == "SAWM-012"
    assert repair["actual_configured_board_admission_cid"] == ""
    assert repair["expected_configured_board_admission_cid"] == (
        "baguqeeraelsagoqf62zk3etgymookxuzxrn763rwbo7i6iqpqte6ehj7giuq"
    )
    assert repair["accepted_transition_cid"] == (
        "sha256:97b5896f0e2f32fbdddc0a904a57ff0d5ddc11abbe1ecf3a9aad73178d77cd1f"
    )
    assert repair["implementation_commit"] == (
        "e9d03bff8b527e2b6c1a216cffb60a52a037c581"
    )
    assert repair["merge_commit"] == (
        "8d2acae71f4d8caaceaf2677d4f0ca87d15e606e"
    )
    assert repair["accepted_completion_changed"] is False
    assert repair["historical_completion_rewritten"] is False
    assert repair["task_completion_authority"] is False
    assert expected["exact_changes"] == {
        "event_suffix_length": 1,
        "evidence_node_changes": 1,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "plan_revision_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "coordination_semantic_changes": 0,
        "sidecar_changes": 0,
        "implementation_provider_invocations": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "store_generation_row_changes": 1,
        "state_server_row_changes": 2,
        "credential_row_changes": 1,
    }
    assert len(expected["operator_control_paths"]) == 9
    assert expected["preservation"]["sidecars_preserved"] is True
    assert expected["preservation"]["generation_bearing_owner_restart"] is True
    assert expected["preservation"]["worker_self_approval"] is False
    control, coordination = materializer._m28_target_paths(
        REPO_ROOT, config, expected
    )
    assert control == (
        REPO_ROOT
        / "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m27/control.duckdb"
    ).resolve()
    assert coordination == (
        REPO_ROOT
        / "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m27/control.coordination.duckdb"
    ).resolve()

    monkeypatch.setattr(
        dependency,
        "_m28_source_chain_errors",
        lambda *_args, **_kwargs: [],
    )
    assert dependency._m28_live_claim_admission_recovery_successor_errors(
        config, seal, inventory, root=REPO_ROOT
    ) == []

    changed = copy.deepcopy(config)
    changed[key]["exact_changes"]["accepted_completion_changes"] = 1
    errors = dependency._m28_live_claim_admission_recovery_successor_errors(
        changed, seal, inventory, root=REPO_ROOT
    )
    assert any("differs across controls" in error for error in errors)


def test_m28_presence_masks_m27_and_keeps_every_predecessor_historical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m28_presence_test",
    )
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m28_presence_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m28_presence_test",
    )
    key = "live_claim_admission_recovery_successor_materialization"
    authority = (
        materializer._expected_m28_live_claim_admission_recovery_authority()
    )
    scheduler = {key: authority}
    migration = {key: authority}
    seal = {f"{key}_cid": materializer._identity(authority)}
    assert dependency._m28_successor_declared(scheduler, {}, {}) is True
    assert dependency._m28_successor_declared({}, {}, migration) is True
    assert dependency._m28_successor_declared({}, seal, {}) is True
    assert dependency._m28_successor_declared({}, {}, {}) is False

    monkeypatch.setattr(
        board,
        "_m28_migration_errors",
        lambda *_args, **_kwargs: ["M28 active"],
    )
    historical_calls: list[bool] = []

    def historical(*_args: object, **kwargs: object) -> list[str]:
        historical_calls.append(kwargs.get("require_active_runtime") is False)
        return []

    for name in (
        "_m27_migration_errors", "_m26_migration_errors",
        "_m25_migration_errors", "_m24_migration_errors",
        "_m23_migration_errors", "_m22_migration_errors",
        "_m21_migration_errors", "_m20_migration_errors",
        "_m19_migration_errors", "_m18_migration_errors",
        "_m17_migration_errors", "_m16_migration_errors",
    ):
        monkeypatch.setattr(board, name, historical)
    assert board._active_successor_migration_errors(
        scheduler, seal, migration
    ) == ["M28 active"]
    assert historical_calls == [True] * 12

    partial = {key: authority}
    errors = board._active_successor_migration_errors(partial, {}, {})
    assert "M28 active" in errors
    assert any("only partially declared" in error for error in errors)


def test_m28_marker_keeps_prior_final_pair_and_source_receipts_distinct(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m28_marker_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m28_marker_test",
    )
    authority = materializer._expected_m28_live_claim_admission_recovery_authority()
    key = "live_claim_admission_recovery_successor_materialization"
    prior_receipt_cid = authority["prior_authority"]["migration_receipt_cid"]
    observed = {
        "schema": "sawm/non-authoritative-live-source-successor-receipt@1",
        "authoritative": False,
        "control_database_is_authority": True,
        "receipt_is_final_pair_commit_marker": False,
        "receipt_is_evidence_source_seal_marker": True,
        "migration_revision": "SAWM-R2-M28",
        f"{key}_cid": materializer._identity(authority),
        "database_path": operator._M28_STORE_ID,
        "coordination_path": operator._M28_COORDINATION_STORE_ID,
        "target_generation": 27,
        "target_plan_revision": 28,
        "target_event_watermark": 273,
        "projection_cid": operator._M28_TARGET_PROJECTION_CID,
        "coordination_projection_digest": authority["prior_authority"][
            "coordination_projection_digest"
        ],
        "coordination_event_count": authority["prior_authority"][
            "coordination_event_count"
        ],
        "generation_bearing_owner_restart_verified": True,
        "prior_owner_generation": 26,
        "live_owner_generation": 27,
        "queried_and_mutated_through_live_quack_only": True,
        "direct_authoritative_file_opened": False,
        "plan_revision_changes": 0,
        "evidence_node_changes": 1,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "coordination_semantic_changes": 0,
        "sidecars_preserved": True,
        "accepted_completion_changes": 0,
        "worker_self_approval": False,
    }
    observed["receipt_cid"] = materializer._identity(observed)
    receipt_path = (
        tmp_path / operator._M28_STORE_ID
    ).resolve().parent / "m28-source-successor-receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(observed, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(operator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        operator,
        "_require_m27_final_pair_marker",
        lambda *_args, **_kwargs: {"receipt_cid": prior_receipt_cid},
    )
    marker = operator._require_m28_source_successor_marker(
        {key: authority}, authority, materializer
    )
    assert marker["prior_final_pair_receipt_cid"] == prior_receipt_cid
    assert marker["source_successor_receipt_cid"] == observed["receipt_cid"]
    assert marker["receipt_is_final_pair_commit_marker"] is False

    monkeypatch.setattr(
        operator,
        "_require_m27_final_pair_marker",
        lambda *_args, **_kwargs: {"receipt_cid": "sha256:" + "0" * 64},
    )
    with pytest.raises(
        operator.OperatorError,
        match="M28 prior M27 final-pair receipt differs",
    ):
        operator._require_m28_source_successor_marker(
            {key: authority}, authority, materializer
        )


def test_m27_dead_owner_resume_authority_runtime_and_source_chain_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m27_authority_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m27_authority_test",
    )
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m27_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "dead_owner_parallel_resume_successor_materialization"
    expected = materializer._expected_m27_dead_owner_parallel_resume_authority()
    m27_config, _historical_inventory, _historical_seal = (
        _historical_successor_controls_at(key, config, inventory, seal)
    )
    # M28 advances only the live owner/store generation.  Reconstruct the
    # historical M27 runtime binding before exercising its closed authority.
    m27_config["database_program"]["store_generation"] = "26"

    assert config[key] == inventory[key] == expected
    assert seal[f"{key}_cid"] == materializer._identity(expected)
    assert materializer._m27_successor_configured(m27_config) is True
    assert dict(operator._active_source_repair_materialization(m27_config)) == expected
    assert expected["migration_revision"] == "SAWM-R2-M27"
    assert expected["precursor_source_head"] == (
        "bbff12f06ff200b0f9280c50ca3866e1a752f3e9"
    )
    assert expected["prior_control_source_head"] == (
        "5e4aa2bc1c90527ba3fa8cb272b785141f50e832"
    )
    assert expected["source_chain"]["sawm_012_worker_commit"] == (
        "e3a1fc0fd04f00e1ca64fba0252905d8e0f144bb"
    )
    assert expected["source_chain"]["two_parent_merge_commit"] == (
        "cdced896c85e331fc32d7ad623aa2eb926b6f1c6"
    )
    assert expected["prior_event_watermark"] == 264
    assert expected["target_event_watermark"] == 268
    assert expected["prior_coordination_event_count"] == 2_582
    assert expected["target_coordination_event_count"] == 2_586
    assert expected["target_coordination_projection_digest"] == (
        "sha256:40d0bcc156301700b145f93b688c0f0e0b7e6387c7c30a2eaa462284de620465"
    )
    assert set(expected["interrupted_claims"]) == {
        "SAWM-006", "SAWM-008", "SAWM-012", "SAWM-015"
    }
    assert set(expected["task_rearms"]) == {"SAWM-006", "SAWM-012"}
    assert expected["unchanged_retrying_tasks"] == ["SAWM-008", "SAWM-015"]
    assert expected["accepted_definition_changes"] == 0
    assert expected["accepted_completion_changes"] == 0
    assert expected["implementation_provider_invocations"] == 0
    assert expected["worker_self_approval"] is False
    assert expected["observed_unaccepted_source"]["completion_authority"] is False
    assert config["database_program"]["store_id"].endswith(
        "run-r2-m27/control.duckdb"
    )
    assert config["database_program"]["store_generation"] == "35"
    assert m27_config["database_program"]["store_generation"] == "26"
    assert config["database_program"]["quack_endpoint"] == (
        "quack:127.0.0.1:24070"
    )
    preserved = (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m26/worktrees"
    )
    active = (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m27/worktrees"
    )
    assert expected["preserved_worktree_root"] == preserved
    assert config["database_program"]["worktree_root"] == active
    assert config["runtime_paths"]["worktrees"] == active
    assert config["quack_owner"]["port"] == 24_070

    monkeypatch.setattr(
        dependency,
        "_m27_source_chain_errors",
        lambda *_args, **_kwargs: [],
    )
    assert dependency._m27_dead_owner_parallel_resume_successor_errors(
        m27_config, seal, inventory, root=REPO_ROOT
    ) == []

    malformed = copy.deepcopy(m27_config)
    malformed[key] = None
    with pytest.raises(
        materializer.MaterializationError,
        match="M27 stopped-run recovery authority is invalid",
    ):
        materializer._m27_successor_configured(malformed)
    with pytest.raises(
        operator.OperatorError,
        match="active M27 dead-owner parallel-resume authority is invalid",
    ):
        operator._active_source_repair_materialization(malformed)


def test_m27_presence_masks_m26_and_keeps_every_predecessor_historical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m27_presence_test",
    )
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m27_presence_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m27_presence_test",
    )
    key = "dead_owner_parallel_resume_successor_materialization"
    authority = materializer._expected_m27_dead_owner_parallel_resume_authority()
    scheduler = {key: authority}
    migration = {key: authority}
    seal = {f"{key}_cid": materializer._identity(authority)}
    assert dependency._m27_successor_declared(scheduler, {}, {}) is True
    assert dependency._m27_successor_declared({}, {}, migration) is True
    assert dependency._m27_successor_declared({}, seal, {}) is True
    assert dependency._m27_successor_declared({}, {}, {}) is False

    monkeypatch.setattr(
        board, "_m27_migration_errors", lambda *_args, **_kwargs: ["M27 active"]
    )
    historical_calls: list[bool] = []

    def historical(*_args: object, **kwargs: object) -> list[str]:
        historical_calls.append(kwargs.get("require_active_runtime") is False)
        return []

    for name in (
        "_m26_migration_errors", "_m25_migration_errors",
        "_m24_migration_errors", "_m23_migration_errors",
        "_m22_migration_errors", "_m21_migration_errors",
        "_m20_migration_errors", "_m19_migration_errors",
        "_m18_migration_errors", "_m17_migration_errors",
        "_m16_migration_errors",
    ):
        monkeypatch.setattr(board, name, historical)
    assert board._active_successor_migration_errors(
        scheduler, seal, migration
    ) == ["M27 active"]
    assert historical_calls == [True] * 11

    partial = {key: authority}
    errors = board._active_successor_migration_errors(partial, {}, {})
    assert "M27 active" in errors
    assert any("only partially declared" in error for error in errors)


def test_m27_final_marker_rejects_self_rehashed_authority_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m27_marker_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m27_marker_test",
    )
    authority = materializer._expected_m27_dead_owner_parallel_resume_authority()
    key = "dead_owner_parallel_resume_successor_materialization"
    config = {key: authority}
    monkeypatch.setattr(operator, "REPO_ROOT", tmp_path)
    control = tmp_path / operator._M27_STORE_ID
    coordination = tmp_path / operator._M27_COORDINATION_STORE_ID
    control.parent.mkdir(parents=True)
    control.write_bytes(b"control-live-head")
    coordination.write_bytes(b"coordination-materialized-head")
    coordination_hash = materializer._stable_regular_sha256(
        coordination,
        root=tmp_path,
        noun="test M27 coordination store",
        required_link_count=1,
    )
    marker = {
        "schema": "sawm/non-authoritative-migration-receipt@25",
        "authoritative": False,
        "receipt_is_final_pair_commit_marker": True,
        "migration_revision": "SAWM-R2-M27",
        f"{key}_cid": materializer._identity(authority),
        "database_path": operator._M27_STORE_ID,
        "coordination_path": operator._M27_COORDINATION_STORE_ID,
        "target_runtime_root": str(Path(operator._M27_STORE_ID).parent),
        "target_generation": 26,
        "target_quack_port": 24_070,
        "target_plan_revision": 28,
        "target_event_watermark": 268,
        "task_revision_changes": 2,
        "task_status_changes": 2,
        "orphan_claim_expirations": 4,
        "coordination_semantic_changes": 4,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "worker_self_approval": False,
        "control_store_sha256": "0" * 64,
        "control_store_size": 1,
        "coordination_store_sha256": coordination_hash[0],
        "coordination_store_size": coordination_hash[1],
        "coordination_projection_digest": authority[
            "target_coordination_projection_digest"
        ],
        "coordination_event_count": 2_586,
        "interrupted_claims": {
            alias: {} for alias in ("SAWM-006", "SAWM-008", "SAWM-012", "SAWM-015")
        },
        "task_rearms": {"SAWM-006": {}, "SAWM-012": {}},
    }

    def publish(body: dict[str, object]) -> None:
        unhashed = dict(body)
        unhashed["receipt_cid"] = materializer._identity(unhashed)
        (control.parent / "migration-receipt.json").write_text(
            json.dumps(unhashed, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    publish(marker)
    assert dict(
        operator._require_m27_final_pair_marker(
            config, authority, materializer
        )
    )["coordination_projection_digest"] == authority[
        "target_coordination_projection_digest"
    ]
    coordination.write_bytes(b"coordination-evolved-after-m27")
    with pytest.raises(
        operator.OperatorError,
        match="M27 materialized final pair marker differs",
    ):
        operator._require_m27_final_pair_marker(
            config, authority, materializer
        )
    assert dict(
        operator._require_m27_final_pair_marker(
            config,
            authority,
            materializer,
            require_current_coordination_store=False,
        )
    )["coordination_event_count"] == 2_586
    coordination.write_bytes(b"coordination-materialized-head")
    for field, bad_value in (
        ("schema", "sawm/non-authoritative-migration-receipt@24"),
        ("coordination_projection_digest", "sha256:" + "f" * 64),
        ("accepted_definition_changes", 1),
    ):
        tampered = dict(marker)
        tampered[field] = bad_value
        publish(tampered)
        with pytest.raises(
            operator.OperatorError,
            match="M27 materialized final pair marker differs",
        ):
            operator._require_m27_final_pair_marker(
                config, authority, materializer
            )


def test_m27_private_stage_expires_four_claims_and_preserves_completion_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m27_private_stage_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m27_private_stage_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    authority = materializer._expected_m27_dead_owner_parallel_resume_authority()
    config, _historical_migration, _historical_seal = (
        _historical_successor_controls_at(
            "dead_owner_parallel_resume_successor_materialization",
            config,
        )
    )
    # Rehearse M27 against its exact historical generation/source projection;
    # M28 advanced only the live owner and M29 preserves that later state.
    config["database_program"]["store_generation"] = "26"
    population = copy.deepcopy(materializer.build_population(REPO_ROOT))
    population["source_binding"]["kit_gitlink"] = authority[
        "current_kit_gitlink"
    ]
    historical_source_binding_cid = (
        "sha256:83e28e01de41699d5b2312ead03e7f33d9989d809d97924ea0a230af2c038856"
    )
    population["source_binding"][
        "source_binding_cid"
    ] = historical_source_binding_cid
    population["repository_tree_id"] = historical_source_binding_cid
    prior_control, prior_coordination = materializer._assert_m27_prior_anchor(
        REPO_ROOT, authority, population
    )
    prior_hashes = (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    )
    monkeypatch.setattr(
        materializer.time,
        "time_ns",
        lambda: materializer._M27_CONTROL_RECORDED_AT_MS * 1_000_000,
    )
    stage_dir = tmp_path / "m27-stage"
    stage_dir.mkdir()
    validation_digest = materializer._identity(
        {
            "schema": "sawm/m27-private-stage-validation@1",
            "source_binding_cid": population["source_binding"][
                "source_binding_cid"
            ],
        }
    )
    staged = materializer._stage_m27_store_pair(
        REPO_ROOT,
        stage_dir,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    verified = staged["verified"]
    assert verified["valid"] is True
    assert verified["event_watermark"] == 268
    assert verified["coordination_event_count"] == 2_586
    assert verified["coordination_projection_digest"] == authority[
        "target_coordination_projection_digest"
    ]
    assert verified["active_claim_count"] == 0
    assert verified["active_attempt_count"] == 0
    assert verified["active_lease_count"] == 0
    assert verified["orphan_claim_expirations"] == 4
    assert verified["task_revision_changes"] == 2
    assert verified["task_status_changes"] == 2
    assert verified["accepted_completion_changes"] == 0
    assert verified["implementation_provider_invocations"] == 0
    assert prior_hashes == (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    )

    import duckdb

    def completion_rows(path: Path) -> list[tuple[object, ...]]:
        connection = duckdb.connect(str(path), read_only=True)
        try:
            return connection.execute(
                "SELECT * FROM completion_receipts ORDER BY receipt_cid"
            ).fetchall()
        finally:
            connection.close()

    assert completion_rows(staged["stage_control"]) == completion_rows(prior_control)
    stage_hash = materializer._store_sha256(staged["stage_control"])
    assert materializer._verify_m27_store_pair_copy(
        staged["stage_control"],
        staged["stage_coordination"],
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    ) == verified
    assert materializer._store_sha256(staged["stage_control"]) == stage_hash
    assert completion_rows(staged["stage_control"]) == completion_rows(prior_control)
    assert not tuple(stage_dir.glob("*.wal"))

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    source = DatabaseTaskSource(staged["stage_control"], install_schema=False)
    try:
        statuses, revisions, _receipts = (
            operator._verify_m27_live_head_task_projection(
                source,
                population,
                materializer,
                expected_projection_cid=verified["projection_cid"],
            )
        )
    finally:
        source.close()
    assert statuses["SAWM-006"] == "retrying"
    assert revisions["SAWM-006"] == 4
    assert statuses["SAWM-008"] == "retrying"
    assert revisions["SAWM-008"] == 8
    assert statuses["SAWM-012"] == "retrying"
    assert revisions["SAWM-012"] == 7
    assert statuses["SAWM-015"] == "retrying"
    assert revisions["SAWM-015"] == 5

    receipt = materializer._expected_m27_migration_receipt(
        tmp_path,
        staged["stage_control"],
        staged["stage_coordination"],
        population,
        verified,
        validation_digest,
    )
    unhashed = dict(receipt)
    assert unhashed.pop("receipt_cid") == materializer._identity(unhashed)
    assert receipt["schema"] == "sawm/non-authoritative-migration-receipt@25"
    assert receipt["authoritative"] is False
    assert receipt["receipt_is_final_pair_commit_marker"] is True
    assert receipt["accepted_completion_changes"] == 0
    assert receipt["worker_self_approval"] is False


def test_m27_staging_refuses_to_time_travel_before_last_lease_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m27_clock_test",
    )
    monkeypatch.setattr(
        materializer.time,
        "time_ns",
        lambda: (materializer._M27_CONTROL_RECORDED_AT_MS - 1) * 1_000_000,
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="before every exact claim has expired",
    ):
        materializer._m27_require_actual_expiry_time()


def test_m26_stopped_recovery_authority_and_runtime_bindings_are_exact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m26_authority_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m26_authority_test",
    )
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m26_authority_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m26_supplied_root_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "automatic_stall_recovery_successor_materialization"
    config, inventory, seal = _historical_successor_controls_at(
        key, config, inventory, seal
    )
    assert inventory is not None
    assert seal is not None
    runtime = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m26"
    store = f"{runtime}/control.duckdb"
    worktrees = f"{runtime}/worktrees"
    config["database_program"].update(
        {
            "store_id": store,
            "store_generation": "25",
            "quack_endpoint": "quack:127.0.0.1:24069",
            "event_store_path": f"{runtime}/events",
            "runtime_registry_path": f"{runtime}/registry",
            "worktree_root": worktrees,
        }
    )
    config["quack_owner"].update(
        {
            "database_path": store,
            "store_id": store,
            "state_dir": f"{runtime}/quack-owner",
            "port": 24_069,
        }
    )
    config["runtime_paths"] = {
        "root": runtime,
        "state": f"{runtime}/state",
        "worktrees": worktrees,
        "merge_queue": f"{runtime}/merge-queue",
        "logs": f"{runtime}/logs",
        "generated_runtime_artifacts_are_completion_authority": False,
    }
    expected = materializer._expected_m26_automatic_stall_recovery_authority()

    assert config[key] == inventory[key] == expected
    assert seal[f"{key}_cid"] == materializer._identity(expected)
    assert materializer._m26_successor_configured(config) is True
    assert dict(operator._active_source_repair_materialization(config)) == expected
    assert expected["migration_revision"] == "SAWM-R2-M26"
    assert expected["repair_source_commit"] == (
        "bd39c5eee1f607ff23bc8632aae1933253fc9e05"
    )
    assert expected["repair_source_tree"] == (
        "cd805dd1ae95c8f76009f9d257179d74fc23230c"
    )
    assert expected["prior_event_watermark"] == 258
    assert expected["target_event_watermark"] == 262
    assert expected["prior_coordination_event_count"] == 2_417
    assert expected["target_coordination_event_count"] == 2_421
    assert expected["target_coordination_projection_digest"] == (
        "sha256:96b801b636e67c62ec35bc6338ba4b39494b0d6c8ca7439e8d88f4beab60c085"
    )
    assert set(expected["orphan_claims"]) == {"SAWM-006", "SAWM-015"}
    assert set(expected["failure_receipts"]) == {"SAWM-008", "SAWM-012"}
    assert set(expected["task_rearms"]) == {"SAWM-008", "SAWM-012"}
    assert expected["observed_source_chain"]["SAWM-010"] == {
        "merge_commit": "17f6c16a8722770db24890ef4b21b64b4b77e813",
        "completion_authority": True,
        "accepted_transition_cid": (
            "sha256:9d77ca0df59a20d23d99b1e10b6ef70c96ff34d1a22a82775fccec73c6127289"
        ),
        "status": "completed",
    }
    assert expected["observed_source_chain"]["SAWM-012"] == {
        "merge_commit": "f4639af1c5cd71f87df26e33b3bd746a9d385309",
        "completion_authority": False,
        "accepted_transition_cid": None,
        "status": "landed_but_unaccepted_false_terminal",
    }
    assert expected["accepted_completion_changes"] == 0
    assert expected["implementation_provider_invocations"] == 0
    assert expected["worker_self_approval"] is False
    assert expected["later_source_admission"] == {
        "accepted_source_transition_schema": (
            "ipfs_accelerate_py/agent-supervisor/accepted-source-transition@3"
        ),
        "bare_descendant_allowed": False,
        "canonical_quack_completion_required": True,
        "exact_control_commit_required_initially": True,
        "two_parent_supervisor_merge_required": True,
    }
    assert config["database_program"]["store_id"].endswith(
        "run-r2-m26/control.duckdb"
    )
    assert config["database_program"]["store_generation"] == "25"
    assert config["database_program"]["quack_endpoint"] == (
        "quack:127.0.0.1:24069"
    )
    assert config["quack_owner"]["port"] == 24_069

    monkeypatch.setattr(
        dependency, "_m26_source_chain_errors", lambda *_, **__: []
    )
    assert dependency._m26_automatic_stall_recovery_successor_errors(
        config,
        seal,
        inventory,
        root=REPO_ROOT,
    ) == []

    malformed = copy.deepcopy(config)
    malformed[key] = None
    with pytest.raises(
        materializer.MaterializationError,
        match="M26 automatic-stall-recovery authority is invalid",
    ):
        materializer._m26_successor_configured(malformed)
    with pytest.raises(
        operator.OperatorError,
        match="active M26 automatic-stall-recovery authority is invalid",
    ):
        operator._active_source_repair_materialization(malformed)

    supplied_root = tmp_path / "staged-repository"
    observed: dict[str, object] = {}
    staged_dependency = object()
    staged_materializer = object()

    def load_staged_dependency(root: Path) -> object:
        observed["dependency_root"] = root
        return staged_dependency

    def staged_spec(_name: str, path: Path) -> object:
        observed["materializer_path"] = path
        return SimpleNamespace(
            loader=SimpleNamespace(exec_module=lambda _module: None)
        )

    monkeypatch.setattr(board, "_dependency_validator_module", load_staged_dependency)
    monkeypatch.setattr(board.importlib.util, "spec_from_file_location", staged_spec)
    monkeypatch.setattr(
        board.importlib.util,
        "module_from_spec",
        lambda _spec: staged_materializer,
    )
    loaded_dependency, loaded_materializer = board._m26_validation_modules(
        supplied_root
    )
    assert loaded_dependency is staged_dependency
    assert loaded_materializer is staged_materializer
    assert observed == {
        "dependency_root": supplied_root,
        "materializer_path": (
            supplied_root
            / "scripts/materialize_semantic_addressed_world_model_program.py"
        ),
    }

def test_m26_key_presence_masks_m25_and_keeps_history_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m26_presence_test",
    )
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m26_presence_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m26_presence_test",
    )
    key = "automatic_stall_recovery_successor_materialization"
    authority = materializer._expected_m26_automatic_stall_recovery_authority()
    scheduler = {
        key: authority,
        "native_duckdb_preload_successor_materialization": {},
    }
    migration = {key: authority}
    seal = {f"{key}_cid": materializer._identity(authority)}
    assert dependency._m26_successor_declared(scheduler, {}, {}) is True
    assert dependency._m26_successor_declared({}, {}, migration) is True
    assert dependency._m26_successor_declared({}, seal, {}) is True
    assert dependency._m26_successor_declared({}, {}, {}) is False

    m25_key = "native_duckdb_preload_successor_materialization"
    m25_authority = materializer._expected_m25_native_duckdb_preload_authority()
    _effective, nested_errors = dependency._effective_nested_source_authorities(
        [{"package": "ipfs_datasets_py"}, {"package": "ipfs_kit_py"}],
        {key: authority, m25_key: m25_authority},
        {m25_key: m25_authority},
        {f"{m25_key}_cid": materializer._identity(m25_authority)},
    )
    assert nested_errors == ["active M26 nested-source authority is partial"]

    monkeypatch.setattr(
        board, "_m26_migration_errors", lambda *_args, **_kwargs: ["M26 active"]
    )
    monkeypatch.setattr(
        board, "_m25_migration_errors", lambda *_args, **_kwargs: ["M25 history"]
    )
    for name in (
        "_m24_migration_errors",
        "_m23_migration_errors",
        "_m22_migration_errors",
        "_m21_migration_errors",
        "_m20_migration_errors",
        "_m19_migration_errors",
        "_m18_migration_errors",
        "_m17_migration_errors",
        "_m16_migration_errors",
    ):
        monkeypatch.setattr(board, name, lambda *_args, **_kwargs: [])
    assert board._active_successor_migration_errors(
        scheduler, seal, migration
    ) == ["M26 active", "M25 history"]

    partial = dict(scheduler)
    errors = board._active_successor_migration_errors(partial, {}, migration)
    assert "M26 active" in errors
    assert any("only partially declared" in error for error in errors)


def test_m26_source_chain_is_one_repair_then_one_nine_control_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m26_source_chain_test",
    )
    authority = materializer._expected_m26_automatic_stall_recovery_authority()
    prior = authority["prior_source_head"]
    repair = authority["repair_source_commit"]
    final = "f" * 40
    final_tree = "e" * 40
    parent_line = [final, repair]
    population = {
        "source_binding": {
            "head": final,
            "tree": final_tree,
            "datasets_gitlink": authority["prior_datasets_gitlink"],
            "kit_gitlink": authority["prior_kit_gitlink"],
        }
    }

    def fake_git(root: Path, *args: str) -> str:
        if args[:3] == ("diff", "--name-status", "--no-renames"):
            before, after = args[3], args[4]
            if (before, after) == (prior, repair):
                statuses = {path: "M" for path in authority["source_repair_paths"]}
                statuses[
                    "test/api/test_agent_supervisor_database_dispatch_watchdog.py"
                ] = "A"
            elif (before, after) == (repair, final):
                statuses = {path: "M" for path in authority["operator_control_paths"]}
            elif (before, after) == (prior, final):
                statuses = {path: "M" for path in authority["source_repair_paths"]}
                statuses[
                    "test/api/test_agent_supervisor_database_dispatch_watchdog.py"
                ] = "A"
                statuses.update(
                    {path: "M" for path in authority["operator_control_paths"]}
                )
            else:
                raise AssertionError(args)
            return "\n".join(
                f"{status}\t{path}" for path, status in sorted(statuses.items())
            )
        if args[:2] == ("merge-base", "--is-ancestor"):
            return ""
        if args[:4] == ("rev-list", "--parents", "-n", "1"):
            return " ".join(parent_line)
        if args[:2] == ("rev-list", "--count"):
            return "1"
        if args[0] != "rev-parse":
            raise AssertionError(args)
        ref = args[1]
        trees = {
            f"{prior}^{{tree}}": authority["prior_source_tree"],
            f"{repair}^{{tree}}": authority["repair_source_tree"],
            f"{final}^{{tree}}": final_tree,
        }
        if ref in trees:
            return str(trees[ref])
        for path, blob in authority["repair_source_blobs"].items():
            if ref == f"{repair}:{path}":
                return str(blob)
        for dependency_name, gitlink_key, tree_key in (
            ("ipfs_datasets_py", "prior_datasets_gitlink", "prior_datasets_tree"),
            ("ipfs_kit_py", "prior_kit_gitlink", "prior_kit_tree"),
        ):
            if ref in {
                f"{head}:{dependency_name}" for head in (prior, repair, final)
            }:
                return str(authority[gitlink_key])
            if root.name == dependency_name and ref == (
                f"{authority[gitlink_key]}^{{tree}}"
            ):
                return str(authority[tree_key])
        raise AssertionError((root, args))

    monkeypatch.setattr(materializer, "_git", fake_git)
    materializer._assert_m26_source_delta(REPO_ROOT, population, authority)
    parent_line.append("0" * 40)
    with pytest.raises(
        materializer.MaterializationError,
        match="one exact repair plus one nine-control commit",
    ):
        materializer._assert_m26_source_delta(REPO_ROOT, population, authority)


def test_m26_private_stage_expires_orphans_and_rearms_only_exact_failures(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m26_private_stage_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m26_private_stage_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    authority = materializer._expected_m26_automatic_stall_recovery_authority()
    population = materializer.build_population(REPO_ROOT)
    population = copy.deepcopy(population)
    population["source_binding"]["kit_gitlink"] = authority["prior_kit_gitlink"]
    population["source_binding"]["kit_tree"] = authority["prior_kit_tree"]
    config.pop("dead_owner_parallel_resume_successor_materialization", None)
    runtime = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m26"
    store = f"{runtime}/control.duckdb"
    config["database_program"].update(
        {
            "store_id": store,
            "store_generation": "25",
            "quack_endpoint": "quack:127.0.0.1:24069",
            "event_store_path": f"{runtime}/events",
            "runtime_registry_path": f"{runtime}/registry",
            "worktree_root": f"{runtime}/worktrees",
        }
    )
    config["quack_owner"].update(
        {
            "database_path": store,
            "store_id": store,
            "state_dir": f"{runtime}/quack-owner",
            "port": 24_069,
        }
    )
    config["runtime_paths"] = {
        "root": runtime,
        "state": f"{runtime}/state",
        "worktrees": f"{runtime}/worktrees",
        "merge_queue": f"{runtime}/merge-queue",
        "logs": f"{runtime}/logs",
        "generated_runtime_artifacts_are_completion_authority": False,
    }
    prior_control, prior_coordination = materializer._assert_m26_prior_anchor(
        REPO_ROOT,
        authority,
        population,
    )
    prior_hashes = (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    )
    stage_dir = tmp_path / "m26-stage"
    stage_dir.mkdir()
    validation_digest = materializer._identity(
        {
            "schema": "sawm/m26-private-stage-validation@1",
            "source_binding_cid": population["source_binding"][
                "source_binding_cid"
            ],
        }
    )
    staged = materializer._stage_m26_store_pair(
        REPO_ROOT,
        stage_dir,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    import duckdb

    def completion_receipt_rows(path: Path) -> list[tuple[object, ...]]:
        connection = duckdb.connect(str(path), read_only=True)
        try:
            return connection.execute(
                "SELECT * FROM completion_receipts ORDER BY receipt_cid"
            ).fetchall()
        finally:
            connection.close()

    verified = staged["verified"]
    assert verified["valid"] is True
    assert verified["event_watermark"] == 262
    assert verified["coordination_event_count"] == 2_421
    assert verified["coordination_projection_digest"] == authority[
        "target_coordination_projection_digest"
    ]
    assert verified["active_claim_count"] == 0
    assert verified["active_attempt_count"] == 0
    assert verified["active_lease_count"] == 0
    assert verified["orphan_claim_expirations"] == 2
    assert verified["logical_completion_removals_for_rearm"] == 2
    assert verified["task_revision_changes"] == 2
    assert verified["task_status_changes"] == 2
    assert verified["accepted_completion_changes"] == 0
    assert verified["implementation_provider_invocations"] == 0
    assert prior_hashes == (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    )
    # M26 has no completion transition.  Its replay qualification must run on
    # a disposable copy and preserve every accepted historical receipt field,
    # including evidence_digests that older completion events cannot rebuild.
    assert completion_receipt_rows(staged["stage_control"]) == (
        completion_receipt_rows(prior_control)
    )
    staged_control_sha256 = materializer._store_sha256(staged["stage_control"])
    reverified = materializer._verify_m26_store_pair_copy(
        staged["stage_control"],
        staged["stage_coordination"],
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    assert reverified == verified
    assert materializer._store_sha256(staged["stage_control"]) == (
        staged_control_sha256
    )
    assert completion_receipt_rows(staged["stage_control"]) == (
        completion_receipt_rows(prior_control)
    )
    assert not tuple(stage_dir.glob("*.wal"))

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    source = DatabaseTaskSource(staged["stage_control"], install_schema=False)
    try:
        statuses, revisions, _receipts = (
            operator._verify_m26_live_head_task_projection(
                source,
                population,
                materializer,
                expected_projection_cid=verified["projection_cid"],
            )
        )
        assert source.get_task("SAWM-012").body["completion_receipt"] == (
            materializer._m26_task_rearm_receipt("SAWM-012")
        )
    finally:
        source.close()
    assert statuses["SAWM-006"] == "todo"
    assert revisions["SAWM-006"] == 2
    assert statuses["SAWM-015"] == "retrying"
    assert revisions["SAWM-015"] == 5
    assert statuses["SAWM-008"] == "retrying"
    assert revisions["SAWM-008"] == 8
    assert statuses["SAWM-012"] == "retrying"
    assert revisions["SAWM-012"] == 5

    receipt = materializer._expected_m26_migration_receipt(
        tmp_path,
        staged["stage_control"],
        staged["stage_coordination"],
        population,
        verified,
        validation_digest,
    )
    unhashed = dict(receipt)
    assert unhashed.pop("receipt_cid") == materializer._identity(unhashed)
    assert receipt["schema"] == "sawm/non-authoritative-migration-receipt@24"
    assert receipt["authoritative"] is False
    assert receipt["receipt_is_final_pair_commit_marker"] is True
    assert receipt["observed_source_chain"]["SAWM-012"][
        "completion_authority"
    ] is False
    assert receipt["accepted_completion_changes"] == 0
    assert receipt["worker_self_approval"] is False


def test_m25_native_preload_authority_and_four_lane_bindings_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m25_authority_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m25_authority_test",
    )
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m25_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "native_duckdb_preload_successor_materialization"
    expected = materializer._expected_m25_native_duckdb_preload_authority()
    config, inventory, seal = _historical_successor_controls_at(
        key, config, inventory, seal
    )
    assert inventory is not None
    assert seal is not None
    runtime = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m25"
    store = f"{runtime}/control.duckdb"
    config["database_program"].update(
        {
            "store_id": store,
            "store_generation": "24",
            "quack_endpoint": "quack:127.0.0.1:24068",
            "event_store_path": f"{runtime}/events",
            "runtime_registry_path": f"{runtime}/registry",
            "worktree_root": f"{runtime}/worktrees",
        }
    )
    config["quack_owner"].update(
        {
            "database_path": store,
            "store_id": store,
            "state_dir": f"{runtime}/quack-owner",
            "port": 24_068,
        }
    )
    config["runtime_paths"] = {
        "root": runtime,
        "state": f"{runtime}/state",
        "worktrees": f"{runtime}/worktrees",
        "merge_queue": f"{runtime}/merge-queue",
        "logs": f"{runtime}/logs",
        "generated_runtime_artifacts_are_completion_authority": False,
    }

    assert config[key] == inventory[key] == expected
    assert seal[f"{key}_cid"] == materializer._identity(expected)
    assert materializer._m25_successor_configured(config) is True
    assert dict(operator._active_source_repair_materialization(config)) == expected
    assert expected["migration_revision"] == "SAWM-R2-M25"
    assert expected["prior_source_head"] == (
        "bb02830699388df9b52c01830c9e970a56c56796"
    )
    assert expected["repair_source_commit"] == (
        "135b077c5ad9482bbb167fdfc81d8b7855ff5fab"
    )
    assert expected["prior_failed_start_control_store_sha256"] == (
        "27c7bf7f923005ec66eec6f4b75b68cbe84fd3c8bf6273ebbb2b6d954ac64f35"
    )
    assert expected["prior_coordination_store_sha256"] == (
        "0671fbc77fa65bb30bf2c7227ccf963b9833223a9ed178461c278b21413e1957"
    )
    assert expected["maximum_persisted_generation"] == 23
    assert expected["failed_attempted_generation"] == 24
    assert expected["failed_attempt_allocated_generation"] is False
    assert expected[
        "prior_materialization_receipt_precedes_failed_start_checkpoint"
    ] is True
    failure = expected["failed_quack_start"]
    diagnosis = failure["diagnosis"]
    assert expected["failed_quack_start_cid"] == materializer._identity(failure)
    assert failure["phase"] == "pre_identity_replica_extension_load"
    assert failure["failed_operation"] == "LOAD httpfs"
    assert failure["load_quack_reached"] is False
    assert failure["quack_serve_reached"] is False
    assert failure["embedded_quack_involved"] is False
    assert diagnosis["failed_process_native_distribution_version"] == "1.5.2"
    assert failure["identity_publication_reached"] is False
    assert failure["state_server_row_created"] is False
    assert failure["store_generation_row_created"] is False
    repair = expected["native_preload_repair"]
    assert repair["ambient_loader_environment_rejected"] is True
    assert repair["sanitized_process_birth_required"] is True
    assert repair["in_process_loader_environment_mutation"] is False
    assert "ambient_loader_paths_removed" not in repair
    assert expected["target_generation"] == 24
    assert expected["target_quack_port"] == 24_068
    assert expected["target_plan_revision"] == 26
    assert expected["target_event_watermark"] == 251
    assert expected["target_coordination_event_count"] == 1_407
    for field in (
        "task_revision_changes",
        "task_status_changes",
        "coordination_semantic_changes",
        "accepted_completion_changes",
        "implementation_provider_invocations",
    ):
        assert expected[field] == 0
    assert config["database_program"]["store_id"].endswith(
        "run-r2-m25/control.duckdb"
    )
    assert config["database_program"]["store_generation"] == "24"
    assert config["database_program"]["quack_endpoint"] == (
        "quack:127.0.0.1:24068"
    )
    assert config["quack_owner"]["port"] == 24_068
    assert [
        (
            lane["index"],
            lane["name"],
            lane["strict_shard_remainder"],
            lane["initial_task_ids"],
        )
        for lane in config["lanes"]
    ] == [
        (0, "sawm-lane-0", 0, ["SAWM-008"]),
        (1, "sawm-lane-1", 1, ["SAWM-006", "SAWM-010"]),
        (2, "sawm-lane-2", 2, ["SAWM-015"]),
        (3, "sawm-lane-3", 3, ["SAWM-012"]),
    ]

    monkeypatch.setattr(
        dependency, "_m25_source_chain_errors", lambda *_, **__: []
    )
    assert dependency._m25_native_duckdb_preload_successor_errors(
        config,
        seal,
        inventory,
        root=REPO_ROOT,
    ) == []

    malformed = copy.deepcopy(config)
    malformed[key] = None
    with pytest.raises(
        materializer.MaterializationError,
        match="M25 native-preload authority is invalid",
    ):
        materializer._m25_successor_configured(malformed)
    with pytest.raises(
        operator.OperatorError,
        match="active M25 native-DuckDB preload successor authority is invalid",
    ):
        operator._active_source_repair_materialization(malformed)


def test_m25_key_presence_masks_m24_and_preserves_historical_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m25_presence_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m25_presence_test",
    )
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m25_presence_test",
    )
    board = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m25_presence_test",
    )
    key = "native_duckdb_preload_successor_materialization"
    authority = materializer._expected_m25_native_duckdb_preload_authority()
    scheduler = {
        key: authority,
        "multi_lane_sidecar_reopen_successor_materialization": {},
    }
    migration = {key: authority}
    seal = {f"{key}_cid": materializer._identity(authority)}
    assert dependency._m25_successor_declared(scheduler, {}, {}) is True
    assert dependency._m25_successor_declared({}, {}, migration) is True
    assert dependency._m25_successor_declared({}, seal, {}) is True
    assert dependency._m25_successor_declared({}, {}, {}) is False

    monkeypatch.setattr(
        board, "_m25_migration_errors", lambda *_args, **_kwargs: ["M25 active"]
    )
    for name, result in (
        ("_m24_migration_errors", ["M24 history"]),
        ("_m23_migration_errors", ["M23 history"]),
        ("_m22_migration_errors", ["M22 history"]),
        ("_m21_migration_errors", ["M21 history"]),
        ("_m20_migration_errors", ["M20 history"]),
        ("_m19_migration_errors", ["M19 history"]),
        ("_m18_migration_errors", ["M18 history"]),
        ("_m17_migration_errors", ["M17 history"]),
        ("_m16_migration_errors", ["M16 history"]),
    ):
        monkeypatch.setattr(
            board,
            name,
            lambda *_args, _result=result, **_kwargs: _result,
        )
    assert board._active_successor_migration_errors(
        scheduler, seal, migration
    ) == [
        "M25 active",
        "M24 history",
        "M23 history",
        "M22 history",
        "M21 history",
        "M20 history",
        "M19 history",
        "M18 history",
        "M17 history",
        "M16 history",
    ]

    partial = dict(scheduler)
    partial.pop(key)
    errors = board._active_successor_migration_errors(partial, seal, migration)
    assert "M25 active" in errors
    assert any("only partially declared" in error for error in errors)

    malformed = {**scheduler, key: None}
    with pytest.raises(
        operator.OperatorError,
        match="active M25 native-DuckDB preload successor authority is invalid",
    ):
        operator._active_source_repair_materialization(malformed)


def test_m25_source_chain_is_one_repair_then_one_nine_control_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m25_source_chain_test",
    )
    authority = materializer._expected_m25_native_duckdb_preload_authority()
    prior = authority["prior_source_head"]
    repair = authority["repair_source_commit"]
    final = "f" * 40
    final_tree = "e" * 40
    counts = {(prior, repair): 1, (repair, final): 1}
    population = {
        "source_binding": {
            "head": final,
            "tree": final_tree,
            "datasets_gitlink": authority["prior_datasets_gitlink"],
            "kit_gitlink": authority["prior_kit_gitlink"],
        }
    }

    def fake_git(root: Path, *args: str) -> str:
        if args[:3] == ("diff", "--name-status", "--no-renames"):
            before, after = args[3], args[4]
            if (before, after) == (prior, repair):
                paths = sorted(authority["source_repair_paths"])
            elif (before, after) in {(repair, final), (prior, final)}:
                paths = sorted(authority["operator_control_paths"])
            else:
                raise AssertionError(args)
            return "\n".join(f"M\t{path}" for path in paths)
        if args[:2] == ("merge-base", "--is-ancestor"):
            return ""
        if args[0] == "rev-list":
            before, after = args[2].split("..", 1)
            return str(counts[(before, after)])
        if args[0] != "rev-parse":
            raise AssertionError(args)
        ref = args[1]
        trees = {
            f"{prior}^{{tree}}": authority["prior_source_tree"],
            f"{repair}^{{tree}}": authority["repair_source_tree"],
            f"{final}^{{tree}}": final_tree,
        }
        if ref in trees:
            return str(trees[ref])
        for path, blob in authority["repair_source_blobs"].items():
            if ref == f"{repair}:{path}":
                return str(blob)
        for dependency_name, gitlink_key, tree_key in (
            ("ipfs_datasets_py", "prior_datasets_gitlink", "prior_datasets_tree"),
            ("ipfs_kit_py", "prior_kit_gitlink", "prior_kit_tree"),
        ):
            if ref in {
                f"{head}:{dependency_name}" for head in (prior, repair, final)
            }:
                return str(authority[gitlink_key])
            if root.name == dependency_name and ref == (
                f"{authority[gitlink_key]}^{{tree}}"
            ):
                return str(authority[tree_key])
        raise AssertionError((root, args))

    monkeypatch.setattr(materializer, "_git", fake_git)
    materializer._assert_m25_source_delta(REPO_ROOT, population, authority)
    counts[(repair, final)] = 2
    with pytest.raises(
        materializer.MaterializationError,
        match="one exact repair plus one nine-control commit",
    ):
        materializer._assert_m25_source_delta(REPO_ROOT, population, authority)


def test_m25_staging_and_pending_receipt_artifacts_fail_closed(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m25_staging_guard_test",
    )
    control = tmp_path / "control.duckdb"
    control.write_bytes(b"control")
    materializer._assert_m25_no_staging_or_pending(control)

    stage = tmp_path / ".m25-installing.test"
    stage.mkdir()
    with pytest.raises(
        materializer.MigrationRequired,
        match="staging or receipt temporary",
    ):
        materializer._assert_m25_no_staging_or_pending(control)
    stage.rmdir()

    pending = tmp_path / ".migration-receipt.json.test.tmp"
    pending.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        materializer.MigrationRequired,
        match="staging or receipt temporary",
    ):
        materializer._assert_m25_no_staging_or_pending(control)


def test_m25_receipt_binds_failed_start_zero_task_delta_and_rehashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m25_receipt_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m25_receipt_test",
    )
    authority = materializer._expected_m25_native_duckdb_preload_authority()
    control = tmp_path / "control.duckdb"
    coordination = tmp_path / "control.coordination.duckdb"
    control.write_bytes(b"m25-control")
    coordination.write_bytes(b"m25-coordination")
    control_sha256 = hashlib.sha256(control.read_bytes()).hexdigest()
    coordination_sha256 = hashlib.sha256(coordination.read_bytes()).hexdigest()
    population = {
        "program_definition_cid": "sha256:" + "1" * 64,
        "source_binding": {"source_binding_cid": "sha256:" + "2" * 64},
    }
    validation_digest = "sha256:" + "3" * 64
    verified = {
        "control_store_sha256": control_sha256,
        "control_store_size": control.stat().st_size,
        "coordination_store_sha256": coordination_sha256,
        "coordination_store_size": coordination.stat().st_size,
        "migration_digest": "sha256:" + "4" * 64,
        "migration_evidence_id": "sha256:" + "5" * 64,
        "plan_migration_event_id": "event:plan-m25",
        "migration_evidence_event_id": "event:evidence-m25",
        "projection_cid": "baguqeera" + "6" * 52,
        "event_watermark": 251,
        "semantic_authority_digest": authority[
            "prior_semantic_authority_digest"
        ],
        "frozen_base_authority_digest": authority[
            "prior_frozen_base_authority_digest"
        ],
        "append_surface_digest": "sha256:" + "7" * 64,
        "catalog_digest": authority["prior_catalog_digest"],
        "coordination_projection_digest": authority[
            "target_coordination_projection_digest"
        ],
        "coordination_event_count": 1_407,
    }
    monkeypatch.setattr(
        materializer,
        "_m25_source_binding_authority",
        lambda *_args, **_kwargs: authority,
    )
    receipt = materializer._expected_m25_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        {},
        verified,
        validation_digest,
    )
    assert set(receipt) == materializer._M25_RECEIPT_KEYS
    unhashed = dict(receipt)
    assert unhashed.pop("receipt_cid") == materializer._identity(unhashed)
    assert receipt["schema"] == "sawm/non-authoritative-migration-receipt@23"
    assert receipt["authoritative"] is False
    assert receipt["receipt_is_final_pair_commit_marker"] is True
    assert receipt["migration_event_watermark"] == 251
    assert receipt["maximum_persisted_generation"] == 23
    assert receipt["failed_attempted_generation"] == 24
    assert receipt["failed_attempt_allocated_generation"] is False
    assert receipt["failed_quack_start"] == authority["failed_quack_start"]
    assert receipt["failed_quack_start_cid"] == authority[
        "failed_quack_start_cid"
    ]
    for field in (
        "task_revision_changes",
        "task_status_changes",
        "coordination_semantic_changes",
        "accepted_definition_changes",
        "accepted_completion_changes",
        "implementation_provider_invocations",
        "effect_claim_changes",
        "implementation_commit_changes",
        "merge_attempt_changes",
    ):
        assert receipt[field] == 0
    assert receipt["worker_self_approval"] is False
    for name, value in operator._m25_receipt_authority_fields(
        authority, materializer
    ).items():
        assert receipt[name] == value

    receipt_path = tmp_path / "migration-receipt.json"
    receipt_path.write_bytes(materializer._canonical(receipt) + b"\n")
    monkeypatch.setattr(
        materializer,
        "_assert_m25_receipt_commit_inputs",
        lambda *_args, **_kwargs: None,
    )
    assert materializer._verify_existing_m25_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        {},
        verified,
        validation_digest,
    ) == receipt

    forged = copy.deepcopy(receipt)
    forged["worker_self_approval"] = True
    forged_unhashed = dict(forged)
    forged_unhashed.pop("receipt_cid")
    forged["receipt_cid"] = materializer._identity(forged_unhashed)
    receipt_path.write_bytes(materializer._canonical(forged) + b"\n")
    with pytest.raises(
        materializer.MigrationRequired,
        match="final pair marker differs",
    ):
        materializer._verify_existing_m25_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            {},
            verified,
            validation_digest,
        )


def test_m24_sidecar_reopen_authority_and_four_lane_bindings_are_exact() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m24_authority_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m24_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "multi_lane_sidecar_reopen_successor_materialization"
    expected = materializer._expected_m24_sidecar_reopen_authority()
    config, inventory, seal = _historical_successor_controls_at(
        key, config, inventory, seal
    )
    assert inventory is not None
    assert seal is not None
    runtime = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m24"
    store = f"{runtime}/control.duckdb"
    config["database_program"].update(
        {
            "store_id": store,
            "store_generation": "24",
            "quack_endpoint": "quack:127.0.0.1:24067",
            "event_store_path": f"{runtime}/events",
            "runtime_registry_path": f"{runtime}/registry",
            "worktree_root": f"{runtime}/worktrees",
        }
    )
    config["quack_owner"].update(
        {
            "database_path": store,
            "store_id": store,
            "state_dir": f"{runtime}/quack-owner",
            "port": 24_067,
        }
    )
    config["runtime_paths"] = {
        "root": runtime,
        "state": f"{runtime}/state",
        "worktrees": f"{runtime}/worktrees",
        "merge_queue": f"{runtime}/merge-queue",
        "logs": f"{runtime}/logs",
        "generated_runtime_artifacts_are_completion_authority": False,
    }

    assert config[key] == expected
    assert inventory[key] == expected
    assert seal[f"{key}_cid"] == materializer._identity(expected)
    assert materializer._m24_successor_configured(config) is True
    assert dict(operator._active_source_repair_materialization(config)) == expected
    assert expected["migration_revision"] == "SAWM-R2-M24"
    assert expected["target_generation"] == 24
    assert expected["target_quack_port"] == 24_067
    assert expected["target_plan_revision"] == 25
    assert expected["target_event_watermark"] == 249
    assert expected["target_coordination_event_count"] == 1_407
    assert expected["task_revision_changes"] == 2
    assert expected["task_status_changes"] == 2
    assert expected["coordination_semantic_changes"] == 2
    assert expected["accepted_completion_changes"] == 0
    assert expected["execution_sidecar_copied"] is False
    assert expected["read_replica_sidecar_copied"] is False
    assert expected["worktrees_copied"] is False
    assert config["database_program"]["store_generation"] == "24"
    assert config["database_program"]["quack_endpoint"] == "quack:127.0.0.1:24067"
    assert config["quack_owner"]["port"] == 24_067
    assert [
        (
            lane["index"],
            lane["name"],
            lane["strict_shard_remainder"],
            lane["initial_task_ids"],
        )
        for lane in config["lanes"]
    ] == [
        (0, "sawm-lane-0", 0, ["SAWM-008"]),
        (1, "sawm-lane-1", 1, ["SAWM-006", "SAWM-010"]),
        (2, "sawm-lane-2", 2, ["SAWM-015"]),
        (3, "sawm-lane-3", 3, ["SAWM-012"]),
    ]

    marker_fields = dict(
        operator._m24_receipt_authority_fields(expected, materializer)
    )
    for name in (
        "prior_lane_runtime_anchors",
        "interrupted_claim_evidence",
        "failure_receipt",
        "task_rearm",
        "lane_contract",
        "repair_source_first_commit",
        "repair_source_commit",
        "repair_source_blobs",
        "accepted_control_plane_repair",
    ):
        assert marker_fields[name] == expected[name]

    for malformed in (
        {**copy.deepcopy(config), key: None},
        copy.deepcopy(config),
    ):
        if malformed[key] is not None:
            malformed["lanes"][1]["initial_task_ids"] = ["SAWM-010"]
        with pytest.raises(
            operator.OperatorError,
            match="active M24 sidecar-reopen successor authority is invalid",
        ):
            operator._active_source_repair_materialization(malformed)


def test_m24_stopped_anchor_private_settlement_rearm_and_receipt_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m24_private_stage_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m24_private_stage_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    authority = materializer._expected_m24_sidecar_reopen_authority()
    population = materializer.build_population(REPO_ROOT)
    prior_control, prior_coordination = materializer._assert_m24_prior_anchor(
        REPO_ROOT,
        authority,
        population,
    )
    validation_digest = materializer._identity(
        {
            "schema": "sawm/m24-private-stage-validation@1",
            "source_binding_cid": population["source_binding"][
                "source_binding_cid"
            ],
        }
    )
    stage_dir = tmp_path / "m24-stage"
    stage_dir.mkdir()
    staged = materializer._stage_m24_store_pair(
        REPO_ROOT,
        stage_dir,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    verified = staged["verified"]
    assert verified["event_watermark"] == 249
    assert verified["plan_revision_changes"] == 1
    assert verified["evidence_node_changes"] == 1
    assert verified["task_revision_changes"] == 2
    assert verified["task_status_changes"] == 2
    assert verified["coordination_semantic_changes"] == 2
    assert verified["coordination_event_count"] == 1_407
    assert verified["coordination_projection_digest"] == (
        authority["target_coordination_projection_digest"]
    )
    assert verified["active_claim_count"] == 0
    assert verified["active_attempt_count"] == 0
    assert verified["active_lease_count"] == 0
    for field in (
        "goal_changes",
        "accepted_definition_changes",
        "accepted_completion_changes",
        "implementation_provider_invocations",
        "effect_claim_changes",
        "implementation_commit_changes",
        "merge_attempt_changes",
    ):
        assert verified[field] == 0
    assert not (stage_dir / "control.execution.duckdb").exists()
    assert not (stage_dir / "control.read-replica.duckdb").exists()
    assert not tuple(stage_dir.glob("*.wal"))

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    source = DatabaseTaskSource(staged["stage_control"], install_schema=False)
    try:
        statuses, revisions, _receipts = (
            operator._verify_m24_live_head_task_projection(
                source,
                population,
                materializer,
                expected_projection_cid=verified["projection_cid"],
            )
        )
    finally:
        source.close()
    assert statuses["SAWM-008"] == "retrying"
    assert revisions["SAWM-008"] == 5
    assert statuses["SAWM-015"] == "retrying"
    assert revisions["SAWM-015"] == 5

    monkeypatch.setattr(
        materializer,
        "_m24_source_binding_authority",
        lambda *_args, **_kwargs: authority,
    )
    receipt = materializer._expected_m24_migration_receipt(
        tmp_path,
        staged["stage_control"],
        staged["stage_coordination"],
        population,
        config,
        verified,
        validation_digest,
    )
    assert set(receipt) == materializer._M24_RECEIPT_KEYS
    unhashed = dict(receipt)
    assert unhashed.pop("receipt_cid") == materializer._identity(unhashed)
    assert receipt["schema"] == "sawm/non-authoritative-migration-receipt@22"
    assert receipt["migration_digest"] == verified["migration_digest"]
    assert receipt["coordination_store_sha256"] == verified[
        "coordination_store_sha256"
    ]
    assert receipt["coordination_projection_digest"] == authority[
        "target_coordination_projection_digest"
    ]
    assert receipt["task_rearm_event_ids"].keys() == {"failure", "retry"}
    assert receipt["task_rearm_receipt_cids"].keys() == {"failure", "retry"}
    for name, value in operator._m24_receipt_authority_fields(
        authority,
        materializer,
    ).items():
        if name not in {"database_path", "coordination_path"}:
            assert receipt[name] == value


def test_m24_presence_masks_historical_materializer_and_operator_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m24_presence_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m24_presence_test",
    )
    key = "multi_lane_sidecar_reopen_successor_materialization"
    cid_key = f"{key}_cid"
    expected = materializer._expected_m24_sidecar_reopen_authority()
    inventory_path = (
        tmp_path
        / "docs/architecture/semantic_addressed_world_model_inventory/"
        "prior_materialization_migration.json"
    )
    seal_path = tmp_path / "config/semantic_addressed_world_model_dependencies.seal.json"
    config_path = tmp_path / "scheduler.json"
    inventory_path.parent.mkdir(parents=True)
    seal_path.parent.mkdir(parents=True)

    def write_controls(
        scheduler: Mapping[str, object],
        migration: Mapping[str, object],
        seal: Mapping[str, object],
    ) -> None:
        config_path.write_text(json.dumps(scheduler), encoding="utf-8")
        inventory_path.write_text(json.dumps(migration), encoding="utf-8")
        seal_path.write_text(json.dumps(seal), encoding="utf-8")

    for scheduler, migration, seal in (
        ({}, {key: expected}, {}),
        ({}, {}, {cid_key: materializer._identity(expected)}),
    ):
        write_controls(scheduler, migration, seal)
        for operation in (materializer.check_materialized, materializer.materialize):
            with pytest.raises(
                materializer.MaterializationError,
                match="M24 .* only partially declared",
            ):
                operation(tmp_path, config_path)

    checked = {"selected": "m24-check"}
    materialized = {"selected": "m24-materialize"}
    monkeypatch.setattr(
        materializer,
        "_check_m24_materialized",
        lambda *_args, **_kwargs: checked,
    )
    monkeypatch.setattr(
        materializer,
        "_materialize_m24",
        lambda *_args, **_kwargs: materialized,
    )
    monkeypatch.setattr(
        materializer,
        "_check_m23_materialized",
        lambda *_args, **_kwargs: pytest.fail("M23 check must be masked"),
    )
    monkeypatch.setattr(
        materializer,
        "_materialize_m23",
        lambda *_args, **_kwargs: pytest.fail("M23 materialize must be masked"),
    )
    write_controls(
        {key: expected},
        {key: expected},
        {cid_key: materializer._identity(expected)},
    )
    assert materializer.check_materialized(tmp_path, config_path) is checked
    assert materializer.materialize(tmp_path, config_path) is materialized

    selected = {"selected": "m24-marker"}
    monkeypatch.setattr(
        operator,
        "_require_m24_final_pair_marker",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        operator,
        "_require_m23_final_pair_marker",
        lambda *_args, **_kwargs: pytest.fail("M23 marker must be masked"),
    )
    assert operator._require_active_final_pair_marker(
        {key: None, "multi_lane_successor_materialization": {}},
        expected,
        materializer,
    ) is selected


def test_m23_multi_lane_authority_scheduler_and_presence_are_exact() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m23_authority_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m23_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "multi_lane_successor_materialization"
    expected = materializer._expected_m23_multi_lane_authority()

    assert config[key] == expected
    assert inventory[key] == expected
    assert seal[f"{key}_cid"] == materializer._identity(expected)
    assert expected["schema"] == (
        "sawm/multi-lane-successor-materialization-authorization@1"
    )
    assert expected["migration_revision"] == "SAWM-R2-M23"
    assert expected["target_generation"] == 23
    assert expected["target_plan_revision"] == 24
    assert expected["target_event_watermark"] == 244
    assert expected["target_quack_port"] == 24_066
    assert expected["event_suffix_length"] == 4
    assert expected["task_revision_changes"] == 2
    assert expected["task_status_changes"] == 2
    assert expected["coordination_semantic_changes"] == 2
    assert expected["accepted_completion_changes"] == 0
    m23_config, _historical_migration, _historical_seal = (
        _historical_successor_controls_at(key, config)
    )
    runtime = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m23"
    m23_config["database_program"].update(
        {
            "quack_endpoint": "quack:127.0.0.1:24066",
            "store_id": f"{runtime}/control.duckdb",
            "store_generation": "23",
            "event_store_path": f"{runtime}/events",
            "runtime_registry_path": f"{runtime}/registry",
            "worktree_root": f"{runtime}/worktrees",
        }
    )
    m23_config["quack_owner"].update(
        {
            "database_path": f"{runtime}/control.duckdb",
            "state_dir": f"{runtime}/quack-owner",
            "port": 24_066,
            "store_id": f"{runtime}/control.duckdb",
        }
    )
    m23_config["runtime_paths"] = {
        "root": runtime,
        "state": f"{runtime}/state",
        "worktrees": f"{runtime}/worktrees",
        "merge_queue": f"{runtime}/merge-queue",
        "logs": f"{runtime}/logs",
        "generated_runtime_artifacts_are_completion_authority": False,
    }
    assert dict(operator._active_source_repair_materialization(m23_config)) == expected
    assert materializer._m23_successor_configured(config) is True
    marker_fields = dict(operator._m23_receipt_authority_fields(expected))
    for name in (
        "prior_receipt_publication_control_store_sha256",
        "prior_receipt_publication_control_store_size",
        "prior_receipt_publication_coordination_store_sha256",
        "prior_receipt_publication_coordination_store_size",
        "prior_receipt_publication_projection_cid",
        "prior_receipt_publication_semantic_authority_digest",
        "prior_receipt_publication_frozen_base_authority_digest",
        "prior_receipt_publication_append_surface_digest",
        "prior_receipt_publication_coordination_projection_digest",
        "prior_receipt_publication_coordination_event_count",
        "prior_receipt_publication_event_watermark",
        "prior_receipt_publication_catalog_digest",
    ):
        assert marker_fields[name] == expected[name]

    assert m23_config["database_program"]["store_id"] == f"{runtime}/control.duckdb"
    assert m23_config["database_program"]["store_generation"] == "23"
    assert m23_config["database_program"]["quack_endpoint"] == "quack:127.0.0.1:24066"
    assert m23_config["quack_owner"]["port"] == 24_066
    assert m23_config["runtime_paths"]["root"] == runtime
    assert m23_config["max_lanes"] == 4
    assert m23_config["strict_task_sharding"] is True
    assert m23_config["idle_lane_work_stealing"] == ""
    assert [lane["index"] for lane in m23_config["lanes"]] == [0, 1, 2, 3]
    assert [lane["strict_shard_remainder"] for lane in m23_config["lanes"]] == [0, 1, 2, 3]
    assert [lane["name"] for lane in m23_config["lanes"]] == [
        "sawm-lane-0", "sawm-lane-1", "sawm-lane-2", "sawm-lane-3"
    ]
    assert m23_config["provider"]["max_concurrency"] >= 4

    for malformed in (
        {**copy.deepcopy(m23_config), key: None},
        copy.deepcopy(m23_config),
    ):
        if malformed[key] is not None:
            malformed["lanes"][0]["index"] = False
        with pytest.raises(
            operator.OperatorError,
            match="active M23 multi-lane successor authority is invalid",
        ):
            operator._active_source_repair_materialization(malformed)


def test_m23_stopped_anchor_private_settlement_rearm_and_receipt_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m23_private_stage_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m23_private_stage_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    authority = materializer._expected_m23_multi_lane_authority()
    population = materializer.build_population(REPO_ROOT)
    anchor_keys = (
        "prior_store_id",
        "prior_coordination_store_id",
        "prior_read_replica_path",
        "prior_execution_sidecar_path",
        "prior_stopped_status_projection_path",
        "prior_supervisor_status_path",
        "prior_migration_receipt_path",
    )
    anchors = tuple(REPO_ROOT / str(authority[key]) for key in anchor_keys)
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)
        for path in anchors
    }
    prior_control, prior_coordination = materializer._assert_m23_prior_anchor(
        REPO_ROOT, authority, population
    )
    assert before == {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)
        for path in anchors
    }

    validation_digest = materializer._identity(
        {
            "schema": "sawm/m23-private-stage-validation@1",
            "source_binding_cid": population["source_binding"]["source_binding_cid"],
        }
    )
    stage_dir = tmp_path / "m23-stage"
    stage_dir.mkdir()
    staged = materializer._stage_m23_store_pair(
        REPO_ROOT,
        stage_dir,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    verified = staged["verified"]
    assert verified["event_watermark"] == 244
    assert verified["plan_revision_changes"] == 1
    assert verified["evidence_node_changes"] == 1
    assert verified["task_revision_changes"] == 2
    assert verified["task_status_changes"] == 2
    assert verified["coordination_semantic_changes"] == 2
    assert verified["coordination_event_count"] == 1357
    assert verified["active_claim_count"] == 0
    assert verified["active_attempt_count"] == 0
    assert verified["active_lease_count"] == 0
    for field in (
        "goal_changes",
        "accepted_definition_changes",
        "accepted_completion_changes",
        "implementation_provider_invocations",
        "effect_claim_changes",
        "implementation_commit_changes",
        "merge_attempt_changes",
    ):
        assert verified[field] == 0
    assert not (stage_dir / "control.execution.duckdb").exists()
    assert not (stage_dir / "control.read-replica.duckdb").exists()
    assert not tuple(stage_dir.glob("*.wal"))

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    source = DatabaseTaskSource(staged["stage_control"], install_schema=False)
    try:
        statuses, revisions, _receipts = operator._verify_m23_live_head_task_projection(
            source,
            population,
            materializer,
            expected_projection_cid=verified["projection_cid"],
        )
    finally:
        source.close()
    assert statuses["SAWM-007"] == "completed"
    assert statuses["SAWM-011"] == "completed"
    assert statuses["SAWM-015"] == "retrying"
    assert revisions["SAWM-015"] == 5

    monkeypatch.setattr(
        materializer,
        "_m23_source_binding_authority",
        lambda *_args, **_kwargs: authority,
    )
    receipt = materializer._expected_m23_migration_receipt(
        tmp_path,
        staged["stage_control"],
        staged["stage_coordination"],
        population,
        config,
        verified,
        validation_digest,
    )
    assert set(receipt) == materializer._M23_RECEIPT_KEYS
    unhashed = dict(receipt)
    assert unhashed.pop("receipt_cid") == materializer._identity(unhashed)
    assert receipt["schema"] == "sawm/non-authoritative-migration-receipt@21"
    assert receipt["task_rearm_event_ids"].keys() == {"failure", "retry"}
    assert receipt["task_rearm_receipt_cids"].keys() == {"failure", "retry"}
    assert receipt["execution_sidecar_copied"] is False
    assert receipt["read_replica_sidecar_copied"] is False


def test_m23_presence_masks_historical_materializer_and_operator_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m23_presence_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m23_presence_test",
    )
    key = "multi_lane_successor_materialization"
    cid_key = f"{key}_cid"
    expected = materializer._expected_m23_multi_lane_authority()
    inventory_path = (
        tmp_path
        / "docs/architecture/semantic_addressed_world_model_inventory/"
        "prior_materialization_migration.json"
    )
    seal_path = tmp_path / "config/semantic_addressed_world_model_dependencies.seal.json"
    config_path = tmp_path / "scheduler.json"
    inventory_path.parent.mkdir(parents=True)
    seal_path.parent.mkdir(parents=True)

    def write_controls(
        scheduler: Mapping[str, object],
        migration: Mapping[str, object],
        seal: Mapping[str, object],
    ) -> None:
        config_path.write_text(json.dumps(scheduler), encoding="utf-8")
        inventory_path.write_text(json.dumps(migration), encoding="utf-8")
        seal_path.write_text(json.dumps(seal), encoding="utf-8")

    for scheduler, migration, seal in (
        ({}, {key: expected}, {}),
        ({}, {}, {cid_key: materializer._identity(expected)}),
    ):
        write_controls(scheduler, migration, seal)
        for operation in (materializer.check_materialized, materializer.materialize):
            with pytest.raises(
                materializer.MaterializationError,
                match="M23 .* only partially declared",
            ):
                operation(tmp_path, config_path)

    checked = {"selected": "m23-check"}
    materialized = {"selected": "m23-materialize"}
    monkeypatch.setattr(
        materializer, "_check_m23_materialized", lambda *_args, **_kwargs: checked
    )
    monkeypatch.setattr(
        materializer, "_materialize_m23", lambda *_args, **_kwargs: materialized
    )
    monkeypatch.setattr(
        materializer,
        "_check_m22_materialized",
        lambda *_args, **_kwargs: pytest.fail("M22 check must be masked"),
    )
    monkeypatch.setattr(
        materializer,
        "_materialize_m22",
        lambda *_args, **_kwargs: pytest.fail("M22 materialize must be masked"),
    )
    write_controls(
        {key: expected},
        {key: expected},
        {cid_key: materializer._identity(expected)},
    )
    assert materializer.check_materialized(tmp_path, config_path) is checked
    assert materializer.materialize(tmp_path, config_path) is materialized

    selected = {"selected": "m23-marker"}
    monkeypatch.setattr(
        operator,
        "_require_m23_final_pair_marker",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        operator,
        "_require_m22_final_pair_marker",
        lambda *_args, **_kwargs: pytest.fail("M22 marker must be masked"),
    )
    assert operator._require_active_final_pair_marker(
        {key: None, "live_preflight_receipt_compatibility_successor_materialization": {}},
        expected,
        materializer,
    ) is selected


def test_m22_live_preflight_receipt_compatibility_authority_is_exact_and_presence_first() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m22_authority_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m22_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "live_preflight_receipt_compatibility_successor_materialization"
    expected = (
        materializer._expected_m22_live_preflight_receipt_compatibility_authority()
    )
    exact_control_paths = {
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        "config/semantic_addressed_world_model_dependencies.seal.json",
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
        (
            "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ),
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "scripts/validate_semantic_addressed_world_model_board.py",
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
    }

    assert config[key] == expected
    assert inventory[key] == expected
    assert seal[f"{key}_cid"] == materializer._identity(expected)
    assert expected["schema"] == (
        "sawm/live-preflight-receipt-compatibility-successor-"
        "materialization-authorization@1"
    )
    assert expected["target_runtime_root"].endswith("run-r2-m22")
    assert expected["target_generation"] == 22
    assert expected["target_plan_revision"] == 23
    assert expected["target_event_watermark"] == 233
    assert expected["target_quack_port"] == 24_065
    assert expected["event_suffix_length"] == 2
    assert expected["plan_revision_changes"] == 1
    assert expected["task_revision_changes"] == 0
    assert expected["task_status_changes"] == 0
    assert expected["accepted_definition_changes"] == 0
    assert expected["accepted_completion_changes"] == 0
    blockers = expected["observed_launch_blockers"]
    assert blockers["direct_configured_scheduler"] == {
        "schema": "sawm/observed-control-plane-launch-blocker@1",
        "route": "configured_board_scheduler_direct",
        "terminal": "sealed_worker_exit",
        "exit_code": 78,
        "typed_reason": "quack_auth_token_unavailable",
        "missing_secret_handle": "SAWM_QUACK_TOKEN",
        "task_claim_count": 0,
        "attempt_count": 0,
        "provider_invocation_count": 0,
        "effect_claim_count": 0,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_completion_changes": 0,
        "secret_persisted": False,
    }
    assert blockers["operator_facade_live_preflight"]["exact_error"] == (
        "M21-head accepted completion differs: SAWM-001"
    )
    assert blockers["operator_facade_live_preflight"][
        "legacy_top_level_worker_self_approval_field_present"
    ] is False
    assert blockers["operator_facade_live_preflight"][
        "separately_pinned_operator_attestation"
    ] is True
    assert blockers["operator_facade_live_preflight"][
        "attested_worker_self_approval"
    ] is False
    assert all(
        blocker["secret_persisted"] is False
        and blocker["task_claim_count"] == 0
        and blocker["attempt_count"] == 0
        and blocker["provider_invocation_count"] == 0
        and blocker["effect_claim_count"] == 0
        and blocker["task_revision_changes"] == 0
        and blocker["task_status_changes"] == 0
        and blocker["accepted_completion_changes"] == 0
        for blocker in blockers.values()
    )
    assert set(expected["operator_control_paths"]) == exact_control_paths
    assert set(expected["bounded_control_plane_repair_paths"]) == (
        exact_control_paths
    )
    historical_config, _historical_migration, _historical_seal = (
        _historical_successor_controls_at(key, config)
    )
    assert materializer._m22_successor_configured(historical_config) is True
    assert dict(operator._active_source_repair_materialization(historical_config)) == expected
    assert operator._successor_materialization_configured(historical_config) is True

    # M22 key presence masks every valid historical successor. A malformed
    # M22 declaration must fail closed instead of falling through to M21.
    malformed = copy.deepcopy(historical_config)
    malformed[key] = None
    with pytest.raises(
        materializer.MaterializationError,
        match="M22 live-preflight receipt compatibility authority is invalid",
    ):
        materializer._m22_successor_configured(malformed)
    with pytest.raises(
        operator.OperatorError,
        match=(
            "active M22 live-preflight receipt compatibility successor "
            "authority is invalid"
        ),
    ):
        operator._active_source_repair_materialization(malformed)


def test_m22_scheduler_authority_is_preserved_under_m27_runtime() -> None:
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    authority = config[
        "live_preflight_receipt_compatibility_successor_materialization"
    ]
    runtime_root = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m22"
    control = f"{runtime_root}/control.duckdb"
    coordination = f"{runtime_root}/control.coordination.duckdb"

    assert authority["target_runtime_root"] == runtime_root
    assert authority["target_store_id"] == control
    assert authority["target_coordination_store_id"] == coordination
    assert authority["target_generation"] == 22
    assert authority["target_plan_revision"] == 23
    assert authority["target_event_watermark"] == 233
    assert authority["target_quack_port"] == 24_065
    current_runtime = (
        "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
    )
    assert config["database_program"]["store_id"] == (
        f"{current_runtime}/control.duckdb"
    )
    # M48 advances the owner to generation 35 without changing the M27
    # runtime namespace.
    assert config["database_program"]["store_generation"] == "35"
    assert config["database_program"]["quack_endpoint"] == (
        "quack:127.0.0.1:24070"
    )
    assert config["quack_owner"]["port"] == 24_070
    assert config["runtime_paths"]["root"] == current_runtime


def test_m22_legacy_completion_compatibility_is_exact_and_fail_closed() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m22_legacy_completion_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m22_legacy_completion_test",
    )
    import duckdb

    stopped_m21 = (
        REPO_ROOT
        / "data/agent_supervisor/semantic_addressed_world_model/run-r2-m21/"
        "control.duckdb"
    )
    connection = duckdb.connect(str(stopped_m21), read_only=True)
    try:
        rows = connection.execute(
            "SELECT task_cid, body_json FROM tasks ORDER BY task_cid"
        ).fetchall()
    finally:
        connection.close()
    accepted = {}
    for task_cid, body_json in rows:
        body = json.loads(str(body_json))
        task_alias = str(
            body.get("task_id")
            or (body.get("definition") or {}).get("task_id")
            or ""
        )
        if task_alias in {
            "SAWM-001",
            "SAWM-002",
            "SAWM-003",
            "SAWM-004",
            "SAWM-005",
        }:
            accepted[task_alias] = (
                str(task_cid),
                body.get("completion_receipt"),
                body.get("operational_validation_revision"),
            )

    completion_identities = {
        "SAWM-001": "sha256:c11d9255c9f70bfac083167cecf140036376196d89c9b3bec9ed42ae341a0ffc",
        "SAWM-002": "sha256:cdce267a0a42216625ee191214f41f5baa7c8dcb055d91f3754cf85825da5b20",
        "SAWM-003": "sha256:57db209a834b82e8d04e32ac51b23b0e3e2e42fab910268d253f010536338f97",
        "SAWM-004": "sha256:8ce1d4c03347c06896d0986985bf36d2bb6d9a4c40930642893718a3aa8a6e59",
        "SAWM-005": "sha256:4ae316b48b95699618da1b2ef7c15b9574c4b88e436fc2393dae14448697ee08",
    }
    operational_receipt_cids = {
        "SAWM-001": "sha256:6e5463ca065664a3e3735f5320f6527a3b85d85cda769e9a3a23c30635f14265",
        "SAWM-002": "sha256:3e931c6a35d1a4a0ee03d306d2ff657fb9bff9cb622b281de695e06e77d12ad2",
        "SAWM-003": "sha256:8462595484a51a67bd5b5f18d3f6dcf4a18079f0762417259b170eba541c44a4",
        "SAWM-004": "sha256:b487fb734f074d958c5bf2026c0016dca835b063cebc5531b80a60218ad3f1cc",
        "SAWM-005": "sha256:041007a8bf7905957ba38550919a860377ec0b54cae051d573b778598935cd61",
    }
    assert set(accepted) == set(completion_identities)

    for task_alias, expected_completion_cid in completion_identities.items():
        task_cid, completion, operational = accepted[task_alias]
        assert isinstance(completion, Mapping)
        assert "worker_self_approval" not in completion
        assert materializer._identity(completion) == expected_completion_cid
        assert isinstance(operational, Mapping)
        assert operational["worker_self_approval"] is False
        assert operational["receipt_cid"] == operational_receipt_cids[task_alias]
        unhashed_operational = dict(operational)
        assert materializer._identity(
            {
                key: value
                for key, value in unhashed_operational.items()
                if key != "receipt_cid"
            }
        ) == operational["receipt_cid"]
        assert operator._m22_legacy_completion_is_compatible(
            materializer,
            task_alias=task_alias,
            task_cid=task_cid,
            completion_receipt=completion,
            operational_validation_revision=operational,
        )

        # Compatibility is deliberately narrower than normal defaulting:
        # legacy absence is accepted, but any explicit top-level value is not.
        for explicit in (False, True, "false", 0, None):
            explicit_completion = copy.deepcopy(completion)
            explicit_completion["worker_self_approval"] = explicit
            assert not operator._m22_legacy_completion_is_compatible(
                materializer,
                task_alias=task_alias,
                task_cid=task_cid,
                completion_receipt=explicit_completion,
                operational_validation_revision=operational,
            )

        tampered_completion = copy.deepcopy(completion)
        tampered_completion["operation"] = "database_complete_tampered"
        assert not operator._m22_legacy_completion_is_compatible(
            materializer,
            task_alias=task_alias,
            task_cid=task_cid,
            completion_receipt=tampered_completion,
            operational_validation_revision=operational,
        )
        assert not operator._m22_legacy_completion_is_compatible(
            materializer,
            task_alias=task_alias,
            task_cid=task_cid,
            completion_receipt=completion,
            operational_validation_revision=None,
        )
        for field, invalid in (
            ("worker_self_approval", None),
            ("worker_self_approval", True),
            ("receipt_cid", "sha256:" + "0" * 64),
            ("task_cid", "sha256:" + "0" * 64),
        ):
            tampered_operational = copy.deepcopy(operational)
            if invalid is None:
                tampered_operational.pop(field)
            else:
                tampered_operational[field] = invalid
            assert not operator._m22_legacy_completion_is_compatible(
                materializer,
                task_alias=task_alias,
                task_cid=task_cid,
                completion_receipt=completion,
                operational_validation_revision=tampered_operational,
            )


def test_m22_stopped_anchor_private_append_and_receipt_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m22_private_stage_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m22_private_stage_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    authority = (
        materializer._expected_m22_live_preflight_receipt_compatibility_authority()
    )
    population = materializer.build_population(REPO_ROOT)
    anchor_keys = (
        "prior_store_id",
        "prior_coordination_store_id",
        "prior_read_replica_path",
        "prior_execution_sidecar_path",
        "prior_stopped_status_projection_path",
        "prior_migration_receipt_path",
    )
    anchors = tuple(REPO_ROOT / str(authority[key]) for key in anchor_keys)
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)
        for path in anchors
    }
    prior_control, prior_coordination = materializer._assert_m22_prior_anchor(
        REPO_ROOT,
        authority,
        population,
    )
    assert before == {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)
        for path in anchors
    }

    validation_digest = materializer._identity(
        {
            "schema": "sawm/m22-private-stage-validation@1",
            "source_binding_cid": population["source_binding"][
                "source_binding_cid"
            ],
        }
    )
    stage_dir = tmp_path / "m22-stage"
    stage_dir.mkdir()
    staged = materializer._stage_m22_store_pair(
        REPO_ROOT,
        stage_dir,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    verified = staged["verified"]
    assert verified["event_watermark"] == 233
    assert verified["plan_revision_changes"] == 1
    assert verified["evidence_node_changes"] == 1
    for field in (
        "task_revision_changes",
        "task_status_changes",
        "goal_changes",
        "accepted_definition_changes",
        "accepted_completion_changes",
        "coordination_semantic_changes",
    ):
        assert verified[field] == 0
    assert materializer._store_sha256(staged["stage_coordination"]) == (
        authority["prior_coordination_store_sha256"]
    )
    assert not (stage_dir / "control.execution.duckdb").exists()
    assert not (stage_dir / "control.read-replica.duckdb").exists()
    assert not tuple(stage_dir.glob("*.wal"))

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    source = DatabaseTaskSource(staged["stage_control"], install_schema=False)
    try:
        statuses, revisions, receipts = (
            operator._verify_m22_live_head_task_projection(
                source,
                population,
                materializer,
                expected_projection_cid=verified["projection_cid"],
            )
        )
    finally:
        source.close()
    assert statuses["SAWM-001"] == "completed"
    assert statuses["SAWM-007"] == "retrying"
    assert revisions["SAWM-001"] == 18
    assert {
        alias: receipts[alias]
        for alias in authority["legacy_operational_validation_receipt_cids"]
    } == authority["legacy_operational_validation_receipt_cids"]

    monkeypatch.setattr(
        materializer,
        "_m22_source_binding_authority",
        lambda *_args, **_kwargs: authority,
    )
    receipt = materializer._expected_m22_migration_receipt(
        tmp_path,
        staged["stage_control"],
        staged["stage_coordination"],
        population,
        config,
        verified,
        validation_digest,
    )
    assert set(receipt) == materializer._M22_RECEIPT_KEYS
    unhashed = dict(receipt)
    claimed = unhashed.pop("receipt_cid")
    assert claimed == materializer._identity(unhashed)
    assert receipt["schema"] == "sawm/non-authoritative-migration-receipt@20"
    assert receipt["legacy_completion_worker_field_required_absent"] is True
    assert receipt["observed_launch_blockers"] == authority[
        "observed_launch_blockers"
    ]
    assert receipt["execution_sidecar_copied"] is False
    assert receipt["read_replica_sidecar_copied"] is False


def test_m22_final_marker_authority_and_dispatch_are_presence_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m22_marker_dispatch_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m22_marker_dispatch_test",
    )
    authority = (
        materializer._expected_m22_live_preflight_receipt_compatibility_authority()
    )
    marker = dict(operator._m22_receipt_authority_fields(authority))
    assert operator._m22_receipt_has_exact_authority_fields(marker, authority)
    for field, invalid in (
        ("legacy_completion_worker_field_required_absent", False),
        ("prior_execution_sidecar_sha256", "0" * 64),
        ("prior_stopped_status_projection_sha256", "0" * 64),
        ("target_generation", 21),
        ("target_plan_revision", 22),
        ("target_event_watermark", 231),
        ("target_quack_port", 24_064),
        ("worker_self_approval", True),
    ):
        tampered = copy.deepcopy(marker)
        tampered[field] = invalid
        assert not operator._m22_receipt_has_exact_authority_fields(
            tampered,
            authority,
        )

    selected = {"selected": "m22"}
    monkeypatch.setattr(
        operator,
        "_require_m22_final_pair_marker",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        operator,
        "_require_m21_final_pair_marker",
        lambda *_args, **_kwargs: pytest.fail("M21 dispatch must be masked"),
    )
    config = {
        "live_preflight_receipt_compatibility_successor_materialization": None,
        "generation_realization_successor_materialization": {},
    }
    assert operator._require_active_final_pair_marker(
        config,
        authority,
        materializer,
    ) is selected


def test_m22_materializer_any_surface_presence_masks_m21(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m22_any_surface_test",
    )
    key = "live_preflight_receipt_compatibility_successor_materialization"
    cid_key = f"{key}_cid"
    expected = (
        materializer._expected_m22_live_preflight_receipt_compatibility_authority()
    )
    inventory_path = (
        tmp_path
        / "docs/architecture/semantic_addressed_world_model_inventory/"
        "prior_materialization_migration.json"
    )
    seal_path = (
        tmp_path / "config/semantic_addressed_world_model_dependencies.seal.json"
    )
    config_path = tmp_path / "scheduler.json"
    inventory_path.parent.mkdir(parents=True)
    seal_path.parent.mkdir(parents=True)

    def write_controls(
        scheduler: Mapping[str, object],
        migration: Mapping[str, object],
        seal: Mapping[str, object],
    ) -> None:
        config_path.write_text(json.dumps(scheduler), encoding="utf-8")
        inventory_path.write_text(json.dumps(migration), encoding="utf-8")
        seal_path.write_text(json.dumps(seal), encoding="utf-8")

    write_controls({}, {key: expected}, {})
    for operation in (materializer.check_materialized, materializer.materialize):
        with pytest.raises(
            materializer.MaterializationError,
            match="M22 .* only partially declared",
        ):
            operation(tmp_path, config_path)

    write_controls({}, {}, {cid_key: materializer._identity(expected)})
    for operation in (materializer.check_materialized, materializer.materialize):
        with pytest.raises(
            materializer.MaterializationError,
            match="M22 .* only partially declared",
        ):
            operation(tmp_path, config_path)

    write_controls(
        {key: None},
        {key: expected},
        {cid_key: materializer._identity(expected)},
    )
    with pytest.raises(
        materializer.MaterializationError,
        match="M22 .* differs across controls",
    ):
        materializer.check_materialized(tmp_path, config_path)

    checked = {"selected": "m22-check"}
    materialized = {"selected": "m22-materialize"}
    monkeypatch.setattr(
        materializer,
        "_check_m22_materialized",
        lambda *_args, **_kwargs: checked,
    )
    monkeypatch.setattr(
        materializer,
        "_materialize_m22",
        lambda *_args, **_kwargs: materialized,
    )
    monkeypatch.setattr(
        materializer,
        "_check_m21_materialized",
        lambda *_args, **_kwargs: pytest.fail("M21 check must be masked"),
    )
    monkeypatch.setattr(
        materializer,
        "_materialize_m21",
        lambda *_args, **_kwargs: pytest.fail("M21 materialize must be masked"),
    )
    write_controls(
        {key: expected},
        {key: expected},
        {cid_key: materializer._identity(expected)},
    )
    assert materializer.check_materialized(tmp_path, config_path) is checked
    assert materializer.materialize(tmp_path, config_path) is materialized


def test_m17_source_binding_authority_is_presence_first_and_exact() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m17_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "source_binding_successor_materialization"
    authority = materializer._expected_m17_source_binding_authority()
    assert config[key] == inventory[key] == authority
    assert seal[f"{key}_cid"] == materializer._identity(authority)
    assert materializer._m17_successor_configured(config) is True
    assert authority["prior_control_store_sha256"] == (
        "11b837c173263c18f24e3d382ae1234771edc5e9ce03f56c8747be272fb8ad06"
    )
    assert authority["prior_frozen_base_authority_digest"] == (
        "sha256:24ace3d6006244240b71822a27a85a25ffe44dbb32d700d427479289d31127b2"
    )
    assert authority["target_generation"] == 18
    assert authority["target_quack_port"] == 24_060
    assert authority["target_plan_revision"] == 18
    assert authority["target_event_watermark"] == 211
    assert authority["task_revision_changes"] == 0
    assert authority["coordination_semantic_changes"] == 0
    malformed = dict(config)
    malformed[key] = None
    with pytest.raises(
        materializer.MaterializationError,
        match="M17 source-binding authority is invalid",
    ):
        materializer._m17_successor_configured(malformed)


def test_m19_live_catalog_inventory_authority_is_presence_first_and_exact() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m19_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "live_catalog_inventory_successor_materialization"
    config, inventory, seal = _historical_successor_controls_at(
        key, config, inventory, seal
    )
    assert inventory is not None
    assert seal is not None
    authority = materializer._expected_m19_live_catalog_inventory_authority()
    assert config[key] == inventory[key] == authority
    assert seal[f"{key}_cid"] == materializer._identity(authority)
    assert materializer._m19_successor_configured(config) is True
    assert materializer._m18_successor_configured(config) is True

    present = dict(config)
    assert materializer._m19_successor_configured(present) is True
    assert authority["schema"] == (
        "sawm/live-quack-catalog-inventory-repair-authorization@1"
    )
    assert authority["migration_revision"] == "SAWM-R2-M19"
    assert authority["prior_control_store_sha256"] == (
        "2050e7a0869590c7744b42e08fa2333326690f5176f9414a429ac9ec44b272bc"
    )
    assert authority["prior_frozen_base_authority_digest"] == (
        "sha256:b7e832646a9e8014f64f61c86530476d219ef10c25a467086cd73638cbbd034b"
    )
    assert authority["target_generation"] == 20
    assert authority["target_quack_port"] == 24_062
    assert authority["target_plan_revision"] == 20
    assert authority["target_event_watermark"] == 227
    assert authority["task_revision_changes"] == 0
    assert authority["task_status_changes"] == 0
    assert authority["coordination_semantic_changes"] == 0
    assert "target_projection_cid" not in authority
    assert authority["accepted_source_repair"]["blob_oids"] == {
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py": (
            "d00dc3bbdf813db1fcaabd57be7abf0436c5d995"
        ),
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py": (
            "6d0c0abae674ffd48b927d66b9e528cb3b8082f6"
        ),
    }
    malformed = dict(config)
    malformed[key] = None
    with pytest.raises(
        materializer.MaterializationError,
        match="M19 live-catalog-inventory authority is invalid",
    ):
        materializer._m19_successor_configured(malformed)

    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m19_presence_test",
    )
    assert dict(operator._active_source_repair_materialization(config)) == authority
    assert dict(operator._active_source_repair_materialization(present)) == authority
    with pytest.raises(
        operator.OperatorError,
        match="active M19 live-catalog-inventory successor authority is invalid",
    ):
        operator._active_source_repair_materialization(malformed)

    dependency_validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_validator_m19_presence_test",
    )
    dependency_errors = (
        dependency_validator._m19_live_catalog_inventory_successor_errors(
            malformed,
            seal,
            inventory,
            root=REPO_ROOT,
            require_active_runtime=False,
        )
    )
    assert "M19 source-binding authority differs across controls" in (
        dependency_errors
    )


def test_m20_test_isolation_authority_is_presence_first_and_exact() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m20_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "test_isolation_successor_materialization"
    config, inventory, seal = _historical_successor_controls_at(
        key, config, inventory, seal
    )
    assert inventory is not None
    assert seal is not None
    authority = materializer._expected_m20_test_isolation_authority()
    assert config[key] == inventory[key] == authority
    assert seal[f"{key}_cid"] == materializer._identity(authority)
    assert seal[f"{key}_cid"] == (
        "sha256:24d19d647e4806dda56a8d76f8eaf60bb1329249e1ca340e5793a4181aee4a7c"
    )
    assert materializer._m20_successor_configured(config) is True
    assert authority["prior_source_head"] == (
        "d5275b900cd223643658afad19c643506d32a748"
    )
    assert authority["prior_source_binding_cid"] == (
        "sha256:79b9fcb269ad29dd30d77084afeca33eab8b9cf8164a548748bf69a9908b298f"
    )
    assert authority["prior_validation_digest"] == (
        "sha256:60cdc6646b46d52d5102ff7c82595fdaa95c0c4e632221e28fac9d126adc2873"
    )
    assert authority["prior_event_prefix_sha256"] == (
        "45cccf8ea81087e168110d3e2abfd85c79656863519f632fcf0795a1d0208c80"
    )
    assert authority["repair_source_commit"] == (
        "3e926a247acd5654887b276afa3d00b896c08a2d"
    )
    assert authority["accepted_source_repair"]["blob_oids"] == {
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py": (
            "fea8d6784b697ce4430460446b42262577d092a5"
        )
    }
    assert authority["target_generation"] == 21
    assert authority["target_quack_port"] == 24_063
    assert authority["target_plan_revision"] == 21
    assert authority["target_event_watermark"] == 229
    assert authority["task_revision_changes"] == 0
    assert authority["task_status_changes"] == 0
    assert authority["coordination_semantic_changes"] == 0
    assert "target_projection_cid" not in authority
    assert set(materializer._m20_migration_body(
        {"source_binding": {"source_binding_cid": "sha256:" + "a" * 64}},
        config,
        "sha256:" + "b" * 64,
    )["changes"]) >= {
        "goal_changes",
        "effect_claim_changes",
        "implementation_commit_changes",
        "merge_attempt_changes",
    }
    malformed = dict(config)
    malformed[key] = None
    with pytest.raises(
        materializer.MaterializationError,
        match="M20 test-isolation authority is invalid",
    ):
        materializer._m20_successor_configured(malformed)

    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m20_presence_test",
    )
    assert dict(operator._active_source_repair_materialization(config)) == authority
    with pytest.raises(
        operator.OperatorError,
        match="active M20 test-isolation successor authority is invalid",
    ):
        operator._active_source_repair_materialization(malformed)


def test_m20_only_presence_routes_operator_and_materializer_cli_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m20_only_presence_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m20_only_presence_test",
    )
    authority = materializer._expected_m20_test_isolation_authority()
    m19 = materializer._expected_m19_live_catalog_inventory_authority()
    m20_only = {"test_isolation_successor_materialization": authority}
    m19_only = {"live_catalog_inventory_successor_materialization": m19}
    assert operator._successor_materialization_configured(m20_only) is True
    assert operator._successor_materialization_configured(m19_only) is True
    assert dict(operator._active_source_repair_materialization(m20_only)) == (
        authority
    )
    assert dict(operator._active_source_repair_materialization(m19_only)) == m19

    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(materializer, "build_population", lambda _root: {})
    monkeypatch.setattr(materializer, "_load_json", lambda _path: m20_only)

    def checked(root: Path, config_path: Path) -> dict[str, object]:
        calls.append((root, config_path))
        return {"valid": True, "action": "checked-m20-only"}

    monkeypatch.setattr(materializer, "check_materialized", checked)
    assert materializer.main(
        [
            "check",
            "--repo-root",
            str(tmp_path),
            "--config",
            "m20-only.json",
        ]
    ) == 0
    assert calls == [(tmp_path.resolve(), Path("m20-only.json"))]
    assert json.loads(capsys.readouterr().out) == {
        "action": "checked-m20-only",
        "valid": True,
    }


def test_m20_validator_selection_preserves_m19_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    config, inventory, seal = _historical_successor_controls_at(
        "test_isolation_successor_materialization", config, inventory, seal
    )
    assert inventory is not None
    assert seal is not None
    board_validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_validator_m20_presence_test",
    )
    for name, result in (
        ("_m20_migration_errors", ["M20 active"]),
        ("_m19_migration_errors", ["M19 history"]),
        ("_m18_migration_errors", ["M18 history"]),
        ("_m17_migration_errors", ["M17 history"]),
        ("_m16_migration_errors", ["M16 history"]),
    ):
        monkeypatch.setattr(
            board_validator,
            name,
            lambda *_args, _result=result, **_kwargs: _result,
        )
    assert board_validator._active_successor_migration_errors(
        config, seal, inventory
    ) == [
        "M20 active",
        "M19 history",
        "M18 history",
        "M17 history",
        "M16 history",
    ]

    partial = dict(config)
    partial.pop("test_isolation_successor_materialization")
    errors = board_validator._active_successor_migration_errors(
        partial, seal, inventory
    )
    assert "M20 active" in errors
    assert any("only partially declared" in error for error in errors)


def test_m20_private_stage_is_an_exact_two_event_append(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m20_private_stage_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._expected_m20_test_isolation_authority()
    validation_digest = materializer._identity(
        {
            "schema": "sawm/m20-private-stage-validation@1",
            "source_binding_cid": population["source_binding"][
                "source_binding_cid"
            ],
        }
    )
    assert validation_digest != authority["prior_validation_digest"]
    assert (
        population["source_binding"]["source_binding_cid"]
        != authority["prior_source_binding_cid"]
    )
    prior_control = REPO_ROOT / authority["prior_store_id"]
    prior_coordination = REPO_ROOT / authority["prior_coordination_store_id"]
    prior_hashes = {
        "control": materializer._stable_regular_sha256(
            prior_control,
            root=REPO_ROOT,
            noun="M20 test predecessor control",
            required_link_count=1,
        ),
        "coordination": materializer._stable_regular_sha256(
            prior_coordination,
            root=REPO_ROOT,
            noun="M20 test predecessor coordination",
            required_link_count=1,
        ),
    }
    prior_entries = tuple(sorted(path.name for path in prior_control.parent.iterdir()))
    stage_dir = tmp_path / "private-stage"
    stage_dir.mkdir()
    staged = materializer._stage_m20_store_pair(
        REPO_ROOT,
        stage_dir,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    verified = staged["verified"]
    assert materializer._stable_regular_sha256(
        prior_control,
        root=REPO_ROOT,
        noun="M20 test predecessor control",
        required_link_count=1,
    ) == prior_hashes["control"]
    assert materializer._stable_regular_sha256(
        prior_coordination,
        root=REPO_ROOT,
        noun="M20 test predecessor coordination",
        required_link_count=1,
    ) == prior_hashes["coordination"]
    assert tuple(sorted(path.name for path in prior_control.parent.iterdir())) == (
        prior_entries
    )
    assert materializer._store_sha256(staged["stage_coordination"]) == (
        authority["prior_coordination_store_sha256"]
    )
    assert verified["event_watermark"] == 229
    assert verified["plan_revision_changes"] == 1
    assert verified["evidence_node_changes"] == 1
    for field in (
        "task_revision_changes",
        "task_status_changes",
        "goal_changes",
        "accepted_definition_changes",
        "accepted_completion_changes",
        "coordination_semantic_changes",
    ):
        assert verified[field] == 0
    assert verified["semantic_authority_digest"] == (
        authority["prior_semantic_authority_digest"]
    )
    assert verified["frozen_base_authority_digest"] == (
        authority["prior_frozen_base_authority_digest"]
    )
    assert not tuple(stage_dir.glob("*.wal"))

    import duckdb

    connection = duckdb.connect(str(staged["stage_control"]), read_only=True)
    try:
        assert connection.execute(
            "SELECT MIN(revision), MAX(revision), COUNT(*) FROM plan_revisions"
        ).fetchone() == (1, 21, 21)
        evidence_body = json.loads(
            connection.execute(
                "SELECT body_json FROM evidence_nodes "
                "WHERE evidence_kind='operator_control_plane_source_migration' "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()[0]
        )
        suffix = connection.execute(
            "SELECT global_sequence, event_type FROM domain_events "
            "WHERE global_sequence > 227 ORDER BY global_sequence"
        ).fetchall()
        assert materializer._event_prefix_digest(connection, 227) == (
            authority["prior_event_prefix_sha256"],
            227,
        )
    finally:
        connection.close()
    assert evidence_body["validation_digest"] == validation_digest
    assert evidence_body["preserved_m19_acceptance"]["receipt_cid"] == (
        authority["prior_migration_receipt_cid"]
    )
    assert suffix == [
        (228, "intent.plan_revision_appended"),
        (229, "intent.evidence_recorded"),
    ]
    with pytest.MonkeyPatch.context() as receipt_patch:
        receipt_patch.setattr(
            materializer,
            "_m20_source_binding_authority",
            lambda *_args, **_kwargs: authority,
        )
        receipt = materializer._expected_m20_migration_receipt(
            tmp_path,
            staged["stage_control"],
            staged["stage_coordination"],
            population,
            config,
            verified,
            validation_digest,
        )
    unhashed_receipt = dict(receipt)
    claimed_receipt_cid = unhashed_receipt.pop("receipt_cid")
    assert set(receipt) == materializer._M20_RECEIPT_KEYS
    assert claimed_receipt_cid == materializer._identity(unhashed_receipt)
    assert receipt["validation_digest"] != receipt["prior_validation_digest"]

    tampered = tmp_path / "tampered-control.duckdb"
    shutil.copyfile(staged["stage_control"], tampered)
    connection = duckdb.connect(str(tampered))
    try:
        connection.execute(
            "UPDATE tasks SET status='todo' WHERE task_alias='SAWM-007'"
        )
    finally:
        connection.close()
    with pytest.raises(
        materializer.MigrationRequired,
        match="frozen control authority",
    ):
        materializer._verify_m20_store_pair_copy(
            tampered,
            staged["stage_coordination"],
            prior_control,
            prior_coordination,
            population,
            config,
            validation_digest,
        )


def test_m20_prior_anchor_is_exact_unlaunched_and_rechecked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m20_prior_anchor_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._expected_m20_test_isolation_authority()
    prior_control = REPO_ROOT / authority["prior_store_id"]
    prior_coordination = REPO_ROOT / authority["prior_coordination_store_id"]
    before = (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    )
    captured: dict[str, str] = {}
    original_verify = materializer._verify_m19_store_pair_copy

    def capture_frozen_m19(*args: object, **kwargs: object) -> object:
        frozen_population = args[4]
        assert isinstance(frozen_population, Mapping)
        captured["source_binding_cid"] = str(
            frozen_population["source_binding"]["source_binding_cid"]
        )
        captured["validation_digest"] = str(args[6])
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(
        materializer, "_verify_m19_store_pair_copy", capture_frozen_m19
    )
    assert materializer._assert_m20_prior_anchor(
        REPO_ROOT, authority, population, config
    ) == (prior_control, prior_coordination)
    assert captured == {
        "source_binding_cid": authority["prior_source_binding_cid"],
        "validation_digest": authority["prior_validation_digest"],
    }
    assert (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    ) == before

    with monkeypatch.context() as final_listener:
        calls = 0

        def listener_appears(port: int) -> bool:
            nonlocal calls
            if port == 24_062:
                calls += 1
                return calls > 1
            return False

        final_listener.setattr(
            materializer, "_m18_prior_listener_is_active", listener_appears
        )
        final_listener.setattr(
            materializer,
            "_verify_m19_store_pair_copy",
            lambda *_args, **_kwargs: {
                "control_store_sha256": authority["prior_control_store_sha256"],
                "coordination_store_sha256": authority[
                    "prior_coordination_store_sha256"
                ],
                "projection_cid": authority["prior_projection_cid"],
                "append_surface_digest": authority["prior_append_surface_digest"],
                "semantic_authority_digest": authority[
                    "prior_semantic_authority_digest"
                ],
                "frozen_base_authority_digest": authority[
                    "prior_frozen_base_authority_digest"
                ],
                "catalog_digest": authority["prior_catalog_digest"],
                "event_watermark": 227,
            },
        )
        with pytest.raises(
            materializer.MaterializationError,
            match="predecessor verification mutated authority",
        ):
            materializer._assert_m20_prior_anchor(
                REPO_ROOT, authority, population, config
            )

    fake_control = tmp_path / "control.duckdb"
    fake_coordination = tmp_path / "control.coordination.duckdb"
    fake_receipt = tmp_path / "migration-receipt.json"
    shutil.copyfile(prior_control, fake_control)
    shutil.copyfile(prior_coordination, fake_coordination)
    shutil.copyfile(REPO_ROOT / authority["prior_migration_receipt_path"], fake_receipt)
    mapping = {
        authority["prior_store_id"]: fake_control,
        authority["prior_coordination_store_id"]: fake_coordination,
        authority["prior_migration_receipt_path"]: fake_receipt,
    }
    monkeypatch.setattr(
        materializer,
        "_confined_leaf",
        lambda _root, relative, **_kwargs: mapping[str(relative)],
    )
    execution = fake_control.with_name("control.execution.duckdb")
    execution.write_bytes(b"forbidden")
    with pytest.raises(
        materializer.MigrationRequired,
        match="byte or lifecycle anchor differs",
    ):
        materializer._assert_m20_prior_anchor(
            tmp_path, authority, population, config
        )
    execution.unlink()
    fake_receipt.write_bytes(b"{}\n")
    with pytest.raises(
        materializer.MigrationRequired,
        match="byte or lifecycle anchor differs",
    ):
        materializer._assert_m20_prior_anchor(
            tmp_path, authority, population, config
        )


def test_m20_receipt_publication_recovers_only_exact_hardlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m20_receipt_recovery_test",
    )
    expected = {key: "" for key in materializer._M20_RECEIPT_KEYS}
    expected.update(
        {
            "schema": "sawm/non-authoritative-migration-receipt@18",
            "authoritative": False,
            "receipt_is_final_pair_commit_marker": True,
            "current_source_binding_cid": "sha256:" + "1" * 64,
            "validation_digest": "sha256:" + "2" * 64,
            "prior_validation_digest": "sha256:" + "3" * 64,
        }
    )
    unhashed = dict(expected)
    unhashed.pop("receipt_cid")
    expected["receipt_cid"] = materializer._identity(unhashed)
    commit_checks = 0

    def commit_inputs(*_args: object, **_kwargs: object) -> None:
        nonlocal commit_checks
        commit_checks += 1

    monkeypatch.setattr(
        materializer,
        "_expected_m20_migration_receipt",
        lambda *_args, **_kwargs: dict(expected),
    )
    monkeypatch.setattr(
        materializer, "_assert_m20_receipt_commit_inputs", commit_inputs
    )

    def case(name: str) -> tuple[Path, Path]:
        root = tmp_path / name
        root.mkdir()
        control = root / "control.duckdb"
        coordination = root / "control.coordination.duckdb"
        control.write_bytes(b"control")
        coordination.write_bytes(b"coordination")
        return control, coordination

    control, coordination = case("normal")
    observed = materializer._ensure_m20_migration_receipt(
        control.parent, control, coordination, {}, {}, {}, "unused"
    )
    assert observed == expected
    assert os.lstat(control.parent / "migration-receipt.json").st_nlink == 1
    assert commit_checks >= 4

    control, coordination = case("pending-only")
    pending = control.parent / ".migration-receipt.json.7.tmp"
    pending.write_bytes(materializer._canonical(expected) + b"\n")
    assert materializer._ensure_m20_migration_receipt(
        control.parent, control, coordination, {}, {}, {}, "unused"
    ) == expected
    assert not pending.exists()
    assert os.lstat(control.parent / "migration-receipt.json").st_nlink == 1

    control, coordination = case("linked-crash")
    receipt = control.parent / "migration-receipt.json"
    receipt.write_bytes(materializer._canonical(expected) + b"\n")
    pending = control.parent / ".migration-receipt.json.8.tmp"
    os.link(receipt, pending)
    assert os.lstat(receipt).st_nlink == 2
    assert materializer._ensure_m20_migration_receipt(
        control.parent, control, coordination, {}, {}, {}, "unused"
    ) == expected
    assert not pending.exists()
    assert os.lstat(receipt).st_nlink == 1

    control, coordination = case("tampered-pending")
    (control.parent / ".migration-receipt.json.bad.tmp").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="pending M20 receipt differs",
    ):
        materializer._ensure_m20_migration_receipt(
            control.parent, control, coordination, {}, {}, {}, "unused"
        )

    control, coordination = case("ambiguous")
    for suffix in ("a", "b"):
        (control.parent / f".migration-receipt.json.{suffix}.tmp").write_bytes(
            materializer._canonical(expected) + b"\n"
        )
    with pytest.raises(
        materializer.MigrationRequired,
        match="ambiguous pending receipts",
    ):
        materializer._ensure_m20_migration_receipt(
            control.parent, control, coordination, {}, {}, {}, "unused"
        )

    control, coordination = case("hardlinked-lock")
    lock = control.parent / ".m20-receipt.publish.lock"
    lock.write_bytes(b"")
    os.link(lock, control.parent / "lock-alias")
    with pytest.raises(
        materializer.MaterializationError,
        match="lock is not a regular leaf",
    ):
        materializer._ensure_m20_migration_receipt(
            control.parent, control, coordination, {}, {}, {}, "unused"
        )

    control, coordination = case("contended-lock")
    monotonic = iter((0.0, 11.0))

    def contended_flock(_descriptor: int, operation: int) -> None:
        if operation & materializer.fcntl.LOCK_NB:
            raise BlockingIOError

    with monkeypatch.context() as contention:
        contention.setattr(materializer.fcntl, "flock", contended_flock)
        contention.setattr(
            materializer.time,
            "monotonic",
            lambda: next(monotonic, 11.0),
        )
        contention.setattr(materializer.time, "sleep", lambda _delay: None)
        with pytest.raises(
            materializer.MaterializationError,
            match="timed out acquiring the M20 receipt lock",
        ):
            materializer._ensure_m20_migration_receipt(
                control.parent, control, coordination, {}, {}, {}, "unused"
            )


def test_m20_receipt_and_validation_bindings_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m20_receipt_tamper_test",
    )
    authority = materializer._expected_m20_test_isolation_authority()
    config = {"test_isolation_successor_materialization": authority}
    population = {
        "source_binding": {"source_binding_cid": "sha256:" + "4" * 64}
    }
    with pytest.raises(
        materializer.MaterializationError,
        match="current source and validation must differ",
    ):
        materializer._m20_migration_body(
            population, config, authority["prior_validation_digest"]
        )

    expected = {key: "" for key in materializer._M20_RECEIPT_KEYS}
    expected.update(
        {
            "schema": "sawm/non-authoritative-migration-receipt@18",
            "authoritative": False,
            "receipt_is_final_pair_commit_marker": True,
            "current_source_binding_cid": population["source_binding"][
                "source_binding_cid"
            ],
            "validation_digest": "sha256:" + "5" * 64,
            "prior_validation_digest": authority["prior_validation_digest"],
            "projection_cid": "projection:exact",
            "prior_migration_receipt_cid": authority[
                "prior_migration_receipt_cid"
            ],
        }
    )
    unhashed = dict(expected)
    unhashed.pop("receipt_cid")
    expected["receipt_cid"] = materializer._identity(unhashed)
    control = tmp_path / "control.duckdb"
    coordination = tmp_path / "control.coordination.duckdb"
    control.write_bytes(b"control")
    coordination.write_bytes(b"coordination")
    receipt_path = tmp_path / "migration-receipt.json"
    monkeypatch.setattr(
        materializer,
        "_expected_m20_migration_receipt",
        lambda *_args, **_kwargs: dict(expected),
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m20_receipt_commit_inputs",
        lambda *_args, **_kwargs: None,
    )
    receipt_path.write_bytes(materializer._canonical(expected) + b"\n")
    assert materializer._verify_existing_m20_migration_receipt(
        tmp_path, control, coordination, population, config, {}, "unused"
    ) == expected
    mutations = (
        lambda item: item.__setitem__("projection_cid", "projection:forged"),
        lambda item: item.__setitem__(
            "current_source_binding_cid", "sha256:" + "6" * 64
        ),
        lambda item: item.__setitem__(
            "prior_migration_receipt_cid", "sha256:" + "7" * 64
        ),
        lambda item: item.pop("worker_self_approval"),
        lambda item: item.__setitem__("unexpected", False),
    )
    for mutate in mutations:
        forged = copy.deepcopy(expected)
        mutate(forged)
        unhashed = dict(forged)
        unhashed.pop("receipt_cid")
        forged["receipt_cid"] = materializer._identity(unhashed)
        receipt_path.write_bytes(materializer._canonical(forged) + b"\n")
        with pytest.raises(
            materializer.MigrationRequired,
            match="final pair marker differs",
        ):
            materializer._verify_existing_m20_migration_receipt(
                tmp_path, control, coordination, population, config, {}, "unused"
            )


def test_m20_target_verification_rechecks_sidecars_listener_and_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m20_target_race_test",
    )
    control = tmp_path / "control.duckdb"
    coordination = tmp_path / "control.coordination.duckdb"
    prior_control = tmp_path / "prior-control.duckdb"
    prior_coordination = tmp_path / "prior-coordination.duckdb"
    for path, body in (
        (control, b"target-control"),
        (coordination, b"coordination"),
        (prior_control, b"prior-control"),
        (prior_coordination, b"prior-coordination"),
    ):
        path.write_bytes(body)
    verified = {"valid": True}
    monkeypatch.setattr(
        materializer, "_m18_prior_listener_is_active", lambda _port: False
    )
    monkeypatch.setattr(
        materializer,
        "_verify_m20_store_pair_copy",
        lambda *_args, **_kwargs: verified,
    )
    assert materializer._verify_m20_store_pair(
        tmp_path,
        control,
        coordination,
        prior_control,
        prior_coordination,
        {},
        {},
        "unused",
    ) == verified

    sidecar = tmp_path / "control.execution.duckdb"

    def inject_sidecar(*_args: object, **_kwargs: object) -> object:
        sidecar.write_bytes(b"appeared")
        return verified

    monkeypatch.setattr(
        materializer, "_verify_m20_store_pair_copy", inject_sidecar
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="forbidden mutable sidecar",
    ):
        materializer._verify_m20_store_pair(
            tmp_path,
            control,
            coordination,
            prior_control,
            prior_coordination,
            {},
            {},
            "unused",
        )
    sidecar.unlink()

    calls = 0

    def listener_appears(_port: int) -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    monkeypatch.setattr(
        materializer, "_m18_prior_listener_is_active", listener_appears
    )
    monkeypatch.setattr(
        materializer,
        "_verify_m20_store_pair_copy",
        lambda *_args, **_kwargs: verified,
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="listener appeared during verification",
    ):
        materializer._verify_m20_store_pair(
            tmp_path,
            control,
            coordination,
            prior_control,
            prior_coordination,
            {},
            {},
            "unused",
        )

    monkeypatch.setattr(
        materializer, "_m18_prior_listener_is_active", lambda _port: False
    )
    original_copy = shutil.copyfile

    def corrupt_copy(source: object, target: object, *args: object, **kwargs: object) -> object:
        result = original_copy(source, target, *args, **kwargs)
        if Path(source) == control:
            with Path(target).open("ab") as stream:
                stream.write(b"corrupt")
        return result

    monkeypatch.setattr(materializer.shutil, "copyfile", corrupt_copy)
    with pytest.raises(
        materializer.MaterializationError,
        match="verification copy differs",
    ):
        materializer._verify_m20_store_pair(
            tmp_path,
            control,
            coordination,
            prior_control,
            prior_coordination,
            {},
            {},
            "unused",
        )


def test_m20_live_projection_is_derived_from_and_bound_to_closed_marker() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m20_live_projection_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m20_live_projection_test",
    )
    authority = materializer._expected_m20_test_isolation_authority()
    assert "target_projection_cid" not in authority
    assert operator._expected_live_projection_cid(
        authority, {"projection_cid": "projection:closed-marker"}
    ) == "projection:closed-marker"
    assert operator._expected_live_projection_cid(authority, {}) == ""
    fake_materializer = SimpleNamespace(
        MigrationRequired=materializer.MigrationRequired,
        _inspect_m20_head_task_projection=lambda *_args: {
            "event_watermark": 229,
            "plan_revision": 21,
            "projection_cid": "projection:closed-marker",
        },
    )
    assert operator._verify_m20_live_head_task_projection(
        object(),
        {"taskboard": []},
        fake_materializer,
        expected_projection_cid="projection:closed-marker",
    ) == ({}, {}, {})
    for projection in ("", "projection:forged"):
        with pytest.raises(
            materializer.MigrationRequired,
            match="live head projection differs",
        ):
            operator._verify_m20_live_head_task_projection(
                object(),
                {"taskboard": []},
                fake_materializer,
                expected_projection_cid=projection,
            )


def test_m20_source_scope_and_partial_declaration_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m20_source_scope_test",
    )
    dependency = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_validator_m20_partial_test",
    )
    authority = materializer._expected_m20_test_isolation_authority()
    final_head = "f" * 40
    final_tree = "e" * 40
    population = {
        "source_binding": {
            "head": final_head,
            "tree": final_tree,
            "datasets_gitlink": authority["prior_datasets_gitlink"],
            "kit_gitlink": authority["prior_kit_gitlink"],
        }
    }
    extra_path = "unexpected.py"
    inject_extra = False

    def fake_git(root: Path, *args: str) -> str:
        if args[:3] == ("diff", "--name-status", "--no-renames"):
            before, after = args[3], args[4]
            if (before, after) == (
                authority["prior_source_head"],
                authority["repair_source_commit"],
            ):
                paths = sorted(materializer._M20_TEST_ISOLATION_REPAIR_BLOBS)
            else:
                paths = sorted(authority["operator_control_paths"])
                if inject_extra and before == authority["repair_source_commit"]:
                    paths.append(extra_path)
            return "\n".join(f"M\t{path}" for path in paths)
        if args[:2] == ("merge-base", "--is-ancestor"):
            return ""
        if args[0] != "rev-parse":
            raise AssertionError(args)
        ref = args[1]
        trees = {
            f"{authority['prior_source_head']}^{{tree}}": authority[
                "prior_source_tree"
            ],
            f"{authority['repair_source_commit']}^{{tree}}": authority[
                "repair_source_tree"
            ],
            f"{final_head}^{{tree}}": final_tree,
        }
        if ref in trees:
            return str(trees[ref])
        for path, blob in materializer._M20_TEST_ISOLATION_REPAIR_BLOBS.items():
            if ref == f"{authority['repair_source_commit']}:{path}":
                return blob
        for dependency_name, gitlink_key, tree_key in (
            ("ipfs_datasets_py", "prior_datasets_gitlink", "prior_datasets_tree"),
            ("ipfs_kit_py", "prior_kit_gitlink", "prior_kit_tree"),
        ):
            if ref in {
                f"{head}:{dependency_name}"
                for head in (
                    authority["prior_source_head"],
                    authority["repair_source_commit"],
                    final_head,
                )
            }:
                return str(authority[gitlink_key])
            if root.name == dependency_name and ref == (
                f"{authority[gitlink_key]}^{{tree}}"
            ):
                return str(authority[tree_key])
        raise AssertionError((root, args))

    monkeypatch.setattr(materializer, "_git", fake_git)
    materializer._assert_m20_source_delta(REPO_ROOT, population, authority)
    inject_extra = True
    with pytest.raises(
        materializer.MaterializationError,
        match="post-repair source delta differs",
    ):
        materializer._assert_m20_source_delta(REPO_ROOT, population, authority)

    key = "test_isolation_successor_materialization"
    assert dependency._m20_successor_declared({}, {}, {key: authority}) is True
    assert dependency._m20_successor_declared(
        {}, {f"{key}_cid": materializer._identity(authority)}, {}
    ) is True
    assert dependency._m20_successor_declared({}, {}, {}) is False
    for scheduler, seal, migration in (
        ({}, {}, {key: authority}),
        ({}, {f"{key}_cid": materializer._identity(authority)}, {}),
    ):
        errors = dependency._m20_test_isolation_successor_errors(
            scheduler,
            seal,
            migration,
            root=REPO_ROOT,
            require_active_runtime=False,
        )
        assert (
            "M20 test-isolation successor authority is only partially declared"
            in errors
        )
        assert "M20 test-isolation authority differs across controls" in errors


def test_m19_operator_and_validators_are_presence_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    config, inventory, seal = _historical_successor_controls_at(
        "live_catalog_inventory_successor_materialization",
        config,
        inventory,
        seal,
    )
    assert inventory is not None
    assert seal is not None
    present = dict(config)
    board_validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_validator_m19_presence_test",
    )
    monkeypatch.setattr(
        board_validator,
        "_m19_migration_errors",
        lambda *_args, **_kwargs: ["M19 malformed"],
    )
    monkeypatch.setattr(
        board_validator,
        "_m18_migration_errors",
        lambda *_args, **_kwargs: ["M18 history checked"],
    )
    monkeypatch.setattr(
        board_validator,
        "_m17_migration_errors",
        lambda *_args, **_kwargs: ["M17 history checked"],
    )
    monkeypatch.setattr(
        board_validator,
        "_m16_migration_errors",
        lambda *_args, **_kwargs: ["M16 history checked"],
    )
    assert board_validator._active_successor_migration_errors(
        present,
        seal,
        inventory,
    ) == [
        "M19 malformed",
        "M18 history checked",
        "M17 history checked",
        "M16 history checked",
    ]


def test_m17_source_seal_binds_repair_and_nine_control_delta() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m17_source_seal_test",
    )
    authority = materializer._expected_m17_source_binding_authority()
    m18 = materializer._expected_m18_portal_completion_persistence_authority()
    historical_head = m18["prior_source_head"]
    assert historical_head == "e675b96cdb3eba7479d3ddb32beefb439489d986"
    assert materializer._git(
        REPO_ROOT, "rev-parse", f"{historical_head}^{{tree}}"
    ) == m18["prior_source_tree"]
    changed = {
        line.split("\t", 1)[1]
        for line in materializer._git(
            REPO_ROOT,
            "diff",
            "--name-status",
            "--no-renames",
            authority["repair_source_commit"],
            historical_head,
            "--",
        ).splitlines()
    }
    assert changed == set(authority["operator_control_paths"])
    assert set(authority["accepted_source_repair"]["changed_paths"]) == {
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
    }
    assert set(authority["operator_control_paths"]) == set(
        authority["bounded_control_plane_repair_paths"]
    )


def test_m18_portal_completion_authority_is_presence_first_and_exact() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m18_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "portal_completion_persistence_successor_materialization"
    authority = materializer._expected_m18_portal_completion_persistence_authority()
    assert config[key] == inventory[key] == authority
    assert seal[f"{key}_cid"] == materializer._identity(authority)
    assert materializer._m18_successor_configured(config) is True
    assert authority["prior_event_watermark"] == 222
    assert authority["target_event_watermark"] == 225
    assert authority["target_generation"] == 19
    assert authority["target_plan_revision"] == 19
    assert authority["target_quack_port"] == 24_061
    assert authority["prior_owner_marker_present"] is False
    assert authority["prior_stop_control_present"] is False
    assert authority["prior_token_handoff_present"] is False
    assert authority["prior_wals_absent"] is True
    assert authority["prior_listener_present"] is False
    assert authority["prior_pid_present"] is False
    assert authority["prior_active_count"] == 0
    assert set(authority["failure_receipts"]) == {"SAWM-007"}
    assert set(authority["task_rearms"]) == {"SAWM-007"}
    assert authority["accepted_runtime_source_transitions"][2]["merge_commit"] == (
        "2675ecdf5462c0be90d4a8825773c403172edf66"
    )
    assert authority["accepted_source_repair"]["blob_oids"] == {
        (
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
            "database_portal_bridge.py"
        ): "5ff1def4410f184c3f41f1b468e230df3a5e4794",
        "test/api/test_agent_supervisor_database_portal_bridge.py": (
            "91459021212f4bc4366950def56962c2ce623a89"
        ),
    }
    malformed = dict(config)
    malformed[key] = None
    with pytest.raises(
        materializer.MaterializationError,
        match="M18 portal completion persistence authority is invalid",
    ):
        materializer._m18_successor_configured(malformed)


def test_m20_historical_authority_retains_generation_21_runtime_namespace() -> None:
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    authority = config["test_isolation_successor_materialization"]
    root = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m20"
    assert authority["target_runtime_root"] == root
    assert authority["target_store_id"] == f"{root}/control.duckdb"
    assert authority["target_coordination_store_id"] == (
        f"{root}/control.coordination.duckdb"
    )
    assert authority["target_generation"] == 21
    assert authority["target_quack_port"] == 24_063
    assert authority["target_plan_revision"] == 21
    assert authority["target_event_watermark"] == 229


def test_m18_operator_and_validators_are_presence_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "portal_completion_persistence_successor_materialization"
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m18_presence_test",
    )
    historical, historical_inventory, historical_seal = (
        _historical_successor_controls_at(key, config, inventory, seal)
    )
    assert historical_inventory is not None
    assert historical_seal is not None
    assert dict(operator._active_source_repair_materialization(historical)) == config[key]
    malformed = copy.deepcopy(historical)
    malformed[key] = None
    with pytest.raises(
        operator.OperatorError,
        match="active M18 portal-completion successor authority is invalid",
    ):
        operator._active_source_repair_materialization(malformed)

    dependency_validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_validator_m18_presence_test",
    )
    dependency_errors = dependency_validator._m18_portal_completion_persistence_errors(
        malformed,
        seal,
        inventory,
        root=REPO_ROOT,
    )
    assert "M18 portal-completion authority differs across controls" in (
        dependency_errors
    )

    board_validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_validator_m18_presence_test",
    )
    monkeypatch.setattr(
        board_validator,
        "_m18_migration_errors",
        lambda *_args, **_kwargs: ["M18 malformed"],
    )
    monkeypatch.setattr(
        board_validator,
        "_m17_migration_errors",
        lambda *_args, **_kwargs: ["M17 history checked"],
    )
    monkeypatch.setattr(
        board_validator,
        "_m16_migration_errors",
        lambda *_args, **_kwargs: ["M16 history checked"],
    )
    assert board_validator._active_successor_migration_errors(
        malformed,
        historical_seal,
        historical_inventory,
    ) == ["M18 malformed", "M17 history checked", "M16 history checked"]


@pytest.mark.parametrize(
    "artifact",
    (
        "quack-state-server.pid",
        "quack-state-server.owner.json",
        "control.execution.duckdb.wal",
        "control.read-replica.duckdb.wal",
        "stale.quack-token",
    ),
)
def test_m18_prior_anchor_rejects_every_stale_lifecycle_class(
    artifact: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        f"sawm_materializer_m18_lifecycle_{artifact.replace('.', '_')}",
    )
    authority = materializer._expected_m18_portal_completion_persistence_authority()
    expected_by_noun = {
        "stopped M17 control store": (
            authority["prior_control_store_sha256"],
            authority["prior_control_store_size"],
        ),
        "stopped M17 coordination store": (
            authority["prior_coordination_store_sha256"],
            authority["prior_coordination_store_size"],
        ),
        "stopped M17 status projection": (
            authority["prior_stopped_status_projection_sha256"],
            authority["prior_stopped_status_projection_size"],
        ),
        "historical M17 migration receipt": (
            authority["prior_migration_receipt_sha256"],
            authority["prior_migration_receipt_size"],
        ),
        "stopped M17 execution sidecar": (
            authority["prior_execution_sidecar_sha256"],
            authority["prior_execution_sidecar_size"],
        ),
        "stopped M17 read replica": (
            authority["prior_read_replica_sha256"],
            authority["prior_read_replica_size"],
        ),
    }

    def sealed_anchor(
        _path: Path, *, noun: str, **_kwargs: object
    ) -> tuple[str, int]:
        return expected_by_noun[noun]

    monkeypatch.setattr(materializer, "_stable_regular_sha256", sealed_anchor)
    control = (REPO_ROOT / authority["prior_store_id"]).resolve()
    owner_dir = control.parent / "quack-owner"
    if artifact == "control.execution.duckdb.wal":
        injected = control.with_name(artifact)
    elif artifact == "control.read-replica.duckdb.wal":
        injected = control.with_name(artifact)
    else:
        injected = owner_dir / artifact

    if artifact.endswith(".quack-token"):
        original_glob = Path.glob

        def lifecycle_glob(path: Path, pattern: str):
            if path == owner_dir and pattern == "*.quack-token":
                return iter((injected,))
            return original_glob(path, pattern)

        monkeypatch.setattr(Path, "glob", lifecycle_glob)
    else:
        original_lexists = os.path.lexists

        def lifecycle_lexists(path: os.PathLike[str] | str) -> bool:
            return Path(path) == injected or original_lexists(path)

        monkeypatch.setattr(materializer.os.path, "lexists", lifecycle_lexists)

    with pytest.raises(
        materializer.MigrationRequired,
        match="stopped M17 byte or lifecycle anchor differs",
    ):
        materializer._assert_m18_prior_anchor(REPO_ROOT, authority)


def test_m18_listener_observation_is_present_and_parse_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m18_listener_observation_test",
    )
    header = (
        "sl local_address rem_address st tx_queue rx_queue tr "
        "tm->when retrnsmt uid timeout inode\n"
    )
    inactive = (
        "0: 0100007F:0001 00000000:0000 01 00000000:00000000 "
        "00:00000000 00000000 0 0 1\n"
    )
    active = (
        "0: 0100007F:5DFC 00000000:0000 0A 00000000:00000000 "
        "00:00000000 00000000 0 0 1\n"
    )
    observations: dict[str, str | BaseException] = {}

    def proc_read_text(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        del encoding, errors
        observed = observations[path.name]
        if isinstance(observed, BaseException):
            raise observed
        return observed

    monkeypatch.setattr(Path, "read_text", proc_read_text)
    observations.update({"tcp": FileNotFoundError(), "tcp6": header + inactive})
    assert materializer._m18_prior_listener_is_active(24_060) is False

    observations.update({"tcp": FileNotFoundError(), "tcp6": FileNotFoundError()})
    with pytest.raises(
        materializer.MigrationRequired,
        match="listener observation is unavailable",
    ):
        materializer._m18_prior_listener_is_active(24_060)

    observations.update({"tcp": header + "malformed\n", "tcp6": header + inactive})
    with pytest.raises(
        materializer.MigrationRequired,
        match="listener observation is malformed",
    ):
        materializer._m18_prior_listener_is_active(24_060)

    observations.update({"tcp": PermissionError(), "tcp6": header + inactive})
    with pytest.raises(
        materializer.MigrationRequired,
        match="listener observation is unavailable",
    ):
        materializer._m18_prior_listener_is_active(24_060)

    observations.update({"tcp": header + active, "tcp6": FileNotFoundError()})
    assert materializer._m18_prior_listener_is_active(24_060) is True


@pytest.mark.parametrize("entrypoint", ("ensure", "verify"))
@pytest.mark.parametrize("race", ("store_swap", "pending_receipt"))
def test_m18_existing_marker_rechecks_pair_and_pending_after_receipt_load(
    entrypoint: str,
    race: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        f"sawm_materializer_m18_marker_race_{entrypoint}_{race}",
    )
    control = tmp_path / "control.duckdb"
    coordination = tmp_path / "control.coordination.duckdb"
    control_bytes = b"sealed-control\n"
    coordination_bytes = b"sealed-coordination\n"
    control.write_bytes(control_bytes)
    coordination.write_bytes(coordination_bytes)
    body = {
        "schema": "sawm/test-m18-final-marker@1",
        "control_store_sha256": hashlib.sha256(control_bytes).hexdigest(),
        "control_store_size": len(control_bytes),
        "coordination_store_sha256": hashlib.sha256(
            coordination_bytes
        ).hexdigest(),
        "coordination_store_size": len(coordination_bytes),
    }
    expected = {**body, "receipt_cid": materializer._identity(body)}
    receipt_path = tmp_path / "migration-receipt.json"
    receipt_path.write_bytes(materializer._canonical(expected) + b"\n")
    pending = tmp_path / ".migration-receipt.json.race.tmp"
    verified = {"verified": True}

    monkeypatch.setattr(
        materializer,
        "_expected_m18_migration_receipt",
        lambda *_args, **_kwargs: dict(expected),
    )
    original_load = materializer._load_nofollow_json

    def load_then_race(*args: object, **kwargs: object):
        observed = original_load(*args, **kwargs)
        if kwargs.get("noun") == "M18 final pair marker":
            if race == "store_swap":
                control.write_bytes(b"swapped-after-receipt-load\n")
            else:
                pending.write_text("pending\n", encoding="utf-8")
        return observed

    monkeypatch.setattr(materializer, "_load_nofollow_json", load_then_race)
    if entrypoint == "ensure":
        monkeypatch.setattr(
            materializer,
            "_m18_portal_completion_persistence_authority",
            lambda *_args, **_kwargs: {},
        )
        monkeypatch.setattr(
            materializer,
            "_assert_committed_clean_source",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            materializer,
            "_assert_m18_source_delta",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            materializer,
            "_assert_m18_prior_anchor",
            lambda *_args, **_kwargs: (control, coordination),
        )
        monkeypatch.setattr(
            materializer,
            "_verify_m18_store_pair",
            lambda *_args, **_kwargs: dict(verified),
        )
        def invoke() -> dict[str, object]:
            return materializer._ensure_m18_migration_receipt(
                tmp_path,
                control,
                coordination,
                {},
                {},
                verified,
                "sha256:test",
            )
    else:
        def invoke() -> dict[str, object]:
            return materializer._verify_existing_m18_migration_receipt(
                tmp_path,
                control,
                coordination,
                {},
                {},
                verified,
                "sha256:test",
            )

    match = (
        "M18 store pair changed at receipt commit"
        if race == "store_swap"
        else "M18 retained a pending receipt temporary"
    )
    with pytest.raises(materializer.MigrationRequired, match=match):
        invoke()


@pytest.mark.parametrize("write_mode", ("short", "zero_progress"))
def test_m18_receipt_writer_handles_short_write_or_fails_closed(
    write_mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        f"sawm_materializer_m18_short_write_{write_mode}",
    )
    control = tmp_path / "control.duckdb"
    coordination = tmp_path / "control.coordination.duckdb"
    control.write_bytes(b"sealed-control\n")
    coordination.write_bytes(b"sealed-coordination\n")
    body = {
        "schema": "sawm/test-m18-final-marker@1",
        "control_store_sha256": hashlib.sha256(control.read_bytes()).hexdigest(),
        "control_store_size": control.stat().st_size,
        "coordination_store_sha256": hashlib.sha256(
            coordination.read_bytes()
        ).hexdigest(),
        "coordination_store_size": coordination.stat().st_size,
    }
    expected = {**body, "receipt_cid": materializer._identity(body)}
    verified = {"verified": True}
    monkeypatch.setattr(
        materializer,
        "_expected_m18_migration_receipt",
        lambda *_args, **_kwargs: dict(expected),
    )
    monkeypatch.setattr(
        materializer,
        "_m18_portal_completion_persistence_authority",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        materializer,
        "_assert_committed_clean_source",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m18_source_delta",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m18_prior_anchor",
        lambda *_args, **_kwargs: (control, coordination),
    )
    monkeypatch.setattr(
        materializer,
        "_verify_m18_store_pair",
        lambda *_args, **_kwargs: dict(verified),
    )
    original_write = materializer.os.write

    def bounded_write(descriptor: int, payload: bytes | memoryview) -> int:
        if write_mode == "zero_progress":
            return 0
        return original_write(descriptor, payload[:7])

    monkeypatch.setattr(materializer.os, "write", bounded_write)

    def publish() -> dict[str, object]:
        return materializer._ensure_m18_migration_receipt(
            tmp_path,
            control,
            coordination,
            {},
            {},
            verified,
            "sha256:test",
        )

    receipt_path = tmp_path / "migration-receipt.json"
    if write_mode == "short":
        assert publish() == expected
        assert receipt_path.read_bytes() == materializer._canonical(expected) + b"\n"
        assert not tuple(tmp_path.glob(".migration-receipt.json.*.tmp"))
        return

    with pytest.raises(
        materializer.MaterializationError,
        match="M18 receipt write made no progress",
    ):
        publish()
    assert not receipt_path.exists()
    pending = tuple(tmp_path.glob(".migration-receipt.json.*.tmp"))
    assert len(pending) == 1
    monkeypatch.setattr(materializer.os, "write", original_write)
    with pytest.raises((json.JSONDecodeError, materializer.MigrationRequired)):
        publish()
    assert not receipt_path.exists()
    pending[0].unlink()
    assert publish() == expected


class _RemoteCatalogResult:
    def __init__(self, rows: tuple[tuple[str, ...], ...]) -> None:
        self._rows = rows

    def fetchall(self) -> tuple[tuple[str, ...], ...]:
        return self._rows


class _RemoteCatalogConnection:
    """Quack-shaped catalog: rows/columns exist, BASE TABLE entries do not."""

    def __init__(
        self,
        *,
        duck_tables: tuple[str, ...],
        columns: tuple[str, ...],
        views: tuple[str, ...] = (),
        base_tables: tuple[str, ...] = (),
    ) -> None:
        self.duck_tables = duck_tables
        self.columns = columns
        self.views = views
        self.base_tables = base_tables

    def execute(self, sql: str, parameters: object = None) -> _RemoteCatalogResult:
        del parameters
        normalized = " ".join(str(sql).split())
        if "FROM duckdb_tables()" in normalized:
            return _RemoteCatalogResult(tuple((name,) for name in self.duck_tables))
        if "table_type='BASE TABLE'" in normalized or 'table_type = \'BASE TABLE\'' in normalized:
            return _RemoteCatalogResult(tuple((name,) for name in self.base_tables))
        if "AND table_type='VIEW'" in normalized or "AND table_type = 'VIEW'" in normalized:
            return _RemoteCatalogResult(tuple((name,) for name in self.views))
        if "FROM information_schema.columns" in normalized and "DISTINCT table_name" in normalized:
            return _RemoteCatalogResult(tuple((name,) for name in self.columns))
        raise AssertionError(f"unexpected catalog SQL: {normalized}")


def test_live_remote_frozen_table_names_accepts_quack_omitted_base_table_entries() -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_live_remote_catalog_test",
    )
    append = ("domain_events", "evidence_nodes", "plan_revisions", "plans")
    frozen = ("artifacts", "credentials", "goals", "state_servers", "tasks")
    views = ("task_board",)
    columns = frozen + append + views
    remote = _RemoteCatalogConnection(
        duck_tables=frozen + append,
        columns=columns,
        views=views,
        base_tables=(),
    )
    assert operator._live_remote_frozen_table_names(remote) == frozen

    empty = _RemoteCatalogConnection(
        duck_tables=(),
        columns=columns,
        views=views,
        base_tables=(),
    )
    assert operator._live_remote_frozen_table_names(empty) == frozen

    with pytest.raises(
        operator.OperatorError,
        match="M18 live frozen table inventory differs",
    ):
        operator._live_remote_frozen_table_names(
            _RemoteCatalogConnection(
                duck_tables=(),
                columns=append + views,
                views=views,
                base_tables=(),
            )
        )


def test_m18_live_marker_rejects_rehashed_fields_and_tail_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m18_live_marker_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m18_live_marker_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._expected_m18_portal_completion_persistence_authority()
    target_runtime_root = authority["target_runtime_root"]
    config.pop("automatic_stall_recovery_successor_materialization", None)
    config.pop("native_duckdb_preload_successor_materialization", None)
    config.pop("multi_lane_sidecar_reopen_successor_materialization")
    config.pop("multi_lane_successor_materialization")
    config.pop("live_preflight_receipt_compatibility_successor_materialization")
    config.pop("generation_realization_successor_materialization")
    config.pop("test_isolation_successor_materialization")
    config.pop("live_catalog_inventory_successor_materialization")
    config["database_program"].update(
        {
            "quack_endpoint": "quack:127.0.0.1:24061",
            "store_id": authority["target_store_id"],
            "store_generation": "19",
            "event_store_path": f"{target_runtime_root}/events",
            "runtime_registry_path": f"{target_runtime_root}/registry",
            "worktree_root": f"{target_runtime_root}/worktrees",
        }
    )
    config["quack_owner"].update(
        {
            "database_path": authority["target_store_id"],
            "state_dir": f"{target_runtime_root}/quack-owner",
            "port": authority["target_quack_port"],
            "store_id": authority["target_store_id"],
        }
    )
    config["runtime_paths"] = {
        "root": target_runtime_root,
        "state": f"{target_runtime_root}/state",
        "worktrees": f"{target_runtime_root}/worktrees",
        "merge_queue": f"{target_runtime_root}/merge-queue",
        "logs": f"{target_runtime_root}/logs",
        "generated_runtime_artifacts_are_completion_authority": False,
    }
    validation_digest = "sha256:" + "1" * 64
    target_root = tmp_path / authority["target_runtime_root"]
    target_root.mkdir(parents=True)
    staged = materializer._stage_m18_store_pair(
        REPO_ROOT,
        target_root,
        REPO_ROOT / authority["prior_store_id"],
        REPO_ROOT / authority["prior_coordination_store_id"],
        population,
        config,
        validation_digest,
    )
    prior_coordination = tmp_path / authority["prior_coordination_store_id"]
    prior_coordination.parent.mkdir(parents=True)
    prior_control = tmp_path / authority["prior_store_id"]
    shutil.copyfile(
        REPO_ROOT / authority["prior_store_id"],
        prior_control,
    )
    shutil.copyfile(
        REPO_ROOT / authority["prior_coordination_store_id"],
        prior_coordination,
    )
    prior_artifacts = {
        noun: tmp_path / authority[path_key]
        for noun, path_key in (
            ("status", "prior_stopped_status_projection_path"),
            ("receipt", "prior_migration_receipt_path"),
            ("execution", "prior_execution_sidecar_path"),
            ("read replica", "prior_read_replica_path"),
        )
    }
    for noun, path_key in (
        ("status", "prior_stopped_status_projection_path"),
        ("receipt", "prior_migration_receipt_path"),
        ("execution", "prior_execution_sidecar_path"),
        ("read replica", "prior_read_replica_path"),
    ):
        target = prior_artifacts[noun]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / authority[path_key], target)
    monkeypatch.setattr(
        materializer,
        "_m18_portal_completion_persistence_authority",
        lambda *_args, **_kwargs: dict(authority),
    )
    receipt = materializer._expected_m18_migration_receipt(
        tmp_path,
        staged["stage_control"],
        staged["stage_coordination"],
        population,
        config,
        staged["verified"],
        validation_digest,
    )
    receipt_path = target_root / "migration-receipt.json"
    receipt_path.write_bytes(materializer._canonical(receipt) + b"\n")
    monkeypatch.setattr(operator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        materializer,
        "build_population",
        lambda _root: population,
    )

    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        PINNED_PROFILE_ID,
        PINNED_QUACK_EXTENSION,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    live_identity = {
        "server_id": "server:00000000-0000-4000-8000-000000000019",
        "store_id": authority["target_store_id"],
        "database_uuid": authority["prior_database_uuid"],
        "process_birth_id": "birth:" + "1" * 32,
        "listen_uri": config["database_program"]["quack_endpoint"],
        "extension_fingerprint": "sha256:"
        + config["quack_owner"]["pinned_extension"]["sha256"],
        "schema_revision": 1,
        "schema_fingerprint": "sha256:" + "2" * 64,
        "generation": 19,
        "fence_epoch": 19,
        "revision": 0,
        "credential_generation": 19,
        "secret_handle": config["quack_owner"]["secret_handle"],
        "repository_id": config["quack_owner"]["repository_id"],
        "startup_epoch": 1_788_016_500,
        "started_at": "2026-08-29T16:35:00Z",
        "status": "ready",
    }
    remote_identity = {
        key: live_identity[key]
        for key in (
            "server_id",
            "store_id",
            "database_uuid",
            "process_birth_id",
            "listen_uri",
            "extension_fingerprint",
            "schema_revision",
            "generation",
            "credential_generation",
            "schema_fingerprint",
        )
    }
    remote_identity.update({"live": True, "canonical_rows_verified": True})
    lifecycle = duckdb.connect(str(staged["stage_control"]))
    try:
        lifecycle.execute(
            "INSERT INTO store_generations VALUES (?,?,?,?,?,?,?,?,?)",
            [
                19,
                1,
                19,
                0,
                live_identity["database_uuid"],
                live_identity["process_birth_id"],
                live_identity["started_at"],
                "",
                "{}",
            ],
        )
        lifecycle.execute(
            "INSERT INTO state_servers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                live_identity["server_id"],
                live_identity["store_id"],
                live_identity["database_uuid"],
                live_identity["process_birth_id"],
                live_identity["listen_uri"],
                live_identity["extension_fingerprint"],
                1,
                19,
                live_identity["started_at"],
                None,
                "ready",
                1,
                "",
                "{}",
            ],
        )
        lifecycle.execute(
            "INSERT INTO server_epochs VALUES (?,?,?,?,?)",
            [
                live_identity["server_id"],
                live_identity["startup_epoch"],
                19,
                live_identity["started_at"],
                None,
            ],
        )
        capability_body = json.dumps(
            {
                "status": "compatible",
                "profile_id": PINNED_PROFILE_ID,
                "extension_fingerprint": live_identity["extension_fingerprint"],
            },
            sort_keys=True,
        )
        lifecycle.execute(
            "INSERT INTO capability_snapshots VALUES (?,?,?,?,?,?,?,?,?)",
            [
                f"cap:{live_identity['server_id']}:19",
                live_identity["server_id"],
                PINNED_PROFILE_ID,
                duckdb.__version__,
                PINNED_QUACK_EXTENSION,
                live_identity["extension_fingerprint"],
                "compatible",
                live_identity["started_at"],
                capability_body,
            ],
        )
        lifecycle.execute(
            "INSERT INTO credentials VALUES (?,?,?,?,?,?,?,?)",
            [
                f"cred:{live_identity['server_id']}:19",
                live_identity["secret_handle"],
                19,
                "quack-auth",
                live_identity["started_at"],
                None,
                None,
                0,
            ],
        )
        lifecycle.execute("CHECKPOINT")
    finally:
        lifecycle.close()

    source = DatabaseTaskSource(staged["stage_control"], install_schema=False)
    try:
        lifecycle_reads = {
            table: 0
            for table in (
                "state_servers",
                "store_generations",
                "server_epochs",
                "capability_snapshots",
                "credentials",
            )
        }
        original_connection = source.intent._connection

        class CountingConnection:
            def __init__(self, connection: object) -> None:
                self._wrapped = connection

            def __getattr__(self, name: str) -> object:
                return getattr(self._wrapped, name)

            def execute(
                self,
                sql: str,
                parameters: object = None,
            ) -> object:
                normalized = " ".join(str(sql).split())
                for table in lifecycle_reads:
                    if f'SELECT * FROM "{table}" ORDER BY' in normalized:
                        lifecycle_reads[table] += 1
                if parameters is None:
                    return self._wrapped.execute(sql)
                return self._wrapped.execute(sql, parameters)

        @contextlib.contextmanager
        def counted_connection(*, write: bool = False):
            with original_connection(write=write) as connection:
                yield CountingConnection(connection)

        source.intent._connection = counted_connection
        canonical_prior_paths = {
            prior_control.resolve(),
            prior_coordination.resolve(),
        }
        connected_paths: list[Path] = []
        real_connect = duckdb.connect

        def guarded_connect(
            database: object = ":memory:",
            *args: object,
            **kwargs: object,
        ) -> object:
            raw = os.fspath(database)
            if raw != ":memory:":
                path = Path(raw).resolve()
                assert path not in canonical_prior_paths
                connected_paths.append(path)
            return real_connect(database, *args, **kwargs)

        with monkeypatch.context() as connect_guard:
            connect_guard.setattr(duckdb, "connect", guarded_connect)
            accepted = operator._require_m18_final_pair_marker(
                config,
                authority,
                materializer,
                live_source=source,
                validation_digest=validation_digest,
                live_identity=live_identity,
                remote_identity=remote_identity,
            )
        assert accepted == receipt
        prior_copy_paths = {
            path.name
            for path in connected_paths
            if "sawm-r2-m18-live-prior-" in str(path.parent)
        }
        assert prior_copy_paths == {
            "control.duckdb",
            "control.coordination.duckdb",
        }
        assert lifecycle_reads == {table: 1 for table in lifecycle_reads}
        source.intent._connection = original_connection
        mutations: dict[str, object] = {
            "control_database_is_authority": False,
            "coordination_database_is_authority": False,
            "validation_digest": "sha256:" + "0" * 64,
            "migration_digest": "sha256:" + "0" * 64,
            "migration_evidence_id": "forged-evidence",
            "plan_migration_event_id": "forged-plan-event",
            "migration_evidence_event_id": "forged-evidence-event",
            "task_rearm_event_ids": {"SAWM-007": "forged-task-event"},
            "task_rearm_receipt_cids": {"SAWM-007": "forged-task-receipt"},
            "coordination_rearm_event_ids": {
                "SAWM-007": "lease-event:" + "0" * 32
            },
            "coordination_rearm_ids": {"SAWM-007": "forged-rearm"},
            "plan_projection_cid": "forged-plan-projection",
            "migration_event_watermark": 224,
            "control_store_sha256": "not-a-sha256",
            "control_store_size": 0,
            "append_surface_digest": "sha256:" + "0" * 64,
            "catalog_digest": "sha256:" + "0" * 64,
        }
        for field, value in mutations.items():
            forged = {**receipt, field: value}
            unhashed = dict(forged)
            unhashed.pop("receipt_cid")
            forged["receipt_cid"] = materializer._identity(unhashed)
            receipt_path.write_bytes(materializer._canonical(forged) + b"\n")
            with pytest.raises(operator.OperatorError):
                operator._require_m18_final_pair_marker(
                    config,
                    authority,
                    materializer,
                    live_source=source,
                    validation_digest=validation_digest,
                    live_identity=live_identity,
                    remote_identity=remote_identity,
                )
        receipt_path.write_bytes(materializer._canonical(receipt) + b"\n")

        for _noun, path in prior_artifacts.items():
            with path.open("r+b") as handle:
                first = handle.read(1)
                assert first
                handle.seek(0)
                handle.write(b"\x00" if first != b"\x00" else b"\x01")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                with pytest.raises(operator.OperatorError):
                    operator._require_m18_final_pair_marker(
                        config,
                        authority,
                        materializer,
                        live_source=source,
                        validation_digest=validation_digest,
                        live_identity=live_identity,
                        remote_identity=remote_identity,
                    )
            finally:
                with path.open("r+b") as handle:
                    handle.seek(0)
                    handle.write(first)
                    handle.flush()
                    os.fsync(handle.fileno())

        prior_owner_dir = prior_control.parent / "quack-owner"
        forbidden_artifacts = (
            prior_control.with_name(prior_control.name + ".wal"),
            prior_coordination.with_name(prior_coordination.name + ".wal"),
            prior_artifacts["execution"].with_name(
                prior_artifacts["execution"].name + ".wal"
            ),
            prior_artifacts["read replica"].with_name(
                prior_artifacts["read replica"].name + ".wal"
            ),
            prior_control.with_name(f".{prior_control.name}.state-owner.json"),
            prior_owner_dir / "quack-state-server.pid",
            prior_owner_dir / "quack-state-server.stop",
            prior_owner_dir / "quack-state-server.owner.json",
            prior_owner_dir / "forged.quack-token",
        )
        for path in forbidden_artifacts:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"forged\n")
            try:
                with pytest.raises(operator.OperatorError):
                    operator._require_m18_final_pair_marker(
                        config,
                        authority,
                        materializer,
                        live_source=source,
                        validation_digest=validation_digest,
                        live_identity=live_identity,
                        remote_identity=remote_identity,
                    )
            finally:
                path.unlink()

        with monkeypatch.context() as listener_patch:
            listener_patch.setattr(
                materializer,
                "_m18_prior_listener_is_active",
                lambda _port: True,
            )
            with pytest.raises(operator.OperatorError):
                operator._require_m18_final_pair_marker(
                    config,
                    authority,
                    materializer,
                    live_source=source,
                    validation_digest=validation_digest,
                    live_identity=live_identity,
                    remote_identity=remote_identity,
                )

        from ipfs_accelerate_py.agent_supervisor.merge import worktree_lifecycle

        for liveness in (
            worktree_lifecycle.OwnerLiveness.ALIVE,
            worktree_lifecycle.OwnerLiveness.UNKNOWN,
        ):
            with monkeypatch.context() as process_patch:
                process_patch.setattr(
                    worktree_lifecycle,
                    "owner_liveness",
                    lambda _identity, state=liveness: state,
                )
                with pytest.raises(operator.OperatorError):
                    operator._require_m18_final_pair_marker(
                        config,
                        authority,
                        materializer,
                        live_source=source,
                        validation_digest=validation_digest,
                        live_identity=live_identity,
                        remote_identity=remote_identity,
                    )

        original_prior_snapshot = materializer._m18_prior_artifact_anchor_snapshot
        snapshot_calls = 0

        def mutate_status_before_final_snapshot(
            root: Path,
            sealed_authority: Mapping[str, object],
        ) -> Mapping[str, object]:
            nonlocal snapshot_calls
            snapshot_calls += 1
            if snapshot_calls == 2:
                prior_artifacts["status"].write_bytes(b"forged-after-first-read\n")
            return original_prior_snapshot(root, sealed_authority)

        monkeypatch.setattr(
            materializer,
            "_m18_prior_artifact_anchor_snapshot",
            mutate_status_before_final_snapshot,
        )
        with pytest.raises(operator.OperatorError):
            operator._require_m18_final_pair_marker(
                config,
                authority,
                materializer,
                live_source=source,
                validation_digest=validation_digest,
                live_identity=live_identity,
                remote_identity=remote_identity,
            )
        shutil.copyfile(
            REPO_ROOT / authority["prior_stopped_status_projection_path"],
            prior_artifacts["status"],
        )
    finally:
        source.close()

    pristine_control = tmp_path / "m18-live-pristine.duckdb"
    shutil.copyfile(staged["stage_control"], pristine_control)
    tamper_cases: tuple[tuple[str, str, list[object]], ...] = (
        (
            "historical plan revision",
            "UPDATE plan_revisions SET body_json=? WHERE revision=18",
            ["{}"],
        ),
        (
            "historical evidence",
            "UPDATE evidence_nodes SET body_json=? WHERE evidence_id=("
            "SELECT evidence_id FROM evidence_nodes WHERE evidence_id<>? "
            "ORDER BY evidence_id LIMIT 1)",
            ["{}", staged["verified"]["migration_evidence_id"]],
        ),
        (
            "catalog",
            "CREATE VIEW forged_m18_catalog AS SELECT 1 AS value",
            [],
        ),
        (
            "unchecked stable metadata",
            "UPDATE control_plane_metadata SET value=? WHERE key='tool_version'",
            ["forged"],
        ),
        (
            "lifecycle",
            "UPDATE capability_snapshots SET profile_id=? WHERE server_id=?",
            ["forged-profile", live_identity["server_id"]],
        ),
        (
            "extra lifecycle row",
            "INSERT INTO state_servers SELECT ?,store_id,database_uuid,"
            "?,listen_uri,extension_fingerprint,schema_revision,"
            "generation,started_at,stopped_at,status,revision,extension_schema,"
            "extension_json FROM state_servers WHERE server_id=?",
            [
                "server:00000000-0000-4000-8000-000000000099",
                "birth:" + "9" * 32,
                live_identity["server_id"],
            ],
        ),
        (
            "intent tail",
            "UPDATE domain_events SET body_json=? WHERE global_sequence=225",
            ["{}"],
        ),
    )
    for noun, sql, parameters in tamper_cases:
        shutil.copyfile(pristine_control, staged["stage_control"])
        database = duckdb.connect(str(staged["stage_control"]))
        try:
            database.execute(sql, parameters)
            database.execute("CHECKPOINT")
            forged_receipt = dict(receipt)
            if noun in {"historical plan revision", "historical evidence"}:
                forged_receipt["append_surface_digest"] = (
                    materializer._authority_table_digest_on(
                        database,
                        (
                            "plans",
                            "plan_revisions",
                            "evidence_nodes",
                            "domain_events",
                        ),
                    )
                )
            elif noun == "catalog":
                forged_receipt["catalog_digest"] = (
                    materializer._main_catalog_digest_on(database)
                )
        finally:
            database.close()
        unhashed = dict(forged_receipt)
        unhashed.pop("receipt_cid")
        forged_receipt["receipt_cid"] = materializer._identity(unhashed)
        receipt_path.write_bytes(materializer._canonical(forged_receipt) + b"\n")
        tampered_source = DatabaseTaskSource(
            staged["stage_control"], install_schema=False
        )
        try:
            with pytest.raises(
                operator.OperatorError,
                match=(
                    "M18 live intent-event tail differs"
                    if noun == "intent tail"
                    else None
                ),
            ):
                operator._require_m18_final_pair_marker(
                    config,
                    authority,
                    materializer,
                    live_source=tampered_source,
                    validation_digest=validation_digest,
                    live_identity=live_identity,
                    remote_identity=remote_identity,
                )
        finally:
            tampered_source.close()
            receipt_path.write_bytes(materializer._canonical(receipt) + b"\n")


def test_m18_verifier_rejects_forged_evidence_parent(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m18_evidence_parent_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._expected_m18_portal_completion_persistence_authority()
    validation_digest = "sha256:" + "1" * 64
    stage = tmp_path / "stage"
    stage.mkdir()
    staged = materializer._stage_m18_store_pair(
        REPO_ROOT,
        stage,
        REPO_ROOT / authority["prior_store_id"],
        REPO_ROOT / authority["prior_coordination_store_id"],
        population,
        config,
        validation_digest,
    )

    import duckdb

    database = duckdb.connect(str(staged["stage_control"]))
    try:
        database.execute(
            "UPDATE evidence_nodes SET parent_evidence_id=? "
            "WHERE evidence_id=?",
            ["forged-parent", staged["verified"]["migration_evidence_id"]],
        )
        database.execute("CHECKPOINT")
    finally:
        database.close()
    with pytest.raises(
        materializer.MigrationRequired,
        match="plan/evidence append differs",
    ):
        materializer._verify_m18_store_pair_copy(
            staged["stage_control"],
            staged["stage_coordination"],
            staged["stage_prior_control"],
            staged["stage_prior_coordination"],
            population,
            config,
            validation_digest,
        )


def test_m16_accepted_source_retry_authority_is_presence_first_and_exact() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m16_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "accepted_source_retry_successor_materialization"
    authority = materializer._expected_m16_accepted_source_retry_authority()
    assert config[key] == authority
    assert inventory[key] == authority
    assert seal[f"{key}_cid"] == materializer._identity(authority)
    assert materializer._m16_successor_configured(config) is True
    malformed = dict(config)
    malformed[key] = None
    with pytest.raises(
        materializer.MaterializationError,
        match="M16 accepted-source retry authority is invalid",
    ):
        materializer._m16_successor_configured(malformed)


def test_m16_source_seal_binds_repair_blobs_and_both_exact_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m16_source_seal_test",
    )
    authority = materializer._expected_m16_accepted_source_retry_authority()
    prior = authority["prior_source_head"]
    repair = authority["repair_source_commit"]
    current = "c" * 40
    repair_paths = tuple(materializer._M16_ACCEPTED_REPAIR_BLOBS)
    operator_paths = tuple(sorted(materializer._M16_OPERATOR_CONTROL_PATHS))
    diffs = {
        (prior, repair): [f"M\t{path}" for path in repair_paths],
        (repair, current): [f"M\t{path}" for path in operator_paths],
        (prior, current): [
            *(f"M\t{path}" for path in repair_paths),
            *(f"M\t{path}" for path in operator_paths),
        ],
    }
    refs = {
        f"{prior}^{{tree}}": authority["prior_source_tree"],
        f"{repair}^{{tree}}": authority["repair_source_tree"],
        f"{current}^{{tree}}": "d" * 40,
        f"{prior}:ipfs_datasets_py": authority["prior_datasets_gitlink"],
        f"{current}:ipfs_datasets_py": authority["prior_datasets_gitlink"],
        f"{prior}:ipfs_kit_py": authority["prior_kit_gitlink"],
        f"{current}:ipfs_kit_py": authority["prior_kit_gitlink"],
    }
    for path, blob in materializer._M16_ACCEPTED_REPAIR_BLOBS.items():
        refs[f"{repair}:{path}"] = blob
        refs[f"{current}:{path}"] = blob

    def exact_git(_root: Path, *args: str, **_kwargs: object) -> str:
        if args[:2] == ("merge-base", "--is-ancestor"):
            return ""
        if args and args[0] == "rev-parse":
            return refs[args[1]]
        if args and args[0] == "diff":
            return "\n".join(diffs[(args[3], args[4])])
        raise AssertionError(f"unexpected M16 source-seal git call: {args}")

    monkeypatch.setattr(materializer, "_git", exact_git)
    population = {
        "source_binding": {
            "head": current,
            "tree": "d" * 40,
            "datasets_gitlink": authority["prior_datasets_gitlink"],
        }
    }
    materializer._assert_m16_source_delta(Path("/unused"), population, authority)

    first_path = repair_paths[0]
    refs[f"{current}:{first_path}"] = "0" * 40
    with pytest.raises(
        materializer.MaterializationError,
        match="accepted-source repair blob differs",
    ):
        materializer._assert_m16_source_delta(
            Path("/unused"), population, authority
        )
    refs[f"{current}:{first_path}"] = materializer._M16_ACCEPTED_REPAIR_BLOBS[
        first_path
    ]

    diffs[(repair, current)].append("M\tunexpected.py")
    with pytest.raises(
        materializer.MaterializationError,
        match="repair successor delta differs from the nine controls",
    ):
        materializer._assert_m16_source_delta(
            Path("/unused"), population, authority
        )


def test_m17_namespace_is_preserved_as_historical_under_m27() -> None:
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    authority = config["source_binding_successor_materialization"]
    root = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m17"
    assert authority["target_store_id"] == f"{root}/control.duckdb"
    assert authority["target_generation"] == 18
    assert authority["target_quack_port"] == 24_060
    assert config["runtime_paths"]["root"].endswith("run-r2-m27")
    # The M17 authority remains historical while M48 owns generation 35 in the
    # M27 runtime namespace.
    assert config["database_program"]["store_generation"] == "35"
    assert config["quack_owner"]["port"] == 24_070


def test_m18_validators_keep_m17_and_m16_historical_authority() -> None:
    dependency_validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_validator_m17_historical_m16_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )

    assert dependency_validator._m17_source_binding_successor_errors(
        config,
        seal,
        inventory,
        root=REPO_ROOT,
        require_active_runtime=False,
    ) == []
    assert dependency_validator._m17_source_binding_successor_errors(
        config,
        seal,
        inventory,
        root=REPO_ROOT,
    ) == ["scheduler M17 target/runtime binding is not exact"]
    assert dependency_validator._m16_accepted_source_retry_errors(
        config,
        seal,
        inventory,
        root=REPO_ROOT,
        require_active_runtime=False,
    ) == []
    assert dependency_validator._m16_accepted_source_retry_errors(
        config,
        seal,
        inventory,
        root=REPO_ROOT,
    ) == ["scheduler M16 target/runtime binding is not exact"]

    corrupted_history = copy.deepcopy(config)
    corrupted_history["accepted_source_retry_successor_materialization"][
        "worker_self_approval"
    ] = True
    assert dependency_validator._m16_accepted_source_retry_errors(
        corrupted_history,
        seal,
        inventory,
        root=REPO_ROOT,
        require_active_runtime=False,
    ) == ["M16 accepted-source retry authority differs across controls"]


def test_m16_disposable_stage_preserves_failures_and_rearms_exactly(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m16_stage_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._expected_m16_accepted_source_retry_authority()
    prior_control = REPO_ROOT / authority["prior_store_id"]
    prior_coordination = REPO_ROOT / authority["prior_coordination_store_id"]
    before = (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    result = materializer._stage_m16_store_pair(
        REPO_ROOT,
        stage,
        prior_control,
        prior_coordination,
        population,
        config,
        "sha256:m16-disposable-stage",
    )
    verified = result["verified"]
    assert verified["event_watermark"] == 209
    assert verified["projection_cid"] == authority["target_projection_cid"]
    assert set(verified["task_rearm_event_ids"]) == {"SAWM-003", "SAWM-004"}
    assert set(verified["coordination_rearm_event_ids"]) == {
        "SAWM-003",
        "SAWM-004",
    }
    assert verified["coordination_event_count"] == 674
    assert verified["coordination_projection_digest"] == authority[
        "target_coordination_projection_digest"
    ]
    assert (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    ) == before

    # The coordination projection intentionally excludes lease_events.  Prove
    # the deep verifier independently seals the exact task scope of the two
    # rearm events rather than trusting that projection alone.
    import duckdb

    connection = duckdb.connect(str(result["stage_coordination"]))
    try:
        connection.execute(
            "UPDATE lease_events SET scope_key='task:forged' "
            "WHERE observed_at_ms=?",
            [materializer._M16_COORDINATION_REARM_OBSERVED_AT_MS["SAWM-003"]],
        )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    with pytest.raises(
        materializer.MigrationRequired,
        match="SAWM-003 coordination rearm differs",
    ):
        materializer._verify_m16_store_pair_copy(
            result["stage_control"],
            result["stage_coordination"],
            result["stage_prior_control"],
            result["stage_prior_coordination"],
            population,
            config,
            "sha256:m16-disposable-stage",
        )


def test_m16_prior_anchor_is_verified_without_replay_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m16_prior_anchor_test",
    )
    authority = materializer._expected_m16_accepted_source_retry_authority()
    prior_control = REPO_ROOT / authority["prior_store_id"]
    prior_coordination = REPO_ROOT / authority["prior_coordination_store_id"]
    before = (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    replay_paths: list[Path] = []
    digest_paths: list[Path] = []
    original_replay = DatabaseTaskSource.projection_matches_events

    def tracked_replay(source: DatabaseTaskSource) -> bool:
        replay_paths.append(Path(source.database_path))
        return original_replay(source)

    monkeypatch.setattr(
        DatabaseTaskSource,
        "projection_matches_events",
        tracked_replay,
    )
    for name in (
        "_semantic_authority_digest",
        "_frozen_base_authority_digest",
        "_append_surface_digest",
    ):
        original_digest = getattr(materializer, name)

        def tracked_digest(path: Path, *, _digest=original_digest) -> str:
            digest_paths.append(Path(path))
            return _digest(path)

        monkeypatch.setattr(materializer, name, tracked_digest)

    assert materializer._assert_m16_prior_anchor(REPO_ROOT, authority) == (
        prior_control,
        prior_coordination,
    )
    assert len(replay_paths) == 1
    assert len(digest_paths) == 3
    assert len(set(digest_paths)) == 1
    assert replay_paths[0] not in set(digest_paths)
    assert (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    ) == before


def test_m16_live_head_uses_read_only_projection_inspection() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m16_live_inspector_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m16_live_inspector_test",
    )

    class ReadOnlySource:
        def snapshot(self) -> object:
            return type(
                "Snapshot",
                (),
                {
                    "event_cursor": materializer._M16_TARGET_EVENT_WATERMARK,
                    "projection_cid": materializer._M16_EXPECTED_PROJECTION_CID,
                },
            )()

        def get_plan(self, _plan_cid: str) -> dict[str, object]:
            return {"revision": materializer._M16_TARGET_PLAN_REVISION}

        def get_task(self, alias: str) -> object:
            failure = materializer._M16_FAILURE_RECEIPTS[alias]
            return type(
                "Task",
                (),
                {
                    "task_cid": failure["task_cid"],
                    "status": "retrying",
                    "revision": 5,
                    "body": {
                        "completion_receipt": materializer._m16_task_rearm_receipt(
                            alias
                        )
                    },
                },
            )()

        def projection_matches_events(self) -> bool:
            raise AssertionError("live inspection must not rebuild projections")

    inspected = materializer._inspect_m16_head_task_projection(
        ReadOnlySource(),
        {"plan_root_cid": "sha256:test-plan"},
    )
    assert inspected["event_watermark"] == materializer._M16_TARGET_EVENT_WATERMARK
    assert set(inspected["tasks"]) == {"SAWM-003", "SAWM-004"}

    class MaterializerStub:
        MigrationRequired = materializer.MigrationRequired

        @staticmethod
        def _inspect_m16_head_task_projection(
            _source: object, _population: object
        ) -> dict[str, object]:
            return {}

        @staticmethod
        def _verify_m16_head_task_projection(
            _source: object, _population: object
        ) -> dict[str, object]:
            raise AssertionError("live operator selected destructive replay")

    with pytest.raises(
        materializer.MigrationRequired,
        match="M16 live head projection differs",
    ):
        operator._verify_m16_live_head_task_projection(
            object(),
            {},
            MaterializerStub(),
        )


def test_m16_receipt_recovers_exact_hardlink_and_live_marker_hashes_coordination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m16_receipt_recovery_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m16_coordination_hash_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._expected_m16_accepted_source_retry_authority()
    historical_config = copy.deepcopy(config)
    historical_config.pop("portal_completion_persistence_successor_materialization")
    historical_config.pop("source_binding_successor_materialization")
    historical_config["runtime_paths"] = {
        "root": authority["target_runtime_root"],
        "state": f"{authority['target_runtime_root']}/state",
        "worktrees": f"{authority['target_runtime_root']}/worktrees",
        "merge_queue": f"{authority['target_runtime_root']}/merge-queue",
        "logs": f"{authority['target_runtime_root']}/logs",
        "generated_runtime_artifacts_are_completion_authority": False,
    }
    dependency_report = {"schema": "sawm/test-dependency@1", "valid": True}
    board_report = {"schema": "sawm/test-board@1", "valid": True}
    validation_digest = materializer._identity(
        {
            "dependency": dependency_report,
            "board": board_report,
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    staged = materializer._stage_m16_store_pair(
        REPO_ROOT,
        stage,
        REPO_ROOT / authority["prior_store_id"],
        REPO_ROOT / authority["prior_coordination_store_id"],
        population,
        historical_config,
        validation_digest,
    )
    target = tmp_path / authority["target_runtime_root"]
    target.mkdir(parents=True)
    control = target / "control.duckdb"
    coordination = target / "control.coordination.duckdb"
    shutil.copyfile(staged["stage_control"], control)
    shutil.copyfile(staged["stage_coordination"], coordination)
    prior_control = tmp_path / authority["prior_store_id"]
    prior_coordination = tmp_path / authority["prior_coordination_store_id"]
    prior_control.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / authority["prior_store_id"], prior_control)
    shutil.copyfile(
        REPO_ROOT / authority["prior_coordination_store_id"],
        prior_coordination,
    )
    verified = dict(staged["verified"])

    monkeypatch.setattr(
        materializer,
        "_m16_accepted_source_retry_authority",
        lambda *_args, **_kwargs: authority,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_committed_clean_source",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m16_source_delta",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m16_prior_anchor",
        lambda *_args, **_kwargs: (prior_control, prior_coordination),
    )
    monkeypatch.setattr(
        materializer,
        "_verify_m16_store_pair",
        lambda *_args, **_kwargs: dict(verified),
    )
    receipt = materializer._ensure_m16_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        historical_config,
        verified,
        validation_digest,
    )
    receipt_path = target / "migration-receipt.json"
    crash_alias = target / ".migration-receipt.json.123.tmp"
    os.link(receipt_path, crash_alias)
    assert receipt_path.stat().st_nlink == 2
    assert materializer._ensure_m16_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    ) == receipt
    assert receipt_path.stat().st_nlink == 1
    assert not crash_alias.exists()

    def validator_report(_root: Path, relative: str) -> dict[str, object]:
        if relative.endswith("dependencies.py"):
            return dependency_report
        if relative.endswith("board.py"):
            return board_report
        raise AssertionError(f"unexpected validator path: {relative}")

    monkeypatch.setattr(operator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(materializer, "build_population", lambda _root: population)
    monkeypatch.setattr(materializer, "_validator_report", validator_report)
    assert dict(
        operator._require_m16_final_pair_marker(
            historical_config, authority, materializer, checked=None
        )
    ) == receipt
    with coordination.open("ab") as handle:
        handle.write(b"tampered")
        handle.flush()
        os.fsync(handle.fileno())
    with pytest.raises(
        operator.OperatorError,
        match="M16 materialized final pair marker differs",
    ):
        operator._require_m16_final_pair_marker(
            historical_config, authority, materializer, checked=None
        )


def test_m13_quack_refresh_authority_is_closed_and_selected() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m13_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "quack_refresh_successor_materialization"
    authority = materializer._expected_m13_quack_refresh_authority()
    assert config[key] == migration[key] == authority
    assert materializer._identity(authority) == (
        materializer._M13_QUACK_REFRESH_AUTHORITY_CID
    )
    assert seal[f"{key}_cid"] == materializer._M13_QUACK_REFRESH_AUTHORITY_CID
    assert authority["schema"] == "sawm/quack-initial-refresh-repair-authorization@1"
    assert authority["target_generation"] == 14
    assert authority["target_quack_port"] == 24_056
    assert authority["target_plan_revision"] == 14
    assert authority["target_event_watermark"] == 191
    assert authority["target_projection_cid"] == (
        "baguqeeraqpkofyd3pnjnaqiwwefz7pkubim7kaklbp65puqu47ckb2vdmtxa"
    )
    assert authority["failed_quack_start"]["phase"] == (
        "pre_identity_initial_replica_bind"
    )
    assert authority["failed_quack_start"]["diagnosis"][
        "latent_lifecycle_hazard_statically_identified"
    ] is True
    assert materializer._m13_successor_configured(config) is True
    malformed = dict(config)
    malformed[key] = []
    with pytest.raises(
        materializer.MaterializationError,
        match="M13 Quack refresh authority is invalid",
    ):
        materializer._m13_successor_configured(malformed)


def test_m14_stale_owner_restart_authority_is_presence_first_and_bound() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m14_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    authority = materializer._expected_m14_stale_owner_restart_authority()
    key = "stale_owner_restart_successor_materialization"
    assert config[key] == inventory[key] == authority
    assert seal[f"{key}_cid"] == materializer._identity(authority)
    assert authority["schema"] == (
        "sawm/stale-owner-restart-repair-authorization@1"
    )
    assert authority["target_generation"] == 15
    assert authority["target_quack_port"] == 24_057
    assert authority["target_plan_revision"] == 15
    assert authority["target_event_watermark"] == 199
    assert authority["prior_database_uuid"] == (
        "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4"
    )
    assert authority["prior_generation"] == 14
    assert authority["prior_plan_revision"] == 14
    assert authority["prior_server_id"] == (
        "server:93f03ce5-e269-46ce-ab00-4d1f4ab269c9"
    )
    assert authority["prior_process_birth_id"] == (
        "birth:88dc7661014afc8d0d55a98d1d2debab"
    )
    assert authority["prior_startup_epoch"] == 1_787_964_894
    assert authority["prior_state_server_revision"] == 2
    assert authority["prior_stopped_at"] == "2026-08-29T07:57:24Z"
    assert authority["prior_owner_marker_present"] is False
    assert authority["prior_stopped_status_projection_present"] is True
    assert authority["prior_stopped_status_projection_sha256"] == (
        "03cced524345a6c66544d0f132fabd082f89ab1aa854a929e73a28052713cdc8"
    )
    assert authority["prior_stopped_status_projection_size"] == 2_491
    assert authority["prior_stale_owner_recovery_receipt_present"] is True
    assert authority["prior_stale_owner_recovery_receipt_sha256"] == (
        "1e5b080270abf275136e90c2c7beb268dd8d96f204c1fa7a56196b65be03d308"
    )
    assert authority["prior_stale_owner_recovery_receipt_size"] == 727
    assert authority["prior_stale_owner_recovery_cid"] == (
        "baguqeeranrymaea55inpt7sfio7rfibhqekv53jowpqkkhze4lzyt2kehcka"
    )
    assert authority["prior_migration_receipt_sha256"] == (
        "84cda5460119c2031c93ad3834a218bbd2598b5408a720f30fe44ce888aa7739"
    )
    assert authority["prior_migration_receipt_size"] == 4_407
    assert authority["prior_migration_receipt_cid"] == (
        "sha256:88054ecca35daf3e68f4beb82109025ed62c40c8350e744c866ad4914bb4aaad"
    )
    assert authority["prior_datasets_gitlink"] == (
        "58e5455a600d9b88e311842541d6612649d5b8cb"
    )
    assert authority["prior_datasets_tree"] == (
        "964b3e949b80191fe1b8e59af62df08f38857102"
    )
    assert authority["post_stop_datasets_gitlink"] == (
        "556a5978ec94a5e6706bfa5126332a838f7b7797"
    )
    assert authority["post_stop_datasets_tree"] == (
        "9b43261123240f2cb725e5d4903f36529e8fc180"
    )
    malformed = dict(config)
    malformed[key] = []
    with pytest.raises(
        materializer.MaterializationError,
        match="M14 stale-owner restart authority is invalid",
    ):
        materializer._m14_successor_configured(malformed)


def test_m14_prior_anchor_is_exact_read_only_and_preserves_m13_receipts() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m14_prior_anchor_test",
    )
    authority = materializer._expected_m14_stale_owner_restart_authority()
    paths = tuple(
        REPO_ROOT / authority[field]
        for field in (
            "prior_store_id",
            "prior_coordination_store_id",
            "prior_stopped_status_projection_path",
            "prior_stale_owner_recovery_receipt_path",
            "prior_migration_receipt_path",
        )
    )
    before = {
        path: (materializer._store_sha256(path), path.stat().st_size)
        for path in paths
    }

    control, coordination = materializer._assert_m14_prior_anchor(
        REPO_ROOT, authority
    )

    assert control == paths[0]
    assert coordination == paths[1]
    assert before == {
        path: (materializer._store_sha256(path), path.stat().st_size)
        for path in paths
    }
    assert not os.path.lexists(control.with_name(control.name + ".wal"))
    assert not os.path.lexists(
        coordination.with_name(coordination.name + ".wal")
    )
    assert not os.path.lexists(
        control.with_name(f".{control.name}.state-owner.json")
    )


def test_m14_disposable_stage_is_deeply_verified_and_forgery_closed(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m14_deep_pair_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._expected_m14_stale_owner_restart_authority()
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    prior_control = prior_dir / "control.duckdb"
    prior_coordination = prior_dir / "control.coordination.duckdb"
    shutil.copyfile(REPO_ROOT / authority["prior_store_id"], prior_control)
    shutil.copyfile(
        REPO_ROOT / authority["prior_coordination_store_id"],
        prior_coordination,
    )
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    validation_digest = "sha256:m14-disposable-deep-verification"
    staged = materializer._stage_m14_store_pair(
        REPO_ROOT,
        stage_dir,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    verified = materializer._verify_m14_store_pair_copy(
        staged["stage_control"],
        staged["stage_coordination"],
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    assert verified == staged["verified"]
    assert verified["event_watermark"] == 199
    assert verified["projection_cid"] == authority["target_projection_cid"]
    assert verified["plan_revision_changes"] == 1
    assert verified["evidence_node_changes"] == 1
    assert verified["task_revision_changes"] == 0
    assert verified["task_status_changes"] == 0
    assert verified["accepted_definition_changes"] == 0
    assert verified["accepted_completion_changes"] == 0
    assert verified["coordination_semantic_changes"] == 0

    import duckdb

    for case, statement, message in (
        (
            "immutable",
            "UPDATE tasks SET status='retrying' WHERE task_alias='SAWM-001'",
            "changed frozen control authority",
        ),
        (
            "plan",
            "UPDATE plans SET body_json='{}' WHERE revision=15",
            "plan/evidence append differs",
        ),
        (
            "evidence",
            "UPDATE evidence_nodes SET digest='sha256:forged' "
            "WHERE created_at='2026-08-29T08:00:00Z'",
            "plan/evidence append differs",
        ),
        (
            "event",
            "UPDATE domain_events SET task_cid='forged' "
            "WHERE global_sequence=199",
            "event (identity or envelope differs|stream bindings differ)",
        ),
    ):
        case_dir = tmp_path / case
        case_dir.mkdir()
        forged_control = case_dir / "control.duckdb"
        forged_coordination = case_dir / "control.coordination.duckdb"
        shutil.copyfile(staged["stage_control"], forged_control)
        shutil.copyfile(staged["stage_coordination"], forged_coordination)
        connection = duckdb.connect(str(forged_control))
        try:
            connection.execute(statement)
            connection.execute("CHECKPOINT")
        finally:
            connection.close()
        with pytest.raises(materializer.MigrationRequired, match=message):
            materializer._verify_m14_store_pair_copy(
                forged_control,
                forged_coordination,
                prior_control,
                prior_coordination,
                population,
                config,
                validation_digest,
            )


@pytest.mark.parametrize(
    ("relative", "directory"),
    (
        ("control.duckdb.wal", False),
        ("control.coordination.duckdb.wal", False),
        ("control.execution.duckdb", False),
        ("control.execution.duckdb.wal", False),
        ("control.read-replica.duckdb", False),
        ("control.read-replica.duckdb.wal", False),
        (".control.duckdb.state-owner.json", False),
        ("state", True),
        ("events", True),
        ("registry", True),
        ("worktrees", True),
        ("merge-queue", True),
        ("quack-owner", True),
    ),
)
def test_m14_target_rejects_every_mutable_sidecar_and_runtime_directory(
    tmp_path: Path,
    relative: str,
    directory: bool,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        f"sawm_materializer_m14_target_clean_{relative.replace('.', '_')}",
    )
    control = tmp_path / "control.duckdb"
    coordination = tmp_path / "control.coordination.duckdb"
    control.write_bytes(b"control")
    coordination.write_bytes(b"coordination")
    materializer._assert_m14_target_clean(control, coordination)
    forbidden = tmp_path / relative
    if directory:
        forbidden.mkdir()
    else:
        forbidden.write_bytes(b"forbidden")
    with pytest.raises(
        materializer.MigrationRequired,
        match="forbidden mutable sidecar",
    ):
        materializer._assert_m14_target_clean(control, coordination)


def test_m14_receipt_recovers_one_exact_pending_hardlink_and_rejects_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m14_receipt_recovery_test",
    )
    control = tmp_path / "control.duckdb"
    coordination = tmp_path / "control.coordination.duckdb"
    control.write_bytes(b"control")
    coordination.write_bytes(b"coordination")
    verified = {"valid": True}
    unhashed = {
        "schema": "sawm/non-authoritative-migration-receipt@12",
        "control_store_sha256": materializer._store_sha256(control),
        "control_store_size": control.stat().st_size,
        "coordination_store_sha256": materializer._store_sha256(coordination),
        "coordination_store_size": coordination.stat().st_size,
        "receipt_is_final_pair_commit_marker": True,
        "worker_self_approval": False,
    }
    expected = {**unhashed, "receipt_cid": materializer._identity(unhashed)}
    authority = materializer._expected_m14_stale_owner_restart_authority()
    monkeypatch.setattr(
        materializer,
        "_m14_stale_owner_restart_authority",
        lambda *_args, **_kwargs: authority,
    )
    monkeypatch.setattr(
        materializer, "_assert_committed_clean_source", lambda *_args: None
    )
    monkeypatch.setattr(
        materializer, "_assert_m14_source_delta", lambda *_args: None
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m14_prior_anchor",
        lambda *_args: (tmp_path / "prior", tmp_path / "prior-coordination"),
    )
    monkeypatch.setattr(
        materializer,
        "_verify_m14_store_pair",
        lambda *_args, **_kwargs: dict(verified),
    )
    monkeypatch.setattr(
        materializer,
        "_expected_m14_migration_receipt",
        lambda *_args, **_kwargs: dict(expected),
    )
    receipt_path = tmp_path / "migration-receipt.json"
    pending = tmp_path / ".migration-receipt.json.101.tmp"
    pending.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        materializer.MigrationRequired,
        match="pending M14 receipt differs",
    ):
        materializer._ensure_m14_migration_receipt(
            tmp_path,
            control,
            coordination,
            {},
            {},
            verified,
            "sha256:m14-receipt-test",
        )
    assert not receipt_path.exists()
    pending.write_bytes(materializer._canonical(expected) + b"\n")

    receipt = materializer._ensure_m14_migration_receipt(
        tmp_path,
        control,
        coordination,
        {},
        {},
        verified,
        "sha256:m14-receipt-test",
    )
    assert receipt == expected
    assert receipt_path.stat().st_nlink == 1
    assert not pending.exists()
    assert materializer._verify_existing_m14_migration_receipt(
        tmp_path,
        control,
        coordination,
        {},
        {},
        verified,
        "sha256:m14-receipt-test",
    ) == expected
    crash_alias = tmp_path / ".migration-receipt.json.202.tmp"
    os.link(receipt_path, crash_alias)
    assert receipt_path.stat().st_nlink == 2
    assert materializer._ensure_m14_migration_receipt(
        tmp_path,
        control,
        coordination,
        {},
        {},
        verified,
        "sha256:m14-receipt-test",
    ) == expected
    assert receipt_path.stat().st_nlink == 1
    assert not crash_alias.exists()

    forged = dict(expected)
    forged["worker_self_approval"] = True
    forged_unhashed = dict(forged)
    forged_unhashed.pop("receipt_cid")
    forged["receipt_cid"] = materializer._identity(forged_unhashed)
    receipt_path.write_bytes(materializer._canonical(forged) + b"\n")
    with pytest.raises(
        materializer.MigrationRequired,
        match="final pair marker differs",
    ):
        materializer._verify_existing_m14_migration_receipt(
            tmp_path,
            control,
            coordination,
            {},
            {},
            verified,
            "sha256:m14-receipt-test",
        )


def test_m14_recovers_only_inode_bound_private_staging_aliases(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m14_staging_alias_recovery_test",
    )
    stage = tmp_path / ".m14-installing.101.deadbeef"
    stage.mkdir()
    stage_control = stage / "control.duckdb"
    stage_coordination = stage / "control.coordination.duckdb"
    stage_control.write_bytes(b"control")
    stage_coordination.write_bytes(b"coordination")
    control = tmp_path / "control.duckdb"
    coordination = tmp_path / "control.coordination.duckdb"
    os.link(stage_control, control)
    os.link(stage_coordination, coordination)
    assert materializer._recover_m14_staging_aliases(control, coordination) is True
    assert not stage.exists()
    assert control.stat().st_nlink == 1
    assert coordination.stat().st_nlink == 1

    forged_stage = tmp_path / ".m14-installing.102.deadbeef"
    forged_stage.mkdir()
    (forged_stage / "control.duckdb").write_bytes(b"forged-control")
    (forged_stage / "control.coordination.duckdb").write_bytes(
        b"forged-coordination"
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="not an inode alias",
    ):
        materializer._recover_m14_staging_aliases(control, coordination)
    assert control.read_bytes() == b"control"
    assert coordination.read_bytes() == b"coordination"


def test_m14_partial_publication_revalidates_source_and_rolls_back_its_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m14_partial_publication_race_test",
    )
    authority = materializer._expected_m14_stale_owner_restart_authority()
    config = {
        "database_program": {
            "store_id": authority["target_store_id"],
            "store_generation": 15,
            "quack_endpoint": "quack:127.0.0.1:24057",
        },
        "quack_owner": {
            "database_path": authority["target_store_id"],
            "store_id": authority["target_store_id"],
            "port": 24_057,
        },
    }
    population = {"source_binding": {"head": "f" * 40}}
    prior_control = tmp_path / "prior-control.duckdb"
    prior_coordination = tmp_path / "prior-control.coordination.duckdb"
    prior_control.write_bytes(b"prior-control")
    prior_coordination.write_bytes(b"prior-coordination")
    checks: list[int] = []

    def assert_clean(_root: Path, _population: object) -> None:
        checks.append(len(checks) + 1)
        if len(checks) == 4:
            raise materializer.MaterializationError(
                "source binding changed after partial publication"
            )

    def stage_pair(
        _root: Path,
        stage_dir: Path,
        _prior_control: Path,
        _prior_coordination: Path,
        _population: object,
        _config: object,
        _validation_digest: str,
    ) -> dict[str, object]:
        stage_control = stage_dir / "control.duckdb"
        stage_coordination = stage_dir / "control.coordination.duckdb"
        stage_control.write_bytes(b"staged-control")
        stage_coordination.write_bytes(b"staged-coordination")
        return {
            "stage_control": stage_control,
            "stage_coordination": stage_coordination,
            "plan_receipt": SimpleNamespace(event_id="event:plan"),
            "evidence": SimpleNamespace(event_id="event:evidence"),
            "migration_digest": "sha256:" + "1" * 64,
            "verified": {"valid": True},
        }

    monkeypatch.setattr(materializer, "build_population", lambda _root: population)
    monkeypatch.setattr(
        materializer,
        "_m14_stale_owner_restart_authority",
        lambda *_args, **_kwargs: authority,
    )
    monkeypatch.setattr(materializer, "_assert_committed_clean_source", assert_clean)
    monkeypatch.setattr(
        materializer, "_assert_m14_source_delta", lambda *_args: None
    )
    monkeypatch.setattr(
        materializer,
        "_m7_validation_digest",
        lambda *_args: "sha256:" + "2" * 64,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m14_prior_anchor",
        lambda *_args: (prior_control, prior_coordination),
    )
    monkeypatch.setattr(materializer, "_stage_m14_store_pair", stage_pair)

    with pytest.raises(
        materializer.MaterializationError,
        match="source binding changed after partial publication",
    ):
        materializer._materialize_m14(
            tmp_path,
            tmp_path
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
            config,
        )
    target = (
        tmp_path
        / "data/agent_supervisor/semantic_addressed_world_model/run-r2-m14"
    )
    assert checks == [1, 2, 3, 4]
    assert not (target / "control.duckdb").exists()
    assert not (target / "control.coordination.duckdb").exists()
    assert not (target / "migration-receipt.json").exists()
    stages = list(target.glob(".m14-installing.*"))
    assert len(stages) == 1
    assert (stages[0] / "control.duckdb").read_bytes() == b"staged-control"
    assert (
        stages[0] / "control.coordination.duckdb"
    ).read_bytes() == b"staged-coordination"


def test_m15_runtime_root_authority_is_presence_first_and_exact() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m15_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "runtime_root_rebind_successor_materialization"
    authority = materializer._expected_m15_runtime_root_rebind_authority()
    assert config[key] == inventory[key] == authority
    assert seal[f"{key}_cid"] == materializer._identity(authority)
    assert materializer._m15_successor_configured(config) is True
    assert authority["target_generation"] == 16
    assert authority["target_quack_port"] == 24_058
    assert authority["target_plan_revision"] == 16
    assert authority["target_event_watermark"] == 201
    assert authority["target_projection_cid"] == (
        "baguqeeraaiqn3rqfg7gr4ks25n5ffjt3k4wcf4du56534qzyk7z3j7hewxbq"
    )
    assert authority["prior_control_store_sha256"] == (
        "af56f7c9af54b36755225d4cb4ef1c4e4e311f5c267077f87af9c4b4e88943ef"
    )
    assert authority["prior_frozen_base_authority_digest"] == (
        "sha256:96c3b52bb87fe9377e26fed3bd7e1976554db56a6301aaab356132e1aa55ebe3"
    )
    blocker = authority["detached_launch_blocker"]
    assert blocker["failure_kind"] == "historical_runtime_pid_collision"
    assert blocker["pid_liveness"] == "dead"
    assert blocker["provider_dispatched"] is False
    assert blocker["worker_dispatched"] is False
    malformed = dict(config)
    malformed[key] = []
    with pytest.raises(
        materializer.MaterializationError,
        match="M15 runtime-root rebind authority is invalid",
    ):
        materializer._m15_successor_configured(malformed)


def test_m15_historical_authority_preserves_fresh_namespace_under_m27() -> None:
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    key = "runtime_root_rebind_successor_materialization"
    authority = config[key]
    root = authority["target_runtime_root"]
    assert root == "data/agent_supervisor/semantic_addressed_world_model/run-r2-m15"
    assert authority["target_store_id"] == f"{root}/control.duckdb"
    assert authority["target_coordination_store_id"] == (
        f"{root}/control.coordination.duckdb"
    )
    assert authority["target_generation"] == 16
    assert authority["target_quack_port"] == 24_058
    assert authority["target_plan_revision"] == 16
    assert authority["target_event_watermark"] == 201
    historical_runtime = {
        "root": root,
        "state": f"{root}/state",
        "worktrees": f"{root}/worktrees",
        "merge_queue": f"{root}/merge-queue",
        "logs": f"{root}/logs",
        "generated_runtime_artifacts_are_completion_authority": False,
    }
    assert historical_runtime["root"] == authority["target_runtime_root"]
    assert config["runtime_paths"]["root"] == (
        "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
    )
    assert config["runtime_paths"] != historical_runtime
    # The M15 authority remains historical while M48 owns generation 35 in the
    # M27 runtime namespace.
    assert config["database_program"]["store_generation"] == "35"
    assert config["quack_owner"]["port"] == 24_070
    assert root != "data/agent_supervisor/semantic_addressed_world_model/run-r2-m13"


def test_m15_stopped_m14_anchor_is_exact_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m15_prior_anchor_test",
    )
    authority = materializer._expected_m15_runtime_root_rebind_authority()
    paths = tuple(
        REPO_ROOT / authority[field]
        for field in (
            "prior_store_id",
            "prior_coordination_store_id",
            "prior_stopped_status_projection_path",
            "prior_migration_receipt_path",
        )
    )
    pid_path = REPO_ROOT / authority["detached_launch_blocker"]["pid_path"]
    before = {
        path: (materializer._store_sha256(path), path.stat().st_size)
        for path in (*paths, pid_path)
    }
    token_scans: list[Path] = []
    original_token_scan = materializer._assert_m15_token_handoffs_absent

    def audited_token_scan(control: Path) -> None:
        token_scans.append(control)
        original_token_scan(control)

    monkeypatch.setattr(
        materializer,
        "_assert_m15_token_handoffs_absent",
        audited_token_scan,
    )
    control, coordination = materializer._assert_m15_prior_anchor(
        REPO_ROOT, authority
    )
    assert (control, coordination) == paths[:2]
    assert token_scans == [control, control]
    assert before == {
        path: (materializer._store_sha256(path), path.stat().st_size)
        for path in (*paths, pid_path)
    }
    assert not os.path.lexists(
        control.with_name(f".{control.name}.state-owner.json")
    )
    assert not os.path.lexists(control.with_name(control.name + ".wal"))
    assert not os.path.lexists(
        coordination.with_name(coordination.name + ".wal")
    )


def test_m15_prior_anchor_rejects_landed_quack_token_suffix(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m15_token_handoff_test",
    )
    control = tmp_path / "control.duckdb"
    owner = tmp_path / "quack-owner"
    owner.mkdir()
    legacy_wrong_name = owner / "quack-state-server.token.json"
    legacy_wrong_name.write_text("not-a-landed-handoff\n", encoding="utf-8")
    materializer._assert_m15_token_handoffs_absent(control)

    landed = owner / "env___SAWM_QUACK_TOKEN.quack-token"
    landed.write_text("secret-must-not-be-read\n", encoding="ascii")
    with pytest.raises(
        materializer.MigrationRequired,
        match="retained a Quack token handoff",
    ):
        materializer._assert_m15_token_handoffs_absent(control)


def test_m15_prior_anchor_rechecks_non_database_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m15_anchor_recheck_test",
    )
    authority = materializer._expected_m15_runtime_root_rebind_authority()
    original = materializer._stable_regular_sha256
    status_checks = 0

    def status_drift(*args: object, **kwargs: object) -> tuple[str, int]:
        nonlocal status_checks
        observed = original(*args, **kwargs)
        if kwargs.get("noun") == "stopped M14 status projection":
            status_checks += 1
            if status_checks == 2:
                return ("0" * 64, observed[1])
        return observed

    monkeypatch.setattr(materializer, "_stable_regular_sha256", status_drift)
    with pytest.raises(
        materializer.MaterializationError,
        match="predecessor verification mutated authority",
    ):
        materializer._assert_m15_prior_anchor(REPO_ROOT, authority)
    assert status_checks == 2


def test_m15_read_only_marker_rejects_pending_receipt_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m15_pending_receipt_test",
    )
    control = tmp_path / "control.duckdb"
    coordination = tmp_path / "control.coordination.duckdb"
    control.write_bytes(b"control")
    coordination.write_bytes(b"coordination")
    body = {
        "schema": "sawm/non-authoritative-migration-receipt@13",
        "receipt_is_final_pair_commit_marker": True,
    }
    expected = {**body, "receipt_cid": materializer._identity(body)}
    (tmp_path / "migration-receipt.json").write_bytes(
        materializer._canonical(expected) + b"\n"
    )
    monkeypatch.setattr(
        materializer,
        "_expected_m15_migration_receipt",
        lambda *_args, **_kwargs: expected,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m15_receipt_commit_inputs",
        lambda *_args, **_kwargs: None,
    )
    pending = tmp_path / ".migration-receipt.json.forged.tmp"
    pending.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        materializer.MigrationRequired,
        match="pending receipt temporary",
    ):
        materializer._verify_existing_m15_migration_receipt(
            tmp_path,
            control,
            coordination,
            {},
            {},
            {},
            "sha256:test",
        )
    pending.unlink()
    assert materializer._verify_existing_m15_migration_receipt(
        tmp_path,
        control,
        coordination,
        {},
        {},
        {},
        "sha256:test",
    ) == expected


def test_m15_disposable_stage_appends_only_events_200_and_201(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m15_stage_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._expected_m15_runtime_root_rebind_authority()
    prior_control = tmp_path / "prior.duckdb"
    prior_coordination = tmp_path / "prior.coordination.duckdb"
    shutil.copyfile(REPO_ROOT / authority["prior_store_id"], prior_control)
    shutil.copyfile(
        REPO_ROOT / authority["prior_coordination_store_id"],
        prior_coordination,
    )
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    validation_digest = "sha256:m15-disposable-stage"
    staged = materializer._stage_m15_store_pair(
        REPO_ROOT,
        stage_dir,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    verified = staged["verified"]
    assert verified["event_watermark"] == 201
    assert verified["projection_cid"] == authority["target_projection_cid"]
    assert verified["semantic_authority_digest"] == authority[
        "prior_semantic_authority_digest"
    ]
    assert verified["frozen_base_authority_digest"] == authority[
        "prior_frozen_base_authority_digest"
    ]
    assert verified["coordination_projection_digest"] == authority[
        "prior_coordination_projection_digest"
    ]
    assert verified["coordination_event_count"] == 484
    assert verified["plan_revision_changes"] == 1
    assert verified["evidence_node_changes"] == 1
    assert verified["task_revision_changes"] == 0
    assert verified["task_status_changes"] == 0


def test_m15_historical_final_marker_helper_survives_m16_supersession(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m15_live_marker_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m15_live_marker_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._expected_m15_runtime_root_rebind_authority()
    assert config["runtime_paths"]["root"] != authority["target_runtime_root"]
    historical_config = copy.deepcopy(config)
    historical_config["runtime_paths"] = {
        "root": authority["target_runtime_root"],
        "state": f"{authority['target_runtime_root']}/state",
        "worktrees": f"{authority['target_runtime_root']}/worktrees",
        "merge_queue": f"{authority['target_runtime_root']}/merge-queue",
        "logs": f"{authority['target_runtime_root']}/logs",
        "generated_runtime_artifacts_are_completion_authority": False,
    }
    dependency_report = {"schema": "sawm/test-dependency@1", "valid": True}
    board_report = {"schema": "sawm/test-board@1", "valid": True}
    validation_digest = materializer._identity(
        {
            "dependency": dependency_report,
            "board": board_report,
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    target = tmp_path / authority["target_runtime_root"]
    target.mkdir(parents=True)
    staged = materializer._stage_m15_store_pair(
        REPO_ROOT,
        target,
        REPO_ROOT / authority["prior_store_id"],
        REPO_ROOT / authority["prior_coordination_store_id"],
        population,
        historical_config,
        validation_digest,
    )
    control = staged["stage_control"]
    coordination = staged["stage_coordination"]
    verified = staged["verified"]
    monkeypatch.setattr(
        materializer,
        "_m15_runtime_root_rebind_authority",
        lambda *_args, **_kwargs: authority,
    )
    receipt = materializer._expected_m15_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        historical_config,
        verified,
        validation_digest,
    )
    receipt_path = target / "migration-receipt.json"
    receipt_path.write_bytes(materializer._canonical(receipt) + b"\n")

    def validator_report(
        _root: Path,
        relative_path: str,
    ) -> dict[str, object]:
        if relative_path.endswith("dependencies.py"):
            return dependency_report
        if relative_path.endswith("board.py"):
            return board_report
        raise AssertionError(f"unexpected validator path: {relative_path}")

    monkeypatch.setattr(operator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(materializer, "build_population", lambda _root: population)
    monkeypatch.setattr(materializer, "_validator_report", validator_report)
    checked = {
        **verified,
        "valid": True,
        "database_path": str(control.resolve()),
        "coordination_path": str(coordination.resolve()),
        "validation_digest": validation_digest,
        "receipt": receipt,
    }
    assert dict(
        operator._require_m15_final_pair_marker(
            historical_config, authority, materializer, checked=checked
        )
    ) == receipt

    owner = target / "quack-owner"
    owner.mkdir()
    (owner / "quack-state-server.status.json").write_text(
        '{"lifecycle":"running"}\n', encoding="utf-8"
    )

    def forbidden_deep_check(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("live final-marker validation deep-checked the store")

    monkeypatch.setattr(materializer, "check_materialized", forbidden_deep_check)
    monkeypatch.setattr(
        materializer,
        "_expected_m15_migration_receipt",
        forbidden_deep_check,
    )
    accepted = operator._require_m15_final_pair_marker(
        historical_config, authority, materializer, checked=None
    )
    assert dict(accepted) == receipt
    pending = target / ".migration-receipt.json.live-race.tmp"
    pending.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        operator.OperatorError,
        match="final pair marker",
    ):
        operator._require_m15_final_pair_marker(
            historical_config, authority, materializer, checked=None
        )


def test_operator_offline_check_falls_back_without_prior_authority_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m15_prior_authority_fallback_test",
    )
    authority = {"schema": "test/active-authority@1"}
    config = {"source_repair_materialization": authority}
    monkeypatch.setattr(
        operator, "_validator", lambda *_args, **_kwargs: {"valid": True}
    )
    monkeypatch.setattr(
        operator,
        "_active_source_repair_materialization",
        lambda _config: authority,
    )
    monkeypatch.setattr(
        operator, "_successor_materialization_configured", lambda _config: True
    )
    monkeypatch.setattr(
        operator,
        "_require_active_final_pair_marker",
        lambda *_args, **_kwargs: {},
    )
    fake_materializer = SimpleNamespace(
        build_population=lambda _root: {},
        _assert_committed_clean_source=lambda *_args: None,
        check_materialized=lambda *_args: {"valid": True, "action": "checked"},
    )
    monkeypatch.setattr(operator, "_materializer", lambda: fake_materializer)
    result = operator._validate_offline_quack_start(config)
    assert result["prior_authority"] == authority
    assert result["store"] == {"valid": True, "action": "checked"}


def test_m13_source_delta_and_target_binding_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m13_source_delta_test",
    )
    authority = materializer._expected_m13_quack_refresh_authority()
    population = {
        "source_binding": {"head": "f" * 40},
    }
    changed = list(authority["bounded_control_plane_repair_paths"])

    def exact_git(_root: Path, *args: str, **_kwargs: object) -> str:
        if args[:2] == ("merge-base", "--is-ancestor"):
            return ""
        if args and args[0] == "diff":
            return "\n".join(f"M\t{path}" for path in changed)
        raise AssertionError(f"unexpected M13 source-delta git call: {args}")

    monkeypatch.setattr(materializer, "_git", exact_git)
    materializer._assert_m13_source_delta(tmp_path, population, authority)
    changed.append("unexpected.py")
    with pytest.raises(
        materializer.MaterializationError,
        match="exact repair paths",
    ):
        materializer._assert_m13_source_delta(tmp_path, population, authority)
    config = {
        "database_program": {
            "store_id": authority["target_store_id"],
            "store_generation": "14",
            "quack_endpoint": "quack:127.0.0.1:24056",
        },
        "quack_owner": {
            "database_path": authority["target_store_id"],
            "store_id": authority["target_store_id"],
            "port": 24_056,
        },
    }
    control, coordination = materializer._m13_target_paths(
        tmp_path, config, authority
    )
    assert control.name == "control.duckdb"
    assert coordination == control.with_name("control.coordination.duckdb")


def test_m13_materializer_rechecks_source_after_private_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m13_source_race_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    authority = materializer._expected_m13_quack_refresh_authority()
    config["database_program"].update({
        "store_id": authority["target_store_id"],
        "store_generation": "14",
        "quack_endpoint": "quack:127.0.0.1:24056",
    })
    config["quack_owner"].update({
        "database_path": authority["target_store_id"],
        "store_id": authority["target_store_id"],
        "port": 24056,
    })
    population = {"source_binding": {"head": "f" * 40}}
    prior_control = tmp_path / "prior-control.duckdb"
    prior_coordination = tmp_path / "prior-control.coordination.duckdb"
    prior_control.write_bytes(b"prior-control")
    prior_coordination.write_bytes(b"prior-coordination")
    clean_checks: list[int] = []

    def assert_clean(_root: Path, _population: object) -> None:
        clean_checks.append(len(clean_checks) + 1)
        if len(clean_checks) == 2:
            raise materializer.MaterializationError(
                "source binding changed after private staging"
            )

    def stage_pair(
        _root: Path,
        stage_dir: Path,
        _prior_control: Path,
        _prior_coordination: Path,
        _population: object,
        _config: object,
        _validation_digest: str,
    ) -> dict[str, object]:
        stage_control = stage_dir / "control.duckdb"
        stage_coordination = stage_dir / "control.coordination.duckdb"
        stage_control.write_bytes(b"staged-control")
        stage_coordination.write_bytes(b"staged-coordination")
        return {
            "stage_control": stage_control,
            "stage_coordination": stage_coordination,
            "plan_receipt": SimpleNamespace(event_id="event:plan"),
            "evidence": SimpleNamespace(event_id="event:evidence"),
            "migration_digest": "sha256:" + "1" * 64,
            "verified": {"valid": True},
        }

    monkeypatch.setattr(materializer, "build_population", lambda _root: population)
    monkeypatch.setattr(
        materializer,
        "_m13_quack_refresh_authority",
        lambda _population, _config: authority,
    )
    monkeypatch.setattr(materializer, "_assert_committed_clean_source", assert_clean)
    monkeypatch.setattr(
        materializer,
        "_assert_m13_source_delta",
        lambda _root, _population, _authority: None,
    )
    monkeypatch.setattr(
        materializer,
        "_m7_validation_digest",
        lambda _root, _population: "sha256:" + "2" * 64,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m13_prior_publication_anchor",
        lambda _root, _authority, _population: (
            prior_control,
            prior_coordination,
            {"valid": True},
        ),
    )
    monkeypatch.setattr(materializer, "_stage_m13_store_pair", stage_pair)

    with pytest.raises(
        materializer.MaterializationError,
        match="source binding changed after private staging",
    ):
        materializer._materialize_m13(
            tmp_path,
            tmp_path
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
            config,
        )
    target = (
        tmp_path
        / "data/agent_supervisor/semantic_addressed_world_model/run-r2-m13"
    )
    assert clean_checks == [1, 2]
    assert not (target / "control.duckdb").exists()
    assert not (target / "control.coordination.duckdb").exists()
    assert not (target / "migration-receipt.json").exists()


def test_m13_receipt_gate_rechecks_offline_pair_and_rejects_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m13_receipt_gate_test",
    )
    control = tmp_path / "control.duckdb"
    coordination = tmp_path / "control.coordination.duckdb"
    control.write_bytes(b"control")
    coordination.write_bytes(b"coordination")
    expected = {
        "control_store_sha256": materializer._store_sha256(control),
        "control_store_size": control.stat().st_size,
        "coordination_store_sha256": materializer._store_sha256(coordination),
        "coordination_store_size": coordination.stat().st_size,
    }
    monkeypatch.setattr(materializer, "_assert_offline", lambda _path: None)
    materializer._assert_m13_receipt_commit_inputs(
        tmp_path, control, coordination, expected
    )
    control.with_name("control.read-replica.duckdb").write_bytes(b"forbidden")
    with pytest.raises(
        materializer.MigrationRequired,
        match="mutable sidecar appeared before final receipt",
    ):
        materializer._assert_m13_receipt_commit_inputs(
            tmp_path, control, coordination, expected
        )


def test_m13_quack_transport_defers_probe_until_post_identity_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m13_refresh_lifecycle_test",
    )
    transport = operator._SawmQuackTransport({})
    database = tmp_path / "control.duckdb"
    database.write_bytes(b"control")

    class Writer:
        path = str(database)

        def execute(self, sql: str) -> None:
            assert sql == "CHECKPOINT"

    class Replica:
        def __init__(self) -> None:
            self.closed = False
            self.statements: list[str] = []

        def execute(self, sql: str, _parameters: object = None) -> None:
            self.statements.append(sql)

        def close(self) -> None:
            self.closed = True

    replicas: list[Replica] = []
    probes: list[bool] = []
    monkeypatch.setattr(
        transport,
        "_copy_replica",
        lambda _source, target: {
            "path": str(target),
            "sha256": "0" * 64,
            "size_bytes": 7,
        },
    )

    def open_replica(_path: Path) -> Replica:
        replica = Replica()
        replicas.append(replica)
        return replica

    monkeypatch.setattr(transport, "_open_replica_connection", open_replica)
    monkeypatch.setattr(transport, "_probe", lambda: probes.append(True))
    transport._serve_uri = "http://127.0.0.1:24056"
    transport._owner_token = "test-token"
    transport._server_identity = {"generation": 14}

    initial = transport.refresh(Writer(), probe=False)
    assert initial["serve_started"] is True
    assert initial["probe_performed"] is False
    assert initial["live"] is False
    assert probes == []

    admitted = transport.refresh(Writer(), probe=True)
    assert replicas[0].closed is True
    assert any("quack_stop" in statement for statement in replicas[0].statements)
    assert admitted["probe_performed"] is True
    assert admitted["live"] is True
    assert probes == [True]

    start_probes: list[bool] = []
    monkeypatch.setattr(
        transport,
        "refresh",
        lambda _writer, *, probe=True: start_probes.append(probe)
        or {"probe_performed": probe, "live": probe},
    )
    identity = SimpleNamespace(
        server_id="server:test",
        store_id="store:test",
        database_uuid="database:test",
        schema_revision=1,
        schema_fingerprint="sha256:" + "1" * 64,
        generation=14,
        process_birth_id="birth:test",
    )
    started = transport.start(
        Writer(),
        host="127.0.0.1",
        port=24_056,
        token="test-token",
        identity=identity,
    )
    assert start_probes == [False]
    assert started["live"] is False

    from ipfs_accelerate_py.agent_supervisor.runtime import quack_state_server

    final_probes: list[bool] = []

    class FinalTransport:
        def refresh(self, _writer: object, *, probe: bool = True) -> None:
            final_probes.append(probe)

    class Server:
        transport = FinalTransport()
        _connection = object()

        def start(self) -> SimpleNamespace:
            return SimpleNamespace(to_dict=lambda: {"generation": 14})

        def ready(self) -> dict[str, object]:
            return {"ready": True}

        def stop(self) -> dict[str, object]:
            return {"stopped": True}

    server = Server()
    monkeypatch.setattr(
        quack_state_server,
        "build_server",
        lambda **_kwargs: server,
    )
    monkeypatch.setattr(
        operator,
        "_serve_sawm_owner",
        lambda observed: {"stopped": observed is server},
    )
    owner = {
        "database_path": "data/control.duckdb",
        "state_dir": "data/quack-owner",
        "host": "127.0.0.1",
        "port": 24_056,
        "repository_id": "repository:test",
        "store_id": "data/control.duckdb",
        "secret_handle": "env://SAWM_QUACK_TOKEN",
    }
    assert operator._start_quack({"quack_owner": owner}) == 0
    assert final_probes == [True]


def test_m13_operator_requires_the_exact_receipt_in_live_and_checked_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m13_operator_marker_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m13_final_marker_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    authority = materializer._expected_m13_quack_refresh_authority()
    config["database_program"].update({
        "store_id": authority["target_store_id"], "store_generation": "14",
        "quack_endpoint": "quack:127.0.0.1:24056",
    })
    config["quack_owner"].update({
        "database_path": authority["target_store_id"], "store_id": authority["target_store_id"], "port": 24056,
    })
    # Exercise the sealed historical M13 binding, which predates M24's
    # four-lane provider concurrency increase.
    config["provider"]["max_concurrency"] = 1
    population = materializer.build_population(REPO_ROOT)
    target = (
        tmp_path
        / "data/agent_supervisor/semantic_addressed_world_model/run-r2-m13"
    )
    target.mkdir(parents=True)
    control = target / "control.duckdb"
    coordination = target / "control.coordination.duckdb"
    control.write_bytes(b"control")
    coordination.write_bytes(b"coordination")

    dependency_report = {
        "schema": "sawm/test-dependency-validation@1",
        "valid": True,
    }
    board_report = {
        "schema": "sawm/test-board-validation@1",
        "valid": True,
    }
    validation_digest = materializer._identity(
        {
            "dependency": dependency_report,
            "board": board_report,
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    migration_body = materializer._m13_migration_body(
        population, config, validation_digest
    )
    migration_digest = materializer._identity(migration_body)
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    migration_evidence_id = content_identity(
        {
            "task_cid": population["migration_inventory"]["prior_task_cids"][
                "SAWM-000"
            ],
            "evidence_kind": "operator_control_plane_source_migration",
            "digest": migration_digest,
            "body": migration_body,
        }
    )
    verified = {
        "control_store_sha256": materializer._store_sha256(control),
        "coordination_store_sha256": materializer._store_sha256(coordination),
        "projection_cid": authority["target_projection_cid"],
        "event_watermark": authority["target_event_watermark"],
        "migration_digest": migration_digest,
        "migration_evidence_id": migration_evidence_id,
        "plan_migration_event_id": "baguqeera" + "1" * 52,
        "migration_evidence_event_id": "baguqeera" + "2" * 52,
        "coordination_projection_digest": authority[
            "target_coordination_projection_digest"
        ],
        "coordination_event_count": authority["target_coordination_event_count"],
        "semantic_authority_digest": authority["target_semantic_authority_digest"],
        "frozen_base_authority_digest": authority[
            "target_frozen_base_authority_digest"
        ],
        "append_surface_digest": "sha256:" + "3" * 64,
    }
    receipt = materializer._expected_m13_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    )
    assert set(receipt) == operator._M13_RECEIPT_KEYS
    assert len(receipt) == 69
    receipt_path = target / "migration-receipt.json"

    def validator_report(
        _root: Path,
        relative_path: str,
    ) -> dict[str, object]:
        if relative_path.endswith("dependencies.py"):
            return dependency_report
        if relative_path.endswith("board.py"):
            return board_report
        raise AssertionError(f"unexpected validator path: {relative_path}")

    monkeypatch.setattr(operator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(materializer, "build_population", lambda _root: population)
    monkeypatch.setattr(materializer, "_validator_report", validator_report)

    def reseal(marker: dict[str, object]) -> dict[str, object]:
        unhashed = dict(marker)
        unhashed.pop("receipt_cid", None)
        return {**unhashed, "receipt_cid": materializer._identity(unhashed)}

    def write_marker(marker: dict[str, object]) -> None:
        receipt_path.write_bytes(materializer._canonical(marker) + b"\n")

    def checked_for(marker: dict[str, object]) -> dict[str, object]:
        return {
            "valid": True,
            "database_path": str(control.resolve()),
            "coordination_path": str(coordination.resolve()),
            "projection_cid": authority["target_projection_cid"],
            "coordination_projection_digest": authority[
                "target_coordination_projection_digest"
            ],
            "event_watermark": authority["target_event_watermark"],
            "receipt": marker,
        }

    write_marker(receipt)
    for checked in (None, checked_for(receipt)):
        accepted = operator._require_m13_final_pair_marker(
            config, authority, materializer, checked=checked
        )
        assert dict(accepted) == receipt

    missing = dict(receipt)
    missing.pop("append_surface_digest")
    malformed = (
        reseal(missing),
        reseal({**receipt, "unexpected_field": True}),
        reseal({**receipt, "target_generation": 15}),
        reseal({**receipt, "implementation_provider_invocations": 1}),
        reseal({**receipt, "append_surface_digest": "sha256:invalid"}),
    )
    for marker in malformed:
        write_marker(marker)
        for checked in (None, checked_for(marker)):
            with pytest.raises(
                operator.OperatorError,
                match="M13 materialized final pair marker differs",
            ):
                operator._require_m13_final_pair_marker(
                    config, authority, materializer, checked=checked
                )


def test_m12_declared_output_retry_authority_is_closed() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m12_authority_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "declared_output_retry_successor_materialization"
    cid_key = "declared_output_retry_successor_materialization_cid"
    authority = materializer._expected_m12_declared_output_retry_authority()

    assert config[key] == migration[key] == authority
    assert materializer._identity(authority) == (
        materializer._M12_DECLARED_OUTPUT_RETRY_AUTHORITY_CID
    )
    assert seal[cid_key] == materializer._M12_DECLARED_OUTPUT_RETRY_AUTHORITY_CID
    assert authority["migration_revision"] == "SAWM-R2-M12"
    assert authority["prior_event_watermark"] == 186
    assert authority["target_event_watermark"] == 189
    assert authority["target_plan_revision"] == 13
    assert authority["target_generation"] == 14
    assert authority["target_quack_port"] == 45_256
    assert authority["task_rearm"]["from_status"] == "blocked"
    assert authority["task_rearm"]["from_revision"] == 15
    assert authority["task_rearm"]["to_status"] == "retrying"
    assert authority["task_rearm"]["to_revision"] == 16
    assert authority["prior_control_wal_present"] is False
    assert authority["prior_coordination_wal_present"] is False
    assert authority["execution_sidecar_copied"] is False
    assert authority["read_replica_sidecar_copied"] is False
    assert authority["accepted_definition_changes"] == 0
    assert authority["accepted_completion_changes"] == 0
    assert authority["implementation_provider_invocations_observed"] == 2
    assert authority["implementation_provider_model_calls_observed"] == 92
    assert authority["implementation_provider_tokens_observed"] == 11_202_083
    assert authority["implementation_provider_cost_usd_observed"] == "1.23726850"
    assert authority["settlement_provider_invocation_count"] == 0
    assert authority["provider_execution_accounting_mismatch"] is True
    observations = authority["live_implementation_failure"][
        "provider_execution_observations"
    ]
    assert observations == [
        {
            "route": "initial",
            "model_calls": 54,
            "tokens": 7_203_829,
            "cost_usd": "0.80458654",
        },
        {
            "route": "automatic_inline_rescue",
            "model_calls": 38,
            "tokens": 3_998_254,
            "cost_usd": "0.43268196",
        },
    ]
    assert set(authority["bounded_control_plane_repair_paths"]) == {
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        "config/semantic_addressed_world_model_dependencies.seal.json",
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
        (
            "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ),
        (
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
            "database_portal_bridge.py"
        ),
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "scripts/validate_semantic_addressed_world_model_board.py",
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
        "test/api/test_agent_supervisor_database_portal_bridge.py",
    }


def test_m12_receipt_commit_gate_rechecks_offline_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m12_receipt_commit_gate_test",
    )
    target = tmp_path / "data/agent_supervisor/semantic_addressed_world_model/run-r2-m12"
    target.mkdir(parents=True)
    control = target / "control.duckdb"
    coordination = target / "control.coordination.duckdb"
    control.write_bytes(b"control")
    coordination.write_bytes(b"coordination")
    expected = {
        "control_store_sha256": materializer._store_sha256(control),
        "control_store_size": control.stat().st_size,
        "coordination_store_sha256": materializer._store_sha256(coordination),
        "coordination_store_size": coordination.stat().st_size,
    }

    original_control = control.read_bytes()

    def mutate_during_offline_check(_path: Path) -> None:
        with control.open("ab") as handle:
            handle.write(b"-changed")
            handle.flush()
            os.fsync(handle.fileno())

    monkeypatch.setattr(
        materializer,
        "_assert_offline",
        mutate_during_offline_check,
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="store pair changed at receipt commit",
    ):
        materializer._assert_m12_receipt_commit_inputs(
            tmp_path,
            control,
            coordination,
            expected,
        )
    control.write_bytes(original_control)

    monkeypatch.undo()
    owner_marker = control.with_name(f".{control.name}.state-owner.json")
    owner_marker.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        materializer.MaterializationError,
        match="offline store verification refused",
    ):
        materializer._assert_m12_receipt_commit_inputs(
            tmp_path,
            control,
            coordination,
            expected,
        )


def test_m12_exact_pair_rehearsal_is_receipt_last_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m12_pair_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._m12_declared_output_retry_authority(
        population,
        config,
    )
    predecessor_fields = (
        "prior_store_id",
        "prior_coordination_store_id",
        "prior_materialization_receipt_path",
        "prior_owner_status_path",
        "prior_execution_store_id",
        "prior_read_replica_store_id",
    )
    original_paths = tuple(REPO_ROOT / authority[field] for field in predecessor_fields)
    original_hashes = {
        path: materializer._store_sha256(path) for path in original_paths
    }
    for field in predecessor_fields:
        source = REPO_ROOT / authority[field]
        copied = tmp_path / authority[field]
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copied)

    prior_control = tmp_path / authority["prior_store_id"]
    prior_coordination = tmp_path / authority["prior_coordination_store_id"]
    anchored = materializer._assert_m12_prior_publication_anchor(
        tmp_path,
        authority,
        population,
    )
    assert anchored[2]["event_watermark"] == 186
    assert anchored[2]["coordination_event_count"] == 223
    assert not prior_control.with_name(prior_control.name + ".wal").exists()
    assert not prior_coordination.with_name(
        prior_coordination.name + ".wal"
    ).exists()

    from ipfs_accelerate_py.agent_supervisor.task_sources import intent_repository

    original_clock = intent_repository._utc_iso
    stage = tmp_path / "m12-stage"
    stage.mkdir()
    validation_digest = "sha256:m12-disposable-pair-receipt"
    staged = materializer._stage_m12_store_pair(
        tmp_path,
        stage,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    assert intent_repository._utc_iso is original_clock
    control = tmp_path / authority["target_store_id"]
    coordination = tmp_path / authority["target_coordination_store_id"]
    control.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(staged["stage_control"], control)
    shutil.copyfile(staged["stage_coordination"], coordination)
    target_hashes = (
        materializer._store_sha256(control),
        materializer._store_sha256(coordination),
    )
    verified = materializer._verify_m12_store_pair(
        tmp_path,
        control,
        coordination,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    assert verified["projection_cid"] == materializer._M12_EXPECTED_PROJECTION_CID
    assert verified["event_watermark"] == 189
    assert verified["coordination_event_count"] == 224
    assert verified["task_revision_changes"] == 1
    assert verified["task_status_changes"] == 1
    assert verified["accepted_definition_changes"] == 0
    assert verified["accepted_completion_changes"] == 0
    assert verified["execution_sidecar_copied"] is False
    assert verified["read_replica_sidecar_copied"] is False
    assert target_hashes == (
        materializer._store_sha256(control),
        materializer._store_sha256(coordination),
    )

    changed_paths = list(authority["bounded_control_plane_repair_paths"])

    def exact_git(_root: Path, *args: str, **_kwargs: object) -> str:
        if args[:2] == ("merge-base", "--is-ancestor"):
            return ""
        if args and args[0] == "diff":
            return "\n".join(f"M\t{path}" for path in changed_paths)
        raise AssertionError(f"unexpected M12 source-delta git call: {args}")

    monkeypatch.setattr(materializer, "_git", exact_git)
    delta_population = copy.deepcopy(population)
    delta_population["source_binding"]["head"] = "f" * 40
    materializer._assert_m12_source_delta(
        tmp_path,
        delta_population,
        authority,
    )
    changed_paths.append("unexpected.py")
    with pytest.raises(
        materializer.MaterializationError,
        match="exact repair paths",
    ):
        materializer._assert_m12_source_delta(
            tmp_path,
            delta_population,
            authority,
        )
    changed_paths.pop()

    monkeypatch.setattr(
        materializer,
        "_assert_m12_source_delta",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_committed_clean_source",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m12_prior_publication_anchor",
        lambda *_args, **_kwargs: anchored,
    )
    monkeypatch.setattr(
        materializer,
        "_verify_m12_store_pair",
        lambda *_args, **_kwargs: dict(verified),
    )

    expected = materializer._expected_m12_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    )
    receipt_path = control.parent / "migration-receipt.json"
    pending = control.parent / ".migration-receipt.json.999999.tmp"
    pending.write_bytes(materializer._canonical(expected) + b"\n")
    receipt = materializer._ensure_m12_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    )
    assert receipt == expected
    assert receipt["schema"] == "sawm/non-authoritative-migration-receipt@10"
    assert receipt["receipt_is_final_pair_commit_marker"] is True
    assert receipt["prior_control_wal_present"] is False
    assert receipt["prior_coordination_wal_present"] is False
    assert receipt["execution_sidecar_copied"] is False
    assert receipt["read_replica_sidecar_copied"] is False
    assert receipt_path.stat().st_nlink == 1
    assert not pending.exists()
    assert materializer._ensure_m12_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    ) == receipt
    assert materializer._verify_existing_m12_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    ) == receipt

    crash_alias = control.parent / ".migration-receipt.json.123.tmp"
    os.link(receipt_path, crash_alias)
    assert materializer._ensure_m12_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    ) == receipt
    assert receipt_path.stat().st_nlink == 1
    assert not crash_alias.exists()

    execution_sidecar = control.with_name("control.execution.duckdb")
    execution_sidecar.write_bytes(b"forbidden")
    with pytest.raises(
        materializer.MigrationRequired,
        match="mutable sidecar appeared before final receipt",
    ):
        materializer._ensure_m12_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    execution_sidecar.unlink()

    pristine_receipt = receipt_path.read_bytes()
    tampered = json.loads(pristine_receipt)
    tampered["worker_self_approval"] = True
    receipt_path.write_bytes(materializer._canonical(tampered) + b"\n")
    with pytest.raises(
        materializer.MigrationRequired,
        match="final pair marker differs",
    ):
        materializer._verify_existing_m12_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    receipt_path.write_bytes(pristine_receipt)
    assert original_hashes == {
        path: materializer._store_sha256(path) for path in original_paths
    }
    assert target_hashes == (
        materializer._store_sha256(control),
        materializer._store_sha256(coordination),
    )

    # The operator consumes the materializer's exact 72-field payload plus
    # receipt CID. Exercise both the offline-check binding and the live-owned
    # marker path with this disposable pair, without duplicating its schema.
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m12_final_marker_test",
    )
    m12_operator_config = copy.deepcopy(config)
    m12_operator_config.pop("quack_refresh_successor_materialization")
    m12_operator_config["database_program"].update(
        {
            "quack_endpoint": "quack:127.0.0.1:45256",
            "store_id": authority["target_store_id"],
            "store_generation": "14",
        }
    )
    m12_operator_config["quack_owner"].update(
        {
            "database_path": authority["target_store_id"],
            "port": 45_256,
            "store_id": authority["target_store_id"],
        }
    )
    # Exercise the sealed historical M12 binding, which predates M24's
    # four-lane provider concurrency increase.
    m12_operator_config["provider"]["max_concurrency"] = 1
    dependency_report = {
        "schema": "sawm/test-dependency-validation@1",
        "valid": True,
    }
    board_report = {
        "schema": "sawm/test-board-validation@1",
        "valid": True,
    }
    operator_validation_digest = materializer._identity(
        {
            "dependency": dependency_report,
            "board": board_report,
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    operator_migration_body = materializer._m12_migration_body(
        population,
        m12_operator_config,
        operator_validation_digest,
    )
    operator_migration_digest = materializer._identity(operator_migration_body)
    operator_receipt = dict(receipt)
    operator_receipt["validation_digest"] = operator_validation_digest
    operator_receipt["migration_digest"] = operator_migration_digest
    operator_receipt["migration_evidence_id"] = content_identity(
        {
            "task_cid": population["migration_inventory"]["prior_task_cids"][
                "SAWM-000"
            ],
            "evidence_kind": "operator_control_plane_source_migration",
            "digest": operator_migration_digest,
            "body": operator_migration_body,
        }
    )
    operator_unhashed = dict(operator_receipt)
    operator_unhashed.pop("receipt_cid")
    operator_receipt["receipt_cid"] = materializer._identity(operator_unhashed)
    assert len(operator_unhashed) == 72
    assert len(operator_receipt) == 73

    def validator_report(
        _root: Path,
        relative_path: str,
    ) -> dict[str, object]:
        if relative_path.endswith("dependencies.py"):
            return dependency_report
        if relative_path.endswith("board.py"):
            return board_report
        raise AssertionError(f"unexpected validator path: {relative_path}")

    monkeypatch.setattr(operator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(materializer, "build_population", lambda _root: population)
    monkeypatch.setattr(materializer, "_validator_report", validator_report)

    def reseal(marker: dict[str, object]) -> dict[str, object]:
        unhashed = dict(marker)
        unhashed.pop("receipt_cid", None)
        return {**unhashed, "receipt_cid": materializer._identity(unhashed)}

    def checked_for(marker: dict[str, object]) -> dict[str, object]:
        return {
            "valid": True,
            "database_path": str(control.resolve()),
            "coordination_path": str(coordination.resolve()),
            "projection_cid": authority["target_projection_cid"],
            "coordination_projection_digest": authority[
                "target_coordination_projection_digest"
            ],
            "event_watermark": authority["target_event_watermark"],
            "receipt": marker,
        }

    def write_marker(marker: dict[str, object]) -> None:
        receipt_path.write_bytes(materializer._canonical(marker) + b"\n")

    write_marker(operator_receipt)
    for checked in (checked_for(operator_receipt), None):
        accepted = operator._require_m12_final_pair_marker(
            m12_operator_config,
            authority,
            materializer,
            checked=checked,
        )
        assert dict(accepted) == operator_receipt

    missing_key = dict(operator_receipt)
    missing_key.pop("append_surface_digest")
    malformed_markers = [
        reseal(missing_key),
        reseal({**operator_receipt, "unexpected_field": True}),
        reseal({**operator_receipt, "authoritative": True}),
        reseal(
            {
                **operator_receipt,
                "current_source_binding_cid": "sha256:" + "0" * 64,
            }
        ),
        reseal(
            {
                **operator_receipt,
                "validation_digest": "sha256:" + "0" * 64,
            }
        ),
        reseal(
            {
                **operator_receipt,
                "task_rearm_receipt_cid": "baguqeera" + "a" * 52,
            }
        ),
    ]
    for malformed_marker in malformed_markers:
        write_marker(malformed_marker)
        for checked in (checked_for(malformed_marker), None):
            with pytest.raises(
                operator.OperatorError,
                match="M12 materialized final pair marker differs",
            ):
                operator._require_m12_final_pair_marker(
                    m12_operator_config,
                    authority,
                    materializer,
                    checked=checked,
                )
    receipt_path.write_bytes(pristine_receipt)
