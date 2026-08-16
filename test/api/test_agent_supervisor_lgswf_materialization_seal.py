"""Focused qualification for the two-stage LGSWF bootstrap authority.

The fixtures deliberately redirect the materializer to ``tmp_path``.  They do
not construct, inspect, or mutate the configured ``run-actual-v3`` namespace.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
)

ROOT = Path(__file__).resolve().parents[2]
MATERIALIZER = ROOT / "scripts/materialize_logic_governed_semantic_work_fabric_control_plane.py"
requires_duckdb = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for the temporary LGSWF control-plane fixture",
)


def _load_materializer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"lgswf_materializer_test_{id(object())}", MATERIALIZER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _temporary_config() -> dict[str, Any]:
    return {
        "database_program": {
            "store_id": "runtime/control.duckdb",
            "schema_revision": "datasets-authoritative-operational-v1",
            "schema_profile": "datasets-authoritative-operational",
            "semantic_relations_permitted": False,
        },
        "runtime_paths": {"evidence": "runtime/evidence"},
        "initial_projection": {
            "ready_task_ids": ["LGSWF-006"],
            "completed_task_ids": [],
        },
    }


def _temporary_population() -> dict[str, Any]:
    aliases = ("LGSWF-006", "LGSWF-001", "LGSWF-002", "LGSWF-003")
    tasks: list[dict[str, Any]] = []
    for ordinal, alias in enumerate(aliases, start=1):
        task: dict[str, Any] = {
            "task_cid": f"task:{alias}",
            "task_id": alias,
            "task_alias": alias,
            "goal_cid": "goal:lgswf-root",
            "plan_cid": "plan:lgswf-test",
            "status": "todo",
            "priority": "P0",
            "ordinal": ordinal,
            "title": alias,
            "dependencies": ([] if alias == "LGSWF-006" else ["task:LGSWF-006"]),
        }
        if alias == "LGSWF-006":
            task.update(
                {
                    "completion": "manual",
                    "review_only": "true",
                    "is_schedulable": "false",
                }
            )
        tasks.append(task)
    return {
        "repository_tree_id": "tree:lgswf-test",
        "source_head": "head:lgswf-test",
        "plan_root_cid": "plan:lgswf-test",
        "objectives": [
            {
                "objective_id": "objective:lgswf-root",
                "objective_alias": "LGSWF-G000",
                "title": "LGSWF root",
                "goal_cid": "goal:lgswf-root",
                "goal_alias": "LGSWF-G000",
                "status": "open",
            }
        ],
        "plans": [
            {
                "plan_cid": "plan:lgswf-test",
                "plan_alias": "LGSWF-PLAN-TEST",
                "goal_cid": "goal:lgswf-root",
                "status": "active",
            }
        ],
        "tasks": tasks,
        "task_cids_by_alias": {alias: f"task:{alias}" for alias in aliases},
        "goal_cids_by_alias": {"LGSWF-G000": "goal:lgswf-root"},
    }


def _materialize_temporary_plane(
    tmp_path: Path,
) -> tuple[ModuleType, dict[str, Any], dict[str, Any], dict[str, Any]]:
    module = _load_materializer()
    module.ROOT = tmp_path
    config = _temporary_config()
    population = _temporary_population()
    receipt = module.materialize(config, population)
    return module, config, population, receipt


def _stub_launch_evidence() -> dict[str, Any]:
    return {
        "launch_plan_cid": "sha256:" + "1" * 64,
        "schema": "lgswf-test-launch-plan@1",
        "authority_mode": "embedded",
        "task_source_kind": "duckdb",
        "schema_revision": "datasets-authoritative-operational-v1",
        "configured_schema_profile": "datasets-authoritative-operational",
        "semantic_relations_permitted": False,
        "lanes": 1,
        "admitted_lanes": 1,
        "plan_bound_dispatch": False,
        "effective_strict_task_sharding": True,
        "plan_bound_promotion_task": "LGSWF-005",
        "implement": True,
        "process_started": False,
    }


def test_dirty_worktree_fails_closed_before_population_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_materializer()
    calls: list[list[str]] = []

    def dirty_status(argv: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        return SimpleNamespace(stdout=" M docs/architecture/board.md\n")

    monkeypatch.setattr(module.subprocess, "run", dirty_status)

    with pytest.raises(
        module.MaterializationError,
        match="dirty execution worktree",
    ):
        module.build_population({})

    assert calls == [["git", "status", "--porcelain=v1", "--untracked-files=all"]]


def test_actual_launch_evidence_reports_bounded_legacy_dispatch_truthfully() -> None:
    module = _load_materializer()
    config = module._load_config()

    evidence = module._render_launch_plan_evidence(config)

    assert evidence["authority_mode"] == "embedded"
    assert evidence["task_source_kind"] == "duckdb"
    assert evidence["lanes"] == 1
    assert evidence["admitted_lanes"] == 1
    assert evidence["effective_strict_task_sharding"] is True
    assert evidence["plan_bound_dispatch"] is False
    assert evidence["plan_bound_promotion_task"] == "LGSWF-005"
    assert evidence["implement"] is True
    assert evidence["process_started"] is False


@requires_duckdb
def test_unsealed_plane_exposes_only_manual_seal_and_daemon_skips_it(
    tmp_path: Path,
) -> None:
    module, config, population, receipt = _materialize_temporary_plane(tmp_path)

    verification = receipt["verification"]
    assert verification["bootstrap_stage"] == "unsealed"
    assert verification["ready_task_aliases"] == ["LGSWF-006"]
    assert verification["completed_task_aliases"] == []
    assert verification["active_task_claim_count"] == 0

    paths = module._paths(config)
    daemon = DatabaseImplementationDaemon(
        database_path=paths["control"],
        coordination_path=paths["coordination"],
        execution_path=paths["execution"],
        owner_session_id="lgswf-test-automatic-daemon",
        authority_mode="embedded",
        task_source_kind="duckdb",
        install_schema=False,
    )
    try:
        assert daemon._automatic_claim_exclusions() == {"task:LGSWF-006"}
        assert daemon.claim_next() is None
        manual = daemon.task_source.get(population["task_cids_by_alias"]["LGSWF-006"])
        assert manual is not None
        assert manual.status == "todo"
    finally:
        daemon.close()


@requires_duckdb
def test_trusted_seal_prepares_cas_promotes_settles_and_replays_immutably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, _receipt = _materialize_temporary_plane(tmp_path)
    monkeypatch.setattr(
        module, "_render_launch_plan_evidence", lambda _config: _stub_launch_evidence()
    )
    monkeypatch.setattr(module, "_sha256_file", lambda _path: "sha256:" + "2" * 64)

    sealed = module.seal(config, population)
    accepted_result_cid = sealed["accepted_result_cid"]
    assert sealed["preparation"]["status"] == "prepared"
    assert sealed["preparation"]["evidence_digest"] == accepted_result_cid
    assert sealed["control_cas"]["changed"] is True
    assert sealed["control_cas"]["task"]["status"] == "completed"
    assert (
        sealed["control_cas"]["task"]["body"]["completion_receipt"]["accepted_result_cid"]
        == accepted_result_cid
    )
    assert sealed["coordination_promotion"]["status"] == "succeeded"
    assert sealed["settled_lease"]["state"] == "released"
    assert sealed["post_verification"]["accepted_result_cid"] == accepted_result_cid
    assert sealed["post_verification"]["ready_task_aliases"] == [
        "LGSWF-001",
        "LGSWF-002",
        "LGSWF-003",
    ]

    receipt_path = module._bootstrap_receipt_path(config, "duckdb-seal.json")
    stored_before = receipt_path.read_bytes()
    persisted = json.loads(stored_before)
    claimed_receipt_cid = persisted.pop("receipt_cid")
    assert module._identity(persisted) == claimed_receipt_cid

    replay = module.seal(config, population)
    assert replay["replayed"] is True
    assert replay["receipt_cid"] == sealed["receipt_cid"]
    assert replay["accepted_result_cid"] == accepted_result_cid
    assert receipt_path.read_bytes() == stored_before


@requires_duckdb
def test_post_settlement_reconstruction_writes_identity_default_verify_accepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, config, population, _receipt = _materialize_temporary_plane(tmp_path)
    monkeypatch.setattr(
        module, "_render_launch_plan_evidence", lambda _config: _stub_launch_evidence()
    )
    monkeypatch.setattr(module, "_sha256_file", lambda _path: "sha256:" + "2" * 64)

    sealed = module.seal(config, population)
    accepted_result_cid = sealed["accepted_result_cid"]
    assert sealed["coordination_promotion"]["status"] == "succeeded"
    assert sealed["settled_lease"]["state"] == "released"
    assert (
        module._verify_store(config, population, expected_stage="sealed")["accepted_result_cid"]
        == accepted_result_cid
    )

    receipt_path = module._bootstrap_receipt_path(config, "duckdb-seal.json")
    receipt_path.unlink()
    assert not receipt_path.exists()

    reconstructed = module.seal(config, population)
    assert reconstructed["accepted_result_cid"] == accepted_result_cid
    assert reconstructed["post_verification"]["accepted_result_cid"] == (accepted_result_cid)
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted["accepted_result_cid"] == accepted_result_cid
    claimed_receipt_cid = persisted.pop("receipt_cid")
    assert module._identity(persisted) == claimed_receipt_cid

    monkeypatch.setattr(module, "_load_config", lambda: config)
    monkeypatch.setattr(module, "build_population", lambda _config: population)
    assert module.main(["verify"]) == 0
    verification = json.loads(capsys.readouterr().out)
    assert verification["valid"] is True
    assert verification["seal_receipt_cid"] == claimed_receipt_cid


@requires_duckdb
def test_self_consistent_receipt_with_wrong_result_identity_fails_seal_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, _receipt = _materialize_temporary_plane(tmp_path)
    monkeypatch.setattr(
        module, "_render_launch_plan_evidence", lambda _config: _stub_launch_evidence()
    )
    monkeypatch.setattr(module, "_sha256_file", lambda _path: "sha256:" + "2" * 64)

    sealed = module.seal(config, population)
    receipt_path = module._bootstrap_receipt_path(config, "duckdb-seal.json")
    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered.pop("receipt_cid")
    tampered["accepted_result_cid"] = "sha256:" + "f" * 64
    assert tampered["accepted_result_cid"] != sealed["accepted_result_cid"]
    tampered["receipt_cid"] = module._identity(tampered)
    module._write_receipt(receipt_path, tampered)

    loaded = module._load_existing_seal_receipt(config, population)
    assert loaded is not None
    assert loaded["receipt_cid"] == tampered["receipt_cid"]
    assert loaded["accepted_result_cid"] == tampered["accepted_result_cid"]
    with pytest.raises(
        module.MaterializationError,
        match="existing bootstrap seal disagrees with control authority",
    ):
        module.seal(config, population)


@pytest.mark.parametrize(
    ("seal_receipt", "expected_error"),
    [
        (None, "accepted bootstrap seal receipt is absent"),
        (
            {
                "receipt_cid": "sha256:" + "3" * 64,
                "accepted_result_cid": "sha256:" + "4" * 64,
            },
            "bootstrap seal receipt disagrees with control authority",
        ),
    ],
)
def test_sealed_verification_requires_matching_receipt_identity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    seal_receipt: dict[str, str] | None,
    expected_error: str,
) -> None:
    module = _load_materializer()
    config = _temporary_config()
    population = _temporary_population()
    accepted_result_cid = "sha256:" + "5" * 64
    monkeypatch.setattr(module, "_load_config", lambda: config)
    monkeypatch.setattr(module, "build_population", lambda _config: population)
    monkeypatch.setattr(
        module,
        "_verify_store",
        lambda *_args, **_kwargs: {"accepted_result_cid": accepted_result_cid},
    )
    monkeypatch.setattr(
        module,
        "_load_existing_seal_receipt",
        lambda *_args, **_kwargs: seal_receipt,
    )

    assert module.main(["verify"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is False
    assert result["error"] == expected_error


def test_sealed_verification_accepts_only_the_exact_result_identity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_materializer()
    config = _temporary_config()
    population = _temporary_population()
    accepted_result_cid = "sha256:" + "6" * 64
    seal_receipt = {
        "receipt_cid": "sha256:" + "7" * 64,
        "accepted_result_cid": accepted_result_cid,
    }
    monkeypatch.setattr(module, "_load_config", lambda: config)
    monkeypatch.setattr(module, "build_population", lambda _config: population)
    monkeypatch.setattr(
        module,
        "_verify_store",
        lambda *_args, **_kwargs: {"accepted_result_cid": accepted_result_cid},
    )
    monkeypatch.setattr(
        module,
        "_load_existing_seal_receipt",
        lambda *_args, **_kwargs: seal_receipt,
    )

    assert module.main(["verify"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["seal_receipt_cid"] == seal_receipt["receipt_cid"]
