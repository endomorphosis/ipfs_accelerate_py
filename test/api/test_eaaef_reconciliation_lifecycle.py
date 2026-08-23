from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import (
    eaaef_reconciliation_lifecycle as lifecycle,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sealed_forest(*, accelerator_commit: str = "1" * 40) -> dict[str, Any]:
    repositories = [
        {
            "name": "ipfs_accelerate_py",
            "relative_path": ".",
            "commit": accelerator_commit,
            "tree": "a" * 40,
            "gitlink": False,
            "initialized": True,
            "clean": True,
        },
        {
            "name": "ipfs_datasets_py",
            "relative_path": "ipfs_datasets_py",
            "commit": "2" * 40,
            "tree": "b" * 40,
            "gitlink": True,
            "initialized": True,
            "clean": True,
        },
        {
            "name": "ipfs_kit_py",
            "relative_path": "ipfs_kit_py",
            "commit": "3" * 40,
            "tree": "c" * 40,
            "gitlink": True,
            "initialized": True,
            "clean": True,
        },
        {
            "name": "mcpplusplus",
            "relative_path": "ipfs_accelerate_py/mcplusplus",
            "commit": "4" * 40,
            "tree": "d" * 40,
            "gitlink": True,
            "initialized": True,
            "clean": True,
        },
    ]
    board_bytes = (
        Path(__file__).resolve().parents[2] / lifecycle.EAAEF_BOARD_PATH
    ).read_bytes()
    blob_oid = hashlib.sha1(
        b"blob " + str(len(board_bytes)).encode("ascii") + b"\0" + board_bytes,
        usedforsecurity=False,
    ).hexdigest()
    board_source = lifecycle._board_source_binding(
        board_bytes,
        source_head=repositories[0]["commit"],
        source_tree=repositories[0]["tree"],
        git_mode="100644",
        blob_oid=blob_oid,
    )
    identity = {
        "schema": lifecycle.EAAEF_FOREST_SCHEMA,
        "repositories": repositories,
        "board_source": board_source,
    }
    root = lifecycle._cid(identity)
    return {
        **identity,
        "valid": True,
        "blockers": [],
        "source_head": repositories[0]["commit"],
        "source_tree": repositories[0]["tree"],
        "source_forest_root": root,
        "source_generation_cid": root,
        "binding_cid": lifecycle._cid({**identity, "source_forest_root": root}),
    }


def _board(repo_root: Path) -> dict[str, Any]:
    return json.loads((repo_root / lifecycle.EAAEF_BOARD_PATH).read_text(encoding="utf-8"))


def _population(repo_root: Path) -> lifecycle.CompiledEAAEFPopulation:
    return lifecycle.compile_fresh_eaaef_population(
        _board(repo_root),
        forest=_sealed_forest(),
    )


def _bootstrap_snapshot(
    population: lifecycle.CompiledEAAEFPopulation,
) -> dict[str, Any]:
    value = {
        "schema": lifecycle.EAAEF_BOOTSTRAP_SNAPSHOT_SCHEMA,
        "source_head": population.source_head,
        "source_tree": population.source_tree,
        "source_forest_root": population.source_forest_root,
        "board_cid": population.board_cid,
        "reconciliation_population_cid": population.population_cid,
        "bootstrap_population_cid": population.bootstrap_population_cid,
        "bootstrap_task_count": lifecycle.EAAEF_BOOTSTRAP_TASK_COUNT,
        "held_task_count": lifecycle.EAAEF_PLAN_R2_TASK_COUNT,
        "terminal_statuses_imported": 0,
        "bootstrap_materialization_mode": "offline_before_exclusive_owner_start",
        "bootstrap_owner_absent_during_materialization": True,
        "owner_started_after_bootstrap": True,
        "direct_database_mutation_after_owner_start": False,
        "bootstrap_admission_cid": "sha256:" + "a" * 64,
        "r1_launch_capsule_cid": "sha256:" + "b" * 64,
        "quack_owner_qualification_cid": "sha256:" + "c" * 64,
        "quack_command_fabric_qualification_cid": "sha256:" + "d" * 64,
        "owner_principal_did": "did:key:z" + "A" * 20,
        "shard_id": "fresh-shard",
        "store_id": "fresh-store",
        "owner_generation": 1,
        "expected_epoch": 1,
        "fencing_token": 1,
        "lease_id": "fresh-lease",
        "expected_version": 1,
        "expected_active_plan_cid": population.plan_r1_cid,
        "expected_active_plan_root_cid": population.plan_r1_cid,
        "expected_active_plan_revision": 1,
        "expected_event_cursor": "0",
        "expected_semantic_root_cid": population.source_forest_root,
        "request_id": "fresh-request",
        "idempotency_key": "fresh-idempotency",
        "deadline_ms": 200_000,
        "issued_at_ms": 100_000,
        "expires_at_ms": 300_000,
        "one_use_nonce": "fresh-nonce",
    }
    value["snapshot_cid"] = lifecycle._cid(value)
    return value


def _qualification(source_forest_root: str) -> dict[str, Any]:
    value = {
        "schema": lifecycle.EAAEF_OWNER_QUALIFICATION_SCHEMA,
        "interface": lifecycle.EAAEF_RECONCILIATION_OWNER_INTERFACE,
        "source_forest_root": source_forest_root,
        "bootstrap_materialization_mode": "offline_before_exclusive_owner_start",
        "bootstrap_materialization_before_owner_start": True,
        "offline_population_includes_execution_contracts": True,
        "direct_database_mutation_after_owner_start": False,
        "typed_task_source_interface": lifecycle.EAAEF_TYPED_TASK_SOURCE_INTERFACE,
        "plan_r2_repository_interface": lifecycle.AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE,
        "plan_r2_remote_gateway_interface": (lifecycle.PLAN_R2_REMOTE_CLIENT_GATEWAY_INTERFACE),
        "plan_r2_wire_channel_interface": lifecycle.PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
        "plan_r2_remote_runtime_qualification_status": "production_qualified",
        "plan_r2_remote_runtime_blockers": [],
        "status_operation": "status.snapshot",
        "stop_tracks_operation": "stop_tracks",
        "launch_modes": ["paused", "plan_r2"],
        "database_authority_crossing_allowed": False,
        "filesystem_path_authority_crossing_allowed": False,
        "transport_token_authority_crossing_allowed": False,
        "sql_crossing_allowed": False,
        "provider_launch_allowed": True,
    }
    value["qualification_cid"] = lifecycle._cid(value)
    return value


class _FakeOwner:
    INTERFACE = lifecycle.EAAEF_RECONCILIATION_OWNER_INTERFACE

    def __init__(
        self,
        source_forest_root: str,
        *,
        birth: lifecycle.ProcessBirth | None = None,
        omit_stopped_birth: bool = False,
    ) -> None:
        self.source_forest_root = source_forest_root
        self.birth = birth
        self.omit_stopped_birth = omit_stopped_birth
        self.stopped = False
        self.offline_request: dict[str, Any] | None = None

    def reconciliation_qualification(self) -> Mapping[str, Any]:
        return _qualification(self.source_forest_root)

    def materialize_offline_population(
        self,
        request: Mapping[str, Any],
        *,
        population: lifecycle.CompiledEAAEFPopulation,
    ) -> Mapping[str, Any]:
        self.offline_request = dict(request)
        snapshot = _bootstrap_snapshot(population)
        value = {
            "schema": lifecycle.EAAEF_OFFLINE_POPULATION_RECEIPT_SCHEMA,
            "interface": lifecycle.EAAEF_RECONCILIATION_OWNER_INTERFACE,
            "request_cid": request["request_cid"],
            "generation_id": request["generation_id"],
            "source_forest_root": population.source_forest_root,
            "population_cid": population.population_cid,
            "goal_population_cid": population.goal_population_cid,
            "execution_contract_population_cid": (
                population.execution_contract_population_cid
            ),
            "bootstrap_population_cid": population.bootstrap_population_cid,
            "held_plan_r2_population_cid": population.plan_r2_population_cid,
            "plan_r1_cid": population.plan_r1_cid,
            "task_count": lifecycle.EAAEF_TASK_COUNT,
            "goal_count": lifecycle.EAAEF_GOAL_COUNT,
            "goal_edge_count": lifecycle.EAAEF_GOAL_EDGE_COUNT,
            "plan_count": 1,
            "bootstrap_task_count": lifecycle.EAAEF_BOOTSTRAP_TASK_COUNT,
            "held_task_count": lifecycle.EAAEF_PLAN_R2_TASK_COUNT,
            "task_status_counts": {
                "blocked": lifecycle.EAAEF_PLAN_R2_TASK_COUNT,
                "todo": lifecycle.EAAEF_BOOTSTRAP_TASK_COUNT,
            },
            "execution_contract_counts": population.execution_contract_counts,
            "execution_contracts_materialized": True,
            "terminal_statuses_imported": 0,
            "bootstrap_materialization_mode": "offline_before_exclusive_owner_start",
            "bootstrap_owner_absent_during_materialization": True,
            "owner_started_after_bootstrap": True,
            "direct_database_mutation_after_owner_start": False,
            "provider_process_started": False,
            "bootstrap_snapshot": snapshot,
        }
        value["receipt_cid"] = lifecycle._cid(value)
        return value

    def apply_signed_plan_r2(
        self,
        request: Mapping[str, Any],
        *,
        population: lifecycle.CompiledEAAEFPopulation,
        authority: lifecycle.VerifiedFreshEAAEFAuthority,
    ) -> Mapping[str, Any]:
        raise AssertionError("signed authority is not created or applied by these tests")

    def launch_reconciliation_supervisor(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        raise AssertionError("no supervisor is launched by these tests")

    def reconciliation_status_snapshot(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        active = not self.stopped
        value = {
            "schema": lifecycle.EAAEF_OWNER_STATUS_RECEIPT_SCHEMA,
            "interface": lifecycle.EAAEF_RECONCILIATION_OWNER_INTERFACE,
            "request_cid": request["request_cid"],
            "active": active,
            "generation_id": "eaaef-test-generation" if active else "",
            "phase": "launched_paused" if active else "absent",
            "source_head": "1" * 40 if active else "",
            "source_forest_root": self.source_forest_root if active else "",
            "task_count": lifecycle.EAAEF_TASK_COUNT if active else 0,
            "task_status_counts": {"todo": lifecycle.EAAEF_TASK_COUNT} if active else {},
            "supervisor_birth": self.birth.to_dict() if active and self.birth else None,
            "provider_process_started": False,
        }
        value["receipt_cid"] = lifecycle._cid(value)
        return value

    def stop_reconciliation_tracks(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        stopped_births = []
        if self.birth is not None and not self.omit_stopped_birth:
            stopped_births.append(self.birth.to_dict())
        self.stopped = True
        value = {
            "schema": lifecycle.EAAEF_OWNER_STOP_RECEIPT_SCHEMA,
            "interface": lifecycle.EAAEF_RECONCILIATION_OWNER_INTERFACE,
            "request_cid": request["request_cid"],
            "generation_id": request["generation_id"],
            "stopped": True,
            "remaining_track_count": 0,
            "stopped_process_births": stopped_births,
            "provider_processes_stopped": True,
            "task_state_mutated": False,
        }
        value["receipt_cid"] = lifecycle._cid(value)
        return value


def _state(
    population: lifecycle.CompiledEAAEFPopulation,
    *,
    phase: str = "launched_paused",
) -> dict[str, Any]:
    value = {
        "schema": lifecycle.EAAEF_STATE_SCHEMA,
        "interface": lifecycle.EAAEF_RECONCILIATION_LIFECYCLE_INTERFACE,
        "generation_id": "eaaef-test-generation",
        "phase": phase,
        "source_head": population.source_head,
        "source_tree": population.source_tree,
        "source_forest_root": population.source_forest_root,
        "population": population.public_dict(),
        "supervisor_birth": None,
        "provider_process_started": False,
        "updated_at_ms": 1,
    }
    value["state_cid"] = lifecycle._cid(value)
    return value


def _parser_destinations(parser: argparse.ArgumentParser) -> set[str]:
    result = {action.dest for action in parser._actions}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                result.update(_parser_destinations(child))
    return result


def test_fresh_population_is_exact_22_plus_94_and_plan_r2_releases_all(
    repo_root: Path,
) -> None:
    population = _population(repo_root)

    assert population.task_count == 116
    assert len(population.bootstrap_tasks) == 22
    assert len(population.plan_r2_tasks) == 94
    assert len(population.dependencies) == 270
    assert Counter(item["status"] for item in population.bootstrap_tasks) == {"todo": 22}
    assert Counter(item["status"] for item in population.plan_r2_tasks) == {"blocked": 94}
    assert population.execution_contract_counts == {
        "task_dependencies": 270,
        "task_outputs": 415,
        "task_validations": 117,
        "task_acceptance": 116,
    }

    statement = lifecycle.build_unsigned_fresh_plan_r2_statement(
        population=population,
        bootstrap_snapshot=_bootstrap_snapshot(population),
    )
    transition = population.plan_r2_transition_tasks(
        plan_cid=str(statement["new_plan"]["plan_cid"])
    )

    assert len(statement["tasks"]) == 116
    assert len(statement["dependencies"]) == 270
    assert statement["protected_tasks"] == []
    assert all(item["status"] == "todo" for item in transition)
    assert all(item["body"]["is_schedulable"] is True for item in transition)
    assert all(item["body"]["blocked_reason"] == "" for item in transition)
    assert all(item["revision"] == 2 for item in transition)
    assert len(lifecycle._canonical_bytes(statement)) <= (
        lifecycle.MAX_PLAN_R2_REMOTE_REQUEST_BYTES
        - lifecycle._PLAN_R2_REMOTE_REQUEST_OVERHEAD_RESERVE
    )
    assert "operator_signature" not in statement
    assert "security_reviewer_signature" not in statement


def test_stale_forest_and_bootstrap_bindings_fail_closed(repo_root: Path) -> None:
    original = _sealed_forest()
    stale = json.loads(json.dumps(original))
    stale["repositories"][0]["tree"] = "e" * 40
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="forest"):
        lifecycle.compile_fresh_eaaef_population(_board(repo_root), forest=stale)

    original_population = lifecycle.compile_fresh_eaaef_population(
        _board(repo_root), forest=original
    )
    fresh_population = lifecycle.compile_fresh_eaaef_population(
        _board(repo_root), forest=_sealed_forest(accelerator_commit="9" * 40)
    )
    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="bootstrap owner snapshot differs",
    ):
        lifecycle.build_unsigned_fresh_plan_r2_statement(
            population=fresh_population,
            bootstrap_snapshot=_bootstrap_snapshot(original_population),
        )


def test_board_source_binding_is_read_from_exact_git_tree_blob(repo_root: Path) -> None:
    head = lifecycle._git(repo_root, "rev-parse", "HEAD")
    tree = lifecycle._git(repo_root, "rev-parse", f"{head}^{{tree}}")
    binding = lifecycle._git_board_source(
        repo_root,
        source_head=head,
        source_tree=tree,
    )
    board_bytes = (repo_root / lifecycle.EAAEF_BOARD_PATH).read_bytes()

    assert binding["relative_path"] == lifecycle.EAAEF_BOARD_PATH
    assert binding["source_head"] == head
    assert binding["source_tree"] == tree
    assert binding["git_mode"] == "100644"
    assert binding["object_type"] == "blob"
    assert binding["byte_count"] == len(board_bytes)
    assert binding["bytes_cid"] == lifecycle._cid(board_bytes)
    assert binding["canonical_json_cid"] == lifecycle._eaaef_source_cid(_board(repo_root))


def test_prepare_materializes_offline_contracts_and_stops_before_authority(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = _population(repo_root)
    owner = _FakeOwner(population.source_forest_root)
    monkeypatch.setattr(
        lifecycle,
        "inspect_current_repository_forest",
        lambda _root: _sealed_forest(),
    )

    result = lifecycle.prepare_fresh_generation(
        repo_root=repo_root,
        state_root=tmp_path / "state",
        owner=owner,
        generation_id="eaaef-prepare-generation",
        now_ms=1,
    )

    assert result["phase"] == "awaiting_external_authority"
    assert result["provider_process_started"] is False
    assert result["unsigned_authority_request"]["unsigned_plan_r2_statement"] is not None
    assert owner.offline_request is not None
    assert owner.offline_request["expected_task_count"] == 116
    assert owner.offline_request["expected_goal_count"] == 20
    assert owner.offline_request["expected_goal_edge_count"] == 18
    assert owner.offline_request["expected_plan_count"] == 1
    assert owner.offline_request["bootstrap_task_count"] == 22
    assert owner.offline_request["held_task_count"] == 94
    assert owner.offline_request["owner_must_be_absent_during_population_write"] is True
    assert owner.offline_request["expected_execution_contract_counts"] == (
        population.execution_contract_counts
    )


@pytest.mark.parametrize(
    "forbidden",
    [
        {"database_path": "/tmp/control.duckdb"},
        {"raw_token": "secret"},
        {"statement_sql": "SELECT * FROM tasks"},
    ],
)
def test_typed_boundary_rejects_database_token_and_sql(forbidden: dict[str, str]) -> None:
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="exposes"):
        lifecycle._assert_no_boundary_authority(forbidden)


def test_public_cli_and_source_have_no_raw_authority_or_historical_run_surface() -> None:
    destinations = _parser_destinations(lifecycle._argument_parser())
    assert destinations.isdisjoint(
        {"database", "database_path", "duckdb_path", "sql", "token", "credential"}
    )
    parsed = lifecycle._argument_parser().parse_args(["launch", "--plan-r2"])
    assert parsed.plan_r2 is True
    source = inspect.getsource(lifecycle)
    assert "duckdb.connect(" not in source
    assert "os.kill(" not in source
    assert "SIGTERM" not in source
    assert "SIGKILL" not in source
    assert "run-v14" not in source


def test_owner_qualification_rejects_stale_forest_and_unclosed_remote_blockers(
    repo_root: Path,
) -> None:
    population = _population(repo_root)
    owner = _FakeOwner(population.source_forest_root)
    assert (
        lifecycle.require_typed_reconciliation_owner(
            owner, source_forest_root=population.source_forest_root
        )
        is owner
    )

    stale = _FakeOwner("sha256:" + "f" * 64)
    with pytest.raises(lifecycle.EAAEFReconciliationBlocked, match="source_forest_root"):
        lifecycle.require_typed_reconciliation_owner(
            stale, source_forest_root=population.source_forest_root
        )

    qualification = _qualification(population.source_forest_root)
    qualification["plan_r2_remote_runtime_blockers"] = ["wire_not_qualified"]
    qualification.pop("qualification_cid")
    qualification["qualification_cid"] = lifecycle._cid(qualification)
    owner.reconciliation_qualification = lambda: qualification  # type: ignore[method-assign]
    with pytest.raises(
        lifecycle.EAAEFReconciliationBlocked,
        match="plan_r2_remote_runtime_blockers",
    ):
        lifecycle.require_typed_reconciliation_owner(owner)


def test_status_and_stop_use_owner_receipts_and_exact_birth_cleanup(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = _population(repo_root)
    birth = lifecycle.ProcessBirth(
        pid=42,
        start_time_ticks=100,
        parent_pid=2,
        boot_id="boot-one",
        argv_sha256="sha256:" + "5" * 64,
    )
    reused = lifecycle.ProcessBirth(
        pid=42,
        start_time_ticks=101,
        parent_pid=2,
        boot_id="boot-one",
        argv_sha256="sha256:" + "6" * 64,
    )
    owner = _FakeOwner(population.source_forest_root, birth=birth)
    store = lifecycle.ReconciliationStateStore(tmp_path / "state")
    state = _state(population)
    store.create_generation("eaaef-test-generation", state)
    store.activate("eaaef-test-generation", state_cid=str(state["state_cid"]))
    for artifact in ("owner.sock", "supervisor.pid.json", "stop.request"):
        (store.generation_dir("eaaef-test-generation") / artifact).write_text(
            "test", encoding="utf-8"
        )

    def probe(pid: int) -> lifecycle.ProcessBirth | None:
        assert pid == birth.pid
        return reused if owner.stopped else birth

    def forbidden_signal(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("client lifecycle must never signal a process directly")

    monkeypatch.setattr(os, "kill", forbidden_signal)
    status = lifecycle.reconciliation_status(
        state_root=store.root,
        owner=owner,
        process_probe=probe,
    )
    assert status["active"] is True
    assert status["local_birth_corroborated"] is True
    assert status["owner_supervisor_birth"] == birth.to_dict()

    stopped = lifecycle.stop_reconciliation_generation(
        state_root=store.root,
        owner=owner,
        process_probe=probe,
    )
    assert stopped["stopped"] is True
    assert stopped["stopped_process_count"] == 1
    assert set(stopped["removed_runtime_artifacts"]) == {
        "owner.sock",
        "supervisor.pid.json",
        "stop.request",
    }
    assert store.active_generation() == ""
    assert store.read_state("eaaef-test-generation")["phase"] == "stopped"


def test_status_rejects_unknown_typed_task_status(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    population = _population(repo_root)
    owner = _FakeOwner(population.source_forest_root)
    original_status = owner.reconciliation_status_snapshot

    def unknown_status(request: Mapping[str, Any]) -> Mapping[str, Any]:
        value = dict(original_status(request))
        value.pop("receipt_cid")
        value["task_status_counts"] = {"todo": 115, "invented_status": 1}
        value["receipt_cid"] = lifecycle._cid(value)
        return value

    owner.reconciliation_status_snapshot = unknown_status  # type: ignore[method-assign]
    store = lifecycle.ReconciliationStateStore(tmp_path / "state")
    state = _state(population)
    store.create_generation("eaaef-test-generation", state)
    store.activate("eaaef-test-generation", state_cid=str(state["state_cid"]))

    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="status counts are malformed",
    ):
        lifecycle.reconciliation_status(state_root=store.root, owner=owner)


def test_stop_rejects_receipt_that_omits_status_bound_birth(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    population = _population(repo_root)
    birth = lifecycle.ProcessBirth(
        pid=43,
        start_time_ticks=100,
        parent_pid=2,
        boot_id="boot-one",
        argv_sha256="sha256:" + "7" * 64,
    )
    owner = _FakeOwner(
        population.source_forest_root,
        birth=birth,
        omit_stopped_birth=True,
    )
    store = lifecycle.ReconciliationStateStore(tmp_path / "state")
    state = _state(population)
    store.create_generation("eaaef-test-generation", state)
    store.activate("eaaef-test-generation", state_cid=str(state["state_cid"]))
    artifact = store.generation_dir("eaaef-test-generation") / "owner.sock"
    artifact.write_text("test", encoding="utf-8")

    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="omits the status-bound supervisor birth",
    ):
        lifecycle.stop_reconciliation_generation(
            state_root=store.root,
            owner=owner,
            process_probe=lambda _pid: birth,
        )
    assert artifact.exists()
    assert store.active_generation() == "eaaef-test-generation"
