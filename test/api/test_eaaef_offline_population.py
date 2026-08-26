from __future__ import annotations

import hashlib
import inspect
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    current_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    eaaef_casf_bootstrap_lifecycle as casf_lifecycle,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    eaaef_casf_owner_management as casf_management,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    eaaef_offline_population as offline,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    eaaef_reconciliation_lifecycle as lifecycle,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    ExclusiveOwnerLease,
    QuackStateServerOwnershipError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_eaaef_reconciliation_owner import (
    EAAEF_BOOTSTRAP_RECONCILIATION_OWNER_INTERFACE,
    EAAEF_CASF_BOOTSTRAP_OWNER_GUARD_INTERFACE,
    EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
    EAAEF_CASF_BOOTSTRAP_REGISTRY_SCHEMA,
    EAAEF_CASF_OWNER_ABORT_RECEIPT_SCHEMA,
    EAAEF_CASF_OWNER_ABSENCE_ATTESTATION_SCHEMA,
    EAAEF_CASF_OWNER_COMMIT_RECEIPT_SCHEMA,
    EAAEF_CASF_OWNER_START_RECEIPT_SCHEMA,
    EAAEF_OWNER_PRODUCTION_BLOCKERS,
    EAAEFCASFBootstrapBinding,
    EAAEFCASFBootstrapOwnerError,
    EAAEFCASFBootstrapRegistry,
    EAAEFTypedReconciliationOwnerUnavailable,
    bind_eaaef_casf_bootstrap_owner,
    open_eaaef_bootstrap_reconciliation_owner,
    open_eaaef_typed_reconciliation_owner,
)

_REAL_INSPECT_CURRENT_REPOSITORY_FOREST = lifecycle.inspect_current_repository_forest


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


@pytest.fixture(autouse=True)
def _explicit_trusted_forest_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        lifecycle,
        "inspect_current_repository_forest",
        lambda _root: _sealed_forest(),
    )


def _population(repo_root: Path) -> lifecycle.CompiledEAAEFPopulation:
    return lifecycle.compile_fresh_eaaef_population(
        _board(repo_root),
        forest=_sealed_forest(),
        repo_root=repo_root,
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
        "owner_principal_did": ed25519_did_key(bytes([11]) * 32),
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


def _concrete_snapshot_bindings() -> (
    casf_lifecycle.EAAEFCASFBootstrapSnapshotBindings
):
    return casf_lifecycle.EAAEFCASFBootstrapSnapshotBindings(
        bootstrap_admission_cid="sha256:" + "a" * 64,
        r1_launch_capsule_cid="sha256:" + "b" * 64,
        quack_owner_qualification_cid="sha256:" + "c" * 64,
        quack_command_fabric_qualification_cid="sha256:" + "d" * 64,
        owner_principal_did=ed25519_did_key(bytes([11]) * 32),
        shard_id="fresh-shard",
        store_id="fresh-store",
        lease_id="fresh-lease",
        expected_event_cursor="0",
        request_id="fresh-request",
        idempotency_key="fresh-idempotency",
        issued_at_ms=100_000,
        deadline_ms=200_000,
        expires_at_ms=300_000,
        one_use_nonce="fresh-nonce",
    )


def _concrete_bootstrap_binding(
    generation_dir: Path,
    *,
    generation_id: str,
) -> EAAEFCASFBootstrapBinding:
    generation_dir.mkdir(mode=0o700)
    generation_dir.chmod(0o700)
    return EAAEFCASFBootstrapBinding(
        generation_id=generation_id,
        source_head="1" * 40,
        source_tree="2" * 40,
        source_forest_root="sha256:" + "3" * 64,
        board_cid="sha256:" + "4" * 64,
        population_cid="sha256:" + "5" * 64,
        bootstrap_population_cid="sha256:" + "6" * 64,
        plan_r1_cid="sha256:" + "7" * 64,
        database_path=generation_dir / "control.duckdb",
        owner_state_dir=generation_dir / "casf-owner",
    )


class _FakeCASFBootstrapGuard:
    """Non-production process double for provisional owner lifecycle tests."""

    INTERFACE = EAAEF_CASF_BOOTSTRAP_OWNER_GUARD_INTERFACE
    NONPRODUCTION_TEST_DOUBLE = True

    def __init__(
        self,
        binding: object,
        population: lifecycle.CompiledEAAEFPopulation,
        events: list[str],
        *,
        start_error: BaseException | None = None,
        start_mutation: tuple[str, Any] | None = None,
    ) -> None:
        self.binding = binding
        self.population = population
        self.events = events
        self.start_error = start_error
        self.start_mutation = start_mutation
        self.held = False
        self.absence: dict[str, Any] | None = None
        self.start_receipt: dict[str, Any] | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.process_birth: lifecycle.ProcessBirth | None = None
        self.committed = False

    def __enter__(self) -> _FakeCASFBootstrapGuard:
        self.held = True
        self.events.append("exclusive_guard_acquired")
        return self

    def __exit__(self, *_args: object) -> None:
        if not self.committed:
            self._terminate_owner()
        self.events.append("exclusive_guard_released")
        self.held = False

    def _terminate_owner(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def cleanup(self) -> None:
        self._terminate_owner()

    def owner_absence_attestation(self) -> dict[str, Any]:
        assert self.held is True
        self.events.append("owner_absence_attested")
        value = {
            "schema": EAAEF_CASF_OWNER_ABSENCE_ATTESTATION_SCHEMA,
            "interface": EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
            "generation_id": self.binding.generation_id,
            "source_forest_root": self.binding.source_forest_root,
            "owner_absent": True,
            "exclusive_owner_lease_held": True,
            "observed_owner_process_birth": None,
        }
        value["attestation_cid"] = lifecycle._cid(value)
        self.absence = value
        return value

    def start_after_offline_commit(
        self,
        *,
        offline_materialization_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        assert self.held is True
        assert self.binding.database_path.is_file()
        record = json.loads(
            (self.binding.database_path.parent / "bootstrap-owner.json").read_text(
                encoding="utf-8"
            )
        )
        assert record["phase"] == "offline_committed"
        assert offline_materialization_receipt["owner_started"] is False
        self.events.append("owner_start_requested")
        if self.start_error is not None:
            raise self.start_error
        assert self.absence is not None
        self.process = subprocess.Popen(
            [sys.executable, "-c", "import signal; signal.pause()"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        birth = lifecycle.inspect_process_birth(self.process.pid)
        for _attempt in range(100):
            if birth is not None:
                break
            os.sched_yield()
            birth = lifecycle.inspect_process_birth(self.process.pid)
        assert birth is not None
        self.process_birth = birth
        value = {
            "schema": EAAEF_CASF_OWNER_START_RECEIPT_SCHEMA,
            "interface": EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
            "generation_id": self.binding.generation_id,
            "source_forest_root": self.binding.source_forest_root,
            "population_cid": self.binding.population_cid,
            "absence_attestation_cid": self.absence["attestation_cid"],
            "offline_materialization_receipt_cid": (
                offline_materialization_receipt["receipt_cid"]
            ),
            "owner_started_after_bootstrap": True,
            "exclusive_owner_lease_handoff_complete": True,
            "owner_start_commit_pending": True,
            "provider_process_started": False,
            "owner_process_birth": birth.to_dict(),
            "bootstrap_snapshot": _bootstrap_snapshot(self.population),
        }
        if self.start_mutation is not None:
            field, selected = self.start_mutation
            value[field] = selected
        value["start_receipt_cid"] = lifecycle._cid(value)
        self.start_receipt = value
        return value

    def abort_started_owner(
        self,
        *,
        start_receipt: Mapping[str, Any] | None,
        reason_code: str,
    ) -> dict[str, Any]:
        assert self.held is True
        self.events.append("owner_abort_requested")
        self._terminate_owner()
        start_cid = ""
        if start_receipt is not None and isinstance(
            start_receipt.get("start_receipt_cid"), str
        ):
            start_cid = start_receipt["start_receipt_cid"]
        value = {
            "schema": EAAEF_CASF_OWNER_ABORT_RECEIPT_SCHEMA,
            "interface": EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
            "generation_id": self.binding.generation_id,
            "owner_start_receipt_cid": start_cid,
            "abort_reason_code": reason_code,
            "owner_abort_completed": True,
            "remaining_started_owner_count": 0,
            "owner_process_birth": (
                None if self.process_birth is None else self.process_birth.to_dict()
            ),
            "owner_process_alive": False,
            "task_state_mutated": False,
        }
        value["abort_receipt_cid"] = lifecycle._cid(value)
        self.committed = False
        return value

    def commit_started_owner(
        self,
        *,
        start_receipt: Mapping[str, Any],
        final_record_cid: str,
    ) -> dict[str, Any]:
        assert self.held is True
        assert self.start_receipt == start_receipt
        assert self.process_birth is not None
        assert lifecycle.inspect_process_birth(self.process_birth.pid) == self.process_birth
        record = json.loads(
            (self.binding.database_path.parent / "bootstrap-owner.json").read_text(
                encoding="utf-8"
            )
        )
        assert record["phase"] == "owner_started"
        assert record["record_cid"] == final_record_cid
        self.events.append("owner_commit_requested")
        self.committed = True
        value = {
            "schema": EAAEF_CASF_OWNER_COMMIT_RECEIPT_SCHEMA,
            "interface": EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
            "generation_id": self.binding.generation_id,
            "owner_start_receipt_cid": start_receipt["start_receipt_cid"],
            "final_record_cid": final_record_cid,
            "owner_commit_completed": True,
            "owner_process_birth": self.process_birth.to_dict(),
            "owner_process_alive": True,
            "provider_process_started": False,
        }
        value["commit_receipt_cid"] = lifecycle._cid(value)
        return value


class _FakeCASFBootstrapLifecycle:
    """Explicitly non-production lifecycle; it never exercises CASF/Quack."""

    INTERFACE = EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE
    NONPRODUCTION_TEST_DOUBLE = True

    def __init__(
        self,
        population: lifecycle.CompiledEAAEFPopulation,
        *,
        start_error: BaseException | None = None,
        start_mutation: tuple[str, Any] | None = None,
    ) -> None:
        self.population = population
        self.events: list[str] = []
        self.start_error = start_error
        self.start_mutation = start_mutation
        self.guards: list[_FakeCASFBootstrapGuard] = []

    def hold_exclusive_bootstrap(
        self,
        binding: object,
    ) -> _FakeCASFBootstrapGuard:
        guard = _FakeCASFBootstrapGuard(
            binding,
            self.population,
            self.events,
            start_error=self.start_error,
            start_mutation=self.start_mutation,
        )
        self.guards.append(guard)
        return guard

    def cleanup(self) -> None:
        for guard in self.guards:
            guard.cleanup()


def _board(repo_root: Path) -> dict[str, Any]:
    return json.loads((repo_root / lifecycle.EAAEF_BOARD_PATH).read_text(encoding="utf-8"))


def _plain(value: Any) -> Any:
    return json.loads(lifecycle._canonical_bytes(value))


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit_test_repository(repo_root: Path, *, marker: str) -> str:
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--quiet")
    _git(repo_root, "config", "user.name", "EAAEF provenance test")
    _git(repo_root, "config", "user.email", "eaaef-provenance@example.invalid")
    (repo_root / "identity.txt").write_text(marker, encoding="utf-8")
    _git(repo_root, "add", "identity.txt")
    _git(repo_root, "commit", "--quiet", "-m", "test repository identity")
    return _git(repo_root, "rev-parse", "HEAD")


def _real_git_forest(repo_root: Path, *, board_bytes: bytes) -> dict[str, Any]:
    repo_root.mkdir(parents=True)
    _git(repo_root, "init", "--quiet")
    _git(repo_root, "config", "user.name", "EAAEF provenance test")
    _git(repo_root, "config", "user.email", "eaaef-provenance@example.invalid")
    board_path = repo_root / lifecycle.EAAEF_BOARD_PATH
    board_path.parent.mkdir(parents=True)
    board_path.write_bytes(board_bytes)
    _git(repo_root, "add", lifecycle.EAAEF_BOARD_PATH)
    for name, relative_path in lifecycle._SUBMODULES:
        nested_head = _commit_test_repository(repo_root / relative_path, marker=name)
        _git(
            repo_root,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{nested_head},{relative_path}",
        )
    _git(repo_root, "commit", "--quiet", "-m", "sealed EAAEF test forest")
    return _REAL_INSPECT_CURRENT_REPOSITORY_FOREST(repo_root)


def _replace_bootstrap_task(
    population: lifecycle.CompiledEAAEFPopulation,
    task: dict[str, Any],
) -> lifecycle.CompiledEAAEFPopulation:
    return replace(population, bootstrap_tasks=(task, *population.bootstrap_tasks[1:]))


def _reseal_mutated_output_population(
    population: lifecycle.CompiledEAAEFPopulation,
) -> lifecycle.CompiledEAAEFPopulation:
    task = _plain(population.bootstrap_tasks[0])
    task["body"]["outputs"][0] = "forged/resealed-output.txt"
    unsigned_task = dict(task)
    unsigned_task.pop("execution_contract_cid")
    task["execution_contract_cid"] = lifecycle._cid(
        {
            "schema": "EAAEFTaskExecutionContract@1",
            "task": unsigned_task,
            "source_forest_root": population.source_forest_root,
        }
    )
    bootstrap = (task, *population.bootstrap_tasks[1:])
    tasks = (*bootstrap, *population.plan_r2_tasks)
    execution_contract_population_cid = lifecycle._cid(
        {
            "schema": lifecycle.EAAEF_EXECUTION_CONTRACT_POPULATION_SCHEMA,
            "contracts": [
                {
                    "task_cid": item["task_cid"],
                    "execution_contract_cid": item["execution_contract_cid"],
                }
                for item in tasks
            ],
            "source_forest_root": population.source_forest_root,
        }
    )
    bootstrap_population_cid = lifecycle._cid(
        {
            "schema": "EAAEFBootstrapPopulation@1",
            "tasks": bootstrap,
            "dependencies": population.dependencies,
            "source_forest_root": population.source_forest_root,
        }
    )
    population_cid = lifecycle._cid(
        {
            "schema": lifecycle.EAAEF_POPULATION_SCHEMA,
            "board_cid": population.board_cid,
            "bootstrap_population_cid": bootstrap_population_cid,
            "plan_r2_population_cid": population.plan_r2_population_cid,
            "goal_population_cid": population.goal_population_cid,
            "execution_contract_population_cid": execution_contract_population_cid,
            "source_forest_root": population.source_forest_root,
            "task_count": lifecycle.EAAEF_TASK_COUNT,
        }
    )
    return replace(
        population,
        bootstrap_tasks=bootstrap,
        execution_contract_population_cid=execution_contract_population_cid,
        bootstrap_population_cid=bootstrap_population_cid,
        population_cid=population_cid,
    )


class _ForbiddenMaterializer:
    INTERFACE = offline.DATABASE_TASK_SOURCE_INTERFACE

    def __init__(self) -> None:
        self.called = False

    def materialize(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        self.called = True
        raise AssertionError("forged population reached the task source")


def test_translation_is_exact_database_task_source_population(repo_root: Path) -> None:
    population = _population(repo_root)
    translated = offline.translate_compiled_eaaef_population(
        population,
        current_board=_board(repo_root),
        current_forest=_sealed_forest(),
        repo_root=repo_root,
        owner_active=False,
    )

    assert translated["schema"] == offline.EAAEF_OFFLINE_TASK_SOURCE_POPULATION_SCHEMA
    assert translated["task_source_interface"] == "DatabaseTaskSource@1"
    assert translated["plan_root_cid"] == population.plan_r1_cid
    assert len(translated["goals"]) == 20
    assert len(translated["goal_edges"]) == 18
    assert len(translated["plans"]) == 1
    assert translated["plans"][0]["plan_alias"] == lifecycle.EAAEF_PLAN_R1_ALIAS
    assert translated["plans"][0]["revision"] == 1
    assert len(translated["tasks"]) == 116
    assert Counter(item["status"] for item in translated["tasks"]) == {
        "blocked": 94,
        "todo": 22,
    }
    assert sum(len(item["depends_on"]) for item in translated["tasks"]) == 270
    assert sum(len(item["outputs"]) for item in translated["tasks"]) == 430
    assert sum(len(item["validations"]) for item in translated["tasks"]) == 117
    assert sum(len(item["acceptance"]) for item in translated["tasks"]) == 116
    assert translated["terminal_statuses_imported"] == 0
    assert translated["owner_absent_required"] is True
    assert translated["provider_launch_allowed"] is False
    assert [
        item["goal_alias"] for item in translated["goals"] if not item["parent_goal_cid"]
    ] == ["EAAEF-G000"]
    assert offline.verify_translated_eaaef_population(
        translated,
        population=population,
        current_board=_board(repo_root),
        current_forest=_sealed_forest(),
        repo_root=repo_root,
    ) == translated


def test_public_population_apis_require_keyword_only_trusted_repo_root(repo_root: Path) -> None:
    population = _population(repo_root)
    translated = offline.translate_compiled_eaaef_population(
        population,
        current_board=_board(repo_root),
        current_forest=_sealed_forest(),
        repo_root=repo_root,
        owner_active=False,
    )
    for function in (
        lifecycle.compile_fresh_eaaef_population,
        lifecycle.verify_compiled_eaaef_population_commitments,
        offline.translate_compiled_eaaef_population,
        offline.verify_translated_eaaef_population,
        offline.materialize_offline_eaaef_population,
    ):
        parameter = inspect.signature(function).parameters["repo_root"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None

    with pytest.raises(lifecycle.EAAEFReconciliationBlocked, match="explicit trusted repo_root"):
        lifecycle.compile_fresh_eaaef_population(
            _board(repo_root),
            forest=_sealed_forest(),
        )
    with pytest.raises(lifecycle.EAAEFReconciliationBlocked, match="explicit trusted repo_root"):
        lifecycle.verify_compiled_eaaef_population_commitments(
            population,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
        )
    with pytest.raises(lifecycle.EAAEFReconciliationBlocked, match="explicit trusted repo_root"):
        offline.translate_compiled_eaaef_population(
            population,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
            owner_active=False,
        )
    with pytest.raises(lifecycle.EAAEFReconciliationBlocked, match="explicit trusted repo_root"):
        offline.verify_translated_eaaef_population(
            translated,
            population=population,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
        )
    sink = _ForbiddenMaterializer()
    with pytest.raises(lifecycle.EAAEFReconciliationBlocked, match="explicit trusted repo_root"):
        offline.materialize_offline_eaaef_population(
            sink,
            population,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
            owner_active=False,
        )
    assert sink.called is False


def test_offline_materialization_preserves_all_contract_rows(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    if not DatabaseTaskSource.available():
        pytest.skip("DuckDB unavailable")
    population = _population(repo_root)
    with DatabaseTaskSource(tmp_path / "offline-population.duckdb") as source:
        receipt = offline.materialize_offline_eaaef_population(
            source,
            population,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
            repo_root=repo_root,
            owner_active=False,
        )
        page = source.list_tasks(limit=200)
        snapshot = source.intent.snapshot()

    assert receipt["qualification_status"] == "offline_population_only"
    assert receipt["owner_absent_during_materialization"] is True
    assert receipt["owner_started"] is False
    assert receipt["provider_process_started"] is False
    assert receipt["task_status_counts"] == {"blocked": 94, "todo": 22}
    assert receipt["execution_contract_counts"] == {
        "task_dependencies": 270,
        "task_outputs": 430,
        "task_validations": 117,
        "task_acceptance": 116,
    }
    assert snapshot.goal_count == 20
    assert snapshot.plan_count == 1
    assert snapshot.task_count == 116
    assert snapshot.dependency_count == 270
    assert len(page.tasks) == 116
    assert Counter(item.status for item in page.tasks) == {"blocked": 94, "todo": 22}
    assert sum(len(item.dependencies) for item in page.tasks) == 270
    assert sum(len(item.outputs) for item in page.tasks) == 430
    assert sum(len(item.validations) for item in page.tasks) == 117
    assert sum(len(item.acceptance) for item in page.tasks) == 116


def test_translation_rejects_live_owner_history_stale_forest_and_terminal_status(
    repo_root: Path,
) -> None:
    population = _population(repo_root)
    with pytest.raises(lifecycle.EAAEFReconciliationBlocked, match="owner to be absent"):
        offline.translate_compiled_eaaef_population(
            population,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
            repo_root=repo_root,
            owner_active=True,
        )
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="historical"):
        offline.translate_compiled_eaaef_population(
            population,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
            repo_root=repo_root,
            owner_active=False,
            historical_task_statuses={"EAAEF-000": "done"},
        )
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="forest"):
        offline.translate_compiled_eaaef_population(
            population,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(accelerator_commit="9" * 40),
            repo_root=repo_root,
            owner_active=False,
        )

    forged_task = _plain(population.bootstrap_tasks[0])
    forged_task["status"] = "done"
    forged_population = _replace_bootstrap_task(population, forged_task)
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="CID differs"):
        offline.translate_compiled_eaaef_population(
            forged_population,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
            repo_root=repo_root,
            owner_active=False,
        )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda task: task["body"]["outputs"].__setitem__(0, "forged/output.txt"), "contract"),
        (
            lambda task: task["body"]["validations"][0]["argv"].__setitem__(0, "false"),
            "contract",
        ),
        (
            lambda task: task["body"]["acceptance"].__setitem__(0, "forged acceptance"),
            "contract",
        ),
        (lambda task: task.__setitem__("priority", "P9"), "contract"),
        (lambda task: task["body"].__setitem__("task_spec_cid", "sha256:" + "f" * 64), "task"),
    ],
)
def test_retained_population_cids_reject_commitment_bearing_task_mutations(
    repo_root: Path,
    mutation: Any,
    expected: str,
) -> None:
    population = _population(repo_root)
    forged_task = _plain(population.bootstrap_tasks[0])
    mutation(forged_task)
    forged = _replace_bootstrap_task(population, forged_task)

    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match=expected):
        offline.translate_compiled_eaaef_population(
            forged,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
            repo_root=repo_root,
            owner_active=False,
        )


def test_forged_dependency_and_held_partition_cids_fail_closed(repo_root: Path) -> None:
    population = _population(repo_root)
    dependency = dict(population.dependencies[0])
    dependency["kind"] = "forged"
    forged_dependencies = replace(
        population,
        dependencies=(dependency, *population.dependencies[1:]),
    )
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="bootstrap"):
        offline.translate_compiled_eaaef_population(
            forged_dependencies,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
            repo_root=repo_root,
            owner_active=False,
        )

    held_task = _plain(population.plan_r2_tasks[0])
    held_task["body"]["outputs"][0] = "forged/held-output.txt"
    forged_held = replace(
        population,
        plan_r2_tasks=(held_task, *population.plan_r2_tasks[1:]),
    )
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="contract"):
        offline.translate_compiled_eaaef_population(
            forged_held,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
            repo_root=repo_root,
            owner_active=False,
        )


@pytest.mark.parametrize(
    ("field_name", "expected"),
    [
        ("execution_contract_population_cid", "execution-contract population"),
        ("goal_population_cid", "goal population"),
        ("bootstrap_population_cid", "bootstrap population"),
        ("plan_r2_population_cid", "held-R2 population"),
        ("population_cid", "overall population"),
    ],
)
def test_each_population_commitment_is_recomputed(
    repo_root: Path,
    field_name: str,
    expected: str,
) -> None:
    population = _population(repo_root)
    forged = replace(population, **{field_name: "sha256:" + "f" * 64})

    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match=expected):
        offline.translate_compiled_eaaef_population(
            forged,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
            repo_root=repo_root,
            owner_active=False,
        )


def test_goal_and_r1_plan_mutations_reject_retained_commitments(repo_root: Path) -> None:
    population = _population(repo_root)
    goal = _plain(population.goals[0])
    goal["title"] = "forged root goal"
    forged_goal = replace(population, goals=(goal, *population.goals[1:]))
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="goal population"):
        offline.translate_compiled_eaaef_population(
            forged_goal,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
            repo_root=repo_root,
            owner_active=False,
        )

    plan = _plain(population.plan_r1)
    plan["body"]["source_head"] = "f" * 40
    forged_plan = replace(population, plan_r1=plan)
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="goal population"):
        offline.translate_compiled_eaaef_population(
            forged_plan,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
            repo_root=repo_root,
            owner_active=False,
        )


def test_current_board_and_task_specs_must_remain_self_addressed(repo_root: Path) -> None:
    board = _board(repo_root)
    board["goals"][0]["title"] = "forged board title"
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="board CID"):
        lifecycle.compile_fresh_eaaef_population(
            board,
            forest=_sealed_forest(),
            repo_root=repo_root,
        )

    board = _board(repo_root)
    board["tasks"][0]["execution_owned_files"][0] = "forged/task-output.txt"
    projection = dict(board)
    projection.pop("board_cid")
    board["board_cid"] = lifecycle._eaaef_source_cid(projection)
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="task spec CID"):
        lifecycle.compile_fresh_eaaef_population(
            board,
            forest=_sealed_forest(),
            repo_root=repo_root,
        )


def test_resealed_forged_rows_still_differ_from_current_sealed_board(repo_root: Path) -> None:
    population = _population(repo_root)
    forged = _reseal_mutated_output_population(population)

    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="sealed board"):
        offline.translate_compiled_eaaef_population(
            forged,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
            repo_root=repo_root,
            owner_active=False,
        )


def test_resealed_board_and_population_must_match_sealed_git_source(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forest = _sealed_forest()
    forged_board = _board(repo_root)
    task = forged_board["tasks"][0]
    task["execution_owned_files"][0] = "forged/resealed-board-output.txt"
    task_projection = dict(task)
    task_projection.pop("task_spec_cid")
    task["task_spec_cid"] = lifecycle._eaaef_source_cid(task_projection)
    board_projection = dict(forged_board)
    board_projection.pop("board_cid")
    forged_board["board_cid"] = lifecycle._eaaef_source_cid(board_projection)

    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="sealed Git board source",
    ):
        lifecycle.compile_fresh_eaaef_population(
            forged_board,
            forest=forest,
            repo_root=repo_root,
        )

    # Reconstruct the fully self-consistent population accepted before the
    # provenance gate, then prove the offline verifier independently rejects it.
    def _allow_unbound_board(
        _board: Mapping[str, Any],
        *,
        sealed_forest: Mapping[str, Any],
    ) -> None:
        assert sealed_forest["source_forest_root"] == forest["source_forest_root"]

    with monkeypatch.context() as bypass:
        bypass.setattr(lifecycle, "_require_current_board_provenance", _allow_unbound_board)
        forged_population = lifecycle.compile_fresh_eaaef_population(
            forged_board,
            forest=forest,
            repo_root=repo_root,
        )

    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="sealed Git board source",
    ):
        offline.translate_compiled_eaaef_population(
            forged_population,
            current_board=forged_board,
            current_forest=forest,
            repo_root=repo_root,
            owner_active=False,
        )


def test_fully_resealed_forest_cannot_claim_blob_absent_from_real_tree(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_root = tmp_path / "trusted-eaaef-repository"
    honest_board_bytes = (repo_root / lifecycle.EAAEF_BOARD_PATH).read_bytes()
    honest_board = json.loads(honest_board_bytes)
    honest_forest = _real_git_forest(trusted_root, board_bytes=honest_board_bytes)
    assert honest_forest["valid"] is True
    monkeypatch.setattr(
        lifecycle,
        "inspect_current_repository_forest",
        _REAL_INSPECT_CURRENT_REPOSITORY_FOREST,
    )
    lifecycle.compile_fresh_eaaef_population(
        honest_board,
        forest=honest_forest,
        repo_root=trusted_root,
    )

    forged_board = _plain(honest_board)
    forged_task = forged_board["tasks"][0]
    forged_task["execution_owned_files"][0] = "forged/absent-tree-output.txt"
    task_projection = dict(forged_task)
    task_projection.pop("task_spec_cid")
    forged_task["task_spec_cid"] = lifecycle._eaaef_source_cid(task_projection)
    board_projection = dict(forged_board)
    board_projection.pop("board_cid")
    forged_board["board_cid"] = lifecycle._eaaef_source_cid(board_projection)
    forged_board_bytes = json.dumps(
        forged_board,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    forged_blob_oid = hashlib.sha1(
        b"blob " + str(len(forged_board_bytes)).encode("ascii") + b"\0" + forged_board_bytes,
        usedforsecurity=False,
    ).hexdigest()
    forged_board_source = lifecycle._board_source_binding(
        forged_board_bytes,
        source_head=honest_forest["source_head"],
        source_tree=honest_forest["source_tree"],
        git_mode="100644",
        blob_oid=forged_blob_oid,
    )
    forged_identity = {
        "schema": lifecycle.EAAEF_FOREST_SCHEMA,
        "repositories": _plain(honest_forest["repositories"]),
        "board_source": forged_board_source,
    }
    forged_forest_root = lifecycle._cid(forged_identity)
    forged_forest = {
        **forged_identity,
        "valid": True,
        "blockers": [],
        "source_head": honest_forest["source_head"],
        "source_tree": honest_forest["source_tree"],
        "source_forest_root": forged_forest_root,
        "source_generation_cid": forged_forest_root,
        "binding_cid": lifecycle._cid(
            {**forged_identity, "source_forest_root": forged_forest_root}
        ),
    }
    tree_entry = _git(
        trusted_root,
        "ls-tree",
        honest_forest["source_tree"],
        "--",
        lifecycle.EAAEF_BOARD_PATH,
    )
    assert honest_forest["board_source"]["blob_oid"] in tree_entry
    assert forged_blob_oid not in tree_entry

    # Reconstruct the population that a descriptor-only gate accepted.
    with monkeypatch.context() as descriptor_only:
        descriptor_only.setattr(
            lifecycle,
            "inspect_current_repository_forest",
            lambda _root: forged_forest,
        )
        forged_population = lifecycle.compile_fresh_eaaef_population(
            forged_board,
            forest=forged_forest,
            repo_root=trusted_root,
        )

    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="trusted current Git inspection",
    ):
        lifecycle.compile_fresh_eaaef_population(
            forged_board,
            forest=forged_forest,
            repo_root=trusted_root,
        )
    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="trusted current Git inspection",
    ):
        offline.translate_compiled_eaaef_population(
            forged_population,
            current_board=forged_board,
            current_forest=forged_forest,
            repo_root=trusted_root,
            owner_active=False,
        )


def test_commitment_failure_occurs_before_offline_sink_call(repo_root: Path) -> None:
    population = _population(repo_root)
    forged_task = _plain(population.bootstrap_tasks[0])
    forged_task["body"]["outputs"][0] = "forged/pre-sink-output.txt"
    forged = _replace_bootstrap_task(population, forged_task)
    sink = _ForbiddenMaterializer()

    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="contract"):
        offline.materialize_offline_eaaef_population(
            sink,
            forged,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
            repo_root=repo_root,
            owner_active=False,
        )
    assert sink.called is False


def _bootstrap_registry_record(
    generation_id: str,
    phase: str,
    *,
    owner_process_birth: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    absence_cid = "sha256:" + "1" * 64
    offline_cid = "sha256:" + "2" * 64 if phase != "absent" else ""
    started = phase == "owner_started"
    value = {
        "schema": EAAEF_CASF_BOOTSTRAP_REGISTRY_SCHEMA,
        "generation_id": generation_id,
        "phase": phase,
        "request_cid": "sha256:" + "3" * 64,
        "source_forest_root": "sha256:" + "4" * 64,
        "population_cid": "sha256:" + "5" * 64,
        "owner_lifecycle_interface": EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
        "absence_attestation_cid": absence_cid,
        "offline_materialization_receipt_cid": offline_cid,
        "owner_start_receipt_cid": "sha256:" + "6" * 64 if started else "",
        "canonical_bootstrap_receipt_cid": (
            "sha256:" + "7" * 64 if started else ""
        ),
        "owner_process_birth": (
            dict(owner_process_birth or {}) if started else None
        ),
    }
    value["record_cid"] = lifecycle._cid(value)
    return value


def test_static_owner_facade_reports_blockers_and_cannot_effect(repo_root: Path) -> None:
    owner = open_eaaef_typed_reconciliation_owner(repo_root=repo_root)
    qualification = owner.reconciliation_qualification()

    assert qualification["interface"] == lifecycle.EAAEF_RECONCILIATION_OWNER_INTERFACE
    assert qualification["source_forest_root"] == ""
    assert qualification["bootstrap_materialization_before_owner_start"] is False
    assert qualification["plan_r2_remote_runtime_qualification_status"] != (
        "production_qualified"
    )
    assert qualification["plan_r2_remote_runtime_blockers"] == list(
        EAAEF_OWNER_PRODUCTION_BLOCKERS
    )
    assert qualification["provider_launch_allowed"] is False
    assert qualification["qualification_cid"] == lifecycle._cid(
        {key: value for key, value in qualification.items() if key != "qualification_cid"}
    )
    facade_source = inspect.getsource(type(owner))
    assert "duckdb.connect(" not in facade_source
    assert "os.kill(" not in facade_source
    assert "subprocess." not in facade_source
    assert "signal." not in facade_source
    assert "read_text(" not in facade_source
    with pytest.raises(lifecycle.EAAEFReconciliationBlocked, match="qualification differs"):
        lifecycle.require_typed_reconciliation_owner(owner)
    effect_methods = (
        owner.materialize_offline_population,
        owner.apply_signed_plan_r2,
        owner.launch_reconciliation_supervisor,
        owner.reconciliation_status_snapshot,
        owner.stop_reconciliation_tracks,
    )
    for method in effect_methods:
        with pytest.raises(EAAEFTypedReconciliationOwnerUnavailable):
            if method.__name__ == "materialize_offline_population":
                method({}, population=object())
            elif method.__name__ == "apply_signed_plan_r2":
                method({}, population=object(), authority=object())
            else:
                method({})


def test_explicit_bootstrap_opener_requires_complete_host_local_binding(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    population = _population(repo_root)
    snapshot_bindings = _concrete_snapshot_bindings()

    # The statically named resolver supplies only repo_root.  Merely adding the
    # source seam must therefore remain an inert, typed no-go.
    assert open_eaaef_bootstrap_reconciliation_owner(repo_root=repo_root) is None
    with pytest.raises(
        lifecycle.EAAEFReconciliationBlocked,
        match="bootstrap_portfolio_materialization_owner_unavailable",
    ):
        lifecycle.resolve_bootstrap_reconciliation_owner(repo_root)
    with pytest.raises(
        EAAEFCASFBootstrapOwnerError,
        match="complete explicit host binding",
    ):
        open_eaaef_bootstrap_reconciliation_owner(
            repo_root=repo_root,
            registry_root=tmp_path / "incomplete",
        )
    with pytest.raises(
        EAAEFCASFBootstrapOwnerError,
        match="exact absolute path",
    ):
        open_eaaef_bootstrap_reconciliation_owner(
            repo_root=repo_root,
            registry_root=Path("relative-state"),
            source_forest_root=population.source_forest_root,
            snapshot_bindings=snapshot_bindings,
        )

    registry_root = tmp_path / "explicit-bootstrap-owner"
    owner = open_eaaef_bootstrap_reconciliation_owner(
        repo_root=repo_root,
        registry_root=registry_root,
        source_forest_root=population.source_forest_root,
        snapshot_bindings=snapshot_bindings,
        startup_timeout_seconds=60,
        operation_timeout_seconds=60,
        shutdown_timeout_seconds=30,
    )
    assert owner is not None
    qualification = owner.bootstrap_reconciliation_qualification()
    assert qualification["bootstrap_owner_ready"] is True
    assert qualification["provider_launch_allowed"] is False
    assert owner._owner_lifecycle.snapshot_bindings is snapshot_bindings
    assert not registry_root.exists()


def test_casf_bootstrap_binding_holds_owner_guard_through_commit_and_start(
    repo_root: Path,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    if not DatabaseTaskSource.available():
        pytest.skip("DuckDB unavailable")
    population = _population(repo_root)
    owner_lifecycle = _FakeCASFBootstrapLifecycle(population)
    request.addfinalizer(owner_lifecycle.cleanup)
    assert owner_lifecycle.NONPRODUCTION_TEST_DOUBLE is True
    registry_root = tmp_path / "casf-bootstrap-registry"
    owner = bind_eaaef_casf_bootstrap_owner(
        repo_root=repo_root,
        registry_root=registry_root,
        source_forest_root=population.source_forest_root,
        owner_lifecycle=owner_lifecycle,
    )
    qualification = owner.bootstrap_reconciliation_qualification()
    assert owner.INTERFACE == EAAEF_BOOTSTRAP_RECONCILIATION_OWNER_INTERFACE
    assert qualification["bootstrap_materialization_before_owner_start"] is True
    assert qualification["bootstrap_owner_ready"] is False
    assert qualification["bootstrap_owner_blockers"] == [
        "casf_quack_exclusive_owner_lifecycle_not_bound"
    ]
    assert qualification["provider_launch_allowed"] is False
    with pytest.raises(
        lifecycle.EAAEFReconciliationBlocked,
        match="bootstrap reconciliation owner qualification differs",
    ):
        lifecycle.require_bootstrap_reconciliation_owner(
            owner,
            source_forest_root=population.source_forest_root,
        )
    with pytest.raises(
        lifecycle.EAAEFReconciliationBlocked,
        match="typed_portfolio_materialization_owner_unavailable",
    ):
        lifecycle.require_typed_reconciliation_owner(
            owner,
            source_forest_root=population.source_forest_root,
        )

    generation_id = "eaaef-bootstrap-test-001"
    offline_request = lifecycle._build_offline_population_request(
        generation_id=generation_id,
        population=population,
    )
    receipt = owner.materialize_offline_population(
        offline_request,
        population=population,
    )

    assert owner_lifecycle.events == [
        "exclusive_guard_acquired",
        "owner_absence_attested",
        "owner_start_requested",
        "owner_commit_requested",
        "exclusive_guard_released",
    ]
    assert receipt["task_count"] == 116
    assert receipt["task_status_counts"] == {"blocked": 94, "todo": 22}
    assert receipt["owner_started_after_bootstrap"] is True
    assert receipt["provider_process_started"] is False
    record_path = (
        registry_root
        / "generations"
        / generation_id
        / "bootstrap-owner.json"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["schema"] == EAAEF_CASF_BOOTSTRAP_REGISTRY_SCHEMA
    assert record["phase"] == "owner_started"
    assert record["request_cid"] == offline_request["request_cid"]
    assert record["canonical_bootstrap_receipt_cid"] == receipt["receipt_cid"]
    assert record["owner_process_birth"] is not None
    assert record["record_cid"] == lifecycle._cid(
        {key: value for key, value in record.items() if key != "record_cid"}
    )
    assert str(registry_root) not in json.dumps(receipt, sort_keys=True)
    generation_dir = record_path.parent
    assert registry_root.stat().st_mode & 0o777 == 0o700
    assert generation_dir.stat().st_mode & 0o777 == 0o700
    assert generation_dir.parent.stat().st_mode & 0o777 == 0o700
    assert (registry_root / ".bootstrap-owner.lock").stat().st_mode & 0o777 == 0o600
    assert record_path.stat().st_mode & 0o777 == 0o600
    assert (generation_dir / "control.duckdb").stat().st_mode & 0o777 == 0o600
    guard = owner_lifecycle.guards[0]
    assert guard.NONPRODUCTION_TEST_DOUBLE is True
    assert guard.start_receipt is not None
    assert guard.absence is not None
    assert guard.start_receipt["absence_attestation_cid"] == (
        guard.absence["attestation_cid"]
    )
    assert guard.start_receipt["offline_materialization_receipt_cid"] == (
        record["offline_materialization_receipt_cid"]
    )
    assert guard.process_birth is not None
    assert lifecycle.inspect_process_birth(guard.process_birth.pid) == guard.process_birth


def test_casf_bootstrap_registry_rejects_intermediate_symlink(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    population = _population(repo_root)
    owner_lifecycle = _FakeCASFBootstrapLifecycle(population)
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked-registry"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(EAAEFCASFBootstrapOwnerError, match="contains a symlink"):
        bind_eaaef_casf_bootstrap_owner(
            repo_root=repo_root,
            registry_root=link / "nested",
            source_forest_root=population.source_forest_root,
            owner_lifecycle=owner_lifecycle,
        )


def test_casf_bootstrap_registry_seals_existing_generations_parent(
    tmp_path: Path,
) -> None:
    registry = EAAEFCASFBootstrapRegistry(tmp_path / "registry")
    with registry.exclusive() as capability:
        generations = registry.root / "generations"
        generations.mkdir(mode=0o700)
        generations.chmod(0o777)
        registry.prepare_generation(capability, "eaaef-bootstrap-mode-001")

    assert generations.stat().st_mode & 0o777 == 0o700


def test_casf_bootstrap_registry_rejects_symlinked_generations_parent(
    tmp_path: Path,
) -> None:
    registry = EAAEFCASFBootstrapRegistry(tmp_path / "registry")
    target = tmp_path / "outside-generations"
    target.mkdir(mode=0o700)
    with registry.exclusive() as capability:
        (registry.root / "generations").symlink_to(target, target_is_directory=True)
        with pytest.raises(EAAEFCASFBootstrapOwnerError, match="parent is unsafe"):
            registry.prepare_generation(capability, "eaaef-bootstrap-symlink-001")


def test_casf_bootstrap_registry_requires_capability_and_monotonic_transitions(
    tmp_path: Path,
) -> None:
    registry = EAAEFCASFBootstrapRegistry(tmp_path / "registry")
    generation_id = "eaaef-bootstrap-monotonic-001"
    birth = lifecycle.inspect_process_birth(os.getpid())
    assert birth is not None
    absent = _bootstrap_registry_record(generation_id, "absent")
    offline_record = _bootstrap_registry_record(generation_id, "offline_committed")
    owner_started = _bootstrap_registry_record(
        generation_id,
        "owner_started",
        owner_process_birth=birth.to_dict(),
    )

    with pytest.raises(EAAEFCASFBootstrapOwnerError, match="held-lock capability"):
        registry.prepare_generation(object(), generation_id)
    with registry.exclusive() as capability:
        registry.prepare_generation(capability, generation_id)
        with pytest.raises(EAAEFCASFBootstrapOwnerError, match="held-lock capability"):
            registry.write_record(object(), generation_id, absent)
        registry.write_record(capability, generation_id, absent)
        with pytest.raises(EAAEFCASFBootstrapOwnerError, match="not monotonic"):
            registry.write_record(capability, generation_id, owner_started)
        registry.write_record(capability, generation_id, offline_record)
        with pytest.raises(EAAEFCASFBootstrapOwnerError, match="not monotonic"):
            registry.write_record(capability, generation_id, absent)

        malformed_birth = birth.to_dict()
        malformed_birth["pid"] = True
        malformed_owner = _bootstrap_registry_record(
            generation_id,
            "owner_started",
            owner_process_birth=malformed_birth,
        )
        with pytest.raises(EAAEFCASFBootstrapOwnerError, match="birth types differ"):
            registry.write_record(capability, generation_id, malformed_owner)

        registry.write_record(capability, generation_id, owner_started)
        with pytest.raises(EAAEFCASFBootstrapOwnerError, match="not monotonic"):
            registry.write_record(capability, generation_id, offline_record)
        stale_capability = capability

    with pytest.raises(EAAEFCASFBootstrapOwnerError, match="held-lock capability"):
        registry.write_record(stale_capability, generation_id, owner_started)


def test_casf_bootstrap_start_failure_persists_offline_commit_and_never_rewrites(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    if not DatabaseTaskSource.available():
        pytest.skip("DuckDB unavailable")
    population = _population(repo_root)
    owner_lifecycle = _FakeCASFBootstrapLifecycle(
        population,
        start_error=RuntimeError("injected owner start failure"),
    )
    registry_root = tmp_path / "casf-bootstrap-registry"
    owner = bind_eaaef_casf_bootstrap_owner(
        repo_root=repo_root,
        registry_root=registry_root,
        source_forest_root=population.source_forest_root,
        owner_lifecycle=owner_lifecycle,
    )
    generation_id = "eaaef-bootstrap-test-002"
    request = lifecycle._build_offline_population_request(
        generation_id=generation_id,
        population=population,
    )

    with pytest.raises(RuntimeError, match="injected owner start failure"):
        owner.materialize_offline_population(request, population=population)
    assert owner_lifecycle.events[-2:] == [
        "owner_abort_requested",
        "exclusive_guard_released",
    ]
    assert owner_lifecycle.events[-1] == "exclusive_guard_released"
    generation_dir = registry_root / "generations" / generation_id
    record_path = generation_dir / "bootstrap-owner.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["phase"] == "offline_committed"
    assert record["owner_start_receipt_cid"] == ""
    assert record["canonical_bootstrap_receipt_cid"] == ""
    assert record["owner_process_birth"] is None
    database_path = generation_dir / "control.duckdb"
    original_database_size = database_path.stat().st_size

    with pytest.raises(EAAEFCASFBootstrapOwnerError, match="durable state"):
        owner.materialize_offline_population(request, population=population)
    assert database_path.stat().st_size == original_database_size


@pytest.mark.parametrize(
    ("start_mutation", "message"),
    [
        (
            ("offline_materialization_receipt_cid", "sha256:" + "f" * 64),
            "start evidence differs",
        ),
        (("owner_started_after_bootstrap", 1), "start evidence differs"),
    ],
)
def test_casf_bootstrap_malformed_post_start_receipt_aborts_provisional_owner(
    repo_root: Path,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    start_mutation: tuple[str, Any],
    message: str,
) -> None:
    if not DatabaseTaskSource.available():
        pytest.skip("DuckDB unavailable")
    population = _population(repo_root)
    owner_lifecycle = _FakeCASFBootstrapLifecycle(
        population,
        start_mutation=start_mutation,
    )
    request.addfinalizer(owner_lifecycle.cleanup)
    registry_root = tmp_path / "casf-bootstrap-registry"
    owner = bind_eaaef_casf_bootstrap_owner(
        repo_root=repo_root,
        registry_root=registry_root,
        source_forest_root=population.source_forest_root,
        owner_lifecycle=owner_lifecycle,
    )
    generation_id = "eaaef-bootstrap-malformed-start-001"
    offline_request = lifecycle._build_offline_population_request(
        generation_id=generation_id,
        population=population,
    )

    with pytest.raises(EAAEFCASFBootstrapOwnerError, match=message):
        owner.materialize_offline_population(offline_request, population=population)

    assert owner_lifecycle.events[-2:] == [
        "owner_abort_requested",
        "exclusive_guard_released",
    ]
    guard = owner_lifecycle.guards[0]
    assert guard.process_birth is not None
    assert lifecycle.inspect_process_birth(guard.process_birth.pid) != guard.process_birth
    record = json.loads(
        (
            registry_root
            / "generations"
            / generation_id
            / "bootstrap-owner.json"
        ).read_text(encoding="utf-8")
    )
    assert record["phase"] == "offline_committed"


def test_casf_bootstrap_final_record_failure_aborts_provisional_owner(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if not DatabaseTaskSource.available():
        pytest.skip("DuckDB unavailable")
    population = _population(repo_root)
    owner_lifecycle = _FakeCASFBootstrapLifecycle(population)
    request.addfinalizer(owner_lifecycle.cleanup)
    registry_root = tmp_path / "casf-bootstrap-registry"
    owner = bind_eaaef_casf_bootstrap_owner(
        repo_root=repo_root,
        registry_root=registry_root,
        source_forest_root=population.source_forest_root,
        owner_lifecycle=owner_lifecycle,
    )
    generation_id = "eaaef-bootstrap-final-write-failure-001"
    offline_request = lifecycle._build_offline_population_request(
        generation_id=generation_id,
        population=population,
    )
    original_write_record = owner._registry.write_record

    def _fail_final_record(
        capability: object,
        selected_generation_id: str,
        record: Mapping[str, Any],
    ) -> None:
        original_write_record(capability, selected_generation_id, record)
        if record.get("phase") == "owner_started":
            raise OSError("injected final registry postcondition failure")

    monkeypatch.setattr(owner._registry, "write_record", _fail_final_record)

    with pytest.raises(OSError, match="injected final registry postcondition failure"):
        owner.materialize_offline_population(offline_request, population=population)

    assert owner_lifecycle.events[-2:] == [
        "owner_abort_requested",
        "exclusive_guard_released",
    ]
    assert "owner_commit_requested" not in owner_lifecycle.events
    guard = owner_lifecycle.guards[0]
    assert guard.process_birth is not None
    assert lifecycle.inspect_process_birth(guard.process_birth.pid) != guard.process_birth
    record = json.loads(
        (
            registry_root
            / "generations"
            / generation_id
            / "bootstrap-owner.json"
        ).read_text(encoding="utf-8")
    )
    assert record["phase"] == "owner_started"
    with pytest.raises(EAAEFCASFBootstrapOwnerError, match="durable state"):
        owner.materialize_offline_population(offline_request, population=population)


def test_persistent_casf_bootstrap_owner_preserves_lease_through_quack_start(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not DatabaseTaskSource.available():
        pytest.skip("DuckDB unavailable")
    population = _population(repo_root)
    concrete = casf_lifecycle.QuackEAAEFCASFBootstrapOwnerLifecycle(
        snapshot_bindings=_concrete_snapshot_bindings(),
        startup_timeout_seconds=60,
        operation_timeout_seconds=180,
        shutdown_timeout_seconds=30,
    )
    registry_root = tmp_path / "persistent-casf-bootstrap"
    owner = bind_eaaef_casf_bootstrap_owner(
        repo_root=repo_root,
        registry_root=registry_root,
        source_forest_root=population.source_forest_root,
        owner_lifecycle=concrete,
    )
    qualification = owner.bootstrap_reconciliation_qualification()
    assert qualification["interface"] == EAAEF_BOOTSTRAP_RECONCILIATION_OWNER_INTERFACE
    assert qualification["bootstrap_owner_ready"] is True
    assert qualification["bootstrap_owner_blockers"] == []
    assert (
        lifecycle.require_bootstrap_reconciliation_owner(
            owner,
            source_forest_root=population.source_forest_root,
        )
        is owner
    )
    with pytest.raises(
        lifecycle.EAAEFReconciliationBlocked,
        match="typed_portfolio_materialization_owner_unavailable",
    ):
        lifecycle.require_typed_reconciliation_owner(
            owner,
            source_forest_root=population.source_forest_root,
        )
    generation_id = "eaaef-persistent-success-001"
    request = lifecycle._build_offline_population_request(
        generation_id=generation_id,
        population=population,
    )
    generation_dir = registry_root / "generations" / generation_id
    database_path = generation_dir / "control.duckdb"
    lock_path = database_path.with_name(f".{database_path.name}.state-owner.lock")
    marker_path = database_path.with_name(f".{database_path.name}.state-owner.json")
    observed: dict[str, Any] = {}
    original_materialize = offline.materialize_offline_eaaef_population

    def _observe_held_offline_lease(*args: Any, **kwargs: Any) -> dict[str, Any]:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        observed["offline_marker"] = marker
        contender = ExclusiveOwnerLease(
            lock_path=lock_path,
            marker_path=marker_path,
        )
        with pytest.raises(QuackStateServerOwnershipError, match="exclusive lock"):
            contender.acquire(
                server_id="eaaef-test-contender",
                process_birth=current_process_birth(),
                database_path=database_path,
                generation=1,
            )
        return original_materialize(*args, **kwargs)

    monkeypatch.setattr(
        offline,
        "materialize_offline_eaaef_population",
        _observe_held_offline_lease,
    )
    receipt = owner.materialize_offline_population(request, population=population)

    assert receipt["task_status_counts"] == {"blocked": 94, "todo": 22}
    assert receipt["provider_process_started"] is False
    offline_marker = observed["offline_marker"]
    live_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert live_marker["fence_token"] == offline_marker["fence_token"]
    assert live_marker["process_birth"] == offline_marker["process_birth"]
    assert live_marker["server_id"] != offline_marker["server_id"]
    assert concrete.committed_generation_ids() == (generation_id,)
    with pytest.raises(EAAEFCASFBootstrapOwnerError, match="later effects remain unavailable"):
        owner.apply_signed_plan_r2({}, population=population, authority=object())
    with pytest.raises(EAAEFCASFBootstrapOwnerError, match="later effects remain unavailable"):
        owner.reconciliation_status_snapshot({})
    with pytest.raises(EAAEFCASFBootstrapOwnerError, match="later effects remain unavailable"):
        owner.stop_reconciliation_tracks({})
    with pytest.raises(EAAEFCASFBootstrapOwnerError, match="later effects remain unavailable"):
        owner.launch_reconciliation_supervisor({})

    broker = concrete._brokers[generation_id]
    assert broker.start_receipt is not None
    replay = casf_lifecycle._build_request(
        generation_id=generation_id,
        sequence=2,
        operation="commit_started_owner",
        arguments={
            "owner_start_receipt_cid": broker.start_receipt["start_receipt_cid"],
            "final_record_cid": broker.final_record_cid,
        },
    )
    replay_raw = casf_lifecycle._canonical_bytes(
        replay, noun="test exact replay"
    )
    replay_response = casf_lifecycle._validate_response(
        broker.exchange_raw_for_test(replay_raw),
        generation_id=generation_id,
        sequence=2,
        operation="commit_started_owner",
        request_cid=replay["request_cid"],
    )
    assert replay_response["ok"] is True

    # The durable owner_started record is the handoff point: closing the
    # caller-death sentinel after commit must leave the owner live.
    os.close(broker.death_writer)
    broker.death_writer = -1
    time.sleep(0.1)
    assert broker.process.poll() is None
    assert marker_path.exists()

    divergent = casf_lifecycle._build_request(
        generation_id=generation_id,
        sequence=2,
        operation="commit_started_owner",
        arguments={
            "owner_start_receipt_cid": broker.start_receipt["start_receipt_cid"],
            "final_record_cid": "sha256:" + "f" * 64,
        },
    )
    divergent_raw = casf_lifecycle._canonical_bytes(
        divergent, noun="test divergent replay"
    )
    divergent_response = casf_lifecycle._validate_response(
        broker.exchange_raw_for_test(divergent_raw),
        generation_id=generation_id,
        sequence=2,
        operation="commit_started_owner",
        request_cid=divergent["request_cid"],
    )
    assert divergent_response["ok"] is False
    assert divergent_response["error_code"] == "broker_frame_diverged"
    broker.close_descriptors()
    assert broker.wait_dead(30)
    concrete._forget_broker(generation_id, broker)
    assert not marker_path.exists()


def test_persistent_casf_bootstrap_aborts_after_final_record_failure(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not DatabaseTaskSource.available():
        pytest.skip("DuckDB unavailable")
    population = _population(repo_root)
    concrete = casf_lifecycle.QuackEAAEFCASFBootstrapOwnerLifecycle(
        snapshot_bindings=_concrete_snapshot_bindings(),
        startup_timeout_seconds=60,
        operation_timeout_seconds=180,
        shutdown_timeout_seconds=30,
    )
    registry_root = tmp_path / "persistent-casf-abort"
    owner = bind_eaaef_casf_bootstrap_owner(
        repo_root=repo_root,
        registry_root=registry_root,
        source_forest_root=population.source_forest_root,
        owner_lifecycle=concrete,
    )
    generation_id = "eaaef-persistent-abort-001"
    request = lifecycle._build_offline_population_request(
        generation_id=generation_id,
        population=population,
    )
    original_write_record = owner._registry.write_record

    def _fail_after_final_record(
        capability: object,
        selected_generation_id: str,
        record: Mapping[str, Any],
    ) -> None:
        original_write_record(capability, selected_generation_id, record)
        if record.get("phase") == "owner_started":
            raise OSError("injected persistent-owner final record failure")

    monkeypatch.setattr(owner._registry, "write_record", _fail_after_final_record)
    with pytest.raises(OSError, match="persistent-owner final record failure"):
        owner.materialize_offline_population(request, population=population)

    generation_dir = registry_root / "generations" / generation_id
    record = json.loads(
        (generation_dir / "bootstrap-owner.json").read_text(encoding="utf-8")
    )
    birth = lifecycle.ProcessBirth.from_mapping(record["owner_process_birth"])
    assert lifecycle.inspect_process_birth(birth.pid) != birth
    assert concrete.committed_generation_ids() == ()
    assert not (generation_dir / ".control.duckdb.state-owner.json").exists()


@pytest.mark.parametrize(
    ("artifact_name", "payload"),
    [
        (casf_management.MANAGEMENT_KEY_NAME, b"k" * 32),
        (casf_management.MANAGEMENT_CAPSULE_NAME, b"{}"),
    ],
)
def test_persistent_casf_partial_management_artifact_never_acquires_lease(
    tmp_path: Path,
    artifact_name: str,
    payload: bytes,
) -> None:
    if not DatabaseTaskSource.available():
        pytest.skip("DuckDB unavailable")
    binding = _concrete_bootstrap_binding(
        tmp_path / f"partial-management-{artifact_name}",
        generation_id="eaaef-partial-management-001",
    )
    binding.owner_state_dir.mkdir(mode=0o700)
    artifact = binding.owner_state_dir / artifact_name
    artifact.write_bytes(payload)
    artifact.chmod(0o600)
    lifecycle_owner = casf_lifecycle.QuackEAAEFCASFBootstrapOwnerLifecycle(
        snapshot_bindings=_concrete_snapshot_bindings(),
        startup_timeout_seconds=30,
        operation_timeout_seconds=30,
        shutdown_timeout_seconds=10,
    )
    marker_path = binding.database_path.with_name(
        f".{binding.database_path.name}.state-owner.json"
    )
    lock_path = binding.database_path.with_name(
        f".{binding.database_path.name}.state-owner.lock"
    )

    for _attempt in range(2):
        with pytest.raises((EAAEFCASFBootstrapOwnerError, EOFError)):
            with lifecycle_owner.hold_exclusive_bootstrap(binding):
                pytest.fail("quarantined generation acquired its owner lease")
        assert not marker_path.exists()
        assert not lock_path.exists()
        assert lifecycle_owner.committed_generation_ids() == ()


def test_persistent_casf_bootstrap_owner_reattaches_and_stops_privately(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not DatabaseTaskSource.available():
        pytest.skip("DuckDB unavailable")
    population = _population(repo_root)
    snapshot_bindings = _concrete_snapshot_bindings()
    registry_root = tmp_path / "persistent-casf-reattach"
    owner = open_eaaef_bootstrap_reconciliation_owner(
        repo_root=repo_root,
        registry_root=registry_root,
        source_forest_root=population.source_forest_root,
        snapshot_bindings=snapshot_bindings,
        startup_timeout_seconds=60,
        operation_timeout_seconds=180,
        shutdown_timeout_seconds=30,
    )
    assert owner is not None
    original_lifecycle = owner._owner_lifecycle
    generation_id = "eaaef-persistent-reattach-001"
    request = lifecycle._build_offline_population_request(
        generation_id=generation_id,
        population=population,
    )
    receipt = owner.materialize_offline_population(request, population=population)
    assert receipt["task_count"] == 116
    assert receipt["task_status_counts"] == {"blocked": 94, "todo": 22}
    assert receipt["provider_process_started"] is False

    generation_dir = registry_root / "generations" / generation_id
    binding = owner._binding_for_population(generation_id, population)
    broker = original_lifecycle._brokers[generation_id]
    owner_birth = dict(broker.start_receipt["owner_process_birth"])
    marker_path = generation_dir / ".control.duckdb.state-owner.json"
    marker_before = marker_path.read_bytes()
    marker_inode = marker_path.stat().st_ino

    # Simulate loss of every inherited caller descriptor and Python object.
    broker.close_descriptors()
    original_lifecycle._forget_broker(generation_id, broker)
    time.sleep(0.1)
    assert broker.process.poll() is None
    assert lifecycle.inspect_process_birth(owner_birth["pid"]).to_dict() == owner_birth

    # An incomplete generation and a snapshot-binding mismatch both fail before
    # any second broker, database connection, or owner lease can be opened.
    partial_generation = "eaaef-persistent-partial-001"
    partial_dir = registry_root / "generations" / partial_generation
    partial_dir.mkdir(mode=0o700)
    (partial_dir / "casf-owner").mkdir(mode=0o700)
    partial_owner = open_eaaef_bootstrap_reconciliation_owner(
        repo_root=repo_root,
        registry_root=registry_root,
        source_forest_root=population.source_forest_root,
        snapshot_bindings=snapshot_bindings,
        startup_timeout_seconds=60,
        operation_timeout_seconds=180,
        shutdown_timeout_seconds=30,
    )
    assert partial_owner is not None
    with pytest.raises(EAAEFCASFBootstrapOwnerError, match="artifact.*unavailable"):
        partial_owner.reattach_committed_owner(partial_generation)
    assert partial_owner._owner_lifecycle._brokers == {}
    assert not (partial_dir / ".control.duckdb.state-owner.lock").exists()
    assert not (partial_dir / ".control.duckdb.state-owner.json").exists()

    divergent_snapshot = replace(snapshot_bindings, lease_id="different-lease")
    divergent_owner = open_eaaef_bootstrap_reconciliation_owner(
        repo_root=repo_root,
        registry_root=registry_root,
        source_forest_root=population.source_forest_root,
        snapshot_bindings=divergent_snapshot,
        startup_timeout_seconds=60,
        operation_timeout_seconds=180,
        shutdown_timeout_seconds=30,
    )
    assert divergent_owner is not None
    with pytest.raises(EAAEFCASFBootstrapOwnerError, match="stale or divergent"):
        divergent_owner.reattach_committed_owner(generation_id)
    assert divergent_owner._owner_lifecycle._brokers == {}
    assert lifecycle.inspect_process_birth(owner_birth["pid"]).to_dict() == owner_birth

    recovered = open_eaaef_bootstrap_reconciliation_owner(
        repo_root=repo_root,
        registry_root=registry_root,
        source_forest_root=population.source_forest_root,
        snapshot_bindings=snapshot_bindings,
        startup_timeout_seconds=60,
        operation_timeout_seconds=180,
        shutdown_timeout_seconds=30,
    )
    assert recovered is not None
    recovered_lifecycle = recovered._owner_lifecycle

    def _forbid_second_broker(_binding: EAAEFCASFBootstrapBinding) -> object:
        pytest.fail("management reattachment attempted to open a second broker")

    monkeypatch.setattr(recovered_lifecycle, "_open_broker", _forbid_second_broker)
    status = recovered.reattach_committed_owner(generation_id)
    assert status["phase"] == "committed"
    assert status["owner_process_birth"] == owner_birth
    assert status["owner_process_alive"] is True
    assert recovered.reattach_committed_owner(generation_id) == status
    assert recovered_lifecycle.committed_owner_status(generation_id) == status
    assert recovered_lifecycle.committed_generation_ids() == (generation_id,)
    assert recovered_lifecycle._brokers == {}
    assert marker_path.read_bytes() == marker_before
    assert marker_path.stat().st_ino == marker_inode
    assert lifecycle.inspect_process_birth(owner_birth["pid"]).to_dict() == owner_birth
    boundary = json.dumps(status, sort_keys=True)
    for forbidden in (
        "database_path",
        "control.duckdb",
        "transport_token",
        "management.key",
        "SELECT ",
    ):
        assert forbidden not in boundary

    result = recovered.shutdown_committed_owner(generation_id)
    assert result["committed_owner_stopped"] is True
    assert result["exclusive_owner_lease_released"] is True
    assert result["task_state_mutated"] is False
    assert broker.wait_dead(30)
    assert lifecycle.inspect_process_birth(owner_birth["pid"]) is None
    assert not marker_path.exists()
    assert recovered_lifecycle.committed_generation_ids() == ()

    # Only after the exact owner birth and lease are gone may the test open the
    # database independently to verify the materialized bootstrap projection.
    with DatabaseTaskSource(binding.database_path) as source:
        page = source.list_tasks(limit=200)
    assert len(page.tasks) == 116
    assert Counter(item.status for item in page.tasks) == {
        "blocked": 94,
        "todo": 22,
    }

    terminal_controller = open_eaaef_bootstrap_reconciliation_owner(
        repo_root=repo_root,
        registry_root=registry_root,
        source_forest_root=population.source_forest_root,
        snapshot_bindings=snapshot_bindings,
        startup_timeout_seconds=60,
        operation_timeout_seconds=180,
        shutdown_timeout_seconds=30,
    )
    assert terminal_controller is not None
    assert terminal_controller.adopt_completed_owner_stop(generation_id) == result
    assert terminal_controller._owner_lifecycle.committed_generation_ids() == ()
    with pytest.raises(EAAEFCASFBootstrapOwnerError, match="stale or divergent"):
        terminal_controller.reattach_committed_owner(generation_id)

    owner_state = binding.owner_state_dir
    assert stat.S_IMODE(owner_state.stat().st_mode) == 0o700
    for name in (
        casf_management.MANAGEMENT_CAPSULE_NAME,
        casf_management.MANAGEMENT_KEY_NAME,
        casf_management.MANAGEMENT_STOP_INTENT_NAME,
        casf_management.MANAGEMENT_STOP_RESULT_NAME,
    ):
        artifact = owner_state / name
        assert artifact.is_file()
        assert stat.S_IMODE(artifact.stat().st_mode) == 0o600


def test_persistent_casf_bootstrap_releases_lease_when_caller_dies(
    tmp_path: Path,
) -> None:
    generation_id = "eaaef-caller-death-001"
    binding = _concrete_bootstrap_binding(
        tmp_path / "caller-death-generation",
        generation_id=generation_id,
    )
    marker_path = binding.database_path.with_name(
        f".{binding.database_path.name}.state-owner.json"
    )
    lock_path = binding.database_path.with_name(
        f".{binding.database_path.name}.state-owner.lock"
    )
    helper = """
import signal
import sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import ed25519_did_key
from ipfs_accelerate_py.agent_supervisor.runtime.eaaef_casf_bootstrap_lifecycle import EAAEFCASFBootstrapSnapshotBindings, QuackEAAEFCASFBootstrapOwnerLifecycle
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_casf_bootstrap_owner import EAAEFCASFBootstrapBinding
root = Path(sys.argv[1])
binding = EAAEFCASFBootstrapBinding(
    generation_id="eaaef-caller-death-001", source_head="1" * 40,
    source_tree="2" * 40, source_forest_root="sha256:" + "3" * 64,
    board_cid="sha256:" + "4" * 64, population_cid="sha256:" + "5" * 64,
    bootstrap_population_cid="sha256:" + "6" * 64,
    plan_r1_cid="sha256:" + "7" * 64, database_path=root / "control.duckdb",
    owner_state_dir=root / "casf-owner",
)
snapshot = EAAEFCASFBootstrapSnapshotBindings(
    bootstrap_admission_cid="sha256:" + "a" * 64,
    r1_launch_capsule_cid="sha256:" + "b" * 64,
    quack_owner_qualification_cid="sha256:" + "c" * 64,
    quack_command_fabric_qualification_cid="sha256:" + "d" * 64,
    owner_principal_did=ed25519_did_key(bytes([11]) * 32), shard_id="fresh-shard",
    store_id="fresh-store", lease_id="fresh-lease", expected_event_cursor="0",
    request_id="fresh-request", idempotency_key="fresh-idempotency",
    issued_at_ms=100000, deadline_ms=200000, expires_at_ms=300000,
    one_use_nonce="fresh-nonce",
)
lifecycle = QuackEAAEFCASFBootstrapOwnerLifecycle(
    snapshot_bindings=snapshot, startup_timeout_seconds=30,
    operation_timeout_seconds=30, shutdown_timeout_seconds=10,
)
with lifecycle.hold_exclusive_bootstrap(binding):
    print("READY", flush=True)
    signal.pause()
"""
    caller = subprocess.Popen(
        [sys.executable, "-B", "-c", helper, str(binding.database_path.parent)],
        cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert caller.stdout is not None
    try:
        ready = caller.stdout.readline().strip()
        if ready != "READY":
            assert caller.stderr is not None
            pytest.fail("caller helper failed: " + caller.stderr.read())
        owner_birth = json.loads(marker_path.read_text(encoding="utf-8"))[
            "process_birth"
        ]
        caller.kill()
        caller.wait(timeout=10)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if (
                not marker_path.exists()
                and lifecycle.inspect_process_birth(int(owner_birth["pid"])) is None
            ):
                break
            time.sleep(0.02)
        assert not marker_path.exists()
        contender = ExclusiveOwnerLease(
            lock_path=lock_path,
            marker_path=marker_path,
        )
        contender.acquire(
            server_id="caller-death-recovery",
            process_birth=current_process_birth(),
            database_path=binding.database_path,
            generation=1,
        )
        contender.release()
        assert lifecycle.inspect_process_birth(int(owner_birth["pid"])) is None
    finally:
        if caller.poll() is None:
            caller.kill()
            caller.wait(timeout=10)


@pytest.mark.parametrize(
    ("forbidden_key", "forbidden_value"),
    [
        ("database_path", "/tmp/attacker/control.duckdb"),
        ("transport_token", "raw-secret"),
        ("sql", "SELECT * FROM tasks"),
    ],
)
def test_persistent_casf_bootstrap_broker_rejects_authority_injection(
    tmp_path: Path,
    forbidden_key: str,
    forbidden_value: str,
) -> None:
    generation_id = "eaaef-frame-attack-001"
    binding = _concrete_bootstrap_binding(
        tmp_path / f"frame-attack-{forbidden_key}",
        generation_id=generation_id,
    )
    concrete = casf_lifecycle.QuackEAAEFCASFBootstrapOwnerLifecycle(
        snapshot_bindings=_concrete_snapshot_bindings(),
        startup_timeout_seconds=30,
        operation_timeout_seconds=30,
        shutdown_timeout_seconds=10,
    )
    with concrete.hold_exclusive_bootstrap(binding) as guard:
        broker = guard._broker
        assert broker is not None
        attack: dict[str, Any] = {
            "schema": casf_lifecycle.EAAEF_CASF_BOOTSTRAP_BROKER_REQUEST_SCHEMA,
            "interface": casf_lifecycle.EAAEF_CASF_BOOTSTRAP_BROKER_INTERFACE,
            "generation_id": generation_id,
            "sequence": 1,
            "operation": "abort_started_owner",
            "arguments": {
                "owner_start_receipt_cid": "",
                "abort_reason_code": "attacker-request",
                forbidden_key: forbidden_value,
            },
        }
        attack["request_cid"] = lifecycle._cid(attack)
        raw = lifecycle._canonical_bytes(attack)
        response = casf_lifecycle._decode_canonical_object(
            broker.exchange_raw_for_test(raw), noun="attack response"
        )
        assert response["ok"] is False
        assert response["error_code"] == "broker_frame_invalid"
    assert forbidden_value not in json.dumps(response, sort_keys=True)


def test_persistent_casf_bootstrap_delayed_response_does_not_resend_request(
    tmp_path: Path,
) -> None:
    binding = _concrete_bootstrap_binding(
        tmp_path / "delayed-response-generation",
        generation_id="eaaef-delayed-response-001",
    )
    client_channel, server_channel = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM
    )
    death_reader, death_writer = os.pipe()
    os.close(death_reader)
    requests: list[bytes] = []

    def _delayed_service() -> None:
        request_raw = casf_lifecycle._recv_packet(server_channel)
        requests.append(request_raw)
        request = casf_lifecycle._validate_request(
            request_raw, generation_id=binding.generation_id
        )
        time.sleep(0.12)
        response = casf_lifecycle._build_response(
            generation_id=binding.generation_id,
            sequence=request["sequence"],
            operation=request["operation"],
            request_cid=request["request_cid"],
            ok=True,
            result={"owner_abort_acknowledged": True},
        )
        casf_lifecycle._send_packet(
            server_channel,
            casf_lifecycle._canonical_bytes(
                response, noun="delayed test response"
            ),
        )
        server_channel.settimeout(0.2)
        try:
            requests.append(casf_lifecycle._recv_packet(server_channel))
        except TimeoutError:
            pass

    service = threading.Thread(target=_delayed_service, daemon=True)
    service.start()
    broker = casf_lifecycle._BrokerClient(
        binding=binding,
        channel=client_channel,
        death_writer=death_writer,
        process=object(),
        absence={},
        timeout_seconds=0.1,
    )
    try:
        result = broker._exchange(
            "abort_started_owner",
            {
                "owner_start_receipt_cid": "",
                "abort_reason_code": "delayed-response-test",
            },
        )
        assert result == {"owner_abort_acknowledged": True}
        service.join(timeout=2)
        assert not service.is_alive()
        assert len(requests) == 1
    finally:
        broker.close_descriptors()
        server_channel.close()
