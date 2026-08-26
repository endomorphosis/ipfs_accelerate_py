from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    eaaef_offline_population as offline,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    eaaef_reconciliation_lifecycle as lifecycle,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_eaaef_reconciliation_owner import (
    EAAEF_CASF_BOOTSTRAP_BOUND_PRODUCTION_BLOCKERS,
    EAAEF_CASF_BOOTSTRAP_OWNER_GUARD_INTERFACE,
    EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE,
    EAAEF_CASF_BOOTSTRAP_REGISTRY_SCHEMA,
    EAAEF_CASF_OWNER_ABORT_RECEIPT_SCHEMA,
    EAAEF_CASF_OWNER_ABSENCE_ATTESTATION_SCHEMA,
    EAAEF_CASF_OWNER_COMMIT_RECEIPT_SCHEMA,
    EAAEF_CASF_OWNER_START_RECEIPT_SCHEMA,
    EAAEF_OWNER_PRODUCTION_BLOCKERS,
    EAAEFCASFBootstrapOwnerError,
    EAAEFCASFBootstrapRegistry,
    EAAEFTypedReconciliationOwnerUnavailable,
    bind_eaaef_casf_bootstrap_owner,
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
    assert sum(len(item["outputs"]) for item in translated["tasks"]) == 415
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
        "task_outputs": 415,
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
    assert sum(len(item.outputs) for item in page.tasks) == 415
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
    qualification = owner.reconciliation_qualification()
    assert qualification["bootstrap_materialization_before_owner_start"] is True
    assert qualification["plan_r2_remote_runtime_blockers"] == list(
        EAAEF_CASF_BOOTSTRAP_BOUND_PRODUCTION_BLOCKERS
    )
    assert qualification["provider_launch_allowed"] is False
    with pytest.raises(lifecycle.EAAEFReconciliationBlocked, match="qualification differs"):
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
