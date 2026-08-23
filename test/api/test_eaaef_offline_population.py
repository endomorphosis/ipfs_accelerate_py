from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import (
    eaaef_offline_population as offline,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    eaaef_reconciliation_lifecycle as lifecycle,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    typed_eaaef_reconciliation_owner as owner_facade,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_eaaef_reconciliation_owner import (
    EAAEF_OWNER_PRODUCTION_BLOCKERS,
    EAAEFTypedReconciliationOwnerUnavailable,
    open_eaaef_typed_reconciliation_owner,
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


def _population(repo_root: Path) -> lifecycle.CompiledEAAEFPopulation:
    return lifecycle.compile_fresh_eaaef_population(_board(repo_root), forest=_sealed_forest())


def _board(repo_root: Path) -> dict[str, Any]:
    return json.loads((repo_root / lifecycle.EAAEF_BOARD_PATH).read_text(encoding="utf-8"))


def _plain(value: Any) -> Any:
    return json.loads(lifecycle._canonical_bytes(value))


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
    ) == translated


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
            owner_active=True,
        )
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="historical"):
        offline.translate_compiled_eaaef_population(
            population,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
            owner_active=False,
            historical_task_statuses={"EAAEF-000": "done"},
        )
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="stale"):
        offline.translate_compiled_eaaef_population(
            population,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(accelerator_commit="9" * 40),
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
            owner_active=False,
        )


def test_current_board_and_task_specs_must_remain_self_addressed(repo_root: Path) -> None:
    board = _board(repo_root)
    board["goals"][0]["title"] = "forged board title"
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="board CID"):
        lifecycle.compile_fresh_eaaef_population(board, forest=_sealed_forest())

    board = _board(repo_root)
    board["tasks"][0]["execution_owned_files"][0] = "forged/task-output.txt"
    projection = dict(board)
    projection.pop("board_cid")
    board["board_cid"] = lifecycle._eaaef_source_cid(projection)
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="task spec CID"):
        lifecycle.compile_fresh_eaaef_population(board, forest=_sealed_forest())


def test_resealed_forged_rows_still_differ_from_current_sealed_board(repo_root: Path) -> None:
    population = _population(repo_root)
    forged = _reseal_mutated_output_population(population)

    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="sealed board"):
        offline.translate_compiled_eaaef_population(
            forged,
            current_board=_board(repo_root),
            current_forest=_sealed_forest(),
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
        lifecycle.compile_fresh_eaaef_population(forged_board, forest=forest)

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
        )

    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="sealed Git board source",
    ):
        offline.translate_compiled_eaaef_population(
            forged_population,
            current_board=forged_board,
            current_forest=forest,
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
            owner_active=False,
        )
    assert sink.called is False


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
    facade_source = inspect.getsource(owner_facade)
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
