from __future__ import annotations

import inspect
import json
from collections import Counter
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
    identity = {"schema": lifecycle.EAAEF_FOREST_SCHEMA, "repositories": repositories}
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
    board = json.loads((repo_root / lifecycle.EAAEF_BOARD_PATH).read_text(encoding="utf-8"))
    return lifecycle.compile_fresh_eaaef_population(board, forest=_sealed_forest())


def test_translation_is_exact_database_task_source_population(repo_root: Path) -> None:
    population = _population(repo_root)
    translated = offline.translate_compiled_eaaef_population(
        population,
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
            current_forest=_sealed_forest(),
            owner_active=True,
        )
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="historical"):
        offline.translate_compiled_eaaef_population(
            population,
            current_forest=_sealed_forest(),
            owner_active=False,
            historical_task_statuses={"EAAEF-000": "done"},
        )
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="stale"):
        offline.translate_compiled_eaaef_population(
            population,
            current_forest=_sealed_forest(accelerator_commit="9" * 40),
            owner_active=False,
        )

    forged_task = dict(population.bootstrap_tasks[0])
    forged_task["status"] = "done"
    forged_population = replace(
        population,
        bootstrap_tasks=(forged_task, *population.bootstrap_tasks[1:]),
    )
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="status split"):
        offline.translate_compiled_eaaef_population(
            forged_population,
            current_forest=_sealed_forest(),
            owner_active=False,
        )


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
