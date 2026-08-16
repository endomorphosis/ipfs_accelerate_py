"""Authority-bound operational control-plane schema profile tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_ID,
    DATASETS_AUTHORITATIVE_OPERATIONAL_PROFILE_ID,
    DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES,
    DATASETS_SEMANTIC_TRUTH_RELATIONS,
    ControlPlaneSchemaInstallError,
    datasets_authoritative_operational_schema_sql,
    install_control_plane_schema,
    install_datasets_authoritative_operational_schema,
    verify_datasets_authoritative_operational_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentRepository,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for operational schema profile tests",
)


def _created_tables(sql_text: str) -> set[str]:
    return {
        match.group(1).lower()
        for match in re.finditer(
            r'(?im)^\s*CREATE\s+TABLE\s+"?([a-z][a-z0-9_]*)"?\s*\(',
            sql_text,
        )
    }


def test_profile_sql_is_derived_with_closed_operational_inventory() -> None:
    first = datasets_authoritative_operational_schema_sql()
    second = datasets_authoritative_operational_schema_sql()
    assert first == second

    created = _created_tables(first)
    assert created == set(DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES)
    assert created.isdisjoint(DATASETS_SEMANTIC_TRUTH_RELATIONS)
    assert "evidence_nodes" in created
    assert "proof_obligations" not in created
    assert "contract:code@1" not in first
    assert "contract:evidence@1" not in first
    assert "ControlPlaneDomain@1" not in first
    assert "semantic proof authority remains in ipfs_datasets_py" in first


def test_profile_install_replay_verify_and_intent_round_trip(tmp_path: Path) -> None:
    database = tmp_path / "operational.duckdb"
    applied = install_datasets_authoritative_operational_schema(
        database,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="profile-test",
    )
    assert applied.changed is True
    assert applied.receipts[0].migration_id == (DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_ID)

    verified = verify_datasets_authoritative_operational_schema(database)
    assert verified["valid"] is True
    assert verified["profile_id"] == DATASETS_AUTHORITATIVE_OPERATIONAL_PROFILE_ID
    assert verified["forbidden_relations"] == []
    assert verified["forbidden_contracts"] == []
    assert verified["operational_evidence"]["semantic_and_proof_authority"] == ("ipfs_datasets_py")
    assert set(verified["required_tables_ok"]) == set(DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES)

    replay = install_datasets_authoritative_operational_schema(
        database,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="profile-test",
    )
    assert replay.changed is False
    assert replay.schema_fingerprint == applied.schema_fingerprint

    # Exercise the existing intent/task authority against the constrained
    # profile.  install_schema=False prevents its legacy full-schema bootstrap.
    with IntentRepository(
        database,
        owner_id="owner:profile-test",
        install_schema=False,
    ) as repository:
        repository.upsert_objective(
            objective_id="objective:profile",
            objective_alias="PROFILE-OBJECTIVE",
            title="Profile objective",
            priority="P0",
        )
        repository.upsert_goal(
            goal_cid="goal:profile",
            goal_alias="PROFILE-GOAL",
            objective_id="objective:profile",
            title="Profile goal",
            ordinal=1,
        )
        repository.upsert_plan(
            plan_cid="plan:profile",
            goal_cid="goal:profile",
            plan_alias="PROFILE-PLAN",
            status="active",
        )
        repository.upsert_task(
            task_cid="task:profile",
            task_alias="PROFILE-TASK",
            goal_cid="goal:profile",
            plan_cid="plan:profile",
            objective_id="objective:profile",
            ordinal=1,
            status="ready",
            acceptance=[{"criterion": "receipt exists"}],
        )
        task = repository.get_task("task:profile")
        assert task is not None
        assert task["task_cid"] == "task:profile"
        assert task["status"] == "ready"

    with DatabaseTaskSource(database, install_schema=False) as task_source:
        task = task_source.get_task("PROFILE-TASK")
        assert task is not None
        assert task.task_cid == "task:profile"
        ready = task_source.ready_tasks(limit=10)
        assert [item.task_cid for item in ready.tasks] == ["task:profile"]

    with open_duckdb_connection(database) as connection:
        contract_domains = {
            str(row[0])
            for row in connection.execute("SELECT domain_name FROM schema_contracts").fetchall()
        }
        assert "code" not in contract_domains
        assert "evidence" not in contract_domains


def test_profile_refuses_a_full_control_plane_database(tmp_path: Path) -> None:
    database = tmp_path / "full.duckdb"
    install_control_plane_schema(
        database,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="full-schema-test",
    )
    with pytest.raises(ControlPlaneSchemaInstallError, match="forbidden_relations"):
        install_datasets_authoritative_operational_schema(database)
    with pytest.raises(ControlPlaneSchemaInstallError, match="forbidden_relations"):
        verify_datasets_authoritative_operational_schema(database)


def test_profile_verification_rejects_later_semantic_relation(tmp_path: Path) -> None:
    database = tmp_path / "tampered.duckdb"
    install_datasets_authoritative_operational_schema(database)
    with open_duckdb_connection(database) as connection:
        connection.execute("CREATE TABLE proof_obligations (obligation_id VARCHAR PRIMARY KEY)")
    with pytest.raises(ControlPlaneSchemaInstallError, match="proof_obligations"):
        verify_datasets_authoritative_operational_schema(database)
