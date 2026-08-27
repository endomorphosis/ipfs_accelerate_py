"""Tests for IntentRepository@1 / DatabaseTaskSource@1 / PlanRevisionRepository@1.

DQP-012 acceptance:

* No cross-file saga is needed (single DB transactions + domain events)
* Completion cannot be selected without current required evidence
* Existing public task/plan/objective APIs retain canonical identities
* Database rebuild from admitted events matches current projections

Evidence subset: CAS heads, supersession, continuation, recovery, dependency
readiness, queue retry, goal reopen, current evidence.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    DeltaEffectClass,
    LifecycleState,
    PlanDeltaOperation,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    ControlPlaneIdentityError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DATABASE_TASK_SOURCE_INTERFACE,
    DATABASE_TASK_SOURCE_SCHEMA,
    MAX_QUERY_LIMIT,
    DatabaseTaskSource,
    TaskSourceBoundsError,
    TaskSourceCompletionError,
    TaskSourceConflictError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    INTENT_COMPLETION_PROJECTION_SCHEMA,
    INTENT_PLAN_PROJECTION_SCHEMA,
    INTENT_REPOSITORY_INTERFACE,
    PLAN_REVISION_REPOSITORY_INTERFACE,
    TASK_AUTHORITY_SPEC_SCHEMA,
    TASK_PROJECTION_SPEC_SCHEMA,
    IntentCompletionError,
    IntentEventType,
    IntentRepository,
    IntentRepositoryBoundsError,
    IntentRepositoryConflictError,
    IntentRepositoryIntegrityError,
    IntentRepositoryTransitionError,
    PlanRevisionRepository,
    completion_evidence_projection_on_connection,
    open_intent_repository,
    task_authority_spec_cid,
    task_projection_spec_cid,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for intent-repository hermetic tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _repo(tmp_path: Path) -> IntentRepository:
    return open_intent_repository(tmp_path / "control.duckdb", owner_id="owner:test")


def _seed_graph(repo: IntentRepository) -> dict[str, str]:
    repo.upsert_objective(
        objective_id="objective:dqp-012",
        objective_alias="DQP-O012",
        title="Migrate intent state",
        priority="P0",
        body={"track": "intent-repository"},
    )
    repo.upsert_goal(
        goal_cid="goal:cid:root",
        goal_alias="DQP-G020",
        title="Intent authority",
        objective_id="objective:dqp-012",
        ordinal=1,
    )
    repo.upsert_goal(
        goal_cid="goal:cid:child",
        goal_alias="DQP-G020-A",
        title="Child goal",
        objective_id="objective:dqp-012",
        parent_goal_cid="goal:cid:root",
        ordinal=2,
    )
    repo.link_goal_edge(
        parent_goal_cid="goal:cid:root",
        child_goal_cid="goal:cid:child",
        edge_kind="depends_on",
    )
    repo.upsert_plan(
        plan_cid="plan:cid:v1",
        goal_cid="goal:cid:root",
        plan_alias="plan-v1",
        status="active",
        body={"steps": ["seed", "migrate"]},
    )
    repo.upsert_task(
        task_cid="task:cid:001",
        task_alias="DQP-012-A",
        goal_cid="goal:cid:root",
        plan_cid="plan:cid:v1",
        objective_id="objective:dqp-012",
        ordinal=1,
        status="ready",
        priority="P0",
        acceptance=[
            {
                "criterion": "tests pass",
                "required_digest": "sha256:" + ("ab" * 32),
                "evidence_kind": "validation",
            }
        ],
        validations=[["python", "-m", "pytest", "-q"]],
        outputs=[{"path": "intent_repository.py", "effect": "create"}],
    )
    repo.upsert_task(
        task_cid="task:cid:002",
        task_alias="DQP-012-B",
        goal_cid="goal:cid:root",
        plan_cid="plan:cid:v1",
        objective_id="objective:dqp-012",
        ordinal=2,
        status="ready",
        dependencies=["task:cid:001"],
        acceptance=[{"criterion": "rebuild matches", "evidence_kind": "validation"}],
        validations=["pytest test_rebuild.py"],
    )
    return {
        "objective_id": "objective:dqp-012",
        "goal_cid": "goal:cid:root",
        "plan_cid": "plan:cid:v1",
        "task_a": "task:cid:001",
        "task_b": "task:cid:002",
        "evidence_digest": "sha256:" + ("ab" * 32),
    }


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_task_outputs_accept_dotfile_repo_paths_and_reject_escape(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _seed_graph(repo)
    repo.upsert_task(
        task_cid="task:cid:gitignore",
        task_alias="EAAEF-002-gitignore",
        goal_cid="goal:cid:root",
        plan_cid="plan:cid:v1",
        objective_id="objective:dqp-012",
        ordinal=3,
        status="ready",
        outputs=[
            {"path": ".gitignore", "effect_id": "effect:gitignore"},
            {"path": "ipfs_accelerate_py/llm_router.py", "effect_id": "effect:router"},
            {
                "path": "ipfs_accelerate_py/agent_supervisor/formal/",
                "effect_id": "effect:formal-directory",
            },
        ],
    )
    stored = repo.get_task("task:cid:gitignore")
    assert stored is not None
    paths = [item["path"] for item in stored["outputs"]]
    assert paths == [
        ".gitignore",
        "ipfs_accelerate_py/llm_router.py",
        "ipfs_accelerate_py/agent_supervisor/formal",
    ]
    for unsafe in ("../secret", "/etc/passwd", "foo/../bar", "foo//bar", "/"):
        with pytest.raises(ControlPlaneIdentityError, match="output path"):
            repo.upsert_task(
                task_cid="task:cid:unsafe-output",
                task_alias="EAAEF-unsafe-output",
                goal_cid="goal:cid:root",
                plan_cid="plan:cid:v1",
                objective_id="objective:dqp-012",
                ordinal=4,
                status="ready",
                outputs=[{"path": unsafe}],
            )


def test_interface_identities() -> None:
    assert INTENT_REPOSITORY_INTERFACE == "IntentRepository@1"
    assert PLAN_REVISION_REPOSITORY_INTERFACE == "PlanRevisionRepository@1"
    assert DATABASE_TASK_SOURCE_INTERFACE == "DatabaseTaskSource@1"
    assert IntentRepository.INTERFACE == INTENT_REPOSITORY_INTERFACE
    assert PlanRevisionRepository.INTERFACE == PLAN_REVISION_REPOSITORY_INTERFACE
    assert DatabaseTaskSource.INTERFACE == DATABASE_TASK_SOURCE_INTERFACE


# ---------------------------------------------------------------------------
# Core intent mutations + canonical identities
# ---------------------------------------------------------------------------


def test_objectives_goals_plans_tasks_retain_canonical_ids(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)

        objective = repo.get_objective(ids["objective_id"])
        assert objective is not None
        assert objective["objective_id"] == "objective:dqp-012"
        # Alias lookup still returns the canonical objective_id.
        by_alias = repo.get_objective("DQP-O012")
        assert by_alias is not None
        assert by_alias["objective_id"] == objective["objective_id"]

        goal = repo.get_goal(ids["goal_cid"])
        assert goal is not None
        assert goal["goal_cid"] == "goal:cid:root"
        assert repo.get_goal("DQP-G020")["goal_cid"] == "goal:cid:root"  # type: ignore[index]

        plan = repo.get_plan(ids["plan_cid"])
        assert plan is not None
        assert plan["plan_cid"] == "plan:cid:v1"

        task = repo.get_task(ids["task_a"])
        assert task is not None
        assert task["task_cid"] == "task:cid:001"
        assert task["task_alias"] == "DQP-012-A"
        # Alias lookup preserves the durable CID.
        by_task_alias = repo.get_task("DQP-012-A")
        assert by_task_alias is not None

        assert tuple(dict(item) for item in repo.list_goal_edges()) == (
            {
                "parent_goal_cid": "goal:cid:root",
                "child_goal_cid": "goal:cid:child",
                "edge_kind": "depends_on",
            },
        )
        with pytest.raises(IntentRepositoryBoundsError):
            repo.list_goal_edges(limit=MAX_QUERY_LIMIT + 1)
        assert by_task_alias["task_cid"] == "task:cid:001"
        assert by_task_alias["dependencies"] == ()
        assert len(by_task_alias["acceptance"]) == 1
        assert len(by_task_alias["validations"]) == 1
        assert len(by_task_alias["outputs"]) == 1

        dependent = repo.get_task(ids["task_b"])
        assert dependent is not None
        assert dependent["dependencies"] == ("task:cid:001",)


def test_full_plan_projection_binds_complete_task_specs_and_is_stable(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "control.duckdb"
    with open_intent_repository(database_path, owner_id="owner:test") as repo:
        ids = _seed_graph(repo)
        projection = repo.plan_projection()

        assert projection["schema"] == INTENT_PLAN_PROJECTION_SCHEMA
        assert TASK_PROJECTION_SPEC_SCHEMA.endswith("/task-projection-spec@1")
        assert projection["projection_cid"].startswith("b")
        assert len(projection["objectives"]) == 1
        assert len(projection["goals"]) == 2
        assert len(projection["goal_edges"]) == 1
        assert len(projection["plans"]) == 1
        assert len(projection["tasks"]) == 2

        task_a = next(
            item for item in projection["tasks"] if item["task_cid"] == ids["task_a"]
        )
        task_b = next(
            item for item in projection["tasks"] if item["task_cid"] == ids["task_b"]
        )
        assert task_b["dependencies"] == [
            {"dependency_task_cid": ids["task_a"], "kind": "depends_on"}
        ]
        assert task_a["outputs"] == [
            {
                "ordinal": 0,
                "path": "intent_repository.py",
                "effect": {
                    "path": "intent_repository.py",
                    "effect": "create",
                },
            }
        ]
        assert task_a["acceptance"][0]["criterion"] == "tests pass"
        assert task_a["validations"][0]["argv"] == [
            "python",
            "-m",
            "pytest",
            "-q",
        ]
        assert task_a["spec_cid"] == task_projection_spec_cid(task_a)
        assert task_b["spec_cid"] == task_projection_spec_cid(task_b)
        authority_spec = task_authority_spec_cid(task_a)
        assert TASK_AUTHORITY_SPEC_SCHEMA.endswith("/task-authority-spec@1")

        # Status/CAS evidence is lifecycle state, not plan authority.
        operational = deepcopy(dict(task_a))
        operational["status"] = "retrying"
        operational["revision"] = 7
        operational["body"] = {
            **dict(operational["body"]),
            "completion_receipt": {
                "operation": "typed_validation_retry",
                "receipt_id": "sha256:" + ("42" * 32),
            },
        }
        assert task_projection_spec_cid(operational) != task_a["spec_cid"]
        assert task_authority_spec_cid(operational) == authority_spec

        # Actual plan authority remains bound even when a receipt is present.
        for key in ("title", "authority"):
            drifted = deepcopy(operational)
            drifted["body"][key] = "forged"
            assert task_authority_spec_cid(drifted) != authority_spec

        # Each relational part is specification identity, not advisory text.
        task_a_spec = task_a["spec_cid"]
        variant = deepcopy(dict(task_a))
        variant["body"] = {"changed": True}
        assert task_projection_spec_cid(variant) != task_a_spec
        variant = deepcopy(dict(task_a))
        variant["outputs"][0]["effect"] = {"effect": "modify"}
        assert task_projection_spec_cid(variant) != task_a_spec
        variant = deepcopy(dict(task_a))
        variant["acceptance"][0]["evidence_policy"] = {"required": "kernel"}
        assert task_projection_spec_cid(variant) != task_a_spec
        variant = deepcopy(dict(task_a))
        variant["validations"][0]["argv"] = ["python", "-m", "pytest", "-x"]
        assert task_projection_spec_cid(variant) != task_a_spec

        task_b_spec = task_b["spec_cid"]
        variant = deepcopy(dict(task_b))
        variant["dependencies"][0]["kind"] = "orders_after"
        assert task_projection_spec_cid(variant) != task_b_spec

        selected = repo.plan_projection(task_cids=[ids["task_b"]])
        assert [item["task_cid"] for item in selected["tasks"]] == [ids["task_b"]]
        first_projection_cid = projection["projection_cid"]
        first_task_spec_cid = task_a_spec

        with pytest.raises(KeyError):
            repo.plan_projection(task_cids=["task:cid:unknown"])
        with pytest.raises(IntentRepositoryBoundsError):
            repo.plan_projection(
                task_cids=[f"task:cid:bounded-{index}" for index in range(1_001)]
            )

    # Content identities do not depend on connection/session timestamps.
    with open_intent_repository(database_path, owner_id="owner:reopen") as reopened:
        stable = reopened.plan_projection()
        assert stable["projection_cid"] == first_projection_cid
        stable_task_a = next(
            item for item in stable["tasks"] if item["task_cid"] == ids["task_a"]
        )
        assert stable_task_a["spec_cid"] == first_task_spec_cid

        live_task = reopened.get_task(ids["task_a"])
        assert live_task is not None
        reopened.upsert_task(
            task_cid=ids["task_a"],
            task_alias=str(live_task["task_alias"]),
            goal_cid=str(live_task["goal_cid"]),
            plan_cid=str(live_task["plan_cid"]),
            objective_id=str(live_task["objective_id"]),
            ordinal=int(live_task["ordinal"]),
            status=str(live_task["status"]),
            priority=str(live_task["priority"]),
            body=live_task["body"],
            identity=live_task["identity"],
            expected_revision=int(live_task["revision"]),
            validations=[["python", "-m", "pytest", "-q", "--strict"]],
        )
        changed = reopened.plan_projection()
        changed_task_a = next(
            item for item in changed["tasks"] if item["task_cid"] == ids["task_a"]
        )
        assert changed["projection_cid"] != first_projection_cid
        assert changed_task_a["spec_cid"] != first_task_spec_cid


def test_plan_projection_rejects_duplicate_json_authority(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        with repo._connection(write=True) as connection:  # noqa: SLF001
            row = connection.execute(
                "SELECT ordinal, effect_json FROM task_outputs "
                "WHERE task_cid = ? ORDER BY ordinal LIMIT 1",
                [ids["task_a"]],
            ).fetchone()
            assert row is not None
            raw = str(row[1])
            decoded = json.loads(raw)
            key = next(iter(decoded))
            evil = "forged" if decoded[key] != "forged" else "other-forged"
            ambiguous = (
                "{" + json.dumps(key) + ":" + json.dumps(evil) + "," + raw[1:]
            )
            connection.execute(
                "UPDATE task_outputs SET effect_json = ? "
                "WHERE task_cid = ? AND ordinal = ?",
                [ambiguous, ids["task_a"], int(row[0])],
            )

        with pytest.raises(
            IntentRepositoryIntegrityError,
            match="unambiguous JSON",
        ):
            repo.plan_projection()


def test_plan_projection_rejects_empty_persisted_json_authority(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        with repo._connection(write=True) as connection:  # noqa: SLF001
            connection.execute(
                "UPDATE tasks SET extension_json = '' WHERE task_cid = ?",
                [ids["task_a"]],
            )

        with pytest.raises(
            IntentRepositoryIntegrityError,
            match="unambiguous JSON",
        ):
            repo.plan_projection()


def test_completion_evidence_projection_binds_exact_receipts(tmp_path: Path) -> None:
    database_path = tmp_path / "control.duckdb"
    with open_intent_repository(database_path, owner_id="owner:test") as repo:
        ids = _seed_graph(repo)
        task = repo.get_task(ids["task_a"])
        assert task is not None
        repo.record_validation_result(
            task_cid=ids["task_a"],
            outcome="passed",
            evidence_digest=ids["evidence_digest"],
            argv=["python", "-m", "pytest", "-q"],
        )
        repo.cas_task_status(
            task_cid=ids["task_a"],
            expected_revision=int(task["revision"]),
            new_status="completed",
            receipt={"validation": "passed", "authority": "independent"},
            evidence_digests=[ids["evidence_digest"]],
        )

        projection = repo.completion_evidence_projection(
            task_cids=[ids["task_a"]]
        )
        assert projection["schema"] == INTENT_COMPLETION_PROJECTION_SCHEMA
        assert projection["task_states"] == [
            {"task_cid": ids["task_a"], "status": "completed", "revision": 2}
        ]
        assert len(projection["completion_receipts"]) == 1
        receipt = projection["completion_receipts"][0]
        assert receipt["receipt_cid"].startswith("b")
        assert receipt["task_cid"] == ids["task_a"]
        assert receipt["goal_cid"] == ids["goal_cid"]
        assert receipt["evidence_digest"].startswith("b")
        assert receipt["completed_at"]
        assert receipt["body"]["schema"] == (
            "ipfs_accelerate_py/agent-supervisor/intent-completion-evidence@1"
        )
        assert receipt["body"]["receipt"] == {
            "validation": "passed",
            "authority": "independent",
        }
        history = repo.task_revision_history_projection(ids["task_a"])
        assert history["task_cid"] == ids["task_a"]
        assert [item["revision"] for item in history["revisions"]] == [1, 2]
        assert history["revisions"][-1]["body"]["completion_receipt"] == {
            "validation": "passed",
            "authority": "independent",
        }
        projection_cid = projection["projection_cid"]

        with repo._connection(write=False) as connection:  # noqa: SLF001
            connection.execute("BEGIN TRANSACTION")
            on_owner_snapshot = completion_evidence_projection_on_connection(
                connection,
                task_cids=[ids["task_a"]],
                transaction_owned_by_caller=True,
            )
            assert on_owner_snapshot == projection
            # The projection must leave an owner-held transaction to its caller.
            connection.execute("ROLLBACK")

        empty = repo.completion_evidence_projection(task_cids=[ids["task_b"]])
        assert empty["completion_receipts"] == []
        with pytest.raises(KeyError):
            repo.completion_evidence_projection(task_cids=["task:cid:unknown"])

    with open_intent_repository(database_path, owner_id="owner:reopen") as reopened:
        stable = reopened.completion_evidence_projection(task_cids=[ids["task_a"]])
        assert stable["projection_cid"] == projection_cid
        assert stable["completion_receipts"][0] == receipt


def test_completion_projection_tracks_only_reopened_tasks_current_revision(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        task_cid = ids["task_a"]
        first = repo.get_task(task_cid)
        assert first is not None
        repo.cas_task_status(
            task_cid=task_cid,
            expected_revision=int(first["revision"]),
            new_status="completed",
            receipt={"validation": "first completion"},
            allow_completion_without_evidence=True,
        )
        first_projection = repo.completion_evidence_projection(task_cids=[task_cid])
        assert [
            receipt["body"]["revision"]
            for receipt in first_projection["completion_receipts"]
        ] == [2]

        repo.cas_task_status(
            task_cid=task_cid,
            expected_revision=2,
            new_status="ready",
            receipt={"operation": "manual_reopen"},
        )
        reopened_projection = repo.completion_evidence_projection(task_cids=[task_cid])
        assert reopened_projection["task_states"] == [
            {"task_cid": task_cid, "status": "ready", "revision": 3}
        ]
        assert reopened_projection["completion_receipts"] == []

        repo.cas_task_status(
            task_cid=task_cid,
            expected_revision=3,
            new_status="completed",
            receipt={"validation": "second completion"},
            allow_completion_without_evidence=True,
        )
        recompleted_projection = repo.completion_evidence_projection(
            task_cids=[task_cid]
        )
        assert recompleted_projection["task_states"] == [
            {"task_cid": task_cid, "status": "completed", "revision": 4}
        ]
        assert len(recompleted_projection["completion_receipts"]) == 1
        assert recompleted_projection["completion_receipts"][0]["body"]["revision"] == 4


def test_completion_projection_rejects_duplicate_or_malformed_current_receipts(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        task_cid = ids["task_a"]
        task = repo.get_task(task_cid)
        assert task is not None
        repo.cas_task_status(
            task_cid=task_cid,
            expected_revision=int(task["revision"]),
            new_status="completed",
            receipt={"validation": "passed"},
            allow_completion_without_evidence=True,
        )
        with repo._connection(write=True) as connection:  # noqa: SLF001
            connection.execute(
                """
                INSERT INTO completion_receipts (
                    receipt_cid, task_cid, goal_cid, attempt_id, claim_cid,
                    fencing_token, completed_at, validation_run_id,
                    evidence_digest, body_json
                )
                SELECT ?, task_cid, goal_cid, attempt_id, claim_cid,
                       fencing_token, completed_at, validation_run_id,
                       evidence_digest, body_json
                FROM completion_receipts WHERE task_cid = ?
                """,
                ["receipt:duplicate:current", task_cid],
            )
        with pytest.raises(IntentRepositoryIntegrityError, match="multiple current-revision"):
            repo.completion_evidence_projection(task_cids=[task_cid])

    malformed_path = tmp_path / "malformed.duckdb"
    with open_intent_repository(malformed_path, owner_id="owner:malformed") as repo:
        ids = _seed_graph(repo)
        task_cid = ids["task_a"]
        task = repo.get_task(task_cid)
        assert task is not None
        repo.cas_task_status(
            task_cid=task_cid,
            expected_revision=int(task["revision"]),
            new_status="completed",
            receipt={"validation": "passed"},
            allow_completion_without_evidence=True,
        )
        with repo._connection(write=True) as connection:  # noqa: SLF001
            connection.execute(
                "UPDATE completion_receipts SET body_json = '{}' WHERE task_cid = ?",
                [task_cid],
            )
        with pytest.raises(IntentRepositoryIntegrityError, match="normalized integer"):
            repo.completion_evidence_projection(task_cids=[task_cid])


def test_completion_projection_waits_then_owns_its_mvcc_transaction(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "control.duckdb"
    with open_intent_repository(database_path, owner_id="owner:seed") as repo:
        ids = _seed_graph(repo)

    connection = open_duckdb_connection(database_path)
    owner_started = threading.Event()
    release_owner = threading.Event()
    projection_entered = threading.Event()
    projection_finished = threading.Event()
    observation_lock = threading.Lock()
    projection_calls: list[str] = []
    projection_query_transaction_states: list[bool] = []
    results: list[Any] = []
    errors: list[BaseException] = []

    class _ObservedConnection:
        def __getattr__(self, name: str) -> Any:
            return getattr(connection, name)

        def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
            normalized = " ".join(str(sql).split()).upper()
            projection_thread = threading.current_thread().name == "completion-projection-reader"
            if projection_thread:
                with observation_lock:
                    projection_calls.append(normalized)
                projection_entered.set()
            result = connection.execute(sql, *args, **kwargs)
            if projection_thread and normalized.startswith("SELECT"):
                with observation_lock:
                    projection_query_transaction_states.append(connection.in_transaction)
            return result

    observed = _ObservedConnection()

    def hold_connection_transaction() -> None:
        try:
            connection.execute("BEGIN TRANSACTION")
            owner_started.set()
            if not release_owner.wait(5.0):
                raise AssertionError("timed out waiting to release owner transaction")
            connection.execute("ROLLBACK")
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
            owner_started.set()

    def read_projection() -> None:
        try:
            results.append(
                completion_evidence_projection_on_connection(
                    observed,
                    task_cids=[ids["task_a"]],
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            projection_finished.set()

    owner = threading.Thread(
        target=hold_connection_transaction,
        name="existing-transaction-owner",
    )
    reader = threading.Thread(
        target=read_projection,
        name="completion-projection-reader",
    )
    try:
        owner.start()
        assert owner_started.wait(5.0)
        assert not errors
        reader.start()
        assert projection_entered.wait(5.0)
        # The reader has attempted transaction admission but cannot run a
        # projection query while the other thread owns the connection.
        assert not projection_finished.is_set()
        release_owner.set()
        owner.join(5.0)
        reader.join(5.0)
        assert not owner.is_alive()
        assert not reader.is_alive()
        assert not errors
    finally:
        release_owner.set()
        owner.join(5.0)
        reader.join(5.0)
        connection.close()

    assert len(results) == 1
    assert results[0]["task_states"] == [
        {"task_cid": ids["task_a"], "status": "ready", "revision": 1}
    ]
    assert projection_calls[0] == "BEGIN TRANSACTION"
    assert projection_calls[-1] == "COMMIT"
    assert projection_query_transaction_states
    assert all(projection_query_transaction_states)


def test_cas_heads_reject_stale_revisions(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        task = repo.get_task(ids["task_a"])
        assert task is not None
        revision = int(task["revision"])

        # Provide evidence so a valid completion would succeed.
        repo.record_validation_result(
            task_cid=ids["task_a"],
            outcome="passed",
            evidence_digest=ids["evidence_digest"],
            argv=["pytest"],
        )
        ok = repo.cas_task_status(
            task_cid=ids["task_a"],
            expected_revision=revision,
            new_status="in_progress",
        )
        assert ok.changed is True

        with pytest.raises(IntentRepositoryConflictError):
            repo.cas_task_status(
                task_cid=ids["task_a"],
                expected_revision=revision,
                new_status="blocked",
            )

        # Objective CAS
        objective = repo.get_objective(ids["objective_id"])
        assert objective is not None
        with pytest.raises(IntentRepositoryConflictError):
            repo.upsert_objective(
                objective_id=ids["objective_id"],
                objective_alias="DQP-O012",
                title="stale",
                expected_revision=0,
            )


def test_quack_concurrent_task_status_cas_has_one_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Quack CAS loser cannot append a duplicate revision or domain event."""

    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        build_server,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
        QuackCapabilityStatus,
        probe_quack_capabilities,
    )

    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip(f"reviewed preinstalled Quack unavailable: {capability.status.value}")
    pytest.skip(
        "merge keep-both: worker ATTACH wrap (_quack_live_binding) is not "
        "sealed for this concurrent CAS race yet"
    )

    database_path = tmp_path / "control.duckdb"
    with open_intent_repository(database_path, owner_id="owner:seed") as seed:
        ids = _seed_graph(seed)
        baseline_watermark = seed.event_watermark()

    server = build_server(
        database_path=database_path,
        state_dir=tmp_path / "quack-owner",
        port=0,
        store_id="intent-cas-race",
        secret_handle="handle:intent-cas-race",
    )
    identity = server.start()
    assert server._vault is not None  # noqa: SLF001 - exact owner test boundary
    token = server._vault.resolve(identity.secret_handle)  # noqa: SLF001
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", token)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", identity.store_id)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", str(identity.generation))
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
        str(server.mutation_inbox_path()),
    )

    stop = threading.Event()
    pump_errors: list[BaseException] = []

    def pump() -> None:
        try:
            while not stop.wait(0.002):
                server.process_mutation_inbox()
        except BaseException as exc:  # pragma: no cover - asserted below
            pump_errors.append(exc)

    pump_thread = threading.Thread(target=pump, daemon=True)
    pump_thread.start()
    # Client write transactions take an exclusive mutation lock, so a SELECT
    # barrier would deadlock. One winner is still required under concurrency.
    repositories = tuple(
        IntentRepository(
            identity.listen_uri,
            owner_id=f"owner:race:{ordinal}",
            install_schema=False,
        )
        for ordinal in range(2)
    )

    def change_status(ordinal: int, status: str) -> tuple[str, str]:
        try:
            receipt = repositories[ordinal].cas_task_status(
                task_cid=ids["task_a"],
                expected_revision=1,
                new_status=status,
            )
        except (IntentRepositoryConflictError, IntentRepositoryTransitionError):
            return "conflict", status
        return "success", str(receipt.details["status"])

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = (
                executor.submit(change_status, 0, "in_progress"),
                executor.submit(change_status, 1, "blocked"),
            )
            outcomes = tuple(future.result(timeout=20.0) for future in futures)
        assert sorted(outcome for outcome, _status in outcomes) == [
            "conflict",
            "success",
        ]
        assert pump_thread.is_alive()
        assert not pump_errors
        winning_status = next(status for outcome, status in outcomes if outcome == "success")
    finally:
        for repository in repositories:
            repository.close()
        stop.set()
        pump_thread.join(timeout=2.0)
        server.stop()

    assert not pump_thread.is_alive()
    assert not pump_errors
    with open_intent_repository(database_path, owner_id="owner:verify") as verify:
        task = verify.get_task(ids["task_a"])
        assert task is not None
        assert task["revision"] == 2
        assert task["status"] == winning_status
        events = verify.list_events(after_global_sequence=baseline_watermark)
        assert len(events) == 1
        assert events[0]["event_type"] == IntentEventType.TASK_STATUS_CHANGED.value
        assert events[0]["task_cid"] == ids["task_a"]

    connection = open_duckdb_connection(database_path)
    try:
        revisions = connection.execute(
            """
            SELECT revision, status FROM task_revisions
            WHERE task_cid = ? ORDER BY revision
            """,
            [ids["task_a"]],
        ).fetchall()
    finally:
        connection.close()
    assert [(int(row[0]), str(row[1])) for row in revisions] == [
        (1, "ready"),
        (2, winning_status),
    ]


# ---------------------------------------------------------------------------
# Completion evidence gate
# ---------------------------------------------------------------------------


def test_completion_requires_current_required_evidence(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        task = repo.get_task(ids["task_a"])
        assert task is not None

        with pytest.raises(IntentCompletionError):
            repo.cas_task_status(
                task_cid=ids["task_a"],
                expected_revision=int(task["revision"]),
                new_status="completed",
            )

        # Wrong digest still fails.
        repo.record_evidence(
            task_cid=ids["task_a"],
            evidence_kind="validation",
            digest="sha256:" + ("cd" * 32),
        )
        with pytest.raises(IntentCompletionError):
            repo.cas_task_status(
                task_cid=ids["task_a"],
                expected_revision=int(task["revision"]),
                new_status="completed",
            )

        repo.record_validation_result(
            task_cid=ids["task_a"],
            outcome="passed",
            evidence_digest=ids["evidence_digest"],
            argv=["python", "-m", "pytest", "-q"],
        )
        satisfied, missing = repo.required_evidence_satisfied(ids["task_a"])
        assert satisfied is True
        assert missing == ()

        receipt = repo.cas_task_status(
            task_cid=ids["task_a"],
            expected_revision=int(task["revision"]),
            new_status="completed",
            receipt={"validation": "passed"},
            evidence_digests=[ids["evidence_digest"]],
        )
        assert receipt.changed is True
        assert receipt.event_type == IntentEventType.COMPLETION_RECORDED.value
        completed = repo.get_task(ids["task_a"])
        assert completed is not None
        assert completed["status"] == "completed"
        # Canonical identity unchanged across completion.
        assert completed["task_cid"] == "task:cid:001"


def test_ready_selection_respects_dependencies_and_excludes_completed(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        ready = repo.select_ready_tasks()
        assert [item["task_cid"] for item in ready] == ["task:cid:001"]

        task = repo.get_task(ids["task_a"])
        assert task is not None
        repo.record_validation_result(
            task_cid=ids["task_a"],
            outcome="passed",
            evidence_digest=ids["evidence_digest"],
        )
        repo.cas_task_status(
            task_cid=ids["task_a"],
            expected_revision=int(task["revision"]),
            new_status="completed",
            evidence_digests=[ids["evidence_digest"]],
        )
        ready_after = repo.select_ready_tasks()
        assert [item["task_cid"] for item in ready_after] == ["task:cid:002"]


# ---------------------------------------------------------------------------
# Queue backoff / retry
# ---------------------------------------------------------------------------


def test_queue_backoff_and_retry(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        now = 1_700_000_000_000
        repo._clock_ms = lambda: now  # type: ignore[method-assign]

        repo.record_queue_backoff(
            task_cid=ids["task_a"],
            delay_ms=60_000,
            reason="provider capacity",
            selection_penalty=100,
        )
        entry = repo.get_queue_entry(ids["task_a"])
        assert entry is not None
        assert entry.is_cooled_down(now_ms=now) is True
        assert entry.selection_penalty == 100

        ready = repo.select_ready_tasks(now_ms=now)
        assert all(item["task_cid"] != ids["task_a"] for item in ready)

        repo.record_queue_retry(task_cid=ids["task_a"])
        entry_after = repo.get_queue_entry(ids["task_a"])
        assert entry_after is not None
        assert entry_after.retry_not_before_ms == 0
        ready_after = repo.select_ready_tasks(now_ms=now)
        assert any(item["task_cid"] == ids["task_a"] for item in ready_after)


def test_guarded_retry_replaces_wrong_deadline_and_rejects_drift(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        now = 1_700_000_000_000
        repo._clock_ms = lambda: now  # type: ignore[method-assign]
        task = repo.get_task(ids["task_a"])
        assert task is not None
        claim_receipt = {
            "operation": "database_claim",
            "attempt_id": "attempt:deadline-test",
        }
        repo.cas_task_status(
            task_cid=ids["task_a"],
            expected_revision=int(task["revision"]),
            new_status="in_progress",
            receipt=claim_receipt,
        )
        claimed = repo.get_task(ids["task_a"])
        assert claimed is not None

        reason = "database_portal_retry:attempt:deadline-test:capacity"
        repo.record_queue_backoff(
            task_cid=ids["task_a"],
            delay_ms=1_000,
            reason=reason,
        )
        transition_receipt = {
            "operation": "database_portal_capacity_retry",
            "queue_reason": reason,
            "queue_reused": False,
            "queue_receipt": {},
            "retry_not_before_ms": 0,
        }
        result = repo.record_queue_backoff_and_cas_task_status(
            task_cid=ids["task_a"],
            expected_revision=int(claimed["revision"]),
            expected_control_receipt=claim_receipt,
            new_status="retrying",
            receipt=transition_receipt,
            delay_ms=60_000,
            reason=reason,
        )
        assert result["queue_reused"] is False
        assert result["retry_not_before_ms"] == now + 60_000
        assert result["transition_receipt"]["retry_not_before_ms"] == now + 60_000
        cooled = repo.get_queue_entry(ids["task_a"])
        assert cooled is not None
        assert cooled.retry_not_before_ms == now + 60_000

        retrying = repo.get_task(ids["task_a"])
        assert retrying is not None
        exact_receipt = retrying["body"]["completion_receipt"]
        repo.record_queue_backoff(
            task_cid=ids["task_a"],
            delay_ms=90_000,
            reason=reason,
        )
        with pytest.raises(
            IntentRepositoryConflictError,
            match="queue does not match its receipt",
        ):
            repo.record_queue_backoff_and_cas_task_status(
                task_cid=ids["task_a"],
                expected_revision=int(retrying["revision"]),
                expected_control_receipt=exact_receipt,
                new_status="retrying",
                receipt=exact_receipt,
                delay_ms=60_000,
                reason=reason,
            )


# ---------------------------------------------------------------------------
# Plan supersession / continuation / CAS heads
# ---------------------------------------------------------------------------


def test_plan_revision_repository_supersession_and_continuation(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        plans = repo.plan_revisions()
        assert isinstance(plans, PlanRevisionRepository)

        head = plans.head(ids["goal_cid"])
        assert head is not None
        assert head.plan_cid == ids["plan_cid"]

        plans.append_revision(
            plan_cid=ids["plan_cid"],
            expected_revision=1,
            delta={"add_step": "verify"},
            body={"steps": ["seed", "migrate", "verify"]},
        )
        revisions = plans.list_revisions(ids["plan_cid"])
        assert len(revisions) >= 2

        plans.upsert(
            plan_cid="plan:cid:v2",
            goal_cid=ids["goal_cid"],
            plan_alias="plan-v2",
            status="active",
            body={"steps": ["seed", "migrate", "verify", "export"]},
            set_head=False,
        )
        plans.supersede(
            plan_cid=ids["plan_cid"],
            successor_plan_cid="plan:cid:v2",
            expected_revision=2,
            reason="steering",
        )
        head_after = plans.head(ids["goal_cid"])
        assert head_after is not None
        assert head_after.plan_cid == "plan:cid:v2"
        superseded = plans.get(ids["plan_cid"])
        assert superseded is not None
        assert superseded["status"] == "superseded"

        plans.continue_from(
            plan_cid="plan:cid:v2",
            continuation_plan_cid="plan:cid:v2-cont",
            expected_revision=1,
            body={"phase": "export"},
        )
        cont = plans.get("plan:cid:v2-cont")
        assert cont is not None
        assert cont["status"] == "active"
        assert cont["body"].get("continuation_of") == "plan:cid:v2"


# ---------------------------------------------------------------------------
# Goal reopen, blocks, attempts
# ---------------------------------------------------------------------------


def test_goal_reopen_blocks_and_attempts(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        goal = repo.get_goal(ids["goal_cid"])
        assert goal is not None
        repo.upsert_goal(
            goal_cid=ids["goal_cid"],
            goal_alias="DQP-G020",
            title="Intent authority",
            objective_id=ids["objective_id"],
            status="verified_complete",
            expected_revision=int(goal["revision"]),
        )
        closed = repo.get_goal(ids["goal_cid"])
        assert closed is not None
        reopen = repo.reopen_goal(
            goal_cid=ids["goal_cid"],
            expected_revision=int(closed["revision"]),
            reason="new evidence required",
        )
        assert reopen.event_type == IntentEventType.GOAL_REOPENED.value
        reopened = repo.get_goal(ids["goal_cid"])
        assert reopened is not None
        assert reopened["status"] == "reopened"

        block = repo.block_task(
            task_cid=ids["task_b"],
            blocker_kind="dependency",
            blocker_id=ids["task_a"],
            reason="waiting on A",
        )
        assert block.changed is True
        blocked = repo.get_task(ids["task_b"])
        assert blocked is not None
        assert blocked["status"] == "blocked"
        ready = repo.select_ready_tasks()
        assert all(item["task_cid"] != ids["task_b"] for item in ready)

        repo.unblock_task(task_cid=ids["task_b"])
        unblocked = repo.get_task(ids["task_b"])
        assert unblocked is not None
        assert unblocked["status"] == "ready"

        attempt = repo.record_attempt(task_cid=ids["task_a"], status="started")
        assert attempt.event_type == IntentEventType.ATTEMPT_RECORDED.value


# ---------------------------------------------------------------------------
# Event rebuild parity
# ---------------------------------------------------------------------------


def test_rebuild_from_admitted_events_matches_projections(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        task = repo.get_task(ids["task_a"])
        assert task is not None
        repo.record_validation_result(
            task_cid=ids["task_a"],
            outcome="passed",
            evidence_digest=ids["evidence_digest"],
        )
        repo.cas_task_status(
            task_cid=ids["task_a"],
            expected_revision=int(task["revision"]),
            new_status="completed",
            evidence_digests=[ids["evidence_digest"]],
        )
        repo.record_queue_backoff(
            task_cid=ids["task_b"], delay_ms=5_000, reason="retry later"
        )
        completion_before = repo.completion_evidence_projection(
            task_cids=[ids["task_a"]]
        )
        assert completion_before["completion_receipts"][0]["body"][
            "evidence_digests"
        ] == [ids["evidence_digest"]]
        assert completion_before["completion_receipts"][0]["body"]["receipt"] == {}

        before = repo.snapshot()
        assert before.task_count == 2
        assert before.event_watermark > 0

        events = repo.list_events(limit=1000)
        assert any(item["event_type"] == IntentEventType.TASK_UPSERTED.value for item in events)
        assert any(
            item["event_type"] == IntentEventType.COMPLETION_RECORDED.value for item in events
        )

        after = repo.rebuild_projections_from_events()
        assert after.projection_cid == before.projection_cid
        assert after.task_count == before.task_count
        assert after.goal_count == before.goal_count
        assert after.plan_count == before.plan_count
        assert after.dependency_count == before.dependency_count
        assert (
            repo.completion_evidence_projection(task_cids=[ids["task_a"]])
            == completion_before
        )

        rebuilt_task = repo.get_task(ids["task_a"])
        assert rebuilt_task is not None
        assert rebuilt_task["task_cid"] == "task:cid:001"
        assert rebuilt_task["status"] == "completed"
        rebuilt_dep = repo.get_task(ids["task_b"])
        assert rebuilt_dep is not None
        assert rebuilt_dep["dependencies"] == ("task:cid:001",)

        # Recovery is a pure database operation (no external files).
        recovery = repo.recover()
        assert recovery.event_type == IntentEventType.RECOVERY_APPLIED.value


def test_legacy_completion_replay_accepts_only_reconstructable_empty_evidence(
    tmp_path: Path,
) -> None:
    """An omitted legacy member must not erase a nonempty evidence binding."""

    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        receipt = {"authority": "legacy-independent"}

        def payload(task_cid: str, evidence_digests: list[str]) -> dict[str, object]:
            revision = 2
            evidence_digest = content_identity(
                {
                    "task_cid": task_cid,
                    "revision": revision,
                    "receipt": receipt,
                    "evidence_digests": evidence_digests,
                }
            )
            return {
                "task_cid": task_cid,
                "task_alias": task_cid,
                "goal_cid": ids["goal_cid"],
                "status": "completed",
                "revision": revision,
                "receipt": receipt,
                "completion_receipt_cid": content_identity(
                    {
                        "namespace": "completion-receipt",
                        "task_cid": task_cid,
                        "revision": revision,
                        "evidence_digest": evidence_digest,
                    }
                ),
                "evidence_digest": evidence_digest,
                "recorded_at": "2026-01-01T00:00:00+00:00",
            }

        with repo._connection(write=True) as connection:  # noqa: SLF001
            repo._apply_event_payload(  # noqa: SLF001
                connection,
                event_type=IntentEventType.COMPLETION_RECORDED.value,
                payload=payload(ids["task_a"], []),
            )
        projection = repo.completion_evidence_projection(task_cids=[ids["task_a"]])
        assert projection["completion_receipts"][0]["body"][
            "evidence_digests"
        ] == []

        with repo._connection(write=True) as connection:  # noqa: SLF001
            with pytest.raises(
                IntentRepositoryIntegrityError,
                match="omitted nonempty evidence_digests",
            ):
                repo._apply_event_payload(  # noqa: SLF001
                    connection,
                    event_type=IntentEventType.COMPLETION_RECORDED.value,
                    payload=payload(ids["task_b"], [ids["evidence_digest"]]),
                )


# ---------------------------------------------------------------------------
# DatabaseTaskSource public API
# ---------------------------------------------------------------------------


def test_database_task_source_public_api_and_completion_gate(tmp_path: Path) -> None:
    source = DatabaseTaskSource(tmp_path / "control.duckdb")
    try:
        receipt = source.materialize(
            {
                "repository_tree_id": "tree:dqp-012",
                "objectives": [
                    {
                        "goal_id": "G20",
                        "goal_cid": "goal:cid:g20",
                        "objective_id": "objective:dqp-012",
                        "title": "Move intent into the database",
                        "acceptance_criteria": ["projections rebuild"],
                    }
                ],
                "taskboard": [
                    {
                        "task_id": "DQP-012",
                        "task_cid": "task:cid:012",
                        "goal_id": "G20",
                        "goal_cid": "goal:cid:g20",
                        "acceptance_criteria": [
                            {
                                "criterion": "tests pass",
                                "required_digest": "sha256:" + ("11" * 32),
                                "evidence_kind": "validation",
                            }
                        ],
                        "validation_commands": [
                            "python -m pytest -q "
                            "test/api/test_agent_supervisor_intent_repository.py"
                        ],
                        "effects": [
                            {
                                "path": (
                                    "ipfs_accelerate_py/agent_supervisor/"
                                    "task_sources/intent_repository.py"
                                ),
                                "effect": "create",
                            }
                        ],
                    },
                    {
                        "task_id": "DQP-012-B",
                        "task_cid": "task:cid:012b",
                        "goal_id": "G20",
                        "goal_cid": "goal:cid:g20",
                        "depends_on": ["DQP-012"],
                        "acceptance_criteria": ["ready after A"],
                        "validation_commands": ["true"],
                    },
                ],
            }
        )
        assert receipt["task_count"] == 2
        assert receipt["plan_root_cid"]
        assert source.SCHEMA == DATABASE_TASK_SOURCE_SCHEMA

        snap = source.snapshot()
        assert snap.task_count == 2
        assert snap.goal_count >= 1
        assert snap.source_schema == DATABASE_TASK_SOURCE_SCHEMA

        task = source.get_task("DQP-012")
        assert task is not None
        assert task.task_cid == "task:cid:012"
        assert source.get_task("task:cid:012") is not None
        assert source.get_task("task:cid:012").task_cid == task.task_cid  # type: ignore[union-attr]

        page = source.list_tasks(limit=1)
        assert len(page.tasks) == 1
        second = source.list_tasks(cursor=page.next_cursor, limit=1)
        assert len(second.tasks) == 1
        assert {page.tasks[0].task_cid, second.tasks[0].task_cid} == {
            "task:cid:012",
            "task:cid:012b",
        }

        ready = source.ready_tasks()
        assert [item.task_cid for item in ready.tasks] == ["task:cid:012"]

        with pytest.raises(TaskSourceCompletionError):
            source.compare_and_set_status(
                task.task_cid,
                task.revision,
                "completed",
            )

        source.record_validation_result(
            task_cid=task.task_cid,
            outcome="passed",
            evidence_digest="sha256:" + ("11" * 32),
            argv=["pytest"],
        )
        result = source.compare_and_set_status(
            task.task_cid,
            task.revision,
            "completed",
            {"validation": "passed"},
            evidence_digests=["sha256:" + ("11" * 32)],
        )
        assert result.changed is True
        assert result.task.task_cid == "task:cid:012"
        assert result.task.status == "completed"

        ready_after = source.ready_tasks()
        assert [item.task_cid for item in ready_after.tasks] == ["task:cid:012b"]

        # Objective / plan identity APIs.
        assert source.get_objective("objective:dqp-012") is not None
        assert source.get_goal("goal:cid:g20") is not None
        assert source.get_plan(str(receipt["plan_root_cid"])) is not None

        assert source.projection_matches_events() is True

        with pytest.raises(TaskSourceBoundsError):
            source.list_tasks(limit=MAX_QUERY_LIMIT + 1)
    finally:
        source.close()


def test_reopened_source_infers_one_task_bound_root_without_cross_goal_guess(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "control.duckdb"
    plan_cid = "plan:root"
    with DatabaseTaskSource(database_path) as source:
        source.materialize(
            {
                "repository_tree_id": "tree:refinement",
                "plan_root_cid": plan_cid,
                "objectives": [
                    {
                        "goal_cid": "goal:root",
                        "goal_id": "G-ROOT",
                        "title": "Root",
                    },
                    {
                        "goal_cid": "goal:child",
                        "goal_id": "G-CHILD",
                        "title": "Child",
                        "parent_goal_cid": "goal:root",
                    },
                ],
                "plans": [
                    {
                        "plan_cid": plan_cid,
                        "goal_cid": "goal:root",
                        "status": "active",
                    }
                ],
                "taskboard": [
                    {
                        "task_cid": "task:child",
                        "task_id": "T-CHILD",
                        "goal_cid": "goal:child",
                        "plan_cid": plan_cid,
                        "acceptance_criteria": ["ok"],
                    }
                ],
            },
            repository_tree_id="tree:refinement",
            plan_root_cid=plan_cid,
        )

    with DatabaseTaskSource(
        database_path,
        install_schema=False,
        repository_tree_id="tree:refinement",
    ) as reopened:
        assert reopened.plan_root_cid == ""
        assert reopened.snapshot().plan_root_cid == plan_cid


def test_reopened_source_refuses_ambiguous_task_bound_plan_roots(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "control.duckdb"
    with DatabaseTaskSource(database_path) as source:
        source.materialize(
            {
                "repository_tree_id": "tree:ambiguous",
                "plans": [
                    {
                        "plan_cid": "plan:one",
                        "goal_cid": "goal:one",
                        "status": "active",
                    },
                    {
                        "plan_cid": "plan:two",
                        "goal_cid": "goal:two",
                        "status": "active",
                    },
                ],
                "objectives": [
                    {"goal_cid": "goal:one", "goal_id": "G-ONE"},
                    {"goal_cid": "goal:two", "goal_id": "G-TWO"},
                ],
                "taskboard": [
                    {
                        "task_cid": "task:one",
                        "task_id": "T-ONE",
                        "goal_cid": "goal:one",
                        "plan_cid": "plan:one",
                        "acceptance_criteria": ["one"],
                    },
                    {
                        "task_cid": "task:two",
                        "task_id": "T-TWO",
                        "goal_cid": "goal:two",
                        "plan_cid": "plan:two",
                        "acceptance_criteria": ["two"],
                    },
                ],
            },
            repository_tree_id="tree:ambiguous",
        )

    with DatabaseTaskSource(
        database_path,
        install_schema=False,
        repository_tree_id="tree:ambiguous",
    ) as reopened:
        assert reopened.snapshot().plan_root_cid == ""


def test_database_task_source_forwards_sorted_bounded_goal_edges(
    tmp_path: Path,
) -> None:
    with DatabaseTaskSource(tmp_path / "goal-edges.duckdb") as source:
        source.materialize(
            {
                "repository_tree_id": "tree:goal-edges",
                "objectives": [
                    {"goal_cid": "goal:root", "goal_id": "G0", "title": "Root"},
                    {"goal_cid": "goal:a", "goal_id": "GA", "title": "A"},
                    {"goal_cid": "goal:b", "goal_id": "GB", "title": "B"},
                ],
                "goal_edges": [
                    {
                        "parent_goal_cid": "goal:a",
                        "child_goal_cid": "goal:b",
                        "edge_kind": "goal_dependency",
                    },
                    {
                        "parent_goal_cid": "goal:root",
                        "child_goal_cid": "goal:a",
                        "edge_kind": "goal_parent",
                    },
                ],
                "taskboard": [
                    {
                        "task_cid": "task:goal-edges",
                        "task_id": "T-GOAL-EDGES",
                        "goal_cid": "goal:a",
                        "acceptance_criteria": ["goal graph is exact"],
                    }
                ],
            }
        )

        assert tuple(dict(item) for item in source.list_goal_edges()) == (
            {
                "parent_goal_cid": "goal:a",
                "child_goal_cid": "goal:b",
                "edge_kind": "goal_dependency",
            },
            {
                "parent_goal_cid": "goal:root",
                "child_goal_cid": "goal:a",
                "edge_kind": "goal_parent",
            },
        )
        with pytest.raises(TaskSourceBoundsError):
            source.list_goal_edges(limit=MAX_QUERY_LIMIT + 1)


def test_database_task_source_stale_cursor_and_cas_conflict(tmp_path: Path) -> None:
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        source.materialize(
            {
                "repository_tree_id": "tree:x",
                "objectives": [
                    {
                        "goal_cid": "goal:1",
                        "goal_id": "G1",
                        "title": "One",
                    }
                ],
                "taskboard": [
                    {
                        "task_cid": "task:1",
                        "task_id": "T1",
                        "goal_cid": "goal:1",
                        "acceptance_criteria": ["ok"],
                    },
                    {
                        "task_cid": "task:2",
                        "task_id": "T2",
                        "goal_cid": "goal:1",
                        "acceptance_criteria": ["ok"],
                    },
                ],
            }
        )
        page = source.list_tasks(limit=1)
        assert page.next_cursor
        # Corrupt cursor revision.
        with pytest.raises(TaskSourceConflictError):
            source.list_tasks(cursor=page.next_cursor[:-1] + "x", limit=1)

        task = source.get_task("task:1")
        assert task is not None
        source.record_evidence(
            task_cid=task.task_cid,
            evidence_kind="validation",
            digest="sha256:" + ("22" * 32),
        )
        source.compare_and_set_status(task.task_cid, task.revision, "in_progress")
        with pytest.raises(TaskSourceConflictError):
            source.compare_and_set_status(task.task_cid, task.revision, "blocked")


def test_database_task_source_applies_add_and_exact_unstarted_amend(
    tmp_path: Path,
) -> None:
    base_population = {
        "repository_tree_id": "tree:steer",
        "objectives": [
            {
                "goal_cid": "goal:steer",
                "goal_id": "G-STEER",
                "title": "Steer safely",
            }
        ],
        "tasks": [
            {
                "task_cid": "task:retained",
                "task_id": "STEER-001",
                "goal_cid": "goal:steer",
                "ordinal": 10,
                "status": "ready",
                "title": "Original task",
                "acceptance_criteria": ["original accepted"],
                "validation_commands": ["true"],
            },
            {
                "task_cid": "task:blocked",
                "task_id": "STEER-BLOCKED",
                "goal_cid": "goal:steer",
                "ordinal": 20,
                "status": "blocked",
                "title": "External authority remains blocked",
                "acceptance_criteria": ["operator authorization"],
                "validation_commands": ["true"],
            },
        ],
    }
    candidate_population = deepcopy(base_population)
    candidate_tasks = candidate_population["tasks"]
    assert isinstance(candidate_tasks, list)
    candidate_tasks[0]["title"] = "Amended before claim"
    candidate_tasks[0]["validation_commands"] = ["python -m pytest -q"]
    candidate_tasks[1]["ordinal"] = 30
    candidate_tasks.append(
        {
            "task_cid": "task:qualification",
            "task_id": "STEER-002",
            "goal_cid": "goal:steer",
            "ordinal": 1,
            "status": "ready",
            "title": "Independent qualification",
            "depends_on": [],
            "acceptance_criteria": ["qualification passes"],
            "validation_commands": ["true"],
        }
    )

    with DatabaseTaskSource(tmp_path / "candidate.duckdb") as candidate:
        candidate.materialize(
            candidate_population,
            repository_tree_id="tree:steer",
            plan_root_cid="plan:steer:v2",
        )
        projected_candidate = candidate.plan_projection()
        candidate_by_cid = {
            str(item["task_cid"]): item for item in projected_candidate["tasks"]
        }
        amended_spec_cid = task_projection_spec_cid(
            candidate_by_cid["task:retained"]
        )
        reprioritized_spec_cid = task_projection_spec_cid(
            candidate_by_cid["task:blocked"]
        )

    with DatabaseTaskSource(tmp_path / "control-steer.duckdb") as source:
        materialized = source.materialize(base_population)
        predecessor_root = str(materialized["plan_root_cid"])
        current_projection = source.plan_projection()
        current_by_cid = {
            str(item["task_cid"]): item for item in current_projection["tasks"]
        }
        current_spec_cid = task_projection_spec_cid(
            current_by_cid["task:retained"]
        )
        current_blocked_spec_cid = task_projection_spec_cid(
            current_by_cid["task:blocked"]
        )
        delta = SimpleNamespace(
            delta_cid="delta:steer:v2",
            items=(
                SimpleNamespace(
                    operation=PlanDeltaOperation.AMEND_UNSTARTED_TASK,
                    target_cid="task:retained",
                    expected_target_lifecycle=LifecycleState.UNSTARTED,
                    expected_target_spec_revision=current_spec_cid,
                    after_record_cid=amended_spec_cid,
                    effect_class=DeltaEffectClass.MATERIALIZABLE_NOW,
                ),
                SimpleNamespace(
                    operation=PlanDeltaOperation.REPRIORITIZE_UNSTARTED_TASK,
                    target_cid="task:blocked",
                    expected_target_lifecycle=LifecycleState.BLOCKED,
                    expected_target_spec_revision=current_blocked_spec_cid,
                    after_record_cid=reprioritized_spec_cid,
                    effect_class=DeltaEffectClass.MATERIALIZABLE_NOW,
                ),
                SimpleNamespace(
                    operation=PlanDeltaOperation.ADD_TASK,
                    target_cid="",
                    expected_target_lifecycle=LifecycleState.PROPOSED,
                    expected_target_spec_revision="",
                    after_record_cid="task:qualification",
                    effect_class=DeltaEffectClass.MATERIALIZABLE_NOW,
                ),
            ),
        )
        revision = SimpleNamespace(
            plan_root_cid="plan:steer:v2",
            revision_cid="revision:steer:v2",
        )
        delta.items[0].expected_target_spec_revision = "cid:stale-spec"
        with pytest.raises(TaskSourceConflictError, match="specification CAS"):
            source.apply_plan_revision(
                revision=revision,
                goal_graph=candidate_population,
                repository_tree_id="tree:steer",
                retained_task_cids=("task:retained", "task:blocked"),
                claimed_task_cids=(),
                origin="steer",
                delta=delta,
                store_continuation=object(),
                idempotency_key="steer:stale",
                fencing_token=1,
            )
        assert source.get_task("task:qualification") is None
        delta.items[0].expected_target_spec_revision = current_spec_cid
        receipt = source.apply_plan_revision(
            revision=revision,
            goal_graph=candidate_population,
            repository_tree_id="tree:steer",
            retained_task_cids=("task:retained", "task:blocked"),
            claimed_task_cids=(),
            origin="steer",
            delta=delta,
            store_continuation=object(),
            idempotency_key="steer:v2",
            fencing_token=1,
        )

        assert receipt["amended_task_cids"] == ["task:blocked", "task:retained"]
        assert receipt["added_task_cids"] == ["task:qualification"]
        assert receipt["projection_cid"] == source.plan_revision_projection_cid()
        retained = source.get_task("task:retained")
        assert retained is not None
        assert retained.status == "ready"
        assert retained.task_cid == "task:retained"
        assert retained.body["title"] == "Amended before claim"
        assert source.get_task("task:qualification") is not None
        blocked = source.get_task("task:blocked")
        assert blocked is not None
        assert blocked.status == "blocked"
        assert blocked.ordinal == 30
        assert source.get_plan(predecessor_root)["status"] == "continued"  # type: ignore[index]
        assert source.get_plan("plan:steer:v2")["status"] == "active"  # type: ignore[index]
        completion = source.completion_evidence_projection(
            task_cids=("task:retained",)
        )
        assert completion["completion_receipts"] == []


def test_single_transaction_emits_events_without_external_files(
    tmp_path: Path,
) -> None:
    """Mutations only touch the database path — no sidecar saga files."""

    db = tmp_path / "control.duckdb"
    with open_intent_repository(db) as repo:
        _seed_graph(repo)
        watermark = repo.event_watermark()
        assert watermark > 0
    # Only the duckdb file (and maybe wal) under tmp_path — no markdown/json saga.
    names = {path.name for path in tmp_path.iterdir()}
    assert "control.duckdb" in names
    unexpected = {
        name
        for name in names
        if name.endswith((".md", ".json", ".jsonl")) and "control" not in name
    }
    assert not unexpected
