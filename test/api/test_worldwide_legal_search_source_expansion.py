"""Bounded native source-child admission for one pinned inventory."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search import (
    ROOT_GOAL_ID, WorldwideAdmissionError, admit_goals_and_tasks,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_blockers import BlockerBoard
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_completion import _TASK
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime import atomic_json
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_source_expansion import (
    ADMISSION_SCHEMA, CHILD_SCHEMA, INVENTORY_SCHEMA, JOIN_SCHEMA, JOIN_TASK_ID, MAX_CHILDREN,
    MAX_UNITS, PARENT_TASK_ID, PHASES, RUN_ID, SourceExpansionError, admit_source_expansion,
    expand_inventory, main, published_contract, verify_child_id,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_watchdog import (
    empty_restart_journal, read_owned_json,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource


CONTRACT = Path("docs/worldwide/source-expansion.json")
RECEIPT = Path("artifacts/worldwide-legal-search/receipts/WWL-018.json")
MODULE = "ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_source_expansion"
ACCEPTANCE = "Native current-tree receipts bind this contract"
SOURCE_REV = "sha256:" + ("12" * 32)
OTHER = "sha256:" + ("ab" * 32)
NOW = 1_700_000_000
SEEDED_AT = NOW - 10
FOREST = {
    "site": {"path": "/site", "head": "1" * 40, "tree": "2" * 40},
    "accelerate": {"path": "/framework", "head": "3" * 40, "tree": "4" * 40},
}
BUDGETS = {"task": 3, "validation": 3, "merge": 3, "provider": 4, "repair": 2, "recover": 2}
INCIDENTS = {"reconcile": 1, "recover": 0, "diagnose": 2, "repair": 4}
PHASE_INDEX = {item["id"]: index for index, item in enumerate(PHASES)}
ARGV = ["python3", "-m", "pytest", "test/api/test_worldwide_legal_search_source_expansion.py", "-q"]


def _task(task_id: str, goal_id: str, depends_on: list[str], repository: str) -> dict:
    return {
        "id": task_id, "title": "Fixture task " + task_id, "status": "done", "depends_on": depends_on,
        "repository": repository, "goal_id": goal_id, "subgoal_id": goal_id,
        "outputs": [f"artifacts/worldwide-legal-search/receipts/{task_id}.json"],
        "evidence_output": f"artifacts/worldwide-legal-search/receipts/{task_id}.json",
        "acceptance": ACCEPTANCE, "validation_argv": [["python3", "-c", "print('validated fixture')"]],
        "bounds": {"max_attempts": 3, "wall_seconds_per_attempt": 5400},
    }


def _goal(goal_id: str, task_ids: list[str], parents: list[str] | None = None) -> dict:
    return {
        "id": goal_id,
        "title": "Fixture root" if goal_id == ROOT_GOAL_ID else "Fixture child " + goal_id,
        "parent_ids": [] if parents is None and goal_id == ROOT_GOAL_ID else (parents if parents is not None else [ROOT_GOAL_ID]),
        "task_ids": list(task_ids),
        "acceptance": ACCEPTANCE,
    }


def campaign() -> tuple[dict, str, str]:
    specials = [
        ("WWL-016", []), ("WWL-047", []), ("WWL-081", []),
        ("WWL-018", ["WWL-016", "WWL-047", "WWL-081"]), ("WWL-017", ["WWL-018"]),
    ]
    tasks = []
    goals = []
    for index, number in enumerate((10, 30, 40, 50, 60, 70, 80, 90, 100), start=1):
        task_id = f"WWL-{index:03d}"
        goal_id = f"WWL-G{number:03d}"
        tasks.append(_task(task_id, goal_id, [], "site"))
        goals.append(_goal(goal_id, [task_id]))
    goals.append(_goal("WWL-G020", [item[0] for item in specials]))
    tasks.extend(_task(task_id, "WWL-G020", depends_on, "accelerate") for task_id, depends_on in specials)
    by_id = {goal["id"]: goal for goal in goals}
    ordered = [by_id[f"WWL-G{number:03d}"] for number in range(10, 110, 10)]
    plan = {
        "schema": "worldwide-legal-search.plan@1", "namespace": RUN_ID,
        "goals": [_goal(ROOT_GOAL_ID, [item["id"] for item in tasks], []), *ordered],
        "tasks": tasks,
    }
    heap = ["# Worldwide legal search goal bootstrap", ""]
    for goal in plan["goals"]:
        heap.extend([
            f"## {goal['id']} {goal['title']}", "- Status: active",
            "- Parent goal ids json: " + json.dumps(goal["parent_ids"]),
            "- Required task ids json: " + json.dumps(goal["task_ids"]),
            "- Acceptance: " + goal["acceptance"], "",
        ])
    markdown = "".join(f"## {task['id']} {task['title']}\n- Status: done\n" for task in tasks)
    return plan, "\n".join(heap), markdown


def sources() -> list[dict]:
    return [
        {
            "source_id": "justicedao", "resolved_revision": SOURCE_REV, "eligible": True,
            "configs": [{
                "config_id": "statutes", "splits": ["train"],
                "jurisdictions": [
                    {"jurisdiction_id": "nl", "level": "national"},
                    {"jurisdiction_id": "nl-amsterdam", "level": "local"},
                    {"jurisdiction_id": "eu", "level": "supranational"},
                ],
                "shards": [{"shard_id": "0000", "bytes": 128}],
            }],
        },
        {
            "source_id": "gated-repo", "resolved_revision": "", "eligible": False,
            "blocker": {"kind": "gated-input", "reason": "license gate closed"},
            "configs": [{
                "config_id": "opinions", "splits": ["holdout"],
                "jurisdictions": [{"jurisdiction_id": "us-ca", "level": "regional"}],
                "shards": [{"shard_id": "0001", "bytes": 64}],
            }],
        },
        {
            "source_id": "court-mirror", "resolved_revision": "", "eligible": True,
            "access_error": {"status": 503, "reason": "upstream status 503"}, "configs": [],
        },
    ]


def inventory(plan_revision: str, *, complete: bool = True, source_rows: list[dict] | None = None, declare_plan: bool = True) -> dict:
    body = {"schema": INVENTORY_SCHEMA, "namespace": RUN_ID, "complete": complete, "sources": sources() if source_rows is None else source_rows}
    if declare_plan:
        body["active_plan_revision"] = plan_revision
    return body


def counted(count: int) -> dict:
    rows = [{
        "source_id": "justicedao", "resolved_revision": SOURCE_REV, "eligible": True,
        "configs": [{
            "config_id": "statutes", "splits": ["train"],
            "jurisdictions": [{"jurisdiction_id": f"j{index:02d}", "level": "national"} for index in range(count)],
            "shards": [{"shard_id": "0000", "bytes": 1}],
        }],
    }]
    return inventory(OTHER, source_rows=rows)


def write_control_plane(state_root: Path, plan_revision: str) -> None:
    atomic_json(state_root / "watchdog" / "state.json", {
        "schema": "wwl.watchdog.state@1", "run_id": RUN_ID, "actions": 7, "repairs": 2,
        "incidents": dict(INCIDENTS), "budgets": dict(BUDGETS), "progress_token": "progress-7",
        "policy_digest": "sha256:" + ("cd" * 32), "required_task_ids": ["WWL-016", "WWL-018"],
    })
    atomic_json(state_root / "campaign-readiness.json", {
        "schema": "wwl.watchdog.ready@1", "run_id": RUN_ID, "repository": "/framework",
        "policy_id": "policy:implementation-daemon", "native_revision": "sha256:" + ("a1" * 32),
        "admitted_plan_revision": plan_revision, "bindings": {"lifecycle_restart": False},
        "required_task_ids": ["WWL-017"], "completion_authority": False,
    })
    journal = empty_restart_journal(RUN_ID)
    journal.update({
        "required_task_ids": ["WWL-081"], "actions": {"reconcile": 3}, "guardian_restarts": ["guardian-1"],
        "sealed_hold_markers": [], "loop_policy_digest": "sha256:" + ("ef" * 32),
    })
    atomic_json(state_root / "watchdog" / "restart-journal.json", journal)


def seed_blockers(path: Path, *, extra: dict | None = None) -> None:
    board = BlockerBoard(RUN_ID)
    board.records["WWL-016"] = {
        "task_id": "WWL-016", "classification": "gated-input", "kind": "gated-input",
        "prerequisite_digest": "sha256:" + ("44" * 32), "exhausted_budgets": ["provider"],
        "exhaustion_source": "ledger", "incident_key": "", "classified_at": SEEDED_AT, "recheckable": True,
    }
    if extra is not None:
        board.records[extra["task_id"]] = extra
    board.admit_probe(task_id="WWL-016", probe_id="recheck-wwl-016", not_before=SEEDED_AT, reason="recheck the gated corpus")
    board.save(path)


def load_tasks(database: Path) -> dict:
    with DatabaseTaskSource(database, owner_id=RUN_ID + ":source-expansion-read") as source:
        page = source.list_tasks(limit=500)
        assert page.next_cursor == ""
        return {task.task_alias: task for task in page.tasks}


def admit(database: Path, state_root: Path, plan_revision: str, document: dict, *, blocker_path: Path | None = None, now: int = NOW) -> dict:
    return admit_source_expansion(
        database, document, active_plan_revision=plan_revision, state_root=state_root,
        blocker_path=blocker_path, now=now,
    )


def test_published_contract_matches_the_document():
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert CONTRACT.read_text(encoding="utf-8") == json.dumps(published_contract(), indent=2, sort_keys=True) + "\n"
    assert payload == published_contract()
    assert payload["schema"] == "worldwide-legal-search.source-expansion@1"
    assert payload["task_id"] == "WWL-018"
    assert payload["join_task_id"] == "WWL-017"
    assert payload["depends_on"] == ["WWL-016", "WWL-047", "WWL-081"]
    assert payload["max_children"] == MAX_CHILDREN == 64
    assert payload["max_units"] == MAX_UNITS == 12
    assert [item["id"] for item in payload["phases"]] == ["normalize", "index", "geo", "publish", "verify"]
    assert [item["title"] for item in payload["phases"]] == ["normalize", "index", "geography", "immutable publication", "verification"]
    assert payload["completion_authority"] is False
    assert payload["execution_completed"] is False
    assert payload["self_certified_completion"] is False


def test_units_are_deterministic_and_stay_inside_the_child_bound():
    forward = expand_inventory(inventory(OTHER), active_plan_revision=OTHER)
    flipped = copy.deepcopy(inventory(OTHER))
    flipped["sources"].reverse()
    for source in flipped["sources"]:
        source["configs"].reverse()
        for config in source["configs"]:
            config["splits"].reverse()
            config["jurisdictions"].reverse()
            config["shards"].reverse()
    backward = expand_inventory(flipped, active_plan_revision=OTHER)
    assert forward["schema"] == ADMISSION_SCHEMA
    assert forward["inventory_revision"] == backward["inventory_revision"]
    assert [item["task_id"] for item in forward["children"]] == [item["task_id"] for item in backward["children"]]
    assert len(forward["children"]) == 25
    assert forward["national_count"] == forward["regional_count"] == forward["local_count"] == 1
    assert forward["supranational_count"] == forward["unknown_count"] == 1
    assert forward["jurisdiction_counts"] == [
        {"jurisdiction_id": "eu", "level": "supranational", "units": 1},
        {"jurisdiction_id": "nl", "level": "national", "units": 1},
        {"jurisdiction_id": "nl-amsterdam", "level": "local", "units": 1},
        {"jurisdiction_id": "unknown", "level": "unknown", "units": 1},
        {"jurisdiction_id": "us-ca", "level": "regional", "units": 1},
    ]
    phases = [item["id"] for item in PHASES]
    by_ordinal: dict[int, list[dict]] = {}
    for child in forward["children"]:
        assert _TASK.fullmatch(child["task_id"])
        assert child["task_id"].endswith(f"-{child['phase']}-{child['ordinal']:04d}")
        assert child["schema"] == CHILD_SCHEMA
        assert child["completion_authority"] is False
        assert child["active_plan_revision"] == OTHER
        assert child["inventory_revision"] == forward["inventory_revision"]
        by_ordinal.setdefault(child["ordinal"], []).append(child)
    assert sorted(by_ordinal) == list(range(5))
    for ordinal, group in by_ordinal.items():
        assert [item["phase"] for item in group] == phases
        assert len({(item["source_id"], item["config_id"], item["split_id"], item["jurisdiction_id"], item["shard_id"]) for item in group}) == 1
        assert group[0]["depends_on"] == [PARENT_TASK_ID]
        assert all(item["depends_on"] == [] for item in group[1:])
    mirror = by_ordinal[0]
    assert mirror[0]["source_id"] == "court-mirror"
    assert mirror[0]["config_id"] == "unavailable"
    assert mirror[0]["status"] == "blocked"
    assert mirror[0]["blocker_kind"] == "source-outage"
    assert mirror[0]["access_status"] == 503
    assert {item["grain"] for item in forward["children"] if item["phase"] == "geo"} == {"jurisdiction"}
    assert {item["phase_title"] for item in forward["children"] if item["phase"] == "publish"} == {"immutable publication"}
    bound = expand_inventory(counted(MAX_UNITS), active_plan_revision=OTHER)
    assert len(bound["children"]) == MAX_UNITS * len(PHASES) == 60
    assert bound["national_count"] == MAX_UNITS
    with pytest.raises(SourceExpansionError, match="expansion exceeds the bounded child task set"):
        expand_inventory(counted(MAX_UNITS + 1), active_plan_revision=OTHER)


def test_partial_secret_and_unblocked_inventory_fail_closed(tmp_path):
    partial = inventory(OTHER, complete=False)
    database = tmp_path / "nested" / "control.duckdb"
    with pytest.raises(SourceExpansionError, match="partial inventory"):
        admit(database, tmp_path, OTHER, partial)
    assert not database.exists()
    assert not database.parent.exists()
    with pytest.raises(SourceExpansionError, match="expansion exceeds the bounded child task set"):
        admit(database, tmp_path, OTHER, counted(MAX_UNITS + 1))
    assert not database.exists()
    secret = sources()
    secret[0]["configs"][0]["token"] = "not-a-value"
    with pytest.raises(SourceExpansionError, match="inventory contains a secret field"):
        expand_inventory(inventory(OTHER, source_rows=secret), active_plan_revision=OTHER)
    reset = sources()
    reset[0]["reset_budgets"] = True
    with pytest.raises(SourceExpansionError, match="inventory asks to reset budgets"):
        expand_inventory(inventory(OTHER, source_rows=reset), active_plan_revision=OTHER)
    open_gate = sources()
    del open_gate[1]["blocker"]
    with pytest.raises(SourceExpansionError, match="explicit blocker"):
        expand_inventory(inventory(OTHER, source_rows=open_gate), active_plan_revision=OTHER)
    unknown = sources()
    unknown[0]["configs"][0]["jurisdictions"][0]["level"] = "unknown"
    with pytest.raises(SourceExpansionError, match="explicit level"):
        expand_inventory(inventory(OTHER, source_rows=unknown), active_plan_revision=OTHER)
    declared = inventory(OTHER)
    declared["inventory_revision"] = "sha256:" + ("00" * 32)
    with pytest.raises(SourceExpansionError, match="inventory revision"):
        expand_inventory(declared, active_plan_revision=OTHER)
    moved = inventory(OTHER)
    moved["active_plan_revision"] = "sha256:" + ("00" * 32)
    with pytest.raises(SourceExpansionError, match="active plan revision"):
        expand_inventory(moved, active_plan_revision=OTHER)
    privileged = sources()
    privileged[1]["blocker"] = {"kind": "credentials", "reason": "signing material missing"}
    privileged[1]["access_error"] = {"status": 503, "reason": "upstream status 503"}
    privileged[1]["eligible"] = True
    expanded = expand_inventory(inventory(OTHER, source_rows=privileged), active_plan_revision=OTHER)
    gated = [item for item in expanded["children"] if item["source_id"] == "gated-repo"]
    assert {item["blocker_kind"] for item in gated} == {"credentials"}
    assert {item["access_status"] for item in gated} == {503}
    assert {item["status"] for item in gated} == {"blocked"}
    with pytest.raises(SourceExpansionError, match="clock"):
        admit_source_expansion(database, inventory(OTHER), active_plan_revision=OTHER, state_root=tmp_path, now=-1)


def test_watchdog_and_database_preconditions_fail_before_writes(tmp_path):
    state_root = tmp_path / "state"
    database = tmp_path / "missing.duckdb"
    with pytest.raises(SourceExpansionError, match="watchdog state root is missing"):
        admit(database, state_root, OTHER, inventory(OTHER))
    assert not database.exists()
    state_root.mkdir()
    with pytest.raises(SourceExpansionError, match="watchdog task set is absent or unreadable"):
        admit(database, state_root, OTHER, inventory(OTHER))
    assert not database.exists()
    write_control_plane(state_root, OTHER)
    before = (state_root / "watchdog" / "state.json").read_bytes()
    with pytest.raises(SourceExpansionError, match="native task database is missing"):
        admit(database, state_root, OTHER, inventory(OTHER))
    assert not database.exists()
    assert (state_root / "watchdog" / "state.json").read_bytes() == before
    broken = json.loads(before)
    broken["actions"] = "7"
    atomic_json(state_root / "watchdog" / "state.json", broken)
    with pytest.raises(SourceExpansionError, match="watchdog budget"):
        admit(database, state_root, OTHER, inventory(OTHER))
    assert not database.exists()
    atomic_json(state_root / "watchdog" / "state.json", json.loads(before))
    readiness = read_owned_json(state_root / "campaign-readiness.json")
    readiness["completion_authority"] = True
    atomic_json(state_root / "campaign-readiness.json", readiness)
    with pytest.raises(SourceExpansionError, match="completion authority"):
        admit(database, state_root, OTHER, inventory(OTHER))
    assert not database.exists()


def test_native_admission_preserves_progress_budgets_and_fails_closed(tmp_path):
    plan, heap, markdown = campaign()
    database = tmp_path / "control.duckdb"
    admitted = admit_goals_and_tasks(
        plan, heap, database_path=database, source_forest=FOREST,
        config_path=tmp_path / "runtime.json", python_executable="python3", task_markdown=markdown,
    )
    plan_cid = admitted["plan_root_cid"]
    assert admitted["task_statuses"]["WWL-018"] == "todo"
    assert admitted["task_statuses"]["WWL-017"] == "todo"
    assert admitted["root_child_ids"] == [f"WWL-G{number:03d}" for number in range(10, 110, 10)]
    state_root = tmp_path / "state"
    write_control_plane(state_root, plan_cid)
    state_path = state_root / "watchdog" / "state.json"
    readiness_path = state_root / "campaign-readiness.json"
    journal_path = state_root / "watchdog" / "restart-journal.json"
    blocker_path = tmp_path / "blockers.json"
    before_tasks = load_tasks(database)
    assert len(before_tasks) == 14
    parent_revision = before_tasks[PARENT_TASK_ID].revision
    join_revision = before_tasks[JOIN_TASK_ID].revision
    state_bytes = state_path.read_bytes()
    with pytest.raises(SourceExpansionError, match="active plan revision"):
        admit(database, state_root, OTHER, inventory(plan_cid, declare_plan=False))
    assert state_path.read_bytes() == state_bytes
    assert {alias: task.revision for alias, task in load_tasks(database).items()} == {alias: task.revision for alias, task in before_tasks.items()}
    preview = expand_inventory(inventory(plan_cid), active_plan_revision=plan_cid)
    victim = next(item for item in preview["children"] if item["source_id"] == "gated-repo" and item["phase"] == "normalize")
    seed_blockers(blocker_path, extra={
        "task_id": victim["task_id"], "classification": "permission", "kind": "permission",
        "prerequisite_digest": "sha256:" + ("55" * 32), "exhausted_budgets": [], "exhaustion_source": "",
        "incident_key": "", "classified_at": SEEDED_AT, "recheckable": True,
    })
    blocker_bytes = blocker_path.read_bytes()
    with pytest.raises(SourceExpansionError, match="blocker kind would change"):
        admit(database, state_root, plan_cid, inventory(plan_cid), blocker_path=blocker_path)
    assert blocker_path.read_bytes() == blocker_bytes
    assert state_path.read_bytes() == state_bytes
    assert len(load_tasks(database)) == 14
    seed_blockers(blocker_path)
    result = admit(database, state_root, plan_cid, inventory(plan_cid), blocker_path=blocker_path)
    assert result["schema"] == ADMISSION_SCHEMA
    assert result["active_plan_revision"] == plan_cid
    assert result["inventory_revision"] == preview["inventory_revision"]
    assert result["inserted_count"] == 25
    assert result["join_rewritten"] is True
    assert result["idempotent_replay"] is False
    assert result["completion_authority"] is False
    assert result["execution_completed"] is False
    assert result["watchdog"]["actions"] == 7
    assert result["watchdog"]["repairs"] == 2
    assert result["watchdog"]["budgets_reset"] is False
    assert result["watchdog"]["written"] is True
    assert result["national_count"] == result["local_count"] == result["regional_count"] == 1
    assert result["supranational_count"] == result["unknown_count"] == 1
    assert result["blockers_preserved"] == ["WWL-016"]
    assert len(result["blocked_task_ids"]) == 10
    assert len(result["runnable_task_ids"]) == 15
    tasks = load_tasks(database)
    assert len(tasks) == 39
    parent = tasks[PARENT_TASK_ID]
    joined = tasks[JOIN_TASK_ID]
    assert parent.revision == parent_revision
    assert parent.status == "todo"
    assert joined.revision == join_revision + 1
    assert joined.status == "todo"
    assert parent.goal_cid == joined.goal_cid
    assert parent.plan_cid == joined.plan_cid == plan_cid
    document = joined.body["source_expansion_join"]
    assert document["schema"] == JOIN_SCHEMA
    assert document["completion_authority"] is False
    assert document["required_receipts"] == result["required_receipts"]
    assert document["required_receipt_ids"] == [item["receipt_id"] for item in preview["children"]]
    assert set(joined.dependencies) >= {item["task_cid"] for item in preview["children"]}
    assert parent.task_cid in joined.dependencies
    child_argv = ["python3", "-m", MODULE, "verify-child"]
    for child in preview["children"]:
        stored = tasks[child["task_id"]]
        assert stored.task_cid == child["task_cid"]
        assert stored.plan_cid == plan_cid
        assert stored.goal_cid == parent.goal_cid
        assert stored.status == child["status"]
        assert stored.body["schema"] == CHILD_SCHEMA
        assert stored.body["receipt_id"] == child["receipt_id"]
        assert stored.body["blocker_kind"] == child["blocker_kind"]
        assert stored.body["completion_authority"] is False
        assert stored.ordinal == 1000 + child["ordinal"] * len(PHASES) + PHASE_INDEX[child["phase"]]
        assert list(stored.validations[0]["argv"]) == [*child_argv, child["task_id"]]
        assert _TASK.fullmatch(stored.task_alias)
    for ordinal, group in _groups(preview["children"]).items():
        stored_group = [tasks[item["task_id"]] for item in group]
        assert stored_group[0].dependencies == (parent.task_cid,)
        for earlier, later in zip(stored_group, stored_group[1:]):
            assert later.dependencies == (earlier.task_cid,)
        assert ordinal >= 0
    state = read_owned_json(state_path)
    assert state["actions"] == 7 and state["repairs"] == 2
    assert state["incidents"] == INCIDENTS
    assert state["budgets"] == BUDGETS
    assert state["progress_token"] == "progress-7"
    assert state["policy_digest"] == "sha256:" + ("cd" * 32)
    assert state["required_task_ids"] == ["WWL-016", "WWL-018", *[item["task_id"] for item in preview["children"]]]
    readiness = read_owned_json(readiness_path)
    assert readiness["required_task_ids"] == ["WWL-017", *[item["task_id"] for item in preview["children"]]]
    assert readiness["bindings"] == {"lifecycle_restart": False}
    assert readiness["completion_authority"] is False
    journal = read_owned_json(journal_path)
    assert journal["actions"] == {"reconcile": 3}
    assert journal["guardian_restarts"] == ["guardian-1"]
    assert journal["loop_policy_digest"] == "sha256:" + ("ef" * 32)
    assert journal["required_task_ids"] == ["WWL-081", *[item["task_id"] for item in preview["children"]]]
    board = BlockerBoard.load(blocker_path)
    preserved = board.records["WWL-016"]
    assert preserved["kind"] == "gated-input"
    assert preserved["exhausted_budgets"] == ["provider"]
    assert preserved["exhaustion_source"] == "ledger"
    assert preserved["classified_at"] == float(SEEDED_AT)
    assert board.probes["recheck-wwl-016"]["consumed"] is False
    assert board.probes["recheck-wwl-016"]["not_before"] == float(SEEDED_AT)
    for task_id in result["blocked_task_ids"]:
        record = board.records[task_id]
        assert record["kind"] == tasks[task_id].body["blocker_kind"]
        assert record["exhausted_budgets"] == []
        assert record["classified_at"] == float(NOW)
        assert record["prerequisite_digest"] == tasks[task_id].body["receipt_id"]
        assert record["recheckable"] is True
    progressed_alias = next(item["task_id"] for item in preview["children"] if item["source_id"] == "justicedao" and item["jurisdiction_id"] == "eu" and item["phase"] == "normalize")
    with DatabaseTaskSource(database, owner_id=RUN_ID + ":source-expansion-read") as source:
        current = source.get_task(progressed_alias)
        progressed = source.compare_and_set_status(current.task_cid, current.revision, "in_progress", {"operation": "fixture-progress"})
        progressed_revision = progressed.task.revision
    assert progressed_revision == tasks[progressed_alias].revision + 1
    frozen_state = state_path.read_bytes()
    frozen_readiness = readiness_path.read_bytes()
    frozen_journal = journal_path.read_bytes()
    replay = admit(database, state_root, plan_cid, inventory(plan_cid), blocker_path=blocker_path, now=NOW + 50)
    assert replay["idempotent_replay"] is True
    assert replay["inserted_count"] == 0
    assert replay["join_rewritten"] is False
    assert replay["watchdog"]["written"] is False
    assert replay["watchdog"]["actions"] == 7
    assert replay["watchdog"]["budgets_reset"] is False
    assert state_path.read_bytes() == frozen_state
    assert readiness_path.read_bytes() == frozen_readiness
    assert journal_path.read_bytes() == frozen_journal
    replayed = load_tasks(database)
    assert replayed[progressed_alias].status == "in_progress"
    assert replayed[progressed_alias].revision == progressed_revision
    assert replayed[JOIN_TASK_ID].revision == joined.revision
    assert replayed[PARENT_TASK_ID].revision == parent_revision
    assert BlockerBoard.load(blocker_path).records["WWL-016"]["exhausted_budgets"] == ["provider"]
    assert BlockerBoard.load(blocker_path).records[result["blocked_task_ids"][0]]["classified_at"] == float(NOW)
    changed = inventory(plan_cid)
    changed["sources"][0]["configs"][0]["shards"][0]["bytes"] = 129
    with pytest.raises(SourceExpansionError, match="source child contract changed"):
        admit(database, state_root, plan_cid, changed, blocker_path=blocker_path, now=NOW + 60)
    assert state_path.read_bytes() == frozen_state
    assert load_tasks(database)[progressed_alias].revision == progressed_revision
    shrunk = inventory(plan_cid)
    shrunk["sources"] = [item for item in shrunk["sources"] if item["source_id"] != "court-mirror"]
    with pytest.raises(SourceExpansionError, match="required source child disappeared"):
        admit(database, state_root, plan_cid, shrunk, blocker_path=blocker_path, now=NOW + 70)
    assert state_path.read_bytes() == frozen_state
    remaining = load_tasks(database)
    assert len(remaining) == 39
    assert remaining[progressed_alias].status == "in_progress"
    assert any(task.body.get("source_id") == "court-mirror" for task in remaining.values())
    with pytest.raises(WorldwideAdmissionError, match="task alias set"):
        admit_goals_and_tasks(
            plan, heap, database_path=database, source_forest=FOREST,
            config_path=tmp_path / "runtime.json", python_executable="python3", task_markdown=markdown,
        )
    assert load_tasks(database)[progressed_alias].status == "in_progress"
    assert load_tasks(database)[progressed_alias].revision == progressed_revision


def _groups(children: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for child in children:
        grouped.setdefault(child["ordinal"], []).append(child)
    return grouped


def test_verify_child_command(capsys):
    sample = "WWL-SRC-0123456789abcdef-verify-0001"
    checked = verify_child_id(sample)
    assert checked["phase"] == "verify"
    assert checked["completion_authority"] is False
    assert checked["execution_completed"] is False
    with pytest.raises(SourceExpansionError, match="source child id is outside the completion contract"):
        verify_child_id("WWL-018")
    assert main(["contract"]) == 0
    assert json.loads(capsys.readouterr().out) == published_contract()
    assert main(["verify-child", sample]) == 0
    assert json.loads(capsys.readouterr().out)["task_id"] == sample
    assert main(["verify-child", "WWL-018"]) == 1
    failure = json.loads(capsys.readouterr().out)
    assert failure["completion_authority"] is False
    assert "source child id is outside the completion contract" in failure["error"]
    completed = subprocess.run([sys.executable, "-m", MODULE, "verify-child", sample], check=False, capture_output=True, text=True)
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["task_id"] == sample
    rejected = subprocess.run([sys.executable, "-m", MODULE, "verify-child", "WWL-018"], check=False, capture_output=True, text=True)
    assert rejected.returncode == 1
    assert json.loads(rejected.stdout)["completion_authority"] is False


def test_receipt_does_not_grant_completion():
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["schema"] == "worldwide-legal-search.task-receipt@1"
    assert receipt["task_id"] == "WWL-018"
    assert receipt["title"] == "Expand the inventory into bounded native source child tasks"
    assert receipt["namespace"] == RUN_ID
    assert receipt["root_goal_id"] == "WWL-G000"
    assert receipt["goal_id"] == "WWL-G020"
    assert receipt["subgoal_id"] == "WWL-G020.S2"
    assert receipt["adapter"] == MODULE
    assert receipt["depends_on"] == ["WWL-016", "WWL-047", "WWL-081"]
    assert receipt["completion_authority"] is False
    assert receipt["execution_completed"] is False
    assert receipt["self_certified_completion"] is False
    assert receipt["outputs"] == [
        "test/api/test_worldwide_legal_search_source_expansion.py",
        "docs/worldwide/source-expansion.json",
        "ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_source_expansion.py",
        "artifacts/worldwide-legal-search/receipts/WWL-018.json",
    ]
    assert receipt["validation_argv"] == ARGV
    assert receipt["validation"]["argv"] == ARGV
    assert receipt["validation"]["result"].endswith(" passed")
    assert receipt["validation"]["result"][0].isdigit()
    assert receipt["validation"]["pytest"] == "9.1.1"
    assert receipt["validation"]["python"] == "3.12.3"
    assert receipt["validation"]["interpreter"] == "/home/barberb/lift_coding/external/ipfs_datasets/.venv/bin/python3"
    text = "\n".join(receipt["limitations"])
    for phrase in ("partial", "disappeared", "inventory", "active plan", "blocker", "WWL-017", "watchdog", "budget", "/usr/bin/python3.12", "pytest"):
        assert phrase in text
    assert receipt["limitations"][-1] == "This receipt does not grant campaign completion or task completion."
