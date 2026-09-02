"""Cross-lane authority coverage for the sealed retained predecessors."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS,
    DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATABASE_FENCED_PROVIDER_HISTORICAL_RETAINED_ADMISSION_SCHEMA,
    DATABASE_FENCED_PROVIDER_HISTORICAL_RETAINED_CONSUMPTION_SCHEMA,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
    database_fenced_provider_historical_retained_admission_valid,
    database_fenced_provider_historical_retained_consumption_valid,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
)
from test.api.test_agent_supervisor_retained_occurrence_recovery_v2 import (
    _admission,
    _arm_retrying_task,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for retained cross-lane authority tests",
)

_CONTROL_STORE_ID = "test-retained-cross-lane-control"
_CONTROL_STORE_GENERATION = "pctdd-v1-g9"


def _pins() -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(item)
        for item in (
            DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN,
            *DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS,
        )
    )


def _open_lane(
    root: Path,
    *,
    lane: int,
    owner: str,
    task_source: Any = None,
) -> DatabaseImplementationDaemon:
    lane_root = root / f"lane-{lane}"
    lane_root.mkdir(parents=True)
    return DatabaseImplementationDaemon(
        database_path=lane_root / "control.duckdb",
        coordination_path=lane_root / "coordination.duckdb",
        execution_path=lane_root / "execution.duckdb",
        owner_session_id=owner,
        authority_mode="embedded",
        task_source_kind="duckdb",
        task_source=task_source,
        task_shard_count=4,
        task_shard_index=lane,
        strict_task_sharding=True,
        control_store_id=_CONTROL_STORE_ID,
        control_store_generation=_CONTROL_STORE_GENERATION,
    )


def _seed_predecessor(
    authority: DatabaseImplementationDaemon,
    pin: Mapping[str, Any],
) -> None:
    authority._require_connection().execute(
        """
        INSERT INTO database_task_attempts(
            attempt_id, claim_id, task_cid, task_alias, attempt_number,
            owner_session_id, fencing_token, fence_epoch, lease_id,
            committed_phase, status, started_at_ms, finished_at_ms,
            revision, body_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, '{}')
        """,
        [
            pin["predecessor_attempt_id"],
            pin["predecessor_claim_id"],
            pin["task_cid"],
            pin["task_alias"],
            pin["predecessor_attempt_number"],
            pin["predecessor_owner_session_id"],
            pin["predecessor_fencing_token"],
            pin["predecessor_fence_epoch"],
            pin["predecessor_lease_id"],
            "failed",
            "failed",
            1,
            1,
        ],
    )
    attempt = authority.get_attempt(str(pin["predecessor_attempt_id"]))
    assert attempt is not None
    retained_authority = (
        daemon_module._database_fenced_provider_recovery_authority_for_occurrence(
            dict(pin)
        )
    )
    assert retained_authority is not None
    credit = dict(retained_authority["credit_builder"](dict(pin)))
    fence = authority._install_retained_recovery_dispatch_fence(
        attempt,
        occurrence=dict(pin),
        credit=credit,
    )
    assert fence["state"] == "sealed"
    admitted = authority._transition_fenced_provider_recovery_dispatch_fence(
        attempt,
        expected_state="sealed",
        new_state="admission_pending",
    )
    assert admitted["state"] == "admission_pending"
    admitted = authority._transition_fenced_provider_recovery_dispatch_fence(
        attempt,
        expected_state="admission_pending",
        new_state="admitted",
    )
    assert admitted["state"] == "admitted"


def _historical_admission(
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_alias: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    occurrence, _credit, _inner, _outer, legacy = _admission(
        monkeypatch,
        task_alias=task_alias,
    )
    admission = dict(legacy)
    admission["schema"] = (
        DATABASE_FENCED_PROVIDER_HISTORICAL_RETAINED_ADMISSION_SCHEMA
    )
    admission["historical_occurrence_authority_id"] = (
        "sha256:"
        + hashlib.sha256(task_alias.encode("utf-8")).hexdigest()
    )
    admission["controller_quiescence_receipt_id"] = "sha256:" + "9" * 64
    admission.pop("admission_id")
    admission["admission_id"] = (
        daemon_module._database_fenced_provider_retained_digest(admission)
    )
    assert database_fenced_provider_historical_retained_admission_valid(
        admission
    )
    return occurrence, admission


def _historical_population(
    records: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "repository_tree_id": "tree:historical-cross-lane-claim",
        "objectives": [
            {
                "objective_id": "objective:historical-cross-lane-claim",
                "objective_alias": "PCTDD-O-HISTORICAL-CLAIM",
                "title": "Historical retained claim handoff",
                "goal_cid": "goal:historical-cross-lane-claim",
                "goal_alias": "PCTDD-G-HISTORICAL-CLAIM",
                "status": "open",
            }
        ],
        "tasks": [
            {
                "task_cid": occurrence["task_cid"],
                "task_id": occurrence["task_alias"],
                "goal_cid": "goal:historical-cross-lane-claim",
                "status": (
                    "retrying"
                    if int(occurrence["blocked_task_revision"]) % 2 == 0
                    else "ready"
                ),
                "priority": "P0",
                "ordinal": ordinal,
                "title": "Historical one-shot retained claim",
                "validations": [["python", "-m", "pytest", "-q"]],
            }
            for ordinal, (occurrence, _admission_record) in enumerate(
                records,
                start=1,
            )
        ],
    }


def test_retained_attempt_population_routes_to_exact_existing_home_lanes(
    tmp_path: Path,
) -> None:
    pins = _pins()
    owners: dict[int, set[str]] = {0: set(), 3: set()}
    for pin in pins:
        digest = hashlib.sha256(str(pin["task_alias"]).encode()).hexdigest()
        home = int(digest[:8], 16) % 4
        assert home in owners
        owners[home].add(str(pin["predecessor_owner_session_id"]))
    assert all(len(values) == 1 for values in owners.values())
    lane0 = _open_lane(tmp_path, lane=0, owner=next(iter(owners[0])))
    lane3 = _open_lane(
        tmp_path,
        lane=3,
        owner=next(iter(owners[3])),
        task_source=lane0.task_source,
    )
    try:
        by_lane = {0: lane0, 3: lane3}
        for pin in pins:
            home = lane0._task_home_shard_index(str(pin["task_alias"]))
            assert home in by_lane
            _seed_predecessor(by_lane[home], pin)

        p034 = next(pin for pin in pins if pin["task_alias"] == "PCTDD-034")
        assert lane0.get_attempt(str(p034["predecessor_attempt_id"])) is None
        assert not lane0._retained_recovery_dispatch_fence_is_current(
            p034,
            allowed_states=frozenset({"admitted"}),
        )

        exact = {
            str(pin["predecessor_attempt_id"]): by_lane[
                lane0._task_home_shard_index(str(pin["task_alias"]))
            ]
            for pin in pins
        }
        missing = dict(exact)
        missing.pop(str(p034["predecessor_attempt_id"]))
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="population is not exact",
        ):
            lane0.bind_retained_recovery_attempt_authorities(missing)
        assert lane0._retained_recovery_attempt_authorities is None

        mismatched = dict(exact)
        mismatched[str(p034["predecessor_attempt_id"])] = lane0
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="binding drifted",
        ):
            lane0.bind_retained_recovery_attempt_authorities(mismatched)
        assert lane0._retained_recovery_attempt_authorities is None

        lane0.bind_retained_recovery_attempt_authorities(exact)
        assert set(lane0._retained_recovery_attempt_authorities or {}) == set(
            exact
        )
        for pin in pins:
            assert lane0._retained_recovery_dispatch_fence_is_current(
                pin,
                allowed_states=frozenset({"admitted"}),
            )
        assert (
            lane0._retained_recovery_attempt_authority(p034) is lane3
        )
    finally:
        lane3.close()
        lane0.close()


def test_retained_attempt_binding_rejects_foreign_home_lane_alias(
    tmp_path: Path,
) -> None:
    pins = _pins()
    lane0_pins = [pin for pin in pins if pin["task_alias"] != "PCTDD-034"]
    lane0 = _open_lane(
        tmp_path,
        lane=0,
        owner=str(lane0_pins[0]["predecessor_owner_session_id"]),
    )
    try:
        for pin in lane0_pins:
            _seed_predecessor(lane0, pin)
        aliases = {
            str(pin["predecessor_attempt_id"]): lane0 for pin in pins
        }
        with pytest.raises(DatabaseImplementationAuthorityError):
            lane0.bind_retained_recovery_attempt_authorities(aliases)
        assert lane0._retained_recovery_attempt_authorities is None
        for pin in lane0_pins:
            assert lane0._retained_recovery_dispatch_fence_is_current(
                pin,
                allowed_states=frozenset({"admitted"}),
            )
    finally:
        lane0.close()


def test_historical_claim_uses_canonical_population_and_only_own_lane_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _historical_admission(monkeypatch, task_alias=task_alias)
        for task_alias in ("PCTDD-006", "PCTDD-007", "PCTDD-034")
    ]
    occurrences = {
        occurrence["task_alias"]: occurrence
        for occurrence, _admission_record in records
    }
    admissions = {
        occurrence["task_alias"]: admission
        for occurrence, admission in records
    }
    lane0 = _open_lane(
        tmp_path,
        lane=0,
        owner=str(
            occurrences["PCTDD-006"]["predecessor_owner_session_id"]
        ),
    )
    try:
        lane0.materialize_population(_historical_population(records))
        for task_alias in ("PCTDD-006", "PCTDD-007"):
            _seed_predecessor(lane0, occurrences[task_alias])
        for occurrence, admission in records:
            _arm_retrying_task(lane0, occurrence, admission)

        target = occurrences["PCTDD-006"]
        admission = admissions["PCTDD-006"]
        assert lane0._retained_recovery_attempt_authorities is None
        assert not (tmp_path / "lane-3").exists()
        assert lane0._retained_recovery_admission_population_admitted_current(
            admission
        )

        original_source = lane0._task_source

        class MissingPeerSource:
            def get(self, task_cid: str):
                if task_cid == occurrences["PCTDD-034"]["task_cid"]:
                    return None
                return original_source.get(task_cid)

        lane0._task_source = MissingPeerSource()
        assert not (
            lane0._retained_recovery_admission_population_admitted_current(
                admission
            )
        )
        lane0._task_source = original_source

        attempt = lane0.claim_next(
            exclude_task_cids={occurrences["PCTDD-007"]["task_cid"]}
        )
        assert attempt is not None
        assert attempt.task_cid == target["task_cid"]
        current = lane0.task_source.get(target["task_cid"])
        assert current is not None and current.status == "in_progress"
        consumption = current.body["completion_receipt"][
            "retained_recovery_consumption"
        ]
        assert (
            consumption["schema"]
            == DATABASE_FENCED_PROVIDER_HISTORICAL_RETAINED_CONSUMPTION_SCHEMA
        )
        assert database_fenced_provider_historical_retained_consumption_valid(
            consumption
        )
        assert lane0._retained_recovery_attempt_authorities is None
        assert not (tmp_path / "lane-3").exists()
    finally:
        lane0.close()


def test_supervisor_cross_lane_open_requires_locks_and_existing_peer(
    tmp_path: Path,
) -> None:
    pins = _pins()
    lane0_owner = next(
        str(pin["predecessor_owner_session_id"])
        for pin in pins
        if pin["task_alias"] == "PCTDD-005"
    )
    state_parent = tmp_path / "state"
    lane0 = _open_lane(state_parent, lane=0, owner=lane0_owner)
    supervisor = object.__new__(PortalImplementationSupervisor)
    supervisor.config = SimpleNamespace(
        repo_root=tmp_path,
        state_dir=state_parent / "lane-0",
        max_task_attempts=3,
        implement=True,
        task_prefix="## PCTDD-",
    )
    program = SimpleNamespace(
        authority_mode="quack",
        task_source_kind="duckdb",
        quack_endpoint="quack:127.0.0.1:1",
        store_id=_CONTROL_STORE_ID,
        store_generation=_CONTROL_STORE_GENERATION,
    )
    bind = supervisor._bind_retained_recovery_lane_attempt_authorities
    try:
        with pytest.raises(RuntimeError, match="owner/launch fences"):
            bind(
                daemon=lane0,
                program=program,
                task_source=lane0.task_source,
                shard_count=4,
                shard_index=0,
                strict_sharding=True,
                owner_fence_held=False,
                managed_daemon_launch_lock_held=True,
            )
        with pytest.raises(RuntimeError, match="peer lane is unavailable"):
            bind(
                daemon=lane0,
                program=program,
                task_source=lane0.task_source,
                shard_count=4,
                shard_index=0,
                strict_sharding=True,
                owner_fence_held=True,
                managed_daemon_launch_lock_held=True,
            )
        assert lane0._retained_recovery_attempt_authorities is None
        assert not (state_parent / "lane-3").exists()
    finally:
        lane0.close()
