from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("duckdb")

from ipfs_accelerate_py.agent_supervisor.authorization_logic import (
    ControlMutationPolicy,
)
from ipfs_accelerate_py.agent_supervisor.checkout_lock import checkout_repository_id
from ipfs_accelerate_py.agent_supervisor.control_contracts import (
    AuthorizationDecision,
    AuthorizationVerdict,
    IdempotencyKey,
    Operation,
    OperationAuthority,
    OperationRequest,
)
from ipfs_accelerate_py.agent_supervisor.duckdb_retry_reset import (
    RETRY_RESET_GRANT,
    RETRY_RESET_OWNER_FILE,
    RETRY_RESET_POLICY_SCHEMA,
    DuckDBRetryResetAuthorizationError,
    DuckDBRetryResetConflict,
    DuckDBRetryResetCorruptionError,
    DuckDBRetryResetQuiescenceError,
    LaneBinding,
    RetryResetOwnerConfig,
    execute_duckdb_retry_reset,
    inspect_incomplete_retry_resets,
    main,
    retry_reset_expected_effect,
)
from ipfs_accelerate_py.agent_supervisor.duckdb_task_source import DuckDBTaskSource

NOW = time.time_ns() // 1_000_000


def _source(repository_tree_id: str) -> dict[str, object]:
    return {
        "schema": "fixture/formal-plan-input@1",
        "repository_tree_id": repository_tree_id,
        "objectives": [
            {
                "goal_id": "G1",
                "goal_cid": "goal:datasets",
                "owner_actor_id": "owner:supervisor",
                "title": "Datasets retry reset",
                "acceptance_criteria": ["The governed reset is replayable."],
            }
        ],
        "taskboard": [
            {
                "task_id": "DQK-007",
                "task_cid": "task:cid:dqk-007",
                "goal_id": "G1",
                "actor_id": "agent:alpha",
                "acceptance_criteria": ["Focused tests pass."],
                "validation_commands": ["pytest -q test_reset.py"],
            },
            {
                "task_id": "DQK-008",
                "task_cid": "task:cid:dqk-008",
                "goal_id": "G1",
                "depends_on": ["DQK-007"],
                "actor_id": "agent:beta",
                "acceptance_criteria": ["Descendant stays valid."],
                "validation_commands": ["pytest -q test_descendant.py"],
            },
        ],
        "proof_policy": {
            "policy_cid": "policy:cid:datasets-reset",
            "minimum_code_assurance": "candidate",
            "freshness_seconds": 3600,
            "fallback_check_ids": ["fallback:pytest"],
        },
    }


def _queue_payload(
    task_cid: str,
    task_alias: str,
    *,
    attempt_count: int,
    schema: str = "persistent_task_queue_v2",
) -> dict[str, Any]:
    entry = {
        "task_id": task_alias,
        "priority": "P1",
        "track": "datasets",
        "canonical_task_cid": task_cid,
        "canonical_task_key": "task:key:dqk-007",
        "aliases": [task_alias],
        "provenance": [],
        "selection_penalty": 900,
        "attempt_count": attempt_count,
        "last_selected_at": 3.0,
        "last_completed_at": 2.0,
        "consecutive_failures": 4,
        "consecutive_no_change": 3,
        "merge_failure_count": 2,
        "cooldown_until": 999_999.0,
        "notes": "validation failed",
    }
    if schema == "persistent_task_queue_v3":
        entry.update(
            {
                "authority_renewal_key": "renewal:bound",
                "authority_renewal_failure_count": 2,
                "authority_renewal_last_failure_at": 40.0,
                "authority_renewal_cooldown_until": 400.0,
                "authority_renewal_quarantined": True,
                "authority_renewal_reason": "authority unavailable",
            }
        )
    return {
        "schema": schema,
        "updated_at": 1.0,
        "entry_count": 1,
        "entries": {task_cid: entry},
        "aliases": {task_alias: task_cid},
    }


def _write_lane(
    state_root: Path,
    lane: str,
    *,
    task_cid: str = "task:cid:dqk-007",
    task_alias: str = "DQK-007",
    attempt_count: int = 7,
    queue_schema: str = "persistent_task_queue_v2",
) -> dict[str, str]:
    lane_root = state_root / lane
    lane_root.mkdir(parents=True)
    state_path = lane_root / f"{lane}_task_state.json"
    queue_path = lane_root / "task_queue.json"
    state_path.write_text(
        json.dumps(
            {
                "implementation_in_progress": False,
                "active_task_id": "",
                "active_task_cid": "",
                "task_identities": {task_alias: {"canonical_task_cid": task_cid}},
                "implementation_attempts": {task_alias: attempt_count},
                "implementation_attempts_by_cid": {task_cid: attempt_count + 1},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    queue_path.write_text(
        json.dumps(
            _queue_payload(
                task_cid,
                task_alias,
                attempt_count=attempt_count,
                schema=queue_schema,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "state_prefix": lane,
        "state_path": state_path.relative_to(state_root).as_posix(),
        "queue_path": queue_path.relative_to(state_root).as_posix(),
    }


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _policy_payload(policy: ControlMutationPolicy) -> dict[str, Any]:
    return {
        "schema": RETRY_RESET_POLICY_SCHEMA,
        "policy_id": policy.policy_id,
        "policy_revision": policy.policy_revision,
        "permits": [item.to_record() for item in policy.permits],
        "current_tree_ids": dict(policy.current_tree_ids),
        "current_objective_revisions": dict(policy.current_objective_revisions),
        "active_lease_fences": dict(policy.active_lease_fences),
    }


def _fixture(
    tmp_path: Path,
    *,
    lanes: int = 2,
    expected_status: str = "completed",
    queue_schema: str = "persistent_task_queue_v2",
) -> tuple[
    DuckDBTaskSource,
    OperationRequest,
    ControlMutationPolicy,
    RetryResetOwnerConfig,
    Path,
]:
    repository = (tmp_path / "repository").resolve()
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Retry Reset Test")
    _git(repository, "config", "user.email", "retry-reset@example.invalid")
    (repository / "plan-root.txt").write_text("admitted plan\n", encoding="utf-8")
    _git(repository, "add", "plan-root.txt")
    _git(repository, "commit", "-q", "-m", "admit plan")
    task_source_tree = _git(repository, "rev-parse", "HEAD^{tree}")

    state_root = (tmp_path / "state").resolve()
    state_root.mkdir()
    source = DuckDBTaskSource(state_root / "workflow.duckdb")
    source.materialize(_source(task_source_tree))
    task = source.get_task("DQK-007")
    assert task is not None
    selected = task
    if task.status != expected_status:
        selected = source.compare_and_set_status(
            "DQK-007", task.revision, expected_status
        ).task
    lane_bindings = [
        _write_lane(
            state_root,
            f"lane{index}",
            queue_schema=queue_schema,
        )
        for index in range(lanes)
    ]

    # Model an admitted implementation merge after immutable task-source
    # materialization.  Reset must bind this current HEAD independently.
    (repository / "implementation.txt").write_text("merged work\n", encoding="utf-8")
    _git(repository, "add", "implementation.txt")
    _git(repository, "commit", "-q", "-m", "merge implementation")
    head_commit = _git(repository, "rev-parse", "HEAD")
    head_tree = _git(repository, "rev-parse", "HEAD^{tree}")

    parameters = {
        "task_source_kind": "duckdb",
        "database_path": "workflow.duckdb",
        "plan_root_cid": source.snapshot().plan_root_cid,
        "task_source_repository_tree_id": task_source_tree,
        "repository_head_commit": head_commit,
        "task_cid": selected.task_cid,
        "task_alias": selected.task_alias,
        "task_revision": selected.revision,
        "expected_status": expected_status,
        "reopen_status": "retrying",
        "writer_id": "local",
        "writer_fencing_token": 1,
        "lanes": lane_bindings,
        "lifecycle_owner_paths": ["multi_supervisor_runner.pid"],
    }
    repository_root = str(repository)
    repository_id = checkout_repository_id(repository)
    effect = retry_reset_expected_effect(
        repository_root=repository_root,
        state_root=str(state_root),
        repository_id=repository_id,
        tree_id=head_tree,
        parameters=parameters,
    )
    common = {
        "operation": Operation.RETRY,
        "repository_root": repository_root,
        "state_root": str(state_root),
        "repository_id": repository_id,
        "tree_id": head_tree,
        "objective_id": "goal:datasets",
        "objective_revision": "goal-revision:1",
        "policy_id": "policy:datasets-reset",
        "policy_revision": "policy-revision:1",
        "caller": "operator:datasets",
    }
    decision = AuthorizationDecision(
        verdict=AuthorizationVerdict.PERMIT,
        granted_authority=OperationAuthority.MUTATION,
        authorized_effect_ids=(effect.effect_id,),
        grant_ids=(RETRY_RESET_GRANT, "grant:duckdb-writer:local"),
        lease_id="lease:datasets-reset",
        fencing_epoch=1,
        evaluated_at_ms=NOW - 100,
        expires_at_ms=NOW + 600_000,
        **common,
    )
    request = OperationRequest(
        expected_effects=(effect,),
        parameters=parameters,
        idempotency=IdempotencyKey(
            key=f"retry-reset:dqk-007:revision-{selected.revision}",
            operation=Operation.RETRY,
            caller=common["caller"],
            repository_id=repository_id,
            objective_id=common["objective_id"],
        ),
        authorization=decision,
        lease_id="lease:datasets-reset",
        fencing_epoch=1,
        **common,
    )
    policy = ControlMutationPolicy(
        policy_id=common["policy_id"],
        policy_revision=common["policy_revision"],
        permits=(decision,),
        current_tree_ids={repository_id: head_tree},
        current_objective_revisions={"goal:datasets": "goal-revision:1"},
        active_lease_fences={"lease:datasets-reset": 1},
    )
    policy_path = state_root / "control" / "retry-reset-policy.json"
    policy_path.parent.mkdir()
    policy_bytes = (
        json.dumps(_policy_payload(policy), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    policy_path.write_bytes(policy_bytes)
    owner = RetryResetOwnerConfig(
        repository_root=repository_root,
        repository_id=repository_id,
        database_path="workflow.duckdb",
        task_source_repository_tree_id=task_source_tree,
        policy_path=policy_path.relative_to(state_root).as_posix(),
        policy_digest="sha256:" + hashlib.sha256(policy_bytes).hexdigest(),
        lanes=tuple(
            LaneBinding(item["state_prefix"], item["state_path"], item["queue_path"])
            for item in lane_bindings
        ),
        lifecycle_owner_paths=("multi_supervisor_runner.pid",),
    )
    owner_path = state_root / RETRY_RESET_OWNER_FILE
    owner_path.write_text(
        json.dumps(owner.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    owner_path.chmod(0o600)
    return source, request, policy, owner, state_root


def _execute(
    request: OperationRequest,
    policy: ControlMutationPolicy,
    owner: RetryResetOwnerConfig,
) -> dict[str, Any]:
    return execute_duckdb_retry_reset(
        request,
        trusted_policy=policy,
        trusted_owner=owner,
        clock_ms=lambda: NOW,
    )


def test_cross_lane_reset_is_content_bound_and_idempotent(tmp_path: Path) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path)

    receipt = _execute(request, policy, owner)
    replay = _execute(request, policy, owner)

    assert replay == receipt
    task = source.get_task("task:cid:dqk-007")
    assert task is not None and task.status == "retrying"
    assert task.revision == request.parameters["task_revision"] + 1
    assert receipt["plan_root_cid"] == source.snapshot().plan_root_cid
    assert (
        receipt["task_source_repository_tree_id"]
        == source.snapshot().repository_tree_id
        != receipt["repository_tree_id"]
    )
    assert (
        receipt["repository_head_commit"]
        == request.parameters["repository_head_commit"]
    )
    assert receipt["writer_id"] == "local"
    assert receipt["writer_fencing_token"] == 1
    assert len(receipt["lanes"]) == 2
    assert not inspect_incomplete_retry_resets(state_root)
    for lane in receipt["lanes"]:
        assert lane["matched"]
        assert lane["display_attempt_count_before"] == 7
        assert lane["display_attempt_count_after"] == 0
        assert lane["canonical_attempt_count_before"] == 8
        assert lane["canonical_attempt_count_after"] == 0
        assert lane["state_digest_before"] != lane["state_digest_after"]
        assert lane["queue_digest_before"] != lane["queue_digest_after"]
        assert lane["queue_entries_before"][0]["attempt_count"] == 7
        assert lane["queue_entries_after"][0]["attempt_count"] == 7
        assert all(
            value in {0, ""}
            for value in lane["queue_entries_after"][0]["retry"].values()
        )
    events = source.events(cursor=0, limit=20).events
    assert sum(event["event_type"] == "retry_reset_completed" for event in events) == 1


def test_v3_queue_preserves_authority_history_while_clearing_retry_penalties(
    tmp_path: Path,
) -> None:
    _source_db, request, policy, owner, state_root = _fixture(
        tmp_path, lanes=1, queue_schema="persistent_task_queue_v3"
    )

    _execute(request, policy, owner)

    queue = json.loads((state_root / "lane0/task_queue.json").read_text())
    entry = queue["entries"][request.parameters["task_cid"]]
    assert queue["schema"] == "persistent_task_queue_v3"
    assert entry["attempt_count"] == 7
    assert entry["selection_penalty"] == 0
    assert entry["cooldown_until"] == 0
    assert entry["authority_renewal_key"] == "renewal:bound"
    assert entry["authority_renewal_failure_count"] == 2
    assert entry["authority_renewal_quarantined"] is True


@pytest.mark.parametrize(
    "expected_status,status_changed", [("pending", True), ("retrying", False)]
)
def test_exhausted_nonterminal_task_is_reset_without_spurious_same_status_cas(
    tmp_path: Path, expected_status: str, status_changed: bool
) -> None:
    source, request, policy, owner, _state_root = _fixture(
        tmp_path, lanes=1, expected_status=expected_status
    )
    before_revision = source.get_task("DQK-007").revision  # type: ignore[union-attr]

    receipt = _execute(request, policy, owner)

    after = source.get_task("DQK-007")
    assert after is not None and after.status == "retrying"
    assert receipt["status_changed"] is status_changed
    assert after.revision == before_revision + int(status_changed)
    intent_events = [
        event
        for event in source.events(cursor=0, limit=50).events
        if event["event_type"] == "retry_reset_intent"
    ]
    assert len(intent_events) == int(not status_changed)


def test_owner_topology_must_declare_every_lane(tmp_path: Path) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    extra = _write_lane(state_root, "lane-extra")
    incomplete_owner = replace(
        owner,
        lanes=(
            *owner.lanes,
            LaneBinding(
                extra["state_prefix"], extra["state_path"], extra["queue_path"]
            ),
        ),
    )

    with pytest.raises(DuckDBRetryResetAuthorizationError, match="complete owner"):
        _execute(request, policy, incomplete_owner)

    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]


def test_undeclared_physical_lane_containing_task_fails_closed(tmp_path: Path) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    _write_lane(state_root, "lane-extra")

    with pytest.raises(
        DuckDBRetryResetAuthorizationError, match="absent from owner topology"
    ):
        _execute(request, policy, owner)

    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]


def test_symlink_and_nonfinite_sidecars_fail_closed(tmp_path: Path) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    queue_path = state_root / "lane0/task_queue.json"
    real_queue = state_root / "queue-outside.json"
    queue_path.rename(real_queue)
    queue_path.symlink_to(real_queue)

    with pytest.raises(DuckDBRetryResetCorruptionError, match="symlink"):
        _execute(request, policy, owner)
    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]

    queue_path.unlink()
    real_queue.replace(queue_path)
    payload = json.loads(queue_path.read_text())
    payload["entries"][request.parameters["task_cid"]]["cooldown_until"] = float("nan")
    queue_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(DuckDBRetryResetCorruptionError, match="malformed"):
        _execute(request, policy, owner)


def test_active_alias_refuses_even_when_cid_identity_is_missing(tmp_path: Path) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    state_path = state_root / "lane0/lane0_task_state.json"
    state = json.loads(state_path.read_text())
    state["active_task_id"] = "DQK-007"
    state["active_task_cid"] = ""
    state["task_identities"] = {}
    state["implementation_attempts_by_cid"] = {}
    state_path.write_text(json.dumps(state) + "\n", encoding="utf-8")

    with pytest.raises(DuckDBRetryResetQuiescenceError, match="active"):
        _execute(request, policy, owner)
    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]


def test_crash_after_database_cas_has_visible_idempotent_recovery(
    tmp_path: Path,
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)

    def crash(phase: str) -> None:
        if phase == "database_mutated":
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        execute_duckdb_retry_reset(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            clock_ms=lambda: NOW,
            fault_injector=crash,
        )

    incomplete = inspect_incomplete_retry_resets(state_root)
    assert len(incomplete) == 1
    assert incomplete[0]["phase"] == "prepared"
    assert source.get_task("DQK-007").status == "retrying"  # type: ignore[union-attr]
    lane_state = json.loads((state_root / "lane0/lane0_task_state.json").read_text())
    assert lane_state["implementation_attempts"]["DQK-007"] == 7

    receipt = _execute(request, policy, owner)

    assert receipt["status_after"] == "retrying"
    assert not inspect_incomplete_retry_resets(state_root)
    events = source.events(cursor=0, limit=20).events
    assert sum(event["event_type"] == "status_changed" for event in events) == 2
    assert sum(event["event_type"] == "retry_reset_completed" for event in events) == 1


@pytest.mark.parametrize(
    ("crash_window", "journal_phase"),
    [
        ("lane_sidecar_mutated:lane0", "database_committed"),
        ("receipt_written", "sidecars_committed"),
        ("completion_event_appended", "receipt_committed"),
    ],
)
def test_nontransactional_crash_windows_replay_idempotently(
    tmp_path: Path,
    crash_window: str,
    journal_phase: str,
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=2)

    def crash(phase: str) -> None:
        if phase == crash_window:
            raise RuntimeError("crash window")

    with pytest.raises(RuntimeError, match="crash window"):
        execute_duckdb_retry_reset(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            clock_ms=lambda: NOW,
            fault_injector=crash,
        )

    incomplete = inspect_incomplete_retry_resets(state_root)
    assert len(incomplete) == 1
    assert incomplete[0]["phase"] == journal_phase

    receipt = _execute(request, policy, owner)

    assert receipt["status_after"] == "retrying"
    assert not inspect_incomplete_retry_resets(state_root)
    events = source.events(cursor=0, limit=50).events
    assert sum(event["event_type"] == "retry_reset_completed" for event in events) == 1


def test_structurally_valid_self_issued_permit_is_not_trusted(tmp_path: Path) -> None:
    source, request, policy, owner, _state_root = _fixture(tmp_path, lanes=1)
    assert request.authorization is not None
    forged = replace(
        request.authorization,
        grant_ids=(*request.authorization.grant_ids, "grant:self-issued"),
    )
    counterfeit = replace(request, authorization=forged)

    with pytest.raises(
        DuckDBRetryResetAuthorizationError,
        match="not issued by the current policy",
    ):
        _execute(counterfeit, policy, owner)

    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]


def test_live_master_refuses_reset_even_when_lane_pids_are_absent(
    tmp_path: Path,
) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    (state_root / "multi_supervisor_runner.pid").write_text(f"{os.getpid()}\n")

    with pytest.raises(
        DuckDBRetryResetQuiescenceError, match="lifecycle owner is live"
    ):
        _execute(request, policy, owner)

    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]
    assert not inspect_incomplete_retry_resets(state_root)


def test_malformed_queue_fails_before_database_cas(tmp_path: Path) -> None:
    source, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    (state_root / "lane0/task_queue.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(DuckDBRetryResetCorruptionError, match="queue"):
        _execute(request, policy, owner)

    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]


def test_completed_descendant_blocks_non_cascading_reopen(tmp_path: Path) -> None:
    source, request, policy, owner, _state_root = _fixture(tmp_path, lanes=1)
    descendant = source.get_task("DQK-008")
    assert descendant is not None
    source.compare_and_set_status("DQK-008", descendant.revision, "completed")

    with pytest.raises(DuckDBRetryResetConflict, match="completed descendants"):
        _execute(request, policy, owner)

    assert source.get_task("DQK-007").status == "completed"  # type: ignore[union-attr]


def test_inspect_cli_is_machine_readable_and_nonzero_for_prepared_intent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _source_db, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)

    def crash(phase: str) -> None:
        if phase == "prepared":
            raise RuntimeError("prepared")

    with pytest.raises(RuntimeError, match="prepared"):
        execute_duckdb_retry_reset(
            request,
            trusted_policy=policy,
            trusted_owner=owner,
            clock_ms=lambda: NOW,
            fault_injector=crash,
        )

    assert main(["--inspect-state-root", str(state_root)]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["incomplete"][0]["phase"] == "prepared"


def test_inspect_verifies_completed_journal_receipt_and_filename(
    tmp_path: Path,
) -> None:
    _source_db, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    receipt = _execute(request, policy, owner)
    assert not inspect_incomplete_retry_resets(state_root)

    receipt_path = (
        Path(receipt["journal_path"]).parents[1]
        / "receipts"
        / (receipt["receipt_cid"] + ".json")
    )
    receipt_payload = json.loads(receipt_path.read_text())
    receipt_payload["writer_id"] = "counterfeit"
    receipt_path.write_text(json.dumps(receipt_payload) + "\n", encoding="utf-8")
    with pytest.raises(DuckDBRetryResetCorruptionError, match="identity"):
        inspect_incomplete_retry_resets(state_root)

    # Restore exact receipt and prove that a renamed journal is rejected too.
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    journal_path = Path(receipt["journal_path"])
    renamed = journal_path.with_name("counterfeit.json")
    journal_path.rename(renamed)
    with pytest.raises(DuckDBRetryResetCorruptionError, match="filename"):
        inspect_incomplete_retry_resets(state_root)


def test_execute_cli_loads_pinned_owner_policy_and_rejects_counterfeit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _source_db, request, policy, owner, state_root = _fixture(tmp_path, lanes=1)
    request_path = tmp_path / "retry-request.json"
    request_path.write_text(request.to_json() + "\n", encoding="utf-8")

    assert main(["--request-file", str(request_path)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["schema"].endswith("duckdb-retry-reset-receipt@1")
    assert receipt["request_id"] == request.request_id

    counterfeit = _policy_payload(policy)
    counterfeit["policy_revision"] = "policy-revision:counterfeit"
    (state_root / owner.policy_path).write_text(
        json.dumps(counterfeit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert main(["--request-file", str(request_path)]) == 2
    failure = json.loads(capsys.readouterr().out)
    assert "digest does not match" in failure["message"]


def test_cli_rejects_symlinked_owner_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _source_db, request, _policy, owner, state_root = _fixture(tmp_path, lanes=1)
    request_path = tmp_path / "retry-request.json"
    request_path.write_text(request.to_json() + "\n", encoding="utf-8")
    configured = state_root / owner.policy_path
    outside = tmp_path / "policy-outside.json"
    configured.rename(outside)
    configured.symlink_to(outside)

    assert main(["--request-file", str(request_path)]) == 2
    failure = json.loads(capsys.readouterr().out)
    assert "symlink" in failure["message"]
