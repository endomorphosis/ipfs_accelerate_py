"""Fail-closed recovery tests for exhausted typed Portal deferrals."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_implementation_route import (
    build_agent_implementation_failure_receipt,
    resolve_agent_implementation_route,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_failure_policy import (
    build_grok_route_outcome,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    database_task_source as database_task_source_module,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
    TaskRecord,
    TaskSourceConflictError,
    TYPED_DEFERRAL_BUDGET_SUPERSESSION_OPERATION,
    TypedDeferralRecoveryError,
    execute_quack_owner_command,
    typed_deferral_budget_supersession_matches,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET,
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentRepository,
)


SOURCE_HEAD = "1" * 40
SOURCE_TREE = "2" * 40
REPAIR_HEAD = "3" * 40
REPAIR_TREE = "4" * 40
ADMISSION_NOW_MS = 3_000_000
OWNER_REQUEST_ID = "9" * 32
OWNER_STORE_ID = "store:typed-deferral-recovery"
OWNER_STORE_GENERATION = "generation:typed-deferral-recovery"


def _sha256_identity(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _materialize(
    source: DatabaseTaskSource,
    *,
    alias: str = "VRIF-029",
    source_head: str = SOURCE_HEAD,
    source_tree: str = SOURCE_TREE,
) -> TaskRecord:
    source.materialize(
        {
            "repository_tree_id": "tree:typed-deferral-recovery",
            "objectives": [
                {
                    "goal_id": "G1",
                    "goal_cid": "goal:typed-deferral-recovery",
                    "objective_id": "objective:typed-deferral-recovery",
                    "title": "Typed deferral recovery",
                }
            ],
            "taskboard": [
                {
                    "task_id": alias,
                    "task_cid": "task:typed-deferral-recovery",
                    "goal_cid": "goal:typed-deferral-recovery",
                    "status": "ready",
                    "base_revision": source_head,
                    "base_repository_tree_id": source_tree,
                }
            ],
        }
    )
    task = source.get_task("task:typed-deferral-recovery")
    assert task is not None
    return task


def _block_exhausted(
    source: DatabaseTaskSource,
    *,
    source_head: str = SOURCE_HEAD,
    source_tree: str = SOURCE_TREE,
) -> tuple[TaskRecord, dict[str, object]]:
    task = _materialize(
        source,
        source_head=source_head,
        source_tree=source_tree,
    )
    running = source.compare_and_set_status(
        task.task_cid,
        task.revision,
        "in_progress",
        {"operation": "test_claim"},
    ).task
    budget_body: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-typed-deferral-budget@1"
        ),
        "task_cid": running.task_cid,
        "task_generation": running.task_cid,
        "generation_fingerprint": "sha256:" + ("5" * 64),
        "current_deferral_fingerprint": "sha256:" + ("6" * 64),
        "typed_deferral_candidate_count": 4,
        "typed_deferral_count": 4,
        "typed_deferral_count_is_lower_bound": False,
        "verified_typed_deferral_count": 4,
        "verified_count_complete": True,
        "max_task_attempts": 4,
        "exhausted": True,
        "attempt_consumed": False,
        "typed_deferral_slot_consumed": True,
        "matching_attempts": [],
        "matching_attempts_digest": "sha256:" + ("7" * 64),
        "matching_attempts_truncated": False,
        "omitted_matching_attempt_count": 0,
    }
    budget = {
        **budget_body,
        "observation_id": _sha256_identity(budget_body),
    }
    block_receipt = {
        "operation": "database_portal_typed_deferral_budget_exhausted",
        "attempt_id": "attempt:typed-deferral-recovery",
        "attempt_number": 4,
        "claim_id": "claim:typed-deferral-recovery",
        "lease_id": "lease:typed-deferral-recovery",
        "owner_session_id": "session:typed-deferral-recovery",
        "fencing_token": 4,
        "fence_epoch": 4,
        "execution_phase": "failed",
        "execution_revision": 3,
        "execution_finished_at_ms": 2_000_000,
        "reason": "typed_portal_deferral_budget_exhausted",
        "retryable": False,
        "attempt_consumed": False,
        "typed_deferral_slot_consumed": True,
        "retry_budget": budget,
        "prior_queue_entry_preserved_inactive": True,
        "coordination": {},
        "control_expected_status": "in_progress",
        "control_expected_revision": running.revision,
    }
    blocked = source.compare_and_set_status(
        running.task_cid,
        running.revision,
        "blocked",
        block_receipt,
    ).task
    assert blocked.status == "blocked"
    return blocked, budget


def _positive_provider_pair(
    *, observed_at_ms: int = ADMISSION_NOW_MS - 1_000
) -> tuple[dict[str, object], dict[str, object]]:
    failure = build_agent_implementation_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce="a" * 64,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
        observed_at_ms=observed_at_ms,
    )
    legacy = resolve_agent_implementation_route(default_route="legacy")
    high = resolve_agent_implementation_route(
        **{**legacy.as_dict(), "fallback_reasoning_effort": "high"}
    )
    outcome = build_grok_route_outcome(
        receipt=failure,
        route_plan=high.as_outcome_dict(),
        quota_evidence_id="sha256:" + ("b" * 64),
        decision="fallback_succeeded",
        verifier_status="confirmed_quota",
        fallback_dispatched=True,
        fallback_returncode=0,
    )
    return failure, outcome


def _owner_request_for_pair(
    task: TaskRecord,
    failure: dict[str, object],
    outcome: dict[str, object],
    *,
    repair_head: str = REPAIR_HEAD,
    repair_tree: str = REPAIR_TREE,
    admitted_at_ms: int = ADMISSION_NOW_MS,
    sentinel: object | None = None,
) -> dict[str, object]:
    return database_task_source_module._build_owner_typed_deferral_budget_supersession_request(
        task_cid=task.task_cid,
        task_revision=task.revision,
        task_body=task.body,
        repair_head=repair_head,
        repair_tree=repair_tree,
        quota_probe_receipt=failure,
        route_outcome=outcome,
        owner_command_request_id=OWNER_REQUEST_ID,
        owner_store_id=OWNER_STORE_ID,
        owner_store_generation=OWNER_STORE_GENERATION,
        admitted_at_ms=admitted_at_ms,
        _owner_admission_sentinel=(
            database_task_source_module._TYPED_DEFERRAL_PROVIDER_EVIDENCE_OWNER_SENTINEL
            if sentinel is None
            else sentinel
        ),
    )


def _request(task: TaskRecord) -> dict[str, object]:
    failure, outcome = _positive_provider_pair()
    return _owner_request_for_pair(task, failure, outcome)


def _attempt_from_receipt(receipt: dict[str, object]) -> dict[str, object]:
    return {
        "task_cid": receipt["task_cid"],
        "attempt_id": receipt["attempt_id"],
        "attempt_number": receipt["attempt_number"],
        "claim_id": receipt["claim_id"],
        "lease_id": receipt["lease_id"],
        "owner_session_id": receipt["owner_session_id"],
        "fencing_token": receipt["fencing_token"],
        "fence_epoch": receipt["fence_epoch"],
    }


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _load_vrif_operator() -> object:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts/run_agent_supervisor_residual_intelligence.py"
    spec = importlib.util.spec_from_file_location("vrif_recovery_operator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quack_owner_recovery_round_trip_runs_provider_boundary_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _load_vrif_operator()
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    _git(repository_root, "init", "-q", "-b", "main")
    _git(repository_root, "config", "user.name", "VRIF recovery test")
    _git(repository_root, "config", "user.email", "vrif@example.invalid")
    recovery_path = (
        repository_root
        / "ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py"
    )
    recovery_path.parent.mkdir(parents=True)
    recovery_path.write_text("# source generation\n", encoding="utf-8")
    _git(repository_root, "add", recovery_path.relative_to(repository_root).as_posix())
    _git(repository_root, "commit", "-q", "-m", "source generation")
    source_head = _git(repository_root, "rev-parse", "HEAD")
    source_tree = _git(repository_root, "rev-parse", "HEAD^{tree}")
    recovery_path.write_text("# admitted repair generation\n", encoding="utf-8")
    _git(repository_root, "add", recovery_path.relative_to(repository_root).as_posix())
    _git(repository_root, "commit", "-q", "-m", "repair generation")
    repair_head = _git(repository_root, "rev-parse", "HEAD")
    repair_tree = _git(repository_root, "rev-parse", "HEAD^{tree}")

    database_path = tmp_path / "control.duckdb"
    with DatabaseTaskSource(database_path) as source:
        blocked, _budget = _block_exhausted(
            source,
            source_head=source_head,
            source_tree=source_tree,
        )

    inbox = tmp_path / "mutations"
    token = "typed_recovery_test_token_0123456789"
    store_id = "data/control.duckdb"
    store_generation = "typed-recovery-generation-1"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", token)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", store_id)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION",
        store_generation,
    )
    monkeypatch.setattr(operator, "ROOT", repository_root)
    config = {
        "merge_target_branch": "main",
        "source_binding": {"accelerator_required_ancestor": source_head},
    }
    failure, outcome = _positive_provider_pair(
        observed_at_ms=int(time.time() * 1000) - 1_000
    )
    canary_calls: list[bool] = []

    def provider_canary(**_kwargs: object) -> dict[str, dict[str, object]]:
        canary_calls.append(True)
        return {
            "quota_probe_receipt": failure,
            "route_outcome": outcome,
        }

    monkeypatch.setattr(
        operator,
        "_run_typed_deferral_provider_canary",
        provider_canary,
    )
    observed_payloads: list[dict[str, object]] = []
    owner_failures: list[BaseException] = []

    def owner() -> None:
        connection = None
        repository = None
        try:
            connection = open_duckdb_connection(database_path)
            repository = IntentRepository(
                database_path,
                bound_connection=connection,
                install_schema=False,
                owner_id="owner:typed-recovery-integration",
                session_id="session:typed-recovery-integration",
            )
            deadline = time.monotonic() + 10
            while len(observed_payloads) < 2 and time.monotonic() < deadline:
                requests = sorted(inbox.glob("*.request.json"))
                if not requests:
                    time.sleep(0.01)
                    continue
                request = json.loads(requests[0].read_text(encoding="utf-8"))
                observed_payloads.append(dict(request["payload"]))
                operator._process_owner_commands(
                    repository,
                    inbox,
                    token=token,
                    expected_store_id=store_id,
                    expected_store_generation=store_generation,
                    typed_deferral_provider_evidence_factory=(
                        lambda **context: operator._owner_typed_deferral_provider_evidence(
                            config,
                            database_program=object(),
                            **context,
                        )
                    ),
                )
            assert len(observed_payloads) == 2
        except BaseException as exc:  # surfaced on the test thread below
            owner_failures.append(exc)
        finally:
            if repository is not None:
                repository.close()
            if connection is not None:
                connection.close()

    owner_thread = threading.Thread(target=owner)
    owner_thread.start()
    client = DatabaseTaskSource("quack:127.0.0.1:45123", install_schema=False)
    try:
        first = client.recover_typed_deferral_budget(
            blocked.task_cid,
            repair_head=repair_head,
            repair_tree=repair_tree,
            timeout_seconds=5,
        )
        replay = client.recover_typed_deferral_budget(
            blocked.task_cid,
            repair_head=repair_head,
            repair_tree=repair_tree,
            timeout_seconds=5,
        )
    finally:
        client.close()
    owner_thread.join(timeout=10)

    assert not owner_thread.is_alive()
    assert not owner_failures
    assert canary_calls == [True]
    assert observed_payloads == [
        {
            "task_cid_or_alias": blocked.task_cid,
            "repair_head": repair_head,
            "repair_tree": repair_tree,
        },
        {
            "task_cid_or_alias": blocked.task_cid,
            "repair_head": repair_head,
            "repair_tree": repair_tree,
        },
    ]
    assert first.changed is True
    assert first.previous_status == "blocked"
    assert first.task.status == "retrying"
    durable = first.task.body["completion_receipt"]
    assert durable["repair_head"] == repair_head
    assert durable["repair_tree"] == repair_tree
    assert durable["provider_evidence_owner_store_id"] == store_id
    assert durable["provider_evidence_owner_store_generation"] == store_generation
    assert replay.changed is False
    assert replay.task.status == "retrying"


def test_owner_recovery_rejects_ordinary_block_before_provider_boundary(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "control.duckdb"
    with DatabaseTaskSource(database_path) as source:
        task = _materialize(source, alias="ordinary-block")
        source.compare_and_set_status(
            task.task_cid,
            task.revision,
            "blocked",
            {"operation": "dependency_wait", "blocker": "task:other"},
        )

    connection = open_duckdb_connection(database_path)
    repository = IntentRepository(
        database_path,
        bound_connection=connection,
        install_schema=False,
        owner_id="owner:ordinary-block",
        session_id="session:ordinary-block",
    )
    provider_calls: list[bool] = []
    try:
        with pytest.raises(
            TypedDeferralRecoveryError,
            match="exact exhausted typed-deferral receipt",
        ):
            execute_quack_owner_command(
                repository,
                QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET,
                {
                    "task_cid_or_alias": task.task_cid,
                    "repair_head": REPAIR_HEAD,
                    "repair_tree": REPAIR_TREE,
                },
                request_id="8" * 32,
                store_id="store:ordinary-block",
                store_generation="generation:ordinary-block",
                typed_deferral_provider_evidence_factory=(
                    lambda **_context: provider_calls.append(True)
                ),
            )
    finally:
        repository.close()
        connection.close()
    assert provider_calls == []


def test_exact_positive_route_supersedes_one_exhausted_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        blocked, budget = _block_exhausted(source)
        result = source.rearm_blocked_task(blocked.task_cid, receipt=_request(blocked))

        assert result.changed is True
        assert result.previous_status == "blocked"
        assert result.task.status == "retrying"
        durable = dict(result.task.body["completion_receipt"])
        assert durable["operation"] == TYPED_DEFERRAL_BUDGET_SUPERSESSION_OPERATION
        assert durable["task_cid"] == blocked.task_cid
        assert durable["attempt_id"] == "attempt:typed-deferral-recovery"
        assert durable["exhausted_observation_id"] == budget["observation_id"]
        assert durable["exhausted_receipt_id"] == content_identity(
            blocked.body["completion_receipt"]
        )
        assert durable["source_head"] == SOURCE_HEAD
        assert durable["source_tree"] == SOURCE_TREE
        assert durable["repair_head"] == REPAIR_HEAD
        assert durable["repair_tree"] == REPAIR_TREE
        assert durable["route_outcome"]["decision"] == "fallback_succeeded"
        assert durable["route_outcome"]["fallback_dispatched"] is True
        assert durable["provider_evidence_admitted_at_ms"] == ADMISSION_NOW_MS
        assert durable["provider_evidence_owner_command"] == (
            "recover_typed_deferral_budget"
        )
        assert durable["provider_evidence_owner_command_request_id"] == (
            OWNER_REQUEST_ID
        )
        assert durable["provider_evidence_owner_store_id"] == OWNER_STORE_ID
        assert durable["provider_evidence_owner_store_generation"] == (
            OWNER_STORE_GENERATION
        )
        assert durable["supersession_id"] == content_identity(
            {key: value for key, value in durable.items() if key != "supersession_id"}
        )

        current = {
            "task_cid": result.task.task_cid,
            "task_alias": result.task.task_alias,
            "task_revision": result.task.revision,
            "task_body": copy.deepcopy(dict(result.task.body)),
            "attempt": _attempt_from_receipt(durable),
            "exhausted_budget": budget,
        }
        # Restart reproduction uses the persisted historical admission time;
        # it does not turn a durable receipt into a live freshness lease.
        monkeypatch.setattr(
            database_task_source_module.time,
            "time",
            lambda: 99_999_999.0,
        )
        assert typed_deferral_budget_supersession_matches(durable, **current)

        wrong_attempt = copy.deepcopy(current)
        wrong_attempt["attempt"]["attempt_id"] = "attempt:foreign"
        assert not typed_deferral_budget_supersession_matches(
            durable, **wrong_attempt
        )
        wrong_budget = copy.deepcopy(current)
        wrong_budget["exhausted_budget"]["observation_id"] = (
            "sha256:" + ("0" * 64)
        )
        assert not typed_deferral_budget_supersession_matches(
            durable, **wrong_budget
        )
        wrong_source = copy.deepcopy(current)
        wrong_source["task_body"]["base_revision"] = "9" * 40
        assert not typed_deferral_budget_supersession_matches(
            durable, **wrong_source
        )
        tampered_owner_binding = copy.deepcopy(durable)
        tampered_owner_binding["provider_evidence_owner_store_generation"] = (
            "generation:foreign"
        )
        tampered_owner_binding["supersession_id"] = content_identity(
            {
                key: value
                for key, value in tampered_owner_binding.items()
                if key != "supersession_id"
            }
        )
        assert not typed_deferral_budget_supersession_matches(
            tampered_owner_binding, **current
        )


def test_generic_stale_forged_and_mismatched_rearms_remain_blocked(
    tmp_path: Path,
) -> None:
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        blocked, _budget = _block_exhausted(source)
        valid = _request(blocked)
        failure, outcome = _positive_provider_pair()
        # The old public shape is deliberately no longer an authorization:
        # both records are reproducible content hashes supplied by the caller.
        caller_supplied_pair = {
            key: value
            for key, value in valid.items()
            if key != "provider_evidence_admission"
        }
        caller_supplied_pair.update(
            {"quota_probe_receipt": failure, "route_outcome": outcome}
        )
        cases: dict[str, dict[str, object]] = {
            "generic": {"operation": "database_declared_outputs_on_head_rearm"},
            "caller_pair": caller_supplied_pair,
            "task": {**valid, "task_cid": "task:foreign"},
            "attempt": {**valid, "attempt_id": "attempt:foreign"},
            "observation": {
                **valid,
                "exhausted_observation_id": "sha256:" + ("0" * 64),
            },
            "source": {**valid, "source_head": "8" * 40},
            "repair": {**valid, "repair_tree": "not-a-tree"},
            "serialized_lookalike": {
                **valid,
                "provider_evidence_admission": {
                    "admission_id": "caller-controlled"
                },
            },
        }

        for name, candidate in cases.items():
            with pytest.raises(
                TaskSourceConflictError,
                match="exhausted typed-deferral task remains blocked",
            ):
                source.rearm_blocked_task(blocked.task_cid, receipt=candidate)
            observed = source.get_task(blocked.task_cid)
            assert observed is not None, name
            assert observed.status == "blocked", name
            assert observed.revision == blocked.revision, name

        with pytest.raises(
            TypedDeferralRecoveryError,
            match="state-owner authority",
        ):
            _owner_request_for_pair(
                blocked,
                failure,
                outcome,
                sentinel=object(),
            )

        # A sealed admission cannot be replayed for another repair identity.
        with pytest.raises(
            TypedDeferralRecoveryError,
            match="stale or mismatched",
        ):
            database_task_source_module.build_typed_deferral_budget_supersession_request(
                task_cid=blocked.task_cid,
                task_revision=blocked.revision,
                task_body=blocked.body,
                repair_head="7" * 40,
                repair_tree=REPAIR_TREE,
                provider_evidence_admission=valid["provider_evidence_admission"],
            )

        high = resolve_agent_implementation_route(
            **{
                **resolve_agent_implementation_route(default_route="legacy").as_dict(),
                "fallback_reasoning_effort": "high",
            }
        )
        denied = build_grok_route_outcome(
            receipt=failure,
            route_plan=high.as_outcome_dict(),
            decision="denied",
            verifier_status="not_confirmed",
            fallback_dispatched=False,
            fallback_returncode=None,
        )
        medium = resolve_agent_implementation_route(default_route="legacy")
        medium_outcome = build_grok_route_outcome(
            receipt=failure,
            route_plan=medium.as_outcome_dict(),
            quota_evidence_id="sha256:" + ("b" * 64),
            decision="fallback_succeeded",
            verifier_status="confirmed_quota",
            fallback_dispatched=True,
            fallback_returncode=0,
        )
        stale_failure, stale_outcome = _positive_provider_pair(
            observed_at_ms=ADMISSION_NOW_MS - 60_001
        )
        future_failure, future_outcome = _positive_provider_pair(
            observed_at_ms=ADMISSION_NOW_MS + 5_001
        )
        forged_outcome = copy.deepcopy(outcome)
        forged_outcome["quota_evidence_id"] = "sha256:" + ("e" * 64)
        rejected_pairs = (
            (failure, denied),
            (failure, medium_outcome),
            (stale_failure, stale_outcome),
            (future_failure, future_outcome),
            (failure, forged_outcome),
        )
        for rejected_failure, rejected_outcome in rejected_pairs:
            with pytest.raises(TypedDeferralRecoveryError):
                _owner_request_for_pair(
                    blocked,
                    rejected_failure,
                    rejected_outcome,
                )

        with pytest.raises(TaskSourceConflictError):
            source.compare_and_set_status(
                blocked.task_cid,
                blocked.revision,
                "retrying",
                {"operation": "generic_owner_cas"},
            )
        assert source.get_task(blocked.task_cid).status == "blocked"


def test_ordinary_dependency_block_keeps_existing_rearm_behavior(tmp_path: Path) -> None:
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        task = _materialize(source, alias="ordinary")
        blocked = source.compare_and_set_status(
            task.task_cid,
            task.revision,
            "blocked",
            {"operation": "dependency_wait", "blocker": "task:other"},
        ).task
        result = source.rearm_blocked_task(blocked.task_cid)
        assert result.changed is True
        assert result.task.status == "retrying"
        assert result.task.body["completion_receipt"]["operation"] == (
            "database_declared_outputs_on_head_rearm"
        )
