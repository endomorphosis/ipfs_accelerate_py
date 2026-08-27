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
    TYPED_DEFERRAL_BUDGET_SUPERSESSION_OPERATION,
    DatabaseTaskSource,
    TaskRecord,
    TaskSourceConflictError,
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


def _matching_attempts_digest(items: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for item in items:
        encoded = json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return "sha256:" + digest.hexdigest()


def _block_leftover_wait_exhausted(
    source: DatabaseTaskSource,
) -> tuple[TaskRecord, dict[str, object]]:
    task = _materialize(source, alias="VRIF-leftover-wait")
    running = source.compare_and_set_status(
        task.task_cid,
        task.revision,
        "in_progress",
        {"operation": "test_leftover_wait_claim"},
    ).task
    attempt_id = "attempt:leftover-wait"
    attempt_number = 1
    matching = [
        {
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "reason": "worktree_lifecycle_claim_exists",
            "deferral_fingerprint": "sha256:" + ("6" * 64),
        }
    ]
    budget_body: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-typed-deferral-budget@1"
        ),
        "task_cid": running.task_cid,
        "task_generation": running.task_cid,
        "generation_fingerprint": "sha256:" + ("5" * 64),
        "current_deferral_fingerprint": matching[0]["deferral_fingerprint"],
        "typed_deferral_candidate_count": 1,
        "typed_deferral_count": 1,
        "typed_deferral_count_is_lower_bound": False,
        "verified_typed_deferral_count": 1,
        "verified_count_complete": True,
        "max_task_attempts": 1,
        "exhausted": True,
        "attempt_consumed": False,
        "typed_deferral_slot_consumed": True,
        "matching_attempts": matching,
        "matching_attempts_digest": _matching_attempts_digest(matching),
        "matching_attempts_truncated": False,
        "omitted_matching_attempt_count": 0,
    }
    budget = {**budget_body, "observation_id": _sha256_identity(budget_body)}
    block_receipt = {
        "operation": "database_portal_typed_deferral_budget_exhausted",
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "claim_id": "claim:leftover-wait",
        "lease_id": "lease:leftover-wait",
        "owner_session_id": "session:leftover-wait",
        "fencing_token": 1,
        "fence_epoch": 1,
        "execution_phase": "failed",
        "execution_revision": 3,
        "execution_finished_at_ms": 2_000_000,
        "reason": "typed_portal_deferral_budget_exhausted",
        "retryable": False,
        "attempt_consumed": False,
        "typed_deferral_slot_consumed": True,
        "retry_budget": budget,
        "prior_queue_entry_preserved_inactive": True,
        "coordination": {
            "attempt_id": attempt_id,
            "claim_id": "claim:leftover-wait",
            "attempt_number": attempt_number,
        },
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


def _leftover_wait_recovery_request(
    blocked: TaskRecord,
    budget: dict[str, object],
) -> dict[str, object]:
    blocked_receipt = dict(blocked.body["completion_receipt"])
    reasons = sorted(
        {
            str(item["reason"])
            for item in budget["matching_attempts"]
            if isinstance(item, dict)
        }
    )
    seed: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-leftover-wait-deferral-budget-recovery@1"
        ),
        "disposition": "retry",
        "reason": "leftover_wait_deferral_budget_cleared",
        "source_reason": "typed_portal_deferral_budget_exhausted",
        "task_cid": blocked.task_cid,
        "task_alias": blocked.task_alias,
        "attempt_id": blocked_receipt["attempt_id"],
        "claim_id": blocked_receipt["claim_id"],
        "lease_id": blocked_receipt["lease_id"],
        "attempt_number": blocked_receipt["attempt_number"],
        "fencing_token": blocked_receipt["fencing_token"],
        "fence_epoch": blocked_receipt["fence_epoch"],
        "exhausting_reasons": reasons,
        "blocked_retry_budget": copy.deepcopy(budget),
        "blocked_retry_budget_observation_id": budget["observation_id"],
        "blocked_retry_budget_digest": _sha256_identity(budget),
        "blocked_matching_attempts_digest": budget[
            "matching_attempts_digest"
        ],
        "identity_bound": True,
        "backoff_seconds": 0,
        "attempt_consumed": False,
    }
    seed["receipt_id"] = _sha256_identity(seed)
    queue_reason = (
        "database_portal_retry:"
        + str(blocked_receipt["attempt_id"])
        + ":leftover_wait_deferral_budget_cleared"
    )[:2048]
    return {
        "operation": (
            "database_portal_leftover_wait_deferral_budget_retry_recovery"
        ),
        "attempt_id": blocked_receipt["attempt_id"],
        "claim_id": blocked_receipt["claim_id"],
        "lease_id": blocked_receipt["lease_id"],
        "owner_session_id": blocked_receipt["owner_session_id"],
        "fencing_token": blocked_receipt["fencing_token"],
        "fence_epoch": blocked_receipt["fence_epoch"],
        "attempt_number": blocked_receipt["attempt_number"],
        "execution_phase": blocked_receipt["execution_phase"],
        "execution_revision": blocked_receipt["execution_revision"],
        "execution_finished_at_ms": blocked_receipt[
            "execution_finished_at_ms"
        ],
        "reason": "leftover_wait_deferral_budget_cleared",
        "backoff_seconds": 0,
        "backoff_ms": 0,
        "retry_not_before_ms": 2_000_001,
        "evidence_source": (
            "typed_portal_leftover_wait_deferral_budget_recovery:"
            + str(seed["receipt_id"])
        ),
        "queue_reason": queue_reason,
        "queue_reused": False,
        "queue_receipt": {
            "task_cid": blocked.task_cid,
            "reason": queue_reason,
            "retry_not_before_ms": 2_000_001,
        },
        "coordination": copy.deepcopy(blocked_receipt["coordination"]),
        "leftover_wait_deferral_budget_recovery_seed": seed,
        "control_expected_status": "blocked",
        "control_expected_revision": blocked.revision,
    }


def _rehash_leftover_wait_recovery(
    recovery: dict[str, object],
    *,
    refresh_budget: bool = False,
) -> None:
    seed = recovery["leftover_wait_deferral_budget_recovery_seed"]
    assert isinstance(seed, dict)
    budget = seed["blocked_retry_budget"]
    assert isinstance(budget, dict)
    if refresh_budget:
        matching = budget["matching_attempts"]
        assert isinstance(matching, list)
        budget["matching_attempts_digest"] = _matching_attempts_digest(matching)
        observation_body = dict(budget)
        observation_body.pop("observation_id", None)
        budget["observation_id"] = _sha256_identity(observation_body)
        seed["exhausting_reasons"] = sorted(
            {
                str(item["reason"])
                for item in matching
                if isinstance(item, dict)
            }
        )
        seed["blocked_retry_budget_observation_id"] = budget["observation_id"]
        seed["blocked_retry_budget_digest"] = _sha256_identity(budget)
        seed["blocked_matching_attempts_digest"] = budget[
            "matching_attempts_digest"
        ]
    seed_body = dict(seed)
    seed_body.pop("receipt_id", None)
    seed["receipt_id"] = _sha256_identity(seed_body)
    recovery["evidence_source"] = (
        "typed_portal_leftover_wait_deferral_budget_recovery:"
        + str(seed["receipt_id"])
    )


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
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE",
        str(tmp_path / "typed-recovery.quack-token"),
    )
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


def test_ordinary_cas_rejects_exact_leftover_wait_recovery_without_queue(
    tmp_path: Path,
) -> None:
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        blocked, budget = _block_leftover_wait_exhausted(source)
        request = _leftover_wait_recovery_request(blocked, budget)
        seed = request["leftover_wait_deferral_budget_recovery_seed"]
        assert isinstance(seed, dict)
        with pytest.raises(
            TaskSourceConflictError,
            match="task-fenced atomic queue/status transition",
        ):
            source.compare_and_set_status(
                blocked.task_cid,
                blocked.revision,
                "retrying",
                request,
                evidence_digests=[str(seed["receipt_id"])],
            )

        observed = source.get_task(blocked.task_cid)
        assert observed is not None
        assert observed.status == "blocked"
        assert observed.revision == blocked.revision
        assert observed.body["completion_receipt"] == blocked.body[
            "completion_receipt"
        ]
        assert source.get_queue_entry(blocked.task_cid) is None


def test_guarded_queue_status_rejects_generic_typed_deferral_rearm(
    tmp_path: Path,
) -> None:
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        blocked, _budget = _block_leftover_wait_exhausted(source)
        blocked_receipt = dict(blocked.body["completion_receipt"])

        with pytest.raises(
            TaskSourceConflictError,
            match="guarded queue/status transition has no admitted recovery",
        ):
            source.record_queue_backoff_and_cas_status(
                task_cid=blocked.task_cid,
                expected_revision=blocked.revision,
                expected_control_receipt=blocked_receipt,
                status="retrying",
                receipt={
                    "operation": "generic_bypass",
                    "queue_reason": "generic_bypass",
                },
                delay_ms=0,
                reason="generic_bypass",
            )

        observed = source.get_task(blocked.task_cid)
        assert observed is not None
        assert observed.status == "blocked"
        assert observed.revision == blocked.revision
        assert observed.body["completion_receipt"] == blocked_receipt
        assert source.get_queue_entry(blocked.task_cid) is None


def test_post_merge_queue_admission_requires_fenced_daemon_authority_and_exact_args(
    tmp_path: Path,
) -> None:
    def transition_for(blocked: TaskRecord) -> dict[str, object]:
        evidence_id = "sha256:" + "e" * 64
        qualification_receipt_id = "sha256:" + "f" * 64
        seed_body = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-post-merge-completion-recovery-seed@2"
            ),
            "task_cid": blocked.task_cid,
            "recovery_control_revision": blocked.revision,
            "qualification_kind": "repair",
            "qualification_receipt_id": qualification_receipt_id,
            "recovery_evidence_id": evidence_id,
            "qualified_target_commit": REPAIR_HEAD,
        }
        return {
            "operation": (
                "database_post_merge_declared_outputs_repair_recovery"
            ),
            "repair_evidence_id": evidence_id,
            "repair_receipt_id": qualification_receipt_id,
            "repair_commit": REPAIR_HEAD,
            "queue_reason": "post-merge-fenced-recovery",
            "post_merge_completion_recovery_seed": {
                **seed_body,
                "seed_id": _sha256_identity(seed_body),
            },
        }

    with DatabaseTaskSource(tmp_path / "unbound.duckdb") as source:
        blocked, _budget = _block_leftover_wait_exhausted(source)
        transition = transition_for(blocked)
        with pytest.raises(
            TaskSourceConflictError,
            match="has no portable authority",
        ):
            source._mint_post_merge_queue_admission(
                task_cid=blocked.task_cid,
                expected_revision=blocked.revision,
                expected_control_receipt=blocked.body["completion_receipt"],
                status="retrying",
                receipt=transition,
                delay_ms=0,
                reason=str(transition["queue_reason"]),
                _portable_authority=object(),
            )

    authority = object()
    with DatabaseTaskSource(
        tmp_path / "bound.duckdb",
        _post_merge_queue_authority=authority,
    ) as source:
        blocked, _budget = _block_leftover_wait_exhausted(source)
        transition = transition_for(blocked)
        admission = source._mint_post_merge_queue_admission(
            task_cid=blocked.task_cid,
            expected_revision=blocked.revision,
            expected_control_receipt=blocked.body["completion_receipt"],
            status="retrying",
            receipt=transition,
            delay_ms=0,
            reason=str(transition["queue_reason"]),
            _portable_authority=authority,
        )
        with pytest.raises(
            TaskSourceConflictError,
            match="does not match the mutation",
        ):
            source.record_queue_backoff_and_cas_status(
                task_cid=blocked.task_cid,
                expected_revision=blocked.revision,
                expected_control_receipt=blocked.body["completion_receipt"],
                status="retrying",
                receipt=transition,
                delay_ms=1,
                reason=str(transition["queue_reason"]),
                _post_merge_recovery_admission=admission,
            )

        observed = source.get_task(blocked.task_cid)
        assert observed is not None
        assert observed.status == "blocked"
        assert observed.revision == blocked.revision
        assert source.get_queue_entry(blocked.task_cid) is None


def test_guarded_protected_recovery_rejects_foreign_queue_atomically(
    tmp_path: Path,
) -> None:
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        ready = _materialize(source)
        running = source.compare_and_set_status(
            ready.task_cid,
            ready.revision,
            "in_progress",
            {"operation": "test_claim"},
        ).task
        blocked_receipt = {
            "operation": "database_portal_terminal_failure",
            "attempt_id": "attempt:protected-path-recovery",
        }
        blocked = source.compare_and_set_status(
            running.task_cid,
            running.revision,
            "blocked",
            blocked_receipt,
        ).task
        source.record_queue_backoff(
            task_cid=blocked.task_cid,
            delay_ms=10,
            reason="foreign-queue-authority",
        )
        before_queue = source.get_queue_entry(blocked.task_cid)
        assert before_queue is not None

        with pytest.raises(
            TaskSourceConflictError,
            match="typed recovery found a foreign queue entry",
        ):
            source.record_queue_backoff_and_cas_status(
                task_cid=blocked.task_cid,
                expected_revision=blocked.revision,
                expected_control_receipt=blocked_receipt,
                status="retrying",
                receipt={
                    "operation": (
                        "database_portal_protected_path_retry_recovery"
                    ),
                    "queue_reason": "protected-path-recovery",
                    "queue_receipt": {},
                    "queue_reused": False,
                    "retry_not_before_ms": 0,
                },
                delay_ms=0,
                reason="protected-path-recovery",
            )

        observed = source.get_task(blocked.task_cid)
        after_queue = source.get_queue_entry(blocked.task_cid)
        assert observed is not None
        assert observed.status == "blocked"
        assert observed.revision == blocked.revision
        assert after_queue is not None
        assert after_queue.reason == before_queue.reason
        assert after_queue.attempt == before_queue.attempt
        assert after_queue.retry_not_before_ms == before_queue.retry_not_before_ms


def test_guarded_leftover_wait_recovery_updates_queue_and_status_atomically(
    tmp_path: Path,
) -> None:
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        blocked, budget = _block_leftover_wait_exhausted(source)
        request = _leftover_wait_recovery_request(blocked, budget)
        blocked_receipt = dict(blocked.body["completion_receipt"])

        result = source.record_queue_backoff_and_cas_status(
            task_cid=blocked.task_cid,
            expected_revision=blocked.revision,
            expected_control_receipt=blocked_receipt,
            status="retrying",
            receipt=request,
            delay_ms=0,
            reason=str(request["queue_reason"]),
            exact_retry_not_before_ms=int(request["retry_not_before_ms"]),
        )

        assert result["cas_result"].changed is True
        assert result["previous_status"] == "blocked"
        assert result["retry_not_before_ms"] == request[
            "retry_not_before_ms"
        ]
        observed = source.get_task(blocked.task_cid)
        queue = source.get_queue_entry(blocked.task_cid)
        assert observed is not None and observed.status == "retrying"
        assert observed.body["completion_receipt"] == result[
            "transition_receipt"
        ]
        assert observed.body["completion_receipt"]["operation"] == request[
            "operation"
        ]
        assert queue is not None
        assert queue.reason == request["queue_reason"]
        assert queue.retry_not_before_ms == request["retry_not_before_ms"]


def test_guarded_leftover_wait_recovery_leaves_no_queue_after_lost_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        blocked, budget = _block_leftover_wait_exhausted(source)
        request = _leftover_wait_recovery_request(blocked, budget)
        blocked_receipt = dict(blocked.body["completion_receipt"])
        original_atomic = (
            source._intent.record_queue_backoff_and_cas_task_status
        )

        def foreign_cas_before_atomic(**kwargs: object) -> object:
            source._intent.cas_task_status(
                task_cid=blocked.task_cid,
                expected_revision=blocked.revision,
                expected_control_receipt=blocked_receipt,
                new_status="in_progress",
                receipt={"operation": "foreign_claim"},
            )
            return original_atomic(**kwargs)

        monkeypatch.setattr(
            source._intent,
            "record_queue_backoff_and_cas_task_status",
            foreign_cas_before_atomic,
        )
        with pytest.raises(TaskSourceConflictError, match="revision CAS is stale"):
            source.record_queue_backoff_and_cas_status(
                task_cid=blocked.task_cid,
                expected_revision=blocked.revision,
                expected_control_receipt=blocked_receipt,
                status="retrying",
                receipt=request,
                delay_ms=0,
                reason=str(request["queue_reason"]),
                exact_retry_not_before_ms=int(request["retry_not_before_ms"]),
            )

        observed = source.get_task(blocked.task_cid)
        assert observed is not None
        assert observed.status == "in_progress"
        assert observed.body["completion_receipt"] == {
            "operation": "foreign_claim"
        }
        assert source.get_queue_entry(blocked.task_cid) is None


def test_forged_special_leftover_wait_rearms_remain_blocked(
    tmp_path: Path,
) -> None:
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        blocked, budget = _block_leftover_wait_exhausted(source)
        valid = _leftover_wait_recovery_request(blocked, budget)
        blocked_receipt = dict(blocked.body["completion_receipt"])

        foreign_attempt = copy.deepcopy(valid)
        foreign_attempt["attempt_id"] = "attempt:foreign"

        foreign_claim_seed = copy.deepcopy(valid)
        claim_seed = foreign_claim_seed[
            "leftover_wait_deferral_budget_recovery_seed"
        ]
        assert isinstance(claim_seed, dict)
        claim_seed["claim_id"] = "claim:foreign"
        _rehash_leftover_wait_recovery(foreign_claim_seed)

        foreign_fence = copy.deepcopy(valid)
        foreign_fence["fencing_token"] = 2
        fence_seed = foreign_fence[
            "leftover_wait_deferral_budget_recovery_seed"
        ]
        assert isinstance(fence_seed, dict)
        fence_seed["fencing_token"] = 2
        _rehash_leftover_wait_recovery(foreign_fence)

        foreign_execution_revision = copy.deepcopy(valid)
        foreign_execution_revision["execution_revision"] = 4

        foreign_control_revision = copy.deepcopy(valid)
        foreign_control_revision["control_expected_revision"] = (
            blocked.revision - 1
        )

        foreign_wait_reason = copy.deepcopy(valid)
        reason_seed = foreign_wait_reason[
            "leftover_wait_deferral_budget_recovery_seed"
        ]
        assert isinstance(reason_seed, dict)
        reason_budget = reason_seed["blocked_retry_budget"]
        assert isinstance(reason_budget, dict)
        reason_matching = reason_budget["matching_attempts"]
        assert isinstance(reason_matching, list)
        assert isinstance(reason_matching[0], dict)
        reason_matching[0]["reason"] = "provider_capacity_backoff"
        _rehash_leftover_wait_recovery(
            foreign_wait_reason,
            refresh_budget=True,
        )

        forged_seed_budget = copy.deepcopy(valid)
        forged_budget_seed = forged_seed_budget[
            "leftover_wait_deferral_budget_recovery_seed"
        ]
        assert isinstance(forged_budget_seed, dict)
        forged_budget = forged_budget_seed["blocked_retry_budget"]
        assert isinstance(forged_budget, dict)
        forged_budget["max_task_attempts"] = 2
        _rehash_leftover_wait_recovery(
            forged_seed_budget,
            refresh_budget=True,
        )

        unknown_field = copy.deepcopy(valid)
        unknown_field["authorization"] = "caller-controlled"

        foreign_expected_receipt = copy.deepcopy(blocked_receipt)
        foreign_expected_receipt["attempt_id"] = "attempt:foreign"

        cases: dict[
            str,
            tuple[
                dict[str, object],
                dict[str, object] | None,
                list[str],
                str,
            ],
        ] = {}
        for name, candidate in {
            "attempt": foreign_attempt,
            "claim_seed": foreign_claim_seed,
            "fence": foreign_fence,
            "execution_revision": foreign_execution_revision,
            "control_revision": foreign_control_revision,
            "non_wait_reason": foreign_wait_reason,
            "seed_budget": forged_seed_budget,
            "unknown_field": unknown_field,
        }.items():
            candidate_seed = candidate[
                "leftover_wait_deferral_budget_recovery_seed"
            ]
            assert isinstance(candidate_seed, dict)
            cases[name] = (
                candidate,
                blocked_receipt,
                [str(candidate_seed["receipt_id"])],
                "retrying",
            )
        valid_seed = valid["leftover_wait_deferral_budget_recovery_seed"]
        assert isinstance(valid_seed, dict)
        cases["foreign_expected_receipt"] = (
            valid,
            foreign_expected_receipt,
            [str(valid_seed["receipt_id"])],
            "retrying",
        )
        cases["foreign_evidence_digest"] = (
            valid,
            blocked_receipt,
            ["sha256:" + ("0" * 64)],
            "retrying",
        )
        cases["wrong_target_status"] = (
            valid,
            blocked_receipt,
            [str(valid_seed["receipt_id"])],
            "in_progress",
        )

        for name, (candidate, expected_receipt, digests, status) in cases.items():
            with pytest.raises(
                TaskSourceConflictError,
                match="exhausted typed-deferral task remains blocked",
            ):
                source.compare_and_set_status(
                    blocked.task_cid,
                    blocked.revision,
                    status,
                    candidate,
                    expected_control_receipt=expected_receipt,
                    evidence_digests=digests,
                )
            observed = source.get_task(blocked.task_cid)
            assert observed is not None, name
            assert observed.status == "blocked", name
            assert observed.revision == blocked.revision, name


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
