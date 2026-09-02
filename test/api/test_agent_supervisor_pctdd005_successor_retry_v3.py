"""Runtime gates for the one-shot PCTDD-005 r26 successor authorization."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    ProcessBirthIdentity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    database_task_source as database_task_source_module,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    duckdb_state as duckdb_state_module,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    intent_repository as intent_repository_module,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    TaskSourceUnknownOutcomeError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    FENCED_PROVIDER_OUTER_AUTHORITY_POPULATION_RECEIPT_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    database_portal_bridge as bridge_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as supervisor_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_ID,
    DATABASE_PCTDD005_SUCCESSOR_MANIFEST_ID,
    DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN,
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
    database_pctdd005_successor_credit,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATABASE_FENCED_PROVIDER_INNER_POPULATION_RECEIPT_SCHEMA,
    DATABASE_FENCED_PROVIDER_RETAINED_ADMISSION_SCHEMA,
    DATABASE_PCTDD005_HISTORICAL_SUCCESSOR_ADMISSION_SCHEMA,
    DATABASE_PCTDD005_HISTORICAL_SUCCESSOR_CONSUMPTION_SCHEMA,
    DATABASE_PCTDD005_HISTORICAL_SUCCESSOR_RECONCILIATION_SCHEMA,
    DATABASE_PCTDD005_SUCCESSOR_ADMISSION_SCHEMA,
    DATABASE_PCTDD005_SUCCESSOR_CONSUMPTION_SCHEMA,
    DatabaseImplementationConflictError,
    DatabaseImplementationDaemon,
    database_fenced_provider_historical_retained_consumption,
    database_fenced_provider_historical_retained_consumption_matches_admission,
    database_fenced_provider_retained_admission,
    database_fenced_provider_retained_admission_valid,
    database_fenced_provider_retained_consumption,
    database_fenced_provider_retained_consumption_matches_admission,
    database_pctdd005_successor_admission_valid,
    database_pctdd005_successor_consumption_valid,
    database_pctdd005_successor_reconciliation_valid,
    database_pctdd005_historical_successor_admission_valid,
    database_pctdd005_historical_successor_consumption_valid,
    database_pctdd005_historical_successor_reconciliation_valid,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.retained_recovery_contracts import (
    database_portal_controller_quiescence_receipt_valid,
)
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
)
from test.api.test_agent_supervisor_retained_occurrence_recovery_v2 import (
    _arm_retrying_task,
    _population_for,
)


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def _receipts(
    pin: Mapping[str, Any],
    credit: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    credit_id = _sha256(dict(credit))
    common = {
        "task_cid": pin["task_cid"],
        "task_alias": pin["task_alias"],
        "task_revision": pin["blocked_task_revision"],
        "attempt_id": pin["predecessor_attempt_id"],
        "claim_id": pin["predecessor_claim_id"],
        "owner_session_id": pin["predecessor_owner_session_id"],
        "fencing_token": pin["predecessor_fencing_token"],
        "fence_epoch": pin["predecessor_fence_epoch"],
    }
    inner = {
        "schema": DATABASE_FENCED_PROVIDER_INNER_POPULATION_RECEIPT_SCHEMA,
        "query_profile_id": pin["inner_query_profile_id"],
        "receipt_nonce": pin["receipt_nonce"],
        "receipt_epoch": pin["receipt_epoch"],
        "authority": {
            "authority_mode": "quack",
            "control_store_id": pin["owner_store_id"],
            "control_store_generation": pin["control_store_generation"],
        },
        "subject": {
            **common,
            "lease_id": pin["predecessor_lease_id"],
            "attempt_number": pin["predecessor_attempt_number"],
            "recovery_manifest_id": DATABASE_PCTDD005_SUCCESSOR_MANIFEST_ID,
            "recovery_credit_id": credit_id,
        },
        "groups": {
            "provider_invocations": {"count": 0, "rows": []},
            "effect_claims": {"count": 0, "rows": []},
        },
        "coordinator_receipt_cid": "sha256:" + "0" * 64,
        "receipt_cid": "sha256:" + "1" * 64,
    }
    generation = pin["owner_generation_floor"]
    outer = {
        "schema": FENCED_PROVIDER_OUTER_AUTHORITY_POPULATION_RECEIPT_SCHEMA,
        "query_profile_id": pin["outer_query_profile_id"],
        "receipt_nonce": pin["receipt_nonce"],
        "receipt_epoch": pin["receipt_epoch"],
        "subject": {
            **common,
            "expected_task_status": pin["blocked_task_status"],
            "expected_store_id": pin["owner_store_id"],
            "expected_store_generation": generation,
        },
        "cross_store_context": {
            "coordination_lease_id": pin["predecessor_lease_id"],
        },
        "authority": {
            "owner_binding": {
                "store_id": pin["owner_store_id"],
                "generation": generation,
                "database_uuid": pin["owner_database_uuid"],
                "schema_fingerprint": pin["owner_schema_fingerprint"],
                "schema_revision": pin["owner_schema_revision"],
            }
        },
        "publication_assessment": {
            "accepted_publication_count": 0,
            "terminal_task_status_count": 0,
            "completion_receipt_count": 0,
            "successful_merge_queue_count": 0,
            "successful_merge_attempt_count": 0,
        },
        "receipt_cid": "sha256:" + "2" * 64,
    }
    return inner, outer


def _admission(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], dict[str, Any]]:
    pin = dict(DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN)
    credit = dict(database_pctdd005_successor_credit(pin))
    inner, outer = _receipts(pin, credit)
    monkeypatch.setattr(
        daemon_module,
        "database_fenced_provider_inner_population_receipt_valid",
        lambda value: isinstance(value, Mapping),
    )
    monkeypatch.setattr(
        intent_repository_module,
        "fenced_provider_outer_authority_population_receipt_valid",
        lambda value: isinstance(value, Mapping),
    )
    admission = dict(
        database_fenced_provider_retained_admission(
            occurrence=pin,
            credit=credit,
            inner_receipt=inner,
            outer_receipt=outer,
            expected_task_revision=pin["blocked_task_revision"],
            expected_task_status=pin["blocked_task_status"],
        )
    )
    return pin, admission


def test_pctdd005_admission_binds_inner_schema_to_authenticated_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin = dict(DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN)
    credit = dict(database_pctdd005_successor_credit(pin))
    inner, outer = _receipts(pin, credit)
    inner["authority"]["control_schema_evidence"] = {
        "schema_fingerprint": pin["owner_schema_fingerprint"],
    }
    monkeypatch.setattr(
        daemon_module,
        "database_fenced_provider_inner_population_receipt_valid",
        lambda value: isinstance(value, Mapping),
    )
    monkeypatch.setattr(
        intent_repository_module,
        "fenced_provider_outer_authority_population_receipt_valid",
        lambda value: isinstance(value, Mapping),
    )

    admission = database_fenced_provider_retained_admission(
        occurrence=pin,
        credit=credit,
        inner_receipt=inner,
        outer_receipt=outer,
        expected_task_revision=pin["blocked_task_revision"],
        expected_task_status=pin["blocked_task_status"],
    )
    assert admission["owner_schema_fingerprint"] == (
        pin["owner_schema_fingerprint"]
    )

    crossed_inner = dict(inner)
    crossed_inner["authority"] = dict(inner["authority"])
    crossed_inner["authority"]["control_schema_evidence"] = {
        "schema_fingerprint": "sha256:" + "0" * 64,
    }
    with pytest.raises(
        DatabaseImplementationConflictError,
        match="receipt bindings drifted",
    ):
        database_fenced_provider_retained_admission(
            occurrence=pin,
            credit=credit,
            inner_receipt=crossed_inner,
            outer_receipt=outer,
            expected_task_revision=pin["blocked_task_revision"],
            expected_task_status=pin["blocked_task_status"],
        )


def test_pctdd005_admission_and_consumption_are_versioned_and_cross_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin, admission = _admission(monkeypatch)
    assert admission["schema"] == DATABASE_PCTDD005_SUCCESSOR_ADMISSION_SCHEMA
    assert database_pctdd005_successor_admission_valid(admission)
    assert database_fenced_provider_retained_admission_valid(admission)

    crossed = dict(admission)
    crossed["schema"] = DATABASE_FENCED_PROVIDER_RETAINED_ADMISSION_SCHEMA
    crossed["manifest_id"] = DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_ID
    unsigned = dict(crossed)
    unsigned.pop("admission_id")
    crossed["admission_id"] = _sha256(unsigned)
    assert not database_fenced_provider_retained_admission_valid(crossed)

    consumption = dict(
        database_fenced_provider_retained_consumption(
            admission=admission,
            attempt_id="attempt:pctdd005-successor",
            claim_id="claim:pctdd005-successor",
            lease_id="lease:pctdd005-successor",
            owner_session_id="owner:pctdd005-successor",
            attempt_number=7,
            fencing_token=7,
            fence_epoch=7,
            expected_task_revision=pin["blocked_task_revision"] + 1,
            expected_task_status="retrying",
            resulting_task_revision=pin["blocked_task_revision"] + 2,
            resulting_task_status="in_progress",
        )
    )

    attempt = SimpleNamespace(
        task_cid=pin["task_cid"],
        task_alias=pin["task_alias"],
        attempt_id=consumption["attempt_id"],
        claim_id=consumption["claim_id"],
        lease_id=consumption["lease_id"],
        owner_session_id=consumption["owner_session_id"],
        attempt_number=consumption["attempt_number"],
        fencing_token=consumption["fencing_token"],
        fence_epoch=consumption["fence_epoch"],
        body={
            "control_claim": {
                "revision": pin["blocked_task_revision"] + 2,
            },
            "retry_budget": {
                "retained_recovery_admission": admission,
                "retained_recovery_consumption": consumption,
            },
        },
    )
    policy = DatabasePortalExecutionBridge._retained_recovery_execution_policy(
        attempt
    )
    assert policy is not None
    assert policy["allow_pool"] is False
    assert policy["seed_prior_attempt"] is False
    assert policy["source_relative_path"] == "external/ipfs_datasets"
    attempt.body = {"control_claim": {"revision": pin["blocked_task_revision"] + 2}}
    with pytest.raises(DatabasePortalBridgeError, match="lost its one-shot policy"):
        DatabasePortalExecutionBridge._retained_recovery_execution_policy(attempt)
    assert consumption["schema"] == DATABASE_PCTDD005_SUCCESSOR_CONSUMPTION_SCHEMA
    assert database_pctdd005_successor_consumption_valid(consumption)
    assert database_fenced_provider_retained_consumption_matches_admission(
        admission=admission,
        consumption=consumption,
    )
    with pytest.raises(DatabaseImplementationConflictError):
        database_fenced_provider_retained_consumption(
            admission=admission,
            attempt_id="attempt:second",
            claim_id="claim:second",
            lease_id="lease:second",
            owner_session_id="owner:second",
            attempt_number=8,
            fencing_token=8,
            fence_epoch=8,
            expected_task_revision=pin["blocked_task_revision"] + 2,
            expected_task_status="retrying",
            resulting_task_revision=pin["blocked_task_revision"] + 3,
            resulting_task_status="in_progress",
        )


def test_pctdd005_historical_chain_is_additive_and_cross_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin, legacy_admission = _admission(monkeypatch)
    admission = dict(legacy_admission)
    admission["schema"] = (
        DATABASE_PCTDD005_HISTORICAL_SUCCESSOR_ADMISSION_SCHEMA
    )
    admission["historical_occurrence_authority_id"] = "sha256:" + "8" * 64
    admission["controller_quiescence_receipt_id"] = "sha256:" + "9" * 64
    admission.pop("admission_id")
    admission["admission_id"] = _sha256(admission)
    assert database_pctdd005_historical_successor_admission_valid(admission)
    assert not database_pctdd005_successor_admission_valid(admission)

    consumption = dict(
        database_fenced_provider_historical_retained_consumption(
            admission=admission,
            attempt_id="attempt:pctdd005-historical",
            claim_id="claim:pctdd005-historical",
            lease_id="lease:pctdd005-historical",
            owner_session_id="owner:pctdd005-historical",
            attempt_number=8,
            fencing_token=8,
            fence_epoch=8,
            expected_task_revision=pin["blocked_task_revision"] + 1,
            expected_task_status="retrying",
            resulting_task_revision=pin["blocked_task_revision"] + 2,
            resulting_task_status="in_progress",
        )
    )
    assert (
        consumption["schema"]
        == DATABASE_PCTDD005_HISTORICAL_SUCCESSOR_CONSUMPTION_SCHEMA
    )
    assert database_pctdd005_historical_successor_consumption_valid(
        consumption
    )
    assert not database_pctdd005_successor_consumption_valid(consumption)
    assert (
        database_fenced_provider_historical_retained_consumption_matches_admission(
            admission=admission,
            consumption=consumption,
        )
    )

    reconciliation = {
        "schema": (
            DATABASE_PCTDD005_HISTORICAL_SUCCESSOR_RECONCILIATION_SCHEMA
        ),
        "attempted": True,
        "reconciled": True,
        "blocked": False,
        "reason": "retained_occurrence_reconciliation_complete",
        "expected_occurrence_count": 1,
        "admitted_count": 1,
        "already_consumed_count": 0,
        "outcomes": [
            {
                "task_alias": pin["task_alias"],
                "task_cid": pin["task_cid"],
                "reconciled": True,
                "blocked": False,
                "reason": "retained_occurrence_admitted",
                "admission_id": admission["admission_id"],
                "consumption_id": "",
            }
        ],
    }
    assert database_pctdd005_historical_successor_reconciliation_valid(
        reconciliation
    )
    assert not database_pctdd005_successor_reconciliation_valid(reconciliation)
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon._task_source = SimpleNamespace(
        get=lambda _task_cid: SimpleNamespace(
            body={
                "completion_receipt": {
                    "retained_recovery_admission": admission
                }
            }
        )
    )
    daemon.open = lambda: daemon
    daemon._retained_recovery_admission_is_current_for_task = (
        lambda *_args, **_kwargs: True
    )
    daemon._retained_recovery_admission_population_admitted_current = (
        lambda *_args, **_kwargs: True
    )
    assert daemon.pctdd005_successor_reconciliation_matches_current(
        reconciliation
    )
    legacy_splice = dict(admission)
    legacy_splice["schema"] = DATABASE_PCTDD005_SUCCESSOR_ADMISSION_SCHEMA
    daemon._task_source = SimpleNamespace(
        get=lambda _task_cid: SimpleNamespace(
            body={
                "completion_receipt": {
                    "retained_recovery_admission": legacy_splice
                }
            }
        )
    )
    assert not daemon.pctdd005_successor_reconciliation_matches_current(
        reconciliation
    )


@pytest.mark.parametrize(
    "changes",
    [
        {
            "recovery_mode": "runner_fenced_clean_removed",
            "candidate_disposition": "clean_removed",
        },
        {
            "recovery_mode": "runner_fenced_dirty_candidate_unavailable",
            "candidate_disposition": "dirty_candidate_unavailable",
        },
        {
            "recovery_mode": "unpublished_rescue_quarantined",
            "candidate_disposition": "rescue_quarantined",
            "retained_ref": "refs/heads/rescue/pctdd-005",
            "retained_commit": "a" * 40,
        },
        {"retained_ref": "refs/heads/rescue/pctdd-005"},
        {"retained_commit": "a" * 40},
        {"retained_worktree_path": "/tmp/foreign-candidate"},
        {"recovery_mode": "foreign_mode"},
        {"allow_pool": True},
        {"seed_prior_attempt": True},
    ],
)
def test_pctdd005_no_task_edits_disposition_is_exact_and_never_rescued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: Mapping[str, Any],
) -> None:
    pin = dict(DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN)
    outer_root = tmp_path / "outer"
    source_root = outer_root / "external" / "ipfs_datasets"
    outer_common = tmp_path / "outer.git"
    source_common = tmp_path / "source.git"
    for path in (outer_root, source_root, outer_common, source_common):
        path.mkdir(parents=True, exist_ok=True)
    pin.update(
        {
            "disposition_repository_root": str(outer_root),
            "disposition_git_common_dir": str(outer_common),
            "source_repository_root": str(source_root),
            "source_git_common_dir": str(source_common),
        }
    )
    git_state = {
        "branch_returncode": 1,
        "gitlink_ref": pin["clean_baseline_ref"],
    }

    def fake_git(
        args: list[str],
        *,
        cwd: Path,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        git_args = args[1:]
        if git_args == ["rev-parse", "--show-toplevel"]:
            return SimpleNamespace(returncode=0, stdout=str(cwd) + "\n")
        if git_args == ["rev-parse", "--git-common-dir"]:
            common = source_common if cwd == source_root else outer_common
            return SimpleNamespace(returncode=0, stdout=str(common) + "\n")
        if git_args[:2] == ["cat-file", "-e"]:
            return SimpleNamespace(returncode=0, stdout="")
        if git_args[:1] == ["ls-tree"]:
            line = (
                f"160000 commit {git_state['gitlink_ref']}\t"
                f"{pin['source_relative_path']}\n"
            )
            return SimpleNamespace(returncode=0, stdout=line)
        if git_args[:3] == ["show-ref", "--verify", "--quiet"]:
            return SimpleNamespace(
                returncode=git_state["branch_returncode"],
                stdout="",
            )
        if git_args == [
            "rev-parse",
            "--verify",
            f"refs/heads/{pin['predecessor_branch']}^{{commit}}",
        ]:
            return SimpleNamespace(
                returncode=0,
                stdout=f"{pin['disposition_baseline_ref']}\n",
            )
        raise AssertionError(f"unexpected git invocation: {git_args!r}")

    monkeypatch.setattr(daemon_module.subprocess, "run", fake_git)
    validate_disposition = (
        daemon_module.DatabaseImplementationDaemon
        ._retained_recovery_disposition_is_current
    )
    assert validate_disposition(pin)
    git_state["branch_returncode"] = 0
    assert not validate_disposition(pin)
    assert validate_disposition(
        pin,
        allow_predecessor_baseline_ref=True,
    )
    git_state["branch_returncode"] = 1
    git_state["gitlink_ref"] = "b" * 40
    assert not validate_disposition(pin)
    git_state["gitlink_ref"] = pin["clean_baseline_ref"]

    near_miss = {**pin, **dict(changes)}
    assert not validate_disposition(near_miss)


def _admit_for_claim(daemon: Any, admission: Mapping[str, Any]) -> None:
    expected = dict(admission)
    daemon._retained_recovery_admission_fence_is_current = (
        lambda candidate, *, allowed_states: bool(
            candidate == expected and "admitted" in allowed_states
        )
    )
    daemon._retained_recovery_admission_population_admitted_current = (
        lambda candidate: candidate == expected
    )


def test_pctdd005_reserved_alias_cannot_be_rebound() -> None:
    daemon = object.__new__(daemon_module.DatabaseImplementationDaemon)
    forged = SimpleNamespace(
        task_cid="task:forged:pctdd-005",
        task_alias="PCTDD-005",
        status="blocked",
        revision=26,
        body={},
    )

    assert daemon._retained_recovery_reserved_epoch_state(forged) == (
        "invalid_reserved_identity"
    )
    assert daemon._automatic_claim_forbidden_current(forged) is True


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_pctdd005_credit_is_consumed_before_attempt_or_provider_work(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin, admission = _admission(monkeypatch)
    provider_calls: list[str] = []
    daemon = _open_daemon(tmp_path, provider_calls=provider_calls)
    try:
        daemon.materialize_population(_population_for(pin))
        _arm_retrying_task(daemon, pin, admission)
        _admit_for_claim(daemon, admission)
        real_insert = daemon._insert_attempt_from_claim
        observed: list[str] = []

        def insert_after_consumption(claim: Any, **kwargs: Any):
            current = daemon.task_source.get(pin["task_cid"])
            receipt = dict(current.body["completion_receipt"])
            assert current.status == "in_progress"
            assert database_pctdd005_successor_consumption_valid(
                receipt["retained_recovery_consumption"]
            )
            assert daemon.get_attempt(str(claim.attempt_id)) is None
            assert provider_calls == []
            observed.append("consumed")
            return real_insert(claim, **kwargs)

        monkeypatch.setattr(daemon, "_insert_attempt_from_claim", insert_after_consumption)
        attempt = daemon.claim_next()
        assert attempt is not None
        assert observed == ["consumed"]
        assert daemon.claim_next() is None
        assert provider_calls == []
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_pctdd005_claim_cas_response_loss_reads_back_once_without_reissue(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin, admission = _admission(monkeypatch)
    provider_calls: list[str] = []
    daemon = _open_daemon(tmp_path, provider_calls=provider_calls)
    try:
        daemon.materialize_population(_population_for(pin))
        _arm_retrying_task(daemon, pin, admission)
        _admit_for_claim(daemon, admission)
        real_cas = daemon._cas_task_status_database
        losses = 0

        def lose_response(task_cid: str, **kwargs: Any):
            nonlocal losses
            result = real_cas(task_cid, **kwargs)
            if kwargs.get("new_status") == "in_progress" and losses == 0:
                losses += 1
                raise TaskSourceUnknownOutcomeError("injected committed response loss")
            return result

        monkeypatch.setattr(daemon, "_cas_task_status_database", lose_response)
        attempt = daemon.claim_next()
        assert attempt is not None
        assert losses == 1
        current = daemon.task_source.get(pin["task_cid"])
        receipt = dict(current.body["completion_receipt"])
        assert current.status == "in_progress"
        assert database_pctdd005_successor_consumption_valid(
            receipt["retained_recovery_consumption"]
        )
        assert daemon.claim_next() is None
        assert provider_calls == []
    finally:
        daemon.close()


def _lease_scoped_reconciliation_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_at: str = "",
) -> tuple[
    PortalImplementationSupervisor,
    DatabaseProgramConfig,
    list[str],
    list[Any],
]:
    """Build a no-I/O supervisor seam around the retained recovery lease."""

    repo = tmp_path / "repo"
    state = repo / "state"
    state.mkdir(parents=True)
    todo = repo / "todo.md"
    todo.write_text("# Tasks\n", encoding="utf-8")
    program = DatabaseProgramConfig(
        authority_mode="quack",
        task_source_kind="duckdb",
        endpoint_secret_handle="env://PCTDD005_TEST_QUACK_TOKEN",
        quack_endpoint="quack://127.0.0.1:41307",
        store_id="state/control.duckdb",
        store_generation="pctdd-v1-g9",
        schema_revision="1",
    )
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=todo,
            state_path=state / "task-state.json",
            strategy_path=state / "strategy.json",
            events_path=state / "events.jsonl",
            state_dir=state,
            repo_root=repo,
            state_prefix="pctdd",
            task_prefix="## PCTDD-",
            database_program=program,
        )
    )
    events: list[str] = []
    daemons: list[Any] = []

    class FakeOwnerIntent:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def close(self) -> None:
            events.append("intent_close")

    class FakeOwnerTaskSource:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def close(self) -> None:
            events.append("source_close")

    class FakeBridge:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        run_provider = apply_effect = validate_effect = lambda *_a, **_k: {}

    class FakeDaemon:
        def __init__(self, **kwargs: Any) -> None:
            self.task_source = kwargs["task_source"]
            self._database_portal_outer_authority_cas = None
            self.controller_quiescence_receipts: list[Any] = []
            daemons.append(self)

        def bind_execution_callbacks(self, **_kwargs: Any) -> None:
            return None

        def bind_database_portal_bridge(self, _bridge: Any) -> None:
            return None

        def bind_database_portal_outer_authority_cas(self, callback: Any) -> None:
            self._database_portal_outer_authority_cas = callback
            assert getattr(
                callback,
                "__database_portal_checkout_mutation_lease_held__",
                None,
            ) is False
            events.append("bind_false")

        def reconcile_quiesced_database_portal_attempts(
            self,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return {
                "reconciled": True,
                "blocked": False,
                "repair_batch_pending": False,
            }

        def _assert_lease(self, event: str) -> None:
            assert getattr(
                self._database_portal_outer_authority_cas,
                "__database_portal_checkout_mutation_lease_held__",
                None,
            ) is True
            events.append(event)
            if fail_at == event:
                raise RuntimeError("injected lease-scoped reconciliation failure")

        def reconcile_pctdd005_successor_orphaned_claims(self) -> list[Any]:
            self._assert_lease("pctdd005_orphan")
            return []

        def reconcile_retained_recovery_orphaned_claims(self) -> list[Any]:
            self._assert_lease("retained_orphan")
            return []

        def reconcile_pctdd005_successor_occurrence(
            self,
            *,
            controller_quiescence_receipt: Any = None,
        ) -> dict[str, Any]:
            self._assert_lease("pctdd005_occurrence")
            self.controller_quiescence_receipts.append(
                controller_quiescence_receipt
            )
            return {"blocked": False}

        def reconcile_retained_fenced_provider_occurrences(
            self,
            *,
            controller_quiescence_receipt: Any = None,
        ) -> dict[str, Any]:
            self._assert_lease("retained_occurrence")
            self.controller_quiescence_receipts.append(
                controller_quiescence_receipt
            )
            return {"blocked": False}

        def pctdd005_successor_reconciliation_matches_current(
            self,
            _value: Any,
        ) -> bool:
            return True

        def retained_recovery_reconciliation_matches_current(
            self,
            _value: Any,
        ) -> bool:
            return True

        def close(self) -> None:
            events.append("daemon_close")

    monkeypatch.setattr(
        supervisor_module,
        "_OwnerFencedIntentRepository",
        FakeOwnerIntent,
    )
    monkeypatch.setattr(
        database_task_source_module,
        "DatabaseTaskSource",
        FakeOwnerTaskSource,
    )
    monkeypatch.setattr(
        duckdb_state_module,
        "_resolve_quack_token_handle",
        lambda **_kwargs: ("opaque-test-handle", {"store_id": program.store_id}),
    )
    monkeypatch.setattr(
        daemon_module,
        "DatabaseImplementationDaemon",
        FakeDaemon,
    )
    monkeypatch.setattr(
        daemon_module,
        "database_fenced_provider_retained_reconciliation_valid",
        lambda _value: True,
    )
    monkeypatch.setattr(
        daemon_module,
        "database_pctdd005_successor_reconciliation_valid",
        lambda _value: True,
    )
    monkeypatch.setattr(
        daemon_module,
        "database_fenced_provider_any_retained_reconciliation_valid",
        lambda _value: True,
    )
    monkeypatch.setattr(
        daemon_module,
        "database_fenced_provider_historical_retained_reconciliation_valid",
        lambda _value: True,
    )
    monkeypatch.setattr(
        daemon_module,
        "database_pctdd005_historical_successor_reconciliation_valid",
        lambda _value: True,
    )
    monkeypatch.setattr(
        bridge_module,
        "DatabasePortalExecutionBridge",
        FakeBridge,
    )
    monkeypatch.setattr(
        supervisor,
        "_normalized_quack_owner_binding",
        lambda _value: {"store_id": program.store_id},
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_mutation_inbox_barrier",
        lambda _store_id: {},
    )
    monkeypatch.setattr(
        supervisor,
        "_retained_fenced_provider_program_applicable",
        lambda _program: True,
    )
    monkeypatch.setattr(
        supervisor,
        "_supervisor_checkout_lock_metadata",
        lambda **_kwargs: {},
    )

    lease = SimpleNamespace(lock_path=repo / "checkout.lock")

    def acquire(*_args: Any, **_kwargs: Any) -> tuple[Any, str, None]:
        assert daemons
        assert getattr(
            daemons[0]._database_portal_outer_authority_cas,
            "__database_portal_checkout_mutation_lease_held__",
            None,
        ) is False
        events.append("acquire_false")
        return lease, "acquired", None

    def release(*_args: Any, **_kwargs: Any) -> bool:
        assert getattr(
            daemons[0]._database_portal_outer_authority_cas,
            "__database_portal_checkout_mutation_lease_held__",
            None,
        ) is False
        events.append("release_false")
        return True

    monkeypatch.setattr(supervisor, "_acquire_supervisor_checkout_lease", acquire)
    monkeypatch.setattr(supervisor, "_release_supervisor_checkout_lease", release)
    return supervisor, program, events, daemons


def _retained_controller_cleanup() -> dict[str, Any]:
    return {
        "pid": None,
        "managed_daemon_identity_record_id": "",
        "managed_daemon_process_birth": None,
        "quiesced": True,
        "remaining_pid": None,
        "markers_removed": True,
        "daemon_fence": {
            "fenced": False,
            "safe_to_restart": True,
            "reason": "managed_daemon_not_recorded",
        },
        "provider_runner_fence": {
            "applicable": False,
            "fenced": False,
            "safe_to_restart": True,
            "reason": "ordinary_provider_runner_receipt_absent",
        },
    }


def test_retained_recovery_requires_both_controller_fences_before_daemon_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor, program, events, daemons = _lease_scoped_reconciliation_supervisor(
        tmp_path,
        monkeypatch,
    )

    missing_launch = supervisor._reconcile_interrupted_database_portal_attempts_bound(
        program,
        owner_fence_held=True,
    )
    assert missing_launch["reason"] == (
        "database_portal_retained_launch_fence_absent"
    )
    assert missing_launch["safe_to_restart"] is False

    missing_cleanup = supervisor._reconcile_interrupted_database_portal_attempts_bound(
        program,
        owner_fence_held=True,
        managed_daemon_launch_lock_held=True,
    )
    assert missing_cleanup["reason"] == "database_portal_retained_cleanup_absent"
    assert missing_cleanup["safe_to_restart"] is False
    assert events == []
    assert daemons == []


def test_retained_recovery_rejects_missing_controller_receipt_before_daemon_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor, program, events, daemons = _lease_scoped_reconciliation_supervisor(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_portal_controller_quiescence_receipt",
        lambda **_kwargs: None,
    )

    result = supervisor._reconcile_interrupted_database_portal_attempts_bound(
        program,
        owner_fence_held=True,
        managed_daemon_launch_lock_held=True,
        managed_daemon_cleanup=_retained_controller_cleanup(),
    )

    assert result["reason"] == (
        "database_portal_retained_controller_quiescence_unavailable"
    )
    assert result["safe_to_restart"] is False
    assert events == []
    assert daemons == []


def test_retained_recovery_checkout_capability_is_true_only_inside_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor, program, events, daemons = _lease_scoped_reconciliation_supervisor(
        tmp_path,
        monkeypatch,
    )

    result = supervisor._reconcile_interrupted_database_portal_attempts_bound(
        program,
        owner_fence_held=True,
        managed_daemon_launch_lock_held=True,
        managed_daemon_cleanup=_retained_controller_cleanup(),
    )

    assert result["reconciled"] is True
    assert events[:7] == [
        "bind_false",
        "acquire_false",
        "pctdd005_orphan",
        "retained_orphan",
        "pctdd005_occurrence",
        "retained_occurrence",
        "release_false",
    ]
    assert getattr(
        daemons[0]._database_portal_outer_authority_cas,
        "__database_portal_checkout_mutation_lease_held__",
        None,
    ) is False


def test_retained_recovery_forwards_one_controller_authenticated_quiescence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor, program, _events, daemons = (
        _lease_scoped_reconciliation_supervisor(tmp_path, monkeypatch)
    )
    controller_birth = ProcessBirthIdentity(
        pid=700,
        start_time_ticks=17,
        boot_id="boot:controller",
        parent_pid=1,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_birth",
        lambda _pid: controller_birth,
    )
    cleanup = _retained_controller_cleanup()

    result = supervisor._reconcile_interrupted_database_portal_attempts_bound(
        program,
        owner_fence_held=True,
        managed_daemon_launch_lock_held=True,
        managed_daemon_cleanup=cleanup,
        trigger="supervisor_startup_prelaunch",
    )

    assert result["reconciled"] is True
    assert len(daemons) == 1
    assert len(daemons[0].controller_quiescence_receipts) == 2
    first, second = daemons[0].controller_quiescence_receipts
    assert first == second
    assert database_portal_controller_quiescence_receipt_valid(first)
    assert first["controller_process_birth"] == controller_birth.to_dict()
    assert first["owner_mutation_fence_held"] is True
    assert first["managed_daemon_launch_lock_held"] is True


@pytest.mark.parametrize(
    "validator_name",
    [
        "database_pctdd005_historical_successor_reconciliation_valid",
        "database_fenced_provider_historical_retained_reconciliation_valid",
    ],
)
def test_retained_recovery_rejects_legacy_or_wrong_result_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validator_name: str,
) -> None:
    supervisor, program, _events, daemons = (
        _lease_scoped_reconciliation_supervisor(tmp_path, monkeypatch)
    )
    monkeypatch.setattr(
        daemon_module,
        validator_name,
        lambda _value: False,
    )

    result = supervisor._reconcile_interrupted_database_portal_attempts_bound(
        program,
        owner_fence_held=True,
        managed_daemon_launch_lock_held=True,
        managed_daemon_cleanup=_retained_controller_cleanup(),
    )

    assert result["blocked"] is True
    assert result["safe_to_restart"] is False
    assert result["reason"] == "database_portal_retained_reconciliation_blocked"
    assert len(daemons) == 1
    assert all(
        receipt is not None
        for receipt in daemons[0].controller_quiescence_receipts
    )


def test_retained_recovery_checkout_capability_resets_when_reconcile_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor, program, events, daemons = _lease_scoped_reconciliation_supervisor(
        tmp_path,
        monkeypatch,
        fail_at="pctdd005_occurrence",
    )

    result = supervisor._reconcile_interrupted_database_portal_attempts_bound(
        program,
        owner_fence_held=True,
        managed_daemon_launch_lock_held=True,
        managed_daemon_cleanup=_retained_controller_cleanup(),
    )

    assert result["blocked"] is True
    assert result["reason"] == "database_portal_retained_reconciliation_failed"
    assert events[:6] == [
        "bind_false",
        "acquire_false",
        "pctdd005_orphan",
        "retained_orphan",
        "pctdd005_occurrence",
        "release_false",
    ]
    assert "retained_occurrence" not in events
    assert getattr(
        daemons[0]._database_portal_outer_authority_cas,
        "__database_portal_checkout_mutation_lease_held__",
        None,
    ) is False
