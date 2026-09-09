"""Runtime gates for the one-shot PCTDD-005 r26 successor authorization."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    link_payload_digest,
)
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
    DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS,
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


def _storage_schema_cid(transport_fingerprint: str) -> str:
    return link_payload_digest(
        transport_fingerprint,
        codec="dag-json",
    ).cid


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
    admission["owner_storage_schema_fingerprint"] = _storage_schema_cid(
        str(admission["owner_schema_fingerprint"])
    )
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

    historical_attempt = SimpleNamespace(
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
    historical_policy = (
        DatabasePortalExecutionBridge._retained_recovery_execution_policy(
            historical_attempt
        )
    )
    assert historical_policy is not None
    assert historical_policy["allow_pool"] is False
    assert historical_policy["seed_prior_attempt"] is False

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


def test_pctdd005_historical_execution_policy_rejects_cross_generation_splice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin, legacy_admission = _admission(monkeypatch)
    historical = dict(legacy_admission)
    historical["schema"] = DATABASE_PCTDD005_HISTORICAL_SUCCESSOR_ADMISSION_SCHEMA
    historical["historical_occurrence_authority_id"] = "sha256:" + "8" * 64
    historical["controller_quiescence_receipt_id"] = "sha256:" + "9" * 64
    historical["owner_storage_schema_fingerprint"] = _storage_schema_cid(
        str(historical["owner_schema_fingerprint"])
    )
    historical.pop("admission_id")
    historical["admission_id"] = _sha256(historical)
    consumption = dict(
        database_fenced_provider_historical_retained_consumption(
            admission=historical,
            attempt_id="attempt:pctdd005-historical",
            claim_id="claim:pctdd005-historical",
            lease_id="lease:pctdd005-historical",
            owner_session_id="owner:pctdd005-historical",
            attempt_number=7,
            fencing_token=7,
            fence_epoch=7,
            expected_task_revision=pin["blocked_task_revision"] + 1,
            expected_task_status="retrying",
            resulting_task_revision=pin["blocked_task_revision"] + 2,
            resulting_task_status="in_progress",
        )
    )
    spliced = SimpleNamespace(
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
                "retained_recovery_admission": legacy_admission,
                "retained_recovery_consumption": consumption,
            },
        },
    )
    with pytest.raises(
        DatabasePortalBridgeError,
        match="not an exact consumed chain",
    ):
        DatabasePortalExecutionBridge._retained_recovery_execution_policy(spliced)


def test_no_provider_rearm_uses_nested_verifier_when_attempt_row_is_missing() -> None:
    daemon = object.__new__(DatabaseImplementationDaemon)
    captured: dict[str, Any] = {}

    class _Bridge:
        def no_provider_dispatch_rearm_evidence(
            self,
            attempt: Any,
            *,
            outer_block_receipt: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            captured["attempt_id"] = attempt.attempt_id
            captured["phase"] = attempt.committed_phase
            captured["receipt"] = dict(outer_block_receipt)
            return {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "database-portal-deferred-provider-rearm-evidence@1"
                ),
                "evidence_id": "sha256:" + "a" * 64,
                "provider_dispatched": False,
            }

    daemon._database_portal_bridge = _Bridge()
    daemon.get_attempt = lambda _attempt_id: None
    daemon._task_execution_spec_cid = lambda _task: "spec:cid"
    daemon._database_portal_filesystem_quiesced_release_rearm_evidence = (
        lambda *_args, **_kwargs: None
    )
    task = SimpleNamespace(
        task_cid="baguqeeralebfcpvwg72mkrku5nngr6kuda22x6bqx257fi4w3ztelab56iza",
        task_alias="PCTDD-005",
        revision=29,
        status="blocked",
    )
    receipt = {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-retry-budget@1",
        "operation": "database_unknown_outcome_blocked",
        "reason": "callback_authority_incomplete_blocked",
        "attempt_id": "attempt:80318cb2b8384964a8b784520d9bba70",
        "claim_id": "claim:84143a5851a8491b849327790b4c3e01",
        "task_cid": task.task_cid,
        "attempt_number": 7,
        "owner_session_id": "embedded-store:5a477a1db9402e639fecebb83f5f0873",
        "fencing_token": 7,
        "fence_epoch": 7,
        "lease_id": "lease:db58ff48ac3b497eb60d9f261983622e",
        "validation_spec_cid": "baguqeeranlxkjuo6ekwqkrwfzj34nuo532hbhk2kb6cn2wzwvit2ioovw73q",
        "attempts_used": 2,
        "max_task_attempts": 2,
        "retry_exhausted": True,
        "forced_block": True,
        "authority_outcome": "unknown",
    }
    evidence = daemon._database_portal_no_provider_rearm_evidence(task, receipt)
    assert evidence is not None
    assert captured["attempt_id"] == receipt["attempt_id"]
    assert captured["phase"] == "failed"
    assert captured["receipt"]["attempt_number"] == 7


def test_historical_population_admits_successive_controller_generations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin, legacy = _admission(monkeypatch)
    def _hist(generation: int) -> dict[str, Any]:
        admission = dict(legacy)
        admission["schema"] = DATABASE_PCTDD005_HISTORICAL_SUCCESSOR_ADMISSION_SCHEMA
        admission["historical_occurrence_authority_id"] = "sha256:" + "8" * 64
        admission["controller_quiescence_receipt_id"] = "sha256:" + "9" * 64
        admission["owner_storage_schema_fingerprint"] = _storage_schema_cid(
            str(admission["owner_schema_fingerprint"])
        )
        admission["owner_live_generation"] = generation
        admission.pop("admission_id", None)
        admission["admission_id"] = _sha256(admission)
        return admission

    first = _hist(70)
    second = _hist(71)
    daemon = object.__new__(DatabaseImplementationDaemon)

    class _Source:
        def get(self, cid: str) -> Any:
            if cid != pin["task_cid"]:
                return None
            return SimpleNamespace(
                task_cid=pin["task_cid"],
                task_alias=pin["task_alias"],
                status="retrying",
                revision=int(pin["blocked_task_revision"]) + 1,
                body={"completion_receipt": {"retained_recovery_admission": second}},
            )

    daemon._task_source = _Source()
    monkeypatch.setattr(
        daemon_module,
        "_database_fenced_provider_historical_recovery_authority",
        lambda _value: {
            "admission_schema": DATABASE_PCTDD005_HISTORICAL_SUCCESSOR_ADMISSION_SCHEMA,
            "manifest_builder": lambda: {"ok": True},
            "manifest_validator": lambda _manifest: True,
            "pins": (pin,),
        },
    )
    monkeypatch.setattr(
        daemon_module,
        "database_fenced_provider_historical_retained_admission_valid",
        lambda _value: True,
    )
    assert daemon._retained_recovery_historical_population_durable_current(first) is True
    assert daemon._retained_recovery_historical_population_durable_current(second) is True


def test_historical_admission_fence_does_not_require_live_attempt_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        daemon_module,
        "_database_fenced_provider_any_retained_admission_valid",
        lambda _value: True,
    )
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon._retained_recovery_dispatch_fence_is_current = (
        lambda *_args, **_kwargs: pytest.fail("historical fence opened ART")
    )
    admission = {
        "schema": DATABASE_PCTDD005_HISTORICAL_SUCCESSOR_ADMISSION_SCHEMA,
    }
    assert (
        DatabaseImplementationDaemon._retained_recovery_admission_fence_is_current(
            daemon,
            admission,
            allowed_states=frozenset({"admitted"}),
        )
        is True
    )
    assert (
        DatabaseImplementationDaemon._retained_recovery_admission_fence_is_current(
            daemon,
            admission,
            allowed_states=frozenset({"admission_pending"}),
        )
        is False
    )


def test_database_portal_wait_for_wake_sleeps_the_caller_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(daemon_module.time, "sleep", slept.append)
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.wait_for_wake(20.0)
    assert slept == [20.0]


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
    replica_transport_fingerprint: str = "",
    replica_storage_fingerprint: str = "",
) -> tuple[
    PortalImplementationSupervisor,
    DatabaseProgramConfig,
    list[str],
    list[Any],
    list[dict[str, Any]],
]:
    """Build a no-I/O supervisor seam around the retained recovery lease."""

    repo = tmp_path / "repo"
    state = repo / "state"
    state.mkdir(parents=True)
    todo = repo / "todo.md"
    todo.write_text("# Tasks\n", encoding="utf-8")
    sealed_pins = (
        DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN,
        *DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS,
    )
    sealed_transport_fingerprints = {
        str(pin["owner_schema_fingerprint"]) for pin in sealed_pins
    }
    sealed_store_ids = {str(pin["owner_store_id"]) for pin in sealed_pins}
    sealed_database_uuids = {
        str(pin["owner_database_uuid"]) for pin in sealed_pins
    }
    assert len(sealed_transport_fingerprints) == 1
    assert len(sealed_store_ids) == 1
    assert len(sealed_database_uuids) == 1
    transport_fingerprint = replica_transport_fingerprint or next(
        iter(sealed_transport_fingerprints)
    )
    storage_fingerprint = replica_storage_fingerprint or _storage_schema_cid(
        next(iter(sealed_transport_fingerprints))
    )
    program = DatabaseProgramConfig(
        authority_mode="quack",
        task_source_kind="duckdb",
        endpoint_secret_handle="env://PCTDD005_TEST_QUACK_TOKEN",
        quack_endpoint="quack://127.0.0.1:41307",
        store_id=next(iter(sealed_store_ids)),
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
    owner_intent_bindings: list[dict[str, Any]] = []
    mutation_owner_binding = {
        "server_id": "server:retained-schema-separation",
        "store_id": program.store_id,
        "database_uuid": next(iter(sealed_database_uuids)),
        "schema_revision": 1,
        "schema_fingerprint": storage_fingerprint,
        "generation": 69,
        "process_birth_id": "birth:retained-schema-separation",
        "listen_uri": program.quack_endpoint,
        "extension_fingerprint": "sha256:" + "e" * 64,
    }
    raw_owner_binding = {
        **mutation_owner_binding,
        "read_replica": {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "read-replica-observation@1"
            ),
            "authority": "non_authoritative_read_replica",
            "path": "/sealed/read-replica.duckdb",
            "source_database_path": "/sealed/control.duckdb",
            "server_id": mutation_owner_binding["server_id"],
            "database_uuid": mutation_owner_binding["database_uuid"],
            "generation": mutation_owner_binding["generation"],
            "schema_revision": mutation_owner_binding["schema_revision"],
            "schema_fingerprint": transport_fingerprint,
            "storage_schema_fingerprint": storage_fingerprint,
            "sha256": "sha256:" + "d" * 64,
            "size_bytes": 4096,
            "refresh_sequence": 69,
            "refreshed_at_ms": 1_700_000_000_069,
            "live": True,
        },
    }

    class FakeOwnerIntent:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            owner_intent_bindings.append(dict(kwargs["owner_binding"]))

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
            self.initialization = dict(kwargs)
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
        lambda **_kwargs: (
            "opaque-test-handle",
            dict(raw_owner_binding),
        ),
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
    return supervisor, program, events, daemons, owner_intent_bindings


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
    supervisor, program, events, daemons, _owner_bindings = (
        _lease_scoped_reconciliation_supervisor(tmp_path, monkeypatch)
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
    supervisor, program, events, daemons, _owner_bindings = (
        _lease_scoped_reconciliation_supervisor(tmp_path, monkeypatch)
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
    supervisor, program, events, daemons, _owner_bindings = (
        _lease_scoped_reconciliation_supervisor(tmp_path, monkeypatch)
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
    supervisor, program, _events, daemons, _owner_bindings = (
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


def test_retained_recovery_separates_transport_schema_from_storage_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor, program, _events, daemons, owner_bindings = (
        _lease_scoped_reconciliation_supervisor(tmp_path, monkeypatch)
    )

    result = supervisor._reconcile_interrupted_database_portal_attempts_bound(
        program,
        owner_fence_held=True,
        managed_daemon_launch_lock_held=True,
        managed_daemon_cleanup=_retained_controller_cleanup(),
    )

    assert result["reconciled"] is True
    assert len(daemons) == 1
    assert len(owner_bindings) == 1
    transport_fingerprint = str(
        DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN[
            "owner_schema_fingerprint"
        ]
    )
    mutation_binding = owner_bindings[0]
    control_binding = daemons[0].initialization[
        "authenticated_control_store_binding"
    ]
    expected_profile = daemons[0].initialization[
        "expected_control_schema_profile"
    ]
    assert mutation_binding["schema_fingerprint"].startswith("baguqeera")
    assert mutation_binding["schema_fingerprint"] == (
        "baguqeerah2s7odhlvt7hjaaxzfkax6dqcydviq7uztbtg5xmbilz5vlw4gia"
    )
    assert control_binding == mutation_binding
    assert (
        expected_profile["storage_schema_fingerprint"]
        == mutation_binding["schema_fingerprint"]
    )
    assert expected_profile["transport_schema_fingerprint"] == (
        transport_fingerprint
    )
    verification = daemon_module._database_quack_control_schema_verification(
        transport_schema_revision=program.schema_revision,
        control_store_id=program.store_id,
        control_store_generation=program.store_generation,
        authenticated_owner_binding=control_binding,
        expected_profile=expected_profile,
    )
    assert verification["valid"] is True
    assert verification["schema_fingerprint"] == transport_fingerprint


@pytest.mark.parametrize("crossed_storage", [False, True])
def test_historical_outer_cas_bridges_transport_digest_to_storage_cid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crossed_storage: bool,
) -> None:
    transport_fingerprint = str(
        DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN[
            "owner_schema_fingerprint"
        ]
    )
    storage_fingerprint = (
        _storage_schema_cid("sha256:" + "f" * 64)
        if crossed_storage
        else _storage_schema_cid(transport_fingerprint)
    )
    supervisor, program, _events, _daemons, _owner_bindings = (
        _lease_scoped_reconciliation_supervisor(
            tmp_path,
            monkeypatch,
            replica_storage_fingerprint=storage_fingerprint,
        )
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        retained_recovery_contracts,
    )

    monkeypatch.setattr(
        retained_recovery_contracts,
        "database_portal_controller_quiescence_receipt_valid",
        lambda _value: True,
    )
    monkeypatch.setattr(
        retained_recovery_contracts,
        "database_fenced_provider_historical_occurrence_authority_valid",
        lambda _value: True,
    )
    controller_birth = {
        "pid": 501,
        "start_time_ticks": 31,
        "boot_id": "boot:pctdd-schema-bridge",
        "parent_pid": 500,
    }
    monkeypatch.setattr(
        supervisor_module,
        "read_process_birth",
        lambda _pid: SimpleNamespace(
            to_dict=lambda: dict(controller_birth)
        ),
    )

    _secret, raw_owner = duckdb_state_module._resolve_quack_token_handle(
        uri=program.quack_endpoint
    )
    owner_binding = supervisor._normalized_quack_owner_binding(raw_owner)
    events: list[str] = []

    class Connection:
        def __init__(self) -> None:
            self._quack_mutation_binding = dict(owner_binding)
            self.in_transaction = False

        def execute(self, statement: str, _parameters: Any = None) -> Any:
            events.append(statement)
            if statement == "BEGIN TRANSACTION":
                self.in_transaction = True
            elif statement in {"COMMIT", "ROLLBACK"}:
                self.in_transaction = False
            return self

        def close(self) -> None:
            events.append("close")

    connection = Connection()
    monkeypatch.setattr(
        duckdb_state_module,
        "open_quack_transport_connection",
        lambda *_args, **_kwargs: connection,
    )

    class OuterTaskSource:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def fenced_provider_outer_authority_population_receipt(
            self,
            **observed_subject: Any,
        ) -> dict[str, Any]:
            events.append("receipt")
            return {
                "authority": {
                    "owner_binding": dict(owner_binding),
                    "read_replica_observation": dict(
                        observed_subject["controller_replica_observation"]
                    ),
                    "mutation_barrier": dict(
                        observed_subject["controller_mutation_barrier"]
                    ),
                }
            }

    monkeypatch.setattr(
        database_task_source_module,
        "DatabaseTaskSource",
        OuterTaskSource,
    )
    monkeypatch.setattr(
        intent_repository_module,
        "fenced_provider_outer_authority_population_receipt_valid",
        lambda _value: True,
    )

    def apply_cas(
        _repository: Any,
        actual_connection: Any,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        assert actual_connection is connection
        events.append("cas")
        return SimpleNamespace(changed=True, revision=1)

    monkeypatch.setattr(
        intent_repository_module.IntentRepository,
        "_cas_task_status_on_connection",
        apply_cas,
    )
    controller_receipt_id = "sha256:" + "9" * 64
    historical_authority = {
        "controller_quiescence_receipt_id": controller_receipt_id,
    }
    controller_receipt = {
        "receipt_id": controller_receipt_id,
        "board_namespace": supervisor.board_namespace,
        "state_prefix": supervisor.config.state_prefix,
        "owner_store_id": program.store_id,
        "control_store_generation": program.store_generation,
        "controller_process_birth": controller_birth,
    }
    subject = {
        "task_cid": str(
            DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN["task_cid"]
        ),
        "task_alias": "PCTDD-005",
        "task_revision": int(
            DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN[
                "blocked_task_revision"
            ]
        ),
        "expected_task_status": "blocked",
        "attempt_id": str(
            DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN[
                "predecessor_attempt_id"
            ]
        ),
        "claim_id": str(
            DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN[
                "predecessor_claim_id"
            ]
        ),
        "owner_session_id": str(
            DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN[
                "predecessor_owner_session_id"
            ]
        ),
        "fencing_token": int(
            DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN[
                "predecessor_fencing_token"
            ]
        ),
        "fence_epoch": int(
            DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN[
                "predecessor_fence_epoch"
            ]
        ),
        "expected_store_id": program.store_id,
        "minimum_store_generation": int(
            DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN[
                "owner_generation_floor"
            ]
        ),
        "expected_database_uuid": str(
            DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN[
                "owner_database_uuid"
            ]
        ),
        "expected_schema_fingerprint": transport_fingerprint,
        "receipt_nonce": str(
            DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN["receipt_nonce"]
        ),
        "receipt_epoch": int(
            DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN["receipt_epoch"]
        ),
        "historical_occurrence_authority": historical_authority,
        "controller_quiescence_receipt": controller_receipt,
    }

    if crossed_storage:
        with pytest.raises(
            RuntimeError,
            match="outside the sealed store lineage",
        ):
            supervisor._database_portal_execute_with_fenced_provider_outer_authority_cas_fenced(
                program,
                subject=subject,
                callback=lambda *_args: None,
            )
        assert events == []
    else:
        def consume(_receipt: Any, pinned: Any) -> str:
            events.append("callback")
            pinned.cas_task_status(
                task_cid=str(subject["task_cid"]),
                expected_revision=int(subject["task_revision"]),
                new_status="retrying",
            )
            return "committed"

        assert (
            supervisor._database_portal_execute_with_fenced_provider_outer_authority_cas_fenced(
                program,
                subject=subject,
                callback=consume,
            )
            == "committed"
        )
        assert events == [
            "BEGIN TRANSACTION",
            "receipt",
            "callback",
            "cas",
            "COMMIT",
            "close",
        ]


def test_retained_recovery_rejects_crossed_transport_schema_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor, program, _events, daemons, owner_bindings = (
        _lease_scoped_reconciliation_supervisor(
            tmp_path,
            monkeypatch,
            replica_transport_fingerprint="sha256:" + "f" * 64,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="transport/storage schema bindings",
    ):
        supervisor._reconcile_interrupted_database_portal_attempts_bound(
            program,
            owner_fence_held=True,
            managed_daemon_launch_lock_held=True,
            managed_daemon_cleanup=_retained_controller_cleanup(),
        )

    assert daemons == []
    assert owner_bindings == []


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
    supervisor, program, _events, daemons, _owner_bindings = (
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
    supervisor, program, events, daemons, _owner_bindings = (
        _lease_scoped_reconciliation_supervisor(
            tmp_path,
            monkeypatch,
            fail_at="pctdd005_occurrence",
        )
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


def test_retained_reconciliation_validator_admits_generic_rearm_retrying() -> None:
    """Home-lane prelaunch must accept extra-gate generic rearm as current."""

    pins = tuple(bridge_module.DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS)
    outcomes = [
        {
            "task_alias": pin["task_alias"],
            "task_cid": pin["task_cid"],
            "reconciled": True,
            "blocked": False,
            "reason": "retained_occurrence_already_rearmed",
            "admission_id": "",
            "consumption_id": "",
        }
        for pin in pins
    ]
    value = {
        "schema": daemon_module.DATABASE_FENCED_PROVIDER_HISTORICAL_RETAINED_RECONCILIATION_SCHEMA,
        "attempted": True,
        "reconciled": True,
        "blocked": False,
        "reason": "retained_occurrence_reconciliation_complete",
        "expected_occurrence_count": len(pins),
        "admitted_count": 0,
        "already_consumed_count": len(pins),
        "outcomes": outcomes,
    }
    assert daemon_module._database_fenced_provider_reconciliation_valid(
        value,
        schema=daemon_module.DATABASE_FENCED_PROVIDER_HISTORICAL_RETAINED_RECONCILIATION_SCHEMA,
        pins=pins,
    )
    # Extra-gate aliases still cannot mint restart authority from a blocked
    # prelaunch result; this only admits an already-retrying generic rearm.
    assert not supervisor_module.PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )
