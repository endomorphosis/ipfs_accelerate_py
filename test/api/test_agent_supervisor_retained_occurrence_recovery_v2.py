"""Fail-closed coverage for the additive PCTDD retained-occurrence recovery.

The ``@2`` path is deliberately separate from the historical unpublished
provider migration.  These tests pin the legacy bytes, exercise the compact
manifest -> credit -> receipts -> admission -> consumption chain, and prove
that the one-shot credit is consumed by the canonical task CAS before local
attempt or provider work can begin.
"""

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
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinationNotReadyError,
    DatabaseCoordinationStaleFenceError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    intent_repository as intent_repository_module,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    FENCED_PROVIDER_OUTER_AUTHORITY_POPULATION_RECEIPT_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as supervisor_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    database_portal_bridge as database_portal_bridge_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_FENCED_PROVIDER_RETAINED_CREDIT_SCHEMA,
    DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_ID,
    DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS,
    DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_SCHEMA,
    DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_ID,
    DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_SCHEMA,
    DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_PINS,
    database_fenced_provider_retained_credit,
    database_fenced_provider_retained_manifest,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATABASE_FENCED_PROVIDER_INNER_POPULATION_RECEIPT_SCHEMA,
    DATABASE_FENCED_PROVIDER_RETAINED_ADMISSION_SCHEMA,
    DATABASE_FENCED_PROVIDER_RETAINED_CONSUMPTION_SCHEMA,
    DATABASE_RETRY_BUDGET_SCHEMA,
    DatabaseImplementationConflictError,
    database_fenced_provider_retained_admission,
    database_fenced_provider_retained_admission_valid,
    database_fenced_provider_retained_consumption,
    database_fenced_provider_retained_consumption_valid,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
)
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
)

_OUTER_QUERY_PROFILE_ID = (
    "sha256:d38e766d6d5cde6ae0aee54c3ff731db7be31aea1826b106400fe1492682f19d"
)
_INNER_QUERY_PROFILE_ID = (
    "sha256:1efeeb4904696fc50e0e0e8508fb131d759cb4566366d39ed5d44d2b27b12685"
)
_LEGACY_PIN_POPULATION_DIGEST = (
    "sha256:f938bc2122bbf56cf101c4bad58a8e37d43ed55a2a0a5d52b51155ee6b6bb762"
)
_OCCURRENCE_FIELDS = frozenset(
    {
        "task_cid",
        "task_alias",
        "board_namespace",
        "blocked_task_revision",
        "blocked_task_status",
        "predecessor_attempt_id",
        "predecessor_claim_id",
        "predecessor_lease_id",
        "predecessor_owner_session_id",
        "predecessor_attempt_number",
        "predecessor_fencing_token",
        "predecessor_fence_epoch",
        "predecessor_branch",
        "recovery_mode",
        "candidate_disposition",
        "disposition_repository_root",
        "disposition_git_common_dir",
        "disposition_baseline_ref",
        "source_repository_root",
        "source_git_common_dir",
        "source_relative_path",
        "clean_baseline_ref",
        "retained_ref",
        "retained_commit",
        "retained_worktree_path",
        "receipt_nonce",
        "receipt_epoch",
        "inner_query_profile_id",
        "outer_query_profile_id",
        "owner_store_id",
        "owner_generation_floor",
        "owner_database_uuid",
        "owner_schema_fingerprint",
        "control_store_generation",
        "owner_schema_revision",
        "allow_pool",
        "seed_prior_attempt",
        "one_shot",
        "retry_policy",
        "credit_ordinal",
    }
)


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _legacy_manifest() -> dict[str, Any]:
    return {
        "schema": (
            DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_SCHEMA
        ),
        "revision": "pctdd-provider-recovery-2026-09-02",
        "operator_owned": True,
        "one_shot": True,
        "occurrences": [
            dict(item)
            for item in DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_PINS
        ],
    }


def _pin(task_alias: str) -> dict[str, Any]:
    matches = [
        dict(item)
        for item in DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS
        if item.get("task_alias") == task_alias
    ]
    assert len(matches) == 1
    return matches[0]


def _credit_id(credit: Mapping[str, Any]) -> str:
    return _sha256(dict(credit))


def _fake_receipts(
    occurrence: Mapping[str, Any],
    credit: Mapping[str, Any],
    *,
    live_generation: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create closed binding projections; leaf validity is injected per test."""

    manifest_id = DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_ID
    credit_id = _credit_id(credit)
    observed_generation = (
        occurrence["owner_generation_floor"]
        if live_generation is None
        else live_generation
    )
    common = {
        "task_cid": occurrence["task_cid"],
        "task_alias": occurrence["task_alias"],
        "task_revision": occurrence["blocked_task_revision"],
        "attempt_id": occurrence["predecessor_attempt_id"],
        "claim_id": occurrence["predecessor_claim_id"],
        "owner_session_id": occurrence["predecessor_owner_session_id"],
        "fencing_token": occurrence["predecessor_fencing_token"],
        "fence_epoch": occurrence["predecessor_fence_epoch"],
    }
    inner = {
        "schema": DATABASE_FENCED_PROVIDER_INNER_POPULATION_RECEIPT_SCHEMA,
        "query_profile_id": occurrence["inner_query_profile_id"],
        "receipt_nonce": occurrence["receipt_nonce"],
        "receipt_epoch": occurrence["receipt_epoch"],
        "authority": {
            "authority_mode": "quack",
            "control_store_id": occurrence["owner_store_id"],
            "control_store_generation": occurrence[
                "control_store_generation"
            ],
        },
        "subject": {
            **common,
            "lease_id": occurrence["predecessor_lease_id"],
            "attempt_number": occurrence["predecessor_attempt_number"],
            "recovery_manifest_id": manifest_id,
            "recovery_credit_id": credit_id,
        },
        "groups": {
            "provider_invocations": {"count": 0, "rows": []},
            "effect_claims": {"count": 0, "rows": []},
        },
        "coordinator_receipt_cid": "sha256:" + "0" * 64,
        "receipt_cid": "sha256:" + "1" * 64,
    }
    outer = {
        "schema": FENCED_PROVIDER_OUTER_AUTHORITY_POPULATION_RECEIPT_SCHEMA,
        "query_profile_id": occurrence["outer_query_profile_id"],
        "receipt_nonce": occurrence["receipt_nonce"],
        "receipt_epoch": occurrence["receipt_epoch"],
        "subject": {
            **common,
            "expected_task_status": occurrence["blocked_task_status"],
            "expected_store_id": occurrence["owner_store_id"],
            "expected_store_generation": observed_generation,
        },
        "cross_store_context": {
            "coordination_lease_id": occurrence["predecessor_lease_id"],
        },
        "authority": {
            "owner_binding": {
                "store_id": occurrence["owner_store_id"],
                "generation": observed_generation,
                "database_uuid": occurrence["owner_database_uuid"],
                "schema_fingerprint": occurrence[
                    "owner_schema_fingerprint"
                ],
                "schema_revision": occurrence["owner_schema_revision"],
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


def _admission(
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_alias: str = "PCTDD-006",
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    occurrence = _pin(task_alias)
    credit = dict(database_fenced_provider_retained_credit(occurrence))
    inner, outer = _fake_receipts(occurrence, credit)
    # Leaf receipt construction has its own real-Quack regression suites.  This
    # suite isolates the successor's cross-receipt admission semantics while
    # still requiring both production leaf validators to be invoked.
    observed: list[str] = []

    def inner_valid(value: Any) -> bool:
        observed.append("inner")
        return isinstance(value, Mapping)

    def outer_valid(value: Any) -> bool:
        observed.append("outer")
        return isinstance(value, Mapping)

    monkeypatch.setattr(
        daemon_module,
        "database_fenced_provider_inner_population_receipt_valid",
        inner_valid,
    )
    monkeypatch.setattr(
        daemon_module,
        "fenced_provider_outer_authority_population_receipt_valid",
        outer_valid,
        raising=False,
    )
    monkeypatch.setattr(
        intent_repository_module,
        "fenced_provider_outer_authority_population_receipt_valid",
        outer_valid,
    )
    admission = dict(
        database_fenced_provider_retained_admission(
            occurrence=occurrence,
            credit=credit,
            inner_receipt=inner,
            outer_receipt=outer,
            expected_task_revision=occurrence["blocked_task_revision"],
            expected_task_status=occurrence["blocked_task_status"],
        )
    )
    assert observed == ["inner", "outer"]
    return occurrence, credit, inner, outer, admission


def _assert_admission_rejected(**kwargs: Any) -> None:
    try:
        result = database_fenced_provider_retained_admission(**kwargs)
    except (TypeError, ValueError, DatabaseImplementationConflictError):
        return
    assert result is None, "invalid retained admission did not fail closed"


def _walk_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _walk_mappings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_mappings(item)


def test_legacy_unpublished_manifest_bytes_and_id_remain_unchanged() -> None:
    manifest = _legacy_manifest()
    assert DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/"
        "fenced-provider-unpublished-migration-manifest@1"
    )
    assert DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_ID == (
        "sha256:3b4e8c471c67839e4ce5e45596065d02a0da180bb8a617c0f7cc1b5ae48bbbe0"
    )
    assert _sha256(manifest) == (
        DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_ID
    )
    assert _sha256(manifest["occurrences"]) == _LEGACY_PIN_POPULATION_DIGEST
    assert [item["task_revision"] for item in manifest["occurrences"]] == [
        23,
        23,
        40,
    ]
    assert [item["attempt_id"] for item in manifest["occurrences"]] == [
        "attempt:266a0841fde1462291b45816376f2b3a",
        "attempt:2b37a94ab4e54b14b27afd1577f70a47",
        "attempt:228fff0dbc7644bf953672d3945abdd8",
    ]


def test_retained_manifest_is_closed_separate_and_pins_exact_dispositions() -> None:
    manifest = dict(database_fenced_provider_retained_manifest())
    pins = [dict(item) for item in DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS]

    assert DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/"
        "database-fenced-provider-no-accepted-publication-manifest@2"
    )
    assert DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_SCHEMA != (
        DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_SCHEMA
    )
    assert manifest["schema"] == DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_SCHEMA
    assert manifest["occurrences"] == pins
    assert _sha256(manifest) == DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_ID
    assert {item["task_alias"] for item in pins} == {
        "PCTDD-006",
        "PCTDD-007",
        "PCTDD-034",
    }
    assert all(set(item) == _OCCURRENCE_FIELDS for item in pins)
    assert all(
        item["board_namespace"] == ("parallel-content-sealing-proof-carrying-tdd-v1")
        for item in pins
    )
    assert all(item["blocked_task_status"] == "blocked" for item in pins)
    assert {item["task_alias"]: item["blocked_task_revision"] for item in pins} == {
        "PCTDD-006": 29,
        "PCTDD-007": 29,
        "PCTDD-034": 48,
    }
    assert {item["task_alias"]: item["predecessor_branch"] for item in pins} == {
        "PCTDD-006": ("implementation/pctdd-006-0c4324168ce4-attempt-1-1788326381"),
        "PCTDD-007": ("implementation/pctdd-007-e272ca1f906f-attempt-1-1788327085"),
        "PCTDD-034": ("implementation/pctdd-034-6cc49f41f873-attempt-1-1788325977"),
    }
    assert all(item["owner_generation_floor"] == 58 for item in pins)
    assert all(item["control_store_generation"] == "pctdd-v1-g9" for item in pins)
    assert all(item["owner_schema_revision"] == 1 for item in pins)
    assert all(
        item["inner_query_profile_id"] == _INNER_QUERY_PROFILE_ID for item in pins
    )
    assert all(
        item["outer_query_profile_id"] == _OUTER_QUERY_PROFILE_ID for item in pins
    )
    assert all(item["credit_ordinal"] == 1 for item in pins)
    assert all(item["one_shot"] is True for item in pins)
    assert all(item["allow_pool"] is False for item in pins)
    assert all(item["seed_prior_attempt"] is False for item in pins)
    assert all(
        type(item["retry_policy"]) is str and item["retry_policy"] for item in pins
    )

    p006 = _pin("PCTDD-006")
    assert p006["recovery_mode"] == "runner_fenced_clean_removed"
    assert p006["candidate_disposition"] == "clean_removed"
    assert p006["retained_ref"] == ""
    assert p006["retained_commit"] == ""
    assert p006["retained_worktree_path"] == ""
    for alias in ("PCTDD-007", "PCTDD-034"):
        pin = _pin(alias)
        assert pin["recovery_mode"] == "unpublished_rescue_quarantined"
        assert pin["candidate_disposition"] == "rescue_quarantined"
        assert str(pin["retained_ref"]).startswith("refs/")
        assert len(str(pin["retained_commit"])) == 40


def test_retained_manifest_uses_explicit_base_namespace_with_g9_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_namespace = "parallel-content-sealing-proof-carrying-tdd-v1"
    g9_namespace = "parallel-content-sealing-proof-carrying-tdd-v1-g9"
    merge_branch = f"agent/{g9_namespace}"
    todo_path = tmp_path / "parallel_content_sealing_proof_carrying_tdd.todo.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    config = supervisor_module.PortalSupervisorConfig(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        state_dir=state_dir,
        task_prefix="## PCTDD-",
        board_namespace=base_namespace,
        merge_target_branch=merge_branch,
        repo_root=tmp_path,
    )
    monkeypatch.setattr(
        supervisor_module,
        "isolate_board_runtime",
        lambda **_kwargs: {
            "implementation_branch": merge_branch,
            "branch_result": {"reason": "already_exists"},
        },
    )
    supervisor = PortalImplementationSupervisor(config)
    assert supervisor.board_namespace == base_namespace
    command = supervisor._build_daemon_command()
    namespace_index = command.index("--board-namespace")
    assert command[namespace_index + 1] == base_namespace
    assert supervisor._managed_daemon_matches_command_line(" ".join(command))
    assert supervisor._managed_daemon_command_belongs_to_scope(command)
    foreign_command = list(command)
    foreign_command[namespace_index + 1] = g9_namespace
    assert not supervisor._managed_daemon_matches_command_line(
        " ".join(foreign_command)
    )
    assert not supervisor._managed_daemon_command_belongs_to_scope(
        foreign_command
    )

    patched_pins = tuple(
        {
            **dict(item),
            "disposition_repository_root": str(tmp_path),
        }
        for item in DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS
    )
    monkeypatch.setattr(
        database_portal_bridge_module,
        "DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS",
        patched_pins,
    )
    program = SimpleNamespace(
        store_id=(
            "data/agent_supervisor/"
            "parallel_content_sealing_proof_carrying_tdd_v1_g9/"
            "control.duckdb"
        ),
        store_generation="pctdd-v1-g9",
    )
    assert supervisor._retained_fenced_provider_program_applicable(program)

    supervisor.config.task_prefix = "## PCTDD-X"
    assert not supervisor._retained_fenced_provider_program_applicable(program)
    supervisor.config.task_prefix = "## PCTDD-"
    supervisor.board_namespace = g9_namespace
    assert not supervisor._retained_fenced_provider_program_applicable(program)


def test_credit_is_acyclic_closed_and_rejects_unknown_or_cross_pin_data() -> None:
    occurrence = _pin("PCTDD-006")
    credit = dict(database_fenced_provider_retained_credit(occurrence))
    assert set(credit) == {"schema", "manifest_id", "occurrence"}
    assert credit["schema"] == DATABASE_FENCED_PROVIDER_RETAINED_CREDIT_SCHEMA
    assert credit["manifest_id"] == DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_ID
    assert credit["occurrence"] == occurrence
    encoded = canonical_json(credit)
    assert "inner_receipt_cid" not in encoded
    assert "outer_receipt_cid" not in encoded
    assert "admission_id" not in encoded
    assert "consumption_id" not in encoded

    unknown = {**occurrence, "unreviewed": True}
    with pytest.raises((TypeError, ValueError, DatabaseImplementationConflictError)):
        database_fenced_provider_retained_credit(unknown)
    crossed = dict(occurrence)
    crossed["task_cid"] = _pin("PCTDD-007")["task_cid"]
    with pytest.raises((TypeError, ValueError, DatabaseImplementationConflictError)):
        database_fenced_provider_retained_credit(crossed)


def test_admission_binds_leaf_cids_without_recursively_persisting_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    occurrence, credit, inner, outer, admission = _admission(monkeypatch)
    assert admission["schema"] == DATABASE_FENCED_PROVIDER_RETAINED_ADMISSION_SCHEMA
    assert database_fenced_provider_retained_admission_valid(admission)
    assert admission["manifest_id"] == DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_ID
    assert admission["credit_id"] == _credit_id(credit)
    assert admission["inner_receipt_cid"] == inner["receipt_cid"]
    assert admission["coordinator_receipt_cid"] == inner["coordinator_receipt_cid"]
    assert admission["outer_receipt_cid"] == outer["receipt_cid"]
    assert admission["receipt_nonce"] == occurrence["receipt_nonce"]
    assert admission["receipt_epoch"] == occurrence["receipt_epoch"]
    assert admission["expected_task_revision"] == occurrence["blocked_task_revision"]
    assert admission["expected_task_status"] == "blocked"
    assert admission["predecessor_branch"] == occurrence["predecessor_branch"]
    assert admission["owner_store_id"] == occurrence["owner_store_id"]
    assert admission["owner_live_generation"] == 58

    assert all(
        node.get("schema")
        not in {
            DATABASE_FENCED_PROVIDER_INNER_POPULATION_RECEIPT_SCHEMA,
            FENCED_PROVIDER_OUTER_AUTHORITY_POPULATION_RECEIPT_SCHEMA,
        }
        for node in _walk_mappings(admission)
    )
    assert "groups" not in canonical_json(admission)
    tampered = dict(admission)
    tampered["owner_live_generation"] = 59
    assert not database_fenced_provider_retained_admission_valid(tampered)


@pytest.mark.parametrize(
    ("target", "field", "replacement"),
    [
        ("inner", "receipt_nonce", "nonce:crossed"),
        ("outer", "receipt_epoch", 2),
        ("inner", "query_profile_id", "sha256:" + "3" * 64),
        ("outer", "query_profile_id", "sha256:" + "4" * 64),
        ("outer_subject", "task_revision", 30),
        ("outer_subject", "expected_task_status", "ready"),
        ("outer_owner", "generation", 59),
    ],
)
def test_admission_rejects_cross_receipt_and_current_authority_drift(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    field: str,
    replacement: Any,
) -> None:
    occurrence = _pin("PCTDD-006")
    credit = dict(database_fenced_provider_retained_credit(occurrence))
    inner, outer = _fake_receipts(occurrence, credit)
    monkeypatch.setattr(
        daemon_module,
        "database_fenced_provider_inner_population_receipt_valid",
        lambda value: isinstance(value, Mapping),
    )
    monkeypatch.setattr(
        daemon_module,
        "fenced_provider_outer_authority_population_receipt_valid",
        lambda value: isinstance(value, Mapping),
        raising=False,
    )
    monkeypatch.setattr(
        intent_repository_module,
        "fenced_provider_outer_authority_population_receipt_valid",
        lambda value: isinstance(value, Mapping),
    )
    if target == "inner":
        inner[field] = replacement
    elif target == "outer":
        outer[field] = replacement
    elif target == "outer_subject":
        outer["subject"][field] = replacement
    else:
        outer["authority"]["owner_binding"][field] = replacement
    _assert_admission_rejected(
        occurrence=occurrence,
        credit=credit,
        inner_receipt=inner,
        outer_receipt=outer,
        expected_task_revision=occurrence["blocked_task_revision"],
        expected_task_status=occurrence["blocked_task_status"],
    )


def test_consumption_is_one_shot_and_binds_the_resulting_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    occurrence, _credit, _inner, _outer, admission = _admission(monkeypatch)
    consumption = dict(
        database_fenced_provider_retained_consumption(
            admission=admission,
            attempt_id="attempt:successor",
            claim_id="claim:successor",
            lease_id="lease:successor",
            owner_session_id="owner:successor",
            attempt_number=12,
            fencing_token=12,
            fence_epoch=12,
            expected_task_revision=occurrence["blocked_task_revision"] + 1,
            expected_task_status="retrying",
            resulting_task_revision=occurrence["blocked_task_revision"] + 2,
            resulting_task_status="in_progress",
        )
    )
    assert consumption["schema"] == DATABASE_FENCED_PROVIDER_RETAINED_CONSUMPTION_SCHEMA
    assert database_fenced_provider_retained_consumption_valid(consumption)
    assert consumption["admission_id"] == admission["admission_id"]
    assert consumption["expected_task_status"] == "retrying"
    assert consumption["resulting_task_status"] == "in_progress"
    assert consumption["resulting_task_revision"] == (
        consumption["expected_task_revision"] + 1
    )
    assert "admission" not in consumption
    assert "inner_receipt" not in consumption
    assert "outer_receipt" not in consumption

    tampered = dict(consumption)
    tampered["claim_id"] = "claim:crossed"
    assert not database_fenced_provider_retained_consumption_valid(tampered)
    with pytest.raises((TypeError, ValueError, DatabaseImplementationConflictError)):
        database_fenced_provider_retained_consumption(
            admission=admission,
            attempt_id="attempt:second-use",
            claim_id="claim:second-use",
            lease_id="lease:second-use",
            owner_session_id="owner:second-use",
            attempt_number=13,
            fencing_token=13,
            fence_epoch=13,
            expected_task_revision=occurrence["blocked_task_revision"] + 2,
            expected_task_status="in_progress",
            resulting_task_revision=occurrence["blocked_task_revision"] + 3,
            resulting_task_status="in_progress",
        )


def test_reload_claim_projection_keeps_retry_budget_distinct_from_claim_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    occurrence, _credit, _inner, _outer, admission = _admission(monkeypatch)
    consumption = dict(
        database_fenced_provider_retained_consumption(
            admission=admission,
            attempt_id="attempt:retained-reload",
            claim_id="claim:retained-reload",
            lease_id="lease:retained-reload",
            owner_session_id="owner:retained-reload",
            attempt_number=12,
            fencing_token=12,
            fence_epoch=12,
            expected_task_revision=occurrence["blocked_task_revision"] + 1,
            expected_task_status="retrying",
            resulting_task_revision=occurrence["blocked_task_revision"] + 2,
            resulting_task_status="in_progress",
        )
    )
    receipt = {
        "schema": DATABASE_RETRY_BUDGET_SCHEMA,
        "operation": "database_claim",
        "task_cid": occurrence["task_cid"],
        "validation_spec_cid": "sha256:" + "9" * 64,
        "attempts_used": 3,
        "max_task_attempts": 3,
        "retry_exhausted": True,
        "process_instance_id": "process:retained-reload",
        "owner_session_id": "owner:retained-reload",
        "attempt_id": "attempt:retained-reload",
        "claim_id": "claim:retained-reload",
        "lease_id": "lease:retained-reload",
        "attempt_number": 12,
        "fencing_token": 12,
        "fence_epoch": 12,
        "retained_recovery_admission": dict(admission),
        "retained_recovery_consumption": consumption,
    }
    task = SimpleNamespace(
        task_cid=occurrence["task_cid"],
        task_alias=occurrence["task_alias"],
        revision=occurrence["blocked_task_revision"] + 2,
        status="in_progress",
        body={"completion_receipt": receipt},
    )

    projected = PortalImplementationSupervisor._database_claim_attempt(task)

    assert projected.attempt_number == 12
    assert receipt["attempts_used"] == 3
    malformed = dict(receipt)
    malformed["attempts_used"] = 2
    task.body = {"completion_receipt": malformed}
    with pytest.raises(RuntimeError, match="retained retry budget"):
        PortalImplementationSupervisor._database_claim_attempt(task)


def _population_for(occurrence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "repository_tree_id": "tree:retained-occurrence-v2",
        "objectives": [
            {
                "objective_id": "objective:retained-occurrence-v2",
                "objective_alias": "PCTDD-O-RECOVERY",
                "title": "Retained occurrence recovery",
                "goal_cid": "goal:retained-occurrence-v2",
                "goal_alias": "PCTDD-G-RECOVERY",
                "status": "open",
            }
        ],
        "tasks": [
            {
                "task_cid": occurrence["task_cid"],
                "task_id": occurrence["task_alias"],
                "goal_cid": "goal:retained-occurrence-v2",
                "status": (
                    "retrying"
                    if int(occurrence["blocked_task_revision"]) % 2 == 0
                    else "ready"
                ),
                "priority": "P0",
                "ordinal": 1,
                "title": "One-shot retained recovery",
                "validations": [["python", "-m", "pytest", "-q"]],
            }
        ],
    }


def _advance_to_blocked_revision(daemon: Any, occurrence: Mapping[str, Any]) -> Any:
    task = daemon.task_source.get(occurrence["task_cid"])
    assert task is not None
    target = occurrence["blocked_task_revision"]
    while int(task.revision) < target:
        next_status = "retrying" if task.status != "retrying" else "blocked"
        daemon._cas_task_status_database(
            task.task_cid,
            expected_revision=int(task.revision),
            new_status=next_status,
            receipt=daemon._retry_budget_receipt(
                task,
                attempts_used=0,
                operation="retained_recovery_test_revision_setup",
            ),
        )
        task = daemon.task_source.get(occurrence["task_cid"])
        assert task is not None
    assert task.status == "blocked"
    assert task.revision == target
    return task


def _arm_retrying_task(
    daemon: Any, occurrence: Mapping[str, Any], admission: Mapping[str, Any]
) -> Any:
    blocked = _advance_to_blocked_revision(daemon, occurrence)
    receipt = daemon._retry_budget_receipt(
        blocked,
        attempts_used=(
            max(0, int(daemon.max_task_attempts) - 1)
            if int(daemon.max_task_attempts) > 0
            else 0
        ),
        operation="database_fenced_provider_retained_rearmed",
    )
    receipt["retained_recovery_admission"] = dict(admission)
    daemon._cas_task_status_database(
        blocked.task_cid,
        expected_revision=int(blocked.revision),
        new_status="retrying",
        receipt=receipt,
    )
    retrying = daemon.task_source.get(blocked.task_cid)
    assert retrying is not None
    return retrying


def _allow_exact_synthetic_retained_admission(
    daemon: Any,
    monkeypatch: pytest.MonkeyPatch,
    admission: Mapping[str, Any],
) -> None:
    """Bypass only the predecessor-store prerequisite absent from this fixture.

    Production correctly requires an independently durable admitted dispatch
    fence for the predecessor attempt.  The focused claim-order tests do not
    synthesize that historical attempt store; they exercise the successor
    admission and canonical consumption boundary instead.  Preserve the real
    gate for every other task or admission.
    """

    real_forbidden = daemon._automatic_claim_forbidden_current
    expected = dict(admission)

    def exact_synthetic_gate(task: Any) -> bool:
        receipt = dict(getattr(task, "body", {}).get("completion_receipt") or {})
        candidate = receipt.get("retained_recovery_admission")
        if (
            candidate == expected
            and database_fenced_provider_retained_admission_valid(candidate)
            and str(getattr(task, "task_cid", "") or "") == expected["task_cid"]
            and str(getattr(task, "status", "") or "") == "retrying"
            and int(getattr(task, "revision", -1))
            == int(expected["expected_task_revision"]) + 1
        ):
            return False
        return real_forbidden(task)

    monkeypatch.setattr(
        daemon, "_automatic_claim_forbidden_current", exact_synthetic_gate
    )
    monkeypatch.setattr(
        daemon,
        "_retained_recovery_admission_is_current_for_task",
        lambda task, candidate, **_kwargs: bool(
            candidate == expected
            and str(getattr(task, "task_cid", "") or "")
            == expected["task_cid"]
            and str(getattr(task, "status", "") or "") == "retrying"
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_retained_recovery_aggregate_admitted_current",
        lambda: True,
    )
    monkeypatch.setattr(
        daemon,
        "_retained_recovery_admission_fence_is_current",
        lambda candidate, *, allowed_states: bool(
            candidate == expected and "admitted" in allowed_states
        ),
    )


def _strict_three_pin_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider_calls: list[str],
) -> tuple[Any, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    aliases = ("PCTDD-006", "PCTDD-007", "PCTDD-034")
    records = [_admission(monkeypatch, task_alias=alias) for alias in aliases]
    occurrences = [dict(record[0]) for record in records]
    admissions = {
        record[0]["task_cid"]: dict(record[4]) for record in records
    }
    daemon = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        max_task_attempts=3,
    )
    population = {
        "repository_tree_id": "tree:strict-retained-orphan-v2",
        "objectives": [
            {
                "objective_id": "objective:strict-retained-orphan-v2",
                "objective_alias": "PCTDD-O-STRICT-RECOVERY",
                "title": "Strict retained orphan recovery",
                "goal_cid": "goal:strict-retained-orphan-v2",
                "goal_alias": "PCTDD-G-STRICT-RECOVERY",
                "status": "open",
            }
        ],
        "tasks": [
            {
                "task_cid": occurrence["task_cid"],
                "task_id": occurrence["task_alias"],
                "goal_cid": "goal:strict-retained-orphan-v2",
                "status": (
                    "retrying"
                    if int(occurrence["blocked_task_revision"]) % 2 == 0
                    else "ready"
                ),
                "priority": "P0",
                "ordinal": index,
                "title": "Strict one-shot retained orphan",
                "validations": [["python", "-m", "pytest", "-q"]],
            }
            for index, occurrence in enumerate(occurrences, start=1)
        ],
    }
    daemon.materialize_population(population)
    for occurrence in occurrences:
        _arm_retrying_task(
            daemon,
            occurrence,
            admissions[occurrence["task_cid"]],
        )
    admitted_ids = {
        admission["admission_id"] for admission in admissions.values()
    }
    monkeypatch.setattr(
        daemon,
        "_retained_recovery_admission_fence_is_current",
        lambda candidate, *, allowed_states: bool(
            isinstance(candidate, Mapping)
            and candidate.get("admission_id") in admitted_ids
            and "admitted" in allowed_states
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_retained_recovery_aggregate_admitted_current",
        lambda: True,
    )
    owner_cas = lambda **_kwargs: None
    setattr(owner_cas, "__database_portal_owner_fence_held__", True)
    setattr(
        owner_cas,
        "__database_portal_checkout_mutation_lease_held__",
        True,
    )
    daemon._database_portal_outer_authority_cas = owner_cas
    daemon.authority_mode = "quack"
    daemon.control_store_id = occurrences[0]["owner_store_id"]
    daemon.control_store_generation = occurrences[0][
        "control_store_generation"
    ]
    return daemon, occurrences, admissions


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_claim_consumes_credit_before_attempt_insert_or_provider_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    occurrence, _credit, _inner, _outer, admission = _admission(monkeypatch)
    provider_calls: list[str] = []
    daemon = _open_daemon(tmp_path, provider_calls=provider_calls)
    try:
        daemon.materialize_population(_population_for(occurrence))
        retrying = _arm_retrying_task(daemon, occurrence, admission)
        assert retrying.status == "retrying"
        _allow_exact_synthetic_retained_admission(daemon, monkeypatch, admission)
        real_insert = daemon._insert_attempt_from_claim
        observed: list[str] = []

        def insert_after_consumption(claim: Any, **kwargs: Any):
            current = daemon.task_source.get(occurrence["task_cid"])
            assert current is not None
            assert current.status == "in_progress"
            receipt = dict(current.body["completion_receipt"])
            assert receipt["schema"] == DATABASE_RETRY_BUDGET_SCHEMA
            assert receipt["retained_recovery_admission"] == admission
            assert database_fenced_provider_retained_consumption_valid(
                receipt["retained_recovery_consumption"]
            )
            assert daemon.get_attempt(str(claim.attempt_id)) is None
            assert provider_calls == []
            observed.append("consumed_before_insert")
            return real_insert(claim, **kwargs)

        monkeypatch.setattr(
            daemon, "_insert_attempt_from_claim", insert_after_consumption
        )
        attempt = daemon.claim_next()
        assert attempt is not None
        assert observed == ["consumed_before_insert"]
        assert provider_calls == []
        assert attempt.body["retry_budget"]["retained_recovery_admission"] == admission
        assert database_fenced_provider_retained_consumption_valid(
            attempt.body["retry_budget"]["retained_recovery_consumption"]
        )
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_attempt_insert_failure_never_reopens_consumed_credit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    occurrence, _credit, _inner, _outer, admission = _admission(monkeypatch)
    provider_calls: list[str] = []
    daemon = _open_daemon(tmp_path, provider_calls=provider_calls)
    try:
        daemon.materialize_population(_population_for(occurrence))
        _arm_retrying_task(daemon, occurrence, admission)
        _allow_exact_synthetic_retained_admission(daemon, monkeypatch, admission)

        def fail_insert(claim: Any, **_kwargs: Any):
            current = daemon.task_source.get(occurrence["task_cid"])
            assert current is not None and current.status == "in_progress"
            receipt = dict(current.body["completion_receipt"])
            assert database_fenced_provider_retained_consumption_valid(
                receipt["retained_recovery_consumption"]
            )
            assert daemon.get_attempt(str(claim.attempt_id)) is None
            assert provider_calls == []
            raise RuntimeError("injected local attempt insert failure")

        monkeypatch.setattr(daemon, "_insert_attempt_from_claim", fail_insert)
        assert daemon.claim_next() is None

        current = daemon.task_source.get(occurrence["task_cid"])
        assert current is not None
        assert current.status == "in_progress"
        receipt = dict(current.body["completion_receipt"])
        consumption = receipt["retained_recovery_consumption"]
        assert database_fenced_provider_retained_consumption_valid(consumption)
        assert receipt["retained_recovery_admission"] == admission
        assert provider_calls == []
        assert all(
            node.get("schema")
            not in {
                DATABASE_FENCED_PROVIDER_INNER_POPULATION_RECEIPT_SCHEMA,
                FENCED_PROVIDER_OUTER_AUTHORITY_POPULATION_RECEIPT_SCHEMA,
            }
            for node in _walk_mappings(receipt)
        )

        # The orphan reconciler may retain the durable in-progress fence or
        # move it to a non-dispatchable blocked terminal, but it must never
        # recreate retrying authority from a consumed one-shot admission.
        daemon.reconcile_orphaned_canonical_claims()
        current = daemon.task_source.get(occurrence["task_cid"])
        assert current is not None
        assert current.status in {"in_progress", "blocked"}
        assert current.status != "retrying"
        assert daemon.claim_next() is None
        assert provider_calls == []
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_strict_orphan_repair_settles_latest_lease_inside_terminal_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliases = ("PCTDD-006", "PCTDD-007", "PCTDD-034")
    records = [_admission(monkeypatch, task_alias=alias) for alias in aliases]
    occurrences = [record[0] for record in records]
    admissions = {
        record[0]["task_cid"]: dict(record[4]) for record in records
    }
    population = {
        "repository_tree_id": "tree:retained-orphan-v2",
        "objectives": [
            {
                "objective_id": "objective:retained-orphan-v2",
                "objective_alias": "PCTDD-O-RECOVERY",
                "title": "Retained orphan recovery",
                "goal_cid": "goal:retained-orphan-v2",
                "goal_alias": "PCTDD-G-RECOVERY",
                "status": "open",
            }
        ],
        "tasks": [
            {
                "task_cid": occurrence["task_cid"],
                "task_id": occurrence["task_alias"],
                "goal_cid": "goal:retained-orphan-v2",
                "status": (
                    "retrying"
                    if int(occurrence["blocked_task_revision"]) % 2 == 0
                    else "ready"
                ),
                "priority": "P0",
                "ordinal": index,
                "title": "One-shot retained orphan",
                "validations": [["python", "-m", "pytest", "-q"]],
            }
            for index, occurrence in enumerate(occurrences, start=1)
        ],
    }
    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(population)
        for occurrence in occurrences:
            _arm_retrying_task(
                daemon,
                occurrence,
                admissions[occurrence["task_cid"]],
            )
        admitted_ids = {
            admission["admission_id"] for admission in admissions.values()
        }
        monkeypatch.setattr(
            daemon,
            "_retained_recovery_admission_fence_is_current",
            lambda candidate, *, allowed_states: bool(
                isinstance(candidate, Mapping)
                and candidate.get("admission_id") in admitted_ids
                and "admitted" in allowed_states
            ),
        )
        monkeypatch.setattr(
            daemon,
            "_retained_recovery_aggregate_admitted_current",
            lambda: True,
        )

        def fail_insert(_claim: Any, **_kwargs: Any) -> None:
            raise RuntimeError("injected retained orphan")

        monkeypatch.setattr(daemon, "_insert_attempt_from_claim", fail_insert)
        assert daemon.claim_next() is None
        target = daemon.task_source.get(occurrences[0]["task_cid"])
        assert target is not None and target.status == "in_progress"
        consumption = target.body["completion_receipt"][
            "retained_recovery_consumption"
        ]
        assert daemon.get_attempt(consumption["attempt_id"]) is None

        owner_cas = lambda **_kwargs: None
        setattr(owner_cas, "__database_portal_owner_fence_held__", True)
        setattr(
            owner_cas,
            "__database_portal_checkout_mutation_lease_held__",
            True,
        )
        daemon._database_portal_outer_authority_cas = owner_cas
        daemon.authority_mode = "quack"
        daemon.control_store_id = occurrences[0]["owner_store_id"]
        daemon.control_store_generation = occurrences[0][
            "control_store_generation"
        ]
        barrier_calls: list[str] = []
        real_barrier = (
            daemon.coordinator.execute_with_terminal_task_claim_barrier
        )

        def tracked_barrier(
            claim: Any,
            callback: Any,
            *,
            lease: Any = None,
        ) -> Any:
            assert lease is not None
            barrier_calls.append(str(claim.claim_id))
            return real_barrier(claim, callback, lease=lease)

        monkeypatch.setattr(
            daemon.coordinator,
            "execute_with_terminal_task_claim_barrier",
            tracked_barrier,
        )

        outcomes = daemon.reconcile_retained_recovery_orphaned_claims()

        assert len(outcomes) == 1
        assert outcomes[0]["task_alias"] == "PCTDD-006"
        assert outcomes[0]["status"] == "blocked"
        assert barrier_calls == [consumption["claim_id"]]
        repaired = daemon.task_source.get(occurrences[0]["task_cid"])
        assert repaired is not None and repaired.status == "blocked"
        assert repaired.revision == occurrences[0]["blocked_task_revision"] + 3
        claim = daemon.coordinator.get_task_claim(consumption["claim_id"])
        lease = daemon.coordinator.get_lease(consumption["lease_id"])
        assert claim is not None and claim.state.value == "released"
        assert lease is not None and lease.state.value == "released"
        assert provider_calls == []

        replay = daemon.reconcile_retained_recovery_orphaned_claims()
        assert len(replay) == 1
        assert replay[0]["reason"] == (
            "retained_consumed_orphan_already_blocked"
        )
        assert barrier_calls == [
            consumption["claim_id"],
            consumption["claim_id"],
        ]

        successor = daemon.coordinator.claim_task(
            task_cid=str(repaired.task_cid),
            owner_session_id="session:retained-successor",
        )
        with pytest.raises(
            DatabaseCoordinationStaleFenceError,
            match="latest fence|successor task authority",
        ):
            daemon.reconcile_retained_recovery_orphaned_claims()
        assert barrier_calls == [
            consumption["claim_id"],
            consumption["claim_id"],
            consumption["claim_id"],
        ]
        successor_readback = daemon.coordinator.get_task_claim(
            successor.claim_id
        )
        assert successor_readback is not None
        assert successor_readback.state.value == "accepted"
        still_blocked = daemon.task_source.get(occurrences[0]["task_cid"])
        assert still_blocked is not None
        assert still_blocked.status == "blocked"
        assert still_blocked.revision == repaired.revision
        assert provider_calls == []
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_already_blocked_retained_orphan_terminalizes_accepted_claim_before_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliases = ("PCTDD-006", "PCTDD-007", "PCTDD-034")
    records = [_admission(monkeypatch, task_alias=alias) for alias in aliases]
    occurrences = [record[0] for record in records]
    admissions = {
        record[0]["task_cid"]: dict(record[4]) for record in records
    }
    population = {
        "repository_tree_id": "tree:retained-orphan-accepted-v2",
        "objectives": [
            {
                "objective_id": "objective:retained-orphan-accepted-v2",
                "objective_alias": "PCTDD-O-RECOVERY-ACCEPTED",
                "title": "Accepted retained orphan recovery",
                "goal_cid": "goal:retained-orphan-accepted-v2",
                "goal_alias": "PCTDD-G-RECOVERY-ACCEPTED",
                "status": "open",
            }
        ],
        "tasks": [
            {
                "task_cid": occurrence["task_cid"],
                "task_id": occurrence["task_alias"],
                "goal_cid": "goal:retained-orphan-accepted-v2",
                "status": (
                    "retrying"
                    if int(occurrence["blocked_task_revision"]) % 2 == 0
                    else "ready"
                ),
                "priority": "P0",
                "ordinal": index,
                "title": "Accepted one-shot retained orphan",
                "validations": [["python", "-m", "pytest", "-q"]],
            }
            for index, occurrence in enumerate(occurrences, start=1)
        ],
    }
    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(population)
        for occurrence in occurrences:
            _arm_retrying_task(
                daemon,
                occurrence,
                admissions[occurrence["task_cid"]],
            )
        admitted_ids = {
            admission["admission_id"] for admission in admissions.values()
        }
        monkeypatch.setattr(
            daemon,
            "_retained_recovery_admission_fence_is_current",
            lambda candidate, *, allowed_states: bool(
                isinstance(candidate, Mapping)
                and candidate.get("admission_id") in admitted_ids
                and "admitted" in allowed_states
            ),
        )
        monkeypatch.setattr(
            daemon,
            "_retained_recovery_aggregate_admitted_current",
            lambda: True,
        )
        monkeypatch.setattr(
            daemon,
            "_insert_attempt_from_claim",
            lambda _claim, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected retained orphan")
            ),
        )
        assert daemon.claim_next() is None

        occurrence = occurrences[0]
        consumed = daemon.task_source.get(occurrence["task_cid"])
        assert consumed is not None and consumed.status == "in_progress"
        consumed_receipt = dict(consumed.body["completion_receipt"])
        consumption = dict(consumed_receipt["retained_recovery_consumption"])
        claim = daemon.coordinator.get_task_claim(consumption["claim_id"])
        assert claim is not None and claim.state.value == "accepted"
        block_receipt = daemon._retry_budget_receipt(
            consumed,
            attempts_used=int(consumed_receipt["attempts_used"]),
            operation="database_retained_recovery_consumed_orphan_blocked",
            attempt=claim,
            reason="retained_recovery_consumed_without_local_attempt",
        )
        block_receipt.update(
            {
                "orphaned_attempt_id": consumption["attempt_id"],
                "orphaned_claim_id": consumption["claim_id"],
                "authority_outcome": "unknown",
                "policy_mismatch": False,
                "malformed": False,
                "configured_max_task_attempts": 3,
                "retained_recovery_consumed": True,
                "retry_exhausted": True,
            }
        )
        daemon._cas_task_status_database(
            consumed.task_cid,
            expected_revision=int(consumed.revision),
            new_status="blocked",
            receipt=block_receipt,
        )

        owner_cas = lambda **_kwargs: None
        setattr(owner_cas, "__database_portal_owner_fence_held__", True)
        setattr(
            owner_cas,
            "__database_portal_checkout_mutation_lease_held__",
            True,
        )
        daemon._database_portal_outer_authority_cas = owner_cas
        daemon.authority_mode = "quack"
        daemon.control_store_id = occurrence["owner_store_id"]
        daemon.control_store_generation = occurrence[
            "control_store_generation"
        ]
        barrier_states: list[str] = []
        real_barrier = (
            daemon.coordinator.execute_with_terminal_task_claim_barrier
        )

        def tracked_barrier(
            terminal_claim: Any,
            callback: Any,
            *,
            lease: Any,
        ) -> Any:
            barrier_states.append(terminal_claim.state.value)
            return real_barrier(terminal_claim, callback, lease=lease)

        monkeypatch.setattr(
            daemon.coordinator,
            "execute_with_terminal_task_claim_barrier",
            tracked_barrier,
        )

        outcomes = daemon.reconcile_retained_recovery_orphaned_claims()

        assert len(outcomes) == 1
        assert outcomes[0]["reason"] == (
            "retained_consumed_orphan_already_blocked"
        )
        assert barrier_states == ["released"]
        terminal_claim = daemon.coordinator.get_task_claim(
            consumption["claim_id"]
        )
        terminal_lease = daemon.coordinator.get_lease(
            consumption["lease_id"]
        )
        assert terminal_claim is not None
        assert terminal_claim.state.value == "released"
        assert terminal_lease is not None
        assert terminal_lease.state.value == "released"
        readback = daemon.task_source.get(occurrence["task_cid"])
        assert readback is not None and readback.status == "blocked"
        assert readback.revision == occurrence["blocked_task_revision"] + 3
        assert provider_calls == []
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
@pytest.mark.parametrize("mutation", ["extra_field", "retry_cap"])
def test_strict_orphan_rejects_noncanonical_r_plus_2_without_laundering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    provider_calls: list[str] = []
    daemon, occurrences, _admissions = _strict_three_pin_daemon(
        tmp_path,
        monkeypatch,
        provider_calls=provider_calls,
    )
    try:
        real_retry_receipt = daemon._retry_budget_receipt

        def tampered_retry_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
            receipt = real_retry_receipt(*args, **kwargs)
            if (
                receipt.get("operation") == "database_claim"
                and "retained_recovery_consumption" in receipt
            ):
                if mutation == "extra_field":
                    receipt["unexpected"] = True
                else:
                    receipt["max_task_attempts"] = 4
            return receipt

        monkeypatch.setattr(
            daemon,
            "_retry_budget_receipt",
            tampered_retry_receipt,
        )
        monkeypatch.setattr(
            daemon,
            "_insert_attempt_from_claim",
            lambda _claim, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected retained orphan")
            ),
        )
        assert daemon.claim_next() is None
        target = daemon.task_source.get(occurrences[0]["task_cid"])
        assert target is not None and target.status == "in_progress"
        before = target.to_dict()
        consumption = target.body["completion_receipt"][
            "retained_recovery_consumption"
        ]

        with pytest.raises(
            DatabaseImplementationConflictError,
            match=r"R\+2 claim receipt is not closed and exact",
        ):
            daemon.reconcile_retained_recovery_orphaned_claims()

        readback = daemon.task_source.get(occurrences[0]["task_cid"])
        assert readback is not None and readback.to_dict() == before
        claim = daemon.coordinator.get_task_claim(consumption["claim_id"])
        lease = daemon.coordinator.get_lease(consumption["lease_id"])
        assert claim is not None and claim.state.value == "accepted"
        assert lease is not None and lease.state.value == "accepted"
        assert provider_calls == []
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
@pytest.mark.parametrize("mutation", ["extra_field", "retry_cap"])
def test_strict_orphan_rejects_noncanonical_r_plus_3_idempotent_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    provider_calls: list[str] = []
    daemon, occurrences, _admissions = _strict_three_pin_daemon(
        tmp_path,
        monkeypatch,
        provider_calls=provider_calls,
    )
    try:
        monkeypatch.setattr(
            daemon,
            "_insert_attempt_from_claim",
            lambda _claim, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected retained orphan")
            ),
        )
        assert daemon.claim_next() is None
        occurrence = occurrences[0]
        consumed = daemon.task_source.get(occurrence["task_cid"])
        assert consumed is not None and consumed.status == "in_progress"
        consumed_receipt = dict(consumed.body["completion_receipt"])
        consumption = dict(consumed_receipt["retained_recovery_consumption"])
        claim = daemon.coordinator.get_task_claim(consumption["claim_id"])
        assert claim is not None and claim.state.value == "accepted"
        block_receipt = daemon._retry_budget_receipt(
            consumed,
            attempts_used=int(consumed_receipt["attempts_used"]),
            operation="database_retained_recovery_consumed_orphan_blocked",
            attempt=claim,
            reason="retained_recovery_consumed_without_local_attempt",
        )
        block_receipt.update(
            {
                "orphaned_attempt_id": consumption["attempt_id"],
                "orphaned_claim_id": consumption["claim_id"],
                "authority_outcome": "unknown",
                "policy_mismatch": False,
                "malformed": False,
                "configured_max_task_attempts": 3,
                "retained_recovery_consumed": True,
                "retry_exhausted": True,
            }
        )
        if mutation == "extra_field":
            block_receipt["unexpected"] = True
        else:
            block_receipt["max_task_attempts"] = 4
        daemon._cas_task_status_database(
            consumed.task_cid,
            expected_revision=int(consumed.revision),
            new_status="blocked",
            receipt=block_receipt,
        )
        before = daemon.task_source.get(occurrence["task_cid"])
        assert before is not None

        with pytest.raises(
            DatabaseImplementationConflictError,
            match="blocked readback is not exact",
        ):
            daemon.reconcile_retained_recovery_orphaned_claims()

        readback = daemon.task_source.get(occurrence["task_cid"])
        assert readback is not None and readback.to_dict() == before.to_dict()
        claim_readback = daemon.coordinator.get_task_claim(
            consumption["claim_id"]
        )
        lease_readback = daemon.coordinator.get_lease(consumption["lease_id"])
        assert claim_readback is not None
        assert claim_readback.state.value == "accepted"
        assert lease_readback is not None
        assert lease_readback.state.value == "accepted"
        assert provider_calls == []
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_prepared_completion_race_cannot_release_retained_orphan_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls: list[str] = []
    daemon, occurrences, _admissions = _strict_three_pin_daemon(
        tmp_path,
        monkeypatch,
        provider_calls=provider_calls,
    )
    try:
        monkeypatch.setattr(
            daemon,
            "_insert_attempt_from_claim",
            lambda _claim, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected retained orphan")
            ),
        )
        assert daemon.claim_next() is None
        occurrence = occurrences[0]
        consumed = daemon.task_source.get(occurrence["task_cid"])
        assert consumed is not None and consumed.status == "in_progress"
        before = consumed.to_dict()
        consumption = consumed.body["completion_receipt"][
            "retained_recovery_consumption"
        ]
        real_terminalize = (
            daemon.coordinator.terminalize_unprepared_task_claim
        )
        injected: list[str] = []

        def prepare_before_terminalization(
            claim: Any,
            *,
            lease: Any,
            reason: str,
            now_ms: int,
        ) -> Any:
            daemon.coordinator.prepare_task_completion(
                claim,
                control_expected_revision=int(consumed.revision),
                control_expected_status="in_progress",
                evidence_digest="sha256:" + "c" * 64,
            )
            injected.append(str(claim.claim_id))
            return real_terminalize(
                claim,
                lease=lease,
                reason=reason,
                now_ms=now_ms,
            )

        monkeypatch.setattr(
            daemon.coordinator,
            "terminalize_unprepared_task_claim",
            prepare_before_terminalization,
        )

        with pytest.raises(
            DatabaseCoordinationNotReadyError,
            match="preparation owns settlement",
        ):
            daemon.reconcile_retained_recovery_orphaned_claims()

        assert injected == [consumption["claim_id"]]
        readback = daemon.task_source.get(occurrence["task_cid"])
        assert readback is not None and readback.to_dict() == before
        claim = daemon.coordinator.get_task_claim(consumption["claim_id"])
        lease = daemon.coordinator.get_lease(consumption["lease_id"])
        assert claim is not None and claim.state.value == "accepted"
        assert lease is not None and lease.state.value == "accepted"
        assert daemon.coordinator.get_prepared_task_completion(
            occurrence["task_cid"]
        ) is not None
        assert provider_calls == []
    finally:
        daemon.close()
