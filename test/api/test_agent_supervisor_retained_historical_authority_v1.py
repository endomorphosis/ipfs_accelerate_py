"""Regression tests for split-store retained occurrence authority."""

from __future__ import annotations

import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.retained_recovery_contracts import (
    database_fenced_provider_historical_occurrence_authority,
    database_fenced_provider_historical_occurrence_authority_valid,
    database_portal_controller_quiescence_receipt,
    database_portal_controller_quiescence_receipt_valid,
    retained_recovery_sha256,
)


def _cleanup() -> dict[str, object]:
    return {
        "pid": 4101,
        "managed_daemon_identity_record_id": "identity:managed:4101",
        "managed_daemon_process_birth": {
            "pid": 4101,
            "start_time_ticks": 71,
            "boot_id": "boot:test",
            "parent_pid": 4000,
        },
        "quiesced": True,
        "remaining_pid": None,
        "markers_removed": True,
        "daemon_fence": {
            "fenced": True,
            "safe_to_restart": True,
            "reason": "managed_daemon_owned_process_fenced",
        },
        "provider_runner_fence": {
            "applicable": False,
            "fenced": False,
            "safe_to_restart": True,
            "reason": "ordinary_provider_runner_receipt_absent",
        },
    }


def _controller_receipt() -> dict[str, object]:
    return dict(
        database_portal_controller_quiescence_receipt(
            cleanup=_cleanup(),
            board_namespace="parallel-content-sealing-proof-carrying-tdd-v1",
            state_prefix="pctdd_g9",
            owner_store_id="data/pctdd/control.duckdb",
            control_store_generation="pctdd-v1-g9",
            trigger="supervisor_startup_prelaunch",
            controller_process_birth={
                "pid": 4000,
                "start_time_ticks": 51,
                "boot_id": "boot:test",
                "parent_pid": 1,
            },
            owner_mutation_fence_held=True,
            managed_daemon_launch_lock_held=True,
        )
    )


def _inner_receipt() -> dict[str, object]:
    subject = {
        "task_cid": "task:pctdd-005",
        "task_alias": "PCTDD-005",
        "task_revision": 26,
        "attempt_id": "attempt:pctdd-005:historical",
        "claim_id": "claim:pctdd-005:historical",
        "lease_id": "lease:pctdd-005:historical",
        "attempt_number": 2,
        "owner_session_id": "owner:pctdd:historical",
        "fencing_token": 9,
        "fence_epoch": 4,
        "recovery_manifest_id": "sha256:" + "1" * 64,
        "recovery_credit_id": "sha256:" + "2" * 64,
    }
    terminal = {
        "attempt_id": subject["attempt_id"],
        "task_cid": subject["task_cid"],
        "claim_id": subject["claim_id"],
        "attempt_number": subject["attempt_number"],
        "owner_session_id": subject["owner_session_id"],
        "lease_id": subject["lease_id"],
        "fencing_token": subject["fencing_token"],
        "fence_epoch": subject["fence_epoch"],
        "intended_database_disposition": "blocked_unknown_outcome",
        "evidence_id": "baguqeera" + "a" * 52,
        "prepared_reconciliation_receipt_id": "sha256:" + "3" * 64,
        "commit_barrier_receipt_id": "sha256:" + "4" * 64,
        "stage": "terminal",
        "receipt_id": "sha256:" + "5" * 64,
        "record_json": {
            "canonical_sha256": "sha256:" + "6" * 64,
            "canonical_byte_length": 4096,
        },
    }
    return {
        "receipt_nonce": "nonce:pctdd-005:historical",
        "receipt_epoch": 1,
        "receipt_cid": "sha256:" + "7" * 64,
        "subject": subject,
        "groups": {
            "database_portal_terminal_reconciliations": {
                "count": 1,
                "rows": [terminal],
            }
        },
    }


def _authority() -> dict[str, object]:
    return dict(
        database_fenced_provider_historical_occurrence_authority(
            inner_receipt=_inner_receipt(),
            controller_quiescence_receipt=_controller_receipt(),
            board_namespace="parallel-content-sealing-proof-carrying-tdd-v1",
            owner_store_id="data/pctdd/control.duckdb",
            control_store_generation="pctdd-v1-g9",
        )
    )


def _rehash(record: dict[str, object], identity: str) -> None:
    unsigned = dict(record)
    unsigned.pop(identity, None)
    record[identity] = retained_recovery_sha256(unsigned)


def test_controller_quiescence_requires_exact_death_and_both_fences() -> None:
    receipt = _controller_receipt()
    assert database_portal_controller_quiescence_receipt_valid(receipt)

    for field, value in (
        ("quiesced", False),
        ("remaining_pid", 4101),
        ("markers_removed", False),
        ("owner_mutation_fence_held", False),
        ("managed_daemon_launch_lock_held", False),
        ("daemon_safe_to_restart", False),
        ("provider_runner_safe_to_restart", False),
    ):
        changed = copy.deepcopy(receipt)
        changed[field] = value
        _rehash(changed, "receipt_id")
        assert not database_portal_controller_quiescence_receipt_valid(changed)

    unsafe = _cleanup()
    unsafe["remaining_pid"] = 4101
    with pytest.raises(ValueError, match="failed closed"):
        database_portal_controller_quiescence_receipt(
            cleanup=unsafe,
            board_namespace="parallel-content-sealing-proof-carrying-tdd-v1",
            state_prefix="pctdd_g9",
            owner_store_id="data/pctdd/control.duckdb",
            control_store_generation="pctdd-v1-g9",
            trigger="supervisor_startup_prelaunch",
            controller_process_birth={
                "pid": 4000,
                "start_time_ticks": 51,
                "boot_id": "boot:test",
                "parent_pid": 1,
            },
            owner_mutation_fence_held=True,
            managed_daemon_launch_lock_held=True,
        )


def test_historical_authority_binds_one_terminal_lane_row_and_quiescence() -> None:
    authority = _authority()
    inner = _inner_receipt()
    assert database_fenced_provider_historical_occurrence_authority_valid(
        authority
    )
    assert authority["inner_receipt_cid"] == inner["receipt_cid"]
    assert authority["terminal_reconciliation"] == inner["groups"][
        "database_portal_terminal_reconciliations"
    ]["rows"][0]
    assert authority["controller_quiescence_receipt_id"] == (
        _controller_receipt()["receipt_id"]
    )

    for mutator in (
        lambda value: value["subject"].__setitem__("claim_id", "claim:other"),
        lambda value: value["terminal_reconciliation"].__setitem__(
            "stage", "commit_barrier"
        ),
        lambda value: value.__setitem__(
            "controller_quiescence_receipt_id", "sha256:" + "8" * 64
        ),
    ):
        changed = copy.deepcopy(authority)
        mutator(changed)
        _rehash(changed, "authority_id")
        assert not database_fenced_provider_historical_occurrence_authority_valid(
            changed
        ) or changed["controller_quiescence_receipt_id"] != authority[
            "controller_quiescence_receipt_id"
        ]

    no_terminal = _inner_receipt()
    no_terminal["groups"]["database_portal_terminal_reconciliations"] = {
        "count": 0,
        "rows": [],
    }
    with pytest.raises(ValueError, match="one exact terminal"):
        database_fenced_provider_historical_occurrence_authority(
            inner_receipt=no_terminal,
            controller_quiescence_receipt=_controller_receipt(),
            board_namespace="parallel-content-sealing-proof-carrying-tdd-v1",
            owner_store_id="data/pctdd/control.duckdb",
            control_store_generation="pctdd-v1-g9",
        )
