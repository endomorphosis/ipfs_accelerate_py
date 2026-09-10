"""Generic recovery preserves its existing eight-stage scan contract."""

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import owner_recovery_runtime as recovery
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerError,
)
from test.api.test_agent_supervisor_owner_recovery_runtime import (
    SCOPE,
    owner as _owner_fixture,
    recovery_owner as _recovery_owner_fixture,
)

owner = _owner_fixture
recovery_owner = _recovery_owner_fixture


def _native_five_stage_map(cursors):
    # Exact earlier native profile, with nonempty evidence to preserve.
    old = dict(cursors)
    old.pop("false_completed_requests")
    old.pop("false_pending_requests")
    old.pop("false_processing_requests")
    old["completed_requests"] = "preserved:completed:064"
    return old


def test_native_five_stage_cas_cannot_reset_generic_scan_progress(recovery_owner):
    own = recovery_owner
    head = own.api.load_cursors()
    with pytest.raises(TypedStateOwnerError):
        own.api.cas_cursors(
            expected_revision=head["revision"],
            expected_state_cid=head["state_cid"],
            cursors=_native_five_stage_map(head["cursors"]),
            operation_id="profile:incompatible-cas",
        )
    assert own.api.load_cursors() == head
    with own.gateway._transaction_lock:
        assert (
            own.connection.execute(
                "SELECT COUNT(*) FROM legacy_merge_recovery_operations"
            ).fetchone()[0]
            == 0
        )


def test_native_five_stage_import_requires_explicit_profile_qualification(
    recovery_owner,
):
    own = recovery_owner
    binding = {**SCOPE, "lane_id": "lane:old-native"}
    scope = recovery.recovery_scope_cid(
        store_id=own.gateway.store_id,
        repository_id="repo:one",
        target_branch="main",
        scope_binding=binding,
    )
    old = _native_five_stage_map(own.api.load_cursors()["cursors"])
    with pytest.raises(recovery.OwnerRecoveryRuntimeError):
        own.gateway.provision_legacy_merge_recovery_schema(
            expected_identity=dict(own.gateway.identity),
            repository_id="repo:one",
            target_branch="main",
            migration_id="profile:incompatible-import",
            scope_bindings=[binding],
            cursor_imports=[
                {
                    "scope_cid": scope,
                    "cursors": old,
                    "state_cid": recovery._cid(old),
                }
            ],
        )
    with own.gateway._transaction_lock:
        assert (
            own.connection.execute(
                "SELECT COUNT(*) FROM legacy_merge_recovery_scopes WHERE scope_cid=?",
                [scope],
            ).fetchone()[0]
            == 0
        )
        assert (
            own.connection.execute(
                "SELECT COUNT(*) FROM legacy_merge_recovery_migrations WHERE migration_id=?",
                ["profile:incompatible-import"],
            ).fetchone()[0]
            == 0
        )


def test_retained_five_stage_state_is_unavailable_never_defaulted(recovery_owner):
    own = recovery_owner
    old = _native_five_stage_map(own.api.load_cursors()["cursors"])
    expected = (recovery._cid(old), recovery._json(old))
    # Reproduce a coherent retained earlier profile in this disposable owner;
    # both head and immutable history match. Denial must be about the profile.
    with own.gateway._transaction_lock:
        for table in (
            "legacy_merge_recovery_cursors",
            "legacy_merge_recovery_cursor_history",
        ):
            own.connection.execute(
                f"UPDATE {table} SET state_cid=?,cursors_json=? WHERE scope_cid=?",
                [*expected, own.scope_cid],
            )
    with pytest.raises(TypedStateOwnerError):
        own.api.load_cursors()
    with own.gateway._transaction_lock:
        row = own.connection.execute(
            "SELECT state_cid,cursors_json FROM legacy_merge_recovery_cursors WHERE scope_cid=?",
            [own.scope_cid],
        ).fetchone()
        assert tuple(row[i] for i in range(2)) == expected
