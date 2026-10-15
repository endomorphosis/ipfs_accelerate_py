"""Paired preparation reads detect changes in the actual native owner catalog.

The current producer, full source/capsule validation, offline tokenizer, Kit
SDK and persistent DuckDB adapter all execute normally. Owner READY rows,
launch/model admission and the deliberately timed external mutations are
authored fixtures; a refused preparation establishes no model, semantic or
Lean qualification. No historical storage fixture or package repinning is used.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.semantic_state import harness as harness_module
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    AcceptanceDisposition,
    RootRef,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.durable_state import (
    DurableStateError,
    DurableStateIntegrityError,
    _parse_root_manifest,
    root_manifest_artifact,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.harness import HarnessRootConflict
from ipfs_kit_py.mcp_server.mcplusplus import duckdb_coordination_storage as native_module
from test.api.semantic_state.test_harness_native_materializer import (  # noqa: F401 -- ordinary fixture imports, no repinning
    REPOSITORY,
    catalog_snapshot,
    current,
    native_case,
)

from test.api.semantic_state import test_harness_native_materializer as native_fixtures


@pytest.fixture
def guarded_native(native_case, monkeypatch):  # noqa: F811
    case = native_case
    effects = []

    def forbidden(*args, **kwargs):
        # Record even an effect that a best-effort mirror tries to suppress.
        effects.append((args, kwargs))
        pytest.fail("prepare-only crossed a provider/index/storage/worktree boundary")

    monkeypatch.setattr(native_fixtures, "fail_effect", forbidden)
    native_fixtures.refuse_preparation_effects(case, monkeypatch)
    monkeypatch.setattr(harness_module, "invoke_model", forbidden)
    return case, effects


def advance_same_root_with_native_cas(case, *, label: str) -> RootRef:
    """An authored external SDK A-to-B-to-A update, without fabricated revisions.

    Kit rejects a same-content CAS. Both actual transitions and their immutable
    blocks are therefore required to obtain the same root CID at a later
    revision. The intermediate manifest is closed and retains the original
    linked content; its authored accepted label grants no semantic acceptance.
    """
    store = case.durable.store
    before = store.current_state_root(REPOSITORY)
    with case.durable.metadata_scope():
        manifest = _parse_root_manifest(store.get(before["root_cid"]))
        alternate = root_manifest_artifact(replace(
            manifest, acceptance_disposition=AcceptanceDisposition.ACCEPTED.value,
        ))
    alternate_cid = store.put(alternate)["cid"]
    assert alternate_cid != before["root_cid"]
    outbound = store.compare_and_swap_state_root(
        REPOSITORY, expected_revision=before["revision"],
        expected_root_cid=before["root_cid"], new_root_cid=alternate_cid,
        operation_id=f"{label}-outbound",
    )
    assert outbound["status"] == "updated"
    returning = store.compare_and_swap_state_root(
        REPOSITORY, expected_revision=outbound["after"]["revision"],
        expected_root_cid=alternate_cid, new_root_cid=before["root_cid"],
        operation_id=f"{label}-returning",
    )
    assert returning["status"] == "updated"
    after = store.current_state_root(REPOSITORY)
    assert after["root_cid"] == before["root_cid"]
    assert after["revision"] == before["revision"] + 2
    return RootRef(root_cid=after["root_cid"], generation=after["revision"])


def corrupt_native_block(case, cid: str) -> None:
    """Change actual bytes at a stable key, keeping length metadata coherent."""
    replacement = b'{"schema":"authored-corruption@1","value":"different content"}'
    row = case.connection.execute(
        "SELECT block_bytes FROM kit_coordination_blocks WHERE namespace=? AND cid=?",
        [REPOSITORY, cid],
    ).fetchone()
    assert row is not None and bytes(row[0]) != replacement
    case.connection.execute(
        "UPDATE kit_coordination_blocks SET block_bytes=?, byte_length=? "
        "WHERE namespace=? AND cid=?",
        [replacement, len(replacement), REPOSITORY, cid],
    )


def refuse_after_real_tokenization(guarded_native, monkeypatch, mutation, *, exception, cause=None):
    """Time only the external mutation; retain real read_root/SDK/lock behavior."""
    case, effects = guarded_native
    history_before = copy.deepcopy(case.harness._terminal_attempts)
    before = catalog_snapshot(case.connection)
    original_count = type(case.port).count_tokens
    observed: dict[str, Any] = {}

    def counted_then_mutated(self, prompt, **bindings):
        count = original_count(self, prompt, **bindings)
        assert self is case.port
        assert count > 0
        assert "count" not in observed
        # Both materializations and their independent before/after owner checks
        # completed before the deliberately authored durable-state mutation.
        assert (case.authority.admit_calls, case.authority.verify_calls) == (2, 4)
        observed["count"] = count
        with case.durable.store._lock:
            observed["mutation_result"] = mutation(case)
            observed["expected_catalog"] = catalog_snapshot(case.connection)
        assert observed["expected_catalog"] != before
        return count

    monkeypatch.setattr(type(case.port), "count_tokens", counted_then_mutated)
    with pytest.raises(exception) as refused:
        case.harness.prepare_model_context(case.request)
    assert type(refused.value) is exception
    if cause is not None:
        assert type(refused.value.__cause__) is cause
    assert observed["count"] > 0
    assert (case.authority.admit_calls, case.authority.verify_calls) == (2, 4)
    # Compare against the catalog after the intended external change: refusal
    # must neither undo it nor append a reservation, transition or harness block.
    assert catalog_snapshot(case.connection) == observed["expected_catalog"]
    assert case.harness._terminal_attempts == history_before
    assert case.provider.calls == []
    assert effects == []
    return refused.value, observed["mutation_result"]


def test_closing_read_detects_real_same_cid_revision_advance(guarded_native, monkeypatch):
    case, _ = guarded_native
    refused, successor = refuse_after_real_tokenization(
        guarded_native, monkeypatch,
        lambda native: advance_same_root_with_native_cas(native, label="prepare-race"),
        exception=HarnessRootConflict,
    )
    assert successor.root_cid == case.root.root_cid
    assert successor.generation == case.root.generation + 2
    assert refused.current_root == successor
    assert case.durable.read_root(REPOSITORY) == successor


@pytest.mark.parametrize("fault", ["fence", "generation", "server_ready"])
def test_closing_read_rechecks_actual_native_owner(guarded_native, monkeypatch, fault):
    def mutate(case):
        if fault == "fence":
            case.connection.execute("UPDATE store_generations SET fence_epoch=fence_epoch+1")
        elif fault == "generation":
            case.connection.execute(
                "INSERT INTO store_generations SELECT generation+1, schema_revision, "
                "fence_epoch, revision+1, database_uuid, birth_id FROM store_generations"
            )
        else:
            case.connection.execute("UPDATE state_servers SET status='stopped'")

    refuse_after_real_tokenization(
        guarded_native, monkeypatch, mutate,
        exception=DurableStateError, cause=native_module.NativeOwnerBindingError,
    )


@pytest.mark.parametrize("field,value", [
    ("store_id", "authored-foreign-store"),
    ("database_uuid", "authored-foreign-database"),
    ("format_revision", 2),
])
def test_closing_read_rechecks_native_namespace_binding(guarded_native, monkeypatch, field, value):
    def mutate(case):
        # Field names come only from the closed parametrization above.
        case.connection.execute(
            f"UPDATE kit_coordination_namespaces SET {field}=? WHERE namespace=?",
            [value, REPOSITORY],
        )

    refuse_after_real_tokenization(
        guarded_native, monkeypatch, mutate,
        exception=DurableStateError, cause=native_module.NativeOwnerBindingError,
    )


def test_closing_read_revalidates_present_linked_block_at_unchanged_cid(guarded_native, monkeypatch):
    case, _ = guarded_native
    with case.durable.metadata_scope():
        manifest = _parse_root_manifest(case.durable.get(case.root.root_cid))
    linked_cid = manifest.capsule_index_cid
    assert linked_cid != case.root.root_cid and case.durable.has(linked_cid)
    refuse_after_real_tokenization(
        guarded_native, monkeypatch,
        lambda native: corrupt_native_block(native, linked_cid),
        # The current adapter's separate has() authenticates this present link
        # first and maps its native integrity failure to DurableStateError.
        exception=DurableStateError, cause=native_module.ArtifactIntegrityError,
    )


def test_closing_read_revalidates_root_block_at_unchanged_cid(guarded_native, monkeypatch):
    case, _ = guarded_native
    refuse_after_real_tokenization(
        guarded_native, monkeypatch,
        lambda native: corrupt_native_block(native, case.root.root_cid),
        exception=DurableStateIntegrityError, cause=native_module.ArtifactIntegrityError,
    )


def test_closing_read_revalidates_historical_transition_at_unchanged_current_cid(guarded_native, monkeypatch):
    case, _ = guarded_native
    # Build a real three-transition history before the opening preparation
    # read, then nominate the first immutable transition rather than its head.
    # Rebuilding the authored native launch validates opaque CID shapes, which
    # must use the same explicit native metadata scope as fixture admission.
    # The public preparation below retains every metadata/effect tripwire.
    with case.durable.metadata_scope():
        case.root = advance_same_root_with_native_cas(case, label="prior-history")
        case.request = replace(case.request, expected_root=case.root)
        case.authority.register(case.request, case.port, case.harness.policy, case.model)
    rows = case.connection.execute(
        "SELECT transition_cid, new_revision FROM kit_coordination_transitions "
        "WHERE namespace=? ORDER BY new_revision", [REPOSITORY],
    ).fetchall()
    assert [row[1] for row in rows] == [1, 2, 3]
    historical_cid = rows[0][0]
    assert historical_cid != rows[-1][0]
    assert case.durable.read_root(REPOSITORY) == case.root
    refuse_after_real_tokenization(
        guarded_native, monkeypatch,
        lambda native: corrupt_native_block(native, historical_cid),
        exception=DurableStateIntegrityError, cause=native_module.ArtifactIntegrityError,
    )
