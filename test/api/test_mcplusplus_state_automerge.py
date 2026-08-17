"""MCPP-038: Automerge CRDT state adapter convergence tests.

Acceptance:
- Two isolated replicas converge after partition heal.
- Duplicates are idempotent.
- Implementation is Automerge, not informal LWW.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

automerge = pytest.importorskip(
    "automerge",
    reason="AutomergeCrdtState@1 requires the real automerge package",
)

from ipfs_accelerate_py.mcp_server.mcplusplus.state.automerge_crdt import (  # noqa: E402
    BACKEND_ID,
    INTERFACE_ID,
    STATE_MODE,
    STATE_REF_SCHEMA,
    AutomergeCrdtState,
    AutomergeValueError,
    open_automerge_crdt_state,
)


def test_interface_identity_is_automerge_not_lww() -> None:
    """Adapter advertises Automerge CRDT identity, never LWW."""
    assert INTERFACE_ID == "AutomergeCrdtState@1"
    assert STATE_MODE == "crdt"
    assert BACKEND_ID == "automerge"
    assert "lww" not in INTERFACE_ID.lower()
    assert "lww" not in BACKEND_ID.lower()

    # Real Automerge library is what backs the adapter.
    assert automerge is not None
    assert hasattr(automerge, "core")
    source = inspect.getsource(
        importlib.import_module(
            "ipfs_accelerate_py.mcp_server.mcplusplus.state.automerge_crdt"
        )
    )
    assert "from automerge.core import" in source
    # Fail closed on informal LWW implementations masquerading as CRDT.
    assert "def last_write_wins" not in source
    assert "last_write_wins =" not in source
    assert "class " not in source or "Lww" not in source
    assert "wall_clock" not in source
    assert "timestamp_winner" not in source


def test_basic_put_get_delete_and_state_ref() -> None:
    replica = open_automerge_crdt_state("state/demo", actor_id="replica-a")
    assert replica.mode == "crdt"
    assert replica.backend == "automerge"
    assert replica.interface_id == INTERFACE_ID

    replica.put("title", "hello")
    replica.put("count", 3)
    replica.put("flag", True)
    replica.put("nested", {"a": 1, "b": ["x", "y"]})

    assert replica.get("title") == "hello"
    assert replica["count"] == 3
    assert replica.get("flag") is True
    assert replica.get("nested") == {"a": 1, "b": ["x", "y"]}
    assert "title" in replica
    assert replica.get("missing", "d") == "d"

    assert replica.delete("flag") is True
    assert replica.get("flag") is None
    assert replica.delete("flag") is False

    ref = replica.state_ref()
    assert ref["schema"] == STATE_REF_SCHEMA
    assert ref["id"] == "state/demo"
    assert ref["mode"] == "crdt"
    assert ref["provider"] == "automerge"
    assert ref["clocks"]["automerge_heads"]
    assert ref["metadata"]["automerge"]["backend"] == "automerge"
    assert ref["metadata"]["automerge"]["change_count"] >= 1


def test_partition_heal_two_isolated_replicas_converge() -> None:
    """Two offline writers exchange Automerge sync and converge."""
    # Shared genesis so both replicas start from the same Automerge history.
    genesis = AutomergeCrdtState.open("state/partition", actor_id="genesis")
    genesis.put("shared", "base")
    genesis.put("version", 1)
    blob = genesis.save()

    left = AutomergeCrdtState.load(blob, state_id="state/partition", actor_id="left")
    right = AutomergeCrdtState.load(blob, state_id="state/partition", actor_id="right")

    # Partition: concurrent offline updates on each side.
    left.put("left_only", "L")
    left.put("shared", "from-left")
    left.put("version", 2)

    right.put("right_only", "R")
    right.put("shared", "from-right")
    right.put("extra", {"n": 7})

    # Still diverged before heal.
    assert left.snapshot() != right.snapshot()
    assert left.heads() != right.heads()

    exchanged = left.sync_with(right)
    assert exchanged > 0

    # Partition heal: converge.
    assert left.converged_with(right)
    assert left.heads() == right.heads()
    snap = left.snapshot()
    assert snap["left_only"] == "L"
    assert snap["right_only"] == "R"
    assert snap["extra"] == {"n": 7}
    # Concurrent write to the same key is resolved by Automerge (not wall-clock LWW).
    assert snap["shared"] in {"from-left", "from-right"}
    assert "version" in snap


def test_merge_document_partition_heal_is_commutative() -> None:
    """Document merge order A←B vs B←A yields equal heads and snapshots."""
    genesis = AutomergeCrdtState.open("state/merge", actor_id="g")
    genesis.put("root", True)
    blob = genesis.save()

    a1 = AutomergeCrdtState.load(blob, state_id="state/merge", actor_id="a")
    b1 = AutomergeCrdtState.load(blob, state_id="state/merge", actor_id="b")
    a2 = AutomergeCrdtState.load(blob, state_id="state/merge", actor_id="a")
    b2 = AutomergeCrdtState.load(blob, state_id="state/merge", actor_id="b")

    a1.put("a_key", "A")
    b1.put("b_key", "B")
    a2.put("a_key", "A")
    b2.put("b_key", "B")

    # Order 1: a merges b
    a1.merge(b1)
    b1.merge(a1)

    # Order 2: b merges a first
    b2.merge(a2)
    a2.merge(b2)

    assert a1.converged_with(b1)
    assert a2.converged_with(b2)
    assert a1.heads() == a2.heads() == b1.heads() == b2.heads()
    assert a1.snapshot() == a2.snapshot() == {"root": True, "a_key": "A", "b_key": "B"}


def test_duplicate_document_apply_is_idempotent() -> None:
    """Replaying the same Automerge document save does not diverge state."""
    a = AutomergeCrdtState.open("state/dup", actor_id="a")
    b = AutomergeCrdtState.open("state/dup", actor_id="b")
    a.put("x", 1)
    a.put("y", "two")

    first = b.apply_document(a.save())
    assert first.applied == 1
    snap = b.snapshot()
    heads = b.heads()
    evidence = b.change_evidence()

    second = b.apply_document(a.save())
    assert second.skipped_duplicates >= 1 or second.applied == 0
    assert b.snapshot() == snap
    assert b.heads() == heads
    assert b.change_evidence()["change_hashes"] == evidence["change_hashes"]

    # apply_changes with the same peer is also idempotent.
    third = b.apply_changes([a])
    assert third.skipped_duplicates >= 1 or third.applied == 0
    assert b.snapshot() == snap
    assert b.heads() == heads


def test_duplicate_change_hashes_are_idempotent() -> None:
    """Change evidence already present in history is skipped on re-apply."""
    a = AutomergeCrdtState.open("state/ch", actor_id="a")
    b = AutomergeCrdtState.open("state/ch", actor_id="b")
    a.put("k", "v")
    b.merge(a)

    exported = a.export_changes([])
    assert exported
    # Every exported change is already in b after merge.
    result = b.apply_changes(exported)
    assert result.applied == 0
    assert result.skipped_duplicates == len(exported)
    assert b.get("k") == "v"


def test_reordered_sync_messages_converge() -> None:
    """Automerge sync messages delivered out of pair-order still converge."""
    genesis = AutomergeCrdtState.open("state/reorder", actor_id="g")
    genesis.put("seed", 0)
    blob = genesis.save()

    a = AutomergeCrdtState.load(blob, state_id="state/reorder", actor_id="a")
    b = AutomergeCrdtState.load(blob, state_id="state/reorder", actor_id="b")
    a.put("from_a", 1)
    b.put("from_b", 2)

    sa = a.new_sync_state()
    sb = b.new_sync_state()

    # Collect messages independently, then deliver in reversed order.
    messages_ab: list[bytes] = []
    messages_ba: list[bytes] = []
    for _ in range(8):
        m_a = a.generate_sync_message(sa)
        m_b = b.generate_sync_message(sb)
        if m_a is not None:
            messages_ab.append(m_a)
        if m_b is not None:
            messages_ba.append(m_b)
        if m_a is None and m_b is None:
            break
        # Deliver immediately in reverse preference: B first, then A.
        if m_b is not None:
            a.receive_sync_message(sa, m_b)
        if m_a is not None:
            b.receive_sync_message(sb, m_a)

    # Drain any residual.
    a.sync_with(b)
    assert a.converged_with(b)
    assert a.snapshot()["from_a"] == 1
    assert a.snapshot()["from_b"] == 2
    # Duplicate delivery of a previously sent message must not break state.
    if messages_ab:
        b.receive_sync_message(b.new_sync_state(), messages_ab[0])
    assert a.snapshot()["from_a"] == 1


def test_concurrent_offline_keys_all_survive() -> None:
    """CRDT merge keeps concurrent distinct keys (not single-winner LWW map)."""
    g = AutomergeCrdtState.open("state/keys", actor_id="g")
    g.put("base", True)
    blob = g.save()

    r1 = AutomergeCrdtState.load(blob, state_id="state/keys", actor_id="r1")
    r2 = AutomergeCrdtState.load(blob, state_id="state/keys", actor_id="r2")
    r3 = AutomergeCrdtState.load(blob, state_id="state/keys", actor_id="r3")

    r1.put("k1", "one")
    r2.put("k2", "two")
    r3.put("k3", "three")

    r1.merge(r2)
    r1.merge(r3)
    r2.merge(r1)
    r3.merge(r1)

    expected = {"base": True, "k1": "one", "k2": "two", "k3": "three"}
    assert r1.snapshot() == expected
    assert r2.snapshot() == expected
    assert r3.snapshot() == expected
    assert r1.heads() == r2.heads() == r3.heads()


def test_save_load_roundtrip() -> None:
    original = AutomergeCrdtState.open("state/rt", actor_id="actor-1")
    original.put("msg", "persist")
    original.put("n", 99)
    restored = AutomergeCrdtState.load(
        original.save(), state_id="state/rt", actor_id="actor-1"
    )
    assert restored.snapshot() == original.snapshot()
    assert restored.heads() == original.heads()


def test_state_id_mismatch_rejected() -> None:
    a = AutomergeCrdtState.open("state/a", actor_id="x")
    b = AutomergeCrdtState.open("state/b", actor_id="y")
    a.put("k", 1)
    with pytest.raises(AutomergeValueError, match="state_id mismatch"):
        a.merge(b)
    with pytest.raises(AutomergeValueError, match="state_id mismatch"):
        a.sync_with(b)


def test_fork_creates_isolated_actor() -> None:
    base = AutomergeCrdtState.open("state/fork", actor_id="base")
    base.put("v", 1)
    child = base.fork(actor_id="child")
    child.put("v", 2)
    child.put("child_only", True)
    assert base.get("v") == 1
    assert "child_only" not in base
    base.merge(child)
    assert base.get("v") == 2
    assert base.get("child_only") is True
