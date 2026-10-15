"""Real CAS manifest bindings for the preview companion's structural reader.

These controls use the existing one-file native DuckDB/CAS fixture. They check
structural metadata reads, not successor training, freshness of a complete
signed bridge, behavioral proofs, or worker execution. Fault-injected loaders
are explicit receiving-protocol negatives and cannot grant native authority.
"""
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import codebase_preview_successor_admission as owner
from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits
from test.api.test_structural_codebase_context import native  # noqa: F401


def _changed_head(native):
    native["source"].write_text("def changed_private_symbol(n):\n    return n + 7\n")
    return native["index"].prepare_current(
        native["repository"], repository_id=native["head"].repository_id,
        operation_id="structural-reader-successor", expected_head=native["head"],
        scheduler=native["scheduler"], memory_mb=64,
        limits=CodebaseScanLimits(max_entries=16, max_file_bytes=1024),
    ).head


def test_native_reader_loads_exact_observed_cas_without_starting_preparation(native, monkeypatch):
    observation = native["index"].observe_current(
        native["repository"], expected_head=native["head"],
        scheduler=native["scheduler"], memory_mb=64)
    expected = observation.manifest
    calls = []
    load = native["index"].load

    def recorded_load(cid):
        calls.append(cid)
        return load(cid)

    def forbidden(*args, **kwargs):
        pytest.fail("structural reader initiated preparation, parsing, or catalog observation")

    monkeypatch.setattr(native["index"], "load", recorded_load)
    for name in ("prepare", "prepare_current", "observe_current"):
        monkeypatch.setattr(native["index"], name, forbidden)
    monkeypatch.setattr(native["index"].ingestor, "ingest_snapshot", forbidden)
    monkeypatch.setattr(native["index"].ingestor, "_parse_or_reuse", forbidden)
    context = owner._structural_context_from_current(
        native["index"], {"head": observation.head.to_dict()})
    assert calls == [observation.head.manifest_cid]
    assert context.head == observation.head
    assert context.semantic_state_cid == expected.semantic_state.state_cid
    assert dict(context.coverage) == expected.coverage
    assert context.coverage["inventory_entries"] == context.coverage["captured_entries"] == 1
    assert context.coverage["semantic_symbols"] >= 1
    assert context.coverage["formalized_properties"] == context.coverage["checked_properties"] == 0
    encoded = context.to_dict()
    assert encoded["proof_authority"] is encoded["execution_authority"] is encoded["completion_authority"] is False
    assert native["scheduler"].snapshot()["active_lease_count"] == 0


def test_foreign_reader_owner_is_rejected_before_access(native):
    class ForeignOwner:
        def load(self, cid):
            pytest.fail("foreign owner was consulted")

    with pytest.raises(ValueError, match="exact current native"):
        owner._structural_context_from_current(ForeignOwner(), {"head": native["head"].to_dict()})


@pytest.mark.parametrize("kind", ("mapping", "object"))
def test_receiving_loader_cannot_substitute_an_untyped_manifest(native, monkeypatch, kind):
    real = native["index"].load(native["head"].manifest_cid)
    wrong = real.to_dict() if kind == "mapping" else object()
    monkeypatch.setattr(native["index"], "load", lambda cid: wrong)
    with pytest.raises(ValueError, match="manifest differs"):
        owner._structural_context_from_current(native["index"], {"head": native["head"].to_dict()})


def test_manifest_locator_rebinding_to_another_actual_cas_snapshot_is_rejected(native):
    original = native["head"]
    changed = _changed_head(native)
    assert changed.manifest_cid != original.manifest_cid
    assert changed.snapshot_cid != original.snapshot_cid
    assert native["index"].load(changed.manifest_cid).snapshot.snapshot_cid == changed.snapshot_cid
    mixed = replace(original, manifest_cid=changed.manifest_cid)
    with pytest.raises(ValueError, match="manifest differs"):
        owner._structural_context_from_current(native["index"], {"head": mixed.to_dict()})
    assert native["index"].current(original.repository_id) == changed
    assert native["scheduler"].snapshot()["active_lease_count"] == 0


def test_faulty_loader_returning_a_different_genuine_cas_manifest_is_rejected(native, monkeypatch):
    changed = _changed_head(native)
    actual_different_manifest = native["index"].load(changed.manifest_cid)
    requested = []

    def incorrect_load(cid):
        requested.append(cid)
        return actual_different_manifest

    monkeypatch.setattr(native["index"], "load", incorrect_load)
    with pytest.raises(ValueError, match="manifest differs"):
        owner._structural_context_from_current(native["index"], {"head": native["head"].to_dict()})
    assert requested == [native["head"].manifest_cid]


def test_same_actual_manifest_cannot_be_relabelled_as_another_repository(native):
    original = native["head"]
    other_id = "worktree:independently-different-view"
    relabelled = replace(original, repository_id=other_id,
        ast_revision_id=f"rev:{other_id}:snapshot:{original.snapshot_cid}")
    with pytest.raises(ValueError, match="manifest differs"):
        owner._structural_context_from_current(native["index"], {"head": relabelled.to_dict()})


def test_actual_manifest_and_another_actual_snapshot_cannot_be_mixed(native):
    original = native["head"]
    changed = _changed_head(native)
    # Both components exist in CAS; only this cross-record association is false.
    mixed = replace(original, snapshot_cid=changed.snapshot_cid,
        ast_revision_id=f"rev:{original.repository_id}:snapshot:{changed.snapshot_cid}")
    with pytest.raises(ValueError, match="manifest differs"):
        owner._structural_context_from_current(native["index"], {"head": mixed.to_dict()})


def test_invalid_observed_revision_is_rejected_before_loading_the_actual_manifest(native, monkeypatch):
    observed = {"head": native["head"].to_dict()}
    observed["head"]["ast_revision_id"] = "rev:unrelated:source"

    def forbidden(cid):
        pytest.fail("invalid observed head reached CAS loading")

    monkeypatch.setattr(native["index"], "load", forbidden)
    with pytest.raises(ValueError, match="AST revision"):
        owner._structural_context_from_current(native["index"], observed)
