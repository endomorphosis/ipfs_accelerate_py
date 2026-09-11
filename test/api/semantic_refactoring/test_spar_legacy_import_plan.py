"""Native offline key provenance is preservation, never callback settlement."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ops.agent_supervisor import spar_legacy_import_plan as producer
from scripts.ops.agent_supervisor import spar_merge_owner as native
from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
from test.api.semantic_refactoring.test_spar_merge_owner_bootstrap import (
    preserved as _preserved,
    start,
    attach,
    attach_recovery,
    add_preserved_imports,
)


@pytest.fixture
def preserved(tmp_path):
    return _preserved.__wrapped__(tmp_path)


def prepare(preserved):
    source, manifest, pending, unknown, _ = preserved
    context = {k: manifest[k] for k in producer.CONTEXT_FIELDS}
    train = SimpleNamespace(
        queue=SimpleNamespace(
            target_repository_id=context["repository_id"],
            target_branch=context["target_branch"],
        )
    )
    primary = MergeTrain._dedupe_key(
        train, unknown.canonical_identity, unknown.commit_sha
    )
    body = {
        "request_id": unknown.request_id,
        "task_id": unknown.task_id,
        "canonical_task_id": unknown.canonical_identity,
        "commit_sha": unknown.commit_sha,
        "target_branch": context["target_branch"],
        "status": "integrated_pending_validation",
        "integrated": True,
        "accepted": False,
        "callback_owned_integration": True,
        "opaque_callback_evidence": {
            "result": "unknown",
            "claim_token": unknown.claim_token,
        },
    }
    train_dir = source / "train" / "receipts"
    train_dir.mkdir(parents=True)
    path = train_dir / (primary + ".json")
    path.write_text(json.dumps(body))
    # Opaque signing bytes must survive, not be loaded into an acceptance API.
    key = source / "private" / "callback-signing-material"
    key.parent.mkdir()
    key.write_bytes(b"retained-test-key-never-used-to-sign")
    key.chmod(0o600)
    return source, context, unknown, path, body


def bytes_before(root):
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def test_real_producer_clone_owner_preserves_unknown_callback_and_claim(
    preserved, tmp_path
):
    source, context, unknown, path, body = prepare(preserved)
    before = bytes_before(source)
    result = producer.produce_offline_import_plan(
        offline_root=source, destination=tmp_path / "inspect", context=context
    )
    assert result["schema"] == producer.SCHEMA
    assert all(
        result[k] is False
        for k in (
            "capture_coherent",
            "consumer_closed",
            "callback_settled",
            "source_admitted",
            "completion_authority",
        )
    )
    manifest = result["manifest"]
    prepared = native.prepare_offline_clone(
        offline_root=source, destination=tmp_path / "migrated", manifest=manifest
    )
    assert (
        tmp_path / "migrated/private/callback-signing-material"
    ).read_bytes() == before["private/callback-signing-material"]
    server = start(prepared, tmp_path / "owner")
    conn = recovery_conn = None
    try:
        conn, queue = attach(server)
        row = json.loads(
            queue.call("get", request_id=unknown.request_id)["request_json"]
        )
        assert row["claim_token"] == unknown.claim_token
        assert row["claim_generation"] == unknown.claim_generation
        assert row["consumer_id"] == "retained-old-consumer"
        assert row["status"] == "processing"
        recovery_conn, recovery = attach_recovery(server, manifest)
        receipt = recovery.get_receipt(path.stem, revision=1)
        assert receipt["receipt"] == body
        assert receipt["receipt_cid"] == native._cid(body)
    finally:
        if conn:
            conn.close()
        if recovery_conn:
            recovery_conn.close()
        server.stop()
    assert bytes_before(source) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "orphan",
        "task",
        "candidate",
        "canonical",
        "target",
        "filename",
        "ambiguous-prefix",
    ],
)
def test_receipt_provenance_mismatch_refuses(preserved, tmp_path, mutation):
    source, context, unknown, path, body = prepare(preserved)
    changes = {
        "orphan": ("request_id", "foreign-request"),
        "task": ("task_id", "SPAR-foreign"),
        "candidate": ("commit_sha", "f" * 40),
        "canonical": ("canonical_task_id", "foreign"),
        "target": ("target_branch", "foreign"),
    }
    if mutation in changes:
        field, value = changes[mutation]
        body[field] = value
        path.write_text(json.dumps(body))
    elif mutation == "filename":
        path.rename(path.with_name("f" * 64 + ".json"))
    else:
        path.rename(path.with_name("quarantine-" + unknown.request_id + "!.json"))
    before = bytes_before(source)
    with pytest.raises(native.SparMergeOwnerError):
        producer.produce_offline_import_plan(
            offline_root=source, destination=tmp_path / "inspect", context=context
        )
    assert bytes_before(source) == before


def test_full_cursor_and_quarantine_mapping_preserves_content(preserved, tmp_path):
    source, context, unknown, path, body = prepare(preserved)
    # Reuse the existing native cursor fixture, but remove its unrelated
    # deliberately caller-supplied historical receipt fixture.
    _, manifest, _, _, _ = preserved
    cursor, _ = add_preserved_imports(source, manifest)
    for item in manifest["receipt_imports"]:
        (source / item["path"]).unlink()
    quarantine = path.with_name("quarantine-" + unknown.request_id + ".json")
    quarantine.write_text(json.dumps({**body, "status": "quarantined"}))
    result = producer.produce_offline_import_plan(
        offline_root=source, destination=tmp_path / "inspect", context=context
    )
    assert len(result["manifest"]["cursor_imports"]) == 1
    assert {x["receipt_key"] for x in result["manifest"]["receipt_imports"]} == {
        path.stem,
        quarantine.stem,
    }
    prepared = native.prepare_offline_clone(
        offline_root=source,
        destination=tmp_path / "migrate",
        manifest=result["manifest"],
    )
    assert prepared.cursor_imports[0]["cursors"] == cursor["cursors"]


@pytest.mark.parametrize(
    "mutation",
    ["cursor-content", "cursor-scope", "symlink", "extra-context", "held-lock"],
)
def test_fail_closed_before_mapping_or_owner_start(preserved, tmp_path, mutation):
    source, context, unknown, path, body = prepare(preserved)
    if mutation.startswith("cursor"):
        _, manifest, _, _, _ = preserved
        add_preserved_imports(source, manifest)
        for entry in manifest["receipt_imports"]:
            (source / entry["path"]).unlink()
        cursor_path = source / manifest["cursor_imports"][0]["path"]
        if mutation == "cursor-content":
            cursor = json.loads(cursor_path.read_text())
            cursor["cursors"]["priority_task_cids"] = "changed-without-native-cid"
            cursor_path.write_text(json.dumps(cursor))
        else:
            context["scope_bindings"] = [
                {
                    **context["scope_bindings"][0],
                    "attempt_root": str(tmp_path / "foreign"),
                }
            ]
    elif mutation == "symlink":
        (source / "foreign").symlink_to(path)
    elif mutation == "extra-context":
        context["consumer_closed"] = True
    descriptor = None
    try:
        if mutation == "held-lock":
            import fcntl

            descriptor = os.open(source / "merge_queue.duckdb", os.O_RDONLY)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(native.SparMergeOwnerError):
            producer.produce_offline_import_plan(
                offline_root=source, destination=tmp_path / "inspect", context=context
            )
    finally:
        if descriptor is not None:
            os.close(descriptor)


def test_context_target_cannot_rebind_a_canonical_receipt(preserved, tmp_path):
    source, context, unknown, path, body = prepare(preserved)
    context["repository_id"] = "repo:foreign"
    with pytest.raises(native.SparMergeOwnerError, match="target is unbound"):
        producer.produce_offline_import_plan(
            offline_root=source, destination=tmp_path / "inspect", context=context
        )


def test_overlap_and_existing_destination_refused_without_input_effect(
    preserved, tmp_path
):
    source, context, unknown, path, body = prepare(preserved)
    before = bytes_before(source)
    with pytest.raises(native.SparMergeOwnerError, match="overlaps"):
        producer.produce_offline_import_plan(
            offline_root=source, destination=source / "nested", context=context
        )
    destination = tmp_path / "exists"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        producer.produce_offline_import_plan(
            offline_root=source, destination=destination, context=context
        )
    assert bytes_before(source) == before


def test_replays_committed_wal_only_in_copy_without_owner_discovery(
    preserved, tmp_path, monkeypatch
):
    import subprocess
    import sys
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state

    source, context, unknown, path, body = prepare(preserved)
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import os, sys
sys.path.insert(0, sys.argv[2])
import duckdb
c = duckdb.connect(sys.argv[1], config={'threads': 1})
c.execute('CREATE TABLE captured_foreign_evidence(value VARCHAR)')
c.execute("INSERT INTO captured_foreign_evidence VALUES ('original committed WAL')")
os._exit(0)
""",
            str(source / "merge_queue.duckdb"),
            str(Path(duckdb.__file__).resolve().parent.parent),
        ],
        env={"PATH": os.defpath},
        capture_output=True,
        timeout=15,
    )
    assert child.returncode == 0
    assert (source / "merge_queue.duckdb.wal").is_file()
    before = bytes_before(source)

    def no_discovery(*args, **kwargs):
        raise AssertionError(
            "offline producer must not discover a live owner or load its credentials"
        )

    monkeypatch.setattr(duckdb_state, "discover_live_quack_endpoint", no_discovery)
    result = producer.produce_offline_import_plan(
        offline_root=source, destination=tmp_path / "inspect", context=context
    )
    assert result["manifest"]["wal"] == "merge_queue.duckdb.wal"
    assert len(result["preserved_inventory"]["captured_foreign_evidence"]["rows"]) == 1
    assert bytes_before(source) == before


def test_changed_input_refuses_before_any_database_open(
    preserved, tmp_path, monkeypatch
):
    source, context, unknown, path, body = prepare(preserved)
    original_copy = native.copy_entry
    changed = False

    def change_after_copy(root, entry, destination, **kwargs):
        nonlocal changed
        result = original_copy(root, entry, destination, **kwargs)
        if destination is not None and not changed:
            changed = True
            path.write_text(json.dumps({**body, "status": "changed-during-copy"}))
        return result

    def no_database_open(*args, **kwargs):
        raise AssertionError("changed input must refuse before database inspection")

    monkeypatch.setattr(native, "copy_entry", change_after_copy)
    monkeypatch.setattr(native, "open_duckdb_connection", no_database_open)
    with pytest.raises(native.SparMergeOwnerError):
        producer.produce_offline_import_plan(
            offline_root=source, destination=tmp_path / "inspect", context=context
        )
    assert changed


def test_produced_manifest_survives_two_real_quack_owner_generations(
    preserved, tmp_path
):
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        probe_quack_capabilities,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
        QuackCapabilityStatus,
    )

    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip("real preinstalled Quack unavailable")
    source, context, unknown, path, body = prepare(preserved)
    before = bytes_before(source)
    plan = producer.produce_offline_import_plan(
        offline_root=source, destination=tmp_path / "inspect", context=context
    )
    result = native.qualify_offline_bundle(
        offline_root=source,
        destination=tmp_path / "real-qualified",
        manifest=plan["manifest"],
    )
    assert result["qualified"] is True, result
    assert result["transport"] == "real_quack"
    assert len(result["cycles"]) == 2
    assert all(
        c["closed"] and c["startup_and_checkpoint_writer_lock"]
        for c in result["cycles"]
    )
    assert result["completion_authority"] is False
    assert result["live_custody_qualified"] is False
    assert bytes_before(source) == before
