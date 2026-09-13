"""Known five-scan snapshots migrate without resetting retained scan history."""

import copy
import hashlib
import json

import pytest

from scripts.ops.agent_supervisor import spar_merge_owner as native
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from test.api.semantic_refactoring.test_spar_merge_owner_bootstrap import (
    add_preserved_imports,
    attach_recovery,
    preserved as _preserved_fixture,
    refresh_manifest,
    start,
)

preserved = _preserved_fixture
LEGACY_STAGES = {
    "priority_task_cids", "completed_requests", "pending_requests",
    "quarantined_requests", "processing_requests",
}
NEW_STAGES = {
    "false_completed_requests", "false_pending_requests", "false_processing_requests",
}


def rewrite_cursor(source, manifest, cursor, *, rehash=True):
    if rehash:
        cursor["state_id"] = content_identity(
            {key: value for key, value in cursor.items() if key != "state_id"}
        )
    path = source / manifest["cursor_imports"][0]["path"]
    path.write_text(json.dumps(cursor, indent=2) + "\n")
    refresh_manifest(source, manifest)
    return path


def legacy_cursor(source, manifest):
    cursor, receipts = add_preserved_imports(source, manifest)
    cursor["cursors"] = {
        key: value for key, value in cursor["cursors"].items() if key in LEGACY_STAGES
    }
    path = rewrite_cursor(source, manifest, cursor)
    return cursor, receipts, path


@pytest.mark.parametrize("legacy", [True, False], ids=["five-stage-v1", "eight-stage"])
def test_cursor_import_preserves_source_identity_and_durable_history(
    preserved, tmp_path, legacy
):
    source, manifest, *_ = preserved
    if legacy:
        cursor, receipts, path = legacy_cursor(source, manifest)
    else:
        cursor, receipts = add_preserved_imports(source, manifest)
        path = rewrite_cursor(source, manifest, cursor)
    original = path.read_bytes()
    original_manifest = copy.deepcopy(manifest)
    normalized = {
        **cursor["cursors"], **({stage: "" for stage in NEW_STAGES} if legacy else {})
    }
    state_cid = native._cid(normalized)
    prepared = native.prepare_offline_clone(
        offline_root=source, destination=tmp_path / "clone", manifest=manifest
    )
    assert prepared.manifest == original_manifest == manifest
    assert path.read_bytes() == original
    assert (prepared.database_path.parent / path.relative_to(source)).read_bytes() == original
    assert prepared.cursor_imports == ({
        "scope_cid": manifest["cursor_imports"][0]["scope_cid"],
        "cursors": normalized,
        "state_cid": state_cid,
    },)
    if legacy:
        assert len(prepared.cursor_normalizations) == 1
        evidence = prepared.cursor_normalizations[0]
        assert evidence == {
            "schema": "ipfs_accelerate_py/agent-supervisor/legacy-recovery-cursor-normalization@1",
            "profile": "schema-v1-five-stage-to-eight-stage",
            "source_manifest_cid": prepared.manifest_cid,
            "source_path": path.relative_to(source).as_posix(),
            "source_sha256": hashlib.sha256(original).hexdigest(),
            "source_schema": cursor["schema"],
            "source_state_id": cursor["state_id"],
            "source_cursor_state_cid": native._cid(cursor["cursors"]),
            "scope_cid": manifest["cursor_imports"][0]["scope_cid"],
            "added_cursors": {stage: "" for stage in NEW_STAGES},
            "normalized_cursor_state_cid": state_cid,
        }
    else:
        assert prepared.cursor_normalizations == ()
        assert state_cid == native._cid(cursor["cursors"])

    advanced = {**normalized, "false_completed_requests": "retained:next-scan"}
    for generation in (1, 2):
        server = start(prepared, tmp_path / "owner")
        client = None
        try:
            assert server.identity.generation == generation
            client, api = attach_recovery(server, manifest)
            head = api.load_cursors()
            if generation == 1:
                assert head["cursors"] == normalized and head["revision"] == 0
                api.cas_cursors(
                    expected_revision=0,
                    expected_state_cid=state_cid,
                    cursors=advanced,
                    operation_id="migration-test:advance",
                )
            else:
                assert head["cursors"] == advanced and head["revision"] == 1
            with server._owner_transaction_lock:
                rows = server._connection.execute(
                    "SELECT revision,state_cid,cursors_json "
                    "FROM legacy_merge_recovery_cursor_history ORDER BY revision"
                ).fetchall()
            assert [(row[0], row[1], json.loads(row[2])) for row in rows] == [
                (0, state_cid, normalized),
                (1, native._cid(advanced), advanced),
            ]
            for revision, receipt in enumerate(receipts, 1):
                assert api.get_receipt("train:preserved", revision=revision)["receipt"] == receipt
        finally:
            if client is not None:
                client.close()
            server.stop()
    assert path.read_bytes() == original
    assert manifest == original_manifest


@pytest.mark.parametrize("failure", [
    "missing_legacy_stage", "partial_new_stages", "unknown_stage", "substituted_stage",
    "boolean_value", "oversize_value", "wrong_state_id", "wrong_scope",
    "wrong_source_digest", "wrong_schema", "future_target_profile",
])
def test_five_stage_migration_refuses_unverified_or_unknown_profiles(
    preserved, tmp_path, monkeypatch, failure
):
    source, manifest, *_ = preserved
    cursor, _receipts, path = legacy_cursor(source, manifest)
    if failure == "missing_legacy_stage":
        cursor["cursors"].pop("pending_requests")
    elif failure == "partial_new_stages":
        cursor["cursors"]["false_pending_requests"] = ""
    elif failure == "unknown_stage":
        cursor["cursors"]["unrecognized_requests"] = ""
    elif failure == "substituted_stage":
        cursor["cursors"].pop("pending_requests")
        cursor["cursors"]["false_pending_requests"] = ""
    elif failure == "boolean_value":
        cursor["cursors"]["pending_requests"] = False
    elif failure == "oversize_value":
        cursor["cursors"]["pending_requests"] = "x" * 4097
    elif failure == "wrong_state_id":
        cursor["state_id"] = "sha256:" + "0" * 64
    elif failure == "wrong_scope":
        cursor["attempt_root"] = str(tmp_path / "foreign-attempts")
    elif failure == "wrong_source_digest":
        pass
    elif failure == "wrong_schema":
        cursor["schema"] = cursor["schema"].replace("@1", "@2")
    elif failure == "future_target_profile":
        monkeypatch.setattr(native, "STAGES", (*native.STAGES, "future_requests"))
    rewrite_cursor(source, manifest, cursor, rehash=failure != "wrong_state_id")
    if failure == "wrong_source_digest":
        entry = next(item for item in manifest["files"] if item["path"] == path.relative_to(source).as_posix())
        entry["sha256"] = "0" * 64
    with pytest.raises(native.SparMergeOwnerError):
        native.prepare_offline_clone(
            offline_root=source, destination=tmp_path / "refused", manifest=manifest
        )
