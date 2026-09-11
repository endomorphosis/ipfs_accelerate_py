"""The deployed five-stage profile cannot silently become main's eight stages."""

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from scripts.ops.agent_supervisor import spar_legacy_import_plan as producer
from scripts.ops.agent_supervisor import spar_merge_owner as native
from test.api.semantic_refactoring.test_spar_legacy_import_plan import (
    bytes_before,
    prepare,
    preserved,
)
from test.api.semantic_refactoring.test_spar_merge_owner_bootstrap import (
    add_preserved_imports,
)


@pytest.mark.parametrize("missing", [
    ("false_completed_requests",),
    ("false_pending_requests",),
    ("false_processing_requests",),
    ("false_completed_requests", "false_pending_requests", "false_processing_requests"),
])
def test_valid_old_cursor_identity_cannot_reset_missing_recovery_stages(
    preserved, tmp_path, missing,
):
    source, context, *_ = prepare(preserved)
    manifest = preserved[1]
    cursor, _ = add_preserved_imports(source, manifest)
    for entry in manifest["receipt_imports"]:
        (source / entry["path"]).unlink()
    for stage in missing:
        del cursor["cursors"][stage]
    cursor["state_id"] = content_identity({
        key: value for key, value in cursor.items() if key != "state_id"
    })
    (source / manifest["cursor_imports"][0]["path"]).write_text(json.dumps(cursor))
    before = bytes_before(source)

    with pytest.raises(native.SparMergeOwnerError, match="cursor differs from native binding"):
        producer.produce_offline_import_plan(
            offline_root=source, destination=tmp_path / "inspection", context=context,
        )

    assert bytes_before(source) == before
