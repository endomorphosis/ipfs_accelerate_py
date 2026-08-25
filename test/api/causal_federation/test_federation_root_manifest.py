"""Exact, non-authoritative projection checks for the CASF root manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Final

import test_qualification_report as report_validation

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_RELATIVE_PATH = (
    "docs/architecture/causal_event_federation_inventory/federation_root_manifest.json"
)
PROJECTION_REVISION: Final = "67e9fe1c82c357b7bf59a91121b0c33103e7ee41"
PROJECTION_TREE: Final = "a548f4a229f459c53ae46712918cb8391033651f"
PROJECTION_PARENT: Final = "e45c6fe6790f1110b061f8cbefb2ce1dce4d8544"
PROJECTION_BLOB: Final = "838a03ad0710892bbffa37970e0763c74df5e733"
PROJECTION_RAW_SHA256: Final = "a23431f8324cf8aa1fb7938b5be61321bd0d079b86fb231c9f4bafbabd3d4034"
PROJECTION_BYTE_COUNT: Final = 6535

REPORT_ARTIFACTS = (
    (
        report_validation.REPORT_RELATIVE_PATH,
        "4dbf6c2e385295a59c4c0beaaadd6b09623d43de",
        "4a0519e991263c09f08f51a9a07be04043ee7fba8a96c05eedb141e7a1332dff",
        31002,
    ),
    (
        report_validation.MARKDOWN_RELATIVE_PATH,
        "5d8388289e03e995078bfe75443ffc7730cd0492",
        "2f53aea535133a041af840faeb4b670fdb1c675c8fc164050bcab1b60080f330",
        12835,
    ),
    (
        report_validation.TEST_RELATIVE_PATH,
        "9abb3df6297a1d1565ff020be6e01b8f7a61efdb",
        "332a3ee335e6358b83c7ff6cf11b8a499d87a9b82a2a636b38003d9f062713b0",
        75777,
    ),
)

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "manifest_id",
        "artifact_type",
        "artifact_version",
        "program_id",
        "repository_id",
        "root_goal_id",
        "tranche_goal_id",
        "goal_id",
        "status",
        "manifest_kind",
        "authority",
        "starting_baseline",
        "qualification_input_snapshot",
        "qualification_report_projection",
        "benchmark_suite_projection",
        "qualification_summary",
        "final_tree_binding",
        "residual_gap_ids",
        "manifest_projection_binding",
        "rollback",
        "provenance",
        "nonclaims",
    }
)


def _read_manifest() -> tuple[dict[str, Any], bytes]:
    payload = report_validation._read_regular_bytes(
        ROOT,
        MANIFEST_RELATIVE_PATH,
        maximum=256 * 1024,
    )
    return report_validation._decode_json_bytes(payload, "federation root manifest"), payload


def _manifest_id(manifest: dict[str, Any]) -> str:
    body = {key: value for key, value in manifest.items() if key != "manifest_id"}
    return "sha256:" + hashlib.sha256(report_validation._canonical_bytes(body)).hexdigest()


def _artifact_at(revision: str, expected: tuple[str, str, str, int]) -> bytes:
    path, expected_oid, expected_sha, expected_size = expected
    oid, payload = report_validation._blob_at(ROOT, revision, path)
    assert oid == expected_oid
    assert hashlib.sha256(payload).hexdigest() == expected_sha
    assert len(payload) == expected_size
    return payload


def test_projection_commit_and_current_manifest_bytes_are_exact() -> None:
    report_validation._reject_internal_object_alternates(ROOT)
    head = report_validation._git_text(ROOT, "rev-parse", "--verify", "HEAD^{commit}")
    assert (
        report_validation._git_text(
            ROOT, "rev-parse", "--verify", f"{PROJECTION_REVISION}^{{tree}}"
        )
        == PROJECTION_TREE
    )
    assert (
        report_validation._git_text(ROOT, "rev-parse", "--verify", f"{PROJECTION_REVISION}^")
        == PROJECTION_PARENT
    )
    report_validation._require_ancestor(ROOT, PROJECTION_REVISION, head)
    changed = report_validation._git_bytes(
        ROOT,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-z",
        "-r",
        PROJECTION_REVISION,
    )
    assert changed == MANIFEST_RELATIVE_PATH.encode() + b"\x00"

    oid, projected = report_validation._blob_at(ROOT, PROJECTION_REVISION, MANIFEST_RELATIVE_PATH)
    assert oid == PROJECTION_BLOB
    assert hashlib.sha256(projected).hexdigest() == PROJECTION_RAW_SHA256
    assert len(projected) == PROJECTION_BYTE_COUNT
    current = report_validation._verify_worktree_matches_head(
        ROOT, MANIFEST_RELATIVE_PATH, 256 * 1024, head
    )
    assert current == projected


def test_manifest_schema_identity_authority_and_pending_binding_are_exact() -> None:
    manifest, payload = _read_manifest()
    assert set(manifest) == TOP_LEVEL_FIELDS
    assert hashlib.sha256(payload).hexdigest() == PROJECTION_RAW_SHA256
    assert manifest["manifest_id"] == _manifest_id(manifest)
    assert (
        manifest["schema"],
        manifest["artifact_type"],
        manifest["artifact_version"],
        manifest["program_id"],
        manifest["repository_id"],
        manifest["root_goal_id"],
        manifest["tranche_goal_id"],
        manifest["goal_id"],
        manifest["status"],
        manifest["manifest_kind"],
    ) == (
        "casf/release-manifest@1",
        "casf_federation_root_manifest",
        1,
        "agent-supervisor-causal-event-federation-v1",
        "endomorphosis/ipfs_accelerate_py",
        "CASF-G000",
        "CASF-G040",
        "CASF-G043",
        "not_qualified",
        "non_authoritative_repository_projection",
    )
    assert set(manifest["authority"]) == {
        "authoritative",
        "qualification_authority",
        "completion_authority",
        "promotion_authority",
        "scheduling_authority",
        "release_authority",
    }
    assert all(value is False for value in manifest["authority"].values())

    summary = manifest["qualification_summary"]
    assert summary["status"] == "not_qualified"
    assert summary["duckdb_quack_status"] == "not_qualified"
    assert summary["ducklake_quack_status"] == "not_qualified"
    assert summary["disposition"] == "quarantine_recommended"
    assert summary["promotion_decision_ref"] is None
    assert summary["completion_receipt_ref"] is None
    for field in (
        "ducklake_authoritative",
        "ducklake_blocks_core_qualification",
        "contention_free_operation_qualified",
        "promotion_eligible",
        "release_eligible",
        "root_completion_eligible",
        "disposition_authoritative",
        "quarantine_applied",
        "promotion_applied",
        "production_state_changed",
        "authority_created",
        "completion_created",
    ):
        assert summary[field] is False

    final = manifest["final_tree_binding"]
    assert final["status"] == "pending_post_merge_state_owner_acceptance"
    assert all(value is None for key, value in final.items() if key != "status")
    projection = manifest["manifest_projection_binding"]
    assert projection["status"] == "pending_external_git_projection_binding"
    assert all(value is None for key, value in projection.items() if key != "status")

    rollback = manifest["rollback"]
    assert rollback == {
        "status": "not_authorized",
        "scope": [MANIFEST_RELATIVE_PATH],
        "target": {
            "revision": PROJECTION_PARENT,
            "tree_id": report_validation._git_text(
                ROOT, "rev-parse", "--verify", f"{PROJECTION_PARENT}^{{tree}}"
            ),
        },
        "applied": False,
        "executable": False,
        "decision_ref": None,
        "history_rewrite_permitted": False,
    }
    assert manifest["provenance"]["evidence_source"] == "repository_only"
    assert all(
        value is False for key, value in manifest["provenance"].items() if key != "evidence_source"
    )
    assert len(manifest["nonclaims"]) == 4
    assert all(type(item) is str and item for item in manifest["nonclaims"])


def test_hardened_report_projection_and_residual_gaps_are_exact() -> None:
    manifest, _payload = _read_manifest()
    projection = manifest["qualification_report_projection"]
    assert projection["revision"] == PROJECTION_PARENT
    assert projection["tree_id"] == report_validation._git_text(
        ROOT, "rev-parse", "--verify", f"{PROJECTION_PARENT}^{{tree}}"
    )
    assert projection["report_id"] == (
        "sha256:d1525586b485407f0642d9155b97fa0eed8ac40f9e007cc1133798632642ad9e"
    )
    assert len(projection["artifacts"]) == len(REPORT_ARTIFACTS)
    for binding, expected in zip(projection["artifacts"], REPORT_ARTIFACTS, strict=True):
        path, oid, raw_sha, byte_count = expected
        assert binding == {
            "path": path,
            "git_blob_oid": oid,
            "raw_sha256": raw_sha,
            "byte_count": byte_count,
        }
        _artifact_at(PROJECTION_PARENT, expected)

    report_payload = _artifact_at(PROJECTION_PARENT, REPORT_ARTIFACTS[0])
    report = report_validation._validate_report(
        report_validation._decode_json_bytes(report_payload, "bound qualification report")
    )
    assert projection["report_id"] == report["report_id"]
    assert manifest["residual_gap_ids"] == [gap["gap_id"] for gap in report["residual_gaps"]]
    assert tuple(manifest["residual_gap_ids"]) == report_validation.GAP_IDS
    assert (
        manifest["qualification_summary"]["duckdb_quack_status"]
        == (report["qualification"]["profiles"]["duckdb_quack"]["status"])
    )
    assert (
        manifest["qualification_summary"]["ducklake_quack_status"]
        == (report["qualification"]["profiles"]["ducklake_quack"]["status"])
    )


def test_benchmark_projection_and_snapshot_lineage_are_exact() -> None:
    manifest, _payload = _read_manifest()
    suite_projection = manifest["benchmark_suite_projection"]
    assert suite_projection == {
        "schema": "casf/benchmark-suite@1",
        "suite_id": report_validation.SUITE_ID,
        "component_revision": report_validation.SUITE_COMPONENT_REVISION,
        "component_tree_id": report_validation.SUITE_COMPONENT_TREE,
        "projection_revision": report_validation.SUITE_PROJECTION_REVISION,
        "projection_tree_id": report_validation.SUITE_PROJECTION_TREE,
        "path": report_validation.SUITE_RELATIVE_PATH,
        "git_blob_oid": report_validation.SUITE_BLOB,
        "raw_sha256": report_validation.SUITE_RAW_SHA256,
        "suite_status": "not_run",
        "result_artifacts": [],
        "metrics_omitted": True,
        "authoritative": False,
        "promotion_eligible": False,
        "release_eligible": False,
    }
    assert manifest["starting_baseline"] == {
        "revision": report_validation.STARTING_REVISION,
        "tree_id": report_validation.STARTING_TREE,
    }
    assert manifest["qualification_input_snapshot"] == {
        "revision": report_validation.INPUT_REVISION,
        "tree_id": report_validation.INPUT_TREE,
    }
    for revision, tree in (
        (report_validation.STARTING_REVISION, report_validation.STARTING_TREE),
        (report_validation.INPUT_REVISION, report_validation.INPUT_TREE),
        (report_validation.SUITE_COMPONENT_REVISION, report_validation.SUITE_COMPONENT_TREE),
        (report_validation.SUITE_PROJECTION_REVISION, report_validation.SUITE_PROJECTION_TREE),
    ):
        assert (
            report_validation._git_text(ROOT, "rev-parse", "--verify", f"{revision}^{{tree}}")
            == tree
        )
    for ancestor, descendant in (
        (report_validation.STARTING_REVISION, report_validation.INPUT_REVISION),
        (report_validation.SUITE_COMPONENT_REVISION, report_validation.INPUT_REVISION),
        (report_validation.SUITE_PROJECTION_REVISION, report_validation.INPUT_REVISION),
        (report_validation.INPUT_REVISION, PROJECTION_PARENT),
        (PROJECTION_PARENT, PROJECTION_REVISION),
    ):
        report_validation._require_ancestor(ROOT, ancestor, descendant)

    suite_oid, suite_payload = report_validation._blob_at(
        ROOT, report_validation.SUITE_PROJECTION_REVISION, report_validation.SUITE_RELATIVE_PATH
    )
    assert suite_oid == report_validation.SUITE_BLOB
    assert hashlib.sha256(suite_payload).hexdigest() == report_validation.SUITE_RAW_SHA256
    suite = report_validation._validate_suite(
        report_validation._decode_json_bytes(suite_payload, "bound benchmark suite")
    )
    report_payload = _artifact_at(PROJECTION_PARENT, REPORT_ARTIFACTS[0])
    report = report_validation._decode_json_bytes(report_payload, "bound qualification report")
    assert report["benchmark_suite"]["manifest"] == suite
