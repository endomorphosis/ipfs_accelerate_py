"""Portable stdlib receiving controls; authored data, no native owners/jobs."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).with_name("audit_codebase_source_delta.py")
SPEC = importlib.util.spec_from_file_location("source_delta_pure_reader", MODULE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def entry(path, raw=None, *, opaque=False, acquisition="captured", disposition="filesystem"):
    result = {"schema": "ipfs-datasets.software-contracts.semantic-snapshot-entry@3", "path": path,
        "raw_path_hex": path.encode().hex(), "kind": "opaque" if opaque else "source",
        "size_bytes": 65537 if opaque else len(raw), "source_cid": None if opaque else audit.cid(raw, source=True),
        "opaque_reason": "over_capture_limit" if opaque else None, "git_blob_oid": None,
        "acquisition": "opaque" if opaque else acquisition, "disposition": "opaque" if opaque else disposition,
        "head_blob_oid": None, "index_blob_oids": {}}
    result["entry_cid"] = audit.structured(result)
    return result


def state(snapshot):
    result = {"schema": "ipfs-datasets.software-contracts.semantic-index@2", "repository_id": snapshot["repository_id"],
        "symbols": [], "edges": [], "extractor_name": "authored-transport-fixture", "extractor_version": "1",
        "artifacts": [{"schema": "ipfs-datasets.software-contracts.semantic-artifact@1",
            "artifact_id": "artifact:snapshot-evidence", "kind": "repository_snapshot", "path": None,
            "source_cid": snapshot["snapshot_cid"], "confidence": "exact", "metadata": {"snapshot": deepcopy(snapshot)}}]}
    identity = {key: value for key, value in result.items() if key != "schema"}
    identity.update(schema="ipfs-datasets.software-contracts.semantic-repository-state@1", semantic_index_schema=result["schema"])
    result["state_cid"] = audit.structured(identity)
    return result


def manifest(items, artifacts, *, untracked=()):
    entries = sorted([entry(path, raw, opaque=raw is None, acquisition="working-captured" if path in untracked else "captured",
        disposition="untracked" if path in untracked else "filesystem") for path, raw in items.items()], key=lambda row: row["raw_path_hex"])
    snapshot = {"schema": "ipfs-datasets.software-contracts.semantic-repository-snapshot@4", "repository_id": "fixture:delta",
        "entries": entries, "mode": "filesystem", "max_entries": 512, "max_file_bytes": 65536,
        "git_tree": None, "git_commit": None, "exclusions": []}
    snapshot["snapshot_cid"] = audit.structured(snapshot)
    units = []
    for row in entries:
        ast_cid, status = None, "opaque" if row["opaque_reason"] else "unindexed"
        if row["opaque_reason"] is None:
            artifacts[row["source_cid"]] = items[row["path"]]
            if row["path"].endswith(".py"):
                ast = {"schema": "ipfs-datasets.software-contracts.ast-ir@1.0.0", "provenance": {
                    "source_cid": row["source_cid"], "path": row["path"], "repository_id": snapshot["repository_id"],
                    "revision": "snapshot:" + snapshot["snapshot_cid"], "repository_tree_cid": snapshot["snapshot_cid"]},
                    "frontend": {}, "module": {}, "scopes": [], "symbols": [], "imports": [], "references": [],
                    "calls": [], "effects": [], "diagnostics": [], "unsupported": []}
                raw = audit.wire(ast); ast_cid = audit.cid(raw); artifacts[ast_cid] = raw; status = "ok"
        units.append({"schema": "codebase-ir-structural-unit@1", "source_key": "raw:" + row["raw_path_hex"],
                      "entry_cid": row["entry_cid"], "ast_cid": ast_cid, "parse_status": status})
    result = {"schema": "codebase-ir-structural-manifest@1", "authority": "structural_only", "snapshot": snapshot,
        "semantic_state": state(snapshot), "ast_revision_id": "rev:" + snapshot["repository_id"] + ":snapshot:" + snapshot["snapshot_cid"],
        "units": units, "coverage": {"inventory_entries": len(entries), "captured_entries": sum(row["opaque_reason"] is None for row in entries),
            "ast_ok": sum(row["parse_status"] == "ok" for row in units), "ast_partial": 0, "ast_failed": 0,
            "opaque_entries": sum(row["parse_status"] == "opaque" for row in units),
            "unindexed_entries": sum(row["parse_status"] == "unindexed" for row in units), "semantic_symbols": 0,
            "formalized_properties": 0, "checked_properties": 0}}
    return result


def receipt(manifest, previous, number):
    result = {"schema": "codebase-publication-receipt@1", "operation_id": "authored-" + str(number),
        "previous_head": previous, "repository_id": manifest["snapshot"]["repository_id"], "generation": number,
        "manifest_cid": audit.structured(manifest), "snapshot_cid": manifest["snapshot"]["snapshot_cid"],
        "ast_revision_id": manifest["ast_revision_id"]}
    result["request_cid"] = audit.structured({"schema": "codebase-publication-request@1", "repository_id": result["repository_id"],
        "manifest_cid": result["manifest_cid"], "expected_head": previous})
    head = {key: result[key] for key in audit.HEAD_FIELDS - {"schema", "receipt_cid"}}
    head.update(schema="codebase-head@1", receipt_cid=audit.structured(result))
    return result, head


def sides(manifest):
    entries = {"raw:" + row["raw_path_hex"]: row for row in manifest["snapshot"]["entries"]}
    result = {}
    for unit in manifest["units"]:
        row = entries[unit["source_key"]]
        member = {"source_key": unit["source_key"], "path": row["path"], "raw_path_hex": row["raw_path_hex"],
            "entry_cid": row["entry_cid"], "source_cid": row["source_cid"], "ast_cid": unit["ast_cid"],
            "parse_status": unit["parse_status"], "source_size_bytes": row["size_bytes"], "opaque_reason": row["opaque_reason"]}
        result[unit["source_key"]] = {"entry": row, "member": member}
    return result


def fixture():
    artifacts = {}
    old = manifest({"calc.py": b"value = 1\n", "delete.py": b"delete = 1\n", "rename.py": b"rename = 1\n",
                    "retained.txt": b"same\n", "opaque.bin": None}, artifacts)
    new = manifest({"calc.py": b"value = 2\n", "added.py": b"added = 1\n", "renamed.py": b"rename = 1\n",
                    "retained.txt": b"same\n", "opaque.bin": None}, artifacts)
    old_receipt, old_head = receipt(old, None, 1)
    new_receipt, new_head = receipt(new, old_head, 2)
    left, right = sides(old), sides(new)
    ledger = []
    for key in sorted(set(left) | set(right)):
        previous, current = left.get(key), right.get(key)
        classification = "added" if previous is None else "removed" if current is None else "retained" if key.endswith("opaque.bin".encode().hex()) or key.endswith("retained.txt".encode().hex()) else "changed"
        source_comparison = "different" if classification == "changed" else "equal" if previous is not None and current is not None and previous["entry"]["opaque_reason"] is None else "unavailable"
        ast_comparison = "different" if previous is not None and current is not None and previous["member"]["ast_cid"] is not None else "unavailable"
        ledger.append({"source_key": key, "classification": classification, "previous": previous, "current": current,
            "source_bytes_comparison": source_comparison, "ast_identity_comparison": ast_comparison})
    value = {"schema": audit.DELTA_SCHEMA, "previous_head": old_head, "current_head": new_head,
        "previous_publication_receipt": old_receipt, "current_publication_receipt": new_receipt,
        "previous_membership_cid": audit.structured([row["member"] for row in left.values()]),
        "current_membership_cid": audit.structured([row["member"] for row in right.values()]),
        "capture_policy": {"max_entries": 512, "max_file_bytes": 65536, "exclusions": []}, "ledger": ledger,
        "coverage": {"previous_entries": 5, "current_entries": 5, "union_entries": 7,
            "classifications": {"retained": 2, "changed": 1, "added": 2, "removed": 2},
            "source_bytes_comparisons": {"equal": 1, "different": 1, "unavailable": 5},
            "ast_identity_comparisons": {"equal": 0, "different": 1, "unavailable": 6}},
        "limits": {"max_inventory_entries": 1024, "max_union_entries": 2048, "max_file_bytes": 65536,
                   "max_manifest_bytes": 4 * audit.MIB, "max_delta_bytes": 8 * audit.MIB}, "optimized": True,
        "implementation": {"files": {"authored.producer": "a" * 64}, "sha256": audit.sha(b'{"authored.producer":"' + b'a' * 64 + b'"}'),
            "scope": "listed_local_files_only_not_execution_attestation"},
        "authority": {key: False for key in audit.AUTHORITY}, "numerical_reuse": False, "model_advanced": False,
        "removal_scope": "absent_from_current_complete_capture", "physical_absence_verified": False}
    return {"artifact_cid": audit.structured(value), "value": value}, old, new, old_receipt, new_receipt, artifacts


class DeltaReceivingControls(unittest.TestCase):
    def setUp(self):
        self.envelope, self.previous, self.current, self.old_receipt, self.new_receipt, self.artifacts = fixture()

    def verify(self):
        return audit.verify_source_delta(self.envelope, self.previous, self.current, self.old_receipt, self.new_receipt,
                                         get_artifact=lambda wanted, source=False: self.artifacts[wanted])

    def rehash(self):
        self.envelope["artifact_cid"] = audit.structured(self.envelope["value"])

    def test_full_join_positive_and_opaque_unknown(self):
        result = self.verify()
        self.assertEqual(result["coverage"]["union_entries"], 7)
        self.assertFalse(result["physical_absence_verified"])
        self.assertFalse(result["native_ast_schema_validation_here"])

    def test_default_reference_only_flag_changes(self):
        self.verify(); self.envelope["value"]["optimized"] = False; self.rehash(); self.verify()

    def test_omitted_complete_member_rehashed_refused(self):
        value = self.envelope["value"]
        value["ledger"] = value["ledger"][:-1]
        value["previous_membership_cid"] = audit.structured([row["previous"]["member"] for row in value["ledger"] if row["previous"]])
        value["current_membership_cid"] = audit.structured([row["current"]["member"] for row in value["ledger"] if row["current"]])
        value["coverage"] = audit.delta_coverage(value["ledger"])
        self.rehash()
        with self.assertRaisesRegex(audit.SourceDeltaAuditError, "complete ordered source union"):
            self.verify()

    def test_duplicated_union_refused(self):
        self.envelope["value"]["ledger"].append(deepcopy(self.envelope["value"]["ledger"][0])); self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_reordered_union_refused(self):
        self.envelope["value"]["ledger"].reverse(); self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_wrong_classification_refused(self):
        self.envelope["value"]["ledger"][0]["classification"] = "retained"; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_opaque_byte_equality_claim_refused(self):
        row = next(row for row in self.envelope["value"]["ledger"] if row["source_key"].endswith("opaque.bin".encode().hex()))
        row["source_bytes_comparison"] = "equal"; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_global_ast_retention_claim_refused(self):
        row = next(row for row in self.envelope["value"]["ledger"] if row["classification"] == "changed")
        row["ast_identity_comparison"] = "equal"; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_prefix_unknown_not_complete_absence(self):
        self.envelope["value"]["complete"] = False; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_physical_absence_authority_refused(self):
        self.envelope["value"]["physical_absence_verified"] = True; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_removal_scope_expansion_refused(self):
        self.envelope["value"]["removal_scope"] = "absent_from_filesystem"; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_numerical_reuse_refused(self):
        self.envelope["value"]["numerical_reuse"] = True; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_model_advanced_refused(self):
        self.envelope["value"]["model_advanced"] = True; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_proof_authority_refused(self):
        self.envelope["value"]["authority"]["proof_authority"] = True; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_coverage_bool_alias_refused(self):
        self.envelope["value"]["coverage"]["classifications"]["changed"] = True; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_limits_bool_alias_refused(self):
        self.envelope["value"]["limits"]["max_union_entries"] = True; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_float_identity_refused(self):
        self.envelope["value"]["coverage"]["current_entries"] = 5.0
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_foreign_head_refused(self):
        self.envelope["value"]["current_head"]["repository_id"] = "foreign"; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_foreign_publication_refused(self):
        self.envelope["value"]["current_publication_receipt"]["operation_id"] = "foreign-operation"; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_stale_publication_request_refused(self):
        self.new_receipt["request_cid"] = audit.structured({"foreign": True})
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_capture_policy_change_refused(self):
        self.envelope["value"]["capture_policy"]["exclusions"] = ["hidden.py"]; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_undersized_union_budget_refused(self):
        self.envelope["value"]["limits"]["max_union_entries"] = 6; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_member_entry_binding_rehashed_refused(self):
        self.envelope["value"]["ledger"][0]["current"]["member"]["source_size_bytes"] += 1; self.rehash()
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_missing_raw_source_refused(self):
        wanted = self.current["snapshot"]["entries"][0]["source_cid"]; del self.artifacts[wanted]
        with self.assertRaises(KeyError): self.verify()

    def test_tampered_raw_source_refused(self):
        wanted = self.current["snapshot"]["entries"][0]["source_cid"]; self.artifacts[wanted] = b"wrong"
        with self.assertRaisesRegex(audit.SourceDeltaAuditError, "captured raw source CAS"): self.verify()

    def test_tampered_ast_cas_refused(self):
        wanted = self.current["units"][0]["ast_cid"]; self.artifacts[wanted] += b" "
        with self.assertRaisesRegex(audit.SourceDeltaAuditError, "native AST CAS"): self.verify()

    def test_rehashed_ast_foreign_provenance_refused(self):
        unit = self.current["units"][0]; ast = audit.parse(self.artifacts[unit["ast_cid"]]); ast["provenance"]["revision"] = "snapshot:foreign"
        raw = audit.wire(ast); unit["ast_cid"] = audit.cid(raw); self.artifacts[unit["ast_cid"]] = raw
        head = deepcopy(self.envelope["value"]["current_head"]); head["manifest_cid"] = audit.structured(self.current)
        with self.assertRaisesRegex(audit.SourceDeltaAuditError, "provenance"):
            audit.manifest_members(self.current, head, get_artifact=lambda wanted, source=False: self.artifacts[wanted])

    def test_rehashed_semantic_snapshot_evidence_refused(self):
        self.current["semantic_state"]["artifacts"][0]["source_cid"] = audit.structured({"wrong": True})
        identity = {key: value for key, value in self.current["semantic_state"].items() if key not in {"schema", "state_cid"}}
        identity.update(schema="ipfs-datasets.software-contracts.semantic-repository-state@1", semantic_index_schema=self.current["semantic_state"]["schema"])
        self.current["semantic_state"]["state_cid"] = audit.structured(identity)
        head = deepcopy(self.envelope["value"]["current_head"]); head["manifest_cid"] = audit.structured(self.current)
        with self.assertRaisesRegex(audit.SourceDeltaAuditError, "snapshot evidence"):
            audit.manifest_members(self.current, head, get_artifact=lambda wanted, source=False: self.artifacts[wanted])

    def test_rehashed_native_unit_duplicate_refused(self):
        self.current["units"][1] = deepcopy(self.current["units"][0])
        head = deepcopy(self.envelope["value"]["current_head"]); head["manifest_cid"] = audit.structured(self.current)
        with self.assertRaisesRegex(audit.SourceDeltaAuditError, "omitted/duplicated/reordered"):
            audit.manifest_members(self.current, head, get_artifact=lambda wanted, source=False: self.artifacts[wanted])

    def test_native_resources_bool_alias_refused(self):
        with self.assertRaises(audit.SourceDeltaAuditError): audit.native_resources({"active_lease_count": False, "waiting_request_count": 0})

    def test_telemetry_floats_remain_outside_cid_encoding(self):
        self.assertTrue(audit.observation_same({"threshold": 70.0}, {"threshold": 70.0}))
        self.assertFalse(audit.observation_same({"threshold": 70.0}, {"threshold": 70}))
        with self.assertRaises(audit.SourceDeltaAuditError): audit.structured({"threshold": 70.0})
        with self.assertRaises(audit.SourceDeltaAuditError): audit.observation_same({"threshold": float("nan")}, {})

    def test_duplicate_json_member_refused(self):
        with self.assertRaises(audit.SourceDeltaAuditError): audit.parse(b'{"x":1,"x":2}')

    def test_nonfinite_json_refused(self):
        with self.assertRaises(audit.SourceDeltaAuditError): audit.parse(b'{"x":NaN}')

    def test_wrong_cid_codec_refused(self):
        with self.assertRaises(audit.SourceDeltaAuditError): audit.check_cid(audit.cid(b"raw", source=True))


class GuardedFileControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.reader = audit.Reader(self.root)

    def test_regular_file_two_archive_passes(self):
        (self.root / "item").write_bytes(b"original")
        self.assertEqual(self.reader.raw("item"), b"original")
        self.assertEqual(self.reader.whole_archive(), self.reader.whole_archive())

    def test_parent_escape_refused(self):
        with self.assertRaises(audit.SourceDeltaAuditError): self.reader.path("../outside")

    def test_absolute_locator_refused(self):
        with self.assertRaises(audit.SourceDeltaAuditError): self.reader.path("/tmp/outside")

    def test_symlink_locator_refused(self):
        (self.root / "item").write_bytes(b"raw"); (self.root / "alias").symlink_to("item")
        with self.assertRaises(audit.SourceDeltaAuditError): self.reader.raw("alias")

    def test_hardlink_refused(self):
        (self.root / "item").write_bytes(b"raw"); os.link(self.root / "item", self.root / "alias")
        with self.assertRaises(audit.SourceDeltaAuditError): self.reader.raw("item")

    def test_oversized_read_refused(self):
        (self.root / "item").write_bytes(b"raw")
        with self.assertRaises(audit.SourceDeltaAuditError): self.reader.raw("item", 2)

    def test_file_change_between_reads_refused(self):
        (self.root / "item").write_bytes(b"original"); self.reader.raw("item")
        (self.root / "item").write_bytes(b"changed")
        with self.assertRaises(audit.SourceDeltaAuditError): self.reader.raw("item")

    def test_membership_change_visible_in_whole_archive(self):
        before = self.reader.whole_archive(); (self.root / "item").write_bytes(b"added")
        self.assertNotEqual(before["inventory_cid"], self.reader.whole_archive()["inventory_cid"])

    def test_fifo_is_refused_without_blocking(self):
        os.mkfifo(self.root / "pipe")
        with self.assertRaises(audit.SourceDeltaAuditError): self.reader.raw("pipe")

    def test_root_and_directory_metadata_are_preserved(self):
        directory = self.root / "directory"; directory.mkdir()
        before = self.reader.whole_archive()
        os.utime(directory, ns=(directory.stat().st_atime_ns, directory.stat().st_mtime_ns + 100))
        self.assertNotEqual(before["inventory_cid"], self.reader.whole_archive()["inventory_cid"])


class IgnoredCaptureControls(unittest.TestCase):
    def setUp(self):
        raw = b"def ignored(n):\n    return n + 1\n"
        self.raw = raw
        artifacts = {}
        self.previous = manifest({"ignored.py": raw, "normal.py": b"normal = 1\n"}, artifacts, untracked={"ignored.py"})
        self.current = manifest({"normal.py": b"normal = 1\n"}, artifacts)
        old_receipt, old_head = receipt(self.previous, None, 1)
        new_receipt, new_head = receipt(self.current, old_head, 2)
        left, right = sides(self.previous), sides(self.current)
        ignored, normal = "raw:" + b"ignored.py".hex(), "raw:" + b"normal.py".hex()
        ledger = [{"source_key": ignored, "classification": "removed", "previous": left[ignored], "current": None,
            "source_bytes_comparison": "unavailable", "ast_identity_comparison": "unavailable"},
            {"source_key": normal, "classification": "retained", "previous": left[normal], "current": right[normal],
            "source_bytes_comparison": "equal", "ast_identity_comparison": "different"}]
        value = deepcopy(fixture()[0]["value"])
        value.update(previous_head=old_head, current_head=new_head, previous_publication_receipt=old_receipt,
            current_publication_receipt=new_receipt, previous_membership_cid=audit.structured([row["member"] for row in left.values()]),
            current_membership_cid=audit.structured([row["member"] for row in right.values()]), ledger=ledger,
            coverage={"previous_entries": 2, "current_entries": 1, "union_entries": 2,
                "classifications": {"retained": 1, "changed": 0, "added": 0, "removed": 1},
                "source_bytes_comparisons": {"equal": 1, "different": 0, "unavailable": 1},
                "ast_identity_comparisons": {"equal": 0, "different": 1, "unavailable": 1}})
        self.envelope = {"artifact_cid": audit.structured(value), "value": value}
        physical = {"path": "ignored.py", "bytes": len(raw), "sha256": audit.sha(raw), "source_cid": audit.cid(raw, source=True), "regular": True}
        self.result = {"schema": "codebase-source-delta-ignore-native-qualification@1", "qualified": True,
            "scope": "two_member_ignored_but_physically_present_capture_removal", "pid": 123,
            "optimized_cid": self.envelope["artifact_cid"], "previous_head": old_head, "current_head": new_head,
            "coverage": value["coverage"], "physical_presence_before": physical, "physical_presence_after": deepcopy(physical),
            "physical_absence_verified": False, "numerical_reuse": False, "model_advanced": False,
            "repository_code_executed": False, "worker_launched": False, "proof_authority": False, "admission_authority": False,
            "removal_scope": "absent_from_current_complete_capture", "authority": value["authority"],
            "model_registry_open_attempts": 0, "training_attempts": 0, "inference_attempts": 0, "new_fitting_epochs": 0,
            "source_owner_reopens": 0, "receiving_verified": True, "physical_bytes_unchanged": True,
            "source_connection_closed": True, "current_owner_counts": {"source": 0, "model": 0}}
        audit.verify_source_delta(self.envelope, self.previous, self.current, old_receipt, new_receipt,
                                 get_artifact=lambda wanted, source=False: artifacts[wanted])

    def verify(self):
        return audit.verify_ignored_present_observation(self.result, self.envelope, self.previous, self.current, raw_physical=self.raw)

    def test_removed_capture_with_present_physical_file_positive(self):
        result = self.verify()
        self.assertTrue(result["removed_from_current_complete_capture"])
        self.assertFalse(result["physical_absence_verified"])

    def test_changed_physical_bytes_refused(self):
        self.raw += b"changed"
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_bool_physical_size_alias_refused(self):
        self.result["physical_presence_after"]["bytes"] = True
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_missing_native_receiving_refused(self):
        self.result["receiving_verified"] = False
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_physical_absence_integer_alias_refused(self):
        self.result["physical_absence_verified"] = 0
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_nonzero_model_owners_refused(self):
        self.result["current_owner_counts"]["model"] = 1
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_inference_counter_bool_alias_refused(self):
        self.result["inference_attempts"] = False
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_ignored_source_still_captured_refused(self):
        self.current["snapshot"]["entries"].append(self.previous["snapshot"]["entries"][0])
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_byte_equal_changed_entry_not_retained(self):
        self.envelope["value"]["ledger"][1]["classification"] = "changed"
        self.envelope["value"]["coverage"]["classifications"] = {"retained": 0, "changed": 1, "added": 0, "removed": 1}
        self.result["coverage"] = self.envelope["value"]["coverage"]
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()

    def test_removed_source_equality_claim_refused(self):
        self.envelope["value"]["ledger"][0]["source_bytes_comparison"] = "equal"
        with self.assertRaises(audit.SourceDeltaAuditError): self.verify()


if __name__ == "__main__":
    unittest.main(verbosity=2)
