"""Bounded stdlib controls for closed successor dispatch transport.

Detached, deliberately non-database fixtures exercise public staging and
materialization guards. They attest no scan, numerical inference, signature,
or native owner execution. The separately requested closed-300 receiving
qualification uses actual frozen bytes and is recorded separately.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import os
from pathlib import Path
import resource
import stat
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from . import source_successor_dispatch_fixture as fixture

BASE = Path("/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench")
OLD_SOURCE = BASE / "source-successor-qualification-20261003-02"
REQUIRED = ("private/source.duckdb", "private/model.duckdb",
    "closed-full-scan/owners-after-cold.json",
    "closed-full-scan/copied-source-catalog-relocation.json",
    "closed-full-scan/successor-scan-completion.json",
    "closed-full-scan/successor-selection.json", "closed-full-scan/scan-root.json")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fixture.inert.numerical_wire(value) + b"\n")


class DetachedTransportControls(unittest.TestCase):
    def test_native_duckdb_row_tuples_normalize_to_plain_json(self):
        self.assertEqual(fixture._native_json([("row", 1, None, [2])]), [["row", 1, None, [2]]])

    def test_native_duckdb_row_nonfinite_scalar_refused(self):
        with self.assertRaises((ValueError, TypeError)):
            fixture._native_json([["row", float("nan")]])

    def test_native_duckdb_row_foreign_object_refused(self):
        with self.assertRaises((ValueError, TypeError)):
            fixture._native_json([[object()]])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="successor-dispatch-controls-")
        self.root = Path(self.temporary.name).resolve()
        self.seed = self.root / "seed"; self.seed.mkdir()
        self.output = self.root / "output"
        historical = fixture.inert.Reader(OLD_SOURCE, seconds=120).json("result.json")
        members = []
        for name in REQUIRED:
            path = self.seed / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("detached transport control only: " + name + "\n").encode())
            path.chmod(0o600 if name.endswith(".duckdb") else 0o444)
            members.append({"path": name,
                "source_path": name.removeprefix("closed-full-scan/"),
                **fixture._pin(path.read_bytes()), "mode": stat.S_IMODE(path.stat().st_mode)})
        for name in ("independent-full-scan-audit.json", "independent-full-scan-reader-controls.json"):
            put(self.seed / name, {"detached_control_fixture": name})
            (self.seed / name).chmod(0o444)
        cid = fixture.inert.structured({"detached_control_fixture": True})
        self.receipt = {"schema": fixture.SCHEMA, "qualified": True,
            "source_namespace": str(self.root / "detached-source"), "staged_destination": str(self.seed),
            "source_archive_inventory_cid": cid,
            "audit": fixture._pin((self.seed / "independent-full-scan-audit.json").read_bytes()),
            "reader_controls": fixture._pin((self.seed / "independent-full-scan-reader-controls.json").read_bytes()),
            "reader": fixture._pin(Path(fixture.__file__).with_name("audit_codebase_full_successor.py").read_bytes()),
            "native_result": fixture._pin(b"detached control result"),
            "reader_control_scope": "detached structural controls; no native execution attestation",
            "reader_control_source_namespace": str(OLD_SOURCE),
            "copied_members": sorted(members, key=lambda row: row["path"]),
            "copied_files": len(members), "copied_bytes": sum(row["bytes"] for row in members),
            "selected_producers": [], "current_head": deepcopy(historical["current_head"]),
            "previous_head": deepcopy(historical["previous_head"]),
            "root_cid": cid, "completion_cid": cid, "selection_cid": cid, "source_delta_cid": cid,
            "selected_version_id": historical["child_version_id"], "previous_version_id": historical["parent_version_id"],
            "checkpoint_states": deepcopy(historical["checkpoint_states"]),
            "inherited_setup_epochs": 2, "inherited_scan_pages": 10, "inherited_reference_pages": 1,
            "new_fitting_epochs": 0, "new_scan_pages": 0, "native_owners_opened": False,
            "fresh_native_receiving_required": True, "proof_authority": False,
            "source_execution_attested": False, "scan_execution_attested": False}
        self.sync()

    def tearDown(self):
        self.temporary.cleanup()

    def sync(self):
        put(self.seed / "staged-successor-dispatch.json", self.receipt)

    def materialize(self, receipt=None, output=None):
        return fixture.materialize_staged_successor_dispatch(self.seed, output or self.output,
            self.receipt if receipt is None else receipt)

    def reject_manifest(self, change):
        change(self.receipt); self.sync()
        before = fixture.inert.Reader(self.seed).whole_archive()
        with self.assertRaises((ValueError, KeyError, TypeError)):
            self.materialize()
        self.assertTrue(fixture.inert.same(before, fixture.inert.Reader(self.seed).whole_archive()))

    def test_detached_positive_independent_bytes_modes_and_source_preservation(self):
        before = fixture.inert.Reader(self.seed).whole_archive()
        result = self.materialize()
        self.assertTrue(result["qualified"])
        self.assertFalse(result["native_owners_opened"])
        self.assertFalse(result["proof_authority"])
        self.assertEqual(result["new_fitting_epochs"], 0)
        self.assertEqual(result["new_scan_pages"], 0)
        self.assertEqual(stat.S_IMODE((self.output / "private").stat().st_mode), 0o700)
        for row in self.receipt["copied_members"]:
            original, copied = self.seed / row["path"], self.output / row["path"]
            self.assertEqual(original.read_bytes(), copied.read_bytes())
            self.assertEqual(stat.S_IMODE(original.stat().st_mode), stat.S_IMODE(copied.stat().st_mode))
            self.assertNotEqual((original.stat().st_dev, original.stat().st_ino),
                (copied.stat().st_dev, copied.stat().st_ino))
        (self.output / "private/source.duckdb").write_bytes(b"new-copy-only mutation")
        self.assertTrue(fixture.inert.same(before, fixture.inert.Reader(self.seed).whole_archive()))
        self.assertNotIn("duckdb", sys.modules)
        self.assertFalse(any(name.startswith("ipfs_datasets_py") for name in sys.modules))

    def test_manifest_disagreement_with_saved_receipt_refused(self):
        changed = deepcopy(self.receipt); changed["native_result"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError): self.materialize(changed)

    def test_staged_member_byte_tamper_refused_and_preserved(self):
        path = self.seed / "private/source.duckdb"; path.write_bytes(path.read_bytes() + b"tamper")
        before = fixture.inert.Reader(self.seed).whole_archive()
        with self.assertRaises(ValueError): self.materialize()
        self.assertTrue(fixture.inert.same(before, fixture.inert.Reader(self.seed).whole_archive()))

    def test_staged_member_mode_tamper_refused(self):
        (self.seed / "private/model.duckdb").chmod(0o640)
        with self.assertRaises(ValueError): self.materialize()

    def test_staged_audit_pin_tamper_refused(self):
        path = self.seed / "independent-full-scan-audit.json"; path.chmod(0o600); path.write_bytes(b"{}")
        with self.assertRaises(ValueError): self.materialize()

    def test_staged_controls_pin_tamper_refused(self):
        path = self.seed / "independent-full-scan-reader-controls.json"; path.chmod(0o600); path.write_bytes(b"{}")
        with self.assertRaises(ValueError): self.materialize()

    def test_staged_missing_member_refused(self):
        (self.seed / "closed-full-scan/scan-root.json").unlink()
        with self.assertRaises(ValueError): self.materialize()

    def test_staged_extra_file_refused(self):
        (self.seed / "unexpected-checkpoint.json").write_bytes(b"{}")
        with self.assertRaises(ValueError): self.materialize()

    def test_staged_owner_lock_refused(self):
        (self.seed / "private/source.duckdb.owner.lock").write_bytes(b"lock")
        with self.assertRaises(ValueError): self.materialize()

    def test_staged_wal_refused(self):
        (self.seed / "private/source.duckdb.wal").write_bytes(b"wal")
        with self.assertRaises(ValueError): self.materialize()

    def test_staged_extra_directory_refused(self):
        (self.seed / "unreviewed-directory").mkdir()
        with self.assertRaises(ValueError): self.materialize()

    def test_staged_symlink_member_refused(self):
        path = self.seed / "private/source.duckdb"; path.unlink()
        path.symlink_to(self.seed / "private/model.duckdb")
        with self.assertRaises(ValueError): self.materialize()

    def test_staged_hardlink_member_refused(self):
        path = self.seed / "private/source.duckdb"; path.unlink()
        os.link(self.seed / "private/model.duckdb", path)
        with self.assertRaises(ValueError): self.materialize()

    def test_seed_root_alias_refused(self):
        alias = self.root / "seed-alias"; alias.symlink_to(self.seed, target_is_directory=True)
        with self.assertRaises(ValueError):
            fixture.materialize_staged_successor_dispatch(alias, self.output, self.receipt)

    def test_output_root_alias_refused(self):
        self.output.mkdir()
        alias = self.root / "output-alias"; alias.symlink_to(self.output, target_is_directory=True)
        with self.assertRaises(ValueError): self.materialize(output=alias)

    def test_output_nested_under_seed_refused_before_copy(self):
        nested = self.seed / "receiving"
        before = fixture.inert.Reader(self.seed).whole_archive()
        with self.assertRaises(ValueError): self.materialize(output=nested)
        self.assertTrue(fixture.inert.same(before, fixture.inert.Reader(self.seed).whole_archive()))

    def test_output_nonempty_refused_before_copy(self):
        self.output.mkdir()
        existing = self.output / "existing"; existing.write_bytes(b"retain")
        with self.assertRaises(ValueError): self.materialize()
        self.assertEqual(list(self.output.iterdir()), [existing])

    def test_materialized_receipt_authority_tamper_rejected_before_owner_import(self):
        self.materialize()
        path = self.output / "materialized-successor-dispatch.json"
        value = fixture.inert.parse(path.read_bytes()); value["proof_authority"] = True; put(path, value)
        with patch.object(fixture, "_relocate", side_effect=AssertionError("invalid receipt reached native relocation")):
            with self.assertRaises(ValueError): fixture.open_materialized_successor_dispatch(self.output)
        self.assertNotIn("duckdb", sys.modules)

    def test_materialized_receipt_extra_field_rejected_before_owner_import(self):
        self.materialize()
        path = self.output / "materialized-successor-dispatch.json"
        value = fixture.inert.parse(path.read_bytes()); value["unreviewed"] = True; put(path, value)
        with patch.object(fixture, "_relocate", side_effect=AssertionError("invalid receipt reached native relocation")):
            with self.assertRaises(ValueError): fixture.open_materialized_successor_dispatch(self.output)
        self.assertNotIn("duckdb", sys.modules)


def _mutation_test(change):
    def test(self): self.reject_manifest(change)
    return test


MUTATIONS = {
    "manifest_extra_field": lambda r: r.__setitem__("unreviewed", True),
    "manifest_missing_field": lambda r: r.pop("source_delta_cid"),
    "unqualified_manifest": lambda r: r.__setitem__("qualified", 1),
    "new_fit_boolean_alias": lambda r: r.__setitem__("new_fitting_epochs", False),
    "new_scan_boolean_alias": lambda r: r.__setitem__("new_scan_pages", False),
    "inherited_page_boolean_alias": lambda r: r.__setitem__("inherited_reference_pages", True),
    "copied_count_boolean_alias": lambda r: r.__setitem__("copied_files", True),
    "copied_bytes_boolean_alias": lambda r: r.__setitem__("copied_bytes", True),
    "copied_member_byte_boolean_alias": lambda r: r["copied_members"][0].__setitem__("bytes", True),
    "copied_member_mode_boolean_alias": lambda r: r["copied_members"][0].__setitem__("mode", True),
    "copied_member_traversal": lambda r: r["copied_members"][0].__setitem__("path", "closed-full-scan/../escape"),
    "copied_member_absolute": lambda r: r["copied_members"][0].__setitem__("path", "/escape"),
    "copied_member_backslash": lambda r: r["copied_members"][0].__setitem__("path", "closed-full-scan/escape\\member"),
    "copied_member_foreign_path": lambda r: r["copied_members"][0].__setitem__("path", "foreign/member"),
    "copied_member_wal": lambda r: r["copied_members"][0].__setitem__("path", "private/source-artifacts/foreign.wal"),
    "copied_member_uppercase_sha": lambda r: r["copied_members"][0].__setitem__("sha256", "A" * 64),
    "copied_member_extra_field": lambda r: r["copied_members"][0].__setitem__("foreign", True),
    "copied_member_source_mapping": lambda r: r["copied_members"][-1].__setitem__("source_path", "unrelated-source.duckdb"),
    "copied_member_duplicate": lambda r: r["copied_members"].append(deepcopy(r["copied_members"][-1])),
    "copied_member_missing_required": lambda r: r["copied_members"].pop(),
    "proof_authority": lambda r: r.__setitem__("proof_authority", True),
    "source_execution_attestation": lambda r: r.__setitem__("source_execution_attested", True),
    "scan_execution_attestation": lambda r: r.__setitem__("scan_execution_attested", True),
    "native_owner_open_claim": lambda r: r.__setitem__("native_owners_opened", True),
    "no_fresh_receiving": lambda r: r.__setitem__("fresh_native_receiving_required", False),
    "staged_path_binding": lambda r: r.__setitem__("staged_destination", "/different/stage"),
    "audit_pin_boolean_size": lambda r: r["audit"].__setitem__("bytes", True),
    "reader_pin_extra_field": lambda r: r["reader"].__setitem__("foreign", True),
    "reader_control_role_namespace": lambda r: r.__setitem__("reader_control_source_namespace", r["source_namespace"]),
    "source_inventory_cid_profile": lambda r: r.__setitem__("source_archive_inventory_cid", "invalid"),
    "current_head_boolean_generation": lambda r: r["current_head"].__setitem__("generation", True),
    "current_head_foreign_repository": lambda r: r["current_head"].__setitem__("repository_id", "other"),
    "head_sequence_reversed": lambda r: r.__setitem__("previous_head", deepcopy(r["current_head"])),
}
for _name, _change in MUTATIONS.items():
    setattr(DetachedTransportControls, "test_guard_" + _name, _mutation_test(_change))


class DetachedStagingControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="successor-dispatch-stage-controls-")
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "detached-source"; self.source.mkdir()
        self.destination = self.root / "new-stage"
        historical = fixture.inert.Reader(OLD_SOURCE, seconds=120).json("result.json")
        self.result = {"qualified": True, "complete_scan_qualified": True,
            "new_fitting_epochs": 0, "inherited_setup_epochs": 2, "inherited_scan_pages": 0,
            "new_default_scan_pages": 10, "new_reference_scan_pages": 1, "fresh_process_count": 4,
            "current_head": deepcopy(historical["current_head"]), "previous_head": deepcopy(historical["previous_head"]),
            "coverage": {"inventory_entries": 300, "pages": 10},
            "root_cid": fixture.inert.structured({"detached_root": True}),
            "completion_cid": fixture.inert.structured({"detached_completion": True}),
            "successor_selection_cid": fixture.inert.structured({"detached_selection": True}),
            "source_delta_cid": historical["source_delta_cid"],
            "child_version_id": historical["child_version_id"], "parent_version_id": historical["parent_version_id"],
            "checkpoint_states": deepcopy(historical["checkpoint_states"])}
        for name in REQUIRED:
            path = self.source / name.removeprefix("closed-full-scan/")
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"detached control bytes\n")
            path.chmod(0o600 if name.endswith(".duckdb") else 0o444)
        put(self.source / "materialized-successor-setup.json", {"source_namespace": str(OLD_SOURCE)})
        producer = Path(fixture.__file__).with_name("qualify_codebase_full_successor.py").resolve()
        producer_raw = producer.read_bytes()
        copy = self.source / "producers/00-main.py"; copy.parent.mkdir(); copy.write_bytes(producer_raw)
        put(self.source / "generation-inputs.json", {"schema": "codebase-full-successor-selected-producers@1",
            "scope": "listed_local_files_only", "files": [{"name": "__main__", "path": str(producer),
                "copy": "producers/00-main.py", **fixture._pin(producer_raw)}]})
        reader = Path(fixture.__file__).with_name("audit_codebase_full_successor.py").resolve()
        self.reader_pin = {"path": str(reader), **fixture._pin(reader.read_bytes())}
        self.controls = {"schema": "full-successor-independent-reader-guard-tests@1", "qualified": True,
            "tests": 1, "failures": 0, "errors": 0, "skipped": 0, "reader": deepcopy(self.reader_pin),
            "current_pins_unchanged": True, "source_namespace": str(OLD_SOURCE), "native_owners_opened": False,
            "sql_executed": False, "git_executed": False, "new_fitting_epochs": 0, "new_inference_pages": 0,
            "scope": "detached structural control receipt; no native execution attestation"}
        self.audit = {"schema": "codebase-full-successor-independent-audit@1", "qualified": True,
            "preserved": True, "errors": [], "complete_scan_qualified": True, "namespace": str(self.source),
            "reader": deepcopy(self.reader_pin), "materialization": {"source_namespace": str(OLD_SOURCE)}}
        self.audit_path = self.root / "detached-audit.json"; self.guard_path = self.root / "detached-guards.json"
        self.sync()

    def tearDown(self):
        self.temporary.cleanup()

    def sync(self):
        put(self.source / "result.json", self.result)
        self.audit["audited_result"] = fixture._pin((self.source / "result.json").read_bytes())
        self.audit["archive"] = fixture.inert.Reader(self.source).whole_archive()
        put(self.audit_path, self.audit); put(self.guard_path, self.controls)

    def reject(self, change):
        change(self); self.sync()
        before = fixture.inert.Reader(self.source).whole_archive()
        with self.assertRaises((ValueError, KeyError, TypeError)):
            fixture.stage_closed_successor_dispatch(self.source, self.destination,
                audit_path=self.audit_path, guard_receipt=self.guard_path)
        self.assertFalse(self.destination.exists(), "invalid inputs reached staging writes")
        self.assertTrue(fixture.inert.same(before, fixture.inert.Reader(self.source).whole_archive()))

    def test_staging_destination_overlap_refused_before_copy(self):
        with self.assertRaises(ValueError):
            fixture.stage_closed_successor_dispatch(self.source, self.source / "nested",
                audit_path=self.audit_path, guard_receipt=self.guard_path)
        self.assertFalse((self.source / "nested").exists())

    def test_staging_receipt_alias_refused(self):
        alias = self.root / "guard-alias"; alias.symlink_to(self.guard_path)
        with self.assertRaises(ValueError):
            fixture.stage_closed_successor_dispatch(self.source, self.destination,
                audit_path=self.audit_path, guard_receipt=alias)
        self.assertFalse(self.destination.exists())


STAGING_MUTATIONS = {
    "guard_schema": lambda s: s.controls.__setitem__("schema", "foreign-reader-tests@1"),
    "guard_source_role_namespace": lambda s: s.controls.__setitem__("source_namespace", str(s.source)),
    "guard_current_pins_changed": lambda s: s.controls.__setitem__("current_pins_unchanged", False),
    "guard_native_owners_claim": lambda s: s.controls.__setitem__("native_owners_opened", True),
    "guard_sql_claim": lambda s: s.controls.__setitem__("sql_executed", True),
    "guard_git_claim": lambda s: s.controls.__setitem__("git_executed", True),
    "guard_new_fit": lambda s: s.controls.__setitem__("new_fitting_epochs", 1),
    "guard_new_fit_boolean_alias": lambda s: s.controls.__setitem__("new_fitting_epochs", False),
    "guard_new_inference": lambda s: s.controls.__setitem__("new_inference_pages", 1),
    "guard_new_inference_boolean_alias": lambda s: s.controls.__setitem__("new_inference_pages", False),
    "guard_failed_case": lambda s: s.controls.__setitem__("failures", 1),
    "guard_failure_boolean_alias": lambda s: s.controls.__setitem__("failures", False),
    "guard_skipped_case": lambda s: s.controls.__setitem__("skipped", 1),
    "guard_test_boolean_alias": lambda s: s.controls.__setitem__("tests", True),
    "guard_reader_generation": lambda s: s.controls["reader"].__setitem__("sha256", "0" * 64),
    "audit_incomplete_scan": lambda s: s.audit.__setitem__("complete_scan_qualified", False),
    "audit_namespace": lambda s: s.audit.__setitem__("namespace", str(OLD_SOURCE)),
    "audit_error": lambda s: s.audit.__setitem__("errors", ["failed"]),
    "native_new_fit_boolean_alias": lambda s: s.result.__setitem__("new_fitting_epochs", False),
    "native_inherited_reference_boolean_alias": lambda s: s.result.__setitem__("new_reference_scan_pages", True),
    "native_incomplete_inventory": lambda s: s.result["coverage"].__setitem__("inventory_entries", 299),
    "native_incomplete_pages": lambda s: s.result["coverage"].__setitem__("pages", 9),
}
for _name, _change in STAGING_MUTATIONS.items():
    def _stage_test(self, change=_change): self.reject(change)
    setattr(DetachedStagingControls, "test_guard_" + _name, _stage_test)


def actual_closed_receiving(args):
    """Qualify one actual fresh receiving copy; never open closed owners."""
    output = args.receipt_directory.absolute(); output.mkdir(parents=True, exist_ok=False)
    source = fixture.transport.canonical_path(args.native_source)
    source_before = fixture.inert.Reader(source, seconds=120).whole_archive()
    old_before = fixture.inert.Reader(OLD_SOURCE, seconds=120).whole_archive()
    paths = [Path(__file__).resolve(), Path(fixture.__file__).resolve(),
        Path(fixture.transport.__file__).resolve(), Path(fixture.inert.__file__).resolve(),
        Path(fixture.__file__).with_name("audit_codebase_full_successor.py").resolve()]
    body = {str(path): path.read_bytes() for path in paths}
    for number, path in enumerate(paths): (output / (str(number) + "-" + path.name)).write_bytes(body[str(path)])
    started = time.monotonic()
    report = {"schema": "source-successor-dispatch-actual-receiving@1", "qualified": False,
        "source_namespace": str(source), "old_source_namespace": str(OLD_SOURCE),
        "source_archive_before": source_before, "old_archive_before": old_before,
        "post_setup_fit_attempt_count": 0, "inference_attempts": 0, "new_fitting_epochs": 0,
        "new_scan_pages": 0, "owner_pairs_opened": 0, "native_owners_in_old_archives_opened": False,
        "git_in_old_archives_executed": False, "proof_authority": False, "signed_admission_qualified": False,
        "source_execution_attested": False, "scan_execution_attested": False,
        "pid": os.getpid(), "cpu_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "native_parent_memory_mb": args.native_memory_mb, "nested_current_source_memory_mb": 1024,
        "scope": "actual closed full300 staging, second independent copy, two local DB relocations and one native owner pair/current source/checkpoint receiving; no page execution or signatures",
        "source_pins": [{"path": str(path), **fixture._pin(body[str(path)])} for path in paths], "error": None}
    scheduler = lease = index = registry = connection = staged_before = None
    try:
        host_path = fixture.transport.canonical_path(args.host_configuration)
        host_raw = fixture._read(host_path); host = fixture.inert.parse(host_raw)
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
            GlobalResourceScheduler, ResourceSchedulerConfig, ResourceLane)
        scheduler = GlobalResourceScheduler(ResourceSchedulerConfig(**host["persisted_config"],
            state_path=host["state_path"], lease_ttl_seconds=host["lease_ttl_seconds"],
            auto_renew_leases=host["auto_renew_leases"]))
        report["host_configuration"] = {"path": str(host_path), **fixture._pin(host_raw)}
        report["resources_before"] = scheduler.snapshot()
        lease = scheduler.acquire(ResourceLane.SNAPSHOT_EVALUATION, cpu_slots=1, memory_mb=args.native_memory_mb,
            child_process_slots=1, timeout=30, request_id="actual-closed-successor-dispatch-receiving")
        report["admission"] = {"state_path": str(scheduler.state_path), "lease_id": lease.lease_id,
            "cpu_slots": lease.cpu_slots, "memory_mb": lease.memory_mb,
            "child_process_slots": lease.child_process_slots, "wait_seconds": lease.wait_seconds}
        fixture.need(not lease.cancelled and lease.renew(), "actual receiving lease cancelled or expired")
        print("ADMITTED", lease.lease_id, flush=True)
        seed, receiver = output / "stage", output / "receiver"
        staged = fixture.stage_closed_successor_dispatch(source, seed,
            audit_path=args.native_audit, guard_receipt=args.native_reader_controls)
        staged_before = fixture.inert.Reader(seed, seconds=120).whole_archive()
        materialized = fixture.materialize_staged_successor_dispatch(seed, receiver, staged)
        report["staged"] = staged; report["materialized"] = materialized
        from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor as delta
        from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as scan
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
        from . import qualify_codebase_inventory_resume as native
        def forbid_inference(*unused, **also_unused):
            report["inference_attempts"] += 1
            raise AssertionError("actual copied-owner receiving attempted numerical inference")
        with fixture.transport.no_fit(report), patch.object(scan, "_worker", forbid_inference), \
                patch.object(features, "infer_projection_features", forbid_inference):
            index, registry, connection = fixture.open_materialized_successor_dispatch(receiver)
            report["owner_pairs_opened"] += 1
            fixture.need(not lease.cancelled and lease.renew(), "actual receiving lease cancelled or expired")
            owners = fixture.transport.owners(index, registry, connection)
            expected = fixture.inert.parse(fixture._read(receiver / "closed-full-scan/owners-after-cold.json"))
            expected["registry"]["meta"][0][4] = str(receiver / "private/model-artifacts")
            expected["registry"]["meta"][0][5] += 1
            fixture.need(fixture.inert.observation_same(owners, expected),
                "fresh native owners changed beyond two local paths and one registry open generation")
            fixture.need(fixture.inert.same(owners["source"]["current_head"], staged["current_head"]),
                "fresh native current source head differs")
            states = {label: native._state(registry, staged[field], include_identity=True)
                for label, field in (("root", "previous_version_id"), ("child", "selected_version_id"))}
            fixture.need(fixture.inert.same(states, staged["checkpoint_states"]),
                "fresh native checkpoint/Adam population differs")
            export = fixture.inert.parse(fixture._read(receiver / "closed-full-scan/source-delta.json"))
            transition = delta.load_codebase_source_delta(index.artifacts, export["artifact_cid"])
            fixture.need(delta.validate_current_codebase_source_delta(transition, index, receiver / "repository",
                parent_lease=lease, timeout_seconds=180, memory_mb=1024) is transition,
                "new copied current source receiving differs")
            fixture.need(fixture.inert.observation_same(fixture.transport.owners(index, registry, connection), owners),
                "current receiving mutated native owner rows")
            after_states = {label: native._state(registry, staged[field], include_identity=True)
                for label, field in (("root", "previous_version_id"), ("child", "selected_version_id"))}
            fixture.need(fixture.inert.same(after_states, states), "current receiving mutated saved checkpoint/Adam")
            relocations = fixture.inert.parse(fixture._read(receiver / "copied-successor-dispatch-store-relocations.json"))
            fixture.need(len(relocations["relocations"]) == 2
                and {row["owner"] for row in relocations["relocations"]} == {"source", "registry"}
                and all(row["only_local_path_changed"] is True and row["owner_generation_advanced"] is False
                    and row["fitting_performed"] is False and row["native_publication_performed"] is False
                    and row["old_native_owners_opened"] is False for row in relocations["relocations"]),
                "two exact local copied-store relocations required")
            report.update(native_owners_before=owners, native_owners_after=fixture.transport.owners(index, registry, connection),
                checkpoint_states_before=states, checkpoint_states_after=after_states,
                actual_relocations=relocations, current_source_receiving_verified=True,
                registry_owner_generation_increment_from_native_open=1,
                current_source_receiving_git_scope="new receiver repository only")
            imported = []
            for row in staged["selected_producers"]:
                module = sys.modules.get(row["name"])
                current = fixture._read(row["path"], 4 * fixture.inert.MIB)
                fixture.need(fixture._pin(current) == {key: row[key] for key in ("bytes", "sha256")},
                    "actual selected receiving producer changed: " + row["name"])
                if module is not None:
                    fixture.need(str(Path(module.__file__).resolve()) == row["path"],
                        "actual imported receiving producer path differs: " + row["name"])
                    imported.append(row)
            fixture.need({"ipfs_datasets_py.duckdb_control.autoencoder_registry",
                "ipfs_datasets_py.duckdb_control.codebase_catalog",
                "ipfs_datasets_py.logic.software_contracts.codebase_ir",
                "ipfs_datasets_py.logic.software_contracts.codebase_inventory_successor",
                "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_projection_features",
                "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_runtime_registry",
                "ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler"}
                <= {row["name"] for row in imported}, "actual selected native owner/runtime bodies required")
            report["actual_imported_selected_producers"] = imported
        fixture.need(report["post_setup_fit_attempt_count"] == report["inference_attempts"] == 0,
            "fit/inference-free actual receiving required")
        report["staged_archive_preserved"] = fixture.inert.same(staged_before,
            fixture.inert.Reader(seed, seconds=120).whole_archive())
        fixture.need(report["staged_archive_preserved"], "frozen staged seed changed during native receiving")
        report["actual_native_receiving_complete"] = True
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        fixture.transport.close_materialized_successor(registry, connection)
        if lease is not None: lease.release(); report["own_lease_released"] = True
        if scheduler is not None: report["resources_after"] = fixture.transport.assert_clean(scheduler)
        if staged_before is not None:
            report["staged_archive_preserved"] = fixture.inert.same(staged_before,
                fixture.inert.Reader(output / "stage", seconds=120).whole_archive())
        report["closed_source_archive_preserved"] = fixture.inert.same(source_before,
            fixture.inert.Reader(source, seconds=120).whole_archive())
        report["old_closed_archive_preserved"] = fixture.inert.same(old_before,
            fixture.inert.Reader(OLD_SOURCE, seconds=120).whole_archive())
        report["current_pins_unchanged"] = all(path.read_bytes() == body[str(path)] for path in paths)
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["rss_reservation_respected"] = report["peak_rss_bytes"] <= args.native_memory_mb * fixture.inert.MIB
        report["kernel_resource_enforcement"] = False
        if not report["rss_reservation_respected"] and report["error"] is None:
            report["error"] = {"type": "ReservationMemoryExceeded",
                "message": "actual native process RSS exceeded its shared memory reservation"}
        report["recorded_seconds"] = time.monotonic() - started
        report["qualified"] = report.get("actual_native_receiving_complete") is True and report["error"] is None \
            and report["closed_source_archive_preserved"] and report["old_closed_archive_preserved"] \
            and report["current_pins_unchanged"] and report.get("own_lease_released") is True \
            and report["rss_reservation_respected"] and report.get("staged_archive_preserved") is True
        fixture._write(output / "receipt.json", report)
        for path in output.iterdir():
            if path.is_file(): path.chmod(0o444)
        print({key: report[key] for key in ("qualified", "recorded_seconds", "owner_pairs_opened",
            "post_setup_fit_attempt_count", "inference_attempts", "closed_source_archive_preserved",
            "old_closed_archive_preserved", "error")}, flush=True)
    return 0 if report["qualified"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-directory", type=Path, required=True)
    parser.add_argument("--native-source", type=Path)
    parser.add_argument("--native-audit", type=Path)
    parser.add_argument("--native-reader-controls", type=Path)
    parser.add_argument("--host-configuration", type=Path)
    parser.add_argument("--native-memory-mb", type=int, choices=(1024, 2048), default=2048)
    args = parser.parse_args()
    if args.native_source is not None:
        if any(value is None for value in (args.native_audit, args.native_reader_controls, args.host_configuration)):
            parser.error("actual native mode requires audit, reader controls and shared host configuration")
        return actual_closed_receiving(args)
    output = args.receipt_directory.absolute(); output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__).resolve(), Path(fixture.__file__).resolve(), Path(fixture.transport.__file__).resolve(),
        Path(fixture.inert.__file__).resolve(),
        Path(fixture.__file__).with_name("audit_codebase_full_successor.py").resolve(),
        Path(fixture.__file__).with_name("qualify_codebase_full_successor.py").resolve()]
    before = {str(path): path.read_bytes() for path in paths}
    primary = fixture.inert.Reader(OLD_SOURCE, seconds=120).whole_archive()
    log = io.StringIO(); started = time.monotonic()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    preserved = fixture.inert.same(primary, fixture.inert.Reader(OLD_SOURCE, seconds=120).whole_archive())
    unchanged = all(path.read_bytes() == before[str(path)] for path in paths)
    receipt = {"schema": "source-successor-dispatch-stdlib-controls@1",
        "qualified": result.wasSuccessful() and not result.skipped and preserved and unchanged,
        "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "recorded_seconds": time.monotonic() - started,
        "source_pins": [{"path": str(path), **fixture._pin(before[str(path)])} for path in paths],
        "source_namespace": str(OLD_SOURCE), "source_archive_before": primary,
        "source_archive_preserved": preserved, "current_pins_unchanged": unchanged,
        "native_owners_opened": False, "sql_executed": False, "git_executed": False,
        "new_fitting_epochs": 0, "new_inference_pages": 0,
        "scope": "detached non-database transport controls; no full300 scan, signed admission or native receiving qualification"}
    for number, path in enumerate(paths):
        (output / (str(number) + "-" + path.name)).write_bytes(before[str(path)])
    (output / "tests.log").write_text(log.getvalue())
    (output / "receipt.json").write_bytes(fixture.inert.numerical_wire(receipt) + b"\n")
    for path in output.iterdir(): path.chmod(0o444)
    print({key: receipt[key] for key in ("qualified", "tests", "failures", "errors", "skipped", "recorded_seconds", "source_archive_preserved", "current_pins_unchanged")})
    return 0 if receipt["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
