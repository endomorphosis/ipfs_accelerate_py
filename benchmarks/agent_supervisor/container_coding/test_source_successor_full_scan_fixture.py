"""Stdlib controls for successor transport and fresh-process input binding.

Native owner execution is qualified separately. These tests touch only inert
closed seed bytes and temporary copies, and never connect a database or run
old Git, a scan worker, a child interpreter, or model fitting. One explicitly
marked receiving test opens only newly materialized owners and samples the
current source in the newly copied repository through the shared scheduler.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from . import source_successor_full_scan_fixture as fixture

BASE = Path("/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench")
SOURCE = Path(fixture.SOURCE_NAMESPACE)
AUDIT_PATH = BASE / "source-successor-qualification-20261003-02-readonly-audit-20261003-01.json"
GUARD = BASE / "source-successor-reader-guards-20261003-01/receipt.json"
NATIVE_RECEIVING = []


class TransportControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="source-successor-transport-controls-")
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def test_fixed_closed_seed_transport_positive_without_native_imports(self):
        destination = self.root / "new-copy"
        before = fixture.inert.Reader(SOURCE, seconds=120).whole_archive()
        result = fixture.stage_closed_successor_setup(SOURCE, AUDIT_PATH, GUARD, destination)
        self.assertTrue(result["qualified"])
        self.assertEqual(result["inherited_setup_epochs"], 2)
        self.assertEqual(result["inherited_scan_pages"], 0)
        self.assertEqual(result["new_fitting_epochs"], 0)
        self.assertFalse(result["native_owners_opened"])
        self.assertEqual(len(result["excluded_historical_scan_objects"]), 6)
        self.assertEqual(stat.S_IMODE((destination / "private").stat().st_mode), 0o700)
        self.assertFalse((destination / "private/model.duckdb.owner.lock").exists())
        self.assertFalse((destination / "resource-admission.json").exists())
        for row in result["excluded_historical_scan_objects"]:
            self.assertFalse((destination / row["path"]).exists())
        raw = (destination / "materialized-successor-setup.json").read_bytes()
        receipt, digest = fixture.materialization(destination)
        self.assertEqual(digest, fixture.inert.sha(raw))
        self.assertEqual(receipt, result)
        for name in ("private/source.duckdb", "private/model.duckdb"):
            original, copied = SOURCE / name, destination / name
            self.assertNotEqual((original.stat().st_dev, original.stat().st_ino), (copied.stat().st_dev, copied.stat().st_ino))
            self.assertEqual(original.read_bytes(), copied.read_bytes())
            self.assertEqual(stat.S_IMODE(original.stat().st_mode), stat.S_IMODE(copied.stat().st_mode))
        self.assertEqual(before, fixture.inert.Reader(SOURCE, seconds=120).whole_archive())
        self.assertNotIn("duckdb", sys.modules)
        self.assertFalse(any(name.startswith("ipfs_datasets_py") for name in sys.modules))

    def test_seed_guard_pin_tamper_refused_before_materialization(self):
        guard = self.root / "forged-guard.json"
        guard.write_bytes(GUARD.read_bytes() + b" ")
        with self.assertRaisesRegex(fixture.SuccessorFullScanFixtureError, "pins differ"):
            fixture.stage_closed_successor_setup(SOURCE, AUDIT_PATH, guard, self.root / "new-copy")
        self.assertFalse((self.root / "new-copy").exists())

    def test_seed_audit_pin_tamper_refused_before_materialization(self):
        report = self.root / "forged-audit.json"
        report.write_bytes(AUDIT_PATH.read_bytes() + b" ")
        with self.assertRaisesRegex(fixture.SuccessorFullScanFixtureError, "pins differ"):
            fixture.stage_closed_successor_setup(SOURCE, report, GUARD, self.root / "new-copy")
        self.assertFalse((self.root / "new-copy").exists())

    def test_source_namespace_alias_refused(self):
        alias = self.root / "source-alias"; alias.symlink_to(SOURCE, target_is_directory=True)
        with self.assertRaises(fixture.SuccessorFullScanFixtureError):
            fixture.source_bundle(alias, AUDIT_PATH, GUARD)

    def test_absolute_member_read_refuses_symlink(self):
        source = self.root / "source"; source.write_bytes(b"expected")
        alias = self.root / "alias"; alias.symlink_to(source)
        with self.assertRaises(fixture.SuccessorFullScanFixtureError):
            fixture.read_absolute(alias)

    def test_member_copy_independent_inode_mode_and_bytes(self):
        source = self.root / "source"; source.mkdir()
        original = source / "artifact"; original.write_bytes(b"frozen source bytes"); original.chmod(0o640)
        row = fixture.inert.Reader(source).whole_archive()["files"][1]
        destination = self.root / "new-member"
        observed = fixture.copy_regular(fixture.inert.Reader(source), row, destination)
        self.assertEqual(observed["sha256"], row["sha256"])
        self.assertEqual(destination.read_bytes(), original.read_bytes())
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o640)
        self.assertNotEqual(destination.stat().st_ino, original.stat().st_ino)
        destination.write_bytes(b"changed new namespace")
        self.assertEqual(original.read_bytes(), b"frozen source bytes")

    def test_member_copy_changed_source_pin_refused(self):
        source = self.root / "source"; source.mkdir()
        original = source / "artifact"; original.write_bytes(b"first")
        row = fixture.inert.Reader(source).whole_archive()["files"][1]
        original.write_bytes(b"other")
        with self.assertRaisesRegex(fixture.SuccessorFullScanFixtureError, "source copy bytes"):
            fixture.copy_regular(fixture.inert.Reader(source), row, self.root / "new-member")
        self.assertFalse((self.root / "new-member").exists())

    def test_member_copy_existing_destination_refused(self):
        source = self.root / "source"; source.mkdir()
        (source / "artifact").write_bytes(b"first")
        row = fixture.inert.Reader(source).whole_archive()["files"][1]
        destination = self.root / "new-member"; destination.write_bytes(b"retained")
        with self.assertRaises(FileExistsError):
            fixture.copy_regular(fixture.inert.Reader(source), row, destination)
        self.assertEqual(destination.read_bytes(), b"retained")

    def test_copy_member_bound_rejects_bool_size(self):
        row = {"kind": "file", "nlink": 1, "bytes": True}
        with self.assertRaises(fixture.SuccessorFullScanFixtureError):
            fixture.copy_regular(fixture.inert.Reader(self.root), row, self.root / "new-member")

    def test_exact_copied_registry_relocation_expectation(self):
        before = {"meta": [[1, "schema", "ddl", "catalog", "/old/artifacts", 1]], "versions": [["frozen"]]}
        after = fixture.relocated_registry_observation(before, "/old/artifacts", "/new/artifacts")
        self.assertEqual(after, {"meta": [[1, "schema", "ddl", "catalog", "/new/artifacts", 1]], "versions": [["frozen"]]})
        self.assertEqual(before["meta"][0][4], "/old/artifacts")

    def test_copied_registry_relocation_wrong_path_refused(self):
        before = {"meta": [[1, "schema", "ddl", "catalog", "/unexpected/artifacts", 1]]}
        with self.assertRaisesRegex(fixture.SuccessorFullScanFixtureError, "old artifact path"):
            fixture.relocated_registry_observation(before, "/old/artifacts", "/new/artifacts")

    def test_copied_registry_owner_generation_bool_alias_refused(self):
        before = {"meta": [[1, "schema", "ddl", "catalog", "/old/artifacts", True]]}
        with self.assertRaises(fixture.SuccessorFullScanFixtureError):
            fixture.relocated_registry_observation(before, "/old/artifacts", "/new/artifacts")

    def test_copied_source_relocation_changes_only_cas_root(self):
        before = {"ast": {"table": {"rows": 1, "sha256": "frozen"}}, "catalog": {
            "meta": [[1, "schema", "ddl", "catalog", "/old/cas"]], "heads": [["current"]], "operations": [["frozen"]]}}
        after = fixture.relocated_source_observation(before, "/old/cas", "/new/cas")
        self.assertEqual(after["catalog"]["meta"][0][4], "/new/cas")
        self.assertEqual(before["catalog"]["meta"][0][4], "/old/cas")
        self.assertEqual(after["ast"], before["ast"])
        self.assertEqual(after["catalog"]["heads"], before["catalog"]["heads"])
        self.assertEqual(after["catalog"]["operations"], before["catalog"]["operations"])

    def test_copied_source_relocation_wrong_cas_root_refused(self):
        before = {"ast": {}, "catalog": {"meta": [[1, "schema", "ddl", "catalog", "/unexpected/cas"]], "heads": [], "operations": []}}
        with self.assertRaises(fixture.SuccessorFullScanFixtureError):
            fixture.relocated_source_observation(before, "/old/cas", "/new/cas")

    def test_native_new_copy_owner_open_and_current_receiving_no_fit(self):
        # This runtime boundary cannot be established by inert transport tests.
        # The old closed owners are never opened. All mutations stay in this
        # temporary new copy and are fully joined to their frozen observations.
        destination = self.root / "new-native-copy"
        setup = fixture.stage_closed_successor_setup(SOURCE, AUDIT_PATH, GUARD, destination)
        host = fixture.inert.parse((BASE / "successor-expansion-resources-20261003-01/configuration.json").read_bytes())
        scheduler = fixture.shared_scheduler(host["state_path"], host["persisted_config"])
        report = {"post_setup_fit_attempt_count": 0, "inference_attempts": 0, "owner_pairs_opened": 0,
            "source_publication_performed": False, "old_native_owners_opened": False,
            "scope": "one new-copy pair, both copied binding relocations, full current source receiving; no page execution"}
        index = registry = connection = None
        from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor as delta
        from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as scan
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
        def forbid_inference(*args, **kwargs):
            report["inference_attempts"] += 1
            raise AssertionError("new copied-owner receiving attempted numerical inference")
        try:
            with fixture.no_fit(report), patch.object(scan, "_worker", forbid_inference), patch.object(features, "infer_projection_features", forbid_inference):
                index, registry, connection = fixture.open_materialized_successor(destination)
                report["owner_pairs_opened"] += 1
                before = fixture.owners(index, registry, connection)
                states = fixture.checkpoint_states(registry, setup)
                self.assertEqual(states, setup["checkpoint_states"])
                self.assertEqual(fixture.owner_generation(before), 3)
                self.assertEqual(before["source"]["current_head"], setup["current_head"])
                raw = fixture.inert.parse((destination / "seed-evidence/source-delta.json").read_bytes())
                transition = delta.load_codebase_source_delta(index.artifacts, raw["artifact_cid"])
                self.assertIs(delta.validate_current_codebase_source_delta(transition, index, destination / "repository",
                    scheduler=scheduler, timeout_seconds=180.0, memory_mb=1024), transition)
                self.assertEqual(fixture.owners(index, registry, connection), before)
                self.assertEqual(fixture.checkpoint_states(registry, setup), states)
                source_relocation = fixture.inert.parse((destination / "copied-source-catalog-relocation.json").read_bytes())
                model_relocation = fixture.inert.parse((destination / "copied-registry-relocation.json").read_bytes())
                self.assertTrue(source_relocation["all13_ast_table_rows_preserved"])
                self.assertFalse(source_relocation["native_source_publication_performed"])
                self.assertEqual(model_relocation["owner_generation_before"], 2)
                self.assertEqual(model_relocation["owner_generation_after"], 2)
                self.assertFalse(model_relocation["owner_generation_advanced"])
            self.assertEqual(report["post_setup_fit_attempt_count"], 0)
            self.assertEqual(report["inference_attempts"], 0)
            expected = fixture.inert.parse((SOURCE / "generation-inputs.json").read_bytes())["files"]
            imported = []
            for row in expected:
                if row["name"] == "__main__":
                    # The seed names its historical executed harness by this
                    # alias; today's test runner has a different main module.
                    # source_bundle still verifies the historical body path.
                    continue
                module = sys.modules.get(row["name"])
                if module is not None:
                    path = Path(module.__file__).resolve(); raw = path.read_bytes()
                    self.assertEqual(str(path), row["path"])
                    self.assertEqual(fixture.pin(raw), {key: row[key] for key in ("bytes", "sha256")})
                    imported.append({"name": row["name"], "path": str(path), **fixture.pin(raw)})
            # Registry/scheduler files were outside the historical 34-file
            # export. Observe them explicitly for this new receiving run.
            for name in ("ipfs_datasets_py.duckdb_control.autoencoder_registry",
                         "ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler"):
                module = sys.modules[name]
                path = Path(module.__file__).resolve()
                expected_path = Path(fixture.__file__).resolve().parents[4] / "ipfs_datasets" / Path(*name.split(".")).with_suffix(".py")
                self.assertEqual(path, expected_path)
                imported.append({"name": name, "path": str(path), **fixture.pin(path.read_bytes()),
                    "scope": "new_current_import_outside_inherited34_pin_set"})
            self.assertTrue(any(row["name"].endswith("resource_scheduler") for row in imported))
            self.assertTrue(any(row["name"].endswith("codebase_catalog") for row in imported))
            self.assertTrue(any(row["name"].endswith("autoencoder_registry") for row in imported))
            report["actual_imported_seed_producers"] = imported
            report["working_directory"] = str(Path.cwd())
            report["PYTHONPATH"] = os.environ.get("PYTHONPATH")
            report["qualified"] = True
        finally:
            fixture.close_materialized_successor(registry, connection)
            report["final_resources"] = fixture.assert_clean(scheduler)
            NATIVE_RECEIVING.append(report)

    def test_relocation_uses_final_closed_owner_generation(self):
        destination = self.root / "new-copy"; evidence = destination / "seed-evidence"; evidence.mkdir(parents=True)
        previous = fixture.inert.parse((SOURCE / "owners-before.json").read_bytes())
        final = fixture.inert.parse((SOURCE / "owners-after-cold.json").read_bytes())
        (evidence / "owners-before.json").write_bytes(fixture.inert.numerical_wire(previous))
        (evidence / "owners-after-cold.json").write_bytes(fixture.inert.numerical_wire(final))
        self.assertEqual(previous["registry"]["meta"][0][5], 1)
        self.assertEqual(final["registry"]["meta"][0][5], 2)
        self.assertEqual(fixture.closed_copied_registry(destination), final["registry"])
        self.assertIn("owners-after-cold.json", fixture.EVIDENCE_NAMES)

    def test_explicit_generation_reopen_preserves_other_rows(self):
        observed = {"registry": {"meta": [[1, "schema", "ddl", "catalog", "/new/artifacts", 2]], "versions": [["frozen"]]}, "source": {"current_head": "frozen"}}
        after = fixture.expected_after_reopens(observed, 4)
        self.assertEqual(fixture.owner_generation(after), 6)
        self.assertEqual(after["registry"]["versions"], observed["registry"]["versions"])
        self.assertEqual(after["source"], observed["source"])
        self.assertEqual(fixture.owner_generation(observed), 2)

    def test_generation_reopen_count_exact_integer(self):
        observed = {"registry": {"meta": [[1, "schema", "ddl", "catalog", "/new/artifacts", 2]]}}
        for count in (True, -1, 9):
            with self.subTest(count=count), self.assertRaises(fixture.SuccessorFullScanFixtureError):
                fixture.expected_after_reopens(observed, count)


class RequestControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="successor-chunk-request-controls-")
        self.root = Path(self.temporary.name).resolve()
        self.state = self.root / "shared-state.json"; self.state.write_bytes(b"{}")
        selected = fixture.inert.parse((SOURCE / "successor-selection.json").read_bytes())["value"]
        self.request = {"schema": fixture.REQUEST_SCHEMA, "output": str(self.root), "run_number": 1,
            "root_cid": selected["root_cid"], "cursor": {"schema": "codebase-inventory-resume-cursor@1", "root_cid": selected["root_cid"],
                "next_offset": 64, "previous_page_cid": fixture.inert.cid(b"page", source=True)},
            "version_id": selected["model"]["version_id"], "expected_head": selected["current_head"],
            "checkpoint_states": fixture.inert.parse((SOURCE / "checkpoint-states-after.json").read_bytes()),
            "expected_registry_owner_generation": 3, "scheduler_state_path": str(self.state),
            "scheduler_configuration": {"proof_safety_enabled": True}, "timeout_seconds": 900,
            "max_pages": 2, "materialization_receipt_sha256": "1" * 64, "helper_sha256": "2" * 64}

    def tearDown(self):
        self.temporary.cleanup()

    def test_bounded_request_positive(self):
        self.assertIs(fixture.validate_chunk_request(self.request), self.request)

    def reject(self, mutation):
        mutation(self.request)
        with self.assertRaises((fixture.SuccessorFullScanFixtureError, fixture.inert.SourceSuccessorAuditError)):
            fixture.validate_chunk_request(self.request)

    def test_request_closed_fields(self):
        self.reject(lambda request: request.__setitem__("unreviewed", True))

    def test_request_run_number_bool_alias(self):
        self.reject(lambda request: request.__setitem__("run_number", True))

    def test_request_page_count_bool_alias(self):
        self.reject(lambda request: request.__setitem__("max_pages", True))

    def test_request_page_count_exceeds_bound(self):
        self.reject(lambda request: request.__setitem__("max_pages", 3))

    def test_request_deadline_bool_alias(self):
        self.reject(lambda request: request.__setitem__("timeout_seconds", True))

    def test_request_deadline_nonfinite(self):
        self.reject(lambda request: request.__setitem__("timeout_seconds", math.inf))

    def test_request_cursor_root_join(self):
        self.reject(lambda request: request["cursor"].__setitem__("root_cid", fixture.inert.structured({"other": True})))

    def test_request_cursor_offset_bool_alias(self):
        self.reject(lambda request: request["cursor"].__setitem__("next_offset", True))

    def test_request_cursor_complete_offset_is_not_resumable(self):
        self.reject(lambda request: request["cursor"].__setitem__("next_offset", 300))

    def test_request_previous_page_cid_codec(self):
        self.reject(lambda request: request["cursor"].__setitem__("previous_page_cid", fixture.inert.structured({"page": True})))

    def test_request_helper_pin_canonical_hex(self):
        self.reject(lambda request: request.__setitem__("helper_sha256", "Z" * 64))

    def test_request_frozen_checkpoint_population(self):
        self.reject(lambda request: request["checkpoint_states"].pop("root"))

    def test_request_proof_sampler_profile_enabled(self):
        self.reject(lambda request: request["scheduler_configuration"].__setitem__("proof_safety_enabled", 1))

    def test_request_shared_state_path_alias(self):
        alias = self.root / "state-alias"; alias.symlink_to(self.state)
        self.reject(lambda request: request.__setitem__("scheduler_state_path", str(alias)))

    def test_request_owner_generation_bool_alias(self):
        self.reject(lambda request: request.__setitem__("expected_registry_owner_generation", True))


class ScopedResourceControls(unittest.TestCase):
    class Scheduler:
        def __init__(self, state):
            self.state = state
        def snapshot(self):
            return {}
        @contextmanager
        def _locked_state(self, *, persist=False):
            yield self.state

    def test_other_host_clients_may_have_active_reservations(self):
        scheduler = self.Scheduler({"leases": {"cuda": {"owner_pid": os.getpid() + 10000}}, "waiters": {}})
        result = fixture.assert_clean(scheduler)
        self.assertEqual(result["active_lease_count"], 0)
        self.assertEqual(result["global_active_lease_count"], 1)
        self.assertEqual(result["scope"], "named_scan_process_owners_only")

    def test_scan_owner_lease_must_drain(self):
        scheduler = self.Scheduler({"leases": {"scan": {"owner_pid": os.getpid()}}, "waiters": {}})
        with self.assertRaisesRegex(fixture.SuccessorFullScanFixtureError, "did not drain"):
            fixture.assert_clean(scheduler)

    def test_scan_owner_waiter_must_drain(self):
        scheduler = self.Scheduler({"leases": {}, "waiters": {"scan": {"owner_pid": os.getpid()}}})
        with self.assertRaises(fixture.SuccessorFullScanFixtureError):
            fixture.assert_clean(scheduler)

    def test_child_and_parent_owned_reservations_are_checked_together(self):
        scheduler = self.Scheduler({"leases": {"child": {"owner_pid": 33333}}, "waiters": {}})
        with self.assertRaises(fixture.SuccessorFullScanFixtureError):
            fixture.assert_clean(scheduler, owner_pids=[os.getpid(), 33333])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-directory", type=Path, required=True)
    args = parser.parse_args()
    output = args.receipt_directory.absolute(); output.mkdir(parents=True, exist_ok=False)
    helper_before, test_before = Path(fixture.__file__).read_bytes(), Path(__file__).read_bytes()
    (output / "helper.py").write_bytes(helper_before); (output / "test.py").write_bytes(test_before)
    log = io.StringIO(); started = time.monotonic()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    unchanged = Path(fixture.__file__).read_bytes() == helper_before and Path(__file__).read_bytes() == test_before
    receipt = {"schema": "source-successor-full-scan-fixture-guard-tests@1", "qualified": result.wasSuccessful() and unchanged and not result.skipped,
        "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "recorded_seconds": time.monotonic() - started, "helper": {"path": str(Path(fixture.__file__).resolve()), **fixture.pin(helper_before)},
        "test": {"path": str(Path(__file__).resolve()), **fixture.pin(test_before)}, "current_pins_unchanged": unchanged,
        "source_namespace": str(SOURCE), "new_native_receiving_controls": NATIVE_RECEIVING,
        "old_native_owner_opens": 0, "old_git_commands": 0, "native_owner_pairs_opened": sum(row["owner_pairs_opened"] for row in NATIVE_RECEIVING),
        "native_sql_and_new_repository_source_observation": bool(NATIVE_RECEIVING),
        "child_interpreters": 0, "scan_workers": 0, "model_fitting": 0, "proof_authority": False,
        "scope": "stdlib transport/guards plus one explicitly marked real new-copy owner/current receiving boundary; native complete scan qualified separately"}
    (output / "tests.log").write_text(log.getvalue())
    fixture.write_exclusive(output / "receipt.json", receipt)
    for path in output.iterdir():
        path.chmod(0o444)
    print(json.dumps({key: receipt[key] for key in ("qualified", "tests", "failures", "errors", "skipped", "recorded_seconds", "current_pins_unchanged")}, sort_keys=True))
    return 0 if receipt["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
