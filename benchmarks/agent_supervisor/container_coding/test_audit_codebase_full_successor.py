"""Pure stdlib controls for full-chain and complete stored role/source joins.

The native fixture is inert. Ten-page synthetic examples qualify structural
receiving only; they execute no new inference and attest no execution. All
mutations are detached JSON, and closed primary bytes stay unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import struct
import time
import unittest

HERE = Path(__file__).resolve().parent
READER_PATH = HERE / "audit_codebase_full_successor.py"
spec = importlib.util.spec_from_file_location("full_successor_standalone_reader_controls", READER_PATH)
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
inert = audit.inert
SOURCE = Path("/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/source-successor-qualification-20261003-02")
FULL_SOURCE = SOURCE.parent / "source-successor-full-scan-qualification-20261003-02"


def envelope(value, *, numerical=False):
    return {"artifact_cid": inert.cid(inert.numerical_wire(value), source=True) if numerical else inert.structured(value), "value": value}


class SourceRoleControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = inert.Reader(SOURCE, seconds=120)
        cls.before = cls.reader.whole_archive()
        cls.selection = cls.reader.json("successor-selection.json")["value"]
        cls.previous = cls.reader.json("previous-manifest.json")["value"]
        cls.current = cls.reader.json("current-manifest.json")["value"]
        cls.current_receipt = cls.reader.json("current-publication-receipt.json")
        cls.models = {}
        for label, field, head in (("root", "previous_model", "previous_head"), ("child", "model", "current_head")):
            model = cls.selection[field]; digest = model["artifact"]["sha256"]
            raw = cls.reader.raw("private/model-artifacts/" + digest[:2] + "/" + digest, 16 * inert.MIB)
            cls.models[label] = inert.verify_model(model, raw, cls.selection[head])

    @classmethod
    def tearDownClass(cls):
        if not inert.same(cls.before, inert.Reader(SOURCE, seconds=120).whole_archive()):
            raise AssertionError("closed native source archive changed during pure role controls")

    def check(self, models):
        return audit.verify_current_evaluation_bindings(models["root"], models["child"], self.previous, self.current, get_artifact=self.reader.cas)

    def refresh(self, models):
        """Produce byte-valid numerical checkpoints after report mutations."""
        for label in ("root", "child"):
            saved = models[label]["saved"]; p = saved["report"]["codebase_provenance"]
            saved["report"]["training_targets_sha256"] = inert.sha(inert.numerical_wire(p["training_targets"]))
            saved["report"]["tuning_targets_sha256"] = saved["state"]["tuning_targets_sha256"] = inert.sha(inert.numerical_wire(p["tuning_targets"]))
            saved["report"]["codebase_request_sha256"] = inert.sha(inert.numerical_wire(inert.training_request(saved)))
            if label == "child":
                saved["report"]["base_state_sha256"] = inert.sha(inert.numerical_wire(models["root"]["saved"]["state"]))
            model = deepcopy(self.selection["previous_model" if label == "root" else "model"])
            raw = inert.numerical_wire(saved)
            model["artifact"] = {"sha256": inert.sha(raw), "bytes": len(raw)}
            model["artifact_cid"] = inert.cid(raw, source=True)
            model["state_sha256"] = inert.sha(inert.numerical_wire(saved["state"]))
            model["ancestry"][0]["artifact"] = deepcopy(model["artifact"])
            models[label] = inert.verify_model(model, raw, self.selection["previous_head" if label == "root" else "current_head"])
        return models

    def current_target(self, target, path):
        result = deepcopy(target)
        rows = [row for row in result["validation"] if row["validator_id"] == "codebase_ir.exact_native_target_replay@1"]
        details = rows[0]["details"]
        binding = audit.current_binding(self.current, self.selection["current_head"], path, get_artifact=self.reader.cas)
        binding.pop("authored_contracts")
        details["source_binding"] = binding
        details["source_bytes_hex"] = self.reader.cas(binding["source_cid"], source=True).hex()
        details["captured_ast"] = inert.parse(self.reader.cas(binding["ast_cid"]))
        details["manifest"] = deepcopy(self.current)
        details["publication_receipt"] = deepcopy(self.current_receipt)
        result["source_digest"] = inert.sha(inert.numerical_wire({"target_schema": details["target_schema"],
            "source_binding": binding, "authored_contracts": details["authored_contracts"]}))
        # The enclosing native custody now agrees; no source lowering is run.
        inert.stored_target_binding(result)
        return result

    def test_actual_inert_models_have_complete_current_evaluation_and_role_bindings(self):
        result = self.check(self.models)
        self.assertTrue(result["complete_current_evaluation_bindings_verified"])
        self.assertTrue(result["frozen_cohort_role_source_bytes_verified"])
        self.assertFalse(result["targets_independently_relowered"])

    def denied(self, models, message):
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, message):
            self.check(models)

    def test_same_roles_changed_current_evaluation_head_rehashed_checkpoint(self):
        models = deepcopy(self.models)
        p = models["child"]["saved"]["report"]["codebase_provenance"]
        p["current_evaluation_bindings"][0]["binding"]["head"] = deepcopy(self.selection["previous_head"])
        self.denied(self.refresh(models), "current evaluation source bindings")

    def test_current_evaluation_role_path_swapped_rehashed_checkpoint(self):
        models = deepcopy(self.models)
        rows = models["child"]["saved"]["report"]["codebase_provenance"]["current_evaluation_bindings"]
        rows[0]["role"], rows[1]["role"] = rows[1]["role"], rows[0]["role"]
        self.denied(self.refresh(models), "current evaluation source bindings")

    def test_current_evaluation_incomplete_population_rehashed_checkpoint(self):
        models = deepcopy(self.models)
        models["child"]["saved"]["report"]["codebase_provenance"]["current_evaluation_bindings"].pop()
        self.denied(self.refresh(models), "current evaluation source bindings")

    def test_current_evaluation_extra_authored_contracts_rehashed_checkpoint(self):
        models = deepcopy(self.models)
        models["child"]["saved"]["report"]["codebase_provenance"]["current_evaluation_bindings"][0]["binding"]["authored_contracts"] = [{"other": True}]
        self.denied(self.refresh(models), "current evaluation source bindings")

    def test_current_evaluation_same_bytes_wrong_entry_custody_rehashed_checkpoint(self):
        models = deepcopy(self.models)
        models["child"]["saved"]["report"]["codebase_provenance"]["current_evaluation_bindings"][0]["binding"]["entry"]["disposition"] = "working"
        self.denied(self.refresh(models), "current evaluation source bindings")

    def test_coherent_frozen_canary_rebound_to_current_head_both_models(self):
        models = deepcopy(self.models)
        target = self.current_target(models["root"]["saved"]["report"]["codebase_provenance"]["canary_targets"][0], "canary.py")
        for model in models.values():
            model["saved"]["report"]["codebase_provenance"]["canary_targets"] = [deepcopy(target)]
        models = self.refresh(models)
        inert.verify_model_lineage(models["root"], models["child"])
        self.denied(models, "complete frozen/replay root-head")

    def test_coherent_frozen_tune_rebound_to_current_head_both_models(self):
        models = deepcopy(self.models)
        target = self.current_target(models["root"]["saved"]["report"]["codebase_provenance"]["tuning_targets"][0], "tune.py")
        for model in models.values():
            model["saved"]["report"]["codebase_provenance"]["tuning_targets"] = [deepcopy(target)]
        models = self.refresh(models)
        inert.verify_model_lineage(models["root"], models["child"])
        self.denied(models, "complete frozen/replay root-head")

    def test_current_binding_raw_cas_byte_drift(self):
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "raw source bytes"):
            audit.current_binding(self.current, self.selection["current_head"], "canary.py", get_artifact=lambda cid, source=False: b"drift")

    def test_current_binding_missing_role_path(self):
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "absent/ambiguous"):
            audit.current_binding(self.current, self.selection["current_head"], "missing.py", get_artifact=self.reader.cas)


class CompleteChainControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reader = inert.Reader(SOURCE, seconds=120)
        cls.root = reader.json("scan-root.json")
        cls.first = reader.json("prefix-page.json")
        cls.pages = [cls.first]
        previous = cls.first["artifact_cid"]
        for start in range(32, 300, 32):
            end = min(start + 32, 300); members = cls.root["value"]["members"][start:end]
            page = deepcopy(cls.first["value"])
            page.update(start=start, end=end, previous_page_cid=previous,
                page_membership_cid=inert.structured(members), entries=[])
            for ordinal, member in enumerate(members, start):
                disposition = "opaque" if member["opaque_reason"] is not None else {"failed": "parse_failed", "partial": "parse_partial", "unindexed": "unindexed"}.get(member["parse_status"])
                if disposition is None:
                    disposition = "deferred_budget" if member["path"].endswith(".py") else "unsupported_target"
                page["entries"].append({"member_index": ordinal, "source_key": member["source_key"], "entry_cid": member["entry_cid"],
                    "disposition": disposition, "reason": "synthetic_structural_receiving_control", "target_sha256": None,
                    "source_digest": None, "coverage": [], "inference_index": None})
            counts = {disposition: sum(row["disposition"] == disposition for row in page["entries"])
                      for disposition in sorted({row["disposition"] for row in page["entries"]})}
            page["coverage"] = {"inventory_entries": end - start, "inferred_rows": 0, "dispositions": counts}
            page["inference"]["rows"] = []; page["inference"]["coverage"] = []; page["worker_receipt"] = None
            current = envelope(page, numerical=True); cls.pages.append(current); previous = current["artifact_cid"]
        descriptors = []; counts = {}; inferred = 0
        for page in cls.pages:
            value = page["value"]
            descriptor = {"page_cid": page["artifact_cid"], "start": value["start"], "end": value["end"],
                "membership_cid": value["page_membership_cid"], "inferred_rows": value["coverage"]["inferred_rows"], "dispositions": value["coverage"]["dispositions"]}
            descriptors.append(descriptor); inferred += descriptor["inferred_rows"]
            for disposition, amount in descriptor["dispositions"].items():
                counts[disposition] = counts.get(disposition, 0) + amount
        root = cls.root["value"]
        cls.completion = envelope({"schema": "codebase-inventory-resume-completion@1", "root_cid": cls.root["artifact_cid"],
            "head_cid": root["head_cid"], "membership_cid": root["membership_cid"], "model_artifact_cid": root["model"]["artifact_cid"],
            "pages": descriptors, "coverage": {"inventory_entries": 300, "pages": 10, "inferred_rows": inferred, "dispositions": dict(sorted(counts.items()))},
            "authority": deepcopy(root["authority"])})

    def test_actual_prefix_and_synthetic_complete300_chain(self):
        result = audit.verify_completion(self.completion, self.root, self.pages)
        self.assertTrue(result["all300_members_have_explicit_dispositions"])
        self.assertFalse(result["numerical_execution_independently_reperformed"])
        self.assertFalse(result["all300_members_numerically_inferred"])

    def test_legitimate_zero_row_page_has_no_worker_receipt(self):
        page = self.pages[1]
        result = audit.verify_page(page, self.root, start=32, previous_page_cid=self.first["artifact_cid"])
        self.assertEqual(result["inferred_rows"], 0)

    def test_zero_row_page_receipt_false_alias_rehashed(self):
        page = deepcopy(self.pages[1]["value"]); page["worker_receipt"] = False
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "no numerical worker"):
            audit.verify_page(envelope(page, numerical=True), self.root, start=32, previous_page_cid=self.first["artifact_cid"])

    def test_zero_row_page_receipt_zero_alias_rehashed(self):
        page = deepcopy(self.pages[1]["value"]); page["worker_receipt"] = 0
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "no numerical worker"):
            audit.verify_page(envelope(page, numerical=True), self.root, start=32, previous_page_cid=self.first["artifact_cid"])

    def test_later_page_root_binding_rehashed(self):
        page = deepcopy(self.pages[1]["value"]); page["root_cid"] = inert.structured({"other": True})
        with self.assertRaises(inert.SourceSuccessorAuditError):
            audit.verify_page(envelope(page, numerical=True), self.root, start=32, previous_page_cid=self.first["artifact_cid"])

    def test_later_page_ordered_entry_swap_rehashed(self):
        page = deepcopy(self.pages[1]["value"]); page["entries"].reverse()
        with self.assertRaises(inert.SourceSuccessorAuditError):
            audit.verify_page(envelope(page, numerical=True), self.root, start=32, previous_page_cid=self.first["artifact_cid"])

    def test_later_page_member_index_bool_alias_rehashed(self):
        page = deepcopy(self.pages[1]["value"]); page["entries"][0]["member_index"] = True
        with self.assertRaises(inert.SourceSuccessorAuditError):
            audit.verify_page(envelope(page, numerical=True), self.root, start=32, previous_page_cid=self.first["artifact_cid"])

    def test_last_page_incomplete_population_rehashed(self):
        page = deepcopy(self.pages[-1]["value"]); page["entries"].pop()
        with self.assertRaises(inert.SourceSuccessorAuditError):
            audit.verify_page(envelope(page, numerical=True), self.root, start=288, previous_page_cid=self.pages[-2]["artifact_cid"])

    def test_last_page_exact12_complete_population(self):
        result = audit.verify_page(self.pages[-1], self.root, start=288, previous_page_cid=self.pages[-2]["artifact_cid"])
        self.assertEqual(result["end"], 300)
        self.assertEqual(len(self.pages[-1]["value"]["entries"]), 12)

    def test_complete_chain_missing_page(self):
        with self.assertRaises(inert.SourceSuccessorAuditError):
            audit.verify_completion(self.completion, self.root, self.pages[:-1])

    def test_complete_chain_page_order(self):
        pages = deepcopy(self.pages); pages[2], pages[3] = pages[3], pages[2]
        with self.assertRaises(inert.SourceSuccessorAuditError):
            audit.verify_completion(self.completion, self.root, pages)

    def test_complete_chain_duplicate_page(self):
        pages = deepcopy(self.pages); pages[2] = deepcopy(pages[1])
        with self.assertRaises(inert.SourceSuccessorAuditError):
            audit.verify_completion(self.completion, self.root, pages)

    def test_complete_coverage_rehashed_count_alias(self):
        value = deepcopy(self.completion["value"]); value["coverage"]["pages"] = True
        with self.assertRaises(inert.SourceSuccessorAuditError):
            audit.verify_completion(envelope(value), self.root, self.pages)

    def test_complete_descriptor_rehashed_wrong_prefix(self):
        value = deepcopy(self.completion["value"]); value["pages"][3]["start"] -= 1
        with self.assertRaises(inert.SourceSuccessorAuditError):
            audit.verify_completion(envelope(value), self.root, self.pages)

    def test_complete_authority_rehashed_bool_alias(self):
        value = deepcopy(self.completion["value"]); value["authority"]["proof_authority"] = 0
        with self.assertRaises(inert.SourceSuccessorAuditError):
            audit.verify_completion(envelope(value), self.root, self.pages)

    def test_page_export_extra_authority(self):
        page = deepcopy(self.pages[1]); page["proof_verified"] = True
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "fields required"):
            audit.verify_page(page, self.root, start=32, previous_page_cid=self.first["artifact_cid"])

    def test_completion_export_extra_authority(self):
        completion = deepcopy(self.completion); completion["proof_verified"] = True
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "fields required"):
            audit.verify_completion(completion, self.root, self.pages)


class OptOutControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reader = inert.Reader(SOURCE, seconds=120)
        cls.default = reader.json("prefix-page.json"); cls.reference = reader.json("reference-prefix-page.json")
        cls.record = {"entries_coverage_and_inference_exact": True, "scope": "first32_ordered_members_only",
            "reference_page_cid": cls.reference["artifact_cid"], "optimized_page_cid": cls.default["artifact_cid"], "throughput_qualified": False}

    def test_actual_inert_first32_exact_scope(self):
        self.assertFalse(audit.verify_opt_out_equivalence(self.record, self.default, self.reference)["throughput_qualified"])

    def test_equivalence_record_different_page_identity(self):
        record = deepcopy(self.record); record["reference_page_cid"] = self.default["artifact_cid"]
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "complete identities"):
            audit.verify_opt_out_equivalence(record, self.default, self.reference)

    def test_equivalence_record_broadens_to_all300(self):
        record = deepcopy(self.record); record["scope"] = "all300_ordered_members"
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "scope"):
            audit.verify_opt_out_equivalence(record, self.default, self.reference)

    def test_equivalence_record_false_bool_alias(self):
        record = deepcopy(self.record); record["throughput_qualified"] = 0
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "scope"):
            audit.verify_opt_out_equivalence(record, self.default, self.reference)


class ScopedResourcesControls(unittest.TestCase):
    def test_other_host_gpu_clients_allowed(self):
        result = audit.verify_scoped_resources({"active_lease_count": 0, "waiting_request_count": 0, "owner_pids": [1234],
            "scope": "named_scan_process_owners_only", "global_active_lease_count": 1, "global_waiting_request_count": 0}, [1234])
        self.assertEqual(result["global_active_lease_count"], 1)

    def test_owned_zero_counter_bool_alias_rejected(self):
        value = {"active_lease_count": False, "waiting_request_count": 0, "owner_pids": [1234],
            "scope": "named_scan_process_owners_only", "global_active_lease_count": 1, "global_waiting_request_count": 0}
        with self.assertRaises(inert.SourceSuccessorAuditError):
            audit.verify_scoped_resources(value, [1234])

    def test_missing_child_owner_pid_rejected(self):
        value = {"active_lease_count": 0, "waiting_request_count": 0, "owner_pids": [1234],
            "scope": "named_scan_process_owners_only", "global_active_lease_count": 0, "global_waiting_request_count": 0}
        with self.assertRaises(inert.SourceSuccessorAuditError):
            audit.verify_scoped_resources(value, [1234, 5678])


class MaterializationControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = SOURCE.parent
        cls.upstream = inert.Reader(parent, seconds=120).json("source-successor-qualification-20261003-02-readonly-audit-20261003-01.json", 2 * inert.MIB)
        cls.setup = inert.Reader(parent / "source-successor-full-scan-qualification-20261003-01", seconds=120).json("materialized-successor-setup.json", 2 * inert.MIB)

    def test_actual_closed_transport_entire_population(self):
        result = audit.verify_materialization(self.setup, self.upstream, self.setup["destination"])
        self.assertTrue(result["complete_population_verified"])
        self.assertEqual(result["copied_files"], 1535)

    def denied(self, mutate, text):
        setup = deepcopy(self.setup); mutate(setup)
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, text):
            audit.verify_materialization(setup, self.upstream, self.setup["destination"])

    def test_recounted_missing_copied_source_member(self):
        def mutate(setup):
            member = setup["copied_members"].pop()
            setup["copied_files"] -= 1; setup["copied_bytes"] -= member["bytes"]
        self.denied(mutate, "entire copied")

    def test_coherent_copied_source_locator_alias(self):
        self.denied(lambda value: value["copied_members"][0].update(path="private/../private/model.duckdb"), "entire copied")

    def test_excluded_six_different_identity(self):
        self.denied(lambda value: value["excluded_historical_scan_objects"][0].update(cid=value["excluded_historical_scan_objects"][1]["cid"]), "exact six")

    def test_excluded_six_duplicate_identity(self):
        self.denied(lambda value: value["excluded_historical_scan_objects"].__setitem__(0, deepcopy(value["excluded_historical_scan_objects"][1])), "exact six")

    def test_child_model_version_changed(self):
        self.denied(lambda value: value["selected_models"]["child"].update(version_id=value["selected_models"]["root"]["version_id"]), "source/model generation")

    def test_current_head_coherently_reverted(self):
        self.denied(lambda value: value.update(current_head=deepcopy(self.upstream["source_delta"]["previous_head"])), "source/model generation")

    def test_unrecognized_materialization_authority(self):
        self.denied(lambda value: value.update(execution_verified=True), "fields required")

    def test_native_owners_false_counter_alias(self):
        self.denied(lambda value: value.update(native_owners_opened=0), "transport/accounting")

    def test_copied_byte_size_boolean_alias(self):
        def mutate(value):
            member = value["copied_members"][0]
            member["bytes"] = False
        self.denied(mutate, "integer")


class SharedHostControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = SOURCE.parent / "successor-expansion-resources-20261003-01/configuration.json"
        cls.raw = inert.Reader(path.parent, seconds=120).raw(path.name, inert.MIB)
        cls.host = inert.parse(cls.raw)
        cls.result = {"host_configuration_pin": {"bytes": len(cls.raw), "sha256": inert.sha(cls.raw)},
            "scheduler_state_path": cls.host["state_path"], "scheduler_configuration": cls.host["persisted_config"]}

    def test_actual_existing_common_host_generation(self):
        self.assertTrue(audit.verify_shared_host(self.host, self.result, self.raw)["shared_pool_bound"])

    def test_coherently_rebound_pool_still_rejects_original_host_bytes(self):
        host, result = deepcopy(self.host), deepcopy(self.result)
        host["state_path"] = result["scheduler_state_path"] = "/synthetic/separate-pool/admission.json"
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "original host configuration"):
            audit.verify_shared_host(host, result, self.raw)

    def test_source_pin_is_exact_not_only_parsed_configuration(self):
        result = deepcopy(self.result); result["host_configuration_pin"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "original host configuration"):
            audit.verify_shared_host(self.host, result, self.raw)

    def test_unrecognized_shared_host_authority(self):
        host = deepcopy(self.host); host["kernel_origin_attested"] = True
        raw = inert.numerical_wire(host)
        result = deepcopy(self.result); result["host_configuration_pin"] = {"bytes": len(raw), "sha256": inert.sha(raw)}
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "fields required"):
            audit.verify_shared_host(host, result, raw)


class MemoryReader:
    """Detached protocol preimages; no fixture producer code or child runs."""
    def __init__(self):
        self.root = Path("/synthetic/new-source-successor"); self.values = {}; self.raws = {}

    def put(self, path, value):
        self.values[path] = deepcopy(value); self.raws[path] = inert.numerical_wire(value) + b"\n"

    def json(self, path, maximum=16 * inert.MIB):
        return deepcopy(self.values[path])

    def raw(self, path, maximum=16 * inert.MIB):
        raw = self.raws[path]
        if len(raw) > maximum: raise AssertionError("control read exceeds declared bound")
        return raw


class FreshProcessControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CompleteChainControls.setUpClass()
        cls.root, cls.pages, cls.completion = CompleteChainControls.root, CompleteChainControls.pages, CompleteChainControls.completion
        cls.frozen = inert.Reader(SOURCE, seconds=120).json("checkpoint-states-after.json")
        cls.host = SharedHostControls.host if hasattr(SharedHostControls, "host") else inert.Reader(SOURCE.parent / "successor-expansion-resources-20261003-01", seconds=120).json("configuration.json")

    def fixture(self):
        reader = MemoryReader(); helper_sha = "1" * 64; helper_path = str(HERE / "source_successor_full_scan_fixture.py")
        reader.put("owners-before.json", {"registry": {"meta": [["m", "s", "c", "v", "/new/model-artifacts", 3]]}})
        reader.put("materialized-successor-setup.json", {"synthetic": True})
        result = {"pid": 1234, "fresh_processes": [], "scheduler_state_path": self.host["state_path"],
            "scheduler_configuration": deepcopy(self.host["persisted_config"])}
        cursor = {"schema": "codebase-inventory-resume-cursor@1", "root_cid": self.root["artifact_cid"],
            "next_offset": 64, "previous_page_cid": self.pages[1]["artifact_cid"]}
        for number in range(1, 5):
            pid = 10000 + number; selected = self.pages[number * 2:number * 2 + 2]
            request = {"schema": "source-successor-fresh-scan-chunk-request@1", "output": str(reader.root), "run_number": number,
                "root_cid": self.root["artifact_cid"], "cursor": deepcopy(cursor), "version_id": self.root["value"]["model"]["version_id"],
                "expected_head": deepcopy(self.root["value"]["head"]), "checkpoint_states": self.frozen,
                "expected_registry_owner_generation": 3 + number, "scheduler_state_path": result["scheduler_state_path"],
                "scheduler_configuration": result["scheduler_configuration"], "timeout_seconds": 900.0, "max_pages": 2,
                "materialization_receipt_sha256": inert.sha(reader.raw("materialized-successor-setup.json")), "helper_sha256": helper_sha}
            reader.put(f"successor-resume-request-{number:02d}.json", request)
            next_cursor = None if number == 4 else {"schema": "codebase-inventory-resume-cursor@1", "root_cid": self.root["artifact_cid"],
                "next_offset": 64 + number * 64, "previous_page_cid": selected[-1]["artifact_cid"]}
            chunk = {"schema": "source-successor-fresh-scan-chunk@1", "qualified": True, "complete": number == 4,
                "pid": pid, "run_number": number, "root_cid": request["root_cid"], "version_id": request["version_id"],
                "request_cursor": cursor, "request_sha256": inert.sha(reader.raw(f"successor-resume-request-{number:02d}.json")),
                "helper_sha256": helper_sha, "materialization_receipt_sha256": request["materialization_receipt_sha256"],
                "scheduler_state_path": request["scheduler_state_path"], "scheduler_configuration": request["scheduler_configuration"],
                "max_pages": 2, "timeout_seconds": 900.0, "post_setup_fit_attempt_count": 0,
                "pages_created": [page["artifact_cid"] for page in selected], "source_execution_attested": False,
                "proof_authority": False, "scan_execution_attested": False, "new_fitting_epochs": 0,
                "fit_guard_scope": "three named owner-process training APIs plus exact frozen checkpoints and native owner rows",
                "registry_owner_generation_before": 3 + number, "registry_owner_generation_after": 3 + number,
                "next_cursor": next_cursor, "prefix_tail_cid": selected[-1]["artifact_cid"], "numerical_before": self.frozen,
                "numerical_after": self.frozen, "source_model_owner_preservation": True,
                "final_resources": {"active_lease_count": 0, "waiting_request_count": 0, "owner_pids": [pid],
                    "scope": "named_scan_process_owners_only", "global_active_lease_count": 1, "global_waiting_request_count": 0}, "recorded_seconds": 1.0}
            if number == 4: chunk.update(completion_cid=self.completion["artifact_cid"], coverage=deepcopy(self.completion["value"]["coverage"]))
            reader.put(f"successor-fresh-process-run-{number:02d}.json", chunk)
            result["fresh_processes"].append(chunk)
            reader.put(f"successor-fresh-process-launch-{number:02d}.json", {"schema": "source-successor-fresh-scan-child-launch@1",
                "run_number": number, "parent_pid": result["pid"], "child_pid": pid, "returncode": 0, "timeout_group_terminated": False,
                "child_working_directory": str(HERE.parent.parent.parent), "recorded_seconds": 1.1,
                "request_sha256": chunk["request_sha256"], "helper_sha256": helper_sha, "scheduler_state_path": request["scheduler_state_path"]})
            for ordinal, page in enumerate(selected, 1):
                reader.put(f"successor-process-{number:02d}-page-{ordinal:02d}.json", page)
                wanted = page["artifact_cid"]
                reader.raws["private/source-artifacts/source/" + wanted[:4] + "/" + wanted] = inert.numerical_wire(page["value"])
            cursor = next_cursor
        result["final_resources"] = {"active_lease_count": 0, "waiting_request_count": 0, "owner_pids": [1234, 10001, 10002, 10003, 10004],
            "scope": "named_scan_process_owners_only", "global_active_lease_count": 1, "global_waiting_request_count": 0}
        return reader, result, helper_sha, helper_path

    def check(self, fixture):
        reader, result, sha, path = fixture
        return audit.verify_processes(reader, result, self.root, self.completion, self.frozen, sha, path)

    def update_chunk(self, fixture, number, **changes):
        reader, result, _, _ = fixture
        result["fresh_processes"][number - 1].update(changes)
        reader.put(f"successor-fresh-process-run-{number:02d}.json", result["fresh_processes"][number - 1])

    def test_four_synthetic_detached_protocol_receipts_positive(self):
        checked = self.check(self.fixture())
        self.assertEqual(checked["fresh_processes"], 4)
        self.assertFalse(checked["process_origin_independently_attested"])

    def test_rehashed_child_receipt_extra_authority(self):
        fixture = self.fixture(); self.update_chunk(fixture, 1, execution_attested=True)
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "fields required"): self.check(fixture)

    def test_rehashed_launch_wrong_child_working_directory(self):
        fixture = self.fixture(); reader = fixture[0]
        value = reader.json("successor-fresh-process-launch-01.json"); value["child_working_directory"] = "/synthetic/wrong-package-root"
        reader.put("successor-fresh-process-launch-01.json", value)
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "child launch"): self.check(fixture)

    def test_rehashed_launch_run_number_boolean_alias(self):
        fixture = self.fixture(); reader = fixture[0]
        value = reader.json("successor-fresh-process-launch-01.json"); value["run_number"] = True
        reader.put("successor-fresh-process-launch-01.json", value)
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "child launch"): self.check(fixture)

    def test_rehashed_child_registry_generation(self):
        fixture = self.fixture(); self.update_chunk(fixture, 3, registry_owner_generation_after=8)
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "frozen owner"): self.check(fixture)

    def test_rehashed_child_fit_guard_scope(self):
        fixture = self.fixture(); self.update_chunk(fixture, 1, fit_guard_scope="all subprocess training independently attested")
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "fit guard observation"): self.check(fixture)

    def test_rehashed_child_completion_early(self):
        fixture = self.fixture(); self.update_chunk(fixture, 2, complete=True)
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "page population/completion"): self.check(fixture)

    def test_child_page_export_does_not_join_complete_chain(self):
        fixture = self.fixture(); reader = fixture[0]
        reader.put("successor-process-01-page-01.json", self.pages[3])
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "page export"): self.check(fixture)


class SelectedProducerControls(unittest.TestCase):
    def fixture(self):
        reader = MemoryReader()
        names = sorted({"ipfs_datasets_py.duckdb_control.autoencoder_registry", "ipfs_datasets_py.duckdb_control.codebase_catalog",
            "ipfs_datasets_py.logic.software_contracts.codebase_resources", "ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler",
            "benchmarks.agent_supervisor.container_coding.source_successor_full_scan_fixture",
            "benchmarks.agent_supervisor.container_coding.qualify_codebase_full_successor"})
        rows = []
        for name in names:
            raw = ((HERE / "source_successor_full_scan_fixture.py").read_bytes() if name.endswith(".source_successor_full_scan_fixture")
                else ("# detached protocol producer " + name + "\n").encode())
            copy = "producers/" + name + ".py"; reader.raws[copy] = raw
            root = "/home/barberb/lift_coding/external/" + ("ipfs_datasets" if name.startswith("ipfs_datasets_py.") else "ipfs_accelerate")
            rows.append({"name": name, "path": root + "/" + name.replace(".", "/") + ".py", "copy": copy,
                "bytes": len(raw), "sha256": inert.sha(raw)})
        return {"schema": "codebase-full-successor-selected-producers@1", "files": rows,
            "scope": "listed_local_files_only", "execution_attestation": False}, reader, {"selected_producers": {rows[0]["name"]: rows[0]["sha256"]}}

    def test_closed_detached_selected_producer_population(self):
        generation, reader, upstream = self.fixture()
        self.assertEqual(len(audit.verify_selected_producers(generation, reader, upstream)), 6)

    def test_coherent_duplicate_selected_producer(self):
        generation, reader, upstream = self.fixture(); generation["files"].append(deepcopy(generation["files"][0]))
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "duplicated/reordered"):
            audit.verify_selected_producers(generation, reader, upstream)

    def test_rehashed_inherited_selected_producer_changes(self):
        generation, reader, upstream = self.fixture(); row = generation["files"][0]
        raw = b"# changed historical producer\n"; reader.raws[row["copy"]] = raw; row.update(bytes=len(raw), sha256=inert.sha(raw))
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "inherited selected source"):
            audit.verify_selected_producers(generation, reader, upstream)

    def test_selected_producer_unknown_authority(self):
        generation, reader, upstream = self.fixture(); generation["execution_verified"] = True
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "fields required"):
            audit.verify_selected_producers(generation, reader, upstream)

    def test_rehashed_selected_producer_locator_escape(self):
        generation, reader, upstream = self.fixture(); generation["files"][0]["copy"] = "../producers/escape.py"
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "locator"):
            audit.verify_selected_producers(generation, reader, upstream)

    def test_selected_producer_coherent_copy_has_wrong_native_module_path(self):
        generation, reader, upstream = self.fixture(); generation["files"][0]["path"] = "/synthetic/alias/main.py"
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "exact workspace module path"):
            audit.verify_selected_producers(generation, reader, upstream)

    def test_coherent_final_helper_copy_does_not_match_executed_guard_generation(self):
        generation, reader, upstream = self.fixture()
        row = next(row for row in generation["files"] if row["name"].endswith(".source_successor_full_scan_fixture"))
        raw = reader.raw(row["copy"]) + b"# unrelated generation\n"
        reader.raws[row["copy"]] = raw; row.update(bytes=len(raw), sha256=inert.sha(raw))
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "final helper guard generation"):
            audit.verify_selected_producers(generation, reader, upstream)


class NativeCostControls(unittest.TestCase):
    def fixture(self):
        result = dict.fromkeys(audit.RESULT_FIELDS)
        result.update(schema="codebase-full-successor-native-qualification@1", qualified=True, complete_scan_qualified=True,
            scope="complete_300_member_successor_cpu8d_scan_and_first32_opt_out_comparison", new_fitting_epochs=0,
            inherited_setup_epochs=2, inherited_scan_pages=0, new_default_scan_pages=10, new_reference_scan_pages=1,
            new_scan_pages_created=11, post_setup_fit_attempt_count=0, inference_attempts_outside_pages=0,
            fresh_process_count=4, owner_open_count=6, cpu_scan_cuda_visible_devices="", recorded_seconds=100.0,
            elapsed_seconds_so_far=99.0, phases=[{"name": "synthetic_protocol", "status": "completed", "elapsed_seconds": 98.0}],
            controls=[{"name": name, "refused": True, "error_type": "SyntheticRefusal", "error": "detached protocol example", "elapsed_seconds": 0.1}
                      for name in ("incomplete_prefix_is_unknown", "precancelled_complete_receiving")],
            operation_deadline_seconds={"default_selection": 120, "reference_selection": 600, "inference_page": 600,
                "completion_receiving": 600, "fresh_chunk": 900, "overall": 4200, "reference_override_is_qualification_only": True})
        for field in ("worker_dispatch_qualified", "cuda_qualified", "384d_qualified", "proof_authority", "source_execution_attested",
            "scan_execution_attested", "production_default_activated", "repository_code_executed", "kernel_resource_enforcement_claimed",
            "numerical_page_reuse", "model_head_promoted"): result[field] = False
        for field in ("cold_receiving_verified", "original_source_artifacts_preserved", "selected_producers_unchanged",
            "native_owner_unchanged_except_copy_path_relocation_and_six_opens", "closed_source_archive_preserved_after_job"): result[field] = True
        return result

    def test_detached_complete_native_accounting_protocol(self):
        self.assertEqual(audit.verify_costs(self.fixture())["new_scan_pages_created"], 11)

    def test_extra_native_execution_authority(self):
        result = self.fixture(); result["independently_executed"] = True
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "fields required"): audit.verify_costs(result)


    def test_zero_native_fit_counter_boolean_alias(self):
        result = self.fixture(); result["new_fitting_epochs"] = False
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "accounting"): audit.verify_costs(result)

    def test_native_inherited_pages_relabelled_fresh(self):
        result = self.fixture(); result["inherited_scan_pages"] = 2
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "accounting"): audit.verify_costs(result)

    def test_native_refusal_extra_unverified_scope(self):
        result = self.fixture(); result["controls"][0]["proof_verified"] = True
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "fields required"): audit.verify_costs(result)


class CopiedGitIndexControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_reader = inert.Reader(SOURCE, seconds=120)
        cls.full_reader = inert.Reader(FULL_SOURCE, seconds=120)
        cls.before = cls.full_reader.whole_archive()
        cls.original = cls.old_reader.raw("repository/.git/index", 2 * inert.MIB)
        cls.current = cls.full_reader.raw("repository/.git/index", 2 * inert.MIB)

    @classmethod
    def tearDownClass(cls):
        if not inert.same(cls.before, inert.Reader(FULL_SOURCE, seconds=120).whole_archive()):
            raise AssertionError("closed complete native fullscan archive changed during pure Git index controls")

    def forged(self, start, value):
        body = bytearray(self.current[:-20]); body[start:start + len(value)] = value
        raw = bytes(body)
        return raw + audit.hashlib.sha1(raw).digest()

    def test_actual_300_entry_copied_index_stat_cache_refresh(self):
        value = audit.verify_copied_git_index(self.original, self.current)
        self.assertEqual(value["changed_stat_cache_entries"], 300)
        self.assertEqual(value["allowed_fields"], ["ctime_ns", "ctime_s", "ino"])
        self.assertTrue(value["tracked_oid_mode_flags_paths_extensions_preserved"])
        self.assertFalse(value["git_executed"])

    def test_unchanged_index_is_also_valid(self):
        self.assertEqual(audit.verify_copied_git_index(self.original, self.original)["changed_stat_cache_entries"], 0)

    def test_index_checksum_failure(self):
        current = bytearray(self.current); current[50] ^= 1
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "header/checksum"):
            audit.verify_copied_git_index(self.original, bytes(current))

    def test_rehashed_index_object_id_change(self):
        current = self.forged(52, bytes([self.current[52] ^ 1]))
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "tracked identities"):
            audit.verify_copied_git_index(self.original, current)

    def test_rehashed_index_permission_mode_change(self):
        current = self.forged(36, struct.pack(">I", 0o100755))
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "tracked identities"):
            audit.verify_copied_git_index(self.original, current)

    def test_rehashed_index_mtime_change_is_not_permitted(self):
        current = self.forged(20, struct.pack(">I", struct.unpack(">I", self.current[20:24])[0] + 1))
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "tracked identities"):
            audit.verify_copied_git_index(self.original, current)

    def test_rehashed_index_tree_extension_change(self):
        current = self.forged(len(self.current) - 21, bytes([self.current[-21] ^ 1]))
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "tracked identities"):
            audit.verify_copied_git_index(self.original, current)

    def test_rehashed_index_flags_gain_assume_valid(self):
        flags = struct.unpack(">H", self.current[72:74])[0] | 0x8000
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "ordinary stage-zero"):
            audit.verify_copied_git_index(self.original, self.forged(72, struct.pack(">H", flags)))

    def test_rehashed_index_unsupported_version(self):
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "header/checksum"):
            audit.verify_copied_git_index(self.original, self.forged(4, struct.pack(">I", 3)))

    def test_rehashed_index_missing_entry_count(self):
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "header/checksum"):
            audit.verify_copied_git_index(self.original, self.forged(8, struct.pack(">I", 299)))

    def test_index_bytearray_alias_rejected(self):
        with self.assertRaisesRegex(inert.SourceSuccessorAuditError, "exact Git index bytes"):
            audit.verify_copied_git_index(self.original, bytearray(self.current))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-directory", type=Path, required=True)
    args = parser.parse_args(); output = args.receipt_directory.absolute(); output.mkdir(parents=True, exist_ok=False)
    before_reader, before_test = READER_PATH.read_bytes(), Path(__file__).read_bytes()
    (output / "reader.py").write_bytes(before_reader); (output / "test.py").write_bytes(before_test)
    log = io.StringIO(); started = time.monotonic()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    unchanged = READER_PATH.read_bytes() == before_reader and Path(__file__).read_bytes() == before_test
    receipt = {"schema": "full-successor-independent-reader-guard-tests@1", "qualified": result.wasSuccessful() and unchanged and not result.skipped,
        "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "recorded_seconds": time.monotonic() - started, "reader": {"path": str(READER_PATH), "bytes": len(before_reader), "sha256": inert.sha(before_reader)},
        "test": {"path": str(Path(__file__).resolve()), "bytes": len(before_test), "sha256": inert.sha(before_test)},
        "current_pins_unchanged": unchanged, "source_namespace": str(SOURCE), "native_owners_opened": False,
        "sql_executed": False, "git_executed": False, "new_fitting_epochs": 0, "new_inference_pages": 0,
        "scope": "actual inert checkpoint/source positives, coherent report rehashes, synthetic complete-page/process custody, exact closed-copy Git v2 stat-cache and scoped resource guards; no native execution attestation"}
    full_result_raw = inert.Reader(FULL_SOURCE, seconds=120).raw("result.json", 4 * inert.MIB)
    receipt["full_source_namespace"] = str(FULL_SOURCE)
    receipt["source_native_result"] = {"path": str(FULL_SOURCE / "result.json"), "bytes": len(full_result_raw), "sha256": inert.sha(full_result_raw)}
    (output / "tests.log").write_text(log.getvalue()); (output / "receipt.json").write_bytes(inert.numerical_wire(receipt) + b"\n")
    for path in output.iterdir(): path.chmod(0o444)
    print(json.dumps({key: receipt[key] for key in ("qualified", "tests", "failures", "errors", "skipped", "recorded_seconds", "current_pins_unchanged")}, sort_keys=True))
    return 0 if receipt["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
