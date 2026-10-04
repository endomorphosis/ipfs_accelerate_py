"""Stdlib receiving controls over inert, closed source-successor exports.

This file deliberately imports only the standalone reader and the standard
library. It never creates native owners, opens SQL, runs Git, or fits a model.
All negative examples are in memory or a temporary directory. Rehashed
examples exercise semantic joins after their enclosing byte identities agree.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
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


HERE = Path(__file__).resolve().parent
READER_PATH = HERE / "audit_codebase_source_successor.py"
spec = importlib.util.spec_from_file_location("source_successor_standalone_guard_reader", READER_PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
DEFAULT_ARCHIVE = Path("/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/source-successor-qualification-20261003-02")
ARCHIVE = DEFAULT_ARCHIVE
CONTROL_IDS = []


def envelope(value, *, numerical=False):
    return {"artifact_cid": audit.cid(audit.numerical_wire(value), source=True)
            if numerical else audit.structured(value), "value": value}


def rehash_planning(result, preimages):
    """Rebuild every consumed material, snapshot, receipt and outer identity."""
    for entry in preimages["entries"]:
        entry["binding"]["field_digests"] = {
            name: "sha256:" + audit.sha(b"plan-create-semantic-input\n" + name.encode() + b"\n" + audit.numerical_wire(value))
            for name, value in entry["preimages"].items()
        }
    preimages["entries"].sort(key=lambda entry: audit.sha(audit.numerical_wire(entry["binding"])))
    final = next(entry for entry in preimages["entries"] if entry["preimages"]["scan"] is not None)
    native = result["repository_preview"]
    snapshot = native["input_snapshot"]
    snapshot["material_binding"] = deepcopy(final["binding"])
    snapshot["snapshot_cid"] = audit.named_digest("plan-create-input-snapshot", {k: v for k, v in snapshot.items() if k != "snapshot_cid"})
    receipt = native["preview"]
    receipt["input_snapshot_cid"] = snapshot["snapshot_cid"]
    receipt["receipt_cid"] = audit.named_digest("plan-create-preview-receipt", {k: v for k, v in receipt.items() if k != "receipt_cid"})
    result["result_cid"] = audit.structured({k: v for k, v in result.items() if k != "result_cid"})


def rehash_result(result):
    native = result["repository_preview"]
    snapshot = native["input_snapshot"]
    snapshot["snapshot_cid"] = audit.named_digest("plan-create-input-snapshot", {k: v for k, v in snapshot.items() if k != "snapshot_cid"})
    receipt = native["preview"]
    receipt["input_snapshot_cid"] = snapshot["snapshot_cid"]
    for stage in receipt["stage_results"]:
        stage["result_cid"] = audit.named_digest("plan-create-stage-result", {k: v for k, v in stage.items() if k != "result_cid"})
    receipt["receipt_cid"] = audit.named_digest("plan-create-preview-receipt", {k: v for k, v in receipt.items() if k != "receipt_cid"})
    result["result_cid"] = audit.structured({k: v for k, v in result.items() if k != "result_cid"})


class OverlayReader:
    """Read an in-memory negative example; fall back only to inert bytes."""
    def __init__(self, overrides):
        self.overrides = overrides
        self.delegate = audit.Reader(ARCHIVE, seconds=120)

    def raw(self, locator, maximum=8 * audit.MIB):
        if locator in self.overrides:
            raw = self.overrides[locator]
            audit.need(type(raw) is bytes and len(raw) <= maximum, "bounded overlay bytes required")
            return raw
        return self.delegate.raw(locator, maximum)

    def json(self, locator, maximum=8 * audit.MIB):
        return audit.parse(self.raw(locator, maximum))


class ExportControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = audit.Reader(ARCHIVE, seconds=120)
        cls.archive_before = cls.reader.whole_archive()
        cls.fixtures = {name: cls.reader.json(name, 16 * audit.MIB) for name in (
            "successor-selection.json", "scan-root.json", "source-delta.json", "prefix-page.json",
            "planning-preview.json", "cold-planning-preview.json", "authored-planning-inputs.json",
            "planning-semantic-preimages.json", "cold-planning-semantic-preimages.json",
            "current-manifest.json", "previous-manifest.json", "current-publication-receipt.json",
            "previous-publication-receipt.json", "root-training-record.json", "child-training-record.json",
            "checkpoint-states-after.json", "owners-before.json", "result.json",
        )}
        selection = cls.fixtures["successor-selection.json"]["value"]
        cls.raw_models = {}
        for label, descriptor in (("root", selection["previous_model"]), ("child", selection["model"])):
            digest = descriptor["artifact"]["sha256"]
            cls.raw_models[label] = cls.reader.raw("private/model-artifacts/" + digest[:2] + "/" + digest, 16 * audit.MIB)

    @classmethod
    def tearDownClass(cls):
        after = audit.Reader(ARCHIVE, seconds=120).whole_archive()
        if not audit.same(cls.archive_before, after):
            raise AssertionError("closed native archive changed during stdlib controls")

    def fixture(self, name):
        return deepcopy(self.fixtures[name])

    def denied(self, function, *args, message=None, **kwargs):
        with self.assertRaises(audit.SourceSuccessorAuditError) as caught:
            function(*args, **kwargs)
        if message is not None:
            self.assertRegex(str(caught.exception), message)

    def planning(self, *, cold=False):
        return (self.fixture("cold-planning-preview.json" if cold else "planning-preview.json"),
                self.fixture("authored-planning-inputs.json"),
                self.fixture("cold-planning-semantic-preimages.json" if cold else "planning-semantic-preimages.json"))

    def check_plan(self, result, authored, preimages):
        return audit.verify_planning(result, authored, preimages, self.fixtures["source-delta.json"], self.fixtures["current-manifest.json"]["value"])

    def altered_child(self, mutator):
        selection = self.fixture("successor-selection.json")
        root_record = self.fixture("root-training-record.json")
        child_record = self.fixture("child-training-record.json")
        states = self.fixture("checkpoint-states-after.json")
        model = selection["value"]["model"]
        saved = audit.parse(self.raw_models["child"])
        mutator(saved)
        provenance = saved["report"]["codebase_provenance"]
        saved["report"]["training_targets_sha256"] = audit.sha(audit.numerical_wire(provenance["training_targets"]))
        saved["report"]["tuning_targets_sha256"] = saved["state"]["tuning_targets_sha256"] = audit.sha(audit.numerical_wire(provenance["tuning_targets"]))
        saved["report"]["codebase_request_sha256"] = audit.sha(audit.numerical_wire(audit.training_request(saved)))
        raw = audit.numerical_wire(saved)
        model["artifact"] = {"sha256": audit.sha(raw), "bytes": len(raw)}
        model["artifact_cid"] = audit.cid(raw, source=True)
        model["state_sha256"] = audit.sha(audit.numerical_wire(saved["state"]))
        model["feature_space_sha256"] = audit.sha(audit.numerical_wire(saved["feature_space"]))
        model["ancestry"][0]["artifact"] = deepcopy(model["artifact"])
        record = child_record["value"]
        record["registry_artifact"] = deepcopy(model["artifact"])
        record["checkpoint_raw_cid"] = model["artifact_cid"]
        for field in ("state_sha256", "feature_space_sha256"):
            record[field] = model[field]
        record["report_json"] = audit.numerical_wire(saved["report"]).decode()
        child_record = envelope(record)
        selection["value"]["training_record_cid"] = child_record["artifact_cid"]
        selection = envelope(selection["value"])
        checked = audit.verify_model(model, raw, selection["value"]["current_head"])
        audit.verify_training_record(child_record, model, checked["saved"], selection["value"]["current_head"], selection["value"]["previous_model"]["version_id"])
        states["child"] = checked["checkpoint_state"]
        digest = model["artifact"]["sha256"]
        overrides = {"private/model-artifacts/" + digest[:2] + "/" + digest: raw}
        return OverlayReader(overrides), selection, root_record, child_record, states

    def test_closed_native_positive_source_models_selection_root_page_and_costs(self):
        f = self.fixtures
        audit.verify_source_delta(f["source-delta.json"], f["previous-manifest.json"]["value"], f["current-manifest.json"]["value"],
                                  f["previous-publication-receipt.json"], f["current-publication-receipt.json"], get_artifact=self.reader.cas)
        models = audit.audit_models(self.reader, f["successor-selection.json"], f["root-training-record.json"], f["child-training-record.json"], f["checkpoint-states-after.json"])
        self.assertEqual(models["child"]["checkpoint_state"]["completed_epochs"], 2)
        self.assertTrue(audit.verify_registry_lineage(f["owners-before.json"]["registry"], f["successor-selection.json"], models)["complete_version_identities_verified"])
        audit.verify_selection(f["successor-selection.json"], f["source-delta.json"], f["scan-root.json"], f["successor-selection.json"]["value"]["previous_model"])
        self.assertEqual(len(audit.verify_root(f["scan-root.json"], f["source-delta.json"], f["successor-selection.json"]["value"]["model"])), 300)
        self.assertFalse(audit.verify_prefix_page(f["prefix-page.json"], f["scan-root.json"])["complete_scan_qualified"])
        self.assertEqual(audit.verify_costs(f["result.json"])["new_fitting_epochs"], 2)

    def test_positive_warm_and_cold_full_semantic_preimages(self):
        for cold in (False, True):
            with self.subTest(cold=cold):
                checked = self.check_plan(*self.planning(cold=cold))
                self.assertTrue(checked["complete_semantic_preimages_verified"])
                self.assertEqual(checked["observed_facts"], 0)
                self.assertEqual(len(checked["tasks"]), 2)

    def test_model_altered_weight_bytes_without_descriptor_change(self):
        model = self.fixtures["successor-selection.json"]["value"]["model"]
        raw = bytearray(self.raw_models["child"]); raw[len(raw) // 2] ^= 1
        self.denied(audit.verify_model, model, bytes(raw), self.fixtures["successor-selection.json"]["value"]["current_head"], message="full selected model byte")

    def test_model_raw_cid_codec_is_not_structured(self):
        model = deepcopy(self.fixtures["successor-selection.json"]["value"]["model"])
        model["artifact_cid"] = audit.cid(self.raw_models["child"])
        self.denied(audit.verify_model, model, self.raw_models["child"], self.fixtures["successor-selection.json"]["value"]["current_head"])

    def test_training_record_rehashed_parent_join(self):
        model = self.fixtures["successor-selection.json"]["value"]["model"]
        record = self.fixture("child-training-record.json")["value"]; record["parent_version_id"] = "sha256:" + "0" * 64
        self.denied(audit.verify_training_record, envelope(record), model, audit.parse(self.raw_models["child"]),
                    self.fixtures["successor-selection.json"]["value"]["current_head"], self.fixtures["successor-selection.json"]["value"]["previous_model"]["version_id"])

    def test_preimage_digest_is_checked(self):
        result, authored, preimages = self.planning()
        preimages["entries"][0]["preimages"]["extra"]["$mapping"]["unhashed"] = "changed"
        self.denied(self.check_plan, result, authored, preimages, message="preimage digest")

    def test_preimages_duplicate_variant_identity(self):
        preimages = self.fixture("planning-semantic-preimages.json")
        preimages["entries"][1] = deepcopy(preimages["entries"][0])
        self.denied(audit.verify_preimages, preimages, message="duplicated or reordered")

    def test_preimages_variant_order(self):
        preimages = self.fixture("planning-semantic-preimages.json")
        preimages["entries"].reverse()
        self.denied(audit.verify_preimages, preimages, message="duplicated or reordered")

    def test_authored_legacy_frozen_goal_digest(self):
        result, authored, preimages = self.planning()
        authored["materials"]["frozen_goal_cid"] = "sha256:" + "0" * 64
        self.denied(self.check_plan, result, authored, preimages, message="frozen|metadata")

    def test_material_meaning_only_one_variant(self):
        result, authored, preimages = self.planning()
        preimages["entries"][0]["preimages"]["task_candidates"]["items"][0]["fields"]["$mapping"]["producer_id"] = "producer:other"
        rehash_planning(result, preimages)
        self.denied(self.check_plan, result, authored, preimages, message="meaning changed")

    def test_planning_authority_bool_alias_rehashed(self):
        result, authored, preimages = self.planning()
        result["authority"]["proof_authority"] = 0; rehash_result(result)
        self.denied(self.check_plan, result, authored, preimages, message="exactly false")

    def test_planning_counter_bool_alias_rehashed(self):
        result, authored, preimages = self.planning()
        result["inference_calls"] = False; rehash_result(result)
        self.denied(self.check_plan, result, authored, preimages, message="non-numerical")

    def test_native_preview_counter_bool_alias_rehashed(self):
        result, authored, preimages = self.planning()
        result["repository_preview"]["model_calls"] = False; rehash_result(result)
        self.denied(self.check_plan, result, authored, preimages, message="model-off")

    def test_native_stage_bool_alias_rehashed(self):
        result, authored, preimages = self.planning()
        result["repository_preview"]["preview"]["stage_results"][0]["passed"] = 1; rehash_result(result)
        self.denied(self.check_plan, result, authored, preimages, message="stage result")

    def test_residual_task_population_is_complete(self):
        result, authored, preimages = self.planning()
        result["residual_task_ids"].pop(); rehash_result(result)
        self.denied(self.check_plan, result, authored, preimages, message="residual population")

    def test_snapshot_request_scope_rehashed(self):
        result, authored, preimages = self.planning()
        result["repository_preview"]["input_snapshot"]["scope_paths"] = ["other.py"]; rehash_result(result)
        self.denied(self.check_plan, result, authored, preimages, message="snapshot/request")

    def test_injected_body_free_refs_must_be_exact(self):
        result, authored, preimages = self.planning()
        for entry in preimages["entries"]:
            refs = entry["preimages"]["extra"]["$mapping"].get("codebase_source_delta_advisory")
            if refs is not None:
                refs["$mapping"]["coverage"]["$mapping"]["current_members"] = -1
        rehash_planning(result, preimages)
        self.denied(self.check_plan, result, authored, preimages, message="source-bound material")

    def test_model_tensor_bool_alias_with_rehashed_full_bytes(self):
        selection = self.fixtures["successor-selection.json"]["value"]
        saved = audit.parse(self.raw_models["child"])
        # Find the actual native tensor, avoiding assumptions about layer names.
        saved["state"]["parameters"][0][0][0] = False
        raw = audit.numerical_wire(saved)
        model = deepcopy(selection["model"])
        model["artifact"] = {"sha256": audit.sha(raw), "bytes": len(raw)}
        model["artifact_cid"] = audit.cid(raw, source=True)
        model["state_sha256"] = audit.sha(audit.numerical_wire(saved["state"]))
        model["ancestry"][0]["artifact"] = deepcopy(model["artifact"])
        self.denied(audit.verify_model, model, raw, selection["current_head"], message="tensor|finite|numerical|alias")

    def test_registry_native_version_content_identity(self):
        selected = self.fixtures["successor-selection.json"]
        models = {label: audit.verify_model(selected["value"][field], self.raw_models[label], selected["value"][head])
                  for label, field, head in (("root", "previous_model", "previous_head"), ("child", "model", "current_head"))}
        registry = self.fixture("owners-before.json")["registry"]
        registry["versions"][1][4] = registry["versions"][1][4].replace('"attempt":1', '"attempt":2')
        self.denied(audit.verify_registry_lineage, registry, selected, models, message="version content identity")

    def test_registry_rehashed_version_metadata_model_join(self):
        selected = self.fixture("successor-selection.json")
        models = {label: audit.verify_model(selected["value"][field], self.raw_models[label], selected["value"][head])
                  for label, field, head in (("root", "previous_model", "previous_head"), ("child", "model", "current_head"))}
        registry = self.fixture("owners-before.json")["registry"]
        row = next(row for row in registry["versions"] if row[0] == selected["value"]["model"]["version_id"])
        metadata = audit.parse(row[4]); metadata["result"]["report_sha256"] = "0" * 64
        row[4] = audit.numerical_wire(metadata).decode()
        row[0] = "sha256:" + audit.sha(audit.wire({"schema": "ipfs_datasets_py/autoencoder-control@1", "variant_id": row[1], "artifact": audit.parse(row[3]), "metadata": metadata, "parent_version_id": row[2]}))
        selected["value"]["model"]["version_id"] = row[0]
        selected["value"]["model"]["ancestry"][0]["version_id"] = row[0]
        self.denied(audit.verify_registry_lineage, registry, envelope(selected["value"]), models, message="full model result")

    def test_registry_completed_run_training_request_meaning(self):
        selected = self.fixtures["successor-selection.json"]
        models = {label: audit.verify_model(selected["value"][field], self.raw_models[label], selected["value"][head])
                  for label, field, head in (("root", "previous_model", "previous_head"), ("child", "model", "current_head"))}
        registry = self.fixture("owners-before.json")["registry"]
        request = audit.parse(registry["runs"][0][3]); request["selections"][0]["path"] = "other.py"
        registry["runs"][0][3] = audit.numerical_wire(request).decode()
        self.denied(audit.verify_registry_lineage, registry, selected, models, message="child request/result")

    def test_registry_historical_lease_attempt_exact_integer(self):
        selected = self.fixtures["successor-selection.json"]
        models = {label: audit.verify_model(selected["value"][field], self.raw_models[label], selected["value"][head])
                  for label, field, head in (("root", "previous_model", "previous_head"), ("child", "model", "current_head"))}
        registry = self.fixture("owners-before.json")["registry"]
        lease = audit.parse(registry["runs"][0][7]); lease["attempt"] = True
        registry["runs"][0][7] = audit.numerical_wire(lease).decode()
        self.denied(audit.verify_registry_lineage, registry, selected, models, message="attempt/fence")

    def test_registry_durable_completion_command_digest(self):
        selected = self.fixtures["successor-selection.json"]
        models = {label: audit.verify_model(selected["value"][field], self.raw_models[label], selected["value"][head])
                  for label, field, head in (("root", "previous_model", "previous_head"), ("child", "model", "current_head"))}
        registry = self.fixture("owners-before.json")["registry"]
        row = next(row for row in registry["operations"] if audit.parse(row[2])["command"] == "CompleteRun")
        row[1] = "0" * 64
        self.denied(audit.verify_registry_lineage, registry, selected, models, message="completion command digest")

    def test_registry_durable_completion_event_full_version(self):
        selected = self.fixtures["successor-selection.json"]
        models = {label: audit.verify_model(selected["value"][field], self.raw_models[label], selected["value"][head])
                  for label, field, head in (("root", "previous_model", "previous_head"), ("child", "model", "current_head"))}
        registry = self.fixture("owners-before.json")["registry"]
        row = next(row for row in registry["events"] if row[1] == "candidate_durable")
        payload = audit.parse(row[2]); payload["version_id"] = selected["value"]["previous_model"]["version_id"]
        row[2] = audit.numerical_wire(payload).decode()
        self.denied(audit.verify_registry_lineage, registry, selected, models, message="event identity")


def add_export_case(name, function):
    CONTROL_IDS.append(name)
    setattr(ExportControls, "test_" + name, function)


def lineage_case(mutator):
    def control(self):
        args = self.altered_child(mutator)
        self.denied(audit.audit_models, *args, message="parent|frozen|ancestr|cohort|selection|base.state|continuation|history")
    return control


LINEAGE_MUTATIONS = {
    "model_rehashed_wrong_adam_parent_base_state": lambda saved: saved["report"].__setitem__("base_state_sha256", "0" * 64),
    "model_rehashed_changed_frozen_tuning_cohort": lambda saved: saved["report"]["codebase_provenance"]["tuning_targets"].__setitem__(0, deepcopy(saved["report"]["codebase_provenance"]["canary_targets"][0])),
    "model_rehashed_changed_frozen_canary_cohort": lambda saved: saved["report"]["codebase_provenance"]["canary_targets"].__setitem__(0, deepcopy(saved["report"]["codebase_provenance"]["tuning_targets"][0])),
    "model_rehashed_changed_parent_replay_cohort": lambda saved: saved["report"]["codebase_provenance"]["replay_targets"].__setitem__(0, deepcopy(saved["report"]["codebase_provenance"]["canary_targets"][0])),
    "model_rehashed_changed_training_selection": lambda saved: saved["report"]["codebase_provenance"]["selections"][0].__setitem__("path", "other.py"),
    "model_rehashed_changed_ancestral_training_history": lambda saved: saved["report"]["codebase_provenance"]["ancestral_training"][0].__setitem__("source_digest", "0" * 64),
}
for _name, _mutator in LINEAGE_MUTATIONS.items():
    add_export_case(_name, lineage_case(_mutator))


def semantic_case(field, mutator):
    def control(self):
        result, authored, preimages = self.planning()
        for entry in preimages["entries"]:
            mutator(entry["preimages"][field])
        rehash_planning(result, preimages)
        # All field digests and native enclosing identities now agree. The
        # complete authored meaning/type gate must still reject this example.
        self.denied(self.check_plan, result, authored, preimages, message="authored|typed|native|enum|record|tuple|meaning|goal|preimage")
    return control


def goal_fields(intent):
    return intent["fields"]["$mapping"]["desired_predicates"]["items"][0]["fields"]["$mapping"]


SEMANTIC_MUTATIONS = {
    "meaning_same_goal_id_changed_predicate_type": ("intent", lambda intent: goal_fields(intent).__setitem__("predicate_type", "arbitrary_other_meaning")),
    "meaning_same_goal_id_changed_object": ("intent", lambda intent: goal_fields(intent).__setitem__("object_ref", "requirement:other")),
    "meaning_same_goal_id_changed_subject": ("intent", lambda intent: goal_fields(intent).__setitem__("subject_ref", "other.py")),
    "meaning_same_goal_id_changed_support": ("intent", lambda intent: goal_fields(intent)["support"].__setitem__("value", "supported")),
    "meaning_same_goal_id_changed_polarity": ("intent", lambda intent: goal_fields(intent)["polarity"].__setitem__("value", "negative")),
    "meaning_same_goal_id_wrong_predicate_record_type": ("intent", lambda intent: intent["fields"]["$mapping"]["desired_predicates"]["items"][0].__setitem__("$record", "foreign.module.TypedPredicate")),
    "meaning_same_goal_id_wrong_enum_type": ("intent", lambda intent: goal_fields(intent)["support"].__setitem__("$enum", "foreign.module.SemanticSupport")),
    "meaning_same_task_id_wrong_task_record_type": ("task_candidates", lambda tasks: tasks["items"][0].__setitem__("$record", "foreign.module.TaskCandidate")),
    "meaning_same_producer_id_wrong_record_type": ("producers", lambda producers: producers["items"][0].__setitem__("$record", "foreign.module.ProducerRule")),
    "meaning_same_producer_id_changed_effect": ("producers", lambda producers: producers["items"][0]["fields"]["$mapping"]["effect_predicate_ids"].__setitem__("items", ["goal:runtime:1"])),
    "meaning_same_task_id_changed_closed_obligation": ("task_candidates", lambda tasks: tasks["items"][0]["fields"]["$mapping"]["closes_obligation_ids"].__setitem__("items", ["obligation:other"])),
    "meaning_same_task_id_changed_producer": ("task_candidates", lambda tasks: tasks["items"][0]["fields"]["$mapping"].__setitem__("producer_id", "producer:runtime:1")),
    "meaning_tuple_sequence_replaced_by_list": ("task_candidates", lambda tasks: tasks.__setitem__("$sequence", "list")),
    "meaning_frozen_goal_wrong_native_record": ("frozen_goal", lambda goal: goal.__setitem__("$record", "foreign.module.FrozenPlanningGoal")),
}
for _name, (_field, _mutator) in SEMANTIC_MUTATIONS.items():
    add_export_case(_name, semantic_case(_field, _mutator))


def selection_case(mutator):
    def control(self):
        selected = self.fixture("successor-selection.json")["value"]
        mutator(selected)
        self.denied(audit.verify_selection, envelope(selected), self.fixtures["source-delta.json"], self.fixtures["scan-root.json"], self.fixtures["successor-selection.json"]["value"]["previous_model"])
    return control


for _name, _mutator in {
    "selection_rehashed_previous_head_generation": lambda selected: selected["previous_head"].__setitem__("generation", selected["previous_head"]["generation"] + 1),
    "selection_rehashed_numerical_reuse_bool_alias": lambda selected: selected.__setitem__("numerical_reuse", 0),
    "selection_rehashed_non_boolean_optimized": lambda selected: selected.__setitem__("optimized", 1),
    "selection_rehashed_root_cid": lambda selected: selected.__setitem__("root_cid", audit.structured({"other": True})),
}.items():
    add_export_case(_name, selection_case(_mutator))


def root_case(mutator):
    def control(self):
        root = self.fixture("scan-root.json")["value"]; mutator(root)
        self.denied(audit.verify_root, envelope(root), self.fixtures["source-delta.json"], self.fixtures["successor-selection.json"]["value"]["model"])
    return control


for _name, _mutator in {
    "root_rehashed_incomplete_membership": lambda root: root["members"].pop(),
    "root_rehashed_reordered_membership": lambda root: root["members"].reverse(),
    "root_rehashed_unreviewed_scan_limit": lambda root: root["limits"].__setitem__("unreviewed", 1),
    "root_rehashed_limit_bool_alias": lambda root: root["limits"].__setitem__("page_entries", True),
}.items():
    add_export_case(_name, root_case(_mutator))


def page_case(mutator, *, rehash=True):
    def control(self):
        page = self.fixture("prefix-page.json"); mutator(page["value"])
        if rehash:
            page = envelope(page["value"], numerical=True)
        self.denied(audit.verify_prefix_page, page, self.fixtures["scan-root.json"])
    return control


for _name, _mutator in {
    "page_rehashed_ordered_entry_swap": lambda page: page["entries"].reverse(),
    "page_rehashed_complete_entry_population": lambda page: page["entries"].pop(),
    "page_rehashed_index_bool_alias": lambda page: page["entries"][0].__setitem__("member_index", False),
    "page_rehashed_start_bool_alias": lambda page: page.__setitem__("start", False),
    "page_rehashed_latent_bool_alias": lambda page: page["inference"]["rows"][0]["latent"].__setitem__(0, False),
    "page_rehashed_short_latent_layout": lambda page: page["inference"]["rows"][0]["latent"].pop(),
    "page_rehashed_worker_returncode_bool_alias": lambda page: page["worker_receipt"].__setitem__("returncode", False),
    "page_rehashed_worker_unbounded_output": lambda page: page["worker_receipt"].__setitem__("output_bytes", page["worker_receipt"]["limits"]["max_output_bytes"] + 1),
    "page_rehashed_inference_authority_alias": lambda page: page["inference"].__setitem__("qualified", 0),
    "page_rehashed_inferred_row_coverage": lambda page: page["coverage"].__setitem__("inferred_rows", 0),
}.items():
    add_export_case(_name, page_case(_mutator))


def cost_case(mutator):
    def control(self):
        result = self.fixture("result.json"); mutator(result)
        self.denied(audit.verify_costs, result)
    return control


for _name, _mutator in {
    "cost_inference_zero_bool_alias": lambda result: result.__setitem__("inference_attempts_during_selection_or_cold_receiving", False),
    "cost_fit_zero_bool_alias": lambda result: result.__setitem__("training_attempts_after_setup", False),
    "cost_unreported_qualified_deadlines": lambda result: result.pop("operation_deadline_seconds"),
    "cost_epoch_bool_alias": lambda result: result["setup_training_attempts"][0].__setitem__("actual_completed_epochs", True),
    "cost_nonfinite_actual_elapsed": lambda result: result.__setitem__("recorded_seconds", math.inf),
    "cost_reference_deadline_changed": lambda result: result["operation_deadline_seconds"].__setitem__("reference_selection", 120),
}.items():
    add_export_case(_name, cost_case(_mutator))


class JsonIdentityControls(unittest.TestCase):
    def test_duplicate_json_fields_rejected(self):
        with self.assertRaises(audit.SourceSuccessorAuditError):
            audit.parse(b'{"state":1,"state":2}')

    def test_nonfinite_json_constants_rejected(self):
        for literal in (b"NaN", b"Infinity", b"-Infinity"):
            with self.subTest(literal=literal), self.assertRaises(audit.SourceSuccessorAuditError):
                audit.parse(b'{"value":' + literal + b'}')

    def test_structured_cid_rejects_numerical_float(self):
        with self.assertRaises(audit.SourceSuccessorAuditError):
            audit.structured({"weights": [1.25]})

    def test_bool_int_are_distinct_semantic_observations(self):
        self.assertFalse(audit.observation_same({"calls": False}, {"calls": 0}))

    def test_structured_and_raw_cid_profiles_are_distinct(self):
        raw = b"retained bytes"
        self.assertNotEqual(audit.cid(raw), audit.cid(raw, source=True))
        with self.assertRaises(audit.SourceSuccessorAuditError):
            audit.check_cid(audit.cid(raw), source=True)

    def test_closed_tag_rejects_extra_fields(self):
        with self.assertRaises(audit.SourceSuccessorAuditError):
            audit.decode_preimage({"$mapping": {}, "ignored": True})

    def test_exact_sequence_tag_is_required(self):
        with self.assertRaises(audit.SourceSuccessorAuditError):
            audit.decode_preimage({"$sequence": "set", "items": []})

    def test_depth_is_bounded(self):
        value = None
        for _ in range(98):
            value = [value]
        with self.assertRaises(audit.SourceSuccessorAuditError):
            audit.wire(value)


class ArchiveReadControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="source-successor-stdlib-guard-")
        self.root = Path(self.temporary.name).resolve()
        self.file = self.root / "artifact.json"
        self.file.write_bytes(b'{"value":1}')
        self.reader = audit.Reader(self.root, seconds=60)

    def tearDown(self):
        self.temporary.cleanup()

    def refused(self, function, *args, **kwargs):
        with self.assertRaises(audit.SourceSuccessorAuditError):
            function(*args, **kwargs)

    def test_positive_regular_file_and_canonical_cas(self):
        self.assertEqual(self.reader.json("artifact.json"), {"value": 1})
        raw = audit.wire({"value": 2}); wanted = audit.cid(raw)
        path = self.root / "private/source-artifacts/structured" / wanted[:4] / wanted
        path.parent.mkdir(parents=True); path.write_bytes(raw)
        self.assertEqual(self.reader.cas(wanted), raw)

    def test_locator_traversal_absolute_and_noncanonical(self):
        for locator in ("../outside", "/tmp/outside", "nested//item", "./artifact.json", "artifact.json\0"):
            with self.subTest(locator=locator):
                self.refused(self.reader.raw, locator)

    def test_file_symlink_refused_without_following(self):
        (self.root / "alias").symlink_to(self.file)
        self.refused(self.reader.raw, "alias")

    def test_parent_symlink_refused(self):
        (self.root / "alias-directory").symlink_to(self.root, target_is_directory=True)
        self.refused(self.reader.raw, "alias-directory/artifact.json")

    def test_hardlink_refused_by_raw_and_whole_archive(self):
        os.link(self.file, self.root / "second-link")
        self.refused(self.reader.raw, "artifact.json")
        self.refused(self.reader.whole_archive)

    def test_fifo_refused_without_blocking(self):
        os.mkfifo(self.root / "pipe")
        before = time.monotonic()
        self.refused(self.reader.raw, "pipe")
        self.assertLess(time.monotonic() - before, 1.0)
        self.refused(self.reader.whole_archive)

    def test_per_file_byte_limit(self):
        self.refused(self.reader.raw, "artifact.json", 1)

    def test_aggregate_read_byte_limit(self):
        self.reader.total_read_bytes = audit.MAX_TOTAL_READ_BYTES
        self.refused(self.reader.raw, "artifact.json")

    def test_deadline_before_read(self):
        self.reader.deadline = time.monotonic() - 1
        self.refused(self.reader.raw, "artifact.json")
        self.refused(self.reader.whole_archive)

    def test_deadline_type_and_bounds(self):
        for seconds in (True, 0, -1, math.inf, math.nan, 601):
            with self.subTest(seconds=seconds):
                self.refused(audit.Reader, self.root, seconds=seconds)

    def test_changed_bytes_between_observations(self):
        self.reader.raw("artifact.json")
        self.file.write_bytes(b'{"value":2}')
        self.refused(self.reader.raw, "artifact.json")

    def test_cas_rejects_mismatched_raw_bytes(self):
        wanted = audit.cid(b"expected", source=True)
        path = self.root / "private/source-artifacts/source" / wanted[:4] / wanted
        path.parent.mkdir(parents=True); path.write_bytes(b"different")
        self.refused(self.reader.cas, wanted, source=True)

    def test_cas_rejects_noncanonical_structured_bytes(self):
        raw = b'{ "value": 2 }'; wanted = audit.cid(raw)
        path = self.root / "private/source-artifacts/structured" / wanted[:4] / wanted
        path.parent.mkdir(parents=True); path.write_bytes(raw)
        self.refused(self.reader.cas, wanted)

    def test_archive_preserves_inert_source_symlinks(self):
        (self.root / "inert-link").symlink_to("unavailable-outside")
        record = self.reader.whole_archive()
        row = next(item for item in record["files"] if item["path"] == "inert-link")
        self.assertEqual(row["kind"], "symlink")
        self.assertEqual(row["target"], "unavailable-outside")
        self.refused(self.reader.raw, "inert-link")

    def test_archive_inventories_file_and_membership_change(self):
        before = self.reader.whole_archive()
        self.file.write_bytes(b'{"value":2}')
        after_bytes = self.reader.whole_archive()
        self.assertFalse(audit.same(before, after_bytes))
        (self.root / "new-file").write_bytes(b"new")
        self.assertFalse(audit.same(after_bytes, self.reader.whole_archive()))

    def test_archive_inventories_mode_change(self):
        before = self.reader.whole_archive()
        self.file.chmod(stat.S_IMODE(self.file.stat().st_mode) ^ stat.S_IXUSR)
        self.assertFalse(audit.same(before, self.reader.whole_archive()))

    def test_archive_entry_and_total_bytes_caps(self):
        with patch.object(audit, "MAX_FILES", 1):
            self.refused(self.reader.whole_archive)
        with patch.object(audit, "MAX_ARCHIVE_BYTES", 1):
            self.refused(self.reader.whole_archive)


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_ids = []

    def startTest(self, test):
        self.case_ids.append(test.id())
        super().startTest(test)


def main():
    global ARCHIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--receipt-directory", type=Path, required=True)
    args = parser.parse_args()
    ARCHIVE = args.archive.resolve(strict=True)
    output = args.receipt_directory.absolute()
    output.mkdir(parents=True, exist_ok=False)
    reader_before, test_before = READER_PATH.read_bytes(), Path(__file__).read_bytes()
    native_result_path = ARCHIVE / "result.json"
    native_result_before = native_result_path.read_bytes()
    (output / "reader.py").write_bytes(reader_before)
    (output / "test.py").write_bytes(test_before)
    log = io.StringIO(); started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordingResult).run(suite)
    elapsed = time.monotonic() - started
    pin_unchanged = READER_PATH.read_bytes() == reader_before and Path(__file__).read_bytes() == test_before
    receipt = {"schema": "source-successor-independent-reader-guard-tests@1",
        "qualified": result.wasSuccessful() and pin_unchanged and not result.skipped,
        "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "elapsed_seconds": elapsed, "archive": str(ARCHIVE), "namespace": str(ARCHIVE),
        "case_ids": result.case_ids, "generated_tamper_control_ids": CONTROL_IDS,
        "reader": {"path": str(READER_PATH), "bytes": len(reader_before), "sha256": audit.sha(reader_before)},
        "source_native_result": {"path": str(native_result_path), "bytes": len(native_result_before), "sha256": audit.sha(native_result_before)},
        "test": {"bytes": len(test_before), "sha256": audit.sha(test_before)}, "current_pins_unchanged": pin_unchanged,
        "scope": "stdlib inert JSON and temporary files only; separate from product pytest distinct-case totals",
        "native_owner_opens": 0, "sql_connections": 0, "git_commands": 0, "model_fitting": 0,
        "numerical_inference": 0, "source_property_solvers": 0,
        "execution_attestation": False}
    (output / "tests.log").write_text(log.getvalue())
    (output / "receipt.json").write_bytes(audit.numerical_wire(receipt) + b"\n")
    for path in output.iterdir():
        path.chmod(0o444)
    print(json.dumps({key: receipt[key] for key in ("qualified", "tests", "failures", "errors", "skipped", "elapsed_seconds", "current_pins_unchanged")}, sort_keys=True))
    return 0 if receipt["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
