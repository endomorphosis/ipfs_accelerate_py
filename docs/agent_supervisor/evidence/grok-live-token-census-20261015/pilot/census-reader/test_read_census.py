import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import read_census as reader


def fixture():
    """Authored tiny serialized objects; no runtime imports or retained content."""
    def size(value):
        return len(reader.canonical(value))

    def partition(obj, labels):
        rows = {key: dict(count=0, value_bytes=0, member_bytes=0) for key in (*labels, "other_fields")}
        for key, value in obj.items():
            row = rows[key if key in labels else "other_fields"]
            row["count"] += 1
            row["value_bytes"] += size(value)
            row["member_bytes"] += size(key) + 1 + size(value)
        return dict(object_bytes=size(obj), outer_punctuation_bytes=2 + max(0, len(obj) - 1), members=rows)

    def evidence(rows):
        groups = {key: dict(count=0, row_bytes=0, summary_utf8_bytes=0,
            summary_json_string_bytes=0, summary_json_string_overhead_bytes=0,
            row_envelope_bytes=0) for key in reader.GROUPS}
        for item in rows:
            key = {"semantic-context": "semantic_context", "source-formalization-context": "source_formalization_context"}.get(item["kind"], "other_evidence")
            row = groups[key]
            raw = len(item["summary"].encode())
            encoded = size(item["summary"])
            row["count"] += 1
            row["row_bytes"] += size(item)
            row["summary_utf8_bytes"] += raw
            row["summary_json_string_bytes"] += encoded
            row["summary_json_string_overhead_bytes"] += encoded - raw
            row["row_envelope_bytes"] += size(item) - encoded
        return dict(array_bytes=size(rows), outer_punctuation_bytes=2 + max(0, len(rows) - 1), groups=groups)

    rows = [dict(kind="semantic-context", summary='{"x":"α"}'),
            dict(kind="source-formalization-context", summary='{"proof":null}'),
            dict(kind="world", summary="pending")]
    native = dict(evidence=rows, goal="preserved", scope=[], acceptance=["check"], authority={})
    remaining = dict(native, evidence=rows[1:])
    semantic = dict(capsules=[{}], admissions=[], raw_sources={}, manifest={})
    provider = dict(native_context=remaining, translated_semantic=semantic,
        translation_table={}, instructions="bounded", native_suffix="")
    value = dict(reader.CONSTANTS, native_prompt_bytes=size(native), native_suffix_bytes=0,
        provider_prompt_bytes=size(provider), native_prompt_sha256=hashlib.sha256(reader.canonical(native)).hexdigest(),
        provider_prompt_sha256=hashlib.sha256(reader.canonical(provider)).hexdigest(),
        native_object=partition(native, reader.LABELS["native_object"]),
        remaining_native_context=partition(remaining, reader.LABELS["remaining_native_context"]),
        provider_object=partition(provider, reader.LABELS["provider_object"]),
        translated_semantic=partition(semantic, reader.LABELS["translated_semantic"]),
        native_evidence=evidence(rows), remaining_native_evidence=evidence(rows[1:]))
    translation = dict(schema="supervisor-semantic-router-encoding@1", completion_authority=False,
        execution_authority=False, semantic_equivalence_claimed=False,
        **{key: value[key] for key in (*reader.HASHES, "native_prompt_bytes", "provider_prompt_bytes")})
    return dict(schema="router-implementation-invocation@1", invocation_id="b" * 32, purpose="coding",
        native_prompt_bytes=size(native), native_prompt_sha256=value["native_prompt_sha256"],
        model_prompt_bytes=size(provider) + 10, model_prompt_sha256="a" * 64,
        semantic_prompt_components=value, semantic_translation=translation,
        raw_provider_text="MUST_NEVER_EXPORT_THIS", arbitrary_secret="MUST_NEVER_EXPORT_THIS")


class CensusTests(unittest.TestCase):
    def test_nominal_counts_and_no_unselected_content(self):
        receipt = fixture()
        result = reader.project_report(dict(provider_invocations=[receipt]))
        row = result["observations"][0]
        self.assertEqual(row["status"], "closed_schema_arithmetic_and_receipt_joins_verified")
        self.assertEqual(row["census"]["native_evidence"]["groups"]["semantic_context"]["summary_utf8_bytes"], 10)
        self.assertEqual(row["model_vs_measured_transport_bytes"], 10)
        self.assertFalse(row["independent_prompt_replay"])
        self.assertNotIn("MUST_NEVER_EXPORT_THIS", json.dumps(result))
        self.assertNotIn("raw_provider_text", json.dumps(result))

    def test_missing_and_explicit_unavailable_remain_unknown(self):
        receipt = fixture()
        receipt.pop("semantic_prompt_components")
        self.assertEqual(reader.project_receipt(receipt)["status"], "absent")
        receipt["semantic_prompt_components"] = dict(schema=reader.SCHEMA, status="unavailable", raw_content_exported=False)
        self.assertEqual(reader.project_receipt(receipt)["status"], "unavailable")

    def test_boolean_unavailable_flag_not_coerced(self):
        receipt = fixture()
        receipt["semantic_prompt_components"] = dict(schema=reader.SCHEMA, status="unavailable", raw_content_exported=0)
        with self.assertRaises(ValueError): reader.project_receipt(receipt)

    def test_unknown_fields_or_labels_refused(self):
        for target in ((), ("provider_object", "members"), ("native_evidence", "groups")):
            value = fixture()["semantic_prompt_components"]
            row = value
            for key in target: row = row[key]
            row["untrusted_text"] = "not allowed"
            with self.assertRaises(ValueError): reader.validate_census(value)

    def test_flags_exact_types(self):
        for key, replacement in (("provider_calls", False), ("proof_authority", 0),
                ("nested_partitions_overlap", 1), ("provider_tokens", 0), ("token_estimate", True)):
            value = fixture()["semantic_prompt_components"]
            value[key] = replacement
            with self.assertRaises(ValueError): reader.validate_census(value)

    def test_count_boolean_negative_oversized_refused(self):
        for replacement in (True, -1, reader.MAX_BYTES + 1):
            value = fixture()["semantic_prompt_components"]
            value["native_evidence"]["groups"]["semantic_context"]["count"] = replacement
            with self.assertRaises(ValueError): reader.validate_census(value)

    def test_partition_and_summary_arithmetic(self):
        for path in (("native_object", "outer_punctuation_bytes"), ("provider_object", "object_bytes"),
                ("native_evidence", "groups", "semantic_context", "summary_json_string_overhead_bytes"),
                ("native_evidence", "groups", "other_evidence", "row_envelope_bytes")):
            value = fixture()["semantic_prompt_components"]
            row = value
            for key in path[:-1]: row = row[key]
            row[path[-1]] += 1
            with self.assertRaises(ValueError): reader.validate_census(value)

    def test_cross_partition_join_rejects_consistent_local_forgery(self):
        value = fixture()["semantic_prompt_components"]
        partition = value["native_object"]
        partition["members"]["evidence"]["value_bytes"] += 1
        partition["members"]["evidence"]["member_bytes"] += 1
        partition["object_bytes"] += 1
        value["native_prompt_bytes"] += 1
        with self.assertRaises(ValueError): reader.validate_census(value)

    def test_native_and_transport_receipt_digest_joins(self):
        for key in (*reader.HASHES, "native_prompt_bytes", "provider_prompt_bytes"):
            receipt = fixture()
            receipt["semantic_translation"][key] = "f" * 64 if key.endswith("sha256") else 1
            with self.assertRaises(ValueError): reader.project_receipt(receipt)
        receipt = fixture()
        receipt["native_prompt_sha256"] = "f" * 64
        with self.assertRaises(ValueError): reader.project_receipt(receipt)

    def test_hash_shape_rejected(self):
        for replacement in ("A" * 64, "z" * 64, "a" * 63, None):
            value = fixture()["semantic_prompt_components"]
            value["native_prompt_sha256"] = replacement
            with self.assertRaises(ValueError): reader.validate_census(value)

    def test_duplicate_and_two_distinct_receipts(self):
        first = fixture()
        second = copy.deepcopy(first)
        second["invocation_id"] = "c" * 32
        second.pop("semantic_prompt_components")
        result = reader.project_report(dict(provider_invocations=[first, copy.deepcopy(first), second]))
        self.assertEqual(result["duplicate_receipts_omitted"], 1)
        self.assertEqual(len(result["observations"]), 2)
        self.assertEqual(result["observations"][1]["status"], "absent")

    def test_conflicting_receipts_fail_whole_population(self):
        first = fixture()
        second = copy.deepcopy(first)
        second["arbitrary_secret"] = "changed"
        with self.assertRaises(ValueError): reader.project_report(dict(provider_invocations=[first, second]))

    def test_no_coding_does_not_invent_census(self):
        result = reader.project_report(dict(provider_invocations=[dict(purpose="planning")]))
        self.assertEqual(result["observations"], [])

    def test_duplicate_json_keys_and_nonfinite_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
                path.write_text(raw)
                with self.assertRaises(ValueError): reader.load(path)

    def test_recipe_single_report_and_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "jobs/arm/trial/agent/supervisor-result.json"
            report.parent.mkdir(parents=True)
            report.write_bytes(reader.canonical(dict(provider_invocations=[fixture()])))
            recipe = root / "recipe.json"
            recipe.write_bytes(reader.canonical(dict(output=str(root))))
            result = reader.read_recipe(recipe)
            self.assertEqual(result["retained_supervisor_report"]["sha256"], hashlib.sha256(report.read_bytes()).hexdigest())
            other = root / "jobs/arm/other/agent/supervisor-result.json"
            other.parent.mkdir(parents=True)
            other.write_bytes(report.read_bytes())
            with self.assertRaises(ValueError): reader.read_recipe(recipe)


if __name__ == "__main__":
    unittest.main()
