"""Closed, observation-only byte accounting; never exports source or prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "supervisor-semantic-prompt-component-census@1"
MAX_BYTES = 16_777_216
LABELS = {
    "native_object": ("evidence", "goal", "scope", "acceptance", "authority"),
    "remaining_native_context": ("evidence", "goal", "scope", "acceptance", "authority"),
    "provider_object": ("native_context", "translated_semantic", "translation_table", "instructions", "native_suffix"),
    "translated_semantic": ("capsules", "admissions", "raw_sources", "manifest"),
}
GROUPS = ("semantic_context", "source_formalization_context", "other_evidence")
CONSTANTS = dict(schema=SCHEMA,
    measurement_scope="canonical-serialized-prompt-utf8-bytes@1",
    partition_rule="member_bytes_plus_outer_punctuation_equals_object_bytes",
    nested_partitions_overlap=True, token_estimate=False, provider_tokens=None,
    model_prompt_wrappers_included=False, provider_calls=0, raw_content_exported=False,
    dynamic_labels_exported=False, source_freshness_verified=False, proof_authority=False,
    omission_authority=False, completion_authority=False, execution_authority=False,
    exact_native_round_trip_verified=True)
SIZES = ("native_prompt_bytes", "native_suffix_bytes", "provider_prompt_bytes")
HASHES = ("native_prompt_sha256", "provider_prompt_sha256")
EVIDENCE = ("native_evidence", "remaining_native_evidence")


def need(condition):
    if not condition:
        raise ValueError("invalid closed census evidence")


def closed(value, keys):
    need(type(value) is dict and set(value) == set(keys))


def integer(value):
    need(type(value) is int and 0 <= value <= MAX_BYTES)
    return value


def digest(value):
    need(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value))
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _partition(value, labels):
    closed(value, ("object_bytes", "outer_punctuation_bytes", "members"))
    closed(value["members"], (*labels, "other_fields"))
    counts = sizes = 0
    for label, row in value["members"].items():
        closed(row, ("count", "value_bytes", "member_bytes"))
        count, size, member = (integer(row[key]) for key in ("count", "value_bytes", "member_bytes"))
        need(count <= 1 if label != "other_fields" else True)
        need((size == member == 0) if count == 0 else member >= size + 3 * count)
        counts += count
        sizes += member
    need(integer(value["outer_punctuation_bytes"]) == 2 + max(0, counts - 1))
    need(integer(value["object_bytes"]) == sizes + value["outer_punctuation_bytes"])


def _evidence(value):
    closed(value, ("array_bytes", "outer_punctuation_bytes", "groups"))
    closed(value["groups"], GROUPS)
    counts = sizes = 0
    keys = ("count", "row_bytes", "summary_utf8_bytes", "summary_json_string_bytes",
            "summary_json_string_overhead_bytes", "row_envelope_bytes")
    for row in value["groups"].values():
        closed(row, keys)
        for number in row.values():
            integer(number)
        need(row["summary_json_string_bytes"] == row["summary_utf8_bytes"] + row["summary_json_string_overhead_bytes"])
        need(row["row_bytes"] == row["summary_json_string_bytes"] + row["row_envelope_bytes"])
        need(row["count"] != 0 or all(v == 0 for v in row.values()))
        counts += row["count"]
        sizes += row["row_bytes"]
    need(integer(value["outer_punctuation_bytes"]) == 2 + max(0, counts - 1))
    need(integer(value["array_bytes"]) == sizes + value["outer_punctuation_bytes"])


def validate_census(value):
    """Validate the producer's full schema and arithmetic, without live replay."""
    closed(value, (*CONSTANTS, *SIZES, *HASHES, *LABELS, *EVIDENCE))
    for key, expected in CONSTANTS.items():
        need(type(value[key]) is type(expected) and value[key] == expected)
    for key in SIZES:
        integer(value[key])
    for key in HASHES:
        digest(value[key])
    for key, labels in LABELS.items():
        _partition(value[key], labels)
    for key in EVIDENCE:
        _evidence(value[key])
    for partition, evidence in (("native_object", "native_evidence"), ("remaining_native_context", "remaining_native_evidence")):
        row = value[partition]["members"]["evidence"]
        need(row["count"] == 1 and row["value_bytes"] == value[evidence]["array_bytes"])
    for label, partition in (("native_context", "remaining_native_context"), ("translated_semantic", "translated_semantic")):
        row = value["provider_object"]["members"][label]
        need(row["count"] == 1 and row["value_bytes"] == value[partition]["object_bytes"])
    need(value["native_prompt_bytes"] == value["native_object"]["object_bytes"] + value["native_suffix_bytes"])
    need(value["provider_prompt_bytes"] == value["provider_object"]["object_bytes"])
    need(len(canonical(value)) <= 16_384)
    return value


def project_receipt(receipt):
    need(type(receipt) is dict and receipt.get("schema") == "router-implementation-invocation@1")
    need(receipt.get("purpose") == "coding")
    value = receipt.get("semantic_prompt_components")
    if value is None:
        return dict(status="absent", independent_prompt_replay=False)
    if value == dict(schema=SCHEMA, status="unavailable", raw_content_exported=False):
        need(type(value["raw_content_exported"]) is bool)
        return dict(status="unavailable", independent_prompt_replay=False)
    validate_census(value)
    translation = receipt.get("semantic_translation")
    need(type(translation) is dict and translation.get("schema") == "supervisor-semantic-router-encoding@1")
    for prefix in ("native_prompt", "provider_prompt"):
        for suffix in ("sha256", "bytes"):
            key = prefix + "_" + suffix
            need(type(translation.get(key)) is type(value[key]) and translation[key] == value[key])
            if prefix == "native_prompt":
                need(type(receipt.get(key)) is type(value[key]) and receipt[key] == value[key])
    need(translation.get("completion_authority") is False and translation.get("execution_authority") is False
         and translation.get("semantic_equivalence_claimed") is False)
    model_bytes = integer(receipt.get("model_prompt_bytes"))
    model_sha = digest(receipt.get("model_prompt_sha256"))
    return dict(status="closed_schema_arithmetic_and_receipt_joins_verified",
        independent_prompt_replay=False, current_source_verified=False, token_measurement=False,
        census=value, model_prompt_bytes=model_bytes, model_prompt_sha256=model_sha,
        model_vs_measured_transport_bytes=model_bytes - value["provider_prompt_bytes"],
        model_vs_transport_difference_is_not_only_wrappers=True)


def project_report(report):
    need(type(report) is dict and type(report.get("provider_invocations")) is list
         and len(report["provider_invocations"]) <= 64)
    observations, seen, duplicate_count = [], {}, 0
    for receipt in report["provider_invocations"]:
        need(type(receipt) is dict)
        if receipt.get("purpose") != "coding":
            continue
        identity = receipt.get("invocation_id")
        need(type(identity) is str and len(identity) == 32 and all(c in "0123456789abcdef" for c in identity))
        raw = canonical(receipt)
        if identity in seen:
            need(seen[identity] == raw)
            duplicate_count += 1
            continue
        seen[identity] = raw
        observations.append(dict(receipt_ordinal=len(observations) + 1,
            invocation_sha256=hashlib.sha256(identity.encode()).hexdigest(), **project_receipt(receipt)))
    return dict(schema="closed-semantic-component-readout@1", observations=observations,
        duplicate_receipts_omitted=duplicate_count, native_usage_total=None,
        broker_reads_are_not_all_provider_reads=True, nested_partitions_must_not_be_summed=True,
        raw_content_exported=False, provider_calls=0, execution_authority=False,
        original_producer_claims_are_not_independently_replayed=True)


def _object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result)
        result[key] = value
    return result


def load(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_BYTES)
    raw = path.read_bytes()
    need(len(raw) <= MAX_BYTES)
    value = json.loads(raw, object_pairs_hook=_object, parse_constant=lambda _: need(False))
    return value, dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def read_recipe(recipe):
    config, recipe_binding = load(recipe)
    need(type(config) is dict and type(config.get("output")) is str)
    paths = list(Path(config["output"]).glob("jobs/*/*/agent/supervisor-result.json"))
    need(len(paths) == 1)
    report, report_binding = load(paths[0])
    return dict(recipe=recipe_binding, retained_supervisor_report=report_binding, **project_report(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(read_recipe(args.recipe), indent=2, sort_keys=True, allow_nan=False))
