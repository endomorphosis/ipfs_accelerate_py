"""Reviewed finite data transformations bound to signed task declarations.

The selector is an explicit operator input, not an inference about arbitrary
instruction meaning. It grants no proof, execution, publication or completion
authority. Runtime consumers must verify the enclosing manifest and independently
check the actual input/output record correspondence before staging a candidate.
"""
from __future__ import annotations

import re
from typing import Mapping

SCHEMA = "reviewed-ndjson-data-transform@1"
CONTRACT_SCHEMA = "intent-plan-requirement-contract@4"
CORRESPONDENCE_POLICY = "ordered-record-field-correspondence@1"
FALSE = {
    "semantic_alignment_verified": False, "proof_authority": False,
    "execution_authority": False, "publication_authority": False,
    "completion_authority": False,
}


def validate_reviewed_data_transform(value, *, operations, manifest=None):
    """Check one reviewed copy/rename; optionally join the signed declarations.

    The fixed policy preserves record order, count, multiplicity and unrelated
    fields. The source must exist and target must be absent in every record;
    runtime checking supplies those facts. No caller-selectable bounds, inferred
    field mappings, sorting, filtering, nested paths or overwrite policy exist.
    This function does not verify a manifest signature or read source bytes.
    """
    from .intent_requirement_adapter import (
        IntentRequirementAdapterError, _json, _object, _path, _plain, _text,
    )

    def require(condition, message):
        if not condition:
            raise IntentRequirementAdapterError(message)

    value = _json(_plain(value))
    _object(value, {"schema", "review_ref", "operation_id", "input_path", "output_path",
                    "mode", "source_field", "target_field", "correspondence_policy",
                    "validation_key", *FALSE}, "reviewed NDJSON transformation")
    require(value["schema"] == SCHEMA and value["correspondence_policy"] == CORRESPONDENCE_POLICY,
            "closed reviewed NDJSON transformation policy required")
    require(all(value[key] is False for key in FALSE), "data selector grants no authority")
    for key in ("review_ref", "operation_id", "validation_key"):
        _text(value[key], key)
    for key in ("source_field", "target_field"):
        _text(value[key], key, 64)
        require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", value[key]) is not None,
                "bounded ASCII field identifier required")
    require(type(value["mode"]) is str and value["mode"] in {"copy", "rename"}
            and value["source_field"] != value["target_field"],
            "distinct source and target fields and closed copy/rename mode required")
    for key in ("input_path", "output_path"):
        _path(value[key])
        require(value[key].endswith(".jsonl"), "explicit NDJSON .jsonl paths required")
    require(value["input_path"] != value["output_path"], "immutable input and distinct create output required")
    require(type(operations) is list and len(operations) == 1, "one reviewed data operation required")
    operation = operations[0]
    require(type(operation) is dict and operation.get("operation_id") == value["operation_id"],
            "data selector differs from the reviewed operation")
    output = {"path": value["output_path"], "effect": "create", "media_type": "application/x-ndjson"}
    require(operation.get("outputs") == [output]
            and operation.get("validation_keys") == [value["validation_key"]]
            and operation.get("dependency_operation_ids") == [],
            "data operation requires exactly one NDJSON create output and validation")
    if manifest is None:
        return value

    require(isinstance(manifest, Mapping), "signed data manifest declarations required")
    payload = manifest.get("payload", manifest)
    require(isinstance(payload, Mapping)
            and payload.get("schema") == "supervisor-local-benchmark-manifest@4",
            "data selector requires the intent manifest version")
    tasks = payload.get("tasks")
    require(type(tasks) is list and len(tasks) == 1 and type(tasks[0]) is dict,
            "data selector requires one independently declared task")
    task = tasks[0]
    require(task.get("task_key") == operation.get("task_key") and task.get("outputs") == [output]
            and task.get("dependencies") == [], "data task differs from the reviewed operation")
    validations = task.get("validations")
    require(type(validations) is list and len(validations) == 1
            and type(validations[0]) is dict
            and validations[0].get("validation_key") == value["validation_key"],
            "data selector differs from the independent validation")
    sources = payload.get("sources")
    require(isinstance(sources, Mapping) and value["input_path"] in sources
            and value["output_path"] not in sources
            and payload.get("created_outputs") == [value["output_path"]],
            "data input must be signed and immutable and output independently declared absent")
    scopes = task.get("scope_paths")
    require(type(scopes) is list and {value["input_path"], value["output_path"]} <= set(scopes),
            "data input or output is outside independent task scope")
    return value
