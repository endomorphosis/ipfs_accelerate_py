"""Reviewed finite integer scheduling declarations joined to signed tasks.

The selector records an authored interpretation; it neither proves instruction
alignment nor authorizes execution. The runtime must verify the manifest and
independently check a witness against the exact immutable input constraints.
"""
from __future__ import annotations

from typing import Mapping

SCHEMA = "reviewed-integer-interval-schedule@1"
CONTRACT_SCHEMA = "intent-plan-requirement-contract@5"
CONSTRAINT_POLICY = "finite-half-open-capacity@1"
FALSE = {
    "semantic_alignment_verified": False, "proof_authority": False,
    "execution_authority": False, "publication_authority": False,
    "completion_authority": False,
}


def validate_reviewed_interval_schedule(value, *, operations, manifest=None):
    """Validate one authored scheduling operation without reading source bytes.

    Durations, release/deadline constraints, availability, resource demand and
    capacities come solely from the signed input JSON. A feasible witness does
    not assert optimality, and a failed search does not establish infeasibility.
    This declaration offers no selectable constraint weakening or trust flag.
    """
    from .intent_requirement_adapter import (
        IntentRequirementAdapterError, _json, _object, _path, _plain, _text,
    )

    def require(condition, message):
        if not condition:
            raise IntentRequirementAdapterError(message)

    value = _json(_plain(value))
    _object(value, {"schema", "review_ref", "operation_id", "input_path", "output_path",
                    "constraint_policy", "validation_key", *FALSE}, "reviewed interval schedule")
    require(value["schema"] == SCHEMA and value["constraint_policy"] == CONSTRAINT_POLICY,
            "closed reviewed interval schedule policy required")
    require(all(value[key] is False for key in FALSE), "schedule selector grants no authority")
    for key in ("review_ref", "operation_id", "validation_key"):
        _text(value[key], key)
    for key in ("input_path", "output_path"):
        _path(value[key])
        require(value[key].endswith(".json"), "explicit schedule .json paths required")
    require(value["input_path"] != value["output_path"], "immutable input and distinct create output required")
    require(type(operations) is list and len(operations) == 1, "one reviewed scheduling operation required")
    operation = operations[0]
    require(type(operation) is dict and operation.get("operation_id") == value["operation_id"],
            "schedule selector differs from the reviewed operation")
    output = {"path": value["output_path"], "effect": "create", "media_type": "application/json"}
    require(operation.get("outputs") == [output]
            and operation.get("validation_keys") == [value["validation_key"]]
            and operation.get("dependency_operation_ids") == [],
            "schedule operation requires exactly one JSON create output and validation")
    if manifest is None:
        return value

    require(isinstance(manifest, Mapping), "signed schedule manifest declarations required")
    payload = manifest.get("payload", manifest)
    require(isinstance(payload, Mapping)
            and payload.get("schema") == "supervisor-local-benchmark-manifest@4",
            "schedule selector requires the intent manifest version")
    tasks = payload.get("tasks")
    require(type(tasks) is list and len(tasks) == 1 and type(tasks[0]) is dict,
            "schedule selector requires one independently declared task")
    task = tasks[0]
    require(task.get("task_key") == operation.get("task_key") and task.get("outputs") == [output]
            and task.get("dependencies") == [], "schedule task differs from the reviewed operation")
    validations = task.get("validations")
    require(type(validations) is list and len(validations) == 1
            and type(validations[0]) is dict
            and validations[0].get("validation_key") == value["validation_key"],
            "schedule selector differs from the independent validation")
    sources = payload.get("sources")
    require(isinstance(sources, Mapping) and value["input_path"] in sources
            and value["output_path"] not in sources
            and payload.get("created_outputs") == [value["output_path"]],
            "schedule input must be signed and immutable and output independently declared absent")
    scopes = task.get("scope_paths")
    require(type(scopes) is list and {value["input_path"], value["output_path"]} <= set(scopes),
            "schedule input or output is outside independent task scope")
    return value
