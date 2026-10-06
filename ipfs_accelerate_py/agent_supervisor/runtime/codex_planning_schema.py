"""Generation-only Codex projection of the exact native planning grammar.

The original schema and native plan admission remain authoritative. This
adapter never repairs output, changes task fields, or certifies task semantics.
Intent-coverage envelopes deliberately remain on their existing route.
"""
from __future__ import annotations

import hashlib
from typing import Any

from ipfs_accelerate_py.cli_runtime.grok_structured_output import (
    MAX_RESPONSE_BYTES,
    _bounded_json,
    _loads,
    native_schema_projection,
    planning_response_format,
    validate_response_format,
)

MAX_PROMPT_BYTES = 256_000
PROJECTION_ID = "canonical-prompt-goal-codex-generation-schema@1"


def planning_schema(prompt: str) -> tuple[dict, dict, Any] | None:
    """Return a bounded wire schema, receipt, and independent native validator.

    Only the direct canonical request is selected. Its complete schema must
    equal the native factory under the schema's dynamic budget fields, rather
    than merely carrying the planner's ID. The returned mappings are snapshots.
    """
    if not isinstance(prompt, str):
        raise ValueError("Codex planning prompt must be text")
    try:
        if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise ValueError("Codex planning prompt exceeds its byte bound")
    except UnicodeError:
        raise ValueError("Codex planning prompt is not valid UTF-8") from None

    selected = planning_response_format(prompt)
    if selected is None:
        return None
    canonical, canonical_encoded, validator = validate_response_format(selected)
    wire, _, native_receipt = native_schema_projection(canonical)
    if (native_receipt["projection_id"] != "canonical-prompt-goal-native-id-omission@1"
            or native_receipt["top_level_id_omitted"] is not True):
        raise ValueError("Codex planning requires the exact canonical native schema")

    # Both are native annotations, not generation assertions. The independent
    # validator retains them and the response budget remains enforced below.
    del wire["bounds"]
    wire["$defs"] = wire.pop("definitions")

    def project(value):
        if isinstance(value, dict):
            if "$ref" in value:
                reference = value["$ref"]
                if not reference.startswith("#/definitions/"):
                    raise ValueError("Codex planning schema reference is unsupported")
                value["$ref"] = "#/$defs/" + reference.removeprefix("#/definitions/")
            if "const" in value:
                constant = value.pop("const")
                if type(constant) is not str:
                    raise ValueError("Codex planning constant type is unsupported")
                value["type"] = "string"
                value["enum"] = [constant]
            elif "enum" in value and "type" not in value:
                if not all(type(item) is str for item in value["enum"]):
                    raise ValueError("Codex planning enum type is unsupported")
                value["type"] = "string"
            for child in value.values():
                project(child)
        elif isinstance(value, list):
            for child in value:
                project(child)

    project(wire)
    wire, encoded, _ = validate_response_format({
        "type": "json_schema",
        "json_schema": {"name": "supervisor_codex_prompt_goal_proposal",
                        "strict": True, "schema": wire},
    })
    # The trusted factory has only single-definition local references. Verify
    # their relocated targets without resolving any external resource.
    def verify_references(value):
        if isinstance(value, dict):
            if "$ref" in value:
                reference = value["$ref"]
                name = reference.removeprefix("#/$defs/")
                if (not reference.startswith("#/$defs/") or "/" in name
                        or name not in wire["$defs"]):
                    raise ValueError("Codex planning schema reference is unresolved")
            for child in value.values():
                verify_references(child)
        elif isinstance(value, list):
            for child in value:
                verify_references(child)

    verify_references(wire)
    receipt = {
        "projection_id": PROJECTION_ID,
        "canonical_schema_sha256": hashlib.sha256(canonical_encoded.encode()).hexdigest(),
        "canonical_schema_bytes": len(canonical_encoded.encode()),
        "native_wire_schema_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "native_wire_schema_bytes": len(encoded.encode()),
        "top_level_id_omitted": True,
        "bounds_annotation_omitted": True,
        "definitions_relocated": True,
        "canonical_validation_preserved": True,
        "shape_only": True,
        "execution_authority": False,
        "proof_authority": False,
        "completion_authority": False,
        "publication_authority": False,
        "scope_expansion_authority": False,
    }
    return wire, receipt, validator


def validate_planning_response(output: str, validator: Any) -> str:
    """Validate original JSON against the native schema without rewriting it.

    Diagnostics are static because validator exceptions can contain provider
    text. Cross-field semantics and independent signed admission remain the
    native planner's responsibility after this transport check.
    """
    try:
        maximum = min(MAX_RESPONSE_BYTES, validator.schema["bounds"]["max_serialized_bytes"])
        if (type(output) is not str or not output or output != output.strip()
                or not output.startswith("{") or not output.endswith("}")
                or len(output.encode("utf-8")) > maximum):
            raise ValueError("invalid original response")
        payload = _loads(output)
        _bounded_json(payload, maximum)
        if type(payload) is not dict:
            raise ValueError("response must be an object")
        validator.validate(payload)
    except Exception:
        raise ValueError("Codex planning response does not match the canonical schema") from None
    return output


__all__ = ["planning_schema", "validate_planning_response"]
