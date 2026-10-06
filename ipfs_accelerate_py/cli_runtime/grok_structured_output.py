"""Bounded native Grok JSON-schema transport, independent of plan admission.

Schema conformance is a transport check. The supervisor still validates graph
references, scope, evidence, resource bounds and authority after this layer.
"""
from __future__ import annotations

import json
from typing import Any

MAX_SCHEMA_BYTES = 65_536
MAX_RESPONSE_BYTES = 262_144
MAX_ENVELOPE_BYTES = 4_194_304


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("structured JSON contains duplicate fields")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("structured JSON contains a non-finite number")


def _loads(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)


def _bounded_json(value: Any, maximum: int, *, schema: bool = False) -> str:
    """Reject cycles/deep trees before serialization or validator traversal."""
    ancestors = set()
    visited = 0

    def visit(item, depth):
        nonlocal visited
        visited += 1
        if depth > 32 or visited > 32_768:
            raise ValueError("structured JSON exceeds its tree bound")
        if isinstance(item, (dict, list)):
            if id(item) in ancestors:
                raise ValueError("structured JSON contains a cycle")
            ancestors.add(id(item))
            if isinstance(item, dict):
                if any(not isinstance(key, str) for key in item):
                    raise ValueError("structured JSON requires string object keys")
                for key, child in item.items():
                    if schema and key in {"$ref", "$dynamicRef", "$recursiveRef"}:
                        if not isinstance(child, str) or not (child == "#" or child.startswith("#/")):
                            raise ValueError("structured schema permits only local references")
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
            ancestors.remove(id(item))
        elif item is not None and type(item) not in {str, int, float, bool}:
            raise ValueError("structured JSON contains a non-JSON value")

    visit(value, 0)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > maximum:
        raise ValueError("structured JSON exceeds its byte bound")
    return encoded


def validate_response_format(response_format):
    """Return the exact supported schema, canonical argv value and validator."""
    if (not isinstance(response_format, dict)
            or set(response_format) != {"type", "json_schema"}
            or response_format["type"] != "json_schema"):
        raise ValueError("Grok response_format requires strict json_schema")
    definition = response_format["json_schema"]
    if (not isinstance(definition, dict) or set(definition) != {"name", "strict", "schema"}
            or definition["strict"] is not True
            or not isinstance(definition["name"], str)
            or not 1 <= len(definition["name"]) <= 64
            or not all(char.isascii() and (char.isalnum() or char in "_-") for char in definition["name"])):
        raise ValueError("Grok response_format requires a named strict JSON schema")
    schema = definition["schema"]
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("Grok structured output requires an object schema")
    encoded = _bounded_json(schema, MAX_SCHEMA_BYTES, schema=True)
    # The validator and argv must bind one immutable snapshot even if a caller
    # later mutates the original response_format mapping.
    schema = _loads(encoded)
    # jsonschema is an explicit runtime dependency. Missing validation support
    # fails before the native call; it never downgrades to unconstrained text.
    from jsonschema.validators import validator_for
    from referencing import Registry

    def reject_external(_uri):
        raise ValueError("structured schema external resolution is forbidden")

    validator_class = validator_for(schema, default=None) if "$schema" in schema else validator_for(schema)
    if validator_class is None:
        raise ValueError("Grok structured schema uses an unsupported dialect")
    try:
        validator_class.check_schema(schema)
    except Exception:
        raise ValueError("Grok structured schema is invalid") from None
    validator = validator_class(schema, registry=Registry(retrieve=reject_external))
    return schema, encoded, validator


def bind_schema_argument(command, encoded):
    """Merge literal argv without accepting conflicting operator overrides."""
    if "--" in command:
        raise ValueError("custom Grok command terminates options before response_format")
    existing = []
    formats = []
    for index, part in enumerate(command):
        if part == "--json-schema":
            existing.append(command[index + 1] if index + 1 < len(command) else None)
        elif part.startswith("--json-schema="):
            existing.append(part.partition("=")[2])
        elif part == "--output-format":
            formats.append(command[index + 1] if index + 1 < len(command) else None)
        elif part.startswith("--output-format="):
            formats.append(part.partition("=")[2])
    if formats and formats != ["json"]:
        raise ValueError("custom Grok output format conflicts with response_format")
    if existing:
        try:
            agrees = len(existing) == 1 and _bounded_json(_loads(existing[0]), MAX_SCHEMA_BYTES, schema=True) == encoded
        except (ValueError, TypeError, RecursionError):
            agrees = False
        if not agrees:
            raise ValueError("custom Grok schema conflicts with response_format")
    else:
        command.extend(["--json-schema", encoded])


def decode_response(stdout, validator):
    """Read native structured fields; never repair prose or drop trailing text."""
    if not isinstance(stdout, str) or len(stdout.encode("utf-8")) > MAX_ENVELOPE_BYTES:
        raise ValueError("Grok structured envelope exceeds its byte bound")
    text = stdout.strip()
    try:
        payload = _loads(text)
    except json.JSONDecodeError:
        # Native NDJSON may precede the final envelope. Only the final nonempty
        # line can provide a result; an earlier successful object cannot hide
        # a later failure. Duplicate-key errors are deliberately not caught.
        lines = text.splitlines()
        payload = _loads(lines[-1] if lines else "")
    _bounded_json(payload, MAX_ENVELOPE_BYTES)
    if not isinstance(payload, dict):
        raise ValueError("Grok structured envelope must be an object")
    stops = [payload[key] for key in ("stopReason", "stop_reason") if key in payload]
    if stops and (any(type(value) is not str or value not in {"end_turn", "stop", "stop_sequence"}
                     for value in stops) or len(set(stops)) != 1):
        raise ValueError("Grok native structured response did not finish normally")
    if (str(payload.get("type", "")).lower() == "error" or payload.get("is_error") is True
            or payload.get("type") == "max_turns_reached"
            or str(payload.get("subtype", "")).startswith("error")
            or any(payload.get(key) not in (None, "") for key in ("structuredOutputError", "structured_output_error"))):
        raise ValueError("Grok native structured output failed")
    structured = [payload[key] for key in ("structuredOutput", "structured_output") if key in payload]
    if structured:
        if any(not isinstance(value, dict) for value in structured):
            raise ValueError("Grok native structured output must be an object")
        if len(structured) > 1 and _bounded_json(structured[0], MAX_RESPONSE_BYTES) != _bounded_json(structured[1], MAX_RESPONSE_BYTES):
            raise ValueError("Grok native structured output aliases conflict")
        result = structured[0]
    elif "text" in payload:
        result = _loads(payload["text"]) if isinstance(payload["text"], str) else payload["text"]
    else:
        # Some CLI versions emit the schema object directly. The exact schema
        # validator distinguishes that object from an incomplete envelope.
        result = payload
    encoded = _bounded_json(result, MAX_RESPONSE_BYTES)
    if not isinstance(result, dict):
        raise ValueError("Grok native structured response must be an object")
    try:
        validator.validate(result)
    except Exception:
        # Validation exception messages include provider content. Keep only a
        # bounded static diagnostic; raw model text is not a receipt.
        raise ValueError("Grok native response does not match the requested schema") from None
    return encoded


def planning_response_format(prompt):
    """Select the canonical prompt planner's existing schema without rewriting it."""
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
        PROMPT_GOAL_PROVIDER_REQUEST_SCHEMA, PROMPT_GOAL_PROPOSAL_SCHEMA,
        PROMPT_GOAL_PLANNER_VERSION, _TOP_FIELDS,
    )
    marker = PROMPT_GOAL_PROVIDER_REQUEST_SCHEMA
    try:
        request = _loads(prompt)
    except (ValueError, TypeError, RecursionError):
        if marker in prompt:
            raise ValueError("malformed canonical prompt-goal provider request") from None
        return None
    if not isinstance(request, dict) or request.get("schema") != marker:
        return None
    _bounded_json(request, 256_000)
    schema = request.get("response_schema")
    if (request.get("stage") != "prompt_goal_planning" or request.get("version") != PROMPT_GOAL_PLANNER_VERSION
            or not isinstance(schema, dict) or schema.get("$id") != PROMPT_GOAL_PROPOSAL_SCHEMA
            or schema.get("type") != "object" or schema.get("additionalProperties") is not False
            or not isinstance(schema.get("properties"), dict)
            or set(schema["properties"]) != _TOP_FIELDS
            or not isinstance(schema.get("required"), list)
            or any(not isinstance(item, str) for item in schema["required"])
            or set(schema["required"]) != _TOP_FIELDS
            or schema["properties"].get("schema") != {"const": PROMPT_GOAL_PROPOSAL_SCHEMA}
            or schema["properties"].get("proposal_version") != {"const": PROMPT_GOAL_PLANNER_VERSION}):
        raise ValueError("canonical prompt-goal response schema is missing or incompatible")
    result = {"type": "json_schema", "json_schema": {
        "name": "supervisor_prompt_goal_proposal", "strict": True, "schema": schema}}
    validate_response_format(result)
    return result
