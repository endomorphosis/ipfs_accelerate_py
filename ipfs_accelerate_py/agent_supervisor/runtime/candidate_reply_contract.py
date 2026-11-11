"""Closed acknowledgment for an independently admitted second coding round.

This grammar refers to the input candidate generation. It is not a claim about
the edited postimage and cannot authorize completion, proof reuse or a retry.
"""
from __future__ import annotations

import hashlib
import json
import re

SCHEMA = "supervisor-candidate-round-completion@1"
MAX_PROMPT_BYTES = 256_000
MAX_REPLY_BYTES = 8192
BINDING_FIELDS = ("task_id", "round_number", "round_context_sha256", "workspace_generation")


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _binding(value):
    if (type(value) is not dict or type(value.get("task_id")) is not str
            or not 0 < len(value["task_id"].encode()) <= 4096
            or any(c in value["task_id"] for c in "\0\r\n")
            or type(value.get("round_number")) is not int or value["round_number"] != 2
            or any(type(value.get(key)) is not str or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None
                   for key in ("round_context_sha256", "workspace_generation"))):
        raise ValueError("exact admitted candidate reply binding required")
    return {key: value[key] for key in BINDING_FIELDS}


def candidate_completion_schema(binding):
    """Return the closed provider schema with exact invocation identifiers."""
    expected = dict(schema=SCHEMA, status="candidate_ready", **_binding(binding))
    return dict(type="object", properties={key: dict(type="integer" if type(value) is int else "string",
        enum=[value]) for key, value in expected.items()}, required=list(expected), additionalProperties=False)


def apply_candidate_reply_contract(*, model_prompt, binding):
    """Append an explicit override after retained instructions and read guidance."""
    if type(model_prompt) is not str:
        raise ValueError("literal candidate model prompt required")
    expected = dict(schema=SCHEMA, status="candidate_ready", **_binding(binding))
    instruction = ("\n\nSupervisor second coding round reply contract:\n"
        "The following final reply grammar supersedes any reply grammar inside the retained original "
        "task text. Preserve all original task requirements. Make authorized edits in the allocated "
        "checkout using the existing coding tools, then return only this exact JSON object:\n"
        + _json(expected) + "\n"
        "The identifiers name the candidate input before your edits. This is an acknowledgment of "
        "candidate edits only; independent supervisor validation decides completion. Do not return "
        "a baseline semantic envelope, semantic aliases, residual candidate, proof claim or patch "
        "inside the final response. Candidate capsules and formal artifacts remain advisory.\n")
    result = model_prompt + instruction
    if len(result.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ValueError("candidate model prompt with reply contract exceeds bound")
    schema = _json(candidate_completion_schema(binding))
    return result, dict(schema="supervisor-candidate-round-reply-contract@1", **_binding(binding),
        instruction_sha256=_sha(instruction), instruction_bytes=len(instruction.encode()),
        model_prompt_before_sha256=_sha(model_prompt), model_prompt_before_bytes=len(model_prompt.encode()),
        model_prompt_after_sha256=_sha(result), model_prompt_after_bytes=len(result.encode()),
        native_output_schema_sha256=_sha(schema), native_output_schema_bytes=len(schema.encode()),
        native_output_schema_requested=True, response_validated=False, proof_authority=False,
        execution_authority=False, completion_authority=False, settlement_authority=False)


def validate_candidate_completion_reply(response, *, binding):
    """Check literal reply only, without requiring pre-edit bytes to stay current."""
    expected = dict(schema=SCHEMA, status="candidate_ready", **_binding(binding))
    error = "candidate coding reply does not match admitted round contract"
    if type(response) is not str or len(response.encode("utf-8")) > MAX_REPLY_BYTES:
        raise ValueError(error)
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(error)
            value[key] = item
        return value
    def nonfinite(_value):
        raise ValueError(error)
    try:
        value = json.loads(response, object_pairs_hook=unique, parse_constant=nonfinite)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError(error) from exc
    if (type(value) is not dict or value != expected or set(value) != set(expected)
            or type(value.get("round_number")) is not int):
        raise ValueError(error)
    return response


__all__ = ["candidate_completion_schema", "apply_candidate_reply_contract",
           "validate_candidate_completion_reply"]
