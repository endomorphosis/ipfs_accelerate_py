"""Owner-selected coding acknowledgment schema, without task authority.

The legacy route does not alter model input or accept a new response grammar.
The opt-in ordinary editing route presents its contract after all task context;
the caller must retain reserved semantic response decoding before validating
this acknowledgment. No residual task family is selected here.
"""
from __future__ import annotations

import hashlib
import json


LEGACY_REPLY_MODE = "legacy"
ORDINARY_COMPLETION_REPLY_MODE = "ordinary-completion@1"
CODING_REPLY_MODES = (LEGACY_REPLY_MODE, ORDINARY_COMPLETION_REPLY_MODE)
CODING_COMPLETION_SCHEMA = "supervisor-coding-completion@1"
_ACKNOWLEDGMENT = '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'
MAX_MODEL_PROMPT_BYTES = 256_000
MAX_COMPLETION_REPLY_BYTES = 1_024

_ORDINARY_INSTRUCTION = (
    "\n\nSupervisor coding reply contract (ordinary-completion@1):\n"
    "This dispatch requests ordinary source editing. Make the authorized edits "
    "in the allocated checkout, then return only this final JSON object:\n"
    + _ACKNOWLEDGMENT + "\n"
    "Do not return a structured residual candidate; no residual task "
    "family is selected for this dispatch.\n"
    "This line acknowledges candidate edits only. The supervisor independently "
    "validates candidate effects and decides task completion.\n"
)


def validate_coding_reply_mode(*, mode: str, purpose: str, provider: str) -> None:
    """Reject unsupported choices before any provider dispatch."""
    if type(mode) is not str or mode not in CODING_REPLY_MODES:
        raise ValueError("unsupported owner-selected coding reply mode")
    if mode != LEGACY_REPLY_MODE and purpose != "coding":
        raise ValueError("ordinary completion reply mode requires a coding route")
    if mode != LEGACY_REPLY_MODE and provider != "codex_cli":
        raise ValueError("ordinary completion reply mode requires the qualified Codex route")


def coding_completion_schema() -> dict:
    """Return a fresh closed object schema for native constrained generation."""
    return {
        "type": "object",
        "properties": {
            "schema": {"type": "string", "enum": [CODING_COMPLETION_SCHEMA]},
            "status": {"type": "string", "enum": ["candidate_ready"]},
        },
        "required": ["schema", "status"],
        "additionalProperties": False,
    }


def apply_coding_reply_contract(*, model_prompt: str, mode: str, purpose: str,
                               provider: str) -> tuple[str, dict | None]:
    """Append a fixed final instruction and attest its complete input overhead."""
    validate_coding_reply_mode(mode=mode, purpose=purpose, provider=provider)
    if mode == LEGACY_REPLY_MODE:
        return model_prompt, None
    if type(model_prompt) is not str:
        raise ValueError("coding reply contract requires a literal model prompt")
    before = model_prompt.encode("utf-8")
    instruction = _ORDINARY_INSTRUCTION.encode("utf-8")
    final_prompt = model_prompt + _ORDINARY_INSTRUCTION
    final = final_prompt.encode("utf-8")
    if len(final) > MAX_MODEL_PROMPT_BYTES:
        raise ValueError("model prompt with coding reply contract exceeds its byte bound")
    schema = json.dumps(coding_completion_schema(), ensure_ascii=False, allow_nan=False,
                        sort_keys=True, separators=(",", ":")).encode("utf-8")
    return final_prompt, {
        "schema": "supervisor-coding-reply-contract@1",
        "mode": mode,
        "owner_selected": True,
        "instruction_sha256": hashlib.sha256(instruction).hexdigest(),
        "instruction_bytes": len(instruction),
        "model_prompt_before_sha256": hashlib.sha256(before).hexdigest(),
        "model_prompt_before_bytes": len(before),
        "model_prompt_after_sha256": hashlib.sha256(final).hexdigest(),
        "model_prompt_after_bytes": len(final),
        "response_validated": False,
        "native_output_schema_sha256": hashlib.sha256(schema).hexdigest(),
        "native_output_schema_bytes": len(schema),
        "native_output_schema_requested": True,
        "residual_task_family_selected": False,
        "execution_authority": False,
        "completion_authority": False,
        "settlement_authority": False,
        "automatic_retry_admitted": False,
    }


def validate_coding_completion_reply(response: str) -> str:
    """Require the closed acknowledgment without repairing model text."""
    error_message = "ordinary coding completion reply does not match selected contract"
    if type(response) is not str or len(response.encode("utf-8")) > MAX_COMPLETION_REPLY_BYTES:
        raise ValueError(error_message)

    def exact_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(error_message)
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError(error_message)

    try:
        value = json.loads(response, object_pairs_hook=exact_pairs, parse_constant=reject_constant)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError(error_message) from error
    if (type(value) is not dict or set(value) != {"schema", "status"}
            or value["schema"] != CODING_COMPLETION_SCHEMA
            or value["status"] != "candidate_ready"):
        raise ValueError(error_message)
    return response
