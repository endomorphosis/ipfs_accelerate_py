"""Keep the signed public task text separate from authority constraint summaries."""
from __future__ import annotations

import hashlib
import json

FIELD = "terminal_public_instruction"


def bind_public_instruction(prompt: str, *, prepared: dict, maximum_bytes=262144) -> str:
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptWorkflowRequest
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise ValueError("positive planner request byte budget required")
    request = PromptWorkflowRequest.from_dict(prepared["request"])
    maximum_bytes = min(maximum_bytes, request.budget.max_serialized_bytes,
                        request.budget.max_prompt_tokens * 4)
    text = prepared["query"]
    if type(text) is not str or not text.strip() or len(text.encode("utf-8")) > 32768:
        raise ValueError("bounded verbatim public instruction required")
    source = ".supervisor-instruction.md"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if prepared["manifest"]["payload"]["sources"][source]["sha256"] != digest:
        raise ValueError("public instruction differs from signed source digest")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate planner request field")
            value[key] = item
        return value

    payload = json.loads(prompt, object_pairs_hook=unique)
    if type(payload) is not dict or FIELD in payload:
        raise ValueError("unmodified canonical planner request required")
    payload[FIELD] = {
        "schema": "terminal-planner-public-instruction@1",
        "source_path": source, "source_sha256": digest, "text": text,
        "role": "task_request_within_independently_declared_constraints",
        "execution_authority": False, "completion_authority": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > maximum_bytes:
        raise ValueError("public instruction exceeds the planner request byte budget")
    return encoded
