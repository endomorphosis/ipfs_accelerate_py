"""Bind a prepared token-selected input without claiming provider ingestion.

The unchanged @2 selection remains recommendation evidence. This distinct
receipt binds the input chosen by the caller under its live source checks;
it grants no execution, acceptance, provider-token or savings authority.
"""
from __future__ import annotations

from hashlib import sha256
import json
import re

from .semantic_working_set_selection import validate_complete_working_set_selection_receipt

PROFILE = "capsule-token-minimum@1"
SCHEMA = "supervisor-semantic-working-set-dispatch@1"
_FIELDS = frozenset({"schema", "requested_profile", "selection", "selected_representation",
    "selected_router_prompt_sha256", "selected_router_prompt_bytes",
    "selected_model_prompt_sha256", "selected_model_prompt_bytes",
    "selected_reply_contract_sha256", "preparation_only", "provider_ingestion_verified",
    "total_session_token_savings"})
MAX_RECEIPT_BYTES = 131072
MAX_PROMPT_BYTES = 256000


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise ValueError("closed working set dispatch JSON required") from error


def _hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_working_set_dispatch(receipt):
    """Validate prepared bindings; actual source/currentness remains external."""
    _need(type(receipt) is dict and set(receipt) == _FIELDS,
        "closed working set dispatch receipt required")
    _need(receipt["schema"] == SCHEMA and receipt["requested_profile"] == PROFILE
        and receipt["preparation_only"] is True
        and receipt["provider_ingestion_verified"] is False
        and receipt["total_session_token_savings"] is None,
        "working set dispatch scope differs")
    selection = validate_complete_working_set_selection_receipt(receipt["selection"])
    _need(selection["historical_replay"] is False
        and selection["measurement_status"] in {"measured", "estimated"}
        and selection["eligible_candidate_measurements_complete"] is True,
        "working set dispatch requires a consistent known local counter")
    _need(receipt["selected_representation"] == selection["selected_representation"],
        "working set dispatch winner differs")
    for prefix in ("selected_router_prompt", "selected_model_prompt"):
        _need(_hash(receipt[prefix + "_sha256"])
            and type(receipt[prefix + "_bytes"]) is int
            and 1 <= receipt[prefix + "_bytes"] <= MAX_PROMPT_BYTES,
            "bounded selected working set input required")
    _need(receipt["selected_model_prompt_sha256"] == selection["selected_complete_sha256"]
        and receipt["selected_model_prompt_bytes"] == selection["selected_complete_bytes"],
        "working set dispatch differs from counted complete input")
    _need(receipt["selected_reply_contract_sha256"] is None
        or _hash(receipt["selected_reply_contract_sha256"]),
        "working set reply contract hash differs")
    raw = _json(receipt)
    _need(len(raw) <= MAX_RECEIPT_BYTES, "working set dispatch receipt exceeds its bound")
    return json.loads(raw)


def bind_working_set_dispatch(*, selection, router_prompt, model_prompt, reply_contract=None):
    """Bind exact prepared strings; do not modify recommendation authority flags."""
    _need(type(router_prompt) is str and type(model_prompt) is str
        and (reply_contract is None or type(reply_contract) is dict),
        "literal selected working set inputs required")
    selection = validate_complete_working_set_selection_receipt(selection)
    router, model = router_prompt.encode("utf-8"), model_prompt.encode("utf-8")
    reply = None if reply_contract is None else _json(reply_contract)
    _need(reply is None or len(reply) <= MAX_RECEIPT_BYTES, "bounded reply contract required")
    return validate_working_set_dispatch(dict(schema=SCHEMA, requested_profile=PROFILE,
        selection=selection, selected_representation=selection["selected_representation"],
        selected_router_prompt_sha256=sha256(router).hexdigest(), selected_router_prompt_bytes=len(router),
        selected_model_prompt_sha256=sha256(model).hexdigest(), selected_model_prompt_bytes=len(model),
        selected_reply_contract_sha256=None if reply is None else sha256(reply).hexdigest(),
        preparation_only=True, provider_ingestion_verified=False, total_session_token_savings=None))


__all__ = ["PROFILE", "SCHEMA", "bind_working_set_dispatch", "validate_working_set_dispatch"]
