"""Match one explicit integer-offset intent against freshly checked source.

The closed sentence grammar below is an optional authored input profile, not a
natural-language interpretation service. Results describe the declared integer
contract only. They do not discharge runtime behavior, admit plans, remove tasks
or turn structural observations into behavioral facts.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import Any

from ipfs_datasets_py.logic.intent_ir.canonicalize import canonical_intent_ir_bytes
from ipfs_datasets_py.logic.intent_ir.decoder import decode_intent_ir
from ipfs_datasets_py.logic.intent_ir.schema import (
    IntentIRDocument, IntentKind, IntentModality, IntentStatement, NodeGrounding,
    ReviewStatus, SourceRef, SourceSpan, StatementKind,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

from .structural_codebase_context import run_with_structural_codebase_context

SCHEMA = "supervisor-checked-integer-intent@1"
QUERY_SCHEMA = "supervisor-integer-offset-query@1"
CNL_PROFILE = "integer-offset-intent-sentence@1"
PREDICATE = "integer_offset"
STATEMENT_ID = "integer-offset-goal"
_PROFILE = "python-integer-offset@1"
_MAX_JSON_BYTES = 1024 * 1024
_SOURCE_FIELDS = ("ref_id", "source_uri", "source_id", "source_revision", "content_sha256")
_SENTENCE = re.compile(
    r"Under python-integer-offset@1, (?P<path>[A-Za-z0-9_./-]{1,1024})::"
    r"(?P<function>[A-Za-z_][A-Za-z0-9_]{0,127})\("
    r"(?P<parameter>[A-Za-z_][A-Za-z0-9_]{0,127})\) must return "
    r"(?P=parameter) (?P<sign>[+-]) (?P<offset>0|[1-9][0-9]{0,19})\."
)
_AUTHORITY = {
    "source_semantics_verified": False, "runtime_behavior_verified": False,
    "proof_authority": False, "execution_authority": False,
    "completion_authority": False, "mutation_authority": False,
    "behavioral_satisfaction": False,
}


class CheckedIntegerIntentError(ValueError):
    """Invalid or inconsistent native input for the bounded intent profile."""


def _json(value):
    pending, count, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > 20_000 or depth > 24:
            raise CheckedIntegerIntentError("bounded intent/check JSON required")
        if type(item) is dict:
            if len(item) > 20_000 - count or any(type(key) is not str for key in item):
                raise CheckedIntegerIntentError("string JSON keys required")
            for key in item:
                if len(key) > _MAX_JSON_BYTES:
                    raise CheckedIntegerIntentError("intent/check JSON key limit exceeded")
                text_bytes += len(key.encode("utf-8", errors="surrogatepass"))
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            if len(item) > 20_000 - count:
                raise CheckedIntegerIntentError("intent/check JSON collection limit exceeded")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > _MAX_JSON_BYTES:
                raise CheckedIntegerIntentError("intent/check JSON string limit exceeded")
            text_bytes += len(item.encode("utf-8", errors="surrogatepass"))
        elif type(item) is int and item.bit_length() > 128:
            raise CheckedIntegerIntentError("intent/check JSON integer limit exceeded")
        elif type(item) not in {str, int, float, bool, type(None)}:
            raise CheckedIntegerIntentError("exact inert JSON values required")
        if text_bytes > _MAX_JSON_BYTES:
            raise CheckedIntegerIntentError("intent/check JSON text limit exceeded")
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    except (ValueError, TypeError, RecursionError) as exc:
        raise CheckedIntegerIntentError("finite bounded JSON required") from exc
    if len(raw) > _MAX_JSON_BYTES:
        raise CheckedIntegerIntentError("intent/check JSON byte limit exceeded")
    return json.loads(raw)


def _source_text(value):
    if type(value) is not str or not value or len(value) > 4096:
        raise CheckedIntegerIntentError("nonempty intent source of at most 4096 bytes required")
    try:
        raw = value.encode("utf-8")
    except UnicodeError as exc:
        raise CheckedIntegerIntentError("valid UTF-8 intent source required") from exc
    if len(raw) > 4096:
        raise CheckedIntegerIntentError("intent source exceeds 4096 bytes")
    return value


def _contract_from_sentence(source_text):
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract

    match = _SENTENCE.fullmatch(_source_text(source_text))
    if match is None:
        raise CheckedIntegerIntentError("exact integer-offset sentence syntax required")
    magnitude = int(match["offset"])
    if match["sign"] == "-" and magnitude == 0:
        raise CheckedIntegerIntentError("negative zero is not canonical")
    try:
        return IntegerOffsetContract(match["path"], match["function"], match["parameter"],
                                     magnitude if match["sign"] == "+" else -magnitude)
    except (ValueError, TypeError) as exc:
        raise CheckedIntegerIntentError("sentence is outside the integer-offset contract profile") from exc


def _arguments(contract):
    return [contract.path, contract.function_name, contract.parameter, str(contract.offset), _PROFILE]


def build_integer_offset_intent(
    source_text: str, *, source_id: str = "integer-offset:request", source_revision: str = "authored:1",
) -> IntentIRDocument:
    """Build native IR for exactly ``Under PROFILE, file.py::f(n) must return n + 1.``.

    A minus sign permits negative offsets. Whitespace, spelling, parameter
    repetition and decimal spelling are exact. No model or solver is invoked.
    """
    contract = _contract_from_sentence(source_text)
    source = SourceRef(
        ref_id="source:integer-offset", source_uri="intent:integer-offset",
        source_id=source_id, source_revision=source_revision,
        content_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
        review_status=ReviewStatus.UNREVIEWED, span=SourceSpan(0, len(source_text)),
    )
    statement = IntentStatement(
        statement_id=STATEMENT_ID, kind=StatementKind.GOAL, modality=IntentModality.REQUIRED,
        normalized_text=source_text, source_ref_ids=(source.ref_id,), predicate=PREDICATE,
        arguments=tuple(_arguments(contract)), grounding=NodeGrounding.GROUNDED,
        review_status=ReviewStatus.UNREVIEWED,
    )
    document = IntentIRDocument(
        document_id="intent:integer-offset", title="Explicit integer offset contract",
        intent_kind=IntentKind.DECLARATIVE, sources=(source,), statements=(statement,),
    )
    document.validate()
    return document


def prepare_integer_offset_query(
    *, intent_document, source_text: str, statement_id: str = STATEMENT_ID, source_identity=None,
) -> dict[str, Any]:
    """Validate a native atom without source scanning, admission or inference.

    Unsupported valid atoms produce an unknown query. Malformed documents and
    source provenance fail. A typed atom may explicitly author its query; only
    an independently rebuilt exact sentence receives closed-syntax alignment.
    """
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract

    _source_text(source_text)
    raw = intent_document.to_dict() if type(intent_document) is IntentIRDocument else intent_document
    try:
        native = decode_intent_ir(_json(raw))
        native.validate()
    except (ValueError, TypeError, KeyError) as exc:
        raise CheckedIntegerIntentError("valid exact native IntentIR required") from exc
    if len(native.statements) > 64 or len(native.sources) > 16 or len(native.actions) > 64:
        raise CheckedIntegerIntentError("native intent collection limit exceeded")
    selected = next((item for item in native.statements if item.statement_id == statement_id), None)
    if selected is None:
        raise CheckedIntegerIntentError("one existing native statement ID required")
    refs = [item for item in native.sources if item.ref_id in selected.source_ref_ids]
    if len(refs) != 1:
        raise CheckedIntegerIntentError("selected statement requires one exact intent source")
    source = refs[0]
    identity = {name: getattr(source, name) for name in _SOURCE_FIELDS}
    if source_identity is not None and _json(source_identity) != identity:
        raise CheckedIntegerIntentError("selected native source identity differs")
    if (source.content_sha256 != hashlib.sha256(source_text.encode()).hexdigest()
            or source.span is None or source.span.start_char != 0 or source.span.end_char != len(source_text)):
        raise CheckedIntegerIntentError("selected source digest and complete character span must match")
    reasons = []
    if (len(native.statements) != 1 or len(native.sources) != 1 or native.actions
            or native.control_edges or native.entry_action_ids or native.terminal_action_ids
            or native.intent_kind is not IntentKind.DECLARATIVE
            or selected.kind is not StatementKind.GOAL or selected.modality is not IntentModality.REQUIRED
            or selected.grounding is not NodeGrounding.GROUNDED):
        reasons.append("one_grounded_required_declarative_goal_required")
    contract = None
    arguments = list(selected.arguments)
    if selected.predicate != PREDICATE or len(arguments) != 5 or arguments[-1] != _PROFILE:
        reasons.append("unsupported_native_predicate_or_profile")
    else:
        try:
            offset_text = arguments[3]
            if (type(offset_text) is not str or len(offset_text) > 21
                    or re.fullmatch(r"0|-?[1-9][0-9]{0,19}", offset_text) is None):
                raise ValueError("canonical integer argument required")
            contract = IntegerOffsetContract(arguments[0], arguments[1], arguments[2], int(offset_text))
            if arguments != _arguments(contract):
                raise ValueError("exact ordered contract arguments required")
        except (ValueError, TypeError):
            reasons.append("unsupported_integer_offset_arguments")
            contract = None
    alignment = False
    try:
        rebuilt = build_integer_offset_intent(source_text)
    except CheckedIntegerIntentError:
        rebuilt = None
    if rebuilt is not None:
        expected = rebuilt.statements[0]
        if (selected.predicate != expected.predicate or selected.arguments != expected.arguments
                or selected.normalized_text != source_text):
            reasons.append("closed_sentence_native_meaning_mismatch")
        elif not reasons:
            alignment = True
    result = {
        "schema": QUERY_SCHEMA, "profile": _PROFILE,
        "native_document_sha256": hashlib.sha256(canonical_intent_ir_bytes(native)).hexdigest(),
        "statement": {"statement_id": selected.statement_id, "predicate": selected.predicate,
                      "arguments": arguments, "kind": selected.kind.value, "modality": selected.modality.value},
        "intent_source": {"identity": identity, "span": source.span.to_dict()},
        "contract": None if contract is None else contract.to_dict(),
        "contract_cid": None if contract is None else contract.cid,
        "supported": not reasons, "reasons": reasons,
        "semantic_alignment_verified": alignment,
        "alignment_scope": CNL_PROFILE if alignment else "explicit_authored_native_atom_only",
        "requirement_ids": [item.statement_id for item in sorted(native.statements, key=lambda item: item.statement_id)],
        **_AUTHORITY,
    }
    result["query_cid"] = cid_for_structured(result)
    return _json(result)


def match_checked_integer_intent(
    *, index, repository, intent_document, source_text: str, repository_id: str,
    statement_id: str = STATEMENT_ID, source_identity=None, expected_head=None,
    scheduler=None, parent_lease=None, cancel_event=None, admission_timeout_seconds: float = 30.0,
    timeout_seconds: float = 120.0, memory_mb: int = 1024, cache=None,
) -> dict[str, Any]:
    """Check source through its owner and return only after live exit observation.

    There is deliberately no caller-supplied evidence/result parameter. Every
    supported query invokes the datasets verifier, including cache reuse. A
    checked result answers this conditional contract; every runtime requirement
    remains residual. This function does not activate PlanCreateService.
    """
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        LeaseCancelledError, LeaseTimeoutError,
    )

    query = prepare_integer_offset_query(
        intent_document=intent_document, source_text=source_text,
        statement_id=statement_id, source_identity=source_identity,
    )
    if type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise CheckedIntegerIntentError("finite positive overall timeout required")
    if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
        raise CheckedIntegerIntentError("cancel_event must provide is_set()")
    deadline = time.monotonic() + timeout_seconds

    def remaining():
        if cancel_event is not None and cancel_event.is_set():
            raise LeaseCancelledError("integer intent check cancelled")
        duration = deadline - time.monotonic()
        if duration <= 0:
            raise LeaseTimeoutError("integer intent check deadline exceeded")
        return duration

    def build(context):
        checked, result_cid, status = None, None, "unknown"
        reasons = list(query["reasons"])
        if query["supported"]:
            from ipfs_datasets_py.logic.software_contracts.codebase_integer_verification import CodebaseIntegerVerifier

            contract = IntegerOffsetContract.from_dict(query["contract"])
            duration = remaining()
            checked = _json(CodebaseIntegerVerifier(index, cache=cache).verify(
                repository, expected_head=context.head, contract=contract,
                scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event,
                admission_timeout_seconds=min(admission_timeout_seconds, duration),
                timeout_seconds=duration, memory_mb=memory_mb,
            ))
            remaining()
            if (type(checked) is not dict or checked.get("schema") != "codebase-integer-verification@1"
                    or checked.get("profile") != _PROFILE or checked.get("head") != context.head.to_dict()
                    or checked.get("contract") != contract.to_dict() or checked.get("contract_cid") != contract.cid
                    or checked.get("kernel_checked") is not False or checked.get("behavior_authority") is not False):
                raise CheckedIntegerIntentError("owner checked result binding or authority differs")
            outcome = checked.get("status")
            if outcome not in {"proved", "refuted", "unknown", "timeout", "unavailable", "disagreement", "unsupported", "error"}:
                raise CheckedIntegerIntentError("unknown owner checked result status")
            if outcome in {"proved", "refuted"}:
                if checked.get("solver_replayed") is not True:
                    raise CheckedIntegerIntentError("usable owner outcome requires fresh solver replay")
                status = "conditional_contract_matched" if outcome == "proved" else "conditional_contract_refuted"
            else:
                reasons.append("checked_contract_" + outcome)
            result_cid = cid_for_structured(checked)
        result = {
            "schema": SCHEMA, "profile": _PROFILE, "status": status,
            "query": query, "structural_context": context.to_dict(),
            "checked_result": checked, "checked_result_cid": result_cid,
            "reasons": reasons, "scope": "declared_integer_contract_only",
            "semantic_alignment_verified": query["semantic_alignment_verified"],
            "alignment_scope": query["alignment_scope"],
            "residual_requirements": [{"statement_id": item, "status": "runtime_behavior_unresolved"}
                                      for item in query["requirement_ids"]],
            "current_behavioral_facts": [], "behavioral_satisfied_requirements": [],
            "runtime_refutations": [], "removed_task_ids": [], **_AUTHORITY,
        }
        result["match_cid"] = cid_for_structured(result)
        return _json(result)

    return run_with_structural_codebase_context(
        index, repository, build, repository_id=repository_id, expected_head=expected_head,
        scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event,
        admission_timeout_seconds=admission_timeout_seconds, timeout_seconds=remaining(), memory_mb=memory_mb,
    )


__all__ = ["SCHEMA", "QUERY_SCHEMA", "CNL_PROFILE", "PREDICATE", "STATEMENT_ID",
           "CheckedIntegerIntentError", "build_integer_offset_intent",
           "prepare_integer_offset_query", "match_checked_integer_intent"]
