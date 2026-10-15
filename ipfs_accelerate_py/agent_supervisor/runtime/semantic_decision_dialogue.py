"""Bounded owner-managed capsule dialogue over pinned provider invocations.

Every round replays the required task packet and delivered evidence. Local
input counts describe that replay; retained provider receipts describe charges.
The owner keeps policy, source bindings and budgets outside model output.
This grants neither additional provider tools nor candidate acceptance.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from . import semantic_router_translation as codec
from .router_implementation_runner import MAX_MODEL_PROMPT_BYTES, PromptTokenCounter, render_model_prompt
from .semantic_context_projection import SemanticProjectionBindings, _encoded, _structure, _text
from .semantic_decision_packet import SemanticDecisionPolicy
from .semantic_projection_dispatch import (
    expand_semantic_router_decision_packet, project_semantic_router_prompt,
)

ROUND_SCHEMA = "supervisor-semantic-decision-round@1"
REQUEST_SCHEMA = "supervisor-semantic-expansion-request@1"
FINAL_SCHEMA = "supervisor-semantic-decision-final@1"
SCHEMA = "supervisor-semantic-decision-dialogue@1"
_REJECTION_OBSERVATION_SCHEMA = "supervisor-semantic-decision-rejection-observation@1"
_REJECTION_REASON_CODES = MappingProxyType({
    "post_return_control_check": "post_return_control_check_refused",
    "post_return_current_verification": "current_context_refused",
    "response_contract_validation": "response_contract_refused",
    "native_fallback_response_validation": "native_response_refused",
})


class _ResponseContractReason(str, Enum):
    BOUNDED_TEXT = "bounded_response_text_refused"
    INVALID_JSON = "invalid_response_json"
    DUPLICATE_KEY = "duplicate_response_key"
    STRUCTURE = "response_structure_refused"
    NONOBJECT = "response_object_required"
    SCHEMA = "unknown_response_schema"
    FIELDS = "response_field_set_mismatch"
    SESSION = "response_session_mismatch"
    GENERATION = "response_generation_mismatch"
    BINDING = "response_binding_mismatch"
    FINAL_TEXT = "invalid_final_response_text"
    CAPSULE_IDS = "invalid_requested_capsule_ids"


@dataclass(frozen=True)
class _ResponseContractRejection:
    reason: _ResponseContractReason


def _mark_response_rejection(error, reason):
    """Attach only a trusted closed check marker; retain the exact exception."""
    error._semantic_response_contract_rejection = _ResponseContractRejection(reason)
    return error


def _observe_rejection(attempt, phase, check):
    """Record only the trusted check identity; preserve its original exception."""
    reason_code = _REJECTION_REASON_CODES[phase]
    try:
        return check()
    except Exception as error:
        attempt["rejection_observation"] = {
            "schema": _REJECTION_OBSERVATION_SCHEMA, "phase": phase,
            "reason_code": reason_code, "exception_type": type(error).__name__,
        }
        if phase == "response_contract_validation":
            marker = getattr(error, "_semantic_response_contract_rejection", None)
            if type(marker) is _ResponseContractRejection and type(marker.reason) is _ResponseContractReason:
                attempt["response_contract_rejection_observation"] = {
                    "schema": "supervisor-semantic-response-contract-rejection@1",
                    "reason_code": marker.reason.value,
                }
        raise


def _validate_native_response(text):
    if type(text) is not str or not text.strip() or len(text.encode()) > 65536:
        raise RuntimeError("invalid_native_response")


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class SemanticDecisionCall:
    call_id: str
    session_id: str
    generation: int
    binding_sha256: str
    provider: str
    model: str
    prompt: str
    timeout: int
    max_output_tokens: int
    model_prompt_sha256: str


@dataclass(frozen=True)
class SemanticDecisionReply:
    text: str
    receipt: dict


class SemanticDecisionProviderFailure(RuntimeError):
    def __init__(self, message="provider failed", *, receipt=None):
        super().__init__(message)
        self._receipt_json = _encoded(receipt or {}).decode()

    @property
    def receipt(self):
        return json.loads(self._receipt_json)


class SemanticDecisionDialogueError(RuntimeError):
    def __init__(self, reason, receipt):
        super().__init__(reason)
        self._receipt_json = _encoded(receipt).decode()

    @property
    def receipt(self):
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class SemanticDecisionResult:
    text: str
    receipt_json: str

    @property
    def receipt(self):
        return json.loads(self.receipt_json)


def _verify_current(repository, bindings):
    raw = codec._read(repository, bindings.artifact)
    if hashlib.sha256(raw).hexdigest() != bindings.expected_sha256:
        raise ValueError("decision parent changed")
    codec._verified_semantic(
        repository, artifact=bindings.artifact, text=raw.decode(),
        task_id=bindings.task_id, current=True,
    )


def _response(text, call):
    if type(text) is not str or not text.strip() or len(text.encode()) > 65536:
        raise _mark_response_rejection(ValueError("bounded nonempty dialogue response required"),
                                       _ResponseContractReason.BOUNDED_TEXT)

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise _mark_response_rejection(ValueError("duplicate dialogue response key"),
                                               _ResponseContractReason.DUPLICATE_KEY)
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=pairs)
    except json.JSONDecodeError as error:
        _mark_response_rejection(error, _ResponseContractReason.INVALID_JSON)
        raise
    except RecursionError as error:
        _mark_response_rejection(error, _ResponseContractReason.STRUCTURE)
        raise
    try:
        _structure(value, 65536)
    except Exception as error:
        _mark_response_rejection(error, _ResponseContractReason.STRUCTURE)
        raise
    common = {"schema", "session_id", "generation", "binding_sha256"}
    if type(value) is not dict:
        raise _mark_response_rejection(ValueError("exact dialogue response object required"),
                                       _ResponseContractReason.NONOBJECT)
    expected = (common | {"capsule_ids"} if value.get("schema") == REQUEST_SCHEMA
                else common | {"text"} if value.get("schema") == FINAL_SCHEMA else set())
    if (set(value) != expected or value.get("session_id") != call.session_id
            or type(value.get("generation")) is not int or value["generation"] != call.generation
            or value.get("binding_sha256") != call.binding_sha256):
        reason = (_ResponseContractReason.SCHEMA if value.get("schema") != REQUEST_SCHEMA
                  and value.get("schema") != FINAL_SCHEMA
                  else _ResponseContractReason.FIELDS if set(value) != expected
                  else _ResponseContractReason.SESSION if value.get("session_id") != call.session_id
                  else _ResponseContractReason.GENERATION if type(value.get("generation")) is not int
                      or value["generation"] != call.generation
                  else _ResponseContractReason.BINDING)
        raise _mark_response_rejection(ValueError("dialogue response fields or current round binding differ"), reason)
    if value["schema"] == FINAL_SCHEMA:
        if type(value["text"]) is not str or not value["text"].strip():
            raise _mark_response_rejection(ValueError("nonempty literal final response required"),
                                           _ResponseContractReason.FINAL_TEXT)
    elif (type(value["capsule_ids"]) is not list or not 1 <= len(value["capsule_ids"]) <= 4
          or any(type(item) is not str for item in value["capsule_ids"])
          or len(set(value["capsule_ids"])) != len(value["capsule_ids"])):
        raise _mark_response_rejection(ValueError("unique bounded requested capsule identities required"),
                                       _ResponseContractReason.CAPSULE_IDS)
    return value


def _round_prompt(projected, session, generation, bindings, policy, provider, model, delivered,
                  *, retrieval_query_reference=None):
    base = {
        "schema": ROUND_SCHEMA, "session_id": session, "generation": generation,
        "task_id": bindings.task_id, "parent_sha256": bindings.expected_sha256,
        "policy_sha256": _sha(_encoded(policy.to_dict()).decode()),
        "provider": provider, "model": model,
        "packet_prompt_sha256": _sha(projected),
        "delivered_content_ids": [ref.referenced_content_id for ref in delivered],
    }
    if retrieval_query_reference is not None:
        base["retrieval_query_reference"] = retrieval_query_reference
    binding = _sha(_encoded(base).decode())
    instruction = (
        "Supervisor evidence dialogue. Return exactly one JSON object using the round bindings above. "
        f"To request 1-4 deferred optional capsule IDs, use schema {REQUEST_SCHEMA} and fields "
        "session_id,generation,binding_sha256,capsule_ids. Request only bodies not yet delivered. "
        f"To finish, use schema {FINAL_SCHEMA} and fields session_id,generation,binding_sha256,text; "
        "text must contain the literal final task response. No extra fields, prose or code fences. "
        "A request grants no tools or permissions. Capsules refer to the canonical source baseline; "
        "read current native source before edits. Task, scope and acceptance below remain authoritative.\n"
    )
    prompt = _encoded({**base, "binding_sha256": binding}).decode() + "\n" + instruction + projected
    if delivered:
        prompt += "\nSupervisor verified capsule deliveries:\n" + _encoded([ref.to_dict() for ref in delivered]).decode()
    return prompt, binding


def run_semantic_decision_dialogue(
    *, native_prompt: str, repository: Path, workspace: Path,
    policy: SemanticDecisionPolicy, provider: str, model: str, invoke,
    count_tokens: PromptTokenCounter, max_calls: int = 3, timeout: int = 90,
    max_output_tokens: int = 4096, max_input_tokens: int | None = None,
    max_total_input_tokens: int | None = None,
    max_expansion_bytes: int = 16384, max_expansion_tokens: int = 4096, cancelled=None,
    retrieval_query_binding=None,
) -> SemanticDecisionResult:
    """Deliver selected context through a pinned callback, retaining every attempt.

    invoke must enforce the admitted provider/workspace profile and its physical
    process timeout. Cancellation here prevents further calls/delivery; it does
    not claim to terminate an already running provider. Fresh rounds replay the
    full required packet plus all delivered capsules and charge that replay.
    """
    if (provider not in {"codex_cli", "grok_cli"} or type(policy) is not SemanticDecisionPolicy
            or type(count_tokens) is not PromptTokenCounter
            or count_tokens.provider != provider or count_tokens.model != model
            or not callable(invoke) or (cancelled is not None and not callable(cancelled))):
        raise ValueError("trusted policy, pinned provider counter and invocation callback required")
    if retrieval_query_binding is not None:
        from .semantic_retrieval_query_reference import SemanticRetrievalQueryBinding
        if (type(retrieval_query_binding) is not SemanticRetrievalQueryBinding
                or policy.task_text_source_path is None):
            raise ValueError("trusted retrieval binding and task-text reference policy required")
    _text(model, "model")
    for value, maximum in ((max_calls, 4), (timeout, 600), (max_output_tokens, 16384),
                           (max_expansion_bytes, 65536), (max_expansion_tokens, 128000)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("positive bounded dialogue settings required")
    for limit in (max_input_tokens, max_total_input_tokens):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("positive complete prompt token budgets required")
    root, work = Path(repository).absolute(), Path(workspace).absolute()
    if root.resolve(strict=True) != root or work.resolve(strict=True) != work:
        raise ValueError("canonical existing repository and workspace required")
    started, session = time.monotonic(), uuid.uuid4().hex
    attempts, deliveries, delivered_ids = [], [], set()
    observed_counts = {}
    verify_held_context = None
    ledger = {
        "schema": SCHEMA, "session_id": session, "provider": provider, "model": model,
        "native_prompt_sha256": _sha(native_prompt), "workspace": str(work),
        "policy_sha256": _sha(_encoded(policy.to_dict()).decode()),
        "attempts": attempts, "expansions": [], "max_calls": max_calls,
        "timeout_seconds": timeout, "max_input_tokens": max_input_tokens,
        "max_total_input_tokens": max_total_input_tokens,
        "total_input_budget_scope": "sum of complete rendered user prompts before each invocation",
        "total_input_budget_basis": "estimated_content_tokens" if count_tokens.count_kind == "estimate"
                                    else "local_tokenizer_content_tokens",
        "token_count_kind": count_tokens.count_kind, "tokenizer_binding": count_tokens.binding(),
        "tokenizer_compatibility_basis": count_tokens.compatibility_basis,
        "status": "preparing", "selected_representation": None,
        "token_budget_guaranteed": False, "provider_output_token_cap_enforced": False,
        "provider_cancellation_guaranteed": False,
        "whole_session_token_savings": None, "completion_authority": False,
    }

    def checkpoint():
        if cancelled is not None:
            value = cancelled()
            if type(value) is not bool or value:
                raise RuntimeError("cancelled")
        if time.monotonic() - started >= timeout:
            raise RuntimeError("deadline_exceeded")

    def measure(text):
        if verify_held_context is not None:
            checkpoint()
            verify_held_context()
        amount = count_tokens.count(text)
        digest = _sha(text)
        if digest in observed_counts and observed_counts[digest] != amount:
            raise RuntimeError("tokenizer_inconsistent")
        observed_counts[digest] = amount
        if verify_held_context is not None:
            checkpoint()
            verify_held_context()
        return amount

    def finish(status, reason=None):
        ledger.update(status=status, seconds=time.monotonic() - started,
                      invocation_attempt_count=len(attempts), failure_reason=reason)
        receipts = [row["provider_receipt"] for row in attempts if row.get("provider_receipt_bound")]
        ledger["missing_provider_receipt_count"] = len(attempts) - len(receipts)
        dispatch = [(row.get("provider_receipt") or {}).get("provider_called") for row in attempts]
        ledger["provider_dispatches_reported"] = sum(value is True for value in dispatch)
        ledger["provider_dispatch_status_unknown_count"] = sum(type(value) is not bool for value in dispatch)
        if provider == "codex_cli":
            from .cumulative_usage_reconciliation import reconcile_router_cumulative_usage
            native = [row.get("native_invocation", row) for row in receipts]
            ledger["codex_cumulative_usage"] = reconcile_router_cumulative_usage(native)
        ledger["provider_usage_complete"] = bool(attempts) and not ledger["missing_provider_receipt_count"] and (
            ledger.get("codex_cumulative_usage", {}).get("supplied_receipt_usage_complete") is True)
        ledger["local_input_content_token_sum"] = sum(row["content_tokens"] for row in attempts)

    def retain_receipt(attempt, raw, call):
        attempt["provider_receipt_bound"] = False
        try:
            _structure(raw, 65536)
            if type(raw) is not dict or len(_encoded(raw)) > 65536:
                raise ValueError("bounded provider receipt required")
            attempt["provider_receipt"] = json.loads(_encoded(raw))
        except Exception as error:
            attempt["provider_receipt_error_type"] = type(error).__name__
            return False
        expected = (("provider", provider), ("model", model),
                    ("model_prompt_sha256", call.model_prompt_sha256))
        bound = all(raw.get(key) == value for key, value in expected)
        if "native_invocation" in raw:
            nested = raw["native_invocation"]
            bound = bound and type(nested) is dict and all(nested.get(key) == value for key, value in expected)
            bound = bound and nested.get("workspace") == str(work)
        attempt["provider_receipt_bound"] = bound
        return bound

    try:
        checkpoint()
        projection_options = {}
        if retrieval_query_binding is not None:
            projection_options["retrieval_query_binding"] = retrieval_query_binding
        projection = project_semantic_router_prompt(
            prompt=native_prompt, repository=root, decision_policy=policy, **projection_options)
        _, _, _, _, text, _ = codec._prompt_parts(projection.projected_prompt)
        view = json.loads(text)
        bindings = SemanticProjectionBindings(**view["origin"])
        ledger.update(task_id=bindings.task_id, parent_sha256=bindings.expected_sha256,
                      packet_prompt_sha256=_sha(projection.projected_prompt))
        round_options = {}
        if retrieval_query_binding is not None:
            from .semantic_retrieval_query_reference import load_live_retrieval_query_context
            original_text = codec._prompt_parts(native_prompt)[4]
            observed = projection.receipt["retrieval_query_reference"]
            held = {name: observed[name] for name in (
                "binding", "native_context_sha256", "view_sha256", "native_reference_records_sha256")}
            if (held["binding"] != retrieval_query_binding.to_dict()
                    or observed.get("source_freshness_verified") is not True
                    or observed.get("historical_replay") is not False):
                raise ValueError("retrieval projection differs from held raw parent binding")

            def verify_retrieval_pair():
                _verify_current(root, bindings)
                current = load_live_retrieval_query_context(
                    repository=root, binding=retrieval_query_binding,
                    semantic_original_text=original_text, semantic_bindings=bindings)
                if _sha(current) != held["native_context_sha256"]:
                    raise ValueError("retrieval query context changed")

            verify_held_context = verify_retrieval_pair
            ledger["retrieval_query_reference"] = held
            verify_held_context()
            round_options["retrieval_query_reference"] = held
        first, _ = _round_prompt(projection.projected_prompt, session, 0, bindings, policy, provider, model, [],
                                 **round_options)
        native_model, _ = render_model_prompt(prompt=native_prompt, purpose="coding", workspace=work)
        first_model, _ = render_model_prompt(prompt=first, purpose="coding", workspace=work)
        baseline_count, candidate_count = measure(native_model), measure(first_model)
        native_fits = len(native_model.encode()) <= MAX_MODEL_PROMPT_BYTES and (
            max_input_tokens is None or baseline_count <= max_input_tokens) and (
            max_total_input_tokens is None or baseline_count <= max_total_input_tokens)
        native_fallback = native_fits and baseline_count <= candidate_count
        ledger.update(selected_representation="native" if native_fallback else "decision_packet",
                      native_content_tokens=baseline_count, packet_first_content_tokens=candidate_count)
        for generation in range(max_calls):
            checkpoint()
            if verify_held_context is None:
                _verify_current(root, bindings)
            else:
                verify_held_context()
            prompt, binding = _round_prompt(
                projection.projected_prompt, session, generation, bindings, policy, provider, model, deliveries,
                **round_options)
            if native_fallback:
                prompt = native_prompt
            model_prompt, _ = render_model_prompt(prompt=prompt, purpose="coding", workspace=work)
            amount = measure(model_prompt)
            if len(model_prompt.encode()) > MAX_MODEL_PROMPT_BYTES or (
                    max_input_tokens is not None and amount > max_input_tokens):
                raise RuntimeError("complete_prompt_budget_exceeded")
            if max_total_input_tokens is not None and (
                    sum(row["content_tokens"] for row in attempts) + amount > max_total_input_tokens):
                raise RuntimeError("total_input_budget_exceeded")
            remaining = math.floor(timeout - (time.monotonic() - started))
            if remaining < 1:
                raise RuntimeError("deadline_exceeded")
            checkpoint()
            # Counting may interleave with a source change too.
            if verify_held_context is None:
                _verify_current(root, bindings)
            else:
                verify_held_context()
            checkpoint()
            remaining = math.floor(timeout - (time.monotonic() - started))
            if remaining < 1:
                raise RuntimeError("deadline_exceeded")
            call = SemanticDecisionCall(
                uuid.uuid4().hex, session, generation, binding, provider, model, prompt,
                remaining, max_output_tokens, _sha(model_prompt),
            )
            attempt = {"call_id": call.call_id, "generation": generation, "status": "started",
                       "prompt_sha256": _sha(prompt), "prompt_bytes": len(prompt.encode()),
                       "model_prompt_sha256": call.model_prompt_sha256,
                       "model_prompt_bytes": len(model_prompt.encode()), "content_tokens": amount,
                       "content_token_count_kind": count_tokens.count_kind, "provider_receipt": None}
            attempts.append(attempt)
            for expansion in ledger["expansions"]:
                expansion["included_in_invocation_attempts"].append(call.call_id)
            try:
                reply = invoke(call)
            except Exception as error:
                attempt.update(status="provider_failed", error_type=type(error).__name__)
                if isinstance(error, SemanticDecisionProviderFailure):
                    retain_receipt(attempt, error.receipt, call)
                raise RuntimeError("provider_failed") from error
            if type(reply) is not SemanticDecisionReply:
                raise RuntimeError("invalid_provider_reply")
            if not retain_receipt(attempt, reply.receipt, call):
                raise RuntimeError("provider_receipt_binding_differs")
            attempt.update(status="provider_returned", response_sha256=_sha(reply.text))
            _observe_rejection(attempt, "post_return_control_check", checkpoint)
            if verify_held_context is None:
                _observe_rejection(attempt, "post_return_current_verification",
                                   lambda: _verify_current(root, bindings))
            else:
                _observe_rejection(attempt, "post_return_current_verification", verify_held_context)
            _observe_rejection(attempt, "post_return_control_check", checkpoint)
            if native_fallback:
                _observe_rejection(attempt, "native_fallback_response_validation",
                                   lambda: _validate_native_response(reply.text))
                finish("returned")
                return SemanticDecisionResult(reply.text, _encoded(ledger).decode())
            response = _observe_rejection(attempt, "response_contract_validation", lambda: _response(reply.text, call))
            if response["schema"] == FINAL_SCHEMA:
                finish("returned")
                return SemanticDecisionResult(response["text"], _encoded(ledger).decode())
            requested = tuple(response["capsule_ids"])
            if delivered_ids & set(requested):
                raise RuntimeError("repeated_capsule_request")
            if generation + 1 == max_calls:
                raise RuntimeError("call_budget_exhausted")
            checkpoint()
            if verify_held_context is not None:
                verify_held_context()
            handles = view["deferred_references"]
            if len(handles) != 1:
                raise RuntimeError("no_deferred_archive")
            expanded = expand_semantic_router_decision_packet(
                repository=root, view=view, reference_id=handles[0]["reference_id"],
                bindings=bindings, policy=policy, count_tokens=measure,
                capsule_ids=requested, max_bytes=max_expansion_bytes,
                max_tokens=max_expansion_tokens, cancelled=cancelled,
            )
            checkpoint()
            if verify_held_context is not None:
                verify_held_context()
            deliveries.append(expanded)
            delivered_ids.update(requested)
            ledger["expansions"].append({
                "after_generation": generation, "capsule_ids": list(requested),
                "referenced_content_id": expanded.referenced_content_id,
                "descriptor_sha256": hashlib.sha256(_encoded(expanded.to_dict())).hexdigest(),
                "descriptor_bytes": len(_encoded(expanded.to_dict())),
                "included_in_invocation_attempts": [],
            })
        raise RuntimeError("call_budget_exhausted")
    except Exception as error:
        reason = str(error) if type(error) is RuntimeError else type(error).__name__
        if attempts and attempts[-1]["status"] in {"started", "provider_returned"}:
            attempts[-1]["status"] = "rejected"
        finish("failed", reason)
        raise SemanticDecisionDialogueError(reason, ledger) from error
