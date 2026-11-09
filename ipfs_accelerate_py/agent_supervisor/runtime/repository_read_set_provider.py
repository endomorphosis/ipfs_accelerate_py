"""Inspect one freshly validated read set through the existing Grok backend.

One guarded backend callback is permitted. This is a text inspection, not a
Terminal Bench launch, editing sandbox, source lock, accepted task or native
request/token hard cap. Source checks are cooperative observations only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import threading
import uuid

from .repository_context_read_set import compile_repository_read_set
from .router_implementation_runner import MAX_MODEL_PROMPT_BYTES

SCHEMA = "repository-read-set-inspection@1"
MAX_RESPONSE_BYTES = 1024 * 1024
_USAGE_COUNTS = ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens", "num_turns")
_USAGE_IDS = ("model_id", "session_id", "thread_id", "request_id")


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha(raw):
    return sha256(raw).hexdigest()


def _bounded_usage(observed):
    result = {}
    for key in _USAGE_COUNTS:
        value = observed.get(key)
        if type(value) is int and 0 <= value < 1 << 63:
            result[key] = value
    for key in _USAGE_IDS:
        value = observed.get(key)
        if type(value) is str and 0 < len(value.encode("utf-8")) <= 256 and all(c.isprintable() for c in value):
            result[key] = value
    if type(observed.get("exit_code")) is int and -(1 << 31) <= observed["exit_code"] < 1 << 31:
        result["exit_code"] = observed["exit_code"]
    if type(observed.get("timed_out")) is bool:
        result["timed_out"] = observed["timed_out"]
    return result


@dataclass(frozen=True)
class ReadSetInspection:
    selected_prompt: str = field(repr=False)
    response_text: str = field(repr=False)
    receipt_json: str = field(repr=False)

    @property
    def receipt(self):
        return json.loads(self.receipt_json)


def inspect_repository_read_set(*, session, generation, query, required_paths, required_texts,
        task_id, provider, model, workspace, token_counter, max_input_tokens, limit=8,
        max_prompt_bytes=MAX_MODEL_PROMPT_BYTES, timeout=1800, max_output_tokens=2048,
        requirements_guard=None, receipt_observer=None) -> ReadSetInspection:
    """Compile, validate again at the backend seam, and inspect exactly once.

    The caller must execute from the declared source workspace. No global cwd,
    sandbox, authentication or provider environment is changed by this helper.
    An observer receives content-free final metadata even when an exception
    prevents a result; an existing provider failure keeps its exact identity.
    """
    _need(provider == "grok_cli" and model == "grok-4.7", "read_set_inspection_route_unsupported")
    _need(type(timeout) is int and 1 <= timeout <= 1800, "read_set_inspection_timeout_invalid")
    _need(type(max_output_tokens) is int and 1 <= max_output_tokens <= 16384,
        "read_set_inspection_output_budget_invalid")
    _need(requirements_guard is None or callable(requirements_guard), "read_set_requirements_guard_invalid")
    _need(receipt_observer is None or callable(receipt_observer), "read_set_receipt_observer_invalid")
    _need(type(required_paths) in (list, tuple) and type(required_texts) in (list, tuple),
        "read_set_inspection_required_population_invalid")
    _need(isinstance(workspace, Path) and Path.cwd().resolve(strict=True) == workspace,
        "read_set_inspection_workspace_differs")
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.router_deps import RouterDeps
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation
    from ipfs_accelerate_py.cli_runtime.grok_native_usage import grok_outcome_receipt, grok_usage_receipt

    options = dict(session=session, generation=generation, query=query,
        required_paths=tuple(required_paths), required_texts=tuple(required_texts), task_id=task_id,
        provider=provider, model=model, workspace=workspace, token_counter=token_counter,
        max_input_tokens=max_input_tokens, limit=limit, max_prompt_bytes=max_prompt_bytes)
    invocation = uuid.uuid4().hex
    receipt = dict(schema=SCHEMA, invocation_id=invocation, provider=provider, model=model,
        status="not_started", failure_reason=None, read_set=None,
        model_prompt_sha256=None, model_prompt_bytes=None, response_sha256=None, response_bytes=None,
        timeout_seconds=timeout, requested_output_tokens=max_output_tokens,
        requested_output_token_limit=max_output_tokens, max_output_token_limit_guaranteed=False,
        provider_callback_attempts=0, backend_dispatches=0, backend_returned=False,
        predispatch_read_set_rechecked=False, closing_read_set_rechecked=False,
        requirements_guard_selected=requirements_guard is not None, closing_requirements_checked=False,
        usage=None, native_rollout_usage=None, native_provider_outcome=None,
        usage_capture_status="not_observed", receipt_observer_status="not_selected" if receipt_observer is None else "pending",
        native_max_turns=1, native_tool_allowlist=["read_file"], native_tool_denylist=["read_file", "search_tool", "use_tool"],
        native_permission_mode="dontAsk", effective_toolset_verified=False,
        local_fallback_allowed=False, cross_provider_fallback_allowed=False, automatic_retry_allowed=False,
        response_cache_allowed=False, provider_output_token_cap_enforced=False,
        source_guard_scope="cooperative_snapshot_checks_no_writer_lease",
        containment_verified=False, provider_delivery_verified=False, task_acceptance_verified=False,
        native_cumulative_cap_enforced=False, whole_task_usage_complete=None, total_session_token_savings=None,
        billing_total_verified=False, proof_authority=False, execution_authority=False, completion_authority=False)
    prepared = None
    response = None
    original_error = None
    closing_error = None

    def guard():
        if requirements_guard is not None:
            requirements_guard()

    def current():
        guard()
        fresh = compile_repository_read_set(**options)
        guard()
        if prepared is not None:
            _need(fresh.selected_prompt.encode("utf-8") == prepared.selected_prompt.encode("utf-8")
                and fresh.receipt == prepared.receipt, "read_set_inspection_current_input_differs")
        return fresh

    try:
        set_last_cli_observation(provider, {})
        prepared = current()
        receipt.update(read_set=prepared.receipt,
            model_prompt_sha256=_sha(prepared.selected_prompt.encode("utf-8")),
            model_prompt_bytes=len(prepared.selected_prompt.encode("utf-8")))
        deps = RouterDeps()
        backend = llm_router.get_llm_provider(provider, deps=deps, use_cache=False)
        _need(backend is not None and callable(getattr(backend, "generate", None)),
            "read_set_inspection_backend_unavailable")
        callback_gate = threading.Lock()

        class GuardedBackend:
            def generate(self, prompt, *, model_name=None, **kwargs):
                with callback_gate:
                    receipt["provider_callback_attempts"] += 1
                    _need(receipt["provider_callback_attempts"] == 1, "read_set_inspection_second_callback_refused")
                _need(type(prompt) is str and prompt == prepared.selected_prompt and model_name == model,
                    "read_set_inspection_backend_input_differs")
                current()
                _need(isinstance(workspace, Path) and Path.cwd().resolve(strict=True) == workspace,
                    "read_set_inspection_workspace_differs")
                receipt["predispatch_read_set_rechecked"] = True
                set_last_cli_observation(provider, {})
                receipt["backend_dispatches"] += 1
                value = backend.generate(prompt, model_name=model_name, **kwargs)
                receipt["backend_returned"] = True
                return value

        receipt["status"] = "provider_callback"
        response = llm_router.generate_text(prepared.selected_prompt, provider=provider, model_name=model,
            provider_instance=GuardedBackend(), deps=deps, allow_local_fallback=False,
            allow_cross_provider_fallback=False, disable_model_retry=True, side_effecting=True,
            task_kind="coding", allocation_path="cli", allocation_session_id=invocation,
            chat_session_id=invocation, resume_session_id="", continue_session=False,
            timeout=timeout, max_tokens=max_output_tokens, max_new_tokens=max_output_tokens,
            grok_tools="read_file", grok_disallowed_tools="read_file,search_tool,use_tool",
            grok_permission_mode="dontAsk", grok_max_turns=1, grok_stream_progress=True)
        _need(receipt["provider_callback_attempts"] == 1
            and receipt["backend_dispatches"] == 1 and receipt["backend_returned"] is True,
            "read_set_inspection_backend_not_observed")
        _need(type(response) is str and bool(response.strip()), "read_set_inspection_empty_response")
        raw = response.encode("utf-8")
        _need(len(raw) <= MAX_RESPONSE_BYTES, "read_set_inspection_response_bound")
        receipt.update(status="returned", response_sha256=_sha(raw), response_bytes=len(raw))
    except BaseException as error:
        original_error = error
        receipt.update(status="failed", failure_reason="inspection_failed")
        raise
    finally:
        try:
            observed = get_last_cli_observation(provider) if receipt["backend_dispatches"] else {}
            receipt.update(usage=_bounded_usage(observed), native_rollout_usage=grok_usage_receipt(observed),
                native_provider_outcome=grok_outcome_receipt(observed), usage_capture_status="observed")
        except BaseException as error:
            receipt["usage_capture_status"] = "unavailable"
            if original_error is None:
                closing_error = error
        try:
            if prepared is not None:
                current()
                receipt["closing_read_set_rechecked"] = True
            else:
                guard()
            receipt["closing_requirements_checked"] = requirements_guard is not None
        except BaseException as error:
            if original_error is None and closing_error is None:
                closing_error = error
            receipt.update(status="failed" if original_error is not None else "refused",
                failure_reason="closing_validation_failed")
        if closing_error is not None and receipt["failure_reason"] is None:
            receipt.update(status="refused", failure_reason="usage_capture_failed")
        if receipt_observer is not None:
            try:
                receipt["receipt_observer_status"] = "called"
                receipt_observer(json.loads(_json(receipt)))
            except BaseException as error:
                receipt["receipt_observer_status"] = "failed"
                if original_error is None and closing_error is None:
                    closing_error = error
                    receipt.update(status="failed", failure_reason="receipt_observer_failed")
        if original_error is None and closing_error is not None:
            raise closing_error
    return ReadSetInspection(prepared.selected_prompt, response, _json(receipt))
