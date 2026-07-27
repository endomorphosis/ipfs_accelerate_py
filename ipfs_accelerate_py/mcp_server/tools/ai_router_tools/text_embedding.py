"""Bounded MCP tools for canonical text and embedding routers.

The catalog owns selection and the modality routers own invocation.  This
module deliberately does not instantiate providers, inspect credentials, or
fall back to the legacy MCP inference and embedding implementations.
"""

from __future__ import annotations

import inspect
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import anyio

from ipfs_accelerate_py import embeddings_router, llm_router


AI_ROUTER_TOOL_SCHEMA_VERSION = "ai.router.mcp.v1"
AI_ROUTER_RECEIPT_SCHEMA_VERSION = "ai.router.receipt.v1"

MAX_INPUT_ITEMS = 128
MAX_TEXT_BYTES = 1_048_576
MAX_TEXT_ITEM_BYTES = 262_144
MAX_EMBEDDING_DIMENSIONS = 16_384
MAX_OUTPUT_BYTES = 4_194_304
MAX_TIMEOUT_SECONDS = 120.0
MAX_STREAM_CHUNKS = 1_024
MAX_RECEIPT_CANDIDATES = 16
MAX_POLICY_ENTRIES = 64
MAX_SELECTOR_BYTES = 256

_TEXT_OPERATION = "text.generate"
_EMBEDDING_OPERATION = "embedding.generate"
_TEXT_ROUTER = "llm_router"
_EMBEDDING_ROUTER = "embeddings_router"


class _RequestError(ValueError):
    """A request violates this MCP tool's bounded public contract."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        schema_version: Optional[str] = None,
        catalog_revision: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.schema_version = schema_version
        self.catalog_revision = catalog_revision


def _error_result(
    code: str,
    message: str,
    *,
    schema_version: Optional[str] = None,
    catalog_revision: Optional[str] = None,
    selected_binding: Optional[Dict[str, Any]] = None,
    receipt: Optional[Dict[str, Any]] = None,
    cause: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a stable error without reflecting prompts, policy values, or secrets."""

    error: Dict[str, Any] = {"code": code, "message": message}
    if cause:
        error["cause"] = str(cause)[:128]
    result: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "tool_schema_version": AI_ROUTER_TOOL_SCHEMA_VERSION,
        "schema_version": schema_version,
        "catalog_revision": catalog_revision,
        "error": error,
        "error_code": code,
        "error_type": code,
    }
    if selected_binding is not None:
        result["selected_binding"] = selected_binding
    if receipt is not None:
        result["receipt"] = receipt
    return result


def _bounded_selector(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _RequestError(
            "invalid_request",
            "%s must be a non-empty string when provided." % field_name,
        )
    selected = value.strip()
    if len(selected.encode("utf-8")) > MAX_SELECTOR_BYTES:
        raise _RequestError(
            "input_limit_exceeded",
            "%s exceeds the maximum encoded size." % field_name,
        )
    return selected


def _bounded_policy(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping) or len(value) > MAX_POLICY_ENTRIES:
        raise _RequestError(
            "invalid_request",
            "policy must be a bounded object.",
        )
    # ResolutionRequest performs canonical key and scalar validation.  Copying
    # here prevents a caller from mutating the constraints during resolution.
    return dict(value)


def _bounded_timeout(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= MAX_TIMEOUT_SECONDS
    ):
        raise _RequestError(
            "invalid_request",
            "timeout must be greater than zero and no more than 120 seconds.",
        )
    return float(value)


def _bounded_output_limit(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_OUTPUT_BYTES
    ):
        raise _RequestError(
            "invalid_request",
            "max_output_bytes is outside the supported range.",
        )
    return value


def _validate_streaming(stream: Any, max_stream_chunks: Any) -> None:
    if not isinstance(stream, bool):
        raise _RequestError("invalid_request", "stream must be a boolean.")
    if (
        isinstance(max_stream_chunks, bool)
        or not isinstance(max_stream_chunks, int)
        or not 1 <= max_stream_chunks <= MAX_STREAM_CHUNKS
    ):
        raise _RequestError(
            "invalid_request",
            "max_stream_chunks is outside the supported range.",
        )
    # llm_router.generate_text has a string-in/string-out contract.  A caller
    # asking for a stream must use a future explicitly streaming router method;
    # silently buffering an unbounded iterator would violate the MCP contract.
    if stream:
        raise _RequestError(
            "streaming_unsupported",
            "This canonical router operation does not expose streaming output.",
        )


def _validate_prompt(prompt: Any) -> Tuple[str, int]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise _RequestError(
            "invalid_request",
            "prompt must be a non-empty string.",
        )
    size = len(prompt.encode("utf-8"))
    if size > MAX_TEXT_ITEM_BYTES or size > MAX_TEXT_BYTES:
        raise _RequestError(
            "input_limit_exceeded",
            "prompt exceeds the maximum encoded text size.",
        )
    return prompt, size


def _validate_texts(texts: Any) -> Tuple[List[str], int]:
    if (
        isinstance(texts, (str, bytes, Mapping))
        or not isinstance(texts, Sequence)
        or not texts
        or len(texts) > MAX_INPUT_ITEMS
    ):
        raise _RequestError(
            "input_limit_exceeded",
            "texts must contain between 1 and %d items." % MAX_INPUT_ITEMS,
        )
    result: List[str] = []
    total = 0
    for item in texts:
        if not isinstance(item, str) or not item.strip():
            raise _RequestError(
                "invalid_request",
                "texts must contain only non-empty strings.",
            )
        size = len(item.encode("utf-8"))
        if size > MAX_TEXT_ITEM_BYTES:
            raise _RequestError(
                "input_limit_exceeded",
                "a text item exceeds the maximum encoded size.",
            )
        total += size
        if total > MAX_TEXT_BYTES:
            raise _RequestError(
                "input_limit_exceeded",
                "texts exceed the maximum total encoded size.",
            )
        result.append(item)
    return result, total


def _validate_dimensions(value: Any) -> Optional[int]:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_EMBEDDING_DIMENSIONS
    ):
        raise _RequestError(
            "dimension_limit_exceeded",
            "dimensions is outside the supported range.",
        )
    return value


def _snapshot_versions(snapshot: Any) -> Tuple[str, str]:
    schema_version = getattr(snapshot, "schema_version", None)
    revision = getattr(snapshot, "revision", None)
    if not isinstance(schema_version, str) or not schema_version:
        raise TypeError("catalog snapshot has no schema version")
    if not isinstance(revision, str) or not revision:
        raise TypeError("catalog snapshot has no revision")
    return schema_version, revision


def _selector_matches(record: Any, selector: Optional[str], identity: str) -> bool:
    if selector is None:
        return True
    wanted = selector.strip().casefold()
    values = {
        str(getattr(record, identity, "") or "").casefold(),
        str(getattr(record, "name", "") or "").casefold(),
    }
    values.update(str(item).casefold() for item in getattr(record, "aliases", ()))
    return wanted in values


def _safe_binding(binding: Any) -> Dict[str, Any]:
    """Project only stable routing identity, never labels or provenance."""

    operations = [
        str(getattr(item, "value", item))
        for item in tuple(getattr(binding, "operations", ()) or ())
    ]
    return {
        "binding_id": getattr(binding, "binding_id", None),
        "router": getattr(binding, "router", None),
        "provider_id": getattr(binding, "provider_id", None),
        "model_id": getattr(binding, "model_id", None),
        "deployment_id": getattr(binding, "deployment_id", None),
        "operations": operations[:16],
    }


def _candidate_capabilities(candidate: Any, operation: str) -> Iterable[Any]:
    for record in (
        getattr(candidate, "deployment", None),
        getattr(candidate, "model", None),
        getattr(candidate, "provider", None),
    ):
        for capability in tuple(getattr(record, "capabilities", ()) or ()):
            operations = {
                str(getattr(item, "value", item))
                for item in tuple(getattr(capability, "operations", ()) or ())
            }
            if operation in operations:
                yield capability


def _candidate_limit(candidate: Any, operation: str, field_name: str) -> Optional[int]:
    values = [
        getattr(capability, field_name, None)
        for capability in _candidate_capabilities(candidate, operation)
    ]
    known = [item for item in values if isinstance(item, int) and item > 0]
    return min(known) if known else None


def _invocation_provider(candidate: Any) -> str:
    binding_labels = dict(getattr(candidate.binding, "labels", ()) or ())
    provider_labels = dict(getattr(candidate.provider, "labels", ()) or ())
    return str(
        binding_labels.get(
            "invocation_provider",
            provider_labels.get("invocation_provider", candidate.provider.name),
        )
    )


def _invocation_model(candidate: Any) -> Optional[str]:
    model = getattr(candidate, "model", None)
    if model is None:
        return None
    binding_labels = dict(getattr(candidate.binding, "labels", ()) or ())
    model_labels = dict(getattr(model, "labels", ()) or ())
    return str(
        binding_labels.get(
            "invocation_model",
            model_labels.get(
                "invocation_model",
                model_labels.get("router_model_name", model.name),
            ),
        )
    )


def _resolve_candidates(
    *,
    router_name: str,
    operation: str,
    service: Optional[str],
    model: Optional[str],
    provider: Optional[str],
    policy: Optional[Dict[str, Any]],
    device: Optional[str],
) -> Tuple[Any, str, str, List[Any], int]:
    """Resolve all invocation constraints against one immutable snapshot."""

    from ipfs_accelerate_py.model_manager import get_default_model_manager

    manager = get_default_model_manager()
    snapshot = manager.snapshot()
    schema_version, revision = _snapshot_versions(snapshot)
    # "service" is the provider/service selector used by catalog list_services.
    # If both spellings are supplied, resolve one and verify the other against
    # the selected canonical provider identity below.
    provider_selector = service if service is not None else provider
    try:
        resolution = manager.resolve(
            operation=operation,
            modality="text",
            model=model,
            provider=provider_selector,
            policy=policy,
            device=device,
            configured=True,
            authorized=True,
            routable=True,
            limit=MAX_RECEIPT_CANDIDATES,
            snapshot=snapshot,
        )
    except (TypeError, ValueError) as exc:
        raise _RequestError(
            "invalid_request",
            "The catalog resolution constraints are invalid.",
            schema_version=schema_version,
            catalog_revision=revision,
        ) from exc
    if getattr(resolution, "snapshot_revision", None) != revision:
        raise _RequestError(
            "catalog_revision_mismatch",
            "Catalog resolution did not use the captured revision.",
            schema_version=schema_version,
            catalog_revision=revision,
        )
    candidates = [
        candidate
        for candidate in tuple(getattr(resolution, "candidates", ()) or ())
        if str(getattr(candidate.binding, "router", "")).casefold() == router_name
        and _selector_matches(candidate.provider, service, "provider_id")
        and _selector_matches(candidate.provider, provider, "provider_id")
    ]
    if not candidates:
        constrained = policy is not None or service is not None or provider is not None
        code = "selection_denied" if constrained else "no_match"
        raise _RequestError(
            code,
            (
                "No authorized router binding satisfies all requested constraints."
                if constrained
                else "No canonical router binding is available for this operation."
            ),
            schema_version=schema_version,
            catalog_revision=revision,
        )
    return snapshot, schema_version, revision, candidates, int(
        getattr(resolution, "total_candidates", len(candidates))
    )


def _receipt(
    *,
    revision: str,
    operation: str,
    candidates: Sequence[Any],
    total_candidates: int,
    selected: Any,
    allow_fallback: bool,
    fallback_used: bool,
    input_count: int,
    input_bytes: int,
    output_bytes: int,
    dimensions: Optional[int] = None,
) -> Dict[str, Any]:
    candidate_ids = [
        str(getattr(item.binding, "binding_id", ""))
        for item in candidates[:MAX_RECEIPT_CANDIDATES]
    ]
    return {
        "schema_version": AI_ROUTER_RECEIPT_SCHEMA_VERSION,
        "catalog_revision": revision,
        "operation": operation,
        "selected_binding_id": str(
            getattr(selected.binding, "binding_id", "")
        ),
        "candidate_binding_ids": candidate_ids,
        "candidate_count": min(max(total_candidates, len(candidates)), 1_000),
        "candidate_count_truncated": total_candidates > len(candidate_ids),
        "fallback": {
            "allowed": bool(allow_fallback),
            "used": bool(fallback_used),
            "boundary_binding_ids": candidate_ids,
        },
        "input": {
            "count": input_count,
            "text_bytes": input_bytes,
        },
        "output": {
            "bytes": output_bytes,
            "dimensions": dimensions,
        },
    }


def _trace_matches(candidate: Any, trace: Mapping[str, Any], *, embedding: bool) -> bool:
    provider_key = "provider_used" if embedding else "effective_provider_name"
    model_key = "model_name" if embedding else "effective_model_name"
    actual_provider = str(trace.get(provider_key, "") or "").strip().casefold()
    actual_model = str(trace.get(model_key, "") or "").strip().casefold()
    provider_values = {
        _invocation_provider(candidate).casefold(),
        str(candidate.provider.name).casefold(),
        str(candidate.provider.provider_id).casefold(),
    }
    provider_values.update(str(item).casefold() for item in candidate.provider.aliases)
    model = getattr(candidate, "model", None)
    model_values = {str(_invocation_model(candidate) or "").casefold()}
    if model is not None:
        model_values.update(
            {
                str(model.name).casefold(),
                str(model.model_id).casefold(),
                *(str(item).casefold() for item in model.aliases),
            }
        )
    provider_ok = not actual_provider or actual_provider in provider_values
    model_ok = not actual_model or actual_model in model_values
    return provider_ok and model_ok


def _select_effective_candidate(
    candidates: Sequence[Any],
    trace: Mapping[str, Any],
    *,
    embedding: bool,
    allow_fallback: bool,
) -> Tuple[Any, bool]:
    selected = candidates[0]
    matches = [
        candidate
        for candidate in candidates
        if _trace_matches(candidate, trace, embedding=embedding)
    ]
    effective = matches[0] if matches else None
    trace_fallback = bool(trace.get("fallback_used", False))
    fallback_used = trace_fallback or (
        effective is not None
        and effective.binding.binding_id != selected.binding.binding_id
    )
    if effective is None or (fallback_used and not allow_fallback):
        raise _RequestError(
            "fallback_boundary_exceeded",
            "The router selected a binding outside the permitted fallback boundary.",
        )
    return effective, fallback_used


async def _invoke_with_timeout(callback: Any, timeout: float) -> Tuple[Any, Dict[str, Any]]:
    """Run a synchronous canonical router call with timeout and cancellation."""

    def invoke() -> Tuple[Any, Any, Dict[str, Any]]:
        value, trace_getter = callback()
        if inspect.isawaitable(value):
            return value, trace_getter, {}
        trace = trace_getter() if callable(trace_getter) else {}
        return (
            value,
            None,
            dict(trace) if isinstance(trace, Mapping) else {},
        )

    with anyio.fail_after(timeout):
        run_sync_parameters = inspect.signature(anyio.to_thread.run_sync).parameters
        if "abandon_on_cancel" in run_sync_parameters:
            result, deferred_trace_getter, trace = await anyio.to_thread.run_sync(
                invoke,
                abandon_on_cancel=True,
            )
        else:  # pragma: no cover - compatibility with AnyIO 3.
            result, deferred_trace_getter, trace = await anyio.to_thread.run_sync(
                invoke,
                cancellable=True,
            )
        if inspect.isawaitable(result):
            result = await result
            deferred_trace = (
                deferred_trace_getter()
                if callable(deferred_trace_getter)
                else {}
            )
            trace = (
                dict(deferred_trace)
                if isinstance(deferred_trace, Mapping)
                else {}
            )
        return result, trace


def _router_error(
    exc: BaseException,
    *,
    schema_version: Optional[str],
    revision: Optional[str],
    selected: Optional[Any] = None,
    receipt: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    selected_binding = None if selected is None else _safe_binding(selected.binding)
    if isinstance(exc, _RequestError):
        return _error_result(
            exc.code,
            exc.safe_message,
            schema_version=schema_version,
            catalog_revision=revision,
            selected_binding=selected_binding,
            receipt=receipt,
        )
    if isinstance(exc, TimeoutError):
        return _error_result(
            "timeout",
            "The canonical router call exceeded its bounded timeout.",
            schema_version=schema_version,
            catalog_revision=revision,
            selected_binding=selected_binding,
            receipt=receipt,
        )
    return _error_result(
        "router_error",
        "The canonical router could not complete the request.",
        schema_version=schema_version,
        catalog_revision=revision,
        selected_binding=selected_binding,
        receipt=receipt,
        cause=type(exc).__name__,
    )


async def llm_generate(
    prompt: str,
    service: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    policy: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    timeout: float = 30.0,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
    allow_fallback: bool = False,
    stream: bool = False,
    max_stream_chunks: int = MAX_STREAM_CHUNKS,
) -> Dict[str, Any]:
    """Resolve and invoke ``llm_router.generate_text`` with bounded I/O."""

    schema_version: Optional[str] = None
    revision: Optional[str] = None
    selected: Optional[Any] = None
    try:
        prompt_value, input_bytes = _validate_prompt(prompt)
        service_value = _bounded_selector(service, "service")
        model_value = _bounded_selector(model, "model")
        provider_value = _bounded_selector(provider, "provider")
        device_value = _bounded_selector(device, "device")
        policy_value = _bounded_policy(policy)
        timeout_value = _bounded_timeout(timeout)
        output_limit = _bounded_output_limit(max_output_bytes)
        _validate_streaming(stream, max_stream_chunks)
        if not isinstance(allow_fallback, bool):
            raise _RequestError(
                "invalid_request",
                "allow_fallback must be a boolean.",
            )
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 1 <= max_tokens <= 1_000_000
        ):
            raise _RequestError(
                "invalid_request",
                "max_tokens is outside the supported range.",
            )
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or not 0 <= float(temperature) <= 2
        ):
            raise _RequestError(
                "invalid_request",
                "temperature must be between 0 and 2.",
            )
        _, schema_version, revision, candidates, total = _resolve_candidates(
            router_name=_TEXT_ROUTER,
            operation=_TEXT_OPERATION,
            service=service_value,
            model=model_value,
            provider=provider_value,
            policy=policy_value,
            device=device_value,
        )
        selected = candidates[0]
        catalog_input_limit = _candidate_limit(
            selected, _TEXT_OPERATION, "max_input_bytes"
        )
        if catalog_input_limit is not None and input_bytes > catalog_input_limit:
            raise _RequestError(
                "input_limit_exceeded",
                "prompt exceeds the selected service input limit.",
            )
        catalog_output_limit = _candidate_limit(
            selected, _TEXT_OPERATION, "max_output_bytes"
        )
        if catalog_output_limit is not None:
            output_limit = min(output_limit, catalog_output_limit)
        invocation_provider = _invocation_provider(selected)
        invocation_model = _invocation_model(selected)

        def call() -> Tuple[Any, Any]:
            value = llm_router.generate_text(
                prompt_value,
                model_name=invocation_model,
                provider=invocation_provider,
                allow_local_fallback=False,
                disable_model_retry=True,
                max_tokens=max_tokens,
                temperature=float(temperature),
            )
            return value, getattr(llm_router, "get_last_generation_trace", None)

        generated, trace = await _invoke_with_timeout(call, timeout_value)
        if not isinstance(generated, str):
            raise _RequestError(
                "invalid_router_output",
                "The text router returned a non-text result.",
            )
        output_bytes = len(generated.encode("utf-8"))
        if output_bytes > output_limit:
            raise _RequestError(
                "output_limit_exceeded",
                "The generated text exceeds the bounded output size.",
            )
        effective, fallback_used = _select_effective_candidate(
            candidates,
            trace,
            embedding=False,
            allow_fallback=allow_fallback,
        )
        selected = effective
        receipt = _receipt(
            revision=revision,
            operation=_TEXT_OPERATION,
            candidates=candidates,
            total_candidates=total,
            selected=effective,
            allow_fallback=allow_fallback,
            fallback_used=fallback_used,
            input_count=1,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
        )
        return {
            "status": "success",
            "success": True,
            "tool_schema_version": AI_ROUTER_TOOL_SCHEMA_VERSION,
            "schema_version": schema_version,
            "catalog_revision": revision,
            "text": generated,
            "selected_binding": _safe_binding(effective.binding),
            "receipt": receipt,
            "streaming": {
                "requested": False,
                "supported": False,
                "mode": "buffered",
                "max_chunks": max_stream_chunks,
            },
        }
    except BaseException as exc:
        # Cancellation is control flow, not a router error envelope.
        if not isinstance(exc, Exception):
            raise
        if isinstance(exc, _RequestError):
            schema_version = exc.schema_version or schema_version
            revision = exc.catalog_revision or revision
        return _router_error(
            exc,
            schema_version=schema_version,
            revision=revision,
            selected=selected,
        )


def _normalize_vectors(
    value: Any,
    *,
    expected_count: int,
    expected_dimensions: Optional[int],
    output_limit: int,
) -> Tuple[List[List[float]], int, int]:
    if (
        isinstance(value, (str, bytes, Mapping))
        or not isinstance(value, Sequence)
        or len(value) != expected_count
    ):
        raise _RequestError(
            "invalid_router_output",
            "The embeddings router returned an invalid item count.",
        )
    vectors: List[List[float]] = []
    dimension: Optional[int] = None
    for vector in value:
        if (
            isinstance(vector, (str, bytes, Mapping))
            or not isinstance(vector, Sequence)
        ):
            raise _RequestError(
                "invalid_router_output",
                "The embeddings router returned a malformed vector.",
            )
        if not vector or len(vector) > MAX_EMBEDDING_DIMENSIONS:
            raise _RequestError(
                "dimension_limit_exceeded",
                "An embedding vector has an unsupported dimension.",
            )
        normalized: List[float] = []
        for component in vector:
            if (
                isinstance(component, bool)
                or not isinstance(component, (int, float))
                or not math.isfinite(float(component))
            ):
                raise _RequestError(
                    "invalid_router_output",
                    "An embedding vector contains a non-finite numeric value.",
                )
            normalized.append(float(component))
        if dimension is None:
            dimension = len(normalized)
        elif len(normalized) != dimension:
            raise _RequestError(
                "invalid_router_output",
                "Embedding vectors do not have a uniform dimension.",
            )
        vectors.append(normalized)
    resolved_dimension = int(dimension or 0)
    if (
        expected_dimensions is not None
        and resolved_dimension != expected_dimensions
    ):
        raise _RequestError(
            "dimension_mismatch",
            "The embeddings router returned an unexpected dimension.",
        )
    output_bytes = len(
        json.dumps(vectors, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if output_bytes > output_limit:
        raise _RequestError(
            "output_limit_exceeded",
            "The embeddings exceed the bounded output size.",
        )
    return vectors, resolved_dimension, output_bytes


async def embeddings_generate(
    texts: List[str],
    service: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    policy: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    dimensions: Optional[int] = None,
    timeout: float = 30.0,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
    allow_fallback: bool = False,
    stream: bool = False,
    max_stream_chunks: int = MAX_STREAM_CHUNKS,
) -> Dict[str, Any]:
    """Resolve and invoke ``embeddings_router.embed_texts`` with bounded I/O."""

    schema_version: Optional[str] = None
    revision: Optional[str] = None
    selected: Optional[Any] = None
    try:
        text_values, input_bytes = _validate_texts(texts)
        service_value = _bounded_selector(service, "service")
        model_value = _bounded_selector(model, "model")
        provider_value = _bounded_selector(provider, "provider")
        device_value = _bounded_selector(device, "device")
        policy_value = _bounded_policy(policy)
        dimensions_value = _validate_dimensions(dimensions)
        timeout_value = _bounded_timeout(timeout)
        output_limit = _bounded_output_limit(max_output_bytes)
        _validate_streaming(stream, max_stream_chunks)
        if not isinstance(allow_fallback, bool):
            raise _RequestError(
                "invalid_request",
                "allow_fallback must be a boolean.",
            )
        _, schema_version, revision, candidates, total = _resolve_candidates(
            router_name=_EMBEDDING_ROUTER,
            operation=_EMBEDDING_OPERATION,
            service=service_value,
            model=model_value,
            provider=provider_value,
            policy=policy_value,
            device=device_value,
        )
        selected = candidates[0]
        catalog_batch_limit = _candidate_limit(
            selected, _EMBEDDING_OPERATION, "max_batch_size"
        )
        if catalog_batch_limit is not None and len(text_values) > catalog_batch_limit:
            raise _RequestError(
                "input_limit_exceeded",
                "texts exceed the selected service batch limit.",
            )
        catalog_input_limit = _candidate_limit(
            selected, _EMBEDDING_OPERATION, "max_input_bytes"
        )
        if catalog_input_limit is not None and input_bytes > catalog_input_limit:
            raise _RequestError(
                "input_limit_exceeded",
                "texts exceed the selected service input limit.",
            )
        catalog_output_limit = _candidate_limit(
            selected, _EMBEDDING_OPERATION, "max_output_bytes"
        )
        if catalog_output_limit is not None:
            output_limit = min(output_limit, catalog_output_limit)
        declared_dimensions = _candidate_limit(
            selected, _EMBEDDING_OPERATION, "embedding_dimensions"
        )
        expected_dimensions = dimensions_value or declared_dimensions
        invocation_provider = _invocation_provider(selected)
        invocation_model = _invocation_model(selected)

        def call() -> Tuple[Any, Any]:
            kwargs: Dict[str, Any] = {
                "model_name": invocation_model,
                "provider": invocation_provider,
                "device": device_value,
            }
            if dimensions_value is not None:
                kwargs["dimensions"] = dimensions_value
            value = embeddings_router.embed_texts(text_values, **kwargs)
            return value, getattr(
                embeddings_router, "get_last_embedding_trace", None
            )

        raw_vectors, trace = await _invoke_with_timeout(call, timeout_value)
        vectors, dimension, output_bytes = _normalize_vectors(
            raw_vectors,
            expected_count=len(text_values),
            expected_dimensions=expected_dimensions,
            output_limit=output_limit,
        )
        effective, fallback_used = _select_effective_candidate(
            candidates,
            trace,
            embedding=True,
            allow_fallback=allow_fallback,
        )
        selected = effective
        receipt = _receipt(
            revision=revision,
            operation=_EMBEDDING_OPERATION,
            candidates=candidates,
            total_candidates=total,
            selected=effective,
            allow_fallback=allow_fallback,
            fallback_used=fallback_used,
            input_count=len(text_values),
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            dimensions=dimension,
        )
        return {
            "status": "success",
            "success": True,
            "tool_schema_version": AI_ROUTER_TOOL_SCHEMA_VERSION,
            "schema_version": schema_version,
            "catalog_revision": revision,
            "embeddings": vectors,
            "count": len(vectors),
            "dimensions": dimension,
            "selected_binding": _safe_binding(effective.binding),
            "receipt": receipt,
            "streaming": {
                "requested": False,
                "supported": False,
                "mode": "buffered",
                "max_chunks": max_stream_chunks,
            },
        }
    except BaseException as exc:
        if not isinstance(exc, Exception):
            raise
        if isinstance(exc, _RequestError):
            schema_version = exc.schema_version or schema_version
            revision = exc.catalog_revision or revision
        return _router_error(
            exc,
            schema_version=schema_version,
            revision=revision,
            selected=selected,
        )


async def generate_text(
    prompt: str,
    model: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compatibility alias delegating to :func:`llm_generate`."""

    if model == "auto":
        model = None
    return await llm_generate(prompt=prompt, model=model, **kwargs)


async def generate_embeddings(
    texts: List[str],
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compatibility alias delegating to :func:`embeddings_generate`."""

    model = kwargs.pop("model", model_name)
    if model_name is not None and model is not None and model != model_name:
        return _error_result(
            "invalid_request",
            "model and model_name specify different values.",
        )
    return await embeddings_generate(texts=texts, model=model, **kwargs)


async def generate_embedding(
    text: str,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Single-item compatibility alias for canonical embedding generation."""

    return await generate_embeddings(
        texts=[text],
        model_name=model_name,
        **kwargs,
    )


def _constraint_schema() -> Dict[str, Any]:
    return {
        "service": {"type": "string", "minLength": 1, "maxLength": 256},
        "model": {"type": "string", "minLength": 1, "maxLength": 256},
        "provider": {"type": "string", "minLength": 1, "maxLength": 256},
        "policy": {
            "type": "object",
            "maxProperties": MAX_POLICY_ENTRIES,
            "additionalProperties": {
                "type": ["string", "number", "boolean"],
            },
        },
        "device": {"type": "string", "minLength": 1, "maxLength": 256},
        "timeout": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": MAX_TIMEOUT_SECONDS,
            "default": 30.0,
        },
        "max_output_bytes": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_OUTPUT_BYTES,
            "default": MAX_OUTPUT_BYTES,
        },
        "allow_fallback": {"type": "boolean", "default": False},
        "stream": {"type": "boolean", "default": False},
        "max_stream_chunks": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_STREAM_CHUNKS,
            "default": MAX_STREAM_CHUNKS,
        },
    }


def register_native_ai_router_tools(manager: Any) -> None:
    """Register canonical router tools and compatibility aliases."""

    text_properties = _constraint_schema()
    text_properties.update(
        {
            "prompt": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_TEXT_ITEM_BYTES,
            },
            "max_tokens": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1_000_000,
                "default": 512,
            },
            "temperature": {
                "type": "number",
                "minimum": 0,
                "maximum": 2,
                "default": 0.7,
            },
        }
    )
    embedding_properties = _constraint_schema()
    embedding_properties.update(
        {
            "texts": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_INPUT_ITEMS,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_TEXT_ITEM_BYTES,
                },
            },
            "dimensions": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_EMBEDDING_DIMENSIONS,
            },
        }
    )
    compatibility_embedding_properties = dict(embedding_properties)
    compatibility_embedding_properties["model_name"] = {
        "type": "string",
        "minLength": 1,
        "maxLength": 256,
    }
    single_embedding_properties = _constraint_schema()
    single_embedding_properties.update(
        {
            "text": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_TEXT_ITEM_BYTES,
            },
            "model_name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
            },
            "dimensions": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_EMBEDDING_DIMENSIONS,
            },
        }
    )
    registrations = (
        (
            "llm_generate",
            llm_generate,
            "Resolve one catalog revision and generate bounded text through llm_router.",
            text_properties,
            ["prompt"],
            (),
        ),
        (
            "embeddings_generate",
            embeddings_generate,
            "Resolve one catalog revision and generate bounded vectors through embeddings_router.",
            embedding_properties,
            ["texts"],
            (),
        ),
        (
            "generate_text",
            generate_text,
            "Compatibility alias for llm_generate.",
            text_properties,
            ["prompt"],
            ("compatibility",),
        ),
        (
            "generate_embeddings",
            generate_embeddings,
            "Compatibility alias for embeddings_generate.",
            compatibility_embedding_properties,
            ["texts"],
            ("compatibility",),
        ),
        (
            "generate_embedding",
            generate_embedding,
            "Single-item compatibility alias for embeddings_generate.",
            single_embedding_properties,
            ["text"],
            ("compatibility",),
        ),
    )
    for name, func, description, properties, required, extra_tags in registrations:
        manager.register_tool(
            category="ai_router_tools",
            name=name,
            func=func,
            description=description,
            input_schema={
                "type": "object",
                "properties": dict(properties),
                "required": list(required),
                "additionalProperties": False,
            },
            runtime="fastapi",
            tags=[
                "native",
                "mcpp",
                "ai-router",
                "catalog",
                "bounded",
                *extra_tags,
            ],
        )


# Consistent spelling for loaders that derive registrar names from categories.
register_native_ai_router_tool = register_native_ai_router_tools
register_ai_router_tools = register_native_ai_router_tools


__all__ = [
    "AI_ROUTER_RECEIPT_SCHEMA_VERSION",
    "AI_ROUTER_TOOL_SCHEMA_VERSION",
    "MAX_EMBEDDING_DIMENSIONS",
    "MAX_INPUT_ITEMS",
    "MAX_OUTPUT_BYTES",
    "MAX_STREAM_CHUNKS",
    "MAX_TEXT_BYTES",
    "MAX_TEXT_ITEM_BYTES",
    "MAX_TIMEOUT_SECONDS",
    "embeddings_generate",
    "generate_embedding",
    "generate_embeddings",
    "generate_text",
    "llm_generate",
    "register_native_ai_router_tool",
    "register_native_ai_router_tools",
    "register_ai_router_tools",
]
