"""Bounded MCP tools for canonical multimodal and voice routers.

Catalog resolution owns selection, the modality routers own provider
invocation, and a separately configured media loader owns URI and artifact
materialization.  This module never fetches remote media itself and never
returns media in receipts or routing metadata.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import inspect
import ipaddress
import math
import re
import struct
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.parse import urlsplit

import anyio

from ipfs_accelerate_py import multimodal_router, voice_router


AI_ROUTER_TOOL_SCHEMA_VERSION = "ai.router.mcp.v1"
AI_ROUTER_RECEIPT_SCHEMA_VERSION = "ai.router.receipt.v1"

MAX_MEDIA_ITEMS = 1
MAX_INLINE_MEDIA_BYTES = 8_388_608
MAX_MEDIA_BYTES = 33_554_432
MAX_TEXT_BYTES = 262_144
MAX_OUTPUT_BYTES = 8_388_608
MAX_MEDIA_DURATION_SECONDS = 600.0
MAX_SAMPLE_RATE_HZ = 192_000
MIN_SAMPLE_RATE_HZ = 8_000
MAX_IMAGE_WIDTH = 16_384
MAX_IMAGE_HEIGHT = 16_384
MAX_IMAGE_PIXELS = 40_000_000
MAX_TIMEOUT_SECONDS = 120.0
MAX_STREAM_CHUNKS = 1_024
MAX_RECEIPT_CANDIDATES = 16
MAX_POLICY_ENTRIES = 64
MAX_SELECTOR_BYTES = 256
MAX_URI_BYTES = 2_048
MAX_ARTIFACT_REF_BYTES = 1_024

_VISION_OPERATION = "vision.generate"
_TRANSCRIBE_OPERATION = "audio.transcribe"
_SYNTHESIZE_OPERATION = "audio.synthesize"
_MULTIMODAL_ROUTER = "multimodal_router"
_VOICE_ROUTER = "voice_router"

_IMAGE_MIME_TYPES = frozenset(
    {"image/gif", "image/jpeg", "image/png", "image/webp"}
)
_AUDIO_MIME_TYPES = frozenset(
    {
        "audio/aac",
        "audio/flac",
        "audio/mp3",
        "audio/mp4",
        "audio/mpeg",
        "audio/ogg",
        "audio/wav",
        "audio/webm",
        "audio/x-wav",
    }
)
_AUDIO_FORMATS = {
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/mp3": "mp3",
    "audio/mp4": "mp4",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/wav": "wav",
    "audio/webm": "webm",
    "audio/x-wav": "wav",
}
_MIME_RE = re.compile(r"^[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+$")


class MediaLoader(Protocol):
    """Allowlisted application media layer used for URI and artifact inputs."""

    def load(
        self,
        descriptor: Mapping[str, Any],
        *,
        max_bytes: int,
        timeout: float,
    ) -> Any:
        """Return bytes or a mapping containing ``data`` and safe metadata."""


_media_loader: Optional[MediaLoader] = None


def configure_media_loader(loader: Optional[MediaLoader]) -> None:
    """Configure the allowlisted media layer; ``None`` disables it.

    The loader is application authority, not request authority.  A URI request
    must additionally set ``allow_remote_media=True``.  The loader is expected
    to re-resolve and allowlist every redirect and DNS result.
    """

    global _media_loader
    if loader is not None and not callable(getattr(loader, "load", loader)):
        raise TypeError("media loader must be callable or expose load()")
    _media_loader = loader


class _RequestError(ValueError):
    """A request violates the bounded public contract."""

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
        raise _RequestError("invalid_request", "policy must be a bounded object.")
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


def _bounded_positive_int(value: Any, field_name: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise _RequestError(
            "invalid_request",
            "%s is outside the supported range." % field_name,
        )
    return value


def _bounded_positive_number(value: Any, field_name: str, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= maximum
    ):
        raise _RequestError(
            "invalid_request",
            "%s is outside the supported range." % field_name,
        )
    return float(value)


def _validate_streaming(stream: Any, max_stream_chunks: Any) -> None:
    if not isinstance(stream, bool):
        raise _RequestError("invalid_request", "stream must be a boolean.")
    _bounded_positive_int(
        max_stream_chunks, "max_stream_chunks", MAX_STREAM_CHUNKS
    )
    if stream:
        raise _RequestError(
            "streaming_unsupported",
            "This canonical router operation does not expose streaming output.",
        )


def _validate_text(value: Any, field_name: str) -> Tuple[str, int]:
    if not isinstance(value, str) or not value.strip():
        raise _RequestError(
            "invalid_request", "%s must be a non-empty string." % field_name
        )
    size = len(value.encode("utf-8"))
    if size > MAX_TEXT_BYTES:
        raise _RequestError(
            "input_limit_exceeded",
            "%s exceeds the maximum encoded text size." % field_name,
        )
    return value, size


def _validate_mime(value: Any, allowed: Iterable[str]) -> str:
    if not isinstance(value, str):
        raise _RequestError("unsupported_mime", "A supported MIME type is required.")
    mime_type = value.strip().casefold()
    if not _MIME_RE.fullmatch(mime_type) or mime_type not in set(allowed):
        raise _RequestError(
            "unsupported_mime", "The requested media MIME type is not supported."
        )
    return mime_type


def _validate_uri(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _RequestError("invalid_media", "uri must be a non-empty string.")
    uri = value.strip()
    if len(uri.encode("utf-8")) > MAX_URI_BYTES:
        raise _RequestError("input_limit_exceeded", "uri exceeds the size limit.")
    try:
        parsed = urlsplit(uri)
    except ValueError as exc:
        raise _RequestError("unsafe_media_uri", "The media URI is not allowed.") from exc
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise _RequestError("unsafe_media_uri", "The media URI is not allowed.")
    host = parsed.hostname.rstrip(".").casefold()
    if (
        host == "localhost"
        or host.endswith(".localhost")
        or host.endswith(".local")
        or host.endswith(".internal")
    ):
        raise _RequestError("unsafe_media_uri", "The media URI is not allowed.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise _RequestError("unsafe_media_uri", "The media URI is not allowed.")
    return uri


def _validate_metadata(
    descriptor: Mapping[str, Any],
    *,
    media_kind: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if media_kind == "image":
        width = _bounded_positive_int(
            descriptor.get("width"), "width", MAX_IMAGE_WIDTH
        )
        height = _bounded_positive_int(
            descriptor.get("height"), "height", MAX_IMAGE_HEIGHT
        )
        if width * height > MAX_IMAGE_PIXELS:
            raise _RequestError(
                "dimension_limit_exceeded",
                "Image dimensions exceed the maximum pixel count.",
            )
        result.update(width=width, height=height)
    else:
        result["duration_seconds"] = _bounded_positive_number(
            descriptor.get("duration_seconds"),
            "duration_seconds",
            MAX_MEDIA_DURATION_SECONDS,
        )
        sample_rate = _bounded_positive_int(
            descriptor.get("sample_rate_hz"),
            "sample_rate_hz",
            MAX_SAMPLE_RATE_HZ,
        )
        if sample_rate < MIN_SAMPLE_RATE_HZ:
            raise _RequestError(
                "invalid_request", "sample_rate_hz is below the supported range."
            )
        result["sample_rate_hz"] = sample_rate
    return result


def _validate_media_descriptor(
    value: Any,
    *,
    media_kind: str,
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _RequestError("invalid_media", "Media input must be an object.")
    source = value.get("source")
    if source not in {"inline", "uri", "artifact"}:
        raise _RequestError(
            "invalid_media",
            "Media source must be inline, uri, or artifact.",
        )
    allowed_mimes = _IMAGE_MIME_TYPES if media_kind == "image" else _AUDIO_MIME_TYPES
    descriptor: Dict[str, Any] = {
        "source": source,
        "mime_type": _validate_mime(value.get("mime_type"), allowed_mimes),
    }
    descriptor.update(_validate_metadata(value, media_kind=media_kind))
    declared_bytes = value.get("byte_length")
    if source != "inline" or declared_bytes is not None:
        descriptor["byte_length"] = _bounded_positive_int(
            declared_bytes, "byte_length", MAX_MEDIA_BYTES
        )
    if source == "inline":
        data = value.get("data_base64")
        if not isinstance(data, str) or not data:
            raise _RequestError(
                "invalid_media", "Inline media requires non-empty data_base64."
            )
        # Base64 expands by 4/3.  Reject obviously oversized strings before
        # allocating the decoded representation.
        if len(data) > ((MAX_INLINE_MEDIA_BYTES + 2) // 3) * 4:
            raise _RequestError(
                "input_limit_exceeded", "Inline media exceeds the byte limit."
            )
        descriptor["data_base64"] = data
    elif source == "uri":
        descriptor["uri"] = _validate_uri(value.get("uri"))
    else:
        artifact_ref = value.get("artifact_ref")
        if not isinstance(artifact_ref, str) or not artifact_ref.strip():
            raise _RequestError(
                "invalid_media",
                "Artifact media requires a non-empty artifact_ref.",
            )
        artifact_ref = artifact_ref.strip()
        if len(artifact_ref.encode("utf-8")) > MAX_ARTIFACT_REF_BYTES:
            raise _RequestError(
                "input_limit_exceeded", "artifact_ref exceeds the size limit."
            )
        descriptor["artifact_ref"] = artifact_ref
    return descriptor


def _inline_bytes(descriptor: Mapping[str, Any]) -> bytes:
    try:
        data = base64.b64decode(descriptor["data_base64"], validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise _RequestError(
            "invalid_media", "Inline media is not valid base64."
        ) from exc
    if not data:
        raise _RequestError("invalid_media", "Inline media must not be empty.")
    if len(data) > MAX_INLINE_MEDIA_BYTES:
        raise _RequestError(
            "input_limit_exceeded", "Inline media exceeds the byte limit."
        )
    declared = descriptor.get("byte_length")
    if declared is not None and declared != len(data):
        raise _RequestError(
            "media_metadata_mismatch",
            "Inline media byte_length does not match decoded content.",
        )
    return data


async def _call_media_loader(
    descriptor: Mapping[str, Any],
    *,
    timeout: float,
) -> Any:
    loader = _media_loader
    if loader is None:
        raise _RequestError(
            "media_loader_unavailable",
            "No allowlisted media loader is configured.",
        )
    callback = getattr(loader, "load", loader)

    def invoke() -> Any:
        return callback(
            dict(descriptor),
            max_bytes=min(int(descriptor["byte_length"]), MAX_MEDIA_BYTES),
            timeout=timeout,
        )

    with anyio.fail_after(timeout):
        run_sync_parameters = inspect.signature(anyio.to_thread.run_sync).parameters
        if "abandon_on_cancel" in run_sync_parameters:
            result = await anyio.to_thread.run_sync(
                invoke, abandon_on_cancel=True
            )
        else:  # pragma: no cover - compatibility with AnyIO 3.
            result = await anyio.to_thread.run_sync(invoke, cancellable=True)
        if inspect.isawaitable(result):
            result = await result
    return result


def _normalize_loaded_media(
    value: Any,
    descriptor: Mapping[str, Any],
) -> Tuple[bytes, Dict[str, Any]]:
    metadata: Dict[str, Any] = {}
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
    elif isinstance(value, Mapping):
        raw = value.get("data", value.get("content"))
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise _RequestError(
                "invalid_media_loader_output",
                "The media loader returned no byte payload.",
            )
        data = bytes(raw)
        loaded_mime = value.get("mime_type")
        if loaded_mime is not None and str(loaded_mime).casefold() != descriptor["mime_type"]:
            raise _RequestError(
                "media_metadata_mismatch",
                "The media loader returned a different MIME type.",
            )
        for key in (
            "duration_seconds",
            "sample_rate_hz",
            "width",
            "height",
        ):
            if key in value:
                metadata[key] = value[key]
    else:
        raise _RequestError(
            "invalid_media_loader_output",
            "The media loader returned an invalid result.",
        )
    if not data or len(data) > min(int(descriptor["byte_length"]), MAX_MEDIA_BYTES):
        raise _RequestError(
            "input_limit_exceeded",
            "Materialized media exceeds its declared byte limit.",
        )
    if len(data) != int(descriptor["byte_length"]):
        raise _RequestError(
            "media_metadata_mismatch",
            "Materialized media does not match byte_length.",
        )
    for key, actual in metadata.items():
        declared = descriptor.get(key)
        if declared is not None and actual != declared:
            raise _RequestError(
                "media_metadata_mismatch",
                "Materialized media metadata does not match the request.",
            )
    return data, metadata


async def _materialize_media(
    descriptor: Dict[str, Any],
    *,
    allow_remote_media: bool,
    timeout: float,
) -> Tuple[bytes, Dict[str, Any]]:
    if descriptor["source"] == "inline":
        data = _inline_bytes(descriptor)
        return data, {}
    if descriptor["source"] == "uri":
        if not isinstance(allow_remote_media, bool):
            raise _RequestError(
                "invalid_request", "allow_remote_media must be a boolean."
            )
        if not allow_remote_media:
            raise _RequestError(
                "remote_media_disabled",
                "Remote media loading is disabled for this request.",
            )
    loaded = await _call_media_loader(descriptor, timeout=timeout)
    return _normalize_loaded_media(loaded, descriptor)


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
    known = [
        getattr(capability, field_name, None)
        for capability in _candidate_capabilities(candidate, operation)
    ]
    values = [item for item in known if isinstance(item, int) and item > 0]
    return min(values) if values else None


def _candidate_supports_mime(candidate: Any, operation: str, mime_type: str) -> bool:
    declared = {
        str(item).casefold()
        for capability in _candidate_capabilities(candidate, operation)
        for item in tuple(getattr(capability, "media_types", ()) or ())
    }
    if not declared:
        return True
    major = mime_type.split("/", 1)[0] + "/*"
    return mime_type in declared or major in declared


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
    modality: str,
    service: Optional[str],
    model: Optional[str],
    provider: Optional[str],
    policy: Optional[Dict[str, Any]],
    device: Optional[str],
    mime_type: Optional[str] = None,
) -> Tuple[str, str, List[Any], int]:
    """Resolve every constraint against one immutable catalog snapshot."""

    from ipfs_accelerate_py.model_manager import get_default_model_manager

    manager = get_default_model_manager()
    snapshot = manager.snapshot()
    schema_version, revision = _snapshot_versions(snapshot)
    provider_selector = service if service is not None else provider
    try:
        resolution = manager.resolve(
            operation=operation,
            modality=modality,
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
    router_value = router_name.casefold()
    candidates = [
        candidate
        for candidate in tuple(getattr(resolution, "candidates", ()) or ())
        if str(getattr(candidate.binding, "router", "")).casefold() == router_value
        and _selector_matches(candidate.provider, service, "provider_id")
        and _selector_matches(candidate.provider, provider, "provider_id")
        and (
            mime_type is None
            or _candidate_supports_mime(candidate, operation, mime_type)
        )
    ]
    if not candidates:
        constrained = any(
            item is not None
            for item in (policy, service, provider, model, device, mime_type)
        )
        code = "selection_denied" if constrained else "no_match"
        raise _RequestError(
            code,
            (
                "No authorized router binding satisfies all requested constraints."
                if constrained
                else "No canonical router binding is available for this operation and modality."
            ),
            schema_version=schema_version,
            catalog_revision=revision,
        )
    return (
        schema_version,
        revision,
        candidates,
        int(getattr(resolution, "total_candidates", len(candidates))),
    )


async def _invoke_with_timeout(callback: Any, timeout: float) -> Any:
    def invoke() -> Any:
        return callback()

    with anyio.fail_after(timeout):
        run_sync_parameters = inspect.signature(anyio.to_thread.run_sync).parameters
        if "abandon_on_cancel" in run_sync_parameters:
            result = await anyio.to_thread.run_sync(
                invoke, abandon_on_cancel=True
            )
        else:  # pragma: no cover - compatibility with AnyIO 3.
            result = await anyio.to_thread.run_sync(invoke, cancellable=True)
        if inspect.isawaitable(result):
            result = await result
        return result


def _remaining_timeout(deadline: float) -> float:
    remaining = deadline - anyio.current_time()
    if remaining <= 0:
        raise TimeoutError
    return remaining


async def _invoke_candidates(
    candidates: Sequence[Any],
    callback: Any,
    *,
    deadline: float,
    allow_fallback: bool,
) -> Tuple[Any, Any, bool]:
    attempted = candidates if allow_fallback else candidates[:1]
    last_error: Optional[Exception] = None
    for index, candidate in enumerate(attempted):
        try:
            return (
                await _invoke_with_timeout(
                    lambda: callback(candidate),
                    _remaining_timeout(deadline),
                ),
                candidate,
                index > 0,
            )
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, TimeoutError):
                raise
            last_error = exc
    assert last_error is not None
    raise last_error


def _receipt(
    *,
    revision: str,
    operation: str,
    candidates: Sequence[Any],
    total_candidates: int,
    selected: Any,
    allow_fallback: bool,
    fallback_used: bool,
    input_summary: Dict[str, Any],
    output_summary: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_ids = [
        str(getattr(item.binding, "binding_id", ""))
        for item in candidates[:MAX_RECEIPT_CANDIDATES]
    ]
    return {
        "schema_version": AI_ROUTER_RECEIPT_SCHEMA_VERSION,
        "catalog_revision": revision,
        "operation": operation,
        "selected_binding_id": str(getattr(selected.binding, "binding_id", "")),
        "candidate_binding_ids": candidate_ids,
        "candidate_count": min(max(total_candidates, len(candidates)), 1_000),
        "candidate_count_truncated": total_candidates > len(candidate_ids),
        "fallback": {
            "allowed": bool(allow_fallback),
            "used": bool(fallback_used),
            "boundary_binding_ids": candidate_ids,
        },
        "input": dict(input_summary),
        "output": dict(output_summary),
    }


def _media_summary(descriptor: Mapping[str, Any], size: int) -> Dict[str, Any]:
    # Deliberately omit URI, artifact reference, and inline payload.
    result = {
        "source": descriptor["source"],
        "mime_type": descriptor["mime_type"],
        "bytes": size,
    }
    for key in ("duration_seconds", "sample_rate_hz", "width", "height"):
        if key in descriptor:
            result[key] = descriptor[key]
    return result


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
            "The bounded media or router operation timed out.",
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


async def multimodal_generate(
    prompt: str,
    media: List[Dict[str, Any]],
    service: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    policy: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    timeout: float = 30.0,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
    allow_remote_media: bool = False,
    allow_fallback: bool = False,
    stream: bool = False,
    max_stream_chunks: int = MAX_STREAM_CHUNKS,
) -> Dict[str, Any]:
    """Resolve vision capability and invoke ``multimodal_router``."""

    schema_version: Optional[str] = None
    revision: Optional[str] = None
    selected: Optional[Any] = None
    try:
        prompt_value, prompt_bytes = _validate_text(prompt, "prompt")
        if (
            isinstance(media, (str, bytes, Mapping))
            or not isinstance(media, Sequence)
            or not 1 <= len(media) <= MAX_MEDIA_ITEMS
        ):
            raise _RequestError(
                "item_count_exceeded",
                "media must contain exactly one bounded image item.",
            )
        descriptor = _validate_media_descriptor(media[0], media_kind="image")
        service_value = _bounded_selector(service, "service")
        model_value = _bounded_selector(model, "model")
        provider_value = _bounded_selector(provider, "provider")
        device_value = _bounded_selector(device, "device")
        policy_value = _bounded_policy(policy)
        timeout_value = _bounded_timeout(timeout)
        deadline = anyio.current_time() + timeout_value
        output_limit = _bounded_positive_int(
            max_output_bytes, "max_output_bytes", MAX_OUTPUT_BYTES
        )
        _validate_streaming(stream, max_stream_chunks)
        if not isinstance(allow_remote_media, bool):
            raise _RequestError(
                "invalid_request", "allow_remote_media must be a boolean."
            )
        if not isinstance(allow_fallback, bool):
            raise _RequestError(
                "invalid_request", "allow_fallback must be a boolean."
            )
        _bounded_positive_int(max_tokens, "max_tokens", 1_000_000)
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or not 0 <= float(temperature) <= 2
        ):
            raise _RequestError(
                "invalid_request", "temperature must be between 0 and 2."
            )
        schema_version, revision, candidates, total = _resolve_candidates(
            router_name=_MULTIMODAL_ROUTER,
            operation=_VISION_OPERATION,
            modality="image",
            service=service_value,
            model=model_value,
            provider=provider_value,
            policy=policy_value,
            device=device_value,
            mime_type=descriptor["mime_type"],
        )
        selected = candidates[0]
        data, _ = await _materialize_media(
            descriptor,
            allow_remote_media=allow_remote_media,
            timeout=_remaining_timeout(deadline),
        )
        input_bytes = prompt_bytes + len(data)
        catalog_limit = _candidate_limit(
            selected, _VISION_OPERATION, "max_input_bytes"
        )
        if catalog_limit is not None and input_bytes > catalog_limit:
            raise _RequestError(
                "input_limit_exceeded",
                "The request exceeds the selected service input limit.",
            )
        catalog_output = _candidate_limit(
            selected, _VISION_OPERATION, "max_output_bytes"
        )
        if catalog_output is not None:
            output_limit = min(output_limit, catalog_output)

        def call(candidate: Any) -> Any:
            return multimodal_router.generate_multimodal(
                prompt_value,
                image=data,
                model_name=_invocation_model(candidate),
                device=device_value,
                provider=_invocation_provider(candidate),
                max_tokens=max_tokens,
                temperature=float(temperature),
            )

        generated, selected, fallback_used = await _invoke_candidates(
            candidates,
            call,
            deadline=deadline,
            allow_fallback=allow_fallback,
        )
        if not isinstance(generated, str):
            raise _RequestError(
                "invalid_router_output",
                "The multimodal router returned a non-text result.",
            )
        output_bytes = len(generated.encode("utf-8"))
        if output_bytes > output_limit:
            raise _RequestError(
                "output_limit_exceeded",
                "Generated text exceeds the bounded output size.",
            )
        receipt = _receipt(
            revision=revision,
            operation=_VISION_OPERATION,
            candidates=candidates,
            total_candidates=total,
            selected=selected,
            allow_fallback=allow_fallback,
            fallback_used=fallback_used,
            input_summary={
                "count": 1,
                "text_bytes": prompt_bytes,
                "media": [_media_summary(descriptor, len(data))],
                "total_bytes": input_bytes,
            },
            output_summary={"mime_type": "text/plain", "bytes": output_bytes},
        )
        return {
            "status": "success",
            "success": True,
            "tool_schema_version": AI_ROUTER_TOOL_SCHEMA_VERSION,
            "schema_version": schema_version,
            "catalog_revision": revision,
            "text": generated,
            "selected_binding": _safe_binding(selected.binding),
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


async def voice_transcribe(
    audio: Dict[str, Any],
    service: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    policy: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    language: Optional[str] = None,
    timeout: float = 30.0,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
    allow_remote_media: bool = False,
    allow_fallback: bool = False,
    stream: bool = False,
    max_stream_chunks: int = MAX_STREAM_CHUNKS,
) -> Dict[str, Any]:
    """Resolve transcription capability and invoke ``voice_router``."""

    schema_version: Optional[str] = None
    revision: Optional[str] = None
    selected: Optional[Any] = None
    try:
        descriptor = _validate_media_descriptor(audio, media_kind="audio")
        service_value = _bounded_selector(service, "service")
        model_value = _bounded_selector(model, "model")
        provider_value = _bounded_selector(provider, "provider")
        device_value = _bounded_selector(device, "device")
        language_value = _bounded_selector(language, "language")
        policy_value = _bounded_policy(policy)
        timeout_value = _bounded_timeout(timeout)
        deadline = anyio.current_time() + timeout_value
        output_limit = _bounded_positive_int(
            max_output_bytes, "max_output_bytes", MAX_OUTPUT_BYTES
        )
        _validate_streaming(stream, max_stream_chunks)
        if not isinstance(allow_remote_media, bool):
            raise _RequestError(
                "invalid_request", "allow_remote_media must be a boolean."
            )
        if not isinstance(allow_fallback, bool):
            raise _RequestError(
                "invalid_request", "allow_fallback must be a boolean."
            )
        schema_version, revision, candidates, total = _resolve_candidates(
            router_name=_VOICE_ROUTER,
            operation=_TRANSCRIBE_OPERATION,
            modality="audio",
            service=service_value,
            model=model_value,
            provider=provider_value,
            policy=policy_value,
            device=device_value,
            mime_type=descriptor["mime_type"],
        )
        selected = candidates[0]
        data, _ = await _materialize_media(
            descriptor,
            allow_remote_media=allow_remote_media,
            timeout=_remaining_timeout(deadline),
        )
        catalog_limit = _candidate_limit(
            selected, _TRANSCRIBE_OPERATION, "max_input_bytes"
        )
        if catalog_limit is not None and len(data) > catalog_limit:
            raise _RequestError(
                "input_limit_exceeded",
                "Audio exceeds the selected service input limit.",
            )

        def call(candidate: Any) -> Any:
            return voice_router.speech_to_text(
                data,
                model_name=_invocation_model(candidate),
                language=language_value,
                device=device_value,
                provider=_invocation_provider(candidate),
            )

        transcript, selected, fallback_used = await _invoke_candidates(
            candidates,
            call,
            deadline=deadline,
            allow_fallback=allow_fallback,
        )
        if not isinstance(transcript, str):
            raise _RequestError(
                "invalid_router_output",
                "The voice router returned a non-text transcription.",
            )
        output_bytes = len(transcript.encode("utf-8"))
        catalog_output = _candidate_limit(
            selected, _TRANSCRIBE_OPERATION, "max_output_bytes"
        )
        if catalog_output is not None:
            output_limit = min(output_limit, catalog_output)
        if output_bytes > output_limit:
            raise _RequestError(
                "output_limit_exceeded",
                "The transcription exceeds the bounded output size.",
            )
        receipt = _receipt(
            revision=revision,
            operation=_TRANSCRIBE_OPERATION,
            candidates=candidates,
            total_candidates=total,
            selected=selected,
            allow_fallback=allow_fallback,
            fallback_used=fallback_used,
            input_summary={
                "count": 1,
                "media": [_media_summary(descriptor, len(data))],
                "total_bytes": len(data),
            },
            output_summary={"mime_type": "text/plain", "bytes": output_bytes},
        )
        return {
            "status": "success",
            "success": True,
            "tool_schema_version": AI_ROUTER_TOOL_SCHEMA_VERSION,
            "schema_version": schema_version,
            "catalog_revision": revision,
            "text": transcript,
            "selected_binding": _safe_binding(selected.binding),
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


def _wav_metadata(data: bytes) -> Optional[Tuple[float, int]]:
    """Return bounded WAV duration/sample-rate metadata without decoding audio."""

    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    position = 12
    byte_rate: Optional[int] = None
    sample_rate: Optional[int] = None
    data_size: Optional[int] = None
    while position + 8 <= len(data) and position <= 1_048_576:
        chunk_id = data[position : position + 4]
        size = struct.unpack_from("<I", data, position + 4)[0]
        start = position + 8
        end = start + size
        if end > len(data):
            return None
        if chunk_id == b"fmt " and size >= 16:
            sample_rate = struct.unpack_from("<I", data, start + 4)[0]
            byte_rate = struct.unpack_from("<I", data, start + 8)[0]
        elif chunk_id == b"data":
            data_size = size
        position = end + (size & 1)
    if not byte_rate or not sample_rate or data_size is None:
        return None
    return data_size / byte_rate, sample_rate


async def voice_synthesize(
    text: str,
    output_mime_type: str = "audio/wav",
    service: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    policy: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    voice: Optional[str] = None,
    sample_rate_hz: int = 24_000,
    max_duration_seconds: float = 120.0,
    timeout: float = 30.0,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
    allow_fallback: bool = False,
    stream: bool = False,
    max_stream_chunks: int = MAX_STREAM_CHUNKS,
) -> Dict[str, Any]:
    """Resolve synthesis capability and invoke ``voice_router``."""

    schema_version: Optional[str] = None
    revision: Optional[str] = None
    selected: Optional[Any] = None
    try:
        text_value, text_bytes = _validate_text(text, "text")
        mime_type = _validate_mime(output_mime_type, _AUDIO_MIME_TYPES)
        service_value = _bounded_selector(service, "service")
        model_value = _bounded_selector(model, "model")
        provider_value = _bounded_selector(provider, "provider")
        device_value = _bounded_selector(device, "device")
        voice_value = _bounded_selector(voice, "voice")
        policy_value = _bounded_policy(policy)
        sample_rate = _bounded_positive_int(
            sample_rate_hz, "sample_rate_hz", MAX_SAMPLE_RATE_HZ
        )
        if sample_rate < MIN_SAMPLE_RATE_HZ:
            raise _RequestError(
                "invalid_request", "sample_rate_hz is below the supported range."
            )
        duration_limit = _bounded_positive_number(
            max_duration_seconds,
            "max_duration_seconds",
            MAX_MEDIA_DURATION_SECONDS,
        )
        timeout_value = _bounded_timeout(timeout)
        deadline = anyio.current_time() + timeout_value
        output_limit = _bounded_positive_int(
            max_output_bytes, "max_output_bytes", MAX_OUTPUT_BYTES
        )
        _validate_streaming(stream, max_stream_chunks)
        if not isinstance(allow_fallback, bool):
            raise _RequestError(
                "invalid_request", "allow_fallback must be a boolean."
            )
        schema_version, revision, candidates, total = _resolve_candidates(
            router_name=_VOICE_ROUTER,
            operation=_SYNTHESIZE_OPERATION,
            modality="audio",
            service=service_value,
            model=model_value,
            provider=provider_value,
            policy=policy_value,
            device=device_value,
            mime_type=mime_type,
        )
        selected = candidates[0]
        catalog_input = _candidate_limit(
            selected, _SYNTHESIZE_OPERATION, "max_input_bytes"
        )
        if catalog_input is not None and text_bytes > catalog_input:
            raise _RequestError(
                "input_limit_exceeded",
                "Text exceeds the selected service input limit.",
            )
        catalog_output = _candidate_limit(
            selected, _SYNTHESIZE_OPERATION, "max_output_bytes"
        )
        if catalog_output is not None:
            output_limit = min(output_limit, catalog_output)

        def call(candidate: Any) -> Any:
            return voice_router.text_to_speech(
                text_value,
                voice=voice_value,
                model_name=_invocation_model(candidate),
                device=device_value,
                output_format=_AUDIO_FORMATS[mime_type],
                provider=_invocation_provider(candidate),
                sample_rate=sample_rate,
                max_duration_seconds=duration_limit,
            )

        raw_audio, selected, fallback_used = await _invoke_candidates(
            candidates,
            call,
            deadline=deadline,
            allow_fallback=allow_fallback,
        )
        if not isinstance(raw_audio, (bytes, bytearray, memoryview)):
            raise _RequestError(
                "invalid_router_output",
                "The voice router returned a non-byte audio result.",
            )
        audio = bytes(raw_audio)
        if not audio:
            raise _RequestError(
                "invalid_router_output", "The voice router returned empty audio."
            )
        if len(audio) > output_limit:
            raise _RequestError(
                "output_limit_exceeded",
                "Synthesized audio exceeds the bounded output size.",
            )
        duration: Optional[float] = None
        actual_sample_rate: Optional[int] = None
        if mime_type in {"audio/wav", "audio/x-wav"}:
            wav = _wav_metadata(audio)
            if wav is None:
                raise _RequestError(
                    "invalid_router_output",
                    "Synthesized WAV audio has invalid bounded metadata.",
                )
            duration, actual_sample_rate = wav
            if duration > duration_limit:
                raise _RequestError(
                    "duration_limit_exceeded",
                    "Synthesized audio exceeds the requested duration limit.",
                )
            if actual_sample_rate != sample_rate:
                raise _RequestError(
                    "sample_rate_mismatch",
                    "Synthesized audio has an unexpected sample rate.",
                )
        output_summary: Dict[str, Any] = {
            "mime_type": mime_type,
            "bytes": len(audio),
            "sample_rate_hz": actual_sample_rate or sample_rate,
            "max_duration_seconds": duration_limit,
        }
        if duration is not None:
            output_summary["duration_seconds"] = round(duration, 6)
        receipt = _receipt(
            revision=revision,
            operation=_SYNTHESIZE_OPERATION,
            candidates=candidates,
            total_candidates=total,
            selected=selected,
            allow_fallback=allow_fallback,
            fallback_used=fallback_used,
            input_summary={"count": 1, "text_bytes": text_bytes},
            output_summary=output_summary,
        )
        # Media is bounded and encoded only in the operation result.  The
        # selected binding and receipt contain metadata alone.
        return {
            "status": "success",
            "success": True,
            "tool_schema_version": AI_ROUTER_TOOL_SCHEMA_VERSION,
            "schema_version": schema_version,
            "catalog_revision": revision,
            "audio": {
                "mime_type": mime_type,
                "byte_length": len(audio),
                "data_base64": base64.b64encode(audio).decode("ascii"),
                "sha256": hashlib.sha256(audio).hexdigest(),
            },
            "selected_binding": _safe_binding(selected.binding),
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


def _constraint_schema() -> Dict[str, Any]:
    return {
        "service": {"type": "string", "minLength": 1, "maxLength": 256},
        "model": {"type": "string", "minLength": 1, "maxLength": 256},
        "provider": {"type": "string", "minLength": 1, "maxLength": 256},
        "policy": {
            "type": "object",
            "maxProperties": MAX_POLICY_ENTRIES,
            "additionalProperties": {"type": ["string", "number", "boolean"]},
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


def _media_schema(media_kind: str) -> Dict[str, Any]:
    mime_types = sorted(_IMAGE_MIME_TYPES if media_kind == "image" else _AUDIO_MIME_TYPES)
    metadata = (
        {
            "width": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_IMAGE_WIDTH,
            },
            "height": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_IMAGE_HEIGHT,
            },
        }
        if media_kind == "image"
        else {
            "duration_seconds": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": MAX_MEDIA_DURATION_SECONDS,
            },
            "sample_rate_hz": {
                "type": "integer",
                "minimum": MIN_SAMPLE_RATE_HZ,
                "maximum": MAX_SAMPLE_RATE_HZ,
            },
        }
    )
    required_metadata = list(metadata)

    def variant(source: str, specific: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
        properties: Dict[str, Any] = {
            "source": {"const": source},
            "mime_type": {"type": "string", "enum": mime_types},
            "byte_length": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_MEDIA_BYTES,
            },
            **metadata,
            **specific,
        }
        return {
            "type": "object",
            "properties": properties,
            "required": ["source", "mime_type", *required_metadata, *required],
            "additionalProperties": False,
        }

    return {
        "oneOf": [
            variant(
                "inline",
                {
                    "data_base64": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": ((MAX_INLINE_MEDIA_BYTES + 2) // 3) * 4,
                    }
                },
                ["data_base64"],
            ),
            variant(
                "uri",
                {
                    "uri": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_URI_BYTES,
                        "pattern": "^https://",
                    }
                },
                ["uri", "byte_length"],
            ),
            variant(
                "artifact",
                {
                    "artifact_ref": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_ARTIFACT_REF_BYTES,
                    }
                },
                ["artifact_ref", "byte_length"],
            ),
        ]
    }


def register_native_ai_router_tools(manager: Any) -> None:
    """Register the three canonical vision and voice MCP tools."""

    vision = _constraint_schema()
    vision.update(
        {
            "prompt": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_TEXT_BYTES,
            },
            "media": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_MEDIA_ITEMS,
                "items": _media_schema("image"),
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
            "allow_remote_media": {"type": "boolean", "default": False},
        }
    )
    transcribe = _constraint_schema()
    transcribe.update(
        {
            "audio": _media_schema("audio"),
            "language": {"type": "string", "minLength": 1, "maxLength": 256},
            "allow_remote_media": {"type": "boolean", "default": False},
        }
    )
    synthesize = _constraint_schema()
    synthesize.update(
        {
            "text": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_TEXT_BYTES,
            },
            "output_mime_type": {
                "type": "string",
                "enum": sorted(_AUDIO_MIME_TYPES),
                "default": "audio/wav",
            },
            "voice": {"type": "string", "minLength": 1, "maxLength": 256},
            "sample_rate_hz": {
                "type": "integer",
                "minimum": MIN_SAMPLE_RATE_HZ,
                "maximum": MAX_SAMPLE_RATE_HZ,
                "default": 24_000,
            },
            "max_duration_seconds": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": MAX_MEDIA_DURATION_SECONDS,
                "default": 120.0,
            },
        }
    )
    registrations = (
        (
            "multimodal_generate",
            multimodal_generate,
            "Resolve one catalog revision and generate text through multimodal_router.",
            vision,
            ["prompt", "media"],
        ),
        (
            "voice_transcribe",
            voice_transcribe,
            "Resolve one catalog revision and transcribe audio through voice_router.",
            transcribe,
            ["audio"],
        ),
        (
            "voice_synthesize",
            voice_synthesize,
            "Resolve one catalog revision and synthesize audio through voice_router.",
            synthesize,
            ["text"],
        ),
    )
    for name, func, description, properties, required in registrations:
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
            tags=["native", "mcpp", "ai-router", "catalog", "media", "bounded"],
        )


register_native_ai_router_tool = register_native_ai_router_tools
register_native_vision_voice_tools = register_native_ai_router_tools
register_ai_router_tools = register_native_ai_router_tools


__all__ = [
    "AI_ROUTER_RECEIPT_SCHEMA_VERSION",
    "AI_ROUTER_TOOL_SCHEMA_VERSION",
    "MAX_IMAGE_HEIGHT",
    "MAX_IMAGE_WIDTH",
    "MAX_INLINE_MEDIA_BYTES",
    "MAX_MEDIA_BYTES",
    "MAX_MEDIA_DURATION_SECONDS",
    "MAX_MEDIA_ITEMS",
    "MAX_OUTPUT_BYTES",
    "MAX_SAMPLE_RATE_HZ",
    "MAX_STREAM_CHUNKS",
    "MAX_TEXT_BYTES",
    "MAX_TIMEOUT_SECONDS",
    "MediaLoader",
    "configure_media_loader",
    "multimodal_generate",
    "register_ai_router_tools",
    "register_native_ai_router_tool",
    "register_native_ai_router_tools",
    "register_native_vision_voice_tools",
    "voice_synthesize",
    "voice_transcribe",
]
