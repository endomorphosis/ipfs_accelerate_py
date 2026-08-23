"""MCP tools for the closed External Agent Handoff API (EAAEF-112).

Registration is cold and side-effect free.  Tools accept canonical request
dicts plus connector/file references; client host paths are not authorization.
Server-configured allowlists remain fail-closed at invocation time.

Tool names are distinct from prompt-lifecycle tools.  Preview never admits a
mutating handoff.  Approve/reject require an independent reviewer distinct
from the worker.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.api.external_handoff import (
    HANDOFF_API_OPERATIONS,
    MAX_INSTRUCTION_BYTES,
    MAX_REASON_BYTES,
    ExternalHandoffAPI,
    ExternalHandoffAPIError,
    ExternalHandoffAuthorityError,
    WorkerSelfApprovalError,
    coerce_request,
    get_default_api,
)

HANDOFF_MCP_CATEGORY: Final = "agent_supervisor"
HANDOFF_MCP_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.external-handoff-mcp@1"
)
REPOSITORY_ALLOWLIST_ENV: Final = "IPFS_ACCELERATE_AGENT_REPOSITORY_ALLOWLIST"
STATE_ALLOWLIST_ENV: Final = "IPFS_ACCELERATE_AGENT_STATE_ALLOWLIST"
CONNECTOR_ALLOWLIST_ENV: Final = "IPFS_ACCELERATE_AGENT_CONNECTOR_ALLOWLIST"
MAX_REQUEST_BYTES: Final[int] = 65_536

HANDOFF_MCP_OPERATIONS: Final[tuple[str, ...]] = (
    "handoff",
    "preview",
    "attach",
    "status",
    "follow",
    "steer",
    "pause",
    "resume",
    "approve",
    "reject",
    "cancel",
    "explain",
    "doctor",
    "report",
)

# Canonical MCP names stay distinct from prompt-lifecycle tools.
HANDOFF_TOOL_NAMES: Final[tuple[str, ...]] = (
    "agent_supervisor_handoff",
    "agent_supervisor_preview_handoff",
    "agent_supervisor_attach",
    "agent_supervisor_handoff_status",
    "agent_supervisor_handoff_follow",
    "agent_supervisor_handoff_steer",
    "agent_supervisor_handoff_pause",
    "agent_supervisor_handoff_resume",
    "agent_supervisor_handoff_approve",
    "agent_supervisor_handoff_reject",
    "agent_supervisor_handoff_cancel",
    "agent_supervisor_handoff_explain",
    "agent_supervisor_handoff_doctor",
    "agent_supervisor_handoff_report",
)

_CANONICAL_TOOL_TO_OPERATION: Final[dict[str, str]] = {
    "agent_supervisor_handoff": "handoff",
    "agent_supervisor_preview_handoff": "preview",
    "agent_supervisor_attach": "attach",
    "agent_supervisor_handoff_status": "status",
    "agent_supervisor_handoff_follow": "follow",
    "agent_supervisor_handoff_steer": "steer",
    "agent_supervisor_handoff_pause": "pause",
    "agent_supervisor_handoff_resume": "resume",
    "agent_supervisor_handoff_approve": "approve",
    "agent_supervisor_handoff_reject": "reject",
    "agent_supervisor_handoff_cancel": "cancel",
    "agent_supervisor_handoff_explain": "explain",
    "agent_supervisor_handoff_doctor": "doctor",
    "agent_supervisor_handoff_report": "report",
}

# Objective-list aliases that do not collide with native catalog or
# prompt-lifecycle tool names (agent_supervisor_status/follow/steer/explain/doctor).
HANDOFF_TOOL_ALIASES: Final[dict[str, str]] = {
    "preview_handoff": "preview",
    "attach": "attach",
    "follow": "follow",
    "steer": "steer",
    "approve": "approve",
    "reject": "reject",
    "explain": "explain",
    "doctor": "doctor",
    "report": "report",
}

TOOL_TO_OPERATION: Final[dict[str, str]] = {
    **_CANONICAL_TOOL_TO_OPERATION,
    **HANDOFF_TOOL_ALIASES,
}

_PROMPT_LIFECYCLE_COLLISIONS: Final[frozenset[str]] = frozenset(
    {
        "agent_supervisor_run",
        "agent_supervisor_preview",
        "agent_supervisor_steer",
        "agent_supervisor_status",
        "agent_supervisor_follow",
        "agent_supervisor_explain",
        "agent_supervisor_doctor",
    }
)
_NATIVE_CATALOG_COLLISIONS: Final[frozenset[str]] = frozenset(
    {
        "status",
        "pause",
        "resume",
        "cancel",
    }
)

_CONTROL_OPERATIONS: Final[frozenset[str]] = frozenset(
    {"steer", "pause", "resume", "cancel"}
)
_ORIGIN_OPERATIONS: Final[frozenset[str]] = frozenset({"handoff", "preview"})
_REVIEW_OPERATIONS: Final[frozenset[str]] = frozenset({"approve", "reject"})
_REQUEST_FIELDS: Final[tuple[str, ...]] = (
    "principal_id",
    "worker_principal_id",
    "reviewer_principal_id",
    "authority_id",
    "run_id",
    "session_id",
    "repository_id",
    "objective_id",
    "idempotency_key",
    "cursor",
    "instruction",
    "reason",
)
_PATH_PREFIXES: Final[tuple[str, ...]] = (
    "/",
    "./",
    "../",
    "~",
    "file:",
    "file://",
)

_lock = RLock()
_injected_api: ExternalHandoffAPI | None = None


class HandoffMCPError(RuntimeError):
    """Typed MCP handoff failure before API dispatch."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class PathInjectionDenied(HandoffMCPError):
    """Client-supplied path or connector is not on the server allowlist."""

    def __init__(self, message: str, *, reason_code: str = "path_denied") -> None:
        super().__init__(message, reason_code=reason_code)


def configure_external_handoff_api(api: ExternalHandoffAPI | None) -> None:
    """Inject an ExternalHandoffAPI instance for later tool invocations."""

    global _injected_api
    if api is not None and not isinstance(api, ExternalHandoffAPI):
        raise TypeError("api must be an ExternalHandoffAPI")
    with _lock:
        _injected_api = api


def get_external_handoff_api() -> ExternalHandoffAPI:
    """Return the injected API, or the process-local in-memory default."""

    with _lock:
        if _injected_api is not None:
            return _injected_api
    return get_default_api()


def external_handoff_discovery_manifest() -> dict[str, Any]:
    """Static tool vocabulary without constructing an API or reading paths."""

    return {
        "schema": HANDOFF_MCP_SCHEMA,
        "category": HANDOFF_MCP_CATEGORY,
        "tools": list(HANDOFF_TOOL_NAMES),
        "aliases": dict(HANDOFF_TOOL_ALIASES),
        "operations": list(HANDOFF_MCP_OPERATIONS),
        "api_operations": list(HANDOFF_API_OPERATIONS),
        "path_authority": "server_allowlist_only",
        "connector_or_file_references": True,
        "raw_host_path_authority": False,
        "cold_registration": True,
        "preview_is_handoff": False,
        "self_approval": False,
        "live_quack": False,
        "live_docker": False,
    }


def _allowlist(env_name: str) -> tuple[Path, ...]:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return ()
    paths: list[Path] = []
    for item in raw.split(os.pathsep):
        item = item.strip()
        if not item:
            continue
        paths.append(Path(item).resolve())
    return tuple(paths)


def _connector_allowlist() -> tuple[str, ...]:
    raw = os.environ.get(CONNECTOR_ALLOWLIST_ENV, "").strip()
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(os.pathsep) if item.strip())


def _looks_like_host_path(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    lowered = text.lower()
    if any(lowered.startswith(prefix) for prefix in _PATH_PREFIXES):
        return True
    if "\\" in text:
        return True
    if len(text) >= 2 and text[1] == ":":
        return True
    return False


def _authorize_file_reference(path: str, *, field: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise PathInjectionDenied(f"{field} must be a string path")
    candidate = Path(path).resolve()
    allowed = _allowlist(STATE_ALLOWLIST_ENV) or _allowlist(REPOSITORY_ALLOWLIST_ENV)
    if not allowed:
        raise PathInjectionDenied(
            f"client {field} paths require server allowlist "
            f"({STATE_ALLOWLIST_ENV} or {REPOSITORY_ALLOWLIST_ENV})"
        )
    for root in allowed:
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    raise PathInjectionDenied(f"{field} path not allowlisted: {candidate}")


def _authorize_connector(connector_id: str) -> str:
    if not isinstance(connector_id, str) or not connector_id.strip():
        raise PathInjectionDenied("connector_id must be a non-empty string")
    token = connector_id.strip()
    if _looks_like_host_path(token):
        raise PathInjectionDenied("connector_id must not be a host path")
    allowed = _connector_allowlist()
    if not allowed:
        raise PathInjectionDenied(
            f"client connector_id values require server allowlist "
            f"({CONNECTOR_ALLOWLIST_ENV})"
        )
    if token not in allowed:
        raise PathInjectionDenied(f"connector_id not allowlisted: {token}")
    return token


def _read_text_reference(path: Path, *, field: str, max_bytes: int) -> str:
    if not path.is_file():
        raise HandoffMCPError(f"{field} not found", reason_code="malformed")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise HandoffMCPError(
            f"{field} exceeds {max_bytes} UTF-8 bytes", reason_code="bounds"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HandoffMCPError(f"{field} is not valid UTF-8", reason_code="malformed") from exc
    if "\x00" in text:
        raise HandoffMCPError(f"{field} must not contain NUL", reason_code="malformed")
    return text


def _read_request_file(path: Path) -> dict[str, Any]:
    raw = _read_text_reference(path, field="request_file", max_bytes=MAX_REQUEST_BYTES)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HandoffMCPError(
            "request_file is not valid JSON", reason_code="malformed"
        ) from exc
    if not isinstance(payload, Mapping):
        raise HandoffMCPError(
            "request_file must be a JSON object", reason_code="malformed"
        )
    return dict(payload)


def _optional_string(value: Any, name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise HandoffMCPError(f"{name} must be a string", reason_code="malformed")
    return value.strip()


def _merge_request(operation: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"operation": operation}
    nested = arguments.get("request")
    if nested not in (None, ""):
        if not isinstance(nested, Mapping):
            raise HandoffMCPError("request must be an object", reason_code="malformed")
        supplied = nested.get("operation")
        if supplied not in (None, "") and str(supplied).strip().lower().replace(
            "-", "_"
        ) != operation:
            raise HandoffMCPError(
                "request operation does not match the invoked MCP tool",
                reason_code="operation_mismatch",
            )
        for field in _REQUEST_FIELDS:
            if field in nested and nested[field] not in (None, ""):
                payload[field] = nested[field]

    request_file = arguments.get("request_file")
    if request_file not in (None, ""):
        loaded = _read_request_file(
            _authorize_file_reference(str(request_file), field="request_file")
        )
        supplied = loaded.get("operation")
        if supplied not in (None, "") and str(supplied).strip().lower().replace(
            "-", "_"
        ) != operation:
            raise HandoffMCPError(
                "request_file operation does not match the invoked MCP tool",
                reason_code="operation_mismatch",
            )
        for field in _REQUEST_FIELDS:
            if field in loaded and loaded[field] not in (None, ""):
                payload[field] = loaded[field]

    connector_id = arguments.get("connector_id")
    if connector_id not in (None, ""):
        _authorize_connector(str(connector_id))

    for field in _REQUEST_FIELDS:
        if field in {"instruction", "reason"}:
            continue
        value = arguments.get(field)
        if value not in (None, ""):
            payload[field] = value

    instruction = _optional_string(arguments.get("instruction"), "instruction")
    instruction_file = arguments.get("instruction_file")
    if instruction and instruction_file not in (None, ""):
        raise HandoffMCPError(
            "supply at most one instruction source", reason_code="malformed"
        )
    if instruction_file not in (None, ""):
        instruction = _read_text_reference(
            _authorize_file_reference(str(instruction_file), field="instruction_file"),
            field="instruction_file",
            max_bytes=MAX_INSTRUCTION_BYTES,
        )
    if instruction:
        payload["instruction"] = instruction

    reason = _optional_string(arguments.get("reason"), "reason")
    reason_file = arguments.get("reason_file")
    if reason and reason_file not in (None, ""):
        raise HandoffMCPError(
            "supply at most one reason source", reason_code="malformed"
        )
    if reason_file not in (None, ""):
        reason = _read_text_reference(
            _authorize_file_reference(str(reason_file), field="reason_file"),
            field="reason_file",
            max_bytes=MAX_REASON_BYTES,
        )
    if reason:
        payload["reason"] = reason

    repository_id = str(payload.get("repository_id") or "")
    if _looks_like_host_path(repository_id):
        raise PathInjectionDenied(
            "repository_id is an identity, not host-path authority"
        )
    return payload


def _result_ok(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "receipt": dict(receipt),
        "result": dict(receipt),
        "preview_is_handoff": False,
        "self_approval": False,
    }


def _result_err(error: str, *, code: str) -> dict[str, Any]:
    return {"ok": False, "error": error, "error_code": code, "reason_code": code}


def execute_external_handoff_operation(
    operation: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    api: ExternalHandoffAPI | None = None,
) -> dict[str, Any]:
    """Dispatch one closed handoff operation onto ExternalHandoffAPI."""

    name = str(operation or "").strip().lower().replace("-", "_")
    if name in TOOL_TO_OPERATION:
        name = TOOL_TO_OPERATION[name]
    if name not in HANDOFF_MCP_OPERATIONS:
        return _result_err("unknown handoff MCP operation", code="unknown_operation")
    try:
        payload = _merge_request(name, arguments or {})
        if name not in _ORIGIN_OPERATIONS and not str(payload.get("run_id") or "").strip():
            raise HandoffMCPError("run_id is required", reason_code="malformed")
        if name in _CONTROL_OPERATIONS and not str(payload.get("authority_id") or "").strip():
            raise HandoffMCPError(
                "authority_id is required", reason_code="authority_mismatch"
            )
        if name == "steer" and not str(payload.get("instruction") or "").strip():
            raise HandoffMCPError("instruction is required", reason_code="malformed")
        if not str(payload.get("principal_id") or "").strip():
            raise HandoffMCPError("principal_id is required", reason_code="malformed")
        request = coerce_request(payload, operation=name)
        service = api if api is not None else get_external_handoff_api()
        receipt = getattr(service, name)(request)
        record = dict(receipt.to_dict())
        record["content_id"] = receipt.content_id
        record["receipt_id"] = receipt.receipt_id
        if name == "preview" and record.get("verdict") != "preview_only":
            return _result_err(
                "preview must not admit a mutating handoff",
                code="preview_mutation_denied",
            )
        if name in _REVIEW_OPERATIONS:
            worker = str(record.get("worker_principal_id") or "")
            reviewer = str(record.get("reviewer_principal_id") or "")
            if worker and reviewer and worker == reviewer:
                return _result_err(
                    "worker self-approval is forbidden",
                    code="worker_self_approval",
                )
        return _result_ok(record)
    except PathInjectionDenied as exc:
        return _result_err(str(exc), code=exc.reason_code)
    except WorkerSelfApprovalError as exc:
        return _result_err(str(exc), code=exc.reason_code)
    except ExternalHandoffAuthorityError as exc:
        return _result_err(str(exc), code=exc.reason_code)
    except HandoffMCPError as exc:
        return _result_err(str(exc), code=exc.reason_code)
    except ExternalHandoffAPIError as exc:
        return _result_err(str(exc), code=exc.reason_code)


def _tool_input_schema(operation: str) -> dict[str, Any]:
    props: dict[str, Any] = {
        "principal_id": {"type": "string", "minLength": 1},
        "worker_principal_id": {"type": "string"},
        "reviewer_principal_id": {"type": "string"},
        "authority_id": {"type": "string"},
        "run_id": {"type": "string"},
        "session_id": {"type": "string"},
        "repository_id": {
            "type": "string",
            "description": "Repository identity, never host-path authority.",
        },
        "objective_id": {"type": "string"},
        "idempotency_key": {"type": "string"},
        "cursor": {"type": "string"},
        "instruction": {"type": "string"},
        "instruction_file": {
            "type": "string",
            "description": "Server-allowlisted instruction file reference.",
        },
        "reason": {"type": "string"},
        "reason_file": {
            "type": "string",
            "description": "Server-allowlisted reason file reference.",
        },
        "request_file": {
            "type": "string",
            "description": "Server-allowlisted JSON request file reference.",
        },
        "connector_id": {
            "type": "string",
            "description": "Server-configured connector alias, not a host path.",
        },
        "request": {
            "type": "object",
            "description": "Canonical ExternalHandoffRequest fields.",
        },
    }
    required = ["principal_id"]
    if operation not in _ORIGIN_OPERATIONS:
        required.append("run_id")
    if operation in _CONTROL_OPERATIONS:
        required.append("authority_id")
    if operation == "steer":
        # Instruction may arrive via instruction_file; enforce at invoke time.
        pass
    if operation in _REVIEW_OPERATIONS:
        required.append("reviewer_principal_id")
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def _tool_description(operation: str) -> str:
    return (
        f"External agent handoff {operation} via ExternalHandoffAPI. "
        "Connector/file references only; client host paths are not authorization. "
        "Preview never admits mutation; workers cannot self-approve."
    )


def _make_tool(operation: str):
    async def tool(**arguments: Any) -> dict[str, Any]:
        return execute_external_handoff_operation(operation, arguments)

    tool.__name__ = next(
        name for name, bound in _CANONICAL_TOOL_TO_OPERATION.items() if bound == operation
    )
    tool.__qualname__ = tool.__name__
    tool.__doc__ = _tool_description(operation)
    tool.__handoff_operation__ = operation  # type: ignore[attr-defined]
    return tool


_TOOL_FUNCS: Final[dict[str, Any]] = {
    name: _make_tool(operation) for name, operation in _CANONICAL_TOOL_TO_OPERATION.items()
}

agent_supervisor_handoff = _TOOL_FUNCS["agent_supervisor_handoff"]
agent_supervisor_preview_handoff = _TOOL_FUNCS["agent_supervisor_preview_handoff"]
agent_supervisor_attach = _TOOL_FUNCS["agent_supervisor_attach"]
agent_supervisor_handoff_status = _TOOL_FUNCS["agent_supervisor_handoff_status"]
agent_supervisor_handoff_follow = _TOOL_FUNCS["agent_supervisor_handoff_follow"]
agent_supervisor_handoff_steer = _TOOL_FUNCS["agent_supervisor_handoff_steer"]
agent_supervisor_handoff_pause = _TOOL_FUNCS["agent_supervisor_handoff_pause"]
agent_supervisor_handoff_resume = _TOOL_FUNCS["agent_supervisor_handoff_resume"]
agent_supervisor_handoff_approve = _TOOL_FUNCS["agent_supervisor_handoff_approve"]
agent_supervisor_handoff_reject = _TOOL_FUNCS["agent_supervisor_handoff_reject"]
agent_supervisor_handoff_cancel = _TOOL_FUNCS["agent_supervisor_handoff_cancel"]
agent_supervisor_handoff_explain = _TOOL_FUNCS["agent_supervisor_handoff_explain"]
agent_supervisor_handoff_doctor = _TOOL_FUNCS["agent_supervisor_handoff_doctor"]
agent_supervisor_handoff_report = _TOOL_FUNCS["agent_supervisor_handoff_report"]


def register_external_handoff_tools(manager: Any) -> None:
    """Register handoff MCP tools without resolving an API or allowlist."""

    registered: set[str] = set()
    definitions: list[tuple[str, str]] = [
        (name, operation) for name, operation in _CANONICAL_TOOL_TO_OPERATION.items()
    ]
    for alias, operation in HANDOFF_TOOL_ALIASES.items():
        if alias in _PROMPT_LIFECYCLE_COLLISIONS or alias in _NATIVE_CATALOG_COLLISIONS:
            continue
        if alias in _CANONICAL_TOOL_TO_OPERATION:
            continue
        definitions.append((alias, operation))
    for name, operation in definitions:
        if name in registered:
            continue
        registered.add(name)
        manager.register_tool(
            category=HANDOFF_MCP_CATEGORY,
            name=name,
            func=_TOOL_FUNCS[
                next(
                    canonical
                    for canonical, bound in _CANONICAL_TOOL_TO_OPERATION.items()
                    if bound == operation
                )
            ],
            description=_tool_description(operation),
            input_schema=_tool_input_schema(operation),
            runtime="fastapi",
            tags=[
                "native",
                "agent-supervisor",
                "external-handoff",
                "policy-controlled",
                "connector-or-file-reference",
            ],
        )


__all__ = [
    "CONNECTOR_ALLOWLIST_ENV",
    "HANDOFF_MCP_CATEGORY",
    "HANDOFF_MCP_OPERATIONS",
    "HANDOFF_MCP_SCHEMA",
    "HANDOFF_TOOL_ALIASES",
    "HANDOFF_TOOL_NAMES",
    "HandoffMCPError",
    "PathInjectionDenied",
    "REPOSITORY_ALLOWLIST_ENV",
    "STATE_ALLOWLIST_ENV",
    "TOOL_TO_OPERATION",
    "agent_supervisor_attach",
    "agent_supervisor_handoff",
    "agent_supervisor_handoff_approve",
    "agent_supervisor_handoff_cancel",
    "agent_supervisor_handoff_doctor",
    "agent_supervisor_handoff_explain",
    "agent_supervisor_handoff_follow",
    "agent_supervisor_handoff_pause",
    "agent_supervisor_handoff_reject",
    "agent_supervisor_handoff_report",
    "agent_supervisor_handoff_resume",
    "agent_supervisor_handoff_status",
    "agent_supervisor_handoff_steer",
    "agent_supervisor_preview_handoff",
    "configure_external_handoff_api",
    "execute_external_handoff_operation",
    "external_handoff_discovery_manifest",
    "get_external_handoff_api",
    "register_external_handoff_tools",
]
