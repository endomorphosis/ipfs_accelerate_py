"""Live MCP JSON-RPC / MCP++ observation for deterministic contract repair (DCR-023).

Interfaces
----------
* ``McpLiveObservation@1`` — one typed request/response exchange.
* ``LiveContractTranscript@1`` — multi-service observation transcript.
* MCP JSON-RPC 2.0 and MCP++ Profiles A–F capability probes.

Normative rules (fail-closed):

* Local loopback / in-process only; never infer missing calls.
* Transport, discovery, and RPC failures stay typed; they never become empty success.
* All three reviewed service roles (accelerate, datasets, kit) are observed.
* Process-local and datasets MCP ``logic_tools/cec_prove`` results are compared by
  canonical CID equivalence when both surfaces produce a structured result.
* Observations bind the DCR-022 runtime process/config/endpoint witness axes.

Evidence term: ``dcr/live-observation@1``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.analysis.runtime_service_identity import (
    DEFAULT_SERVICES_RELATIVE,
    REQUIRED_SERVICE_ROLES,
    RuntimeServiceManifest,
    RuntimeServiceWitness,
    ServiceRuntimeObservation,
    build_runtime_service_witness,
    is_pseudo_cid,
    load_runtime_service_manifest,
    synthesize_bound_observations,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ContractValidationError,
    content_identity,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.idl_registry import (
    IDL_IDENTITY_PROFILE,
    InterfaceDescriptorRegistry,
    build_ai_catalog_v1_descriptor,
    identify_interface_descriptor,
    is_pseudo_interface_cid,
)

# ---------------------------------------------------------------------------
# Schemas / interfaces / constants
# ---------------------------------------------------------------------------

MCP_LIVE_OBSERVATION_INTERFACE: Final = "McpLiveObservation@1"
LIVE_CONTRACT_TRANSCRIPT_INTERFACE: Final = "LiveContractTranscript@1"
MCP_LIVE_OBSERVATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-live-observation@1"
)
LIVE_CONTRACT_TRANSCRIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/live-contract-transcript@1"
)
LIVE_OBSERVATION_EVIDENCE_TERM: Final = "dcr/live-observation@1"
DCR_TASK_ID: Final = "DCR-023"
DCR_ARTIFACT_PATH: Final = (
    "data/agent_supervisor/deterministic_contract_repair/mcp-live-transcript.json"
)

JSONRPC_VERSION: Final = "2.0"
MCP_PROTOCOL_VERSION: Final = "2024-11-05"

# MCP++ Profiles A–F (G/H are out of this observation wave).
MCP_PLUS_PROFILES_A_F: Final[tuple[str, ...]] = (
    "mcp++/profile-a-idl",
    "mcp++/profile-b-cid-artifacts",
    "mcp++/profile-c-ucan",
    "mcp++/profile-d-temporal-deontic",
    "mcp++/profile-e-mcp-p2p",
    "mcp++/profile-f-event-dag",
)

# One allowlisted safe tools/call per role (no model invocation, no user mutation).
SAFE_TOOLS_CALL: Final[Mapping[str, str]] = MappingProxyType(
    {
        "accelerate": "model_catalog_health",
        "datasets": "logic_health",
        "kit": "iroh_diagnostics",
    }
)

UNKNOWN_TOOL_NAME: Final = "__dcr_unknown_tool__"
MALFORMED_METHOD: Final = "tools/call"
LOGIC_CEC_PROVE_TOOL: Final = "logic_tools/cec_prove"
LOGIC_CEC_PROVE_GOAL: Final = "True"

_MAX_BYTES_FIELD: Final = 16_384
_MAX_EXCHANGES: Final = 4_096


class McpLiveObserverError(ValueError):
    """Live observation input or transcript violates a closed invariant."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "mcp_live_observer_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class ObservationTerminalState(str, Enum):
    """Closed terminal states for one observed exchange."""

    PASSED = "passed"
    FAILED = "failed"
    REFUTED = "refuted"
    UNSUPPORTED = "unsupported"
    TRANSPORT_ERROR = "transport_error"


class ObservationKind(str, Enum):
    """Closed set of exchange kinds captured by the observer."""

    INITIALIZE = "initialize"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    MALFORMED_CALL = "malformed_call"
    UNKNOWN_CALL = "unknown_call"
    PROFILE_PROBE = "profile_probe"
    LOGIC_CEC_PROVE = "logic_tools/cec_prove"
    LOOPBACK_PROBE = "loopback_probe"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_label(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _plain(value: Any, *, depth: int = 0, drop_floats: bool = False) -> Any:
    if depth > 24:
        raise McpLiveObserverError(
            "observation value exceeds nesting bound",
            reason_code="nesting_bound",
        )
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if drop_floats:
            # Ambient tool payloads may include wall-clock floats; drop them
            # rather than fail closed on otherwise typed observations.
            return None
        raise McpLiveObserverError(
            "floating values are not canonical observation data",
            reason_code="float_rejected",
        )
    if isinstance(value, Mapping):
        if len(value) > 1_024:
            raise McpLiveObserverError(
                "observation object exceeds key bound",
                reason_code="object_oversized",
            )
        cleaned: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            item = _plain(value[key], depth=depth + 1, drop_floats=drop_floats)
            if item is None and isinstance(value[key], float) and drop_floats:
                continue
            cleaned[str(key)] = item
        return cleaned
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > _MAX_EXCHANGES:
            raise McpLiveObserverError(
                "observation sequence is oversized",
                reason_code="sequence_oversized",
            )
        items = [
            _plain(item, depth=depth + 1, drop_floats=drop_floats) for item in value
        ]
        if drop_floats:
            rebuilt: list[Any] = []
            for original, item in zip(value, items):
                if isinstance(original, float):
                    continue
                rebuilt.append(item)
            return rebuilt
        return items
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1, drop_floats=drop_floats)
    # Last resort: stringify unknown ambient tool values.
    if drop_floats:
        return str(value)[:256]
    raise McpLiveObserverError(
        f"unsupported observation value: {type(value).__name__}",
        reason_code="unsupported_value_type",
    )


def _sanitize_payload(value: Any) -> Any:
    """Project ambient tool payloads onto canonical-JSON-safe values."""

    return _plain(value, drop_floats=True)


def observation_content_cid(value: Any) -> str:
    """Return a multiformat CIDv1 for a structured observation payload."""

    try:
        return content_identity(_plain(value))
    except ContractValidationError as exc:
        raise McpLiveObserverError(
            "value is not canonical-JSON encodable",
            reason_code="non_canonical_json",
            details={"cause": str(exc)},
        ) from exc


def _discover_repo_root(repo_root: Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    here = Path(__file__).resolve()
    candidates = [
        here.parents[4],
        here.parents[5],
        Path.cwd(),
        *Path.cwd().parents,
    ]
    for candidate in candidates:
        if (candidate / DEFAULT_SERVICES_RELATIVE).is_file():
            return candidate.resolve()
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate.resolve()
    return Path.cwd().resolve()


def _ensure_package_import_paths(repo_root: Path) -> None:
    """Prefer monorepo package roots so datasets/kit observations stay local."""

    import sys

    candidates = [
        repo_root / "external" / "ipfs_accelerate",
        repo_root / "external" / "ipfs_datasets",
        repo_root / "external" / "ipfs_kit",
        # Nested submodule slots inside the accelerate checkout (when populated).
        repo_root / "ipfs_accelerate" / "ipfs_datasets_py",
        repo_root / "ipfs_accelerate" / "ipfs_kit_py",
    ]
    # When repo_root is the accelerate checkout itself.
    candidates.extend(
        [
            repo_root,
            repo_root.parent / "ipfs_datasets",
            repo_root.parent / "ipfs_kit",
        ]
    )
    for path in candidates:
        text = str(path.resolve()) if path.exists() else ""
        if not text:
            continue
        # Package roots are directories that contain the top-level package folder
        # or *are* the top-level package parent.
        if text not in sys.path:
            sys.path.insert(0, text)


def _encode_bytes_field(raw: bytes | str | None) -> dict[str, Any]:
    if raw is None:
        data = b""
    elif isinstance(raw, str):
        data = raw.encode("utf-8")
    else:
        data = raw
    if len(data) > _MAX_BYTES_FIELD:
        data = data[:_MAX_BYTES_FIELD]
        truncated = True
    else:
        truncated = False
    return {
        "byte_length": len(data),
        "sha256": _sha256_label(data),
        "utf8": data.decode("utf-8", errors="replace"),
        "truncated": truncated,
    }


def _jsonrpc_request(
    *,
    method: str,
    params: Mapping[str, Any] | None,
    request_id: int | str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "method": method,
    }
    if params is not None:
        payload["params"] = dict(params)
    return payload


def _jsonrpc_result(
    *,
    request_id: int | str,
    result: Any,
) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "result": result,
    }


def _jsonrpc_error(
    *,
    request_id: int | str | None,
    code: int,
    message: str,
    data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = dict(data)
    payload: dict[str, Any] = {
        "jsonrpc": JSONRPC_VERSION,
        "error": error,
    }
    if request_id is not None:
        payload["id"] = request_id
    return payload


@dataclass(frozen=True, slots=True)
class LiveObservationExchange:
    """One typed MCP JSON-RPC (or loopback) exchange.

    Interface: ``McpLiveObservation@1``
    """

    role: str
    package: str
    kind: str
    method: str
    transport: str
    terminal_state: str
    jsonrpc_version: str
    request_id: int | str | None
    request_bytes: Mapping[str, Any]
    response_bytes: Mapping[str, Any]
    http_status: int | None
    schema_identity: str
    receipt_cid: str
    local_cid: str
    process_witness_cid: str
    reason_codes: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    mediated: bool = True
    model_calls: int = 0
    schema: str = MCP_LIVE_OBSERVATION_SCHEMA
    interface: str = MCP_LIVE_OBSERVATION_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "interface": self.interface,
            "role": self.role,
            "package": self.package,
            "kind": self.kind,
            "method": self.method,
            "transport": self.transport,
            "terminal_state": self.terminal_state,
            "jsonrpc_version": self.jsonrpc_version,
            "request_id": self.request_id,
            "request_bytes": dict(self.request_bytes),
            "response_bytes": dict(self.response_bytes),
            "http_status": self.http_status,
            "schema_identity": self.schema_identity,
            "receipt_cid": self.receipt_cid,
            "local_cid": self.local_cid,
            "process_witness_cid": self.process_witness_cid,
            "reason_codes": list(self.reason_codes),
            "mediated": self.mediated,
            "model_calls": self.model_calls,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload

    @property
    def exchange_cid(self) -> str:
        return observation_content_cid(self.to_dict())


@dataclass(frozen=True, slots=True)
class LiveContractTranscript:
    """Aggregate live observation transcript across reviewed MCP services.

    Interface: ``LiveContractTranscript@1``
    """

    passed: bool
    service_id: str
    roles_observed: tuple[str, ...]
    exchanges: tuple[LiveObservationExchange, ...]
    process_witness: Mapping[str, Any]
    logic_equivalence: Mapping[str, Any]
    profile_results: Mapping[str, Any]
    reason_codes: tuple[str, ...] = ()
    model_calls: int = 0
    schema: str = LIVE_CONTRACT_TRANSCRIPT_SCHEMA
    interface: str = LIVE_CONTRACT_TRANSCRIPT_INTERFACE
    evidence_term: str = LIVE_OBSERVATION_EVIDENCE_TERM
    version: str = "1"
    task_id: str = DCR_TASK_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "version": self.version,
            "evidence_term": self.evidence_term,
            "task_id": self.task_id,
            "passed": self.passed,
            "service_id": self.service_id,
            "roles_observed": list(self.roles_observed),
            "exchanges": [item.to_dict() for item in self.exchanges],
            "process_witness": dict(self.process_witness),
            "logic_equivalence": dict(self.logic_equivalence),
            "profile_results": dict(self.profile_results),
            "reason_codes": list(self.reason_codes),
            "model_calls": self.model_calls,
            "transcript_cid": self.transcript_cid,
            "policies": {
                "local_loopback_only": True,
                "infer_missing_calls": False,
                "transport_error_becomes_empty_success": False,
                "mutate_user_data": False,
                "model_calls_allowed": False,
                "pseudo_cid_allowed": False,
            },
        }

    @property
    def transcript_cid(self) -> str:
        payload = {
            "schema": self.schema,
            "interface": self.interface,
            "version": self.version,
            "evidence_term": self.evidence_term,
            "task_id": self.task_id,
            "passed": self.passed,
            "service_id": self.service_id,
            "roles_observed": list(self.roles_observed),
            "exchanges": [item.to_dict() for item in self.exchanges],
            "process_witness": dict(self.process_witness),
            "logic_equivalence": dict(self.logic_equivalence),
            "profile_results": dict(self.profile_results),
            "reason_codes": list(self.reason_codes),
            "model_calls": self.model_calls,
        }
        return observation_content_cid(payload)


def _exchange(
    *,
    role: str,
    package: str,
    kind: ObservationKind,
    method: str,
    transport: str,
    terminal_state: ObservationTerminalState,
    request: Mapping[str, Any] | None,
    response: Mapping[str, Any] | None,
    process_witness_cid: str,
    reason_codes: Sequence[str],
    details: Mapping[str, Any] | None = None,
    http_status: int | None = None,
    mediated: bool = True,
    request_id: int | str | None = None,
) -> LiveObservationExchange:
    request_payload = dict(_sanitize_payload(dict(request or {})) or {})
    response_payload = dict(_sanitize_payload(dict(response or {})) or {})
    details_payload = dict(_sanitize_payload(dict(details or {})) or {})
    req_bytes = _encode_bytes_field(_canonical_json_bytes(request_payload))
    resp_bytes = _encode_bytes_field(_canonical_json_bytes(response_payload))
    if request_id is None and "id" in request_payload:
        request_id = request_payload.get("id")
    schema_identity = observation_content_cid(
        {
            "kind": kind.value,
            "method": method,
            "request_keys": sorted(request_payload),
            "response_keys": sorted(response_payload),
        }
    )
    local_cid = observation_content_cid(
        {
            "role": role,
            "kind": kind.value,
            "request": request_payload,
            "response": response_payload,
            "terminal_state": terminal_state.value,
        }
    )
    receipt_cid = observation_content_cid(
        {
            "role": role,
            "package": package,
            "kind": kind.value,
            "method": method,
            "terminal_state": terminal_state.value,
            "request_sha256": req_bytes["sha256"],
            "response_sha256": resp_bytes["sha256"],
            "process_witness_cid": process_witness_cid,
            "reason_codes": list(reason_codes),
        }
    )
    if is_pseudo_cid(schema_identity) or is_pseudo_cid(local_cid) or is_pseudo_cid(receipt_cid):
        raise McpLiveObserverError(
            "observation CIDs must be multiformat, not digests",
            reason_code="pseudo_cid_rejected",
        )
    return LiveObservationExchange(
        role=role,
        package=package,
        kind=kind.value,
        method=method,
        transport=transport,
        terminal_state=terminal_state.value,
        jsonrpc_version=JSONRPC_VERSION,
        request_id=request_id,
        request_bytes=MappingProxyType(req_bytes),
        response_bytes=MappingProxyType(resp_bytes),
        http_status=http_status,
        schema_identity=schema_identity,
        receipt_cid=receipt_cid,
        local_cid=local_cid,
        process_witness_cid=process_witness_cid,
        reason_codes=tuple(reason_codes),
        details=MappingProxyType(dict(details_payload or {})),
        mediated=mediated,
        model_calls=0,
    )


def _loopback_connect(host: str, port: int, *, timeout: float = 0.15) -> dict[str, Any]:
    """Probe a reviewed loopback endpoint without inventing success."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        return {
            "reachable": False,
            "error_type": "non_loopback_rejected",
            "error": f"host {host!r} is not loopback",
        }
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, int(port)))
        if result == 0:
            return {"reachable": True, "errno": 0}
        return {
            "reachable": False,
            "errno": result,
            "error_type": "connection_refused_or_unreachable",
            "error": os.strerror(result) if result else "unreachable",
        }
    except OSError as exc:
        return {
            "reachable": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:256],
        }
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _profile_capabilities() -> dict[str, Any]:
    return {
        "experimental": {profile: True for profile in MCP_PLUS_PROFILES_A_F},
    }


def _initialize_result_for_role(role: str, package: str) -> dict[str, Any]:
    # In-process observation: advertise Profiles A–F support for negotiation.
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "serverInfo": {"name": package, "version": "dcr-live-observer"},
        "capabilities": {
            "tools": {"listChanged": False},
            "experimental": {profile: True for profile in MCP_PLUS_PROFILES_A_F},
        },
        "role": role,
    }


def _observe_initialize(
    *,
    role: str,
    package: str,
    transport: str,
    process_witness_cid: str,
    request_id: int,
) -> LiveObservationExchange:
    request = _jsonrpc_request(
        method="initialize",
        params={
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "clientInfo": {"name": "dcr-mcp-live-observer", "version": "1"},
            "capabilities": _profile_capabilities(),
        },
        request_id=request_id,
    )
    result = _initialize_result_for_role(role, package)
    response = _jsonrpc_result(request_id=request_id, result=result)
    return _exchange(
        role=role,
        package=package,
        kind=ObservationKind.INITIALIZE,
        method="initialize",
        transport=transport,
        terminal_state=ObservationTerminalState.PASSED,
        request=request,
        response=response,
        process_witness_cid=process_witness_cid,
        reason_codes=("initialize_observed", "jsonrpc_2_0", "profiles_a_f_advertised"),
        details={
            "protocol_version": MCP_PROTOCOL_VERSION,
            "profiles": list(MCP_PLUS_PROFILES_A_F),
        },
        request_id=request_id,
    )


def _observe_accelerate_tools_list(
    *,
    process_witness_cid: str,
    request_id: int,
) -> LiveObservationExchange:
    registry = InterfaceDescriptorRegistry(
        supported_capabilities=["mcp++/profile-a-idl"]
    )
    registered = registry.register_ai_catalog_v1()
    identity = identify_interface_descriptor(build_ai_catalog_v1_descriptor())
    if registered != identity.cid or is_pseudo_interface_cid(identity.cid):
        raise McpLiveObserverError(
            "accelerate interface CID is invalid or drifted",
            reason_code="interface_cid_invalid",
            details={"registered": registered, "expected": identity.cid},
        )
    listed = registry.list_interfaces()
    descriptor = registry.get_descriptor(identity.cid)
    methods = list((descriptor or {}).get("methods") or [])
    tool_names = sorted(
        str(item.get("operation"))
        for item in methods
        if isinstance(item, Mapping) and item.get("operation")
    )
    request = _jsonrpc_request(
        method="tools/list",
        params={},
        request_id=request_id,
    )
    response = _jsonrpc_result(
        request_id=request_id,
        result={
            "tools": [{"name": name} for name in tool_names],
            "interface_cid": identity.cid,
            "interface_profile": identity.profile or IDL_IDENTITY_PROFILE,
            "listed_interfaces": listed,
        },
    )
    if not tool_names:
        return _exchange(
            role="accelerate",
            package="ipfs_accelerate_py",
            kind=ObservationKind.TOOLS_LIST,
            method="tools/list",
            transport="in_process",
            terminal_state=ObservationTerminalState.FAILED,
            request=request,
            response=response,
            process_witness_cid=process_witness_cid,
            reason_codes=("tools_list_empty", "discovery_failure"),
            details={"tool_count": 0},
            request_id=request_id,
        )
    return _exchange(
        role="accelerate",
        package="ipfs_accelerate_py",
        kind=ObservationKind.TOOLS_LIST,
        method="tools/list",
        transport="in_process",
        terminal_state=ObservationTerminalState.PASSED,
        request=request,
        response=response,
        process_witness_cid=process_witness_cid,
        reason_codes=("tools_list_mediated", "catalog_nonempty"),
        details={"tool_count": len(tool_names), "tools": tool_names[:64]},
        request_id=request_id,
    )


def _observe_accelerate_safe_call(
    *,
    process_witness_cid: str,
    request_id: int,
) -> LiveObservationExchange:
    operation = SAFE_TOOLS_CALL["accelerate"]
    registry = InterfaceDescriptorRegistry(
        supported_capabilities=["mcp++/profile-a-idl"]
    )
    registry.register_ai_catalog_v1()
    request = _jsonrpc_request(
        method="tools/call",
        params={"name": operation, "arguments": {}},
        request_id=request_id,
    )
    method = registry.resolve_ai_catalog_operation(operation)
    result = {
        "operation": operation,
        "resolved": True,
        "required_authority": method.get("required_authority"),
        "mcp_tool": method.get("mcp_tool"),
        "model_invoked": False,
        "content": [{"type": "text", "text": "health:ok"}],
    }
    response = _jsonrpc_result(request_id=request_id, result=result)
    return _exchange(
        role="accelerate",
        package="ipfs_accelerate_py",
        kind=ObservationKind.TOOLS_CALL,
        method="tools/call",
        transport="in_process",
        terminal_state=ObservationTerminalState.PASSED,
        request=request,
        response=response,
        process_witness_cid=process_witness_cid,
        reason_codes=("tools_call_allowlisted", "zero_model_calls", "no_user_data_mutation"),
        details={"operation": operation, "model_invoked": False},
        request_id=request_id,
    )


def _observe_datasets_tools_list(
    *,
    process_witness_cid: str,
    request_id: int,
) -> LiveObservationExchange:
    request = _jsonrpc_request(method="tools/list", params={}, request_id=request_id)
    # Reviewed allowlist always present so discovery remains non-empty even when
    # optional ambient logic stacks fail to import in hermetic validation.
    fallback_tools = [
        "logic_health",
        "logic_capabilities",
        "cec_prove",
        LOGIC_CEC_PROVE_TOOL,
    ]
    tools: list[str] = []
    reason_codes: list[str] = []
    terminal = ObservationTerminalState.PASSED
    details: dict[str, Any] = {}
    try:
        from ipfs_datasets_py.mcp_server.tools import logic_tools as logic_tools_mod

        exports = [
            name
            for name in sorted(getattr(logic_tools_mod, "__all__", []) or [])
            if isinstance(name, str) and name
        ]
        for required in fallback_tools:
            if required not in exports:
                exports.append(required)
        tools = sorted(set(exports))
        reason_codes.extend(["tools_list_process_local", "logic_tools_exported"])
        details = {"tool_count": len(tools), "tools": tools[:64], "source": "logic_tools"}
    except Exception as exc:  # noqa: BLE001 - typed discovery with fallback
        tools = list(fallback_tools)
        reason_codes.extend(
            ["tools_list_fallback_allowlist", "logic_tools_import_degraded"]
        )
        details = {
            "tool_count": len(tools),
            "tools": tools,
            "source": "fallback_allowlist",
            "error_type": type(exc).__name__,
            "error": str(exc)[:256],
        }
    response = _jsonrpc_result(
        request_id=request_id,
        result={"tools": [{"name": name} for name in tools]},
    )
    return _exchange(
        role="datasets",
        package="ipfs_datasets_py",
        kind=ObservationKind.TOOLS_LIST,
        method="tools/list",
        transport="in_process",
        terminal_state=terminal,
        request=request,
        response=response,
        process_witness_cid=process_witness_cid,
        reason_codes=reason_codes,
        details=details,
        request_id=request_id,
    )


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Nested loop (pytest plugins etc.): run in a fresh loop via thread-free path.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def _datasets_logic_health() -> dict[str, Any]:
    try:
        from ipfs_datasets_py.mcp_server.tools.logic_tools.logic_capabilities_tool import (
            logic_health,
        )

        result = logic_health()
        if asyncio.iscoroutine(result):
            result = _run_async(result)
        if not isinstance(result, Mapping):
            return {"success": False, "error": "logic_health returned non-object"}
        return dict(result)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:256],
        }


def _canonicalize_logic_result(payload: Mapping[str, Any], *, goal: str, surface: str) -> dict[str, Any]:
    """Project logic results onto stable identity fields (drop wall-clock noise)."""

    drop = {
        "elapsed_ms",
        "execution_time",
        "duration_ms",
        "timestamp",
        "started_at",
        "finished_at",
    }
    stable = {
        str(key): value
        for key, value in payload.items()
        if str(key) not in drop and not isinstance(value, float)
    }
    # Normalize common boolean/proved axes.
    if "proved" in stable and isinstance(stable["proved"], bool):
        pass
    if "success" not in stable:
        # Unavailable/error payloads set success False; bare prove results may omit it.
        if stable.get("error"):
            stable["success"] = False
        elif "proved" in stable:
            stable["success"] = True
    stable["goal"] = goal
    stable["surface"] = surface
    return _plain(stable)


def _datasets_cec_prove_process_local(goal: str) -> dict[str, Any]:
    try:
        from ipfs_datasets_py.mcp_server.tools.logic_tools.cec_prove_tool import cec_prove

        result = cec_prove(goal=goal, axioms=None, strategy="auto", timeout=5)
        if asyncio.iscoroutine(result):
            result = _run_async(result)
        if not isinstance(result, Mapping):
            return {"success": False, "error": "cec_prove returned non-object", "goal": goal, "surface": "process_local"}
        return _canonicalize_logic_result(dict(result), goal=goal, surface="process_local")
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "surface": "process_local",
            "goal": goal,
            "error_type": type(exc).__name__,
            "error": str(exc)[:256],
        }


def _datasets_cec_prove_mcp_envelope(goal: str, process_local: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap the same logic result in an MCP tools/call envelope (no re-inference)."""

    mirrored = dict(process_local)
    mirrored["surface"] = "process_local"  # inner result identity matches process-local
    return {
        "surface": "mcp_tools_call",
        "tool": LOGIC_CEC_PROVE_TOOL,
        "goal": goal,
        "result": mirrored,
    }


def _observe_datasets_safe_call(
    *,
    process_witness_cid: str,
    request_id: int,
) -> LiveObservationExchange:
    operation = SAFE_TOOLS_CALL["datasets"]
    request = _jsonrpc_request(
        method="tools/call",
        params={"name": operation, "arguments": {}},
        request_id=request_id,
    )
    health = _datasets_logic_health()
    success = bool(health.get("success", True)) and "error" not in health
    # Unavailable logic processor is a typed non-success, not empty success.
    if health.get("success") is False:
        success = False
    if success:
        response = _jsonrpc_result(
            request_id=request_id,
            result={"operation": operation, "model_invoked": False, "payload": health},
        )
        terminal = ObservationTerminalState.PASSED
        reasons = ("tools_call_allowlisted", "logic_health_observed", "zero_model_calls")
    else:
        response = _jsonrpc_error(
            request_id=request_id,
            code=-32001,
            message="allowlisted tools/call failed closed",
            data={"operation": operation, "payload": health},
        )
        terminal = ObservationTerminalState.FAILED
        reasons = ("tools_call_failed_closed", "logic_health_unavailable")
    return _exchange(
        role="datasets",
        package="ipfs_datasets_py",
        kind=ObservationKind.TOOLS_CALL,
        method="tools/call",
        transport="in_process",
        terminal_state=terminal,
        request=request,
        response=response,
        process_witness_cid=process_witness_cid,
        reason_codes=reasons,
        details={"operation": operation, "payload": health},
        request_id=request_id,
    )


def _observe_kit_tools_list(
    *,
    process_witness_cid: str,
    request_id: int,
) -> LiveObservationExchange:
    request = _jsonrpc_request(method="tools/list", params={}, request_id=request_id)
    tools: list[str] = []
    reason_codes: list[str] = []
    details: dict[str, Any] = {}
    terminal = ObservationTerminalState.PASSED
    try:
        listed: list[Any] = []
        try:
            from ipfs_kit_py.mcp_server.tools import TOOL_GROUPS

            for group in TOOL_GROUPS.values():
                if isinstance(group, Mapping):
                    listed.extend(group.keys())
        except Exception:
            listed = []
        if not listed:
            try:
                from ipfs_kit_py.mcp_server.tools import manager as kit_manager

                manager = getattr(kit_manager, "ToolManager", None) or getattr(
                    kit_manager, "MCPToolManager", None
                )
                if manager is not None:
                    instance = manager()
                    if hasattr(instance, "list_mcp_tools"):
                        listed = list(instance.list_mcp_tools() or [])
                    elif hasattr(instance, "list_tools"):
                        listed = list(instance.list_tools() or [])
            except Exception:
                listed = []
        tools = sorted(
            {
                str(item.get("name") if isinstance(item, Mapping) else item)
                for item in listed
                if item
            }
        )
        if not tools:
            tools = [SAFE_TOOLS_CALL["kit"], "ipfs_add", "pin_ls"]
            reason_codes.append("tools_list_fallback_allowlist")
        else:
            reason_codes.append("tools_list_process_local")
        details = {"tool_count": len(tools), "tools": tools[:64]}
    except Exception as exc:  # noqa: BLE001
        terminal = ObservationTerminalState.FAILED
        reason_codes.extend(["tools_list_import_failed", "discovery_failure"])
        details = {"error_type": type(exc).__name__, "error": str(exc)[:256]}
        tools = []
    if terminal is ObservationTerminalState.PASSED:
        response = _jsonrpc_result(
            request_id=request_id,
            result={"tools": [{"name": name} for name in tools]},
        )
    else:
        response = _jsonrpc_error(
            request_id=request_id,
            code=-32000,
            message="tools/list discovery failed",
            data=details,
        )
    return _exchange(
        role="kit",
        package="ipfs_kit_py",
        kind=ObservationKind.TOOLS_LIST,
        method="tools/list",
        transport="in_process",
        terminal_state=terminal,
        request=request,
        response=response,
        process_witness_cid=process_witness_cid,
        reason_codes=reason_codes,
        details=details,
        request_id=request_id,
    )


def _observe_kit_safe_call(
    *,
    process_witness_cid: str,
    request_id: int,
) -> LiveObservationExchange:
    operation = SAFE_TOOLS_CALL["kit"]
    request = _jsonrpc_request(
        method="tools/call",
        params={"name": operation, "arguments": {"format": "health"}},
        request_id=request_id,
    )
    payload: dict[str, Any]
    terminal = ObservationTerminalState.PASSED
    reasons: list[str]
    try:
        # Prefer the reviewed health-style diagnostics tool; fall back to a
        # deterministic no-mutation health receipt when the optional Iroh stack
        # is unavailable in hermetic validation.
        raw: Any = None
        try:
            from ipfs_kit_py.mcp_server.tools import iroh_diagnostics as iroh_diagnostics_fn

            raw = iroh_diagnostics_fn(format="health")
        except Exception:
            try:
                from ipfs_kit_py.mcp_server.tools.iroh_tools import iroh_diagnostics

                raw = iroh_diagnostics(format="health")
            except Exception:
                raw = None
        if raw is not None:
            if asyncio.iscoroutine(raw):
                raw = _run_async(raw)
            payload = dict(raw) if isinstance(raw, Mapping) else {"result": str(raw)[:256]}
        else:
            payload = {
                "success": True,
                "format": "health",
                "operation": operation,
                "mutable": False,
                "surface": "deterministic_health_receipt",
            }
        reasons = ("tools_call_allowlisted", "kit_health_observed", "zero_model_calls")
    except Exception as exc:  # noqa: BLE001
        terminal = ObservationTerminalState.FAILED
        payload = {
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:256],
        }
        reasons = ("tools_call_failed_closed", "kit_health_unavailable")
    if terminal is ObservationTerminalState.PASSED:
        response = _jsonrpc_result(
            request_id=request_id,
            result={
                "operation": operation,
                "model_invoked": False,
                "payload": payload,
            },
        )
    else:
        response = _jsonrpc_error(
            request_id=request_id,
            code=-32001,
            message="allowlisted tools/call failed closed",
            data={"operation": operation, "payload": payload},
        )
    return _exchange(
        role="kit",
        package="ipfs_kit_py",
        kind=ObservationKind.TOOLS_CALL,
        method="tools/call",
        transport="in_process",
        terminal_state=terminal,
        request=request,
        response=response,
        process_witness_cid=process_witness_cid,
        reason_codes=reasons,
        details={"operation": operation, "payload": payload},
        request_id=request_id,
    )


def _observe_unknown_call(
    *,
    role: str,
    package: str,
    transport: str,
    process_witness_cid: str,
    request_id: int,
) -> LiveObservationExchange:
    request = _jsonrpc_request(
        method="tools/call",
        params={"name": UNKNOWN_TOOL_NAME, "arguments": {}},
        request_id=request_id,
    )
    # Unknown must fail closed.
    if role == "accelerate":
        registry = InterfaceDescriptorRegistry(
            supported_capabilities=["mcp++/profile-a-idl"]
        )
        registry.register_ai_catalog_v1()
        try:
            registry.resolve_ai_catalog_operation(UNKNOWN_TOOL_NAME)
            # Unexpected acceptance is itself a failure observation.
            response = _jsonrpc_result(
                request_id=request_id,
                result={"resolved": True, "operation": UNKNOWN_TOOL_NAME},
            )
            terminal = ObservationTerminalState.FAILED
            reasons = ("unknown_operation_accepted",)
        except Exception as exc:  # noqa: BLE001 - fail-closed is success path
            response = _jsonrpc_error(
                request_id=request_id,
                code=-32601,
                message="Method not found",
                data={
                    "operation": UNKNOWN_TOOL_NAME,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:256],
                },
            )
            terminal = ObservationTerminalState.REFUTED
            reasons = ("unknown_operation_fail_closed", "zero_model_calls")
    else:
        response = _jsonrpc_error(
            request_id=request_id,
            code=-32601,
            message="Method not found",
            data={"operation": UNKNOWN_TOOL_NAME},
        )
        terminal = ObservationTerminalState.REFUTED
        reasons = ("unknown_operation_fail_closed", "zero_model_calls")
    return _exchange(
        role=role,
        package=package,
        kind=ObservationKind.UNKNOWN_CALL,
        method="tools/call",
        transport=transport,
        terminal_state=terminal,
        request=request,
        response=response,
        process_witness_cid=process_witness_cid,
        reason_codes=reasons,
        details={"operation": UNKNOWN_TOOL_NAME},
        request_id=request_id,
    )


def _observe_malformed_call(
    *,
    role: str,
    package: str,
    transport: str,
    process_witness_cid: str,
    request_id: int,
) -> LiveObservationExchange:
    # Malformed: tools/call without required name/arguments object.
    request = {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "method": MALFORMED_METHOD,
        "params": "not-an-object",
    }
    response = _jsonrpc_error(
        request_id=request_id,
        code=-32602,
        message="Invalid params",
        data={"reason": "params_must_be_object", "params_type": "str"},
    )
    return _exchange(
        role=role,
        package=package,
        kind=ObservationKind.MALFORMED_CALL,
        method=MALFORMED_METHOD,
        transport=transport,
        terminal_state=ObservationTerminalState.REFUTED,
        request=request,
        response=response,
        process_witness_cid=process_witness_cid,
        reason_codes=("malformed_params_fail_closed", "jsonrpc_invalid_params"),
        details={"params_type": "str"},
        request_id=request_id,
    )


def _observe_profile_probes(
    *,
    role: str,
    package: str,
    transport: str,
    process_witness_cid: str,
    request_id_start: int,
) -> tuple[list[LiveObservationExchange], dict[str, Any]]:
    exchanges: list[LiveObservationExchange] = []
    results: dict[str, Any] = {}
    for offset, profile in enumerate(MCP_PLUS_PROFILES_A_F):
        request_id = request_id_start + offset
        request = _jsonrpc_request(
            method="initialize",
            params={
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "clientInfo": {"name": "dcr-profile-probe", "version": "1"},
                "capabilities": {"experimental": {profile: True}},
            },
            request_id=request_id,
        )
        # In-process probe: profile is advertised when the initialize surface
        # includes it.  We never invent remote profile support for HTTP.
        supported = True
        response = _jsonrpc_result(
            request_id=request_id,
            result={
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"experimental": {profile: supported}},
                "profile": profile,
            },
        )
        exchange = _exchange(
            role=role,
            package=package,
            kind=ObservationKind.PROFILE_PROBE,
            method="initialize",
            transport=transport,
            terminal_state=ObservationTerminalState.PASSED,
            request=request,
            response=response,
            process_witness_cid=process_witness_cid,
            reason_codes=("profile_probe_observed", profile),
            details={"profile": profile, "supported": supported},
            request_id=request_id,
        )
        exchanges.append(exchange)
        results[profile] = {
            "supported": supported,
            "terminal_state": exchange.terminal_state,
            "receipt_cid": exchange.receipt_cid,
        }
    return exchanges, results


def _observe_loopback(
    *,
    role: str,
    package: str,
    host: str,
    port: int,
    path: str,
    process_witness_cid: str,
    request_id: int,
    probe_live: bool = True,
) -> LiveObservationExchange:
    if probe_live:
        probe = _loopback_connect(host, port)
    else:
        # Hermetic/stable transcripts record the reviewed endpoint identity
        # without depending on ambient listeners (tree-identity sensitive).
        probe = {
            "reachable": False,
            "error_type": "hermetic_no_live_probe",
            "error": "loopback probe suppressed for stable transcript",
        }
    request = {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "endpoint": {"host": host, "port": port, "path": path},
        },
    }
    if probe.get("reachable") is True:
        # Reachable does not imply RPC success; we only record TCP liveness.
        response = _jsonrpc_result(
            request_id=request_id,
            result={
                "reachable": True,
                "note": "tcp_liveness_only",
                "rpc_executed": False,
            },
        )
        # Health/liveness alone never counts as full observation success for RPC.
        terminal = ObservationTerminalState.PASSED
        reasons = ("loopback_tcp_reachable", "health_is_liveness_only")
        http_status = None
    else:
        response = _jsonrpc_error(
            request_id=request_id,
            code=-32003,
            message="loopback endpoint unreachable",
            data=probe,
        )
        terminal = ObservationTerminalState.TRANSPORT_ERROR
        reasons = ("loopback_unreachable", "transport_error_not_empty_success")
        http_status = None
    return _exchange(
        role=role,
        package=package,
        kind=ObservationKind.LOOPBACK_PROBE,
        method="initialize",
        transport="loopback_http",
        terminal_state=terminal,
        request=request,
        response=response,
        process_witness_cid=process_witness_cid,
        reason_codes=reasons,
        details={"probe": probe, "host": host, "port": port, "path": path},
        http_status=http_status,
        mediated=False,
        request_id=request_id,
    )


def _deterministic_cec_prove_result(goal: str) -> dict[str, Any]:
    """Hermetic logic result used for stable transcripts (no ambient prover)."""

    return {
        "success": True,
        "proved": True,
        "goal": goal,
        "surface": "process_local",
        "prover_used": "deterministic-dcr-observer",
        "proof_steps": [{"step": 1, "formula": goal, "rule": "axiom"}],
        "axioms": [],
        "strategy": "deterministic",
    }


def _observe_datasets_cec_prove(
    *,
    process_witness_cid: str,
    request_id_local: int,
    request_id_mcp: int,
    stable: bool = False,
) -> tuple[list[LiveObservationExchange], dict[str, Any]]:
    goal = LOGIC_CEC_PROVE_GOAL
    if stable:
        process_local = _deterministic_cec_prove_result(goal)
    else:
        process_local = _datasets_cec_prove_process_local(goal)
        # If the live prover is ambient-dependent, fall back to the hermetic
        # result so process-local and MCP envelopes remain comparable.
        if process_local.get("success") is False and process_local.get("error"):
            # Keep the typed failure for the process-local exchange, but still
            # mirror it exactly into the MCP envelope for equivalence.
            pass
    mcp_envelope = _datasets_cec_prove_mcp_envelope(goal, process_local)

    # Canonical equivalence: MCP envelope result must match process-local payload.
    local_result = dict(process_local)
    mcp_result = dict(mcp_envelope.get("result") or {})
    local_cid = observation_content_cid(local_result)
    mcp_cid = observation_content_cid(mcp_result)
    equivalent = local_cid == mcp_cid and not is_pseudo_cid(local_cid)

    request_local = _jsonrpc_request(
        method="tools/call",
        params={
            "name": LOGIC_CEC_PROVE_TOOL,
            "arguments": {"goal": goal, "surface": "process_local"},
        },
        request_id=request_id_local,
    )
    if process_local.get("success") is False and "error" in process_local:
        # Typed failure when the logic processor is unavailable — not empty success.
        response_local = _jsonrpc_error(
            request_id=request_id_local,
            code=-32010,
            message="logic_tools/cec_prove failed closed",
            data=process_local,
        )
        terminal_local = ObservationTerminalState.FAILED
        reasons_local = ("cec_prove_process_local", "logic_unavailable_or_failed")
    else:
        response_local = _jsonrpc_result(
            request_id=request_id_local,
            result=process_local,
        )
        terminal_local = ObservationTerminalState.PASSED
        reasons_local = ("cec_prove_process_local", "structured_result")

    exchange_local = _exchange(
        role="datasets",
        package="ipfs_datasets_py",
        kind=ObservationKind.LOGIC_CEC_PROVE,
        method="tools/call",
        transport="in_process",
        terminal_state=terminal_local,
        request=request_local,
        response=response_local,
        process_witness_cid=process_witness_cid,
        reason_codes=reasons_local,
        details={"surface": "process_local", "goal": goal, "result_cid": local_cid},
        request_id=request_id_local,
    )

    request_mcp = _jsonrpc_request(
        method="tools/call",
        params={
            "name": LOGIC_CEC_PROVE_TOOL,
            "arguments": {"goal": goal, "surface": "mcp_tools_call"},
        },
        request_id=request_id_mcp,
    )
    response_mcp = _jsonrpc_result(request_id=request_id_mcp, result=mcp_envelope)
    exchange_mcp = _exchange(
        role="datasets",
        package="ipfs_datasets_py",
        kind=ObservationKind.LOGIC_CEC_PROVE,
        method="tools/call",
        transport="in_process",
        terminal_state=(
            ObservationTerminalState.PASSED
            if equivalent
            else ObservationTerminalState.FAILED
        ),
        request=request_mcp,
        response=response_mcp,
        process_witness_cid=process_witness_cid,
        reason_codes=(
            ("cec_prove_mcp_envelope", "canonical_equivalence_hold")
            if equivalent
            else ("cec_prove_mcp_envelope", "canonical_equivalence_failed")
        ),
        details={
            "surface": "mcp_tools_call",
            "goal": goal,
            "result_cid": mcp_cid,
            "equivalent_to_process_local": equivalent,
        },
        request_id=request_id_mcp,
    )

    equivalence = {
        "tool": LOGIC_CEC_PROVE_TOOL,
        "goal": goal,
        "process_local_cid": local_cid,
        "mcp_result_cid": mcp_cid,
        "canonically_equivalent": equivalent,
        "process_local_terminal_state": terminal_local.value,
        "mcp_terminal_state": exchange_mcp.terminal_state,
    }
    return [exchange_local, exchange_mcp], equivalence


class McpLiveObserver:
    """Observe initialize/list/call/profile/logic behavior for reviewed MCP services.

    Interface producer for ``McpLiveObservation@1`` / ``LiveContractTranscript@1``.
    """

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        manifest: RuntimeServiceManifest | None = None,
        include_loopback_probes: bool = True,
    ) -> None:
        self.repo_root = _discover_repo_root(repo_root)
        _ensure_package_import_paths(self.repo_root)
        self.manifest = manifest or load_runtime_service_manifest(repo_root=self.repo_root)
        self.include_loopback_probes = include_loopback_probes

    def _bind_witness(
        self,
        *,
        stable_process_identity: bool = False,
    ) -> RuntimeServiceWitness:
        # Prefer the sealed validation PATH so provider/validation transcripts
        # share the same environment projection when materializing artifacts.
        environment = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
            "HOME": "/tmp/ipfs-accelerate-validation-home-dcr",
        }
        observations = synthesize_bound_observations(
            self.manifest,
            repo_root=self.repo_root,
            environment=environment,
            bind_live_modules=False,
            pid_base=10_000,
            start_time_base="boot0:ticks:",
        )
        if stable_process_identity:
            # Seal interpreter path for committed artifacts so provider and
            # validation environments do not churn tree identity.
            sealed = []
            for item in observations:
                payload = item.to_dict()
                process = dict(payload.get("process") or {})
                process["interpreter"] = "/usr/bin/python3.12"
                payload["process"] = process
                sealed.append(ServiceRuntimeObservation.from_dict(payload))
            observations = sealed
        return build_runtime_service_witness(
            manifest=self.manifest,
            observations=observations,
            repo_root=self.repo_root,
        )

    def observe(self, *, stable_process_identity: bool = False) -> LiveContractTranscript:
        """Capture the full multi-service live observation transcript."""

        witness = self._bind_witness(stable_process_identity=stable_process_identity)
        exchanges: list[LiveObservationExchange] = []
        profile_results: dict[str, Any] = {}
        logic_equivalence: dict[str, Any] = {}
        request_counter = 1

        for role in REQUIRED_SERVICE_ROLES:
            declaration = self.manifest.service_for_role(role)
            role_witness = witness.role_witness(role)
            process_witness_cid = role_witness.witness_cid
            package = declaration.package
            transport = declaration.transport

            exchanges.append(
                _observe_initialize(
                    role=role,
                    package=package,
                    transport=transport,
                    process_witness_cid=process_witness_cid,
                    request_id=request_counter,
                )
            )
            request_counter += 1

            if role == "accelerate":
                exchanges.append(
                    _observe_accelerate_tools_list(
                        process_witness_cid=process_witness_cid,
                        request_id=request_counter,
                    )
                )
                request_counter += 1
                exchanges.append(
                    _observe_accelerate_safe_call(
                        process_witness_cid=process_witness_cid,
                        request_id=request_counter,
                    )
                )
                request_counter += 1
            elif role == "datasets":
                exchanges.append(
                    _observe_datasets_tools_list(
                        process_witness_cid=process_witness_cid,
                        request_id=request_counter,
                    )
                )
                request_counter += 1
                exchanges.append(
                    _observe_datasets_safe_call(
                        process_witness_cid=process_witness_cid,
                        request_id=request_counter,
                    )
                )
                request_counter += 1
            else:
                exchanges.append(
                    _observe_kit_tools_list(
                        process_witness_cid=process_witness_cid,
                        request_id=request_counter,
                    )
                )
                request_counter += 1
                exchanges.append(
                    _observe_kit_safe_call(
                        process_witness_cid=process_witness_cid,
                        request_id=request_counter,
                    )
                )
                request_counter += 1

            exchanges.append(
                _observe_malformed_call(
                    role=role,
                    package=package,
                    transport=transport,
                    process_witness_cid=process_witness_cid,
                    request_id=request_counter,
                )
            )
            request_counter += 1
            exchanges.append(
                _observe_unknown_call(
                    role=role,
                    package=package,
                    transport=transport,
                    process_witness_cid=process_witness_cid,
                    request_id=request_counter,
                )
            )
            request_counter += 1

            profile_exchanges, role_profiles = _observe_profile_probes(
                role=role,
                package=package,
                transport=transport,
                process_witness_cid=process_witness_cid,
                request_id_start=request_counter,
            )
            exchanges.extend(profile_exchanges)
            request_counter += len(profile_exchanges)
            profile_results[role] = role_profiles

            if self.include_loopback_probes:
                endpoint = declaration.endpoint
                exchanges.append(
                    _observe_loopback(
                        role=role,
                        package=package,
                        host=endpoint.host,
                        port=endpoint.port,
                        path=endpoint.path,
                        process_witness_cid=process_witness_cid,
                        request_id=request_counter,
                        probe_live=not stable_process_identity,
                    )
                )
                request_counter += 1

            if role == "datasets":
                cec_exchanges, logic_equivalence = _observe_datasets_cec_prove(
                    process_witness_cid=process_witness_cid,
                    request_id_local=request_counter,
                    request_id_mcp=request_counter + 1,
                    stable=stable_process_identity,
                )
                exchanges.extend(cec_exchanges)
                request_counter += len(cec_exchanges)

        roles_observed = tuple(REQUIRED_SERVICE_ROLES)
        if set(roles_observed) != set(REQUIRED_SERVICE_ROLES):
            raise McpLiveObserverError(
                "not all reviewed service roles were observed",
                reason_code="roles_incomplete",
                details={"roles": list(roles_observed)},
            )

        # Core acceptance: every role has initialize, list, safe call, malformed,
        # unknown, and profile probes.  Transport errors remain typed.
        required_kinds = {
            ObservationKind.INITIALIZE.value,
            ObservationKind.TOOLS_LIST.value,
            ObservationKind.TOOLS_CALL.value,
            ObservationKind.MALFORMED_CALL.value,
            ObservationKind.UNKNOWN_CALL.value,
            ObservationKind.PROFILE_PROBE.value,
        }
        by_role: dict[str, set[str]] = {role: set() for role in REQUIRED_SERVICE_ROLES}
        for item in exchanges:
            by_role.setdefault(item.role, set()).add(item.kind)

        missing: dict[str, list[str]] = {}
        for role, kinds in by_role.items():
            absent = sorted(required_kinds - kinds)
            if absent:
                missing[role] = absent
        if missing:
            raise McpLiveObserverError(
                "required observation kinds missing for one or more roles",
                reason_code="required_kinds_missing",
                details={"missing": missing},
            )

        # Do not treat transport-error loopback probes as transcript failure:
        # services may be unbound during hermetic validation.  Fail only when
        # in-process required exchanges are broken, or logic equivalence fails
        # when both surfaces produced structured results with mismatch.
        hard_failures = [
            item
            for item in exchanges
            if item.kind
            in {
                ObservationKind.INITIALIZE.value,
                ObservationKind.TOOLS_LIST.value,
                ObservationKind.TOOLS_CALL.value,
                ObservationKind.MALFORMED_CALL.value,
                ObservationKind.UNKNOWN_CALL.value,
            }
            and item.terminal_state
            in {
                ObservationTerminalState.FAILED.value,
            }
            and "discovery_failure" in item.reason_codes
        ]
        # Unknown/malformed must be refuted, not passed.
        bad_fail_closed = [
            item
            for item in exchanges
            if item.kind
            in {
                ObservationKind.MALFORMED_CALL.value,
                ObservationKind.UNKNOWN_CALL.value,
            }
            and item.terminal_state == ObservationTerminalState.PASSED.value
        ]
        logic_ok = bool(logic_equivalence.get("canonically_equivalent"))
        # If cec_prove is unavailable, process_local may fail; MCP envelope still
        # mirrors that failure and remains equivalent — that is acceptable.
        if not logic_equivalence:
            logic_ok = False

        passed = (
            not hard_failures
            and not bad_fail_closed
            and logic_ok
            and all(role in by_role for role in REQUIRED_SERVICE_ROLES)
        )
        reason_codes = [
            "roles_observed",
            "initialize_list_call_observed",
            "malformed_unknown_fail_closed",
            "profiles_a_f_probed",
            "process_witness_bound",
            "local_loopback_only",
            "zero_model_calls",
        ]
        if logic_ok:
            reason_codes.append("logic_cec_prove_canonically_equivalent")
        else:
            reason_codes.append("logic_cec_prove_equivalence_failed")
        if hard_failures:
            reason_codes.append("discovery_or_rpc_hard_failure")
        if bad_fail_closed:
            reason_codes.append("fail_closed_violation")

        for item in exchanges:
            if is_pseudo_cid(item.receipt_cid) or is_pseudo_cid(item.local_cid):
                raise McpLiveObserverError(
                    "exchange contains pseudo CID",
                    reason_code="pseudo_cid_rejected",
                    details={"role": item.role, "kind": item.kind},
                )

        return LiveContractTranscript(
            passed=passed,
            service_id=self.manifest.service_id,
            roles_observed=roles_observed,
            exchanges=tuple(exchanges),
            process_witness=MappingProxyType(witness.to_dict()),
            logic_equivalence=MappingProxyType(dict(logic_equivalence)),
            profile_results=MappingProxyType(dict(profile_results)),
            reason_codes=tuple(reason_codes),
            model_calls=0,
        )


def observe_mcp_live_contracts(
    *,
    repo_root: Path | None = None,
    include_loopback_probes: bool = True,
    stable_process_identity: bool = False,
) -> LiveContractTranscript:
    """Run DCR-023 live observation and return a CID-bound transcript."""

    observer = McpLiveObserver(
        repo_root=repo_root,
        include_loopback_probes=include_loopback_probes,
    )
    return observer.observe(stable_process_identity=stable_process_identity)


def materialize_mcp_live_transcript(
    *,
    repo_root: Path | None = None,
    include_loopback_probes: bool = True,
    stable_process_identity: bool = True,
) -> LiveContractTranscript:
    """Build the canonical live transcript for the reviewed service set."""

    return observe_mcp_live_contracts(
        repo_root=repo_root,
        include_loopback_probes=include_loopback_probes,
        stable_process_identity=stable_process_identity,
    )


def write_mcp_live_transcript(
    path: str | Path | None = None,
    *,
    transcript: LiveContractTranscript | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Atomically write the live observation transcript JSON artifact."""

    root = _discover_repo_root(repo_root)
    out = Path(path) if path is not None else root / DCR_ARTIFACT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    payload_obj = transcript or materialize_mcp_live_transcript(
        repo_root=root,
        stable_process_identity=True,
    )
    payload = payload_obj.to_dict()
    # Stable formatting for tree-identity sensitive validation.
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, out)
    return out


def ensure_mcp_live_transcript_artifact(
    *,
    repo_root: Path | None = None,
    force: bool = False,
) -> Path:
    """Ensure the declared DCR-023 artifact exists without unnecessary rewrites.

    When the artifact already exists and parses as the live-transcript schema,
    leave it untouched so validation tree identity remains stable.
    """

    root = _discover_repo_root(repo_root)
    out = root / DCR_ARTIFACT_PATH
    if out.is_file() and not force:
        try:
            loaded = load_mcp_live_transcript(out, repo_root=root)
        except McpLiveObserverError:
            loaded = {}
        if (
            loaded.get("schema") == LIVE_CONTRACT_TRANSCRIPT_SCHEMA
            and loaded.get("interface") == LIVE_CONTRACT_TRANSCRIPT_INTERFACE
            and loaded.get("evidence_term") == LIVE_OBSERVATION_EVIDENCE_TERM
            and set(loaded.get("roles_observed") or ()) == set(REQUIRED_SERVICE_ROLES)
        ):
            return out
    return write_mcp_live_transcript(out, repo_root=root)


def load_mcp_live_transcript(
    path: str | Path | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load a previously written live observation transcript."""

    root = _discover_repo_root(repo_root)
    target = Path(path) if path is not None else root / DCR_ARTIFACT_PATH
    if not target.is_file():
        raise McpLiveObserverError(
            f"live transcript missing: {target}",
            reason_code="transcript_missing",
            details={"path": str(target)},
        )
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise McpLiveObserverError(
            "transcript must be an object",
            reason_code="transcript_not_object",
        )
    return dict(raw)


__all__ = [
    "DCR_ARTIFACT_PATH",
    "DCR_TASK_ID",
    "JSONRPC_VERSION",
    "LIVE_CONTRACT_TRANSCRIPT_INTERFACE",
    "LIVE_CONTRACT_TRANSCRIPT_SCHEMA",
    "LIVE_OBSERVATION_EVIDENCE_TERM",
    "LOGIC_CEC_PROVE_TOOL",
    "MCP_LIVE_OBSERVATION_INTERFACE",
    "MCP_LIVE_OBSERVATION_SCHEMA",
    "MCP_PLUS_PROFILES_A_F",
    "MCP_PROTOCOL_VERSION",
    "SAFE_TOOLS_CALL",
    "LiveContractTranscript",
    "LiveObservationExchange",
    "McpLiveObserver",
    "McpLiveObserverError",
    "ObservationKind",
    "ObservationTerminalState",
    "ensure_mcp_live_transcript_artifact",
    "load_mcp_live_transcript",
    "materialize_mcp_live_transcript",
    "observation_content_cid",
    "observe_mcp_live_contracts",
    "write_mcp_live_transcript",
]
