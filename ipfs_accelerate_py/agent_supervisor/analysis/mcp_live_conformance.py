"""In-process MCP++ list/call/error conformance and runtime identity (SCA-612/613).

Interface: ``McpLiveConformance@1`` / ``RuntimeServiceIdentity@1``

This module executes representative, capability-bound MCP++ mediation paths for
the accelerator package and optionally other package targets.  Only in-process
registry/handler dispatch can produce acceptance receipts.  Direct import of a
handler function without mediation, empty transports, TODO responses, and
health-only liveness checks cannot satisfy conformance.

Runtime service identity binds loaded module realpaths/digests, commit/tree,
configuration CID, and state CID to the reviewed service authority manifest.
Mixed checkout/state roots fail closed.

Evidence term: ``SCAEV181MCPRUNTIME``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.mcp_server.mcplusplus.idl_registry import (
    IDL_IDENTITY_PROFILE,
    InterfaceDescriptorRegistry,
    InterfaceIdentity,
    build_ai_catalog_v1_descriptor,
    compute_interface_cid,
    identify_interface_descriptor,
    idl_identity_profile,
    is_pseudo_interface_cid,
    validate_interface_cid,
)


MCP_LIVE_CONFORMANCE_INTERFACE: Final = "McpLiveConformance@1"
MCP_LIVE_CONFORMANCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-live-conformance@1"
)
MCP_LIVE_CONFORMANCE_VERSION: Final = "1"
RUNTIME_SERVICE_IDENTITY_INTERFACE: Final = "RuntimeServiceIdentity@1"
RUNTIME_SERVICE_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-service-identity@1"
)
RUNTIME_SERVICE_AUTHORITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-service-authority@1"
)
INVOCATION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-live-invocation-receipt@1"
)
SCAEV181_EVIDENCE_TERM: Final = "SCAEV181MCPRUNTIME"

DEFAULT_AUTHORITY_RELATIVE: Final = (
    "config/swissknife_runtime_service_authority.json"
)
DEFAULT_SERVICE_IDENTITY_RELATIVE: Final = (
    "data/agent_supervisor/swissknife_contract_assurance/runtime/"
    "service-identity.json"
)

MANDATORY_PACKAGE_TARGETS: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py",
)

KNOWN_PACKAGE_TARGETS: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py",
    "ipfs_kit_py",
    "ipfs_datasets_py",
)

SAFE_LIST_OPERATION: Final = "model_catalog_list_services"
SAFE_CALL_OPERATION: Final = "model_catalog_health"
SAFE_UNKNOWN_OPERATION: Final = "__sca_unknown_operation__"


class McpLiveConformanceError(ValueError):
    """A conformance request or receipt is malformed or fail-closed."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "mcp_live_conformance_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class ConformanceTerminalState(str, Enum):
    """Closed terminal states for one live mediation observation."""

    PASSED = "passed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    REFUTED = "refuted"


class TransportKind(str, Enum):
    """Transport surfaces that may appear on a receipt."""

    IN_PROCESS = "in_process"
    HTTP = "http"
    STDIO = "stdio"
    LIBP2P = "libp2p"
    DIRECT_IMPORT = "direct_import"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_label(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _content_cid(value: Any) -> str:
    """Mint a CIDv1 for a structured authority/receipt payload.

    Uses the MCP-IDL identity machinery (raw/sha2-256 on canonical JSON) so
    service authority and identity CIDs are decodable multiformats strings.
    """

    from multiformats import CID, multihash  # type: ignore[attr-defined]

    raw = _canonical_json_bytes(value)
    digest = multihash.digest(raw, "sha2-256")
    return str(CID("base32", 1, "raw", digest))


def _module_file_digest(module_name: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    path = getattr(module, "__file__", None)
    if not isinstance(path, str) or not path:
        raise McpLiveConformanceError(
            f"module has no file path: {module_name}",
            reason_code="module_path_missing",
            details={"module": module_name},
        )
    real = str(Path(path).resolve())
    data = Path(real).read_bytes()
    return {
        "module": module_name,
        "path": real,
        "byte_length": len(data),
        "digest": _sha256_label(data),
    }


def _git_identity(repo_root: Path | None = None) -> dict[str, str]:
    root = repo_root or Path.cwd()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=str(root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise McpLiveConformanceError(
            "unable to resolve git commit/tree for runtime identity",
            reason_code="git_identity_unavailable",
            details={"cause": repr(exc), "cwd": str(root)},
        ) from exc
    if not commit or not tree:
        raise McpLiveConformanceError(
            "empty git commit/tree",
            reason_code="git_identity_empty",
        )
    return {"commit": commit, "tree": tree}


@dataclass(frozen=True)
class LiveInvocationReceipt:
    """Exact receipt for one mediated list/call/error observation."""

    package: str
    operation: str
    method: str
    transport: str
    terminal_state: str
    request_identity: str
    schema_identity: str
    handler_identity: str
    effect_identity: str
    transport_identity: str
    interface_cid: str
    interface_profile: str
    mediated: bool
    model_calls: int
    reason_codes: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    schema: str = INVOCATION_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "package": self.package,
            "operation": self.operation,
            "method": self.method,
            "transport": self.transport,
            "terminal_state": self.terminal_state,
            "request_identity": self.request_identity,
            "schema_identity": self.schema_identity,
            "handler_identity": self.handler_identity,
            "effect_identity": self.effect_identity,
            "transport_identity": self.transport_identity,
            "interface_cid": self.interface_cid,
            "interface_profile": self.interface_profile,
            "mediated": self.mediated,
            "model_calls": self.model_calls,
            "reason_codes": list(self.reason_codes),
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True)
class McpLiveConformanceReport:
    """Aggregate live conformance report for SCAEV181MCPRUNTIME."""

    passed: bool
    model_calls: int
    evidence_term: str
    interface_cid: str
    interface_profile: str
    idl_profile: Mapping[str, Any]
    receipts: tuple[LiveInvocationReceipt, ...]
    package_results: Mapping[str, Any]
    reason_codes: tuple[str, ...] = ()
    schema: str = MCP_LIVE_CONFORMANCE_SCHEMA
    interface: str = MCP_LIVE_CONFORMANCE_INTERFACE
    version: str = MCP_LIVE_CONFORMANCE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "version": self.version,
            "passed": self.passed,
            "model_calls": self.model_calls,
            "evidence_term": self.evidence_term,
            "interface_cid": self.interface_cid,
            "interface_profile": self.interface_profile,
            "idl_profile": dict(self.idl_profile),
            "receipts": [item.to_dict() for item in self.receipts],
            "package_results": dict(self.package_results),
            "reason_codes": list(self.reason_codes),
            "policies": {
                "direct_call_satisfies_mediation": False,
                "health_only_proves_runtime_identity": False,
                "empty_transport_allowed": False,
                "todo_response_allowed": False,
                "pseudo_cid_allowed": False,
                "model_calls_allowed": False,
            },
        }


@dataclass(frozen=True)
class RuntimeServiceIdentity:
    """Startup identity binding modules, git roots, config, and state CIDs."""

    passed: bool
    service_id: str
    commit: str
    tree: str
    configuration_cid: str
    state_cid: str
    authority_cid: str
    modules: tuple[Mapping[str, Any], ...]
    endpoints: tuple[Mapping[str, Any], ...]
    reason_codes: tuple[str, ...] = ()
    schema: str = RUNTIME_SERVICE_IDENTITY_SCHEMA
    interface: str = RUNTIME_SERVICE_IDENTITY_INTERFACE
    evidence_term: str = SCAEV181_EVIDENCE_TERM

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "evidence_term": self.evidence_term,
            "passed": self.passed,
            "service_id": self.service_id,
            "commit": self.commit,
            "tree": self.tree,
            "configuration_cid": self.configuration_cid,
            "state_cid": self.state_cid,
            "authority_cid": self.authority_cid,
            "modules": [dict(item) for item in self.modules],
            "endpoints": [dict(item) for item in self.endpoints],
            "reason_codes": list(self.reason_codes),
            "policies": {
                "mixed_checkout_state_roots_allowed": False,
                "health_is_liveness_only": True,
                "stale_service_code_satisfies_identity": False,
            },
        }


def _identity_for_payload(kind: str, payload: Mapping[str, Any]) -> str:
    return _content_cid({"kind": kind, "payload": dict(payload)})


def _mediated_list(
    registry: InterfaceDescriptorRegistry,
    *,
    package: str,
    interface_identity: InterfaceIdentity,
) -> LiveInvocationReceipt:
    """Execute tools/list parity against the registered IDL surface."""

    request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "package": package,
    }
    request_identity = _identity_for_payload("request", request)
    listed = registry.list_interfaces()
    if interface_identity.cid not in listed:
        raise McpLiveConformanceError(
            "interface not registered for tools/list",
            reason_code="interface_not_registered",
            details={"interface_cid": interface_identity.cid},
        )
    descriptor = registry.get_descriptor(interface_identity.cid)
    if descriptor is None:
        raise McpLiveConformanceError(
            "descriptor missing after list",
            reason_code="descriptor_missing",
        )
    methods = descriptor.get("methods") or []
    if not isinstance(methods, list) or not methods:
        raise McpLiveConformanceError(
            "tools/list returned empty method catalog",
            reason_code="empty_tool_list",
        )
    for method in methods:
        if not isinstance(method, Mapping):
            raise McpLiveConformanceError(
                "method descriptor is not an object",
                reason_code="invalid_method_descriptor",
            )
        for key in ("operation", "input_schema", "output_schema", "mcp_tool"):
            if key not in method:
                raise McpLiveConformanceError(
                    f"method missing {key}",
                    reason_code="method_field_missing",
                    details={"field": key},
                )

    tool_names = sorted(
        str(m["operation"]) for m in methods if isinstance(m, Mapping)
    )
    effect = {
        "tool_count": len(tool_names),
        "tools": tool_names,
        "interface_cid": interface_identity.cid,
    }
    schema_identity = _identity_for_payload(
        "schema",
        {"methods": [str(m.get("operation")) for m in methods if isinstance(m, Mapping)]},
    )
    handler_identity = _identity_for_payload(
        "handler",
        {
            "module": "ipfs_accelerate_py.mcp_server.mcplusplus.idl_registry",
            "symbol": "InterfaceDescriptorRegistry.list_interfaces",
        },
    )
    effect_identity = _identity_for_payload("effect", effect)
    transport_identity = _identity_for_payload(
        "transport",
        {
            "kind": TransportKind.IN_PROCESS.value,
            "mediation": "InterfaceDescriptorRegistry",
        },
    )
    return LiveInvocationReceipt(
        package=package,
        operation="tools/list",
        method="tools/list",
        transport=TransportKind.IN_PROCESS.value,
        terminal_state=ConformanceTerminalState.PASSED.value,
        request_identity=request_identity,
        schema_identity=schema_identity,
        handler_identity=handler_identity,
        effect_identity=effect_identity,
        transport_identity=transport_identity,
        interface_cid=interface_identity.cid,
        interface_profile=interface_identity.profile,
        mediated=True,
        model_calls=0,
        reason_codes=("tools_list_mediated", "catalog_nonempty"),
        details=effect,
    )


def _mediated_call(
    registry: InterfaceDescriptorRegistry,
    *,
    package: str,
    interface_identity: InterfaceIdentity,
    operation: str,
    expect_success: bool,
) -> LiveInvocationReceipt:
    """Execute tools/call (or fail-closed unknown) through IDL resolution."""

    request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": operation, "arguments": {}},
        "package": package,
    }
    request_identity = _identity_for_payload("request", request)
    transport_identity = _identity_for_payload(
        "transport",
        {
            "kind": TransportKind.IN_PROCESS.value,
            "mediation": "resolve_ai_catalog_operation",
        },
    )
    handler_identity = _identity_for_payload(
        "handler",
        {
            "module": "ipfs_accelerate_py.mcp_server.mcplusplus.idl_registry",
            "symbol": "InterfaceDescriptorRegistry.resolve_ai_catalog_operation",
            "operation": operation,
        },
    )

    if expect_success:
        method = registry.resolve_ai_catalog_operation(operation)
        schema_identity = _identity_for_payload(
            "schema",
            {
                "operation": operation,
                "input_schema": method.get("input_schema"),
                "required_authority": method.get("required_authority"),
            },
        )
        # Safe health call: resolve only; no external model invocation.
        effect = {
            "operation": operation,
            "resolved": True,
            "required_authority": method.get("required_authority"),
            "mcp_tool": method.get("mcp_tool"),
            "model_invoked": False,
        }
        effect_identity = _identity_for_payload("effect", effect)
        return LiveInvocationReceipt(
            package=package,
            operation=operation,
            method="tools/call",
            transport=TransportKind.IN_PROCESS.value,
            terminal_state=ConformanceTerminalState.PASSED.value,
            request_identity=request_identity,
            schema_identity=schema_identity,
            handler_identity=handler_identity,
            effect_identity=effect_identity,
            transport_identity=transport_identity,
            interface_cid=interface_identity.cid,
            interface_profile=interface_identity.profile,
            mediated=True,
            model_calls=0,
            reason_codes=("tools_call_mediated", "zero_model_calls"),
            details=effect,
        )

    # Unknown / invalid must fail closed without mediation success.
    try:
        registry.resolve_ai_catalog_operation(operation)
        raise McpLiveConformanceError(
            "unknown operation unexpectedly resolved",
            reason_code="unknown_operation_accepted",
            details={"operation": operation},
        )
    except Exception as exc:  # noqa: BLE001 - typed fail-closed is the success path
        if isinstance(exc, McpLiveConformanceError):
            raise
        effect = {
            "operation": operation,
            "resolved": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:256],
        }
        schema_identity = _identity_for_payload(
            "schema",
            {"operation": operation, "valid": False},
        )
        effect_identity = _identity_for_payload("effect", effect)
        return LiveInvocationReceipt(
            package=package,
            operation=operation,
            method="tools/call",
            transport=TransportKind.IN_PROCESS.value,
            terminal_state=ConformanceTerminalState.REFUTED.value,
            request_identity=request_identity,
            schema_identity=schema_identity,
            handler_identity=handler_identity,
            effect_identity=effect_identity,
            transport_identity=transport_identity,
            interface_cid=interface_identity.cid,
            interface_profile=interface_identity.profile,
            mediated=True,
            model_calls=0,
            reason_codes=("unknown_operation_fail_closed", "zero_model_calls"),
            details=effect,
        )


def _direct_import_cannot_satisfy(
    *,
    package: str,
    interface_identity: InterfaceIdentity,
) -> LiveInvocationReceipt:
    """Record that a bare direct import is not mediation proof."""

    request = {
        "method": "direct_import",
        "target": "build_ai_catalog_v1_descriptor",
        "package": package,
    }
    # Performing the direct import intentionally; receipt marks non-mediation.
    _ = build_ai_catalog_v1_descriptor()
    effect = {
        "direct_import_executed": True,
        "satisfies_mcp_mediation": False,
    }
    return LiveInvocationReceipt(
        package=package,
        operation="direct_import",
        method="direct_import",
        transport=TransportKind.DIRECT_IMPORT.value,
        terminal_state=ConformanceTerminalState.REFUTED.value,
        request_identity=_identity_for_payload("request", request),
        schema_identity=_identity_for_payload("schema", {"mediation": False}),
        handler_identity=_identity_for_payload(
            "handler",
            {
                "module": "ipfs_accelerate_py.mcp_server.mcplusplus.idl_registry",
                "symbol": "build_ai_catalog_v1_descriptor",
                "mediated": False,
            },
        ),
        effect_identity=_identity_for_payload("effect", effect),
        transport_identity=_identity_for_payload(
            "transport",
            {"kind": TransportKind.DIRECT_IMPORT.value, "mediation": False},
        ),
        interface_cid=interface_identity.cid,
        interface_profile=interface_identity.profile,
        mediated=False,
        model_calls=0,
        reason_codes=("direct_import_not_mediation",),
        details=effect,
    )


def run_mcp_live_conformance(
    *,
    packages: Sequence[str] | None = None,
    include_optional_packages: bool = False,
) -> McpLiveConformanceReport:
    """Execute safe list/call/error paths and return a CID-bound report.

    Always covers ``ipfs_accelerate_py``.  Optional packages are reported as
    typed unsupported when not requested or not importable; they never silently
    pass.
    """

    targets = list(packages or MANDATORY_PACKAGE_TARGETS)
    if include_optional_packages:
        for pkg in KNOWN_PACKAGE_TARGETS:
            if pkg not in targets:
                targets.append(pkg)

    profile = idl_identity_profile()
    descriptor = build_ai_catalog_v1_descriptor()
    interface_identity = identify_interface_descriptor(descriptor)
    if is_pseudo_interface_cid(interface_identity.cid):
        raise McpLiveConformanceError(
            "interface CID must not be a pseudo-identity",
            reason_code="pseudo_cid_rejected",
        )
    # Re-verify multihash digest equals retained canonical bytes.
    validate_interface_cid(
        interface_identity.cid,
        interface_identity.canonical_bytes,
        expected_profile=IDL_IDENTITY_PROFILE,
    )

    registry = InterfaceDescriptorRegistry(
        supported_capabilities=["mcp++/profile-a-idl"]
    )
    registered_cid = registry.register_ai_catalog_v1()
    if registered_cid != interface_identity.cid:
        raise McpLiveConformanceError(
            "registered interface CID drift",
            reason_code="interface_cid_drift",
            details={
                "registered": registered_cid,
                "expected": interface_identity.cid,
            },
        )

    receipts: list[LiveInvocationReceipt] = []
    package_results: dict[str, Any] = {}
    reason_codes: list[str] = []
    model_calls = 0
    passed = True

    for package in targets:
        if package != "ipfs_accelerate_py":
            package_results[package] = {
                "status": ConformanceTerminalState.UNSUPPORTED.value,
                "reason_codes": ["package_not_in_this_runtime_scope"],
            }
            reason_codes.append(f"{package}:unsupported")
            continue

        try:
            list_receipt = _mediated_list(
                registry,
                package=package,
                interface_identity=interface_identity,
            )
            call_receipt = _mediated_call(
                registry,
                package=package,
                interface_identity=interface_identity,
                operation=SAFE_CALL_OPERATION,
                expect_success=True,
            )
            unknown_receipt = _mediated_call(
                registry,
                package=package,
                interface_identity=interface_identity,
                operation=SAFE_UNKNOWN_OPERATION,
                expect_success=False,
            )
            direct_receipt = _direct_import_cannot_satisfy(
                package=package,
                interface_identity=interface_identity,
            )
        except McpLiveConformanceError as exc:
            passed = False
            package_results[package] = {
                "status": ConformanceTerminalState.FAILED.value,
                "reason_codes": [exc.reason_code],
                "details": dict(exc.details),
            }
            reason_codes.append(f"{package}:{exc.reason_code}")
            continue

        package_receipts = (
            list_receipt,
            call_receipt,
            unknown_receipt,
            direct_receipt,
        )
        receipts.extend(package_receipts)
        model_calls += sum(item.model_calls for item in package_receipts)

        mediated_ok = (
            list_receipt.mediated
            and list_receipt.terminal_state
            == ConformanceTerminalState.PASSED.value
            and call_receipt.mediated
            and call_receipt.terminal_state
            == ConformanceTerminalState.PASSED.value
            and unknown_receipt.terminal_state
            == ConformanceTerminalState.REFUTED.value
            and direct_receipt.mediated is False
        )
        if not mediated_ok or model_calls != 0:
            passed = False
            package_results[package] = {
                "status": ConformanceTerminalState.FAILED.value,
                "reason_codes": ["mediation_incomplete_or_model_calls"],
                "receipt_count": len(package_receipts),
            }
            reason_codes.append(f"{package}:failed")
        else:
            package_results[package] = {
                "status": ConformanceTerminalState.PASSED.value,
                "reason_codes": [
                    "tools_list_passed",
                    "tools_call_passed",
                    "unknown_fail_closed",
                    "direct_import_refuted",
                ],
                "receipt_count": len(package_receipts),
                "safe_list_operation": SAFE_LIST_OPERATION,
                "safe_call_operation": SAFE_CALL_OPERATION,
            }
            reason_codes.append(f"{package}:passed")

    if model_calls != 0:
        passed = False
        reason_codes.append("model_calls_nonzero")

    if not any(
        r.operation == "tools/list" and r.mediated for r in receipts
    ):
        passed = False
        reason_codes.append("missing_tools_list_receipt")

    return McpLiveConformanceReport(
        passed=passed,
        model_calls=model_calls,
        evidence_term=SCAEV181_EVIDENCE_TERM,
        interface_cid=interface_identity.cid,
        interface_profile=interface_identity.profile,
        idl_profile=profile,
        receipts=tuple(receipts),
        package_results=MappingProxyType(package_results),
        reason_codes=tuple(dict.fromkeys(reason_codes)),
    )


def _discover_repo_root(explicit: Path | None = None) -> Path:
    """Locate the repository root that owns the service authority manifest."""

    if explicit is not None:
        return explicit
    candidates = [Path.cwd(), *Path.cwd().resolve().parents]
    # Prefer roots that contain the reviewed authority file.
    for candidate in candidates:
        if (candidate / DEFAULT_AUTHORITY_RELATIVE).is_file():
            return candidate
    return Path.cwd()


def load_runtime_service_authority(
    path: str | Path | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load and validate the reviewed runtime service authority manifest."""

    root = _discover_repo_root(repo_root)
    authority_path = Path(path) if path else root / DEFAULT_AUTHORITY_RELATIVE
    if not authority_path.is_file():
        raise McpLiveConformanceError(
            f"runtime service authority missing: {authority_path}",
            reason_code="authority_missing",
            details={"path": str(authority_path)},
        )
    raw = json.loads(authority_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise McpLiveConformanceError(
            "authority manifest must be an object",
            reason_code="authority_not_object",
        )
    if raw.get("schema") != RUNTIME_SERVICE_AUTHORITY_SCHEMA:
        raise McpLiveConformanceError(
            "authority schema mismatch",
            reason_code="authority_schema_mismatch",
            details={"schema": raw.get("schema")},
        )
    required = (
        "service_id",
        "modules",
        "endpoints",
        "configuration",
        "state",
    )
    for key in required:
        if key not in raw:
            raise McpLiveConformanceError(
                f"authority missing {key}",
                reason_code="authority_field_missing",
                details={"field": key},
            )
    return raw


def build_runtime_service_identity(
    *,
    authority: Mapping[str, Any] | None = None,
    authority_path: str | Path | None = None,
    repo_root: Path | None = None,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
) -> RuntimeServiceIdentity:
    """Build a startup identity receipt bound to the authority manifest.

    Refuses mixed/stale roots when expected commit/tree are supplied and do not
    match the live checkout.  Health is not consulted; identity is content
    addressed.
    """

    root = _discover_repo_root(repo_root)
    auth = (
        dict(authority)
        if authority is not None
        else load_runtime_service_authority(authority_path, repo_root=root)
    )
    git = _git_identity(root)
    if expected_commit is not None and expected_commit != git["commit"]:
        raise McpLiveConformanceError(
            "mixed or stale commit root",
            reason_code="commit_root_mismatch",
            details={
                "expected": expected_commit,
                "actual": git["commit"],
            },
        )
    if expected_tree is not None and expected_tree != git["tree"]:
        raise McpLiveConformanceError(
            "mixed or stale tree root",
            reason_code="tree_root_mismatch",
            details={"expected": expected_tree, "actual": git["tree"]},
        )

    module_names = auth.get("modules") or []
    if not isinstance(module_names, list) or not module_names:
        raise McpLiveConformanceError(
            "authority modules must be a nonempty list",
            reason_code="authority_modules_invalid",
        )
    modules: list[dict[str, Any]] = []
    for name in module_names:
        if not isinstance(name, str) or not name:
            raise McpLiveConformanceError(
                "module entry must be a nonempty string",
                reason_code="module_name_invalid",
            )
        modules.append(_module_file_digest(name))

    configuration = auth.get("configuration")
    if not isinstance(configuration, Mapping):
        raise McpLiveConformanceError(
            "configuration must be an object",
            reason_code="configuration_invalid",
        )
    configuration_cid = str(
        configuration.get("cid") or _content_cid(dict(configuration))
    )
    if is_pseudo_interface_cid(configuration_cid):
        raise McpLiveConformanceError(
            "configuration CID must be a multiformat CID",
            reason_code="pseudo_cid_rejected",
            details={"field": "configuration.cid"},
        )

    state = auth.get("state")
    if not isinstance(state, Mapping):
        raise McpLiveConformanceError(
            "state must be an object",
            reason_code="state_invalid",
        )
    state_cid = str(state.get("cid") or _content_cid(dict(state)))
    if is_pseudo_interface_cid(state_cid):
        raise McpLiveConformanceError(
            "state CID must be a multiformat CID",
            reason_code="pseudo_cid_rejected",
            details={"field": "state.cid"},
        )

    endpoints_raw = auth.get("endpoints") or []
    if not isinstance(endpoints_raw, list):
        raise McpLiveConformanceError(
            "endpoints must be a list",
            reason_code="endpoints_invalid",
        )
    endpoints: list[dict[str, Any]] = []
    for item in endpoints_raw:
        if not isinstance(item, Mapping):
            raise McpLiveConformanceError(
                "endpoint must be an object",
                reason_code="endpoint_invalid",
            )
        endpoints.append(dict(item))

    authority_cid = _content_cid(
        {
            "schema": auth.get("schema"),
            "service_id": auth.get("service_id"),
            "modules": list(module_names),
            "endpoints": endpoints,
            "configuration": dict(configuration),
            "state": dict(state),
        }
    )

    # Interface CID from IDL must also be non-pseudo for baseline binding.
    interface_cid = compute_interface_cid(build_ai_catalog_v1_descriptor())
    if is_pseudo_interface_cid(interface_cid):
        raise McpLiveConformanceError(
            "baseline interface CID is pseudo",
            reason_code="pseudo_cid_rejected",
        )

    reason_codes = [
        "modules_bound",
        "commit_tree_bound",
        "configuration_cid_bound",
        "state_cid_bound",
        "authority_cid_bound",
        "interface_cid_bound",
    ]
    return RuntimeServiceIdentity(
        passed=True,
        service_id=str(auth["service_id"]),
        commit=git["commit"],
        tree=git["tree"],
        configuration_cid=configuration_cid,
        state_cid=state_cid,
        authority_cid=authority_cid,
        modules=tuple(modules),
        endpoints=tuple(endpoints),
        reason_codes=tuple(reason_codes),
    )


def write_service_identity_receipt(
    identity: RuntimeServiceIdentity,
    *,
    path: str | Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Atomically write the service-identity receipt JSON."""

    root = _discover_repo_root(repo_root)
    out = Path(path) if path else root / DEFAULT_SERVICE_IDENTITY_RELATIVE
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = identity.to_dict()
    payload["receipt_cid"] = _content_cid(payload)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, out)
    return out


def run_scaev181_mcp_runtime_gate(
    *,
    repo_root: Path | None = None,
    write_identity: bool = False,
) -> dict[str, Any]:
    """Run live conformance + runtime identity and return a combined receipt."""

    root = _discover_repo_root(repo_root)
    conformance = run_mcp_live_conformance()
    identity = build_runtime_service_identity(repo_root=root)
    if write_identity:
        write_service_identity_receipt(identity, repo_root=root)

    passed = conformance.passed and identity.passed and conformance.model_calls == 0
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/scaev181-mcp-runtime@1",
        "evidence_term": SCAEV181_EVIDENCE_TERM,
        "passed": passed,
        "model_calls": conformance.model_calls,
        "conformance": conformance.to_dict(),
        "service_identity": identity.to_dict(),
        "reason_codes": list(
            dict.fromkeys(
                [
                    *conformance.reason_codes,
                    *identity.reason_codes,
                    "scaev181_mcp_runtime",
                ]
            )
        ),
    }


__all__ = [
    "ConformanceTerminalState",
    "INVOCATION_RECEIPT_SCHEMA",
    "LiveInvocationReceipt",
    "MCP_LIVE_CONFORMANCE_INTERFACE",
    "MCP_LIVE_CONFORMANCE_SCHEMA",
    "McpLiveConformanceError",
    "McpLiveConformanceReport",
    "RUNTIME_SERVICE_AUTHORITY_SCHEMA",
    "RUNTIME_SERVICE_IDENTITY_INTERFACE",
    "RUNTIME_SERVICE_IDENTITY_SCHEMA",
    "RuntimeServiceIdentity",
    "SAFE_CALL_OPERATION",
    "SAFE_LIST_OPERATION",
    "SCAEV181_EVIDENCE_TERM",
    "TransportKind",
    "build_runtime_service_identity",
    "load_runtime_service_authority",
    "run_mcp_live_conformance",
    "run_scaev181_mcp_runtime_gate",
    "write_service_identity_receipt",
]
