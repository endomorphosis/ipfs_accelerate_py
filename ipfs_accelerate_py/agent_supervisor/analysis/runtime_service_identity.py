"""Bind launched MCP services to exact runtime identities (DCR-022).

Interfaces
----------
* ``RuntimeServiceManifest@1`` — reviewed one-endpoint-per-role service authority.
* ``RuntimeServiceWitness@1`` — process/config/checkout/endpoint binding receipt.

Normative rules (fail-closed):

* Exactly one reviewed endpoint per service role (accelerate, datasets, kit).
* Port disagreements across roles or against the reviewed endpoint are rejected.
* Endpoint availability without process + configuration/state identity is
  insufficient to authorize observations.
* Process replacement, configuration/state CID change, wrong checkout, or an
  unbound endpoint invalidates later observations.
* Digests are never promoted to multiformat CIDs.

Evidence term: ``dcr/runtime-witness@1``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ContractValidationError,
    content_identity,
)

RUNTIME_SERVICE_MANIFEST_INTERFACE: Final = "RuntimeServiceManifest@1"
RUNTIME_SERVICE_WITNESS_INTERFACE: Final = "RuntimeServiceWitness@1"
RUNTIME_SERVICE_MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-contract-repair-services@1"
)
RUNTIME_SERVICE_WITNESS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-contract-repair-runtime-witness@1"
)
RUNTIME_WITNESS_EVIDENCE_TERM: Final = "dcr/runtime-witness@1"
DEFAULT_SERVICES_RELATIVE: Final = "config/deterministic_contract_repair_services.json"
DEFAULT_WITNESS_RELATIVE: Final = (
    "data/agent_supervisor/deterministic_contract_repair/runtime-witness.json"
)

REQUIRED_SERVICE_ROLES: Final[tuple[str, ...]] = ("accelerate", "datasets", "kit")
_DEFAULT_ENV_ALLOWLIST: Final[tuple[str, ...]] = (
    "HOME",
    "PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR",
    "IPFS_ACCELERATE_AGENT_VALIDATION_PATH",
)


class RuntimeServiceIdentityError(ValueError):
    """A runtime service identity request fails closed."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "runtime_service_identity_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class ObservationDisposition(str, Enum):
    """Closed outcomes for validating an observation against a witness."""

    VALID = "valid"
    INVALIDATED = "invalidated"
    UNBOUND = "unbound"


class InvalidationReason(str, Enum):
    """Typed reasons that invalidate runtime observations."""

    PORT_DISAGREEMENT = "port_disagreement"
    PROCESS_REPLACEMENT = "process_replacement"
    CONFIG_STATE_CHANGED = "config_state_changed"
    WRONG_CHECKOUT = "wrong_checkout"
    UNBOUND_ENDPOINT = "unbound_endpoint"
    ENDPOINT_WITHOUT_PROCESS_IDENTITY = "endpoint_without_process_identity"
    ROLE_MISMATCH = "role_mismatch"
    ARGUMENT_MISMATCH = "argument_mismatch"
    ENVIRONMENT_MISMATCH = "environment_mismatch"
    INTERPRETER_MISMATCH = "interpreter_mismatch"
    MODULE_MISMATCH = "module_mismatch"
    TRANSPORT_MISMATCH = "transport_mismatch"


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


def content_cid(value: Any) -> str:
    """Return a CIDv1 dag-json/sha2-256 identity for a structured payload."""

    try:
        return content_identity(value)
    except ContractValidationError as exc:
        raise RuntimeServiceIdentityError(
            "value is not canonical-JSON encodable",
            reason_code="non_canonical_json",
            details={"cause": str(exc)},
        ) from exc


def is_pseudo_cid(value: Any) -> bool:
    """Return True when *value* is not a lowercase base32 CIDv1 dag-json string."""

    if not isinstance(value, str) or not value:
        return True
    text = value.strip()
    if text.startswith("sha256:") or text.startswith("repository:sha256:"):
        return True
    if not text.startswith("b") or len(text) < 50:
        return True
    try:
        import base64

        padding = "=" * ((8 - len(text[1:]) % 8) % 8)
        decoded = base64.b32decode((text[1:].upper() + padding).encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        return True
    expected_prefix = b"\x01\xa9\x02\x12\x20"
    if len(decoded) != len(expected_prefix) + 32:
        return True
    if not decoded.startswith(expected_prefix):
        return True
    canonical = "b" + base64.b32encode(decoded).decode("ascii").rstrip("=").lower()
    return canonical != text


def require_multiformat_cid(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or is_pseudo_cid(text):
        raise RuntimeServiceIdentityError(
            f"{field_name} must be a multiformat CIDv1, not a digest or pseudo-CID",
            reason_code="pseudo_cid_rejected",
            details={"field": field_name, "value_prefix": text[:24]},
        )
    return text


def _require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeServiceIdentityError(
            f"{field_name} must be an object",
            reason_code="invalid_field_type",
            details={"field": field_name},
        )
    return value


def _require_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeServiceIdentityError(
            f"{field_name} must be a nonempty string",
            reason_code="invalid_field_type",
            details={"field": field_name},
        )
    return value.strip()


def _require_int(value: Any, *, field_name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeServiceIdentityError(
            f"{field_name} must be an integer",
            reason_code="invalid_field_type",
            details={"field": field_name},
        )
    if minimum is not None and value < minimum:
        raise RuntimeServiceIdentityError(
            f"{field_name} must be >= {minimum}",
            reason_code="invalid_field_value",
            details={"field": field_name, "value": value},
        )
    return value


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RuntimeServiceIdentityError(
            f"{field_name} must be an array of strings",
            reason_code="invalid_field_type",
            details={"field": field_name},
        )
    items: list[str] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, str) or not raw:
            raise RuntimeServiceIdentityError(
                f"{field_name}[{index}] must be a nonempty string",
                reason_code="invalid_field_type",
                details={"field": field_name, "index": index},
            )
        items.append(raw)
    return tuple(items)


def _discover_repo_root(repo_root: Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    here = Path(__file__).resolve()
    candidates = [here.parents[4], here.parents[5], Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / DEFAULT_SERVICES_RELATIVE).is_file():
            return candidate.resolve()
    return Path.cwd().resolve()


def _git_identity(repo_root: Path) -> dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeServiceIdentityError(
            "unable to resolve git commit/tree for runtime identity",
            reason_code="git_identity_unavailable",
            details={"cause": repr(exc), "cwd": str(repo_root)},
        ) from exc
    if not commit or not tree:
        raise RuntimeServiceIdentityError(
            "empty git commit/tree",
            reason_code="git_identity_empty",
        )
    return {"commit": commit, "tree": tree}


def module_file_digest(module_name: str) -> dict[str, Any]:
    """Resolve a loaded module to realpath + SHA-256 digest."""

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - import failures are fail-closed
        raise RuntimeServiceIdentityError(
            f"unable to import module for identity binding: {module_name}",
            reason_code="module_import_failed",
            details={"module": module_name, "cause": repr(exc)},
        ) from exc
    path = getattr(module, "__file__", None)
    if not isinstance(path, str) or not path:
        raise RuntimeServiceIdentityError(
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


def filter_environment(
    environment: Mapping[str, Any] | None,
    allowlist: Sequence[str],
) -> dict[str, str]:
    """Project an environment map onto the reviewed allowlist only."""

    allowed = {str(name) for name in allowlist}
    source = dict(environment or {})
    filtered: dict[str, str] = {}
    for name in sorted(allowed):
        if name in source and source[name] is not None:
            filtered[name] = str(source[name])
    return filtered


@dataclass(frozen=True, slots=True)
class ServiceEndpoint:
    """One reviewed loopback/in-process endpoint identity."""

    kind: str
    host: str
    port: int
    path: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "host": self.host,
            "port": self.port,
            "path": self.path,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServiceEndpoint":
        data = _require_mapping(payload, field_name="endpoint")
        kind = _require_text(data.get("kind"), field_name="endpoint.kind")
        host = _require_text(data.get("host"), field_name="endpoint.host")
        port = _require_int(data.get("port"), field_name="endpoint.port", minimum=1)
        path = _require_text(data.get("path"), field_name="endpoint.path")
        url = data.get("url")
        if not isinstance(url, str) or not url.strip():
            url = f"http://{host}:{port}{path}"
        return cls(kind=kind, host=host, port=port, path=path, url=url.strip())

    def identity_payload(self) -> dict[str, Any]:
        return self.to_dict()

    @property
    def endpoint_id(self) -> str:
        return content_cid(self.identity_payload())


@dataclass(frozen=True, slots=True)
class ServiceRoleDeclaration:
    """Reviewed declaration for one MCP service role."""

    role: str
    package: str
    root: str
    transport: str
    modules: tuple[str, ...]
    arguments: tuple[str, ...]
    endpoint: ServiceEndpoint
    configuration: Mapping[str, Any]
    state: Mapping[str, Any]
    configuration_cid: str
    state_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "package": self.package,
            "root": self.root,
            "transport": self.transport,
            "modules": list(self.modules),
            "arguments": list(self.arguments),
            "endpoint": self.endpoint.to_dict(),
            "configuration": dict(self.configuration),
            "state": dict(self.state),
            "configuration_cid": self.configuration_cid,
            "state_cid": self.state_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServiceRoleDeclaration":
        data = _require_mapping(payload, field_name="service")
        role = _require_text(data.get("role"), field_name="service.role")
        package = _require_text(data.get("package"), field_name="service.package")
        root = _require_text(data.get("root"), field_name="service.root")
        transport = _require_text(data.get("transport"), field_name="service.transport")
        modules = _string_tuple(data.get("modules"), field_name="service.modules")
        if not modules:
            raise RuntimeServiceIdentityError(
                "service.modules must be nonempty",
                reason_code="modules_empty",
                details={"role": role},
            )
        arguments = _string_tuple(data.get("arguments"), field_name="service.arguments")
        endpoint = ServiceEndpoint.from_dict(
            _require_mapping(data.get("endpoint"), field_name="service.endpoint")
        )
        configuration = dict(
            _require_mapping(data.get("configuration"), field_name="service.configuration")
        )
        state = dict(_require_mapping(data.get("state"), field_name="service.state"))
        configuration_cid = content_cid(configuration)
        state_cid = content_cid(state)
        claimed_config = data.get("configuration_cid")
        if claimed_config is not None:
            claimed = require_multiformat_cid(
                claimed_config, field_name="service.configuration_cid"
            )
            if claimed != configuration_cid:
                raise RuntimeServiceIdentityError(
                    "service.configuration_cid does not match local recomputation",
                    reason_code="configuration_cid_mismatch",
                    details={
                        "role": role,
                        "claimed": claimed,
                        "local": configuration_cid,
                    },
                )
        claimed_state = data.get("state_cid")
        if claimed_state is not None:
            claimed = require_multiformat_cid(claimed_state, field_name="service.state_cid")
            if claimed != state_cid:
                raise RuntimeServiceIdentityError(
                    "service.state_cid does not match local recomputation",
                    reason_code="state_cid_mismatch",
                    details={"role": role, "claimed": claimed, "local": state_cid},
                )
        return cls(
            role=role,
            package=package,
            root=root,
            transport=transport,
            modules=modules,
            arguments=arguments,
            endpoint=endpoint,
            configuration=MappingProxyType(configuration),
            state=MappingProxyType(state),
            configuration_cid=configuration_cid,
            state_cid=state_cid,
        )


@dataclass(frozen=True, slots=True)
class RuntimeServiceManifest:
    """Reviewed multi-package MCP runtime service authority.

    Interface: ``RuntimeServiceManifest@1``
    """

    service_id: str
    services: tuple[ServiceRoleDeclaration, ...]
    environment_allowlist: tuple[str, ...]
    policies: Mapping[str, Any] = field(default_factory=dict)
    conflict_policy: str = (
        "One reviewed endpoint per service role; endpoint availability without "
        "process/config identity is insufficient."
    )
    schema: str = RUNTIME_SERVICE_MANIFEST_SCHEMA
    interface: str = RUNTIME_SERVICE_MANIFEST_INTERFACE
    version: str = "1"
    evidence_term: str = RUNTIME_WITNESS_EVIDENCE_TERM

    def __post_init__(self) -> None:
        roles = [item.role for item in self.services]
        if sorted(roles) != sorted(REQUIRED_SERVICE_ROLES):
            raise RuntimeServiceIdentityError(
                "manifest must declare exactly the accelerate, datasets, and kit roles",
                reason_code="required_roles_missing",
                details={"roles": roles, "required": list(REQUIRED_SERVICE_ROLES)},
            )
        if len(set(roles)) != len(roles):
            raise RuntimeServiceIdentityError(
                "duplicate service roles are forbidden",
                reason_code="duplicate_service_role",
                details={"roles": roles},
            )
        ports = [item.endpoint.port for item in self.services]
        if len(set(ports)) != len(ports):
            raise RuntimeServiceIdentityError(
                "port disagreements: each service role must own a unique reviewed port",
                reason_code=InvalidationReason.PORT_DISAGREEMENT.value,
                details={"ports": ports},
            )
        urls = [item.endpoint.url for item in self.services]
        if len(set(urls)) != len(urls):
            raise RuntimeServiceIdentityError(
                "endpoint URL disagreements across service roles",
                reason_code=InvalidationReason.PORT_DISAGREEMENT.value,
                details={"urls": urls},
            )
        if not self.environment_allowlist:
            raise RuntimeServiceIdentityError(
                "environment_allowlist must be nonempty",
                reason_code="environment_allowlist_empty",
            )

    def service_for_role(self, role: str) -> ServiceRoleDeclaration:
        for item in self.services:
            if item.role == role:
                return item
        raise RuntimeServiceIdentityError(
            f"unknown service role: {role}",
            reason_code="unknown_service_role",
            details={"role": role},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "version": self.version,
            "evidence_term": self.evidence_term,
            "service_id": self.service_id,
            "conflict_policy": self.conflict_policy,
            "policies": dict(self.policies),
            "environment_allowlist": list(self.environment_allowlist),
            "services": [item.to_dict() for item in self.services],
            "manifest_cid": self.manifest_cid,
        }

    @property
    def manifest_cid(self) -> str:
        payload = {
            "schema": self.schema,
            "interface": self.interface,
            "version": self.version,
            "service_id": self.service_id,
            "environment_allowlist": list(self.environment_allowlist),
            "services": [item.to_dict() for item in self.services],
        }
        return content_cid(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeServiceManifest":
        data = _require_mapping(payload, field_name="manifest")
        schema = _require_text(data.get("schema"), field_name="schema")
        if schema != RUNTIME_SERVICE_MANIFEST_SCHEMA:
            raise RuntimeServiceIdentityError(
                "manifest schema mismatch",
                reason_code="manifest_schema_mismatch",
                details={"schema": schema},
            )
        interface = data.get("interface") or RUNTIME_SERVICE_MANIFEST_INTERFACE
        if interface != RUNTIME_SERVICE_MANIFEST_INTERFACE:
            raise RuntimeServiceIdentityError(
                "manifest interface mismatch",
                reason_code="manifest_interface_mismatch",
                details={"interface": interface},
            )
        service_id = _require_text(data.get("service_id"), field_name="service_id")
        services_raw = data.get("services")
        if not isinstance(services_raw, Sequence) or isinstance(
            services_raw, (str, bytes, bytearray)
        ):
            raise RuntimeServiceIdentityError(
                "services must be an array",
                reason_code="invalid_field_type",
                details={"field": "services"},
            )
        services = tuple(
            ServiceRoleDeclaration.from_dict(_require_mapping(item, field_name="service"))
            for item in services_raw
        )
        allowlist = _string_tuple(
            data.get("environment_allowlist") or list(_DEFAULT_ENV_ALLOWLIST),
            field_name="environment_allowlist",
        )
        policies = dict(data.get("policies") or {})
        conflict_policy = str(
            data.get("conflict_policy")
            or (
                "One reviewed endpoint per service role; endpoint availability "
                "without process/config identity is insufficient."
            )
        )
        return cls(
            service_id=service_id,
            services=services,
            environment_allowlist=allowlist,
            policies=MappingProxyType(policies),
            conflict_policy=conflict_policy,
            schema=schema,
            interface=str(interface),
            version=str(data.get("version") or "1"),
            evidence_term=str(
                data.get("evidence_term") or RUNTIME_WITNESS_EVIDENCE_TERM
            ),
        )


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """PID-reuse-resistant process birth identity."""

    pid: int
    start_time: str
    interpreter: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "start_time": self.start_time,
            "interpreter": self.interpreter,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProcessIdentity":
        data = _require_mapping(payload, field_name="process")
        return cls(
            pid=_require_int(data.get("pid"), field_name="process.pid", minimum=1),
            start_time=_require_text(
                data.get("start_time"), field_name="process.start_time"
            ),
            interpreter=_require_text(
                data.get("interpreter"), field_name="process.interpreter"
            ),
        )

    @property
    def process_id(self) -> str:
        return content_cid(self.to_dict())


@dataclass(frozen=True, slots=True)
class ServiceRuntimeObservation:
    """Observed runtime facts for one launched MCP service role."""

    role: str
    process: ProcessIdentity | None
    arguments: tuple[str, ...]
    environment: Mapping[str, str]
    endpoint: ServiceEndpoint | None
    endpoint_bound: bool
    commit: str
    tree: str
    modules: tuple[Mapping[str, Any], ...]
    configuration_cid: str
    state_cid: str
    transport: str
    endpoint_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "process": None if self.process is None else self.process.to_dict(),
            "arguments": list(self.arguments),
            "environment": dict(self.environment),
            "endpoint": None if self.endpoint is None else self.endpoint.to_dict(),
            "endpoint_bound": self.endpoint_bound,
            "endpoint_available": self.endpoint_available,
            "commit": self.commit,
            "tree": self.tree,
            "modules": [dict(item) for item in self.modules],
            "configuration_cid": self.configuration_cid,
            "state_cid": self.state_cid,
            "transport": self.transport,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServiceRuntimeObservation":
        data = _require_mapping(payload, field_name="observation")
        process_raw = data.get("process")
        process = (
            None
            if process_raw is None
            else ProcessIdentity.from_dict(
                _require_mapping(process_raw, field_name="observation.process")
            )
        )
        endpoint_raw = data.get("endpoint")
        endpoint = (
            None
            if endpoint_raw is None
            else ServiceEndpoint.from_dict(
                _require_mapping(endpoint_raw, field_name="observation.endpoint")
            )
        )
        modules_raw = data.get("modules") or ()
        if not isinstance(modules_raw, Sequence) or isinstance(
            modules_raw, (str, bytes, bytearray)
        ):
            raise RuntimeServiceIdentityError(
                "observation.modules must be an array",
                reason_code="invalid_field_type",
            )
        modules = tuple(dict(item) for item in modules_raw)
        return cls(
            role=_require_text(data.get("role"), field_name="observation.role"),
            process=process,
            arguments=_string_tuple(
                data.get("arguments") or (), field_name="observation.arguments"
            ),
            environment=MappingProxyType(
                {
                    str(key): str(value)
                    for key, value in dict(data.get("environment") or {}).items()
                }
            ),
            endpoint=endpoint,
            endpoint_bound=bool(data.get("endpoint_bound")),
            endpoint_available=bool(data.get("endpoint_available")),
            commit=_require_text(data.get("commit"), field_name="observation.commit"),
            tree=_require_text(data.get("tree"), field_name="observation.tree"),
            modules=modules,
            configuration_cid=require_multiformat_cid(
                data.get("configuration_cid"),
                field_name="observation.configuration_cid",
            ),
            state_cid=require_multiformat_cid(
                data.get("state_cid"), field_name="observation.state_cid"
            ),
            transport=_require_text(
                data.get("transport"), field_name="observation.transport"
            ),
        )


@dataclass(frozen=True, slots=True)
class ServiceRoleWitness:
    """Exact runtime witness for one service role."""

    role: str
    package: str
    root: str
    transport: str
    process: ProcessIdentity
    arguments: tuple[str, ...]
    environment: Mapping[str, str]
    endpoint: ServiceEndpoint
    endpoint_bound: bool
    commit: str
    tree: str
    modules: tuple[Mapping[str, Any], ...]
    configuration_cid: str
    state_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "package": self.package,
            "root": self.root,
            "transport": self.transport,
            "process": self.process.to_dict(),
            "arguments": list(self.arguments),
            "environment": dict(self.environment),
            "endpoint": self.endpoint.to_dict(),
            "endpoint_bound": self.endpoint_bound,
            "commit": self.commit,
            "tree": self.tree,
            "modules": [dict(item) for item in self.modules],
            "configuration_cid": self.configuration_cid,
            "state_cid": self.state_cid,
            "witness_cid": self.witness_cid,
        }

    @property
    def witness_cid(self) -> str:
        payload = {
            "role": self.role,
            "package": self.package,
            "root": self.root,
            "transport": self.transport,
            "process": self.process.to_dict(),
            "arguments": list(self.arguments),
            "environment": dict(self.environment),
            "endpoint": self.endpoint.to_dict(),
            "endpoint_bound": self.endpoint_bound,
            "commit": self.commit,
            "tree": self.tree,
            "modules": [dict(item) for item in self.modules],
            "configuration_cid": self.configuration_cid,
            "state_cid": self.state_cid,
        }
        return content_cid(payload)


@dataclass(frozen=True, slots=True)
class RuntimeServiceWitness:
    """Aggregate runtime witness across accelerate, datasets, and kit.

    Interface: ``RuntimeServiceWitness@1``
    """

    service_id: str
    manifest_cid: str
    commit: str
    tree: str
    roles: tuple[ServiceRoleWitness, ...]
    reason_codes: tuple[str, ...] = ()
    schema: str = RUNTIME_SERVICE_WITNESS_SCHEMA
    interface: str = RUNTIME_SERVICE_WITNESS_INTERFACE
    evidence_term: str = RUNTIME_WITNESS_EVIDENCE_TERM
    passed: bool = True

    def role_witness(self, role: str) -> ServiceRoleWitness:
        for item in self.roles:
            if item.role == role:
                return item
        raise RuntimeServiceIdentityError(
            f"witness missing role: {role}",
            reason_code="unknown_service_role",
            details={"role": role},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "evidence_term": self.evidence_term,
            "passed": self.passed,
            "service_id": self.service_id,
            "manifest_cid": self.manifest_cid,
            "commit": self.commit,
            "tree": self.tree,
            "roles": [item.to_dict() for item in self.roles],
            "reason_codes": list(self.reason_codes),
            "witness_cid": self.witness_cid,
            "policies": {
                "one_endpoint_per_service_role": True,
                "endpoint_availability_without_process_identity_insufficient": True,
                "port_disagreements_allowed": False,
                "process_replacement_invalidates": True,
                "config_state_change_invalidates": True,
                "wrong_checkout_invalidates": True,
                "unbound_endpoint_invalidates": True,
                "health_is_liveness_only": True,
            },
        }

    @property
    def witness_cid(self) -> str:
        payload = {
            "schema": self.schema,
            "interface": self.interface,
            "service_id": self.service_id,
            "manifest_cid": self.manifest_cid,
            "commit": self.commit,
            "tree": self.tree,
            "roles": [item.to_dict() for item in self.roles],
            "passed": self.passed,
        }
        return content_cid(payload)


@dataclass(frozen=True, slots=True)
class ObservationVerdict:
    """Result of validating one observation against a role witness."""

    disposition: ObservationDisposition
    role: str
    valid: bool
    reason_codes: tuple[str, ...]
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "role": self.role,
            "valid": self.valid,
            "reason_codes": list(self.reason_codes),
            "details": dict(self.details),
        }


def load_runtime_service_manifest(
    path: str | Path | None = None,
    *,
    repo_root: Path | None = None,
) -> RuntimeServiceManifest:
    """Load and validate the reviewed DCR runtime services manifest."""

    root = _discover_repo_root(repo_root)
    manifest_path = Path(path) if path else root / DEFAULT_SERVICES_RELATIVE
    if not manifest_path.is_file():
        raise RuntimeServiceIdentityError(
            f"runtime services manifest missing: {manifest_path}",
            reason_code="manifest_missing",
            details={"path": str(manifest_path)},
        )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise RuntimeServiceIdentityError(
            "manifest must be an object",
            reason_code="manifest_not_object",
        )
    return RuntimeServiceManifest.from_dict(raw)


def _bind_modules(module_names: Sequence[str]) -> tuple[dict[str, Any], ...]:
    return tuple(module_file_digest(name) for name in module_names)


def build_service_observation(
    *,
    role: str,
    process: ProcessIdentity | Mapping[str, Any],
    arguments: Sequence[str],
    environment: Mapping[str, Any] | None,
    endpoint: ServiceEndpoint | Mapping[str, Any],
    commit: str,
    tree: str,
    configuration_cid: str,
    state_cid: str,
    transport: str,
    modules: Sequence[Mapping[str, Any]] | None = None,
    module_names: Sequence[str] | None = None,
    endpoint_bound: bool = True,
    endpoint_available: bool = True,
    environment_allowlist: Sequence[str] = _DEFAULT_ENV_ALLOWLIST,
) -> ServiceRuntimeObservation:
    """Construct a structured observation for later witness binding/validation."""

    process_obj = (
        process
        if isinstance(process, ProcessIdentity)
        else ProcessIdentity.from_dict(process)
    )
    endpoint_obj = (
        endpoint
        if isinstance(endpoint, ServiceEndpoint)
        else ServiceEndpoint.from_dict(endpoint)
    )
    if modules is not None:
        module_bindings = tuple(dict(item) for item in modules)
    elif module_names is not None:
        module_bindings = _bind_modules(module_names)
    else:
        module_bindings = ()
    return ServiceRuntimeObservation(
        role=role,
        process=process_obj,
        arguments=tuple(arguments),
        environment=MappingProxyType(
            filter_environment(environment, environment_allowlist)
        ),
        endpoint=endpoint_obj,
        endpoint_bound=endpoint_bound,
        endpoint_available=endpoint_available,
        commit=commit,
        tree=tree,
        modules=module_bindings,
        configuration_cid=require_multiformat_cid(
            configuration_cid, field_name="configuration_cid"
        ),
        state_cid=require_multiformat_cid(state_cid, field_name="state_cid"),
        transport=transport,
    )


def bind_role_witness(
    declaration: ServiceRoleDeclaration,
    observation: ServiceRuntimeObservation,
    *,
    environment_allowlist: Sequence[str],
) -> ServiceRoleWitness:
    """Bind one observation to a reviewed service declaration as a role witness.

    Endpoint availability alone is insufficient: process identity, reviewed
    endpoint, config/state CIDs, and checkout roots must all be present and
    consistent with the declaration.
    """

    if observation.role != declaration.role:
        raise RuntimeServiceIdentityError(
            "observation role does not match declaration",
            reason_code=InvalidationReason.ROLE_MISMATCH.value,
            details={"expected": declaration.role, "actual": observation.role},
        )
    if observation.process is None:
        if observation.endpoint_available or observation.endpoint is not None:
            raise RuntimeServiceIdentityError(
                "endpoint availability without process identity is insufficient",
                reason_code=InvalidationReason.ENDPOINT_WITHOUT_PROCESS_IDENTITY.value,
                details={"role": declaration.role},
            )
        raise RuntimeServiceIdentityError(
            "process identity is required to bind a runtime witness",
            reason_code=InvalidationReason.UNBOUND_ENDPOINT.value,
            details={"role": declaration.role},
        )
    if observation.endpoint is None or not observation.endpoint_bound:
        raise RuntimeServiceIdentityError(
            "endpoint is unbound for service role",
            reason_code=InvalidationReason.UNBOUND_ENDPOINT.value,
            details={"role": declaration.role},
        )
    if observation.endpoint.port != declaration.endpoint.port:
        raise RuntimeServiceIdentityError(
            "observed port disagrees with reviewed endpoint",
            reason_code=InvalidationReason.PORT_DISAGREEMENT.value,
            details={
                "role": declaration.role,
                "reviewed_port": declaration.endpoint.port,
                "observed_port": observation.endpoint.port,
            },
        )
    if observation.endpoint.to_dict() != declaration.endpoint.to_dict():
        raise RuntimeServiceIdentityError(
            "observed endpoint disagrees with reviewed endpoint",
            reason_code=InvalidationReason.PORT_DISAGREEMENT.value,
            details={
                "role": declaration.role,
                "reviewed": declaration.endpoint.to_dict(),
                "observed": observation.endpoint.to_dict(),
            },
        )
    if observation.transport != declaration.transport:
        raise RuntimeServiceIdentityError(
            "observed transport disagrees with reviewed transport",
            reason_code=InvalidationReason.TRANSPORT_MISMATCH.value,
            details={
                "role": declaration.role,
                "reviewed": declaration.transport,
                "observed": observation.transport,
            },
        )
    if tuple(observation.arguments) != tuple(declaration.arguments):
        raise RuntimeServiceIdentityError(
            "observed arguments disagree with reviewed arguments",
            reason_code=InvalidationReason.ARGUMENT_MISMATCH.value,
            details={
                "role": declaration.role,
                "reviewed": list(declaration.arguments),
                "observed": list(observation.arguments),
            },
        )
    if observation.configuration_cid != declaration.configuration_cid:
        raise RuntimeServiceIdentityError(
            "observed configuration CID disagrees with reviewed configuration",
            reason_code=InvalidationReason.CONFIG_STATE_CHANGED.value,
            details={
                "role": declaration.role,
                "reviewed": declaration.configuration_cid,
                "observed": observation.configuration_cid,
            },
        )
    if observation.state_cid != declaration.state_cid:
        raise RuntimeServiceIdentityError(
            "observed state CID disagrees with reviewed state",
            reason_code=InvalidationReason.CONFIG_STATE_CHANGED.value,
            details={
                "role": declaration.role,
                "reviewed": declaration.state_cid,
                "observed": observation.state_cid,
            },
        )
    filtered_env = filter_environment(observation.environment, environment_allowlist)
    modules = observation.modules
    if not modules:
        modules = _bind_modules(declaration.modules)
    expected_names = set(declaration.modules)
    observed_names = {str(item.get("module")) for item in modules}
    if not expected_names.issubset(observed_names):
        raise RuntimeServiceIdentityError(
            "observed modules do not cover reviewed module set",
            reason_code=InvalidationReason.MODULE_MISMATCH.value,
            details={
                "role": declaration.role,
                "expected": sorted(expected_names),
                "observed": sorted(observed_names),
            },
        )
    for item in modules:
        digest = item.get("digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise RuntimeServiceIdentityError(
                "module digest missing or not sha256-labeled",
                reason_code=InvalidationReason.MODULE_MISMATCH.value,
                details={"module": item.get("module")},
            )
        path = item.get("path")
        if not isinstance(path, str) or not path:
            raise RuntimeServiceIdentityError(
                "module path missing",
                reason_code=InvalidationReason.MODULE_MISMATCH.value,
                details={"module": item.get("module")},
            )
    return ServiceRoleWitness(
        role=declaration.role,
        package=declaration.package,
        root=declaration.root,
        transport=declaration.transport,
        process=observation.process,
        arguments=tuple(declaration.arguments),
        environment=MappingProxyType(filtered_env),
        endpoint=declaration.endpoint,
        endpoint_bound=True,
        commit=observation.commit,
        tree=observation.tree,
        modules=tuple(dict(item) for item in modules),
        configuration_cid=declaration.configuration_cid,
        state_cid=declaration.state_cid,
    )


def build_runtime_service_witness(
    *,
    manifest: RuntimeServiceManifest | None = None,
    observations: Sequence[ServiceRuntimeObservation | Mapping[str, Any]],
    repo_root: Path | None = None,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
) -> RuntimeServiceWitness:
    """Build a multi-role runtime witness from launch observations.

    Mixed or stale checkout roots fail closed when expected commit/tree are
    supplied.  Port disagreements and unbound endpoints fail closed.
    """

    root = _discover_repo_root(repo_root)
    auth = manifest if manifest is not None else load_runtime_service_manifest(repo_root=root)
    git = _git_identity(root)
    commit = git["commit"]
    tree = git["tree"]
    if expected_commit is not None and expected_commit != commit:
        raise RuntimeServiceIdentityError(
            "mixed or stale commit root",
            reason_code=InvalidationReason.WRONG_CHECKOUT.value,
            details={"expected": expected_commit, "actual": commit},
        )
    if expected_tree is not None and expected_tree != tree:
        raise RuntimeServiceIdentityError(
            "mixed or stale tree root",
            reason_code=InvalidationReason.WRONG_CHECKOUT.value,
            details={"expected": expected_tree, "actual": tree},
        )

    normalized: list[ServiceRuntimeObservation] = []
    for item in observations:
        if isinstance(item, ServiceRuntimeObservation):
            normalized.append(item)
        else:
            normalized.append(ServiceRuntimeObservation.from_dict(item))

    by_role = {item.role: item for item in normalized}
    if set(by_role) != set(REQUIRED_SERVICE_ROLES):
        raise RuntimeServiceIdentityError(
            "observations must cover accelerate, datasets, and kit exactly once",
            reason_code="observation_roles_incomplete",
            details={"roles": sorted(by_role)},
        )
    if len(by_role) != len(normalized):
        raise RuntimeServiceIdentityError(
            "duplicate observation roles are forbidden",
            reason_code="duplicate_observation_role",
        )

    # Reject port disagreements across the observation set before binding.
    observed_ports = []
    for item in normalized:
        if item.endpoint is None:
            continue
        observed_ports.append(item.endpoint.port)
    if len(observed_ports) != len(set(observed_ports)):
        raise RuntimeServiceIdentityError(
            "port disagreements across observed service roles",
            reason_code=InvalidationReason.PORT_DISAGREEMENT.value,
            details={"ports": observed_ports},
        )

    role_witnesses: list[ServiceRoleWitness] = []
    for role in REQUIRED_SERVICE_ROLES:
        declaration = auth.service_for_role(role)
        observation = by_role[role]
        if observation.commit != commit or observation.tree != tree:
            raise RuntimeServiceIdentityError(
                "observation checkout disagrees with repository checkout",
                reason_code=InvalidationReason.WRONG_CHECKOUT.value,
                details={
                    "role": role,
                    "repo_commit": commit,
                    "repo_tree": tree,
                    "observation_commit": observation.commit,
                    "observation_tree": observation.tree,
                },
            )
        role_witnesses.append(
            bind_role_witness(
                declaration,
                observation,
                environment_allowlist=auth.environment_allowlist,
            )
        )

    reason_codes = (
        "roles_bound",
        "process_identity_bound",
        "module_origins_bound",
        "commit_tree_bound",
        "arguments_bound",
        "environment_allowlist_bound",
        "configuration_cid_bound",
        "state_cid_bound",
        "transport_bound",
        "endpoint_bound",
        "pid_start_time_bound",
        "port_disagreements_removed",
    )
    return RuntimeServiceWitness(
        service_id=auth.service_id,
        manifest_cid=auth.manifest_cid,
        commit=commit,
        tree=tree,
        roles=tuple(role_witnesses),
        reason_codes=reason_codes,
        passed=True,
    )


def validate_observation_against_witness(
    witness: RuntimeServiceWitness | ServiceRoleWitness,
    observation: ServiceRuntimeObservation | Mapping[str, Any],
) -> ObservationVerdict:
    """Validate a later observation against a previously bound witness.

    Process replacement, config/state change, wrong checkout, or an unbound
    endpoint invalidate the observation.  Endpoint availability without process
    identity is also insufficient.
    """

    obs = (
        observation
        if isinstance(observation, ServiceRuntimeObservation)
        else ServiceRuntimeObservation.from_dict(observation)
    )
    if isinstance(witness, RuntimeServiceWitness):
        role_witness = witness.role_witness(obs.role)
        expected_commit = witness.commit
        expected_tree = witness.tree
    else:
        role_witness = witness
        expected_commit = role_witness.commit
        expected_tree = role_witness.tree

    reasons: list[str] = []
    details: dict[str, Any] = {"role": obs.role}

    if obs.endpoint is None or not obs.endpoint_bound:
        reasons.append(InvalidationReason.UNBOUND_ENDPOINT.value)
        details["endpoint_bound"] = obs.endpoint_bound
    if obs.process is None:
        if obs.endpoint_available or obs.endpoint is not None:
            reasons.append(InvalidationReason.ENDPOINT_WITHOUT_PROCESS_IDENTITY.value)
        else:
            reasons.append(InvalidationReason.UNBOUND_ENDPOINT.value)
    if obs.endpoint is not None and obs.endpoint.port != role_witness.endpoint.port:
        reasons.append(InvalidationReason.PORT_DISAGREEMENT.value)
        details["reviewed_port"] = role_witness.endpoint.port
        details["observed_port"] = obs.endpoint.port
    if obs.endpoint is not None and obs.endpoint.to_dict() != role_witness.endpoint.to_dict():
        if InvalidationReason.PORT_DISAGREEMENT.value not in reasons:
            reasons.append(InvalidationReason.PORT_DISAGREEMENT.value)
        details["reviewed_endpoint"] = role_witness.endpoint.to_dict()
        details["observed_endpoint"] = obs.endpoint.to_dict()
    if obs.process is not None and (
        obs.process.pid != role_witness.process.pid
        or obs.process.start_time != role_witness.process.start_time
    ):
        reasons.append(InvalidationReason.PROCESS_REPLACEMENT.value)
        details["reviewed_process"] = role_witness.process.to_dict()
        details["observed_process"] = obs.process.to_dict()
    if obs.process is not None and obs.process.interpreter != role_witness.process.interpreter:
        reasons.append(InvalidationReason.INTERPRETER_MISMATCH.value)
    if (
        obs.configuration_cid != role_witness.configuration_cid
        or obs.state_cid != role_witness.state_cid
    ):
        reasons.append(InvalidationReason.CONFIG_STATE_CHANGED.value)
        details["reviewed_configuration_cid"] = role_witness.configuration_cid
        details["observed_configuration_cid"] = obs.configuration_cid
        details["reviewed_state_cid"] = role_witness.state_cid
        details["observed_state_cid"] = obs.state_cid
    if obs.commit != expected_commit or obs.tree != expected_tree:
        reasons.append(InvalidationReason.WRONG_CHECKOUT.value)
        details["reviewed_commit"] = expected_commit
        details["observed_commit"] = obs.commit
        details["reviewed_tree"] = expected_tree
        details["observed_tree"] = obs.tree
    if obs.transport != role_witness.transport:
        reasons.append(InvalidationReason.TRANSPORT_MISMATCH.value)
    if tuple(obs.arguments) != tuple(role_witness.arguments):
        reasons.append(InvalidationReason.ARGUMENT_MISMATCH.value)
    if dict(obs.environment) != dict(role_witness.environment):
        # Only invalidate when allowlisted keys diverge; extras are ignored by
        # observation construction, so equality is exact on the allowlist.
        reasons.append(InvalidationReason.ENVIRONMENT_MISMATCH.value)

    if reasons:
        disposition = (
            ObservationDisposition.UNBOUND
            if InvalidationReason.UNBOUND_ENDPOINT.value in reasons
            or InvalidationReason.ENDPOINT_WITHOUT_PROCESS_IDENTITY.value in reasons
            else ObservationDisposition.INVALIDATED
        )
        return ObservationVerdict(
            disposition=disposition,
            role=obs.role,
            valid=False,
            reason_codes=tuple(dict.fromkeys(reasons)),
            details=MappingProxyType(details),
        )
    return ObservationVerdict(
        disposition=ObservationDisposition.VALID,
        role=obs.role,
        valid=True,
        reason_codes=("observation_matches_witness",),
        details=MappingProxyType({"role": obs.role}),
    )


def write_runtime_witness(
    witness: RuntimeServiceWitness,
    *,
    path: str | Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Atomically write a runtime-witness receipt JSON."""

    root = _discover_repo_root(repo_root)
    out = Path(path) if path else root / DEFAULT_WITNESS_RELATIVE
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = witness.to_dict()
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, out)
    return out


def current_interpreter() -> str:
    """Return the absolute path of the running interpreter."""

    return str(Path(sys.executable).resolve())


def synthesize_bound_observations(
    manifest: RuntimeServiceManifest,
    *,
    repo_root: Path | None = None,
    pid_base: int = 10_000,
    start_time_base: str = "boot0:ticks:",
    environment: Mapping[str, Any] | None = None,
    bind_live_modules: bool = False,
) -> list[ServiceRuntimeObservation]:
    """Synthesize launch-consistent observations for hermetic tests and dry runs.

    When *bind_live_modules* is false, module digests are derived from the
    module name + reviewed package root (stable, import-free).  Live imports are
    optional and only used when explicitly requested.
    """

    root = _discover_repo_root(repo_root)
    git = _git_identity(root)
    env = filter_environment(environment or os.environ, manifest.environment_allowlist)
    observations: list[ServiceRuntimeObservation] = []
    for index, declaration in enumerate(manifest.services):
        if bind_live_modules:
            modules: tuple[Mapping[str, Any], ...] = _bind_modules(declaration.modules)
        else:
            modules = tuple(
                {
                    "module": name,
                    "path": str((root / declaration.root / name.replace(".", "/")).with_suffix(".py")),
                    "byte_length": len(name.encode("utf-8")),
                    "digest": _sha256_label(
                        f"{declaration.package}:{name}:{declaration.root}".encode("utf-8")
                    ),
                }
                for name in declaration.modules
            )
        observations.append(
            ServiceRuntimeObservation(
                role=declaration.role,
                process=ProcessIdentity(
                    pid=pid_base + index,
                    start_time=f"{start_time_base}{index}",
                    interpreter=current_interpreter(),
                ),
                arguments=declaration.arguments,
                environment=MappingProxyType(dict(env)),
                endpoint=declaration.endpoint,
                endpoint_bound=True,
                endpoint_available=True,
                commit=git["commit"],
                tree=git["tree"],
                modules=modules,
                configuration_cid=declaration.configuration_cid,
                state_cid=declaration.state_cid,
                transport=declaration.transport,
            )
        )
    return observations


__all__ = [
    "DEFAULT_SERVICES_RELATIVE",
    "DEFAULT_WITNESS_RELATIVE",
    "InvalidationReason",
    "ObservationDisposition",
    "ObservationVerdict",
    "ProcessIdentity",
    "REQUIRED_SERVICE_ROLES",
    "RUNTIME_SERVICE_MANIFEST_INTERFACE",
    "RUNTIME_SERVICE_MANIFEST_SCHEMA",
    "RUNTIME_SERVICE_WITNESS_INTERFACE",
    "RUNTIME_SERVICE_WITNESS_SCHEMA",
    "RUNTIME_WITNESS_EVIDENCE_TERM",
    "RuntimeServiceIdentityError",
    "RuntimeServiceManifest",
    "RuntimeServiceWitness",
    "ServiceEndpoint",
    "ServiceRoleDeclaration",
    "ServiceRoleWitness",
    "ServiceRuntimeObservation",
    "bind_role_witness",
    "build_runtime_service_witness",
    "build_service_observation",
    "content_cid",
    "current_interpreter",
    "filter_environment",
    "is_pseudo_cid",
    "load_runtime_service_manifest",
    "module_file_digest",
    "require_multiformat_cid",
    "synthesize_bound_observations",
    "validate_observation_against_witness",
    "write_runtime_witness",
]
