"""Bounded interprocedural security-property and dataflow analysis (VFS-030).

Conservative symbolic rules operate over declared flow edges and security
properties.  There is no LLM vulnerability classification.  A finding is
labelled a *vulnerability* only when all four are present:

* a declared security property for the rule family;
* a reachable (resolved) or declared threat path;
* a concrete impact statement; and
* at least one evidence artifact reference (CID/handle, never a body).

Anything weaker is classified as correctness drift, suspicion, or an unknown
dynamic frontier.  Large source, AST, proof, and secret bodies stay outside
these records as artifact references.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, TypeVar

from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)


# ---------------------------------------------------------------------------
# Schema / version / bounds
# ---------------------------------------------------------------------------

SECURITY_CONTRACT_ANALYSIS_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = SECURITY_CONTRACT_ANALYSIS_VERSION
ANALYZER_VERSION: Final[str] = "security-contract-analysis@1"
GOAL_ID: Final[str] = "VFS-030"

SECURITY_ANALYSIS_IS_COMPLETION_EVIDENCE: Final[bool] = False
SECURITY_ANALYSIS_AUTHORIZES_REPAIR: Final[bool] = False

FLOW_NODE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-flow-node@1"
)
FLOW_EDGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-flow-edge@1"
)
SECURITY_PROPERTY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-property@1"
)
THREAT_PATH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-threat-path@1"
)
SECURITY_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-evidence@1"
)
SECURITY_FINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-finding@1"
)
SECURITY_ANALYSIS_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-analysis-report@1"
)
SECURITY_RULE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-rule@1"
)

MAX_TEXT_BYTES: Final[int] = 8_192
MAX_CLAUSE_BYTES: Final[int] = 4_096
MAX_COLLECTION_ITEMS: Final[int] = 256
MAX_GRAPH_NODES: Final[int] = 4_096
MAX_GRAPH_EDGES: Final[int] = 16_384
MAX_PATH_HOPS: Final[int] = 16
MAX_PATHS_PER_RULE: Final[int] = 64
MAX_FINDINGS: Final[int] = 512
MAX_RECORD_BYTES: Final[int] = 262_144
DEFAULT_MAX_HOPS: Final[int] = 8
DEFAULT_MAX_FINDINGS: Final[int] = 128
DEFAULT_MAX_PATHS_PER_RULE: Final[int] = 16

# Forbidden body-like fields: never accepted as evidence material.
_FORBIDDEN_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "source",
        "source_text",
        "source_body",
        "body",
        "code",
        "code_body",
        "secret",
        "secret_value",
        "password",
        "token",
        "api_key",
        "private_key",
        "payload_body",
        "raw_source",
        "ast_body",
        "proof_body",
        "witness_body",
    }
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SecurityRuleFamily(str, Enum):
    """Closed set of bounded interprocedural security rule families."""

    PATH_TRAVERSAL_SCOPE_LOSS = "path_traversal_scope_loss"
    AUTHORIZATION_CAPABILITY_BYPASS = "authorization_capability_bypass"
    UNSAFE_DESERIALIZATION_COMMAND = "unsafe_deserialization_command"
    SECRET_FLOW = "secret_flow"
    CID_INTEGRITY_BYPASS = "cid_integrity_bypass"
    CACHE_POISONING_STALENESS = "cache_poisoning_staleness"
    SYMLINK_ESCAPE = "symlink_escape"
    SILENT_FALLBACK_MOCK_SUCCESS = "silent_fallback_mock_success"
    JOURNAL_ATOMICITY_VIOLATION = "journal_atomicity_violation"
    MCP_SCHEMA_DISPATCH_CONFUSION = "mcp_schema_dispatch_confusion"


class FindingClassification(str, Enum):
    """How a matched rule is labelled after the vulnerability gate."""

    VULNERABILITY = "vulnerability"
    CORRECTNESS_DRIFT = "correctness_drift"
    SUSPICION = "suspicion"
    UNKNOWN_DYNAMIC = "unknown_dynamic"


class FlowRole(str, Enum):
    """Role of a node in the security flow graph."""

    SOURCE = "source"
    SINK = "sink"
    SANITIZER = "sanitizer"
    CHECK = "check"
    PASSTHROUGH = "passthrough"
    ENTRY = "entry"
    EXIT = "exit"


class EdgeResolution(str, Enum):
    """Whether a flow edge is closed or an unknown frontier."""

    RESOLVED = "resolved"
    DECLARED = "declared"
    UNRESOLVED = "unresolved"
    DYNAMIC = "dynamic"
    AMBIGUOUS = "ambiguous"


class ThreatPathOrigin(str, Enum):
    REACHABLE = "reachable"
    DECLARED = "declared"


class AnalysisVerdict(str, Enum):
    CLEAN = "clean"
    FINDINGS = "findings"
    BOUNDED = "bounded"
    EMPTY = "empty"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SecurityContractAnalysisError(ContractValidationError):
    """Base error for security contract analysis."""


class SecurityContractBoundsError(SecurityContractAnalysisError):
    """A bound (items, hops, bytes) was exceeded."""


class ForgedSecurityIdentityError(SecurityContractAnalysisError):
    """A caller-supplied identity did not match the derived content id."""


class ForbiddenBodyError(SecurityContractAnalysisError):
    """Source/secret body material was presented as evidence."""


E = TypeVar("E", bound=Enum)
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    *,
    field_name: str,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        if required:
            raise SecurityContractAnalysisError(f"{field_name} is required")
        return ""
    if not isinstance(value, str):
        raise SecurityContractAnalysisError(f"{field_name} must be a string")
    if "\x00" in value:
        raise SecurityContractAnalysisError(
            f"{field_name} must not contain NUL"
        )
    if len(value.encode("utf-8")) > maximum:
        raise SecurityContractBoundsError(
            f"{field_name} exceeds {maximum} bytes"
        )
    if required and not value.strip():
        raise SecurityContractAnalysisError(f"{field_name} must be non-empty")
    return value


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise SecurityContractAnalysisError(f"{field_name} must be a boolean")
    return value


def _integer(
    value: Any,
    *,
    field_name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SecurityContractAnalysisError(f"{field_name} must be an integer")
    if value < minimum:
        raise SecurityContractAnalysisError(
            f"{field_name} must be >= {minimum}"
        )
    if maximum is not None and value > maximum:
        raise SecurityContractBoundsError(
            f"{field_name} exceeds maximum {maximum}"
        )
    return value


def _enum(value: Any, enum_type: type[E], *, field_name: str) -> E:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:
            raise SecurityContractAnalysisError(
                f"{field_name} is not a valid {enum_type.__name__}"
            ) from exc
    raise SecurityContractAnalysisError(
        f"{field_name} must be a {enum_type.__name__} or string"
    )


def _strings(
    values: Any,
    *,
    field_name: str,
    required: bool = False,
    maximum: int = MAX_COLLECTION_ITEMS,
    item_bytes: int = MAX_CLAUSE_BYTES,
    sort: bool = False,
    unique: bool = False,
) -> tuple[str, ...]:
    if values is None:
        if required:
            raise SecurityContractAnalysisError(f"{field_name} is required")
        return ()
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise SecurityContractAnalysisError(
            f"{field_name} must be a sequence of strings"
        )
    if len(values) > maximum:
        raise SecurityContractBoundsError(
            f"{field_name} exceeds {maximum} items"
        )
    items: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        text = _text(
            raw,
            field_name=f"{field_name}[{index}]",
            required=True,
            maximum=item_bytes,
        )
        if unique and text in seen:
            raise SecurityContractAnalysisError(
                f"{field_name} contains duplicate entry {text!r}"
            )
        seen.add(text)
        items.append(text)
    if sort:
        items = sorted(items)
    if required and not items:
        raise SecurityContractAnalysisError(f"{field_name} must be non-empty")
    return tuple(items)


def _check_header(payload: Mapping[str, Any], expected_schema: str) -> None:
    if not isinstance(payload, Mapping):
        raise SecurityContractAnalysisError("payload must be an object")
    schema = payload.get("schema")
    if schema != expected_schema:
        raise SecurityContractAnalysisError(
            f"unsupported schema {schema!r}; expected {expected_schema!r}"
        )
    version = payload.get("schema_version", payload.get("contract_version"))
    if version is not None and int(version) != SECURITY_CONTRACT_ANALYSIS_VERSION:
        raise SecurityContractAnalysisError(
            f"unsupported schema_version {version!r}"
        )


def _reject_unknown(
    payload: Mapping[str, Any],
    allowed: set[str],
    *,
    artifact_name: str,
) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise SecurityContractAnalysisError(
            f"{artifact_name} contains unknown fields: {sorted(unknown)}"
        )


def _check_identity(
    payload: Mapping[str, Any],
    actual: str,
    *,
    names: Sequence[str],
    artifact_name: str,
) -> None:
    for name in names:
        if name in payload and payload[name] not in (None, "", actual):
            raise ForgedSecurityIdentityError(
                f"{artifact_name} {name} does not match derived identity"
            )


def _bounded(record: CanonicalContract, *, artifact_name: str) -> None:
    encoded = record.canonical_bytes()
    if len(encoded) > MAX_RECORD_BYTES:
        raise SecurityContractBoundsError(
            f"{artifact_name} exceeds {MAX_RECORD_BYTES} serialized bytes"
        )


def _header_fields() -> set[str]:
    return {
        "schema",
        "schema_version",
        "contract_version",
        "content_id",
        "cid",
    }


def _reject_body_keys(payload: Mapping[str, Any], *, field_name: str) -> None:
    for key in payload:
        lowered = str(key).lower()
        if lowered in _FORBIDDEN_BODY_KEYS:
            raise ForbiddenBodyError(
                f"{field_name} must not include body/secret field {key!r}"
            )


def _record(
    value: Any,
    cls: type[T],
    *,
    field_name: str,
    optional: bool = False,
) -> T | None:
    if value is None:
        if optional:
            return None
        raise SecurityContractAnalysisError(f"{field_name} is required")
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        from_dict = getattr(cls, "from_dict", None)
        if from_dict is None:
            raise SecurityContractAnalysisError(
                f"{field_name} cannot be decoded from mapping"
            )
        return from_dict(value)
    raise SecurityContractAnalysisError(
        f"{field_name} must be a {cls.__name__} or mapping"
    )


def _records(
    values: Any,
    cls: type[T],
    *,
    field_name: str,
    maximum: int = MAX_COLLECTION_ITEMS,
) -> tuple[T, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SecurityContractAnalysisError(
            f"{field_name} must be a sequence of {cls.__name__}"
        )
    if len(values) > maximum:
        raise SecurityContractBoundsError(
            f"{field_name} exceeds {maximum} items"
        )
    result: list[T] = []
    for index, item in enumerate(values):
        field = f"{field_name}[{index}]"
        decoded = _record(item, cls, field_name=field)
        if decoded is None:
            raise SecurityContractAnalysisError(f"{field} is required")
        result.append(decoded)
    return tuple(result)


# ---------------------------------------------------------------------------
# Rule catalog (closed, deterministic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecurityRuleSpec:
    """Closed rule specification: source tags, sink tags, sanitizer tags."""

    family: SecurityRuleFamily
    rule_id: str
    name: str
    short_description: str
    source_tags: frozenset[str]
    sink_tags: frozenset[str]
    sanitizer_tags: frozenset[str]
    default_impact: str
    default_severity: str  # info|low|medium|high|critical


_RULE_SPECS: Final[tuple[SecurityRuleSpec, ...]] = (
    SecurityRuleSpec(
        family=SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS,
        rule_id="sec/path-traversal-scope-loss",
        name="Path traversal / scope loss",
        short_description=(
            "Untrusted path component reaches a filesystem operation "
            "without scope confinement."
        ),
        source_tags=frozenset(
            {"untrusted_path", "user_path", "request_path", "cli_path"}
        ),
        sink_tags=frozenset(
            {"fs_open", "fs_read", "fs_write", "fs_unlink", "path_resolve"}
        ),
        sanitizer_tags=frozenset(
            {"path_canonicalize", "scope_confine", "root_jail"}
        ),
        default_impact="Arbitrary file read/write outside the declared root.",
        default_severity="high",
    ),
    SecurityRuleSpec(
        family=SecurityRuleFamily.AUTHORIZATION_CAPABILITY_BYPASS,
        rule_id="sec/authorization-capability-bypass",
        name="Authorization / capability bypass",
        short_description=(
            "Privileged action is reachable without a matching capability check."
        ),
        source_tags=frozenset(
            {"unauthenticated", "untrusted_principal", "forged_token"}
        ),
        sink_tags=frozenset(
            {
                "privileged_action",
                "capability_grant",
                "lease_mutate",
                "admin_tool",
            }
        ),
        sanitizer_tags=frozenset(
            {"authz_check", "capability_check", "principal_bind"}
        ),
        default_impact="Unauthorized privileged action execution.",
        default_severity="critical",
    ),
    SecurityRuleSpec(
        family=SecurityRuleFamily.UNSAFE_DESERIALIZATION_COMMAND,
        rule_id="sec/unsafe-deserialization-command",
        name="Unsafe deserialization / command construction",
        short_description=(
            "Untrusted data reaches a deserializer or shell/command builder."
        ),
        source_tags=frozenset(
            {"untrusted_bytes", "network_payload", "user_input"}
        ),
        sink_tags=frozenset(
            {
                "pickle_loads",
                "yaml_load",
                "eval_exec",
                "shell_exec",
                "subprocess_shell",
            }
        ),
        sanitizer_tags=frozenset(
            {"safe_deserializer", "command_allowlist", "shell_escape"}
        ),
        default_impact="Remote code execution via deserialization or shell.",
        default_severity="critical",
    ),
    SecurityRuleSpec(
        family=SecurityRuleFamily.SECRET_FLOW,
        rule_id="sec/secret-flow",
        name="Secret flow",
        short_description=(
            "Secret-labelled value reaches a log, export, or remote channel."
        ),
        source_tags=frozenset(
            {"secret_material", "credential", "private_key", "api_token"}
        ),
        sink_tags=frozenset(
            {"log_emit", "http_export", "telemetry", "error_report"}
        ),
        sanitizer_tags=frozenset(
            {"secret_redact", "secret_mask", "secret_hash"}
        ),
        default_impact="Credential or secret disclosure outside trust boundary.",
        default_severity="high",
    ),
    SecurityRuleSpec(
        family=SecurityRuleFamily.CID_INTEGRITY_BYPASS,
        rule_id="sec/cid-integrity-bypass",
        name="CID / integrity bypass",
        short_description=(
            "Content is consumed without verifying the declared CID binding."
        ),
        source_tags=frozenset(
            {"unverified_bytes", "mutable_alias", "cache_bytes"}
        ),
        sink_tags=frozenset(
            {"cid_accept", "content_use", "pin_commit", "verify_skip"}
        ),
        sanitizer_tags=frozenset(
            {"cid_verify", "digest_check", "integrity_bind"}
        ),
        default_impact="Tampered content accepted under a trusted identity.",
        default_severity="high",
    ),
    SecurityRuleSpec(
        family=SecurityRuleFamily.CACHE_POISONING_STALENESS,
        rule_id="sec/cache-poisoning-staleness",
        name="Cache poisoning / staleness",
        short_description=(
            "Stale or unauthenticated cache entry is promoted as current."
        ),
        source_tags=frozenset(
            {"stale_cache", "unpinned_cache", "poisoned_entry"}
        ),
        sink_tags=frozenset(
            {"cache_serve", "authority_promote", "freshness_claim"}
        ),
        sanitizer_tags=frozenset(
            {"freshness_check", "pin_coherence", "cache_invalidate"}
        ),
        default_impact="Stale or attacker-controlled data served as current.",
        default_severity="medium",
    ),
    SecurityRuleSpec(
        family=SecurityRuleFamily.SYMLINK_ESCAPE,
        rule_id="sec/symlink-escape",
        name="Symlink escape",
        short_description=(
            "Symlink or link-following path escapes the declared root."
        ),
        source_tags=frozenset(
            {"symlink_path", "link_target", "user_link"}
        ),
        sink_tags=frozenset(
            {"fs_follow", "fs_open", "fs_read", "fs_write"}
        ),
        sanitizer_tags=frozenset(
            {"no_follow", "lstat_only", "symlink_reject"}
        ),
        default_impact="Symlink-based escape from the declared filesystem root.",
        default_severity="high",
    ),
    SecurityRuleSpec(
        family=SecurityRuleFamily.SILENT_FALLBACK_MOCK_SUCCESS,
        rule_id="sec/silent-fallback-mock-success",
        name="Silent fallback / mock success",
        short_description=(
            "Failure path is replaced by a mock or silent success without "
            "declaring degradation."
        ),
        source_tags=frozenset(
            {"backend_error", "capability_missing", "timeout"}
        ),
        sink_tags=frozenset(
            {"mock_success", "silent_ok", "fabricated_result"}
        ),
        sanitizer_tags=frozenset(
            {"degradation_declare", "error_propagate", "fallback_audit"}
        ),
        default_impact=(
            "Callers treat degraded/mock results as real success authority."
        ),
        default_severity="medium",
    ),
    SecurityRuleSpec(
        family=SecurityRuleFamily.JOURNAL_ATOMICITY_VIOLATION,
        rule_id="sec/journal-atomicity-violation",
        name="Journal / atomicity violation",
        short_description=(
            "Mutable state is committed without journal fencing or atomic "
            "replace."
        ),
        source_tags=frozenset(
            {"partial_write", "unfenced_mutate", "torn_update"}
        ),
        sink_tags=frozenset(
            {"commit_visible", "journal_skip", "alias_swap"}
        ),
        sanitizer_tags=frozenset(
            {"atomic_replace", "journal_fence", "fsync_commit"}
        ),
        default_impact="Torn or non-atomic state becomes externally visible.",
        default_severity="high",
    ),
    SecurityRuleSpec(
        family=SecurityRuleFamily.MCP_SCHEMA_DISPATCH_CONFUSION,
        rule_id="sec/mcp-schema-dispatch-confusion",
        name="MCP schema / dispatch confusion",
        short_description=(
            "Tool dispatch binds a name to a schema or implementation that "
            "does not match the declared contract."
        ),
        source_tags=frozenset(
            {
                "schema_drift",
                "alias_collision",
                "unresolved_dispatch",
                "mock_dispatch",
            }
        ),
        sink_tags=frozenset(
            {
                "mcp_invoke",
                "tool_dispatch",
                "schema_accept",
                "capability_claim",
            }
        ),
        sanitizer_tags=frozenset(
            {"schema_bind", "dispatch_resolve", "path_prove"}
        ),
        default_impact=(
            "Wrong tool, schema, or mock implementation is invoked under a "
            "trusted name."
        ),
        default_severity="high",
    ),
)

_RULES_BY_FAMILY: Final[Mapping[SecurityRuleFamily, SecurityRuleSpec]] = (
    MappingProxyType({spec.family: spec for spec in _RULE_SPECS})
)
_RULES_BY_ID: Final[Mapping[str, SecurityRuleSpec]] = MappingProxyType(
    {spec.rule_id: spec for spec in _RULE_SPECS}
)


def security_rule_families() -> tuple[SecurityRuleFamily, ...]:
    """Return the closed, ordered rule family set."""

    return tuple(spec.family for spec in _RULE_SPECS)


def security_rule_spec(family: SecurityRuleFamily | str) -> SecurityRuleSpec:
    """Look up a closed rule specification by family."""

    family_e = _enum(family, SecurityRuleFamily, field_name="family")
    return _RULES_BY_FAMILY[family_e]


def security_rule_specs() -> tuple[SecurityRuleSpec, ...]:
    """Return all closed rule specifications in deterministic order."""

    return _RULE_SPECS


# ---------------------------------------------------------------------------
# Content-addressed records
# ---------------------------------------------------------------------------


class _SecurityContract(CanonicalContract):
    """Shared helpers for security analysis records."""

    @property
    def schema_version(self) -> int:
        return SECURITY_CONTRACT_ANALYSIS_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "schema_version": self.schema_version,
            "contract_version": SECURITY_CONTRACT_ANALYSIS_VERSION,
            **self._payload(),
        }


@dataclass(frozen=True)
class FlowNode(_SecurityContract):
    """One node in the bounded security flow graph."""

    SCHEMA: ClassVar[str] = FLOW_NODE_SCHEMA

    node_id: str
    symbol: str
    role: FlowRole = FlowRole.PASSTHROUGH
    tags: tuple[str, ...] = ()
    path: str = ""
    repository_id: str = ""
    interface: str = ""
    line: int = 0
    column: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "node_id", _text(self.node_id, field_name="node_id")
        )
        object.__setattr__(
            self, "symbol", _text(self.symbol, field_name="symbol")
        )
        object.__setattr__(
            self, "role", _enum(self.role, FlowRole, field_name="role")
        )
        object.__setattr__(
            self,
            "tags",
            _strings(self.tags, field_name="tags", unique=True, sort=True),
        )
        for name in ("path", "repository_id", "interface"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=False),
            )
        object.__setattr__(
            self,
            "line",
            _integer(self.line, field_name="line", minimum=0, maximum=10_000_000),
        )
        object.__setattr__(
            self,
            "column",
            _integer(
                self.column, field_name="column", minimum=0, maximum=1_000_000
            ),
        )
        _bounded(self, artifact_name="flow node")

    @property
    def content_cid(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "symbol": self.symbol,
            "role": self.role.value,
            "tags": self.tags,
            "path": self.path,
            "repository_id": self.repository_id,
            "interface": self.interface,
            "line": self.line,
            "column": self.column,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FlowNode":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            _header_fields()
            | {
                "node_id",
                "symbol",
                "role",
                "tags",
                "path",
                "repository_id",
                "interface",
                "line",
                "column",
            },
            artifact_name="flow node",
        )
        result = cls(
            node_id=payload.get("node_id", ""),
            symbol=payload.get("symbol", ""),
            role=payload.get("role", FlowRole.PASSTHROUGH),
            tags=tuple(payload.get("tags") or ()),
            path=payload.get("path", ""),
            repository_id=payload.get("repository_id", ""),
            interface=payload.get("interface", ""),
            line=int(payload.get("line") or 0),
            column=int(payload.get("column") or 0),
        )
        _check_identity(
            payload,
            result.content_id,
            names=("content_id", "cid"),
            artifact_name="flow node",
        )
        return result


@dataclass(frozen=True)
class FlowEdge(_SecurityContract):
    """Directed flow edge between two nodes."""

    SCHEMA: ClassVar[str] = FLOW_EDGE_SCHEMA

    edge_id: str
    source_id: str
    target_id: str
    resolution: EdgeResolution = EdgeResolution.RESOLVED
    labels: tuple[str, ...] = ()
    kind: str = "dataflow"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "edge_id", _text(self.edge_id, field_name="edge_id")
        )
        object.__setattr__(
            self, "source_id", _text(self.source_id, field_name="source_id")
        )
        object.__setattr__(
            self, "target_id", _text(self.target_id, field_name="target_id")
        )
        object.__setattr__(
            self,
            "resolution",
            _enum(self.resolution, EdgeResolution, field_name="resolution"),
        )
        object.__setattr__(
            self,
            "labels",
            _strings(self.labels, field_name="labels", unique=True, sort=True),
        )
        object.__setattr__(
            self,
            "kind",
            _text(self.kind, field_name="kind", required=False) or "dataflow",
        )
        _bounded(self, artifact_name="flow edge")

    @property
    def is_closed(self) -> bool:
        return self.resolution in {
            EdgeResolution.RESOLVED,
            EdgeResolution.DECLARED,
        }

    @property
    def is_unknown_dynamic(self) -> bool:
        return self.resolution in {
            EdgeResolution.UNRESOLVED,
            EdgeResolution.DYNAMIC,
            EdgeResolution.AMBIGUOUS,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "resolution": self.resolution.value,
            "labels": self.labels,
            "kind": self.kind,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FlowEdge":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            _header_fields()
            | {
                "edge_id",
                "source_id",
                "target_id",
                "resolution",
                "labels",
                "kind",
            },
            artifact_name="flow edge",
        )
        result = cls(
            edge_id=payload.get("edge_id", ""),
            source_id=payload.get("source_id", ""),
            target_id=payload.get("target_id", ""),
            resolution=payload.get("resolution", EdgeResolution.RESOLVED),
            labels=tuple(payload.get("labels") or ()),
            kind=payload.get("kind", "dataflow"),
        )
        _check_identity(
            payload,
            result.content_id,
            names=("content_id", "cid"),
            artifact_name="flow edge",
        )
        return result


@dataclass(frozen=True)
class SecurityPropertyDeclaration(_SecurityContract):
    """A declared security property that a rule family is meant to uphold."""

    SCHEMA: ClassVar[str] = SECURITY_PROPERTY_SCHEMA

    property_id: str
    family: SecurityRuleFamily
    resource: str
    statement: str
    constraints: tuple[str, ...] = ()
    repository_id: str = ""
    interface: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "property_id",
            _text(self.property_id, field_name="property_id"),
        )
        object.__setattr__(
            self,
            "family",
            _enum(self.family, SecurityRuleFamily, field_name="family"),
        )
        object.__setattr__(
            self, "resource", _text(self.resource, field_name="resource")
        )
        object.__setattr__(
            self, "statement", _text(self.statement, field_name="statement")
        )
        object.__setattr__(
            self,
            "constraints",
            _strings(
                self.constraints,
                field_name="constraints",
                unique=True,
                sort=True,
            ),
        )
        for name in ("repository_id", "interface"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=False),
            )
        _bounded(self, artifact_name="security property")

    def _payload(self) -> dict[str, Any]:
        return {
            "property_id": self.property_id,
            "family": self.family.value,
            "resource": self.resource,
            "statement": self.statement,
            "constraints": self.constraints,
            "repository_id": self.repository_id,
            "interface": self.interface,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "SecurityPropertyDeclaration":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            _header_fields()
            | {
                "property_id",
                "family",
                "resource",
                "statement",
                "constraints",
                "repository_id",
                "interface",
            },
            artifact_name="security property",
        )
        result = cls(
            property_id=payload.get("property_id", ""),
            family=payload.get("family", ""),
            resource=payload.get("resource", ""),
            statement=payload.get("statement", ""),
            constraints=tuple(payload.get("constraints") or ()),
            repository_id=payload.get("repository_id", ""),
            interface=payload.get("interface", ""),
        )
        _check_identity(
            payload,
            result.content_id,
            names=("content_id", "cid"),
            artifact_name="security property",
        )
        return result


@dataclass(frozen=True)
class ThreatPath(_SecurityContract):
    """A reachable or declared threat path through the flow graph."""

    SCHEMA: ClassVar[str] = THREAT_PATH_SCHEMA

    path_id: str
    node_ids: tuple[str, ...]
    origin: ThreatPathOrigin = ThreatPathOrigin.REACHABLE
    edge_ids: tuple[str, ...] = ()
    has_unknown_dynamic: bool = False
    hop_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "path_id", _text(self.path_id, field_name="path_id")
        )
        object.__setattr__(
            self,
            "node_ids",
            _strings(
                self.node_ids,
                field_name="node_ids",
                required=True,
                maximum=MAX_PATH_HOPS + 1,
            ),
        )
        object.__setattr__(
            self,
            "origin",
            _enum(self.origin, ThreatPathOrigin, field_name="origin"),
        )
        object.__setattr__(
            self,
            "edge_ids",
            _strings(
                self.edge_ids,
                field_name="edge_ids",
                maximum=MAX_PATH_HOPS,
            ),
        )
        object.__setattr__(
            self,
            "has_unknown_dynamic",
            _boolean(
                self.has_unknown_dynamic, field_name="has_unknown_dynamic"
            ),
        )
        hops = len(self.node_ids) - 1 if self.node_ids else 0
        object.__setattr__(
            self,
            "hop_count",
            _integer(
                self.hop_count if self.hop_count else hops,
                field_name="hop_count",
                minimum=0,
                maximum=MAX_PATH_HOPS,
            ),
        )
        if len(self.node_ids) < 2:
            raise SecurityContractAnalysisError(
                "threat path requires at least two nodes"
            )
        _bounded(self, artifact_name="threat path")

    def _payload(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "node_ids": self.node_ids,
            "origin": self.origin.value,
            "edge_ids": self.edge_ids,
            "has_unknown_dynamic": self.has_unknown_dynamic,
            "hop_count": self.hop_count,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ThreatPath":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            _header_fields()
            | {
                "path_id",
                "node_ids",
                "origin",
                "edge_ids",
                "has_unknown_dynamic",
                "hop_count",
            },
            artifact_name="threat path",
        )
        result = cls(
            path_id=payload.get("path_id", ""),
            node_ids=tuple(payload.get("node_ids") or ()),
            origin=payload.get("origin", ThreatPathOrigin.REACHABLE),
            edge_ids=tuple(payload.get("edge_ids") or ()),
            has_unknown_dynamic=bool(payload.get("has_unknown_dynamic", False)),
            hop_count=int(payload.get("hop_count") or 0),
        )
        _check_identity(
            payload,
            result.content_id,
            names=("content_id", "cid"),
            artifact_name="threat path",
        )
        return result


@dataclass(frozen=True)
class SecurityEvidence(_SecurityContract):
    """Artifact references only — never source/secret bodies."""

    SCHEMA: ClassVar[str] = SECURITY_EVIDENCE_SCHEMA

    artifact_cids: tuple[str, ...] = ()
    counterexample_cids: tuple[str, ...] = ()
    proof_cids: tuple[str, ...] = ()
    runtime_cids: tuple[str, ...] = ()
    graph_slice_cids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "artifact_cids",
            "counterexample_cids",
            "proof_cids",
            "runtime_cids",
            "graph_slice_cids",
            "notes",
        ):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    field_name=name,
                    unique=True,
                    sort=True,
                ),
            )
        # Notes must not look like embedded secrets/source.
        for note in self.notes:
            lowered = note.lower()
            for banned in ("password=", "api_key=", "-----begin", "secret="):
                if banned in lowered:
                    raise ForbiddenBodyError(
                        "evidence notes must not embed secret material"
                    )
        _bounded(self, artifact_name="security evidence")

    @property
    def has_evidence(self) -> bool:
        return bool(
            self.artifact_cids
            or self.counterexample_cids
            or self.proof_cids
            or self.runtime_cids
            or self.graph_slice_cids
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_cids": self.artifact_cids,
            "counterexample_cids": self.counterexample_cids,
            "proof_cids": self.proof_cids,
            "runtime_cids": self.runtime_cids,
            "graph_slice_cids": self.graph_slice_cids,
            "notes": self.notes,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SecurityEvidence":
        _check_header(payload, cls.SCHEMA)
        _reject_body_keys(payload, field_name="security evidence")
        _reject_unknown(
            payload,
            _header_fields()
            | {
                "artifact_cids",
                "counterexample_cids",
                "proof_cids",
                "runtime_cids",
                "graph_slice_cids",
                "notes",
            },
            artifact_name="security evidence",
        )
        result = cls(
            artifact_cids=tuple(payload.get("artifact_cids") or ()),
            counterexample_cids=tuple(
                payload.get("counterexample_cids") or ()
            ),
            proof_cids=tuple(payload.get("proof_cids") or ()),
            runtime_cids=tuple(payload.get("runtime_cids") or ()),
            graph_slice_cids=tuple(payload.get("graph_slice_cids") or ()),
            notes=tuple(payload.get("notes") or ()),
        )
        _check_identity(
            payload,
            result.content_id,
            names=("content_id", "cid"),
            artifact_name="security evidence",
        )
        return result


# ---------------------------------------------------------------------------
# Classification gate
# ---------------------------------------------------------------------------


def vulnerability_requirements_met(
    *,
    security_property: SecurityPropertyDeclaration | None,
    threat_path: ThreatPath | None,
    impact: str,
    evidence: SecurityEvidence | None,
) -> tuple[bool, tuple[str, ...]]:
    """Return whether vulnerability labelling is justified, with missing keys."""

    missing: list[str] = []
    if security_property is None:
        missing.append("security_property")
    if threat_path is None:
        missing.append("threat_path")
    elif threat_path.has_unknown_dynamic and threat_path.origin is not (
        ThreatPathOrigin.DECLARED
    ):
        # Reachable-but-dynamic paths cannot alone justify vulnerability.
        missing.append("closed_threat_path")
    if not (impact or "").strip():
        missing.append("impact")
    if evidence is None or not evidence.has_evidence:
        missing.append("evidence")
    return (not missing, tuple(missing))


def classify_security_finding(
    *,
    security_property: SecurityPropertyDeclaration | None,
    threat_path: ThreatPath | None,
    impact: str,
    evidence: SecurityEvidence | None,
    sanitized: bool = False,
    family_matched: bool = True,
) -> FindingClassification:
    """Apply the vulnerability gate; otherwise drift / suspicion / dynamic.

    Policy (fail-closed, no LLM):

    * sanitizer on path → not a positive match (caller should suppress);
    * unknown dynamic path without declared property → ``unknown_dynamic``;
    * full property + closed/declared path + impact + evidence →
      ``vulnerability``;
    * property present but incomplete evidence/path → ``suspicion``;
    * match without declared property → ``correctness_drift``.
    """

    if sanitized or not family_matched:
        return FindingClassification.CORRECTNESS_DRIFT

    if threat_path is not None and threat_path.has_unknown_dynamic:
        if security_property is None:
            return FindingClassification.UNKNOWN_DYNAMIC
        # Declared property + dynamic path is suspicion, not vulnerability.
        return FindingClassification.SUSPICION

    ok, missing = vulnerability_requirements_met(
        security_property=security_property,
        threat_path=threat_path,
        impact=impact,
        evidence=evidence,
    )
    if ok:
        return FindingClassification.VULNERABILITY
    if security_property is not None and missing:
        return FindingClassification.SUSPICION
    if security_property is None and threat_path is not None:
        return FindingClassification.CORRECTNESS_DRIFT
    return FindingClassification.SUSPICION


# ---------------------------------------------------------------------------
# Finding + report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecurityFinding(_SecurityContract):
    """One security-property/dataflow finding with classification."""

    SCHEMA: ClassVar[str] = SECURITY_FINDING_SCHEMA

    family: SecurityRuleFamily
    classification: FindingClassification
    rule_id: str
    summary: str
    impact: str = ""
    severity: str = "medium"
    confidence_millionths: int = 500_000
    security_property: SecurityPropertyDeclaration | None = None
    threat_path: ThreatPath | None = None
    evidence: SecurityEvidence = field(default_factory=SecurityEvidence)
    symbols: tuple[str, ...] = ()
    interfaces: tuple[str, ...] = ()
    repositories: tuple[str, ...] = ()
    source_node_id: str = ""
    sink_node_id: str = ""
    missing_requirements: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    remediation_hints: tuple[str, ...] = ()
    tree_id: str = ""
    policy_revision: str = ""
    analyzer_version: str = ANALYZER_VERSION
    root_cause_family: str = ""
    seed_label: str = ""  # true_positive | false_positive | unknown_dynamic | ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family",
            _enum(self.family, SecurityRuleFamily, field_name="family"),
        )
        object.__setattr__(
            self,
            "classification",
            _enum(
                self.classification,
                FindingClassification,
                field_name="classification",
            ),
        )
        object.__setattr__(
            self, "rule_id", _text(self.rule_id, field_name="rule_id")
        )
        object.__setattr__(
            self, "summary", _text(self.summary, field_name="summary")
        )
        object.__setattr__(
            self,
            "impact",
            _text(self.impact, field_name="impact", required=False),
        )
        object.__setattr__(
            self,
            "severity",
            _text(self.severity, field_name="severity", required=False)
            or "medium",
        )
        object.__setattr__(
            self,
            "confidence_millionths",
            _integer(
                self.confidence_millionths,
                field_name="confidence_millionths",
                minimum=0,
                maximum=1_000_000,
            ),
        )
        prop = _record(
            self.security_property,
            SecurityPropertyDeclaration,
            field_name="security_property",
            optional=True,
        )
        object.__setattr__(self, "security_property", prop)
        path = _record(
            self.threat_path,
            ThreatPath,
            field_name="threat_path",
            optional=True,
        )
        object.__setattr__(self, "threat_path", path)
        evidence = _record(
            self.evidence,
            SecurityEvidence,
            field_name="evidence",
            optional=True,
        )
        object.__setattr__(
            self, "evidence", evidence if evidence is not None else SecurityEvidence()
        )
        for name in ("symbols", "interfaces", "repositories"):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    field_name=name,
                    unique=True,
                    sort=True,
                ),
            )
        for name in (
            "source_node_id",
            "sink_node_id",
            "tree_id",
            "policy_revision",
            "analyzer_version",
            "root_cause_family",
            "seed_label",
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name) or "",
                    field_name=name,
                    required=False,
                ),
            )
        if not self.analyzer_version:
            object.__setattr__(self, "analyzer_version", ANALYZER_VERSION)
        if not self.root_cause_family:
            object.__setattr__(
                self, "root_cause_family", self.family.value
            )
        object.__setattr__(
            self,
            "missing_requirements",
            _strings(
                self.missing_requirements,
                field_name="missing_requirements",
                unique=True,
                sort=True,
            ),
        )
        object.__setattr__(
            self,
            "assumptions",
            _strings(
                self.assumptions,
                field_name="assumptions",
                unique=True,
                sort=True,
            ),
        )
        object.__setattr__(
            self,
            "remediation_hints",
            _strings(
                self.remediation_hints,
                field_name="remediation_hints",
                unique=True,
                sort=True,
            ),
        )
        # Vulnerability gate consistency.
        if self.classification is FindingClassification.VULNERABILITY:
            ok, missing = vulnerability_requirements_met(
                security_property=self.security_property,
                threat_path=self.threat_path,
                impact=self.impact,
                evidence=self.evidence,
            )
            if not ok:
                raise SecurityContractAnalysisError(
                    "vulnerability classification requires "
                    f"{', '.join(missing)}"
                )
        _bounded(self, artifact_name="security finding")

    @property
    def finding_id(self) -> str:
        return self.content_id

    @property
    def is_vulnerability(self) -> bool:
        return self.classification is FindingClassification.VULNERABILITY

    def _payload(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "classification": self.classification.value,
            "rule_id": self.rule_id,
            "summary": self.summary,
            "impact": self.impact,
            "severity": self.severity,
            "confidence_millionths": self.confidence_millionths,
            "security_property": (
                self.security_property.to_dict()
                if self.security_property is not None
                else None
            ),
            "threat_path": (
                self.threat_path.to_dict()
                if self.threat_path is not None
                else None
            ),
            "evidence": self.evidence.to_dict(),
            "symbols": self.symbols,
            "interfaces": self.interfaces,
            "repositories": self.repositories,
            "source_node_id": self.source_node_id,
            "sink_node_id": self.sink_node_id,
            "missing_requirements": self.missing_requirements,
            "assumptions": self.assumptions,
            "remediation_hints": self.remediation_hints,
            "tree_id": self.tree_id,
            "policy_revision": self.policy_revision,
            "analyzer_version": self.analyzer_version,
            "root_cause_family": self.root_cause_family,
            "seed_label": self.seed_label,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "finding_id": self.finding_id,
            "content_id": self.content_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SecurityFinding":
        _check_header(payload, cls.SCHEMA)
        _reject_body_keys(payload, field_name="security finding")
        _reject_unknown(
            payload,
            _header_fields()
            | {
                "family",
                "classification",
                "rule_id",
                "summary",
                "impact",
                "severity",
                "confidence_millionths",
                "security_property",
                "threat_path",
                "evidence",
                "symbols",
                "interfaces",
                "repositories",
                "source_node_id",
                "sink_node_id",
                "missing_requirements",
                "assumptions",
                "remediation_hints",
                "tree_id",
                "policy_revision",
                "analyzer_version",
                "root_cause_family",
                "seed_label",
                "finding_id",
            },
            artifact_name="security finding",
        )
        result = cls(
            family=payload.get("family", ""),
            classification=payload.get("classification", ""),
            rule_id=payload.get("rule_id", ""),
            summary=payload.get("summary", ""),
            impact=payload.get("impact", ""),
            severity=payload.get("severity", "medium"),
            confidence_millionths=int(
                payload.get("confidence_millionths") or 0
            ),
            security_property=payload.get("security_property"),
            threat_path=payload.get("threat_path"),
            evidence=payload.get("evidence") or SecurityEvidence(),
            symbols=tuple(payload.get("symbols") or ()),
            interfaces=tuple(payload.get("interfaces") or ()),
            repositories=tuple(payload.get("repositories") or ()),
            source_node_id=payload.get("source_node_id", ""),
            sink_node_id=payload.get("sink_node_id", ""),
            missing_requirements=tuple(
                payload.get("missing_requirements") or ()
            ),
            assumptions=tuple(payload.get("assumptions") or ()),
            remediation_hints=tuple(payload.get("remediation_hints") or ()),
            tree_id=payload.get("tree_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            analyzer_version=payload.get("analyzer_version", ANALYZER_VERSION),
            root_cause_family=payload.get("root_cause_family", ""),
            seed_label=payload.get("seed_label", ""),
        )
        _check_identity(
            payload,
            result.finding_id,
            names=("finding_id", "content_id", "cid"),
            artifact_name="security finding",
        )
        return result


@dataclass(frozen=True)
class SecurityAnalysisReport(_SecurityContract):
    """Deterministic report over a bounded analysis pass."""

    SCHEMA: ClassVar[str] = SECURITY_ANALYSIS_REPORT_SCHEMA

    findings: tuple[SecurityFinding, ...] = ()
    verdict: AnalysisVerdict = AnalysisVerdict.EMPTY
    node_count: int = 0
    edge_count: int = 0
    property_count: int = 0
    paths_explored: int = 0
    truncated: bool = False
    truncation_reasons: tuple[str, ...] = ()
    analyzer_version: str = ANALYZER_VERSION
    tree_id: str = ""
    policy_revision: str = ""
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "findings",
            _records(
                self.findings,
                SecurityFinding,
                field_name="findings",
                maximum=MAX_FINDINGS,
            ),
        )
        # Deterministic order: rule_id, classification, finding_id.
        ordered = tuple(
            sorted(
                self.findings,
                key=lambda f: (
                    f.rule_id,
                    f.classification.value,
                    f.finding_id,
                ),
            )
        )
        object.__setattr__(self, "findings", ordered)
        object.__setattr__(
            self,
            "verdict",
            _enum(self.verdict, AnalysisVerdict, field_name="verdict"),
        )
        for name in ("node_count", "edge_count", "property_count", "paths_explored"):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), field_name=name, minimum=0),
            )
        object.__setattr__(
            self,
            "truncated",
            _boolean(self.truncated, field_name="truncated"),
        )
        object.__setattr__(
            self,
            "truncation_reasons",
            _strings(
                self.truncation_reasons,
                field_name="truncation_reasons",
                unique=True,
                sort=True,
            ),
        )
        for name in ("analyzer_version", "tree_id", "policy_revision"):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name) or "",
                    field_name=name,
                    required=False,
                ),
            )
        if not self.analyzer_version:
            object.__setattr__(self, "analyzer_version", ANALYZER_VERSION)
        object.__setattr__(
            self,
            "assumptions",
            _strings(
                self.assumptions,
                field_name="assumptions",
                unique=True,
                sort=True,
            ),
        )
        _bounded(self, artifact_name="security analysis report")

    @property
    def report_id(self) -> str:
        return self.content_id

    @property
    def vulnerabilities(self) -> tuple[SecurityFinding, ...]:
        return tuple(f for f in self.findings if f.is_vulnerability)

    @property
    def by_classification(
        self,
    ) -> Mapping[FindingClassification, tuple[SecurityFinding, ...]]:
        buckets: dict[FindingClassification, list[SecurityFinding]] = {
            c: [] for c in FindingClassification
        }
        for finding in self.findings:
            buckets[finding.classification].append(finding)
        return MappingProxyType(
            {k: tuple(v) for k, v in buckets.items()}
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "findings": tuple(f.to_dict() for f in self.findings),
            "verdict": self.verdict.value,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "property_count": self.property_count,
            "paths_explored": self.paths_explored,
            "truncated": self.truncated,
            "truncation_reasons": self.truncation_reasons,
            "analyzer_version": self.analyzer_version,
            "tree_id": self.tree_id,
            "policy_revision": self.policy_revision,
            "assumptions": self.assumptions,
            "goal_id": GOAL_ID,
            "is_completion_evidence": SECURITY_ANALYSIS_IS_COMPLETION_EVIDENCE,
            "authorizes_repair": SECURITY_ANALYSIS_AUTHORIZES_REPAIR,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "report_id": self.report_id,
            "content_id": self.content_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SecurityAnalysisReport":
        _check_header(payload, cls.SCHEMA)
        _reject_body_keys(payload, field_name="security analysis report")
        _reject_unknown(
            payload,
            _header_fields()
            | {
                "findings",
                "verdict",
                "node_count",
                "edge_count",
                "property_count",
                "paths_explored",
                "truncated",
                "truncation_reasons",
                "analyzer_version",
                "tree_id",
                "policy_revision",
                "assumptions",
                "goal_id",
                "is_completion_evidence",
                "authorizes_repair",
                "report_id",
            },
            artifact_name="security analysis report",
        )
        result = cls(
            findings=tuple(payload.get("findings") or ()),
            verdict=payload.get("verdict", AnalysisVerdict.EMPTY),
            node_count=int(payload.get("node_count") or 0),
            edge_count=int(payload.get("edge_count") or 0),
            property_count=int(payload.get("property_count") or 0),
            paths_explored=int(payload.get("paths_explored") or 0),
            truncated=bool(payload.get("truncated", False)),
            truncation_reasons=tuple(payload.get("truncation_reasons") or ()),
            analyzer_version=payload.get("analyzer_version", ANALYZER_VERSION),
            tree_id=payload.get("tree_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            assumptions=tuple(payload.get("assumptions") or ()),
        )
        _check_identity(
            payload,
            result.report_id,
            names=("report_id", "content_id", "cid"),
            artifact_name="security analysis report",
        )
        return result


# ---------------------------------------------------------------------------
# Flow graph + bounded path search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecurityAnalysisConfig:
    """Bounds for interprocedural analysis."""

    max_hops: int = DEFAULT_MAX_HOPS
    max_findings: int = DEFAULT_MAX_FINDINGS
    max_paths_per_rule: int = DEFAULT_MAX_PATHS_PER_RULE
    include_sanitized_as_false_positive: bool = True
    tree_id: str = ""
    policy_revision: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_hops",
            _integer(
                self.max_hops,
                field_name="max_hops",
                minimum=1,
                maximum=MAX_PATH_HOPS,
            ),
        )
        object.__setattr__(
            self,
            "max_findings",
            _integer(
                self.max_findings,
                field_name="max_findings",
                minimum=1,
                maximum=MAX_FINDINGS,
            ),
        )
        object.__setattr__(
            self,
            "max_paths_per_rule",
            _integer(
                self.max_paths_per_rule,
                field_name="max_paths_per_rule",
                minimum=1,
                maximum=MAX_PATHS_PER_RULE,
            ),
        )
        object.__setattr__(
            self,
            "include_sanitized_as_false_positive",
            _boolean(
                self.include_sanitized_as_false_positive,
                field_name="include_sanitized_as_false_positive",
            ),
        )
        for name in ("tree_id", "policy_revision"):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name) or "",
                    field_name=name,
                    required=False,
                ),
            )


@dataclass
class FlowGraph:
    """In-memory adjacency for bounded path search."""

    nodes: dict[str, FlowNode]
    outgoing: dict[str, list[FlowEdge]]
    edges_by_id: dict[str, FlowEdge]

    @classmethod
    def from_parts(
        cls,
        nodes: Sequence[FlowNode],
        edges: Sequence[FlowEdge],
    ) -> "FlowGraph":
        if len(nodes) > MAX_GRAPH_NODES:
            raise SecurityContractBoundsError(
                f"nodes exceed {MAX_GRAPH_NODES}"
            )
        if len(edges) > MAX_GRAPH_EDGES:
            raise SecurityContractBoundsError(
                f"edges exceed {MAX_GRAPH_EDGES}"
            )
        node_map: dict[str, FlowNode] = {}
        for node in nodes:
            if node.node_id in node_map:
                raise SecurityContractAnalysisError(
                    f"duplicate node_id {node.node_id!r}"
                )
            node_map[node.node_id] = node
        outgoing: dict[str, list[FlowEdge]] = {nid: [] for nid in node_map}
        edges_by_id: dict[str, FlowEdge] = {}
        for edge in edges:
            if edge.edge_id in edges_by_id:
                raise SecurityContractAnalysisError(
                    f"duplicate edge_id {edge.edge_id!r}"
                )
            if edge.source_id not in node_map or edge.target_id not in node_map:
                raise SecurityContractAnalysisError(
                    f"edge {edge.edge_id!r} references unknown nodes"
                )
            edges_by_id[edge.edge_id] = edge
            outgoing[edge.source_id].append(edge)
        # Deterministic adjacency order.
        for nid in outgoing:
            outgoing[nid].sort(key=lambda e: (e.target_id, e.edge_id))
        return cls(nodes=node_map, outgoing=outgoing, edges_by_id=edges_by_id)


def _node_has_tags(node: FlowNode, tags: frozenset[str]) -> bool:
    return bool(tags.intersection(node.tags))


def _node_is_sanitizer(node: FlowNode, sanitizer_tags: frozenset[str]) -> bool:
    if node.role is FlowRole.SANITIZER:
        return True
    return _node_has_tags(node, sanitizer_tags)


def _path_id(node_ids: Sequence[str], edge_ids: Sequence[str]) -> str:
    return content_identity(
        {
            "schema": "security-threat-path-key@1",
            "node_ids": list(node_ids),
            "edge_ids": list(edge_ids),
        }
    )


def bounded_source_sink_paths(
    graph: FlowGraph,
    *,
    source_ids: Sequence[str],
    sink_ids: Sequence[str],
    sanitizer_tags: frozenset[str],
    max_hops: int,
    max_paths: int,
) -> tuple[list[ThreatPath], list[ThreatPath], int, bool]:
    """BFS source→sink paths.

    Returns ``(open_paths, sanitized_paths, explored, truncated)``.
    Paths that touch an unknown-dynamic edge are still returned with
    ``has_unknown_dynamic=True``.
    """

    sinks = set(sink_ids)
    open_paths: list[ThreatPath] = []
    sanitized_paths: list[ThreatPath] = []
    explored = 0
    truncated = False

    for source_id in sorted(source_ids):
        if source_id not in graph.nodes:
            continue
        # state: (node_id, node_path, edge_path, hops, unknown, sanitized)
        queue: deque[
            tuple[str, tuple[str, ...], tuple[str, ...], int, bool, bool]
        ] = deque()
        queue.append((source_id, (source_id,), (), 0, False, False))
        visited: set[tuple[str, int, bool, bool]] = set()

        while queue:
            if len(open_paths) + len(sanitized_paths) >= max_paths:
                truncated = True
                break
            node_id, npath, epath, hops, unknown, sanitized = queue.popleft()
            key = (node_id, hops, unknown, sanitized)
            if key in visited:
                continue
            visited.add(key)
            explored += 1

            if node_id in sinks and hops > 0:
                path = ThreatPath(
                    path_id=_path_id(npath, epath),
                    node_ids=npath,
                    origin=ThreatPathOrigin.REACHABLE,
                    edge_ids=epath,
                    has_unknown_dynamic=unknown,
                    hop_count=hops,
                )
                if sanitized:
                    sanitized_paths.append(path)
                else:
                    open_paths.append(path)
                continue

            if hops >= max_hops:
                continue

            for edge in graph.outgoing.get(node_id, ()):
                nxt = edge.target_id
                if nxt in npath:
                    continue  # simple paths only
                nxt_node = graph.nodes[nxt]
                nxt_sanitized = sanitized or _node_is_sanitizer(
                    nxt_node, sanitizer_tags
                )
                nxt_unknown = unknown or edge.is_unknown_dynamic
                queue.append(
                    (
                        nxt,
                        npath + (nxt,),
                        epath + (edge.edge_id,),
                        hops + 1,
                        nxt_unknown,
                        nxt_sanitized,
                    )
                )
        if truncated:
            break

    open_paths.sort(key=lambda p: (p.hop_count, p.path_id))
    sanitized_paths.sort(key=lambda p: (p.hop_count, p.path_id))
    return open_paths, sanitized_paths, explored, truncated


def _property_for_family(
    properties: Sequence[SecurityPropertyDeclaration],
    family: SecurityRuleFamily,
) -> SecurityPropertyDeclaration | None:
    matches = [p for p in properties if p.family is family]
    if not matches:
        return None
    # Prefer stable order by property_id.
    matches.sort(key=lambda p: p.property_id)
    return matches[0]


def _symbols_for_path(
    graph: FlowGraph, path: ThreatPath
) -> tuple[str, ...]:
    symbols: list[str] = []
    for nid in path.node_ids:
        node = graph.nodes.get(nid)
        if node is not None and node.symbol:
            symbols.append(node.symbol)
    return tuple(dict.fromkeys(symbols))


def _interfaces_for_path(
    graph: FlowGraph, path: ThreatPath
) -> tuple[str, ...]:
    interfaces: list[str] = []
    for nid in path.node_ids:
        node = graph.nodes.get(nid)
        if node is not None and node.interface:
            interfaces.append(node.interface)
    return tuple(sorted(set(interfaces)))


def _repositories_for_path(
    graph: FlowGraph, path: ThreatPath
) -> tuple[str, ...]:
    repos: list[str] = []
    for nid in path.node_ids:
        node = graph.nodes.get(nid)
        if node is not None and node.repository_id:
            repos.append(node.repository_id)
    return tuple(sorted(set(repos)))


def _build_finding(
    *,
    spec: SecurityRuleSpec,
    path: ThreatPath,
    graph: FlowGraph,
    prop: SecurityPropertyDeclaration | None,
    evidence: SecurityEvidence,
    sanitized: bool,
    config: SecurityAnalysisConfig,
    seed_label: str = "",
) -> SecurityFinding:
    impact = (
        prop.statement
        if prop is not None and prop.statement
        else spec.default_impact
    )
    # For vulnerability, impact must be non-empty (always true via default).
    if sanitized:
        classification = FindingClassification.CORRECTNESS_DRIFT
        missing: tuple[str, ...] = ("sanitized_path",)
        conf = 200_000
        severity = "info"
        summary = (
            f"Sanitized path for {spec.name}: not a vulnerability "
            f"({path.node_ids[0]} -> {path.node_ids[-1]})."
        )
        seed = seed_label or "false_positive"
    else:
        classification = classify_security_finding(
            security_property=prop,
            threat_path=path,
            impact=impact if prop is not None else "",
            evidence=evidence,
            sanitized=False,
            family_matched=True,
        )
        ok, missing = vulnerability_requirements_met(
            security_property=prop,
            threat_path=path,
            impact=impact if prop is not None else "",
            evidence=evidence,
        )
        if classification is FindingClassification.VULNERABILITY:
            conf = 900_000
            severity = spec.default_severity
            summary = (
                f"Vulnerability: {spec.name} on path "
                f"{path.node_ids[0]} -> {path.node_ids[-1]}."
            )
            seed = seed_label or "true_positive"
            # Ensure impact is the concrete one used for the gate.
            impact = impact or spec.default_impact
        elif classification is FindingClassification.UNKNOWN_DYNAMIC:
            conf = 300_000
            severity = "low"
            summary = (
                f"Unknown dynamic frontier for {spec.name} on path "
                f"{path.node_ids[0]} -> {path.node_ids[-1]}."
            )
            seed = seed_label or "unknown_dynamic"
            impact = impact if prop is not None else ""
        elif classification is FindingClassification.SUSPICION:
            conf = 500_000
            severity = "medium"
            summary = (
                f"Suspicion: {spec.name} incomplete requirements "
                f"({', '.join(missing) or 'partial'}) on "
                f"{path.node_ids[0]} -> {path.node_ids[-1]}."
            )
            seed = seed_label or "false_positive"
            if prop is None:
                impact = ""
        else:
            conf = 400_000
            severity = "low"
            summary = (
                f"Correctness drift: {spec.name} without declared "
                f"security property on {path.node_ids[0]} -> "
                f"{path.node_ids[-1]}."
            )
            seed = seed_label or "false_positive"
            impact = ""
            # Recompute missing for drift (no property).
            _, missing = vulnerability_requirements_met(
                security_property=None,
                threat_path=path,
                impact="",
                evidence=evidence,
            )

        # For true vulnerability, force impact to default if property
        # statement was empty (already handled).  Gate needs impact text.
        if classification is FindingClassification.VULNERABILITY and not impact:
            impact = spec.default_impact

        # Re-check: if we classified vulnerability, impact must be set.
        if classification is FindingClassification.VULNERABILITY:
            ok2, missing2 = vulnerability_requirements_met(
                security_property=prop,
                threat_path=path,
                impact=impact,
                evidence=evidence,
            )
            if not ok2:
                classification = FindingClassification.SUSPICION
                missing = missing2
                conf = 500_000
                severity = "medium"
                seed = seed_label or "false_positive"

    source_id = path.node_ids[0]
    sink_id = path.node_ids[-1]
    return SecurityFinding(
        family=spec.family,
        classification=classification,
        rule_id=spec.rule_id,
        summary=summary,
        impact=impact,
        severity=severity,
        confidence_millionths=conf,
        security_property=prop,
        threat_path=path,
        evidence=evidence,
        symbols=_symbols_for_path(graph, path),
        interfaces=_interfaces_for_path(graph, path),
        repositories=_repositories_for_path(graph, path),
        source_node_id=source_id,
        sink_node_id=sink_id,
        missing_requirements=missing,
        assumptions=(
            "bounded simple-path search",
            f"max_hops={config.max_hops}",
            "conservative symbolic edges only",
        ),
        remediation_hints=tuple(sorted(spec.sanitizer_tags)),
        tree_id=config.tree_id,
        policy_revision=config.policy_revision,
        analyzer_version=ANALYZER_VERSION,
        root_cause_family=spec.family.value,
        seed_label=seed,
    )


def analyze_security_contracts(
    *,
    nodes: Sequence[FlowNode | Mapping[str, Any]],
    edges: Sequence[FlowEdge | Mapping[str, Any]],
    properties: Sequence[
        SecurityPropertyDeclaration | Mapping[str, Any]
    ] = (),
    evidence_by_family: Mapping[str, SecurityEvidence | Mapping[str, Any]]
    | None = None,
    declared_paths: Sequence[ThreatPath | Mapping[str, Any]] = (),
    default_evidence: SecurityEvidence | Mapping[str, Any] | None = None,
    config: SecurityAnalysisConfig | None = None,
    families: Sequence[SecurityRuleFamily | str] | None = None,
) -> SecurityAnalysisReport:
    """Run bounded interprocedural security rules over a flow graph.

    Parameters
    ----------
    nodes, edges:
        Flow graph parts (objects or dicts).
    properties:
        Declared security properties.  Required for vulnerability labels.
    evidence_by_family:
        Optional map of family value → evidence refs.
    declared_paths:
        Optional operator-declared threat paths (origin=declared).
    default_evidence:
        Fallback evidence applied when a family has no specific entry.
    config:
        Analysis bounds.
    families:
        Optional subset of rule families to run (default: all).
    """

    cfg = config or SecurityAnalysisConfig()
    decoded_nodes = _records(
        list(nodes), FlowNode, field_name="nodes", maximum=MAX_GRAPH_NODES
    )
    decoded_edges = _records(
        list(edges), FlowEdge, field_name="edges", maximum=MAX_GRAPH_EDGES
    )
    decoded_props = _records(
        list(properties),
        SecurityPropertyDeclaration,
        field_name="properties",
        maximum=MAX_COLLECTION_ITEMS,
    )
    decoded_declared = _records(
        list(declared_paths),
        ThreatPath,
        field_name="declared_paths",
        maximum=MAX_COLLECTION_ITEMS,
    )

    if default_evidence is None:
        base_evidence = SecurityEvidence()
    else:
        decoded = _record(
            default_evidence,
            SecurityEvidence,
            field_name="default_evidence",
        )
        base_evidence = decoded if decoded is not None else SecurityEvidence()

    family_evidence: dict[str, SecurityEvidence] = {}
    if evidence_by_family:
        for key, value in evidence_by_family.items():
            decoded = _record(
                value,
                SecurityEvidence,
                field_name=f"evidence_by_family[{key}]",
            )
            if decoded is not None:
                family_evidence[str(key)] = decoded

    graph = FlowGraph.from_parts(decoded_nodes, decoded_edges)

    if families is None:
        specs = list(_RULE_SPECS)
    else:
        specs = [
            security_rule_spec(f) for f in families
        ]
        specs.sort(key=lambda s: s.rule_id)

    findings: list[SecurityFinding] = []
    paths_explored = 0
    truncated = False
    truncation_reasons: list[str] = []

    for spec in specs:
        if len(findings) >= cfg.max_findings:
            truncated = True
            truncation_reasons.append("max_findings")
            break

        source_ids = [
            n.node_id
            for n in graph.nodes.values()
            if n.role is FlowRole.SOURCE
            or _node_has_tags(n, spec.source_tags)
        ]
        sink_ids = [
            n.node_id
            for n in graph.nodes.values()
            if n.role is FlowRole.SINK
            or _node_has_tags(n, spec.sink_tags)
        ]
        if not source_ids or not sink_ids:
            continue

        open_paths, sanitized_paths, explored, path_trunc = (
            bounded_source_sink_paths(
                graph,
                source_ids=source_ids,
                sink_ids=sink_ids,
                sanitizer_tags=spec.sanitizer_tags,
                max_hops=cfg.max_hops,
                max_paths=cfg.max_paths_per_rule,
            )
        )
        paths_explored += explored
        if path_trunc:
            truncated = True
            truncation_reasons.append(f"max_paths:{spec.rule_id}")

        prop = _property_for_family(decoded_props, spec.family)
        evidence = family_evidence.get(spec.family.value, base_evidence)

        # Merge declared paths that match this family's source/sink tags.
        for dpath in decoded_declared:
            if dpath.node_ids[0] not in source_ids:
                continue
            if dpath.node_ids[-1] not in sink_ids:
                continue
            # Ensure origin is declared.
            if dpath.origin is not ThreatPathOrigin.DECLARED:
                dpath = ThreatPath(
                    path_id=dpath.path_id,
                    node_ids=dpath.node_ids,
                    origin=ThreatPathOrigin.DECLARED,
                    edge_ids=dpath.edge_ids,
                    has_unknown_dynamic=dpath.has_unknown_dynamic,
                    hop_count=dpath.hop_count,
                )
            open_paths.append(dpath)

        # Deduplicate paths by path_id.
        seen_path_ids: set[str] = set()
        unique_open: list[ThreatPath] = []
        for path in open_paths:
            if path.path_id in seen_path_ids:
                continue
            seen_path_ids.add(path.path_id)
            unique_open.append(path)
        unique_open.sort(key=lambda p: (p.hop_count, p.path_id))

        for path in unique_open:
            if len(findings) >= cfg.max_findings:
                truncated = True
                truncation_reasons.append("max_findings")
                break
            findings.append(
                _build_finding(
                    spec=spec,
                    path=path,
                    graph=graph,
                    prop=prop,
                    evidence=evidence,
                    sanitized=False,
                    config=cfg,
                )
            )

        if cfg.include_sanitized_as_false_positive:
            for path in sanitized_paths:
                if len(findings) >= cfg.max_findings:
                    truncated = True
                    truncation_reasons.append("max_findings")
                    break
                if path.path_id in seen_path_ids:
                    continue
                seen_path_ids.add(path.path_id)
                findings.append(
                    _build_finding(
                        spec=spec,
                        path=path,
                        graph=graph,
                        prop=prop,
                        evidence=evidence,
                        sanitized=True,
                        config=cfg,
                        seed_label="false_positive",
                    )
                )

    if not findings:
        verdict = AnalysisVerdict.EMPTY if not decoded_nodes else AnalysisVerdict.CLEAN
    elif truncated:
        verdict = AnalysisVerdict.BOUNDED
    else:
        verdict = AnalysisVerdict.FINDINGS

    return SecurityAnalysisReport(
        findings=tuple(findings),
        verdict=verdict,
        node_count=len(decoded_nodes),
        edge_count=len(decoded_edges),
        property_count=len(decoded_props),
        paths_explored=paths_explored,
        truncated=truncated,
        truncation_reasons=tuple(sorted(set(truncation_reasons))),
        analyzer_version=ANALYZER_VERSION,
        tree_id=cfg.tree_id,
        policy_revision=cfg.policy_revision,
        assumptions=(
            "no LLM vulnerability classification",
            "conservative simple-path BFS",
            "vulnerability requires property+path+impact+evidence",
        ),
    )


def build_security_finding(
    *,
    family: SecurityRuleFamily | str,
    classification: FindingClassification | str,
    summary: str,
    impact: str = "",
    evidence: SecurityEvidence | Mapping[str, Any] | None = None,
    security_property: SecurityPropertyDeclaration
    | Mapping[str, Any]
    | None = None,
    threat_path: ThreatPath | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> SecurityFinding:
    """Construct a validated security finding."""

    family_e = _enum(family, SecurityRuleFamily, field_name="family")
    spec = security_rule_spec(family_e)
    return SecurityFinding(
        family=family_e,
        classification=classification,
        rule_id=kwargs.pop("rule_id", spec.rule_id),
        summary=summary,
        impact=impact,
        severity=kwargs.pop("severity", spec.default_severity),
        evidence=evidence if evidence is not None else SecurityEvidence(),
        security_property=security_property,
        threat_path=threat_path,
        **kwargs,
    )


def make_flow_node(
    node_id: str,
    symbol: str,
    *,
    role: FlowRole | str = FlowRole.PASSTHROUGH,
    tags: Sequence[str] = (),
    **kwargs: Any,
) -> FlowNode:
    """Convenience constructor for tests and fixtures."""

    return FlowNode(
        node_id=node_id,
        symbol=symbol,
        role=role,
        tags=tuple(tags),
        **kwargs,
    )


def make_flow_edge(
    edge_id: str,
    source_id: str,
    target_id: str,
    *,
    resolution: EdgeResolution | str = EdgeResolution.RESOLVED,
    **kwargs: Any,
) -> FlowEdge:
    """Convenience constructor for tests and fixtures."""

    return FlowEdge(
        edge_id=edge_id,
        source_id=source_id,
        target_id=target_id,
        resolution=resolution,
        **kwargs,
    )


def make_security_property(
    property_id: str,
    family: SecurityRuleFamily | str,
    *,
    resource: str,
    statement: str,
    **kwargs: Any,
) -> SecurityPropertyDeclaration:
    """Convenience constructor for declared security properties."""

    return SecurityPropertyDeclaration(
        property_id=property_id,
        family=family,
        resource=resource,
        statement=statement,
        **kwargs,
    )


def make_evidence(*artifact_cids: str, **kwargs: Any) -> SecurityEvidence:
    """Convenience constructor for artifact-reference evidence."""

    return SecurityEvidence(
        artifact_cids=tuple(artifact_cids),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fixed-point security (IntentIR / code facts / hyperproperties)
# ---------------------------------------------------------------------------


FIXED_POINT_SECURITY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-fixed-point-receipt@1"
)
CODE_SECURITY_FACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-security-fact@1"
)


@dataclass(frozen=True)
class CodeSecurityFact:
    """Extracted code-side security fact used at fixed-point recheck."""

    fact_id: str
    path: str
    symbol: str
    kind: str
    tags: tuple[str, ...] = ()
    effect_id: str = ""
    evidence_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _text(self.fact_id, field_name="fact_id"))
        object.__setattr__(self, "path", _text(self.path, field_name="path"))
        object.__setattr__(self, "symbol", _text(self.symbol, field_name="symbol"))
        object.__setattr__(self, "kind", _text(self.kind, field_name="kind"))
        object.__setattr__(
            self, "tags", _strings(self.tags, field_name="tags", required=False)
        )
        object.__setattr__(
            self,
            "effect_id",
            _text(self.effect_id, field_name="effect_id", required=False),
        )
        object.__setattr__(
            self,
            "evidence_cid",
            _text(self.evidence_cid, field_name="evidence_cid", required=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODE_SECURITY_FACT_SCHEMA,
            "fact_id": self.fact_id,
            "path": self.path,
            "symbol": self.symbol,
            "kind": self.kind,
            "tags": list(self.tags),
            "effect_id": self.effect_id,
            "evidence_cid": self.evidence_cid,
        }


@dataclass(frozen=True)
class ForbiddenLogicCheckResult:
    """Dual-stream IntentIR/code forbidden-logic evaluation at fixed point."""

    passed: bool
    intent_effect_ids: tuple[str, ...]
    code_effect_ids: tuple[str, ...]
    forbidden_logic_ids: tuple[str, ...]
    uncovered_intent_ids: tuple[str, ...]
    uncovered_code_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "intent_effect_ids": list(self.intent_effect_ids),
            "code_effect_ids": list(self.code_effect_ids),
            "forbidden_logic_ids": list(self.forbidden_logic_ids),
            "uncovered_intent_ids": list(self.uncovered_intent_ids),
            "uncovered_code_ids": list(self.uncovered_code_ids),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class FixedPointSecurityReceipt:
    """Sealed security stage receipt for doctor live fixed-point."""

    candidate_tree_id: str
    code_facts: tuple[CodeSecurityFact, ...]
    forbidden: ForbiddenLogicCheckResult
    analysis_report: SecurityAnalysisReport | None
    hyperproperty_receipt_ids: tuple[str, ...]
    failed_hyperproperty_ids: tuple[str, ...]
    required_hyperproperty_ids: tuple[str, ...]
    all_passed: bool
    reason_codes: tuple[str, ...]
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_tree_id",
            _text(self.candidate_tree_id, field_name="candidate_tree_id"),
        )
        if not isinstance(self.code_facts, tuple):
            object.__setattr__(self, "code_facts", tuple(self.code_facts or ()))
        if not isinstance(self.forbidden, ForbiddenLogicCheckResult):
            raise SecurityContractAnalysisError(
                "forbidden must be ForbiddenLogicCheckResult"
            )
        if self.analysis_report is not None and not isinstance(
            self.analysis_report, SecurityAnalysisReport
        ):
            raise SecurityContractAnalysisError(
                "analysis_report must be SecurityAnalysisReport"
            )
        object.__setattr__(
            self,
            "hyperproperty_receipt_ids",
            _strings(
                self.hyperproperty_receipt_ids,
                field_name="hyperproperty_receipt_ids",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "failed_hyperproperty_ids",
            _strings(
                self.failed_hyperproperty_ids,
                field_name="failed_hyperproperty_ids",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "required_hyperproperty_ids",
            _strings(
                self.required_hyperproperty_ids,
                field_name="required_hyperproperty_ids",
                required=False,
            ),
        )
        object.__setattr__(self, "all_passed", _boolean(self.all_passed, field_name="all_passed"))
        object.__setattr__(
            self,
            "reason_codes",
            _strings(self.reason_codes, field_name="reason_codes", required=False),
        )
        if self.all_passed and (
            not self.forbidden.passed
            or self.failed_hyperproperty_ids
            or (
                self.analysis_report is not None
                and any(f.is_vulnerability for f in self.analysis_report.findings)
            )
        ):
            raise SecurityContractAnalysisError(
                "all_passed fixed-point security forbids open failures"
            )
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(
            self,
            "receipt_id",
            rid or content_identity(self.to_dict(include_receipt_id=False)),
        )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": FIXED_POINT_SECURITY_RECEIPT_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "code_facts": [fact.to_dict() for fact in self.code_facts],
            "forbidden": self.forbidden.to_dict(),
            "analysis_report_id": (
                self.analysis_report.report_id
                if self.analysis_report is not None
                else ""
            ),
            "vulnerability_ids": (
                [f.finding_id for f in self.analysis_report.vulnerabilities]
                if self.analysis_report is not None
                else []
            ),
            "hyperproperty_receipt_ids": list(self.hyperproperty_receipt_ids),
            "failed_hyperproperty_ids": list(self.failed_hyperproperty_ids),
            "required_hyperproperty_ids": list(self.required_hyperproperty_ids),
            "all_passed": self.all_passed,
            "reason_codes": list(self.reason_codes),
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


def extract_code_security_facts(
    *,
    paths: Sequence[str],
    symbols: Sequence[str] = (),
    tags_by_path: Mapping[str, Sequence[str]] | None = None,
    effects_by_path: Mapping[str, Sequence[str]] | None = None,
    kind: str = "code_effect",
    tree_id: str = "",
) -> tuple[CodeSecurityFact, ...]:
    """Extract deterministic code security facts for fixed-point recheck.

    Facts are identity-bound to path/symbol/tags and never include bodies.
    """

    tags_map = {
        str(k): tuple(str(t) for t in v)
        for k, v in (tags_by_path or {}).items()
    }
    effects_map = {
        str(k): tuple(str(e) for e in v)
        for k, v in (effects_by_path or {}).items()
    }
    facts: list[CodeSecurityFact] = []
    symbol_list = [str(s) for s in symbols]
    for index, path in enumerate(paths):
        path_s = _text(path, field_name=f"paths[{index}]")
        symbol = (
            symbol_list[index]
            if index < len(symbol_list) and symbol_list[index]
            else path_s.replace("/", ".").removesuffix(".py")
        )
        tags = tags_map.get(path_s, ())
        effect_ids = effects_map.get(path_s, ())
        if not effect_ids:
            effect_ids = (f"effect:{path_s}",)
        for effect_id in effect_ids:
            fact_id = content_identity(
                {
                    "schema": CODE_SECURITY_FACT_SCHEMA,
                    "tree_id": tree_id,
                    "path": path_s,
                    "symbol": symbol,
                    "kind": kind,
                    "effect_id": effect_id,
                    "tags": list(tags),
                }
            )
            facts.append(
                CodeSecurityFact(
                    fact_id=fact_id,
                    path=path_s,
                    symbol=symbol,
                    kind=kind,
                    tags=tags,
                    effect_id=effect_id,
                    evidence_cid=fact_id,
                )
            )
    facts.sort(key=lambda f: (f.path, f.symbol, f.effect_id, f.fact_id))
    return tuple(facts)


def check_intent_code_forbidden_logic(
    *,
    intent_effects: Sequence[str | Mapping[str, Any]],
    code_effects: Sequence[str | Mapping[str, Any]],
    forbidden_effect_ids: Sequence[str] = (),
    covered_effect_ids: Sequence[str] = (),
) -> ForbiddenLogicCheckResult:
    """Check forbidden logic against *both* intent and code effect streams.

    Dual-stream coverage is mandatory: a single-stream evaluation cannot pass.
    Any effect present in either stream that is listed as forbidden fails closed.
    Every effect must also be covered by a security/authorization name when
    ``covered_effect_ids`` is supplied; otherwise coverage is treated as
    identity-of-union (each effect covers itself).
    """

    def _normalize(items: Sequence[str | Mapping[str, Any]]) -> tuple[str, ...]:
        out: list[str] = []
        for item in items:
            if isinstance(item, Mapping):
                # Prefer explicit effect identity, else canonical projection.
                if "effect_id" in item and item["effect_id"]:
                    out.append(str(item["effect_id"]))
                else:
                    out.append(
                        content_identity(
                            {str(k): item[k] for k in sorted(item.keys())}
                        )
                    )
            else:
                text = str(item).strip()
                if text:
                    out.append(text)
        return tuple(sorted(set(out)))

    intent_ids = _normalize(intent_effects)
    code_ids = _normalize(code_effects)
    forbidden = tuple(sorted({str(x).strip() for x in forbidden_effect_ids if str(x).strip()}))
    covered = tuple(sorted({str(x).strip() for x in covered_effect_ids if str(x).strip()}))

    reasons: list[str] = []
    if not intent_ids or not code_ids:
        reasons.append("intent_code_stream_gap")
        return ForbiddenLogicCheckResult(
            passed=False,
            intent_effect_ids=intent_ids,
            code_effect_ids=code_ids,
            forbidden_logic_ids=(),
            uncovered_intent_ids=intent_ids,
            uncovered_code_ids=code_ids,
            reason_codes=tuple(reasons),
        )

    matched_forbidden = tuple(
        sorted(
            {
                effect
                for effect in forbidden
                if effect in intent_ids or effect in code_ids
            }
        )
    )
    if matched_forbidden:
        reasons.append("forbidden_logic_matched")

    # Coverage: if explicit covered set provided, every effect must be named.
    # Otherwise each stream must be non-empty (already checked) and no
    # forbidden hits.
    if covered:
        uncovered_intent = tuple(sorted(set(intent_ids) - set(covered)))
        uncovered_code = tuple(sorted(set(code_ids) - set(covered)))
        if uncovered_intent or uncovered_code:
            reasons.append("security_stream_gap")
    else:
        uncovered_intent = ()
        uncovered_code = ()

    passed = not reasons and not matched_forbidden
    return ForbiddenLogicCheckResult(
        passed=passed,
        intent_effect_ids=intent_ids,
        code_effect_ids=code_ids,
        forbidden_logic_ids=matched_forbidden,
        uncovered_intent_ids=uncovered_intent,
        uncovered_code_ids=uncovered_code,
        reason_codes=tuple(sorted(set(reasons))),
    )


def check_required_security_hyperproperties(
    *,
    required_ids: Sequence[str],
    held_receipt_ids: Sequence[str] = (),
    failed_ids: Sequence[str] = (),
    unavailable_ids: Sequence[str] = (),
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Require every mandatory hyperproperty to hold with a sealed receipt.

    Returns ``(passed, held_receipt_ids, failed_or_missing_ids)``.
    Unavailable required engines fail closed (listed in failed_or_missing).
    """

    required = tuple(sorted({str(x).strip() for x in required_ids if str(x).strip()}))
    held = tuple(sorted({str(x).strip() for x in held_receipt_ids if str(x).strip()}))
    failed = {str(x).strip() for x in failed_ids if str(x).strip()}
    unavailable = {str(x).strip() for x in unavailable_ids if str(x).strip()}

    missing: set[str] = set()
    for req in required:
        # Receipts may be keyed as hyperproperty:<id> or bare id.
        held_match = any(
            h == req or h.endswith(f":{req}") or h == f"hyperproperty:{req}"
            for h in held
        )
        if req in failed or req in unavailable or not held_match:
            missing.add(req)
    passed = not missing and not (failed & set(required))
    return passed, held, tuple(sorted(missing))


def evaluate_fixed_point_security(
    *,
    candidate_tree_id: str,
    code_facts: Sequence[CodeSecurityFact] = (),
    intent_effects: Sequence[str | Mapping[str, Any]] = (),
    code_effects: Sequence[str | Mapping[str, Any]] = (),
    forbidden_effect_ids: Sequence[str] = (),
    covered_effect_ids: Sequence[str] = (),
    flow_nodes: Sequence[FlowNode | Mapping[str, Any]] = (),
    flow_edges: Sequence[FlowEdge | Mapping[str, Any]] = (),
    properties: Sequence[
        SecurityPropertyDeclaration | Mapping[str, Any]
    ] = (),
    evidence_by_family: Mapping[str, SecurityEvidence | Mapping[str, Any]]
    | None = None,
    default_evidence: SecurityEvidence | Mapping[str, Any] | None = None,
    config: SecurityAnalysisConfig | None = None,
    required_hyperproperty_ids: Sequence[str] = (),
    held_hyperproperty_receipt_ids: Sequence[str] = (),
    failed_hyperproperty_ids: Sequence[str] = (),
    unavailable_hyperproperty_ids: Sequence[str] = (),
    run_flow_analysis: bool = True,
) -> FixedPointSecurityReceipt:
    """Aggregate fixed-point security: facts, forbidden dual-stream, flow, hyperprops.

    Prebuilt boolean/mapping claims are never accepted: callers must supply
    concrete fact/effect identities and sealed hyperproperty receipts.
    """

    tree = _text(candidate_tree_id, field_name="candidate_tree_id")
    facts = tuple(code_facts)
    if not facts and code_effects:
        # Derive minimal facts from code effect stream when paths are absent.
        facts = tuple(
            CodeSecurityFact(
                fact_id=content_identity(
                    {"schema": CODE_SECURITY_FACT_SCHEMA, "effect": effect}
                ),
                path=f"effect/{effect}",
                symbol=str(effect),
                kind="code_effect",
                effect_id=str(effect),
                evidence_cid=content_identity({"effect": effect}),
            )
            for effect in (
                str(e.get("effect_id") if isinstance(e, Mapping) else e)
                for e in code_effects
            )
            if str(e).strip()
        )

    # Prefer explicit code_effects; fall back to extracted fact effect ids.
    effective_code_effects: Sequence[str | Mapping[str, Any]] = code_effects or tuple(
        f.effect_id for f in facts if f.effect_id
    )
    effective_intent = intent_effects or effective_code_effects

    covered = covered_effect_ids or tuple(
        str(e.get("effect_id") if isinstance(e, Mapping) else e)
        for e in (*effective_intent, *effective_code_effects)
    )

    forbidden = check_intent_code_forbidden_logic(
        intent_effects=effective_intent,
        code_effects=effective_code_effects,
        forbidden_effect_ids=forbidden_effect_ids,
        covered_effect_ids=covered,
    )

    report: SecurityAnalysisReport | None = None
    if run_flow_analysis and (flow_nodes or flow_edges):
        cfg = config or SecurityAnalysisConfig(tree_id=tree)
        if not cfg.tree_id:
            cfg = SecurityAnalysisConfig(
                max_hops=cfg.max_hops,
                max_findings=cfg.max_findings,
                max_paths_per_rule=cfg.max_paths_per_rule,
                include_sanitized_as_false_positive=cfg.include_sanitized_as_false_positive,
                tree_id=tree,
                policy_revision=cfg.policy_revision,
            )
        report = analyze_security_contracts(
            nodes=flow_nodes,
            edges=flow_edges,
            properties=properties,
            evidence_by_family=evidence_by_family,
            default_evidence=default_evidence,
            config=cfg,
        )

    hyper_ok, held, missing_hyper = check_required_security_hyperproperties(
        required_ids=required_hyperproperty_ids,
        held_receipt_ids=held_hyperproperty_receipt_ids,
        failed_ids=failed_hyperproperty_ids,
        unavailable_ids=unavailable_hyperproperty_ids,
    )

    reasons: list[str] = []
    if not forbidden.passed:
        reasons.extend(forbidden.reason_codes or ("forbidden_logic_failed",))
    vulnerabilities: tuple[str, ...] = ()
    if report is not None:
        vulnerabilities = tuple(f.finding_id for f in report.vulnerabilities)
        if vulnerabilities:
            reasons.append("security_vulnerability_open")
        if report.truncated:
            reasons.append("security_analysis_truncated")
    if not hyper_ok:
        reasons.append("required_hyperproperty_failed")
        reasons.extend(f"hyperproperty_missing:{hid}" for hid in missing_hyper)

    all_passed = not reasons
    return FixedPointSecurityReceipt(
        candidate_tree_id=tree,
        code_facts=facts,
        forbidden=forbidden,
        analysis_report=report,
        hyperproperty_receipt_ids=held,
        failed_hyperproperty_ids=missing_hyper,
        required_hyperproperty_ids=tuple(
            sorted({str(x).strip() for x in required_hyperproperty_ids if str(x).strip()})
        ),
        all_passed=all_passed,
        reason_codes=tuple(sorted(set(reasons))),
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "ANALYZER_VERSION",
    "AnalysisVerdict",
    "CODE_SECURITY_FACT_SCHEMA",
    "CodeSecurityFact",
    "DEFAULT_MAX_FINDINGS",
    "DEFAULT_MAX_HOPS",
    "DEFAULT_MAX_PATHS_PER_RULE",
    "EdgeResolution",
    "FIXED_POINT_SECURITY_RECEIPT_SCHEMA",
    "FindingClassification",
    "FixedPointSecurityReceipt",
    "FlowEdge",
    "FlowGraph",
    "FlowNode",
    "FlowRole",
    "ForbiddenBodyError",
    "ForbiddenLogicCheckResult",
    "ForgedSecurityIdentityError",
    "GOAL_ID",
    "MAX_FINDINGS",
    "MAX_PATH_HOPS",
    "SCHEMA_VERSION",
    "SECURITY_ANALYSIS_AUTHORIZES_REPAIR",
    "SECURITY_ANALYSIS_IS_COMPLETION_EVIDENCE",
    "SECURITY_ANALYSIS_REPORT_SCHEMA",
    "SECURITY_CONTRACT_ANALYSIS_VERSION",
    "SECURITY_EVIDENCE_SCHEMA",
    "SECURITY_FINDING_SCHEMA",
    "SECURITY_PROPERTY_SCHEMA",
    "SecurityAnalysisConfig",
    "SecurityAnalysisReport",
    "SecurityContractAnalysisError",
    "SecurityContractBoundsError",
    "SecurityEvidence",
    "SecurityFinding",
    "SecurityPropertyDeclaration",
    "SecurityRuleFamily",
    "SecurityRuleSpec",
    "THREAT_PATH_SCHEMA",
    "ThreatPath",
    "ThreatPathOrigin",
    "analyze_security_contracts",
    "bounded_source_sink_paths",
    "build_security_finding",
    "check_intent_code_forbidden_logic",
    "check_required_security_hyperproperties",
    "classify_security_finding",
    "evaluate_fixed_point_security",
    "extract_code_security_facts",
    "make_evidence",
    "make_flow_edge",
    "make_flow_node",
    "make_security_property",
    "security_rule_families",
    "security_rule_spec",
    "security_rule_specs",
    "vulnerability_requirements_met",
]
