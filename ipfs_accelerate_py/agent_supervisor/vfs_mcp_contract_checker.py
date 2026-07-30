"""End-to-end VFS manifest / SDK / MCP / MCP++ parity checking (VFS-028).

Compares the closed surface layers that must agree for a VFS tool to be
invokable through MCP and MCP++:

```text
Python signature
  <-> registered tools
  <-> tools/list schemas
  <-> generated JSON manifests / TypeScript SDKs
  <-> SwissKnife connector calls
  <-> transport profiles
  <-> result / error mappings
  <-> capability / degradation claims
  <-> real implementation targets
```

Consumption policy (conflict-safe):

* Consume MCP++ resolver inventories, resolution receipts, runtime witnesses,
  and the canonical VFS contract pack.
* Never regenerate package manifests or promote observations into expectations.
* Same-named text without a **resolved call path** is insufficient for
  ``proved_parity``; emit an explicit ``missing_resolved_call_path`` finding.

Findings cover stale generated artifacts, missing registrations, extra
unreachable tools, wrong aliases/schema/errors, direct local bypass,
mock/fallback dispatch, and ambiguous paths. Each finding carries a minimal
witness (surface pair + values + optional path/runtime evidence refs).

This module is not completion evidence and does not authorize repairs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .mcplusplus_contract_resolver import (
    PATH_STAGE_ORDER,
    ArtifactRole,
    CallPathClaim,
    DriftKind,
    InventoryArtifact,
    MCPlusPlusCallPath,
    MCPlusPlusContractResolver,
    MCPlusPlusInventory,
    MCPlusPlusResolutionResult,
    ManifestDriftWitness,
    PathVerdict,
    ReasonCode,
    TransportKind,
    classify_non_invocation,
    make_artifact,
    normalize_tool_name,
    schema_fingerprint,
    tool_name_aliases,
)
from .mcplusplus_runtime_witness import (
    ImplementationKind,
    RuntimeWitnessReceipt,
    WitnessOutcome,
)
from .program_graph import ResolverStatus
from .proof.formal_verification_contracts import content_identity
from .vfs_contract_pack import (
    OperationSupport,
    PublicSurface,
    VfsContractPack,
    VfsInvariantKind,
    VfsOperation,
    canonical_vfs_contract_pack,
)


# ---------------------------------------------------------------------------
# Schema / evidence identities
# ---------------------------------------------------------------------------

VFS_MCP_CONTRACT_CHECKER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vfs-mcp-contract-checker@1"
)
VFS_MCP_PARITY_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vfs-mcp-parity-report@1"
)
VFS_MCP_PARITY_FINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vfs-mcp-parity-finding@1"
)
VFS_MCP_PARITY_WITNESS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vfs-mcp-parity-witness@1"
)
VFS_MCP_SURFACE_VIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vfs-mcp-surface-view@1"
)
VFS_MCP_TOOL_PARITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vfs-mcp-tool-parity@1"
)

EVIDENCE_VFS_MCP_PARITY: Final[str] = "vfs/mcp-manifest-sdk-parity@1"
EVIDENCE_VFS_MCP_CALL_PATH: Final[str] = "vfs/mcplusplus-call-path@1"
EVIDENCE_MANIFEST_PARITY: Final[str] = "vfs/mcplusplus-manifest-parity@1"
EVIDENCE_RUNTIME_WITNESS: Final[str] = "vfs/mcplusplus-runtime-witness@1"

CHECKER_VERSION: Final[str] = "vfs-mcp-contract-checker@1"
CHECKER_PRODUCER: Final[str] = "vfs-mcp-contract-checker@1"
CONTRACT_VERSION: Final[int] = 1
GOAL_ID: Final[str] = "VFS-028"

DEFAULT_MAX_TOOLS: Final[int] = 10_000
DEFAULT_MAX_FINDINGS: Final[int] = 50_000
DEFAULT_MAX_WITNESSES: Final[int] = 50_000
DEFAULT_MAX_LABEL_BYTES: Final[int] = 4_096
DEFAULT_MAX_NOTES_BYTES: Final[int] = 8_192
DEFAULT_MAX_SCHEMA_BYTES: Final[int] = 262_144

# Authority: comparison only.
CHECKER_IS_COMPLETION_EVIDENCE: Final[bool] = False
CHECKER_IS_CORRECTNESS_EVIDENCE: Final[bool] = False
CHECKER_AUTHORIZES_REPAIR: Final[bool] = False

# Stages required for a *proved* invocation path (excludes optional caller).
REQUIRED_PROVED_STAGES: Final[tuple[str, ...]] = (
    "connector",
    "profile_transport",
    "tools_list",
    "tools_call",
    "server_registry",
    "adapter",
    "package_implementation",
    "result_error_mapping",
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class VfsMcpCheckerError(ValueError):
    """Malformed or unsafe VFS/MCP parity checker input."""


class VfsMcpCheckerBoundsError(VfsMcpCheckerError):
    """A compact parity record exceeded an explicit bound."""


class ParitySurface(str, Enum):
    """Closed set of layers compared for VFS MCP / MCP++ parity."""

    PYTHON_SIGNATURE = "python_signature"
    REGISTRATION = "registration"
    TOOLS_LIST = "tools_list"
    JSON_MANIFEST = "json_manifest"
    TYPESCRIPT_SDK = "typescript_sdk"
    SWISSKNIFE_CONNECTOR = "swissknife_connector"
    TRANSPORT_PROFILE = "transport_profile"
    RESULT_ERROR_MAP = "result_error_map"
    CAPABILITY_DEGRADATION = "capability_degradation"
    IMPLEMENTATION_TARGET = "implementation_target"
    RUNTIME_WITNESS = "runtime_witness"
    CONTRACT_PACK = "contract_pack"


class ParityFindingKind(str, Enum):
    """Closed finding kinds required by VFS-028 acceptance."""

    STALE_GENERATED_ARTIFACT = "stale_generated_artifact"
    MISSING_REGISTRATION = "missing_registration"
    EXTRA_UNREACHABLE_TOOL = "extra_unreachable_tool"
    WRONG_ALIAS = "wrong_alias"
    SCHEMA_MISMATCH = "schema_mismatch"
    ERROR_MAP_MISMATCH = "error_map_mismatch"
    RESULT_MAP_MISMATCH = "result_map_mismatch"
    DIRECT_LOCAL_BYPASS = "direct_local_bypass"
    MOCK_FALLBACK_DISPATCH = "mock_fallback_dispatch"
    AMBIGUOUS_PATH = "ambiguous_path"
    MISSING_RESOLVED_CALL_PATH = "missing_resolved_call_path"
    TRANSPORT_MISMATCH = "transport_mismatch"
    PROFILE_MISMATCH = "profile_mismatch"
    CAPABILITY_CLAIM_MISMATCH = "capability_claim_mismatch"
    DEGRADATION_CLAIM_MISMATCH = "degradation_claim_mismatch"
    NAME_MISMATCH = "name_mismatch"
    VERSION_MISMATCH = "version_mismatch"
    PYTHON_SIGNATURE_MISMATCH = "python_signature_mismatch"
    IMPLEMENTATION_TARGET_MISMATCH = "implementation_target_mismatch"
    RUNTIME_MOCK_AUTHORITY = "runtime_mock_authority"
    CONTRACT_PACK_GAP = "contract_pack_gap"


class ParitySeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ToolParityVerdict(str, Enum):
    """Per-tool overall verdict.

    ``proved_parity`` requires a resolved (proved) call path *and* zero drift
    findings. Same-named text without a resolved path yields
    ``insufficient_path``, never ``proved_parity``.
    """

    PROVED_PARITY = "proved_parity"
    WITNESSED_DRIFT = "witnessed_drift"
    INSUFFICIENT_PATH = "insufficient_path"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"
    UNKNOWN = "unknown"
    EXTERNAL = "external"


class ReportVerdict(str, Enum):
    """Aggregate report verdict (fail-closed)."""

    ALL_PROVED = "all_proved"
    HAS_DRIFT = "has_drift"
    HAS_INSUFFICIENT_PATH = "has_insufficient_path"
    HAS_AMBIGUOUS = "has_ambiguous"
    EMPTY = "empty"
    UNKNOWN = "unknown"


# Map resolver drift kinds onto parity finding kinds where applicable.
_DRIFT_TO_FINDING: Mapping[DriftKind, ParityFindingKind] = MappingProxyType(
    {
        DriftKind.NAME_MISMATCH: ParityFindingKind.NAME_MISMATCH,
        DriftKind.SCHEMA_MISMATCH: ParityFindingKind.SCHEMA_MISMATCH,
        DriftKind.VERSION_MISMATCH: ParityFindingKind.VERSION_MISMATCH,
        DriftKind.PROFILE_MISMATCH: ParityFindingKind.PROFILE_MISMATCH,
        DriftKind.ALIAS_MISMATCH: ParityFindingKind.WRONG_ALIAS,
        DriftKind.ERROR_MAP_MISMATCH: ParityFindingKind.ERROR_MAP_MISMATCH,
        DriftKind.RESULT_MAP_MISMATCH: ParityFindingKind.RESULT_MAP_MISMATCH,
        DriftKind.MISSING_REGISTRATION: ParityFindingKind.MISSING_REGISTRATION,
        DriftKind.EXTRA_UNREACHABLE: ParityFindingKind.EXTRA_UNREACHABLE_TOOL,
        DriftKind.STALE_MANIFEST: ParityFindingKind.STALE_GENERATED_ARTIFACT,
        DriftKind.COPIED_WITHOUT_BINDING: ParityFindingKind.MISSING_RESOLVED_CALL_PATH,
        DriftKind.TRANSPORT_MISMATCH: ParityFindingKind.TRANSPORT_MISMATCH,
        DriftKind.LANGUAGE_NAME_MISMATCH: ParityFindingKind.NAME_MISMATCH,
    }
)

_FINDING_SEVERITY: Mapping[ParityFindingKind, ParitySeverity] = MappingProxyType(
    {
        ParityFindingKind.STALE_GENERATED_ARTIFACT: ParitySeverity.ERROR,
        ParityFindingKind.MISSING_REGISTRATION: ParitySeverity.ERROR,
        ParityFindingKind.EXTRA_UNREACHABLE_TOOL: ParitySeverity.ERROR,
        ParityFindingKind.WRONG_ALIAS: ParitySeverity.ERROR,
        ParityFindingKind.SCHEMA_MISMATCH: ParitySeverity.ERROR,
        ParityFindingKind.ERROR_MAP_MISMATCH: ParitySeverity.ERROR,
        ParityFindingKind.RESULT_MAP_MISMATCH: ParitySeverity.ERROR,
        ParityFindingKind.DIRECT_LOCAL_BYPASS: ParitySeverity.ERROR,
        ParityFindingKind.MOCK_FALLBACK_DISPATCH: ParitySeverity.ERROR,
        ParityFindingKind.AMBIGUOUS_PATH: ParitySeverity.WARNING,
        ParityFindingKind.MISSING_RESOLVED_CALL_PATH: ParitySeverity.ERROR,
        ParityFindingKind.TRANSPORT_MISMATCH: ParitySeverity.ERROR,
        ParityFindingKind.PROFILE_MISMATCH: ParitySeverity.ERROR,
        ParityFindingKind.CAPABILITY_CLAIM_MISMATCH: ParitySeverity.WARNING,
        ParityFindingKind.DEGRADATION_CLAIM_MISMATCH: ParitySeverity.WARNING,
        ParityFindingKind.NAME_MISMATCH: ParitySeverity.ERROR,
        ParityFindingKind.VERSION_MISMATCH: ParitySeverity.ERROR,
        ParityFindingKind.PYTHON_SIGNATURE_MISMATCH: ParitySeverity.ERROR,
        ParityFindingKind.IMPLEMENTATION_TARGET_MISMATCH: ParitySeverity.ERROR,
        ParityFindingKind.RUNTIME_MOCK_AUTHORITY: ParitySeverity.ERROR,
        ParityFindingKind.CONTRACT_PACK_GAP: ParitySeverity.WARNING,
    }
)

# Roles that contribute named tool inventory for cross-surface tool discovery.
_TOOL_BEARING_ROLES: Final[frozenset[ArtifactRole]] = frozenset(
    {
        ArtifactRole.TOOL_LIST_ENTRY,
        ArtifactRole.REGISTRATION,
        ArtifactRole.MANIFEST,
        ArtifactRole.ALIAS,
        ArtifactRole.IMPLEMENTATION,
        ArtifactRole.ADAPTER,
        ArtifactRole.TOOL_CALL_SITE,
        ArtifactRole.CONNECTOR,
        ArtifactRole.RESULT_MAP,
        ArtifactRole.ERROR_MAP,
        ArtifactRole.JSON_SCHEMA,
    }
)

_GENERATED_ROLES: Final[frozenset[ArtifactRole]] = frozenset(
    {
        ArtifactRole.MANIFEST,
        ArtifactRole.COPIED_MANIFEST,
    }
)

_NON_INVOCATION_REASONS: Final[frozenset[ReasonCode]] = frozenset(
    {
        ReasonCode.SAME_NAME_HELPER,
        ReasonCode.MOCK_IMPLEMENTATION,
        ReasonCode.TEST_SERVER,
        ReasonCode.COPIED_MANIFEST,
        ReasonCode.STATIC_DASHBOARD,
        ReasonCode.LEGACY_FALLBACK,
        ReasonCode.IMPORT_WITHOUT_CALL,
    }
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise VfsMcpCheckerError(f"{name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise VfsMcpCheckerError(f"{name} is required")
    if len(text.encode("utf-8")) > DEFAULT_MAX_LABEL_BYTES:
        raise VfsMcpCheckerBoundsError(f"{name} exceeds label bound")
    return text


def _enum(value: Any, enum_type: type[Enum], label: str) -> Any:
    if isinstance(value, enum_type):
        return value
    text = str(value or "").strip()
    try:
        return enum_type(text)
    except ValueError as exc:
        raise VfsMcpCheckerError(f"unsupported {label}: {text!r}") from exc


def _mapping(
    value: Any,
    name: str,
    *,
    max_bytes: int = DEFAULT_MAX_NOTES_BYTES,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise VfsMcpCheckerError(f"{name} must be a mapping")
    plain: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise VfsMcpCheckerError(f"{name} keys must be strings")
        if isinstance(item, Enum):
            plain[key] = item.value
        elif isinstance(item, (str, bool, int)) or item is None:
            plain[key] = item
        elif isinstance(item, float):
            plain[key] = item
        elif isinstance(item, Mapping):
            plain[key] = dict(_mapping(item, f"{name}.{key}", max_bytes=max_bytes))
        elif isinstance(item, (list, tuple)):
            plain[key] = [
                (
                    x.value
                    if isinstance(x, Enum)
                    else (
                        dict(_mapping(x, f"{name}.{key}[]", max_bytes=max_bytes))
                        if isinstance(x, Mapping)
                        else x
                    )
                )
                for x in item
            ]
        else:
            plain[key] = str(item)
    encoded = str(plain).encode("utf-8")
    if len(encoded) > max_bytes:
        raise VfsMcpCheckerBoundsError(f"{name} exceeds notes bound")
    return MappingProxyType(plain)


def _schema_payload(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise VfsMcpCheckerError(f"{name} must be a mapping")
    plain = {str(k): v for k, v in value.items()}
    # Bound by fingerprinting; reject pathological sizes.
    try:
        blob = str(plain).encode("utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        raise VfsMcpCheckerError(f"{name} is not serializable") from exc
    if len(blob) > DEFAULT_MAX_SCHEMA_BYTES:
        raise VfsMcpCheckerBoundsError(f"{name} exceeds schema bound")
    return MappingProxyType(plain)


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(v).strip() for v in values if str(v).strip()}))


def _tool_key(name: str) -> str:
    return normalize_tool_name(name) or _text(name, "tool_name", required=False)


# ---------------------------------------------------------------------------
# Surface view + witnesses + findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceView:
    """One layer's observation for a single VFS tool name."""

    surface: ParitySurface
    present: bool
    tool_name: str = ""
    qualified_name: str = ""
    language: str = ""
    package: str = ""
    version: str = ""
    input_schema_fingerprint: str = ""
    output_schema_fingerprint: str = ""
    error_codes: tuple[str, ...] = ()
    transport: TransportKind = TransportKind.UNKNOWN
    profiles: tuple[str, ...] = ()
    alias_of: str = ""
    implementation_target: str = ""
    capability_claims: tuple[str, ...] = ()
    degradation_claims: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    has_call_edge: bool = False
    is_generated: bool = False
    is_mock_or_fallback: bool = False
    is_local_bypass: bool = False
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "surface", _enum(self.surface, ParitySurface, "surface")
        )
        if not isinstance(self.present, bool):
            raise VfsMcpCheckerError("present must be a boolean")
        object.__setattr__(
            self, "tool_name", _text(self.tool_name, "tool_name", required=False)
        )
        object.__setattr__(
            self,
            "qualified_name",
            _text(self.qualified_name, "qualified_name", required=False),
        )
        object.__setattr__(
            self, "language", _text(self.language, "language", required=False)
        )
        object.__setattr__(
            self, "package", _text(self.package, "package", required=False)
        )
        object.__setattr__(
            self, "version", _text(self.version, "version", required=False)
        )
        object.__setattr__(
            self,
            "input_schema_fingerprint",
            _text(
                self.input_schema_fingerprint,
                "input_schema_fingerprint",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "output_schema_fingerprint",
            _text(
                self.output_schema_fingerprint,
                "output_schema_fingerprint",
                required=False,
            ),
        )
        object.__setattr__(
            self, "error_codes", _sorted_unique(self.error_codes or ())
        )
        object.__setattr__(
            self,
            "transport",
            _enum(self.transport, TransportKind, "transport"),
        )
        object.__setattr__(self, "profiles", _sorted_unique(self.profiles or ()))
        object.__setattr__(
            self, "alias_of", _text(self.alias_of, "alias_of", required=False)
        )
        object.__setattr__(
            self,
            "implementation_target",
            _text(
                self.implementation_target,
                "implementation_target",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "capability_claims",
            _sorted_unique(self.capability_claims or ()),
        )
        object.__setattr__(
            self,
            "degradation_claims",
            _sorted_unique(self.degradation_claims or ()),
        )
        object.__setattr__(
            self, "artifact_ids", _sorted_unique(self.artifact_ids or ())
        )
        for flag_name in (
            "has_call_edge",
            "is_generated",
            "is_mock_or_fallback",
            "is_local_bypass",
        ):
            flag = getattr(self, flag_name)
            if not isinstance(flag, bool):
                raise VfsMcpCheckerError(f"{flag_name} must be a boolean")
        object.__setattr__(self, "notes", _mapping(self.notes, "surface.notes"))

    @property
    def view_id(self) -> str:
        return "vfsurf-" + content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": VFS_MCP_SURFACE_VIEW_SCHEMA,
            "surface": self.surface.value,
            "present": self.present,
            "tool_name": self.tool_name,
            "qualified_name": self.qualified_name,
            "language": self.language,
            "package": self.package,
            "version": self.version,
            "input_schema_fingerprint": self.input_schema_fingerprint,
            "output_schema_fingerprint": self.output_schema_fingerprint,
            "error_codes": list(self.error_codes),
            "transport": self.transport.value,
            "profiles": list(self.profiles),
            "alias_of": self.alias_of,
            "implementation_target": self.implementation_target,
            "capability_claims": list(self.capability_claims),
            "degradation_claims": list(self.degradation_claims),
            "artifact_ids": list(self.artifact_ids),
            "has_call_edge": self.has_call_edge,
            "is_generated": self.is_generated,
            "is_mock_or_fallback": self.is_mock_or_fallback,
            "is_local_bypass": self.is_local_bypass,
            "notes": dict(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "view_id": self.view_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SurfaceView":
        if not isinstance(payload, Mapping):
            raise VfsMcpCheckerError("surface view payload must be a mapping")
        return cls(
            surface=payload.get("surface", ParitySurface.REGISTRATION.value),
            present=bool(payload.get("present", False)),
            tool_name=str(payload.get("tool_name") or ""),
            qualified_name=str(payload.get("qualified_name") or ""),
            language=str(payload.get("language") or ""),
            package=str(payload.get("package") or ""),
            version=str(payload.get("version") or ""),
            input_schema_fingerprint=str(
                payload.get("input_schema_fingerprint") or ""
            ),
            output_schema_fingerprint=str(
                payload.get("output_schema_fingerprint") or ""
            ),
            error_codes=tuple(payload.get("error_codes") or ()),
            transport=payload.get("transport", TransportKind.UNKNOWN.value),
            profiles=tuple(payload.get("profiles") or ()),
            alias_of=str(payload.get("alias_of") or ""),
            implementation_target=str(
                payload.get("implementation_target") or ""
            ),
            capability_claims=tuple(payload.get("capability_claims") or ()),
            degradation_claims=tuple(payload.get("degradation_claims") or ()),
            artifact_ids=tuple(payload.get("artifact_ids") or ()),
            has_call_edge=bool(payload.get("has_call_edge", False)),
            is_generated=bool(payload.get("is_generated", False)),
            is_mock_or_fallback=bool(payload.get("is_mock_or_fallback", False)),
            is_local_bypass=bool(payload.get("is_local_bypass", False)),
            notes=payload.get("notes") or {},
        )

    @classmethod
    def absent(cls, surface: ParitySurface, *, tool_name: str = "") -> "SurfaceView":
        return cls(surface=surface, present=False, tool_name=tool_name)


@dataclass(frozen=True)
class ParityWitness:
    """Minimal witness binding two surfaces (or one surface + path evidence)."""

    kind: ParityFindingKind
    tool_name: str
    left_surface: ParitySurface
    right_surface: ParitySurface
    left_value: str = ""
    right_value: str = ""
    left_ref: str = ""
    right_ref: str = ""
    path_id: str = ""
    path_verdict: str = ""
    evidence_refs: tuple[str, ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, ParityFindingKind, "kind")
        )
        object.__setattr__(
            self, "tool_name", _text(self.tool_name, "tool_name", required=False)
        )
        object.__setattr__(
            self,
            "left_surface",
            _enum(self.left_surface, ParitySurface, "left_surface"),
        )
        object.__setattr__(
            self,
            "right_surface",
            _enum(self.right_surface, ParitySurface, "right_surface"),
        )
        object.__setattr__(
            self, "left_value", _text(self.left_value, "left_value", required=False)
        )
        object.__setattr__(
            self,
            "right_value",
            _text(self.right_value, "right_value", required=False),
        )
        object.__setattr__(
            self, "left_ref", _text(self.left_ref, "left_ref", required=False)
        )
        object.__setattr__(
            self, "right_ref", _text(self.right_ref, "right_ref", required=False)
        )
        object.__setattr__(
            self, "path_id", _text(self.path_id, "path_id", required=False)
        )
        object.__setattr__(
            self,
            "path_verdict",
            _text(self.path_verdict, "path_verdict", required=False),
        )
        object.__setattr__(
            self, "evidence_refs", _sorted_unique(self.evidence_refs or ())
        )
        object.__setattr__(self, "notes", _mapping(self.notes, "witness.notes"))

    @property
    def witness_id(self) -> str:
        return "vfswit-" + content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": VFS_MCP_PARITY_WITNESS_SCHEMA,
            "kind": self.kind.value,
            "tool_name": self.tool_name,
            "left_surface": self.left_surface.value,
            "right_surface": self.right_surface.value,
            "left_value": self.left_value,
            "right_value": self.right_value,
            "left_ref": self.left_ref,
            "right_ref": self.right_ref,
            "path_id": self.path_id,
            "path_verdict": self.path_verdict,
            "evidence_refs": list(self.evidence_refs),
            "notes": dict(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "witness_id": self.witness_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParityWitness":
        if not isinstance(payload, Mapping):
            raise VfsMcpCheckerError("witness payload must be a mapping")
        return cls(
            kind=payload.get(
                "kind", ParityFindingKind.MISSING_RESOLVED_CALL_PATH.value
            ),
            tool_name=str(payload.get("tool_name") or ""),
            left_surface=payload.get(
                "left_surface", ParitySurface.REGISTRATION.value
            ),
            right_surface=payload.get(
                "right_surface", ParitySurface.TOOLS_LIST.value
            ),
            left_value=str(payload.get("left_value") or ""),
            right_value=str(payload.get("right_value") or ""),
            left_ref=str(payload.get("left_ref") or ""),
            right_ref=str(payload.get("right_ref") or ""),
            path_id=str(payload.get("path_id") or ""),
            path_verdict=str(payload.get("path_verdict") or ""),
            evidence_refs=tuple(payload.get("evidence_refs") or ()),
            notes=payload.get("notes") or {},
        )


@dataclass(frozen=True)
class ParityFinding:
    """One parity defect with severity and minimal witnesses."""

    kind: ParityFindingKind
    tool_name: str
    severity: ParitySeverity
    summary: str
    witnesses: tuple[ParityWitness, ...]
    surfaces: tuple[ParitySurface, ...] = ()
    path_ids: tuple[str, ...] = ()
    confidence: int = 100
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, ParityFindingKind, "kind")
        )
        object.__setattr__(
            self, "tool_name", _text(self.tool_name, "tool_name", required=False)
        )
        object.__setattr__(
            self, "severity", _enum(self.severity, ParitySeverity, "severity")
        )
        object.__setattr__(self, "summary", _text(self.summary, "summary"))
        witnesses = tuple(
            item if isinstance(item, ParityWitness) else ParityWitness.from_dict(item)
            for item in (self.witnesses or ())
        )
        if not witnesses:
            raise VfsMcpCheckerError("parity finding requires at least one witness")
        if len(witnesses) > DEFAULT_MAX_WITNESSES:
            raise VfsMcpCheckerBoundsError("too many witnesses on one finding")
        object.__setattr__(self, "witnesses", witnesses)
        surfaces = tuple(
            _enum(item, ParitySurface, "surface") for item in (self.surfaces or ())
        )
        object.__setattr__(
            self,
            "surfaces",
            tuple(sorted(set(surfaces), key=lambda s: s.value)),
        )
        object.__setattr__(self, "path_ids", _sorted_unique(self.path_ids or ()))
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, int)
            or self.confidence < 0
            or self.confidence > 100
        ):
            raise VfsMcpCheckerError("confidence must be an int in [0, 100]")
        object.__setattr__(self, "notes", _mapping(self.notes, "finding.notes"))

    @property
    def finding_id(self) -> str:
        return "vfsfind-" + content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": VFS_MCP_PARITY_FINDING_SCHEMA,
            "kind": self.kind.value,
            "tool_name": self.tool_name,
            "severity": self.severity.value,
            "summary": self.summary,
            "witnesses": [item.to_dict() for item in self.witnesses],
            "surfaces": [item.value for item in self.surfaces],
            "path_ids": list(self.path_ids),
            "confidence": self.confidence,
            "notes": dict(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "finding_id": self.finding_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParityFinding":
        if not isinstance(payload, Mapping):
            raise VfsMcpCheckerError("finding payload must be a mapping")
        return cls(
            kind=payload.get(
                "kind", ParityFindingKind.MISSING_RESOLVED_CALL_PATH.value
            ),
            tool_name=str(payload.get("tool_name") or ""),
            severity=payload.get("severity", ParitySeverity.ERROR.value),
            summary=str(payload.get("summary") or ""),
            witnesses=tuple(payload.get("witnesses") or ()),
            surfaces=tuple(payload.get("surfaces") or ()),
            path_ids=tuple(payload.get("path_ids") or ()),
            confidence=int(payload.get("confidence", 100)),
            notes=payload.get("notes") or {},
        )


@dataclass(frozen=True)
class ToolParityResult:
    """Parity result for one tool across all compared surfaces."""

    tool_name: str
    verdict: ToolParityVerdict
    surfaces: Mapping[str, SurfaceView]
    findings: tuple[ParityFinding, ...]
    path_ids: tuple[str, ...] = ()
    path_verdicts: tuple[str, ...] = ()
    proved_call_path: bool = False
    text_names_agree: bool = False
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tool_name", _text(self.tool_name, "tool_name")
        )
        object.__setattr__(
            self, "verdict", _enum(self.verdict, ToolParityVerdict, "verdict")
        )
        if not isinstance(self.surfaces, Mapping):
            raise VfsMcpCheckerError("surfaces must be a mapping")
        normalized: dict[str, SurfaceView] = {}
        for key, view in self.surfaces.items():
            surface_key = str(key)
            if isinstance(view, SurfaceView):
                normalized[surface_key] = view
            elif isinstance(view, Mapping):
                normalized[surface_key] = SurfaceView.from_dict(view)
            else:
                raise VfsMcpCheckerError("surface views must be SurfaceView")
        object.__setattr__(self, "surfaces", MappingProxyType(normalized))
        findings = tuple(
            item if isinstance(item, ParityFinding) else ParityFinding.from_dict(item)
            for item in (self.findings or ())
        )
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "path_ids", _sorted_unique(self.path_ids or ()))
        object.__setattr__(
            self, "path_verdicts", _sorted_unique(self.path_verdicts or ())
        )
        if not isinstance(self.proved_call_path, bool):
            raise VfsMcpCheckerError("proved_call_path must be a boolean")
        if not isinstance(self.text_names_agree, bool):
            raise VfsMcpCheckerError("text_names_agree must be a boolean")
        # Fail closed: text agreement without a proved path is never proved_parity.
        if (
            self.verdict is ToolParityVerdict.PROVED_PARITY
            and not self.proved_call_path
        ):
            raise VfsMcpCheckerError(
                "proved_parity requires a resolved call path; "
                "same text without a path is insufficient"
            )
        object.__setattr__(self, "notes", _mapping(self.notes, "tool.notes"))

    @property
    def result_id(self) -> str:
        return "vfsparty-" + content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": VFS_MCP_TOOL_PARITY_SCHEMA,
            "tool_name": self.tool_name,
            "verdict": self.verdict.value,
            "surfaces": {
                key: view.to_dict() for key, view in sorted(self.surfaces.items())
            },
            "findings": [item.to_dict() for item in self.findings],
            "path_ids": list(self.path_ids),
            "path_verdicts": list(self.path_verdicts),
            "proved_call_path": self.proved_call_path,
            "text_names_agree": self.text_names_agree,
            "notes": dict(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "result_id": self.result_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolParityResult":
        if not isinstance(payload, Mapping):
            raise VfsMcpCheckerError("tool parity payload must be a mapping")
        return cls(
            tool_name=str(payload.get("tool_name") or ""),
            verdict=payload.get("verdict", ToolParityVerdict.UNKNOWN.value),
            surfaces=payload.get("surfaces") or {},
            findings=tuple(payload.get("findings") or ()),
            path_ids=tuple(payload.get("path_ids") or ()),
            path_verdicts=tuple(payload.get("path_verdicts") or ()),
            proved_call_path=bool(payload.get("proved_call_path", False)),
            text_names_agree=bool(payload.get("text_names_agree", False)),
            notes=payload.get("notes") or {},
        )


@dataclass(frozen=True)
class VfsMcpParityReport:
    """Content-addressed end-to-end VFS MCP/MCP++ parity report."""

    forest_id: str
    inventory_id: str
    contract_pack_id: str
    tools: tuple[ToolParityResult, ...]
    findings: tuple[ParityFinding, ...]
    verdict: ReportVerdict
    checker_version: str = CHECKER_VERSION
    goal_id: str = GOAL_ID
    truncated: bool = False
    truncation_reason: str = ""
    evidence_kinds: tuple[str, ...] = (
        EVIDENCE_VFS_MCP_PARITY,
        EVIDENCE_VFS_MCP_CALL_PATH,
        EVIDENCE_MANIFEST_PARITY,
    )
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "forest_id")
        )
        object.__setattr__(
            self,
            "inventory_id",
            _text(self.inventory_id, "inventory_id", required=False),
        )
        object.__setattr__(
            self,
            "contract_pack_id",
            _text(self.contract_pack_id, "contract_pack_id", required=False),
        )
        tools = tuple(
            item if isinstance(item, ToolParityResult) else ToolParityResult.from_dict(item)
            for item in (self.tools or ())
        )
        if len(tools) > DEFAULT_MAX_TOOLS:
            raise VfsMcpCheckerBoundsError("too many tools in report")
        object.__setattr__(self, "tools", tools)
        findings = tuple(
            item if isinstance(item, ParityFinding) else ParityFinding.from_dict(item)
            for item in (self.findings or ())
        )
        if len(findings) > DEFAULT_MAX_FINDINGS:
            raise VfsMcpCheckerBoundsError("too many findings in report")
        object.__setattr__(self, "findings", findings)
        object.__setattr__(
            self, "verdict", _enum(self.verdict, ReportVerdict, "verdict")
        )
        object.__setattr__(
            self,
            "checker_version",
            _text(self.checker_version, "checker_version"),
        )
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id"))
        if not isinstance(self.truncated, bool):
            raise VfsMcpCheckerError("truncated must be a boolean")
        object.__setattr__(
            self,
            "truncation_reason",
            _text(self.truncation_reason, "truncation_reason", required=False),
        )
        object.__setattr__(
            self,
            "evidence_kinds",
            _sorted_unique(self.evidence_kinds or ()),
        )
        object.__setattr__(self, "notes", _mapping(self.notes, "report.notes"))

    @property
    def report_id(self) -> str:
        return "vfsprpt-" + content_identity(self._identity_payload())

    @property
    def is_completion_evidence(self) -> bool:
        return CHECKER_IS_COMPLETION_EVIDENCE

    @property
    def authorizes_repair(self) -> bool:
        return CHECKER_AUTHORIZES_REPAIR

    def findings_of(self, kind: ParityFindingKind | str) -> tuple[ParityFinding, ...]:
        target = _enum(kind, ParityFindingKind, "kind")
        return tuple(item for item in self.findings if item.kind is target)

    def tool_result(self, tool_name: str) -> ToolParityResult | None:
        key = _tool_key(tool_name)
        for item in self.tools:
            if _tool_key(item.tool_name) == key or item.tool_name == tool_name:
                return item
        return None

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": VFS_MCP_PARITY_REPORT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "checker_version": self.checker_version,
            "goal_id": self.goal_id,
            "forest_id": self.forest_id,
            "inventory_id": self.inventory_id,
            "contract_pack_id": self.contract_pack_id,
            "tools": [item.to_dict() for item in self.tools],
            "findings": [item.to_dict() for item in self.findings],
            "verdict": self.verdict.value,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
            "evidence_kinds": list(self.evidence_kinds),
            "notes": dict(self.notes),
            "authority": {
                "completion_evidence": CHECKER_IS_COMPLETION_EVIDENCE,
                "correctness_evidence": CHECKER_IS_CORRECTNESS_EVIDENCE,
                "authorizes_repair": CHECKER_AUTHORIZES_REPAIR,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "report_id": self.report_id,
            "producer": CHECKER_PRODUCER,
        }

    def to_record(self) -> dict[str, Any]:
        return self.to_dict()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VfsMcpParityReport":
        if not isinstance(payload, Mapping):
            raise VfsMcpCheckerError("report payload must be a mapping")
        return cls(
            forest_id=str(payload.get("forest_id") or ""),
            inventory_id=str(payload.get("inventory_id") or ""),
            contract_pack_id=str(payload.get("contract_pack_id") or ""),
            tools=tuple(payload.get("tools") or ()),
            findings=tuple(payload.get("findings") or ()),
            verdict=payload.get("verdict", ReportVerdict.UNKNOWN.value),
            checker_version=str(payload.get("checker_version") or CHECKER_VERSION),
            goal_id=str(payload.get("goal_id") or GOAL_ID),
            truncated=bool(payload.get("truncated", False)),
            truncation_reason=str(payload.get("truncation_reason") or ""),
            evidence_kinds=tuple(payload.get("evidence_kinds") or ()),
            notes=payload.get("notes") or {},
        )


# ---------------------------------------------------------------------------
# Inventory surface extraction
# ---------------------------------------------------------------------------


def _artifact_is_mock_or_fallback(item: InventoryArtifact) -> bool:
    reason = item.non_invocation_reason
    if reason in {
        ReasonCode.MOCK_IMPLEMENTATION,
        ReasonCode.LEGACY_FALLBACK,
        ReasonCode.TEST_SERVER,
        ReasonCode.STATIC_DASHBOARD,
    }:
        return True
    if item.role in {
        ArtifactRole.MOCK,
        ArtifactRole.LEGACY_FALLBACK,
        ArtifactRole.TEST_SERVER,
        ArtifactRole.STATIC_DASHBOARD,
    }:
        return True
    markers = {m.lower() for m in item.markers}
    return bool(markers & {"mock", "fallback", "legacy_fallback", "stub"})


def _artifact_is_local_bypass(item: InventoryArtifact) -> bool:
    if item.role is ArtifactRole.LOCAL_HELPER:
        return True
    if item.non_invocation_reason is ReasonCode.SAME_NAME_HELPER:
        return True
    markers = {m.lower() for m in item.markers}
    if "local_bypass" in markers or "direct_local" in markers:
        return True
    record = item.record or {}
    if str(record.get("dispatch") or "").lower() in {
        "local_bypass",
        "direct_local",
        "same_name_helper",
    }:
        return True
    return False


def _claims_from_record(
    record: Mapping[str, Any],
    *keys: str,
) -> tuple[str, ...]:
    values: list[str] = []
    for key in keys:
        raw = record.get(key)
        if raw is None:
            continue
        if isinstance(raw, str):
            if raw.strip():
                values.append(raw.strip())
        elif isinstance(raw, (list, tuple)):
            for item in raw:
                text = str(item or "").strip()
                if text:
                    values.append(text)
    return _sorted_unique(values)


def _view_from_artifacts(
    surface: ParitySurface,
    artifacts: Sequence[InventoryArtifact],
    *,
    tool_name: str,
    prefer_language: str = "",
    is_generated: bool = False,
) -> SurfaceView:
    if not artifacts:
        return SurfaceView.absent(surface, tool_name=tool_name)
    ordered = sorted(
        artifacts,
        key=lambda item: (
            0 if prefer_language and item.language == prefer_language else 1,
            item.artifact_id,
        ),
    )
    primary = ordered[0]
    input_fps = {
        schema_fingerprint(item.input_schema)
        for item in ordered
        if item.input_schema
    }
    output_fps = {
        schema_fingerprint(item.output_schema)
        for item in ordered
        if item.output_schema
    }
    errors: set[str] = set()
    profiles: set[str] = set()
    transports = {
        item.transport
        for item in ordered
        if item.transport is not TransportKind.UNKNOWN
    }
    for item in ordered:
        errors.update(item.error_codes)
        profiles.update(item.profiles)
    capability: list[str] = []
    degradation: list[str] = []
    impl_targets: list[str] = []
    for item in ordered:
        capability.extend(
            _claims_from_record(
                item.record,
                "capabilities",
                "capability_claims",
                "capability",
            )
        )
        degradation.extend(
            _claims_from_record(
                item.record,
                "degradation",
                "degradation_claims",
                "fallback",
            )
        )
        target = str(
            item.record.get("implementation")
            or item.record.get("implementation_target")
            or ""
        ).strip()
        if not target and item.role is ArtifactRole.IMPLEMENTATION:
            target = item.qualified_name
        if target:
            impl_targets.append(target)
    mock = any(_artifact_is_mock_or_fallback(item) for item in ordered)
    bypass = any(_artifact_is_local_bypass(item) for item in ordered)
    return SurfaceView(
        surface=surface,
        present=True,
        tool_name=primary.effective_tool_name or tool_name,
        qualified_name=primary.qualified_name,
        language=primary.language,
        package=primary.package,
        version=primary.version,
        input_schema_fingerprint=sorted(input_fps)[0] if len(input_fps) == 1 else (
            "multi:" + ",".join(sorted(input_fps)) if input_fps else ""
        ),
        output_schema_fingerprint=(
            sorted(output_fps)[0]
            if len(output_fps) == 1
            else ("multi:" + ",".join(sorted(output_fps)) if output_fps else "")
        ),
        error_codes=tuple(sorted(errors)),
        transport=(
            next(iter(transports))
            if len(transports) == 1
            else TransportKind.UNKNOWN
        ),
        profiles=tuple(sorted(profiles)),
        alias_of=primary.alias_of,
        implementation_target=impl_targets[0] if len(set(impl_targets)) == 1 else (
            "multi:" + ",".join(sorted(set(impl_targets))) if impl_targets else ""
        ),
        capability_claims=tuple(sorted(set(capability))),
        degradation_claims=tuple(sorted(set(degradation))),
        artifact_ids=tuple(item.artifact_id for item in ordered),
        has_call_edge=any(item.has_call_edge for item in ordered),
        is_generated=is_generated
        or any(item.role in _GENERATED_ROLES for item in ordered)
        or any("generated" in m.lower() for item in ordered for m in item.markers),
        is_mock_or_fallback=mock,
        is_local_bypass=bypass,
        notes={
            "artifact_count": len(ordered),
            "transport_count": len(transports),
            "input_schema_variants": len(input_fps),
            "output_schema_variants": len(output_fps),
        },
    )


def _match_tool(
    inventory: MCPlusPlusInventory,
    tool_name: str,
    *,
    role: ArtifactRole | None = None,
    language: str = "",
) -> tuple[InventoryArtifact, ...]:
    key = _tool_key(tool_name)
    aliases = set(tool_name_aliases(tool_name))
    aliases.add(key)
    matched: list[InventoryArtifact] = []
    for item in inventory.artifacts:
        if role is not None and item.role is not role:
            continue
        if language and item.language and item.language != language:
            continue
        candidates = {
            _tool_key(item.effective_tool_name),
            _tool_key(item.name),
            _tool_key(item.alias_of) if item.alias_of else "",
            *tool_name_aliases(item.effective_tool_name),
            *tool_name_aliases(item.name),
        }
        candidates.discard("")
        if aliases & candidates:
            matched.append(item)
    return tuple(sorted(matched, key=lambda item: item.artifact_id))


def discover_tool_names(
    inventory: MCPlusPlusInventory,
    *,
    server_name: str = "",
    vfs_prefix_only: bool = False,
) -> tuple[str, ...]:
    """Discover tool names present on any tool-bearing surface."""

    names: set[str] = set()
    for item in inventory.artifacts:
        if item.role not in _TOOL_BEARING_ROLES:
            continue
        if server_name and item.server_name and item.server_name != server_name:
            continue
        name = item.effective_tool_name or item.name
        if not name:
            continue
        if vfs_prefix_only:
            normalized = normalize_tool_name(name)
            if not (
                normalized.startswith("vfs.")
                or normalized.startswith("vfs/")
                or normalized.startswith("ipfs.")
                or "/vfs" in item.path.lower()
                or "vfs" in (item.package or "").lower()
            ):
                # Keep explicit tool_name even without vfs prefix when role is
                # registration/list (caller may pass non-prefixed names).
                if item.role not in {
                    ArtifactRole.REGISTRATION,
                    ArtifactRole.TOOL_LIST_ENTRY,
                    ArtifactRole.MANIFEST,
                    ArtifactRole.IMPLEMENTATION,
                }:
                    continue
        names.add(name)
    return tuple(sorted(names, key=lambda n: (_tool_key(n), n)))


def build_surface_views(
    inventory: MCPlusPlusInventory,
    tool_name: str,
    *,
    contract_pack: VfsContractPack | None = None,
    runtime_receipts: Sequence[RuntimeWitnessReceipt] = (),
) -> dict[str, SurfaceView]:
    """Build the closed surface views for one tool from inventory evidence."""

    py_impl = _match_tool(
        inventory, tool_name, role=ArtifactRole.IMPLEMENTATION, language="python"
    )
    if not py_impl:
        py_impl = tuple(
            item
            for item in _match_tool(inventory, tool_name, role=ArtifactRole.IMPLEMENTATION)
            if not item.language or item.language == "python"
        )
    py_sig = _match_tool(
        inventory, tool_name, role=ArtifactRole.JSON_SCHEMA, language="python"
    )
    # Python signature prefers implementation + python registration signatures.
    py_regs = _match_tool(
        inventory, tool_name, role=ArtifactRole.REGISTRATION, language="python"
    )
    python_artifacts = py_impl or py_regs or py_sig

    registrations = _match_tool(inventory, tool_name, role=ArtifactRole.REGISTRATION)
    tools_list = _match_tool(inventory, tool_name, role=ArtifactRole.TOOL_LIST_ENTRY)
    manifests = _match_tool(inventory, tool_name, role=ArtifactRole.MANIFEST)
    if not manifests:
        manifests = _match_tool(inventory, tool_name, role=ArtifactRole.COPIED_MANIFEST)
    ts_sdk = tuple(
        item
        for item in _match_tool(inventory, tool_name)
        if item.language in {"typescript", "ts", "javascript", "js"}
        and item.role
        in {
            ArtifactRole.MANIFEST,
            ArtifactRole.ALIAS,
            ArtifactRole.JSON_SCHEMA,
            ArtifactRole.CONNECTOR,
            ArtifactRole.TOOL_CALL_SITE,
        }
    )
    # Explicit SDK marker support.
    ts_sdk = ts_sdk + tuple(
        item
        for item in _match_tool(inventory, tool_name)
        if "sdk" in {m.lower() for m in item.markers}
        or str(item.record.get("artifact_kind") or "").lower()
        in {"typescript_sdk", "sdk", "generated_sdk"}
    )
    # Deduplicate ts_sdk by artifact_id.
    seen_ts: set[str] = set()
    unique_ts: list[InventoryArtifact] = []
    for item in ts_sdk:
        if item.artifact_id not in seen_ts:
            seen_ts.add(item.artifact_id)
            unique_ts.append(item)
    ts_sdk = tuple(unique_ts)

    connectors = _match_tool(inventory, tool_name, role=ArtifactRole.CONNECTOR)
    if not connectors:
        connectors = _match_tool(inventory, tool_name, role=ArtifactRole.TOOL_CALL_SITE)
    transports = _match_tool(inventory, tool_name, role=ArtifactRole.TRANSPORT)
    if not transports:
        # Transport may be bound on connector/registration without a transport role.
        transports = tuple(
            item
            for item in registrations + connectors
            if item.transport is not TransportKind.UNKNOWN or item.profiles
        )
    result_maps = _match_tool(inventory, tool_name, role=ArtifactRole.RESULT_MAP)
    error_maps = _match_tool(inventory, tool_name, role=ArtifactRole.ERROR_MAP)
    result_error = result_maps + error_maps
    adapters = _match_tool(inventory, tool_name, role=ArtifactRole.ADAPTER)
    implementations = _match_tool(inventory, tool_name, role=ArtifactRole.IMPLEMENTATION)
    local_helpers = _match_tool(inventory, tool_name, role=ArtifactRole.LOCAL_HELPER)
    mocks = (
        _match_tool(inventory, tool_name, role=ArtifactRole.MOCK)
        + _match_tool(inventory, tool_name, role=ArtifactRole.LEGACY_FALLBACK)
        + _match_tool(inventory, tool_name, role=ArtifactRole.TEST_SERVER)
    )

    views: dict[str, SurfaceView] = {
        ParitySurface.PYTHON_SIGNATURE.value: _view_from_artifacts(
            ParitySurface.PYTHON_SIGNATURE,
            python_artifacts,
            tool_name=tool_name,
            prefer_language="python",
        ),
        ParitySurface.REGISTRATION.value: _view_from_artifacts(
            ParitySurface.REGISTRATION,
            registrations,
            tool_name=tool_name,
            prefer_language="python",
        ),
        ParitySurface.TOOLS_LIST.value: _view_from_artifacts(
            ParitySurface.TOOLS_LIST,
            tools_list,
            tool_name=tool_name,
        ),
        ParitySurface.JSON_MANIFEST.value: _view_from_artifacts(
            ParitySurface.JSON_MANIFEST,
            manifests,
            tool_name=tool_name,
            is_generated=True,
        ),
        ParitySurface.TYPESCRIPT_SDK.value: _view_from_artifacts(
            ParitySurface.TYPESCRIPT_SDK,
            ts_sdk,
            tool_name=tool_name,
            prefer_language="typescript",
            is_generated=True,
        ),
        ParitySurface.SWISSKNIFE_CONNECTOR.value: _view_from_artifacts(
            ParitySurface.SWISSKNIFE_CONNECTOR,
            connectors,
            tool_name=tool_name,
            prefer_language="typescript",
        ),
        ParitySurface.TRANSPORT_PROFILE.value: _view_from_artifacts(
            ParitySurface.TRANSPORT_PROFILE,
            transports,
            tool_name=tool_name,
        ),
        ParitySurface.RESULT_ERROR_MAP.value: _view_from_artifacts(
            ParitySurface.RESULT_ERROR_MAP,
            result_error,
            tool_name=tool_name,
        ),
        ParitySurface.IMPLEMENTATION_TARGET.value: _view_from_artifacts(
            ParitySurface.IMPLEMENTATION_TARGET,
            implementations + adapters + local_helpers + mocks,
            tool_name=tool_name,
            prefer_language="python",
        ),
    }

    # Capability / degradation: prefer explicit claims, then contract pack.
    cap_sources = registrations + implementations + adapters + tools_list
    cap_view = _view_from_artifacts(
        ParitySurface.CAPABILITY_DEGRADATION,
        cap_sources,
        tool_name=tool_name,
    )
    if contract_pack is not None:
        op = _tool_name_to_vfs_operation(tool_name)
        pack_caps: list[str] = []
        pack_deg: list[str] = []
        if op is not None:
            for surface in (PublicSurface.MCP, PublicSurface.MCP_PLUS_PLUS):
                try:
                    sc = contract_pack.surface_contract(surface)
                except Exception:
                    sc = None
                if sc is None:
                    continue
                entry = sc.support_for(op)
                pack_caps.append(f"{surface.value}:{entry.support.value}")
                if entry.entrypoint:
                    pack_caps.append(f"entrypoint:{entry.entrypoint}")
            # Degradation invariant must be present for MCP surfaces.
            try:
                inv = contract_pack.invariant_contract(VfsInvariantKind.DEGRADATION)
                pack_deg.append(inv.invariant_id if hasattr(inv, "invariant_id") else "degradation")
            except Exception:
                for inv in getattr(contract_pack, "invariants", ()):
                    kind = getattr(inv, "kind", None) or getattr(inv, "invariant_kind", None)
                    if kind is VfsInvariantKind.DEGRADATION or str(kind) == "degradation":
                        pack_deg.append(getattr(inv, "invariant_id", "degradation"))
        # Merge pack claims into capability surface.
        cap_view = SurfaceView(
            surface=ParitySurface.CAPABILITY_DEGRADATION,
            present=cap_view.present or bool(pack_caps or pack_deg),
            tool_name=tool_name,
            qualified_name=cap_view.qualified_name,
            language=cap_view.language,
            package=cap_view.package,
            version=cap_view.version,
            input_schema_fingerprint=cap_view.input_schema_fingerprint,
            output_schema_fingerprint=cap_view.output_schema_fingerprint,
            error_codes=cap_view.error_codes,
            transport=cap_view.transport,
            profiles=cap_view.profiles,
            alias_of=cap_view.alias_of,
            implementation_target=cap_view.implementation_target,
            capability_claims=_sorted_unique(
                list(cap_view.capability_claims) + pack_caps
            ),
            degradation_claims=_sorted_unique(
                list(cap_view.degradation_claims) + pack_deg
            ),
            artifact_ids=cap_view.artifact_ids,
            has_call_edge=cap_view.has_call_edge,
            is_generated=cap_view.is_generated,
            is_mock_or_fallback=cap_view.is_mock_or_fallback,
            is_local_bypass=cap_view.is_local_bypass,
            notes={**dict(cap_view.notes), "contract_pack_bound": True},
        )
        views[ParitySurface.CONTRACT_PACK.value] = SurfaceView(
            surface=ParitySurface.CONTRACT_PACK,
            present=True,
            tool_name=tool_name,
            qualified_name=getattr(contract_pack, "content_id", "") or "vfs-contract-pack",
            capability_claims=tuple(pack_caps),
            degradation_claims=tuple(pack_deg),
            notes={
                "contract_pack_id": getattr(contract_pack, "content_id", ""),
                "operation": op.value if op is not None else "",
            },
        )
    views[ParitySurface.CAPABILITY_DEGRADATION.value] = cap_view

    # Runtime witness surface (optional, non-authoritative for mocks).
    runtime_views: list[SurfaceView] = []
    for receipt in runtime_receipts:
        if not _receipt_covers_tool(receipt, tool_name):
            continue
        runtime_views.append(_surface_from_runtime_receipt(receipt, tool_name))
    if runtime_views:
        # Prefer production-class positive witnesses.
        runtime_views.sort(
            key=lambda v: (
                0 if not v.is_mock_or_fallback else 1,
                v.view_id,
            )
        )
        views[ParitySurface.RUNTIME_WITNESS.value] = runtime_views[0]
    else:
        views[ParitySurface.RUNTIME_WITNESS.value] = SurfaceView.absent(
            ParitySurface.RUNTIME_WITNESS, tool_name=tool_name
        )

    return views


def _tool_name_to_vfs_operation(tool_name: str) -> VfsOperation | None:
    normalized = normalize_tool_name(tool_name)
    # Common forms: vfs.read, vfs/read, mcp_vfs_read, ipfs.vfs.stat
    tail = normalized.split(".")[-1].split("/")[-1]
    if tail.startswith("vfs_"):
        tail = tail[4:]
    aliases = {
        "path_resolve": "path.resolve",
        "path-resolve": "path.resolve",
        "resolve": "path.resolve",
        "ls": "list",
        "listdir": "list",
        "rm": "remove",
        "unlink": "remove",
        "mv": "rename",
        "cp": "copy",
        "makedirs": "mkdir",
    }
    candidate = aliases.get(tail, tail)
    # Also try full normalized forms like path.resolve
    for op in VfsOperation:
        if op.value == candidate or op.value == normalized or op.value.endswith(
            "." + candidate
        ):
            return op
        if op.value.split(".")[-1] == candidate:
            return op
    return None


def _iter_runtime_witnesses(receipt: RuntimeWitnessReceipt) -> tuple[Any, ...]:
    """Yield nested RuntimeWitness records from a receipt (or a bare witness)."""

    witnesses = getattr(receipt, "witnesses", None)
    if witnesses is not None:
        return tuple(witnesses)
    # Allow a bare RuntimeWitness-like object to be treated as a single witness.
    if hasattr(receipt, "observation") and hasattr(receipt, "request"):
        return (receipt,)
    return ()


def _receipt_covers_tool(receipt: RuntimeWitnessReceipt, tool_name: str) -> bool:
    key = _tool_key(tool_name)
    aliases = set(tool_name_aliases(tool_name))
    aliases.add(key)

    def _name_hits(name: Any) -> bool:
        if not name:
            return False
        text = str(name)
        return _tool_key(text) in aliases or text in aliases

    for witness in _iter_runtime_witnesses(receipt):
        request = getattr(witness, "request", None)
        observation = getattr(witness, "observation", None)
        discovery = getattr(witness, "discovery", None)
        if request is not None and _name_hits(getattr(request, "tool_name", None)):
            return True
        if observation is not None and _name_hits(
            getattr(observation, "tool_name", None)
        ):
            return True
        if discovery is not None:
            for name in getattr(discovery, "tool_names", ()) or ():
                if _name_hits(name):
                    return True
            for name in getattr(discovery, "production_tools", ()) or ():
                if _name_hits(name):
                    return True
            for name in getattr(discovery, "mock_tools", ()) or ():
                if _name_hits(name):
                    return True
    return False


def _surface_from_runtime_receipt(
    receipt: RuntimeWitnessReceipt,
    tool_name: str,
) -> SurfaceView:
    """Project the best-matching nested witness into a runtime surface view."""

    key = _tool_key(tool_name)
    aliases = set(tool_name_aliases(tool_name))
    aliases.add(key)
    selected = None
    for witness in _iter_runtime_witnesses(receipt):
        request = getattr(witness, "request", None)
        observation = getattr(witness, "observation", None)
        name = ""
        if request is not None:
            name = str(getattr(request, "tool_name", "") or "")
        if not name and observation is not None:
            name = str(getattr(observation, "tool_name", "") or "")
        if name and (_tool_key(name) in aliases or name in aliases):
            selected = witness
            break
    if selected is None:
        witnesses = _iter_runtime_witnesses(receipt)
        selected = witnesses[0] if witnesses else None
    if selected is None:
        return SurfaceView.absent(ParitySurface.RUNTIME_WITNESS, tool_name=tool_name)

    observation = getattr(selected, "observation", None)
    request = getattr(selected, "request", None)
    negotiation = getattr(selected, "negotiation", None)
    discovery = getattr(selected, "discovery", None)

    kind = getattr(observation, "implementation_kind", None) if observation else None
    mock = False
    if kind is ImplementationKind.MOCK:
        mock = True
    elif isinstance(kind, str) and kind == ImplementationKind.MOCK.value:
        mock = True
    if discovery is not None and tool_name:
        mock_tools = {
            normalize_tool_name(str(n))
            for n in (getattr(discovery, "mock_tools", ()) or ())
        }
        if key in mock_tools or normalize_tool_name(tool_name) in mock_tools:
            mock = True

    outcome = getattr(observation, "outcome", None) if observation else None
    outcome_value = (
        outcome.value if isinstance(outcome, Enum) else str(outcome or "")
    )
    target = str(
        (getattr(observation, "implementation_target", "") if observation else "")
        or ""
    )
    profiles: list[str] = []
    if negotiation is not None:
        profiles.extend(getattr(negotiation, "admitted_profiles", ()) or ())
        active = getattr(negotiation, "active_profile", None)
        if active:
            profiles.append(str(active))
    if request is not None:
        profiles.extend(getattr(request, "requested_profiles", ()) or ())

    transport_raw = getattr(selected, "transport", None)
    if transport_raw is None and negotiation is not None:
        transport_raw = getattr(negotiation, "active_transport", None)
    if isinstance(transport_raw, TransportKind):
        transport = transport_raw
    else:
        try:
            transport = TransportKind(str(transport_raw or "unknown"))
        except ValueError:
            transport = TransportKind.UNKNOWN

    receipt_id = str(
        getattr(receipt, "receipt_id", "")
        or getattr(receipt, "content_id", "")
        or getattr(receipt, "fixture_id", "")
        or ""
    )
    errors: list[str] = []
    if observation is not None:
        code = getattr(observation, "error_code", None)
        if code:
            errors.append(str(code))
        for item in getattr(observation, "input_errors", ()) or ():
            if item:
                errors.append(str(item))
        for item in getattr(observation, "output_errors", ()) or ():
            if item:
                errors.append(str(item))

    grants_authority = bool(
        getattr(observation, "grants_runtime_authority", False)
        if observation is not None
        else False
    )
    return SurfaceView(
        surface=ParitySurface.RUNTIME_WITNESS,
        present=True,
        tool_name=tool_name,
        qualified_name=target or receipt_id,
        error_codes=tuple(sorted(set(errors))),
        transport=transport,
        profiles=tuple(sorted({str(p) for p in profiles if p})),
        implementation_target=target,
        artifact_ids=(receipt_id,) if receipt_id else (),
        is_mock_or_fallback=mock,
        notes={
            "outcome": outcome_value,
            "implementation_kind": (
                kind.value if isinstance(kind, Enum) else str(kind or "")
            ),
            "authoritative": bool(grants_authority and not mock),
            "fixture_id": str(getattr(receipt, "fixture_id", "") or ""),
        },
    )


# ---------------------------------------------------------------------------
# Comparison rules
# ---------------------------------------------------------------------------


def _make_finding(
    kind: ParityFindingKind,
    tool_name: str,
    summary: str,
    witnesses: Sequence[ParityWitness],
    *,
    surfaces: Sequence[ParitySurface] = (),
    path_ids: Sequence[str] = (),
    confidence: int = 100,
    notes: Mapping[str, Any] | None = None,
) -> ParityFinding:
    return ParityFinding(
        kind=kind,
        tool_name=tool_name,
        severity=_FINDING_SEVERITY[kind],
        summary=summary,
        witnesses=tuple(witnesses),
        surfaces=tuple(surfaces),
        path_ids=tuple(path_ids),
        confidence=confidence,
        notes=notes or {},
    )


def _pair_witness(
    kind: ParityFindingKind,
    tool_name: str,
    left: SurfaceView,
    right: SurfaceView,
    *,
    left_value: str,
    right_value: str,
    path_id: str = "",
    path_verdict: str = "",
    notes: Mapping[str, Any] | None = None,
) -> ParityWitness:
    return ParityWitness(
        kind=kind,
        tool_name=tool_name,
        left_surface=left.surface,
        right_surface=right.surface,
        left_value=left_value,
        right_value=right_value,
        left_ref=left.qualified_name or left.tool_name or left.view_id,
        right_ref=right.qualified_name or right.tool_name or right.view_id,
        path_id=path_id,
        path_verdict=path_verdict,
        evidence_refs=_sorted_unique(
            list(left.artifact_ids) + list(right.artifact_ids)
        ),
        notes=notes or {},
    )


def _present_name_surfaces(
    views: Mapping[str, SurfaceView],
) -> list[SurfaceView]:
    name_surfaces = (
        ParitySurface.PYTHON_SIGNATURE,
        ParitySurface.REGISTRATION,
        ParitySurface.TOOLS_LIST,
        ParitySurface.JSON_MANIFEST,
        ParitySurface.TYPESCRIPT_SDK,
        ParitySurface.SWISSKNIFE_CONNECTOR,
        ParitySurface.IMPLEMENTATION_TARGET,
    )
    return [
        views[s.value]
        for s in name_surfaces
        if s.value in views and views[s.value].present and views[s.value].tool_name
    ]


def text_names_agree(views: Mapping[str, SurfaceView]) -> bool:
    """True when all present name-bearing surfaces share an alias-compatible name."""

    present = _present_name_surfaces(views)
    if len(present) < 2:
        return len(present) == 1
    alias_sets = [set(tool_name_aliases(v.tool_name)) for v in present]
    shared = set.intersection(*alias_sets) if alias_sets else set()
    return bool(shared)


def call_path_is_proved(path: MCPlusPlusCallPath) -> bool:
    """Whether a path is a fully resolved invocation (not text-only)."""

    if path.verdict is not PathVerdict.PROVED:
        return False
    stage_values = {hop.stage.value if hasattr(hop.stage, "value") else str(hop.stage) for hop in path.hops}
    # All required stages must be present and resolved.
    for stage in REQUIRED_PROVED_STAGES:
        if stage not in stage_values:
            return False
    for hop in path.hops:
        stage = hop.stage.value if hasattr(hop.stage, "value") else str(hop.stage)
        if stage == "caller":
            continue
        if hop.status is not ResolverStatus.RESOLVED_STATIC:
            return False
        if hop.reason_code in _NON_INVOCATION_REASONS:
            return False
    return True


def _paths_for_tool(
    paths: Sequence[MCPlusPlusCallPath],
    tool_name: str,
) -> tuple[MCPlusPlusCallPath, ...]:
    key = _tool_key(tool_name)
    aliases = set(tool_name_aliases(tool_name))
    aliases.add(key)
    matched = [
        path
        for path in paths
        if _tool_key(path.tool_name) in aliases or path.tool_name in aliases
    ]
    return tuple(sorted(matched, key=lambda p: getattr(p, "path_name", "") or p.tool_name))


def compare_tool_surfaces(
    tool_name: str,
    views: Mapping[str, SurfaceView],
    *,
    paths: Sequence[MCPlusPlusCallPath] = (),
    drift_witnesses: Sequence[ManifestDriftWitness] = (),
) -> ToolParityResult:
    """Compare all surface views for one tool and emit findings + verdict."""

    findings: list[ParityFinding] = []
    tool_paths = _paths_for_tool(paths, tool_name)
    path_ids = tuple(
        getattr(p, "path_id", "")
        or getattr(p, "content_id", "")
        or p.path_name
        for p in tool_paths
    )
    path_verdicts = tuple(p.verdict.value for p in tool_paths)
    proved = any(call_path_is_proved(p) for p in tool_paths)
    names_agree = text_names_agree(views)

    reg = views.get(ParitySurface.REGISTRATION.value) or SurfaceView.absent(
        ParitySurface.REGISTRATION, tool_name=tool_name
    )
    listed = views.get(ParitySurface.TOOLS_LIST.value) or SurfaceView.absent(
        ParitySurface.TOOLS_LIST, tool_name=tool_name
    )
    manifest = views.get(ParitySurface.JSON_MANIFEST.value) or SurfaceView.absent(
        ParitySurface.JSON_MANIFEST, tool_name=tool_name
    )
    sdk = views.get(ParitySurface.TYPESCRIPT_SDK.value) or SurfaceView.absent(
        ParitySurface.TYPESCRIPT_SDK, tool_name=tool_name
    )
    connector = views.get(
        ParitySurface.SWISSKNIFE_CONNECTOR.value
    ) or SurfaceView.absent(ParitySurface.SWISSKNIFE_CONNECTOR, tool_name=tool_name)
    transport = views.get(
        ParitySurface.TRANSPORT_PROFILE.value
    ) or SurfaceView.absent(ParitySurface.TRANSPORT_PROFILE, tool_name=tool_name)
    result_err = views.get(
        ParitySurface.RESULT_ERROR_MAP.value
    ) or SurfaceView.absent(ParitySurface.RESULT_ERROR_MAP, tool_name=tool_name)
    py_sig = views.get(
        ParitySurface.PYTHON_SIGNATURE.value
    ) or SurfaceView.absent(ParitySurface.PYTHON_SIGNATURE, tool_name=tool_name)
    impl = views.get(
        ParitySurface.IMPLEMENTATION_TARGET.value
    ) or SurfaceView.absent(ParitySurface.IMPLEMENTATION_TARGET, tool_name=tool_name)
    cap = views.get(
        ParitySurface.CAPABILITY_DEGRADATION.value
    ) or SurfaceView.absent(ParitySurface.CAPABILITY_DEGRADATION, tool_name=tool_name)
    runtime = views.get(
        ParitySurface.RUNTIME_WITNESS.value
    ) or SurfaceView.absent(ParitySurface.RUNTIME_WITNESS, tool_name=tool_name)

    primary_path_id = path_ids[0] if path_ids else ""
    primary_path_verdict = path_verdicts[0] if path_verdicts else ""

    # --- Missing registration ---
    if (listed.present or manifest.present or sdk.present) and not reg.present:
        left = listed if listed.present else (manifest if manifest.present else sdk)
        findings.append(
            _make_finding(
                ParityFindingKind.MISSING_REGISTRATION,
                tool_name,
                f"tool {tool_name!r} appears on {left.surface.value} but has no registration",
                (
                    _pair_witness(
                        ParityFindingKind.MISSING_REGISTRATION,
                        tool_name,
                        left,
                        reg,
                        left_value=left.tool_name or "present",
                        right_value="absent",
                        path_id=primary_path_id,
                        path_verdict=primary_path_verdict,
                    ),
                ),
                surfaces=(left.surface, ParitySurface.REGISTRATION),
                path_ids=path_ids,
            )
        )

    # --- Extra unreachable registration ---
    if reg.present and not listed.present and not proved:
        # Registration without tools/list and without proved path is unreachable.
        findings.append(
            _make_finding(
                ParityFindingKind.EXTRA_UNREACHABLE_TOOL,
                tool_name,
                f"registration for {tool_name!r} is not listed and has no proved call path",
                (
                    _pair_witness(
                        ParityFindingKind.EXTRA_UNREACHABLE_TOOL,
                        tool_name,
                        reg,
                        listed,
                        left_value=reg.qualified_name or reg.tool_name,
                        right_value="absent",
                        path_id=primary_path_id,
                        path_verdict=primary_path_verdict or "none",
                    ),
                ),
                surfaces=(ParitySurface.REGISTRATION, ParitySurface.TOOLS_LIST),
                path_ids=path_ids,
            )
        )

    # --- Schema mismatches between registration / tools_list / manifest / sdk / python ---
    schema_pairs = (
        (py_sig, reg),
        (reg, listed),
        (listed, manifest),
        (listed, sdk),
        (reg, sdk),
        (py_sig, listed),
    )
    for left, right in schema_pairs:
        if not (left.present and right.present):
            continue
        if (
            left.input_schema_fingerprint
            and right.input_schema_fingerprint
            and left.input_schema_fingerprint != right.input_schema_fingerprint
            and not left.input_schema_fingerprint.startswith("multi:")
            and not right.input_schema_fingerprint.startswith("multi:")
        ):
            findings.append(
                _make_finding(
                    ParityFindingKind.SCHEMA_MISMATCH,
                    tool_name,
                    (
                        f"input schema fingerprint differs between "
                        f"{left.surface.value} and {right.surface.value}"
                    ),
                    (
                        _pair_witness(
                            ParityFindingKind.SCHEMA_MISMATCH,
                            tool_name,
                            left,
                            right,
                            left_value=left.input_schema_fingerprint,
                            right_value=right.input_schema_fingerprint,
                            path_id=primary_path_id,
                            path_verdict=primary_path_verdict,
                            notes={"schema_side": "input"},
                        ),
                    ),
                    surfaces=(left.surface, right.surface),
                    path_ids=path_ids,
                )
            )
        if (
            left.output_schema_fingerprint
            and right.output_schema_fingerprint
            and left.output_schema_fingerprint != right.output_schema_fingerprint
            and not left.output_schema_fingerprint.startswith("multi:")
            and not right.output_schema_fingerprint.startswith("multi:")
        ):
            findings.append(
                _make_finding(
                    ParityFindingKind.SCHEMA_MISMATCH,
                    tool_name,
                    (
                        f"output schema fingerprint differs between "
                        f"{left.surface.value} and {right.surface.value}"
                    ),
                    (
                        _pair_witness(
                            ParityFindingKind.SCHEMA_MISMATCH,
                            tool_name,
                            left,
                            right,
                            left_value=left.output_schema_fingerprint,
                            right_value=right.output_schema_fingerprint,
                            path_id=primary_path_id,
                            path_verdict=primary_path_verdict,
                            notes={"schema_side": "output"},
                        ),
                    ),
                    surfaces=(left.surface, right.surface),
                    path_ids=path_ids,
                )
            )

    # --- Error / result map mismatches ---
    if reg.present and result_err.present and result_err.error_codes:
        if reg.error_codes and set(reg.error_codes) != set(result_err.error_codes):
            findings.append(
                _make_finding(
                    ParityFindingKind.ERROR_MAP_MISMATCH,
                    tool_name,
                    f"error codes differ between registration and result/error map for {tool_name!r}",
                    (
                        _pair_witness(
                            ParityFindingKind.ERROR_MAP_MISMATCH,
                            tool_name,
                            reg,
                            result_err,
                            left_value=",".join(reg.error_codes),
                            right_value=",".join(result_err.error_codes),
                            path_id=primary_path_id,
                            path_verdict=primary_path_verdict,
                        ),
                    ),
                    surfaces=(
                        ParitySurface.REGISTRATION,
                        ParitySurface.RESULT_ERROR_MAP,
                    ),
                    path_ids=path_ids,
                )
            )
    if listed.present and reg.present:
        if listed.error_codes and reg.error_codes and set(listed.error_codes) != set(
            reg.error_codes
        ):
            findings.append(
                _make_finding(
                    ParityFindingKind.ERROR_MAP_MISMATCH,
                    tool_name,
                    f"error codes differ between tools/list and registration for {tool_name!r}",
                    (
                        _pair_witness(
                            ParityFindingKind.ERROR_MAP_MISMATCH,
                            tool_name,
                            listed,
                            reg,
                            left_value=",".join(listed.error_codes),
                            right_value=",".join(reg.error_codes),
                            path_id=primary_path_id,
                            path_verdict=primary_path_verdict,
                        ),
                    ),
                    surfaces=(ParitySurface.TOOLS_LIST, ParitySurface.REGISTRATION),
                    path_ids=path_ids,
                )
            )

    # --- Stale generated artifacts (version drift on generated surfaces) ---
    for gen in (manifest, sdk):
        if not (gen.present and reg.present):
            continue
        if gen.version and reg.version and gen.version != reg.version:
            findings.append(
                _make_finding(
                    ParityFindingKind.STALE_GENERATED_ARTIFACT,
                    tool_name,
                    (
                        f"generated {gen.surface.value} version {gen.version!r} "
                        f"does not match registration version {reg.version!r}"
                    ),
                    (
                        _pair_witness(
                            ParityFindingKind.STALE_GENERATED_ARTIFACT,
                            tool_name,
                            gen,
                            reg,
                            left_value=gen.version,
                            right_value=reg.version,
                            path_id=primary_path_id,
                            path_verdict=primary_path_verdict,
                        ),
                    ),
                    surfaces=(gen.surface, ParitySurface.REGISTRATION),
                    path_ids=path_ids,
                )
            )
        # Schema drift against registration also marks generated stale when generated.
        if (
            gen.is_generated
            and gen.input_schema_fingerprint
            and reg.input_schema_fingerprint
            and gen.input_schema_fingerprint != reg.input_schema_fingerprint
            and not gen.input_schema_fingerprint.startswith("multi:")
            and not reg.input_schema_fingerprint.startswith("multi:")
        ):
            # Already covered by SCHEMA_MISMATCH; also flag stale when versions equal
            # or missing so callers can filter generated-only drift.
            if not gen.version or not reg.version or gen.version == reg.version:
                findings.append(
                    _make_finding(
                        ParityFindingKind.STALE_GENERATED_ARTIFACT,
                        tool_name,
                        (
                            f"generated {gen.surface.value} schema is stale relative "
                            f"to registration for {tool_name!r}"
                        ),
                        (
                            _pair_witness(
                                ParityFindingKind.STALE_GENERATED_ARTIFACT,
                                tool_name,
                                gen,
                                reg,
                                left_value=gen.input_schema_fingerprint,
                                right_value=reg.input_schema_fingerprint,
                                path_id=primary_path_id,
                                path_verdict=primary_path_verdict,
                                notes={"stale_reason": "schema"},
                            ),
                        ),
                        surfaces=(gen.surface, ParitySurface.REGISTRATION),
                        path_ids=path_ids,
                    )
                )

    # --- Wrong aliases ---
    if sdk.present and sdk.alias_of and reg.present:
        reg_aliases = set(tool_name_aliases(reg.tool_name))
        reg_aliases.add(_tool_key(reg.tool_name))
        alias_key = _tool_key(sdk.alias_of)
        alias_binds = (
            alias_key in reg_aliases
            or sdk.alias_of in reg_aliases
            or bool(set(tool_name_aliases(sdk.alias_of)) & reg_aliases)
        )
        if not alias_binds:
            findings.append(
                _make_finding(
                    ParityFindingKind.WRONG_ALIAS,
                    tool_name,
                    f"SDK alias {sdk.alias_of!r} does not bind to registration {reg.tool_name!r}",
                    (
                        _pair_witness(
                            ParityFindingKind.WRONG_ALIAS,
                            tool_name,
                            sdk,
                            reg,
                            left_value=sdk.alias_of,
                            right_value=reg.tool_name,
                            path_id=primary_path_id,
                            path_verdict=primary_path_verdict,
                        ),
                    ),
                    surfaces=(
                        ParitySurface.TYPESCRIPT_SDK,
                        ParitySurface.REGISTRATION,
                    ),
                    path_ids=path_ids,
                )
            )

    # --- Name mismatches across present surfaces ---
    present_named = _present_name_surfaces(views)
    if len(present_named) >= 2 and not names_agree:
        left, right = present_named[0], present_named[1]
        findings.append(
            _make_finding(
                ParityFindingKind.NAME_MISMATCH,
                tool_name,
                (
                    f"tool names disagree across surfaces "
                    f"({left.tool_name!r} vs {right.tool_name!r})"
                ),
                (
                    _pair_witness(
                        ParityFindingKind.NAME_MISMATCH,
                        tool_name,
                        left,
                        right,
                        left_value=left.tool_name,
                        right_value=right.tool_name,
                        path_id=primary_path_id,
                        path_verdict=primary_path_verdict,
                    ),
                ),
                surfaces=tuple(v.surface for v in present_named),
                path_ids=path_ids,
            )
        )

    # --- Python signature vs registration name / schema ---
    if py_sig.present and reg.present:
        if py_sig.tool_name and reg.tool_name:
            if not (
                set(tool_name_aliases(py_sig.tool_name))
                & set(tool_name_aliases(reg.tool_name))
            ):
                findings.append(
                    _make_finding(
                        ParityFindingKind.PYTHON_SIGNATURE_MISMATCH,
                        tool_name,
                        "Python signature name does not alias-match registration",
                        (
                            _pair_witness(
                                ParityFindingKind.PYTHON_SIGNATURE_MISMATCH,
                                tool_name,
                                py_sig,
                                reg,
                                left_value=py_sig.tool_name,
                                right_value=reg.tool_name,
                                path_id=primary_path_id,
                                path_verdict=primary_path_verdict,
                            ),
                        ),
                        surfaces=(
                            ParitySurface.PYTHON_SIGNATURE,
                            ParitySurface.REGISTRATION,
                        ),
                        path_ids=path_ids,
                    )
                )

    # --- Transport / profile ---
    if transport.present and connector.present:
        if (
            transport.transport is not TransportKind.UNKNOWN
            and connector.transport is not TransportKind.UNKNOWN
            and transport.transport is not connector.transport
        ):
            findings.append(
                _make_finding(
                    ParityFindingKind.TRANSPORT_MISMATCH,
                    tool_name,
                    "transport disagrees between transport profile and connector",
                    (
                        _pair_witness(
                            ParityFindingKind.TRANSPORT_MISMATCH,
                            tool_name,
                            transport,
                            connector,
                            left_value=transport.transport.value,
                            right_value=connector.transport.value,
                            path_id=primary_path_id,
                            path_verdict=primary_path_verdict,
                        ),
                    ),
                    surfaces=(
                        ParitySurface.TRANSPORT_PROFILE,
                        ParitySurface.SWISSKNIFE_CONNECTOR,
                    ),
                    path_ids=path_ids,
                )
            )
        if transport.profiles and connector.profiles:
            if not set(transport.profiles) & set(connector.profiles):
                findings.append(
                    _make_finding(
                        ParityFindingKind.PROFILE_MISMATCH,
                        tool_name,
                        "no overlapping MCP++ profiles between transport and connector",
                        (
                            _pair_witness(
                                ParityFindingKind.PROFILE_MISMATCH,
                                tool_name,
                                transport,
                                connector,
                                left_value=",".join(transport.profiles),
                                right_value=",".join(connector.profiles),
                                path_id=primary_path_id,
                                path_verdict=primary_path_verdict,
                            ),
                        ),
                        surfaces=(
                            ParitySurface.TRANSPORT_PROFILE,
                            ParitySurface.SWISSKNIFE_CONNECTOR,
                        ),
                        path_ids=path_ids,
                    )
                )

    # --- Implementation target consistency ---
    if impl.present and reg.present:
        reg_target = str(
            (reg.implementation_target or reg.notes.get("implementation") or "")
        )
        # Prefer record-backed target from notes already folded into view.
        if (
            impl.implementation_target
            and reg.implementation_target
            and impl.implementation_target != reg.implementation_target
            and not impl.implementation_target.startswith("multi:")
            and not reg.implementation_target.startswith("multi:")
        ):
            findings.append(
                _make_finding(
                    ParityFindingKind.IMPLEMENTATION_TARGET_MISMATCH,
                    tool_name,
                    "implementation target disagrees between implementation and registration",
                    (
                        _pair_witness(
                            ParityFindingKind.IMPLEMENTATION_TARGET_MISMATCH,
                            tool_name,
                            impl,
                            reg,
                            left_value=impl.implementation_target,
                            right_value=reg.implementation_target,
                            path_id=primary_path_id,
                            path_verdict=primary_path_verdict,
                        ),
                    ),
                    surfaces=(
                        ParitySurface.IMPLEMENTATION_TARGET,
                        ParitySurface.REGISTRATION,
                    ),
                    path_ids=path_ids,
                )
            )
        del reg_target  # unused; kept for clarity in reviews

    # --- Direct local bypass ---
    if impl.is_local_bypass or any(
        v.is_local_bypass for v in views.values() if v.present
    ):
        findings.append(
            _make_finding(
                ParityFindingKind.DIRECT_LOCAL_BYPASS,
                tool_name,
                f"tool {tool_name!r} resolves through a direct local bypass / same-name helper",
                (
                    _pair_witness(
                        ParityFindingKind.DIRECT_LOCAL_BYPASS,
                        tool_name,
                        impl if impl.present else reg,
                        connector if connector.present else listed,
                        left_value=impl.qualified_name or "local_bypass",
                        right_value=connector.qualified_name or "no_mcp_path",
                        path_id=primary_path_id,
                        path_verdict=primary_path_verdict or "rejected",
                        notes={"local_bypass": True},
                    ),
                ),
                surfaces=(
                    ParitySurface.IMPLEMENTATION_TARGET,
                    ParitySurface.SWISSKNIFE_CONNECTOR,
                ),
                path_ids=path_ids,
            )
        )

    # --- Mock / fallback dispatch ---
    if impl.is_mock_or_fallback or any(
        v.is_mock_or_fallback
        for k, v in views.items()
        if v.present and k != ParitySurface.RUNTIME_WITNESS.value
    ):
        findings.append(
            _make_finding(
                ParityFindingKind.MOCK_FALLBACK_DISPATCH,
                tool_name,
                f"tool {tool_name!r} dispatches to mock/fallback/test-server target",
                (
                    _pair_witness(
                        ParityFindingKind.MOCK_FALLBACK_DISPATCH,
                        tool_name,
                        impl if impl.present else reg,
                        listed if listed.present else reg,
                        left_value=impl.qualified_name or "mock_or_fallback",
                        right_value=listed.tool_name or reg.tool_name or tool_name,
                        path_id=primary_path_id,
                        path_verdict=primary_path_verdict or "rejected",
                        notes={"mock_or_fallback": True},
                    ),
                ),
                surfaces=(
                    ParitySurface.IMPLEMENTATION_TARGET,
                    ParitySurface.REGISTRATION,
                ),
                path_ids=path_ids,
            )
        )

    # --- Runtime mock presented as authority ---
    if runtime.present and runtime.is_mock_or_fallback:
        auth = bool(runtime.notes.get("authoritative"))
        if auth or str(runtime.notes.get("outcome") or "") == WitnessOutcome.PASSED.value:
            findings.append(
                _make_finding(
                    ParityFindingKind.RUNTIME_MOCK_AUTHORITY,
                    tool_name,
                    f"runtime witness for {tool_name!r} is mock-class and cannot grant parity authority",
                    (
                        _pair_witness(
                            ParityFindingKind.RUNTIME_MOCK_AUTHORITY,
                            tool_name,
                            runtime,
                            impl if impl.present else reg,
                            left_value=str(runtime.notes.get("implementation_kind") or "mock"),
                            right_value=impl.implementation_target or "unknown",
                            path_id=primary_path_id,
                            path_verdict=primary_path_verdict,
                        ),
                    ),
                    surfaces=(
                        ParitySurface.RUNTIME_WITNESS,
                        ParitySurface.IMPLEMENTATION_TARGET,
                    ),
                    path_ids=path_ids,
                    confidence=0,
                )
            )

    # --- Capability / degradation claims ---
    if cap.present and reg.present:
        # If registration claims capabilities that connector/runtime lack, warn.
        if connector.present and cap.capability_claims and connector.profiles:
            # Soft check: capability claims referencing profiles must intersect.
            claim_profiles = {
                c.split(":", 1)[-1]
                for c in cap.capability_claims
                if c.startswith("profile:") or c.startswith("mcp++/")
            }
            if claim_profiles and not (claim_profiles & set(connector.profiles)):
                findings.append(
                    _make_finding(
                        ParityFindingKind.CAPABILITY_CLAIM_MISMATCH,
                        tool_name,
                        "capability claims do not intersect connector profiles",
                        (
                            _pair_witness(
                                ParityFindingKind.CAPABILITY_CLAIM_MISMATCH,
                                tool_name,
                                cap,
                                connector,
                                left_value=",".join(sorted(claim_profiles)),
                                right_value=",".join(connector.profiles),
                                path_id=primary_path_id,
                                path_verdict=primary_path_verdict,
                            ),
                        ),
                        surfaces=(
                            ParitySurface.CAPABILITY_DEGRADATION,
                            ParitySurface.SWISSKNIFE_CONNECTOR,
                        ),
                        path_ids=path_ids,
                        confidence=70,
                    )
                )
        silent = {
            c.lower()
            for c in cap.degradation_claims
            if "silent" in c.lower() or c.lower() in {"swallow", "placeholder_success"}
        }
        if silent:
            findings.append(
                _make_finding(
                    ParityFindingKind.DEGRADATION_CLAIM_MISMATCH,
                    tool_name,
                    "degradation claims include silent/placeholder success (forbidden)",
                    (
                        _pair_witness(
                            ParityFindingKind.DEGRADATION_CLAIM_MISMATCH,
                            tool_name,
                            cap,
                            reg,
                            left_value=",".join(sorted(silent)),
                            right_value="explicit_degradation_required",
                            path_id=primary_path_id,
                            path_verdict=primary_path_verdict,
                        ),
                    ),
                    surfaces=(
                        ParitySurface.CAPABILITY_DEGRADATION,
                        ParitySurface.REGISTRATION,
                    ),
                    path_ids=path_ids,
                )
            )

    # --- Ambiguous / rejected paths ---
    for path in tool_paths:
        if path.verdict is PathVerdict.AMBIGUOUS:
            findings.append(
                _make_finding(
                    ParityFindingKind.AMBIGUOUS_PATH,
                    tool_name,
                    f"call path {path.path_name!r} is ambiguous",
                    (
                        ParityWitness(
                            kind=ParityFindingKind.AMBIGUOUS_PATH,
                            tool_name=tool_name,
                            left_surface=ParitySurface.SWISSKNIFE_CONNECTOR,
                            right_surface=ParitySurface.IMPLEMENTATION_TARGET,
                            left_value=path.connector_ref or path.caller_ref,
                            right_value=path.implementation_ref or "",
                            left_ref=path.connector_ref,
                            right_ref=path.implementation_ref,
                            path_id=path.path_name,
                            path_verdict=path.verdict.value,
                            evidence_refs=tuple(
                                aid
                                for hop in path.hops
                                for aid in getattr(hop, "artifact_ids", ()) or ()
                            ),
                            notes={"path_name": path.path_name},
                        ),
                    ),
                    surfaces=(
                        ParitySurface.SWISSKNIFE_CONNECTOR,
                        ParitySurface.IMPLEMENTATION_TARGET,
                    ),
                    path_ids=(path.path_name,),
                    confidence=25,
                )
            )
        if path.verdict is PathVerdict.REJECTED:
            # Non-invocation already covered by mock/bypass when markers present;
            # still record ambiguous-style rejected path if no other finding.
            reasons = {
                hop.reason_code
                for hop in path.hops
                if hop.reason_code in _NON_INVOCATION_REASONS
            }
            if reasons & {
                ReasonCode.MOCK_IMPLEMENTATION,
                ReasonCode.LEGACY_FALLBACK,
                ReasonCode.TEST_SERVER,
            } and not any(
                f.kind is ParityFindingKind.MOCK_FALLBACK_DISPATCH for f in findings
            ):
                findings.append(
                    _make_finding(
                        ParityFindingKind.MOCK_FALLBACK_DISPATCH,
                        tool_name,
                        f"call path {path.path_name!r} rejected as mock/fallback",
                        (
                            ParityWitness(
                                kind=ParityFindingKind.MOCK_FALLBACK_DISPATCH,
                                tool_name=tool_name,
                                left_surface=ParitySurface.IMPLEMENTATION_TARGET,
                                right_surface=ParitySurface.REGISTRATION,
                                left_value=",".join(sorted(r.value for r in reasons)),
                                right_value=path.verdict.value,
                                path_id=path.path_name,
                                path_verdict=path.verdict.value,
                                evidence_refs=(),
                            ),
                        ),
                        surfaces=(ParitySurface.IMPLEMENTATION_TARGET,),
                        path_ids=(path.path_name,),
                        confidence=0,
                    )
                )
            if ReasonCode.SAME_NAME_HELPER in reasons and not any(
                f.kind is ParityFindingKind.DIRECT_LOCAL_BYPASS for f in findings
            ):
                findings.append(
                    _make_finding(
                        ParityFindingKind.DIRECT_LOCAL_BYPASS,
                        tool_name,
                        f"call path {path.path_name!r} rejected as same-name local helper",
                        (
                            ParityWitness(
                                kind=ParityFindingKind.DIRECT_LOCAL_BYPASS,
                                tool_name=tool_name,
                                left_surface=ParitySurface.IMPLEMENTATION_TARGET,
                                right_surface=ParitySurface.SWISSKNIFE_CONNECTOR,
                                left_value=ReasonCode.SAME_NAME_HELPER.value,
                                right_value=path.connector_ref or "",
                                path_id=path.path_name,
                                path_verdict=path.verdict.value,
                            ),
                        ),
                        surfaces=(ParitySurface.IMPLEMENTATION_TARGET,),
                        path_ids=(path.path_name,),
                        confidence=0,
                    )
                )

    # --- Resolver drift witnesses (consume, do not re-derive manifests) ---
    for drift in drift_witnesses:
        if _tool_key(drift.tool_name) not in {
            _tool_key(tool_name),
            *{_tool_key(a) for a in tool_name_aliases(tool_name)},
        } and drift.tool_name not in tool_name_aliases(tool_name):
            if drift.tool_name and _tool_key(drift.tool_name) != _tool_key(tool_name):
                continue
        kind = _DRIFT_TO_FINDING.get(drift.drift_kind)
        if kind is None:
            continue
        # Avoid duplicate kind+tool+values already present.
        already = any(
            f.kind is kind
            and any(
                w.left_value == drift.left_value and w.right_value == drift.right_value
                for w in f.witnesses
            )
            for f in findings
        )
        if already:
            continue
        left_surface = ParitySurface.JSON_MANIFEST
        right_surface = ParitySurface.REGISTRATION
        if drift.drift_kind is DriftKind.SCHEMA_MISMATCH:
            left_surface = ParitySurface.TOOLS_LIST
        elif drift.drift_kind is DriftKind.TRANSPORT_MISMATCH:
            left_surface = ParitySurface.TRANSPORT_PROFILE
            right_surface = ParitySurface.SWISSKNIFE_CONNECTOR
        elif drift.drift_kind is DriftKind.ERROR_MAP_MISMATCH:
            left_surface = ParitySurface.RESULT_ERROR_MAP
        elif drift.drift_kind is DriftKind.RESULT_MAP_MISMATCH:
            left_surface = ParitySurface.RESULT_ERROR_MAP
        elif drift.drift_kind is DriftKind.MISSING_REGISTRATION:
            left_surface = ParitySurface.TOOLS_LIST
        elif drift.drift_kind is DriftKind.EXTRA_UNREACHABLE:
            left_surface = ParitySurface.REGISTRATION
            right_surface = ParitySurface.TOOLS_LIST
        findings.append(
            _make_finding(
                kind,
                tool_name,
                (
                    f"resolver drift {drift.drift_kind.value} for {tool_name!r}: "
                    f"{drift.left_ref} vs {drift.right_ref}"
                ),
                (
                    ParityWitness(
                        kind=kind,
                        tool_name=tool_name,
                        left_surface=left_surface,
                        right_surface=right_surface,
                        left_value=drift.left_value,
                        right_value=drift.right_value,
                        left_ref=drift.left_ref,
                        right_ref=drift.right_ref,
                        path_id=primary_path_id,
                        path_verdict=primary_path_verdict,
                        evidence_refs=tuple(
                            getattr(ev, "evidence_id", "")
                            or getattr(ev, "content_id", "")
                            or ""
                            for ev in drift.evidence
                        ),
                        notes={"drift_kind": drift.drift_kind.value},
                    ),
                ),
                surfaces=(left_surface, right_surface),
                path_ids=path_ids,
                notes={"source": "mcplusplus_contract_resolver"},
            )
        )

    # --- Same text without resolved call path is insufficient ---
    if names_agree and not proved:
        # Only emit when at least two name-bearing surfaces agree (text match).
        if len(present_named) >= 2:
            findings.append(
                _make_finding(
                    ParityFindingKind.MISSING_RESOLVED_CALL_PATH,
                    tool_name,
                    (
                        f"surfaces agree on name {tool_name!r} but no proved "
                        f"caller→connector→…→implementation call path is bound"
                    ),
                    (
                        _pair_witness(
                            ParityFindingKind.MISSING_RESOLVED_CALL_PATH,
                            tool_name,
                            present_named[0],
                            present_named[1],
                            left_value=present_named[0].tool_name,
                            right_value=present_named[1].tool_name,
                            path_id=primary_path_id,
                            path_verdict=primary_path_verdict or "none",
                            notes={
                                "text_names_agree": True,
                                "proved_call_path": False,
                                "rule": "same_text_without_resolved_call_path_insufficient",
                            },
                        ),
                    ),
                    surfaces=tuple(v.surface for v in present_named),
                    path_ids=path_ids,
                    confidence=100,
                )
            )

    # Deduplicate findings by finding_id.
    unique: dict[str, ParityFinding] = {}
    for item in findings:
        unique[item.finding_id] = item
    findings_t = tuple(
        sorted(unique.values(), key=lambda f: (f.kind.value, f.finding_id))
    )

    # Verdict aggregation (fail-closed).
    kinds = {f.kind for f in findings_t}
    if any(p.verdict is PathVerdict.AMBIGUOUS for p in tool_paths) or (
        ParityFindingKind.AMBIGUOUS_PATH in kinds
    ):
        # Ambiguity may coexist with drift; prefer ambiguous when present.
        if findings_t and kinds - {ParityFindingKind.AMBIGUOUS_PATH}:
            # Still report drift, but mark ambiguous if no hard reject.
            if any(
                k
                in {
                    ParityFindingKind.MOCK_FALLBACK_DISPATCH,
                    ParityFindingKind.DIRECT_LOCAL_BYPASS,
                }
                for k in kinds
            ):
                verdict = ToolParityVerdict.REJECTED
            else:
                verdict = ToolParityVerdict.AMBIGUOUS
        else:
            verdict = ToolParityVerdict.AMBIGUOUS
    elif any(p.verdict is PathVerdict.EXTERNAL for p in tool_paths) and not findings_t:
        verdict = ToolParityVerdict.EXTERNAL
    elif any(
        k
        in {
            ParityFindingKind.MOCK_FALLBACK_DISPATCH,
            ParityFindingKind.DIRECT_LOCAL_BYPASS,
            ParityFindingKind.RUNTIME_MOCK_AUTHORITY,
        }
        for k in kinds
    ):
        verdict = ToolParityVerdict.REJECTED
    elif kinds - {
        ParityFindingKind.MISSING_RESOLVED_CALL_PATH,
        ParityFindingKind.CONTRACT_PACK_GAP,
    }:
        # Concrete surface drift outranks the path-insufficiency residual.
        verdict = ToolParityVerdict.WITNESSED_DRIFT
    elif ParityFindingKind.MISSING_RESOLVED_CALL_PATH in kinds and not proved:
        verdict = ToolParityVerdict.INSUFFICIENT_PATH
    elif findings_t:
        verdict = ToolParityVerdict.WITNESSED_DRIFT
    elif proved and not findings_t:
        verdict = ToolParityVerdict.PROVED_PARITY
    elif any(p.verdict is PathVerdict.PROVED for p in tool_paths) and not findings_t:
        verdict = ToolParityVerdict.PROVED_PARITY
    elif not any(v.present for v in views.values()):
        verdict = ToolParityVerdict.UNKNOWN
    else:
        verdict = ToolParityVerdict.UNKNOWN

    # Final fail-closed guard: never emit proved_parity without proved path.
    if verdict is ToolParityVerdict.PROVED_PARITY and not proved:
        verdict = ToolParityVerdict.INSUFFICIENT_PATH
        if ParityFindingKind.MISSING_RESOLVED_CALL_PATH not in kinds:
            findings_t = findings_t + (
                _make_finding(
                    ParityFindingKind.MISSING_RESOLVED_CALL_PATH,
                    tool_name,
                    "refusing proved_parity without resolved call path",
                    (
                        ParityWitness(
                            kind=ParityFindingKind.MISSING_RESOLVED_CALL_PATH,
                            tool_name=tool_name,
                            left_surface=ParitySurface.REGISTRATION,
                            right_surface=ParitySurface.IMPLEMENTATION_TARGET,
                            left_value=reg.tool_name or tool_name,
                            right_value=impl.tool_name or "",
                            path_verdict="none",
                            notes={
                                "rule": "same_text_without_resolved_call_path_insufficient"
                            },
                        ),
                    ),
                    surfaces=(
                        ParitySurface.REGISTRATION,
                        ParitySurface.IMPLEMENTATION_TARGET,
                    ),
                ),
            )

    return ToolParityResult(
        tool_name=tool_name,
        verdict=verdict,
        surfaces={k: v for k, v in views.items()},
        findings=findings_t,
        path_ids=path_ids,
        path_verdicts=path_verdicts,
        proved_call_path=proved,
        text_names_agree=names_agree,
        notes={
            "path_count": len(tool_paths),
            "finding_count": len(findings_t),
        },
    )


def _aggregate_report_verdict(
    tools: Sequence[ToolParityResult],
) -> ReportVerdict:
    if not tools:
        return ReportVerdict.EMPTY
    verdicts = {item.verdict for item in tools}
    if ToolParityVerdict.WITNESSED_DRIFT in verdicts or ToolParityVerdict.REJECTED in verdicts:
        return ReportVerdict.HAS_DRIFT
    if ToolParityVerdict.INSUFFICIENT_PATH in verdicts:
        return ReportVerdict.HAS_INSUFFICIENT_PATH
    if ToolParityVerdict.AMBIGUOUS in verdicts:
        return ReportVerdict.HAS_AMBIGUOUS
    if verdicts == {ToolParityVerdict.PROVED_PARITY}:
        return ReportVerdict.ALL_PROVED
    if ToolParityVerdict.PROVED_PARITY in verdicts and verdicts <= {
        ToolParityVerdict.PROVED_PARITY,
        ToolParityVerdict.EXTERNAL,
    }:
        return ReportVerdict.ALL_PROVED
    return ReportVerdict.UNKNOWN


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


class VfsMcpContractChecker:
    """End-to-end VFS manifest / SDK / MCP / MCP++ parity checker."""

    def __init__(
        self,
        inventory: MCPlusPlusInventory,
        *,
        contract_pack: VfsContractPack | None = None,
        max_tools: int = DEFAULT_MAX_TOOLS,
        max_findings: int = DEFAULT_MAX_FINDINGS,
    ) -> None:
        if not isinstance(inventory, MCPlusPlusInventory):
            raise VfsMcpCheckerError("inventory must be MCPlusPlusInventory")
        if (
            isinstance(max_tools, bool)
            or not isinstance(max_tools, int)
            or max_tools < 1
            or max_tools > DEFAULT_MAX_TOOLS
        ):
            raise VfsMcpCheckerBoundsError("max_tools out of bounds")
        if (
            isinstance(max_findings, bool)
            or not isinstance(max_findings, int)
            or max_findings < 1
            or max_findings > DEFAULT_MAX_FINDINGS
        ):
            raise VfsMcpCheckerBoundsError("max_findings out of bounds")
        self._inventory = inventory
        self._contract_pack = contract_pack
        self._max_tools = max_tools
        self._max_findings = max_findings

    @property
    def inventory(self) -> MCPlusPlusInventory:
        return self._inventory

    @property
    def contract_pack(self) -> VfsContractPack | None:
        return self._contract_pack

    def check(
        self,
        *,
        tool_names: Sequence[str] | None = None,
        claims: Sequence[CallPathClaim | Mapping[str, Any]] | None = None,
        resolution: MCPlusPlusResolutionResult | None = None,
        runtime_receipts: Sequence[RuntimeWitnessReceipt] = (),
        server_name: str = "",
        vfs_prefix_only: bool = False,
    ) -> VfsMcpParityReport:
        """Run end-to-end parity checks and return a content-addressed report."""

        paths: tuple[MCPlusPlusCallPath, ...] = ()
        drift: tuple[ManifestDriftWitness, ...] = ()
        inventory_id = self._inventory.inventory_id

        if resolution is not None:
            if not isinstance(resolution, MCPlusPlusResolutionResult):
                raise VfsMcpCheckerError(
                    "resolution must be MCPlusPlusResolutionResult"
                )
            paths = resolution.paths
            drift = resolution.drift_witnesses
            inventory_id = resolution.inventory_id or inventory_id
        elif claims is not None:
            resolver = MCPlusPlusContractResolver(
                self._inventory, max_paths=self._max_tools
            )
            resolution = resolver.resolve(claims)
            paths = resolution.paths
            drift = resolution.drift_witnesses
            inventory_id = resolution.inventory_id or inventory_id
        else:
            # Still run global manifest comparison from the resolver.
            resolver = MCPlusPlusContractResolver(
                self._inventory, max_paths=self._max_tools
            )
            drift = resolver.compare_manifests(server_name=server_name)

        if tool_names is None:
            names = list(
                discover_tool_names(
                    self._inventory,
                    server_name=server_name,
                    vfs_prefix_only=vfs_prefix_only,
                )
            )
            # Include tools only present on resolved paths.
            for path in paths:
                if path.tool_name and path.tool_name not in names:
                    names.append(path.tool_name)
            names = sorted(set(names), key=lambda n: (_tool_key(n), n))
        else:
            names = [
                _text(name, "tool_name") for name in tool_names if str(name or "").strip()
            ]

        truncated = False
        truncation_reason = ""
        if len(names) > self._max_tools:
            names = names[: self._max_tools]
            truncated = True
            truncation_reason = "max_tools"

        tool_results: list[ToolParityResult] = []
        all_findings: list[ParityFinding] = []

        # Contract-pack coverage for MCP / MCP++ surfaces (once per report).
        pack_findings = self._contract_pack_findings()
        all_findings.extend(pack_findings)

        for name in names:
            if len(all_findings) >= self._max_findings:
                truncated = True
                truncation_reason = truncation_reason or "max_findings"
                break
            views = build_surface_views(
                self._inventory,
                name,
                contract_pack=self._contract_pack,
                runtime_receipts=runtime_receipts,
            )
            tool_drift = tuple(
                item
                for item in drift
                if not item.tool_name
                or _tool_key(item.tool_name) == _tool_key(name)
                or item.tool_name in tool_name_aliases(name)
            )
            result = compare_tool_surfaces(
                name,
                views,
                paths=paths,
                drift_witnesses=tool_drift,
            )
            tool_results.append(result)
            for finding in result.findings:
                if len(all_findings) >= self._max_findings:
                    truncated = True
                    truncation_reason = truncation_reason or "max_findings"
                    break
                all_findings.append(finding)

        # Deduplicate findings.
        unique_findings: dict[str, ParityFinding] = {}
        for item in all_findings:
            unique_findings[item.finding_id] = item
        findings_t = tuple(
            sorted(
                unique_findings.values(),
                key=lambda f: (f.kind.value, f.tool_name, f.finding_id),
            )
        )
        tools_t = tuple(
            sorted(tool_results, key=lambda t: (_tool_key(t.tool_name), t.tool_name))
        )

        pack_id = ""
        if self._contract_pack is not None:
            pack_id = str(getattr(self._contract_pack, "content_id", "") or "")

        evidence = [
            EVIDENCE_VFS_MCP_PARITY,
            EVIDENCE_VFS_MCP_CALL_PATH,
            EVIDENCE_MANIFEST_PARITY,
        ]
        if runtime_receipts:
            evidence.append(EVIDENCE_RUNTIME_WITNESS)

        return VfsMcpParityReport(
            forest_id=self._inventory.forest_id,
            inventory_id=inventory_id,
            contract_pack_id=pack_id,
            tools=tools_t,
            findings=findings_t,
            verdict=_aggregate_report_verdict(tools_t),
            truncated=truncated,
            truncation_reason=truncation_reason,
            evidence_kinds=tuple(evidence),
            notes={
                "tool_count": len(tools_t),
                "finding_count": len(findings_t),
                "path_count": len(paths),
                "drift_count": len(drift),
                "required_proved_stages": list(REQUIRED_PROVED_STAGES),
                "path_stage_order": list(PATH_STAGE_ORDER),
            },
        )

    def _contract_pack_findings(self) -> list[ParityFinding]:
        if self._contract_pack is None:
            return []
        findings: list[ParityFinding] = []
        for surface in (PublicSurface.MCP, PublicSurface.MCP_PLUS_PLUS):
            try:
                sc = self._contract_pack.surface_contract(surface)
            except Exception:
                findings.append(
                    _make_finding(
                        ParityFindingKind.CONTRACT_PACK_GAP,
                        "",
                        f"contract pack missing surface {surface.value}",
                        (
                            ParityWitness(
                                kind=ParityFindingKind.CONTRACT_PACK_GAP,
                                tool_name="",
                                left_surface=ParitySurface.CONTRACT_PACK,
                                right_surface=ParitySurface.REGISTRATION,
                                left_value=surface.value,
                                right_value="missing",
                            ),
                        ),
                        surfaces=(ParitySurface.CONTRACT_PACK,),
                        confidence=80,
                    )
                )
                continue
            unresolved = list(sc.unresolved_operations)
            if unresolved:
                findings.append(
                    _make_finding(
                        ParityFindingKind.CONTRACT_PACK_GAP,
                        "",
                        (
                            f"contract pack surface {surface.value} has "
                            f"{len(unresolved)} unresolved operations"
                        ),
                        (
                            ParityWitness(
                                kind=ParityFindingKind.CONTRACT_PACK_GAP,
                                tool_name="",
                                left_surface=ParitySurface.CONTRACT_PACK,
                                right_surface=ParitySurface.REGISTRATION,
                                left_value=surface.value,
                                right_value=",".join(
                                    op.value for op in unresolved[:16]
                                ),
                                notes={"unresolved_count": len(unresolved)},
                            ),
                        ),
                        surfaces=(ParitySurface.CONTRACT_PACK,),
                        confidence=60,
                    )
                )
            # Supported operations should eventually have entrypoints; gap if none.
            supported_without = [
                op.operation.value
                for op in sc.operations
                if op.support is OperationSupport.SUPPORTED and not op.entrypoint
            ]
            if supported_without:
                findings.append(
                    _make_finding(
                        ParityFindingKind.CONTRACT_PACK_GAP,
                        "",
                        (
                            f"supported {surface.value} operations lack entrypoints: "
                            f"{', '.join(supported_without[:8])}"
                        ),
                        (
                            ParityWitness(
                                kind=ParityFindingKind.CONTRACT_PACK_GAP,
                                tool_name="",
                                left_surface=ParitySurface.CONTRACT_PACK,
                                right_surface=ParitySurface.REGISTRATION,
                                left_value=surface.value,
                                right_value=",".join(supported_without[:16]),
                                notes={"gap": "missing_entrypoint"},
                            ),
                        ),
                        surfaces=(ParitySurface.CONTRACT_PACK,),
                        confidence=50,
                    )
                )
        return findings


# ---------------------------------------------------------------------------
# Public convenience API
# ---------------------------------------------------------------------------


def check_vfs_mcp_parity(
    inventory: MCPlusPlusInventory,
    *,
    claims: Sequence[CallPathClaim | Mapping[str, Any]] | None = None,
    resolution: MCPlusPlusResolutionResult | None = None,
    contract_pack: VfsContractPack | None = None,
    runtime_receipts: Sequence[RuntimeWitnessReceipt] = (),
    tool_names: Sequence[str] | None = None,
    server_name: str = "",
    use_canonical_contract_pack: bool = True,
) -> VfsMcpParityReport:
    """Check VFS manifest/SDK/MCP/MCP++ parity end to end.

    When ``contract_pack`` is omitted and ``use_canonical_contract_pack`` is
    true, the canonical pack from VFS-026 is bound for capability/degradation
    surface comparison.
    """

    pack = contract_pack
    if pack is None and use_canonical_contract_pack:
        pack = canonical_vfs_contract_pack()
    checker = VfsMcpContractChecker(inventory, contract_pack=pack)
    return checker.check(
        tool_names=tool_names,
        claims=claims,
        resolution=resolution,
        runtime_receipts=runtime_receipts,
        server_name=server_name,
    )


def make_surface_view(**kwargs: Any) -> SurfaceView:
    """Construct a validated ``SurfaceView`` (test/helper entrypoint)."""

    return SurfaceView(**kwargs)


def make_parity_witness(**kwargs: Any) -> ParityWitness:
    """Construct a validated ``ParityWitness`` (test/helper entrypoint)."""

    return ParityWitness(**kwargs)


def report_content_identity(report: VfsMcpParityReport | Mapping[str, Any]) -> str:
    """Stable content identity for a parity report."""

    if isinstance(report, VfsMcpParityReport):
        return report.report_id
    return "vfsprpt-" + content_identity(
        VfsMcpParityReport.from_dict(report)._identity_payload()
    )


def finding_kinds() -> tuple[str, ...]:
    """Closed finding-kind vocabulary (deterministic order)."""

    return tuple(sorted(item.value for item in ParityFindingKind))


def parity_surfaces() -> tuple[str, ...]:
    """Closed surface vocabulary (deterministic order)."""

    return tuple(sorted(item.value for item in ParitySurface))


__all__ = [
    "CHECKER_AUTHORIZES_REPAIR",
    "CHECKER_IS_COMPLETION_EVIDENCE",
    "CHECKER_IS_CORRECTNESS_EVIDENCE",
    "CHECKER_PRODUCER",
    "CHECKER_VERSION",
    "CONTRACT_VERSION",
    "EVIDENCE_MANIFEST_PARITY",
    "EVIDENCE_RUNTIME_WITNESS",
    "EVIDENCE_VFS_MCP_CALL_PATH",
    "EVIDENCE_VFS_MCP_PARITY",
    "GOAL_ID",
    "REQUIRED_PROVED_STAGES",
    "ParityFinding",
    "ParityFindingKind",
    "ParitySeverity",
    "ParitySurface",
    "ParityWitness",
    "ReportVerdict",
    "SurfaceView",
    "ToolParityResult",
    "ToolParityVerdict",
    "VfsMcpCheckerBoundsError",
    "VfsMcpCheckerError",
    "VfsMcpContractChecker",
    "VfsMcpParityReport",
    "build_surface_views",
    "call_path_is_proved",
    "check_vfs_mcp_parity",
    "compare_tool_surfaces",
    "discover_tool_names",
    "finding_kinds",
    "make_parity_witness",
    "make_surface_view",
    "parity_surfaces",
    "report_content_identity",
    "text_names_agree",
]
