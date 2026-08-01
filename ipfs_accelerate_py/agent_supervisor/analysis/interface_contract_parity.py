"""Generic interface, manifest, SDK, MCP, and transport parity analysis (LPR-024).

Compares caller-provided surface layers that must agree for a tool to be
invokable end-to-end:

```text
python signature
  <-> registration
  <-> tools/list schemas
  <-> generated JSON manifests / TypeScript SDKs
  <-> connectors
  <-> transport profiles
  <-> result / error mappings
  <-> capability / degradation claims
  <-> real implementation targets
```

Surface kinds, tool selection, alias mapping, required call-path stages, and
contract-pack projection are **injected** via :class:`ToolSelectionPolicy`,
:class:`ParitySurfaceSpec`, and :class:`ContractProfileAdapter`.  The generic
engine never embeds product-domain tool aliases, fixed connector names, fixed
source paths, or board/goal identifiers.

Consumption policy (conflict-safe):

* Consume inventories, resolved call paths, runtime witness records, and
  contract-profile projections supplied by the caller.
* Never regenerate package manifests or promote observations into expectations.
* Same-named text without a **resolved call path** is insufficient for
  ``proved_parity``; emit an explicit ``missing_resolved_call_path`` finding.

Findings cover stale generated artifacts, missing registrations, extra
unreachable tools, wrong aliases/schema/errors, direct local bypass,
mock/fallback dispatch, and ambiguous paths.  Each finding carries a minimal
witness (surface pair + values + optional path/runtime evidence refs).

This module is not completion evidence and does not authorize repairs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

# ---------------------------------------------------------------------------
# Schema / evidence identities (domain-neutral)
# ---------------------------------------------------------------------------

INTERFACE_CONTRACT_PARITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/interface-contract-parity@1"
)
INTERFACE_PARITY_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/interface-parity-report@1"
)
INTERFACE_PARITY_FINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/interface-parity-finding@1"
)
INTERFACE_PARITY_WITNESS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/interface-parity-witness@1"
)
INTERFACE_SURFACE_VIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/interface-surface-view@1"
)
INTERFACE_TOOL_PARITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/interface-tool-parity@1"
)

EVIDENCE_INTERFACE_PARITY: Final[str] = "interface/manifest-sdk-parity@1"
EVIDENCE_CALL_PATH: Final[str] = "interface/resolved-call-path@1"
EVIDENCE_MANIFEST_PARITY: Final[str] = "interface/manifest-parity@1"
EVIDENCE_RUNTIME_WITNESS: Final[str] = "interface/runtime-witness@1"

CHECKER_VERSION: Final[str] = "interface-contract-parity@1"
CHECKER_PRODUCER: Final[str] = "interface-contract-parity@1"
CONTRACT_VERSION: Final[int] = 1

DEFAULT_MAX_TOOLS: Final[int] = 10_000
DEFAULT_MAX_FINDINGS: Final[int] = 50_000
DEFAULT_MAX_WITNESSES: Final[int] = 50_000
DEFAULT_MAX_LABEL_BYTES: Final[int] = 4_096
DEFAULT_MAX_NOTES_BYTES: Final[int] = 8_192
DEFAULT_MAX_SCHEMA_BYTES: Final[int] = 262_144
DEFAULT_MAX_ARTIFACTS: Final[int] = 50_000
DEFAULT_MAX_HOPS: Final[int] = 256
DEFAULT_MAX_PATHS: Final[int] = 10_000

# Authority: comparison only.
CHECKER_IS_COMPLETION_EVIDENCE: Final[bool] = False
CHECKER_IS_CORRECTNESS_EVIDENCE: Final[bool] = False
CHECKER_AUTHORIZES_REPAIR: Final[bool] = False

# Default stages required for a *proved* invocation path (caller-overridable).
DEFAULT_REQUIRED_PROVED_STAGES: Final[tuple[str, ...]] = (
    "connector",
    "profile_transport",
    "tools_list",
    "tools_call",
    "server_registry",
    "adapter",
    "package_implementation",
    "result_error_mapping",
)

# Default closed surface vocabulary (domain-neutral names; no product aliases).
DEFAULT_SURFACE_KINDS: Final[tuple[str, ...]] = (
    "python_signature",
    "registration",
    "tools_list",
    "json_manifest",
    "typescript_sdk",
    "connector",
    "transport_profile",
    "result_error_map",
    "capability_degradation",
    "implementation_target",
    "runtime_witness",
    "contract_pack",
)

# Name-bearing surfaces used for text-name agreement (overridable via policy).
DEFAULT_NAME_BEARING_SURFACES: Final[tuple[str, ...]] = (
    "python_signature",
    "registration",
    "tools_list",
    "json_manifest",
    "typescript_sdk",
    "implementation_target",
)

# Roles that contribute named tool inventory for cross-surface discovery.
DEFAULT_TOOL_BEARING_ROLES: Final[frozenset[str]] = frozenset(
    {
        "tool_list_entry",
        "registration",
        "manifest",
        "alias",
        "implementation",
        "adapter",
        "tool_call_site",
        "connector",
        "result_map",
        "error_map",
        "json_schema",
    }
)

DEFAULT_GENERATED_ROLES: Final[frozenset[str]] = frozenset(
    {
        "manifest",
        "copied_manifest",
    }
)

DEFAULT_NON_INVOCATION_REASONS: Final[frozenset[str]] = frozenset(
    {
        "same_name_helper",
        "mock_implementation",
        "test_server",
        "copied_manifest",
        "static_dashboard",
        "legacy_fallback",
        "import_without_call",
    }
)

_FORBIDDEN_SILENT_DEGRADATION: Final[frozenset[str]] = frozenset(
    {
        "silent_success",
        "placeholder_success",
        "swallow",
    }
)

_CONTENT_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:sha256:)?[0-9a-fA-F]{64}$|^(?:baguqeera|bafy)[a-z2-7]{20,}$"
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class InterfaceParityError(ValueError):
    """Malformed or unsafe interface parity input."""

    def __init__(
        self,
        message: str,
        *,
        reason_codes: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.reason_codes = tuple(reason_codes)


class InterfaceParityBoundsError(InterfaceParityError):
    """A compact parity record exceeded an explicit bound."""


class ParityFindingKind(str, Enum):
    """Closed finding kinds for interface contract parity."""

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
    UNRESOLVED_PATH = "unresolved_path"
    FORGED_ARTIFACT = "forged_artifact"
    UNBOUNDED_ARTIFACT = "unbounded_artifact"


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


class PathVerdict(str, Enum):
    """Call-path resolution verdict (domain-neutral)."""

    PROVED = "proved"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class HopStatus(str, Enum):
    """Per-hop resolution status."""

    RESOLVED_STATIC = "resolved_static"
    RESOLVED_DYNAMIC = "resolved_dynamic"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


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
        ParityFindingKind.UNRESOLVED_PATH: ParitySeverity.ERROR,
        ParityFindingKind.FORGED_ARTIFACT: ParitySeverity.ERROR,
        ParityFindingKind.UNBOUNDED_ARTIFACT: ParitySeverity.ERROR,
    }
)

_DRIFT_KIND_TO_FINDING: Mapping[str, ParityFindingKind] = MappingProxyType(
    {
        "name_mismatch": ParityFindingKind.NAME_MISMATCH,
        "schema_mismatch": ParityFindingKind.SCHEMA_MISMATCH,
        "version_mismatch": ParityFindingKind.VERSION_MISMATCH,
        "profile_mismatch": ParityFindingKind.PROFILE_MISMATCH,
        "alias_mismatch": ParityFindingKind.WRONG_ALIAS,
        "error_map_mismatch": ParityFindingKind.ERROR_MAP_MISMATCH,
        "result_map_mismatch": ParityFindingKind.RESULT_MAP_MISMATCH,
        "missing_registration": ParityFindingKind.MISSING_REGISTRATION,
        "extra_unreachable": ParityFindingKind.EXTRA_UNREACHABLE_TOOL,
        "stale_manifest": ParityFindingKind.STALE_GENERATED_ARTIFACT,
        "copied_without_binding": ParityFindingKind.MISSING_RESOLVED_CALL_PATH,
        "transport_mismatch": ParityFindingKind.TRANSPORT_MISMATCH,
        "language_name_mismatch": ParityFindingKind.NAME_MISMATCH,
    }
)


# ---------------------------------------------------------------------------
# Validation + identity helpers
# ---------------------------------------------------------------------------


def _json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 32:
        raise InterfaceParityBoundsError("json depth exceeded")
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value.encode("utf-8")) > DEFAULT_MAX_SCHEMA_BYTES:
            raise InterfaceParityBoundsError("string exceeds schema bound")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if len(value) > DEFAULT_MAX_ARTIFACTS:
            raise InterfaceParityBoundsError("mapping too large")
        return {
            str(k): _json_value(v, depth=depth + 1)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        if len(value) > DEFAULT_MAX_ARTIFACTS:
            raise InterfaceParityBoundsError("sequence too large")
        return [_json_value(item, depth=depth + 1) for item in value]
    return str(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _content_id(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def schema_fingerprint(schema: Mapping[str, Any] | None) -> str:
    """Stable fingerprint for a JSON-schema-like mapping."""

    if not schema:
        return ""
    if not isinstance(schema, Mapping):
        raise InterfaceParityError("schema must be a mapping")
    payload = _canonical_json(dict(schema))
    if len(payload) > DEFAULT_MAX_SCHEMA_BYTES:
        raise InterfaceParityBoundsError("schema exceeds byte bound")
    return "sch-" + hashlib.sha256(payload).hexdigest()[:32]


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise InterfaceParityError(f"{name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise InterfaceParityError(f"{name} is required")
    if len(text.encode("utf-8")) > DEFAULT_MAX_LABEL_BYTES:
        raise InterfaceParityBoundsError(f"{name} exceeds label bound")
    return text


def _enum(value: Any, enum_type: type[Enum], label: str) -> Any:
    if isinstance(value, enum_type):
        return value
    text = str(value or "").strip()
    try:
        return enum_type(text)
    except ValueError as exc:
        raise InterfaceParityError(f"unsupported {label}: {text!r}") from exc


def _mapping(
    value: Any,
    name: str,
    *,
    max_bytes: int = DEFAULT_MAX_NOTES_BYTES,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise InterfaceParityError(f"{name} must be a mapping")
    plain: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise InterfaceParityError(f"{name} keys must be strings")
        plain[key] = item
    encoded = _canonical_json(plain)
    if len(encoded) > max_bytes:
        raise InterfaceParityBoundsError(f"{name} exceeds bound")
    return MappingProxyType(plain)


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        if len(text.encode("utf-8")) > DEFAULT_MAX_LABEL_BYTES:
            raise InterfaceParityBoundsError("label exceeds bound")
        seen.add(text)
        ordered.append(text)
    return tuple(sorted(ordered))


def _require_identifier(value: str, field_name: str) -> str:
    text = _text(value, field_name)
    if not re.fullmatch(r"[A-Za-z0-9_./:+@-]{1,512}", text):
        raise InterfaceParityError(
            f"{field_name} is not a closed identifier",
            reason_codes=("invalid_identifier",),
        )
    return text


def normalize_tool_name(name: str) -> str:
    """Domain-neutral tool-name normalization (case-fold + slash→dot)."""

    text = str(name or "").strip()
    if not text:
        return ""
    text = text.replace("\\", "/").replace("/", ".")
    # Collapse runs of dots / underscores for alias comparison only.
    text = re.sub(r"[._]+", ".", text)
    return text.strip(".").lower()


def _tool_key(name: str) -> str:
    return normalize_tool_name(name) or _text(name, "tool_name", required=False)


# ---------------------------------------------------------------------------
# Policy injection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class ParitySurfaceSpec:
    """One injectable surface kind for parity comparison.

    Domain products supply their own surface kinds and role mappings.  The
    generic engine never hard-codes product-specific surface names beyond the
    neutral default vocabulary.
    """

    kind: str
    roles: tuple[str, ...] = ()
    name_bearing: bool = False
    generated: bool = False
    languages: tuple[str, ...] = ()
    prefer_language: str = ""
    artifact_kind: str = ""
    required_for_proved: bool = False
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _require_identifier(self.kind, "kind"))
        roles = _sorted_unique(self.roles or ())
        object.__setattr__(self, "roles", roles)
        for flag_name in ("name_bearing", "generated", "required_for_proved"):
            if not isinstance(getattr(self, flag_name), bool):
                raise InterfaceParityError(f"{flag_name} must be a boolean")
        object.__setattr__(
            self, "languages", _sorted_unique(self.languages or ())
        )
        object.__setattr__(
            self,
            "prefer_language",
            _text(self.prefer_language, "prefer_language", required=False),
        )
        object.__setattr__(
            self,
            "artifact_kind",
            _text(self.artifact_kind, "artifact_kind", required=False),
        )
        object.__setattr__(self, "notes", _mapping(self.notes, "surface_spec.notes"))


def default_surface_specs() -> tuple[ParitySurfaceSpec, ...]:
    """Neutral default surface set used when a policy omits surface specs."""

    return (
        ParitySurfaceSpec(
            kind="python_signature",
            roles=("implementation",),
            name_bearing=True,
            languages=("python",),
            prefer_language="python",
        ),
        ParitySurfaceSpec(
            kind="registration",
            roles=("registration",),
            name_bearing=True,
            required_for_proved=True,
        ),
        ParitySurfaceSpec(
            kind="tools_list",
            roles=("tool_list_entry",),
            name_bearing=True,
            required_for_proved=True,
        ),
        ParitySurfaceSpec(
            kind="json_manifest",
            roles=("manifest", "copied_manifest"),
            name_bearing=True,
            generated=True,
            languages=("json",),
            artifact_kind="json_manifest",
        ),
        ParitySurfaceSpec(
            kind="typescript_sdk",
            roles=("manifest",),
            name_bearing=True,
            generated=True,
            languages=("typescript",),
            artifact_kind="typescript_sdk",
        ),
        ParitySurfaceSpec(
            kind="connector",
            roles=("connector",),
            required_for_proved=True,
        ),
        ParitySurfaceSpec(
            kind="transport_profile",
            roles=("transport",),
            required_for_proved=True,
        ),
        ParitySurfaceSpec(
            kind="result_error_map",
            roles=("result_map", "error_map"),
        ),
        ParitySurfaceSpec(
            kind="capability_degradation",
            roles=("registration", "connector"),
        ),
        ParitySurfaceSpec(
            kind="implementation_target",
            roles=("implementation", "adapter"),
            name_bearing=True,
            required_for_proved=True,
        ),
        ParitySurfaceSpec(
            kind="runtime_witness",
            roles=(),
        ),
        ParitySurfaceSpec(
            kind="contract_pack",
            roles=(),
        ),
    )


@dataclass(frozen=True)
class ToolSelectionPolicy:
    """Immutable tool selection, alias mapping, and surface configuration.

    All domain vocabulary lives here — never in module constants beyond the
    neutral defaults above.
    """

    policy_id: str
    surface_specs: tuple[ParitySurfaceSpec, ...] = field(
        default_factory=default_surface_specs
    )
    alias_groups: tuple[tuple[str, ...], ...] = ()
    alias_map: Mapping[str, str] = field(default_factory=dict)
    tool_name_filter: Callable[[str], bool] | None = None
    required_proved_stages: tuple[str, ...] = DEFAULT_REQUIRED_PROVED_STAGES
    tool_bearing_roles: frozenset[str] = DEFAULT_TOOL_BEARING_ROLES
    generated_roles: frozenset[str] = DEFAULT_GENERATED_ROLES
    non_invocation_reasons: frozenset[str] = DEFAULT_NON_INVOCATION_REASONS
    server_name: str = ""
    max_tools: int = DEFAULT_MAX_TOOLS
    max_findings: int = DEFAULT_MAX_FINDINGS
    max_artifacts: int = DEFAULT_MAX_ARTIFACTS
    schema: str = INTERFACE_CONTRACT_PARITY_SCHEMA
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _require_identifier(self.policy_id, "policy_id")
        )
        specs = tuple(self.surface_specs or default_surface_specs())
        if not specs:
            raise InterfaceParityError("surface_specs must not be empty")
        if len(specs) > 256:
            raise InterfaceParityBoundsError("too many surface_specs")
        kinds = [spec.kind for spec in specs]
        if len(set(kinds)) != len(kinds):
            raise InterfaceParityError(
                "surface_specs kinds must be unique",
                reason_codes=("duplicate_surface_kind",),
            )
        object.__setattr__(self, "surface_specs", specs)
        groups: list[tuple[str, ...]] = []
        for group in self.alias_groups or ():
            cleaned = _sorted_unique(group)
            if cleaned:
                groups.append(cleaned)
        object.__setattr__(self, "alias_groups", tuple(groups))
        alias_map: dict[str, str] = {}
        for key, value in dict(self.alias_map or {}).items():
            k = normalize_tool_name(str(key))
            v = normalize_tool_name(str(value))
            if k and v:
                alias_map[k] = v
        object.__setattr__(self, "alias_map", MappingProxyType(alias_map))
        stages = _sorted_unique(self.required_proved_stages or ())
        if not stages:
            raise InterfaceParityError("required_proved_stages must not be empty")
        object.__setattr__(self, "required_proved_stages", stages)
        roles = frozenset(
            _require_identifier(r, "role") for r in (self.tool_bearing_roles or ())
        )
        object.__setattr__(self, "tool_bearing_roles", roles)
        gen_roles = frozenset(
            _require_identifier(r, "role") for r in (self.generated_roles or ())
        )
        object.__setattr__(self, "generated_roles", gen_roles)
        reasons = frozenset(
            _require_identifier(r, "reason")
            for r in (self.non_invocation_reasons or ())
        )
        object.__setattr__(self, "non_invocation_reasons", reasons)
        object.__setattr__(
            self, "server_name", _text(self.server_name, "server_name", required=False)
        )
        for bound_name in ("max_tools", "max_findings", "max_artifacts"):
            bound = getattr(self, bound_name)
            if (
                isinstance(bound, bool)
                or not isinstance(bound, int)
                or bound < 1
                or bound > DEFAULT_MAX_FINDINGS
            ):
                raise InterfaceParityBoundsError(f"{bound_name} out of bounds")
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(self, "notes", _mapping(self.notes, "policy.notes"))
        if self.tool_name_filter is not None and not callable(self.tool_name_filter):
            raise InterfaceParityError("tool_name_filter must be callable or None")

    @property
    def content_id(self) -> str:
        return _content_id(self.to_dict())

    def surface_kinds(self) -> tuple[str, ...]:
        return tuple(spec.kind for spec in self.surface_specs)

    def name_bearing_kinds(self) -> frozenset[str]:
        return frozenset(spec.kind for spec in self.surface_specs if spec.name_bearing)

    def spec_for(self, kind: str) -> ParitySurfaceSpec | None:
        for spec in self.surface_specs:
            if spec.kind == kind:
                return spec
        return None

    def aliases_for(self, tool_name: str) -> frozenset[str]:
        """Return the closed alias set for a tool name under this policy."""

        key = _tool_key(tool_name)
        aliases: set[str] = {key, normalize_tool_name(tool_name)}
        if tool_name:
            aliases.add(tool_name.strip())
        # alias_map (normalized key -> canonical)
        canonical = self.alias_map.get(key, key)
        aliases.add(canonical)
        for raw, mapped in self.alias_map.items():
            if mapped == canonical or raw == key or mapped == key:
                aliases.add(raw)
                aliases.add(mapped)
        for group in self.alias_groups:
            group_keys = {_tool_key(item) for item in group}
            if key in group_keys or canonical in group_keys:
                aliases.update(group_keys)
                aliases.update(group)
        # Built-in orthography variants (slash/dot/case) only — no product aliases.
        for item in list(aliases):
            aliases.add(normalize_tool_name(item))
            aliases.add(item.replace(".", "/"))
            aliases.add(item.replace("/", "."))
        aliases.discard("")
        return frozenset(aliases)

    def accepts_tool(self, tool_name: str) -> bool:
        if not tool_name or not str(tool_name).strip():
            return False
        if self.tool_name_filter is None:
            return True
        return bool(self.tool_name_filter(tool_name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "surface_specs": [
                {
                    "kind": s.kind,
                    "roles": list(s.roles),
                    "name_bearing": s.name_bearing,
                    "generated": s.generated,
                    "languages": list(s.languages),
                    "prefer_language": s.prefer_language,
                    "artifact_kind": s.artifact_kind,
                    "required_for_proved": s.required_for_proved,
                    "notes": dict(s.notes),
                }
                for s in self.surface_specs
            ],
            "alias_groups": [list(g) for g in self.alias_groups],
            "alias_map": dict(self.alias_map),
            "required_proved_stages": list(self.required_proved_stages),
            "tool_bearing_roles": sorted(self.tool_bearing_roles),
            "generated_roles": sorted(self.generated_roles),
            "non_invocation_reasons": sorted(self.non_invocation_reasons),
            "server_name": self.server_name,
            "max_tools": self.max_tools,
            "max_findings": self.max_findings,
            "max_artifacts": self.max_artifacts,
            "notes": dict(self.notes),
            "has_tool_name_filter": self.tool_name_filter is not None,
        }


@dataclass(frozen=True)
class ContractProfileAdapter:
    """Projects a program contract profile (or duck-typed pack) into parity.

    Accepts either a :class:`ProgramContractProfile`-like object with
    ``operations``, ``surfaces``, and ``content_id``, or an explicit mapping of
    operation → surface projection data.
    """

    adapter_id: str
    operations: tuple[str, ...] = ()
    surfaces: tuple[str, ...] = ()
    operation_entrypoints: Mapping[str, str] = field(default_factory=dict)
    unresolved_operations: tuple[str, ...] = ()
    capability_claims: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    degradation_claims: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    error_codes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    profile_content_id: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "adapter_id", _require_identifier(self.adapter_id, "adapter_id")
        )
        object.__setattr__(self, "operations", _sorted_unique(self.operations or ()))
        object.__setattr__(self, "surfaces", _sorted_unique(self.surfaces or ()))
        entrypoints: dict[str, str] = {}
        for key, value in dict(self.operation_entrypoints or {}).items():
            k = _text(str(key), "operation")
            v = _text(str(value), "entrypoint", required=False)
            entrypoints[k] = v
        object.__setattr__(
            self, "operation_entrypoints", MappingProxyType(entrypoints)
        )
        object.__setattr__(
            self,
            "unresolved_operations",
            _sorted_unique(self.unresolved_operations or ()),
        )
        caps: dict[str, tuple[str, ...]] = {}
        for key, values in dict(self.capability_claims or {}).items():
            caps[_text(str(key), "operation")] = _sorted_unique(values or ())
        object.__setattr__(self, "capability_claims", MappingProxyType(caps))
        degs: dict[str, tuple[str, ...]] = {}
        for key, values in dict(self.degradation_claims or {}).items():
            degs[_text(str(key), "operation")] = _sorted_unique(values or ())
        object.__setattr__(self, "degradation_claims", MappingProxyType(degs))
        errs: dict[str, tuple[str, ...]] = {}
        for key, values in dict(self.error_codes or {}).items():
            errs[_text(str(key), "operation")] = _sorted_unique(values or ())
        object.__setattr__(self, "error_codes", MappingProxyType(errs))
        object.__setattr__(
            self,
            "profile_content_id",
            _text(self.profile_content_id, "profile_content_id", required=False),
        )
        object.__setattr__(self, "notes", _mapping(self.notes, "adapter.notes"))

    @property
    def content_id(self) -> str:
        return self.profile_content_id or _content_id(self.to_dict())

    @classmethod
    def from_program_contract_profile(
        cls,
        profile: Any,
        *,
        adapter_id: str = "contract-profile",
        surface_names: Sequence[str] | None = None,
    ) -> "ContractProfileAdapter":
        """Build an adapter from a ProgramContractProfile-like object."""

        operations: list[str] = []
        entrypoints: dict[str, str] = {}
        caps: dict[str, tuple[str, ...]] = {}
        degs: dict[str, tuple[str, ...]] = {}
        errs: dict[str, tuple[str, ...]] = {}
        unresolved: list[str] = []

        for op in getattr(profile, "operations", ()) or ():
            name = str(
                getattr(op, "operation", None)
                or getattr(op, "name", None)
                or op
                or ""
            ).strip()
            if not name:
                continue
            operations.append(name)
            entry = str(getattr(op, "entrypoint", "") or "").strip()
            if entry:
                entrypoints[name] = entry
            support = getattr(op, "support", None)
            support_value = (
                support.value if isinstance(support, Enum) else str(support or "")
            )
            if support_value.lower() in {"supported", "required"} and not entry:
                unresolved.append(name)
            err_codes = tuple(
                str(c) for c in (getattr(op, "error_codes", ()) or ()) if c
            )
            if err_codes:
                errs[name] = err_codes
            op_caps = tuple(
                str(c) for c in (getattr(op, "capability_claims", ()) or ()) if c
            )
            if op_caps:
                caps[name] = op_caps

        surfaces: list[str] = []
        if surface_names is not None:
            surfaces = [str(s) for s in surface_names]
        else:
            for surface in getattr(profile, "surfaces", ()) or ():
                sname = str(
                    getattr(surface, "surface", None)
                    or getattr(surface, "name", None)
                    or surface
                    or ""
                ).strip()
                if sname:
                    surfaces.append(sname)
                for uop in getattr(surface, "unresolved_operations", ()) or ():
                    unresolved.append(
                        str(getattr(uop, "value", uop) if uop else "").strip()
                    )

        profile_id = str(
            getattr(profile, "content_id", None)
            or getattr(profile, "profile_id", None)
            or ""
        )
        return cls(
            adapter_id=adapter_id,
            operations=tuple(operations),
            surfaces=tuple(surfaces),
            operation_entrypoints=entrypoints,
            unresolved_operations=tuple(unresolved),
            capability_claims=caps,
            degradation_claims=degs,
            error_codes=errs,
            profile_content_id=profile_id,
        )

    def projection_for(self, tool_name: str) -> Mapping[str, Any]:
        key = tool_name
        aliases = {tool_name, normalize_tool_name(tool_name)}
        matched = next(
            (
                op
                for op in self.operations
                if op in aliases or normalize_tool_name(op) in aliases
            ),
            "",
        )
        if not matched:
            return MappingProxyType(
                {
                    "operation": "",
                    "entrypoint": "",
                    "capability_claims": (),
                    "degradation_claims": (),
                    "error_codes": (),
                    "unresolved": tool_name in self.unresolved_operations,
                }
            )
        return MappingProxyType(
            {
                "operation": matched,
                "entrypoint": self.operation_entrypoints.get(matched, ""),
                "capability_claims": self.capability_claims.get(matched, ()),
                "degradation_claims": self.degradation_claims.get(matched, ()),
                "error_codes": self.error_codes.get(matched, ()),
                "unresolved": matched in self.unresolved_operations,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "operations": list(self.operations),
            "surfaces": list(self.surfaces),
            "operation_entrypoints": dict(self.operation_entrypoints),
            "unresolved_operations": list(self.unresolved_operations),
            "capability_claims": {
                k: list(v) for k, v in self.capability_claims.items()
            },
            "degradation_claims": {
                k: list(v) for k, v in self.degradation_claims.items()
            },
            "error_codes": {k: list(v) for k, v in self.error_codes.items()},
            "profile_content_id": self.profile_content_id,
            "notes": dict(self.notes),
        }


# ---------------------------------------------------------------------------
# Inventory + call-path input records (domain-neutral)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceArtifact:
    """One inventory artifact observed on a surface role."""

    artifact_id: str
    role: str
    name: str
    tool_name: str = ""
    language: str = ""
    package: str = ""
    version: str = ""
    qualified_name: str = ""
    path: str = ""
    server_name: str = ""
    transport: str = ""
    profiles: tuple[str, ...] = ()
    alias_of: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    error_codes: tuple[str, ...] = ()
    markers: tuple[str, ...] = ()
    has_call_edge: bool = False
    non_invocation_reason: str = ""
    implementation_target: str = ""
    content_id: str = ""
    record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_id", _require_identifier(self.artifact_id, "artifact_id")
        )
        object.__setattr__(self, "role", _require_identifier(self.role, "role"))
        object.__setattr__(self, "name", _text(self.name, "name", required=False))
        object.__setattr__(
            self, "tool_name", _text(self.tool_name, "tool_name", required=False)
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
            "qualified_name",
            _text(self.qualified_name, "qualified_name", required=False),
        )
        # Paths are caller-supplied evidence labels; bounded but not domain-checked.
        path = _text(self.path, "path", required=False)
        if len(path.encode("utf-8")) > DEFAULT_MAX_LABEL_BYTES:
            raise InterfaceParityBoundsError("path exceeds bound")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self, "server_name", _text(self.server_name, "server_name", required=False)
        )
        object.__setattr__(
            self, "transport", _text(self.transport, "transport", required=False)
        )
        object.__setattr__(self, "profiles", _sorted_unique(self.profiles or ()))
        object.__setattr__(
            self, "alias_of", _text(self.alias_of, "alias_of", required=False)
        )
        object.__setattr__(
            self, "input_schema", _mapping(self.input_schema, "input_schema", max_bytes=DEFAULT_MAX_SCHEMA_BYTES)
        )
        object.__setattr__(
            self,
            "output_schema",
            _mapping(self.output_schema, "output_schema", max_bytes=DEFAULT_MAX_SCHEMA_BYTES),
        )
        object.__setattr__(
            self, "error_codes", _sorted_unique(self.error_codes or ())
        )
        object.__setattr__(self, "markers", _sorted_unique(self.markers or ()))
        if not isinstance(self.has_call_edge, bool):
            raise InterfaceParityError("has_call_edge must be a boolean")
        object.__setattr__(
            self,
            "non_invocation_reason",
            _text(self.non_invocation_reason, "non_invocation_reason", required=False),
        )
        object.__setattr__(
            self,
            "implementation_target",
            _text(
                self.implementation_target, "implementation_target", required=False
            ),
        )
        object.__setattr__(
            self, "content_id", _text(self.content_id, "content_id", required=False)
        )
        object.__setattr__(
            self, "record", _mapping(self.record, "record", max_bytes=DEFAULT_MAX_NOTES_BYTES)
        )

    @property
    def effective_tool_name(self) -> str:
        return self.tool_name or self.name or self.alias_of

    def identity_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "role": self.role,
            "name": self.name,
            "tool_name": self.tool_name,
            "language": self.language,
            "package": self.package,
            "version": self.version,
            "qualified_name": self.qualified_name,
            "path": self.path,
            "server_name": self.server_name,
            "transport": self.transport,
            "profiles": list(self.profiles),
            "alias_of": self.alias_of,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "error_codes": list(self.error_codes),
            "markers": list(self.markers),
            "has_call_edge": self.has_call_edge,
            "non_invocation_reason": self.non_invocation_reason,
            "implementation_target": self.implementation_target,
            "record": dict(self.record),
        }

    def computed_content_id(self) -> str:
        return _content_id(self.identity_payload())

    def validate_integrity(self) -> None:
        """Reject forged content_id and unbounded payloads."""

        if self.content_id:
            expected = self.computed_content_id()
            # Accept either full match or digest-only equivalence.
            left = self.content_id.removeprefix("sha256:")
            right = expected.removeprefix("sha256:")
            if left != right and self.content_id != expected:
                raise InterfaceParityError(
                    f"forged artifact content_id for {self.artifact_id!r}",
                    reason_codes=("forged_artifact",),
                )


def make_artifact(
    artifact_id: str,
    role: str,
    name: str,
    **kwargs: Any,
) -> SurfaceArtifact:
    """Construct a validated :class:`SurfaceArtifact`."""

    return SurfaceArtifact(
        artifact_id=artifact_id,
        role=role,
        name=name,
        **kwargs,
    )


@dataclass(frozen=True)
class SurfaceInventory:
    """Bounded inventory of surface artifacts for parity comparison."""

    inventory_id: str
    artifacts: tuple[SurfaceArtifact, ...] = ()
    forest_id: str = ""
    admitted_transports: tuple[str, ...] = ()
    required_profiles: tuple[str, ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "inventory_id",
            _require_identifier(self.inventory_id, "inventory_id"),
        )
        artifacts = tuple(self.artifacts or ())
        if len(artifacts) > DEFAULT_MAX_ARTIFACTS:
            raise InterfaceParityBoundsError(
                "too many artifacts",
                reason_codes=("unbounded_artifact",),
            )
        seen_ids: set[str] = set()
        for item in artifacts:
            if not isinstance(item, SurfaceArtifact):
                raise InterfaceParityError("artifacts must be SurfaceArtifact")
            if item.artifact_id in seen_ids:
                raise InterfaceParityError(
                    f"duplicate artifact_id {item.artifact_id!r}",
                    reason_codes=("duplicate_artifact",),
                )
            seen_ids.add(item.artifact_id)
            item.validate_integrity()
        object.__setattr__(
            self,
            "artifacts",
            tuple(sorted(artifacts, key=lambda a: a.artifact_id)),
        )
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "forest_id", required=False)
        )
        object.__setattr__(
            self,
            "admitted_transports",
            _sorted_unique(self.admitted_transports or ()),
        )
        object.__setattr__(
            self,
            "required_profiles",
            _sorted_unique(self.required_profiles or ()),
        )
        object.__setattr__(self, "notes", _mapping(self.notes, "inventory.notes"))

    @property
    def content_id(self) -> str:
        return _content_id(
            {
                "inventory_id": self.inventory_id,
                "forest_id": self.forest_id,
                "admitted_transports": list(self.admitted_transports),
                "required_profiles": list(self.required_profiles),
                "artifacts": [a.identity_payload() for a in self.artifacts],
                "notes": dict(self.notes),
            }
        )


@dataclass(frozen=True)
class CallPathHop:
    """One hop on a resolved (or unresolved) call path."""

    stage: str
    status: HopStatus = HopStatus.UNKNOWN
    reason_code: str = ""
    ref: str = ""
    artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _require_identifier(self.stage, "stage"))
        object.__setattr__(self, "status", _enum(self.status, HopStatus, "status"))
        object.__setattr__(
            self,
            "reason_code",
            _text(self.reason_code, "reason_code", required=False),
        )
        object.__setattr__(self, "ref", _text(self.ref, "ref", required=False))
        object.__setattr__(
            self, "artifact_ids", _sorted_unique(self.artifact_ids or ())
        )


@dataclass(frozen=True)
class ResolvedCallPath:
    """Caller-supplied call path with explicit verdict and hops.

    Unresolved paths remain explicit: they are never silently promoted to
    proved.  Same-name text alone does not set ``verdict=proved``.
    """

    path_id: str
    tool_name: str
    verdict: PathVerdict
    hops: tuple[CallPathHop, ...] = ()
    connector_ref: str = ""
    implementation_ref: str = ""
    caller_ref: str = ""
    transport: str = ""
    profiles: tuple[str, ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "path_id", _require_identifier(self.path_id, "path_id")
        )
        object.__setattr__(self, "tool_name", _text(self.tool_name, "tool_name"))
        object.__setattr__(
            self, "verdict", _enum(self.verdict, PathVerdict, "verdict")
        )
        hops = tuple(self.hops or ())
        if len(hops) > DEFAULT_MAX_HOPS:
            raise InterfaceParityBoundsError("too many hops")
        for hop in hops:
            if not isinstance(hop, CallPathHop):
                raise InterfaceParityError("hops must be CallPathHop")
        object.__setattr__(self, "hops", hops)
        object.__setattr__(
            self,
            "connector_ref",
            _text(self.connector_ref, "connector_ref", required=False),
        )
        object.__setattr__(
            self,
            "implementation_ref",
            _text(self.implementation_ref, "implementation_ref", required=False),
        )
        object.__setattr__(
            self, "caller_ref", _text(self.caller_ref, "caller_ref", required=False)
        )
        object.__setattr__(
            self, "transport", _text(self.transport, "transport", required=False)
        )
        object.__setattr__(self, "profiles", _sorted_unique(self.profiles or ()))
        object.__setattr__(self, "notes", _mapping(self.notes, "path.notes"))


@dataclass(frozen=True)
class DriftWitnessRecord:
    """External resolver drift witness consumed (not re-derived) by the checker."""

    drift_kind: str
    tool_name: str = ""
    left_value: str = ""
    right_value: str = ""
    left_ref: str = ""
    right_ref: str = ""
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "drift_kind", _require_identifier(self.drift_kind, "drift_kind")
        )
        object.__setattr__(
            self, "tool_name", _text(self.tool_name, "tool_name", required=False)
        )
        object.__setattr__(
            self, "left_value", _text(self.left_value, "left_value", required=False)
        )
        object.__setattr__(
            self, "right_value", _text(self.right_value, "right_value", required=False)
        )
        object.__setattr__(
            self, "left_ref", _text(self.left_ref, "left_ref", required=False)
        )
        object.__setattr__(
            self, "right_ref", _text(self.right_ref, "right_ref", required=False)
        )
        object.__setattr__(
            self, "evidence_refs", _sorted_unique(self.evidence_refs or ())
        )


@dataclass(frozen=True)
class RuntimeWitnessObservation:
    """Minimal runtime witness observation for non-authoritative surface views."""

    tool_name: str
    implementation_kind: str = ""
    implementation_target: str = ""
    outcome: str = ""
    grants_runtime_authority: bool = False
    is_mock: bool = False
    transport: str = ""
    profiles: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ()
    receipt_id: str = ""
    fixture_id: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_name", _text(self.tool_name, "tool_name"))
        object.__setattr__(
            self,
            "implementation_kind",
            _text(self.implementation_kind, "implementation_kind", required=False),
        )
        object.__setattr__(
            self,
            "implementation_target",
            _text(
                self.implementation_target, "implementation_target", required=False
            ),
        )
        object.__setattr__(
            self, "outcome", _text(self.outcome, "outcome", required=False)
        )
        if not isinstance(self.grants_runtime_authority, bool):
            raise InterfaceParityError("grants_runtime_authority must be a boolean")
        if not isinstance(self.is_mock, bool):
            raise InterfaceParityError("is_mock must be a boolean")
        object.__setattr__(
            self, "transport", _text(self.transport, "transport", required=False)
        )
        object.__setattr__(self, "profiles", _sorted_unique(self.profiles or ()))
        object.__setattr__(
            self, "error_codes", _sorted_unique(self.error_codes or ())
        )
        object.__setattr__(
            self, "receipt_id", _text(self.receipt_id, "receipt_id", required=False)
        )
        object.__setattr__(
            self, "fixture_id", _text(self.fixture_id, "fixture_id", required=False)
        )
        object.__setattr__(self, "notes", _mapping(self.notes, "runtime.notes"))


# ---------------------------------------------------------------------------
# Surface view + witnesses + findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceView:
    """One layer's observation for a single tool name."""

    surface: str
    present: bool
    tool_name: str = ""
    qualified_name: str = ""
    language: str = ""
    package: str = ""
    version: str = ""
    input_schema_fingerprint: str = ""
    output_schema_fingerprint: str = ""
    error_codes: tuple[str, ...] = ()
    transport: str = ""
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
            self, "surface", _require_identifier(self.surface, "surface")
        )
        if not isinstance(self.present, bool):
            raise InterfaceParityError("present must be a boolean")
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
            self, "transport", _text(self.transport, "transport", required=False)
        )
        object.__setattr__(self, "profiles", _sorted_unique(self.profiles or ()))
        object.__setattr__(
            self, "alias_of", _text(self.alias_of, "alias_of", required=False)
        )
        object.__setattr__(
            self,
            "implementation_target",
            _text(
                self.implementation_target, "implementation_target", required=False
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
                raise InterfaceParityError(f"{flag_name} must be a boolean")
        object.__setattr__(self, "notes", _mapping(self.notes, "surface.notes"))

    @property
    def view_id(self) -> str:
        return "icpsurf-" + _content_id(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": INTERFACE_SURFACE_VIEW_SCHEMA,
            "surface": self.surface,
            "present": self.present,
            "tool_name": self.tool_name,
            "qualified_name": self.qualified_name,
            "language": self.language,
            "package": self.package,
            "version": self.version,
            "input_schema_fingerprint": self.input_schema_fingerprint,
            "output_schema_fingerprint": self.output_schema_fingerprint,
            "error_codes": list(self.error_codes),
            "transport": self.transport,
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
            raise InterfaceParityError("surface view payload must be a mapping")
        return cls(
            surface=str(payload.get("surface") or "registration"),
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
            transport=str(payload.get("transport") or ""),
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
    def absent(cls, surface: str, *, tool_name: str = "") -> "SurfaceView":
        return cls(surface=surface, present=False, tool_name=tool_name)


@dataclass(frozen=True)
class ParityWitness:
    """Minimal witness binding two surfaces (or one surface + path evidence)."""

    kind: ParityFindingKind
    tool_name: str
    left_surface: str
    right_surface: str
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
            _require_identifier(self.left_surface, "left_surface"),
        )
        object.__setattr__(
            self,
            "right_surface",
            _require_identifier(self.right_surface, "right_surface"),
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
        return "icpwit-" + _content_id(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": INTERFACE_PARITY_WITNESS_SCHEMA,
            "kind": self.kind.value,
            "tool_name": self.tool_name,
            "left_surface": self.left_surface,
            "right_surface": self.right_surface,
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
            raise InterfaceParityError("witness payload must be a mapping")
        return cls(
            kind=payload.get(
                "kind", ParityFindingKind.MISSING_RESOLVED_CALL_PATH.value
            ),
            tool_name=str(payload.get("tool_name") or ""),
            left_surface=str(payload.get("left_surface") or "registration"),
            right_surface=str(payload.get("right_surface") or "tools_list"),
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
    surfaces: tuple[str, ...] = ()
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
            raise InterfaceParityError("parity finding requires at least one witness")
        if len(witnesses) > DEFAULT_MAX_WITNESSES:
            raise InterfaceParityBoundsError("too many witnesses on one finding")
        object.__setattr__(self, "witnesses", witnesses)
        surfaces = _sorted_unique(self.surfaces or ())
        object.__setattr__(self, "surfaces", surfaces)
        object.__setattr__(self, "path_ids", _sorted_unique(self.path_ids or ()))
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, int)
            or self.confidence < 0
            or self.confidence > 100
        ):
            raise InterfaceParityError("confidence must be an int in [0, 100]")
        object.__setattr__(self, "notes", _mapping(self.notes, "finding.notes"))

    @property
    def finding_id(self) -> str:
        return "icpfind-" + _content_id(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": INTERFACE_PARITY_FINDING_SCHEMA,
            "kind": self.kind.value,
            "tool_name": self.tool_name,
            "severity": self.severity.value,
            "summary": self.summary,
            "witnesses": [item.to_dict() for item in self.witnesses],
            "surfaces": list(self.surfaces),
            "path_ids": list(self.path_ids),
            "confidence": self.confidence,
            "notes": dict(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "finding_id": self.finding_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParityFinding":
        if not isinstance(payload, Mapping):
            raise InterfaceParityError("finding payload must be a mapping")
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
        object.__setattr__(self, "tool_name", _text(self.tool_name, "tool_name"))
        object.__setattr__(
            self, "verdict", _enum(self.verdict, ToolParityVerdict, "verdict")
        )
        if not isinstance(self.surfaces, Mapping):
            raise InterfaceParityError("surfaces must be a mapping")
        normalized: dict[str, SurfaceView] = {}
        for key, view in self.surfaces.items():
            surface_key = str(key)
            if isinstance(view, SurfaceView):
                normalized[surface_key] = view
            elif isinstance(view, Mapping):
                normalized[surface_key] = SurfaceView.from_dict(view)
            else:
                raise InterfaceParityError("surface views must be SurfaceView")
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
            raise InterfaceParityError("proved_call_path must be a boolean")
        if not isinstance(self.text_names_agree, bool):
            raise InterfaceParityError("text_names_agree must be a boolean")
        if (
            self.verdict is ToolParityVerdict.PROVED_PARITY
            and not self.proved_call_path
        ):
            raise InterfaceParityError(
                "proved_parity requires a resolved call path; "
                "same text without a path is insufficient"
            )
        object.__setattr__(self, "notes", _mapping(self.notes, "tool.notes"))

    @property
    def result_id(self) -> str:
        return "icpparty-" + _content_id(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": INTERFACE_TOOL_PARITY_SCHEMA,
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
            raise InterfaceParityError("tool parity payload must be a mapping")
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
class InterfaceParityReport:
    """Content-addressed end-to-end interface parity report."""

    inventory_id: str
    policy_id: str
    tools: tuple[ToolParityResult, ...]
    findings: tuple[ParityFinding, ...]
    verdict: ReportVerdict
    forest_id: str = ""
    contract_pack_id: str = ""
    checker_version: str = CHECKER_VERSION
    truncated: bool = False
    truncation_reason: str = ""
    evidence_kinds: tuple[str, ...] = (
        EVIDENCE_INTERFACE_PARITY,
        EVIDENCE_CALL_PATH,
        EVIDENCE_MANIFEST_PARITY,
    )
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "inventory_id",
            _text(self.inventory_id, "inventory_id", required=False),
        )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id", required=False)
        )
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "forest_id", required=False)
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
            raise InterfaceParityBoundsError("too many tools in report")
        object.__setattr__(self, "tools", tools)
        findings = tuple(
            item if isinstance(item, ParityFinding) else ParityFinding.from_dict(item)
            for item in (self.findings or ())
        )
        if len(findings) > DEFAULT_MAX_FINDINGS:
            raise InterfaceParityBoundsError("too many findings in report")
        object.__setattr__(self, "findings", findings)
        object.__setattr__(
            self, "verdict", _enum(self.verdict, ReportVerdict, "verdict")
        )
        object.__setattr__(
            self,
            "checker_version",
            _text(self.checker_version, "checker_version"),
        )
        if not isinstance(self.truncated, bool):
            raise InterfaceParityError("truncated must be a boolean")
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
        return "icprpt-" + _content_id(self._identity_payload())

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
            "schema": INTERFACE_PARITY_REPORT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "checker_version": self.checker_version,
            "inventory_id": self.inventory_id,
            "policy_id": self.policy_id,
            "forest_id": self.forest_id,
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
    def from_dict(cls, payload: Mapping[str, Any]) -> "InterfaceParityReport":
        if not isinstance(payload, Mapping):
            raise InterfaceParityError("report payload must be a mapping")
        return cls(
            inventory_id=str(payload.get("inventory_id") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            tools=tuple(payload.get("tools") or ()),
            findings=tuple(payload.get("findings") or ()),
            verdict=payload.get("verdict", ReportVerdict.UNKNOWN.value),
            forest_id=str(payload.get("forest_id") or ""),
            contract_pack_id=str(payload.get("contract_pack_id") or ""),
            checker_version=str(payload.get("checker_version") or CHECKER_VERSION),
            truncated=bool(payload.get("truncated", False)),
            truncation_reason=str(payload.get("truncation_reason") or ""),
            evidence_kinds=tuple(payload.get("evidence_kinds") or ()),
            notes=payload.get("notes") or {},
        )


# ---------------------------------------------------------------------------
# View construction
# ---------------------------------------------------------------------------


def _artifact_is_mock_or_fallback(
    item: SurfaceArtifact,
    policy: ToolSelectionPolicy,
) -> bool:
    reason = item.non_invocation_reason
    if reason in {
        "mock_implementation",
        "legacy_fallback",
        "test_server",
        "static_dashboard",
    }:
        return True
    if item.role in {"mock", "legacy_fallback", "test_server", "static_dashboard"}:
        return True
    markers = {m.lower() for m in item.markers}
    return bool(markers & {"mock", "fallback", "legacy_fallback", "stub"})


def _artifact_is_local_bypass(item: SurfaceArtifact) -> bool:
    if item.role == "local_helper":
        return True
    if item.non_invocation_reason == "same_name_helper":
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
    surface: str,
    artifacts: Sequence[SurfaceArtifact],
    *,
    tool_name: str,
    policy: ToolSelectionPolicy,
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
        schema_fingerprint(dict(item.input_schema))
        for item in ordered
        if item.input_schema
    }
    output_fps = {
        schema_fingerprint(dict(item.output_schema))
        for item in ordered
        if item.output_schema
    }
    errors: set[str] = set()
    profiles: set[str] = set()
    transports: set[str] = set()
    for item in ordered:
        errors.update(item.error_codes)
        profiles.update(item.profiles)
        if item.transport:
            transports.add(item.transport)
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
            item.implementation_target
            or item.record.get("implementation")
            or item.record.get("implementation_target")
            or ""
        ).strip()
        if not target and item.role == "implementation":
            target = item.qualified_name
        if target:
            impl_targets.append(target)
    mock = any(_artifact_is_mock_or_fallback(item, policy) for item in ordered)
    bypass = any(_artifact_is_local_bypass(item) for item in ordered)
    return SurfaceView(
        surface=surface,
        present=True,
        tool_name=primary.effective_tool_name or tool_name,
        qualified_name=primary.qualified_name,
        language=primary.language,
        package=primary.package,
        version=primary.version,
        input_schema_fingerprint=(
            sorted(input_fps)[0]
            if len(input_fps) == 1
            else ("multi:" + ",".join(sorted(input_fps)) if input_fps else "")
        ),
        output_schema_fingerprint=(
            sorted(output_fps)[0]
            if len(output_fps) == 1
            else ("multi:" + ",".join(sorted(output_fps)) if output_fps else "")
        ),
        error_codes=tuple(sorted(errors)),
        transport=(
            next(iter(sorted(transports)))
            if len(transports) == 1
            else ("multi:" + ",".join(sorted(transports)) if transports else "")
        ),
        profiles=tuple(sorted(profiles)),
        alias_of=primary.alias_of,
        implementation_target=(
            impl_targets[0]
            if len(set(impl_targets)) == 1
            else ("multi:" + ",".join(sorted(set(impl_targets))) if impl_targets else "")
        ),
        capability_claims=tuple(sorted(set(capability))),
        degradation_claims=tuple(sorted(set(degradation))),
        artifact_ids=tuple(item.artifact_id for item in ordered),
        has_call_edge=any(item.has_call_edge for item in ordered),
        is_generated=is_generated
        or any(item.role in policy.generated_roles for item in ordered)
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
    inventory: SurfaceInventory,
    tool_name: str,
    policy: ToolSelectionPolicy,
    *,
    role: str | None = None,
    language: str = "",
    roles: Sequence[str] | None = None,
    artifact_kind: str = "",
) -> tuple[SurfaceArtifact, ...]:
    aliases = set(policy.aliases_for(tool_name))
    role_set = set(roles or ())
    if role:
        role_set.add(role)
    matched: list[SurfaceArtifact] = []
    for item in inventory.artifacts:
        if role_set and item.role not in role_set:
            continue
        if language and item.language and item.language != language:
            continue
        if artifact_kind:
            kind = str(item.record.get("artifact_kind") or "")
            if kind and kind != artifact_kind:
                continue
            # For SDK surfaces without explicit artifact_kind, require language match
            # already handled; markers may include "sdk".
            if not kind and artifact_kind == "typescript_sdk":
                markers = {m.lower() for m in item.markers}
                if "sdk" not in markers and item.language != "typescript":
                    continue
        candidates = {
            _tool_key(item.effective_tool_name),
            _tool_key(item.name),
            _tool_key(item.alias_of) if item.alias_of else "",
            *policy.aliases_for(item.effective_tool_name),
            *policy.aliases_for(item.name),
        }
        candidates.discard("")
        if aliases & candidates:
            matched.append(item)
    return tuple(sorted(matched, key=lambda item: item.artifact_id))


def discover_tool_names(
    inventory: SurfaceInventory,
    policy: ToolSelectionPolicy,
    *,
    server_name: str = "",
) -> tuple[str, ...]:
    """Discover tool names present on any tool-bearing surface."""

    if not isinstance(inventory, SurfaceInventory):
        raise InterfaceParityError("inventory must be SurfaceInventory")
    if not isinstance(policy, ToolSelectionPolicy):
        raise InterfaceParityError("policy must be ToolSelectionPolicy")
    server = server_name or policy.server_name
    names: set[str] = set()
    for item in inventory.artifacts:
        if item.role not in policy.tool_bearing_roles:
            continue
        if server and item.server_name and item.server_name != server:
            continue
        name = item.effective_tool_name or item.name
        if not name:
            continue
        if not policy.accepts_tool(name):
            continue
        names.add(name)
    return tuple(sorted(names, key=lambda n: (_tool_key(n), n)))


def build_surface_views(
    inventory: SurfaceInventory,
    tool_name: str,
    policy: ToolSelectionPolicy,
    *,
    contract_adapter: ContractProfileAdapter | None = None,
    runtime_observations: Sequence[RuntimeWitnessObservation] = (),
) -> dict[str, SurfaceView]:
    """Build surface views for one tool from inventory evidence + policy."""

    if not isinstance(inventory, SurfaceInventory):
        raise InterfaceParityError("inventory must be SurfaceInventory")
    if not isinstance(policy, ToolSelectionPolicy):
        raise InterfaceParityError("policy must be ToolSelectionPolicy")
    tool_name = _text(tool_name, "tool_name")
    views: dict[str, SurfaceView] = {}

    for spec in policy.surface_specs:
        if spec.kind == "runtime_witness":
            continue  # filled below
        if spec.kind == "contract_pack":
            continue  # filled below
        if spec.kind == "capability_degradation":
            # Filled after registration/connector views exist.
            continue

        languages = spec.languages
        if languages:
            matched: list[SurfaceArtifact] = []
            for lang in languages:
                matched.extend(
                    _match_tool(
                        inventory,
                        tool_name,
                        policy,
                        roles=spec.roles,
                        language=lang,
                        artifact_kind=spec.artifact_kind,
                    )
                )
            # De-dupe by artifact_id
            by_id = {a.artifact_id: a for a in matched}
            arts = tuple(sorted(by_id.values(), key=lambda a: a.artifact_id))
        else:
            arts = _match_tool(
                inventory,
                tool_name,
                policy,
                roles=spec.roles,
                artifact_kind=spec.artifact_kind,
            )
            # For typescript_sdk without language filter on empty roles match,
            # prefer typescript language artifacts with sdk marker.
            if spec.kind == "typescript_sdk":
                arts = tuple(
                    a
                    for a in arts
                    if a.language == "typescript"
                    or "sdk" in {m.lower() for m in a.markers}
                    or str(a.record.get("artifact_kind") or "") == "typescript_sdk"
                )
            if spec.kind == "json_manifest":
                arts = tuple(
                    a
                    for a in arts
                    if a.language in {"", "json"}
                    or str(a.record.get("artifact_kind") or "") == "json_manifest"
                )
            if spec.kind == "python_signature":
                arts = _match_tool(
                    inventory,
                    tool_name,
                    policy,
                    role="implementation",
                    language="python",
                )

        views[spec.kind] = _view_from_artifacts(
            spec.kind,
            arts,
            tool_name=tool_name,
            policy=policy,
            prefer_language=spec.prefer_language,
            is_generated=spec.generated,
        )

    # Capability / degradation surface folds claims from registration + connector.
    reg = views.get("registration") or SurfaceView.absent(
        "registration", tool_name=tool_name
    )
    connector = views.get("connector") or SurfaceView.absent(
        "connector", tool_name=tool_name
    )
    pack_caps: list[str] = []
    pack_deg: list[str] = []
    if contract_adapter is not None:
        proj = contract_adapter.projection_for(tool_name)
        pack_caps.extend(proj.get("capability_claims") or ())
        pack_deg.extend(proj.get("degradation_claims") or ())
    cap_claims = _sorted_unique(
        list(reg.capability_claims)
        + list(connector.capability_claims)
        + pack_caps
    )
    deg_claims = _sorted_unique(
        list(reg.degradation_claims)
        + list(connector.degradation_claims)
        + pack_deg
    )
    views["capability_degradation"] = SurfaceView(
        surface="capability_degradation",
        present=bool(cap_claims or deg_claims or reg.present or connector.present),
        tool_name=tool_name,
        capability_claims=cap_claims,
        degradation_claims=deg_claims,
        artifact_ids=tuple(
            sorted(set(reg.artifact_ids) | set(connector.artifact_ids))
        ),
        notes={"contract_pack_bound": contract_adapter is not None},
    )

    if contract_adapter is not None:
        proj = contract_adapter.projection_for(tool_name)
        views["contract_pack"] = SurfaceView(
            surface="contract_pack",
            present=True,
            tool_name=tool_name,
            qualified_name=contract_adapter.content_id,
            capability_claims=tuple(proj.get("capability_claims") or ()),
            degradation_claims=tuple(proj.get("degradation_claims") or ()),
            notes={
                "contract_pack_id": contract_adapter.content_id,
                "operation": str(proj.get("operation") or ""),
                "entrypoint": str(proj.get("entrypoint") or ""),
                "unresolved": bool(proj.get("unresolved")),
            },
        )
    elif "contract_pack" in policy.surface_kinds():
        views["contract_pack"] = SurfaceView.absent(
            "contract_pack", tool_name=tool_name
        )

    # Runtime witness surface (optional, non-authoritative for mocks).
    runtime_views: list[SurfaceView] = []
    aliases = policy.aliases_for(tool_name)
    for observation in runtime_observations:
        if not isinstance(observation, RuntimeWitnessObservation):
            # Duck-type thin wrappers.
            observation = RuntimeWitnessObservation(
                tool_name=str(getattr(observation, "tool_name", "") or tool_name),
                implementation_kind=str(
                    getattr(observation, "implementation_kind", "") or ""
                ),
                implementation_target=str(
                    getattr(observation, "implementation_target", "") or ""
                ),
                outcome=str(getattr(observation, "outcome", "") or ""),
                grants_runtime_authority=bool(
                    getattr(observation, "grants_runtime_authority", False)
                ),
                is_mock=bool(getattr(observation, "is_mock", False)),
                transport=str(getattr(observation, "transport", "") or ""),
                profiles=tuple(getattr(observation, "profiles", ()) or ()),
                error_codes=tuple(getattr(observation, "error_codes", ()) or ()),
                receipt_id=str(getattr(observation, "receipt_id", "") or ""),
                fixture_id=str(getattr(observation, "fixture_id", "") or ""),
            )
        if _tool_key(observation.tool_name) not in aliases and observation.tool_name not in aliases:
            continue
        mock = observation.is_mock or observation.implementation_kind.lower() in {
            "mock",
            "fixture",
            "stub",
            "fallback",
        }
        runtime_views.append(
            SurfaceView(
                surface="runtime_witness",
                present=True,
                tool_name=tool_name,
                qualified_name=observation.implementation_target or observation.receipt_id,
                error_codes=observation.error_codes,
                transport=observation.transport,
                profiles=observation.profiles,
                implementation_target=observation.implementation_target,
                artifact_ids=(observation.receipt_id,) if observation.receipt_id else (),
                is_mock_or_fallback=mock,
                notes={
                    "outcome": observation.outcome,
                    "implementation_kind": observation.implementation_kind,
                    "authoritative": bool(
                        observation.grants_runtime_authority and not mock
                    ),
                    "fixture_id": observation.fixture_id,
                },
            )
        )
    if runtime_views:
        runtime_views.sort(
            key=lambda v: (0 if not v.is_mock_or_fallback else 1, v.view_id)
        )
        views["runtime_witness"] = runtime_views[0]
    else:
        views["runtime_witness"] = SurfaceView.absent(
            "runtime_witness", tool_name=tool_name
        )

    # Ensure every policy surface kind is present (absent if missing).
    for kind in policy.surface_kinds():
        if kind not in views:
            views[kind] = SurfaceView.absent(kind, tool_name=tool_name)
    return views


# ---------------------------------------------------------------------------
# Comparison rules
# ---------------------------------------------------------------------------


def _make_finding(
    kind: ParityFindingKind,
    tool_name: str,
    summary: str,
    witnesses: Sequence[ParityWitness],
    *,
    surfaces: Sequence[str] = (),
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
    left_value: str = "",
    right_value: str = "",
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
        left_ref=(left.artifact_ids[0] if left.artifact_ids else left.qualified_name),
        right_ref=(
            right.artifact_ids[0] if right.artifact_ids else right.qualified_name
        ),
        path_id=path_id,
        path_verdict=path_verdict,
        notes=notes or {},
    )


def _present_name_surfaces(
    views: Mapping[str, SurfaceView],
    policy: ToolSelectionPolicy,
) -> tuple[SurfaceView, ...]:
    name_kinds = policy.name_bearing_kinds()
    present = [
        views[kind]
        for kind in sorted(name_kinds)
        if kind in views and views[kind].present and views[kind].tool_name
    ]
    return tuple(present)


def text_names_agree(
    views: Mapping[str, SurfaceView],
    policy: ToolSelectionPolicy | None = None,
) -> bool:
    """True when all present name-bearing surfaces share an alias-compatible name."""

    policy = policy or ToolSelectionPolicy(policy_id="default-text-names")
    present = _present_name_surfaces(views, policy)
    if len(present) < 2:
        return len(present) == 1
    alias_sets = [set(policy.aliases_for(v.tool_name)) for v in present]
    shared = set.intersection(*alias_sets) if alias_sets else set()
    return bool(shared)


def call_path_is_proved(
    path: ResolvedCallPath,
    policy: ToolSelectionPolicy | None = None,
) -> bool:
    """Whether a path is a fully resolved invocation (not text-only)."""

    if not isinstance(path, ResolvedCallPath):
        raise InterfaceParityError("path must be ResolvedCallPath")
    if path.verdict is not PathVerdict.PROVED:
        return False
    policy = policy or ToolSelectionPolicy(policy_id="default-path-proof")
    stage_values = {hop.stage for hop in path.hops}
    for stage in policy.required_proved_stages:
        if stage not in stage_values:
            return False
    for hop in path.hops:
        if hop.stage == "caller":
            continue
        if hop.status is not HopStatus.RESOLVED_STATIC:
            return False
        if hop.reason_code in policy.non_invocation_reasons:
            return False
    return True


def _paths_for_tool(
    paths: Sequence[ResolvedCallPath],
    tool_name: str,
    policy: ToolSelectionPolicy,
) -> tuple[ResolvedCallPath, ...]:
    aliases = policy.aliases_for(tool_name)
    matched = [
        path
        for path in paths
        if _tool_key(path.tool_name) in aliases or path.tool_name in aliases
    ]
    return tuple(sorted(matched, key=lambda p: p.path_id))


def compare_tool_surfaces(
    tool_name: str,
    views: Mapping[str, SurfaceView],
    *,
    policy: ToolSelectionPolicy | None = None,
    paths: Sequence[ResolvedCallPath] = (),
    drift_witnesses: Sequence[DriftWitnessRecord] = (),
) -> ToolParityResult:
    """Compare surface views for one tool and emit findings + verdict."""

    policy = policy or ToolSelectionPolicy(policy_id="default-compare")
    tool_name = _text(tool_name, "tool_name")
    views = {str(k): (v if isinstance(v, SurfaceView) else SurfaceView.from_dict(v)) for k, v in views.items()}

    tool_paths = _paths_for_tool(paths, tool_name, policy)
    path_ids = tuple(p.path_id for p in tool_paths)
    path_verdicts = tuple(p.verdict.value for p in tool_paths)
    proved = any(call_path_is_proved(p, policy) for p in tool_paths)
    names_agree = text_names_agree(views, policy)
    findings: list[ParityFinding] = []

    def _get(kind: str) -> SurfaceView:
        return views.get(kind) or SurfaceView.absent(kind, tool_name=tool_name)

    reg = _get("registration")
    listed = _get("tools_list")
    manifest = _get("json_manifest")
    sdk = _get("typescript_sdk")
    connector = _get("connector")
    transport = _get("transport_profile")
    result_err = _get("result_error_map")
    py_sig = _get("python_signature")
    impl = _get("implementation_target")
    cap = _get("capability_degradation")
    runtime = _get("runtime_witness")

    primary_path_id = path_ids[0] if path_ids else ""
    primary_path_verdict = path_verdicts[0] if path_verdicts else ""

    # --- Explicit unresolved paths ---
    for path in tool_paths:
        if path.verdict is PathVerdict.UNRESOLVED:
            findings.append(
                _make_finding(
                    ParityFindingKind.UNRESOLVED_PATH,
                    tool_name,
                    f"call path {path.path_id!r} remains unresolved",
                    (
                        ParityWitness(
                            kind=ParityFindingKind.UNRESOLVED_PATH,
                            tool_name=tool_name,
                            left_surface="connector",
                            right_surface="implementation_target",
                            left_value=path.connector_ref or path.caller_ref,
                            right_value=path.implementation_ref or "",
                            path_id=path.path_id,
                            path_verdict=path.verdict.value,
                            notes={"explicit": True},
                        ),
                    ),
                    surfaces=("connector", "implementation_target"),
                    path_ids=(path.path_id,),
                )
            )

    # --- Missing registration ---
    if (listed.present or manifest.present or sdk.present) and not reg.present:
        left = listed if listed.present else (manifest if manifest.present else sdk)
        findings.append(
            _make_finding(
                ParityFindingKind.MISSING_REGISTRATION,
                tool_name,
                f"tool {tool_name!r} appears on {left.surface} but has no registration",
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
                surfaces=(left.surface, "registration"),
                path_ids=path_ids,
            )
        )

    # --- Extra unreachable registration ---
    if reg.present and not listed.present and not proved:
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
                surfaces=("registration", "tools_list"),
                path_ids=path_ids,
            )
        )

    # --- Schema mismatches ---
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
                        f"{left.surface} and {right.surface}"
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
                        f"{left.surface} and {right.surface}"
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

    # --- Error map mismatches ---
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
                    surfaces=("registration", "result_error_map"),
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
                    surfaces=("tools_list", "registration"),
                    path_ids=path_ids,
                )
            )

    # --- Stale generated artifacts ---
    for gen in (manifest, sdk):
        if not (gen.present and reg.present):
            continue
        if gen.version and reg.version and gen.version != reg.version:
            findings.append(
                _make_finding(
                    ParityFindingKind.STALE_GENERATED_ARTIFACT,
                    tool_name,
                    (
                        f"generated {gen.surface} version {gen.version!r} "
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
                    surfaces=(gen.surface, "registration"),
                    path_ids=path_ids,
                )
            )
        if (
            gen.is_generated
            and gen.input_schema_fingerprint
            and reg.input_schema_fingerprint
            and gen.input_schema_fingerprint != reg.input_schema_fingerprint
            and not gen.input_schema_fingerprint.startswith("multi:")
            and not reg.input_schema_fingerprint.startswith("multi:")
        ):
            if not gen.version or not reg.version or gen.version == reg.version:
                findings.append(
                    _make_finding(
                        ParityFindingKind.STALE_GENERATED_ARTIFACT,
                        tool_name,
                        (
                            f"generated {gen.surface} schema is stale relative "
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
                        surfaces=(gen.surface, "registration"),
                        path_ids=path_ids,
                    )
                )

    # --- Wrong aliases ---
    if sdk.present and sdk.alias_of and reg.present:
        reg_aliases = set(policy.aliases_for(reg.tool_name))
        alias_key = _tool_key(sdk.alias_of)
        alias_binds = (
            alias_key in reg_aliases
            or sdk.alias_of in reg_aliases
            or bool(set(policy.aliases_for(sdk.alias_of)) & reg_aliases)
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
                    surfaces=("typescript_sdk", "registration"),
                    path_ids=path_ids,
                )
            )

    # --- Name mismatches ---
    present_named = _present_name_surfaces(views, policy)
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

    # --- Python signature vs registration ---
    if py_sig.present and reg.present and py_sig.tool_name and reg.tool_name:
        if not (set(policy.aliases_for(py_sig.tool_name)) & set(policy.aliases_for(reg.tool_name))):
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
                    surfaces=("python_signature", "registration"),
                    path_ids=path_ids,
                )
            )

    # --- Transport / profile ---
    if transport.present and connector.present:
        if (
            transport.transport
            and connector.transport
            and not transport.transport.startswith("multi:")
            and not connector.transport.startswith("multi:")
            and transport.transport != connector.transport
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
                            left_value=transport.transport,
                            right_value=connector.transport,
                            path_id=primary_path_id,
                            path_verdict=primary_path_verdict,
                        ),
                    ),
                    surfaces=("transport_profile", "connector"),
                    path_ids=path_ids,
                )
            )
        if transport.profiles and connector.profiles:
            if not set(transport.profiles) & set(connector.profiles):
                findings.append(
                    _make_finding(
                        ParityFindingKind.PROFILE_MISMATCH,
                        tool_name,
                        "no overlapping profiles between transport and connector",
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
                        surfaces=("transport_profile", "connector"),
                        path_ids=path_ids,
                    )
                )

    # --- Implementation target consistency ---
    if impl.present and reg.present:
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
                    surfaces=("implementation_target", "registration"),
                    path_ids=path_ids,
                )
            )

    # --- Local bypass / mock-fallback on surfaces ---
    for view in views.values():
        if not view.present:
            continue
        if view.is_local_bypass:
            findings.append(
                _make_finding(
                    ParityFindingKind.DIRECT_LOCAL_BYPASS,
                    tool_name,
                    f"surface {view.surface} dispatches via direct local bypass",
                    (
                        _pair_witness(
                            ParityFindingKind.DIRECT_LOCAL_BYPASS,
                            tool_name,
                            view,
                            reg if reg.present else view,
                            left_value=view.qualified_name or view.tool_name,
                            right_value="local_bypass",
                            path_id=primary_path_id,
                            path_verdict=primary_path_verdict,
                        ),
                    ),
                    surfaces=(view.surface,),
                    path_ids=path_ids,
                    confidence=0,
                )
            )
        if view.is_mock_or_fallback and view.surface != "runtime_witness":
            findings.append(
                _make_finding(
                    ParityFindingKind.MOCK_FALLBACK_DISPATCH,
                    tool_name,
                    f"surface {view.surface} is mock/fallback dispatch",
                    (
                        _pair_witness(
                            ParityFindingKind.MOCK_FALLBACK_DISPATCH,
                            tool_name,
                            view,
                            reg if reg.present else view,
                            left_value=view.qualified_name or view.tool_name,
                            right_value="mock_or_fallback",
                            path_id=primary_path_id,
                            path_verdict=primary_path_verdict,
                        ),
                    ),
                    surfaces=(view.surface,),
                    path_ids=path_ids,
                    confidence=0,
                )
            )

    # --- Runtime mock authority ---
    if runtime.present and runtime.is_mock_or_fallback:
        auth = bool(runtime.notes.get("authoritative"))
        if auth or str(runtime.notes.get("outcome") or "").lower() in {
            "passed",
            "pass",
            "ok",
            "success",
        }:
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
                            left_value=str(
                                runtime.notes.get("implementation_kind") or "mock"
                            ),
                            right_value=impl.implementation_target or "unknown",
                            path_id=primary_path_id,
                            path_verdict=primary_path_verdict,
                        ),
                    ),
                    surfaces=("runtime_witness", "implementation_target"),
                    path_ids=path_ids,
                    confidence=0,
                )
            )

    # --- Capability / degradation ---
    if cap.present and reg.present:
        if connector.present and cap.capability_claims and connector.profiles:
            claim_profiles = {
                c.split(":", 1)[-1]
                for c in cap.capability_claims
                if c.startswith("profile:") or "/" in c
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
                        surfaces=("capability_degradation", "connector"),
                        path_ids=path_ids,
                        confidence=70,
                    )
                )
        silent = {
            c.lower()
            for c in cap.degradation_claims
            if "silent" in c.lower()
            or c.lower() in _FORBIDDEN_SILENT_DEGRADATION
            or "placeholder" in c.lower()
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
                    surfaces=("capability_degradation", "registration"),
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
                    f"call path {path.path_id!r} is ambiguous",
                    (
                        ParityWitness(
                            kind=ParityFindingKind.AMBIGUOUS_PATH,
                            tool_name=tool_name,
                            left_surface="connector",
                            right_surface="implementation_target",
                            left_value=path.connector_ref or path.caller_ref,
                            right_value=path.implementation_ref or "",
                            left_ref=path.connector_ref,
                            right_ref=path.implementation_ref,
                            path_id=path.path_id,
                            path_verdict=path.verdict.value,
                            evidence_refs=tuple(
                                aid
                                for hop in path.hops
                                for aid in hop.artifact_ids
                            ),
                            notes={"path_id": path.path_id},
                        ),
                    ),
                    surfaces=("connector", "implementation_target"),
                    path_ids=(path.path_id,),
                    confidence=25,
                )
            )
        if path.verdict is PathVerdict.REJECTED:
            reasons = {
                hop.reason_code
                for hop in path.hops
                if hop.reason_code in policy.non_invocation_reasons
            }
            if reasons & {
                "mock_implementation",
                "legacy_fallback",
                "test_server",
            } and not any(
                f.kind is ParityFindingKind.MOCK_FALLBACK_DISPATCH for f in findings
            ):
                findings.append(
                    _make_finding(
                        ParityFindingKind.MOCK_FALLBACK_DISPATCH,
                        tool_name,
                        f"call path {path.path_id!r} rejected as mock/fallback",
                        (
                            ParityWitness(
                                kind=ParityFindingKind.MOCK_FALLBACK_DISPATCH,
                                tool_name=tool_name,
                                left_surface="implementation_target",
                                right_surface="registration",
                                left_value=",".join(sorted(reasons)),
                                right_value=path.verdict.value,
                                path_id=path.path_id,
                                path_verdict=path.verdict.value,
                            ),
                        ),
                        surfaces=("implementation_target",),
                        path_ids=(path.path_id,),
                        confidence=0,
                    )
                )
            if "same_name_helper" in reasons and not any(
                f.kind is ParityFindingKind.DIRECT_LOCAL_BYPASS for f in findings
            ):
                findings.append(
                    _make_finding(
                        ParityFindingKind.DIRECT_LOCAL_BYPASS,
                        tool_name,
                        f"call path {path.path_id!r} rejected as same-name local helper",
                        (
                            ParityWitness(
                                kind=ParityFindingKind.DIRECT_LOCAL_BYPASS,
                                tool_name=tool_name,
                                left_surface="implementation_target",
                                right_surface="connector",
                                left_value="same_name_helper",
                                right_value=path.connector_ref or "",
                                path_id=path.path_id,
                                path_verdict=path.verdict.value,
                            ),
                        ),
                        surfaces=("implementation_target",),
                        path_ids=(path.path_id,),
                        confidence=0,
                    )
                )

    # --- Resolver drift witnesses ---
    for drift in drift_witnesses:
        if drift.tool_name and _tool_key(drift.tool_name) not in policy.aliases_for(
            tool_name
        ):
            continue
        kind = _DRIFT_KIND_TO_FINDING.get(drift.drift_kind)
        if kind is None:
            continue
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
        left_surface = "json_manifest"
        right_surface = "registration"
        if drift.drift_kind == "schema_mismatch":
            left_surface = "tools_list"
        elif drift.drift_kind == "transport_mismatch":
            left_surface = "transport_profile"
            right_surface = "connector"
        elif drift.drift_kind in {"error_map_mismatch", "result_map_mismatch"}:
            left_surface = "result_error_map"
        elif drift.drift_kind == "missing_registration":
            left_surface = "tools_list"
        elif drift.drift_kind == "extra_unreachable":
            left_surface = "registration"
            right_surface = "tools_list"
        findings.append(
            _make_finding(
                kind,
                tool_name,
                (
                    f"resolver drift {drift.drift_kind} for {tool_name!r}: "
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
                        evidence_refs=drift.evidence_refs,
                        notes={"drift_kind": drift.drift_kind},
                    ),
                ),
                surfaces=(left_surface, right_surface),
                path_ids=path_ids,
                notes={"source": "external_drift_witness"},
            )
        )

    # --- Same text without resolved call path is insufficient ---
    if names_agree and not proved:
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

    kinds = {f.kind for f in findings_t}
    if any(p.verdict is PathVerdict.AMBIGUOUS for p in tool_paths) or (
        ParityFindingKind.AMBIGUOUS_PATH in kinds
    ):
        if findings_t and kinds - {ParityFindingKind.AMBIGUOUS_PATH}:
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
            ParityFindingKind.FORGED_ARTIFACT,
            ParityFindingKind.UNBOUNDED_ARTIFACT,
        }
        for k in kinds
    ):
        verdict = ToolParityVerdict.REJECTED
    elif kinds - {
        ParityFindingKind.MISSING_RESOLVED_CALL_PATH,
        ParityFindingKind.CONTRACT_PACK_GAP,
        ParityFindingKind.UNRESOLVED_PATH,
    }:
        verdict = ToolParityVerdict.WITNESSED_DRIFT
    elif (
        ParityFindingKind.MISSING_RESOLVED_CALL_PATH in kinds
        or ParityFindingKind.UNRESOLVED_PATH in kinds
    ) and not proved:
        verdict = ToolParityVerdict.INSUFFICIENT_PATH
    elif findings_t:
        verdict = ToolParityVerdict.WITNESSED_DRIFT
    elif proved and not findings_t:
        verdict = ToolParityVerdict.PROVED_PARITY
    elif not any(v.present for v in views.values()):
        verdict = ToolParityVerdict.UNKNOWN
    else:
        verdict = ToolParityVerdict.UNKNOWN

    # Final fail-closed guard.
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
                            left_surface="registration",
                            right_surface="implementation_target",
                            left_value=reg.tool_name or tool_name,
                            right_value=impl.tool_name or "",
                            path_verdict="none",
                            notes={
                                "rule": "same_text_without_resolved_call_path_insufficient"
                            },
                        ),
                    ),
                    surfaces=("registration", "implementation_target"),
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
    if (
        ToolParityVerdict.WITNESSED_DRIFT in verdicts
        or ToolParityVerdict.REJECTED in verdicts
    ):
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
# Analyzer
# ---------------------------------------------------------------------------


class InterfaceContractParityAnalyzer:
    """End-to-end interface / manifest / SDK / transport parity analyzer."""

    def __init__(
        self,
        inventory: SurfaceInventory,
        policy: ToolSelectionPolicy,
        *,
        contract_adapter: ContractProfileAdapter | None = None,
    ) -> None:
        if not isinstance(inventory, SurfaceInventory):
            raise InterfaceParityError("inventory must be SurfaceInventory")
        if not isinstance(policy, ToolSelectionPolicy):
            raise InterfaceParityError("policy must be ToolSelectionPolicy")
        if contract_adapter is not None and not isinstance(
            contract_adapter, ContractProfileAdapter
        ):
            raise InterfaceParityError(
                "contract_adapter must be ContractProfileAdapter"
            )
        self._inventory = inventory
        self._policy = policy
        self._contract_adapter = contract_adapter

    @property
    def inventory(self) -> SurfaceInventory:
        return self._inventory

    @property
    def policy(self) -> ToolSelectionPolicy:
        return self._policy

    @property
    def contract_adapter(self) -> ContractProfileAdapter | None:
        return self._contract_adapter

    def check(
        self,
        *,
        tool_names: Sequence[str] | None = None,
        paths: Sequence[ResolvedCallPath] = (),
        drift_witnesses: Sequence[DriftWitnessRecord] = (),
        runtime_observations: Sequence[RuntimeWitnessObservation] = (),
    ) -> InterfaceParityReport:
        """Run end-to-end parity checks and return a content-addressed report."""

        if len(paths) > DEFAULT_MAX_PATHS:
            raise InterfaceParityBoundsError("too many call paths")
        for path in paths:
            if not isinstance(path, ResolvedCallPath):
                raise InterfaceParityError("paths must be ResolvedCallPath")

        if tool_names is None:
            names = list(discover_tool_names(self._inventory, self._policy))
            for path in paths:
                if path.tool_name and path.tool_name not in names:
                    if self._policy.accepts_tool(path.tool_name):
                        names.append(path.tool_name)
            names = sorted(set(names), key=lambda n: (_tool_key(n), n))
        else:
            names = [
                _text(name, "tool_name")
                for name in tool_names
                if str(name or "").strip()
            ]

        truncated = False
        truncation_reason = ""
        if len(names) > self._policy.max_tools:
            names = names[: self._policy.max_tools]
            truncated = True
            truncation_reason = "max_tools"

        tool_results: list[ToolParityResult] = []
        all_findings: list[ParityFinding] = []

        pack_findings = self._contract_pack_findings()
        all_findings.extend(pack_findings)

        for name in names:
            if len(all_findings) >= self._policy.max_findings:
                truncated = True
                truncation_reason = truncation_reason or "max_findings"
                break
            views = build_surface_views(
                self._inventory,
                name,
                self._policy,
                contract_adapter=self._contract_adapter,
                runtime_observations=runtime_observations,
            )
            tool_drift = tuple(
                item
                for item in drift_witnesses
                if not item.tool_name
                or _tool_key(item.tool_name) in self._policy.aliases_for(name)
            )
            result = compare_tool_surfaces(
                name,
                views,
                policy=self._policy,
                paths=paths,
                drift_witnesses=tool_drift,
            )
            tool_results.append(result)
            for finding in result.findings:
                if len(all_findings) >= self._policy.max_findings:
                    truncated = True
                    truncation_reason = truncation_reason or "max_findings"
                    break
                all_findings.append(finding)

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
        if self._contract_adapter is not None:
            pack_id = self._contract_adapter.content_id

        evidence = [
            EVIDENCE_INTERFACE_PARITY,
            EVIDENCE_CALL_PATH,
            EVIDENCE_MANIFEST_PARITY,
        ]
        if runtime_observations:
            evidence.append(EVIDENCE_RUNTIME_WITNESS)

        return InterfaceParityReport(
            inventory_id=self._inventory.inventory_id,
            policy_id=self._policy.policy_id,
            forest_id=self._inventory.forest_id,
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
                "drift_count": len(drift_witnesses),
                "required_proved_stages": list(self._policy.required_proved_stages),
                "surface_kinds": list(self._policy.surface_kinds()),
            },
        )

    def _contract_pack_findings(self) -> list[ParityFinding]:
        if self._contract_adapter is None:
            return []
        findings: list[ParityFinding] = []
        if self._contract_adapter.unresolved_operations:
            findings.append(
                _make_finding(
                    ParityFindingKind.CONTRACT_PACK_GAP,
                    "",
                    (
                        f"contract profile has "
                        f"{len(self._contract_adapter.unresolved_operations)} "
                        f"unresolved operations"
                    ),
                    (
                        ParityWitness(
                            kind=ParityFindingKind.CONTRACT_PACK_GAP,
                            tool_name="",
                            left_surface="contract_pack",
                            right_surface="registration",
                            left_value=self._contract_adapter.adapter_id,
                            right_value=",".join(
                                self._contract_adapter.unresolved_operations[:16]
                            ),
                            notes={
                                "unresolved_count": len(
                                    self._contract_adapter.unresolved_operations
                                )
                            },
                        ),
                    ),
                    surfaces=("contract_pack",),
                    confidence=60,
                )
            )
        missing_entry = [
            op
            for op in self._contract_adapter.operations
            if not self._contract_adapter.operation_entrypoints.get(op)
            and op not in self._contract_adapter.unresolved_operations
        ]
        # Only report if operations were declared supported via entrypoints map
        # emptiness while listed — keep as soft gap when entrypoints empty for all.
        if missing_entry and any(self._contract_adapter.operation_entrypoints.values()):
            findings.append(
                _make_finding(
                    ParityFindingKind.CONTRACT_PACK_GAP,
                    "",
                    (
                        "supported operations lack entrypoints: "
                        f"{', '.join(missing_entry[:8])}"
                    ),
                    (
                        ParityWitness(
                            kind=ParityFindingKind.CONTRACT_PACK_GAP,
                            tool_name="",
                            left_surface="contract_pack",
                            right_surface="registration",
                            left_value=self._contract_adapter.adapter_id,
                            right_value=",".join(missing_entry[:16]),
                            notes={"gap": "missing_entrypoint"},
                        ),
                    ),
                    surfaces=("contract_pack",),
                    confidence=50,
                )
            )
        return findings


def check_interface_parity(
    inventory: SurfaceInventory,
    policy: ToolSelectionPolicy,
    *,
    paths: Sequence[ResolvedCallPath] = (),
    drift_witnesses: Sequence[DriftWitnessRecord] = (),
    contract_adapter: ContractProfileAdapter | None = None,
    runtime_observations: Sequence[RuntimeWitnessObservation] = (),
    tool_names: Sequence[str] | None = None,
) -> InterfaceParityReport:
    """Check interface/manifest/SDK/transport parity end to end."""

    analyzer = InterfaceContractParityAnalyzer(
        inventory,
        policy,
        contract_adapter=contract_adapter,
    )
    return analyzer.check(
        tool_names=tool_names,
        paths=paths,
        drift_witnesses=drift_witnesses,
        runtime_observations=runtime_observations,
    )


def make_surface_view(**kwargs: Any) -> SurfaceView:
    """Construct a validated ``SurfaceView`` (test/helper entrypoint)."""

    return SurfaceView(**kwargs)


def make_parity_witness(**kwargs: Any) -> ParityWitness:
    """Construct a validated ``ParityWitness`` (test/helper entrypoint)."""

    return ParityWitness(**kwargs)


def report_content_identity(
    report: InterfaceParityReport | Mapping[str, Any],
) -> str:
    """Stable content identity for a parity report."""

    if isinstance(report, InterfaceParityReport):
        return report.report_id
    return "icprpt-" + _content_id(
        InterfaceParityReport.from_dict(report)._identity_payload()
    )


def finding_kinds() -> tuple[str, ...]:
    """Closed finding-kind vocabulary (deterministic order)."""

    return tuple(sorted(item.value for item in ParityFindingKind))


def parity_surfaces(
    policy: ToolSelectionPolicy | None = None,
) -> tuple[str, ...]:
    """Surface vocabulary for a policy (or neutral defaults)."""

    if policy is None:
        return tuple(sorted(DEFAULT_SURFACE_KINDS))
    return tuple(sorted(policy.surface_kinds()))


def proved_stages(
    policy: ToolSelectionPolicy | None = None,
) -> tuple[str, ...]:
    """Required proved stages for a policy (or neutral defaults)."""

    if policy is None:
        return DEFAULT_REQUIRED_PROVED_STAGES
    return policy.required_proved_stages


def make_proved_path(
    tool_name: str,
    *,
    path_id: str = "",
    policy: ToolSelectionPolicy | None = None,
    connector_ref: str = "GenericConnector.callTool",
    implementation_ref: str = "",
    transport: str = "http",
    profiles: Sequence[str] = ("rpc/basic",),
) -> ResolvedCallPath:
    """Build a minimal proved call path covering policy-required stages."""

    policy = policy or ToolSelectionPolicy(policy_id="default-proved-path")
    hops = tuple(
        CallPathHop(
            stage=stage,
            status=HopStatus.RESOLVED_STATIC,
            ref=f"{stage}:{tool_name}",
        )
        for stage in policy.required_proved_stages
    )
    return ResolvedCallPath(
        path_id=path_id or f"path:{tool_name}",
        tool_name=tool_name,
        verdict=PathVerdict.PROVED,
        hops=hops,
        connector_ref=connector_ref,
        implementation_ref=implementation_ref or f"impl:{tool_name}",
        transport=transport,
        profiles=tuple(profiles),
    )


__all__ = [
    "CHECKER_AUTHORIZES_REPAIR",
    "CHECKER_IS_COMPLETION_EVIDENCE",
    "CHECKER_IS_CORRECTNESS_EVIDENCE",
    "CHECKER_PRODUCER",
    "CHECKER_VERSION",
    "CONTRACT_VERSION",
    "DEFAULT_REQUIRED_PROVED_STAGES",
    "DEFAULT_SURFACE_KINDS",
    "EVIDENCE_CALL_PATH",
    "EVIDENCE_INTERFACE_PARITY",
    "EVIDENCE_MANIFEST_PARITY",
    "EVIDENCE_RUNTIME_WITNESS",
    "INTERFACE_CONTRACT_PARITY_SCHEMA",
    "INTERFACE_PARITY_FINDING_SCHEMA",
    "INTERFACE_PARITY_REPORT_SCHEMA",
    "INTERFACE_PARITY_WITNESS_SCHEMA",
    "INTERFACE_SURFACE_VIEW_SCHEMA",
    "INTERFACE_TOOL_PARITY_SCHEMA",
    "CallPathHop",
    "ContractProfileAdapter",
    "DriftWitnessRecord",
    "HopStatus",
    "InterfaceContractParityAnalyzer",
    "InterfaceParityBoundsError",
    "InterfaceParityError",
    "InterfaceParityReport",
    "ParityFinding",
    "ParityFindingKind",
    "ParitySeverity",
    "ParitySurfaceSpec",
    "ParityWitness",
    "PathVerdict",
    "ReportVerdict",
    "ResolvedCallPath",
    "RuntimeWitnessObservation",
    "SurfaceArtifact",
    "SurfaceInventory",
    "SurfaceView",
    "ToolParityResult",
    "ToolParityVerdict",
    "ToolSelectionPolicy",
    "build_surface_views",
    "call_path_is_proved",
    "check_interface_parity",
    "compare_tool_surfaces",
    "default_surface_specs",
    "discover_tool_names",
    "finding_kinds",
    "make_artifact",
    "make_parity_witness",
    "make_proved_path",
    "make_surface_view",
    "normalize_tool_name",
    "parity_surfaces",
    "proved_stages",
    "report_content_identity",
    "schema_fingerprint",
    "text_names_agree",
]
