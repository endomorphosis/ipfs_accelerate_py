"""Public-surface manifest for exported symbols (PCAR-013).

`PublicSurfaceManifest` classifies selected exports as stable, provisional,
internal, compatibility, deprecated, simulation, test_only, or
accidentally_public. Stable records bind owner, versioned schema, effects,
errors, authority, tests, proofs, and consumers. Python, CLI, and MCP
projections of one canonical operation are compared for parity. The manifest
observes owners and public contracts; it does not make internal symbols
public, deprecate them, or authorize removal.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ipfs_accelerate_py.utils.cid_utils import (
    canonical_dag_json_bytes,
    cid_for_dag_json,
    validate_cid,
)

from .architecture_ir import ArchitectureIR, ArchitectureNode
from .contracts import (
    ArchitectureContractError,
    Confidence,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    SourceSpan,
    _closed_enum,
    _require_int,
    _require_mapping,
    _require_text,
)

PUBLIC_SURFACE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/public-surface-manifest@1"
)
PUBLIC_SURFACE_VERSION = 1
PUBLIC_SURFACE_EVIDENCE = "pcar/public-surface-manifest@1"
STABLE_SYMBOL_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/stable-public-symbol@1"
)
STABLE_SYMBOL_VERSION = 1
EXPORT_RECORD_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/export-record@1"
)
EXPORT_RECORD_VERSION = 1
PROJECTION_FINDING_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/projection-parity-finding@1"
)
PROJECTION_FINDING_VERSION = 1
ACCIDENTAL_EXPORT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/accidental-export-finding@1"
)
ACCIDENTAL_EXPORT_VERSION = 1
REMOVAL_GATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/public-symbol-removal-gate@1"
)
REMOVAL_GATE_VERSION = 1
EXTRACTOR_IDENTITY = "pcar-013-public-surface-manifest"
TASK_ID = "PCAR-013"
DEFAULT_FRESHNESS = "pcar-013-public-surface"
EFFECT_CLASS = "read_only_analysis"
MANIFEST_CAN_AUTHORIZE_REMOVAL = False
MANIFEST_CAN_PROMOTE_INTERNAL = False
MANIFEST_CAN_DEPRECATE = False
MANIFEST_CAN_CHANGE_PUBLIC_API = False

_UNKNOWN_FIELD_MESSAGE = "unknown public-surface field"
_MISSING_FIELD_MESSAGE = "missing public-surface field"
_STABLE_INCOMPLETE_MESSAGE = "stable public symbol metadata is incomplete"
_EXPORT_CLOSURE_MESSAGE = "selected export is missing classification or provenance"
_DUPLICATE_CLASSIFICATION_MESSAGE = "selected export has conflicting classifications"
_CID_PREFIXES = ("bagu", "bafy", "bafk", "sha256:")
_PRIVATE_NAME = re.compile(r"^_[^_].*")
_PROJECT_SCRIPTS_HEADER = re.compile(r"^\[project\.scripts\]\s*$")
_TOML_ASSIGNMENT = re.compile(
    r'^([A-Za-z0-9_./-]+)\s*=\s*["\']([^"\']+)["\']\s*$'
)
_EFFECTFUL_CALLEES = frozenset(
    {
        "print",
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "exit",
        "quit",
        "breakpoint",
        "os.system",
        "os.popen",
        "os.remove",
        "os.unlink",
        "os.replace",
        "os.rename",
        "os.makedirs",
        "subprocess.run",
        "subprocess.call",
        "subprocess.Popen",
        "subprocess.check_call",
        "subprocess.check_output",
        "socket.socket",
        "urllib.request.urlopen",
        "requests.get",
        "requests.post",
        "pathlib.Path.write_text",
        "pathlib.Path.write_bytes",
        "sys.path.insert",
        "sys.path.append",
        "sys.exit",
    }
)
_FILESYSTEM_CALLEES = frozenset(
    {
        "open",
        "os.remove",
        "os.unlink",
        "os.replace",
        "os.rename",
        "os.makedirs",
        "pathlib.Path.write_text",
        "pathlib.Path.write_bytes",
    }
)
_PROCESS_CALLEES = frozenset(
    {
        "os.system",
        "os.popen",
        "subprocess.run",
        "subprocess.call",
        "subprocess.Popen",
        "subprocess.check_call",
        "subprocess.check_output",
        "sys.exit",
        "exit",
        "quit",
    }
)
_NETWORK_CALLEES = frozenset(
    {
        "socket.socket",
        "urllib.request.urlopen",
        "requests.get",
        "requests.post",
    }
)
_MUTATION_CALLEES = frozenset(
    {
        "sys.path.insert",
        "sys.path.append",
        "setattr",
        "delattr",
        "globals",
    }
)


class PublicSurfaceError(ArchitectureContractError):
    """Fail-closed public-surface contract violation."""


class PublicSurfaceAuthorityError(PublicSurfaceError):
    """Raised when the manifest is asked to change the public API."""


class ExportClassification(str, Enum):
    """Closed export-class vocabulary (PCAR-PLAN-R1)."""

    STABLE = "stable"
    PROVISIONAL = "provisional"
    INTERNAL = "internal"
    COMPATIBILITY = "compatibility"
    DEPRECATED = "deprecated"
    SIMULATION = "simulation"
    TEST_ONLY = "test_only"
    ACCIDENTALLY_PUBLIC = "accidentally_public"


CLOSED_EXPORT_CLASSES: frozenset[str] = frozenset(
    item.value for item in ExportClassification
)
REQUIRED_EXPORT_CLASSES: tuple[ExportClassification, ...] = tuple(ExportClassification)
PUBLIC_EXPORT_CLASSES: frozenset[ExportClassification] = frozenset(
    {
        ExportClassification.STABLE,
        ExportClassification.PROVISIONAL,
        ExportClassification.COMPATIBILITY,
        ExportClassification.DEPRECATED,
        ExportClassification.SIMULATION,
        ExportClassification.TEST_ONLY,
        ExportClassification.ACCIDENTALLY_PUBLIC,
    }
)


class ProjectionKind(str, Enum):
    """Typed projections of the canonical operation catalog."""

    PYTHON = "python"
    CLI = "cli"
    MCP = "mcp"


CLOSED_PROJECTIONS: frozenset[str] = frozenset(item.value for item in ProjectionKind)
REQUIRED_PROJECTIONS: tuple[ProjectionKind, ...] = tuple(ProjectionKind)


class ConsumerEvidenceKind(str, Enum):
    """Whether consumers of a public symbol are inventoried."""

    KNOWN = "known"
    UNKNOWN = "unknown"


CLOSED_CONSUMER_EVIDENCE: frozenset[str] = frozenset(
    item.value for item in ConsumerEvidenceKind
)


class ImportLaziness(str, Enum):
    """Whether an import is deferred until use."""

    LAZY = "lazy"
    EAGER = "eager"
    UNKNOWN = "unknown"


CLOSED_IMPORT_LAZINESS: frozenset[str] = frozenset(item.value for item in ImportLaziness)


class ImportEffectKind(str, Enum):
    """Closed import-time effect vocabulary."""

    NONE = "none"
    FILESYSTEM = "filesystem"
    PROCESS = "process"
    NETWORK = "network"
    MUTATION = "mutation"
    EXCEPTION = "exception"
    UNKNOWN = "unknown"


CLOSED_IMPORT_EFFECTS: frozenset[str] = frozenset(item.value for item in ImportEffectKind)


class AccidentalExportKind(str, Enum):
    """Closed accidental-export finding vocabulary."""

    UNDECLARED_PUBLIC = "undeclared_public"
    INTERNAL_REEXPORT = "internal_reexport"
    PRIVATE_NAME_IN_ALL = "private_name_in_all"
    STAR_REEXPORT = "star_reexport"
    WILDCARD_SURFACE = "wildcard_surface"


CLOSED_ACCIDENTAL_KINDS: frozenset[str] = frozenset(
    item.value for item in AccidentalExportKind
)


class ProjectionMismatchKind(str, Enum):
    """Closed Python/CLI/MCP projection-parity vocabulary."""

    MISSING_PYTHON = "missing_python"
    MISSING_CLI = "missing_cli"
    MISSING_MCP = "missing_mcp"
    SCHEMA_MISMATCH = "schema_mismatch"
    VERSION_MISMATCH = "version_mismatch"
    EFFECT_MISMATCH = "effect_mismatch"
    ERROR_MISMATCH = "error_mismatch"
    SEMANTIC_INVENTION = "semantic_invention"


CLOSED_PROJECTION_MISMATCHES: frozenset[str] = frozenset(
    item.value for item in ProjectionMismatchKind
)


class RemovalBlockerKind(str, Enum):
    """Closed blockers that prevent public-symbol removal."""

    UNKNOWN_CONSUMERS = "unknown_consumers"
    CONSUMERS_REMAIN = "consumers_remain"
    NOT_DEPRECATED = "not_deprecated"
    MISSING_REPLACEMENT = "missing_replacement"
    MISSING_COMPATIBILITY = "missing_compatibility"
    MISSING_NEGATIVE_IMPORT_TESTS = "missing_negative_import_tests"
    MISSING_RELEASE_NOTES = "missing_release_notes"
    STILL_EXPORTED = "still_exported"
    MANIFEST_CANNOT_AUTHORIZE = "manifest_cannot_authorize"


CLOSED_REMOVAL_BLOCKERS: frozenset[str] = frozenset(
    item.value for item in RemovalBlockerKind
)


class DiscoveryOrigin(str, Enum):
    """How a selected export entered the manifest."""

    DECLARATION = "declaration"
    ALL_LIST = "all_list"
    PACKAGE_REEXPORT = "package_reexport"
    ENTRYPOINT = "entrypoint"
    CLI_REGISTRY = "cli_registry"
    MCP_REGISTRY = "mcp_registry"
    PYPROJECT_SCRIPT = "pyproject_script"
    ARCHITECTURE_IR = "architecture_ir"


CLOSED_DISCOVERY_ORIGINS: frozenset[str] = frozenset(
    item.value for item in DiscoveryOrigin
)

_STABLE_FIELDS = frozenset(
    {
        "authority",
        "consumers",
        "content_identity",
        "effects",
        "errors",
        "owner",
        "proofs",
        "schema",
        "symbol",
        "tests",
        "version",
    }
)
_IMPORT_ASSESSMENT_FIELDS = frozenset(
    {
        "effects",
        "imported_symbol",
        "laziness",
        "module",
        "provenance",
        "side_effect_free",
    }
)
_EXPORT_FIELDS = frozenset(
    {
        "classification",
        "consumer_evidence",
        "consumers",
        "content_identity",
        "import_assessment",
        "origins",
        "projections",
        "provenance",
        "qualified_name",
        "schema",
        "stable",
        "symbol",
        "version",
    }
)
_ACCIDENTAL_FIELDS = frozenset(
    {
        "content_identity",
        "declared_classification",
        "kind",
        "message",
        "origins",
        "provenance",
        "schema",
        "symbol",
        "version",
    }
)
_PROJECTION_FINDING_FIELDS = frozenset(
    {
        "cli_schema",
        "content_identity",
        "kind",
        "mcp_schema",
        "message",
        "operation",
        "present_projections",
        "python_schema",
        "schema",
        "version",
    }
)
_REMOVAL_FIELDS = frozenset(
    {
        "blockers",
        "content_identity",
        "gates_satisfied",
        "schema",
        "symbol",
        "version",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "accidental_exports",
        "can_authorize_removal",
        "can_change_public_api",
        "can_deprecate",
        "can_promote_internal",
        "content_identity",
        "effect_class",
        "exports",
        "freshness",
        "import_traces",
        "projection_findings",
        "removal_gates",
        "repository_tree",
        "schema",
        "stable_symbols",
        "version",
    }
)
_DECLARATION_FIELDS = frozenset(
    {
        "authority",
        "classification",
        "consumer_evidence",
        "consumers",
        "effects",
        "errors",
        "module",
        "owner",
        "proofs",
        "projections",
        "provenance",
        "qualified_name",
        "schema",
        "symbol",
        "tests",
        "version",
    }
)
_PROJECTION_BINDING_FIELDS = frozenset(
    {
        "effects",
        "errors",
        "operation",
        "projection",
        "provenance",
        "schema",
        "version",
    }
)
_CONSUMER_FIELDS = frozenset(
    {
        "consumer",
        "kind",
        "provenance",
        "symbol",
    }
)
_REMOVAL_EVIDENCE_FIELDS = frozenset(
    {
        "compatibility_satisfied",
        "consumers_migrated",
        "deprecated",
        "negative_import_tests",
        "release_notes",
        "replacement",
        "still_exported",
        "symbol",
    }
)


def _content_identity(payload: Mapping[str, Any]) -> str:
    return cid_for_dag_json(payload)


def _validate_dag_json_cid(value: str) -> str:
    try:
        return validate_cid(value, codecs=("dag-json",))
    except (TypeError, ValueError) as exc:
        raise PublicSurfaceError("content identity must be a dag-json CIDv1") from exc


def _reject_unknown(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    extra = sorted(set(payload) - set(allowed))
    if extra:
        raise PublicSurfaceError(f"{_UNKNOWN_FIELD_MESSAGE}: {extra}")


def _require_fields(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    allowed_fields = set(allowed)
    _reject_unknown(payload, allowed_fields)
    missing = sorted(allowed_fields - set(payload))
    if missing:
        raise PublicSurfaceError(f"{_MISSING_FIELD_MESSAGE}: {missing}")


def _require_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PublicSurfaceError(f"{name} must be a list of strings")
    items = tuple(
        _require_text(item, f"{name} item", error_type=PublicSurfaceError)
        for item in value
    )
    return tuple(sorted(set(items)))


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise PublicSurfaceError(f"{name} must be a boolean")
    return value


def _looks_like_content_identity(value: str) -> bool:
    return value.startswith(_CID_PREFIXES)


def _wrap_contract(exc: ArchitectureContractError) -> PublicSurfaceError:
    if isinstance(exc, PublicSurfaceError):
        return exc
    return PublicSurfaceError(str(exc))


def _node_identity(node: ArchitectureNode) -> str:
    prefix = f"n:{node.kind.value}:"
    if node.node_id.startswith(prefix):
        return node.node_id[len(prefix) :]
    return node.node_id


def _symbol_leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1] if name else name


def _normalize_qualified(symbol: str, module: str, qualified_name: str) -> str:
    if qualified_name:
        return qualified_name
    if module:
        return f"{module}.{symbol}"
    return symbol


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _constant_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_strings(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        names = []
        for elt in node.elts:
            literal = _constant_str(elt)
            if literal:
                names.append(literal)
        return tuple(names)
    literal = _constant_str(node)
    return (literal,) if literal else ()


def _effect_kinds_for(callee: str) -> tuple[ImportEffectKind, ...]:
    kinds: list[ImportEffectKind] = []
    if callee in _FILESYSTEM_CALLEES or callee.endswith(".write_text") or callee.endswith(
        ".write_bytes"
    ):
        kinds.append(ImportEffectKind.FILESYSTEM)
    if callee in _PROCESS_CALLEES:
        kinds.append(ImportEffectKind.PROCESS)
    if callee in _NETWORK_CALLEES:
        kinds.append(ImportEffectKind.NETWORK)
    if callee in _MUTATION_CALLEES:
        kinds.append(ImportEffectKind.MUTATION)
    if callee in {"exec", "eval", "compile", "__import__"}:
        kinds.append(ImportEffectKind.UNKNOWN)
    if not kinds and callee in _EFFECTFUL_CALLEES:
        kinds.append(ImportEffectKind.MUTATION)
    return tuple(kinds)


@dataclass(frozen=True)
class SurfaceSourceBinding:
    """Current-tree observational binding for a public surface root."""

    name: str
    path: str
    nominated_symbol: str
    origin: DiscoveryOrigin
    start_line: int
    end_line: int


CURRENT_SURFACE_BINDINGS: tuple[SurfaceSourceBinding, ...] = (
    SurfaceSourceBinding(
        "package-root-all",
        "ipfs_accelerate_py/__init__.py",
        "__all__",
        DiscoveryOrigin.ALL_LIST,
        889,
        889,
    ),
    SurfaceSourceBinding(
        "pyproject-scripts",
        "pyproject.toml",
        "ipfs-accelerate",
        DiscoveryOrigin.PYPROJECT_SCRIPT,
        35,
        47,
    ),
    SurfaceSourceBinding(
        "cli-main",
        "ipfs_accelerate_py/cli_entry.py",
        "main",
        DiscoveryOrigin.ENTRYPOINT,
        12,
        12,
    ),
    SurfaceSourceBinding(
        "supervisor-cli-commands",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py",
        "SUPERVISOR_COMMANDS",
        DiscoveryOrigin.CLI_REGISTRY,
        24,
        33,
    ),
    SurfaceSourceBinding(
        "mcp-tool-registry",
        "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py",
        "AGENT_SUPERVISOR_OPERATION_TOOLS",
        DiscoveryOrigin.MCP_REGISTRY,
        222,
        227,
    ),
    SurfaceSourceBinding(
        "mcp-package-all",
        "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/__init__.py",
        "__all__",
        DiscoveryOrigin.ALL_LIST,
        47,
        77,
    ),
    SurfaceSourceBinding(
        "canonical-operation-catalog",
        "ipfs_accelerate_py/agent_supervisor/control/control_contracts.py",
        "OPERATION_CATALOG_V2",
        DiscoveryOrigin.DECLARATION,
        8346,
        8346,
    ),
)


@dataclass(frozen=True)
class ImportAssessment:
    """Laziness and import-time effects for one selected import."""

    module: str
    imported_symbol: str
    laziness: ImportLaziness
    effects: tuple[ImportEffectKind, ...]
    provenance: SourceFactIdentity
    side_effect_free: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "module",
            _require_text(self.module, "module", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "imported_symbol",
            _require_text(
                self.imported_symbol, "imported_symbol", error_type=PublicSurfaceError
            ),
        )
        object.__setattr__(
            self,
            "laziness",
            _closed_enum(
                self.laziness,
                ImportLaziness,
                "import laziness",
                error_type=PublicSurfaceError,
            ),
        )
        if isinstance(self.effects, ImportEffectKind):
            effects = (self.effects,)
        elif isinstance(self.effects, str):
            effects = (
                _closed_enum(
                    self.effects,
                    ImportEffectKind,
                    "import effect",
                    error_type=PublicSurfaceError,
                ),
            )
        elif isinstance(self.effects, (bytes, bytearray)) or not isinstance(
            self.effects, Sequence
        ):
            raise PublicSurfaceError("effects must be a list of import effects")
        else:
            effects = tuple(
                _closed_enum(
                    item,
                    ImportEffectKind,
                    "import effect",
                    error_type=PublicSurfaceError,
                )
                for item in self.effects
            )
        if not effects:
            effects = (ImportEffectKind.NONE,)
        ordered = tuple(sorted(set(effects), key=lambda item: item.value))
        if ImportEffectKind.NONE in ordered and len(ordered) > 1:
            ordered = tuple(item for item in ordered if item is not ImportEffectKind.NONE)
        object.__setattr__(self, "effects", ordered)
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        object.__setattr__(self, "provenance", provenance)
        side_effect_free = _require_bool(self.side_effect_free, "side_effect_free")
        inferred = not any(item is not ImportEffectKind.NONE for item in ordered)
        if side_effect_free != inferred:
            raise PublicSurfaceError(
                "side_effect_free must match recorded import-time effects"
            )
        object.__setattr__(self, "side_effect_free", inferred)

    def to_dict(self) -> dict[str, Any]:
        return {
            "effects": [item.value for item in self.effects],
            "imported_symbol": self.imported_symbol,
            "laziness": self.laziness.value,
            "module": self.module,
            "provenance": self.provenance.to_dict(),
            "side_effect_free": self.side_effect_free,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ImportAssessment":
        mapping = _require_mapping(payload, error_type=PublicSurfaceError)
        _require_fields(mapping, _IMPORT_ASSESSMENT_FIELDS)
        provenance_payload = mapping["provenance"]
        if not isinstance(provenance_payload, Mapping):
            raise PublicSurfaceError("import assessment provenance must be an object")
        try:
            return cls(
                module=mapping["module"],
                imported_symbol=mapping["imported_symbol"],
                laziness=mapping["laziness"],
                effects=mapping["effects"],
                provenance=SourceFactIdentity.from_mapping(provenance_payload),
                side_effect_free=mapping["side_effect_free"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc

    from_dict = from_mapping


@dataclass(frozen=True)
class StablePublicSymbolRecord:
    """Complete stable public-contract binding for one exported symbol."""

    symbol: str
    owner: str
    schema: str
    version: str
    effects: tuple[str, ...]
    errors: tuple[str, ...]
    authority: str
    tests: tuple[str, ...]
    proofs: tuple[str, ...]
    consumers: tuple[str, ...]
    content_identity: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "symbol",
            _require_text(self.symbol, "symbol", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "owner",
            _require_text(self.owner, "owner", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "schema",
            _require_text(self.schema, "schema", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "version",
            _require_text(self.version, "version", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "authority",
            _require_text(self.authority, "authority", error_type=PublicSurfaceError),
        )
        object.__setattr__(self, "effects", _require_text_tuple(self.effects, "effects"))
        object.__setattr__(self, "errors", _require_text_tuple(self.errors, "errors"))
        object.__setattr__(self, "tests", _require_text_tuple(self.tests, "tests"))
        object.__setattr__(self, "proofs", _require_text_tuple(self.proofs, "proofs"))
        object.__setattr__(
            self, "consumers", _require_text_tuple(self.consumers, "consumers")
        )
        if not self.tests:
            raise PublicSurfaceError(_STABLE_INCOMPLETE_MESSAGE)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=PublicSurfaceError,
                )
            )
            if claimed != identity:
                raise PublicSurfaceError("stable symbol content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "consumers": list(self.consumers),
            "effects": list(self.effects),
            "errors": list(self.errors),
            "owner": self.owner,
            "proofs": list(self.proofs),
            "schema": self.schema,
            "symbol": self.symbol,
            "tests": list(self.tests),
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise PublicSurfaceError("stable symbol content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StablePublicSymbolRecord":
        mapping = _require_mapping(payload, error_type=PublicSurfaceError)
        _require_fields(mapping, _STABLE_FIELDS)
        record = cls(
            symbol=mapping["symbol"],
            owner=mapping["owner"],
            schema=mapping["schema"],
            version=mapping["version"],
            effects=mapping["effects"],
            errors=mapping["errors"],
            authority=mapping["authority"],
            tests=mapping["tests"],
            proofs=mapping["proofs"],
            consumers=mapping["consumers"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise PublicSurfaceError("stable symbol content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class ExportRecord:
    """One selected export with exactly one classification and provenance."""

    symbol: str
    classification: ExportClassification
    provenance: SourceFactIdentity
    qualified_name: str = ""
    origins: tuple[DiscoveryOrigin, ...] = (DiscoveryOrigin.DECLARATION,)
    projections: tuple[ProjectionKind, ...] = ()
    consumers: tuple[str, ...] = ()
    consumer_evidence: ConsumerEvidenceKind = ConsumerEvidenceKind.UNKNOWN
    import_assessment: ImportAssessment | None = None
    stable: StablePublicSymbolRecord | None = None
    schema: str = EXPORT_RECORD_SCHEMA
    version: int = EXPORT_RECORD_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=PublicSurfaceError)
        if schema != EXPORT_RECORD_SCHEMA:
            raise PublicSurfaceError("unexpected export-record schema")
        version = _require_int(self.version, "version", error_type=PublicSurfaceError)
        if version != EXPORT_RECORD_VERSION:
            raise PublicSurfaceError("unexpected export-record version")
        symbol = _require_text(self.symbol, "symbol", error_type=PublicSurfaceError)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(
            self,
            "classification",
            _closed_enum(
                self.classification,
                ExportClassification,
                "export classification",
                error_type=PublicSurfaceError,
            ),
        )
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        object.__setattr__(self, "provenance", provenance)
        qualified = _require_text(
            self.qualified_name or symbol,
            "qualified_name",
            error_type=PublicSurfaceError,
        )
        object.__setattr__(self, "qualified_name", qualified)
        if isinstance(self.origins, DiscoveryOrigin):
            origins = (self.origins,)
        elif isinstance(self.origins, str):
            origins = (
                _closed_enum(
                    self.origins,
                    DiscoveryOrigin,
                    "discovery origin",
                    error_type=PublicSurfaceError,
                ),
            )
        elif isinstance(self.origins, (bytes, bytearray)) or not isinstance(
            self.origins, Sequence
        ):
            raise PublicSurfaceError("origins must be a list of discovery origins")
        else:
            origins = tuple(
                _closed_enum(
                    item,
                    DiscoveryOrigin,
                    "discovery origin",
                    error_type=PublicSurfaceError,
                )
                for item in self.origins
            )
        if not origins:
            raise PublicSurfaceError(_EXPORT_CLOSURE_MESSAGE)
        object.__setattr__(
            self,
            "origins",
            tuple(sorted(set(origins), key=lambda item: item.value)),
        )
        if isinstance(self.projections, ProjectionKind):
            projections = (self.projections,)
        elif isinstance(self.projections, str):
            projections = (
                _closed_enum(
                    self.projections,
                    ProjectionKind,
                    "projection",
                    error_type=PublicSurfaceError,
                ),
            )
        elif isinstance(self.projections, (bytes, bytearray)) or not isinstance(
            self.projections, Sequence
        ):
            raise PublicSurfaceError("projections must be a list of projection kinds")
        else:
            projections = tuple(
                _closed_enum(
                    item,
                    ProjectionKind,
                    "projection",
                    error_type=PublicSurfaceError,
                )
                for item in self.projections
            )
        object.__setattr__(
            self,
            "projections",
            tuple(sorted(set(projections), key=lambda item: item.value)),
        )
        object.__setattr__(
            self, "consumers", _require_text_tuple(self.consumers, "consumers")
        )
        object.__setattr__(
            self,
            "consumer_evidence",
            _closed_enum(
                self.consumer_evidence,
                ConsumerEvidenceKind,
                "consumer evidence",
                error_type=PublicSurfaceError,
            ),
        )
        assessment = self.import_assessment
        if assessment is None:
            object.__setattr__(self, "import_assessment", None)
        elif isinstance(assessment, ImportAssessment):
            object.__setattr__(self, "import_assessment", assessment)
        elif isinstance(assessment, Mapping):
            object.__setattr__(
                self, "import_assessment", ImportAssessment.from_mapping(assessment)
            )
        else:
            raise PublicSurfaceError("import_assessment must be an object or null")
        stable = self.stable
        if self.classification is ExportClassification.STABLE:
            if stable is None:
                raise PublicSurfaceError(_STABLE_INCOMPLETE_MESSAGE)
            if isinstance(stable, Mapping):
                stable = StablePublicSymbolRecord.from_mapping(stable)
            elif not isinstance(stable, StablePublicSymbolRecord):
                raise PublicSurfaceError("stable record must be an object")
            if stable.symbol not in {self.symbol, self.qualified_name}:
                raise PublicSurfaceError("stable record symbol mismatch")
            object.__setattr__(self, "stable", stable)
        else:
            if stable is not None:
                raise PublicSurfaceError(
                    "stable metadata is only valid for stable exports"
                )
            object.__setattr__(self, "stable", None)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=PublicSurfaceError,
                )
            )
            if claimed != identity:
                raise PublicSurfaceError("export record content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "classification": self.classification.value,
            "consumer_evidence": self.consumer_evidence.value,
            "consumers": list(self.consumers),
            "import_assessment": (
                None if self.import_assessment is None else self.import_assessment.to_dict()
            ),
            "origins": [item.value for item in self.origins],
            "projections": [item.value for item in self.projections],
            "provenance": self.provenance.to_dict(),
            "qualified_name": self.qualified_name,
            "schema": self.schema,
            "stable": None if self.stable is None else self.stable.to_dict(),
            "symbol": self.symbol,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise PublicSurfaceError("export record content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExportRecord":
        mapping = _require_mapping(payload, error_type=PublicSurfaceError)
        _require_fields(mapping, _EXPORT_FIELDS)
        provenance_payload = mapping["provenance"]
        if not isinstance(provenance_payload, Mapping):
            raise PublicSurfaceError("export provenance must be an object")
        try:
            record = cls(
                symbol=mapping["symbol"],
                classification=mapping["classification"],
                provenance=SourceFactIdentity.from_mapping(provenance_payload),
                qualified_name=mapping["qualified_name"],
                origins=mapping["origins"],
                projections=mapping["projections"],
                consumers=mapping["consumers"],
                consumer_evidence=mapping["consumer_evidence"],
                import_assessment=mapping["import_assessment"],
                stable=mapping["stable"],
                schema=mapping["schema"],
                version=mapping["version"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        if mapping["content_identity"] != record.content_identity:
            raise PublicSurfaceError("export record content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class AccidentalExportFinding:
    """One public name that is not an intentional public contract."""

    symbol: str
    kind: AccidentalExportKind
    provenance: SourceFactIdentity
    message: str
    origins: tuple[DiscoveryOrigin, ...] = (DiscoveryOrigin.ALL_LIST,)
    declared_classification: ExportClassification | None = None
    schema: str = ACCIDENTAL_EXPORT_SCHEMA
    version: int = ACCIDENTAL_EXPORT_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=PublicSurfaceError)
        if schema != ACCIDENTAL_EXPORT_SCHEMA:
            raise PublicSurfaceError("unexpected accidental-export schema")
        version = _require_int(self.version, "version", error_type=PublicSurfaceError)
        if version != ACCIDENTAL_EXPORT_VERSION:
            raise PublicSurfaceError("unexpected accidental-export version")
        object.__setattr__(
            self,
            "symbol",
            _require_text(self.symbol, "symbol", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "kind",
            _closed_enum(
                self.kind,
                AccidentalExportKind,
                "accidental export kind",
                error_type=PublicSurfaceError,
            ),
        )
        object.__setattr__(
            self,
            "message",
            _require_text(self.message, "message", error_type=PublicSurfaceError),
        )
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        object.__setattr__(self, "provenance", provenance)
        if isinstance(self.origins, DiscoveryOrigin):
            origins = (self.origins,)
        elif isinstance(self.origins, str):
            origins = (
                _closed_enum(
                    self.origins,
                    DiscoveryOrigin,
                    "discovery origin",
                    error_type=PublicSurfaceError,
                ),
            )
        elif isinstance(self.origins, (bytes, bytearray)) or not isinstance(
            self.origins, Sequence
        ):
            raise PublicSurfaceError("origins must be a list of discovery origins")
        else:
            origins = tuple(
                _closed_enum(
                    item,
                    DiscoveryOrigin,
                    "discovery origin",
                    error_type=PublicSurfaceError,
                )
                for item in self.origins
            )
        object.__setattr__(
            self,
            "origins",
            tuple(sorted(set(origins), key=lambda item: item.value)),
        )
        declared = self.declared_classification
        if declared is None:
            object.__setattr__(self, "declared_classification", None)
        else:
            object.__setattr__(
                self,
                "declared_classification",
                _closed_enum(
                    declared,
                    ExportClassification,
                    "export classification",
                    error_type=PublicSurfaceError,
                ),
            )
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=PublicSurfaceError,
                )
            )
            if claimed != identity:
                raise PublicSurfaceError("accidental export content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "declared_classification": (
                None
                if self.declared_classification is None
                else self.declared_classification.value
            ),
            "kind": self.kind.value,
            "message": self.message,
            "origins": [item.value for item in self.origins],
            "provenance": self.provenance.to_dict(),
            "schema": self.schema,
            "symbol": self.symbol,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise PublicSurfaceError("accidental export content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AccidentalExportFinding":
        mapping = _require_mapping(payload, error_type=PublicSurfaceError)
        _require_fields(mapping, _ACCIDENTAL_FIELDS)
        provenance_payload = mapping["provenance"]
        if not isinstance(provenance_payload, Mapping):
            raise PublicSurfaceError("accidental export provenance must be an object")
        try:
            finding = cls(
                symbol=mapping["symbol"],
                kind=mapping["kind"],
                provenance=SourceFactIdentity.from_mapping(provenance_payload),
                message=mapping["message"],
                origins=mapping["origins"],
                declared_classification=mapping["declared_classification"],
                schema=mapping["schema"],
                version=mapping["version"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        if mapping["content_identity"] != finding.content_identity:
            raise PublicSurfaceError("accidental export content identity mismatch")
        return finding

    from_dict = from_mapping


@dataclass(frozen=True)
class ProjectionParityFinding:
    """One Python/CLI/MCP divergence from the canonical operation catalog."""

    operation: str
    kind: ProjectionMismatchKind
    message: str
    present_projections: tuple[ProjectionKind, ...]
    python_schema: str = ""
    cli_schema: str = ""
    mcp_schema: str = ""
    schema: str = PROJECTION_FINDING_SCHEMA
    version: int = PROJECTION_FINDING_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=PublicSurfaceError)
        if schema != PROJECTION_FINDING_SCHEMA:
            raise PublicSurfaceError("unexpected projection-finding schema")
        version = _require_int(self.version, "version", error_type=PublicSurfaceError)
        if version != PROJECTION_FINDING_VERSION:
            raise PublicSurfaceError("unexpected projection-finding version")
        object.__setattr__(
            self,
            "operation",
            _require_text(self.operation, "operation", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "kind",
            _closed_enum(
                self.kind,
                ProjectionMismatchKind,
                "projection mismatch kind",
                error_type=PublicSurfaceError,
            ),
        )
        object.__setattr__(
            self,
            "message",
            _require_text(self.message, "message", error_type=PublicSurfaceError),
        )
        if isinstance(self.present_projections, ProjectionKind):
            present = (self.present_projections,)
        elif isinstance(self.present_projections, str):
            present = (
                _closed_enum(
                    self.present_projections,
                    ProjectionKind,
                    "projection",
                    error_type=PublicSurfaceError,
                ),
            )
        elif isinstance(self.present_projections, (bytes, bytearray)) or not isinstance(
            self.present_projections, Sequence
        ):
            raise PublicSurfaceError(
                "present_projections must be a list of projection kinds"
            )
        else:
            present = tuple(
                _closed_enum(
                    item,
                    ProjectionKind,
                    "projection",
                    error_type=PublicSurfaceError,
                )
                for item in self.present_projections
            )
        object.__setattr__(
            self,
            "present_projections",
            tuple(sorted(set(present), key=lambda item: item.value)),
        )
        object.__setattr__(
            self,
            "python_schema",
            ""
            if self.python_schema == ""
            else _require_text(
                self.python_schema, "python_schema", error_type=PublicSurfaceError
            ),
        )
        object.__setattr__(
            self,
            "cli_schema",
            ""
            if self.cli_schema == ""
            else _require_text(
                self.cli_schema, "cli_schema", error_type=PublicSurfaceError
            ),
        )
        object.__setattr__(
            self,
            "mcp_schema",
            ""
            if self.mcp_schema == ""
            else _require_text(
                self.mcp_schema, "mcp_schema", error_type=PublicSurfaceError
            ),
        )
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=PublicSurfaceError,
                )
            )
            if claimed != identity:
                raise PublicSurfaceError(
                    "projection finding content identity mismatch"
                )
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "cli_schema": self.cli_schema,
            "kind": self.kind.value,
            "mcp_schema": self.mcp_schema,
            "message": self.message,
            "operation": self.operation,
            "present_projections": [item.value for item in self.present_projections],
            "python_schema": self.python_schema,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise PublicSurfaceError("projection finding content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProjectionParityFinding":
        mapping = _require_mapping(payload, error_type=PublicSurfaceError)
        _require_fields(mapping, _PROJECTION_FINDING_FIELDS)
        finding = cls(
            operation=mapping["operation"],
            kind=mapping["kind"],
            message=mapping["message"],
            present_projections=mapping["present_projections"],
            python_schema=mapping["python_schema"],
            cli_schema=mapping["cli_schema"],
            mcp_schema=mapping["mcp_schema"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != finding.content_identity:
            raise PublicSurfaceError("projection finding content identity mismatch")
        return finding

    from_dict = from_mapping


@dataclass(frozen=True)
class RemovalGateRecord:
    """Observed removal-precondition status. Never authorizes deletion."""

    symbol: str
    blockers: tuple[RemovalBlockerKind, ...]
    gates_satisfied: bool
    schema: str = REMOVAL_GATE_SCHEMA
    version: int = REMOVAL_GATE_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=PublicSurfaceError)
        if schema != REMOVAL_GATE_SCHEMA:
            raise PublicSurfaceError("unexpected removal-gate schema")
        version = _require_int(self.version, "version", error_type=PublicSurfaceError)
        if version != REMOVAL_GATE_VERSION:
            raise PublicSurfaceError("unexpected removal-gate version")
        object.__setattr__(
            self,
            "symbol",
            _require_text(self.symbol, "symbol", error_type=PublicSurfaceError),
        )
        if isinstance(self.blockers, RemovalBlockerKind):
            blockers = (self.blockers,)
        elif isinstance(self.blockers, str):
            blockers = (
                _closed_enum(
                    self.blockers,
                    RemovalBlockerKind,
                    "removal blocker",
                    error_type=PublicSurfaceError,
                ),
            )
        elif isinstance(self.blockers, (bytes, bytearray)) or not isinstance(
            self.blockers, Sequence
        ):
            raise PublicSurfaceError("blockers must be a list of removal blockers")
        else:
            blockers = tuple(
                _closed_enum(
                    item,
                    RemovalBlockerKind,
                    "removal blocker",
                    error_type=PublicSurfaceError,
                )
                for item in self.blockers
            )
        ordered = tuple(sorted(set(blockers), key=lambda item: item.value))
        if RemovalBlockerKind.MANIFEST_CANNOT_AUTHORIZE not in ordered:
            ordered = ordered + (RemovalBlockerKind.MANIFEST_CANNOT_AUTHORIZE,)
            ordered = tuple(sorted(set(ordered), key=lambda item: item.value))
        object.__setattr__(self, "blockers", ordered)
        gates = _require_bool(self.gates_satisfied, "gates_satisfied")
        if gates:
            raise PublicSurfaceError(
                "public-surface manifest cannot satisfy removal authorization"
            )
        object.__setattr__(self, "gates_satisfied", False)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=PublicSurfaceError,
                )
            )
            if claimed != identity:
                raise PublicSurfaceError("removal gate content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "blockers": [item.value for item in self.blockers],
            "gates_satisfied": self.gates_satisfied,
            "schema": self.schema,
            "symbol": self.symbol,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise PublicSurfaceError("removal gate content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RemovalGateRecord":
        mapping = _require_mapping(payload, error_type=PublicSurfaceError)
        _require_fields(mapping, _REMOVAL_FIELDS)
        record = cls(
            symbol=mapping["symbol"],
            blockers=mapping["blockers"],
            gates_satisfied=mapping["gates_satisfied"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise PublicSurfaceError("removal gate content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class ExportDeclaration:
    """Reviewed classification input for one selected export."""

    symbol: str
    classification: ExportClassification
    provenance: SourceFactIdentity
    qualified_name: str = ""
    module: str = ""
    owner: str = ""
    schema: str = ""
    version: str = ""
    effects: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    authority: str = ""
    tests: tuple[str, ...] = ()
    proofs: tuple[str, ...] = ()
    consumers: tuple[str, ...] = ()
    consumer_evidence: ConsumerEvidenceKind = ConsumerEvidenceKind.UNKNOWN
    projections: tuple[ProjectionKind, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "symbol",
            _require_text(self.symbol, "symbol", error_type=PublicSurfaceError),
        )
        if _looks_like_content_identity(self.symbol):
            raise PublicSurfaceError("content identity is not an exported symbol")
        object.__setattr__(
            self,
            "classification",
            _closed_enum(
                self.classification,
                ExportClassification,
                "export classification",
                error_type=PublicSurfaceError,
            ),
        )
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        object.__setattr__(self, "provenance", provenance)
        qualified = _normalize_qualified(
            self.symbol, self.module, self.qualified_name
        )
        object.__setattr__(
            self,
            "qualified_name",
            _require_text(qualified, "qualified_name", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "module",
            ""
            if self.module == ""
            else _require_text(self.module, "module", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "owner",
            ""
            if self.owner == ""
            else _require_text(self.owner, "owner", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "schema",
            ""
            if self.schema == ""
            else _require_text(self.schema, "schema", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "version",
            ""
            if self.version == ""
            else _require_text(self.version, "version", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "authority",
            ""
            if self.authority == ""
            else _require_text(
                self.authority, "authority", error_type=PublicSurfaceError
            ),
        )
        object.__setattr__(self, "effects", _require_text_tuple(self.effects, "effects"))
        object.__setattr__(self, "errors", _require_text_tuple(self.errors, "errors"))
        object.__setattr__(self, "tests", _require_text_tuple(self.tests, "tests"))
        object.__setattr__(self, "proofs", _require_text_tuple(self.proofs, "proofs"))
        object.__setattr__(
            self, "consumers", _require_text_tuple(self.consumers, "consumers")
        )
        object.__setattr__(
            self,
            "consumer_evidence",
            _closed_enum(
                self.consumer_evidence,
                ConsumerEvidenceKind,
                "consumer evidence",
                error_type=PublicSurfaceError,
            ),
        )
        if isinstance(self.projections, ProjectionKind):
            projections = (self.projections,)
        elif isinstance(self.projections, str):
            projections = (
                _closed_enum(
                    self.projections,
                    ProjectionKind,
                    "projection",
                    error_type=PublicSurfaceError,
                ),
            )
        elif isinstance(self.projections, (bytes, bytearray)) or not isinstance(
            self.projections, Sequence
        ):
            raise PublicSurfaceError("projections must be a list of projection kinds")
        else:
            projections = tuple(
                _closed_enum(
                    item,
                    ProjectionKind,
                    "projection",
                    error_type=PublicSurfaceError,
                )
                for item in self.projections
            )
        object.__setattr__(
            self,
            "projections",
            tuple(sorted(set(projections), key=lambda item: item.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "classification": self.classification.value,
            "consumer_evidence": self.consumer_evidence.value,
            "consumers": list(self.consumers),
            "effects": list(self.effects),
            "errors": list(self.errors),
            "module": self.module,
            "owner": self.owner,
            "proofs": list(self.proofs),
            "projections": [item.value for item in self.projections],
            "provenance": self.provenance.to_dict(),
            "qualified_name": self.qualified_name,
            "schema": self.schema,
            "symbol": self.symbol,
            "tests": list(self.tests),
            "version": self.version,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExportDeclaration":
        mapping = _require_mapping(payload, error_type=PublicSurfaceError)
        _require_fields(mapping, _DECLARATION_FIELDS)
        provenance_payload = mapping["provenance"]
        if not isinstance(provenance_payload, Mapping):
            raise PublicSurfaceError("export declaration provenance must be an object")
        try:
            return cls(
                symbol=mapping["symbol"],
                classification=mapping["classification"],
                provenance=SourceFactIdentity.from_mapping(provenance_payload),
                qualified_name=mapping["qualified_name"],
                module=mapping["module"],
                owner=mapping["owner"],
                schema=mapping["schema"],
                version=mapping["version"],
                effects=mapping["effects"],
                errors=mapping["errors"],
                authority=mapping["authority"],
                tests=mapping["tests"],
                proofs=mapping["proofs"],
                consumers=mapping["consumers"],
                consumer_evidence=mapping["consumer_evidence"],
                projections=mapping["projections"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc

    from_dict = from_mapping


@dataclass(frozen=True)
class ProjectionBinding:
    """One Python, CLI, or MCP projection of a canonical operation."""

    operation: str
    projection: ProjectionKind
    schema: str
    version: str
    provenance: SourceFactIdentity
    effects: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation",
            _require_text(self.operation, "operation", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "projection",
            _closed_enum(
                self.projection,
                ProjectionKind,
                "projection",
                error_type=PublicSurfaceError,
            ),
        )
        object.__setattr__(
            self,
            "schema",
            _require_text(self.schema, "schema", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "version",
            _require_text(self.version, "version", error_type=PublicSurfaceError),
        )
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "effects", _require_text_tuple(self.effects, "effects"))
        object.__setattr__(self, "errors", _require_text_tuple(self.errors, "errors"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "effects": list(self.effects),
            "errors": list(self.errors),
            "operation": self.operation,
            "projection": self.projection.value,
            "provenance": self.provenance.to_dict(),
            "schema": self.schema,
            "version": self.version,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProjectionBinding":
        mapping = _require_mapping(payload, error_type=PublicSurfaceError)
        _require_fields(mapping, _PROJECTION_BINDING_FIELDS)
        provenance_payload = mapping["provenance"]
        if not isinstance(provenance_payload, Mapping):
            raise PublicSurfaceError("projection binding provenance must be an object")
        try:
            return cls(
                operation=mapping["operation"],
                projection=mapping["projection"],
                schema=mapping["schema"],
                version=mapping["version"],
                provenance=SourceFactIdentity.from_mapping(provenance_payload),
                effects=mapping["effects"],
                errors=mapping["errors"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc

    from_dict = from_mapping


@dataclass(frozen=True)
class ConsumerReference:
    """One inventoried consumer of a selected export."""

    symbol: str
    consumer: str
    provenance: SourceFactIdentity
    kind: str = "import"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "symbol",
            _require_text(self.symbol, "symbol", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "consumer",
            _require_text(self.consumer, "consumer", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self,
            "kind",
            _require_text(self.kind, "kind", error_type=PublicSurfaceError),
        )
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        object.__setattr__(self, "provenance", provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumer": self.consumer,
            "kind": self.kind,
            "provenance": self.provenance.to_dict(),
            "symbol": self.symbol,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ConsumerReference":
        mapping = _require_mapping(payload, error_type=PublicSurfaceError)
        _require_fields(mapping, _CONSUMER_FIELDS)
        provenance_payload = mapping["provenance"]
        if not isinstance(provenance_payload, Mapping):
            raise PublicSurfaceError("consumer reference provenance must be an object")
        try:
            return cls(
                symbol=mapping["symbol"],
                consumer=mapping["consumer"],
                provenance=SourceFactIdentity.from_mapping(provenance_payload),
                kind=mapping["kind"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc

    from_dict = from_mapping


@dataclass(frozen=True)
class RemovalEvidence:
    """Candidate removal evidence. The manifest still cannot authorize."""

    symbol: str
    deprecated: bool = False
    replacement: str = ""
    consumers_migrated: bool = False
    compatibility_satisfied: bool = False
    negative_import_tests: tuple[str, ...] = ()
    release_notes: str = ""
    still_exported: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "symbol",
            _require_text(self.symbol, "symbol", error_type=PublicSurfaceError),
        )
        object.__setattr__(
            self, "deprecated", _require_bool(self.deprecated, "deprecated")
        )
        object.__setattr__(
            self,
            "replacement",
            ""
            if self.replacement == ""
            else _require_text(
                self.replacement, "replacement", error_type=PublicSurfaceError
            ),
        )
        object.__setattr__(
            self,
            "consumers_migrated",
            _require_bool(self.consumers_migrated, "consumers_migrated"),
        )
        object.__setattr__(
            self,
            "compatibility_satisfied",
            _require_bool(self.compatibility_satisfied, "compatibility_satisfied"),
        )
        object.__setattr__(
            self,
            "negative_import_tests",
            _require_text_tuple(self.negative_import_tests, "negative_import_tests"),
        )
        object.__setattr__(
            self,
            "release_notes",
            ""
            if self.release_notes == ""
            else _require_text(
                self.release_notes, "release_notes", error_type=PublicSurfaceError
            ),
        )
        object.__setattr__(
            self,
            "still_exported",
            _require_bool(self.still_exported, "still_exported"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatibility_satisfied": self.compatibility_satisfied,
            "consumers_migrated": self.consumers_migrated,
            "deprecated": self.deprecated,
            "negative_import_tests": list(self.negative_import_tests),
            "release_notes": self.release_notes,
            "replacement": self.replacement,
            "still_exported": self.still_exported,
            "symbol": self.symbol,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RemovalEvidence":
        mapping = _require_mapping(payload, error_type=PublicSurfaceError)
        _require_fields(mapping, _REMOVAL_EVIDENCE_FIELDS)
        return cls(
            symbol=mapping["symbol"],
            deprecated=mapping["deprecated"],
            replacement=mapping["replacement"],
            consumers_migrated=mapping["consumers_migrated"],
            compatibility_satisfied=mapping["compatibility_satisfied"],
            negative_import_tests=mapping["negative_import_tests"],
            release_notes=mapping["release_notes"],
            still_exported=mapping["still_exported"],
        )

    from_dict = from_mapping


@dataclass(frozen=True)
class DiscoveredExport:
    """A public name discovered without executing inspected modules."""

    symbol: str
    qualified_name: str
    origin: DiscoveryOrigin
    provenance: SourceFactIdentity
    private_name: bool = False
    star: bool = False


@dataclass(frozen=True)
class PublicSurfaceManifest:
    """Canonical public-surface classification for one repository tree."""

    repository_tree: str
    freshness: str
    exports: tuple[ExportRecord, ...]
    stable_symbols: tuple[StablePublicSymbolRecord, ...]
    accidental_exports: tuple[AccidentalExportFinding, ...]
    projection_findings: tuple[ProjectionParityFinding, ...]
    import_traces: tuple[ImportAssessment, ...]
    removal_gates: tuple[RemovalGateRecord, ...]
    schema: str = PUBLIC_SURFACE_SCHEMA
    version: int = PUBLIC_SURFACE_VERSION
    effect_class: str = EFFECT_CLASS
    can_authorize_removal: bool = False
    can_promote_internal: bool = False
    can_deprecate: bool = False
    can_change_public_api: bool = False
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=PublicSurfaceError)
        if schema != PUBLIC_SURFACE_SCHEMA:
            raise PublicSurfaceError("unexpected public-surface schema")
        version = _require_int(self.version, "version", error_type=PublicSurfaceError)
        if version != PUBLIC_SURFACE_VERSION:
            raise PublicSurfaceError("unexpected public-surface version")
        object.__setattr__(
            self,
            "repository_tree",
            _require_text(
                self.repository_tree, "repository_tree", error_type=PublicSurfaceError
            ),
        )
        object.__setattr__(
            self,
            "freshness",
            _require_text(self.freshness, "freshness", error_type=PublicSurfaceError),
        )
        effect_class = _require_text(
            self.effect_class, "effect_class", error_type=PublicSurfaceError
        )
        if effect_class != EFFECT_CLASS:
            raise PublicSurfaceError("public-surface effect class is read_only_analysis")
        if self.can_authorize_removal or self.can_promote_internal or self.can_deprecate:
            raise PublicSurfaceAuthorityError(
                "public-surface manifest cannot change the public API"
            )
        if self.can_change_public_api:
            raise PublicSurfaceAuthorityError(
                "public-surface manifest cannot change the public API"
            )
        object.__setattr__(self, "can_authorize_removal", False)
        object.__setattr__(self, "can_promote_internal", False)
        object.__setattr__(self, "can_deprecate", False)
        object.__setattr__(self, "can_change_public_api", False)
        exports = _record_tuple(self.exports, ExportRecord, "exports")
        seen: set[str] = set()
        for record in exports:
            key = record.qualified_name
            if key in seen:
                raise PublicSurfaceError(_DUPLICATE_CLASSIFICATION_MESSAGE)
            seen.add(key)
            if record.provenance.repository_tree != self.repository_tree:
                raise PublicSurfaceError(
                    "export provenance repository_tree must match the manifest"
                )
            if record.provenance.freshness != self.freshness:
                raise PublicSurfaceError(
                    "export provenance freshness must match the manifest"
                )
        object.__setattr__(self, "exports", exports)
        stables = _record_tuple(
            self.stable_symbols, StablePublicSymbolRecord, "stable_symbols"
        )
        export_stables = tuple(
            sorted(
                (record.stable for record in exports if record.stable is not None),
                key=lambda item: item.symbol,
            )
        )
        if tuple(item.symbol for item in stables) != tuple(
            item.symbol for item in export_stables
        ):
            raise PublicSurfaceError("stable_symbols must match stable export records")
        object.__setattr__(self, "stable_symbols", export_stables)
        object.__setattr__(
            self,
            "accidental_exports",
            _record_tuple(
                self.accidental_exports, AccidentalExportFinding, "accidental_exports"
            ),
        )
        object.__setattr__(
            self,
            "projection_findings",
            _record_tuple(
                self.projection_findings,
                ProjectionParityFinding,
                "projection_findings",
            ),
        )
        object.__setattr__(
            self,
            "import_traces",
            _record_tuple(self.import_traces, ImportAssessment, "import_traces"),
        )
        object.__setattr__(
            self,
            "removal_gates",
            _record_tuple(self.removal_gates, RemovalGateRecord, "removal_gates"),
        )
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=PublicSurfaceError,
                )
            )
            if claimed != identity:
                raise PublicSurfaceError("public-surface content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "accidental_exports": [item.to_dict() for item in self.accidental_exports],
            "can_authorize_removal": False,
            "can_change_public_api": False,
            "can_deprecate": False,
            "can_promote_internal": False,
            "effect_class": self.effect_class,
            "exports": [item.to_dict() for item in self.exports],
            "freshness": self.freshness,
            "import_traces": [item.to_dict() for item in self.import_traces],
            "projection_findings": [item.to_dict() for item in self.projection_findings],
            "removal_gates": [item.to_dict() for item in self.removal_gates],
            "repository_tree": self.repository_tree,
            "schema": self.schema,
            "stable_symbols": [item.to_dict() for item in self.stable_symbols],
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise PublicSurfaceError("public-surface content identity mismatch")
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PublicSurfaceManifest":
        mapping = _require_mapping(payload, error_type=PublicSurfaceError)
        _require_fields(mapping, _MANIFEST_FIELDS)
        try:
            manifest = cls(
                repository_tree=mapping["repository_tree"],
                freshness=mapping["freshness"],
                exports=mapping["exports"],
                stable_symbols=mapping["stable_symbols"],
                accidental_exports=mapping["accidental_exports"],
                projection_findings=mapping["projection_findings"],
                import_traces=mapping["import_traces"],
                removal_gates=mapping["removal_gates"],
                schema=mapping["schema"],
                version=mapping["version"],
                effect_class=mapping["effect_class"],
                can_authorize_removal=mapping["can_authorize_removal"],
                can_promote_internal=mapping["can_promote_internal"],
                can_deprecate=mapping["can_deprecate"],
                can_change_public_api=mapping["can_change_public_api"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        if mapping["content_identity"] != manifest.content_identity:
            raise PublicSurfaceError("public-surface content identity mismatch")
        return manifest

    from_dict = from_mapping

    @classmethod
    def from_json(cls, payload: str) -> "PublicSurfaceManifest":
        if type(payload) is not str or not payload:
            raise PublicSurfaceError("public-surface JSON must be a nonempty string")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise PublicSurfaceError("public-surface JSON is malformed") from exc
        if not isinstance(decoded, Mapping):
            raise PublicSurfaceError("public-surface JSON must contain an object")
        return cls.from_mapping(decoded)

    @property
    def covered_classifications(self) -> frozenset[ExportClassification]:
        return frozenset(item.classification for item in self.exports)

    @property
    def export_closure_complete(self) -> bool:
        return bool(self.exports) and all(
            item.classification in ExportClassification and item.provenance
            for item in self.exports
        )

    def export_for(self, symbol: str) -> ExportRecord:
        matches = [
            item
            for item in self.exports
            if item.symbol == symbol or item.qualified_name == symbol
        ]
        if not matches:
            raise PublicSurfaceError(f"unknown export: {symbol}")
        if len(matches) != 1:
            raise PublicSurfaceError(_DUPLICATE_CLASSIFICATION_MESSAGE)
        return matches[0]

    def stable_record(self, symbol: str) -> StablePublicSymbolRecord:
        record = self.export_for(symbol)
        if record.stable is None:
            raise PublicSurfaceError("export is not a complete stable public symbol")
        return record.stable

    def removal_gate(self, symbol: str) -> RemovalGateRecord:
        matches = [item for item in self.removal_gates if item.symbol == symbol]
        if not matches:
            raise PublicSurfaceError(f"unknown removal gate: {symbol}")
        return matches[0]

    def removal_blocked(self, symbol: str) -> bool:
        return True

    def authorize_removal(self, symbol: str) -> None:
        raise PublicSurfaceAuthorityError(
            "public-surface manifest cannot authorize removal"
        )

    def promote_internal(self, symbol: str) -> None:
        raise PublicSurfaceAuthorityError(
            "public-surface manifest does not make internal symbols public"
        )

    def deprecate_symbol(self, symbol: str) -> None:
        raise PublicSurfaceAuthorityError(
            "public-surface manifest does not deprecate symbols"
        )


def _record_tuple(value: Any, record_type: type, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PublicSurfaceError(f"{name} must be a list of objects")
    records = []
    for item in value:
        if isinstance(item, record_type):
            records.append(item)
        elif isinstance(item, Mapping):
            records.append(record_type.from_mapping(item))
        else:
            raise PublicSurfaceError(f"{name} must be a list of objects")
    if record_type is ExportRecord:
        key = lambda item: item.qualified_name
    elif record_type is StablePublicSymbolRecord:
        key = lambda item: item.symbol
    elif record_type is AccidentalExportFinding:
        key = lambda item: (item.kind.value, item.symbol)
    elif record_type is ProjectionParityFinding:
        key = lambda item: (item.kind.value, item.operation)
    elif record_type is ImportAssessment:
        key = lambda item: (item.module, item.imported_symbol, item.laziness.value)
    elif record_type is RemovalGateRecord:
        key = lambda item: item.symbol
    else:
        key = lambda item: repr(item)
    return tuple(sorted(records, key=key))


def _fact(
    path: str,
    start: int,
    end: int,
    *,
    repository_tree: str,
    freshness: str,
    confidence: Confidence = Confidence.EXACT,
    extractor_identity: str = EXTRACTOR_IDENTITY,
) -> SourceFactIdentity:
    return SourceFactIdentity(
        extractor_identity=extractor_identity,
        span=SourceSpan(path, start, end),
        confidence=confidence,
        freshness=freshness,
        repository_tree=repository_tree,
    )


def discover_exports_from_sources(
    sources: Mapping[str, str],
    *,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
) -> tuple[DiscoveredExport, ...]:
    """Discover public names from declared text without importing modules."""

    if not isinstance(sources, Mapping) or isinstance(sources, (str, bytes, bytearray)):
        raise PublicSurfaceError("sources must be an object mapping paths to text")
    discovered: list[DiscoveredExport] = []
    for raw_path, raw_text in sources.items():
        path = _require_text(raw_path, "source path", error_type=PublicSurfaceError)
        if type(raw_text) is not str:
            raise PublicSurfaceError("source text must be a string")
        if path.endswith(".toml"):
            discovered.extend(
                _discover_pyproject_scripts(
                    path, raw_text, repository_tree=repository_tree, freshness=freshness
                )
            )
            continue
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(raw_text)
        except SyntaxError as exc:
            raise PublicSurfaceError(f"source is not parseable: {path}") from exc
        discovered.extend(
            _discover_python_exports(
                path, tree, repository_tree=repository_tree, freshness=freshness
            )
        )
    ordered = sorted(
        discovered,
        key=lambda item: (item.origin.value, item.qualified_name, item.symbol),
    )
    return tuple(ordered)


def _discover_pyproject_scripts(
    path: str,
    text: str,
    *,
    repository_tree: str,
    freshness: str,
) -> list[DiscoveredExport]:
    discovered: list[DiscoveredExport] = []
    in_scripts = False
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_scripts = bool(_PROJECT_SCRIPTS_HEADER.match(stripped))
            continue
        if not in_scripts or not stripped or stripped.startswith("#"):
            continue
        match = _TOML_ASSIGNMENT.match(stripped)
        if match is None:
            continue
        name, target = match.group(1), match.group(2)
        symbol = target.rsplit(":", 1)[-1]
        discovered.append(
            DiscoveredExport(
                symbol=symbol,
                qualified_name=target,
                origin=DiscoveryOrigin.PYPROJECT_SCRIPT,
                provenance=_fact(
                    path, index, index, repository_tree=repository_tree, freshness=freshness
                ),
            )
        )
        discovered.append(
            DiscoveredExport(
                symbol=name,
                qualified_name=target,
                origin=DiscoveryOrigin.ENTRYPOINT,
                provenance=_fact(
                    path, index, index, repository_tree=repository_tree, freshness=freshness
                ),
            )
        )
    return discovered


def _discover_python_exports(
    path: str,
    tree: ast.Module,
    *,
    repository_tree: str,
    freshness: str,
) -> list[DiscoveredExport]:
    discovered: list[DiscoveredExport] = []
    module_name = _module_name_from_path(path)
    init_module = path.endswith("/__init__.py") or path.endswith("__init__.py")
    all_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    discovered.extend(
                        _discover_named_assignment(
                            path,
                            module_name,
                            target.id,
                            node.value,
                            node,
                            all_names,
                            repository_tree=repository_tree,
                            freshness=freshness,
                        )
                    )
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            discovered.extend(
                _discover_named_assignment(
                    path,
                    module_name,
                    node.target.id,
                    node.value,
                    node,
                    all_names,
                    repository_tree=repository_tree,
                    freshness=freshness,
                )
            )
        if isinstance(node, ast.FunctionDef) and node.name == "main" and init_module is False:
            start = int(getattr(node, "lineno", 1) or 1)
            end = int(getattr(node, "end_lineno", start) or start)
            discovered.append(
                DiscoveredExport(
                    symbol="main",
                    qualified_name=f"{module_name}.main",
                    origin=DiscoveryOrigin.ENTRYPOINT,
                    provenance=_fact(
                        path,
                        start,
                        end,
                        repository_tree=repository_tree,
                        freshness=freshness,
                    ),
                )
            )
        if isinstance(node, ast.ImportFrom):
            start = int(getattr(node, "lineno", 1) or 1)
            end = int(getattr(node, "end_lineno", start) or start)
            star = any(alias.name == "*" for alias in node.names)
            if star:
                discovered.append(
                    DiscoveredExport(
                        symbol="*",
                        qualified_name=f"{module_name}.*",
                        origin=DiscoveryOrigin.PACKAGE_REEXPORT,
                        provenance=_fact(
                            path,
                            start,
                            end,
                            repository_tree=repository_tree,
                            freshness=freshness,
                            confidence=Confidence.OPAQUE,
                        ),
                        star=True,
                    )
                )
            elif init_module:
                for alias in node.names:
                    local = alias.asname or alias.name
                    if local == "*":
                        continue
                    discovered.append(
                        DiscoveredExport(
                            symbol=local,
                            qualified_name=f"{module_name}.{local}",
                            origin=DiscoveryOrigin.PACKAGE_REEXPORT,
                            provenance=_fact(
                                path,
                                start,
                                end,
                                repository_tree=repository_tree,
                                freshness=freshness,
                            ),
                            private_name=bool(_PRIVATE_NAME.match(local)),
                        )
                    )
    return discovered


def _discover_named_assignment(
    path: str,
    module_name: str,
    name: str,
    value: ast.AST,
    node: ast.AST,
    all_names: set[str],
    *,
    repository_tree: str,
    freshness: str,
) -> list[DiscoveredExport]:
    start = int(getattr(node, "lineno", 1) or 1)
    end = int(getattr(node, "end_lineno", start) or start)
    discovered: list[DiscoveredExport] = []
    if name == "__all__":
        names = _literal_strings(value)
        if not names and not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            discovered.append(
                DiscoveredExport(
                    symbol="*",
                    qualified_name=f"{module_name}.*",
                    origin=DiscoveryOrigin.ALL_LIST,
                    provenance=_fact(
                        path,
                        start,
                        end,
                        repository_tree=repository_tree,
                        freshness=freshness,
                        confidence=Confidence.OPAQUE,
                    ),
                    star=True,
                )
            )
        if "*" in names:
            discovered.append(
                DiscoveredExport(
                    symbol="*",
                    qualified_name=f"{module_name}.*",
                    origin=DiscoveryOrigin.ALL_LIST,
                    provenance=_fact(
                        path,
                        start,
                        end,
                        repository_tree=repository_tree,
                        freshness=freshness,
                        confidence=Confidence.OPAQUE,
                    ),
                    star=True,
                )
            )
        for item in names:
            if item == "*":
                continue
            all_names.add(item)
            discovered.append(
                DiscoveredExport(
                    symbol=item,
                    qualified_name=f"{module_name}.{item}",
                    origin=DiscoveryOrigin.ALL_LIST,
                    provenance=_fact(
                        path,
                        start,
                        end,
                        repository_tree=repository_tree,
                        freshness=freshness,
                    ),
                    private_name=bool(_PRIVATE_NAME.match(item)),
                )
            )
        return discovered
    if name in {"SUPERVISOR_COMMANDS", "COMMANDS", "CLI_COMMANDS"}:
        for item in _literal_strings(value):
            discovered.append(
                DiscoveredExport(
                    symbol=item,
                    qualified_name=f"{module_name}.{item}",
                    origin=DiscoveryOrigin.CLI_REGISTRY,
                    provenance=_fact(
                        path,
                        start,
                        end,
                        repository_tree=repository_tree,
                        freshness=freshness,
                    ),
                )
            )
        return discovered
    if name in {
        "AGENT_SUPERVISOR_OPERATION_TOOLS",
        "PROMPT_LIFECYCLE_TOOLS",
        "OPERATION_TOOLS",
        "MCP_TOOLS",
    }:
        discovered.append(
            DiscoveredExport(
                symbol=name,
                qualified_name=f"{module_name}.{name}",
                origin=DiscoveryOrigin.MCP_REGISTRY,
                provenance=_fact(
                    path,
                    start,
                    end,
                    repository_tree=repository_tree,
                    freshness=freshness,
                ),
            )
        )
    return discovered


def _module_name_from_path(path: str) -> str:
    text = path.replace("\\", "/")
    if text.endswith(".py"):
        text = text[:-3]
    parts = [part for part in text.split("/") if part and part != "__init__"]
    return ".".join(parts) if parts else path


def discover_exports_from_architecture_ir(
    graph: ArchitectureIR,
) -> tuple[DiscoveredExport, ...]:
    """Select INTERFACE, ENTRYPOINT, and REEXPORTS facts as public-surface candidates."""

    nodes = {node.node_id: node for node in graph.nodes}
    discovered: list[DiscoveredExport] = []
    for node in graph.nodes:
        if node.kind not in {NodeKind.INTERFACE, NodeKind.ENTRYPOINT}:
            continue
        origin = (
            DiscoveryOrigin.ENTRYPOINT
            if node.kind is NodeKind.ENTRYPOINT
            else DiscoveryOrigin.ARCHITECTURE_IR
        )
        identity = _node_identity(node)
        discovered.append(
            DiscoveredExport(
                symbol=_symbol_leaf(identity),
                qualified_name=identity,
                origin=origin,
                provenance=node.provenance,
            )
        )
    for edge in graph.edges:
        if edge.kind is not EdgeKind.REEXPORTS:
            continue
        target = nodes.get(edge.target)
        if target is None:
            continue
        identity = _node_identity(target)
        discovered.append(
            DiscoveredExport(
                symbol=_symbol_leaf(identity),
                qualified_name=identity,
                origin=DiscoveryOrigin.PACKAGE_REEXPORT,
                provenance=edge.provenance,
            )
        )
    return tuple(
        sorted(
            discovered,
            key=lambda item: (item.origin.value, item.qualified_name, item.symbol),
        )
    )


def assess_imports_from_sources(
    sources: Mapping[str, str],
    *,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
) -> tuple[ImportAssessment, ...]:
    """Assess import laziness and effects from declared text, never executing it."""

    if not isinstance(sources, Mapping) or isinstance(sources, (str, bytes, bytearray)):
        raise PublicSurfaceError("sources must be an object mapping paths to text")
    traces: list[ImportAssessment] = []
    for raw_path, raw_text in sources.items():
        path = _require_text(raw_path, "source path", error_type=PublicSurfaceError)
        if not path.endswith(".py") or type(raw_text) is not str:
            continue
        try:
            tree = ast.parse(raw_text)
        except SyntaxError as exc:
            raise PublicSurfaceError(f"source is not parseable: {path}") from exc
        traces.extend(
            _assess_module_imports(
                path, tree, repository_tree=repository_tree, freshness=freshness
            )
        )
    return tuple(
        sorted(
            traces,
            key=lambda item: (item.module, item.imported_symbol, item.laziness.value),
        )
    )


def _assess_module_imports(
    path: str,
    tree: ast.Module,
    *,
    repository_tree: str,
    freshness: str,
) -> list[ImportAssessment]:
    module_name = _module_name_from_path(path)
    traces: list[ImportAssessment] = []
    effects = _module_effects(tree)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        lazy = _import_is_nested(tree, node)
        start = int(getattr(node, "lineno", 1) or 1)
        end = int(getattr(node, "end_lineno", start) or start)
        provenance = _fact(
            path,
            start,
            end,
            repository_tree=repository_tree,
            freshness=freshness,
            confidence=Confidence.EXACT if not node.names or node.names[0].name != "*" else Confidence.OPAQUE,
        )
        names = ["*"] if isinstance(node, ast.ImportFrom) and any(
            alias.name == "*" for alias in node.names
        ) else [alias.asname or alias.name for alias in node.names]
        imported_module = (
            node.names[0].name
            if isinstance(node, ast.Import)
            else (node.module or module_name)
        )
        recorded_effects = effects if not lazy else (ImportEffectKind.NONE,)
        for name in names:
            traces.append(
                ImportAssessment(
                    module=imported_module or module_name,
                    imported_symbol=name,
                    laziness=ImportLaziness.LAZY if lazy else ImportLaziness.EAGER,
                    effects=recorded_effects,
                    provenance=provenance,
                    side_effect_free=recorded_effects == (ImportEffectKind.NONE,),
                )
            )
    if not traces and effects != (ImportEffectKind.NONE,):
        start = 1
        traces.append(
            ImportAssessment(
                module=module_name,
                imported_symbol=module_name,
                laziness=ImportLaziness.EAGER,
                effects=effects,
                provenance=_fact(
                    path,
                    start,
                    start,
                    repository_tree=repository_tree,
                    freshness=freshness,
                    confidence=Confidence.CONSERVATIVE,
                ),
                side_effect_free=False,
            )
        )
    return traces


def _import_is_nested(tree: ast.Module, target: ast.AST) -> bool:
    for node in ast.walk(tree):
        if node is target:
            continue
        body: Sequence[ast.AST] | None = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
        if body is None:
            continue
        if any(child is target for child in ast.walk(node) if child is not node):
            return True
    return False


def _module_effects(tree: ast.Module) -> tuple[ImportEffectKind, ...]:
    kinds: set[ImportEffectKind] = set()
    for node in tree.body:
        if isinstance(node, ast.Raise):
            kinds.add(ImportEffectKind.EXCEPTION)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            kinds.update(_effect_kinds_for(_call_name(node.value.func)))
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            kinds.update(_effect_kinds_for(_call_name(node.value.func)))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
            kinds.update(_effect_kinds_for(_call_name(node.value.func)))
    if not kinds:
        return (ImportEffectKind.NONE,)
    return tuple(sorted(kinds, key=lambda item: item.value))


def detect_projection_mismatch(
    bindings: Sequence[ProjectionBinding | Mapping[str, Any]],
) -> tuple[ProjectionParityFinding, ...]:
    """Compare Python, CLI, and MCP projections of each canonical operation."""

    parsed = tuple(
        item if isinstance(item, ProjectionBinding) else ProjectionBinding.from_mapping(item)
        for item in bindings
    )
    by_operation: dict[str, dict[ProjectionKind, ProjectionBinding]] = {}
    for item in parsed:
        slot = by_operation.setdefault(item.operation, {})
        if item.projection in slot:
            existing = slot[item.projection]
            if (
                existing.schema != item.schema
                or existing.version != item.version
                or existing.effects != item.effects
                or existing.errors != item.errors
            ):
                raise PublicSurfaceError(
                    "conflicting projection bindings for the same operation"
                )
            continue
        slot[item.projection] = item
    findings: list[ProjectionParityFinding] = []
    for operation, present in sorted(by_operation.items()):
        kinds = tuple(sorted(present, key=lambda item: item.value))
        python = present.get(ProjectionKind.PYTHON)
        cli = present.get(ProjectionKind.CLI)
        mcp = present.get(ProjectionKind.MCP)
        python_schema = python.schema if python is not None else ""
        cli_schema = cli.schema if cli is not None else ""
        mcp_schema = mcp.schema if mcp is not None else ""
        if python is None and (cli is not None or mcp is not None):
            findings.append(
                ProjectionParityFinding(
                    operation=operation,
                    kind=ProjectionMismatchKind.SEMANTIC_INVENTION,
                    message="CLI or MCP projection is not present on the canonical Python catalog",
                    present_projections=kinds,
                    python_schema=python_schema,
                    cli_schema=cli_schema,
                    mcp_schema=mcp_schema,
                )
            )
            findings.append(
                ProjectionParityFinding(
                    operation=operation,
                    kind=ProjectionMismatchKind.MISSING_PYTHON,
                    message="canonical Python projection is missing",
                    present_projections=kinds,
                    python_schema=python_schema,
                    cli_schema=cli_schema,
                    mcp_schema=mcp_schema,
                )
            )
        if python is not None and cli is None:
            findings.append(
                ProjectionParityFinding(
                    operation=operation,
                    kind=ProjectionMismatchKind.MISSING_CLI,
                    message="CLI projection is missing for a canonical operation",
                    present_projections=kinds,
                    python_schema=python_schema,
                    cli_schema=cli_schema,
                    mcp_schema=mcp_schema,
                )
            )
        if python is not None and mcp is None:
            findings.append(
                ProjectionParityFinding(
                    operation=operation,
                    kind=ProjectionMismatchKind.MISSING_MCP,
                    message="MCP projection is missing for a canonical operation",
                    present_projections=kinds,
                    python_schema=python_schema,
                    cli_schema=cli_schema,
                    mcp_schema=mcp_schema,
                )
            )
        comparable = [item for item in (python, cli, mcp) if item is not None]
        if len(comparable) >= 2:
            schemas = {item.schema for item in comparable}
            versions = {item.version for item in comparable}
            effects = {item.effects for item in comparable}
            errors = {item.errors for item in comparable}
            if len(schemas) > 1:
                findings.append(
                    ProjectionParityFinding(
                        operation=operation,
                        kind=ProjectionMismatchKind.SCHEMA_MISMATCH,
                        message="projection schemas diverge from the canonical catalog",
                        present_projections=kinds,
                        python_schema=python_schema,
                        cli_schema=cli_schema,
                        mcp_schema=mcp_schema,
                    )
                )
            if len(versions) > 1:
                findings.append(
                    ProjectionParityFinding(
                        operation=operation,
                        kind=ProjectionMismatchKind.VERSION_MISMATCH,
                        message="projection versions diverge from the canonical catalog",
                        present_projections=kinds,
                        python_schema=python_schema,
                        cli_schema=cli_schema,
                        mcp_schema=mcp_schema,
                    )
                )
            if len(effects) > 1:
                findings.append(
                    ProjectionParityFinding(
                        operation=operation,
                        kind=ProjectionMismatchKind.EFFECT_MISMATCH,
                        message="projection effects diverge from the canonical catalog",
                        present_projections=kinds,
                        python_schema=python_schema,
                        cli_schema=cli_schema,
                        mcp_schema=mcp_schema,
                    )
                )
            if len(errors) > 1:
                findings.append(
                    ProjectionParityFinding(
                        operation=operation,
                        kind=ProjectionMismatchKind.ERROR_MISMATCH,
                        message="projection errors diverge from the canonical catalog",
                        present_projections=kinds,
                        python_schema=python_schema,
                        cli_schema=cli_schema,
                        mcp_schema=mcp_schema,
                    )
                )
    return tuple(
        sorted(findings, key=lambda item: (item.kind.value, item.operation))
    )


def detect_accidental_exports(
    discovered: Sequence[DiscoveredExport],
    declarations: Sequence[ExportDeclaration],
) -> tuple[AccidentalExportFinding, ...]:
    """Classify public names that are not intentional public contracts."""

    declared = {_declaration_key(item): item for item in declarations}
    findings: list[AccidentalExportFinding] = []
    seen: set[tuple[str, str]] = set()
    for item in discovered:
        if item.star:
            marker = ("*", item.qualified_name)
            if marker in seen:
                continue
            seen.add(marker)
            findings.append(
                AccidentalExportFinding(
                    symbol=item.symbol,
                    kind=(
                        AccidentalExportKind.STAR_REEXPORT
                        if item.origin is DiscoveryOrigin.PACKAGE_REEXPORT
                        else AccidentalExportKind.WILDCARD_SURFACE
                    ),
                    provenance=item.provenance,
                    message="star export expands the public surface opaquely",
                    origins=(item.origin,),
                )
            )
            continue
        if item.private_name:
            marker = ("private", item.qualified_name)
            if marker not in seen:
                seen.add(marker)
                findings.append(
                    AccidentalExportFinding(
                        symbol=item.symbol,
                        kind=AccidentalExportKind.PRIVATE_NAME_IN_ALL,
                        provenance=item.provenance,
                        message="private name appears on a public export list",
                        origins=(item.origin,),
                    )
                )
        declaration = _matching_declaration(item, declared)
        if declaration is None:
            marker = ("undeclared", item.qualified_name)
            if marker in seen:
                continue
            seen.add(marker)
            findings.append(
                AccidentalExportFinding(
                    symbol=item.symbol,
                    kind=AccidentalExportKind.UNDECLARED_PUBLIC,
                    provenance=item.provenance,
                    message="public name has no reviewed export classification",
                    origins=(item.origin,),
                )
            )
            continue
        if declaration.classification is ExportClassification.INTERNAL:
            marker = ("internal", item.qualified_name)
            if marker in seen:
                continue
            seen.add(marker)
            findings.append(
                AccidentalExportFinding(
                    symbol=item.symbol,
                    kind=AccidentalExportKind.INTERNAL_REEXPORT,
                    provenance=item.provenance,
                    message="internal symbol is publicly re-exported",
                    origins=(item.origin,),
                    declared_classification=ExportClassification.INTERNAL,
                )
            )
    return tuple(sorted(findings, key=lambda item: (item.kind.value, item.symbol)))


def _declaration_key(item: ExportDeclaration) -> str:
    return item.qualified_name or item.symbol


def _matching_declaration(
    discovered: DiscoveredExport,
    declared: Mapping[str, ExportDeclaration],
) -> ExportDeclaration | None:
    for key in (
        discovered.qualified_name,
        discovered.symbol,
        _symbol_leaf(discovered.qualified_name),
    ):
        if key in declared:
            return declared[key]
    for item in declared.values():
        if item.symbol == discovered.symbol or item.qualified_name == discovered.qualified_name:
            return item
        if _symbol_leaf(item.qualified_name) == discovered.symbol:
            return item
    return None


def classify_export(declaration: ExportDeclaration | Mapping[str, Any]) -> ExportRecord:
    """Bind one reviewed export to exactly one classification and provenance."""

    parsed = (
        declaration
        if isinstance(declaration, ExportDeclaration)
        else ExportDeclaration.from_mapping(declaration)
    )
    stable = None
    if parsed.classification is ExportClassification.STABLE:
        missing = [
            name
            for name, value in (
                ("owner", parsed.owner),
                ("schema", parsed.schema),
                ("version", parsed.version),
                ("authority", parsed.authority),
            )
            if not value
        ]
        if missing or not parsed.tests:
            raise PublicSurfaceError(_STABLE_INCOMPLETE_MESSAGE)
        stable = StablePublicSymbolRecord(
            symbol=parsed.symbol,
            owner=parsed.owner,
            schema=parsed.schema,
            version=parsed.version,
            effects=parsed.effects,
            errors=parsed.errors,
            authority=parsed.authority,
            tests=parsed.tests,
            proofs=parsed.proofs,
            consumers=parsed.consumers,
        )
    return ExportRecord(
        symbol=parsed.symbol,
        classification=parsed.classification,
        provenance=parsed.provenance,
        qualified_name=parsed.qualified_name,
        origins=(DiscoveryOrigin.DECLARATION,),
        projections=parsed.projections,
        consumers=parsed.consumers,
        consumer_evidence=parsed.consumer_evidence,
        stable=stable,
    )


def _stable_from_declaration(parsed: ExportDeclaration) -> StablePublicSymbolRecord:
    return StablePublicSymbolRecord(
        symbol=parsed.symbol,
        owner=parsed.owner,
        schema=parsed.schema,
        version=parsed.version,
        effects=parsed.effects,
        errors=parsed.errors,
        authority=parsed.authority,
        tests=parsed.tests,
        proofs=parsed.proofs,
        consumers=parsed.consumers,
    )


def _removal_gate_for(
    record: ExportRecord,
    evidence: RemovalEvidence | None,
) -> RemovalGateRecord:
    blockers: list[RemovalBlockerKind] = [RemovalBlockerKind.MANIFEST_CANNOT_AUTHORIZE]
    if record.consumer_evidence is ConsumerEvidenceKind.UNKNOWN:
        blockers.append(RemovalBlockerKind.UNKNOWN_CONSUMERS)
    if record.consumers:
        migrated = bool(evidence and evidence.consumers_migrated)
        if not migrated:
            blockers.append(RemovalBlockerKind.CONSUMERS_REMAIN)
    deprecated = record.classification is ExportClassification.DEPRECATED or (
        evidence is not None and evidence.deprecated
    )
    if not deprecated:
        blockers.append(RemovalBlockerKind.NOT_DEPRECATED)
    if evidence is None or not evidence.replacement:
        blockers.append(RemovalBlockerKind.MISSING_REPLACEMENT)
    if evidence is None or not evidence.compatibility_satisfied:
        blockers.append(RemovalBlockerKind.MISSING_COMPATIBILITY)
    if evidence is None or not evidence.negative_import_tests:
        blockers.append(RemovalBlockerKind.MISSING_NEGATIVE_IMPORT_TESTS)
    if evidence is None or not evidence.release_notes:
        blockers.append(RemovalBlockerKind.MISSING_RELEASE_NOTES)
    still_exported = True if evidence is None else evidence.still_exported
    if still_exported:
        blockers.append(RemovalBlockerKind.STILL_EXPORTED)
    return RemovalGateRecord(
        symbol=record.symbol,
        blockers=tuple(blockers),
        gates_satisfied=False,
    )


def _merge_consumers(
    declaration: ExportDeclaration,
    references: Sequence[ConsumerReference],
) -> tuple[tuple[str, ...], ConsumerEvidenceKind]:
    matched = tuple(
        sorted(
            {
                item.consumer
                for item in references
                if item.symbol in {declaration.symbol, declaration.qualified_name}
            }
        )
    )
    consumers = tuple(sorted(set(declaration.consumers) | set(matched)))
    if matched or declaration.consumer_evidence is ConsumerEvidenceKind.KNOWN:
        return consumers, ConsumerEvidenceKind.KNOWN
    if consumers:
        return consumers, ConsumerEvidenceKind.KNOWN
    return consumers, declaration.consumer_evidence


def _assessment_for(
    record_key: str,
    traces: Sequence[ImportAssessment],
) -> ImportAssessment | None:
    leaf = _symbol_leaf(record_key)
    matches = [
        item
        for item in traces
        if item.imported_symbol in {record_key, leaf, "*"}
        or item.module == record_key
    ]
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda item: (item.laziness.value, item.imported_symbol, item.module),
    )[0]


def build_public_surface_manifest(
    declarations: Sequence[ExportDeclaration | Mapping[str, Any]] | None = None,
    *,
    projections: Sequence[ProjectionBinding | Mapping[str, Any]] | None = None,
    consumers: Sequence[ConsumerReference | Mapping[str, Any]] | None = None,
    import_traces: Sequence[ImportAssessment | Mapping[str, Any]] | None = None,
    removal_evidence: Sequence[RemovalEvidence | Mapping[str, Any]] | None = None,
    sources: Mapping[str, str] | None = None,
    architecture: ArchitectureIR | Mapping[str, Any] | None = None,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
) -> PublicSurfaceManifest:
    """Classify selected exports and record accidental and projection findings."""

    parsed_declarations = tuple(
        item if isinstance(item, ExportDeclaration) else ExportDeclaration.from_mapping(item)
        for item in (declarations or ())
    )
    parsed_projections = tuple(
        item if isinstance(item, ProjectionBinding) else ProjectionBinding.from_mapping(item)
        for item in (projections or ())
    )
    parsed_consumers = tuple(
        item if isinstance(item, ConsumerReference) else ConsumerReference.from_mapping(item)
        for item in (consumers or ())
    )
    parsed_traces = tuple(
        item if isinstance(item, ImportAssessment) else ImportAssessment.from_mapping(item)
        for item in (import_traces or ())
    )
    parsed_removal = {
        item.symbol: item
        for item in (
            value if isinstance(value, RemovalEvidence) else RemovalEvidence.from_mapping(value)
            for value in (removal_evidence or ())
        )
    }
    graph: ArchitectureIR | None
    if architecture is None:
        graph = None
    elif isinstance(architecture, ArchitectureIR):
        graph = architecture
    else:
        try:
            graph = ArchitectureIR.from_mapping(architecture)
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
    if graph is not None:
        if graph.repository_tree != repository_tree:
            raise PublicSurfaceError(
                "ArchitectureIR repository_tree must match the public-surface manifest"
            )
        if graph.freshness != freshness:
            raise PublicSurfaceError(
                "ArchitectureIR freshness must match the public-surface manifest"
            )
    discovered: list[DiscoveredExport] = []
    if sources:
        discovered.extend(
            discover_exports_from_sources(
                sources, repository_tree=repository_tree, freshness=freshness
            )
        )
        if not parsed_traces:
            parsed_traces = assess_imports_from_sources(
                sources, repository_tree=repository_tree, freshness=freshness
            )
    if graph is not None:
        discovered.extend(discover_exports_from_architecture_ir(graph))
    accidental = detect_accidental_exports(discovered, parsed_declarations)
    projection_findings = detect_projection_mismatch(parsed_projections)
    declared_keys: dict[str, ExportDeclaration] = {}
    for item in parsed_declarations:
        key = item.qualified_name
        existing = declared_keys.get(key)
        if existing is None:
            declared_keys[key] = item
            continue
        if existing.classification is not item.classification:
            raise PublicSurfaceError(_DUPLICATE_CLASSIFICATION_MESSAGE)
        if (
            existing.owner
            and item.owner
            and existing.owner != item.owner
        ) or (
            existing.schema
            and item.schema
            and existing.schema != item.schema
        ):
            raise PublicSurfaceError(_DUPLICATE_CLASSIFICATION_MESSAGE)
    exports: list[ExportRecord] = []
    used_discovered: set[str] = set()
    for declaration in declared_keys.values():
        consumers_for, evidence = _merge_consumers(declaration, parsed_consumers)
        origins = [DiscoveryOrigin.DECLARATION]
        for item in discovered:
            if _matching_declaration(item, {declaration.qualified_name: declaration}) is declaration:
                origins.append(item.origin)
                used_discovered.add(item.qualified_name)
        classification = declaration.classification
        if classification is ExportClassification.INTERNAL and any(
            finding.symbol in {declaration.symbol, _symbol_leaf(declaration.qualified_name)}
            and finding.kind is AccidentalExportKind.INTERNAL_REEXPORT
            for finding in accidental
        ):
            classification = ExportClassification.ACCIDENTALLY_PUBLIC
        stable = None
        if classification is ExportClassification.STABLE:
            patched = ExportDeclaration(
                symbol=declaration.symbol,
                classification=declaration.classification,
                provenance=declaration.provenance,
                qualified_name=declaration.qualified_name,
                module=declaration.module,
                owner=declaration.owner,
                schema=declaration.schema,
                version=declaration.version,
                effects=declaration.effects,
                errors=declaration.errors,
                authority=declaration.authority,
                tests=declaration.tests,
                proofs=declaration.proofs,
                consumers=consumers_for,
                consumer_evidence=evidence,
                projections=declaration.projections,
            )
            if not patched.owner or not patched.schema or not patched.version or not patched.authority:
                raise PublicSurfaceError(_STABLE_INCOMPLETE_MESSAGE)
            stable = _stable_from_declaration(patched)
        projections_for = declaration.projections
        if not projections_for:
            projections_for = tuple(
                item.projection
                for item in parsed_projections
                if item.operation in {declaration.symbol, declaration.qualified_name}
            )
        exports.append(
            ExportRecord(
                symbol=declaration.symbol,
                classification=classification,
                provenance=declaration.provenance,
                qualified_name=declaration.qualified_name,
                origins=tuple(origins),
                projections=projections_for,
                consumers=consumers_for,
                consumer_evidence=evidence,
                import_assessment=_assessment_for(declaration.qualified_name, parsed_traces),
                stable=stable,
            )
        )
    for item in discovered:
        if item.qualified_name in used_discovered:
            continue
        if any(
            record.symbol == item.symbol or record.qualified_name == item.qualified_name
            for record in exports
        ):
            continue
        if item.star:
            continue
        exports.append(
            ExportRecord(
                symbol=item.symbol,
                classification=ExportClassification.ACCIDENTALLY_PUBLIC,
                provenance=item.provenance,
                qualified_name=item.qualified_name,
                origins=(item.origin,),
                consumer_evidence=ConsumerEvidenceKind.UNKNOWN,
                import_assessment=_assessment_for(item.qualified_name, parsed_traces),
            )
        )
    if not exports:
        raise PublicSurfaceError(_EXPORT_CLOSURE_MESSAGE)
    removal_gates = tuple(
        _removal_gate_for(record, parsed_removal.get(record.symbol) or parsed_removal.get(record.qualified_name))
        for record in exports
    )
    stable_symbols = tuple(
        record.stable for record in exports if record.stable is not None
    )
    return PublicSurfaceManifest(
        repository_tree=repository_tree,
        freshness=freshness,
        exports=tuple(exports),
        stable_symbols=stable_symbols,
        accidental_exports=accidental,
        projection_findings=projection_findings,
        import_traces=parsed_traces,
        removal_gates=removal_gates,
    )


classify_exports = build_public_surface_manifest


def refuse_public_promotion(symbol: str) -> None:
    """Reject attempts to treat the manifest as a promotion authority."""

    _require_text(symbol, "symbol", error_type=PublicSurfaceError)
    raise PublicSurfaceAuthorityError(
        "public-surface manifest does not make internal symbols public"
    )


def refuse_deprecation(symbol: str) -> None:
    """Reject attempts to treat the manifest as a deprecation authority."""

    _require_text(symbol, "symbol", error_type=PublicSurfaceError)
    raise PublicSurfaceAuthorityError(
        "public-surface manifest does not deprecate symbols"
    )


def refuse_removal(symbol: str) -> None:
    """Reject attempts to treat the manifest as a removal authority."""

    _require_text(symbol, "symbol", error_type=PublicSurfaceError)
    raise PublicSurfaceAuthorityError(
        "public-surface manifest cannot authorize removal"
    )


def unknown_consumers_block_removal(record: ExportRecord | Mapping[str, Any]) -> bool:
    """Unknown consumers are a hard removal blocker."""

    parsed = record if isinstance(record, ExportRecord) else ExportRecord.from_mapping(record)
    return parsed.consumer_evidence is ConsumerEvidenceKind.UNKNOWN


__all__ = [
    "ACCIDENTAL_EXPORT_SCHEMA",
    "ACCIDENTAL_EXPORT_VERSION",
    "CLOSED_ACCIDENTAL_KINDS",
    "CLOSED_CONSUMER_EVIDENCE",
    "CLOSED_DISCOVERY_ORIGINS",
    "CLOSED_EXPORT_CLASSES",
    "CLOSED_IMPORT_EFFECTS",
    "CLOSED_IMPORT_LAZINESS",
    "CLOSED_PROJECTIONS",
    "CLOSED_PROJECTION_MISMATCHES",
    "CLOSED_REMOVAL_BLOCKERS",
    "CURRENT_SURFACE_BINDINGS",
    "DEFAULT_FRESHNESS",
    "EFFECT_CLASS",
    "EXPORT_RECORD_SCHEMA",
    "EXPORT_RECORD_VERSION",
    "EXTRACTOR_IDENTITY",
    "MANIFEST_CAN_AUTHORIZE_REMOVAL",
    "MANIFEST_CAN_CHANGE_PUBLIC_API",
    "MANIFEST_CAN_DEPRECATE",
    "MANIFEST_CAN_PROMOTE_INTERNAL",
    "PROJECTION_FINDING_SCHEMA",
    "PROJECTION_FINDING_VERSION",
    "PUBLIC_EXPORT_CLASSES",
    "PUBLIC_SURFACE_EVIDENCE",
    "PUBLIC_SURFACE_SCHEMA",
    "PUBLIC_SURFACE_VERSION",
    "REMOVAL_GATE_SCHEMA",
    "REMOVAL_GATE_VERSION",
    "REQUIRED_EXPORT_CLASSES",
    "REQUIRED_PROJECTIONS",
    "STABLE_SYMBOL_SCHEMA",
    "STABLE_SYMBOL_VERSION",
    "TASK_ID",
    "AccidentalExportFinding",
    "AccidentalExportKind",
    "ConsumerEvidenceKind",
    "ConsumerReference",
    "DiscoveredExport",
    "DiscoveryOrigin",
    "ExportClassification",
    "ExportDeclaration",
    "ExportRecord",
    "ImportAssessment",
    "ImportEffectKind",
    "ImportLaziness",
    "ProjectionBinding",
    "ProjectionKind",
    "ProjectionMismatchKind",
    "ProjectionParityFinding",
    "PublicSurfaceAuthorityError",
    "PublicSurfaceError",
    "PublicSurfaceManifest",
    "RemovalBlockerKind",
    "RemovalEvidence",
    "RemovalGateRecord",
    "StablePublicSymbolRecord",
    "SurfaceSourceBinding",
    "assess_imports_from_sources",
    "build_public_surface_manifest",
    "classify_export",
    "classify_exports",
    "detect_accidental_exports",
    "detect_projection_mismatch",
    "discover_exports_from_architecture_ir",
    "discover_exports_from_sources",
    "refuse_deprecation",
    "refuse_public_promotion",
    "refuse_removal",
    "unknown_consumers_block_removal",
]
