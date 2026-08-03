"""Profile-driven static inventory of repository code surfaces.

This module deliberately performs *static* discovery.  It never imports or
executes files from the repository being inspected.  Domain vocabulary
(signals, kind taxonomy, scan roots, classification bounds, schema identity)
is supplied by an immutable :class:`SurfaceInventoryPolicy`.  Historical
filename suffixes listed in the policy are search signals only; their presence
is not a defect and is not, by itself, enough to classify a file as a shadow
or duplicate.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final, Mapping, Pattern, Sequence


# Authority bounds.  An inventory can expose drift; it cannot decide that a
# variant is broken, select a repair, or prove repository correctness.
INVENTORY_IS_COMPLETION_EVIDENCE: Final[bool] = False
INVENTORY_IS_CORRECTNESS_EVIDENCE: Final[bool] = False
INVENTORY_AUTHORIZES_REPAIR: Final[bool] = False
VARIANT_PRESENCE_IS_DEFECT: Final[bool] = False

REPOSITORY_SURFACE_INVENTORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repository-surface-inventory@1"
)
REPOSITORY_SURFACE_INVENTORY_CONTRACT_VERSION: Final[str] = (
    "repository-surface-inventory/v1"
)

_DEFAULT_MAX_TEXT_BYTES: Final[int] = 4 * 1024 * 1024
_DEFAULT_TEXT_SUFFIXES: Final[tuple[str, ...]] = (
    ".py",
    ".pyi",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
    ".rst",
    ".txt",
    ".in",
    ".sh",
)
_DEFAULT_ARCHIVE_PARTS: Final[tuple[str, ...]] = (
    "archive",
    "archives",
    "archived",
    "backup",
    "backups",
    "obsolete",
    "attic",
    "archived_stale_tests",
    "reorganization_backup_root",
    "reorganization_backup_final",
)
_DEFAULT_TEST_PARTS: Final[tuple[str, ...]] = ("test", "tests", "testing", "fixtures")
_DEFAULT_DOC_PARTS: Final[tuple[str, ...]] = ("doc", "docs", "documentation")
_DEFAULT_TOOL_PARTS: Final[tuple[str, ...]] = ("tool", "tools", "scripts", "cli")
_DEFAULT_SERVER_PARTS: Final[tuple[str, ...]] = ("server", "servers")
_DEFAULT_HANDLER_PARTS: Final[tuple[str, ...]] = ("handler", "handlers")
_DEFAULT_ENDPOINT_PARTS: Final[tuple[str, ...]] = (
    "endpoint",
    "endpoints",
    "api",
    "apis",
)
_DEFAULT_CONTROLLER_PARTS: Final[tuple[str, ...]] = ("controller", "controllers")
_DEFAULT_SDK_MANIFEST_PARTS: Final[tuple[str, ...]] = (
    "sdk",
    "sdks",
    "manifest",
    "manifests",
    "package.json",
    "pyproject.toml",
)
_DEFAULT_ROLE_NAME_TOKENS: Final[tuple[str, ...]] = (
    "backend",
    "adapter",
    "manager",
    "journal",
    "wal",
    "snapshot",
    "version",
    "export",
    "__init__",
)
_DEFAULT_VARIANT_SUFFIXES: Final[tuple[str, ...]] = (
    ".fixed",
    ".full",
    ".new",
    ".clean",
    ".optimized",
    ".broken",
)

_GENERATED_MARKER = re.compile(
    r"(?im)^\s*(?:#|//|/\*)?\s*(?:auto[- ]?generated|generated file|do not edit)\b"
)
_COMPATIBILITY_MARKER = re.compile(
    r"(?i)\b(?:backwards? compatibility|compatibility (?:alias|shim|wrapper)|"
    r"deprecated (?:alias|entrypoint|wrapper)|legacy (?:alias|entrypoint|wrapper)|"
    r"thin wrapper around|proxy to avoid duplication)\b"
)
_PLACEHOLDER_MARKER = re.compile(
    r"(?im)^\s*(?:#\s*)?(?:todo:\s*implement|placeholder(?:\s+implementation)?|"
    r"not implemented(?: yet)?)\s*$"
)
_REGISTRATION_NAME = re.compile(
    r"(?i)(?:register|registry|route|router|endpoint|handler|controller|"
    r"add_(?:api_)?route|include_router|tool|server|mount)"
)
_EXPORT_NAME = re.compile(r"(?i)(?:__all__|export|entry[_-]?point|console_scripts)")
_REFERENCE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*(?:[./:][A-Za-z0-9_-]+)*")


class SurfaceInventoryError(RuntimeError):
    """Inventory failure with stable machine-readable reason codes."""

    def __init__(self, message: str, *, reason_codes: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.reason_codes = tuple(str(code) for code in reason_codes if str(code))


class SurfaceClassification(str, Enum):
    CANONICAL = "canonical"
    COMPATIBILITY = "compatibility"
    GENERATED = "generated"
    TEST = "test"
    ARCHIVE = "archive"
    PLACEHOLDER = "placeholder"
    DUPLICATE = "duplicate"
    SHADOW = "shadow"
    UNKNOWN = "unknown"


class EvidenceKind(str, Enum):
    DEFINITION = "definition"
    IMPORT = "import"
    CALLER = "caller"
    REGISTRATION = "registration"
    TEST = "test"
    DOCUMENTATION = "documentation"
    EXPORT = "export"
    CLASSIFICATION = "classification"
    RELATIONSHIP = "relationship"
    CONTRADICTION = "contradiction"


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class SignalTarget(str, Enum):
    PATH = "path"
    CONTENT = "content"
    PATH_OR_CONTENT = "path_or_content"


@dataclass(frozen=True, order=True)
class SurfaceSignal:
    """A bounded byte-level or path-level discovery/classification signal."""

    name: str
    pattern: str
    target: SignalTarget = SignalTarget.PATH_OR_CONTENT
    flags: int = re.IGNORECASE | re.VERBOSE

    def compile(self) -> Pattern[str]:
        return re.compile(self.pattern, self.flags)


@dataclass(frozen=True, order=True)
class SurfaceKindSpec:
    """Declarative rule that attaches a surface kind id when matched."""

    kind: str
    path_patterns: tuple[str, ...] = ()
    content_patterns: tuple[str, ...] = ()
    combined_patterns: tuple[str, ...] = ()
    path_parts: tuple[str, ...] = ()
    stem_tokens: tuple[str, ...] = ()
    require_domain_signal: bool = False
    flags: int = re.IGNORECASE

    def matches(
        self,
        *,
        path: str,
        text: str,
        path_parts: Sequence[str],
        domain_signal: bool,
    ) -> bool:
        if self.require_domain_signal and not domain_signal:
            return False
        lowered = path.lower()
        stem = Path(lowered).stem
        combined = lowered + "\n" + text[:16384].lower()
        if self.path_parts and set(path_parts) & set(self.path_parts):
            return True
        if self.stem_tokens and any(token in stem for token in self.stem_tokens):
            return True
        for pattern in self.path_patterns:
            if re.search(pattern, lowered, self.flags):
                return True
        for pattern in self.content_patterns:
            if re.search(pattern, text[:16384], self.flags):
                return True
        for pattern in self.combined_patterns:
            if re.search(pattern, combined, self.flags):
                return True
        return False


@dataclass(frozen=True)
class SurfaceInventoryPolicy:
    """Immutable inventory profile: signals, kinds, bounds, and schema identity."""

    profile_id: str
    schema: str = REPOSITORY_SURFACE_INVENTORY_SCHEMA
    contract_version: str = REPOSITORY_SURFACE_INVENTORY_CONTRACT_VERSION
    content_signals: tuple[SurfaceSignal, ...] = ()
    path_signals: tuple[SurfaceSignal, ...] = ()
    kind_specs: tuple[SurfaceKindSpec, ...] = ()
    variant_suffixes: tuple[str, ...] = _DEFAULT_VARIANT_SUFFIXES
    text_suffixes: tuple[str, ...] = _DEFAULT_TEXT_SUFFIXES
    archive_parts: tuple[str, ...] = _DEFAULT_ARCHIVE_PARTS
    test_parts: tuple[str, ...] = _DEFAULT_TEST_PARTS
    doc_parts: tuple[str, ...] = _DEFAULT_DOC_PARTS
    tool_parts: tuple[str, ...] = _DEFAULT_TOOL_PARTS
    server_parts: tuple[str, ...] = _DEFAULT_SERVER_PARTS
    handler_parts: tuple[str, ...] = _DEFAULT_HANDLER_PARTS
    endpoint_parts: tuple[str, ...] = _DEFAULT_ENDPOINT_PARTS
    controller_parts: tuple[str, ...] = _DEFAULT_CONTROLLER_PARTS
    sdk_manifest_parts: tuple[str, ...] = _DEFAULT_SDK_MANIFEST_PARTS
    role_name_tokens: tuple[str, ...] = _DEFAULT_ROLE_NAME_TOKENS
    default_scan_root_names: tuple[str, ...] = ()
    max_text_bytes: int = _DEFAULT_MAX_TEXT_BYTES
    other_kind: str = "other"
    fallback_kind_when_empty: bool = True

    def __post_init__(self) -> None:
        if not self.profile_id or not str(self.profile_id).strip():
            raise SurfaceInventoryError(
                "policy.profile_id is required",
                reason_codes=("policy_profile_id_missing",),
            )
        if self.max_text_bytes <= 0:
            raise SurfaceInventoryError(
                "policy.max_text_bytes must be positive",
                reason_codes=("policy_bounds_invalid",),
            )

    @property
    def text_suffix_set(self) -> frozenset[str]:
        return frozenset(item.lower() for item in self.text_suffixes)

    @property
    def archive_part_set(self) -> frozenset[str]:
        return frozenset(self.archive_parts)

    @property
    def test_part_set(self) -> frozenset[str]:
        return frozenset(self.test_parts)

    @property
    def doc_part_set(self) -> frozenset[str]:
        return frozenset(self.doc_parts)

    def role_part_set(self) -> frozenset[str]:
        return frozenset(
            {
                *self.test_parts,
                *self.doc_parts,
                *self.tool_parts,
                *self.server_parts,
                *self.handler_parts,
                *self.endpoint_parts,
                *self.controller_parts,
                *self.sdk_manifest_parts,
            }
        )

    def identity(self) -> str:
        payload = {
            "profile_id": self.profile_id,
            "schema": self.schema,
            "contract_version": self.contract_version,
            "content_signals": [
                {"name": s.name, "pattern": s.pattern, "target": s.target.value}
                for s in self.content_signals
            ],
            "path_signals": [
                {"name": s.name, "pattern": s.pattern, "target": s.target.value}
                for s in self.path_signals
            ],
            "kind_specs": [
                {
                    "kind": k.kind,
                    "path_patterns": list(k.path_patterns),
                    "content_patterns": list(k.content_patterns),
                    "combined_patterns": list(k.combined_patterns),
                    "path_parts": list(k.path_parts),
                    "stem_tokens": list(k.stem_tokens),
                    "require_domain_signal": k.require_domain_signal,
                }
                for k in self.kind_specs
            ],
            "variant_suffixes": list(self.variant_suffixes),
            "text_suffixes": list(self.text_suffixes),
            "default_scan_root_names": list(self.default_scan_root_names),
            "max_text_bytes": self.max_text_bytes,
            "other_kind": self.other_kind,
        }
        return _content_id(payload)

    def to_record(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "schema": self.schema,
            "contract_version": self.contract_version,
            "policy_identity": self.identity(),
            "variant_suffixes": list(self.variant_suffixes),
            "default_scan_root_names": list(self.default_scan_root_names),
            "max_text_bytes": self.max_text_bytes,
            "kind_ids": [item.kind for item in self.kind_specs],
            "content_signal_names": [item.name for item in self.content_signals],
            "path_signal_names": [item.name for item in self.path_signals],
        }


@dataclass(frozen=True)
class _CompiledPolicy:
    policy: SurfaceInventoryPolicy
    content_patterns: tuple[Pattern[str], ...]
    path_patterns: tuple[Pattern[str], ...]

    @classmethod
    def from_policy(cls, policy: SurfaceInventoryPolicy) -> "_CompiledPolicy":
        return cls(
            policy=policy,
            content_patterns=tuple(item.compile() for item in policy.content_signals),
            path_patterns=tuple(item.compile() for item in policy.path_signals),
        )

    def path_signal(self, path: str) -> bool:
        return any(pattern.search(path) for pattern in self.path_patterns)

    def content_signal(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.content_patterns)

    def domain_signal(self, path: str, text: str) -> bool:
        return self.path_signal(path) or self.content_signal(text)


@dataclass(frozen=True, order=True)
class Definition:
    """A statically observed definition and its comparable signature."""

    name: str
    kind: str
    signature: str
    line: int

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "signature": self.signature,
            "line": self.line,
        }


@dataclass(frozen=True)
class SurfaceEvidence:
    kind: EvidenceKind
    source_path: str
    detail: str
    line: int | None = None
    target_path: str | None = None
    target_symbol: str | None = None
    claim_level: str = "observed_syntax"

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "source_path": self.source_path,
            "detail": self.detail,
            "line": self.line,
            "target_path": self.target_path,
            "target_symbol": self.target_symbol,
            "claim_level": self.claim_level,
        }


@dataclass(frozen=True, order=True)
class SurfaceContradiction:
    """A disagreement, not a declaration that either side is defective."""

    code: str
    symbol: str
    paths: tuple[str, ...]
    observations: tuple[str, ...]
    disposition: str = "inconclusive"
    is_defect: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "symbol": self.symbol,
            "paths": list(self.paths),
            "observations": list(self.observations),
            "disposition": self.disposition,
            "is_defect": self.is_defect,
        }


@dataclass(frozen=True)
class InventoryDiagnostic:
    code: str
    severity: DiagnosticSeverity
    detail: str
    path: str | None = None
    explained: bool = True
    is_defect: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "detail": self.detail,
            "path": self.path,
            "explained": self.explained,
            "is_defect": self.is_defect,
        }


@dataclass(frozen=True)
class SurfaceRecord:
    path: str
    kinds: tuple[str, ...]
    classification: SurfaceClassification
    classifications: tuple[SurfaceClassification, ...]
    classification_evidence: tuple[str, ...]
    variant_suffix: str | None
    sha256: str
    size_bytes: int
    definitions: tuple[Definition, ...] = ()
    imports: tuple[str, ...] = ()
    calls: tuple[str, ...] = ()
    registrations: tuple[str, ...] = ()
    exports: tuple[str, ...] = ()
    imported_by: tuple[str, ...] = ()
    called_by: tuple[str, ...] = ()
    tested_by: tuple[str, ...] = ()
    documented_by: tuple[str, ...] = ()
    duplicate_of: str | None = None
    shadows: str | None = None
    contradictions: tuple[SurfaceContradiction, ...] = ()
    evidence: tuple[SurfaceEvidence, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kinds": list(self.kinds),
            "classification": self.classification.value,
            "classifications": [item.value for item in self.classifications],
            "classification_evidence": list(self.classification_evidence),
            "variant_suffix": self.variant_suffix,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "definitions": [item.to_record() for item in self.definitions],
            "imports": list(self.imports),
            "calls": list(self.calls),
            "registrations": list(self.registrations),
            "exports": list(self.exports),
            "imported_by": list(self.imported_by),
            "called_by": list(self.called_by),
            "tested_by": list(self.tested_by),
            "documented_by": list(self.documented_by),
            "duplicate_of": self.duplicate_of,
            "shadows": self.shadows,
            "contradictions": [item.to_record() for item in self.contradictions],
            "evidence": [item.to_record() for item in self.evidence],
        }


@dataclass(frozen=True)
class InventoryCompleteness:
    enumerated_paths: int
    eligible_text_paths: int
    candidate_paths: int
    analyzed_paths: int
    explained_paths: int
    unexplained_paths: tuple[str, ...] = ()
    excluded_by_reason: Mapping[str, int] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return (
            self.candidate_paths == self.analyzed_paths
            and self.candidate_paths == self.explained_paths
            and not self.unexplained_paths
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "enumerated_paths": self.enumerated_paths,
            "eligible_text_paths": self.eligible_text_paths,
            "candidate_paths": self.candidate_paths,
            "analyzed_paths": self.analyzed_paths,
            "explained_paths": self.explained_paths,
            "unexplained_paths": list(self.unexplained_paths),
            "excluded_by_reason": dict(sorted(self.excluded_by_reason.items())),
        }


@dataclass(frozen=True)
class RepositorySurfaceInventory:
    repository_root: str
    scan_roots: tuple[str, ...]
    source_revision: str | None
    surfaces: tuple[SurfaceRecord, ...]
    contradictions: tuple[SurfaceContradiction, ...]
    diagnostics: tuple[InventoryDiagnostic, ...]
    completeness: InventoryCompleteness
    policy: SurfaceInventoryPolicy
    schema: str = REPOSITORY_SURFACE_INVENTORY_SCHEMA
    contract_version: str = REPOSITORY_SURFACE_INVENTORY_CONTRACT_VERSION
    is_completion_evidence: bool = INVENTORY_IS_COMPLETION_EVIDENCE
    is_correctness_evidence: bool = INVENTORY_IS_CORRECTNESS_EVIDENCE

    @property
    def coverage_complete(self) -> bool:
        return self.completeness.complete

    @property
    def unexplained_surface_diagnostics(self) -> tuple[InventoryDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if not item.explained)

    def by_path(self) -> dict[str, SurfaceRecord]:
        return {item.path: item for item in self.surfaces}

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema": self.schema,
            "contract_version": self.contract_version,
            "profile_id": self.policy.profile_id,
            "policy": self.policy.to_record(),
            "repository_root": self.repository_root,
            "scan_roots": list(self.scan_roots),
            "source_revision": self.source_revision,
            "authority": {
                "is_completion_evidence": self.is_completion_evidence,
                "is_correctness_evidence": self.is_correctness_evidence,
                "authorizes_repair": INVENTORY_AUTHORIZES_REPAIR,
                "variant_presence_is_defect": VARIANT_PRESENCE_IS_DEFECT,
            },
            "completeness": self.completeness.to_record(),
            "surfaces": [item.to_record() for item in self.surfaces],
            "contradictions": [item.to_record() for item in self.contradictions],
            "diagnostics": [item.to_record() for item in self.diagnostics],
        }
        record["content_id"] = _content_id(record)
        return record

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_record(), sort_keys=True, indent=indent, separators=None
        )


@dataclass
class _Analysis:
    path: str
    text: str
    data: bytes
    kinds: tuple[str, ...]
    variant_suffix: str | None
    definitions: tuple[Definition, ...] = ()
    imports: tuple[str, ...] = ()
    calls: tuple[str, ...] = ()
    registrations: tuple[str, ...] = ()
    exports: tuple[str, ...] = ()
    evidence: list[SurfaceEvidence] = field(default_factory=list)
    syntax_error: str | None = None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _content_id(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _run_git(root: Path, args: Sequence[str]) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _git_revision(root: Path) -> str | None:
    value = _run_git(root, ("rev-parse", "HEAD"))
    return value.decode("ascii", "replace").strip() if value else None


def _enumerate_root(root: Path) -> tuple[tuple[Path, ...], str]:
    """Return tracked plus dirty/untracked overlay paths for one Git worktree."""

    raw = _run_git(
        root,
        ("ls-files", "--cached", "--modified", "--others", "--exclude-standard", "-z"),
    )
    if raw is not None:
        paths: list[Path] = []
        for item in raw.split(b"\0"):
            if not item:
                continue
            decoded = item.decode("utf-8", "surrogateescape")
            candidate = root / decoded
            if candidate.exists() or candidate.is_symlink():
                paths.append(candidate)
        return tuple(sorted(set(paths), key=lambda item: item.as_posix())), "git"

    paths = tuple(
        sorted(
            (item for item in root.rglob("*") if item.is_file() or item.is_symlink()),
            key=lambda item: item.as_posix(),
        )
    )
    return paths, "filesystem"


def _default_scan_roots(
    repository_root: Path, policy: SurfaceInventoryPolicy
) -> tuple[Path, ...]:
    for name in policy.default_scan_root_names:
        candidate = repository_root / name
        if candidate.is_dir():
            return (candidate,)
    return (repository_root,)


def _resolve_scan_roots(
    repository_root: Path,
    scan_roots: Sequence[str | os.PathLike[str]] | None,
    policy: SurfaceInventoryPolicy,
) -> tuple[Path, ...]:
    if scan_roots is None:
        return _default_scan_roots(repository_root, policy)

    resolved: list[Path] = []
    for item in scan_roots:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = repository_root / candidate
        resolved.append(candidate.resolve())
    return tuple(resolved)


def _relative(path: Path, repository_root: Path) -> str:
    return path.relative_to(repository_root).as_posix()


def _variant_suffix(path: str, policy: SurfaceInventoryPolicy) -> str | None:
    lowered = PurePosixPath(path).name.lower()
    for suffix in policy.variant_suffixes:
        if lowered.endswith(suffix) or any(
            lowered.endswith(suffix + extension)
            for extension in policy.text_suffix_set
        ):
            return suffix
    return None


def _base_without_variant(path: str, policy: SurfaceInventoryPolicy) -> str:
    suffix = _variant_suffix(path, policy)
    if not suffix:
        return path
    pure = PurePosixPath(path)
    name = pure.name
    lowered = name.lower()
    if lowered.endswith(suffix):
        name = name[: -len(suffix)]
    else:
        index = lowered.rfind(suffix)
        name = name[:index] + name[index + len(suffix) :]
    return pure.with_name(name).as_posix()


def _is_python_path(path: str, policy: SurfaceInventoryPolicy) -> bool:
    return Path(_base_without_variant(path, policy)).suffix.lower() in {".py", ".pyi"}


def _is_text_path(path: str, policy: SurfaceInventoryPolicy) -> bool:
    base = _base_without_variant(path, policy)
    return Path(base).suffix.lower() in policy.text_suffix_set


def _path_parts(path: str) -> tuple[str, ...]:
    return tuple(part.lower() for part in PurePosixPath(path).parts)


def _has_part(parts: Sequence[str], vocabulary: frozenset[str]) -> bool:
    return bool(set(parts) & vocabulary)


def _is_archive(path: str, policy: SurfaceInventoryPolicy) -> bool:
    parts = _path_parts(path)
    return _has_part(parts, policy.archive_part_set) or any(
        part.startswith(("backup_", "archive_")) for part in parts
    )


def _is_test(path: str, policy: SurfaceInventoryPolicy) -> bool:
    parts = _path_parts(path)
    name = parts[-1] if parts else ""
    return _has_part(parts, policy.test_part_set) or name.startswith("test_")


def _is_doc(path: str, policy: SurfaceInventoryPolicy) -> bool:
    return _has_part(_path_parts(path), policy.doc_part_set) or Path(
        path
    ).suffix.lower() in {".md", ".rst"}


def _is_candidate(
    path: str,
    text_sample: str,
    compiled: _CompiledPolicy,
) -> bool:
    policy = compiled.policy
    if compiled.path_signal(path):
        return True
    if _variant_suffix(path, policy) is not None:
        return True
    if not compiled.content_signal(text_sample):
        return False
    if compiled.path_signal(text_sample):
        return True
    parts = _path_parts(path)
    name = parts[-1] if parts else ""
    return _has_part(parts, policy.role_part_set()) or any(
        token in name for token in policy.role_name_tokens
    )


def _surface_kinds(
    path: str,
    text: str,
    compiled: _CompiledPolicy,
) -> tuple[str, ...]:
    policy = compiled.policy
    parts = _path_parts(path)
    domain = compiled.domain_signal(path, text[:16384])
    kinds: set[str] = set()
    for spec in policy.kind_specs:
        if spec.matches(
            path=path,
            text=text,
            path_parts=parts,
            domain_signal=domain,
        ):
            kinds.add(spec.kind)
    if not kinds and policy.fallback_kind_when_empty:
        kinds.add(policy.other_kind)
    return tuple(sorted(kinds))


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return ""


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    positional = [item.arg for item in (*args.posonlyargs, *args.args)]
    defaults = len(args.defaults)
    required = len(positional) - defaults
    kwonly = [
        item.arg + ("" if default is None else "=")
        for item, default in zip(args.kwonlyargs, args.kw_defaults)
    ]
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    vararg = f"*{args.vararg.arg}" if args.vararg else ""
    kwarg = f"**{args.kwarg.arg}" if args.kwarg else ""
    pieces = [
        *(
            name if index < required else name + "="
            for index, name in enumerate(positional)
        ),
        vararg,
        *kwonly,
        kwarg,
    ]
    return f"{prefix}({','.join(piece for piece in pieces if piece)})"


def _analyze_python(analysis: _Analysis) -> None:
    try:
        tree = ast.parse(analysis.text, filename=analysis.path)
    except (SyntaxError, ValueError) as exc:
        analysis.syntax_error = f"{exc.__class__.__name__}: {exc}"
        return

    definitions: list[Definition] = []
    imports: set[str] = set()
    calls: set[str] = set()
    registrations: set[str] = set()
    exports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.append(
                Definition(node.name, "function", _signature(node), node.lineno)
            )
            for decorator in node.decorator_list:
                name = _call_name(decorator)
                if _REGISTRATION_NAME.search(name):
                    registrations.add(f"decorator:{name}:{node.name}")
        elif isinstance(node, ast.ClassDef):
            bases = ",".join(filter(None, (_call_name(item) for item in node.bases)))
            definitions.append(
                Definition(node.name, "class", f"bases({bases})", node.lineno)
            )
            for decorator in node.decorator_list:
                name = _call_name(decorator)
                if _REGISTRATION_NAME.search(name):
                    registrations.add(f"decorator:{name}:{node.name}")
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                imports.add(f"{module}:{alias.name}")
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name:
                calls.add(name)
                if _REGISTRATION_NAME.search(name):
                    label = ""
                    if node.args and isinstance(node.args[0], ast.Constant):
                        if isinstance(node.args[0].value, (str, int)):
                            label = f":{node.args[0].value}"
                    registrations.add(f"call:{name}{label}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in targets
            ):
                value = node.value
                if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                    for item in value.elts:
                        if isinstance(item, ast.Constant) and isinstance(
                            item.value, str
                        ):
                            exports.add(item.value)

    analysis.definitions = tuple(sorted(set(definitions)))
    analysis.imports = tuple(sorted(imports))
    analysis.calls = tuple(sorted(calls))
    analysis.registrations = tuple(sorted(registrations))
    analysis.exports = tuple(sorted(exports))
    for item in analysis.definitions:
        analysis.evidence.append(
            SurfaceEvidence(
                EvidenceKind.DEFINITION,
                analysis.path,
                f"{item.kind} {item.name}{item.signature}",
                line=item.line,
                target_symbol=item.name,
            )
        )
    for item in analysis.imports:
        analysis.evidence.append(
            SurfaceEvidence(EvidenceKind.IMPORT, analysis.path, item)
        )
    for item in analysis.registrations:
        analysis.evidence.append(
            SurfaceEvidence(EvidenceKind.REGISTRATION, analysis.path, item)
        )
    for item in analysis.exports:
        analysis.evidence.append(
            SurfaceEvidence(
                EvidenceKind.EXPORT, analysis.path, item, target_symbol=item
            )
        )


def _analyze_text(analysis: _Analysis, compiled: _CompiledPolicy) -> None:
    imports: set[str] = set()
    registrations: set[str] = set()
    exports: set[str] = set()
    for number, line in enumerate(analysis.text.splitlines(), 1):
        if compiled.domain_signal(analysis.path, line):
            if re.search(r"(?i)\b(?:import|require|from)\b", line):
                imports.add(line.strip()[:300])
            if _REGISTRATION_NAME.search(line):
                registrations.add(f"line:{number}:{line.strip()[:240]}")
            if _EXPORT_NAME.search(line):
                exports.add(f"line:{number}:{line.strip()[:240]}")
    analysis.imports = tuple(sorted(imports))
    analysis.registrations = tuple(sorted(registrations))
    analysis.exports = tuple(sorted(exports))


def _looks_placeholder(
    analysis: _Analysis, policy: SurfaceInventoryPolicy
) -> bool:
    if _PLACEHOLDER_MARKER.search(analysis.text[:16384]):
        return True
    if not _is_python_path(analysis.path, policy) or analysis.syntax_error:
        return False
    try:
        tree = ast.parse(analysis.text, filename=analysis.path)
    except (SyntaxError, ValueError):
        return False
    body = list(tree.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return bool(body) and all(
        isinstance(item, (ast.Pass, ast.Import, ast.ImportFrom))
        or (
            isinstance(item, ast.Expr)
            and (
                isinstance(item.value, ast.Constant) and item.value.value is Ellipsis
            )
        )
        for item in body
    )


def _module_keys(path: str, policy: SurfaceInventoryPolicy) -> set[str]:
    base = _base_without_variant(path, policy)
    pure = PurePosixPath(base)
    no_suffix = pure.with_suffix("").as_posix()
    dotted = no_suffix.replace("/", ".")
    keys = {pure.stem, no_suffix, dotted}
    parts = dotted.split(".")
    for index in range(len(parts)):
        keys.add(".".join(parts[index:]))
    if pure.name == "__init__.py":
        package = pure.parent.as_posix().replace("/", ".")
        keys.add(package)
    return {key for key in keys if key}


def _reference_tokens(text: str, policy: SurfaceInventoryPolicy) -> set[str]:
    tokens: set[str] = set()
    for match in _REFERENCE_TOKEN.finditer(text):
        observed = match.group(0).strip("./:")
        if not observed:
            continue
        normalized = _base_without_variant(observed, policy)
        for raw_value in (observed, normalized):
            value = raw_value
            tokens.add(value)
            suffix = Path(value).suffix.lower()
            if suffix in policy.text_suffix_set:
                value = value[: -len(suffix)]
                tokens.add(value)
            dotted = value.replace("/", ".").replace(":", ".")
            tokens.add(dotted)
            parts = tuple(part for part in dotted.split(".") if part)
            tokens.update(part for part in parts if len(part) >= 4)
            for index in range(len(parts)):
                tokens.add(".".join(parts[index:]))
    return tokens


def _reference_index(
    analyses: Mapping[str, _Analysis],
    policy: SurfaceInventoryPolicy,
) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for path, analysis in analyses.items():
        pure = PurePosixPath(path)
        explicit_keys = {path, pure.name}
        for key in explicit_keys:
            if len(key) >= 4:
                index[key].add(path)
        if analysis.variant_suffix is None:
            for key in _module_keys(path, policy):
                if len(key) >= 4:
                    index[key].add(path)
            for definition in analysis.definitions:
                if len(definition.name) >= 4:
                    index[definition.name].add(path)
    return index


def _referenced_paths(
    text: str,
    index: Mapping[str, set[str]],
    policy: SurfaceInventoryPolicy,
) -> set[str]:
    referenced: set[str] = set()
    for token in _reference_tokens(text, policy):
        referenced.update(index.get(token, ()))
    return referenced


def _primary_classification(
    classifications: set[SurfaceClassification],
) -> SurfaceClassification:
    precedence = (
        SurfaceClassification.ARCHIVE,
        SurfaceClassification.TEST,
        SurfaceClassification.GENERATED,
        SurfaceClassification.PLACEHOLDER,
        SurfaceClassification.DUPLICATE,
        SurfaceClassification.COMPATIBILITY,
        SurfaceClassification.SHADOW,
        SurfaceClassification.CANONICAL,
        SurfaceClassification.UNKNOWN,
    )
    return next(item for item in precedence if item in classifications)


def _logical_identity(path: str, policy: SurfaceInventoryPolicy) -> str:
    base = PurePosixPath(_base_without_variant(path, policy))
    return base.stem.lower()


def _discover_contradictions(
    analyses: Mapping[str, _Analysis],
    classifications: Mapping[str, set[SurfaceClassification]],
    policy: SurfaceInventoryPolicy,
) -> tuple[SurfaceContradiction, ...]:
    by_identity_and_symbol: dict[
        tuple[str, str], list[tuple[str, Definition]]
    ] = defaultdict(list)
    for path, analysis in analyses.items():
        excluded = {
            SurfaceClassification.ARCHIVE,
            SurfaceClassification.TEST,
            SurfaceClassification.GENERATED,
            SurfaceClassification.PLACEHOLDER,
        }
        if classifications[path] & excluded:
            continue
        for definition in analysis.definitions:
            if definition.kind in {"class", "function"}:
                by_identity_and_symbol[
                    (_logical_identity(path, policy), definition.name)
                ].append((path, definition))

    contradictions: list[SurfaceContradiction] = []
    for (_, symbol), observations in sorted(by_identity_and_symbol.items()):
        paths = {path for path, _ in observations}
        signatures = {item.signature for _, item in observations}
        if len(paths) < 2 or len(signatures) < 2:
            continue
        contradictions.append(
            SurfaceContradiction(
                code="definition_signature_disagreement",
                symbol=symbol,
                paths=tuple(sorted(paths)),
                observations=tuple(
                    sorted(
                        f"{path}:{item.kind}:{item.signature}"
                        for path, item in observations
                    )
                ),
            )
        )
    return tuple(contradictions)


def _evidence_sort_key(
    item: SurfaceEvidence,
) -> tuple[str, str, int, str, str, str, str]:
    return (
        item.kind.value,
        item.source_path,
        item.line if item.line is not None else -1,
        item.detail,
        item.target_path or "",
        item.target_symbol or "",
        item.claim_level,
    )


def _diagnostic_sort_key(
    item: InventoryDiagnostic,
) -> tuple[str, str, str, str, bool, bool]:
    return (
        item.code,
        item.severity.value,
        item.path or "",
        item.detail,
        item.explained,
        item.is_defect,
    )


def discover_surface_paths(
    repository_root: str | os.PathLike[str],
    policy: SurfaceInventoryPolicy,
    *,
    scan_roots: Sequence[str | os.PathLike[str]] | None = None,
) -> tuple[str, ...]:
    """Discover candidate paths without parsing them.

    Git indexes are preferred and include modified/untracked overlays.  A
    filesystem walk is used for non-Git fixtures.
    """

    root = Path(repository_root).resolve()
    compiled = _CompiledPolicy.from_policy(policy)
    roots = _resolve_scan_roots(root, scan_roots, policy)
    discovered: set[str] = set()
    for scan_root in roots:
        if not scan_root.is_dir() or not scan_root.is_relative_to(root):
            continue
        paths, _ = _enumerate_root(scan_root)
        for path in paths:
            rel = _relative(path, root)
            if not _is_text_path(rel, policy) or path.is_symlink():
                continue
            try:
                sample = path.read_bytes()[:65536].decode("utf-8", "replace")
            except OSError:
                if compiled.path_signal(rel):
                    discovered.add(rel)
                continue
            if _is_candidate(rel, sample, compiled):
                discovered.add(rel)
    return tuple(sorted(discovered))


def inventory_repository_surfaces(
    repository_root: str | os.PathLike[str],
    policy: SurfaceInventoryPolicy,
    *,
    scan_roots: Sequence[str | os.PathLike[str]] | None = None,
    max_text_bytes: int | None = None,
) -> RepositorySurfaceInventory:
    """Build a deterministic, evidence-backed repository surface inventory."""

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise SurfaceInventoryError(
            f"repository root is not a directory: {root}",
            reason_codes=("repository_root_missing",),
        )
    compiled = _CompiledPolicy.from_policy(policy)
    selected_roots = _resolve_scan_roots(root, scan_roots, policy)
    if not selected_roots:
        raise SurfaceInventoryError(
            "at least one scan root is required", reason_codes=("scan_root_empty",)
        )
    for scan_root in selected_roots:
        if not scan_root.is_dir() or not scan_root.is_relative_to(root):
            raise SurfaceInventoryError(
                f"scan root must be a directory beneath repository root: {scan_root}",
                reason_codes=("scan_root_outside_repository",),
            )

    bound = policy.max_text_bytes if max_text_bytes is None else max_text_bytes
    diagnostics: list[InventoryDiagnostic] = []
    excluded: dict[str, int] = defaultdict(int)
    all_paths: set[Path] = set()
    for scan_root in selected_roots:
        paths, source = _enumerate_root(scan_root)
        all_paths.update(paths)
        diagnostics.append(
            InventoryDiagnostic(
                "enumeration_source",
                DiagnosticSeverity.INFO,
                f"{source} enumeration selected for scan root",
                _relative(scan_root, root) or ".",
            )
        )

    eligible = 0
    analyses: dict[str, _Analysis] = {}
    unexplained: set[str] = set()
    for path in sorted(all_paths, key=lambda item: item.as_posix()):
        rel = _relative(path, root)
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            if compiled.path_signal(rel):
                unexplained.add(rel)
                diagnostics.append(
                    InventoryDiagnostic(
                        "surface_stat_failed",
                        DiagnosticSeverity.ERROR,
                        str(exc),
                        rel,
                        explained=False,
                    )
                )
            else:
                excluded["stat_failed_non_surface"] += 1
            continue
        if stat.S_ISLNK(mode):
            excluded["symlink_not_followed"] += 1
            if compiled.path_signal(rel):
                unexplained.add(rel)
                diagnostics.append(
                    InventoryDiagnostic(
                        "surface_symlink_not_followed",
                        DiagnosticSeverity.WARNING,
                        "candidate symlink was not followed",
                        rel,
                        explained=False,
                    )
                )
            continue
        if not stat.S_ISREG(mode):
            excluded["non_regular"] += 1
            continue
        if not _is_text_path(rel, policy):
            excluded["non_text_extension"] += 1
            continue
        eligible += 1
        if mode and path.stat().st_size > bound:
            excluded["text_size_limit"] += 1
            if compiled.path_signal(rel):
                unexplained.add(rel)
                diagnostics.append(
                    InventoryDiagnostic(
                        "surface_size_limit",
                        DiagnosticSeverity.ERROR,
                        f"candidate exceeds {bound} byte static-analysis limit",
                        rel,
                        explained=False,
                    )
                )
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            if compiled.path_signal(rel):
                unexplained.add(rel)
                diagnostics.append(
                    InventoryDiagnostic(
                        "surface_read_failed",
                        DiagnosticSeverity.ERROR,
                        str(exc),
                        rel,
                        explained=False,
                    )
                )
            else:
                excluded["read_failed_non_surface"] += 1
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", "replace")
            diagnostics.append(
                InventoryDiagnostic(
                    "lossy_utf8_decode",
                    DiagnosticSeverity.WARNING,
                    "invalid UTF-8 was replaced for static analysis",
                    rel,
                )
            )
        if not _is_candidate(rel, text[:65536], compiled):
            excluded["no_domain_signal"] += 1
            continue
        analysis = _Analysis(
            path=rel,
            text=text,
            data=data,
            kinds=_surface_kinds(rel, text, compiled),
            variant_suffix=_variant_suffix(rel, policy),
        )
        if _is_python_path(rel, policy):
            _analyze_python(analysis)
        else:
            _analyze_text(analysis, compiled)
        analyses[rel] = analysis
        if analysis.syntax_error:
            unexplained.add(rel)
            diagnostics.append(
                InventoryDiagnostic(
                    "python_syntax_unexplained",
                    DiagnosticSeverity.ERROR,
                    analysis.syntax_error,
                    rel,
                    explained=False,
                )
            )
        if analysis.variant_suffix:
            diagnostics.append(
                InventoryDiagnostic(
                    "variant_observed",
                    DiagnosticSeverity.INFO,
                    (
                        f"historical suffix {analysis.variant_suffix!r} observed; "
                        "suffix presence alone is not a defect or classification"
                    ),
                    rel,
                    is_defect=False,
                )
            )

    imported_by: dict[str, set[str]] = defaultdict(set)
    called_by: dict[str, set[str]] = defaultdict(set)
    tested_by: dict[str, set[str]] = defaultdict(set)
    documented_by: dict[str, set[str]] = defaultdict(set)
    relation_evidence: dict[str, list[SurfaceEvidence]] = defaultdict(list)
    reference_index = _reference_index(analyses, policy)
    for source_path, source in analyses.items():
        searchable_imports = "\n".join(source.imports)
        searchable_calls = "\n".join(source.calls)
        searchable_text = source.text[:262144]
        import_targets = _referenced_paths(
            searchable_imports, reference_index, policy
        )
        call_targets = _referenced_paths(searchable_calls, reference_index, policy)
        test_targets = (
            _referenced_paths(searchable_text, reference_index, policy)
            if _is_test(source_path, policy)
            else set()
        )
        documentation_targets = (
            _referenced_paths(searchable_text, reference_index, policy)
            if _is_doc(source_path, policy)
            else set()
        )
        target_paths = (
            import_targets | call_targets | test_targets | documentation_targets
        )
        for target_path in target_paths - {source_path}:
            if target_path in import_targets:
                imported_by[target_path].add(source_path)
                relation_evidence[target_path].append(
                    SurfaceEvidence(
                        EvidenceKind.IMPORT,
                        source_path,
                        "static import references surface",
                        target_path=target_path,
                    )
                )
            if target_path in call_targets:
                called_by[target_path].add(source_path)
                relation_evidence[target_path].append(
                    SurfaceEvidence(
                        EvidenceKind.CALLER,
                        source_path,
                        "static call references a definition on surface",
                        target_path=target_path,
                    )
                )
            if target_path in test_targets:
                tested_by[target_path].add(source_path)
                relation_evidence[target_path].append(
                    SurfaceEvidence(
                        EvidenceKind.TEST,
                        source_path,
                        "test text references surface",
                        target_path=target_path,
                    )
                )
            if target_path in documentation_targets:
                documented_by[target_path].add(source_path)
                relation_evidence[target_path].append(
                    SurfaceEvidence(
                        EvidenceKind.DOCUMENTATION,
                        source_path,
                        "documentation references surface",
                        target_path=target_path,
                    )
                )

    digest_groups: dict[str, list[str]] = defaultdict(list)
    identities: dict[str, list[str]] = defaultdict(list)
    for path, analysis in analyses.items():
        digest_groups[hashlib.sha256(analysis.data).hexdigest()].append(path)
        identities[_logical_identity(path, policy)].append(path)

    duplicate_of: dict[str, str] = {}
    for paths in digest_groups.values():
        if len(paths) < 2:
            continue
        ranked = sorted(
            paths,
            key=lambda item: (
                _is_archive(item, policy),
                _is_test(item, policy),
                _variant_suffix(item, policy) is not None,
                len(PurePosixPath(item).parts),
                item,
            ),
        )
        for path in ranked[1:]:
            duplicate_of[path] = ranked[0]

    shadows: dict[str, str] = {}
    incoming = {
        path: len(imported_by[path]) + len(called_by[path]) for path in analyses
    }
    for paths in identities.values():
        if len(paths) < 2:
            continue
        ranked = sorted(
            paths,
            key=lambda item: (
                _is_archive(item, policy),
                _is_test(item, policy),
                _variant_suffix(item, policy) is not None,
                -incoming[item],
                len(PurePosixPath(item).parts),
                item,
            ),
        )
        active = ranked[0]
        for path in ranked[1:]:
            if (
                path not in duplicate_of
                and incoming[path] == 0
                and not _is_archive(path, policy)
            ):
                shadows[path] = active

    classifications: dict[str, set[SurfaceClassification]] = {}
    classification_reasons: dict[str, list[str]] = defaultdict(list)
    for path, analysis in analyses.items():
        values: set[SurfaceClassification] = set()
        if _is_archive(path, policy):
            values.add(SurfaceClassification.ARCHIVE)
            classification_reasons[path].append("archive/backup path component")
        if _is_test(path, policy):
            values.add(SurfaceClassification.TEST)
            classification_reasons[path].append("test/fixture path convention")
        if _GENERATED_MARKER.search(analysis.text[:8192]):
            values.add(SurfaceClassification.GENERATED)
            classification_reasons[path].append("generated-file marker")
        if _looks_placeholder(analysis, policy):
            values.add(SurfaceClassification.PLACEHOLDER)
            classification_reasons[path].append("placeholder-only syntax or marker")
        if path in duplicate_of:
            values.add(SurfaceClassification.DUPLICATE)
            classification_reasons[path].append(
                f"byte-identical to {duplicate_of[path]}"
            )
        if _COMPATIBILITY_MARKER.search(analysis.text[:32768]):
            values.add(SurfaceClassification.COMPATIBILITY)
            classification_reasons[path].append("explicit compatibility/wrapper prose")
        if path in shadows:
            values.add(SurfaceClassification.SHADOW)
            classification_reasons[path].append(
                f"same logical identity as referenced peer {shadows[path]} with no incoming reference"
            )

        historical = {
            SurfaceClassification.ARCHIVE,
            SurfaceClassification.TEST,
            SurfaceClassification.GENERATED,
            SurfaceClassification.PLACEHOLDER,
            SurfaceClassification.DUPLICATE,
            SurfaceClassification.SHADOW,
        }
        canonical_evidence: list[str] = []
        if analysis.registrations:
            canonical_evidence.append("contains static registration")
        if analysis.exports:
            canonical_evidence.append("contains static export")
        if imported_by[path]:
            canonical_evidence.append("referenced by static import")
        if called_by[path]:
            canonical_evidence.append("referenced by static call")
        if documented_by[path]:
            canonical_evidence.append("referenced by documentation")
        if not values & historical and (
            canonical_evidence
            or (
                compiled.path_signal(path)
                and not analysis.variant_suffix
                and not _is_doc(path, policy)
            )
        ):
            values.add(SurfaceClassification.CANONICAL)
            classification_reasons[path].extend(
                canonical_evidence or ["live unsuffixed implementation path"]
            )
        if not values:
            values.add(SurfaceClassification.UNKNOWN)
            classification_reasons[path].append(
                "surface signal observed without stronger classification evidence"
            )
            unexplained.add(path)
            diagnostics.append(
                InventoryDiagnostic(
                    "unexplained_surface_classification",
                    DiagnosticSeverity.WARNING,
                    classification_reasons[path][-1],
                    path,
                    explained=False,
                )
            )
        classifications[path] = values

    contradictions = _discover_contradictions(analyses, classifications, policy)
    contradictions_by_path: dict[str, list[SurfaceContradiction]] = defaultdict(list)
    for contradiction in contradictions:
        diagnostics.append(
            InventoryDiagnostic(
                contradiction.code,
                DiagnosticSeverity.WARNING,
                (
                    f"{contradiction.symbol} has differing observed signatures; "
                    "without a typed canonical contract the finding is inconclusive"
                ),
                ",".join(contradiction.paths),
                is_defect=False,
            )
        )
        for path in contradiction.paths:
            contradictions_by_path[path].append(contradiction)
            relation_evidence[path].append(
                SurfaceEvidence(
                    EvidenceKind.CONTRADICTION,
                    path,
                    f"{contradiction.code}:{contradiction.symbol}",
                    target_symbol=contradiction.symbol,
                )
            )

    surfaces: list[SurfaceRecord] = []
    for path, analysis in sorted(analyses.items()):
        values = classifications[path]
        evidence = list(analysis.evidence) + relation_evidence[path]
        for reason in classification_reasons[path]:
            evidence.append(
                SurfaceEvidence(
                    EvidenceKind.CLASSIFICATION,
                    path,
                    reason,
                )
            )
        surfaces.append(
            SurfaceRecord(
                path=path,
                kinds=analysis.kinds,
                classification=_primary_classification(values),
                classifications=tuple(sorted(values, key=lambda item: item.value)),
                classification_evidence=tuple(classification_reasons[path]),
                variant_suffix=analysis.variant_suffix,
                sha256=hashlib.sha256(analysis.data).hexdigest(),
                size_bytes=len(analysis.data),
                definitions=analysis.definitions,
                imports=analysis.imports,
                calls=analysis.calls,
                registrations=analysis.registrations,
                exports=analysis.exports,
                imported_by=tuple(sorted(imported_by[path])),
                called_by=tuple(sorted(called_by[path])),
                tested_by=tuple(sorted(tested_by[path])),
                documented_by=tuple(sorted(documented_by[path])),
                duplicate_of=duplicate_of.get(path),
                shadows=shadows.get(path),
                contradictions=tuple(contradictions_by_path[path]),
                evidence=tuple(sorted(set(evidence), key=_evidence_sort_key)),
            )
        )

    analyzed = sum(1 for item in analyses.values() if item.syntax_error is None)
    explained = len(analyses) - len(unexplained & set(analyses))
    completeness = InventoryCompleteness(
        enumerated_paths=len(all_paths),
        eligible_text_paths=eligible,
        candidate_paths=len(analyses) + len(unexplained - set(analyses)),
        analyzed_paths=analyzed,
        explained_paths=explained,
        unexplained_paths=tuple(sorted(unexplained)),
        excluded_by_reason=dict(excluded),
    )
    if not completeness.complete:
        diagnostics.append(
            InventoryDiagnostic(
                "inventory_incomplete",
                DiagnosticSeverity.ERROR,
                (
                    f"{len(completeness.unexplained_paths)} candidate surface(s) "
                    "remain unexplained"
                ),
                explained=False,
            )
        )

    return RepositorySurfaceInventory(
        repository_root=root.as_posix(),
        scan_roots=tuple(_relative(item, root) or "." for item in selected_roots),
        source_revision=_git_revision(root),
        surfaces=tuple(surfaces),
        contradictions=contradictions,
        diagnostics=tuple(sorted(diagnostics, key=_diagnostic_sort_key)),
        completeness=completeness,
        policy=policy,
        schema=policy.schema,
        contract_version=policy.contract_version,
    )


def assert_inventory_complete(inventory: RepositorySurfaceInventory) -> None:
    """Fail closed when a discovered surface has no complete explanation."""

    if inventory.coverage_complete:
        return
    paths = ", ".join(inventory.completeness.unexplained_paths) or "<count mismatch>"
    raise SurfaceInventoryError(
        f"repository surface inventory is incomplete: {paths}",
        reason_codes=("inventory_incomplete", "unexplained_surface"),
    )


def publish_surface_inventory(
    inventory: RepositorySurfaceInventory,
    output_path: str | os.PathLike[str],
) -> Path:
    """Atomically publish the deterministic inventory JSON.

    Publication preserves the inventory's non-authoritative flags.  It does not
    promote a finding to completion or correctness evidence.
    """

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    payload = inventory.to_json(indent=2) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def discover_inventory_schemas() -> tuple[str, ...]:
    return (REPOSITORY_SURFACE_INVENTORY_SCHEMA,)


__all__ = [
    "INVENTORY_AUTHORIZES_REPAIR",
    "INVENTORY_IS_COMPLETION_EVIDENCE",
    "INVENTORY_IS_CORRECTNESS_EVIDENCE",
    "REPOSITORY_SURFACE_INVENTORY_CONTRACT_VERSION",
    "REPOSITORY_SURFACE_INVENTORY_SCHEMA",
    "VARIANT_PRESENCE_IS_DEFECT",
    "Definition",
    "DiagnosticSeverity",
    "EvidenceKind",
    "InventoryCompleteness",
    "InventoryDiagnostic",
    "RepositorySurfaceInventory",
    "SignalTarget",
    "SurfaceClassification",
    "SurfaceContradiction",
    "SurfaceEvidence",
    "SurfaceInventoryError",
    "SurfaceInventoryPolicy",
    "SurfaceKindSpec",
    "SurfaceRecord",
    "SurfaceSignal",
    "assert_inventory_complete",
    "discover_inventory_schemas",
    "discover_surface_paths",
    "inventory_repository_surfaces",
    "publish_surface_inventory",
]
