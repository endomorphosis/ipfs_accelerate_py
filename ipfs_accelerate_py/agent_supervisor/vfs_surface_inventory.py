"""Evidence-backed inventory of the IPFS Kit virtual-filesystem surface.

This module deliberately performs *static* discovery.  It never imports or
executes files from the repository being inspected.  The resulting inventory
separates observations (definitions, imports, calls, registrations, tests,
documentation, and exports) from conclusions about which implementation ought
to be retained.

In particular, historical suffixes such as ``.fixed`` and ``.full`` are search
signals only.  Their presence is not a defect and is not, by itself, enough to
classify a file as a shadow or duplicate.
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
from typing import Any, Final, Mapping, Sequence


VFS_SURFACE_INVENTORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vfs-surface-inventory@1"
)
VFS_SURFACE_INVENTORY_CONTRACT_VERSION: Final[str] = "vfs-surface-inventory/v1"
VFS_SURFACE_INVENTORY_GOAL_ID: Final[str] = "VFS-025"

# Authority bounds.  An inventory can expose drift; it cannot decide that a
# variant is broken, select a repair, or prove repository correctness.
INVENTORY_IS_COMPLETION_EVIDENCE: Final[bool] = False
INVENTORY_IS_CORRECTNESS_EVIDENCE: Final[bool] = False
INVENTORY_AUTHORIZES_REPAIR: Final[bool] = False
VARIANT_PRESENCE_IS_DEFECT: Final[bool] = False

VARIANT_SUFFIXES: Final[tuple[str, ...]] = (
    ".fixed",
    ".full",
    ".new",
    ".clean",
    ".optimized",
    ".broken",
)

_MAX_TEXT_BYTES = 4 * 1024 * 1024
_TEXT_SUFFIXES = frozenset(
    {
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
    }
)
_ARCHIVE_PARTS = frozenset(
    {
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
    }
)
_TEST_PARTS = frozenset({"test", "tests", "testing", "fixtures"})
_DOC_PARTS = frozenset({"doc", "docs", "documentation"})
_TOOL_PARTS = frozenset({"tool", "tools", "scripts", "cli"})
_SERVER_PARTS = frozenset({"server", "servers"})
_HANDLER_PARTS = frozenset({"handler", "handlers"})
_ENDPOINT_PARTS = frozenset({"endpoint", "endpoints", "api", "apis"})
_CONTROLLER_PARTS = frozenset({"controller", "controllers"})
_SDK_MANIFEST_PARTS = frozenset(
    {"sdk", "sdks", "manifest", "manifests", "package.json", "pyproject.toml"}
)

_VFS_SIGNAL = re.compile(
    r"""(?ix)
    (?<![a-z0-9])(?:vfs|virtual[\s_-]*file[\s_-]*system|ipfs[\s_.-]*fsspec|
    enhanced[\s_.-]*fsspec|filesystem[\s_.-]*journal|fs[\s_.-]*journal|
    vfs[\s_.-]*(?:version|snapshot)|bucket[\s_.-]*vfs|
    (?:car|pin|storage|filesystem)?[\s_.-]*wal)(?![a-z0-9])
    """
)
_PATH_SIGNAL = re.compile(
    r"""(?ix)
    (?:^|[./_-])(?:
      ipfs_fsspec|enhanced_fsspec|iroh_fsspec|iroh_vfs|
      vfs(?:_[a-z0-9]+)*|[a-z0-9]+_vfs(?:_[a-z0-9]+)*|
      filesystem_journal|fs_journal(?:_[a-z0-9]+)*|
      (?:car_|pin_|storage_|enhanced_)?wal(?:_[a-z0-9]+)*|
      vfs_version(?:_[a-z0-9]+)*|vfs_snapshot(?:_[a-z0-9]+)*
    )(?=[^a-z0-9]|$)
    """
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


class VfsSurfaceInventoryError(RuntimeError):
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


class SurfaceKind(str, Enum):
    FSSPEC = "fsspec"
    VFS_MANAGER = "vfs_manager"
    BUCKET_MANAGER = "bucket_manager"
    JOURNAL_WAL = "journal_wal"
    VERSION_SNAPSHOT = "version_snapshot"
    BACKEND_ADAPTER = "backend_adapter"
    HANDLER = "handler"
    ENDPOINT = "endpoint"
    CONTROLLER = "controller"
    TOOL = "tool"
    SERVER = "server"
    SDK_MANIFEST = "sdk_manifest"
    EXPORT = "export"
    DOCUMENTATION = "documentation"
    EXAMPLE = "example"
    OTHER = "other"


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
class VfsSurface:
    path: str
    kinds: tuple[SurfaceKind, ...]
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
            "kinds": [item.value for item in self.kinds],
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
class VfsSurfaceInventory:
    repository_root: str
    scan_roots: tuple[str, ...]
    source_revision: str | None
    surfaces: tuple[VfsSurface, ...]
    contradictions: tuple[SurfaceContradiction, ...]
    diagnostics: tuple[InventoryDiagnostic, ...]
    completeness: InventoryCompleteness
    schema: str = VFS_SURFACE_INVENTORY_SCHEMA
    contract_version: str = VFS_SURFACE_INVENTORY_CONTRACT_VERSION
    is_completion_evidence: bool = INVENTORY_IS_COMPLETION_EVIDENCE
    is_correctness_evidence: bool = INVENTORY_IS_CORRECTNESS_EVIDENCE

    @property
    def coverage_complete(self) -> bool:
        return self.completeness.complete

    @property
    def unexplained_surface_diagnostics(self) -> tuple[InventoryDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if not item.explained)

    def by_path(self) -> dict[str, VfsSurface]:
        return {item.path: item for item in self.surfaces}

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema": self.schema,
            "contract_version": self.contract_version,
            "goal_id": VFS_SURFACE_INVENTORY_GOAL_ID,
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
    kinds: tuple[SurfaceKind, ...]
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
            # Deleted tracked files appear in --modified.  They are an explained
            # overlay state, not an inspectable surface.
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


def _default_scan_roots(repository_root: Path) -> tuple[Path, ...]:
    # VFS-025 is scoped to IPFS Kit.  If invoked against a fixture or the
    # package worktree itself, scan that root instead.
    ipfs_kit = repository_root / "ipfs_kit_py"
    if ipfs_kit.is_dir():
        return (ipfs_kit,)
    return (repository_root,)


def _resolve_scan_roots(
    repository_root: Path,
    scan_roots: Sequence[str | os.PathLike[str]] | None,
) -> tuple[Path, ...]:
    if scan_roots is None:
        return _default_scan_roots(repository_root)

    resolved: list[Path] = []
    for item in scan_roots:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = repository_root / candidate
        resolved.append(candidate.resolve())
    return tuple(resolved)


def _relative(path: Path, repository_root: Path) -> str:
    return path.relative_to(repository_root).as_posix()


def _variant_suffix(path: str) -> str | None:
    lowered = PurePosixPath(path).name.lower()
    for suffix in VARIANT_SUFFIXES:
        if lowered.endswith(suffix) or any(
            lowered.endswith(suffix + extension) for extension in _TEXT_SUFFIXES
        ):
            return suffix
    return None


def _base_without_variant(path: str) -> str:
    suffix = _variant_suffix(path)
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


def _is_python_path(path: str) -> bool:
    return Path(_base_without_variant(path)).suffix.lower() in {".py", ".pyi"}


def _is_text_path(path: str) -> bool:
    base = _base_without_variant(path)
    return Path(base).suffix.lower() in _TEXT_SUFFIXES


def _path_parts(path: str) -> tuple[str, ...]:
    return tuple(part.lower() for part in PurePosixPath(path).parts)


def _has_part(parts: Sequence[str], vocabulary: frozenset[str]) -> bool:
    return bool(set(parts) & vocabulary)


def _is_archive(path: str) -> bool:
    parts = _path_parts(path)
    return _has_part(parts, _ARCHIVE_PARTS) or any(
        part.startswith(("backup_", "archive_")) for part in parts
    )


def _is_test(path: str) -> bool:
    parts = _path_parts(path)
    name = parts[-1] if parts else ""
    return _has_part(parts, _TEST_PARTS) or name.startswith("test_")


def _is_doc(path: str) -> bool:
    return _has_part(_path_parts(path), _DOC_PARTS) or Path(path).suffix.lower() in {
        ".md",
        ".rst",
    }


def _is_candidate(path: str, text_sample: str) -> bool:
    if _PATH_SIGNAL.search(path):
        return True
    # Historical filename suffixes are explicit discovery signals.  They widen
    # coverage only: downstream evidence must still classify what was found,
    # and ``variant_observed`` is always a non-defect diagnostic.
    if _variant_suffix(path) is not None:
        return True
    if not _VFS_SIGNAL.search(text_sample):
        return False
    # A direct module/path reference makes an otherwise generically named file
    # part of the caller/import/registration surface.  This captures generic
    # entrypoints such as ``app.py`` without selecting every file that merely
    # uses a broad word such as "filesystem".
    if _PATH_SIGNAL.search(text_sample):
        return True
    parts = _path_parts(path)
    name = parts[-1] if parts else ""
    role_parts = (
        _TEST_PARTS
        | _DOC_PARTS
        | _TOOL_PARTS
        | _SERVER_PARTS
        | _HANDLER_PARTS
        | _ENDPOINT_PARTS
        | _CONTROLLER_PARTS
        | _SDK_MANIFEST_PARTS
    )
    return (
        _has_part(parts, role_parts)
        or any(
            token in name
            for token in (
                "backend",
                "adapter",
                "manager",
                "journal",
                "wal",
                "fsspec",
                "snapshot",
                "version",
                "export",
                "__init__",
            )
        )
    )


def _surface_kinds(path: str, text: str) -> tuple[SurfaceKind, ...]:
    lowered = path.lower()
    combined = lowered + "\n" + text[:16384].lower()
    parts = _path_parts(path)
    kinds: set[SurfaceKind] = set()

    if "fsspec" in combined:
        kinds.add(SurfaceKind.FSSPEC)
    if (
        re.search(r"(?:^|[/_.-])vfs[_-]?manager(?:[/_.-]|$)", lowered)
        or "class vfsmanager" in combined
    ):
        kinds.add(SurfaceKind.VFS_MANAGER)
    if "bucket" in combined and (
        "vfs" in combined or "virtual filesystem" in combined
    ) and ("manager" in combined or "bucket_vfs" in lowered):
        kinds.add(SurfaceKind.BUCKET_MANAGER)
    if re.search(
        r"(?:^|[/_.-])(?:[a-z]+[_-])?wal(?:[/_.-]|$)|"
        r"(?:filesystem|fs)[_-]?journal",
        combined,
    ):
        kinds.add(SurfaceKind.JOURNAL_WAL)
    if (
        ("vfs" in combined and ("version" in combined or "snapshot" in combined))
        or "vfsversiontracker" in combined
    ):
        kinds.add(SurfaceKind.VERSION_SNAPSHOT)
    if (
        ("backend" in lowered or "adapter" in lowered or "integration" in lowered)
        and (_VFS_SIGNAL.search(combined) or _PATH_SIGNAL.search(lowered))
    ):
        kinds.add(SurfaceKind.BACKEND_ADAPTER)
    if _has_part(parts, _HANDLER_PARTS) or "handler" in Path(lowered).stem:
        kinds.add(SurfaceKind.HANDLER)
    if _has_part(parts, _ENDPOINT_PARTS) or any(
        token in Path(lowered).stem for token in ("endpoint", "_api")
    ):
        kinds.add(SurfaceKind.ENDPOINT)
    if _has_part(parts, _CONTROLLER_PARTS) or "controller" in Path(lowered).stem:
        kinds.add(SurfaceKind.CONTROLLER)
    if _has_part(parts, _TOOL_PARTS) or any(
        token in Path(lowered).stem for token in ("_cli", "_tool")
    ):
        kinds.add(SurfaceKind.TOOL)
    if _has_part(parts, _SERVER_PARTS) or "server" in Path(lowered).stem:
        kinds.add(SurfaceKind.SERVER)
    if (
        _has_part(parts, _SDK_MANIFEST_PARTS)
        or re.search(r"(?:^|[/_.-])sdk(?:[/_.-]|$)", lowered)
        or any(
            token in lowered
            for token in ("manifest", "package.json", "pyproject.toml")
        )
    ):
        kinds.add(SurfaceKind.SDK_MANIFEST)
    if "__init__" in lowered or _EXPORT_NAME.search(text):
        kinds.add(SurfaceKind.EXPORT)
    if _is_doc(path):
        kinds.add(SurfaceKind.DOCUMENTATION)
    if "example" in parts or "examples" in parts or "demo" in Path(lowered).stem:
        kinds.add(SurfaceKind.EXAMPLE)
    if not kinds:
        kinds.add(SurfaceKind.OTHER)
    return tuple(sorted(kinds, key=lambda item: item.value))


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
        *(name if index < required else name + "=" for index, name in enumerate(positional)),
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
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
            )
            if any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in targets
            ):
                value = node.value
                if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                    for item in value.elts:
                        if isinstance(item, ast.Constant) and isinstance(item.value, str):
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


def _analyze_text(analysis: _Analysis) -> None:
    imports: set[str] = set()
    registrations: set[str] = set()
    exports: set[str] = set()
    for number, line in enumerate(analysis.text.splitlines(), 1):
        if _VFS_SIGNAL.search(line) or _PATH_SIGNAL.search(line):
            if re.search(r"(?i)\b(?:import|require|from)\b", line):
                imports.add(line.strip()[:300])
            if _REGISTRATION_NAME.search(line):
                registrations.add(f"line:{number}:{line.strip()[:240]}")
            if _EXPORT_NAME.search(line):
                exports.add(f"line:{number}:{line.strip()[:240]}")
    analysis.imports = tuple(sorted(imports))
    analysis.registrations = tuple(sorted(registrations))
    analysis.exports = tuple(sorted(exports))


def _looks_placeholder(analysis: _Analysis) -> bool:
    if _PLACEHOLDER_MARKER.search(analysis.text[:16384]):
        return True
    if not _is_python_path(analysis.path) or analysis.syntax_error:
        return False
    try:
        tree = ast.parse(analysis.text, filename=analysis.path)
    except (SyntaxError, ValueError):
        return False
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(
        body[0].value, ast.Constant
    ) and isinstance(body[0].value.value, str):
        body = body[1:]
    return bool(body) and all(
        isinstance(item, (ast.Pass, ast.Import, ast.ImportFrom))
        or (
            isinstance(item, ast.Expr)
            and (
                isinstance(item.value, ast.Constant)
                and item.value.value is Ellipsis
            )
        )
        for item in body
    )


def _module_keys(path: str) -> set[str]:
    base = _base_without_variant(path)
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


_REFERENCE_TOKEN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_-]*(?:[./:][A-Za-z0-9_-]+)*"
)


def _reference_tokens(text: str) -> set[str]:
    """Return comparable module, path, and symbol tokens from static text."""

    tokens: set[str] = set()
    for match in _REFERENCE_TOKEN.finditer(text):
        observed = match.group(0).strip("./:")
        if not observed:
            continue
        normalized = _base_without_variant(observed)
        for raw_value in (observed, normalized):
            value = raw_value
            tokens.add(value)
            suffix = Path(value).suffix.lower()
            if suffix in _TEXT_SUFFIXES:
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
) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for path, analysis in analyses.items():
        pure = PurePosixPath(path)
        # An explicit variant filename can be mapped to that variant.  Generic
        # module and symbol references belong to the unsuffixed implementation;
        # attributing them to every sibling would manufacture callers and make
        # an otherwise unreferenced shadow appear canonical.
        explicit_keys = {path, pure.name}
        for key in explicit_keys:
            if len(key) >= 4:
                index[key].add(path)
        if analysis.variant_suffix is None:
            for key in _module_keys(path):
                if len(key) >= 4:
                    index[key].add(path)
            for definition in analysis.definitions:
                if len(definition.name) >= 4:
                    index[definition.name].add(path)
    return index


def _referenced_paths(text: str, index: Mapping[str, set[str]]) -> set[str]:
    referenced: set[str] = set()
    for token in _reference_tokens(text):
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


def _logical_identity(path: str) -> str:
    base = PurePosixPath(_base_without_variant(path))
    return base.stem.lower()


def _discover_contradictions(
    analyses: Mapping[str, _Analysis],
    classifications: Mapping[str, set[SurfaceClassification]],
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
                    (_logical_identity(path), definition.name)
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
                    sorted(f"{path}:{item.kind}:{item.signature}" for path, item in observations)
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


def discover_vfs_surface_paths(
    repository_root: str | os.PathLike[str],
    *,
    scan_roots: Sequence[str | os.PathLike[str]] | None = None,
) -> tuple[str, ...]:
    """Discover candidate paths without parsing them.

    Git indexes are preferred and include modified/untracked overlays.  A
    filesystem walk is used for non-Git fixtures.
    """

    root = Path(repository_root).resolve()
    roots = _resolve_scan_roots(root, scan_roots)
    discovered: set[str] = set()
    for scan_root in roots:
        if not scan_root.is_dir() or not scan_root.is_relative_to(root):
            continue
        paths, _ = _enumerate_root(scan_root)
        for path in paths:
            rel = _relative(path, root)
            if not _is_text_path(rel) or path.is_symlink():
                continue
            try:
                sample = path.read_bytes()[:65536].decode("utf-8", "replace")
            except OSError:
                if _PATH_SIGNAL.search(rel):
                    discovered.add(rel)
                continue
            if _is_candidate(rel, sample):
                discovered.add(rel)
    return tuple(sorted(discovered))


def inventory_vfs_surfaces(
    repository_root: str | os.PathLike[str],
    *,
    scan_roots: Sequence[str | os.PathLike[str]] | None = None,
    max_text_bytes: int = _MAX_TEXT_BYTES,
) -> VfsSurfaceInventory:
    """Build a deterministic, evidence-backed VFS surface inventory."""

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise VfsSurfaceInventoryError(
            f"repository root is not a directory: {root}",
            reason_codes=("repository_root_missing",),
        )
    selected_roots = _resolve_scan_roots(root, scan_roots)
    if not selected_roots:
        raise VfsSurfaceInventoryError(
            "at least one scan root is required", reason_codes=("scan_root_empty",)
        )
    for scan_root in selected_roots:
        if not scan_root.is_dir() or not scan_root.is_relative_to(root):
            raise VfsSurfaceInventoryError(
                f"scan root must be a directory beneath repository root: {scan_root}",
                reason_codes=("scan_root_outside_repository",),
            )

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
            if _PATH_SIGNAL.search(rel):
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
            if _PATH_SIGNAL.search(rel):
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
        if not _is_text_path(rel):
            excluded["non_text_extension"] += 1
            continue
        eligible += 1
        if mode and path.stat().st_size > max_text_bytes:
            excluded["text_size_limit"] += 1
            if _PATH_SIGNAL.search(rel):
                unexplained.add(rel)
                diagnostics.append(
                    InventoryDiagnostic(
                        "surface_size_limit",
                        DiagnosticSeverity.ERROR,
                        f"candidate exceeds {max_text_bytes} byte static-analysis limit",
                        rel,
                        explained=False,
                    )
                )
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            if _PATH_SIGNAL.search(rel):
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
        if not _is_candidate(rel, text[:65536]):
            excluded["no_vfs_signal"] += 1
            continue
        analysis = _Analysis(
            path=rel,
            text=text,
            data=data,
            kinds=_surface_kinds(rel, text),
            variant_suffix=_variant_suffix(rel),
        )
        if _is_python_path(rel):
            _analyze_python(analysis)
        else:
            _analyze_text(analysis)
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

    # Map imports/calls/tests/docs to the target surfaces.  Candidate texts are
    # retained in memory, so this relationship pass performs no execution.
    imported_by: dict[str, set[str]] = defaultdict(set)
    called_by: dict[str, set[str]] = defaultdict(set)
    tested_by: dict[str, set[str]] = defaultdict(set)
    documented_by: dict[str, set[str]] = defaultdict(set)
    relation_evidence: dict[str, list[SurfaceEvidence]] = defaultdict(list)
    reference_index = _reference_index(analyses)
    for source_path, source in analyses.items():
        searchable_imports = "\n".join(source.imports)
        searchable_calls = "\n".join(source.calls)
        searchable_text = source.text[:262144]
        import_targets = _referenced_paths(searchable_imports, reference_index)
        call_targets = _referenced_paths(searchable_calls, reference_index)
        test_targets = (
            _referenced_paths(searchable_text, reference_index)
            if _is_test(source_path)
            else set()
        )
        documentation_targets = (
            _referenced_paths(searchable_text, reference_index)
            if _is_doc(source_path)
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
        identities[_logical_identity(path)].append(path)

    duplicate_of: dict[str, str] = {}
    for paths in digest_groups.values():
        if len(paths) < 2:
            continue
        ranked = sorted(
            paths,
            key=lambda item: (
                _is_archive(item),
                _is_test(item),
                _variant_suffix(item) is not None,
                len(PurePosixPath(item).parts),
                item,
            ),
        )
        for path in ranked[1:]:
            duplicate_of[path] = ranked[0]

    shadows: dict[str, str] = {}
    incoming = {
        path: len(imported_by[path]) + len(called_by[path])
        for path in analyses
    }
    for paths in identities.values():
        if len(paths) < 2:
            continue
        ranked = sorted(
            paths,
            key=lambda item: (
                _is_archive(item),
                _is_test(item),
                _variant_suffix(item) is not None,
                -incoming[item],
                len(PurePosixPath(item).parts),
                item,
            ),
        )
        active = ranked[0]
        for path in ranked[1:]:
            if path not in duplicate_of and incoming[path] == 0 and not _is_archive(path):
                shadows[path] = active

    classifications: dict[str, set[SurfaceClassification]] = {}
    classification_reasons: dict[str, list[str]] = defaultdict(list)
    for path, analysis in analyses.items():
        values: set[SurfaceClassification] = set()
        if _is_archive(path):
            values.add(SurfaceClassification.ARCHIVE)
            classification_reasons[path].append("archive/backup path component")
        if _is_test(path):
            values.add(SurfaceClassification.TEST)
            classification_reasons[path].append("test/fixture path convention")
        if _GENERATED_MARKER.search(analysis.text[:8192]):
            values.add(SurfaceClassification.GENERATED)
            classification_reasons[path].append("generated-file marker")
        if _looks_placeholder(analysis):
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
        if (
            not values & historical
            and (
                canonical_evidence
                or (
                    _PATH_SIGNAL.search(path)
                    and not analysis.variant_suffix
                    and not _is_doc(path)
                )
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

    contradictions = _discover_contradictions(analyses, classifications)
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

    surfaces: list[VfsSurface] = []
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
            VfsSurface(
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

    return VfsSurfaceInventory(
        repository_root=root.as_posix(),
        scan_roots=tuple(_relative(item, root) or "." for item in selected_roots),
        source_revision=_git_revision(root),
        surfaces=tuple(surfaces),
        contradictions=contradictions,
        diagnostics=tuple(sorted(diagnostics, key=_diagnostic_sort_key)),
        completeness=completeness,
    )


def build_vfs_surface_inventory(
    repository_root: str | os.PathLike[str],
    **kwargs: Any,
) -> VfsSurfaceInventory:
    """Compatibility spelling for :func:`inventory_vfs_surfaces`."""

    return inventory_vfs_surfaces(repository_root, **kwargs)


def inventory_repository_vfs_surfaces(
    repository_root: str | os.PathLike[str],
    **kwargs: Any,
) -> VfsSurfaceInventory:
    """Explicit repository-oriented spelling used by supervisor callers."""

    return inventory_vfs_surfaces(repository_root, **kwargs)


def assert_inventory_complete(inventory: VfsSurfaceInventory) -> None:
    """Fail closed when a discovered surface has no complete explanation."""

    if inventory.coverage_complete:
        return
    paths = ", ".join(inventory.completeness.unexplained_paths) or "<count mismatch>"
    raise VfsSurfaceInventoryError(
        f"VFS surface inventory is incomplete: {paths}",
        reason_codes=("inventory_incomplete", "unexplained_surface"),
    )


def publish_vfs_surface_inventory(
    inventory: VfsSurfaceInventory,
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
    return (VFS_SURFACE_INVENTORY_SCHEMA,)


__all__ = [
    "INVENTORY_AUTHORIZES_REPAIR",
    "INVENTORY_IS_COMPLETION_EVIDENCE",
    "INVENTORY_IS_CORRECTNESS_EVIDENCE",
    "VARIANT_PRESENCE_IS_DEFECT",
    "VARIANT_SUFFIXES",
    "VFS_SURFACE_INVENTORY_CONTRACT_VERSION",
    "VFS_SURFACE_INVENTORY_GOAL_ID",
    "VFS_SURFACE_INVENTORY_SCHEMA",
    "Definition",
    "DiagnosticSeverity",
    "EvidenceKind",
    "InventoryCompleteness",
    "InventoryDiagnostic",
    "SurfaceClassification",
    "SurfaceContradiction",
    "SurfaceEvidence",
    "SurfaceKind",
    "VfsSurface",
    "VfsSurfaceInventory",
    "VfsSurfaceInventoryError",
    "assert_inventory_complete",
    "build_vfs_surface_inventory",
    "discover_inventory_schemas",
    "discover_vfs_surface_paths",
    "inventory_repository_vfs_surfaces",
    "inventory_vfs_surfaces",
    "publish_vfs_surface_inventory",
]
